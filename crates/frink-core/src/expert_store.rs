//! A bounded, lease-protected byte cache for routed-expert weights --
//! the storage foundation for running MoE checkpoints whose experts do
//! not all fit in RAM at once (stream cold experts from SSD, keep hot
//! ones resident under one global byte budget).
//!
//! Status: wired into both decode paths as an opt-in
//! (`Decoder::from_gguf_with_expert_cache` for GGUF,
//! `load_kimi_checkpoint_with_expert_cache` for Kimi safetensors, both
//! behind the server's `FRINK_EXPERT_CACHE_BYTES` or
//! `FRINK_SSD_STREAMING=1` which defaults the cache to 2 GiB), each
//! proven
//! bit-identical to its resident path on the committed fixtures at
//! both generous and smaller-than-one-expert budgets. Without the
//! opt-in, experts still load resident (mmap) exactly as before.
//! [`ExpertStore::prefetch`] warms keys for ds4-style SSD streaming
//! overlap (caller supplies the hotlist). Never yet exercised against
//! a real large checkpoint. Reference clone: `.scratch/ds4` (gitignored).
//!
//! Design:
//!
//! - [`ExpertSource`] abstracts where bytes come from (a file with
//!   positional reads, a shard set, an in-memory test source). The
//!   store never caches a partial expert: one `(layer, expert)` key
//!   maps to one complete byte buffer (gate+up+down and any auxiliary
//!   tensors, concatenated by the source in a layout the consumer
//!   defines).
//! - [`ExpertStore::acquire`] returns an [`ExpertLease`] -- a cheap
//!   `Arc` handle that *pins* the entry: eviction skips any entry with
//!   an outstanding lease, so a slot can never be reused while
//!   CPU (or, later, GPU) work still reads it. This is the
//!   slot-reuse-corruption guard, enforced structurally (an `Arc` with
//!   `strong_count > 1` is simply not freeable), not by convention.
//! - When an expert doesn't fit even after evicting every unleased
//!   entry (e.g. a cache configured smaller than one decode step's
//!   expert union), `acquire` still succeeds: the bytes are read and
//!   returned as an *uncached* pass-through lease. Capacity pressure
//!   degrades to more I/O, never to a wrong answer or a deadlock.
//! - Reads happen outside the store lock, so concurrent misses on
//!   different experts overlap their I/O. Two concurrent misses on the
//!   *same* expert may both read it; the first to insert wins and the
//!   loser's buffer becomes that caller's private pass-through copy --
//!   duplicated work under a rare race, never wrong bytes.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Stable identity of one routed expert within a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

/// Where expert bytes come from. Implementations must be cheap to call
/// concurrently (positional reads, not a shared seek cursor).
pub trait ExpertSource: Send + Sync {
    /// The exact byte length of this expert, or `None` if the key does
    /// not exist. Used for budget accounting *before* reading.
    fn expert_len(&self, key: ExpertKey) -> Option<usize>;

    /// Reads this expert's complete bytes.
    fn read_expert(&self, key: ExpertKey) -> io::Result<Vec<u8>>;
}

/// A pinned handle to one expert's bytes. While any lease for an entry
/// is alive, the store cannot evict or reuse that entry's memory.
pub struct ExpertLease {
    data: Arc<Vec<u8>>,
}

impl ExpertLease {
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// The shared buffer behind this lease, for building
    /// `WeightBytes::Shared` sub-range views (one per matrix packed
    /// into the expert's combined bytes). Every clone extends the
    /// entry's pin: the store cannot evict while any of these views is
    /// alive.
    pub fn shared_buf(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.data)
    }
}

/// Monotonic counters, readable at any time without taking the store
/// lock. `resident_bytes` is a gauge (current cache footprint).
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpertStoreStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Acquires that could not be cached (entry larger than what the
    /// budget could free) and were served as uncached pass-throughs.
    pub pass_throughs: u64,
    pub bytes_read: u64,
    pub resident_bytes: u64,
}

struct Entry {
    data: Arc<Vec<u8>>,
    /// Monotonic recency stamp; smallest = least recently used. A
    /// stamp per touch is simpler and cheaper than reshuffling a
    /// dedicated LRU list under the same lock, at the cost of an O(n)
    /// scan on eviction -- fine for the hundreds-to-low-thousands of
    /// entries a real expert cache holds.
    last_used: u64,
}

struct Inner {
    entries: HashMap<ExpertKey, Entry>,
    resident_bytes: usize,
    clock: u64,
}

/// Bytes every live [`ExpertStore`] in this process has committed to
/// holding expert weights.
///
/// This module is the SINGLE holder of the expert byte budget, because
/// on unified memory two budgets are the same RAM counted twice. That
/// rule only bites when something *else* also wants to retain weight
/// bytes, and something else now does: `weight_matrix::repack_cache`
/// keeps interleaved copies of matrices it has packed. This gauge is
/// how that cache subtracts rather than restates -- see
/// [`crate::host_memory::derived_copy_budget`].
static COMMITTED_EXPERT_BYTES: AtomicU64 = AtomicU64::new(0);

/// Bytes currently committed to expert caches across the process. See
/// [`COMMITTED_EXPERT_BYTES`].
pub fn committed_expert_bytes() -> u64 {
    COMMITTED_EXPERT_BYTES.load(Ordering::Relaxed)
}

/// The bounded cache. See the module docs for the design contract.
pub struct ExpertStore<S: ExpertSource> {
    source: S,
    budget_bytes: usize,
    inner: Mutex<Inner>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    pass_throughs: AtomicU64,
    bytes_read: AtomicU64,
}

impl<S: ExpertSource> Drop for ExpertStore<S> {
    fn drop(&mut self) {
        COMMITTED_EXPERT_BYTES.fetch_sub(self.budget_bytes as u64, Ordering::Relaxed);
    }
}

impl<S: ExpertSource> ExpertStore<S> {
    pub fn new(source: S, budget_bytes: usize) -> Self {
        COMMITTED_EXPERT_BYTES.fetch_add(budget_bytes as u64, Ordering::Relaxed);
        ExpertStore {
            source,
            budget_bytes,
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                resident_bytes: 0,
                clock: 0,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            pass_throughs: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub fn stats(&self) -> ExpertStoreStats {
        let resident = self
            .inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .resident_bytes as u64;
        ExpertStoreStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            pass_throughs: self.pass_throughs.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            resident_bytes: resident,
        }
    }

    /// Best-effort warm of `keys` into the cache (ds4-style SSD
    /// streaming prefetch): acquires each key and drops the lease so
    /// the entry stays resident under the budget until eviction.
    /// Concurrent acquires on the same key may duplicate I/O; never
    /// returns wrong bytes. Failures on individual keys are skipped.
    pub fn prefetch(&self, keys: &[ExpertKey]) {
        for &key in keys {
            let _ = self.acquire(key);
        }
    }

    /// Returns a pinned lease on `key`'s bytes, reading them from the
    /// source on a miss. Never blocks waiting for other leases to be
    /// released: if the entry cannot fit in the budget right now, the
    /// bytes are returned uncached (see module docs).
    pub fn acquire(&self, key: ExpertKey) -> io::Result<ExpertLease> {
        // Fast path: cache hit under the lock.
        {
            let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
            inner.clock += 1;
            let clock = inner.clock;
            if let Some(entry) = inner.entries.get_mut(&key) {
                entry.last_used = clock;
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(ExpertLease {
                    data: Arc::clone(&entry.data),
                });
            }
        }

        // Miss: read outside the lock so concurrent misses overlap I/O.
        self.misses.fetch_add(1, Ordering::Relaxed);
        let data = self.source.read_expert(key)?;
        self.bytes_read
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        let size = data.len();
        let data = Arc::new(data);

        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        // Someone else may have inserted this key while we read -- use
        // theirs (keeps exactly one cached copy), our read becomes the
        // rare-race duplicate described in the module docs. Already
        // counted as a miss above (each acquire increments exactly one
        // of hits/misses), so no hit is recorded here.
        if let Some(entry) = inner.entries.get_mut(&key) {
            return Ok(ExpertLease {
                data: Arc::clone(&entry.data),
            });
        }

        if size > self.budget_bytes {
            self.pass_throughs.fetch_add(1, Ordering::Relaxed);
            return Ok(ExpertLease { data });
        }

        // Evict least-recently-used *unleased* entries until it fits.
        while inner.resident_bytes + size > self.budget_bytes {
            let victim = inner
                .entries
                .iter()
                .filter(|(_, e)| Arc::strong_count(&e.data) == 1)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| *k);
            match victim {
                Some(k) => {
                    let e = inner.entries.remove(&k).expect("victim key just found");
                    inner.resident_bytes -= e.data.len();
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                }
                None => {
                    // Every resident entry is pinned by a live lease --
                    // serve uncached rather than waiting (a wait here
                    // could deadlock against our own caller's leases).
                    self.pass_throughs.fetch_add(1, Ordering::Relaxed);
                    return Ok(ExpertLease { data });
                }
            }
        }

        inner.clock += 1;
        let clock = inner.clock;
        inner.resident_bytes += size;
        inner.entries.insert(
            key,
            Entry {
                data: Arc::clone(&data),
                last_used: clock,
            },
        );
        Ok(ExpertLease { data })
    }
}

mod file_source;
pub use file_source::FileRangeSource;

#[cfg(test)]
mod tests;
