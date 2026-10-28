//! A bounded, lease-protected cache for routed-expert weights: the
//! storage under a MoE checkpoint whose experts do not all fit in RAM
//! (stream cold experts from SSD, keep hot ones resident under one
//! global byte budget).
//!
//! Status: wired into both decode paths as an opt-in
//! (`Decoder::from_gguf_with_expert_cache` for GGUF,
//! `load_kimi_checkpoint_with_expert_cache` for Kimi safetensors, both
//! behind `FRINK_EXPERT_CACHE_BYTES` or `FRINK_SSD_STREAMING=1`, which
//! defaults the cache to 2 GiB), each proven bit-identical to its
//! resident path on the committed fixtures at generous, quarter and
//! smaller-than-one-expert budgets. Without the opt-in, experts load
//! resident (mmap) exactly as before. Reference clone: `.scratch/ds4`
//! (gitignored).
//!
//! Design:
//!
//! - [`ExpertSource`] abstracts where bytes come from (a file with
//!   positional reads, a shard set, an in-memory test source). The
//!   store never caches a partial expert: one `(layer, expert)` key
//!   maps to one complete byte buffer (gate+up+down and any auxiliary
//!   tensors, concatenated by the source in a layout the consumer
//!   defines).
//! - Two storage modes behind ONE policy. [`ExpertStore::new`] keeps
//!   each entry in its own heap buffer, bounded by bytes.
//!   [`ExpertStore::with_slots`] keeps entries in the fixed slots of a
//!   [`SlotMemory`] pool ([`crate::expert_slab`]: host RAM, or on Apple
//!   silicon one Shared `MTLBuffer` the GPU reads in place), bounded by
//!   slot count. Which entry is evicted, and whether a miss is admitted
//!   at all, is decided by the same code for both.
//! - The policy is hotness first, recency second. With an
//!   [`ExpertHotness`] attached ([`ExpertStore::with_hotness`]), the
//!   victim is the unpinned entry with the fewest real routing
//!   selections, ties broken by least recent use, and a miss that is
//!   COLDER than every evictable entry is served without being
//!   admitted, so one cold expert cannot push out a hot one. Without
//!   it every expert reads as equally hot and the policy is plain LRU.
//!   Selections are counted by the caller at routing time, misses
//!   included, which is what keeps an expert that is routed to
//!   constantly but evicted before its second hit from looking cold.
//! - [`ExpertStore::acquire`] returns an [`ExpertLease`]: a cheap `Arc`
//!   handle that *pins* the entry. Eviction skips any entry with an
//!   outstanding lease, so a slot can never be reused while CPU or GPU
//!   work still reads it. This is the slot-reuse-corruption guard,
//!   enforced structurally (an `Arc` with `strong_count > 1` is simply
//!   not freeable), not by convention. [`ExpertStore::acquire_many`]
//!   pins a whole top-k before reading any of it, so a miss on the last
//!   expert cannot evict the first.
//! - When an expert doesn't fit even after evicting every unleased
//!   entry (a cache configured smaller than one decode step's expert
//!   union, or a slot smaller than this expert), `acquire` still
//!   succeeds: the bytes are read and returned as an *uncached*
//!   pass-through lease. Capacity pressure degrades to more I/O, never
//!   to a wrong answer or a deadlock.
//! - Reads happen outside the store lock, so concurrent misses on
//!   different experts overlap their I/O. Two concurrent misses on the
//!   *same* expert may both read it; the first to insert wins and the
//!   loser's buffer becomes that caller's private pass-through copy:
//!   duplicated work under a rare race, never wrong bytes.
//!
//! Bounded work per token: a decode step acquires each layer's top-k
//! once, so it reads at most `top_k` experts per routed layer and
//! admits (and so evicts) at most that many. Nothing here prefetches a
//! layer ahead or speculatively.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::expert_slab::SlotMemory;

mod bytes;
pub use bytes::ExpertBytes;
use bytes::SlotPool;

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

/// How often each expert has been ROUTED to: the hotness the eviction
/// policy ranks by. Implemented by whoever routes (the decoder's
/// per-layer activation counters), so the store reads the one count
/// the rest of the engine already keeps rather than a second tally of
/// its own that could disagree with it.
pub trait ExpertHotness: Send + Sync {
    fn selections(&self, key: ExpertKey) -> u64;
}

/// A pinned handle to one expert's bytes. While any lease for an entry
/// is alive, the store cannot evict or reuse that entry's memory.
#[derive(Clone)]
pub struct ExpertLease {
    data: Arc<ExpertBytes>,
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
    pub fn shared_buf(&self) -> Arc<ExpertBytes> {
        Arc::clone(&self.data)
    }

    /// The slot this lease reads, or `None` for heap bytes (heap mode,
    /// or an uncached pass-through).
    pub fn slot(&self) -> Option<u32> {
        self.data.slot_index()
    }
}

/// Monotonic counters, readable at any time without taking the store
/// lock. `resident_bytes` is a gauge (current cache footprint) and
/// `peak_resident_bytes` its high-water mark.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpertStoreStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Acquires that were not cached and were served as uncached
    /// pass-throughs: the entry was larger than what the budget could
    /// free, or every resident entry was pinned, or the policy refused
    /// to admit it (`refused_admissions` counts that last case alone).
    pub pass_throughs: u64,
    /// Misses the policy declined to admit because every evictable
    /// entry was hotter.
    pub refused_admissions: u64,
    pub bytes_read: u64,
    pub resident_bytes: u64,
    pub peak_resident_bytes: u64,
}

struct Entry {
    data: Arc<ExpertBytes>,
    /// Bytes this entry holds against the budget: its length in heap
    /// mode, one whole slot in slot mode.
    charge: usize,
    /// Monotonic recency stamp; smallest = least recently used. A
    /// stamp per touch is simpler and cheaper than reshuffling a
    /// dedicated LRU list under the same lock, at the cost of an O(n)
    /// scan on eviction: fine for the hundreds-to-low-thousands of
    /// entries a real expert cache holds.
    last_used: u64,
}

struct Inner {
    entries: HashMap<ExpertKey, Entry>,
    resident_bytes: usize,
    peak_resident_bytes: usize,
    clock: u64,
    /// Slot mode only: slots no entry occupies.
    free_slots: Vec<u32>,
}

/// Where admitted entries live.
enum Storage {
    Heap,
    Slots(Arc<SlotPool>),
}

/// Bytes every live [`ExpertStore`] in this process has committed to
/// holding expert weights.
///
/// This module is the SINGLE holder of the expert byte budget, because
/// on unified memory two budgets are the same RAM counted twice. That
/// rule only bites when something *else* also wants to retain weight
/// bytes, and two things do: `weight_matrix::repack_cache` keeps
/// interleaved copies of matrices it has packed, and subtracts this
/// gauge rather than restating it (see
/// [`crate::host_memory::derived_copy_budget`]); and the Metal weight
/// cache, which charges nothing for a slice of a Metal slot slab
/// because it aliases the slab instead of copying it.
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
    storage: Storage,
    hotness: Option<Arc<dyn ExpertHotness>>,
    inner: Mutex<Inner>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    pass_throughs: AtomicU64,
    refused_admissions: AtomicU64,
    bytes_read: AtomicU64,
}

impl<S: ExpertSource> Drop for ExpertStore<S> {
    fn drop(&mut self) {
        COMMITTED_EXPERT_BYTES.fetch_sub(self.budget_bytes as u64, Ordering::Relaxed);
    }
}

/// What one miss turns into once its bytes are read.
enum Placement {
    /// Admitted: the entry's bytes, now resident.
    Cached(Arc<ExpertBytes>),
    /// Not admitted: the reader's own copy, gone when its lease drops.
    PassThrough(Vec<u8>),
}

impl<S: ExpertSource> ExpertStore<S> {
    /// Heap mode: each entry its own buffer, `budget_bytes` in total.
    pub fn new(source: S, budget_bytes: usize) -> Self {
        Self::build(source, budget_bytes, Storage::Heap, Vec::new())
    }

    /// Slot mode: entries live in `memory`'s slots, one expert per
    /// slot, and the budget is the pool's size. An expert larger than a
    /// slot is served uncached.
    pub fn with_slots(source: S, memory: Box<dyn SlotMemory>) -> Self {
        let pool = Arc::new(SlotPool::new(memory));
        let budget = pool.slots() * pool.slot_bytes;
        // Popped from the back, so slot 0 is used first.
        let free: Vec<u32> = (0..pool.slots() as u32).rev().collect();
        Self::build(source, budget, Storage::Slots(pool), free)
    }

    fn build(source: S, budget_bytes: usize, storage: Storage, free_slots: Vec<u32>) -> Self {
        COMMITTED_EXPERT_BYTES.fetch_add(budget_bytes as u64, Ordering::Relaxed);
        ExpertStore {
            source,
            budget_bytes,
            storage,
            hotness: None,
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                resident_bytes: 0,
                peak_resident_bytes: 0,
                clock: 0,
                free_slots,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            pass_throughs: AtomicU64::new(0),
            refused_admissions: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }

    /// Ranks eviction by `hotness` (see the module docs). Without it
    /// the policy is plain LRU.
    pub fn with_hotness(mut self, hotness: Arc<dyn ExpertHotness>) -> Self {
        self.hotness = Some(hotness);
        self
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Slots in the pool, or `None` in heap mode.
    pub fn slot_count(&self) -> Option<usize> {
        match &self.storage {
            Storage::Heap => None,
            Storage::Slots(pool) => Some(pool.slots()),
        }
    }

    /// Where the slots live (`"host"`, `"metal"`), or `None` in heap
    /// mode.
    pub fn slot_memory_kind(&self) -> Option<&'static str> {
        match &self.storage {
            Storage::Heap => None,
            Storage::Slots(pool) => Some(pool.kind),
        }
    }

    pub fn stats(&self) -> ExpertStoreStats {
        let (resident, peak) = {
            let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
            (
                inner.resident_bytes as u64,
                inner.peak_resident_bytes as u64,
            )
        };
        ExpertStoreStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            pass_throughs: self.pass_throughs.load(Ordering::Relaxed),
            refused_admissions: self.refused_admissions.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            resident_bytes: resident,
            peak_resident_bytes: peak,
        }
    }

    /// True when `key` is cached right now. For tests and reports.
    pub fn is_resident(&self, key: ExpertKey) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entries
            .contains_key(&key)
    }

    /// Best-effort warm of `keys` into the cache (ds4-style SSD
    /// streaming prefetch): acquires each key and drops the lease so
    /// the entry stays resident under the budget until eviction.
    /// Concurrent acquires on the same key may duplicate I/O; never
    /// returns wrong bytes. Failures on individual keys are skipped.
    /// No caller yet: it is the applicator for a shipped hotlist
    /// (`docs/plans/out-of-core-moe.md`, `shipped-profile-and-runtime-
    /// learning`).
    pub fn prefetch(&self, keys: &[ExpertKey]) {
        for &key in keys {
            let _ = self.acquire(key);
        }
    }

    /// Returns a pinned lease on `key`'s bytes, reading them from the
    /// source on a miss. Never blocks waiting for other leases to be
    /// released: if the entry cannot be cached right now, the bytes are
    /// returned uncached (see module docs).
    pub fn acquire(&self, key: ExpertKey) -> io::Result<ExpertLease> {
        let mut leases = self.acquire_many(&[key])?;
        Ok(leases.pop().expect("one key, one lease"))
    }

    /// Leases for every key, `out[i]` being `keys[i]`'s, all pinned at
    /// once: hits are pinned before any miss is read, so admitting a
    /// later miss can never evict an earlier key of the same batch. A
    /// key repeated in `keys` is read once and shares one lease.
    pub fn acquire_many(&self, keys: &[ExpertKey]) -> io::Result<Vec<ExpertLease>> {
        let mut out: Vec<Option<ExpertLease>> = vec![None; keys.len()];
        // Indices of the first occurrence of each missed key.
        let mut missed: Vec<usize> = Vec::new();
        {
            let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
            for (i, key) in keys.iter().enumerate() {
                if keys[..i].contains(key) {
                    continue; // resolved from its first occurrence below
                }
                inner.clock += 1;
                let clock = inner.clock;
                if let Some(entry) = inner.entries.get_mut(key) {
                    entry.last_used = clock;
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    out[i] = Some(ExpertLease {
                        data: Arc::clone(&entry.data),
                    });
                } else {
                    missed.push(i);
                }
            }
        }

        // Read outside the lock so concurrent misses overlap I/O.
        let mut read: Vec<(usize, Vec<u8>)> = Vec::with_capacity(missed.len());
        for &i in &missed {
            self.misses.fetch_add(1, Ordering::Relaxed);
            let data = self.source.read_expert(keys[i])?;
            self.bytes_read
                .fetch_add(data.len() as u64, Ordering::Relaxed);
            read.push((i, data));
        }

        if !read.is_empty() {
            let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
            for (i, data) in read {
                let lease = match self.place(&mut inner, keys[i], data) {
                    Placement::Cached(data) => ExpertLease { data },
                    Placement::PassThrough(data) => {
                        self.pass_throughs.fetch_add(1, Ordering::Relaxed);
                        ExpertLease {
                            data: Arc::new(ExpertBytes::from(data)),
                        }
                    }
                };
                out[i] = Some(lease);
            }
        }

        for i in 0..keys.len() {
            if out[i].is_none() {
                let first = keys[..i]
                    .iter()
                    .position(|k| *k == keys[i])
                    .expect("an unresolved index is a repeat of an earlier key");
                out[i] = out[first].clone();
            }
        }
        Ok(out
            .into_iter()
            .map(|l| l.expect("every key resolved"))
            .collect())
    }

    fn selections(&self, key: ExpertKey) -> u64 {
        self.hotness.as_ref().map_or(0, |h| h.selections(key))
    }

    /// The entry the policy would evict to admit `candidate`, or `None`
    /// when nothing may go: every entry is pinned, or every evictable
    /// entry is hotter than `candidate` (counted as a refusal).
    fn victim(&self, inner: &Inner, candidate: ExpertKey) -> Option<ExpertKey> {
        let (key, hot) = inner
            .entries
            .iter()
            .filter(|(_, e)| Arc::strong_count(&e.data) == 1)
            .map(|(k, e)| (*k, (self.selections(*k), e.last_used)))
            .min_by_key(|(_, rank)| *rank)
            .map(|(k, (hot, _))| (k, hot))?;
        if hot > self.selections(candidate) {
            self.refused_admissions.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(key)
    }

    fn evict(&self, inner: &mut Inner, key: ExpertKey) {
        let e = inner.entries.remove(&key).expect("victim key just found");
        inner.resident_bytes -= e.charge;
        if let Some(slot) = e.data.slot_index() {
            // The map held the only `Arc` (the victim filter), so no
            // reader can see this slot any more.
            inner.free_slots.push(slot);
        }
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Admits `key` under the lock, or says why not.
    fn place(&self, inner: &mut Inner, key: ExpertKey, mut data: Vec<u8>) -> Placement {
        // Someone else may have inserted this key while we read: use
        // theirs (exactly one cached copy); our read is the rare-race
        // duplicate the module docs describe. Already counted as a
        // miss, so no hit is recorded here.
        if let Some(entry) = inner.entries.get(&key) {
            return Placement::Cached(Arc::clone(&entry.data));
        }
        let size = data.len();
        let (bytes, charge) = match &self.storage {
            Storage::Heap => {
                if size > self.budget_bytes {
                    return Placement::PassThrough(data);
                }
                while inner.resident_bytes + size > self.budget_bytes {
                    match self.victim(inner, key) {
                        Some(v) => self.evict(inner, v),
                        None => return Placement::PassThrough(data),
                    }
                }
                (Arc::new(ExpertBytes::from(data)), size)
            }
            Storage::Slots(pool) => {
                if size > pool.slot_bytes {
                    return Placement::PassThrough(data);
                }
                if inner.free_slots.is_empty() {
                    match self.victim(inner, key) {
                        Some(v) => self.evict(inner, v),
                        None => return Placement::PassThrough(data),
                    }
                }
                let slot = inner.free_slots.pop().expect("a slot was just freed");
                // A slot is written whole: the padding replaces whatever
                // the previous occupant left past this expert's end.
                data.resize(pool.slot_bytes, 0);
                let written = pool
                    .device
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .write_slot(0, slot, &data);
                data.truncate(size);
                if let Err(e) = written {
                    eprintln!("frink: expert slot write failed, serving uncached: {e}");
                    inner.free_slots.push(slot);
                    return Placement::PassThrough(data);
                }
                (
                    Arc::new(ExpertBytes::slot(Arc::clone(pool), slot, size)),
                    pool.slot_bytes,
                )
            }
        };
        inner.clock += 1;
        let clock = inner.clock;
        inner.resident_bytes += charge;
        inner.peak_resident_bytes = inner.peak_resident_bytes.max(inner.resident_bytes);
        inner.entries.insert(
            key,
            Entry {
                data: Arc::clone(&bytes),
                charge,
                last_used: clock,
            },
        );
        Placement::Cached(bytes)
    }
}

mod file_source;
pub use file_source::FileRangeSource;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod policy_tests;
