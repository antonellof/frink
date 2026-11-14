//! The one lookup path every address-keyed resident-buffer cache takes,
//! and the proof each entry has to give before it may be served.
//!
//! # What went wrong (GitHub issue #180)
//!
//! Two caches in `gpu.rs` mapped `(host pointer, host length)` to an
//! `MTLBuffer` built from those bytes. A host address is not an
//! identity: free the allocation and the next one of the same size gets
//! the address back, and the cache then serves the new tensor the old
//! tensor's uploaded bytes.
//!
//! `resident_weight_buffer` knew that and defended itself twice -- a
//! mmap keepalive on the zero-copy path, a sampled fingerprint on the
//! copy path. `resident_f32_buffer`, sitting seventy lines below it in
//! the same file with the same key type, defended itself not at all.
//! That is this repo's dominant bug shape: two structures that must
//! agree about one thing with nothing enforcing it.
//!
//! What it cost: `POST /admin/models/load` swaps checkpoints in one
//! process, and a `Decoder`'s RMSNorm gammas are owned `Vec<f32>`s that
//! are freed with it. Loading Llama-3.2-1B-Q4_K_M and then
//! Llama-3.2-1B-Q6_K put the second model's gammas at the first
//! model's freed addresses -- 49 stale hits in a 24-token decode,
//! measured -- and the answer came back fluent and wrong. The Studio
//! model selector is that endpoint.
//!
//! # The rule now
//!
//! An entry may be served only if it can still prove it holds the
//! caller's bytes ([`Resident::still_holds`]), and [`get_or_build`] is
//! the only way to read one of these caches. A new cache added here
//! inherits the check by construction: it cannot be looked up without
//! an entry type that answers the question, and answering it `true`
//! unconditionally is a thing a reviewer can see, unlike an absent
//! check nobody wrote.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::LocalKey;

use crate::gpu::MetalError;

/// Where a cached upload came from: a host address and a length.
///
/// Deliberately NOT called an identity. It is the lookup key and
/// nothing more -- what makes a hit legitimate is
/// [`Resident::still_holds`], not this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct HostKey {
    addr: usize,
    len: usize,
}

impl HostKey {
    pub(crate) fn of(bytes: &[u8]) -> Self {
        HostKey {
            addr: bytes.as_ptr() as usize,
            len: bytes.len(),
        }
    }
}

/// What a cached device buffer must be able to say about itself.
pub(crate) trait Resident {
    /// True when this entry still holds exactly `host`.
    ///
    /// Called on EVERY hit, including thread-local ones, because the
    /// window this closes is "the allocation moved on and the address
    /// did not". An implementation that cannot answer cheaply and
    /// exactly must answer conservatively (`false`) rather than
    /// optimistically.
    fn still_holds(&self, host: &[u8]) -> bool;

    /// Bytes this entry retains, for the cache budget. Zero for an
    /// entry that aliases host memory it does not own.
    fn resident_bytes(&self) -> usize;
}

/// A process-wide cache and its per-thread mirror.
type Global<V> = Mutex<Option<HashMap<HostKey, Arc<V>>>>;
type Mirror<V> = LocalKey<std::cell::RefCell<HashMap<HostKey, Arc<V>>>>;

/// Byte budget shared by every resident cache.
///
/// One budget, because on unified memory two budgets are the same RAM
/// counted twice. `usize::MAX` by default: on Apple Silicon the weights
/// are already resident and the aliasing path retains nothing.
pub(crate) fn budget_bytes() -> usize {
    match std::env::var("FRINK_METAL_WEIGHT_CACHE_BYTES") {
        Ok(v) => v.parse().unwrap_or(usize::MAX),
        Err(_) => usize::MAX,
    }
}

/// Look `host` up, building and caching an entry when there is no valid
/// one.
///
/// The single reader of every resident cache: the thread-local mirror,
/// the process map, the freshness check, the budget and the insert are
/// written once here rather than once per cache, which is what stopped
/// one of them from having a check the other lacked.
pub(crate) fn get_or_build<V: Resident>(
    global: &Global<V>,
    mirror: &'static Mirror<V>,
    host: &[u8],
    build: impl FnOnce() -> Result<V, MetalError>,
) -> Result<Arc<V>, MetalError> {
    get_or_build_within(global, mirror, host, budget_bytes, build)
}

/// [`get_or_build`] with the budget stated rather than read from the
/// environment, so a test can exercise a tight budget without a
/// process-wide variable two tests would race over.
///
/// The budget is asked for only on a miss: a hit is the per-token path,
/// and reading the environment there was a `getenv` per resident
/// lookup, a dozen per layer per token.
fn get_or_build_within<V: Resident>(
    global: &Global<V>,
    mirror: &'static Mirror<V>,
    host: &[u8],
    budget: impl FnOnce() -> usize,
    build: impl FnOnce() -> Result<V, MetalError>,
) -> Result<Arc<V>, MetalError> {
    let key = HostKey::of(host);

    if let Some(cached) = mirror.with(|c| c.borrow().get(&key).cloned()) {
        if cached.still_holds(host) {
            return Ok(cached);
        }
        // The address was recycled under us. Drop the alias here and in
        // the process map, then rebuild: a stale entry is a miss, never
        // a hit.
        mirror.with(|c| {
            c.borrow_mut().remove(&key);
        });
    }

    let cached = {
        let mut guard = global.lock().unwrap();
        let cache = guard.get_or_insert_with(HashMap::new);
        match cache.get(&key).filter(|c| c.still_holds(host)) {
            Some(cached) => cached.clone(),
            None => {
                let budget = budget();
                let used: usize = cache.values().map(|b| b.resident_bytes()).sum();
                if used.saturating_add(host.len()) > budget {
                    // Drop everything and retry with a clean slate for
                    // this buffer. Better than silently re-uploading
                    // forever under a tight budget.
                    cache.clear();
                    mirror.with(|c| c.borrow_mut().clear());
                }
                let built = Arc::new(build()?);
                if built.resident_bytes() > budget {
                    // This buffer alone exceeds the budget: one-shot
                    // upload, do not cache.
                    return Ok(built);
                }
                cache.insert(key, built.clone());
                built
            }
        }
    };
    mirror.with(|c| {
        c.borrow_mut().insert(key, cached.clone());
    });
    Ok(cached)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A stand-in for a device buffer: it remembers the bytes it was
    /// built from, which is exactly what a real `MTLBuffer` holds.
    struct FakeBuffer {
        bytes: Vec<u8>,
        /// Distinguishes two entries built from equal-length host
        /// buffers, so a test can tell WHICH one came back.
        tag: u32,
    }

    impl Resident for FakeBuffer {
        fn still_holds(&self, host: &[u8]) -> bool {
            self.bytes == host
        }
        fn resident_bytes(&self) -> usize {
            self.bytes.len()
        }
    }

    /// The whole bug: one allocation is freed, the next one lands on
    /// its address with different contents, and the cache must not
    /// serve the first one's upload.
    ///
    /// The recycled address is FORCED rather than hoped for: the bytes
    /// under one address are overwritten in place, which is exactly
    /// what an allocator handing that address to a second tensor looks
    /// like to a cache that keys on the address.
    ///
    /// Sabotage: make `still_holds` return `true` unconditionally and
    /// this goes red with `tag` 1 where 2 is expected.
    #[test]
    fn an_address_reused_by_different_bytes_is_a_miss_not_a_hit() {
        static CACHE: Global<FakeBuffer> = Mutex::new(None);
        thread_local! {
            static TL: RefCell<HashMap<HostKey, Arc<FakeBuffer>>> = RefCell::new(HashMap::new());
        }
        let fetch = |host: &[u8], tag: u32| {
            get_or_build_within(
                &CACHE,
                &TL,
                host,
                || usize::MAX,
                || {
                    Ok(FakeBuffer {
                        bytes: host.to_vec(),
                        tag,
                    })
                },
            )
            .expect("build")
        };

        let mut region = vec![0xAAu8; 64];
        assert_eq!(fetch(&region, 1).tag, 1);
        assert_eq!(
            fetch(&region, 99).tag,
            1,
            "unchanged bytes at one address must still hit"
        );

        region.iter_mut().for_each(|b| *b = 0x55);
        let second = fetch(&region, 2);
        assert_eq!(
            second.tag, 2,
            "the address was recycled by different bytes, so the old upload must not be served"
        );
        assert_eq!(second.bytes, region);
    }

    /// A budget of zero means nothing is retained, which is the
    /// behaviour before any of these caches existed.
    #[test]
    fn a_zero_budget_retains_nothing() {
        static CACHE: Global<FakeBuffer> = Mutex::new(None);
        thread_local! {
            static TL: RefCell<HashMap<HostKey, Arc<FakeBuffer>>> = RefCell::new(HashMap::new());
        }
        let fetch = |host: &[u8], tag: u32| {
            get_or_build_within(
                &CACHE,
                &TL,
                host,
                || 0,
                || {
                    Ok(FakeBuffer {
                        bytes: host.to_vec(),
                        tag,
                    })
                },
            )
            .expect("build")
        };

        let region = vec![0x11u8; 32];
        assert_eq!(fetch(&region, 7).tag, 7);
        assert_eq!(
            fetch(&region, 8).tag,
            8,
            "nothing may be retained under a zero budget"
        );
    }
}
