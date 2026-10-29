//! A process-wide, byte-budgeted cache of interleaved ("repacked")
//! weight bytes, keyed by the identity of the mapping the bytes came
//! from.
//!
//! A repack rewrites a whole matrix into the row-interleaved layout the
//! `x4` / `x8` GEMV kernels read, so it costs a pass over every weight
//! byte. Done once per matrix that is the load-time cost llama.cpp pays
//! for its `repack` backend. Done once per CALL it is a full copy of the
//! matrix on every token, and that is what shipped: `apply_cpu_q8`, the
//! int-dot matvec the dense FFN gate/up and the MoE experts take, passed
//! a hand-written `/* uncacheable */ None` where every other matvec
//! passed `data.map_id()`. Profiled at one thread on TinyLlama Q8_0,
//! roughly 85% of a decode token was `pack_q8_0_matrix_x4` and the
//! `Arc` copy behind it, against under 10% in the GEMV it fed (#128).
//!
//! The `None` was a second, hand-restated copy of a decision
//! [`WeightBytes::map_id`] already makes: `Owned` and `Shared` bytes
//! answer `None` there, because their address is not an identity, and
//! `Mapped` bytes answer `Some`. Two places deciding one thing, with
//! nothing making them agree, is this repo's dominant bug shape, so the
//! typed lookups below take the [`WeightBytes`] and ask it themselves.
//! A call site can no longer say "uncacheable" on its own authority.
//!
//! # The budget
//!
//! Caching a packing RETAINS a second copy of a matrix that is already
//! mapped. Measured on TinyLlama-1.1B Q8_0, retaining the dense FFN
//! gate/up packings cost **+527 MB of peak footprint** (685 MB to 1213
//! MB), and that scales with gate/up bytes: an 8B checkpoint pays
//! several GB, and a resident MoE pays it once per expert that has ever
//! been routed to. Unbounded, it grows with the number of distinct
//! matrices the process touches, which for an MoE is unbounded in
//! practice.
//!
//! So there is ONE cache, not one per format, holding
//! [`budget_bytes`] at most, evicting least-recently-used entries to
//! stay under it. A matrix that does not fit is packed and returned
//! uncached, so pressure degrades to recomputation and never to a wrong
//! answer -- the same degradation
//! [`crate::expert_store::ExpertStore::acquire`] makes, for the same
//! reason.
//!
//! Five caches would be five budgets over one pool of RAM, which is the
//! defect the budget exists to close, so the format is part of the key
//! instead.
//!
//! The budget itself is DERIVED, in
//! [`crate::host_memory::derived_copy_budget`], from a live probe of the
//! host minus what `expert_store` has already committed. It can be zero,
//! and zero means nothing is ever retained: exactly the behaviour before
//! this cache existed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use super::WeightBytes;

/// Identity of the memory mapping a repacked buffer was built from.
///
/// The repack caches key on a weight's **address**, and an address is
/// only a stable identity for as long as the mapping that published it
/// is alive. Unmap one file and map another and the kernel will hand
/// the same address straight back -- a textbook ABA. The cache then
/// serves one matrix another matrix's interleaved bytes, which panicked
/// with an out-of-range slice when the two shapes differed and was
/// SILENT, i.e. wrong output, when they matched.
///
/// Holding a [`std::sync::Weak`] is what closes it, and it closes both
/// halves at once:
///
/// * while the `Weak` lives, the `Arc`'s control block cannot be
///   recycled, so [`Self::id`] is a unique name for exactly one mapping
///   for as long as the cache entry exists; and
/// * `upgrade()` succeeding proves the mapping itself is still alive,
///   which is what makes the address it published still mean what it
///   meant when the entry was written.
///
/// A dead `Weak` is therefore a *stale entry*, not a hit, and is
/// repacked and replaced. The `Weak` holds no mapping open, so nothing
/// here keeps a file resident.
#[derive(Clone)]
pub struct MapId {
    map: std::sync::Weak<memmap2::Mmap>,
    id: usize,
    offset: usize,
}

impl MapId {
    /// The identity of `range.start` inside `mmap`. Only
    /// [`WeightBytes::map_id`] builds one, and only for `Mapped` bytes.
    pub(super) fn of(mmap: &Arc<memmap2::Mmap>, offset: usize) -> Self {
        MapId {
            map: Arc::downgrade(mmap),
            id: Arc::as_ptr(mmap) as usize,
            offset,
        }
    }

    /// True when `other` names the same, still-live mapping.
    fn matches(&self, other: &MapId) -> bool {
        self.id == other.id
            && self.offset == other.offset
            && self
                .map
                .upgrade()
                .is_some_and(|m| Arc::as_ptr(&m) as usize == other.id)
    }

    /// The cache key this identity contributes to, for `format` at
    /// `rows x cols`.
    fn key(&self, format: Format, rows: usize, cols: usize) -> RepackKey {
        (format, self.id, self.offset, rows, cols)
    }
}

/// Which interleaved layout a cached packing is in.
///
/// Part of the KEY rather than the identity of a separate cache: one
/// budget over one map is the whole point, and five maps would be five
/// budgets spending the same RAM.
///
/// Carrying it in the key is defensive rather than load-bearing, and
/// saying so is the honest version: one mapping offset is one tensor,
/// a tensor has one quant kind, and each kind reaches exactly one
/// packer, so no call site today can ask for two formats at one
/// address. The key names the packing anyway, so that invariant is not
/// something a future format has to rediscover. A test cannot
/// distinguish it for the same reason it cannot happen, so there is no
/// test claiming to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Format {
    /// `block_q4_Kx8`
    Q4Kx8,
    /// `block_q5_Kx8`
    Q5Kx8,
    /// `block_q6_Kx8`
    Q6Kx8,
    /// `block_q8_0x4`
    Q8_0x4,
    /// `block_q4_0x4`
    Q4_0x4,
}

/// `(format, mapping id, byte offset, rows, cols)`.
///
/// `cols` is in the key because two tensors of equal row count and
/// unequal width are different matrices with different repacked lengths,
/// and the old `(address, rows)` key called them the same one.
type RepackKey = (Format, usize, usize, usize, usize);

/// Interleaved bytes, beside the mapping identity that makes the key
/// meaningful ([`MapId`]) and the recency stamp eviction orders by.
struct Entry {
    id: MapId,
    packed: Arc<[u8]>,
    /// Monotonic; smallest is least recently used. A stamp per touch
    /// rather than an LRU list, which is what
    /// [`crate::expert_store`] does and for the same reason: an O(n)
    /// scan is cheap at the few-hundred entries a checkpoint produces,
    /// and a list is another structure to keep in agreement.
    last_used: u64,
}

/// The one cache. See the module docs for why it is one and not five.
#[derive(Default)]
struct Cache {
    entries: HashMap<RepackKey, Entry>,
    /// Sum of `entries[..].packed.len()`, maintained on every insert and
    /// every eviction so the budget check is O(1).
    resident_bytes: usize,
    clock: u64,
}

impl Cache {
    /// Drops one entry and un-accounts its bytes.
    ///
    /// The only way an entry leaves the map. A `remove` that forgot to
    /// subtract would leak budget until the cache stopped caching
    /// anything, which is exactly the kind of silent divergence this
    /// repo keeps paying for, so there is one of these and everything
    /// calls it.
    fn evict(&mut self, key: &RepackKey) {
        if let Some(entry) = self.entries.remove(key) {
            self.resident_bytes = self.resident_bytes.saturating_sub(entry.packed.len());
        }
    }

    /// The least recently used key, or `None` when the map is empty.
    fn lru(&self) -> Option<RepackKey> {
        self.entries
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| *k)
    }

    /// Serves `key` if it holds a live packing for `id`, dropping a
    /// stale entry rather than returning it.
    ///
    /// A dead `Weak` means the mapping that published this address is
    /// gone and the address has been handed to somebody else, so the
    /// entry is a textbook ABA and must not be served.
    fn take_hit(&mut self, key: &RepackKey, id: &MapId) -> Option<Arc<[u8]>> {
        match self.entries.get_mut(key) {
            Some(entry) if entry.id.matches(id) => {
                self.clock += 1;
                entry.last_used = self.clock;
                Some(Arc::clone(&entry.packed))
            }
            Some(_) => {
                self.evict(key);
                None
            }
            None => None,
        }
    }

    /// Retains `packed` under `key` if the budget can hold it, evicting
    /// least-recently-used entries to make room.
    ///
    /// Returns without inserting when one packing alone exceeds the
    /// budget -- including when the budget is zero, which is how "never
    /// retain anything" is expressed. The caller already holds the
    /// packing, so declining costs a recomputation next time and
    /// nothing else.
    fn insert_within_budget(&mut self, key: RepackKey, id: MapId, packed: Arc<[u8]>) {
        let budget = budget_bytes();
        let size = packed.len();
        if size > budget {
            return;
        }
        while self.resident_bytes + size > budget {
            let Some(victim) = self.lru() else { break };
            self.evict(&victim);
        }
        // The loop can only exit early when the map is empty, and an
        // empty map holds zero bytes, so this cannot fail after it --
        // but assert rather than assume, because the budget is the
        // property the tests pin.
        if self.resident_bytes + size > budget {
            return;
        }
        self.clock += 1;
        self.resident_bytes += size;
        self.entries.insert(
            key,
            Entry {
                id,
                packed,
                last_used: self.clock,
            },
        );
    }
}

/// The process-wide cache, built on first use.
fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Cache::default()))
}

/// The one way any of this module takes the cache lock.
///
/// A poisoned lock is recovered from rather than propagated. The map
/// holds no invariant a panic can leave half-built: entries are
/// inserted whole, and every read re-checks the identity before trusting
/// the bytes. Propagating the poison instead would let one panic
/// anywhere in the process turn EVERY later matvec into a panic, which
/// is a much worse failure than serving a correct cached packing.
fn lock() -> MutexGuard<'static, Cache> {
    cache().lock().unwrap_or_else(|e| e.into_inner())
}

/// Bytes this cache may retain, decided once for the process.
///
/// `FRINK_REPACK_CACHE_BYTES` overrides it, and `0` is a legal value
/// meaning "never retain anything" -- the behaviour before this cache
/// existed, and the reason a memory-constrained host is expressible
/// rather than merely given a smaller number.
///
/// Otherwise it is DERIVED by
/// [`crate::host_memory::derived_copy_budget`] from what the host says
/// is available, less the standard fit headroom, less what
/// `expert_store` has already committed. That subtraction is the whole
/// relationship between this budget and the expert one: they are not
/// two independent numbers, they are one pool spent in a fixed order.
fn budget_bytes() -> usize {
    #[cfg(test)]
    {
        if let Some(bytes) = tests::budget_override() {
            return bytes;
        }
    }
    static BYTES: OnceLock<usize> = OnceLock::new();
    *BYTES.get_or_init(|| {
        if let Some(explicit) = std::env::var("FRINK_REPACK_CACHE_BYTES")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
        {
            return usize::try_from(explicit).unwrap_or(usize::MAX);
        }
        let derived = crate::host_memory::derived_copy_budget(
            crate::host_memory::available_bytes(),
            crate::host_memory::FIT_HEADROOM_BYTES,
            crate::expert_store::committed_expert_bytes(),
        );
        usize::try_from(derived).unwrap_or(usize::MAX)
    })
}

/// The one lookup every format's repack shares.
///
/// `id` is `None` for bytes whose address may be recycled under us
/// (owned buffers, and an expert store's leases -- see
/// [`WeightBytes::map_id`]), and those always repack. Private: the only
/// way for a matvec to reach this is through a typed lookup below, which
/// derives `id` from the [`WeightBytes`] rather than accepting one.
fn get_or_repack(
    format: Format,
    id: Option<MapId>,
    rows: usize,
    cols: usize,
    repack: impl FnOnce() -> Vec<u8>,
) -> Arc<[u8]> {
    let Some(id) = id else {
        return Arc::from(repack().into_boxed_slice());
    };
    let key = id.key(format, rows, cols);
    if let Some(hit) = lock().take_hit(&key, &id) {
        return hit;
    }
    let arc: Arc<[u8]> = Arc::from(repack().into_boxed_slice());
    let mut cache = lock();
    // Another thread may have won the race; prefer the existing entry,
    // but only if it is one this caller would have accepted above.
    match cache.take_hit(&key, &id) {
        Some(hit) => hit,
        None => {
            cache.insert_within_budget(key, id, Arc::clone(&arc));
            arc
        }
    }
}

/// Whether the packing of `data` (as `format` at `rows x cols`) is
/// currently held under a live identity. Tests only.
#[cfg(test)]
fn is_cached(format: Format, data: &WeightBytes, rows: usize, cols: usize) -> bool {
    let Some(id) = data.map_id() else {
        return false;
    };
    lock()
        .entries
        .get(&id.key(format, rows, cols))
        .is_some_and(|e| e.id.matches(&id))
}

/// Bytes the cache currently holds. Diagnostics and tests.
#[cfg(test)]
fn resident_bytes() -> usize {
    lock().resident_bytes
}

/// Empties the cache. Tests only: the cache is process-wide, so a test
/// that asserts about the budget has to start from a known footprint.
#[cfg(test)]
fn clear() {
    let mut cache = lock();
    cache.entries.clear();
    cache.resident_bytes = 0;
}

pub(super) fn get_or_repack_q4k(data: &WeightBytes, rows: usize, cols: usize) -> Arc<[u8]> {
    get_or_repack(Format::Q4Kx8, data.map_id(), rows, cols, || {
        frink_quant::pack_q4_k_matrix_x8(
            data.as_slice(),
            rows,
            cols,
            frink_quant::q4_kx8_interleave(),
        )
    })
}

pub(super) fn get_or_repack_q5k(data: &WeightBytes, rows: usize, cols: usize) -> Arc<[u8]> {
    get_or_repack(Format::Q5Kx8, data.map_id(), rows, cols, || {
        frink_quant::pack_q5_k_matrix_x8(
            data.as_slice(),
            rows,
            cols,
            frink_quant::q5_kx8_interleave(),
        )
    })
}

pub(super) fn get_or_repack_q6k(data: &WeightBytes, rows: usize, cols: usize) -> Arc<[u8]> {
    get_or_repack(Format::Q6Kx8, data.map_id(), rows, cols, || {
        frink_quant::pack_q6_k_matrix_x8(
            data.as_slice(),
            rows,
            cols,
            frink_quant::q6_kx8_interleave(),
        )
    })
}

pub(super) fn get_or_repack_q8x4(data: &WeightBytes, rows: usize, cols: usize) -> Arc<[u8]> {
    get_or_repack(Format::Q8_0x4, data.map_id(), rows, cols, || {
        frink_quant::pack_q8_0_matrix_x4(
            data.as_slice(),
            rows,
            cols,
            frink_quant::q8_0x4_interleave(),
        )
    })
}

pub(super) fn get_or_repack_q4_0x4(data: &WeightBytes, rows: usize, cols: usize) -> Arc<[u8]> {
    get_or_repack(Format::Q4_0x4, data.map_id(), rows, cols, || {
        frink_quant::pack_q4_0_matrix_x4(
            data.as_slice(),
            rows,
            cols,
            frink_quant::q4_0x4_interleave(),
        )
    })
}

/// Whether the `Q8_0x4` packing of `data` is held in the cache. What
/// the `apply_cpu_q8` test below asks after one call.
#[cfg(test)]
pub(super) fn q8x4_is_cached(data: &WeightBytes, rows: usize, cols: usize) -> bool {
    is_cached(Format::Q8_0x4, data, rows, cols)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{f16_le, ForceIntDot};
    use super::super::{QuantKind, WeightMatrix};
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `usize::MAX` means "no override": a real budget of `usize::MAX`
    /// is not reachable, since it is a quarter of a byte count that
    /// came out of a memory probe.
    static BUDGET_OVERRIDE: AtomicUsize = AtomicUsize::new(usize::MAX);

    /// The budget [`super::budget_bytes`] should report, if a test has
    /// pinned one.
    pub(super) fn budget_override() -> Option<usize> {
        match BUDGET_OVERRIDE.load(Ordering::Acquire) {
            usize::MAX => None,
            bytes => Some(bytes),
        }
    }

    /// Pins the cache budget, and empties the cache, for the lifetime of
    /// the guard.
    ///
    /// Both halves are necessary and both are here rather than at the
    /// call sites: the cache and the budget are process-wide, so a test
    /// that asserts about either has to own both, and two tests holding
    /// different budgets at once would see each other's. The mutex is
    /// what serializes them, the same shape as
    /// `weight_matrix::tests::ForceIntDot`.
    pub(super) struct ForceBudget {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl ForceBudget {
        fn new(bytes: usize) -> Self {
            static LOCK: Mutex<()> = Mutex::new(());
            let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
            BUDGET_OVERRIDE.store(bytes, Ordering::Release);
            clear();
            ForceBudget { _lock: lock }
        }

        /// Enough for any fixture here: the budget is not what the test
        /// is about.
        fn generous() -> Self {
            Self::new(1 << 20)
        }
    }

    impl Drop for ForceBudget {
        fn drop(&mut self) {
            clear();
            BUDGET_OVERRIDE.store(usize::MAX, Ordering::Release);
        }
    }

    // -----------------------------------------------------------------
    // Repack cache identity (see `MapId`)
    //
    // The bug these cover: the caches used to key on `(address, rows)`
    // and gate on an `address_is_stable() -> bool`. Drop one mmap, make
    // another, and the kernel hands the same address back, so the cache
    // served the previous matrix's interleaved bytes -- an out-of-range
    // panic when the shapes differed, silent wrong output when they
    // matched.
    //
    // Address reuse is the OS's decision and cannot be demanded from a
    // test, so these do not wait for it. They fabricate exactly what the
    // cache would SEE in that moment -- a key that collides while the
    // mapping behind it is gone, or while the width differs -- and
    // assert the cache refuses to serve it.
    // -----------------------------------------------------------------

    /// Writes `bytes` to a temp file and maps it. The caller holds the
    /// `Arc`, so when the mapping dies is explicit, which is the whole
    /// subject of these tests.
    fn mapped(tag: &str, bytes: &[u8]) -> (Arc<memmap2::Mmap>, WeightBytes) {
        let path = std::env::temp_dir().join(format!(
            "frink_repack_{tag}_{}_{:?}.bin",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, bytes).expect("write fixture");
        let file = std::fs::File::open(&path).expect("open fixture");
        // SAFETY: the file was written and closed above, is named for
        // this process and thread, and nothing mutates it while mapped.
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).expect("map fixture") });
        let _ = std::fs::remove_file(&path);
        let view = WeightBytes::Mapped {
            mmap: Arc::clone(&mmap),
            range: 0..bytes.len(),
        };
        (mmap, view)
    }

    /// Q8_0 bytes with finite scales, `rows * cols/32` blocks.
    fn q8_0_matrix_bytes(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let mut data = Vec::with_capacity(rows * (cols / 32) * 34);
        for _ in 0..rows * (cols / 32) {
            data.extend_from_slice(&f16_le(0.02 + f32::from(next()) * 0.0004));
            for _ in 0..32 {
                data.push(next());
            }
        }
        data
    }

    /// A `MapId` is only an identity while its mapping is alive. This is
    /// the check the old boolean could not express, and it is the one
    /// thing standing between the cache and an ABA.
    #[test]
    fn map_id_stops_matching_once_its_mapping_is_dropped() {
        let (mmap, view) = mapped("live", &q8_0_matrix_bytes(4, 32, 7));
        let id = view.map_id().expect("Mapped bytes must have an identity");
        let held = id.clone();
        assert!(
            held.matches(&id),
            "a live mapping must match its own identity"
        );

        // Everything that could witness the mapping is gone: this is
        // precisely the moment the address becomes reusable.
        drop(view);
        drop(mmap);
        assert!(
            !held.matches(&id),
            "an identity whose mapping is dead must not match, or the \
             cache will trust an address the kernel has already reissued"
        );
    }

    /// A cache entry left behind by a dead mapping must be replaced, not
    /// served. Fabricates the entry rather than waiting on the OS to
    /// reissue an address; the entry is byte-for-byte what the old code
    /// would have left there.
    #[test]
    fn stale_repack_entry_is_replaced_not_served() {
        let _budget = ForceBudget::generous();
        let (rows, cols) = (8usize, 64usize);
        let bytes = q8_0_matrix_bytes(rows, cols, 11);
        let (_mmap, view) = mapped("stale", &bytes);
        let id = view.map_id().expect("Mapped bytes must have an identity");

        // Some other matrix's packing, parked at the key this live
        // matrix will look up, under an identity that can never upgrade.
        let poison = vec![0xABu8; 16];
        {
            let mut cache = lock();
            cache.insert_within_budget(
                id.key(Format::Q8_0x4, rows, cols),
                MapId {
                    map: std::sync::Weak::new(),
                    id: id.id,
                    offset: id.offset,
                },
                Arc::from(poison.clone().into_boxed_slice()),
            );
        }

        let got = get_or_repack_q8x4(&view, rows, cols);
        let want = frink_quant::pack_q8_0_matrix_x4(
            view.as_slice(),
            rows,
            cols,
            frink_quant::q8_0x4_interleave(),
        );
        assert_ne!(&got[..], &poison[..], "served a dead mapping's bytes");
        assert_eq!(&got[..], &want[..], "stale entry was not repacked");

        // And the dead entry is gone rather than pinning a control block.
        let cache = lock();
        let entry = cache
            .entries
            .get(&id.key(Format::Q8_0x4, rows, cols))
            .expect("the live packing should now be cached");
        assert!(
            entry.id.matches(&id),
            "the replacement entry must carry the LIVE identity"
        );
    }

    /// Two widths at one address are two matrices. The old key was
    /// `(address, rows)`, so a 576x576 and a 576x1536 collided and the
    /// second was served the first's shorter buffer.
    ///
    /// Drives the primitive directly: the collision needs two byte
    /// buffers under ONE identity, which is exactly what the typed
    /// lookups exist to make impossible for production code.
    #[test]
    fn repack_key_separates_two_widths_at_one_address() {
        let _budget = ForceBudget::generous();
        let rows = 8usize;
        let narrow = q8_0_matrix_bytes(rows, 32, 3);
        let wide = q8_0_matrix_bytes(rows, 64, 5);
        let (_mmap, view) = mapped("widths", &narrow);
        let id = view.map_id().expect("Mapped bytes must have an identity");

        let il = frink_quant::q8_0x4_interleave();
        let a = get_or_repack(Format::Q8_0x4, Some(id.clone()), rows, 32, || {
            frink_quant::pack_q8_0_matrix_x4(&narrow, rows, 32, il)
        });
        let b = get_or_repack(Format::Q8_0x4, Some(id.clone()), rows, 64, || {
            frink_quant::pack_q8_0_matrix_x4(&wide, rows, 64, il)
        });
        assert_eq!(
            &a[..],
            &frink_quant::pack_q8_0_matrix_x4(&narrow, rows, 32, il)[..]
        );
        assert_eq!(
            &b[..],
            &frink_quant::pack_q8_0_matrix_x4(&wide, rows, 64, il)[..],
            "the wider matrix was served the narrower one's packing"
        );
        assert!(b.len() > a.len(), "widths must not share a cache entry");
    }

    /// Owned buffers and expert-store leases are never cacheable. The
    /// lease is the interesting one: its allocation stays alive and keeps
    /// its address while its CONTENTS are replaced by another expert's,
    /// so no liveness check could rescue it.
    #[test]
    fn map_id_is_none_for_owned_and_shared_bytes() {
        let owned = WeightBytes::Owned(q8_0_matrix_bytes(4, 32, 9));
        assert!(owned.map_id().is_none(), "an owned Vec's address is reused");

        let buf = Arc::new(crate::expert_store::ExpertBytes::from(q8_0_matrix_bytes(
            4, 32, 13,
        )));
        let leased = WeightBytes::Shared {
            buf,
            range: 0..34 * 4,
        };
        assert!(
            leased.map_id().is_none(),
            "an expert lease keeps its address across a content swap"
        );
    }

    /// A mapped matrix is packed once, and a second lookup is a hit that
    /// hands back the SAME allocation. An owned matrix is never cached.
    /// Both halves are the contract the matvecs rely on.
    #[test]
    fn a_mapped_matrix_is_packed_once_and_an_owned_one_never_cached() {
        let _budget = ForceBudget::generous();
        let (rows, cols) = (8usize, 64usize);
        let bytes = q8_0_matrix_bytes(rows, cols, 17);
        let (_mmap, view) = mapped("once", &bytes);
        assert!(!q8x4_is_cached(&view, rows, cols));
        let first = get_or_repack_q8x4(&view, rows, cols);
        assert!(q8x4_is_cached(&view, rows, cols));
        let second = get_or_repack_q8x4(&view, rows, cols);
        assert!(
            Arc::ptr_eq(&first, &second),
            "a second lookup of a live mapping must be a cache hit"
        );

        let owned = WeightBytes::Owned(bytes);
        let a = get_or_repack_q8x4(&owned, rows, cols);
        let b = get_or_repack_q8x4(&owned, rows, cols);
        assert!(!q8x4_is_cached(&owned, rows, cols));
        assert!(
            !Arc::ptr_eq(&a, &b),
            "owned bytes have no identity and must repack every time"
        );
    }

    /// The #128 regression, at the call site that carried it.
    ///
    /// `apply_cpu_q8` is the int-dot matvec the dense FFN gate/up and
    /// every MoE expert take, and it repacked its matrix on EVERY call:
    /// it passed a hand-written `None` identity where the other matvecs
    /// passed `data.map_id()`. Measured at one thread on TinyLlama Q8_0,
    /// that repack was ~85% of a decode token. The typed lookup now
    /// derives the identity itself, and this asserts the packing of a
    /// mapped matrix is in the cache after one call through that path.
    ///
    /// Sabotage: make `get_or_repack_q8x4` pass `None` instead of
    /// `data.map_id()` and this goes red.
    ///
    /// Two preconditions, both of which have to hold before there is
    /// anything to assert. The budget must be non-zero, or nothing is
    /// retained by design. And `apply_cpu_q8` is a MATVEC, so it only
    /// exists on a host that takes the matvec half of the int-dot tier
    /// (#152 turned that half off on x86, where it measured 4x to 8.8x
    /// slower than the AVX2 f32 dot). Where it does not, the guard skips
    /// rather than asserting a `None` is a `Some`.
    #[test]
    fn apply_cpu_q8_caches_the_packing_of_a_mapped_matrix() {
        let _force = ForceIntDot::new(true);
        let _budget = ForceBudget::generous();
        if !super::super::cpu_int_dot_for(super::super::IntDotShape::Matvec) {
            return;
        }
        let (rows, cols) = (8usize, 64usize);
        let bytes = q8_0_matrix_bytes(rows, cols, 23);
        let (_mmap, view) = mapped("apply_q8", &bytes);
        let m = WeightMatrix::Quantized {
            data: view,
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        let WeightMatrix::Quantized { data, .. } = &m else {
            unreachable!()
        };
        assert!(!q8x4_is_cached(data, rows, cols), "fresh mapping");

        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let act = frink_quant::quantize_activations_q8(&x);
        let out = m
            .apply_cpu_q8(&act)
            .expect("Q8_0 with int-dot on takes the interleaved path");
        assert_eq!(out.len(), rows);
        assert!(
            q8x4_is_cached(data, rows, cols),
            "apply_cpu_q8 repacked a mapped matrix without caching it: \
             that is a full copy of the matrix per token (#128)"
        );
    }

    // -----------------------------------------------------------------
    // The budget
    //
    // The cache retains a SECOND copy of a matrix that is already
    // mapped. Measured at +527 MB of peak footprint on TinyLlama-1.1B
    // Q8_0 for the dense FFN gate/up packings alone, and a resident MoE
    // pays that once per expert ever routed to. These tests are the
    // bound on that.
    // -----------------------------------------------------------------

    /// The interleave the Q8_0 fixtures pack with, spelled once.
    fn il() -> usize {
        frink_quant::q8_0x4_interleave()
    }

    /// Ten distinct matrices under a budget that holds three of them.
    ///
    /// Three separate properties, because a cache can fail each one on
    /// its own:
    ///
    /// 1. it never exceeds the budget, at any step;
    /// 2. it still answers correctly for what it dropped;
    /// 3. it EVICTS rather than stops caching. A cache that filled up
    ///    and then refused every later matrix would satisfy (1) and (2)
    ///    and be useless: decode walks every layer, so the first three
    ///    matrices would be cached forever and every other matrix would
    ///    repack per token, which is #128 again for all but three of
    ///    them.
    ///
    /// Sabotage: insert unconditionally and (1) goes red; delete the
    /// eviction loop and (3) goes red. Both were run.
    #[test]
    fn the_cache_never_exceeds_its_budget() {
        let (rows, cols) = (8usize, 64usize);
        let one_packing =
            frink_quant::pack_q8_0_matrix_x4(&q8_0_matrix_bytes(rows, cols, 1), rows, cols, il())
                .len();
        // Room for three packings, and not a byte more.
        let budget = one_packing * 3;
        let _guard = ForceBudget::new(budget);

        let mut held = Vec::new();
        for seed in 0..10u32 {
            let bytes = q8_0_matrix_bytes(rows, cols, seed + 1);
            let (mmap, view) = mapped(&format!("budget{seed}"), &bytes);
            let got = get_or_repack_q8x4(&view, rows, cols);
            assert_eq!(
                &got[..],
                &frink_quant::pack_q8_0_matrix_x4(&bytes, rows, cols, il())[..],
                "an evicting cache must still answer correctly"
            );
            assert!(
                resident_bytes() <= budget,
                "cache grew past its budget at matrix {seed}: {} > {budget}",
                resident_bytes()
            );
            // Hold the mappings so no address is reused mid-test, which
            // would make an eviction indistinguishable from an ABA drop.
            held.push((mmap, view));
        }
        assert!(resident_bytes() <= budget);
        assert!(
            resident_bytes() >= one_packing,
            "a budget that fits three packings must be holding some"
        );

        // (3): the LAST matrix is resident and the FIRST is not, which
        // only an evicting cache can manage under this budget.
        let (_, last) = held.last().expect("ten matrices were packed");
        assert!(
            q8x4_is_cached(last, rows, cols),
            "the most recent matrix must be cached: a cache that stops \
             caching once full leaves every later matrix repacking per \
             token, which is the #128 defect for all but the first few"
        );
        let (_, first) = &held[0];
        assert!(
            !q8x4_is_cached(first, rows, cols),
            "ten packings into a three-packing budget must have evicted \
             the least recently used one"
        );
    }

    /// The batched GEMM path answers the same with the cache holding its
    /// packing and with the cache disabled.
    ///
    /// This is where #152 and #158 have to compose. #152 turns the
    /// batch half of the int-dot tier ON for x86, so a prefill there now
    /// repacks matrices that nothing repacked before; #158 bounds what
    /// those packings may retain. They meet at `get_or_repack_*`, which
    /// is the ONE budgeted lookup either half reaches -- the batch path
    /// opens no cache of its own, so there is one pool, not two.
    ///
    /// What must hold across that meeting is that the budget decides
    /// only where the interleaved bytes LIVE, never what they are. A
    /// miss under a tight budget returns a fresh packing and the GEMM
    /// must produce bit-identical output, or a memory-constrained host
    /// would silently answer differently from a roomy one.
    ///
    /// So the generous side runs the GEMM TWICE: once cold, which
    /// misses and inserts, and once warm, which is served the retained
    /// packing. Comparing warm against cold is what puts the hit path
    /// under test; comparing either against the zero-budget run is what
    /// puts the budget under test. A first draft compared one cold run
    /// against one zero-budget run and survived zeroing the miss path,
    /// because that corrupts both sides identically -- it asserted that
    /// two equally wrong answers agreed.
    ///
    /// Sabotage: return zeroed bytes from `Cache::take_hit`, and this
    /// goes red where the miss-path version did not.
    #[test]
    fn the_batch_path_answers_the_same_with_the_cache_full_and_disabled() {
        let _force = ForceIntDot::new(true);
        if !super::super::cpu_int_dot_for(super::super::IntDotShape::BatchGemm) {
            return;
        }
        // Two row-groups plus a tail, and a batch that straddles the
        // 4-wide activation quad, so the GEMM takes both its full-tile
        // and partial-tile paths under each budget.
        let (rows, cols, batch) = (19usize, 64usize, 6usize);
        let bytes = q8_0_matrix_bytes(rows, cols, 71);
        let (_mmap, view) = mapped("batch_budget", &bytes);
        let m = WeightMatrix::Quantized {
            data: view,
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        let x: Vec<f32> = (0..batch * cols)
            .map(|i| ((i as f32) * 0.013 - 0.7).sin() * 1.4)
            .collect();

        let cached = {
            let _budget = ForceBudget::generous();
            let cold = m.apply_batch(&x, batch);
            assert!(
                resident_bytes() > 0,
                "a generous budget retained nothing, so the second call \
                 below would miss too and the hit path would go untested"
            );
            let warm = m.apply_batch(&x, batch);
            assert_eq!(
                cold.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                warm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "the retained packing served a different answer than the \
                 one that built it"
            );
            warm
        };
        let uncached = {
            let _budget = ForceBudget::new(0);
            let out = m.apply_batch(&x, batch);
            assert_eq!(resident_bytes(), 0, "a zero budget retained something");
            out
        };

        assert_eq!(
            cached.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            uncached.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the repack budget changed the answer, not just where the \
             interleaved bytes live"
        );
    }

    /// A zero budget retains nothing, which is the behaviour before the
    /// cache existed: every call repacks, into its own allocation, and
    /// the answer is unchanged.
    ///
    /// This is what makes the memory-constrained host expressible rather
    /// than merely given a smaller number.
    ///
    /// Sabotage: make `insert_within_budget` skip its `size > budget`
    /// return and this goes red.
    #[test]
    fn a_zero_budget_retains_nothing_and_matches_the_pre_cache_behaviour() {
        let _guard = ForceBudget::new(0);
        let (rows, cols) = (8usize, 64usize);
        let bytes = q8_0_matrix_bytes(rows, cols, 29);
        let (_mmap, view) = mapped("zero_budget", &bytes);

        let first = get_or_repack_q8x4(&view, rows, cols);
        let second = get_or_repack_q8x4(&view, rows, cols);
        assert_eq!(resident_bytes(), 0, "a zero budget retained something");
        assert!(!q8x4_is_cached(&view, rows, cols));
        assert!(
            !Arc::ptr_eq(&first, &second),
            "a zero budget must repack every call, as the engine did \
             before this cache existed"
        );
        let want = frink_quant::pack_q8_0_matrix_x4(&bytes, rows, cols, il());
        assert_eq!(&first[..], &want[..]);
        assert_eq!(&second[..], &want[..], "same bytes, different allocation");
    }

    /// Every format spends the SAME budget. Five caches would be five
    /// budgets over one pool of RAM, which is the defect
    /// `expert_store`'s single-holder rule exists to prevent.
    ///
    /// Sabotage: ignore the budget in `insert_within_budget`, or delete
    /// its eviction loop, and this goes red. Collapsing `Format` out of
    /// the key does NOT turn it red, and the doc on `Format` says why:
    /// the two fixtures are two mappings, so their keys differ with or
    /// without it.
    #[test]
    fn every_format_spends_one_budget() {
        let (rows, cols) = (8usize, 64usize);
        let bytes = q8_0_matrix_bytes(rows, cols, 31);
        let q8_len = frink_quant::pack_q8_0_matrix_x4(&bytes, rows, cols, il()).len();
        let _guard = ForceBudget::new(q8_len);

        let (_mmap, view) = mapped("one_budget_q8", &bytes);
        let _ = get_or_repack_q8x4(&view, rows, cols);
        assert!(q8x4_is_cached(&view, rows, cols), "the budget holds one");

        // A Q4_0 packing of a DIFFERENT mapping, into a budget with room
        // for exactly one entry.
        let q4_bytes = vec![7u8; rows * (cols / 32) * 18];
        let (_mmap4, view4) = mapped("one_budget_q4", &q4_bytes);
        let _ = get_or_repack_q4_0x4(&view4, rows, cols);

        assert!(
            resident_bytes() <= q8_len,
            "the two formats spent one budget, not two"
        );
        assert!(
            !q8x4_is_cached(&view, rows, cols),
            "the Q4_0 packing must have displaced the Q8_0 one"
        );
    }
}
