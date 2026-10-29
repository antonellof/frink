//! The paged half of the KV store: one shared physical block store per
//! layer, the per-sequence block tables that address into it, and the
//! page groups the server leases a block in every layer through.

use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::kv_store::{KvBuf, KvDtype, KvRows};

use super::KvCache;

/// The other half of PagedAttention that `KvBlockPool`/`KvCache::with_pool`
/// deliberately don't implement (see this module's doc comment): real,
/// *shared* physical block storage that many sequences' block tables can
/// address into, instead of each `KvCache` still owning its own private,
/// contiguous `Vec`. `KvBlockPool` only ever bounds a *count* of blocks
/// each cache may grow to; `PagedKvStore` is the actual backing memory,
/// and a sequence's `PagedKvCache` holds a block table (an ordered list
/// of block IDs into this shared store) instead of owning K/V data
/// directly. This is what makes non-contiguous-block reads during
/// attention (`causal_gqa_attention_paged`, in `attention.rs`) possible
/// at all -- `causal_gqa_attention`'s existing contiguous-slice read
/// pattern has no way to express "position 37 lives in block 12, cached
/// out of order relative to block 5."
pub struct PagedKvStore {
    block_size: usize,
    n_kv_heads: usize,
    /// K head width; see [`KvCache::head_dim`].
    head_dim: usize,
    /// V head width; see [`KvCache::v_head_dim`].
    v_head_dim: usize,
    k: KvBuf, // [total_blocks * block_size, n_kv_heads, head_dim], flattened
    v: KvBuf, // [total_blocks * block_size, n_kv_heads, v_head_dim], flattened
    free_block_ids: Vec<usize>,
}

impl PagedKvStore {
    /// One width for K and V; [`Self::new_split`] is the general form.
    pub fn new(block_size: usize, total_blocks: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self::new_split(block_size, total_blocks, n_kv_heads, head_dim, head_dim)
    }

    pub fn new_split(
        block_size: usize,
        total_blocks: usize,
        n_kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
    ) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        PagedKvStore {
            block_size,
            n_kv_heads,
            head_dim,
            v_head_dim,
            k: KvBuf::zeroed(
                KvDtype::F32,
                total_blocks * block_size * n_kv_heads * head_dim,
            ),
            v: KvBuf::zeroed(
                KvDtype::F32,
                total_blocks * block_size * n_kv_heads * v_head_dim,
            ),
            // Pushed in descending order so `pop()` hands out ascending
            // block IDs -- not load-bearing for correctness (any free ID
            // works), just makes manual debugging/inspection saner.
            free_block_ids: (0..total_blocks).rev().collect(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn free_block_count(&self) -> usize {
        self.free_block_ids.len()
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn v_head_dim(&self) -> usize {
        self.v_head_dim
    }

    /// One position's K row: `n_kv_heads * head_dim` elements.
    pub fn k_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// One position's V row: `n_kv_heads * v_head_dim` elements.
    pub fn v_width(&self) -> usize {
        self.n_kv_heads * self.v_head_dim
    }

    fn acquire_block(&mut self) -> Option<usize> {
        self.free_block_ids.pop()
    }

    fn release_block(&mut self, id: usize) {
        self.free_block_ids.push(id);
    }

    fn k_elems_per_block(&self) -> usize {
        self.block_size * self.k_width()
    }

    fn v_elems_per_block(&self) -> usize {
        self.block_size * self.v_width()
    }

    /// One position's K (or V) row within block `id` at `offset` (0-based
    /// within the block) -- `[n_kv_heads * head_dim]` long. Used by
    /// `causal_gqa_attention_paged` to read attention inputs directly out
    /// of shared physical storage via a block table, and by
    /// `PagedKvCache::push` to write a new position into it.
    pub fn k_row(&self, id: usize, offset: usize) -> KvRows<'_> {
        let width = self.k_width();
        let start = id * self.k_elems_per_block() + offset * width;
        self.k.view().slice(start..start + width)
    }

    pub fn v_row(&self, id: usize, offset: usize) -> KvRows<'_> {
        let width = self.v_width();
        let start = id * self.v_elems_per_block() + offset * width;
        self.v.view().slice(start..start + width)
    }

    /// The same store at `dtype`. Only before any sequence has written
    /// a row, for the reason [`KvCache::with_dtype`] gives.
    pub fn with_dtype(mut self, dtype: KvDtype) -> Self {
        // A zero-width store (a recurrent layer's: no rows at all) has
        // nothing to convert and no block count to recover from its
        // buffer length.
        assert!(
            self.k_elems_per_block() == 0
                || self.free_block_ids.len() == self.k.len() / self.k_elems_per_block(),
            "with_dtype on a paged store with blocks in use"
        );
        if self.k.dtype() != dtype {
            self.k = KvBuf::zeroed(dtype, self.k.len());
            self.v = KvBuf::zeroed(dtype, self.v.len());
        }
        self
    }

    /// What this store's rows are stored as.
    pub fn dtype(&self) -> KvDtype {
        self.k.dtype()
    }

    /// Copies every position of block `src` over block `dst`.
    ///
    /// The one operation copy-on-write needs and nothing else does. A
    /// sequence that forks shares its full blocks and copies only the
    /// PART-FULL tail, because that is the single block two forks would
    /// both write into. Sharing it instead would let one fork's token
    /// appear in the other's context, which is a wrong answer served
    /// with a 200 rather than a crash.
    ///
    /// The whole block is copied rather than the written prefix of it:
    /// the unwritten rows are within the block either way, and a
    /// length-aware copy would need the caller's `seq_len`, which is
    /// the one number this type deliberately does not hold.
    pub fn copy_block(&mut self, src: usize, dst: usize) {
        assert_ne!(src, dst, "copy_block onto itself");
        let k_span = self.k_elems_per_block();
        let (ks, kd) = (src * k_span, dst * k_span);
        self.k.copy_within(ks..ks + k_span, kd);
        let v_span = self.v_elems_per_block();
        let (vs, vd) = (src * v_span, dst * v_span);
        self.v.copy_within(vs..vs + v_span, vd);
    }

    fn write_k_row(&mut self, id: usize, offset: usize, row: KvRows<'_>) {
        let width = self.k_width();
        assert_eq!(row.len(), width);
        let start = id * self.k_elems_per_block() + offset * width;
        self.k.write_rows(start, row);
    }

    fn write_v_row(&mut self, id: usize, offset: usize, row: KvRows<'_>) {
        let width = self.v_width();
        assert_eq!(row.len(), width);
        let start = id * self.v_elems_per_block() + offset * width;
        self.v.write_rows(start, row);
    }
}

/// Returned when a `PagedKvCache` needs another block but its
/// `PagedKvStore` has none free -- the paged-storage analog of
/// `KvPoolExhausted`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedStoreExhausted;

impl std::fmt::Display for PagedStoreExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "paged KV store exhausted: no free blocks remain")
    }
}

impl std::error::Error for PagedStoreExhausted {}

/// One sequence's view into a shared `PagedKvStore`: a block table
/// (which physical blocks this sequence's positions live in, in order)
/// plus how many positions have been written so far. Unlike `KvCache`,
/// this holds no K/V data itself -- every read and write goes through
/// the shared store.
#[derive(Debug, Clone, Default)]
pub struct PagedKvCache {
    block_table: Vec<usize>,
    seq_len: usize,
    /// A recurrent layer's state, per sequence and NOT paged: the store
    /// holds rows and a Mamba layer has none. Same rules as
    /// `KvCache::recurrent`.
    pub recurrent: Option<crate::recurrent_state::RecurrentState>,
}

impl PagedKvCache {
    pub fn new() -> Self {
        PagedKvCache {
            block_table: Vec::new(),
            seq_len: 0,
            recurrent: None,
        }
    }

    /// See `KvCache::can_truncate_to`.
    pub fn can_truncate_to(&self, new_seq_len: usize) -> bool {
        self.recurrent.is_none() || new_seq_len == 0 || new_seq_len == self.seq_len
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn block_table(&self) -> &[usize] {
        &self.block_table
    }

    /// Appends one position's key/value vectors, acquiring a new block
    /// from `store` first if the current tail block is full (or none
    /// held yet). Mirrors `KvCache::push`'s signature/semantics exactly,
    /// just against shared storage instead of a private buffer.
    pub fn push(
        &mut self,
        store: &mut PagedKvStore,
        k_step: &[f32],
        v_step: &[f32],
    ) -> Result<(), PagedStoreExhausted> {
        self.push_rows(store, KvRows::F32(k_step), KvRows::F32(v_step))
    }

    /// [`Self::push`] for a row already at some store's dtype, so a row
    /// moved from one f16 store to another is copied rather than
    /// widened and rounded again.
    fn push_rows(
        &mut self,
        store: &mut PagedKvStore,
        k_step: KvRows<'_>,
        v_step: KvRows<'_>,
    ) -> Result<(), PagedStoreExhausted> {
        let block_size = store.block_size();
        let offset_in_block = self.seq_len % block_size;
        // Index by position rather than taking the tail block: a
        // sequence that pre-reserved (see `reserve`) already holds the
        // block this position belongs in, and appending another would
        // both leak a block and write the row in the wrong place.
        let block_index = self.seq_len / block_size;
        if block_index >= self.block_table.len() {
            let id = store.acquire_block().ok_or(PagedStoreExhausted)?;
            self.block_table.push(id);
        }
        let block_id = self.block_table[block_index];
        store.write_k_row(block_id, offset_in_block, k_step);
        store.write_v_row(block_id, offset_in_block, v_step);
        self.seq_len += 1;
        Ok(())
    }

    /// Appends a block the caller already owns, without taking one from
    /// the store.
    ///
    /// This is how a sliding window recycles. A block whose positions
    /// have fallen behind the window is never read again -- the paged
    /// attention kernel indexes `block_table[t / block_size]` only for
    /// `t >= seq_len - window` -- so its storage can back a *later*
    /// position instead of being handed back and re-acquired. The table
    /// keeps its absolute-position indexing and simply names the same
    /// physical block at two indices: the stale one, which nothing
    /// reads, and the live one.
    ///
    /// That aliasing is the reason this is a separate method rather than
    /// a flag on [`Self::reserve`]. A caller that recycles owns the
    /// obligation to release each distinct block exactly once, and to
    /// have established that the donor index really is out of window --
    /// neither of which this type can check for itself.
    pub fn append_block(&mut self, block_id: usize) {
        self.block_table.push(block_id);
    }

    /// Points this sequence at a different set of physical blocks,
    /// keeping its length.
    ///
    /// What a FORK installs. The forked sequence is at the same
    /// position with the same history, and the only thing that differs
    /// is where some of those bytes live: the full blocks are the
    /// original's, shared, and the tail is a copy. The caller owns
    /// establishing that the new table really holds this sequence's
    /// content, which is why this is a plain setter and not a `fork`
    /// method here -- the copying happens a layer up, where the page
    /// GROUP is the unit and one call covers every layer at once.
    ///
    /// The new table must be at least as long as the old, because a
    /// shorter one would drop blocks the caller still accounts for.
    pub fn retable(&mut self, block_table: Vec<usize>) {
        assert!(
            block_table.len() >= self.block_table.len(),
            "a retable must not shorten the block table: {} to {}",
            self.block_table.len(),
            block_table.len()
        );
        self.block_table = block_table;
    }

    /// Releases every block this sequence holds back to `store`. Must be
    /// called explicitly (there's no `Drop` here, since dropping needs a
    /// `&mut PagedKvStore` this type doesn't own a reference to) --
    /// mirrors `KvCache::release_to_pool`, just not automatic.
    ///
    /// Each *distinct* block once: a table that has recycled through
    /// [`Self::append_block`] names one block at more than one index, and
    /// releasing per index would put the same id on the free list twice,
    /// after which two sequences are handed the same memory.
    pub fn release(&mut self, store: &mut PagedKvStore) {
        let mut seen: Vec<usize> = Vec::new();
        for id in self.block_table.drain(..) {
            if !seen.contains(&id) {
                seen.push(id);
                store.release_block(id);
            }
        }
        self.seq_len = 0;
        self.recurrent = None;
    }

    /// How many *additional* blocks appending `n_new` positions would
    /// take from `store`, given what this sequence already holds.
    ///
    /// Counted against held CAPACITY rather than against `seq_len`, so
    /// it is right in both cases. The tail block is usually part-full,
    /// so the answer is never simply `n_new / block_size`: positions
    /// that land in a block already held cost nothing. And a sequence
    /// that pre-reserved (see [`Self::reserve`]) holds blocks beyond
    /// its length, which a `seq_len`-only sum would ask for twice.
    ///
    /// Callers that must not fail part-way through a write check this
    /// against [`PagedKvStore::free_block_count`] before touching
    /// anything.
    pub fn blocks_needed_for(&self, store: &PagedKvStore, n_new: usize) -> usize {
        let held_capacity = self.block_table.len() * store.block_size();
        let unused = held_capacity.saturating_sub(self.seq_len);
        n_new.saturating_sub(unused).div_ceil(store.block_size())
    }

    /// Takes the blocks `n_new` more positions will need, without
    /// advancing `seq_len`.
    ///
    /// This is what makes a multi-layer append all-or-nothing. The
    /// check and the taking happen together, so every later
    /// [`Self::push`] writes into a block this sequence already owns
    /// and cannot fail. Reserving and then not filling is harmless: the
    /// blocks are this sequence's until it releases, and `seq_len`
    /// still says how far it really got.
    pub fn reserve(
        &mut self,
        store: &mut PagedKvStore,
        n_new: usize,
    ) -> Result<(), PagedStoreExhausted> {
        let need = self.blocks_needed_for(store, n_new);
        if need > store.free_block_count() {
            return Err(PagedStoreExhausted);
        }
        for _ in 0..need {
            let id = store
                .acquire_block()
                .expect("checked against free_block_count immediately above");
            self.block_table.push(id);
        }
        Ok(())
    }

    /// Installs a block table the caller allocated, with `seq_len`
    /// positions already computed in it.
    ///
    /// This is how a sequence starts life on top of a cached prefix:
    /// the blocks are somebody else's, already full, and this sequence
    /// appends past them.
    ///
    /// The `seq_len` installed here is therefore also the POSITION the
    /// caller's next forward pass must run at, and the caller has no
    /// second source for that number: [`Self::push`] writes at `seq_len`
    /// and ignores whatever position its caller believes it is at. A
    /// prefill that started from zero over an adopted prefix put the
    /// prompt in the rows *after* the prefix while carrying positions
    /// `0..n`, which is a wrong answer served with a 200.
    ///
    /// `seq_len` MUST be a whole number of blocks,
    /// because the first append writes at `seq_len` and a shared block
    /// must never be written -- another sequence is attending over it.
    /// A ragged length would put that write inside the last shared
    /// block, corrupting a prefix every other holder is reading.
    pub fn adopt_blocks(&mut self, block_table: Vec<usize>, seq_len: usize, block_size: usize) {
        assert_eq!(
            seq_len % block_size,
            0,
            "an adopted prefix must end on a block boundary, or the first \
             append writes into a block another sequence is reading"
        );
        assert!(
            seq_len / block_size <= block_table.len(),
            "block table too short for the adopted length"
        );
        self.block_table = block_table;
        self.seq_len = seq_len;
    }

    /// Copies this sequence's KV out of the shared store into a plain
    /// contiguous [`KvCache`].
    ///
    /// This is what lets the batched prefill path run *unchanged* over
    /// paged storage. Its fast arm hands `cache.k` / `cache.v` to a
    /// blocked kernel that reads them as flat slices, and a block table
    /// cannot be expressed that way. Rather than maintain a second
    /// prefill kernel that reads through the table -- a copy that could
    /// drift from the one every other model path uses -- the pages are
    /// materialised once per layer, the existing kernel runs, and the
    /// new rows go back with [`Self::append_contiguous`].
    ///
    /// The cost is one `seq_len * n_kv_heads * head_dim` copy per layer
    /// per prefill call, against matmuls that dominate prefill. Decode
    /// still reads through the block table and copies nothing, which is
    /// where page sharing actually pays.
    pub fn to_contiguous(&self, store: &PagedKvStore) -> KvCache {
        let mut cache = KvCache::with_capacity_split(
            store.n_kv_heads,
            store.head_dim,
            store.v_head_dim,
            self.seq_len,
        )
        .with_dtype(store.dtype());
        cache.k.reserve_exact(self.seq_len * store.k_width());
        cache.v.reserve_exact(self.seq_len * store.v_width());
        for pos in 0..self.seq_len {
            let block_id = self.block_table[pos / store.block_size];
            let offset = pos % store.block_size;
            cache.k.extend_from_rows(store.k_row(block_id, offset));
            cache.v.extend_from_rows(store.v_row(block_id, offset));
        }
        // The paged store does not evict either, so its positions and
        // its rows agree and this one assignment is both.
        cache.set_positions(self.seq_len);
        // And the copy must not start: the paged prefill scatters rows
        // `seq_len..` of it back into the store after the forward, by
        // absolute offset (#61). See `KvCache::forbid_eviction`.
        cache.forbid_eviction();
        cache.recurrent = self.recurrent.clone();
        cache
    }

    /// Appends `count` positions' worth of contiguous K/V rows, the
    /// inverse of [`Self::to_contiguous`].
    ///
    /// Blocks are reserved for the whole append *before* the first row
    /// is written, so a store that cannot hold the request refuses it
    /// having changed nothing. Writing rows until the store runs dry
    /// would leave the sequence with a `seq_len` that disagrees with
    /// the model's own idea of how far it has got, which is not a
    /// recoverable state.
    pub fn append_contiguous<'a>(
        &mut self,
        store: &mut PagedKvStore,
        k: impl Into<KvRows<'a>>,
        v: impl Into<KvRows<'a>>,
        count: usize,
    ) -> Result<(), PagedStoreExhausted> {
        let (k, v) = (k.into(), v.into());
        let (k_width, v_width) = (store.k_width(), store.v_width());
        assert_eq!(k.len(), count * k_width, "k row count");
        assert_eq!(v.len(), count * v_width, "v row count");
        if self.blocks_needed_for(store, count) > store.free_block_count() {
            return Err(PagedStoreExhausted);
        }
        for i in 0..count {
            self.push_rows(
                store,
                k.slice(i * k_width..(i + 1) * k_width),
                v.slice(i * v_width..(i + 1) * v_width),
            )
            .expect("blocks reserved above, so no push here can exhaust the store");
        }
        Ok(())
    }
}

/// Per-layer [`PagedKvStore`]s that many concurrent requests share.
///
/// # Why a lock per layer, and why two phases
///
/// `frink-server` runs generation on `spawn_blocking` with, in its own
/// words, "no I/O and no shared lock". `KvBlockPool` survives that
/// because it only bounds a *count*: each `KvCache` owns a private
/// `Vec`, and the pool mutex is taken briefly at acquire and release,
/// never during a forward. A `PagedKvStore` is the opposite -- it IS
/// the backing memory -- so sharing one across concurrent requests
/// needs an answer to "who may touch these bytes when".
///
/// The answer the API already implies: attention takes
/// `&PagedKvStore` and only `push` takes `&mut`. So the accesses split
/// cleanly into many concurrent readers and one short exclusive write
/// per position, which is exactly an `RwLock` -- and one per LAYER
/// rather than one for the whole model, so two requests contend only
/// when both are writing the same layer at the same instant.
///
/// A caller must therefore take the write guard for the push alone and
/// drop it before attending under a read guard. Holding the write
/// guard across attention would serialise the expensive half and give
/// back a global lock with extra steps. Nothing breaks in the gap: a
/// sequence's block table and length are its own, and another
/// request's push in between only touches blocks it exclusively holds.
///
/// # Deadlock
///
/// [`Self::write_all`] is the one place several layers are held at
/// once, and it takes them in ascending layer order. Every caller
/// getting the same order is what makes that safe; there is no other
/// multi-layer acquisition in the codebase, and a new one must follow
/// the same rule.
///
/// # Poisoning
///
/// A panic while holding a store leaves the KV mid-write, which is not
/// recoverable state, but it is also not *unsound* -- the bytes are
/// plain `f32`. Poison is stepped over with `into_inner`, matching how
/// `frink-server` already treats its pool mutex: a poisoned lock
/// should not turn one request's panic into a permanently dead server.
pub struct SharedPagedKv {
    layers: Vec<RwLock<PagedKvStore>>,
    /// Guarded separately from the layers, and always taken BEFORE
    /// them, never while a layer guard is held. That one-way order is
    /// what keeps group allocation and the per-layer push paths from
    /// deadlocking against each other.
    groups: Mutex<GroupTable>,
}

impl SharedPagedKv {
    /// One store per layer, each with `blocks_per_layer` blocks.
    pub fn new(
        n_layers: usize,
        block_size: usize,
        blocks_per_layer: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        SharedPagedKv {
            layers: (0..n_layers)
                .map(|_| {
                    RwLock::new(PagedKvStore::new(
                        block_size,
                        blocks_per_layer,
                        n_kv_heads,
                        head_dim,
                    ))
                })
                .collect(),
            groups: Mutex::new(GroupTable::default()),
        }
    }

    /// Wraps stores the caller built, for tests and for callers that
    /// size layers differently.
    pub fn from_stores(stores: Vec<PagedKvStore>) -> Self {
        SharedPagedKv {
            layers: stores.into_iter().map(RwLock::new).collect(),
            groups: Mutex::new(GroupTable::default()),
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Shared access to one layer, for attention.
    pub fn read(&self, layer: usize) -> RwLockReadGuard<'_, PagedKvStore> {
        self.layers[layer]
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Exclusive access to one layer, for a push. Hold it for the push
    /// and nothing else -- see the type docs.
    pub fn write(&self, layer: usize) -> RwLockWriteGuard<'_, PagedKvStore> {
        self.layers[layer]
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Every layer at once, in ascending order, so a multi-layer append
    /// is atomic against other requests.
    ///
    /// This is what makes "all layers advance or none do" hold under
    /// concurrency rather than only single-threaded: checking free
    /// space and then appending are separate steps, and without the
    /// guards spanning both, another request can take the blocks in
    /// between and leave this one half-written.
    ///
    /// Ascending order is the deadlock rule; see the type docs.
    pub fn write_all(&self) -> Vec<RwLockWriteGuard<'_, PagedKvStore>> {
        self.layers
            .iter()
            .map(|l| l.write().unwrap_or_else(|poisoned| poisoned.into_inner()))
            .collect()
    }

    /// Free blocks in one layer, for admission control. A snapshot: by
    /// the time a caller acts on it another request may have taken
    /// them, which is why the append itself re-checks under the guard.
    pub fn free_blocks(&self, layer: usize) -> usize {
        self.read(layer).free_block_count()
    }

    /// Takes one block from EVERY layer as a single group, refcount 1.
    ///
    /// All layers or none: a group that existed in some layers and not
    /// others could not answer "which block holds position p in layer
    /// l", which is the only question it exists to answer.
    pub fn acquire_group(&self) -> Option<PageGroup> {
        let mut guards = self.write_all();
        if guards.iter().any(|s| s.free_block_count() == 0) {
            return None;
        }
        let blocks: Vec<usize> = guards
            .iter_mut()
            .map(|s| {
                s.acquire_block()
                    .expect("checked every layer under these same guards")
            })
            .collect();
        let mut groups = self
            .groups
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Some(PageGroup(groups.insert(blocks)))
    }

    /// One more holder of `group`.
    ///
    /// Called when a second sequence adopts a cached prefix. Without
    /// it, the first sequence to finish frees pages the second is
    /// still attending over -- a use-after-free that shows up as
    /// another conversation's tokens rather than as a crash.
    pub fn retain_group(&self, group: PageGroup) {
        let mut groups = self
            .groups
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        groups.retain(group.0);
    }

    /// One fewer holder. At zero the blocks go back to their layers.
    ///
    /// Returns whether this was the last holder, so a caller can assert
    /// on it rather than guess.
    pub fn release_group(&self, group: PageGroup) -> bool {
        let blocks = {
            let mut groups = self
                .groups
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match groups.release(group.0) {
                Some(blocks) => blocks,
                None => return false,
            }
        };
        // The groups lock is dropped before the layer guards are taken,
        // so the lock order is always groups-then-layers and never the
        // reverse. See the type docs on deadlock.
        let mut guards = self.write_all();
        for (store, block) in guards.iter_mut().zip(blocks) {
            store.release_block(block);
        }
        true
    }

    /// Which block in each layer this group owns, indexed by layer.
    pub fn group_blocks(&self, group: PageGroup) -> Vec<usize> {
        let groups = self
            .groups
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        groups.blocks(group.0).to_vec()
    }

    /// How many holders `group` has. Zero means it does not exist.
    pub fn group_refs(&self, group: PageGroup) -> u32 {
        let groups = self
            .groups
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        groups.refs(group.0)
    }

    /// Copies every layer's block from `src` into `dst`.
    ///
    /// The group-level form of [`PagedKvStore::copy_block`], which is
    /// the unit a sequence forks at: a position's KV lives in one block
    /// per layer, and a fork that copied some layers and shared others
    /// would answer with half of each.
    pub fn copy_group(&self, src: PageGroup, dst: PageGroup) {
        let (from, to) = {
            let groups = self
                .groups
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (groups.blocks(src.0).to_vec(), groups.blocks(dst.0).to_vec())
        };
        // Groups lock released before the layer guards, as everywhere
        // else here: see the type docs on deadlock.
        let mut guards = self.write_all();
        for ((store, &s), &d) in guards.iter_mut().zip(&from).zip(&to) {
            store.copy_block(s, d);
        }
    }

    /// Groups that could still be allocated, bounded by the layer with
    /// the fewest free blocks: a group needs one from each.
    pub fn free_groups(&self) -> usize {
        (0..self.layers.len())
            .map(|l| self.free_blocks(l))
            .min()
            .unwrap_or(0)
    }
}

/// A handle to one block in every layer.
///
/// The unit of sharing between sequences, and the only thing small
/// enough to be what a radix prefix cache stores: that cache maps a
/// token prefix to ONE index per token, while a position's KV lives in
/// `n_layers` different blocks. A group is the name for all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageGroup(pub u32);

/// Group ids, their per-layer blocks, and how many holders each has.
#[derive(Debug, Default)]
struct GroupTable {
    /// Indexed by group id. `None` for an id currently on the free list.
    blocks: Vec<Option<Vec<usize>>>,
    refs: Vec<u32>,
    free_ids: Vec<u32>,
}

impl GroupTable {
    fn insert(&mut self, blocks: Vec<usize>) -> u32 {
        if let Some(id) = self.free_ids.pop() {
            self.blocks[id as usize] = Some(blocks);
            self.refs[id as usize] = 1;
            return id;
        }
        self.blocks.push(Some(blocks));
        self.refs.push(1);
        (self.blocks.len() - 1) as u32
    }

    fn retain(&mut self, id: u32) {
        let refs = &mut self.refs[id as usize];
        assert!(*refs > 0, "cannot retain group {id}, which has no holders");
        *refs += 1;
    }

    /// Drops one holder, returning the blocks to free only when the
    /// last one goes.
    fn release(&mut self, id: u32) -> Option<Vec<usize>> {
        let refs = &mut self.refs[id as usize];
        assert!(*refs > 0, "double free of group {id}");
        *refs -= 1;
        if *refs > 0 {
            return None;
        }
        // The id is reusable now, but only after the blocks are out:
        // handing the id back while it still named blocks would let a
        // later `acquire_group` believe it owns them too.
        let blocks = self.blocks[id as usize]
            .take()
            .expect("a group with holders always has blocks");
        self.free_ids.push(id);
        Some(blocks)
    }

    fn blocks(&self, id: u32) -> &[usize] {
        self.blocks[id as usize]
            .as_deref()
            .expect("group has no blocks; it was already released")
    }

    fn refs(&self, id: u32) -> u32 {
        self.refs.get(id as usize).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests;
