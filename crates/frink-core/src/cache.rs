//! A per-layer KV cache, growable one position at a time during decode.
//! Two growth strategies exist:
//!
//! - `with_pool`: PagedAttention-style block allocation. Many caches
//!   (one per concurrent request, typically) draw fixed-size blocks
//!   from one shared, bounded `KvBlockPool` instead of each
//!   independently pre-committing to a worst-case context length.
//!   Growth happens in fixed block-sized quanta, and a cache's blocks
//!   return to the shared pool when it's dropped, so the pool's free
//!   count is a real, live admission-control signal a caller can check
//!   before accepting a new request. This is the block-*allocation*
//!   half of PagedAttention; it does not (yet) change how attention
//!   reads a cache -- `k`/`v` are still read as one contiguous slice
//!   per sequence (see `Decoder::forward_token`/`forward_batch`), just
//!   backed by capacity that grows in block-sized steps instead of
//!   Rust's default exponential `Vec` growth. Wiring this into
//!   `frink-server` as live per-request admission control via
//!   `FRINK_KV_POOL_BLOCKS`/`FRINK_KV_POOL_BLOCK_SIZE`.

use std::sync::{Arc, Mutex};

use crate::kv_store::{KvBuf, KvDtype, KvRows};
use crate::kv_swa::KvWindow;

/// Returned by `KvCache::push` (and `with_pool`) when a pool-backed
/// cache needs another block but its shared `KvBlockPool` has none
/// free. Caches built with `new`/`with_capacity` never return this --
/// their growth is unconditional, matching their pre-paging behavior
/// exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPoolExhausted;

impl std::fmt::Display for KvPoolExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KV cache block pool exhausted: no free blocks remain")
    }
}

impl std::error::Error for KvPoolExhausted {}

/// A bounded pool of fixed-size KV-cache blocks (in positions) shared
/// across many `KvCache` instances, typically one pool per server
/// process. Each `KvCache::with_pool` acquires one block up front and
/// one more each time it grows past its currently held capacity;
/// `free_blocks` is therefore a live, accurate admission-control
/// signal -- a caller can check it before accepting a new request
/// rather than discovering exhaustion only after committing memory.
pub struct KvBlockPool {
    block_size: usize,
    total_blocks: usize,
    free_blocks: usize,
}

impl KvBlockPool {
    /// `block_size` positions per block, `total_blocks` blocks in the
    /// whole shared budget (so `block_size * total_blocks` positions
    /// total, across however many caches draw from this pool at once).
    pub fn new(block_size: usize, total_blocks: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        KvBlockPool {
            block_size,
            total_blocks,
            free_blocks: total_blocks,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub fn free_blocks(&self) -> usize {
        self.free_blocks
    }

    /// Re-budget the pool.
    ///
    /// The pool is an *accounting* budget, not an allocator: each
    /// `KvCache` owns its own buffer and this counts how many blocks
    /// the deployment has promised. So a resize is arithmetic, with one
    /// rule that is not.
    ///
    /// Shrinking below what is currently held is REFUSED and the pool
    /// is left exactly as it was. `free_blocks` would have to go
    /// negative to represent it, and the alternative -- clamping it to
    /// zero -- silently over-promises: the caches already holding those
    /// blocks do not give them back, so every later `try_acquire`
    /// would be deciding against a budget that does not describe the
    /// memory in use.
    ///
    /// Returns the number of blocks currently held when it refuses, so
    /// the caller can say what the floor actually is rather than making
    /// the operator find it by being rejected.
    pub fn resize(&mut self, total_blocks: usize) -> Result<(), usize> {
        let in_use = self.total_blocks - self.free_blocks;
        if total_blocks < in_use {
            return Err(in_use);
        }
        self.free_blocks = total_blocks - in_use;
        self.total_blocks = total_blocks;
        Ok(())
    }

    fn try_acquire(&mut self, n: usize) -> bool {
        if n <= self.free_blocks {
            self.free_blocks -= n;
            true
        } else {
            false
        }
    }

    fn release(&mut self, n: usize) {
        self.free_blocks = (self.free_blocks + n).min(self.total_blocks);
    }
}

struct PooledState {
    pool: Arc<Mutex<KvBlockPool>>,
    block_size: usize,
    blocks_held: usize,
}

pub struct KvCache {
    pub n_kv_heads: usize,
    /// The K head width. Also the Q head width and the width RoPE
    /// rotates within (`n_embd_head_k`).
    pub head_dim: usize,
    /// The V head width (`n_embd_head_v`). Equal to `head_dim` for every
    /// architecture but MiMo-V2 (`head_dim: 192, v_head_dim: 128`);
    /// `frink_models::kv_head_dims` is the seam. Every row of `k` is
    /// `n_kv_heads * head_dim` wide and every row of `v` is
    /// `n_kv_heads * v_head_dim`, so the two buffers are sized and
    /// indexed by their OWN width throughout this file.
    pub v_head_dim: usize,
    /// `[rows, n_kv_heads, head_dim]`, flattened, at the cache's dtype
    /// (`crate::kv_store`): f32 from every constructor, f16 once
    /// [`Self::with_dtype`] says so, which is what a model's caches do.
    pub k: KvBuf,
    /// `[rows, n_kv_heads, v_head_dim]`, flattened, same dtype as `k`.
    pub v: KvBuf,
    /// This sequence's KV on the Metal device, when a layer's attention
    /// runs there.
    ///
    /// It lives HERE and not on the model because it is per-sequence
    /// state: two requests in flight have two histories, and a mirror
    /// hung off a shared `Decoder` would hand one sequence the other's
    /// keys. Travelling with the cache also means the lifecycle is the
    /// cache's.
    ///
    /// It is a MIRROR, not the authority: `k` and `v` above stay
    /// complete, so truncation, the prefix cache, a slot file and every
    /// host reader go on working untouched. It is used only to attend,
    /// and only while its own `seq_len` equals the cache's `rows()`.
    /// Anything that moves a cache backwards -- a truncate, a restored
    /// prefix, a clone -- leaves the two disagreeing and the next use
    /// re-uploads from `k`/`v` rather than trusting it. One upload, in
    /// exchange for not having to find every such place.
    #[cfg(feature = "metal")]
    pub metal_attn: Option<frink_metal::attn::MetalKvBuffers>,
    /// Positions this sequence has consumed.
    ///
    /// **Not the same thing as the number of rows in `k`/`v`**, and the
    /// distinction is the whole reason this field is private. They are
    /// equal today because nothing evicts, and they stop being equal
    /// the moment a windowed layer drops a position behind its window
    /// (#61): `positions` keeps counting, `rows` does not.
    ///
    /// Every reader has to say which one it meant, so there is no
    /// `seq_len` any more. [`Self::positions`] is what RoPE, a resume
    /// point and a truncate target mean; [`Self::rows`] is what
    /// attention iterates and what the bytes cost.
    ///
    /// Two bugs have already been caused by the two being one field.
    /// `PrefillState` read the KV's length as the position to resume at
    /// (#37), and `DraftModelSpeculator::sync` trusted a counter beside
    /// a cache that a device-resident backend leaves empty. Both were
    /// right to read the store rather than keep a copy; both would be
    /// wrong the day the store evicts.
    positions: usize,
    /// The capacity (in positions) this cache was pre-allocated for,
    /// if any. `None` for caches built with `new` or `with_pool`.
    planned_capacity: Option<usize>,
    /// `Some` for caches built with `with_pool`; tracks the shared
    /// pool and how many blocks this cache currently holds, so its
    /// blocks can be returned on drop.
    pool_state: Option<PooledState>,
    /// `Some` once a windowed layer's cache has been told it may drop
    /// rows behind its window (#61). `None` -- the default, and what
    /// every constructor produces -- means this cache keeps every
    /// position it was ever pushed, which is what every store in this
    /// engine did before.
    ///
    /// Armed by the decoder, never by a constructor, because the fact
    /// that a layer is windowed lives in `ModelConfig` and the decision
    /// that eviction is *safe for this run* lives in
    /// `frink_models::decoder::kv_window`. See [`Self::arm_window`].
    window: Option<KvWindow>,
    /// Set on a cache whose holder needs every row it was pushed, so
    /// [`Self::arm_window`] must not arm it whatever the layer's window
    /// is. Three holders today, each of which reads the rows back as a
    /// whole history rather than attending over them:
    ///
    /// - the paged prefill's gathered scratch copy
    ///   ([`PagedKvCache::to_contiguous`]): the forward appends the
    ///   prompt's rows to it and the scatter then copies rows
    ///   `seq_len..` back into the page store by offset. Evicting the
    ///   scratch's front would shift every one of those rows, and the
    ///   scatter would copy the wrong rows or panic on a short buffer;
    /// - a slot save's prefill, whose file holds one row per position;
    /// - `frink layer-divergence`, which compares every row with a
    ///   device copy that does not evict.
    eviction_forbidden: bool,
    /// A recurrent layer's state (`crate::recurrent_state`), `None` for
    /// every attention layer and for a recurrent layer's sequence that
    /// has not run a token yet. Created by the layer's block at the
    /// size its weights name; cloned with the cache; cleared with it;
    /// and the reason [`Self::truncate`] can refuse
    /// ([`Self::can_truncate_to`]). Public because the block that owns
    /// the geometry lives in another crate.
    pub recurrent: Option<crate::recurrent_state::RecurrentState>,
}

/// Cloning a pool-backed cache detaches the clone from pool accounting
/// (its `k`/`v`/`seq_len` data is copied normally, but the clone does
/// not hold or later release any blocks itself) -- mirroring how
/// `frink-models::prefix_cache` already uses `KvCache::clone` to fork
/// a cached prefix into a new, independent request's cache. Only the
/// original cache's blocks are released, exactly once, when it drops.
impl Clone for KvCache {
    fn clone(&self) -> Self {
        KvCache {
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            v_head_dim: self.v_head_dim,
            k: self.k.clone(),
            v: self.v.clone(),
            positions: self.positions,
            planned_capacity: self.planned_capacity,
            pool_state: None,
            // Carried, not reset: a clone of a cache that has already
            // dropped rows is a cache that has already dropped rows,
            // and pretending otherwise would let the clone's
            // `positions` be read as a row count again.
            window: self.window,
            // Reset, not carried: the mark is about one holder reading
            // rows back by offset, and a clone is a new holder. A clone
            // of a cache that has never evicted is a whole history, and
            // a decoder may window it from here like any other.
            eviction_forbidden: false,
            #[cfg(feature = "metal")]
            metal_attn: None,
            recurrent: self.recurrent.clone(),
        }
    }
}

impl Drop for KvCache {
    fn drop(&mut self) {
        if let Some(state) = &self.pool_state {
            if let Ok(mut pool) = state.pool.lock() {
                pool.release(state.blocks_held);
            }
        }
    }
}

impl KvCache {
    /// Positions this sequence has consumed: what RoPE means, what a
    /// resume point means, and what a truncate target is measured in.
    ///
    /// Monotonic except through [`Self::truncate`] and [`Self::clear`].
    /// Equal to [`Self::rows`] today, and deliberately a different
    /// method so it stops being equal safely (#61).
    #[inline]
    pub fn positions(&self) -> usize {
        self.positions
    }

    /// Sets the position counter directly, for a constructor that
    /// filled `k`/`v` by hand rather than through `push`.
    ///
    /// Deliberately narrow and deliberately not `pub`: the only honest
    /// caller is one that has just written exactly this many rows, and
    /// a public setter on a counter the buffer should imply is how the
    /// two drift apart again.
    pub(crate) fn set_positions(&mut self, positions: usize) {
        debug_assert_eq!(
            positions,
            self.rows(),
            "set_positions must agree with the rows just written"
        );
        self.positions = positions;
    }

    /// Test-only: sets the position counter WITHOUT the agreement check
    /// [`Self::set_positions`] makes.
    ///
    /// Exists for one caller: `kv_signature`'s test that a serialized
    /// payload whose declared count contradicts its buffers is
    /// rejected. That contradiction is the thing under test, so it has
    /// to be constructible, and it must not be constructible anywhere
    /// else.
    #[cfg(test)]
    pub(crate) fn force_positions_for_test(&mut self, positions: usize) {
        self.positions = positions;
    }

    /// Rows of K/V actually resident: what attention iterates over and
    /// what the memory costs.
    ///
    /// Derived from the buffer rather than counted alongside it, so it
    /// cannot drift from what is really there. That is the same rule
    /// the batched prefill learned in #37: read the cursor, do not keep
    /// a copy of it.
    #[inline]
    /// The positions this cache was pre-allocated for, when it was.
    ///
    /// A device-side mirror of the KV needs a fixed buffer, so it can
    /// only be made for a cache that already knows how far it can
    /// grow; an unbounded one takes the host path
    /// (`crate::decoder::device_attention`).
    pub fn capacity_positions(&self) -> Option<usize> {
        self.planned_capacity
    }

    pub fn rows(&self) -> usize {
        let k_width = self.k_width();
        if k_width == 0 {
            return 0;
        }
        self.k.len() / k_width
    }

    /// One position's K row: `n_kv_heads * head_dim` elements.
    #[inline]
    pub fn k_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// One position's V row: `n_kv_heads * v_head_dim` elements.
    #[inline]
    pub fn v_width(&self) -> usize {
        self.n_kv_heads * self.v_head_dim
    }

    /// A cache whose K and V heads share one width -- every architecture
    /// but MiMo-V2. [`Self::new_split`] is the general form.
    pub fn new(n_kv_heads: usize, head_dim: usize) -> Self {
        Self::new_split(n_kv_heads, head_dim, head_dim)
    }

    pub fn new_split(n_kv_heads: usize, head_dim: usize, v_head_dim: usize) -> Self {
        KvCache {
            n_kv_heads,
            head_dim,
            v_head_dim,
            k: KvBuf::new(KvDtype::F32),
            v: KvBuf::new(KvDtype::F32),
            positions: 0,
            planned_capacity: None,
            pool_state: None,
            window: None,
            eviction_forbidden: false,
            #[cfg(feature = "metal")]
            metal_attn: None,
            recurrent: None,
        }
    }

    /// Pre-allocates storage for up to `max_seq_len` positions, so
    /// `push` never triggers a reallocation-and-copy during decode.
    /// Use this when the maximum context length is known ahead of time
    pub fn with_capacity(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self::with_capacity_split(n_kv_heads, head_dim, head_dim, max_seq_len)
    }

    /// [`Self::with_capacity`] with a V head width of its own.
    pub fn with_capacity_split(
        n_kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        KvCache {
            n_kv_heads,
            head_dim,
            v_head_dim,
            k: KvBuf::with_capacity(KvDtype::F32, max_seq_len * n_kv_heads * head_dim),
            v: KvBuf::with_capacity(KvDtype::F32, max_seq_len * n_kv_heads * v_head_dim),
            positions: 0,
            planned_capacity: Some(max_seq_len),
            pool_state: None,
            window: None,
            eviction_forbidden: false,
            #[cfg(feature = "metal")]
            metal_attn: None,
            recurrent: None,
        }
    }

    /// Acquires up front however many blocks from `pool` are needed to
    /// cover `max_seq_len` positions (at least one, even if
    /// `max_seq_len` is `0`), so a caller that knows its worst-case
    /// sequence length ahead of time (as `frink-server` does: prompt
    /// length + `max_tokens`) never needs to acquire another block
    /// mid-decode. This matters beyond performance: `push` growing past
    /// its currently held capacity can fail if the pool is exhausted by
    /// *other* requests by then, and callers like
    /// `frink_models::Decoder::forward_token` treat `push` as
    /// infallible for non-pooled caches -- a pooled cache that
    /// under-reserves at construction and then fails to grow later
    /// would violate that assumption and panic mid-decode. Sizing to
    /// `max_seq_len` up front turns that into an admission-control
    /// decision made once, honestly, before any generation work starts,
    /// exactly mirroring `with_capacity`'s worst-case pre-allocation --
    /// just drawn from a shared pool instead of a private allocation.
    /// Returns `Err(KvPoolExhausted)` without mutating anything if the
    /// pool doesn't have that many blocks free.
    pub fn with_pool(
        n_kv_heads: usize,
        head_dim: usize,
        pool: Arc<Mutex<KvBlockPool>>,
        max_seq_len: usize,
    ) -> Result<Self, KvPoolExhausted> {
        Self::with_pool_split(n_kv_heads, head_dim, head_dim, pool, max_seq_len)
    }

    /// [`Self::with_pool`] with a V head width of its own.
    pub fn with_pool_split(
        n_kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        pool: Arc<Mutex<KvBlockPool>>,
        max_seq_len: usize,
    ) -> Result<Self, KvPoolExhausted> {
        let block_size = pool.lock().unwrap().block_size();
        let blocks_needed = max_seq_len.div_ceil(block_size).max(1);
        if !pool.lock().unwrap().try_acquire(blocks_needed) {
            return Err(KvPoolExhausted);
        }
        Ok(KvCache {
            n_kv_heads,
            head_dim,
            v_head_dim,
            k: KvBuf::with_capacity(
                KvDtype::F32,
                blocks_needed * block_size * n_kv_heads * head_dim,
            ),
            v: KvBuf::with_capacity(
                KvDtype::F32,
                blocks_needed * block_size * n_kv_heads * v_head_dim,
            ),
            positions: 0,
            planned_capacity: None,
            pool_state: Some(PooledState {
                pool,
                block_size,
                blocks_held: blocks_needed,
            }),
            window: None,
            eviction_forbidden: false,
            #[cfg(feature = "metal")]
            metal_attn: None,
            recurrent: None,
        })
    }

    /// Appends one position's key/value vectors (each
    /// `n_kv_heads * head_dim` long) to the cache. For pool-backed
    /// caches, this may need to acquire another block first; if the
    /// shared pool has none free, no data is appended and
    /// `Err(KvPoolExhausted)` is returned. Caches built with `new` or
    /// `with_capacity` always return `Ok`.
    pub fn push(&mut self, k_step: &[f32], v_step: &[f32]) -> Result<(), KvPoolExhausted> {
        assert_eq!(k_step.len(), self.k_width());
        assert_eq!(v_step.len(), self.v_width());

        let (k_width, v_width) = (self.k_width(), self.v_width());
        if let Some(state) = &mut self.pool_state {
            let capacity_positions = self.k.capacity() / k_width;
            // ROWS, not positions: this asks whether the buffer is
            // full, and an evicting cache's buffer is shorter than its
            // position count. Equal for a cache that never evicts.
            let rows = self.k.len() / k_width;
            if rows == capacity_positions {
                if !state.pool.lock().unwrap().try_acquire(1) {
                    return Err(KvPoolExhausted);
                }
                state.blocks_held += 1;
                self.k.reserve_exact(state.block_size * k_width);
                self.v.reserve_exact(state.block_size * v_width);
            }
        }

        self.k.extend_from_f32(k_step);
        self.v.extend_from_f32(v_step);
        self.positions += 1;
        Ok(())
    }

    /// The same cache storing its rows at `dtype`.
    ///
    /// Only for a cache that holds nothing yet: converting rows already
    /// written would round them a second time on an f16 round trip,
    /// and no caller has a reason to. Capacity is kept, so a cache
    /// pre-allocated for a context stays pre-allocated.
    pub fn with_dtype(mut self, dtype: KvDtype) -> Self {
        assert!(
            self.k.is_empty() && self.v.is_empty(),
            "with_dtype on a cache that already holds rows"
        );
        if self.k.dtype() != dtype {
            self.k = KvBuf::with_capacity(dtype, self.k.capacity());
            self.v = KvBuf::with_capacity(dtype, self.v.capacity());
        }
        self
    }

    /// What this cache's rows are stored as.
    pub fn dtype(&self) -> KvDtype {
        self.k.dtype()
    }

    /// Every K row, borrowed at the store's dtype: what the attention
    /// kernels read.
    pub fn k_rows(&self) -> KvRows<'_> {
        self.k.view()
    }

    /// Every V row, borrowed at the store's dtype.
    pub fn v_rows(&self) -> KvRows<'_> {
        self.v.view()
    }

    /// Advance length by `n` positions without storing real K/V values
    /// (zero-fill). Used when Metal owns the KV plane and the host cache
    /// only needs matching `seq_len` for sync checks.
    pub fn advance_len(&mut self, n: usize) -> Result<(), KvPoolExhausted> {
        if n == 0 {
            return Ok(());
        }
        let k_zeros = vec![0f32; self.k_width()];
        let v_zeros = vec![0f32; self.v_width()];
        for _ in 0..n {
            self.push(&k_zeros, &v_zeros)?;
        }
        Ok(())
    }

    /// Returns this cache's blocks to its shared pool immediately
    /// (rather than waiting for `Drop`) and detaches it from pool
    /// accounting; a no-op for caches that aren't pool-backed, and
    /// idempotent if called more than once.
    pub fn release_to_pool(&mut self) {
        if let Some(state) = self.pool_state.take() {
            if let Ok(mut pool) = state.pool.lock() {
                pool.release(state.blocks_held);
            }
        }
    }

    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.positions = 0;
        self.recurrent = None;
    }

    /// Whether [`Self::truncate`] to `new_seq_len` is possible.
    ///
    /// Always, for an attention layer's cache: its rows are a history.
    /// For a cache holding a recurrent state only to zero (the state
    /// resets) or to where it already is: the state is a reduction
    /// over the whole prefix and has no "one token ago"
    /// (`crate::recurrent_state`). Every caller that rolls a cache back
    /// to a middle position asks this first, or is fenced off the model.
    ///
    /// For a windowed cache (#61), only while the rows left behind still
    /// cover the window the next query reads
    /// ([`KvWindow::rows_needed_at`]). A cache that has drained to
    /// `window` rows and is rolled back `d` positions would otherwise
    /// hand the next query `d - 1` rows fewer than the model attends
    /// over, with nothing to say so. [`Self::rollback_limit`] is how far
    /// a caller may plan to roll back before it writes.
    pub fn can_truncate_to(&self, new_seq_len: usize) -> bool {
        if new_seq_len > self.positions {
            return false;
        }
        let recurrent_ok =
            self.recurrent.is_none() || new_seq_len == 0 || new_seq_len == self.positions;
        let dropped = self.positions - new_seq_len;
        let rows = self.rows();
        // Only a windowed cache can hold fewer rows than positions. An
        // unwindowed one with no rows at all is a recurrent layer's,
        // whose history is its state and not its rows.
        let window_ok = match self.window {
            None => true,
            Some(w) => {
                dropped <= rows
                    && (new_seq_len == 0 || rows - dropped >= w.rows_needed_at(new_seq_len))
            }
        };
        recurrent_ok && window_ok
    }

    /// How many of the positions written from now on a later
    /// [`Self::truncate`] is guaranteed to be able to undo.
    ///
    /// The question a speculative decoder asks BEFORE it writes a draft
    /// block, because asking [`Self::can_truncate_to`] afterwards is too
    /// late: the rows are written and some of them are already wrong.
    ///
    /// - Unbounded for an ordinary attention cache: its rows are a
    ///   history.
    /// - Zero for one holding a recurrent state, which is a reduction
    ///   over the whole prefix with no "one token ago".
    /// - [`KvWindow::max_rollback`] for a windowed one, because the
    ///   drain may run between the write and the rollback and only the
    ///   window's reserve is kept for it.
    pub fn rollback_limit(&self) -> usize {
        if self.recurrent.is_some() {
            return 0;
        }
        match self.window {
            None => usize::MAX,
            Some(w) => w.max_rollback(),
        }
    }

    /// Rolls the cache back to exactly `new_seq_len` positions,
    /// discarding everything after. Used to reject speculatively
    /// decoded draft tokens that turned out wrong: their K/V were
    /// already pushed during batched verification, and rejection means
    /// removing them so the next real decode step continues from the
    /// last *accepted* position, not the last *attempted* one. The
    /// target is in POSITIONS.
    ///
    /// A windowed cache can only roll back into rows it still holds, and
    /// only as far as leaves the window the next query reads still
    /// resident ([`Self::can_truncate_to`]). Further back names a
    /// position this cache dropped, or leaves a hole the next query
    /// would attend across, and there is no honest thing to return for
    /// either. So it stops, rather than silently answering the next
    /// token out of a shorter history than the model reads.
    ///
    /// Unreachable for a cache that has not been armed by
    /// [`Self::arm_window`]: `rows == positions` there, so every target
    /// at or below `positions` is a row it holds.
    pub fn truncate(&mut self, new_seq_len: usize) {
        assert!(
            new_seq_len <= self.positions,
            "truncate target {new_seq_len} must not exceed current seq_len {}",
            self.positions
        );
        let rows = self.rows();
        let dropped = self.positions - new_seq_len;
        assert!(
            self.recurrent.is_some() || self.can_truncate_to(new_seq_len),
            "truncate target {new_seq_len} is {dropped} positions back but only {rows} rows \
             are resident: this cache evicted behind a {:?} window (#61), and what is left \
             would not cover the window the next query reads. A caller that rolls back asks \
             `rollback_limit` before it writes, or `can_truncate_to` before it truncates; \
             FRINK_KV_WINDOW=0 keeps every row for a workload that needs more.",
            self.window.map(|w| w.window())
        );
        assert!(
            self.can_truncate_to(new_seq_len),
            "truncate target {new_seq_len} on a cache holding a recurrent state at {} positions: \
             a Mamba state is a reduction over the whole prefix and cannot be rolled back to a \
             middle position (`crate::recurrent_state`); the caller should have asked \
             `can_truncate_to` or been fenced off a recurrent model",
            self.positions
        );
        if new_seq_len == 0 {
            self.recurrent = None;
        }
        let keep_rows = rows.saturating_sub(dropped);
        self.k.truncate(keep_rows * self.k_width());
        self.v.truncate(keep_rows * self.v_width());
        self.positions = new_seq_len;
    }

    /// Tells this cache it may drop rows that have fallen behind
    /// `window` (#61 step 2).
    ///
    /// Arming alone drops nothing: [`Self::evict_behind_window`] is what
    /// drops, and the holder calls it at a point where it knows nothing
    /// is mid-read. That split is deliberate. `push` is the obvious
    /// place to evict and it is the wrong one: `Decoder::forward_batch`
    /// writes a whole prefill batch into the cache and only then reads
    /// it back, against a row offset it captured BEFORE the writes.
    /// Evicting inside `push` would move every row out from under that
    /// offset, and the prompt would be attended over shifted keys. So
    /// eviction is something the holder asks for, and asking for it too
    /// rarely only costs memory -- over-retention is always correct,
    /// under-retention is wrong logits.
    ///
    /// Idempotent; a second call with a different window replaces the
    /// first, and rows already dropped stay dropped.
    ///
    /// A no-op on a cache that [`Self::forbid_eviction`] has marked.
    pub fn arm_window(&mut self, window: KvWindow) {
        if !self.eviction_forbidden {
            self.window = Some(window);
        }
    }

    /// Marks this cache as one that must keep every row it is pushed,
    /// whatever window its layer has. See the field it sets for the
    /// callers and why. Disarms a window already armed.
    ///
    /// Panics on a cache that has already evicted: rows already dropped
    /// stay dropped, and a disarmed cache holding fewer rows than
    /// positions would read as an unwindowed one with a hole in it.
    pub fn forbid_eviction(&mut self) {
        assert!(
            self.k_width() == 0 || self.rows() == self.positions,
            "forbid_eviction on a cache that already dropped {} of its {} positions",
            self.positions - self.rows(),
            self.positions
        );
        self.eviction_forbidden = true;
        self.window = None;
    }

    /// The window this cache evicts behind, or `None` if it keeps
    /// everything.
    pub fn window(&self) -> Option<KvWindow> {
        self.window
    }

    /// Drops rows that have fallen behind the armed window, returning
    /// how many rows went. Zero, always, for an unarmed cache.
    ///
    /// The rows dropped are the OLDEST ones, so what remains is still a
    /// contiguous suffix of the sequence: row `i` of `rows` holds
    /// absolute position `positions - rows + i`. Every windowed
    /// attention kernel reads the last `window` rows and nothing else,
    /// and [`KvWindow::rows_after`] guarantees at least that many
    /// survive, so the set of rows a kernel reads is byte-for-byte the
    /// set it would have read with no eviction at all.
    pub fn evict_behind_window(&mut self) -> usize {
        let Some(window) = self.window else {
            return 0;
        };
        let (k_width, v_width) = (self.k_width(), self.v_width());
        if k_width == 0 {
            return 0;
        }
        let rows = self.k.len() / k_width;
        let keep = window.rows_after(self.positions).min(rows);
        let drop_rows = rows - keep;
        if drop_rows == 0 {
            return 0;
        }
        // One `drain` per eviction, not one per token: the whole reason
        // `KvWindow` carries slack. This moves `keep` rows down, and it
        // happens once every `slack + 1` positions.
        self.k.drain_front(drop_rows * k_width);
        self.v.drain_front(drop_rows * v_width);
        // `drain` frees no memory, and the saving this exists for is
        // memory. A cache that just absorbed a 32k-token prefill holds
        // 32k rows of capacity behind `window + slack` rows of data
        // until something hands it back. Only when the excess is large
        // enough to be worth a realloc-and-copy: shrinking on every
        // eviction would trade the block drain for a full copy.
        //
        // **Never for a pool-backed cache**, and the reason is a bug
        // rather than a preference. `push` asks whether the buffer is
        // full by comparing rows against `k.capacity()`, and takes
        // another block from the shared pool when they meet. Shrinking
        // the buffer to a window's worth would make that condition true
        // every `slack + 1` tokens forever, so a windowed pooled cache
        // would draw a fresh block from the pool on a cadence, hold
        // every one of them until it drops, and exhaust the pool
        // mid-answer -- where `push` is documented infallible and the
        // caller panics. The pool has already promised this cache its
        // blocks; handing the capacity back without handing the blocks
        // back saves nothing and costs that.
        if self.pool_state.is_none() {
            let want_k = window.max_rows() * k_width;
            if self.k.capacity() > want_k.saturating_mul(2) {
                self.k.shrink_to(want_k);
                self.v.shrink_to(window.max_rows() * v_width);
            }
        }
        drop_rows
    }

    /// Bytes currently resident for this cache's K and V buffers
    /// combined (actual allocated capacity, not just used length) --
    /// the number that matters for "does this fit in the context
    /// budget,"
    pub fn allocated_bytes(&self) -> usize {
        self.k.allocated_bytes() + self.v.allocated_bytes()
    }

    /// True if this cache was pre-allocated via `with_capacity` and
    /// has not yet grown past that planned capacity (i.e. `push` has
    /// never had to reallocate). Useful for tests/diagnostics
    /// confirming the pre-allocation path actually avoided reallocs.
    pub fn is_within_planned_capacity(&self) -> bool {
        match self.planned_capacity {
            Some(cap) => {
                // POSITIONS against the plan (the plan was made in
                // positions), ROWS against the buffer (the buffer holds
                // rows). Equal unless this cache evicts.
                self.positions <= cap && self.k.capacity() >= self.rows() * self.k_width()
            }
            None => false,
        }
    }
}

mod paged;
pub use paged::*;

#[cfg(test)]
mod tests;
