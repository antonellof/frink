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

use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

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
    pub k: Vec<f32>, // [rows, n_kv_heads, head_dim], flattened
    pub v: Vec<f32>, // [rows, n_kv_heads, v_head_dim], flattened
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
            k: Vec::new(),
            v: Vec::new(),
            positions: 0,
            planned_capacity: None,
            pool_state: None,
            window: None,
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
            k: Vec::with_capacity(max_seq_len * n_kv_heads * head_dim),
            v: Vec::with_capacity(max_seq_len * n_kv_heads * v_head_dim),
            positions: 0,
            planned_capacity: Some(max_seq_len),
            pool_state: None,
            window: None,
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
            k: Vec::with_capacity(blocks_needed * block_size * n_kv_heads * head_dim),
            v: Vec::with_capacity(blocks_needed * block_size * n_kv_heads * v_head_dim),
            positions: 0,
            planned_capacity: None,
            pool_state: Some(PooledState {
                pool,
                block_size,
                blocks_held: blocks_needed,
            }),
            window: None,
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

        self.k.extend_from_slice(k_step);
        self.v.extend_from_slice(v_step);
        self.positions += 1;
        Ok(())
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
    pub fn can_truncate_to(&self, new_seq_len: usize) -> bool {
        self.recurrent.is_none() || new_seq_len == 0 || new_seq_len == self.positions
    }

    /// Rolls the cache back to exactly `new_seq_len` positions,
    /// discarding everything after. Used to reject speculatively
    /// decoded draft tokens that turned out wrong: their K/V were
    /// already pushed during batched verification, and rejection means
    /// removing them so the next real decode step continues from the
    /// last *accepted* position, not the last *attempted* one.
    /// Rolls the cache back to exactly `new_seq_len` POSITIONS.
    ///
    /// A windowed cache can only roll back into rows it still holds. A
    /// target further back than [`Self::rows`] names a position this
    /// cache dropped behind its window, and there is no honest thing to
    /// return for it -- so it stops, rather than silently rolling back
    /// to the oldest row it happens to have and answering the next
    /// token out of a history with a hole in it.
    ///
    /// Unreachable for a cache that has not been armed by
    /// [`Self::arm_window`], which is every cache unless
    /// `FRINK_KV_WINDOW` is on: `rows == positions` there, so the
    /// second precondition is implied by the first.
    pub fn truncate(&mut self, new_seq_len: usize) {
        assert!(
            new_seq_len <= self.positions,
            "truncate target {new_seq_len} must not exceed current seq_len {}",
            self.positions
        );
        let rows = self.rows();
        let dropped = self.positions - new_seq_len;
        assert!(
            dropped <= rows,
            "truncate target {new_seq_len} is {dropped} positions back but only {rows} rows \
             are resident: this cache evicted behind a {:?} window (#61). Turn off \
             FRINK_KV_WINDOW for a workload that rolls the KV cache back this far.",
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
        let keep_rows = rows - dropped;
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
    pub fn arm_window(&mut self, window: KvWindow) {
        self.window = Some(window);
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
        self.k.drain(..drop_rows * k_width);
        self.v.drain(..drop_rows * v_width);
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
        (self.k.capacity() + self.v.capacity()) * std::mem::size_of::<f32>()
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
    k: Vec<f32>, // [total_blocks * block_size, n_kv_heads, head_dim], flattened
    v: Vec<f32>, // [total_blocks * block_size, n_kv_heads, v_head_dim], flattened
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
            k: vec![0.0; total_blocks * block_size * n_kv_heads * head_dim],
            v: vec![0.0; total_blocks * block_size * n_kv_heads * v_head_dim],
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
    pub fn k_row(&self, id: usize, offset: usize) -> &[f32] {
        let width = self.k_width();
        let start = id * self.k_elems_per_block() + offset * width;
        &self.k[start..start + width]
    }

    pub fn v_row(&self, id: usize, offset: usize) -> &[f32] {
        let width = self.v_width();
        let start = id * self.v_elems_per_block() + offset * width;
        &self.v[start..start + width]
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

    fn k_row_mut(&mut self, id: usize, offset: usize) -> &mut [f32] {
        let width = self.k_width();
        let start = id * self.k_elems_per_block() + offset * width;
        &mut self.k[start..start + width]
    }

    fn v_row_mut(&mut self, id: usize, offset: usize) -> &mut [f32] {
        let width = self.v_width();
        let start = id * self.v_elems_per_block() + offset * width;
        &mut self.v[start..start + width]
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
        store
            .k_row_mut(block_id, offset_in_block)
            .copy_from_slice(k_step);
        store
            .v_row_mut(block_id, offset_in_block)
            .copy_from_slice(v_step);
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
        );
        cache.k.reserve_exact(self.seq_len * store.k_width());
        cache.v.reserve_exact(self.seq_len * store.v_width());
        for pos in 0..self.seq_len {
            let block_id = self.block_table[pos / store.block_size];
            let offset = pos % store.block_size;
            cache.k.extend_from_slice(store.k_row(block_id, offset));
            cache.v.extend_from_slice(store.v_row(block_id, offset));
        }
        // The paged store does not evict either, so its positions and
        // its rows agree and this one assignment is both.
        cache.set_positions(self.seq_len);
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
    pub fn append_contiguous(
        &mut self,
        store: &mut PagedKvStore,
        k: &[f32],
        v: &[f32],
        count: usize,
    ) -> Result<(), PagedStoreExhausted> {
        let (k_width, v_width) = (store.k_width(), store.v_width());
        assert_eq!(k.len(), count * k_width, "k row count");
        assert_eq!(v.len(), count * v_width, "v row count");
        if self.blocks_needed_for(store, count) > store.free_block_count() {
            return Err(PagedStoreExhausted);
        }
        for i in 0..count {
            self.push(
                store,
                &k[i * k_width..(i + 1) * k_width],
                &v[i * v_width..(i + 1) * v_width],
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
mod tests {
    use super::*;

    /// `blocks_needed_for` is the reservation the whole no-partial-write
    /// guarantee rests on, and it is wrong in two opposite directions
    /// that fail very differently.
    ///
    /// UNDER-counting is the dangerous one: `append_contiguous` reserves
    /// on this answer and then pushes with an `expect`, so too small a
    /// number panics part-way through a layer -- exactly the corrupted
    /// state the reservation exists to prevent. Over-counting merely
    /// refuses a request that would have fitted.
    ///
    /// Both mistakes are one edit away. Flooring instead of ceiling
    /// under-counts whenever the append does not land on a block
    /// boundary; ignoring the part-full tail over-counts whenever a
    /// sequence is mid-block, which after the first token is almost
    /// always. Neither shows up when the numbers happen to divide
    /// evenly, so the cases here are chosen so that they do not.
    #[test]
    fn blocks_needed_for_accounts_for_the_part_full_tail_block() {
        let mut store = PagedKvStore::new(/* block_size = */ 4, 64, 1, 1);
        let mut cache = PagedKvCache::new();
        let row = [1.0f32];
        // Real pushes rather than poking `seq_len`: the count is
        // against blocks this sequence HOLDS, so a length with no
        // blocks behind it is a state that cannot occur and would only
        // let the test agree with an arithmetic nothing produces.
        let advance = |cache: &mut PagedKvCache, store: &mut PagedKvStore, n: usize| {
            for _ in 0..n {
                cache.push(store, &row, &row).unwrap();
            }
        };

        // Empty: a whole-block boundary, and a remainder that a floor
        // would round away.
        assert_eq!(cache.blocks_needed_for(&store, 0), 0);
        assert_eq!(cache.blocks_needed_for(&store, 1), 1);
        assert_eq!(cache.blocks_needed_for(&store, 4), 1);
        assert_eq!(cache.blocks_needed_for(&store, 5), 2, "5 into 4s needs 2");

        // One position in: three slots free in the tail, so appending up
        // to three costs NOTHING. Ignoring the tail would say 1.
        advance(&mut cache, &mut store, 1);
        assert_eq!(cache.blocks_needed_for(&store, 3), 0, "fits in the tail");
        assert_eq!(cache.blocks_needed_for(&store, 4), 1);
        assert_eq!(cache.blocks_needed_for(&store, 8), 2);

        // The awkward case: 1 free in the tail, 6 to append. 5 spill
        // over 4-wide blocks, so 2. A floor gives 1 and a tail-blind
        // ceil gives 2 for the wrong reason, so this pins the shape.
        advance(&mut cache, &mut store, 2); // seq_len = 3
        assert_eq!(cache.blocks_needed_for(&store, 6), 2);
        assert_eq!(cache.blocks_needed_for(&store, 5), 1);

        // Tail exactly full: no free slots, so this behaves like empty.
        advance(&mut cache, &mut store, 1); // seq_len = 4
        assert_eq!(cache.blocks_needed_for(&store, 1), 1);
        assert_eq!(cache.blocks_needed_for(&store, 4), 1);

        // A RESERVED block is capacity this sequence already holds, so
        // it must not be asked for twice. Counting from `seq_len` alone
        // would say 1 here and take a second block for positions the
        // reservation already covers.
        cache.reserve(&mut store, 4).unwrap();
        assert_eq!(
            cache.blocks_needed_for(&store, 4),
            0,
            "a reserved block is already held"
        );
        assert_eq!(cache.blocks_needed_for(&store, 5), 1);
    }

    /// A group takes one block from every layer, and gives them all
    /// back together.
    ///
    /// All-or-nothing is the point: a group holding blocks in some
    /// layers and not others cannot answer "which block holds position
    /// p in layer l", which is the only question it exists for.
    #[test]
    fn a_group_takes_one_block_from_every_layer_and_returns_them_together() {
        let kv = SharedPagedKv::new(3, 2, 4, 1, 1);
        assert_eq!(kv.free_groups(), 4);

        let g = kv.acquire_group().expect("4 groups available");
        let blocks = kv.group_blocks(g);
        assert_eq!(blocks.len(), 3, "one block per layer");
        for l in 0..3 {
            assert_eq!(kv.free_blocks(l), 3, "layer {l} gave up exactly one");
        }
        assert_eq!(kv.free_groups(), 3);

        assert!(kv.release_group(g), "sole holder, so this frees it");
        for l in 0..3 {
            assert_eq!(kv.free_blocks(l), 4, "layer {l} got its block back");
        }
        assert_eq!(kv.free_groups(), 4);
    }

    /// A group survives until its LAST holder releases it.
    ///
    /// This is what makes prefix sharing safe. Two sequences off one
    /// system prompt hold the same pages; if the first to finish freed
    /// them, the second would keep attending over blocks the store had
    /// already handed to somebody else -- surfacing as another
    /// conversation's tokens, not as a crash.
    #[test]
    fn a_group_shared_by_two_holders_survives_the_first_release() {
        let kv = SharedPagedKv::new(2, 2, 2, 1, 1);
        let g = kv.acquire_group().unwrap();
        let blocks = kv.group_blocks(g);
        kv.retain_group(g);
        assert_eq!(kv.group_refs(g), 2);

        assert!(
            !kv.release_group(g),
            "one holder remains, so nothing is freed"
        );
        assert_eq!(kv.group_refs(g), 1);
        assert_eq!(kv.free_blocks(0), 1, "the blocks are still held");
        assert_eq!(kv.group_blocks(g), blocks, "and still name the same blocks");

        assert!(kv.release_group(g), "last holder frees it");
        assert_eq!(kv.group_refs(g), 0);
        assert_eq!(kv.free_blocks(0), 2);
    }

    /// Exhaustion is per group, bounded by the tightest layer.
    ///
    /// A layer with one block left caps the whole pool at one more
    /// group however much room the others have, because a group needs
    /// one block from each.
    #[test]
    fn group_capacity_is_bounded_by_the_layer_with_the_fewest_blocks() {
        let kv = SharedPagedKv::from_stores(vec![
            PagedKvStore::new(2, 5, 1, 1),
            PagedKvStore::new(2, 1, 1, 1),
        ]);
        assert_eq!(kv.free_groups(), 1, "layer 1 has only one block");

        let g = kv.acquire_group().expect("one group fits");
        assert_eq!(kv.free_groups(), 0);
        assert!(
            kv.acquire_group().is_none(),
            "layer 1 is empty, so no group can be formed"
        );
        // The refused attempt must not have taken layer 0's block.
        assert_eq!(kv.free_blocks(0), 4, "a refused group leaks nothing");
        kv.release_group(g);
        assert_eq!(kv.free_blocks(0), 5);
    }

    /// A released id is reused, with a refcount that starts over.
    #[test]
    fn a_released_group_id_is_reused_with_a_fresh_refcount() {
        let kv = SharedPagedKv::new(1, 2, 2, 1, 1);
        let first = kv.acquire_group().unwrap();
        kv.retain_group(first);
        assert_eq!(kv.group_refs(first), 2);
        kv.release_group(first);
        kv.release_group(first);
        assert_eq!(kv.group_refs(first), 0, "gone, not merely decremented");

        let second = kv.acquire_group().unwrap();
        assert_eq!(second, first, "the id is reused");
        assert_eq!(
            kv.group_refs(second),
            1,
            "a reused id must not inherit the old count"
        );
        assert_eq!(kv.group_blocks(second).len(), 1);
        assert_eq!(kv.free_blocks(0), 1);
    }

    /// Reading a group after its last holder released it PANICS rather
    /// than answering with stale blocks.
    ///
    /// This is the observable half of clearing the entry on release,
    /// and the reason it is `take` rather than `clone`: a caller still
    /// holding a `PageGroup` after releasing it is exactly the bug
    /// refcounting exists to prevent, and blocks that now belong to
    /// somebody else are the worst possible answer -- the caller reads
    /// another sequence's KV and nothing says so.
    ///
    /// Written after sabotage showed the previous test here passed with
    /// `clone` in place of `take`: `insert` overwrites the entry on
    /// reuse, so a stale entry was never reachable through the path
    /// that test took. This one reaches it.
    #[test]
    #[should_panic(expected = "already released")]
    fn reading_a_released_group_panics_rather_than_returning_stale_blocks() {
        let kv = SharedPagedKv::new(2, 2, 2, 1, 1);
        let g = kv.acquire_group().unwrap();
        assert!(kv.release_group(g));
        let _ = kv.group_blocks(g);
    }

    /// Releasing a group nobody holds is a bug, not a no-op.
    ///
    /// Silently ignoring it would let a double release return the same
    /// blocks to the store twice, after which two sequences are handed
    /// the same page and both write it.
    #[test]
    #[should_panic(expected = "double free of group")]
    fn releasing_a_group_twice_panics_rather_than_freeing_it_twice() {
        let kv = SharedPagedKv::new(1, 2, 2, 1, 1);
        let g = kv.acquire_group().unwrap();
        assert!(kv.release_group(g));
        kv.release_group(g);
    }

    /// A recycled block backs a later position without the store ever
    /// being asked for another one, and the later position's writes are
    /// what a read at that position returns.
    ///
    /// This is the whole sliding-window mechanism in miniature. Blocks
    /// of two, four positions, and only two blocks in the store: without
    /// recycling, position 2 has nowhere to go.
    #[test]
    fn a_recycled_block_backs_a_later_position_without_touching_the_store() {
        let mut store = PagedKvStore::new(2, 2, 1, 2);
        let mut cache = PagedKvCache::new();
        cache.push(&mut store, &[1.0, 1.0], &[1.0, 1.0]).unwrap();
        cache.push(&mut store, &[2.0, 2.0], &[2.0, 2.0]).unwrap();
        assert_eq!(store.free_block_count(), 1, "one block per position pair");

        // Positions 0..2 have fallen behind a window of two. Their block
        // backs positions 2..4 instead, and the store is untouched.
        let recycled = cache.block_table()[0];
        cache.append_block(recycled);
        assert_eq!(
            store.free_block_count(),
            1,
            "recycling must not take a block from the store"
        );
        cache.push(&mut store, &[3.0, 3.0], &[3.0, 3.0]).unwrap();
        assert_eq!(cache.seq_len(), 3);
        assert_eq!(
            cache.block_table(),
            &[recycled, recycled],
            "the same block at the stale index and the live one"
        );

        // Reading position 2 sees the new row. Position 0's row is gone,
        // which is exactly what "behind the window" means -- the kernel
        // never indexes it.
        let flat = cache.to_contiguous(&store);
        assert_eq!(&flat.k[4..6], &[3.0, 3.0], "position 2 reads its own row");
        assert_eq!(
            &flat.k[0..2],
            &[3.0, 3.0],
            "position 0 now reads the recycled row, and nothing may read it"
        );
    }

    /// Releasing an aliased table hands each block back ONCE.
    ///
    /// Per index instead of per distinct block would put the recycled id
    /// on the free list twice, and the next two acquisitions would hand
    /// two sequences the same memory -- which does not fail, it
    /// interleaves two conversations' KV.
    #[test]
    fn releasing_a_recycled_table_gives_each_block_back_once() {
        // Exactly one block in the store, so "handed back twice" is
        // observable as a second acquisition succeeding.
        let mut store = PagedKvStore::new(2, 1, 1, 2);
        let mut cache = PagedKvCache::new();
        cache.push(&mut store, &[1.0, 1.0], &[1.0, 1.0]).unwrap();
        let held = cache.block_table()[0];
        cache.append_block(held);
        cache.append_block(held);

        let free_before = store.free_block_count();
        cache.release(&mut store);
        assert_eq!(
            store.free_block_count(),
            free_before + 1,
            "three table entries naming one block are one block back"
        );
        // And the store agrees: it can hand out that block once.
        assert!(store.acquire_block().is_some());
        assert!(store.acquire_block().is_none());
    }

    #[test]
    fn a_gathered_sequence_round_trips_through_the_store() {
        let mut store = PagedKvStore::new(2, 8, 2, 2);
        let mut cache = PagedKvCache::new();
        // Five positions over blocks of two: the tail block is half
        // full, which is where an off-by-one in the gather shows up.
        let rows: Vec<[f32; 4]> = (0..5)
            .map(|i| {
                let b = i as f32 * 10.0;
                [b + 1.0, b + 2.0, b + 3.0, b + 4.0]
            })
            .collect();
        for r in &rows {
            cache.push(&mut store, r, r).unwrap();
        }

        let flat = cache.to_contiguous(&store);
        assert_eq!(flat.positions(), 5);
        assert_eq!(flat.k.len(), 5 * 4);
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(&flat.k[i * 4..(i + 1) * 4], r, "position {i} k");
            assert_eq!(&flat.v[i * 4..(i + 1) * 4], r, "position {i} v");
        }

        // And appending those same rows back onto a fresh sequence
        // reproduces the store's view of them exactly.
        let mut rebuilt = PagedKvCache::new();
        let mut store2 = PagedKvStore::new(2, 8, 2, 2);
        rebuilt
            .append_contiguous(&mut store2, &flat.k, &flat.v, 5)
            .unwrap();
        let again = rebuilt.to_contiguous(&store2);
        assert_eq!(again.k, flat.k);
        assert_eq!(again.v, flat.v);
        assert_eq!(again.positions(), flat.positions());
    }

    #[test]
    fn push_grows_seq_len_and_stores_values() {
        let mut cache = KvCache::new(2, 2);
        cache
            .push(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert_eq!(cache.positions(), 1);
        cache
            .push(&[9.0, 10.0, 11.0, 12.0], &[13.0, 14.0, 15.0, 16.0])
            .unwrap();
        assert_eq!(cache.positions(), 2);
        assert_eq!(cache.k.len(), 2 * 2 * 2);
        assert_eq!(cache.k[4], 9.0);
    }

    #[test]
    #[should_panic]
    fn push_wrong_size_panics() {
        let mut cache = KvCache::new(2, 2);
        let _ = cache.push(&[1.0, 2.0], &[1.0, 2.0]); // too short
    }

    #[test]
    fn clear_resets_state() {
        let mut cache = KvCache::new(1, 1);
        cache.push(&[1.0], &[2.0]).unwrap();
        cache.clear();
        assert_eq!(cache.positions(), 0);
        assert!(cache.k.is_empty());
    }

    #[test]
    fn truncate_rolls_back_to_exact_length_preserving_earlier_data() {
        let mut cache = KvCache::new(2, 2);
        cache
            .push(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();
        cache
            .push(&[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
            .unwrap();
        cache
            .push(&[9.0, 9.0, 9.0, 9.0], &[90.0, 90.0, 90.0, 90.0])
            .unwrap();
        assert_eq!(cache.positions(), 3);

        cache.truncate(1);
        assert_eq!(cache.positions(), 1);
        assert_eq!(cache.k, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.v, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn truncate_to_current_length_is_a_no_op() {
        let mut cache = KvCache::new(1, 2);
        cache.push(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.truncate(1);
        assert_eq!(cache.positions(), 1);
        assert_eq!(cache.k, vec![1.0, 2.0]);
    }

    #[test]
    fn truncate_to_zero_empties_the_cache() {
        let mut cache = KvCache::new(1, 2);
        cache.push(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.truncate(0);
        assert_eq!(cache.positions(), 0);
        assert!(cache.k.is_empty());
        assert!(cache.v.is_empty());
    }

    #[test]
    #[should_panic]
    fn truncate_beyond_current_length_panics() {
        let mut cache = KvCache::new(1, 2);
        cache.push(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.truncate(5);
    }

    #[test]
    fn push_after_truncate_continues_correctly() {
        let mut cache = KvCache::new(1, 1);
        cache.push(&[1.0], &[10.0]).unwrap();
        cache.push(&[2.0], &[20.0]).unwrap();
        cache.push(&[3.0], &[30.0]).unwrap(); // this one will be "rejected"
        cache.truncate(2);
        cache.push(&[99.0], &[990.0]).unwrap(); // real continuation after rejection
        assert_eq!(cache.positions(), 3);
        assert_eq!(cache.k, vec![1.0, 2.0, 99.0]);
        assert_eq!(cache.v, vec![10.0, 20.0, 990.0]);
    }

    #[test]
    fn with_capacity_preallocates_and_never_reallocates_within_plan() {
        let n_kv_heads = 4;
        let head_dim = 8;
        let max_seq_len = 16;
        let mut cache = KvCache::with_capacity(n_kv_heads, head_dim, max_seq_len);

        let expected_elems = max_seq_len * n_kv_heads * head_dim;
        assert!(cache.k.capacity() >= expected_elems);
        assert!(cache.v.capacity() >= expected_elems);

        let step = vec![0.5f32; n_kv_heads * head_dim];
        let k_ptr_before = cache.k.as_ptr();
        for _ in 0..max_seq_len {
            cache.push(&step, &step).unwrap();
        }
        let k_ptr_after = cache.k.as_ptr();
        assert_eq!(
            k_ptr_before, k_ptr_after,
            "pushing exactly up to the planned capacity must not reallocate"
        );
        assert!(cache.is_within_planned_capacity());
    }

    #[test]
    fn allocated_bytes_reflects_preallocated_capacity_not_just_used_length() {
        let cache = KvCache::with_capacity(4, 8, 100);
        // 100 positions * 4 kv_heads * 8 head_dim * 2 (k+v) * 4 bytes/f32
        let expected_min = 100 * 4 * 8 * 2 * 4;
        assert!(
            cache.allocated_bytes() >= expected_min,
            "allocated_bytes={} expected_min={expected_min}",
            cache.allocated_bytes()
        );
        // Nothing has been pushed yet, but the memory is already reserved.
        assert_eq!(cache.positions(), 0);
    }

    #[test]
    fn grow_as_you_go_cache_reports_not_within_planned_capacity() {
        let mut cache = KvCache::new(2, 2);
        cache
            .push(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert!(
            !cache.is_within_planned_capacity(),
            "a cache built with `new` has no plan to be within"
        );
    }

    #[test]
    fn with_pool_acquires_one_block_and_reports_it_in_free_blocks() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 10)));
        let cache = KvCache::with_pool(2, 2, pool.clone(), 0).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 9);
        assert_eq!(cache.positions(), 0);
    }

    #[test]
    fn with_pool_fails_without_mutating_the_pool_when_exhausted() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 0)));
        let result = KvCache::with_pool(2, 2, pool.clone(), 0);
        assert!(result.is_err());
        assert_eq!(pool.lock().unwrap().free_blocks(), 0);
    }

    #[test]
    fn push_acquires_additional_blocks_as_the_cache_crosses_block_boundaries() {
        let block_size = 2;
        let pool = Arc::new(Mutex::new(KvBlockPool::new(block_size, 10)));
        let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 9);

        // First block holds `block_size` = 2 positions; pushing them
        // must not need a second block.
        cache.push(&[1.0], &[1.0]).unwrap();
        cache.push(&[2.0], &[2.0]).unwrap();
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            9,
            "filling exactly the first block must not acquire a second one"
        );

        // The third position crosses into a second block.
        cache.push(&[3.0], &[3.0]).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 8);
        assert_eq!(cache.positions(), 3);
        assert_eq!(cache.k, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn push_returns_pool_exhausted_and_leaves_state_unchanged_when_no_blocks_remain() {
        let block_size = 1;
        let pool = Arc::new(Mutex::new(KvBlockPool::new(block_size, 1)));
        let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 0);

        cache.push(&[1.0], &[1.0]).unwrap(); // fills the one held block

        let before_k = cache.k.clone();
        let result = cache.push(&[2.0], &[2.0]);
        assert_eq!(result, Err(KvPoolExhausted));
        assert_eq!(
            cache.positions(),
            1,
            "a failed push must not change seq_len"
        );
        assert_eq!(cache.k, before_k, "a failed push must not append data");
    }

    #[test]
    fn dropping_a_pooled_cache_returns_its_blocks_to_the_pool() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(1, 2)));
        {
            let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
            cache.push(&[1.0], &[1.0]).unwrap(); // fills the first (only held) block
            cache.push(&[2.0], &[2.0]).unwrap(); // crosses into a second block
            assert_eq!(pool.lock().unwrap().free_blocks(), 0);
        }
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            2,
            "both blocks held by the dropped cache must return to the pool"
        );
    }

    #[test]
    fn release_to_pool_is_explicit_and_idempotent() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 5)));
        let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 4);

        cache.release_to_pool();
        assert_eq!(pool.lock().unwrap().free_blocks(), 5);

        cache.release_to_pool(); // no-op, must not over-release
        assert_eq!(pool.lock().unwrap().free_blocks(), 5);

        drop(cache); // must not release again either
        assert_eq!(pool.lock().unwrap().free_blocks(), 5);
    }

    #[test]
    fn two_pooled_caches_share_one_bounded_budget() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(1, 1)));
        let cache_a = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        let cache_b = KvCache::with_pool(1, 1, pool.clone(), 0);
        assert!(
            cache_b.is_err(),
            "a second concurrent request must not be admitted when the shared budget is full"
        );

        drop(cache_a);
        let cache_c = KvCache::with_pool(1, 1, pool, 0);
        assert!(
            cache_c.is_ok(),
            "once the first request's cache is dropped, its budget must become available again"
        );
    }

    #[test]
    fn cloning_a_pooled_cache_detaches_the_clone_from_pool_accounting() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 3)));
        let original = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 2);

        let clone = original.clone();
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            2,
            "cloning must not acquire additional blocks"
        );
        assert_eq!(clone.k, original.k);

        drop(clone);
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            2,
            "dropping a detached clone must not release the original's blocks"
        );

        drop(original);
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            3,
            "dropping the original must release its blocks exactly once"
        );
    }

    /// A resize is arithmetic, and the one rule that is not: shrinking
    /// past what is held is refused, and the pool is left exactly as it
    /// was.
    ///
    /// Clamping to zero instead would silently over-promise -- the
    /// caches holding those blocks do not give them back, so every
    /// later acquire would decide against a budget that does not
    /// describe the memory in use. This test fails under that clamp.
    #[test]
    fn a_pool_refuses_to_shrink_below_what_is_already_held() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 10)));
        let held = KvCache::with_pool(2, 4, Arc::clone(&pool), 24).expect("blocks");
        let in_use = {
            let p = pool.lock().unwrap();
            p.total_blocks() - p.free_blocks()
        };
        assert!(in_use > 0, "the fixture must actually hold blocks");

        let mut p = pool.lock().unwrap();
        assert_eq!(p.resize(in_use - 1), Err(in_use));
        assert_eq!(p.total_blocks(), 10, "a refused resize changes nothing");
        assert_eq!(p.free_blocks(), 10 - in_use);

        // Down to exactly what is held is legal, and leaves nothing free.
        assert_eq!(p.resize(in_use), Ok(()));
        assert_eq!(p.free_blocks(), 0);
        drop(p);
        drop(held);
    }

    /// Growing hands the new blocks to the free list without disturbing
    /// what is held, which is the whole point of a live re-split.
    #[test]
    fn growing_a_pool_adds_to_what_is_free_and_not_to_what_is_held() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 8)));
        let held = KvCache::with_pool(2, 4, Arc::clone(&pool), 16).expect("blocks");
        let mut p = pool.lock().unwrap();
        let in_use = p.total_blocks() - p.free_blocks();

        assert_eq!(p.resize(32), Ok(()));
        assert_eq!(p.total_blocks(), 32);
        assert_eq!(p.free_blocks(), 32 - in_use);
        drop(p);
        drop(held);
    }

    /// Positions and rows agree for every store that does not evict,
    /// and this pins that they are DERIVED separately rather than one
    /// being an alias for the other.
    ///
    /// The point of the split is #61: a windowed layer will drop rows
    /// behind its window while its position count keeps climbing. Until
    /// then the two are equal, so this test cannot prove much about
    /// eviction. What it CAN prove, and what matters, is that `rows`
    /// reads the buffer rather than the counter: set the counter to a
    /// lie and `rows` still reports the truth.
    #[test]
    fn rows_is_read_from_the_buffer_and_positions_from_the_counter() {
        let mut cache = KvCache::new(2, 4);
        for i in 0..5 {
            let step = vec![i as f32; 8];
            cache.push(&step, &step).expect("unbounded growth");
        }
        assert_eq!(cache.positions(), 5);
        assert_eq!(cache.rows(), 5, "nothing evicts, so they agree");

        // The counter lies; the buffer does not.
        cache.force_positions_for_test(99);
        assert_eq!(cache.positions(), 99);
        assert_eq!(
            cache.rows(),
            5,
            "rows must come from k/v, or it is just a second name for the counter"
        );
    }

    /// `truncate` is measured in POSITIONS, and it moves both, because
    /// nothing evicts yet. Written down because it is the first thing
    /// eviction changes: a truncate target will stop being a row index.
    #[test]
    fn truncate_moves_positions_and_rows_together_while_nothing_evicts() {
        let mut cache = KvCache::new(1, 2);
        for i in 0..4 {
            let step = vec![i as f32; 2];
            cache.push(&step, &step).expect("unbounded growth");
        }
        cache.truncate(2);
        assert_eq!(cache.positions(), 2);
        assert_eq!(cache.rows(), 2);
        assert_eq!(cache.k.len(), 4, "two positions of two elements each");
    }

    /// The store's rule must be *the* rule, not a second copy of it.
    ///
    /// `KvWindow::rows_after` is what `frink_models::kv_budget` prices
    /// against. If the store drained to anything else the budget would
    /// be describing a cache that does not exist, which is #33 again.
    #[test]
    fn a_windowed_cache_holds_exactly_what_the_rule_says_it_holds() {
        let window = KvWindow::new(8, 3).expect("positive window");
        let mut cache = KvCache::new(2, 4);
        cache.arm_window(window);
        let step = vec![1.0f32; 8];
        for p in 1..=200usize {
            cache.push(&step, &step).expect("unbounded growth");
            cache.evict_behind_window();
            assert_eq!(cache.positions(), p);
            assert_eq!(
                cache.rows(),
                window.rows_after(p),
                "at {p} positions the store and the rule disagree"
            );
        }
    }

    /// The point of the issue: positions keep counting, rows do not.
    #[test]
    fn a_windowed_layers_resident_rows_stop_growing() {
        let window = KvWindow::with_default_slack(16).expect("positive window");
        let mut cache = KvCache::new(2, 4);
        cache.arm_window(window);
        let step = vec![0.5f32; 8];
        for _ in 0..2000 {
            cache.push(&step, &step).expect("unbounded growth");
            cache.evict_behind_window();
        }
        assert_eq!(cache.positions(), 2000);
        assert!(
            cache.rows() <= window.max_rows(),
            "{} rows resident after 2000 positions",
            cache.rows()
        );
        // And the bytes really went back, not just the length: `drain`
        // frees nothing, and memory is the whole point.
        assert!(
            cache.allocated_bytes() <= window.max_rows() * 8 * 2 * 4 * 2,
            "capacity was never handed back: {} bytes",
            cache.allocated_bytes()
        );
    }

    /// **The correctness argument, measured.**
    ///
    /// Eviction is only allowed to be token-identical because the rows a
    /// windowed kernel reads -- the last `window` of them -- are the
    /// same bytes whether or not anything behind them was dropped. This
    /// pushes distinguishable rows into an evicting cache and a plain
    /// one and compares exactly that slice.
    #[test]
    fn the_rows_a_windowed_kernel_reads_are_identical_with_and_without_eviction() {
        let window = KvWindow::new(6, 2).expect("positive window");
        let (n_kv_heads, head_dim) = (2usize, 4usize);
        let per = n_kv_heads * head_dim;
        let mut evicting = KvCache::new(n_kv_heads, head_dim);
        evicting.arm_window(window);
        let mut plain = KvCache::new(n_kv_heads, head_dim);

        for p in 0..120usize {
            // A row nothing else could produce, so a misplaced row is
            // not merely a different number but an identifiable one.
            let k: Vec<f32> = (0..per).map(|i| (p * 100 + i) as f32).collect();
            let v: Vec<f32> = k.iter().map(|x| -x).collect();
            evicting.push(&k, &v).expect("unbounded growth");
            evicting.evict_behind_window();
            plain.push(&k, &v).expect("unbounded growth");

            let read = window.window().min(p + 1);
            let e_start = (evicting.rows() - read) * per;
            let p_start = (plain.rows() - read) * per;
            assert_eq!(
                &evicting.k[e_start..],
                &plain.k[p_start..],
                "K read set diverged at position {p}"
            );
            assert_eq!(
                &evicting.v[e_start..],
                &plain.v[p_start..],
                "V read set diverged at position {p}"
            );
            assert_eq!(evicting.positions(), plain.positions());
        }
    }

    /// An unarmed cache is the cache this engine has always had.
    #[test]
    fn an_unarmed_cache_never_drops_a_row() {
        let mut cache = KvCache::new(2, 4);
        let step = vec![1.0f32; 8];
        for _ in 0..64 {
            cache.push(&step, &step).expect("unbounded growth");
            assert_eq!(cache.evict_behind_window(), 0);
        }
        assert_eq!(cache.rows(), 64);
        assert_eq!(cache.positions(), 64);
        assert!(cache.window().is_none());
    }

    /// Speculative decoding rolls the cache back by the number of
    /// rejected draft tokens. That is representable while the target is
    /// still resident.
    #[test]
    fn truncate_still_works_inside_the_resident_window() {
        let window = KvWindow::new(8, 3).expect("positive window");
        let mut cache = KvCache::new(1, 2);
        cache.arm_window(window);
        let step = vec![1.0f32; 2];
        for _ in 0..50 {
            cache.push(&step, &step).expect("unbounded growth");
            cache.evict_behind_window();
        }
        let rows_before = cache.rows();
        cache.truncate(46);
        assert_eq!(cache.positions(), 46);
        assert_eq!(cache.rows(), rows_before - 4);
    }

    /// ...and stops, loudly, when it is not. A cache that rolled back to
    /// the oldest row it happened to have would answer the next token
    /// out of a history with a hole in it.
    #[test]
    #[should_panic(expected = "rows are resident")]
    fn truncate_refuses_to_roll_back_past_what_the_window_kept() {
        let window = KvWindow::new(4, 1).expect("positive window");
        let mut cache = KvCache::new(1, 2);
        cache.arm_window(window);
        let step = vec![1.0f32; 2];
        for _ in 0..50 {
            cache.push(&step, &step).expect("unbounded growth");
            cache.evict_behind_window();
        }
        cache.truncate(3);
    }

    /// A clone of an evicting cache is still an evicting cache. Reset
    /// the window on clone and `positions` becomes a row count again in
    /// whatever reads the clone.
    #[test]
    fn a_clone_carries_the_window() {
        let window = KvWindow::new(4, 1).expect("positive window");
        let mut cache = KvCache::new(1, 2);
        cache.arm_window(window);
        let step = vec![1.0f32; 2];
        for _ in 0..20 {
            cache.push(&step, &step).expect("unbounded growth");
            cache.evict_behind_window();
        }
        let copy = cache.clone();
        assert_eq!(copy.window(), Some(window));
        assert_eq!(copy.rows(), cache.rows());
        assert_eq!(copy.positions(), cache.positions());
    }

    /// A pool-backed cache grows when its BUFFER is full, not when its
    /// position counter reaches the buffer's size. Those are the same
    /// number until something evicts, and an evicting pooled cache keyed
    /// on positions would acquire a block per token forever and exhaust
    /// the pool.
    #[test]
    fn a_pooled_windowed_cache_stops_acquiring_blocks() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 8)));
        let mut cache =
            KvCache::with_pool(1, 2, Arc::clone(&pool), 8).expect("the pool was sized for this");
        cache.arm_window(KvWindow::new(4, 1).expect("positive window"));
        let step = vec![1.0f32; 2];
        for _ in 0..64 {
            cache
                .push(&step, &step)
                .expect("the buffer never fills again");
            cache.evict_behind_window();
        }
        assert_eq!(cache.positions(), 64);
        assert!(cache.rows() <= 5);
    }

    /// **The sibling above passes for the wrong reason on its own
    /// sizing, so this is the same property where the trap is armed.**
    ///
    /// Eviction hands surplus CAPACITY back with `shrink_to`, which is
    /// where its memory saving actually comes from. For a pool-backed
    /// cache that is a bug rather than a saving: the pool has already
    /// promised these blocks and does not take them back, while `push`
    /// decides it needs another block by comparing rows against
    /// `k.capacity()`. Shrink that capacity to a window and the
    /// condition is true again every `slack + 1` tokens, forever -- so
    /// the cache draws a fresh block from the shared pool on a cadence,
    /// never releases one, and runs the pool dry underneath every OTHER
    /// request in the process. It surfaces where `push` is documented
    /// infallible, so the caller panics mid-answer.
    ///
    /// The sibling's cache is 8 positions against a 5-row ceiling,
    /// which is under the 2x threshold `shrink_to` is gated on, so it
    /// never reaches the shrink at all. This one is 64 positions
    /// against the same ceiling, and the pool is given spare blocks so
    /// the leak shows up as blocks quietly gone rather than only as the
    /// error at the end of them.
    #[test]
    fn an_evicting_pooled_cache_never_hands_its_capacity_back_to_reacquire_it() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(8, 24)));
        let mut cache =
            KvCache::with_pool(1, 2, Arc::clone(&pool), 64).expect("8 of the 24 blocks");
        let free_after_construction = pool.lock().unwrap().free_blocks();
        assert_eq!(free_after_construction, 16, "8 blocks cover 64 positions");

        cache.arm_window(KvWindow::new(4, 1).expect("positive window"));
        let step = vec![1.0f32; 2];
        for _ in 0..512 {
            cache
                .push(&step, &step)
                .expect("a reservation made up front must not be re-made per token");
            cache.evict_behind_window();
        }

        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            free_after_construction,
            "the cache drew more blocks from the shared pool while evicting"
        );
        assert_eq!(cache.positions(), 512);
        assert!(cache.rows() <= 5);
    }
}
