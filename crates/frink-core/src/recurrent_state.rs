//! What a layer with no KV history carries between tokens instead.
//!
//! A Mamba layer (and every other recurrent block llama.cpp keeps in
//! `llama_memory_recurrent`) has no per-position rows to attend over;
//! it has a fixed-size state that the next token reads and overwrites.
//! LFM2's short convolution is the exception that proves the rule: its
//! state IS the last `l_cache - 1` inputs, so `frink_models::shortconv`
//! keeps it as the layer's KV history and needs nothing here. A Mamba
//! state is a reduction over the whole prefix, not a window of it, and
//! that is the one property every consumer of a per-layer cache has to
//! know about:
//!
//! - it CLONES with the cache (a prefix-cache fork is a fork of the
//!   state), and CLEARS with it;
//! - it cannot be TRUNCATED to a middle position. llama.cpp's
//!   `llama_memory_recurrent::seq_rm` refuses a `p0 > 0` too, unless
//!   per-token snapshots were kept (`n_rs_seq`). So [`KvCache::truncate`]
//!   on a cache that holds one refuses anything but "to zero" or "to
//!   where it is", and the callers that roll back -- the prefix cache,
//!   the draft model, the whole-response cache's back-off -- ask
//!   [`KvCache::can_truncate_to`] first or are fenced off the model;
//! - it can be CHECKPOINTED and restored ([`KvCheckpoint`]), which is
//!   how speculative verification undoes a rejected block on such a
//!   target, as llama.cpp's server does: restore, then replay the
//!   committed tokens.
//!
//! The buffers are flat and the LAYER owns their geometry (its weights
//! say what `d_conv`, the conv width and the scan dims are), so this
//! type cannot disagree with the block about a shape: it is created by
//! the block, on first use, at the size the block asks for.
//!
//! [`KvCache::truncate`]: crate::cache::KvCache::truncate
//! [`KvCache::can_truncate_to`]: crate::cache::KvCache::can_truncate_to
//! [`KvCheckpoint`]: crate::cache::KvCheckpoint

/// One sequence's state for one recurrent layer.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentState {
    /// The conv window, `[d_conv - 1][width]`, oldest row first
    /// (`llama_hparams::n_embd_r`).
    ///
    /// Page-aligned for the same reason `ssm` is, and the reason is
    /// measured: a fused recurrent branch that UPLOADED this window and
    /// read it back cost 123 KB of copy per layer per token on Bonsai,
    /// 11.8 MB a token, which was most of why the first version of that
    /// launch ran slower than the host body it replaced.
    pub conv: AlignedF32,
    /// The SSM state, `[n_head][head_dim][d_state]`
    /// (`llama_hparams::n_embd_s`).
    ///
    /// Page-aligned ([`AlignedF32`]) so a Metal kernel can read and
    /// write these very bytes instead of a copy of them; it derefs to
    /// `[f32]`, so a reader sees no difference.
    pub ssm: AlignedF32,
}

impl RecurrentState {
    /// A fresh sequence's state: zeros, as `build_rs` zeroes a new
    /// sequence's (`llama-graph.cpp`, `llm_graph_input_rs`).
    pub fn zeros(conv_len: usize, ssm_len: usize) -> Self {
        Self {
            conv: AlignedF32::zeros(conv_len),
            ssm: AlignedF32::zeros(ssm_len),
        }
    }

    /// Bytes this state holds.
    pub fn bytes(&self) -> usize {
        (self.conv.len() + self.ssm.len()) * std::mem::size_of::<f32>()
    }
}

/// A page-aligned `f32` buffer.
///
/// Apple Silicon's GPU shares the CPU's memory, and Metal will wrap a
/// host allocation as a buffer WITHOUT copying it
/// (`newBufferWithBytesNoCopy`) when the pointer and the length are
/// page-aligned. A recurrent state is the one buffer where that matters:
/// Bonsai-2-27B's is 3.1 MB per layer, so a kernel that reads it by
/// UPLOADING it and writes it back by downloading costs 300 MB of copies
/// per token, which measured slower than leaving the recurrence on the
/// host (`1ebda4fc:docs/plans/gdn-resident-state.md`). Aligned, the same kernel
/// reads and writes the host's own bytes.
///
/// It derefs to `[f32]`, so every existing reader keeps working.
pub struct AlignedF32 {
    ptr: std::ptr::NonNull<f32>,
    len: usize,
    /// The allocation's byte length, which is `len * 4` rounded up to a
    /// page: Metal requires the LENGTH to be page-aligned too, and
    /// `dealloc` must be handed the layout `alloc` got.
    bytes: usize,
}

/// 16 KiB on Apple Silicon, 4 KiB elsewhere; over-aligning is never
/// wrong, so the larger value is used on every target rather than
/// guessed per platform (`crate::weight_matrix` makes the same choice
/// for mmap-backed weights).
pub const PAGE: usize = 16384;

// SAFETY: the allocation is owned exclusively by this value and holds
// plain `f32`, so moving it between threads and sharing `&` are both
// sound. A Metal buffer wrapping it is created and consumed inside one
// call under `&mut`, which is what keeps the GPU's view exclusive.
unsafe impl Send for AlignedF32 {}
unsafe impl Sync for AlignedF32 {}

impl AlignedF32 {
    /// `len` zeroed floats, page-aligned, with the allocation rounded
    /// up to a whole page.
    pub fn zeros(len: usize) -> Self {
        let bytes = (len * std::mem::size_of::<f32>()).max(1).div_ceil(PAGE) * PAGE;
        let layout = std::alloc::Layout::from_size_align(bytes, PAGE).expect("page layout");
        // SAFETY: a non-zero layout; the pointer is checked below and
        // freed in `Drop` with the same layout.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) } as *mut f32;
        let ptr = std::ptr::NonNull::new(raw).expect("allocation failed");
        Self { ptr, len, bytes }
    }

    /// The whole allocation's byte length, page-aligned: what
    /// `newBufferWithBytesNoCopy` must be given.
    pub fn alloc_bytes(&self) -> usize {
        self.bytes
    }

    /// The allocation's base pointer, page-aligned.
    pub fn as_ptr(&self) -> *mut f32 {
        self.ptr.as_ptr()
    }
}

impl Drop for AlignedF32 {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.bytes, PAGE)
            .expect("the layout it was made with");
        // SAFETY: allocated by `zeros` with this exact layout, and this
        // is the only owner.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout) };
    }
}

impl std::ops::Deref for AlignedF32 {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        // SAFETY: `len` floats were allocated and zeroed by `zeros`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl std::ops::DerefMut for AlignedF32 {
    fn deref_mut(&mut self) -> &mut [f32] {
        // SAFETY: as `deref`, with the exclusive borrow this takes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Clone for AlignedF32 {
    fn clone(&self) -> Self {
        let mut copy = Self::zeros(self.len);
        copy.copy_from_slice(self);
        copy
    }
}

impl std::fmt::Debug for AlignedF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedF32({} floats)", self.len)
    }
}

impl PartialEq for AlignedF32 {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl FromIterator<f32> for AlignedF32 {
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        let v: Vec<f32> = iter.into_iter().collect();
        let mut out = Self::zeros(v.len());
        out.copy_from_slice(&v);
        out
    }
}

#[cfg(test)]
mod aligned_tests {
    use super::*;

    #[test]
    fn an_aligned_buffer_is_page_aligned_in_pointer_and_length() {
        for len in [1usize, 1024, 48 * 128 * 128] {
            let b = AlignedF32::zeros(len);
            assert_eq!(b.as_ptr() as usize % PAGE, 0, "pointer");
            assert_eq!(b.alloc_bytes() % PAGE, 0, "length");
            assert!(b.alloc_bytes() >= len * 4);
            assert_eq!(b.len(), len);
            assert!(b.iter().all(|v| *v == 0.0), "zeroed");
        }
    }

    #[test]
    fn it_clones_by_value_and_compares_by_contents() {
        let mut a = AlignedF32::zeros(8);
        a[3] = 1.5;
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(b[3], 1.5);
    }
}
