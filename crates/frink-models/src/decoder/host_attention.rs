//! The host attention kernels the decoder bodies call: the blocked
//! prefill attention and the single-query GQA reduction.

use frink_core::attention::{
    causal_gqa_attention_prefill_shared_kv_split, causal_gqa_attention_softcap,
};
use frink_core::kv_store::KvRows;

#[cfg(feature = "cuda")]
use super::cuda_gqa_enabled;
use super::Decoder;

impl Decoder {
    /// The blocked prefill attention, on the GPU when this shape is one
    /// the Metal flash kernel serves and on the Rayon host kernel
    /// otherwise, which is the ONE place that choice is made.
    ///
    /// It is reached only by layers the FUSED Metal attention block
    /// cannot take -- a gated softmax attention, a V width that differs
    /// from K's, a projection with no Metal launch -- so the layer's
    /// Q, K and V are on the host either way and there is nothing to
    /// fuse; what is left is whether the `n_q x n_kv` score matrix is
    /// built by six cores or by the GPU. On Bonsai's 16 attention
    /// layers that matrix is the largest single host cost of a prefill
    /// (`dot_f32`, `pv_tile` and `qk_tile` are 11178 of a sampled
    /// prefill's top-of-stack against 1504 for the next thing).
    ///
    /// The refusals are the kernel's, not a guess: it takes no sliding
    /// window, no ALiBi slopes, no attention sink and one head width
    /// for K and V, and a shape it does not serve falls through to the
    /// host body below rather than being approximated.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_blocked(
        &self,
        q_batch: &[f32],
        cache_k: &[f32],
        cache_v: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        batch_size: usize,
        base_seq_len: usize,
        softcap: Option<f32>,
        window: Option<usize>,
    ) -> Vec<f32> {
        #[cfg(feature = "metal")]
        if window.is_none()
            && self.alibi_slopes.is_none()
            && v_head_dim == head_dim
            && frink_core::weight_matrix::metal_dense_enabled()
        {
            if let Ok(out) = frink_metal::attn::launch_gqa_prefill_host_ex(
                q_batch,
                cache_k,
                cache_v,
                n_heads,
                n_kv_heads,
                head_dim,
                batch_size,
                base_seq_len,
                softcap,
            ) {
                return out;
            }
        }
        causal_gqa_attention_prefill_shared_kv_split(
            q_batch,
            cache_k,
            cache_v,
            n_heads,
            n_kv_heads,
            head_dim,
            v_head_dim,
            batch_size,
            base_seq_len,
            softcap,
            window,
            self.alibi_slopes.as_deref(),
        )
    }

    /// GQA decode reduction for one token. Uses the CUDA `gqa_decode`
    /// kernel when built with `--features cuda` and `FRINK_CUDA_GQA=1`
    /// (falling back to the host path on any launch error), else the
    /// portable [`causal_gqa_attention`]. With residency enabled the
    /// K/V append stays in [`frink_cuda::attn::CudaKvBuffers`] so only
    /// Q crosses the bus per call (plus a prefix refresh on append).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gqa_attention(
        &self,
        layer: usize,
        q: &[f32],
        k: KvRows<'_>,
        v: KvRows<'_>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        #[cfg(feature = "cuda")]
        {
            // The CUDA kernels read f32 rows; an f16 host store takes the
            // host kernel below rather than being widened per token.
            if let (KvRows::F32(k), KvRows::F32(v)) = (k, v) {
                if cuda_gqa_enabled()
                    && self.config.layer_shapes.is_uniform()
                    && self.alibi_slopes.is_none()
                {
                    match frink_cuda::attn::launch_gqa_decode_resident(
                        layer, q, k, v, n_heads, n_kv_heads, head_dim, seq_len,
                    ) {
                        Ok(out) => return out,
                        Err(e) => {
                            eprintln!(
                                "frink: CUDA GQA resident decode failed, trying full upload: {e}"
                            );
                        }
                    }
                    match frink_cuda::attn::launch_gqa_decode(
                        q, k, v, n_heads, n_kv_heads, head_dim, seq_len,
                    ) {
                        Ok(out) => return out,
                        Err(e) => {
                            eprintln!("frink: CUDA GQA decode failed, host fallback: {e}");
                        }
                    }
                }
            }
        }
        let _ = layer;
        causal_gqa_attention_softcap(
            q,
            k,
            v,
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            self.config.attn_logit_softcap,
        )
    }
}
