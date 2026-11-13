//! The CUDA routed experts of a MoE layer, all of them in one step
//! (`frink_cuda::moe_id_launch`): `mul_mv_id` for a decode token and
//! short batches, `mul_mm_id` once an expert gathers enough rows.
//!
//! Without it a CUDA MoE layer ran each routed expert as its own fused
//! FFN launch with a synchronous round trip, `top_k` per token per
//! layer. The fence is the planes: a layer whose experts were not packed
//! contiguously at load (streamed, mixed kinds, F32), or whose packed
//! kinds CUDA has no id kernel for, answers `None` here before touching
//! the device, and the host body runs one matvec per expert as before.
//! A launch that fails after admission is reported once and also falls
//! back.

use super::{Decoder, LayerWeights};

impl Decoder {
    /// The layer's routed planes as CUDA's id kernels take them, or
    /// `None` (no planes, or a kind with no id kernel). The fence every
    /// CUDA routed-expert launch goes through, decode and prefill alike.
    pub(super) fn cuda_moe_planes(
        layer: &LayerWeights,
    ) -> Option<frink_cuda::moe_id_launch::MoePlanesCuda<'_>> {
        layer.moe.packed_q4.as_ref()?.cuda_view()
    }

    /// Every routed slot of `n_tokens` rows of `x` (`ids[t * top_k + j]`
    /// is token `t`'s `j`-th expert) through the layer's SwiGLU experts
    /// on the device. Returns `[n_slots][hidden]`, unweighted, or `None`
    /// to send the layer to the per-expert path.
    pub(super) fn try_cuda_moe_routed(
        layer: &LayerWeights,
        x: &[f32],
        n_tokens: usize,
        ids: &[usize],
        top_k: usize,
    ) -> Option<Vec<f32>> {
        if !frink_core::weight_matrix::cuda_dense_enabled() || ids.is_empty() {
            return None;
        }
        let planes = Self::cuda_moe_planes(layer)?;
        let ids: Vec<u32> = ids.iter().map(|&e| e as u32).collect();
        match frink_cuda::moe_id_launch::launch_moe_swiglu_id(&planes, x, n_tokens, &ids, top_k) {
            Ok(out) => Some(out),
            Err(e) => {
                warn_once(&format!(
                    "frink: CUDA MoE id kernels declined, one matvec per expert runs: {e}"
                ));
                None
            }
        }
    }
}

fn warn_once(msg: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("{msg}"));
}
