//! The host-side shape agreement of a stack layer, before any device
//! work, moved out of `prefill/mod.rs` (ADR 0008).

use super::{check_expert_shapes, PrefillDenseLayerCuda, PrefillParams, RouterCuda, StackFfn};
use crate::gpu::CudaError;

/// The host-side shape agreement, before any device work: a mismatch
/// is a named error and never a kernel reading past a buffer.
pub(super) fn check_layer_shapes(
    layer: &PrefillDenseLayerCuda<'_>,
    params: &PrefillParams<'_>,
    batch: usize,
    hidden_dim: usize,
) -> Result<(), CudaError> {
    let PrefillParams {
        n_heads,
        n_kv_heads,
        head_dim,
        prefix_k,
        prefix_v,
        start_pos,
        ..
    } = *params;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    if let StackFfn::Moe {
        router, ffn_dim, ..
    } = layer.ffn
    {
        let fits = match router {
            RouterCuda::F32 { data, rows } => data.len() == rows * hidden_dim,
            RouterCuda::Quant(p) => p.cols() == hidden_dim && p.validate(batch).is_ok(),
        };
        if !fits || router.rows() == 0 || ffn_dim == 0 {
            return Err(CudaError::Unsupported(
                "resident stack: an MoE router that does not fit its layer".to_string(),
            ));
        }
    }
    if layer.attn_norm_w.len() != hidden_dim
        || layer.ffn_norm_w.len() != hidden_dim
        || layer.q.rows() != q_width
        || layer.k.rows() != kv_width
        || layer.v.rows() != kv_width
        || layer.o.rows() != hidden_dim
        || layer.o.cols() != q_width
        || [layer.q, layer.k, layer.v]
            .iter()
            .any(|m| m.cols() != hidden_dim)
        || prefix_k.len() != start_pos * kv_width
        || prefix_v.len() != start_pos * kv_width
    {
        return Err(CudaError::Unsupported(
            "prefill dense layer: projection shapes do not agree with the layer geometry"
                .to_string(),
        ));
    }
    for m in [&layer.q, &layer.k, &layer.v, &layer.o] {
        m.validate(batch)?;
    }
    if let StackFfn::Dense(ex) = &layer.ffn {
        check_expert_shapes(ex, hidden_dim, ex.gate.rows())?;
        for m in [&ex.gate, &ex.up, &ex.down] {
            m.validate(batch)?;
        }
    }
    if batch == 1 && !crate::decode_attn::head_dim_ok(head_dim) {
        return Err(CudaError::Unsupported(format!(
            "resident decode: head_dim {head_dim} (the one-row attention takes a multiple of 4 \
             up to 256)"
        )));
    }
    if let Some(rope) = &layer.rope {
        if rope.rot_dim == 0 || !rope.rot_dim.is_multiple_of(2) || rope.rot_dim > head_dim {
            return Err(CudaError::Unsupported(format!(
                "prefill dense layer: rotary width {} on a head of {head_dim}",
                rope.rot_dim
            )));
        }
        if let Some(ff) = rope.freq_factors {
            if ff.len() != rope.rot_dim / 2 {
                return Err(CudaError::Unsupported(format!(
                    "prefill dense layer: {} freq_factors for {} rotation bands",
                    ff.len(),
                    rope.rot_dim / 2
                )));
            }
        }
    }
    norm_rows(layer.extras.q_norm, q_width)?;
    norm_rows(layer.extras.k_norm, kv_width)?;
    if let Some(sinks) = layer.extras.sinks {
        if sinks.len() != params.n_heads {
            return Err(CudaError::Unsupported(format!(
                "prefill dense layer: {} attention sinks for {} heads",
                sinks.len(),
                params.n_heads
            )));
        }
    }
    for (bias, width) in [
        (layer.extras.q_bias, q_width),
        (layer.extras.k_bias, kv_width),
        (layer.extras.v_bias, kv_width),
        (layer.extras.o_bias, hidden_dim),
    ] {
        if let Some(bias) = bias {
            if bias.len() != width {
                return Err(CudaError::Unsupported(format!(
                    "prefill dense layer: a bias of {} on a projection of {width}",
                    bias.len()
                )));
            }
        }
    }
    Ok(())
}

/// A QK norm weight as `(n, rows per position)`: per head (`head_dim`
/// long) or whole vector, which is how the loader's length rule reads
/// them.
pub(super) fn norm_rows(
    w: Option<&[f32]>,
    width: usize,
) -> Result<Option<(usize, usize)>, CudaError> {
    match w {
        None => Ok(None),
        Some(w) if w.is_empty() || !width.is_multiple_of(w.len()) => {
            Err(CudaError::Unsupported(format!(
                "prefill dense layer: a QK norm of {} over a width of {width}",
                w.len()
            )))
        }
        Some(w) => Ok(Some((w.len(), width / w.len()))),
    }
}
