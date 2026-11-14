//! gpt-oss's routed FFN on the device: the router's bias, the per-expert
//! biases and the clamped SwiGLU (`LLM_FFN_SWIGLU_OAI_MOE` in
//! `llama-graph.cpp::build_moe_ffn`).
//!
//! The decode MoE stack and the prefill stack run the same packed-expert
//! pipeline for every MoE model (router, top-k, `gate ∥ up` id GEMMs, the
//! GLU, `down` id GEMM, weighted sum). gpt-oss changes three steps of it,
//! and this module is those three steps, encoded by both stacks:
//!
//! - the router logits get `ffn_gate_inp.bias` (`vec_add_rows`);
//! - the GLU becomes ggml's `swiglu_oai` with the gate and up expert
//!   biases folded in: `ggml_add_id` on each, then the activation,
//!   one dispatch ([`encode_swiglu_oai_id`]);
//! - `down` gets its expert's bias before the weighted sum
//!   ([`encode_add_id`], `ggml_add_id`).
//!
//! The routing itself needs nothing new: llama's `SOFTMAX_WEIGHT` gating
//! (top-k on the raw logits, softmax over the winners) picks and weighs
//! exactly what a softmax over all experts followed by top-k and
//! renormalization does, which is `moe_topk_softmax_batch` with
//! `renormalize`.

use crate::dispatch::dispatch_counted;
use crate::gpu::{
    ensure_pipeline, resident_f32_buffer, resident_f32_shared, MetalError, ResidentF32Buffer,
};
use crate::moe_ids::IdsBinding;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};
use std::ptr::NonNull;
use std::sync::Arc;

/// One gpt-oss layer's FFN additions, as the stacks take them. The
/// expert planes are expert-major: expert `e`'s gate bias is
/// `gate_bias[e * ffn_rows .. (e + 1) * ffn_rows]`.
///
/// The three expert planes are shared allocations, not slices: they are
/// 368 KB each on gpt-oss-20b, and a slice can only be proven to be the
/// upload the cache holds by comparing every byte, per layer per token
/// (`gpu::resident_f32_shared`). The router bias is 32 floats and stays
/// a slice.
#[derive(Clone, Copy)]
pub struct MoeOaiMetal<'a> {
    /// `ffn_gate_inp.bias`, `[n_experts]`.
    pub router_bias: &'a [f32],
    /// `ffn_gate_exps.bias`, `[n_experts * ffn_rows]`.
    pub gate_bias: &'a Arc<[f32]>,
    /// `ffn_up_exps.bias`, `[n_experts * ffn_rows]`.
    pub up_bias: &'a Arc<[f32]>,
    /// `ffn_down_exps.bias`, `[n_experts * hidden_rows]`.
    pub down_bias: &'a Arc<[f32]>,
    /// The sigmoid's steepness (llama's `alpha = 1.702`).
    pub alpha: f32,
    /// The clamp (llama's `limit = 7.0`).
    pub limit: f32,
}

impl MoeOaiMetal<'_> {
    /// Panics when a plane does not match the layer it is handed with:
    /// a short plane would read the next layer's biases, or past the end.
    pub(crate) fn check(&self, n_experts: usize, ffn_rows: usize, hidden_rows: usize) {
        assert_eq!(self.router_bias.len(), n_experts, "router bias");
        assert_eq!(self.gate_bias.len(), n_experts * ffn_rows, "gate bias");
        assert_eq!(self.up_bias.len(), n_experts * ffn_rows, "up bias");
        assert_eq!(self.down_bias.len(), n_experts * hidden_rows, "down bias");
    }
}

/// [`MoeOaiMetal`] with its planes resident.
pub(crate) struct MoeOaiResident {
    pub(crate) router_bias: Arc<ResidentF32Buffer>,
    gate_bias: Arc<ResidentF32Buffer>,
    up_bias: Arc<ResidentF32Buffer>,
    pub(crate) down_bias: Arc<ResidentF32Buffer>,
    alpha: f32,
    limit: f32,
}

impl MoeOaiResident {
    pub(crate) fn new(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        oai: &MoeOaiMetal<'_>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            router_bias: resident_f32_buffer(device, oai.router_bias)?,
            gate_bias: resident_f32_shared(device, oai.gate_bias)?,
            up_bias: resident_f32_shared(device, oai.up_bias)?,
            down_bias: resident_f32_shared(device, oai.down_bias)?,
            alpha: oai.alpha,
            limit: oai.limit,
        })
    }
}

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ggml `swiglu_oai` (kernel_swiglu_oai) over the routed slots, with
// `ggml_add_id` of the gate and up expert biases folded in front. Slot
// `s` holds rows [s * ffn, (s + 1) * ffn) of `gate` / `up` and ran
// expert `ids[s]`.
template <typename T>
kernel void moe_swiglu_oai_id(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const uint* ids [[buffer(2)]],
    device const float* gate_b [[buffer(3)]],
    device const float* up_b [[buffer(4)]],
    device T* act [[buffer(5)]],
    constant uint& ffn [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    constant float2& al [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    const uint s = i / ffn;
    const uint b = ids[s] * ffn + (i - s * ffn);
    float x0 = gate[i] + gate_b[b];
    float x1 = up[i] + up_b[b];
    x0 = min(x0, al.y);
    x1 = max(min(x1, al.y), -al.y);
    float out_glu = x0 / (1.0f + exp(-x0 * al.x));
    out_glu = out_glu * (1.0f + x1);
    act[i] = (T)out_glu;
}

typedef decltype(moe_swiglu_oai_id<float>) moe_swiglu_oai_id_f32_t;
typedef decltype(moe_swiglu_oai_id<half>) moe_swiglu_oai_id_f16_t;
template [[host_name("moe_swiglu_oai_id_f32")]] kernel moe_swiglu_oai_id_f32_t moe_swiglu_oai_id<float>;
template [[host_name("moe_swiglu_oai_id_f16")]] kernel moe_swiglu_oai_id_f16_t moe_swiglu_oai_id<half>;

// ggml `add_id`: `dst[s][r] += bias[ids[s]][r]`, `rows` per slot.
kernel void moe_add_id(
    device float* dst [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    const uint s = i / rows;
    dst[i] += bias[ids[s] * rows + (i - s * rows)];
}
"#;

const TG: usize = 256;

fn dispatch_1d(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, n: usize) {
    dispatch_counted(
        encoder,
        MTLSize {
            width: n.div_ceil(TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: TG,
            height: 1,
            depth: 1,
        },
    );
}

/// `act = swiglu_oai(gate + gate_bias[id], up + up_bias[id])` over
/// `n_slots` routed slots of `ffn_rows`; `act` is f16 when `f16_out`
/// (the prefill GEMM's input), f32 otherwise (the decode matvec's).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_swiglu_oai_id(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    oai: &MoeOaiResident,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    ids: IdsBinding<'_>,
    act: &ProtocolObject<dyn MTLBuffer>,
    ffn_rows: usize,
    n_slots: usize,
    f16_out: bool,
) -> Result<(), MetalError> {
    let name = if f16_out {
        "moe_swiglu_oai_id_f16"
    } else {
        "moe_swiglu_oai_id_f32"
    };
    let pipe = ensure_pipeline(device, SRC, name)?;
    let n = ffn_rows * n_slots;
    encoder.setComputePipelineState(&pipe.0);
    let mut al = [oai.alpha, oai.limit];
    // SAFETY: the bound buffers outlive the command buffer (the stacks
    // hold them to the wait); `setBytes` copies the locals.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(gate), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(up), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(ids.buf), ids.offset, 2);
        encoder.setBuffer_offset_atIndex(Some(&oai.gate_bias.buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&oai.up_bias.buffer), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(act), 0, 5);
        for (idx, mut v) in [(6usize, ffn_rows as u32), (7, n as u32)] {
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                4,
                idx,
            );
        }
        encoder.setBytes_length_atIndex(NonNull::new(al.as_mut_ptr() as *mut _).unwrap(), 8, 8);
    }
    dispatch_1d(encoder, n);
    Ok(())
}

/// `dst[s] += bias[ids[s]]` over `n_slots` slots of `rows` (ggml
/// `add_id`): `down`'s expert bias.
pub(crate) fn encode_add_id(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    ids: IdsBinding<'_>,
    bias: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    n_slots: usize,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, SRC, "moe_add_id")?;
    let n = rows * n_slots;
    encoder.setComputePipelineState(&pipe.0);
    // SAFETY: as in `encode_swiglu_oai_id`.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(ids.buf), ids.offset, 1);
        encoder.setBuffer_offset_atIndex(Some(bias), 0, 2);
        for (idx, mut v) in [(3usize, rows as u32), (4, n as u32)] {
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                4,
                idx,
            );
        }
    }
    dispatch_1d(encoder, n);
    Ok(())
}

#[cfg(test)]
mod tests;
