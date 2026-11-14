//! The ROUTED half of a decode layer's FFN, for the hybrids whose FFN
//! is a mixture of experts with one shared expert (Qwen3-Next,
//! `qwen35moe`): the router, the softmax top-k, the selected experts
//! through `mul_mv_id`, their weighted sum into the residual, and the
//! sigmoid gate on the shared expert.
//!
//! # Why
//!
//! Issue #540. Before this every MoE layer of a gated delta-net model
//! decoded on the host: the routed experts on the CPU and the shared one
//! beside them, a submission per matrix, while the recurrent and
//! attention layers waited. Qwen3-Next-80B-A3B ran 8.6 tok/s against
//! llama.cpp's 94. With this the whole layer, routed experts included,
//! rides in the decode run (`crate::gdn_branch::GdnRun`) like a dense
//! one, and the token still waits once.
//!
//! The dense FFN tail (`crate::gdn_branch`'s `encode_ffn_tail`) runs
//! the SHARED expert as it runs a dense FFN; this adds what a routed
//! layer has beyond that. llama.cpp's `build_moe_ffn` is the order:
//! softmax over the router logits, the top `k`, renormalised when the
//! model says so, `silu(gate) * up`, `down`, the weighted sum
//! (`qwen3next.cpp:521-539`), and the shared expert scaled by
//! `sigmoid(ffn_gate_inp_shexp . x)`.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};
use std::ptr::NonNull;

use crate::gpu::{
    encode_matvec, ensure_pipeline, resident_f32_buffer, resident_weight_buffer, MatvecLaunch,
    MetalError, MoePackedQ4,
};
use crate::moe_ids::IdsBinding;

/// The routed experts of one layer, beside the shared expert the dense
/// tail already runs.
pub struct RoutedExperts<'a> {
    /// `blk.N.ffn_gate_inp.weight`, `[n_experts, hidden]`.
    pub router: &'a MatvecLaunch<'a>,
    /// The experts' gate, up and down planes.
    pub packed: &'a MoePackedQ4<'a>,
    pub top_k: usize,
    /// `norm_topk_prob`: the selected probabilities renormalised.
    pub renormalize: bool,
    /// `blk.N.ffn_gate_inp_shexp.weight`, `[hidden]`: the shared
    /// expert's output is scaled by `sigmoid(gate . x)`. `None` adds it
    /// unscaled.
    pub shared_gate: Option<&'a [f32]>,
    /// The caller's name for this layer, handed back with the experts
    /// the run chose for it (`GdnRun::finish_with_head`).
    pub tag: usize,
}

/// Where the routed half's buffers sit in the tail scratch, after the
/// dense tail's six.
pub(crate) mod slot {
    pub const LOGITS: usize = 6;
    pub const IDS: usize = 7;
    pub const ROUTE: usize = 8;
    pub const GATE: usize = 9;
    pub const UP: usize = 10;
    pub const ACT: usize = 11;
    pub const EXPERT_OUT: usize = 12;
}

fn cols(m: &MatvecLaunch<'_>) -> usize {
    m.row_bytes / m.block_bytes * m.block_elems
}

impl RoutedExperts<'_> {
    /// The routed half's scratch, in [`slot`] order.
    pub(crate) fn scratch_lens(&self, hidden: usize) -> [usize; 7] {
        let (k, f) = (self.top_k, self.packed.ffn_rows);
        [self.router.rows, k, k, k * f, k * f, k * f, k * hidden]
    }

    /// Whether the widths agree with each other and with `hidden`, and
    /// the planes have `mul_mv_id` kernels.
    pub(crate) fn fits(&self, hidden: usize) -> bool {
        let p = self.packed;
        self.router.rows == p.n_experts
            && cols(self.router) == hidden
            && self.top_k >= 1
            && self.top_k <= p.n_experts
            && p.hidden_rows == hidden
            && p.gate_stride == p.up_stride
            && self.shared_gate.is_none_or(|g| g.len() == hidden)
            && crate::gpu::moe_packed_mul_mv_id_supported(p.gate_kind, p.up_kind, p.down_kind)
    }
}

/// `y_r *= sigmoid(g . x_r)` over rows of `n` floats, one threadgroup a
/// row: the dot is reduced across the group, then every thread scales
/// its share.
const SIGMOID_DOT_SCALE_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_dot_scale(
    device const float* g [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    x += (size_t)row * n;
    y += (size_t)row * n;
    float s = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        s += g[i] * x[i];
    }
    s = simd_sum(s);
    if (lane == 0u) {
        partial[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float t = lane < (tg_size + 31u) / 32u ? partial[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0u) {
            partial[0] = t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / (1.0f + exp(-partial[0]));
    for (uint i = tid; i < n; i += tg_size) {
        y[i] *= scale;
    }
}
"#;

/// The shared expert's output `y` scaled by `sigmoid(gate . x)`, each
/// of `rows` rows by its own.
pub(crate) fn encode_shared_gate(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    gate: &[f32],
    x: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
) -> Result<(), MetalError> {
    let g = resident_f32_buffer(device, gate)?;
    let pipe = ensure_pipeline(device, SIGMOID_DOT_SCALE_SRC, "sigmoid_dot_scale")?;
    encoder.setComputePipelineState(&pipe.0);
    // SAFETY: three buffers that outlive the command buffer (the gate
    // is resident, the other two are the caller's held scratch) and a
    // 4-byte constant that outlives the call.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&g.buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(x), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(y), 0, 2);
        let mut n = gate.len() as u32;
        encoder.setBytes_length_atIndex(NonNull::new(&mut n as *mut u32 as *mut _).unwrap(), 4, 3);
    }
    crate::dispatch::dispatch_counted(
        encoder,
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// `h += Σ_k route[k] · down_k(silu(gate_k(x)) · up_k(x))` over the
/// top `k` of `softmax(router . x)`, into the caller's SERIAL encoder:
/// each dispatch reads what the one before it wrote.
///
/// `x` is the FFN-normed input, `sc` the tail scratch holding the
/// routed buffers at [`slot`].
pub(crate) fn encode_routed(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    r: &RoutedExperts<'_>,
    x: &ProtocolObject<dyn MTLBuffer>,
    h: &ProtocolObject<dyn MTLBuffer>,
    sc: &crate::scratch_pool::Scratch,
) -> Result<(), MetalError> {
    let p = r.packed;
    let k = r.top_k as u32;
    let router_w = resident_weight_buffer(device, r.router.weights)?;
    let ids = IdsBinding::whole(sc.buf(slot::IDS));
    encode_matvec(
        encoder,
        device,
        r.router,
        &router_w,
        x,
        sc.buf(slot::LOGITS),
    )?;
    crate::gpu::encode_moe_topk_softmax_batch(
        encoder,
        device,
        sc.buf(slot::LOGITS),
        IdsBinding::whole(sc.buf(slot::IDS)),
        sc.buf(slot::ROUTE),
        p.n_experts as u32,
        k,
        r.renormalize,
        1,
        None,
    )?;
    crate::gpu::encode_q4_0_moe_gate_up_id(
        encoder,
        device,
        x,
        p,
        IdsBinding::whole(sc.buf(slot::IDS)),
        sc.buf(slot::GATE),
        sc.buf(slot::UP),
        k,
        1,
    )?;
    crate::elem::encode_silu_mul(
        encoder,
        device,
        sc.buf(slot::GATE),
        sc.buf(slot::UP),
        sc.buf(slot::ACT),
        (r.top_k * p.ffn_rows) as u32,
    )?;
    crate::gpu::encode_moe_down_id_planes(
        encoder,
        device,
        p,
        sc.buf(slot::ACT),
        sc.buf(slot::EXPERT_OUT),
        ids,
        k,
    )?;
    crate::gpu::encode_moe_weighted_sum_residual(
        encoder,
        device,
        sc.buf(slot::EXPERT_OUT),
        sc.buf(slot::ROUTE),
        h,
        p.hidden_rows,
        k,
        None,
    )
}

/// The expert ids a completed tail chose, read from its scratch.
pub(crate) fn read_ids(sc: &crate::scratch_pool::Scratch, top_k: usize) -> Vec<usize> {
    // SAFETY: shared storage of at least `top_k` 4-byte ids, written by
    // the top-k kernel of a command buffer the caller has waited for.
    let ids = unsafe {
        std::slice::from_raw_parts(sc.buf(slot::IDS).contents().as_ptr() as *const i32, top_k)
    };
    ids.iter().map(|&i| i.max(0) as usize).collect()
}
