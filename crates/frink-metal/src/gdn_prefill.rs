//! A RUN of recurrent prefill layers on the device: every row of a
//! batch through `attn_norm`, the four projections, the convolution,
//! the gates, the chunked delta rule, the gated norm, `ssm_out`, the
//! residual, the FFN and the second residual, with the hidden batch
//! resident from the first layer of the run to the last.
//!
//! # Why
//!
//! Issue #540. A Bonsai-2-27B `pp512` on an M5 Max spent 3.7 s against
//! the reference fork's 0.71, and the GPU part of it was about a
//! second: the rest was a recurrent layer coming back to the host five
//! times (the two projection launches, the host convolution and gates,
//! the chunk launch, `ssm_out`) and its FFN three more, each with the
//! activations downloaded and uploaded again and the convolution and
//! the gate projections run single-threaded or on six cores. This is
//! the decode path's `crate::gdn_branch` with a batch in place of a
//! token: GEMMs where it has matvecs, the chunked recurrence where it
//! has the row step, and the same head kernels stepping a row at a
//! time inside one dispatch.
//!
//! The GEMMs read f32 activations, the input the per-launch path gives
//! them, so the projections round exactly as they did; what moves is
//! where the small ops run, and those are pinned against the host body
//! by `frink_models`' prefill test.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLSize,
};
use std::ptr::NonNull;

use crate::gdn::{buffer_no_copy, encode_gated_norm, DeltaShape};
use crate::gdn_chunk::{encode_delta_chunk, ChunkInputs};
use crate::gdn_head::{
    encode_conv_silu_rows, encode_gates_rows, encode_l2_norm_heads_rows, HeadShape,
};
use crate::gpu::{
    encode_mul_mm_sg, ensure_pipeline, resident_f32_buffer, resident_weight_buffer, shared_metal,
    MetalError, MulMmSgLaunch,
};
use crate::hadamard::FoldPlan;

/// The two gate projections, `[n_out][cols]` f32 weights each, over
/// `rows` activations: one simdgroup per (output, row), f32 throughout,
/// as the host's `matmul_f32` does them. They are 48 outputs wide on
/// Bonsai, far too narrow for a GEMM tile and too small to round to
/// half for one.
const GATE_PROJ_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gdn_gate_proj_rows(
    device const float* x [[buffer(0)]],
    device const float* w_beta [[buffer(1)]],
    device const float* w_alpha [[buffer(2)]],
    device float* beta [[buffer(3)]],
    device float* alpha [[buffer(4)]],
    constant uint2& dims [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint cols = dims.x;
    const uint n_out = dims.y;
    const uint o = tg.x;
    const uint r = tg.y;
    const bool is_alpha = o >= n_out;
    const uint row = is_alpha ? o - n_out : o;
    device const float4* w =
        (device const float4*)((is_alpha ? w_alpha : w_beta) + (size_t)row * cols);
    device const float4* xr = (device const float4*)(x + (size_t)r * cols);
    float acc = 0.0f;
    for (uint i = lane; i < cols / 4; i += 32) {
        acc += dot(w[i], xr[i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        (is_alpha ? alpha : beta)[(size_t)r * n_out + row] = acc;
    }
}
"#;

/// One recurrent layer as the prefill run takes it. Built by the
/// caller once per layer, and checked by [`Self::is_supported`] before
/// anything is encoded.
pub struct PrefillGdnLayer<'a> {
    pub shape: DeltaShape,
    pub head: HeadShape,
    /// `blk.N.attn_norm.weight` and the model's RMS epsilon, which the
    /// l2 norms and the gated norm use too.
    pub attn_norm: &'a [f32],
    pub eps: f32,
    pub qkv: MulMmSgLaunch<'a>,
    pub z: MulMmSgLaunch<'a>,
    /// The rotation `qkv` and `z` share on their input. The gate
    /// projections are stored unfolded and read `attn_norm(x)` before it.
    pub fold_x: Option<FoldPlan<'a>>,
    /// `blk.N.ssm_beta.weight` / `ssm_alpha.weight`, `[n_v_heads][hidden]`.
    pub beta: &'a [f32],
    pub alpha: &'a [f32],
    pub conv1d: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub out_proj: MulMmSgLaunch<'a>,
    pub fold_y: Option<FoldPlan<'a>>,
    pub ffn: PrefillFfn<'a>,
}

/// The dense half of the layer: `h += down(silu(gate(n)) * up(n))`
/// with `n = rms_norm(h + branch)`.
pub struct PrefillFfn<'a> {
    pub norm: &'a [f32],
    pub gate: MulMmSgLaunch<'a>,
    pub up: MulMmSgLaunch<'a>,
    pub down: MulMmSgLaunch<'a>,
    pub fold_x: Option<FoldPlan<'a>>,
    pub fold_act: Option<FoldPlan<'a>>,
}

/// One layer's recurrent state as the host owns it: two page-aligned
/// allocations the GPU reads and writes in place.
pub struct PrefillGdnState {
    pub ssm_ptr: *mut f32,
    pub ssm_bytes: usize,
    pub conv_ptr: *mut f32,
    pub conv_bytes: usize,
    pub conv_len: usize,
}

fn cols_of(m: &MulMmSgLaunch<'_>) -> usize {
    (m.row_bytes / m.block_bytes) * m.block_elems
}

fn fold_ok(plan: Option<&FoldPlan<'_>>, width: usize) -> bool {
    plan.is_none_or(|p| p.check(width).is_ok())
}

impl PrefillGdnLayer<'_> {
    /// Whether every width agrees with every other and with what the
    /// kernels serve, so a run is refused before it encodes anything
    /// rather than halfway through a state it has already advanced.
    pub fn is_supported(&self) -> bool {
        let h = self.head;
        let hidden = self.attn_norm.len();
        let (conv_dim, value_dim) = (h.conv_dim(), h.value_dim());
        let f = &self.ffn;
        h.n_k_heads > 0
            && h.n_v_heads > 0
            && h.head_dim.is_power_of_two()
            && h.head_dim <= crate::gdn_chunk::MAX_HEAD_DIM
            && h.d_conv >= 2
            && h.d_conv <= 8
            && h.n_v_heads.is_multiple_of(h.n_k_heads)
            && self.shape.n_k_heads == h.n_k_heads
            && self.shape.n_v_heads == h.n_v_heads
            && self.shape.head_dim == h.head_dim
            && hidden > 0
            && hidden.is_multiple_of(4)
            && self.qkv.rows == conv_dim
            && cols_of(&self.qkv) == hidden
            && self.z.rows == value_dim
            && cols_of(&self.z) == hidden
            && self.beta.len() == h.n_v_heads * hidden
            && self.alpha.len() == h.n_v_heads * hidden
            && self.conv1d.len() == conv_dim * h.d_conv
            && self.dt_bias.len() == h.n_v_heads
            && self.a.len() == h.n_v_heads
            && self.ssm_norm.len() == h.head_dim
            && self.out_proj.rows == hidden
            && cols_of(&self.out_proj) == value_dim
            && fold_ok(self.fold_x.as_ref(), hidden)
            && fold_ok(self.fold_y.as_ref(), value_dim)
            && f.norm.len() == hidden
            && f.gate.rows == f.up.rows
            && cols_of(&f.gate) == hidden
            && cols_of(&f.up) == hidden
            && f.down.rows == hidden
            && cols_of(&f.down) == f.gate.rows
            && fold_ok(f.fold_x.as_ref(), hidden)
            && fold_ok(f.fold_act.as_ref(), f.gate.rows)
    }
}

/// Buffer indices into the run's scratch: sized for its widest layer
/// and reused by every layer of it, because a serial encoder orders the
/// dispatches and the next layer's writes wait for this one's reads.
mod slot {
    pub const H: usize = 0;
    pub const XN: usize = 1;
    pub const QKV: usize = 2;
    pub const Z: usize = 3;
    pub const BETA_IN: usize = 4;
    pub const ALPHA_IN: usize = 5;
    pub const BETA: usize = 6;
    pub const G: usize = 7;
    pub const CONV: usize = 8;
    pub const O: usize = 9;
    pub const Y: usize = 10;
    pub const OUT: usize = 11;
    pub const GATE: usize = 12;
    pub const UP: usize = 13;
    pub const BRANCH: usize = 14;
}

fn signs_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    plan: Option<&FoldPlan<'_>>,
) -> Result<Option<std::sync::Arc<crate::gpu::ResidentWeightBuffer>>, MetalError> {
    match plan.and_then(|p| p.signs) {
        None => Ok(None),
        Some(signs) => {
            // SAFETY: a `&[f32]` viewed as its own bytes, read only.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    signs.as_ptr() as *const u8,
                    std::mem::size_of_val(signs),
                )
            };
            resident_weight_buffer(device, bytes).map(Some)
        }
    }
}

fn encode_fold(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    buf: &ProtocolObject<dyn MTLBuffer>,
    width: usize,
    rows: usize,
    plan: Option<&FoldPlan<'_>>,
) -> Result<(), MetalError> {
    let Some(plan) = plan else {
        return Ok(());
    };
    let signs = signs_buffer(device, Some(plan))?;
    crate::hadamard::encode_fold_rows(
        encoder,
        device,
        buf,
        width,
        rows,
        plan,
        signs.as_ref().map(|b| &*b.buffer),
    )
}

fn encode_gemm(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    m: &MulMmSgLaunch<'_>,
    x: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
) -> Result<(), MetalError> {
    let w = resident_weight_buffer(device, m.weights)?;
    encode_mul_mm_sg(encoder, device, m, &w, x, out, rows)
}

#[allow(clippy::too_many_arguments)]
fn encode_gate_proj(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    w_beta: &ProtocolObject<dyn MTLBuffer>,
    w_alpha: &ProtocolObject<dyn MTLBuffer>,
    beta: &ProtocolObject<dyn MTLBuffer>,
    alpha: &ProtocolObject<dyn MTLBuffer>,
    cols: usize,
    n_out: usize,
    rows: usize,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, GATE_PROJ_SRC, "gdn_gate_proj_rows")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        for (idx, buf) in [x, w_beta, w_alpha, beta, alpha].into_iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buf), 0, idx);
        }
        let mut dims: [u32; 2] = [cols as u32, n_out as u32];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 8, 5);
    }
    crate::dispatch::dispatch_counted(
        encoder,
        MTLSize {
            width: 2 * n_out,
            height: rows,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// The run's encoder, and the clock that splits it into `pf_*` spans
/// under `FRINK_METAL_KERNEL_TIMING` (serial spans: the steps rely on a
/// serial encoder's ordering). Off, `span` hands back the one encoder.
struct Enc<'c> {
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    spans: crate::kernel_timing::SpanClock,
    cmd: &'c ProtocolObject<dyn MTLCommandBuffer>,
}

impl Enc<'_> {
    fn span(
        &mut self,
        label: &'static str,
    ) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>, MetalError> {
        self.spans.span_serial(self.cmd, &mut self.encoder, label)?;
        Ok(&self.encoder)
    }
}

/// One layer into `encoder`, reading and writing the run's residual
/// stream in `sc.buf(slot::H)`.
///
/// # Safety
///
/// As [`launch_gdn_prefill_run`] for this layer's `state`.
unsafe fn encode_layer(
    enc: &mut Enc<'_>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    w: &PrefillGdnLayer<'_>,
    state: &PrefillGdnState,
    sc: &crate::scratch_pool::Scratch,
    rows: usize,
) -> Result<(), MetalError> {
    let h = w.head;
    let hidden = w.attn_norm.len();
    let (key_dim, value_dim, conv_dim) = (h.key_dim(), h.value_dim(), h.conv_dim());
    if state.conv_len != h.conv_state_len() {
        return Err(MetalError::CommandFailed);
    }
    // SAFETY: the caller's contract, forwarded.
    let ssm_buf = unsafe { buffer_no_copy(device, state.ssm_ptr, state.ssm_bytes) }
        .ok_or(MetalError::BufferAllocFailed)?;
    // SAFETY: the caller's contract, forwarded.
    let conv_state = unsafe { buffer_no_copy(device, state.conv_ptr, state.conv_bytes) }
        .ok_or(MetalError::BufferAllocFailed)?;
    let attn_norm = resident_f32_buffer(device, w.attn_norm)?;
    let w_beta = resident_f32_buffer(device, w.beta)?;
    let w_alpha = resident_f32_buffer(device, w.alpha)?;
    let taps = resident_f32_buffer(device, w.conv1d)?;
    let dt = resident_f32_buffer(device, w.dt_bias)?;
    let a = resident_f32_buffer(device, w.a)?;
    let ssm_norm = resident_f32_buffer(device, w.ssm_norm)?;
    let b = |i: usize| sc.buf(i);
    let (n, eps) = (rows as u32, w.eps);

    let encoder = enc.span("pf_attn_norm")?;
    crate::norm::encode_rms_norm_batch(
        encoder,
        device,
        b(slot::H),
        &attn_norm.buffer,
        b(slot::XN),
        hidden as u32,
        n,
        eps,
    )?;
    let encoder = enc.span("pf_gdn_gate_proj")?;
    // BEFORE the rotation: the gate projections are stored unfolded
    // and read `attn_norm(x)` in the primal basis; the fold below
    // rewrites that buffer in place.
    encode_gate_proj(
        encoder,
        device,
        b(slot::XN),
        &w_beta.buffer,
        &w_alpha.buffer,
        b(slot::BETA_IN),
        b(slot::ALPHA_IN),
        hidden,
        h.n_v_heads,
        rows,
    )?;
    let encoder = enc.span("pf_fold")?;
    encode_fold(
        encoder,
        device,
        b(slot::XN),
        hidden,
        rows,
        w.fold_x.as_ref(),
    )?;
    let encoder = enc.span("pf_gdn_qkv_z")?;
    encode_gemm(encoder, device, &w.qkv, b(slot::XN), b(slot::QKV), rows)?;
    encode_gemm(encoder, device, &w.z, b(slot::XN), b(slot::Z), rows)?;
    let encoder = enc.span("pf_gdn_gates")?;
    encode_gates_rows(
        encoder,
        device,
        h.n_v_heads,
        rows,
        b(slot::BETA_IN),
        b(slot::ALPHA_IN),
        &dt.buffer,
        &a.buffer,
        b(slot::BETA),
        b(slot::G),
    )?;
    let encoder = enc.span("pf_gdn_conv")?;
    encode_conv_silu_rows(
        encoder,
        device,
        h,
        &conv_state,
        &taps.buffer,
        b(slot::QKV),
        b(slot::CONV),
        rows,
    )?;
    let encoder = enc.span("pf_gdn_l2")?;
    // Q and K of every row in one dispatch: the first `2 * n_k_heads`
    // heads of each `[q | k | v]` row the convolution wrote.
    encode_l2_norm_heads_rows(
        encoder,
        device,
        b(slot::CONV),
        0,
        h.head_dim,
        2 * h.n_k_heads,
        rows,
        conv_dim,
        eps,
    )?;
    let encoder = enc.span("pf_gdn_chunk")?;
    encode_delta_chunk(
        encoder,
        device,
        w.shape,
        rows,
        &ssm_buf,
        ChunkInputs {
            q: (b(slot::CONV), 0),
            k: (b(slot::CONV), key_dim),
            v: (b(slot::CONV), 2 * key_dim),
            row_stride: conv_dim,
        },
        b(slot::G),
        b(slot::BETA),
        b(slot::O),
    )?;
    let encoder = enc.span("pf_gdn_gated_norm")?;
    encode_gated_norm(
        encoder,
        device,
        rows * h.n_v_heads,
        h.head_dim,
        eps,
        b(slot::O),
        b(slot::Z),
        &ssm_norm.buffer,
        b(slot::Y),
    )?;
    let encoder = enc.span("pf_fold")?;
    encode_fold(
        encoder,
        device,
        b(slot::Y),
        value_dim,
        rows,
        w.fold_y.as_ref(),
    )?;
    let encoder = enc.span("pf_gdn_out")?;
    encode_gemm(encoder, device, &w.out_proj, b(slot::Y), b(slot::OUT), rows)?;

    encode_ffn_tail(enc, device, &w.ffn, eps, sc, rows, hidden)
}

/// `h += out; h += down(silu(gate(n)) * up(n))` with `n = rms_norm(h)`,
/// reading the branch's projection from `slot::OUT`. The ONE FFN tail:
/// a recurrent layer ends with it, and so does an attention layer's
/// tail ([`PrefillAttnTail`]), which has the same shape after `wo`.
#[allow(clippy::too_many_arguments)]
fn encode_ffn_tail(
    enc: &mut Enc<'_>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    f: &PrefillFfn<'_>,
    eps: f32,
    sc: &crate::scratch_pool::Scratch,
    rows: usize,
    hidden: usize,
) -> Result<(), MetalError> {
    let b = |i: usize| sc.buf(i);
    let ffn_norm = resident_f32_buffer(device, f.norm)?;
    let encoder = enc.span("pf_ffn_norm")?;
    crate::norm::encode_add_rms_norm_batch(
        encoder,
        device,
        b(slot::H),
        b(slot::OUT),
        &ffn_norm.buffer,
        b(slot::XN),
        hidden as u32,
        rows as u32,
        eps,
    )?;
    let encoder = enc.span("pf_fold")?;
    encode_fold(
        encoder,
        device,
        b(slot::XN),
        hidden,
        rows,
        f.fold_x.as_ref(),
    )?;
    let encoder = enc.span("pf_gate_up")?;
    encode_gemm(encoder, device, &f.gate, b(slot::XN), b(slot::GATE), rows)?;
    encode_gemm(encoder, device, &f.up, b(slot::XN), b(slot::UP), rows)?;
    let encoder = enc.span("pf_glu")?;
    // In place: elementwise, each element read before it is written.
    crate::elem::encode_silu_mul(
        encoder,
        device,
        b(slot::GATE),
        b(slot::UP),
        b(slot::GATE),
        (rows * f.gate.rows) as u32,
    )?;
    let encoder = enc.span("pf_fold")?;
    encode_fold(
        encoder,
        device,
        b(slot::GATE),
        f.gate.rows,
        rows,
        f.fold_act.as_ref(),
    )?;
    let encoder = enc.span("pf_down")?;
    encode_gemm(encoder, device, &f.down, b(slot::GATE), b(slot::OUT), rows)?;
    let encoder = enc.span("pf_residual")?;
    crate::elem::encode_vec_add(
        encoder,
        device,
        b(slot::H),
        b(slot::OUT),
        (rows * hidden) as u32,
    )
}

/// An ATTENTION layer's tail at the head of a run: `wo` over the
/// attention branch (gated, before `wo`), then the FFN tail. A hybrid's
/// attention layer feeds the recurrent layers after it a residual
/// stream the host never looks at, so it rides in their command buffer.
pub struct PrefillAttnTail<'a> {
    /// `[rows][o_proj cols]`, the branch as `attn_branch_rows` leaves it.
    pub branch: &'a [f32],
    pub o_proj: MulMmSgLaunch<'a>,
    pub fold: Option<FoldPlan<'a>>,
    pub ffn: PrefillFfn<'a>,
    pub eps: f32,
}

impl PrefillAttnTail<'_> {
    fn is_supported(&self, rows: usize, hidden: usize) -> bool {
        let width = cols_of(&self.o_proj);
        let f = &self.ffn;
        self.branch.len() == rows * width
            && self.o_proj.rows == hidden
            && fold_ok(self.fold.as_ref(), width)
            && f.norm.len() == hidden
            && f.gate.rows == f.up.rows
            && cols_of(&f.gate) == hidden
            && cols_of(&f.up) == hidden
            && f.down.rows == hidden
            && cols_of(&f.down) == f.gate.rows
            && fold_ok(f.fold_x.as_ref(), hidden)
            && fold_ok(f.fold_act.as_ref(), f.gate.rows)
    }
}

/// `rows` tokens of one sequence through `layers`, in order, in ONE
/// command buffer, the hidden batch uploaded once and read back once.
/// Returns the batch as it leaves the last layer.
///
/// Every layer is checked before anything is encoded, so an `Err` has
/// touched no state.
///
/// # Safety
///
/// Each state's two pointers must address that many readable-writable,
/// page-aligned bytes, held exclusively by the caller across the call,
/// which waits for the GPU: the GPU writes those very bytes.
pub unsafe fn launch_gdn_prefill_run(
    hidden_batch: &[f32],
    rows: usize,
    head: Option<&PrefillAttnTail<'_>>,
    layers: &[(PrefillGdnLayer<'_>, PrefillGdnState)],
) -> Result<Vec<f32>, MetalError> {
    let hidden = match (head, layers.first()) {
        (Some(t), _) => t.o_proj.rows,
        (None, Some((first, _))) => first.attn_norm.len(),
        (None, None) => return Err(MetalError::CommandFailed),
    };
    if rows == 0
        || hidden_batch.len() != rows * hidden
        || head.is_some_and(|t| !t.is_supported(rows, hidden))
        || layers
            .iter()
            .any(|(l, _)| !l.is_supported() || l.attn_norm.len() != hidden)
    {
        return Err(MetalError::CommandFailed);
    }
    let widest = |f: fn(&PrefillGdnLayer<'_>) -> usize| {
        layers.iter().map(|(l, _)| f(l)).max().unwrap_or(0).max(1) * rows
    };
    let mut lens = [0usize; 15];
    lens[slot::H] = rows * hidden;
    lens[slot::XN] = rows * hidden;
    lens[slot::QKV] = widest(|l| l.head.conv_dim());
    lens[slot::Z] = widest(|l| l.head.value_dim());
    lens[slot::BETA_IN] = widest(|l| l.head.n_v_heads);
    lens[slot::ALPHA_IN] = lens[slot::BETA_IN];
    lens[slot::BETA] = lens[slot::BETA_IN];
    lens[slot::G] = lens[slot::BETA_IN];
    lens[slot::CONV] = lens[slot::QKV];
    lens[slot::O] = lens[slot::Z];
    lens[slot::Y] = lens[slot::Z];
    lens[slot::BRANCH] = head.map_or(1, |t| t.branch.len());
    lens[slot::OUT] = rows * hidden;
    lens[slot::GATE] = widest(|l| l.ffn.gate.rows).max(head.map_or(0, |t| t.ffn.gate.rows * rows));

    lens[slot::UP] = lens[slot::GATE];

    let shared = shared_metal()?;
    let device = &shared.device;
    let sc =
        crate::scratch_pool::Scratch::take(device, &lens).ok_or(MetalError::BufferAllocFailed)?;
    sc.write(slot::H, hidden_batch)
        .ok_or(MetalError::CommandFailed)?;
    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let mut enc = Enc {
        encoder: cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?,
        spans: crate::kernel_timing::SpanClock::begin(device, 20 * (layers.len() + 1)),
        cmd: &cmd_buf,
    };
    let encoded = (|| {
        if let Some(t) = head {
            let width = cols_of(&t.o_proj);
            let branch = sc
                .write(slot::BRANCH, t.branch)
                .ok_or(MetalError::CommandFailed)?;
            let encoder = enc.span("pf_fold")?;
            encode_fold(encoder, device, branch, width, rows, t.fold.as_ref())?;
            let encoder = enc.span("pf_o_proj")?;
            encode_gemm(encoder, device, &t.o_proj, branch, sc.buf(slot::OUT), rows)?;
            encode_ffn_tail(&mut enc, device, &t.ffn, t.eps, &sc, rows, hidden)?;
        }
        for (layer, state) in layers {
            // SAFETY: the caller's contract, forwarded per layer.
            unsafe { encode_layer(&mut enc, device, layer, state, &sc, rows) }?;
        }
        Ok(())
    })();
    // Ended either way; on an error nothing is committed and the buffer
    // is dropped unsubmitted, so no state has moved.
    enc.encoder.endEncoding();
    encoded?;
    let clock = crate::timing::SubmitClock::start();
    crate::timing::commit_wait_note(&cmd_buf, "gdn-prefill", 16, clock);
    enc.spans.finish(&cmd_buf, "gdn-prefill", 4);
    if cmd_buf.status() != objc2_metal::MTLCommandBufferStatus::Completed {
        return Err(MetalError::CommandFailed);
    }
    // SAFETY: shared storage of `rows * hidden` floats, written by the
    // kernels this call waited for.
    unsafe {
        let h = sc.buf(slot::H);
        Ok(std::slice::from_raw_parts(h.contents().as_ptr() as *const f32, rows * hidden).to_vec())
    }
}
