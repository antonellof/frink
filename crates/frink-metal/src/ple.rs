//! Per-layer embeddings (Gemma-4's "PLE") for the fused dense stacks.
//!
//! Gemma-4 adds two things to an otherwise ordinary Gemma layer
//! (`src/models/gemma4.cpp` at the pinned llama.cpp):
//!
//! - a prologue, once per token, that turns the token's row of
//!   `per_layer_token_embd` and a projection of the scaled embedding into
//!   one `n_embd_per_layer`-wide input per layer ([`PleInputMetal`]);
//! - an epilogue per layer, after the FFN residual, that gates the hidden
//!   state with that layer's input and adds it back
//!   ([`PerLayerGateMetal`]), followed by the layer's scalar output scale.
//!
//! Both are encoded INTO the dense decode and prefill stacks
//! (`crate::decode_dense::launch_decode_dense_stack`,
//! `crate::attn::launch_prefill_dense_stack`), so a Gemma-4 token is one
//! command buffer like every other dense model's, rather than a parallel
//! stack of its own (ADR 0004). This module only holds what those two
//! stacks need and nothing else does: the descriptions, the three small
//! kernels, the batched f32 GEMM the prefill takes for f32 weights, and
//! the plane the per-layer inputs live in.

use crate::dispatch::dispatch_counted;
use crate::gpu::{
    encode_matvec, encode_mul_mm_sg_f16, ensure_pipeline, resident_f32_buffer,
    resident_weight_buffer, MatvecLaunch, MetalError, MulMmSgLaunch,
};
use crate::mem_ranges::MemRanges;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize};
use std::ptr::NonNull;
use std::sync::Mutex;

/// A weight the PREFILL stack multiplies a whole batch by: a quantized
/// GEMM (`mul_mm_sg`, which reads a half activation) or plain f32 rows,
/// which no `mul_mm_sg` kernel takes. Gemma-4 E2B ships its per-layer
/// gate and projection as F32 and `per_layer_model_proj` as BF16 (which
/// the loader widens to f32), so the f32 arm is the common one there.
#[derive(Clone, Copy)]
pub enum PrefillMatMetal<'a> {
    Quant(MulMmSgLaunch<'a>),
    F32 {
        weights: &'a [f32],
        rows: usize,
        cols: usize,
    },
}

impl PrefillMatMetal<'_> {
    pub fn rows(&self) -> usize {
        match self {
            Self::Quant(l) => l.rows,
            Self::F32 { rows, .. } => *rows,
        }
    }

    /// Whether the GEMM reads a half activation.
    fn wants_half(&self) -> bool {
        matches!(self, Self::Quant(_))
    }
}

/// The once-per-token prologue: `n_layer` inputs of `n_embd` each.
///
/// ```text
/// pl  = model_proj · h            (h already scaled by sqrt(hidden))
/// pl *= proj_scale                (1 / sqrt(hidden))
/// pl  = rms_norm(pl) * proj_norm  (per n_embd chunk)
/// pl  = (pl + tok_rows) * input_scale   (1 / sqrt(2))
/// ```
///
/// `tok_rows` is the token's row of `per_layer_token_embd`, already
/// multiplied by `sqrt(n_embd)`, `batch` rows of `n_layer * n_embd`. It is
/// gathered on the host, as llama.cpp does ("do the PLE first to
/// guarantee it is done in the host buffer"): the table is 262k rows of
/// 8960 and only one row per token is read.
///
/// The two scales are passed rather than recomputed here so that the
/// host body and this one take them from the same expression.
pub struct PleInputMetal<'a, L> {
    pub model_proj: L,
    pub proj_norm_w: &'a [f32],
    pub tok_rows: &'a [f32],
    pub n_embd: usize,
    pub n_layer: usize,
    pub proj_scale: f32,
    pub input_scale: f32,
}

/// The per-layer epilogue, applied after the FFN residual:
///
/// ```text
/// g  = gelu(gate · h) * pl[slot]
/// h += rms_norm(proj · g) * post_norm
/// ```
pub struct PerLayerGateMetal<'a, L> {
    pub gate: L,
    pub proj: L,
    pub post_norm: &'a [f32],
    /// Which `n_embd` chunk of the prologue's plane this layer reads.
    pub slot: usize,
}

const PLE_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// y *= a
kernel void ple_scale_f32(
    device float* y [[buffer(0)]],
    constant float& a [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        y[i] = y[i] * a;
    }
}

// y = (y + x) * a, the order the host body adds and scales in.
kernel void ple_add_scale_f32(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant float& a [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        y[i] = (y[i] + x[i]) * a;
    }
}

// out[r, c] = gelu(gate[r, c]) * up[r * up_stride + up_off + c], for
// `total / n` rows of `n`. The gelu is `gelu_mul_f32`'s, character for
// character (tanh approximation, `frink_core::matmul::gelu`).
template <typename T>
kernel void ple_gelu_mul_rows(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& total [[buffer(4)]],
    constant uint& up_stride [[buffer(5)]],
    constant uint& up_off [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < total) {
        uint r = i / n;
        uint c = i - r * n;
        float g = gate[i];
        const float K = 0.7978845608028654f; // sqrt(2/pi)
        const float C = 0.044715f;
        float gelu = 0.5f * g * (1.0f + precise::tanh(K * (g + C * g * g * g)));
        out[i] = T(gelu * up[r * up_stride + up_off + c]);
    }
}

typedef decltype(ple_gelu_mul_rows<float>) ple_gelu_mul_rows_t;
template [[host_name("ple_gelu_mul_rows_f32")]] kernel ple_gelu_mul_rows_t ple_gelu_mul_rows<float>;
typedef decltype(ple_gelu_mul_rows<half>) ple_gelu_mul_rows_h_t;
template [[host_name("ple_gelu_mul_rows_f16")]] kernel ple_gelu_mul_rows_h_t ple_gelu_mul_rows<half>;

// out[t, r] = dot(w[r, :], x[t, :]) for f32 weights and activations.
// Four simdgroups per threadgroup, one output row each; the token index
// is the FASTER grid axis, so threadgroups in flight share a weight row
// and the (much smaller) activation plane stays in cache while the
// weights stream once.
kernel void ple_f32_mm(
    device const float* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& n_tokens [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    const uint t = tgpig.x;
    const uint r = tgpig.y * 4u + sg;
    if (t >= n_tokens || r >= rows) {
        return;
    }
    device const float* wr = w + (size_t)r * cols;
    device const float* xr = x + (size_t)t * cols;
    float acc = 0.0f;
    for (uint i = lane; i < cols; i += 32u) {
        acc += wr[i] * xr[i];
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        out[(size_t)t * rows + r] = acc;
    }
}
"#;

fn set_u32(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, v: u32, idx: usize) {
    let mut v = v;
    // SAFETY: `setBytes` copies the 4 bytes before returning.
    unsafe {
        enc.setBytes_length_atIndex(NonNull::new(&mut v as *mut u32 as *mut _).unwrap(), 4, idx);
    }
}

fn set_f32(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, v: f32, idx: usize) {
    let mut v = v;
    // SAFETY: `setBytes` copies the 4 bytes before returning.
    unsafe {
        enc.setBytes_length_atIndex(NonNull::new(&mut v as *mut f32 as *mut _).unwrap(), 4, idx);
    }
}

fn dispatch_1d(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, n: usize) {
    let tg = 256usize;
    dispatch_counted(
        enc,
        MTLSize {
            width: n.div_ceil(tg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
}

/// `y[..n] *= a` in place.
pub(crate) fn encode_scale(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    y: &ProtocolObject<dyn MTLBuffer>,
    a: f32,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, PLE_KERNEL_SRC, "ple_scale_f32")?;
    enc.setComputePipelineState(&pipe.0);
    // SAFETY: binds a live buffer the dispatch below reads and writes.
    unsafe { enc.setBuffer_offset_atIndex(Some(y), 0, 0) };
    set_f32(enc, a, 1);
    set_u32(enc, n, 2);
    dispatch_1d(enc, n as usize);
    Ok(())
}

/// `y[..n] = (y + x) * a` in place.
fn encode_add_scale(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    y: &ProtocolObject<dyn MTLBuffer>,
    x: &ProtocolObject<dyn MTLBuffer>,
    a: f32,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, PLE_KERNEL_SRC, "ple_add_scale_f32")?;
    enc.setComputePipelineState(&pipe.0);
    // SAFETY: binds live buffers the dispatch below reads and writes.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(y), 0, 0);
        enc.setBuffer_offset_atIndex(Some(x), 0, 1);
    }
    set_f32(enc, a, 2);
    set_u32(enc, n, 3);
    dispatch_1d(enc, n as usize);
    Ok(())
}

/// `out = gelu(gate) * up[row * up_stride + up_off ..]`, `rows` rows of `n`.
#[allow(clippy::too_many_arguments)]
fn encode_gelu_mul_rows(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    out_half: bool,
    n: u32,
    rows: u32,
    up_stride: u32,
    up_off: u32,
) -> Result<(), MetalError> {
    let name = if out_half {
        "ple_gelu_mul_rows_f16"
    } else {
        "ple_gelu_mul_rows_f32"
    };
    let pipe = ensure_pipeline(device, PLE_KERNEL_SRC, name)?;
    enc.setComputePipelineState(&pipe.0);
    // SAFETY: binds live buffers the dispatch below reads and writes.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
    }
    let total = n * rows;
    set_u32(enc, n, 3);
    set_u32(enc, total, 4);
    set_u32(enc, up_stride, 5);
    set_u32(enc, up_off, 6);
    dispatch_1d(enc, total as usize);
    Ok(())
}

/// `out[t, :] = w · x[t, :]` for f32 weights, `n_tokens` rows.
#[allow(clippy::too_many_arguments)]
fn encode_f32_mm(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    w: &ProtocolObject<dyn MTLBuffer>,
    x: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    cols: u32,
    rows: u32,
    n_tokens: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, PLE_KERNEL_SRC, "ple_f32_mm")?;
    enc.setComputePipelineState(&pipe.0);
    // SAFETY: binds live buffers the dispatch below reads and writes.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w), 0, 0);
        enc.setBuffer_offset_atIndex(Some(x), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
    }
    set_u32(enc, cols, 3);
    set_u32(enc, rows, 4);
    set_u32(enc, n_tokens, 5);
    dispatch_counted(
        enc,
        MTLSize {
            width: n_tokens as usize,
            height: (rows as usize).div_ceil(4),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// The scratch buffers a PLE-carrying stack owns for the length of one
/// command buffer: the per-layer input plane and the host rows it is
/// built from. Its own mutex, taken inside the stack's scratch guard and
/// released after the wait; nothing else takes it.
pub(crate) struct PleScratch {
    pub(crate) plane: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) tok: Retained<ProtocolObject<dyn MTLBuffer>>,
    cap: usize,
}

// SAFETY: gated by `PLE_SCRATCH`; one encode at a time.
unsafe impl Send for PleScratch {}

static PLE_SCRATCH: Mutex<Option<PleScratch>> = Mutex::new(None);

pub(crate) fn borrow_ple_scratch(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    elems: usize,
) -> Result<std::sync::MutexGuard<'static, Option<PleScratch>>, MetalError> {
    let mut guard = PLE_SCRATCH.lock().unwrap_or_else(|p| p.into_inner());
    if guard.as_ref().is_none_or(|s| s.cap < elems) {
        let alloc = |n: usize| {
            device
                .newBufferWithLength_options(n.max(1) * 4, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)
        };
        *guard = Some(PleScratch {
            plane: alloc(elems)?,
            tok: alloc(elems)?,
            cap: elems,
        });
    }
    Ok(guard)
}

impl<L> PleInputMetal<'_, L> {
    /// Elements of the plane for `batch` tokens.
    pub(crate) fn plane_elems(&self, batch: usize) -> usize {
        batch * self.n_layer * self.n_embd
    }

    pub(crate) fn check(&self, batch: usize, proj_rows: usize) {
        assert!(self.n_embd > 0 && self.n_layer > 0);
        assert_eq!(self.proj_norm_w.len(), self.n_embd);
        assert_eq!(self.tok_rows.len(), self.plane_elems(batch));
        assert_eq!(proj_rows, self.n_layer * self.n_embd);
    }

    /// Copy the host rows in. Before the encode, on the host.
    pub(crate) fn upload_tok(&self, scratch: &PleScratch) {
        crate::attn::copy_f32_into(&scratch.tok, self.tok_rows);
    }

    /// The three steps after the projection, shared by both stacks.
    #[allow(clippy::too_many_arguments)]
    fn encode_finish(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mrs: &mut MemRanges,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        scratch: &PleScratch,
        batch: usize,
        rms_eps: f32,
    ) -> Result<(), MetalError> {
        let plane = scratch.plane.as_ref();
        let tok = scratch.tok.as_ref();
        let n = self.plane_elems(batch) as u32;
        mrs.begin_op(enc, &[plane], &[plane]);
        encode_scale(enc, device, plane, self.proj_scale, n)?;
        mrs.end_op(&[plane], &[plane]);
        let norm_w = resident_f32_buffer(device, self.proj_norm_w)?;
        mrs.begin_op(enc, &[plane], &[plane]);
        crate::norm::encode_rms_norm_batch(
            enc,
            device,
            plane,
            &norm_w.buffer,
            plane,
            self.n_embd as u32,
            (batch * self.n_layer) as u32,
            rms_eps,
        )?;
        mrs.end_op(&[plane], &[plane]);
        mrs.begin_op(enc, &[plane, tok], &[plane]);
        encode_add_scale(enc, device, plane, tok, self.input_scale, n)?;
        mrs.end_op(&[plane, tok], &[plane]);
        Ok(())
    }
}

impl PleInputMetal<'_, MatvecLaunch<'_>> {
    /// Decode prologue: `h` is the scaled embedding of one token.
    pub(crate) fn encode_decode(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mrs: &mut MemRanges,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        scratch: &PleScratch,
        h: &ProtocolObject<dyn MTLBuffer>,
        rms_eps: f32,
    ) -> Result<(), MetalError> {
        let plane = scratch.plane.as_ref();
        let w = resident_weight_buffer(device, self.model_proj.weights)?;
        mrs.begin_op(enc, &[h], &[plane]);
        encode_matvec(enc, device, &self.model_proj, &w, h, plane)?;
        mrs.end_op(&[h], &[plane]);
        self.encode_finish(enc, mrs, device, scratch, 1, rms_eps)
    }
}

/// Batched `out = m · x` for the prefill stack, staging `x` through the
/// half plane when the weight is quantized.
#[allow(clippy::too_many_arguments)]
fn encode_prefill_mat(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    m: &PrefillMatMetal<'_>,
    x: &ProtocolObject<dyn MTLBuffer>,
    x_is_half: bool,
    half: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    cols: usize,
    batch: usize,
) -> Result<(), MetalError> {
    match m {
        PrefillMatMetal::Quant(l) => {
            let src = if x_is_half {
                x
            } else {
                mrs.begin_op(enc, &[x], &[half]);
                crate::elem::encode_f32_to_f16(enc, device, x, half, (batch * cols) as u32)?;
                mrs.end_op(&[x], &[half]);
                half
            };
            let w = resident_weight_buffer(device, l.weights)?;
            mrs.begin_op(enc, &[src], &[out]);
            encode_mul_mm_sg_f16(enc, device, l, &w, src, out, batch)?;
            mrs.end_op(&[src], &[out]);
        }
        PrefillMatMetal::F32 {
            weights,
            rows,
            cols: w_cols,
        } => {
            assert!(!x_is_half, "the f32 GEMM reads f32 activations");
            assert_eq!(*w_cols, cols);
            assert_eq!(weights.len(), rows * w_cols);
            let w = resident_f32_buffer(device, weights)?;
            mrs.begin_op(enc, &[x], &[out]);
            encode_f32_mm(
                enc,
                device,
                &w.buffer,
                x,
                out,
                cols as u32,
                *rows as u32,
                batch as u32,
            )?;
            mrs.end_op(&[x], &[out]);
        }
    }
    Ok(())
}

impl PleInputMetal<'_, PrefillMatMetal<'_>> {
    /// Prefill prologue: `h` holds `batch` scaled embeddings.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_prefill(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mrs: &mut MemRanges,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        scratch: &PleScratch,
        h: &ProtocolObject<dyn MTLBuffer>,
        half: &ProtocolObject<dyn MTLBuffer>,
        hidden_dim: usize,
        batch: usize,
        rms_eps: f32,
    ) -> Result<(), MetalError> {
        encode_prefill_mat(
            enc,
            mrs,
            device,
            &self.model_proj,
            h,
            false,
            half,
            &scratch.plane,
            hidden_dim,
            batch,
        )?;
        self.encode_finish(enc, mrs, device, scratch, batch, rms_eps)
    }
}

/// Buffers the epilogue may clobber: all are free once the FFN residual
/// has been added to `h`.
pub(crate) struct GateScratch<'a> {
    pub(crate) h: &'a ProtocolObject<dyn MTLBuffer>,
    /// `>= batch * n_embd` f32.
    pub(crate) gate: &'a ProtocolObject<dyn MTLBuffer>,
    /// `>= batch * n_embd` f32.
    pub(crate) act: &'a ProtocolObject<dyn MTLBuffer>,
    /// `>= batch * hidden` f32.
    pub(crate) out: &'a ProtocolObject<dyn MTLBuffer>,
    /// `>= batch * max(hidden, n_embd)` half; prefill only.
    pub(crate) half: Option<&'a ProtocolObject<dyn MTLBuffer>>,
}

impl<L> PerLayerGateMetal<'_, L> {
    /// The tail both stacks share once the projection is in `s.out`.
    #[allow(clippy::too_many_arguments)]
    fn encode_tail(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mrs: &mut MemRanges,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        s: &GateScratch<'_>,
        hidden_dim: usize,
        batch: usize,
        rms_eps: f32,
    ) -> Result<(), MetalError> {
        assert_eq!(self.post_norm.len(), hidden_dim);
        let pw = resident_f32_buffer(device, self.post_norm)?;
        mrs.begin_op(enc, &[s.out], &[s.out]);
        crate::norm::encode_rms_norm_batch(
            enc,
            device,
            s.out,
            &pw.buffer,
            s.out,
            hidden_dim as u32,
            batch as u32,
            rms_eps,
        )?;
        mrs.end_op(&[s.out], &[s.out]);
        mrs.begin_op(enc, &[s.h, s.out], &[s.h]);
        crate::elem::encode_vec_add(enc, device, s.h, s.out, (batch * hidden_dim) as u32)?;
        mrs.end_op(&[s.h, s.out], &[s.h]);
        Ok(())
    }
}

impl PerLayerGateMetal<'_, MatvecLaunch<'_>> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_decode(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mrs: &mut MemRanges,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        ple: &PleScratch,
        ple_layers: usize,
        s: &GateScratch<'_>,
        hidden_dim: usize,
        rms_eps: f32,
    ) -> Result<(), MetalError> {
        let n = self.gate.rows;
        assert_eq!(self.proj.rows, hidden_dim);
        assert!(self.slot < ple_layers);
        let plane = ple.plane.as_ref();
        let gate_w = resident_weight_buffer(device, self.gate.weights)?;
        let proj_w = resident_weight_buffer(device, self.proj.weights)?;
        mrs.begin_op(enc, &[s.h], &[s.gate]);
        encode_matvec(enc, device, &self.gate, &gate_w, s.h, s.gate)?;
        mrs.end_op(&[s.h], &[s.gate]);
        mrs.begin_op(enc, &[s.gate, plane], &[s.act]);
        encode_gelu_mul_rows(
            enc,
            device,
            s.gate,
            plane,
            s.act,
            false,
            n as u32,
            1,
            0,
            (self.slot * n) as u32,
        )?;
        mrs.end_op(&[s.gate, plane], &[s.act]);
        mrs.begin_op(enc, &[s.act], &[s.out]);
        encode_matvec(enc, device, &self.proj, &proj_w, s.act, s.out)?;
        mrs.end_op(&[s.act], &[s.out]);
        self.encode_tail(enc, mrs, device, s, hidden_dim, 1, rms_eps)
    }
}

impl PerLayerGateMetal<'_, PrefillMatMetal<'_>> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_prefill(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mrs: &mut MemRanges,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        ple: &PleScratch,
        ple_layers: usize,
        s: &GateScratch<'_>,
        hidden_dim: usize,
        batch: usize,
        rms_eps: f32,
    ) -> Result<(), MetalError> {
        let n = self.gate.rows();
        assert_eq!(self.proj.rows(), hidden_dim);
        assert!(self.slot < ple_layers);
        let half = s
            .half
            .expect("the prefill epilogue stages through the half plane");
        let plane = ple.plane.as_ref();
        encode_prefill_mat(
            enc, mrs, device, &self.gate, s.h, false, half, s.gate, hidden_dim, batch,
        )?;
        // The gated product feeds `proj`, so it is written in whatever
        // precision that GEMM reads.
        let to_half = self.proj.wants_half();
        let act = if to_half { half } else { s.act };
        mrs.begin_op(enc, &[s.gate, plane], &[act]);
        encode_gelu_mul_rows(
            enc,
            device,
            s.gate,
            plane,
            act,
            to_half,
            n as u32,
            batch as u32,
            (ple_layers * n) as u32,
            (self.slot * n) as u32,
        )?;
        mrs.end_op(&[s.gate, plane], &[act]);
        encode_prefill_mat(
            enc, mrs, device, &self.proj, act, to_half, half, s.out, n, batch,
        )?;
        self.encode_tail(enc, mrs, device, s, hidden_dim, batch, rms_eps)
    }
}

/// `h *= scale` over `n` elements, hazard-tracked: the layer's scalar
/// output scale (`layer_output_scale`).
pub(crate) fn encode_out_scale(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    h: &ProtocolObject<dyn MTLBuffer>,
    scale: f32,
    n: usize,
) -> Result<(), MetalError> {
    mrs.begin_op(enc, &[h], &[h]);
    encode_scale(enc, device, h, scale, n as u32)?;
    mrs.end_op(&[h], &[h]);
    Ok(())
}
