//! A Mamba block over any number of rows: a token mixer of the dense
//! decode stack (`crate::decode_dense::TokenMixer::Mamba1`, a matvec per
//! projection) and a layer of a prefill run (`crate::ssm_prefill`, a
//! GEMM per projection), one body over [`SsmProj`]. The conv, the small
//! projections, the scan and the gate are each a dispatch in the
//! caller's command buffer, with the layer's state the host's own memory
//! wrapped in place (`crate::gdn::buffer_no_copy`).
//!
//! The arithmetic is `frink_models::mamba1` (`mamba-base.cpp:4-148`):
//!
//! ```text
//! xz   = in_proj(x)                          [2 d_inner]       q buffer
//! xc   = silu(conv(state, xz[..d_inner]) + b) [d_inner]        attn buffer
//! x_db = x_proj(xc)                          [dt_rank + 2 d_state]  k buffer
//! dt   = dt_proj(x_db[..dt_rank])            [d_inner]         v buffer
//! y    = scan(state, xc, dt + dt_b, A, B, C) [d_inner]         gate buffer
//! y    = (y + xc * D) * silu(xz[d_inner..])
//! out  = out_proj(y)                                           o buffer
//! ```

use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::gpu::{ensure_pipeline, MatvecLaunch, MetalError, MulMmSgLaunch};
use crate::mem_ranges::MemRanges;

pub const SSM_DECODE_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// y = (y + x * D[c / head_dim]) * silu(z), one thread per element of
// `rows` rows of `n` (`mamba-base.cpp:121-122,268-269`; Mamba-1's heads
// are one channel wide). `y` is `[rows][n]`; row r's x and z are
// `r * x_stride` and `r * z_stride` into theirs, since both are read
// out of wider rows (the conv output, `in_proj`).
kernel void ssm_skip_gate(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* z [[buffer(2)]],
    device const float* d [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    const uint n = dims.x;
    if (i >= n * rows) {
        return;
    }
    const uint r = i / n;
    const uint c = i - r * n;
    const float zi = z[(size_t)r * dims.w + c];
    const float xi = x[(size_t)r * dims.z + c];
    y[i] = (y[i] + xi * d[c / dims.y]) * (zi / (1.0f + exp(-zi)));
}

// An RMSNorm over each of `n_group` groups of `gw` floats in each of
// `rows` rows of `x` (row stride `stride`), in place, times the group's
// slice of `weight` (`[n_group][gw]`), or by nothing when `dims.w` is 0
// (`build_norm(x, NULL, ..)`). One threadgroup per (row, group).
// `mamba-base.cpp:271-274` on Mamba-2's `y`, and Mamba-1's dt / B / C
// norms (`:84-88`, a "group" per vector, the weights concatenated).
kernel void ssm_group_rms_norm(
    device float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint4& dims [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint n_threads [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint gw = dims.x;
    const uint n_group = dims.y;
    const uint stride = dims.z;
    const uint r = tg / n_group;
    const uint g = tg - r * n_group;
    device float* row = x + (size_t)r * stride + (size_t)g * gw;
    device const float* w = weight + (size_t)g * gw;
    float partial = 0.0f;
    for (uint i = tid; i < gw; i += n_threads) {
        const float v = row[i];
        partial += v * v;
    }
    partial = simd_sum(partial);
    if (lane == 0u) {
        scratch[sg] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (n_threads + 31u) / 32u;
    if (lane < nsg) {
        total = scratch[lane];
    }
    total = simd_sum(total);
    const float scale = rsqrt(total / float(gw) + eps);
    for (uint i = tid; i < gw; i += n_threads) {
        row[i] = dims.w != 0u ? row[i] * scale * w[i] : row[i] * scale;
    }
}

// `out[r][c] = x[r * stride + c]` for `c < n`: one column band of a
// batch of rows made contiguous, for the projection that reads it
// (Mamba-1's `dt_proj` reads the first `dt_rank` of each `x_proj` row).
kernel void ssm_copy_cols(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint4& dims [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    const uint n = dims.x;
    if (i >= n * dims.z) {
        return;
    }
    const uint r = i / n;
    out[i] = x[(size_t)r * dims.y + (i - r * n)];
}
"#;

/// The conv, its bias and SiLU over `rows` tokens of `width` channels
/// read `x_stride` apart from `x`: the gated delta-net's conv kernel
/// with the bias on (`crate::gdn_head::encode_conv_silu_strided`), so a
/// decode token and a prompt batch run one body.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv_bias_silu(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    state: &ProtocolObject<dyn MTLBuffer>,
    taps: &ProtocolObject<dyn MTLBuffer>,
    bias: &ProtocolObject<dyn MTLBuffer>,
    d_conv: usize,
    x: (&ProtocolObject<dyn MTLBuffer>, usize),
    x_stride: usize,
    out: &ProtocolObject<dyn MTLBuffer>,
    width: usize,
    rows: usize,
) -> Result<(), MetalError> {
    crate::gdn_head::encode_conv_silu_strided(
        encoder,
        device,
        crate::gdn_head::ConvRows {
            width,
            d_conv,
            state,
            taps,
            bias: Some(bias),
            x,
            x_stride,
            out,
            rows,
        },
    )
}

/// `y = (y + x * D) * silu(z)` over `rows` rows of `n`; `x` and `z` are
/// (buffer, byte offset, row stride in floats).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_skip_gate(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    y: &ProtocolObject<dyn MTLBuffer>,
    x: (&ProtocolObject<dyn MTLBuffer>, usize, usize),
    z: (&ProtocolObject<dyn MTLBuffer>, usize, usize),
    d: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    head_dim: usize,
    rows: usize,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, SSM_DECODE_KERNEL_SRC, "ssm_skip_gate")?;
    encoder.setComputePipelineState(&pipe.0);
    // SAFETY: the buffers outlive the caller's command buffer; the
    // uniforms are copied by `setBytes`.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(y), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(x.0), x.1, 1);
        encoder.setBuffer_offset_atIndex(Some(z.0), z.1, 2);
        encoder.setBuffer_offset_atIndex(Some(d), 0, 3);
        let mut dims: [u32; 4] = [n as u32, head_dim as u32, x.2 as u32, z.2 as u32];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 16, 4);
        let mut r = rows as u32;
        encoder.setBytes_length_atIndex(NonNull::new(&mut r as *mut u32 as *mut _).unwrap(), 4, 5);
    }
    dispatch_1d(encoder, &pipe.0, n * rows);
    Ok(())
}

/// [`SSM_DECODE_KERNEL_SRC`]'s `ssm_group_rms_norm`: `n_group` groups of
/// `gw` in each of `rows` rows `stride` floats apart, in place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_group_rms_norm(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: (&ProtocolObject<dyn MTLBuffer>, usize),
    weight: Option<&ProtocolObject<dyn MTLBuffer>>,
    gw: usize,
    n_group: usize,
    stride: usize,
    rows: usize,
    eps: f32,
) -> Result<(), MetalError> {
    if gw == 0 || n_group == 0 || stride < gw * n_group || rows == 0 {
        return Err(MetalError::Unsupported("ssm group norm shape"));
    }
    let pipe = ensure_pipeline(device, SSM_DECODE_KERNEL_SRC, "ssm_group_rms_norm")?;
    encoder.setComputePipelineState(&pipe.0);
    let threads = gw.next_power_of_two().clamp(32, 1024);
    // SAFETY: as `encode_skip_gate`'s.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x.0), x.1, 0);
        // Unweighted, index 1 is bound to `x` and never read.
        encoder.setBuffer_offset_atIndex(Some(weight.unwrap_or(x.0)), 0, 1);
        let mut dims: [u32; 4] = [
            gw as u32,
            n_group as u32,
            stride as u32,
            u32::from(weight.is_some()),
        ];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 16, 2);
        let mut e = eps;
        encoder.setBytes_length_atIndex(NonNull::new(&mut e as *mut f32 as *mut _).unwrap(), 4, 3);
        encoder.setThreadgroupMemoryLength_atIndex(32 * 4, 0);
    }
    crate::dispatch::dispatch_counted(
        encoder,
        MTLSize {
            width: rows * n_group,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// `out = x[.., ..n]`: the first `n` floats of each of `rows` rows
/// `stride` apart, made contiguous.
pub(crate) fn encode_copy_cols(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: (&ProtocolObject<dyn MTLBuffer>, usize),
    stride: usize,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    rows: usize,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, SSM_DECODE_KERNEL_SRC, "ssm_copy_cols")?;
    encoder.setComputePipelineState(&pipe.0);
    // SAFETY: as `encode_skip_gate`'s.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x.0), x.1, 0);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 1);
        let mut dims: [u32; 4] = [n as u32, stride as u32, rows as u32, 0];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 16, 2);
    }
    dispatch_1d(encoder, &pipe.0, n * rows);
    Ok(())
}

/// A projection a Mamba block runs: a matvec for a decode token, a GEMM
/// for a prompt batch. The block bodies below are written once over
/// this, so a decode token and a prefill batch cannot run different
/// arithmetic around their projections.
pub trait SsmProj {
    fn out_rows(&self) -> usize;
    fn in_cols(&self) -> usize;
    /// `out = W x` over `rows` rows of `x`.
    fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        x: &ProtocolObject<dyn MTLBuffer>,
        out: &ProtocolObject<dyn MTLBuffer>,
        rows: usize,
    ) -> Result<(), MetalError>;
}

impl SsmProj for MatvecLaunch<'_> {
    fn out_rows(&self) -> usize {
        self.rows
    }
    fn in_cols(&self) -> usize {
        self.row_bytes / self.block_bytes * self.block_elems
    }
    fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        x: &ProtocolObject<dyn MTLBuffer>,
        out: &ProtocolObject<dyn MTLBuffer>,
        rows: usize,
    ) -> Result<(), MetalError> {
        if rows != 1 {
            return Err(MetalError::Unsupported("a matvec projects one row"));
        }
        let w = crate::gpu::resident_weight_buffer(device, self.weights)?;
        crate::gpu::encode_matvec(encoder, device, self, &w, x, out)
    }
}

impl SsmProj for MulMmSgLaunch<'_> {
    fn out_rows(&self) -> usize {
        self.rows
    }
    fn in_cols(&self) -> usize {
        self.row_bytes / self.block_bytes * self.block_elems
    }
    fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        x: &ProtocolObject<dyn MTLBuffer>,
        out: &ProtocolObject<dyn MTLBuffer>,
        rows: usize,
    ) -> Result<(), MetalError> {
        let w = crate::gpu::resident_weight_buffer(device, self.weights)?;
        crate::gpu::encode_mul_mm_sg(encoder, device, self, &w, x, out, rows)
    }
}

/// `mamba-base.cpp:84-88`'s RMS norms on dt, B and C: the three weights
/// (Jamba), or none of them (FalconMamba's `ssm.dt_b_c_rms`, a
/// weightless norm on each).
#[derive(Clone, Copy)]
pub struct DtBcNormMetal<'a> {
    pub weights: Option<(&'a [f32], &'a [f32], &'a [f32])>,
    pub eps: f32,
}

/// One Mamba-1 layer's block, for the dense stack (a matvec per
/// projection) or a prefill run (a GEMM, `crate::ssm_prefill`). Built
/// only through [`Mamba1Mixer::new`], whose contract is what makes the
/// state pointers sound to hand a safe launch.
pub struct Mamba1Mixer<'a, P = MatvecLaunch<'a>> {
    w: Mamba1Weights<'a, P>,
    conv_state: (*mut f32, usize),
    ssm_state: (*mut f32, usize),
}

/// The weights [`Mamba1Mixer::new`] takes, by name. The per-channel
/// planes are shared allocations, so the resident copy a decode token
/// binds is found by address (`gpu::resident_f32_shared`); as slices,
/// proving the cache still held them was a compare of every byte, 35 MB
/// a token on Mamba-1.4B, which halved its decode rate whenever the
/// GPU was busy with the token before (the early-commit split).
pub struct Mamba1Weights<'a, P = MatvecLaunch<'a>> {
    pub in_proj: P,
    pub conv1d: &'a Arc<[f32]>,
    pub conv_bias: &'a Arc<[f32]>,
    pub d_conv: usize,
    pub x_proj: P,
    pub dt_bc_norm: Option<DtBcNormMetal<'a>>,
    pub dt_proj: P,
    pub dt_bias: &'a Arc<[f32]>,
    pub a: &'a Arc<[f32]>,
    pub d: &'a Arc<[f32]>,
    pub out_proj: P,
    pub d_inner: usize,
    pub d_state: usize,
    pub dt_rank: usize,
}

impl<'a, P: SsmProj> Mamba1Mixer<'a, P> {
    /// `None` when a shape disagrees with the others.
    ///
    /// # Safety
    ///
    /// The two state allocations (`ptr`, byte length) must be page-aligned
    /// and whole pages, hold `(d_conv - 1) * d_inner` and `d_inner *
    /// d_state` floats, stay valid, and be touched by nothing else until
    /// the launch this mixer is handed to returns.
    pub unsafe fn new(
        w: Mamba1Weights<'a, P>,
        conv_state: (*mut f32, usize),
        ssm_state: (*mut f32, usize),
    ) -> Option<Self> {
        let norm_ok = w.dt_bc_norm.is_none_or(|n| {
            n.weights.is_none_or(|(dt, b, c)| {
                dt.len() == w.dt_rank && b.len() == w.d_state && c.len() == w.d_state
            })
        });
        let ok = w.d_conv >= 2
            && w.d_conv <= 8
            && w.in_proj.out_rows() == 2 * w.d_inner
            && w.conv1d.len() == w.d_conv * w.d_inner
            && w.conv_bias.len() == w.d_inner
            && w.x_proj.out_rows() == w.dt_rank + 2 * w.d_state
            && w.x_proj.in_cols() == w.d_inner
            && w.dt_proj.out_rows() == w.d_inner
            && w.dt_proj.in_cols() == w.dt_rank
            && w.dt_bias.len() == w.d_inner
            && w.a.len() == w.d_inner * w.d_state
            && w.d.len() == w.d_inner
            && w.out_proj.in_cols() == w.d_inner
            && norm_ok
            && (1..=1024).contains(&w.d_state)
            && conv_state.1 >= (w.d_conv - 1) * w.d_inner * 4
            && ssm_state.1 >= w.d_inner * w.d_state * 4;
        ok.then_some(Self {
            w,
            conv_state,
            ssm_state,
        })
    }

    /// The block's output width, `ssm_out`'s rows.
    pub(crate) fn out_rows(&self) -> usize {
        self.w.out_proj.out_rows()
    }

    /// The input width, `ssm_in`'s columns.
    pub(crate) fn in_cols(&self) -> usize {
        self.w.in_proj.in_cols()
    }

    /// `dt_rank`: the width of a batch's contiguous `dt` input.
    pub(crate) fn dt_rank(&self) -> usize {
        self.w.dt_rank
    }

    /// The scratch this block needs PER ROW: `xz` in the Q buffer, `x_db`
    /// and `dt` in the K and V buffers, `xc` in the attention buffer and
    /// `y` in the gate buffer.
    pub(crate) fn widths(&self) -> ScratchNeed {
        let w = &self.w;
        ScratchNeed {
            q: 2 * w.d_inner,
            kv: (w.dt_rank + 2 * w.d_state).max(w.d_inner),
            attn: w.d_inner,
            gate: w.d_inner,
        }
    }
}

/// How wide a block needs each of the dense stack's scratch buffers.
/// Named, because the two blocks put different things in each and a
/// tuple let one block's widths size the other's buffers.
#[derive(Clone, Copy, Default)]
pub(crate) struct ScratchNeed {
    pub q: usize,
    pub kv: usize,
    pub attn: usize,
    pub gate: usize,
}

impl ScratchNeed {
    pub(crate) fn max(self, o: Self) -> Self {
        Self {
            q: self.q.max(o.q),
            kv: self.kv.max(o.kv),
            attn: self.attn.max(o.attn),
            gate: self.gate.max(o.gate),
        }
    }
}

/// The scratch, as this block uses it.
pub(crate) struct Mamba1Bufs<'b> {
    pub x: &'b ProtocolObject<dyn MTLBuffer>,
    pub xz: &'b ProtocolObject<dyn MTLBuffer>,
    pub xc: &'b ProtocolObject<dyn MTLBuffer>,
    pub x_db: &'b ProtocolObject<dyn MTLBuffer>,
    pub dt: &'b ProtocolObject<dyn MTLBuffer>,
    /// `x_db`'s first `dt_rank` columns made contiguous, which a batch's
    /// GEMM needs and a token's matvec does not (it reads the head of
    /// the one row).
    pub dt_low: Option<&'b ProtocolObject<dyn MTLBuffer>>,
    pub y: &'b ProtocolObject<dyn MTLBuffer>,
    pub out: &'b ProtocolObject<dyn MTLBuffer>,
}

/// `Err` unless `buf` holds `floats` floats: the stack sizes its
/// scratch from [`ScratchNeed`], and a block that wrote past one would
/// corrupt its neighbour silently (it did, once, when the widths were a
/// tuple the two blocks read differently).
fn fits(buf: &ProtocolObject<dyn MTLBuffer>, floats: usize) -> Result<(), MetalError> {
    if buf.length() < floats * 4 {
        return Err(MetalError::Unsupported("ssm block scratch too small"));
    }
    Ok(())
}

fn dispatch_1d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipe: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    n: usize,
) {
    let tg = pipe.maxTotalThreadsPerThreadgroup().clamp(1, 256);
    crate::dispatch::dispatch_counted(
        encoder,
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

/// `out = proj(x)` through the hazard tracker.
#[allow(clippy::too_many_arguments)]
fn project<P: SsmProj>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    p: &P,
    x: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
) -> Result<(), MetalError> {
    mrs.begin_op(encoder, &[x], &[out]);
    let r = p.encode(encoder, device, x, out, rows);
    mrs.end_op(&[x], &[out]);
    r
}

/// The block over `rows` consecutive tokens, `out = mamba1(x)`
/// (`[rows][n_embd]` in, `[rows][n_embd]` out), into the caller's
/// encoder: one token in the decode stack, a prompt batch in a prefill
/// run.
pub(crate) fn encode_mamba1<P: SsmProj>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    m: &Mamba1Mixer<'_, P>,
    b: &Mamba1Bufs<'_>,
    rows: usize,
) -> Result<(), MetalError> {
    use crate::gpu::{resident_f32_buffer, resident_f32_shared};
    let w = &m.w;
    let n = w.d_inner;
    let x_db_w = w.dt_rank + 2 * w.d_state;
    let need = m.widths();
    fits(b.xz, rows * need.q)?;
    fits(b.x_db, rows * need.kv)?;
    fits(b.dt, rows * need.kv)?;
    fits(b.xc, rows * need.attn)?;
    fits(b.y, rows * need.gate)?;
    let dt_low = match (rows, b.dt_low) {
        (1, _) => None,
        (_, Some(buf)) => {
            fits(buf, rows * w.dt_rank)?;
            Some(buf)
        }
        (_, None) => return Err(MetalError::Unsupported("a batch's dt needs its own buffer")),
    };
    // SAFETY: `Mamba1Mixer::new`'s contract.
    let conv_state = unsafe { crate::gdn::buffer_no_copy(device, m.conv_state.0, m.conv_state.1) }
        .ok_or(MetalError::BufferAllocFailed)?;
    // SAFETY: as above.
    let ssm_state = unsafe { crate::gdn::buffer_no_copy(device, m.ssm_state.0, m.ssm_state.1) }
        .ok_or(MetalError::BufferAllocFailed)?;
    let (conv_state, ssm_state): (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ) = (&conv_state, &ssm_state);
    let taps = resident_f32_shared(device, w.conv1d)?;
    let conv_b = resident_f32_shared(device, w.conv_bias)?;
    let dt_b = resident_f32_shared(device, w.dt_bias)?;
    let a = resident_f32_shared(device, w.a)?;
    let d = resident_f32_shared(device, w.d)?;
    // Compiled before the first dispatch of this block is encoded.
    ensure_pipeline(
        device,
        crate::gdn_head::GDN_HEAD_KERNEL_SRC,
        "gdn_conv_silu",
    )?;
    ensure_pipeline(device, SSM_DECODE_KERNEL_SRC, "ssm_skip_gate")?;

    // :34 xz = in_proj(x).
    project(encoder, mrs, device, &w.in_proj, b.x, b.xz, rows)?;

    // :43-70 xc = silu(conv(x) + b), the window stepped in place.
    mrs.begin_op(encoder, &[b.xz, conv_state], &[b.xc, conv_state]);
    encode_conv_bias_silu(
        encoder,
        device,
        conv_state,
        &taps.buffer,
        &conv_b.buffer,
        w.d_conv,
        (b.xz, 0),
        2 * n,
        b.xc,
        n,
        rows,
    )?;
    mrs.end_op(&[b.xz, conv_state], &[b.xc, conv_state]);

    // :74 x_db = x_proj(xc).
    project(encoder, mrs, device, &w.x_proj, b.xc, b.x_db, rows)?;

    // :84-88 the norms on dt, B and C, in place in `x_db`.
    if let Some(norm) = w.dt_bc_norm {
        let (wdt, wb, wc) = match norm.weights {
            Some((dt, bw, cw)) => (
                Some(resident_f32_buffer(device, dt)?),
                Some(resident_f32_buffer(device, bw)?),
                Some(resident_f32_buffer(device, cw)?),
            ),
            None => (None, None, None),
        };
        mrs.begin_op(encoder, &[b.x_db], &[b.x_db]);
        for (off, width, weight) in [
            (0, w.dt_rank, &wdt),
            (w.dt_rank, w.d_state, &wb),
            (w.dt_rank + w.d_state, w.d_state, &wc),
        ] {
            encode_group_rms_norm(
                encoder,
                device,
                (b.x_db, off * 4),
                weight.as_ref().map(|r| &*r.buffer),
                width,
                1,
                x_db_w,
                rows,
                norm.eps,
            )?;
        }
        mrs.end_op(&[b.x_db], &[b.x_db]);
    }

    // :90 dt = dt_proj(x_db[..dt_rank]).
    let dt_in = match dt_low {
        None => b.x_db,
        Some(buf) => {
            mrs.begin_op(encoder, &[b.x_db], &[buf]);
            encode_copy_cols(encoder, device, (b.x_db, 0), x_db_w, buf, w.dt_rank, rows)?;
            mrs.end_op(&[b.x_db], &[buf]);
            buf
        }
    };
    project(encoder, mrs, device, &w.dt_proj, dt_in, b.dt, rows)?;

    // :104-117 the scan, dt's bias added in the kernel.
    mrs.begin_op(encoder, &[b.xc, b.x_db, b.dt, ssm_state], &[b.y, ssm_state]);
    crate::ssm_scan::encode_scan(
        encoder,
        device,
        crate::ssm_scan::ScanShape {
            n_head: n,
            head_dim: 1,
            d_state: w.d_state,
            n_group: 1,
        },
        crate::ssm_scan::ScanOperands {
            state: ssm_state,
            x: (b.xc, 0),
            bc: (b.x_db, w.dt_rank * 4),
            x_stride: n,
            bc_stride: x_db_w,
            dt: (b.dt, 0),
            dt_stride: n,
            dt_bias: Some(&dt_b.buffer),
            a: &a.buffer,
            per_state: true,
            y: b.y,
            rows,
        },
    )?;
    mrs.end_op(&[b.xc, b.x_db, b.dt, ssm_state], &[b.y, ssm_state]);

    // :121-122 y = (y + xc * D) * silu(z).
    mrs.begin_op(encoder, &[b.y, b.xc, b.xz], &[b.y]);
    encode_skip_gate(
        encoder,
        device,
        b.y,
        (b.xc, 0, n),
        (b.xz, n * 4, 2 * n),
        &d.buffer,
        n,
        1,
        rows,
    )?;
    mrs.end_op(&[b.y, b.xc, b.xz], &[b.y]);

    // :124 out = out_proj(y).
    project(encoder, mrs, device, &w.out_proj, b.y, b.out, rows)
}

/// One Mamba-2 layer's block (`frink_models::mamba2`,
/// `mamba-base.cpp:149-288`), for the dense stack or a prefill run:
///
/// ```text
/// zxBCdt = in_proj(x)                        q buffer
/// xBC    = silu(conv(state, xBC) + b)        attention buffer
/// y      = scan(state, x, dt + dt_b, A, B, C) gate buffer
/// y      = (y + x * D[head]) * silu(z)
/// y      = rms_norm(y per group) * ssm_norm
/// out    = out_proj(y)
/// ```
pub struct Mamba2Mixer<'a, P = MatvecLaunch<'a>> {
    w: Mamba2Weights<'a, P>,
    conv_state: (*mut f32, usize),
    ssm_state: (*mut f32, usize),
}

/// The weights [`Mamba2Mixer::new`] takes, by name. The conv planes and
/// `ssm_norm` are shared, as [`Mamba1Weights`]' are; the per-head
/// vectors are `n_head` floats, cheap to compare.
pub struct Mamba2Weights<'a, P = MatvecLaunch<'a>> {
    pub in_proj: P,
    pub conv1d: &'a Arc<[f32]>,
    pub conv_bias: &'a Arc<[f32]>,
    pub d_conv: usize,
    pub dt_bias: &'a [f32],
    pub a: &'a [f32],
    pub d: &'a [f32],
    /// `ssm_norm`, `[d_inner]`, applied per group of `d_inner / n_group`.
    pub norm: Option<&'a Arc<[f32]>>,
    pub norm_eps: f32,
    pub out_proj: P,
    pub d_inner: usize,
    pub d_state: usize,
    pub n_head: usize,
    pub n_group: usize,
}

impl<P> Mamba2Weights<'_, P> {
    /// The conv's channels, `d_inner + 2 n_group d_state`.
    fn conv_width(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }
}

impl<'a, P: SsmProj> Mamba2Mixer<'a, P> {
    /// `None` when a shape disagrees with the others.
    ///
    /// # Safety
    ///
    /// As [`Mamba1Mixer::new`], with the states `(d_conv - 1) *
    /// conv_width` and `d_inner * d_state` floats.
    pub unsafe fn new(
        w: Mamba2Weights<'a, P>,
        conv_state: (*mut f32, usize),
        ssm_state: (*mut f32, usize),
    ) -> Option<Self> {
        let width = w.conv_width();
        let ok = w.d_conv >= 2
            && w.d_conv <= 8
            && w.n_head > 0
            && w.d_inner.is_multiple_of(w.n_head)
            && w.n_group > 0
            && w.n_head.is_multiple_of(w.n_group)
            && w.in_proj.out_rows() == w.d_inner + width + w.n_head
            && w.conv1d.len() == w.d_conv * width
            && w.conv_bias.len() == width
            && w.dt_bias.len() == w.n_head
            && w.a.len() == w.n_head
            && w.d.len() == w.n_head
            && w.d_inner.is_multiple_of(w.n_group)
            && w.norm.is_none_or(|n| n.len() == w.d_inner)
            && w.out_proj.in_cols() == w.d_inner
            && (1..=1024).contains(&w.d_state)
            && conv_state.1 >= (w.d_conv - 1) * width * 4
            && ssm_state.1 >= w.d_inner * w.d_state * 4;
        ok.then_some(Self {
            w,
            conv_state,
            ssm_state,
        })
    }

    pub(crate) fn out_rows(&self) -> usize {
        self.w.out_proj.out_rows()
    }

    pub(crate) fn in_cols(&self) -> usize {
        self.w.in_proj.in_cols()
    }

    /// The scratch PER ROW: `zxBCdt` in the Q buffer, `xBC` in the
    /// attention buffer, `y` in the gate buffer.
    pub(crate) fn widths(&self) -> ScratchNeed {
        ScratchNeed {
            q: self.w.in_proj.out_rows(),
            kv: 0,
            attn: self.w.conv_width(),
            gate: self.w.d_inner,
        }
    }
}

/// The scratch, as the Mamba-2 block uses it.
pub(crate) struct Mamba2Bufs<'b> {
    pub x: &'b ProtocolObject<dyn MTLBuffer>,
    pub zxbcdt: &'b ProtocolObject<dyn MTLBuffer>,
    pub xbc: &'b ProtocolObject<dyn MTLBuffer>,
    pub y: &'b ProtocolObject<dyn MTLBuffer>,
    pub out: &'b ProtocolObject<dyn MTLBuffer>,
}

/// The Mamba-2 block over `rows` consecutive tokens, `out = mamba2(x)`.
pub(crate) fn encode_mamba2<P: SsmProj>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    m: &Mamba2Mixer<'_, P>,
    b: &Mamba2Bufs<'_>,
    rows: usize,
) -> Result<(), MetalError> {
    use crate::gpu::{resident_f32_buffer, resident_f32_shared};
    let w = &m.w;
    let (n, width) = (w.d_inner, w.conv_width());
    let stride = w.in_proj.out_rows();
    let need = m.widths();
    fits(b.zxbcdt, rows * need.q)?;
    fits(b.xbc, rows * need.attn)?;
    fits(b.y, rows * need.gate)?;
    // SAFETY: `Mamba2Mixer::new`'s contract.
    let conv_state = unsafe { crate::gdn::buffer_no_copy(device, m.conv_state.0, m.conv_state.1) }
        .ok_or(MetalError::BufferAllocFailed)?;
    // SAFETY: as above.
    let ssm_state = unsafe { crate::gdn::buffer_no_copy(device, m.ssm_state.0, m.ssm_state.1) }
        .ok_or(MetalError::BufferAllocFailed)?;
    let (conv_state, ssm_state): (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ) = (&conv_state, &ssm_state);
    let taps = resident_f32_shared(device, w.conv1d)?;
    let conv_b = resident_f32_shared(device, w.conv_bias)?;
    let dt_b = resident_f32_buffer(device, w.dt_bias)?;
    let a = resident_f32_buffer(device, w.a)?;
    let d = resident_f32_buffer(device, w.d)?;
    let norm = match w.norm {
        Some(nw) => Some(resident_f32_shared(device, nw)?),
        None => None,
    };
    ensure_pipeline(
        device,
        crate::gdn_head::GDN_HEAD_KERNEL_SRC,
        "gdn_conv_silu",
    )?;
    ensure_pipeline(device, SSM_DECODE_KERNEL_SRC, "ssm_skip_gate")?;

    // :180 zxBCdt = in_proj(x).
    project(encoder, mrs, device, &w.in_proj, b.x, b.zxbcdt, rows)?;

    // :195-226 xBC = silu(conv(xBC) + b), the window stepped in place.
    mrs.begin_op(encoder, &[b.zxbcdt, conv_state], &[b.xbc, conv_state]);
    encode_conv_bias_silu(
        encoder,
        device,
        conv_state,
        &taps.buffer,
        &conv_b.buffer,
        w.d_conv,
        (b.zxbcdt, n * 4),
        stride,
        b.xbc,
        width,
        rows,
    )?;
    mrs.end_op(&[b.zxbcdt, conv_state], &[b.xbc, conv_state]);

    // :239-258 the scan, dt (the last n_head of each zxBCdt row) plus
    // its bias.
    mrs.begin_op(encoder, &[b.xbc, b.zxbcdt, ssm_state], &[b.y, ssm_state]);
    crate::ssm_scan::encode_scan(
        encoder,
        device,
        crate::ssm_scan::ScanShape {
            n_head: w.n_head,
            head_dim: n / w.n_head,
            d_state: w.d_state,
            n_group: w.n_group,
        },
        crate::ssm_scan::ScanOperands {
            state: ssm_state,
            x: (b.xbc, 0),
            bc: (b.xbc, n * 4),
            x_stride: width,
            bc_stride: width,
            dt: (b.zxbcdt, (n + width) * 4),
            dt_stride: stride,
            dt_bias: Some(&dt_b.buffer),
            a: &a.buffer,
            per_state: false,
            y: b.y,
            rows,
        },
    )?;
    mrs.end_op(&[b.xbc, b.zxbcdt, ssm_state], &[b.y, ssm_state]);

    // :268-269 y = (y + x * D[head]) * silu(z), z the first d_inner.
    mrs.begin_op(encoder, &[b.y, b.xbc, b.zxbcdt], &[b.y]);
    encode_skip_gate(
        encoder,
        device,
        b.y,
        (b.xbc, 0, width),
        (b.zxbcdt, 0, stride),
        &d.buffer,
        n,
        n / w.n_head,
        rows,
    )?;
    mrs.end_op(&[b.y, b.xbc, b.zxbcdt], &[b.y]);

    // :271-274 an RMSNorm over each group's `d_inner / n_group`
    // channels, times the group's slice of `ssm_norm`.
    if let Some(nw) = &norm {
        mrs.begin_op(encoder, &[b.y], &[b.y]);
        encode_group_rms_norm(
            encoder,
            device,
            (b.y, 0),
            Some(&nw.buffer),
            n / w.n_group,
            w.n_group,
            n,
            rows,
            w.norm_eps,
        )?;
        mrs.end_op(&[b.y], &[b.y]);
    }

    // :278 out = out_proj(y).
    project(encoder, mrs, device, &w.out_proj, b.y, b.out, rows)
}
