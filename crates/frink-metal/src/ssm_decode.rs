//! A Mamba block as a token mixer of the dense decode stack
//! (`crate::decode_dense::TokenMixer::Mamba1`): the conv, the two small
//! projections, the scan and the gate, every one a dispatch in the
//! stack's command buffer, with the layer's state the host's own memory
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

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::gpu::{ensure_pipeline, MatvecLaunch, MetalError};

pub const SSM_DECODE_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One token of the causal conv, its bias and SiLU, one thread per
// channel, the window shifted in place (`frink_core::mamba2::conv_step`
// then `silu(v + b)`).
kernel void ssm_conv_bias_silu(
    device float* state [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint2& dims [[buffer(5)]],
    uint c [[thread_position_in_grid]]
) {
    const uint width = dims.x;
    const uint d_conv = dims.y;
    if (c >= width) {
        return;
    }
    device const float* t = taps + (size_t)c * d_conv;
    float acc = 0.0f;
    for (uint i = 0; i + 1u < d_conv; ++i) {
        acc += state[i * width + c] * t[i];
    }
    const float xc = x[c];
    acc += xc * t[d_conv - 1u];
    const float v = acc + bias[c];
    out[c] = v / (1.0f + exp(-v));
    for (uint i = 0; i + 2u < d_conv; ++i) {
        state[i * width + c] = state[(i + 1u) * width + c];
    }
    state[(d_conv - 2u) * width + c] = xc;
}

// y = (y + x * D[i / head_dim]) * silu(z), one thread per element
// (`mamba-base.cpp:121-122`; Mamba-1's heads are one channel wide).
kernel void ssm_skip_gate(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* z [[buffer(2)]],
    device const float* d [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= dims.x) {
        return;
    }
    const float zi = z[i];
    y[i] = (y[i] + x[i] * d[i / dims.y]) * (zi / (1.0f + exp(-zi)));
}
"#;

/// One Mamba-1 layer's block, for the dense stack. Built only through
/// [`Mamba1Mixer::new`], whose contract is what makes the state
/// pointers sound to hand a safe launch.
pub struct Mamba1Mixer<'a> {
    w: Mamba1Weights<'a>,
    conv_state: (*mut f32, usize),
    ssm_state: (*mut f32, usize),
}

/// The weights [`Mamba1Mixer::new`] takes, by name.
pub struct Mamba1Weights<'a> {
    pub in_proj: MatvecLaunch<'a>,
    pub conv1d: &'a [f32],
    pub conv_bias: &'a [f32],
    pub d_conv: usize,
    pub x_proj: MatvecLaunch<'a>,
    pub dt_proj: MatvecLaunch<'a>,
    pub dt_bias: &'a [f32],
    pub a: &'a [f32],
    pub d: &'a [f32],
    pub out_proj: MatvecLaunch<'a>,
    pub d_inner: usize,
    pub d_state: usize,
    pub dt_rank: usize,
}

impl<'a> Mamba1Mixer<'a> {
    /// `None` when a shape disagrees with the others.
    ///
    /// # Safety
    ///
    /// The two state allocations (`ptr`, byte length) must be page-aligned
    /// and whole pages, hold `(d_conv - 1) * d_inner` and `d_inner *
    /// d_state` floats, stay valid, and be touched by nothing else until
    /// the launch this mixer is handed to returns.
    pub unsafe fn new(
        w: Mamba1Weights<'a>,
        conv_state: (*mut f32, usize),
        ssm_state: (*mut f32, usize),
    ) -> Option<Self> {
        let cols = |m: &MatvecLaunch<'_>| m.row_bytes / m.block_bytes * m.block_elems;
        let ok = w.d_conv >= 2
            && w.in_proj.rows == 2 * w.d_inner
            && w.conv1d.len() == w.d_conv * w.d_inner
            && w.conv_bias.len() == w.d_inner
            && w.x_proj.rows == w.dt_rank + 2 * w.d_state
            && cols(&w.x_proj) == w.d_inner
            && w.dt_proj.rows == w.d_inner
            && cols(&w.dt_proj) == w.dt_rank
            && w.dt_bias.len() == w.d_inner
            && w.a.len() == w.d_inner * w.d_state
            && w.d.len() == w.d_inner
            && cols(&w.out_proj) == w.d_inner
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
        self.w.out_proj.rows
    }

    /// The scratch widths this block needs from the stack: `xz` in the
    /// Q buffer, `x_db` and `dt` in the K and V buffers, `xc` in the
    /// attention buffer and `y` in the gate buffer.
    pub(crate) fn widths(&self) -> (usize, usize, usize) {
        let w = &self.w;
        (
            2 * w.d_inner,
            (w.dt_rank + 2 * w.d_state).max(w.d_inner),
            w.d_inner,
        )
    }
}

/// The stack's scratch, as this block uses it.
pub(crate) struct Mamba1Bufs<'b> {
    pub x: &'b ProtocolObject<dyn MTLBuffer>,
    pub xz: &'b ProtocolObject<dyn MTLBuffer>,
    pub xc: &'b ProtocolObject<dyn MTLBuffer>,
    pub x_db: &'b ProtocolObject<dyn MTLBuffer>,
    pub dt: &'b ProtocolObject<dyn MTLBuffer>,
    pub y: &'b ProtocolObject<dyn MTLBuffer>,
    pub out: &'b ProtocolObject<dyn MTLBuffer>,
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

/// The block for one token, `out = mamba1(x)`, into the stack's encoder.
pub(crate) fn encode_mamba1(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut crate::mem_ranges::MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    m: &Mamba1Mixer<'_>,
    b: &Mamba1Bufs<'_>,
) -> Result<(), MetalError> {
    use crate::gpu::{encode_matvec, resident_f32_buffer, resident_weight_buffer};
    let w = &m.w;
    let n = w.d_inner;
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
    let in_w = resident_weight_buffer(device, w.in_proj.weights)?;
    let x_w = resident_weight_buffer(device, w.x_proj.weights)?;
    let dt_w = resident_weight_buffer(device, w.dt_proj.weights)?;
    let out_w = resident_weight_buffer(device, w.out_proj.weights)?;
    let taps = resident_f32_buffer(device, w.conv1d)?;
    let conv_b = resident_f32_buffer(device, w.conv_bias)?;
    let dt_b = resident_f32_buffer(device, w.dt_bias)?;
    let a = resident_f32_buffer(device, w.a)?;
    let d = resident_f32_buffer(device, w.d)?;
    // Compiled before the first dispatch of this block is encoded.
    let conv_pipe = ensure_pipeline(device, SSM_DECODE_KERNEL_SRC, "ssm_conv_bias_silu")?;
    let gate_pipe = ensure_pipeline(device, SSM_DECODE_KERNEL_SRC, "ssm_skip_gate")?;

    // :34 xz = in_proj(x).
    mrs.begin_op(encoder, &[b.x], &[b.xz]);
    encode_matvec(encoder, device, &w.in_proj, &in_w, b.x, b.xz)?;
    mrs.end_op(&[b.x], &[b.xz]);

    // :43-70 xc = silu(conv(x) + b), the window shifted in place.
    mrs.begin_op(encoder, &[b.xz, conv_state], &[b.xc, conv_state]);
    encoder.setComputePipelineState(&conv_pipe.0);
    // SAFETY: the buffers outlive the stack's command buffer; `dims` is
    // copied by `setBytes`.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(conv_state), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&taps.buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&conv_b.buffer), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(b.xz), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(b.xc), 0, 4);
        let mut dims: [u32; 2] = [n as u32, w.d_conv as u32];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 8, 5);
    }
    dispatch_1d(encoder, &conv_pipe.0, n);
    mrs.end_op(&[b.xz, conv_state], &[b.xc, conv_state]);

    // :74 x_db = x_proj(xc); :90 dt = dt_proj(x_db[..dt_rank]).
    mrs.begin_op(encoder, &[b.xc], &[b.x_db]);
    encode_matvec(encoder, device, &w.x_proj, &x_w, b.xc, b.x_db)?;
    mrs.end_op(&[b.xc], &[b.x_db]);
    mrs.begin_op(encoder, &[b.x_db], &[b.dt]);
    encode_matvec(encoder, device, &w.dt_proj, &dt_w, b.x_db, b.dt)?;
    mrs.end_op(&[b.x_db], &[b.dt]);

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
            bc_stride: w.dt_rank + 2 * w.d_state,
            dt: b.dt,
            dt_bias: Some(&dt_b.buffer),
            a: &a.buffer,
            per_state: true,
            y: b.y,
            rows: 1,
        },
    )?;
    mrs.end_op(&[b.xc, b.x_db, b.dt, ssm_state], &[b.y, ssm_state]);

    // :121-122 y = (y + xc * D) * silu(z).
    mrs.begin_op(encoder, &[b.y, b.xc, b.xz], &[b.y]);
    encoder.setComputePipelineState(&gate_pipe.0);
    // SAFETY: as the conv's.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(b.y), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b.xc), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(b.xz), n * 4, 2);
        encoder.setBuffer_offset_atIndex(Some(&d.buffer), 0, 3);
        let mut dims: [u32; 2] = [n as u32, 1];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 8, 4);
    }
    dispatch_1d(encoder, &gate_pipe.0, n);
    mrs.end_op(&[b.y, b.xc, b.xz], &[b.y]);

    // :124 out = out_proj(y).
    mrs.begin_op(encoder, &[b.y], &[b.out]);
    encode_matvec(encoder, device, &w.out_proj, &out_w, b.y, b.out)?;
    mrs.end_op(&[b.y], &[b.out]);
    Ok(())
}
