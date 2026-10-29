//! The HEAD of a gated delta-net layer on the device: the causal
//! convolution with its SiLU, the per-head l2 norms on Q and K, and the
//! two scalar gates.
//!
//! # Why
//!
//! `docs/plans/gdn-resident-state.md` prices a Bonsai decode token at
//! 192 command buffers, 71.6 ms of GPU and 29.9 ms of latency BEYOND
//! it, with the GPU work already faster than the reference's whole
//! token. The 0.15 ms a submission costs is the OS wake-up and every
//! way around it has been measured and lost, so the only lever is the
//! COUNT, and the count only falls to one per layer when the host has
//! nothing to do inside a layer.
//!
//! These three are what the host still does inside a recurrent layer
//! between the QKV projection and the recurrence. They are small --
//! elementwise, a short window, a 128-wide reduction -- and that is the
//! point: they are not worth a submission of their own and they are
//! exactly what makes a submission of their own necessary.
//!
//! Each is an `encode_` function so it composes into somebody else's
//! command buffer, with one `launch_gdn_head` that runs all three for
//! the test that pins them against `frink_models::gdn`'s host body.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};
use std::ptr::NonNull;

use crate::gpu::{ensure_pipeline, MetalError};

/// Shapes of one recurrent layer's head, so the three encoders take one
/// argument and cannot be given a conv width from one layer and a head
/// count from another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadShape {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub head_dim: usize,
    /// Taps in the causal convolution (`d_conv`, 4 on every real
    /// export).
    pub d_conv: usize,
}

impl HeadShape {
    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.head_dim
    }
    pub fn value_dim(&self) -> usize {
        self.n_v_heads * self.head_dim
    }
    /// The convolution runs over `[q | k | v]` as one vector, which is
    /// how the converter lays `conv1d` out.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// Rows of convolution history, `(d_conv - 1) * conv_dim`.
    pub fn conv_state_len(&self) -> usize {
        (self.d_conv - 1) * self.conv_dim()
    }
}

pub const GDN_HEAD_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// `frink_core::mamba2::conv_step` followed by SiLU, one thread per
// convolution channel.
//
// The shift that follows the read is safe inside the thread because a
// channel's history column is touched by NOBODY else: thread `c` reads
// `state[i * width + c]` for every `i` and writes the same column back.
// The whole window is held in registers between the two, so the write
// cannot race the read.
kernel void gdn_conv_silu(
    device float* state [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    uint c [[thread_position_in_grid]]
) {
    const uint width = dims.x;
    const uint d_conv = dims.y;
    if (c >= width) {
        return;
    }
    // ops.cpp:9598-9603: a float accumulator over the window in order,
    // the newest input last, which is the order the host keeps too.
    float window[8];
    for (uint i = 0; i + 1 < d_conv; ++i) {
        window[i] = state[i * width + c];
    }
    const float xc = x[c];
    device const float* t = taps + (size_t)c * d_conv;
    float acc = 0.0f;
    for (uint i = 0; i + 1 < d_conv; ++i) {
        acc += window[i] * t[i];
    }
    acc += xc * t[d_conv - 1];
    out[c] = acc / (1.0f + exp(-acc));

    // Drop the oldest row, append `x`.
    for (uint i = 0; i + 2 < d_conv; ++i) {
        state[i * width + c] = window[i + 1];
    }
    if (d_conv > 1) {
        state[(d_conv - 2) * width + c] = xc;
    }
}

// `frink_core::gdn::l2_normalize` over one head of Q or K: one
// threadgroup per head, one thread per element.
//
// The host sums in f64 and this cannot, so the reduction is done in
// pairs down the threadgroup, which is the arrangement whose error
// grows with log(n) rather than n. At head_dim 128 the two agree to
// better than the f32 the result is stored in.
kernel void gdn_l2_norm_heads(
    device float* x [[buffer(0)]],
    constant uint2& dims [[buffer(1)]],
    constant float& eps [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint s = dims.x;
    const uint n_heads = dims.y;
    if (tgid >= n_heads || tid >= s) {
        return;
    }
    device float* head = x + (size_t)tgid * s;
    threadgroup float partial[1024];
    const float v = head[tid];
    partial[tid] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = s >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = 1.0f / max(sqrt(partial[0]), eps);
    head[tid] = v * scale;
}

// The two per-head scalars: `sigmoid(beta)` and
// `softplus(alpha + dt_bias) * a`, one thread per value head.
kernel void gdn_gates(
    device const float* beta_in [[buffer(0)]],
    device const float* alpha_in [[buffer(1)]],
    device const float* dt_bias [[buffer(2)]],
    device const float* a [[buffer(3)]],
    device float* beta_out [[buffer(4)]],
    device float* g_out [[buffer(5)]],
    constant uint& n_v [[buffer(6)]],
    uint h [[thread_position_in_grid]]
) {
    if (h >= n_v) {
        return;
    }
    beta_out[h] = 1.0f / (1.0f + exp(-beta_in[h]));
    // `frink_core::mamba2::softplus`, threshold and all: above 20 the
    // host returns the argument, and a kernel that took the log anyway
    // would differ exactly where the argument is large.
    const float z = alpha_in[h] + dt_bias[h];
    const float sp = (z > 20.0f) ? z : log(1.0f + exp(z));
    g_out[h] = sp * a[h];
}
"#;

/// The causal convolution and its SiLU, in `encoder`.
pub fn encode_conv_silu(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    shape: HeadShape,
    state: &ProtocolObject<dyn MTLBuffer>,
    taps: &ProtocolObject<dyn MTLBuffer>,
    x: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), MetalError> {
    // The window is held in registers, so the tap count is bounded by
    // the array the kernel declares; every real export is 4.
    if shape.d_conv == 0 || shape.d_conv > 8 {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, GDN_HEAD_KERNEL_SRC, "gdn_conv_silu")?;
    encoder.setComputePipelineState(&pipe.0);
    let width = shape.conv_dim();
    unsafe {
        for (idx, buf) in [state, taps, x, out].into_iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buf), 0, idx);
        }
        let mut dims: [u32; 2] = [width as u32, shape.d_conv as u32];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 8, 4);
    }
    dispatch_1d(encoder, &pipe.0, width);
    Ok(())
}

/// The per-head l2 norm over `n_heads` heads of `head_dim`, in place.
pub fn encode_l2_norm_heads(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    offset_floats: usize,
    head_dim: usize,
    n_heads: usize,
    eps: f32,
) -> Result<(), MetalError> {
    // The reduction halves its stride from `head_dim`, so a width that
    // is not a power of two would drop the odd tail silently.
    if head_dim == 0 || !head_dim.is_power_of_two() || head_dim > 1024 {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, GDN_HEAD_KERNEL_SRC, "gdn_l2_norm_heads")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x), offset_floats * 4, 0);
        let mut dims: [u32; 2] = [head_dim as u32, n_heads as u32];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 8, 1);
        let mut e = eps;
        encoder.setBytes_length_atIndex(NonNull::new(&mut e as *mut f32 as *mut _).unwrap(), 4, 2);
    }
    crate::dispatch::dispatch_counted(
        encoder,
        MTLSize {
            width: n_heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: head_dim,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// The two per-head gates, in `encoder`.
#[allow(clippy::too_many_arguments)]
pub fn encode_gates(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n_v_heads: usize,
    beta_in: &ProtocolObject<dyn MTLBuffer>,
    alpha_in: &ProtocolObject<dyn MTLBuffer>,
    dt_bias: &ProtocolObject<dyn MTLBuffer>,
    a: &ProtocolObject<dyn MTLBuffer>,
    beta_out: &ProtocolObject<dyn MTLBuffer>,
    g_out: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, GDN_HEAD_KERNEL_SRC, "gdn_gates")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        for (idx, buf) in [beta_in, alpha_in, dt_bias, a, beta_out, g_out]
            .into_iter()
            .enumerate()
        {
            encoder.setBuffer_offset_atIndex(Some(buf), 0, idx);
        }
        let mut n = n_v_heads as u32;
        encoder.setBytes_length_atIndex(NonNull::new(&mut n as *mut u32 as *mut _).unwrap(), 4, 6);
    }
    dispatch_1d(encoder, &pipe.0, n_v_heads);
    Ok(())
}

fn dispatch_1d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipe: &Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,
    n: usize,
) {
    use objc2_metal::MTLComputePipelineState;
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

/// One row's whole head -- gates, convolution with SiLU, the two l2
/// norms -- end to end, for the test that pins it against the host.
///
/// Returns `(q, k, v, g, beta)` exactly as
/// `frink_models::gdn::Gdn::conv_and_gates_for_rows` does for one row.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn launch_gdn_head(
    shape: HeadShape,
    conv_state: &mut [f32],
    taps: &[f32],
    qkv: &[f32],
    beta_in: &[f32],
    alpha_in: &[f32],
    dt_bias: &[f32],
    a: &[f32],
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), MetalError> {
    let (key_dim, value_dim, conv_dim) = (shape.key_dim(), shape.value_dim(), shape.conv_dim());
    if conv_state.len() != shape.conv_state_len()
        || taps.len() != conv_dim * shape.d_conv
        || qkv.len() != conv_dim
        || beta_in.len() != shape.n_v_heads
        || alpha_in.len() != shape.n_v_heads
        || dt_bias.len() != shape.n_v_heads
        || a.len() != shape.n_v_heads
    {
        return Err(MetalError::CommandFailed);
    }
    let shared = crate::gpu::shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;
    let upload = |xs: &[f32]| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let mut owned = xs.to_vec();
        // SAFETY: `owned` is live across the call and the buffer copies
        // its bytes.
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(owned.as_mut_ptr() as *mut _).ok_or(MetalError::BufferAllocFailed)?,
                std::mem::size_of_val(owned.as_slice()),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocFailed)
    };
    let state_buf = upload(conv_state)?;
    let (taps_buf, qkv_buf) = (upload(taps)?, upload(qkv)?);
    let (beta_buf, alpha_buf) = (upload(beta_in)?, upload(alpha_in)?);
    let (dt_buf, a_buf) = (upload(dt_bias)?, upload(a)?);
    let mk = |n: usize| {
        device
            .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocFailed)
    };
    let conv_out = mk(conv_dim)?;
    let (beta_out, g_out) = (mk(shape.n_v_heads)?, mk(shape.n_v_heads)?);

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_gates(
        &encoder,
        device,
        shape.n_v_heads,
        &beta_buf,
        &alpha_buf,
        &dt_buf,
        &a_buf,
        &beta_out,
        &g_out,
    )?;
    encode_conv_silu(
        &encoder, device, shape, &state_buf, &taps_buf, &qkv_buf, &conv_out,
    )?;
    // Q and K only: `v` is the convolution's output as it stands.
    encode_l2_norm_heads(
        &encoder,
        device,
        &conv_out,
        0,
        shape.head_dim,
        shape.n_k_heads,
        eps,
    )?;
    encode_l2_norm_heads(
        &encoder,
        device,
        &conv_out,
        key_dim,
        shape.head_dim,
        shape.n_k_heads,
        eps,
    )?;
    encoder.endEncoding();
    let clock = crate::timing::SubmitClock::start();
    crate::timing::commit_wait_note(&cmd_buf, "gdn-head", 64, clock);

    // SAFETY: every buffer below is shared storage of exactly the length
    // read, written by kernels this call has waited for.
    unsafe {
        let conv = std::slice::from_raw_parts(conv_out.contents().as_ptr() as *const f32, conv_dim);
        let st = std::slice::from_raw_parts(
            state_buf.contents().as_ptr() as *const f32,
            conv_state.len(),
        );
        conv_state.copy_from_slice(st);
        let beta =
            std::slice::from_raw_parts(beta_out.contents().as_ptr() as *const f32, shape.n_v_heads);
        let g =
            std::slice::from_raw_parts(g_out.contents().as_ptr() as *const f32, shape.n_v_heads);
        Ok((
            conv[..key_dim].to_vec(),
            conv[key_dim..2 * key_dim].to_vec(),
            conv[2 * key_dim..2 * key_dim + value_dim].to_vec(),
            g.to_vec(),
            beta.to_vec(),
        ))
    }
}
