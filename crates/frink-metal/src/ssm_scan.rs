//! The Mamba selective scan over a batch of rows, on the device:
//! `ggml_ssm_scan` for both generations (`frink_core::mamba2::scan_rows`
//! is the host body, and the twin this is tested against).
//!
//! # Why a batch
//!
//! A prompt's scan is the one part of a Mamba layer's prefill that is
//! neither a GEMM nor cheap: every row reads and writes the whole
//! `[n_head][head_dim][d_state]` state, which is 2 MiB a layer on
//! Granite-4.0-H. On the host that is most of a prompt even spread over
//! every core. Here it is one dispatch per layer, the state held in
//! registers across the rows, the way `kernel_ssm_scan_f32`
//! (`ggml-metal/kernels/ssm.metal:167-296`) holds it: one threadgroup
//! per `(head, channel)`, one thread per state element, a SIMD sum per
//! row for `y`. A decode token (one row) stays on the host, where the
//! submission would cost more than the scan.
//!
//! The state is the host's own memory, wrapped in place
//! (`crate::gdn::buffer_no_copy`), so the host stays its only owner.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLSize,
};

use crate::gpu::{ensure_pipeline, shared_metal, MetalError};

pub const SSM_SCAN_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// `rows` tokens of the scan. Threadgroup `tg` is state row `tg` of
// `[n_head][head_dim]` (head `tg / head_dim`); thread `k` is its state
// element `k`. Row r's `x` is at `r * x_stride` of `x`, its `[B | C]`
// at `r * bc_stride` of `bc` (one buffer at two offsets for a prompt's
// conv output, two buffers for a decode token's). `dt` is
// `[rows][n_head]` before softplus, plus `dt_bias` when flag bit 1 is
// set; `a` is `[n_head]` (Mamba-2) or `[n_head][d_state]` (Mamba-1,
// flag bit 0).
kernel void ssm_scan_rows(
    device float* state [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* bc [[buffer(2)]],
    device const float* dt [[buffer(3)]],
    device const float* a [[buffer(4)]],
    device float* y [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    constant uint4& batch [[buffer(7)]],
    device const float* dt_bias [[buffer(8)]],
    threadgroup float* partial [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint k [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint n_sg [[simdgroups_per_threadgroup]]
) {
    const uint n_head = dims.x;
    const uint head_dim = dims.y;
    const uint d_state = dims.z;
    const uint n_group = dims.w;
    const uint x_stride = batch.x;
    const uint bc_stride = batch.y;
    const uint rows = batch.z;
    const bool per_state = (batch.w & 1u) != 0u;
    const bool has_dt_bias = (batch.w & 2u) != 0u;
    const uint h = tg / head_dim;
    const uint g = h / (n_head / n_group);
    const uint n_x = n_head * head_dim;
    const uint n_bc = n_group * d_state;
    const uint b_at = g * d_state + k;
    const uint c_at = n_bc + g * d_state + k;
    float s = state[(size_t)tg * d_state + k];
    const float av = per_state ? a[h * d_state + k] : a[h];
    const float bias = has_dt_bias ? dt_bias[h] : 0.0f;
    for (uint r = 0; r < rows; ++r) {
        device const float* bcr = bc + (size_t)r * bc_stride;
        float dt0 = dt[(size_t)r * n_head + h];
        if (has_dt_bias) {
            dt0 = dt0 + bias;
        }
        // `frink_core::mamba2::softplus`, threshold and all.
        const float dt_sp = dt0 > 20.0f ? dt0 : log(1.0f + exp(dt0));
        const float x_dt = x[(size_t)r * x_stride + tg] * dt_sp;
        s = s * exp(dt_sp * av) + bcr[b_at] * x_dt;
        const float p = simd_sum(s * bcr[c_at]);
        if (n_sg == 1u) {
            if (lane == 0u) {
                y[(size_t)r * n_x + tg] = p;
            }
        } else {
            if (lane == 0u) {
                partial[sg] = p;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k == 0u) {
                float sum = 0.0f;
                for (uint i = 0; i < n_sg; ++i) {
                    sum += partial[i];
                }
                y[(size_t)r * n_x + tg] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    state[(size_t)tg * d_state + k] = s;
}
"#;

/// The scan's geometry (`frink_core::mamba2::ScanDims`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanShape {
    pub n_head: usize,
    pub head_dim: usize,
    pub d_state: usize,
    pub n_group: usize,
}

impl ScanShape {
    /// Whether the kernel serves this shape: one thread per state
    /// element in a threadgroup, and whole groups.
    pub fn is_supported(self) -> bool {
        self.n_head > 0
            && self.head_dim > 0
            && (1..=1024).contains(&self.d_state)
            && self.n_group > 0
            && self.n_head.is_multiple_of(self.n_group)
    }

    fn state_len(self) -> usize {
        self.n_head * self.head_dim * self.d_state
    }
}

/// `rows` tokens of the scan on the device, `state` advanced in place,
/// `y` (`[rows][n_head][head_dim]`) written.
///
/// # Safety
///
/// `state_ptr` must point at `state_bytes` page-aligned, readable and
/// writable bytes holding at least the shape's state, with
/// `state_bytes` a whole number of pages, and nothing else may touch
/// them until this returns (it waits for the GPU before returning).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scan_rows(
    shape: ScanShape,
    state_ptr: *mut f32,
    state_bytes: usize,
    xbc: &[f32],
    stride: usize,
    dt: &[f32],
    rows: usize,
    a: &[f32],
    per_state: bool,
    y: &mut [f32],
) -> Result<(), MetalError> {
    let n_x = shape.n_head * shape.head_dim;
    let n_bc = shape.n_group * shape.d_state;
    let want_a = if per_state {
        shape.n_head * shape.d_state
    } else {
        shape.n_head
    };
    if !shape.is_supported()
        || rows == 0
        || stride < n_x + 2 * n_bc
        || xbc.len() < (rows - 1) * stride + n_x + 2 * n_bc
        || dt.len() != rows * shape.n_head
        || a.len() != want_a
        || y.len() != rows * n_x
        || state_bytes < shape.state_len() * 4
    {
        return Err(MetalError::Unsupported("ssm scan shape"));
    }
    let shared = shared_metal()?;
    let device = &shared.device;
    // SAFETY: the caller's contract, forwarded.
    let state = unsafe { crate::gdn::buffer_no_copy(device, state_ptr, state_bytes) }
        .ok_or(MetalError::BufferAllocFailed)?;
    let sc = crate::scratch_pool::Scratch::take(device, &[xbc.len(), dt.len(), a.len(), y.len()])
        .ok_or(MetalError::BufferAllocFailed)?;
    let xbc_b = sc.write(0, xbc).ok_or(MetalError::CommandFailed)?;
    let dt_b = sc.write(1, dt).ok_or(MetalError::CommandFailed)?;
    let a_b = sc.write(2, a).ok_or(MetalError::CommandFailed)?;
    let y_b = sc.buf(3);
    // Compiled before an encoder exists, so a failure leaves none open.
    ensure_pipeline(device, SSM_SCAN_KERNEL_SRC, "ssm_scan_rows")?;
    let cmd = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_scan(
        &enc,
        device,
        shape,
        ScanOperands {
            state: &state,
            x: (xbc_b, 0),
            bc: (xbc_b, n_x * 4),
            x_stride: stride,
            bc_stride: stride,
            dt: dt_b,
            dt_bias: None,
            a: a_b,
            per_state,
            y: y_b,
            rows,
        },
    )?;
    enc.endEncoding();
    let clock = crate::timing::SubmitClock::start();
    crate::timing::commit_wait_note(&cmd, "ssm-scan", 32, clock);
    // SAFETY: shared storage of exactly `y.len()` floats, written by the
    // command buffer waited for above.
    unsafe {
        std::ptr::copy_nonoverlapping(
            y_b.contents().as_ptr() as *const f32,
            y.as_mut_ptr(),
            y.len(),
        )
    };
    Ok(())
}

/// Where one scan's operands are: buffers and byte offsets, for
/// [`encode_scan`].
pub(crate) struct ScanOperands<'b> {
    pub state: &'b ProtocolObject<dyn MTLBuffer>,
    pub x: (&'b ProtocolObject<dyn MTLBuffer>, usize),
    pub bc: (&'b ProtocolObject<dyn MTLBuffer>, usize),
    pub x_stride: usize,
    pub bc_stride: usize,
    pub dt: &'b ProtocolObject<dyn MTLBuffer>,
    /// Added to `dt` before the softplus, when the caller has not.
    pub dt_bias: Option<&'b ProtocolObject<dyn MTLBuffer>>,
    pub a: &'b ProtocolObject<dyn MTLBuffer>,
    pub per_state: bool,
    pub y: &'b ProtocolObject<dyn MTLBuffer>,
    pub rows: usize,
}

/// The scan, encoded into the caller's command buffer: a prompt's
/// batch here ([`launch_scan_rows`]), one decode token in the dense
/// stack (`crate::ssm_decode`).
pub(crate) fn encode_scan(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    shape: ScanShape,
    o: ScanOperands<'_>,
) -> Result<(), MetalError> {
    if !shape.is_supported() || o.rows == 0 {
        return Err(MetalError::Unsupported("ssm scan shape"));
    }
    let pipe = ensure_pipeline(device, SSM_SCAN_KERNEL_SRC, "ssm_scan_rows")?;
    enc.setComputePipelineState(&pipe.0);
    let n_sg = shape.d_state.div_ceil(32);
    let flags = o.per_state as u32 | (u32::from(o.dt_bias.is_some()) << 1);
    // SAFETY: every bound buffer outlives the command buffer (the caller
    // waits on it before dropping any of them); the uniforms are copied
    // by `setBytes`.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(o.state), 0, 0);
        enc.setBuffer_offset_atIndex(Some(o.x.0), o.x.1, 1);
        enc.setBuffer_offset_atIndex(Some(o.bc.0), o.bc.1, 2);
        enc.setBuffer_offset_atIndex(Some(o.dt), 0, 3);
        enc.setBuffer_offset_atIndex(Some(o.a), 0, 4);
        enc.setBuffer_offset_atIndex(Some(o.y), 0, 5);
        let mut dims: [u32; 4] = [
            shape.n_head as u32,
            shape.head_dim as u32,
            shape.d_state as u32,
            shape.n_group as u32,
        ];
        enc.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 16, 6);
        let mut batch: [u32; 4] = [o.x_stride as u32, o.bc_stride as u32, o.rows as u32, flags];
        enc.setBytes_length_atIndex(NonNull::new(batch.as_mut_ptr() as *mut _).unwrap(), 16, 7);
        // A buffer must be bound at every index the kernel names; with no
        // bias the flag keeps it unread.
        enc.setBuffer_offset_atIndex(Some(o.dt_bias.unwrap_or(o.dt)), 0, 8);
        enc.setThreadgroupMemoryLength_atIndex(n_sg.max(1) * 4, 0);
    }
    crate::dispatch::dispatch_counted(
        enc,
        MTLSize {
            width: shape.n_head * shape.head_dim,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: shape.d_state,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// A fresh, page-aligned allocation of `len` floats for the tests: the
/// shape `frink_core::recurrent_state::AlignedF32` has, which this crate
/// cannot name.
#[cfg(test)]
pub(crate) struct PageBuf {
    ptr: *mut f32,
    len: usize,
    layout: std::alloc::Layout,
}

#[cfg(test)]
impl PageBuf {
    pub(crate) fn new(data: &[f32]) -> Self {
        const PAGE: usize = 16384;
        let bytes = (data.len() * 4).max(1).div_ceil(PAGE) * PAGE;
        let layout = std::alloc::Layout::from_size_align(bytes, PAGE).unwrap();
        // SAFETY: a non-zero layout; the memory is initialised below.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut f32;
        assert!(!ptr.is_null());
        // SAFETY: `ptr` holds at least `data.len()` floats.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
        Self {
            ptr,
            len: data.len(),
            layout,
        }
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        // SAFETY: `len` initialised floats this value owns.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[cfg(test)]
impl Drop for PageBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: &mut u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((*seed >> 9) as f32 / (1u32 << 23) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The host scan, written out: `frink_core::mamba2::scan_rows`'s
    /// arithmetic with a plain running sum, which the kernel's SIMD sum
    /// differs from only in rounding.
    #[allow(clippy::too_many_arguments)]
    fn host_scan(
        s: ScanShape,
        state: &mut [f32],
        xbc: &[f32],
        stride: usize,
        dt: &[f32],
        rows: usize,
        a: &[f32],
        per_state: bool,
    ) -> Vec<f32> {
        let n_x = s.n_head * s.head_dim;
        let n_bc = s.n_group * s.d_state;
        let mut y = vec![0.0f32; rows * n_x];
        for r in 0..rows {
            let row = &xbc[r * stride..];
            for t in 0..n_x {
                let h = t / s.head_dim;
                let g = h / (s.n_head / s.n_group);
                let d0 = dt[r * s.n_head + h];
                let sp = if d0 > 20.0 { d0 } else { (1.0 + d0.exp()).ln() };
                let x_dt = row[t] * sp;
                let mut sum = 0.0f32;
                for k in 0..s.d_state {
                    let av = if per_state {
                        a[h * s.d_state + k]
                    } else {
                        a[h]
                    };
                    let v = state[t * s.d_state + k] * (sp * av).exp()
                        + row[n_x + g * s.d_state + k] * x_dt;
                    sum += v * row[n_x + n_bc + g * s.d_state + k];
                    state[t * s.d_state + k] = v;
                }
                y[r * n_x + t] = sum;
            }
        }
        y
    }

    /// Both generations against the host twin: Mamba-2 with groups and
    /// a state too wide for one SIMD group, Mamba-1 with one channel a
    /// head and a per-state decay, each over several rows from a state
    /// that is not zero.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn scan_rows_matches_the_host() {
        for (shape, per_state) in [
            (
                ScanShape {
                    n_head: 4,
                    head_dim: 3,
                    d_state: 96,
                    n_group: 2,
                },
                false,
            ),
            (
                ScanShape {
                    n_head: 40,
                    head_dim: 1,
                    d_state: 16,
                    n_group: 1,
                },
                true,
            ),
        ] {
            let mut seed = 3u32;
            let rows = 7;
            let n_x = shape.n_head * shape.head_dim;
            let stride = n_x + 2 * shape.n_group * shape.d_state + 5;
            let xbc = rnd(&mut seed, rows * stride);
            let dt = rnd(&mut seed, rows * shape.n_head);
            let a_len = if per_state {
                shape.n_head * shape.d_state
            } else {
                shape.n_head
            };
            let a: Vec<f32> = rnd(&mut seed, a_len).iter().map(|v| -v.abs()).collect();
            let init = rnd(&mut seed, shape.state_len());
            let mut host_state = init.clone();
            let want = host_scan(
                shape,
                &mut host_state,
                &xbc,
                stride,
                &dt,
                rows,
                &a,
                per_state,
            );
            let dev_state = PageBuf::new(&init);
            let mut got = vec![0.0f32; rows * n_x];
            // SAFETY: a page-aligned allocation this test owns, of whole
            // pages, untouched until the launch returns.
            unsafe {
                launch_scan_rows(
                    shape,
                    dev_state.ptr,
                    dev_state.layout.size(),
                    &xbc,
                    stride,
                    &dt,
                    rows,
                    &a,
                    per_state,
                    &mut got,
                )
            }
            .expect("launch");
            let close = |g: f32, w: f32| (g - w).abs() <= 1e-4 * w.abs().max(1.0);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(close(*g, *w), "{shape:?} y[{i}]: {g} vs {w}");
            }
            for (i, (g, w)) in dev_state.as_slice().iter().zip(&host_state).enumerate() {
                assert!(close(*g, *w), "{shape:?} state[{i}]: {g} vs {w}");
            }
        }
    }
}
