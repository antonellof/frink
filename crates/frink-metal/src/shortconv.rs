//! LFM2's short convolution as a device op, so a conv layer can sit in
//! the dense decode stack (`crate::decode_dense`) beside the attention
//! layers instead of sending every token's whole model to the host.
//!
//! The op is `frink_models::shortconv` (`lfm-base.cpp:9-112`): `bcx =
//! in_proj(x)` split in three, `bx = b * x`, a causal depthwise conv of
//! width `l_cache` over the previous `l_cache - 1` inputs and `bx`, and
//! `y = c * conv`, which `out_proj` then projects. The two projections
//! are the stack's own matvecs; this file is the middle.
//!
//! # Where the window lives
//!
//! The HOST stays the authority for the conv history: it is the layer's
//! `KvCache`, which every prefix, rollback and paging path already
//! knows how to carry. Each token the caller copies the last
//! `l_cache - 1` inputs into [`ConvWindows`], the stack appends `bx` as
//! the window's last row, and after the wait the caller pushes that row
//! to the host cache. Two rows of `n_embd` floats each way per conv
//! layer, against a device mirror that every one of those paths would
//! have had to keep in step.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLResourceOptions,
    MTLSize,
};

use crate::gpu::{ensure_pipeline, MetalError};

pub const SHORTCONV_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One token of the short convolution, one thread per channel.
//
// `window` is `[l][n]`: rows 0 .. l-2 are the previous inputs, oldest
// first, and row l-1 is where this token's `bx` is written. `bcx` is
// `in_proj`'s output `[b | c | x]`, `taps` the kernel `[n][l]` with the
// newest input on tap l-1 (ops.cpp:9601-9603), `y` the gated output.
kernel void shortconv_step(
    device float* window [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device const float* bcx [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    uint ch [[thread_position_in_grid]]
) {
    const uint n = dims.x;
    const uint l = dims.y;
    if (ch >= n) {
        return;
    }
    const float bx = bcx[ch] * bcx[2u * n + ch];
    window[(l - 1u) * n + ch] = bx;
    device const float* t = taps + (size_t)ch * l;
    // A float accumulator over the window in order, the newest input
    // last, as the host and ggml_ssm_conv sum it.
    float acc = 0.0f;
    for (uint i = 0; i + 1u < l; ++i) {
        acc += window[i * n + ch] * t[i];
    }
    acc += bx * t[l - 1u];
    y[ch] = bcx[n + ch] * acc;
}
"#;

/// The widest conv the kernel serves. Every LFM export is 3; the bound
/// only keeps a hostile `l_cache` from becoming a huge window buffer.
pub const MAX_L_CACHE: usize = 16;

/// Every conv layer's window for one token, in ONE shared buffer:
/// window `i` is `[l_cache][n]` floats at `i * l_cache * n`.
pub struct ConvWindows {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    n: usize,
    l_cache: usize,
    count: usize,
}

// SAFETY: one shared-mode MTLBuffer, read and written by the host only
// between launches (every launch that binds it waits before returning)
// and by the GPU only inside them; the same justification as
// `MetalKvBuffers`.
unsafe impl Send for ConvWindows {}
unsafe impl Sync for ConvWindows {}

impl ConvWindows {
    /// `count` windows of `l_cache` rows of `n` floats, zeroed.
    pub fn new(count: usize, n: usize, l_cache: usize) -> Result<Self, MetalError> {
        if count == 0 || n == 0 || !(2..=MAX_L_CACHE).contains(&l_cache) {
            return Err(MetalError::Unsupported("short conv window shape"));
        }
        let shared = crate::gpu::shared_metal()?;
        let floats = count * l_cache * n;
        let buf = shared
            .device
            .newBufferWithLength_options(floats * 4, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocFailed)?;
        let mut w = Self {
            buf,
            n,
            l_cache,
            count,
        };
        w.all_mut().fill(0.0);
        Ok(w)
    }

    /// Whether this holds `count` windows of this shape.
    pub fn fits(&self, count: usize, n: usize, l_cache: usize) -> bool {
        (self.count, self.n, self.l_cache) == (count, n, l_cache)
    }

    fn all(&self) -> &[f32] {
        // SAFETY: shared storage of exactly `count * l_cache * n` floats
        // this value owns. The GPU touches it only inside a launch, and
        // every launch that binds it waits before returning, so no
        // command buffer is writing it while the host reads.
        unsafe {
            std::slice::from_raw_parts(
                self.buf.contents().as_ptr() as *const f32,
                self.count * self.l_cache * self.n,
            )
        }
    }

    fn all_mut(&mut self) -> &mut [f32] {
        // SAFETY: as `all`, and `&mut self` makes this the only view.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buf.contents().as_ptr() as *mut f32,
                self.count * self.l_cache * self.n,
            )
        }
    }

    /// Window `i`'s history rows, `[l_cache - 1][n]`, oldest first, for
    /// the host to fill before a launch.
    pub fn history_mut(&mut self, i: usize) -> &mut [f32] {
        assert!(i < self.count);
        let (n, l) = (self.n, self.l_cache);
        &mut self.all_mut()[i * l * n..(i * l + l - 1) * n]
    }

    /// Window `i`'s newest row, the `bx` the last launch appended.
    pub fn newest(&self, i: usize) -> &[f32] {
        assert!(i < self.count);
        let (n, l) = (self.n, self.l_cache);
        &self.all()[(i * l + l - 1) * n..(i + 1) * l * n]
    }

    pub(crate) fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buf
    }

    pub(crate) fn offset_bytes(&self, i: usize) -> usize {
        assert!(i < self.count);
        i * self.l_cache * self.n * 4
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn l_cache(&self) -> usize {
        self.l_cache
    }
}

/// `y = c * conv(window, b * x)` for one token, appending `b * x` to
/// window `i`. `taps` is the resident `[n][l_cache]` kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_shortconv_step(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    windows: &ConvWindows,
    i: usize,
    taps: &ProtocolObject<dyn MTLBuffer>,
    bcx: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, SHORTCONV_KERNEL_SRC, "shortconv_step")?;
    encoder.setComputePipelineState(&pipe.0);
    let n = windows.n;
    // SAFETY: every buffer outlives the encoder's command buffer (the
    // caller waits on it before any of them is dropped), and `dims` is
    // copied into the command stream by `setBytes`.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(windows.buffer()), windows.offset_bytes(i), 0);
        encoder.setBuffer_offset_atIndex(Some(taps), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(bcx), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(y), 0, 3);
        let mut dims: [u32; 2] = [n as u32, windows.l_cache as u32];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 8, 4);
    }
    let tg = pipe.0.maxTotalThreadsPerThreadgroup().clamp(1, 256);
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

    /// The host's arithmetic (`frink_models::shortconv::ShortConv::
    /// forward_rows` between its projections), spelled out: the scalar
    /// twin the kernel is checked against.
    fn host_step(history: &[f32], taps: &[f32], bcx: &[f32], n: usize, l: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; n];
        for ch in 0..n {
            let bx = bcx[ch] * bcx[2 * n + ch];
            let t = &taps[ch * l..(ch + 1) * l];
            let mut acc = 0.0f32;
            for i in 0..l - 1 {
                acc += history[i * n + ch] * t[i];
            }
            acc += bx * t[l - 1];
            y[ch] = bcx[n + ch] * acc;
        }
        y
    }

    /// One step against the host twin, on the second of two windows so
    /// the offset is exercised, with a history that is not zero and taps
    /// that are not symmetric (an off-by-one in the window or the tap
    /// order changes the answer).
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn one_step_matches_the_host_and_appends_bx() {
        let (n, l) = (300usize, 3usize);
        let val = |i: usize, s: f32| ((i * 37 % 101) as f32 / 50.0 - 1.0) * s;
        let taps: Vec<f32> = (0..n * l).map(|i| val(i, 0.7) + (i % l) as f32).collect();
        let bcx: Vec<f32> = (0..3 * n).map(|i| val(i + 11, 1.3)).collect();
        let history: Vec<f32> = (0..(l - 1) * n).map(|i| val(i + 5, 2.0)).collect();
        let mut windows = ConvWindows::new(2, n, l).expect("windows");
        windows.history_mut(1).copy_from_slice(&history);

        let shared = crate::gpu::shared_metal().expect("metal");
        let device = &shared.device;
        // Fresh buffers, not the resident cache: that is keyed on the
        // slice's address, which a later test's vector can reuse.
        let upload = |v: &[f32]| {
            let b = device
                .newBufferWithLength_options(v.len() * 4, MTLResourceOptions::StorageModeShared)
                .expect("buffer");
            // SAFETY: shared storage of exactly `v.len()` floats, not yet
            // bound to any command buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    v.as_ptr(),
                    b.contents().as_ptr() as *mut f32,
                    v.len(),
                )
            };
            b
        };
        let (taps_b, bcx_b) = (upload(&taps), upload(&bcx));
        let y_b = device
            .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
            .expect("y");
        let cmd = shared.queue.commandBuffer().expect("cb");
        let enc = cmd.computeCommandEncoder().expect("enc");
        encode_shortconv_step(&enc, device, &windows, 1, &taps_b, &bcx_b, &y_b).expect("encode");
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        // SAFETY: shared storage of `n` floats written by the waited
        // command buffer.
        let got = unsafe { std::slice::from_raw_parts(y_b.contents().as_ptr() as *const f32, n) };
        let want = host_step(&history, &taps, &bcx, n, l);
        for (ch, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                "channel {ch}: {g} vs {w}"
            );
        }
        let bx: Vec<f32> = (0..n).map(|c| bcx[c] * bcx[2 * n + c]).collect();
        assert_eq!(windows.newest(1), &bx[..], "bx is the window's last row");
        assert!(
            windows.newest(0).iter().all(|&v| v == 0.0),
            "the other window is untouched"
        );
    }
}
