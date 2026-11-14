//! Serialized per-dispatch GPU cost of the decode stack's small kernels,
//! measured the way llama.cpp's `test-backend-ops perf` measures its own
//! under `GGML_METAL_CONCURRENCY_DISABLE=1`: N copies of one dispatch in
//! ONE encoder, a barrier between each, command-buffer GPU time over N.
//!
//! `crate::kernel_timing` attributes a real token's GPU time to kinds
//! but has to put every kind in its own encoder, so its small-kernel
//! rows carry an encoder boundary each. This is the number without it,
//! and the one to put beside a llama.cpp `us/run`. Hardware only.

use crate::attn::{encode_gqa_for_bench, encode_rope, MetalRope, MetalRopeLayout, RopeTarget};
use crate::elem::{encode_gelu_mul, encode_vec_add};
use crate::gpu::{
    compute_encoder_concurrent, memory_barrier_buffers, shared_metal, MetalError, SharedMetal,
};
use crate::norm::{encode_add_rms_norm, encode_rms_norm};
use crate::timing::{commit_wait_note, SubmitClock};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLResourceOptions,
};
use std::sync::Arc;

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

fn f32_buf(device: &ProtocolObject<dyn MTLDevice>, n: usize) -> Result<Buf, MetalError> {
    device
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)
}

fn f16_buf(device: &ProtocolObject<dyn MTLDevice>, n: usize) -> Result<Buf, MetalError> {
    device
        .newBufferWithLength_options(n * 2, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)
}

/// GPU microseconds per dispatch of `encode`, serialized `n` deep: the
/// MINIMUM over `REPEATS` command buffers, because this host is never
/// quiet and interference only ever adds time.
const REPEATS: usize = 7;

fn serialized_us(
    shared: &Arc<SharedMetal>,
    n: usize,
    mut encode: impl FnMut(
        &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    ) -> Result<(), MetalError>,
) -> Result<f64, MetalError> {
    let mut best = f64::INFINITY;
    for _ in 0..REPEATS {
        best = best.min(serialized_us_once(shared, n, &mut encode)?);
    }
    Ok(best)
}

fn serialized_us_once(
    shared: &Arc<SharedMetal>,
    n: usize,
    mut encode: impl FnMut(
        &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    ) -> Result<(), MetalError>,
) -> Result<f64, MetalError> {
    // Warm the pipeline outside the timed buffer.
    {
        let cmd_buf = shared
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandFailed)?;
        let encoder = compute_encoder_concurrent(&cmd_buf)?;
        encode(&encoder)?;
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }
    let clock = SubmitClock::start();
    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = compute_encoder_concurrent(&cmd_buf)?;
    for i in 0..n {
        if i > 0 {
            memory_barrier_buffers(&encoder);
        }
        encode(&encoder)?;
    }
    encoder.endEncoding();
    commit_wait_note(&cmd_buf, "kernel-bench", u64::MAX, clock);
    let gpu_s = cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime();
    Ok(gpu_s * 1e6 / n as f64)
}

/// Prints the serialized cost of each decode-stack small kernel at
/// Gemma-2-2B's shapes (hidden 2304, 8 q heads and 4 kv heads of 256,
/// ffn 9216) and Llama-3.2-3B's RoPE shape (24 + 8 heads of 128).
///
/// Run: `cargo test -p frink-metal --features metal -- --ignored
/// --nocapture serialized_small_kernel_costs`.
#[test]
#[ignore = "needs a real Metal GPU; prints a table"]
fn serialized_small_kernel_costs() {
    let shared = shared_metal().expect("Metal device");
    let device = &shared.device;
    const N: usize = 256;
    let hidden = 2304usize;
    let ffn = 9216usize;
    let (n_heads, n_kv, d) = (8u32, 4u32, 256u32);
    let kv_len = 96usize;

    let h = f32_buf(device, hidden).unwrap();
    let o = f32_buf(device, hidden).unwrap();
    let w = f32_buf(device, hidden).unwrap();
    let x = f32_buf(device, hidden).unwrap();
    let gate = f32_buf(device, ffn).unwrap();
    let up = f32_buf(device, ffn).unwrap();
    let act = f32_buf(device, ffn).unwrap();
    let q = f32_buf(device, (n_heads * d) as usize).unwrap();
    let k = f32_buf(device, (n_kv * d) as usize).unwrap();
    let kc = f16_buf(device, kv_len * (n_kv * d) as usize).unwrap();
    let vc = f16_buf(device, kv_len * (n_kv * d) as usize).unwrap();
    let attn = f32_buf(device, (n_heads * d) as usize).unwrap();
    let q3b = f32_buf(device, 24 * 128).unwrap();
    let k3b = f32_buf(device, 8 * 128).unwrap();

    let rope = MetalRope::new(MetalRopeLayout::Neox);
    let rope_norm = MetalRope::new(MetalRopeLayout::Norm);
    let mut rows: Vec<(&str, f64)> = Vec::new();
    let mut row = |name: &'static str, r: Result<f64, MetalError>| {
        rows.push((name, r.expect(name)));
    };

    row(
        "vec_add 2304",
        serialized_us(&shared, N, |e| {
            encode_vec_add(e, device, &h, &o, hidden as u32)
        }),
    );
    row(
        "rms_norm 2304",
        serialized_us(&shared, N, |e| {
            encode_rms_norm(e, device, &h, &w, &x, hidden as u32, 1e-6)
        }),
    );
    row(
        "add_rms_norm 2304",
        serialized_us(&shared, N, |e| {
            encode_add_rms_norm(e, device, &h, &o, &w, &x, hidden as u32, 1e-6)
        }),
    );
    row(
        "gelu_mul 9216",
        serialized_us(&shared, N, |e| {
            encode_gelu_mul(e, device, &gate, &up, &act, ffn as u32)
        }),
    );
    row(
        "rope neox 8+4 heads d256",
        serialized_us(&shared, N, |e| {
            encode_rope(
                e,
                device,
                rope,
                RopeTarget { vecs: &q, n_heads },
                Some(RopeTarget {
                    vecs: &k,
                    n_heads: n_kv,
                }),
                d,
                10000.0,
                50,
                None,
            )
        }),
    );
    row(
        "rope norm 24+8 heads d128",
        serialized_us(&shared, N, |e| {
            encode_rope(
                e,
                device,
                rope_norm,
                RopeTarget {
                    vecs: &q3b,
                    n_heads: 24,
                },
                Some(RopeTarget {
                    vecs: &k3b,
                    n_heads: 8,
                }),
                128,
                500000.0,
                50,
                None,
            )
        }),
    );
    row(
        "gqa d256 8/4 heads kv96 softcap",
        serialized_us(&shared, N, |e| {
            encode_gqa_for_bench(
                e,
                device,
                &q,
                &kc,
                &vc,
                &attn,
                n_heads,
                n_kv,
                d,
                kv_len as u32,
                0,
                Some(50.0),
            )
        }),
    );
    row(
        "gqa d256 8/4 heads kv96 no softcap",
        serialized_us(&shared, N, |e| {
            encode_gqa_for_bench(
                e,
                device,
                &q,
                &kc,
                &vc,
                &attn,
                n_heads,
                n_kv,
                d,
                kv_len as u32,
                0,
                None,
            )
        }),
    );

    eprintln!("serialized GPU us per dispatch, {N} deep, barrier between each, min of {REPEATS}:");
    for (name, us) in &rows {
        eprintln!("  {name:<36} {us:>8.2} us");
    }
}

/// The prefill GEMM's rate on Bonsai-2-27B's FFN shape (`ffn_up`,
/// 17408 x 5120, PTQ1_0) at 512 tokens, beside Q8_0 and Q4_0 on the
/// same shape, with f32 and f16 activations: the number to put beside
/// `test-backend-ops perf -o MUL_MAT`.
///
/// Run: `cargo test -p frink-metal --features metal --release --
/// --ignored --nocapture prefill_gemm_rates`.
#[test]
#[ignore = "needs a real Metal GPU; prints a table"]
fn prefill_gemm_rates() {
    use crate::gpu::{encode_mul_mm_sg, encode_mul_mm_sg_f16, resident_weight_buffer};
    let shared = shared_metal().expect("Metal");
    let device = &shared.device;
    let (rows, cols, batch) = (17408usize, 5120usize, 512usize);
    let x = f32_buf(device, batch * cols).unwrap();
    let xh = f16_buf(device, batch * cols).unwrap();
    let out = f32_buf(device, batch * rows).unwrap();
    let out2 = f32_buf(device, batch * rows).unwrap();
    const N: usize = 8;
    let mut seed = 99u32;
    // SAFETY: shared buffers of exactly these lengths, no GPU work yet.
    unsafe {
        let xs = std::slice::from_raw_parts_mut(x.contents().as_ptr() as *mut f32, batch * cols);
        let hs = std::slice::from_raw_parts_mut(xh.contents().as_ptr() as *mut u16, batch * cols);
        for (a, h) in xs.iter_mut().zip(hs.iter_mut()) {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *a = ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5;
            *h = half::f16::from_f32(*a).to_bits();
        }
    }
    // (kind, entry, block bytes, block elements, scale offset in a block)
    for (name, fn_name, block_bytes, block_elems, scale_at) in [
        ("ptq1_0", "ptq1_0_mul_mm_sg", 28usize, 128usize, 26usize),
        ("q8_0", "q8_0_mul_mm_sg", 34, 32, 0),
        ("q4_0", "q4_0_mul_mm_sg", 18, 32, 0),
    ] {
        let row_bytes = cols / block_elems * block_bytes;
        let mut w = vec![0u8; rows * row_bytes];
        for (i, b) in w.iter_mut().enumerate() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let at = i % block_bytes;
            // Every block scale a small finite half (0x2000).
            *b = if at == scale_at {
                0
            } else if at == scale_at + 1 {
                0x20
            } else {
                (seed >> 24) as u8
            };
        }
        let l = crate::gpu::MulMmSgLaunch {
            weights: &w,
            rows,
            row_bytes,
            fn_name,
            block_bytes,
            block_elems,
        };
        let wb = resident_weight_buffer(device, &w).unwrap();
        let flops = 2.0 * (rows * cols * batch) as f64;
        let f32_us = serialized_us(&shared, N, |e| {
            encode_mul_mm_sg(e, device, &l, &wb, &x, &out, batch)
        })
        .unwrap();
        let f16_us = serialized_us(&shared, N, |e| {
            encode_mul_mm_sg_f16(e, device, &l, &wb, &xh, &out2, batch)
        })
        .unwrap();
        // SAFETY: both written by command buffers that were waited for.
        let (a, b) = unsafe {
            (
                std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, batch * rows),
                std::slice::from_raw_parts(out2.contents().as_ptr() as *const f32, batch * rows),
            )
        };
        let same = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() == y.to_bits())
            .count();
        let maxd = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        eprintln!(
            "  {name}: f32 vs f16-rounded activations: {same}/{} bit-identical, max |d| {maxd}",
            a.len()
        );
        eprintln!(
            "  {name:<8} f32 act {f32_us:>8.1} us {:>6.1} TFLOP/s   f16 act {f16_us:>8.1} us {:>6.1} TFLOP/s",
            flops / f32_us / 1e6,
            flops / f16_us / 1e6
        );
    }
}
