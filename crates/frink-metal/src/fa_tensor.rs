//! The Metal 4 tensor-API arm of prefill attention (llama
//! `kernel_flash_attn_ext_tensor`, `kernels/fa_tensor.metal`).
//!
//! On an M5 the simdgroup-MMA `flash_attn_ext` port in `attn.rs` does not
//! reach the per-core matrix units; llama's tensor kernel does, through
//! two `mpp::tensor_ops::matmul2d` ops per KV block (Q·Kᵀ into
//! threadgroup scores, P·V into a cooperative O). This is that kernel for
//! frink's buffers: Q `[n_q, n_heads, D]` f32, K/V `[pos, n_kv_heads, D]`
//! f16, out `[n_q, n_heads, D]` f32.
//!
//! Two things differ from llama's host side, neither in the arithmetic:
//!
//! - frink has no mask tensor. Causality comes from positions: query
//!   `i` sees keys `< kv_prefix_len + i + 1`, set to `-INF` exactly where
//!   llama's mask adds `-INF`. KV blocks past the tile's last visible key
//!   are skipped, which is what llama's `blk` prepass marks 0 for a
//!   causal mask, so no prepass and no mask buffer.
//! - llama's KV view is padded to a multiple of `C` (its cache is padded
//!   to 256) and the host asserts `ne11 % C == 0`. frink's is not: the
//!   last block's K/V tensors carry the real row count as their extent,
//!   so `matmul2d` never reads past the cache, and the rows it does not
//!   read sit at keys every real query masks anyway.
//!
//! Selection ([`selected`]) is llama's `ggml_metal_op_flash_attn_ext_use_tensor`
//! for an f16 cache.

use crate::dispatch::dispatch_counted;
use crate::gpu::{ensure_pipeline, MetalError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};
use std::ptr::NonNull;
use std::sync::LazyLock;

/// Queries per threadgroup (llama `OP_FLASH_ATTN_EXT_TENSOR_NQPSG`).
const NQPTG: u32 = 32;
/// KV rows per block (llama `OP_FLASH_ATTN_EXT_TENSOR_NCPSG`).
const NCPSG: u32 = 64;
/// Simdgroups per threadgroup (llama `OP_FLASH_ATTN_EXT_TENSOR_NSG`).
const NSG: u32 = 8;

/// The head widths frink instantiates. llama's list also has 192/128,
/// 512 and 576/512; frink's prefill caches have `DK == DV` and its FA
/// paths stop at 256, so those widths stay on the paths they use today.
pub(crate) fn supported(head_dim: u32) -> bool {
    matches!(head_dim, 64 | 128 | 256)
}

/// llama `ggml_metal_op_flash_attn_ext_use_tensor` for frink's shapes:
/// the device runs the tensor API, the batch is not a vec batch
/// (`ne01 < 20`), the width is instantiated, and the work clears llama's
/// small-work cutoff (`ceil(n_q / 32) * n_heads * D >= 8192`; below it
/// the half8x8 kernel, frink's `fa_ext`, is faster).
///
/// `kv_dequantized` is a cache that reaches attention through the f16
/// scratch. llama only dequantizes a quantized cache up front when the
/// batch is at least 32 queries (`ggml_metal_op_flash_attn_ext_use_kv_f16`),
/// and below that it runs the half8x8 kernel on the quantized cache, so
/// the tensor arm needs `n_q >= 32` there.
pub(crate) fn selected(head_dim: u32, n_heads: u32, n_q: u32, kv_dequantized: bool) -> bool {
    shape_selected(head_dim, n_heads, n_q, kv_dequantized) && crate::tensor_mm::has_tensor()
}

/// [`selected`] without the device probe, so the thresholds are tested
/// on every host.
fn shape_selected(head_dim: u32, n_heads: u32, n_q: u32, kv_dequantized: bool) -> bool {
    if n_q < 20 || !supported(head_dim) || (kv_dequantized && n_q < 32) {
        return false;
    }
    let work = u64::from(n_q.div_ceil(NQPTG)) * u64::from(n_heads) * u64::from(head_dim);
    work >= 8192
}

/// Threadgroup bytes (llama: queries f16, scores f32, probabilities f16,
/// row scale f32, rescale flag i32, padded to 16).
fn smem(head_dim: u32) -> usize {
    let q = NQPTG as usize;
    let c = NCPSG as usize;
    let raw = q * head_dim as usize * 2 + q * c * (4 + 2) + q * 4 + 4;
    raw.div_ceil(16) * 16
}

fn pipeline_name(head_dim: u32) -> Result<&'static str, MetalError> {
    Ok(match head_dim {
        64 => "gqa_prefill_fa_tensor_d64",
        128 => "gqa_prefill_fa_tensor_d128",
        256 => "gqa_prefill_fa_tensor_d256",
        _ => return Err(MetalError::CommandFailed),
    })
}

static SRC: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}#include <metal_stdlib>\n{}{BODY}",
        crate::tensor_mm::TENSOR_DEFINE,
        crate::tensor_mm::HEADERS
    )
});

/// Compile the pipeline for `head_dim` ahead of the first prefill.
pub(crate) fn warm(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    head_dim: u32,
) -> Result<&'static str, MetalError> {
    let name = pipeline_name(head_dim)?;
    ensure_pipeline(device, SRC.as_str(), name)?;
    Ok(name)
}

/// Encode causal prefill attention through the tensor kernel. The caller
/// has checked [`crate::tensor_mm::has_tensor`]; an unsupported width is
/// an error, never a silent fallback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    v: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_q: u32,
    kv_prefix_len: u32,
    softcap: f32,
) -> Result<(), MetalError> {
    if !crate::tensor_mm::has_tensor() || n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(MetalError::CommandFailed);
    }
    let name = pipeline_name(head_dim)?;
    let pipe = ensure_pipeline(device, SRC.as_str(), name)?;
    // llama: `scale = 1/sqrt(D)` from the graph, then `scale /= softcap`
    // on the host when a softcap is set.
    let mut scale = 1.0f32 / (head_dim as f32).sqrt();
    if softcap > 0.0 {
        scale /= softcap;
    }
    encoder.setComputePipelineState(&pipe.0);
    let mut args: [u32; 4] = [n_heads, n_kv_heads, n_q, kv_prefix_len];
    let mut fargs: [f32; 2] = [scale, softcap];
    // SAFETY: the bound buffers outlive the command buffer (the callers
    // hold them across commit/wait or in the stack scratch), and the
    // `setBytes` sources are locals copied by Metal before the call returns.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        encoder.setBytes_length_atIndex(
            NonNull::new(args.as_mut_ptr() as *mut _).unwrap(),
            std::mem::size_of_val(&args),
            4,
        );
        encoder.setBytes_length_atIndex(
            NonNull::new(fargs.as_mut_ptr() as *mut _).unwrap(),
            std::mem::size_of_val(&fargs),
            5,
        );
        encoder.setThreadgroupMemoryLength_atIndex(smem(head_dim), 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_q.div_ceil(NQPTG) as usize,
            height: n_heads as usize,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: NSG as usize,
            depth: 1,
        },
    );
    Ok(())
}

/// llama `kernel_flash_attn_ext_tensor<half, half4, D, D>` with mask =
/// causal-from-positions, no sinks, no ALiBi. The host splices the
/// tensor define and headers in front.
const BODY: &str = r#"
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// args: n_heads, n_kv_heads, n_q, kv_prefix_len; fargs: scale, softcap.
template<short DK, short DV, short Q, short C, short NSG>
kernel void gqa_prefill_fa_tensor(
        device const float * q    [[buffer(0)]],
        device const half  * k    [[buffer(1)]],
        device const half  * v    [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        constant uint4     & args [[buffer(4)]],
        constant float2    & fargs [[buffer(5)]],
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = 32;
    constexpr short NT = NW*NSG;
    constexpr short NQ = Q/NSG;
    constexpr short NC = C/NW; // columns per thread

    static_assert(DK % 4 == 0,  "DK must be divisible by 4");
    static_assert(Q % NSG == 0, "Q must be divisible by NSG");
    static_assert(C % NW == 0,  "C must be divisible by NW");

    const int n_heads    = int(args[0]);
    const int n_kv_heads = int(args[1]);
    const int n_q        = int(args[2]);
    const int kv_prefix  = int(args[3]);
    const float scale    = fargs[0];
    const float softcap  = fargs[1];
    const bool has_scap  = softcap != 0.0f;

    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    const short tiitg = sgitg*NW + tiisg;

    threadgroup half  * sq = (threadgroup half  *) shmem;       // [Q, DK] queries
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK); // [Q, C]  scores
    threadgroup half  * sp = (threadgroup half  *) (ss + Q*C);  // [Q, C]  probabilities
    threadgroup float * sr = (threadgroup float *) (sp + Q*C);  // [Q]     per-row scale of O
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);    // [1]     last iteration (ic0 + 1) that rescaled O

    const int q_row = n_heads*DK; // floats between consecutive queries
    q += (int64_t) iq1*q_row + iq2*DK;

    {
        const int ikv2 = iq2/(n_heads/n_kv_heads);

        k += ikv2*DK;
        v += ikv2*DV;
    }

    // with softcap the scale is small (scale/softcap), so it is applied to the scores to keep the precision of Q
    const float qscale = has_scap ? 1.0f : scale;

    // load the queries, with the scale folded in
    for (int i = tiitg; i < Q*DK/4; i += NT) {
        const int j = i/(DK/4);

        float4 q4 = 0.0f;
        if (iq1 + j < n_q) {
            q4 = ((device const float4 *) (q + (int64_t) j*q_row))[i%(DK/4)];
        }

        ((threadgroup half4 *) sq)[i] = (half4) (q4*qscale);
    }

    float M[NQ];
    float S[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        M[jj] = -FLT_MAX/2;
        S[jj] = 0.0f;
    }

    const int sk = n_kv_heads*DK;
    const int sv = n_kv_heads*DV;

    // Rows of K/V that exist, and the first key no query of this tile sees
    // (llama's blk prepass marks every block from there on 0).
    const int kv_valid = kv_prefix + n_q;
    const int kv_end   = kv_prefix + min(iq1 + Q, n_q);
    // The first key that the tile's first query does not see: blocks
    // wholly below it need no mask (llama's blk = 2).
    const int kv_full  = kv_prefix + iq1 + 1;

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C,  Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C,  Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto tv0 = tensor((device half *) v, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

    // the O matrix from the paper
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg == 0) {
        sf[0] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ic0 = 0, ic = 0; ic < kv_end; ++ic0, ic += C) {
        // the last block reads only the rows that exist; the rest are keys
        // past kv_valid, which every query masks
        const int nc = min(int(C), kv_valid - ic);
        const bool need_mask = ic + C > kv_full;

        // Q*K^T
        {
            auto tk = tensor((device half *) (k + (int64_t) ic*sk), dextents<int32_t, 2>(DK, nc), array<int, 2>({1, sk}));

            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            float s[NC];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                s[ii] = ss[j*C + ii*NW + tiisg];
            }

            if (has_scap) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] = softcap*precise::tanh(s[ii]*scale);
                }
            }

            // causal mask: query iq1 + j sees keys < kv_prefix + iq1 + j + 1.
            // The kv_valid clamp only binds on the padding rows past n_q,
            // so the columns the last block did not read are -INF there too.
            if (need_mask) {
                const int clen = min(kv_prefix + iq1 + j + 1, kv_valid);
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    if (ic + ii*NW + tiisg >= clen) {
                        s[ii] = -INFINITY;
                    }
                }
            }

            float m = M[jj];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }

            m = simd_max(m);

            // lazy rescaling: move the running max only when it grows by more than 8 (e^8 fits in half)
            float ms = 1.0f;

            if (m > M[jj] + 8.0f) {
                ms    = exp(M[jj] - m);
                M[jj] = m;

                if (tiisg == 0) {
                    sf[0] = ic0 + 1;
                }
            }

            float sum = 0.0f;

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                // the sum uses the same rounded values as P*V
                const half p = (half) exp(s[ii] - M[jj]);

                sp[j*C + ii*NW + tiisg] = p;

                sum += (float) p;
            }

            S[jj] = S[jj]*ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms)*O + P*V
        if (sf[0] == ic0 + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        {
            auto tv = tensor((device half *) (v + (int64_t) ic*sv), dextents<int32_t, 2>(DV, nc), array<int, 2>({1, sv}));

            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        if (tiisg == 0) {
            sr[j] = S[jj] == 0.0f ? 0.0f : 1.0f/S[jj];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
    }

    // store to global memory - rows past n_q are clipped by the tensor extents
    device float * pdst = dst + ((int64_t) iq1*n_heads + iq2)*DV;

    auto td = tensor(pdst, dextents<int32_t, 2>(DV, n_q - iq1), array<int, 2>({1, n_heads*DV}));

    co.store(td);
}

typedef decltype(gqa_prefill_fa_tensor<64, 64, 32, 64, 8>) gqa_prefill_fa_tensor_t;

template [[host_name("gqa_prefill_fa_tensor_d64")]]  kernel gqa_prefill_fa_tensor_t gqa_prefill_fa_tensor<64,  64,  32, 64, 8>;
template [[host_name("gqa_prefill_fa_tensor_d128")]] kernel gqa_prefill_fa_tensor_t gqa_prefill_fa_tensor<128, 128, 32, 64, 8>;
template [[host_name("gqa_prefill_fa_tensor_d256")]] kernel gqa_prefill_fa_tensor_t gqa_prefill_fa_tensor<256, 256, 32, 64, 8>;
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attn::{
        alloc_f32_buffer, launch_gqa_prefill_host_kernel, upload_f16_from_f32, upload_f32,
        PrefillAttnKernel,
    };
    use crate::gpu::shared_metal;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

    /// llama's thresholds: vec batch below 20, small-work cutoff at 8192,
    /// a dequantized cache only from 32 queries, the instantiated widths.
    #[test]
    fn selection_follows_llama_thresholds() {
        // ceil(n_q / 32) * n_heads * D against 8192.
        assert!(!shape_selected(128, 32, 20, false)); // 1 * 32 * 128 = 4096
        assert!(shape_selected(128, 64, 20, false)); // 1 * 64 * 128 = 8192
        assert!(!shape_selected(128, 64, 19, false)); // a vec batch
        assert!(shape_selected(128, 32, 33, false)); // 2 * 32 * 128 = 8192
        assert!(!shape_selected(64, 32, 65, false)); // 3 * 32 * 64 = 6144
        assert!(shape_selected(256, 32, 20, false)); // 1 * 32 * 256 = 8192
                                                     // A cache that went through the f16 scratch needs 32 queries.
        assert!(!shape_selected(128, 64, 31, true));
        assert!(shape_selected(128, 64, 32, true));
        // Widths the kernel is not instantiated at.
        assert!(!shape_selected(96, 128, 512, false));
        assert!(!shape_selected(512, 128, 512, false));
    }

    struct Case {
        n_heads: usize,
        n_kv_heads: usize,
        d: usize,
        n_q: usize,
        prefix: usize,
        softcap: Option<f32>,
        /// Scale keys up with position, so the running max moves by more
        /// than 8 between blocks and the lazy O rescale runs.
        ramp: bool,
    }

    const fn c(
        n_heads: usize,
        n_kv_heads: usize,
        d: usize,
        n_q: usize,
        prefix: usize,
        softcap: Option<f32>,
        ramp: bool,
    ) -> Case {
        Case {
            n_heads,
            n_kv_heads,
            d,
            n_q,
            prefix,
            softcap,
            ramp,
        }
    }

    /// n_q off the 32-query tile, kv_prefix_len > 0, caches off the
    /// 64-row block and exactly on it, GQA ratios 1/2/4/8, softcap.
    const CASES: &[Case] = &[
        c(8, 2, 64, 40, 9, None, false),
        c(4, 4, 64, 64, 0, Some(30.0), false),
        c(4, 2, 64, 64, 64, None, true),
        c(8, 8, 128, 33, 0, None, false),
        c(16, 2, 128, 100, 70, Some(50.0), true),
        c(4, 2, 128, 300, 200, None, true),
        c(4, 1, 256, 40, 9, Some(50.0), false),
        c(8, 4, 256, 65, 130, None, true),
        c(8, 1, 256, 1, 0, None, false),
    ];

    fn inputs(t: &Case) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let total = t.prefix + t.n_q;
        let row = t.n_kv_heads * t.d;
        let q = (0..t.n_q * t.n_heads * t.d)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k = (0..total * row)
            .map(|i| {
                let ramp = if t.ramp {
                    1.0 + (i / row) as f32 / 16.0
                } else {
                    1.0
                };
                (i as f32 * 0.03).cos() * ramp
            })
            .collect();
        let v = (0..total * row).map(|i| (i as f32 * 0.05).sin()).collect();
        (q, k, v)
    }

    /// Causal GQA in f64 over the f16-rounded K/V the kernel reads.
    fn cpu(t: &Case, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let r = |x: f32| f64::from(half::f16::from_f32(x).to_f32());
        let scale = 1.0 / (t.d as f64).sqrt();
        let group = t.n_heads / t.n_kv_heads;
        let mut out = vec![0f32; t.n_q * t.n_heads * t.d];
        for i in 0..t.n_q {
            let clen = t.prefix + i + 1;
            for h in 0..t.n_heads {
                let kh = h / group;
                let qh = &q[(i * t.n_heads + h) * t.d..][..t.d];
                let mut s: Vec<f64> = (0..clen)
                    .map(|p| {
                        let kp = &k[(p * t.n_kv_heads + kh) * t.d..][..t.d];
                        let dot: f64 = qh.iter().zip(kp).map(|(a, b)| f64::from(*a) * r(*b)).sum();
                        let x = dot * scale;
                        match t.softcap {
                            Some(c) => f64::from(c) * (x / f64::from(c)).tanh(),
                            None => x,
                        }
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for x in s.iter_mut() {
                    *x = (*x - m).exp();
                    sum += *x;
                }
                for dd in 0..t.d {
                    let acc: f64 = (0..clen)
                        .map(|p| s[p] * r(v[(p * t.n_kv_heads + kh) * t.d + dd]))
                        .sum();
                    out[(i * t.n_heads + h) * t.d + dd] = (acc / sum) as f32;
                }
            }
        }
        out
    }

    /// The tensor kernel with `poison` NaN rows after the cache: a read
    /// past `kv_prefix_len + n_q` turns the output NaN.
    fn run_tensor(t: &Case, q: &[f32], k: &[f32], v: &[f32], poison: usize) -> Vec<f32> {
        let shared = shared_metal().expect("metal device");
        let device = &shared.device;
        let pad = vec![f32::NAN; poison * t.n_kv_heads * t.d];
        let kp: Vec<f32> = k.iter().chain(&pad).copied().collect();
        let vp: Vec<f32> = v.iter().chain(&pad).copied().collect();
        let qb = upload_f32(device, q).unwrap();
        let kb = upload_f16_from_f32(device, &kp).unwrap();
        let vb = upload_f16_from_f32(device, &vp).unwrap();
        let n_out = t.n_q * t.n_heads * t.d;
        let ob = alloc_f32_buffer(device, n_out).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        let res = encode(
            &enc,
            device,
            &qb,
            &kb,
            &vb,
            &ob,
            t.n_heads as u32,
            t.n_kv_heads as u32,
            t.d as u32,
            t.n_q as u32,
            t.prefix as u32,
            t.softcap.unwrap_or(0.0),
        );
        enc.endEncoding();
        res.expect("tensor prefill");
        cmd.commit();
        cmd.waitUntilCompleted();
        // SAFETY: `ob` holds `n_out` f32 and the GPU finished writing it.
        unsafe { std::slice::from_raw_parts(ob.contents().as_ptr() as *const f32, n_out) }.to_vec()
    }

    fn run_named(t: &Case, q: &[f32], k: &[f32], v: &[f32], kernel: PrefillAttnKernel) -> Vec<f32> {
        launch_gqa_prefill_host_kernel(
            q,
            k,
            v,
            t.n_heads,
            t.n_kv_heads,
            t.d,
            t.n_q,
            t.prefix,
            t.softcap,
            kernel,
        )
        .expect("prefill")
    }

    /// Largest `|a - b| / max(|a|, 1)`; NaN counts as the worst.
    fn max_rel_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
        let mut worst = (0f32, 0usize);
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs() / x.abs().max(1.0);
            if d.is_nan() || d > worst.0 {
                worst = (if d.is_nan() { f32::INFINITY } else { d }, i);
            }
        }
        worst
    }

    fn tensor_device() -> bool {
        std::env::set_var("FRINK_METAL_FA_VEC", "1");
        crate::tensor_mm::has_tensor()
    }

    /// The tensor kernel against an f64 causal reference. Q and P round
    /// to half on the way into the matrix units, as llama's do, so the
    /// tolerance is half precision.
    #[test]
    #[ignore = "needs a Metal 4 tensor GPU"]
    fn tensor_prefill_matches_cpu() {
        if !tensor_device() {
            return;
        }
        for t in CASES {
            let (q, k, v) = inputs(t);
            let want = cpu(t, &q, &k, &v);
            for poison in [0, 64] {
                let got = run_tensor(t, &q, &k, &v, poison);
                let (d, i) = max_rel_diff(&want, &got);
                assert!(
                    d <= 4e-3,
                    "d={} heads={}/{} n_q={} pre={} sc={:?} poison={poison}: \
                     max rel diff {d} at {i} (cpu {} gpu {})",
                    t.d,
                    t.n_heads,
                    t.n_kv_heads,
                    t.n_q,
                    t.prefix,
                    t.softcap,
                    want[i],
                    got[i]
                );
            }
        }
    }

    /// The tensor kernel against the half8x8 `fa_ext` port it replaces on
    /// M5, through the named-kernel entry the parity tests use.
    #[test]
    #[ignore = "needs a Metal 4 tensor GPU"]
    fn tensor_prefill_matches_fa_ext() {
        if !tensor_device() {
            return;
        }
        for t in CASES.iter().filter(|t| t.n_q >= 8) {
            let (q, k, v) = inputs(t);
            let ext = run_named(t, &q, &k, &v, PrefillAttnKernel::FaExt);
            let ten = run_named(t, &q, &k, &v, PrefillAttnKernel::Tensor);
            let (d, i) = max_rel_diff(&ext, &ten);
            assert!(
                d <= 4e-3,
                "d={} n_q={} pre={}: tensor vs fa_ext {d} at {i} ({} vs {})",
                t.d,
                t.n_q,
                t.prefix,
                ext[i],
                ten[i]
            );
        }
    }

    /// `Auto` sends a shape llama gives the tensor kernel to it, and one
    /// under the small-work cutoff to `fa_ext`, bit for bit.
    #[test]
    #[ignore = "needs a Metal 4 tensor GPU"]
    fn auto_takes_the_tensor_kernel_where_llama_does() {
        if !tensor_device() {
            return;
        }
        let big = c(64, 8, 128, 40, 5, None, false);
        assert!(selected(128, 64, 40, false));
        let (q, k, v) = inputs(&big);
        let auto = run_named(&big, &q, &k, &v, PrefillAttnKernel::Auto);
        let ten = run_named(&big, &q, &k, &v, PrefillAttnKernel::Tensor);
        let ext = run_named(&big, &q, &k, &v, PrefillAttnKernel::FaExt);
        assert_eq!(auto, ten, "Auto did not take the tensor kernel");
        assert_ne!(
            auto, ext,
            "tensor and fa_ext agree bit for bit; the test cannot tell them apart"
        );

        let small = c(8, 2, 128, 40, 5, None, false);
        assert!(!selected(128, 8, 40, false));
        let (q, k, v) = inputs(&small);
        let auto = run_named(&small, &q, &k, &v, PrefillAttnKernel::Auto);
        let ext = run_named(&small, &q, &k, &v, PrefillAttnKernel::FaExt);
        assert_eq!(auto, ext, "Auto left fa_ext under llama's cutoff");
    }
}
