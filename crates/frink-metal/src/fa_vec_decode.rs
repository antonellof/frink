//! The FA-vec decode attention kernels: one query row against an f16
//! KV cache, one threadgroup per head, NSG simdgroups each walking
//! every NSG-th tile of 32 keys with an online softmax, merged at the
//! end. Specialised per head width (64 / 96 / 128 / 256) because the
//! lane-to-slice mapping differs.
//!
//! Split out of `attn.rs` along the seam the Gemma-2 decode work
//! touches next: per token, these kernels cost a Gemma-2-2B layer
//! ~3x what llama.cpp's `flash_attn_ext_vec` does at the same shape.

use crate::dispatch::dispatch_counted;
use crate::gpu::{ensure_pipeline, MetalError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};
use std::ptr::NonNull;

/// llama.cpp-style FA-vec decode for **head_dim=64**, f16 KV, NE=2, C=32.
/// Same tile/merge structure as the d=128 kernel, but D4=16 float4
/// slices only cover half a simdgroup — so each warp processes **two**
/// KV positions per pass (half-warp `ty=0` gets even `cc`, `ty=1` odd),
/// with a 16-lane shuffle-xor dot reduce and a cross-half xor-16 merge
/// of the V accumulators. This keeps all 32 lanes busy where a naive
/// D4=16 port would idle half the warp (TinyLlama / Llama-3.2-1B are
/// d=64).
const GQA_DECODE_FA_VEC_D64_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_fa_vec_d64(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    device const float* sinks [[buffer(10)]],
    constant uint& has_sinks [[buffer(11)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Specialized for D=64 (host only dispatches when head_dim==64).
    constexpr uint D = 64u;
    constexpr uint D4 = 16u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    // Per-SG floats: C scores + D output.
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;
    const uint tx = tiisg % D4; // float4 slice of the head
    const uint ty = tiisg / D4; // 0/1: token parity within a warp pass

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    if (tiisg < D4) {
        so4[tiisg] = float4(0.0f);
    }
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    // Each SG walks KV tiles: ic0 = sgitg, sgitg+nsg, ...
    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        // Q·K, two positions per warp pass: half-warp `ty` owns token
        // ic+cc; 16-lane xor reduce leaves the full dot in every lane
        // of that half; lane tx==0 publishes it.
        for (uint cc = ty; cc < chunk; cc += 2u) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float p = dot(sq4[tx], float4(k4[tx]));
            p += simd_shuffle_xor(p, 8u);
            p += simd_shuffle_xor(p, 4u);
            p += simd_shuffle_xor(p, 2u);
            p += simd_shuffle_xor(p, 1u);
            if (tx == 0u) {
                float sc = p * scale;
                if (softcap > 0.0f) {
                    sc = softcap * tanh(sc / softcap);
                }
                ss[cc] = sc;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax over this tile (one score per lane).
        float s_lane = (tiisg < chunk) ? ss[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        if (tiisg < D4) {
            so4[tiisg] *= ms;
        }
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O += P · V, two positions per warp pass; merge the two token
        // halves with an xor-16 shuffle before accumulating.
        float4 lo = float4(0.0f);
        for (uint cc = ty; cc < chunk; cc += 2u) {
            device const half4* v4 =
                (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            lo += float4(v4[tx]) * ss[cc];
        }
        lo += simd_shuffle_xor(lo, 16u);
        if (ty == 0u) {
            so4[tx] += lo;
        }
    }

    // Publish S,M for cross-SG reduce (reuse ss[0], ss[1]).
    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG online-softmax merge.
    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            if (tiisg < D4) {
                so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u && tiisg < D4) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        // gpt-oss's attention sink: one more logit in the denominator
        // with no value, so it only rescales (llama `fa_vec_common`).
        float S_f = ss0[0];
        float a_f = 1.0f;
        if (has_sinks != 0u) {
            const float s_k = sinks[h];
            const float M_f = max(ss0[1], s_k);
            a_f = (ss0[1] == -INFINITY) ? 0.0f : exp(ss0[1] - M_f);
            S_f = S_f * a_f + exp(s_k - M_f);
        }
        float inv = (S_f == 0.0f) ? 0.0f : (a_f / S_f);
        device float4* out4 = (device float4*)(out + h * D);
        out4[tiisg] = so0[tiisg] * inv;
    }
}
"#;

/// FA-vec decode for **head_dim=96** (Phi-3-mini). D4=24 float4 slices —
/// lanes `tiisg < 24` own Q/K/V float4 work; remaining warp lanes contribute
/// zeros to the simd_sum score reduce so the tile/merge structure stays
/// identical to the d=128 kernel.
const GQA_DECODE_FA_VEC_D96_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_fa_vec_d96(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    device const float* sinks [[buffer(10)]],
    constant uint& has_sinks [[buffer(11)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = 96u;
    constexpr uint D4 = 24u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    if (tiisg < D4) {
        so4[tiisg] = float4(0.0f);
    }
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = (tiisg < D4) ? dot(sq4[tiisg], float4(k4[tiisg])) : 0.0f;
            float sc = simd_sum(partial) * scale;
            if (softcap > 0.0f) {
                sc = softcap * tanh(sc / softcap);
            }
            scores[cc] = sc;
        }

        float s_lane = (tiisg < chunk) ? scores[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        if (tiisg < D4) {
            so4[tiisg] *= ms;
        }
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo = float4(0.0f);
        if (tiisg < D4) {
            for (uint cc = 0; cc < chunk; cc++) {
                device const half4* v4 =
                    (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
                lo += float4(v4[tiisg]) * ss[cc];
            }
            so4[tiisg] += lo;
        }
    }

    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            if (tiisg < D4) {
                so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u && tiisg < D4) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        // gpt-oss's attention sink: one more logit in the denominator
        // with no value, so it only rescales (llama `fa_vec_common`).
        float S_f = ss0[0];
        float a_f = 1.0f;
        if (has_sinks != 0u) {
            const float s_k = sinks[h];
            const float M_f = max(ss0[1], s_k);
            a_f = (ss0[1] == -INFINITY) ? 0.0f : exp(ss0[1] - M_f);
            S_f = S_f * a_f + exp(s_k - M_f);
        }
        float inv = (S_f == 0.0f) ? 0.0f : (a_f / S_f);
        device float4* out4 = (device float4*)(out + h * D);
        out4[tiisg] = so0[tiisg] * inv;
    }
}
"#;

/// FA-vec decode for the two head widths a lane can own whole float4
/// slices of: **128** (one slice per lane) and **256** (two). f16 KV,
/// NE=1 (one key per simdgroup at a time, `simd_sum` per key), C=32
/// keys per tile. One TG per head; NSG simdgroups each own every
/// NSG-th tile, then an online-softmax merge across simdgroups.
///
/// The tile loops are COMPILE-TIME. The previous kernels walked
/// `cc < chunk` with `chunk` a runtime value, and kept the tile's
/// scores in a register array indexed by that runtime `cc`; an Apple
/// GPU then issues each key's loads only after the previous key's
/// `simd_sum`, so a tile of 32 keys was 32 memory latencies in a row
/// twice, once for K and once for V. Measured serialized
/// (`crate::kernel_bench`): 38.9 us for Gemma-2-2B's 8 heads of 256 at
/// 96 keys, against llama.cpp's 13.4 us for the same op, and the
/// difference is that its `FOR_UNROLL` tile loop lets every key's
/// loads go out together. Here the loop is unrolled over all C keys;
/// a key past the tile's valid count reads the tile's last valid row
/// (always in bounds) and scores -INF, so its softmax weight is
/// exactly zero and it adds exactly `v * 0.0f` to the output. The
/// arithmetic order for every valid key is unchanged, so the result
/// is bit-identical to the loop it replaces.
const GQA_DECODE_FA_VEC_WIDE_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)") for

template <uint D>
kernel void gqa_decode_fa_vec_wide(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    device const float* sinks [[buffer(10)]],
    constant uint& has_sinks [[buffer(11)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D4 = D / 4u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    // float4 slices of the head each lane owns: lane t owns slices
    // t, t + 32, ... so the simd reduce covers the whole head.
    constexpr uint SL = D4 / NW;
    // Per-SG floats: C scores + D output.
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));
    // float4 stride from one cached token's row of this kv head to the next.
    const uint row4 = n_kv_heads * D4;

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    FOR_UNROLL (uint sl = 0; sl < SL; sl++) {
        so4[tiisg + sl * NW] = float4(0.0f);
    }
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        device const half4* k4 = (device const half4*)(k_cache + (ic * n_kv_heads + kv_h) * D);
        device const half4* v4 = (device const half4*)(v_cache + (ic * n_kv_heads + kv_h) * D);

        // Q.K for all C keys of the tile, every key's loads independent
        // of the previous key's reduce. Lane `cc` keeps key `cc`'s
        // score; a key past `chunk` reads the last valid row and is
        // masked to -INF.
        float s_lane = -INFINITY;
        FOR_UNROLL (uint cc = 0; cc < C; cc++) {
            const uint r = min(cc, chunk - 1u);
            device const half4* kr = k4 + r * row4;
            float partial = 0.0f;
            FOR_UNROLL (uint sl = 0; sl < SL; sl++) {
                partial += dot(sq4[tiisg + sl * NW], float4(kr[tiisg + sl * NW]));
            }
            float sc = simd_sum(partial) * scale;
            if (tiisg == cc) {
                s_lane = (cc < chunk) ? sc : -INFINITY;
            }
        }
        // Softcap once per lane on the score it kept, not 32 times per
        // lane on every key's broadcast score: the same expression on
        // the same value, so the same result.
        if (softcap > 0.0f && s_lane != -INFINITY) {
            s_lane = softcap * tanh(s_lane / softcap);
        }

        // Online softmax over this tile (one score per lane).
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        FOR_UNROLL (uint sl = 0; sl < SL; sl++) {
            so4[tiisg + sl * NW] *= ms;
        }
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O += P . V, again over all C keys: a masked key's weight is
        // exactly zero, and its clamped row is finite, so it adds 0.
        float4 lo[SL];
        FOR_UNROLL (uint sl = 0; sl < SL; sl++) {
            lo[sl] = float4(0.0f);
        }
        FOR_UNROLL (uint cc = 0; cc < C; cc++) {
            const uint r = min(cc, chunk - 1u);
            device const half4* vr = v4 + r * row4;
            const float p = ss[cc];
            FOR_UNROLL (uint sl = 0; sl < SL; sl++) {
                lo[sl] += float4(vr[tiisg + sl * NW]) * p;
            }
        }
        FOR_UNROLL (uint sl = 0; sl < SL; sl++) {
            so4[tiisg + sl * NW] += lo[sl];
        }
    }

    // Publish S,M for cross-SG reduce (reuse ss[0], ss[1]).
    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG online-softmax merge.
    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            FOR_UNROLL (uint sl = 0; sl < SL; sl++) {
                so0[tiisg + sl * NW] = so0[tiisg + sl * NW] * a0 + so1[tiisg + sl * NW] * a1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        // gpt-oss's attention sink: one more logit in the denominator
        // with no value, so it only rescales (llama `fa_vec_common`).
        float S_f = ss0[0];
        float a_f = 1.0f;
        if (has_sinks != 0u) {
            const float s_k = sinks[h];
            const float M_f = max(ss0[1], s_k);
            a_f = (ss0[1] == -INFINITY) ? 0.0f : exp(ss0[1] - M_f);
            S_f = S_f * a_f + exp(s_k - M_f);
        }
        float inv = (S_f == 0.0f) ? 0.0f : (a_f / S_f);
        device float4* out4 = (device float4*)(out + h * D);
        FOR_UNROLL (uint sl = 0; sl < SL; sl++) {
            out4[tiisg + sl * NW] = so0[tiisg + sl * NW] * inv;
        }
    }
}

typedef decltype(gqa_decode_fa_vec_wide<128>) gqa_decode_fa_vec_wide_t;
template [[host_name("gqa_decode_fa_vec_d128")]] kernel gqa_decode_fa_vec_wide_t gqa_decode_fa_vec_wide<128>;
template [[host_name("gqa_decode_fa_vec_d256")]] kernel gqa_decode_fa_vec_wide_t gqa_decode_fa_vec_wide<256>;
template [[host_name("gqa_decode_fa_vec_d512")]] kernel gqa_decode_fa_vec_wide_t gqa_decode_fa_vec_wide<512>;
"#;

/// TG size for FA-vec decode (d=64/96/128/256/512): NSG=8 × NW=32.
pub(crate) fn gqa_fa_vec_threadgroup_size(_head_dim: u32) -> u32 {
    256
}

/// Head dims the FA-vec decode kernels cover (dedicated specializations).
///
/// 512 is Gemma-4's full-attention width. It is the wide kernel at
/// `D = 512` (four float4 slices per lane), and its threadgroup memory,
/// `(512 + 8 * (32 + 512)) * 4` = 19 KiB, is under the 32 KiB limit.
pub(crate) fn gqa_fa_vec_supported(head_dim: u32) -> bool {
    matches!(head_dim, 64 | 96 | 128 | 256 | 512)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gqa_fa_vec(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    v: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_start: u32,
    softcap: f32,
    sinks: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<(), MetalError> {
    let pipe = match head_dim {
        512 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_WIDE_KERNEL_SRC,
            "gqa_decode_fa_vec_d512",
        )?,
        256 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_WIDE_KERNEL_SRC,
            "gqa_decode_fa_vec_d256",
        )?,
        128 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_WIDE_KERNEL_SRC,
            "gqa_decode_fa_vec_d128",
        )?,
        96 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_D96_KERNEL_SRC,
            "gqa_decode_fa_vec_d96",
        )?,
        64 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_D64_KERNEL_SRC,
            "gqa_decode_fa_vec_d64",
        )?,
        _ => return Err(MetalError::CommandFailed),
    };
    encoder.setComputePipelineState(&pipe.0);
    let tg = gqa_fa_vec_threadgroup_size(head_dim);
    let nsg = tg / 32;
    // Q[D] + NSG * (C=32 scores + D output)
    let tg_mem = ((head_dim + nsg * (32 + head_dim)) * 4) as usize;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut nh = n_heads;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nh as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut nkv = n_kv_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut nkv as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut hd = head_dim;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 6);
        let mut sl = seq_len;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sl as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut ks = kv_start;
        encoder.setBytes_length_atIndex(NonNull::new(&mut ks as *mut u32 as *mut _).unwrap(), 4, 8);
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        crate::attn::bind_sinks(encoder, sinks, q, 10, 11);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_heads as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
