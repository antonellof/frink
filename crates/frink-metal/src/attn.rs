//! Device K/V caches default to **f16** (llama.cpp `-ctk f16 -ctv f16`).
//! Quantized stores: `FRINK_CTK=q8_0|fp8|q4` (an unknown value falls
//! back to f16). Quant caches dequant to a process-wide f16 scratch before
//! FA/GQA. Q/activations stay f32. Append converts on GPU.
//!
//! **Decode (B=1):** Q/K/V/O matvecs fused with RoPE→KV→GQA in one command
//! buffer ([`launch_decode_attn_block`]); dense layers can continue through
//! residual → FFN in the same CB ([`launch_decode_dense_layer`] /
//! [`launch_decode_dense_stack`]).
//!
//! **Prefill (T≥1):** dense layers with B≥4 can run
//! [`launch_prefill_dense_layer`] (one layer) or [`launch_prefill_dense_stack`]
//! (consecutive layers, one CB, one host readback) — RMSNorm → Q/K/V
//! `mul_mm_sg` → RoPE/KV/GQA → O → FFN (gate∥up→act→down) with activations
//! resident on GPU scratch (decode-stack barriers). Otherwise host (or
//! batched) Q/K/V feed [`launch_prefill_attn_block`] — multi-pos RoPE,
//! batch KV append into [`MetalKvBuffers`], causal GQA — so decode can
//! skip [`MetalKvBuffers::upload_from_host`] when seq_lens already match.
//!
//! Prefill GEMM timing: `FRINK_METAL_MM_TIMING=1` (see
//! [`crate::gpu::launch_dense_ffn_swiglu_batch`] / mul_mm_sg launches, and
//! [`launch_prefill_dense_layer`] / [`launch_prefill_dense_stack`]
//! setup/gpu/readback totals).
//!
//! Scope:
//! - `LLAMA_ROPE_TYPE_NORM` (interleaved) or `NEOX` ± Llama-3 freq factors
//! - Full-causal GQA (no sliding window); online-softmax kernels
//! - Prefill dense stack supports QKV bias + QK-norm via [`AttnExtras`]
//!   (same order as CPU: bias → norm → RoPE)
//!
//! Enable with `FRINK_METAL_ATTN=1` (also requires dense Metal / `FRINK_METAL`).
//! Sampled decode downloads the hidden vector and runs lm_head on the host:
//! folding the full vocab projection into the stack instead measured ~2x
//! slower on Llama-3.1-8B Q4_K_M.
//!
//! Greedy decode (`temperature<=0`, hooked from `frink-server::generate` /
//! `frink-cli`) folds final_norm + lm_head + **argmax** into the same CB
//! (dense [`launch_decode_dense_stack`] or MoE [`launch_moe_decode_stack`])
//! and downloads only the top-1 token id, so the vocab never crosses the
//! bus. Embedding gather can also run on-GPU (`get_rows`) so the dense
//! stack needs no host `dequant_row` upload.
//!
//! `FRINK_METAL_FA_VEC=0` disables llama-style FA-vec decode **and** prefill
//! (default **on** for `head_dim` in {64,96,128,256}). Other head dims keep
//! legacy online-softmax GQA. Prefill at `head_dim` 64/128/256 with `n_q >= 8`
//! goes further and takes the simdgroup-MMA `flash_attn_ext` kernel; the
//! parity tests reach a specific one through [`PrefillAttnKernel`], not an
//! environment variable.
//!
//! `FRINK_CTK` selects KV dtype ([`MetalKvDtype`]); see [`is_implemented`].
pub use crate::decode_dense::{
    launch_decode_dense_stack, AttnExtras, DenseLayerMetal, EmbdGatherMetal,
};
pub(crate) use crate::dispatch::dispatch_counted;
use crate::elem::{
    encode_act_mul_f32_to_f16, encode_argmax, encode_f32_to_f16, encode_silu_mul, encode_vec_add,
    encode_vec_add_at, warm_prefill_elem_pipelines,
};
use crate::embd::encode_get_rows;
use crate::fa_vec_decode::{encode_gqa_fa_vec, gqa_fa_vec_supported};
use crate::gpu::{
    compute_encoder_concurrent, encode_matvec, encode_moe_topk_softmax_batch, encode_mul_mm_sg_f16,
    encode_q4_0_moe_gate_up_id, encode_q4_0_moe_id, encode_q4_0_moe_topk, ensure_pipeline,
    memory_barrier_resources, resident_f32_buffer, resident_weight_buffer, shared_metal,
    warm_mul_mm_sg_pipeline, MatvecLaunch, MetalError, MoeExpertLaunch, MoePackedQ4, MulMmSgLaunch,
    ResidentF32Buffer, ResidentWeightBuffer,
};
use crate::mem_ranges::MemRanges;
use crate::moe_ids::MoeIdsLog;
use crate::norm::{
    encode_add_rms_norm, encode_add_rms_norm_batch, encode_add_rms_norm_f32_to_f16_batch,
    encode_rms_norm, encode_rms_norm_at, encode_rms_norm_batch, encode_rms_norm_f32_to_f16_batch,
    encode_rms_norm_per_head_batch,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};
// RoPE moved to `crate::rope`; re-exported here because `frink-models`
// and the sibling stacks read it at `frink_metal::attn::`.
pub(crate) use crate::rope::{
    assert_freq_factors_len, encode_rope, encode_rope_batch, EncodedRope, RopeTarget,
};
pub use crate::rope::{
    launch_rope_heads_batch_host, launch_rope_heads_host, LayerRope, MetalRope, MetalRopeLayout,
};
use std::cell::RefCell;
use std::collections::HashSet;
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

/// Whether the fused Metal attention block should run (in addition to
/// dense Metal matvecs). Default off until measured; `1|true|on` enables.
pub fn metal_attn_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("FRINK_METAL_ATTN").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("attn")
        )
    })
}

// The greedy GPU argmax fold's setting lives in `crate::greedy_fold`,
// which states what it is and why running a decode step on the wrong
// thread used to change the answer (GitHub issue #166). Re-exported
// here because `frink-models` reads it at `frink_metal::attn::`.
// The KV wire moved to `crate::kv_wire` when the K rotation
// landed. Re-exported here because `frink-models`, the bench guard and
// the tests all read these at `frink_metal::attn::`, and a wire format
// is not worth a rename across nine call sites.
pub use crate::kv_wire::{
    effective_metal_kv_dtype, encode_kv_dequant_to_f16, encode_kv_store_append, metal_kv_dtype,
    metal_kv_q4_viable, metal_kv_q8_0_viable, parse_metal_kv_dtype, MetalKvDtype,
};

pub use crate::greedy_fold::{
    adopt_greedy_fold, greedy_fold_setting, metal_greedy_argmax_active, set_metal_greedy_argmax,
    GreedyFold, GreedyFoldGuard,
};

/// Whether FA-vec GQA decode is enabled.
///
/// Default: **on** for supported head dims (64 / 96 / 128 / 256) via
/// [`encode_gqa`]. `FRINK_METAL_FA_VEC=0|false|off` forces the legacy
/// online-softmax kernel. `=1|true|on|vec` keeps FA on (still only
/// dispatched for supported dims).
pub fn metal_fa_vec_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        match std::env::var("FRINK_METAL_FA_VEC").ok().as_deref() {
            Some("0") | Some("false") | Some("off") => false,
            Some("1") | Some("true") | Some("on") | Some("vec") => true,
            // Default on: llama-parity FA for d=128 is required to close the
            // ~1.15× decode gap (legacy GQA ≈ llama `-ctk f32`).
            _ => true,
        }
    })
}

const GQA_DECODE_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Online-softmax GQA decode (B=1, full causal).
// One threadgroup per query head. V accumulators stay register-local
// through the seq scan (float4 dots/axpy when head_dim % 4 == 0); each
// simdgroup reduces via simd_shuffle_xor, then N_SG partials merge in a
// compact TG footprint O(nsg * head_dim) instead of O(tg * head_dim).
// `tg` must be a multiple of 32 (host enforces). head_dim <= 256.

inline float online_rescale(float m_old, float m_new) {
    // exp(m_old - m_new); 0 when m_old is -inf (empty / inactive partial).
    return (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
}

kernel void gqa_decode(
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
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    if (h >= n_heads || seq_len == 0u) return;

    constexpr uint MAX_D = 256u;
    constexpr uint NW = 32u;
    const uint nsg = tg / NW;
    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;

    // Compact layout after per-SG reduce: m[nsg] | s[nsg] | acc[nsg * head_dim]
    threadgroup float* m_sh = shared;
    threadgroup float* s_sh = shared + nsg;
    threadgroup float* acc_base = shared + 2u * nsg;

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(head_dim));
    device const float* q_h = q + h * head_dim;

    float m = -INFINITY;
    float s = 0.0f;
    float my_acc[MAX_D];
    for (uint d = 0; d < head_dim; d++) {
        my_acc[d] = 0.0f;
    }

    const bool vec4 = (head_dim & 3u) == 0u;
    // kv_start > 0 = sliding-window attention: only positions
    // [kv_start, seq_len) are visible (Gemma-style SWA).
    for (uint t = kv_start + tid; t < seq_len; t += tg) {
        device const half* k_t =
            k_cache + (t * n_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        if (vec4) {
            float4 acc4 = float4(0.0f);
            for (uint d = 0; d < head_dim; d += 4u) {
                acc4 += float4(
                    q_h[d], q_h[d + 1u], q_h[d + 2u], q_h[d + 3u])
                    * float4(
                        float(k_t[d]), float(k_t[d + 1u]), float(k_t[d + 2u]), float(k_t[d + 3u]));
            }
            dot = acc4[0] + acc4[1] + acc4[2] + acc4[3];
        } else {
            for (uint d = 0; d < head_dim; d++) {
                dot += q_h[d] * float(k_t[d]);
            }
        }
        float score = dot * scale;
        if (softcap > 0.0f) {
            score = softcap * tanh(score / softcap);
        }
        float m2 = max(m, score);
        float a = online_rescale(m, m2);
        float b = exp(score - m2);
        s = s * a + b;
        device const half* v_t =
            v_cache + (t * n_kv_heads + kv_h) * head_dim;
        if (vec4) {
            for (uint d = 0; d < head_dim; d += 4u) {
                my_acc[d] = my_acc[d] * a + b * float(v_t[d]);
                my_acc[d + 1u] = my_acc[d + 1u] * a + b * float(v_t[d + 1u]);
                my_acc[d + 2u] = my_acc[d + 2u] * a + b * float(v_t[d + 2u]);
                my_acc[d + 3u] = my_acc[d + 3u] * a + b * float(v_t[d + 3u]);
            }
        } else {
            for (uint d = 0; d < head_dim; d++) {
                my_acc[d] = my_acc[d] * a + b * float(v_t[d]);
            }
        }
        m = m2;
    }

    // Intra-simdgroup butterfly reduce (no TG traffic).
    for (ushort offset = ushort(NW >> 1); offset > 0u; offset >>= 1) {
        float m_o = simd_shuffle_xor(m, offset);
        float s_o = simd_shuffle_xor(s, offset);
        float m_new = max(m, m_o);
        float a = online_rescale(m, m_new);
        float a_o = online_rescale(m_o, m_new);
        s = s * a + s_o * a_o;
        for (uint d = 0; d < head_dim; d++) {
            float ao = simd_shuffle_xor(my_acc[d], offset);
            my_acc[d] = my_acc[d] * a + ao * a_o;
        }
        m = m_new;
    }

    if (tiisg == 0u) {
        m_sh[sgitg] = m;
        s_sh[sgitg] = s;
        threadgroup float* slot = acc_base + sgitg * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            slot[d] = my_acc[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG tree reduce over nsg << tg partials.
    for (uint stride = nsg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            uint other = tid + stride;
            float m1 = m_sh[tid];
            float m2 = m_sh[other];
            float s1 = s_sh[tid];
            float s2 = s_sh[other];
            float m_new = max(m1, m2);
            float a1 = online_rescale(m1, m_new);
            float a2 = online_rescale(m2, m_new);
            m_sh[tid] = m_new;
            s_sh[tid] = s1 * a1 + s2 * a2;
            threadgroup float* acc1 = acc_base + tid * head_dim;
            threadgroup float* acc2 = acc_base + other * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                acc1[d] = acc1[d] * a1 + acc2[d] * a2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_s = 1.0f / s_sh[0];
    threadgroup float* acc0 = acc_base;
    for (uint d = tid; d < head_dim; d += tg) {
        out[h * head_dim + d] = acc0[d] * inv_s;
    }
}
"#;

/// FA-vec multi-query causal prefill for head_dim **64 / 96 / 128**, f16
/// KV, C=32. One TG per `(head, query_token)`; same tile/merge as decode
/// FA-vec, but each query `qi` attends only over `[0 ..= kv_prefix_len + qi]`.
///
/// One body, three entry points. Only 128 existed before, which meant
/// every d=64 model (SmolLM2, TinyLlama, Llama-3.2-1B, Qwen2.5-0.5B) ran
/// the legacy `gqa_prefill` kernel instead. That kernel keeps a
/// *per-thread* accumulator in threadgroup memory — `tg * head_dim`
/// floats, ~27 KB at 108 threads — which caps occupancy at roughly one
/// threadgroup per core. This one needs `D + nsg*(C+D)` floats, 3.3 KB at
/// d=64. Profiling SmolLM2 metal `pp512` put 62% of prefill inside that
/// legacy kernel's `waitUntilCompleted`.
///
/// `D4 = D/4` is how many float4 lanes carry the query and the output.
/// At d=128 that is exactly the 32-lane simdgroup; at d=64 and d=96 it is
/// fewer, so the lanes above `D4` sit out the dot product and the
/// accumulator updates. They still participate in `simd_sum`/`simd_max`,
/// which is what makes the masking safe rather than merely lucky.
macro_rules! gqa_prefill_fa_vec_src {
    ($name:literal, $d:literal) => {
        concat!(
            r#"
#include <metal_stdlib>
using namespace metal;

kernel void "#,
            $name,
            r#"(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_q [[buffer(7)]],
    constant uint& kv_prefix_len [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = "#,
            $d,
            r#"u;
    constexpr uint D4 = D / 4u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    constexpr uint SG_F = C + D;

    uint h = tgpig.x;
    uint qi = tgpig.y;
    if (h >= n_heads || qi >= n_q || head_dim != D) return;

    uint causal_len = kv_prefix_len + qi + 1u;
    if (causal_len == 0u) return;

    uint tid = tid_tg.x;
    uint tg = tg_size.x;
    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;
    const bool own = tiisg < D4;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + (qi * n_heads + h) * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    if (own) {
        so4[tiisg] = float4(0.0f);
    }
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = ic0 * C;
        if (ic >= causal_len) break;
        uint chunk = min(C, causal_len - ic);

        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = own ? dot(sq4[tiisg], float4(k4[tiisg])) : 0.0f;
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
        if (own) {
            so4[tiisg] *= ms;
        }
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (own) {
            float4 lo = float4(0.0f);
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
            if (own) {
                so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u && own) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + (qi * n_heads + h) * D);
        out4[tiisg] = so0[tiisg] * inv;
    }
}
"#
        )
    };
}

const GQA_PREFILL_FA_VEC_KERNEL_SRC: &str = gqa_prefill_fa_vec_src!("gqa_prefill_fa_vec", "128");
const GQA_PREFILL_FA_VEC_D64_KERNEL_SRC: &str =
    gqa_prefill_fa_vec_src!("gqa_prefill_fa_vec_d64", "64");
const GQA_PREFILL_FA_VEC_D96_KERNEL_SRC: &str =
    gqa_prefill_fa_vec_src!("gqa_prefill_fa_vec_d96", "96");

/// llama `kernel_flash_attn_ext` for d=64, ported with **real 8×8 simdgroup
/// MMA** on both Q·Kᵀ and P·V (`ggml-metal.metal`, `kernel_flash_attn_ext_impl`
/// — Q·Kᵀ at :6693-6729, P·V at :6841-6910; d=64 instantiation at :7126).
///
/// Shape is llama's: QN=8 queries and C=64 keys per threadgroup, NSG=4
/// simdgroups, `ss[QN][2C]` scores, `so[QN][64]` accumulator. Its scalar
/// predecessor (deleted with the switch that selected it) had the same
/// *tiling* but computed
/// scores with one `dot`+`simd_sum` per (query,key) inside a single simdgroup
/// with 16 of 32 lanes live — 16 of 128 threads doing arithmetic. Here all
/// four simdgroups run MMA over disjoint 8-key column blocks of `ss`, and P·V
/// runs over disjoint 8-wide column blocks of `so`.
///
/// Two things differ from llama and are load-bearing:
///
/// 1. **Where scale/softcap/mask are applied.** llama has an explicit mask
///    tensor and folds `scale`/`softcap`/`mask` into the online-softmax loop.
///    The predecessor kernel had to fold them into the *score* loop instead,
///    because an earlier version ran them as a separate all-simdgroups pass
///    over shared `ss` slots: a read-modify-write with four readers and four
///    writers and no barrier between them, which double-softcapped whichever
///    slots lost the race (see the comment on the scalar kernel). MMA scores
///    are written by `simdgroup_store`, so that fold is no longer possible —
///    they move back into the softmax loop, which is safe for the reason
///    llama's is: there each `ss` slot has exactly **one** owner
///    (`j = jj*NSG + sgitg` picks disjoint rows per simdgroup, `tiisg` picks
///    disjoint `float2` columns per lane), so the RMW has a single writer.
///    Any future edit must preserve that ownership, not the `sgitg == 0` guard
///    it replaces.
///
/// 2. **Tail handling.** `simdgroup_load` reads a full 8 rows of K/V, so the
///    last partial group of a cache whose length is not a multiple of 8 would
///    read past the end of the buffer. llama pre-pads its KV cache; Frink's
///    is exactly `kv_prefix_len + n_q` rows, so instead the ≤7 leftover rows
///    are staged once into a zero-filled `kpad`/`vpad` tile in threadgroup
///    memory and the MMA reads that. Groups entirely past the cache are
///    skipped: the causal mask forces their `ss` columns to `-INFINITY`
///    (`ic + cc >= kv_valid >= clen`), hence P = 0, hence no P·V contribution.
///
/// Parameterised over the head dim: every shape constant below is derived from
/// `D`, so the same body serves d=64, d=128 and d=256.
///
/// The three places that touch the `so` accumulator row-wise — zero-init,
/// the online-softmax rescale, and the epilogue — walk it with llama's lane
/// loop `for (i = tiisg; i < D4; i += NW)` (`ggml-metal.metal:6529-6535`,
/// `:6826-6838`, `:7024-7034`). Below `D4 <= NW` that loop runs exactly one
/// iteration on exactly the lanes the previous `own = tiisg < D4` guard
/// selected, with the same index, so d=64 and d=128 are unchanged; at d=256
/// (`D4 == 64`) it is what lets one lane carry two `float4` columns.
///
/// Remaining shape constraints: `D % 16 == 0`, because the Q·Kᵀ loop walks the
/// head 16 columns at a time as a pair of 8×8 MMAs; `(D/8) % NSG == 0`, so the
/// P·V output blocks partition evenly across simdgroups; and threadgroup
/// memory `(2·QN·D + QN·SH)·4 + 4·8·D` bytes, which is 28 KiB at d=256 and so
/// the last width that fits Apple's 32 KiB limit at this tiling.
macro_rules! gqa_prefill_fa_ext_mma_src {
    ($name:literal, $d:literal) => {
        concat!(
            r#"
#include <metal_stdlib>
using namespace metal;

kernel void "#,
            $name,
            r#"(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_q [[buffer(7)]],
    constant uint& kv_prefix_len [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = "#,
            $d,
            r#"u;          // DK == DV == PV
    constexpr uint D4 = D / 4u;      // output float4 columns == owning lanes
    constexpr uint D8 = D / 8u;      // D / 8: 8-wide MMA steps along the head
    constexpr uint QN = 8u;          // queries per threadgroup
    constexpr uint C = 64u;          // keys per threadgroup chunk
    constexpr uint NW = 32u;
    constexpr uint NSG = 4u;
    constexpr uint NQ = QN / NSG;    // softmax rows per simdgroup
    constexpr uint SH = 2u * C;      // ss row stride
    constexpr uint CB = C / 8u;      // 8-key blocks per chunk
    constexpr uint NC = CB / NSG;    // score blocks per simdgroup
    constexpr uint NO = D8 / NSG;    // output column blocks per simdgroup

    const uint h = tgpig.x;
    const uint qi0 = tgpig.y * QN;
    if (h >= n_heads || qi0 >= n_q || head_dim != D) return;

    const uint group_size = n_heads / max(n_kv_heads, 1u);
    const uint kv_h = h / max(group_size, 1u);
    const uint kv_stride = n_kv_heads * D;
    const float scale = 1.0f / sqrt(float(D));
    const uint n_local = min(QN, n_q - qi0);

    // sq[QN,D] f32 | so[QN,D] f32 | ss[QN,SH] f32 | kpad[8,D] f16 | vpad[8,D] f16
    threadgroup float* sq = shared;
    threadgroup float* so = shared + QN * D;
    threadgroup float* ss = shared + 2u * QN * D;
    threadgroup half* kpad = (threadgroup half*)(shared + 2u * QN * D + QN * SH);
    threadgroup half* vpad = kpad + 8u * D;

    // Rows of K/V that physically exist, and the 8-row-aligned prefix of them
    // that `simdgroup_load` may read straight out of device memory.
    const uint kv_valid = kv_prefix_len + n_q;
    const uint kv_full = (kv_valid / 8u) * 8u;
    const uint kv_rem = kv_valid - kv_full;

    for (uint j = 0u; j < QN; j++) {
        const uint gqi = qi0 + j;
        threadgroup float4* sq4 = (threadgroup float4*)(sq + j * D);
        if (gqi < n_q) {
            device const float4* q4 =
                (device const float4*)(q + (gqi * n_heads + h) * D);
            for (uint i = tiisg; i < D4; i += NW) sq4[i] = q4[i];
        } else {
            // Zero query rows: their MMA scores are 0, the softmax skips them,
            // and nothing reads their `so` rows back out.
            for (uint i = tiisg; i < D4; i += NW) sq4[i] = float4(0.0f);
        }
        threadgroup float4* so4 = (threadgroup float4*)(so + j * D);
        for (uint i = tiisg; i < D4; i += NW) so4[i] = float4(0.0f);
    }

    if (kv_rem > 0u) {
        const uint tid = uint(sgitg) * NW + uint(tiisg);
        for (uint idx = tid; idx < 8u * D; idx += NSG * NW) {
            const uint r = idx / D;
            const uint c = idx - r * D;
            half kk = (half)0.0f;
            half vv = (half)0.0f;
            if (r < kv_rem) {
                const uint base = (kv_full + r) * kv_stride + kv_h * D + c;
                kk = k_cache[base];
                vv = v_cache[base];
            }
            kpad[idx] = kk;
            vpad[idx] = vv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (uint jj = 0u; jj < NQ; jj++) {
        S[jj] = 0.0f;
        M[jj] = -INFINITY;
    }

    uint max_causal = 0u;
    for (uint j = 0u; j < n_local; j++) {
        max_causal = max(max_causal, kv_prefix_len + qi0 + j + 1u);
    }
    if (max_causal == 0u) return;

    for (uint ic0 = 0u; ; ic0++) {
        const uint ic = ic0 * C;
        if (ic >= max_causal) break;

        // Q·Kᵀ — 8x8 MMA. Simdgroup `sgitg` owns key blocks
        // {sgitg, sgitg + NSG}, i.e. ss columns [8g, 8g+8): disjoint, so
        // `simdgroup_store` is a single writer per slot.
        for (uint cb = 0u; cb < NC; cb++) {
            const uint g = uint(sgitg) + cb * NSG;
            const uint key0 = ic + 8u * g;
            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
            if (key0 + 8u <= kv_full) {
                device const half* pk = k_cache + key0 * kv_stride + kv_h * D;
                for (uint i = 0u; i < D8 / 2u; i++) {
                    simdgroup_float8x8 mq0, mq1;
                    simdgroup_half8x8 mk0, mk1;
                    simdgroup_load(mq0, sq + 16u * i, D);
                    simdgroup_load(mq1, sq + 16u * i + 8u, D);
                    // transpose: [key,dim] -> [dim,key]
                    simdgroup_load(mk0, pk + 16u * i, kv_stride, 0, true);
                    simdgroup_load(mk1, pk + 16u * i + 8u, kv_stride, 0, true);
                    simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                    simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
                }
            } else if (key0 == kv_full && kv_rem > 0u) {
                threadgroup const half* pk = kpad;
                for (uint i = 0u; i < D8 / 2u; i++) {
                    simdgroup_float8x8 mq0, mq1;
                    simdgroup_half8x8 mk0, mk1;
                    simdgroup_load(mq0, sq + 16u * i, D);
                    simdgroup_load(mq1, sq + 16u * i + 8u, D);
                    simdgroup_load(mk0, pk + 16u * i, D, 0, true);
                    simdgroup_load(mk1, pk + 16u * i + 8u, D, 0, true);
                    simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                    simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
                }
            } else {
                // Entirely past the cache: every column here is masked below.
                continue;
            }
            simdgroup_store(mqk, ss + 8u * g, SH, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax. Single-writer per ss slot: row j is owned by one
        // simdgroup, column pair `tiisg` by one lane. scale/softcap/causal
        // mask are applied here, on a value read once and written once.
        for (uint jj = 0u; jj < NQ; jj++) {
            const uint j = jj * NSG + sgitg;
            if (j >= n_local) continue;
            threadgroup float2* ss2 = (threadgroup float2*)(ss + j * SH);
            float2 s2 = ss2[tiisg] * scale;
            if (softcap > 0.0f) s2 = softcap * tanh(s2 / softcap);
            const uint clen = kv_prefix_len + qi0 + j + 1u;
            const uint c0 = ic + 2u * uint(tiisg);
            // `cc >= chunk` is subsumed: it implies ic+cc >= max_causal >= clen.
            if (c0 >= clen) s2[0] = -INFINITY;
            if (c0 + 1u >= clen) s2[1] = -INFINITY;

            const float m = M[jj];
            M[jj] = simd_max(max(m, max(s2[0], s2[1])));
            const float ms = (m == -INFINITY) ? 0.0f : exp(m - M[jj]);
            const float2 vs2 = float2(
                (s2[0] == -INFINITY) ? 0.0f : exp(s2[0] - M[jj]),
                (s2[1] == -INFINITY) ? 0.0f : exp(s2[1] - M[jj])
            );
            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[tiisg] = vs2;

            threadgroup float4* so4 = (threadgroup float4*)(so + j * D);
            for (uint i = tiisg; i < D4; i += NW) so4[i] *= ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += P·V — 8x8 MMA. Simdgroup `sgitg` owns output columns
        // {8*sgitg + 8*NSG*ii}, disjoint across simdgroups; every simdgroup
        // walks all CB key blocks.
        {
            simdgroup_float8x8 lo[NO];
            for (uint ii = 0u; ii < NO; ii++) {
                simdgroup_load(lo[ii], so + 8u * sgitg + 8u * NSG * ii, D, 0, false);
            }
            for (uint cc = 0u; cc < CB; cc++) {
                const uint key0 = ic + 8u * cc;
                const bool fullblk = (key0 + 8u <= kv_full);
                const bool padblk = (key0 == kv_full) && (kv_rem > 0u);
                if (!fullblk && !padblk) continue;
                simdgroup_float8x8 vs;
                simdgroup_load(vs, ss + 8u * cc, SH, 0, false);
                if (fullblk) {
                    device const half* pv =
                        v_cache + key0 * kv_stride + kv_h * D + 8u * sgitg;
                    for (uint ii = 0u; ii < NO; ii++) {
                        simdgroup_half8x8 mv;
                        simdgroup_load(mv, pv + 8u * NSG * ii, kv_stride, 0, false);
                        simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                    }
                } else {
                    threadgroup const half* pv = vpad + 8u * sgitg;
                    for (uint ii = 0u; ii < NO; ii++) {
                        simdgroup_half8x8 mv;
                        simdgroup_load(mv, pv + 8u * NSG * ii, D, 0, false);
                        simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                    }
                }
            }
            for (uint ii = 0u; ii < NO; ii++) {
                simdgroup_store(lo[ii], so + 8u * sgitg + 8u * NSG * ii, D, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Each SG writes its NQ queries (no cross-SG KV reduce — llama layout).
    for (uint jj = 0u; jj < NQ; jj++) {
        const uint j = jj * NSG + sgitg;
        if (j >= n_local) continue;
        const float inv = (S[jj] == 0.0f) ? 0.0f : (1.0f / S[jj]);
        device float4* out4 =
            (device float4*)(out + ((qi0 + j) * n_heads + h) * D);
        threadgroup float4* so4 = (threadgroup float4*)(so + j * D);
        for (uint i = tiisg; i < D4; i += NW) out4[i] = so4[i] * inv;
    }
}
"#
        )
    };
}

const GQA_PREFILL_FA_EXT_MMA_D64_KERNEL_SRC: &str =
    gqa_prefill_fa_ext_mma_src!("gqa_prefill_fa_ext_mma_d64", "64");
/// Qwen3-0.6B / Phi-4-mini / Mistral shape: head_dim 128. There is no scalar
/// `fa_ext` predecessor at this width (that kernel is d=64-only), so the A/B
/// reference for both correctness and timing is `gqa_prefill_fa_vec`.
const GQA_PREFILL_FA_EXT_MMA_D128_KERNEL_SRC: &str =
    gqa_prefill_fa_ext_mma_src!("gqa_prefill_fa_ext_mma_d128", "128");
/// Gemma-2 / Gemma-3 shape: head_dim 256, the width the `own`-guarded epilogue
/// could not reach. `D4 == 64` means each lane carries two `float4` output
/// columns and `NO == 8` accumulator matrices per simdgroup; threadgroup memory
/// is 28 KiB. Like d=128 there is no scalar `fa_ext` at this width, so the A/B
/// reference for correctness and timing is `gqa_prefill_fa_vec_d256`.
const GQA_PREFILL_FA_EXT_MMA_D256_KERNEL_SRC: &str =
    gqa_prefill_fa_ext_mma_src!("gqa_prefill_fa_ext_mma_d256", "256");

const GQA_PREFILL_FA_VEC_D256_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_prefill_fa_vec_d256(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_q [[buffer(7)]],
    constant uint& kv_prefix_len [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = 256u;
    constexpr uint D4 = 64u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    constexpr uint SG_F = C + D;

    uint h = tgpig.x;
    uint qi = tgpig.y;
    if (h >= n_heads || qi >= n_q || head_dim != D) return;

    uint causal_len = kv_prefix_len + qi + 1u;
    if (causal_len == 0u) return;

    uint tid = tid_tg.x;
    uint tg = tg_size.x;
    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + (qi * n_heads + h) * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    // D=256 is 64 float4 spread over 32 lanes, so each lane owns *two*
    // of them: tiisg and tiisg+NW. Touching only the first truncated both
    // the Q.K dot and the output to the first 128 of 256 head dims. This
    // kernel was cloned from the d=128 one, where one float4 per lane is
    // exactly right; at d=256 it silently dropped half of every head.
    so4[tiisg] = float4(0.0f);
    so4[tiisg + NW] = float4(0.0f);
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = ic0 * C;
        if (ic >= causal_len) break;
        uint chunk = min(C, causal_len - ic);

        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = dot(sq4[tiisg], float4(k4[tiisg]))
                          + dot(sq4[tiisg + NW], float4(k4[tiisg + NW]));
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
        so4[tiisg] *= ms;
        so4[tiisg + NW] *= ms;
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo0 = float4(0.0f);
        float4 lo1 = float4(0.0f);
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* v4 =
                (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            lo0 += float4(v4[tiisg]) * ss[cc];
            lo1 += float4(v4[tiisg + NW]) * ss[cc];
        }
        so4[tiisg] += lo0;
        so4[tiisg + NW] += lo1;
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
            so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            so0[tiisg + NW] = so0[tiisg + NW] * a0 + so1[tiisg + NW] * a1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + (qi * n_heads + h) * D);
        out4[tiisg] = so0[tiisg] * inv;
        out4[tiisg + NW] = so0[tiisg + NW] * inv;
    }
}
"#;

// Multi-token causal GQA prefill: one threadgroup per (query token, head).
// Query `qi` at absolute cache index `kv_prefix_len + qi` attends over
// K/V[0 ..= kv_prefix_len + qi] (inclusive), matching host
// `causal_gqa_attention` per position after the batch KV append.
// Used when FA-vec is off or head_dim lacks a specialized prefill kernel.
const GQA_PREFILL_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float online_rescale_prefill(float m_old, float m_new) {
    return (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
}

kernel void gqa_prefill(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_q [[buffer(7)]],
    constant uint& kv_prefix_len [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    uint h = tgpig.x;
    uint qi = tgpig.y;
    if (h >= n_heads || qi >= n_q) return;

    // Metal requires all thread-index attrs to share scalar vs vector shape.
    uint tid = tid_tg.x;
    uint tg = tg_size.x;
    uint causal_len = kv_prefix_len + qi + 1u;
    if (causal_len == 0u) return;

    threadgroup float* m_sh = shared;
    threadgroup float* s_sh = shared + tg;
    threadgroup float* acc_base = shared + 2u * tg;
    threadgroup float* my_acc = acc_base + tid * head_dim;

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(head_dim));
    device const float* q_h = q + (qi * n_heads + h) * head_dim;

    float m = -INFINITY;
    float s = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        my_acc[d] = 0.0f;
    }

    for (uint t = tid; t < causal_len; t += tg) {
        device const half* k_t =
            k_cache + (t * n_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_h[d] * float(k_t[d]);
        }
        float score = dot * scale;
        if (softcap > 0.0f) {
            score = softcap * tanh(score / softcap);
        }
        float m2 = max(m, score);
        float a = online_rescale_prefill(m, m2);
        float b = exp(score - m2);
        s = s * a + b;
        device const half* v_t =
            v_cache + (t * n_kv_heads + kv_h) * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            my_acc[d] = my_acc[d] * a + b * float(v_t[d]);
        }
        m = m2;
    }

    m_sh[tid] = m;
    s_sh[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            uint other = tid + stride;
            float m1 = m_sh[tid];
            float m2 = m_sh[other];
            float s1 = s_sh[tid];
            float s2 = s_sh[other];
            float m_new = max(m1, m2);
            float a1 = online_rescale_prefill(m1, m_new);
            float a2 = online_rescale_prefill(m2, m_new);
            m_sh[tid] = m_new;
            s_sh[tid] = s1 * a1 + s2 * a2;
            threadgroup float* acc1 = acc_base + tid * head_dim;
            threadgroup float* acc2 = acc_base + other * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                acc1[d] = acc1[d] * a1 + acc2[d] * a2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_s = 1.0f / s_sh[0];
    threadgroup float* acc0 = acc_base;
    device float* out_h = out + (qi * n_heads + h) * head_dim;
    for (uint d = tid; d < head_dim; d += tg) {
        out_h[d] = acc0[d] * inv_s;
    }
}
"#;

/// Growable Metal-resident KV for one layer (`[seq, n_kv, head_dim]`).
///
/// Default **f16** matches llama.cpp `-ctk f16`. With `FRINK_CTK=q8_0` and a
/// viable head layout, stores ggml Q8_0 (~½ the bytes); attention kernels still
/// read f16 via a process-wide dequant scratch shared across layers.
pub struct MetalKvBuffers {
    pub(crate) dtype: MetalKvDtype,
    /// Whether the stored K went through the Hadamard rotation.
    ///
    /// Decided once, at construction, by
    /// [`crate::kv_wire::q4_rotation_viable`], and read by exactly
    /// two places: the append, which rotates, and the attention sites,
    /// which rotate the query to match. A wire written one way and read
    /// the other is a wrong answer rather than an error, so there is one
    /// field and no second derivation of it.
    pub(crate) k_rotated: bool,
    pub(crate) k: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) v: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub(crate) capacity: usize,
}

// SAFETY: shared-mode MTLBuffers created once and mutated only from the
// decode thread that owns this cache (same justification as ResidentWeightBuffer).
unsafe impl Send for MetalKvBuffers {}
unsafe impl Sync for MetalKvBuffers {}

fn kv_store_nbytes(dtype: MetalKvDtype, elems: usize) -> Result<usize, MetalError> {
    match dtype {
        MetalKvDtype::F16 => Ok(elems * 2),
        d if d.is_q8_wire() => {
            if !elems.is_multiple_of(frink_quant::Q8_0_BLOCK_ELEMS) {
                return Err(MetalError::CommandFailed);
            }
            Ok((elems / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES)
        }
        MetalKvDtype::Q4_0 => {
            if !elems.is_multiple_of(frink_quant::Q4_KV_GROUP) {
                return Err(MetalError::CommandFailed);
            }
            Ok((elems / frink_quant::Q4_KV_GROUP) * frink_quant::Q4_KV_BLOCK_BYTES)
        }
        _ => Ok(elems * 2),
    }
}

impl MetalKvBuffers {
    pub fn with_capacity(
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self, MetalError> {
        let dtype = effective_metal_kv_dtype(n_kv_heads, head_dim);
        Self::with_capacity_dtype(n_kv_heads, head_dim, max_seq_len, dtype)
    }

    /// Allocate KV buffers with an explicit store dtype (tests / callers that
    /// bypass `FRINK_CTK`).
    pub fn with_capacity_dtype(
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: MetalKvDtype,
    ) -> Result<Self, MetalError> {
        let shared = shared_metal()?;
        let dtype = match dtype {
            d if d.is_q8_wire() && !metal_kv_q8_0_viable(n_kv_heads, head_dim) => {
                return Err(MetalError::CommandFailed);
            }
            MetalKvDtype::Q4_0 if !metal_kv_q4_viable(n_kv_heads, head_dim) => {
                return Err(MetalError::CommandFailed);
            }
            d if d.is_implemented() => d,
            _ => MetalKvDtype::F16,
        };
        let capacity = max_seq_len.max(1);
        let elems = capacity * n_kv_heads * head_dim;
        let nbytes = kv_store_nbytes(dtype, elems)?;
        let k = shared
            .device
            .newBufferWithLength_options(nbytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocFailed)?;
        let v = shared
            .device
            .newBufferWithLength_options(nbytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocFailed)?;
        Ok(Self {
            dtype,
            k_rotated: dtype == MetalKvDtype::Q4_0 && crate::kv_wire::q4_rotation_viable(head_dim),
            k,
            v,
            n_kv_heads,
            head_dim,
            seq_len: 0,
            capacity,
        })
    }

    pub fn dtype(&self) -> MetalKvDtype {
        self.dtype
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn elems_per_token(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Overwrites device K/V from host f32 caches (e.g. after CPU prefill),
    /// converting to the store dtype on the host.
    pub fn upload_from_host(
        &mut self,
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<(), MetalError> {
        assert_eq!(k.len(), seq_len * self.elems_per_token());
        assert_eq!(v.len(), seq_len * self.elems_per_token());
        if seq_len > self.capacity {
            return Err(MetalError::CommandFailed);
        }
        let n = seq_len * self.elems_per_token();
        match self.dtype {
            d if d.is_q8_wire() => {
                let k_q = frink_quant::quantize_q8_0(&k[..n]);
                let v_q = frink_quant::quantize_q8_0(&v[..n]);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_q.as_ptr(),
                        self.k.contents().as_ptr() as *mut u8,
                        k_q.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        v_q.as_ptr(),
                        self.v.contents().as_ptr() as *mut u8,
                        v_q.len(),
                    );
                }
            }
            MetalKvDtype::Q4_0 => {
                // The store holds K rotated, so host rows are rotated on
                // the way in exactly as `kv_append_q4` rotates the
                // ones the GPU writes. The pair with the unrotate in
                // `tokens_host` is what keeps a sequence that crosses
                // between the two paths reading the same K.
                let k_rot: Vec<f32>;
                let k_in: &[f32] = if self.k_rotated {
                    let mut owned = k[..n].to_vec();
                    for row in owned.chunks_exact_mut(self.elems_per_token()) {
                        frink_quant::kv_rotation::rotate_row_inplace(row, self.head_dim);
                    }
                    k_rot = owned;
                    &k_rot
                } else {
                    &k[..n]
                };
                let k_q = frink_quant::pack_q4_kv_blocks(k_in);
                let v_q = frink_quant::pack_q4_kv_blocks(&v[..n]);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_q.as_ptr(),
                        self.k.contents().as_ptr() as *mut u8,
                        k_q.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        v_q.as_ptr(),
                        self.v.contents().as_ptr() as *mut u8,
                        v_q.len(),
                    );
                }
            }
            _ => {
                let k_f16: Vec<u16> = k[..n]
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let v_f16: Vec<u16> = v[..n]
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let nbytes = n * 2;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_f16.as_ptr() as *const u8,
                        self.k.contents().as_ptr() as *mut u8,
                        nbytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        v_f16.as_ptr() as *const u8,
                        self.v.contents().as_ptr() as *mut u8,
                        nbytes,
                    );
                }
            }
        }
        self.seq_len = seq_len;
        Ok(())
    }

    /// Copies the last appended token's K/V to host f32 (after a completed CB).
    pub fn last_token_host(&self) -> (Vec<f32>, Vec<f32>) {
        assert!(self.seq_len > 0);
        let (k, v) = self.tokens_host(self.seq_len - 1, 1);
        (k, v)
    }

    /// Downloads `n` tokens starting at `start` as f32 (after a completed CB).
    pub fn tokens_host(&self, start: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(start + n <= self.seq_len);
        let per = self.elems_per_token();
        let off = start * per;
        let elems = n * per;
        match self.dtype {
            d if d.is_q8_wire() => {
                let nbytes =
                    (elems / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES;
                let byte_off =
                    (off / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES;
                let k_ptr = self.k.contents();
                let v_ptr = self.v.contents();
                let k_bytes = unsafe {
                    std::slice::from_raw_parts(k_ptr.as_ptr().add(byte_off) as *const u8, nbytes)
                };
                let v_bytes = unsafe {
                    std::slice::from_raw_parts(v_ptr.as_ptr().add(byte_off) as *const u8, nbytes)
                };
                (
                    frink_quant::dequant_q8_0(k_bytes).expect("q8 k aligned"),
                    frink_quant::dequant_q8_0(v_bytes).expect("q8 v aligned"),
                )
            }
            MetalKvDtype::Q4_0 => {
                let nbytes = (elems / frink_quant::Q4_KV_GROUP) * frink_quant::Q4_KV_BLOCK_BYTES;
                let byte_off = (off / frink_quant::Q4_KV_GROUP) * frink_quant::Q4_KV_BLOCK_BYTES;
                let k_ptr = self.k.contents();
                let v_ptr = self.v.contents();
                let k_bytes = unsafe {
                    std::slice::from_raw_parts(k_ptr.as_ptr().add(byte_off) as *const u8, nbytes)
                };
                let v_bytes = unsafe {
                    std::slice::from_raw_parts(v_ptr.as_ptr().add(byte_off) as *const u8, nbytes)
                };
                let mut k = frink_quant::unpack_q4_kv_blocks(k_bytes).expect("q4 k");
                // The device store holds K rotated; the host cache this
                // feeds is read by host kernels whose queries are not,
                // so the rotation is undone on the way out. This is the
                // one site that reads a rotated store as plain K.
                if self.k_rotated {
                    for row in k.chunks_exact_mut(per) {
                        frink_quant::kv_rotation::unrotate_row_inplace(row, self.head_dim);
                    }
                }
                (k, frink_quant::unpack_q4_kv_blocks(v_bytes).expect("q4 v"))
            }
            _ => {
                let k_ptr = self.k.contents();
                let v_ptr = self.v.contents();
                let k = unsafe {
                    std::slice::from_raw_parts(k_ptr.as_ptr() as *const u16, off + elems)
                };
                let v = unsafe {
                    std::slice::from_raw_parts(v_ptr.as_ptr() as *const u16, off + elems)
                };
                (
                    k[off..off + elems]
                        .iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .collect(),
                    v[off..off + elems]
                        .iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .collect(),
                )
            }
        }
    }
}

/// Process-wide f16 view of Q8_0 KV for FA/GQA (one pair shared across layers).
struct Q8AttnScratch {
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Rotated Q for a `k_rotated` store. f32, and sized by the query
    /// rather than the cache, so it rides on the same guard: the
    /// rotated K and the query that can read it are produced under one
    /// lock and cannot be taken apart.
    q: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_elems_cap: usize,
    elems_cap: usize,
}

// SAFETY: gated by Q8_ATTN_SCRATCH mutex; encode holds the lock for the CB encode.
unsafe impl Send for Q8AttnScratch {}

static Q8_ATTN_SCRATCH: Mutex<Option<Q8AttnScratch>> = Mutex::new(None);

fn borrow_q8_attn_scratch(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    elems: usize,
    q_elems: usize,
) -> Result<std::sync::MutexGuard<'static, Option<Q8AttnScratch>>, MetalError> {
    let mut guard = Q8_ATTN_SCRATCH.lock().unwrap();
    let fits = guard
        .as_ref()
        .is_some_and(|s| s.elems_cap >= elems && s.q_elems_cap >= q_elems);
    if !fits {
        let elems = elems.max(guard.as_ref().map_or(0, |s| s.elems_cap)).max(1);
        let q_elems = q_elems
            .max(guard.as_ref().map_or(0, |s| s.q_elems_cap))
            .max(1);
        let nbytes = elems * 2;
        *guard = Some(Q8AttnScratch {
            k: device
                .newBufferWithLength_options(nbytes, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
            v: device
                .newBufferWithLength_options(nbytes, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
            q: device
                .newBufferWithLength_options(q_elems * 4, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
            q_elems_cap: q_elems,
            elems_cap: elems,
        });
    }
    Ok(guard)
}

fn alloc_f32_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    device
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)
}

fn alloc_half_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    device
        .newBufferWithLength_options(n * 2, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)
}

fn alloc_u32_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    device
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)
}

pub(crate) fn upload_f32(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let mut owned = data.to_vec();
    unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(owned.as_mut_ptr() as *mut _).unwrap(),
            owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)
}

/// Upload host f32 as packed f16 (Metal KV / host-probe GQA).
fn upload_f16_from_f32(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let mut bits: Vec<u16> = data
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(bits.as_mut_ptr() as *mut _).unwrap(),
            bits.len() * 2,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)
}

pub(crate) fn copy_f32_into(buf: &ProtocolObject<dyn MTLBuffer>, data: &[f32]) {
    let nbytes = data.len() * 4;
    debug_assert!(buf.length() >= nbytes);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            nbytes,
        );
    }
}

/// Process-wide activation scratch for [`launch_decode_dense_stack`].
/// Avoids allocating ~10 MTLBuffers every decode token.
pub(crate) struct DecodeScratch {
    pub(crate) h: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) x: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) x2: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) q: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) k: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) v: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) attn: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) o: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) up: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) act: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) down: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) logits: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Single u32 slot for greedy argmax-in-stack (always resident; 4 bytes).
    pub(crate) argmax_idx: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// What `x` currently holds, when the dense stack has said so.
    ///
    /// Lives HERE, inside the thing the mutex protects, so the claim
    /// and the buffer it describes cannot be read apart. See
    /// [`crate::resident_act`] for what happened when it did not.
    pub(crate) resident: Option<crate::resident_act::ResidentPublication>,
    hidden_cap: usize,
    max_q_cap: usize,
    max_kv_cap: usize,
    attn_cap: usize,
    max_gate_cap: usize,
    logits_cap: usize,
}

// SAFETY: scratch is gated by DECODE_SCRATCH mutex; only one encode at a time.
unsafe impl Send for DecodeScratch {}

static DECODE_SCRATCH: Mutex<Option<DecodeScratch>> = Mutex::new(None);

/// Process-wide activation scratch for [`launch_prefill_dense_layer`].
/// Same residency idea as [`DecodeScratch`], sized for batch B≥4.
struct PrefillScratch {
    h: Retained<ProtocolObject<dyn MTLBuffer>>,
    x: Retained<ProtocolObject<dyn MTLBuffer>>,
    x2: Retained<ProtocolObject<dyn MTLBuffer>>,
    q: Retained<ProtocolObject<dyn MTLBuffer>>,
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn: Retained<ProtocolObject<dyn MTLBuffer>>,
    o: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    up: Retained<ProtocolObject<dyn MTLBuffer>>,
    down: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Reused f16 activation plane for `mul_mm_sg_f16` (max of hidden/q/gate).
    half_act: Retained<ProtocolObject<dyn MTLBuffer>>,
    half_act_cap: usize,
    batch_cap: usize,
    hidden_cap: usize,
    max_q_cap: usize,
    max_kv_cap: usize,
    max_gate_cap: usize,
}

// SAFETY: gated by PREFILL_SCRATCH mutex; one encode at a time.
unsafe impl Send for PrefillScratch {}

static PREFILL_SCRATCH: Mutex<Option<PrefillScratch>> = Mutex::new(None);

/// Shape key for a retained prefill command-buffer plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefillCbKey {
    pub layer: u32,
    pub batch: u32,
    pub hidden: u32,
    pub ffn: u32,
    pub q_rows: u32,
}

/// Shape key for a multi-layer prefill command-buffer plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefillStackCbKey {
    pub start_layer: u32,
    pub depth: u32,
    pub batch: u32,
    pub hidden: u32,
}

/// Stub cache for one-CB-per-layer encoding plans.
///
/// Partial step toward llama.cpp `ggml_metal_graph_compute`: retain encoded
/// `MTLCommandBuffer` templates (or ICB / graph nodes) keyed by
/// [`PrefillCbKey`] and replay with updated buffer bindings. Today we
/// record shape keys and which compute pipelines are warmed (compiled once
/// via [`MetalGraph::warm_prefill_pipelines`]) so the first dense-prefill
/// layer avoids repeated `ensure_pipeline` / Metal compile latency.
#[derive(Default, Debug)]
pub struct PrefillCbCache {
    keys: HashSet<PrefillCbKey>,
    stack_keys: HashSet<PrefillStackCbKey>,
    /// Kernel function names compiled and resident (process pipeline cache).
    hot_pipelines: HashSet<&'static str>,
}

impl PrefillCbCache {
    pub fn note(&mut self, key: PrefillCbKey) -> bool {
        self.keys.insert(key)
    }

    pub fn note_stack(&mut self, key: PrefillStackCbKey) -> bool {
        self.stack_keys.insert(key)
    }

    pub fn contains(&self, key: &PrefillCbKey) -> bool {
        self.keys.contains(key)
    }

    pub fn contains_stack(&self, key: &PrefillStackCbKey) -> bool {
        self.stack_keys.contains(key)
    }

    pub fn mark_pipeline_hot(&mut self, fn_name: &'static str) {
        self.hot_pipelines.insert(fn_name);
    }

    pub fn is_pipeline_hot(&self, fn_name: &str) -> bool {
        self.hot_pipelines.contains(fn_name)
    }

    pub fn hot_pipeline_count(&self) -> usize {
        self.hot_pipelines.len()
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Parameters for [`MetalGraph::warm_prefill_pipelines`].
pub struct PrefillWarmParams<'a> {
    pub layer: &'a PrefillDenseLayerMetal<'a>,
    pub rope_layout: MetalRope,
    pub head_dim: u32,
    pub gelu_ffn: bool,
    pub kv_dtype: MetalKvDtype,
}

/// Minimal Metal encode-plan holder (prefill CB cache; decode replay later).
#[derive(Default, Debug)]
pub struct MetalGraph {
    pub prefill: PrefillCbCache,
    prefill_pipelines_warmed: bool,
}

impl MetalGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn prefill_pipelines_warmed(&self) -> bool {
        self.prefill_pipelines_warmed
    }

    /// Compile mul_mm_sg, RMSNorm, RoPE, KV append/dequant, GQA, and FFN
    /// elementwise pipelines used by [`launch_prefill_dense_layer`]. Mirrors
    /// the pipeline residency half of llama.cpp `ggml_metal_graph_compute`
    /// (full CB / graph replay is still TODO).
    pub fn warm_prefill_pipelines(
        &mut self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        params: PrefillWarmParams<'_>,
    ) -> Result<(), MetalError> {
        let mut mark = |name: &'static str| self.prefill.mark_pipeline_hot(name);

        warm_prefill_elem_pipelines(device, params.gelu_ffn)?;
        mark("rms_norm_f32");
        mark("vec_add_f32");
        mark(if params.gelu_ffn {
            "gelu_mul_f32"
        } else {
            "silu_mul_f32"
        });

        let mut launches = vec![
            &params.layer.q,
            &params.layer.k,
            &params.layer.v,
            &params.layer.o,
        ];
        // MoE layers take `mul_mm_id`, which compiles on first encode.
        if let Some((gate, up, down)) = params.layer.ffn.dense() {
            launches.extend([gate, up, down]);
        }
        for launch in launches {
            warm_mul_mm_sg_pipeline(device, launch.fn_name)?;
            mark(launch.fn_name);
            let f16_static: &'static str = match launch.fn_name {
                "q4_k_mul_mm_sg" => "q4_k_mul_mm_sg_f16",
                "q5_k_mul_mm_sg" => "q5_k_mul_mm_sg_f16",
                "q6_k_mul_mm_sg" => "q6_k_mul_mm_sg_f16",
                "q8_0_mul_mm_sg" => "q8_0_mul_mm_sg_f16",
                "q4_0_mul_mm_sg" => "q4_0_mul_mm_sg_f16",
                "q5_0_mul_mm_sg" => "q5_0_mul_mm_sg_f16",
                "iq4_xs_mul_mm_sg" => "iq4_xs_mul_mm_sg_f16",
                _ => continue,
            };
            warm_mul_mm_sg_pipeline(device, f16_static)?;
            mark(f16_static);
            // The exact-tile (`bc_out=false`) siblings: compiled here too,
            // so a prefill whose rows/batch happen to tile exactly does not
            // pay a pipeline build inside its first command buffer.
            for base in [launch.fn_name, f16_static] {
                if let Some(a) = crate::gpu::mul_mm_sg_aligned_fn(base) {
                    warm_mul_mm_sg_pipeline(device, a)?;
                    mark(a);
                }
            }
        }

        let (rope_src, rope_name) = crate::rope::rope_kernel(params.rope_layout.layout);
        ensure_pipeline(device, rope_src, rope_name)?;
        mark(rope_name);

        for (src, name) in crate::kv_wire::kv_wire_pipelines(params.kv_dtype) {
            ensure_pipeline(device, src, name)?;
            mark(name);
        }

        warm_gqa_prefill_pipeline(device, params.head_dim, &mut mark)?;

        self.prefill_pipelines_warmed = true;
        Ok(())
    }
}

fn warm_gqa_prefill_pipeline(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    head_dim: u32,
    mark: &mut dyn FnMut(&'static str),
) -> Result<(), MetalError> {
    if metal_fa_vec_enabled() && gqa_prefill_fa_vec_supported(head_dim) {
        let (src, name) = match head_dim {
            64 => (GQA_PREFILL_FA_VEC_D64_KERNEL_SRC, "gqa_prefill_fa_vec_d64"),
            96 => (GQA_PREFILL_FA_VEC_D96_KERNEL_SRC, "gqa_prefill_fa_vec_d96"),
            128 => (GQA_PREFILL_FA_VEC_KERNEL_SRC, "gqa_prefill_fa_vec"),
            256 => (
                GQA_PREFILL_FA_VEC_D256_KERNEL_SRC,
                "gqa_prefill_fa_vec_d256",
            ),
            _ => return Err(MetalError::CommandFailed),
        };
        ensure_pipeline(device, src, name)?;
        mark(name);
    } else {
        ensure_pipeline(device, GQA_PREFILL_KERNEL_SRC, "gqa_prefill")?;
        mark("gqa_prefill");
    }
    Ok(())
}

static PREFILL_GRAPH: OnceLock<Mutex<MetalGraph>> = OnceLock::new();

/// Process-wide [`MetalGraph`] (stub). Intended for decode-stack replay
/// and future `ggml_metal_graph_compute`-style CB retention.
pub fn metal_graph() -> std::sync::MutexGuard<'static, MetalGraph> {
    PREFILL_GRAPH
        .get_or_init(|| Mutex::new(MetalGraph::new()))
        .lock()
        .unwrap()
}

pub(crate) struct ScratchCaps {
    pub(crate) hidden: usize,
    pub(crate) max_q: usize,
    pub(crate) max_kv: usize,
    pub(crate) attn: usize,
    pub(crate) max_gate: usize,
    pub(crate) logits: usize,
}

struct PrefillScratchCaps {
    batch: usize,
    hidden: usize,
    max_q: usize,
    max_kv: usize,
    max_gate: usize,
}

/// The decode scratch, if no one else is using it right now.
///
/// `try_lock` and not `lock`: the only caller outside the decode stack
/// is [`crate::resident_act`], whose whole contract is that a busy
/// scratch means "upload normally". Blocking there would mean waiting
/// out another thread's entire decode step, and it would invert a lock
/// order. Poisoning is treated as busy for the same reason.
pub(crate) fn try_lock_decode_scratch(
) -> Option<std::sync::MutexGuard<'static, Option<DecodeScratch>>> {
    DECODE_SCRATCH.try_lock().ok()
}

pub(crate) fn borrow_decode_scratch(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    caps: ScratchCaps,
) -> Result<std::sync::MutexGuard<'static, Option<DecodeScratch>>, MetalError> {
    let mut guard = DECODE_SCRATCH.lock().unwrap();
    // Handing out the guard is handing out the right to WRITE `x`, so
    // any standing claim about what `x` holds stops being true here.
    // This is the invalidation the old thread-local publication had no
    // way to express: it was not stored with the buffer.
    if let Some(scratch) = guard.as_mut() {
        scratch.resident = None;
    }
    let fits = match guard.as_ref() {
        Some(s) => {
            s.hidden_cap >= caps.hidden
                && s.max_q_cap >= caps.max_q
                && s.max_kv_cap >= caps.max_kv
                && s.attn_cap >= caps.attn
                && s.max_gate_cap >= caps.max_gate
                && s.logits_cap >= caps.logits
        }
        None => false,
    };
    if !fits {
        let logits = if caps.logits > 0 {
            Some(alloc_f32_buffer(device, caps.logits)?)
        } else {
            None
        };
        *guard = Some(DecodeScratch {
            h: alloc_f32_buffer(device, caps.hidden)?,
            x: alloc_f32_buffer(device, caps.hidden)?,
            x2: alloc_f32_buffer(device, caps.hidden)?,
            q: alloc_f32_buffer(device, caps.max_q)?,
            k: alloc_f32_buffer(device, caps.max_kv)?,
            v: alloc_f32_buffer(device, caps.max_kv)?,
            attn: alloc_f32_buffer(device, caps.attn)?,
            o: alloc_f32_buffer(device, caps.hidden)?,
            gate: alloc_f32_buffer(device, caps.max_gate)?,
            up: alloc_f32_buffer(device, caps.max_gate)?,
            act: alloc_f32_buffer(device, caps.max_gate)?,
            down: alloc_f32_buffer(device, caps.hidden)?,
            logits,
            argmax_idx: alloc_u32_buffer(device, 1)?,
            resident: None,
            hidden_cap: caps.hidden,
            max_q_cap: caps.max_q,
            max_kv_cap: caps.max_kv,
            attn_cap: caps.attn,
            max_gate_cap: caps.max_gate,
            logits_cap: caps.logits,
        });
    }
    Ok(guard)
}

fn borrow_prefill_scratch(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    caps: PrefillScratchCaps,
) -> Result<std::sync::MutexGuard<'static, Option<PrefillScratch>>, MetalError> {
    let mut guard = PREFILL_SCRATCH.lock().unwrap();
    let fits = match guard.as_ref() {
        Some(s) => {
            s.batch_cap >= caps.batch
                && s.hidden_cap >= caps.hidden
                && s.max_q_cap >= caps.max_q
                && s.max_kv_cap >= caps.max_kv
                && s.max_gate_cap >= caps.max_gate
                && s.half_act_cap >= caps.batch * caps.hidden.max(caps.max_q).max(caps.max_gate)
        }
        None => false,
    };
    if !fits {
        let bh = caps.batch * caps.hidden;
        let half_cap = caps.batch * caps.hidden.max(caps.max_q).max(caps.max_gate);
        *guard = Some(PrefillScratch {
            h: alloc_f32_buffer(device, bh)?,
            x: alloc_f32_buffer(device, bh)?,
            x2: alloc_f32_buffer(device, bh)?,
            q: alloc_f32_buffer(device, caps.batch * caps.max_q)?,
            k: alloc_f32_buffer(device, caps.batch * caps.max_kv)?,
            v: alloc_f32_buffer(device, caps.batch * caps.max_kv)?,
            attn: alloc_f32_buffer(device, caps.batch * caps.max_q)?,
            o: alloc_f32_buffer(device, bh)?,
            gate: alloc_f32_buffer(device, caps.batch * caps.max_gate)?,
            up: alloc_f32_buffer(device, caps.batch * caps.max_gate)?,
            // No f32 `act` plane: SwiGLU/GeGLU writes straight into
            // `half_act` for the down `mul_mm_sg_f16`, which also saves
            // `batch * max_gate * 4` bytes of scratch.
            down: alloc_f32_buffer(device, bh)?,
            half_act: alloc_half_buffer(device, half_cap)?,
            half_act_cap: half_cap,
            batch_cap: caps.batch,
            hidden_cap: caps.hidden,
            max_q_cap: caps.max_q,
            max_kv_cap: caps.max_kv,
            max_gate_cap: caps.max_gate,
        });
    }
    Ok(guard)
}

/// Decode GQA against the layer's KV cache, hazard-tracked through `mrs`.
/// Same shared-f16-scratch caveat as [`encode_gqa_prefill_with_kv`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gqa_with_kv(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    kv: &MetalKvBuffers,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_start: u32,
    softcap: Option<f32>,
) -> Result<(), MetalError> {
    if kv.dtype.needs_f16_scratch() {
        let elems = (seq_len as usize) * kv.elems_per_token();
        let q_elems = (n_heads as usize) * (head_dim as usize);
        let mut guard = borrow_q8_attn_scratch(device, elems, q_elems)?;
        let scratch = guard.as_mut().unwrap();
        let (sk, sv) = (scratch.k.as_ref(), scratch.v.as_ref());
        let (kk, vv) = (kv.k.as_ref(), kv.v.as_ref());
        mrs.begin_op(encoder, &[kk, vv], &[sk, sv]);
        encode_kv_dequant_to_f16(encoder, device, kv.dtype, &kv.k, &scratch.k, elems as u32)?;
        encode_kv_dequant_to_f16(encoder, device, kv.dtype, &kv.v, &scratch.v, elems as u32)?;
        mrs.end_op(&[kk, vv], &[sk, sv]);
        // A rotated store is only readable by a rotated query, and the
        // two are produced in the same block for that reason.
        let q = if kv.k_rotated {
            let sq = scratch.q.as_ref();
            mrs.begin_op(encoder, &[q], &[sq]);
            let r = crate::kv_wire::encode_rotate_q_q4(
                encoder, device, q, &scratch.q, 1, n_heads, n_kv_heads, head_dim,
            );
            mrs.end_op(&[q], &[sq]);
            r?;
            scratch.q.as_ref()
        } else {
            q
        };
        mrs.begin_op(encoder, &[q, sk, sv], &[out]);
        let res = encode_gqa(
            encoder, device, q, &scratch.k, &scratch.v, out, n_heads, n_kv_heads, head_dim,
            seq_len, kv_start, softcap,
        );
        mrs.end_op(&[q, sk, sv], &[out]);
        res
    } else {
        let (kk, vv) = (kv.k.as_ref(), kv.v.as_ref());
        mrs.begin_op(encoder, &[q, kk, vv], &[out]);
        let res = encode_gqa(
            encoder, device, q, &kv.k, &kv.v, out, n_heads, n_kv_heads, head_dim, seq_len,
            kv_start, softcap,
        );
        mrs.end_op(&[q, kk, vv], &[out]);
        res
    }
}

/// Prefill GQA against the layer's KV cache, hazard-tracked through `mrs`.
///
/// A quantized KV cache is dequantized into a *shared, process-wide* f16
/// scratch first. That scratch is the one buffer in the prefill layer that
/// is not visible to the caller, so the tracking has to happen in here:
/// the next layer's dequant writes it again (WAR against this layer's GQA
/// read) and nothing outside would know to order those.
#[allow(clippy::too_many_arguments)]
fn encode_gqa_prefill_with_kv(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    kv: &MetalKvBuffers,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_q: u32,
    kv_prefix_len: u32,
    attn_softcap: Option<f32>,
) -> Result<(), MetalError> {
    let total_seq = kv_prefix_len + n_q;
    if kv.dtype.needs_f16_scratch() {
        let elems = (total_seq as usize) * kv.elems_per_token();
        let q_elems = (n_q as usize) * (n_heads as usize) * (head_dim as usize);
        let mut guard = borrow_q8_attn_scratch(device, elems, q_elems)?;
        let scratch = guard.as_mut().unwrap();
        let (sk, sv) = (scratch.k.as_ref(), scratch.v.as_ref());
        let (kk, vv) = (kv.k.as_ref(), kv.v.as_ref());
        mrs.begin_op(encoder, &[kk, vv], &[sk, sv]);
        encode_kv_dequant_to_f16(encoder, device, kv.dtype, &kv.k, &scratch.k, elems as u32)?;
        encode_kv_dequant_to_f16(encoder, device, kv.dtype, &kv.v, &scratch.v, elems as u32)?;
        mrs.end_op(&[kk, vv], &[sk, sv]);
        let q = if kv.k_rotated {
            let sq = scratch.q.as_ref();
            mrs.begin_op(encoder, &[q], &[sq]);
            let r = crate::kv_wire::encode_rotate_q_q4(
                encoder, device, q, &scratch.q, n_q, n_heads, n_kv_heads, head_dim,
            );
            mrs.end_op(&[q], &[sq]);
            r?;
            scratch.q.as_ref()
        } else {
            q
        };
        // GQA must not race the dequant writes it just queued.
        mrs.begin_op(encoder, &[q, sk, sv], &[out]);
        let res = encode_gqa_prefill(
            encoder,
            device,
            q,
            &scratch.k,
            &scratch.v,
            out,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
            attn_softcap,
            PrefillAttnKernel::Auto,
        );
        mrs.end_op(&[q, sk, sv], &[out]);
        res
    } else {
        let (kk, vv) = (kv.k.as_ref(), kv.v.as_ref());
        mrs.begin_op(encoder, &[q, kk, vv], &[out]);
        let res = encode_gqa_prefill(
            encoder,
            device,
            q,
            &kv.k,
            &kv.v,
            out,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
            attn_softcap,
            PrefillAttnKernel::Auto,
        );
        mrs.end_op(&[q, kk, vv], &[out]);
        res
    }
}

/// TG size for decode GQA: multiple of 32 with **power-of-two** N_SG
/// (cross-SG tree reduce). Compact TG mem after per-SG register reduce:
/// `(2 * nsg + nsg * head_dim) * 4`.
fn gqa_decode_threadgroup_size(seq_len: u32, head_dim: u32) -> u32 {
    const TG_BUDGET_BYTES: u32 = 28 * 1024;
    const NW: u32 = 32;
    let per_sg = head_dim.saturating_add(2).saturating_mul(4).max(1);
    let max_nsg = (TG_BUDGET_BYTES / per_sg).clamp(1, 8);
    let raw = seq_len.div_ceil(NW).clamp(1, 4).min(max_nsg);
    // Power-of-two N_SG only (1/2/4) so the cross-SG tree reduce is complete.
    let nsg = if raw >= 4 && max_nsg >= 4 {
        4
    } else if raw >= 2 && max_nsg >= 2 {
        2
    } else {
        1
    };
    nsg * NW
}

/// Prefill FA-vec TG size. d=64 wastes half of each simdgroup on Q·K
/// (`D4=16` of 32 lanes), so prefer fewer simdgroups → more concurrent
/// TGs (one TG still owns one query). Measured on SmolLM2 Metal pp512.
fn gqa_prefill_fa_vec_threadgroup_size(head_dim: u32) -> u32 {
    match head_dim {
        // d=64: D4=16 → half of each SG idle on Q·K. Prefer 2 SG (64
        // threads) so more (head,query) TGs stay in flight on tiny pp512.
        64 => 64,
        96 => 128,
        _ => 256,
    }
}

/// Prefill GQA keeps the legacy per-thread TG-acc layout when FA-vec is
/// off or head_dim is unsupported; TG must be a power of two for its tree
/// reduce. Separate from decode so NeoX/prefill agents can evolve this
/// without fighting decode occupancy tweaks.
fn gqa_prefill_threadgroup_size(seq_len: u32, head_dim: u32) -> u32 {
    const TG_BUDGET_BYTES: u32 = 28 * 1024;
    let per_thread = head_dim.saturating_add(2).saturating_mul(4).max(1);
    let max_by_mem = (TG_BUDGET_BYTES / per_thread).max(1);
    let want = seq_len.clamp(32, 128).min(max_by_mem).max(1);
    let mut tg = 1u32 << (31 - want.leading_zeros());
    while tg > max_by_mem && tg > 1 {
        tg >>= 1;
    }
    tg.max(1)
}

/// FA-vec prefill head dims: 64 (SmolLM2 / TinyLlama / Llama-3.2-1B),
/// 96, 128 (Llama-3), 256 (Gemma-2/3). Anything else falls back to the
/// legacy per-thread-accumulator `gqa_prefill`.
fn gqa_prefill_fa_vec_supported(head_dim: u32) -> bool {
    matches!(head_dim, 64 | 96 | 128 | 256)
}

/// llama `kernel_flash_attn_ext` dispatch: QN=8, C=64, NSG=4 (128 threads/TG).
///
/// `head_dim` must be 64, 128 or 256 — the three widths the MMA kernel is
/// instantiated at.
#[allow(clippy::too_many_arguments)]
fn encode_gqa_prefill_fa_ext(
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
    const QN: u32 = 8;
    const C: u32 = 64;
    const NSG: u32 = 4;
    const SH: u32 = 2 * C;
    let d = head_dim;
    let pipe = match d {
        64 => ensure_pipeline(
            device,
            GQA_PREFILL_FA_EXT_MMA_D64_KERNEL_SRC,
            "gqa_prefill_fa_ext_mma_d64",
        )?,
        128 => ensure_pipeline(
            device,
            GQA_PREFILL_FA_EXT_MMA_D128_KERNEL_SRC,
            "gqa_prefill_fa_ext_mma_d128",
        )?,
        256 => ensure_pipeline(
            device,
            GQA_PREFILL_FA_EXT_MMA_D256_KERNEL_SRC,
            "gqa_prefill_fa_ext_mma_d256",
        )?,
        _ => return Err(MetalError::CommandFailed),
    };
    encoder.setComputePipelineState(&pipe.0);
    let tg = 32 * NSG;
    // sq[QN,D] + so[QN,D] + ss[QN,SH], plus kpad[8,D]+vpad[8,D] as f16 for the
    // ≤7-row cache tail. 10 KiB at d=64, 16 KiB at d=128, 28 KiB at d=256 —
    // the last width under Apple's 32 KiB threadgroup limit.
    let tg_mem = ((2 * QN * d + QN * SH) * 4) as usize + (2 * 8 * d * 2) as usize;
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
        let mut hd = d;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 6);
        let mut nq = n_q;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nq as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut prefix = kv_prefix_len;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut prefix as *mut u32 as *mut _).unwrap(),
            4,
            8,
        );
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_heads as usize,
            height: n_q.div_ceil(QN) as usize,
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

/// Which prefill attention kernel [`encode_gqa_prefill`] encodes.
///
/// Production only ever asks for [`PrefillAttnKernel::Auto`]: there is one
/// best kernel per shape and no reason to run a slower one. The variants
/// exist so the parity tests can name a kernel directly and check each
/// against the CPU reference and against its neighbour, rather than
/// steering the dispatch through an environment variable that would then
/// have to survive in shipped builds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrefillAttnKernel {
    /// Best kernel for the shape. The only thing production asks for.
    Auto,
    /// llama `flash_attn_ext`, simdgroup-MMA score + P·V. `head_dim` 64,
    /// 128 or 256 with `n_q >= 8`; other shapes fall through to FA-vec.
    FaExt,
    /// llama-style FA-vec, every `head_dim` with a specialized kernel.
    FaVec,
}

#[allow(clippy::too_many_arguments)]
fn encode_gqa_prefill_fa_vec(
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
    kernel: PrefillAttnKernel,
) -> Result<(), MetalError> {
    // llama flash_attn_ext (MMA Q·Kᵀ + P·V, QN=8/C=64) for d=64, d=128 and
    // d=256 prefill; every other width, and every batch under QN=8, stays on
    // FA-vec.
    if kernel != PrefillAttnKernel::FaVec && matches!(head_dim, 64 | 128 | 256) && n_q >= 8 {
        return encode_gqa_prefill_fa_ext(
            encoder,
            device,
            q,
            k,
            v,
            out,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
            softcap,
        );
    }
    let pipe = match head_dim {
        64 => ensure_pipeline(
            device,
            GQA_PREFILL_FA_VEC_D64_KERNEL_SRC,
            "gqa_prefill_fa_vec_d64",
        )?,
        96 => ensure_pipeline(
            device,
            GQA_PREFILL_FA_VEC_D96_KERNEL_SRC,
            "gqa_prefill_fa_vec_d96",
        )?,
        128 => ensure_pipeline(device, GQA_PREFILL_FA_VEC_KERNEL_SRC, "gqa_prefill_fa_vec")?,
        256 => ensure_pipeline(
            device,
            GQA_PREFILL_FA_VEC_D256_KERNEL_SRC,
            "gqa_prefill_fa_vec_d256",
        )?,
        _ => return Err(MetalError::CommandFailed),
    };
    encoder.setComputePipelineState(&pipe.0);
    let tg = gqa_prefill_fa_vec_threadgroup_size(head_dim);
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
        let mut nq = n_q;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nq as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut prefix = kv_prefix_len;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut prefix as *mut u32 as *mut _).unwrap(),
            4,
            8,
        );
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_heads as usize,
            height: n_q as usize,
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

/// [`encode_gqa`] for `crate::kernel_bench`, which times the kernel the
/// decode stack picks for a shape without a `MetalKvBuffers`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gqa_for_bench(
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
    attn_softcap: Option<f32>,
) -> Result<(), MetalError> {
    encode_gqa(
        encoder,
        device,
        q,
        k,
        v,
        out,
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        kv_start,
        attn_softcap,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_gqa(
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
    attn_softcap: Option<f32>,
) -> Result<(), MetalError> {
    let softcap = attn_softcap.filter(|&c| c > 0.0).unwrap_or(0.0);
    // FA-vec supports SWA (`kv_start`) and softcap; use it whenever
    // head_dim has a specialized kernel.
    if metal_fa_vec_enabled() && gqa_fa_vec_supported(head_dim) {
        return encode_gqa_fa_vec(
            encoder, device, q, k, v, out, n_heads, n_kv_heads, head_dim, seq_len, kv_start,
            softcap,
        );
    }
    if head_dim > 256 {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, GQA_DECODE_KERNEL_SRC, "gqa_decode")?;
    encoder.setComputePipelineState(&pipe.0);
    let tg = gqa_decode_threadgroup_size(seq_len, head_dim);
    let nsg = tg / 32;
    // m[nsg] + s[nsg] + acc[nsg * head_dim]
    let tg_mem = ((2 * nsg + nsg * head_dim) * 4) as usize;
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

#[allow(clippy::too_many_arguments)]
fn encode_gqa_prefill(
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
    attn_softcap: Option<f32>,
    kernel: PrefillAttnKernel,
) -> Result<(), MetalError> {
    let softcap = attn_softcap.filter(|&c| c > 0.0).unwrap_or(0.0);
    if metal_fa_vec_enabled() && gqa_prefill_fa_vec_supported(head_dim) {
        return encode_gqa_prefill_fa_vec(
            encoder,
            device,
            q,
            k,
            v,
            out,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
            softcap,
            kernel,
        );
    }
    let pipe = ensure_pipeline(device, GQA_PREFILL_KERNEL_SRC, "gqa_prefill")?;
    encoder.setComputePipelineState(&pipe.0);
    let max_causal = kv_prefix_len + n_q;
    let tg = gqa_prefill_threadgroup_size(max_causal.max(1), head_dim);
    // Prefill kernel: per-thread TG acc — m[tg]|s[tg]|acc[tg*head_dim].
    let tg_mem = ((2 * tg + tg * head_dim) * 4) as usize;
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
        let mut nq = n_q;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nq as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut prefix = kv_prefix_len;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut prefix as *mut u32 as *mut _).unwrap(),
            4,
            8,
        );
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_heads as usize,
            height: n_q as usize,
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

/// Encode the optional QKV bias adds + QK-RMSNorms (CPU-path order:
/// bias → norm → RoPE). Per-head (`weight.len() == head_dim`) or
/// whole-vector (`weight.len() == q_rows` / `k_rows`, OLMoE). No-ops
/// when `extras` is empty. Single-token path used by decode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attn_extras(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    extras: &AttnExtras<'_>,
    q_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    k_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    v_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    q_rows: usize,
    k_rows: usize,
    v_rows: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) -> Result<(), MetalError> {
    let resident = resident_attn_extras(device, extras)?;
    encode_attn_extras_batch(
        encoder, device, extras, q_buf, k_buf, v_buf, q_rows, k_rows, v_rows, n_heads, n_kv_heads,
        head_dim, 1, rms_eps, &resident,
    )
}

/// Prefill (`batch ≥ 1`) extras using already-resident bias/norm buffers.
#[allow(clippy::too_many_arguments)]
fn encode_attn_extras_batch(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    extras: &AttnExtras<'_>,
    q_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    k_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    v_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    q_rows: usize,
    k_rows: usize,
    v_rows: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    batch: usize,
    rms_eps: f32,
    resident: &AttnExtrasResident,
) -> Result<(), MetalError> {
    if batch == 0 {
        return Ok(());
    }
    if let Some(bb) = resident.q_bias.as_ref() {
        debug_assert_eq!(extras.q_bias.map(|b| b.len()), Some(q_rows));
        for t in 0..batch {
            encode_vec_add_at(
                encoder,
                device,
                q_buf,
                t * q_rows * 4,
                &bb.buffer,
                q_rows as u32,
            )?;
        }
    }
    if let Some(bb) = resident.k_bias.as_ref() {
        debug_assert_eq!(extras.k_bias.map(|b| b.len()), Some(k_rows));
        for t in 0..batch {
            encode_vec_add_at(
                encoder,
                device,
                k_buf,
                t * k_rows * 4,
                &bb.buffer,
                k_rows as u32,
            )?;
        }
    }
    if let Some(bb) = resident.v_bias.as_ref() {
        debug_assert_eq!(extras.v_bias.map(|b| b.len()), Some(v_rows));
        for t in 0..batch {
            encode_vec_add_at(
                encoder,
                device,
                v_buf,
                t * v_rows * 4,
                &bb.buffer,
                v_rows as u32,
            )?;
        }
    }
    // Q/K norm style is inferred from the weight LENGTH, and that is
    // sound rather than a guess: `loader.rs`'s `refined_qk_norm` derives
    // `ModelConfig::qk_norm_style` from exactly this rule
    // (`len == head_dim` is PerHead, `len == n_heads * head_dim` is
    // WholeVector, anything else is a load error), so the host enum and
    // this length check cannot disagree about a checkpoint that loaded.
    // `AttnExtras` therefore does not carry the discriminator.
    //
    // The two rules agreeing is pinned host-side by
    // `the_metal_qk_norm_length_rule_is_the_one_the_loader_derives_the_style_from`.
    if let (Some(w), Some(wb)) = (extras.q_norm, resident.q_norm.as_ref()) {
        if w.len() == head_dim {
            encode_rms_norm_per_head_batch(
                encoder,
                device,
                q_buf,
                &wb.buffer,
                n_heads as u32,
                head_dim as u32,
                batch as u32,
                rms_eps,
            )?;
        } else {
            debug_assert_eq!(w.len(), q_rows, "Q norm weight must be head_dim or q_rows");
            for t in 0..batch {
                let off = t * q_rows * 4;
                encode_rms_norm_at(
                    encoder,
                    device,
                    q_buf,
                    off,
                    &wb.buffer,
                    q_buf,
                    off,
                    q_rows as u32,
                    rms_eps,
                )?;
            }
        }
    }
    if let (Some(w), Some(wb)) = (extras.k_norm, resident.k_norm.as_ref()) {
        if w.len() == head_dim {
            encode_rms_norm_per_head_batch(
                encoder,
                device,
                k_buf,
                &wb.buffer,
                n_kv_heads as u32,
                head_dim as u32,
                batch as u32,
                rms_eps,
            )?;
        } else {
            debug_assert_eq!(w.len(), k_rows, "K norm weight must be head_dim or k_rows");
            for t in 0..batch {
                let off = t * k_rows * 4;
                encode_rms_norm_at(
                    encoder,
                    device,
                    k_buf,
                    off,
                    &wb.buffer,
                    k_buf,
                    off,
                    k_rows as u32,
                    rms_eps,
                )?;
            }
        }
    }
    Ok(())
}

struct AttnExtrasResident {
    q_bias: Option<std::sync::Arc<ResidentF32Buffer>>,
    k_bias: Option<std::sync::Arc<ResidentF32Buffer>>,
    v_bias: Option<std::sync::Arc<ResidentF32Buffer>>,
    q_norm: Option<std::sync::Arc<ResidentF32Buffer>>,
    k_norm: Option<std::sync::Arc<ResidentF32Buffer>>,
}

fn resident_attn_extras(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    extras: &AttnExtras<'_>,
) -> Result<AttnExtrasResident, MetalError> {
    Ok(AttnExtrasResident {
        q_bias: extras
            .q_bias
            .map(|b| resident_f32_buffer(device, b))
            .transpose()?,
        k_bias: extras
            .k_bias
            .map(|b| resident_f32_buffer(device, b))
            .transpose()?,
        v_bias: extras
            .v_bias
            .map(|b| resident_f32_buffer(device, b))
            .transpose()?,
        q_norm: extras
            .q_norm
            .map(|b| resident_f32_buffer(device, b))
            .transpose()?,
        k_norm: extras
            .k_norm
            .map(|b| resident_f32_buffer(device, b))
            .transpose()?,
    })
}

/// Fused Q/K/V matvec → RoPE → KV append → GQA → O matvec.
/// Returns the O-projection output on the host. Updates `kv.seq_len`.
#[allow(clippy::too_many_arguments)]
pub fn launch_decode_attn_block(
    x: &[f32],
    q_launch: &MatvecLaunch<'_>,
    k_launch: &MatvecLaunch<'_>,
    v_launch: &MatvecLaunch<'_>,
    o_launch: &MatvecLaunch<'_>,
    kv: &mut MetalKvBuffers,
    n_heads: usize,
    rope_layout: MetalRope,
    // This layer's rotation, BOTH halves in one value. Two loose
    // parameters here is how a per-layer base ended up beside a
    // stack-wide divisor set at four call sites; see `LayerRope`.
    rope: LayerRope<'_>,
    pos: usize,
    extras: &AttnExtras<'_>,
    rms_eps: f32,
) -> Result<Vec<f32>, MetalError> {
    // Exhaustive destructure, no `..`: adding a third half to a
    // layer's rotation must break every launch that ropes.
    let LayerRope {
        theta: rope_theta,
        freq_factors,
    } = rope;
    let head_dim = kv.head_dim;
    let n_kv_heads = kv.n_kv_heads;
    assert_eq!(q_launch.rows, n_heads * head_dim);
    assert_eq!(k_launch.rows, n_kv_heads * head_dim);
    assert_eq!(v_launch.rows, n_kv_heads * head_dim);
    assert_eq!(o_launch.rows, n_heads * head_dim);
    assert_eq!(
        pos, kv.seq_len,
        "decode pos must equal current Metal KV length"
    );
    if kv.seq_len >= kv.capacity {
        return Err(MetalError::CommandFailed);
    }
    assert_freq_factors_len(freq_factors, rope_layout, head_dim);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let x_buf = upload_f32(device, x)?;
    let q_w = resident_weight_buffer(device, q_launch.weights)?;
    let k_w = resident_weight_buffer(device, k_launch.weights)?;
    let v_w = resident_weight_buffer(device, v_launch.weights)?;
    let o_w = resident_weight_buffer(device, o_launch.weights)?;

    let q_buf = alloc_f32_buffer(device, q_launch.rows)?;
    let k_buf = alloc_f32_buffer(device, k_launch.rows)?;
    let v_buf = alloc_f32_buffer(device, v_launch.rows)?;
    let attn_buf = alloc_f32_buffer(device, n_heads * head_dim)?;
    let o_buf = alloc_f32_buffer(device, o_launch.rows)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    encode_matvec(&encoder, device, q_launch, &q_w, &x_buf, &q_buf)?;
    encode_matvec(&encoder, device, k_launch, &k_w, &x_buf, &k_buf)?;
    encode_matvec(&encoder, device, v_launch, &v_w, &x_buf, &v_buf)?;
    encode_attn_extras(
        &encoder,
        device,
        extras,
        &q_buf,
        &k_buf,
        &v_buf,
        q_launch.rows,
        k_launch.rows,
        v_launch.rows,
        n_heads,
        n_kv_heads,
        head_dim,
        rms_eps,
    )?;

    encode_rope(
        &encoder,
        device,
        rope_layout,
        RopeTarget {
            vecs: &q_buf,
            n_heads: n_heads as u32,
        },
        Some(RopeTarget {
            vecs: &k_buf,
            n_heads: n_kv_heads as u32,
        }),
        head_dim as u32,
        rope_theta,
        pos as u32,
        ff_buf.as_deref(),
    )?;

    let token_elems = (n_kv_heads * head_dim) as u32;
    let offset = (kv.seq_len * n_kv_heads * head_dim) as u32;
    encode_kv_store_append(&encoder, device, &k_buf, &v_buf, kv, offset, token_elems)?;

    let new_seq = (kv.seq_len + 1) as u32;
    encode_gqa_with_kv(
        &encoder,
        &mut MemRanges::new(),
        device,
        &q_buf,
        kv,
        &attn_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        new_seq,
        0,
        extras.attn_logit_softcap,
    )?;

    // O matvec reads attn_buf as activation `x`.
    encode_matvec(&encoder, device, o_launch, &o_w, &attn_buf, &o_buf)?;

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    kv.seq_len += 1;

    let out_ptr = o_buf.contents();
    let out = unsafe {
        std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, o_launch.rows).to_vec()
    };
    Ok(out)
}

/// Retained Metal buffers for MoE decode so residual stays on GPU across
/// layers. With packed-id MoE, host never sees activations mid-layer.
struct MoeDecodeScratch {
    h: Retained<ProtocolObject<dyn MTLBuffer>>,
    x_attn: Retained<ProtocolObject<dyn MTLBuffer>>,
    x2: Retained<ProtocolObject<dyn MTLBuffer>>,
    q: Retained<ProtocolObject<dyn MTLBuffer>>,
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn: Retained<ProtocolObject<dyn MTLBuffer>>,
    o: Retained<ProtocolObject<dyn MTLBuffer>>,
    router: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// One routed-expert id slot per stack layer, so every layer's
    /// selection survives the single-command-buffer fold instead of being
    /// overwritten by the next layer (see [`crate::moe_ids`]).
    ids: MoeIdsLog,
    route: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Pre-SiLU gate projection (unfused MoE matvec_id).
    gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Pre-SiLU up projection (unfused MoE matvec_id).
    up: Retained<ProtocolObject<dyn MTLBuffer>>,
    act: Retained<ProtocolObject<dyn MTLBuffer>>,
    expert_out: Retained<ProtocolObject<dyn MTLBuffer>>,
    moe_out: Retained<ProtocolObject<dyn MTLBuffer>>,
    logits: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    argmax_idx: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_dim: usize,
    q_rows: usize,
    k_rows: usize,
    ffn_rows: usize,
    top_k_cap: usize,
    n_router: usize,
    logits_cap: usize,
}

thread_local! {
    static MOE_SCRATCH: RefCell<Option<MoeDecodeScratch>> = const { RefCell::new(None) };
}

/// Ensure MoE decode scratch exists for `hidden_dim` (no host upload).
pub fn moe_decode_ensure(hidden_dim: usize) -> Result<(), MetalError> {
    let shared = shared_metal()?;
    let device = &shared.device;
    MOE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let need_new = match slot.as_ref() {
            None => true,
            Some(s) => s.hidden_dim != hidden_dim,
        };
        if need_new {
            // Caps match OLMoE-class decode (top-k≤8, ffn≤8×hidden, ≤256 experts).
            // Grown on demand in phase-1/2 if a layer exceeds them.
            let q_rows = hidden_dim * 2;
            let k_rows = hidden_dim;
            let ffn_rows = hidden_dim * 8;
            let top_k_cap = 8;
            let n_router = 256;
            *slot = Some(MoeDecodeScratch {
                h: alloc_f32_buffer(device, hidden_dim)?,
                x_attn: alloc_f32_buffer(device, hidden_dim)?,
                x2: alloc_f32_buffer(device, hidden_dim)?,
                q: alloc_f32_buffer(device, q_rows)?,
                k: alloc_f32_buffer(device, k_rows)?,
                v: alloc_f32_buffer(device, k_rows)?,
                attn: alloc_f32_buffer(device, q_rows)?,
                o: alloc_f32_buffer(device, hidden_dim)?,
                router: alloc_f32_buffer(device, n_router)?,
                ids: MoeIdsLog::new(device, top_k_cap, 1)?,
                route: alloc_f32_buffer(device, top_k_cap)?,
                gate: alloc_f32_buffer(device, top_k_cap * ffn_rows)?,
                up: alloc_f32_buffer(device, top_k_cap * ffn_rows)?,
                act: alloc_f32_buffer(device, top_k_cap * ffn_rows)?,
                expert_out: alloc_f32_buffer(device, top_k_cap * hidden_dim)?,
                moe_out: alloc_f32_buffer(device, hidden_dim)?,
                logits: None,
                argmax_idx: alloc_u32_buffer(device, 1)?,
                hidden_dim,
                q_rows,
                k_rows,
                ffn_rows,
                top_k_cap,
                n_router,
                logits_cap: 0,
            });
        }
        Ok(())
    })
}

/// Seed residual `h` from host hidden (call once before the MoE layer loop).
pub fn moe_decode_seed(hidden: &[f32]) -> Result<(), MetalError> {
    moe_decode_ensure(hidden.len())?;
    MOE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let scratch = slot.as_mut().ok_or(MetalError::CommandFailed)?;
        let hidden_dim = scratch.hidden_dim;
        let dst = scratch.h.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(hidden.as_ptr(), dst.as_ptr() as *mut f32, hidden_dim);
        }
        Ok(())
    })
}

/// Download residual hidden. Keeps scratch buffers for the next token
/// (`moe_decode_seed` overwrites `h`). `None` if never seeded.
pub fn moe_decode_take_hidden() -> Option<Vec<f32>> {
    MOE_SCRATCH.with(|cell| {
        let slot = cell.borrow();
        let scratch = slot.as_ref()?;
        let ptr = scratch.h.contents();
        Some(unsafe {
            std::slice::from_raw_parts(ptr.as_ptr() as *const f32, scratch.hidden_dim).to_vec()
        })
    })
}

fn moe_scratch_ensure_caps(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    scratch: &mut MoeDecodeScratch,
    q_rows: usize,
    k_rows: usize,
    ffn_rows: usize,
    top_k: usize,
    n_router: usize,
) -> Result<(), MetalError> {
    if q_rows > scratch.q_rows {
        scratch.q = alloc_f32_buffer(device, q_rows)?;
        scratch.attn = alloc_f32_buffer(device, q_rows)?;
        scratch.q_rows = q_rows;
    }
    if k_rows > scratch.k_rows {
        scratch.k = alloc_f32_buffer(device, k_rows)?;
        scratch.v = alloc_f32_buffer(device, k_rows)?;
        scratch.k_rows = k_rows;
    }
    if n_router > scratch.n_router {
        scratch.router = alloc_f32_buffer(device, n_router)?;
        scratch.n_router = n_router;
    }
    if ffn_rows > scratch.ffn_rows || top_k > scratch.top_k_cap {
        let fk = ffn_rows.max(scratch.ffn_rows);
        let tk = top_k.max(scratch.top_k_cap);
        scratch.gate = alloc_f32_buffer(device, tk * fk)?;
        scratch.up = alloc_f32_buffer(device, tk * fk)?;
        scratch.act = alloc_f32_buffer(device, tk * fk)?;
        scratch.expert_out = alloc_f32_buffer(device, tk * scratch.hidden_dim)?;
        scratch.route = alloc_f32_buffer(device, tk)?;
        scratch.ffn_rows = fk;
        scratch.top_k_cap = tk;
    }
    Ok(())
}

fn moe_scratch_ensure_logits(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    scratch: &mut MoeDecodeScratch,
    vocab: usize,
) -> Result<(), MetalError> {
    if vocab > scratch.logits_cap {
        scratch.logits = Some(alloc_f32_buffer(device, vocab)?);
        scratch.logits_cap = vocab;
    }
    Ok(())
}

/// Phase 1 (llama mul_mat_id graph style): on-device
/// `rms_attn → QKV→RoPE→KV→GQA→O → h+=o → rms_ffn → router`.
/// Downloads **only** router logits. Residual + FFN-normed stay resident.
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_decode_pre(
    attn_norm_w: &[f32],
    q_launch: &MatvecLaunch<'_>,
    k_launch: &MatvecLaunch<'_>,
    v_launch: &MatvecLaunch<'_>,
    o_launch: &MatvecLaunch<'_>,
    kv: &mut MetalKvBuffers,
    ffn_norm_w: &[f32],
    router_launch: &MatvecLaunch<'_>,
    n_heads: usize,
    rope_layout: MetalRope,
    // This layer's rotation, BOTH halves in one value. Two loose
    // parameters here is how a per-layer base ended up beside a
    // stack-wide divisor set at four call sites; see `LayerRope`.
    rope: LayerRope<'_>,
    pos: usize,
    rms_eps: f32,
    extras: &AttnExtras<'_>,
) -> Result<Vec<f32>, MetalError> {
    // Exhaustive destructure, no `..`: adding a third half to a
    // layer's rotation must break every launch that ropes.
    let LayerRope {
        theta: rope_theta,
        freq_factors,
    } = rope;
    let head_dim = kv.head_dim;
    let n_kv_heads = kv.n_kv_heads;
    let hidden_dim = attn_norm_w.len();
    assert_eq!(ffn_norm_w.len(), hidden_dim);
    assert_eq!(q_launch.rows, n_heads * head_dim);
    assert_eq!(k_launch.rows, n_kv_heads * head_dim);
    assert_eq!(v_launch.rows, n_kv_heads * head_dim);
    assert_eq!(o_launch.rows, hidden_dim);
    assert_eq!(pos, kv.seq_len);
    if kv.seq_len >= kv.capacity {
        return Err(MetalError::CommandFailed);
    }
    assert_freq_factors_len(freq_factors, rope_layout, head_dim);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    MOE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let scratch = slot.as_mut().ok_or(MetalError::CommandFailed)?;
        if scratch.hidden_dim != hidden_dim {
            return Err(MetalError::CommandFailed);
        }
        moe_scratch_ensure_caps(
            device,
            scratch,
            q_launch.rows,
            k_launch.rows,
            1, // ffn sized in experts phase
            1,
            router_launch.rows,
        )?;

        let attn_nw = resident_f32_buffer(device, attn_norm_w)?;
        let ffn_nw = resident_f32_buffer(device, ffn_norm_w)?;
        let q_w = resident_weight_buffer(device, q_launch.weights)?;
        let k_w = resident_weight_buffer(device, k_launch.weights)?;
        let v_w = resident_weight_buffer(device, v_launch.weights)?;
        let o_w = resident_weight_buffer(device, o_launch.weights)?;
        let r_w = resident_weight_buffer(device, router_launch.weights)?;
        let ff_buf = match freq_factors {
            Some(ff) => Some(upload_f32(device, ff)?),
            None => None,
        };

        let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?;

        encode_rms_norm(
            &encoder,
            device,
            &scratch.h,
            &attn_nw.buffer,
            &scratch.x_attn,
            hidden_dim as u32,
            rms_eps,
        )?;
        encode_matvec(
            &encoder,
            device,
            q_launch,
            &q_w,
            &scratch.x_attn,
            &scratch.q,
        )?;
        encode_matvec(
            &encoder,
            device,
            k_launch,
            &k_w,
            &scratch.x_attn,
            &scratch.k,
        )?;
        encode_matvec(
            &encoder,
            device,
            v_launch,
            &v_w,
            &scratch.x_attn,
            &scratch.v,
        )?;
        encode_attn_extras(
            &encoder,
            device,
            extras,
            &scratch.q,
            &scratch.k,
            &scratch.v,
            q_launch.rows,
            k_launch.rows,
            v_launch.rows,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_eps,
        )?;
        encode_rope(
            &encoder,
            device,
            rope_layout,
            RopeTarget {
                vecs: &scratch.q,
                n_heads: n_heads as u32,
            },
            Some(RopeTarget {
                vecs: &scratch.k,
                n_heads: n_kv_heads as u32,
            }),
            head_dim as u32,
            rope_theta,
            pos as u32,
            ff_buf.as_deref(),
        )?;
        let token_elems = (n_kv_heads * head_dim) as u32;
        let offset = (kv.seq_len * n_kv_heads * head_dim) as u32;
        encode_kv_store_append(
            &encoder,
            device,
            &scratch.k,
            &scratch.v,
            kv,
            offset,
            token_elems,
        )?;
        let new_seq = (kv.seq_len + 1) as u32;
        encode_gqa_with_kv(
            &encoder,
            &mut MemRanges::new(),
            device,
            &scratch.q,
            kv,
            &scratch.attn,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            new_seq,
            0,
            extras.attn_logit_softcap,
        )?;
        encode_matvec(&encoder, device, o_launch, &o_w, &scratch.attn, &scratch.o)?;
        encode_add_rms_norm(
            &encoder,
            device,
            &scratch.h,
            &scratch.o,
            &ffn_nw.buffer,
            &scratch.x2,
            hidden_dim as u32,
            rms_eps,
        )?;
        encode_matvec(
            &encoder,
            device,
            router_launch,
            &r_w,
            &scratch.x2,
            &scratch.router,
        )?;

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        kv.seq_len += 1;

        let ptr = scratch.router.contents();
        let logits = unsafe {
            std::slice::from_raw_parts(ptr.as_ptr() as *const f32, router_launch.rows).to_vec()
        };
        Ok(logits)
    })
}

/// Phase 2: batched MoE SwiGLU on resident FFN-normed `x2`, then `h += moe`.
/// Prefers the Q4_0 top-k kernels (≤8 experts); otherwise falls back to
/// host [`crate::gpu::launch_moe_topk_swiglu`] + upload/add (slower).
pub fn launch_moe_decode_experts(experts: &[MoeExpertLaunch<'_>]) -> Result<(), MetalError> {
    if experts.is_empty() {
        return Ok(());
    }
    let hidden = experts[0].down.rows;
    let ffn = experts[0].gate.rows;
    let q4_0_batched = experts.len() <= 8
        && experts.iter().all(|ex| {
            ex.gate.fn_name == "q4_0_matvec"
                && ex.up.fn_name == "q4_0_matvec"
                && ex.down.fn_name == "q4_0_matvec"
                && ex.gate.block_bytes == 18
                && ex.up.block_bytes == 18
                && ex.down.block_bytes == 18
        });

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    MOE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let scratch = slot.as_mut().ok_or(MetalError::CommandFailed)?;
        if scratch.hidden_dim != hidden {
            return Err(MetalError::CommandFailed);
        }

        if q4_0_batched {
            moe_scratch_ensure_caps(
                device,
                scratch,
                scratch.q_rows,
                scratch.k_rows,
                ffn,
                experts.len(),
                scratch.n_router,
            )?;
            let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(MetalError::CommandFailed)?;
            encode_q4_0_moe_topk(
                &encoder,
                device,
                &scratch.x2,
                experts,
                &scratch.act,
                &scratch.expert_out,
                &scratch.moe_out,
            )?;
            encode_vec_add(
                &encoder,
                device,
                &scratch.h,
                &scratch.moe_out,
                hidden as u32,
            )?;
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            return Ok(());
        }

        // Generic path: download x2, run existing fuse, upload + add.
        let x2 = unsafe {
            std::slice::from_raw_parts(scratch.x2.contents().as_ptr() as *const f32, hidden)
                .to_vec()
        };
        drop(slot);
        let moe = crate::gpu::launch_moe_topk_swiglu(&x2, experts)?;
        MOE_SCRATCH.with(|cell| {
            let slot = cell.borrow();
            let scratch = slot.as_ref().ok_or(MetalError::CommandFailed)?;
            let moe_buf = upload_f32(device, &moe)?;
            let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(MetalError::CommandFailed)?;
            encode_vec_add(&encoder, device, &scratch.h, &moe_buf, hidden as u32)?;
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            Ok(())
        })
    })
}

/// One MoE layer's launches for [`launch_moe_decode_stack`].
pub struct MoeLayerMetal<'a> {
    pub attn_norm_w: &'a [f32],
    pub ffn_norm_w: &'a [f32],
    pub q: MatvecLaunch<'a>,
    pub k: MatvecLaunch<'a>,
    pub v: MatvecLaunch<'a>,
    pub o: MatvecLaunch<'a>,
    pub router: MatvecLaunch<'a>,
    pub packed: MoePackedQ4<'a>,
    pub extras: AttnExtras<'a>,
}

/// Pre-bound MTLBuffers for one MoE layer (llama: bind weights once).
struct MoeLayerResident {
    attn_nw: std::sync::Arc<ResidentF32Buffer>,
    ffn_nw: std::sync::Arc<ResidentF32Buffer>,
    q_w: std::sync::Arc<ResidentWeightBuffer>,
    k_w: std::sync::Arc<ResidentWeightBuffer>,
    v_w: std::sync::Arc<ResidentWeightBuffer>,
    o_w: std::sync::Arc<ResidentWeightBuffer>,
    r_w: std::sync::Arc<ResidentWeightBuffer>,
}

/// Seven cache lookups, exactly as the dense stack does per layer per
/// token.
///
/// This used to memoise the seven behind a thread-local keyed on
/// `attn_norm_w.as_ptr()` -- an address, with no length beside it and
/// no check that the bytes were still that layer's. It was a second,
/// weaker copy of a decision [`crate::resident_cache`] already makes,
/// in front of the caches that make it, so a recycled address served a
/// whole layer of another model's weights past two caches that would
/// have caught it. Two structures deciding one thing, and only one of
/// them checking.
fn moe_layer_resident(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    layer: &MoeLayerMetal<'_>,
) -> Result<MoeLayerResident, MetalError> {
    Ok(MoeLayerResident {
        attn_nw: resident_f32_buffer(device, layer.attn_norm_w)?,
        ffn_nw: resident_f32_buffer(device, layer.ffn_norm_w)?,
        q_w: resident_weight_buffer(device, layer.q.weights)?,
        k_w: resident_weight_buffer(device, layer.k.weights)?,
        v_w: resident_weight_buffer(device, layer.v.weights)?,
        o_w: resident_weight_buffer(device, layer.o.weights)?,
        r_w: resident_weight_buffer(device, layer.router.weights)?,
    })
}

/// One MoE layer into a Concurrent encoder using llama-style [`MemRanges`]
/// barriers (only on SRC↔DST / DST↔DST conflicts). Same shape as dense
/// [`launch_decode_dense_stack`] and llama `ggml_metal_op` + `mem_ranges`.
///
/// Fused Concurrent groups (fewer barriers): Q∥K∥V, extras+RoPE, KV stores.
#[allow(clippy::too_many_arguments)]
fn encode_moe_layer_fused(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    scratch: &MoeDecodeScratch,
    layer_idx: usize,
    layer: &MoeLayerMetal<'_>,
    kv: &MetalKvBuffers,
    top_k: usize,
    norm_topk_prob: bool,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    rope_layout: MetalRope,
    rope_theta: f32,
    ff_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    pos: usize,
    rms_eps: f32,
) -> Result<(), MetalError> {
    let bound = moe_layer_resident(device, layer)?;
    let attn_nw = &bound.attn_nw;
    let ffn_nw = &bound.ffn_nw;
    let q_w = &bound.q_w;
    let k_w = &bound.k_w;
    let v_w = &bound.v_w;
    let o_w = &bound.o_w;
    let r_w = &bound.r_w;

    // attn_norm: h → x_attn
    {
        let srcs = [scratch.h.as_ref()];
        let dsts = [scratch.x_attn.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_rms_norm(
            encoder,
            device,
            &scratch.h,
            &attn_nw.buffer,
            &scratch.x_attn,
            hidden_dim as u32,
            rms_eps,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    // Q∥K∥V — one Concurrent set (shared src, disjoint dsts).
    {
        let srcs = [scratch.x_attn.as_ref()];
        let dsts = [scratch.q.as_ref(), scratch.k.as_ref(), scratch.v.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_matvec(encoder, device, &layer.q, q_w, &scratch.x_attn, &scratch.q)?;
        encode_matvec(encoder, device, &layer.k, k_w, &scratch.x_attn, &scratch.k)?;
        encode_matvec(encoder, device, &layer.v, v_w, &scratch.x_attn, &scratch.v)?;
        mrs.end_op(&srcs, &dsts);
    }
    // extras + RoPE on q/k — fused group (one barrier before in-place chain).
    {
        let srcs = [scratch.q.as_ref(), scratch.k.as_ref(), scratch.v.as_ref()];
        let dsts = [scratch.q.as_ref(), scratch.k.as_ref(), scratch.v.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_attn_extras(
            encoder,
            device,
            &layer.extras,
            &scratch.q,
            &scratch.k,
            &scratch.v,
            layer.q.rows,
            layer.k.rows,
            layer.v.rows,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_eps,
        )?;
        // Concurrent: barrier before RoPE reads q/k/v written by extras.
        memory_barrier_resources(
            encoder,
            &[scratch.q.as_ref(), scratch.k.as_ref(), scratch.v.as_ref()],
        );
        encode_rope(
            encoder,
            device,
            rope_layout,
            RopeTarget {
                vecs: &scratch.q,
                n_heads: n_heads as u32,
            },
            Some(RopeTarget {
                vecs: &scratch.k,
                n_heads: n_kv_heads as u32,
            }),
            head_dim as u32,
            rope_theta,
            pos as u32,
            ff_buf,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    let token_elems = (n_kv_heads * head_dim) as u32;
    let offset = (pos * n_kv_heads * head_dim) as u32;
    {
        let srcs = [scratch.k.as_ref(), scratch.v.as_ref()];
        let dsts = [kv.k.as_ref(), kv.v.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        // RoPE must complete before KV store (begin_op may already barrier).
        encode_kv_store_append(
            encoder,
            device,
            &scratch.k,
            &scratch.v,
            kv,
            offset,
            token_elems,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    {
        // `encode_gqa_with_kv` tracks itself: with a quantized KV cache it
        // also writes a shared f16 dequant scratch no caller can name.
        encode_gqa_with_kv(
            encoder,
            mrs,
            device,
            &scratch.q,
            kv,
            &scratch.attn,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            (pos + 1) as u32,
            0,
            layer.extras.attn_logit_softcap,
        )?;
    }
    {
        let srcs = [scratch.attn.as_ref()];
        let dsts = [scratch.o.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_matvec(encoder, device, &layer.o, o_w, &scratch.attn, &scratch.o)?;
        mrs.end_op(&srcs, &dsts);
    }
    {
        let srcs = [scratch.h.as_ref(), scratch.o.as_ref()];
        let dsts = [scratch.h.as_ref(), scratch.x2.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_add_rms_norm(
            encoder,
            device,
            &scratch.h,
            &scratch.o,
            &ffn_nw.buffer,
            &scratch.x2,
            hidden_dim as u32,
            rms_eps,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    {
        let srcs = [scratch.x2.as_ref()];
        let dsts = [scratch.router.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_matvec(
            encoder,
            device,
            &layer.router,
            r_w,
            &scratch.x2,
            &scratch.router,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    {
        let srcs = [scratch.router.as_ref()];
        let dsts = [scratch.ids.buffer(), scratch.route.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        // One token, but the *batch* kernel: it spreads the softmax and
        // each of the k selection passes over a simdgroup and keeps `probs`
        // in threadgroup memory. Its single-token twin ran the whole thing
        // on lane 0 out of a private `float[256]` and cost ~1.3-2.2 ms/tok
        // across 16 layers (see the plan's MoE decode diagnosis).
        encode_moe_topk_softmax_batch(
            encoder,
            device,
            &scratch.router,
            scratch.ids.binding(layer_idx),
            &scratch.route,
            layer.router.rows as u32,
            top_k as u32,
            norm_topk_prob,
            1,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    {
        let srcs = [scratch.x2.as_ref(), scratch.ids.buffer()];
        let dsts = [scratch.gate.as_ref(), scratch.up.as_ref()];
        mrs.begin_op(encoder, &srcs, &dsts);
        encode_q4_0_moe_gate_up_id(
            encoder,
            device,
            &scratch.x2,
            &layer.packed,
            scratch.ids.binding(layer_idx),
            &scratch.gate,
            &scratch.up,
            top_k as u32,
            1,
        )?;
        mrs.end_op(&srcs, &dsts);
    }
    let srcs = [
        scratch.x2.as_ref(),
        scratch.ids.buffer(),
        scratch.route.as_ref(),
        scratch.gate.as_ref(),
        scratch.up.as_ref(),
        scratch.h.as_ref(),
    ];
    let dsts = [
        scratch.act.as_ref(),
        scratch.expert_out.as_ref(),
        scratch.h.as_ref(),
    ];
    mrs.begin_op(encoder, &srcs, &dsts);
    encode_q4_0_moe_id(
        encoder,
        device,
        &scratch.x2,
        &layer.packed,
        scratch.ids.binding(layer_idx),
        &scratch.route,
        &scratch.gate,
        &scratch.up,
        &scratch.act,
        &scratch.expert_out,
        &scratch.h,
        top_k as u32,
        1,
        true,
        true,
    )?;
    mrs.end_op(&srcs, &dsts);
    Ok(())
}

/// One MoE layer, one CB (llama graph style): attn → residual → ffn_norm →
/// F32 router → GPU softmax top-k → packed Q4_0 experts → residual.
/// Returns selected expert ids (for hotness accounting). Updates `kv.seq_len`.
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_decode_layer_fused(
    attn_norm_w: &[f32],
    q_launch: &MatvecLaunch<'_>,
    k_launch: &MatvecLaunch<'_>,
    v_launch: &MatvecLaunch<'_>,
    o_launch: &MatvecLaunch<'_>,
    kv: &mut MetalKvBuffers,
    ffn_norm_w: &[f32],
    router_launch: &MatvecLaunch<'_>,
    packed: &MoePackedQ4<'_>,
    top_k: usize,
    norm_topk_prob: bool,
    n_heads: usize,
    rope_layout: MetalRope,
    // This layer's rotation, BOTH halves in one value. Two loose
    // parameters here is how a per-layer base ended up beside a
    // stack-wide divisor set at four call sites; see `LayerRope`.
    rope: LayerRope<'_>,
    pos: usize,
    rms_eps: f32,
    extras: &AttnExtras<'_>,
) -> Result<Vec<usize>, MetalError> {
    // No destructure here: this launch forwards the whole `LayerRope`
    // to the stack below, which is the point of it being one value.
    let layer = MoeLayerMetal {
        attn_norm_w,
        ffn_norm_w,
        q: MatvecLaunch {
            kernel_src: q_launch.kernel_src,
            fn_name: q_launch.fn_name,
            block_bytes: q_launch.block_bytes,
            block_elems: q_launch.block_elems,
            weights: q_launch.weights,
            rows: q_launch.rows,
            row_bytes: q_launch.row_bytes,
            rows_per_tg: q_launch.rows_per_tg,
        },
        k: MatvecLaunch {
            kernel_src: k_launch.kernel_src,
            fn_name: k_launch.fn_name,
            block_bytes: k_launch.block_bytes,
            block_elems: k_launch.block_elems,
            weights: k_launch.weights,
            rows: k_launch.rows,
            row_bytes: k_launch.row_bytes,
            rows_per_tg: k_launch.rows_per_tg,
        },
        v: MatvecLaunch {
            kernel_src: v_launch.kernel_src,
            fn_name: v_launch.fn_name,
            block_bytes: v_launch.block_bytes,
            block_elems: v_launch.block_elems,
            weights: v_launch.weights,
            rows: v_launch.rows,
            row_bytes: v_launch.row_bytes,
            rows_per_tg: v_launch.rows_per_tg,
        },
        o: MatvecLaunch {
            kernel_src: o_launch.kernel_src,
            fn_name: o_launch.fn_name,
            block_bytes: o_launch.block_bytes,
            block_elems: o_launch.block_elems,
            weights: o_launch.weights,
            rows: o_launch.rows,
            row_bytes: o_launch.row_bytes,
            rows_per_tg: o_launch.rows_per_tg,
        },
        router: MatvecLaunch {
            kernel_src: router_launch.kernel_src,
            fn_name: router_launch.fn_name,
            block_bytes: router_launch.block_bytes,
            block_elems: router_launch.block_elems,
            weights: router_launch.weights,
            rows: router_launch.rows,
            row_bytes: router_launch.row_bytes,
            rows_per_tg: router_launch.rows_per_tg,
        },
        packed: MoePackedQ4 {
            gate: packed.gate,
            up: packed.up,
            down: packed.down,
            gate_stride: packed.gate_stride,
            up_stride: packed.up_stride,
            down_stride: packed.down_stride,
            n_experts: packed.n_experts,
            ffn_rows: packed.ffn_rows,
            hidden_rows: packed.hidden_rows,
            gate_row_bytes: packed.gate_row_bytes,
            down_row_bytes: packed.down_row_bytes,
            gate_kind: packed.gate_kind,
            up_kind: packed.up_kind,
            down_kind: packed.down_kind,
        },
        extras: AttnExtras {
            q_bias: extras.q_bias,
            k_bias: extras.k_bias,
            v_bias: extras.v_bias,
            q_norm: extras.q_norm,
            k_norm: extras.k_norm,
            attn_logit_softcap: extras.attn_logit_softcap,
        },
    };
    let out = launch_moe_decode_stack(
        &[], // seeded externally
        std::slice::from_ref(&layer),
        std::slice::from_mut(kv),
        top_k,
        norm_topk_prob,
        n_heads,
        rope_layout,
        rope,
        pos,
        rms_eps,
        None,
        None,
        false,
        true, // reuse existing scratch.h
        None,
    )?;
    Ok(out.1.into_iter().next().unwrap_or_default())
}

/// All MoE layers in **one** command buffer (one wait) — dense-stack
/// equivalent for OLMoE. Returns `(hidden_or_logits_or_argmax, per_layer_expert_ids)`.
/// When `reuse_scratch_h` is true, `hidden` is ignored and scratch `h`
/// from a prior [`moe_decode_seed`] is used. When `embd` is `Some`,
/// gathers the token row into `h` on-GPU (no host seed upload).
///
/// With `final_norm_w` + `output`, folds lm_head on-GPU. `argmax_only`
/// downloads a 1-element `vec![token_id as f32]` (same contract as dense).
#[allow(clippy::too_many_arguments)]
pub fn launch_moe_decode_stack(
    hidden: &[f32],
    layers: &[MoeLayerMetal<'_>],
    kvs: &mut [MetalKvBuffers],
    top_k: usize,
    norm_topk_prob: bool,
    n_heads: usize,
    rope_layout: MetalRope,
    // This layer's rotation, BOTH halves in one value. Two loose
    // parameters here is how a per-layer base ended up beside a
    // stack-wide divisor set at four call sites; see `LayerRope`.
    rope: LayerRope<'_>,
    pos: usize,
    rms_eps: f32,
    final_norm_w: Option<&[f32]>,
    output: Option<&MatvecLaunch<'_>>,
    argmax_only: bool,
    reuse_scratch_h: bool,
    embd: Option<&EmbdGatherMetal<'_>>,
) -> Result<(Vec<f32>, Vec<Vec<usize>>), MetalError> {
    // Exhaustive destructure, no `..`: adding a third half to a
    // layer's rotation must break every launch that ropes.
    let LayerRope {
        theta: rope_theta,
        freq_factors,
    } = rope;
    assert!(!layers.is_empty());
    assert_eq!(layers.len(), kvs.len());
    assert!(top_k > 0 && top_k <= 8);
    let hidden_dim = layers[0].packed.hidden_rows;
    let head_dim = kvs[0].head_dim;
    let n_kv_heads = kvs[0].n_kv_heads;
    for kv in kvs.iter() {
        assert_eq!(kv.head_dim, head_dim);
        assert_eq!(kv.n_kv_heads, n_kv_heads);
        assert_eq!(pos, kv.seq_len);
        if kv.seq_len >= kv.capacity {
            return Err(MetalError::CommandFailed);
        }
    }

    if let Some(e) = embd {
        assert_eq!(e.n_cols, hidden_dim);
        assert!(e.token_id < e.rows);
        moe_decode_ensure(hidden_dim)?;
    } else if !reuse_scratch_h {
        moe_decode_seed(hidden)?;
    } else {
        moe_decode_ensure(hidden_dim)?;
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;
    let ff_resident = match freq_factors {
        Some(ff) => Some(resident_f32_buffer(device, ff)?),
        None => None,
    };

    MOE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let scratch = slot.as_mut().ok_or(MetalError::CommandFailed)?;
        if scratch.hidden_dim != hidden_dim {
            return Err(MetalError::CommandFailed);
        }
        let max_q = layers.iter().map(|l| l.q.rows).max().unwrap();
        let max_k = layers.iter().map(|l| l.k.rows).max().unwrap();
        let max_ffn = layers.iter().map(|l| l.packed.ffn_rows).max().unwrap();
        let max_router = layers.iter().map(|l| l.router.rows).max().unwrap();
        moe_scratch_ensure_caps(device, scratch, max_q, max_k, max_ffn, top_k, max_router)?;
        // One id slot per layer. Grow before encoding: the harvest below
        // reads it after the command buffer this call already waits on,
        // so recording hotness costs no dispatch and no extra sync.
        scratch.ids.ensure(device, top_k, layers.len())?;
        if let Some(out_l) = output {
            moe_scratch_ensure_logits(device, scratch, out_l.rows)?;
        }

        let clock = crate::timing::SubmitClock::start();
        let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
        // llama.cpp / dense-stack: one Concurrent encoder for the full graph,
        // barriers only via MemRanges (ggml_mem_ranges).
        let encoder = compute_encoder_concurrent(&cmd_buf)?;
        let mut mrs = MemRanges::new();

        let _embd_w = if let Some(e) = embd {
            let w = resident_weight_buffer(device, e.weights)?;
            let srcs: [&ProtocolObject<dyn MTLBuffer>; 0] = [];
            let dsts = [scratch.h.as_ref()];
            mrs.begin_op(&encoder, &srcs, &dsts);
            encode_get_rows(
                &encoder,
                device,
                e.kind,
                &w,
                &scratch.h,
                e.row_bytes as u32,
                e.n_cols as u32,
                e.token_id as u32,
            )?;
            mrs.end_op(&srcs, &dsts);
            Some(w)
        } else {
            None
        };

        for (layer_idx, (layer, kv)) in layers.iter().zip(kvs.iter()).enumerate() {
            encode_moe_layer_fused(
                &encoder,
                &mut mrs,
                device,
                scratch,
                layer_idx,
                layer,
                kv,
                top_k,
                norm_topk_prob,
                n_heads,
                n_kv_heads,
                head_dim,
                hidden_dim,
                rope_layout,
                rope_theta,
                ff_resident.as_ref().map(|b| b.buffer.as_ref()),
                pos,
                rms_eps,
            )?;
        }

        // final_norm → optional lm_head → optional argmax (dense-stack parity).
        let (download_n, download_logits, download_argmax) = if let Some(fnw) = final_norm_w {
            assert_eq!(fnw.len(), hidden_dim);
            let fn_buf = resident_f32_buffer(device, fnw)?;
            {
                let srcs = [scratch.h.as_ref()];
                let dsts = [scratch.x_attn.as_ref()];
                mrs.begin_op(&encoder, &srcs, &dsts);
                encode_rms_norm(
                    &encoder,
                    device,
                    &scratch.h,
                    &fn_buf.buffer,
                    &scratch.x_attn,
                    hidden_dim as u32,
                    rms_eps,
                )?;
                mrs.end_op(&srcs, &dsts);
            }
            if let Some(out_l) = output {
                let logits = scratch.logits.as_ref().ok_or(MetalError::CommandFailed)?;
                assert_eq!(out_l.rows, scratch.logits_cap);
                let out_w = resident_weight_buffer(device, out_l.weights)?;
                {
                    let srcs = [scratch.x_attn.as_ref()];
                    let dsts = [logits.as_ref()];
                    mrs.begin_op(&encoder, &srcs, &dsts);
                    encode_matvec(&encoder, device, out_l, &out_w, &scratch.x_attn, logits)?;
                    mrs.end_op(&srcs, &dsts);
                }
                if argmax_only {
                    {
                        let srcs = [logits.as_ref()];
                        let dsts = [scratch.argmax_idx.as_ref()];
                        mrs.begin_op(&encoder, &srcs, &dsts);
                        encode_argmax(
                            &encoder,
                            device,
                            logits,
                            &scratch.argmax_idx,
                            out_l.rows as u32,
                        )?;
                        mrs.end_op(&srcs, &dsts);
                    }
                    (1, false, true)
                } else {
                    (out_l.rows, true, false)
                }
            } else {
                (hidden_dim, false, false)
            }
        } else {
            assert!(output.is_none(), "MoE stack lm_head requires final_norm");
            (hidden_dim, false, false)
        };

        encoder.endEncoding();
        crate::timing::commit_wait_note(&cmd_buf, "moe-decode/tok", 32, clock);

        for kv in kvs.iter_mut() {
            kv.seq_len = pos + 1;
        }

        // Every layer's routed-expert selection, read once from the id log
        // after the wait above (no extra sync). This is what feeds
        // `MoeWeights::activation_counts`, and through it the expert
        // residency/placement plan -- which ranked every expert equal at
        // zero on Metal for as long as this returned empty vectors.
        let all_ids = scratch.ids.harvest(layers.len(), top_k);

        if download_argmax {
            let ptr = scratch.argmax_idx.contents();
            let idx = unsafe { *(ptr.as_ptr() as *const u32) as usize };
            return Ok((vec![idx as f32], all_ids));
        }

        let src: &ProtocolObject<dyn MTLBuffer> = if download_logits {
            scratch
                .logits
                .as_ref()
                .ok_or(MetalError::CommandFailed)?
                .as_ref()
        } else if final_norm_w.is_some() {
            scratch.x_attn.as_ref()
        } else {
            scratch.h.as_ref()
        };
        let n = if download_logits {
            download_n
        } else {
            hidden_dim
        };
        let out = unsafe {
            std::slice::from_raw_parts(src.contents().as_ptr() as *const f32, n).to_vec()
        };
        Ok((out, all_ids))
    })
}

/// Full dense decode layer on one CB: RMSNorm → QKV→RoPE→KV→GQA→O →
/// residual → RMSNorm → gate/up → SiLU×up → down → residual.
/// Returns the updated residual `hidden` on the host. Updates `kv.seq_len`.
#[allow(clippy::too_many_arguments)]
pub fn launch_decode_dense_layer(
    hidden: &[f32],
    attn_norm_w: &[f32],
    q_launch: &MatvecLaunch<'_>,
    k_launch: &MatvecLaunch<'_>,
    v_launch: &MatvecLaunch<'_>,
    o_launch: &MatvecLaunch<'_>,
    kv: &mut MetalKvBuffers,
    ffn_norm_w: &[f32],
    gate_launch: &MatvecLaunch<'_>,
    up_launch: &MatvecLaunch<'_>,
    down_launch: &MatvecLaunch<'_>,
    n_heads: usize,
    rope_layout: MetalRope,
    // This layer's rotation, BOTH halves in one value. Two loose
    // parameters here is how a per-layer base ended up beside a
    // stack-wide divisor set at four call sites; see `LayerRope`.
    rope: LayerRope<'_>,
    pos: usize,
    rms_eps: f32,
    extras: &AttnExtras<'_>,
) -> Result<Vec<f32>, MetalError> {
    // Exhaustive destructure, no `..`: adding a third half to a
    // layer's rotation must break every launch that ropes.
    let LayerRope {
        theta: rope_theta,
        freq_factors,
    } = rope;
    let head_dim = kv.head_dim;
    let n_kv_heads = kv.n_kv_heads;
    let hidden_dim = hidden.len();
    assert_eq!(attn_norm_w.len(), hidden_dim);
    assert_eq!(ffn_norm_w.len(), hidden_dim);
    assert_eq!(q_launch.rows, n_heads * head_dim);
    assert_eq!(k_launch.rows, n_kv_heads * head_dim);
    assert_eq!(v_launch.rows, n_kv_heads * head_dim);
    assert_eq!(o_launch.rows, hidden_dim);
    assert_eq!(down_launch.rows, hidden_dim);
    assert_eq!(gate_launch.rows, up_launch.rows);
    assert_eq!(
        pos, kv.seq_len,
        "decode pos must equal current Metal KV length"
    );
    if kv.seq_len >= kv.capacity {
        return Err(MetalError::CommandFailed);
    }
    assert_freq_factors_len(freq_factors, rope_layout, head_dim);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let h_buf = upload_f32(device, hidden)?;
    let attn_nw = resident_f32_buffer(device, attn_norm_w)?;
    let ffn_nw = resident_f32_buffer(device, ffn_norm_w)?;
    let x_buf = alloc_f32_buffer(device, hidden_dim)?;
    let x2_buf = alloc_f32_buffer(device, hidden_dim)?;

    let q_w = resident_weight_buffer(device, q_launch.weights)?;
    let k_w = resident_weight_buffer(device, k_launch.weights)?;
    let v_w = resident_weight_buffer(device, v_launch.weights)?;
    let o_w = resident_weight_buffer(device, o_launch.weights)?;
    let gate_w = resident_weight_buffer(device, gate_launch.weights)?;
    let up_w = resident_weight_buffer(device, up_launch.weights)?;
    let down_w = resident_weight_buffer(device, down_launch.weights)?;

    let q_buf = alloc_f32_buffer(device, q_launch.rows)?;
    let k_buf = alloc_f32_buffer(device, k_launch.rows)?;
    let v_buf = alloc_f32_buffer(device, v_launch.rows)?;
    let attn_buf = alloc_f32_buffer(device, n_heads * head_dim)?;
    let o_buf = alloc_f32_buffer(device, o_launch.rows)?;
    let gate_buf = alloc_f32_buffer(device, gate_launch.rows)?;
    let up_buf = alloc_f32_buffer(device, up_launch.rows)?;
    let act_buf = alloc_f32_buffer(device, gate_launch.rows)?;
    let down_buf = alloc_f32_buffer(device, down_launch.rows)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    encode_rms_norm(
        &encoder,
        device,
        &h_buf,
        &attn_nw.buffer,
        &x_buf,
        hidden_dim as u32,
        rms_eps,
    )?;
    encode_matvec(&encoder, device, q_launch, &q_w, &x_buf, &q_buf)?;
    encode_matvec(&encoder, device, k_launch, &k_w, &x_buf, &k_buf)?;
    encode_matvec(&encoder, device, v_launch, &v_w, &x_buf, &v_buf)?;
    encode_attn_extras(
        &encoder,
        device,
        extras,
        &q_buf,
        &k_buf,
        &v_buf,
        q_launch.rows,
        k_launch.rows,
        v_launch.rows,
        n_heads,
        n_kv_heads,
        head_dim,
        rms_eps,
    )?;

    encode_rope(
        &encoder,
        device,
        rope_layout,
        RopeTarget {
            vecs: &q_buf,
            n_heads: n_heads as u32,
        },
        Some(RopeTarget {
            vecs: &k_buf,
            n_heads: n_kv_heads as u32,
        }),
        head_dim as u32,
        rope_theta,
        pos as u32,
        ff_buf.as_deref(),
    )?;

    let token_elems = (n_kv_heads * head_dim) as u32;
    let offset = (kv.seq_len * n_kv_heads * head_dim) as u32;
    encode_kv_store_append(&encoder, device, &k_buf, &v_buf, kv, offset, token_elems)?;

    let new_seq = (kv.seq_len + 1) as u32;
    encode_gqa_with_kv(
        &encoder,
        &mut MemRanges::new(),
        device,
        &q_buf,
        kv,
        &attn_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        new_seq,
        0,
        extras.attn_logit_softcap,
    )?;
    encode_matvec(&encoder, device, o_launch, &o_w, &attn_buf, &o_buf)?;
    encode_add_rms_norm(
        &encoder,
        device,
        &h_buf,
        &o_buf,
        &ffn_nw.buffer,
        &x2_buf,
        hidden_dim as u32,
        rms_eps,
    )?;
    encode_matvec(&encoder, device, gate_launch, &gate_w, &x2_buf, &gate_buf)?;
    encode_matvec(&encoder, device, up_launch, &up_w, &x2_buf, &up_buf)?;
    encode_silu_mul(
        &encoder,
        device,
        &gate_buf,
        &up_buf,
        &act_buf,
        gate_launch.rows as u32,
    )?;
    encode_matvec(&encoder, device, down_launch, &down_w, &act_buf, &down_buf)?;
    encode_vec_add(&encoder, device, &h_buf, &down_buf, hidden_dim as u32)?;

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    kv.seq_len += 1;

    let out_ptr = h_buf.contents();
    Ok(unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, hidden_dim).to_vec() })
}

/// Per-layer `mul_mm_sg` launches for [`launch_prefill_dense_layer`] /
/// [`launch_prefill_dense_stack`].
pub struct PrefillDenseLayerMetal<'a> {
    pub attn_norm_w: &'a [f32],
    pub ffn_norm_w: &'a [f32],
    pub q: MulMmSgLaunch<'a>,
    pub k: MulMmSgLaunch<'a>,
    pub v: MulMmSgLaunch<'a>,
    pub o: MulMmSgLaunch<'a>,
    /// Dense SwiGLU/GeGLU FFN, or a routed-expert MoE FFN. The attention
    /// half of the layer is identical either way.
    pub ffn: PrefillFfnMetal<'a>,
    pub post_attn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
    /// QKV bias / QK-norm (Qwen2.5, Qwen3, Gemma-3). Applied after GEMM,
    /// before RoPE — same order as the CPU / decode paths.
    pub extras: AttnExtras<'a>,
    /// This layer's RoPE base AND divisors, or `None` where this layer
    /// does not rotate at all; see [`LayerRope`].
    pub rope: Option<LayerRope<'a>>,
    /// Layer index for [`PrefillCbCache`] keying only.
    pub layer_idx: u32,
}

/// FFN half of a fused-prefill-stack layer.
pub enum PrefillFfnMetal<'a> {
    Dense {
        gate: MulMmSgLaunch<'a>,
        up: MulMmSgLaunch<'a>,
        down: MulMmSgLaunch<'a>,
    },
    /// Routed experts through `mul_mm_id` with the router and top-k run
    /// on the GPU, so the layer still adds no command buffer of its own.
    Moe(crate::gpu::PrefillMoeMetal<'a>),
}

impl<'a> PrefillFfnMetal<'a> {
    /// Intermediate FFN width (dense rows, or one expert's rows).
    fn ffn_rows(&self) -> usize {
        match self {
            Self::Dense { gate, .. } => gate.rows,
            Self::Moe(moe) => moe.packed.ffn_rows,
        }
    }

    fn hidden_out_rows(&self) -> usize {
        match self {
            Self::Dense { down, .. } => down.rows,
            Self::Moe(moe) => moe.packed.hidden_rows,
        }
    }

    fn dense(&self) -> Option<(&MulMmSgLaunch<'a>, &MulMmSgLaunch<'a>, &MulMmSgLaunch<'a>)> {
        match self {
            Self::Dense { gate, up, down } => Some((gate, up, down)),
            Self::Moe(_) => None,
        }
    }
}

struct PrefillScratchView<'a> {
    h: &'a ProtocolObject<dyn MTLBuffer>,
    x: &'a ProtocolObject<dyn MTLBuffer>,
    x2: &'a ProtocolObject<dyn MTLBuffer>,
    q: &'a ProtocolObject<dyn MTLBuffer>,
    k: &'a ProtocolObject<dyn MTLBuffer>,
    v: &'a ProtocolObject<dyn MTLBuffer>,
    attn: &'a ProtocolObject<dyn MTLBuffer>,
    o: &'a ProtocolObject<dyn MTLBuffer>,
    gate: &'a ProtocolObject<dyn MTLBuffer>,
    up: &'a ProtocolObject<dyn MTLBuffer>,
    down: &'a ProtocolObject<dyn MTLBuffer>,
    half_act: &'a ProtocolObject<dyn MTLBuffer>,
}

struct PrefillDenseLayerResident {
    attn_nw: std::sync::Arc<ResidentF32Buffer>,
    ffn_nw: std::sync::Arc<ResidentF32Buffer>,
    q_w: std::sync::Arc<ResidentWeightBuffer>,
    k_w: std::sync::Arc<ResidentWeightBuffer>,
    v_w: std::sync::Arc<ResidentWeightBuffer>,
    o_w: std::sync::Arc<ResidentWeightBuffer>,
    ffn_w: PrefillFfnResident,
    post_attn_w: Option<std::sync::Arc<ResidentF32Buffer>>,
    post_ffn_w: Option<std::sync::Arc<ResidentF32Buffer>>,
    extras: AttnExtrasResident,
}

enum PrefillFfnResident {
    Dense {
        gate_w: std::sync::Arc<ResidentWeightBuffer>,
        up_w: std::sync::Arc<ResidentWeightBuffer>,
        down_w: std::sync::Arc<ResidentWeightBuffer>,
    },
    Moe {
        packed: crate::gpu::MoePackedResident,
        router_w: std::sync::Arc<ResidentF32Buffer>,
    },
}

fn resident_prefill_dense_layer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    layer: &PrefillDenseLayerMetal<'_>,
    hidden_dim: usize,
) -> Result<PrefillDenseLayerResident, MetalError> {
    let post_attn_w = if let Some(post) = layer.post_attn_norm {
        assert_eq!(post.len(), hidden_dim);
        Some(resident_f32_buffer(device, post)?)
    } else {
        None
    };
    let post_ffn_w = if let Some(post) = layer.post_ffn_norm {
        assert_eq!(post.len(), hidden_dim);
        Some(resident_f32_buffer(device, post)?)
    } else {
        None
    };
    let ffn_w = match &layer.ffn {
        PrefillFfnMetal::Dense { gate, up, down } => PrefillFfnResident::Dense {
            gate_w: resident_weight_buffer(device, gate.weights)?,
            up_w: resident_weight_buffer(device, up.weights)?,
            down_w: resident_weight_buffer(device, down.weights)?,
        },
        PrefillFfnMetal::Moe(moe) => PrefillFfnResident::Moe {
            packed: crate::gpu::moe_packed_resident(device, &moe.packed)?,
            router_w: resident_f32_buffer(device, moe.router_w)?,
        },
    };
    Ok(PrefillDenseLayerResident {
        attn_nw: resident_f32_buffer(device, layer.attn_norm_w)?,
        ffn_nw: resident_f32_buffer(device, layer.ffn_norm_w)?,
        q_w: resident_weight_buffer(device, layer.q.weights)?,
        k_w: resident_weight_buffer(device, layer.k.weights)?,
        v_w: resident_weight_buffer(device, layer.v.weights)?,
        o_w: resident_weight_buffer(device, layer.o.weights)?,
        ffn_w,
        post_attn_w,
        post_ffn_w,
        extras: resident_attn_extras(device, &layer.extras)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_prefill_dense_layer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    layer: &PrefillDenseLayerMetal<'_>,
    resident: &PrefillDenseLayerResident,
    scratch: &PrefillScratchView<'_>,
    kv: &MetalKvBuffers,
    n_heads: usize,
    batch: usize,
    hidden_dim: usize,
    rope_layout: MetalRope,
    rope: EncodedRope<'_>,
    start_pos: usize,
    rms_eps: f32,
    gelu_ffn: bool,
    attn_softcap: Option<f32>,
) -> Result<(), MetalError> {
    let n_kv_heads = kv.n_kv_heads;
    let head_dim = kv.head_dim;
    let h_buf = scratch.h;
    let _x_buf = scratch.x;
    let x2_buf = scratch.x2;
    let q_buf = scratch.q;
    let k_buf = scratch.k;
    let v_buf = scratch.v;
    let attn_buf = scratch.attn;
    let o_buf = scratch.o;
    let gate_buf = scratch.gate;
    let up_buf = scratch.up;
    let down_buf = scratch.down;
    let half_act = scratch.half_act;
    let kv_k = kv.k.as_ref();
    let kv_v = kv.v.as_ref();

    // attn_norm: h → half_act (f32→f16 already folded into the norm).
    mrs.begin_op(encoder, &[h_buf], &[half_act]);
    encode_rms_norm_f32_to_f16_batch(
        encoder,
        device,
        h_buf,
        &resident.attn_nw.buffer,
        half_act,
        hidden_dim as u32,
        batch as u32,
        rms_eps,
    )?;
    mrs.end_op(&[h_buf], &[half_act]);

    // Q ∥ K ∥ V — one Concurrent set (shared src, disjoint dsts).
    mrs.begin_op(encoder, &[half_act], &[q_buf, k_buf, v_buf]);
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.q,
        &resident.q_w,
        half_act,
        q_buf,
        batch,
    )?;
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.k,
        &resident.k_w,
        half_act,
        k_buf,
        batch,
    )?;
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.v,
        &resident.v_w,
        half_act,
        v_buf,
        batch,
    )?;
    mrs.end_op(&[half_act], &[q_buf, k_buf, v_buf]);

    // QKV bias / QK-norm, in place on q/k/v.
    mrs.begin_op(encoder, &[q_buf, k_buf, v_buf], &[q_buf, k_buf, v_buf]);
    encode_attn_extras_batch(
        encoder,
        device,
        &layer.extras,
        q_buf,
        k_buf,
        v_buf,
        layer.q.rows,
        layer.k.rows,
        layer.v.rows,
        n_heads,
        n_kv_heads,
        head_dim,
        batch,
        rms_eps,
        &resident.extras,
    )?;
    mrs.end_op(&[q_buf, k_buf, v_buf], &[q_buf, k_buf, v_buf]);

    // RoPE q ∥ RoPE k, in place (disjoint buffers, so they overlap).
    //
    // `None` = this layer does not rotate (llama.cpp's per-layer
    // `use_rope`), and then there is simply no dispatch: q and k reach
    // the KV store and the attention exactly as the projections left
    // them, which is what `ggml` does when the graph never builds the
    // `ggml_rope_ext` node.
    if let Some((rope_theta, ff_buf)) = rope {
        mrs.begin_op(encoder, &[q_buf, k_buf], &[q_buf, k_buf]);
        encode_rope_batch(
            encoder,
            device,
            rope_layout,
            q_buf,
            n_heads as u32,
            head_dim as u32,
            rope_theta,
            start_pos as u32,
            batch as u32,
            ff_buf,
        )?;
        encode_rope_batch(
            encoder,
            device,
            rope_layout,
            k_buf,
            n_kv_heads as u32,
            head_dim as u32,
            rope_theta,
            start_pos as u32,
            batch as u32,
            ff_buf,
        )?;
        mrs.end_op(&[q_buf, k_buf], &[q_buf, k_buf]);
    }

    let kv_width = n_kv_heads * head_dim;
    let token_elems = (batch * kv_width) as u32;
    let offset = (kv.seq_len * kv_width) as u32;
    mrs.begin_op(encoder, &[k_buf, v_buf], &[kv_k, kv_v]);
    encode_kv_store_append(encoder, device, k_buf, v_buf, kv, offset, token_elems)?;
    mrs.end_op(&[k_buf, v_buf], &[kv_k, kv_v]);

    encode_gqa_prefill_with_kv(
        encoder,
        mrs,
        device,
        q_buf,
        kv,
        attn_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        batch as u32,
        start_pos as u32,
        attn_softcap,
    )?;

    mrs.begin_op(encoder, &[attn_buf], &[half_act]);
    encode_f32_to_f16(
        encoder,
        device,
        attn_buf,
        half_act,
        (batch * layer.q.rows) as u32,
    )?;
    mrs.end_op(&[attn_buf], &[half_act]);

    mrs.begin_op(encoder, &[half_act], &[o_buf]);
    encode_mul_mm_sg_f16(
        encoder,
        device,
        &layer.o,
        &resident.o_w,
        half_act,
        o_buf,
        batch,
    )?;
    mrs.end_op(&[half_act], &[o_buf]);

    if let Some(pw) = resident.post_attn_w.as_ref() {
        mrs.begin_op(encoder, &[o_buf], &[o_buf]);
        encode_rms_norm_batch(
            encoder,
            device,
            o_buf,
            &pw.buffer,
            o_buf,
            hidden_dim as u32,
            batch as u32,
            rms_eps,
        )?;
        mrs.end_op(&[o_buf], &[o_buf]);
    }

    // FFN input staging. The dense FFN only wants f16, so residual-add +
    // RMSNorm + the f32→f16 convert collapse into one dispatch. The MoE FFN
    // also routes on the f32 activations, so it keeps the f32 `x2` write and
    // converts separately.
    let ffn_is_moe = matches!(layer.ffn, PrefillFfnMetal::Moe(_));
    if resident.post_attn_w.is_none() && !ffn_is_moe {
        mrs.begin_op(encoder, &[h_buf, o_buf], &[h_buf, half_act]);
        encode_add_rms_norm_f32_to_f16_batch(
            encoder,
            device,
            h_buf,
            o_buf,
            &resident.ffn_nw.buffer,
            half_act,
            hidden_dim as u32,
            batch as u32,
            rms_eps,
        )?;
        mrs.end_op(&[h_buf, o_buf], &[h_buf, half_act]);
    } else {
        if resident.post_attn_w.is_none() {
            mrs.begin_op(encoder, &[h_buf, o_buf], &[h_buf, x2_buf]);
            encode_add_rms_norm_batch(
                encoder,
                device,
                h_buf,
                o_buf,
                &resident.ffn_nw.buffer,
                x2_buf,
                hidden_dim as u32,
                batch as u32,
                rms_eps,
            )?;
            mrs.end_op(&[h_buf, o_buf], &[h_buf, x2_buf]);
        } else {
            mrs.begin_op(encoder, &[h_buf, o_buf], &[h_buf]);
            encode_vec_add(encoder, device, h_buf, o_buf, (batch * hidden_dim) as u32)?;
            mrs.end_op(&[h_buf, o_buf], &[h_buf]);
            if ffn_is_moe {
                mrs.begin_op(encoder, &[h_buf], &[x2_buf]);
                encode_rms_norm_batch(
                    encoder,
                    device,
                    h_buf,
                    &resident.ffn_nw.buffer,
                    x2_buf,
                    hidden_dim as u32,
                    batch as u32,
                    rms_eps,
                )?;
                mrs.end_op(&[h_buf], &[x2_buf]);
            } else {
                mrs.begin_op(encoder, &[h_buf], &[half_act]);
                encode_rms_norm_f32_to_f16_batch(
                    encoder,
                    device,
                    h_buf,
                    &resident.ffn_nw.buffer,
                    half_act,
                    hidden_dim as u32,
                    batch as u32,
                    rms_eps,
                )?;
                mrs.end_op(&[h_buf], &[half_act]);
            }
        }
        if ffn_is_moe {
            mrs.begin_op(encoder, &[x2_buf], &[half_act]);
            encode_f32_to_f16(
                encoder,
                device,
                x2_buf,
                half_act,
                (batch * hidden_dim) as u32,
            )?;
            mrs.end_op(&[x2_buf], &[half_act]);
        }
    }

    match (&layer.ffn, &resident.ffn_w) {
        (
            PrefillFfnMetal::Dense { gate, up, down },
            PrefillFfnResident::Dense {
                gate_w,
                up_w,
                down_w,
            },
        ) => {
            // gate ∥ up
            mrs.begin_op(encoder, &[half_act], &[gate_buf, up_buf]);
            encode_mul_mm_sg_f16(encoder, device, gate, gate_w, half_act, gate_buf, batch)?;
            encode_mul_mm_sg_f16(encoder, device, up, up_w, half_act, up_buf, batch)?;
            mrs.end_op(&[half_act], &[gate_buf, up_buf]);

            // SwiGLU/GeGLU straight to f16 — the staging convert is folded in.
            let ffn_elems = (batch * gate.rows) as u32;
            mrs.begin_op(encoder, &[gate_buf, up_buf], &[half_act]);
            encode_act_mul_f32_to_f16(
                encoder, device, gate_buf, up_buf, half_act, ffn_elems, gelu_ffn,
            )?;
            mrs.end_op(&[gate_buf, up_buf], &[half_act]);

            mrs.begin_op(encoder, &[half_act], &[down_buf]);
            encode_mul_mm_sg_f16(encoder, device, down, down_w, half_act, down_buf, batch)?;
            mrs.end_op(&[half_act], &[down_buf]);
        }
        (PrefillFfnMetal::Moe(moe), PrefillFfnResident::Moe { packed, router_w }) => {
            crate::gpu::encode_moe_prefill_ffn(
                encoder, mrs, device, moe, packed, router_w, x2_buf, half_act, down_buf, batch,
            )?;
        }
        // resident_prefill_dense_layer builds the resident half from the
        // same enum, so the mixed arms are unreachable.
        _ => return Err(MetalError::CommandFailed),
    }
    if let Some(pw) = resident.post_ffn_w.as_ref() {
        mrs.begin_op(encoder, &[down_buf], &[down_buf]);
        encode_rms_norm_batch(
            encoder,
            device,
            down_buf,
            &pw.buffer,
            down_buf,
            hidden_dim as u32,
            batch as u32,
            rms_eps,
        )?;
        mrs.end_op(&[down_buf], &[down_buf]);
    }

    mrs.begin_op(encoder, &[h_buf, down_buf], &[h_buf]);
    encode_vec_add(
        encoder,
        device,
        h_buf,
        down_buf,
        (batch * hidden_dim) as u32,
    )?;
    mrs.end_op(&[h_buf, down_buf], &[h_buf]);
    Ok(())
}

/// Consecutive dense prefill layers in **one** command buffer (B≥4).
///
/// Layer 0 copies `hidden` into scratch `h`; later layers leave `h` in
/// place after each residual add. Activations stay in [`PrefillScratch`]
/// across layers; host readback happens once at the end. Records
/// [`PrefillStackCbKey`] plus per-layer [`PrefillCbKey`] entries in the
/// process [`MetalGraph`].
///
/// Timing: `FRINK_METAL_MM_TIMING=1` logs setup/gpu/readback totals.
#[allow(clippy::too_many_arguments)]
pub fn launch_prefill_dense_stack(
    hidden: &[f32],
    layers: &[PrefillDenseLayerMetal<'_>],
    kvs: &mut [MetalKvBuffers],
    n_heads: usize,
    batch: usize,
    rope_layout: MetalRope,
    start_pos: usize,
    rms_eps: f32,
    gelu_ffn: bool,
    attn_softcap: Option<f32>,
) -> Result<Vec<f32>, MetalError> {
    if batch < 4 {
        return Err(MetalError::CommandFailed);
    }
    assert_eq!(layers.len(), kvs.len());
    assert!(!layers.is_empty());

    let hidden_dim = layers[0].attn_norm_w.len();
    assert_eq!(hidden.len(), batch * hidden_dim);
    let head_dim = kvs[0].head_dim;
    let n_kv_heads = kvs[0].n_kv_heads;
    for (layer, kv) in layers.iter().zip(kvs.iter()) {
        assert_eq!(layer.attn_norm_w.len(), hidden_dim);
        assert_eq!(layer.ffn_norm_w.len(), hidden_dim);
        assert_eq!(layer.q.rows, n_heads * head_dim);
        assert_eq!(layer.k.rows, n_kv_heads * head_dim);
        assert_eq!(layer.v.rows, n_kv_heads * head_dim);
        assert_eq!(layer.o.rows, hidden_dim);
        assert_eq!(layer.ffn.hidden_out_rows(), hidden_dim);
        if let PrefillFfnMetal::Dense { gate, up, .. } = &layer.ffn {
            assert_eq!(gate.rows, up.rows);
        }
        if let PrefillFfnMetal::Moe(moe) = &layer.ffn {
            if !moe.is_supported() {
                return Err(MetalError::CommandFailed);
            }
        }
        assert_eq!(start_pos, kv.seq_len);
        if kv.seq_len + batch > kv.capacity {
            return Err(MetalError::CommandFailed);
        }
        assert_eq!(kv.head_dim, head_dim);
        assert_eq!(kv.n_kv_heads, n_kv_heads);
        assert_freq_factors_len(
            layer.rope.and_then(|r| r.freq_factors),
            rope_layout,
            head_dim,
        );
    }

    let max_q = layers.iter().map(|l| l.q.rows).max().unwrap();
    let max_kv = layers.iter().map(|l| l.k.rows.max(l.v.rows)).max().unwrap();
    let max_gate = layers.iter().map(|l| l.ffn.ffn_rows()).max().unwrap();

    let timing = std::env::var_os("FRINK_METAL_MM_TIMING").is_some();
    let t_setup = std::time::Instant::now();

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let scratch_guard = borrow_prefill_scratch(
        device,
        PrefillScratchCaps {
            batch,
            hidden: hidden_dim,
            max_q,
            max_kv,
            max_gate,
        },
    )?;
    let scratch = scratch_guard.as_ref().expect("prefill scratch ensured");
    copy_f32_into(&scratch.h, hidden);

    let scratch_view = PrefillScratchView {
        h: &scratch.h,
        x: &scratch.x,
        x2: &scratch.x2,
        q: &scratch.q,
        k: &scratch.k,
        v: &scratch.v,
        attn: &scratch.attn,
        o: &scratch.o,
        gate: &scratch.gate,
        up: &scratch.up,
        down: &scratch.down,
        half_act: &scratch.half_act,
    };

    // Per layer, cached on (pointer, len) -- see the same comment in
    // `launch_decode_dense_stack`.
    let ff_resident = layers
        .iter()
        .map(|l| match l.rope.and_then(|r| r.freq_factors) {
            Some(ff) => resident_f32_buffer(device, ff).map(Some),
            None => Ok(None),
        })
        .collect::<Result<Vec<_>, MetalError>>()?;

    {
        let mut graph = metal_graph();
        if !graph.prefill_pipelines_warmed() {
            graph.warm_prefill_pipelines(
                device,
                PrefillWarmParams {
                    layer: &layers[0],
                    rope_layout,
                    head_dim: head_dim as u32,
                    gelu_ffn,
                    kv_dtype: kvs[0].dtype,
                },
            )?;
        }
        graph.prefill.note_stack(PrefillStackCbKey {
            start_layer: layers[0].layer_idx,
            depth: layers.len() as u32,
            batch: batch as u32,
            hidden: hidden_dim as u32,
        });
        for layer in layers {
            graph.prefill.note(PrefillCbKey {
                layer: layer.layer_idx,
                batch: batch as u32,
                hidden: hidden_dim as u32,
                ffn: layer.ffn.ffn_rows() as u32,
                q_rows: layer.q.rows as u32,
            });
        }
    }

    let setup_us = t_setup.elapsed().as_micros();
    let clock = crate::timing::SubmitClock::start();
    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = compute_encoder_concurrent(&cmd_buf)?;
    // One tracker for the whole stack: layer N+1's first dispatch only
    // barriers when it actually touches something layer N left dirty.
    let mut mrs = MemRanges::new();

    for (layer_idx, (layer, kv)) in layers.iter().zip(kvs.iter()).enumerate() {
        let resident = resident_prefill_dense_layer(device, layer, hidden_dim)?;
        encode_prefill_dense_layer(
            &encoder,
            &mut mrs,
            device,
            layer,
            &resident,
            &scratch_view,
            kv,
            n_heads,
            batch,
            hidden_dim,
            rope_layout,
            layer.rope.map(|r| {
                (
                    r.theta,
                    ff_resident[layer_idx].as_ref().map(|b| b.buffer.as_ref()),
                )
            }),
            start_pos,
            rms_eps,
            gelu_ffn,
            attn_softcap,
        )?;
    }

    encoder.endEncoding();
    let gpu_us =
        crate::timing::commit_wait_note(&cmd_buf, "prefill-dense-stack", 1, clock).as_micros();

    for kv in kvs.iter_mut() {
        kv.seq_len += batch;
    }

    let t_read = std::time::Instant::now();
    let out_ptr = scratch.h.contents();
    let out = unsafe {
        std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, batch * hidden_dim).to_vec()
    };
    if timing {
        crate::timing::mm_timing_add(setup_us, gpu_us, t_read.elapsed().as_micros());
    }
    Ok(out)
}

/// One dense prefill layer in **one** command buffer (B≥4).
///
/// Encodes: attn RMSNorm (per row) → Q∥K∥V `mul_mm_sg` → RoPE → KV append →
/// causal GQA → O `mul_mm_sg` → residual → FFN RMSNorm → gate∥up → act →
/// down → residual. Activations stay in [`PrefillScratch`]; barriers match
/// the decode-stack Concurrent pattern. Records a [`PrefillCbKey`] in the
/// process [`MetalGraph`] and warms prefill pipelines on the first call
/// (llama.cpp `ggml_metal_graph_compute` residency; CB replay still TODO).
///
/// Rejects `batch < 4` and models that need QKV bias / QK-norm on this path
/// (caller should fall back to host proj + [`launch_prefill_attn_block`]).
///
/// Timing: `FRINK_METAL_MM_TIMING=1` logs setup/gpu/readback like mul_mm_sg.
#[allow(clippy::too_many_arguments)]
pub fn launch_prefill_dense_layer(
    hidden: &[f32],
    layer: &PrefillDenseLayerMetal<'_>,
    kv: &mut MetalKvBuffers,
    n_heads: usize,
    batch: usize,
    rope_layout: MetalRope,
    start_pos: usize,
    rms_eps: f32,
    gelu_ffn: bool,
    attn_softcap: Option<f32>,
) -> Result<Vec<f32>, MetalError> {
    launch_prefill_dense_stack(
        hidden,
        std::slice::from_ref(layer),
        std::slice::from_mut(kv),
        n_heads,
        batch,
        rope_layout,
        start_pos,
        rms_eps,
        gelu_ffn,
        attn_softcap,
    )
}

/// Host-upload GQA only (parity testing / fallback probe).
#[allow(clippy::too_many_arguments)]
pub fn launch_gqa_decode_host(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_gqa_decode_host_ex(
        q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, 0, None,
    )
}

/// Host-upload GQA with optional sliding-window start and logit softcap.
#[allow(clippy::too_many_arguments)]
pub fn launch_gqa_decode_host_ex(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_start: usize,
    attn_softcap: Option<f32>,
) -> Result<Vec<f32>, MetalError> {
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(k_cache.len(), seq_len * n_kv_heads * head_dim);
    assert_eq!(v_cache.len(), seq_len * n_kv_heads * head_dim);
    assert!(kv_start <= seq_len);

    let shared = shared_metal()?;
    let device = &shared.device;
    let q_buf = upload_f32(device, q)?;
    let k_buf = upload_f16_from_f32(device, k_cache)?;
    let v_buf = upload_f16_from_f32(device, v_cache)?;
    let out_buf = alloc_f32_buffer(device, n_heads * head_dim)?;

    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_gqa(
        &encoder,
        device,
        &q_buf,
        &k_buf,
        &v_buf,
        &out_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        seq_len as u32,
        kv_start as u32,
        attn_softcap,
    )?;
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let ptr = out_buf.contents();
    Ok(unsafe {
        std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n_heads * head_dim).to_vec()
    })
}

/// Multi-token RoPE → batch KV append → causal GQA prefill.
///
/// `q`/`k`/`v` are **pre-RoPE**, packed `[n_q, n_heads|n_kv_heads, head_dim]`.
/// `start_pos` must equal `kv.seq_len` (prefix already resident on Metal, or
/// empty). Returns attention output `[n_q, n_heads, head_dim]` and RoPE'd
/// K/V for host [`KvCache`] sync. Updates `kv.seq_len` by `n_q`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn launch_prefill_attn_block(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv: &mut MetalKvBuffers,
    n_heads: usize,
    n_q: usize,
    rope_layout: MetalRope,
    // This layer's rotation, BOTH halves in one value. Two loose
    // parameters here is how a per-layer base ended up beside a
    // stack-wide divisor set at four call sites; see `LayerRope`.
    rope: LayerRope<'_>,
    start_pos: usize,
    attn_softcap: Option<f32>,
    return_kv: bool,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), MetalError> {
    // Exhaustive destructure, no `..`: adding a third half to a
    // layer's rotation must break every launch that ropes.
    let LayerRope {
        theta: rope_theta,
        freq_factors,
    } = rope;
    let head_dim = kv.head_dim;
    let n_kv_heads = kv.n_kv_heads;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    assert_eq!(q.len(), n_q * q_width);
    assert_eq!(k.len(), n_q * kv_width);
    assert_eq!(v.len(), n_q * kv_width);
    assert_eq!(
        start_pos, kv.seq_len,
        "prefill start_pos must equal current Metal KV length"
    );
    if kv.seq_len + n_q > kv.capacity {
        return Err(MetalError::CommandFailed);
    }
    assert_freq_factors_len(freq_factors, rope_layout, head_dim);
    if n_q == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let q_buf = upload_f32(device, q)?;
    let k_buf = upload_f32(device, k)?;
    let v_buf = upload_f32(device, v)?;
    let attn_buf = alloc_f32_buffer(device, n_q * q_width)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    encode_rope_batch(
        &encoder,
        device,
        rope_layout,
        &q_buf,
        n_heads as u32,
        head_dim as u32,
        rope_theta,
        start_pos as u32,
        n_q as u32,
        ff_buf.as_deref(),
    )?;
    encode_rope_batch(
        &encoder,
        device,
        rope_layout,
        &k_buf,
        n_kv_heads as u32,
        head_dim as u32,
        rope_theta,
        start_pos as u32,
        n_q as u32,
        ff_buf.as_deref(),
    )?;

    let token_elems = (n_q * kv_width) as u32;
    let offset = (kv.seq_len * kv_width) as u32;
    encode_kv_store_append(&encoder, device, &k_buf, &v_buf, kv, offset, token_elems)?;

    let prefill_result = encode_gqa_prefill_with_kv(
        &encoder,
        &mut MemRanges::new(),
        device,
        &q_buf,
        kv,
        &attn_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        n_q as u32,
        start_pos as u32,
        attn_softcap,
    );
    encoder.endEncoding();
    prefill_result?;
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    kv.seq_len += n_q;

    let attn_ptr = attn_buf.contents();
    let attn = unsafe {
        std::slice::from_raw_parts(attn_ptr.as_ptr() as *const f32, n_q * q_width).to_vec()
    };
    if !return_kv {
        return Ok((attn, Vec::new(), Vec::new()));
    }
    let k_ptr = k_buf.contents();
    let v_ptr = v_buf.contents();
    let k_roped = unsafe {
        std::slice::from_raw_parts(k_ptr.as_ptr() as *const f32, n_q * kv_width).to_vec()
    };
    let v_roped = unsafe {
        std::slice::from_raw_parts(v_ptr.as_ptr() as *const f32, n_q * kv_width).to_vec()
    };
    Ok((attn, k_roped, v_roped))
}

/// Host-upload multi-query causal GQA (parity testing).
/// `q` is `[n_q, n_heads, head_dim]`; K/V are full caches of length
/// `kv_prefix_len + n_q` (already including the new tokens).
#[allow(clippy::too_many_arguments)]
pub fn launch_gqa_prefill_host(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_q: usize,
    kv_prefix_len: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_gqa_prefill_host_ex(
        q,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        n_q,
        kv_prefix_len,
        None,
    )
}

/// [`launch_gqa_prefill_host`] with an attention-logit softcap, so the
/// Gemma prefill path can be checked against the CPU reference.
#[allow(clippy::too_many_arguments)]
pub fn launch_gqa_prefill_host_ex(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_q: usize,
    kv_prefix_len: usize,
    attn_softcap: Option<f32>,
) -> Result<Vec<f32>, MetalError> {
    launch_gqa_prefill_host_kernel(
        q,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        n_q,
        kv_prefix_len,
        attn_softcap,
        PrefillAttnKernel::Auto,
    )
}

/// [`launch_gqa_prefill_host_ex`] against one named kernel.
///
/// The dispatch picks the fastest kernel for the shape and there is no way
/// to ask it for a different one, which is correct for production and
/// useless for a parity test: `Auto` would silently send every d=64 case
/// with `n_q >= 8` to `fa_ext` and leave FA-vec unchecked at that width.
/// Naming the kernel here is how the tests reach both, without a switch
/// that shipped builds would also honour.
#[allow(clippy::too_many_arguments)]
pub fn launch_gqa_prefill_host_kernel(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_q: usize,
    kv_prefix_len: usize,
    attn_softcap: Option<f32>,
    kernel: PrefillAttnKernel,
) -> Result<Vec<f32>, MetalError> {
    let total_seq = kv_prefix_len + n_q;
    assert_eq!(q.len(), n_q * n_heads * head_dim);
    assert_eq!(k_cache.len(), total_seq * n_kv_heads * head_dim);
    assert_eq!(v_cache.len(), total_seq * n_kv_heads * head_dim);
    if n_q == 0 {
        return Ok(Vec::new());
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let q_buf = upload_f32(device, q)?;
    let k_buf = upload_f16_from_f32(device, k_cache)?;
    let v_buf = upload_f16_from_f32(device, v_cache)?;
    let out_buf = alloc_f32_buffer(device, n_q * n_heads * head_dim)?;

    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    let enc_result = encode_gqa_prefill(
        &encoder,
        device,
        &q_buf,
        &k_buf,
        &v_buf,
        &out_buf,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        n_q as u32,
        kv_prefix_len as u32,
        attn_softcap,
        kernel,
    );
    encoder.endEncoding();
    enc_result?;
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let ptr = out_buf.contents();
    Ok(unsafe {
        std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n_q * n_heads * head_dim).to_vec()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope::tests::cpu_rope_norm;

    #[test]
    fn prefill_cb_cache_tracks_keys_and_hot_pipelines() {
        let mut cache = PrefillCbCache::default();
        let key = PrefillCbKey {
            layer: 3,
            batch: 8,
            hidden: 4096,
            ffn: 14336,
            q_rows: 4096,
        };
        assert!(cache.note(key));
        assert!(!cache.note(key));
        assert!(cache.contains(&key));
        assert_eq!(cache.len(), 1);
        let stack = PrefillStackCbKey {
            start_layer: 0,
            depth: 32,
            batch: 8,
            hidden: 4096,
        };
        assert!(cache.note_stack(stack));
        assert!(!cache.note_stack(stack));
        assert!(cache.contains_stack(&stack));
        cache.mark_pipeline_hot("q4_k_mul_mm_sg");
        cache.mark_pipeline_hot("rms_norm_f32");
        assert!(cache.is_pipeline_hot("q4_k_mul_mm_sg"));
        assert!(!cache.is_pipeline_hot("gqa_prefill"));
        assert_eq!(cache.hot_pipeline_count(), 2);
    }

    #[test]
    fn metal_mm_timing_env_is_optional() {
        // Documented hook for [`launch_prefill_dense_layer`]: when set, setup/gpu/readback
        // microseconds accumulate via [`crate::timing::mm_timing_add`].
        let enabled = std::env::var_os("FRINK_METAL_MM_TIMING").is_some();
        let _ = enabled;
    }

    #[test]
    fn parse_metal_kv_dtype_f16_default_and_aliases() {
        assert_eq!(parse_metal_kv_dtype(None), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(Some("")), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(Some("f16")), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(Some("FP16")), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(Some("half")), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(Some("bogus")), MetalKvDtype::F16);
    }

    #[test]
    fn parse_metal_kv_dtype_q8_0() {
        assert_eq!(parse_metal_kv_dtype(Some("q8_0")), MetalKvDtype::Q8_0);
        assert_eq!(parse_metal_kv_dtype(Some("Q8_0")), MetalKvDtype::Q8_0);
        assert_eq!(parse_metal_kv_dtype(Some("q8")), MetalKvDtype::Q8_0);
    }

    /// Every spelling this layer resolves, and the rule that an
    /// unknown one is f16 rather than a guess.
    ///
    /// The spellings are llama.cpp's (`q4_0`, not a name of frink's
    /// own), so a copied command line resolves to the same store. The
    /// CLI refuses what is outside the set before reaching here
    /// (`frink_models::ctk`); this fallback is the last resort for the
    /// environment variable, which has no parser in front of it.
    #[test]
    fn parse_metal_kv_dtype_covers_every_spelling() {
        assert_eq!(parse_metal_kv_dtype(Some("fp8")), MetalKvDtype::Fp8);
        assert_eq!(parse_metal_kv_dtype(Some("e4m3")), MetalKvDtype::Fp8);
        assert_eq!(parse_metal_kv_dtype(Some("q4_0")), MetalKvDtype::Q4_0);
        assert_eq!(parse_metal_kv_dtype(Some("Q4_0")), MetalKvDtype::Q4_0);
        assert_eq!(parse_metal_kv_dtype(Some("nonsense")), MetalKvDtype::F16);
        assert_eq!(parse_metal_kv_dtype(None), MetalKvDtype::F16);
        assert!(MetalKvDtype::Q4_0.is_implemented());
        assert!(MetalKvDtype::Fp8.is_implemented());
        assert!(MetalKvDtype::F16.is_implemented());
        assert!(MetalKvDtype::Q8_0.is_implemented());
        assert!(metal_kv_q8_0_viable(4, 64));
        assert!(!metal_kv_q8_0_viable(2, 8));
        assert!(metal_kv_q4_viable(4, 64));
    }

    fn cpu_gqa(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        cpu_gqa_ex(
            q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, 0, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_gqa_ex(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_start: usize,
        softcap: Option<f32>,
    ) -> Vec<f32> {
        let group_size = n_heads / n_kv_heads.max(1);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let kv_h = h / group_size.max(1);
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for t in kv_start..seq_len {
                let k_t = &k_cache
                    [(t * n_kv_heads + kv_h) * head_dim..(t * n_kv_heads + kv_h + 1) * head_dim];
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += q_h[d] * k_t[d];
                }
                let mut score = dot * scale;
                if let Some(c) = softcap.filter(|&c| c > 0.0) {
                    score = c * (score / c).tanh();
                }
                scores[t] = score;
            }
            let max = scores[kv_start..]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in scores[kv_start..].iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in scores[kv_start..].iter_mut() {
                *s /= sum.max(f32::MIN_POSITIVE);
            }
            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            for t in kv_start..seq_len {
                let v_t = &v_cache
                    [(t * n_kv_heads + kv_h) * head_dim..(t * n_kv_heads + kv_h + 1) * head_dim];
                let w = scores[t];
                for d in 0..head_dim {
                    out_h[d] += w * v_t[d];
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_decode_matches_cpu() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let seq_len = 5;
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let cpu = cpu_gqa(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);
        let gpu = launch_gqa_decode_host(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len)
            .expect("metal gqa");
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            // Device KV is f16; allow round-trip vs f32 CPU reference.
            let tol = 2e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "elem {i}: cpu={a} gpu={b} tol={tol}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_fa_vec_matches_cpu() {
        // FA-vec dedicated kernels: d=128 (Llama-3.x), d=64 (TinyLlama /
        // Llama-3.2-1B), d=96 (Phi-3), d=256 (Gemma-3).
        let test_cases = vec![
            (4, 2, 128, 17),
            (8, 2, 128, 33),
            (8, 4, 128, 65),
            (8, 4, 128, 128),
            (4, 2, 64, 17),
            (8, 2, 64, 33),
            (8, 4, 64, 65),
            (32, 4, 64, 128),
            (8, 4, 64, 1),
            (4, 4, 96, 17),
            (8, 8, 96, 33),
            (32, 32, 96, 65),
            (4, 1, 256, 17),
            (8, 4, 256, 33),
            (4, 1, 256, 65),
        ];
        for (n_heads, n_kv_heads, head_dim, seq_len) in test_cases {
            let q: Vec<f32> = (0..n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let cpu = cpu_gqa(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);

            // FA is default-on for d=128; force-on for clarity.
            std::env::set_var("FRINK_METAL_FA_VEC", "1");
            let gpu = launch_gqa_decode_host(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len)
                .expect("metal gqa fa-vec");

            assert_eq!(
                cpu.len(),
                gpu.len(),
                "nh={n_heads} nkv={n_kv_heads} hd={head_dim} seq={seq_len}"
            );
            for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let tol = 2e-3 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "nh={n_heads} nkv={n_kv_heads} hd={head_dim} seq={seq_len} elem {i}: cpu={a} gpu={b} tol={tol}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_fa_vec_window_softcap_matches_cpu() {
        // Windowed (kv_start>0) + softcap paths through FA-vec kernels.
        let cases = [
            // (n_heads, n_kv, head_dim, seq, kv_start, softcap)
            (4, 2, 128, 65, 17, None),
            (8, 4, 128, 65, 33, Some(50.0f32)),
            (4, 2, 64, 48, 16, None),
            (8, 4, 64, 48, 8, Some(30.0)),
            (4, 4, 96, 40, 8, Some(50.0)),
            (4, 1, 256, 40, 12, Some(50.0)),
            (8, 4, 128, 33, 0, Some(50.0)), // softcap only
        ];
        std::env::set_var("FRINK_METAL_FA_VEC", "1");
        for (n_heads, n_kv_heads, head_dim, seq_len, kv_start, softcap) in cases {
            let q: Vec<f32> = (0..n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let cpu = cpu_gqa_ex(
                &q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_start, softcap,
            );
            let gpu = launch_gqa_decode_host_ex(
                &q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_start, softcap,
            )
            .expect("metal fa-vec window/softcap");
            assert_eq!(cpu.len(), gpu.len());
            for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let tol = 3e-3 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "hd={head_dim} seq={seq_len} ks={kv_start} sc={softcap:?} elem {i}: cpu={a} gpu={b}"
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_gqa_prefill(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_q: usize,
        kv_prefix_len: usize,
    ) -> Vec<f32> {
        let q_width = n_heads * head_dim;
        let mut out = vec![0f32; n_q * q_width];
        for qi in 0..n_q {
            let causal_len = kv_prefix_len + qi + 1;
            let kv_elems = causal_len * n_kv_heads * head_dim;
            let row = cpu_gqa(
                &q[qi * q_width..(qi + 1) * q_width],
                &k_cache[..kv_elems],
                &v_cache[..kv_elems],
                n_heads,
                n_kv_heads,
                head_dim,
                causal_len,
            );
            out[qi * q_width..(qi + 1) * q_width].copy_from_slice(&row);
        }
        out
    }

    #[test]
    fn gqa_threadgroup_sizes_are_well_formed() {
        for seq in [1u32, 7, 32, 100, 512, 2048] {
            for hd in [8u32, 64, 128] {
                let pre = gqa_prefill_threadgroup_size(seq, hd);
                assert!(pre.is_power_of_two(), "prefill tg={pre} seq={seq} hd={hd}");
                assert!(pre >= 1);
                let dec = gqa_decode_threadgroup_size(seq, hd);
                assert_eq!(dec % 32, 0, "decode tg={dec} seq={seq} hd={hd}");
                assert!((dec / 32).is_power_of_two(), "decode nsg not pot: {dec}");
                assert!(dec >= 32);
            }
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_matches_cpu() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let n_q = 4;
        let kv_prefix_len = 2;
        let total = kv_prefix_len + n_q;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let cpu = cpu_gqa_prefill(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
        );
        let gpu = launch_gqa_prefill_host(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
        )
        .expect("metal gqa prefill");
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 2e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "elem {i}: cpu={a} gpu={b} tol={tol}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_vec_d128_matches_cpu() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 128;
        let n_q = 3;
        let kv_prefix_len = 5;
        let total = kv_prefix_len + n_q;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let cpu = cpu_gqa_prefill(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
        );
        let gpu = launch_gqa_prefill_host(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix_len,
        )
        .expect("metal gqa prefill fa-vec");
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 5e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "elem {i}: cpu={a} gpu={b} tol={tol}");
        }
    }

    /// Gemma-2 attends through the **prefill** FA-vec kernel at head_dim
    /// 256 with an attention-logit softcap, and nothing covered that: the
    /// only prefill parity test was d=128 without softcap, and every
    /// d=256 decode case fit inside a single 32-wide KV chunk. The kernel
    /// was dropping the upper half of every head and no test noticed.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_vec_softcap_matches_cpu() {
        std::env::set_var("FRINK_METAL_FA_VEC", "1");
        // Every head dim the FA-vec prefill path claims to cover. d=64
        // and d=96 are the ones where fewer than 32 lanes own a float4 of
        // the output, so the lane masking is what these cases pin down --
        // an unmasked lane reads `sq4` past the end of the query and the
        // score is silently wrong for every token.
        // (n_heads, n_kv, head_dim, n_q, kv_prefix, softcap)
        let cases = [
            (8usize, 4usize, 256usize, 3usize, 5usize, Some(50.0f32)),
            (8, 4, 256, 16, 0, Some(50.0)),
            (8, 4, 256, 40, 9, Some(50.0)),
            (8, 4, 256, 3, 5, None),
            (8, 4, 128, 40, 9, Some(50.0)),
            (8, 4, 128, 33, 0, None),
            (9, 3, 64, 40, 9, Some(50.0)),
            (8, 4, 64, 65, 0, None),
            (8, 8, 64, 3, 5, None),
            (4, 2, 96, 40, 9, Some(50.0)),
            (4, 4, 96, 33, 7, None),
        ];
        for (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) in cases {
            let total = kv_prefix_len + n_q;
            let q: Vec<f32> = (0..n_q * n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();

            let q_width = n_heads * head_dim;
            let mut cpu = vec![0f32; n_q * q_width];
            for qi in 0..n_q {
                let causal_len = kv_prefix_len + qi + 1;
                let kv_elems = causal_len * n_kv_heads * head_dim;
                let row = cpu_gqa_ex(
                    &q[qi * q_width..(qi + 1) * q_width],
                    &k[..kv_elems],
                    &v[..kv_elems],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    causal_len,
                    0,
                    softcap,
                );
                cpu[qi * q_width..(qi + 1) * q_width].copy_from_slice(&row);
            }

            let gpu = launch_gqa_prefill_host_kernel(
                &q,
                &k,
                &v,
                n_heads,
                n_kv_heads,
                head_dim,
                n_q,
                kv_prefix_len,
                softcap,
                PrefillAttnKernel::FaVec,
            )
            .expect("metal gqa prefill fa-vec d256");
            assert_eq!(cpu.len(), gpu.len());
            for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let tol = 5e-3 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "hd={head_dim} n_q={n_q} pre={kv_prefix_len} sc={softcap:?} elem {i}: cpu={a} gpu={b}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_ext_d64_matches_cpu() {
        std::env::set_var("FRINK_METAL_FA_VEC", "1");
        // Shapes chosen around the MMA kernel's 8-row K/V granularity:
        // kv_valid = kv_prefix_len + n_q is 49 / 65 / 49 / 128 / 137 / 8 /
        // 128 / 64, covering both the padded tail (kv_valid % 8 != 0) and the
        // exact fit, at 1, 2 and 3 chunks of C=64 keys. The last two are the
        // prefix-cache shape: a long shared prefix with a short new batch, so
        // whole key blocks sit past `max_causal` and must contribute nothing.
        let cases = [
            (9usize, 3usize, 64usize, 40usize, 9usize, Some(50.0f32)),
            (8usize, 4usize, 64usize, 65usize, 0usize, None),
            (8usize, 4usize, 64usize, 40usize, 9usize, Some(50.0f32)),
            (8usize, 4usize, 64usize, 128usize, 0usize, None),
            (6usize, 2usize, 64usize, 130usize, 7usize, Some(30.0f32)),
            (4usize, 4usize, 64usize, 8usize, 0usize, None),
            (8usize, 4usize, 64usize, 8usize, 120usize, None),
            (8usize, 2usize, 64usize, 9usize, 55usize, Some(20.0f32)),
        ];
        for (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) in cases {
            let total = kv_prefix_len + n_q;
            let q: Vec<f32> = (0..n_q * n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let q_width = n_heads * head_dim;
            let mut cpu = vec![0f32; n_q * q_width];
            for qi in 0..n_q {
                let causal_len = kv_prefix_len + qi + 1;
                let kv_elems = causal_len * n_kv_heads * head_dim;
                let row = cpu_gqa_ex(
                    &q[qi * q_width..(qi + 1) * q_width],
                    &k[..kv_elems],
                    &v[..kv_elems],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    causal_len,
                    0,
                    softcap,
                );
                cpu[qi * q_width..(qi + 1) * q_width].copy_from_slice(&row);
            }
            let gpu = launch_gqa_prefill_host_ex(
                &q,
                &k,
                &v,
                n_heads,
                n_kv_heads,
                head_dim,
                n_q,
                kv_prefix_len,
                softcap,
            )
            .expect("fa_ext prefill");
            let mut max_diff = 0f32;
            let mut worst = (0usize, 0f32, 0f32);
            for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (i, *a, *b);
                }
            }
            let tol = 5e-3 * worst.1.abs().max(1.0);
            assert!(
                max_diff <= tol,
                "hd={head_dim} n_q={n_q} pre={kv_prefix_len} sc={softcap:?} max_diff={max_diff} worst={worst:?} tol={tol}"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_ext_matches_fa_vec_d64() {
        std::env::set_var("FRINK_METAL_FA_VEC", "1");
        let (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) =
            (9usize, 3usize, 64usize, 40usize, 9usize, Some(50.0f32));
        let total = kv_prefix_len + n_q;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let run = |kernel| {
            launch_gqa_prefill_host_kernel(
                &q,
                &k,
                &v,
                n_heads,
                n_kv_heads,
                head_dim,
                n_q,
                kv_prefix_len,
                softcap,
                kernel,
            )
            .expect("prefill")
        };
        let fa_vec = run(PrefillAttnKernel::FaVec);
        let fa_ext = run(PrefillAttnKernel::FaExt);
        let mut max_diff = 0f32;
        let mut worst = (0usize, 0f32, 0f32);
        for (i, (a, b)) in fa_vec.iter().zip(fa_ext.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
                worst = (i, *a, *b);
            }
        }
        assert!(
            max_diff <= 1e-4,
            "fa_ext vs fa_vec max_diff={max_diff} worst={worst:?}"
        );
    }

    /// The d=64 MMA kernel against FA-vec over the shape sweep, which is the
    /// A/B this used to run against the kernel's own scalar predecessor. That
    /// predecessor was only ever reachable through an environment variable
    /// nobody would set in production, so it went; FA-vec computes the same
    /// attention by a different tiling and is the surviving second opinion.
    /// Tolerance is the FA-vec one (1e-3, not the scalar's 1e-4): the two
    /// kernels accumulate the softmax in a different order.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_ext_mma_d64_matches_fa_vec() {
        std::env::set_var("FRINK_METAL_FA_VEC", "1");
        // The MMA pipeline must exist: a compile failure here would otherwise
        // surface as a `.expect()` panic that reads like a device problem.
        {
            let shared = shared_metal().expect("metal device");
            ensure_pipeline(
                &shared.device,
                GQA_PREFILL_FA_EXT_MMA_D64_KERNEL_SRC,
                "gqa_prefill_fa_ext_mma_d64",
            )
            .expect("fa_ext MMA pipeline compiles");
        }
        let cases = [
            (9usize, 3usize, 64usize, 40usize, 9usize, Some(50.0f32)),
            (8usize, 4usize, 64usize, 128usize, 0usize, None),
            (6usize, 2usize, 64usize, 130usize, 7usize, Some(30.0f32)),
            (8usize, 8usize, 64usize, 65usize, 0usize, None),
            (8usize, 4usize, 64usize, 8usize, 120usize, None),
            (8usize, 2usize, 64usize, 9usize, 55usize, Some(20.0f32)),
        ];
        for (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) in cases {
            let total = kv_prefix_len + n_q;
            let q: Vec<f32> = (0..n_q * n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let run = |kernel| {
                launch_gqa_prefill_host_kernel(
                    &q,
                    &k,
                    &v,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    n_q,
                    kv_prefix_len,
                    softcap,
                    kernel,
                )
                .expect("fa_ext")
            };
            let fa_vec = run(PrefillAttnKernel::FaVec);
            let mma = run(PrefillAttnKernel::FaExt);
            let mut max_diff = 0f32;
            let mut worst = (0usize, 0f32, 0f32);
            for (i, (a, b)) in fa_vec.iter().zip(mma.iter()).enumerate() {
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (i, *a, *b);
                }
            }
            assert!(
                max_diff <= 1e-3,
                "mma vs fa_vec hd={head_dim} n_q={n_q} pre={kv_prefix_len} \
                 sc={softcap:?} max_diff={max_diff} worst={worst:?}"
            );
        }
    }

    /// The d=128 MMA kernel (Qwen3-0.6B / Phi-4-mini / Mistral shape) against
    /// the f32 CPU reference **and** against FA-vec, which is the only other
    /// kernel at this width and therefore the A/B baseline. Same shape sweep as
    /// the d=64 test: padded cache tails, exact 8-row fits, and the long-prefix
    /// / short-batch case where whole key blocks sit past `max_causal`.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_ext_mma_d128_matches_cpu_and_fa_vec() {
        std::env::set_var("FRINK_METAL_FA_VEC", "1");
        {
            let shared = shared_metal().expect("metal device");
            ensure_pipeline(
                &shared.device,
                GQA_PREFILL_FA_EXT_MMA_D128_KERNEL_SRC,
                "gqa_prefill_fa_ext_mma_d128",
            )
            .expect("d128 fa_ext MMA pipeline compiles");
        }
        let cases = [
            (16usize, 8usize, 128usize, 40usize, 9usize, None),
            (16usize, 8usize, 128usize, 65usize, 0usize, None),
            (8usize, 4usize, 128usize, 128usize, 0usize, Some(50.0f32)),
            (6usize, 2usize, 128usize, 130usize, 7usize, Some(30.0f32)),
            (8usize, 4usize, 128usize, 8usize, 120usize, None),
            (8usize, 2usize, 128usize, 9usize, 55usize, Some(20.0f32)),
        ];
        for (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) in cases {
            let total = kv_prefix_len + n_q;
            let q: Vec<f32> = (0..n_q * n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let q_width = n_heads * head_dim;
            let mut cpu = vec![0f32; n_q * q_width];
            for qi in 0..n_q {
                let causal_len = kv_prefix_len + qi + 1;
                let kv_elems = causal_len * n_kv_heads * head_dim;
                let row = cpu_gqa_ex(
                    &q[qi * q_width..(qi + 1) * q_width],
                    &k[..kv_elems],
                    &v[..kv_elems],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    causal_len,
                    0,
                    softcap,
                );
                cpu[qi * q_width..(qi + 1) * q_width].copy_from_slice(&row);
            }
            let run = |kernel| {
                launch_gqa_prefill_host_kernel(
                    &q,
                    &k,
                    &v,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    n_q,
                    kv_prefix_len,
                    softcap,
                    kernel,
                )
                .expect("d128 prefill")
            };
            let fa_vec = run(PrefillAttnKernel::FaVec);
            let mma = run(PrefillAttnKernel::FaExt);
            let worst_of = |a: &[f32], b: &[f32]| {
                let mut max_diff = 0f32;
                let mut worst = (0usize, 0f32, 0f32);
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    let d = (x - y).abs();
                    if d > max_diff {
                        max_diff = d;
                        worst = (i, *x, *y);
                    }
                }
                (max_diff, worst)
            };
            let (d_cpu, w_cpu) = worst_of(&cpu, &mma);
            let tol = 5e-3 * w_cpu.1.abs().max(1.0);
            assert!(
                d_cpu <= tol,
                "mma vs cpu hd={head_dim} n_q={n_q} pre={kv_prefix_len} \
                 sc={softcap:?} max_diff={d_cpu} worst={w_cpu:?} tol={tol}"
            );
            let (d_vec, w_vec) = worst_of(&fa_vec, &mma);
            assert!(
                d_vec <= 1e-3,
                "mma vs fa_vec hd={head_dim} n_q={n_q} pre={kv_prefix_len} \
                 sc={softcap:?} max_diff={d_vec} worst={w_vec:?}"
            );
        }
    }

    /// The d=256 MMA kernel (Gemma-2 / Gemma-3 shape) against the f32 CPU
    /// reference **and** against FA-vec, the only other kernel at this width.
    /// This is the width the old `own = tiisg < D4` epilogue could not reach:
    /// `D4 == 64` needs two `float4` columns per lane, so the cases below
    /// deliberately include Gemma-3-1B's own 4-head / 1-kv-head / softcap
    /// shape alongside padded cache tails, exact 8-row fits, and a long-prefix
    /// / short-batch case where whole key blocks sit past `max_causal`.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_fa_ext_mma_d256_matches_cpu_and_fa_vec() {
        std::env::set_var("FRINK_METAL_FA_VEC", "1");
        {
            let shared = shared_metal().expect("metal device");
            ensure_pipeline(
                &shared.device,
                GQA_PREFILL_FA_EXT_MMA_D256_KERNEL_SRC,
                "gqa_prefill_fa_ext_mma_d256",
            )
            .expect("d256 fa_ext MMA pipeline compiles");
        }
        let cases = [
            // Gemma-3-1B: 4 heads, 1 kv head, softcap.
            (4usize, 1usize, 256usize, 40usize, 9usize, Some(50.0f32)),
            (4usize, 1usize, 256usize, 64usize, 0usize, None),
            (8usize, 4usize, 256usize, 65usize, 0usize, None),
            (6usize, 2usize, 256usize, 130usize, 7usize, Some(30.0f32)),
            (4usize, 1usize, 256usize, 8usize, 120usize, None),
            (8usize, 2usize, 256usize, 9usize, 55usize, Some(20.0f32)),
        ];
        for (n_heads, n_kv_heads, head_dim, n_q, kv_prefix_len, softcap) in cases {
            let total = kv_prefix_len + n_q;
            let q: Vec<f32> = (0..n_q * n_heads * head_dim)
                .map(|i| (i as f32 * 0.07).sin())
                .collect();
            let k: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.03).cos())
                .collect();
            let v: Vec<f32> = (0..total * n_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.05).sin())
                .collect();
            let q_width = n_heads * head_dim;
            let mut cpu = vec![0f32; n_q * q_width];
            for qi in 0..n_q {
                let causal_len = kv_prefix_len + qi + 1;
                let kv_elems = causal_len * n_kv_heads * head_dim;
                let row = cpu_gqa_ex(
                    &q[qi * q_width..(qi + 1) * q_width],
                    &k[..kv_elems],
                    &v[..kv_elems],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    causal_len,
                    0,
                    softcap,
                );
                cpu[qi * q_width..(qi + 1) * q_width].copy_from_slice(&row);
            }
            let run = |kernel| {
                launch_gqa_prefill_host_kernel(
                    &q,
                    &k,
                    &v,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    n_q,
                    kv_prefix_len,
                    softcap,
                    kernel,
                )
                .expect("d256 prefill")
            };
            let fa_vec = run(PrefillAttnKernel::FaVec);
            let mma = run(PrefillAttnKernel::FaExt);
            let worst_of = |a: &[f32], b: &[f32]| {
                let mut max_diff = 0f32;
                let mut worst = (0usize, 0f32, 0f32);
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    let d = (x - y).abs();
                    if d > max_diff {
                        max_diff = d;
                        worst = (i, *x, *y);
                    }
                }
                (max_diff, worst)
            };
            let (d_cpu, w_cpu) = worst_of(&cpu, &mma);
            let tol = 5e-3 * w_cpu.1.abs().max(1.0);
            assert!(
                d_cpu <= tol,
                "mma vs cpu hd={head_dim} n_q={n_q} pre={kv_prefix_len} \
                 sc={softcap:?} max_diff={d_cpu} worst={w_cpu:?} tol={tol}"
            );
            let (d_vec, w_vec) = worst_of(&fa_vec, &mma);
            assert!(
                d_vec <= 1e-3,
                "mma vs fa_vec hd={head_dim} n_q={n_q} pre={kv_prefix_len} \
                 sc={softcap:?} max_diff={d_vec} worst={w_vec:?}"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gqa_prefill_empty_prefix_matches_per_token_decode() {
        // n_q positions with no prior KV must match running decode GQA
        // at each causal length (same math as forward_batch).
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let n_q = 3;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.09).sin())
            .collect();
        let k: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.04).cos())
            .collect();
        let v: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.06).sin())
            .collect();
        let gpu = launch_gqa_prefill_host(&q, &k, &v, n_heads, n_kv_heads, head_dim, n_q, 0)
            .expect("metal gqa prefill");
        let q_width = n_heads * head_dim;
        for qi in 0..n_q {
            let causal = qi + 1;
            let kv_elems = causal * n_kv_heads * head_dim;
            let decode = launch_gqa_decode_host(
                &q[qi * q_width..(qi + 1) * q_width],
                &k[..kv_elems],
                &v[..kv_elems],
                n_heads,
                n_kv_heads,
                head_dim,
                causal,
            )
            .expect("metal gqa decode");
            for (i, (a, b)) in gpu[qi * q_width..(qi + 1) * q_width]
                .iter()
                .zip(decode.iter())
                .enumerate()
            {
                let tol = 1e-4 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "qi={qi} elem {i}: prefill={a} decode={b}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn prefill_attn_block_matches_cpu_and_updates_kv() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let n_q = 3;
        let start_pos = 0usize;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.08).sin())
            .collect();
        let k: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.04).sin())
            .collect();
        let mut kv =
            MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 16).expect("alloc metal kv");
        let (attn, k_roped, v_roped) = launch_prefill_attn_block(
            &q,
            &k,
            &v,
            &mut kv,
            n_heads,
            n_q,
            MetalRope::new(MetalRopeLayout::Norm),
            LayerRope {
                theta: 10000.0,
                freq_factors: None,
            },
            start_pos,
            None,
            true,
        )
        .expect("prefill attn");
        assert_eq!(kv.seq_len, n_q);
        assert_eq!(attn.len(), n_q * n_heads * head_dim);
        assert_eq!(k_roped.len(), n_q * n_kv_heads * head_dim);

        // CPU reference: per-token RoPE + causal GQA over growing cache.
        let mut k_cpu = k.clone();
        let v_cpu = v.clone();
        let mut q_cpu = q.clone();
        for t in 0..n_q {
            let pos = start_pos + t;
            for h in 0..n_heads {
                let off = (t * n_heads + h) * head_dim;
                cpu_rope_norm(&mut q_cpu[off..off + head_dim], pos, 10000.0, None);
            }
            for h in 0..n_kv_heads {
                let off = (t * n_kv_heads + h) * head_dim;
                cpu_rope_norm(&mut k_cpu[off..off + head_dim], pos, 10000.0, None);
            }
        }
        for (a, b) in k_cpu.iter().zip(k_roped.iter()) {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol);
        }
        for (a, b) in v_cpu.iter().zip(v_roped.iter()) {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol);
        }
        let cpu_attn = cpu_gqa_prefill(
            &q_cpu, &k_cpu, &v_cpu, n_heads, n_kv_heads, head_dim, n_q, 0,
        );
        for (i, (a, b)) in cpu_attn.iter().zip(attn.iter()).enumerate() {
            let tol = 2e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "attn elem {i}: cpu={a} gpu={b}");
        }
        let (k_dl, v_dl) = kv.tokens_host(0, n_q);
        for (i, (a, b)) in k_roped.iter().zip(k_dl.iter()).enumerate() {
            let tol = 2e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "k dl elem {i}: host={a} metal={b}");
        }
        for (i, (a, b)) in v_roped.iter().zip(v_dl.iter()).enumerate() {
            let tol = 2e-3 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "v dl elem {i}: host={a} metal={b}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn q8_0_kv_prefill_matches_f16_path() {
        // elems/token = 2*64 = 128 (Q8_0 block-aligned).
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 64;
        let n_q = 2;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.04).sin())
            .collect();
        let mut kv_f16 =
            MetalKvBuffers::with_capacity_dtype(n_kv_heads, head_dim, 16, MetalKvDtype::F16)
                .expect("f16 kv");
        let mut kv_q8 =
            MetalKvBuffers::with_capacity_dtype(n_kv_heads, head_dim, 16, MetalKvDtype::Q8_0)
                .expect("q8 kv");
        assert_eq!(kv_q8.dtype(), MetalKvDtype::Q8_0);
        let (attn_f16, _, _) = launch_prefill_attn_block(
            &q,
            &k,
            &v,
            &mut kv_f16,
            n_heads,
            n_q,
            MetalRope::new(MetalRopeLayout::Norm),
            LayerRope {
                theta: 10000.0,
                freq_factors: None,
            },
            0,
            None,
            false,
        )
        .expect("f16 prefill");
        let (attn_q8, _, _) = launch_prefill_attn_block(
            &q,
            &k,
            &v,
            &mut kv_q8,
            n_heads,
            n_q,
            MetalRope::new(MetalRopeLayout::Norm),
            LayerRope {
                theta: 10000.0,
                freq_factors: None,
            },
            0,
            None,
            false,
        )
        .expect("q8 prefill");
        assert_eq!(attn_f16.len(), attn_q8.len());
        for (i, (a, b)) in attn_f16.iter().zip(attn_q8.iter()).enumerate() {
            // Q8_0 KV is lossy; keep a loose absolute+relative bound.
            let tol = 5e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "attn elem {i}: f16={a} q8={b} tol={tol}"
            );
        }
        let (k_dl, _) = kv_q8.tokens_host(0, n_q);
        assert_eq!(k_dl.len(), n_q * n_kv_heads * head_dim);
    }

    /// A rotated 4-bit store has to answer what an f16 store answers.
    ///
    /// The rotation is invisible by construction: K is stored rotated
    /// and Q is rotated to match, so the attention output is the same
    /// up to the 4-bit quantization error. A rotation applied to one
    /// side and not the other is not a small error, it is a different
    /// model, which is what this test is for; it went red on the first
    /// run and stayed red until the append and the query agreed about
    /// which head's sign pattern to use.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn q4_kv_prefill_matches_f16_path() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 64;
        let n_q = 2;
        let q: Vec<f32> = (0..n_q * n_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..n_q * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.04).sin())
            .collect();
        let mut kv_f16 =
            MetalKvBuffers::with_capacity_dtype(n_kv_heads, head_dim, 16, MetalKvDtype::F16)
                .expect("f16 kv");
        let mut kv_t4 =
            MetalKvBuffers::with_capacity_dtype(n_kv_heads, head_dim, 16, MetalKvDtype::Q4_0)
                .expect("q4 kv");
        assert!(kv_t4.k_rotated, "head_dim 64 should rotate");
        let rope = MetalRope::new(MetalRopeLayout::Norm);
        let layer_rope = LayerRope {
            theta: 10000.0,
            freq_factors: None,
        };
        let (attn_f16, _, _) = launch_prefill_attn_block(
            &q,
            &k,
            &v,
            &mut kv_f16,
            n_heads,
            n_q,
            rope,
            layer_rope,
            0,
            None,
            false,
        )
        .expect("f16 prefill");
        let (attn_t4, _, _) = launch_prefill_attn_block(
            &q, &k, &v, &mut kv_t4, n_heads, n_q, rope, layer_rope, 0, None, false,
        )
        .expect("q4 prefill");
        assert_eq!(attn_f16.len(), attn_t4.len());
        for (i, (a, b)) in attn_f16.iter().zip(attn_t4.iter()).enumerate() {
            let tol = 8e-2 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "attn elem {i}: f16={a} q4={b} tol={tol}"
            );
        }

        // The host cache is filled from the device one when the dense
        // stack runs ahead of it, and the host attention does not rotate
        // its query, so what comes back here has to be plain K. Without
        // the unrotate in `tokens_host` these rows are the rotated ones,
        // which is a different vector entirely rather than a lossier one.
        //
        // Compared per head in L2, not per element: the quantization
        // error is introduced in the rotated basis and the inverse
        // rotation spreads it over the whole head, so a channel whose
        // own value is small can carry a large share of it. The norm is
        // what an orthogonal transform preserves, so the norm is what
        // has a bound.
        let (k_f16, v_f16) = kv_f16.tokens_host(0, n_q);
        let (k_t4, v_t4) = kv_t4.tokens_host(0, n_q);
        assert_eq!(k_f16.len(), k_t4.len());
        // 0.20 against a measured 0.15 worst head: 4-bit is 4-bit.
        // What this catches is the rotate/unrotate pair disagreeing,
        // which is off by more than 1, not by the quantization step.
        assert_head_l2_close(&k_f16, &k_t4, head_dim, 0.20, "k");
        assert_head_l2_close(&v_f16, &v_t4, head_dim, 0.20, "v");
    }

    /// Relative L2 error per head, the bound a rotated 4-bit store has.
    fn assert_head_l2_close(a: &[f32], b: &[f32], head_dim: usize, tol: f32, what: &str) {
        assert_eq!(a.len(), b.len());
        for (h, (ha, hb)) in a
            .chunks_exact(head_dim)
            .zip(b.chunks_exact(head_dim))
            .enumerate()
        {
            let num: f32 = ha.iter().zip(hb).map(|(x, y)| (x - y) * (x - y)).sum();
            let den: f32 = ha.iter().map(|x| x * x).sum::<f32>().max(1e-12);
            let rel = (num / den).sqrt();
            assert!(rel <= tol, "{what} head {h}: relative L2 {rel} > {tol}");
        }
    }

    /// The host upload and the host download have to agree about the
    /// rotation, because a sequence can cross between the CPU path and
    /// the Metal one in both directions: `upload_from_host` after a CPU
    /// prefill, `tokens_host` when the dense stack runs ahead. Rotating
    /// on one side only is silent, so it gets a round trip of its own.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn q4_host_round_trip_is_plain_k() {
        let n_kv_heads = 2;
        let head_dim = 64;
        let seq = 3;
        let per = n_kv_heads * head_dim;
        // Deliberately NOT a smooth sweep: a smooth vector is exactly
        // what a Hadamard transform concentrates into a few bands, so
        // its rotated absmax is far above its plain one and a 4-bit
        // store of it is much lossier than a real K row. This is a
        // scattered sequence for that reason.
        let k: Vec<f32> = (0..seq * per)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let v: Vec<f32> = (0..seq * per)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0)
            .collect();
        let mut kv =
            MetalKvBuffers::with_capacity_dtype(n_kv_heads, head_dim, 16, MetalKvDtype::Q4_0)
                .expect("q4 kv");
        assert!(kv.k_rotated);
        kv.upload_from_host(&k, &v, seq).expect("upload");
        let (k_back, v_back) = kv.tokens_host(0, seq);
        // Per head in L2, for the reason given on `assert_head_l2_close`.
        // Without the rotate/unrotate pair agreeing, K comes back as a
        // different vector and this is off by more than 1, not by the
        // 4-bit step.
        assert_head_l2_close(&k, &k_back, head_dim, 0.20, "k");
        assert_head_l2_close(&v, &v_back, head_dim, 0.20, "v");
    }

    /// Deterministic weights for one dense decode layer, owned so the
    /// `MatvecLaunch` byte views below can borrow them.
    struct DenseLayerBytes {
        attn_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        q: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        o: Vec<u8>,
        gate: Vec<u8>,
        up: Vec<u8>,
        down: Vec<u8>,
    }

    fn f32_weight_bytes(rows: usize, cols: usize, seed: f32) -> Vec<u8> {
        (0..rows * cols)
            .flat_map(|i| (((i as f32 + seed) * 0.017).sin() * 0.25).to_le_bytes())
            .collect()
    }

    impl DenseLayerBytes {
        fn new(hidden: usize, n_q: usize, n_kv: usize, ffn: usize, seed: f32) -> Self {
            Self {
                attn_norm: (0..hidden)
                    .map(|i| 1.0 + (i as f32 * 0.01 + seed).sin() * 0.1)
                    .collect(),
                ffn_norm: (0..hidden)
                    .map(|i| 1.0 + (i as f32 * 0.02 + seed).cos() * 0.1)
                    .collect(),
                q: f32_weight_bytes(n_q, hidden, seed + 1.0),
                k: f32_weight_bytes(n_kv, hidden, seed + 2.0),
                v: f32_weight_bytes(n_kv, hidden, seed + 3.0),
                o: f32_weight_bytes(hidden, n_q, seed + 4.0),
                gate: f32_weight_bytes(ffn, hidden, seed + 5.0),
                up: f32_weight_bytes(ffn, hidden, seed + 6.0),
                down: f32_weight_bytes(hidden, ffn, seed + 7.0),
            }
        }
    }

    fn f32_matvec(bytes: &[u8], rows: usize, cols: usize) -> MatvecLaunch<'_> {
        let (kernel_src, fn_name, block_bytes, block_elems, rows_per_tg) =
            crate::gpu::matvec_launch_meta("F32").expect("F32 matvec kernel");
        MatvecLaunch {
            kernel_src,
            fn_name,
            block_bytes,
            block_elems,
            weights: bytes,
            rows,
            row_bytes: cols * 4,
            rows_per_tg,
        }
    }

    fn dense_layer_metal<'a>(
        w: &'a DenseLayerBytes,
        rope: Option<LayerRope<'a>>,
        hidden: usize,
        n_q: usize,
        n_kv: usize,
        ffn: usize,
    ) -> DenseLayerMetal<'a> {
        DenseLayerMetal {
            attn_norm_w: &w.attn_norm,
            ffn_norm_w: &w.ffn_norm,
            q: f32_matvec(&w.q, n_q, hidden),
            k: f32_matvec(&w.k, n_kv, hidden),
            v: f32_matvec(&w.v, n_kv, hidden),
            o: f32_matvec(&w.o, hidden, n_q),
            gate: f32_matvec(&w.gate, ffn, hidden),
            up: f32_matvec(&w.up, ffn, hidden),
            down: f32_matvec(&w.down, hidden, ffn),
            extras: AttnExtras::default(),
            rope,
            window: None,
            post_attn_norm: None,
            post_ffn_norm: None,
        }
    }

    /// The fused decode stack must rope EVERY layer with that layer's
    /// own per-band divisors, not with the first layer's.
    ///
    /// Gemma-3 4B/12B/27B are the model that needs this: `rope_scaling
    /// {linear, factor 8}` folded into the full-attention layers'
    /// divisors, nothing on the sliding ones (`gemma3.cpp` never assigns
    /// `rope_freq_scale_train_swa`), five layers in six sliding. The
    /// stack used to take ONE `freq_factors` slice for a whole run, so
    /// `Decoder::metal_stack_needs_per_layer_rope_freqs` refused those
    /// checkpoints off the fused path rather than rotate them wrongly
    /// (issue #63). This test is what makes deleting that refusal safe.
    ///
    /// The oracle is the per-layer launch, which has always taken its
    /// own `freq_factors` per call. The last assertion is what stops the
    /// test being vacuous: feed the stack ONE set for both layers --
    /// exactly the old bug -- and it must answer differently.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn a_decode_stack_ropes_each_layer_with_its_own_freq_factors() {
        let (hidden, n_heads, n_kv_heads, head_dim, ffn) =
            (32usize, 2usize, 1usize, 16usize, 32usize);
        let n_q = n_heads * head_dim;
        let n_kv = n_kv_heads * head_dim;
        let rope = MetalRope::new(MetalRopeLayout::Norm);
        let eps = 1e-5f32;
        // Gemma-3's two sets: the full-attention layers divide every
        // band by the trained linear factor, the sliding ones do not.
        let full_ff = vec![8.0f32; head_dim / 2];
        let swa_ff = vec![1.0f32; head_dim / 2];
        // ... and its two bases, so the layers differ in BOTH halves of
        // `LayerRope` at once, as a real checkpoint does.
        let full_rope = LayerRope {
            theta: 1_000_000.0,
            freq_factors: Some(&full_ff),
        };
        let swa_rope = LayerRope {
            theta: 10_000.0,
            freq_factors: Some(&swa_ff),
        };

        let w: Vec<DenseLayerBytes> = (0..2)
            .map(|i| DenseLayerBytes::new(hidden, n_q, n_kv, ffn, i as f32 * 11.0))
            .collect();
        // Layer 0 slides, layer 1 attends in full: the alternating
        // pattern, in the smallest stack that can show it.
        let ropes = [swa_rope, full_rope];

        let steps = 6usize;
        let hidden0: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.13).sin()).collect();

        let mut kv_stack: Vec<MetalKvBuffers> = (0..2)
            .map(|_| MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 32).expect("kv"))
            .collect();
        let mut kv_ref: Vec<MetalKvBuffers> = (0..2)
            .map(|_| MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 32).expect("kv"))
            .collect();
        let mut kv_one: Vec<MetalKvBuffers> = (0..2)
            .map(|_| MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 32).expect("kv"))
            .collect();

        let mut h_stack = hidden0.clone();
        let mut h_ref = hidden0.clone();
        let mut h_one = hidden0.clone();

        for pos in 0..steps {
            let layers: Vec<DenseLayerMetal<'_>> = (0..2)
                .map(|i| dense_layer_metal(&w[i], Some(ropes[i]), hidden, n_q, n_kv, ffn))
                .collect();
            h_stack = launch_decode_dense_stack(
                &h_stack,
                &layers,
                &mut kv_stack,
                n_heads,
                rope,
                pos,
                eps,
                None,
                None,
                false,
                None,
                false,
            )
            .expect("decode stack");

            // Oracle: the same two layers, one command buffer each.
            for i in 0..2 {
                h_ref = launch_decode_dense_layer(
                    &h_ref,
                    &w[i].attn_norm,
                    &f32_matvec(&w[i].q, n_q, hidden),
                    &f32_matvec(&w[i].k, n_kv, hidden),
                    &f32_matvec(&w[i].v, n_kv, hidden),
                    &f32_matvec(&w[i].o, hidden, n_q),
                    &mut kv_ref[i],
                    &w[i].ffn_norm,
                    &f32_matvec(&w[i].gate, ffn, hidden),
                    &f32_matvec(&w[i].up, ffn, hidden),
                    &f32_matvec(&w[i].down, hidden, ffn),
                    n_heads,
                    rope,
                    ropes[i],
                    pos,
                    eps,
                    &AttnExtras::default(),
                )
                .expect("decode layer");
            }

            // The bug, run deliberately: layer 0's rope for both layers.
            let one_set: Vec<DenseLayerMetal<'_>> = (0..2)
                .map(|i| dense_layer_metal(&w[i], Some(ropes[0]), hidden, n_q, n_kv, ffn))
                .collect();
            h_one = launch_decode_dense_stack(
                &h_one,
                &one_set,
                &mut kv_one,
                n_heads,
                rope,
                pos,
                eps,
                None,
                None,
                false,
                None,
                false,
            )
            .expect("decode stack, one set");
        }

        assert_eq!(h_stack.len(), h_ref.len());
        for (i, (a, b)) in h_ref.iter().zip(h_stack.iter()).enumerate() {
            // Same kernels and the same f16 KV; the stack only
            // reassociates the residual add into the next layer's norm,
            // and on this device that is bit-identical. The tolerance is
            // room for a future reassociation, not for a rope that
            // rotated at the wrong scale: sharing one set instead of two
            // moves the answer ~300x further than this (asserted below).
            let tol = 1e-6 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "elem {i}: per-layer={a} stack={b} tol={tol}"
            );
        }
        let drift = h_stack
            .iter()
            .zip(h_one.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift > 1e-3,
            "one shared rope for both layers answered the same as two: \
             this test proves nothing (max drift {drift})"
        );
    }

    /// A layer whose `rope` is `None` must not be rotated by the fused
    /// decode stack -- llama.cpp's per-layer `use_rope`, which EXAONE-4
    /// 32B, `exaone-moe` and `smollm3` all gate (`frink_models::
    /// rope_layers`). The stacks used to have the RoPE dispatch written
    /// in unconditionally, the same way they had the final norm written
    /// in for OLMo-1.
    ///
    /// The oracle is the per-layer launch, which always rotates, fed
    /// divisors so large that every angle rounds to zero: `angle /=
    /// freq_factors[i]` (the kernel source above) with `1e30` is the
    /// identity rotation to float precision, so "rotated by nothing"
    /// and "not rotated" must agree. The last assertion is what keeps
    /// it honest: the same stack with a REAL rope on that layer must
    /// answer differently, or the test could not tell `None` from
    /// `Some`.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn a_decode_stack_leaves_a_layer_with_no_rope_unrotated() {
        let (hidden, n_heads, n_kv_heads, head_dim, ffn) =
            (32usize, 2usize, 1usize, 16usize, 32usize);
        let n_q = n_heads * head_dim;
        let n_kv = n_kv_heads * head_dim;
        let rope = MetalRope::new(MetalRopeLayout::Norm);
        let eps = 1e-5f32;
        let real = LayerRope {
            theta: 10_000.0,
            freq_factors: None,
        };
        let identity_ff = vec![1e30f32; head_dim / 2];
        // Rotation by an angle of zero on every band: the oracle's
        // spelling of "no rotation" through a kernel that always ropes.
        let identity = LayerRope {
            theta: 10_000.0,
            freq_factors: Some(&identity_ff),
        };

        let w: Vec<DenseLayerBytes> = (0..2)
            .map(|i| DenseLayerBytes::new(hidden, n_q, n_kv, ffn, i as f32 * 11.0))
            .collect();
        // Layer 0 rotates, layer 1 does not: EXAONE-4 32B's shape in
        // the smallest stack that can show it.
        let stack_ropes = [Some(real), None];
        let oracle_ropes = [real, identity];
        let all_rotate = [Some(real), Some(real)];

        let steps = 6usize;
        let hidden0: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut kv_stack: Vec<MetalKvBuffers> = (0..2)
            .map(|_| MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 32).expect("kv"))
            .collect();
        let mut kv_ref: Vec<MetalKvBuffers> = (0..2)
            .map(|_| MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 32).expect("kv"))
            .collect();
        let mut kv_all: Vec<MetalKvBuffers> = (0..2)
            .map(|_| MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 32).expect("kv"))
            .collect();
        let mut h_stack = hidden0.clone();
        let mut h_ref = hidden0.clone();
        let mut h_all = hidden0.clone();

        for pos in 0..steps {
            let layers: Vec<DenseLayerMetal<'_>> = (0..2)
                .map(|i| dense_layer_metal(&w[i], stack_ropes[i], hidden, n_q, n_kv, ffn))
                .collect();
            h_stack = launch_decode_dense_stack(
                &h_stack,
                &layers,
                &mut kv_stack,
                n_heads,
                rope,
                pos,
                eps,
                None,
                None,
                false,
                None,
                false,
            )
            .expect("decode stack");

            for i in 0..2 {
                h_ref = launch_decode_dense_layer(
                    &h_ref,
                    &w[i].attn_norm,
                    &f32_matvec(&w[i].q, n_q, hidden),
                    &f32_matvec(&w[i].k, n_kv, hidden),
                    &f32_matvec(&w[i].v, n_kv, hidden),
                    &f32_matvec(&w[i].o, hidden, n_q),
                    &mut kv_ref[i],
                    &w[i].ffn_norm,
                    &f32_matvec(&w[i].gate, ffn, hidden),
                    &f32_matvec(&w[i].up, ffn, hidden),
                    &f32_matvec(&w[i].down, hidden, ffn),
                    n_heads,
                    rope,
                    oracle_ropes[i],
                    pos,
                    eps,
                    &AttnExtras::default(),
                )
                .expect("decode layer");
            }

            // The bug, run deliberately: rotate layer 1 too.
            let rotated: Vec<DenseLayerMetal<'_>> = (0..2)
                .map(|i| dense_layer_metal(&w[i], all_rotate[i], hidden, n_q, n_kv, ffn))
                .collect();
            h_all = launch_decode_dense_stack(
                &h_all,
                &rotated,
                &mut kv_all,
                n_heads,
                rope,
                pos,
                eps,
                None,
                None,
                false,
                None,
                false,
            )
            .expect("decode stack, everything rotated");
        }

        assert_eq!(h_stack.len(), h_ref.len());
        for (i, (a, b)) in h_ref.iter().zip(h_stack.iter()).enumerate() {
            // `1e30` divisors leave angles of ~1e-30 radians: cos is
            // exactly 1 and sin is ~1e-30 in f32, so the oracle's
            // rotation is the identity to well under this tolerance.
            let tol = 1e-5 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "elem {i}: identity-roped={a} unroped={b} tol={tol}"
            );
        }
        let drift = h_stack
            .iter()
            .zip(h_all.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift > 1e-3,
            "rotating the no-rope layer answered the same as leaving it: \
             this test proves nothing (max drift {drift})"
        );
    }

    /// Q8_0 weights for one dense PREFILL layer, owned like
    /// [`DenseLayerBytes`] so the `MulMmSgLaunch` views can borrow them.
    struct PrefillLayerBytes {
        attn_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        q: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        o: Vec<u8>,
        gate: Vec<u8>,
        up: Vec<u8>,
        down: Vec<u8>,
    }

    fn q8_0_weight_bytes(rows: usize, cols: usize, seed: f32) -> Vec<u8> {
        let mut out = Vec::new();
        for r in 0..rows {
            let row: Vec<f32> = (0..cols)
                .map(|c| (((r * cols + c) as f32 + seed) * 0.017).sin() * 0.25)
                .collect();
            out.extend(frink_quant::quantize_q8_0(&row));
        }
        out
    }

    impl PrefillLayerBytes {
        fn new(hidden: usize, n_q: usize, n_kv: usize, ffn: usize, seed: f32) -> Self {
            Self {
                attn_norm: (0..hidden)
                    .map(|i| 1.0 + (i as f32 * 0.01 + seed).sin() * 0.1)
                    .collect(),
                ffn_norm: (0..hidden)
                    .map(|i| 1.0 + (i as f32 * 0.02 + seed).cos() * 0.1)
                    .collect(),
                q: q8_0_weight_bytes(n_q, hidden, seed + 1.0),
                k: q8_0_weight_bytes(n_kv, hidden, seed + 2.0),
                v: q8_0_weight_bytes(n_kv, hidden, seed + 3.0),
                o: q8_0_weight_bytes(hidden, n_q, seed + 4.0),
                gate: q8_0_weight_bytes(ffn, hidden, seed + 5.0),
                up: q8_0_weight_bytes(ffn, hidden, seed + 6.0),
                down: q8_0_weight_bytes(hidden, ffn, seed + 7.0),
            }
        }
    }

    fn q8_0_mul_mm_sg(bytes: &[u8], rows: usize, cols: usize) -> MulMmSgLaunch<'_> {
        let (fn_name, block_bytes, block_elems) =
            crate::gpu::mul_mm_sg_meta("Q8_0").expect("Q8_0 mul_mm_sg kernel");
        assert!(cols.is_multiple_of(block_elems));
        MulMmSgLaunch {
            weights: bytes,
            rows,
            row_bytes: (cols / block_elems) * block_bytes,
            fn_name,
            block_bytes,
            block_elems,
        }
    }

    fn prefill_layer_metal<'a>(
        w: &'a PrefillLayerBytes,
        rope: Option<LayerRope<'a>>,
        layer_idx: u32,
        hidden: usize,
        n_q: usize,
        n_kv: usize,
        ffn: usize,
    ) -> PrefillDenseLayerMetal<'a> {
        PrefillDenseLayerMetal {
            attn_norm_w: &w.attn_norm,
            ffn_norm_w: &w.ffn_norm,
            q: q8_0_mul_mm_sg(&w.q, n_q, hidden),
            k: q8_0_mul_mm_sg(&w.k, n_kv, hidden),
            v: q8_0_mul_mm_sg(&w.v, n_kv, hidden),
            o: q8_0_mul_mm_sg(&w.o, hidden, n_q),
            ffn: PrefillFfnMetal::Dense {
                gate: q8_0_mul_mm_sg(&w.gate, ffn, hidden),
                up: q8_0_mul_mm_sg(&w.up, ffn, hidden),
                down: q8_0_mul_mm_sg(&w.down, hidden, ffn),
            },
            post_attn_norm: None,
            post_ffn_norm: None,
            extras: AttnExtras::default(),
            rope,
            layer_idx,
        }
    }

    /// The prefill twin of
    /// `a_decode_stack_ropes_each_layer_with_its_own_freq_factors`, and
    /// it matters as much: prefill is where a long Gemma-3 prompt spends
    /// its time, and the wrong divisors get worse the longer the prompt.
    ///
    /// Oracle and non-vacuity check are the same shape as the decode
    /// test. The one-layer launch is the oracle because a run of one
    /// cannot index the wrong layer's divisors.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn a_prefill_stack_ropes_each_layer_with_its_own_freq_factors() {
        let (hidden, n_heads, n_kv_heads, head_dim, ffn) =
            (32usize, 2usize, 2usize, 16usize, 64usize);
        let n_q = n_heads * head_dim;
        let n_kv = n_kv_heads * head_dim;
        let batch = 8usize;
        let rope = MetalRope::new(MetalRopeLayout::Norm);
        let eps = 1e-5f32;
        let full_ff = vec![8.0f32; head_dim / 2];
        let swa_ff = vec![1.0f32; head_dim / 2];
        let ropes = [
            LayerRope {
                theta: 10_000.0,
                freq_factors: Some(&swa_ff),
            },
            LayerRope {
                theta: 1_000_000.0,
                freq_factors: Some(&full_ff),
            },
        ];

        let w: Vec<PrefillLayerBytes> = (0..2)
            .map(|i| PrefillLayerBytes::new(hidden, n_q, n_kv, ffn, i as f32 * 11.0))
            .collect();
        let hidden0: Vec<f32> = (0..batch * hidden)
            .map(|i| (i as f32 * 0.031).sin())
            .collect();

        let mut kv_stack: Vec<MetalKvBuffers> = (0..2)
            .map(|_| MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 32).expect("kv"))
            .collect();
        let layers: Vec<PrefillDenseLayerMetal<'_>> = (0..2)
            .map(|i| prefill_layer_metal(&w[i], Some(ropes[i]), i as u32, hidden, n_q, n_kv, ffn))
            .collect();
        let h_stack = launch_prefill_dense_stack(
            &hidden0,
            &layers,
            &mut kv_stack,
            n_heads,
            batch,
            rope,
            0,
            eps,
            false,
            None,
        )
        .expect("prefill stack");

        let mut h_ref = hidden0.clone();
        for i in 0..2 {
            let mut kv = MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 32).expect("kv");
            let layer =
                prefill_layer_metal(&w[i], Some(ropes[i]), i as u32, hidden, n_q, n_kv, ffn);
            let input = std::mem::take(&mut h_ref);
            h_ref = launch_prefill_dense_layer(
                &input, &layer, &mut kv, n_heads, batch, rope, 0, eps, false, None,
            )
            .expect("prefill layer");
        }

        // The bug: layer 0's rope for the whole run.
        let mut kv_one: Vec<MetalKvBuffers> = (0..2)
            .map(|_| MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 32).expect("kv"))
            .collect();
        let one_set: Vec<PrefillDenseLayerMetal<'_>> = (0..2)
            .map(|i| prefill_layer_metal(&w[i], Some(ropes[0]), i as u32, hidden, n_q, n_kv, ffn))
            .collect();
        let h_one = launch_prefill_dense_stack(
            &hidden0,
            &one_set,
            &mut kv_one,
            n_heads,
            batch,
            rope,
            0,
            eps,
            false,
            None,
        )
        .expect("prefill stack, one set");

        assert_eq!(h_stack.len(), h_ref.len());
        for (i, (a, b)) in h_ref.iter().zip(h_stack.iter()).enumerate() {
            let tol = 1e-6 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "elem {i}: per-layer={a} stack={b} tol={tol}"
            );
        }
        let drift = h_stack
            .iter()
            .zip(h_one.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift > 1e-3,
            "one shared rope for both layers answered the same as two: \
             this test proves nothing (max drift {drift})"
        );
    }

    /// The prefill twin of
    /// `a_decode_stack_leaves_a_layer_with_no_rope_unrotated`. The two
    /// stacks are two encoders (`encode_prefill_dense_layer` here,
    /// `launch_decode_dense_stack` in `decode_dense.rs`), each with its
    /// own `if let Some(..) = rope` around the dispatch, so each needs
    /// its own proof that `None` means "unrotated" rather than "rotated
    /// by whatever the buffer held".
    ///
    /// Same oracle: divisors of `1e30` turn every angle to zero, so a
    /// layer roped by the identity must agree with one not roped at
    /// all, and a layer roped for real must not.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn a_prefill_stack_leaves_a_layer_with_no_rope_unrotated() {
        let (hidden, n_heads, n_kv_heads, head_dim, ffn) =
            (32usize, 2usize, 2usize, 16usize, 64usize);
        let n_q = n_heads * head_dim;
        let n_kv = n_kv_heads * head_dim;
        let batch = 8usize;
        let rope = MetalRope::new(MetalRopeLayout::Norm);
        let eps = 1e-5f32;
        let real = LayerRope {
            theta: 10_000.0,
            freq_factors: None,
        };
        let identity_ff = vec![1e30f32; head_dim / 2];
        let identity = LayerRope {
            theta: 10_000.0,
            freq_factors: Some(&identity_ff),
        };
        let w: Vec<PrefillLayerBytes> = (0..2)
            .map(|i| PrefillLayerBytes::new(hidden, n_q, n_kv, ffn, i as f32 * 11.0))
            .collect();
        let hidden0: Vec<f32> = (0..batch * hidden)
            .map(|i| (i as f32 * 0.031).sin())
            .collect();

        let run = |ropes: [Option<LayerRope<'_>>; 2]| -> Vec<f32> {
            let mut kvs: Vec<MetalKvBuffers> = (0..2)
                .map(|_| MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 32).expect("kv"))
                .collect();
            let layers: Vec<PrefillDenseLayerMetal<'_>> = (0..2)
                .map(|i| prefill_layer_metal(&w[i], ropes[i], i as u32, hidden, n_q, n_kv, ffn))
                .collect();
            launch_prefill_dense_stack(
                &hidden0, &layers, &mut kvs, n_heads, batch, rope, 0, eps, false, None,
            )
            .expect("prefill stack")
        };

        // Layer 0 rotates, layer 1 does not.
        let h_stack = run([Some(real), None]);
        let h_ref = run([Some(real), Some(identity)]);
        // The bug, run deliberately.
        let h_all = run([Some(real), Some(real)]);

        assert_eq!(h_stack.len(), h_ref.len());
        for (i, (a, b)) in h_ref.iter().zip(h_stack.iter()).enumerate() {
            let tol = 1e-5 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "elem {i}: identity-roped={a} unroped={b} tol={tol}"
            );
        }
        let drift = h_stack
            .iter()
            .zip(h_all.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift > 1e-3,
            "rotating the no-rope layer answered the same as leaving it: \
             this test proves nothing (max drift {drift})"
        );
    }
}
