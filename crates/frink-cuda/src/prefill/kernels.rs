//! CUDA C for the five elementwise / per-row ops a dense prefill layer
//! needs between its matmuls, so that the layer never returns to the
//! host between them (`docs/plans/cpu-cuda-parity.md`, step 2b, #259).
//!
//! Each kernel is the CUDA spelling of ONE host function, named in its
//! doc comment; the hardware tests in [`super::tests`] hold every one
//! against a naive host loop of the same arithmetic. Always compiled
//! (it is text), so the source is available to a GPU-less build's
//! tests and to `tools/` that execute the emitted C on a CPU.

/// `frink_core::matmul::rms_norm`, once per row: `out[r][i] = x[r][i]
/// * rsqrt(mean(x[r]^2) + eps) * w[i]`.
///
/// One block per row, 256 threads, a shared-memory tree reduction.
/// Serves the two pre-norms, the two post-norms (Gemma-2) and the QK
/// norms: a per-head QK norm is this kernel with `n = head_dim` and
/// `n_rows = batch * n_heads`, a whole-vector one with `n = n_heads *
/// head_dim`, which is how the loader's length rule reads them
/// (`frink_models::capability::QkNormStyle`).
pub const RMSNORM_ROWS_KERNEL_SRC: &str = r#"
// One block per row, any multiple of 32 threads up to 1024: each warp
// reduces its sum of squares by shuffles, warp 0 reduces the warps'
// sums, and the block scales its row. Two barriers, where the old tree
// over 256 shared slots took nine; a decode token's one row of 4096
// took 6.5 us on an L40S.
extern "C" __global__ void rmsnorm_rows_f32(
    const float* x,
    const float* w,
    float* out,
    int n_rows,
    int n,
    float eps
) {
    __shared__ float partial[32];
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const float* xr = x + (size_t)row * n;
    float* orow = out + (size_t)row * n;
    const int tid = threadIdx.x;
    const int tg = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    float acc = 0.0f;
    for (int i = tid; i < n; i += tg) {
        const float v = xr[i];
        acc += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float total = lane < (tg >> 5) ? partial[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) total += __shfl_xor_sync(0xffffffffu, total, off);
        if (lane == 0) partial[0] = total;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(partial[0] / (float)n + eps);
    for (int i = tid; i < n; i += tg) {
        orow[i] = xr[i] * inv_rms * w[i];
    }
}
"#;

/// `x[r][i] += bias[i]` for every row: the QKV biases (Qwen2) that
/// `Decoder::apply_qkv_bias_and_clamp` adds on the host. Elementwise
/// over `n_rows * n`.
pub const ADD_BIAS_ROWS_KERNEL_SRC: &str = r#"
extern "C" __global__ void add_bias_rows_f32(
    float* x,
    const float* bias,
    int n_rows,
    int n
) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n_rows * n;
    if (i < total) {
        x[i] += bias[i % n];
    }
}
"#;

/// `a[i] += b[i]`: the residual add (`residual_add` with no scale;
/// a model with `residual_scale` never reaches this path, see
/// `Decoder::metal_can_serve_model`).
pub const ADD_ROWS_KERNEL_SRC: &str = r#"
extern "C" __global__ void add_rows_f32(
    float* a,
    const float* b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += b[i];
    }
}
"#;

/// RoPE over a `[n_rows, n_heads, head_dim]` batch, one thread per
/// (row, head, band). The CUDA spelling of `Decoder::apply_rope_head_
/// theta` plus `apply_rope_attn_factor`:
///
/// * `neox == 0` rotates adjacent pairs `(2i, 2i+1)` (llama.cpp
///   `LLAMA_ROPE_TYPE_NORM`, `apply_rope_interleaved`); `neox != 0`
///   rotates split halves `(i, i + rot/2)` (`apply_rope`).
/// * `rot_dim` channels of each head rotate and the tail passes through
///   (Phi-3/Phi-4).
/// * `freq_factors`, when non-null, divides band `i`'s angle
///   (Llama 3's `rope_freqs.weight`), as `apply_rope_with_freq_factors`.
/// * `mscale` multiplies the ROTATED channels before rotation, which is
///   what `apply_rope_attn_factor` does on the host (rotation is
///   linear, so the order is unobservable); `1.0` when the model has no
///   YaRN magnitude term.
/// * The band frequency is `1 / theta^(2i / rot_dim)` in f32, the same
///   expression as `frink_core::attention::rope_freqs`.
///
/// Position of row `r` is `start_pos + r`.
pub const ROPE_ROWS_KERNEL_SRC: &str = r#"
extern "C" __global__ void rope_rows_f32(
    float* x,
    const float* freq_factors,
    int n_rows,
    int n_heads,
    int head_dim,
    int rot_dim,
    int start_pos,
    float theta,
    int neox,
    float mscale
) {
    int half = rot_dim / 2;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t per_row = (size_t)n_heads * half;
    if (idx >= (size_t)n_rows * per_row) return;
    int row = (int)(idx / per_row);
    int rem = (int)(idx % per_row);
    int h = rem / half;
    int i = rem % half;
    float* v = x + ((size_t)row * n_heads + h) * head_dim;
    float freq = 1.0f / powf(theta, (float)(2 * i) / (float)rot_dim);
    float angle = (float)(start_pos + row) * freq;
    if (freq_factors != 0) {
        angle = angle / freq_factors[i];
    }
    float s, c;
    sincosf(angle, &s, &c);
    int ia = neox ? i : 2 * i;
    int ib = neox ? i + half : 2 * i + 1;
    float a = v[ia] * mscale;
    float b = v[ib] * mscale;
    v[ia] = a * c - b * s;
    v[ib] = a * s + b * c;
}
"#;

/// Causal GQA over a prefill batch: `frink_core::attention::
/// causal_gqa_attention_row` for every query row at once. Four warps
/// per block, one query row per warp, the four rows consecutive so the
/// K/V rows they share sit in L1; each lane holds a `float4` slice of
/// the query and of the running V sum, so a key costs one `float4`
/// load of K, one warp reduction and one `float4` load of V. The
/// online softmax is the decode kernel's
/// (`crate::attn::GQA_DECODE_KERNEL_SRC`), extended by:
///
/// * the causal bound: query row `r` sits at position `start_pos + r`
///   and sees keys `0..=start_pos + r` of a K/V buffer laid out
///   `[start_pos + n_q, n_kv_heads, head_dim]` (prefix, then batch);
/// * a sliding window (`window > 0`): keys below `pos + 1 - window`
///   are skipped, the host's `seq_len.saturating_sub(w)`;
/// * the logit softcap (`softcap > 0`): `softcap * tanh(score /
///   softcap)` after the scale, Gemma-2's `attn_logit_softcapping`.
///
/// `scale` is passed in rather than derived so the caller and the host
/// body cannot disagree about it. `head_dim` is a multiple of 4 and at
/// most 256 (two `float4` per lane); the first version of this kernel,
/// one warp per block and scalar loads, measured 2.5 ms per
/// Llama-3.2-3B layer at pp512 on an RTX 3090, a sixth of the step.
pub const CAUSAL_GQA_PREFILL_KERNEL_SRC: &str = r#"
__device__ __forceinline__ float frink_inf_pf() {
    return __int_as_float(0x7f800000);
}

#define FX_ATTN_WARPS 4

extern "C" __global__ void causal_gqa_prefill_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_all,
    const float* __restrict__ v_all,
    float* __restrict__ out,
    int n_q,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int start_pos,
    int window,
    float scale,
    float softcap
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int r = blockIdx.x * FX_ATTN_WARPS + warp;
    const int h = blockIdx.y;
    if (r >= n_q || h >= n_heads) return;
    const int group_size = n_heads / max(n_kv_heads, 1);
    const int kv_h = h / max(group_size, 1);
    const int seq_len = start_pos + r + 1;
    int lo = 0;
    if (window > 0 && seq_len > window) lo = seq_len - window;
    const int d4 = head_dim >> 2;          // float4s per row, <= 64
    const int n_slices = (d4 + 31) >> 5;   // float4s per lane, 1 or 2
    const float4* q_h = (const float4*)(q + ((size_t)r * n_heads + h) * head_dim);

    float4 qv[2];
    float4 acc[2];
#pragma unroll
    for (int s = 0; s < 2; s++) {
        const int i = lane + 32 * s;
        qv[s] = (s < n_slices && i < d4) ? q_h[i] : make_float4(0.f, 0.f, 0.f, 0.f);
        acc[s] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    float m = -frink_inf_pf();
    float l = 0.f;
    const unsigned mask = 0xffffffffu;

    for (int t = lo; t < seq_len; t++) {
        const float4* k_t = (const float4*)(k_all + ((size_t)t * n_kv_heads + kv_h) * head_dim);
        float pdot = 0.f;
#pragma unroll
        for (int s = 0; s < 2; s++) {
            const int i = lane + 32 * s;
            if (s < n_slices && i < d4) {
                const float4 kv = k_t[i];
                pdot += qv[s].x * kv.x + qv[s].y * kv.y + qv[s].z * kv.z + qv[s].w * kv.w;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            pdot += __shfl_xor_sync(mask, pdot, off);
        }
        float score = pdot * scale;
        if (softcap > 0.f) score = softcap * tanhf(score / softcap);
        const float m2 = fmaxf(m, score);
        const float a = (m == -frink_inf_pf()) ? 0.f : expf(m - m2);
        const float b = expf(score - m2);
        l = l * a + b;
        const float4* v_t = (const float4*)(v_all + ((size_t)t * n_kv_heads + kv_h) * head_dim);
#pragma unroll
        for (int s = 0; s < 2; s++) {
            const int i = lane + 32 * s;
            if (s < n_slices && i < d4) {
                const float4 vv = v_t[i];
                acc[s].x = acc[s].x * a + b * vv.x;
                acc[s].y = acc[s].y * a + b * vv.y;
                acc[s].z = acc[s].z * a + b * vv.z;
                acc[s].w = acc[s].w * a + b * vv.w;
            }
        }
        m = m2;
    }
    const float inv = (l > 0.f) ? (1.f / l) : 0.f;
    float4* out_h = (float4*)(out + ((size_t)r * n_heads + h) * head_dim);
#pragma unroll
    for (int s = 0; s < 2; s++) {
        const int i = lane + 32 * s;
        if (s < n_slices && i < d4) {
            out_h[i] = make_float4(acc[s].x * inv, acc[s].y * inv, acc[s].z * inv, acc[s].w * inv);
        }
    }
}
"#;

/// Module and function names, in one place so the enqueue helpers and
/// the tests cannot spell them differently.
pub const RMSNORM_ROWS: (&str, &str) = ("frink_prefill_rmsnorm_rows", "rmsnorm_rows_f32");
pub const ADD_BIAS_ROWS: (&str, &str) = ("frink_prefill_add_bias_rows", "add_bias_rows_f32");
pub const ADD_ROWS: (&str, &str) = ("frink_prefill_add_rows", "add_rows_f32");
pub const ROPE_ROWS: (&str, &str) = ("frink_prefill_rope_rows", "rope_rows_f32");
pub const CAUSAL_GQA_PREFILL: (&str, &str) = ("frink_prefill_causal_gqa", "causal_gqa_prefill_f32");
