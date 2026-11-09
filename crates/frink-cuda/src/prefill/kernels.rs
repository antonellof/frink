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
extern "C" __global__ void rmsnorm_rows_f32(
    const float* x,
    const float* w,
    float* out,
    int n_rows,
    int n,
    float eps
) {
    __shared__ float partial[256];
    int row = blockIdx.x;
    if (row >= n_rows) return;
    const float* xr = x + (size_t)row * n;
    float* orow = out + (size_t)row * n;
    int tid = threadIdx.x;
    int tg = blockDim.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += tg) {
        float v = xr[i];
        acc += v * v;
    }
    partial[tid] = acc;
    __syncthreads();
    for (int stride = tg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf(partial[0] / (float)n + eps);
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

// Causal GQA for a block of query rows of one head, tiled over the keys.
//
// Grid (ceil(n_q / (4 * rpw)), n_heads), 128 threads. Warp w owns rows
// w, w + 4, w + 8, ... of the block's 4 * rpw. Per tile of bc <= 32
// keys the block stages K and V once in shared memory, and every row
// of the block reads them there: lane j scores key j of the tile
// against the row (K rows padded to head_dim + 4 floats so the lanes'
// float4 reads land on distinct banks), the row's running max and sum
// update once per tile, and lane l accumulates the row's float4
// channels 4l.. (and 4l + 128.. for a head of 256) over the tile's
// probabilities. Before this a warp walked one row's keys alone and
// read each key from global memory, about 4 GB of L2 traffic per
// Llama-3.1-8B layer at pp512.
//
// Dynamic shared memory: Q [4 * rpw][head_dim], K [bc][head_dim + 4],
// V [bc][head_dim], P [4][32] floats.
extern "C" __global__ void causal_gqa_prefill_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_all,
    const float* __restrict__ v_all,
    float* __restrict__ out,
    int n_q,
    int heads,      // n_heads | n_kv_heads << 16
    int head_dim,
    int start_pos,
    int window,
    float scale,
    float softcap,
    int tile        // rows per warp | keys per tile << 8
) {
    extern __shared__ __align__(16) float fx_smem[];
    const int n_heads = heads & 0xffff;
    const int n_kv_heads = heads >> 16;
    const int rpw = tile & 0xff;
    const int bc = tile >> 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int h = blockIdx.y;
    const int rows_per_block = FX_ATTN_WARPS * rpw;
    const int r0 = blockIdx.x * rows_per_block;
    const int group_size = n_heads / max(n_kv_heads, 1);
    const int kv_h = h / max(group_size, 1);
    const int d4 = head_dim >> 2;
    const int n_slices = (d4 + 31) >> 5;   // float4s per lane, 1 or 2
    const int ks = head_dim + 4;           // padded K row, in floats

    float* q_s = fx_smem;
    float* k_s = q_s + rows_per_block * head_dim;
    float* v_s = k_s + bc * ks;
    float* p_s = v_s + bc * head_dim;

    // The block's rows into shared memory; a row past n_q is zeros and
    // is never stored.
    for (int i = threadIdx.x; i < rows_per_block * d4; i += blockDim.x) {
        const int rr = i / d4;
        const int c = i - rr * d4;
        const int r = r0 + rr;
        float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < n_q) v4 = ((const float4*)(q + ((size_t)r * n_heads + h) * head_dim))[c];
        ((float4*)(q_s + rr * head_dim))[c] = v4;
    }

    float m[8];
    float l[8];
    float4 acc[8][2];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        m[i] = -frink_inf_pf();
        l[i] = 0.f;
        acc[i][0] = make_float4(0.f, 0.f, 0.f, 0.f);
        acc[i][1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    // Keys any row of the block can see: from the first row's window
    // start to the last row's own position.
    const int r_last = min(r0 + rows_per_block, n_q) - 1;
    const int seq_first = start_pos + r0 + 1;
    int t_lo = 0;
    if (window > 0 && seq_first > window) t_lo = seq_first - window;
    const int t_hi = start_pos + r_last + 1;
    const unsigned mask = 0xffffffffu;

    for (int t0 = t_lo; t0 < t_hi; t0 += bc) {
        const int n_t = min(bc, t_hi - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < n_t * d4; i += blockDim.x) {
            const int j = i / d4;
            const int c = i - j * d4;
            const size_t at = ((size_t)(t0 + j) * n_kv_heads + kv_h) * head_dim;
            ((float4*)(k_s + j * ks))[c] = ((const float4*)(k_all + at))[c];
            ((float4*)(v_s + j * head_dim))[c] = ((const float4*)(v_all + at))[c];
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i >= rpw) break;
            const int rr = warp + FX_ATTN_WARPS * i;
            const int r = r0 + rr;
            if (r >= n_q) break;
            const int seq_len = start_pos + r + 1;
            int lo = 0;
            if (window > 0 && seq_len > window) lo = seq_len - window;

            // Lane j scores key t0 + j.
            const int t = t0 + lane;
            float score = -frink_inf_pf();
            if (lane < n_t && t >= lo && t < seq_len) {
                const float4* qr = (const float4*)(q_s + rr * head_dim);
                const float4* kr = (const float4*)(k_s + lane * ks);
                float dot = 0.f;
                for (int c = 0; c < d4; c++) {
                    const float4 a = qr[c];
                    const float4 b = kr[c];
                    dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
                score = dot * scale;
                if (softcap > 0.f) score = softcap * tanhf(score / softcap);
            }
            float tmax = score;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(mask, tmax, off));
            if (tmax == -frink_inf_pf()) continue;   // no key of this tile is the row's

            const float m2 = fmaxf(m[i], tmax);
            const float alpha = (m[i] == -frink_inf_pf()) ? 0.f : expf(m[i] - m2);
            const float p = (score == -frink_inf_pf()) ? 0.f : expf(score - m2);
            float psum = p;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) psum += __shfl_xor_sync(mask, psum, off);
            l[i] = l[i] * alpha + psum;
            m[i] = m2;
            p_s[warp * 32 + lane] = p;
            __syncwarp();
#pragma unroll
            for (int s = 0; s < 2; s++) {
                const int c = lane + 32 * s;
                if (s < n_slices && c < d4) {
                    float4 a4 = acc[i][s];
                    a4.x *= alpha; a4.y *= alpha; a4.z *= alpha; a4.w *= alpha;
                    for (int j = 0; j < n_t; j++) {
                        const float pj = p_s[warp * 32 + j];
                        const float4 vv = ((const float4*)(v_s + j * head_dim))[c];
                        a4.x += pj * vv.x; a4.y += pj * vv.y; a4.z += pj * vv.z; a4.w += pj * vv.w;
                    }
                    acc[i][s] = a4;
                }
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i >= rpw) break;
        const int r = r0 + warp + FX_ATTN_WARPS * i;
        if (r >= n_q) break;
        const float inv = (l[i] > 0.f) ? (1.f / l[i]) : 0.f;
        float4* out_h = (float4*)(out + ((size_t)r * n_heads + h) * head_dim);
#pragma unroll
        for (int s = 0; s < 2; s++) {
            const int c = lane + 32 * s;
            if (s < n_slices && c < d4) {
                const float4 a4 = acc[i][s];
                out_h[c] = make_float4(a4.x * inv, a4.y * inv, a4.z * inv, a4.w * inv);
            }
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
