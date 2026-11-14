use crate::tensor::Tensor;

/// Row-major matmul: `a` is [m, k], `b_t` is [n, k] (i.e. already
/// transposed, which is how GGUF stores weight matrices for a `y = W x`
/// projection). Output is [m, n]. Parallelized over output rows with
/// rayon, matching the "each output row is independent" decomposition
/// used across llama.cpp / ggml's matmul kernels -- except for the
/// single-token decode case (`m == 1`, the common case for this path,
/// since `WeightMatrix::F32` is only used for small tensors like
/// embeddings/synthetic weights), where parallelizing over `m` would
/// give exactly one chunk regardless of thread count, i.e. no
/// parallelism at all no matter how large `n` is. That case instead
/// parallelizes over `n` (output features) directly, since `out` is
/// exactly `n` elements long when `m == 1` and needs no layout
/// transpose to do so.
pub fn matmul_f32(a: &Tensor, b_t: &Tensor) -> Tensor {
    let m = a.rows();
    let k = a.cols();
    let n = b_t.rows();
    assert_eq!(
        b_t.cols(),
        k,
        "matmul shape mismatch: a is [{m},{k}], b_t is [{},{}]",
        b_t.rows(),
        b_t.cols()
    );

    let mut out = vec![0f32; m * n];
    // The SIMD dot attention uses, resolved once: the single-accumulator
    // loop this replaced could not be vectorised (a float sum cannot be
    // reordered without fast-math) and was 12% of a Bonsai-2-27B decode
    // step on two 48 x 5120 F32 projections per layer.
    let dot = crate::attention::dot_f32_fn();
    if m == 1 {
        let a_row = a.row(0);
        crate::par::items_mut(&mut out, 1, |col, out_val| {
            *out_val = dot(a_row, b_t.row(col));
        });
    } else {
        crate::par::chunks_mut(&mut out, n, 1, |row, out_row| {
            let a_row = a.row(row);
            for (col, out_val) in out_row.iter_mut().enumerate() {
                *out_val = dot(a_row, b_t.row(col));
            }
        });
    }

    Tensor::new(out, vec![m, n])
}

/// RMSNorm as used by LLaMA-family and DeepSeek/GLM/Kimi-family decoders:
/// x_normalized = x / sqrt(mean(x^2) + eps) * weight
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len());
    let mut out = vec![0f32; x.len()];
    mul3_scale(x, weight, rms_scale(x, eps), &mut out);
    out
}

/// Per-head RMSNorm (Qwen3 / Gemma3 `attn_q_norm` / `attn_k_norm`):
/// `weight` has length `head_dim` and is reused for every head in
/// `x` (layout `[n_heads, head_dim]` row-major).
pub fn rms_norm_per_head(x: &[f32], weight: &[f32], head_dim: usize, eps: f32) -> Vec<f32> {
    assert_eq!(weight.len(), head_dim);
    assert_eq!(x.len() % head_dim, 0);
    let mut out = vec![0f32; x.len()];
    for (head, out_h) in x.chunks_exact(head_dim).zip(out.chunks_exact_mut(head_dim)) {
        mul3_scale(head, weight, rms_scale(head, eps), out_h);
    }
    out
}

/// `1 / sqrt(mean(x^2) + eps)` the way `ggml_compute_forward_rms_norm_f32`
/// (`ggml-cpu/ops.cpp`) computes it, bit for bit: each square in f32,
/// summed IN ORDER in f64, the mean rounded to f32, then `1.0f/sqrtf`.
///
/// The order is the point. ggml's loop is scalar (`ggml_float` is
/// double), so every libllama build gets the same bits, and the norm is
/// the input of the first activation quantizer of every layer. An f32
/// SIMD sum here differed from it in the last bit on 93-99% of
/// elements, and one ulp at an input that scales to `k + 0.5` flips a
/// quant: 78 flips in layer 0 alone on Llama-3.2-1B Q8_0 (#476). With
/// this, layer 0's Q/K/V are bit-identical to libllama's there and on
/// gpt-oss-20b.
///
/// Every RMS norm on the CPU goes through here, so there is one answer.
pub fn rms_scale(x: &[f32], eps: f32) -> f32 {
    debug_assert!(!x.is_empty(), "a norm over no elements");
    let mut sum = 0f64;
    for &v in x {
        sum += (v * v) as f64;
    }
    let mean = (sum / x.len() as f64) as f32;
    1.0 / (mean + eps).sqrt()
}

#[inline]
fn mul3_scale(x: &[f32], w: &[f32], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), w.len());
    debug_assert_eq!(x.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { mul3_scale_neon(x, w, scale, out) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { mul3_scale_avx2(x, w, scale, out) };
            return;
        }
    }
    for ((o, &xv), &wv) in out.iter_mut().zip(x).zip(w) {
        *o = xv * scale * wv;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mul3_scale_neon(x: &[f32], w: &[f32], scale: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let vs = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let wv = vld1q_f32(w.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(vmulq_f32(xv, vs), wv));
        i += 4;
    }
    while i < n {
        out[i] = x[i] * scale * w[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul3_scale_avx2(x: &[f32], w: &[f32], scale: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let vs = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        _mm256_storeu_ps(
            out.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_mul_ps(xv, vs), wv),
        );
        i += 8;
    }
    while i < n {
        out[i] = x[i] * scale * w[i];
        i += 1;
    }
}

/// Soft-cap used by Gemma 2+ attention / final logits:
/// `softcap * tanh(x / softcap)`.
pub fn softcap_inplace(x: &mut [f32], softcap: f32) {
    if softcap <= 0.0 {
        return;
    }
    let inv = 1.0 / softcap;
    let cap = |chunk: &mut [f32]| {
        for v in chunk.iter_mut() {
            *v = softcap * (*v * inv).tanh();
        }
    };
    // A final-logit cap runs over the whole vocabulary every decode
    // token (Gemma-2/3/4: 256k), and a scalar `tanh` loop over that is
    // a millisecond or more on one core. Each element is capped by the
    // same expression either way, so the split changes no value.
    const PAR_MIN: usize = 1 << 15;
    if x.len() >= PAR_MIN {
        use rayon::prelude::*;
        x.par_chunks_mut(PAR_MIN / 4).for_each(cap);
    } else {
        cap(x);
    }
}

/// GELU (tanh approximation) used by Gemma GeGLU FFNs.
pub fn gelu(x: f32) -> f32 {
    // HuggingFace / llama.cpp GELU tanh approx.
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    const C: f32 = 0.044_715;
    0.5 * x * (1.0 + (K * (x + C * x * x * x)).tanh())
}

/// Elementwise gated FFN combine: gelu(gate) * up (Gemma GeGLU).
///
/// **The tanh is gone and the answers moved. Both are on purpose.**
///
/// `gelu` above spends a libm `tanhf` per element, which does not
/// vectorize and does not inline: on Gemma-3-1B CPU `pp512` it was 10.7%
/// of all non-idle samples in the process. The identity
/// `0.5·x·(1 + tanh(u)) == x / (1 + exp(-2u))` turns that into one
/// vectorized exponential, and [`expf_neon`] / [`expf_avx2`] do four or
/// eight at a time.
///
/// The rewrite is also *more* accurate on the side that matters, which is
/// why it clears the "no less accurate than what it replaces" bar rather
/// than merely getting close to it. For `x` negative, `tanh(u) → -1`, so
/// `1 + tanh(u)` is a difference of nearly equal numbers and loses most of
/// its significant bits; `1 + exp(-2u)` is a sum of a large term and 1 and
/// loses none. `geglu_is_no_less_accurate_than_the_tanh_form_it_replaces`
/// measures both against an `f64` evaluation of the same formula and
/// pins it.
pub fn geglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    par_gated_chunks(gate, up, gelu_mul)
}

/// Threshold above which the elementwise FFN activations fork to Rayon.
/// Decode passes one row (`ffn_dim`, a few thousand elements) and would
/// only pay fork-join; prefill passes `batch × ffn_dim`, which on
/// Gemma-3-1B is 3.5 M elements per layer.
const GATED_PAR_MIN: usize = 1 << 15;

/// Run a gated-activation kernel over the whole pair, forked to Rayon
/// past [`GATED_PAR_MIN`].
///
/// Prefill ran these serially on the calling thread while every other
/// core sat in the FFN's fork-join: on Gemma-3-1B CPU `pp512`, `tanhf`
/// under `geglu` was 10.7% of *all* non-idle samples in the process and
/// 3682 of its 3688 samples were on the main thread alone.
///
/// `f` takes slices rather than one element, so the kernel can hold a
/// vector register across a run instead of being called per lane.
/// Chunking stays bit-exact for the same reason it always was — no
/// reduction, each output element depends only on its own inputs — and
/// the chunk length is a multiple of 16, so it never splits a 4- or
/// 8-lane group and the vector arms cannot see a boundary either.
#[inline]
fn par_gated_chunks<F>(gate: &[f32], up: &[f32], f: F) -> Vec<f32>
where
    F: Fn(&[f32], &[f32], &mut [f32]) + Sync + Send,
{
    let n = gate.len();
    let mut out = vec![0f32; n];
    if n < GATED_PAR_MIN {
        f(gate, up, &mut out);
        return out;
    }
    // Cache-line-aligned chunks so no two tasks share a 64-byte line.
    let chunk = (n.div_ceil(crate::par::num_threads() * 4)).next_multiple_of(16);
    crate::par::chunks_mut(&mut out, chunk, 1, |c, o| {
        let start = c * chunk;
        let end = start + o.len();
        f(&gate[start..end], &up[start..end], o);
    });
    out
}

/// `out[i] = gelu(gate[i]) * up[i]`, vectorized where there is a vector
/// exponential and falling back to the scalar [`gelu`] where there is not.
fn gelu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { gelu_mul_neon(gate, up, out) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { gelu_mul_avx2(gate, up, out) };
            return;
        }
    }
    for ((o, g), u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *o = gelu(*g) * *u;
    }
}

/// `out[i] = silu(gate[i]) * up[i]`, same shape as [`gelu_mul`].
fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { silu_mul_neon(gate, up, out) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { silu_mul_avx2(gate, up, out) };
            return;
        }
    }
    for ((o, g), u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *o = silu(*g) * *u;
    }
}

// The `ggml_v_expf` constants (`ggml/src/ggml-cpu/vec.h`), used only by
// the vector arms below -- gated so a host with neither SIMD arm still
// compiles clean under `-D warnings`.
//
// `crate::attention` carries its own private copy of this kernel for the
// softmax. THEY ARE THE SAME ROUTINE AND SHOULD LIVE IN ONE MODULE; this
// copy exists only because that one is private to `attention` and this
// branch does not own that file. Whoever merges them: the two differ in
// exactly one way, the clamp, and the difference is load-bearing --
// see [`EXP_CLAMP`].
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod exp_consts {
    // The nine ggml_v_expf polynomial constants live in
    // `frink_core::vexp`, shared with `attention`, which had a
    // byte-identical copy. Only the clamp differs between the two, and
    // it differs on purpose: see EXP_CLAMP.
    pub use crate::vexp::*;

    /// **Clamped on BOTH sides, unlike the softmax copy in `attention`,
    /// and that is the whole difference between the two.**
    ///
    /// This kernel drops `ggml_v_expf`'s overflow branch, which is only
    /// sound while `|n| <= 126`. A softmax argument is `score - row_max`,
    /// hence never positive, so the softmax copy clamps from below alone.
    /// A gated-activation argument is `-x` (SiLU) or `-2u` (GELU) and has
    /// no sign at all, so an upper clamp is required or a large negative
    /// input walks `bits(z) << 23` straight out of the exponent field and
    /// returns garbage rather than infinity.
    ///
    /// `87` is where clamping stops costing anything *below*: `exp(-87)`
    /// is already under `f32::MIN_POSITIVE`, and `87 · log2(e)` is
    /// `125.5`, which keeps `n` inside the fast path's range with room.
    ///
    /// Above it the clamp alone is NOT harmless, and this is the trap:
    /// the value appears as the denominator `x / (1 + exp(t))`, so a
    /// saturated `exp` divides a numerator that has no bound of its own.
    /// At `x = -1e30` the true `t` is infinite and the answer is `-0`,
    /// while `x / (1 + exp(87))` is `-1.6e-8` -- a real number where there
    /// should be nothing. So the vector arms do not merely clamp on the
    /// high side, they *select*: `t >= EXP_CLAMP` yields zero. That is
    /// exact to within `1.5e-37`, because `t >= 87` only happens where the
    /// activation itself has collapsed (`x <= -10` for GELU, `x <= -87`
    /// for SiLU) and it collapses far faster than `|x|` grows.
    pub const EXP_CLAMP: f32 = 87.0;
    /// `sqrt(2/pi)`, the GELU tanh approximation's outer constant.
    pub const GELU_K: f32 = 0.797_884_6;
    /// `-2 · GELU_K`: `gelu(x) = x / (1 + exp(x·(A + B·x²)))` with
    /// `A = -2K` and `B = -2KC`, which is `0.5·x·(1 + tanh(K·(x + C·x³)))`
    /// rearranged so no cancellation is left in it.
    pub const GELU_A: f32 = -2.0 * GELU_K;
    /// `-2 · GELU_K · 0.044715`.
    pub const GELU_B: f32 = -2.0 * GELU_K * 0.044_715;
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use exp_consts::*;

/// `exp(x)` for four lanes: ARM optimized-routines' `expf`, the shape
/// llama.cpp vendors as `ggml_v_expf`. Accurate to under an ulp.
///
/// `z = fma(x, log2(e), 0x1.8p23)` rounds `x·log2(e)` to an integer `n`
/// and leaves it in the low mantissa bits of `z`, so `bits(z) << 23` is
/// the exponent field of `2^n`. `b = x - n·ln2_hi - n·ln2_lo` is the
/// reduced argument in `[-ln2/2, ln2/2]`, and the degree-5 polynomial
/// evaluates `e^b - 1` there.
///
/// The input is clamped to `±EXP_CLAMP` first, which is what lets the
/// overflow branch of the original be dropped; read [`EXP_CLAMP`] before
/// widening it.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn expf_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vminq_f32(
        vmaxq_f32(x, vdupq_n_f32(-EXP_CLAMP)),
        vdupq_n_f32(EXP_CLAMP),
    );
    let r = vdupq_n_f32(EXP_SHIFT);
    let z = vfmaq_f32(r, x, vdupq_n_f32(EXP_LOG2E));
    let n = vsubq_f32(z, r);
    // `b = x - n*ln2_hi - n*ln2_lo`; `vfmsq_f32(a, b, c) == a - b*c`.
    let b = vfmsq_f32(
        vfmsq_f32(x, n, vdupq_n_f32(EXP_LN2_HI)),
        n,
        vdupq_n_f32(EXP_LN2_LO),
    );
    let e = vshlq_n_u32::<23>(vreinterpretq_u32_f32(z));
    let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
    let u = vmulq_f32(b, b);
    let j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(EXP_C0), b),
        vfmaq_f32(
            vfmaq_f32(vdupq_n_f32(EXP_C1), vdupq_n_f32(EXP_C2), b),
            vfmaq_f32(vdupq_n_f32(EXP_C3), vdupq_n_f32(EXP_C4), b),
            u,
        ),
        u,
    );
    vfmaq_f32(k, j, k)
}

/// One-lane [`expf_neon`], op for op, so a kernel's scalar tail computes
/// the same bits as its vector body.
///
/// A libm `expf` here instead would make the answer depend on where the
/// slice happens to end, and prefill and decode do not end in the same
/// place. `mul_add` is the scalar FMA the vector arms use lane-wise, and
/// both round once, so this is equality and not approximation --
/// `parallel_gated_activations_are_bit_identical_to_the_serial_form`
/// asserts it by pushing every element through this path.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn expf_scalar(x: f32) -> f32 {
    let x = x.clamp(-EXP_CLAMP, EXP_CLAMP);
    let z = x.mul_add(EXP_LOG2E, EXP_SHIFT);
    let n = z - EXP_SHIFT;
    let b = (-n).mul_add(EXP_LN2_LO, (-n).mul_add(EXP_LN2_HI, x));
    let k = f32::from_bits((z.to_bits() << 23).wrapping_add(1.0f32.to_bits()));
    let u = b * b;
    let j = EXP_C4
        .mul_add(b, EXP_C3)
        .mul_add(u, EXP_C2.mul_add(b, EXP_C1))
        .mul_add(u, EXP_C0 * b);
    j.mul_add(k, k)
}

/// AVX2 sibling of [`expf_neon`]: same constants, same polynomial, same
/// two-sided clamp, eight lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn expf_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(-EXP_CLAMP)),
        _mm256_set1_ps(EXP_CLAMP),
    );
    let r = _mm256_set1_ps(EXP_SHIFT);
    let z = _mm256_fmadd_ps(x, _mm256_set1_ps(EXP_LOG2E), r);
    let n = _mm256_sub_ps(z, r);
    let b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(EXP_LN2_LO),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(EXP_LN2_HI), x),
    );
    let e = _mm256_slli_epi32::<23>(_mm256_castps_si256(z));
    let k = _mm256_castsi256_ps(_mm256_add_epi32(
        e,
        _mm256_castps_si256(_mm256_set1_ps(1.0)),
    ));
    let u = _mm256_mul_ps(b, b);
    let j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(EXP_C4), b, _mm256_set1_ps(EXP_C3)),
            u,
            _mm256_fmadd_ps(_mm256_set1_ps(EXP_C2), b, _mm256_set1_ps(EXP_C1)),
        ),
        u,
        _mm256_mul_ps(_mm256_set1_ps(EXP_C0), b),
    );
    _mm256_fmadd_ps(j, k, k)
}

/// `x / (1 + exp(t))`, with the saturating branch [`EXP_CLAMP`] describes:
/// once `t` reaches the clamp the true quotient is below `1.5e-37`, and
/// returning it rather than dividing by a saturated exponential is what
/// keeps a large negative input from producing a real number.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn gate_by_exp_scalar(x: f32, t: f32) -> f32 {
    if t >= EXP_CLAMP {
        0.0
    } else {
        x / (1.0 + expf_scalar(t))
    }
}

/// `t = -2·K·(g + C·g³)`, written as `g·(A + B·g²)` so it is one multiply
/// and one FMA. GELU's exponent argument.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn gelu_exp_arg(g: f32) -> f32 {
    g * GELU_B.mul_add(g * g, GELU_A)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gelu_mul_neon(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let nv = n & !3;
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let a = vdupq_n_f32(GELU_A);
    let b = vdupq_n_f32(GELU_B);
    let sat = vdupq_n_f32(EXP_CLAMP);
    let mut i = 0;
    while i < nv {
        let g = vld1q_f32(gate.as_ptr().add(i));
        // `t = g * (A + B*g^2)`, i.e. `-2*K*(g + C*g^3)`.
        let t = vmulq_f32(g, vfmaq_f32(a, b, vmulq_f32(g, g)));
        let y = vdivq_f32(g, vaddq_f32(one, expf_neon(t)));
        let y = vbslq_f32(vcgeq_f32(t, sat), zero, y);
        vst1q_f32(
            out.as_mut_ptr().add(i),
            vmulq_f32(y, vld1q_f32(up.as_ptr().add(i))),
        );
        i += 4;
    }
    for j in nv..n {
        let g = *gate.get_unchecked(j);
        *out.get_unchecked_mut(j) = gate_by_exp_scalar(g, gelu_exp_arg(g)) * *up.get_unchecked(j);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn silu_mul_neon(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let nv = n & !3;
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let sat = vdupq_n_f32(EXP_CLAMP);
    let mut i = 0;
    while i < nv {
        let g = vld1q_f32(gate.as_ptr().add(i));
        let t = vnegq_f32(g);
        let y = vdivq_f32(g, vaddq_f32(one, expf_neon(t)));
        let y = vbslq_f32(vcgeq_f32(t, sat), zero, y);
        vst1q_f32(
            out.as_mut_ptr().add(i),
            vmulq_f32(y, vld1q_f32(up.as_ptr().add(i))),
        );
        i += 4;
    }
    for j in nv..n {
        let g = *gate.get_unchecked(j);
        *out.get_unchecked_mut(j) = gate_by_exp_scalar(g, -g) * *up.get_unchecked(j);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_mul_avx2(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let nv = n & !7;
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let a = _mm256_set1_ps(GELU_A);
    let b = _mm256_set1_ps(GELU_B);
    let sat = _mm256_set1_ps(EXP_CLAMP);
    let mut i = 0;
    while i < nv {
        let g = _mm256_loadu_ps(gate.as_ptr().add(i));
        let t = _mm256_mul_ps(g, _mm256_fmadd_ps(b, _mm256_mul_ps(g, g), a));
        let y = _mm256_div_ps(g, _mm256_add_ps(one, expf_avx2(t)));
        let y = _mm256_blendv_ps(y, zero, _mm256_cmp_ps::<_CMP_GE_OQ>(t, sat));
        _mm256_storeu_ps(
            out.as_mut_ptr().add(i),
            _mm256_mul_ps(y, _mm256_loadu_ps(up.as_ptr().add(i))),
        );
        i += 8;
    }
    for j in nv..n {
        let g = *gate.get_unchecked(j);
        *out.get_unchecked_mut(j) = gate_by_exp_scalar(g, gelu_exp_arg(g)) * *up.get_unchecked(j);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_mul_avx2(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let nv = n & !7;
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let neg = _mm256_set1_ps(-0.0);
    let sat = _mm256_set1_ps(EXP_CLAMP);
    let mut i = 0;
    while i < nv {
        let g = _mm256_loadu_ps(gate.as_ptr().add(i));
        // `-g` as a sign flip, so `-0.0` negates to `0.0` exactly as
        // NEON's `vnegq_f32` does and the two arms cannot disagree there.
        let t = _mm256_xor_ps(g, neg);
        let y = _mm256_div_ps(g, _mm256_add_ps(one, expf_avx2(t)));
        let y = _mm256_blendv_ps(y, zero, _mm256_cmp_ps::<_CMP_GE_OQ>(t, sat));
        _mm256_storeu_ps(
            out.as_mut_ptr().add(i),
            _mm256_mul_ps(y, _mm256_loadu_ps(up.as_ptr().add(i))),
        );
        i += 8;
    }
    for j in nv..n {
        let g = *gate.get_unchecked(j);
        *out.get_unchecked_mut(j) = gate_by_exp_scalar(g, -g) * *up.get_unchecked(j);
    }
}

/// Plain (non-RMS) LayerNorm -- ggml's `LLM_NORM` (as opposed to
/// `LLM_NORM_RMS`, what [`rms_norm`] implements): subtract the mean,
/// divide by the standard deviation, then apply an elementwise
/// affine `* weight + bias`. GLM-5.2's real DSA lightning indexer
/// normalizes its compressed key through exactly this
/// (`indexer_k_norm` carries both a `weight` *and* a `bias` GGUF
/// tensor -- confirmed against llama.cpp PR #23346/#25407's real
/// `create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight"|"bias", i),
/// ...)` calls and the `build_norm(indexer_k, ..., LLM_NORM, il)` call
/// site, `LLM_NORM` being ggml's plain-LayerNorm op, distinct from
/// every other norm in this codebase so far, which are all RMSNorm).
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), bias.len());
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .map(|((v, w), b)| (v - mean) * inv_std * w + b)
        .collect()
}

/// SiLU / swish activation: x * sigmoid(x). Used by the SwiGLU-style
/// gated MLP and MoE expert feed-forward blocks in this family of models.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Elementwise gated FFN combine: silu(gate) * up, the standard SwiGLU
/// pairing used inside both dense and MoE-expert feed-forward blocks.
///
/// Same rewrite as [`geglu`], and it reaches far more of the ledger:
/// [`silu`] was already the cancellation-free `x / (1 + exp(-x))` form,
/// so this changes only *which* exponential runs, from a scalar libm
/// `expf` per element to four or eight lanes at a time. Every SwiGLU
/// model on the CPU rows goes through here — TinyLlama, SmolLM2, Qwen,
/// Mistral, Llama — not just the Gemma family that named the todo.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    par_gated_chunks(gate, up, silu_mul)
}

/// ReLU, the gate nonlinearity of [`reglu`].
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Elementwise gated FFN combine: `relu(gate) * up`.
///
/// This is how the generic decoder spells llama.cpp's **ungated**
/// `LLM_FFN_RELU_SQR` FFN (`down(relu(up(x))^2)`, `arcee.cpp:123-128`,
/// `LLM_FFN_SEQ` with a null gate): the loader aliases `gate` to the
/// same `up` matrix, and `relu(x) * x` is `relu(x)^2` exactly -- `x*x`
/// for `x >= 0`, and `0 * x` for `x < 0`, which is a zero of one sign
/// or the other and vanishes in the down-projection's sum. Keeping it a
/// GLU form means every gated path (routed, placed, batched, slotted)
/// serves it with no branch; the two dense hot paths skip the aliased
/// matmul through `GluAct::ungated`.
///
/// No SIMD arm: `max` and a multiply auto-vectorise, and the relu forms
/// have no exponential to amortise.
pub fn reglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    par_gated_chunks(gate, up, |g, u, o| {
        for ((o, g), u) in o.iter_mut().zip(g).zip(u) {
            *o = relu(*g) * u;
        }
    })
}

/// Kimi K3's `situ` activation (`hidden_act: "situ"` in its real
/// `config.json`, registered as `ACT2FN["situ"] -> SituAndMul` in
/// `modeling_kimi_linear.py`): `beta*tanh(gate/beta)*sigmoid(gate) *
/// linear_beta*tanh(up/linear_beta)`. Not SiLU/SwiGLU -- a real,
/// non-obvious fact confirmed by reading Kimi K3's actual reference
/// source and config (`activation_situ_beta`=4.0,
/// `activation_situ_linear_beta`=25.0) rather than assuming the more
/// common SwiGLU convention every other model in this codebase uses.
pub fn situ_and_mul(gate: &[f32], up: &[f32], beta: f32, linear_beta: f32) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let situ_a = beta * (g / beta).tanh() * (1.0 / (1.0 + (-g).exp()));
            let up_t = linear_beta * (u / linear_beta).tanh();
            situ_a * up_t
        })
        .collect()
}

#[cfg(test)]
mod tests {
    /// The vocabulary-sized cap is split across threads; every element
    /// must still be exactly `sc * tanh(x / sc)`, the scalar expression.
    #[test]
    fn a_vocabulary_sized_softcap_equals_the_scalar_expression() {
        let sc = 30.0f32;
        let raw: Vec<f32> = (0..262_144)
            .map(|i| ((i as f32) * 0.37).sin() * 80.0)
            .collect();
        let mut got = raw.clone();
        softcap_inplace(&mut got, sc);
        for (g, r) in got.iter().zip(&raw) {
            assert_eq!(g.to_bits(), (sc * (r * (1.0 / sc)).tanh()).to_bits());
        }
    }

    use super::*;

    /// Decode and prefill must compute the same bits from the same
    /// weights. Decode runs one row through the serial arm, prefill runs
    /// `batch x ffn_dim` through the Rayon arm, and the vector kernel sees
    /// a different tail split in each -- so the claim is not "chunking is
    /// elementwise, therefore fine", it is that the kernel's own vector
    /// and scalar-tail paths agree everywhere the split can fall.
    ///
    /// The reference is the kernel applied to one element at a time, which
    /// puts every lane through the scalar tail. It is deliberately NOT
    /// `silu(g) * u`: the scalar [`silu`] and [`gelu`] keep libm and are
    /// still the reference for the CUDA and Metal ports and for `kda` /
    /// `gdn`, while these two run the vector exponential. The distance
    /// between them is what the accuracy test below measures.
    #[test]
    fn parallel_gated_activations_are_bit_identical_to_the_serial_form() {
        // Straddle GATED_PAR_MIN so both arms are covered, and use a
        // length that is not a multiple of the chunk size or of a vector
        // width. Elementwise with no reduction, so "close enough" is not
        // the bar: every bit must match, or prefill and decode disagree
        // on the same model.
        for n in [7usize, GATED_PAR_MIN - 1, GATED_PAR_MIN, 300_007] {
            let gate: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.0037 - 4.0).sin() * 6.0)
                .collect();
            let up: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.0041 + 1.0).cos() * 2.5)
                .collect();

            let one_at_a_time = |f: fn(&[f32], &[f32], &mut [f32])| -> Vec<f32> {
                let mut out = vec![0f32; n];
                for i in 0..n {
                    f(&gate[i..i + 1], &up[i..i + 1], &mut out[i..i + 1]);
                }
                out
            };
            assert_eq!(
                swiglu(&gate, &up),
                one_at_a_time(silu_mul),
                "swiglu at n = {n}"
            );
            assert_eq!(
                geglu(&gate, &up),
                one_at_a_time(gelu_mul),
                "geglu at n = {n}"
            );
        }
    }

    /// The bar for replacing a libm call in a hot loop is not "the output
    /// did not change" -- it did change -- but "the output is no less
    /// accurate than what it replaced", measured against a reference
    /// neither side can flatter.
    ///
    /// **The obvious reference is the wrong one, and finding that out is
    /// half of what this test is for.** Evaluating
    /// `0.5·x·(1 + tanh(u))` in `f64` cancels for the same reason the
    /// `f32` version does: `tanh(u) → -1` for `x` negative, so `1 + tanh(u)`
    /// is a difference of nearly equal numbers in `f64` too, and past
    /// `x ≈ -7` the "reference" is itself noise. It scored the vector form
    /// at `1.97e22` relative error against a value that had already
    /// collapsed. The reference here is therefore the algebraically
    /// identical `x / (1 + exp(-2u))` in `f64`, which has no subtraction
    /// in it at all.
    ///
    /// Two metrics, because they answer different questions and the
    /// rewrite does not win both.
    ///
    /// 1. Scaled by `|x|` -- how much absolute error this contributes to
    ///    the sum it feeds. GELU: `1.13e-7` vs `9.74e-8`, so the vector
    ///    form is 1.16x behind, both at one to two ulp of `|x|`; the extra
    ///    rounding is the divide. SiLU: identical to the last digit.
    /// 2. Scaled by `|gelu(x)|` -- whether the returned number is itself
    ///    right. Here the rewrite is not marginally better but categorically
    ///    so: over 240001 samples the tanh form loses more than 0.1% of the
    ///    value at 5047 of them, up to and including every significant bit,
    ///    and the new form at none.
    ///
    /// So the assertions are: nobody may lose a value (metric 2), and the
    /// absolute contribution may not slip by more than a quarter ulp
    /// (metric 1). Both numbers above are what the code measures today, so
    /// a regression in either direction fires.
    #[test]
    fn geglu_and_swiglu_are_no_less_accurate_than_the_libm_forms_they_replace() {
        /// Worst error under both metrics, plus how many samples lost more
        /// than 0.1% of the value.
        ///
        /// The `1e-30` floor on metric 2 is not a fudge: below it the
        /// activation cannot change any `f32` sum it participates in, and
        /// both forms flush to zero there anyway, which would otherwise
        /// score as 100% error for both and hide the real difference.
        fn sweep(
            reference: fn(f64) -> f64,
            vector: fn(f32) -> f32,
            libm: fn(f32) -> f32,
        ) -> (f64, f64, u32, u32) {
            let (mut v_abs, mut l_abs) = (0f64, 0f64);
            let (mut v_lost, mut l_lost) = (0u32, 0u32);
            let mut i = -120_000i32;
            while i <= 120_000 {
                let x = i as f32 * 0.001;
                let want = reference(x as f64);
                let (v, l) = (vector(x) as f64 - want, libm(x) as f64 - want);
                let scale = (x as f64).abs().max(1e-30);
                v_abs = v_abs.max(v.abs() / scale);
                l_abs = l_abs.max(l.abs() / scale);
                if want.abs() >= 1e-30 {
                    v_lost += u32::from(v.abs() / want.abs() > 1e-3);
                    l_lost += u32::from(l.abs() / want.abs() > 1e-3);
                }
                i += 1;
            }
            (v_abs, l_abs, v_lost, l_lost)
        }

        /// The tanh-approximation GELU rearranged so it does not cancel,
        /// with `sqrt(2/pi)` to full precision rather than the `f32`
        /// constant either implementation rounds it to.
        fn gelu_f64(x: f64) -> f64 {
            const K: f64 = 0.797_884_560_802_865_4;
            let u = K * (x + 0.044_715 * x * x * x);
            x / (1.0 + (-2.0 * u).exp())
        }
        fn silu_f64(x: f64) -> f64 {
            x / (1.0 + (-x).exp())
        }
        fn vec_gelu(x: f32) -> f32 {
            let mut out = [0f32; 1];
            gelu_mul(&[x], &[1.0], &mut out);
            out[0]
        }
        fn vec_silu(x: f32) -> f32 {
            let mut out = [0f32; 1];
            silu_mul(&[x], &[1.0], &mut out);
            out[0]
        }

        for (what, reference, vector, libm) in [
            (
                "GELU",
                gelu_f64 as fn(f64) -> f64,
                vec_gelu as fn(f32) -> f32,
                gelu as fn(f32) -> f32,
            ),
            ("SiLU", silu_f64, vec_silu, silu),
        ] {
            let (v_abs, l_abs, v_lost, l_lost) = sweep(reference, vector, libm);
            assert_eq!(
                v_lost, 0,
                "vector {what} lost more than 0.1% of the value at {v_lost} samples \
                 (the form it replaces: {l_lost})"
            );
            assert!(
                v_lost <= l_lost,
                "vector {what} loses values the form it replaces kept: {v_lost} vs {l_lost}"
            );
            assert!(
                v_abs <= l_abs * 1.25,
                "vector {what} contributes more absolute error than the form it \
                 replaces: {v_abs:e} vs {l_abs:e}"
            );
        }
    }

    /// The clamp is only free where [`EXP_CLAMP`] says it is, and the
    /// saturating select is what makes the high side true.
    ///
    /// Without it a large negative input walks `bits(z) << 23` out of the
    /// exponent field, or -- worse, because it looks like an answer --
    /// divides an unbounded numerator by a saturated exponential and
    /// returns `-1.6e-8` for `gelu(-1e30)`, which should be `-0`. The
    /// tolerance is absolute rather than relative because everything here
    /// is either exactly the input or smaller than any `f32` sum can
    /// notice.
    #[test]
    fn the_two_sided_clamp_leaves_the_saturating_tails_correct() {
        fn gelu_f64(x: f64) -> f64 {
            const K: f64 = 0.797_884_560_802_865_4;
            let u = K * (x + 0.044_715 * x * x * x);
            x / (1.0 + (-2.0 * u).exp())
        }
        for x in [
            -1e30f32, -1e10, -1000.0, -120.0, -88.0, -12.0, -10.0, 10.0, 12.0, 20.0, 120.0, 1000.0,
            1e30,
        ] {
            let mut g = [0f32; 1];
            gelu_mul(&[x], &[1.0], &mut g);
            let mut s = [0f32; 1];
            silu_mul(&[x], &[1.0], &mut s);
            assert!(g[0].is_finite() || x.abs() > 1e20, "gelu({x}) = {}", g[0]);
            assert!(s[0].is_finite() || x.abs() > 1e20, "silu({x}) = {}", s[0]);

            // A `f64` reference that does not cancel; anything past the
            // clamp has collapsed far below what an `f32` sum resolves.
            let want_g = gelu_f64(x as f64);
            let want_s = (x as f64) / (1.0 + (-(x as f64)).exp());
            assert!(
                (g[0] as f64 - want_g).abs() <= 1e-30 + 1e-6 * want_g.abs(),
                "gelu({x}) = {} want {want_g:e}",
                g[0]
            );
            assert!(
                (s[0] as f64 - want_s).abs() <= 1e-30 + 1e-6 * want_s.abs(),
                "silu({x}) = {} want {want_s:e}",
                s[0]
            );
            // The two saturate to the identity at different places, and
            // the gap is the point of the `-2·K·(x + C·x³)` argument:
            // GELU's exponent runs away cubically, so it is already `x`
            // by 10, while SiLU's is linear and still `9.9995` there.
            if x >= 10.0 {
                assert_eq!(g[0], x, "gelu should saturate to the identity at {x}");
            }
            if x >= 20.0 {
                assert_eq!(s[0], x, "silu should saturate to the identity at {x}");
            }
        }
    }

    #[test]
    fn layer_norm_zero_mean_unit_var_input_is_unchanged_by_weight_one_bias_zero() {
        // A hand-picked vector with mean 0 and population variance 1:
        // [-1, 1] has mean 0, var = ((1)+(1))/2 = 1.
        let x = vec![-1.0, 1.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![0.0, 0.0];
        let out = layer_norm(&x, &weight, &bias, 0.0);
        assert!((out[0] - (-1.0)).abs() < 1e-4);
        assert!((out[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_applies_affine_weight_and_bias_after_normalizing() {
        let x = vec![-1.0, 1.0];
        let weight = vec![2.0, 3.0];
        let bias = vec![10.0, -10.0];
        let out = layer_norm(&x, &weight, &bias, 0.0);
        // normalized = [-1, 1] (same as above), then * weight + bias:
        assert!((out[0] - (-2.0 + 10.0)).abs() < 1e-4);
        assert!((out[1] - (3.0 - 10.0)).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_constant_input_is_zero_before_bias() {
        // Zero variance input: every normalized value must be exactly 0
        // (mean-subtracted, so all zero) regardless of eps, then affine.
        let x = vec![5.0, 5.0, 5.0];
        let weight = vec![1.0, 1.0, 1.0];
        let bias = vec![0.25, 0.25, 0.25];
        let out = layer_norm(&x, &weight, &bias, 1e-5);
        for v in out {
            assert!((v - 0.25).abs() < 1e-4);
        }
    }

    #[test]
    fn matmul_identity_returns_input() {
        // a = [[1,2],[3,4]], b_t = identity transposed = identity
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let identity = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let out = matmul_f32(&a, &identity);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_known_values() {
        // a = [1, 2, 3] (1x3), b_t = [[1,1,1]] (1x3) => dot = 6
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let b_t = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]);
        let out = matmul_f32(&a, &b_t);
        assert_eq!(out.shape, vec![1, 1]);
        assert_eq!(out.data[0], 6.0);
    }

    #[test]
    fn matmul_single_row_batch_matches_sequential_dot_products() {
        // m=1 exercises the dedicated single-token-decode path
        // (parallelized over n, not m) -- check it against a plain
        // sequential dot product per output column, not just m=1/n=1.
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let b_t = Tensor::new(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0],
            vec![4, 3],
        );
        let out = matmul_f32(&a, &b_t);
        assert_eq!(out.shape, vec![1, 4]);
        assert_eq!(out.data, vec![1.0, 2.0, 6.0, 2.0]);
    }

    #[test]
    fn rms_norm_unit_weight_preserves_direction() {
        let x = vec![3.0, 4.0];
        let w = vec![1.0, 1.0];
        let out = rms_norm(&x, &w, 1e-6);
        // ratio between components should be preserved
        assert!((out[0] / out[1] - 3.0 / 4.0).abs() < 1e-4);
    }

    /// Expected bits are `ggml_rms_norm`'s at the pin (libggml-cpu on an
    /// M2, one thread), not a re-derivation. `1e8 + 1` is `1e8` in f32,
    /// so a single-precision sum of these squares loses all 63 ones and
    /// lands on a scale 0x3a51b717; ggml's f64 sum keeps them (#476).
    #[test]
    fn rms_scale_is_ggmls_f64_ordered_sum() {
        let mut x = vec![1.0f32; 64];
        x[0] = 1e4;
        let scale = rms_scale(&x, 1e-5);
        assert_eq!(scale.to_bits(), 0x3a51_b713);
        let f32_sum: f32 = x.iter().map(|v| v * v).sum();
        assert_ne!(
            (1.0 / (f32_sum / 64.0 + 1e-5).sqrt()).to_bits(),
            scale.to_bits()
        );
        let y = rms_norm(&x, &[1.0; 64], 1e-5);
        assert_eq!((y[0].to_bits(), y[1].to_bits()), (0x40ff_fffb, 0x3a51_b713));
    }

    /// The same, on a 2048-wide row with no structure to it: the first
    /// four outputs of `ggml_rms_norm` at the pin, eps 1e-6.
    #[test]
    fn rms_norm_matches_ggml_on_a_hidden_sized_row() {
        let x: Vec<f32> = (0..2048)
            .map(|i| ((i * 7919) % 2003) as f32 / 1000.0 - 1.0)
            .collect();
        let y = rms_norm(&x, &vec![1.0; 2048], 1e-6);
        let got: Vec<u32> = y[..4].iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, [0xbfdd_3007, 0x3fc9_47dd, 0x3fb4_b5d5, 0x3fa0_23cb]);
        let per_head = rms_norm_per_head(&x, &vec![1.0; 2048], 2048, 1e-6);
        assert_eq!(per_head, y);
    }

    #[test]
    fn silu_is_zero_at_zero_and_monotonic_ish() {
        assert!((silu(0.0)).abs() < 1e-6);
        assert!(silu(5.0) > silu(1.0));
    }

    // Golden values independently computed in Python from the same
    // formula transcribed from Kimi K3's real `SituAndMul` source
    // (`beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)`),
    // using its real config values (`activation_situ_beta`=4.0,
    // `activation_situ_linear_beta`=25.0).
    #[test]
    fn situ_and_mul_matches_independent_python_reference() {
        let cases = [
            (0.0f32, 0.0f32, 0.0f32),
            (2.0, -3.0, -4.861_066_3),
            (-1.5, 10.0, -2.483_860_7),
        ];
        for (gate, up, expected) in cases {
            let got = situ_and_mul(&[gate], &[up], 4.0, 25.0)[0];
            assert!(
                (got - expected).abs() < 1e-4,
                "situ({gate},{up}): rust={got} python={expected}"
            );
        }
    }
}
