//! Rotary position embedding (RoPE, both the split-half `apply_rope`
//! and interleaved `apply_rope_interleaved` conventions) and
//! grouped-query causal attention (GQA). This is the "vanilla"
//! attention path used as the correctness baseline.
//! `causal_mla_attention`/`causal_mla_attention_sparse` add
//! DeepSeek-style latent attention and its DSA sparse-selection variant
//! (GLM-5.2, DeepSeek V3.2/V4); both mechanisms are now backed by real,
//! public reference implementations (see docs/MODELS.md).
//! `frink_models::mla`/`frink_models::glm_dsa` compose these
//! primitives into full RoPE-carrying MLA forward passes.

use crate::cache::PagedKvStore;
use crate::kv_store::KvRows;

/// The RoPE frequency table for `(theta, dim)`, computed once per
/// thread.
///
/// `1.0 / theta.powf(2i/dim)` depends on nothing but the band index and
/// two values that are fixed for the life of a model, yet it was
/// evaluated inside the band loop on every call. For Llama-3-8B decode
/// that is (32 query + 8 kv heads) x 64 bands x 32 layers, roughly 82k
/// `powf` per token, all producing the same couple of 64-entry tables.
///
/// Thread-local rather than a shared map because this is a per-token
/// path and a mutex here would trade one cost for another. The tables
/// are tens of floats, so the duplication across worker threads is
/// nothing.
///
/// The expression is unchanged, so the values are bit-identical to what
/// the loop produced.
fn rope_freqs(theta: f32, dim: usize) -> std::rc::Rc<Vec<f32>> {
    /// Keyed on `(theta bits, dim)`, both of which are fixed per model.
    type FreqTables = std::collections::HashMap<(u32, usize), std::rc::Rc<Vec<f32>>>;
    thread_local! {
        static CACHE: std::cell::RefCell<FreqTables> =
            std::cell::RefCell::new(FreqTables::new());
    }
    // Keyed on the bit pattern: theta is a config value copied around,
    // never computed, so equal thetas are bit-equal.
    let key = (theta.to_bits(), dim);
    CACHE.with(|c| {
        if let Some(hit) = c.borrow().get(&key) {
            return std::rc::Rc::clone(hit);
        }
        let table: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
            .collect();
        let table = std::rc::Rc::new(table);
        c.borrow_mut().insert(key, std::rc::Rc::clone(&table));
        table
    })
}

/// Applies rotary position embedding in place to a single head's vector,
/// split-half (GPT-NeoX / `LLAMA_ROPE_TYPE_NEOX`) style: each pair
/// `(i, i+half)` is rotated together, for position `pos` with base
/// `theta`. This is what llama.cpp calls NEOX-style RoPE (used by e.g.
/// DeepSeek-V3.2's lightning indexer); see [`apply_rope_interleaved`]
/// for the other real convention.
pub fn apply_rope(vec: &mut [f32], pos: usize, theta: f32) {
    let dim = vec.len();
    let half = dim / 2;
    let freqs = rope_freqs(theta, dim);
    for i in 0..half {
        let freq = freqs[i];
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = vec[i];
        let b = vec[i + half];
        vec[i] = a * cos - b * sin;
        vec[i + half] = a * sin + b * cos;
    }
}

/// Inverse of [`apply_rope`] (split-half / NeoX): rotates each pair by
/// `-angle`. DeepSeek V4 applies this ("derope" / `ggml_rope_ext_back`)
/// to the rope slice of attention output before the grouped `wo_a`
/// projection — see `.scratch/NOTES_DS4_INFERENCE.md`.
pub fn apply_rope_back(vec: &mut [f32], pos: usize, theta: f32) {
    let dim = vec.len();
    let half = dim / 2;
    let freqs = rope_freqs(theta, dim);
    for i in 0..half {
        let freq = freqs[i];
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = vec[i];
        let b = vec[i + half];
        // Inverse of (a,b) -> (a cos - b sin, a sin + b cos).
        vec[i] = a * cos + b * sin;
        vec[i + half] = -a * sin + b * cos;
    }
}

/// Inverse of [`apply_rope_interleaved`] (adjacent-pair / Norm RoPE).
pub fn apply_rope_interleaved_back(vec: &mut [f32], pos: usize, theta: f32) {
    let dim = vec.len();
    let half = dim / 2;
    let freqs = rope_freqs(theta, dim);
    for i in 0..half {
        let freq = freqs[i];
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = vec[2 * i];
        let b = vec[2 * i + 1];
        vec[2 * i] = a * cos + b * sin;
        vec[2 * i + 1] = -a * sin + b * cos;
    }
}

/// SIMD `q·k` for one attention head. NEON (`vfmaq_f32`) / AVX2+FMA
/// (`_mm256_fmadd_ps`) 4-/8-wide accumulation with a scalar tail, falling
/// back to a plain scalar sum elsewhere. The grouped accumulation
/// reassociates the sum, so results match the scalar dot only within
/// float noise -- which the online-softmax path already tolerates.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { dot_f32_neon(a, b) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// [`dot_f32`] resolved ONCE to the arm this CPU takes, for a caller
/// that dots many rows against one vector and should not re-run the
/// feature probe per row (`matmul::matmul_f32`).
pub(crate) fn dot_f32_fn() -> fn(&[f32], &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON was just detected on this CPU, and the
            // returned function is only ever called on this process.
            return |a, b| unsafe { dot_f32_neon(a, b) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA were just detected on this CPU.
            return |a, b| unsafe { dot_f32_avx2(a, b) };
        }
    }
    |a, b| a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc = vfmaq_f32(acc, va, vb);
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut sum = hsum256_ps(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Horizontal sum of an AVX2 f32 vector. Factored out of
/// [`dot_f32_avx2`] so [`qk_tile_avx2`] can close its register tile with
/// the *same* reduction, which is what keeps the tiled scores
/// bit-identical to the row-at-a-time ones.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps(acc: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut s128 = _mm_add_ps(lo, hi);
    s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    s128 = _mm_add_ss(s128, _mm_shuffle_ps(s128, s128, 0x55));
    _mm_cvtss_f32(s128)
}

/// Online (flash-style) softmax·V accumulate for one head: one pass over
/// K/V, no `seq_len` score buffer. Numerically matches classic
/// max-subtract softmax within float noise (see unit tests).
///
/// When `attn_softcap` is `Some(sc)` with `sc > 0`, each score is remapped
/// with Gemma-2-style `sc * tanh(score / sc)` before the online softmax
/// (llama.cpp `attention.logit_softcapping`).
///
/// When `sink` is `Some(s)`, one extra virtual key participates in the
/// softmax denominator with logit `s` and a **zero** value vector, so it
/// bleeds probability mass away from the real keys without contributing
/// anything to the output. That is gpt-oss's attention sink, and this is
/// exactly llama.cpp's own online form
/// (`ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_flash_attn_ext_f16`,
/// the `// sinks - apply only on the first kv-chunk` block):
///
/// ```text
/// if (s > M) { ms = expf(M - s); M = s; scale VKQ by ms; } else { vs = expf(s - M); }
/// S = S*ms + vs;
/// ```
///
/// The sink logit is *not* multiplied by `scale` — it is a learned logit
/// already in score space, matching both the flash-attention path above
/// and `ggml_compute_forward_soft_max_f32`, which applies `scale` to the
/// KQ row before taking `MAX(max, sk[head])`.
fn online_attn_accumulate(
    q_h: &[f32],
    scale: f32,
    out_h: &mut [f32],
    attn_softcap: Option<f32>,
    sink: Option<f32>,
    mut for_each_kv: impl FnMut(&mut dyn FnMut(&[f32], &[f32], f32)),
) {
    // `q_h` is one K-width head and `out_h` one V-width head; the two
    // agree everywhere but MiMo-V2, and nothing here needs them to.
    // The third visitor argument is an ADDITIVE score bias for that
    // key -- ALiBi's `slope * (p_key - p_query)` (`crate::alibi`), or
    // `0.0` -- applied after the scale and the softcap, where
    // `ggml_soft_max_ext` adds `slope * mask`.
    let mut m = f32::NEG_INFINITY;
    let mut l = 0f32;
    out_h.fill(0.0);
    for_each_kv(&mut |k_t, v_t, bias| {
        let mut s = dot_f32(q_h, k_t) * scale;
        if let Some(sc) = attn_softcap.filter(|&c| c > 0.0) {
            s = sc * (s / sc).tanh();
        }
        s += bias;
        let m_new = m.max(s);
        let alpha = (m - m_new).exp();
        let p = (s - m_new).exp();
        l = l * alpha + p;
        axpy_scale(out_h, alpha, v_t, p);
        m = m_new;
    });
    if let Some(s) = sink {
        let m_new = m.max(s);
        let alpha = (m - m_new).exp();
        l = l * alpha + (s - m_new).exp();
        scale_inplace(out_h, alpha);
    }
    if l > 0.0 {
        let inv = 1.0 / l;
        scale_inplace(out_h, inv);
    }
}

/// `out[i] = out[i] * alpha + p * v[i]` (online-softmax V accumulate).
#[inline]
fn axpy_scale(out: &mut [f32], alpha: f32, v: &[f32], p: f32) {
    debug_assert_eq!(out.len(), v.len());
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { axpy_scale_neon(out, alpha, v, p) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { axpy_scale_avx2(out, alpha, v, p) };
            return;
        }
    }
    for (o, &vv) in out.iter_mut().zip(v) {
        *o = *o * alpha + p * vv;
    }
}

/// `out[i] += p * v[i]` (plain axpy; the blocked-softmax V accumulate,
/// which never rescales what is already accumulated).
#[inline]
fn axpy(out: &mut [f32], v: &[f32], p: f32) {
    debug_assert_eq!(out.len(), v.len());
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { axpy_neon(out, v, p) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { axpy_avx2(out, v, p) };
            return;
        }
    }
    for (o, &vv) in out.iter_mut().zip(v) {
        *o += p * vv;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_neon(out: &mut [f32], v: &[f32], p: f32) {
    use std::arch::aarch64::*;
    let n = out.len();
    let vp = vdupq_n_f32(p);
    let mut i = 0;
    while i + 4 <= n {
        let o = vld1q_f32(out.as_ptr().add(i));
        let vv = vld1q_f32(v.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vfmaq_f32(o, vv, vp));
        i += 4;
    }
    while i < n {
        out[i] += p * v[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(out: &mut [f32], v: &[f32], p: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    let vp = _mm256_set1_ps(p);
    let mut i = 0;
    while i + 8 <= n {
        let o = _mm256_loadu_ps(out.as_ptr().add(i));
        let vv = _mm256_loadu_ps(v.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_fmadd_ps(vv, vp, o));
        i += 8;
    }
    while i < n {
        out[i] += p * v[i];
        i += 1;
    }
}

#[inline]
fn scale_inplace(x: &mut [f32], s: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { scale_inplace_neon(x, s) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { scale_inplace_avx2(x, s) };
            return;
        }
    }
    for v in x.iter_mut() {
        *v *= s;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_scale_neon(out: &mut [f32], alpha: f32, v: &[f32], p: f32) {
    use std::arch::aarch64::*;
    let n = out.len();
    let va = vdupq_n_f32(alpha);
    let vp = vdupq_n_f32(p);
    let mut i = 0;
    while i + 4 <= n {
        let o = vld1q_f32(out.as_ptr().add(i));
        let vv = vld1q_f32(v.as_ptr().add(i));
        let r = vfmaq_f32(vmulq_f32(o, va), vv, vp);
        vst1q_f32(out.as_mut_ptr().add(i), r);
        i += 4;
    }
    while i < n {
        out[i] = out[i] * alpha + p * v[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_inplace_neon(x: &mut [f32], s: f32) {
    use std::arch::aarch64::*;
    let n = x.len();
    let vs = vdupq_n_f32(s);
    let mut i = 0;
    while i + 4 <= n {
        let v = vld1q_f32(x.as_ptr().add(i));
        vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(v, vs));
        i += 4;
    }
    while i < n {
        x[i] *= s;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_scale_avx2(out: &mut [f32], alpha: f32, v: &[f32], p: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    let va = _mm256_set1_ps(alpha);
    let vp = _mm256_set1_ps(p);
    let mut i = 0;
    while i + 8 <= n {
        let o = _mm256_loadu_ps(out.as_ptr().add(i));
        let vv = _mm256_loadu_ps(v.as_ptr().add(i));
        let r = _mm256_fmadd_ps(vv, vp, _mm256_mul_ps(o, va));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), r);
        i += 8;
    }
    while i < n {
        out[i] = out[i] * alpha + p * v[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_inplace_avx2(x: &mut [f32], s: f32) {
    use std::arch::x86_64::*;
    let n = x.len();
    let vs = _mm256_set1_ps(s);
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(v, vs));
        i += 8;
    }
    while i < n {
        x[i] *= s;
        i += 1;
    }
}

/// Same split-half rotation as [`apply_rope`], but each frequency band
/// `i` has its angle divided by `freq_factors[i]` before the rotation --
/// Llama 3/3.1/3.2's real per-band RoPE frequency correction (the
/// `rope_freqs.weight` GGUF tensor, `n_rot/2` elements, `TENSOR_NOT_REQUIRED`
/// so most non-Llama-3 checkpoints don't carry it). Confirmed against
/// real llama.cpp source, not guessed: `ggml_rope_cache_init`
/// (`ggml/src/ggml-cpu/ops.cpp`) computes `theta/freq_factors[i0/2]`
/// per band before `rope_yarn`. `freq_factors` all-`1.0` is
/// mathematically identical to plain `apply_rope` (pinned by
/// `rope_with_all_ones_freq_factors_matches_plain_rope`).
///
/// Found via a real end-to-end run: serving a real
/// Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf checkpoint (this tensor
/// present and non-trivial) produced short answers correctly but
/// degenerated into a spurious early EOS a few dozen tokens into a
/// longer generation -- a real, independent llama.cpp oracle run
/// against the exact same file continued correctly with no early stop.
/// Root cause: every RoPE angle was computed without this per-band
/// correction, an error that compounds with position and eventually
/// produces wrong logits.
pub fn apply_rope_with_freq_factors(vec: &mut [f32], pos: usize, theta: f32, freq_factors: &[f32]) {
    let dim = vec.len();
    let half = dim / 2;
    assert_eq!(
        freq_factors.len(),
        half,
        "freq_factors must have one entry per rotation band (dim/2)"
    );
    let freqs = rope_freqs(theta, dim);
    for i in 0..half {
        let freq = freqs[i];
        let angle = pos as f32 * freq / freq_factors[i];
        let (sin, cos) = angle.sin_cos();
        let a = vec[i];
        let b = vec[i + half];
        vec[i] = a * cos - b * sin;
        vec[i + half] = a * sin + b * cos;
    }
}

/// Applies rotary position embedding in place, interleaved (GPT-J /
/// llama.cpp's `LLAMA_ROPE_TYPE_NORM`) style: adjacent pairs
/// `(2*i, 2*i+1)` are rotated together, rather than `apply_rope`'s
/// split-half pairing. GLM-5.2 uses this convention for both its main
/// attention (`rope_interleave: true`) and its lightning indexer
/// (`indexer_rope_interleave: true`) per its real `config.json`
/// (`huggingface.co/zai-org/GLM-5.2`) — confirmed against llama.cpp PR
/// #25407, which rotates the indexer with `LLAMA_ROPE_TYPE_NORM` where
/// DeepSeek-V3.2's PR #23346 uses `LLAMA_ROPE_TYPE_NEOX`.
pub fn apply_rope_interleaved(vec: &mut [f32], pos: usize, theta: f32) {
    let dim = vec.len();
    let half = dim / 2;
    let freqs = rope_freqs(theta, dim);
    for i in 0..half {
        let freq = freqs[i];
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = vec[2 * i];
        let b = vec[2 * i + 1];
        vec[2 * i] = a * cos - b * sin;
        vec[2 * i + 1] = a * sin + b * cos;
    }
}

/// Interleaved (GPT-J / `LLAMA_ROPE_TYPE_NORM`) RoPE with Llama 3/3.1/3.2's
/// per-band frequency correction -- the combination real llama.cpp uses
/// for `general.architecture = "llama"` checkpoints that carry
/// `rope_freqs.weight`. Pairing is adjacent `(2*i, 2*i+1)` as in
/// [`apply_rope_interleaved`]; each band's angle is divided by
/// `freq_factors[i]` as in [`apply_rope_with_freq_factors`].
/// `freq_factors` all-`1.0` is mathematically identical to plain
/// `apply_rope_interleaved` (pinned by
/// `rope_interleaved_with_all_ones_freq_factors_matches_plain_interleaved`).
pub fn apply_rope_interleaved_with_freq_factors(
    vec: &mut [f32],
    pos: usize,
    theta: f32,
    freq_factors: &[f32],
) {
    let dim = vec.len();
    let half = dim / 2;
    assert_eq!(
        freq_factors.len(),
        half,
        "freq_factors must have one entry per rotation band (dim/2)"
    );
    let freqs = rope_freqs(theta, dim);
    for i in 0..half {
        let freq = freqs[i];
        let angle = pos as f32 * freq / freq_factors[i];
        let (sin, cos) = angle.sin_cos();
        let a = vec[2 * i];
        let b = vec[2 * i + 1];
        vec[2 * i] = a * cos - b * sin;
        vec[2 * i + 1] = a * sin + b * cos;
    }
}

/// YaRN RoPE scaling exactly as a checkpoint declares it, in the shape
/// the reference reads out of `rope_scaling` (FreeToken
/// `python/freetoken/layers/rotary.py:139`, the `"yarn"` arm of
/// `_get_rope`). `beta_fast` / `beta_slow` / `truncate` carry that arm's
/// own defaults, because a real YaRN checkpoint usually declares only
/// `factor` and `original_max_position_embeddings`.
///
/// This type is only the declaration. The frequency rewrite it implies
/// is [`yarn_freq_factors`], whose output feeds
/// [`apply_rope_with_freq_factors`] /
/// [`apply_rope_interleaved_with_freq_factors`] like any other per-band
/// correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnScaling {
    /// `rope_scaling["factor"]`: how much longer the served context is
    /// than the context the checkpoint was trained at. `1.0` makes the
    /// whole rewrite a no-op (every band's divisor is `1.0`).
    pub factor: f32,
    /// `rope_scaling["beta_fast"]`, the number of full rotations that
    /// marks the *high*-frequency end of the correction ramp -- bands
    /// faster than this are left extrapolated. Reference default `32.0`.
    pub beta_fast: f32,
    /// `rope_scaling["beta_slow"]`, the rotation count marking the
    /// *low*-frequency end -- bands slower than this are fully
    /// interpolated. Reference default `1.0`.
    pub beta_slow: f32,
    /// `rope_scaling["original_max_position_embeddings"]`: the context
    /// the checkpoint was actually trained at, which is the length the
    /// rotation counts above are counted against.
    pub orig_max_pos: usize,
    /// `rope_scaling["truncate"]`, reference default `true`: floor the
    /// low end and ceil the high end of the correction range to whole
    /// band indices. `false` keeps the fractional range, which is the
    /// case the `low == high` nudge in [`yarn_correction_range`] exists
    /// for.
    pub truncate: bool,
}

impl YarnScaling {
    /// The reference's defaults for everything a checkpoint may omit
    /// (`rope_scaling.get("beta_fast", 32.0)`,
    /// `.get("beta_slow", 1.0)`, `.get("truncate", True)`). Using
    /// anything else here silently moves the correction range and
    /// therefore which dims extrapolate.
    pub fn new(factor: f32, orig_max_pos: usize) -> Self {
        YarnScaling {
            factor,
            beta_fast: 32.0,
            beta_slow: 1.0,
            orig_max_pos,
            truncate: true,
        }
    }
}

/// The (fractional) band index at which a frequency completes exactly
/// `num_rotations` full rotations over the checkpoint's *original*
/// trained context -- the reference's `_find_correction_dim`
/// (`rotary.py:161`):
/// `rotary_dim * ln(orig_max_pos / (num_rotations * 2π)) / (2 * ln(base))`.
///
/// Computed in `f64`: `ln` of a 128k context over `2 * ln(base)` is a
/// ratio of two large logs, and doing it in `f32` moves the floored /
/// ceiled band index by a whole band often enough to matter.
fn yarn_correction_dim(
    num_rotations: f64,
    rotary_dim: usize,
    base: f64,
    orig_max_pos: usize,
) -> f64 {
    rotary_dim as f64 * (orig_max_pos as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln()
        / (2.0 * base.ln())
}

/// The `[low, high]` band range the YaRN ramp interpolates across, as
/// the reference computes it (`rotary.py:167-179`): both ends from
/// [`yarn_correction_dim`], floored / ceiled when `truncate`, `low`
/// clamped up to `0`, and -- the load-bearing detail, called out in the
/// reference's own comment at `rotary.py:176` -- `high` clamped to
/// **`rotary_dim - 1`, not `rotary_dim / 2 - 1`**.
///
/// The ramp only has `rotary_dim / 2` entries, so a `high` above
/// `rotary_dim / 2 - 1` means the ramp never reaches `1.0`: the
/// longest-wavelength dims stay *partly* extrapolated. Clamping to the
/// ramp's own last index instead (the naive reading) forces the ramp to
/// hit `1.0` at the last band and fully interpolates dims the reference
/// deliberately leaves partly extrapolated -- a checkpoint-wide change
/// to the lowest frequencies, i.e. exactly the dims long-context
/// behaviour rides on. Pinned by
/// `yarn_high_is_clamped_to_rotary_dim_minus_one_not_half_minus_one`.
///
/// Returned as `f64` because `high` may be fractional: when the range
/// collapses (`low == high`, which is what `truncate: false` with
/// `beta_fast == beta_slow` produces) the reference nudges `high` by
/// `+0.001` rather than flooring the gap at `1`, and that nudge is what
/// makes the ramp a step at `low` instead of a division by zero.
pub fn yarn_correction_range(scaling: YarnScaling, rotary_dim: usize, base: f32) -> (f64, f64) {
    let base = base as f64;
    let mut low = yarn_correction_dim(
        scaling.beta_fast as f64,
        rotary_dim,
        base,
        scaling.orig_max_pos,
    );
    let mut high = yarn_correction_dim(
        scaling.beta_slow as f64,
        rotary_dim,
        base,
        scaling.orig_max_pos,
    );
    if scaling.truncate {
        low = low.floor();
        high = high.ceil();
    }
    low = low.max(0.0);
    high = high.min(rotary_dim as f64 - 1.0);
    if low == high {
        high += 0.001;
    }
    (low, high)
}

/// YaRN's frequency rewrite, expressed as the per-band **divisors**
/// [`apply_rope_with_freq_factors`] already consumes: one entry per
/// rotation band (`rotary_dim / 2`), each the number the band's RoPE
/// angle is divided by.
///
/// The reference rewrites the frequencies themselves
/// (`rotary.py:181-187`):
/// `inv_freq_new = (inv_freq / factor) * ramp + inv_freq * (1 - ramp)`
/// with `ramp = clamp((band - low) / (high - low), 0, 1)` over
/// `rotary_dim / 2` bands. Dividing an angle by `d` is scaling its
/// frequency by `1/d`, so the identical rewrite in divisor form is
/// `d = 1 / (ramp / factor + (1 - ramp))` -- exactly `1.0` on the
/// extrapolated (fast) bands and exactly `factor` on any fully
/// interpolated one. Keeping it in this form is what lets a YaRN
/// checkpoint ride the *existing* CPU and Metal RoPE paths (llama.cpp's
/// `rope_freqs` semantics: `theta / freq_factors[i]`) instead of needing
/// a second rotation kernel; it also composes with a checkpoint's own
/// per-band factors by multiplication.
///
/// Skipping this rewrite entirely -- what frink did before this
/// existed, since it read neither `rope.scaling.type` nor
/// `rope.scaling.factor` -- ropes a long-context YaRN checkpoint as if
/// it declared no scaling at all: correct near position 0 and
/// progressively wrong with position, which is the failure that looks
/// like quality "degrading over long prompts" rather than like a bug.
///
/// # Panics
/// If `rotary_dim` is odd or zero (a band would have no partner
/// channel), or `factor` is not positive (the divisor would be
/// non-finite and every angle with it).
pub fn yarn_freq_factors(scaling: YarnScaling, rotary_dim: usize, base: f32) -> Vec<f32> {
    assert!(
        rotary_dim > 0 && rotary_dim.is_multiple_of(2),
        "rotary_dim must be a positive even number of channels, got {rotary_dim}"
    );
    assert!(
        scaling.factor > 0.0,
        "YaRN factor must be positive, got {}",
        scaling.factor
    );
    let (low, high) = yarn_correction_range(scaling, rotary_dim, base);
    let factor = scaling.factor as f64;
    (0..rotary_dim / 2)
        .map(|band| {
            let ramp = ((band as f64 - low) / (high - low)).clamp(0.0, 1.0);
            // Reference form: inv_freq * (ramp / factor + (1 - ramp)).
            let freq_scale = ramp / factor + (1.0 - ramp);
            (1.0 / freq_scale) as f32
        })
        .collect()
}

/// The reference's `"proportional"` arm (`rotary.py:103`), in the same
/// per-band divisor form as [`yarn_freq_factors`].
///
/// Partial rope normally spaces its frequencies over the *rotated*
/// width (`base^(2i / rotary_dim)`, what [`apply_rope`] and friends
/// compute from the slice they are handed). The proportional arm spaces
/// them over the **full head** instead (`base^(2i / head_size)`) and
/// zeroes every band past `rotary_dim / 2` -- i.e. the untouched tail of
/// the head is exactly the tail this crate already leaves unrotated, so
/// only the spacing differs. The returned divisor,
/// `base^(2i/head_size - 2i/rotary_dim)`, converts one spacing into the
/// other and is all-`1.0` when `rotary_dim == head_size` (full rope,
/// where the two spacings coincide).
///
/// Using the wrong spacing for a checkpoint that declares this is not a
/// long-context-only error: every rotated band below the last is turned
/// at the wrong rate from position 1 onward.
///
/// # Panics
/// If `rotary_dim` is odd, zero, or wider than `head_size`.
pub fn proportional_freq_factors(head_size: usize, rotary_dim: usize, base: f32) -> Vec<f32> {
    assert!(
        rotary_dim > 0 && rotary_dim.is_multiple_of(2) && rotary_dim <= head_size,
        "rotary_dim {rotary_dim} must be positive, even, and no wider than head_size {head_size}"
    );
    let base = base as f64;
    (0..rotary_dim / 2)
        .map(|band| {
            let exponent =
                (2 * band) as f64 / head_size as f64 - (2 * band) as f64 / rotary_dim as f64;
            base.powf(exponent) as f32
        })
        .collect()
}

/// Single-token causal attention for one query against all cached
/// key/value positions (0..=pos), grouped-query style: `n_kv_heads` may
/// be fewer than `n_heads`, with each KV head shared by
/// `n_heads / n_kv_heads` query heads.
///
/// `q` is [n_heads, head_dim]; `k_cache`/`v_cache` are
/// [seq_len, n_kv_heads, head_dim] flattened row-major. Returns
/// [n_heads, head_dim].
pub fn causal_gqa_attention<'a>(
    q: &[f32],
    k_cache: impl Into<KvRows<'a>>,
    v_cache: impl Into<KvRows<'a>>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    causal_gqa_attention_softcap(
        q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, None,
    )
}

/// [`causal_gqa_attention`] with optional Gemma-2 attention logit softcap.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_softcap<'a>(
    q: &[f32],
    k_cache: impl Into<KvRows<'a>>,
    v_cache: impl Into<KvRows<'a>>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    attn_softcap: Option<f32>,
) -> Vec<f32> {
    causal_gqa_attention_row(
        q,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        head_dim,
        seq_len,
        None,
        None,
        attn_softcap,
        None,
    )
}

/// Sliding-window variant of [`causal_gqa_attention`]: the query (the
/// last cached position) sees only the most recent `window` positions,
/// including its own.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_windowed<'a>(
    q: &[f32],
    k_cache: impl Into<KvRows<'a>>,
    v_cache: impl Into<KvRows<'a>>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    window: usize,
) -> Vec<f32> {
    causal_gqa_attention_windowed_softcap(
        q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, window, None,
    )
}

/// [`causal_gqa_attention_windowed`] with optional attention logit softcap.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_windowed_softcap<'a>(
    q: &[f32],
    k_cache: impl Into<KvRows<'a>>,
    v_cache: impl Into<KvRows<'a>>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    window: usize,
    attn_softcap: Option<f32>,
) -> Vec<f32> {
    assert!(window > 0, "window must be positive");
    causal_gqa_attention_row(
        q,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        head_dim,
        seq_len,
        Some(window),
        None,
        attn_softcap,
        None,
    )
}

/// [`causal_gqa_attention`] with gpt-oss / MiMo-V2 attention sinks: one
/// learned logit per query head that joins the softmax and contributes
/// nothing to the output (`ggml_soft_max_add_sinks`). Carries no
/// softcap, as llama.cpp's sink arm carries none.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_sinks<'a>(
    q: &[f32],
    k_cache: impl Into<KvRows<'a>>,
    v_cache: impl Into<KvRows<'a>>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    window: Option<usize>,
    sinks: &[f32],
) -> Vec<f32> {
    causal_gqa_attention_row(
        q,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        head_dim,
        seq_len,
        window,
        Some(sinks),
        None,
        None,
    )
}

/// THE contiguous single-query GQA kernel: every arm the decode path
/// dispatches -- plain, softcapped, windowed, with sinks -- and the
/// contiguous twin of [`causal_gqa_attention_paged_sinks`].
///
/// One body rather than three, because the three it replaced were one
/// loop varied by a window bound and a sink term, and the V head width
/// would have been a fourth decoration to add to each of them. `q` is
/// `[n_heads, head_dim]`; `k_cache` is `[seq_len, n_kv_heads,
/// head_dim]`; `v_cache` is `[seq_len, n_kv_heads, v_head_dim]`; the
/// result is `[n_heads, v_head_dim]`. `v_head_dim == head_dim` for every
/// architecture but MiMo-V2 (`frink_models::kv_head_dims`), whose
/// `head_dim: 192, v_head_dim: 128` is why the two are two arguments:
/// the score is a dot over the K width and the accumulate is an axpy
/// over the V width, and nothing in between cares that they agree.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_row<'a>(
    q: &[f32],
    k_cache: impl Into<KvRows<'a>>,
    v_cache: impl Into<KvRows<'a>>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
    window: Option<usize>,
    sinks: Option<&[f32]>,
    attn_softcap: Option<f32>,
    alibi: Option<&[f32]>,
) -> Vec<f32> {
    let (k_cache, v_cache) = (k_cache.into(), v_cache.into());
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(k_cache.len(), seq_len * n_kv_heads * head_dim);
    assert_eq!(v_cache.len(), seq_len * n_kv_heads * v_head_dim);
    if let Some(sinks) = sinks {
        assert_eq!(
            sinks.len(),
            n_heads,
            "attention sinks are per query head (llama.cpp `attn_sinks` is {{n_head}})"
        );
    }
    if let Some(slopes) = alibi {
        assert_eq!(slopes.len(), n_heads, "ALiBi slopes are per query head");
    }

    let group_size = n_heads / n_kv_heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; n_heads * v_head_dim];
    // The query is the last cached position; a windowed layer sees only
    // the most recent `window` positions including its own.
    let start = match window {
        Some(w) => {
            assert!(w > 0, "window must be positive");
            seq_len.saturating_sub(w)
        }
        None => 0,
    };

    // ALiBi: the query is the last cached position, key `t` is
    // `(seq_len - 1) - t` behind it, and the bias is `-slope * distance`.
    let q_pos = seq_len as f32 - 1.0;
    // Over HEADS, because they share nothing: each reads its own slice
    // of `q` and its own KV group and writes its own slice of `out`.
    //
    // This loop was serial, and on a hybrid model it is the whole of
    // what a decode token still does on the host -- 6.8 ms of a 94 ms
    // Bonsai token across sixteen attention layers of twenty-four heads
    // each, while six cores sat idle
    // (`docs/plans/gdn-resident-state.md`).
    //
    // `min_len` is 1 because a head is already a real unit of work
    // (`seq_len` keys against `head_dim`), and the scheduler's own
    // published-work rule decides whether to fork at all: a short
    // prompt's tiny region stays on one core through
    // `crate::par::backend`, which is exactly the case a hard-coded
    // threshold here would have had to guess at.
    crate::par::chunks_mut(&mut out, v_head_dim, 1, |h, out_h| {
        let kv_h = h / group_size.max(1);
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let sink = sinks.map(|s| s[h]);
        let slope = alibi.map_or(0.0, |s| s[h]);
        // One scratch row per head for an f16 store to widen into; an f32
        // store borrows and never touches it (`crate::kv_store`).
        let (mut k_scratch, mut v_scratch) = (Vec::new(), Vec::new());
        online_attn_accumulate(q_h, scale, out_h, attn_softcap, sink, |visit| {
            for t in start..seq_len {
                let k_t = k_cache.read(
                    (t * n_kv_heads + kv_h) * head_dim..(t * n_kv_heads + kv_h + 1) * head_dim,
                    &mut k_scratch,
                );
                let v_t = v_cache.read(
                    (t * n_kv_heads + kv_h) * v_head_dim..(t * n_kv_heads + kv_h + 1) * v_head_dim,
                    &mut v_scratch,
                );
                visit(k_t, v_t, slope * (t as f32 - q_pos));
            }
        });
    });

    out
}

/// Prefill (multi-query) causal GQA: `q`/`k_cache`/`v_cache` are all length
/// `seq_len` in the time dimension. Query at position `t` attends only to
/// keys/values `0..=t` (same math as looping [`causal_gqa_attention`] per
/// token). Layout: q/out `[seq_len, n_heads, head_dim]`; k/v
/// `[seq_len, n_kv_heads, head_dim]`. Metal prefill kernels must match.
pub fn causal_gqa_attention_prefill(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * n_heads * head_dim);
    assert_eq!(k_cache.len(), seq_len * n_kv_heads * head_dim);
    assert_eq!(v_cache.len(), seq_len * n_kv_heads * head_dim);

    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv_heads * head_dim;
    let mut out = vec![0f32; seq_len * q_stride];
    for t in 0..seq_len {
        let q_t = &q[t * q_stride..(t + 1) * q_stride];
        let k_prefix = &k_cache[..(t + 1) * kv_stride];
        let v_prefix = &v_cache[..(t + 1) * kv_stride];
        let attn = causal_gqa_attention(
            q_t,
            k_prefix,
            v_prefix,
            n_heads,
            n_kv_heads,
            head_dim,
            t + 1,
        );
        out[t * q_stride..(t + 1) * q_stride].copy_from_slice(&attn);
    }
    out
}

/// Prefill attention parallelized over `(query, head)` slots. Same math as
/// calling [`causal_gqa_attention_softcap`] per query; used by the decoder
/// CPU pp path so Rayon owns the full `[n_q × n_heads]` grid instead of
/// only the query axis (better for large-head models like Phi-4).
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_prefill_shared_kv(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_q: usize,
    kv_prefix: usize,
    attn_softcap: Option<f32>,
) -> Vec<f32> {
    causal_gqa_attention_prefill_shared_kv_windowed(
        q,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        n_q,
        kv_prefix,
        attn_softcap,
        None,
    )
}

/// [`causal_gqa_attention_prefill_shared_kv`] with an optional sliding
/// window, so SWA models (Gemma-2/3, Mistral, Qwen2-MoE) get the same
/// blocked kernel instead of the per-query
/// [`causal_gqa_attention_windowed_softcap`] fallback.
///
/// `window = Some(w)` restricts the query at absolute position `p`
/// (`p = kv_prefix + b`) to keys `p + 1 - w ..= p`, matching
/// [`causal_gqa_attention_windowed_softcap`]'s
/// `window_start = seq_len.saturating_sub(window)` exactly — the
/// per-query function is called with `seq_len = p + 1`. `None` is
/// full causal.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_prefill_shared_kv_windowed(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_q: usize,
    kv_prefix: usize,
    attn_softcap: Option<f32>,
    window: Option<usize>,
) -> Vec<f32> {
    causal_gqa_attention_prefill_shared_kv_split(
        q,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        head_dim,
        n_q,
        kv_prefix,
        attn_softcap,
        window,
        None,
    )
}

/// THE batched prefill kernel; [`causal_gqa_attention_prefill_shared_kv_windowed`]
/// is this with one head width. `v_cache` is `[kv_len, n_kv_heads,
/// v_head_dim]` and the result `[n_q, n_heads, v_head_dim]`; the
/// score tile is over the K width and the PV tile over the V width
/// (`frink_models::kv_head_dims`).
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_prefill_shared_kv_split(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    v_head_dim: usize,
    n_q: usize,
    kv_prefix: usize,
    attn_softcap: Option<f32>,
    window: Option<usize>,
    alibi: Option<&[f32]>,
) -> Vec<f32> {
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv_heads * head_dim;
    let v_stride = n_kv_heads * v_head_dim;
    let out_stride = n_heads * v_head_dim;
    assert_eq!(q.len(), n_q * q_stride);
    if let Some(slopes) = alibi {
        assert_eq!(slopes.len(), n_heads, "ALiBi slopes are per query head");
    }
    let kv_len = kv_prefix + n_q;
    assert!(k_cache.len() >= kv_len * kv_stride);
    assert!(v_cache.len() >= kv_len * v_stride);

    let group_size = n_heads / n_kv_heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; n_q * out_stride];

    // Blocked three-pass attention in llama.cpp's CPU shape: `KQ` as one
    // real `ggml_mul_mat`, one vectorized softmax over each score row,
    // then `KQV` as a second `ggml_mul_mat`. Tasks own a block of
    // queries for one head, so the K/V rows they stream stay hot across
    // the block; the raw pointer only bridges Send/Sync -- tasks write
    // disjoint `(query, head)` slices.
    struct OutPtr(*mut f32);
    unsafe impl Send for OutPtr {}
    unsafe impl Sync for OutPtr {}
    impl OutPtr {
        /// Safety: no two concurrent callers may overlap `[off, off+len)`.
        #[inline]
        unsafe fn write(&self, off: usize, src: &[f32]) {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.0.add(off), src.len());
        }
    }

    /// Per-worker scratch, reused across every task a Rayon worker
    /// runs: a packed Q tile, the `[Q_BLOCK, span]` score tile and the
    /// `[Q_BLOCK, head_dim]` output accumulator. Allocating these per
    /// task cost a malloc/free pair and a memset per `(query-block,
    /// head)`, of which a `pp512` layer has `n_q/8 * n_heads`.
    #[derive(Default)]
    struct Scratch {
        q_tile: Vec<f32>,
        scores: Vec<f32>,
        acc: Vec<f32>,
    }

    const Q_BLOCK: usize = 8;
    let n_blocks = n_q.div_ceil(Q_BLOCK);
    let out_w = OutPtr(out.as_mut_ptr());
    let softcap = attn_softcap.filter(|&c| c > 0.0);

    crate::par::indices_init(n_blocks * n_heads, 1, Scratch::default, |scratch, task| {
        let Scratch {
            q_tile,
            scores,
            acc,
        } = scratch;
        let blk = task / n_heads;
        let h = task % n_heads;
        let kv_h = h / group_size.max(1);
        let b_start = blk * Q_BLOCK;
        let b_end = (b_start + Q_BLOCK).min(n_q);
        let n_b = b_end - b_start;

        // The block's visible KV span. `t_hi` is the widest causal
        // length in the block; `t_lo` is the earliest position its
        // first query can still see under the window.
        let t_hi = kv_prefix + b_end;
        let t_lo = match window {
            Some(w) => (kv_prefix + b_start + 1).saturating_sub(w),
            None => 0,
        };
        let span = t_hi - t_lo;
        let kv_off = t_lo * kv_stride + kv_h * head_dim;
        let v_off = t_lo * v_stride + kv_h * v_head_dim;

        // Pack the block's Q rows for this head contiguously. The
        // GEMM then reads them with `lda = head_dim` instead of
        // `n_heads * head_dim`, which for a 32-head model is the
        // difference between one tile living in L1 and touching 32
        // cache lines per step.
        q_tile.clear();
        for b in b_start..b_end {
            q_tile.extend_from_slice(&q[b * q_stride + h * head_dim..][..head_dim]);
        }

        // Pass 1: `scores[n_b, span] = scale * Q_tile * Kᵀ` as a
        // register-tiled GEMM, computed over the **full** rectangle
        // with no mask. Pass 2 zeroes every entry outside a query's
        // visible range before pass 3 reads it, so the masked-out
        // corners are dead values, never wrong ones: at most
        // `Q_BLOCK-1` extra columns per row (0.7% of a 512-wide
        // span) bought in exchange for a dense inner loop.
        scores.resize(n_b * span, 0.0);
        qk_tile(
            q_tile, n_b, head_dim, k_cache, kv_off, kv_stride, span, scale, scores,
        );

        // Pass 2: softcap, then max-subtract softmax, over exactly
        // each query's visible range -- and an explicit zero
        // everywhere else, which is what turns pass 3 into a dense
        // GEMM (llama.cpp reaches the same state by adding a `-INF`
        // mask row before `ggml_soft_max_ext`).
        let mut norms = [0f32; Q_BLOCK];
        for b in b_start..b_end {
            let causal_len = kv_prefix + b + 1;
            // Same visible range as `causal_gqa_attention_windowed_softcap`
            // called with `seq_len = causal_len`.
            let t_start = match window {
                Some(w) => causal_len.saturating_sub(w),
                None => 0,
            };
            let row = &mut scores[(b - b_start) * span..][..span];
            let lo = t_start - t_lo;
            let hi = causal_len - t_lo;
            row[..lo].fill(0.0);
            row[hi..].fill(0.0);
            let live = &mut row[lo..hi];
            if let Some(sc) = softcap {
                for s in live.iter_mut() {
                    *s = sc * (*s / sc).tanh();
                }
            }
            // ALiBi (`crate::alibi`): `slope_h * (p_key - p_query)` on
            // every visible key, after the softcap, as
            // `ggml_soft_max_ext` adds `slope * mask`. Key `j` of the
            // live range is position `t_start + j`; the query is
            // `causal_len - 1`.
            if let Some(slopes) = alibi {
                let slope = slopes[h];
                let q_pos = (causal_len - 1) as f32;
                for (j, s) in live.iter_mut().enumerate() {
                    *s += slope * ((t_start + j) as f32 - q_pos);
                }
            }
            norms[b - b_start] = softmax_row_exp_sum(live);
        }

        // Pass 3: `acc[n_b, head_dim] += P * V`, the second GEMM.
        // Zero probabilities contribute `fma(v, 0, acc) == acc`
        // exactly, so dropping the mask here is bit-identical to
        // skipping those positions.
        acc.resize(n_b * v_head_dim, 0.0);
        acc.fill(0.0);
        pv_tile(scores, n_b, span, v_cache, v_off, v_stride, v_head_dim, acc);

        for b in b_start..b_end {
            let out_h = &mut acc[(b - b_start) * v_head_dim..][..v_head_dim];
            let l = norms[b - b_start];
            if l > 0.0 {
                scale_inplace(out_h, 1.0 / l);
            }
            unsafe {
                out_w.write(b * out_stride + h * v_head_dim, out_h);
            }
        }
    });

    out
}

/// In-place row softmax for the blocked prefill kernel: `x[i]` becomes
/// `exp(x[i] - max(x))` and the sum of those exponentials is returned,
/// so the caller divides once at the end instead of normalising per
/// position.
///
/// This is the third pass of the blocked form, and on small models it
/// was the expensive one. `pass 1` and `pass 3` are register-tiled
/// GEMMs; this pass was a **scalar `f32::exp` per (query, KV position)**,
/// i.e. one libm `expf` call for every score the GEMM had just produced
/// four-at-a-time. At `pp512` a single layer of a 32-head model issues
/// `512 × 32 × ~256 ≈ 4.2 M` of them.
///
/// llama.cpp does not pay that: `ggml_vec_soft_max_f32`
/// (`ggml/src/ggml-cpu/vec.cpp`) exponentiates a whole row through
/// `ggml_v_expf` (`ggml/src/ggml-cpu/vec.h`), which is ARM's
/// optimized-routines `expf` rewritten over a vector register. This is
/// that same routine; see [`expf_neon`] for the derivation.
///
/// **This changes CPU prefill numerics**, and deliberately: the
/// polynomial is not libm's `expf` to the last bit, and the vector
/// accumulator reassociates the sum. On a near-tie that is enough to
/// move a greedy argmax, so a CPU generation is not token-identical to
/// what the scalar form produced. It is not *less* accurate -- the
/// probabilities land at the same handful of ulps and the normaliser
/// lands closer to the truth, which
/// `the_vectorised_softmax_is_no_less_accurate_than_the_scalar_one`
/// measures against an `f64` reference.
///
/// The reduction order of the max is irrelevant (max is associative and
/// the scores are finite). Both the kernel and the row-at-a-time
/// reference it is pinned against call this one function, which is what
/// keeps `position_outer_prefill_is_bit_identical_to_the_query_outer_form`
/// an equality test rather than a tolerance;
/// `vectorised_softmax_row_matches_the_scalar_libm_form` is what checks
/// this function against `f32::exp` itself.
#[inline]
fn softmax_row_exp_sum(x: &mut [f32]) -> f32 {
    if x.is_empty() {
        // A zero-width visible range (`window == 0`) leaves the caller's
        // accumulator at zero and skips the normalisation, which is what
        // the scalar form did too: `l` never left `0.0`.
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { softmax_row_exp_sum_neon(x) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { softmax_row_exp_sum_avx2(x) };
        }
    }
    softmax_row_exp_sum_scalar(x)
}

/// Scalar `softmax_row_exp_sum` for hosts with neither NEON nor AVX2 --
/// and the shape the SIMD arms are tested against.
fn softmax_row_exp_sum_scalar(x: &mut [f32]) -> f32 {
    let m = x.iter().fold(f32::NEG_INFINITY, |a, &s| a.max(s));
    let mut l = 0f32;
    for s in x.iter_mut() {
        *s = (*s - m).exp();
        l += *s;
    }
    l
}

/// `exp(x)` for four lanes at once: ARM optimized-routines' `expf` in
/// the shape llama.cpp vendors as `ggml_v_expf`
/// (`ggml/src/ggml-cpu/vec.h`, the `__ARM_NEON` arm).
///
/// `z = fma(x, log2(e), 0x1.8p23)` rounds `x·log2(e)` to an integer `n`
/// by the round-to-nearest of the add itself, and leaves that integer in
/// the low mantissa bits of `z`, so `bits(z) << 23` is exactly the
/// exponent field of `2^n` -- one shift instead of a conversion and a
/// scalb. `b = x - n·ln2_hi - n·ln2_lo` is the reduced argument in
/// `[-ln2/2, ln2/2]` (split so the product is exact in `f32`), and the
/// degree-5 minimax polynomial evaluates `e^b - 1` there. The result is
/// `2^n · (1 + j)`, accurate to under an ulp.
///
/// **The overflow branch of the original is dropped, and the clamp is
/// what makes that sound.** llama.cpp keeps a slow path for `|n| > 126`
/// because `ggml_v_expf` is a general `expf`. Here every argument is
/// `score - row_max`, hence `<= 0`, and `exp` of anything below about
/// `-87.3` is already smaller than the smallest normal `f32` -- so
/// clamping the input at `-87` changes no representable output (the row
/// max itself contributes `exp(0) == 1.0` exactly, so a clamped term is
/// at most `1.6e-38` of the sum) while pinning `n` to `[-125.5, 0]`,
/// where the fast path is the only path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn expf_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vmaxq_f32(x, vdupq_n_f32(EXP_MIN_ARG));
    let r = vdupq_n_f32(EXP_SHIFT);
    let z = vfmaq_f32(r, x, vdupq_n_f32(EXP_LOG2E));
    let n = vsubq_f32(z, r);
    // `b = x - n*ln2_hi - n*ln2_lo`; `vfmsq_f32(a, b, c) == a - b*c`.
    let b = vfmsq_f32(
        vfmsq_f32(x, n, vdupq_n_f32(EXP_LN2_HI)),
        n,
        vdupq_n_f32(EXP_LN2_LO),
    );
    // `2^n`, built by dropping `n` into the exponent field. The add
    // wraps for negative `n`, which is exactly the intended borrow.
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

/// AVX2 sibling of [`expf_neon`]: same constants, same polynomial, same
/// clamp, eight lanes (`ggml_v_expf`'s `__AVX2__ && __FMA__` arm).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn expf_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_max_ps(x, _mm256_set1_ps(EXP_MIN_ARG));
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

// The `ggml_v_expf` constants, shared by both vector arms and used by
// neither scalar path -- gated so a host with no SIMD arm at all still
// compiles clean under `-D warnings`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod exp_consts {
    // Shared with `matmul`, which had a byte-identical copy of these
    // nine. Only EXP_MIN_ARG below is ours: a softmax argument is
    // always <= 0, so this clamps below only, where `matmul` must also
    // select zero above because its argument is an unbounded
    // denominator.
    pub use crate::vexp::*;

    /// Below this the exponential is smaller than `f32::MIN_POSITIVE`, so
    /// clamping here costs nothing and keeps `n` inside the fast path.
    pub const EXP_MIN_ARG: f32 = -87.0;
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use exp_consts::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_row_exp_sum_neon(x: &mut [f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len();
    let p = x.as_mut_ptr();
    let nv = n & !3;

    let mut mv = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0;
    while i < nv {
        mv = vmaxq_f32(mv, vld1q_f32(p.add(i)));
        i += 4;
    }
    let mut m = if nv == 0 {
        f32::NEG_INFINITY
    } else {
        vmaxvq_f32(mv)
    };
    for j in nv..n {
        m = m.max(*p.add(j));
    }

    let mvec = vdupq_n_f32(m);
    let mut sv = vdupq_n_f32(0.0);
    let mut i = 0;
    while i < nv {
        let e = expf_neon(vsubq_f32(vld1q_f32(p.add(i)), mvec));
        vst1q_f32(p.add(i), e);
        sv = vaddq_f32(sv, e);
        i += 4;
    }
    let mut l = vaddvq_f32(sv);
    if nv < n {
        // The tail goes through the same approximation rather than
        // `f32::exp`, so a row's values do not change character at the
        // width boundary. Padding lanes hold `0.0`; they are exponentiated
        // and then simply not read.
        let mut buf = [0f32; 4];
        for (j, slot) in (nv..n).zip(buf.iter_mut()) {
            *slot = *p.add(j) - m;
        }
        vst1q_f32(buf.as_mut_ptr(), expf_neon(vld1q_f32(buf.as_ptr())));
        for (j, &e) in (nv..n).zip(buf.iter()) {
            *p.add(j) = e;
            l += e;
        }
    }
    l
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row_exp_sum_avx2(x: &mut [f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    let p = x.as_mut_ptr();
    let nv = n & !7;

    let mut mv = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i < nv {
        mv = _mm256_max_ps(mv, _mm256_loadu_ps(p.add(i)));
        i += 8;
    }
    let mut m = if nv == 0 {
        f32::NEG_INFINITY
    } else {
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), mv);
        lanes.iter().fold(f32::NEG_INFINITY, |a, &s| a.max(s))
    };
    for j in nv..n {
        m = m.max(*p.add(j));
    }

    let mvec = _mm256_set1_ps(m);
    let mut sv = _mm256_setzero_ps();
    let mut i = 0;
    while i < nv {
        let e = expf_avx2(_mm256_sub_ps(_mm256_loadu_ps(p.add(i)), mvec));
        _mm256_storeu_ps(p.add(i), e);
        sv = _mm256_add_ps(sv, e);
        i += 8;
    }
    let mut l = hsum256_ps(sv);
    if nv < n {
        let mut buf = [0f32; 8];
        for (j, slot) in (nv..n).zip(buf.iter_mut()) {
            *slot = *p.add(j) - m;
        }
        _mm256_storeu_ps(buf.as_mut_ptr(), expf_avx2(_mm256_loadu_ps(buf.as_ptr())));
        for (j, &e) in (nv..n).zip(buf.iter()) {
            *p.add(j) = e;
            l += e;
        }
    }
    l
}

/// `scores[b][t] = scale · Σ_d q_tile[b][d]·k[t][d]` for one query block
/// against one head's K rows: the `KQ` matmul that llama.cpp expresses as
/// a plain `ggml_mul_mat` and dispatches into tinyBLAS.
///
/// `q_tile` is packed contiguous `[n_b, head_dim]`; K row `t` lives at
/// `k[k_off + t*k_stride ..][..head_dim]`, so the caller's
/// `[pos, kv_head, dim]` cache needs no repack.
///
/// The port is of tinyBLAS's `gemm_bloc_<RM>x<RN>`
/// (`ggml/src/ggml-cpu/llamafile/sgemm.cpp`): hold an `RM × RN` register
/// tile of vector accumulators, load `RM` A-vectors and `RN` B-vectors per
/// step along `k`, and horizontally sum once at the end. What it replaces
/// was a `dot_f32` per `(query, KV position)`, i.e. a whole K row re-read
/// per query -- two loads for every FMA. The 4×4 NEON tile issues eight
/// loads for sixteen FMAs, and each K row is read once per query block
/// rather than once per query.
///
/// The reduction order is deliberately the same as [`dot_f32`]'s on each
/// backend (4-wide + `vaddvq` under NEON, 8-wide + the same horizontal sum
/// under AVX2, scalar tail after the horizontal sum), so this is
/// bit-identical to the row-at-a-time loop rather than merely close.
// Kept out of line: one call per `(query-block, head)` costs nothing
// against a 512x64x64 tile of FMAs, and it keeps this kernel a named
// symbol in a `sample` profile instead of vanishing into the Rayon
// closure -- which is how its cost was found in the first place.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn qk_tile(
    q_tile: &[f32],
    n_b: usize,
    head_dim: usize,
    k: &[f32],
    k_off: usize,
    k_stride: usize,
    span: usize,
    scale: f32,
    scores: &mut [f32],
) {
    debug_assert_eq!(q_tile.len(), n_b * head_dim);
    debug_assert_eq!(scores.len(), n_b * span);
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                qk_tile_neon(
                    q_tile, n_b, head_dim, k, k_off, k_stride, span, scale, scores,
                )
            };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                qk_tile_avx2(
                    q_tile, n_b, head_dim, k, k_off, k_stride, span, scale, scores,
                )
            };
            return;
        }
    }
    qk_rows(
        q_tile,
        head_dim,
        k,
        k_off,
        k_stride,
        span,
        scale,
        scores,
        0..n_b,
        0..span,
    );
}

/// Row-at-a-time `Q·Kᵀ` over a sub-rectangle of the score tile: the
/// edges the register tile does not cover, and the whole tile on hosts
/// with neither NEON nor AVX2.
#[allow(clippy::too_many_arguments)]
fn qk_rows(
    q_tile: &[f32],
    head_dim: usize,
    k: &[f32],
    k_off: usize,
    k_stride: usize,
    span: usize,
    scale: f32,
    scores: &mut [f32],
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) {
    for b in rows {
        let q_b = &q_tile[b * head_dim..][..head_dim];
        for t in cols.clone() {
            let k_t = &k[k_off + t * k_stride..][..head_dim];
            scores[b * span + t] = dot_f32(q_b, k_t) * scale;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn qk_tile_neon(
    q_tile: &[f32],
    n_b: usize,
    head_dim: usize,
    k: &[f32],
    k_off: usize,
    k_stride: usize,
    span: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use std::arch::aarch64::*;
    let qp = q_tile.as_ptr();
    let kp = k.as_ptr().add(k_off);
    let sp = scores.as_mut_ptr();
    // Rows/columns the 4×4 tile covers, and the 4-wide part of `head_dim`
    // -- the same boundary `dot_f32_neon` uses, which is what keeps the
    // scalar leftovers bit-identical.
    let bt = n_b & !3;
    let tt = span & !3;
    let dv = head_dim & !3;

    // KV position outer, query block inner: the four K rows of a tile are
    // loaded once and reused by every query tile, so one pass over this
    // head's K slab serves the whole query block.
    let mut t0 = 0;
    while t0 < tt {
        let k0 = kp.add(t0 * k_stride);
        let k1 = k0.add(k_stride);
        let k2 = k1.add(k_stride);
        let k3 = k2.add(k_stride);
        let mut b0 = 0;
        while b0 < bt {
            let a0 = qp.add(b0 * head_dim);
            let a1 = a0.add(head_dim);
            let a2 = a1.add(head_dim);
            let a3 = a2.add(head_dim);
            let z = vdupq_n_f32(0.0);
            // `cIJ` accumulates query `b0+I` against key `t0+J`.
            let (mut c00, mut c01, mut c02, mut c03) = (z, z, z, z);
            let (mut c10, mut c11, mut c12, mut c13) = (z, z, z, z);
            let (mut c20, mut c21, mut c22, mut c23) = (z, z, z, z);
            let (mut c30, mut c31, mut c32, mut c33) = (z, z, z, z);
            let mut d = 0;
            while d < dv {
                let av0 = vld1q_f32(a0.add(d));
                let av1 = vld1q_f32(a1.add(d));
                let av2 = vld1q_f32(a2.add(d));
                let av3 = vld1q_f32(a3.add(d));
                let kv0 = vld1q_f32(k0.add(d));
                c00 = vfmaq_f32(c00, av0, kv0);
                c10 = vfmaq_f32(c10, av1, kv0);
                c20 = vfmaq_f32(c20, av2, kv0);
                c30 = vfmaq_f32(c30, av3, kv0);
                let kv1 = vld1q_f32(k1.add(d));
                c01 = vfmaq_f32(c01, av0, kv1);
                c11 = vfmaq_f32(c11, av1, kv1);
                c21 = vfmaq_f32(c21, av2, kv1);
                c31 = vfmaq_f32(c31, av3, kv1);
                let kv2 = vld1q_f32(k2.add(d));
                c02 = vfmaq_f32(c02, av0, kv2);
                c12 = vfmaq_f32(c12, av1, kv2);
                c22 = vfmaq_f32(c22, av2, kv2);
                c32 = vfmaq_f32(c32, av3, kv2);
                let kv3 = vld1q_f32(k3.add(d));
                c03 = vfmaq_f32(c03, av0, kv3);
                c13 = vfmaq_f32(c13, av1, kv3);
                c23 = vfmaq_f32(c23, av2, kv3);
                c33 = vfmaq_f32(c33, av3, kv3);
                d += 4;
            }
            let mut r = [
                [
                    vaddvq_f32(c00),
                    vaddvq_f32(c01),
                    vaddvq_f32(c02),
                    vaddvq_f32(c03),
                ],
                [
                    vaddvq_f32(c10),
                    vaddvq_f32(c11),
                    vaddvq_f32(c12),
                    vaddvq_f32(c13),
                ],
                [
                    vaddvq_f32(c20),
                    vaddvq_f32(c21),
                    vaddvq_f32(c22),
                    vaddvq_f32(c23),
                ],
                [
                    vaddvq_f32(c30),
                    vaddvq_f32(c31),
                    vaddvq_f32(c32),
                    vaddvq_f32(c33),
                ],
            ];
            // Leftover dims after the horizontal sum, exactly where
            // `dot_f32_neon` adds them.
            let arow = [a0, a1, a2, a3];
            let krow = [k0, k1, k2, k3];
            for d in dv..head_dim {
                for (i, ai) in arow.iter().enumerate() {
                    let av = *ai.add(d);
                    for (j, kj) in krow.iter().enumerate() {
                        r[i][j] += av * *kj.add(d);
                    }
                }
            }
            for (i, ri) in r.iter().enumerate() {
                for (j, v) in ri.iter().enumerate() {
                    *sp.add((b0 + i) * span + t0 + j) = v * scale;
                }
            }
            b0 += 4;
        }
        t0 += 4;
    }
    qk_rows(
        q_tile,
        head_dim,
        k,
        k_off,
        k_stride,
        span,
        scale,
        scores,
        0..bt,
        tt..span,
    );
    qk_rows(
        q_tile,
        head_dim,
        k,
        k_off,
        k_stride,
        span,
        scale,
        scores,
        bt..n_b,
        0..span,
    );
}

/// AVX2 sibling of [`qk_tile_neon`]. The register file is half as wide
/// (16 YMM), so the tile is 4 queries × 2 keys -- 8 accumulators plus 4
/// A-vectors and a B-vector -- instead of 4×4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn qk_tile_avx2(
    q_tile: &[f32],
    n_b: usize,
    head_dim: usize,
    k: &[f32],
    k_off: usize,
    k_stride: usize,
    span: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use std::arch::x86_64::*;
    let qp = q_tile.as_ptr();
    let kp = k.as_ptr().add(k_off);
    let sp = scores.as_mut_ptr();
    let bt = n_b & !3;
    let tt = span & !1;
    let dv = head_dim & !7;

    let mut t0 = 0;
    while t0 < tt {
        let k0 = kp.add(t0 * k_stride);
        let k1 = k0.add(k_stride);
        let mut b0 = 0;
        while b0 < bt {
            let a0 = qp.add(b0 * head_dim);
            let a1 = a0.add(head_dim);
            let a2 = a1.add(head_dim);
            let a3 = a2.add(head_dim);
            let z = _mm256_setzero_ps();
            let (mut c00, mut c01) = (z, z);
            let (mut c10, mut c11) = (z, z);
            let (mut c20, mut c21) = (z, z);
            let (mut c30, mut c31) = (z, z);
            let mut d = 0;
            while d < dv {
                let av0 = _mm256_loadu_ps(a0.add(d));
                let av1 = _mm256_loadu_ps(a1.add(d));
                let av2 = _mm256_loadu_ps(a2.add(d));
                let av3 = _mm256_loadu_ps(a3.add(d));
                let kv0 = _mm256_loadu_ps(k0.add(d));
                c00 = _mm256_fmadd_ps(av0, kv0, c00);
                c10 = _mm256_fmadd_ps(av1, kv0, c10);
                c20 = _mm256_fmadd_ps(av2, kv0, c20);
                c30 = _mm256_fmadd_ps(av3, kv0, c30);
                let kv1 = _mm256_loadu_ps(k1.add(d));
                c01 = _mm256_fmadd_ps(av0, kv1, c01);
                c11 = _mm256_fmadd_ps(av1, kv1, c11);
                c21 = _mm256_fmadd_ps(av2, kv1, c21);
                c31 = _mm256_fmadd_ps(av3, kv1, c31);
                d += 8;
            }
            let mut r = [
                [hsum256_ps(c00), hsum256_ps(c01)],
                [hsum256_ps(c10), hsum256_ps(c11)],
                [hsum256_ps(c20), hsum256_ps(c21)],
                [hsum256_ps(c30), hsum256_ps(c31)],
            ];
            let arow = [a0, a1, a2, a3];
            let krow = [k0, k1];
            for d in dv..head_dim {
                for (i, ai) in arow.iter().enumerate() {
                    let av = *ai.add(d);
                    for (j, kj) in krow.iter().enumerate() {
                        r[i][j] += av * *kj.add(d);
                    }
                }
            }
            for (i, ri) in r.iter().enumerate() {
                for (j, v) in ri.iter().enumerate() {
                    *sp.add((b0 + i) * span + t0 + j) = v * scale;
                }
            }
            b0 += 4;
        }
        t0 += 2;
    }
    qk_rows(
        q_tile,
        head_dim,
        k,
        k_off,
        k_stride,
        span,
        scale,
        scores,
        0..bt,
        tt..span,
    );
    qk_rows(
        q_tile,
        head_dim,
        k,
        k_off,
        k_stride,
        span,
        scale,
        scores,
        bt..n_b,
        0..span,
    );
}

/// `acc[b][d] += Σ_t p[b][t]·v[t][d]` for one query block against one
/// head's V rows: the `KQV` matmul, the second of llama.cpp's two
/// attention `ggml_mul_mat`s.
///
/// V row `t` lives at `v[v_off + t*v_stride ..][..head_dim]`. `p` is the
/// `[n_b, span]` probability tile pass 2 produced, already zeroed outside
/// each query's visible range, so no mask is needed here.
///
/// Register-tiled the other way round from [`qk_tile`]: the output tile
/// (8 queries × 8 dims) lives in the accumulators and `t` is the
/// reduction axis, so a V row is loaded once and feeds all eight
/// queries. What it replaces was an `axpy` per `(KV position, query)`,
/// which re-loaded and re-stored the whole `head_dim`-wide accumulator
/// row for every position -- three L1 accesses per FMA against this
/// version's ten loads per sixteen vector FMAs.
///
/// Accumulation order along `t` is unchanged (ascending, one `fma` per
/// position), and the vector/scalar boundary matches [`axpy`]'s on each
/// backend, so this is bit-identical to the row-at-a-time loop.
// Out of line for the same reason as [`qk_tile`].
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn pv_tile(
    p: &[f32],
    n_b: usize,
    span: usize,
    v: &[f32],
    v_off: usize,
    v_stride: usize,
    head_dim: usize,
    acc: &mut [f32],
) {
    debug_assert_eq!(p.len(), n_b * span);
    debug_assert_eq!(acc.len(), n_b * head_dim);
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { pv_tile_neon(p, n_b, span, v, v_off, v_stride, head_dim, acc) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { pv_tile_avx2(p, n_b, span, v, v_off, v_stride, head_dim, acc) };
            return;
        }
    }
    pv_rows(p, span, v, v_off, v_stride, head_dim, acc, 0..n_b);
}

/// Row-at-a-time `P·V` for the query rows the register tile does not
/// cover, and for hosts with neither NEON nor AVX2. Zero probabilities
/// are skipped rather than accumulated -- `acc + v*0` is exactly `acc`
/// for finite `v`, so this is a pure work saving on the masked padding.
#[allow(clippy::too_many_arguments)]
fn pv_rows(
    p: &[f32],
    span: usize,
    v: &[f32],
    v_off: usize,
    v_stride: usize,
    head_dim: usize,
    acc: &mut [f32],
    rows: std::ops::Range<usize>,
) {
    for b in rows {
        let out_b = &mut acc[b * head_dim..][..head_dim];
        for t in 0..span {
            let w = p[b * span + t];
            if w == 0.0 {
                continue;
            }
            axpy(out_b, &v[v_off + t * v_stride..][..head_dim], w);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn pv_tile_neon(
    p: &[f32],
    n_b: usize,
    span: usize,
    v: &[f32],
    v_off: usize,
    v_stride: usize,
    head_dim: usize,
    acc: &mut [f32],
) {
    use std::arch::aarch64::*;
    let vp = v.as_ptr().add(v_off);
    let pp = p.as_ptr();
    let ap = acc.as_mut_ptr();
    let bt = n_b & !7;
    let dv = head_dim & !7;
    // `axpy_neon` vectorizes up to `head_dim & !3` and goes scalar after;
    // matching both boundaries is what makes the leftovers bit-identical.
    let dv4 = head_dim & !3;

    let mut b0 = 0;
    while b0 < bt {
        let mut d0 = 0;
        while d0 < dv {
            let mut c0l = vld1q_f32(ap.add(b0 * head_dim + d0));
            let mut c0h = vld1q_f32(ap.add(b0 * head_dim + d0 + 4));
            let mut c1l = vld1q_f32(ap.add((b0 + 1) * head_dim + d0));
            let mut c1h = vld1q_f32(ap.add((b0 + 1) * head_dim + d0 + 4));
            let mut c2l = vld1q_f32(ap.add((b0 + 2) * head_dim + d0));
            let mut c2h = vld1q_f32(ap.add((b0 + 2) * head_dim + d0 + 4));
            let mut c3l = vld1q_f32(ap.add((b0 + 3) * head_dim + d0));
            let mut c3h = vld1q_f32(ap.add((b0 + 3) * head_dim + d0 + 4));
            let mut c4l = vld1q_f32(ap.add((b0 + 4) * head_dim + d0));
            let mut c4h = vld1q_f32(ap.add((b0 + 4) * head_dim + d0 + 4));
            let mut c5l = vld1q_f32(ap.add((b0 + 5) * head_dim + d0));
            let mut c5h = vld1q_f32(ap.add((b0 + 5) * head_dim + d0 + 4));
            let mut c6l = vld1q_f32(ap.add((b0 + 6) * head_dim + d0));
            let mut c6h = vld1q_f32(ap.add((b0 + 6) * head_dim + d0 + 4));
            let mut c7l = vld1q_f32(ap.add((b0 + 7) * head_dim + d0));
            let mut c7h = vld1q_f32(ap.add((b0 + 7) * head_dim + d0 + 4));
            for t in 0..span {
                let vr = vp.add(t * v_stride + d0);
                let v0 = vld1q_f32(vr);
                let v1 = vld1q_f32(vr.add(4));
                let s0 = vdupq_n_f32(*pp.add(b0 * span + t));
                c0l = vfmaq_f32(c0l, v0, s0);
                c0h = vfmaq_f32(c0h, v1, s0);
                let s1 = vdupq_n_f32(*pp.add((b0 + 1) * span + t));
                c1l = vfmaq_f32(c1l, v0, s1);
                c1h = vfmaq_f32(c1h, v1, s1);
                let s2 = vdupq_n_f32(*pp.add((b0 + 2) * span + t));
                c2l = vfmaq_f32(c2l, v0, s2);
                c2h = vfmaq_f32(c2h, v1, s2);
                let s3 = vdupq_n_f32(*pp.add((b0 + 3) * span + t));
                c3l = vfmaq_f32(c3l, v0, s3);
                c3h = vfmaq_f32(c3h, v1, s3);
                let s4 = vdupq_n_f32(*pp.add((b0 + 4) * span + t));
                c4l = vfmaq_f32(c4l, v0, s4);
                c4h = vfmaq_f32(c4h, v1, s4);
                let s5 = vdupq_n_f32(*pp.add((b0 + 5) * span + t));
                c5l = vfmaq_f32(c5l, v0, s5);
                c5h = vfmaq_f32(c5h, v1, s5);
                let s6 = vdupq_n_f32(*pp.add((b0 + 6) * span + t));
                c6l = vfmaq_f32(c6l, v0, s6);
                c6h = vfmaq_f32(c6h, v1, s6);
                let s7 = vdupq_n_f32(*pp.add((b0 + 7) * span + t));
                c7l = vfmaq_f32(c7l, v0, s7);
                c7h = vfmaq_f32(c7h, v1, s7);
            }
            vst1q_f32(ap.add(b0 * head_dim + d0), c0l);
            vst1q_f32(ap.add(b0 * head_dim + d0 + 4), c0h);
            vst1q_f32(ap.add((b0 + 1) * head_dim + d0), c1l);
            vst1q_f32(ap.add((b0 + 1) * head_dim + d0 + 4), c1h);
            vst1q_f32(ap.add((b0 + 2) * head_dim + d0), c2l);
            vst1q_f32(ap.add((b0 + 2) * head_dim + d0 + 4), c2h);
            vst1q_f32(ap.add((b0 + 3) * head_dim + d0), c3l);
            vst1q_f32(ap.add((b0 + 3) * head_dim + d0 + 4), c3h);
            vst1q_f32(ap.add((b0 + 4) * head_dim + d0), c4l);
            vst1q_f32(ap.add((b0 + 4) * head_dim + d0 + 4), c4h);
            vst1q_f32(ap.add((b0 + 5) * head_dim + d0), c5l);
            vst1q_f32(ap.add((b0 + 5) * head_dim + d0 + 4), c5h);
            vst1q_f32(ap.add((b0 + 6) * head_dim + d0), c6l);
            vst1q_f32(ap.add((b0 + 6) * head_dim + d0 + 4), c6h);
            vst1q_f32(ap.add((b0 + 7) * head_dim + d0), c7l);
            vst1q_f32(ap.add((b0 + 7) * head_dim + d0 + 4), c7h);
            d0 += 8;
        }
        // Leftover dims, still `t`-ascending per `(query, dim)`: fused
        // below `head_dim & !3` and plain below `head_dim`, which is
        // where `axpy_neon`'s own vector/scalar split falls.
        if dv < head_dim {
            for t in 0..span {
                for i in 0..8 {
                    let w = *pp.add((b0 + i) * span + t);
                    if w == 0.0 {
                        continue;
                    }
                    let row = ap.add((b0 + i) * head_dim);
                    for d in dv..dv4 {
                        *row.add(d) = f32::mul_add(w, *vp.add(t * v_stride + d), *row.add(d));
                    }
                    for d in dv4..head_dim {
                        *row.add(d) += w * *vp.add(t * v_stride + d);
                    }
                }
            }
        }
        b0 += 8;
    }
    pv_rows(p, span, v, v_off, v_stride, head_dim, acc, bt..n_b);
}

/// AVX2 sibling of [`pv_tile_neon`]: same 8-query × 8-dim output tile,
/// but one YMM accumulator per query instead of two NEON quads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pv_tile_avx2(
    p: &[f32],
    n_b: usize,
    span: usize,
    v: &[f32],
    v_off: usize,
    v_stride: usize,
    head_dim: usize,
    acc: &mut [f32],
) {
    use std::arch::x86_64::*;
    let vp = v.as_ptr().add(v_off);
    let pp = p.as_ptr();
    let ap = acc.as_mut_ptr();
    let bt = n_b & !7;
    // `axpy_avx2` vectorizes up to `head_dim & !7`, so its scalar tail
    // and this kernel's are the same elements.
    let dv = head_dim & !7;

    let mut b0 = 0;
    while b0 < bt {
        let mut d0 = 0;
        while d0 < dv {
            let mut c0 = _mm256_loadu_ps(ap.add(b0 * head_dim + d0));
            let mut c1 = _mm256_loadu_ps(ap.add((b0 + 1) * head_dim + d0));
            let mut c2 = _mm256_loadu_ps(ap.add((b0 + 2) * head_dim + d0));
            let mut c3 = _mm256_loadu_ps(ap.add((b0 + 3) * head_dim + d0));
            let mut c4 = _mm256_loadu_ps(ap.add((b0 + 4) * head_dim + d0));
            let mut c5 = _mm256_loadu_ps(ap.add((b0 + 5) * head_dim + d0));
            let mut c6 = _mm256_loadu_ps(ap.add((b0 + 6) * head_dim + d0));
            let mut c7 = _mm256_loadu_ps(ap.add((b0 + 7) * head_dim + d0));
            for t in 0..span {
                let vv = _mm256_loadu_ps(vp.add(t * v_stride + d0));
                c0 = _mm256_fmadd_ps(vv, _mm256_set1_ps(*pp.add(b0 * span + t)), c0);
                c1 = _mm256_fmadd_ps(vv, _mm256_set1_ps(*pp.add((b0 + 1) * span + t)), c1);
                c2 = _mm256_fmadd_ps(vv, _mm256_set1_ps(*pp.add((b0 + 2) * span + t)), c2);
                c3 = _mm256_fmadd_ps(vv, _mm256_set1_ps(*pp.add((b0 + 3) * span + t)), c3);
                c4 = _mm256_fmadd_ps(vv, _mm256_set1_ps(*pp.add((b0 + 4) * span + t)), c4);
                c5 = _mm256_fmadd_ps(vv, _mm256_set1_ps(*pp.add((b0 + 5) * span + t)), c5);
                c6 = _mm256_fmadd_ps(vv, _mm256_set1_ps(*pp.add((b0 + 6) * span + t)), c6);
                c7 = _mm256_fmadd_ps(vv, _mm256_set1_ps(*pp.add((b0 + 7) * span + t)), c7);
            }
            _mm256_storeu_ps(ap.add(b0 * head_dim + d0), c0);
            _mm256_storeu_ps(ap.add((b0 + 1) * head_dim + d0), c1);
            _mm256_storeu_ps(ap.add((b0 + 2) * head_dim + d0), c2);
            _mm256_storeu_ps(ap.add((b0 + 3) * head_dim + d0), c3);
            _mm256_storeu_ps(ap.add((b0 + 4) * head_dim + d0), c4);
            _mm256_storeu_ps(ap.add((b0 + 5) * head_dim + d0), c5);
            _mm256_storeu_ps(ap.add((b0 + 6) * head_dim + d0), c6);
            _mm256_storeu_ps(ap.add((b0 + 7) * head_dim + d0), c7);
            d0 += 8;
        }
        if dv < head_dim {
            for t in 0..span {
                for i in 0..8 {
                    let w = *pp.add((b0 + i) * span + t);
                    if w == 0.0 {
                        continue;
                    }
                    let row = ap.add((b0 + i) * head_dim);
                    for d in dv..head_dim {
                        *row.add(d) += w * *vp.add(t * v_stride + d);
                    }
                }
            }
        }
        b0 += 8;
    }
    pv_rows(p, span, v, v_off, v_stride, head_dim, acc, bt..n_b);
}

/// Same math as `causal_gqa_attention`, but K/V positions are read
/// through a `PagedKvStore` block table instead of one contiguous
/// slice: position `t` lives in block `block_table[t / block_size]`
/// at offset `t % block_size`, so blocks need not be physically
/// adjacent or in order. Must match `causal_gqa_attention` given the
/// same logical K/V contents (float noise only) — the block table is a
/// storage-layout detail, not a math change.
pub fn causal_gqa_attention_paged(
    q: &[f32],
    store: &PagedKvStore,
    block_table: &[usize],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    causal_gqa_attention_paged_sinks(
        q,
        store,
        block_table,
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        None,
        None,
        None,
        None,
    )
}

/// [`causal_gqa_attention_paged`] with per-head attention sinks and an
/// optional sliding window: the paged twin of
/// [`causal_gqa_attention_sinks`].
///
/// # Why this had to exist before the paged path could serve anything
///
/// `causal_gqa_attention_paged` had neither term, and
/// `Decoder::forward_token_paged` therefore refused gpt-oss with an
/// assert rather than answer it differently from the contiguous path.
/// That assert was the right call and a dead end: a sliding-window or
/// sink-carrying model could never move onto paged KV, and paged KV is
/// what a radix prefix cache hands back page indices for. So this is a
/// correctness item before it is a caching one.
///
/// # Bit-identity is by construction, not by tolerance
///
/// Both this and the contiguous kernel funnel the same `(k, v)` rows,
/// in the same order, through the same [`online_attn_accumulate`] with
/// the same scale and the same sink. Nothing is re-associated and no
/// sum is reordered, so the results are bit-identical rather than
/// close -- which is the only useful bar here, since the whole point is
/// that moving a model onto paged KV must not change its distribution.
/// The tests assert exact equality.
///
/// `sinks` is `None` for a model that ships none, which is the ordinary
/// case; `window` is `Some(w)` for a sliding-window layer and `None`
/// for full causal. `attn_softcap` is carried too, so this one entry
/// point can mirror every arm of the contiguous dispatch: a softcapped
/// model moved onto paged KV without it would differ silently, which is
/// the same class of bug this function exists to close.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_paged_sinks(
    q: &[f32],
    store: &PagedKvStore,
    block_table: &[usize],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    window: Option<usize>,
    sinks: Option<&[f32]>,
    attn_softcap: Option<f32>,
    alibi: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * head_dim);
    if let Some(slopes) = alibi {
        assert_eq!(slopes.len(), n_heads, "ALiBi slopes are per query head");
    }
    assert_eq!(
        head_dim,
        store.head_dim(),
        "the query's head width must be the store's K head width"
    );
    // The V head width is the store's own fact, not an argument: a
    // store built for a model is built at that model's widths
    // (`PagedKvStore::new_split`), and a kernel that took it separately
    // could be handed the K width for it -- which is what every arm did
    // before MiMo-V2 made the two differ.
    let v_head_dim = store.v_head_dim();
    let block_size = store.block_size();
    assert!(
        block_table.len() * block_size >= seq_len,
        "block table too short for seq_len"
    );
    if let Some(sinks) = sinks {
        assert_eq!(
            sinks.len(),
            n_heads,
            "attention sinks are per query head (llama.cpp `attn_sinks` is {{n_head}})"
        );
    }

    let group_size = n_heads / n_kv_heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; n_heads * v_head_dim];
    // The query is the last cached position; a windowed layer sees only
    // the most recent `window` positions including its own. Identical
    // to the contiguous kernel's `start`, deliberately: a different
    // rounding here would silently shift which token a window drops.
    let start = match window {
        Some(w) => {
            assert!(w > 0, "window must be positive");
            seq_len.saturating_sub(w)
        }
        None => 0,
    };

    let q_pos = seq_len as f32 - 1.0;
    for h in 0..n_heads {
        let kv_h = h / group_size.max(1);
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let sink = sinks.map(|s| s[h]);
        let slope = alibi.map_or(0.0, |s| s[h]);
        let out_h = &mut out[h * v_head_dim..(h + 1) * v_head_dim];
        let (mut k_scratch, mut v_scratch) = (Vec::new(), Vec::new());
        online_attn_accumulate(q_h, scale, out_h, attn_softcap, sink, |visit| {
            for t in start..seq_len {
                let block_id = block_table[t / block_size];
                let offset = t % block_size;
                let k_t = store
                    .k_row(block_id, offset)
                    .read(kv_h * head_dim..(kv_h + 1) * head_dim, &mut k_scratch);
                let v_t = store
                    .v_row(block_id, offset)
                    .read(kv_h * v_head_dim..(kv_h + 1) * v_head_dim, &mut v_scratch);
                visit(k_t, v_t, slope * (t as f32 - q_pos));
            }
        });
    }

    out
}

/// Single-token causal attention for DeepSeek/Kimi-style Multi-head
/// Latent Attention (MLA): every query head has its own key/value (no
/// GQA-style grouping -- verified directly against Kimi K3's real
/// `KimiMLAAttention.forward`, where `kv_b_proj` expands to the full
/// `num_heads` count and the `num_key_value_heads`/`num_key_value_groups`
/// fields computed in `__init__` go unused), but the key/query head
/// dimension (`qk_head_dim` = `qk_nope_head_dim + qk_rope_head_dim`) can
/// differ from the value head dimension (`v_head_dim`) -- unlike
/// `causal_gqa_attention`, which assumes one shared `head_dim` for both.
///
/// `q` is [n_heads, qk_head_dim]; `k_cache` is [seq_len, n_heads,
/// qk_head_dim]; `v_cache` is [seq_len, n_heads, v_head_dim]. Returns
/// [n_heads, v_head_dim].
pub fn causal_mla_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    mla_attention_inner(
        q,
        k_cache,
        v_cache,
        n_heads,
        qk_head_dim,
        v_head_dim,
        seq_len,
        None,
        None,
        1.0 / (qk_head_dim as f32).sqrt(),
    )
}

/// The one MLA attention body, shared by the dense, sparse and
/// sink-carrying entry points.
///
/// `visible` restricts which key positions participate (`None` is every
/// position through `seq_len`); `sinks` is one learned logit per query
/// head. Sharing the body is deliberate rather than tidy: the four
/// public forms differ only in those two options, and a second copy of
/// the softmax is how one of them quietly stops matching the others.
///
/// A sink joins the softmax denominator with a **zero** value vector,
/// so it takes probability mass away from the real keys without
/// contributing to the output -- the same semantics as
/// [`causal_gqa_attention_sinks`], and for the same reason: it lets a
/// head decline to attend to anything rather than being forced to
/// spread a full unit of weight over keys it does not want. The sink
/// logit is **not** scaled by `1/sqrt(qk_head_dim)`; it is a learned
/// logit already in score space.
///
/// A head whose sink dominates gets an output near zero, which is the
/// intended behaviour and not a bug to guard against -- clamping it
/// would remove the only thing the sink is for.
#[allow(clippy::too_many_arguments)]
fn mla_attention_inner(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
    visible: Option<&[usize]>,
    sinks: Option<&[f32]>,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * qk_head_dim);
    assert_eq!(k_cache.len(), seq_len * n_heads * qk_head_dim);
    assert_eq!(v_cache.len(), seq_len * n_heads * v_head_dim);
    if let Some(visible) = visible {
        assert!(
            visible.iter().all(|&t| t < seq_len),
            "visible positions must be within seq_len"
        );
    }
    if let Some(sinks) = sinks {
        assert_eq!(
            sinks.len(),
            n_heads,
            "one sink logit per query head, or none at all"
        );
    }

    // Indexed rather than materialized: `None` means every position
    // through `seq_len`, and building that list would allocate one
    // `usize` per cached token on every decode step of every layer --
    // paid on the dense path, which is the common one.
    let n_positions = visible.map_or(seq_len, |v| v.len());
    let position_at = |i: usize| visible.map_or(i, |v| v[i]);

    let mut out = vec![0f32; n_heads * v_head_dim];

    for h in 0..n_heads {
        let q_h = &q[h * qk_head_dim..(h + 1) * qk_head_dim];

        let mut scores = vec![0f32; n_positions];
        for (i, score) in scores.iter_mut().enumerate() {
            let t = position_at(i);
            let k_t =
                &k_cache[(t * n_heads + h) * qk_head_dim..(t * n_heads + h + 1) * qk_head_dim];
            let mut dot = 0f32;
            for d in 0..qk_head_dim {
                dot += q_h[d] * k_t[d];
            }
            *score = dot * scale;
        }

        let sink = sinks.map(|s| s[h]);
        let mut max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if let Some(s) = sink {
            max = max.max(s);
        }
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        // The sink's mass lands in the denominator only: it has no value
        // vector, which is exactly how it removes weight from the real
        // keys instead of redistributing it among them.
        if let Some(s) = sink {
            sum += (s - max).exp();
        }
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }

        let out_h = &mut out[h * v_head_dim..(h + 1) * v_head_dim];
        for (i, &w) in scores.iter().enumerate() {
            let t = position_at(i);
            let v_t = &v_cache[(t * n_heads + h) * v_head_dim..(t * n_heads + h + 1) * v_head_dim];
            for d in 0..v_head_dim {
                out_h[d] += w * v_t[d];
            }
        }
    }

    out
}

/// [`causal_mla_attention`] with the caller's softmax scale instead of
/// `1/sqrt(qk_head_dim)`: DeepSeek-2 under YaRN folds `mscale^2` into
/// it (`src/models/deepseek2.cpp:446-448`, `frink_models::mla_yarn`).
#[allow(clippy::too_many_arguments)]
pub fn causal_mla_attention_scaled(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    mla_attention_inner(
        q,
        k_cache,
        v_cache,
        n_heads,
        qk_head_dim,
        v_head_dim,
        seq_len,
        None,
        None,
        scale,
    )
}

/// [`causal_mla_attention`] with DeepSeek V4's per-head attention sinks.
///
/// `sinks` is one learned logit per query head. See
/// [`mla_attention_inner`] for what a sink does and why it is not
/// scaled.
#[allow(clippy::too_many_arguments)]
pub fn causal_mla_attention_sinks(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
    sinks: Option<&[f32]>,
) -> Vec<f32> {
    mla_attention_inner(
        q,
        k_cache,
        v_cache,
        n_heads,
        qk_head_dim,
        v_head_dim,
        seq_len,
        None,
        sinks,
        1.0 / (qk_head_dim as f32).sqrt(),
    )
}

/// [`causal_mla_attention_sparse`] with per-head attention sinks.
///
/// The sink matters more here than on the dense path: a sparse query
/// sees only the positions the indexer selected, and without a sink its
/// softmax is forced to spend a full unit of weight on them however
/// poorly they match.
#[allow(clippy::too_many_arguments)]
pub fn causal_mla_attention_sparse_sinks(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
    visible: &[usize],
    sinks: Option<&[f32]>,
) -> Vec<f32> {
    mla_attention_inner(
        q,
        k_cache,
        v_cache,
        n_heads,
        qk_head_dim,
        v_head_dim,
        seq_len,
        Some(visible),
        sinks,
        1.0 / (qk_head_dim as f32).sqrt(),
    )
}

/// The DeepSeek-V3.2 / GLM-5.2 "lightning indexer" (arXiv 2512.02556;
/// real, merged, tested reference implementations in llama.cpp PR
/// #23346 and PR #25407): scores every causally-visible key position
/// against the query using a cheap multi-head dot-product indexer, then
/// keeps only the `top_k` highest-scoring positions.
///
/// `indexer_q` is `[n_index_heads][index_head_dim]` for this query
/// position; `indexer_keys` is `[num_causal_positions][index_head_dim]`
/// (one MQA key per causal position, `0..=query_pos`); `indexer_weights`
/// is `[n_index_heads]`. Returns the kept key positions, ascending.
///
/// The real implementation additionally rotates `indexer_q`/`indexer_k`
/// through a fixed orthogonal Hadamard matrix before the dot product (to
/// spread values evenly for FP8 quantization on real hardware). An
/// orthogonal transform applied identically to both operands leaves
/// their dot product unchanged in exact arithmetic
/// (`(Hq)·(Hk) = q^T H^T H k = q^T k`), so this f32 CPU path omits it —
/// the score computed here is exact, not an approximation of the real
/// one.
pub fn lightning_indexer_topk(
    indexer_q: &[Vec<f32>],
    indexer_keys: &[Vec<f32>],
    indexer_weights: &[f32],
    top_k: usize,
) -> Vec<usize> {
    let n_heads = indexer_q.len();
    assert_eq!(indexer_weights.len(), n_heads);
    let index_head_dim = indexer_q.first().map_or(0, |q| q.len());
    let scale = 1.0 / ((index_head_dim * n_heads) as f32).sqrt();

    let mut scored: Vec<(usize, f32)> = indexer_keys
        .iter()
        .enumerate()
        .map(|(j, k)| {
            let score: f32 = indexer_q
                .iter()
                .zip(indexer_weights.iter())
                .map(|(q, w)| {
                    let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                    dot.max(0.0) * w * scale
                })
                .sum();
            (j, score)
        })
        .collect();

    let keep = top_k.min(scored.len());
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut kept: Vec<usize> = scored.into_iter().take(keep).map(|(j, _)| j).collect();
    kept.sort_unstable();
    kept
}

/// Same as [`causal_mla_attention`] for a single query position, but
/// attention is restricted to the explicit `visible` key positions
/// (ascending, a subset of `0..seq_len`) rather than the full causal
/// history — the sparse-attention half of GLM-5.2/DeepSeek-V3.2's DSA,
/// applied after [`lightning_indexer_topk`] selects `visible`.
#[allow(clippy::too_many_arguments)]
pub fn causal_mla_attention_sparse(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
    visible: &[usize],
) -> Vec<f32> {
    mla_attention_inner(
        q,
        k_cache,
        v_cache,
        n_heads,
        qk_head_dim,
        v_head_dim,
        seq_len,
        Some(visible),
        None,
        1.0 / (qk_head_dim as f32).sqrt(),
    )
}

#[cfg(test)]
mod tests {

    /// Deterministic pseudo-random data for the split-width tests.
    fn lcg_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// A V head width that differs from the K width (MiMo-V2's
    /// `192 / 128`): the row kernel agrees with the MLA kernel, which
    /// has always taken the two widths separately, on the MHA shape the
    /// two share -- and with itself on the GQA shape the MLA kernel
    /// cannot express, through the paged twin over the same rows.
    #[test]
    fn split_kv_head_widths_agree_across_the_row_paged_and_mla_kernels() {
        let (n_heads, n_kv_heads, hd, vd, seq) = (4usize, 2usize, 6usize, 4usize, 7usize);
        let q = lcg_vec(1, n_heads * hd);
        let k = lcg_vec(2, seq * n_kv_heads * hd);
        let v = lcg_vec(3, seq * n_kv_heads * vd);
        let sinks = lcg_vec(4, n_heads);
        for (window, sink, cap) in [
            (None, None, None),
            (Some(3), None, None),
            (None, Some(sinks.as_slice()), None),
            (Some(4), Some(sinks.as_slice()), None),
            (None, None, Some(5.0)),
        ] {
            let row = super::causal_gqa_attention_row(
                &q, &k, &v, n_heads, n_kv_heads, hd, vd, seq, window, sink, cap, None,
            );
            assert_eq!(row.len(), n_heads * vd);
            // Paged twin over the same rows.
            let mut store = crate::cache::PagedKvStore::new_split(2, 8, n_kv_heads, hd, vd);
            let mut paged = crate::cache::PagedKvCache::new();
            for t in 0..seq {
                paged
                    .push(
                        &mut store,
                        &k[t * n_kv_heads * hd..(t + 1) * n_kv_heads * hd],
                        &v[t * n_kv_heads * vd..(t + 1) * n_kv_heads * vd],
                    )
                    .unwrap();
            }
            let via_pages = super::causal_gqa_attention_paged_sinks(
                &q,
                &store,
                paged.block_table(),
                n_heads,
                n_kv_heads,
                hd,
                seq,
                window,
                sink,
                cap,
                None,
            );
            for (a, b) in row.iter().zip(via_pages.iter()) {
                assert!((a - b).abs() < 1e-6, "row {a} vs paged {b} ({window:?})");
            }
        }
        // MHA shape: the MLA kernel is the independent reference.
        let q1 = lcg_vec(5, n_heads * hd);
        let k1 = lcg_vec(6, seq * n_heads * hd);
        let v1 = lcg_vec(7, seq * n_heads * vd);
        let row = super::causal_gqa_attention_row(
            &q1, &k1, &v1, n_heads, n_heads, hd, vd, seq, None, None, None, None,
        );
        let mla = super::causal_mla_attention(&q1, &k1, &v1, n_heads, hd, vd, seq);
        for (a, b) in row.iter().zip(mla.iter()) {
            assert!((a - b).abs() < 1e-6, "row {a} vs mla {b}");
        }
        // And the width is not merely tolerated: a V width read as the
        // K width would be a different vector.
        assert_ne!(row.len(), n_heads * hd);
    }

    /// ALiBi (`crate::alibi`): the three kernels agree with each other
    /// and with a naive softmax reference that adds `slope_h * (p_key -
    /// p_query)` to every scaled score, windowed and not; and the bias
    /// really moves the output (a slope of zero on every head is the
    /// plain kernel).
    #[test]
    fn alibi_is_one_bias_across_the_row_paged_and_prefill_kernels() {
        let (n_heads, n_kv_heads, hd, prefix, n_q) = (6usize, 3usize, 8usize, 5usize, 9usize);
        let kv_len = prefix + n_q;
        let q = lcg_vec(21, n_q * n_heads * hd);
        let k = lcg_vec(22, kv_len * n_kv_heads * hd);
        let v = lcg_vec(23, kv_len * n_kv_heads * hd);
        let slopes = crate::alibi::slopes(n_heads, 8.0).unwrap();
        let group = n_heads / n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();

        // The naive reference for query row `b` (position `prefix + b`).
        let naive = |b: usize, window: Option<usize>| -> Vec<f32> {
            let q_pos = prefix + b;
            let t_start = window.map_or(0, |w| (q_pos + 1).saturating_sub(w));
            let mut out = vec![0f32; n_heads * hd];
            for h in 0..n_heads {
                let kv_h = h / group;
                let q_h = &q[(b * n_heads + h) * hd..(b * n_heads + h + 1) * hd];
                let scores: Vec<f32> = (t_start..=q_pos)
                    .map(|t| {
                        let k_t =
                            &k[(t * n_kv_heads + kv_h) * hd..(t * n_kv_heads + kv_h + 1) * hd];
                        let dot: f32 = q_h.iter().zip(k_t).map(|(a, b)| a * b).sum();
                        dot * scale + slopes[h] * (t as f32 - q_pos as f32)
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let l: f32 = e.iter().sum();
                for (j, t) in (t_start..=q_pos).enumerate() {
                    let v_t = &v[(t * n_kv_heads + kv_h) * hd..(t * n_kv_heads + kv_h + 1) * hd];
                    for d in 0..hd {
                        out[h * hd + d] += e[j] / l * v_t[d];
                    }
                }
            }
            out
        };

        for window in [None, Some(4)] {
            let prefill = super::causal_gqa_attention_prefill_shared_kv_split(
                &q,
                &k,
                &v,
                n_heads,
                n_kv_heads,
                hd,
                hd,
                n_q,
                prefix,
                None,
                window,
                Some(&slopes),
            );
            let plain = super::causal_gqa_attention_prefill_shared_kv_split(
                &q, &k, &v, n_heads, n_kv_heads, hd, hd, n_q, prefix, None, window, None,
            );
            let mut moved = false;
            for b in 0..n_q {
                let t = prefix + b + 1;
                let want = naive(b, window);
                let row = super::causal_gqa_attention_row(
                    &q[b * n_heads * hd..(b + 1) * n_heads * hd],
                    &k[..t * n_kv_heads * hd],
                    &v[..t * n_kv_heads * hd],
                    n_heads,
                    n_kv_heads,
                    hd,
                    hd,
                    t,
                    window,
                    None,
                    None,
                    Some(&slopes),
                );
                let got = &prefill[b * n_heads * hd..(b + 1) * n_heads * hd];
                for ((g, r), w) in got.iter().zip(&row).zip(&want) {
                    assert!(
                        (g - w).abs() < 1e-5,
                        "prefill {g} vs naive {w} ({window:?})"
                    );
                    assert!((r - w).abs() < 1e-5, "row {r} vs naive {w} ({window:?})");
                }
                let unbiased = &plain[b * n_heads * hd..(b + 1) * n_heads * hd];
                moved |= got.iter().zip(unbiased).any(|(a, b)| (a - b).abs() > 1e-3);
            }
            assert!(moved, "the bias must change the output ({window:?})");
        }

        // The paged kernel reads the same slopes.
        let seq = kv_len;
        let mut store = crate::cache::PagedKvStore::new_split(2, 8, n_kv_heads, hd, hd);
        let mut paged = crate::cache::PagedKvCache::new();
        for t in 0..seq {
            paged
                .push(
                    &mut store,
                    &k[t * n_kv_heads * hd..(t + 1) * n_kv_heads * hd],
                    &v[t * n_kv_heads * hd..(t + 1) * n_kv_heads * hd],
                )
                .unwrap();
        }
        let qn = &q[(n_q - 1) * n_heads * hd..];
        let row = super::causal_gqa_attention_row(
            qn,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            hd,
            hd,
            seq,
            None,
            None,
            None,
            Some(&slopes),
        );
        let via_pages = super::causal_gqa_attention_paged_sinks(
            qn,
            &store,
            paged.block_table(),
            n_heads,
            n_kv_heads,
            hd,
            seq,
            None,
            None,
            None,
            Some(&slopes),
        );
        for (a, b) in row.iter().zip(via_pages.iter()) {
            assert!((a - b).abs() < 1e-6, "row {a} vs paged {b}");
        }
    }

    /// The batched prefill kernel at split widths equals the row kernel
    /// applied position by position, windowed and not, softcapped and
    /// not -- the same equivalence the one-width tests below pin, at
    /// the shape that used to be a refusal.
    #[test]
    fn split_kv_head_width_prefill_matches_the_row_kernel_per_position() {
        let (n_heads, n_kv_heads, hd, vd, prefix, n_q) =
            (4usize, 2usize, 6usize, 4usize, 3usize, 11usize);
        let kv_len = prefix + n_q;
        let q = lcg_vec(11, n_q * n_heads * hd);
        let k = lcg_vec(12, kv_len * n_kv_heads * hd);
        let v = lcg_vec(13, kv_len * n_kv_heads * vd);
        for (window, cap) in [
            (None, None),
            (Some(4), None),
            (None, Some(6.0)),
            (Some(5), Some(6.0)),
        ] {
            let got = super::causal_gqa_attention_prefill_shared_kv_split(
                &q, &k, &v, n_heads, n_kv_heads, hd, vd, n_q, prefix, cap, window, None,
            );
            assert_eq!(got.len(), n_q * n_heads * vd);
            for b in 0..n_q {
                let t = prefix + b + 1;
                let want = super::causal_gqa_attention_row(
                    &q[b * n_heads * hd..(b + 1) * n_heads * hd],
                    &k[..t * n_kv_heads * hd],
                    &v[..t * n_kv_heads * vd],
                    n_heads,
                    n_kv_heads,
                    hd,
                    vd,
                    t,
                    window,
                    None,
                    cap,
                    None,
                );
                let row = &got[b * n_heads * vd..(b + 1) * n_heads * vd];
                for (a, w) in row.iter().zip(want.iter()) {
                    assert!(
                        (a - w).abs() < 1e-5,
                        "b={b} {a} vs {w} ({window:?}, {cap:?})"
                    );
                }
            }
        }
    }

    /// A sink takes probability mass away from the real keys without
    /// contributing to the output, so the result shrinks toward zero
    /// rather than being redistributed. Without a sink the softmax must
    /// spend a full unit of weight on the keys it has, however poorly
    /// they match; with one, a head can decline.
    #[test]
    fn an_mla_sink_removes_weight_from_the_real_keys_instead_of_moving_it() {
        let (n_heads, qk, vd, seq) = (2, 2, 2, 2);
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let v = vec![4.0, 8.0, 1.0, 2.0, 4.0, 8.0, 1.0, 2.0];

        let plain = super::causal_mla_attention(&q, &k, &v, n_heads, qk, vd, seq);
        let none = super::causal_mla_attention_sinks(&q, &k, &v, n_heads, qk, vd, seq, None);
        assert_eq!(plain, none, "no sink must be exactly the old path");

        // A sink far above every score takes nearly all the mass.
        let big = super::causal_mla_attention_sinks(
            &q,
            &k,
            &v,
            n_heads,
            qk,
            vd,
            seq,
            Some(&[40.0, 40.0]),
        );
        for (b, p) in big.iter().zip(plain.iter()) {
            assert!(b.abs() < 1e-6, "a dominant sink leaves ~0, got {b} vs {p}");
        }

        // A sink far below every score changes almost nothing.
        let tiny = super::causal_mla_attention_sinks(
            &q,
            &k,
            &v,
            n_heads,
            qk,
            vd,
            seq,
            Some(&[-40.0, -40.0]),
        );
        for (s, p) in tiny.iter().zip(plain.iter()) {
            assert!((s - p).abs() < 1e-5, "negligible sink: {s} vs {p}");
        }
    }

    /// The sink is per HEAD, so one head may decline while another
    /// attends normally. A single shared sink would be a different
    /// mechanism, and one that cannot express this.
    #[test]
    fn each_head_gets_its_own_mla_sink() {
        let (n_heads, qk, vd, seq) = (2, 1, 1, 1);
        let q = vec![1.0, 1.0];
        let k = vec![1.0, 1.0];
        let v = vec![5.0, 5.0];

        let out = super::causal_mla_attention_sinks(
            &q,
            &k,
            &v,
            n_heads,
            qk,
            vd,
            seq,
            Some(&[40.0, -40.0]),
        );
        assert!(out[0].abs() < 1e-6, "head 0 declined: {}", out[0]);
        assert!(
            (out[1] - 5.0).abs() < 1e-5,
            "head 1 attended normally: {}",
            out[1]
        );
    }

    /// The sink logit is NOT multiplied by the `1/sqrt(qk_head_dim)`
    /// score scale -- it is a learned logit already in score space. If
    /// it were scaled, the same checkpoint would sink differently at
    /// different head widths, which is what this pins down.
    #[test]
    fn the_mla_sink_logit_is_not_scaled_by_the_head_width() {
        // One key whose score is exactly 0 before and after scaling, so
        // the only thing the head width can affect is the sink.
        let sink = 0.0f32;
        let mut outs = Vec::new();
        for qk in [1usize, 4, 16] {
            let q = vec![0.0; qk];
            let k = vec![0.0; qk];
            let v = vec![10.0];
            outs.push(super::causal_mla_attention_sinks(&q, &k, &v, 1, qk, 1, 1, Some(&[sink]))[0]);
        }
        // score 0 and sink 0 split the mass evenly, at every width.
        for o in &outs {
            assert!((o - 5.0).abs() < 1e-5, "expected 5.0, got {o}");
        }
    }

    /// The sparse path takes a sink too, and it matters more there: a
    /// query that sees only the indexer's selection would otherwise be
    /// forced to spend a full unit of weight on it.
    #[test]
    fn the_sparse_mla_path_honours_a_sink_over_the_selected_positions() {
        let (n_heads, qk, vd, seq) = (1, 1, 1, 3);
        let q = vec![1.0];
        let k = vec![1.0, 1.0, 1.0];
        let v = vec![2.0, 4.0, 6.0];
        let visible = [0usize, 2];

        let plain = super::causal_mla_attention_sparse(&q, &k, &v, n_heads, qk, vd, seq, &visible);
        let none = super::causal_mla_attention_sparse_sinks(
            &q, &k, &v, n_heads, qk, vd, seq, &visible, None,
        );
        assert_eq!(plain, none);
        assert!((plain[0] - 4.0).abs() < 1e-5, "mean of 2 and 6");

        let sunk = super::causal_mla_attention_sparse_sinks(
            &q,
            &k,
            &v,
            n_heads,
            qk,
            vd,
            seq,
            &visible,
            Some(&[40.0]),
        );
        assert!(sunk[0].abs() < 1e-6, "a dominant sink leaves ~0");
    }
    #[test]
    fn prefill_shared_kv_matches_per_query_reference() {
        // Shapes chosen to cross the query-block boundary (n_q > 2 blocks,
        // with a partial last block), with a nonzero decoded prefix and
        // grouped KV heads; softcap both off and on. The blocked
        // three-pass softmax must agree with the per-query online
        // accumulator within float noise.
        let n_heads = 6;
        let n_kv_heads = 2;
        let head_dim = 16;
        let n_q = 19;
        let kv_prefix = 5;
        let kv_len = kv_prefix + n_q;
        let q_stride = n_heads * head_dim;
        let kv_stride = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..n_q * q_stride)
            .map(|i| ((i as f32) * 0.013 - 0.7).sin() * 1.3)
            .collect();
        let k_cache: Vec<f32> = (0..kv_len * kv_stride)
            .map(|i| ((i as f32) * 0.017 - 0.3).cos() * 1.1)
            .collect();
        let v_cache: Vec<f32> = (0..kv_len * kv_stride)
            .map(|i| ((i as f32) * 0.011 + 0.2).sin() * 0.9)
            .collect();

        for softcap in [None, Some(30.0)] {
            let got = super::causal_gqa_attention_prefill_shared_kv(
                &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, n_q, kv_prefix, softcap,
            );
            assert_eq!(got.len(), n_q * q_stride);
            for b in 0..n_q {
                let causal_len = kv_prefix + b + 1;
                let want = super::causal_gqa_attention_softcap(
                    &q[b * q_stride..(b + 1) * q_stride],
                    &k_cache[..causal_len * kv_stride],
                    &v_cache[..causal_len * kv_stride],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    causal_len,
                    softcap,
                );
                for (i, (g, w)) in got[b * q_stride..(b + 1) * q_stride]
                    .iter()
                    .zip(want.iter())
                    .enumerate()
                {
                    assert!(
                        (g - w).abs() < 1e-5,
                        "softcap {softcap:?} query {b} slot {i}: blocked {g} vs online {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn windowed_prefill_shared_kv_matches_the_per_query_windowed_reference() {
        // Same shapes as `prefill_shared_kv_matches_per_query_reference`,
        // now against `causal_gqa_attention_windowed_softcap` — the
        // per-query path the decoder's SWA arm used to call. Windows are
        // chosen to sit below, across and above the causal prefix so the
        // `saturating_sub` boundary is exercised on both sides; the last
        // one degenerates to full causal and must match the unwindowed
        // kernel too.
        let n_heads = 6;
        let n_kv_heads = 2;
        let head_dim = 16;
        let n_q = 19;
        let kv_prefix = 5;
        let kv_len = kv_prefix + n_q;
        let q_stride = n_heads * head_dim;
        let kv_stride = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..n_q * q_stride)
            .map(|i| ((i as f32) * 0.013 - 0.7).sin() * 1.3)
            .collect();
        let k_cache: Vec<f32> = (0..kv_len * kv_stride)
            .map(|i| ((i as f32) * 0.017 - 0.3).cos() * 1.1)
            .collect();
        let v_cache: Vec<f32> = (0..kv_len * kv_stride)
            .map(|i| ((i as f32) * 0.011 + 0.2).sin() * 0.9)
            .collect();

        for window in [1usize, 3, 7, kv_prefix, kv_len, kv_len + 8] {
            for softcap in [None, Some(30.0)] {
                let got = super::causal_gqa_attention_prefill_shared_kv_windowed(
                    &q,
                    &k_cache,
                    &v_cache,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    n_q,
                    kv_prefix,
                    softcap,
                    Some(window),
                );
                assert_eq!(got.len(), n_q * q_stride);
                for b in 0..n_q {
                    let causal_len = kv_prefix + b + 1;
                    let want = super::causal_gqa_attention_windowed_softcap(
                        &q[b * q_stride..(b + 1) * q_stride],
                        &k_cache[..causal_len * kv_stride],
                        &v_cache[..causal_len * kv_stride],
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        causal_len,
                        window,
                        softcap,
                    );
                    for (i, (g, w)) in got[b * q_stride..(b + 1) * q_stride]
                        .iter()
                        .zip(want.iter())
                        .enumerate()
                    {
                        assert!(
                            (g - w).abs() < 1e-5,
                            "window {window} softcap {softcap:?} query {b} slot {i}: \
                             blocked {g} vs per-query {w}"
                        );
                    }
                }
            }
        }

        // `window >= kv_len` is full causal: identical to `None`.
        let windowed = super::causal_gqa_attention_prefill_shared_kv_windowed(
            &q,
            &k_cache,
            &v_cache,
            n_heads,
            n_kv_heads,
            head_dim,
            n_q,
            kv_prefix,
            None,
            Some(kv_len + 8),
        );
        let full = super::causal_gqa_attention_prefill_shared_kv(
            &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, n_q, kv_prefix, None,
        );
        assert_eq!(windowed, full);
    }

    /// The query-outer form the blocked kernel had before the K/V rows
    /// were hoisted to the outer loop: one query at a time, streaming
    /// the whole visible K slab and then the whole visible V slab.
    /// Built from the same `dot_f32` / `softmax_row_exp_sum` / `axpy` /
    /// `scale_inplace` primitives, so the kernel must match it **bit for
    /// bit**, not within a tolerance: reordering which rows are loaded
    /// when must not reorder any arithmetic. What those primitives
    /// compute is checked separately, `dot_f32` against a scalar sum and
    /// `softmax_row_exp_sum` against libm's own `expf` in
    /// `vectorised_softmax_row_matches_the_scalar_libm_form`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_query_outer_reference(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_q: usize,
        kv_prefix: usize,
        attn_softcap: Option<f32>,
        window: Option<usize>,
    ) -> Vec<f32> {
        let q_stride = n_heads * head_dim;
        let group_size = n_heads / n_kv_heads.max(1);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let softcap = attn_softcap.filter(|&c| c > 0.0);
        let mut out = vec![0f32; n_q * q_stride];
        let mut acc = vec![0f32; head_dim];
        for h in 0..n_heads {
            let kv_h = h / group_size.max(1);
            for b in 0..n_q {
                let causal_len = kv_prefix + b + 1;
                let t_start = match window {
                    Some(w) => causal_len.saturating_sub(w),
                    None => 0,
                };
                let q_h = &q[b * q_stride + h * head_dim..][..head_dim];
                let mut scores = vec![0f32; causal_len - t_start];
                for (i, s) in scores.iter_mut().enumerate() {
                    let base = ((t_start + i) * n_kv_heads + kv_h) * head_dim;
                    let mut v = super::dot_f32(q_h, &k_cache[base..base + head_dim]) * scale;
                    if let Some(sc) = softcap {
                        v = sc * (v / sc).tanh();
                    }
                    *s = v;
                }
                let l = super::softmax_row_exp_sum(&mut scores);
                acc.fill(0.0);
                for (i, &p) in scores.iter().enumerate() {
                    let base = ((t_start + i) * n_kv_heads + kv_h) * head_dim;
                    super::axpy(&mut acc, &v_cache[base..base + head_dim], p);
                }
                if l > 0.0 {
                    super::scale_inplace(&mut acc, 1.0 / l);
                }
                out[b * q_stride + h * head_dim..][..head_dim].copy_from_slice(&acc);
            }
        }
        out
    }

    #[test]
    fn position_outer_prefill_is_bit_identical_to_the_query_outer_form() {
        // Shapes cross the query-block boundary with a partial last
        // block, a nonzero decoded prefix and grouped KV heads. Windows
        // are chosen so the visible span is narrower than, equal to and
        // wider than the block, plus the unwindowed case.
        let n_heads = 6;
        let n_kv_heads = 2;
        let head_dim = 16;
        let n_q = 19;
        let kv_prefix = 5;
        let kv_len = kv_prefix + n_q;
        let q_stride = n_heads * head_dim;
        let kv_stride = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..n_q * q_stride)
            .map(|i| ((i as f32) * 0.013 - 0.7).sin() * 1.3)
            .collect();
        let k_cache: Vec<f32> = (0..kv_len * kv_stride)
            .map(|i| ((i as f32) * 0.017 - 0.3).cos() * 1.1)
            .collect();
        let v_cache: Vec<f32> = (0..kv_len * kv_stride)
            .map(|i| ((i as f32) * 0.011 + 0.2).sin() * 0.9)
            .collect();

        for window in [None, Some(1), Some(3), Some(8), Some(9), Some(kv_len + 4)] {
            for softcap in [None, Some(30.0)] {
                let got = super::causal_gqa_attention_prefill_shared_kv_windowed(
                    &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, n_q, kv_prefix, softcap,
                    window,
                );
                let want = prefill_query_outer_reference(
                    &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, n_q, kv_prefix, softcap,
                    window,
                );
                assert_eq!(got, want, "window {window:?} softcap {softcap:?}");
            }
        }
    }

    #[test]
    fn tiled_prefill_gemm_is_bit_identical_across_awkward_shapes() {
        // `qk_tile` works a 4-query × 4-key register tile and `pv_tile`
        // an 8-query × 8-dim one, each with a row-at-a-time edge path
        // for what the tile does not cover. Every one of those edges,
        // and every `head_dim` width a real checkpoint uses, has to land
        // on the *same* arithmetic as the row-at-a-time form — so this
        // asserts bit equality against `prefill_query_outer_reference`,
        // not a tolerance.
        //
        // Swept: head_dim 64 (Llama/SmolLM2/Qwen3), 80 (Phi-4-mini), 128
        // (Qwen2.5/Mistral), 256 (Gemma-3); head counts that are not a
        // multiple of the tile; `n_q` below, on and off both tile
        // boundaries; GQA and MQA grouping; windows narrower than the
        // prompt (so the visible span is narrower than a query block);
        // and softcap on and off.
        let shapes = [
            (4usize, 4usize, 64usize),
            (6, 2, 64),
            (5, 1, 80),
            (3, 3, 128),
            (2, 1, 256),
        ];
        let batches = [(19usize, 5usize), (8, 0), (3, 7), (16, 1), (7, 0)];
        for &(n_heads, n_kv_heads, head_dim) in &shapes {
            let q_stride = n_heads * head_dim;
            let kv_stride = n_kv_heads * head_dim;
            for &(n_q, kv_prefix) in &batches {
                let kv_len = kv_prefix + n_q;
                let q: Vec<f32> = (0..n_q * q_stride)
                    .map(|i| ((i as f32) * 0.013 - 0.7).sin() * 1.3)
                    .collect();
                let k_cache: Vec<f32> = (0..kv_len * kv_stride)
                    .map(|i| ((i as f32) * 0.017 - 0.3).cos() * 1.1)
                    .collect();
                let v_cache: Vec<f32> = (0..kv_len * kv_stride)
                    .map(|i| ((i as f32) * 0.011 + 0.2).sin() * 0.9)
                    .collect();
                for window in [None, Some(2), Some(5), Some(9), Some(kv_len + 3)] {
                    for softcap in [None, Some(30.0)] {
                        let got = super::causal_gqa_attention_prefill_shared_kv_windowed(
                            &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, n_q, kv_prefix,
                            softcap, window,
                        );
                        let want = prefill_query_outer_reference(
                            &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, n_q, kv_prefix,
                            softcap, window,
                        );
                        assert_eq!(
                            got, want,
                            "heads {n_heads}/{n_kv_heads} head_dim {head_dim} n_q {n_q} \
                             kv_prefix {kv_prefix} window {window:?} softcap {softcap:?}"
                        );
                    }
                }
            }
        }
    }

    /// The blocked kernel and its query-outer reference share
    /// `softmax_row_exp_sum`, so their bit-equality test cannot see a
    /// wrong exponential -- it would be equally wrong on both sides.
    /// This is the test that can: the vectorised routine against
    /// `f32::exp`, i.e. against libm, which is what the kernel called
    /// before.
    ///
    /// Swept: every length from 0 through twice the widest vector so all
    /// four NEON and all eight AVX2 tail positions are hit; rows whose
    /// spread is far past the `-87` clamp, where the vector form floors
    /// at `~1.6e-38` and libm returns a true zero; a constant row, where
    /// every term is `exp(0)` and the sum must come out at exactly the
    /// row length; and a row of one.
    #[test]
    fn vectorised_softmax_row_matches_the_scalar_libm_form() {
        fn libm_reference(x: &[f32]) -> (Vec<f32>, f32) {
            let m = x.iter().fold(f32::NEG_INFINITY, |a, &s| a.max(s));
            let out: Vec<f32> = x.iter().map(|s| (s - m).exp()).collect();
            let mut l = 0f32;
            for &e in out.iter() {
                l += e;
            }
            (out, l)
        }

        // `spread` scales the score range: 200.0 pushes the low tail
        // past the clamp, 0.0 makes every score identical.
        for spread in [1.0f32, 8.0, 200.0, 0.0] {
            for n in (0..=17).chain([31, 32, 33, 64, 127, 512]) {
                let row: Vec<f32> = (0..n)
                    .map(|i| ((i as f32) * 0.37 - 1.1).sin() * spread)
                    .collect();
                let (want, want_l) = libm_reference(&row);

                let mut got = row.clone();
                let got_l = super::softmax_row_exp_sum(&mut got);

                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-6 * w + 1e-30,
                        "spread {spread} n {n} slot {i}: vector {g} vs libm {w}"
                    );
                }
                assert!(
                    (got_l - want_l).abs() <= 1e-5 * want_l.max(1.0),
                    "spread {spread} n {n}: sum {got_l} vs libm {want_l}"
                );
                if n == 0 {
                    assert_eq!(got_l, 0.0, "an empty visible range normalises to nothing");
                }
                if spread == 0.0 && n > 0 {
                    // Every score equal means every term is `exp(0)`, and
                    // the routine has to return that as *exactly* 1.0 --
                    // an approximation that drifts at zero would bias
                    // every uniform attention row.
                    for (i, g) in got.iter().enumerate() {
                        assert_eq!(*g, 1.0, "n {n} slot {i}: exp(0) must be exact");
                    }
                }
            }
        }
    }

    /// Replacing libm's `expf` and a sequential `f32` sum changes the
    /// last bits of every attention probability, and across a 26-layer
    /// prefill that is enough to move a greedy argmax on a near-tie. So
    /// "different from what the scalar form produced" is not the
    /// question worth asking, because the answer is yes and will stay
    /// yes. "Further from the true softmax" is the question, and this
    /// answers it against an `f64` ground truth.
    ///
    /// Measured on this sweep: the probabilities come out at the same
    /// accuracy as the scalar form (both within a factor of two of each
    /// other, both growing together with the spread of the row, because
    /// the shared error term is rounding `score - max` into `f32`, not
    /// the exponential); the normaliser comes out **better**, by 3x to
    /// 10x, because four partial sums is a pairwise reduction and
    /// `l += *s` down the row is not.
    #[test]
    fn the_vectorised_softmax_is_no_less_accurate_than_the_scalar_one() {
        for spread in [1.0f64, 6.0, 20.0] {
            for n in [64usize, 253, 512] {
                let row: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.37 - 1.1).sin() * spread)
                    .collect();

                // Ground truth: the same reduction in `f64`.
                let m64 = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let truth: Vec<f64> = row.iter().map(|s| (s - m64).exp()).collect();
                let truth_l: f64 = truth.iter().sum();

                let f32_row: Vec<f32> = row.iter().map(|&s| s as f32).collect();

                let mut vector = f32_row.clone();
                let vector_l = super::softmax_row_exp_sum(&mut vector);
                let mut scalar = f32_row.clone();
                let scalar_l = super::softmax_row_exp_sum_scalar(&mut scalar);

                let worst = |got: &[f32]| -> f64 {
                    got.iter()
                        .zip(truth.iter())
                        .map(|(&g, &t)| ((g as f64) - t).abs() / t)
                        .fold(0.0, f64::max)
                };
                let (ev, es) = (worst(&vector), worst(&scalar));
                let lv = ((vector_l as f64) - truth_l).abs() / truth_l;
                let ls = ((scalar_l as f64) - truth_l).abs() / truth_l;
                let eps = f64::from(f32::EPSILON);

                // Probabilities: the same accuracy, within a factor of
                // two either way. Neither form is the error term here --
                // rounding `score - max` into `f32` is, which is why the
                // error grows with the spread of the row and why both
                // forms grow with it together.
                assert!(
                    ev <= 2.0 * es.max(eps) && ev <= 64.0 * eps,
                    "spread {spread} n {n}: vector probabilities err {ev:e} \
                     against scalar {es:e}"
                );

                // The normaliser: the vector form is the better one,
                // every time. Four (or eight) partial sums is a pairwise
                // reduction; `l += *s` down a 512-wide row is not.
                assert!(
                    lv <= ls.max(eps),
                    "spread {spread} n {n}: vector normaliser err {lv:e} \
                     against scalar {ls:e}"
                );
            }
        }
    }

    /// The row max must be the true max whichever lane it lands in, and
    /// the largest term must come back as exactly `1.0`, because the
    /// caller divides by the sum rather than tracking a running maximum.
    #[test]
    fn softmax_row_finds_its_maximum_in_every_lane_position() {
        for n in 1usize..=20 {
            for peak in 0..n {
                let mut row: Vec<f32> = (0..n).map(|i| -(i as f32) - 3.0).collect();
                row[peak] = 12.5;
                let l = super::softmax_row_exp_sum(&mut row);
                assert_eq!(row[peak], 1.0, "n {n} peak {peak}: the max term is exp(0)");
                for (i, &p) in row.iter().enumerate() {
                    assert!(p <= 1.0, "n {n} peak {peak} slot {i}: {p} exceeds the max");
                }
                assert!(l >= 1.0, "n {n} peak {peak}: sum {l} must include the max");
            }
        }
    }

    use super::*;

    #[test]
    fn simd_dot_f32_matches_scalar_across_lengths() {
        // Cover exact-multiple and tail lengths around the SIMD width.
        for n in [1usize, 3, 4, 7, 8, 15, 16, 63, 128, 129] {
            let a: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.31 - 2.0).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.17 + 1.0).cos()).collect();
            let simd = dot_f32(&a, &b);
            let scalar: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!(
                (simd - scalar).abs() <= 1e-4 * scalar.abs().max(1.0),
                "n={n} simd={simd} scalar={scalar}"
            );
        }
    }

    #[test]
    fn rope_preserves_vector_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope(&mut v, 5, 10000.0);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE is a rotation and must preserve norm"
        );
    }

    #[test]
    fn rope_back_inverts_rope() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut v = original.clone();
        apply_rope(&mut v, 11, 10000.0);
        apply_rope_back(&mut v, 11, 10000.0);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_interleaved_back_inverts_interleaved() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut v = original.clone();
        apply_rope_interleaved(&mut v, 11, 10000.0);
        apply_rope_interleaved_back(&mut v, 11, 10000.0);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let original = v.clone();
        apply_rope(&mut v, 0, 10000.0);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_with_all_ones_freq_factors_matches_plain_rope() {
        let mut with_factors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut plain = with_factors.clone();
        let ones = vec![1.0; 3];
        apply_rope_with_freq_factors(&mut with_factors, 7, 10000.0, &ones);
        apply_rope(&mut plain, 7, 10000.0);
        for (a, b) in with_factors.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_with_freq_factors_diverges_from_plain_rope_when_factors_are_not_one() {
        let mut with_factors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut plain = with_factors.clone();
        let factors = vec![0.5, 2.0, 1.0];
        apply_rope_with_freq_factors(&mut with_factors, 7, 10000.0, &factors);
        apply_rope(&mut plain, 7, 10000.0);
        let differs = with_factors
            .iter()
            .zip(plain.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "non-1.0 freq_factors must change the rotation");
    }

    #[test]
    fn rope_with_freq_factors_preserves_vector_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_with_freq_factors(&mut v, 5, 10000.0, &[0.8, 1.3]);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-4);
    }

    /// The correction range's high end is clamped to `rotary_dim - 1`,
    /// which is *above* the ramp's own last index (`rotary_dim/2 - 1`),
    /// so the ramp never reaches `1.0` and the longest-wavelength bands
    /// stay partly extrapolated.
    ///
    /// **This test fails if `high` is clamped the naive way** (to
    /// `rotary_dim / 2 - 1`): that makes `ramp == 1.0` at the last band,
    /// whose divisor then becomes the full scaling factor `8.0` instead
    /// of the reference's `2.5366`. Numbers below are computed by hand
    /// from `_find_correction_dim` (`rotary.py:161`), not from this
    /// implementation: for `rotary_dim` 64, base 10000, original context
    /// 131072, `low = floor(22.5134) = 22` and
    /// `high = ceil(34.5546) = 35`.
    #[test]
    fn yarn_high_is_clamped_to_rotary_dim_minus_one_not_half_minus_one() {
        let scaling = YarnScaling::new(8.0, 131_072);
        let (low, high) = yarn_correction_range(scaling, 64, 10_000.0);
        assert!((low - 22.0).abs() < 1e-9, "low was {low}");
        assert!((high - 35.0).abs() < 1e-9, "high was {high}");

        let factors = yarn_freq_factors(scaling, 64, 10_000.0);
        let last = factors[31];
        let ramp = (31.0 - 22.0) / (35.0 - 22.0);
        let want = 1.0 / (ramp / 8.0 + (1.0 - ramp));
        assert!(
            (last - want).abs() < 1e-4,
            "last band divisor {last} must be the reference's {want}"
        );
        assert!(
            (last - 8.0).abs() > 1.0,
            "clamping high to rotary_dim/2 - 1 would fully interpolate this band \
             (divisor 8.0, the whole factor); got {last}"
        );
    }

    /// Band-by-band against the reference ramp
    /// (`rotary.py:181-187`), with `low`/`high` computed by hand as in
    /// the test above: bands at or below `low` are pure extrapolation
    /// (divisor exactly 1.0) and each band past it interpolates by
    /// `1 / (ramp/factor + 1 - ramp)`.
    #[test]
    fn yarn_freq_factors_match_the_reference_ramp_formula_band_by_band() {
        let scaling = YarnScaling::new(8.0, 131_072);
        let factors = yarn_freq_factors(scaling, 64, 10_000.0);
        assert_eq!(factors.len(), 32, "one divisor per rotation band");
        for band in [0usize, 10, 22] {
            assert!(
                (factors[band] - 1.0).abs() < 1e-6,
                "band {band} is at or below low=22 and must be left extrapolated, \
                 got {}",
                factors[band]
            );
        }
        for band in [23usize, 27, 31] {
            let ramp = (band as f32 - 22.0) / (35.0 - 22.0);
            let want = 1.0 / (ramp / 8.0 + (1.0 - ramp));
            assert!(
                (factors[band] - want).abs() < 1e-4,
                "band {band}: got {}, reference {want}",
                factors[band]
            );
        }
    }

    /// A collapsed correction range (`low == high`, which
    /// `truncate: false` with `beta_fast == beta_slow` produces) is
    /// nudged by `+0.001`, making the ramp a step at `low`: the band
    /// below stays fully extrapolated and the band above is fully
    /// interpolated at the whole factor.
    ///
    /// **This test fails if the collapse is handled by flooring the gap
    /// at 1** (`high = low + 1`), the other obvious repair: band 23 is
    /// then only `0.4866` of the way up the ramp and its divisor is
    /// `1.7414`, not `8.0`.
    #[test]
    fn yarn_nudges_a_collapsed_correction_range_instead_of_flooring_the_gap_at_one() {
        let scaling = YarnScaling {
            beta_slow: 32.0,
            truncate: false,
            ..YarnScaling::new(8.0, 131_072)
        };
        let (low, high) = yarn_correction_range(scaling, 64, 10_000.0);
        // Hand-computed: _find_correction_dim(32) = 22.513440...
        assert!((low - 22.513_44).abs() < 1e-4, "low was {low}");
        assert!(
            (high - low - 0.001).abs() < 1e-9,
            "high must be low + 0.001, got {high}"
        );

        let factors = yarn_freq_factors(scaling, 64, 10_000.0);
        assert!(
            (factors[22] - 1.0).abs() < 1e-6,
            "band below the step must be untouched, got {}",
            factors[22]
        );
        assert!(
            (factors[23] - 8.0).abs() < 1e-4,
            "band above the step must take the whole factor (a gap of 1 would \
             give 1.7414); got {}",
            factors[23]
        );
    }

    /// `factor = 1.0` means "the served context is the trained context":
    /// every band's divisor must be exactly 1.0, i.e. mathematically the
    /// same rotation as no scaling at all. A checkpoint that declares
    /// YaRN with a no-op factor must not have its RoPE moved.
    #[test]
    fn yarn_with_a_factor_of_one_leaves_every_band_untouched() {
        let factors = yarn_freq_factors(YarnScaling::new(1.0, 4096), 32, 10_000.0);
        for (band, f) in factors.iter().enumerate() {
            assert!((f - 1.0).abs() < 1e-6, "band {band} moved to {f}");
        }
    }

    /// The divisors are only a re-expression of the reference's
    /// rewritten `inv_freq`, so rotating through
    /// [`apply_rope_with_freq_factors`] must land on exactly the angle
    /// the reference's `inv_freq_new` implies. Checked against the
    /// reference formula (`inv_freq * (ramp/factor + 1 - ramp)`)
    /// evaluated here, not against this module's own divisor.
    #[test]
    fn yarn_divisors_reproduce_the_references_rewritten_frequencies() {
        let scaling = YarnScaling::new(8.0, 131_072);
        let factors = yarn_freq_factors(scaling, 64, 10_000.0);

        let band = 31usize;
        let pos = 1024usize;
        let mut v = vec![0.0f32; 64];
        v[band] = 1.0;
        apply_rope_with_freq_factors(&mut v, pos, 10_000.0, &factors);

        let ramp = (band as f64 - 22.0) / (35.0 - 22.0);
        let inv_freq = 1.0 / 10_000f64.powf((2 * band) as f64 / 64.0);
        let inv_freq_new = inv_freq * (ramp / 8.0 + (1.0 - ramp));
        let angle = pos as f64 * inv_freq_new;
        assert!(
            (v[band] as f64 - angle.cos()).abs() < 1e-5,
            "cos: {} vs {}",
            v[band],
            angle.cos()
        );
        assert!(
            (v[band + 32] as f64 - angle.sin()).abs() < 1e-5,
            "sin: {} vs {}",
            v[band + 32],
            angle.sin()
        );
    }

    /// The proportional arm spaces frequencies over the *full* head
    /// while only the first `rotary_dim` channels rotate. Values are
    /// hand-computed from `rotary.py:103`'s
    /// `base ** (arange(0, head_size, 2) / head_size)` against this
    /// crate's own `base ** (2i / rotary_dim)` spacing: the divisor is
    /// their ratio.
    #[test]
    fn proportional_freq_factors_respace_frequencies_over_the_full_head() {
        let factors = proportional_freq_factors(128, 96, 10_000.0);
        assert_eq!(factors.len(), 48, "one divisor per rotated band");
        assert!((factors[0] - 1.0).abs() < 1e-6, "band 0 is 1/1");
        // 10000^(2/128 - 2/96) = 0.95316188...
        assert!(
            (factors[1] - 0.953_161_9).abs() < 1e-5,
            "band 1 was {}",
            factors[1]
        );
        // 10000^(94/128 - 94/96) = 0.10491397...
        assert!(
            (factors[47] - 0.104_913_97).abs() < 1e-5,
            "last band was {}",
            factors[47]
        );
    }

    /// Full-width rope is the case where both spacings coincide, so the
    /// arm must be a no-op there rather than quietly re-scaling every
    /// band of an ordinary checkpoint.
    #[test]
    fn proportional_freq_factors_are_all_ones_when_the_whole_head_rotates() {
        for f in proportional_freq_factors(128, 128, 500_000.0) {
            assert!((f - 1.0).abs() < 1e-6, "full-width band moved to {f}");
        }
    }

    #[test]
    fn rope_interleaved_preserves_vector_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_interleaved(&mut v, 5, 10000.0);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE is a rotation and must preserve norm"
        );
    }

    #[test]
    fn rope_interleaved_at_position_zero_is_identity() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let original = v.clone();
        apply_rope_interleaved(&mut v, 0, 10000.0);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_interleaved_rotates_adjacent_pairs_not_split_halves() {
        // With a single frequency band (dim=2), interleaved and split-half
        // RoPE are mathematically identical (both rotate the one (v[0],
        // v[1]) pair). The two conventions only diverge once dim > 2 and
        // there's more than one frequency band to route pairs into --
        // that's the real bug class this test guards against: mixing up
        // which components get paired together.
        let mut interleaved = vec![1.0, 0.0, 0.0, 1.0];
        let mut split_half = interleaved.clone();
        apply_rope_interleaved(&mut interleaved, 3, 10000.0);
        apply_rope(&mut split_half, 3, 10000.0);
        // Different frequency assigned to each pair in the two
        // conventions (interleaved pairs (0,1)+(2,3), split-half pairs
        // (0,2)+(1,3)) so with two distinct frequency bands the outputs
        // must differ.
        let differs = interleaved
            .iter()
            .zip(split_half.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "the two RoPE conventions must not coincide here");
    }

    #[test]
    fn rope_interleaved_with_all_ones_freq_factors_matches_plain_interleaved() {
        let mut with_factors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut plain = with_factors.clone();
        let ones = vec![1.0; 3];
        apply_rope_interleaved_with_freq_factors(&mut with_factors, 7, 10000.0, &ones);
        apply_rope_interleaved(&mut plain, 7, 10000.0);
        for (a, b) in with_factors.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_interleaved_with_freq_factors_diverges_when_factors_are_not_one() {
        let mut with_factors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut plain = with_factors.clone();
        let factors = vec![0.5, 2.0, 1.0];
        apply_rope_interleaved_with_freq_factors(&mut with_factors, 7, 10000.0, &factors);
        apply_rope_interleaved(&mut plain, 7, 10000.0);
        let differs = with_factors
            .iter()
            .zip(plain.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "non-1.0 freq_factors must change the rotation");
    }

    #[test]
    fn rope_interleaved_with_freq_factors_preserves_vector_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_interleaved_with_freq_factors(&mut v, 5, 10000.0, &[0.8, 1.3]);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-4);
    }

    #[test]
    fn attention_with_single_position_returns_that_value() {
        // With one cached position, attention weight is trivially 1.0,
        // so output must equal that single V vector regardless of Q/K.
        let q = vec![1.0, 0.0]; // 1 head, head_dim=2
        let k_cache = vec![0.5, 0.5]; // seq_len=1, 1 kv head
        let v_cache = vec![9.0, -3.0];
        let out = causal_gqa_attention(&q, &k_cache, &v_cache, 1, 1, 2, 1);
        assert!((out[0] - 9.0).abs() < 1e-4);
        assert!((out[1] - (-3.0)).abs() < 1e-4);
    }

    #[test]
    fn gqa_group_mapping_shares_kv_heads_correctly() {
        // 4 query heads, 2 kv heads -> heads 0,1 use kv head 0; heads 2,3 use kv head 1.
        let head_dim = 2;
        let q = vec![
            1.0, 0.0, // head 0
            1.0, 0.0, // head 1
            1.0, 0.0, // head 2
            1.0, 0.0, // head 3
        ];
        // seq_len = 1, 2 kv heads
        let k_cache = vec![1.0, 0.0, 1.0, 0.0];
        let v_cache = vec![100.0, 100.0, 200.0, 200.0];
        let out = causal_gqa_attention(&q, &k_cache, &v_cache, 4, 2, head_dim, 1);
        // heads 0,1 -> kv head 0 -> v = [100,100]; heads 2,3 -> kv head 1 -> v=[200,200]
        assert_eq!(&out[0..2], &[100.0, 100.0][..]);
        assert_eq!(&out[2..4], &[100.0, 100.0][..]);
        assert_eq!(&out[4..6], &[200.0, 200.0][..]);
        assert_eq!(&out[6..8], &[200.0, 200.0][..]);
    }

    #[test]
    fn prefill_gqa_matches_per_token_causal() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 5;
        let q: Vec<f32> = (0..seq_len * n_heads * head_dim)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.19).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let batched =
            causal_gqa_attention_prefill(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);
        let q_stride = n_heads * head_dim;
        let kv_stride = n_kv_heads * head_dim;
        for t in 0..seq_len {
            let expect = causal_gqa_attention(
                &q[t * q_stride..(t + 1) * q_stride],
                &k[..(t + 1) * kv_stride],
                &v[..(t + 1) * kv_stride],
                n_heads,
                n_kv_heads,
                head_dim,
                t + 1,
            );
            let got = &batched[t * q_stride..(t + 1) * q_stride];
            for (a, b) in got.iter().zip(expect.iter()) {
                assert!((a - b).abs() < 1e-5, "t={t}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn windowed_attention_with_window_covering_full_history_matches_full_causal() {
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 3;
        let seq_len = 4;

        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let k_cache: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.17).cos())
            .collect();
        let v_cache: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();

        let full = causal_gqa_attention(
            &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, seq_len,
        );
        let windowed = causal_gqa_attention_windowed(
            &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, seq_len, seq_len,
        );
        assert_eq!(full.len(), windowed.len());
        for (a, b) in full.iter().zip(windowed.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "window >= seq_len must be bit-identical to full causal"
            );
        }
    }

    #[test]
    fn windowed_attention_ignores_positions_outside_the_window() {
        // 1 head, seq_len=3, window=1: only the current position (t=2)
        // should ever be attended to, so the output must equal exactly
        // that position's V vector regardless of Q/K -- masking every
        // earlier position out means there is exactly one candidate
        // left, and softmax over one candidate is trivially 1.0.
        let head_dim = 2;
        let q = vec![1.0, 0.0];
        let k_cache = vec![9.0, -9.0, 0.5, 0.5, -3.0, 7.0]; // seq_len=3, 1 kv head
        let v_cache = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let out = causal_gqa_attention_windowed(&q, &k_cache, &v_cache, 1, 1, head_dim, 3, 1);
        assert!((out[0] - 50.0).abs() < 1e-4);
        assert!((out[1] - 60.0).abs() < 1e-4);
    }

    #[test]
    fn paged_attention_matches_contiguous_attention_bit_identical() {
        use crate::cache::{PagedKvCache, PagedKvStore};

        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 3;
        let block_size = 2;
        let seq_len = 5;

        // Deterministic pseudo-random-ish K/V/Q values, no RNG needed.
        let k_flat: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| ((i * 7 + 1) % 13) as f32 * 0.1)
            .collect();
        let v_flat: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| ((i * 5 + 3) % 11) as f32 * 0.1)
            .collect();
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i * 3 + 2) % 9) as f32 * 0.1)
            .collect();

        let contiguous =
            causal_gqa_attention(&q, &k_flat, &v_flat, n_heads, n_kv_heads, head_dim, seq_len);

        let mut store = PagedKvStore::new(block_size, seq_len, n_kv_heads, head_dim);
        let mut cache = PagedKvCache::new();
        for t in 0..seq_len {
            let start = t * n_kv_heads * head_dim;
            let end = start + n_kv_heads * head_dim;
            cache
                .push(&mut store, &k_flat[start..end], &v_flat[start..end])
                .expect("store sized for seq_len blocks, must not exhaust");
        }

        let paged = causal_gqa_attention_paged(
            &q,
            &store,
            cache.block_table(),
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
        );

        assert_eq!(contiguous.len(), paged.len());
        for (a, b) in contiguous.iter().zip(paged.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "paged path must be bit-identical");
        }
    }

    /// A helper for the paged/contiguous comparisons below: the same
    /// K/V pushed into a paged store, so only the ADDRESSING differs
    /// between the two kernels under test.
    fn paged_fixture(
        seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> (Vec<f32>, Vec<f32>, crate::cache::PagedKvStore, Vec<usize>) {
        use crate::cache::{PagedKvCache, PagedKvStore};
        let k_flat: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| ((i * 7 + 1) % 13) as f32 * 0.1)
            .collect();
        let v_flat: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| ((i * 5 + 3) % 11) as f32 * 0.1)
            .collect();
        let mut store = PagedKvStore::new(block_size, seq_len, n_kv_heads, head_dim);
        let mut cache = PagedKvCache::new();
        for t in 0..seq_len {
            let start = t * n_kv_heads * head_dim;
            let end = start + n_kv_heads * head_dim;
            cache
                .push(&mut store, &k_flat[start..end], &v_flat[start..end])
                .expect("store sized for seq_len blocks, must not exhaust");
        }
        let table = cache.block_table().to_vec();
        (k_flat, v_flat, store, table)
    }

    /// The paged kernel's sink term must be BIT-identical to the
    /// contiguous one, not merely close.
    ///
    /// This is what let `forward_token_paged` stop refusing gpt-oss.
    /// The whole premise of moving a model onto paged KV is that its
    /// distribution does not change, so "within tolerance" is not the
    /// bar -- a distribution that differs in the last bit is still a
    /// different distribution, and it would show up as a model that
    /// answers differently depending on which cache it happened to be
    /// served from.
    #[test]
    fn the_paged_sink_term_is_bit_identical_to_the_contiguous_one() {
        let (n_heads, n_kv_heads, head_dim, seq_len, block_size) = (4, 2, 3, 5, 2);
        let (k_flat, v_flat, store, table) =
            paged_fixture(seq_len, n_kv_heads, head_dim, block_size);
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i * 3 + 2) % 9) as f32 * 0.1)
            .collect();

        // A spread of sinks, including one that dominates and one that
        // is negligible, so the comparison covers both ends of the
        // online-softmax rescale rather than a single middling value.
        let sinks = vec![-30.0f32, 0.0, 1.5, 30.0];
        let contiguous = causal_gqa_attention_sinks(
            &q, &k_flat, &v_flat, n_heads, n_kv_heads, head_dim, seq_len, None, &sinks,
        );
        let paged = causal_gqa_attention_paged_sinks(
            &q,
            &store,
            &table,
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            None,
            Some(&sinks),
            None,
            None,
        );
        assert_eq!(contiguous.len(), paged.len());
        for (i, (a, b)) in contiguous.iter().zip(paged.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} vs {b}");
        }
    }

    /// The window arm too, at a width that really drops positions --
    /// and at one that covers the whole history, which must degenerate
    /// to full causal rather than to an off-by-one.
    #[test]
    fn the_paged_window_arm_is_bit_identical_to_the_contiguous_one() {
        let (n_heads, n_kv_heads, head_dim, seq_len, block_size) = (4, 2, 3, 7, 2);
        let (k_flat, v_flat, store, table) =
            paged_fixture(seq_len, n_kv_heads, head_dim, block_size);
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i * 3 + 2) % 9) as f32 * 0.1)
            .collect();
        let sinks = vec![0.5f32; n_heads];

        for window in [1usize, 2, 3, 6, 7, 99] {
            let contiguous = causal_gqa_attention_sinks(
                &q,
                &k_flat,
                &v_flat,
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                Some(window),
                &sinks,
            );
            let paged = causal_gqa_attention_paged_sinks(
                &q,
                &store,
                &table,
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                Some(window),
                Some(&sinks),
                None,
                None,
            );
            for (i, (a, b)) in contiguous.iter().zip(paged.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "window {window} element {i}: {a} vs {b}"
                );
            }
        }
    }

    /// With no sinks and no window it must reproduce the plain paged
    /// kernel exactly, so the new entry point is a strict superset
    /// rather than a second implementation that drifts from it.
    #[test]
    fn the_paged_sink_kernel_without_sinks_or_window_is_the_plain_paged_kernel() {
        let (n_heads, n_kv_heads, head_dim, seq_len, block_size) = (4, 2, 3, 5, 2);
        let (_k, _v, store, table) = paged_fixture(seq_len, n_kv_heads, head_dim, block_size);
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i * 3 + 2) % 9) as f32 * 0.1)
            .collect();

        let plain =
            causal_gqa_attention_paged(&q, &store, &table, n_heads, n_kv_heads, head_dim, seq_len);
        let via_sinks = causal_gqa_attention_paged_sinks(
            &q, &store, &table, n_heads, n_kv_heads, head_dim, seq_len, None, None, None, None,
        );
        for (i, (a, b)) in plain.iter().zip(via_sinks.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "element {i}");
        }
    }

    #[test]
    fn mla_attention_with_single_position_returns_that_value() {
        // Same reasoning as `attention_with_single_position_returns_that_value`,
        // but with distinct qk/v head dims (5 vs 3) to exercise the one real
        // difference from `causal_gqa_attention`.
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0]; // 1 head, qk_head_dim=5
        let k_cache = vec![0.2, 0.2, 0.2, 0.2, 0.2]; // seq_len=1
        let v_cache = vec![9.0, -3.0, 1.0]; // v_head_dim=3
        let out = causal_mla_attention(&q, &k_cache, &v_cache, 1, 5, 3, 1);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 9.0).abs() < 1e-4);
        assert!((out[1] - (-3.0)).abs() < 1e-4);
        assert!((out[2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn mla_attention_every_head_gets_its_own_kv_no_grouping() {
        // Unlike GQA, MLA has no shared-kv-head grouping: with 2 heads and
        // 2 cached kv-head-slots, head 0 must only ever see kv slot 0 and
        // head 1 only kv slot 1.
        let qk_head_dim = 2;
        let v_head_dim = 2;
        let q = vec![1.0, 0.0, 1.0, 0.0]; // 2 heads
        let k_cache = vec![1.0, 0.0, 1.0, 0.0]; // seq_len=1, 2 heads
        let v_cache = vec![100.0, 100.0, 200.0, 200.0];
        let out = causal_mla_attention(&q, &k_cache, &v_cache, 2, qk_head_dim, v_head_dim, 1);
        assert_eq!(&out[0..2], &[100.0, 100.0][..]);
        assert_eq!(&out[2..4], &[200.0, 200.0][..]);
    }

    #[test]
    fn lightning_indexer_topk_keeps_all_positions_when_top_k_covers_them() {
        let indexer_q = vec![vec![1.0, 0.0]];
        let indexer_keys = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.1, 0.9]];
        let indexer_weights = vec![1.0];
        let kept = lightning_indexer_topk(&indexer_q, &indexer_keys, &indexer_weights, 10);
        assert_eq!(kept, vec![0, 1, 2]);
    }

    #[test]
    fn lightning_indexer_topk_selects_highest_scoring_positions() {
        // Query aligned with key 0 (dot=1.0), partially with key 2 (dot=0.9),
        // orthogonal to key 1 (dot=0.0, relu'd score 0). Top-2 must be {0, 2}.
        let indexer_q = vec![vec![1.0, 0.0]];
        let indexer_keys = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.9, 0.1]];
        let indexer_weights = vec![1.0];
        let kept = lightning_indexer_topk(&indexer_q, &indexer_keys, &indexer_weights, 2);
        assert_eq!(kept, vec![0, 2]);
    }

    #[test]
    fn lightning_indexer_topk_relu_zeroes_negative_dot_products() {
        // Key 1's dot product with the query is negative; ReLU floors its
        // score at 0, so it must lose to key 0 (positive) even at top_k=1.
        let indexer_q = vec![vec![1.0, 0.0]];
        let indexer_keys = vec![vec![0.3, 0.0], vec![-1.0, 0.0]];
        let indexer_weights = vec![1.0];
        let kept = lightning_indexer_topk(&indexer_q, &indexer_keys, &indexer_weights, 1);
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn mla_attention_sparse_with_all_positions_visible_matches_full_causal() {
        let qk_head_dim = 3;
        let v_head_dim = 2;
        let seq_len = 4;
        let n_heads = 2;
        let q: Vec<f32> = (0..n_heads * qk_head_dim).map(|i| i as f32 * 0.1).collect();
        let k_cache: Vec<f32> = (0..seq_len * n_heads * qk_head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let v_cache: Vec<f32> = (0..seq_len * n_heads * v_head_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();

        let full = causal_mla_attention(
            &q,
            &k_cache,
            &v_cache,
            n_heads,
            qk_head_dim,
            v_head_dim,
            seq_len,
        );
        let visible: Vec<usize> = (0..seq_len).collect();
        let sparse = causal_mla_attention_sparse(
            &q,
            &k_cache,
            &v_cache,
            n_heads,
            qk_head_dim,
            v_head_dim,
            seq_len,
            &visible,
        );

        assert_eq!(full.len(), sparse.len());
        for (a, b) in full.iter().zip(sparse.iter()) {
            assert!((a - b).abs() < 1e-6, "full={a} sparse={b}");
        }
    }

    #[test]
    fn mla_attention_sparse_ignores_positions_outside_visible_set() {
        // Only position 0 is visible; a wildly different value at position 1
        // must have zero influence on the output.
        let qk_head_dim = 2;
        let v_head_dim = 1;
        let q = vec![1.0, 0.0];
        let k_cache = vec![1.0, 0.0, 1.0, 0.0]; // seq_len=2, identical keys
        let v_cache = vec![5.0, 999.0]; // position 0 -> 5.0, position 1 -> 999.0
        let out = causal_mla_attention_sparse(
            &q,
            &k_cache,
            &v_cache,
            1,
            qk_head_dim,
            v_head_dim,
            2,
            &[0],
        );
        assert_eq!(out.len(), 1);
        assert!((out[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn attn_logit_softcap_changes_output_vs_uncapped() {
        // Softcap must change the attended output relative to the uncapped
        // path (and must not be a no-op identity for large scores).
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let seq_len = 3;
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 + 1.0) * 2.5)
            .collect();
        let k: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.7).sin() * 3.0)
            .collect();
        let v: Vec<f32> = (0..seq_len * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.3).cos())
            .collect();
        let plain = causal_gqa_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);
        let capped = causal_gqa_attention_softcap(
            &q,
            &k,
            &v,
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            Some(30.0),
        );
        assert_eq!(plain.len(), capped.len());
        let differs = plain
            .iter()
            .zip(capped.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(differs, "softcap must change attention output");
        // Softcap None / <=0 must match the uncapped path.
        let none =
            causal_gqa_attention_softcap(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, None);
        for (a, b) in plain.iter().zip(none.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
