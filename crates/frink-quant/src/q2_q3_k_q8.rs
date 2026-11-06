//! **Q2_K x Q8_K and Q3_K x Q8_K**: the int8 dots of a Q2_K or Q3_K row
//! against activations quantized to Q8_K once per matmul, llama.cpp's
//! `ggml_vec_dot_q2_K_q8_K` and `ggml_vec_dot_q3_K_q8_K`.
//!
//! # Why they exist
//!
//! Both kinds had only fused f32 kernels here (`dot_q2_k_f32`,
//! `dot_q3_k_f32`), so the CPU column of `docs/KERNELS.md` read `f32`
//! for them: every row dot decoded the 2- or 3-bit codes and multiplied
//! in f32, and the batched matmul did that once per (row, activation).
//! ggml declares `vec_dot_type = GGML_TYPE_Q8_K` for both
//! (`ggml-cpu.c`, `type_traits_cpu`), so llama.cpp quantizes the
//! activation once and dots in int8. These are that path, and with them
//! a Q2_K or Q3_K matmul quantizes its activations the way ggml does
//! (ADR 0009), as Q4_K, Q5_K, Q6_K and IQ4_XS already do.
//!
//! # What they compute
//!
//! Both kinds are sixteen 16-element sub-blocks per 256-element
//! super-block, in the element order `dequant_q2_k` / `dequant_q3_k`
//! define: for each 128-element half, four 2-bit planes of its 32
//! `qs` bytes (shift 0, 2, 4, 6), each plane two sub-blocks (bytes
//! 0..16, then 16..32).
//!
//! - **Q2_K** (`quants.c:565-615`): value `d*sc*q - dmin*m` with
//!   `q` in 0..=3 and `sc`, `m` the low and high nibble of one byte per
//!   sub-block. The min term needs only the activation's per-16 sums,
//!   which `Q8KActivations::bsums` already carries:
//!   `y.d * (d * sum(sc_j * sum(q * q8)) - dmin * sum(m_j * bsum_j))`.
//! - **Q3_K** (`quants.c:617-693`): value `d*(s-32)*(q - 4*!h)` with
//!   `q` the 2-bit code, `h` bit `m` of `hmask` (one bit per plane) and
//!   `s` the 6-bit scale `q3_k_unpack_scales` unpacks:
//!   `y.d * d * sum((s_j - 32) * sum((q - 4*!h) * q8))`.
//!
//! The scalar bodies are the twins every SIMD arm is checked against,
//! and the f32 kernels are a second reference: on activations exact in
//! Q8_K the two agree to rounding.

use half::f16;

use crate::{
    q3_k_unpack_scales, Q8KActivations, Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMS, Q3_K_BLOCK_BYTES,
    Q3_K_BLOCK_ELEMS, Q3_K_SCALE_BYTES,
};

/// Q3_K's sixteen sub-block scales, already minus 32, as ggml's
/// `utmp`/`aux` unpacking leaves them.
#[inline]
fn q3_k_scales(block: &[u8; Q3_K_BLOCK_BYTES]) -> [i8; 16] {
    let raw: &[u8; Q3_K_SCALE_BYTES] = block[96..108].try_into().unwrap();
    q3_k_unpack_scales(raw).map(|s| s - 32)
}

/// Q2_K row x Q8_K activations. Dispatches to the SIMD arm the host has
/// (AVX2 on x86, SDOT on aarch64) and falls back to the scalar twin.
pub fn dot_q2_k_q8_k(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the features are detected above; `row_bytes` is
            // whole Q2_K blocks with one Q8_K block each in `act`
            // (asserted in debug inside).
            return unsafe { x86::dot_q2_k_q8_k_avx2(row_bytes, act) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: as the x86 arm, for `dotprod`.
            return unsafe { aarch64::dot_q2_k_q8_k_neon_sdot(row_bytes, act) };
        }
    }
    dot_q2_k_q8_k_scalar(row_bytes, act)
}

/// Q3_K row x Q8_K activations. Same dispatch as [`dot_q2_k_q8_k`].
pub fn dot_q3_k_q8_k(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: as in `dot_q2_k_q8_k`, for Q3_K blocks.
            return unsafe { x86::dot_q3_k_q8_k_avx2(row_bytes, act) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: as in `dot_q2_k_q8_k`, for Q3_K blocks.
            return unsafe { aarch64::dot_q3_k_q8_k_neon_sdot(row_bytes, act) };
        }
    }
    dot_q3_k_q8_k_scalar(row_bytes, act)
}

/// The Q2_K scalar twin, `ggml_vec_dot_q2_K_q8_K_generic`.
pub fn dot_q2_k_q8_k_scalar(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q2_K_BLOCK_BYTES, 0);
    debug_assert_eq!(row_bytes.len() / Q2_K_BLOCK_BYTES, act.n_blocks());
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q2_K_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let sc = &block[0..16];
        let qs = &block[16..80];
        let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
        let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();
        let q8 = &act.q[b * Q2_K_BLOCK_ELEMS..(b + 1) * Q2_K_BLOCK_ELEMS];
        let bsums = &act.bsums[b * 16..(b + 1) * 16];

        let summs: i32 = sc
            .iter()
            .zip(bsums)
            .map(|(&s, &bs)| (s >> 4) as i32 * bs as i32)
            .sum();
        let mut isum = 0i32;
        for half in 0..2 {
            let q2 = &qs[half * 32..half * 32 + 32];
            for plane in 0..4 {
                let shift = 2 * plane;
                let y = &q8[half * 128 + plane * 32..half * 128 + plane * 32 + 32];
                for sub in 0..2 {
                    let is = half * 8 + plane * 2 + sub;
                    let s: i32 = (0..16)
                        .map(|l| y[sub * 16 + l] as i32 * ((q2[sub * 16 + l] >> shift) & 3) as i32)
                        .sum();
                    isum += (sc[is] & 0xF) as i32 * s;
                }
            }
        }
        acc += act.d[b] * (d * isum as f32 - dmin * summs as f32);
    }
    acc
}

/// The Q3_K scalar twin, `ggml_vec_dot_q3_K_q8_K_generic` with the
/// element order of `dequant_q3_k`.
pub fn dot_q3_k_q8_k_scalar(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q3_K_BLOCK_BYTES, 0);
    debug_assert_eq!(row_bytes.len() / Q3_K_BLOCK_BYTES, act.n_blocks());
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q3_K_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let d = f16::from_le_bytes([block[108], block[109]]).to_f32();
        let scales = q3_k_scales(block);
        let q8 = &act.q[b * Q3_K_BLOCK_ELEMS..(b + 1) * Q3_K_BLOCK_ELEMS];

        let mut isum = 0i32;
        for half in 0..2 {
            let q3 = &qs[half * 32..half * 32 + 32];
            for plane in 0..4 {
                let shift = 2 * plane;
                let m = 1u8 << (half * 4 + plane);
                let y = &q8[half * 128 + plane * 32..half * 128 + plane * 32 + 32];
                for sub in 0..2 {
                    let is = half * 8 + plane * 2 + sub;
                    let s: i32 = (0..16)
                        .map(|l| {
                            let i = sub * 16 + l;
                            let low = ((q3[i] >> shift) & 3) as i32;
                            let bias = if hmask[i] & m != 0 { 0 } else { 4 };
                            y[i] as i32 * (low - bias)
                        })
                        .sum();
                    isum += scales[is] as i32 * s;
                }
            }
        }
        acc += d * act.d[b] * isum as f32;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    use super::*;
    use crate::repack::sdot;

    /// `ggml_vec_dot_q2_K_q8_K`'s NEON body: each plane masked out of the
    /// 32 `qs` bytes, SDOT per 16 lanes, the sub-block scale applied to
    /// the horizontal sum, the min term from `bsums`.
    ///
    /// # Safety
    /// The host has `dotprod`; `row_bytes` is whole Q2_K blocks and
    /// `act` has one Q8_K block per row block (asserted in debug).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q2_k_q8_k_neon_sdot(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q2_K_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q2_K_BLOCK_BYTES, act.n_blocks());
        let m3 = vdupq_n_u8(3);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q2_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
            let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();
            let sc = block.as_ptr();
            let qs = block.as_ptr().add(16);
            let q8 = act.q.as_ptr().add(b * Q2_K_BLOCK_ELEMS);

            // The min term: sixteen (min nibble x bsum) products.
            let mins = vshrq_n_u8(vld1q_u8(sc), 4);
            let bs = act.bsums.as_ptr().add(b * 16);
            let (bs0, bs1) = (vld1q_s16(bs), vld1q_s16(bs.add(8)));
            let mins0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
            let mins1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
            let s0 = vaddq_s32(
                vmull_s16(vget_low_s16(mins0), vget_low_s16(bs0)),
                vmull_s16(vget_high_s16(mins0), vget_high_s16(bs0)),
            );
            let s1 = vaddq_s32(
                vmull_s16(vget_low_s16(mins1), vget_low_s16(bs1)),
                vmull_s16(vget_high_s16(mins1), vget_high_s16(bs1)),
            );
            let summs = vaddvq_s32(vaddq_s32(s0, s1));

            let mut isum = 0i32;
            for half in 0..2 {
                let bits0 = vld1q_u8(qs.add(half * 32));
                let bits1 = vld1q_u8(qs.add(half * 32 + 16));
                for plane in 0..4 {
                    let shift = vdupq_n_s8(-2 * plane as i8);
                    let q2a = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(bits0, shift), m3));
                    let q2b = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(bits1, shift), m3));
                    let y = q8.add(half * 128 + plane * 32);
                    let is = half * 8 + plane * 2;
                    let pa = vaddvq_s32(sdot(vdupq_n_s32(0), q2a, vld1q_s8(y)));
                    let pb = vaddvq_s32(sdot(vdupq_n_s32(0), q2b, vld1q_s8(y.add(16))));
                    isum += pa * (*sc.add(is) & 0xF) as i32;
                    isum += pb * (*sc.add(is + 1) & 0xF) as i32;
                }
            }
            acc += act.d[b] * (d * isum as f32 - dmin * summs as f32);
        }
        acc
    }

    /// `ggml_vec_dot_q3_K_q8_K`'s NEON body: the 2-bit plane minus 4
    /// where its `hmask` bit is clear, SDOT per 16 lanes, the signed
    /// sub-block scale applied to the horizontal sum.
    ///
    /// # Safety
    /// As [`dot_q2_k_q8_k_neon_sdot`], for Q3_K blocks.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q3_k_q8_k_neon_sdot(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q3_K_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q3_K_BLOCK_BYTES, act.n_blocks());
        let m3 = vdupq_n_u8(3);
        let four = vdupq_n_u8(4);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q3_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[108], block[109]]).to_f32();
            let scales = q3_k_scales(block);
            let hm0 = vld1q_u8(block.as_ptr());
            let hm1 = vld1q_u8(block.as_ptr().add(16));
            let qs = block.as_ptr().add(32);
            let q8 = act.q.as_ptr().add(b * Q3_K_BLOCK_ELEMS);

            let mut isum = 0i32;
            for half in 0..2 {
                let bits0 = vld1q_u8(qs.add(half * 32));
                let bits1 = vld1q_u8(qs.add(half * 32 + 16));
                for plane in 0..4 {
                    let shift = vdupq_n_s8(-2 * plane as i8);
                    let m = vdupq_n_u8(1 << (half * 4 + plane));
                    // 4 where the high bit is clear, 0 where it is set.
                    let bias0 = vbicq_u8(four, vtstq_u8(hm0, m));
                    let bias1 = vbicq_u8(four, vtstq_u8(hm1, m));
                    let qa = vsubq_s8(
                        vreinterpretq_s8_u8(vandq_u8(vshlq_u8(bits0, shift), m3)),
                        vreinterpretq_s8_u8(bias0),
                    );
                    let qb = vsubq_s8(
                        vreinterpretq_s8_u8(vandq_u8(vshlq_u8(bits1, shift), m3)),
                        vreinterpretq_s8_u8(bias1),
                    );
                    let y = q8.add(half * 128 + plane * 32);
                    let is = half * 8 + plane * 2;
                    let pa = vaddvq_s32(sdot(vdupq_n_s32(0), qa, vld1q_s8(y)));
                    let pb = vaddvq_s32(sdot(vdupq_n_s32(0), qb, vld1q_s8(y.add(16))));
                    isum += pa * scales[is] as i32 + pb * scales[is + 1] as i32;
                }
            }
            acc += d * act.d[b] * isum as f32;
        }
        acc
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::*;

    /// ggml's `get_scale_shuffle_q3k`: plane `i` of a 128-element half
    /// takes sub-block scale `2i` in its low 16 lanes and `2i + 1` in
    /// its high 16, as `i16` pairs for `madd_epi16`.
    const SCALE_SHUFFLE: [[u8; 32]; 4] = {
        let mut t = [[0u8; 32]; 4];
        let mut i = 0;
        while i < 4 {
            let mut l = 0;
            while l < 16 {
                t[i][l] = (4 * i + l % 2) as u8;
                t[i][16 + l] = (4 * i + 2 + l % 2) as u8;
                l += 1;
            }
            i += 1;
        }
        t
    };

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shuffle(i: usize) -> __m256i {
        _mm256_loadu_si256(SCALE_SHUFFLE[i].as_ptr() as *const __m256i)
    }

    /// The eight `i16` scales of one half, in both 128-bit lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn half_scales(sc16: &[i16; 16], half: usize) -> __m256i {
        _mm256_broadcastsi128_si256(_mm_loadu_si128(
            sc16.as_ptr().add(half * 8) as *const __m128i
        ))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum(accum: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(accum, 1);
        let lo = _mm256_castps256_ps128(accum);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }

    /// `ggml_vec_dot_q2_K_q8_K`'s AVX2 body. The 2-bit codes are
    /// unsigned, so `maddubs` takes them directly.
    ///
    /// # Safety
    /// The host has AVX2 and FMA; `row_bytes` is whole Q2_K blocks and
    /// `act` has one Q8_K block per row block (asserted in debug).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q2_k_q8_k_avx2(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q2_K_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q2_K_BLOCK_BYTES, act.n_blocks());
        let m3 = _mm256_set1_epi8(3);
        let mut accum = _mm256_setzero_ps();
        for (b, block) in row_bytes
            .as_chunks::<Q2_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = act.d[b] * f16::from_le_bytes([block[80], block[81]]).to_f32();
            let dmin = -act.d[b] * f16::from_le_bytes([block[82], block[83]]).to_f32();
            let qs = block.as_ptr().add(16);
            let q8 = act.q.as_ptr().add(b * Q2_K_BLOCK_ELEMS);

            let mut sc16 = [0i16; 16];
            let mut mins16 = [0i16; 16];
            for j in 0..16 {
                sc16[j] = (block[j] & 0xF) as i16;
                mins16[j] = (block[j] >> 4) as i16;
            }
            let prod = _mm256_madd_epi16(
                _mm256_loadu_si256(mins16.as_ptr() as *const __m256i),
                _mm256_loadu_si256(act.bsums.as_ptr().add(b * 16) as *const __m256i),
            );
            accum = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), accum);

            let mut sumi = _mm256_setzero_si256();
            for half in 0..2 {
                let scales = half_scales(&sc16, half);
                let bits = _mm256_loadu_si256(qs.add(half * 32) as *const __m256i);
                let y = q8.add(half * 128);
                let q = [
                    _mm256_and_si256(bits, m3),
                    _mm256_and_si256(_mm256_srli_epi16(bits, 2), m3),
                    _mm256_and_si256(_mm256_srli_epi16(bits, 4), m3),
                    _mm256_and_si256(_mm256_srli_epi16(bits, 6), m3),
                ];
                for (plane, &qp) in q.iter().enumerate() {
                    let y8 = _mm256_loadu_si256(y.add(plane * 32) as *const __m256i);
                    let p16 = _mm256_maddubs_epi16(qp, y8);
                    let s = _mm256_shuffle_epi8(scales, shuffle(plane));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s, p16));
                }
            }
            accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        }
        hsum(accum)
    }

    /// `ggml_vec_dot_q3_K_q8_K`'s AVX2 body: `maddubs` on the unsigned
    /// 2-bit code and on the 4 owed where the `hmask` bit is clear, the
    /// second subtracted from the first, then the signed scales.
    ///
    /// # Safety
    /// As [`dot_q2_k_q8_k_avx2`], for Q3_K blocks.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q3_k_q8_k_avx2(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q3_K_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q3_K_BLOCK_BYTES, act.n_blocks());
        let m3 = _mm256_set1_epi8(3);
        let four = _mm256_set1_epi8(4);
        let zero = _mm256_setzero_si256();
        let mut accum = _mm256_setzero_ps();
        for (b, block) in row_bytes
            .as_chunks::<Q3_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = act.d[b] * f16::from_le_bytes([block[108], block[109]]).to_f32();
            let sc16 = q3_k_scales(block).map(|s| s as i16);
            let hbits = _mm256_loadu_si256(block.as_ptr() as *const __m256i);
            let qs = block.as_ptr().add(32);
            let q8 = act.q.as_ptr().add(b * Q3_K_BLOCK_ELEMS);

            let mut sumi = _mm256_setzero_si256();
            for half in 0..2 {
                let scales = half_scales(&sc16, half);
                let bits = _mm256_loadu_si256(qs.add(half * 32) as *const __m256i);
                let y = q8.add(half * 128);
                let q = [
                    _mm256_and_si256(bits, m3),
                    _mm256_and_si256(_mm256_srli_epi16(bits, 2), m3),
                    _mm256_and_si256(_mm256_srli_epi16(bits, 4), m3),
                    _mm256_and_si256(_mm256_srli_epi16(bits, 6), m3),
                ];
                for (plane, &ql) in q.iter().enumerate() {
                    let m = _mm256_set1_epi8((1u8 << (half * 4 + plane)) as i8);
                    let clear = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, m), zero);
                    let qh = _mm256_and_si256(clear, four);
                    let y8 = _mm256_loadu_si256(y.add(plane * 32) as *const __m256i);
                    let p16 = _mm256_sub_epi16(
                        _mm256_maddubs_epi16(ql, y8),
                        _mm256_maddubs_epi16(qh, y8),
                    );
                    let s = _mm256_shuffle_epi8(scales, shuffle(plane));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s, p16));
                }
            }
            accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        }
        hsum(accum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic bytes: an xorshift stream.
    fn stream(seed: u32) -> impl FnMut() -> u32 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        }
    }

    /// A Q2_K row: random scales, mins and codes, small f16 `d`/`dmin`.
    fn q2_row(n_blocks: usize, seed: u32) -> Vec<u8> {
        let mut next = stream(seed);
        let mut out = Vec::with_capacity(n_blocks * Q2_K_BLOCK_BYTES);
        for _ in 0..n_blocks {
            for _ in 0..80 {
                out.push(next() as u8);
            }
            let d = f16::from_f32(0.01 + (next() % 1000) as f32 / 5000.0);
            let dmin = f16::from_f32(0.005 + (next() % 1000) as f32 / 9000.0);
            out.extend_from_slice(&d.to_le_bytes());
            out.extend_from_slice(&dmin.to_le_bytes());
        }
        out
    }

    /// A Q3_K row: random hmask, codes and packed scales, small f16 `d`.
    fn q3_row(n_blocks: usize, seed: u32) -> Vec<u8> {
        let mut next = stream(seed);
        let mut out = Vec::with_capacity(n_blocks * Q3_K_BLOCK_BYTES);
        for _ in 0..n_blocks {
            for _ in 0..108 {
                out.push(next() as u8);
            }
            let d = f16::from_f32(0.01 + (next() % 1000) as f32 / 5000.0);
            out.extend_from_slice(&d.to_le_bytes());
        }
        out
    }

    /// Activations exact in Q8_K: `amax` is 1.0 and every value is a
    /// multiple of 1/127, so the int8 path and the f32 kernel differ only
    /// by f32 summation order.
    fn exact_acts(n: usize) -> Vec<f32> {
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 255) as f32 - 127.0) / 127.0)
            .collect();
        let act = crate::quantize_activations_q8_k(&x);
        for (i, q) in act.q.iter().enumerate() {
            let back = *q as f32 * act.d[i / Q2_K_BLOCK_ELEMS];
            assert!((back - x[i]).abs() < 1e-6, "activation {i} not exact");
        }
        x
    }

    fn lossy_acts(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect()
    }

    fn close(got: f32, want: f32, rel: f32, what: &str) {
        let tol = rel * want.abs().max(1.0) + 1e-4;
        assert!((got - want).abs() <= tol, "{what}: {got} vs {want}");
    }

    #[test]
    fn q2_scalar_agrees_with_the_f32_kernel_on_q8_exact_activations() {
        for (n_blocks, seed) in [(1usize, 7u32), (4, 11), (9, 23)] {
            let bytes = q2_row(n_blocks, seed);
            let x = exact_acts(n_blocks * Q2_K_BLOCK_ELEMS);
            let act = crate::quantize_activations_q8_k(&x);
            let want = crate::dot_q2_k_f32_scalar(&bytes, &x);
            let got = dot_q2_k_q8_k_scalar(&bytes, &act);
            close(got, want, 1e-5, &format!("q2 n_blocks={n_blocks}"));
        }
    }

    #[test]
    fn q3_scalar_agrees_with_the_f32_kernel_on_q8_exact_activations() {
        for (n_blocks, seed) in [(1usize, 5u32), (4, 13), (9, 29)] {
            let bytes = q3_row(n_blocks, seed);
            let x = exact_acts(n_blocks * Q3_K_BLOCK_ELEMS);
            let act = crate::quantize_activations_q8_k(&x);
            let want = crate::dot_q3_k_f32_scalar(&bytes, &x);
            let got = dot_q3_k_q8_k_scalar(&bytes, &act);
            close(got, want, 1e-5, &format!("q3 n_blocks={n_blocks}"));
        }
    }

    /// Every SIMD arm the host has, against the scalar twins, on
    /// activations with a real quantization step.
    #[test]
    fn the_simd_arms_match_the_scalar_twins() {
        for (n_blocks, seed) in [(1usize, 3u32), (3, 5), (8, 9)] {
            let n = n_blocks * Q2_K_BLOCK_ELEMS;
            let act = crate::quantize_activations_q8_k(&lossy_acts(n));
            let q2 = q2_row(n_blocks, seed);
            let q3 = q3_row(n_blocks, seed + 1);
            let want2 = dot_q2_k_q8_k_scalar(&q2, &act);
            let want3 = dot_q3_k_q8_k_scalar(&q3, &act);
            close(dot_q2_k_q8_k(&q2, &act), want2, 1e-5, "q2 dispatch");
            close(dot_q3_k_q8_k(&q3, &act), want3, 1e-5, "q3 dispatch");
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                // SAFETY: `dotprod` detected; whole blocks, one act block each.
                let (n2, n3) = unsafe {
                    (
                        aarch64::dot_q2_k_q8_k_neon_sdot(&q2, &act),
                        aarch64::dot_q3_k_q8_k_neon_sdot(&q3, &act),
                    )
                };
                close(n2, want2, 1e-5, "q2 neon");
                close(n3, want3, 1e-5, "q3 neon");
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2 and FMA detected; whole blocks, one act block each.
                let (a2, a3) = unsafe {
                    (
                        x86::dot_q2_k_q8_k_avx2(&q2, &act),
                        x86::dot_q3_k_q8_k_avx2(&q3, &act),
                    )
                };
                close(a2, want2, 1e-5, "q2 avx2");
                close(a3, want3, 1e-5, "q3 avx2");
            }
        }
    }

    /// The extreme codes: every Q3_K high bit clear (all values owe 4)
    /// and every Q2_K code 3, at full-range activations, so a sign or
    /// saturation slip in an arm cannot hide behind random data.
    #[test]
    fn the_simd_arms_match_the_scalar_twins_at_the_extremes() {
        let n_blocks = 2;
        let x: Vec<f32> = (0..n_blocks * Q2_K_BLOCK_ELEMS)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let act = crate::quantize_activations_q8_k(&x);
        let mut q2 = q2_row(n_blocks, 41);
        let mut q3 = q3_row(n_blocks, 43);
        for b in 0..n_blocks {
            q2[b * Q2_K_BLOCK_BYTES + 16..b * Q2_K_BLOCK_BYTES + 80].fill(0xFF);
            q3[b * Q3_K_BLOCK_BYTES..b * Q3_K_BLOCK_BYTES + 32].fill(0);
        }
        close(
            dot_q2_k_q8_k(&q2, &act),
            dot_q2_k_q8_k_scalar(&q2, &act),
            1e-5,
            "q2 extremes",
        );
        close(
            dot_q3_k_q8_k(&q3, &act),
            dot_q3_k_q8_k_scalar(&q3, &act),
            1e-5,
            "q3 extremes",
        );
    }
}
