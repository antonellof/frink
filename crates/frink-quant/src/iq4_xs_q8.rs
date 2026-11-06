//! **IQ4_XS x Q8_K**: the int8 dot of an IQ4_XS row against activations
//! quantized to Q8_K once per matmul, llama.cpp's
//! `ggml_vec_dot_iq4_xs_q8_K`.
//!
//! # Why it exists
//!
//! IQ4_XS had one CPU kernel here, `dot_iq4_xs_f32`: the codebook lookup
//! fused with an f32 FMA against f32 activations. That is the right
//! shape for one activation and the wrong one for a batch. The batched
//! matmul's fallback arm ran it once per (row, activation), which
//! decodes every row's nibbles `batch` times and multiplies in f32
//! where an int8 lane does four; measured on a rented Ryzen 9 3900X
//! (2026-09-15, `benchmarks/RESULTS.md`), Llama-3.2-1B IQ4_XS prefilled
//! at 56.5 tok/s against llama.cpp's 251, **4.45x**, while every K-quant
//! row on the same host sat at 1.0x to 1.4x -- because those kinds have
//! this kernel (`dot_q4_k_q8` and friends) and IQ4_XS did not.
//!
//! # What it computes
//!
//! Per 256-element super-block (`arch/arm/quants.c:4256-4316`,
//! `arch/x86/quants.c`, same arithmetic): for each of the eight
//! 32-element sub-blocks, `sumi = sum(codebook[nibble] * q8)` over the
//! sixteen low nibbles then the sixteen high nibbles of the sub-block's
//! 16 bytes, times the sub-block's 6-bit scale `ls - 32` (four low bits
//! in `scales_l`, two high bits in `scales_h`); the block contributes
//! `d * y.d * sum(ls_i * sumi_i)`. The element order is the one
//! `dequant_iq4_xs` defines -- sixteen lows, then sixteen highs, per 16
//! bytes -- and the scalar body below is written from that function so
//! the two cannot disagree about it.
//!
//! The scalar body is the twin every SIMD arm is checked against, and
//! the f32 kernel is a second reference: on activations that are
//! exactly representable in Q8_K the two agree to rounding.

use half::f16;

use crate::{Q8KActivations, IQ4_XS_BLOCK_BYTES, IQ4_XS_BLOCK_ELEMS, KVALUES_IQ4NL};

/// The 6-bit sub-block scale of sub-block `ib` (0..8), already minus 32.
#[inline]
fn sub_scale(scales_l: &[u8], scales_h: u16, ib: usize) -> i32 {
    let lo = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32;
    let hi = ((scales_h >> (2 * ib)) & 3) as i32;
    (lo | (hi << 4)) - 32
}

/// IQ4_XS row x Q8_K activations. Dispatches to the SIMD arm the host
/// has (AVX2 on x86, SDOT on aarch64) and falls back to the scalar
/// twin.
pub fn dot_iq4_xs_q8_k(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { x86::dot_iq4_xs_q8_k_avx2(row_bytes, act) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { aarch64::dot_iq4_xs_q8_k_neon_sdot(row_bytes, act) };
        }
    }
    dot_iq4_xs_q8_k_scalar(row_bytes, act)
}

/// The scalar twin. Same element order as `dequant_iq4_xs`.
pub fn dot_iq4_xs_q8_k_scalar(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    debug_assert_eq!(row_bytes.len() % IQ4_XS_BLOCK_BYTES, 0);
    debug_assert_eq!(row_bytes.len() / IQ4_XS_BLOCK_BYTES, act.n_blocks());
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<IQ4_XS_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];
        let q8 = &act.q[b * IQ4_XS_BLOCK_ELEMS..(b + 1) * IQ4_XS_BLOCK_ELEMS];
        let mut sumi = 0i32;
        for ib in 0..8 {
            let sub = &qs[ib * 16..ib * 16 + 16];
            let y = &q8[ib * 32..ib * 32 + 32];
            let mut s = 0i32;
            for (j, &byte) in sub.iter().enumerate() {
                s += KVALUES_IQ4NL[(byte & 0xf) as usize] as i32 * y[j] as i32;
                s += KVALUES_IQ4NL[(byte >> 4) as usize] as i32 * y[16 + j] as i32;
            }
            sumi += sub_scale(scales_l, scales_h, ib) * s;
        }
        acc += d * act.d[b] * sumi as f32;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    use super::*;
    use crate::repack::sdot;

    /// `ggml_vec_dot_iq4_xs_q8_K`'s NEON body: codebook lookup with
    /// `tbl`, SDOT per 16 lanes, the sub-block scale applied to the
    /// horizontal sum. Safety: `row_bytes` is whole IQ4_XS blocks and
    /// `act` has one Q8_K block per row block (asserted in debug).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_iq4_xs_q8_k_neon_sdot(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % IQ4_XS_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / IQ4_XS_BLOCK_BYTES, act.n_blocks());
        let low_mask = vdupq_n_u8(0x0F);
        let codebook = vld1q_s8(KVALUES_IQ4NL.as_ptr());
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<IQ4_XS_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let scales_h = u16::from_le_bytes([block[2], block[3]]);
            let scales_l = &block[4..8];
            let qs = block.as_ptr().add(8);
            let q8 = act.q.as_ptr().add(b * IQ4_XS_BLOCK_ELEMS);
            let mut sumi = 0i32;
            for ib in 0..8 {
                let bytes = vld1q_u8(qs.add(ib * 16));
                let lo = vqtbl1q_s8(codebook, vandq_u8(bytes, low_mask));
                let hi = vqtbl1q_s8(codebook, vshrq_n_u8(bytes, 4));
                let y_lo = vld1q_s8(q8.add(ib * 32));
                let y_hi = vld1q_s8(q8.add(ib * 32 + 16));
                let prod = sdot(sdot(vdupq_n_s32(0), lo, y_lo), hi, y_hi);
                sumi += vaddvq_s32(prod) * sub_scale(scales_l, scales_h, ib);
            }
            acc += d * act.d[b] * sumi as f32;
        }
        acc
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::*;

    /// `mul_add_epi8` from ggml's x86 quants: `maddubs` wants an
    /// unsigned left operand, so the codebook value's sign moves onto
    /// the activation.
    #[target_feature(enable = "avx2")]
    unsafe fn mul_add_epi8(x: __m256i, y: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(x, x);
        let sy = _mm256_sign_epi8(y, x);
        _mm256_maddubs_epi16(ax, sy)
    }

    /// `ggml_vec_dot_iq4_xs_q8_K`'s AVX2 body: two 16-byte sub-blocks per
    /// step, codebook through `pshufb`, the sub-block scales applied
    /// with `madd_epi16`. Safety: as the NEON arm.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_iq4_xs_q8_k_avx2(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % IQ4_XS_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / IQ4_XS_BLOCK_BYTES, act.n_blocks());
        let codebook = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let m4b = _mm_set1_epi8(0x0f);
        let mut accum = _mm256_setzero_ps();
        for (b, block) in row_bytes
            .as_chunks::<IQ4_XS_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let scales_h = u16::from_le_bytes([block[2], block[3]]);
            let scales_l = &block[4..8];
            let qs = block.as_ptr().add(8);
            let q8 = act.q.as_ptr().add(b * IQ4_XS_BLOCK_ELEMS);
            let mut sumi = _mm256_setzero_si256();
            for ib in (0..8).step_by(2) {
                let q4bits_1 = _mm_loadu_si128(qs.add(ib * 16) as *const __m128i);
                let q4bits_2 = _mm_loadu_si128(qs.add(ib * 16 + 16) as *const __m128i);
                let q8b_1 = _mm256_loadu_si256(q8.add(ib * 32) as *const __m256i);
                let q8b_2 = _mm256_loadu_si256(q8.add(ib * 32 + 32) as *const __m256i);
                // Low nibbles in the low 128 lanes, high nibbles in the
                // high lanes: the dequant order, sixteen lows then
                // sixteen highs, matches `q8`'s 32 consecutive values.
                let q4b_1 = _mm256_set_m128i(
                    _mm_shuffle_epi8(codebook, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                    _mm_shuffle_epi8(codebook, _mm_and_si128(q4bits_1, m4b)),
                );
                let q4b_2 = _mm256_set_m128i(
                    _mm_shuffle_epi8(codebook, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                    _mm_shuffle_epi8(codebook, _mm_and_si128(q4bits_2, m4b)),
                );
                let p16_1 = mul_add_epi8(q4b_1, q8b_1);
                let p16_2 = mul_add_epi8(q4b_2, q8b_2);
                let ls1 = sub_scale(scales_l, scales_h, ib) as i16;
                let ls2 = sub_scale(scales_l, scales_h, ib + 1) as i16;
                let p_1 = _mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1));
                let p_2 = _mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p_1, p_2));
            }
            accum = _mm256_fmadd_ps(
                _mm256_set1_ps(d * act.d[b]),
                _mm256_cvtepi32_ps(sumi),
                accum,
            );
        }
        // Horizontal sum of the eight lanes.
        let hi = _mm256_extractf128_ps(accum, 1);
        let lo = _mm256_castps256_ps128(accum);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic IQ4_XS row: random-looking nibbles and scales, an
    /// f16 block scale, `n_blocks` super-blocks.
    fn row(n_blocks: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut out = Vec::with_capacity(n_blocks * IQ4_XS_BLOCK_BYTES);
        for _ in 0..n_blocks {
            let d = f16::from_f32(0.01 + (next() % 1000) as f32 / 5000.0);
            out.extend_from_slice(&d.to_le_bytes());
            out.extend_from_slice(&(next() as u16).to_le_bytes());
            for _ in 0..4 {
                out.push(next() as u8);
            }
            for _ in 0..128 {
                out.push(next() as u8);
            }
        }
        out
    }

    /// The scalar twin against the f32 kernel on activations that are
    /// exact in Q8_K (small integers times a power of two): the only
    /// difference is then f32 summation order.
    #[test]
    fn scalar_agrees_with_the_f32_kernel_on_q8_exact_activations() {
        for (n_blocks, seed) in [(1usize, 7u32), (4, 11), (9, 23)] {
            let bytes = row(n_blocks, seed);
            let x: Vec<f32> = (0..n_blocks * IQ4_XS_BLOCK_ELEMS)
                .map(|i| ((i * 37 % 255) as f32 - 127.0) / 127.0)
                .collect();
            let act = crate::quantize_activations_q8_k(&x);
            // `d = amax/127` with amax exactly 1.0 makes every x an exact
            // multiple of the step; check the round trip so the
            // comparison below is between two exact evaluations.
            for (i, q) in act.q.iter().enumerate() {
                let back = *q as f32 * act.d[i / IQ4_XS_BLOCK_ELEMS];
                assert!(
                    (back - x[i]).abs() < 1e-6,
                    "activation {i} not exact in Q8_K"
                );
            }
            let want = crate::dot_iq4_xs_f32(&bytes, &x);
            let got = dot_iq4_xs_q8_k_scalar(&bytes, &act);
            let tol = 1e-5 * want.abs().max(1.0) + 1e-4;
            assert!(
                (got - want).abs() <= tol,
                "n_blocks={n_blocks}: q8 {got} vs f32 {want}"
            );
        }
    }

    /// Every SIMD arm the host has against the scalar twin, on
    /// activations with a real quantization step so the int8 path is
    /// exercised at full range.
    #[test]
    fn the_simd_arms_match_the_scalar_twin() {
        for (n_blocks, seed) in [(1usize, 3u32), (3, 5), (8, 9)] {
            let bytes = row(n_blocks, seed);
            let x: Vec<f32> = (0..n_blocks * IQ4_XS_BLOCK_ELEMS)
                .map(|i| ((i as f32) * 0.37).sin() * 3.0)
                .collect();
            let act = crate::quantize_activations_q8_k(&x);
            let want = dot_iq4_xs_q8_k_scalar(&bytes, &act);
            let got = dot_iq4_xs_q8_k(&bytes, &act);
            let tol = 1e-5 * want.abs().max(1.0);
            assert!((got - want).abs() <= tol, "dispatch {got} vs scalar {want}");
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                let neon = unsafe { aarch64::dot_iq4_xs_q8_k_neon_sdot(&bytes, &act) };
                assert!((neon - want).abs() <= tol, "neon {neon} vs scalar {want}");
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let avx2 = unsafe { x86::dot_iq4_xs_q8_k_avx2(&bytes, &act) };
                assert!((avx2 - want).abs() <= tol, "avx2 {avx2} vs scalar {want}");
            }
        }
    }

    /// The sub-block scale unpacking, against the spelling
    /// `dequant_iq4_xs` uses.
    #[test]
    fn sub_scales_unpack_as_the_dequantizer_unpacks_them() {
        let scales_l = [0x21u8, 0x43, 0x65, 0x87];
        let scales_h: u16 = 0b11_10_01_00_11_10_01_00;
        for ib in 0..8 {
            let want = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
                | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
            assert_eq!(sub_scale(&scales_l, scales_h, ib), want - 32, "ib={ib}");
        }
    }
}
