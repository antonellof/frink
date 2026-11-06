//! **Q4_1, Q5_0, Q5_1 and IQ4_NL x Q8 activations**: the int8 dots of a
//! 32-element-block row against activations quantized once per matmul,
//! llama.cpp's `ggml_vec_dot_q4_1_q8_1`, `ggml_vec_dot_q5_0_q8_0`,
//! `ggml_vec_dot_q5_1_q8_1` and `ggml_vec_dot_iq4_nl_q8_0`.
//!
//! # Why they exist
//!
//! All four had only fused f32 kernels here, so the CPU column of
//! `docs/KERNELS.md` read `f32` for them. ggml declares `vec_dot_type`
//! `Q8_0` for Q5_0 and IQ4_NL and `Q8_1` for Q4_1 and Q5_1
//! (`ggml-cpu.c`, `type_traits_cpu`): llama.cpp quantizes the activation
//! to 8 bits per 32-element block and dots in int8 (ADR 0009).
//!
//! # One body, four decoders
//!
//! Each kind is a 32-element block that decodes to 32 small integers
//! `w` (element `j < 16` from the low nibble of byte `j`, `j >= 16` from
//! the high nibble of byte `j - 16`, the order `dequant_*` defines),
//! one f16 scale `d`, and for the `_1` kinds an f16 offset `m`:
//!
//! - **Q4_1**: `w = nibble` (0..=15), value `d*w + m`.
//! - **Q5_0**: `w = (nibble | fifth << 4) - 16` (-16..=15), value `d*w`.
//! - **Q5_1**: `w = nibble | fifth << 4` (0..=31), value `d*w + m`.
//! - **IQ4_NL**: `w = kvalues_iq4nl[nibble]` (-127..=113), value `d*w`.
//!
//! So a block contributes `y.d * (d * sum(w * q8) + m * sum(q8))`. The
//! `m` term is what ggml's `Q8_1` carries precomputed as `s`; here it is
//! the block's int8 sum times the f32 scale, which is the same number
//! before ggml rounds `s` to f16. Only the decoder differs per kind
//! ([`Block`]); the dot and the min term are written once (ADR 0004).
//!
//! The scalar body is the twin every SIMD arm is checked against, and
//! the f32 kernels are a second reference.

// `as_chunks::<B::BYTES>` would need a const generic from a trait's
// associated const, which stable Rust cannot express.
#![allow(clippy::chunks_exact_to_as_chunks)]

use half::f16;

use crate::{
    q5_fifth_bits, Q8Activations, IQ4_NL_BLOCK_BYTES, KVALUES_IQ4NL, Q4_1_BLOCK_BYTES,
    Q5_0_BLOCK_BYTES, Q5_1_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};

/// One 32-element block format: where its scale, offset and codes live,
/// and how its codes become 32 signed integers.
trait Block {
    const BYTES: usize;
    /// Byte offset of the sixteen nibble bytes.
    const QS: usize;
    /// Byte offset of the f16 offset `m`, for the `_1` kinds.
    const MIN: Option<usize>;
    /// Byte offset of the 32-bit fifth-bit plane, for the Q5 kinds.
    const QH: Option<usize>;
    /// Subtracted from every decoded code (16 for Q5_0).
    const BIAS: i8;
    /// Codes through the IQ4_NL codebook instead of linear.
    const CODEBOOK: bool;

    /// The 32 decoded integers of one block, in element order.
    fn decode(block: &[u8]) -> [i8; 32] {
        let qs = &block[Self::QS..Self::QS + 16];
        let qh = Self::QH.map(|o| u32::from_le_bytes(block[o..o + 4].try_into().unwrap()));
        let mut w = [0i8; 32];
        for j in 0..16 {
            let (mut lo, mut hi) = (qs[j] & 0xF, qs[j] >> 4);
            if let Some(qh) = qh {
                let (xh0, xh1) = q5_fifth_bits(qh, j);
                lo |= xh0;
                hi |= xh1;
            }
            if Self::CODEBOOK {
                w[j] = KVALUES_IQ4NL[lo as usize];
                w[j + 16] = KVALUES_IQ4NL[hi as usize];
            } else {
                w[j] = lo as i8 - Self::BIAS;
                w[j + 16] = hi as i8 - Self::BIAS;
            }
        }
        w
    }
}

struct Q4_1;
impl Block for Q4_1 {
    const BYTES: usize = Q4_1_BLOCK_BYTES;
    const QS: usize = 4;
    const MIN: Option<usize> = Some(2);
    const QH: Option<usize> = None;
    const BIAS: i8 = 0;
    const CODEBOOK: bool = false;
}

struct Q5_0;
impl Block for Q5_0 {
    const BYTES: usize = Q5_0_BLOCK_BYTES;
    const QS: usize = 6;
    const MIN: Option<usize> = None;
    const QH: Option<usize> = Some(2);
    const BIAS: i8 = 16;
    const CODEBOOK: bool = false;
}

struct Q5_1;
impl Block for Q5_1 {
    const BYTES: usize = Q5_1_BLOCK_BYTES;
    const QS: usize = 8;
    const MIN: Option<usize> = Some(2);
    const QH: Option<usize> = Some(4);
    const BIAS: i8 = 0;
    const CODEBOOK: bool = false;
}

struct Iq4Nl;
impl Block for Iq4Nl {
    const BYTES: usize = IQ4_NL_BLOCK_BYTES;
    const QS: usize = 2;
    const MIN: Option<usize> = None;
    const QH: Option<usize> = None;
    const BIAS: i8 = 0;
    const CODEBOOK: bool = true;
}

#[inline]
fn f16_at(block: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([block[o], block[o + 1]]).to_f32()
}

/// The scalar twin every SIMD arm is checked against.
fn dot_scalar<B: Block>(row_bytes: &[u8], act: &Q8Activations) -> f32 {
    debug_assert_eq!(row_bytes.len() % B::BYTES, 0);
    debug_assert_eq!(row_bytes.len() / B::BYTES, act.n_blocks());
    let mut acc = 0f32;
    for (b, block) in row_bytes.chunks_exact(B::BYTES).enumerate() {
        let w = B::decode(block);
        let y = &act.q[b * Q8_0_BLOCK_ELEMS..(b + 1) * Q8_0_BLOCK_ELEMS];
        let isum: i32 = w.iter().zip(y).map(|(&w, &y)| w as i32 * y as i32).sum();
        let mut block_acc = f16_at(block, 0) * isum as f32;
        if let Some(o) = B::MIN {
            let ysum: i32 = y.iter().map(|&y| y as i32).sum();
            block_acc += f16_at(block, o) * ysum as f32;
        }
        acc += act.d[b] * block_acc;
    }
    acc
}

/// Picks the SIMD arm the host has, else the scalar twin.
fn dot_dispatch<B: Block>(row_bytes: &[u8], act: &Q8Activations) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA detected; `row_bytes` is whole blocks
            // with one Q8 block each in `act` (asserted in debug inside).
            return unsafe { x86::dot_avx2::<B>(row_bytes, act) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: as the x86 arm, for `dotprod`.
            return unsafe { aarch64::dot_neon_sdot::<B>(row_bytes, act) };
        }
    }
    dot_scalar::<B>(row_bytes, act)
}

macro_rules! kind_fns {
    ($block:ty, $dot:ident, $scalar:ident, $what:literal) => {
        #[doc = concat!($what, " row x Q8 activations, SIMD where the host has it.")]
        pub fn $dot(row_bytes: &[u8], act: &Q8Activations) -> f32 {
            dot_dispatch::<$block>(row_bytes, act)
        }
        #[doc = concat!("The ", $what, " scalar twin.")]
        pub fn $scalar(row_bytes: &[u8], act: &Q8Activations) -> f32 {
            dot_scalar::<$block>(row_bytes, act)
        }
    };
}

kind_fns!(Q4_1, dot_q4_1_q8, dot_q4_1_q8_scalar, "Q4_1");
kind_fns!(Q5_0, dot_q5_0_q8, dot_q5_0_q8_scalar, "Q5_0");
kind_fns!(Q5_1, dot_q5_1_q8, dot_q5_1_q8_scalar, "Q5_1");
kind_fns!(Iq4Nl, dot_iq4_nl_q8, dot_iq4_nl_q8_scalar, "IQ4_NL");

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    use super::*;
    use crate::repack::sdot;

    /// One block's codes as two 16-lane vectors (elements 0..16, 16..32).
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn decode<B: Block>(block: *const u8) -> (int8x16_t, int8x16_t) {
        let bytes = vld1q_u8(block.add(B::QS));
        let mut lo = vandq_u8(bytes, vdupq_n_u8(0xF));
        let mut hi = vshrq_n_u8(bytes, 4);
        if let Some(o) = B::QH {
            // Lane j of the low half reads bit j of qh, lane j of the high
            // half bit 16 + j; each set bit becomes 0x10.
            const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let bit = vld1q_u8(BIT.as_ptr());
            let qh = block.add(o);
            let b01 = vcombine_u8(vdup_n_u8(*qh), vdup_n_u8(*qh.add(1)));
            let b23 = vcombine_u8(vdup_n_u8(*qh.add(2)), vdup_n_u8(*qh.add(3)));
            let fifth = vdupq_n_u8(0x10);
            lo = vorrq_u8(lo, vandq_u8(vtstq_u8(b01, bit), fifth));
            hi = vorrq_u8(hi, vandq_u8(vtstq_u8(b23, bit), fifth));
        }
        if B::CODEBOOK {
            let codebook = vld1q_s8(KVALUES_IQ4NL.as_ptr());
            return (vqtbl1q_s8(codebook, lo), vqtbl1q_s8(codebook, hi));
        }
        let bias = vdupq_n_s8(B::BIAS);
        (
            vsubq_s8(vreinterpretq_s8_u8(lo), bias),
            vsubq_s8(vreinterpretq_s8_u8(hi), bias),
        )
    }

    /// SDOT per 16 lanes; the min term from the block's int8 sum.
    ///
    /// # Safety
    /// The host has `dotprod`; `row_bytes` is whole `B` blocks and `act`
    /// has one Q8 block per row block (asserted in debug).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_neon_sdot<B: Block>(row_bytes: &[u8], act: &Q8Activations) -> f32 {
        debug_assert_eq!(row_bytes.len() % B::BYTES, 0);
        debug_assert_eq!(row_bytes.len() / B::BYTES, act.n_blocks());
        let mut acc = 0f32;
        for (b, block) in row_bytes.chunks_exact(B::BYTES).enumerate() {
            let (w_lo, w_hi) = decode::<B>(block.as_ptr());
            let y = act.q.as_ptr().add(b * Q8_0_BLOCK_ELEMS);
            let (y_lo, y_hi) = (vld1q_s8(y), vld1q_s8(y.add(16)));
            let isum = vaddvq_s32(sdot(sdot(vdupq_n_s32(0), w_lo, y_lo), w_hi, y_hi));
            let mut block_acc = f16_at(block, 0) * isum as f32;
            if let Some(o) = B::MIN {
                let ysum = vaddlvq_s8(y_lo) as i32 + vaddlvq_s8(y_hi) as i32;
                block_acc += f16_at(block, o) * ysum as f32;
            }
            acc += act.d[b] * block_acc;
        }
        acc
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::*;

    /// ggml's `bytes_from_bits_32`: lane j is 0xFF where bit j of the
    /// 32-bit word at `p` is set.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn bytes_from_bits_32(p: *const u8) -> __m256i {
        let x32 = u32::from_le_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]) as i32;
        let shuf = _mm256_set_epi64x(
            0x0303030303030303,
            0x0202020202020202,
            0x0101010101010101,
            0x0000000000000000,
        );
        let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf);
        let bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
        _mm256_cmpeq_epi8(_mm256_or_si256(bytes, bit_mask), _mm256_set1_epi64x(-1))
    }

    /// One block's 32 codes, element order, as signed bytes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn decode<B: Block>(block: *const u8) -> __m256i {
        let bytes = _mm_loadu_si128(block.add(B::QS) as *const __m128i);
        let m4 = _mm_set1_epi8(0xF);
        let mut lo = _mm_and_si128(bytes, m4);
        let mut hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), m4);
        if B::CODEBOOK {
            let codebook = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
            lo = _mm_shuffle_epi8(codebook, lo);
            hi = _mm_shuffle_epi8(codebook, hi);
        }
        let mut w = _mm256_set_m128i(hi, lo);
        if let Some(o) = B::QH {
            let fifth = _mm256_and_si256(bytes_from_bits_32(block.add(o)), _mm256_set1_epi8(0x10));
            w = _mm256_or_si256(w, fifth);
        }
        _mm256_sub_epi8(w, _mm256_set1_epi8(B::BIAS))
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

    /// `maddubs` wants an unsigned left operand, so the weight's sign
    /// moves onto the activation (ggml's `mul_sum_i8_pairs`). `|w| <=
    /// 127` and `|y| <= 127`, so a pair sum stays inside `i16`.
    ///
    /// # Safety
    /// The host has AVX2 and FMA; `row_bytes` is whole `B` blocks and
    /// `act` has one Q8 block per row block (asserted in debug).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_avx2<B: Block>(row_bytes: &[u8], act: &Q8Activations) -> f32 {
        debug_assert_eq!(row_bytes.len() % B::BYTES, 0);
        debug_assert_eq!(row_bytes.len() / B::BYTES, act.n_blocks());
        let ones16 = _mm256_set1_epi16(1);
        let ones8 = _mm256_set1_epi8(1);
        let mut accum = _mm256_setzero_ps();
        for (b, block) in row_bytes.chunks_exact(B::BYTES).enumerate() {
            let w = decode::<B>(block.as_ptr());
            let y = _mm256_loadu_si256(act.q.as_ptr().add(b * Q8_0_BLOCK_ELEMS) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(y, w));
            let p32 = _mm256_madd_epi16(p16, ones16);
            let da = act.d[b];
            let dw = _mm256_set1_ps(da * f16_at(block, 0));
            accum = _mm256_fmadd_ps(dw, _mm256_cvtepi32_ps(p32), accum);
            if let Some(o) = B::MIN {
                let ysum = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, y), ones16);
                let m = _mm256_set1_ps(da * f16_at(block, o));
                accum = _mm256_fmadd_ps(m, _mm256_cvtepi32_ps(ysum), accum);
            }
        }
        hsum(accum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(seed: u32) -> impl FnMut() -> u32 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        }
    }

    /// Random code bytes with small f16 scale (and offset) fields.
    fn row<B: Block>(n_blocks: usize, seed: u32) -> Vec<u8> {
        let mut next = stream(seed);
        let mut out = Vec::with_capacity(n_blocks * B::BYTES);
        for _ in 0..n_blocks {
            let mut block: Vec<u8> = (0..B::BYTES).map(|_| next() as u8).collect();
            let d = f16::from_f32(0.01 + (next() % 1000) as f32 / 5000.0);
            block[0..2].copy_from_slice(&d.to_le_bytes());
            if let Some(o) = B::MIN {
                let m = f16::from_f32(-0.05 + (next() % 1000) as f32 / 9000.0);
                block[o..o + 2].copy_from_slice(&m.to_le_bytes());
            }
            out.extend_from_slice(&block);
        }
        out
    }

    /// Activations exact in Q8 (32-blocks, `amax` 1.0, multiples of
    /// 1/127), so the int8 path and the f32 kernel differ only by f32
    /// summation order.
    fn exact_acts(n: usize) -> Vec<f32> {
        let x: Vec<f32> = (0..n)
            .map(|i| {
                if i % 32 == 0 {
                    1.0
                } else {
                    ((i * 37 % 255) as f32 - 127.0) / 127.0
                }
            })
            .collect();
        let act = crate::quantize_activations_q8(&x);
        for (i, q) in act.q.iter().enumerate() {
            let back = *q as f32 * act.d[i / 32];
            assert!((back - x[i]).abs() < 1e-6, "activation {i} not exact");
        }
        x
    }

    fn close(got: f32, want: f32, what: &str) {
        let tol = 1e-5 * want.abs().max(1.0) + 1e-4;
        assert!((got - want).abs() <= tol, "{what}: {got} vs {want}");
    }

    type F32Dot = fn(&[u8], &[f32]) -> f32;

    fn check_against_f32<B: Block>(f32_dot: F32Dot, what: &str) {
        for (n_blocks, seed) in [(1usize, 7u32), (4, 11), (9, 23)] {
            let bytes = row::<B>(n_blocks, seed);
            let x = exact_acts(n_blocks * 32);
            let act = crate::quantize_activations_q8(&x);
            close(
                dot_scalar::<B>(&bytes, &act),
                f32_dot(&bytes, &x),
                &format!("{what} n_blocks={n_blocks}"),
            );
        }
    }

    #[test]
    fn the_scalar_twins_agree_with_the_f32_kernels_on_q8_exact_activations() {
        check_against_f32::<Q4_1>(crate::dot_q4_1_f32_scalar, "q4_1");
        check_against_f32::<Q5_0>(crate::dot_q5_0_f32_scalar, "q5_0");
        check_against_f32::<Q5_1>(crate::dot_q5_1_f32_scalar, "q5_1");
        check_against_f32::<Iq4Nl>(crate::dot_iq4_nl_f32_scalar, "iq4_nl");
    }

    fn check_simd<B: Block>(what: &str) {
        for (n_blocks, seed) in [(1usize, 3u32), (3, 5), (8, 9)] {
            let x: Vec<f32> = (0..n_blocks * 32)
                .map(|i| ((i as f32) * 0.37).sin() * 3.0)
                .collect();
            let act = crate::quantize_activations_q8(&x);
            let bytes = row::<B>(n_blocks, seed);
            let want = dot_scalar::<B>(&bytes, &act);
            close(
                dot_dispatch::<B>(&bytes, &act),
                want,
                &format!("{what} dispatch"),
            );
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                // SAFETY: `dotprod` detected; whole blocks, one act block each.
                let got = unsafe { aarch64::dot_neon_sdot::<B>(&bytes, &act) };
                close(got, want, &format!("{what} neon"));
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2 and FMA detected; whole blocks, one act block each.
                let got = unsafe { x86::dot_avx2::<B>(&bytes, &act) };
                close(got, want, &format!("{what} avx2"));
            }
        }
    }

    #[test]
    fn the_simd_arms_match_the_scalar_twins() {
        check_simd::<Q4_1>("q4_1");
        check_simd::<Q5_0>("q5_0");
        check_simd::<Q5_1>("q5_1");
        check_simd::<Iq4Nl>("iq4_nl");
    }

    /// The decoder against the dequantizers, element by element: `d *
    /// w + m` must be the value `dequant_*` writes, so the element order
    /// and the fifth-bit plane are checked directly, not through a sum.
    #[test]
    fn decode_is_the_dequantizer_element_for_element() {
        type Deq = fn(&[u8]) -> Result<Vec<f32>, crate::QuantError>;
        fn check<B: Block>(deq: Deq, what: &str) {
            let bytes = row::<B>(3, 31);
            let want = deq(&bytes).unwrap();
            for (b, block) in bytes.chunks_exact(B::BYTES).enumerate() {
                let w = B::decode(block);
                let d = f16_at(block, 0);
                let m = B::MIN.map_or(0.0, |o| f16_at(block, o));
                for j in 0..32 {
                    let got = d * w[j] as f32 + m;
                    let exp = want[b * 32 + j];
                    assert!((got - exp).abs() <= 1e-6, "{what} block {b} elem {j}");
                }
            }
        }
        check::<Q4_1>(crate::dequant_q4_1, "q4_1");
        check::<Q5_0>(crate::dequant_q5_0, "q5_0");
        check::<Q5_1>(crate::dequant_q5_1, "q5_1");
        check::<Iq4Nl>(crate::dequant_iq4_nl, "iq4_nl");
    }
}
