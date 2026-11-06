//! **IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S x Q8_K**: the int8 dots
//! of the codebook-grid formats against activations quantized to Q8_K
//! once per matmul, llama.cpp's `ggml_vec_dot_iq{2_xxs,2_xs,2_s,3_xxs,
//! 3_s}_q8_K` (`ggml-cpu/quants.c:906-1148`).
//!
//! # Why they exist
//!
//! The five had only scalar f32 kernels here, so the CPU column of
//! `docs/KERNELS.md` read `f32` for them, and they are what the
//! "Dynamic" low-bit GGUFs of large MoE models are made of. ggml
//! declares `vec_dot_type = Q8_K` for each (ADR 0009).
//!
//! # One body, five decoders
//!
//! ggml's five dots share one shape: a block's 256 values are
//! `grid * sign` small integers, each 16-element group carries an odd
//! integer scale `ls`, and the block contributes
//! `FACTOR * d * y.d * sum_g(ls_g * sum(w * q8))`, with `FACTOR` 1/8 for
//! the IQ2 kinds, 1/4 for IQ3_XXS and 1 for IQ3_S. So each kind is a
//! decoder ([`IqBlock::decode`]) to `w` and `ls`, written from ggml's
//! generic dot with the grids and sign tables in `iq_tables`, and the
//! dot is written once (ADR 0004). The decoders are checked element by
//! element against the dequantizers, which are pinned to ggml's bytes.

// `as_chunks::<B::BYTES>` would need a const generic from a trait's
// associated const, which stable Rust cannot express.
#![allow(clippy::chunks_exact_to_as_chunks)]

use half::f16;

use crate::iq_tables::{
    IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS,
};
use crate::{
    Q8KActivations, IQ2_S_BLOCK_BYTES, IQ2_XS_BLOCK_BYTES, IQ2_XXS_BLOCK_BYTES, IQ3_S_BLOCK_BYTES,
    IQ3_XXS_BLOCK_BYTES,
};

const QK_K: usize = 256;

/// A block's 256 signed values and its sixteen per-16-element scales.
struct Decoded {
    w: [i8; QK_K],
    ls: [i32; 16],
}

trait IqBlock {
    const BYTES: usize;
    /// ggml's final multiplier (`*s = FACTOR * sumf`).
    const FACTOR: f32;
    fn decode(block: &[u8]) -> Decoded;
}

#[inline]
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[inline]
fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

/// Byte `j` of a packed grid entry, signed by bit `j` of `signs`.
#[inline]
fn signed(grid: u64, j: usize, signs: u8, bit: usize) -> i8 {
    let mag = ((grid >> (8 * j)) & 0xFF) as i8;
    if signs & KMASK_IQ2XS[bit] != 0 {
        -mag
    } else {
        mag
    }
}

struct Iq2Xxs;
impl IqBlock for Iq2Xxs {
    const BYTES: usize = IQ2_XXS_BLOCK_BYTES;
    const FACTOR: f32 = 0.125;
    fn decode(block: &[u8]) -> Decoded {
        let mut out = Decoded {
            w: [0; QK_K],
            ls: [0; 16],
        };
        for ib32 in 0..8 {
            let o = 2 + 8 * ib32;
            let aux1 = u32_at(block, o + 4);
            let ls = 2 * (aux1 >> 28) as i32 + 1;
            out.ls[2 * ib32] = ls;
            out.ls[2 * ib32 + 1] = ls;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[block[o + l] as usize];
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    out.w[32 * ib32 + 8 * l + j] = signed(grid, j, signs, j);
                }
            }
        }
        out
    }
}

struct Iq2Xs;
impl IqBlock for Iq2Xs {
    const BYTES: usize = IQ2_XS_BLOCK_BYTES;
    const FACTOR: f32 = 0.125;
    fn decode(block: &[u8]) -> Decoded {
        let mut out = Decoded {
            w: [0; QK_K],
            ls: [0; 16],
        };
        for ib32 in 0..8 {
            let sc = block[66 + ib32];
            out.ls[2 * ib32] = 2 * (sc & 0xF) as i32 + 1;
            out.ls[2 * ib32 + 1] = 2 * (sc >> 4) as i32 + 1;
            for l in 0..4 {
                let q = u16_at(block, 2 + 2 * (4 * ib32 + l));
                let grid = IQ2XS_GRID[(q & 511) as usize];
                let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                for j in 0..8 {
                    out.w[32 * ib32 + 8 * l + j] = signed(grid, j, signs, j);
                }
            }
        }
        out
    }
}

struct Iq2S;
impl IqBlock for Iq2S {
    const BYTES: usize = IQ2_S_BLOCK_BYTES;
    const FACTOR: f32 = 0.125;
    fn decode(block: &[u8]) -> Decoded {
        let (qs, sign_bytes, qh, scales) = (
            &block[2..34],
            &block[34..66],
            &block[66..74],
            &block[74..82],
        );
        let mut out = Decoded {
            w: [0; QK_K],
            ls: [0; 16],
        };
        for ib32 in 0..8 {
            out.ls[2 * ib32] = 1 + 2 * (scales[ib32] & 0xF) as i32;
            out.ls[2 * ib32 + 1] = 1 + 2 * (scales[ib32] >> 4) as i32;
            for l in 0..4 {
                let idx =
                    qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[idx];
                let signs = sign_bytes[4 * ib32 + l];
                for j in 0..8 {
                    out.w[32 * ib32 + 8 * l + j] = signed(grid, j, signs, j);
                }
            }
        }
        out
    }
}

struct Iq3Xxs;
impl IqBlock for Iq3Xxs {
    const BYTES: usize = IQ3_XXS_BLOCK_BYTES;
    const FACTOR: f32 = 0.25;
    fn decode(block: &[u8]) -> Decoded {
        let (qs, gas) = (&block[2..66], &block[66..98]);
        let mut out = Decoded {
            w: [0; QK_K],
            ls: [0; 16],
        };
        for ib32 in 0..8 {
            let aux = u32_at(gas, 4 * ib32);
            let ls = 2 * (aux >> 28) as i32 + 1;
            out.ls[2 * ib32] = ls;
            out.ls[2 * ib32 + 1] = ls;
            let q3 = &qs[8 * ib32..8 * ib32 + 8];
            for l in 0..4 {
                let g1 = IQ3XXS_GRID[q3[2 * l] as usize] as u64;
                let g2 = IQ3XXS_GRID[q3[2 * l + 1] as usize] as u64;
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                for j in 0..4 {
                    out.w[32 * ib32 + 8 * l + j] = signed(g1, j, signs, j);
                    out.w[32 * ib32 + 8 * l + 4 + j] = signed(g2, j, signs, j + 4);
                }
            }
        }
        out
    }
}

struct Iq3S;
impl IqBlock for Iq3S {
    const BYTES: usize = IQ3_S_BLOCK_BYTES;
    const FACTOR: f32 = 1.0;
    fn decode(block: &[u8]) -> Decoded {
        let (qs, qh, sign_bytes, scales) = (
            &block[2..66],
            &block[66..74],
            &block[74..106],
            &block[106..110],
        );
        let mut out = Decoded {
            w: [0; QK_K],
            ls: [0; 16],
        };
        for ib32 in 0..8 {
            let sc = scales[ib32 / 2];
            let ls = if ib32 % 2 == 0 { sc & 0xF } else { sc >> 4 };
            let ls = 2 * ls as i32 + 1;
            out.ls[2 * ib32] = ls;
            out.ls[2 * ib32 + 1] = ls;
            let h = qh[ib32] as usize;
            for l in 0..4 {
                let i1 = qs[8 * ib32 + 2 * l] as usize | ((h << (8 - 2 * l)) & 256);
                let i2 = qs[8 * ib32 + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256);
                let (g1, g2) = (IQ3S_GRID[i1] as u64, IQ3S_GRID[i2] as u64);
                let signs = sign_bytes[4 * ib32 + l];
                for j in 0..4 {
                    out.w[32 * ib32 + 8 * l + j] = signed(g1, j, signs, j);
                    out.w[32 * ib32 + 8 * l + 4 + j] = signed(g2, j, signs, j + 4);
                }
            }
        }
        out
    }
}

/// The scalar twin every SIMD arm is checked against.
fn dot_scalar<B: IqBlock>(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    debug_assert_eq!(row_bytes.len() % B::BYTES, 0);
    debug_assert_eq!(row_bytes.len() / B::BYTES, act.n_blocks());
    let mut acc = 0f32;
    for (b, block) in row_bytes.chunks_exact(B::BYTES).enumerate() {
        let dec = B::decode(block);
        let y = &act.q[b * QK_K..(b + 1) * QK_K];
        let mut bsum = 0i32;
        for g in 0..16 {
            let s: i32 = (0..16)
                .map(|l| dec.w[16 * g + l] as i32 * y[16 * g + l] as i32)
                .sum();
            bsum += dec.ls[g] * s;
        }
        acc += f16::from_le_bytes([block[0], block[1]]).to_f32() * act.d[b] * bsum as f32;
    }
    B::FACTOR * acc
}

fn dot_dispatch<B: IqBlock>(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA detected; whole blocks, one Q8_K block
            // each in `act` (asserted in debug inside).
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
        #[doc = concat!($what, " row x Q8_K activations, SIMD where the host has it.")]
        pub fn $dot(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
            dot_dispatch::<$block>(row_bytes, act)
        }
        #[doc = concat!("The ", $what, " scalar twin.")]
        pub fn $scalar(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
            dot_scalar::<$block>(row_bytes, act)
        }
    };
}

kind_fns!(Iq2Xxs, dot_iq2_xxs_q8_k, dot_iq2_xxs_q8_k_scalar, "IQ2_XXS");
kind_fns!(Iq2Xs, dot_iq2_xs_q8_k, dot_iq2_xs_q8_k_scalar, "IQ2_XS");
kind_fns!(Iq2S, dot_iq2_s_q8_k, dot_iq2_s_q8_k_scalar, "IQ2_S");
kind_fns!(Iq3Xxs, dot_iq3_xxs_q8_k, dot_iq3_xxs_q8_k_scalar, "IQ3_XXS");
kind_fns!(Iq3S, dot_iq3_s_q8_k, dot_iq3_s_q8_k_scalar, "IQ3_S");

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    use super::*;
    use crate::repack::sdot;

    /// The decoded block against Q8_K, SDOT per 16-element group.
    ///
    /// # Safety
    /// The host has `dotprod`; `row_bytes` is whole `B` blocks and `act`
    /// has one Q8_K block per row block (asserted in debug).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_neon_sdot<B: IqBlock>(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % B::BYTES, 0);
        debug_assert_eq!(row_bytes.len() / B::BYTES, act.n_blocks());
        let mut acc = 0f32;
        for (b, block) in row_bytes.chunks_exact(B::BYTES).enumerate() {
            let dec = B::decode(block);
            let y = act.q.as_ptr().add(b * QK_K);
            let mut bsum = 0i32;
            for g in 0..16 {
                let w = vld1q_s8(dec.w.as_ptr().add(16 * g));
                let s = vaddvq_s32(sdot(vdupq_n_s32(0), w, vld1q_s8(y.add(16 * g))));
                bsum += dec.ls[g] * s;
            }
            acc += f16::from_le_bytes([block[0], block[1]]).to_f32() * act.d[b] * bsum as f32;
        }
        B::FACTOR * acc
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::*;

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

    /// The decoded block against Q8_K, 32 lanes per step: the weight's
    /// sign moves onto the activation for `maddubs`, and each 16-lane
    /// half takes its group's scale in `madd_epi16`. `|w| <= 62` and
    /// `|y| <= 127`, so a pair sum stays inside `i16`.
    ///
    /// # Safety
    /// The host has AVX2 and FMA; `row_bytes` is whole `B` blocks and
    /// `act` has one Q8_K block per row block (asserted in debug).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_avx2<B: IqBlock>(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % B::BYTES, 0);
        debug_assert_eq!(row_bytes.len() / B::BYTES, act.n_blocks());
        let mut accum = _mm256_setzero_ps();
        for (b, block) in row_bytes.chunks_exact(B::BYTES).enumerate() {
            let dec = B::decode(block);
            let y = act.q.as_ptr().add(b * QK_K);
            let mut sumi = _mm256_setzero_si256();
            for k in 0..8 {
                let w = _mm256_loadu_si256(dec.w.as_ptr().add(32 * k) as *const __m256i);
                let y8 = _mm256_loadu_si256(y.add(32 * k) as *const __m256i);
                let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(y8, w));
                let sc = _mm256_set_m128i(
                    _mm_set1_epi16(dec.ls[2 * k + 1] as i16),
                    _mm_set1_epi16(dec.ls[2 * k] as i16),
                );
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p16, sc));
            }
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32() * act.d[b];
            accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        }
        B::FACTOR * hsum(accum)
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

    /// Random payload with a small positive f16 `d` in bytes 0..2.
    fn row<B: IqBlock>(n_blocks: usize, seed: u32) -> Vec<u8> {
        let mut next = stream(seed);
        let mut out = Vec::with_capacity(n_blocks * B::BYTES);
        for _ in 0..n_blocks {
            let mut block: Vec<u8> = (0..B::BYTES).map(|_| next() as u8).collect();
            let d = f16::from_f32(0.01 + (next() % 1000) as f32 / 5000.0);
            block[0..2].copy_from_slice(&d.to_le_bytes());
            out.extend_from_slice(&block);
        }
        out
    }

    type Deq = fn(&[u8]) -> Result<Vec<f32>, crate::QuantError>;
    type F32Dot = fn(&[u8], &[f32]) -> f32;

    /// `FACTOR * d * ls * w` must be the value the dequantizer (pinned to
    /// ggml's bytes) writes, element for element: the grid index, the
    /// sign bit and the scale nibble are each checked directly.
    fn check_decode<B: IqBlock>(deq: Deq, what: &str) {
        let bytes = row::<B>(4, 31);
        let want = deq(&bytes).unwrap();
        for (b, block) in bytes.chunks_exact(B::BYTES).enumerate() {
            let dec = B::decode(block);
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            for e in 0..QK_K {
                let got = B::FACTOR * d * (dec.ls[e / 16] * dec.w[e] as i32) as f32;
                let exp = want[b * QK_K + e];
                let tol = 1e-6 * exp.abs().max(1.0);
                assert!(
                    (got - exp).abs() <= tol,
                    "{what} block {b} elem {e}: {got} vs {exp}"
                );
            }
        }
    }

    #[test]
    fn decode_is_the_dequantizer_element_for_element() {
        check_decode::<Iq2Xxs>(crate::dequant_iq2_xxs, "iq2_xxs");
        check_decode::<Iq2Xs>(crate::dequant_iq2_xs, "iq2_xs");
        check_decode::<Iq2S>(crate::dequant_iq2_s, "iq2_s");
        check_decode::<Iq3Xxs>(crate::dequant_iq3_xxs, "iq3_xxs");
        check_decode::<Iq3S>(crate::dequant_iq3_s, "iq3_s");
    }

    fn close(got: f32, want: f32, what: &str) {
        let tol = 1e-5 * want.abs().max(1.0) + 1e-4;
        assert!((got - want).abs() <= tol, "{what}: {got} vs {want}");
    }

    /// Against the f32 kernels on activations exact in Q8_K.
    fn check_f32<B: IqBlock>(f32_dot: F32Dot, what: &str) {
        for (n_blocks, seed) in [(1usize, 7u32), (3, 11)] {
            let bytes = row::<B>(n_blocks, seed);
            let x: Vec<f32> = (0..n_blocks * QK_K)
                .map(|i| ((i * 37 % 255) as f32 - 127.0) / 127.0)
                .collect();
            let act = crate::quantize_activations_q8_k(&x);
            close(dot_scalar::<B>(&bytes, &act), f32_dot(&bytes, &x), what);
        }
    }

    #[test]
    fn the_scalar_twins_agree_with_the_f32_kernels_on_q8_exact_activations() {
        check_f32::<Iq2Xxs>(crate::dot_iq2_xxs_f32_scalar, "iq2_xxs");
        check_f32::<Iq2Xs>(crate::dot_iq2_xs_f32_scalar, "iq2_xs");
        check_f32::<Iq2S>(crate::dot_iq2_s_f32_scalar, "iq2_s");
        check_f32::<Iq3Xxs>(crate::dot_iq3_xxs_f32_scalar, "iq3_xxs");
        check_f32::<Iq3S>(crate::dot_iq3_s_f32_scalar, "iq3_s");
    }

    fn check_simd<B: IqBlock>(what: &str) {
        for (n_blocks, seed) in [(1usize, 3u32), (4, 5)] {
            let x: Vec<f32> = (0..n_blocks * QK_K)
                .map(|i| ((i as f32) * 0.37).sin() * 3.0)
                .collect();
            let act = crate::quantize_activations_q8_k(&x);
            let bytes = row::<B>(n_blocks, seed);
            let want = dot_scalar::<B>(&bytes, &act);
            close(dot_dispatch::<B>(&bytes, &act), want, what);
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                // SAFETY: `dotprod` detected; whole blocks, one act block each.
                close(
                    unsafe { aarch64::dot_neon_sdot::<B>(&bytes, &act) },
                    want,
                    what,
                );
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2 and FMA detected; whole blocks, one act block each.
                close(unsafe { x86::dot_avx2::<B>(&bytes, &act) }, want, what);
            }
        }
    }

    #[test]
    fn the_simd_arms_match_the_scalar_twins() {
        check_simd::<Iq2Xxs>("iq2_xxs");
        check_simd::<Iq2Xs>("iq2_xs");
        check_simd::<Iq2S>("iq2_s");
        check_simd::<Iq3Xxs>("iq3_xxs");
        check_simd::<Iq3S>("iq3_s");
    }
}
