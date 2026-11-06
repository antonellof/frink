//! The base-3 trit formats: ggml's `TQ1_0` (tag 34, 256 weights per
//! block) and PrismML's `PTQ1_0` (tag 143, 128 per block, one f16 scale
//! per 128 instead of per 256), which is what Bonsai 2 27B ships in.
//! Same codec, two block geometries: a value in `{-1, 0, +1}` is stored
//! as a trit `{0, 1, 2}`, five trits to a byte in `qs` (`3^5 = 243 <
//! 256`) and four to a byte in the `qh` tail, the scale LAST in the
//! block (`block_ptq1_0 { qs[24]; qh[2]; half d }`, `block_tq1_0 {
//! qs[48]; qh[4]; half d }`).
//!
//! The packing order is the one llama.cpp's `dequantize_row_tq1_0` /
//! `dequantize_row_ptq1_0` walk (PrismML `ggml-quants.c:2255-2285`,
//! upstream's is the same loop at 256): `qs` is consumed in STAGES of
//! 32, 16 and 8 bytes -- as many whole stages of each width as fit,
//! largest first -- and within a stage of `c` bytes the five trits of
//! byte `j + m` are elements `n * c + m` for `n = 0..5`, most
//! significant trit first; then each `qh` byte holds four trits for
//! elements `n * qh_len + h`. A trit is read out as `((q * 3^n) as u16
//! * 3) >> 8`, which is the top base-3 digit of the byte after the
//! digits above it have wrapped away in u8 multiplication. That trick
//! is exact because the quantizer packs `q = ceil(digits * 256 / 243)`
//! (`quantize_row_ptq1_0_ref`), so `q * 3^n` never lands on a digit
//! boundary; it is transcribed here and pinned by a round trip through
//! the quantizer below.
//!
//! What this module does NOT do is guess the Hadamard rotation Bonsai
//! folds into these weights: that is activation-side and lives in
//! `frink-core` (`weight_matrix::hadamard`), and a PTQ1_0 tensor
//! without it is just a ternary matrix.

use half::f16;

/// One trit format's block geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TritLayout {
    /// Bytes of five-trit packing.
    pub qs_len: usize,
    /// Bytes of four-trit packing.
    pub qh_len: usize,
}

impl TritLayout {
    pub const fn block_elems(self) -> usize {
        self.qs_len * 5 + self.qh_len * 4
    }
    /// `qs + qh + f16 d`.
    pub const fn block_bytes(self) -> usize {
        self.qs_len + self.qh_len + 2
    }
}

/// ggml `TQ1_0`: `block_tq1_0 { qs[48]; qh[4]; half d }`, 256 weights.
pub const TQ1_0: TritLayout = TritLayout {
    qs_len: 48,
    qh_len: 4,
};
/// PrismML `PTQ1_0`: `block_ptq1_0 { qs[24]; qh[2]; half d }`, 128 weights.
pub const PTQ1_0: TritLayout = TritLayout {
    qs_len: 24,
    qh_len: 2,
};

pub const TQ1_0_BLOCK_BYTES: usize = TQ1_0.block_bytes();
pub const TQ1_0_BLOCK_ELEMS: usize = TQ1_0.block_elems();
pub const PTQ1_0_BLOCK_BYTES: usize = PTQ1_0.block_bytes();
pub const PTQ1_0_BLOCK_ELEMS: usize = PTQ1_0.block_elems();

const _: () = assert!(TQ1_0_BLOCK_BYTES == 54 && TQ1_0_BLOCK_ELEMS == 256);
const _: () = assert!(PTQ1_0_BLOCK_BYTES == 28 && PTQ1_0_BLOCK_ELEMS == 128);

/// The stage widths `qs` is consumed in (`ggml-quants.c`: `{32, 16, 8}`).
const STAGES: [usize; 3] = [32, 16, 8];
const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// Unpacks one block's trits to `out[..block_elems]` as `-1 | 0 | 1`.
/// The scalar transcription of the two dequantizers; every other
/// reader of a trit block (the dot products, the NEON unpack when it
/// lands) is held to this one.
#[inline]
pub fn unpack_trits(block: &[u8], layout: TritLayout, out: &mut [i8]) {
    debug_assert!(block.len() >= layout.block_bytes());
    debug_assert!(out.len() >= layout.block_elems());
    let qs = &block[..layout.qs_len];
    let qh = &block[layout.qs_len..layout.qs_len + layout.qh_len];
    let mut o = 0usize;
    let mut j = 0usize;
    for &c in &STAGES {
        while j + c <= layout.qs_len {
            for &p in &POW3 {
                for &b in &qs[j..j + c] {
                    let q = b.wrapping_mul(p);
                    let xi = ((q as u16) * 3) >> 8;
                    out[o] = xi as i8 - 1;
                    o += 1;
                }
            }
            j += c;
        }
    }
    for &p in &POW3[..4] {
        for &b in qh {
            let q = b.wrapping_mul(p);
            let xi = ((q as u16) * 3) >> 8;
            out[o] = xi as i8 - 1;
            o += 1;
        }
    }
    debug_assert_eq!(o, layout.block_elems());
}

/// The block's scale, the last two bytes.
#[inline]
pub fn block_scale(block: &[u8], layout: TritLayout) -> f32 {
    let at = layout.qs_len + layout.qh_len;
    f16::from_le_bytes([block[at], block[at + 1]]).to_f32()
}

/// Dequantizes a whole row (any number of blocks) to f32.
pub fn dequant_trits(row_bytes: &[u8], layout: TritLayout) -> Result<Vec<f32>, crate::QuantError> {
    let bb = layout.block_bytes();
    if !row_bytes.len().is_multiple_of(bb) {
        return Err(crate::QuantError::Misaligned(row_bytes.len(), bb));
    }
    let n = row_bytes.len() / bb;
    let mut out = Vec::with_capacity(n * layout.block_elems());
    let mut trits = [0i8; 256];
    for block in row_bytes.chunks_exact(bb) {
        let d = block_scale(block, layout);
        unpack_trits(block, layout, &mut trits);
        out.extend(trits[..layout.block_elems()].iter().map(|&t| t as f32 * d));
    }
    Ok(out)
}

/// `sum_i w_i * x_i` over one row against f32 activations, the scalar
/// twin every faster path is checked against.
pub fn dot_trits_f32(row_bytes: &[u8], layout: TritLayout, x: &[f32]) -> f32 {
    let bb = layout.block_bytes();
    let be = layout.block_elems();
    debug_assert_eq!(row_bytes.len() / bb * be, x.len());
    let mut trits = [0i8; 256];
    let mut acc = 0f32;
    for (block, xs) in row_bytes.chunks_exact(bb).zip(x.chunks_exact(be)) {
        unpack_trits(block, layout, &mut trits);
        // Ternary: the block's contribution is (sum of x where +1) minus
        // (sum of x where -1), times the scale. No multiply per element.
        let mut plus = 0f32;
        let mut minus = 0f32;
        for (&t, &v) in trits[..be].iter().zip(xs) {
            if t > 0 {
                plus += v;
            } else if t < 0 {
                minus += v;
            }
        }
        acc += block_scale(block, layout) * (plus - minus);
    }
    acc
}

/// `sum_i w_i * x_i` against Q8_0 activations (blocks of 32): the block
/// is `be / 32` activation blocks, each an integer dot of trits and
/// int8 times that sub-block's scale. Exact against the f32 twin up to
/// the activation quantization.
pub fn dot_trits_q8(row_bytes: &[u8], layout: TritLayout, act: &crate::Q8Activations) -> f32 {
    let bb = layout.block_bytes();
    let be = layout.block_elems();
    let sub = be / crate::Q8_0_BLOCK_ELEMS;
    debug_assert_eq!(row_bytes.len() / bb * sub, act.n_blocks());
    let mut trits = [0i8; 256];
    let mut acc = 0f32;
    for (bi, block) in row_bytes.chunks_exact(bb).enumerate() {
        unpack_trits(block, layout, &mut trits);
        let mut block_sum = 0f32;
        for s in 0..sub {
            let ab = bi * sub + s;
            let q = &act.q[ab * 32..(ab + 1) * 32];
            let t = &trits[s * 32..(s + 1) * 32];
            let isum: i32 = t.iter().zip(q).map(|(&t, &q)| t as i32 * q as i32).sum();
            block_sum += isum as f32 * act.d[ab];
        }
        acc += block_scale(block, layout) * block_sum;
    }
    acc
}

/// `quantize_row_ptq1_0_ref` / `quantize_row_tq1_0_ref`, transcribed:
/// the block scale is the max magnitude, each value rounds to a trit,
/// and the bytes are `ceil(digits * 256 / 243)`. Here for the tests
/// (a round trip pins the unpack order) and for anyone writing a
/// fixture; frink does not quantize models to these formats.
pub fn quantize_trits(x: &[f32], layout: TritLayout) -> Vec<u8> {
    let be = layout.block_elems();
    assert!(x.len().is_multiple_of(be));
    let mut out = Vec::with_capacity(x.len() / be * layout.block_bytes());
    for xs in x.chunks_exact(be) {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let id = if amax > 0.0 { 1.0 / amax } else { 0.0 };
        let trit = |v: f32| -> u8 { ((v * id).round() as i32 + 1) as u8 };
        let mut block = Vec::with_capacity(layout.block_bytes());
        let mut e = 0usize;
        let mut j = 0usize;
        for &c in &STAGES {
            while j + c <= layout.qs_len {
                for m in 0..c {
                    let mut q: u8 = 0;
                    for n in 0..5 {
                        q = q.wrapping_mul(3).wrapping_add(trit(xs[e + m + n * c]));
                    }
                    block.push(((q as u16) * 256).div_ceil(243) as u8);
                }
                e += 5 * c;
                j += c;
            }
        }
        for h in 0..layout.qh_len {
            let mut q: u8 = 0;
            for m in 0..4 {
                q = q
                    .wrapping_mul(3)
                    .wrapping_add(trit(xs[e + h + m * layout.qh_len]));
            }
            q = q.wrapping_mul(3);
            block.push(((q as u16) * 256).div_ceil(243) as u8);
        }
        block.extend_from_slice(&f16::from_f32(amax).to_le_bytes());
        debug_assert_eq!(block.len(), layout.block_bytes());
        out.extend(block);
    }
    out
}

/// PrismML `PTQ1_0` row x Q8_0 activations, `ggml_vec_dot_ptq1_0_q8_0`
/// (the format's `vec_dot_type` is `Q8_0`): the block's trits against
/// its four activation blocks, SDOT or AVX2 where the host has them and
/// [`dot_trits_q8`] (the scalar twin) otherwise.
pub fn dot_ptq1_0_q8(row_bytes: &[u8], act: &crate::Q8Activations) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA detected; `row_bytes` is whole PTQ1_0
            // blocks with four Q8_0 blocks each in `act` (asserted inside).
            return unsafe { simd::dot_ptq1_0_q8_avx2(row_bytes, act) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: as the x86 arm, for `dotprod`.
            return unsafe { simd::dot_ptq1_0_q8_neon_sdot(row_bytes, act) };
        }
    }
    dot_trits_q8(row_bytes, PTQ1_0, act)
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod simd {
    use super::*;

    const SUB: usize = PTQ1_0_BLOCK_ELEMS / crate::Q8_0_BLOCK_ELEMS;

    /// # Safety
    /// The host has `dotprod`; `row_bytes` is whole PTQ1_0 blocks and
    /// `act` has four Q8_0 blocks per row block (asserted in debug).
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_ptq1_0_q8_neon_sdot(row_bytes: &[u8], act: &crate::Q8Activations) -> f32 {
        use crate::repack::sdot;
        use std::arch::aarch64::*;
        debug_assert_eq!(row_bytes.len() / PTQ1_0_BLOCK_BYTES * SUB, act.n_blocks());
        let mut trits = [0i8; PTQ1_0_BLOCK_ELEMS];
        let mut acc = 0f32;
        for (bi, block) in row_bytes
            .as_chunks::<PTQ1_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            unpack_trits(block, PTQ1_0, &mut trits);
            let mut block_sum = 0f32;
            for s in 0..SUB {
                let ab = bi * SUB + s;
                let y = act.q.as_ptr().add(ab * 32);
                let t = trits.as_ptr().add(s * 32);
                let p = sdot(vdupq_n_s32(0), vld1q_s8(t), vld1q_s8(y));
                let isum = vaddvq_s32(sdot(p, vld1q_s8(t.add(16)), vld1q_s8(y.add(16))));
                block_sum += isum as f32 * act.d[ab];
            }
            acc += block_scale(block, PTQ1_0) * block_sum;
        }
        acc
    }

    /// # Safety
    /// The host has AVX2 and FMA; as the NEON arm otherwise.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_ptq1_0_q8_avx2(row_bytes: &[u8], act: &crate::Q8Activations) -> f32 {
        use std::arch::x86_64::*;
        debug_assert_eq!(row_bytes.len() / PTQ1_0_BLOCK_BYTES * SUB, act.n_blocks());
        let ones = _mm256_set1_epi16(1);
        let mut trits = [0i8; PTQ1_0_BLOCK_ELEMS];
        let mut acc = 0f32;
        for (bi, block) in row_bytes
            .as_chunks::<PTQ1_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            unpack_trits(block, PTQ1_0, &mut trits);
            let mut block_sum = 0f32;
            for s in 0..SUB {
                let ab = bi * SUB + s;
                let t = _mm256_loadu_si256(trits.as_ptr().add(s * 32) as *const __m256i);
                let y = _mm256_loadu_si256(act.q.as_ptr().add(ab * 32) as *const __m256i);
                // `maddubs` wants an unsigned left operand: the trit's
                // sign moves onto the activation.
                let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(t, t), _mm256_sign_epi8(y, t));
                let p32 = _mm256_madd_epi16(p16, ones);
                let lo = _mm256_castsi256_si128(p32);
                let hi = _mm256_extracti128_si256(p32, 1);
                let v = _mm_add_epi32(lo, hi);
                let v = _mm_hadd_epi32(v, v);
                let v = _mm_hadd_epi32(v, v);
                block_sum += _mm_cvtsi128_si32(v) as f32 * act.d[ab];
            }
            acc += block_scale(block, PTQ1_0) * block_sum;
        }
        acc
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;

    /// The SIMD arm against [`dot_trits_q8`], on random trit bytes and
    /// activations with a real quantization step. A wrong sub-block
    /// pairing or a dropped sign shows here.
    #[test]
    fn ptq1_0_simd_matches_the_scalar_twin() {
        let mut s = 0x9e37_79b9u32;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for n_blocks in [1usize, 3, 8] {
            let mut row = Vec::new();
            for _ in 0..n_blocks {
                let mut b: Vec<u8> = (0..PTQ1_0_BLOCK_BYTES).map(|_| next() as u8).collect();
                let d = f16::from_f32(0.02 + (next() % 100) as f32 / 1000.0);
                b[PTQ1_0_BLOCK_BYTES - 2..].copy_from_slice(&d.to_le_bytes());
                row.extend_from_slice(&b);
            }
            let x: Vec<f32> = (0..n_blocks * PTQ1_0_BLOCK_ELEMS)
                .map(|i| ((i as f32) * 0.41).sin() * 2.0)
                .collect();
            let act = crate::quantize_activations_q8(&x);
            let want = dot_trits_q8(&row, PTQ1_0, &act);
            let got = dot_ptq1_0_q8(&row, &act);
            let tol = 1e-5 * want.abs().max(1.0) + 1e-5;
            assert!(
                (got - want).abs() <= tol,
                "n_blocks={n_blocks}: {got} vs {want}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ternary_row(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                match (s >> 24) % 3 {
                    0 => -0.75,
                    1 => 0.0,
                    _ => 0.75,
                }
            })
            .collect()
    }

    #[test]
    fn the_two_layouts_are_the_structs_in_ggml_common_h() {
        assert_eq!((TQ1_0.block_bytes(), TQ1_0.block_elems()), (54, 256));
        assert_eq!((PTQ1_0.block_bytes(), PTQ1_0.block_elems()), (28, 128));
        assert_eq!(POW3, [1, 3, 9, 27, 81]);
    }

    /// A ternary row survives quantize -> unpack exactly, in both
    /// layouts, which pins the stage order and the qh tail together:
    /// any disagreement between the packer's and the unpacker's
    /// element order shows as a permuted row.
    #[test]
    fn a_ternary_row_round_trips_exactly_in_both_layouts() {
        for (layout, n) in [(PTQ1_0, 128 * 3), (TQ1_0, 256 * 2)] {
            let x = ternary_row(n, 7);
            let packed = quantize_trits(&x, layout);
            assert_eq!(
                packed.len(),
                n / layout.block_elems() * layout.block_bytes()
            );
            let back = dequant_trits(&packed, layout).unwrap();
            assert_eq!(back, x, "{layout:?}");
        }
    }

    /// The scale is the LAST two bytes and every packed byte is below
    /// 256 * 243 / 243: a hand-built block with known trits decodes to
    /// them in order.
    #[test]
    fn a_hand_packed_ptq1_0_block_decodes_in_stage_order() {
        // Element e = (e % 3) - 1 pattern, quantized, then read back
        // with the twin unpack and compared to the definition.
        let x: Vec<f32> = (0..128).map(|e| (e % 3) as f32 - 1.0).collect();
        let packed = quantize_trits(&x, PTQ1_0);
        let mut trits = [0i8; 256];
        unpack_trits(&packed, PTQ1_0, &mut trits);
        for (e, &t) in trits[..128].iter().enumerate() {
            assert_eq!(t as i32, (e % 3) as i32 - 1, "element {e}");
        }
        assert_eq!(block_scale(&packed, PTQ1_0), 1.0);
        // The first byte holds elements 0, 16, 32, 48, 64 (stage 16, m = 0).
        let digits = [x[0], x[16], x[32], x[48], x[64]]
            .iter()
            .fold(0u32, |q, v| q * 3 + (*v as i32 + 1) as u32);
        assert_eq!(packed[0] as u32, (digits * 256).div_ceil(243));
    }

    #[test]
    fn the_dots_agree_with_a_dequantized_f32_dot() {
        for (layout, n) in [(PTQ1_0, 128 * 4), (TQ1_0, 256 * 2)] {
            let w = ternary_row(n, 3);
            let packed = quantize_trits(&w, layout);
            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin()).collect();
            let want: f32 = dequant_trits(&packed, layout)
                .unwrap()
                .iter()
                .zip(&x)
                .map(|(a, b)| a * b)
                .sum();
            let got = dot_trits_f32(&packed, layout, &x);
            assert!((got - want).abs() < 1e-4, "{layout:?}: {got} vs {want}");
            let act = crate::quantize_activations_q8(&x);
            let got_q8 = dot_trits_q8(&packed, layout, &act);
            assert!(
                (got_q8 - want).abs() < 2e-2 * want.abs().max(1.0),
                "{layout:?}: q8 {got_q8} vs {want}"
            );
        }
    }
}
