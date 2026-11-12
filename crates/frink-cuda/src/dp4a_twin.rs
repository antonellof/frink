//! Scalar twins of the q8_1 quantizer and the dp4a matvecs in
//! [`crate::matvec_kinds::dp4a`].
//!
//! Each twin performs the kernel's float operations in the kernel's
//! order (one f32 accumulator per thread, walked block by block; warp
//! 0 adding the other warps' partials in warp order; the xor butterfly
//! across its lanes), and its integer operations exactly. With FP
//! contraction off that is bit for bit what the CUDA C computes, which
//! `tools/matvec_host_check` checks by running the kernel text on a
//! host CPU. A GPU contracts into FMAs, so the hardware test holds the
//! device to the twins with a relative tolerance instead.
//!
//! The twins read the block layouts their own way where an independent
//! reading is cheap, so a mapping slip in the kernel text cannot be
//! copied into its twin: the Q4_K scales through
//! [`crate::coalesced_twin`]'s `q4_k_scale_min` rather than upstream's
//! branch-free unpack, and the MXFP4 codes through a plain table rather
//! than byte permutes.

use crate::coalesced_twin::warp_sum;
use crate::matvec_kinds::dp4a::{Dp4aKind, Q8_1_BLOCK_BYTES, Q8_1_BLOCK_ELEMS};

fn f16_at(b: &[u8]) -> f32 {
    crate::mul_mm::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

fn i32_at(b: &[u8]) -> i32 {
    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// f32 to f16 bits, round to nearest even, as `fq_f32_to_f16` computes
/// it on the host: `__float2half`. NaN becomes a quiet NaN of the same
/// sign (the device's `cvt` canonicalises a NaN's payload instead).
pub fn f32_to_f16_bits(f: f32) -> u16 {
    let u = f.to_bits();
    let sign = (u >> 16) & 0x8000;
    let ax = u & 0x7FFF_FFFF;
    let h = if ax >= 0x7F80_0000 {
        sign | 0x7C00 | if ax > 0x7F80_0000 { 0x0200 } else { 0 }
    } else if ax >= 0x477F_F000 {
        sign | 0x7C00
    } else if ax < 0x3880_0000 {
        if ax < 0x3300_0000 {
            sign
        } else {
            let m = (ax & 0x7F_FFFF) | 0x80_0000;
            let shift = 126 - (ax >> 23);
            let mut k = m >> shift;
            let rem = m & ((1 << shift) - 1);
            let half = 1 << (shift - 1);
            if rem > half || (rem == half && k & 1 == 1) {
                k += 1;
            }
            sign | k
        }
    } else {
        let r = ax - 0x3800_0000;
        sign | ((r + 0x0FFF + ((r >> 13) & 1)) >> 13)
    };
    h as u16
}

/// The xor butterfly `warp_reduce_max` / `warp_reduce_sum` run, as
/// every lane ends it: `x = op(x, x[lane ^ o])` for `o = 16 .. 1`.
fn butterfly(mut lanes: [f32; 32], op: impl Fn(f32, f32) -> f32) -> [f32; 32] {
    for o in [16, 8, 4, 2, 1] {
        let prev = lanes;
        for (l, v) in lanes.iter_mut().enumerate() {
            *v = op(prev[l], prev[l ^ o]);
        }
    }
    lanes
}

/// Twin of `quantize_q8_1`: `x` (a multiple of 32 long) as `block_q8_1`
/// bytes, 36 per 32 values: f16 `d = amax / 127`, f16 of the warp's f32
/// sum, then `roundf(x / d)` as int8 (all zero when `amax == 0`).
pub fn quantize_q8_1(x: &[f32]) -> Vec<u8> {
    assert!(
        x.len().is_multiple_of(Q8_1_BLOCK_ELEMS),
        "q8_1 quantizes whole 32-value blocks, not {}",
        x.len()
    );
    let mut out = Vec::with_capacity(x.len() / Q8_1_BLOCK_ELEMS * Q8_1_BLOCK_BYTES);
    for blk in x.as_chunks::<Q8_1_BLOCK_ELEMS>().0 {
        let mut abs = [0f32; 32];
        let mut val = [0f32; 32];
        for (l, v) in blk.iter().enumerate() {
            abs[l] = v.abs();
            val[l] = *v;
        }
        // `fmaxf` and Rust's `max` both return the non-NaN operand.
        let amax = butterfly(abs, f32::max)[0];
        let sum = butterfly(val, |a, b| a + b)[0];
        let d = amax / 127.0;
        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16_bits(sum).to_le_bytes());
        for v in blk {
            let q = if amax == 0.0 {
                0
            } else {
                (v / d).round() as i32
            };
            out.push(q as i8 as u8);
        }
    }
    out
}

/// The values `block_q8_1` bytes stand for: `q * d`, `d` the stored
/// f16 widened. Exact in f32 (a 7-bit integer times an 11-bit mantissa).
pub fn q8_1_dequant(y: &[u8]) -> Vec<f32> {
    y.as_chunks::<Q8_1_BLOCK_BYTES>()
        .0
        .iter()
        .flat_map(|b| {
            let d = f16_at(b);
            b[4..].iter().map(move |q| f32::from(*q as i8) * d)
        })
        .collect()
}

/// `x` as a dp4a matvec reads it: through [`quantize_q8_1`] and back.
/// The integer dot is exact, so a dp4a matvec's result is the dot of
/// its weights with these values up to f32 rounding, and a reference
/// that multiplies them models the device. The batch GEMM's quantizer
/// rounds differently (`v * (127 / amax)`), so its twins have their own.
pub fn q8_1_round(x: &[f32]) -> Vec<f32> {
    q8_1_dequant(&quantize_q8_1(x))
}

/// `dp4a.s32.s32`: `c` plus the four signed byte products of `a` and `b`.
pub fn dp4a(a: i32, b: i32, c: i32) -> i32 {
    let (a, b) = (a.to_le_bytes(), b.to_le_bytes());
    (0..4).fold(c, |acc, i| {
        acc.wrapping_add(i32::from(a[i] as i8) * i32::from(b[i] as i8))
    })
}

/// `vec_dot_q4_K_q8_1` for thread word `iqs` of the 144-byte block
/// `blk`, against `y`, the eight q8_1 blocks of its 256 activations.
pub fn vec_dot_q4_k(blk: &[u8], y: &[u8], iqs: usize) -> f32 {
    let bq8_offset = 2 * ((iqs / 2) / 4);
    let r = (iqs / 2) % 4;
    let q4 = 16 + 16 * bq8_offset + 4 * r;
    let (v0, v1) = (i32_at(&blk[q4..]), i32_at(&blk[q4 + 16..]));
    let scales = &blk[4..16];
    let mut sumf_d = 0f32;
    let mut sumf_m = 0f32;
    for i in 0..2 {
        let (sc, m) = crate::coalesced_twin::q4_k_scale_min(bq8_offset + i, scales);
        let b8 = &y[(bq8_offset + i) * Q8_1_BLOCK_BYTES..];
        let d8 = f16_at(b8);
        let u0 = i32_at(&b8[4 + 4 * r..]);
        let u1 = i32_at(&b8[4 + 16 + 4 * r..]);
        let v0i = (v0 >> (4 * i)) & 0x0F0F_0F0F;
        let v1i = (v1 >> (4 * i)) & 0x0F0F_0F0F;
        let dot1 = dp4a(v1i, u1, dp4a(v0i, u0, 0));
        let dot2 = dp4a(0x0101_0101, u1, dp4a(0x0101_0101, u0, 0));
        sumf_d += d8 * (dot1 * i32::from(sc)) as f32;
        sumf_m += d8 * (dot2 * i32::from(m)) as f32;
    }
    f16_at(&blk[0..]) * sumf_d - f16_at(&blk[2..]) * sumf_m
}

/// `vec_dot_q6_K_q8_1` (vdr 2) for thread word `iqs` of the 210-byte
/// block `blk`.
pub fn vec_dot_q6_k(blk: &[u8], y: &[u8], iqs: usize) -> f32 {
    let bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    let scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    let vh_shift = 2 * ((iqs % 16) / 8);
    let vl: [u32; 2] = std::array::from_fn(|j| i32_at(&blk[4 * (iqs + j)..]) as u32);
    let vh: [u32; 2] = std::array::from_fn(|j| {
        (i32_at(&blk[128 + 4 * (8 * (iqs / 16) + iqs % 8 + j)..]) as u32) >> vh_shift
    });
    let mut sumf = 0f32;
    for i in 0..2 {
        let b8 = &y[(bq8_offset + 2 * i) * Q8_1_BLOCK_BYTES..];
        let sc = i32::from(blk[192 + scale_offset + 4 * i] as i8);
        let mut sumi = 0i32;
        for j in 0..2 {
            let u = i32_at(&b8[4 + 4 * (iqs % 8 + j)..]);
            let vil = ((vl[j] >> (4 * i)) & 0x0F0F_0F0F) as i32;
            let vih = (((vh[j] >> (4 * i)) << 4) & 0x3030_3030) as i32;
            sumi = dp4a(vil | vih, u, sumi);
            sumi = dp4a(0xE0E0_E0E0_u32 as i32, u, sumi);
        }
        sumf += f16_at(b8) * (sumi * sc) as f32;
    }
    f16_at(&blk[208..]) * sumf
}

/// `vec_dot_q8_0_q8_1` (vdr 2) for thread word `iqs` of the 34-byte
/// block `blk`: `(d_w * d_x) * sumi`.
pub fn vec_dot_q8_0(blk: &[u8], y: &[u8], iqs: usize) -> f32 {
    let mut sumi = 0i32;
    for i in 0..2 {
        sumi = dp4a(
            i32_at(&blk[2 + 4 * (iqs + i)..]),
            i32_at(&y[4 + 4 * (iqs + i)..]),
            sumi,
        );
    }
    f16_at(blk) * f16_at(y) * sumi as f32
}

/// ggml's `kvalues_mxfp4`: E2M1 doubled, so every value is an int8.
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// `vec_dot_mxfp4_q8_1` (vdr 2) for thread word `iqs` of the 17-byte
/// block `blk`. A code byte's low nibble is element `b`, its high nibble
/// element `b + 16`; the scale is `(e8m0 * 0.5) * d_x`.
pub fn vec_dot_mxfp4(blk: &[u8], y: &[u8], iqs: usize) -> f32 {
    let mut sumi = 0i32;
    for l in 0..2 {
        let codes = &blk[1 + 4 * (iqs + l)..1 + 4 * (iqs + l) + 4];
        let lo = i32::from_le_bytes(std::array::from_fn(|b| {
            KVALUES_MXFP4[usize::from(codes[b] & 15)] as u8
        }));
        let hi = i32::from_le_bytes(std::array::from_fn(|b| {
            KVALUES_MXFP4[usize::from(codes[b] >> 4)] as u8
        }));
        sumi = dp4a(lo, i32_at(&y[4 + 4 * (iqs + l)..]), sumi);
        sumi = dp4a(hi, i32_at(&y[4 + 4 * (iqs + l + 4)..]), sumi);
    }
    let d = crate::coalesced_twin::e8m0(blk[0]) * 0.5 * f16_at(y);
    d * sumi as f32
}

/// One thread word's dot: a weight block, its q8_1 blocks, the word.
pub type VecDot = fn(&[u8], &[u8], usize) -> f32;

/// The vec-dot twin of a dp4a kernel's entry point.
pub fn vec_dot_for(entry: &str) -> Option<VecDot> {
    match entry {
        "q4_k_matvec_dp4a" => Some(vec_dot_q4_k),
        "q6_k_matvec_dp4a" => Some(vec_dot_q6_k),
        "q8_0_matvec_dp4a" => Some(vec_dot_q8_0),
        "mxfp4_matvec_dp4a" => Some(vec_dot_mxfp4),
        _ => None,
    }
}

/// Which `(block, word)` pairs thread `tid` of an `nwarps`-warp block
/// takes in a row of `n_blocks` blocks of `kind`, in its loop order.
/// The one mapping the row twin and the coverage test both use.
pub fn thread_blocks(
    kind: &Dp4aKind,
    tid: usize,
    nwarps: usize,
    n_blocks: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let tpb = kind.threads_per_block();
    let iqs = kind.vdr * (tid % tpb);
    (tid / tpb..n_blocks)
        .step_by(32 * nwarps / tpb)
        .map(move |b| (b, iqs))
}

/// Twin of a dp4a kernel for one row: `row` is its `n_blocks` blocks,
/// `y` the q8_1 activations (`n_blocks * qk / 32` blocks), `nwarps` the
/// `FQ_NWARPS` it was compiled with. `FQ_ROWS` does not enter: a
/// thread's partial for a row is the same sum whichever rows share it.
pub fn dp4a_matvec_row(
    kind: &Dp4aKind,
    row: &[u8],
    y: &[u8],
    n_blocks: usize,
    nwarps: usize,
) -> f32 {
    let dot = vec_dot_for(kind.entry).expect("every dp4a kernel has a vec-dot twin");
    let partial = |tid: usize| {
        let mut acc = 0f32;
        for (b, iqs) in thread_blocks(kind, tid, nwarps, n_blocks) {
            let blk = &row[b * kind.block_bytes..(b + 1) * kind.block_bytes];
            let y8 = &y[b * kind.q8_per_block() * Q8_1_BLOCK_BYTES..];
            acc += dot(blk, y8, iqs);
        }
        acc
    };
    let mut lanes = [0f32; 32];
    for (lane, v) in lanes.iter_mut().enumerate() {
        let mut s = partial(lane);
        for w in 1..nwarps {
            s += partial(32 * w + lane);
        }
        *v = s;
    }
    // Lane 0 of the xor butterfly sums exactly as the shuffle-down tree
    // does, and every lane ends with lane 0's value (tested below).
    warp_sum(lanes)
}

/// Fixtures the host tests here and the hardware tests in
/// `dp4a_launch` share.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    fn pseudo(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// Blocks with finite headers: f16 scales near 2^-4 and 2^-5,
    /// MXFP4's E8M0 byte near 2^0.
    pub(crate) fn weights(kind: &Dp4aKind, n_blocks: usize, seed: u32) -> Vec<u8> {
        let bb = kind.block_bytes;
        let mut r = pseudo(n_blocks * bb, seed);
        for b in r.chunks_exact_mut(bb) {
            match bb {
                144 => b[0..4].copy_from_slice(&[0x00, 0x2C, 0x00, 0x28]),
                210 => b[208..210].copy_from_slice(&[0x00, 0x2C]),
                34 => b[0..2].copy_from_slice(&[0x00, 0x2C]),
                17 => b[0] = 120 + b[0] % 16,
                _ => unreachable!("{bb}"),
            }
        }
        r
    }

    /// The rows' weights as f32, by `frink_quant`'s dequant (MXFP4 from
    /// the GGUF block definition, apart from the twin's own table use).
    pub(crate) fn dequant(kind: &Dp4aKind, r: &[u8]) -> Vec<f32> {
        match kind.block_bytes {
            144 => frink_quant::dequant_q4_k(r).expect("q4_k"),
            210 => frink_quant::dequant_q6_k(r).expect("q6_k"),
            34 => frink_quant::dequant_q8_0(r).expect("q8_0"),
            17 => r
                .as_chunks::<17>()
                .0
                .iter()
                .flat_map(|b| {
                    let half = 2f64.powi(i32::from(b[0]) - 128) as f32;
                    (0..32).map(move |i| {
                        f32::from(
                            KVALUES_MXFP4[usize::from((b[1 + i % 16] >> (4 * (i / 16))) & 15)],
                        ) * half
                    })
                })
                .collect(),
            _ => unreachable!(),
        }
    }

    /// The q8_1 values back as f32: `q * d`, `d` the stored f16.
    pub(crate) fn dequant_q8_1(y: &[u8]) -> Vec<f32> {
        q8_1_dequant(y)
    }

    /// Activations whose blocks differ in scale, so `amax` varies.
    pub(crate) fn activations(n: usize, phase: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.021 + phase).sin() * (1.0 + (i % 7) as f32))
            .collect()
    }

    /// The bound a q8_1 activation costs against the f32 dot, per term:
    /// `|x - q*d16| <= d32/2 + 127*|d32 - d16|`, with `d16` the f16 of
    /// `d32 = amax/127`, so at most `amax/127 * (1/2 + 127 * 2^-11)`,
    /// about 2^-7.8 of the block's largest activation. Summed with each
    /// weight's magnitude, plus f32 rounding on both sides. The hardware
    /// test holds the device's dp4a result to its coalesced f32 one with
    /// it.
    pub(crate) fn q8_1_bound(w: &[f32], x: &[f32]) -> f64 {
        let per = 0.5 + 127.0 * 2f64.powi(-11);
        let mut bound = 0f64;
        for (wb, xb) in w.chunks(32).zip(x.chunks(32)) {
            let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
            let wsum: f64 = wb.iter().map(|v| f64::from(v.abs())).sum();
            bound += wsum * f64::from(amax) / 127.0 * per;
        }
        let terms: f64 = w
            .iter()
            .zip(x)
            .map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs())
            .sum();
        bound + 1e-5 * terms + 1e-6
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{activations, dequant, dequant_q8_1, q8_1_bound, weights};
    use super::*;
    use crate::matvec_kinds::dp4a::{dp4a_kind, DP4A};

    /// Every dp4a twin against a dequantize-then-dot of the SAME
    /// quantized activations, in f64. The integer dot is exact, so all
    /// that separates the two is f32 rounding of a few products per
    /// block: under `1e-5` of the term scale. A wrong lane mapping, a
    /// misread scale or a sign slip is orders of magnitude past it.
    ///
    /// Sabotage: `0xE0E0E0E0` to `0xE1E1E1E1` in the Q6_K twin, or
    /// `iqs % 8` to `iqs % 4` in its `u` offset, and Q6_K fails here.
    #[test]
    fn every_twin_matches_dequantize_then_dot_of_its_q8_1_input() {
        for kind in DP4A {
            for nwarps in [1usize, 2, 4] {
                for (n_blocks, seed) in [(1usize, 3u32), (3, 7), (6, 11), (16, 13), (37, 17)] {
                    let row = weights(kind, n_blocks, seed);
                    let x = activations(n_blocks * kind.qk, seed as f32);
                    let y = quantize_q8_1(&x);
                    let got = dp4a_matvec_row(kind, &row, &y, n_blocks, nwarps);
                    let w = dequant(kind, &row);
                    let xq = dequant_q8_1(&y);
                    let want: f64 = w
                        .iter()
                        .zip(&xq)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum();
                    let scale: f64 = w
                        .iter()
                        .zip(&xq)
                        .map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs())
                        .sum::<f64>()
                        .max(1e-6);
                    let err = (f64::from(got) - want).abs() / scale;
                    assert!(
                        err < 1e-5,
                        "{} nwarps={nwarps} n_blocks={n_blocks}: twin {got} vs {want} (rel {err:e})",
                        kind.entry
                    );
                }
            }
        }
    }

    /// The dp4a twin against the coalesced f32 twin of the same kind:
    /// the only difference is the q8_1 rounding of the activation, so
    /// they agree within [`q8_1_bound`] and not much better. The second
    /// assertion keeps the fixture honest: a bound a thousand times the
    /// actual gap would pass a kernel that ignored half the row. A slip
    /// smaller than the q8_1 step is the tight test's above to catch.
    ///
    /// Sabotage: drop the Q4_K twin's `- dmin * sumf_m` and Q4_K fails.
    #[test]
    fn the_dp4a_and_f32_twins_differ_by_the_q8_1_rounding() {
        for kind in DP4A {
            let f32_twin =
                crate::coalesced_twin::twin_for(&kind.entry.replace("_dp4a", "_coalesced"))
                    .expect("a coalesced twin");
            let n_blocks = 12;
            let row = weights(kind, n_blocks, 29);
            // Positive, so Q4_K's min term (`dmin * m * sum(x)`) adds up
            // across sub-blocks instead of cancelling.
            let x: Vec<f32> = activations(n_blocks * kind.qk, 0.3)
                .iter()
                .map(|v| v.abs() + 0.25)
                .collect();
            let y = quantize_q8_1(&x);
            let a = dp4a_matvec_row(kind, &row, &y, n_blocks, 4);
            let b = f32_twin(&row, &x, n_blocks, 32);
            let w = dequant(kind, &row);
            let bound = q8_1_bound(&w, &x);
            let gap = f64::from((a - b).abs());
            assert!(
                gap <= bound,
                "{}: dp4a {a} f32 {b}, gap {gap} > bound {bound}",
                kind.entry
            );
            assert!(
                bound
                    < 0.05
                        * w.iter()
                            .zip(&x)
                            .map(|(p, q)| f64::from((p * q).abs()))
                            .sum::<f64>(),
                "{}: the bound is too loose to mean anything",
                kind.entry
            );
        }
    }

    /// Every quant word of every block is read by exactly one thread at
    /// every warp count, including rows shorter than one loop trip.
    #[test]
    fn the_threads_cover_every_quant_word_exactly_once() {
        for kind in DP4A {
            let words = kind.qi;
            for nwarps in [1usize, 2, 4] {
                for n_blocks in [1usize, 5, 33, 70] {
                    let mut seen = vec![0u8; n_blocks * words];
                    for tid in 0..32 * nwarps {
                        for (b, iqs) in thread_blocks(kind, tid, nwarps, n_blocks) {
                            for w in iqs..iqs + kind.vdr {
                                seen[b * words + w] += 1;
                            }
                        }
                    }
                    assert!(
                        seen.iter().all(|c| *c == 1),
                        "{} nwarps={nwarps} n={n_blocks}",
                        kind.entry
                    );
                }
            }
        }
    }

    /// The quantizer on blocks built to tell its rules apart: an all-zero
    /// block (`d = 0`, every `q = 0`, no division), ties that `roundf`
    /// takes away from zero, the f16 of the f32 sum, and `amax` mapping to
    /// exactly +-127.
    ///
    /// Sabotage: `.round()` to `.round_ties_even()` fails the tie block.
    #[test]
    fn the_quantizer_follows_quantize_q8_1() {
        let mut x = vec![0f32; 32];
        // Block 1: amax 127 so d = 1, with exact halves.
        x.extend((0..32).map(|i| match i {
            0 => 127.0,
            1 => -127.0,
            2 => 2.5,
            3 => -2.5,
            4 => 0.5,
            _ => (i as f32) - 16.0,
        }));
        // Block 2: a small amax, whose f16 d is not d.
        x.extend((0..32).map(|i| ((i as f32) * 0.37).sin() * 1e-3));
        let y = quantize_q8_1(&x);
        assert_eq!(y.len(), 3 * Q8_1_BLOCK_BYTES);
        assert!(
            y[..Q8_1_BLOCK_BYTES].iter().all(|b| *b == 0),
            "a zero block is all zero"
        );
        let b1 = &y[36..72];
        assert_eq!(f16_at(b1), 1.0);
        let q: Vec<i8> = b1[4..].iter().map(|v| *v as i8).collect();
        assert_eq!(
            &q[..5],
            &[127, -127, 3, -3, 1],
            "roundf ties away from zero"
        );
        let sum: f32 = x[32..64].iter().sum();
        assert!(
            (f16_at(&b1[2..]) - sum).abs() <= sum.abs() * 1e-3,
            "s is the block sum"
        );
        let b2 = &y[72..108];
        let amax = x[64..96].iter().fold(0f32, |m, v| m.max(v.abs()));
        assert_eq!(
            f16_at(b2),
            crate::mul_mm::f16_to_f32(f32_to_f16_bits(amax / 127.0))
        );
        assert!(b2[4..].iter().any(|v| (*v as i8).unsigned_abs() == 127));
    }

    /// The f16 rounding against the `half` crate's, which is IEEE
    /// round-to-nearest-even: every exponent with a spread of mantissas
    /// (ties, one ulp either side), the subnormal and overflow edges
    /// exhaustively, and a stride over the whole f32 space.
    #[test]
    fn f32_to_f16_bits_rounds_to_nearest_even() {
        let check = |bits: u32| {
            let f = f32::from_bits(bits);
            let want = half::f16::from_f32(f);
            let got = f32_to_f16_bits(f);
            if want.is_nan() {
                assert!(half::f16::from_bits(got).is_nan(), "{bits:#x}");
            } else {
                assert_eq!(got, want.to_bits(), "{f:e} ({bits:#x})");
            }
        };
        for e in 0u32..256 {
            for m in [
                0u32, 1, 0xFFF, 0x1000, 0x1001, 0x2000, 0x3000, 0x7F_E000, 0x7F_F000, 0x7F_FFFF,
            ] {
                check((e << 23) | m);
                check(0x8000_0000 | (e << 23) | m);
            }
        }
        for edge in [0x3300_0000u32, 0x3880_0000, 0x477F_F000] {
            for bits in edge - (1 << 16)..edge + (1 << 16) {
                check(bits);
            }
        }
        let mut bits = 0u32;
        while let Some(next) = bits.checked_add(65_537) {
            check(bits);
            bits = next;
        }
    }

    /// The software dp4a is four signed byte products: 0xFF is -1, not
    /// 255, on both sides.
    #[test]
    fn dp4a_multiplies_signed_bytes() {
        assert_eq!(dp4a(-1, -1, 0), 4);
        assert_eq!(
            dp4a(0x0101_0101, i32::from_le_bytes([0x80, 0x7F, 1, 0xFF]), 10),
            10 - 128 + 127 + 1 - 1
        );
        assert_eq!(dp4a(0xE0E0_E0E0_u32 as i32, 0x0101_0101, 0), -128);
    }

    /// The xor butterfly ends with one value on every lane (so lane `i`
    /// may store row `i`), and lane 0's is the shuffle-down tree's. A
    /// shuffle-DOWN reduction gives lane 0 the same sum and the other
    /// lanes partial ones, which is why the kernels that store a row per
    /// lane reduce with xor; the fixture is checked to show it.
    ///
    /// Sabotage: the butterfly's offsets ascending (`[1, 2, 4, 8, 16]`)
    /// fails here. (Partner `(l + o) % 32` instead of `l ^ o` does not,
    /// and cannot: each step's partials repeat with period `o`, so the
    /// rotation adds the same pairs.)
    #[test]
    fn the_xor_butterfly_is_the_shuffle_down_tree_on_every_lane() {
        // Magnitudes from 1 to 2^24 in a cycle of nine, so which values
        // meet first decides which low bits survive.
        let mut lanes = [0f32; 32];
        for (i, l) in lanes.iter_mut().enumerate() {
            *l = ((i as f32) * 0.37 + 0.1).sin() * 2f32.powi(3 * (i % 9) as i32);
        }
        let all = butterfly(lanes, |a, b| a + b);
        let tree = warp_sum(lanes);
        for (lane, v) in all.iter().enumerate() {
            assert_eq!(v.to_bits(), tree.to_bits(), "lane {lane}");
        }
        // Two other reductions, to show the fixture separates them:
        // shuffle-down (lane 0 right, the rest partial) and the xor
        // butterfly with its offsets ascending (a different order).
        let model = |offsets: [usize; 5], partner: fn(usize, usize) -> usize| {
            let mut v = lanes;
            for o in offsets {
                let prev = v;
                for (l, x) in v.iter_mut().enumerate() {
                    *x = prev[l] + prev[partner(l, o)];
                }
            }
            v
        };
        let down = model([16, 8, 4, 2, 1], |l, o| if l + o < 32 { l + o } else { l });
        assert_eq!(down[0].to_bits(), tree.to_bits());
        assert!(down[1..].iter().any(|v| v.to_bits() != tree.to_bits()));
        let up = model([1, 2, 4, 8, 16], |l, o| l ^ o);
        assert_ne!(
            up[0].to_bits(),
            tree.to_bits(),
            "the fixture cannot tell the offset order"
        );
        let plain: f32 = lanes.iter().sum();
        assert_ne!(
            plain.to_bits(),
            tree.to_bits(),
            "the fixture cannot tell the orders apart"
        );
    }

    /// Every table row has a twin, and the Q4_K twin's scales come from
    /// the reading the coalesced twin uses, not the kernel's.
    #[test]
    fn every_dp4a_kernel_has_a_twin() {
        for k in DP4A {
            assert!(vec_dot_for(k.entry).is_some(), "{}", k.entry);
        }
        assert!(
            dp4a_kind("q5_k_matvec").is_none(),
            "Q5_K keeps the f32 kernel"
        );
    }
}
