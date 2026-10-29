//! Scalar twins of the coalesced matvec kernels in
//! [`crate::matvec_kinds::coalesced`].
//!
//! The CUDA C cannot run here, so the same arithmetic and the same
//! lane-to-data mapping are transcribed in Rust and held against a
//! dequantize-then-dot reference from `frink_quant`. The kernels' risk
//! is not the arithmetic but the MAPPING: which bytes each lane reads
//! and which activations they pair with, at every `lanes` (the
//! kernels' `FX_LANES`) the table can ask for.
//!
//! Each twin performs the kernel's float operations in the kernel's
//! order: one f32 accumulator per lane, walked block by block, then
//! the `__shfl_down_sync` tree the warp reduces with
//! ([`warp_sum`]). With contraction off that is bit-for-bit what the
//! CUDA C computes, which is what `tools/matvec_host_check` checks by
//! running the kernel text on a host CPU. A GPU contracts into FMAs,
//! so the hardware test compares with a relative tolerance instead.

fn f16_to_f32(bits: u16) -> f32 {
    crate::mul_mm::f16_to_f32(bits)
}

fn f16_at(b: &[u8]) -> f32 {
    f16_to_f32(u16::from(b[0]) | (u16::from(b[1]) << 8))
}

fn q4_k_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// The reduction `fx_warp_sum` performs, as lane 0 sees it:
/// `acc += shfl_down(acc, s)` for `s = 16, 8, 4, 2, 1`. A lane whose
/// source is past the warp keeps its own value, but no such lane feeds
/// lane 0, so the tree below is exact.
pub fn warp_sum(mut lanes: [f32; 32]) -> f32 {
    let mut s = 16;
    while s > 0 {
        for l in 0..s {
            lanes[l] += lanes[l + s];
        }
        s >>= 1;
    }
    lanes[0]
}

/// Which blocks lane `lane` visits and where its run starts, for a
/// kernel compiled with `FX_LANES = lanes` and `run` quants per lane:
/// `(first block, blocks per trip, run start)`. The one mapping every
/// twin uses, so the coverage test below checks the twins' mapping and
/// not a restatement of it.
pub fn lane_mapping(lane: usize, lanes: usize, run: usize) -> (usize, usize, usize) {
    assert!(
        (1..=32).contains(&lanes) && 32usize.is_multiple_of(lanes),
        "FX_LANES = {lanes} does not divide the warp"
    );
    (lane / lanes, 32 / lanes, (lane % lanes) * run)
}

/// Twin of `q4_k_matvec_coalesced`. Lane `sub` reads
/// `qs[sub*B .. sub*B + B)`, `B = 128 / lanes`, which sits in 32-byte
/// group `oi = sub*B / 32`; that group's low nibbles are activations
/// `oi*64 + (sub*B % 32) ..` and its high nibbles the same plus 32.
pub fn q4_k_matvec_coalesced_row(row: &[u8], x: &[f32], n_blocks: usize, lanes: usize) -> f32 {
    let run = 128 / lanes;
    assert!(
        run <= 32 && run.is_multiple_of(4),
        "Q4_K: lanes must be 4, 8, 16 or 32"
    );
    let mut acc = [0f32; 32];
    for (lane, a) in acc.iter_mut().enumerate() {
        let (first, step, off) = lane_mapping(lane, lanes, run);
        let oi = off / 32;
        let within = off % 32;
        for blk in (first..n_blocks).step_by(step) {
            let block = &row[blk * 144..(blk + 1) * 144];
            let d = f16_at(&block[0..2]);
            let dmin = f16_at(&block[2..4]);
            let scales = &block[4..16];
            let qs = &block[16..144];
            let (sc1, m1) = q4_k_scale_min(2 * oi, scales);
            let (sc2, m2) = q4_k_scale_min(2 * oi + 1, scales);
            let d1 = d * f32::from(sc1);
            let min1 = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let min2 = dmin * f32::from(m2);
            let xb = blk * 256 + oi * 64 + within;
            for i in 0..run {
                let b = qs[off + i];
                *a += (d1 * f32::from(b & 0x0F) - min1) * x[xb + i];
                *a += (d2 * f32::from(b >> 4) - min2) * x[xb + 32 + i];
            }
        }
    }
    warp_sum(acc)
}

/// Twin of `q5_k_matvec_coalesced`.
///
/// Q5_K shares Q4_K's six-bit scale/min packing; what it adds is a
/// fifth bit per weight, held in `qh` one BIT-PLANE per 32-element
/// group. Lane `sub` owns elements `[sub*R, sub*R + R)` of each group,
/// `R = 32 / lanes`, reads the same `qh` bytes for all four groups,
/// and picks bit `2*oi` for the low nibble and bit `2*oi + 1` for the
/// high one. Reading the wrong bit-plane perturbs weights by exactly
/// 16 and is easy to miss by eye.
pub fn q5_k_matvec_coalesced_row(row: &[u8], x: &[f32], n_blocks: usize, lanes: usize) -> f32 {
    let run = 32 / lanes;
    let mut acc = [0f32; 32];
    for (lane, a) in acc.iter_mut().enumerate() {
        let (first, step, l0) = lane_mapping(lane, lanes, run);
        for blk in (first..n_blocks).step_by(step) {
            let block = &row[blk * 176..(blk + 1) * 176];
            let d = f16_at(&block[0..2]);
            let dmin = f16_at(&block[2..4]);
            let scales = &block[4..16];
            let qh = &block[16..48];
            let qs = &block[48..176];
            for oi in 0..4usize {
                let (sc1, m1) = q4_k_scale_min(2 * oi, scales);
                let (sc2, m2) = q4_k_scale_min(2 * oi + 1, scales);
                let d1 = d * f32::from(sc1);
                let min1 = dmin * f32::from(m1);
                let d2 = d * f32::from(sc2);
                let min2 = dmin * f32::from(m2);
                let u1 = 1u8 << (2 * oi);
                let u2 = 2u8 << (2 * oi);
                let xb = blk * 256 + oi * 64 + l0;
                for e in 0..run {
                    let h = qh[l0 + e];
                    let q = qs[oi * 32 + l0 + e];
                    let hi1 = if h & u1 != 0 { 16u8 } else { 0 };
                    let hi2 = if h & u2 != 0 { 16u8 } else { 0 };
                    *a += (d1 * f32::from((q & 0x0F) + hi1) - min1) * x[xb + e];
                    *a += (d2 * f32::from((q >> 4) + hi2) - min2) * x[xb + 32 + e];
                }
            }
        }
    }
    warp_sum(acc)
}

/// Twin of `q6_k_matvec_coalesced`. Lane `sub` takes indices
/// `[sub*R, sub*R + R)` of the 32 each half's inner loop iterates,
/// `R = 32 / lanes`; the four sub-scales it uses are `is`, `is + 2`,
/// `is + 4`, `is + 6` with `is = sub*R / 16`.
pub fn q6_k_matvec_coalesced_row(row: &[u8], x: &[f32], n_blocks: usize, lanes: usize) -> f32 {
    let run = 32 / lanes;
    assert!(run <= 16, "Q6_K: a run must stay inside one scale group");
    let mut acc = [0f32; 32];
    for (lane, a) in acc.iter_mut().enumerate() {
        let (first, step, l0) = lane_mapping(lane, lanes, run);
        let is = l0 / 16;
        for blk in (first..n_blocks).step_by(step) {
            let block = &row[blk * 210..(blk + 1) * 210];
            let d = f16_at(&block[208..210]);
            for half in 0..2 {
                let ql = &block[half * 64..];
                let qh = &block[128 + half * 32..];
                let sc = &block[192 + half * 8..192 + half * 8 + 8];
                let xh = blk * 256 + half * 128 + l0;
                for e in 0..run {
                    let l = l0 + e;
                    let q1 = i32::from((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) - 32;
                    let q2 = i32::from((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32;
                    let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) - 32;
                    let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) - 32;
                    *a += d * f32::from(sc[is] as i8) * q1 as f32 * x[xh + e];
                    *a += d * f32::from(sc[is + 2] as i8) * q2 as f32 * x[xh + 32 + e];
                    *a += d * f32::from(sc[is + 4] as i8) * q3 as f32 * x[xh + 64 + e];
                    *a += d * f32::from(sc[is + 6] as i8) * q4 as f32 * x[xh + 96 + e];
                }
            }
        }
    }
    warp_sum(acc)
}

/// Twin of `q8_0_matvec_coalesced`. Lane `sub` takes quants
/// `[sub*R, sub*R + R)` of each block it visits, `R = 32 / lanes`; the
/// scale is per block, so every lane on a block reads the same two
/// header bytes and the hardware broadcasts them.
pub fn q8_0_matvec_coalesced_row(row: &[u8], x: &[f32], n_blocks: usize, lanes: usize) -> f32 {
    let run = 32 / lanes;
    let mut acc = [0f32; 32];
    for (lane, a) in acc.iter_mut().enumerate() {
        let (first, step, l0) = lane_mapping(lane, lanes, run);
        for blk in (first..n_blocks).step_by(step) {
            let block = &row[blk * 34..(blk + 1) * 34];
            let d = f16_at(&block[0..2]);
            for e in 0..run {
                let q = block[2 + l0 + e] as i8;
                *a += d * f32::from(q) * x[blk * 32 + l0 + e];
            }
        }
    }
    warp_sum(acc)
}

/// A twin's signature: one row, its activations, its block count, and
/// the `FX_LANES` the kernel was compiled with.
pub type CoalescedTwin = fn(&[u8], &[f32], usize, usize) -> f32;

/// The twin for a coalesced kernel's entry point, keyed the way
/// [`crate::matvec_kinds::coalesced::COALESCED`] is so the host check
/// and the hardware test can walk that table rather than a list of
/// their own.
pub fn twin_for(entry: &str) -> Option<CoalescedTwin> {
    match entry {
        "q4_k_matvec_coalesced" => Some(q4_k_matvec_coalesced_row),
        "q5_k_matvec_coalesced" => Some(q5_k_matvec_coalesced_row),
        "q6_k_matvec_coalesced" => Some(q6_k_matvec_coalesced_row),
        "q8_0_matvec_coalesced" => Some(q8_0_matvec_coalesced_row),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matvec_kinds::coalesced::{MatvecWidth, COALESCED};

    fn pseudo(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// Pseudo-random blocks with their f16 header fields pinned finite
    /// (`0x2C00` and `0x2800`), so a comparison against the f64
    /// reference measures the mapping and not a NaN scale.
    fn row(kind: &str, n_blocks: usize, seed: u32) -> Vec<u8> {
        let (bytes, finite_at): (usize, &[usize]) = match kind {
            "Q4_K" => (144, &[0, 2]),
            "Q5_K" => (176, &[0, 2]),
            "Q6_K" => (210, &[208]),
            "Q8_0" => (34, &[0]),
            _ => unreachable!("{kind}"),
        };
        let mut r = pseudo(n_blocks * bytes, seed);
        for blk in 0..n_blocks {
            for (i, at) in finite_at.iter().enumerate() {
                r[blk * bytes + at] = 0x00;
                r[blk * bytes + at + 1] = if i == 0 { 0x2C } else { 0x28 };
            }
        }
        r
    }

    fn dequant(kind: &str, r: &[u8]) -> Vec<f32> {
        match kind {
            "Q4_K" => frink_quant::dequant_q4_k(r),
            "Q5_K" => frink_quant::dequant_q5_k(r),
            "Q6_K" => frink_quant::dequant_q6_k(r),
            "Q8_0" => frink_quant::dequant_q8_0(r),
            _ => unreachable!("{kind}"),
        }
        .expect("dequant")
    }

    fn block_elems(kind: &str) -> usize {
        if kind == "Q8_0" {
            32
        } else {
            256
        }
    }

    const KINDS: [(&str, &str); 4] = [
        ("Q4_K", "q4_k_matvec_coalesced"),
        ("Q5_K", "q5_k_matvec_coalesced"),
        ("Q6_K", "q6_k_matvec_coalesced"),
        ("Q8_0", "q8_0_matvec_coalesced"),
    ];

    /// Every `FX_LANES` the table can compile, per entry point.
    fn widths(entry: &str) -> Vec<usize> {
        let k = COALESCED
            .iter()
            .find(|k| k.entry == entry)
            .unwrap_or_else(|| panic!("{entry} is not in COALESCED"));
        vec![k.lanes(MatvecWidth::Narrow), k.lanes(MatvecWidth::Wide)]
    }

    /// `|got - want|` against `sum |w_i x_i|`. The twin accumulates in
    /// f32 per lane exactly as the kernel does, over at most a few
    /// hundred terms per lane, so its error against an f64 reference is
    /// f32 summation error: under `1e-5` of the term scale here.
    fn assert_matches(got: f32, w: &[f32], x: &[f32], what: &str) {
        let want: f64 = w
            .iter()
            .zip(x)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let scale: f64 = w
            .iter()
            .zip(x)
            .map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs())
            .sum::<f64>()
            .max(1.0);
        let err = (f64::from(got) - want).abs() / scale;
        assert!(
            err < 1e-5,
            "{what}: twin {got} vs dequant-dot {want} (rel {err:e})"
        );
    }

    /// The table and the twins name the same kernels, so the hardware
    /// test and the host check can walk the table.
    #[test]
    fn every_coalesced_kernel_has_a_twin() {
        for k in COALESCED {
            assert!(twin_for(k.entry).is_some(), "{} has no twin", k.entry);
        }
        assert_eq!(COALESCED.len(), KINDS.len());
    }

    /// Every twin, at every width the table compiles, against
    /// dequantize-then-dot. Block counts include ones that are not a
    /// multiple of the blocks a wide warp takes per trip (1, 3, 5, 6),
    /// so a lane whose stride overruns or skips a block shows here.
    ///
    /// Sabotage: change `lane / lanes` to `lane % lanes` in
    /// `lane_mapping` and every wide case fails.
    #[test]
    fn every_twin_matches_dequantize_then_dot_at_every_width() {
        for (kind, entry) in KINDS {
            let twin = twin_for(entry).expect("twin");
            for lanes in widths(entry) {
                for (n_blocks, seed) in [(1usize, 7u32), (3, 19), (5, 41), (6, 53), (9, 61)] {
                    let r = row(kind, n_blocks, seed);
                    let n = n_blocks * block_elems(kind);
                    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.021).sin()).collect();
                    let got = twin(&r, &x, n_blocks, lanes);
                    assert_matches(
                        got,
                        &dequant(kind, &r),
                        &x,
                        &format!("{kind} lanes={lanes} n_blocks={n_blocks}"),
                    );
                }
            }
        }
    }

    /// The narrow and wide twins of one kind sum the same terms in a
    /// different order, so they agree to rounding and not bit for bit.
    /// A wide mapping that dropped a run would still be "close" to
    /// nothing; this pins the two widths to each other directly.
    #[test]
    fn the_narrow_and_wide_twins_agree() {
        for (kind, entry) in KINDS {
            let twin = twin_for(entry).expect("twin");
            let ws = widths(entry);
            let n_blocks = 12; // a 3072-wide K-quant row, 384 for Q8_0
            let r = row(kind, n_blocks, 97);
            let n = n_blocks * block_elems(kind);
            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).cos()).collect();
            let narrow = twin(&r, &x, n_blocks, ws[0]);
            let wide = twin(&r, &x, n_blocks, ws[1]);
            assert_matches(wide, &dequant(kind, &r), &x, &format!("{kind} wide"));
            assert!(
                (narrow - wide).abs() <= 1e-4 * narrow.abs().max(1.0),
                "{kind}: narrow {narrow} vs wide {wide}"
            );
        }
    }

    /// Every quant of every block must be read by exactly one lane at
    /// every width. An off-by-one in the run start or the block stride
    /// silently drops or double-counts a run, and a short row can hide
    /// it in the tolerance.
    #[test]
    fn the_lanes_cover_every_quant_exactly_once_at_every_width() {
        for (kind, entry) in KINDS {
            // Quant positions per block as the kernels index them: Q4_K
            // walks 128 `qs` bytes, the others a 32-wide index.
            let per_block = if kind == "Q4_K" { 128 } else { 32 };
            for lanes in widths(entry) {
                let run = per_block / lanes;
                let n_blocks = 7;
                let mut seen = vec![0u8; n_blocks * per_block];
                for lane in 0..32 {
                    let (first, step, start) = lane_mapping(lane, lanes, run);
                    for blk in (first..n_blocks).step_by(step) {
                        for e in 0..run {
                            seen[blk * per_block + start + e] += 1;
                        }
                    }
                }
                assert!(
                    seen.iter().all(|c| *c == 1),
                    "{kind} lanes={lanes}: lanes do not tile the row: {seen:?}"
                );
            }
        }
    }

    /// Only the second 32-element sub-block is non-zero, so a lane that
    /// pairs Q4_K's high nibbles with the FIRST activation block scores
    /// zero here instead of the right answer.
    #[test]
    fn the_q4_k_high_nibbles_pair_with_the_second_activation_block() {
        let r = row("Q4_K", 1, 13);
        let mut x = vec![0f32; 256];
        for (i, v) in x.iter_mut().enumerate().take(64).skip(32) {
            *v = ((i % 5) as f32) + 1.0;
        }
        let w = dequant("Q4_K", &r);
        for lanes in widths("q4_k_matvec_coalesced") {
            let got = q4_k_matvec_coalesced_row(&r, &x, 1, lanes);
            assert_matches(got, &w, &x, &format!("high nibbles, lanes={lanes}"));
        }
    }

    /// Q6_K's four sub-scales are 2 apart, not adjacent. Getting that
    /// wrong scales three of the four outputs by the wrong factor.
    #[test]
    fn the_q6_k_scale_indices_are_two_apart() {
        let r = row("Q6_K", 1, 9);
        let x: Vec<f32> = (0..256).map(|i| ((i % 11) as f32) - 5.0).collect();
        let w = dequant("Q6_K", &r);
        for lanes in widths("q6_k_matvec_coalesced") {
            let got = q6_k_matvec_coalesced_row(&r, &x, 1, lanes);
            assert_matches(got, &w, &x, &format!("q6_k scales, lanes={lanes}"));
        }
    }

    /// The Q5_K fifth bit lives in a per-group BIT-PLANE, so the low
    /// nibble of group `oi` takes bit `2*oi` and the high nibble bit
    /// `2*oi + 1`. Both wrong readings shift weights by 16.
    #[test]
    fn the_q5_k_fifth_bit_comes_from_the_right_bit_plane() {
        let r = row("Q5_K", 1, 13);
        let x: Vec<f32> = (0..256).map(|i| ((i % 9) as f32) - 4.0).collect();
        let w = dequant("Q5_K", &r);
        for lanes in widths("q5_k_matvec_coalesced") {
            let got = q5_k_matvec_coalesced_row(&r, &x, 1, lanes);
            assert_matches(got, &w, &x, &format!("q5_k bit-plane, lanes={lanes}"));
        }
    }

    /// Q8_0's quants are SIGNED. Reading them unsigned agrees with
    /// itself and with nothing else.
    #[test]
    fn the_q8_0_quants_are_read_signed() {
        let mut r = row("Q8_0", 1, 1);
        for b in r.iter_mut().skip(2) {
            *b = 0xF0; // -16 signed, 240 unsigned
        }
        let x = vec![1.0f32; 32];
        for lanes in widths("q8_0_matvec_coalesced") {
            let got = q8_0_matvec_coalesced_row(&r, &x, 1, lanes);
            assert!(got < 0.0, "lanes={lanes}: signed quants gave {got}");
        }
    }

    /// The tree must be the one `__shfl_down_sync` builds, not a plain
    /// sum: the host check compares the kernel to the twin bit for bit,
    /// and a different association order is a different f32. Checked
    /// against a simultaneous-update model of the shuffle, in which
    /// every lane reads its partner's value from BEFORE the step.
    #[test]
    fn warp_sum_is_the_shuffle_tree() {
        let mut lanes = [0f32; 32];
        for (i, l) in lanes.iter_mut().enumerate() {
            *l = (((i * 7919) % 97) as f32 - 48.0) * 10f32.powi((i % 7) as i32);
        }
        let mut v = lanes;
        for s in [16, 8, 4, 2, 1] {
            let prev = v;
            for l in 0..32 {
                if l + s < 32 {
                    v[l] = prev[l] + prev[l + s];
                }
            }
        }
        assert_eq!(warp_sum(lanes).to_bits(), v[0].to_bits());
        let plain: f32 = lanes.iter().sum();
        assert_ne!(
            plain.to_bits(),
            v[0].to_bits(),
            "this fixture cannot tell the tree from a plain sum"
        );
    }
}
