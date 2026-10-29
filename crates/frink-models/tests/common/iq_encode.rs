//! Fixture encoders for the seven grid-codebook kinds `frink_quant`
//! loads but does not write (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
//! IQ1_S, IQ1_M). Like `quant_encode`'s, each only has to be a valid
//! packing of SOME weights close to the input; the suites' round-trip
//! tests hold them to `frink_quant`'s dequant, which defines the model.
//!
//! Each group of eight (or four) weights is snapped to a grid row by
//! [`Grid::nearest`]: every element is rounded to the nearest of the
//! grid's levels, and that lattice point maps to the grid row nearest
//! it, a table built once per grid by brute force. Not the best row for
//! the continuous input, but cheap enough to encode a whole MoE fixture
//! and close enough to keep the model's activations sane.
//!
//! The layouts are `frink_quant`'s `for_each_iq*` walks: signs and
//! parity, scale nibbles and high index bits sit where they read them.

use frink_gguf::GgmlType;
use frink_quant::iq_tables as t;
use half::f16;
use std::sync::OnceLock;

/// A grid as `dim`-wide rows of levels, and the nearest row to every
/// point of the level lattice.
pub struct Grid {
    dim: usize,
    levels: Vec<f32>,
    nearest: Vec<u16>,
}

impl Grid {
    fn new(rows: &[[f32; 8]], dim: usize) -> Self {
        let mut levels: Vec<f32> = rows.iter().flat_map(|r| r[..dim].to_vec()).collect();
        levels.sort_by(f32::total_cmp);
        levels.dedup();
        let n = levels.len().pow(dim as u32);
        assert!(n <= 1 << 16, "a {}-level, {dim}-wide lattice", levels.len());
        let nearest = (0..n)
            .map(|mut c| {
                let mut p = [0f32; 8];
                for v in p.iter_mut().take(dim) {
                    *v = levels[c % levels.len()];
                    c /= levels.len();
                }
                let dist = |r: &[f32; 8]| -> f32 { (0..dim).map(|j| (r[j] - p[j]).powi(2)).sum() };
                (0..rows.len())
                    .min_by(|&a, &b| dist(&rows[a]).total_cmp(&dist(&rows[b])))
                    .expect("a grid row") as u16
            })
            .collect();
        Grid {
            dim,
            levels,
            nearest,
        }
    }

    /// The grid's largest level, the one a scale maps `max |v|` to.
    fn top(&self) -> f32 {
        *self.levels.last().expect("levels")
    }

    /// The row nearest `target` (in levels' units) through the lattice.
    fn nearest(&self, target: &[f32]) -> usize {
        let mut code = 0usize;
        for j in (0..self.dim).rev() {
            let k = (0..self.levels.len())
                .min_by(|&a, &b| {
                    (self.levels[a] - target[j])
                        .abs()
                        .total_cmp(&(self.levels[b] - target[j]).abs())
                })
                .expect("levels");
            code = code * self.levels.len() + k;
        }
        self.nearest[code] as usize
    }
}

fn bytes8(row: u64) -> [f32; 8] {
    std::array::from_fn(|j| ((row >> (8 * j)) & 0xFF) as u8 as f32)
}
fn bytes4(row: u32) -> [f32; 8] {
    std::array::from_fn(|j| {
        if j < 4 {
            ((row >> (8 * j)) & 0xFF) as f32
        } else {
            0.0
        }
    })
}
fn signed8(row: u64) -> [f32; 8] {
    std::array::from_fn(|j| ((row >> (8 * j)) & 0xFF) as u8 as i8 as f32)
}

fn grid(kind: GgmlType) -> &'static Grid {
    static G: [OnceLock<Grid>; 6] = [const { OnceLock::new() }; 6];
    let (slot, build): (usize, fn() -> Grid) = match kind {
        GgmlType::IQ2XXS => (0, || Grid::new(&t::IQ2XXS_GRID.map(bytes8), 8)),
        GgmlType::IQ2XS => (1, || Grid::new(&t::IQ2XS_GRID.map(bytes8), 8)),
        GgmlType::IQ2S => (2, || Grid::new(&t::IQ2S_GRID.map(bytes8), 8)),
        GgmlType::IQ3XXS => (3, || Grid::new(&t::IQ3XXS_GRID.map(bytes4), 4)),
        GgmlType::IQ3S => (4, || Grid::new(&t::IQ3S_GRID.map(bytes4), 4)),
        GgmlType::IQ1S | GgmlType::IQ1M => (5, || Grid::new(&t::IQ1S_GRID.map(signed8), 8)),
        other => panic!("{other:?} is not a grid kind"),
    };
    G[slot].get_or_init(build)
}

fn amax(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |m, x| m.max(x.abs()))
}

/// The `half` super-scale putting the largest wanted step at `top`
/// scale units, and its value.
fn super_scale(steps: &[f32], top: f32) -> (f16, f32) {
    let d = f16::from_f32(steps.iter().copied().fold(0f32, f32::max) / top);
    (d, d.to_f32())
}

/// The sign mask of eight values with an even number of negatives, as
/// `KSIGNS_IQ2XS` can express: an odd count flips the smallest one.
fn even_signs(v: &[f32]) -> u8 {
    let mut neg = (0..8).fold(0u8, |m, j| m | (u8::from(v[j] < 0.0) << j));
    if neg.count_ones() % 2 == 1 {
        let j = (0..8)
            .min_by(|&a, &b| v[a].abs().total_cmp(&v[b].abs()))
            .expect("eight");
        neg ^= 1 << j;
    }
    neg
}

fn literal_signs(v: &[f32]) -> u8 {
    (0..8).fold(0u8, |m, j| m | (u8::from(v[j] < 0.0) << j))
}

/// `|v| / dl`, the magnitudes a grid row is chosen for.
fn mags(v: &[f32], dl: f32) -> Vec<f32> {
    v.iter()
        .map(|x| if dl > 0.0 { x.abs() / dl } else { 0.0 })
        .collect()
}

/// The 4-bit IQ2 scale `s` with `d (0.5 + s) / 4` nearest `step`.
fn iq2_nibble(step: f32, d: f32) -> u8 {
    if d > 0.0 {
        (4.0 * step / d - 0.5).round().clamp(0.0, 15.0) as u8
    } else {
        0
    }
}

fn iq2_xxs(v: &[f32], out: &mut Vec<u8>) {
    let g = grid(GgmlType::IQ2XXS);
    let steps: Vec<f32> = v.chunks(32).map(|s| amax(s) / g.top()).collect();
    let (d, df) = super_scale(&steps, 15.5 / 4.0);
    let mut blk = [0u8; 66];
    blk[0..2].copy_from_slice(&d.to_le_bytes());
    for (ib, sub) in v.chunks(32).enumerate() {
        let s = iq2_nibble(steps[ib], df);
        let dl = df * (0.5 + f32::from(s)) * 0.25;
        let mut aux = u32::from(s) << 28;
        for (l, grp) in sub.chunks(8).enumerate() {
            blk[2 + 8 * ib + l] = g.nearest(&mags(grp, dl)) as u8;
            aux |= u32::from(even_signs(grp) & 127) << (7 * l);
        }
        blk[2 + 8 * ib + 4..2 + 8 * ib + 8].copy_from_slice(&aux.to_le_bytes());
    }
    out.extend_from_slice(&blk);
}

fn iq2_xs(v: &[f32], out: &mut Vec<u8>) {
    let g = grid(GgmlType::IQ2XS);
    let steps: Vec<f32> = v.chunks(16).map(|s| amax(s) / g.top()).collect();
    let (d, df) = super_scale(&steps, 15.5 / 4.0);
    let mut blk = [0u8; 74];
    blk[0..2].copy_from_slice(&d.to_le_bytes());
    for (h, half16) in v.chunks(16).enumerate() {
        let s = iq2_nibble(steps[h], df);
        blk[66 + h / 2] |= s << (4 * (h % 2));
        let dl = df * (0.5 + f32::from(s)) * 0.25;
        for (k, grp) in half16.chunks(8).enumerate() {
            let code = g.nearest(&mags(grp, dl)) as u16 | (u16::from(even_signs(grp) & 127) << 9);
            let at = 2 + 2 * (2 * h + k);
            blk[at..at + 2].copy_from_slice(&code.to_le_bytes());
        }
    }
    out.extend_from_slice(&blk);
}

fn iq2_s(v: &[f32], out: &mut Vec<u8>) {
    let g = grid(GgmlType::IQ2S);
    let steps: Vec<f32> = v.chunks(16).map(|s| amax(s) / g.top()).collect();
    let (d, df) = super_scale(&steps, 15.5 / 4.0);
    let mut blk = [0u8; 82];
    blk[0..2].copy_from_slice(&d.to_le_bytes());
    for (h, half16) in v.chunks(16).enumerate() {
        let s = iq2_nibble(steps[h], df);
        blk[74 + h / 2] |= s << (4 * (h % 2));
        let dl = df * (0.5 + f32::from(s)) * 0.25;
        for (k, grp) in half16.chunks(8).enumerate() {
            let gi = 2 * h + k;
            let idx = g.nearest(&mags(grp, dl));
            blk[2 + gi] = idx as u8;
            blk[66 + gi / 4] |= ((idx >> 8) as u8) << (2 * (gi % 4));
            blk[34 + gi] = literal_signs(grp);
        }
    }
    out.extend_from_slice(&blk);
}

fn iq3_xxs(v: &[f32], out: &mut Vec<u8>) {
    let g = grid(GgmlType::IQ3XXS);
    let steps: Vec<f32> = v.chunks(32).map(|s| amax(s) / g.top()).collect();
    let (d, df) = super_scale(&steps, 15.5 / 2.0);
    let mut blk = [0u8; 98];
    blk[0..2].copy_from_slice(&d.to_le_bytes());
    for (ib, sub) in v.chunks(32).enumerate() {
        let s = if df > 0.0 {
            (2.0 * steps[ib] / df - 0.5).round().clamp(0.0, 15.0) as u32
        } else {
            0
        };
        let dl = df * (0.5 + s as f32) * 0.5;
        let mut aux = s << 28;
        for (l, grp) in sub.chunks(8).enumerate() {
            let m = mags(grp, dl);
            blk[2 + 8 * ib + 2 * l] = g.nearest(&m[..4]) as u8;
            blk[2 + 8 * ib + 2 * l + 1] = g.nearest(&m[4..]) as u8;
            aux |= u32::from(even_signs(grp) & 127) << (7 * l);
        }
        blk[66 + 4 * ib..70 + 4 * ib].copy_from_slice(&aux.to_le_bytes());
    }
    out.extend_from_slice(&blk);
}

fn iq3_s(v: &[f32], out: &mut Vec<u8>) {
    let g = grid(GgmlType::IQ3S);
    let steps: Vec<f32> = v.chunks(32).map(|s| amax(s) / g.top()).collect();
    let (d, df) = super_scale(&steps, 31.0);
    let mut blk = [0u8; 110];
    blk[0..2].copy_from_slice(&d.to_le_bytes());
    for (ib, sub) in v.chunks(32).enumerate() {
        let s = if df > 0.0 {
            ((steps[ib] / df - 1.0) / 2.0).round().clamp(0.0, 15.0) as u8
        } else {
            0
        };
        blk[106 + ib / 2] |= s << (4 * (ib % 2));
        let dl = df * (1.0 + 2.0 * f32::from(s));
        for (l, grp) in sub.chunks(8).enumerate() {
            let m = mags(grp, dl);
            for (k, half4) in m.chunks(4).enumerate() {
                let idx = g.nearest(half4);
                blk[2 + 8 * ib + 2 * l + k] = idx as u8;
                blk[66 + ib] |= ((idx >> 8) as u8) << (2 * l + k);
            }
            blk[74 + 4 * ib + l] = literal_signs(grp);
        }
    }
    out.extend_from_slice(&blk);
}

/// IQ1's grid row for `v` at `dl`, and whether `-delta` fits better
/// than `+delta`: the row and the sign the group is stored with.
fn iq1_group(g: &Grid, v: &[f32], dl: f32) -> (usize, bool, f32) {
    let mut best = (0usize, false, f32::INFINITY);
    for neg in [false, true] {
        let delta = if neg { -0.125 } else { 0.125 };
        let target: Vec<f32> = v
            .iter()
            .map(|x| if dl > 0.0 { x / dl - delta } else { 0.0 })
            .collect();
        let idx = g.nearest(&target);
        let row = signed8(t::IQ1S_GRID[idx]);
        let err: f32 = (0..8).map(|j| (dl * (row[j] + delta) - v[j]).powi(2)).sum();
        if err < best.2 {
            best = (idx, neg, err);
        }
    }
    best
}

/// The 3-bit IQ1 scale `s` with `d (2 s + 1)` nearest `step`.
fn iq1_scale(step: f32, d: f32) -> u16 {
    if d > 0.0 {
        ((step / d - 1.0) / 2.0).round().clamp(0.0, 7.0) as u16
    } else {
        0
    }
}

fn iq1_s(v: &[f32], out: &mut Vec<u8>) {
    let g = grid(GgmlType::IQ1S);
    let steps: Vec<f32> = v.chunks(32).map(|s| amax(s) / 1.125).collect();
    let (d, df) = super_scale(&steps, 15.0);
    let mut blk = [0u8; 50];
    blk[0..2].copy_from_slice(&d.to_le_bytes());
    for (ib, sub) in v.chunks(32).enumerate() {
        let s = iq1_scale(steps[ib], df);
        let dl = df * (2.0 * f32::from(s) + 1.0);
        // One delta sign per sub-block: the one with less error summed.
        let pick = |neg: bool| -> (Vec<usize>, f32) {
            let delta = if neg { -0.125 } else { 0.125 };
            sub.chunks(8).fold((Vec::new(), 0f32), |(mut ids, e), grp| {
                let target: Vec<f32> = grp
                    .iter()
                    .map(|x| if dl > 0.0 { x / dl - delta } else { 0.0 })
                    .collect();
                let idx = g.nearest(&target);
                let row = signed8(t::IQ1S_GRID[idx]);
                ids.push(idx);
                let err: f32 = (0..8)
                    .map(|j| (dl * (row[j] + delta) - grp[j]).powi(2))
                    .sum();
                (ids, e + err)
            })
        };
        let (pos, neg) = (pick(false), pick(true));
        let (ids, is_neg) = if neg.1 < pos.1 {
            (neg.0, true)
        } else {
            (pos.0, false)
        };
        let mut qh = (s << 12) | (u16::from(is_neg) << 15);
        for (l, &idx) in ids.iter().enumerate() {
            blk[2 + 4 * ib + l] = idx as u8;
            qh |= ((idx >> 8) as u16) << (3 * l);
        }
        blk[34 + 2 * ib..36 + 2 * ib].copy_from_slice(&qh.to_le_bytes());
    }
    out.extend_from_slice(&blk);
}

fn iq1_m(v: &[f32], out: &mut Vec<u8>) {
    let g = grid(GgmlType::IQ1M);
    let steps: Vec<f32> = v.chunks(16).map(|s| amax(s) / 1.125).collect();
    let (d, df) = super_scale(&steps, 15.0);
    let mut blk = [0u8; 56];
    let mut sc = [0u16; 4];
    for (h, half16) in v.chunks(16).enumerate() {
        let s = iq1_scale(steps[h], df);
        let (ib, k) = (h / 2, h % 2);
        sc[ib / 2] |= s << (6 * (ib % 2) + 3 * k);
        let dl = df * (2.0 * f32::from(s) + 1.0);
        for (j, grp) in half16.chunks(8).enumerate() {
            let gi = 2 * h + j;
            let (idx, neg, _) = iq1_group(g, grp, dl);
            blk[gi] = idx as u8;
            let field = ((idx >> 8) as u8) | (u8::from(neg) << 3);
            blk[32 + gi / 2] |= field << (4 * (gi % 2));
        }
    }
    let bits = d.to_bits();
    for (i, w) in sc.iter_mut().enumerate() {
        *w |= ((bits >> (4 * i)) & 0xF) << 12;
        blk[48 + 2 * i..50 + 2 * i].copy_from_slice(&w.to_le_bytes());
    }
    out.extend_from_slice(&blk);
}

/// The seven kinds this module encodes.
pub const KINDS: [GgmlType; 7] = [
    GgmlType::IQ2XXS,
    GgmlType::IQ2XS,
    GgmlType::IQ2S,
    GgmlType::IQ3XXS,
    GgmlType::IQ3S,
    GgmlType::IQ1S,
    GgmlType::IQ1M,
];

/// `w` (whole 256-weight blocks) as `dtype`, if it is one of [`KINDS`].
pub fn encode(dtype: GgmlType, w: &[f32]) -> Option<Vec<u8>> {
    let block: fn(&[f32], &mut Vec<u8>) = match dtype {
        GgmlType::IQ2XXS => iq2_xxs,
        GgmlType::IQ2XS => iq2_xs,
        GgmlType::IQ2S => iq2_s,
        GgmlType::IQ3XXS => iq3_xxs,
        GgmlType::IQ3S => iq3_s,
        GgmlType::IQ1S => iq1_s,
        GgmlType::IQ1M => iq1_m,
        _ => return None,
    };
    assert!(
        w.len().is_multiple_of(256),
        "an IQ row is whole 256-weight blocks"
    );
    let mut out = Vec::new();
    for chunk in w.as_chunks::<256>().0 {
        block(chunk, &mut out);
    }
    Some(out)
}

/// The CPU dequant of `dtype`, one of [`KINDS`].
pub fn dequant(dtype: GgmlType) -> fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError> {
    match dtype {
        GgmlType::IQ2XXS => frink_quant::dequant_iq2_xxs,
        GgmlType::IQ2XS => frink_quant::dequant_iq2_xs,
        GgmlType::IQ2S => frink_quant::dequant_iq2_s,
        GgmlType::IQ3XXS => frink_quant::dequant_iq3_xxs,
        GgmlType::IQ3S => frink_quant::dequant_iq3_s,
        GgmlType::IQ1S => frink_quant::dequant_iq1_s,
        GgmlType::IQ1M => frink_quant::dequant_iq1_m,
        other => panic!("{other:?}"),
    }
}
