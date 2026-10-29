//! Fixture encoders for the quant kinds `frink_quant` loads but does
//! not write (Q2_K, Q3_K, IQ4_NL, IQ4_XS, MXFP4), shared by the suites that
//! store a synthetic model in them. Each is a plain quantizer: it only
//! has to be a valid packing of SOME weights, and
//! `metal_low_k_quant_decode`'s round-trip test holds them to
//! `frink_quant`'s dequant, which defines the model every engine runs.

use frink_gguf::GgmlType;
use half::f16;

/// One Q2_K block (256 weights, 84 bytes): per sub-block, the range
/// `[min(lo, 0), hi]` in four steps.
pub fn q2_k_block(v: &[f32], out: &mut Vec<u8>) {
    let mut step = [0f32; 16];
    let mut min = [0f32; 16];
    for s in 0..16 {
        let sub = &v[16 * s..16 * s + 16];
        let lo = sub.iter().copied().fold(0f32, f32::min);
        let hi = sub.iter().copied().fold(f32::MIN, f32::max);
        step[s] = (hi - lo) / 3.0;
        min[s] = -lo;
    }
    let d = f16::from_f32(step.iter().copied().fold(0f32, f32::max) / 15.0);
    let dmin = f16::from_f32(min.iter().copied().fold(0f32, f32::max) / 15.0);
    let (df, dminf) = (d.to_f32(), dmin.to_f32());
    let mut blk = [0u8; 84];
    for s in 0..16 {
        let sc = if df > 0.0 {
            (step[s] / df).round().clamp(0.0, 15.0)
        } else {
            0.0
        };
        let m = if dminf > 0.0 {
            (min[s] / dminf).round().clamp(0.0, 15.0)
        } else {
            0.0
        };
        blk[s] = sc as u8 | ((m as u8) << 4);
        let (dl, ml) = (df * sc, dminf * m);
        let (n, j, h) = (s / 8, (s / 2) % 4, s % 2);
        for l in 0..16 {
            let raw = if dl > 0.0 {
                ((v[16 * s + l] + ml) / dl).round().clamp(0.0, 3.0)
            } else {
                0.0
            };
            blk[16 + 32 * n + 16 * h + l] |= (raw as u8) << (2 * j);
        }
    }
    blk[80..82].copy_from_slice(&d.to_le_bytes());
    blk[82..84].copy_from_slice(&dmin.to_le_bytes());
    out.extend_from_slice(&blk);
}

/// One Q3_K block (256 weights, 110 bytes): per sub-block, a symmetric
/// scale putting `max |v|` at 4 steps, the value stored as `q + 4` in
/// three bits (two in `qs`, the high one in `hmask`).
pub fn q3_k_block(v: &[f32], out: &mut Vec<u8>) {
    let mut step = [0f32; 16];
    for s in 0..16 {
        step[s] = v[16 * s..16 * s + 16]
            .iter()
            .fold(0f32, |m, x| m.max(x.abs()))
            / 4.0;
    }
    let d = f16::from_f32(step.iter().copied().fold(0f32, f32::max) / 31.0);
    let df = d.to_f32();
    let mut blk = [0u8; 110];
    for s in 0..16 {
        let sc = if df > 0.0 {
            (step[s] / df).round().clamp(0.0, 31.0)
        } else {
            0.0
        };
        let sc6 = sc as u8 + 32;
        blk[96 + s % 8] |= (sc6 & 0xF) << (4 * (s / 8));
        blk[104 + s % 4] |= (sc6 >> 4) << (2 * (s / 4));
        let dl = df * sc;
        let (n, j, h) = (s / 8, (s / 2) % 4, s % 2);
        for l in 0..16 {
            let q = if dl > 0.0 {
                (v[16 * s + l] / dl).round().clamp(-4.0, 3.0)
            } else {
                0.0
            };
            let q = (q + 4.0) as u8;
            blk[32 + 32 * n + 16 * h + l] |= (q & 3) << (2 * j);
            blk[16 * h + l] |= (q >> 2) << (4 * n + j);
        }
    }
    blk[108..110].copy_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&blk);
}

/// E2M1, as `frink_quant` and ggml's Metal shaders read it.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// One row of f32 as GGUF MXFP4 blocks: an E8M0 exponent putting the
/// block's largest magnitude at or under 6, then each value's nearest
/// E2M1 code, low nibble element `j`, high nibble `j + 16`.
pub fn mxfp4_row(row: &[f32]) -> Vec<u8> {
    assert!(row.len().is_multiple_of(32), "MXFP4 needs whole blocks");
    let mut out = Vec::with_capacity(row.len() / 32 * 17);
    for block in row.as_chunks::<32>().0 {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let e = if amax == 0.0 {
            1
        } else {
            ((amax / 6.0).log2().ceil() as i32 + 127).clamp(1, 254)
        };
        let d = f32::from_bits((e as u32) << 23);
        let code = |v: f32| {
            (0..16u8)
                .min_by(|&a, &b| {
                    let da = (E2M1[a as usize] * d - v).abs();
                    let db = (E2M1[b as usize] * d - v).abs();
                    da.total_cmp(&db)
                })
                .expect("sixteen codes")
        };
        out.push(e as u8);
        for j in 0..16 {
            out.push(code(block[j]) | (code(block[j + 16]) << 4));
        }
    }
    out
}

/// The IQ4 codebook (ggml `kvalues_iq4nl`), shared by IQ4_NL and IQ4_XS.
const KVALUES_IQ4: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

/// The nearest codebook entry to `v` at scale `d`.
fn iq4_code(v: f32, d: f32) -> u8 {
    (0..16u8)
        .min_by(|&a, &b| {
            let da = (KVALUES_IQ4[a as usize] * d - v).abs();
            let db = (KVALUES_IQ4[b as usize] * d - v).abs();
            da.total_cmp(&db)
        })
        .expect("sixteen codes")
}

/// One row of f32 as IQ4_XS blocks (256 weights, 136 bytes): a `half`
/// super-scale `d`, then per 32-weight sub-block a 6-bit scale `ls`
/// (low nibble in `scales_l`, top two bits in `scales_h`) read as
/// `d * (ls - 32)`, and sixteen bytes of codes laid out as IQ4_NL's.
/// Each sub-block's scale puts its largest magnitude at `-127`.
pub fn iq4_xs_row(row: &[f32]) -> Vec<u8> {
    assert!(row.len().is_multiple_of(256), "IQ4_XS needs whole blocks");
    let mut out = Vec::with_capacity(row.len() / 256 * 136);
    for block in row.as_chunks::<256>().0 {
        let want: Vec<f32> = block
            .as_chunks::<32>()
            .0
            .iter()
            .map(|s| s.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0)
            .collect();
        let d = f16::from_f32(want.iter().copied().fold(0f32, f32::max) / 31.0);
        let df = d.to_f32();
        let mut blk = [0u8; 136];
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        let mut scales_h = 0u16;
        for (ib, sub) in block.as_chunks::<32>().0.iter().enumerate() {
            let s = if df > 0.0 {
                (want[ib] / df).round().clamp(0.0, 31.0)
            } else {
                0.0
            };
            let ls = s as u8 + 32;
            blk[4 + ib / 2] |= (ls & 0xF) << (4 * (ib % 2));
            scales_h |= u16::from(ls >> 4) << (2 * ib);
            let dl = df * s;
            for j in 0..16 {
                blk[8 + 16 * ib + j] = iq4_code(sub[j], dl) | (iq4_code(sub[j + 16], dl) << 4);
            }
        }
        blk[2..4].copy_from_slice(&scales_h.to_le_bytes());
        out.extend_from_slice(&blk);
    }
    out
}

/// One row of f32 as IQ4_NL blocks: `d` puts the block's largest
/// magnitude at the codebook's end (`-127`), then each value's nearest
/// code, low nibble element `j`, high nibble `j + 16`.
pub fn iq4_nl_row(row: &[f32]) -> Vec<u8> {
    assert!(row.len().is_multiple_of(32), "IQ4_NL needs whole blocks");
    let mut out = Vec::with_capacity(row.len() / 32 * 18);
    for block in row.as_chunks::<32>().0 {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = f16::from_f32(amax / 127.0);
        out.extend_from_slice(&d.to_le_bytes());
        for j in 0..16 {
            out.push(iq4_code(block[j], d.to_f32()) | (iq4_code(block[j + 16], d.to_f32()) << 4));
        }
    }
    out
}

/// `w`, a whole number of `cols`-wide rows, as `dtype`: the four kinds
/// above, and Q8_0, Q5_K and Q6_K through `frink_quant`'s own encoders.
pub fn encode(dtype: GgmlType, cols: usize, w: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    match dtype {
        GgmlType::Q2K | GgmlType::Q3K => {
            assert_eq!(cols % 256, 0, "a K-quant row is whole 256-weight blocks");
            let block: fn(&[f32], &mut Vec<u8>) = if dtype == GgmlType::Q2K {
                q2_k_block
            } else {
                q3_k_block
            };
            for chunk in w.as_chunks::<256>().0 {
                block(chunk, &mut out);
            }
        }
        GgmlType::IQ4NL => w.chunks_exact(cols).for_each(|r| out.extend(iq4_nl_row(r))),
        GgmlType::MXFP4 => w.chunks_exact(cols).for_each(|r| out.extend(mxfp4_row(r))),
        GgmlType::IQ4XS => w.chunks_exact(cols).for_each(|r| out.extend(iq4_xs_row(r))),
        GgmlType::Q5K => w.chunks_exact(cols).for_each(|r| {
            frink_quant::encode_row_q5_k(r, None, &mut out).expect("whole blocks");
        }),
        GgmlType::Q6K => w.chunks_exact(cols).for_each(|r| {
            frink_quant::encode_row_q6_k(r, None, &mut out).expect("whole blocks");
        }),
        GgmlType::Q8_0 => w.chunks_exact(cols).for_each(|r| {
            frink_quant::encode_row_q8_0(r, &mut out).expect("whole blocks");
        }),
        other => panic!("no fixture encoder for {other:?}"),
    }
    out
}
