//! The tensor-core prefill attention's arithmetic on the host, and the
//! error its f16 operands can put on an output.
//!
//! [`prefill_attn_tc_twin`] performs the kernel's f32 operations in the
//! kernel's order: the same blocks and key tiles, f16 rounding where the
//! kernel rounds, each product summed in the order `mma.m16n8k16` sums
//! it when every add rounds (as the host check's CPU stand-in does), and
//! each lane's share of a row's sum kept apart until the quad's
//! butterfly. With FP contraction off the kernel text agrees with it bit
//! for bit (`tools/prefill_attn_tc_host_check/run.sh`). A GPU's tensor
//! cores add in their own order, so the hardware tests hold the kernel to
//! [`f16_error_bound`] against the plain definition instead.

use super::{block_row, first_key, grid, variant_for, AttnArgs, ROWS};

/// `x` rounded to the nearest f16 (ties to even), as an f32: what
/// `cvt.rn.f16.f32` stores. Overflow is infinity, a NaN stays a NaN.
pub fn round_f16(x: f32) -> f32 {
    /// f16's least normal, 2^-14.
    const MIN_NORMAL: f32 = f32::from_bits(0x3880_0000);
    let a = x.abs();
    if a.is_nan() {
        return x;
    }
    let r = if a >= 65_520.0 {
        f32::INFINITY
    } else if a < MIN_NORMAL {
        // Below f16's least normal the quantum is 2^-24. Adding and
        // removing 0.5, where an f32's quantum is also 2^-24, rounds to
        // it with ties to even.
        (a + 0.5) - 0.5
    } else {
        // Ten mantissa bits of f32's 23, ties to even.
        let b = a.to_bits();
        f32::from_bits((b + 0x0fff + ((b >> 13) & 1)) & !0x1fff)
    };
    r.copysign(x)
}

/// The kernel's output for `n_q` query rows (`[n_q][n_heads][head_dim]`)
/// over K/V rows `[start_pos + n_q][n_kv_heads][head_dim]`, at `group`
/// query heads per block. Panics on a shape the kernel does not take.
pub fn prefill_attn_tc_twin(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    args: &AttnArgs,
    sinks: Option<&[f32]>,
    group: usize,
) -> Vec<f32> {
    let var = variant_for(args).expect("a shape the tensor-core kernel takes");
    let AttnArgs {
        n_heads,
        n_kv_heads,
        head_dim: d,
        start_pos,
        window: _,
        scale: _,
        softcap: _,
    } = *args;
    let (bc, gqa, window) = (var.keys_per_tile, args.gqa(), args.window_or_zero());
    let (qf, cap) = (args.q_factor(), args.softcap_or_zero());
    let k16: Vec<f32> = k.iter().map(|&x| round_f16(x)).collect();
    let v16: Vec<f32> = v.iter().map(|&x| round_f16(x)).collect();
    let row = |t: usize, kvh: usize| (t * n_kv_heads + kvh) * d..(t * n_kv_heads + kvh + 1) * d;
    let mut out = vec![0f32; n_q * n_heads * d];
    let (gx, gy, _) = grid(n_q, n_kv_heads, gqa, group);
    for by in 0..gy as usize {
        for bx in 0..gx as usize {
            let j0 = bx * (ROWS / group);
            let last_j = (j0 + ROWS / group).min(n_q) - 1;
            let k_hi = start_pos + last_j + 1;
            let k_lo = first_key(start_pos + j0, window);
            for r in 0..ROWS {
                let Some((j, h)) = block_row((bx, by), r, group, gqa, n_q) else {
                    continue;
                };
                let (pos, kvh) = (start_pos + j, h / gqa);
                let q16: Vec<f32> = q[(j * n_heads + h) * d..][..d]
                    .iter()
                    .map(|&x| round_f16(x * qf))
                    .collect();
                let mut m = -f32::from_bits(0x7eff_ffff);
                let mut l = [0f32; 4];
                let mut o = vec![0f32; d];
                let mut s = vec![0f32; bc];
                let mut t0 = k_lo;
                while t0 < k_hi {
                    // A tile the row's warp skips leaves m, l and o as
                    // this does (every p is 0, alpha is 1), and one it
                    // takes unmasked has every key visible to the row.
                    let mut mx = f32::NEG_INFINITY;
                    for (jj, sj) in s.iter_mut().enumerate() {
                        let t = t0 + jj;
                        *sj = f32::NEG_INFINITY;
                        if t > pos || (window > 0 && pos - t >= window) {
                            continue;
                        }
                        let mut dot = 0f32;
                        for (a, b) in q16.iter().zip(&k16[row(t, kvh)]) {
                            dot += a * b;
                        }
                        if cap > 0.0 {
                            dot = cap * dot.tanh();
                        }
                        *sj = dot;
                        mx = mx.max(dot);
                    }
                    let mn = m.max(mx);
                    let alpha = (m - mn).exp();
                    m = mn;
                    // Lane tg of the quad holds keys 8 nt + 2 tg and
                    // 8 nt + 2 tg + 1 of every key tile nt, in that order.
                    let mut ls = [0f32; 4];
                    for (jj, sj) in s.iter_mut().enumerate() {
                        *sj = (*sj - m).exp();
                        ls[(jj % 8) / 2] += *sj;
                    }
                    for (lt, st) in l.iter_mut().zip(ls) {
                        *lt = *lt * alpha + st;
                    }
                    for x in o.iter_mut() {
                        *x *= alpha;
                    }
                    for (jj, &p) in s.iter().enumerate() {
                        let t = t0 + jj;
                        if t >= k_hi {
                            break;
                        }
                        let p16 = round_f16(p);
                        for (x, vv) in o.iter_mut().zip(&v16[row(t, kvh)]) {
                            *x += p16 * vv;
                        }
                    }
                    t0 += bc;
                }
                // The quad's butterfly: lane tg adds lane tg ^ 1, then
                // lane tg ^ 2's pair.
                let lsum = (l[0] + l[1]) + (l[2] + l[3]);
                let (sc, li) = match sinks {
                    Some(sk) => {
                        let sk = sk[h];
                        let mn = m.max(sk);
                        let sc = (m - mn).exp();
                        (sc, lsum * sc + (sk - mn).exp())
                    }
                    None => (1.0, lsum),
                };
                let inv = if li > 0.0 { sc / li } else { 0.0 };
                for (y, x) in out[(j * n_heads + h) * d..][..d].iter_mut().zip(&o) {
                    *y = x * inv;
                }
            }
        }
    }
    out
}

/// The largest error the kernel's f16 operands can put on an output,
/// against the definition in exact arithmetic, for `n_q` rows of `args`
/// over K/V rows `[start_pos + n_q][n_kv_heads][head_dim]`.
///
/// With `u = 2^-11`, f16's unit roundoff, and `V` the largest `|v|`,
/// three terms add up.
///
/// * The logits. `s = scale * q . k` takes two f16 roundings per
///   product and an f32 sum, so it moves by at most `delta = (2u + u^2 +
///   (2 * head_dim + 1) * 2^-24) * max sum |q_i k_i| * scale`. The
///   softcap cannot add to that, as its slope is at most 1. A softmax
///   whose logits each move by `delta` moves its weights by a ratio
///   within `e^(+-2 delta)`, which moves the weighted sum of V by at
///   most `V * (e^(2 delta) - 1)`.
/// * The operands of P * V. Each probability rounds to f16 (relative
///   `u`, or `2^-25` absolute below f16's normal range, over a row sum of
///   at least 1) and so does each value (relative `u`), which is at most
///   `V * (2u + u^2 + n_keys * 2^-25)`.
/// * The f32 accumulations, with room for the tensor cores' own add
///   order, at most `V * 2 * (n_keys + 4) * 2^-24`.
///
/// The bound is their sum with a quarter added for the second-order
/// terms each one leaves out.
pub fn f16_error_bound(q: &[f32], k: &[f32], v: &[f32], n_q: usize, args: &AttnArgs) -> f32 {
    const U: f64 = 1.0 / 2048.0;
    const E32: f64 = 1.0 / 16_777_216.0;
    let (d, gqa) = (args.head_dim, args.gqa());
    let total = args.start_pos + n_q;
    let vmax = v[..total * args.n_kv_heads * d]
        .iter()
        .fold(0f64, |a, x| a.max(f64::from(x.abs())));
    let mut qk1 = 0f64;
    for j in 0..n_q {
        for h in 0..args.n_heads {
            let qr = &q[(j * args.n_heads + h) * d..][..d];
            for t in 0..=args.start_pos + j {
                let kr = &k[(t * args.n_kv_heads + h / gqa) * d..][..d];
                let sum: f64 = qr
                    .iter()
                    .zip(kr)
                    .map(|(a, b)| f64::from((a * b).abs()))
                    .sum();
                qk1 = qk1.max(sum * f64::from(args.scale));
            }
        }
    }
    let n_keys = args.window.map_or(total, |w| w.min(total)) as f64;
    let delta = (2.0 * U + U * U + (2.0 * d as f64 + 1.0) * E32) * qk1;
    let logits = (2.0 * delta).exp_m1();
    let operands = 2.0 * U + U * U + n_keys * E32 / 2.0;
    let sums = 2.0 * (n_keys + 4.0) * E32;
    (1.25 * vmax * (logits + operands + sums)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefill_attn_tc::{heads_per_block, GROUPS};

    /// The definition, in f64: every visible key's softmax weight,
    /// the sink in the max and the sum.
    fn naive(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q: usize,
        a: &AttnArgs,
        sinks: Option<&[f32]>,
        skip: Option<(usize, usize, usize)>,
    ) -> Vec<f32> {
        let (d, gqa) = (a.head_dim, a.gqa());
        let mut out = vec![0f32; n_q * a.n_heads * d];
        for j in 0..n_q {
            let pos = a.start_pos + j;
            let lo = a.window.map_or(0, |w| (pos + 1).saturating_sub(w));
            for h in 0..a.n_heads {
                let kvh = h / gqa;
                let qr = &q[(j * a.n_heads + h) * d..][..d];
                let keys: Vec<usize> = (lo..=pos).filter(|&t| skip != Some((j, h, t))).collect();
                let s: Vec<f64> = keys
                    .iter()
                    .map(|&t| {
                        let kr = &k[(t * a.n_kv_heads + kvh) * d..][..d];
                        let dot: f64 = qr
                            .iter()
                            .zip(kr)
                            .map(|(x, y)| f64::from(*x) * f64::from(*y))
                            .sum();
                        let s = dot * f64::from(a.scale);
                        match a.softcap.filter(|c| *c > 0.0) {
                            Some(c) => f64::from(c) * (s / f64::from(c)).tanh(),
                            None => s,
                        }
                    })
                    .collect();
                let sink = sinks.map(|sk| f64::from(sk[h]));
                let m = s.iter().copied().chain(sink).fold(f64::MIN, f64::max);
                let l: f64 = s.iter().map(|x| (x - m).exp()).sum::<f64>()
                    + sink.map_or(0.0, |x| (x - m).exp());
                let o = &mut out[(j * a.n_heads + h) * d..][..d];
                for (c, oc) in o.iter_mut().enumerate() {
                    let acc: f64 = keys
                        .iter()
                        .zip(&s)
                        .map(|(&t, x)| {
                            (x - m).exp() * f64::from(v[(t * a.n_kv_heads + kvh) * d + c])
                        })
                        .sum();
                    *oc = (acc / l) as f32;
                }
            }
        }
        out
    }

    fn wave(n: usize, seed: f32, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * ((i as f32) * 0.037 + seed).sin())
            .collect()
    }

    fn worst(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn round_f16_is_cvt_rn_f16_f32() {
        let mut xs = vec![
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            65_504.0,
            65_519.0,
            65_520.0,
            1e6,
            6.0e-5,
            6.1e-5,
            5.96e-8,
            2.98e-8,
            2.99e-8,
            1e-9,
            0.1,
            1.0 + 1.0 / 2048.0,
            1.0 + 3.0 / 2048.0,
        ];
        // Random mantissas from 2^-30 (f16 subnormals and below) to
        // 2^18 (past f16's largest finite), both signs.
        let mut state = 7u32;
        for _ in 0..20_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let m = f32::from_bits(0x3f80_0000 | (state & 0x007f_ffff));
            let e = (state >> 24) % 48;
            let sign = if state & 0x0080_0000 != 0 { -1.0 } else { 1.0 };
            xs.push(sign * m * 2f32.powi(e as i32 - 30));
        }
        for x in xs {
            let want = half::f16::from_f32(x).to_f32();
            assert_eq!(round_f16(x).to_bits(), want.to_bits(), "{x:e}");
        }
        assert!(round_f16(f32::NAN).is_nan());
        assert_eq!(round_f16(f32::NEG_INFINITY), f32::NEG_INFINITY);
    }

    /// The twin is attention: for every head width, group size, GQA
    /// ratio (seven among them, which pads rows), a prefix, a window
    /// that binds, a softcap that binds and sinks, it is within the f16
    /// bound of the definition, and the bound is tight enough to mean
    /// something.
    #[test]
    fn the_twin_is_attention_within_the_f16_bound() {
        type Case = (
            usize,
            usize,
            usize,
            usize,
            usize,
            Option<usize>,
            Option<f32>,
        );
        let cases: [Case; 8] = [
            (13, 4, 4, 64, 0, None, None),
            (40, 8, 4, 64, 9, None, None),
            (70, 8, 2, 128, 0, Some(37), None),
            (33, 8, 1, 128, 70, None, Some(1.5)),
            (21, 7, 1, 64, 5, Some(9), None),
            (19, 4, 1, 256, 40, Some(45), Some(2.0)),
            (9, 2, 1, 256, 0, None, None),
            (66, 2, 2, 128, 3, None, None),
        ];
        for (n_q, n_heads, n_kv_heads, head_dim, start_pos, window, softcap) in cases {
            let total = start_pos + n_q;
            let q = wave(n_q * n_heads * head_dim, 0.2, 1.5);
            let k = wave(total * n_kv_heads * head_dim, 0.4, 1.0);
            let v = wave(total * n_kv_heads * head_dim, 0.6, 1.0);
            let args = AttnArgs {
                n_heads,
                n_kv_heads,
                head_dim,
                start_pos,
                window,
                scale: (head_dim as f32).sqrt().recip(),
                softcap,
            };
            let bound = f16_error_bound(&q, &k, &v, n_q, &args);
            assert!(bound < 0.05, "a bound of {bound} holds nothing");
            for sink in [None, Some(-1.0f32), Some(3.0)] {
                let sinks: Option<Vec<f32>> =
                    sink.map(|s| (0..n_heads).map(|h| s + 0.5 * h as f32).collect());
                let want = naive(&q, &k, &v, n_q, &args, sinks.as_deref(), None);
                let mut groups: Vec<usize> = GROUPS.to_vec();
                groups.push(heads_per_block(args.gqa(), n_q));
                for g in groups {
                    let got = prefill_attn_tc_twin(&q, &k, &v, n_q, &args, sinks.as_deref(), g);
                    let err = worst(&got, &want);
                    assert!(
                        err <= bound,
                        "n_q {n_q} hd {head_dim} gqa {} g {g} start {start_pos} window \
                         {window:?} softcap {softcap:?} sink {sink:?}: error {err} over {bound}",
                        args.gqa()
                    );
                }
            }
        }
    }

    /// One key left out of one row moves that row by ten times the bound
    /// or more (25 to 118 times, measured): the tolerance the hardware
    /// tests use can see an off-by-one in the causal or window bound. A
    /// small Q keeps the logits flat, so every key carries about 1/n of
    /// its row.
    #[test]
    fn the_f16_bound_sees_one_missing_key() {
        let (n_q, n_heads, head_dim) = (13, 2, 64);
        let q = wave(n_q * n_heads * head_dim, 0.2, 0.1);
        let k = wave(n_q * head_dim, 0.4, 1.0);
        let v = wave(n_q * head_dim, 0.6, 1.0);
        for window in [None, Some(5)] {
            let args = AttnArgs {
                n_heads,
                n_kv_heads: 1,
                head_dim,
                start_pos: 0,
                window,
                scale: 0.125,
                softcap: None,
            };
            let bound = f16_error_bound(&q, &k, &v, n_q, &args);
            let full = naive(&q, &k, &v, n_q, &args, None, None);
            // The diagonal key of the last row, and its oldest one.
            let oldest = window.map_or(0, |w| n_q - w);
            for t in [n_q - 1, oldest] {
                let short = naive(&q, &k, &v, n_q, &args, None, Some((n_q - 1, 1, t)));
                let err = worst(&short, &full);
                assert!(
                    err > 10.0 * bound,
                    "key {t}: {err} against a bound of {bound}"
                );
            }
        }
    }
}
