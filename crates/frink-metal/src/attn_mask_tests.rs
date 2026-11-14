//! Attention sinks and the prefill sliding window, on every Metal
//! attention kernel, against a scalar reference.
//!
//! gpt-oss gives every head a sink (one more logit in its softmax, with
//! no value) and slides a 128-key window over every other layer. Each
//! kernel that serves a decode or a prefill takes both
//! ([`crate::attn::AttnMask`]); this suite holds each one to the
//! reference on its own, by naming the kernel, so a kernel that drops
//! either is caught here and not as a drift in a model's logits.

use crate::attn::{
    launch_gqa_decode_host_sinks, launch_gqa_prefill_host_masked, PrefillAttnKernel,
};

/// One query row against `[0, seq_len)` with the first `lo` keys hidden,
/// and `sink` (if any) in the denominator: llama.cpp's
/// `ggml_compute_forward_flash_attn_ext_f16` sinks block.
#[allow(clippy::too_many_arguments)]
fn reference_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv: usize,
    d: usize,
    lo: usize,
    seq_len: usize,
    sinks: Option<&[f32]>,
) -> Vec<f32> {
    let group = n_heads / n_kv;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0f32; n_heads * d];
    for h in 0..n_heads {
        let kv_h = h / group;
        let qh = &q[h * d..(h + 1) * d];
        let scores: Vec<f32> = (lo..seq_len)
            .map(|t| {
                let kt = &k[(t * n_kv + kv_h) * d..(t * n_kv + kv_h + 1) * d];
                qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale
            })
            .collect();
        let mut m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if let Some(s) = sinks {
            m = m.max(s[h]);
        }
        let mut denom: f32 = scores.iter().map(|s| (s - m).exp()).sum();
        if let Some(s) = sinks {
            denom += (s[h] - m).exp();
        }
        let oh = &mut out[h * d..(h + 1) * d];
        for (i, t) in (lo..seq_len).enumerate() {
            let p = (scores[i] - m).exp() / denom;
            let vt = &v[(t * n_kv + kv_h) * d..(t * n_kv + kv_h + 1) * d];
            for (o, x) in oh.iter_mut().zip(vt) {
                *o += p * x;
            }
        }
    }
    out
}

fn fill(n: usize, step: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * step).sin() * scale).collect()
}

/// Sinks well inside the score range, one per head and all different:
/// big enough to take a third or more of the mass, so a kernel that
/// ignored them, or read another head's, is off by far more than the
/// tolerance.
fn sinks(n_heads: usize) -> Vec<f32> {
    (0..n_heads).map(|h| 2.0 + 0.5 * h as f32).collect()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 3e-3 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{what} elem {i}: gpu {g} vs {w}");
    }
}

/// Decode: the FA-vec kernel at every width it has, and the legacy one
/// (d=32 has no FA-vec kernel), each with sinks, with and without a
/// window start.
#[test]
#[ignore = "needs a real Metal GPU"]
fn decode_kernels_apply_sinks() {
    for (n_heads, n_kv, d, seq) in [
        (8, 2, 64, 70),
        (4, 2, 96, 40),
        (8, 4, 128, 65),
        (4, 1, 256, 33),
        (4, 2, 512, 45),
        (4, 2, 32, 37),
    ] {
        let q = fill(n_heads * d, 0.07, 1.0);
        let k = fill(seq * n_kv * d, 0.031, 1.0);
        // Offset so every head's output is order one, and the mass the
        // sink takes away shows in it.
        let v: Vec<f32> = fill(seq * n_kv * d, 0.053, 1.0)
            .into_iter()
            .map(|x| x + 1.0)
            .collect();
        let s = sinks(n_heads);
        for kv_start in [0, seq / 3] {
            let want = reference_row(&q, &k, &v, n_heads, n_kv, d, kv_start, seq, Some(&s));
            let plain = reference_row(&q, &k, &v, n_heads, n_kv, d, kv_start, seq, None);
            let got = launch_gqa_decode_host_sinks(
                &q,
                &k,
                &v,
                n_heads,
                n_kv,
                d,
                seq,
                kv_start,
                None,
                Some(&s),
            )
            .expect("decode with sinks");
            assert_close(&got, &want, &format!("decode d={d} kv_start={kv_start}"));
            let moved = got
                .iter()
                .zip(&plain)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            // Ten times the tolerance: dropping the sinks is not noise.
            assert!(moved > 3e-2, "d={d}: the sinks moved nothing ({moved})");
        }
    }
}

/// Prefill: every kernel the dispatch can pick, named, with sinks and a
/// window smaller than both the prefix and the batch, so the window
/// cuts inside the batch and inside the cache.
#[test]
#[ignore = "needs a real Metal GPU"]
fn prefill_kernels_apply_sinks_and_the_window() {
    let kernels = [
        PrefillAttnKernel::FaVec,
        PrefillAttnKernel::FaExt,
        PrefillAttnKernel::Legacy,
        PrefillAttnKernel::Auto,
    ];
    for (n_heads, n_kv, d) in [
        (8, 2, 64),
        (4, 2, 96),
        (4, 1, 128),
        (4, 2, 256),
        (4, 2, 512),
        (4, 2, 32),
    ] {
        for (prefix, n_q, window) in [(0, 40, 0), (19, 45, 16), (70, 33, 128), (5, 96, 7)] {
            let total = prefix + n_q;
            let q = fill(n_q * n_heads * d, 0.071, 1.0);
            let k = fill(total * n_kv * d, 0.029, 1.0);
            let v: Vec<f32> = fill(total * n_kv * d, 0.047, 1.0)
                .into_iter()
                .map(|x| x + 1.0)
                .collect();
            let s = sinks(n_heads);
            let (mut want, mut plain) = (Vec::new(), Vec::new());
            for qi in 0..n_q {
                let seq = prefix + qi + 1;
                let lo = if window > 0 {
                    seq.saturating_sub(window)
                } else {
                    0
                };
                let row = &q[qi * n_heads * d..(qi + 1) * n_heads * d];
                want.extend(reference_row(
                    row,
                    &k,
                    &v,
                    n_heads,
                    n_kv,
                    d,
                    lo,
                    seq,
                    Some(&s),
                ));
                plain.extend(reference_row(row, &k, &v, n_heads, n_kv, d, 0, seq, None));
            }
            // The case is one a kernel that dropped either would fail.
            let moved = want
                .iter()
                .zip(&plain)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                moved > 3e-2,
                "d={d}: sinks and window moved nothing ({moved})"
            );
            let mut ks: Vec<PrefillAttnKernel> = kernels.to_vec();
            if crate::tensor_mm::has_tensor() && crate::fa_tensor::supported(d as u32) {
                ks.push(PrefillAttnKernel::Tensor);
            }
            for kernel in ks {
                let got = launch_gqa_prefill_host_masked(
                    &q,
                    &k,
                    &v,
                    n_heads,
                    n_kv,
                    d,
                    n_q,
                    prefix,
                    None,
                    kernel,
                    Some(&s),
                    (window > 0).then_some(window),
                )
                .expect("prefill with sinks and window");
                assert_close(
                    &got,
                    &want,
                    &format!("prefill {kernel:?} d={d} prefix={prefix} n_q={n_q} window={window}"),
                );
            }
        }
    }
}
