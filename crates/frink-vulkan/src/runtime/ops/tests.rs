//! Each op against a scalar twin written from the CPU function it
//! replaces. `frink-core`'s seam tests hold the same ops to those
//! functions themselves.

use super::*;
use crate::dispatch::tests::{kind_row, pseudo_random, runtime, twin};
use crate::shaders::matvec_shader;

fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}[{i}]: gpu {g} vs twin {w}"
        );
    }
}

fn upload(rt: &mut Runtime, data: &[f32]) -> DeviceBuf {
    let b = rt.alloc_f32(data.len()).unwrap();
    rt.write_f32(b, 0, data).unwrap();
    b
}

/// ggml's RMSNorm (`ggml_compute_forward_rms_norm_f32`): squares summed
/// in order in f64.
fn rms_norm_twin(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum: f64 = x.iter().map(|v| (v * v) as f64).sum();
    let scale = 1.0 / ((sum / x.len() as f64) as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * scale * w).collect()
}

#[test]
fn gpu_rms_norm_matches_its_twin_per_row() {
    let Some(mut rt) = runtime() else { return };
    // One 2048-wide row (a hidden state) and eight 128-wide heads
    // sharing one weight (`rms_norm_per_head`).
    for (n, rows) in [(2048usize, 1usize), (128, 8), (5, 3)] {
        let x = pseudo_random(n as u64 * 7 + rows as u64, n * rows);
        let w: Vec<f32> = pseudo_random(99, n).iter().map(|v| 1.0 + v).collect();
        let (xb, wb, ob) = (
            upload(&mut rt, &x),
            upload(&mut rt, &w),
            rt.alloc_f32(n * rows).unwrap(),
        );
        let mut batch = Batch::new();
        batch.rms_norm(wb, xb, ob, n, rows, 1e-5);
        rt.submit_batch(&batch).unwrap();
        let want: Vec<f32> = x
            .chunks(n)
            .flat_map(|r| rms_norm_twin(r, &w, 1e-5))
            .collect();
        close(
            &format!("rms_norm {n}x{rows}"),
            &rt.read_f32(ob, 0, n * rows).unwrap(),
            &want,
            1e-5,
        );
    }
}

#[test]
fn gpu_add_matches_its_twin() {
    let Some(mut rt) = runtime() else { return };
    let n = 1000;
    let (a, b) = (pseudo_random(1, n), pseudo_random(2, n));
    let (ab, bb, ob) = (
        upload(&mut rt, &a),
        upload(&mut rt, &b),
        rt.alloc_f32(n).unwrap(),
    );
    let mut batch = Batch::new();
    batch.add(ab, bb, ob, n);
    rt.submit_batch(&batch).unwrap();
    let want: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
    assert_eq!(rt.read_f32(ob, 0, n).unwrap(), want, "an add is exact");
}

/// A bias is added to every position's row, and the add is exact.
#[test]
fn gpu_a_bias_is_added_to_every_position() {
    let Some(mut rt) = runtime() else { return };
    let (n, positions) = (130usize, 7usize);
    let a = pseudo_random(3, n * positions);
    let bias = pseudo_random(4, n);
    let (ab, bb, ob) = (
        upload(&mut rt, &a),
        upload(&mut rt, &bias),
        rt.alloc_f32(n * positions).unwrap(),
    );
    let mut batch = Batch::for_positions(positions);
    batch.add_bias(ab, bb, ob, n);
    rt.submit_batch(&batch).unwrap();
    let want: Vec<f32> = a.iter().enumerate().map(|(i, a)| a + bias[i % n]).collect();
    assert_eq!(rt.read_f32(ob, 0, n * positions).unwrap(), want);
}

/// `frink_core::attention::apply_rope_interleaved` and `apply_rope`
/// (NeoX) over the first `rot_dim` channels, from the same table.
fn rope_twin(x: &[f32], head_dim: usize, rot_dim: usize, tab: &[f32], neox: bool) -> Vec<f32> {
    let mut out = x.to_vec();
    let half = rot_dim / 2;
    for head in out.chunks_mut(head_dim) {
        for k in 0..half {
            let (cs, sn) = (tab[2 * k], tab[2 * k + 1]);
            let (a, b) = if neox {
                (k, k + half)
            } else {
                (2 * k, 2 * k + 1)
            };
            let (xa, xb) = (head[a], head[b]);
            head[a] = xa * cs - xb * sn;
            head[b] = xa * sn + xb * cs;
        }
    }
    out
}

#[test]
fn gpu_rope_matches_its_twin_for_both_pairings_and_a_partial_width() {
    let Some(mut rt) = runtime() else { return };
    let (n_heads, head_dim) = (6usize, 64usize);
    for (rot_dim, neox) in [(64usize, false), (64, true), (32, false), (48, true)] {
        let pos = 4093.0f32;
        let tab: Vec<f32> = (0..rot_dim / 2)
            .flat_map(|k| {
                let (s, c) = (pos / 10000f32.powf(2.0 * k as f32 / rot_dim as f32)).sin_cos();
                [c, s]
            })
            .collect();
        let x = pseudo_random(rot_dim as u64 + neox as u64, n_heads * head_dim);
        let (tb, xb, ob) = (
            upload(&mut rt, &tab),
            upload(&mut rt, &x),
            rt.alloc_f32(x.len()).unwrap(),
        );
        let mut batch = Batch::new();
        batch.rope(tb, xb, ob, n_heads, head_dim, rot_dim, neox);
        rt.submit_batch(&batch).unwrap();
        let want = rope_twin(&x, head_dim, rot_dim, &tab, neox);
        close(
            &format!("rope rot {rot_dim} neox {neox}"),
            &rt.read_f32(ob, 0, x.len()).unwrap(),
            &want,
            1e-6,
        );
    }
}

/// The f16 store, bit for bit against `half` (round to nearest even, as
/// ggml's F16C conversion is), over values that hit every branch: ties,
/// subnormals, overflow, signed zero, NaN.
#[test]
fn gpu_kv_store_rounds_to_f16_as_the_cpu_does() {
    let Some(mut rt) = runtime() else { return };
    let mut src = pseudo_random(5, 254);
    src.extend([
        1.0 + 2f32.powi(-11),       // a tie, rounds to even (down)
        1.0 + 3.0 * 2f32.powi(-11), // a tie, rounds to even (up)
        6.0e-8,                     // f16 subnormal
        -2.9e-5,                    // f16 subnormal, negative
        1e-9,                       // underflows to zero
        65520.0,                    // rounds up to inf
        -0.0,
        f32::NAN,
        65504.0,        // max f16
        2f32.powi(-14), // min normal
    ]);
    let n = src.len();
    let sb = upload(&mut rt, &src);
    let cache_words = 3 * n / 2;
    let cache = rt.alloc_f32(cache_words).unwrap();
    let mut batch = Batch::new();
    batch.kv_store(sb, cache, n, n / 2, cache_words);
    rt.submit_batch(&batch).unwrap();
    let got = rt.read_f32(cache, n / 2, n / 2).unwrap();
    for (i, pair) in src.chunks(2).enumerate() {
        let w = got[i].to_bits();
        for (half_i, v) in pair.iter().enumerate() {
            let bits = (w >> (16 * half_i)) as u16;
            let want = half::f16::from_f32(*v).to_bits();
            if v.is_nan() {
                assert!(half::f16::from_bits(bits).is_nan(), "NaN stays NaN");
            } else {
                assert_eq!(bits, want, "value {v}: gpu {bits:#06x} vs {want:#06x}");
            }
        }
    }
    let before = rt.read_f32(cache, 0, n / 2).unwrap();
    assert!(
        before.iter().all(|v| v.to_bits() == 0),
        "words before `dst` untouched"
    );
}

/// `causal_gqa_attention_row` without window, sinks, softcap or ALiBi,
/// over K and V as the f16 cache holds them.
fn attention_twin(q: &[f32], k: &[f32], v: &[f32], s: AttnShape) -> Vec<f32> {
    let group = s.n_heads / s.n_kv_heads;
    let mut out = vec![0f32; s.n_heads * s.head_dim];
    for h in 0..s.n_heads {
        let kvh = h / group;
        let qh = &q[h * s.head_dim..(h + 1) * s.head_dim];
        let at = |t: usize| (t * s.n_kv_heads + kvh) * s.head_dim;
        let scores: Vec<f32> = (0..s.seq_len)
            .map(|t| {
                qh.iter()
                    .zip(&k[at(t)..at(t) + s.head_dim])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * s.scale
            })
            .collect();
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let p: Vec<f32> = scores.iter().map(|x| (x - m).exp()).collect();
        let l: f32 = p.iter().sum();
        for d in 0..s.head_dim {
            out[h * s.head_dim + d] = (0..s.seq_len).map(|t| p[t] * v[at(t) + d]).sum::<f32>() / l;
        }
    }
    out
}

fn f16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| half::f16::from_f32(*x).to_f32()).collect()
}

/// The KV append and attention chained in one submission: every
/// position stored through `kv_store`, then attention over all of them,
/// for MHA, GQA (4:1) and MQA, at a length past one workgroup.
#[test]
fn gpu_attention_over_the_f16_cache_matches_its_twin() {
    let Some(mut rt) = runtime() else { return };
    for (n_heads, n_kv_heads, head_dim, seq_len) in [
        (4usize, 4usize, 64usize, 7usize),
        (8, 2, 128, 130),
        (6, 1, 32, 65),
    ] {
        let row = n_kv_heads * head_dim;
        let k = pseudo_random(11 + seq_len as u64, seq_len * row);
        let v = pseudo_random(23 + seq_len as u64, seq_len * row);
        let q = pseudo_random(37 + seq_len as u64, n_heads * head_dim);
        // K at word 0, V after it: halves, two per word.
        let half_words = seq_len * row / 2;
        let cache_words = 2 * half_words;
        let cache = rt.alloc_f32(cache_words).unwrap();
        let (kb, vb, qb) = (
            upload(&mut rt, &k),
            upload(&mut rt, &v),
            upload(&mut rt, &q),
        );
        let scores = rt.alloc_f32(n_heads * seq_len).unwrap();
        let out = rt.alloc_f32(n_heads * head_dim).unwrap();
        let shape = AttnShape {
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            k_off: 0,
            v_off: half_words,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };
        let mut batch = Batch::new();
        batch.kv_store(kb, cache, seq_len * row, 0, cache_words);
        batch.kv_store(vb, cache, seq_len * row, half_words, cache_words);
        batch.attention(cache, cache_words, qb, scores, out, shape);
        let before = rt.submissions();
        rt.submit_batch(&batch).unwrap();
        assert_eq!(
            rt.submissions() - before,
            1,
            "store and attend in one submission"
        );
        let want = attention_twin(&q, &f16_round(&k), &f16_round(&v), shape);
        close(
            &format!("attention {n_heads}/{n_kv_heads}x{head_dim} over {seq_len}"),
            &rt.read_f32(out, 0, n_heads * head_dim).unwrap(),
            &want,
            1e-4,
        );
    }
}

/// Ops feed each other within one submission: norm, a matvec of the
/// normed vector, and a residual add, against the same three steps on
/// the host.
#[test]
fn gpu_a_batch_chains_norm_matvec_and_residual_in_one_submission() {
    let Some(mut rt) = runtime() else { return };
    let (cols, rows) = (256usize, 256usize);
    let s = matvec_shader("Q8_0").unwrap();
    let blocks = cols / s.block_elems;
    let weights: Vec<u8> = (0..rows)
        .flat_map(|r| kind_row("Q8_0", blocks, 5 + r as u64))
        .collect();
    let x = pseudo_random(77, cols);
    let w: Vec<f32> = pseudo_random(78, cols).iter().map(|v| 1.0 + v).collect();
    let (xb, wb) = (upload(&mut rt, &x), upload(&mut rt, &w));
    let (normed, proj, out) = (
        rt.alloc_f32(cols).unwrap(),
        rt.alloc_f32(rows).unwrap(),
        rt.alloc_f32(rows).unwrap(),
    );
    let mut batch = Batch::new();
    batch.rms_norm(wb, xb, normed, cols, 1, 1e-6);
    batch.matvec(
        MatvecJob {
            shader: s,
            weights: &weights,
            rows,
        },
        (normed, 0),
        (proj, 0),
    );
    batch.add(xb, proj, out, rows);
    let before = rt.submissions();
    rt.submit_batch(&batch).unwrap();
    assert_eq!(rt.submissions() - before, 1);

    let n = rms_norm_twin(&x, &w, 1e-6);
    let p: Vec<f32> = (0..rows)
        .map(|r| twin("Q8_0")(&weights[r * blocks * 34..(r + 1) * blocks * 34], &n))
        .collect();
    let want: Vec<f32> = x.iter().zip(&p).map(|(a, b)| a + b).collect();
    close(
        "norm -> matvec -> add",
        &rt.read_f32(out, 0, rows).unwrap(),
        &want,
        1e-4,
    );
}

#[test]
fn gpu_a_view_past_its_buffer_is_an_error_not_a_dispatch() {
    let Some(mut rt) = runtime() else { return };
    let (a, out) = (rt.alloc_f32(8).unwrap(), rt.alloc_f32(16).unwrap());
    let mut batch = Batch::new();
    batch.add(a, a, out, 16);
    assert!(rt.submit_batch(&batch).is_err());
    assert!(rt.submit_batch(&Batch::new()).is_err(), "an empty batch");
    assert!(rt.write_f32(a, 4, &[0.0; 5]).is_err());
}

#[test]
fn gpu_a_batched_swiglu_is_silu_gate_times_up() {
    let Some(mut rt) = runtime() else { return };
    let n = 300;
    let g: Vec<f32> = (0..n).map(|i| (i as f32 - 150.0) * 0.07).collect();
    let u: Vec<f32> = (0..n).map(|i| ((i * 7 % 31) as f32 - 15.0) * 0.3).collect();
    let (gb, ub, out) = (
        rt.alloc_f32(n).unwrap(),
        rt.alloc_f32(n).unwrap(),
        rt.alloc_f32(n).unwrap(),
    );
    rt.write_f32(gb, 0, &g).unwrap();
    rt.write_f32(ub, 0, &u).unwrap();
    let mut batch = Batch::new();
    batch.swiglu(gb, ub, out, n);
    rt.submit_batch(&batch).unwrap();
    let got = rt.read_f32(out, 0, n).unwrap();
    for i in 0..n {
        let want = g[i] / (1.0 + (-g[i]).exp()) * u[i];
        assert!(
            (got[i] - want).abs() <= 1e-5 * want.abs().max(1.0),
            "{i}: {} vs {want}",
            got[i]
        );
    }
}

#[test]
fn gpu_scratch_is_reused_until_it_is_too_small() {
    let Some(mut rt) = runtime() else { return };
    let a = rt.scratch("a", 0, 100).unwrap();
    assert_eq!(rt.scratch("a", 0, 100).unwrap(), a);
    assert_eq!(rt.scratch("a", 0, 128).unwrap(), a, "100 rounds up to 128");
    assert_ne!(rt.scratch("a", 1, 100).unwrap(), a, "another index");
    assert_ne!(rt.scratch("b", 0, 100).unwrap(), a, "another name");
    let grown = rt.scratch("a", 0, 129).unwrap();
    assert_ne!(grown, a);
    assert!(rt.capacity_f32(grown) >= 129);
    assert!(rt.write_f32(grown, 0, &[1.0; 129]).is_ok());
}

#[test]
fn gpu_a_constant_is_uploaded_once_and_again_when_its_bytes_change() {
    let Some(mut rt) = runtime() else { return };
    let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
    let first = rt.constant_f32(&data).unwrap();
    assert_eq!(rt.constant_f32(&data).unwrap(), first);
    assert_eq!(rt.read_f32(first, 0, 4).unwrap(), data);
    // The same address with other bytes is another constant.
    data[2] = -3.0;
    let second = rt.constant_f32(&data).unwrap();
    assert_eq!(rt.read_f32(second, 0, 4).unwrap(), data);
}

#[test]
fn gpu_a_queued_free_reaches_only_the_runtime_that_owns_the_buffer() {
    let (Some(mut a), Some(mut b)) = (runtime(), runtime()) else {
        return;
    };
    assert_ne!(a.id(), b.id());
    // The same index in both runtimes.
    let (in_a, in_b) = (a.alloc_f32(8).unwrap(), b.alloc_f32(8).unwrap());
    assert_eq!(in_a, in_b);
    b.write_f32(in_b, 0, &[7.0; 8]).unwrap();
    free_later(a.id(), in_a);
    // `b` drains the queue and must leave `a`'s entry on it.
    let _ = b.submit_batch(&Batch::new());
    assert_eq!(b.read_f32(in_b, 0, 8).unwrap(), [7.0; 8]);
    assert!(a.write_f32(in_a, 0, &[1.0; 8]).is_ok(), "not freed yet");
    // `a` frees it at its own next submission.
    let _ = a.submit_batch(&Batch::new());
    let freed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.read_f32(in_a, 0, 8)));
    assert!(freed.is_err(), "a's buffer was not freed by a's submission");
}

/// A batch recorded for several positions is each position recorded
/// alone, bit for bit: RoPE with a table per position, the KV append,
/// causal attention (position z sees the prefix and positions up to its
/// own), a per-row norm and a matvec, the ops a prefill layer chains.
/// The prefix is 70 positions, so a later position's scores run past a
/// workgroup.
#[test]
fn gpu_a_batch_of_positions_is_each_position_alone() {
    let Some(mut rt) = runtime() else { return };
    let (n_heads, n_kv, hd, rot, prefix, positions) =
        (4usize, 2usize, 8usize, 6usize, 70usize, 5usize);
    let (qn, kn) = (n_heads * hd, n_kv * hd);
    let capacity = prefix + positions;
    let half_words = capacity * kn / 2;
    let cache_words = 2 * half_words;
    let shader = matvec_shader("Q8_0").unwrap();
    let rows = 7;
    let weights: Vec<u8> = (0..rows)
        .flat_map(|r| kind_row("Q8_0", 1, r as u64 + 3))
        .collect();
    let job = MatvecJob {
        shader,
        weights: &weights,
        rows,
    };
    let norm_w: Vec<f32> = pseudo_random(5, hd).iter().map(|v| 1.0 + v).collect();
    let q = pseudo_random(41, positions * qn);
    let k = pseudo_random(43, positions * kn);
    let v = pseudo_random(47, positions * kn);
    let tab = pseudo_random(53, positions * rot);
    let old_k = pseudo_random(59, prefix * kn);
    let old_v = pseudo_random(61, prefix * kn);

    // One run of `p` positions starting at batch position `at`, over a
    // cache holding the prefix and the positions before `at`. Answers
    // the attention output, its norm, and the matvec of the norm.
    let run = |rt: &mut Runtime, cache: DeviceBuf, at: usize, p: usize| {
        let s = |x: &[f32], w: usize| x[at * w..(at + p) * w].to_vec();
        let (qb, kb, vb, tb) = (
            upload(rt, &s(&q, qn)),
            upload(rt, &s(&k, kn)),
            upload(rt, &s(&v, kn)),
            upload(rt, &s(&tab, rot)),
        );
        let wb = upload(rt, &norm_w);
        let seq = prefix + at + 1;
        let [qr, kr, out, normed] = [qn, kn, qn, qn].map(|n| rt.alloc_f32(p * n).unwrap());
        let scores = rt.alloc_f32(p * n_heads * (seq + p - 1)).unwrap();
        let y = rt.alloc_f32(p * rows).unwrap();
        let mut batch = Batch::for_positions(p);
        batch.rope(tb, qb, qr, n_heads, hd, rot, false);
        batch.rope(tb, kb, kr, n_kv, hd, rot, false);
        batch.kv_store(kr, cache, kn, (prefix + at) * kn / 2, cache_words);
        batch.kv_store(
            vb,
            cache,
            kn,
            half_words + (prefix + at) * kn / 2,
            cache_words,
        );
        batch.attention(
            cache,
            cache_words,
            qr,
            scores,
            out,
            AttnShape {
                n_heads,
                n_kv_heads: n_kv,
                head_dim: hd,
                seq_len: seq,
                k_off: 0,
                v_off: half_words,
                scale: 1.0 / (hd as f32).sqrt(),
            },
        );
        batch.rms_norm(wb, out, normed, hd, n_heads, 1e-5);
        batch.matvec(job, (normed, 0), (y, 0));
        let before = rt.submissions();
        rt.submit_batch(&batch).unwrap();
        assert_eq!(rt.submissions() - before, 1);
        let bits = |b: DeviceBuf, n: usize| -> Vec<u32> {
            rt.read_f32(b, 0, n)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect()
        };
        (bits(out, p * qn), bits(normed, p * qn), bits(y, p * rows))
    };
    let cache_with_prefix = |rt: &mut Runtime| {
        let cache = rt.alloc_f32(cache_words).unwrap();
        let (kb, vb) = (upload(rt, &old_k), upload(rt, &old_v));
        let mut batch = Batch::new();
        batch.kv_store(kb, cache, prefix * kn, 0, cache_words);
        batch.kv_store(vb, cache, prefix * kn, half_words, cache_words);
        rt.submit_batch(&batch).unwrap();
        cache
    };

    let together = cache_with_prefix(&mut rt);
    let (out, normed, y) = run(&mut rt, together, 0, positions);
    let alone = cache_with_prefix(&mut rt);
    for z in 0..positions {
        let (o1, n1, y1) = run(&mut rt, alone, z, 1);
        assert_eq!(out[z * qn..(z + 1) * qn], o1[..], "attention, position {z}");
        assert_eq!(normed[z * qn..(z + 1) * qn], n1[..], "norm, position {z}");
        assert_eq!(y[z * rows..(z + 1) * rows], y1[..], "matvec, position {z}");
    }
    // The two caches hold the same rows whichever way they were filled.
    assert_eq!(
        rt.read_u32(together, 0, cache_words).unwrap(),
        rt.read_u32(alone, 0, cache_words).unwrap()
    );
    // And a later position did not just repeat the first.
    assert_ne!(out[..qn], out[qn..2 * qn]);
}
