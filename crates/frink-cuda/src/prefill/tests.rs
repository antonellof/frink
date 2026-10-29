//! Each kernel against a naive host loop of the same arithmetic, then
//! the whole layer against a host twin built from those loops and
//! [`crate::mul_mm_ref::mul_mm_reference`]. Everything that needs a
//! device is `#[ignore]`d; run it with
//! `cargo test -p frink-cuda --features cuda -- --ignored` on real
//! hardware and write down what happened in the `ignore` text.

use super::*;
use crate::gpu::shared_device;
use crate::mul_mm::Q8_0;
use crate::mul_mm_ref::{fixtures, mul_mm_reference};

fn host_rms_norm_rows(x: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
    x.chunks_exact(n)
        .flat_map(|row| {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
            let s = 1.0 / (ms + eps).sqrt();
            row.iter().zip(w).map(move |(v, w)| v * s * w)
        })
        .collect()
}

/// `Decoder::apply_rope_head_theta` + `apply_rope_attn_factor`, naive.
#[allow(clippy::too_many_arguments)]
fn host_rope_rows(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rot_dim: usize,
    start_pos: usize,
    theta: f32,
    ff: Option<&[f32]>,
    neox: bool,
    mscale: f32,
) {
    let half = rot_dim / 2;
    for (r, row) in x.chunks_exact_mut(n_heads * head_dim).enumerate() {
        let pos = (start_pos + r) as f32;
        for head in row.chunks_exact_mut(head_dim) {
            for i in 0..half {
                let freq = 1.0 / theta.powf((2 * i) as f32 / rot_dim as f32);
                let mut angle = pos * freq;
                if let Some(ff) = ff {
                    angle /= ff[i];
                }
                let (s, c) = angle.sin_cos();
                let (ia, ib) = if neox {
                    (i, i + half)
                } else {
                    (2 * i, 2 * i + 1)
                };
                let a = head[ia] * mscale;
                let b = head[ib] * mscale;
                head[ia] = a * c - b * s;
                head[ib] = a * s + b * c;
            }
        }
    }
}

/// `frink_core::attention::causal_gqa_attention_row` per query, naive.
fn host_causal_gqa(q: &[f32], k: &[f32], v: &[f32], n_q: usize, a: &AttnArgs) -> Vec<f32> {
    let hd = a.head_dim;
    let group = a.n_heads / a.n_kv_heads.max(1);
    let mut out = vec![0f32; n_q * a.n_heads * hd];
    for r in 0..n_q {
        let seq_len = a.start_pos + r + 1;
        let lo = a.window.map_or(0, |w| seq_len.saturating_sub(w));
        for h in 0..a.n_heads {
            let kv_h = h / group.max(1);
            let q_h = &q[(r * a.n_heads + h) * hd..(r * a.n_heads + h + 1) * hd];
            let mut scores: Vec<f32> = (lo..seq_len)
                .map(|t| {
                    let k_t =
                        &k[(t * a.n_kv_heads + kv_h) * hd..(t * a.n_kv_heads + kv_h + 1) * hd];
                    let mut s = q_h.iter().zip(k_t).map(|(a, b)| a * b).sum::<f32>() * a.scale;
                    if let Some(c) = a.softcap.filter(|c| *c > 0.0) {
                        s = c * (s / c).tanh();
                    }
                    s
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut l = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                l += *s;
            }
            let o = &mut out[(r * a.n_heads + h) * hd..(r * a.n_heads + h + 1) * hd];
            for (ti, t) in (lo..seq_len).enumerate() {
                let v_t = &v[(t * a.n_kv_heads + kv_h) * hd..(t * a.n_kv_heads + kv_h + 1) * hd];
                for d in 0..hd {
                    o[d] += scores[ti] / l * v_t[d];
                }
            }
        }
    }
    out
}

/// A Q8_0 projection with both views, as `frink-core` builds one:
/// the GEMM's and the matvec's, over the same bytes.
pub(super) fn q8_0_proj(data: &'static [u8], rows: usize, cols: usize) -> ProjWeights<'static> {
    let (kernel_src, module_name, fn_name) =
        crate::gpu::matvec_launch_meta("Q8_0").expect("Q8_0 has a matvec");
    ProjWeights {
        gemm: MulMmWeights {
            kind: &Q8_0,
            data,
            rows,
            cols,
            row_bytes: cols / 32 * 34,
        },
        matvec: Some(crate::gpu::MatvecLaunch {
            kernel_src,
            module_name,
            fn_name,
            weights: data,
            rows,
            row_bytes: cols / 32 * 34,
            n_blocks_per_row: cols / 32,
        }),
    }
}

pub(super) fn wave(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.037 + seed).sin()).collect()
}

/// `|got - want| <= tol * max(|want|, rms(want))` per element: the
/// floor is the vector's own scale, not 1.0, so a small element of a
/// vector whose terms cancel (a residual stream after two adds) is
/// held to the same absolute error as its neighbours rather than to a
/// relative one it cannot meet through f16 operands.
fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let rms = (want.iter().map(|w| w * w).sum::<f32>() / want.len().max(1) as f32).sqrt();
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(rms),
            "{what}: element {i}: GPU={g} host={w} (rms {rms})"
        );
    }
}

#[test]
fn a_layer_whose_shapes_disagree_is_refused_before_the_device() {
    // `o.cols` is not `n_heads * head_dim`: refused by the host check,
    // which on a GPU-less host is the only way this test can pass.
    let m = |rows, cols| {
        ProjWeights::gemm_only(MulMmWeights {
            kind: &Q8_0,
            data: &[],
            rows,
            cols,
            row_bytes: cols / 32 * 34,
        })
    };
    let w = vec![1.0f32; 64];
    let layer = PrefillDenseLayerCuda {
        attn_norm_w: &w,
        ffn_norm_w: &w,
        q: m(64, 64),
        k: m(32, 64),
        v: m(32, 64),
        o: m(64, 96),
        gate: m(96, 64),
        up: m(96, 64),
        down: m(64, 96),
        post_attn_norm: None,
        post_ffn_norm: None,
        extras: AttnExtrasCuda::default(),
        rope: None,
    };
    let params = PrefillParams {
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        rms_eps: 1e-5,
        attn_scale: 32f32.sqrt().recip(),
        attn_softcap: None,
        window: None,
        prefix_k: &[],
        prefix_v: &[],
        start_pos: 0,
    };
    let err = launch_prefill_dense_layer(&vec![0.0; 4 * 64], &layer, &params, 4)
        .err()
        .expect("a disagreeing shape must be refused");
    assert!(matches!(err, CudaError::Unsupported(_)), "{err:?}");
}

#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn rmsnorm_rows_matches_the_host() {
    let dev = shared_device().unwrap();
    let (rows, n) = (37, 96);
    let x = wave(rows * n, 0.1);
    let w = wave(n, 0.7);
    let d_x = dev.htod_copy(x.clone()).unwrap();
    let d_w = dev.htod_copy(w.clone()).unwrap();
    let out = enqueue_rmsnorm_rows(&dev, &d_x, &d_w, rows, n, 1e-5).unwrap();
    let got = dev.dtoh_sync_copy(&out).unwrap();
    assert_close(&got, &host_rms_norm_rows(&x, &w, n, 1e-5), 1e-5, "rmsnorm");
}

#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn rope_rows_matches_the_host_in_both_layouts_with_a_partial_width_and_divisors() {
    let dev = shared_device().unwrap();
    let (rows, n_heads, head_dim) = (9, 3, 32);
    let ff: Vec<f32> = (0..16).map(|i| 1.0 + i as f32 * 0.25).collect();
    for (neox, rot_dim, use_ff, mscale) in [
        (false, 32, false, 1.0),
        (true, 32, false, 1.0),
        (false, 24, true, 1.19),
        (true, 24, true, 1.0),
    ] {
        let x = wave(rows * n_heads * head_dim, 0.3);
        let mut want = x.clone();
        host_rope_rows(
            &mut want,
            n_heads,
            head_dim,
            rot_dim,
            5,
            10000.0,
            use_ff.then_some(&ff[..rot_dim / 2]),
            neox,
            mscale,
        );
        let mut d_x = dev.htod_copy(x).unwrap();
        let d_ff = use_ff.then(|| dev.htod_copy(ff[..rot_dim / 2].to_vec()).unwrap());
        enqueue_rope_rows(
            &dev,
            &mut d_x,
            rows,
            n_heads,
            head_dim,
            5,
            &RopeArgs {
                theta: 10000.0,
                freq_factors: d_ff.as_ref(),
                rot_dim,
                neox,
                mscale,
            },
        )
        .unwrap();
        let got = dev.dtoh_sync_copy(&d_x).unwrap();
        assert_close(
            &got,
            &want,
            2e-5,
            &format!("rope neox={neox} rot={rot_dim} ff={use_ff}"),
        );
    }
}

#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn causal_gqa_prefill_matches_the_host_with_a_prefix_a_window_and_a_softcap() {
    let dev = shared_device().unwrap();
    let (n_q, n_heads, n_kv_heads, head_dim) = (13, 4, 2, 64);
    for (start_pos, window, softcap) in [
        (0, None, None),
        (7, None, None),
        (7, Some(5), None),
        (0, None, Some(50.0)),
    ] {
        let total = start_pos + n_q;
        let q = wave(n_q * n_heads * head_dim, 0.2);
        let k = wave(total * n_kv_heads * head_dim, 0.4);
        let v = wave(total * n_kv_heads * head_dim, 0.6);
        let args = AttnArgs {
            n_heads,
            n_kv_heads,
            head_dim,
            start_pos,
            window,
            scale: (head_dim as f32).sqrt().recip(),
            softcap,
        };
        let want = host_causal_gqa(&q, &k, &v, n_q, &args);
        let d_q = dev.htod_copy(q).unwrap();
        let d_k = dev.htod_copy(k).unwrap();
        let d_v = dev.htod_copy(v).unwrap();
        let out = enqueue_causal_gqa_prefill(&dev, &d_q, &d_k, &d_v, n_q, &args).unwrap();
        let got = dev.dtoh_sync_copy(&out).unwrap();
        assert_close(
            &got,
            &want,
            1e-4,
            &format!("attn start={start_pos} window={window:?} softcap={softcap:?}"),
        );
    }
}

/// The whole layer against a host twin: the same fixtures through
/// `mul_mm_reference` and the naive loops above, in the order the
/// host batched body runs them.
#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn the_dense_layer_matches_a_host_twin_with_prefix_biases_norms_and_rope() {
    let (batch, hidden, n_heads, n_kv_heads, head_dim, ffn) =
        (40usize, 64usize, 2usize, 1usize, 32usize, 96usize);
    let start_pos = 6;
    let q_w = n_heads * head_dim;
    let kv_w = n_kv_heads * head_dim;
    // The shared fixtures pin every Q8_0 scale near 0.1, which puts
    // this layer's `down` input in the thousands and its output in the
    // tens: an f32 sum that cancels three orders of magnitude, where
    // the GPU's FMA contraction and the twin's plain accumulate differ
    // by 4e-3 without either being wrong. Small scales keep every
    // activation near 1 so the comparison measures the kernels.
    let mk = |rows, cols, seed| {
        let mut data = fixtures::weights(&Q8_0, rows, cols, seed);
        for block in data.as_chunks_mut::<34>().0 {
            let bits = u16::from(block[0]) | (u16::from(block[1]) << 8);
            let bits = (bits & 0x83FF) | (6 << 10);
            block[0] = bits as u8;
            block[1] = (bits >> 8) as u8;
        }
        (data, rows, cols, cols / 32 * 34)
    };
    let (qd, ..) = mk(q_w, hidden, 1);
    let (kd, ..) = mk(kv_w, hidden, 2);
    let (vd, ..) = mk(kv_w, hidden, 3);
    let (od, ..) = mk(hidden, q_w, 4);
    let (gd, ..) = mk(ffn, hidden, 5);
    let (ud, ..) = mk(ffn, hidden, 6);
    let (dd, ..) = mk(hidden, ffn, 7);
    let view = |data: &'static [u8], rows, cols| q8_0_proj(data, rows, cols);
    // Leak the fixtures: the views borrow them for the whole test.
    let leak = |v: Vec<u8>| -> &'static [u8] { Box::leak(v.into_boxed_slice()) };
    let (qd, kd, vd, od, gd, ud, dd) = (
        leak(qd),
        leak(kd),
        leak(vd),
        leak(od),
        leak(gd),
        leak(ud),
        leak(dd),
    );
    let attn_norm_w = wave(hidden, 1.1)
        .iter()
        .map(|v| 1.0 + 0.1 * v)
        .collect::<Vec<_>>();
    let ffn_norm_w = wave(hidden, 1.3)
        .iter()
        .map(|v| 1.0 + 0.1 * v)
        .collect::<Vec<_>>();
    let post_attn = wave(hidden, 1.5)
        .iter()
        .map(|v| 1.0 + 0.1 * v)
        .collect::<Vec<_>>();
    let q_bias = wave(q_w, 2.1);
    let k_bias = wave(kv_w, 2.3);
    let v_bias = wave(kv_w, 2.5);
    let q_norm = wave(head_dim, 2.7)
        .iter()
        .map(|v| 1.0 + 0.1 * v)
        .collect::<Vec<_>>();
    let k_norm = wave(head_dim, 2.9)
        .iter()
        .map(|v| 1.0 + 0.1 * v)
        .collect::<Vec<_>>();
    let ff: Vec<f32> = (0..head_dim / 2).map(|i| 1.0 + i as f32 * 0.1).collect();
    let hidden_in = wave(batch * hidden, 0.05);
    let prefix_k = wave(start_pos * kv_w, 3.1);
    let prefix_v = wave(start_pos * kv_w, 3.3);
    let eps = 1e-5;
    let scale = (head_dim as f32).sqrt().recip();
    let window = Some(20);

    // --- host twin ---
    let gemm = |data: &[u8], rows, cols, x: &[f32]| {
        mul_mm_reference(&Q8_0, data, x, rows, cols, batch, cols / 32 * 34).unwrap()
    };
    let normed = host_rms_norm_rows(&hidden_in, &attn_norm_w, hidden, eps);
    let mut q = gemm(qd, q_w, hidden, &normed);
    let mut k = gemm(kd, kv_w, hidden, &normed);
    let mut v = gemm(vd, kv_w, hidden, &normed);
    for (x, b, w) in [
        (&mut q, &q_bias, q_w),
        (&mut k, &k_bias, kv_w),
        (&mut v, &v_bias, kv_w),
    ] {
        for row in x.chunks_exact_mut(w) {
            for (a, bb) in row.iter_mut().zip(b) {
                *a += bb;
            }
        }
    }
    let mut q = host_rms_norm_rows(&q, &q_norm, head_dim, eps);
    let mut k = host_rms_norm_rows(&k, &k_norm, head_dim, eps);
    host_rope_rows(
        &mut q,
        n_heads,
        head_dim,
        head_dim,
        start_pos,
        500000.0,
        Some(&ff),
        false,
        1.0,
    );
    host_rope_rows(
        &mut k,
        n_kv_heads,
        head_dim,
        head_dim,
        start_pos,
        500000.0,
        Some(&ff),
        false,
        1.0,
    );
    let mut k_all = prefix_k.clone();
    k_all.extend_from_slice(&k);
    let mut v_all = prefix_v.clone();
    v_all.extend_from_slice(&v);
    let args = AttnArgs {
        n_heads,
        n_kv_heads,
        head_dim,
        start_pos,
        window,
        scale,
        softcap: None,
    };
    let attn = host_causal_gqa(&q, &k_all, &v_all, batch, &args);
    let o = gemm(od, hidden, q_w, &attn);
    let o = host_rms_norm_rows(&o, &post_attn, hidden, eps);
    let mut h: Vec<f32> = hidden_in.iter().zip(&o).map(|(a, b)| a + b).collect();
    let normed2 = host_rms_norm_rows(&h, &ffn_norm_w, hidden, eps);
    let g = gemm(gd, ffn, hidden, &normed2);
    let u = gemm(ud, ffn, hidden, &normed2);
    let act: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let d = gemm(dd, hidden, ffn, &act);
    for (a, b) in h.iter_mut().zip(&d) {
        *a += b;
    }

    // --- device ---
    let layer = PrefillDenseLayerCuda {
        attn_norm_w: &attn_norm_w,
        ffn_norm_w: &ffn_norm_w,
        q: view(qd, q_w, hidden),
        k: view(kd, kv_w, hidden),
        v: view(vd, kv_w, hidden),
        o: view(od, hidden, q_w),
        gate: view(gd, ffn, hidden),
        up: view(ud, ffn, hidden),
        down: view(dd, hidden, ffn),
        post_attn_norm: Some(&post_attn),
        post_ffn_norm: None,
        extras: AttnExtrasCuda {
            q_bias: Some(&q_bias),
            k_bias: Some(&k_bias),
            v_bias: Some(&v_bias),
            q_norm: Some(&q_norm),
            k_norm: Some(&k_norm),
        },
        rope: Some(LayerRopeCuda {
            theta: 500000.0,
            freq_factors: Some(&ff),
            rot_dim: head_dim,
            layout: RopeLayoutCuda::Norm,
            mscale: 1.0,
        }),
    };
    let params = PrefillParams {
        n_heads,
        n_kv_heads,
        head_dim,
        rms_eps: eps,
        attn_scale: scale,
        attn_softcap: None,
        window,
        prefix_k: &prefix_k,
        prefix_v: &prefix_v,
        start_pos,
    };
    let out = launch_prefill_dense_layer(&hidden_in, &layer, &params, batch).unwrap();
    // 1e-2 of the vector's scale, not 1e-3 of each element: on sm_80+
    // the seven GEMMs run on the tensor cores with f16 operands
    // (`mul_mm_tc`) and the twin here is the f32 reference. Measured on
    // an RTX 3090: K rows within 2e-3, and one hidden element of 0.39
    // off by 6e-3 in a vector of RMS about 1 after two residual adds.
    // A wrong kernel is off by a whole term. The GEMM's own exactness
    // is `mul_mm_launch`'s test.
    assert_close(&out.k_rows, &k, 1e-2, "K rows");
    assert_close(&out.v_rows, &v, 1e-2, "V rows");
    assert_close(&out.hidden, &h, 1e-2, "hidden");

    // The stack: the same layer twice with the hidden batch resident
    // between them must equal two single launches, K/V rows per layer.
    let second = launch_prefill_dense_layer(&out.hidden, &layer, &params, batch).unwrap();
    let stacked = launch_prefill_dense_stack(
        StackInput::Hidden(&hidden_in),
        &[(&layer, &params), (&layer, &params)],
        batch,
        StackKv::UploadPrefix,
    )
    .unwrap();
    assert_eq!(stacked.kv_rows.len(), 2);
    assert_close(&stacked.kv_rows[0].0, &out.k_rows, 1e-5, "stack layer 0 K");
    assert_close(
        &stacked.kv_rows[1].0,
        &second.k_rows,
        1e-5,
        "stack layer 1 K",
    );
    assert_close(
        &stacked.kv_rows[1].1,
        &second.v_rows,
        1e-5,
        "stack layer 1 V",
    );
    assert_close(&stacked.hidden, &second.hidden, 1e-5, "stack hidden");
}

/// Q8_0 fixture bytes with every block scale near `2^-9`, so a
/// layer's activations stay near 1 (see the note in the layer test
/// above), leaked for the `'static` views.
pub(super) fn small_q8_0(rows: usize, cols: usize, seed: u32) -> &'static [u8] {
    let mut data = fixtures::weights(&Q8_0, rows, cols, seed);
    for block in data.as_chunks_mut::<34>().0 {
        let bits = u16::from(block[0]) | (u16::from(block[1]) << 8);
        let bits = (bits & 0x83FF) | (6 << 10);
        block[0] = bits as u8;
        block[1] = (bits >> 8) as u8;
    }
    Box::leak(data.into_boxed_slice())
}

/// A one-row step multiplies by the matvec; a projection that has
/// only the GEMM's view must be refused before the device, not run
/// through a tile that wastes all but one column, and not silently.
#[test]
fn a_one_row_step_without_a_matvec_is_refused_before_the_device() {
    let gemm = |rows, cols| {
        ProjWeights::gemm_only(MulMmWeights {
            kind: &Q8_0,
            data: &[],
            rows,
            cols,
            row_bytes: cols / 32 * 34,
        })
    };
    let w = vec![1.0f32; 64];
    let layer = PrefillDenseLayerCuda {
        attn_norm_w: &w,
        ffn_norm_w: &w,
        q: gemm(64, 64),
        k: gemm(32, 64),
        v: gemm(32, 64),
        o: gemm(64, 64),
        gate: gemm(96, 64),
        up: gemm(96, 64),
        down: gemm(64, 96),
        post_attn_norm: None,
        post_ffn_norm: None,
        extras: AttnExtrasCuda::default(),
        rope: None,
    };
    let params = PrefillParams {
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        rms_eps: 1e-5,
        attn_scale: 32f32.sqrt().recip(),
        attn_softcap: None,
        window: None,
        prefix_k: &[],
        prefix_v: &[],
        start_pos: 0,
    };
    let err = launch_prefill_dense_stack(
        StackInput::Hidden(&w),
        &[(&layer, &params)],
        1,
        StackKv::UploadPrefix,
    )
    .err()
    .expect("a one-row step with no matvec must be refused");
    assert!(
        matches!(&err, CudaError::Unsupported(m) if m.contains("matvec")),
        "{err:?}"
    );
}

/// The two views of a projection must describe the same bytes: a
/// matvec over one tensor beside the shape of another would multiply
/// the wrong weights with every check passing.
#[test]
fn a_matvec_view_of_other_bytes_is_refused() {
    let a: &'static [u8] = Box::leak(vec![0u8; 64 * 2 * 34].into_boxed_slice());
    let b: &'static [u8] = Box::leak(vec![0u8; 64 * 2 * 34].into_boxed_slice());
    let mut p = q8_0_proj(a, 64, 64);
    assert!(p.validate(1).is_ok());
    p.matvec.as_mut().unwrap().weights = b;
    let err = p.validate(1).expect_err("different bytes");
    assert!(matches!(err, CudaError::Unsupported(_)), "{err:?}");
    // A GEMM step does not read the matvec view at all.
    assert!(p.validate(4).is_ok());
}

#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn decode_attn_matches_its_twin_across_chunks_a_window_and_a_softcap() {
    let dev = shared_device().unwrap();
    for (n_heads, n_kv_heads, head_dim) in [(24usize, 8usize, 128usize), (4, 2, 64), (2, 1, 256)] {
        for (seq_len, window, softcap) in [
            (1usize, None, None),
            (63, None, None),
            (64, None, None),
            (65, None, None),
            (777, None, None),
            (777, Some(300), None),
            (130, None, Some(50.0)),
        ] {
            let q = wave(n_heads * head_dim, 0.2);
            let k = wave(seq_len * n_kv_heads * head_dim, 0.4);
            let v = wave(seq_len * n_kv_heads * head_dim, 0.6);
            let scale = (head_dim as f32).sqrt().recip();
            let want = crate::decode_attn::decode_attn_twin(
                &q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, window, scale, softcap,
            );
            let args = AttnArgs {
                n_heads,
                n_kv_heads,
                head_dim,
                start_pos: seq_len - 1,
                window,
                scale,
                softcap,
            };
            let d_q = dev.htod_copy(q.clone()).unwrap();
            let d_k = dev.htod_copy(k.clone()).unwrap();
            let d_v = dev.htod_copy(v.clone()).unwrap();
            let out = enqueue_decode_attn(&dev, &d_q, &d_k, &d_v, &args).unwrap();
            let got = dev.dtoh_sync_copy(&out).unwrap();
            let what = format!("decode attn h={n_heads} hd={head_dim} seq={seq_len} w={window:?}");
            assert_close(&got, &want, 1e-4, &what);
            // And against the definition, through the prefill kernel's
            // host loop at one query row.
            let naive = host_causal_gqa(&q, &k, &v, 1, &args);
            assert_close(&got, &naive, 1e-4, &format!("{what} vs naive"));
        }
    }
}

/// A Llama-shaped layer (no biases, no QK norm, NORM RoPE with
/// divisors) on the host at one row: the arithmetic the stack's
/// one-row step must reproduce. Returns the new hidden row, K, V.
#[allow(clippy::too_many_arguments)]
fn host_llama_row(
    w: &[&[u8]; 7],
    dims: (usize, usize, usize, usize, usize),
    norms: (&[f32], &[f32]),
    ff: &[f32],
    hidden_in: &[f32],
    prefix_k: &[f32],
    prefix_v: &[f32],
    pos: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (hidden, n_heads, n_kv_heads, head_dim, ffn) = dims;
    let (q_w, kv_w) = (n_heads * head_dim, n_kv_heads * head_dim);
    let mv = |data: &[u8], rows, cols, x: &[f32]| {
        mul_mm_reference(&Q8_0, data, x, rows, cols, 1, cols / 32 * 34).unwrap()
    };
    let eps = 1e-5;
    let normed = host_rms_norm_rows(hidden_in, norms.0, hidden, eps);
    let mut q = mv(w[0], q_w, hidden, &normed);
    let mut k = mv(w[1], kv_w, hidden, &normed);
    let v = mv(w[2], kv_w, hidden, &normed);
    for (x, heads) in [(&mut q, n_heads), (&mut k, n_kv_heads)] {
        host_rope_rows(
            x,
            heads,
            head_dim,
            head_dim,
            pos,
            500000.0,
            Some(ff),
            false,
            1.0,
        );
    }
    let mut k_all = prefix_k.to_vec();
    k_all.extend_from_slice(&k);
    let mut v_all = prefix_v.to_vec();
    v_all.extend_from_slice(&v);
    let args = AttnArgs {
        n_heads,
        n_kv_heads,
        head_dim,
        start_pos: pos,
        window: None,
        scale: (head_dim as f32).sqrt().recip(),
        softcap: None,
    };
    let attn = host_causal_gqa(&q, &k_all, &v_all, 1, &args);
    let o = mv(w[3], hidden, q_w, &attn);
    let mut h: Vec<f32> = hidden_in.iter().zip(&o).map(|(a, b)| a + b).collect();
    let normed2 = host_rms_norm_rows(&h, norms.1, hidden, eps);
    let g = mv(w[4], ffn, hidden, &normed2);
    let u = mv(w[5], ffn, hidden, &normed2);
    let act: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let d = mv(w[6], hidden, ffn, &act);
    for (a, b) in h.iter_mut().zip(&d) {
        *a += b;
    }
    (h, k, v)
}

/// Decode is the stack at one row. Token by token over a prefix, two
/// layers, through each layer's device mirror: every token's hidden
/// row and K/V rows must match the host twin, the mirror must agree
/// with uploading the prefix every time (same kernels, different home
/// for the history), and a mirror the host cache no longer matches
/// must be rebuilt rather than trusted.
#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn decode_through_the_mirror_matches_the_host_twin_token_by_token() {
    let (hidden, n_heads, n_kv_heads, head_dim, ffn) = (64usize, 2usize, 1usize, 32usize, 96usize);
    let (q_w, kv_w) = (n_heads * head_dim, n_kv_heads * head_dim);
    let w: [&'static [u8]; 7] = [
        small_q8_0(q_w, hidden, 11),
        small_q8_0(kv_w, hidden, 12),
        small_q8_0(kv_w, hidden, 13),
        small_q8_0(hidden, q_w, 14),
        small_q8_0(ffn, hidden, 15),
        small_q8_0(ffn, hidden, 16),
        small_q8_0(hidden, ffn, 17),
    ];
    let norm = |seed| {
        wave(hidden, seed)
            .iter()
            .map(|v| 1.0 + 0.1 * v)
            .collect::<Vec<_>>()
    };
    let (an, fnw) = (norm(1.1), norm(1.3));
    let ff: Vec<f32> = (0..head_dim / 2).map(|i| 1.0 + i as f32 * 0.1).collect();
    let layer = PrefillDenseLayerCuda {
        attn_norm_w: &an,
        ffn_norm_w: &fnw,
        q: q8_0_proj(w[0], q_w, hidden),
        k: q8_0_proj(w[1], kv_w, hidden),
        v: q8_0_proj(w[2], kv_w, hidden),
        o: q8_0_proj(w[3], hidden, q_w),
        gate: q8_0_proj(w[4], ffn, hidden),
        up: q8_0_proj(w[5], ffn, hidden),
        down: q8_0_proj(w[6], hidden, ffn),
        post_attn_norm: None,
        post_ffn_norm: None,
        extras: AttnExtrasCuda::default(),
        rope: Some(LayerRopeCuda {
            theta: 500000.0,
            freq_factors: Some(&ff),
            rot_dim: head_dim,
            layout: RopeLayoutCuda::Norm,
            mscale: 1.0,
        }),
    };
    let leak = |v: Vec<f32>| -> &'static [f32] { Box::leak(v.into_boxed_slice()) };
    let params_at = |pos: usize, pk: &'static [f32], pv: &'static [f32]| PrefillParams {
        n_heads,
        n_kv_heads,
        head_dim,
        rms_eps: 1e-5,
        attn_scale: (head_dim as f32).sqrt().recip(),
        attn_softcap: None,
        window: None,
        prefix_k: pk,
        prefix_v: pv,
        start_pos: pos,
    };
    // Two layers' host caches, seeded with a prefix of 70 rows (two
    // decode-attention chunks once the token is added).
    let start = 70;
    let mut host_k = [wave(start * kv_w, 3.1), wave(start * kv_w, 3.7)];
    let mut host_v = [wave(start * kv_w, 3.3), wave(start * kv_w, 3.9)];
    let mut mirrors: [Option<CudaKvMirror>; 2] = [None, None];
    let mut row = wave(hidden, 0.05);
    let step = |row: &[f32],
                pos: usize,
                host_k: &[Vec<f32>; 2],
                host_v: &[Vec<f32>; 2],
                mirrors: &mut [Option<CudaKvMirror>; 2]| {
        let p: Vec<PrefillParams<'static>> = (0..2)
            .map(|l| params_at(pos, leak(host_k[l].clone()), leak(host_v[l].clone())))
            .collect();
        let launch = [(&layer, &p[0]), (&layer, &p[1])];
        let [m0, m1] = mirrors;
        let mut slots = [m0, m1];
        let via_mirror = launch_prefill_dense_stack(
            StackInput::Hidden(row),
            &launch,
            1,
            StackKv::Mirrors(&mut slots),
        )
        .unwrap();
        let via_upload =
            launch_prefill_dense_stack(StackInput::Hidden(row), &launch, 1, StackKv::UploadPrefix)
                .unwrap();
        (via_mirror, via_upload)
    };
    for pos in start..start + 4 {
        let (via_mirror, via_upload) = step(&row, pos, &host_k, &host_v, &mut mirrors);
        // Same kernels, same inputs: only where the history lives
        // differs, so the two agree to the bit.
        assert_eq!(
            via_mirror.hidden, via_upload.hidden,
            "pos {pos}: mirror vs upload"
        );
        assert_eq!(
            via_mirror.kv_rows, via_upload.kv_rows,
            "pos {pos}: K/V rows"
        );
        // And the host twin, layer by layer.
        let mut h = row.clone();
        for l in 0..2 {
            let (h2, k, v) = host_llama_row(
                &w,
                (hidden, n_heads, n_kv_heads, head_dim, ffn),
                (&an, &fnw),
                &ff,
                &h,
                &host_k[l],
                &host_v[l],
                pos,
            );
            assert_close(
                &via_mirror.kv_rows[l].0,
                &k,
                1e-3,
                &format!("pos {pos} layer {l} K"),
            );
            assert_close(
                &via_mirror.kv_rows[l].1,
                &v,
                1e-3,
                &format!("pos {pos} layer {l} V"),
            );
            h = h2;
        }
        assert_close(&via_mirror.hidden, &h, 1e-3, &format!("pos {pos} hidden"));
        for l in 0..2 {
            host_k[l].extend_from_slice(&via_mirror.kv_rows[l].0);
            host_v[l].extend_from_slice(&via_mirror.kv_rows[l].1);
            assert_eq!(mirrors[l].as_ref().unwrap().rows(), pos + 1);
        }
        row = via_mirror.hidden;
    }

    // Truncate by one and grow back with a DIFFERENT row through a host
    // path: same length, different history. The mirror must notice.
    for k in host_k.iter_mut() {
        let n = k.len();
        k[n - 1] += 0.5;
    }
    let (via_mirror, via_upload) = step(&row, start + 4, &host_k, &host_v, &mut mirrors);
    assert_eq!(
        via_mirror.hidden, via_upload.hidden,
        "a stale mirror was trusted"
    );
}
