//! Each kind of layer `decoder::ssm_prefill` hands a Metal prefill run,
//! against the host body the decoder runs otherwise, from a state that
//! is not zero, with Q8_0 projections (the run's GEMMs need a quantized
//! kind). The run's GEMMs round their activation tiles to half, so the
//! comparison is within a tolerance; a block run out of order, a state
//! not carried, a window read from zero or a scale applied twice is
//! far outside it.

use frink_core::matmul::rms_norm;
use frink_core::recurrent_state::RecurrentState;
use frink_core::weight_matrix::{QuantKind, WeightBytes};
use frink_core::WeightMatrix;
use frink_metal::gdn_prefill::RunOptions;
use frink_metal::prefill_run::{launch_prefill_run, PrefillRunLayer};
use frink_metal::ssm_decode::{
    DtBcNormMetal, Mamba1Mixer, Mamba1Weights, Mamba2Mixer, Mamba2Weights,
};
use frink_metal::ssm_prefill::{PrefillConvLayer, PrefillConvState, PrefillMambaLayer};

use super::{gemm, state_conv, state_ssm};
use crate::fused_layer::LayerFfnParts;
use crate::mamba1::{DtBcNorm, Mamba1, Mamba1Hparams};
use crate::mamba2::{Mamba2, Mamba2Hparams};
use crate::shortconv::{window_from_history, ShortConv};

struct Rng(u32);

impl Rng {
    fn v(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
                (((self.0 >> 9) as f32 / (1u32 << 23) as f32) - 0.5) * scale
            })
            .collect()
    }
}

fn q8(rows: usize, cols: usize, v: Vec<f32>) -> WeightMatrix {
    WeightMatrix::Quantized {
        data: WeightBytes::Owned(frink_quant::quantize_q8_0(&v)),
        rows,
        cols,
        kind: QuantKind::Q8_0,
    }
}

fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1.0);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * scale,
            "{what}[{i}]: device {g} host {w}"
        );
    }
}

fn normed(hidden: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    hidden
        .chunks(w.len())
        .flat_map(|r| rms_norm(r, w, eps))
        .collect()
}

/// `h += s * out`, the host's `residual_add`.
fn add(h: &mut [f32], out: &[f32], s: Option<f32>) {
    crate::scalar_multipliers::residual_add(h, out, s);
}

/// `ffn_norm`, gate, up, down.
type Ffn<'a> = (
    &'a [f32],
    &'a WeightMatrix,
    &'a WeightMatrix,
    &'a WeightMatrix,
);

/// The FFN tail on the host: `h += s * down(silu(gate(n)) * up(n))`.
fn host_ffn(h: &mut [f32], f: Ffn<'_>, rows: usize, s: Option<f32>) {
    let n = normed(h, f.0, EPS);
    let (g, u) = (f.1.apply_batch(&n, rows), f.2.apply_batch(&n, rows));
    let act: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(a, b)| a / (1.0 + (-a).exp()) * b)
        .collect();
    add(h, &f.3.apply_batch(&act, rows), s);
}

const EPS: f32 = 1e-5;

/// LFM2's conv layer with three rows of history already pushed, its
/// FFN, and Granite's residual multiplier: the output, and the history
/// rows the run hands back, against `ShortConv::forward_rows` over a
/// cache that holds the same history.
#[test]
#[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
fn a_conv_layer_in_the_run_matches_the_host_body() {
    let (n, ffn_dim, l, rows, prior) = (128usize, 192usize, 3usize, 24usize, 3usize);
    let mut r = Rng(9_101);
    let conv = ShortConv {
        conv: r.v(n * l, 1.0),
        l_cache: l,
        in_proj: q8(3 * n, n, r.v(3 * n * n, 0.3)),
        out_proj: q8(n, n, r.v(n * n, 0.3)),
    };
    let (gate, up, down) = (
        q8(ffn_dim, n, r.v(ffn_dim * n, 0.2)),
        q8(ffn_dim, n, r.v(ffn_dim * n, 0.2)),
        q8(n, ffn_dim, r.v(n * ffn_dim, 0.2)),
    );
    let attn_norm: Vec<f32> = r.v(n, 0.5).iter().map(|x| 1.0 + x).collect();
    let ffn_norm: Vec<f32> = r.v(n, 0.5).iter().map(|x| 1.0 + x).collect();
    let ffn = LayerFfnParts::from_parts(&ffn_norm, EPS, &gate, &up, &down);
    let hidden = r.v(rows * n, 2.0);
    let scale = Some(0.37f32);
    let mut history: Vec<Vec<f32>> = (0..prior).map(|_| r.v(n, 2.0)).collect();
    let window = window_from_history(l - 1, n, prior, |i| &history[i]);

    // The host body.
    let mut want = hidden.clone();
    let out = conv.forward_rows(&normed(&hidden, &attn_norm, EPS), rows, |bx| {
        history.push(bx.to_vec());
        window_from_history(l, n, history.len(), |i| &history[i])
    });
    add(&mut want, &out, scale);
    host_ffn(&mut want, (&ffn_norm, &gate, &up, &down), rows, scale);

    let mut bx_out = Vec::new();
    let layer = PrefillConvLayer {
        attn_norm: &attn_norm,
        eps: EPS,
        in_proj: gemm(&conv.in_proj).unwrap(),
        taps: &conv.conv,
        l_cache: l,
        out_proj: gemm(&conv.out_proj).unwrap(),
        ffn: Some(ffn.prefill_launches().unwrap()),
    };
    // SAFETY: no recurrent state is handed over.
    let got = unsafe {
        launch_prefill_run(
            &hidden,
            rows,
            prior,
            None,
            &mut [PrefillRunLayer::Conv(
                layer,
                PrefillConvState {
                    window: &window,
                    bx_out: &mut bx_out,
                },
            )],
            RunOptions {
                residual_scale: scale,
            },
        )
    }
    .expect("the run launches");
    close("out", &got, &want, 3e-3);
    close("history", &bx_out, &history[prior..].concat(), 3e-3);
}

fn mamba1(r: &mut Rng, n: usize, dt_bc_norm: DtBcNorm) -> Mamba1 {
    let h = Mamba1Hparams {
        d_conv: 4,
        d_inner: 256,
        d_state: 16,
        dt_rank: 32,
    };
    let x_rows = h.dt_rank + 2 * h.d_state;
    Mamba1 {
        h,
        in_proj: q8(2 * h.d_inner, n, r.v(2 * h.d_inner * n, 0.3)),
        conv1d: r.v(h.d_inner * h.d_conv, 1.0),
        conv1d_bias: r.v(h.d_inner, 0.5),
        x_proj: q8(x_rows, h.d_inner, r.v(x_rows * h.d_inner, 0.3)),
        dt_bc_norm,
        dt_proj: q8(h.d_inner, h.dt_rank, r.v(h.d_inner * h.dt_rank, 0.5)),
        dt_bias: r.v(h.d_inner, 1.0),
        a: r.v(h.d_inner * h.d_state, 2.0)
            .iter()
            .map(|v| -v.abs())
            .collect(),
        d: r.v(h.d_inner, 1.0),
        out_proj: q8(n, h.d_inner, r.v(n * h.d_inner, 0.3)),
    }
}

/// A state that is not zero, at the block's size.
fn warm(mut s: RecurrentState, r: &mut Rng) -> RecurrentState {
    let (c, k) = (s.conv.len(), s.ssm.len());
    s.conv.copy_from_slice(&r.v(c, 1.0));
    s.ssm.copy_from_slice(&r.v(k, 1.0));
    s
}

/// A pure Mamba layer (no FFN) for each of the three dt/B/C norm forms
/// (`mamba-base.cpp:84-88`: none, FalconMamba's weightless one, Jamba's
/// weighted one), from a state that is not zero: the output and both
/// states against `Mamba1::forward_rows`.
#[test]
#[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
fn a_mamba1_layer_in_the_run_matches_the_host_body() {
    let (n, rows) = (128usize, 20usize);
    for form in 0..3 {
        let mut r = Rng(4_321 + form);
        let norm = match form {
            0 => DtBcNorm::None,
            1 => DtBcNorm::Weightless,
            _ => DtBcNorm::Weighted {
                dt: r.v(32, 0.5).iter().map(|x| 1.0 + x).collect(),
                b: r.v(16, 0.5).iter().map(|x| 1.0 + x).collect(),
                c: r.v(16, 0.5).iter().map(|x| 1.0 + x).collect(),
            },
        };
        let m = mamba1(&mut r, n, norm);
        let attn_norm: Vec<f32> = r.v(n, 0.5).iter().map(|x| 1.0 + x).collect();
        let hidden = r.v(rows * n, 2.0);
        let state0 = warm(m.zero_state(), &mut r);

        let mut host_state = state0.clone();
        let mut want = hidden.clone();
        let out = m.forward_rows(
            &normed(&hidden, &attn_norm, EPS),
            rows,
            &mut host_state,
            EPS,
        );
        add(&mut want, &out, None);

        let mut dev_state = state0.clone();
        let dt_bc_norm = match &m.dt_bc_norm {
            DtBcNorm::None => None,
            DtBcNorm::Weightless => Some(DtBcNormMetal {
                weights: None,
                eps: EPS,
            }),
            DtBcNorm::Weighted { dt, b, c } => Some(DtBcNormMetal {
                weights: Some((&dt[..], &b[..], &c[..])),
                eps: EPS,
            }),
        };
        let w = Mamba1Weights {
            in_proj: gemm(&m.in_proj).unwrap(),
            conv1d: &m.conv1d,
            conv_bias: &m.conv1d_bias,
            d_conv: m.h.d_conv,
            x_proj: gemm(&m.x_proj).unwrap(),
            dt_bc_norm,
            dt_proj: gemm(&m.dt_proj).unwrap(),
            dt_bias: &m.dt_bias,
            a: &m.a,
            d: &m.d,
            out_proj: gemm(&m.out_proj).unwrap(),
            d_inner: m.h.d_inner,
            d_state: m.h.d_state,
            dt_rank: m.h.dt_rank,
        };
        // SAFETY: this test's page-aligned state, untouched until the
        // launch, which waits for the GPU, returns.
        let block =
            unsafe { Mamba1Mixer::new(w, state_conv(&mut dev_state), state_ssm(&mut dev_state)) }
                .expect("the shapes agree");
        // SAFETY: as above.
        let got = unsafe {
            launch_prefill_run(
                &hidden,
                rows,
                0,
                None,
                &mut [PrefillRunLayer::Mamba1(PrefillMambaLayer {
                    attn_norm: &attn_norm,
                    eps: EPS,
                    block,
                    ffn: None,
                })],
                RunOptions::default(),
            )
        }
        .expect("the run launches");
        close(&format!("form {form} out"), &got, &want, 3e-3);
        close(
            &format!("form {form} conv"),
            &dev_state.conv,
            &host_state.conv,
            3e-3,
        );
        close(
            &format!("form {form} ssm"),
            &dev_state.ssm,
            &host_state.ssm,
            3e-3,
        );
    }
}

/// A Mamba-2 layer with two groups and the group norm (Nemotron-H's
/// shape, `mamba-base.cpp:271-274`), its FFN and the residual
/// multiplier, from a state that is not zero, at a row count that is not
/// a multiple of the scan's row batch.
#[test]
#[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
fn a_mamba2_layer_in_the_run_matches_the_host_body() {
    let (n, ffn_dim, rows) = (128usize, 192usize, 21usize);
    let mut r = Rng(77);
    let h = Mamba2Hparams {
        d_conv: 4,
        d_inner: 256,
        d_state: 64,
        n_head: 8,
        n_group: 2,
    };
    let m = Mamba2 {
        h,
        in_proj: q8(h.in_proj_rows(), n, r.v(h.in_proj_rows() * n, 0.3)),
        conv1d: r.v(h.conv_width() * h.d_conv, 1.0),
        conv1d_bias: r.v(h.conv_width(), 0.5),
        dt_bias: r.v(h.n_head, 1.0),
        a: r.v(h.n_head, 2.0).iter().map(|v| -v.abs()).collect(),
        d: r.v(h.n_head, 1.0),
        norm: Some(r.v(h.d_inner, 0.5).iter().map(|x| 1.0 + x).collect()),
        out_proj: q8(n, h.d_inner, r.v(n * h.d_inner, 0.3)),
    };
    let (gate, up, down) = (
        q8(ffn_dim, n, r.v(ffn_dim * n, 0.2)),
        q8(ffn_dim, n, r.v(ffn_dim * n, 0.2)),
        q8(n, ffn_dim, r.v(n * ffn_dim, 0.2)),
    );
    let attn_norm: Vec<f32> = r.v(n, 0.5).iter().map(|x| 1.0 + x).collect();
    let ffn_norm: Vec<f32> = r.v(n, 0.5).iter().map(|x| 1.0 + x).collect();
    let ffn = LayerFfnParts::from_parts(&ffn_norm, EPS, &gate, &up, &down);
    let hidden = r.v(rows * n, 2.0);
    let scale = Some(0.5f32);
    let state0 = warm(m.zero_state(), &mut r);

    let mut host_state = state0.clone();
    let mut want = hidden.clone();
    let out = m.forward_rows(
        &normed(&hidden, &attn_norm, EPS),
        rows,
        &mut host_state,
        EPS,
    );
    add(&mut want, &out, scale);
    host_ffn(&mut want, (&ffn_norm, &gate, &up, &down), rows, scale);

    let mut dev_state = state0.clone();
    let w = Mamba2Weights {
        in_proj: gemm(&m.in_proj).unwrap(),
        conv1d: &m.conv1d,
        conv_bias: &m.conv1d_bias,
        d_conv: h.d_conv,
        dt_bias: &m.dt_bias,
        a: &m.a,
        d: &m.d,
        norm: m.norm.as_deref(),
        norm_eps: EPS,
        out_proj: gemm(&m.out_proj).unwrap(),
        d_inner: h.d_inner,
        d_state: h.d_state,
        n_head: h.n_head,
        n_group: h.n_group,
    };
    // SAFETY: as the Mamba-1 test's.
    let block =
        unsafe { Mamba2Mixer::new(w, state_conv(&mut dev_state), state_ssm(&mut dev_state)) }
            .expect("the shapes agree");
    // SAFETY: as above.
    let got = unsafe {
        launch_prefill_run(
            &hidden,
            rows,
            0,
            None,
            &mut [PrefillRunLayer::Mamba2(PrefillMambaLayer {
                attn_norm: &attn_norm,
                eps: EPS,
                block,
                ffn: Some(ffn.prefill_launches().unwrap()),
            })],
            RunOptions {
                residual_scale: scale,
            },
        )
    }
    .expect("the run launches");
    close("out", &got, &want, 3e-3);
    close("conv", &dev_state.conv, &host_state.conv, 3e-3);
    close("ssm", &dev_state.ssm, &host_state.ssm, 3e-3);
}
