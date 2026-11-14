//! The Metal path against the host body, on a synthetic Gemma-4 small
//! enough to run in a test and shaped to reach every feature the
//! stacks were parameterised for: sliding (256-wide) and full (512-wide)
//! heads, a full layer's `rope_freqs`, shared-KV layers reading both
//! kinds of slot, per-layer embeddings with f32 gate and projection,
//! per-layer FFN widths, the output scale, and a window small enough
//! that both decode and prefill cross it.

use super::*;
use crate::engine::Engine;
use crate::gemma4_engine::{Gemma4AttnWeights, Gemma4Hparams, Gemma4LayerWeights, Gemma4Weights};
use frink_core::tensor::Tensor;
use frink_core::weight_matrix::{QuantKind, WeightBytes};

const HIDDEN: usize = 256;
const VOCAB: usize = 64;
const WINDOW: usize = 8;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    fn vals(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }

    /// Q8_0, so the prefill stack has a GEMM for it.
    fn q8(&mut self, rows: usize, cols: usize) -> WeightMatrix {
        let v = self.vals(rows * cols, 1.0 / (cols as f32).sqrt());
        WeightMatrix::Quantized {
            data: WeightBytes::Owned(frink_quant::quantize_q8_0(&v)),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        }
    }

    /// F32, as E2B ships its per-layer gate and projection.
    fn f32m(&mut self, rows: usize, cols: usize) -> WeightMatrix {
        let v = self.vals(rows * cols, 1.0 / (cols as f32).sqrt());
        WeightMatrix::F32(Tensor::new(v, vec![rows, cols]))
    }

    fn norm(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 1.0 + 0.2 * self.next()).collect()
    }
}

/// Nine layers in E2B's 4:1-style rhythm compressed to 2:1 (sliding,
/// sliding, full), the last three sharing KV: layers 6 and 7 (sliding)
/// read slot 4, layer 8 (full) reads slot 5.
fn fixture() -> Gemma4Engine {
    let n_layer = 9;
    let hp = Gemma4Hparams {
        arch: "gemma4".into(),
        n_layer,
        hidden_dim: HIDDEN,
        ffn_dims: (0..n_layer)
            .map(|il| if il < 6 { 512 } else { 768 })
            .collect(),
        n_heads: 4,
        n_kv_heads: 1,
        head_dim_full: 512,
        head_dim_swa: 256,
        rot_dim_full: 512,
        rot_dim_swa: 256,
        sliding_window: WINDOW,
        is_swa: (0..n_layer).map(|il| (il + 1) % 3 != 0).collect(),
        n_layer_kv_from_start: 6,
        embd_per_layer: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        rope_theta_swa: 10_000.0,
        final_logit_softcap: Some(30.0),
        attention_scale: 1.0,
    };
    let mut r = Rng(0x6e6d_6134);
    let layers = (0..n_layer)
        .map(|il| {
            let hd = hp.head_dim(il);
            let kv = hp.n_kv_heads * hd;
            let q = hp.n_heads * hd;
            let ffn = hp.ffn_dim(il);
            let own = hp.has_kv(il);
            Gemma4LayerWeights {
                attn_norm: r.norm(HIDDEN),
                attn: Gemma4AttnWeights {
                    q_proj: r.q8(q, HIDDEN),
                    k_proj: own.then(|| r.q8(kv, HIDDEN)),
                    v_proj: own.then(|| r.q8(kv, HIDDEN)),
                    o_proj: r.q8(HIDDEN, q),
                    // Small: with `attention_scale = 1` a unit-RMS Q·K at
                    // these widths scores ~sqrt(head_dim), and a softmax
                    // that peaked would hide every difference behind one key.
                    q_norm: r.norm(hd).iter().map(|w| 0.15 * w).collect(),
                    k_norm: own.then(|| r.norm(hd)),
                    post_attn_norm: r.norm(HIDDEN),
                },
                ffn_norm: r.norm(HIDDEN),
                ffn_gate: r.q8(ffn, HIDDEN),
                ffn_up: r.q8(ffn, HIDDEN),
                ffn_down: r.q8(HIDDEN, ffn),
                ffn_post_norm: r.norm(HIDDEN),
                per_layer_inp_gate: Some(r.f32m(hp.embd_per_layer, HIDDEN)),
                per_layer_proj: Some(r.f32m(HIDDEN, hp.embd_per_layer)),
                per_layer_post_norm: Some(r.norm(HIDDEN)),
                out_scale: Some(0.6 + 0.05 * il as f32),
            }
        })
        .collect();
    let ple = n_layer * hp.embd_per_layer;
    let weights = Gemma4Weights {
        token_embd: r.q8(VOCAB, HIDDEN),
        per_layer_token_embd: Some(r.q8(VOCAB, ple)),
        per_layer_model_proj: Some(r.f32m(ple, HIDDEN)),
        per_layer_proj_norm: Some(r.norm(hp.embd_per_layer)),
        layers,
        output_norm: r.norm(HIDDEN),
        output_head: r.q8(VOCAB, HIDDEN),
        // Proportional RoPE: half the bands rotate at their own rate,
        // half three times slower, so a stack that drops the divisors
        // (or hands them to a sliding layer) rotates differently.
        rope_freqs: Some(
            (0..256)
                .map(|i| if i % 2 == 0 { 1.0 } else { 3.0 })
                .collect(),
        ),
    };
    Gemma4Engine {
        weights,
        hp,
        // The device's store, so the host body rounds K and V as the
        // stacks do.
        kv_dtype: frink_core::kv_store::KvDtype::F16,
        device_aux: Default::default(),
    }
}

fn tokens(n: usize) -> Vec<usize> {
    (0..n).map(|i| (i * 7 + 3) % VOCAB).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// `|a - b| / |a|` over the whole vector.
fn rel_err(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    let num: f32 = want.iter().zip(got).map(|(a, b)| (a - b) * (a - b)).sum();
    let den: f32 = want.iter().map(|a| a * a).sum();
    (num / den).sqrt()
}

/// The device answers from f16 KV and f32 activations where the host
/// quantizes activations to Q8 for its dot products: agreement is
/// close, not bitwise. A wrong slot, scale or width moves logits by
/// tens of percent on this fixture.
const TOL: f32 = 2e-2;

fn assert_close(what: &str, want: &[f32], got: &[f32]) {
    let e = rel_err(want, got);
    assert!(e < TOL, "{what}: relative error {e} (tolerance {TOL})");
    assert_eq!(argmax(want), argmax(got), "{what}: greedy token differs");
}

/// No GPU needed: the fixture, and E2B's shapes, are fully described to
/// both stacks. A layer the stacks could not describe would silently
/// keep the whole model on the host.
#[test]
fn the_stacks_describe_every_gemma4_layer() {
    let e = fixture();
    assert!(e.metal_stacks_cover());
    let layers = e.decode_layers().expect("decode layers");
    for (il, l) in layers.iter().enumerate() {
        let a = l.mixer.attn().expect("every Gemma-4 layer attends");
        match a.kv {
            LayerKv::Own { slot, .. } => {
                assert!(e.hp.has_kv(il));
                assert_eq!(slot, il);
            }
            LayerKv::Shared { slot } => assert_eq!(slot, e.hp.kv_reuse_layer(il)),
        }
        assert_eq!(a.window.is_some(), e.hp.is_swa_layer(il));
        assert_eq!(
            a.rope.and_then(|r| r.freq_factors).is_some(),
            !e.hp.is_swa_layer(il),
            "rope_freqs on the full layers only"
        );
        assert_eq!(l.out_scale, Some(0.6 + 0.05 * il as f32));
    }
    let prefill = e.prefill_layers().expect("prefill layers");
    for (il, l) in prefill.iter().enumerate() {
        assert_eq!(l.window, e.hp.is_swa_layer(il).then_some(WINDOW));
    }
    // The attention scale folded into Q's norm, as the host's `compensate`.
    let aux = e.device_aux();
    let hd = e.hp.head_dim(4) as f32;
    assert_eq!(
        aux.q_norm[4][0],
        e.weights.layers[4].attn.q_norm[0] * hd.sqrt()
    );
}

/// llama.cpp reads the raw K as V when a layer has no `attn_v`, as the
/// host body does; the stacks have no arm for it, so they must not
/// claim such a file at all.
#[test]
fn a_kv_layer_without_attn_v_stays_on_the_host() {
    let mut e = fixture();
    e.weights.layers[1].attn.v_proj = None;
    assert!(e.decode_layers().is_none());
    assert!(e.prefill_layers().is_none());
    assert!(!e.metal_stacks_cover());
}

#[test]
#[ignore = "needs Metal"]
fn metal_decode_matches_the_host_body_across_the_window() {
    let e = fixture();
    let toks = tokens(3 * WINDOW);
    let (mut host, mut dev) = (e.new_state(), e.new_state());
    for (pos, &t) in toks.iter().enumerate() {
        let want = e.forward_token_host(t, pos, &mut host);
        let got = e
            .decode_on_metal(t, pos, &mut dev)
            .expect("the fixture is Metal-eligible");
        assert_close(&format!("decode pos {pos}"), &want, &got);
    }
    assert!(dev.metal.kv.is_some(), "the device held the KV throughout");
    for (il, c) in dev.kv.iter().enumerate() {
        assert_eq!(
            c.as_ref().map(|c| c.positions()),
            e.hp.has_kv(il).then_some(toks.len()),
            "layer {il}: the host record advanced with the device"
        );
    }
}

#[test]
#[ignore = "needs Metal"]
fn metal_prefill_matches_the_host_body() {
    let e = fixture();
    // Within the window: the whole prompt is one prefill command buffer.
    let short = tokens(WINDOW - 2);
    let mut host = e.new_state();
    let mut want = Vec::new();
    for (pos, &t) in short.iter().enumerate() {
        want = e.forward_token_host(t, pos, &mut host);
    }
    let mut dev = e.new_state();
    let (done, got) = e.prefill_on_metal(&short, 0, &mut dev);
    assert_eq!(done, short.len(), "batched in one chunk");
    assert_close("prefill inside the window", &want, &got);
    // ...and decoding on from it reads the KV the prefill wrote.
    let next = e.forward_token_host(5, short.len(), &mut host);
    let got = e.decode_on_metal(5, short.len(), &mut dev).expect("decode");
    assert_close("decode after prefill", &next, &got);

    // Past the window: the whole prompt is still one batch, its sliding
    // layers masking the keys a query has slid past.
    let long = tokens(2 * WINDOW + 3);
    let mut host = e.new_state();
    let mut want = Vec::new();
    for (pos, &t) in long.iter().enumerate() {
        want = e.forward_token_host(t, pos, &mut host);
    }
    let mut dev = e.new_state();
    let (done, got) = e.prefill_on_metal(&long, 0, &mut dev);
    assert_eq!(done, long.len(), "batched past the window");
    assert_close("prefill past the window", &want, &got);

    // A second batch that starts past the window, on a prefix the first
    // one wrote: its first query's window begins inside the old rows.
    let split = WINDOW + 3;
    let mut dev = e.new_state();
    let (done, _) = e.prefill_on_metal(&long[..split], 0, &mut dev);
    assert_eq!(done, split);
    let (done, got) = e.prefill_on_metal(&long[split..], split, &mut dev);
    assert_eq!(done, long.len() - split);
    assert_close("second batch past the window", &want, &got);
    // ...and decoding on from it reads the right rows.
    let n = long.len();
    let next = e.forward_token_host(7, n, &mut host);
    let got = e.decode_on_metal(7, n, &mut dev).expect("decode");
    assert_close("decode after a windowed prefill", &next, &got);
}

/// The device KV goes back to the host intact: a state that leaves the
/// Metal path mid-sequence continues on the host as if it had never
/// left.
#[test]
#[ignore = "needs Metal"]
fn leaving_metal_hands_the_host_the_device_rows() {
    let e = fixture();
    let toks = tokens(2 * WINDOW);
    let (mut host, mut dev) = (e.new_state(), e.new_state());
    let split = WINDOW + 3;
    for (pos, &t) in toks.iter().enumerate().take(split) {
        e.forward_token_host(t, pos, &mut host);
        e.decode_on_metal(t, pos, &mut dev).expect("decode");
    }
    Gemma4Engine::drop_device_kv(&mut dev);
    for (pos, &t) in toks.iter().enumerate().skip(split) {
        let want = e.forward_token_host(t, pos, &mut host);
        let got = e.forward_token_host(t, pos, &mut dev);
        assert_close(&format!("host after Metal, pos {pos}"), &want, &got);
    }
}

/// `Engine::prefill` is `forward_token` per token unless an engine
/// overrides it; on a fresh state with Metal switched off here, the
/// override must give the same answer as the loop.
#[test]
fn engine_prefill_matches_the_token_loop_on_the_host() {
    if Gemma4Engine::metal_switched_on() {
        return;
    }
    let e = fixture();
    let toks = tokens(5);
    let mut a = e.new_state();
    let mut want = Vec::new();
    for (pos, &t) in toks.iter().enumerate() {
        want = e.forward_token(t, pos, &mut a);
    }
    let mut b = e.new_state();
    assert_eq!(e.prefill(&toks, 0, &mut b), want);
}

/// The fixture rotating `rot` channels of every head on both layer
/// kinds (`rope.dimension_count = rot`, no `_swa` key: llama.cpp's
/// `n_rot_swa` defaults to `n_rot_full`).
fn partial_rope_fixture(rot: usize) -> Gemma4Engine {
    let mut e = fixture();
    e.hp.rot_dim_full = rot;
    e.hp.rot_dim_swa = rot;
    e
}

/// `MetalRope::rot_dim` is one width for the run: whole heads on every
/// layer is no width at all, one partial width on every layer rides the
/// stacks with `rope_freqs` cut to its `n_rot/2` bands, and a full width
/// different from the sliding one must not be claimed (#635).
#[test]
fn the_stacks_take_one_rope_width_or_none() {
    let e = fixture();
    assert_eq!(e.metal_rope().and_then(|r| r.rot_dim), None);

    let e = partial_rope_fixture(128);
    assert_eq!(e.metal_rope().and_then(|r| r.rot_dim), Some(128));
    assert!(e.metal_stacks_cover());
    let full = 2;
    assert!(!e.hp.is_swa_layer(full));
    let ff = e
        .layer_rope(full)
        .freq_factors
        .expect("full layer divisors");
    assert_eq!(ff.len(), 64, "ggml reads n_rot/2 bands of rope_freqs");

    // 256 of 512 on the full layers and 256 of 256 on the sliding ones
    // is one width; 128 beside 256 is two.
    let mut e = fixture();
    e.hp.rot_dim_full = 256;
    assert_eq!(e.metal_rope().and_then(|r| r.rot_dim), Some(256));
    e.hp.rot_dim_full = 128;
    assert!(e.metal_rope().is_none());
    assert!(!e.metal_stacks_cover());
}

#[test]
#[ignore = "needs Metal"]
fn metal_matches_the_host_body_under_partial_rope() {
    let e = partial_rope_fixture(128);
    let toks = tokens(2 * WINDOW + 3);
    let mut host = e.new_state();
    let mut want = Vec::new();
    for (pos, &t) in toks.iter().enumerate() {
        want = e.forward_token_host(t, pos, &mut host);
    }
    let mut dev = e.new_state();
    let (done, got) = e.prefill_on_metal(&toks, 0, &mut dev);
    assert_eq!(done, toks.len(), "batched past the window");
    assert_close("partial-RoPE prefill", &want, &got);
    for (i, t) in [5usize, 9, 13].into_iter().enumerate() {
        let pos = toks.len() + i;
        let want = e.forward_token_host(t, pos, &mut host);
        let got = e.decode_on_metal(t, pos, &mut dev).expect("decode");
        assert_close(&format!("partial-RoPE decode pos {pos}"), &want, &got);
    }
}
