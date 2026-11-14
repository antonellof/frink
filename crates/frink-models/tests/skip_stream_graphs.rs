//! Talkie, checked against llama.cpp itself: weightless norms, a
//! per-head scalar Q gain, an embedding skip stream, and two projection
//! gains.
//!
//! `talkie` was triaged NEW CODE on four things (`src/models/talkie.cpp`),
//! each measured to be one graph of 155: every norm is
//! `build_norm(x, nullptr, nullptr, LLM_NORM_RMS)` (`:50,68,90,110,137`;
//! `NormOp::RmsNoParams`); `attn_q_norm` is `{1, n_head}` applied after
//! RoPE with a weightless per-head K norm (`:26,82-91`;
//! `QkNormStyle::PerHeadScalar`); the normed embedding is added into
//! every layer's output times `layer_output_scale` (`:50-52,123-126`;
//! `frink_models::skip_stream`); and the converter writes
//! `attn_output.scale` / `ffn_down.scale` (`conversion/talkie.py:26-31`),
//! which `build_lora_mm` multiplies in (`frink_models::weight_scales`).
//! `logit_scale` is REQUIRED and multiplied (`:5,141`).
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `talkie` | the converter's shape: all four, gains included |
//! | `talkie_nogains` | no `.scale` companions (a hand-written shape); its golden differs, which is what pins that the gains are applied and not dropped |
//!
//! The Q gains, `out_scale` and the projection gains are drawn away
//! from one (or zero), so any of them dropped is visible; the sabotage
//! tests measure each.
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own `talkie`
//! graph over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `talkie` | 6.43e-14 | 9.54e-07 |
//! | `talkie_nogains` | 1.47e-14 | 4.17e-07 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_talkie_fixture.py \
//!     crates/frink-models/tests/fixtures/talkie_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_talkie_fixture.py \
//!     crates/frink-models/tests/fixtures/talkie_nogains_tiny.gguf --no-gains
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, kl_vs_golden, load_graph_fixture, worst_vs,
    GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::QkNormStyle;
use frink_models::config::RopeLayout;
use frink_models::norm::NormOp;
use frink_models::Decoder;

const WITH_GAINS: &str = "talkie";
const NO_GAINS: &str = "talkie_nogains";

const TALKIE_GOLDEN: [f32; 48] = [
    -1.2266144,
    1.82581,
    -1.1032155,
    -0.8537476,
    -0.15707164,
    -0.4309752,
    0.26436344,
    0.7653789,
    0.4522379,
    0.6228169,
    0.7763749,
    -1.369439,
    1.1275764,
    0.17016041,
    -0.5913589,
    -0.21517682,
    0.5504663,
    1.2972579,
    0.8483514,
    2.3139849,
    0.701385,
    0.3523885,
    0.8748827,
    0.110369995,
    -0.6576546,
    -1.1839297,
    1.0010854,
    0.519807,
    -1.2560184,
    0.053080916,
    -1.3863722,
    -1.2479892,
    0.9099202,
    0.27374917,
    2.588185,
    0.48087206,
    0.9983089,
    0.47070318,
    1.6834686,
    0.48162612,
    0.31500107,
    0.80107236,
    0.66740495,
    -1.208982,
    -1.0716579,
    -2.0485408,
    -1.8180509,
    -0.8358284,
];

const TALKIE_NOGAINS_GOLDEN: [f32; 48] = [
    -1.1474556,
    2.0127413,
    -0.94759905,
    -0.8410317,
    -0.16483386,
    -0.13562112,
    0.37099952,
    0.33961317,
    0.7920251,
    0.770155,
    0.7428873,
    -1.0716972,
    1.0690823,
    -0.3888023,
    -0.83508015,
    -0.42683503,
    0.81114507,
    1.4192772,
    0.9137515,
    2.446857,
    0.8867185,
    0.3292379,
    0.8382579,
    0.21753533,
    -0.78214645,
    -1.0423167,
    0.42302468,
    0.009655014,
    -1.1473806,
    -0.17310324,
    -0.7246405,
    -1.1860048,
    0.78681886,
    0.15595865,
    2.555512,
    0.7344698,
    1.3467735,
    0.5453557,
    1.3823574,
    0.28480965,
    0.91963553,
    1.2098575,
    0.668623,
    -1.3727905,
    -1.0116423,
    -2.1390824,
    -1.6982249,
    -0.78897643,
];

#[test]
fn the_converter_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(WITH_GAINS, &TALKIE_GOLDEN);
}

#[test]
fn the_shape_without_gains_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(NO_GAINS, &TALKIE_NOGAINS_GOLDEN);
}

/// The measurement the module doc quotes.
#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (WITH_GAINS, &TALKIE_GOLDEN),
        (NO_GAINS, &TALKIE_NOGAINS_GOLDEN),
    ] {
        let d = load_graph_fixture(name);
        let mut kv = graph_caches(&d);
        let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
        let kl = kl_vs_golden(&got, golden);
        let worst = worst_vs(&got, golden);
        println!("| `{name}` | {kl:.2e} | {worst:.2e} |");
        assert!(kl < 1e-8, "{name}: KL {kl}");
    }
}

/// What the loader resolved, value by value.
#[test]
fn the_loader_resolves_the_row_the_way_llama_cpp_does() {
    let d = load_graph_fixture(WITH_GAINS);
    let c = &d.config;
    assert!(c.skip_stream);
    assert_eq!(c.qk_norm_style, QkNormStyle::PerHeadScalar);
    assert!(d.qk_norm_after_rope, "talkie.cpp:82-91, after RoPE");
    assert_eq!(c.rope_layout, RopeLayout::Neox);
    assert_eq!(
        c.logit_multiplier,
        Some(0.75),
        "logit_scale, multiplied as is"
    );
    assert_eq!(d.final_norm, NormOp::RmsNoParams, "no output_norm tensor");
    for (il, layer) in d.layers.iter().enumerate() {
        assert_eq!(layer.attn.norm_weight, NormOp::RmsNoParams, "layer {il}");
        assert_eq!(layer.moe.norm_weight, NormOp::RmsNoParams, "layer {il}");
        let q = layer.attn.q_norm.as_ref().expect("the per-head gains");
        assert_eq!(q.len(), c.n_heads, "layer {il}: one gain per head");
        assert!(
            layer.attn.k_norm.is_none(),
            "layer {il}: K norms without a weight"
        );
        let near = |got: Option<f32>, want: f32| (got.unwrap() - want).abs() < 1e-6;
        assert!(
            near(layer.out_scale, 0.6 + 0.2 * il as f32),
            "layer {il} out_scale"
        );
        assert!(
            near(layer.attn.o_scale, 1.7 - 0.2 * il as f32),
            "layer {il} o_scale"
        );
        assert!(
            near(layer.moe.down_scale, 0.5 + 0.3 * il as f32),
            "layer {il} down_scale"
        );
    }
    let plain = load_graph_fixture(NO_GAINS);
    assert!(plain
        .layers
        .iter()
        .all(|l| l.attn.o_scale.is_none() && l.moe.down_scale.is_none()));
}

/// Each thing dropped in turn diverges from llama.cpp: the skip stream
/// (`out_scale` zeroed), the Q gains (set to one), and the two
/// projection gains (dropping them IS the no-gains file, whose golden
/// is the other one).
///
/// What is deliberately NOT here: the norm ORDER. A per-head RMSNorm
/// is invariant under RoPE (a rotation of pairs inside the head keeps
/// the head's norm) and a per-head scalar commutes with it, so
/// `talkie.cpp:82-91`'s "after rope" is honoured (`qk_norm_after_rope`
/// is set, the loader test above pins it) and cannot be measured on
/// this shape -- unlike hunyuan's per-element weights, where it can.
#[test]
fn each_dropped_piece_is_visible() {
    let golden = &TALKIE_GOLDEN;
    let mut d = load_graph_fixture(WITH_GAINS);
    for layer in d.layers.iter_mut() {
        layer.out_scale = Some(0.0);
    }
    assert!(decode_worst(&d, golden) > 100.0 * GRAPH_TOL, "skip stream");

    let mut d = load_graph_fixture(WITH_GAINS);
    for layer in d.layers.iter_mut() {
        let n = layer.attn.q_norm.as_ref().unwrap().len();
        layer.attn.q_norm = Some(vec![1.0; n].into());
    }
    assert!(decode_worst(&d, golden) > 100.0 * GRAPH_TOL, "Q gains");

    let mut d = load_graph_fixture(WITH_GAINS);
    for layer in d.layers.iter_mut() {
        layer.attn.o_scale = None;
        layer.moe.down_scale = None;
    }
    let worst_vs_own = decode_worst(&d, golden);
    let worst_vs_nogains = decode_worst(&d, &TALKIE_NOGAINS_GOLDEN);
    assert!(
        worst_vs_own > 100.0 * GRAPH_TOL,
        "projection gains dropped: {worst_vs_own}"
    );
    assert!(
        worst_vs_nogains < GRAPH_TOL,
        "dropping the gains IS the no-gains file: {worst_vs_nogains}"
    );
}

/// The batched body and the row body agree over twelve positions.
#[test]
fn the_prefill_body_agrees_with_the_row_body() {
    let d = load_graph_fixture(WITH_GAINS);
    let prompt: Vec<usize> = GRAPH_PROMPT
        .iter()
        .chain(GRAPH_PROMPT.iter())
        .copied()
        .collect();
    let mut kv = graph_caches(&d);
    let batched = d.forward_batch_last(&prompt, 0, &mut kv);
    let mut kv = graph_caches(&d);
    let mut rowwise = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        rowwise = d.forward_token(tok, pos, &mut kv);
    }
    let worst = worst_vs(&batched, &rowwise);
    assert!(worst < GRAPH_TOL, "batched vs row-wise differ by {worst}");
}

fn decode_worst(d: &Decoder, golden: &[f32]) -> f32 {
    let mut kv = graph_caches(d);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = d.forward_token(tok, pos, &mut kv);
    }
    worst_vs(&out, golden)
}
