//! Kolibri-1, checked against llama.cpp itself (`src/models/kolibri1.cpp`
//! at `604cc430`, `crate::models::kolibri1` for the facts).
//!
//! The one new op is the router. `kolibri1.cpp:138` ranks the experts by
//! `logits + exp_probs_b` BEFORE the sigmoid and hands the ids to
//! `build_moe_ffn` (`:155`); every other biased router ranks
//! `sigmoid(logits) + exp_probs_b` (`llama-graph.cpp:2279`). That is
//! `frink_moe::SelectionRule`, and this fixture is drawn so the two
//! rules pick different experts: router logits spread about 2.5 against
//! a bias uniform in [-1.5, 1.5], so the bias reorders the gated scores
//! almost entirely and the raw logits only partly.
//! `each_seam_is_visible_in_the_logits` measures it.
//!
//! The rest is rows on served seams: the window ARRAY (`:17`) with layer
//! 2 full and unrotated (`:88`), per-head QK RMS (`:47-48`), REQUIRED
//! sandwich norms with `ffn_post_norm` over routed + shared
//! (`:41,51,165-167`), one ungated shared expert, gating and `norm_w`
//! from the file (`:8-9`). Three keys the graph never reads are written
//! at values that would move the logits if they were honoured:
//! `rope.freq_base_swa` (7, against a model base of 50; `:19` seeds the
//! sliding base from the model's), `leading_dense_block_count` (1) and
//! `expert_weights_scale` (2.5).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays are llama.cpp's own graph over each fixture,
//! through `scripts/gptoss_reference_logits.cpp` linked against a
//! `libllama` built from llama.cpp at `604cc430` (CPU, F32 KV, flash
//! attention off); CI regenerates them (`tools/oracle/goldens.tsv`). The
//! 32-step greedy decode is `fixtures/decode_steps/kolibri1.txt`, run by
//! `tests/decode_step_graphs.rs`.
//!
//! | fixture | what it carries |
//! |---|---|
//! | `kolibri1` | sigmoid gating, `expert_weights_norm = true` |
//! | `kolibri1_softmax` | softmax gating, `expert_weights_norm = false` |
//! | `kolibri1_no_bias` | no `exp_probs_b` (`:54`, REQUIRED): libllama refuses it, so does frink |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_kolibri1_fixture.py \
//!     crates/frink-models/tests/fixtures/kolibri1_tiny.gguf [--softmax | --no-bias]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/kolibri1_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    graph_fixture_path, kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath, QkNormStyle};
use frink_models::config::RopeLayout;
use frink_models::rope_layers::RopeLayers;
use frink_models::{Decoder, LoadError, ModelConfig};
use frink_moe::{GatingFunction, SelectionRule};

const KOLIBRI1: &str = "kolibri1";
const KOLIBRI1_SOFTMAX: &str = "kolibri1_softmax";
const KOLIBRI1_NO_BIAS: &str = "kolibri1_no_bias";

const KOLIBRI1_GOLDEN: [f32; 48] = [
    -0.8791073,
    1.4592007,
    -0.03869921,
    -0.5347499,
    -1.5297122,
    0.17696203,
    -0.1102838,
    -0.071661234,
    -1.1705444,
    -1.388371,
    -1.5843723,
    -2.1758115,
    2.183664,
    1.6696239,
    -1.1543493,
    0.8608104,
    2.685273,
    0.018152446,
    -0.029788733,
    -1.8109256,
    0.8986406,
    1.1057873,
    0.24528068,
    -0.73611945,
    0.30004317,
    1.0775893,
    -2.3257551,
    -1.9981058,
    -0.059066713,
    -1.06089,
    -2.5711627,
    -0.91058826,
    -1.9354644,
    1.3023019,
    -1.6918095,
    -0.72833395,
    -1.8193686,
    -1.3216647,
    0.07604593,
    -0.3123489,
    -0.49774456,
    -0.9741819,
    0.38736376,
    0.7859921,
    2.8471262,
    -0.7183766,
    -0.9413678,
    -1.1434343,
];

const KOLIBRI1_SOFTMAX_GOLDEN: [f32; 48] = [
    -1.0401052,
    0.8423997,
    0.7842827,
    -1.2144084,
    -1.2773819,
    -0.17690712,
    -0.46097827,
    1.2655966,
    0.5229932,
    -1.5206184,
    -0.34382045,
    -0.7061531,
    2.8139799,
    1.7575668,
    -2.7654696,
    1.7383617,
    1.1699398,
    -0.29892677,
    -1.462467,
    -1.6102831,
    1.963362,
    1.5679622,
    -0.009984136,
    -1.2956657,
    1.2289457,
    2.1829264,
    -2.8904114,
    -0.042182535,
    0.1501311,
    1.6620368,
    -2.9984086,
    -1.2742969,
    -1.3503804,
    0.6871415,
    -0.42404783,
    -0.716764,
    -1.3415068,
    -1.7184637,
    -0.46664774,
    -1.4339191,
    -0.7551242,
    0.24251273,
    1.4997609,
    0.7329362,
    2.756067,
    -1.2960609,
    -0.3379761,
    -1.7769586,
];

/// The fixture's layers: sliding, sliding, full, sliding.
const SLIDING: [usize; 3] = [0, 1, 3];
const FULL: usize = 2;

fn decode(decoder: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    out
}

#[test]
fn kolibri1_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(KOLIBRI1, &KOLIBRI1_GOLDEN);
}

/// The other branch of both routing keys (`kolibri1.cpp:8-9`).
#[test]
fn softmax_gating_without_renormalising_matches_llama_cpp() {
    assert_all_three_paths_match(KOLIBRI1_SOFTMAX, &KOLIBRI1_SOFTMAX_GOLDEN);
    let c = load_graph_fixture(KOLIBRI1_SOFTMAX).config;
    assert_eq!(c.moe.gating, GatingFunction::Softmax);
    assert!(!c.moe.norm_topk_prob);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (KOLIBRI1, &KOLIBRI1_GOLDEN),
        (KOLIBRI1_SOFTMAX, &KOLIBRI1_SOFTMAX_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built from the file.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture(KOLIBRI1),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(KOLIBRI1);
    let c = &d.config;
    assert_eq!(c.qk_norm_style, QkNormStyle::PerHead);
    assert_eq!(c.n_layers, 4);
    assert_eq!(c.moe.selection, SelectionRule::BiasBeforeGating);
    assert_eq!(c.moe.gating, GatingFunction::Sigmoid);
    assert!(c.moe.norm_topk_prob);
    assert_eq!(c.moe.n_shared_experts, 1);
    // The three dead keys.
    assert_eq!(c.rope_theta, 50.0);
    assert_eq!(c.rope_theta_swa, Some(50.0), "kolibri1.cpp:19");
    assert_eq!(c.moe.expert_weights_scale, 1.0, "kolibri1.cpp:150");
    assert_eq!(c.n_dense_leading_layers, 0, "kolibri1.cpp:52-61");
    assert_eq!(c.rope_layers, RopeLayers::SlidingOnly);
    for il in SLIDING {
        assert_eq!(c.layer_sliding_window(il), Some(3), "blk.{il}");
    }
    assert_eq!(c.layer_sliding_window(FULL), None);
    for (il, layer) in d.layers.iter().enumerate() {
        assert!(layer.attn.post_attn_norm.is_some(), "blk.{il}");
        assert!(layer.attn.post_ffn_norm.is_some(), "blk.{il}");
        assert!(layer.moe.exp_probs_bias.is_some(), "blk.{il}");
        assert_eq!(layer.moe.shared_experts.len(), 1, "blk.{il}");
    }
}

/// Each seam moves the logits on its own, so a loader that dropped any
/// of them is caught by the golden rather than agreeing with it by
/// accident. The first is the reason the fixture exists.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let base = load_graph_fixture(KOLIBRI1);
    assert_decoder_matches_on_all_three_paths(&base, &KOLIBRI1_GOLDEN, GRAPH_TOL, "baseline");
    let diverges = |d: &Decoder, what: &str| {
        let worst = worst_vs(&decode(d), &KOLIBRI1_GOLDEN);
        println!("{what}: max |delta| {worst:.3e}");
        assert!(worst > 1e-2, "{what} not seen: {worst}");
    };

    // DeepSeek-V3's rule, `sigmoid(l) + b`, where the graph ranks `l + b`.
    let mut d = load_graph_fixture(KOLIBRI1);
    d.config.moe.selection = SelectionRule::BiasAfterGating;
    diverges(&d, "the bias added after the sigmoid");

    // The dead keys, honoured.
    let mut d = load_graph_fixture(KOLIBRI1);
    d.config.rope_theta_swa = Some(7.0);
    diverges(&d, "rope.freq_base_swa");
    let mut d = load_graph_fixture(KOLIBRI1);
    d.config.moe.expert_weights_scale = 2.5;
    diverges(&d, "expert_weights_scale");

    // The full layer rotated.
    let mut d = load_graph_fixture(KOLIBRI1);
    d.config.rope_layers = RopeLayers::All;
    diverges(&d, "RoPE on the full layer");

    // No renormalisation.
    let mut d = load_graph_fixture(KOLIBRI1);
    d.config.moe.norm_topk_prob = false;
    diverges(&d, "expert_weights_norm");

    // The post-FFN norm, which covers the shared expert too.
    let mut d = load_graph_fixture(KOLIBRI1);
    for layer in d.layers.iter_mut() {
        layer.attn.post_ffn_norm = None;
    }
    diverges(&d, "ffn_post_norm");

    // The shared expert.
    let mut d = load_graph_fixture(KOLIBRI1);
    for layer in d.layers.iter_mut() {
        layer.moe.shared_experts.clear();
    }
    diverges(&d, "the shared expert");
}

/// `kolibri1.cpp:54` creates `exp_probs_b` with flag 0, so libllama
/// refuses a file without it; the generic loader reads it as optional,
/// and `crate::models::kolibri1`'s load checks are what refuse it here.
#[test]
fn a_file_without_the_selection_bias_is_refused_by_name() {
    let path = graph_fixture_path(KOLIBRI1_NO_BIAS);
    let file = frink_gguf::GgufFile::open(&path).expect("fixture opens");
    let err = ModelConfig::from_gguf(&file).expect_err("refused");
    assert!(matches!(err, LoadError::UnsupportedFeature(..)), "{err}");
    let msg = format!("{err}");
    assert!(msg.contains("blk.0.exp_probs_b.bias"), "{msg}");
    assert!(msg.contains("kolibri1.cpp"), "{msg}");
}
