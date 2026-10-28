//! LFM3, checked against llama.cpp itself: LFM2's hybrid layer stack
//! (`llm_build_lfm_base`, `src/models/lfm-base.cpp`) with four seams
//! frink already served for other rows, each made visible here.
//!
//! `lfm3.cpp:8-10` marks layer `il` a short convolution when
//! `n_head_kv(il) == 0`; `:159-204` is LFM2's residual topology with the
//! sandwich post norms OUTSIDE the `is_recr` branch (`:178-181`,
//! `:195-198`), so a conv layer's output is normed too. The attention
//! layers (`:101-145`) are GQA with a per-head RMS QK norm, NEOX RoPE,
//! and three things LFM2 does not have:
//!
//! - a per-HEAD sigmoid gate projected from the `attn_norm` output and
//!   multiplied in before `wo` (`:77,132-139`, `crate::attn_gate`);
//! - the window ARRAY (`:13`, `crate::swa_layers`) with the sliding
//!   layers at `rope.freq_base_swa` (`:20`, REQUIRED) and the full ones
//!   at `rope.freq_base`;
//! - `rope_freqs.weight` on the FULL layers only (`:79-82,113`,
//!   `swa_geometry::swa_layers_drop_rope_factors`), which the converter
//!   writes as a partial-RoPE mask (`conversion/lfm3.py:101-111`).
//!
//! The final norm is `output_norm` (not lfm2's `token_embd_norm`
//! spelling); `output` tied when absent (`:37-41`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays are llama.cpp's own graph over each fixture,
//! through `scripts/gptoss_reference_logits.cpp` linked against a
//! `libllama` built from llama.cpp at `604cc430` (CPU, F32 KV, flash
//! attention off). The 32-step greedy decode is
//! `fixtures/decode_steps/lfm3.txt`, run by `tests/decode_step_graphs.rs`.
//!
//! | fixture | what it carries |
//! |---|---|
//! | `lfm3` | both post norms on every layer, conv layers included |
//! | `lfm3_no_post_norm` | neither (`:50-55`) |
//! | `lfm3_non_causal` | `attention.causal = false`, the d1 graph: libllama runs it (logits differ from `lfm3`'s), frink refuses it |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_lfm3_fixture.py \
//!     crates/frink-models/tests/fixtures/lfm3_tiny.gguf [--no-post-norm | --non-causal]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/lfm3_tiny.gguf 3 7 11 19 23 5
//! ```
//!
//! The fixture's layers are `[conv, sliding, full, conv, sliding]` with a
//! window of 3, head_dim 8 and the full layers rotating 2 of 4 pairs, and
//! the two bases 10000 and 500, so each seam below moves the logits on
//! its own.

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    graph_fixture_path, kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::attn_gate::GateWidth;
use frink_models::capability::{resolve_architecture, ArchPath, QkNormStyle};
use frink_models::config::RopeLayout;
use frink_models::layer_shapes::AttnShape;
use frink_models::{Decoder, LoadError, ModelConfig};

const LFM3: &str = "lfm3";
const LFM3_NO_POST_NORM: &str = "lfm3_no_post_norm";
const LFM3_NON_CAUSAL: &str = "lfm3_non_causal";

const LFM3_GOLDEN: [f32; 48] = [
    0.58787006,
    0.14019239,
    -0.8605895,
    -1.8186958,
    -0.24603444,
    2.224465,
    -0.64142704,
    1.483648,
    0.38113907,
    0.32298103,
    -0.88586175,
    1.2090461,
    0.5389815,
    -0.84506685,
    -2.0623322,
    1.4190376,
    1.3402047,
    -1.962446,
    0.7071994,
    -0.21188143,
    0.43728685,
    -0.3334981,
    -0.14865214,
    -0.5275305,
    0.04276274,
    0.12671542,
    -0.66341007,
    -1.2619722,
    -1.9058759,
    0.8329835,
    0.15964298,
    -1.892856,
    -0.9800455,
    -2.2816143,
    -0.28499207,
    2.0957756,
    -0.82427865,
    -0.8311041,
    -0.46354946,
    0.71463263,
    -1.4189851,
    -0.072505385,
    0.07706443,
    2.6908596,
    -0.24492426,
    1.1190628,
    0.8743363,
    1.6037673,
];

const LFM3_NO_POST_NORM_GOLDEN: [f32; 48] = [
    1.190139,
    -1.5253534,
    0.16093211,
    1.3198348,
    -1.9216067,
    0.54794186,
    -1.4012852,
    1.7506913,
    -0.057964526,
    -1.2550633,
    -1.1357446,
    0.7633026,
    1.1514536,
    -1.5506332,
    0.322231,
    0.3501821,
    -1.6617405,
    -0.06880698,
    0.16437805,
    -0.5395237,
    0.24613863,
    0.98703724,
    0.55515236,
    2.0668495,
    -1.4482067,
    0.36904395,
    -0.82186925,
    2.0804286,
    -0.31657124,
    0.9722032,
    1.8104296,
    1.0351534,
    -0.840288,
    -1.1093581,
    0.31823257,
    1.7767928,
    2.2543788,
    -2.318903,
    -0.33617097,
    -1.8694553,
    -2.3898175,
    0.7038256,
    0.03389918,
    1.3214649,
    0.22077352,
    1.8535094,
    -0.3325284,
    0.15271275,
];

/// Layer kinds of the fixture: conv, sliding, full, conv, sliding.
const CONV: [usize; 2] = [0, 3];
const SLIDING: [usize; 2] = [1, 4];
const FULL: usize = 2;

fn decode(decoder: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    out
}

fn refusal(name: &str) -> String {
    let path = graph_fixture_path(name);
    let file = frink_gguf::GgufFile::open(&path).expect("fixture opens");
    let err = match ModelConfig::from_gguf(&file) {
        Err(e) => e,
        Ok(config) => Decoder::from_gguf(&path, config)
            .err()
            .unwrap_or_else(|| panic!("{name} is refused")),
    };
    assert!(matches!(err, LoadError::UnsupportedFeature(..)), "{err}");
    format!("{err}")
}

#[test]
fn lfm3_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(LFM3, &LFM3_GOLDEN);
}

/// `lfm3.cpp:50-55`: the post norms are optional, both or neither.
#[test]
fn without_the_post_norms_it_matches_llama_cpp() {
    assert_all_three_paths_match(LFM3_NO_POST_NORM, &LFM3_NO_POST_NORM_GOLDEN);
    let d = load_graph_fixture(LFM3_NO_POST_NORM);
    for (il, layer) in d.layers.iter().enumerate() {
        assert!(layer.attn.post_attn_norm.is_none(), "blk.{il}");
        assert!(layer.attn.post_ffn_norm.is_none(), "blk.{il}");
    }
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (LFM3, &LFM3_GOLDEN),
        (LFM3_NO_POST_NORM, &LFM3_NO_POST_NORM_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built from the file: the layer kinds, the window on
/// the sliding layers alone, the gate per head, both RoPE bases, the
/// mask on the full layers only, and the post norm on the CONV layers.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("lfm3"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(LFM3);
    let c = &d.config;
    assert_eq!(c.qk_norm_style, QkNormStyle::PerHead);
    assert_eq!(c.n_layers, 5);
    assert_eq!(c.sliding_window, Some(3));
    assert_eq!(c.rope_theta, 25.0);
    assert_eq!(c.rope_theta_swa, Some(500.0));
    for il in CONV {
        assert_eq!(
            c.layer_shape(il).attention,
            AttnShape::ShortConv,
            "blk.{il}"
        );
        assert_eq!(
            c.layer_sliding_window(il),
            None,
            "a conv layer never slides"
        );
        let layer = &d.layers[il];
        assert!(layer.attn.output_gate.is_none(), "blk.{il}");
        assert!(
            layer.attn.post_attn_norm.is_some(),
            "lfm3.cpp:178-181 on blk.{il}"
        );
    }
    for il in SLIDING.into_iter().chain([FULL]) {
        assert_eq!(
            c.layer_shape(il).attention,
            AttnShape::Gqa {
                n_heads: 4,
                n_kv_heads: 2
            },
            "blk.{il}"
        );
        let gate = d.layers[il].attn.output_gate.as_ref().expect("attn_gate");
        assert_eq!(gate.width, GateWidth::PerHead, "blk.{il}");
        assert!(d.layers[il].attn.post_attn_norm.is_some(), "blk.{il}");
    }
    for il in SLIDING {
        assert_eq!(c.layer_sliding_window(il), Some(3), "blk.{il}");
    }
    assert_eq!(c.layer_sliding_window(FULL), None);
    let freqs = c.rope_freqs.as_ref().expect("rope_freqs.weight");
    assert_eq!(freqs.for_layer(false), &[1.0, 1.0, 1e30, 1e30]);
    assert_eq!(
        freqs.for_layer(true),
        &[1.0; 4],
        "lfm3.cpp:113: no factors on a sliding layer"
    );
}

/// Each seam moves the logits on its own, so a loader that dropped any
/// of them would be caught by the golden rather than agreeing with it
/// by accident.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let base = load_graph_fixture(LFM3);
    assert_decoder_matches_on_all_three_paths(&base, &LFM3_GOLDEN, GRAPH_TOL, "baseline");
    let diverges = |d: &Decoder, what: &str| {
        let worst = worst_vs(&decode(d), &LFM3_GOLDEN);
        println!("{what}: max |delta| {worst:.3e}");
        assert!(worst > 1e-2, "{what} not seen: {worst}");
    };

    // The per-head gate.
    let mut d = load_graph_fixture(LFM3);
    for layer in d.layers.iter_mut() {
        layer.attn.output_gate = None;
    }
    diverges(&d, "the attention gate");

    // The conv block's post norm (`:178-181` sit outside `is_recr`).
    let mut d = load_graph_fixture(LFM3);
    for il in CONV {
        d.layers[il].attn.post_attn_norm = None;
    }
    diverges(&d, "the conv layers' post norm");

    // The partial-RoPE mask on the sliding layers too.
    let mut d = load_graph_fixture(LFM3);
    if let Some(f) = d.config.rope_freqs.as_mut() {
        f.swa = None;
    }
    diverges(&d, "the mask kept off the sliding layers");

    // No mask on the full layers either.
    let mut d = load_graph_fixture(LFM3);
    d.config.rope_freqs = None;
    diverges(&d, "the full layers' mask");

    // The sliding layers at the full base.
    let mut d = load_graph_fixture(LFM3);
    d.config.rope_theta_swa = Some(d.config.rope_theta);
    diverges(&d, "rope.freq_base_swa");

    // No window.
    let mut d = load_graph_fixture(LFM3);
    d.config.sliding_window = None;
    diverges(&d, "the window");
}

/// The d1 decision checkpoints (`conversion/lfm3.py:167`) set
/// `attention.causal = false`, and `lfm3.cpp:22-25,92-96` then build a
/// graph with no memory that reads the prompt both ways. libllama runs
/// the fixture (its logits differ from `lfm3`'s); frink refuses it.
#[test]
fn the_non_causal_decision_graph_is_refused_by_name() {
    let msg = refusal(LFM3_NON_CAUSAL);
    assert!(msg.contains("attention.causal"), "{msg}");
    assert!(msg.contains("lfm3.cpp:22-25"), "{msg}");
}
