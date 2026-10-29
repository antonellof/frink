//! Command-R7B, checked against llama.cpp itself: `command-r`'s graph
//! (the weighted LayerNorm without a bias, the shared-norm parallel
//! residual, a `logit_scale` multiply, a tied lm_head, NORM RoPE) with a
//! sliding window whose SLIDING layers alone are rotated.
//!
//! `cohere2.cpp:4-7` pin `swa_type = STANDARD`, seed a period of 4 and
//! let the scalar `attention.sliding_window_pattern` override it; `:13`
//! reads `attention.sliding_window` as REQUIRED; `:72,91` rotate Q and K
//! `if (is_swa)` and not otherwise. That is `rope_layers::RopeLayers::
//! SlidingOnly`, the `exaone-moe` rule, which the module's first census
//! had missed because it grepped for `use_rope`; `:14` reads
//! `logit_scale` as REQUIRED (`LogitScaleUse::AsIs`, the `talkie` use).
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `cohere2` | 4 layers, window 3 over the 6-token prompt, the seeded period (layers 0-2 slide and rotate, layer 3 is full and unrotated), `logit_scale 0.25` |
//! | `cohere2_pattern` | the same weights with `sliding_window_pattern = 2` (layers 0 and 2 slide); libllama's logits move by 0.36, so the scalar key is live and honoured |
//! | `cohere2_nowindow` | the REQUIRED window key left out; libllama refuses the file, and so does frink |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `cohere2` | 1.03e-14 | 4.77e-07 |
//! | `cohere2_pattern` | 8.95e-14 | 1.13e-06 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_cohere2_fixture.py \
//!     crates/frink-models/tests/fixtures/cohere2_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_cohere2_fixture.py \
//!     crates/frink-models/tests/fixtures/cohere2_pattern_tiny.gguf --pattern-key
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    graph_fixture_path, kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::{ModelConfig, RopeLayout};
use frink_models::loader::LoadError;
use frink_models::norm::NormOp;
use frink_models::parallel_residual::ParallelNorm;
use frink_models::rope_layers::RopeLayers;
use frink_models::Decoder;

const COHERE2: &str = "cohere2";
const PATTERN: &str = "cohere2_pattern";
const NOWINDOW: &str = "cohere2_nowindow";

const COHERE2_GOLDEN: [f32; 48] = [
    -0.053444654,
    -0.2945618,
    -0.5240313,
    0.23941867,
    0.0153282955,
    -0.20903893,
    0.037728816,
    -0.05422908,
    0.26514,
    0.12005758,
    0.031209648,
    -0.24886842,
    -0.05153007,
    -0.17520654,
    0.32748678,
    0.15941642,
    0.26152772,
    -0.2623593,
    0.019688867,
    0.11148635,
    0.2026538,
    0.35271636,
    -0.048427552,
    -0.5440997,
    0.68449545,
    0.00057941675,
    0.4054489,
    0.29115608,
    -0.54106575,
    0.19864336,
    -0.9096316,
    -0.022025362,
    -0.4299309,
    -0.05480747,
    0.004694402,
    0.31159785,
    -0.14953396,
    0.052480593,
    -0.21736613,
    0.9779248,
    0.22852951,
    -0.08411124,
    -0.88865626,
    -0.53500533,
    0.08720142,
    -0.2744,
    -0.10360582,
    0.37675047,
];

const COHERE2_PATTERN_GOLDEN: [f32; 48] = [
    0.2208364,
    -0.2998417,
    -0.39995843,
    0.30566198,
    0.055625163,
    -0.2997071,
    0.23857908,
    -0.03192523,
    0.20956117,
    0.2848893,
    -0.0003990978,
    -0.28001106,
    -0.093539596,
    -0.17675892,
    0.07347078,
    0.003204763,
    -0.09599219,
    -0.242819,
    -0.14148289,
    0.06196405,
    0.44811422,
    0.39965704,
    -0.23303573,
    -0.4134945,
    0.5024576,
    -0.013341993,
    0.68771166,
    0.5362823,
    -0.4089411,
    0.3504594,
    -0.6724138,
    0.048972473,
    -0.29651448,
    -0.07718365,
    0.055388622,
    0.40473723,
    -0.14715862,
    0.1476892,
    -0.18024069,
    0.9101582,
    0.18045573,
    -0.14951098,
    -0.8368826,
    -0.5015953,
    0.046836138,
    -0.33440673,
    -0.056326002,
    0.35608572,
];
fn decode(decoder: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    out
}

#[test]
fn cohere2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(COHERE2, &COHERE2_GOLDEN);
}

#[test]
fn the_scalar_pattern_key_is_honoured() {
    assert!(worst_vs(&COHERE2_GOLDEN, &COHERE2_PATTERN_GOLDEN) > 0.1);
    assert_all_three_paths_match(PATTERN, &COHERE2_PATTERN_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (COHERE2, &COHERE2_GOLDEN),
        (PATTERN, &COHERE2_PATTERN_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: the sliding-only rotation, the seeded period
/// (three sliding layers then one full) or the key's, the window, the
/// weighted LayerNorm with no pre-FFN slot, the logit multiplier, the
/// sliding layers' rope base following the model's.
#[test]
fn the_loaded_layers_are_the_graph() {
    assert!(frink_models::capability::uses_weighted_layer_norm(COHERE2));
    assert!(matches!(
        resolve_architecture(COHERE2),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Norm
        })
    ));
    let d = load_graph_fixture(COHERE2);
    assert_eq!(d.config.rope_layers, RopeLayers::SlidingOnly);
    assert_eq!(d.config.sliding_window, Some(3));
    let slides: Vec<bool> = (0..4)
        .map(|il| d.config.layer_sliding_window(il).is_some())
        .collect();
    assert_eq!(slides, [true, true, true, false], "period 4, last dense");
    let rotates: Vec<bool> = (0..4).map(|il| d.config.layer_rope(il).is_some()).collect();
    assert_eq!(rotates, slides, "cohere2.cpp:91: rotated iff sliding");
    assert_eq!(d.config.logit_multiplier, Some(0.25));
    assert_eq!(d.config.rope_theta, 50_000.0);
    assert!(d.config.parallel_residual);
    for layer in &d.layers {
        assert_eq!(layer.moe.parallel, Some(ParallelNorm::SharedNorm));
        assert!(matches!(layer.attn.norm_weight, NormOp::LayerNorm(_)));
        assert!(matches!(layer.moe.norm_weight, NormOp::None));
    }

    let p = load_graph_fixture(PATTERN);
    let slides: Vec<bool> = (0..4)
        .map(|il| p.config.layer_sliding_window(il).is_some())
        .collect();
    assert_eq!(slides, [true, false, true, false], "the key's period 2");
}

/// The window is REQUIRED upstream (`cohere2.cpp:13`): libllama refuses
/// a file without it (`key not found in model:
/// cohere2.attention.sliding_window`, measured on this fixture), and
/// so does frink, by name, rather than run it with no layer rotated.
#[test]
fn a_file_without_the_window_key_is_refused() {
    let file = frink_gguf::GgufFile::open(graph_fixture_path(NOWINDOW)).expect("fixture opens");
    assert_eq!(
        frink_models::swa_geometry::window_required(COHERE2),
        Some("cohere2.cpp:13")
    );
    match ModelConfig::from_gguf(&file) {
        Err(LoadError::UnsupportedFeature(_, msg)) => {
            assert!(msg.contains("attention.sliding_window"), "{msg}");
            assert!(msg.contains("cohere2.cpp:13"), "{msg}");
        }
        Err(other) => panic!("expected the missing window refused by name, got {other:?}"),
        Ok(_) => panic!("a windowless cohere2 file loaded"),
    }
}

/// Each seam sabotaged on the loaded decoder: rotating every layer
/// (what the census miss would have built), the window dropped, the
/// LayerNorm read as RMSNorm, the multiplier dropped.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(COHERE2);
    assert_decoder_matches_on_all_three_paths(&d, &COHERE2_GOLDEN, GRAPH_TOL, "baseline");

    d.config.rope_layers = RopeLayers::All;
    let worst = worst_vs(&decode(&d), &COHERE2_GOLDEN);
    assert!(worst > 1e-2, "rotating the full layer not seen: {worst}");
    d.config.rope_layers = RopeLayers::SlidingOnly;

    let saved = d.config.logit_multiplier.take();
    let worst = worst_vs(&decode(&d), &COHERE2_GOLDEN);
    assert!(worst > 1.0, "the logit multiply not seen: {worst}");
    d.config.logit_multiplier = saved;

    let weight_of = |op: &NormOp| -> Vec<f32> {
        let NormOp::LayerNorm(w) = op else {
            unreachable!()
        };
        w.clone()
    };
    let saved: Vec<NormOp> = d
        .layers
        .iter_mut()
        .map(|l| {
            let w = weight_of(&l.attn.norm_weight);
            std::mem::replace(&mut l.attn.norm_weight, NormOp::Rms(w))
        })
        .collect();
    let worst = worst_vs(&decode(&d), &COHERE2_GOLDEN);
    assert!(worst > 1e-2, "LayerNorm vs RMSNorm not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.norm_weight = s;
    }
    assert_decoder_matches_on_all_three_paths(&d, &COHERE2_GOLDEN, GRAPH_TOL, "restored");
}
