//! Command-R, checked against llama.cpp itself: the shared-norm parallel
//! residual (`frink_models::parallel_residual`) over the weighted
//! LayerNorm WITHOUT a bias (`capability::WEIGHTED_LAYER_NORM`, the
//! variant `dbrx` gave its first caller), a `logit_scale` MULTIPLY that
//! the graph skips when the key is absent or zero
//! (`frink_models::scalar_multipliers`, `LogitScaleUse::AsIsOptional`),
//! a tied lm_head, NORM RoPE. Command-R 35B and Aya-23 are this shape.
//!
//! Command-R+ (104B, 64 layers) is the same graph with the per-head
//! LayerNorm QK norm `command-r.cpp:28-31` REQUIRE at `n_layer >= 64`,
//! and that op is refused by name (`frink_models::qk_layer_norm`) from
//! a 64-layer fixture libllama runs.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `command_r` | the served shape with `logit_scale = 0.0625`, as every real export declares |
//! | `command_r_noscale` | the same weights with no `logit_scale` key: libllama's logits are the scaled file's divided by 0.0625 (measured, ratio 0.0625 to 1e-9), so absent means no scale and not a refusal |
//! | `command_r_plus` | 64 layers with `attn_q_norm` `{8, 2}` / `attn_k_norm` `{8, 1}`; libllama runs it; refused by name |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `command_r` | 1.02e-15 | 1.49e-07 |
//! | `command_r_noscale` | 2.26e-13 | 2.38e-06 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_command_r_fixture.py \
//!     crates/frink-models/tests/fixtures/command_r_tiny.gguf
//! ... command_r_noscale_tiny.gguf --no-logit-scale
//! ... command_r_plus_tiny.gguf --plus
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
use frink_models::Decoder;

const COMMAND_R: &str = "command_r";
const NOSCALE: &str = "command_r_noscale";
const PLUS: &str = "command_r_plus";

const COMMAND_R_GOLDEN: [f32; 48] = [
    0.07120129,
    -0.0658564,
    0.075167604,
    -0.03013523,
    -0.005875893,
    -0.098735176,
    0.090072975,
    -0.050787985,
    -0.0146826655,
    0.016819555,
    -0.014875419,
    -0.073848434,
    -0.12244835,
    -0.0037116595,
    -0.062389098,
    -0.05820402,
    -0.04317017,
    0.20395382,
    -0.06363838,
    -0.09944917,
    0.035045836,
    -0.15262796,
    -0.004360944,
    -0.10448907,
    -0.0140059665,
    -0.11419049,
    -0.14803372,
    0.16377491,
    -0.06749434,
    0.03222097,
    -0.057085104,
    0.08956219,
    0.0523116,
    -0.28689998,
    -0.026316267,
    -0.0629535,
    -0.05267317,
    -0.09888828,
    -0.034774955,
    -0.020797234,
    -0.16666314,
    -0.1091035,
    0.115461305,
    -0.105705485,
    -0.061054923,
    -0.12041834,
    0.015898105,
    -0.030510504,
];

const COMMAND_R_NOSCALE_GOLDEN: [f32; 48] = [
    1.1392206,
    -1.0537024,
    1.2026817,
    -0.48216367,
    -0.09401429,
    -1.5797628,
    1.4411676,
    -0.81260777,
    -0.23492265,
    0.26911288,
    -0.23800671,
    -1.181575,
    -1.9591736,
    -0.05938655,
    -0.99822557,
    -0.93126434,
    -0.6907227,
    3.263261,
    -1.0182141,
    -1.5911868,
    0.5607334,
    -2.4420474,
    -0.069775105,
    -1.6718252,
    -0.22409546,
    -1.8270478,
    -2.3685396,
    2.6203985,
    -1.0799094,
    0.51553553,
    -0.91336167,
    1.4329951,
    0.8369856,
    -4.5903997,
    -0.42106026,
    -1.007256,
    -0.8427707,
    -1.5822124,
    -0.5563993,
    -0.33275574,
    -2.6666102,
    -1.745656,
    1.8473809,
    -1.6912878,
    -0.97687876,
    -1.9266934,
    0.25436968,
    -0.48816806,
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
fn command_r_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(COMMAND_R, &COMMAND_R_GOLDEN);
}

#[test]
fn command_r_without_the_logit_scale_key_matches_llama_cpp() {
    assert_all_three_paths_match(NOSCALE, &COMMAND_R_NOSCALE_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (COMMAND_R, &COMMAND_R_GOLDEN),
        (NOSCALE, &COMMAND_R_NOSCALE_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: every layer the shared-norm parallel residual
/// with the attention norm a weighted LayerNorm and NO pre-FFN norm, the
/// final norm the same function, the logit multiplier the file's, the
/// lm_head tied, NORM RoPE at the Command-R base.
#[test]
fn the_loaded_layers_are_the_graph() {
    assert!(frink_models::capability::uses_weighted_layer_norm(
        "command-r"
    ));
    assert!(matches!(
        resolve_architecture("command-r"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Norm
        })
    ));
    let d = load_graph_fixture(COMMAND_R);
    assert!(d.config.parallel_residual);
    assert_eq!(d.config.logit_multiplier, Some(0.0625));
    assert_eq!(d.config.rope_theta, 8_000_000.0);
    assert_eq!(d.config.rms_norm_eps, 1e-5, "attention.layer_norm_epsilon");
    // `command-r.cpp:21`: `output` is `TENSOR_DUPLICATED` from the
    // embeddings, and the file has no `output.weight`.
    let file = frink_gguf::GgufFile::open(graph_fixture_path(COMMAND_R)).unwrap();
    assert!(file.find_tensor("output.weight").is_none());
    assert_eq!(
        (d.output_head.rows(), d.output_head.cols()),
        (d.embedding.rows(), d.embedding.cols())
    );
    for layer in &d.layers {
        assert_eq!(layer.moe.parallel, Some(ParallelNorm::SharedNorm));
        assert!(matches!(layer.attn.norm_weight, NormOp::LayerNorm(_)));
        assert!(matches!(layer.moe.norm_weight, NormOp::None));
        assert!(layer.attn.q_norm.is_none() && layer.attn.k_norm.is_none());
    }
    assert!(matches!(d.final_norm, NormOp::LayerNorm(_)));

    let n = load_graph_fixture(NOSCALE);
    assert_eq!(n.config.logit_multiplier, None, "absent means no scale");
    // The two goldens ARE the multiply: the same logits, scaled.
    let ratio = COMMAND_R_GOLDEN[3] / COMMAND_R_NOSCALE_GOLDEN[3];
    assert!((ratio - 0.0625).abs() < 1e-6, "{ratio}");
}

/// Command-R+'s per-head LayerNorm QK norm: refused by name, from a
/// 64-layer fixture whose header loads (nothing in it says QK norm) and
/// whose tensors llama.cpp REQUIRES at that depth.
#[test]
fn command_r_plus_is_refused_by_name() {
    let file = frink_gguf::GgufFile::open(graph_fixture_path(PLUS)).expect("fixture opens");
    let q = file
        .find_tensor("blk.63.attn_q_norm.weight")
        .expect("REQUIRED at 64 layers");
    assert_eq!(
        q.shape.iter().product::<u64>(),
        16,
        "{{8, 2}}: two heads of eight"
    );
    let config = ModelConfig::from_gguf(&file).expect("the header is the served shape's");
    assert_eq!(config.n_layers, 64);
    match Decoder::from_gguf(graph_fixture_path(PLUS), config) {
        Err(LoadError::UnsupportedFeature(_, msg)) => {
            assert!(msg.contains("per-head LayerNorm"), "{msg}");
            assert!(msg.contains("command-r.cpp:28-31,80,87"), "{msg}");
        }
        Err(other) => panic!("expected the QK LayerNorm refused by name, got {other:?}"),
        Ok(_) => panic!("Command-R+ loaded"),
    }
}

/// Each seam sabotaged on the loaded decoder: the norm turned into the
/// biased form with a zero bias is a no-op (so that is NOT the
/// sabotage); the norm turned into RMSNorm (what the generic path
/// would have computed), the parallel residual switched off, the
/// logit multiplier dropped, each diverge past the tolerance.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(COMMAND_R);
    assert_decoder_matches_on_all_three_paths(&d, &COMMAND_R_GOLDEN, GRAPH_TOL, "baseline");

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
    let worst = worst_vs(&decode(&d), &COMMAND_R_GOLDEN);
    assert!(worst > 1e-2, "LayerNorm vs RMSNorm not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.norm_weight = s;
    }

    for l in d.layers.iter_mut() {
        l.moe.parallel = None;
    }
    let worst = worst_vs(&decode(&d), &COMMAND_R_GOLDEN);
    assert!(worst > 1e-2, "the parallel residual not seen: {worst}");
    for l in d.layers.iter_mut() {
        l.moe.parallel = Some(ParallelNorm::SharedNorm);
    }

    let saved = d.config.logit_multiplier.take();
    let worst = worst_vs(&decode(&d), &COMMAND_R_GOLDEN);
    assert!(worst > 1.0, "the logit multiply not seen: {worst}");
    d.config.logit_multiplier = saved;

    assert_decoder_matches_on_all_three_paths(&d, &COMMAND_R_GOLDEN, GRAPH_TOL, "restored");
}
