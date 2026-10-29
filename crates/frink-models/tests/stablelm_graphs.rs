//! StableLM, checked against llama.cpp itself: the sixth row of the old
//! "LayerNorm-with-bias group" (`tests/attn_bias.rs`) on
//! `NormOp::LayerNormBias`, and the two OTHER shapes `stablelm.cpp`
//! builds behind the same architecture string, each refused by name
//! from a fixture libllama runs.
//!
//! `stablelm.cpp` decides three things by TENSOR PRESENCE and none by a
//! key:
//!
//! - `ffn_norm` present (`:38-39`, `TENSOR_NOT_REQUIRED`): the ordinary
//!   sequential layer (`:129-134`). StableLM-2-1.6B, StableLM-3B-4E1T.
//!   SERVED, matched below.
//! - `ffn_norm` absent: the PARALLEL residual, `cur = inpSA` (`:135-137`)
//!   -- the FFN reads the normed input attention read and `:124,147`
//!   sum `inpL + attn + ffn`. StableLM-2-12B. SERVED
//!   (`frink_models::parallel_residual`, the `SharedNorm` arm; refused
//!   by name for one PR, matched below).
//! - `attn_q_norm` / `attn_k_norm` present (`:34-35`, `{n_embd_head_k,
//!   n_head}`, applied as `LLM_NORM` per head, `:84-97`): a per-head
//!   LAYERNORM with a distinct weight per head. StableLM-2-12B again.
//!   REFUSED by name (`frink_models::qk_layer_norm`).
//!
//! `{arch}.use_parallel_residual` is written by every export
//! (`conversion/stablelm.py:35`) and read by NOTHING in the graph:
//! libllama's logits for the sequential file with the key `true` are
//! byte-identical to the file with it `false` (measured, below), and
//! frink ignores it the same way.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `stablelm` | the served shape: six LayerNorm tensors per layer plus two on the output, Q/K/V biases, a quarter-width NEOX rotary (`rope.dimension_count = 2` of `head_dim = 8`), SwiGLU |
//! | `stablelm_parkey` | the same file with `use_parallel_residual = true`; libllama byte-identical |
//! | `stablelm_parallel` | no `ffn_norm` tensors; libllama's logits move by 8.85; matched on the parallel residual |
//! | `stablelm_qknorm` | `attn_q_norm` `{8, 4}` / `attn_k_norm` `{8, 2}`; libllama's logits move by 8.73; refused by name |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `stablelm` | 3.39e-13 | 5.96e-06 |
//! | `stablelm_parkey` | 3.39e-13 | 5.96e-06 (the same golden) |
//! | `stablelm_parallel` | 1.95e-12 | 3.93e-06 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_stablelm_fixture.py \
//!     crates/frink-models/tests/fixtures/stablelm_tiny.gguf
//! ... stablelm_parkey_tiny.gguf --par-key
//! ... stablelm_parallel_tiny.gguf --parallel
//! ... stablelm_qknorm_tiny.gguf --qk-norm
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    graph_fixture_path, kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_gguf::TensorSource;
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::{ModelConfig, RopeLayout};
use frink_models::loader::LoadError;
use frink_models::norm::NormOp;
use frink_models::parallel_residual::{layer_is_parallel, layer_parallel_norm, ParallelNorm};
use frink_models::qk_layer_norm::uses_per_head_layer_norm_qk;
use frink_models::{Decoder, FfnActivation};

const STABLELM: &str = "stablelm";
const PARKEY: &str = "stablelm_parkey";
const PARALLEL: &str = "stablelm_parallel";
const QKNORM: &str = "stablelm_qknorm";

const STABLELM_GOLDEN: [f32; 48] = [
    3.5349085,
    2.1987376,
    -1.0546974,
    0.80937195,
    -5.852913,
    0.6558612,
    -4.0061474,
    -1.091073,
    -0.016633064,
    -3.557351,
    0.9694009,
    -4.7876997,
    -1.0171344,
    -4.715365,
    8.494166,
    0.18439966,
    3.7130413,
    3.7588103,
    -1.4541855,
    -0.34355915,
    -4.2078915,
    5.6470323,
    0.6497342,
    -3.493856,
    -0.98314524,
    1.8562548,
    -2.5408292,
    -1.1405362,
    -0.94195175,
    1.4491676,
    -4.2586555,
    0.4026252,
    -4.8530436,
    -1.0419786,
    4.8740597,
    -1.1855567,
    1.176153,
    -0.23482698,
    3.959227,
    2.510057,
    -1.0698466,
    -0.086925566,
    -3.0041733,
    0.4988827,
    0.41968492,
    -2.2483754,
    -0.27294743,
    0.4179212,
];

const STABLELM_PARALLEL_GOLDEN: [f32; 48] = [
    1.2758195,
    0.374847,
    2.2136497,
    0.28281575,
    0.8447069,
    3.1614554,
    -1.8220232,
    2.9400074,
    0.13048303,
    2.282848,
    -1.9335396,
    2.955791,
    4.2844634,
    2.8044033,
    -0.35587132,
    -0.8806746,
    -1.1278564,
    -1.6981791,
    -1.9637599,
    -1.2375296,
    3.15817,
    1.049355,
    -2.9624171,
    -1.4916255,
    -1.3107316,
    4.641596,
    -2.3528836,
    1.5127604,
    -2.08676,
    -2.9950008,
    1.7161475,
    -4.832929,
    -3.0630536,
    -1.9723961,
    -3.2645502,
    1.7738419,
    -2.2311192,
    -1.6054455,
    -3.2711082,
    -1.8463331,
    3.398006,
    0.1639309,
    3.3589563,
    -1.5282083,
    2.459272,
    -0.28737998,
    1.5689645,
    1.0649877,
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
fn stablelm_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(STABLELM, &STABLELM_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (STABLELM, &STABLELM_GOLDEN),
        (PARALLEL, &STABLELM_PARALLEL_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: the biased LayerNorm at all three sites with
/// the biases read from the file (the pre-FFN pair is OPTIONAL upstream
/// and REQUIRED here as a pair), NEOX, a quarter-width rotary, the Q/K/V
/// biases, SwiGLU.
#[test]
fn the_loaded_layers_are_the_graph() {
    assert!(frink_models::capability::uses_biased_layer_norm(STABLELM));
    assert!(matches!(
        resolve_architecture(STABLELM),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(STABLELM);
    for layer in &d.layers {
        for (site, op) in [
            ("attn", &layer.attn.norm_weight),
            ("ffn", &layer.moe.norm_weight),
        ] {
            match op {
                NormOp::LayerNormBias { weight, bias } => {
                    assert_eq!(weight.len(), 32, "{site}");
                    assert_eq!(bias.len(), 32, "{site}");
                    assert!(
                        bias.iter().any(|b| b.abs() > 0.1),
                        "{site}: the bias is the file's"
                    );
                }
                other => panic!("{site}: {other:?} is not the biased LayerNorm"),
            }
        }
        assert!(layer.attn.q_bias.is_some() && layer.attn.k_bias.is_some());
        assert!(layer.attn.v_bias.is_some());
        assert!(layer.attn.o_bias.is_none(), "stablelm.cpp:31: no `wo` bias");
        assert!(layer.attn.q_norm.is_none() && layer.attn.k_norm.is_none());
    }
    assert!(matches!(d.final_norm, NormOp::LayerNormBias { .. }));
    assert_eq!(
        d.config.rope_dim,
        Some(2),
        "partial_rotary_factor 0.25 of 8"
    );
    assert_eq!(d.config.rms_norm_eps, 1e-5, "attention.layer_norm_epsilon");
    assert_eq!(d.config.ffn_activation, FfnActivation::Swiglu);
}

/// The key every export writes is dead metadata upstream: libllama's
/// logits for the file with it `true` are the sequential file's byte
/// for byte, and so are frink's. The file differs from the served one
/// by that one key.
#[test]
fn use_parallel_residual_is_ignored_as_llama_cpp_ignores_it() {
    let file = frink_gguf::GgufFile::open(graph_fixture_path(PARKEY)).expect("fixture opens");
    assert_eq!(
        file.metadata_bool("stablelm.use_parallel_residual"),
        Some(true),
        "the fixture must carry the key"
    );
    let plain = frink_gguf::GgufFile::open(graph_fixture_path(STABLELM)).unwrap();
    assert_eq!(
        plain.metadata_bool("stablelm.use_parallel_residual"),
        Some(false)
    );
    // The rule is the tensor, not the key.
    assert!(!layer_is_parallel(&file, STABLELM, 0));
    assert!(!ModelConfig::from_gguf(&file).unwrap().parallel_residual);
    assert_all_three_paths_match(PARKEY, &STABLELM_GOLDEN);
}

/// A layer with no `ffn_norm` is the parallel residual, and it is
/// SERVED (`frink_models::parallel_residual`, `SharedNorm`): the
/// fixture that evidenced its refusal for one PR matches libllama,
/// whose logits differ from the sequential file's by 8.85.
#[test]
fn the_parallel_residual_matches_llama_cpp() {
    let file = frink_gguf::GgufFile::open(graph_fixture_path(PARALLEL)).expect("fixture opens");
    assert!(file.find_tensor("blk.0.ffn_norm.weight").is_none());
    assert_eq!(
        layer_parallel_norm(&file, STABLELM, 0),
        Some(ParallelNorm::SharedNorm)
    );
    assert!(worst_vs(&STABLELM_GOLDEN, &STABLELM_PARALLEL_GOLDEN) > 1.0);
    assert_all_three_paths_match(PARALLEL, &STABLELM_PARALLEL_GOLDEN);
    let d = load_graph_fixture(PARALLEL);
    assert!(d.config.parallel_residual);
    for layer in &d.layers {
        assert_eq!(layer.moe.parallel, Some(ParallelNorm::SharedNorm));
        assert!(matches!(layer.moe.norm_weight, NormOp::None));
        assert!(matches!(
            layer.attn.norm_weight,
            NormOp::LayerNormBias { .. }
        ));
    }
}

/// A layer with `attn_q_norm` is the per-head LayerNorm: refused by
/// name from a file libllama runs, whose logits differ from the plain
/// file's by 8.73.
#[test]
fn the_per_head_layer_norm_qk_norm_is_refused_by_name() {
    assert!(uses_per_head_layer_norm_qk(STABLELM));
    let file = frink_gguf::GgufFile::open(graph_fixture_path(QKNORM)).expect("fixture opens");
    let q = file
        .find_tensor("blk.0.attn_q_norm.weight")
        .expect("the fixture carries it");
    // ne = {n_embd_head_k, n_head}: exactly the length the loader's rule
    // would have read as one RMS over the whole projection.
    assert_eq!(q.shape.iter().product::<u64>(), 32);
    let config = ModelConfig::from_gguf(&file).expect("the header is the served shape's");
    match Decoder::from_gguf(graph_fixture_path(QKNORM), config) {
        Err(LoadError::UnsupportedFeature(_, msg)) => {
            assert!(msg.contains("per-head LayerNorm"), "{msg}");
            assert!(msg.contains("stablelm.cpp:34-35,84-97"), "{msg}");
        }
        Err(other) => panic!("expected the QK LayerNorm refused by name, got {other:?}"),
        Ok(_) => panic!("the QK LayerNorm shape loaded"),
    }
}

/// Each thing the golden checks, sabotaged on the loaded decoder: the
/// pre-FFN bias zeroed (the OPTIONAL pair, so the one a loader could
/// drop quietly), the attention norm turned into dbrx's unbiased
/// LayerNorm, the K bias dropped, the rotary widened to the whole head.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(STABLELM);
    assert_decoder_matches_on_all_three_paths(&d, &STABLELM_GOLDEN, GRAPH_TOL, "baseline");

    let weight_of = |op: &NormOp| -> Vec<f32> {
        let NormOp::LayerNormBias { weight, .. } = op else {
            unreachable!()
        };
        weight.clone()
    };

    let saved: Vec<NormOp> = d
        .layers
        .iter_mut()
        .map(|l| {
            let weight = weight_of(&l.moe.norm_weight);
            std::mem::replace(
                &mut l.moe.norm_weight,
                NormOp::LayerNormBias {
                    weight,
                    bias: vec![0.0; 32],
                },
            )
        })
        .collect();
    let worst = worst_vs(&decode(&d), &STABLELM_GOLDEN);
    assert!(worst > 1e-2, "ffn_norm.bias not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.moe.norm_weight = s;
    }

    let saved: Vec<NormOp> = d
        .layers
        .iter_mut()
        .map(|l| {
            let weight = weight_of(&l.attn.norm_weight);
            std::mem::replace(&mut l.attn.norm_weight, NormOp::LayerNorm(weight))
        })
        .collect();
    let worst = worst_vs(&decode(&d), &STABLELM_GOLDEN);
    assert!(worst > 1e-2, "attn_norm.bias not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.norm_weight = s;
    }

    let saved: Vec<Option<Vec<f32>>> = d.layers.iter_mut().map(|l| l.attn.k_bias.take()).collect();
    let worst = worst_vs(&decode(&d), &STABLELM_GOLDEN);
    assert!(worst > 1e-2, "attn_k.bias not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.k_bias = s;
    }

    let saved = d.config.rope_dim.take();
    let worst = worst_vs(&decode(&d), &STABLELM_GOLDEN);
    assert!(worst > 1e-2, "the quarter-width rotary not seen: {worst}");
    d.config.rope_dim = saved;

    assert_decoder_matches_on_all_three_paths(&d, &STABLELM_GOLDEN, GRAPH_TOL, "restored");
}
