//! Orion and Nemotron, checked against llama.cpp itself: the LayerNorm
//! WITH A BIAS at every norm site, the variant `capability::
//! WEIGHTED_LAYER_NORM` had named as having no caller.
//!
//! Eight architectures create `attn_norm.bias` / `ffn_norm.bias` /
//! `output_norm.bias` as REQUIRED and `build_norm(x, w, b, LLM_NORM,
//! il)` multiplies by `w` and then adds `b`. `tests/attn_bias.rs` had
//! them all refused with the bias named. Read one by one, two of them
//! need NOTHING ELSE of the generic decoder, and those two close here
//! on `NormOp::LayerNormBias` (`capability::BIASED_LAYER_NORM`):
//!
//! - `orion` (Orion-14B): a Llama with NEOX RoPE, no `rope.dimension_count`
//!   and no `rope.freq_base` in the file (`conversion/orion.py:13-37`).
//! - `nemotron` (Nemotron-4, Minitron): the ungated ReLU-squared FFN
//!   `arcee` already serves, partial NEOX RoPE, and three OPTIONAL
//!   biases (`nemotron.cpp:31,40-41`), served through
//!   `frink_models::proj_bias`.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `orion` | the converter's shape: six per-layer norm tensors and two output-norm tensors, NEOX at the defaults |
//! | `nemotron` | the same norm on the ReLU-squared graph with a half-width rotary and `rope.scaling.type = none` |
//! | `nemotron_biases` | the optional `attn_output.bias` / `ffn_up.bias` / `ffn_down.bias` present; libllama applies them (its logits move by 8.07) and so does frink (`frink_models::proj_bias`; the file was refused as unread until then) |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `orion` | 2.33e-11 | 1.64e-05 (see `ORION_TOL`) |
//! | `nemotron` | 5.13e-13 | 4.86e-06 |
//! | `nemotron_biases` | 2.12e-13 | 1.67e-06 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_orion_fixture.py \
//!     crates/frink-models/tests/fixtures/orion_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nemotron_fixture.py \
//!     crates/frink-models/tests/fixtures/nemotron_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nemotron_fixture.py \
//!     crates/frink-models/tests/fixtures/nemotron_biases_tiny.gguf --biases
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_all_three_paths_match_within,
    assert_decoder_matches_on_all_three_paths, graph_caches, kl_vs_golden, load_graph_fixture,
    worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath, BIASED_LAYER_NORM};
use frink_models::config::RopeLayout;
use frink_models::norm::NormOp;
use frink_models::{Decoder, FfnActivation};

const ORION: &str = "orion";
const NEMOTRON: &str = "nemotron";
const NEMOTRON_BIASES: &str = "nemotron_biases";

/// Orion sits at 1.6e-5 max |delta| (KL 2.3e-11), above the 1e-5 line
/// the suite holds every other fixture to and two orders under the next
/// tolerance the suite has. It is noise, and it was measured rather
/// than assumed: drawing every projection at unit scale still leaves
/// 1.2e-5 (KL 6e-12), where `nemotron` -- the same norm on a ReLU-squared
/// FFN -- is at 4.9e-6 and `dbrx` / `olmo`, the other two LayerNorm
/// rows, are at 1e-12 or better. What is different is a SwiGLU fed a
/// biased (non-zero-mean) LayerNorm output on a libllama built with
/// Accelerate (`vDSP_measqv`, `sgemm`), which is where f32 summation
/// order shows. The line here is 5e-5, and the sabotages below move the
/// logits by three orders more than it.
const ORION_TOL: f32 = 5e-5;

const ORION_GOLDEN: [f32; 48] = [
    -1.0600793,
    2.4058762,
    1.1829354,
    2.4633183,
    1.2401165,
    -4.8361464,
    0.13678956,
    -1.2572477,
    -2.4184341,
    0.30853915,
    0.39239192,
    3.318696,
    2.663734,
    0.6739796,
    -4.361993,
    2.0904078,
    1.5585349,
    -1.7527168,
    0.0007056892,
    -1.1774929,
    -1.5299386,
    -2.4000227,
    -0.23690438,
    -0.9865655,
    -6.229175,
    2.3851454,
    -4.575431,
    -2.1592624,
    -0.32663417,
    1.8916459,
    2.5984278,
    -0.46903995,
    -0.612654,
    -3.0993576,
    -0.49888408,
    3.3013432,
    0.4252932,
    -0.14596611,
    -0.26242423,
    -2.664662,
    -1.8994002,
    1.4912342,
    2.2139351,
    1.3733842,
    0.8680297,
    0.922374,
    -2.8612056,
    0.6219318,
];

const NEMOTRON_GOLDEN: [f32; 48] = [
    0.021555483,
    1.8955895,
    -1.3320845,
    6.056012,
    3.3412895,
    0.912037,
    -1.1567304,
    1.1021042,
    -0.4614247,
    0.8820132,
    1.3762077,
    0.72714293,
    2.0539532,
    -0.36992747,
    0.05811405,
    0.05993949,
    -2.9562664,
    -0.99142957,
    0.92990506,
    0.5021541,
    0.31386447,
    0.8486861,
    0.76540935,
    -1.5778431,
    -3.19172,
    0.8465306,
    2.014451,
    2.4971147,
    -1.4107871,
    3.3293924,
    1.2479118,
    -3.9631605,
    1.7998744,
    2.641799,
    -6.124082,
    -1.3780648,
    0.8110308,
    -0.9004832,
    0.13881707,
    -1.7073963,
    -0.5648134,
    -0.123278975,
    -0.69075954,
    2.6854124,
    1.5081806,
    0.19908547,
    1.4768732,
    -4.9410467,
];

const NEMOTRON_BIASES_GOLDEN: [f32; 48] = [
    -0.09904046,
    1.3401713,
    2.5483193,
    3.2855296,
    0.75442386,
    0.72979254,
    -0.56891817,
    -0.63956404,
    3.7740107,
    1.2493708,
    -0.21804845,
    -1.5416408,
    4.3261623,
    3.3956466,
    1.4229823,
    6.0028987,
    -1.0376859,
    3.2586093,
    1.7087011,
    0.9277673,
    1.2896769,
    -3.8718057,
    0.20172787,
    2.3414621,
    -2.0978537,
    6.1882367,
    -0.6020052,
    -0.21269107,
    1.4116989,
    -4.145981,
    3.3045237,
    -3.8645291,
    1.1280963,
    3.9081454,
    -2.9068398,
    3.806529,
    3.723837,
    -2.737557,
    0.4906417,
    1.7376441,
    3.0176263,
    -0.11307931,
    0.8661195,
    -1.4732631,
    0.4466896,
    0.3666165,
    0.53472936,
    3.1325316,
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
fn orion_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(ORION, &ORION_GOLDEN, ORION_TOL);
}

#[test]
fn nemotron_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(NEMOTRON, &NEMOTRON_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (ORION, &ORION_GOLDEN),
        (NEMOTRON, &NEMOTRON_GOLDEN),
        (NEMOTRON_BIASES, &NEMOTRON_BIASES_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: every norm site is the biased LayerNorm,
/// with the bias read from the file; the two rows are NEOX; orion's
/// rotary defaults and nemotron's half-width rotary and ReLU-squared
/// FFN are what the graphs read.
#[test]
fn the_loaded_layers_are_the_two_graphs() {
    assert!(BIASED_LAYER_NORM.starts_with(&["orion", "nemotron"]));
    for name in [ORION, NEMOTRON] {
        assert!(matches!(
            resolve_architecture(name),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        ));
        let d = load_graph_fixture(name);
        for layer in &d.layers {
            for (site, op) in [
                ("attn", &layer.attn.norm_weight),
                ("ffn", &layer.moe.norm_weight),
            ] {
                match op {
                    NormOp::LayerNormBias { weight, bias } => {
                        assert_eq!(weight.len(), 32, "{name} {site}");
                        assert_eq!(bias.len(), 32, "{name} {site}");
                        assert!(
                            bias.iter().any(|b| b.abs() > 0.1),
                            "{name} {site}: the bias is the file's"
                        );
                    }
                    other => panic!("{name} {site}: {other:?} is not the biased LayerNorm"),
                }
            }
        }
        assert!(matches!(d.final_norm, NormOp::LayerNormBias { .. }));
    }
    let orion = load_graph_fixture(ORION);
    assert_eq!(
        orion.config.rope_dim, None,
        "no rope.dimension_count: the whole head rotates"
    );
    assert_eq!(
        orion.config.rope_theta, 10000.0,
        "no rope.freq_base: llama.cpp's default"
    );
    assert_eq!(
        orion.config.rms_norm_eps, 1e-5,
        "attention.layer_norm_epsilon"
    );
    let nemotron = load_graph_fixture(NEMOTRON);
    assert_eq!(nemotron.config.rope_dim, Some(4));
    assert_eq!(nemotron.config.ffn_activation, FfnActivation::ReluSqr);
}

/// Nemotron's optional projection biases: `nemotron.cpp:31,40-41`
/// create them `TENSOR_NOT_REQUIRED` and llama.cpp applies them (the
/// two goldens differ by 8.07). They are served through
/// `frink_models::proj_bias` -- the file used to be refused as
/// carrying unread tensors -- and match libllama.
#[test]
fn nemotrons_optional_projection_biases_are_applied() {
    assert_all_three_paths_match(NEMOTRON_BIASES, &NEMOTRON_BIASES_GOLDEN);
    let d = load_graph_fixture(NEMOTRON_BIASES);
    for layer in &d.layers {
        assert!(layer.attn.o_bias.is_some());
        let bias = layer.moe.dense_bias.as_ref().expect("read");
        assert!(bias.up.is_some() && bias.down.is_some() && bias.gate.is_none());
    }
    assert!(worst_vs(&NEMOTRON_GOLDEN, &NEMOTRON_BIASES_GOLDEN) > 1.0);
}

/// Each half of the norm, sabotaged on the loaded decoder: the bias
/// zeroed and the weight flattened each move the logits past the
/// tolerance, on both rows.
#[test]
fn the_bias_and_the_weight_are_each_visible_in_the_logits() {
    for (name, golden, tol) in [
        (ORION, &ORION_GOLDEN, ORION_TOL),
        (NEMOTRON, &NEMOTRON_GOLDEN, GRAPH_TOL),
    ] {
        let mut d = load_graph_fixture(name);
        assert_decoder_matches_on_all_three_paths(&d, golden, tol, name);

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
                let weight = weight_of(&l.attn.norm_weight);
                std::mem::replace(
                    &mut l.attn.norm_weight,
                    NormOp::LayerNormBias {
                        weight,
                        bias: vec![0.0; 32],
                    },
                )
            })
            .collect();
        let worst = worst_vs(&decode(&d), golden);
        assert!(worst > 1e-2, "{name}: attn_norm.bias not seen: {worst}");
        for (l, s) in d.layers.iter_mut().zip(saved) {
            l.attn.norm_weight = s;
        }

        // The weight without the bias: `NormOp::LayerNorm`, dbrx's form,
        // is the plausible wrong variant.
        let saved: Vec<NormOp> = d
            .layers
            .iter_mut()
            .map(|l| {
                let weight = weight_of(&l.moe.norm_weight);
                std::mem::replace(&mut l.moe.norm_weight, NormOp::LayerNorm(weight))
            })
            .collect();
        let worst = worst_vs(&decode(&d), golden);
        assert!(worst > 1e-2, "{name}: ffn_norm.bias not seen: {worst}");
        for (l, s) in d.layers.iter_mut().zip(saved) {
            l.moe.norm_weight = s;
        }

        // And RMSNorm with the weight -- what the generic path used to
        // compute for these files -- on the final norm.
        let weight = weight_of(&d.final_norm);
        let saved = std::mem::replace(&mut d.final_norm, NormOp::Rms(weight));
        let worst = worst_vs(&decode(&d), golden);
        assert!(
            worst > 1e-2,
            "{name}: output_norm as RMSNorm not seen: {worst}"
        );
        d.final_norm = saved;

        assert_decoder_matches_on_all_three_paths(&d, golden, tol, "restored");
    }
}
