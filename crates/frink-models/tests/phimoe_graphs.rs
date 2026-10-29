//! Phi-3.5-MoE, checked against llama.cpp itself: `phi3`'s graph
//! (`models.h:632`, `using graph = llama_model_phi3::graph`) on
//! `phimoe.cpp`'s tensors, which differ from a Phi-3 file in exactly the
//! biases: an RMSNorm WITH a bias at every norm site (`phimoe.cpp:20-21,
//! 28-29,35-36` create the pairs REQUIRED; `phi3.cpp:99-102,137-139,
//! 174-177` pass them to `LLM_NORM_RMS`) -- `NormOp::RmsBias`, one
//! graph of 155 on the generic path -- plus `attn_output.bias` and
//! `output.bias` (`:33,23`, REQUIRED, `frink_models::proj_bias`). The
//! old refusal had called the norm biases LayerNorm biases; they are
//! not, and the fixture is what says so (an RMS body with the bias
//! added matches, a LayerNorm body would not).
//!
//! The rest is served already: softmax top-2 routing renormalised
//! (`phi3.cpp:153-163`, `norm_w = true`), LongRoPE's factor pair picked
//! by context with `rope.scaling.attn_factor`, NEOX, and the window key
//! every export writes that `phimoe.cpp:3-10` never read
//! (`capability::swa_window_override`, the `phi3` answer; libllama
//! reports `n_swa = 0` for this fixture, measured).
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `phimoe` | `context_length 64` over `original 32`: the LONG factor pair in use, `attn_factor 1.0955`, a window of 8 that must be ignored |
//! | `phimoe_plain` | no factor tensors, no attn factor, context = original: plain NEOX; libllama's logits differ from the LongRoPE file's by 7.7 |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `phimoe` | 1.91e-11 | 1.48e-05 (see `PHIMOE_TOL`) |
//! | `phimoe_plain` | 1.86e-12 | 8.82e-06 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_phimoe_fixture.py \
//!     crates/frink-models/tests/fixtures/phimoe_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_phimoe_fixture.py \
//!     crates/frink-models/tests/fixtures/phimoe_plain_tiny.gguf --no-longrope
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match_within, assert_decoder_matches_on_all_three_paths, graph_caches,
    kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::RopeLayout;
use frink_models::norm::NormOp;
use frink_models::proj_bias::{attn_out_presence, output_presence, Presence};
use frink_models::Decoder;

const PHIMOE: &str = "phimoe";
const PLAIN: &str = "phimoe_plain";

/// The LongRoPE file sits at 1.5e-5 max |delta| (KL 1.9e-11), the plain
/// one at 8.8e-6 here and 1.07e-5 on CI's x86 host: the `orion` class
/// (`tests/biased_layer_norm_graphs.rs`,
/// `ORION_TOL`), a SwiGLU fed a biased, non-zero-mean norm output on a
/// libllama built with Accelerate, where f32 summation order shows. The
/// line is 5e-5, as there; every sabotage below moves the logits by
/// three orders more.
const PHIMOE_TOL: f32 = 5e-5;

const PHIMOE_GOLDEN: [f32; 48] = [
    0.5518457,
    1.0194962,
    3.6445029,
    1.9097971,
    -0.5623789,
    -4.2708077,
    1.3803577,
    -2.9604185,
    -0.022438288,
    2.4192545,
    -1.0429044,
    4.034354,
    -2.1667616,
    3.951693,
    -1.2134359,
    -0.7885203,
    0.58892286,
    0.7717161,
    -3.7656345,
    0.346936,
    3.3737583,
    -0.38977277,
    1.4990072,
    0.5621855,
    3.6714025,
    -2.5282867,
    2.665083,
    -0.20153293,
    0.39950007,
    -1.9833242,
    0.68917286,
    -4.4505634,
    1.6545969,
    0.37214202,
    -7.477748,
    0.064747095,
    -3.523653,
    -1.4030945,
    0.86356497,
    1.2528995,
    0.40594435,
    4.1130033,
    -3.2555504,
    1.4591715,
    -1.3850882,
    -2.4866161,
    0.96216166,
    -2.040977,
];

const PHIMOE_PLAIN_GOLDEN: [f32; 48] = [
    1.7346637,
    1.779487,
    7.4098587,
    -0.7568199,
    3.2711396,
    0.22519729,
    0.079294205,
    -3.0003068,
    -2.8026862,
    -0.23447502,
    -2.7861438,
    1.2287228,
    2.2784212,
    -1.8864423,
    -1.166738,
    1.2637557,
    2.326755,
    0.94796157,
    -1.5825233,
    -2.6106663,
    -2.2103841,
    0.68422055,
    -1.3489866,
    -2.9420874,
    -4.0652657,
    -0.42382622,
    -0.28999028,
    0.0064498186,
    4.2788534,
    -2.1246562,
    -3.5965977,
    2.7146492,
    -3.3606505,
    -0.3124584,
    -3.1161742,
    -1.3153855,
    0.39527917,
    0.4667015,
    1.8999867,
    -1.3434141,
    -1.8320209,
    3.033251,
    -3.0209146,
    3.436065,
    -1.033559,
    -2.75827,
    -4.7559814,
    -0.5117618,
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
fn phimoe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(PHIMOE, &PHIMOE_GOLDEN, PHIMOE_TOL);
}

#[test]
fn phimoe_without_longrope_matches_llama_cpp() {
    assert!(worst_vs(&PHIMOE_GOLDEN, &PHIMOE_PLAIN_GOLDEN) > 0.5);
    assert_all_three_paths_match_within(PLAIN, &PHIMOE_PLAIN_GOLDEN, PHIMOE_TOL);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [(PHIMOE, &PHIMOE_GOLDEN), (PLAIN, &PHIMOE_PLAIN_GOLDEN)] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: every norm site the biased RMSNorm with the
/// file's bias, the two projection biases, routed experts top-2 of 4,
/// the window dropped, the long factor pair in use with the converter's
/// attn factor.
#[test]
fn the_loaded_decoder_is_the_graph() {
    // The rows live in `frink_models::models::phimoe`; the census is over
    // the catalog, so `phimoe` stays the one biased-RMSNorm graph.
    let biased_rms: Vec<&str> = frink_models::capability::architecture_catalog()
        .iter()
        .map(|p| p.gguf_name)
        .filter(|n| frink_models::capability::uses_biased_rms_norm(n))
        .collect();
    assert_eq!(biased_rms, [PHIMOE]);
    assert_eq!(attn_out_presence(PHIMOE), Some(Presence::Required));
    assert_eq!(output_presence(PHIMOE), Some(Presence::Required));
    assert!(matches!(
        resolve_architecture(PHIMOE),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(PHIMOE);
    for layer in &d.layers {
        for (site, op) in [
            ("attn", &layer.attn.norm_weight),
            ("ffn", &layer.moe.norm_weight),
        ] {
            match op {
                NormOp::RmsBias { weight, bias } => {
                    assert_eq!((weight.len(), bias.len()), (32, 32), "{site}");
                    assert!(
                        bias.iter().any(|b| b.abs() > 0.1),
                        "{site}: the file's bias"
                    );
                }
                other => panic!("{site}: {other:?} is not the biased RMSNorm"),
            }
        }
        assert!(layer.attn.o_bias.is_some());
        assert!(layer.attn.q_bias.is_some() && layer.attn.k_bias.is_some());
        assert_eq!(layer.moe.router.rows(), 4, "four routed experts");
    }
    assert!(matches!(d.final_norm, NormOp::RmsBias { .. }));
    assert!(d.output_bias.is_some());
    assert_eq!(
        (d.config.moe.n_experts, d.config.moe.n_experts_active),
        (4, 2)
    );
    assert_eq!(
        d.config.sliding_window, None,
        "phimoe.cpp:3-10 read no window key"
    );
    assert!((d.config.rope_attn_factor - 1.0955).abs() < 1e-3);
    assert!(
        d.config.rope_freqs.is_some(),
        "the long pair, context 64 over 32"
    );

    let p = load_graph_fixture(PLAIN);
    assert_eq!(p.config.rope_attn_factor, 1.0);
    assert!(p.config.rope_freqs.is_none());
}

/// Each seam sabotaged on the loaded decoder: the norm bias zeroed, the
/// norm turned into the LayerNorm-with-bias (what the old refusal
/// called it), the output bias dropped.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(PHIMOE);
    assert_decoder_matches_on_all_three_paths(&d, &PHIMOE_GOLDEN, PHIMOE_TOL, "baseline");

    let parts = |op: &NormOp| -> (Vec<f32>, Vec<f32>) {
        let NormOp::RmsBias { weight, bias } = op else {
            unreachable!()
        };
        (weight.clone(), bias.clone())
    };
    let saved: Vec<NormOp> = d
        .layers
        .iter_mut()
        .map(|l| {
            let (weight, _) = parts(&l.attn.norm_weight);
            std::mem::replace(&mut l.attn.norm_weight, NormOp::Rms(weight))
        })
        .collect();
    let worst = worst_vs(&decode(&d), &PHIMOE_GOLDEN);
    assert!(worst > 1e-2, "the norm bias not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.norm_weight = s;
    }

    let saved: Vec<NormOp> = d
        .layers
        .iter_mut()
        .map(|l| {
            let (weight, bias) = parts(&l.moe.norm_weight);
            std::mem::replace(
                &mut l.moe.norm_weight,
                NormOp::LayerNormBias { weight, bias },
            )
        })
        .collect();
    let worst = worst_vs(&decode(&d), &PHIMOE_GOLDEN);
    assert!(worst > 1e-2, "LayerNorm for RMSNorm not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.moe.norm_weight = s;
    }

    let saved = d.output_bias.take();
    let worst = worst_vs(&decode(&d), &PHIMOE_GOLDEN);
    assert!(worst > 1.0, "output.bias not seen: {worst}");
    d.output_bias = saved;
    assert_decoder_matches_on_all_three_paths(&d, &PHIMOE_GOLDEN, PHIMOE_TOL, "restored");
}
