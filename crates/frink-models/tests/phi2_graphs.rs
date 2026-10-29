//! Phi-2, checked against llama.cpp itself: the shared-norm parallel
//! residual (`frink_models::parallel_residual`) over the biased
//! LayerNorm, Q/K/V biases, the REQUIRED `attn_output.bias` / `ffn_up.
//! bias` / `ffn_down.bias` with the ungated GELU (`frink_models::
//! proj_bias`), a partial NEOX rotary, and the one thing no earlier row
//! had: `output.bias` on the LM head (`phi2.cpp:22,136`, REQUIRED),
//! `Decoder::output_bias`, added right after the head in
//! `decoder::lm_head::Logits` and never folded into a fused Metal
//! decode stack (a bias moves the argmax where the cap and the
//! multiplier cannot).
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `phi2` | split `attn_q` / `attn_k` / `attn_v` with biases, as the current converter writes |
//! | `phi2_fused` | the same weights as one `attn_qkv.weight` / `.bias` (older exports); libllama byte-identical to the split file |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//! A GELU row, at the f16-table line (`GELU_TABLE_TOL_BIASED`).
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `phi2` | 2.95e-07 | 2.50e-03 |
//! | `phi2_fused` | 2.95e-07 | 2.50e-03 (the same golden) |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_phi2_fixture.py \
//!     crates/frink-models/tests/fixtures/phi2_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_phi2_fixture.py \
//!     crates/frink-models/tests/fixtures/phi2_fused_tiny.gguf --fused
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
use frink_models::parallel_residual::ParallelNorm;
use frink_models::proj_bias::{output_presence, Presence};
use frink_models::{Decoder, FfnActivation};

const PHI2: &str = "phi2";
const PHI2_FUSED: &str = "phi2_fused";

/// The f16 GELU-table line, as in `tests/proj_bias_graphs.rs`; the
/// sabotages below move the logits by more than 1.
const GELU_TABLE_TOL_BIASED: f32 = 1e-2;

const PHI2_GOLDEN: [f32; 48] = [
    -2.703486,
    -6.481769,
    -1.7040737,
    1.4405402,
    1.0704855,
    1.539808,
    -4.351997,
    -1.717865,
    3.3333216,
    1.2004324,
    2.103045,
    -1.4339964,
    1.5129533,
    -3.1447134,
    -1.5147669,
    3.5478418,
    1.0659884,
    2.2116117,
    -0.15988258,
    -6.2265286,
    0.7879978,
    1.7139347,
    1.8020049,
    1.3367853,
    2.9507613,
    -1.4044921,
    -2.086196,
    0.06277639,
    5.179845,
    3.6416256,
    -1.7835414,
    -1.7161425,
    -1.1067437,
    -0.42966536,
    -0.56274307,
    -0.045191407,
    1.3812802,
    -3.0431554,
    2.634523,
    3.1698017,
    1.0008919,
    1.6742024,
    -1.3365207,
    0.79682136,
    -2.5168622,
    -4.505355,
    1.7743274,
    0.75843775,
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
fn phi2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(PHI2, &PHI2_GOLDEN, GELU_TABLE_TOL_BIASED);
}

/// The fused spelling loads to the same decoder and the same logits;
/// libllama's goldens for the two files are byte-identical.
#[test]
fn the_fused_qkv_spelling_matches_the_same_golden() {
    assert_all_three_paths_match_within(PHI2_FUSED, &PHI2_GOLDEN, GELU_TABLE_TOL_BIASED);
}

#[test]
fn report_kl_against_llama_cpp() {
    let out = decode(&load_graph_fixture(PHI2));
    println!(
        "{PHI2}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
        kl_vs_golden(&out, &PHI2_GOLDEN),
        worst_vs(&out, &PHI2_GOLDEN)
    );
}

/// What the loader built: the output bias read from the file at vocab
/// width; every layer the shared-norm parallel residual with the biased
/// LayerNorm and no pre-FFN norm; the Q/K/V, `wo` and FFN biases; the
/// ungated GELU; a half-width rotary.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture(PHI2),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    assert!(frink_models::capability::uses_biased_layer_norm(PHI2));
    assert_eq!(output_presence(PHI2), Some(Presence::Required));
    for name in [PHI2, PHI2_FUSED] {
        let d = load_graph_fixture(name);
        let bias = d.output_bias.as_ref().expect("phi2.cpp:22: REQUIRED");
        assert_eq!(bias.len(), d.output_head.rows());
        assert!(bias.iter().any(|b| b.abs() > 0.5), "the file's, not zeros");
        assert!(d.config.parallel_residual);
        assert_eq!(d.config.rope_dim, Some(4), "partial_rotary_factor 0.5 of 8");
        assert_eq!(d.config.ffn_activation, FfnActivation::GeluUngated);
        for layer in &d.layers {
            assert_eq!(layer.moe.parallel, Some(ParallelNorm::SharedNorm));
            assert!(matches!(
                layer.attn.norm_weight,
                NormOp::LayerNormBias { .. }
            ));
            assert!(matches!(layer.moe.norm_weight, NormOp::None));
            assert!(layer.attn.q_bias.is_some() && layer.attn.k_bias.is_some());
            assert!(layer.attn.v_bias.is_some() && layer.attn.o_bias.is_some());
            let b = layer.moe.dense_bias.as_ref().expect("REQUIRED");
            assert!(b.up.is_some() && b.down.is_some() && b.gate.is_none());
        }
        assert!(matches!(d.final_norm, NormOp::LayerNormBias { .. }));
    }
}

/// The output bias is visible in the logits (the fold refusal for a
/// head that has one is pinned in `decoder::lm_head`'s own tests); the
/// residual sabotage diverges too.
#[test]
fn the_output_bias_and_the_residual_are_visible_in_the_logits() {
    let mut d = load_graph_fixture(PHI2);
    assert_decoder_matches_on_all_three_paths(&d, &PHI2_GOLDEN, GELU_TABLE_TOL_BIASED, "baseline");

    let saved = d.output_bias.take();
    let worst = worst_vs(&decode(&d), &PHI2_GOLDEN);
    assert!(worst > 1.0, "output.bias not seen: {worst}");
    d.output_bias = saved;

    for l in d.layers.iter_mut() {
        l.moe.parallel = None;
    }
    let worst = worst_vs(&decode(&d), &PHI2_GOLDEN);
    assert!(worst > 1e-1, "the parallel residual not seen: {worst}");
    for l in d.layers.iter_mut() {
        l.moe.parallel = Some(ParallelNorm::SharedNorm);
    }
    assert_decoder_matches_on_all_three_paths(&d, &PHI2_GOLDEN, GELU_TABLE_TOL_BIASED, "restored");
}
