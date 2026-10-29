//! Falcon, checked against llama.cpp itself: both of `falcon.cpp`'s
//! shapes, decided per layer by one OPTIONAL tensor.
//!
//! - **Falcon-7B** (no `attn_norm_2`): the shared-norm parallel residual
//!   (`frink_models::parallel_residual`, `SharedNorm`; `:71-74,124-135`)
//!   over the biased LayerNorm, a fused `attn_qkv` with NO bias and one
//!   KV head (`:38`, `head_count_kv 1`), the ungated GELU FFN with no
//!   biases (`:127-131`), NEOX RoPE over the whole head.
//! - **Falcon-40B / 180B** (`attn_norm_2` on every layer, `:35-36`):
//!   `:79-85` norm the layer input with `attn_norm_2` FOR ATTENTION and
//!   `:124` keeps feeding the FFN `attn_norm(x)`. That is the two-norm
//!   arm (`TwoNorms`) with the tensor names crossed relative to
//!   `gptneox`: `norm_sites::ATTN_NORM_2_FEEDS_ATTENTION` puts
//!   `attn_norm_2` in the attention slot and `attn_norm` in the pre-FFN
//!   slot, per layer, by tensor presence.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `falcon` | the 7B shape: multi-query fused QKV, one norm read by both branches |
//! | `falcon_40b` | `attn_norm_2` present, `head_count_kv 2` of 4: attention under the second norm, the FFN under the first |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//! Both rows are GELU rows and sit at the f16-table line
//! (`GELU_TABLE_TOL_BIASED`, the `starcoder2` / `gptneox` class).
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `falcon` | 3.80e-08 | 1.28e-03 |
//! | `falcon_40b` | 1.94e-07 | 2.54e-03 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_falcon_fixture.py \
//!     crates/frink-models/tests/fixtures/falcon_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_falcon_fixture.py \
//!     crates/frink-models/tests/fixtures/falcon_40b_tiny.gguf --40b
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
use frink_models::{Decoder, FfnActivation};

const FALCON: &str = "falcon";
const FALCON_40B: &str = "falcon_40b";

/// The f16 GELU-table line, as in `tests/proj_bias_graphs.rs` and
/// `tests/parallel_residual_graphs.rs`; the sabotages below move the
/// logits by more than 1.
const GELU_TABLE_TOL_BIASED: f32 = 1e-2;

const FALCON_GOLDEN: [f32; 48] = [
    -0.023851305,
    1.8990972,
    -0.27492452,
    0.07032436,
    -1.0992731,
    1.4236183,
    0.95483935,
    5.6469665,
    1.9979279,
    -3.5172772,
    -3.3640475,
    -3.2298155,
    1.7517385,
    2.2749195,
    2.6972399,
    0.44880116,
    -0.18968296,
    -1.5660871,
    0.7771499,
    -0.7613571,
    -3.1649864,
    -4.707929,
    -3.34941,
    3.8261957,
    -4.243885,
    -3.3405528,
    -2.0032394,
    4.9316454,
    -0.7285447,
    -0.23031873,
    1.4154136,
    2.3430972,
    -0.21827066,
    3.2599332,
    -0.7211382,
    -1.1105232,
    0.29800284,
    -3.0425372,
    -1.2951528,
    -2.2307794,
    1.9442586,
    0.7917534,
    -0.4735676,
    -4.2697406,
    1.2268739,
    3.8539133,
    2.4665067,
    1.0224755,
];

const FALCON_40B_GOLDEN: [f32; 48] = [
    2.9144917,
    1.582704,
    -0.09995645,
    0.62020063,
    -2.1127467,
    -2.5087535,
    -3.9740348,
    2.0259497,
    0.67024565,
    6.0801587,
    2.5100932,
    -3.5481682,
    -0.8439546,
    -4.189772,
    1.3911984,
    0.5786557,
    -1.8855684,
    0.1855017,
    -1.7835027,
    -1.0305117,
    0.6751963,
    -3.209982,
    3.6008422,
    -2.4633152,
    -1.8548427,
    1.0278114,
    -0.5450892,
    -0.9814405,
    -2.9492235,
    1.0006919,
    0.037578344,
    0.66227806,
    5.2264647,
    2.331304,
    -3.5922213,
    -2.338593,
    0.036880076,
    -0.73757315,
    1.2658523,
    2.2593307,
    3.5195594,
    -2.099323,
    -1.7900335,
    3.323327,
    5.3911104,
    0.09161067,
    -1.3365704,
    -2.6108572,
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
fn falcon_7b_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(FALCON, &FALCON_GOLDEN, GELU_TABLE_TOL_BIASED);
}

#[test]
fn falcon_40b_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(FALCON_40B, &FALCON_40B_GOLDEN, GELU_TABLE_TOL_BIASED);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [(FALCON, &FALCON_GOLDEN), (FALCON_40B, &FALCON_40B_GOLDEN)] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built. 7B: every layer `SharedNorm`, one biased
/// LayerNorm in the attention slot and none in the pre-FFN slot, one KV
/// head, the fused QKV split with no bias, the ungated GELU, the whole
/// head rotating. 40B: every layer `TwoNorms` with BOTH slots the biased
/// LayerNorm, the attention slot read from `attn_norm_2` and the pre-FFN
/// slot from `attn_norm`, two KV heads.
#[test]
fn the_loaded_layers_are_the_two_shapes() {
    assert!(matches!(
        resolve_architecture(FALCON),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    assert!(frink_models::capability::uses_biased_layer_norm(FALCON));
    // Over the catalog: the row lives in `frink_models::models::falcon`.
    let crossed: Vec<&str> = frink_models::capability::architecture_catalog()
        .iter()
        .map(|p| p.gguf_name)
        .filter(|n| frink_models::norm_sites::attn_norm_2_feeds_attention(n))
        .collect();
    assert_eq!(crossed, [FALCON]);

    let d = load_graph_fixture(FALCON);
    assert!(d.config.parallel_residual);
    assert_eq!(d.config.rope_dim, None, "falcon.cpp:51: the whole head");
    assert_eq!((d.config.n_heads, d.config.n_kv_heads), (4, 1));
    assert_eq!(d.config.ffn_activation, FfnActivation::GeluUngated);
    assert_eq!(d.config.rms_norm_eps, 1e-5, "attention.layer_norm_epsilon");
    for layer in &d.layers {
        assert_eq!(layer.moe.parallel, Some(ParallelNorm::SharedNorm));
        assert!(matches!(
            layer.attn.norm_weight,
            NormOp::LayerNormBias { .. }
        ));
        assert!(matches!(layer.moe.norm_weight, NormOp::None));
        assert!(layer.attn.q_bias.is_none() && layer.attn.o_bias.is_none());
        assert!(layer.moe.dense_bias.is_none());
    }
    assert!(matches!(d.final_norm, NormOp::LayerNormBias { .. }));

    let d = load_graph_fixture(FALCON_40B);
    assert!(d.config.parallel_residual);
    assert_eq!((d.config.n_heads, d.config.n_kv_heads), (4, 2));
    for layer in &d.layers {
        assert_eq!(layer.moe.parallel, Some(ParallelNorm::TwoNorms));
        assert!(matches!(
            layer.attn.norm_weight,
            NormOp::LayerNormBias { .. }
        ));
        assert!(matches!(
            layer.moe.norm_weight,
            NormOp::LayerNormBias { .. }
        ));
    }
}

/// The 40B's crossed slots are visible: swapping the two norms back
/// (attention under `attn_norm`, the FFN under `attn_norm_2`, which is
/// what a loader that read the names by their `gptneox` meaning would
/// build) diverges by more than 1; so does running either file's
/// layers as the sequential graph.
#[test]
fn the_crossed_slots_and_the_residual_are_visible_in_the_logits() {
    let mut d = load_graph_fixture(FALCON_40B);
    assert_decoder_matches_on_all_three_paths(
        &d,
        &FALCON_40B_GOLDEN,
        GELU_TABLE_TOL_BIASED,
        "baseline",
    );
    for l in d.layers.iter_mut() {
        std::mem::swap(&mut l.attn.norm_weight, &mut l.moe.norm_weight);
    }
    let worst = worst_vs(&decode(&d), &FALCON_40B_GOLDEN);
    assert!(worst > 1.0, "the crossed slots not seen: {worst}");
    for l in d.layers.iter_mut() {
        std::mem::swap(&mut l.attn.norm_weight, &mut l.moe.norm_weight);
    }
    for l in d.layers.iter_mut() {
        l.moe.parallel = None;
    }
    let worst = worst_vs(&decode(&d), &FALCON_40B_GOLDEN);
    assert!(worst > 1e-1, "the 40B parallel residual not seen: {worst}");
    for l in d.layers.iter_mut() {
        l.moe.parallel = Some(ParallelNorm::TwoNorms);
    }
    assert_decoder_matches_on_all_three_paths(
        &d,
        &FALCON_40B_GOLDEN,
        GELU_TABLE_TOL_BIASED,
        "restored",
    );

    let mut d = load_graph_fixture(FALCON);
    for l in d.layers.iter_mut() {
        l.moe.parallel = None;
    }
    let worst = worst_vs(&decode(&d), &FALCON_GOLDEN);
    assert!(worst > 1e-1, "the 7B parallel residual not seen: {worst}");
}
