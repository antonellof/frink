//! GPT-2 and StarCoder, checked against llama.cpp itself: the learned
//! absolute position table (`frink_models::position_embd`) added to the
//! token embedding before layer 0, and NO rotation anywhere
//! (`rope_layers::RopeLayers::Never`).
//!
//! `gpt2.cpp` and `starcoder.cpp` are one graph: the biased LayerNorm, a
//! fused `attn_qkv` with its bias, REQUIRED `attn_output.bias` / FFN
//! biases with the ungated GELU (`frink_models::proj_bias`), a
//! sequential residual, `output` tied when absent, and
//! `position_embd.weight` `{n_embd, n_ctx_train}` gathered at `inp_pos`
//! and added (`gpt2.cpp:19,74-77`; `starcoder.cpp:19,75-78`). They
//! differ in `head_count_kv 1` (StarCoder is multi-query) and in a size
//! table. `llama_model_rope_type` answers NONE for `gpt2` and NORM for
//! `starcoder`; neither graph calls `ggml_rope`, so the layout the
//! profile carries is a filler no rotation site reads.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `gpt2` | multi-head, no `head_count_kv` key, the table at `context_length` rows |
//! | `starcoder` | multi-query (`head_count_kv 1`), the same graph |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//! Both are GELU rows, at the f16-table line (`GELU_TABLE_TOL_BIASED`).
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `gpt2` | 1.85e-07 | 1.92e-03 |
//! | `starcoder` | 1.89e-07 | 2.22e-03 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_gpt2_fixture.py \
//!     crates/frink-models/tests/fixtures/gpt2_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_gpt2_fixture.py \
//!     crates/frink-models/tests/fixtures/starcoder_tiny.gguf --starcoder
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match_within, assert_decoder_matches_on_all_three_paths, graph_caches,
    kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::norm::NormOp;
use frink_models::position_embd::learned_positions;
use frink_models::proj_bias::Presence;
use frink_models::rope_layers::RopeLayers;
use frink_models::{Decoder, FfnActivation};

const GPT2: &str = "gpt2";
const STARCODER: &str = "starcoder";

/// The f16 GELU-table line, as in `tests/proj_bias_graphs.rs`; the
/// sabotages below move the logits by more than 1.
const GELU_TABLE_TOL_BIASED: f32 = 1e-2;

const GPT2_GOLDEN: [f32; 48] = [
    -0.6573621,
    2.9633462,
    3.2659283,
    -1.3273627,
    0.08076489,
    -1.7954323,
    -1.1176579,
    2.8109498,
    -3.9097128,
    0.01209116,
    1.4695457,
    1.5429434,
    -1.5182111,
    -3.9310834,
    0.13187599,
    -1.659733,
    0.5722728,
    1.4608746,
    -2.8142061,
    3.1388316,
    -3.6804242,
    -1.9039344,
    2.8494217,
    -1.6299663,
    1.980264,
    3.123711,
    0.15463078,
    -2.2403169,
    -4.069517,
    -1.2410057,
    0.08672428,
    -0.20686132,
    -1.9231515,
    -0.41513622,
    -2.5273848,
    -0.89849216,
    1.3546169,
    -1.8897996,
    -0.24963951,
    2.9334893,
    -3.6576576,
    2.508842,
    -3.8780282,
    0.43352234,
    3.319735,
    -1.3278563,
    -6.625536,
    -2.5700781,
];

const STARCODER_GOLDEN: [f32; 48] = [
    2.7122505,
    -0.87900597,
    1.9939574,
    0.52471054,
    0.32130376,
    1.0395184,
    -0.8511578,
    -2.133499,
    -1.9509056,
    2.658618,
    -0.6289139,
    0.77263606,
    0.6364243,
    -1.0310364,
    1.5179756,
    -2.0868213,
    -3.563622,
    1.3383784,
    0.80170465,
    3.4460893,
    0.90810513,
    -2.703003,
    -2.0542274,
    2.0577478,
    0.5137366,
    -1.8185408,
    -2.2759106,
    1.1423565,
    3.7036502,
    -0.22664452,
    -0.9481406,
    0.8348339,
    0.58086765,
    1.3786743,
    2.3406892,
    1.733942,
    -0.88097894,
    -1.4998661,
    -1.6387432,
    -0.62362313,
    -1.7426552,
    4.1450834,
    0.7006407,
    -2.7372146,
    -0.34740624,
    1.5327535,
    -0.2832946,
    -1.1003811,
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
fn gpt2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(GPT2, &GPT2_GOLDEN, GELU_TABLE_TOL_BIASED);
}

#[test]
fn starcoder_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(STARCODER, &STARCODER_GOLDEN, GELU_TABLE_TOL_BIASED);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [(GPT2, &GPT2_GOLDEN), (STARCODER, &STARCODER_GOLDEN)] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: the table at `context_length` rows, no layer
/// rotating, the biased LayerNorm, the fused QKV bias split, the
/// projection biases, the ungated GELU; StarCoder with one KV head.
#[test]
fn the_loaded_decoders_are_the_graph() {
    for (name, n_kv) in [(GPT2, 4), (STARCODER, 1)] {
        assert!(matches!(
            resolve_architecture(name),
            Some(ArchPath::GenericGqa { .. })
        ));
        assert!(learned_positions(name));
        assert!(matches!(
            frink_models::models::ArchFacts::of(name)
                .tensors
                .position_embd,
            Some((Presence::Required, _))
        ));
        assert!(frink_models::capability::uses_biased_layer_norm(name));
        let d = load_graph_fixture(name);
        let table = d.position_embd.as_ref().expect("REQUIRED");
        assert_eq!((table.rows(), table.cols()), (64, 32));
        assert!(d.config.learned_positions);
        assert_eq!(d.config.rope_layers, RopeLayers::Never);
        for il in 0..d.config.n_layers {
            assert!(
                d.config.layer_rope(il).is_none(),
                "{name} layer {il} must not rotate"
            );
        }
        assert_eq!(d.config.n_kv_heads, n_kv, "{name}");
        assert_eq!(d.config.ffn_activation, FfnActivation::GeluUngated);
        for layer in &d.layers {
            assert!(matches!(
                layer.attn.norm_weight,
                NormOp::LayerNormBias { .. }
            ));
            assert!(matches!(
                layer.moe.norm_weight,
                NormOp::LayerNormBias { .. }
            ));
            assert!(layer.attn.q_bias.is_some() && layer.attn.o_bias.is_some());
            let b = layer.moe.dense_bias.as_ref().expect("REQUIRED");
            assert!(b.up.is_some() && b.down.is_some());
        }
        assert!(matches!(d.final_norm, NormOp::LayerNormBias { .. }));
    }
}

/// The table and the no-rotation rule are each visible: dropping the
/// table, and rotating every layer as the generic path used to
/// (the finding `tests/rope_layout.rs`'s `LLAMA_NO_ROPE` group was
/// built on), each move the logits by more than 1.
#[test]
fn the_position_table_and_the_absence_of_rotation_are_visible_in_the_logits() {
    let mut d = load_graph_fixture(GPT2);
    assert_decoder_matches_on_all_three_paths(&d, &GPT2_GOLDEN, GELU_TABLE_TOL_BIASED, "baseline");

    let saved = d.position_embd.take();
    let worst = worst_vs(&decode(&d), &GPT2_GOLDEN);
    assert!(worst > 1.0, "the position table not seen: {worst}");
    d.position_embd = saved;

    d.config.rope_layers = RopeLayers::All;
    let worst = worst_vs(&decode(&d), &GPT2_GOLDEN);
    assert!(
        worst > 1e-1,
        "rotating a non-rotating model not seen: {worst}"
    );
    d.config.rope_layers = RopeLayers::Never;

    assert_decoder_matches_on_all_three_paths(&d, &GPT2_GOLDEN, GELU_TABLE_TOL_BIASED, "restored");
}
