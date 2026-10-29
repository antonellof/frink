//! The PARALLEL residual, `x + attn(norm(x)) + ffn(norm(x))`, checked
//! against llama.cpp itself on both of its spellings
//! (`frink_models::parallel_residual`):
//!
//! - **Two norms** (`gptneox.cpp:143-166`, under `use_parallel_residual`):
//!   the FFN reads `ffn_norm(x)`, its OWN LayerNorm of the layer input.
//!   Pythia, GPT-NeoX-20B, Dolly-v2. The same file with the key `false`
//!   is the sequential graph (`:167-195`), and libllama's logits for the
//!   two differ by 3.73, so the key is live and both values are matched.
//! - **One shared norm** (`plamo.cpp:64,97-98,111-112`, always): the
//!   FFN reads `attn_norm(x)`, the vector attention read; there is no
//!   `ffn_norm` tensor. PLaMo-13B. The `stablelm` layer without an
//!   `ffn_norm` is the same arm over a biased LayerNorm, and the fixture
//!   that evidenced its refusal for one PR matches now
//!   (`tests/stablelm_graphs.rs`).
//!
//! What the seam is: the sequential bodies compute `h = x + attn` and
//! then `h + ffn(normed2)`, which IS the three-term sum, so the only
//! difference is what `normed2` is -- and on a parallel layer it is a
//! function of the LAYER INPUT, which attention has already been added
//! on top of by the time the FFN body runs. So it is captured before
//! attention, beside the router's operand, as one `BranchInputs` value
//! built by one constructor; `MoeWeights::parallel` is the per-layer
//! fact, `ModelConfig::parallel_residual` the model-level one the fused
//! Metal launches refuse on.
//!
//! `gptneox` also needed three things that were already seams: the
//! biased LayerNorm (`capability::BIASED_LAYER_NORM`), the fused
//! `attn_qkv` with its bias (`qkv_fused`), and the REQUIRED
//! `attn_output.bias` / `ffn_up.bias` / `ffn_down.bias`
//! (`frink_models::proj_bias`) with the ungated GELU FFN.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `gptneox` | `use_parallel_residual = true`: two norms, the key's live value |
//! | `gptneox_sequential` | the same weights with the key `false`: the ordinary layer; libllama differs by 3.73 |
//! | `plamo` | the shared norm on every layer, RMSNorm, GQA 4:1, no `rope.dimension_count` |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `gptneox` | 5.10e-07 | 1.90e-03 (f16 GELU table, see `GELU_TABLE_TOL_BIASED`) |
//! | `gptneox_sequential` | 2.99e-07 | 2.06e-03 (the same) |
//! | `plamo` | 1.64e-13 | 2.03e-06 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_gptneox_fixture.py \
//!     crates/frink-models/tests/fixtures/gptneox_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_gptneox_fixture.py \
//!     crates/frink-models/tests/fixtures/gptneox_sequential_tiny.gguf --sequential
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_plamo_fixture.py \
//!     crates/frink-models/tests/fixtures/plamo_tiny.gguf
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_all_three_paths_match_within,
    assert_decoder_matches_on_all_three_paths, graph_caches, kl_vs_golden, load_graph_fixture,
    worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::RopeLayout;
use frink_models::norm::NormOp;
use frink_models::parallel_residual::{parallel_residual_rows, ParallelNorm};
use frink_models::{Decoder, FfnActivation};

const GPTNEOX: &str = "gptneox";
const GPTNEOX_SEQUENTIAL: &str = "gptneox_sequential";
const PLAMO: &str = "plamo";

/// The two GELU rows sit at 1.9e-3 and 2.1e-3 max |delta| (KL 5.1e-7
/// and 3.0e-7): the same class and the same magnitude as `starcoder2` /
/// `codeshell` in `tests/proj_bias_graphs.rs`, which was traced to
/// ggml's f16 GELU lookup table by making frink's GELU emulate it (both
/// agree to 2e-6 then). The SwiGLU row on the same seam, `plamo`, is at
/// 2.0e-6 with no table in the way, and the parallel and sequential
/// gptneox files are off by the SAME amount, so the residual is not what
/// the delta measures. The line is 1e-2, as there; the residual sabotage
/// below moves the logits by 3.7.
const GELU_TABLE_TOL_BIASED: f32 = 1e-2;

const GPTNEOX_GOLDEN: [f32; 48] = [
    -1.2541133,
    -1.9010253,
    1.6264493,
    -0.5944097,
    3.8079586,
    -0.45556077,
    1.4795055,
    -0.7485163,
    -0.1749692,
    2.2264998,
    -2.1929324,
    -2.1642048,
    1.6396415,
    -0.8502722,
    1.8260491,
    -1.2009183,
    -0.24953943,
    -1.4830208,
    1.1673291,
    -1.3590267,
    0.16149724,
    1.1505806,
    0.12267977,
    1.6437703,
    -0.3338021,
    -1.1078942,
    3.16778,
    0.6824665,
    1.4459947,
    0.28464678,
    -1.708386,
    -2.3415737,
    1.2089655,
    1.5562145,
    3.4985552,
    -0.83757293,
    3.1340835,
    1.0693827,
    -0.6520877,
    4.146714,
    2.273557,
    2.755667,
    -1.6995779,
    1.9658943,
    0.1468414,
    1.0106877,
    -5.382864,
    -0.6808712,
];

const GPTNEOX_SEQUENTIAL_GOLDEN: [f32; 48] = [
    -0.4067453,
    1.0910151,
    -2.099649,
    0.67372674,
    1.7033515,
    1.5465364,
    2.5024862,
    -0.5937606,
    0.20811868,
    1.6730058,
    0.29726514,
    -0.59954625,
    -0.12069702,
    0.81368446,
    1.1319603,
    -1.454434,
    1.2883321,
    -3.1368785,
    1.4435287,
    -1.6823636,
    1.6885241,
    -0.9895759,
    1.2326796,
    0.18501186,
    0.0031306744,
    0.020078778,
    0.6717562,
    -1.301095,
    2.140298,
    -1.352277,
    -0.8707237,
    -0.73309684,
    2.4703054,
    -1.3343924,
    1.975771,
    -1.436978,
    2.049243,
    0.6629174,
    0.6889976,
    2.3441858,
    -0.5461322,
    1.1231091,
    -1.5890052,
    -1.4878935,
    2.8101852,
    0.9883346,
    -5.1894064,
    -3.4441085,
];

const PLAMO_GOLDEN: [f32; 48] = [
    -1.0749183,
    -0.33061773,
    -0.14498457,
    1.1303196,
    0.21189189,
    1.1881695,
    -0.3930282,
    -0.8383394,
    -0.15709972,
    -0.68993443,
    -1.1800187,
    1.5050642,
    -1.5283421,
    -0.42264193,
    0.55689865,
    2.8455243,
    0.7183703,
    1.8230822,
    1.1520581,
    0.6484574,
    -1.1084138,
    1.5664897,
    -0.16960575,
    -0.9853859,
    -1.1229429,
    1.4508116,
    -0.37287265,
    1.9889243,
    1.7143767,
    -0.4890281,
    -0.28409266,
    -0.2671595,
    -0.7156272,
    -0.39051664,
    0.5521982,
    1.4728514,
    0.052200437,
    0.8882669,
    -1.0759233,
    1.7858328,
    -1.7430079,
    1.6180019,
    -2.4213986,
    -0.3482054,
    -2.6941166,
    1.1135948,
    4.579192,
    0.6856148,
];

fn decode(decoder: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    out
}

/// The GELU rows at the f16-table tolerance (`GELU_TABLE_TOL_BIASED`),
/// the SwiGLU row at the suite's line.
#[test]
fn gptneox_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(GPTNEOX, &GPTNEOX_GOLDEN, GELU_TABLE_TOL_BIASED);
}

#[test]
fn gptneox_sequential_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(
        GPTNEOX_SEQUENTIAL,
        &GPTNEOX_SEQUENTIAL_GOLDEN,
        GELU_TABLE_TOL_BIASED,
    );
}

#[test]
fn plamo_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(PLAMO, &PLAMO_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (GPTNEOX, &GPTNEOX_GOLDEN),
        (GPTNEOX_SEQUENTIAL, &GPTNEOX_SEQUENTIAL_GOLDEN),
        (PLAMO, &PLAMO_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built. `gptneox`: every layer `TwoNorms` with both
/// norms the biased LayerNorm, the fused QKV bias split, the three
/// projection biases, the ungated GELU, a quarter-width rotary, the
/// model-level flag set; with the key `false`, no layer is parallel and
/// the flag is clear. `plamo`: every layer `SharedNorm` with NO pre-FFN
/// norm, SwiGLU, the whole head rotating.
#[test]
fn the_loaded_layers_are_the_graphs() {
    for name in [GPTNEOX, PLAMO] {
        assert!(matches!(
            resolve_architecture(name),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        ));
    }
    assert!(frink_models::capability::uses_biased_layer_norm(GPTNEOX));

    let d = load_graph_fixture(GPTNEOX);
    assert!(d.config.parallel_residual);
    assert_eq!(d.config.rope_dim, Some(2), "rotary_pct 0.25 of 8");
    assert_eq!(
        d.config.n_kv_heads, d.config.n_heads,
        "no head_count_kv key"
    );
    assert_eq!(d.config.ffn_activation, FfnActivation::GeluUngated);
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
        assert!(layer.attn.q_bias.is_some() && layer.attn.k_bias.is_some());
        assert!(layer.attn.v_bias.is_some());
        assert!(layer.attn.o_bias.is_some());
        let bias = layer.moe.dense_bias.as_ref().expect("REQUIRED");
        assert!(bias.up.is_some() && bias.down.is_some() && bias.gate.is_none());
    }

    let seq = load_graph_fixture(GPTNEOX_SEQUENTIAL);
    assert!(!seq.config.parallel_residual);
    for layer in &seq.layers {
        assert_eq!(layer.moe.parallel, None);
        assert!(matches!(
            layer.moe.norm_weight,
            NormOp::LayerNormBias { .. }
        ));
    }

    let p = load_graph_fixture(PLAMO);
    assert!(p.config.parallel_residual);
    assert_eq!(p.config.rope_dim, None, "plamo.cpp:41: the whole head");
    assert_eq!(p.config.ffn_activation, FfnActivation::Swiglu);
    assert_eq!((p.config.n_heads, p.config.n_kv_heads), (4, 1));
    for layer in &p.layers {
        assert_eq!(layer.moe.parallel, Some(ParallelNorm::SharedNorm));
        assert!(matches!(layer.attn.norm_weight, NormOp::Rms(_)));
        assert!(
            matches!(layer.moe.norm_weight, NormOp::None),
            "no ffn_norm tensor and no pre-FFN norm: the FFN reads attention's input"
        );
    }
}

/// The key is live for `gptneox`: the two goldens differ by 3.73, and
/// running the parallel weights as the sequential graph (or the other
/// way round) misses by that much. This is the sabotage that matters,
/// because a seam that ignored `MoeWeights::parallel` would still load
/// every tensor and answer fluently.
#[test]
fn the_residual_topology_is_visible_in_the_logits() {
    assert!(worst_vs(&GPTNEOX_GOLDEN, &GPTNEOX_SEQUENTIAL_GOLDEN) > 1.0);

    let mut d = load_graph_fixture(GPTNEOX);
    assert_decoder_matches_on_all_three_paths(
        &d,
        &GPTNEOX_GOLDEN,
        GELU_TABLE_TOL_BIASED,
        "baseline",
    );
    for l in d.layers.iter_mut() {
        l.moe.parallel = None;
    }
    let worst = worst_vs(&decode(&d), &GPTNEOX_GOLDEN);
    assert!(worst > 1.0, "the parallel residual not seen: {worst}");
    // ... and the same weights ARE the sequential golden then: the two
    // files share every tensor, so this is the sequential graph run on
    // the parallel file's weights.
    let worst = worst_vs(&decode(&d), &GPTNEOX_SEQUENTIAL_GOLDEN);
    assert!(
        worst < GELU_TABLE_TOL_BIASED,
        "sequential on the same weights should be the sequential golden: {worst}"
    );
    for l in d.layers.iter_mut() {
        l.moe.parallel = Some(ParallelNorm::TwoNorms);
    }
    assert_decoder_matches_on_all_three_paths(
        &d,
        &GPTNEOX_GOLDEN,
        GELU_TABLE_TOL_BIASED,
        "restored",
    );

    // plamo: the shared norm swapped for the sequential reading of the
    // post-attention residual (with no ffn_norm to apply, the raw one).
    let mut p = load_graph_fixture(PLAMO);
    assert_decoder_matches_on_all_three_paths(&p, &PLAMO_GOLDEN, GRAPH_TOL, "baseline");
    for l in p.layers.iter_mut() {
        l.moe.parallel = None;
    }
    let worst = worst_vs(&decode(&p), &PLAMO_GOLDEN);
    assert!(worst > 1e-2, "plamo's shared norm not seen: {worst}");
    // The plausible wrong arm: two norms, with the pre-FFN slot empty,
    // norms the layer input with NOTHING.
    for l in p.layers.iter_mut() {
        l.moe.parallel = Some(ParallelNorm::TwoNorms);
    }
    let worst = worst_vs(&decode(&p), &PLAMO_GOLDEN);
    assert!(worst > 1e-2, "the wrong arm not seen: {worst}");
    for l in p.layers.iter_mut() {
        l.moe.parallel = Some(ParallelNorm::SharedNorm);
    }
    assert_decoder_matches_on_all_three_paths(&p, &PLAMO_GOLDEN, GRAPH_TOL, "restored");
}

/// The table's generic-path rows are exactly the ones with a golden
/// here or in their own graph test (`stablelm`, `command_r`, `falcon`,
/// `phi2`, `cohere2`, `cohere2moe`).
#[test]
fn every_generic_path_row_of_the_table_has_a_golden() {
    let with_golden = [
        "gptneox",
        "plamo",
        "stablelm",
        "command-r",
        "falcon",
        "phi2",
        "cohere2",
        "cohere2moe",
    ];
    for row in parallel_residual_rows() {
        let generic = matches!(
            resolve_architecture(row.arch),
            Some(ArchPath::GenericGqa { .. })
        );
        assert_eq!(
            generic,
            with_golden.contains(&row.arch),
            "`{}`: on the generic path without a golden, or with one while refused",
            row.arch
        );
    }
}
