//! StarCoder2, CodeShell and Jais-2, checked against llama.cpp itself:
//! the projection biases on the generic dense path.
//!
//! The three were the rows of the "LayerNorm-with-bias group" whose
//! other blocker was `attn_output.bias`, `ffn_up.bias` and
//! `ffn_down.bias`, all REQUIRED, with no slot on the generic dense
//! path. `frink_models::proj_bias` is that slot -- `AttnWeights::
//! o_bias` after `wo` (and after `o_scale`, `build_attn`'s order) and
//! `MoeWeights::dense_bias` (`frink_moe::DenseBias`) before the
//! activation and after `down` (`build_ffn`'s) -- filled for exactly
//! the architectures whose graph creates the tensors, which the module
//! measured over all 155 graphs; gpt-oss's `o_bias` moved onto the
//! same slot from its side table. Two of the three also needed the
//! ungated GELU (`FfnActivation::GeluUngated`, `LLM_FFN_GELU` under
//! `LLM_FFN_SEQ`), the third the ReLU-squared FFN `arcee` had.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `starcoder2` | biased LayerNorm, Q/K/V biases, `wo_b`, ungated GELU with `up_b` / `down_b`, NEOX, `rope.dimension_count` |
//! | `codeshell` | the same with no `rope.dimension_count` and a linear scaling of exactly 1 (`conversion/codeshell.py:19-21`) |
//! | `jais2` | the same biases on the ReLU-squared FFN, MHA (`jais2.cpp:37-39` size the K/V biases `{n_embd}`, so a grouped file cannot load upstream) |
//! | `llama_biases` | the OPTIONAL case: `llama.cpp`'s own graph creates `attn_output.bias` and all three FFN biases `TENSOR_NOT_REQUIRED` and applies them when present, on a GATED SwiGLU with RMSNorm -- the one file that exercises `ffn_gate.bias`; frink used to refuse it as carrying unread tensors |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//! The two GELU rows are compared at [`GELU_TABLE_TOL_BIASED`]:
//! llama.cpp's f16 GELU table is the approximate side, measured below.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `starcoder2` | 4.68e-07 | 3.00e-03 (2.1e-13 / 1.9e-06 with the f16 table emulated) |
//! | `codeshell` | 4.73e-06 | 7.89e-03 (1.1e-12 / 2.4e-06 with the f16 table emulated) |
//! | `jais2` | 6.63e-13 | 2.15e-06 |
//! | `llama_biases` | 2.68e-13 | 3.58e-06 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_proj_bias_fixture.py starcoder2 \
//!     crates/frink-models/tests/fixtures/starcoder2_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_proj_bias_fixture.py codeshell \
//!     crates/frink-models/tests/fixtures/codeshell_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_proj_bias_fixture.py jais2 \
//!     crates/frink-models/tests/fixtures/jais2_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_proj_bias_fixture.py llama \
//!     crates/frink-models/tests/fixtures/llama_biases_tiny.gguf
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
use frink_models::{Decoder, FfnActivation};

const STARCODER2: &str = "starcoder2";
const CODESHELL: &str = "codeshell";
const JAIS2: &str = "jais2";
const LLAMA_BIASES: &str = "llama_biases";

/// The two GELU rows sit at 3.0e-3 and 7.9e-3 max |delta| (KL 4.7e-7 and
/// 4.7e-6), above the 2e-4 the Grok and DBRX rows hold to, and the
/// reason was measured the way theirs was: with frink's `gelu` made to
/// emulate ggml's f16 lookup table (`ggml_vec_gelu_f32` under
/// `GGML_GELU_FP16` rounds its input AND its output to f16), both files
/// agree to 1.9e-6 and 2.4e-6 (KL 2.1e-13 and 1.1e-12). The table's
/// error grows with |x|, and a biased pre-activation under a norm gain
/// of 1.5 is larger than Grok's; `jais2`, the ReLU-squared row on the
/// same biases, is at 2.1e-6 with no table in the way. The line here is
/// 1e-2, and every sabotage below moves the logits past it.
const GELU_TABLE_TOL_BIASED: f32 = 1e-2;

const STARCODER2_GOLDEN: [f32; 48] = [
    -1.0926175,
    1.3312448,
    0.41609192,
    -4.461484,
    3.6872644,
    1.1913576,
    -0.43057662,
    1.7436688,
    5.7481265,
    -1.7655408,
    1.5445454,
    -0.48526508,
    -2.380407,
    4.014509,
    0.104914784,
    -2.3526652,
    0.9192513,
    0.4783864,
    2.8612916,
    0.102033615,
    2.691976,
    -2.235642,
    -2.7812681,
    3.9165442,
    2.0605571,
    1.4872739,
    0.24452853,
    -2.594925,
    0.044754863,
    0.5747465,
    -4.051844,
    -3.9240882,
    -7.3347073,
    1.8196559,
    -2.4402351,
    1.8711305,
    1.9376856,
    0.28916943,
    0.39922032,
    1.5159407,
    -5.3955007,
    -0.6525061,
    3.7791579,
    2.5250416,
    -1.0006173,
    -0.33861667,
    -2.6075583,
    -0.12971675,
];

const CODESHELL_GOLDEN: [f32; 48] = [
    -0.7604214,
    1.4210099,
    3.208003,
    0.30369046,
    -0.57189655,
    -1.8431914,
    0.32023126,
    4.302043,
    -0.4233386,
    0.5303428,
    2.7965388,
    0.42303014,
    -3.140205,
    -1.1184019,
    -0.81505793,
    1.8018755,
    -2.952871,
    0.5999783,
    5.1441684,
    -0.15363646,
    1.5476677,
    4.254599,
    2.7911677,
    -1.3333856,
    2.3988729,
    -1.6992905,
    3.1499743,
    -2.8028686,
    -1.4014108,
    3.9597104,
    -0.059210487,
    5.8862185,
    -1.288259,
    -2.0099502,
    1.9441185,
    4.1866045,
    1.3203459,
    -1.0139891,
    -4.5193367,
    -4.0032606,
    -3.2097235,
    0.3254612,
    -1.1565952,
    -2.6408563,
    -2.1032991,
    1.3824763,
    1.7117057,
    2.711108,
];

const JAIS2_GOLDEN: [f32; 48] = [
    1.4032304,
    0.9597486,
    1.3264621,
    -0.47690594,
    -3.2832594,
    2.0583823,
    -2.5881028,
    2.3167944,
    -0.16592729,
    -0.039674878,
    -0.83522713,
    4.3458004,
    -2.5247142,
    2.572974,
    -2.1449852,
    1.706517,
    -1.8208816,
    2.413216,
    -1.7430935,
    0.66168875,
    0.82790565,
    -0.2811213,
    -2.7168362,
    0.03764248,
    -0.22280526,
    3.88866,
    -0.7145385,
    -2.6349354,
    0.72087777,
    0.81244624,
    -0.7041393,
    0.9938104,
    2.909649,
    2.4377909,
    -0.67587656,
    3.6400495,
    1.9442037,
    -0.6949006,
    -0.3682681,
    -4.8855,
    2.2625203,
    -3.6931248,
    3.59339,
    1.9955797,
    -0.2846263,
    -3.2653992,
    2.630516,
    2.4742665,
];

const LLAMA_BIASES_GOLDEN: [f32; 48] = [
    -3.4940364,
    -3.664318,
    0.016628921,
    0.14024544,
    4.383847,
    1.5438089,
    -1.8578327,
    -1.5735847,
    -3.0979557,
    4.1399736,
    -2.5926094,
    -2.5082493,
    1.0640879,
    2.972954,
    -1.2864491,
    -5.9918375,
    -0.29181662,
    -0.2851377,
    0.842005,
    -3.9406488,
    -3.574434,
    -0.5199969,
    2.6717472,
    -1.943846,
    -1.4883559,
    8.087,
    -0.28577638,
    -1.1128571,
    1.3617675,
    3.5949144,
    -4.0096145,
    2.3283005,
    0.28883553,
    0.10383916,
    0.48079693,
    0.19965363,
    -0.6059859,
    -0.01877147,
    0.29294163,
    0.37415075,
    0.6761262,
    -1.6171567,
    0.48089367,
    -2.403988,
    -4.177381,
    -1.1211095,
    -3.5456724,
    -4.9050136,
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
fn starcoder2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(STARCODER2, &STARCODER2_GOLDEN, GELU_TABLE_TOL_BIASED);
}

#[test]
fn codeshell_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(CODESHELL, &CODESHELL_GOLDEN, GELU_TABLE_TOL_BIASED);
}

#[test]
fn jais2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(JAIS2, &JAIS2_GOLDEN);
}

/// A `llama` file WITH the optional biases: applied as `build_ffn` /
/// `build_attn` apply them, gate bias included, where the loader used
/// to refuse the file for unread tensors.
#[test]
fn a_llama_file_with_the_optional_biases_matches_llama_cpp() {
    assert_all_three_paths_match(LLAMA_BIASES, &LLAMA_BIASES_GOLDEN);
    let d = load_graph_fixture(LLAMA_BIASES);
    assert!(!d.config.ffn_is_ungated());
    for layer in &d.layers {
        let bias = layer.moe.dense_bias.as_ref().expect("the biases are read");
        assert!(bias.gate.is_some() && bias.up.is_some() && bias.down.is_some());
        assert!(layer.attn.o_bias.is_some());
        assert!(matches!(layer.attn.norm_weight, NormOp::Rms(_)));
    }
    // The gate bias is the one the ungated rows cannot see.
    let mut d = d;
    for l in d.layers.iter_mut() {
        l.moe.dense_bias.as_mut().unwrap().gate = None;
    }
    let worst = worst_vs(&decode(&d), &LLAMA_BIASES_GOLDEN);
    assert!(worst > 1e-2, "ffn_gate.bias not seen: {worst}");
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (STARCODER2, &STARCODER2_GOLDEN),
        (CODESHELL, &CODESHELL_GOLDEN),
        (JAIS2, &JAIS2_GOLDEN),
        (LLAMA_BIASES, &LLAMA_BIASES_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: both bias slots filled from the file on
/// every layer, the biased norm, the activation each graph names,
/// NEOX, and codeshell's whole-head rotary under a factor-1 linear
/// scaling.
#[test]
fn the_loaded_layers_are_the_three_graphs() {
    for (name, act, kv_heads) in [
        (STARCODER2, FfnActivation::GeluUngated, 2),
        (CODESHELL, FfnActivation::GeluUngated, 2),
        (JAIS2, FfnActivation::ReluSqr, 4),
    ] {
        assert!(matches!(
            resolve_architecture(name),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        ));
        let d = load_graph_fixture(name);
        assert_eq!(d.config.ffn_activation, act, "{name}");
        assert_eq!(d.config.n_kv_heads, kv_heads, "{name}");
        assert!(d.config.ffn_is_ungated(), "{name}");
        for layer in &d.layers {
            assert_eq!(layer.attn.o_bias.as_ref().map(Vec::len), Some(32), "{name}");
            let bias = layer
                .moe
                .dense_bias
                .as_ref()
                .unwrap_or_else(|| panic!("{name}: no FFN bias"));
            assert_eq!(bias.up.as_ref().map(Vec::len), Some(48), "{name}");
            assert_eq!(bias.down.as_ref().map(Vec::len), Some(32), "{name}");
            assert!(
                bias.gate.is_none(),
                "{name}: an ungated FFN has no gate bias"
            );
            assert!(layer.attn.q_bias.is_some() && layer.attn.k_bias.is_some());
            assert!(matches!(
                layer.attn.norm_weight,
                NormOp::LayerNormBias { .. }
            ));
        }
    }
    let codeshell = load_graph_fixture(CODESHELL);
    assert_eq!(
        codeshell.config.rope_dim, None,
        "no rope.dimension_count: n_rot = head_dim"
    );
}

/// Each bias, sabotaged on the loaded decoder, moves the logits past
/// the tolerance: `wo_b`, `up_b` (before the activation), `down_b`.
#[test]
fn each_projection_bias_is_visible_in_the_logits() {
    for (name, golden, tol) in [
        (STARCODER2, &STARCODER2_GOLDEN, GELU_TABLE_TOL_BIASED),
        (JAIS2, &JAIS2_GOLDEN, GRAPH_TOL),
    ] {
        let mut d = load_graph_fixture(name);
        assert_decoder_matches_on_all_three_paths(&d, golden, tol, name);

        let saved: Vec<_> = d.layers.iter_mut().map(|l| l.attn.o_bias.take()).collect();
        let worst = worst_vs(&decode(&d), golden);
        assert!(worst > 1e-2, "{name}: attn_output.bias not seen: {worst}");
        for (l, s) in d.layers.iter_mut().zip(saved) {
            l.attn.o_bias = s;
        }

        let saved: Vec<_> = d
            .layers
            .iter_mut()
            .map(|l| l.moe.dense_bias.as_mut().unwrap().up.take())
            .collect();
        let worst = worst_vs(&decode(&d), golden);
        assert!(worst > 1e-2, "{name}: ffn_up.bias not seen: {worst}");
        for (l, s) in d.layers.iter_mut().zip(saved) {
            l.moe.dense_bias.as_mut().unwrap().up = s;
        }

        let saved: Vec<_> = d
            .layers
            .iter_mut()
            .map(|l| l.moe.dense_bias.as_mut().unwrap().down.take())
            .collect();
        let worst = worst_vs(&decode(&d), golden);
        assert!(worst > 1e-2, "{name}: ffn_down.bias not seen: {worst}");
        for (l, s) in d.layers.iter_mut().zip(saved) {
            l.moe.dense_bias.as_mut().unwrap().down = s;
        }

        assert_decoder_matches_on_all_three_paths(&d, golden, tol, "restored");
    }
}

/// **`llama-embed` is `llama`, and the fixture is the evidence.**
///
/// llama.cpp's `llama_model_llama_embed` INHERITS `llama_model_llama`
/// (`models.h:175-182`): the same `load_arch_hparams`, the same
/// `load_arch_tensors`, and a `build_arch_graph` that is `llama`'s
/// graph with the `embed` template argument set. That flag skips the
/// output head and changes nothing in the decoder body, so llama.cpp
/// cannot compute a different graph for the two names.
///
/// frink had it DEFERRED as an "embedding variant", which was read off
/// the name rather than the graph -- the mistake `pangu-embedded`
/// already cost this project once.
///
/// The fixture is `llama_biases_tiny.gguf`'s weights written under the
/// other architecture string (`--spelled`), so the two differ in the
/// name and in nothing else. It is held to `llama`'s own libllama
/// golden rather than to frink's output, which is what makes this
/// evidence rather than a tautology.
#[test]
fn llama_embed_is_llama_and_meets_llamas_golden() {
    // It is on the decoder path, not the encoder loader: it has a KV
    // cache and it generates.
    let path = resolve_architecture("llama-embed").expect("a catalog row");
    assert!(
        matches!(
            path,
            ArchPath::GenericGqa {
                rope: RopeLayout::Norm,
                ..
            }
        ),
        "llama-embed must resolve on the shared decoder with llama's rotation, not as a \
         deferred scope: {path:?}"
    );

    // The same golden the `llama` spelling is held to.
    assert_all_three_paths_match("llama_embed", &LLAMA_BIASES_GOLDEN);

    // And the two really are the same weights: identical logits, not
    // merely both close to the golden.
    let same = decode(&load_graph_fixture("llama_embed"));
    let base = decode(&load_graph_fixture(LLAMA_BIASES));
    assert_eq!(
        same.len(),
        base.len(),
        "the two spellings must produce the same vocabulary"
    );
    let worst = same
        .iter()
        .zip(&base)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        worst, 0.0,
        "the alias must compute bit-identical logits to the row it aliases"
    );
}
