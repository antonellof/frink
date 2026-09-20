//! MiniMax-Text-01 (`minimax-01`), checked against llama.cpp itself:
//! lightning attention as a recurrent block on the generic path
//! (`crate::lightning`, `frink_core::lightning`,
//! `layer_shapes::AttnShape::Lightning`).
//!
//! `minimax-01.cpp:243-459` runs every layer as `attn_norm` -> block ->
//! residual -> `ffn_norm` -> MoE -> residual, with three things the
//! rest of the generic path does not have:
//!
//! * the block on the layers `attention.recurrent_layers` or
//!   `(i + 1) % full_attention_interval != 0` name (`:11-17`, the same
//!   two keys Qwen3.5 reads with a default of 8 instead of 4), and
//!   ordinary GQA with partial NEOX RoPE on the rest (`:252-277`);
//! * the block itself (`:293-420`): a fused `attn_qkv` through SiLU
//!   BEFORE the split (`:303`), read HEAD-major (`:305-309`), a
//!   `head_dim x head_dim` KV per head decayed by `exp(-c s_h)` per
//!   token, then `rms_norm(o, attn_norm_2) * sigmoid(attn_gate(x))`
//!   and `attn_output`;
//! * the residual (`:249,428-431,440,455-458`): each sublayer's own
//!   PRE-NORM output, times a REQUIRED `residual_scale`, REPLACES the
//!   stream its branch joins, and the layer input is discarded
//!   (`crate::normed_residual`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | what it adds | KL | max abs delta |
//! |---|---|---|---|
//! | `minimax_01` | `full_attention_interval 2`, separate Q/K/V, tied head | 5.17e-10 | 8.2e-5 |
//! | `minimax_01_array` | the same layout as `attention.recurrent_layers` (libllama byte-identical to `minimax_01`, measured) | 5.17e-10 | 8.2e-5 |
//! | `minimax_01_fused_qkv` | one `attn_qkv` on the full-attention layers | 9.48e-11 | 4.9e-5 |
//! | `minimax_01_output` | a separate `output.weight` | 1.30e-9 | 1.1e-4 |
//! | `minimax_01_unit_scale` | `residual_scale = 1.0`: still this topology | 1.33e-11 | 1.5e-5 |
//!
//! Four sabotages, each confirmed red: dropping the SiLU (7.8 off),
//! splitting the fused projection into three blocks instead of
//! head-major (5.0), leaving the residual stream as the layer input
//! (4.8), and taking the decay scale from the far end of the layer
//! stack (4.8).
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_minimax_01_fixture.py \
//!     crates/frink-models/tests/fixtures/minimax_01_tiny.gguf
//! /tmp/ref_logits crates/frink-models/tests/fixtures/minimax_01_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match_within, graph_caches, kl_vs_golden, load_graph_fixture, worst_vs,
    GRAPH_PROMPT,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::RopeLayout;
use frink_models::layer_shapes::AttnShape;
use frink_models::Decoder;

const MM: &str = "minimax_01";
const MM_ARRAY: &str = "minimax_01_array";
const MM_FUSED: &str = "minimax_01_fused_qkv";
const MM_OUTPUT: &str = "minimax_01_output";
const MM_UNIT: &str = "minimax_01_unit_scale";

/// Four routed SwiGLU experts on EVERY layer, two used, accumulated in
/// a different order from ggml's `mul_mat_id`, on logits of magnitude
/// ~3: the `phimoe` / `orion` class. The KL is 1e-9 and below and the
/// deltas take either sign, which is what says reduction order rather
/// than a disagreement about the graph; the four sabotages above each
/// move the logits by more than 3, so the margin is four orders wide.
const TOL: f32 = 2e-4;

const MINIMAX_01_GOLDEN: [f32; 48] = [
    0.6681399,
    1.6755197,
    0.54836357,
    0.5210681,
    1.1550399,
    0.1687491,
    -2.6534922,
    -2.2283223,
    -2.1806085,
    0.051287755,
    0.4645192,
    -0.4998206,
    -0.73287284,
    0.7498232,
    1.082952,
    -2.8100972,
    -0.45855528,
    -0.6859754,
    -0.95748824,
    -2.1284328,
    1.9551492,
    -2.6328795,
    -4.013103,
    -0.24415612,
    0.4345622,
    2.7079797,
    -2.809811,
    -0.9154463,
    -0.56320006,
    -0.059639454,
    1.3010942,
    1.605186,
    -0.058279514,
    1.0571958,
    2.6907346,
    2.6605017,
    1.4147716,
    -0.3919657,
    -3.0708857,
    0.019455016,
    0.5191119,
    2.8503556,
    -2.0300927,
    -0.61165977,
    0.05758767,
    -0.47143316,
    1.1523659,
    -1.2111936,
];

const MINIMAX_01_FUSED_QKV_GOLDEN: [f32; 48] = [
    -1.5538177,
    0.18436944,
    -0.12492418,
    1.5650862,
    1.2923541,
    -1.4466057,
    -1.0032556,
    -2.2260303,
    -0.7751756,
    -1.1309888,
    0.7147875,
    -0.7969637,
    0.6282594,
    1.7542589,
    -0.19856691,
    -1.8410815,
    0.5958377,
    -2.0973747,
    -0.9027867,
    -2.105464,
    0.11448485,
    0.61515355,
    -2.067038,
    1.1761019,
    0.7492411,
    -0.84665596,
    3.2510538,
    0.027956069,
    0.2884857,
    -1.3473221,
    -1.1803219,
    -1.5219578,
    -0.36797106,
    -3.7478817,
    0.8820106,
    -0.49543488,
    -0.91445386,
    -1.6836829,
    0.88053775,
    2.8950386,
    2.4386199,
    -0.23221713,
    -0.18290046,
    -1.6431955,
    0.38166726,
    0.9470525,
    0.28706807,
    1.2439424,
];

const MINIMAX_01_OUTPUT_GOLDEN: [f32; 48] = [
    1.1459455,
    0.6140426,
    -0.5988216,
    0.29733756,
    1.3257124,
    3.2396116,
    1.3980083,
    -2.7862368,
    0.7591679,
    -0.65291053,
    -1.1870277,
    2.5711038,
    0.45417136,
    0.2753234,
    0.20997328,
    0.40022683,
    2.021968,
    -1.2547636,
    0.7504816,
    2.855443,
    -1.513439,
    -0.39180183,
    0.057823896,
    0.32759953,
    -0.8601077,
    -0.00031137466,
    -1.5367748,
    0.73966897,
    -2.0450027,
    -0.45371705,
    -0.26459557,
    0.61748916,
    0.28151095,
    2.6728096,
    -0.71357,
    -1.5592737,
    0.8839338,
    1.0610983,
    1.1878753,
    1.0660193,
    -0.1695627,
    0.029941797,
    0.43424672,
    -0.4001,
    -1.62096,
    2.6242142,
    -1.2471907,
    -1.9754444,
];

const MINIMAX_01_UNIT_SCALE_GOLDEN: [f32; 48] = [
    0.9033364,
    1.7487625,
    1.6392235,
    -1.3655696,
    -0.37487948,
    -1.9757394,
    -0.54350036,
    -1.6539364,
    -0.49685174,
    1.9024687,
    0.49952143,
    0.8788247,
    -1.5061786,
    0.6091176,
    -4.488391,
    -2.6861947,
    -0.61249065,
    -0.90620846,
    -0.5051411,
    3.610125,
    -1.6718173,
    0.5027287,
    1.271646,
    -0.045512557,
    -3.2773442,
    1.1984375,
    -4.126746,
    -2.72092,
    0.5687664,
    -1.6882799,
    -0.056620836,
    -3.0365207,
    -0.89261603,
    -0.3331084,
    1.3379717,
    1.9813147,
    -2.283633,
    -0.5496623,
    1.7847466,
    -1.0670989,
    0.28690767,
    -0.99251765,
    3.3261094,
    -0.81351733,
    0.37256,
    1.193477,
    -1.4080385,
    -3.155875,
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
fn minimax_01_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(MM, &MINIMAX_01_GOLDEN, TOL);
}

/// The same layout declared with `attention.recurrent_layers`, which
/// `minimax-01.cpp:12` takes over the interval. libllama's logits for
/// the two files are byte-identical, measured, so one golden serves
/// both and the array arm is evidenced rather than assumed.
#[test]
fn the_recurrent_layers_array_matches_the_same_golden() {
    assert_all_three_paths_match_within(MM_ARRAY, &MINIMAX_01_GOLDEN, TOL);
}

/// `llama-model.cpp:3289` prefers a fused `attn_qkv` for every
/// architecture that goes through `create_tensor_qkv`, which
/// `minimax-01.cpp:46` does for its full-attention layers.
#[test]
fn a_fused_qkv_on_the_attention_layers_matches_llama_cpp() {
    assert_all_three_paths_match_within(MM_FUSED, &MINIMAX_01_FUSED_QKV_GOLDEN, TOL);
    let d = load_graph_fixture(MM_FUSED);
    assert!(d.layers[1].attn.q_proj.rows() > 0, "blk.1 is attention");
}

#[test]
fn a_separate_output_weight_matches_llama_cpp() {
    assert_all_three_paths_match_within(MM_OUTPUT, &MINIMAX_01_OUTPUT_GOLDEN, TOL);
}

/// `residual_scale = 1.0` is still this topology: the layer input is
/// discarded whatever the multiplier is. A `scale_or_none` on that
/// value -- which every OTHER multiplier here takes -- would turn the
/// one architecture that has it back into every other one, and this
/// golden is what catches that: libllama's logits for the unit-scale
/// file are not the ordinary graph's.
#[test]
fn a_unit_residual_scale_is_still_the_pre_norm_topology() {
    assert_all_three_paths_match_within(MM_UNIT, &MINIMAX_01_UNIT_SCALE_GOLDEN, TOL);
    let d = load_graph_fixture(MM_UNIT);
    assert_eq!(d.config.normed_residual_scale, Some(1.0));
    assert_eq!(d.config.residual_scale, None);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (MM, &MINIMAX_01_GOLDEN),
        (MM_ARRAY, &MINIMAX_01_GOLDEN),
        (MM_FUSED, &MINIMAX_01_FUSED_QKV_GOLDEN),
        (MM_OUTPUT, &MINIMAX_01_OUTPUT_GOLDEN),
        (MM_UNIT, &MINIMAX_01_UNIT_SCALE_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: two lightning layers and two GQA layers, the
/// block's four tensors on the first kind and none of them on the
/// second, the decay scale falling with the layer index, and the
/// pre-norm residual topology with its REQUIRED scale.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("minimax-01"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(MM);
    assert!(d.config.has_recurrent_layers());
    assert_eq!(d.config.rope_dim, Some(4));
    assert_eq!(d.config.normed_residual_scale, Some(0.35));
    assert_eq!(d.config.residual_scale, None);
    for il in [0usize, 2] {
        assert_eq!(
            d.config.layer_shape(il).attention,
            AttnShape::Lightning,
            "blk.{il}"
        );
        let l = match d.layers[il].attn.ssm.as_ref().expect("lightning block") {
            frink_models::ssm_block::SsmBlock::Lightning(l) => l,
            other => panic!(
                "blk.{il} is not lightning: {:?}",
                std::mem::discriminant(other)
            ),
        };
        assert_eq!((l.n_head, l.head_dim), (4, 8));
        assert_eq!(l.norm.len(), 32);
        assert_eq!(l.qkv.rows(), 3 * 32);
        assert_eq!(l.gate.rows(), 32);
        assert_eq!(
            frink_models::lightning::Lightning::state_len(l.n_head, l.head_dim),
            4 * 8 * 8
        );
    }
    // `minimax-01.cpp:288`: `1 - il/(n_layer - 1) + 1e-5`, so a later
    // layer decays LESS. Four layers, so layer 2 is at one third.
    let scale = |il: usize| match d.layers[il].attn.ssm.as_ref().unwrap() {
        frink_models::ssm_block::SsmBlock::Lightning(l) => l.decay.scale,
        _ => unreachable!(),
    };
    assert!((scale(0) - 1.00001).abs() < 1e-6, "{}", scale(0));
    assert!((scale(2) - 0.33334).abs() < 1e-4, "{}", scale(2));
    for il in [1usize, 3] {
        assert!(
            matches!(
                d.config.layer_shape(il).attention,
                AttnShape::Gqa {
                    n_heads: 4,
                    n_kv_heads: 2
                }
            ),
            "blk.{il}"
        );
        assert!(d.layers[il].attn.ssm.is_none(), "blk.{il}");
    }
    assert_eq!(d.config.moe.n_experts, 4);
    assert_eq!(d.config.moe.n_experts_active, 2);
}

/// The paged backing carries the lightning state, rows and state
/// agreeing with the contiguous one.
#[test]
fn paged_decode_matches_contiguous() {
    let d = load_graph_fixture(MM);
    let store = std::sync::Arc::new(d.config.new_paged_kv(4, 8));
    let mut paged: Vec<frink_core::cache::PagedKvCache> = (0..d.config.n_layers)
        .map(|_| frink_core::cache::PagedKvCache::new())
        .collect();
    let mut contiguous = graph_caches(&d);
    let mut want = Vec::new();
    let mut got = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        want = d.forward_token(tok, pos, &mut contiguous);
        got = d
            .forward_token_paged(tok, pos, &mut paged, &store)
            .expect("the store has room");
    }
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-5, "logit {i}: paged {a} vs {b}");
    }
}

/// A file that does not declare the REQUIRED key is refused by name,
/// as llama.cpp refuses it (`minimax-01.cpp:6` reads it with no
/// default).
#[test]
fn a_missing_residual_scale_is_refused_by_name() {
    use frink_models::scalar_multipliers::{
        multiplier_support, resolve, DeclaredMultipliers, MultiplierDims,
    };
    let err = resolve(
        multiplier_support("minimax-01"),
        DeclaredMultipliers::default(),
        MultiplierDims {
            head_dim: 8,
            n_layer: 4,
            n_embd: 32,
        },
    )
    .expect_err("no residual_scale");
    let msg = err.message("minimax-01");
    assert!(msg.contains("minimax-01.residual_scale"), "{msg}");
    assert!(msg.contains("REQUIRED"), "{msg}");
}
