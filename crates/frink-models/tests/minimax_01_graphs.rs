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
/// move the logits by more than 3.
///
/// 5e-4, up from 2e-4, since the goldens became CI's answer (ADR 0011: a
/// portable AVX2 libllama on x86_64, not an arm64 native one). The two
/// libllama builds alone differ by 1.6e-4 (`minimax_01`) and 1.9e-4
/// (`minimax_01_output`), and frink on x86_64 CI then sits 2.35e-4 and
/// 2.59e-4 from the new goldens on the prefill path (arm64: inside
/// 2e-4). The sabotages are still almost four orders over the line.
const TOL: f32 = 5e-4;

const MINIMAX_01_GOLDEN: [f32; 48] = [
    0.66824585,
    1.6755114,
    0.54837406,
    0.52108043,
    1.1550103,
    0.16876064,
    -2.6534996,
    -2.2283218,
    -2.1807551,
    0.05115971,
    0.4645205,
    -0.4997689,
    -0.7327945,
    0.7498171,
    1.0828277,
    -2.810112,
    -0.45857427,
    -0.6860819,
    -0.95748293,
    -2.1284966,
    1.9552443,
    -2.6329615,
    -4.0131454,
    -0.24420214,
    0.4345796,
    2.7080176,
    -2.8099303,
    -0.91550803,
    -0.5632334,
    -0.05972123,
    1.3011812,
    1.6051451,
    -0.05827506,
    1.057216,
    2.6908393,
    2.6604834,
    1.4149175,
    -0.39187092,
    -3.0710444,
    0.01939106,
    0.5190912,
    2.8503928,
    -2.030086,
    -0.61170405,
    0.057593122,
    -0.471462,
    1.1523254,
    -1.2112093,
];

const MINIMAX_01_FUSED_QKV_GOLDEN: [f32; 48] = [
    -1.5538161,
    0.18436855,
    -0.1249229,
    1.565089,
    1.2923499,
    -1.4466064,
    -1.0032593,
    -2.2260268,
    -0.77517575,
    -1.13099,
    0.7147856,
    -0.7969661,
    0.62825215,
    1.7542577,
    -0.1985631,
    -1.841079,
    0.59583557,
    -2.0973747,
    -0.9027889,
    -2.1054688,
    0.11448419,
    0.6151551,
    -2.0670385,
    1.176101,
    0.74924153,
    -0.846655,
    3.2510486,
    0.02795124,
    0.2884857,
    -1.3473189,
    -1.1803176,
    -1.5219637,
    -0.36797184,
    -3.7478814,
    0.8820178,
    -0.49543583,
    -0.9144577,
    -1.683687,
    0.8805351,
    2.8950386,
    2.438622,
    -0.23221415,
    -0.18290132,
    -1.643198,
    0.38166493,
    0.94704986,
    0.28707248,
    1.243948,
];

const MINIMAX_01_OUTPUT_GOLDEN: [f32; 48] = [
    1.1459701,
    0.61399806,
    -0.5987601,
    0.29744333,
    1.325903,
    3.2395597,
    1.3979886,
    -2.7863042,
    0.759284,
    -0.6530173,
    -1.1869439,
    2.5712419,
    0.45429423,
    0.27523708,
    0.21005821,
    0.40025944,
    2.0218887,
    -1.2547214,
    0.7503343,
    2.855513,
    -1.5133799,
    -0.39181554,
    0.057765245,
    0.3275042,
    -0.8599144,
    -0.00026649237,
    -1.5367789,
    0.73983514,
    -2.0450869,
    -0.45371526,
    -0.2646246,
    0.6175356,
    0.2813534,
    2.6726363,
    -0.71346575,
    -1.5592294,
    0.8838768,
    1.061053,
    1.1879032,
    1.0659946,
    -0.16956994,
    0.029929966,
    0.43427098,
    -0.39995193,
    -1.6210182,
    2.624225,
    -1.2472099,
    -1.975461,
];

const MINIMAX_01_UNIT_SCALE_GOLDEN: [f32; 48] = [
    0.9033351,
    1.7487543,
    1.6392146,
    -1.3655577,
    -0.37488908,
    -1.9757447,
    -0.5435107,
    -1.6539354,
    -0.49686086,
    1.9024792,
    0.49952018,
    0.87883323,
    -1.5061895,
    0.6091199,
    -4.4883957,
    -2.686212,
    -0.61249197,
    -0.90622264,
    -0.5051459,
    3.6101217,
    -1.6718059,
    0.5027277,
    1.2716429,
    -0.045524,
    -3.2773595,
    1.1984423,
    -4.126742,
    -2.7209222,
    0.5687667,
    -1.6882763,
    -0.056616127,
    -3.0365305,
    -0.8926109,
    -0.33311337,
    1.3379872,
    1.9813068,
    -2.2836359,
    -0.54966545,
    1.7847652,
    -1.0670828,
    0.28691858,
    -0.99251103,
    3.3261096,
    -0.81353784,
    0.37255782,
    1.1934747,
    -1.4080226,
    -3.1558623,
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
