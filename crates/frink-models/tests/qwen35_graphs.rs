//! Qwen3.5 (`qwen35`), checked against llama.cpp itself: the gated delta
//! net on the generic path (`crate::gdn`, `frink_core::gdn`,
//! `layer_shapes::AttnShape::Gdn`).
//!
//! `qwen35.cpp:126-152` runs every layer as `attn_norm` -> block ->
//! residual -> `post_attention_norm` -> SwiGLU -> residual, the block
//! being the delta net on the layers `attention.recurrent_layers` or
//! `(i + 1) % full_attention_interval != 0` name (`:17-24`) and gated
//! full attention on the rest (`:186-234`: the gate interleaved with
//! the query in `wq`, per-head QK norm, partial IMROPE over the
//! `rope.dimension_sections` -- NEOX band for band on text positions --
//! `sigmoid(gate) * attn` before `wo`). The delta net
//! (`:236-317`, `delta-net-base.cpp:289-365`) reads V head `h`'s keys
//! from K head `h % n_k_heads` (`llama-model.cpp:524-526`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `qwen35` | see `report_kl_against_llama_cpp` | |
//! | `qwen35_array` | (`attention.recurrent_layers`; libllama byte-identical to `qwen35`) | |
//! | `qwen35_output` | (a separate `output.weight`) | |
//! | `qwen35moe` | (`qwen35moe`: `qwen2moe`'s FFN, softmax, a sigmoid-gated shared expert) | |
//! | `qwen3next` | (`qwen3next`: grouped V heads, fused `ssm_ba`, plain NEOX RoPE) | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_qwen35_fixture.py \
//!     crates/frink-models/tests/fixtures/qwen35_tiny.gguf [--array | --output]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/qwen35_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_all_three_paths_match_within,
    assert_decoder_matches_on_all_three_paths, graph_caches, kl_vs_golden, load_graph_fixture,
    worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath, QkNormStyle};
use frink_models::config::RopeLayout;
use frink_models::layer_shapes::AttnShape;
use frink_models::norm::NormOp;
use frink_models::Decoder;

const Q35: &str = "qwen35";
const Q35_ARRAY: &str = "qwen35_array";
const Q35_OUTPUT: &str = "qwen35_output";
const Q35MOE: &str = "qwen35moe";
const Q3NEXT: &str = "qwen3next";

/// The MoE row sits at 2e-5 max delta against libllama (KL 2.9e-11),
/// with the differences of either sign: the four routed SwiGLU experts
/// and the shared expert accumulate in a different order from ggml's
/// `mul_mat_id`. The `phimoe` / `orion` class; the dense rows stay at
/// `GRAPH_TOL`.
const Q35MOE_TOL: f32 = 5e-5;

const Q35_GOLDEN: [f32; 48] = [
    1.1752617,
    -0.10627127,
    0.33335793,
    0.6561886,
    -1.4498894,
    -0.19351739,
    -1.8791107,
    0.798954,
    1.6245198,
    -0.49334693,
    -1.5738757,
    0.45307195,
    -1.0303729,
    -1.7160652,
    -0.57821476,
    -0.52908224,
    1.0848951,
    0.109870985,
    -0.07334429,
    1.8102084,
    1.2377963,
    1.6395442,
    0.6890778,
    0.3326063,
    -0.10412532,
    0.7999805,
    -1.788507,
    -1.6497498,
    -1.8424459,
    -0.061935186,
    0.3562174,
    -0.8260913,
    -0.4012126,
    1.1811497,
    0.14384389,
    0.56542516,
    1.0739757,
    -0.89419425,
    -1.2602042,
    1.235939,
    1.5505302,
    -1.009949,
    2.3749595,
    -0.5552903,
    2.3903658,
    0.7304524,
    1.7889483,
    -0.71491814,
];

const Q35_OUTPUT_GOLDEN: [f32; 48] = [
    -2.2100215,
    1.0246589,
    0.6213379,
    -3.0916371,
    -1.3098698,
    -0.23730496,
    0.7887312,
    0.8281351,
    -0.8026626,
    -0.8554641,
    -1.1705763,
    0.6187867,
    1.5126362,
    2.7802546,
    -2.774306,
    -0.040545344,
    0.8870721,
    -0.119152784,
    -2.8073406,
    1.018529,
    0.3077031,
    -0.43384475,
    0.85149944,
    -0.86601126,
    -2.2314308,
    0.10326272,
    -2.277334,
    -0.29860097,
    0.19373459,
    -3.8206537,
    1.8511173,
    -0.53988445,
    0.53501016,
    -3.0734982,
    0.6476718,
    0.3977803,
    1.7783042,
    0.06702441,
    0.38871485,
    -0.03947854,
    0.08354396,
    0.19557267,
    -0.5621463,
    -0.47500193,
    -0.5736393,
    -0.73621905,
    0.40045023,
    0.02514121,
];

const Q35MOE_GOLDEN: [f32; 48] = [
    0.22362602,
    -0.6776387,
    1.437111,
    1.2822045,
    -1.5060251,
    -1.1283212,
    0.38018948,
    -0.5652575,
    1.2369487,
    -0.1832569,
    -1.6393082,
    3.4682868,
    -1.5699699,
    1.0261121,
    2.450604,
    2.7590237,
    1.3118762,
    1.7313893,
    1.0267415,
    1.1709214,
    0.34072128,
    0.7808993,
    -0.6962054,
    1.2979488,
    1.1081667,
    -1.7758297,
    0.5941986,
    0.89649326,
    -0.3366027,
    -0.6407424,
    1.7617235,
    -2.418105,
    0.10036731,
    -1.8583269,
    -1.1456118,
    0.50613666,
    0.05913782,
    1.0523102,
    -2.3145566,
    0.1743756,
    -0.29794914,
    2.1630335,
    -0.31022805,
    -0.7142283,
    -0.21806842,
    0.23067427,
    0.037080467,
    0.87471694,
];

const Q3NEXT_GOLDEN: [f32; 48] = [
    -1.822452,
    2.3457441,
    1.3742802,
    -0.83531225,
    0.7206263,
    -0.26172385,
    -0.89559764,
    -1.3595402,
    -0.39871687,
    -0.41712087,
    -2.866317,
    -2.0913453,
    -1.2430425,
    -1.6470587,
    0.74825215,
    -0.67276496,
    -1.1653938,
    0.19282657,
    0.80315477,
    -1.8707292,
    -0.8722274,
    -0.038630247,
    1.489897,
    1.2567722,
    -2.8511395,
    0.2656275,
    1.2639767,
    -0.15691194,
    -3.245615,
    -1.1053054,
    0.6681192,
    -1.1433287,
    -0.572135,
    2.0743706,
    0.18879177,
    -0.43414372,
    -1.9876318,
    0.189032,
    0.65705353,
    -1.8302181,
    1.729482,
    0.5731125,
    0.49497312,
    1.6742837,
    0.50558585,
    1.0427566,
    -0.30545086,
    -2.0598104,
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
fn qwen35_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(Q35, &Q35_GOLDEN);
}

/// The same layout declared with `attention.recurrent_layers`, which
/// `qwen35.cpp:17` takes over the interval.
#[test]
fn the_recurrent_layers_array_matches_the_same_golden() {
    assert_all_three_paths_match(Q35_ARRAY, &Q35_GOLDEN);
}

#[test]
fn a_separate_output_weight_matches_llama_cpp() {
    assert_all_three_paths_match(Q35_OUTPUT, &Q35_OUTPUT_GOLDEN);
}

/// `qwen35moe`: the same layers with `qwen2moe`'s FFN
/// (`qwen35moe.cpp:496-538`): softmax over four experts, two used,
/// renormalised, plus a shared expert scaled by the sigmoid of its own
/// one-logit gate.
#[test]
fn qwen35moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(Q35MOE, &Q35MOE_GOLDEN, Q35MOE_TOL);
    let d = load_graph_fixture(Q35MOE);
    assert!(matches!(
        resolve_architecture("qwen35moe"),
        Some(ArchPath::GenericGqa { .. })
    ));
    assert_eq!(d.config.moe.n_experts, 4);
    assert_eq!(d.config.moe.n_shared_experts, 1);
    assert!(d.config.moe.norm_topk_prob);
    assert!(
        d.layers[0].moe.shared_expert_gate.is_some(),
        "ffn_gate_inp_shexp"
    );
    assert_eq!(d.config.layer_shape(0).attention, AttnShape::Gdn);
}

/// `qwen3next`: `qwen35moe`'s layers with the V heads GROUPED over the
/// K heads (`qwen3next.cpp:521-539`), beta and alpha from one `ssm_ba`
/// projection (`:422-436`) and plain NEOX RoPE (`:282-291`). Tiling the
/// heads instead is a different graph: this golden is the grouped one.
#[test]
fn qwen3next_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(Q3NEXT, &Q3NEXT_GOLDEN, Q35MOE_TOL);
    let d = load_graph_fixture(Q3NEXT);
    assert!(matches!(
        resolve_architecture("qwen3next"),
        Some(ArchPath::GenericGqa { .. })
    ));
    let g = d.layers[0].attn.ssm.as_ref().unwrap().gdn().unwrap();
    assert_eq!(g.h.map, frink_core::gdn::HeadMap::Grouped);
    assert!(matches!(
        g.beta_alpha,
        frink_models::gdn::BetaAlpha::Fused { .. }
    ));
    assert_eq!(d.config.moe.n_experts, 4);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (Q35, &Q35_GOLDEN),
        (Q35_ARRAY, &Q35_GOLDEN),
        (Q35_OUTPUT, &Q35_OUTPUT_GOLDEN),
        (Q35MOE, &Q35MOE_GOLDEN),
        (Q3NEXT, &Q3NEXT_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: three delta-net layers and one gated
/// attention layer, its `wq` twice the query width, the pre-FFN norm
/// from `post_attention_norm`, per-head QK norm, partial rotation.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("qwen35"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(Q35);
    assert_eq!(d.config.qk_norm_style, QkNormStyle::PerHead);
    assert_eq!(d.config.rope_dim, Some(4));
    assert!(d.config.has_recurrent_layers());
    for il in 0..3 {
        assert_eq!(
            d.config.layer_shape(il).attention,
            AttnShape::Gdn,
            "blk.{il}"
        );
        let g = d.layers[il]
            .attn
            .ssm
            .as_ref()
            .unwrap()
            .gdn()
            .expect("delta net");
        assert_eq!(
            (g.h.d_conv, g.h.head_dim, g.h.n_k_heads, g.h.n_v_heads),
            (4, 8, 2, 4)
        );
        assert!(
            d.layers[il].moe.norm_weight != NormOp::None,
            "blk.{il} post_attention_norm"
        );
    }
    assert!(matches!(
        d.config.layer_shape(3).attention,
        AttnShape::Gqa {
            n_heads: 4,
            n_kv_heads: 2
        }
    ));
    let attn = &d.layers[3].attn;
    assert!(attn.q_gate_interleaved);
    assert_eq!(attn.q_proj.rows(), 2 * 4 * 8);
    assert_eq!(attn.q_norm.as_ref().map(Vec::len), Some(8));
    assert!(attn.ssm.is_none());
}

/// The paged backing carries the delta state, rows and state agreeing
/// with the contiguous one.
#[test]
fn paged_decode_matches_contiguous() {
    let d = load_graph_fixture(Q35);
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
            .expect("8 blocks of 4 hold 6 positions");
    }
    assert_eq!(got, want);
    assert!(worst_vs(&got, &Q35_GOLDEN) < GRAPH_TOL);
    assert_eq!(paged[0].recurrent, contiguous[0].recurrent);
    assert_eq!(
        contiguous[3].rows(),
        GRAPH_PROMPT.len(),
        "the attention layer's rows"
    );
}

/// The delta state is visible: a decay of -30 forgets everything.
/// (The head map and the attention gate are pinned by the golden
/// itself: tiling the V heads the other way, or skipping the gate's
/// sigmoid, each moved the logits by more than 1e-2 when sabotaged.)
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(Q35);
    assert_decoder_matches_on_all_three_paths(&d, &Q35_GOLDEN, GRAPH_TOL, "baseline");
    let saved: Vec<f32> = d.layers[0]
        .attn
        .ssm
        .as_ref()
        .unwrap()
        .gdn()
        .unwrap()
        .a
        .clone();
    for a in d.layers[0]
        .attn
        .ssm
        .as_mut()
        .unwrap()
        .gdn_mut()
        .unwrap()
        .a
        .iter_mut()
    {
        *a = -30.0;
    }
    let worst = worst_vs(&decode(&d), &Q35_GOLDEN);
    assert!(worst > 1e-2, "the delta state not seen: {worst}");
    d.layers[0].attn.ssm.as_mut().unwrap().gdn_mut().unwrap().a = saved;
    assert_decoder_matches_on_all_three_paths(&d, &Q35_GOLDEN, GRAPH_TOL, "restored");
}
