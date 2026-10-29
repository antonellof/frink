//! Nemotron-H (`nemotron_h`), checked against llama.cpp itself: the
//! one-block-per-layer hybrid, on the Mamba-2 seam `granitehybrid`
//! opened (`crate::mamba2`, `layer_shapes::AttnShape::Mamba2`).
//!
//! `nemotron-h.cpp:143-158` runs every layer as ONE block under ONE
//! `attn_norm` with ONE residual add: Mamba-2 where `n_head_kv(i) == 0
//! && n_ff(i) == 0` (`:9-11`), attention where `n_ff(i) == 0` (no RoPE,
//! `:181-193`; optional `attn_output.bias`, `:75`), the ungated
//! ReLU-squared FFN otherwise (`:227-231`; optional biases, `:96-97`).
//! On the generic layer those are "a block with `ffn_dim 0`"
//! (`layer_shapes::BLOCK_WITHOUT_FFN_KEEPS_ITS_OUTPUT`: the output IS
//! added, unlike deci's) and "no block, an FFN" whose pre-norm is
//! `attn_norm` (`norm_sites::ONE_NORM_PER_LAYER`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `nemotron_h` | see `report_kl_against_llama_cpp` | |
//! | `nemotron_h_biases` | (`attn_output.bias`, `ffn_up.bias`, `ffn_down.bias`) | |
//! | `nemotron_h_output` | (a separate `output.weight`) | |
//! | `nemotron_h_moe` | (`nemotron_h_moe`: the FFN layer a sigmoid MoE of ungated ReLU-squared experts with the router bias, plus an ungated shared expert) | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nemotron_h_fixture.py \
//!     crates/frink-models/tests/fixtures/nemotron_h_tiny.gguf [--biases | --output]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/nemotron_h_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::{FfnActivation, RopeLayout};
use frink_models::layer_shapes::AttnShape;
use frink_models::norm::NormOp;
use frink_models::rope_layers::RopeLayers;
use frink_models::Decoder;
use frink_moe::GatingFunction;

const NH: &str = "nemotron_h";
const NH_BIASES: &str = "nemotron_h_biases";
const NH_OUTPUT: &str = "nemotron_h_output";
const NH_MOE: &str = "nemotron_h_moe";

const NH_GOLDEN: [f32; 48] = [
    0.49835175,
    -1.4659572,
    0.5571566,
    -1.1211269,
    -1.738946,
    0.7609806,
    -2.7705464,
    -0.1412692,
    -0.87078756,
    0.41768342,
    -0.006975338,
    1.0850604,
    -0.082371354,
    -2.8229394,
    0.8018638,
    -1.6038654,
    -0.35132965,
    -1.5974526,
    0.22312164,
    1.365032,
    2.35661,
    -0.26617113,
    1.3530823,
    0.6897775,
    1.8871486,
    0.6253885,
    -0.8807533,
    0.56874305,
    -3.1587472,
    2.3875623,
    -2.2404647,
    0.37548348,
    1.9546769,
    -0.9002607,
    -1.3283824,
    0.25733936,
    2.4777699,
    0.41277292,
    -0.09976137,
    0.75689775,
    -0.6550976,
    0.9939175,
    -0.04035145,
    1.7089225,
    0.81528676,
    -1.2833877,
    1.4878092,
    0.4959755,
];

const NH_BIASES_GOLDEN: [f32; 48] = [
    1.169713,
    -2.233602,
    0.7801914,
    -0.20783573,
    -3.6183696,
    -0.20179734,
    -0.03114283,
    -2.2031968,
    -0.60560894,
    -2.6798854,
    0.28114414,
    -1.6729867,
    0.21278661,
    -0.8144615,
    1.0800893,
    -1.3217417,
    -0.03867519,
    -0.84645796,
    -3.6989594,
    -0.4224767,
    1.4953208,
    -0.9601126,
    -1.721971,
    0.18882114,
    -0.7266179,
    1.5732248,
    -1.5485091,
    0.8623589,
    -3.5940688,
    0.12307876,
    -2.2586539,
    0.5310467,
    1.6465752,
    -3.6216488,
    1.8478397,
    1.736959,
    1.0122354,
    -0.64001906,
    0.22576511,
    -0.42475343,
    0.11844772,
    0.9707775,
    0.42944288,
    0.08975434,
    -0.9211603,
    0.451342,
    1.9414262,
    1.8472195,
];

const NH_OUTPUT_GOLDEN: [f32; 48] = [
    -0.08316015,
    -1.6332952,
    -0.6035934,
    1.1308761,
    -1.1232843,
    -0.91058457,
    -0.63595533,
    -0.63871837,
    0.8588956,
    -0.3319887,
    0.11025202,
    0.071849555,
    -1.5116341,
    -1.6171095,
    -0.58905846,
    -0.20399922,
    0.6905763,
    0.91684955,
    -1.1913571,
    -0.8628144,
    -0.72883093,
    -1.5927418,
    -1.7983365,
    4.6099052,
    0.08835381,
    0.1566664,
    -0.6917397,
    0.4165231,
    1.0937591,
    -3.5335515,
    0.38027567,
    -1.8192315,
    0.7732649,
    -0.5447896,
    -1.154996,
    0.43965346,
    -0.63115203,
    -2.7141054,
    0.977726,
    -2.2609303,
    0.18054594,
    1.2854426,
    -0.7552408,
    3.0653586,
    -0.835613,
    1.2382689,
    0.10847488,
    -0.6347828,
];

const NH_MOE_GOLDEN: [f32; 48] = [
    -0.08564463,
    -1.6987882,
    1.8891695,
    -1.4673432,
    0.29491013,
    0.91232616,
    -0.8917312,
    0.025637537,
    -1.7253764,
    -0.0134396255,
    0.70770055,
    0.62906206,
    0.2945255,
    -3.4463587,
    -0.27748364,
    -0.4468332,
    -1.6725576,
    0.98237514,
    -1.7984445,
    0.9663483,
    2.6736376,
    -0.88111633,
    -0.14342141,
    -1.6324458,
    0.9185461,
    1.219005,
    -1.8341842,
    0.47902396,
    -1.6169834,
    1.8832691,
    -3.6289985,
    1.2772204,
    0.22597614,
    -1.5320957,
    -0.75469035,
    0.52414393,
    2.7344248,
    -0.4001045,
    1.48726,
    0.31474167,
    -0.31410033,
    1.9143553,
    -0.88485783,
    0.8758001,
    0.6030649,
    -1.224995,
    1.2363113,
    0.4540789,
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
fn nemotron_h_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(NH, &NH_GOLDEN);
}

/// The three optional biases (`:75,96-97`), each visible.
#[test]
fn the_optional_biases_match_llama_cpp_and_are_visible() {
    assert_all_three_paths_match(NH_BIASES, &NH_BIASES_GOLDEN);
    let mut d = load_graph_fixture(NH_BIASES);
    assert!(d.layers[1].attn.o_bias.is_some());
    assert!(d.layers[2].moe.dense_bias.is_some());
    d.layers[1].attn.o_bias = None;
    d.layers[2].moe.dense_bias = None;
    // Without them the file IS the plain fixture (same seed, same draws
    // up to the biases): the logits land on the plain golden's side of
    // the biased one, not on it.
    let worst = worst_vs(&decode(&d), &NH_BIASES_GOLDEN);
    assert!(worst > 1e-2, "biases not seen: {worst}");
}

#[test]
fn a_separate_output_weight_matches_llama_cpp() {
    assert_all_three_paths_match(NH_OUTPUT, &NH_OUTPUT_GOLDEN);
}

/// `nemotron_h_moe` (Nemotron-3 Nano 30B-A3B): the FFN layer is a
/// sigmoid MoE (`nemotron-h.cpp:206-231`, the gating function a literal,
/// `expert_weights_norm` / `_scale` from the file, the router bias
/// required) of UNGATED ReLU-squared experts, plus an ungated
/// ReLU-squared shared expert added to the routed sum. The fixture
/// declares `expert_weights_scale 2.5` and `expert_weights_norm true`,
/// both READ here (unlike `mimo2`'s), so the golden carries both.
#[test]
fn nemotron_h_moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(NH_MOE, &NH_MOE_GOLDEN);
    let d = load_graph_fixture(NH_MOE);
    assert!(matches!(
        resolve_architecture("nemotron_h_moe"),
        Some(ArchPath::GenericGqa { .. })
    ));
    assert_eq!(d.config.moe.gating, GatingFunction::Sigmoid);
    assert!(d.config.moe.norm_topk_prob);
    assert_eq!(d.config.moe.expert_weights_scale, 2.5);
    assert_eq!(d.config.moe.n_experts, 4);
    assert_eq!(d.config.moe.n_shared_experts, 1);
    assert_eq!(d.config.ffn_activation, FfnActivation::ReluSqr);
    // The block layers carry no experts; the FFN layer routes with its
    // bias and its gate is the aliased `up` (no `ffn_gate_exps` in the
    // file).
    assert!(d.layers[0].moe.exp_probs_bias.is_none());
    assert!(d.layers[2].moe.exp_probs_bias.is_some());
    assert_eq!(d.layers[2].moe.router.rows(), 4);
    assert_eq!(d.config.layer_shape(2).attention, AttnShape::Absent);
    assert_eq!(d.config.layer_shape(2).ffn_dim, 16);
}

/// The routed sum's scale and the shared expert are each visible.
#[test]
fn the_moe_seams_are_visible_in_the_logits() {
    let mut d = load_graph_fixture(NH_MOE);
    assert_decoder_matches_on_all_three_paths(&d, &NH_MOE_GOLDEN, GRAPH_TOL, "baseline");
    d.config.moe.expert_weights_scale = 1.0;
    let worst = worst_vs(&decode(&d), &NH_MOE_GOLDEN);
    assert!(worst > 1e-2, "expert_weights_scale not seen: {worst}");
    d.config.moe.expert_weights_scale = 2.5;
    assert_decoder_matches_on_all_three_paths(&d, &NH_MOE_GOLDEN, GRAPH_TOL, "restored");
    let saved = std::mem::take(&mut d.layers[2].moe.shared_experts);
    let worst = worst_vs(&decode(&d), &NH_MOE_GOLDEN);
    assert!(worst > 1e-2, "the shared expert not seen: {worst}");
    d.layers[2].moe.shared_experts = saved;
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (NH, &NH_GOLDEN),
        (NH_BIASES, &NH_BIASES_GOLDEN),
        (NH_OUTPUT, &NH_OUTPUT_GOLDEN),
        (NH_MOE, &NH_MOE_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: three layer kinds from the two arrays, one
/// norm per layer (the FFN-only layer's is `attn_norm`), no rotation,
/// the ungated ReLU-squared FFN.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("nemotron_h"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(NH);
    assert_eq!(
        d.config.rope_layers,
        RopeLayers::Never,
        "no ggml_rope_ext in the graph"
    );
    assert_eq!(d.config.ffn_activation, FfnActivation::ReluSqr);
    let kinds: Vec<(AttnShape, usize)> = (0..4)
        .map(|il| {
            let s = d.config.layer_shape(il);
            (s.attention, s.ffn_dim)
        })
        .collect();
    assert_eq!(
        kinds,
        [
            (AttnShape::Mamba2, 0),
            (
                AttnShape::Gqa {
                    n_heads: 4,
                    n_kv_heads: 2
                },
                0
            ),
            (AttnShape::Absent, 40),
            (AttnShape::Mamba2, 0),
        ]
    );
    // The block layers norm at the attention slot and have no FFN norm;
    // the FFN-only layer norms at the FFN slot, with `attn_norm`.
    for il in [0, 1, 3] {
        assert!(
            d.layers[il].attn.norm_weight != NormOp::None,
            "blk.{il} attn_norm"
        );
        assert_eq!(
            d.layers[il].moe.norm_weight,
            NormOp::None,
            "blk.{il} has no FFN"
        );
    }
    assert_eq!(
        d.layers[2].attn.norm_weight,
        NormOp::None,
        "blk.2 has no block"
    );
    assert!(
        d.layers[2].moe.norm_weight != NormOp::None,
        "blk.2 norms with attn_norm"
    );
    assert!(d.layers[0].attn.ssm.is_some() && d.layers[3].attn.ssm.is_some());
}

/// Every layer's cache counts positions, whichever block it holds, and
/// the paged path agrees with the contiguous one.
#[test]
fn every_layer_counts_positions_and_paged_matches_contiguous() {
    let d = load_graph_fixture(NH);
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
    assert!(worst_vs(&got, &NH_GOLDEN) < GRAPH_TOL);
    for il in [0, 1, 3] {
        assert_eq!(contiguous[il].positions(), GRAPH_PROMPT.len(), "blk.{il}");
        assert_eq!(paged[il].seq_len(), GRAPH_PROMPT.len(), "blk.{il}");
    }
    assert!(contiguous[0].recurrent.is_some() && contiguous[1].recurrent.is_none());
}

/// The one residual add per layer is the block's output (nemotron-h.cpp:
/// 157), not deci's discarded branch: zeroing the attention layer's
/// `wo` moves the logits.
#[test]
fn a_block_without_an_ffn_still_reaches_the_residual() {
    let mut d = load_graph_fixture(NH);
    assert_decoder_matches_on_all_three_paths(&d, &NH_GOLDEN, GRAPH_TOL, "baseline");
    let zero = frink_core::weight_matrix::WeightMatrix::F32(frink_core::Tensor::new(
        vec![0.0; d.layers[1].attn.o_proj.rows() * d.layers[1].attn.o_proj.cols()],
        vec![
            d.layers[1].attn.o_proj.rows(),
            d.layers[1].attn.o_proj.cols(),
        ],
    ));
    let saved = std::mem::replace(&mut d.layers[1].attn.o_proj, zero);
    let worst = worst_vs(&decode(&d), &NH_GOLDEN);
    assert!(
        worst > 1e-2,
        "the attention layer's output not seen: {worst}"
    );
    d.layers[1].attn.o_proj = saved;
    assert_decoder_matches_on_all_three_paths(&d, &NH_GOLDEN, GRAPH_TOL, "restored");
}
