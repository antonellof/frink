//! ALiBi, checked against llama.cpp itself: the per-head linear position
//! bias `slope_h * (p_key - p_query)` added to every scaled attention
//! score in place of any rotation (`frink_core::alibi` is the slope
//! formula, `frink_models::alibi` the graphs and where each gets its
//! `f_max_alibi_bias`, `rope_layers::RopeLayers::Never` the other half).
//!
//! Five graphs, three ways of arriving at the bias:
//!
//! | fixture | arch | the bias | the rest |
//! |---|---|---|---|
//! | `refact` | `refact` | the literal 8 (`refact.cpp:12`) | RMSNorm, split Q/K/V, SwiGLU, multi-query |
//! | `bloom` | `bloom` | the literal 8 (`bloom.cpp:18`) | a biased LayerNorm on the EMBEDDINGS (`token_embd_norm`, `:77-80`) and every site, fused `attn_qkv` with bias, the required projection biases, the ungated GELU |
//! | `mpt` | `mpt` | `attention.max_alibi_bias` (`mpt.cpp:6`) | the weighted LayerNorm (no biases), fused `attn_qkv`, the ungated GELU, `attention.clamp_kqv = 4` |
//! | `mpt_posembd` | `mpt` | the key | the same with an OPTIONAL `position_embd` (`:19,80-84`) added as well |
//! | `jais` | `jais` | the key (`jais.cpp:5`) | the biased LayerNorm, fused `attn_qkv` with bias, the required projection biases including `ffn_gate.bias`, SwiGLU |
//! | `baichuan13b` | `baichuan` | the literal 8 at 40 layers ONLY (`baichuan.cpp:11-14`) | the audited Baichuan-7B graph, 40 layers deep |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`
//! (`f_max_alibi_bias = 8.0e+00` in its log for all six). The GELU rows
//! (`bloom`, `mpt`) sit at the f16-table line.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `refact` | 6.44e-13 | 3.40e-06 |
//! | `bloom` | 8.31e-08 | 1.60e-03 (GELU table) |
//! | `mpt` | 3.56e-07 | 2.94e-03 (GELU table) |
//! | `mpt_posembd` | 1.59e-07 | 1.20e-03 (GELU table) |
//! | `jais` | 7.42e-13 | 4.29e-06 |
//! | `baichuan13b` | 1.12e-12 | 4.11e-06 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_alibi_fixture.py \
//!     crates/frink-models/tests/fixtures/<arch>_tiny.gguf --arch <arch> [--clamp] [--pos-embd]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_all_three_paths_match_within,
    assert_decoder_matches_on_all_three_paths, graph_caches, kl_vs_golden, load_graph_fixture,
    worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::alibi::{max_alibi_bias, ALIBI_ARCHITECTURES};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::norm::NormOp;
use frink_models::rope_layers::RopeLayers;
use frink_models::Decoder;

const REFACT: &str = "refact";
const BLOOM: &str = "bloom";
const MPT: &str = "mpt";
const MPT_POSEMBD: &str = "mpt_posembd";
const JAIS: &str = "jais";
const BAICHUAN13B: &str = "baichuan13b";

/// The f16 GELU-table line for the two GELU rows, as in
/// `tests/proj_bias_graphs.rs`.
const GELU_TABLE_TOL_BIASED: f32 = 1e-2;

const REFACT_GOLDEN: [f32; 48] = [
    1.6549939,
    2.8482842,
    -0.010387003,
    -0.41634226,
    -0.5085094,
    0.49147654,
    -0.29592305,
    -2.3339252,
    0.14024913,
    0.036277294,
    -0.07211794,
    0.93730116,
    3.6560688,
    -0.18318093,
    -2.2409916,
    1.357959,
    1.8118945,
    -0.02551955,
    -0.9263804,
    0.36650777,
    -1.8676319,
    -0.13419831,
    -0.9283725,
    0.453684,
    0.020440638,
    -0.5229931,
    0.043088913,
    0.46281737,
    0.03896594,
    -0.21679384,
    2.200129,
    1.2730231,
    -1.3949137,
    -2.9630156,
    -0.21439451,
    1.3479078,
    0.4216987,
    0.860394,
    -2.0292616,
    1.781734,
    -0.67120147,
    0.008468032,
    -2.8120832,
    0.53199613,
    -1.1806753,
    1.7514107,
    1.2578211,
    1.1648668,
];

const BLOOM_GOLDEN: [f32; 48] = [
    -1.7867191,
    0.7388108,
    -3.4743052,
    -2.270637,
    -0.21849859,
    5.5797367,
    1.6390393,
    -0.2807964,
    2.1608799,
    -3.0618083,
    -1.7874427,
    -2.1353576,
    1.091575,
    2.7827163,
    -0.51322424,
    -4.1432314,
    -1.4548144,
    0.35735118,
    -1.7310395,
    0.24778095,
    0.77306867,
    1.4844954,
    -2.227355,
    1.8942745,
    -1.3949765,
    -2.0248065,
    -4.269501,
    0.34574983,
    -3.4594676,
    0.009799838,
    -0.15839553,
    3.19238,
    0.17135137,
    -0.24496937,
    0.18882442,
    -1.6016178,
    -3.366631,
    0.9095422,
    0.6825838,
    -1.668202,
    -0.77305305,
    -1.8451796,
    0.82004434,
    1.6339889,
    0.42025173,
    2.7688892,
    3.5912695,
    1.9232414,
];

const MPT_GOLDEN: [f32; 48] = [
    -1.8736517,
    0.16775644,
    1.7705475,
    -1.885526,
    1.9999611,
    2.931221,
    -1.0334865,
    0.4051898,
    1.2494981,
    0.7812804,
    0.3531245,
    2.9738972,
    -1.7825515,
    2.5045507,
    0.23783487,
    -1.8221941,
    0.81695116,
    0.18903816,
    -0.75063586,
    -0.5714999,
    0.6644517,
    -1.9378046,
    -5.52606,
    0.21358788,
    -0.12632793,
    2.5263638,
    -0.2686491,
    -2.357956,
    1.5284567,
    2.6456342,
    1.5448105,
    0.07085681,
    -1.7139376,
    4.41261,
    -0.036240816,
    1.1686822,
    1.8823804,
    0.27017784,
    -0.23226589,
    3.6014638,
    0.16359799,
    2.8566403,
    1.3327211,
    -2.3698103,
    0.06921953,
    -1.1255114,
    2.422467,
    -3.683622,
];

const MPT_POSEMBD_GOLDEN: [f32; 48] = [
    -1.2202072,
    -0.39005,
    0.88348174,
    -0.017091632,
    2.139869,
    -2.045758,
    -1.1558509,
    -0.13109487,
    -0.39916813,
    1.2367465,
    -0.94237185,
    1.2566822,
    -0.19132155,
    0.84722596,
    -1.572878,
    1.228665,
    -0.594174,
    -1.0190809,
    -2.40279,
    1.5575473,
    -1.0982574,
    1.3794795,
    0.8028422,
    4.070152,
    1.806757,
    2.205661,
    -0.37273294,
    1.0423446,
    3.1094134,
    0.5287744,
    2.2650137,
    2.2150214,
    -2.7188394,
    1.0528259,
    0.049096763,
    4.896858,
    1.1792839,
    -0.27762175,
    -0.5201824,
    0.4254904,
    3.4040837,
    -2.658591,
    4.4674606,
    1.3085467,
    -3.8832135,
    1.053467,
    -5.8463607,
    -1.4078035,
];

const JAIS_GOLDEN: [f32; 48] = [
    1.3309035,
    -4.6245885,
    2.8254647,
    -0.046314,
    -4.654948,
    -5.721507,
    0.37810123,
    1.068189,
    1.2230241,
    -0.5782789,
    -3.4870183,
    -1.1545666,
    -0.6440997,
    -0.6330321,
    0.546469,
    0.6720067,
    0.94705176,
    -2.0071821,
    -1.3295844,
    2.9072833,
    -0.75390744,
    -0.96537673,
    1.9158897,
    0.5769355,
    -4.281908,
    1.1462736,
    -1.0244932,
    0.8130402,
    1.5750359,
    3.0617673,
    -1.1567116,
    1.2914208,
    -3.3444161,
    3.8609219,
    -0.6813557,
    -0.92626154,
    0.4321928,
    -3.6219506,
    -0.3001632,
    5.5139484,
    -0.3059013,
    2.3683224,
    1.0771976,
    -1.6149892,
    2.4883344,
    -2.9018698,
    0.12143296,
    0.95315313,
];

const BAICHUAN13B_GOLDEN: [f32; 48] = [
    0.19693953,
    -0.25763318,
    0.7779955,
    -0.87053114,
    0.916278,
    -0.9924694,
    -0.4748236,
    -0.84000295,
    0.087464735,
    -1.2840397,
    0.85798836,
    0.39043665,
    -0.18626124,
    -2.6698356,
    1.4937129,
    0.99435663,
    0.89514685,
    0.63037527,
    0.017613858,
    0.8068818,
    -0.68519485,
    -0.97518575,
    1.2378159,
    -1.2489398,
    0.8728918,
    0.2719179,
    1.0279036,
    -2.2037148,
    0.7479422,
    0.19265911,
    0.42815858,
    0.23705375,
    -0.30178246,
    0.43060532,
    1.7478276,
    1.8214768,
    -1.9803348,
    -0.68179095,
    0.8169286,
    -0.14931655,
    1.4600272,
    -0.15389845,
    0.56421554,
    0.324741,
    2.8162992,
    0.8135909,
    1.1746902,
    0.23719889,
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
fn refact_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(REFACT, &REFACT_GOLDEN);
}

#[test]
fn bloom_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(BLOOM, &BLOOM_GOLDEN, GELU_TABLE_TOL_BIASED);
}

#[test]
fn mpt_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(MPT, &MPT_GOLDEN, GELU_TABLE_TOL_BIASED);
    assert_all_three_paths_match_within(MPT_POSEMBD, &MPT_POSEMBD_GOLDEN, GELU_TABLE_TOL_BIASED);
}

#[test]
fn jais_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(JAIS, &JAIS_GOLDEN);
}

#[test]
fn baichuan_13b_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(BAICHUAN13B, &BAICHUAN13B_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (REFACT, &REFACT_GOLDEN),
        (BLOOM, &BLOOM_GOLDEN),
        (MPT, &MPT_GOLDEN),
        (MPT_POSEMBD, &MPT_POSEMBD_GOLDEN),
        (JAIS, &JAIS_GOLDEN),
        (BAICHUAN13B, &BAICHUAN13B_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built on every row: the bias at 8 and the slopes
/// derived from it, no layer rotating; plus each graph's own things.
#[test]
fn the_loaded_decoders_are_the_graphs() {
    for (arch, _, _) in ALIBI_ARCHITECTURES {
        assert!(matches!(
            resolve_architecture(arch),
            Some(ArchPath::GenericGqa { .. })
        ));
    }
    for name in [REFACT, BLOOM, MPT, MPT_POSEMBD, JAIS, BAICHUAN13B] {
        let d = load_graph_fixture(name);
        assert_eq!(d.config.alibi_max_bias, Some(8.0), "{name}");
        let slopes = d.alibi_slopes.as_ref().expect("derived from the bias");
        assert_eq!(slopes.len(), d.config.n_heads, "{name}");
        assert_eq!(d.config.rope_layers, RopeLayers::Never, "{name}");
        for il in 0..d.config.n_layers {
            assert!(
                d.config.layer_rope(il).is_none(),
                "{name} layer {il} rotates"
            );
        }
    }

    let refact = load_graph_fixture(REFACT);
    assert_eq!(refact.config.n_kv_heads, 1, "conversion/refact.py:41");
    assert!(matches!(refact.layers[0].attn.norm_weight, NormOp::Rms(_)));

    let bloom = load_graph_fixture(BLOOM);
    assert!(
        matches!(bloom.embedding_norm, NormOp::LayerNormBias { .. }),
        "bloom.cpp:77-80: the embeddings are normed"
    );
    assert!(bloom.layers[0].attn.o_bias.is_some());
    assert!(bloom.layers[0].moe.dense_bias.is_some());

    let mpt = load_graph_fixture(MPT);
    assert!(
        matches!(mpt.layers[0].attn.norm_weight, NormOp::LayerNorm(_)),
        "no biases"
    );
    assert!(matches!(mpt.embedding_norm, NormOp::None));
    assert!(mpt.position_embd.is_none());
    assert_eq!(mpt.config.clamp_kqv, Some(4.0));
    let mpt_pos = load_graph_fixture(MPT_POSEMBD);
    assert!(
        mpt_pos.position_embd.is_some(),
        "mpt.cpp:19: optional, present here"
    );
    assert!(worst_vs(&MPT_GOLDEN, &MPT_POSEMBD_GOLDEN) > 0.1);

    let jais = load_graph_fixture(JAIS);
    let b = jais.layers[0]
        .moe
        .dense_bias
        .as_ref()
        .expect("jais.cpp:41-47");
    assert!(b.gate.is_some() && b.up.is_some() && b.down.is_some());

    let baichuan = load_graph_fixture(BAICHUAN13B);
    assert_eq!(baichuan.config.n_layers, 40);
    assert_eq!(max_alibi_bias("baichuan", 32, None), None, "the 7B rotates");
}

/// The bias is visible: dropping the slopes (what the generic path
/// computed for these files before, minus the rotation it also did),
/// rotating on top of them, and the slopes of the wrong head count each
/// diverge; and the two GELU rows stay within the table line.
#[test]
fn the_bias_and_the_absence_of_rotation_are_visible_in_the_logits() {
    let mut d = load_graph_fixture(REFACT);
    assert_decoder_matches_on_all_three_paths(&d, &REFACT_GOLDEN, GRAPH_TOL, "baseline");

    let saved = d.alibi_slopes.take();
    let worst = worst_vs(&decode(&d), &REFACT_GOLDEN);
    assert!(worst > 1e-1, "the bias not seen: {worst}");
    d.alibi_slopes = saved;

    d.config.rope_layers = RopeLayers::All;
    let worst = worst_vs(&decode(&d), &REFACT_GOLDEN);
    assert!(worst > 1e-1, "rotating an ALiBi model not seen: {worst}");
    d.config.rope_layers = RopeLayers::Never;

    // The wrong slope table: half the bias (m0 for 16 heads on a
    // 4-head model) is the plausible slip in the formula.
    let saved = d
        .alibi_slopes
        .replace(frink_core::alibi::slopes(4, 4.0).unwrap());
    let worst = worst_vs(&decode(&d), &REFACT_GOLDEN);
    assert!(worst > 1e-2, "the slope formula not seen: {worst}");
    d.alibi_slopes = saved;
    assert_decoder_matches_on_all_three_paths(&d, &REFACT_GOLDEN, GRAPH_TOL, "restored");

    // Baichuan-13B: the 40-layer file positioned by ALiBi; served as the
    // 7B (rotated, unbiased) it is a different model.
    let mut b = load_graph_fixture(BAICHUAN13B);
    b.alibi_slopes = None;
    b.config.rope_layers = RopeLayers::All;
    let worst = worst_vs(&decode(&b), &BAICHUAN13B_GOLDEN);
    assert!(worst > 1e-1, "the 13B run as the 7B not seen: {worst}");
}
