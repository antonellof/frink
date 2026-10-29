//! Cohere2 MoE (`cohere2moe`), checked against llama.cpp itself.
//!
//! `cohere2moe.cpp` is `cohere2.cpp` (the shared-norm PARALLEL
//! residual, `crate::parallel_residual`; a REQUIRED window and
//! `logit_scale`; NORM RoPE) with routed experts, and three things the
//! generic path did not have: a layer rotates when it slides OR sits in
//! the dense prefix (`:177-179,192`, `rope_layers::RopeLayers::
//! SlidingOrLeadingDense`); with a shared expert the branch is
//! `(moe_out + shexp) * 0.5` (`:248-260`, `parallel_dense_ffn::
//! SHARED_EXPERT_SUM_SCALE`); and the norm FUNCTION comes from which
//! epsilon key the file carries (`:4-11,166`, `norm::
//! NORM_BY_RMS_EPS_KEY`). The router reads `ffn_inp = attn_norm(inpL)`
//! (`:234`), which on a parallel layer IS the normed FFN input the
//! generic router reads (`crate::router_input`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `cohere2moe` (LayerNorm, the per-layer array `[F, T, T, F]`, dense lead 1, sigmoid, shared expert) | see `report_kl_against_llama_cpp` | |
//! | `cohere2moe_rms` (`layer_norm_rms_epsilon`: RMSNorm) | | |
//! | `cohere2moe_mtp` (one NextN block inside `block_count`; libllama's golden is byte-identical to the trunk's) | | |
//! | `cohere2moe_normw` (softmax from the key, `expert_weights_norm = true`) | | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_cohere2moe_fixture.py \
//!     crates/frink-models/tests/fixtures/cohere2moe_tiny.gguf [--rms | --mtp | --norm-w]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/cohere2moe_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::RopeLayout;
use frink_models::norm::{NormFunction, NormOp};
use frink_models::rope_layers::RopeLayers;
use frink_models::Decoder;
use frink_moe::GatingFunction;

const C2M: &str = "cohere2moe";
const C2M_RMS: &str = "cohere2moe_rms";
const C2M_MTP: &str = "cohere2moe_mtp";
const C2M_NORMW: &str = "cohere2moe_normw";

const C2M_GOLDEN: [f32; 48] = [
    -0.1998614,
    0.37380674,
    -0.19802338,
    0.61683923,
    -0.24817622,
    0.27394673,
    0.33277515,
    0.46428928,
    0.8476461,
    0.53871274,
    -0.4580558,
    0.29034784,
    0.07116176,
    -0.17490497,
    -0.47162175,
    0.6247042,
    0.13002525,
    -0.10363444,
    -0.22540446,
    -0.34327394,
    0.06621427,
    -0.01933147,
    0.32743692,
    -0.3973254,
    -0.16289756,
    0.47000232,
    0.021436084,
    0.23537947,
    0.25207362,
    -0.28439453,
    -0.18058646,
    0.17354926,
    0.12614155,
    0.17252661,
    -0.4985469,
    -0.18669313,
    0.61947477,
    -0.5215068,
    0.04284755,
    0.050296135,
    -0.2349889,
    0.24461979,
    -0.12719606,
    0.23694229,
    0.416066,
    -0.70547587,
    0.38758397,
    -0.021045685,
];

const C2M_RMS_GOLDEN: [f32; 48] = [
    -0.43148744,
    0.23549667,
    -0.11741187,
    0.34025976,
    -0.26691404,
    0.19328812,
    0.4128005,
    0.16230115,
    0.7562505,
    0.46654668,
    -0.070075646,
    0.31787497,
    -0.22104454,
    -0.1797974,
    -0.22936016,
    0.64344937,
    0.038543046,
    -0.16798586,
    0.15880302,
    -0.50610054,
    -0.040432096,
    0.11790031,
    0.25467703,
    -0.56803834,
    -0.025621396,
    0.45654762,
    0.048178762,
    0.19698687,
    0.28510842,
    -0.21242456,
    -0.0412717,
    0.21372958,
    0.31924596,
    0.23058257,
    -0.7089049,
    0.032102898,
    0.6030538,
    -0.33516952,
    -0.077708,
    0.10184398,
    -0.092901275,
    0.3082506,
    0.005192158,
    0.15502135,
    0.3849722,
    -0.5768224,
    0.24720299,
    0.21895123,
];

const C2M_NORMW_GOLDEN: [f32; 48] = [
    -0.14975457,
    0.41398874,
    -0.17658284,
    0.7056119,
    -0.25934166,
    0.2688579,
    0.23348208,
    0.45345914,
    0.7950763,
    0.4779842,
    -0.47900882,
    0.35028276,
    0.122362055,
    -0.20576477,
    -0.4641306,
    0.6002639,
    0.054598294,
    -0.06358287,
    -0.17015333,
    -0.39523208,
    0.07257469,
    -0.10447755,
    0.34128127,
    -0.44539028,
    -0.18468432,
    0.35598153,
    -0.004257124,
    0.22213942,
    0.21330935,
    -0.28551766,
    -0.20770809,
    0.2822167,
    0.12721626,
    0.30126446,
    -0.4994516,
    -0.22059251,
    0.6337675,
    -0.53521687,
    0.025233077,
    -0.059645183,
    -0.26314467,
    0.21333067,
    -0.14011842,
    0.23075938,
    0.33885512,
    -0.6646887,
    0.4640737,
    -0.031161204,
];

fn decode(d: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(d);
    d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv)
}

#[test]
fn cohere2moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(C2M, &C2M_GOLDEN);
}

/// A nonzero `layer_norm_rms_epsilon` switches every norm to RMS
/// (`cohere2moe.cpp:166`).
#[test]
fn cohere2moe_rms_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(C2M_RMS, &C2M_RMS_GOLDEN);
    let d = load_graph_fixture(C2M_RMS);
    assert_eq!(d.config.norm_function, NormFunction::Rms);
    assert!(matches!(d.layers[0].attn.norm_weight, NormOp::Rms(_)));
    assert!(matches!(d.final_norm, NormOp::Rms(_)));
}

/// The MTP block is inside `block_count` and outside the graph
/// (`crate::mtp_blocks`): libllama's logits for this file are
/// byte-identical to the trunk-only file's, and so are frink's.
#[test]
fn cohere2moe_mtp_matches_llama_cpp_and_the_trunk() {
    assert_all_three_paths_match(C2M_MTP, &C2M_GOLDEN);
    let d = load_graph_fixture(C2M_MTP);
    assert_eq!(d.config.n_layers, 4);
    assert_eq!(d.config.n_mtp_blocks, 1);
}

/// The gating key and `expert_weights_norm` are READ (`cohere2moe.cpp:
/// 19-21`): softmax with renormalisation lands on its own golden.
#[test]
fn cohere2moe_normw_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(C2M_NORMW, &C2M_NORMW_GOLDEN);
    let d = load_graph_fixture(C2M_NORMW);
    assert_eq!(d.config.moe.gating, GatingFunction::Softmax);
    assert!(d.config.moe.norm_topk_prob);
    assert!(
        worst_vs(&C2M_NORMW_GOLDEN, &C2M_GOLDEN) > 1e-2,
        "the two goldens differ"
    );
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (C2M, &C2M_GOLDEN),
        (C2M_RMS, &C2M_RMS_GOLDEN),
        (C2M_MTP, &C2M_GOLDEN),
        (C2M_NORMW, &C2M_NORMW_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: the weighted LayerNorm from the
/// `layer_norm_epsilon` key alone, the parallel residual on every
/// layer, the dense prefix rotated although it does not slide, the
/// full layer past it unrotated, sigmoid from no key, `norm_w` false,
/// the shared expert with the `0.5` on the sum, `logit_scale`.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("cohere2moe"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Norm
        })
    ));
    let d = load_graph_fixture(C2M);
    let c = &d.config;
    assert_eq!(
        c.norm_function,
        NormFunction::LayerNorm,
        "cohere2moe.cpp:166"
    );
    assert!(matches!(d.layers[0].attn.norm_weight, NormOp::LayerNorm(_)));
    assert!(c.parallel_residual);
    assert_eq!(c.sliding_window, Some(3));
    let slides: Vec<bool> = (0..4)
        .map(|il| c.layer_sliding_window(il).is_some())
        .collect();
    assert_eq!(slides, [false, true, true, false], "the per-layer array");
    assert_eq!(
        c.rope_layers,
        RopeLayers::SlidingOrLeadingDense { n_dense_lead: 1 }
    );
    let rotates: Vec<bool> = (0..4).map(|il| c.layer_rotates(il)).collect();
    assert_eq!(
        rotates,
        [true, true, true, false],
        "cohere2moe.cpp:177-179,192"
    );
    assert_eq!(c.rope_theta_swa, Some(c.rope_theta), "cohere2moe.cpp:38");
    assert_eq!(c.n_dense_leading_layers, 1);
    assert_eq!(
        c.moe.gating,
        GatingFunction::Sigmoid,
        "cohere2moe.cpp:27-29"
    );
    assert!(!c.moe.norm_topk_prob);
    assert_eq!(c.moe.n_experts, 4);
    assert_eq!(c.moe.n_experts_active, 2);
    assert_eq!(c.moe.n_shared_experts, 1);
    assert_eq!(c.logit_multiplier, Some(0.25));
    assert_eq!(d.layers[0].moe.n_experts(), 1);
    assert_eq!(d.layers[0].moe.parallel_sum_scale, None);
    for il in 1..4 {
        assert_eq!(d.layers[il].moe.n_experts(), 4, "blk.{il}");
        assert_eq!(d.layers[il].moe.shared_experts.len(), 1);
        assert_eq!(
            d.layers[il].moe.parallel_sum_scale,
            Some(0.5),
            "cohere2moe.cpp:259"
        );
    }
}

/// Each seam is visible: the `0.5`, the dense prefix's rotation, the
/// unrotated full layer, the norm function.
#[test]
fn every_cohere2moe_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(C2M);
    assert_decoder_matches_on_all_three_paths(&d, &C2M_GOLDEN, GRAPH_TOL, "baseline");

    for il in 1..4 {
        d.layers[il].moe.parallel_sum_scale = None;
    }
    let worst = worst_vs(&decode(&d), &C2M_GOLDEN);
    assert!(worst > 1e-2, "the 0.5 on the sum not seen: {worst}");
    for il in 1..4 {
        d.layers[il].moe.parallel_sum_scale = Some(0.5);
    }

    // `cohere2`'s rule: the dense prefix would not rotate.
    d.config.rope_layers = RopeLayers::SlidingOnly;
    let worst = worst_vs(&decode(&d), &C2M_GOLDEN);
    assert!(
        worst > 1e-2,
        "the dense prefix's rotation not seen: {worst}"
    );
    d.config.rope_layers = RopeLayers::All;
    let worst = worst_vs(&decode(&d), &C2M_GOLDEN);
    assert!(worst > 1e-2, "the unrotated full layer not seen: {worst}");
    d.config.rope_layers = RopeLayers::SlidingOrLeadingDense { n_dense_lead: 1 };

    d.config.moe.norm_topk_prob = true;
    let worst = worst_vs(&decode(&d), &C2M_GOLDEN);
    assert!(worst > 1e-2, "norm_w not seen: {worst}");
    d.config.moe.norm_topk_prob = false;
    assert_decoder_matches_on_all_three_paths(&d, &C2M_GOLDEN, GRAPH_TOL, "restored");

    // The RMS file under the LayerNorm reading, and vice versa.
    let rms = load_graph_fixture(C2M_RMS);
    assert!(worst_vs(&decode(&rms), &C2M_GOLDEN) > 1e-2);
}

/// The two throws in `cohere2moe.cpp`'s `load_arch_hparams` no table
/// states (#464), refused by name as libllama refuses the same files:
/// `cohere2moe_noeps_tiny.gguf` (`missing Cohere2 MoE norm epsilon`,
/// `:4-9`) and `cohere2moe_nodense_tiny.gguf` (`key not found in model:
/// cohere2moe.leading_dense_block_count`, `:15`), both measured.
#[test]
fn a_cohere2moe_file_missing_a_required_hparam_is_refused_by_name() {
    for (name, needle) in [
        ("cohere2moe_noeps", "cohere2moe.cpp:4-9"),
        ("cohere2moe_nodense", "cohere2moe.cpp:15"),
    ] {
        let path = format!(
            "{}/tests/fixtures/{name}_tiny.gguf",
            env!("CARGO_MANIFEST_DIR")
        );
        let file = frink_gguf::GgufFile::open(&path).expect("fixture opens");
        match frink_models::ModelConfig::from_gguf(&file) {
            Err(frink_models::LoadError::UnsupportedFeature(arch, msg)) => {
                assert_eq!(arch, "cohere2moe", "{name}");
                assert!(msg.contains(needle), "{name}: {msg}");
            }
            other => panic!("{name} must be refused, got {other:?}"),
        }
    }
}
