//! Gated attention, checked against llama.cpp itself: `afmoe` and
//! `laguna` on ONE seam (`frink_models::attn_gate`).
//!
//! Both were triaged NEW CODE naming `wqkv_gate` -- llama.cpp's
//! `LLM_TENSOR_ATTN_GATE`, `blk.N.attn_gate.weight` -- as what was left
//! after the per-layer shape seam closed. `step35` names it too. The
//! three graphs were read SIDE BY SIDE before the seam was built, and
//! they are one op with two free parameters rather than one graph:
//!
//! | arch | activation | width | presence |
//! |---|---|---|---|
//! | `afmoe` (`afmoe.cpp:73,154,183-185`) | sigmoid | per element | required |
//! | `laguna` (`laguna.cpp:110-124,211,246-257`) | softplus | per head OR per element, off the tensor | required |
//! | `step35` (`step35.cpp:96,268-284`) | sigmoid | per head | optional |
//!
//! Identical in all three: the gate is projected from the SAME normed
//! input Q/K/V read, multiplies the attention output after the
//! softmax-weighted V sum and before `wo`, and a per-head value
//! broadcasts over its head's `head_dim` channels. Six of llama.cpp's
//! 155 graphs create the tensor; the other three (`qwen3next`, `qwen35`,
//! `qwen35moe`) store the gated delta-net's `z` projection under the
//! same name, a different op on a different engine
//! (`attn_gate::GDN_Z_GATE_ARCHS`).
//!
//! **Three fixtures cover the seam's corners.** `afmoe` is sigmoid /
//! per element; `laguna` (M.1 shape) is softplus / per element;
//! `laguna_swa` (XS.2 shape) is softplus / per head, and its per-layer
//! `head_count` array means the per-head gate is sized by each layer's
//! own count. `step35`'s corner, sigmoid / per head / optional, is the
//! product of two axes each evidenced here, and the row closed on its
//! own suite (`tests/clamped_swiglu_graphs.rs`).
//!
//! **One laguna refusal is evidenced from a file that has the thing,
//! and one former refusal is served from one.** `laguna_swa_rot`
//! declares `rope.dimension_count_swa` half the head, which libllama
//! honours (`n_rot(il)` at llama-hparams.cpp:85-91 -- its golden
//! differs from `laguna_swa`'s by up to 0.3); frink honours it too
//! now, through `ModelConfig::rope_dim_swa` (`frink_models::
//! swa_geometry`, the seam Step-3.5's halved full-layer width landed
//! on), and matches that golden. `laguna_swa_yarn` declares a YaRN
//! scaling that `laguna.cpp:48,184-192` switch off on the sliding
//! layers, and stays refused by name.
//!
//! **Where the numbers come from.** Each golden was produced by running
//! llama.cpp's own graph over its fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! Regenerating:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_afmoe_fixture.py \
//!     crates/frink-models/tests/fixtures/afmoe_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_laguna_fixture.py \
//!     crates/frink-models/tests/fixtures/laguna_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_laguna_fixture.py \
//!     crates/frink-models/tests/fixtures/laguna_swa_tiny.gguf --swa
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_laguna_fixture.py \
//!     crates/frink-models/tests/fixtures/laguna_swa_rot_tiny.gguf --swa-rot
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_laguna_fixture.py \
//!     crates/frink-models/tests/fixtures/laguna_swa_yarn_tiny.gguf --swa-yarn
//! /tmp/ref_logits <fixture> 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_all_three_paths_match_within, graph_caches,
    graph_fixture_path, kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT,
};
use frink_models::attn_gate::{GateAct, GateWidth};
use frink_models::layer_shapes::AttnShape;
use frink_models::ModelConfig;

const AFMOE: &str = "afmoe";
const LAGUNA: &str = "laguna";
const LAGUNA_SWA: &str = "laguna_swa";
const SPARK25: &str = "spark25";

/// `spark2_5` is the suite's one GeGLU row, so llama.cpp's f16 GELU
/// table is the approximate side and `GELU_TABLE_TOL` (2e-4, set from
/// a Gemma-shaped fixture) is too tight for it. MEASURED rather than
/// widened by feel: with frink's `gelu_mul` made to emulate ggml's
/// table -- input rounded to f16, the tanh form, the result rounded to
/// f16 again, which is what `GGML_GELU_FP16` does at
/// `ggml/src/ggml-cpu/vec.h:1414-1425` -- this fixture agrees with
/// libllama to **6.08e-6** (KL 3.34e-12) where the unemulated build
/// reads **2.86e-3** (KL 7.70e-7). So the whole of the gap is the
/// table, on a four-layer file whose residual stream the attention
/// amplifies; the same shape as `starcoder2` and `codeshell` in
/// `tests/proj_bias_graphs.rs`, which carry their own constant for the
/// same reason.
///
/// 5e-3 is under every sabotage below -- dropping the gate moves these
/// logits by more than 1, and reading the window array as a period
/// moves them by 3.5 (both measured while the row was built).
const SPARK25_GELU_TOL: f32 = 5e-3;

const AFMOE_GOLDEN: [f32; 48] = [
    -0.17821455,
    1.3820034,
    0.0337075,
    -0.3012852,
    -0.4508564,
    2.4009683,
    -1.082543,
    -0.5653495,
    0.5819897,
    -1.5246291,
    -1.8619847,
    0.90694714,
    -1.15742,
    -2.6498086,
    -2.7920437,
    -0.8673738,
    -0.11308196,
    0.44642755,
    -1.2857102,
    1.2340804,
    -1.6425024,
    0.45722282,
    2.0952306,
    -0.11279556,
    1.6935093,
    -1.7845697,
    -0.41834974,
    -2.0837069,
    -4.2632957,
    2.8340116,
    2.0910254,
    0.20614135,
    1.2104105,
    1.3846397,
    0.33692575,
    -1.8186575,
    0.4888968,
    -1.2707634,
    0.1681897,
    0.23549247,
    -3.128251,
    -0.6434635,
    -2.3148022,
    -0.5317296,
    2.23949,
    1.0068989,
    -1.3572114,
    -5.9129972,
];

const LAGUNA_GOLDEN: [f32; 48] = [
    -1.5119162,
    1.4036855,
    0.6291794,
    0.66163397,
    -1.9894987,
    1.9093504,
    -0.49081546,
    -0.08843517,
    0.07342255,
    -1.3001451,
    0.15341505,
    0.28918207,
    0.24776477,
    -1.3133051,
    -2.1886113,
    3.267044,
    -0.42774922,
    -2.783679,
    1.2199318,
    1.0702016,
    4.7013416,
    3.0918355,
    0.53851604,
    1.7211139,
    3.4720519,
    -0.7194342,
    1.7778969,
    -1.0225552,
    -1.3829987,
    1.2933049,
    -0.39992058,
    2.2462418,
    0.05706048,
    1.406198,
    0.8671186,
    0.7189059,
    -0.6231334,
    0.14192748,
    0.2384373,
    -0.57008594,
    -2.0872383,
    1.2017615,
    1.9997605,
    -1.5388064,
    -0.192842,
    -1.0803405,
    -1.803505,
    -1.0824968,
];

const LAGUNA_SWA_GOLDEN: [f32; 48] = [
    -2.5766973,
    -0.027115047,
    1.2975299,
    2.3285065,
    -0.8408443,
    1.3943012,
    0.07298869,
    2.459518,
    -0.88847154,
    1.1134903,
    -3.5601313,
    -0.5942931,
    -1.3527534,
    -1.5217843,
    3.3961315,
    0.7684222,
    0.5481303,
    -1.2206059,
    -2.996244,
    1.202471,
    2.7001061,
    -2.5136843,
    -1.2062895,
    -0.23398805,
    -0.8197799,
    -2.3169868,
    1.2953389,
    -0.16158393,
    1.9031371,
    3.1668959,
    -2.3783312,
    -1.4109818,
    -4.083351,
    1.7414421,
    -2.3296978,
    -0.3878163,
    -2.0514178,
    -1.467369,
    0.5651611,
    -0.42475224,
    -1.4477028,
    0.77115947,
    -1.9698552,
    -0.9428121,
    -0.12386471,
    0.46544915,
    0.7005613,
    0.4321354,
];

const SPARK25_GOLDEN: [f32; 48] = [
    1.1904615,
    -1.5599897,
    -0.4287688,
    -0.79204,
    4.722593,
    2.877309,
    -1.7242424,
    2.707797,
    1.2134902,
    0.6893058,
    0.9260141,
    -0.33815855,
    -1.0772187,
    1.4401295,
    -1.3031607,
    -0.7704283,
    -0.66947174,
    0.182432,
    0.6031899,
    -0.2503235,
    -0.65955424,
    -0.12335199,
    -0.056718677,
    0.5392068,
    1.0678068,
    -1.328313,
    -0.39162576,
    2.3502724,
    1.288382,
    0.23902476,
    0.58983445,
    0.939625,
    -0.3550618,
    -0.18232766,
    -1.142412,
    1.8027571,
    -0.12490639,
    -2.3195045,
    2.1821742,
    1.4608009,
    -0.59154177,
    -2.1486416,
    -0.22655094,
    0.832515,
    -0.28606927,
    -2.0903263,
    -0.39845926,
    0.6822282,
];

#[test]
fn afmoe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(AFMOE, &AFMOE_GOLDEN);
}

#[test]
fn laguna_m1_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(LAGUNA, &LAGUNA_GOLDEN);
}

#[test]
fn laguna_xs2_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(LAGUNA_SWA, &LAGUNA_SWA_GOLDEN);
}

/// Spark-2.5, the first row closed against the llama.cpp pin moved on
/// 2026-09-19.
///
/// It is `step35`'s corner of the two axes -- sigmoid, per head -- with
/// the tensor REQUIRED instead of optional, so what this golden adds to
/// the suite is not the arithmetic (`laguna_swa` already evidences a
/// per-head gate) but the ROW: an architecture whose graph landed
/// upstream six weeks after the last census, gated the same way, and
/// was refused here until somebody read it.
#[test]
fn spark25_matches_llama_cpp_on_all_three_paths() {
    // GeGLU: `spark2-5.cpp:124` is `LLM_FFN_GELU` under `LLM_FFN_PAR`,
    // and llama.cpp's CPU GELU is an f16 lookup table, so the
    // reference is the approximate side here exactly as it is for
    // `grok`, `falcon` and the two StarCoder rows.
    assert_all_three_paths_match_within(SPARK25, &SPARK25_GOLDEN, SPARK25_GELU_TOL);
}

/// The numbers in the report, so they can be regenerated rather than
/// trusted. Run with `--nocapture` to see them.
#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden, kl_line) in [
        (AFMOE, &AFMOE_GOLDEN, 1e-8),
        (LAGUNA, &LAGUNA_GOLDEN, 1e-8),
        (LAGUNA_SWA, &LAGUNA_SWA_GOLDEN, 1e-8),
        // The GeGLU row's line is llama.cpp's own f16 GELU table, and
        // the constant above says what it measures to with the table
        // emulated (3.34e-12).
        (SPARK25, &SPARK25_GOLDEN, 1e-5),
    ] {
        let d = load_graph_fixture(name);
        let mut kv = graph_caches(&d);
        let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
        let kl = kl_vs_golden(&got, golden);
        let worst = worst_vs(&got, golden);
        println!("| `{name}` | {kl:.2e} | {worst:.2e} |");
        assert!(kl < kl_line, "{name}: KL {kl}");
    }
}

/// Each fixture loaded the gate the table says it has, at the width
/// its tensor has: structural, so a loader that read the wrong
/// activation or the wrong width says which before the numeric test
/// says "diverged".
#[test]
fn each_fixture_loads_the_gate_its_architecture_and_tensor_declare() {
    let d = load_graph_fixture(AFMOE);
    for (il, layer) in d.layers.iter().enumerate() {
        let g = layer
            .attn
            .output_gate
            .as_ref()
            .unwrap_or_else(|| panic!("afmoe blk.{il}"));
        assert_eq!(g.act, GateAct::Sigmoid);
        assert_eq!(g.width, GateWidth::PerElement);
        assert_eq!(g.proj.rows(), d.config.n_heads * d.config.head_dim);
        assert_eq!(g.proj.cols(), d.config.hidden_dim);
    }
    // afmoe.cpp:120: the only non-Gemma graph that scales its
    // embeddings by sqrt(n_embd).
    assert_eq!(
        d.config.embedding_scale,
        Some((d.config.hidden_dim as f32).sqrt())
    );

    let d = load_graph_fixture(LAGUNA);
    assert_eq!(
        d.config.sliding_window, None,
        "laguna.cpp:34-35: M.1 has no window"
    );
    for (il, layer) in d.layers.iter().enumerate() {
        let g = layer
            .attn
            .output_gate
            .as_ref()
            .unwrap_or_else(|| panic!("laguna blk.{il}"));
        assert_eq!(g.act, GateAct::Softplus);
        assert_eq!(g.width, GateWidth::PerElement);
    }

    let d = load_graph_fixture(LAGUNA_SWA);
    assert_eq!(d.config.sliding_window, Some(3));
    // laguna.cpp:41: dense_first, so layer 0 is full and 1-3 slide.
    assert_eq!(d.config.layer_sliding_window(0), None);
    for il in 1..4 {
        assert_eq!(d.config.layer_sliding_window(il), Some(3), "blk.{il}");
    }
    assert_eq!(d.config.rope_theta_swa, Some(5000.0));
    for (il, layer) in d.layers.iter().enumerate() {
        let n_heads = match d.config.layer_shape(il).attention {
            AttnShape::Gqa { n_heads, .. } => n_heads,
            other => panic!("blk.{il}: {other:?}"),
        };
        assert_eq!(
            n_heads,
            [2, 4, 4, 4][il],
            "conversion/laguna.py:79 writes an array"
        );
        let g = layer
            .attn
            .output_gate
            .as_ref()
            .unwrap_or_else(|| panic!("laguna_swa blk.{il}"));
        assert_eq!(g.act, GateAct::Softplus);
        assert_eq!(g.width, GateWidth::PerHead);
        assert_eq!(
            g.proj.rows(),
            n_heads,
            "laguna.cpp:110: sized by THIS layer's count"
        );
    }
}

/// The gate is load-bearing on every fixture: dropping it moves the
/// output by orders of magnitude more than the comparison tolerance,
/// and so does swapping the activation, so the suite cannot pass with
/// a gate that is applied but wrong.
#[test]
fn the_gate_and_its_activation_are_visible_in_the_logits() {
    for (name, golden) in [
        (AFMOE, &AFMOE_GOLDEN),
        (LAGUNA, &LAGUNA_GOLDEN),
        (LAGUNA_SWA, &LAGUNA_SWA_GOLDEN),
    ] {
        // 1. No gate at all.
        let mut d = load_graph_fixture(name);
        for layer in d.layers.iter_mut() {
            layer.attn.output_gate = None;
        }
        let mut kv = graph_caches(&d);
        let worst = worst_vs(&d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv), golden);
        assert!(
            worst > 1e-2,
            "{name}: removing the gate moved the output by only {worst}; the gate is dead code"
        );
        // 2. The other activation.
        let mut d = load_graph_fixture(name);
        for layer in d.layers.iter_mut() {
            let g = layer.attn.output_gate.as_mut().unwrap();
            g.act = match g.act {
                GateAct::Sigmoid => GateAct::Softplus,
                GateAct::Softplus => GateAct::Sigmoid,
            };
        }
        let mut kv = graph_caches(&d);
        let worst = worst_vs(&d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv), golden);
        assert!(
            worst > 1e-2,
            "{name}: swapping the activation moved the output by only {worst}"
        );
    }
}

/// A per-head gate read as per-element, or the reverse, is a shape
/// error the loader refuses rather than a broadcast it guesses: the
/// M.1 fixture's per-element tensor under the XS.2 fixture's per-head
/// architecture is fine (laguna admits both), but under `afmoe`, which
/// admits only per-element, a per-head tensor is refused naming the
/// admissible width.
#[test]
fn a_gate_width_the_architecture_does_not_admit_is_refused_naming_the_tensor() {
    // Load the XS.2 fixture's per-head tensors as if for afmoe.
    let path = graph_fixture_path(LAGUNA_SWA);
    let file = frink_gguf::GgufFile::open(&path).expect("fixture opens");
    let err = frink_models::attn_gate::AttnGate::load(&file, "afmoe", 1, 4, 8, 32)
        .expect_err("afmoe admits only a per-element gate");
    let msg = format!("{err}");
    assert!(
        msg.contains("blk.1.attn_gate.weight has 4 output rows"),
        "{msg}"
    );
    assert!(msg.contains("32 (per element)"), "{msg}");
    // And laguna takes the same tensor as per head.
    let g = frink_models::attn_gate::AttnGate::load(&file, "laguna", 1, 4, 8, 32)
        .expect("laguna admits per head")
        .expect("present");
    assert_eq!(g.width, GateWidth::PerHead);
    // An architecture with no gate reads none, even from a file that
    // has one -- the tensor is then refused as unconsumed by the full
    // loader, which is a different test's business.
    assert!(
        frink_models::attn_gate::AttnGate::load(&file, "llama", 1, 4, 8, 32)
            .expect("no spec, no read")
            .is_none()
    );
}

/// llama.cpp's logits for `laguna_swa_rot_tiny.gguf`: the sliding
/// layers rotate 4 of 8 dims, the full layers all 8.
const LAGUNA_SWA_ROT_GOLDEN: [f32; 48] = [
    -2.900597,
    -0.21544212,
    1.1636807,
    2.0950882,
    -0.66421103,
    1.5038196,
    0.1658414,
    2.6975534,
    -1.0402565,
    0.99273187,
    -3.1984298,
    -0.96442115,
    -1.9525335,
    -1.4564579,
    3.614548,
    1.1388557,
    0.92276263,
    -1.3367314,
    -3.053658,
    1.1712291,
    3.2525735,
    -2.7948487,
    -0.9251003,
    -0.23932657,
    -0.8467276,
    -2.6862788,
    0.9524888,
    -0.51321614,
    1.3821557,
    2.6605582,
    -1.7582645,
    -1.4891844,
    -4.2057133,
    1.7533894,
    -2.1401525,
    -0.6105618,
    -2.5806613,
    -1.5213596,
    0.009132922,
    -0.0018042773,
    -1.1191428,
    0.8924065,
    -1.6698679,
    -0.83779836,
    0.3776102,
    0.081464976,
    0.7033006,
    1.0798461,
];

/// A second rotary width for the sliding layers, from a file libllama
/// loads and runs differently with it (this golden differs from
/// `laguna_swa`'s), is SERVED: each layer rotates its own width
/// (`ModelConfig::layer_rope`), and rotating every layer at either one
/// width diverges. This used to be a refusal by name.
#[test]
fn a_swa_rotary_width_differing_from_the_full_one_is_served_per_layer() {
    assert_all_three_paths_match("laguna_swa_rot", &LAGUNA_SWA_ROT_GOLDEN);
    let d = load_graph_fixture("laguna_swa_rot");
    assert_eq!(d.config.rope_dim, None, "full layers: the whole head");
    assert_eq!(d.config.rope_dim_swa, Some(4), "sliding layers: half");
    assert!(d.config.rope_dim_varies_by_layer());
    assert!(
        worst_vs(&LAGUNA_SWA_ROT_GOLDEN, &LAGUNA_SWA_GOLDEN) > 1e-1,
        "the two fixtures must disagree, or the width is invisible"
    );
    for (what, full, swa) in [("all whole", None, None), ("all half", Some(4), None)] {
        let mut d = load_graph_fixture("laguna_swa_rot");
        d.config.rope_dim = full;
        d.config.rope_dim_swa = swa;
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &LAGUNA_SWA_ROT_GOLDEN,
        );
        assert!(worst > 1e-2, "{what}: the output moved by only {worst}");
    }
}

/// A window together with a RoPE scaling is Laguna-XS.2's real shape
/// and is refused by name, because `laguna.cpp:48,184-192` run the
/// sliding layers with the scaling switched off and frink carries one
/// `attn_factor` for the whole model. The same rule `olmo2` (Olmo-3)
/// and `mellum` carry, one table (`swa_geometry::swa_layers_unscaled_rope`).
#[test]
fn a_window_together_with_a_rope_scaling_is_refused_as_the_olmo3_rule() {
    let file =
        frink_gguf::GgufFile::open(graph_fixture_path("laguna_swa_yarn")).expect("fixture opens");
    let err = ModelConfig::from_gguf(&file).expect_err("refused at the header");
    let msg = format!("{err}");
    assert!(msg.contains("sliding window"), "{msg}");
    assert!(msg.contains("yarn"), "{msg}");
    assert!(msg.contains("laguna.cpp:48,181-193"), "{msg}");
}

/// The paged decode path carries the gate too: it runs `attn_block`
/// with a different KV backing and nothing else, and this is the check
/// that a gated model answers the same out of a paged pool.
#[test]
fn paged_decode_matches_contiguous_on_a_gated_model() {
    let d = load_graph_fixture(LAGUNA_SWA);
    let store = std::sync::Arc::new(d.config.new_paged_kv(4, 8));
    let mut paged: Vec<frink_core::cache::PagedKvCache> = (0..d.layers.len())
        .map(|_| frink_core::cache::PagedKvCache::new())
        .collect();
    let mut contiguous = d.config.new_kv_caches();
    let mut want = Vec::new();
    let mut got = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        want = d.forward_token(tok, pos, &mut contiguous);
        got = d
            .forward_token_paged(tok, pos, &mut paged, &store)
            .expect("8 blocks of 4 hold 6 positions");
    }
    assert_eq!(
        got, want,
        "paged and contiguous decode must be bit-identical"
    );
    common::assert_close(&got, &LAGUNA_SWA_GOLDEN, common::GRAPH_TOL, "paged decode");
}

/// With Metal switched ON, a gated model is kept off the fused
/// attention path entirely: `forward_token` takes it only when
/// `layer_supports_metal_attn` holds for EVERY layer, that predicate
/// asks `Decoder::metal_attn_view`, and the view answers `None` for a
/// layer with a gate. This is the hardware half of the
/// exhaustive-destructure fence: the unit test pins the predicate, this
/// pins that the launch decision reads it.
///
/// The assertion that carries the proof is the Metal KV probe, not the
/// logits. These fixtures are F32 MoE files, and no fused MoE kernel
/// takes F32 experts, so with the fence REMOVED (measured, under
/// sabotage) the per-layer launches still fall back to the host and
/// the logits still match -- what changes is that the fused path was
/// ENTERED and allocated its KV. So: the gated model must allocate NO
/// Metal KV, and the same file with its gates stripped -- the same
/// weights minus the one thing the kernels cannot do -- MUST, which is
/// what proves the switch was live in this process at all.
///
/// `cargo test -p frink-models --features metal --test gated_attention_graphs -- --ignored`
#[test]
#[ignore = "needs Apple Metal GPU"]
fn a_gated_model_matches_llama_cpp_with_metal_switched_on() {
    #[cfg(feature = "metal")]
    common::pin_backend(common::Backend::Metal);
    #[cfg(not(feature = "metal"))]
    {
        eprintln!("skip: built without --features metal");
    }
    #[cfg(feature = "metal")]
    {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        for (name, golden) in [
            (AFMOE, &AFMOE_GOLDEN),
            (LAGUNA, &LAGUNA_GOLDEN),
            (LAGUNA_SWA, &LAGUNA_SWA_GOLDEN),
        ] {
            let d = load_graph_fixture(name);
            let mut kv = graph_caches(&d);
            common::assert_close(
                &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
                golden,
                common::GRAPH_TOL,
                &format!("{name}: prefill with Metal on"),
            );
            let mut kv = graph_caches(&d);
            let mut out = Vec::new();
            for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
                out = d.forward_token(tok, pos, &mut kv);
            }
            common::assert_close(
                &out,
                golden,
                common::GRAPH_TOL,
                &format!("{name}: decode with Metal on"),
            );
            assert!(
                !d.metal_attn_kv_allocated(),
                "{name}: a gated layer must never reach a fused Metal attention launch"
            );
        }
        // The control: the M.1-shaped file with its gates stripped is
        // uniform, un-windowed and F32, so with the switch on the fused
        // decode launch must take it -- else the assertions above
        // proved only that Metal was off.
        let mut d = load_graph_fixture(LAGUNA);
        for layer in d.layers.iter_mut() {
            layer.attn.output_gate = None;
        }
        let mut kv = graph_caches(&d);
        for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
            let _ = d.forward_token(tok, pos, &mut kv);
        }
        assert!(
            d.metal_attn_kv_allocated(),
            "the ungated control did not reach a fused launch; the Metal switch is not live \
             in this process and the gated assertions above proved nothing"
        );
    }
}
