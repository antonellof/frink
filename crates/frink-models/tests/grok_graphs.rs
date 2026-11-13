//! Grok-1, checked against llama.cpp itself.
//!
//! `grok` was triaged NEW CODE, and its verdict named the MiniCPM
//! shape: `src/models/grok.cpp:5-12` seeds seven hyper-parameters
//! BEFORE `:14-27` let the file override them, so a Grok-1 export that
//! declares none of them is still scaled by all of them and a
//! key-presence gate sees an ordinary file. That is
//! `scalar_multipliers::MultiplierDefaults::Grok`, and it is why this
//! suite has TWO goldens: `GROK_GOLDEN` is for a file declaring NO key
//! -- the only fixture shape that can tell the hook from its absence --
//! and `GROK_DECLARED_GOLDEN` is for a file declaring every key at a
//! value nowhere near its default, which pins that the file still wins.
//! A hook merged the wrong way round agrees with llama.cpp on exactly
//! the files that prove it exists.
//!
//! What the verdict called new code turned out to be three seams that
//! had landed the day before, each extended by one column:
//!
//! - **The defaults hook**, plus two things MiniCPM did not need:
//!   `grok.cpp:211` MULTIPLIES by `logit_scale` (`LogitScaleUse::AsIs`),
//!   and the attention scale comes from a fifth key,
//!   `{arch}.attention.output_scale` (`AttentionScaleKey::OutputScale`),
//!   which the graph applies INSIDE its tanh softcap with
//!   `kq_scale = 1.0f` (`:137`, `llama-graph.cpp:2572-2582`) -- which is
//!   arithmetically `ModelConfig::attention_scale` plus the existing
//!   `attn_logit_softcap`, so no new attention code.
//! - **The norm-site table** (`crate::norm_sites`): `attn_output_norm`
//!   is Grok's POST-attention norm (`:143-148`), the same tensor name
//!   `dbrx` uses for its PRE-FFN norm, and `layer_output_norm` its
//!   post-FFN norm (`:75-78`, `:185-190`).
//! - **The softcap gate**: `ScalingFacts::logit_softcap`,
//!   because `conversion/grok.py:34` writes `attn_logit_softcapping` for
//!   every export and the generic refusal would have stopped all of them.
//!
//! Two keys Grok reads and llama.cpp never applies --
//! `router_logit_softcapping` (`:10,:20`) and
//! `attention.temperature_length` (`:23`; no other reference to either
//! field under `src/`, measured) -- are neither applied nor refused, and
//! the declared file carries both to prove it. Grok-2's parallel dense
//! FFN (`:171-184`) is `crate::parallel_dense_ffn`, served since
//! `arctic` closed on the same seam, and `grok_dense_ffn_tiny.gguf` is
//! checked in `tests/parallel_dense_ffn_graphs.rs`.
//!
//! **Tolerance.** This is a GeGLU row (`:165`, `LLM_FFN_GELU`), so it
//! is compared at [`GELU_TABLE_TOL`]: llama.cpp's CPU GELU is a 65536
//! entry f16 lookup table and frink's is exact. Measured rather than
//! assumed: with frink's GELU temporarily made to emulate the table
//! (round the input to f16, look up, round the output to f16) both
//! files agree with libllama to **1.0e-7 / 1.5e-7** (KL 7e-16 / 2e-15),
//! so the whole of the 6.3e-5 / 4.0e-5 gap below is the table.
//!
//! **Where the numbers come from.** Both goldens were produced by
//! running llama.cpp's own graph over the fixtures through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! Regenerating (all three files together):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_grok_fixture.py \
//!     crates/frink-models/tests/fixtures/grok_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_grok_fixture.py \
//!     crates/frink-models/tests/fixtures/grok_declared_tiny.gguf --declared
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_grok_fixture.py \
//!     crates/frink-models/tests/fixtures/grok_dense_ffn_tiny.gguf --dense-ffn
//! /tmp/ref_logits crates/frink-models/tests/fixtures/grok_tiny.gguf 3 7 11 19 23 5
//! /tmp/ref_logits crates/frink-models/tests/fixtures/grok_declared_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match_within, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GELU_TABLE_TOL, GRAPH_PROMPT,
};
use frink_gguf::TensorSource;
use frink_models::{FfnActivation, ModelConfig, NormOp, RopeLayout};

const GROK: &str = "grok";

/// The file declaring NO key, scaled by every `grok.cpp:5-12` default.
const GROK_GOLDEN: [f32; 48] = [
    -0.11251373,
    -0.026473513,
    -0.055697076,
    -0.19734381,
    -0.0529336,
    -0.054044608,
    0.06723128,
    -0.13850914,
    0.11881044,
    0.070129976,
    -0.2639037,
    0.26145476,
    0.10589446,
    -0.1721888,
    -0.041680638,
    0.042815816,
    -0.10318534,
    -0.038024586,
    0.059208687,
    -0.03599476,
    -0.012259253,
    0.012435135,
    -0.06356341,
    -0.130161,
    -0.050142236,
    -0.11604546,
    -0.110850506,
    0.16574205,
    -0.12435809,
    0.0034513173,
    -0.11575936,
    -0.0193332,
    -0.2006187,
    -0.12878717,
    0.0034563676,
    0.13198729,
    0.09669709,
    0.031157292,
    0.09666561,
    0.162525,
    -0.057125907,
    -0.1393247,
    -0.13197218,
    -0.15255743,
    -0.03449362,
    -0.011501383,
    -0.03715561,
    0.11378722,
];

/// The same weights with every key declared: `logit_scale = 0.9`,
/// `embedding_scale = 40`, `attention.output_scale = 0.2`,
/// `attn_logit_softcapping = 10`, `final_logit_softcapping = 5`, plus
/// the two keys llama.cpp reads and never applies.
const GROK_DECLARED_GOLDEN: [f32; 48] = [
    -0.20590317,
    -0.05821289,
    -0.08407988,
    -0.48076674,
    0.041526847,
    0.016847923,
    0.25774103,
    -0.15762882,
    -0.011562021,
    0.024056517,
    -0.41214567,
    -0.11709563,
    -0.10771391,
    -0.042149857,
    0.05325322,
    0.08525145,
    -0.21518523,
    -0.123669714,
    -0.11176306,
    -0.18987104,
    0.24006866,
    0.020933907,
    -0.13480619,
    0.0773401,
    -0.15065041,
    -0.24687304,
    0.28445444,
    -0.065707356,
    0.09487598,
    -0.07064034,
    0.01691522,
    -0.01752319,
    -0.3981459,
    0.17475381,
    0.020597251,
    0.073743105,
    0.28815252,
    -0.1600543,
    0.009261235,
    -0.03378719,
    0.17028067,
    -0.24544583,
    -0.08273651,
    -0.13020343,
    0.1967583,
    0.012938257,
    -0.109115504,
    0.29917872,
];

/// The row itself, on all three forward paths: the file declaring
/// nothing, scaled by every default.
#[test]
fn grok_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(GROK, &GROK_GOLDEN, GELU_TABLE_TOL);
}

/// The file declaring every key, on all three paths: the file wins over
/// the defaults.
#[test]
fn a_grok_file_declaring_every_key_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within("grok_declared", &GROK_DECLARED_GOLDEN, GELU_TABLE_TOL);
}

/// The numbers in the report, so they can be regenerated rather than
/// trusted. Run with `--nocapture` to see them.
#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (GROK, &GROK_GOLDEN),
        ("grok_declared", &GROK_DECLARED_GOLDEN),
    ] {
        let d = load_graph_fixture(name);
        let mut kv = graph_caches(&d);
        let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
        let kl = kl_vs_golden(&got, golden);
        let worst = worst_vs(&got, golden);
        println!("| `{name}` | {kl:.2e} | {worst:.2e} |");
        assert!(kl < 1e-7, "{name}: KL {kl}");
    }
}

/// A file declaring nothing resolves to `grok.cpp:5-12`'s seeds, and
/// nothing else the graph does not do.
///
/// `attention_scale` is `Some` only because this fixture's head width
/// is 6: at Grok-1's real 128 the default `1/sqrt(128)` restates the
/// kernels' own scale and resolves to `None`
/// (`scalar_multipliers::tests`). Here it must survive, or the next
/// test could not sabotage it.
#[test]
fn a_grok_file_declaring_nothing_is_scaled_by_every_default() {
    let d = load_graph_fixture(GROK);
    assert_eq!(d.config.embedding_scale, Some(78.383_67));
    assert_eq!(
        d.config.logit_multiplier,
        Some(0.577_350_3),
        "grok.cpp:211 multiplies; the reciprocal would be 1.732"
    );
    assert_eq!(d.config.attention_scale, Some(0.088_388_35));
    assert_eq!(d.config.attn_logit_softcap, Some(30.0));
    assert_eq!(d.config.final_logit_softcap, None, "grok.cpp:12: 0.0, off");
    assert_eq!(
        d.config.residual_scale, None,
        "no residual multiplier in the graph"
    );
    assert_eq!(d.config.clamp_kqv, None);
    assert_eq!(d.config.ffn_activation, FfnActivation::Gelu, "grok.cpp:165");
    assert!(
        d.config.moe.norm_topk_prob,
        "build_moe_ffn(..., true, ...) at :165"
    );
    assert_eq!(d.config.rope_layout, RopeLayout::Neox);
    assert_eq!(d.config.n_kv_heads, 2, "the fixture exercises GQA");
}

/// The file's declarations win over every default, key by key.
#[test]
fn a_grok_file_declaring_its_keys_overrides_every_default() {
    let d = load_graph_fixture("grok_declared");
    assert_eq!(d.config.logit_multiplier, Some(0.9));
    assert_eq!(d.config.embedding_scale, Some(40.0));
    assert_eq!(d.config.attention_scale, Some(0.2));
    assert_eq!(d.config.attn_logit_softcap, Some(10.0));
    assert_eq!(d.config.final_logit_softcap, Some(5.0));

    // The two keys llama.cpp reads and never applies are in the file
    // and the file loaded: neither applied, neither refused.
    let file =
        frink_gguf::GgufFile::open(graph_fixture_path("grok_declared")).expect("fixture opens");
    assert_eq!(
        file.metadata_f32("grok.router_logit_softcapping"),
        Some(30.0),
        "the fixture must carry the dead key for this to prove anything"
    );
    assert_eq!(
        file.metadata_u64("grok.attention.temperature_length"),
        Some(4096)
    );
}

/// Each default, sabotaged on its own, diverges from llama.cpp.
///
/// The margin per sabotage is what makes `GROK_GOLDEN` evidence FOR the
/// defaults rather than merely consistent with them. Each is the
/// substitution a wrong reading would make: the reciprocal of the logit
/// scale (Granite's direction), no embedding multiplier, the kernels'
/// own attention scale, and no softcap.
#[test]
fn sabotaging_any_one_default_diverges_from_llama_cpp() {
    type Sabotage = fn(&mut ModelConfig);
    let cases: [(&str, Sabotage); 4] = [
        ("logit_scale inverted", |c| {
            c.logit_multiplier = Some(1.0 / 0.577_350_3)
        }),
        ("embedding_scale dropped", |c| c.embedding_scale = None),
        ("attention.output_scale dropped", |c| {
            c.attention_scale = None
        }),
        ("attn softcap dropped", |c| c.attn_logit_softcap = None),
    ];
    for (what, sabotage) in cases {
        let mut d = load_graph_fixture(GROK);
        sabotage(&mut d.config);
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &GROK_GOLDEN,
        );
        println!("{what}: worst {worst:.3e}");
        assert!(
            worst > 1e-2,
            "{what} moved the output by only {worst}; the fixture cannot see the default it \
             exists to pin"
        );
    }
}

/// The two post-norms are Grok's, under Grok's names, and each is
/// load-bearing.
///
/// `attn_output_norm` is the same tensor name `dbrx` stores its pre-FFN
/// norm under; here it must land in the POST-attention slot
/// (`grok.cpp:143-148`) and `ffn_norm` stay the pre-FFN norm (`:152`).
/// Dropping either post-norm diverges, so the table row is not
/// decoration.
#[test]
fn attn_output_norm_is_the_post_attention_norm_and_both_post_norms_are_load_bearing() {
    let d = load_graph_fixture(GROK);
    for (il, layer) in d.layers.iter().enumerate() {
        assert!(
            matches!(layer.attn.norm_weight, NormOp::Rms(_)),
            "blk.{il}: attn_norm"
        );
        assert!(
            matches!(layer.moe.norm_weight, NormOp::Rms(_)),
            "blk.{il}: ffn_norm is its own tensor (grok.cpp:64)"
        );
        assert!(
            layer.attn.post_attn_norm.is_some(),
            "blk.{il}: attn_output_norm must land in the post-attention slot"
        );
        assert!(
            layer.attn.post_ffn_norm.is_some(),
            "blk.{il}: layer_output_norm must land in the post-FFN slot"
        );
    }
    for site in ["post_attn", "post_ffn"] {
        let mut d = load_graph_fixture(GROK);
        for layer in d.layers.iter_mut() {
            match site {
                "post_attn" => layer.attn.post_attn_norm = None,
                _ => layer.attn.post_ffn_norm = None,
            }
        }
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &GROK_GOLDEN,
        );
        println!("{site} norm dropped: worst {worst:.3e}");
        assert!(
            worst > 1e-2,
            "dropping the {site} norm moved the output by only {worst}"
        );
    }
}

/// Grok's RoPE is the split-half variant, and the fixture can see the
/// other one.
#[test]
fn grok_ropes_neox_and_the_fixture_can_see_the_other_variant() {
    let mut d = load_graph_fixture(GROK);
    d.config.rope_layout = RopeLayout::Norm;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &GROK_GOLDEN,
    );
    assert!(
        worst > 1e-3,
        "rotating the NORM pairs moved the output by only {worst}; the attention in this \
         fixture is too flat to see a positional bug"
    );
}

// The Grok-2 shape, `grok_dense_ffn_tiny.gguf`, was refused from here
// until 2026-09-12; it is served now and checked against its own
// libllama golden in `tests/parallel_dense_ffn_graphs.rs`.
