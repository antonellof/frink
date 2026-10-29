//! The POST-NORM-ONLY family, checked against llama.cpp itself.
//!
//! `olmo2` and `exaone4` were both triaged NEW CODE in `capability.rs`
//! for the same blocker: neither has an `attn_norm` or an `ffn_norm`
//! tensor, both sublayers read the RAW residual, and each branch's
//! output is normed before its residual add.
//!
//! ```text
//! ffn_inp = x       + post_attn_norm(attn(x))
//! out     = ffn_inp + post_ffn_norm(ffn(ffn_inp))
//! ```
//!
//! **They are one topology, not two that look alike**, and the claim is
//! a line-by-line reading of both graphs rather than a family
//! resemblance -- `crate::norm`'s module docs carry the table. So
//! there is ONE implementation, `NormOp`, and this suite is what
//! proves the shared body is right for both rows rather than right for
//! one and plausible for the other.
//!
//! What the two rows do NOT share is their QK-norm style, which is why
//! they still need a fixture each:
//!
//! | | `olmo2` | `exaone4` |
//! |---|---|---|
//! | QK-norm width | `{n_embd}` / `{n_head_kv * n_embd_head}` (olmo2.cpp:45-46) | `{n_embd_head_k}` (exaone4.cpp:61-62) |
//! | applied to | the 2-D projection, BEFORE `ggml_reshape_3d` (:106-116) | the 3-D tensor `build_qkv` already reshaped (:124-128) |
//! | frink style | `QkNormStyle::WholeVector` | `QkNormStyle::PerHead` |
//!
//! frink derives that style from the weight LENGTH at load
//! (`loader.rs`'s `refined_qk_norm`), so a fixture per style is the only
//! way to check the derivation against a real file of each shape.
//!
//! # What each fixture deliberately does NOT carry
//!
//! Both llama.cpp graphs have a second, per-layer RoPE behaviour that
//! frink cannot express, and in both cases llama.cpp switches it on
//! with NO GGUF key:
//!
//! * `exaone4.cpp:4-9` turns SWA on inside `if (n_layer() == 64)`, and
//!   :116 then ropes ONLY the sliding layers -- so EXAONE-4 32B gives
//!   its full-attention layers no rotation at all. Refused by name on
//!   `block_count == 64` in `loader.rs`, the `baichuan` precedent.
//! * `olmo2.cpp:120-134` ropes the sliding layers with the model's
//!   scaling switched off while :136-146 ropes the full ones with it on.
//!   Refused by name when a file carries BOTH a window and a
//!   `rope.scaling.type`.
//!
//! Both refusals have their own reachability tests in `loader.rs`
//! (`exaone4_32b_is_refused_because_its_full_attention_layers_get_no_rope`,
//! `olmo2_is_refused_only_when_it_has_a_window_and_a_rope_scaling_together`),
//! each with the negative half that proves the gate discriminates rather
//! than refusing the architecture outright. These two fixtures sit on
//! the other side of both lines, which is what makes them evidence about
//! the topology instead of evidence about a gate.
//!
//! `olmo` (OLMo-1) is a THIRD shape and is not in this suite: it norms
//! before both sublayers, so it is pre-norm like llama, and what it
//! lacks is a norm FUNCTION (non-parametric LayerNorm, `olmo.cpp:65-67`,
//! `:104-106`, all three `build_norm` calls with a NULL weight) rather
//! than the residual shape. It stays refused, and its triage row says so.
//!
//! **Where the numbers come from.** Each `GOLDEN` array was produced by
//! running llama.cpp's own graph for that architecture over the same
//! fixture file, through `scripts/gptoss_reference_logits.cpp` linked
//! against a real `libllama` built from `.scratch/llama.cpp`. Not by
//! re-reading a spec, and not by frink checking itself.
//!
//! Measured against that reference over `GRAPH_PROMPT`, both fixtures
//! being F32 so no `vec_dot_type` question arises:
//!
//! | arch | KL(llama.cpp \|\| frink) | max abs logit delta | top-1 |
//! |---|---|---|---|
//! | `olmo2` | 3.01e-14 | 6.74e-07 | agrees |
//! | `exaone4` | 1.24e-14 | 3.91e-07 | agrees |
//!
//! That is float32 accumulation noise -- ggml blocks its matmuls and
//! frink does not -- and it is six orders of magnitude under every
//! sabotage below, each of which is required to move the logits by more
//! than 1e-2.
//!
//! Regenerating (both halves must be redone together if a fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_olmo2_fixture.py \
//!     crates/frink-models/tests/fixtures/olmo2_tiny.gguf
//! clang++ -std=c++17 -O2 scripts/gptoss_reference_logits.cpp \
//!     -I$LLAMA/include -I$LLAMA/ggml/include -L$BUILD/bin -lllama \
//!     -Wl,-rpath,$BUILD/bin -o /tmp/ref_logits
//! /tmp/ref_logits crates/frink-models/tests/fixtures/olmo2_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, load_graph_fixture, worst_vs,
    GRAPH_PROMPT,
};
use frink_models::capability::QkNormStyle;
use frink_models::norm::NormOp;
use frink_models::{ModelConfig, RopeLayout};

/// Both rows, named once so no test below can quietly cover one and
/// claim the family.
const ROWS: [&str; 2] = ["olmo2", "exaone4"];

// --- olmo2 ---------------------------------------------------------

const OLMO2_GOLDEN: [f32; 48] = [
    0.08746515,
    0.17119634,
    -0.25714567,
    0.6422479,
    -0.3069752,
    0.2974551,
    0.08578196,
    -0.011812593,
    0.42666468,
    0.34420896,
    -0.16842516,
    0.08995975,
    -0.41947117,
    -0.019727454,
    -0.10010234,
    -0.14203572,
    0.19076586,
    -0.023054836,
    -0.31898016,
    0.024437977,
    0.2810498,
    -0.27231476,
    0.15113845,
    0.1411128,
    -0.11158176,
    -0.21863683,
    0.34922016,
    -0.31178498,
    -0.1140763,
    0.14966111,
    -0.2308585,
    -0.17571855,
    0.10948058,
    0.18218815,
    0.28228983,
    0.25900576,
    0.30450112,
    0.0922729,
    -0.3080338,
    -0.10226583,
    -0.081689626,
    -0.14788833,
    0.022490028,
    0.0284947,
    0.22814226,
    0.33456734,
    0.008091899,
    -0.26187766,
];

// --- exaone4 -------------------------------------------------------

const EXAONE4_GOLDEN: [f32; 48] = [
    0.22023313,
    0.11032572,
    0.22996083,
    -0.028935976,
    -0.032483645,
    -0.2924301,
    -0.13335375,
    0.5377056,
    0.1966051,
    -0.0069604963,
    0.06706336,
    -0.24097686,
    -0.2501576,
    -0.3881526,
    -0.194803,
    0.24884875,
    -0.22297607,
    -0.11595694,
    0.39182818,
    0.15815882,
    -0.50697553,
    0.32023728,
    0.12748046,
    0.7513782,
    -0.008413196,
    -0.58900523,
    0.45524126,
    -0.050441377,
    0.41747662,
    -0.22276221,
    -0.0007682629,
    0.37388653,
    -0.02264306,
    0.083368115,
    0.4577073,
    0.22389793,
    0.03977763,
    -0.08297233,
    -0.08565287,
    0.11946838,
    -0.008339636,
    -0.16977824,
    0.0016572177,
    0.27790904,
    -0.0969176,
    0.17013024,
    0.44302016,
    -0.09636323,
];

fn golden(name: &str) -> &'static [f32] {
    match name {
        "olmo2" => &OLMO2_GOLDEN,
        "exaone4" => &EXAONE4_GOLDEN,
        other => panic!("no golden logits for `{other}`"),
    }
}

#[test]
fn olmo2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("olmo2", &OLMO2_GOLDEN);
}

#[test]
fn exaone4_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("exaone4", &EXAONE4_GOLDEN);
}

/// The topology itself: neither row has a pre-norm in either slot.
///
/// Structural, not numeric, because a decoder that loaded an
/// `attn_norm` from somewhere and applied it would still have to agree
/// with llama.cpp -- and the point of these two rows is that there is
/// nothing to load. If this assertion ever fails, `loader.rs` found a
/// tensor these architectures do not have.
#[test]
fn neither_row_has_a_pre_attention_norm_or_a_pre_ffn_norm() {
    for name in ROWS {
        let d = load_graph_fixture(name);
        assert_eq!(d.layers.len(), 2, "{name}: layer count");
        for (il, layer) in d.layers.iter().enumerate() {
            assert_eq!(
                layer.attn.norm_weight,
                NormOp::None,
                "{name} blk.{il}: Q/K/V must be projected off the raw residual"
            );
            assert_eq!(
                layer.moe.norm_weight,
                NormOp::None,
                "{name} blk.{il}: the FFN must read the raw post-attention residual"
            );
            // And both post-norms ARE there, non-zero, because they are
            // the only norms in the layer.
            let post_attn = layer
                .attn
                .post_attn_norm
                .as_ref()
                .unwrap_or_else(|| panic!("{name} blk.{il}: attn_post_norm must be loaded"));
            let post_ffn = layer
                .attn
                .post_ffn_norm
                .as_ref()
                .unwrap_or_else(|| panic!("{name} blk.{il}: ffn_post_norm must be loaded"));
            assert!(post_attn.iter().any(|w| *w != 0.0), "{name} blk.{il}");
            assert!(post_ffn.iter().any(|w| *w != 0.0), "{name} blk.{il}");
        }
    }
}

/// Putting an all-ones RMSNorm back in either pre-norm slot diverges.
///
/// This is the sabotage that matters, because it is the shortcut
/// somebody will reach for: "these architectures just have identity
/// pre-norms, load a vector of ones". An RMSNorm with unit weights is
/// not the identity -- it still divides by the RMS of the residual --
/// and this test measures how far from the truth that lands. Without
/// it, `NormOp::None` and `NormOp::Rms(vec![1.0; n])` would be
/// indistinguishable to this suite and the enum would be decoration.
#[test]
fn restoring_an_all_ones_pre_norm_in_either_slot_diverges_from_llama_cpp() {
    for name in ROWS {
        for slot in ["attn", "ffn"] {
            let mut d = load_graph_fixture(name);
            let hidden = d.config.hidden_dim;
            for layer in d.layers.iter_mut() {
                if slot == "attn" {
                    layer.attn.norm_weight = NormOp::Rms(vec![1.0; hidden]);
                } else {
                    layer.moe.norm_weight = NormOp::Rms(vec![1.0; hidden]);
                }
            }
            let mut kv = graph_caches(&d);
            let worst = worst_vs(
                &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
                golden(name),
            );
            assert!(
                worst > 1e-2,
                "{name}: normalising before the {slot} branch moved the output by only \
                 {worst}; this fixture cannot tell the post-norm-only topology from a \
                 pre-norm one"
            );
        }
    }
}

/// Dropping either post-norm diverges.
///
/// `post_attn_norm` and `post_ffn_norm` are two of the eight model
/// features a copied decode path in this repo has silently lost. On
/// these two architectures they are the ONLY norms inside a layer, so a
/// suite that could not see them missing would be checking almost
/// nothing.
#[test]
fn dropping_either_post_norm_diverges_from_llama_cpp() {
    for name in ROWS {
        for which in ["attn", "ffn"] {
            let mut d = load_graph_fixture(name);
            for layer in d.layers.iter_mut() {
                if which == "attn" {
                    layer.attn.post_attn_norm = None;
                } else {
                    layer.attn.post_ffn_norm = None;
                }
            }
            let mut kv = graph_caches(&d);
            let worst = worst_vs(
                &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
                golden(name),
            );
            assert!(
                worst > 1e-2,
                "{name}: dropping post_{which}_norm moved the output by only {worst}; \
                 the fixture cannot see this slot"
            );
        }
    }
}

/// The two rows resolve to DIFFERENT QK-norm styles, off the weight
/// length in the file.
///
/// `olmo2.cpp:45-46` sizes its norms `{n_embd}` and
/// `{n_head_kv * n_embd_head}` and applies them before the reshape;
/// `exaone4.cpp:61-62` sizes both `{n_embd_head_k}` and applies them
/// after `build_qkv` has reshaped. Same topology, opposite style, and
/// nothing in the GGUF says which -- frink derives it from the length,
/// so these are the two lengths that derivation has to tell apart.
#[test]
fn the_two_rows_derive_opposite_qk_norm_styles_from_their_weight_lengths() {
    let olmo2 = load_graph_fixture("olmo2");
    assert_eq!(olmo2.config.qk_norm_style, QkNormStyle::WholeVector);
    assert_eq!(olmo2.config.head_dim, 6);
    assert_eq!(olmo2.config.n_heads, 4);
    let q_norm = olmo2.layers[0]
        .attn
        .q_norm
        .as_ref()
        .expect("olmo2 has attn_q_norm");
    assert_eq!(
        q_norm.len(),
        24,
        "olmo2's Q norm is n_heads * head_dim wide"
    );

    let exaone4 = load_graph_fixture("exaone4");
    assert_eq!(exaone4.config.qk_norm_style, QkNormStyle::PerHead);
    assert_eq!(exaone4.config.head_dim, 8);
    assert_eq!(exaone4.config.n_heads, 4);
    let q_norm = exaone4.layers[0]
        .attn
        .q_norm
        .as_ref()
        .expect("exaone4 has attn_q_norm");
    assert_eq!(q_norm.len(), 8, "exaone4's Q norm is head_dim wide");
}

/// The half that makes the assertion above worth its runtime: each
/// fixture can SEE its QK-norm, and can see it on the wrong side of
/// RoPE.
///
/// The style itself cannot be flipped and measured -- a whole-vector
/// weight is 24 long where a per-head one is 6, so `rms_norm_per_head`
/// would not be applying the wrong style, it would be reading off the
/// end -- which is why the style is pinned by its length above and this
/// test covers the two things that CAN silently go wrong instead:
///
/// * the norms being dropped entirely (both files draw them centred
///   near 1.5, so a dropped norm is far from the truth);
/// * the norms landing AFTER RoPE, which is `maincoder` and
///   `hunyuan-moe`'s order and NOT these two (`olmo2.cpp:106-112`
///   precede :124-146, `exaone4.cpp:127-128` precede :132-138). No GGUF
///   key distinguishes the two orders; llama.cpp writes it into each
///   hand-written graph.
#[test]
fn dropping_the_qk_norms_or_moving_them_after_rope_diverges_from_llama_cpp() {
    for name in ROWS {
        let mut dropped = load_graph_fixture(name);
        for layer in dropped.layers.iter_mut() {
            layer.attn.q_norm = None;
            layer.attn.k_norm = None;
        }
        let mut kv = graph_caches(&dropped);
        let worst = worst_vs(
            &dropped.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            golden(name),
        );
        assert!(
            worst > 1e-2,
            "{name}: dropping the QK norms moved the output by only {worst}; \
             the fixture cannot see them"
        );

        let mut reordered = load_graph_fixture(name);
        assert!(
            !reordered.qk_norm_after_rope,
            "{name}: both graphs norm BEFORE rotating"
        );
        reordered.qk_norm_after_rope = true;
        let mut kv = graph_caches(&reordered);
        let worst = worst_vs(
            &reordered.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            golden(name),
        );
        assert!(
            worst > 1e-2,
            "{name}: norming after RoPE instead of before moved the output by only \
             {worst}; the fixture cannot see the order"
        );
    }
}

/// Both rows are NEOX, and both fixtures can see a flip.
///
/// `llama_model_rope_type` puts `LLM_ARCH_OLMO2` and `LLM_ARCH_EXAONE4`
/// in the `LLAMA_ROPE_TYPE_NEOX` group. Rotating the wrong pairs of
/// every Q/K head is the defect that produced the Llama-3.1-8B
/// wrong-output bug.
#[test]
fn both_rows_are_neox_and_rotating_the_wrong_pairs_diverges_from_llama_cpp() {
    for name in ROWS {
        let path = graph_fixture_path(name);
        let file = frink_gguf::GgufFile::open(&path).expect("opens");
        let mut config = ModelConfig::from_gguf(&file).expect("parses");
        assert_eq!(config.rope_layout, RopeLayout::Neox, "{name}: rope layout");
        config.rope_layout = RopeLayout::Norm;
        let d = common::load_decoder(&path, config);
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            golden(name),
        );
        assert!(
            worst > 1e-2,
            "{name}: flipping the RoPE variant moved the output by only {worst}; \
             the fixture cannot see this"
        );
    }
}

/// The rest of the per-architecture checklist, as ABSENCES.
///
/// Both graphs pass a literal `1.0f/sqrtf(float(n_embd_head))` to
/// `build_attn` (olmo2.cpp:154, exaone4.cpp:145) and neither reads
/// `LLM_KV_ATTENTION_SCALE`, so `attention_scale` must stay unset and
/// let the kernels' own scale stand. Neither fixture declares a window,
/// which is what keeps both of them on the implemented side of the two
/// per-layer RoPE refusals; and neither row is MoE, so no gating or
/// renormalisation question arises. Each of these is silent when wrong:
/// the model still runs.
#[test]
fn neither_row_overrides_the_attention_scale_or_declares_a_window() {
    for name in ROWS {
        let d = load_graph_fixture(name);
        assert!(
            d.config.attention_scale.is_none(),
            "{name}: attention_scale must stay unset"
        );
        assert_eq!(d.config.sliding_window, None, "{name}: no window");
        assert_eq!(d.config.layer_sliding_window(0), None, "{name}: layer 0");
        assert_eq!(d.config.layer_sliding_window(1), None, "{name}: layer 1");
        assert!(d.config.moe.n_experts <= 1, "{name}: neither row is MoE");
        assert!(
            d.gpt_oss.is_none(),
            "{name}: no attention sinks, no router bias, no SwiGLU clamp"
        );
    }
}

/// Both are on the audited generic path, and by this suite.
///
/// A name in `AUDITED_GENERIC_GQA` with no evidence defeats the whole
/// point of the list, so the list and the evidence are asserted in the
/// same place.
#[test]
fn both_rows_are_admitted_to_the_audited_generic_path() {
    for name in ROWS {
        assert!(
            frink_models::capability::is_audited_generic(name),
            "{name} must be audited: this file is its evidence"
        );
        assert!(
            frink_models::capability::is_post_norm_only(name),
            "{name} must be on the post-norm-only list"
        );
    }
    // And the list is exactly these two: a third name would be a third
    // `*.cpp` somebody read, and would need a row here.
    assert_eq!(
        frink_models::capability::POST_NORM_ONLY_ARCHITECTURES,
        &ROWS[..]
    );
    // `olmo` is OLMo-1 and a different shape: pre-norm, with a
    // non-parametric LayerNorm. It is audited now too
    // (`tests/olmo_graphs.rs`), on a different variant of the same enum,
    // and what still has to hold is that it is not on THIS list -- a
    // decoder that read OLMo-1 as post-norm-only would drop both its
    // norms and answer fluently.
    assert!(!frink_models::capability::is_post_norm_only("olmo"));
    assert!(frink_models::capability::uses_non_parametric_layer_norm(
        "olmo"
    ));
}
