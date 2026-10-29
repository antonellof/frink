//! The per-architecture tables pinned against llama.cpp and against
//! the architectures the generic path actually serves.

use super::*;

/// The no-renormalise list is keyed on what llama.cpp's GRAPH does,
/// not on what a GGUF says, because for these architectures the
/// GGUF says nothing.
///
/// `expert_weights_norm` is only written by converters that set it.
/// `deepseek.cpp:145` passes `norm_w=false`, and
/// `conversion/deepseek.py`'s `DeepseekModel` never writes the key
/// -- only `DeepseekV2Model` does. So a real `deepseek` checkpoint
/// carries no key at all and frink fell through to its default,
/// renormalising the selected experts' softmax weights where
/// llama.cpp leaves them alone.
///
/// The same mistake made OLMoE emit garbage, which is why that list
/// exists. This pins the membership so a later edit cannot quietly
/// drop a name back into the renormalising default.
#[test]
fn the_architectures_llama_cpp_does_not_renormalise_are_pinned() {
    // Through the facts: `qwen2moe`'s row is in `crate::models`.
    let no_renorm = |a: &str| crate::models::ArchFacts::of(a).moe.no_topk_renormalize;
    for arch in ["deepseek", "olmoe", "qwen2moe"] {
        assert!(
            no_renorm(arch),
            "{arch} passes norm_w=false in llama.cpp and must not be renormalised"
        );
    }
    // `deepseek2` is a DIFFERENT architecture whose converter DOES
    // write the key, so it must not be on this list -- it gets its
    // answer from the file.
    assert!(!no_renorm("deepseek2"));
    assert!(!no_renorm("qwen3moe"));
}

/// Every architecture llama.cpp defaults to SIGMOID gating must be
/// on the list, because for these the GGUF carries no key to say so.
///
/// Each of these reads `LLM_KV_EXPERT_GATING_FUNC` as optional and
/// then sets SIGMOID when it is absent, so a converted checkpoint
/// has nothing in it that would correct frink's softmax default.
/// Same shape as the `deepseek` top-k renormalisation bug, and as
/// `phi3`'s sliding window: the file is silent and the architecture
/// decides.
/// The literal table and its name list are two spellings of one
/// fact; this is what keeps them one.
#[test]
fn the_gating_literal_names_are_the_gating_literal_table() {
    let from_table: Vec<&str> = GATING_LITERAL_ARCHITECTURES
        .iter()
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(from_table, GATING_LITERAL_NAMES);
    // `mimo2.cpp:227` passes the SIGMOID literal, so the key is
    // never read there; a hand-written SOFTMAX key must not turn it.
    assert!(matches!(
        GATING_LITERAL_ARCHITECTURES
            .iter()
            .find(|(n, _)| *n == "mimo2")
            .map(|(_, g)| *g),
        Some(GatingFunction::Sigmoid)
    ));
}

#[test]
fn the_architectures_llama_cpp_defaults_to_sigmoid_gating_are_pinned() {
    for arch in ["afmoe", "deepseek2", "glm4moe", "laguna", "step35"] {
        assert!(
            SIGMOID_GATING_ARCHITECTURES.contains(&arch),
            "{arch} sets SIGMOID when the gating key is absent"
        );
    }
    // Architectures that HARDCODE softmax must stay off it, or the
    // fix becomes the opposite bug: `ernie4-5-moe.cpp:90` and
    // `qwen3moe` both gate with softmax unconditionally.
    for softmax in ["ernie4_5-moe", "qwen3moe", "olmoe", "llama"] {
        assert!(
            !SIGMOID_GATING_ARCHITECTURES.contains(&softmax),
            "{softmax} does not default to sigmoid"
        );
    }
}

/// Every name in every architecture-keyed behaviour table is a name
/// the catalog actually resolves, on the generic-GQA path.
///
/// These five tables are the repo's dominant bug shape in its purest
/// form: five lists of strings that have to agree with a sixth
/// structure (`capability::architecture_catalog`) about what an
/// architecture is called, with nothing checking it. A typo, a
/// hyphen where the GGUF has an underscore, or a name that later
/// moves to a dedicated stack all produce the same thing -- an entry
/// that reads as coverage and can never fire. This repo has shipped
/// exactly that once already, in `unsupported_feature_keys`, keyed
/// on a GGUF spelling no converter writes.
///
/// The generic-path check is the second half and the sharper one: a
/// behaviour flag on an architecture that is `DedicatedOnly` or
/// `Deferred` never reaches this loader, so it is dead text.
///
/// Sabotage to confirm: add `"seedoss"` to any list below.
#[test]
fn every_architecture_keyed_behaviour_table_names_a_real_generic_row() {
    let tables: &[(&str, &[&str])] = &[
        ("SIGMOID_GATING_ARCHITECTURES", SIGMOID_GATING_ARCHITECTURES),
        ("GATING_LITERAL_ARCHITECTURES", GATING_LITERAL_NAMES),
        ("EXPERT_WEIGHTS_SCALE_READERS", EXPERT_WEIGHTS_SCALE_READERS),
        ("EXPERT_WEIGHTS_NORM_READERS", EXPERT_WEIGHTS_NORM_READERS),
        ("YARN_BETA_READERS", YARN_BETA_READERS),
        (
            "NO_TOPK_RENORMALIZE_ARCHITECTURES",
            NO_TOPK_RENORMALIZE_ARCHITECTURES,
        ),
        (
            "PRE_FFN_NORM_IS_POST_ATTENTION_NORM",
            crate::norm_sites::PRE_FFN_NORM_IS_POST_ATTENTION_NORM,
        ),
        ("LEADING_DENSE_KEY_IS_INERT", LEADING_DENSE_KEY_IS_INERT),
        (
            "QK_NORM_AFTER_ROPE_ARCHITECTURES",
            QK_NORM_AFTER_ROPE_ARCHITECTURES,
        ),
    ];
    for (table, names) in tables {
        for arch in *names {
            let profile = crate::capability::resolve_profile(arch).unwrap_or_else(|| {
                panic!("{table} names `{arch}`, which the catalog does not have")
            });
            if matches!(profile.path, crate::capability::ArchPath::GenericGqa { .. }) {
                continue;
            }
            // Not a generic row, so the entry cannot fire HERE.
            // That is allowed only when something else is named as
            // applying the behaviour instead. An unexplained dead
            // entry still fails, which is the whole point.
            let owner = DEDICATED_OWNS_ITS_BEHAVIOUR
                .iter()
                .find(|(name, _)| name == arch)
                .map(|(_, owner)| *owner);
            assert!(
                owner.is_some(),
                "{table} names `{arch}`, which resolves to {:?} and never reaches this \
                 loader, so the entry cannot fire. Either drop it, or add it to \
                 DEDICATED_OWNS_ITS_BEHAVIOUR naming what applies the behaviour instead",
                profile.path
            );
        }
    }
}

/// The three tables that describe how a layer is BUILT, rather than
/// how it is routed, only carry architectures that are audited.
///
/// The distinction matters and is not pedantry. A routing default
/// (`SIGMOID_GATING_ARCHITECTURES`, `NO_TOPK_RENORMALIZE_ARCHITECTURES`)
/// is allowed to name an architecture that still refuses: it is
/// written down ahead of time so a later admission inherits the
/// right answer, and the tables say so. But the three below change
/// which TENSOR a layer reads and in what order -- and each was
/// added for exactly one architecture, whose fixture is the only
/// thing proving the change is right. A fourth name appearing here
/// without evidence would be a claim about a graph nobody read,
/// carried by a list whose doc comment cites two.
#[test]
fn the_layer_shape_tables_only_name_audited_architectures() {
    for (table, names) in [
        (
            "PRE_FFN_NORM_IS_POST_ATTENTION_NORM",
            crate::norm_sites::PRE_FFN_NORM_IS_POST_ATTENTION_NORM,
        ),
        ("LEADING_DENSE_KEY_IS_INERT", LEADING_DENSE_KEY_IS_INERT),
        (
            "QK_NORM_AFTER_ROPE_ARCHITECTURES",
            QK_NORM_AFTER_ROPE_ARCHITECTURES,
        ),
    ] {
        for arch in names {
            assert!(
                crate::capability::is_audited_generic(arch),
                "{table} names `{arch}`, which is not in AUDITED_GENERIC_GQA. Either it \
                 has a fixture proving the change is right -- audit it -- or the entry \
                 is a guess about a graph"
            );
        }
    }
}

#[test]
fn qwen2moe_disables_topk_renorm() {
    assert!(
        crate::models::ArchFacts::of("qwen2moe")
            .moe
            .no_topk_renormalize,
        "qwen2moe must have norm_topk_prob=false (llama.cpp build_moe_ffn norm_w=false)"
    );
}
