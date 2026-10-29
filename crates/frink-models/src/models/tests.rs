//! The registry guards from `docs/plans/arch-registry-migration.md`.

use super::facts::tests::{assert_same, behaviour_table_names};
use super::{ArchFacts, REGISTRY};

fn registry_names() -> impl Iterator<Item = &'static str> {
    REGISTRY.iter().flat_map(|def| def.names.iter().copied())
}

/// A name in two definitions would answer from whichever came first; a
/// name the catalog does not know would never be read.
#[test]
fn every_registry_name_resolves_exactly_once() {
    for name in registry_names() {
        let defs = REGISTRY.iter().filter(|d| d.names.contains(&name)).count();
        assert_eq!(defs, 1, "`{name}` is named by {defs} registry entries");
        assert!(
            crate::capability::resolve_profile(name).is_some(),
            "`{name}` is in the registry but not in the catalog"
        );
    }
}

/// `AUDITED_GENERIC_GQA` still owns the audited flag (step 10 derives
/// it from here); until then the two must say the same thing.
#[test]
fn a_registry_name_is_audited_if_and_only_if_it_has_evidence() {
    for def in REGISTRY {
        for name in def.names {
            assert_eq!(
                def.evidence.is_some(),
                crate::capability::is_audited_generic(name),
                "`{name}`: evidence {:?} against AUDITED_GENERIC_GQA",
                def.evidence
            );
        }
    }
}

/// A definition states every row for its names, so no behaviour table
/// may still carry one: the legacy tables answer [`ArchFacts::DEFAULT`]
/// for every registry name. The catalog and `AUDITED_GENERIC_GQA` are
/// exempt until steps 10 and 11 move them, and so is what they decide
/// (the profile, the audited flag, the family's embedding scale).
#[test]
fn no_registry_name_remains_in_a_legacy_table() {
    let tabled = behaviour_table_names();
    for name in registry_names() {
        assert!(
            !tabled.contains(&name),
            "`{name}` is in the registry and still has a legacy table row"
        );
        let legacy = ArchFacts::from_tables(name);
        let mut want = ArchFacts::DEFAULT;
        want.profile = legacy.profile;
        want.audited = legacy.audited;
        want.scaling.embeddings_scaled_by_sqrt_n_embd =
            legacy.scaling.embeddings_scaled_by_sqrt_n_embd;
        assert_same(name, legacy, want);
    }
}

/// The catalog fills these in [`ArchFacts::of`]; a definition stating
/// them would be overwritten without a word.
#[test]
fn a_definition_leaves_the_catalog_fields_to_the_catalog() {
    for def in REGISTRY {
        assert!(def.facts.profile.is_none(), "{:?}", def.names);
        assert!(!def.facts.audited, "{:?}", def.names);
        assert!(def.source.starts_with("src/models/"), "{:?}", def.names);
    }
}

/// Two moved rows no fixture in `tests/snapshots/model_config/` reaches
/// (there is no `qwen2` or `qwen2moe` fixture, and
/// `tests/qwen2moe_paris.rs` needs a real checkpoint), so they are pinned
/// here against the llama.cpp lines their files cite.
#[test]
fn the_qwen_rows_no_fixture_reaches_are_pinned() {
    use crate::proj_bias::{output_presence, Presence};
    // qwen2.cpp:27: `output_b`, TENSOR_NOT_REQUIRED.
    assert_eq!(output_presence("qwen2"), Some(Presence::Optional));
    // qwen2moe.cpp:139: `norm_w = false`.
    assert!(ArchFacts::of("qwen2moe").moe.no_topk_renormalize);
    // qwen3moe.cpp:144: `norm_w = true`.
    assert!(!ArchFacts::of("qwen3moe").moe.no_topk_renormalize);
}

/// How each registry graph reads `attention.sliding_window_pattern`,
/// measured as `swa_layers`' two tables are (`load_swa_pattern` callers
/// read an array then a scalar, the `get_arr` readers an array only).
/// No snapshot fixture declares the key on most of these, so the census
/// is pinned here: a definition that drops its row reads the key as dead
/// metadata and serves the seed instead of the file's layout.
#[test]
fn the_registry_pattern_readers_are_pinned() {
    use crate::swa_layers::PatternKeyRead;
    let callers = [
        "cohere2",    // cohere2.cpp:5
        "cohere2moe", // cohere2moe.cpp:28
        "exaone4",    // exaone4.cpp:7
        "gpt-oss",    // openai-moe.cpp:9
        "llama4",     // llama4.cpp:19
        "olmo2",      // olmo2.cpp:9
    ];
    for name in registry_names() {
        let want = if callers.contains(&name) {
            PatternKeyRead::ArrayThenScalar
        } else if name == "kolibri1" {
            // kolibri1.cpp:17, `get_arr` with no scalar attempt.
            PatternKeyRead::PerLayerBool
        } else {
            PatternKeyRead::Unread
        };
        assert_eq!(crate::swa_layers::pattern_key_read(name), want, "{name}");
    }
}
