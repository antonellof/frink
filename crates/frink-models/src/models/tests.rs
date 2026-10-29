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
