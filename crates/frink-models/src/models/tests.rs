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
        "cohere2",      // cohere2.cpp:5
        "cohere2moe",   // cohere2moe.cpp:28
        "exaone4",      // exaone4.cpp:7
        "gemma2",       // gemma2.cpp:6
        "gemma3",       // gemma3.cpp:7
        "mellum",       // mellum.cpp:16
        "plamo3",       // plamo3.cpp:9
        "gpt-oss",      // openai-moe.cpp:9
        "llama4",       // llama4.cpp:19
        "olmo2",        // olmo2.cpp:9
        "afmoe",        // afmoe.cpp:17
        "exaone-moe",   // exaone-moe.cpp:6
        "laguna",       // laguna.cpp:42
        "smallthinker", // smallthinker.cpp:9
        "muse-glimmer", // muse-glimmer.cpp:13
    ];
    for name in registry_names() {
        let want = if callers.contains(&name) {
            PatternKeyRead::ArrayThenScalar
        } else if [
            "kolibri1",
            "maple",
            "spark2_5",
            "granite_swa",
            "step35",
            "mimo2",
            "lfm3",
        ]
        .contains(&name)
        {
            // `get_arr` with no scalar attempt: kolibri1.cpp:17,
            // maple.cpp:10, spark2-5.cpp:8, granite-swa.cpp:17,
            // step35.cpp:26, mimo2.cpp:12, lfm3.cpp:13.
            PatternKeyRead::PerLayerBool
        } else {
            PatternKeyRead::Unread
        };
        assert_eq!(crate::swa_layers::pattern_key_read(name), want, "{name}");
    }
}

/// The four graphs that norm Q and K AFTER RoPE (`maincoder.cpp:78-95`,
/// `hunyuan-moe.cpp:93-118`, `hunyuan-vl.cpp:56-81` for `hunyuan-dense`,
/// `talkie.cpp:82-91`). No GGUF key states the order, so a definition
/// that drops its row serves the generic "before" order and shifts every
/// layer's attention scores without a shape change.
#[test]
fn the_registry_qk_norm_after_rope_rows_are_pinned() {
    let after = ["hunyuan-dense", "hunyuan-moe", "maincoder", "talkie"];
    for name in registry_names() {
        assert_eq!(
            ArchFacts::of(name).tensors.qk_norm_after_rope,
            after.contains(&name),
            "{name}"
        );
    }
}

/// `hunyuan-dense` inherits `hunyuan-vl`'s NTK-alpha RoPE rescale
/// (`models.h:1830-1834`, `hunyuan-vl.cpp:8-12`); no other registry
/// graph applies it.
#[test]
fn only_hunyuan_dense_rescales_the_rope_base_in_the_registry() {
    for name in registry_names() {
        assert_eq!(
            ArchFacts::of(name).scaling.rope_ntk_alpha,
            name == "hunyuan-dense",
            "{name}"
        );
    }
}

/// `talkie` is the one graph with the embedding skip stream, the
/// per-head scalar Q gain and the weightless RMSNorm at every site
/// (`talkie.cpp:26,50-52,68,90,110,123-126,137`) among the registry
/// rows, and its REQUIRED `logit_scale` is multiplied onto the logits
/// (`:5,141`).
#[test]
fn the_talkie_rows_are_pinned() {
    use crate::norm::NormFunction;
    use crate::scalar_multipliers::MultiplierSupport;
    for name in registry_names() {
        let facts = ArchFacts::of(name);
        let is_talkie = name == "talkie";
        assert_eq!(facts.topology.skip_stream, is_talkie, "{name}");
        assert_eq!(facts.tensors.per_head_scalar_qk_gain, is_talkie, "{name}");
        assert_eq!(
            facts.tensors.norm_sites.function == NormFunction::RmsNoParams,
            is_talkie,
            "{name}"
        );
    }
    // `cohere2` carries the same row (`cohere2.cpp:14,153-154`), so this
    // one is pinned for `talkie` alone.
    assert_eq!(
        ArchFacts::of("talkie").scaling.multipliers,
        MultiplierSupport::TALKIE
    );
}
