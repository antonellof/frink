//! `ArchFacts` against the tables it is built from, and the loader
//! against the facts.

use super::*;
use crate::loader::arch_tables as t;

/// A name no table has a row for.
pub(crate) const ABSENT: &str = "frink-test-absent-from-every-table";

/// Every name some table keys on, plus the catalog, the registry, an
/// alias and a name nobody knows.
pub(crate) fn every_tabled_name() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = crate::capability::architecture_catalog()
        .iter()
        .map(|p| p.gguf_name)
        .collect();
    names.extend(crate::capability::AUDITED_GENERIC_GQA);
    names.extend(crate::models::REGISTRY.iter().flat_map(|d| d.names));
    names.extend(behaviour_table_names());
    names.extend(["llama-embed", "granite-moe", "gpt-oss", ABSENT]);
    names.sort_unstable();
    names.dedup();
    names
}

/// Every name a BEHAVIOUR table keys on: everything above but the
/// catalog and `AUDITED_GENERIC_GQA`, which plan steps 10 and 11 move.
pub(crate) fn behaviour_table_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = Vec::new();
    names.extend(t::SIGMOID_GATING_ARCHITECTURES);
    names.extend(t::GATING_LITERAL_ARCHITECTURES.iter().map(|(n, _)| *n));
    names.extend(t::EXPERT_WEIGHTS_NORM_READERS);
    names.extend(t::EXPERT_WEIGHTS_SCALE_READERS);
    names.extend(t::NO_TOPK_RENORMALIZE_ARCHITECTURES);
    names.extend(t::LEADING_DENSE_KEY_IS_INERT);
    names.extend(crate::router_input::ROUTER_INPUT_TABLE.iter().map(|r| r.0));
    names.extend(crate::layer_shapes::PURE_RECURRENT.iter().map(|r| r.0));
    names.extend(crate::rope_layers::ROPE_PATTERN_READERS.iter().map(|r| r.0));
    names.extend(crate::norm::NORM_BY_RMS_EPS_KEY.iter().map(|r| r.0));
    names.extend(crate::capability::LOGIT_SOFTCAP_ARCHITECTURES);
    names.extend(crate::act_layers::SWIGLU_CLAMP_READERS);
    names.extend(crate::act_layers::XIELU_ARCHITECTURES);
    names.extend(t::QK_NORM_AFTER_ROPE_ARCHITECTURES);
    names.extend(
        crate::qk_layer_norm::PER_HEAD_LAYER_NORM_QK
            .iter()
            .map(|r| r.0),
    );
    names.extend(crate::capability::PER_HEAD_SCALAR_QK_GAIN);
    names.extend(crate::capability::PER_HEAD_DISTINCT_QK_NORM);
    names.extend(
        crate::attn_gate::Q_INTERLEAVED_GATE_ARCHS
            .iter()
            .map(|r| r.0),
    );
    names.extend(
        crate::parallel_dense_ffn::SHARED_EXPERT_SUM_SCALE
            .iter()
            .map(|r| r.0),
    );
    {
        use crate::norm_sites as ns;
        names.extend(ns::POST_NORMS_BOTH_OR_NEITHER.iter().map(|r| r.0));
        for table in [
            ns::PRE_FFN_NORM_IS_POST_ATTENTION_NORM,
            ns::PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM,
            ns::POST_NORMS_UNDER_GROK_NAMES,
            ns::EMBEDDING_NORM_ARCHITECTURES,
            ns::WEIGHTLESS_EMBEDDING_NORM,
            ns::NO_OUTPUT_NORM,
            ns::OUTPUT_NORM_UNDER_EMBEDDING_NAME,
            ns::ONE_NORM_PER_LAYER,
        ] {
            names.extend(table);
        }
    }
    names.extend(crate::rope_finetuned::ROPE_GATED_ON_FINETUNED);
    names.extend(crate::clamp_kqv::CLAMPED_QKV_ARCHITECTURES);
    names.extend(crate::clamp_kqv::CLAMP_KQV_REQUIRED);
    names.extend(crate::mtp_blocks::NEXTN_READERS.iter().map(|r| r.0));
    names.extend(crate::proj_bias::ATTN_OUT_BIAS_CREATORS.iter().map(|r| r.0));
    names.extend(crate::proj_bias::FFN_BIAS_CREATORS.iter().map(|r| r.0));
    names.extend(crate::proj_bias::OUTPUT_BIAS_CREATORS.iter().map(|r| r.0));
    names.sort_unstable();
    names.dedup();
    names
}

/// The rules agree when they answer alike on a grid that reaches every
/// hparam-keyed arm today's functions have (EXAONE-4's 64 layers,
/// Gemma's 46 and 62, Baichuan's 40, a zero head width).
fn assert_rules_agree(name: &str, a: HparamRules, b: HparamRules) {
    let HparamRules {
        swa_window_override: wa,
        attention_scale_override: sa,
        rope_layers: ra,
        load_checks: ca,
    } = a;
    let HparamRules {
        swa_window_override: wb,
        attention_scale_override: sb,
        rope_layers: rb,
        load_checks: cb,
    } = b;
    // An empty file: every check that reads a key or a tensor answers.
    let empty = crate::test_source::StubSource::with_tensors(&[]);
    for n_layers in [0, 1, 4] {
        assert_eq!(
            ca(&empty, name, n_layers),
            cb(&empty, name, n_layers),
            "{name}: load_checks({n_layers})"
        );
    }
    for n_layers in [1, 4, 26, 40, 46, 62, 63, 64, 65] {
        assert_eq!(
            wa(name, n_layers),
            wb(name, n_layers),
            "{name}: swa_window_override({n_layers})"
        );
        for head_dim in [0, 128, 256] {
            assert_eq!(
                sa(name, n_layers, 4608, 32, head_dim),
                sb(name, n_layers, 4608, 32, head_dim),
                "{name}: attention_scale_override({n_layers}, head_dim {head_dim})"
            );
        }
        for window in [false, true] {
            for dense in [0, 1, 3] {
                assert_eq!(
                    ra(name, n_layers, window, dense),
                    rb(name, n_layers, window, dense),
                    "{name}: rope_layers({n_layers}, {window}, {dense})"
                );
            }
        }
    }
}

/// Field by field, with no `..`: a field added to [`ArchFacts`] does not
/// compile here until the agreement tests compare it.
pub(crate) fn assert_same(name: &str, got: ArchFacts, want: ArchFacts) {
    let ArchFacts {
        profile,
        audited,
        moe,
        swa,
        ffn,
        norm,
        scaling,
        topology,
        tensors,
        rules,
    } = got;
    assert_eq!(
        profile.map(|p| p.gguf_name),
        want.profile.map(|p| p.gguf_name),
        "{name}: profile"
    );
    assert_eq!(profile, want.profile, "{name}: profile row");
    assert_eq!(audited, want.audited, "{name}: audited");
    assert_eq!(moe, want.moe, "{name}: moe");
    assert_eq!(swa, want.swa, "{name}: swa");
    assert_eq!(ffn, want.ffn, "{name}: ffn");
    assert_eq!(norm, want.norm, "{name}: norm");
    assert_eq!(scaling, want.scaling, "{name}: scaling");
    assert_eq!(topology, want.topology, "{name}: topology");
    assert_eq!(tensors, want.tensors, "{name}: tensors");
    assert_rules_agree(name, rules, want.rules);
}

/// `DEFAULT` is what a name absent from every table gets, so a legacy
/// default that changes (a table's `unwrap_or`, a `match` arm's `_`)
/// cannot leave the constant stating the old one.
#[test]
fn default_is_a_name_absent_from_every_table() {
    assert_same(ABSENT, ArchFacts::from_tables(ABSENT), ArchFacts::DEFAULT);
}

/// For every name any table keys on, the facts the loaders read
/// ([`ArchFacts::of`]) state what each public lookup answers. For a
/// registry name, a row left behind in (or added to) a legacy table the
/// definition does not state turns this red.
#[test]
fn the_facts_agree_with_every_public_lookup() {
    use crate::capability as cap;
    for name in every_tabled_name() {
        let profile = cap::resolve_profile(name);
        // A row no public lookup answers for: the registry entry's, else
        // the table's. `no_registry_name_remains_in_a_legacy_table` holds
        // the table to nothing for a registry name.
        let def = crate::models::find(name).map(|d| d.facts);
        let want = ArchFacts {
            profile,
            audited: cap::is_audited_generic(name),
            moe: MoeFacts {
                sigmoid_by_default: t::SIGMOID_GATING_ARCHITECTURES.contains(&name),
                gating_literal: t::GATING_LITERAL_ARCHITECTURES
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, g)| *g),
                reads_expert_weights_norm: def
                    .map_or(t::EXPERT_WEIGHTS_NORM_READERS.contains(&name), |f| {
                        f.moe.reads_expert_weights_norm
                    }),
                reads_expert_weights_scale: t::EXPERT_WEIGHTS_SCALE_READERS.contains(&name),
                // A table with no public lookup: the definition's
                // statement, else the table (`no_registry_name_remains_in_a_legacy_table`
                // holds the table empty for registry names).
                no_topk_renormalize: def
                    .map_or(t::NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&name), |f| {
                        f.moe.no_topk_renormalize
                    }),
                leading_dense_key_is_inert: def
                    .map_or(t::LEADING_DENSE_KEY_IS_INERT.contains(&name), |f| {
                        f.moe.leading_dense_key_is_inert
                    }),
                routed_weight_before_ffn: crate::routed_weight_site::weight_before_ffn(name),
                router_input: crate::router_input::router_input(name),
                selection: def.map_or(frink_moe::SelectionRule::BiasAfterGating, |f| {
                    f.moe.selection
                }),
            },
            swa: SwaFacts {
                default_layout: cap::default_swa_layout(name),
                window_required: crate::swa_geometry::window_required(name),
                rope_base_required: crate::swa_geometry::swa_rope_base_required(name),
                layers_unscaled_rope: crate::swa_geometry::swa_layers_unscaled_rope(name),
                full_layers_rotate_half: crate::swa_geometry::full_layers_rotate_half(name),
                layers_drop_rope_factors: crate::swa_geometry::swa_layers_drop_rope_factors(name),
                rope_base_follows_model: cap::swa_rope_base_follows_model(name),
                rope_scale_follows_model: cap::swa_rope_scale_follows_model(name),
                rope_base_key_unread: def.and_then(|f| f.swa.rope_base_key_unread),
            },
            ffn: FfnFacts {
                // No legacy table: only a definition states it.
                length_counts_gate_and_up: def.is_some_and(|f| f.ffn.length_counts_gate_and_up),
                geglu: cap::uses_geglu(name),
                relu_sqr: cap::uses_relu_sqr(name),
                gelu_ungated: cap::uses_gelu_ungated(name),
                reglu: cap::uses_reglu(name),
                xielu: crate::act_layers::uses_xielu(name),
                reads_swiglu_clamps: crate::act_layers::reads_swiglu_clamps(name),
            },
            norm: NormFacts {
                norm_by_rms_eps_key: crate::norm::NORM_BY_RMS_EPS_KEY
                    .iter()
                    .any(|(a, _)| *a == name),
                block_sub_norms: crate::sub_norms::block_sub_norms(name),
            },
            scaling: ScalingFacts {
                multipliers: crate::scalar_multipliers::multiplier_support(name),
                logit_softcap: cap::LOGIT_SOFTCAP_ARCHITECTURES.contains(&name),
                embeddings_scaled_by_sqrt_n_embd: profile
                    .is_some_and(|p| cap::embeddings_scaled_by_sqrt_n_embd(name, p.family)),
                // Read: a declared clamp survives; required: an absent
                // one is refused.
                clamp_kqv: matches!(
                    crate::clamp_kqv::resolve_clamp(name, Some(1.0)),
                    Ok(Some(_))
                )
                .then(|| crate::clamp_kqv::resolve_clamp(name, None).is_err()),
            },
            topology: TopologyFacts {
                pure_recurrent_block: crate::layer_shapes::pure_recurrent_block(name),
                reads_rope_pattern: crate::rope_layers::reads_rope_pattern(name),
                skip_stream: crate::skip_stream::has_skip_stream(name),
                parallel_ssm: crate::mamba2::parallel_with_attention(name),
                learned_positions: crate::position_embd::learned_positions(name),
                rope_gated_on_finetuned: crate::rope_finetuned::unrotated(name, Some(false)),
                reads_nextn: crate::mtp_blocks::reads_nextn(name),
            },
            tensors: TensorFacts {
                gpt_oss_layers: name == "gpt-oss",
                norm_sites: crate::norm_sites::NormSites::for_arch(name),
                per_head_layer_norm_qk: crate::qk_layer_norm::PER_HEAD_LAYER_NORM_QK
                    .iter()
                    .find(|r| r.0 == name)
                    .map(|r| r.2),
                per_head_scalar_qk_gain: cap::uses_per_head_scalar_qk_gain(name),
                per_head_distinct_qk_norm: cap::uses_per_head_distinct_qk_norm(name),
                q_gate_interleaved: crate::attn_gate::q_gate_interleaved(name),
                post_norms_both_or_neither: crate::norm_sites::POST_NORMS_BOTH_OR_NEITHER
                    .iter()
                    .find(|r| r.0 == name)
                    .map(|r| r.1),
                shared_expert_sum_scale: crate::parallel_dense_ffn::shared_expert_sum_scale(
                    name, true,
                ),
                weightless_embedding_norm: crate::norm_sites::weightless_embedding_norm(name),
                qk_norm_after_rope: t::QK_NORM_AFTER_ROPE_ARCHITECTURES.contains(&name),
                attn_out_bias: crate::proj_bias::attn_out_presence(name),
                ffn_bias: crate::proj_bias::ffn_presence(name),
                output_bias: crate::proj_bias::output_presence(name),
            },
            rules: HparamRules {
                swa_window_override: cap::swa_window_override,
                attention_scale_override: cap::attention_scale_override,
                rope_layers: crate::rope_layers::rope_layers,
                load_checks: def.map_or(super::no_load_checks, |f| f.rules.load_checks),
            },
        };
        assert_same(name, ArchFacts::of(name), want);
    }
}

/// Plan risk 1: the config loader keys on the RAW name. `llama-embed`
/// is the one alias `canonical_architecture` folds, and on the config
/// side it keeps its own catalog row.
#[test]
fn config_facts_are_keyed_on_the_raw_name() {
    let raw = ArchFacts::of("llama-embed");
    assert_eq!(raw.profile.map(|p| p.gguf_name), Some("llama-embed"));
    assert_eq!(
        crate::capability::canonical_architecture("llama-embed"),
        "llama"
    );
    assert!(raw.audited);
}

/// A row on a name the catalog does not know can never be read: the
/// loader refuses the name before it reads any facts.
#[test]
fn every_tabled_name_is_a_catalog_name() {
    // `NEXTN_READERS` is a census of llama.cpp's graphs, measured over
    // all of `src/models/`, and names some frink has no row for.
    let census = |n: &str| crate::mtp_blocks::NEXTN_READERS.iter().any(|r| r.0 == n);
    for name in every_tabled_name() {
        if name == ABSENT || census(name) {
            continue;
        }
        assert!(
            crate::capability::resolve_profile(name).is_some(),
            "`{name}` has a table row but no catalog row, so the row is dead"
        );
    }
}

/// The derived key lists answer as they did when they lived in
/// `capability.rs` and read the tables by name.
#[test]
fn the_refused_key_lists_derive_from_the_facts() {
    let gemma = ArchFacts::of("gemma3");
    assert!(gemma.unsupported_feature_keys("gemma3").is_empty());
    let grok = ArchFacts::of("grok");
    let keys: Vec<String> = grok
        .unsupported_feature_keys("grok")
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert!(!keys.iter().any(|k| k.contains("softcapping")), "{keys:?}");
    let llama = ArchFacts::of("llama");
    assert!(llama
        .unsupported_feature_keys("llama")
        .iter()
        .any(|(k, _)| k == "llama.attn_logit_softcapping"));
    // The scaling keys are the complement of the multiplier row.
    let granite: Vec<String> = ArchFacts::of("granite")
        .unsupported_scaling_keys("granite")
        .into_iter()
        .map(|(k, _, _)| k)
        .collect();
    assert!(!granite.contains(&"granite.residual_scale".to_string()));
    assert!(llama
        .unsupported_scaling_keys("llama")
        .iter()
        .any(|(k, _, _)| k == "llama.residual_scale"));
}

/// `ModelConfig::from_gguf` reads these through [`ArchFacts`]; a direct
/// call creeping back would bypass whatever a `models/<arch>.rs` later
/// supplies. The source is scanned rather than trusted.
#[test]
fn the_config_loader_reads_the_tables_only_through_the_facts() {
    let src = code_only(include_str!("../loader/model_config.rs"));
    // A table is used as `NAME.contains(..)` / `NAME.iter()`; the bare
    // name also appears inside the best-effort strings the snapshots pin.
    for direct in [
        "arch_tables",
        "from_tables(",
        "rope_finetuned::unrotated(",
        "resolve_profile(",
        "capability::unsupported_feature_keys",
        "capability::unsupported_scaling_keys",
        "pure_recurrent_block(",
        "NORM_BY_RMS_EPS_KEY.",
        "LEADING_DENSE_KEY_IS_INERT.",
        "FFN_LENGTH_COUNTS_GATE_AND_UP.",
        "reads_rope_pattern(",
        "GATING_LITERAL_ARCHITECTURES.",
        "SIGMOID_GATING_ARCHITECTURES.",
        "EXPERT_WEIGHTS_NORM_READERS.",
        "NO_TOPK_RENORMALIZE_ARCHITECTURES.",
        "EXPERT_WEIGHTS_SCALE_READERS.",
        "capability::swa_window_override",
        "window_required(",
        "swa_rope_base_required(",
        "swa_layers_unscaled_rope(",
        "default_swa_layout(",
        "multiplier_support(",
        "embeddings_scaled_by_sqrt_n_embd(",
        "capability::attention_scale_override",
        "swa_rope_base_follows_model(",
        "uses_geglu(",
        "uses_relu_sqr(",
        "uses_gelu_ungated(",
        "uses_reglu(",
        "uses_xielu(",
        "reads_swiglu_clamps(",
        "full_layers_rotate_half(",
        "swa_layers_drop_rope_factors(",
        "swa_rope_scale_follows_model(",
        "is_audited_generic(",
        "has_skip_stream(",
        "parallel_with_attention(",
        "rope_layers::rope_layers(",
        "router_input::router_input(",
        "block_sub_norms(",
        "learned_positions(",
        "weight_before_ffn(",
    ] {
        assert!(
            !src.contains(direct),
            "loader/model_config.rs calls `{direct}` directly; read it from `ArchFacts`"
        );
    }
}

/// Code only: the comments still name the tables they explain.
fn code_only(src: &str) -> String {
    src.lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The same for `Decoder::from_gguf`, which builds its tensor rows from
/// the CANONICAL name and `family` / `memory` from the raw one.
#[test]
fn the_decoder_loader_reads_the_tables_only_through_the_facts() {
    let src = code_only(include_str!("../loader/decoder_from_gguf.rs"));
    for direct in [
        "arch_tables",
        "from_tables(",
        "rope_finetuned::unrotated(",
        "resolve_profile(",
        "== \"gpt-oss\"",
        "NormSites::with_function(",
        "NormSites::for_arch(",
        "per_head_layer_norm_refusal(",
        "uses_per_head_scalar_qk_gain(",
        "uses_per_head_distinct_qk_norm(",
        "q_gate_interleaved(",
        "post_norm_pairing_refusal(",
        "shared_expert_sum_scale(",
        "weightless_embedding_norm(",
        "QK_NORM_AFTER_ROPE_ARCHITECTURES.",
    ] {
        assert!(
            !src.contains(direct),
            "loader/decoder_from_gguf.rs calls `{direct}` directly; read it from `ArchFacts`"
        );
    }
    // Plan risk 1, in the order the loader reads them: the canonical
    // facts first, from the name `canonical_architecture` returned; the
    // raw facts from `general.architecture` itself.
    let canonical = src
        .find("ArchFacts::of(&arch)")
        .expect("tensor facts built from the canonical `arch`");
    assert!(src[..canonical].contains("canonical_architecture("));
    let raw = src
        .find("ArchFacts::of(\n            file.metadata_str(\"general.architecture\")")
        .expect("family / memory facts built from the raw name");
    assert!(raw > canonical);
}
