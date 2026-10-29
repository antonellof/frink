//! [`ArchFacts`]: everything `ModelConfig::from_gguf` knows about an
//! architecture before it reads the file, in one value.
//!
//! Step 1 of `docs/plans/arch-registry-migration.md` (#403). The facts
//! are built by [`ArchFacts::from_tables`] from the same name-keyed
//! tables the loader used to call one at a time, so nothing about an
//! answer changes; what changes is that the loader reads one value, and
//! a later `models/<arch>.rs` can supply that value without editing a
//! dozen tables.
//!
//! Field groups follow the table families. Every row type is today's.
//! [`ArchFacts::DEFAULT`] is what a name absent from every table gets,
//! and a test holds it to that.
//!
//! Two names reach the loaders (plan risk 1). `ModelConfig::from_gguf`
//! and `Decoder::from_gguf`'s `family` / `memory` build the facts from
//! the RAW `general.architecture`; the decoder's tensor rows
//! ([`TensorFacts`]) come from the CANONICAL one
//! (`capability::canonical_architecture`), so an alias reads the tables
//! of the graph llama.cpp computes for it.
//!
//! What is NOT here yet (still keyed on the name at the call site):
//! every resolver that reads the file or takes a hyperparameter beside
//! the name (`mtp_blocks::trunk_layers`, `kv_head_dims`, `clamp_kqv`,
//! `swa_geometry::rotary_widths`, the decoder's `load_*` helpers, ...).

use frink_moe::GatingFunction;

use crate::capability::{ArchProfile, DecoderFamily, SwaPattern, SwaWindowOverride};
use crate::layer_shapes::ZeroKvLayer;
use crate::norm_sites::NormSites;
use crate::rope_layers::RopeLayers;
use crate::router_input::RouterInput;
use crate::scalar_multipliers::MultiplierSupport;

/// The per-architecture facts the loaders read.
#[derive(Debug, Clone, Copy)]
pub struct ArchFacts {
    /// The catalog row (`capability::resolve_profile`). `None` is an
    /// unknown name, which the loader refuses first.
    pub profile: Option<&'static ArchProfile>,
    /// `capability::is_audited_generic`.
    pub audited: bool,
    pub moe: MoeFacts,
    pub swa: SwaFacts,
    pub ffn: FfnFacts,
    pub norm: NormFacts,
    pub scaling: ScalingFacts,
    pub topology: TopologyFacts,
    pub tensors: TensorFacts,
    pub rules: HparamRules,
}

/// MoE routing rows (`loader/arch_tables.rs`, `routed_weight_site`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeFacts {
    /// `SIGMOID_GATING_ARCHITECTURES`: sigmoid when the file has no key.
    pub sigmoid_by_default: bool,
    /// `GATING_LITERAL_ARCHITECTURES`: the graph's literal, which wins
    /// over the file's key.
    pub gating_literal: Option<GatingFunction>,
    /// `EXPERT_WEIGHTS_NORM_READERS`.
    pub reads_expert_weights_norm: bool,
    /// `EXPERT_WEIGHTS_SCALE_READERS`.
    pub reads_expert_weights_scale: bool,
    /// `NO_TOPK_RENORMALIZE_ARCHITECTURES`.
    pub no_topk_renormalize: bool,
    /// `LEADING_DENSE_KEY_IS_INERT`.
    pub leading_dense_key_is_inert: bool,
    /// `routed_weight_site::weight_before_ffn`.
    pub routed_weight_before_ffn: bool,
    /// `router_input::router_input`.
    pub router_input: RouterInput,
}

/// Sliding-window rows (`capability`, `swa_geometry`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwaFacts {
    /// `capability::default_swa_layout`: the seeded period and phase.
    pub default_layout: Option<SwaPattern>,
    /// `swa_geometry::window_required`: the llama.cpp line, if the
    /// window key is REQUIRED.
    pub window_required: Option<&'static str>,
    /// `swa_geometry::swa_rope_base_required`.
    pub rope_base_required: Option<&'static str>,
    /// `swa_geometry::swa_layers_unscaled_rope`.
    pub layers_unscaled_rope: Option<&'static str>,
    /// `swa_geometry::full_layers_rotate_half`.
    pub full_layers_rotate_half: Option<&'static str>,
    /// `swa_geometry::swa_layers_drop_rope_factors`.
    pub layers_drop_rope_factors: Option<&'static str>,
    /// `capability::swa_rope_base_follows_model`.
    pub rope_base_follows_model: bool,
    /// `capability::swa_rope_scale_follows_model`.
    pub rope_scale_follows_model: bool,
}

/// FFN rows, in the order the loader tests them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfnFacts {
    /// `FFN_LENGTH_COUNTS_GATE_AND_UP`.
    pub length_counts_gate_and_up: bool,
    /// `capability::uses_geglu`.
    pub geglu: bool,
    /// `capability::uses_relu_sqr`.
    pub relu_sqr: bool,
    /// `capability::uses_gelu_ungated`.
    pub gelu_ungated: bool,
    /// `capability::uses_reglu`.
    pub reglu: bool,
    /// `act_layers::uses_xielu`.
    pub xielu: bool,
    /// `act_layers::reads_swiglu_clamps`.
    pub reads_swiglu_clamps: bool,
}

/// Norm rows (`norm`, `sub_norms`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormFacts {
    /// `norm::NORM_BY_RMS_EPS_KEY`: a zero RMS epsilon means "absent".
    pub norm_by_rms_eps_key: bool,
    /// `sub_norms::block_sub_norms`.
    pub block_sub_norms: bool,
}

/// Metadata-multiplier rows (`scalar_multipliers`, `capability`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalingFacts {
    /// `scalar_multipliers::multiplier_support`.
    pub multipliers: MultiplierSupport,
    /// `capability::LOGIT_SOFTCAP_ARCHITECTURES`.
    pub logit_softcap: bool,
    /// `capability::embeddings_scaled_by_sqrt_n_embd` for the row's own
    /// family.
    pub embeddings_scaled_by_sqrt_n_embd: bool,
}

/// Layer-topology rows read at config time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyFacts {
    /// `layer_shapes::pure_recurrent_block`.
    pub pure_recurrent_block: Option<ZeroKvLayer>,
    /// `rope_layers::reads_rope_pattern`.
    pub reads_rope_pattern: bool,
    /// `skip_stream::has_skip_stream`.
    pub skip_stream: bool,
    /// `mamba2::parallel_with_attention`.
    pub parallel_ssm: bool,
    /// `position_embd::learned_positions`.
    pub learned_positions: bool,
}

/// Decoder tensor-table rows (`Decoder::from_gguf`, canonical name).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorFacts {
    /// gpt-oss's four extra per-layer tensors (`decoder::GptOssWeights`).
    /// A name test, not a table: no other graph has them.
    pub gpt_oss_layers: bool,
    /// `norm_sites::NormSites::for_arch`: where each norm site is
    /// stored. The loader replaces `function` with the one the file
    /// decided (`ModelConfig::norm_function`).
    pub norm_sites: NormSites,
    /// `qk_layer_norm::per_head_layer_norm_lines`.
    pub per_head_layer_norm_qk: Option<&'static str>,
    /// `capability::uses_per_head_scalar_qk_gain`.
    pub per_head_scalar_qk_gain: bool,
    /// `capability::uses_per_head_distinct_qk_norm`.
    pub per_head_distinct_qk_norm: bool,
    /// `attn_gate::q_gate_interleaved`.
    pub q_gate_interleaved: bool,
    /// `norm_sites::post_norms_both_or_neither`.
    pub post_norms_both_or_neither: Option<&'static str>,
    /// `parallel_dense_ffn::shared_expert_sum_scale_row`.
    pub shared_expert_sum_scale: Option<f32>,
    /// `norm_sites::weightless_embedding_norm`.
    pub weightless_embedding_norm: bool,
    /// `QK_NORM_AFTER_ROPE_ARCHITECTURES`.
    pub qk_norm_after_rope: bool,
}

/// The decisions that are code rather than rows: each takes a
/// hyperparameter beside the architecture. `from_tables` points every
/// one at today's name-keyed function; a `models/<arch>.rs` will point
/// it at its own.
#[derive(Debug, Clone, Copy)]
pub struct HparamRules {
    /// `capability::swa_window_override(arch, block_count)`:
    /// `exaone4.cpp:4`'s layer-count gate.
    pub swa_window_override: fn(&str, usize) -> SwaWindowOverride,
    /// `capability::attention_scale_override(arch, n_layers, n_embd,
    /// n_heads, head_dim)`: Gemma-2/3 27B and `jais`.
    pub attention_scale_override: fn(&str, usize, usize, usize, usize) -> Option<f32>,
    /// `rope_layers::rope_layers(arch, n_layers, has_window, n_dense_lead)`.
    pub rope_layers: fn(&str, usize, bool, usize) -> RopeLayers,
}

impl HparamRules {
    /// Today's name-keyed functions.
    pub const LEGACY: Self = Self {
        swa_window_override: crate::capability::swa_window_override,
        attention_scale_override: crate::capability::attention_scale_override,
        rope_layers: crate::rope_layers::rope_layers,
    };
}

impl ArchFacts {
    /// A name absent from every table.
    pub const DEFAULT: Self = Self {
        profile: None,
        audited: false,
        moe: MoeFacts {
            sigmoid_by_default: false,
            gating_literal: None,
            reads_expert_weights_norm: false,
            reads_expert_weights_scale: false,
            no_topk_renormalize: false,
            leading_dense_key_is_inert: false,
            routed_weight_before_ffn: false,
            router_input: RouterInput::NormedFfnInput,
        },
        swa: SwaFacts {
            default_layout: None,
            window_required: None,
            rope_base_required: None,
            layers_unscaled_rope: None,
            full_layers_rotate_half: None,
            layers_drop_rope_factors: None,
            rope_base_follows_model: false,
            rope_scale_follows_model: false,
        },
        ffn: FfnFacts {
            length_counts_gate_and_up: false,
            geglu: false,
            relu_sqr: false,
            gelu_ungated: false,
            reglu: false,
            xielu: false,
            reads_swiglu_clamps: false,
        },
        norm: NormFacts {
            norm_by_rms_eps_key: false,
            block_sub_norms: false,
        },
        scaling: ScalingFacts {
            multipliers: MultiplierSupport::NONE,
            logit_softcap: false,
            embeddings_scaled_by_sqrt_n_embd: false,
        },
        topology: TopologyFacts {
            pure_recurrent_block: None,
            reads_rope_pattern: false,
            skip_stream: false,
            parallel_ssm: false,
            learned_positions: false,
        },
        tensors: TensorFacts {
            gpt_oss_layers: false,
            norm_sites: NormSites::GENERIC,
            per_head_layer_norm_qk: None,
            per_head_scalar_qk_gain: false,
            per_head_distinct_qk_norm: false,
            q_gate_interleaved: false,
            post_norms_both_or_neither: None,
            shared_expert_sum_scale: None,
            weightless_embedding_norm: false,
            qk_norm_after_rope: false,
        },
        rules: HparamRules::LEGACY,
    };

    /// The facts for `arch` as the legacy tables state them.
    ///
    /// Keyed on the name exactly as given; the callers decide raw or
    /// canonical (module doc).
    pub fn from_tables(arch: &str) -> Self {
        use crate::loader::arch_tables as t;
        let profile = crate::capability::resolve_profile(arch);
        Self {
            profile,
            audited: crate::capability::is_audited_generic(arch),
            moe: MoeFacts {
                sigmoid_by_default: t::SIGMOID_GATING_ARCHITECTURES.contains(&arch),
                gating_literal: t::GATING_LITERAL_ARCHITECTURES
                    .iter()
                    .find(|(name, _)| *name == arch)
                    .map(|(_, g)| *g),
                reads_expert_weights_norm: t::EXPERT_WEIGHTS_NORM_READERS.contains(&arch),
                reads_expert_weights_scale: t::EXPERT_WEIGHTS_SCALE_READERS.contains(&arch),
                no_topk_renormalize: t::NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&arch),
                leading_dense_key_is_inert: t::LEADING_DENSE_KEY_IS_INERT.contains(&arch),
                routed_weight_before_ffn: crate::routed_weight_site::weight_before_ffn(arch),
                router_input: crate::router_input::router_input(arch),
            },
            swa: SwaFacts {
                default_layout: crate::capability::default_swa_layout(arch),
                window_required: crate::swa_geometry::window_required(arch),
                rope_base_required: crate::swa_geometry::swa_rope_base_required(arch),
                layers_unscaled_rope: crate::swa_geometry::swa_layers_unscaled_rope(arch),
                full_layers_rotate_half: crate::swa_geometry::full_layers_rotate_half(arch),
                layers_drop_rope_factors: crate::swa_geometry::swa_layers_drop_rope_factors(arch),
                rope_base_follows_model: crate::capability::swa_rope_base_follows_model(arch),
                rope_scale_follows_model: crate::capability::swa_rope_scale_follows_model(arch),
            },
            ffn: FfnFacts {
                length_counts_gate_and_up: t::FFN_LENGTH_COUNTS_GATE_AND_UP.contains(&arch),
                geglu: crate::capability::uses_geglu(arch),
                relu_sqr: crate::capability::uses_relu_sqr(arch),
                gelu_ungated: crate::capability::uses_gelu_ungated(arch),
                reglu: crate::capability::uses_reglu(arch),
                xielu: crate::act_layers::uses_xielu(arch),
                reads_swiglu_clamps: crate::act_layers::reads_swiglu_clamps(arch),
            },
            norm: NormFacts {
                norm_by_rms_eps_key: crate::norm::NORM_BY_RMS_EPS_KEY
                    .iter()
                    .any(|(a, _)| *a == arch),
                block_sub_norms: crate::sub_norms::block_sub_norms(arch),
            },
            scaling: ScalingFacts {
                multipliers: crate::scalar_multipliers::multiplier_support(arch),
                logit_softcap: crate::capability::LOGIT_SOFTCAP_ARCHITECTURES.contains(&arch),
                // Only read once the profile resolved; an unknown name
                // has no family and is refused before this is read.
                embeddings_scaled_by_sqrt_n_embd: profile.is_some_and(|p| {
                    crate::capability::embeddings_scaled_by_sqrt_n_embd(arch, p.family)
                }),
            },
            topology: TopologyFacts {
                pure_recurrent_block: crate::layer_shapes::pure_recurrent_block(arch),
                reads_rope_pattern: crate::rope_layers::reads_rope_pattern(arch),
                skip_stream: crate::skip_stream::has_skip_stream(arch),
                parallel_ssm: crate::mamba2::parallel_with_attention(arch),
                learned_positions: crate::position_embd::learned_positions(arch),
            },
            tensors: TensorFacts {
                gpt_oss_layers: arch == "gpt-oss",
                norm_sites: NormSites::for_arch(arch),
                per_head_layer_norm_qk: crate::qk_layer_norm::per_head_layer_norm_lines(arch),
                per_head_scalar_qk_gain: crate::capability::uses_per_head_scalar_qk_gain(arch),
                per_head_distinct_qk_norm: crate::capability::uses_per_head_distinct_qk_norm(arch),
                q_gate_interleaved: crate::attn_gate::q_gate_interleaved(arch),
                post_norms_both_or_neither: crate::norm_sites::post_norms_both_or_neither(arch),
                shared_expert_sum_scale: crate::parallel_dense_ffn::shared_expert_sum_scale_row(
                    arch,
                ),
                weightless_embedding_norm: crate::norm_sites::weightless_embedding_norm(arch),
                qk_norm_after_rope: t::QK_NORM_AFTER_ROPE_ARCHITECTURES.contains(&arch),
            },
            rules: HparamRules::LEGACY,
        }
    }

    /// The catalog row's family, if the name resolved.
    pub fn family(&self) -> Option<DecoderFamily> {
        self.profile.map(|p| p.family)
    }
}

#[cfg(test)]
#[path = "facts_tests.rs"]
mod tests;
