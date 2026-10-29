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

use frink_gguf::TensorSource;
use frink_moe::{GatingFunction, SelectionRule};

use crate::capability::{ArchProfile, DecoderFamily, SwaPattern, SwaWindowOverride};
use crate::layer_shapes::ZeroKvLayer;
use crate::norm_sites::NormSites;
use crate::proj_bias::Presence;
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
    /// Where `exp_probs_b` enters the top-k ([`SelectionRule`]). No
    /// legacy table names the other rule: every tabled row is
    /// `BiasAfterGating`, `build_moe_ffn`'s own.
    pub selection: SelectionRule,
    /// `moe_interleave`: the llama.cpp lines when the loader honours
    /// `interleave_moe_layer_step` (`llama4`).
    pub interleave_step_honoured: Option<&'static str>,
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
    /// The llama.cpp line where `arch` seeds the sliding layers' RoPE
    /// base from the model's and never reads `rope.freq_base_swa`, so a
    /// file's key is dead. `None` where the key is read (the base loader
    /// does not read it; each graph's `load_arch_hparams` does or not).
    pub rope_base_key_unread: Option<&'static str>,
    /// `chunked_swa`: the literal chunk and the lines, for a CHUNKED
    /// window (`llama4`).
    pub chunked: Option<(usize, &'static str)>,
}

/// FFN rows, in the order the loader tests them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfnFacts {
    /// `feed_forward_length` counts gate and up together, so each FFN
    /// matrix is half as wide (`crate::models::qwen`).
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalingFacts {
    /// `scalar_multipliers::multiplier_support`.
    pub multipliers: MultiplierSupport,
    /// `capability::LOGIT_SOFTCAP_ARCHITECTURES`.
    pub logit_softcap: bool,
    /// `capability::embeddings_scaled_by_sqrt_n_embd` for the row's own
    /// family.
    pub embeddings_scaled_by_sqrt_n_embd: bool,
    /// `clamp_kqv`: `None` where the graph reads no clamp, else whether
    /// the key is REQUIRED.
    pub clamp_kqv: Option<bool>,
    /// `rope_scaling::YARN_BETA_READERS`: the YaRN `beta_fast` /
    /// `beta_slow` keys are read (dead metadata everywhere else).
    pub reads_yarn_beta: bool,
    /// `attn_temperature`: a LITERAL attention temperature, read from no
    /// key (`llama4`).
    pub attn_temperature_literal: Option<crate::attn_temperature::AttnTemperature>,
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
    /// `rope_finetuned::ROPE_GATED_ON_FINETUNED`: a file declaring
    /// `rope.scaling.finetuned = false` rotates nothing.
    pub rope_gated_on_finetuned: bool,
    /// `mtp_blocks::NEXTN_READERS`: `nextn_predict_layers` blocks are
    /// subtracted from the layer count.
    pub reads_nextn: bool,
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
    /// `proj_bias::ATTN_OUT_BIAS_CREATORS`: `blk.N.attn_output.bias`.
    pub attn_out_bias: Option<Presence>,
    /// `proj_bias::FFN_BIAS_CREATORS`: the dense FFN biases, and
    /// whether `ffn_gate.bias` is among them.
    pub ffn_bias: Option<(Presence, bool)>,
    /// `proj_bias::OUTPUT_BIAS_CREATORS`: `output.bias` on the LM head.
    pub output_bias: Option<Presence>,
    /// `qkv_fused::QKV_TENSOR_CREATORS`: which Q/K/V biases the graph
    /// creates, and how.
    pub qkv_tensors: crate::qkv_fused::QkvTensors,
    /// `parallel_dense_ffn`: a dense FFN summed with the routed experts.
    pub parallel_dense: Option<&'static crate::parallel_dense_ffn::ParallelDenseFfn>,
}

/// The decisions that are code rather than rows: each takes a
/// hyperparameter beside the architecture. `from_tables` points every
/// one at today's name-keyed function; a `models/<arch>.rs` will point
/// it at its own.
#[derive(Debug, Clone, Copy)]
pub struct HparamRules {
    /// `capability::swa_window_override(arch, n_layers)`, on the trunk:
    /// `exaone4.cpp:4`'s layer-count gate.
    pub swa_window_override: fn(&str, usize) -> SwaWindowOverride,
    /// `capability::attention_scale_override(arch, n_layers, n_embd,
    /// n_heads, head_dim)`: Gemma-2/3 27B and `jais`.
    pub attention_scale_override: fn(&str, usize, usize, usize, usize) -> Option<f32>,
    /// `rope_layers::rope_layers(arch, n_layers, has_window, n_dense_lead)`.
    pub rope_layers: fn(&str, usize, bool, usize) -> RopeLayers,
    /// The refusals in the architecture's own `load_arch_hparams` and
    /// `load_arch_tensors` that no table states: a REQUIRED key, a
    /// value the graph throws on, a REQUIRED tensor the generic loader
    /// treats as optional. `(file, arch, n_layers)`; `None` loads.
    pub load_checks: fn(&dyn TensorSource, &str, usize) -> Option<String>,
    /// `weightless_qk_norm(arch, n_experts)`: Q and K normed with no
    /// weight (`llama4.cpp:43`).
    pub weightless_qk_norm: fn(&str, usize) -> bool,
}

/// [`HparamRules::load_checks`] for an architecture with none.
pub fn no_load_checks(_: &dyn TensorSource, _: &str, _: usize) -> Option<String> {
    None
}

impl HparamRules {
    /// Today's name-keyed functions.
    pub const LEGACY: Self = Self {
        swa_window_override: crate::capability::swa_window_override_from_table,
        attention_scale_override: crate::capability::attention_scale_override,
        rope_layers: crate::rope_layers::rope_layers_from_table,
        load_checks: no_load_checks,
        weightless_qk_norm: crate::weightless_qk_norm::no_weightless_qk_norm,
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
            selection: SelectionRule::BiasAfterGating,
            interleave_step_honoured: None,
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
            rope_base_key_unread: None,
            chunked: None,
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
            clamp_kqv: None,
            reads_yarn_beta: false,
            attn_temperature_literal: None,
        },
        topology: TopologyFacts {
            pure_recurrent_block: None,
            reads_rope_pattern: false,
            skip_stream: false,
            parallel_ssm: false,
            learned_positions: false,
            rope_gated_on_finetuned: false,
            reads_nextn: false,
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
            attn_out_bias: None,
            ffn_bias: None,
            output_bias: None,
            qkv_tensors: crate::qkv_fused::QkvTensors::Helper,
            parallel_dense: None,
        },
        rules: HparamRules::LEGACY,
    };

    /// The facts for `arch` as the legacy tables state them.
    ///
    /// Keyed on the name exactly as given; the callers decide raw or
    /// canonical (module doc).
    pub fn from_tables(arch: &str) -> Self {
        // Only the legacy tables: a public lookup that consults the
        // registry (`multiplier_support`, `attn_out_presence`, ...) would
        // recurse through `of`.
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
                // No table: `llama4` was the only row.
                routed_weight_before_ffn: false,
                router_input: crate::router_input::router_input(arch),
                selection: SelectionRule::BiasAfterGating,
                // No table: `llama4` was the only row.
                interleave_step_honoured: None,
            },
            swa: SwaFacts {
                default_layout: crate::capability::default_swa_layout_from_table(arch),
                window_required: crate::swa_geometry::window_required_from_table(arch),
                rope_base_required: crate::swa_geometry::swa_rope_base_required(arch),
                layers_unscaled_rope: crate::swa_geometry::swa_layers_unscaled_rope_from_table(
                    arch,
                ),
                full_layers_rotate_half: crate::swa_geometry::full_layers_rotate_half(arch),
                layers_drop_rope_factors: crate::swa_geometry::swa_layers_drop_rope_factors(arch),
                rope_base_follows_model: crate::capability::swa_rope_base_follows_model_from_table(
                    arch,
                ),
                rope_scale_follows_model:
                    crate::capability::swa_rope_scale_follows_model_from_table(arch),
                rope_base_key_unread: None,
                // No table: `llama4` was the only row.
                chunked: None,
            },
            ffn: FfnFacts {
                // No table: `qwen` was the only row (`crate::models::qwen`).
                length_counts_gate_and_up: false,
                geglu: crate::capability::uses_geglu_from_table(arch),
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
                multipliers: crate::scalar_multipliers::multiplier_support_from_table(arch),
                logit_softcap: crate::capability::LOGIT_SOFTCAP_ARCHITECTURES.contains(&arch),
                // Only read once the profile resolved; an unknown name
                // has no family and is refused before this is read.
                embeddings_scaled_by_sqrt_n_embd: profile.is_some_and(|p| {
                    crate::capability::embeddings_scaled_by_sqrt_n_embd(arch, p.family)
                }),
                clamp_kqv: crate::clamp_kqv::clamp_row_from_table(arch),
                reads_yarn_beta: crate::loader::rope_scaling::YARN_BETA_READERS.contains(&arch),
                // No table: `llama4` was the only row.
                attn_temperature_literal: None,
            },
            topology: TopologyFacts {
                pure_recurrent_block: crate::layer_shapes::pure_recurrent_block(arch),
                reads_rope_pattern: crate::rope_layers::reads_rope_pattern(arch),
                skip_stream: crate::skip_stream::has_skip_stream(arch),
                parallel_ssm: crate::mamba2::parallel_with_attention(arch),
                learned_positions: crate::position_embd::learned_positions(arch),
                rope_gated_on_finetuned: crate::rope_finetuned::ROPE_GATED_ON_FINETUNED
                    .contains(&arch),
                reads_nextn: crate::mtp_blocks::reads_nextn_from_table(arch),
            },
            tensors: TensorFacts {
                // No table: `gpt-oss` states it in `crate::models::gpt_oss`.
                gpt_oss_layers: false,
                norm_sites: NormSites::with_function(
                    arch,
                    crate::norm::norm_function_from_table(arch),
                ),
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
                attn_out_bias: crate::proj_bias::attn_out_presence_from_table(arch),
                ffn_bias: crate::proj_bias::ffn_presence_from_table(arch),
                output_bias: crate::proj_bias::output_presence_from_table(arch),
                qkv_tensors: crate::qkv_fused::qkv_tensors_from_table(arch),
                parallel_dense: crate::parallel_dense_ffn::parallel_dense_ffn_from_table(arch),
            },
            rules: HparamRules::LEGACY,
        }
    }

    /// The facts every loader reads: the `crate::models` registry
    /// entry naming `arch`, else the legacy tables.
    ///
    /// The catalog row and the audited flag still come from
    /// `capability.rs` for a registry name (steps 10 and 11 move them).
    pub fn of(arch: &str) -> Self {
        match super::find(arch) {
            Some(def) => Self {
                profile: crate::capability::resolve_profile(arch),
                audited: crate::capability::is_audited_generic(arch),
                ..def.facts
            },
            None => Self::from_tables(arch),
        }
    }

    /// The catalog row's family, if the name resolved.
    pub fn family(&self) -> Option<DecoderFamily> {
        self.profile.map(|p| p.family)
    }
}

#[cfg(test)]
#[path = "facts_tests.rs"]
pub(crate) mod tests;
