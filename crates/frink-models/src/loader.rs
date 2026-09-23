//! Loads a real `Decoder` from an on-disk GGUF file, using the
//! llama.cpp-style tensor naming convention
//! (`token_embd.weight`, `blk.N.attn_q.weight`, `blk.N.ffn_gate.weight`
//! or, for MoE, `blk.N.ffn_gate_exps.weight`, `output_norm.weight`,
//! `output.weight`). Until this module existed, frink could only run
//! correctly-shaped *random* weights.
//!
//! Quantized tensors (Q8_0 / Q4_0) are loaded as `WeightMatrix::Quantized`
//! backed by `WeightBytes::Mapped` -- a zero-copy view into the same
//! mmap `GgufFile` already holds, with no intermediate heap copy of the
//! tensor's bytes at all. So a checkpoint's resident memory is the
//! mmap page cache, not the mmap plus a second in-process copy of every
//! weight. `WeightMatrix::apply` dispatches to frink-quant's fused
//! dequant+dot kernels directly against those mapped bytes at inference
//! time. F32 tensors (norms, embeddings, and any weight not natively
//! quantized) still copy into an owned `Tensor`, since they're small
//! relative to the quantized weight matrices and need per-element
//! access patterns a raw byte view doesn't support as cleanly.
//!
//! Verified end to end (see `crates/frink-models/tests/gguf_roundtrip.rs`)
//! against a genuinely Q8_0-quantized, generated on-disk GGUF fixture
//! for the dense (single-expert) case, and against real OLMoE / Qwen2-MoE
//! checkpoints for the multi-expert 3D-packed-tensor path.

use frink_core::expert_store::{ExpertKey, ExpertSource, ExpertStore};
use frink_core::tensor::Tensor;
use frink_core::weight_matrix::quant_kind_for;
use frink_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
use frink_gguf::{GgmlType, GgufError, GgufValue, ShardedGguf, TensorInfo, TensorSource};
use frink_moe::{ExpertWeights, GatingFunction, MoeLayerConfig};
use std::sync::Arc;
use thiserror::Error;

use crate::config::ModelConfig;
#[cfg(feature = "metal")]
use crate::decoder::MoePackedQ4Planes;
use crate::decoder::{AttnWeights, Decoder, ExpertBacking, LayerWeights, MoeWeights};
use crate::norm::NormOp;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Shard(#[from] frink_gguf::ShardError),
    #[error("tensor '{0}' has unsupported dtype {1:?}")]
    UnsupportedDtype(String, GgmlType),
    #[error(
        "MoE tensor '{0}' is not 3D or its expert count {1} does not match config n_experts {2}"
    )]
    ExpertCountMismatch(String, usize, usize),
    #[error("GGUF file is missing required hparam metadata key '{0}'")]
    MissingHparam(String),
    /// `general.architecture` is not in the capability registry -- refuse
    /// to guess RoPE/gating rather than emit fluent-but-wrong logits.
    #[error(
        "unsupported GGUF architecture '{0}': not in frink's capability registry \
         (unknown required features fail closed; see frink_models::capability)"
    )]
    UnsupportedArchitecture(String),
    /// Architecture exists but must not use the generic GQA decoder.
    #[error("architecture '{0}' cannot use the generic Decoder: {1}")]
    DedicatedArchitectureRequired(String, &'static str),
    /// Metadata advertises a feature the generic decoder does not implement.
    #[error("architecture '{0}' requires unimplemented feature: {1}")]
    UnsupportedFeature(String, String),
    #[error(
        "architecture '{0}' has never been verified against llama.cpp. It would run on \
         frink's shared generic-GQA path, which ASSUMES plain GQA with {1:?} RoPE and no \
         ALiBi, no learned position embeddings and no per-layer rope skipping. That \
         assumption has already been wrong for gpt2, mpt, refact, bloom and jais, each of \
         which loaded clean and answered as a different model. {2} Set \
         FRINK_ALLOW_UNAUDITED_ARCH=1 to run it anyway and compare the output against \
         llama.cpp yourself"
    )]
    UnauditedArchitecture(String, crate::config::RopeLayout, String),
    /// The checkpoint carries per-block tensors this build never reads,
    /// i.e. weights that contribute to the real graph and would simply
    /// be missing from ours. See [`assert_every_tensor_consumed`].
    #[error(
        "checkpoint carries {0} tensor(s) this build never reads, so its graph is not the one \
         frink would run: {1}. This is a missing feature, not a corrupt file. Override with \
         FRINK_ALLOW_UNKNOWN_TENSORS=1 to load anyway and accept wrong output."
    )]
    UnconsumedTensors(usize, String),
    /// `FRINK_STRICT_KERNELS=1` and the model has weights with no
    /// kernel on the selected accelerator, i.e. it would run, correctly,
    /// on a silently slower path. Refusing is the point: a benchmark or
    /// CI run must not be able to publish a number taken off the
    /// backend it claims. See [`frink_core::kernel_registry`].
    #[error("{0}")]
    StrictKernels(String),
}

/// Architecture-family name strings (GGUF's `general.architecture` value)
/// known, from reading ik_llama.cpp's `llama-hparams.cpp`
/// (`LLM_ARCH_DEEPSEEK2`, `LLM_ARCH_GLM4_MOE` cases), to default to
/// sigmoid MoE gating with post-selection renormalization rather than
/// softmax. Every member's citation is inline here; `docs/MODELS.md`
/// carries none and the pointer that used to send readers there was
/// dangling.
/// `afmoe`, `laguna` and `step35` added 2026-09-01 by the
/// unaudited-refusal triage's gating sweep. Each reads
/// `LLM_KV_EXPERT_GATING_FUNC` as OPTIONAL and then, when the key is
/// absent, sets `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`
/// (`afmoe.cpp:29-30`, `laguna.cpp:55-56`, `step35.cpp:19-20`). Frink
/// fell back to softmax for all three.
///
/// This is the `deepseek` shape a third, fourth and fifth time: a
/// default that is right for most architectures and silently wrong for
/// one, where the GGUF carries no key to correct it. Nothing is live
/// today -- all three are `NewCode` for other reasons and refuse before
/// reaching here -- but the list is what a later admission would trust.
const SIGMOID_GATING_ARCHITECTURES: &[&str] = &[
    "afmoe",
    // cohere2moe.cpp:27-29: the key read optional, SIGMOID when absent.
    "cohere2moe",
    "deepseek2",
    "glm4moe",
    "laguna",
    "step35",
];

/// Architectures whose graph passes a gating LITERAL into
/// `build_moe_ffn`, so the file's `expert_gating_func` is never read:
/// the literal wins even over a key that says otherwise.
///
/// Measured 2026-09-12 by parsing every `build_moe_ffn(` call's
/// arguments in all 155 `src/models/*.cpp`: three graphs pass
/// `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` (`llama4.cpp`, `mimo2.cpp:227`,
/// `nemotron-h.cpp`), twenty-six pass `_SOFTMAX`, nineteen pass
/// `hparams.expert_gating_func`. Only `mimo2` of the three is on this
/// loader. The twenty-six softmax literals are not tabled: every
/// converter for them writes no key or writes SOFTMAX, so the key and
/// the literal agree on every real file, and a table of twenty-six
/// hand-copied rows would be a bigger risk than the hand-written file
/// it guards against. `conversion/mimo.py` writes SIGMOID from
/// `scoring_func`, so on a real MiMo file the two agree too; the row
/// exists because the literal is what llama.cpp runs.
const GATING_LITERAL_ARCHITECTURES: &[(&str, GatingFunction)] = &[
    ("mimo2", GatingFunction::Sigmoid),
    // `nemotron-h.cpp:218`: the SIGMOID literal; the converter writes no
    // `expert_gating_func` (`conversion/nemotron.py:238-250`).
    ("nemotron_h_moe", GatingFunction::Sigmoid),
    // `llama4.cpp:230`: the SIGMOID literal, with `norm_w = false` at
    // `:228` (`NO_TOPK_RENORMALIZE_ARCHITECTURES`); the converter
    // writes no key (`conversion/llama.py:374-394`).
    ("llama4", GatingFunction::Sigmoid),
];
/// The names alone, for the cross-table test.
#[cfg(test)]
const GATING_LITERAL_NAMES: &[&str] = &["mimo2", "nemotron_h_moe", "llama4"];

/// Architectures whose `load_arch_hparams` reads
/// `{arch}.expert_weights_scale` (`LLM_KV_EXPERT_WEIGHTS_SCALE`) -- and
/// so the only ones whose `build_moe_ffn` call sees a nonzero
/// `hparams.expert_weights_scale`. On every other architecture the key
/// is dead metadata upstream: the field stays 0 and the multiply is
/// skipped, whatever the file says.
///
/// Measured 2026-09-12: `grep -l LLM_KV_EXPERT_WEIGHTS_SCALE
/// src/models/*.cpp` is twenty graphs; these are the eight on this
/// loader (`deepseek2` / `deepseek32` / `deepseek2ocr` / `deepseek4` /
/// `glm-dsa` / `glm4-moe` / `kimi-linear` / `minimax-m3` / `dflash` /
/// `nemotron-h` / `hy-v3` are on other engines, refused, or unknown
/// here; `cohere2moe.cpp:20` joined on 2026-09-14). Found by `mimo2`'s fixture: `mimo2.cpp` reads the
/// key nowhere, libllama ran the fixture unscaled, and frink -- which
/// honoured the key for any architecture -- scaled it by 2.5.
const EXPERT_WEIGHTS_SCALE_READERS: &[&str] = &[
    "afmoe",
    "bailingmoe",
    "bailingmoe2",
    // `cohere2moe.cpp:19-20` read both the norm and the scale.
    "cohere2moe",
    "deepseek",
    "dots1",
    "exaone-moe",
    // `glm4-moe.cpp:13-14` read both the scale and the norm.
    "glm4moe",
    "laguna",
    // `nemotron-h.cpp:19` (`routed_scaling_factor`, 2.5 on Nemotron-3 Nano).
    "nemotron_h_moe",
    "step35",
];

/// The same for `{arch}.expert_weights_norm` (`LLM_KV_EXPERT_WEIGHTS_NORM`):
/// eighteen graphs read it upstream, seven on this loader; every other
/// graph passes `norm_w` as a LITERAL into `build_moe_ffn`, and the
/// literal is what `NO_TOPK_RENORMALIZE_ARCHITECTURES` and its default
/// transcribe. `deepseek` reads the scale but not the norm
/// (`deepseek.cpp` passes `false`).
const EXPERT_WEIGHTS_NORM_READERS: &[&str] = &[
    "afmoe",
    "bailingmoe",
    "bailingmoe2",
    "cohere2moe",
    "dots1",
    "exaone-moe",
    "glm4moe",
    "laguna",
    // `nemotron-h.cpp:18` (`norm_topk_prob`).
    "nemotron_h_moe",
    "step35",
];

/// Names that appear in a behaviour table above but are `DedicatedOnly`
/// or `Deferred`, together with the module that actually applies the
/// behaviour for them.
///
/// Two true things were in conflict here, and deleting either would
/// have lost one. `SIGMOID_GATING_ARCHITECTURES` records a fact about
/// llama.cpp (these architectures default to sigmoid when the GGUF
/// carries no `expert_gating_func`), and a test pins it as such. The
/// cross-table test records a different fact: an entry for an
/// architecture that never reaches THIS loader cannot fire, and a gate
/// that cannot fire is worse than no gate because it reads as coverage.
///
/// Both hold. `deepseek2` is genuinely sigmoid-gated and genuinely
/// never arrives here. So the resolution is not to drop a name
/// from either place, it is to say out loud who owns it instead, and to
/// make an unexplained dead entry still fail.
///
/// Adding a name here is a claim that the named module applies the
/// behaviour. It is checked no further than that, so it is the one line
/// in this file to be suspicious of.
/// Test-only: it asserts a relationship rather than driving one, and a
/// production reader would have to be told that.
#[cfg(test)]
const DEDICATED_OWNS_ITS_BEHAVIOUR: &[(&str, &str)] = &[
    // `mla_gguf_loader` reads `expert_gating_func` and falls back to
    // Sigmoid itself, so deepseek2's gating is decided there.
    ("deepseek2", "mla_gguf_loader"),
    // `glm4moe` was here while it was refused; it is a generic-path
    // row now (2026-09-12) and the sigmoid default is live in THIS
    // loader.
];

/// Architecture-family names whose real reference implementation skips
/// renormalizing top-k softmax routing weights after selection (GGUF
/// carries no metadata key for this -- it's hardcoded per-architecture in
/// both the real HF `transformers` model code and llama.cpp's
/// `build_moe_ffn` call sites, not read from the file). Confirmed for
/// `olmoe` against `OlmoeTopKRouter.forward` in
/// `transformers/models/olmoe/modeling_olmoe.py` (`config.norm_topk_prob`
/// is `false` in the real published config.json) and llama.cpp's
/// `src/models/olmoe.cpp` (`build_moe_ffn(..., false, ...,
/// LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, ...)`). See
/// `MoeLayerConfig::norm_topk_prob`'s doc comment for why this matters:
/// getting it wrong silently produces wrong generation output even
/// though the file loads and shape-validates fine.
// Architectures whose reference graphs pass `norm_w=false` to
// `build_moe_ffn` (llama.cpp) / `norm_topk_prob=false` in HF config.
// Qwen2-MoE: `.scratch/llama.cpp/src/models/qwen2moe.cpp` -- Softmax +
// `false` for the norm_topk slot. Renormalizing top-k weights made
// Qwen1.5-MoE greedy decode emit garbage despite shared-expert load.
// `deepseek` (V1) added 2026-09-01 by the unaudited-refusal triage.
// `src/models/deepseek.cpp:145-155` passes `norm_w=false`, and
// `conversion/deepseek.py`'s `DeepseekModel` never writes
// `{arch}.expert_weights_norm` -- only `DeepseekV2Model` does -- so no
// real `deepseek` GGUF carries the key to override the default with.
// Frink therefore renormalised where llama.cpp does not. Same class of
// bug as the OLMoE one above, and latent only because `deepseek` is
// unaudited and refuses first.
// `jamba.cpp:164` passes `norm_w = false` and its converter writes no
// `expert_weights_norm` (`conversion/jamba.py:24-54`).
// `llama4.cpp:228` passes `false` beside its SIGMOID literal
// (`GATING_LITERAL_ARCHITECTURES`): the top-k sigmoid scores weight
// the experts unrenormalised.
const NO_TOPK_RENORMALIZE_ARCHITECTURES: &[&str] =
    &["deepseek", "jamba", "llama4", "olmoe", "qwen2moe"];

/// Architectures whose `{arch}.feed_forward_length` counts the gate and
/// the up projection TOGETHER, so each FFN matrix is half as wide as the
/// key says.
///
/// Qwen-1 (`QWenLMHeadModel`, GGUF string `qwen` -- not `qwen2` and not
/// `qwen3`) is the only one. HF's `QWenMLP` sets
/// `ff_dim_in = config.intermediate_size // 2` and builds `w1` and `w2`
/// at that width; `conversion/qwen.py`'s `QwenModel` inherits the base
/// `set_gguf_parameters`, which writes `intermediate_size` through
/// unchanged (`conversion/base.py:1206`); and `src/models/qwen.cpp:33-35`
/// therefore creates `ffn_gate`, `ffn_up` and `ffn_down` at `n_ff / 2`.
///
/// **This costs no logits and is still worth fixing.** frink loads the
/// dense FFN by tensor NAME and uses each matrix's own shape, so the
/// forward pass was always right; what was wrong was `expert_ffn_dim`,
/// which is what every memory estimate and `frink inspect-plan` row
/// prices the FFN from. That is this repo's dominant bug shape --
/// `ModelConfig` and the weights disagreeing about one number with
/// nothing comparing them -- so
/// `the_declared_ffn_width_matches_the_matrices_that_load`
/// (tests/one_match_arm_graphs.rs) now compares them.
const FFN_LENGTH_COUNTS_GATE_AND_UP: &[&str] = &["qwen"];

// The norm-slot lists (`PRE_FFN_NORM_IS_POST_ATTENTION_NORM` and its
// siblings) live in `crate::norm_sites`, beside the table that reads
// them; the tests below still walk them.

/// Architectures whose checkpoints carry `{arch}.leading_dense_block_count`
/// while their reference graph never branches on it: **every** layer is
/// MoE regardless of what the key says.
///
/// `bailingmoe` is the case this list exists for.
/// `src/models/bailingmoe.cpp:5` reads
/// `LLM_KV_LEADING_DENSE_BLOCK_COUNT` into `n_layer_dense_lead` and then
/// `load_arch_tensors` creates `ffn_gate_inp`, the expert tensors and
/// the shared-expert tensors unconditionally for every layer (:39-54 --
/// there is no `if (i < n_layer_dense_lead)` anywhere in the file) and
/// the graph has no dense branch either (:119-152). Meanwhile
/// `conversion/bailingmoe.py:27` writes `first_k_dense_replace` into the
/// key verbatim, so real Ling checkpoints DO carry a nonzero value.
///
/// Frink's `ModelConfig::layer_is_dense` does branch on it, so without
/// this list frink looks for `blk.0.ffn_gate.weight` on a layer that
/// only ships experts and dies on a missing tensor. That is a load
/// failure rather than wrong logits, which is why it stayed latent.
///
/// Do not read this as "the key is meaningless": for `deepseek`,
/// `dots1`, `glm4moe` and every other leading-dense architecture the key
/// is load-bearing and must be honoured. Membership here is a statement
/// about ONE architecture's graph, checked in that graph.
const LEADING_DENSE_KEY_IS_INERT: &[&str] = &["bailingmoe"];

/// Architectures whose reference graph applies `attn_q_norm` /
/// `attn_k_norm` AFTER `ggml_rope_ext`, not before it.
///
/// There is no GGUF key for this. llama.cpp writes the order into each
/// hand-written graph, so the only place it can come from is the
/// architecture string, and getting it wrong changes every layer's
/// attention scores without changing a single tensor shape.
///
/// - `maincoder`: `src/models/maincoder.cpp:78-90` ropes Q and K, then
///   norms them at `:92` and `:95`.
/// - `hunyuan-moe`: `src/models/hunyuan-moe.cpp:93,104` rope, `:110,115`
///   norm.
///
/// - `hunyuan-dense`: it has no graph of its own --
///   `src/models/models.h:1830-1834` derives `llama_model_hunyuan_dense`
///   from `llama_model_hunyuan_vl` and reuses its graph -- so the lines
///   are `hunyuan-vl.cpp:56-66` (rope) then `:73-81` (norm). Its second
///   blocker, the NTK-alpha RoPE base rescale, is implemented too; see
///   [`crate::rope_ntk_alpha`].
///
/// The audited majority is the other way round -- `qwen3moe.cpp:99,108`
/// and `bailingmoe2.cpp:123-135` both norm first -- which is why the
/// decoder's default is "before" and this list is the exception.
const QK_NORM_AFTER_ROPE_ARCHITECTURES: &[&str] =
    &["hunyuan-dense", "hunyuan-moe", "maincoder", "talkie"];

fn metadata_u64_any(file: &impl TensorSource, keys: &[String]) -> Option<u64> {
    keys.iter().find_map(|k| file.metadata_u64(k))
}

fn metadata_f32_any(file: &impl TensorSource, keys: &[String]) -> Option<f32> {
    keys.iter()
        .find_map(|k| file.metadata(k).and_then(GgufValue::as_f32))
}

impl ModelConfig {
    /// Derives a `ModelConfig` from a real GGUF file's own hyperparameter
    /// metadata, following llama.cpp's `general.architecture`-prefixed key
    /// convention (`{arch}.block_count`, `{arch}.embedding_length`,
    /// `{arch}.attention.head_count`, `{arch}.expert_count`, ...) rather
    /// than requiring a hand-written preset to already match the file's
    /// shape exactly. This is what lets `frink-server` (and `frink
    /// run-real`) load an arbitrary checkpoint, not just the three
    /// hand-tuned presets in `config.rs`.
    ///
    /// Fields with no corresponding metadata key fall back to widely-used
    /// llama.cpp defaults (documented inline) and are listed in the
    /// returned config's `best_effort_fields`, following the same
    /// confirmed-vs-estimated discipline as the hand-written presets.
    pub fn from_gguf(file: &impl TensorSource) -> Result<Self, LoadError> {
        let arch = file
            .metadata_str("general.architecture")
            .ok_or_else(|| LoadError::MissingHparam("general.architecture".to_string()))?
            .to_string();
        let arch_profile = crate::capability::resolve_profile(&arch)
            .ok_or_else(|| LoadError::UnsupportedArchitecture(arch.clone()))?;
        let rope_layout = match arch_profile.path {
            crate::capability::ArchPath::GenericGqa { rope }
            | crate::capability::ArchPath::TestFixture { rope } => rope,
            crate::capability::ArchPath::DedicatedOnly { reason } => {
                return Err(LoadError::DedicatedArchitectureRequired(
                    arch.clone(),
                    reason,
                ));
            }
            crate::capability::ArchPath::Deferred { reason } => {
                return Err(LoadError::UnsupportedFeature(
                    arch.clone(),
                    format!("architecture deferred from Frink text-generation scope: {reason}"),
                ));
            }
        };
        let qk_norm_style = arch_profile.qk_norm;
        // A vision export's text tower declaring M-RoPE sections on an
        // architecture whose text rotation is NORM (`crate::mrope`).
        if let Some(reason) = crate::mrope::mrope_refusal(file, &arch) {
            return Err(LoadError::UnsupportedFeature(arch.clone(), reason));
        }
        for (meta_key, feature) in crate::capability::unsupported_feature_keys(&arch) {
            if let Some(v) = metadata_f32_any(file, std::slice::from_ref(&meta_key)) {
                if v > 0.0 {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!("{feature} (metadata {meta_key}={v})"),
                    ));
                }
            }
            if let Some(v) = metadata_u64_any(file, std::slice::from_ref(&meta_key)) {
                if v > 0 {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        feature.to_string(),
                    ));
                }
            }
        }
        // Metadata-declared multipliers the generic decoder does not
        // apply. Unlike the tensor-consumption gate, nothing about these
        // is visible in the weights, so a Granite checkpoint would load
        // and answer at the wrong scale. See
        // `capability::unsupported_scaling_keys`.
        for (meta_key, feature, no_op) in crate::capability::unsupported_scaling_keys(&arch) {
            if let Some(v) = metadata_f32_any(file, std::slice::from_ref(&meta_key)) {
                if (v - no_op).abs() > 1e-6 {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!("{feature} (metadata {meta_key}={v})"),
                    ));
                }
            }
        }
        let key = |suffix: &str| format!("{arch}.{suffix}");

        let name: &'static str = Box::leak(
            file.metadata_str("general.name")
                .unwrap_or(&arch)
                .to_string()
                .into_boxed_str(),
        );

        let block_count =
            file.metadata_u64(&key("block_count"))
                .ok_or_else(|| LoadError::MissingHparam(key("block_count")))? as usize;
        // llama.cpp's `n_layer()` is `block_count` MINUS the NextN/MTP
        // blocks the converter appended inside it, for the graphs that
        // read `nextn_predict_layers` (`crate::mtp_blocks`). `n_layers`
        // is the trunk from here on; `block_count` is handed ONLY to the
        // two things llama.cpp decides before it has read the key --
        // `exaone4.cpp:4`'s layer-count gate and the per-layer array
        // lengths -- and nowhere else.
        let trunk = crate::mtp_blocks::trunk_layers(file, &arch, block_count)?;
        // Nanbeige's `num_loops`: the trunk is the PHYSICAL count and
        // `n_layers` the logical one from here on (`crate::layer_loops`);
        // the per-layer arrays below are read at physical length and
        // replicated per pass, as `nanbeige.cpp:24-26` replicate them.
        let layer_loops = crate::layer_loops::read_layer_loops(file, &arch, trunk.n_layers)?;
        let n_layers = layer_loops
            .map(|l| l.logical_layers())
            .unwrap_or(trunk.n_layers);
        // Baichuan-13B (block_count 40) used to be refused HERE: one
        // architecture string, two positional schemes, decided by layer
        // count with no key (`baichuan.cpp:11-14`). It is served now
        // through `crate::alibi` (the bias) and `crate::rope_layers`
        // (no rotation), both keyed on the same layer count.
        // EXAONE-4 32B used to be refused HERE, on the same shape:
        // `exaone4.cpp:4-14` switches the whole SWA machinery on inside
        // `if (hparams.n_layer() == 64)` and :116 then ropes only the
        // sliding layers, so its full-attention layers get no rotation
        // at all and no GGUF key says so. That is now IMPLEMENTED rather
        // than refused -- `capability::swa_disabled_by_arch` carries the
        // layer-count gate and `crate::rope_layers` the per-layer
        // rotation rule it feeds -- so the two EXAONE-4 sizes are one
        // code path with two answers instead of one running and one
        // stopping. `tests/no_rope_layer_graphs.rs` has a 64-layer
        // fixture against libllama's own logits.
        let hidden_dim = file
            .metadata_u64(&key("embedding_length"))
            .ok_or_else(|| LoadError::MissingHparam(key("embedding_length")))?
            as usize;
        // Scalar OR per-layer array, as llama.cpp reads all three
        // (`get_key_or_arr`, llama-model.cpp:1149-1158). The scalars
        // below are the WIDEST layer's; `ModelConfig::layer_shape` is
        // what a layer body reads. See `crate::layer_shapes`.
        let heads_per_layer =
            crate::layer_shapes::read_u64_trunk_layers(file, &key("attention.head_count"), &trunk)?
                .ok_or_else(|| LoadError::MissingHparam(key("attention.head_count")))?;
        let n_heads = heads_per_layer.iter().copied().max().unwrap_or(0) as usize;

        let mut best_effort_fields: Vec<&'static str> = Vec::new();

        let kv_heads_per_layer = match crate::layer_shapes::read_u64_trunk_layers(
            file,
            &key("attention.head_count_kv"),
            &trunk,
        )? {
            Some(v) => v,
            None => {
                best_effort_fields.push("n_kv_heads (no attention.head_count_kv key; assumed equal to n_heads, i.e. plain MHA)");
                heads_per_layer.clone()
            }
        };
        let n_kv_heads = kv_heads_per_layer.iter().copied().max().unwrap_or(0) as usize;
        let head_dim = match file.metadata_u64(&key("attention.key_length")) {
            Some(v) => v as usize,
            None => {
                // llama.cpp derives it from LAYER 0's head count
                // (`n_embd / n_head()`, llama-model.cpp:1195), which on
                // a file whose layer 0 has none is a division by zero
                // there and a refusal here.
                let h0 = heads_per_layer.first().copied().unwrap_or(0) as usize;
                // A pure recurrent model has no heads and no head width
                // (`layer_shapes::PURE_RECURRENT`); nothing reads one.
                match hidden_dim.checked_div(h0) {
                    _ if h0 == 0 && crate::layer_shapes::pure_recurrent_block(&arch).is_some() => 0,
                    None => {
                        return Err(LoadError::MissingHparam(format!(
                            "{} (layer 0 declares head_count 0, so it cannot be derived as \
                             hidden_dim / n_heads)",
                            key("attention.key_length")
                        )));
                    }
                    Some(derived) => {
                        best_effort_fields.push(
                            "head_dim (no attention.key_length key; derived as hidden_dim / n_heads)",
                        );
                        derived
                    }
                }
            }
        };
        let v_head_dim = crate::kv_head_dims::resolve_v_head_dim(
            &arch,
            head_dim,
            file.metadata_u64(&key("attention.value_length"))
                .map(|v| v as usize),
        )?;
        // `Some` only when it differs: see `ModelConfig::v_head_dim`.
        let v_head_dim = (v_head_dim != head_dim).then_some(v_head_dim);
        let vocab_size = file
            .metadata("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GgufValue::Array(items) => Some(items.len()),
                _ => None,
            })
            .or_else(|| file.metadata_u64(&key("vocab_size")).map(|v| v as usize))
            .unwrap_or_else(|| {
                best_effort_fields.push("vocab_size (no tokenizer.ggml.tokens array or {arch}.vocab_size key; fell back to output.weight's own row count)");
                // `output.weight`'s real raw shape is `[hidden_dim,
                // vocab_size]` (ggml's fastest-first `ne[]` order --
                // see `load_weight_matrix`'s doc comment), so vocab_size
                // is the *last* element, not the first.
                file.find_tensor("output.weight")
                    .and_then(|t| t.shape.last().copied())
                    .unwrap_or(0) as usize
            });
        let rope_theta = metadata_f32_any(file, &[key("rope.freq_base")]).unwrap_or_else(|| {
            best_effort_fields.push("rope_theta (no rope.freq_base key; defaulted to 10000.0)");
            10000.0
        });
        // NTK-alpha: `{arch}.rope.scaling.alpha` is read for every
        // architecture (llama-model.cpp:1186) and APPLIED by two
        // (`hunyuan-vl.cpp:8-12`, inherited by `hunyuan-dense`). The
        // list and the arithmetic live together in one module so the
        // key's readers and its appliers cannot drift apart -- see
        // `crate::rope_ntk_alpha`, which also records why a converted
        // `hunyuan-dense` file carries the already-scaled base instead.
        let rope_theta = crate::rope_ntk_alpha::ntk_alpha_scaled_rope_base(
            &arch,
            rope_theta,
            head_dim,
            metadata_f32_any(file, &[key("rope.scaling.alpha")]),
        );
        // The norm FUNCTION is the architecture's, except where the file
        // decides it (`crate::norm::NORM_BY_RMS_EPS_KEY`): a present and
        // nonzero RMS epsilon means RMSNorm there, and a zero one is
        // llama.cpp's "absent", so it is dropped before the epsilon
        // itself is read below.
        let declared_rms_eps = metadata_f32_any(file, &[key("attention.layer_norm_rms_epsilon")])
            .filter(|eps| {
                *eps != 0.0
                    || !crate::norm::NORM_BY_RMS_EPS_KEY
                        .iter()
                        .any(|(a, _)| *a == arch)
            });
        let norm_function = crate::norm::norm_function_for_file(&arch, declared_rms_eps);
        let rms_norm_eps = declared_rms_eps
            .or_else(|| metadata_f32_any(file, &[key("attention.layer_norm_epsilon")]))
            .unwrap_or_else(|| {
                best_effort_fields
                    .push("rms_norm_eps (no layer_norm_rms_epsilon key; defaulted to 1e-5)");
                1e-5
            });

        let n_experts = metadata_u64_any(file, &[key("expert_count")]).unwrap_or(0) as usize;
        let is_moe = n_experts > 1;

        // `expert_used_count` is scalar OR an array at `block_count`
        // length: `llama-model.cpp:1266` reads it with `get_key_or_arr`
        // in the COMMON loader, for every architecture, and
        // `gguf_writer.py:869-873` writes whichever it is handed.
        // `conversion/nemotron.py:574` hands it a LIST (Nemotron-H
        // Puzzle, one entry per block), and `nemotron_h` is an
        // architecture frink serves -- so before 2026-09-19 such a
        // file read no scalar here, fell into the default, and routed
        // top-2 on every layer whatever the file said. A uniform array
        // is that one value; a varying one needs a per-layer top-k the
        // MoE layer does not have and stops by name rather than
        // picking a number.
        let n_experts_active = if is_moe {
            match crate::layer_shapes::read_u64_per_layer(
                file,
                &key("expert_used_count"),
                // `n_layer_all`, i.e. `block_count` including any MTP
                // blocks, which is the length llama.cpp asks for at
                // `llama-model.cpp:1266` -- BEFORE `n_layer()` drops
                // them.
                block_count,
            )? {
                Some(per_layer) => {
                    let first = per_layer[0];
                    if per_layer.iter().any(|v| *v != first) {
                        return Err(LoadError::UnsupportedFeature(
                            key("expert_used_count"),
                            format!(
                                "a PER-LAYER expert count ({per_layer:?}). llama.cpp reads this \
                                 key with `get_key_or_arr` for every architecture \
                                 (llama-model.cpp:1266) and routes layer `il` to \
                                 `n_expert_used_arr[il]` experts; frink carries one top-k for \
                                 the model, so it would route every layer to {first} and answer \
                                 something else. conversion/nemotron.py:574 writes the array for \
                                 Nemotron-H Puzzle"
                            ),
                        ));
                    }
                    first as usize
                }
                None => {
                    best_effort_fields
                        .push("moe.n_experts_active (no expert_used_count key; defaulted to 2)");
                    2
                }
            }
        } else {
            1
        };
        // Read here, ahead of the shared-expert inference below, because
        // the tensor that inference probes lives on the first MoE
        // layer, not on layer 0.
        let n_dense_leading_layers = if LEADING_DENSE_KEY_IS_INERT.contains(&arch.as_str()) {
            0
        } else {
            metadata_u64_any(file, &[key("leading_dense_block_count")]).unwrap_or(0) as usize
        };
        // Prefer the GGUF hparam when present. Qwen2MoE (and some other
        // HF→GGUF exports) omit `expert_shared_count` but still ship
        // `blk.N.ffn_{gate,up,down}_shexp.weight` -- without a tensor-
        // presence fallback those weights are silently dropped and the
        // model runs with a large chunk of active FFN missing.
        //
        // The probe is the FIRST MoE LAYER, not `blk.0`: a leading-dense
        // model has no shared expert on layer 0, and probing there
        // answered 0 for every such file. `laguna` is the case that
        // found it -- `laguna.cpp:20` assigns `n_expert_shared = 1`
        // before reading the key, `conversion/laguna.py` never writes
        // the key, and its layer 0 is dense (:105), so a real Laguna
        // export loaded with its three REQUIRED `_shexp` tensors
        // (:138-140) unread on every MoE layer.
        // The interleave step the loader honours (`crate::moe_interleave`,
        // `llama4.cpp:64`), read here because the shared-expert probe
        // below needs the first layer it makes MoE.
        let moe_interleave_step = crate::moe_interleave::interleave_step(
            &arch,
            metadata_u64_any(file, &[key("interleave_moe_layer_step")]),
            n_experts,
        )
        .map_err(|reason| LoadError::UnsupportedFeature(arch.clone(), reason))?;
        let first_moe_layer = (n_dense_leading_layers..n_layers)
            .find(|&il| !moe_interleave_step.is_some_and(|step| !(il + 1).is_multiple_of(step)))
            .unwrap_or(n_layers.saturating_sub(1));
        let shexp_probe = format!("blk.{first_moe_layer}.ffn_gate_shexp.weight");
        let n_shared_experts = match metadata_u64_any(file, &[key("expert_shared_count")]) {
            Some(n) => n as usize,
            None if is_moe && file.find_tensor(&shexp_probe).is_some() => {
                best_effort_fields.push(
                    "moe.n_shared_experts (no expert_shared_count; inferred 1 from the first \
                     MoE layer's ffn_gate_shexp.weight)",
                );
                1
            }
            None => 0,
        };
        // MoE GGUFs often only set `feed_forward_length` (OLMoE=1024,
        // Qwen2-MoE=5632 for the shared expert). `expert_feed_forward_length`
        // is optional. llama.cpp `qwen2moe.cpp` uses
        // `n_ff_exp = n_ff_exp ? n_ff_exp : n_ff / n_expert_used` (1408 for
        // Qwen1.5-MoE); the shared expert keeps the full `n_ff` (5632).
        let ffn_per_layer =
            crate::layer_shapes::read_u64_trunk_layers(file, &key("feed_forward_length"), &trunk)?
                // Qwen-1 declares gate and up as one number; see
                // `FFN_LENGTH_COUNTS_GATE_AND_UP`.
                .map(|v| {
                    if FFN_LENGTH_COUNTS_GATE_AND_UP.contains(&arch.as_str()) {
                        v.into_iter().map(|ff| ff / 2).collect()
                    } else {
                        v
                    }
                });
        let feed_forward_length = ffn_per_layer.as_ref().and_then(|v| v.iter().copied().max());
        // Scalar OR an array, exactly as `expert_used_count` above:
        // llama.cpp reads it with `get_key_or_arr` (`maple.cpp:6`,
        // `dots3note.cpp:11`, `nemotron-h.cpp`), and
        // `conversion/nemotron.py:573` writes a LIST for Nemotron-H
        // Puzzle -- an architecture frink serves. Read as a scalar
        // alone, an array answered `None` here and the fallback below
        // silently sized every expert at `feed_forward_length /
        // n_experts_used`, which is a different FFN and loads without
        // complaint when the tensors happen to be that wide.
        let expert_ffn_per_layer = crate::layer_shapes::read_u64_per_layer(
            file,
            &key("expert_feed_forward_length"),
            block_count,
        )?;
        if let Some(per_layer) = expert_ffn_per_layer.as_ref() {
            let first = per_layer[0];
            if per_layer.iter().any(|v| *v != first) {
                return Err(LoadError::UnsupportedFeature(
                    key("expert_feed_forward_length"),
                    format!(
                        "a PER-LAYER expert FFN width ({per_layer:?}). llama.cpp sizes the \
                         expert tensors from layer 0's entry and `LayerShapes` carries one \
                         expert width for the model, so frink would build every layer at \
                         {first} and read the others' weights at the wrong stride"
                    ),
                ));
            }
        }
        let expert_ffn_dim = expert_ffn_per_layer
            .map(|v| v[0])
            .or_else(|| {
                feed_forward_length.map(|ff| {
                    if is_moe && n_experts_active > 0 {
                        ff / n_experts_active as u64
                    } else {
                        ff
                    }
                })
            })
            .unwrap_or_else(|| {
                best_effort_fields.push(
                    "moe.expert_ffn_dim (no expert_feed_forward_length/feed_forward_length; defaulted to 4x hidden_dim)",
                );
                (hidden_dim * 4) as u64
            }) as usize;
        // `{arch}.attention.rope_pattern`: one entry per layer, nonzero
        // meaning "this layer rotates" (`llama-hparams.cpp:333-343`).
        // Read only where llama.cpp reads it, because
        // `llama-model.cpp:1314` seeds the array with 1 for every
        // architecture and only `granite-swa.cpp:43` reads it back --
        // so honouring it elsewhere would answer differently from
        // upstream on a file that carries it as dead metadata.
        let rope_pattern: Option<std::sync::Arc<[bool]>> =
            if crate::rope_layers::reads_rope_pattern(&arch) {
                crate::layer_shapes::read_u64_per_layer(
                    file,
                    &key("attention.rope_pattern"),
                    block_count,
                )?
                .map(|v| v.into_iter().map(|x| x != 0).collect())
            } else {
                None
            };

        // Qwen3.5's recurrent layers come from two keys, not from the
        // head counts (`crate::gdn::recurrent_layers`).
        let recurrent_layers =
            crate::gdn::recurrent_layers(file, &arch, trunk.block_count, n_layers)?;
        let layer_shapes = crate::layer_shapes::LayerShapes::resolve(
            &arch,
            &heads_per_layer,
            &kv_heads_per_layer,
            ffn_per_layer.as_deref(),
            expert_ffn_dim,
            recurrent_layers.as_ref(),
        )?
        // `nanbeige.cpp:24-26` copies each physical layer's shape
        // arrays to every logical slot; HRM-Text's two stacks are
        // uniform and its arrays are scalars, so the replication is
        // spelled for the one schedule that needs it rather than
        // divided out of the other's counts.
        .replicated(match layer_loops {
            Some(crate::layer_loops::LayerLoops::Repeat { n_loops, .. }) => n_loops,
            _ => 1,
        });
        // The OTHER half of llama.cpp's dense-vs-MoE rule.
        // `ModelConfig::layer_is_dense` implements the leading-dense
        // prefix and not the `(il + 1) % n_moe_layer_step == 0` at
        // `src/models/ernie4-5-moe.cpp:64`, so a file whose step would
        // change the answer stops here rather than looking for expert
        // tensors on a layer that stores dense ones. `moe_interleave`
        // records what building the fixture found: llama.cpp cannot load
        // such a file either, because its own tensor loader has no step
        // in it.
        if let Some(reason) = crate::moe_interleave::interleave_step_refusal(
            &arch,
            metadata_u64_any(file, &[key("interleave_moe_layer_step")]),
        ) {
            return Err(LoadError::UnsupportedFeature(arch.clone(), reason));
        }

        // ik_llama.cpp's real gating-function hparam
        // (LLM_KV_EXPERT_GATING_FUNC: 1=softmax, 2=sigmoid) if the file
        // carries it; otherwise fall back to the same architecture-name
        // convention the hand-written presets in config.rs use (see
        // docs/MODELS.md for the citations behind that list).
        let gating_literal = GATING_LITERAL_ARCHITECTURES
            .iter()
            .find(|(name, _)| *name == arch)
            .map(|(_, g)| *g);
        let gating = match (
            gating_literal,
            metadata_u64_any(file, &[key("expert_gating_func")]),
        ) {
            (Some(literal), _) => literal,
            (None, Some(2)) => GatingFunction::Sigmoid,
            (None, Some(1)) => GatingFunction::Softmax,
            (None, _) => {
                if SIGMOID_GATING_ARCHITECTURES.contains(&arch.as_str()) {
                    GatingFunction::Sigmoid
                } else {
                    if is_moe {
                        best_effort_fields.push(
                            "moe.gating (no expert_gating_func key and architecture not in the known-sigmoid list; defaulted to softmax)",
                        );
                    }
                    GatingFunction::Softmax
                }
            }
        };

        // `{arch}.expert_weights_norm` (llama.cpp
        // `LLM_KV_EXPERT_WEIGHTS_NORM`) is the real metadata key for
        // whether the selected experts' weights are renormalised. Most
        // checkpoints do not carry it, which is why the fallback below
        // exists at all -- but when one does, the file's own answer wins
        // over an architecture-name guess.
        // The key only where llama.cpp reads it (`EXPERT_WEIGHTS_NORM_READERS`);
        // everywhere else the graph's literal, which the table below
        // transcribes, whatever the file says.
        let norm_key = if EXPERT_WEIGHTS_NORM_READERS.contains(&arch.as_str()) {
            file.metadata_bool(&key("expert_weights_norm"))
        } else {
            None
        };
        let norm_topk_prob = match norm_key {
            Some(v) => v,
            None => {
                // See `NO_TOPK_RENORMALIZE_ARCHITECTURES`'s doc comment:
                // an architecture-name lookup, the same convention
                // `gating`'s fallback above uses.
                if is_moe && matches!(gating, GatingFunction::Softmax) {
                    best_effort_fields.push(
                        "moe.norm_topk_prob (no expert_weights_norm key; defaulted by architecture-name lookup against NO_TOPK_RENORMALIZE_ARCHITECTURES)",
                    );
                }
                !NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&arch.as_str())
            }
        };

        // `{arch}.expert_weights_scale` (`LLM_KV_EXPERT_WEIGHTS_SCALE`).
        // llama.cpp's `build_moe_ffn` skips the multiply for both 0.0 and
        // 1.0, so both mean "no scaling" and both land on 1.0 here.
        // And only where llama.cpp reads it (`EXPERT_WEIGHTS_SCALE_READERS`).
        let expert_weights_scale = if EXPERT_WEIGHTS_SCALE_READERS.contains(&arch.as_str()) {
            metadata_f32_any(file, &[key("expert_weights_scale")])
                .filter(|s| *s != 0.0)
                .unwrap_or(1.0)
        } else {
            1.0
        };

        // Real GGUF key (`{arch}.attention.sliding_window`, confirmed
        // against `gguf-py/gguf/constants.py`'s real
        // `LLM_KV_ATTENTION_SLIDING_WINDOW`). Some checkpoints
        // (confirmed for real published Qwen1.5-MoE/Qwen2-MoE GGUFs)
        // carry a nonzero window value even when the model's own
        // config disables sliding-window attention entirely
        // (`use_sliding_window: false`) -- llama.cpp's own convention
        // is that a window of 0 means "unused," so only a real nonzero
        // value here is treated as active.
        let declared_window = metadata_u64_any(file, &[key("attention.sliding_window")]);
        let sliding_window = declared_window
            .map(|v| v as usize)
            .filter(|&w| w > 0)
            // `phi3` declares a window that llama.cpp deliberately does
            // NOT honour -- see `capability::swa_disabled_by_arch`. This
            // has to drop the window rather than pick a period, because
            // upstream is declining to use the file's value, not
            // choosing a different one.
            //
            // `block_count`, NOT `n_layers`: `exaone4.cpp:4` tests
            // `n_layer() == 64` at a point where `n_layer_nextn` has not
            // been read yet (`:18`), so a 64-trunk EXAONE-4 with an MTP
            // block appended sees 65 there and gets no window.
            //
            // `smallthinker` declares a window that llama.cpp REPLACES:
            // `smallthinker.cpp:8` assigns `n_swa = 4096` on the branch
            // the file's nonzero value selected. One table decides all
            // three answers (`capability::swa_window_override`), so a
            // row cannot be dropped by one reader and pinned by another.
            .and_then(
                |w| match crate::capability::swa_window_override(&arch, trunk.block_count) {
                    crate::capability::SwaWindowOverride::Honour => Some(w),
                    crate::capability::SwaWindowOverride::Drop => None,
                    crate::capability::SwaWindowOverride::Pin(pinned) => Some(pinned),
                },
            );

        // A CHUNKED window (`crate::chunked_swa`): the literal chunk on
        // the branch the file takes, whatever nonzero value it declares
        // and whether it declares one at all; a declared ZERO is the
        // branch libllama aborts on, refused there by name.
        let swa_chunked = crate::chunked_swa::chunked_window(&arch, declared_window)?;
        let sliding_window = swa_chunked.or(sliding_window);
        let swa_chunked = swa_chunked.is_some();

        // A window llama.cpp REQUIRES (`crate::swa_geometry::
        // window_required`): without it the file does not load upstream,
        // and here it would run with no layer rotated.
        if let (None, Some(line)) = (sliding_window, crate::swa_geometry::window_required(&arch)) {
            return Err(LoadError::UnsupportedFeature(
                arch.clone(),
                format!(
                    "`{arch}.attention.sliding_window` is absent or zero, and llama.cpp reads it \
                     as a REQUIRED key for this architecture (src/models/{line}); every real \
                     export writes it, and a file without it does not load upstream"
                ),
            ));
        }

        // A window on a short-conv architecture: `lfm2.cpp:24-29`
        // honours it on the ATTENTION layers alone (`is_swa_impl[il] =
        // !is_recr_impl[il]`), a per-layer answer `crate::swa_layers`
        // has no variant for, and one that would also arm eviction
        // against the history the conv indexes by row
        // (`crate::shortconv`). No published export writes the key.
        if let (Some(w), true) = (
            sliding_window,
            crate::shortconv::is_shortconv_architecture(&arch),
        ) {
            return Err(LoadError::UnsupportedFeature(
                arch.clone(),
                format!(
                    "`{arch}.attention.sliding_window` {w}: lfm2.cpp:24-29 windows the attention \
                     layers and not the conv layers, which `swa_layers` cannot yet spell, and a \
                     window on a conv layer would evict the history its convolution reads \
                     (`crate::shortconv`); no published LFM2 export writes the key"
                ),
            ));
        }

        // Three graphs rope their SLIDING layers with the scaling
        // switched off -- freq_scale = 1, ext_factor = 0, attn_factor =
        // 1 -- while the full-attention layers use the model's:
        // `olmo2` (Olmo-3), `mellum` and `laguna` (Laguna-XS.2), each
        // at the lines `crate::swa_geometry` cites. frink's
        // `RopeFreqs` already keeps their sliding layers' divisors
        // unscaled, but `rope_attn_factor` is one value for the whole
        // model, so honouring the file would mean rotating half the
        // layers at a magnitude the checkpoint never trained at.
        //
        // A window with NO scaling is not this case and is not refused:
        // both branches then reduce to the same plain RoPE, and the
        // difference is masking alone, which frink implements.
        if let (Some(lines), true) = (
            crate::swa_geometry::swa_layers_unscaled_rope(&arch),
            sliding_window.is_some(),
        ) {
            let scaling_type = file
                .metadata_str(&key("rope.scaling.type"))
                .unwrap_or("none")
                .to_string();
            if !scaling_type.eq_ignore_ascii_case("none") {
                return Err(LoadError::UnsupportedFeature(
                    arch.clone(),
                    format!(
                        "this {arch} checkpoint declares BOTH a sliding window and \
                         rope.scaling.type = \"{scaling_type}\". llama.cpp ropes the \
                         sliding layers with the scaling switched off (freq_scale = 1, \
                         ext_factor = 0, attn_factor = 1; {lines}) and the \
                         full-attention layers with it on, and frink carries one RoPE \
                         scaling for the whole model. A {arch} file with a window and no \
                         scaling, or with scaling and no window, is unaffected"
                    ),
                ));
            }
        }

        // WHICH LAYERS SLIDE (`attention.sliding_window_pattern`).
        //
        // llama.cpp seeds a period per architecture (gemma2 2, gemma3
        // 6, exaone4 4, ...) and only then reads the key, so a missing
        // key must NOT mean "all SWA": the seed is
        // `capability::default_swa_layout`, with the gemma3+ period for
        // any Gemma variant the table does not name. The phase is a
        // property of the architecture -- `dense_first` is an argument
        // to `set_swa_pattern`, not a GGUF key -- so it comes from the
        // seed either way.
        //
        // The key itself is a scalar period OR a per-layer bool array,
        // and which of the two an architecture's graph honours -- and
        // what it does with the other -- is `crate::swa_layers`'s
        // table, transcribed from the `get_key_or_arr` overload each
        // `load_arch_hparams` calls. The array used to be REFUSED here
        // for every architecture, which stopped every real EXAONE-4
        // 32B, EXAONE-MoE and Olmo-3 export at the door over a value
        // llama.cpp never reads for them.
        let swa_seed = crate::capability::default_swa_layout(&arch).or(match arch_profile.family {
            crate::capability::DecoderFamily::GemmaFamily => Some(crate::capability::SwaPattern {
                period: 6,
                dense_first: false,
            }),
            _ => None,
        });
        let swa_layers = match sliding_window {
            // No window: no graph consults `is_swa`, and reading the
            // key would only refuse a file over a value nothing uses.
            None => crate::swa_layers::SwaLayers::All,
            Some(_) => crate::swa_layers::read_swa_layers(
                file,
                &arch,
                &key("attention.sliding_window_pattern"),
                &trunk,
                swa_seed,
            )?,
        };

        // The metadata-declared scalar multipliers, resolved once for
        // whichever subset this architecture's reference graph applies.
        // See `crate::scalar_multipliers`; the keys the graph does NOT
        // apply were already refused above, by a list derived from the
        // same table. Read first because its `defaults` also seed the
        // softcap below: `grok.cpp:5-12` assigns all of them in one
        // place, and so does this.
        let multiplier_support = crate::scalar_multipliers::multiplier_support(&arch);

        // The file's softcap, then the architecture's default for a
        // file that declares none (`grok.cpp:9`), then llama.cpp's own
        // "off". `> 0.0` is what every graph tests before applying one.
        let attn_logit_softcap = metadata_f32_any(
            file,
            &[
                key("attention.logit_softcapping"),
                key("attn_logit_softcapping"),
            ],
        )
        .or(multiplier_support.defaults.attn_logit_softcap())
        .filter(|&v| v > 0.0);
        let final_logit_softcap =
            metadata_f32_any(file, &[key("final_logit_softcapping")]).filter(|&v| v > 0.0);

        let declared = crate::scalar_multipliers::DeclaredMultipliers {
            logit: metadata_f32_any(file, &[key("logit_scale")]),
            residual: metadata_f32_any(file, &[key("residual_scale")]),
            embedding: metadata_f32_any(file, &[key("embedding_scale")]),
            // Exactly the spelling this architecture's graph reads --
            // `attention.scale` for Granite, `attention.output_scale`
            // for Grok -- and nothing for the rest. The other spelling
            // was refused above, by the same table.
            attention: multiplier_support
                .attention
                .suffix()
                .and_then(|suffix| metadata_f32_any(file, &[key(suffix)])),
        };
        let multipliers = crate::scalar_multipliers::resolve(
            multiplier_support,
            declared,
            // `n_layer` and `n_embd` are here for MiniCPM's defaults
            // (`minicpm.cpp:6-7`), which are computed from the model's
            // own shape rather than declared: an older MiniCPM export
            // carries none of the three keys and is still scaled by all
            // three.
            crate::scalar_multipliers::MultiplierDims {
                head_dim,
                n_layer: n_layers,
                n_embd: hidden_dim,
            },
        )
        .map_err(|e| LoadError::UnsupportedFeature(arch.clone(), e.message(&arch)))?;

        // Gemma and afmoe: embeddings are scaled by sqrt(hidden_dim) at
        // input. That is ARITHMETIC, not a key -- those graphs read no
        // `embedding_scale` at all -- so it comes from the table and a
        // file declaring the key on one of them is refused above rather
        // than honoured. Granite's comes out of `{arch}.embedding_scale`.
        let embedding_scale =
            if crate::capability::embeddings_scaled_by_sqrt_n_embd(&arch, arch_profile.family) {
                Some((hidden_dim as f32).sqrt())
            } else {
                multipliers.embedding_scale
            };

        // llama.cpp's `f_attention_scale`, and ONLY where it differs from
        // the `1/sqrt(head_dim)` frink's attention kernels already
        // apply -- `Some` here means "pre-scale Q", so restating the
        // kernels' own scale would double-scale every score.
        //
        // For Gemma-2 and Gemma-3 that difference is real at 27B and
        // nowhere else (`capability::attention_scale_override` carries
        // the llama.cpp lines). This used to be a hardcoded `None` under
        // a comment that NAMED the 27B exception without implementing
        // it, so Gemma-2-27B scored 1.061x and Gemma-3-27B 1.146x too
        // large on every layer: a sharper softmax than the trained one,
        // fluent and wrong, with no error.
        //
        // Granite reaches the same slot from the file's own
        // `{arch}.attention.scale` (`granite.cpp:225`, whose `0.0f`
        // sentinel means "use the kernels' scale"). The two sources
        // cannot both be live on one architecture: `attention_scale_override`
        // covers the architectures that COMPUTE the scale and
        // `scalar_multipliers` the ones that READ it, and no llama.cpp
        // architecture does both. `.or` rather than a match because the
        // computed one is the one that cannot be turned off by a file.
        let attention_scale = crate::capability::attention_scale_override(
            &arch, n_layers, hidden_dim, n_heads, head_dim,
        )
        .or(multipliers.attention_scale);

        // Granite reads `{arch}.rope.scaling.finetuned` as a switch for
        // RoPE itself, not as a note about the scaling: a file declaring
        // it false runs UNROTATED in llama.cpp (every Granite-4.0 hybrid
        // export), which is `RopeLayers::Never` below
        // (`crate::rope_finetuned`).
        let rope_switched_off = crate::rope_finetuned::unrotated(
            &arch,
            file.metadata(&key("rope.scaling.finetuned"))
                .and_then(GgufValue::as_bool),
        );

        // OLMo-1 and DBRX clamp Q, K and V by `{arch}.attention.clamp_kqv`
        // inside the shared `build_qkv`. Resolved here for the
        // architectures whose loader reads the key, REQUIRED where
        // llama.cpp's is (`dbrx.cpp:5`), and applied by the one helper
        // every host body shares (`decoder/qkv_bias.rs`). See
        // `crate::clamp_kqv`, which also records that both converters
        // really write this key.
        let clamp_kqv = crate::clamp_kqv::resolve_clamp(
            &arch,
            metadata_f32_any(file, &[key("attention.clamp_kqv")]),
        )
        .map_err(|e| LoadError::UnsupportedFeature(arch.clone(), e.message(&arch)))?;

        // SWA-layer RoPE base. `llama_hparams` defaults it to 10000 and
        // the Gemma-3 lineage relies on that default; the architectures
        // in `swa_rope_base_follows_model` instead seed it from the
        // model's own base before the key can override.
        let rope_theta_swa = if sliding_window.is_some() {
            let fallback = if crate::capability::swa_rope_base_follows_model(&arch) {
                rope_theta
            } else {
                10_000.0
            };
            Some(
                metadata_f32_any(
                    file,
                    &[key("rope.freq_base_swa"), key("rope_freq_base_swa")],
                )
                .unwrap_or(fallback),
            )
        } else {
            None
        };

        let ffn_activation = match arch_profile.family {
            // Per-ARCHITECTURE first, because llama.cpp's choice is per
            // architecture and the family partition does not match it:
            // `grok` is StandardGqa and passes `LLM_FFN_GELU`.
            _ if crate::capability::uses_geglu(&arch) => crate::config::FfnActivation::Gelu,
            _ if crate::capability::uses_relu_sqr(&arch) => crate::config::FfnActivation::ReluSqr,
            _ if crate::capability::uses_gelu_ungated(&arch) => {
                crate::config::FfnActivation::GeluUngated
            }
            // The GATED ReLU (`ggml_reglu_split`), a real gate tensor:
            // NOT the row above, which aliases gate to up.
            _ if crate::capability::uses_reglu(&arch) => crate::config::FfnActivation::Reglu,
            // The four per-layer arrays travel IN the variant, read as
            // `apertus.cpp:6-9` reads them (`crate::act_layers`).
            _ if crate::act_layers::uses_xielu(&arch) => crate::config::FfnActivation::Xielu(
                crate::act_layers::read_xielu_layers(file, trunk.n_layers)?,
            ),
            // The two clamp arrays, read as `step35.cpp:28-29` read them
            // (optional; a file with neither is plain SwiGLU).
            _ if crate::act_layers::reads_swiglu_clamps(&arch) => {
                match crate::act_layers::read_swiglu_clamps(file, &arch, &trunk)? {
                    Some(clamps) => crate::config::FfnActivation::SwigluClamped(clamps),
                    None => crate::config::FfnActivation::Swiglu,
                }
            }
            crate::capability::DecoderFamily::GemmaFamily => crate::config::FfnActivation::Gelu,
            crate::capability::DecoderFamily::PhiFamily => {
                crate::config::FfnActivation::SwigluFused
            }
            _ => crate::config::FfnActivation::Swiglu,
        };

        // Llama 3/3.1/3.2's real per-band RoPE frequency correction: one
        // model-level tensor (`TENSOR_NOT_REQUIRED`, `TENSOR_DUPLICATED`
        // for every layer but the first in the real llama.cpp source --
        // i.e. every layer shares this same array), not per-layer. See
        // `frink_core::attention::apply_rope_with_freq_factors`'s doc
        // comment for why this matters.
        let rope_freqs = load_f32_vec_optional(file, "rope_freqs.weight")?;

        // Phi-3/Phi-4 LongRoPE: two per-band factor tensors instead of
        // Llama's single `rope_freqs.weight`, selected by context size
        // (llama.cpp `llama_model::get_rope_factors`: `rope_freqs` wins if
        // present, else `rope_long` when the run's context exceeds
        // `rope.scaling.original_context_length`, else `rope_short`).
        //
        // Provisional pick from the checkpoint's advertised context length
        // (llama.cpp's default `n_ctx`). The definitive pick happens in
        // `ModelConfig::apply_runtime_context`, called from `frink run`
        // (`--ctx-size`) and from `verify_engine::load_and_tokenize`
        // (`n_tokens + 8`, matching `tools/llama_logits.c`).
        let rope_orig_ctx = metadata_u64_any(file, &[key("rope.scaling.original_context_length")])
            .map(|v| v as usize);

        // The per-position attention temperature (`crate::attn_temperature`).
        // `mistral3.cpp:15` floors it on `hparams.n_ctx_orig_yarn`, which
        // `llama-model.cpp:1164-1165` seeds from `context_length` BEFORE
        // the YaRN key overrides it -- so a Ministral file with no YaRN
        // key floors on its context length, and the resolver is handed
        // that value rather than the key. Before this existed the key
        // loaded and was silently dropped on the one generic-path
        // architecture whose graph applies it.
        let attn_temperature = crate::attn_temperature::resolve_attn_temperature(
            &arch,
            crate::attn_temperature::DeclaredTemperature {
                scale: metadata_f32_any(file, &[key("attention.temperature_scale")]),
                length: metadata_u64_any(file, &[key("attention.temperature_length")]),
                n_ctx_orig_yarn: rope_orig_ctx
                    .map(|v| v as u64)
                    .or_else(|| metadata_u64_any(file, &[key("context_length")])),
            },
        )
        .map_err(|e| LoadError::UnsupportedFeature(arch.clone(), e.message(&arch)))?;
        // `rope_freqs.weight` outranks the LongRoPE pair (llama.cpp
        // `get_rope_factors` checks it first), so a checkpoint carrying
        // it never populates these and the runtime re-pick below cannot
        // overwrite a Llama-3 correction with a Phi one.
        let (rope_freqs_long, rope_freqs_short) = if rope_freqs.is_some() {
            (None, None)
        } else {
            (
                load_f32_vec_optional(file, "rope_factors_long.weight")?,
                load_f32_vec_optional(file, "rope_factors_short.weight")?,
            )
        };
        // Provisional pick from the checkpoint's own advertised context;
        // `ModelConfig::apply_runtime_context` re-picks once the run's
        // `--ctx-size` is known, which is the number llama.cpp decides on.
        let rope_freqs = match (rope_freqs, rope_orig_ctx) {
            (Some(f), _) => Some(f),
            (None, Some(orig)) => {
                let model_ctx = metadata_u64_any(file, &[key("context_length")])
                    .unwrap_or(orig as u64) as usize;
                if model_ctx > orig {
                    rope_freqs_long.clone().or_else(|| rope_freqs_short.clone())
                } else {
                    rope_freqs_short.clone().or_else(|| rope_freqs_long.clone())
                }
            }
            (None, None) => None,
        };

        // Partial rotary: the file's `rope.dimension_count`, or the head
        // width when absent -- llama.cpp's seeded `n_rot_full`
        // (`llama-model.cpp:1200-1202`).
        let rope_dim_seeded = metadata_u64_any(file, &[key("rope.dimension_count")])
            .map(|d| d as usize)
            .filter(|d| *d > 0)
            .unwrap_or(head_dim);

        // The sliding layers' OWN rotary and head widths
        // (`llama-model.cpp:1215-1223`). The rotary one is honoured --
        // `crate::swa_geometry` resolves the key and step35's halving
        // into the two widths `ModelConfig::layer_rope` hands out -- and
        // the head ones are refused, since frink carries one head width
        // in every cache. Only a model with a sliding layer reads the
        // keys; on one that has none they are dead metadata, as they
        // are upstream (`n_rot(il)` never takes the `_swa` branch). The
        // halving is not a key and applies regardless.
        let geometry = crate::swa_geometry::SwaGeometry {
            rope_dim_swa: metadata_u64_any(file, &[key("rope.dimension_count_swa")]),
            key_length_swa: metadata_u64_any(file, &[key("attention.key_length_swa")]),
            value_length_swa: metadata_u64_any(file, &[key("attention.value_length_swa")]),
            rope_dim_full: rope_dim_seeded as u64,
            head_dim: head_dim as u64,
        };
        let widths = if sliding_window.is_some()
            || crate::swa_geometry::full_layers_rotate_half(&arch).is_some()
        {
            if let Some(reason) = crate::swa_geometry::swa_geometry_refusal(&arch, geometry) {
                return Err(LoadError::UnsupportedFeature(arch.clone(), reason));
            }
            crate::swa_geometry::rotary_widths(&arch, geometry)
        } else {
            crate::swa_geometry::RotaryWidths {
                full: Some(rope_dim_seeded).filter(|d| *d < head_dim),
                swa: None,
            }
        };
        // Equal values mean "whole head", which is the same thing as
        // `None` and stays `None` so nothing downstream has to
        // special-case it.
        let rope_dim = widths.full;
        let rope_dim_swa = widths.swa;

        // See `ModelConfig::rope_attn_factor`. `mut` because YaRN's
        // magnitude term is folded into it below.
        let mut rope_attn_factor = metadata_f32_any(file, &[key("rope.scaling.attn_factor")])
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or(1.0);

        // YaRN long-context scaling. `rope.scaling.attn_factor` above is
        // only YaRN's *magnitude* term (ggml `rope_yarn`'s `mscale`); the
        // frequency half -- which bands get interpolated toward the
        // trained context and which stay extrapolated -- lives in
        // `rope.scaling.type` + `rope.scaling.factor`, and frink read
        // neither before this. A YaRN checkpoint was therefore roped as
        // if it declared no scaling at all: right near position 0 and
        // progressively wrong further in, i.e. the failure that reads as
        // long-prompt quality decay rather than as a bug.
        //
        // The rewrite is folded into `rope_freqs`, the same per-band
        // divisor array Llama-3's `rope_freqs.weight` supplies (ggml
        // divides each band's theta by it), so it rides the existing CPU
        // and Metal RoPE paths unchanged. When a file carries both, the
        // two corrections compose by multiplication, as they do in
        // llama.cpp (`ggml_rope_cache_init` divides by `freq_factors`
        // *and then* runs `rope_yarn`).
        // Linear scaling, which was silently DROPPED before this.
        //
        // `rope.scaling.type = "linear"` with factor s means rotating
        // position `p/s` instead of `p`. Since the angle is `p * freq`,
        // that is exactly `p * (freq / s)`, and `rope_freqs` already
        // divides each band's frequency. So a uniform vector of `s`
        // expresses it exactly and rides the existing CPU and Metal RoPE
        // paths unchanged, the same way YaRN does below.
        //
        // Before this, the type was compared against "yarn" and anything
        // else returned None, so a checkpoint declaring linear scaling
        // with factor 4 loaded and roped at UNSCALED positions where
        // llama.cpp divides them by 4. It answered as a different model
        // with no error. Affects the long-context community rescales
        // (`*-16k`, `*-32k` Llama-2 derivatives).

        // The file's own per-band factors, BEFORE any position-scaling
        // fold. That is what a sliding layer uses on an architecture
        // whose SWA layers do not inherit the trained scale -- llama.cpp
        // keeps the two apart as `freq_factors` (a tensor, the same for
        // every layer) and `freq_scale` (per layer,
        // `llama-model.cpp:2033`), while frink folds them into one
        // vector. See `config::RopeFreqs`.
        let rope_freqs_unscaled = rope_freqs.clone();

        let rope_freqs = match linear_scaling_from_gguf(file, &arch) {
            None => rope_freqs,
            Some(factor) => {
                let rotary_dim = rope_dim.unwrap_or(head_dim);
                if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
                    best_effort_fields.push(
                        "rope_freqs (linear scaling declared but the rotary width is odd; \
                         scaling not applied)",
                    );
                    rope_freqs
                } else {
                    let linear = vec![factor; rotary_dim / 2];
                    match rope_freqs {
                        None => Some(linear),
                        // Compose by multiplication, as a file carrying
                        // its own `rope_freqs.weight` tensor and a
                        // declared linear factor means both.
                        Some(own) if own.len() == linear.len() => {
                            Some(own.iter().zip(linear.iter()).map(|(a, b)| a * b).collect())
                        }
                        Some(own) => {
                            best_effort_fields.push(
                                "rope_freqs (linear scaling declared but the file's own \
                                 rope_freqs tensor has a different width; scaling not applied)",
                            );
                            Some(own)
                        }
                    }
                }
            }
        };
        let rope_freqs = match yarn_scaling_from_gguf(file, &arch, rope_orig_ctx) {
            None => rope_freqs,
            Some(scaling) => {
                // YaRN's MAGNITUDE half (`crate::yarn_magnitude`):
                // llama.cpp multiplies the rotated channels of q and k
                // by `get_mscale(factor, 1) / get_mscale(factor,
                // log_mul)` on top of `rope.scaling.attn_factor`
                // (`llama-context.cpp:196-231` with ggml's `rope_yarn`
                // term cancelled), and frink applied only the key.
                // Folded into the same field so it reaches the CPU
                // helper and the Metal `mscale` uniform through one
                // value. Gated on the same `Some(scaling)` as the
                // frequency half, so a file frink does not rewrite
                // (no `original_context_length`) takes neither half.
                rope_attn_factor *= crate::yarn_magnitude::yarn_attn_magnitude(
                    scaling.factor,
                    crate::yarn_magnitude::yarn_log_mul_for(
                        &arch,
                        metadata_f32_any(file, &[key("rope.scaling.yarn_log_multiplier")]),
                    ),
                );
                let rotary_dim = rope_dim.unwrap_or(head_dim);
                if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
                    best_effort_fields.push(
                        "rope_freqs (YaRN declared but the rotary width is odd; scaling not applied)",
                    );
                    rope_freqs
                } else {
                    let yarn =
                        frink_core::attention::yarn_freq_factors(scaling, rotary_dim, rope_theta);
                    match rope_freqs {
                        None => Some(yarn),
                        Some(own) if own.len() == yarn.len() => {
                            Some(own.iter().zip(yarn.iter()).map(|(a, b)| a * b).collect())
                        }
                        Some(own) => {
                            best_effort_fields.push(
                                "rope_freqs (YaRN declared alongside a per-band factor tensor of a \
                                 different width; the file's own tensor is used unscaled)",
                            );
                            Some(own)
                        }
                    }
                }
            }
        };

        // The SWA half of the split. llama.cpp defaults
        // `rope_freq_scale_train_swa` to `1.0f`
        // (`src/llama-hparams.h:129`) and only the architectures in
        // `swa_rope_scale_follows_model` assign it from
        // `rope_freq_scale_train`; `get_rope_freq_scale`
        // (`llama-model.cpp:2033-2035`) then picks between them per
        // layer. `gemma3.cpp` is not on that list and its converter
        // writes the FULL-ATTENTION factor
        // (`conversion/base.py:1222-1230`), so a Gemma-3 4B/12B/27B was
        // rotating five layers in six at `p/8` where llama.cpp rotates
        // at `p`.
        //
        // "No scaling" is spelled as an all-ones divisor vector, which
        // is what dividing by nothing is, so the sliding layers need no
        // second code path anywhere downstream.
        // With TWO rotary widths, one divisor vector cannot serve both
        // kinds of layer. `step35.cpp:247` passes NO factors to its
        // sliding layers (`crate::swa_geometry::swa_layers_drop_rope_
        // factors`), so for it the full layers take the first
        // `rope_dim/2` bands of the tensor -- ggml reads only that many
        // -- and the sliding layers divide by nothing at their own
        // width; every other architecture is refused by name, because
        // nothing upstream says which layers would take which.
        if rope_freqs.is_some() {
            if let Some(reason) =
                crate::swa_geometry::two_widths_with_factors_refusal(&arch, widths)
            {
                return Err(LoadError::UnsupportedFeature(arch.clone(), reason));
            }
        }
        let rope_freqs = rope_freqs
            .map(|full| -> Result<crate::config::RopeFreqs, LoadError> {
                if let Some(swa_width) = rope_dim_swa {
                    let full_width = rope_dim.unwrap_or(head_dim);
                    if full.len() < full_width / 2 {
                        return Err(LoadError::UnsupportedFeature(
                            arch.clone(),
                            format!(
                                "rope_freqs.weight has {} bands; the full-attention layers rotate \
                                 {full_width} dims and need {}",
                                full.len(),
                                full_width / 2
                            ),
                        ));
                    }
                    let full: Vec<f32> = full[..full_width / 2].to_vec();
                    return Ok(crate::config::RopeFreqs {
                        full,
                        swa: Some(vec![1.0; swa_width / 2]),
                    });
                }
                let swa = (sliding_window.is_some()
                    && !crate::capability::swa_rope_scale_follows_model(&arch))
                .then(|| rope_freqs_unscaled.unwrap_or_else(|| vec![1.0; full.len()]))
                .filter(|swa| *swa != full);
                Ok(crate::config::RopeFreqs { full, swa })
            })
            .transpose()?;

        // RoPE layout comes from the capability registry above (fail-
        // closed). Getting this wrong for `llama` (needs Norm) was the
        // real root cause of the Llama-3.1-8B early-stop/wrong-logits bug.

        if best_effort_fields.is_empty() {
            best_effort_fields.push(
                "none -- every field above was read directly from this file's own GGUF metadata",
            );
        }

        // LAST, deliberately. The generic path is a GUESS, so it has to
        // be opted into rather than fallen onto: it assumes plain GQA
        // with no ALiBi, no learned position embeddings and no
        // per-layer rope skipping, and that assumption was already
        // wrong for gpt2, mpt, refact, bloom and jais.
        //
        // But it runs AFTER every architecture-specific refusal, so a
        // checkpoint with a NAMED problem still reports that problem.
        // Checking first would have replaced "this uses ALiBi" with
        // "this is unaudited", which is true and much less useful.
        if matches!(
            arch_profile.path,
            crate::capability::ArchPath::GenericGqa { .. }
        ) && !crate::capability::is_audited_generic(&arch)
            && !matches!(
                std::env::var("FRINK_ALLOW_UNAUDITED_ARCH").ok().as_deref(),
                Some("1") | Some("true") | Some("on")
            )
        {
            return Err(LoadError::UnauditedArchitecture(
                arch.clone(),
                rope_layout,
                crate::capability::unaudited_refusal_detail(&arch),
            ));
        }

        Ok(ModelConfig {
            name,
            n_layers,
            n_mtp_blocks: trunk.n_mtp_blocks,
            layer_loops,
            skip_stream: crate::skip_stream::has_skip_stream(&arch),
            parallel_ssm: crate::mamba2::parallel_with_attention(&arch),
            swa_chunked,
            weightless_qk_norm: crate::weightless_qk_norm::weightless_qk_norm(&arch, n_experts),
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            v_head_dim,
            vocab_size,
            rope_theta,
            rms_norm_eps,
            // `crate::norm::POST_NORM_EPS_LITERAL`: the architecture's, which
            // is the model's for all but one graph of 155.
            post_norm_eps: crate::norm::post_norm_eps(&arch, rms_norm_eps),
            // No GGUF file encodes a hybrid KDA/Gated-MLA attention
            // topology today; every real checkpoint loaded this way
            // runs the standard Gqa path.
            attention: crate::config::AttentionKind::Gqa,
            sliding_window,
            swa_layers,
            // llama.cpp's per-layer `use_rope`. Fed the POST-gate window
            // answer (`sliding_window`, not the raw key), because
            // `exaone4` decides both off the same layer count and the
            // two must not be able to disagree.
            rope_layers: if rope_switched_off {
                crate::rope_layers::RopeLayers::Never
            } else if let Some(mask) = rope_pattern.clone() {
                // The FILE's answer, for the one architecture that
                // reads the key (`rope_layers::ROPE_PATTERN_READERS`).
                crate::rope_layers::RopeLayers::FileMask(mask)
            } else {
                crate::rope_layers::rope_layers(
                    &arch,
                    n_layers,
                    sliding_window.is_some(),
                    n_dense_leading_layers,
                )
            },
            router_input: crate::router_input::router_input(&arch),
            block_sub_norms: crate::sub_norms::block_sub_norms(&arch),
            parallel_residual: crate::parallel_residual::model_has_parallel_layer(
                file, &arch, n_layers,
            ),
            learned_positions: crate::position_embd::learned_positions(&arch),
            attn_value_scale: crate::attn_value_scale::resolve_attn_value_scale(
                &arch,
                file.metadata_f32(&key("attention.value_scale")),
            ),
            alibi_max_bias: crate::alibi::max_alibi_bias(
                &arch,
                n_layers,
                file.metadata_f32(&key("attention.max_alibi_bias")),
            ),
            layer_shapes,
            moe: MoeLayerConfig {
                n_experts: n_experts.max(1),
                n_experts_active,
                n_shared_experts,
                hidden_dim,
                expert_ffn_dim,
                gating,
                norm_topk_prob,
                expert_group_count: metadata_u64_any(file, &[key("expert_group_count")])
                    .map(|v| v as usize)
                    .filter(|&c| c > 1),
                expert_group_used_count: metadata_u64_any(file, &[key("expert_group_used_count")])
                    .map(|v| v as usize)
                    .filter(|&c| c > 0),
                expert_weights_scale,
                routed_weight_before_ffn: crate::routed_weight_site::weight_before_ffn(&arch),
            },
            n_dense_leading_layers,
            moe_interleave_step,
            norm_function,
            rope_freqs,
            rope_layout,
            qk_norm_style,
            attn_logit_softcap,
            final_logit_softcap,
            embedding_scale,
            residual_scale: multipliers.residual_scale,
            normed_residual_scale: multipliers.normed_residual_scale,
            clamp_kqv,
            attn_temperature,
            logit_multiplier: multipliers.logit_multiplier,
            attention_scale,
            rope_attn_factor,
            rope_dim,
            rope_dim_swa,
            rope_freqs_long,
            rope_freqs_short,
            rope_orig_ctx,
            rope_theta_swa,
            ffn_activation,
            best_effort_fields: Box::leak(best_effort_fields.into_boxed_slice()),
        })
    }
}

impl crate::sampling::RecommendedSampling {
    /// The sampling a GGUF recommends for itself, from the
    /// `general.sampling.*` metadata keys llama.cpp's converter writes
    /// when the source checkpoint carried a `generation_config.json`.
    ///
    /// This is the GGUF half of FreeToken's `load_generation_sampling`
    /// (`python/freetoken/utils/hf.py:92`), which checks the GGUF
    /// metadata *first* and only falls back to a `generation_config.json`
    /// sidecar for non-GGUF checkpoints -- a GGUF is a single file and
    /// has no sidecar to read.
    ///
    /// Key names are llama.cpp's own (`general.sampling.temp`, not
    /// `temperature`). Each key is independent: a file that names only
    /// `top_k` recommends only `top_k`, and the two fields it did not
    /// mention stay `None` so the server's own defaults keep speaking
    /// for them.
    ///
    /// `temp` / `top_p` are read as float *or* integer, because a
    /// converter that wrote `temp = 1` stores a GGUF integer and
    /// dropping that value would silently serve the checkpoint greedy --
    /// the exact repetition-loop failure the recommendation exists to
    /// prevent.
    pub fn from_gguf(file: &impl TensorSource) -> Self {
        let number = |k: &str| -> Option<f32> {
            file.metadata(k)
                .and_then(|v| v.as_f32().or_else(|| v.as_u64().map(|u| u as f32)))
        };
        crate::sampling::RecommendedSampling {
            temperature: number("general.sampling.temp"),
            top_p: number("general.sampling.top_p"),
            top_k: file
                .metadata("general.sampling.top_k")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        }
    }
}

/// The `linear` RoPE scaling factor, if this file declares one.
///
/// Deliberately separate from [`yarn_scaling_from_gguf`]: YaRN needs an
/// original context length and per-band betas, and linear needs neither.
/// Any factor at or below one is not a correction, and is treated as
/// absent rather than applied as a no-op.
fn linear_scaling_from_gguf(file: &impl TensorSource, arch: &str) -> Option<f32> {
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let scaling_type = file.metadata_str(&key("rope.scaling.type"))?;
    if !scaling_type.eq_ignore_ascii_case("linear") {
        return None;
    }
    metadata_f32_any(file, &[key("rope.scaling.factor")]).filter(|f| f.is_finite() && *f > 1.0)
}

/// The YaRN RoPE scaling a GGUF declares, or `None` when this file
/// declares none that changes the rotation.
///
/// llama.cpp's key names (`llama-arch.cpp`
/// `LLM_KV_ROPE_SCALING_TYPE` / `_FACTOR`): `<arch>.rope.scaling.type`
/// is a string (`"none"`, `"linear"`, `"yarn"`, `"longrope"`) and
/// `<arch>.rope.scaling.factor` the ratio of served to trained context.
/// `beta_fast` / `beta_slow` are read from both the plain and the
/// `yarn_`-prefixed spelling and otherwise fall back to the reference's
/// own defaults (32.0 / 1.0), which is what a real checkpoint relies on
/// -- almost none of them write those two keys.
///
/// `None` is returned for every case where applying YaRN would be a
/// guess or a no-op rather than a correction, so that no checkpoint's
/// rotation moves without the file having asked for it:
///
/// * a scaling type other than `yarn` (`linear` divides positions,
///   `longrope` rides the `rope_factors_long`/`_short` tensors this
///   loader already reads -- neither is this rewrite, and treating them
///   as YaRN would rope them wrong in a *new* way instead of leaving
///   them as they are),
/// * a missing, non-finite or `<= 1.0` factor (the reference's own
///   `get_mscale` treats `scale <= 1` as unscaled, and a factor of 1.0
///   makes every band's divisor exactly 1.0 anyway),
/// * a missing `rope.scaling.original_context_length` -- the trained
///   context is what the correction range is measured against, and
///   inventing one (say, from `context_length`, which on a YaRN file is
///   the *extended* length) would put the ramp in the wrong place and
///   quietly rope the checkpoint at frequencies nobody trained.
pub(crate) fn yarn_scaling_from_gguf(
    file: &impl TensorSource,
    arch: &str,
    orig_ctx: Option<usize>,
) -> Option<frink_core::attention::YarnScaling> {
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let scaling_type = file.metadata_str(&key("rope.scaling.type"))?;
    if !scaling_type.eq_ignore_ascii_case("yarn") {
        return None;
    }
    let factor = metadata_f32_any(file, &[key("rope.scaling.factor")])
        .filter(|f| f.is_finite() && *f > 1.0)?;
    let orig_max_pos = orig_ctx?;
    let beta = |suffix: &str, default: f32| -> f32 {
        metadata_f32_any(
            file,
            &[
                key(&format!("rope.scaling.{suffix}")),
                key(&format!("rope.scaling.yarn_{suffix}")),
            ],
        )
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
    };
    // `llama-hparams.h:137` seeds `yarn_beta_fast = 32.0f` for every
    // architecture and `grok.cpp:5` reseeds it to 8.0 before the key is
    // read; the table that holds Grok's other defaults holds that one
    // too, so it is not a second literal here.
    let beta_fast_default = crate::scalar_multipliers::multiplier_support(arch)
        .defaults
        .yarn_beta_fast()
        .unwrap_or(32.0);
    Some(frink_core::attention::YarnScaling {
        factor,
        beta_fast: beta("beta_fast", beta_fast_default),
        beta_slow: beta("beta_slow", 1.0),
        orig_max_pos,
        // No GGUF key carries the reference's `truncate` flag, and its
        // default is `true`; a file that wanted the fractional range
        // would have no way to say so here.
        truncate: true,
    })
}

pub(crate) fn find_info<'a>(
    file: &'a impl TensorSource,
    name: &str,
) -> Result<&'a TensorInfo, LoadError> {
    file.find_tensor(name)
        .ok_or_else(|| LoadError::Gguf(GgufError::TensorNotFound(name.to_string())))
}

/// Like `load_f32_vec`, but for tensors that only exist on some
/// checkpoints (e.g. `attn_q_norm`/`attn_k_norm` -- OLMoE-style
/// per-projection QK-RMSNorm applied to the full q_proj/k_proj output
/// before RoPE, confirmed against `OlmoeAttention.forward` in
/// `transformers/models/olmoe/modeling_olmoe.py`: `q_norm(q_proj(x))`,
/// `k_norm(k_proj(x))`, both plain RMSNorm over the whole projected
/// width, not per-head). Absent for every other preset/fixture this
/// loader already handles -- `None` there is correct, not a missing
/// feature.
/// Loads the four gpt-oss-only side-table tensors for one layer, and
/// checks that the fifth, the attention sinks, was loaded onto the
/// layer's [`AttnWeights`] by the generic tensor-presence read.
///
/// Every one of them is **required**: a gpt-oss checkpoint that is
/// missing any of these is not a gpt-oss checkpoint frink can run, and
/// quietly substituting zeros would reintroduce exactly the
/// silently-wrong-graph failure this path exists to remove. The lengths
/// are asserted against the config for the same reason -- a bias of the
/// wrong width would otherwise be applied to a `zip`-truncated prefix
/// and produce a plausible, wrong answer.
///
/// Shapes follow `src/models/openai-moe.cpp::load_arch_tensors`:
/// `attn_sinks {n_head}` (`:44`, flags `0`, so REQUIRED there too),
/// `attn_output.bias {n_embd}`, `ffn_gate_inp.bias {n_expert}`,
/// `ffn_{gate,up}_exps.bias {n_ff_exp, n_expert}`,
/// `ffn_down_exps.bias {n_embd, n_expert}`. GGUF stores the fastest
/// dimension first, so the 2-D bias tensors arrive expert-major and
/// split by simple chunking.
fn load_gpt_oss_layer(
    file: &impl TensorSource,
    l: usize,
    config: &ModelConfig,
    sinks_loaded: bool,
) -> Result<crate::decoder::GptOssLayer, LoadError> {
    let n_experts = config.moe.n_experts;
    let ff = config.moe.expert_ffn_dim;

    let want = |name: &str, got: usize, expect: usize| -> Result<(), LoadError> {
        if got == expect {
            Ok(())
        } else {
            Err(LoadError::UnsupportedFeature(
                config.name.to_string(),
                format!("{name} has {got} elements, expected {expect}"),
            ))
        }
    };

    if !sinks_loaded {
        return Err(LoadError::UnsupportedFeature(
            config.name.to_string(),
            format!(
                "blk.{l}.attn_sinks.weight is missing; gpt-oss requires it \
                 (src/models/openai-moe.cpp:44) and llama.cpp refuses the file without it"
            ),
        ));
    }
    // `attn_output.bias` is `AttnWeights::o_bias` now, read by
    // `crate::proj_bias` (gpt-oss is a REQUIRED row of its table).
    let router_bias = load_f32_vec(file, &format!("blk.{l}.ffn_gate_inp.bias"))?;
    want(
        &format!("blk.{l}.ffn_gate_inp.bias"),
        router_bias.len(),
        n_experts,
    )?;

    let gate_b = load_f32_vec(file, &format!("blk.{l}.ffn_gate_exps.bias"))?;
    want(
        &format!("blk.{l}.ffn_gate_exps.bias"),
        gate_b.len(),
        n_experts * ff,
    )?;
    let up_b = load_f32_vec(file, &format!("blk.{l}.ffn_up_exps.bias"))?;
    want(
        &format!("blk.{l}.ffn_up_exps.bias"),
        up_b.len(),
        n_experts * ff,
    )?;
    let down_b = load_f32_vec(file, &format!("blk.{l}.ffn_down_exps.bias"))?;
    want(
        &format!("blk.{l}.ffn_down_exps.bias"),
        down_b.len(),
        n_experts * config.hidden_dim,
    )?;

    let expert_bias = (0..n_experts)
        .map(|e| frink_moe::ExpertBias {
            gate: gate_b[e * ff..(e + 1) * ff].to_vec(),
            up: up_b[e * ff..(e + 1) * ff].to_vec(),
            down: down_b[e * config.hidden_dim..(e + 1) * config.hidden_dim].to_vec(),
        })
        .collect();

    Ok(crate::decoder::GptOssLayer {
        router_bias,
        expert_bias,
    })
}

/// `blk.N.attn_sinks.weight` when the file carries it, checked to be
/// one logit per query head of THIS layer (`{n_head}` in every graph
/// that creates it: `openai-moe.cpp:44`, `mimo2.cpp:58`).
///
/// Optional here because that is what the tensor's consumers make it:
/// `build_attn_mha` takes a nullable `sinks` and `mimo2.cpp:58` creates
/// it `TENSOR_NOT_REQUIRED`. gpt-oss, which requires it, checks the
/// result where its side table loads.
fn load_attn_sinks(
    file: &impl TensorSource,
    l: usize,
    n_heads: usize,
) -> Result<Option<Vec<f32>>, LoadError> {
    let name = format!("blk.{l}.attn_sinks.weight");
    let Some(sinks) = load_f32_vec_optional(file, &name)? else {
        return Ok(None);
    };
    if sinks.len() != n_heads {
        return Err(LoadError::UnsupportedFeature(
            name,
            format!(
                "attention sinks are one logit per query head; this layer has {n_heads} heads \
                 and the tensor {} entries",
                sinks.len()
            ),
        ));
    }
    Ok(Some(sinks))
}

pub(crate) fn load_f32_vec_optional(
    file: &impl TensorSource,
    name: &str,
) -> Result<Option<Vec<f32>>, LoadError> {
    if file.find_tensor(name).is_none() {
        return Ok(None);
    }
    Ok(Some(load_f32_vec(file, name)?))
}

/// Slice `n` rows starting at `start` out of a quantized matrix without
/// dequantizing: every `Quantized` kind stores one interleaved block
/// buffer per row (fixed `row_bytes`), so a row range is a contiguous
/// byte range. Mapped sources stay zero-copy (sub-range of the same
/// mmap); other backings get an owned copy. Returns `None` for non-
/// quantized matrices (F32 / MXFP4) -- callers fall back to dequant.
pub(crate) fn slice_quantized_rows(
    m: &WeightMatrix,
    start: usize,
    n: usize,
) -> Option<WeightMatrix> {
    // A folded fused projection splits into folded parts sharing ONE
    // fold: the input transform is the same for q, k and v, and
    // `apply_gpu_multi` recognises the shared `Arc`.
    if let WeightMatrix::Folded { base, fold } = m {
        let mut part = slice_quantized_rows(base, start, n)?;
        part.fold_hadamard(fold.clone());
        return Some(part);
    }
    let WeightMatrix::Quantized {
        data,
        rows,
        cols,
        kind,
    } = m
    else {
        return None;
    };
    let total = data.len();
    if *rows == 0 || total % *rows != 0 || start + n > *rows {
        return None;
    }
    let row_bytes = total / *rows;
    let (b0, b1) = (start * row_bytes, (start + n) * row_bytes);
    let bytes = match data {
        WeightBytes::Mapped { mmap, range } => WeightBytes::Mapped {
            mmap: mmap.clone(),
            range: range.start + b0..range.start + b1,
        },
        other => WeightBytes::Owned(other.as_slice()[b0..b1].to_vec()),
    };
    Some(WeightMatrix::Quantized {
        data: bytes,
        rows: n,
        cols: *cols,
        kind: *kind,
    })
}

/// Dense-layer FFN tensors: standard gate/up/down, Phi-3 fused
/// `ffn_up` with `2 * ffn_dim` rows and no separate gate, the UNGATED
/// two-matrix FFN (`FfnActivation::ReluSqr`), or nothing at all for an
/// FFN-free layer (`ffn_dim == 0`).
///
/// `ffn_dim` is THIS layer's width (`ModelConfig::layer_shape`), which
/// is the model's for every architecture but the per-layer ones.
fn load_dense_expert(
    file: &impl TensorSource,
    layer: usize,
    config: &ModelConfig,
    ffn_dim: usize,
) -> Result<ExpertWeights, LoadError> {
    if ffn_dim == 0 {
        return Ok(crate::layer_shapes::absent_ffn(config.hidden_dim));
    }
    let gate_name = format!("blk.{layer}.ffn_gate.weight");
    let up_name = format!("blk.{layer}.ffn_up.weight");
    let down_name = format!("blk.{layer}.ffn_down.weight");
    if config.ffn_is_ungated() {
        // `arcee.cpp:39-40` and `apertus.cpp:45-46` create `ffn_up` and
        // `ffn_down` and no gate; a file carrying one describes a graph
        // this architecture does not compute, and would otherwise be
        // left as an unread tensor with a less specific message.
        if file.find_tensor(&gate_name).is_some() {
            return Err(LoadError::UnsupportedFeature(
                config.name.to_string(),
                format!(
                    "{gate_name} is present but this architecture's FFN is ungated \
                     ({:?}: LLM_FFN_RELU_SQR under LLM_FFN_SEQ with a null gate, \
                     arcee.cpp:123-128, or ggml_xielu over ffn_up alone, apertus.cpp:129-142)",
                    config.ffn_activation
                ),
            ));
        }
        let up = load_weight_matrix(file, &up_name)?;
        if up.rows() != ffn_dim {
            return Err(LoadError::UnsupportedFeature(
                config.name.to_string(),
                format!(
                    "{up_name} has {} rows; the ungated FFN expects feed_forward_length = \
                     {ffn_dim}",
                    up.rows()
                ),
            ));
        }
        // The alias: the same tensor read again. A zero-copy view of the
        // same bytes for a quantized mmapped file; an owned widening for
        // an F32/F16 one. See `FfnActivation::ReluSqr` for why the pair
        // is aliased rather than the struct given an `Option`.
        return Ok(ExpertWeights {
            gate: load_weight_matrix(file, &up_name)?,
            up,
            down: load_weight_matrix(file, &down_name)?,
        });
    }
    if file.find_tensor(&gate_name).is_some() {
        return Ok(ExpertWeights {
            gate: load_weight_matrix(file, &gate_name)?,
            up: load_weight_matrix(file, &up_name)?,
            down: load_weight_matrix(file, &down_name)?,
        });
    }
    // Phi-3 fused SwiGLU: up is [hidden, 2*ff], first half gate, second up.
    let fused = load_weight_matrix(file, &up_name)?;
    let ff = ffn_dim;
    if fused.rows() != 2 * ff {
        return Err(LoadError::UnsupportedFeature(
            config.name.to_string(),
            format!(
                "{up_name} has {} rows without a companion ffn_gate; \
                 expected fused SwiGLU with 2*ffn_dim = {} rows",
                fused.rows(),
                2 * ff
            ),
        ));
    }
    let cols = fused.cols();
    // Quantized fused gate+up: split by rows, no dequant (Metal-capable).
    if let (Some(gate), Some(up)) = (
        slice_quantized_rows(&fused, 0, ff),
        slice_quantized_rows(&fused, ff, ff),
    ) {
        return Ok(ExpertWeights {
            gate,
            up,
            down: load_weight_matrix(file, &down_name)?,
        });
    }
    let mut full = Vec::with_capacity(fused.rows() * cols);
    for r in 0..fused.rows() {
        full.extend_from_slice(&fused.dequant_row(r));
    }
    let gate = WeightMatrix::F32(Tensor::new(full[..ff * cols].to_vec(), vec![ff, cols]));
    let up = WeightMatrix::F32(Tensor::new(full[ff * cols..].to_vec(), vec![ff, cols]));
    Ok(ExpertWeights {
        gate,
        up,
        down: load_weight_matrix(file, &down_name)?,
    })
}

/// Widen a raw plain-float tensor (`F32` / `F16` / `BF16`) to `f32`.
///
/// The three unquantized element types are handled identically at every
/// call site (eager widening to an owned buffer -- none of them has a
/// block structure a fused dot kernel could exploit), and each of the
/// seven GGUF loaders used to inline the same two-way match. F16 had no
/// arm in any of them, which made every `*-f16.gguf` a hard
/// `UnsupportedDtype` even though the type was parsed and sized.
pub(crate) fn widen_plain_float(
    dtype: GgmlType,
    raw: &[u8],
    name: &str,
) -> Result<Vec<f32>, LoadError> {
    match dtype {
        GgmlType::F32 => {
            let mut out = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes(*chunk));
            }
            Ok(out)
        }
        GgmlType::F16 => frink_quant::dequant_f16(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::F16)),
        GgmlType::BF16 => frink_quant::dequant_bf16(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::BF16)),
        // MXFP4 is accepted as a weight matrix and as an MoE expert
        // tensor, and `WeightMatrix::dequant` already calls this
        // dequantizer, so refusing it here made a 1-D MXFP4 norm or
        // bias a hard load error on a checkpoint whose 2-D tensors of
        // the same type load fine. That contradicted this function's
        // own contract, which is to widen whatever the loaders accept.
        GgmlType::MXFP4 => frink_quant::dequant_mxfp4_gguf(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::MXFP4)),
        other => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
    }
}

pub(crate) fn load_f32_vec(file: &impl TensorSource, name: &str) -> Result<Vec<f32>, LoadError> {
    let info = find_info(file, name)?;
    let raw = file.tensor_bytes(name)?;
    match info.dtype {
        // MXFP4 rides with the plain floats because `widen_plain_float`
        // is where its arm already lives -- routing it here rather than
        // giving this table its own `dequant_mxfp4_gguf` call keeps ONE
        // MXFP4 arm in this file instead of two that can drift.
        //
        // It has to be in *both* tables' reach, and it was in neither's:
        // `load_weight_matrix` accepts MXFP4 as a 2-D weight and
        // `load_moe_expert_matrices` accepts it as an expert tensor, so
        // a checkpoint whose norms happen to be MXFP4 failed here with
        // `UnsupportedDtype` while its far larger tensors of the exact
        // same dtype loaded fine.
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::MXFP4 => {
            widen_plain_float(info.dtype, raw, name)
        }
        GgmlType::Q8_0 => frink_quant::dequant_q8_0(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q8_0)),
        GgmlType::Q4_0 => frink_quant::dequant_q4_0(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q4_0)),
        GgmlType::Q4K => frink_quant::dequant_q4_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q4K)),
        GgmlType::Q5K => frink_quant::dequant_q5_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q5K)),
        GgmlType::Q6K => frink_quant::dequant_q6_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q6K)),
        GgmlType::Q2K => frink_quant::dequant_q2_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q2K)),
        GgmlType::Q3K => frink_quant::dequant_q3_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q3K)),
        GgmlType::Q4_1 => frink_quant::dequant_q4_1(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q4_1)),
        GgmlType::Q5_0 => frink_quant::dequant_q5_0(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q5_0)),
        GgmlType::Q5_1 => frink_quant::dequant_q5_1(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q5_1)),
        GgmlType::Q8_1 => frink_quant::dequant_q8_1(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q8_1)),
        GgmlType::IQ4NL => frink_quant::dequant_iq4_nl(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ4NL)),
        GgmlType::IQ4XS => frink_quant::dequant_iq4_xs(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ4XS)),
        // The codebook-grid tiers. Rare on the 1-D tensors this
        // function widens (norms and biases are almost always F32),
        // but a dtype frink can decode should never be rejected here
        // just because the *other* dispatch table below knows it --
        // that split is how a supported format turns into a load
        // failure on the one checkpoint that uses it.
        GgmlType::IQ1S => frink_quant::dequant_iq1_s(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ1S)),
        GgmlType::IQ1M => frink_quant::dequant_iq1_m(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ1M)),
        GgmlType::IQ2XXS => frink_quant::dequant_iq2_xxs(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ2XXS)),
        GgmlType::IQ2XS => frink_quant::dequant_iq2_xs(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ2XS)),
        GgmlType::IQ2S => frink_quant::dequant_iq2_s(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ2S)),
        GgmlType::IQ3XXS => frink_quant::dequant_iq3_xxs(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ3XXS)),
        GgmlType::IQ3S => frink_quant::dequant_iq3_s(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ3S)),
        other => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
    }
}

/// Loads a 2D weight matrix, keeping Q8_0/Q4_0 tensors quantized (raw
/// bytes copied out, never dequantized) and only expanding truly F32
/// tensors. This is the memory- and bandwidth-saving path: for a
/// multi-billion-parameter checkpoint the difference between this and
/// "dequant everything on load" is the difference between fitting in
/// RAM and not.
pub(crate) fn load_weight_matrix(
    file: &impl TensorSource,
    name: &str,
) -> Result<WeightMatrix, LoadError> {
    let mut m = load_weight_matrix_unfolded(file, name)?;
    // A PrismML checkpoint folds a Hadamard rotation into the listed
    // weights (`crate::hadamard_fold`); the matrix carries the
    // activation-side transform so every `apply` undoes it.
    if let Some(fold) = crate::hadamard_fold::fold_for(file, name, m.cols())? {
        m.fold_hadamard(fold);
    }
    Ok(m)
}

fn load_weight_matrix_unfolded(
    file: &impl TensorSource,
    name: &str,
) -> Result<WeightMatrix, LoadError> {
    let info = find_info(file, name)?;
    // GGUF's on-disk `ne[]` shape array is fastest-varying-dimension-first
    // (ggml convention), i.e. `[in_features, out_features]` for a 2D
    // weight matrix -- the *reverse* of the row-major `[rows, cols]` =
    // `[out_features, in_features]` order `WeightMatrix`/`matmul_f32`
    // need. Reversed here once so every consumer below gets the correct
    // orientation. Before this reversal existed, every 2D tensor in an
    // externally-produced GGUF file was silently loaded transposed -- a
    // real bug found by running a real downloaded checkpoint
    // (TinyLlama-1.1B-Chat, e.g. `attn_k.weight`'s real raw shape is
    // `[2048, 256]` = `[hidden_dim, kv_dim]` = `[in, out]`) -- found
    // as a real transposition bug affecting every externally-produced
    // GGUF file, caught by serving a real downloaded checkpoint.
    let shape: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
    // A ggml tensor's `ne[]` is always four long and trailing 1s are
    // implicit, so a GGUF writer is free to store a `[in, 1]` matrix
    // with `n_dims = 1`. llama.cpp reads it back as a matrix anyway --
    // `check_tensor_dims` compares each requested dimension against
    // `cur->ne[i]` and requires 1 for the dimensions the file does not
    // carry -- so a single-output projection is a 2-D weight there and
    // must be one here. Refusing it instead made a real checkpoint
    // unloadable: `cross-encoder/ms-marco-MiniLM-L6-v2` writes
    // `cls.output.weight` as `[384]`, i.e. the 1x384 relevance head
    // that `/v1/rerank` exists to run, and the whole route died at load
    // with "expected 2D".
    let (rows, cols) = match shape.as_slice() {
        [r, c] => (*r, *c),
        [c] => (1, *c),
        other => {
            return Err(LoadError::UnsupportedDtype(
                format!("{name} (expected 2D, got shape {other:?})"),
                info.dtype,
            ))
        }
    };
    // `shape` is what the file said; `[rows, cols]` is what the matrix
    // is. They differ exactly in the 1-D case above, and the `Tensor`
    // must carry the matrix shape or `apply` reads it as a vector.
    let shape = vec![rows, cols];

    match info.dtype {
        // BF16 has no block/scale structure to keep quantized-in-place
        // the way Q4_0/Q8_0/K-quants do -- there's no fused dot kernel
        // that would make sense for a plain narrowed float, so it's
        // eagerly widened to an owned f32 Tensor exactly like F32
        // tensors already are.
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            let data = load_f32_vec(file, name)?;
            Ok(WeightMatrix::F32(Tensor::new(data, shape)))
        }
        other => match quant_kind_for(other) {
            Some(kind) => {
                let (mmap, range) = file.tensor_mapped_range(name)?;
                #[cfg(feature = "metal")]
                frink_metal::gpu::register_weight_mmap(Arc::clone(&mmap));
                Ok(WeightMatrix::Quantized {
                    data: WeightBytes::Mapped { mmap, range },
                    rows,
                    cols,
                    kind,
                })
            }
            None => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
        },
    }
}

/// Splits a packed 3D MoE expert tensor `blk.N.ffn_{gate,up,down}_exps.weight`
/// (shape `[n_experts, out_dim, in_dim]`) into per-expert `WeightMatrix`es,
/// slicing raw bytes directly (quantized tensors stay quantized; block
/// boundaries never cross expert boundaries since `in_dim` is a whole
/// number of quantization blocks). Matches llama.cpp/ik_llama.cpp layout
/// confirmed on real OLMoE and Qwen2-MoE GGUF checkpoints.
pub(crate) fn split_expert_tensor(
    file: &impl TensorSource,
    name: &str,
    n_experts: usize,
) -> Result<Vec<WeightMatrix>, LoadError> {
    let info = find_info(file, name)?;
    // Real raw shape is `[in_dim, out_dim, n_experts]` (ggml's
    // fastest-first `ne[]` order -- see `load_weight_matrix`'s doc
    // comment for the confirmed 2D case this generalizes from). `n_experts`
    // is the slowest-varying (last, i.e. outermost/most-major) dimension,
    // so each expert's `out_dim*in_dim` block is contiguous with experts
    // back-to-back in the mmap.
    if info.shape.len() != 3 || info.shape[2] as usize != n_experts {
        let file_experts = info.shape.last().map(|&d| d as usize).unwrap_or(0);
        return Err(LoadError::ExpertCountMismatch(
            name.to_string(),
            file_experts,
            n_experts,
        ));
    }
    let out_dim = info.shape[1] as usize;
    let in_dim = info.shape[0] as usize;
    let raw = file.tensor_bytes(name)?;

    match info.dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            let all = crate::loader::widen_plain_float(info.dtype, raw, name)?;
            let per_expert = out_dim * in_dim;
            Ok((0..n_experts)
                .map(|e| {
                    WeightMatrix::F32(Tensor::new(
                        all[e * per_expert..(e + 1) * per_expert].to_vec(),
                        vec![out_dim, in_dim],
                    ))
                })
                .collect())
        }
        other => match quant_kind_for(other) {
            Some(kind) => {
                let (mmap, full_range) = file.tensor_mapped_range(name)?;
                #[cfg(feature = "metal")]
                frink_metal::gpu::register_weight_mmap(Arc::clone(&mmap));
                let bytes_per_expert = raw.len() / n_experts;
                Ok((0..n_experts)
                    .map(|e| WeightMatrix::Quantized {
                        data: WeightBytes::Mapped {
                            mmap: Arc::clone(&mmap),
                            range: (full_range.start + e * bytes_per_expert)
                                ..(full_range.start + (e + 1) * bytes_per_expert),
                        },
                        rows: out_dim,
                        cols: in_dim,
                        kind,
                    })
                    .collect())
            }
            None => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
        },
    }
}

/// When every routed expert is mmap-backed with a Metal simdgroup-GEMM
/// kind and back-to-back slices, record the combined gate/up/down planes
/// for Metal packed MoE. Gate/up/down may differ in kind (e.g. Q4_K /
/// Q4_K / Q8_0) but must be uniform across experts per role.
#[cfg(feature = "metal")]
fn try_build_moe_packed_q4_planes(experts: &[ExpertWeights]) -> Option<MoePackedQ4Planes> {
    use frink_core::weight_matrix::{QuantKind, WeightBytes};
    use std::sync::Arc;

    if experts.is_empty() {
        return None;
    }

    fn mapped_sg(m: &WeightMatrix) -> Option<(WeightBytes, usize, &'static str)> {
        match m {
            WeightMatrix::Quantized {
                data: WeightBytes::Mapped { mmap, range },
                rows,
                kind,
                ..
            } => {
                let kind_str = match kind {
                    QuantKind::Q4_0 => "Q4_0",
                    QuantKind::Q5_0 => "Q5_0",
                    QuantKind::Q4K => "Q4_K",
                    QuantKind::Q5K => "Q5_K",
                    QuantKind::Q6K => "Q6_K",
                    QuantKind::Q8_0 => "Q8_0",
                    QuantKind::IQ4XS => "IQ4_XS",
                    _ => return None,
                };
                let _ = frink_metal::gpu::mul_mm_sg_meta(kind_str)?;
                Some((
                    WeightBytes::Mapped {
                        mmap: Arc::clone(mmap),
                        range: range.clone(),
                    },
                    *rows,
                    kind_str,
                ))
            }
            _ => None,
        }
    }

    let (gate0, ffn_rows, gate_kind) = mapped_sg(&experts[0].gate)?;
    let (up0, up_rows, up_kind) = mapped_sg(&experts[0].up)?;
    let (down0, hidden_rows, down_kind) = mapped_sg(&experts[0].down)?;
    if up_rows != ffn_rows {
        return None;
    }
    let WeightBytes::Mapped {
        mmap: gate_mmap,
        range: gate0_range,
    } = &gate0
    else {
        return None;
    };
    let WeightBytes::Mapped {
        mmap: up_mmap,
        range: up0_range,
    } = &up0
    else {
        return None;
    };
    let WeightBytes::Mapped {
        mmap: down_mmap,
        range: down0_range,
    } = &down0
    else {
        return None;
    };

    let gate_stride = gate0_range.len();
    let up_stride = up0_range.len();
    let down_stride = down0_range.len();
    if gate_stride == 0 || up_stride == 0 || down_stride == 0 {
        return None;
    }

    let n = experts.len();
    for (i, ex) in experts.iter().enumerate().skip(1) {
        let (g, fr, gk) = mapped_sg(&ex.gate)?;
        let (u, ur, uk) = mapped_sg(&ex.up)?;
        let (d, hr, dk) = mapped_sg(&ex.down)?;
        if gk != gate_kind || uk != up_kind || dk != down_kind {
            return None;
        }
        let WeightBytes::Mapped { mmap, range } = &g else {
            return None;
        };
        if fr != ffn_rows {
            return None;
        }
        if !Arc::ptr_eq(mmap, gate_mmap)
            || range.len() != gate_stride
            || range.start != gate0_range.start + i * gate_stride
        {
            return None;
        }
        let WeightBytes::Mapped { mmap, range } = &u else {
            return None;
        };
        if ur != ffn_rows
            || !Arc::ptr_eq(mmap, up_mmap)
            || range.len() != up_stride
            || range.start != up0_range.start + i * up_stride
        {
            return None;
        }
        let WeightBytes::Mapped { mmap, range } = &d else {
            return None;
        };
        if hr != hidden_rows
            || !Arc::ptr_eq(mmap, down_mmap)
            || range.len() != down_stride
            || range.start != down0_range.start + i * down_stride
        {
            return None;
        }
    }

    Some(MoePackedQ4Planes::new(
        WeightBytes::Mapped {
            mmap: Arc::clone(gate_mmap),
            range: gate0_range.start..gate0_range.start + n * gate_stride,
        },
        WeightBytes::Mapped {
            mmap: Arc::clone(up_mmap),
            range: up0_range.start..up0_range.start + n * up_stride,
        },
        WeightBytes::Mapped {
            mmap: Arc::clone(down_mmap),
            range: down0_range.start..down0_range.start + n * down_stride,
        },
        gate_stride,
        up_stride,
        down_stride,
        n,
        ffn_rows,
        hidden_rows,
        gate_kind,
        up_kind,
        down_kind,
    ))
}

/// One matrix's place inside a store-backed expert's combined byte
/// buffer (gate bytes, then up, then down, concatenated by
/// `GgufExpertSource::read_expert`).
#[derive(Debug, Clone, Copy)]
pub struct StoredMatrixSpec {
    pub offset: usize,
    pub len: usize,
    pub rows: usize,
    pub cols: usize,
    pub kind: QuantKind,
}

/// Byte-range layout of one store-backed routed expert.
#[derive(Debug, Clone, Copy)]
pub struct StoredExpertLayout {
    pub gate: StoredMatrixSpec,
    pub up: StoredMatrixSpec,
    pub down: StoredMatrixSpec,
}

impl StoredExpertLayout {
    pub fn total_bytes(&self) -> usize {
        self.gate.len + self.up.len + self.down.len
    }

    /// Builds temporary zero-copy `WeightMatrix` views over a leased
    /// buffer. Each view's `WeightBytes::Shared` clone of the lease's
    /// `Arc` keeps the cache entry pinned for the view's lifetime.
    pub fn materialize(&self, lease: &frink_core::expert_store::ExpertLease) -> ExpertWeights {
        let mk = |spec: &StoredMatrixSpec| WeightMatrix::Quantized {
            data: WeightBytes::Shared {
                buf: lease.shared_buf(),
                range: spec.offset..spec.offset + spec.len,
            },
            rows: spec.rows,
            cols: spec.cols,
            kind: spec.kind,
        };
        ExpertWeights {
            gate: mk(&self.gate),
            up: mk(&self.up),
            down: mk(&self.down),
        }
    }
}

/// [`ExpertSource`] over a (possibly sharded) GGUF checkpoint: each
/// expert's gate/up/down byte ranges are read positionally from the
/// owning shard file and concatenated, so a store miss touches exactly
/// that expert's bytes -- no mmap of the expert region, no shared seek
/// cursor.
pub struct GgufExpertSource {
    files: Vec<std::fs::File>,
    /// (layer, expert) -> the three (file index, offset, len) segments
    /// in gate/up/down order.
    segments: std::collections::HashMap<ExpertKey, [(usize, u64, usize); 3]>,
}

impl ExpertSource for GgufExpertSource {
    fn expert_len(&self, key: ExpertKey) -> Option<usize> {
        self.segments
            .get(&key)
            .map(|segs| segs.iter().map(|&(_, _, len)| len).sum())
    }

    fn read_expert(&self, key: ExpertKey) -> std::io::Result<Vec<u8>> {
        let segs = self
            .segments
            .get(&key)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, format!("{key:?}")))?;
        let total: usize = segs.iter().map(|&(_, _, len)| len).sum();
        let mut buf = vec![0u8; total];
        let mut written = 0;
        for &(fi, offset, len) in segs {
            let dst = &mut buf[written..written + len];
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                self.files[fi].read_exact_at(dst, offset)?;
            }
            #[cfg(not(unix))]
            {
                use std::io::{Read, Seek, SeekFrom};
                let mut f = &self.files[fi];
                f.seek(SeekFrom::Start(offset))?;
                f.read_exact(dst)?;
            }
            written += len;
        }
        Ok(buf)
    }
}

/// Collects the per-expert `(file, offset, len)` segments and layout
/// for one packed 3D expert tensor -- the store-backed counterpart of
/// `split_expert_tensor`, sharing its shape/offset math. Only
/// quantized dtypes are supported (an F32/BF16 expert tensor keeps the
/// resident path; the store exists for the quantized multi-hundred-GB
/// case).
/// One packed 3D expert tensor's store-backed description: the owning
/// shard index, each expert's `(offset, len)` within that shard file,
/// and the matrix spec shared by every expert's slice.
struct StoredTensorSpecs {
    shard: usize,
    per_expert: Vec<(u64, usize)>,
    spec: StoredMatrixSpec,
}

fn stored_expert_specs(
    file: &ShardedGguf,
    name: &str,
    n_experts: usize,
) -> Result<Option<StoredTensorSpecs>, LoadError> {
    let info = find_info(file, name)?;
    if info.shape.len() != 3 || info.shape[2] as usize != n_experts {
        let file_experts = info.shape.last().map(|&d| d as usize).unwrap_or(0);
        return Err(LoadError::ExpertCountMismatch(
            name.to_string(),
            file_experts,
            n_experts,
        ));
    }
    let out_dim = info.shape[1] as usize;
    let in_dim = info.shape[0] as usize;
    let Some(kind) = quant_kind_for(info.dtype) else {
        return Ok(None); // F32/BF16 (or unsupported): resident fallback
    };
    let shard = file
        .tensor_shard_index(name)
        .expect("find_info succeeded, shard index must exist");
    // The mmap range of a tensor within a GgufFile IS its byte offset
    // range within that shard file (the mmap covers the whole file).
    let (_, full_range) = file.tensor_mapped_range(name)?;
    let total_len = full_range.end - full_range.start;
    let bytes_per_expert = total_len / n_experts;
    let per_expert: Vec<(u64, usize)> = (0..n_experts)
        .map(|e| {
            (
                (full_range.start + e * bytes_per_expert) as u64,
                bytes_per_expert,
            )
        })
        .collect();
    let spec = StoredMatrixSpec {
        offset: 0, // caller assigns the position within the combined buffer
        len: bytes_per_expert,
        rows: out_dim,
        cols: in_dim,
        kind,
    };
    Ok(Some(StoredTensorSpecs {
        shard,
        per_expert,
        spec,
    }))
}

impl Decoder {
    /// Loads real weights from `path` for the given `config`. `config`
    /// supplies the architecture shape (layer count, head counts, MoE
    /// topology); tensor names are resolved against it using the
    /// llama.cpp naming convention described in the module docs.
    ///
    /// A `config.moe.n_experts <= 1` model is treated as dense: expert
    /// weights are read from the plain `blk.N.ffn_{gate,up,down}.weight`
    /// tensor names rather than the packed 3D `_exps` variant.
    pub fn from_gguf(
        path: impl AsRef<std::path::Path>,
        config: ModelConfig,
    ) -> Result<Self, LoadError> {
        Self::from_gguf_with_expert_cache(path, config, None)
    }

    /// Like `from_gguf`, but with `expert_cache_bytes: Some(budget)`
    /// routed experts are NOT loaded resident: each layer holds only
    /// byte-range layouts, and expert bytes are read on demand through
    /// one bounded, lease-protected `ExpertStore` shared by every
    /// layer (a single global byte budget; see
    /// `frink_core::expert_store`). Dense layers, shared experts,
    /// attention, embeddings, and the output head stay resident/mapped
    /// exactly as before -- only routed experts stream. Layers whose
    /// expert tensors are F32/BF16 fall back to resident loading (the
    /// store exists for the quantized case). Output is bit-identical
    /// to the resident path -- same bytes, same kernels -- pinned by
    /// the roundtrip suite's equivalence test.
    pub fn from_gguf_with_expert_cache(
        path: impl AsRef<std::path::Path>,
        mut config: ModelConfig,
        expert_cache_bytes: Option<u64>,
    ) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let file = ShardedGguf::open(path)?;

        // gpt-oss carries four per-layer tensors the generic GQA layer
        // structs have no home for (its fifth, the attention sinks, is
        // `AttnWeights::sinks` and loads by tensor presence). That is
        // decided by the architecture string, so resolve it once here.
        // See `crate::decoder::GptOssWeights`.
        //
        // This used to be ONE flag with the norm-slot fact below,
        // `arch == "gpt-oss"`, standing for two unrelated facts.
        // Splitting them is what let `seed_oss` -- which shares the norm
        // slot and has none of the extra tensors -- be admitted without
        // also being handed attention sinks.
        // Canonicalised: an architecture that llama.cpp computes
        // another one's graph for reads that one's tables, so an alias
        // needs one entry rather than a row in each of a dozen
        // per-architecture lists (`capability::canonical_architecture`).
        let arch = crate::capability::canonical_architecture(
            file.metadata_str("general.architecture")
                .unwrap_or_default(),
        )
        .to_string();
        let is_gpt_oss = arch == "gpt-oss";
        // A `<projection>.scale` companion is a multiply llama.cpp
        // applies and frink does not; refused by name here, before
        // the unread-tensor gate can be talked past
        // (`crate::weight_scales`).
        crate::weight_scales::refuse_weight_scale_tensors(
            &arch,
            file.tensors().map(|(_, t)| t.name.as_str()),
        )?;
        // Which tensor each of the five norm sites is stored under and
        // which FUNCTION norms it, resolved ONCE. Four shapes reach this
        // loader -- the plain pre-norm layer, the post-norm-only
        // topology (`olmo2`, `exaone4`), the non-parametric LayerNorm
        // (`olmo`) and the weighted LayerNorm (`dbrx`) -- and three
        // architectures keep a norm under a name another architecture
        // uses for a different site (`gpt-oss` / `seed_oss`, `dbrx`,
        // `grok`). `crate::norm_sites` is the one table for all of it;
        // this loader used to restate the decision at every site.
        let norm_sites = crate::norm_sites::NormSites::with_function(&arch, config.norm_function);
        let mut gpt_oss_layers: Vec<crate::decoder::GptOssLayer> = Vec::new();

        // One store for the whole model (keys are (layer, expert)),
        // built up-front with every stored expert's segments; created
        // only when the cache is enabled AND some layer can use it.
        let mut store_segments: std::collections::HashMap<ExpertKey, [(usize, u64, usize); 3]> =
            std::collections::HashMap::new();
        let mut stored_layouts: Vec<Option<Vec<StoredExpertLayout>>> = Vec::new();

        // Loaded like any other weight matrix: a quantized embedding
        // table stays quantized (zero-copy mmap) and token lookup
        // dequantizes one row via `WeightMatrix::dequant_row`, instead
        // of the whole vocabulary tensor being widened to f32 up front.
        let embedding = load_weight_matrix(&file, "token_embd.weight")?;
        // The learned position table (`crate::position_embd`), one row
        // per trained position, for the graphs that add one.
        let position_embd = crate::position_embd::load_position_embd(
            &file,
            &arch,
            config.hidden_dim,
            metadata_u64_any(&file, &[format!("{arch}.context_length")]).map(|v| v as usize),
        )?;

        // PHYSICAL layers: the blocks the file holds tensors for. A
        // looped model (`crate::layer_loops`) has more logical layers
        // than this, and they run these same weights.
        let n_physical = config
            .layer_loops
            .map_or(config.n_layers, |loops| loops.physical_layers());
        let mut layers = Vec::with_capacity(n_physical);
        let mut refined_qk_norm = config.qk_norm_style;
        for l in 0..n_physical {
            // THIS layer's head counts and FFN width. Uniform for every
            // architecture but the per-layer ones (`crate::layer_shapes`),
            // and the loader reads the shape rather than the scalars so
            // that a deci / openelm layer is sized by its own header.
            let shape = config.layer_shape(l);
            // Whether THIS layer's FFN reads the layer input rather
            // than the post-attention residual, and under which norm
            // (`crate::parallel_residual`).
            let parallel = crate::parallel_residual::layer_parallel_norm(&file, &arch, l);
            // THIS layer's norm slots: the architecture's row, with
            // Falcon-40B's `attn_norm_2` crossing the two pre-norm names
            // on the layers that carry it (`crate::norm_sites`).
            let layer_sites = norm_sites.for_layer(&arch, &file, l);
            // BitNet's two inner norms, REQUIRED when the architecture
            // has them and untouched otherwise (`crate::sub_norms`).
            let sub_norms = crate::sub_norms::load_sub_norms(
                &file,
                &arch,
                config.block_sub_norms,
                l,
                config.hidden_dim,
                shape.ffn_dim,
            )?;
            let attn = match shape.attention {
                crate::layer_shapes::AttnShape::Gqa { n_heads, .. } => {
                    // Q/K/V and their biases come out of ONE decision about
                    // which spelling this layer uses -- see `qkv_fused`. They
                    // used to be resolved independently, and a checkpoint that
                    // fused both (ChatGLM, Qwen-1) had its bias dropped.
                    let crate::qkv_fused::QkvProjections {
                        q: q_proj,
                        k: k_proj,
                        v: v_proj,
                        q_bias,
                        k_bias,
                        v_bias,
                    } = crate::qkv_fused::load_fused_or_split_qkv(&file, l, &config)?;
                    let q_norm =
                        load_f32_vec_optional(&file, &format!("blk.{l}.attn_q_norm.weight"))?;
                    let k_norm =
                        load_f32_vec_optional(&file, &format!("blk.{l}.attn_k_norm.weight"))?;
                    // The per-head LAYERNORM (`crate::qk_layer_norm`), whose
                    // weight is `n_heads * head_dim` long and would pass
                    // the length rule below as `WholeVector`.
                    if let Some(reason) = crate::qk_layer_norm::per_head_layer_norm_refusal(
                        &arch,
                        l,
                        q_norm.is_some() || k_norm.is_some(),
                    ) {
                        return Err(LoadError::UnsupportedFeature(
                            config.name.to_string(),
                            reason,
                        ));
                    }
                    // Refine WholeVector vs PerHead from the first observed norm length.
                    // The per-head SCALAR gain is decided by architecture first
                    // (`capability::PER_HEAD_SCALAR_QK_GAIN`): its length is
                    // `n_heads`, which a length test alone could confuse with
                    // `head_dim`.
                    if let Some(ref w) = q_norm {
                        if crate::capability::uses_per_head_scalar_qk_gain(&arch) {
                            if w.len() != n_heads {
                                return Err(LoadError::UnsupportedFeature(
                                    config.name.to_string(),
                                    format!(
                                        "blk.{l}.attn_q_norm.weight length {} is not one gain per \
                                         head (n_heads={n_heads}; talkie.cpp:26 creates it {{1, \
                                         n_head}})",
                                        w.len()
                                    ),
                                ));
                            }
                            refined_qk_norm = crate::capability::QkNormStyle::PerHeadScalar;
                        } else if crate::capability::uses_per_head_distinct_qk_norm(&arch) {
                            // `plamo2.cpp:92-93`: `{head_dim, n_head}`, one
                            // row per head, RMS per head. The same length as
                            // a whole-vector weight, so the architecture
                            // decides (`capability::PER_HEAD_DISTINCT_QK_NORM`).
                            if w.len() != n_heads * config.head_dim {
                                return Err(LoadError::UnsupportedFeature(
                                    config.name.to_string(),
                                    format!(
                                        "blk.{l}.attn_q_norm.weight length {} is not one row per \
                                         head (n_heads={n_heads} x head_dim={}; plamo2.cpp:92 \
                                         creates it {{head_dim, n_head}})",
                                        w.len(),
                                        config.head_dim
                                    ),
                                ));
                            }
                            refined_qk_norm = crate::capability::QkNormStyle::PerHeadDistinct;
                        } else if w.len() == config.head_dim {
                            refined_qk_norm = crate::capability::QkNormStyle::PerHead;
                        } else if w.len() == n_heads * config.head_dim {
                            refined_qk_norm = crate::capability::QkNormStyle::WholeVector;
                        } else {
                            return Err(LoadError::UnsupportedFeature(
                                config.name.to_string(),
                                format!(
                                    "blk.{l}.attn_q_norm.weight length {} matches neither \
                                     head_dim={} nor n_heads*head_dim={}",
                                    w.len(),
                                    config.head_dim,
                                    n_heads * config.head_dim
                                ),
                            ));
                        }
                    }
                    let attn = AttnWeights {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj: load_weight_matrix(&file, &format!("blk.{l}.attn_output.weight"))?,
                        // Which tensor, which function, and whether there is a
                        // norm here at all: all three answered by the table.
                        norm_weight: layer_sites.load_pre_norm(layer_sites.attn, &file, Some(l))?,
                        q_norm,
                        k_norm,
                        // Qwen2/Qwen2-MoE-family real QKV bias (`attn_{q,k,v}.bias`,
                        // real config `qkv_bias`, `o_proj` has none) -- see
                        // `AttnWeights::q_bias`'s doc comment. Resolved above,
                        // alongside the projections they belong to, because a
                        // file that fuses the weight fuses the bias too.
                        q_bias,
                        k_bias,
                        v_bias,
                        post_attn_norm: crate::norm_sites::NormSites::load_post_norm(
                            norm_sites.post_attn,
                            &file,
                            l,
                        )?,
                        post_ffn_norm: crate::norm_sites::NormSites::load_post_norm(
                            norm_sites.post_ffn,
                            &file,
                            l,
                        )?,
                        // The architecture table decides whether there is
                        // a gate and how it is applied; the tensor decides
                        // its width. See `crate::attn_gate`.
                        output_gate: crate::attn_gate::AttnGate::load(
                            &file,
                            &arch,
                            l,
                            n_heads,
                            config.head_dim,
                            config.hidden_dim,
                        )?,
                        // The TENSOR decides. Four llama.cpp graphs pass it
                        // into the one `build_attn_mha`; on the generic
                        // path a file that has it gets the sink term and
                        // a file that does not gets none, whatever the
                        // architecture string. gpt-oss's requirement is
                        // checked where its side table loads.
                        sinks: load_attn_sinks(&file, l, n_heads)?,
                        attn_sub_norm: sub_norms.as_ref().map(|n| n.attn.clone()),
                        o_scale: crate::weight_scales::load_projection_gain(
                            &file,
                            &arch,
                            l,
                            "attn_output",
                        )?,
                        o_bias: crate::proj_bias::load_attn_out_bias(
                            &file,
                            &arch,
                            l,
                            config.hidden_dim,
                        )?,
                        shortconv: None,
                        // falcon-h1.cpp:55-71: the Mamba-2 block beside
                        // attention on every layer (`crate::mamba2::
                        // PARALLEL_WITH_ATTENTION`).
                        ssm: if config.parallel_ssm {
                            Some(crate::ssm_block::SsmBlock::Mamba2(
                                crate::mamba2::Mamba2::load(&file, &arch, l, config.hidden_dim)?,
                            ))
                        } else {
                            None
                        },
                        q_gate_interleaved: crate::attn_gate::q_gate_interleaved(&arch),
                    };
                    crate::layer_shapes::check_gqa_projection_widths(
                        l,
                        shape.attention,
                        config.head_dim,
                        config.v_head_dim(),
                        config.hidden_dim,
                        &attn,
                    )?;
                    attn
                }
                other => crate::layer_shapes::load_non_gqa_attention(
                    other,
                    &file,
                    &arch,
                    l,
                    &norm_sites,
                    &config,
                )?,
            };

            // Leading dense layers (see ModelConfig::layer_is_dense's
            // doc comment) load from the plain dense tensor names
            // regardless of this model's global MoE topology, matching
            // the DeepSeek-2/3-family convention found in
            // ik_llama.cpp's source. A model with n_experts<=1
            // globally (the dense test fixture) is dense on every
            // layer either way.
            // A layer with NO FFN at all (`ffn_dim 0`: deci's, Nemotron-H's
            // block-only layers) takes the dense arm, whose loader answers
            // `absent_ffn` for that width, whatever the model's MoE says.
            let is_dense_layer = config.layer_is_dense(l)
                || config.moe.n_experts <= 1
                || shape.ffn_dim == 0
                || crate::moe_interleave::dense_by_router_absence(&arch, &file, l);
            // The ungated experts (`nemotron-h.cpp:82-86,209-215`: a null
            // gate into `build_moe_ffn`, `LLM_FFN_RELU_SQR`) are spelled
            // the way the dense ungated FFN is (`load_dense_expert`): the
            // gate ALIASED to `up`, so `relu(up)^2` runs through the gated
            // body with no branch. A file that carries a gate anyway is
            // refused, as the dense loader refuses one.
            let routed_gate_name = if config.ffn_is_ungated() && !is_dense_layer {
                if file
                    .find_tensor(&format!("blk.{l}.ffn_gate_exps.weight"))
                    .is_some()
                {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!(
                            "blk.{l}.ffn_gate_exps.weight is present but this architecture's \
                             experts are ungated ({:?}: a null gate into build_moe_ffn, \
                             nemotron-h.cpp:212)",
                            config.ffn_activation
                        ),
                    ));
                }
                format!("blk.{l}.ffn_up_exps.weight")
            } else {
                format!("blk.{l}.ffn_gate_exps.weight")
            };
            // The inner FFN norm has a site in the dense body only
            // (`build_ffn` with a NULL down, `bitnet.cpp:127-141`);
            // `build_moe_ffn` has none, so a routed layer that carried
            // one would have nowhere to apply it.
            if sub_norms.is_some() && !is_dense_layer {
                return Err(LoadError::UnsupportedFeature(
                    arch.clone(),
                    format!(
                        "blk.{l}.ffn_sub_norm on a MoE layer: llama.cpp applies the inner FFN \
                         norm in the dense `build_ffn` body only (bitnet.cpp:127-141), and no \
                         routed-expert graph has that site"
                    ),
                ));
            }
            let n_experts = if is_dense_layer {
                1
            } else {
                config.moe.n_experts
            };
            let experts: ExpertBacking = if is_dense_layer {
                ExpertBacking::Resident(vec![load_dense_expert(&file, l, &config, shape.ffn_dim)?])
            } else {
                // Try store-backed layouts first when the cache is
                // enabled; fall back to resident when any of the three
                // tensors isn't a supported quantized dtype.
                let stored = if expert_cache_bytes.is_some() {
                    let g = stored_expert_specs(&file, &routed_gate_name, n_experts)?;
                    let u = stored_expert_specs(
                        &file,
                        &format!("blk.{l}.ffn_up_exps.weight"),
                        n_experts,
                    )?;
                    let d = stored_expert_specs(
                        &file,
                        &format!("blk.{l}.ffn_down_exps.weight"),
                        n_experts,
                    )?;
                    match (g, u, d) {
                        (Some(gt), Some(ut), Some(dt)) => {
                            let mut layouts = Vec::with_capacity(n_experts);
                            for e in 0..n_experts {
                                let key = ExpertKey {
                                    layer: l as u32,
                                    expert: e as u32,
                                };
                                store_segments.insert(
                                    key,
                                    [
                                        (gt.shard, gt.per_expert[e].0, gt.per_expert[e].1),
                                        (ut.shard, ut.per_expert[e].0, ut.per_expert[e].1),
                                        (dt.shard, dt.per_expert[e].0, dt.per_expert[e].1),
                                    ],
                                );
                                let mut gate = gt.spec;
                                let mut up = ut.spec;
                                let mut down = dt.spec;
                                gate.offset = 0;
                                up.offset = gate.len;
                                down.offset = gate.len + up.len;
                                layouts.push(StoredExpertLayout { gate, up, down });
                            }
                            Some(layouts)
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                match stored {
                    Some(layouts) => {
                        // Placeholder; the shared store is attached in a
                        // second pass below once every layer's segments
                        // are collected.
                        stored_layouts.push(Some(layouts));
                        ExpertBacking::Resident(Vec::new())
                    }
                    None => {
                        let gates = split_expert_tensor(&file, &routed_gate_name, n_experts)?;
                        let ups = split_expert_tensor(
                            &file,
                            &format!("blk.{l}.ffn_up_exps.weight"),
                            n_experts,
                        )?;
                        let downs = split_expert_tensor(
                            &file,
                            &format!("blk.{l}.ffn_down_exps.weight"),
                            n_experts,
                        )?;
                        ExpertBacking::Resident(
                            gates
                                .into_iter()
                                .zip(ups)
                                .zip(downs)
                                .map(|((gate, up), down)| ExpertWeights { gate, up, down })
                                .collect(),
                        )
                    }
                }
            };
            if stored_layouts.len() < layers.len() + 1 {
                stored_layouts.push(None);
            }

            let mut shared_experts: Vec<ExpertWeights> =
                if config.moe.n_shared_experts > 0 && !is_dense_layer {
                    // The shared expert takes the architecture's dense
                    // activation, so an ungated one aliases its gate as
                    // `load_dense_expert` does (`nemotron-h.cpp:222-227`).
                    let shexp_gate = if config.ffn_is_ungated() {
                        if file
                            .find_tensor(&format!("blk.{l}.ffn_gate_shexp.weight"))
                            .is_some()
                        {
                            return Err(LoadError::UnsupportedFeature(
                                arch.clone(),
                                format!(
                                    "blk.{l}.ffn_gate_shexp.weight is present but this \
                                     architecture's shared expert is ungated"
                                ),
                            ));
                        }
                        format!("blk.{l}.ffn_up_shexp.weight")
                    } else {
                        format!("blk.{l}.ffn_gate_shexp.weight")
                    };
                    vec![ExpertWeights {
                        gate: load_weight_matrix(&file, &shexp_gate)?,
                        up: load_weight_matrix(&file, &format!("blk.{l}.ffn_up_shexp.weight"))?,
                        down: load_weight_matrix(&file, &format!("blk.{l}.ffn_down_shexp.weight"))?,
                    }]
                } else {
                    Vec::new()
                };
            // A dense FFN SUMMED with the experts (Grok-2, Arctic) is the
            // shared-expert slot under the dense names, plus the row's
            // scale on the sum (`crate::parallel_dense_ffn`). Decided per
            // layer: Grok-1's layers have no triple and take neither.
            // ...and the same scale under the `_shexp` names
            // (`SHARED_EXPERT_SUM_SCALE`, cohere2moe's `* 0.5`), on a
            // layer that loaded a shared expert above.
            let parallel_sum_scale = if is_dense_layer {
                None
            } else {
                match crate::parallel_dense_ffn::parallel_dense_for_layer(&arch, &file, l)? {
                    Some(row) => {
                        shared_experts.push(ExpertWeights {
                            gate: load_weight_matrix(&file, &format!("blk.{l}.ffn_gate.weight"))?,
                            up: load_weight_matrix(&file, &format!("blk.{l}.ffn_up.weight"))?,
                            down: load_weight_matrix(&file, &format!("blk.{l}.ffn_down.weight"))?,
                        });
                        row.sum_scale
                    }
                    None => crate::parallel_dense_ffn::shared_expert_sum_scale(
                        &arch,
                        !shared_experts.is_empty(),
                    ),
                }
            };
            // Arctic's second per-layer norm, the routed branch's operand
            // (`crate::router_input::RouterInput::NormedLayerInput`):
            // REQUIRED on its routed layers, unread everywhere else.
            let exps_norm = if config.router_input.needs_exps_norm() && !is_dense_layer {
                Some(load_f32_vec(
                    &file,
                    &format!("blk.{l}.ffn_norm_exps.weight"),
                )?)
            } else {
                None
            };

            let router = if !is_dense_layer {
                load_weight_matrix(&file, &format!("blk.{l}.ffn_gate_inp.weight"))?
            } else {
                // dense layer: no real router; a zero [1, hidden] matrix
                // always selects the single expert deterministically.
                WeightMatrix::F32(Tensor::zeros(vec![1, config.hidden_dim]))
            };

            let n_for_counts = match &experts {
                ExpertBacking::Resident(v) if v.is_empty() => n_experts,
                other => other.n_experts(),
            };
            let activation_counts = (0..n_for_counts)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect();
            // Qwen2-MoE-specific real tensor (`blk.N.ffn_gate_inp_shexp.weight`,
            // real on-disk shape `[hidden_dim]`, confirmed against
            // llama.cpp's real `qwen2moe.cpp`) -- see
            // `MoeWeights::shared_expert_gate`'s doc comment. Presence
            // of the tensor itself is the real signal (not an
            // architecture-name list): every other supported
            // architecture's checkpoints simply don't carry this
            // tensor, so this naturally stays `None` there.
            let shared_expert_gate = if is_dense_layer {
                None
            } else {
                load_f32_vec_optional(&file, &format!("blk.{l}.ffn_gate_inp_shexp.weight"))?
            };
            #[cfg(feature = "metal")]
            let packed_q4 = match &experts {
                ExpertBacking::Resident(v) if !v.is_empty() => try_build_moe_packed_q4_planes(v),
                _ => None,
            };
            // DeepSeek-V3's aux-loss-free selection bias. The on-disk
            // name carries no `ffn_` prefix -- llama.cpp's
            // `LLM_TENSOR_FFN_EXP_PROBS_B` maps to `blk.%d.exp_probs_b`
            // (`llama-arch.cpp:416`, `gguf-py/gguf/constants.py:1240`).
            // Optional: only the DeepSeek-V3-lineage MoE recipes carry
            // it, and this same generic loader serves OLMoE / Qwen2-MoE /
            // Mixtral, which do not.
            let exp_probs_bias = if is_dense_layer {
                None
            } else {
                load_f32_vec_optional(&file, &format!("blk.{l}.exp_probs_b.bias"))?
            };
            if let Some(bias) = &exp_probs_bias {
                if bias.len() != config.moe.n_experts {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!(
                            "blk.{l}.exp_probs_b.bias has {} entries but the model has {} experts",
                            bias.len(),
                            config.moe.n_experts
                        ),
                    ));
                }
                // Grouped selection masks the *biased* scores before the
                // global top-k (`build_moe_ffn`, the `n_expert_groups > 1`
                // block). frink's `route_top_k_grouped` takes a fixed
                // count from every group instead, which is a different
                // algorithm, so combining the two here would be a guess.
                // Refuse rather than route wrongly.
                if config.moe.expert_group_count.is_some() {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!(
                            "blk.{l}.exp_probs_b.bias together with expert groups \
                             ({:?}): llama.cpp masks the biased scores per group \
                             before a global top-k, which is not the per-group \
                             top-k frink implements",
                            config.moe.expert_group_count
                        ),
                    ));
                }
            }
            let moe = MoeWeights {
                router,
                experts,
                shared_experts,
                shared_expert_gate,
                exp_probs_bias,
                exps_norm,
                parallel_sum_scale,
                // The dense FFN's biases (`crate::proj_bias`), on a dense
                // layer; a routed layer's experts carry none on the
                // generic path (gpt-oss's are its side table's).
                dense_bias: if is_dense_layer && shape.ffn_dim > 0 {
                    let bias = crate::proj_bias::load_dense_ffn_bias(
                        &file,
                        &arch,
                        l,
                        config.hidden_dim,
                        shape.ffn_dim,
                        config.ffn_is_ungated(),
                    )?;
                    if bias.is_some() && sub_norms.is_some() {
                        return Err(LoadError::UnsupportedFeature(
                            arch.clone(),
                            format!(
                                "layer {l} has both an inner FFN norm and FFN biases; no llama.cpp \
                                 graph has both and the dense body has one arm for each"
                            ),
                        ));
                    }
                    bias
                } else {
                    None
                },
                ffn_sub_norm: sub_norms.map(|n| n.ffn),
                down_scale: {
                    let gain =
                        crate::weight_scales::load_projection_gain(&file, &arch, l, "ffn_down")?;
                    if gain.is_some() && !is_dense_layer {
                        return Err(LoadError::UnsupportedFeature(
                            arch.clone(),
                            format!(
                                "blk.{l}.ffn_down.scale on a MoE layer: the routed experts' \
                                 scales are `ffn_down_exps.scale`, one per expert, which is \
                                 not applied here"
                            ),
                        ));
                    }
                    gain
                },
                // The same table as the attention slot, so the two
                // pre-norms cannot disagree about the function, and the
                // pre-FFN tensor's NAME comes from the same row that
                // decided the post-attention slot must not read it. An
                // FFN-free layer (`deci.cpp:52-54`) has no such tensor.
                // A parallel layer with ONE shared norm has no pre-FFN
                // tensor and no pre-FFN norm: the FFN reads the vector
                // attention read (`crate::parallel_residual`).
                norm_weight: if shape.ffn_dim == 0
                    || parallel == Some(crate::parallel_residual::ParallelNorm::SharedNorm)
                {
                    NormOp::None
                } else {
                    layer_sites.load_pre_norm(layer_sites.ffn, &file, Some(l))?
                },
                parallel,
                activation_counts,
                #[cfg(feature = "metal")]
                packed_q4,
            };

            if is_gpt_oss {
                gpt_oss_layers.push(load_gpt_oss_layer(&file, l, &config, attn.sinks.is_some())?);
            }

            // Talkie's per-layer skip scalar (`crate::skip_stream`);
            // REQUIRED there, untouched everywhere else.
            let out_scale =
                crate::skip_stream::load_out_scale(&file, &arch, config.skip_stream, l)?;
            layers.push(LayerWeights {
                attn,
                moe,
                out_scale,
            });
        }

        // `olmo.cpp:15-36` creates no `output_norm` at all and
        // `:128-130` norms the final hidden state with a null weight, so
        // asking for the tensor would refuse every real OLMo-1 file;
        // the table's function decides whether the read happens.
        let final_norm = norm_sites.load_pre_norm(norm_sites.output, &file, None)?;
        // `hrm-text.cpp:46` creates `hrm_z_l_init` REQUIRED, and only
        // that graph does (`crate::hrm`): the learned LOW stream, one
        // `[n_embd]` row broadcast over the tokens at `:182`.
        let hrm_z_l_init = match config.layer_loops {
            Some(crate::layer_loops::LayerLoops::Hrm { .. }) => {
                Some(load_f32_vec(&file, "hrm.z_l_init")?)
            }
            _ => None,
        };
        // The embedding norm (`norm_sites::EMBEDDING_NORM_ARCHITECTURES`),
        // `NormOp::None` where the site is absent.
        // Two answers, one field: a STORED embedding norm (`bloom`) or
        // a weightless one (`muse-glimmer.cpp:69`), and the tables that
        // decide them are disjoint by construction
        // (`norm_sites::WEIGHTLESS_EMBEDDING_NORM`).
        let embedding_norm = if crate::norm_sites::weightless_embedding_norm(&arch) {
            crate::norm::NormOp::RmsNoParams
        } else {
            norm_sites.load_pre_norm(norm_sites.embedding, &file, None)?
        };
        // Many small Llama/Gemma-family GGUFs tie the lm-head to
        // `token_embd.weight` and omit `output.weight` (llama.cpp
        // `llama_model_loader` falls back the same way). Prefer the
        // explicit head when present.
        let output_head = match load_weight_matrix(&file, "output.weight") {
            Ok(w) => w,
            Err(_) => load_weight_matrix(&file, "token_embd.weight")?,
        };
        // `output.bias` for the graphs that create it (`crate::proj_bias`).
        let output_bias = crate::proj_bias::load_output_bias(&file, &arch, output_head.rows())?;

        // Second pass: attach the one shared store to every
        // store-backed layer. Opening the shard files fresh (plain
        // `File` handles for positional reads, not mmaps) keeps the
        // stored experts' bytes out of the process's mapped footprint
        // entirely.
        if !store_segments.is_empty() {
            let budget = expert_cache_bytes
                .expect("store_segments only populated when a cache budget is set")
                as usize;
            let files: Result<Vec<std::fs::File>, std::io::Error> =
                file.shard_paths().iter().map(std::fs::File::open).collect();
            let files = files.map_err(GgufError::from)?;
            let store = std::sync::Arc::new(ExpertStore::new(
                GgufExpertSource {
                    files,
                    segments: store_segments,
                },
                budget,
            ));
            for (l, layer) in layers.iter_mut().enumerate() {
                if let Some(layouts) = stored_layouts.get_mut(l).and_then(Option::take) {
                    layer.moe.experts = ExpertBacking::Stored {
                        store: std::sync::Arc::clone(&store),
                        layouts,
                        layer: l as u32,
                    };
                }
            }
        }

        config.qk_norm_style = refined_qk_norm;

        let family = crate::capability::resolve_profile(
            file.metadata_str("general.architecture").unwrap_or("llama"),
        )
        .map(|p| p.family)
        .unwrap_or(crate::capability::DecoderFamily::StandardGqa);
        let memory_kind = crate::capability::resolve_profile(
            file.metadata_str("general.architecture").unwrap_or("llama"),
        )
        .map(|p| p.memory)
        .unwrap_or(crate::capability::MemoryKind::KvGqa);
        let execution_plan = crate::execution_plan::ExecutionPlan::from_config(
            &config,
            family,
            memory_kind,
            crate::execution_plan::ExecutionPlan::probe_metal_caps(),
        );

        let alibi_slopes = crate::decoder::config_alibi_slopes(&config);
        let decoder = Decoder {
            config,
            embedding,
            position_embd,
            embedding_norm,
            // `hrm-text.cpp:46` creates it REQUIRED, and only that
            // graph does (`crate::hrm`); the loader reads it for the
            // architecture whose schedule needs it and for no other.
            hrm_z_l_init,
            alibi_slopes,
            layers,
            final_norm,
            output_head,
            output_bias,
            gpu_vram_budget_bytes: None,
            gpt_oss: if is_gpt_oss {
                Some(crate::decoder::GptOssWeights {
                    layers: gpt_oss_layers,
                })
            } else {
                None
            },
            qk_norm_after_rope: QK_NORM_AFTER_ROPE_ARCHITECTURES.contains(&arch.as_str()),
            #[cfg(feature = "metal")]
            metal_attn_kv: std::sync::Mutex::new(None),
            execution_plan,
            kv_window: crate::decoder::KvWindowPolicy::from_env(),
            plan_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            lora_adapters: Vec::new(),
        };
        // Resolve every kernel the model will need while we still have a
        // load-time error path to report it on, then seal: from here a
        // lookup that misses is an unpredicted slow path and says so.
        decoder.probe_kernels();
        frink_core::kernel_registry::seal_or_error()
            .map_err(|e| LoadError::StrictKernels(e.to_string()))?;
        // `ModelConfig` is parsed from a *different* handle on the same
        // file (the CLI opens its own `GgufFile`, then hands the config
        // here), so the model-level tensors it consumed were recorded on
        // that handle, not this one. Replay them before the gate, or
        // every Llama-3.x checkpoint reads as carrying an unread
        // `rope_freqs.weight` it in fact uses on every RoPE call.
        for name in crate::config::MODEL_LEVEL_TENSORS_READ_BY_CONFIG {
            file.note_consumed(name);
        }
        // The NextN/MTP blocks llama.cpp creates `TENSOR_SKIP` and never
        // runs (`crate::mtp_blocks`): deliberately unread, and said so,
        // rather than left for the gate below to report as a term the
        // graph is missing. The range is the config's, so the layer
        // loop above and this mark cannot disagree about where the
        // trunk ends.
        let skipped = crate::mtp_blocks::note_mtp_blocks_skipped(
            &file,
            &crate::mtp_blocks::TrunkLayers {
                block_count: n_physical + decoder.config.n_mtp_blocks,
                n_layers: n_physical,
                n_mtp_blocks: decoder.config.n_mtp_blocks,
            },
        );
        if skipped > 0 {
            eprintln!(
                "frink: skipping {} NextN/MTP block(s) after layer {} ({skipped} tensors), as \
                 llama.cpp does",
                decoder.config.n_mtp_blocks,
                n_physical - 1
            );
        }
        // Slots llama.cpp creates and never reads (`crate::unread_tensors`):
        // ignored as upstream ignores them, and said so.
        let ignored = crate::unread_tensors::note_unread_layer_tensors(&file, &arch, n_physical);
        if !ignored.is_empty() {
            eprintln!(
                "frink: ignoring {} tensor(s) llama.cpp creates and never reads for `{}` \
                 (first: {}), as llama.cpp does",
                ignored.len(),
                arch,
                ignored[0]
            );
        }
        assert_every_tensor_consumed(&file)?;
        Ok(decoder)
    }
}

/// Tensor-name prefixes a text-generation load legitimately never
/// reads. Everything here is consumed by a *different* code path, not by
/// nothing: multimodal projector planes belong to `mmproj`, and the
/// per-shard split bookkeeping is metadata, not weights.
const IGNORED_TENSOR_PREFIXES: &[&str] = &["mm.", "v.", "mmproj.", "resampler.", "audio."];

/// Fails the load when the checkpoint carries tensors this build never
/// looked at.
///
/// A tensor nobody reads is not a harmless extra: it is a term of the
/// real graph that ours is missing. gpt-oss ships `blk.N.attn_sinks`
/// and frink has no attention-sink code anywhere, so the file loads,
/// runs at full speed, and emits a different distribution than the model
/// it claims to be; the newer MoE recipes ship `ffn_exp_probs_b` the
/// same way. Both are silent today, and both are exactly what the
/// architecture registry cannot catch, because the architecture *string*
/// is one frink does support -- it is the checkpoint that carries more
/// than the registry entry promises.
///
/// This is deliberately the last check in the load: by here every loader
/// arm has had its chance to ask for what it needs, so what is left over
/// is what nothing in this build knows about.
///
/// `FRINK_ALLOW_UNKNOWN_TENSORS=1` downgrades it to a warning, for the
/// case where a human has decided the missing term does not matter (a
/// bias tensor of zeros, an auxiliary head that never runs). The default
/// is refusal: a wrong answer is worse than no answer.
pub fn assert_every_tensor_consumed(file: &ShardedGguf) -> Result<(), LoadError> {
    let mut left: Vec<String> = file
        .unconsumed_tensors()
        .into_iter()
        .filter(|n| !IGNORED_TENSOR_PREFIXES.iter().any(|p| n.starts_with(p)))
        .collect();
    if left.is_empty() {
        return Ok(());
    }
    left.sort();
    let shown = left.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
    let listing = if left.len() > 8 {
        format!("{shown}, … (+{} more)", left.len() - 8)
    } else {
        shown
    };
    if matches!(
        std::env::var("FRINK_ALLOW_UNKNOWN_TENSORS").ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    ) {
        eprintln!(
            "frink: WARNING -- {} tensor(s) in this checkpoint are never read \
             ({listing}); output may be wrong (FRINK_ALLOW_UNKNOWN_TENSORS=1)",
            left.len()
        );
        return Ok(());
    }
    Err(LoadError::UnconsumedTensors(left.len(), listing))
}

#[cfg(test)]
mod tests {

    /// A quantized 1-D tensor loads through the shared helper.
    ///
    /// This used to be six copies of `load_f32_vec`, and they had
    /// drifted badly: this one decoded twenty dtypes while the five
    /// architecture loaders decoded three (F32/F16/BF16). A quantizer
    /// that emits a Q8_0 norm or bias -- ordinary for aggressive
    /// quants -- loaded on the generic path and was rejected with
    /// `UnsupportedDtype` on GLM-5.2, Kimi, DeepSeek-MLA, Gemma-4 and
    /// the hybrid stack.
    ///
    /// This file's own comment predicted exactly that, about the same
    /// split one level down: "a dtype frink can decode should never be
    /// rejected here just because the *other* dispatch table below
    /// knows it -- that split is how a supported format turns into a
    /// load failure on the one checkpoint that uses it."
    #[test]
    fn a_quantized_one_dimensional_tensor_widens_through_the_shared_helper() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.25).collect();
        let quantized = frink_quant::quantize_q8_0(&values);

        struct OneTensor {
            info: TensorInfo,
            bytes: Vec<u8>,
        }
        impl TensorSource for OneTensor {
            fn metadata(&self, _key: &str) -> Option<&frink_gguf::GgufValue> {
                None
            }
            fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
                (name == self.info.name).then_some(&self.info)
            }
            fn tensor_bytes(&self, _name: &str) -> Result<&[u8], GgufError> {
                Ok(&self.bytes)
            }
            fn tensor_mapped_range(
                &self,
                name: &str,
            ) -> Result<
                (
                    std::sync::Arc<frink_gguf::MmapHandle>,
                    std::ops::Range<usize>,
                ),
                GgufError,
            > {
                // Never reached: `load_f32_vec` widens from bytes.
                Err(GgufError::TensorNotFound(name.to_string()))
            }
        }

        let source = OneTensor {
            info: TensorInfo {
                name: "blk.0.attn_norm.weight".to_string(),
                shape: vec![64],
                dtype: GgmlType::Q8_0,
                offset: 0,
            },
            bytes: quantized,
        };

        let widened = load_f32_vec(&source, "blk.0.attn_norm.weight")
            .expect("a Q8_0 norm must load, not report an unsupported dtype");
        assert_eq!(widened.len(), values.len());
        for (got, want) in widened.iter().zip(values.iter()) {
            assert!(
                (got - want).abs() < 0.05,
                "q8_0 round trip: got {got}, want {want}"
            );
        }
    }
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        buf.write_all(s.as_bytes()).unwrap();
    }

    fn write_kv_str(buf: &mut Vec<u8>, key: &str, val: &str) {
        write_string(buf, key);
        buf.write_u32::<LittleEndian>(8).unwrap(); // type = string
        write_string(buf, val);
    }

    /// A minimal, tensor-free GGUF byte buffer declaring only
    /// `general.architecture` (no `{arch}.block_count` or any other
    /// hparam key) -- the shape a stripped-down or malformed file might
    /// take, and the exact case `ModelConfig::from_gguf` must reject
    /// loudly rather than silently default around.
    fn build_arch_only_gguf(arch: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count
        write_kv_str(&mut buf, "general.architecture", arch);
        buf
    }

    #[test]
    fn model_config_from_gguf_fails_loudly_when_required_hparams_are_missing() {
        let tmp =
            std::env::temp_dir().join(format!("frink_test_arch_only_{}.gguf", std::process::id()));
        // Use a registered architecture so the failure is MissingHparam,
        // not UnsupportedArchitecture.
        std::fs::write(&tmp, build_arch_only_gguf("llama")).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
        std::fs::remove_file(&tmp).ok();

        match ModelConfig::from_gguf(&file) {
            Err(LoadError::MissingHparam(key)) => {
                assert_eq!(key, "llama.block_count");
            }
            other => panic!(
                "expected LoadError::MissingHparam for a file with no hparam keys, got {other:?}"
            ),
        }
    }

    #[test]
    fn model_config_from_gguf_fails_closed_on_unknown_architecture() {
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_unknown_arch_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&tmp, build_arch_only_gguf("bogus-arch-with-no-hparams")).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
        std::fs::remove_file(&tmp).ok();

        match ModelConfig::from_gguf(&file) {
            Err(LoadError::UnsupportedArchitecture(arch)) => {
                assert_eq!(arch, "bogus-arch-with-no-hparams");
            }
            other => panic!(
                "expected LoadError::UnsupportedArchitecture for an unregistered arch, got {other:?}"
            ),
        }
    }

    fn write_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
        write_string(buf, key);
        buf.write_u32::<LittleEndian>(6).unwrap(); // type = float32
        buf.write_f32::<LittleEndian>(val).unwrap();
    }

    /// `arch` plus one f32 hparam, so a metadata-only feature gate can be
    /// exercised without building a whole checkpoint.
    fn build_arch_plus_f32_gguf(arch: &str, key: &str, val: f32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(2).unwrap(); // kv_count
        write_kv_str(&mut buf, "general.architecture", arch);
        write_kv_f32(&mut buf, key, val);
        buf
    }

    fn config_error_for(arch: &str, key: &str, val: f32, tag: &str) -> LoadError {
        let tmp = std::env::temp_dir().join(format!("frink_test_scale_{tag}.gguf"));
        std::fs::write(&tmp, build_arch_plus_f32_gguf(arch, key, val)).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
        std::fs::remove_file(&tmp).ok();
        ModelConfig::from_gguf(&file).expect_err("must not succeed")
    }

    /// Granite / MiniCPM / Command-R multipliers are hparams, not
    /// tensors, so `assert_every_tensor_consumed` cannot see them: a
    /// checkpoint declaring one loads, runs at full speed, and computes
    /// a differently-scaled graph than it was trained as. An
    /// architecture whose reference graph does not apply one must refuse
    /// it by name.
    ///
    /// Driven on `llama` rather than on `granite`, and that swap is the
    /// point: `granite` APPLIES all four now
    /// (`crate::scalar_multipliers`), so leaving the case here would
    /// have turned this test into a test of nothing the day the feature
    /// landed. llama.cpp's llama graph reads none of the four keys, so a
    /// `llama` checkpoint declaring one is exactly the silent divergence
    /// the gate exists for.
    #[test]
    fn a_declared_multiplier_this_decoder_does_not_apply_is_refused_by_name() {
        for (key, val) in [
            ("llama.logit_scale", 6.0f32),
            ("llama.residual_scale", 0.22),
            ("llama.embedding_scale", 12.0),
            ("llama.attention.scale", 0.015_625),
        ] {
            let tag = key.replace('.', "_");
            match config_error_for("llama", key, val, &tag) {
                LoadError::UnsupportedFeature(arch, msg) => {
                    assert_eq!(arch, "llama");
                    assert!(msg.contains(key), "error must name the key: {msg}");
                }
                other => panic!("expected UnsupportedFeature for {key}, got {other:?}"),
            }
        }
    }

    /// The complement, and the half that would otherwise have gone
    /// missing: `granite` must NOT be refused for the keys its graph
    /// applies.
    ///
    /// The refusal list and the implementation are two views of ONE
    /// table (`scalar_multipliers::multiplier_support`), so this test
    /// and the one above cannot both pass while they disagree -- which
    /// is the whole value of deriving the list rather than restating it.
    #[test]
    fn granite_is_not_refused_for_the_multipliers_it_applies() {
        for (key, val) in [
            ("granite.logit_scale", 8.0f32),
            ("granite.residual_scale", 0.22),
            ("granite.embedding_scale", 12.0),
            ("granite.attention.scale", 0.015_625),
        ] {
            let tag = format!("granite_ok_{}", key.replace('.', "_"));
            // The file carries no `block_count`, so the load still fails
            // -- but on the *missing hparam*, having passed this gate.
            match config_error_for("granite", key, val, &tag) {
                LoadError::MissingHparam(k) => assert_eq!(k, "granite.block_count"),
                other => panic!("{key}={val} must pass the scaling gate, got {other:?}"),
            }
        }
    }

    /// The gate must not fire on a multiplier that is a no-op. A file
    /// writing `residual_scale = 1.0` describes the graph frink already
    /// computes, and refusing it would be a false alarm. llama.cpp's
    /// `f_attention_scale` uses `0.0` rather than `1.0` as its "unset"
    /// sentinel, so the two are checked against their own no-op values.
    #[test]
    fn a_multiplier_that_is_a_no_op_is_not_refused() {
        for (key, val) in [
            ("llama.logit_scale", 1.0f32),
            ("llama.residual_scale", 1.0),
            ("llama.embedding_scale", 1.0),
            ("llama.attention.scale", 0.0),
        ] {
            let tag = format!("noop_{}", key.replace('.', "_"));
            // The file carries no `block_count`, so the load still fails
            // -- but on the *missing hparam*, having passed this gate.
            match config_error_for("llama", key, val, &tag) {
                LoadError::MissingHparam(k) => assert_eq!(k, "llama.block_count"),
                other => panic!("no-op {key}={val} must pass the scaling gate, got {other:?}"),
            }
        }
    }

    /// One GGUF metadata value, in the three types these header-only
    /// fixtures need.
    enum Kv<'a> {
        Str(&'a str),
        U32(u32),
        F32(f32),
        /// A uint32 ARRAY. Only one gate needs it -- the sliding-window
        /// pattern, which llama.cpp reads with `ml.get_key_or_arr` --
        /// and without it that gate could only be tested through a
        /// value of some other type, which is not the case it exists
        /// for.
        Arr32(&'a [u32]),
    }

    /// A tensor-free GGUF carrying exactly `kvs` -- enough for
    /// `ModelConfig::from_gguf` to run without a single weight on disk.
    fn build_metadata_gguf(kvs: &[(&str, Kv)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(kvs.len() as u64).unwrap();
        for (k, v) in kvs {
            match v {
                Kv::Str(s) => write_kv_str(&mut buf, k, s),
                Kv::U32(n) => {
                    write_string(&mut buf, k);
                    buf.write_u32::<LittleEndian>(4).unwrap(); // type = uint32
                    buf.write_u32::<LittleEndian>(*n).unwrap();
                }
                Kv::F32(f) => write_kv_f32(&mut buf, k, *f),
                Kv::Arr32(values) => {
                    write_string(&mut buf, k);
                    buf.write_u32::<LittleEndian>(9).unwrap(); // type = array
                    buf.write_u32::<LittleEndian>(4).unwrap(); // element type = uint32
                    buf.write_u64::<LittleEndian>(values.len() as u64).unwrap();
                    for v in *values {
                        buf.write_u32::<LittleEndian>(*v).unwrap();
                    }
                }
            }
        }
        buf
    }

    fn open_metadata_gguf(tag: &str, kvs: &[(&str, Kv)]) -> frink_gguf::GgufFile {
        let tmp = std::env::temp_dir().join(format!("frink_test_meta_{tag}.gguf"));
        std::fs::write(&tmp, build_metadata_gguf(kvs)).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("header-only file must parse");
        std::fs::remove_file(&tmp).ok();
        file
    }

    /// A minimal `llama` hparam set (64-wide single head, base 10000)
    /// plus whatever RoPE-scaling keys a test wants to add.
    fn llama_config_with(tag: &str, extra: &[(&str, Kv)]) -> ModelConfig {
        let mut kvs: Vec<(&str, Kv)> = vec![
            ("general.architecture", Kv::Str("llama")),
            ("llama.block_count", Kv::U32(1)),
            ("llama.embedding_length", Kv::U32(64)),
            ("llama.attention.head_count", Kv::U32(1)),
            ("llama.attention.head_count_kv", Kv::U32(1)),
            ("llama.attention.key_length", Kv::U32(64)),
            ("llama.rope.freq_base", Kv::F32(10_000.0)),
        ];
        for (k, v) in extra {
            kvs.push((
                k,
                match v {
                    Kv::Str(s) => Kv::Str(s),
                    Kv::U32(n) => Kv::U32(*n),
                    Kv::F32(f) => Kv::F32(*f),
                    Kv::Arr32(a) => Kv::Arr32(a),
                },
            ));
        }
        ModelConfig::from_gguf(&open_metadata_gguf(tag, &kvs)).expect("fixture must load")
    }

    /// Builds a config for an arbitrary architecture tag, returning the
    /// error rather than unwrapping it.
    /// llama.cpp chooses the FFN gate activation PER ARCHITECTURE;
    /// frink chose it per family. Those are different partitions, and
    /// `grok` is where they disagree: `src/models/grok.cpp:165` passes
    /// `LLM_FFN_GELU` to `build_moe_ffn`, while `grok` is
    /// `DecoderFamily::StandardGqa` and so was handed SwiGLU -- a
    /// different FFN on every layer.
    ///
    /// It was pinned here while `grok` still refused, because the
    /// failure mode is that auditing it later makes it silently wrong,
    /// and an audit is exactly when nobody thinks to re-check the
    /// activation. `grok` is audited now (tests/grok_graphs.rs), and the
    /// fixture's GELU experts are what that suite compares.
    #[test]
    fn the_ffn_activation_follows_the_architecture_not_the_family() {
        use crate::capability::uses_geglu;
        use crate::config::FfnActivation;

        assert!(uses_geglu("grok"), "grok's MoE FFN gate is GELU upstream");
        // Same family, SiLU upstream (`src/models/dbrx.cpp:122`), so the
        // family rule alone cannot be what selects grok.
        assert!(!uses_geglu("dbrx"));
        assert!(!uses_geglu("llama"));

        // The Gemma lineage keeps its GELU through the FAMILY rule, so
        // the new per-architecture arm must not have displaced it.
        // gemma2/gemma3 only: `gemma` v1 is unaudited and refuses, so
        // it cannot be loaded to check its activation.
        for gemma in ["gemma2", "gemma3"] {
            assert!(
                !uses_geglu(gemma),
                "{gemma} is GELU via GemmaFamily; listing it here too \
                 would hide a later regression in the family rule"
            );
            assert_eq!(
                config_for_arch(gemma).expect("gemma loads").ffn_activation,
                FfnActivation::Gelu,
                "{gemma}"
            );
        }

        // And a plain SwiGLU architecture stays SwiGLU.
        assert_eq!(
            config_for_arch("llama")
                .expect("llama loads")
                .ffn_activation,
            FfnActivation::Swiglu
        );

        // The ungated ReLU-squared row, and the four that share its FFN
        // and refuse for something else (`capability::uses_relu_sqr`).
        assert_eq!(
            config_for_arch("arcee")
                .expect("arcee loads")
                .ffn_activation,
            FfnActivation::ReluSqr
        );
        for shared in ["plm", "nemotron", "jais2", "nemotron_h"] {
            assert!(crate::capability::uses_relu_sqr(shared), "{shared}");
        }
        assert!(!crate::capability::uses_relu_sqr("llama"));
    }

    /// A per-layer array whose entries differ, on an architecture whose
    /// llama.cpp graph reads layer 0, is refused naming the table; the
    /// same arrays with equal entries are the uniform model, for any
    /// architecture, because a converter may spell a scalar as a list.
    ///
    /// Reachability, not only `LayerShapes::resolve`'s own unit test:
    /// this goes through `from_gguf` on a header-only file, which is
    /// where `openelm` used to die on `MissingHparam` for a key its
    /// file carried.
    #[test]
    fn a_varying_per_layer_array_is_refused_on_a_layer_zero_architecture_and_equal_ones_are_uniform(
    ) {
        let kvs = [
            ("general.architecture", Kv::Str("llama")),
            ("llama.block_count", Kv::U32(2)),
            ("llama.embedding_length", Kv::U32(64)),
            ("llama.attention.head_count", Kv::Arr32(&[2, 2])),
            ("llama.attention.head_count_kv", Kv::Arr32(&[2, 1])),
            ("llama.attention.key_length", Kv::U32(32)),
            ("llama.rope.freq_base", Kv::F32(10_000.0)),
        ];
        let err = ModelConfig::from_gguf(&open_metadata_gguf("layer_shapes_vary", &kvs))
            .expect_err("llama takes layer 0 upstream");
        let msg = err.to_string();
        assert!(msg.contains("PER_LAYER_SHAPE_ARCHS"), "{msg}");
        assert!(msg.contains("LLAMA_LOAD_LOCALS"), "{msg}");

        let kvs = [
            ("general.architecture", Kv::Str("llama")),
            ("llama.block_count", Kv::U32(2)),
            ("llama.embedding_length", Kv::U32(64)),
            ("llama.attention.head_count", Kv::Arr32(&[2, 2])),
            ("llama.attention.head_count_kv", Kv::Arr32(&[1, 1])),
            ("llama.attention.key_length", Kv::U32(32)),
            ("llama.rope.freq_base", Kv::F32(10_000.0)),
        ];
        let cfg = ModelConfig::from_gguf(&open_metadata_gguf("layer_shapes_equal", &kvs))
            .expect("equal arrays are the uniform model");
        assert!(cfg.layer_shapes.is_uniform());
        assert_eq!((cfg.n_heads, cfg.n_kv_heads), (2, 1));

        // An array of the wrong length is refused as llama.cpp refuses
        // it (`key has wrong array length`).
        let kvs = [
            ("general.architecture", Kv::Str("llama")),
            ("llama.block_count", Kv::U32(2)),
            ("llama.embedding_length", Kv::U32(64)),
            ("llama.attention.head_count", Kv::Arr32(&[2, 2, 2])),
            ("llama.attention.key_length", Kv::U32(32)),
            ("llama.rope.freq_base", Kv::F32(10_000.0)),
        ];
        let err = ModelConfig::from_gguf(&open_metadata_gguf("layer_shapes_len", &kvs))
            .expect_err("three entries for two layers");
        assert!(err.to_string().contains("wrong array length"), "{err}");
    }

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
        for arch in ["deepseek", "olmoe", "qwen2moe"] {
            assert!(
                NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&arch),
                "{arch} passes norm_w=false in llama.cpp and must not be renormalised"
            );
        }
        // `deepseek2` is a DIFFERENT architecture whose converter DOES
        // write the key, so it must not be on this list -- it gets its
        // answer from the file.
        assert!(!NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&"deepseek2"));
        assert!(!NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&"qwen3moe"));
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
            (
                "NO_TOPK_RENORMALIZE_ARCHITECTURES",
                NO_TOPK_RENORMALIZE_ARCHITECTURES,
            ),
            (
                "PRE_FFN_NORM_IS_POST_ATTENTION_NORM",
                crate::norm_sites::PRE_FFN_NORM_IS_POST_ATTENTION_NORM,
            ),
            (
                "PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM",
                crate::norm_sites::PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM,
            ),
            (
                "POST_NORMS_UNDER_GROK_NAMES",
                crate::norm_sites::POST_NORMS_UNDER_GROK_NAMES,
            ),
            (
                "ATTN_NORM_2_FEEDS_ATTENTION",
                crate::norm_sites::ATTN_NORM_2_FEEDS_ATTENTION,
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
            (
                "PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM",
                crate::norm_sites::PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM,
            ),
            (
                "POST_NORMS_UNDER_GROK_NAMES",
                crate::norm_sites::POST_NORMS_UNDER_GROK_NAMES,
            ),
            (
                "ATTN_NORM_2_FEEDS_ATTENTION",
                crate::norm_sites::ATTN_NORM_2_FEEDS_ATTENTION,
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

    fn config_for_arch(arch: &'static str) -> Result<ModelConfig, LoadError> {
        // The per-arch hyperparameter keys are looked up by the arch's
        // own prefix, so they have to be built for the arch under test.
        let keys: Vec<String> = [
            "block_count",
            "embedding_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
        ]
        .iter()
        .map(|k| format!("{arch}.{k}"))
        .collect();
        let theta = format!("{arch}.rope.freq_base");
        let kvs: Vec<(&str, Kv)> = vec![
            ("general.architecture", Kv::Str(arch)),
            (keys[0].as_str(), Kv::U32(1)),
            (keys[1].as_str(), Kv::U32(64)),
            (keys[2].as_str(), Kv::U32(1)),
            (keys[3].as_str(), Kv::U32(1)),
            (keys[4].as_str(), Kv::U32(64)),
            (theta.as_str(), Kv::F32(10_000.0)),
        ];
        ModelConfig::from_gguf(&open_metadata_gguf(arch, &kvs))
    }

    /// The generic path is OPT-IN, and this is what proves it.
    ///
    /// An architecture nobody has checked used to FALL ONTO generic GQA
    /// and run. Five did exactly that and computed the wrong thing for
    /// the life of the project. The refusal exists; nothing tested it,
    /// so a reordering or an unevidenced addition to
    /// `AUDITED_GENERIC_GQA` would have gone unnoticed.
    #[test]
    fn an_unaudited_generic_architecture_refuses_rather_than_guessing() {
        // `grovemoe` is on the generic path and is not in the audited
        // list. It is the sixth name to hold this slot: `starcoder` was
        // first, until an audit found it REQUIRES a fused
        // `attn_qkv.bias` and a learned `position_embd` the generic
        // decoder has no slot for, so it refuses for a stronger reason;
        // then `xverse`, until it was admitted with a libllama-golden
        // fixture (`tests/fixture_away_graphs.rs`); then `nanbeige`,
        // until the layer loop became `crate::layer_loops`; then
        // `talkie`, until `crate::skip_stream`; then `arctic`, until
        // `crate::parallel_dense_ffn`. `grovemoe` runs a SECOND expert
        // bank (`src/models/grovemoe.cpp:57-59,137-164`) whose upstream
        // graph diverges from the reference, and its blocker is
        // invisible in metadata, so nothing but this gate stops it.
        assert!(
            !crate::capability::is_audited_generic("grovemoe"),
            "this test needs an arch that is generic AND unaudited"
        );
        match config_for_arch("grovemoe") {
            Err(LoadError::UnauditedArchitecture(name, ..)) => assert_eq!(name, "grovemoe"),
            other => panic!("expected an unaudited refusal, got {other:?}"),
        }
    }

    /// An architecture with evidence still loads, or the inversion would
    /// have turned every model off.
    #[test]
    fn an_audited_architecture_still_loads() {
        assert!(crate::capability::is_audited_generic("llama"));
        assert!(config_for_arch("llama").is_ok());
    }

    /// A NAMED problem must outrank "unaudited".
    ///
    /// `grovemoe` is unaudited AND names its second expert bank; a
    /// `llama4` file declaring a window of zero names the branch
    /// libllama aborts on (`crate::chunked_swa`), and that is what its
    /// refusal should say. Reporting "unaudited" instead would be true
    /// and far less useful, and it is the ordering the loader's own
    /// comment claims. Nothing checked that claim. (`gpt2`, `bloom` and
    /// then a plain `llama4` were the example until each was served.)
    #[test]
    fn a_named_refusal_outranks_the_unaudited_one() {
        let kvs: Vec<(&str, Kv)> = vec![
            ("general.architecture", Kv::Str("llama4")),
            ("llama4.block_count", Kv::U32(1)),
            ("llama4.embedding_length", Kv::U32(64)),
            ("llama4.attention.head_count", Kv::U32(1)),
            ("llama4.attention.head_count_kv", Kv::U32(1)),
            ("llama4.attention.key_length", Kv::U32(64)),
            ("llama4.rope.freq_base", Kv::F32(10_000.0)),
            ("llama4.expert_count", Kv::U32(16)),
            ("llama4.interleave_moe_layer_step", Kv::U32(1)),
            ("llama4.attention.sliding_window", Kv::U32(0)),
        ];
        let err = ModelConfig::from_gguf(&open_metadata_gguf("llama4", &kvs))
            .expect_err("a zero window must refuse");
        assert!(
            !matches!(err, LoadError::UnauditedArchitecture(..)),
            "llama4 should report its own reason, not that nobody audited it: {err:?}"
        );
        assert!(err.to_string().contains("llama-graph.cpp:159"), "{err}");
    }

    /// A checkpoint that declares YaRN gets the per-band divisors the
    /// reference's `"yarn"` arm implies, folded into `rope_freqs` so the
    /// existing RoPE kernels apply them. Expected values are hand-derived
    /// from `_find_correction_dim` for this fixture (rotary width 64,
    /// base 10000, original context 131072): `low = 22`, `high = 35`.
    ///
    /// Before this, frink read neither `rope.scaling.type` nor
    /// `rope.scaling.factor`, so this file roped exactly like an
    /// unscaled one -- correct near position 0, progressively wrong
    /// further in.
    #[test]
    fn a_gguf_declaring_yarn_gets_its_rope_frequencies_rewritten() {
        let cfg = llama_config_with(
            "yarn",
            &[
                ("llama.rope.scaling.type", Kv::Str("yarn")),
                ("llama.rope.scaling.factor", Kv::F32(8.0)),
                (
                    "llama.rope.scaling.original_context_length",
                    Kv::U32(131_072),
                ),
            ],
        );
        let factors = cfg
            .rope_freqs
            .expect("a YaRN checkpoint must carry rewritten per-band frequencies")
            .full;
        assert_eq!(factors.len(), 32, "one divisor per rotation band");
        assert!(
            (factors[0] - 1.0).abs() < 1e-6,
            "the fastest band is left extrapolated, got {}",
            factors[0]
        );
        let ramp = (31.0 - 22.0) / (35.0 - 22.0);
        let want = 1.0 / (ramp / 8.0 + (1.0 - ramp));
        assert!(
            (factors[31] - want).abs() < 1e-4,
            "slowest band: got {}, reference {want}",
            factors[31]
        );
    }

    /// The rewrite must not fire on a file that did not ask for it. A
    /// scaling type frink does not implement (`linear`, `longrope`) is
    /// left exactly as it was rather than being roped as YaRN, which
    /// would be a new kind of wrong rather than the current known one.
    /// `rope.scaling.type = "linear"` must actually scale.
    ///
    /// Rotating position `p/s` is the same as rotating `p` with every
    /// band's frequency divided by `s`, and `rope_freqs` is exactly a
    /// per-band frequency divisor, so a uniform vector of `s` expresses
    /// linear scaling with no new code on the RoPE paths.
    ///
    /// Before this, the scaling type was compared against "yarn" and
    /// anything else returned None, so such a file loaded and roped at
    /// unscaled positions: a different model, no error.
    #[test]
    fn linear_scaling_is_applied_as_a_uniform_frequency_divisor() {
        let cfg = llama_config_with(
            "linear",
            &[
                ("llama.rope.scaling.type", Kv::Str("linear")),
                ("llama.rope.scaling.factor", Kv::F32(4.0)),
            ],
        );
        let freqs = &cfg
            .rope_freqs
            .as_ref()
            .expect("linear scaling must produce frequency factors")
            .full;
        assert_eq!(freqs.len(), cfg.head_dim / 2, "one factor per rotated pair");
        assert!(
            freqs.iter().all(|f| (*f - 4.0).abs() < 1e-6),
            "linear scaling is uniform across bands, unlike YaRN: got {freqs:?}"
        );
    }

    /// A factor that corrects nothing is not a correction.
    #[test]
    fn a_linear_factor_of_one_is_treated_as_absent() {
        assert!(llama_config_with(
            "linear_one",
            &[
                ("llama.rope.scaling.type", Kv::Str("linear")),
                ("llama.rope.scaling.factor", Kv::F32(1.0)),
            ],
        )
        .rope_freqs
        .is_none());
    }

    #[test]
    fn a_gguf_without_yarn_scaling_keeps_its_rope_frequencies_untouched() {
        assert!(llama_config_with("noscale", &[]).rope_freqs.is_none());
        // Linear scaling is NOT "no scaling". It used to land here,
        // asserted as `is_none()`, on the reasoning that leaving
        // positions alone beat roping them wrong in a new way. Both are
        // wrong output: llama.cpp divides the positions by the factor.
        // See `linear_scaling_is_applied_as_a_uniform_frequency_divisor`.
        // YaRN with a no-op factor is not a correction either.
        assert!(llama_config_with(
            "yarn_factor_one",
            &[
                ("llama.rope.scaling.type", Kv::Str("yarn")),
                ("llama.rope.scaling.factor", Kv::F32(1.0)),
                (
                    "llama.rope.scaling.original_context_length",
                    Kv::U32(131_072),
                ),
            ],
        )
        .rope_freqs
        .is_none());
    }

    /// The correction range is measured against the context the
    /// checkpoint was *trained* at, so a file that declares YaRN without
    /// `rope.scaling.original_context_length` leaves the rotation alone
    /// rather than inventing a trained length (`context_length` on such
    /// a file is the *extended* one, which would put the ramp in the
    /// wrong place at every band).
    #[test]
    fn yarn_without_an_original_context_length_is_not_guessed_at() {
        let cfg = llama_config_with(
            "yarn_noctx",
            &[
                ("llama.rope.scaling.type", Kv::Str("yarn")),
                ("llama.rope.scaling.factor", Kv::F32(8.0)),
            ],
        );
        assert!(cfg.rope_freqs.is_none());
    }

    /// `general.sampling.*` is the checkpoint's own recommendation, and
    /// only the keys the file carries become one: a file naming just
    /// `top_k` must leave temperature and top_p to the server's
    /// defaults.
    #[test]
    fn gguf_sampling_metadata_is_read_as_the_checkpoints_recommendation() {
        use crate::sampling::RecommendedSampling;
        let full = RecommendedSampling::from_gguf(&open_metadata_gguf(
            "sampling_full",
            &[
                ("general.architecture", Kv::Str("llama")),
                ("general.sampling.temp", Kv::F32(1.0)),
                ("general.sampling.top_k", Kv::U32(20)),
                ("general.sampling.top_p", Kv::F32(0.95)),
            ],
        ));
        assert_eq!(
            full,
            RecommendedSampling {
                temperature: Some(1.0),
                top_p: Some(0.95),
                top_k: Some(20),
            }
        );

        let partial = RecommendedSampling::from_gguf(&open_metadata_gguf(
            "sampling_partial",
            &[
                ("general.architecture", Kv::Str("llama")),
                ("general.sampling.top_k", Kv::U32(40)),
            ],
        ));
        assert_eq!(partial.top_k, Some(40));
        assert_eq!(partial.temperature, None);
        assert_eq!(partial.top_p, None);
    }

    /// A converter that wrote `temp = 1` stores a GGUF integer, not a
    /// float. Dropping it would serve a checkpoint that asked for
    /// temperature 1.0 at the framework's greedy default -- the
    /// repetition-loop failure the recommendation exists to prevent.
    #[test]
    fn an_integer_valued_sampling_temp_is_still_a_recommendation() {
        let recommended = crate::sampling::RecommendedSampling::from_gguf(&open_metadata_gguf(
            "sampling_int_temp",
            &[
                ("general.architecture", Kv::Str("llama")),
                ("general.sampling.temp", Kv::U32(1)),
            ],
        ));
        assert_eq!(recommended.temperature, Some(1.0));
    }

    /// The overwhelming majority of checkpoints recommend nothing, and
    /// those must keep frink's existing defaults exactly.
    #[test]
    fn a_gguf_without_sampling_metadata_recommends_nothing() {
        let recommended = crate::sampling::RecommendedSampling::from_gguf(&open_metadata_gguf(
            "sampling_absent",
            &[("general.architecture", Kv::Str("llama"))],
        ));
        assert!(recommended.is_empty());
    }

    #[test]
    fn model_config_from_gguf_rejects_dedicated_architectures() {
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_dedicated_arch_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&tmp, build_arch_only_gguf("deepseek4")).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
        std::fs::remove_file(&tmp).ok();

        match ModelConfig::from_gguf(&file) {
            Err(LoadError::DedicatedArchitectureRequired(arch, _)) => {
                assert_eq!(arch, "deepseek4");
            }
            other => panic!(
                "expected LoadError::DedicatedArchitectureRequired for deepseek4, got {other:?}"
            ),
        }
    }

    /// The same Q5_K block bytes cross-validated against an independent
    /// Python reference in `frink-quant`'s own tests, reused here for
    /// the same full-path proof as the Q6_K test below.
    #[rustfmt::skip]
    const Q5_K_TEST_BLOCK: [u8; 176] = [
        0x66, 0x2a, 0x66, 0x2a, 0x01, 0x01, 0x01, 0x01, 0x4f, 0x4b, 0x10, 0x12, 0x41, 0xe2, 0xc1,
        0xb1, 0x72, 0x2f, 0x20, 0x07, 0x31, 0x0c, 0x38, 0xb3, 0x9c, 0xb8, 0xad, 0x2f, 0x9a, 0xea,
        0x17, 0xd0, 0xee, 0x93, 0x9e, 0x3e, 0x74, 0xbb, 0x28, 0x18, 0x39, 0x25, 0xb6, 0x09, 0x18,
        0x29, 0x1c, 0x1d, 0x29, 0x41, 0x40, 0x0a, 0x74, 0x7d, 0xfd, 0x21, 0xdd, 0x6d, 0x45, 0x73,
        0x0e, 0x1e, 0xc0, 0x4a, 0xfc, 0xf3, 0x8e, 0x24, 0x6b, 0x34, 0x7d, 0xbe, 0x94, 0xde, 0x59,
        0x7a, 0x35, 0x30, 0x36, 0x0a, 0xf9, 0x4a, 0x9b, 0xa2, 0x26, 0x21, 0xa2, 0xfa, 0xdf, 0x4b,
        0x29, 0x64, 0x6f, 0xbb, 0xca, 0x0f, 0x3c, 0xda, 0x20, 0xf4, 0x93, 0x86, 0xab, 0x6e, 0xb9,
        0xe5, 0xd5, 0xa0, 0x82, 0xd6, 0x41, 0xff, 0x12, 0xbc, 0x34, 0xbb, 0xab, 0xb8, 0x20, 0x2f,
        0xbb, 0x5f, 0x0c, 0x10, 0xcf, 0x49, 0xc5, 0x86, 0x5c, 0xdf, 0xff, 0x78, 0x44, 0x26, 0x3b,
        0xc2, 0x23, 0x3d, 0x2b, 0xe9, 0x00, 0x12, 0xf8, 0xea, 0xe2, 0x9e, 0x5e, 0x50, 0x20, 0x9f,
        0x9d, 0x8d, 0x7d, 0x7f, 0xcc, 0x1d, 0x0e, 0x13, 0xf8, 0xc2, 0xf1, 0x3d, 0x08, 0x2f, 0x23,
        0x13, 0xac, 0x0d, 0xa7, 0xe7, 0x20, 0xa3, 0x90, 0xb7, 0xc8, 0x28,
    ];

    fn build_single_q5_k_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-q5k-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols,
                                                   // rows] -- reversed from the semantic [rows, cols] this tensor
                                                   // represents (1 row, 256 cols / 1 Q5_K block).
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q5_K block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(13).unwrap(); // dtype tag: Q5_K
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&Q5_K_TEST_BLOCK);
        buf
    }

    /// How far a fused dot may sit from an exact dequantized dot.
    ///
    /// Two regimes, and one fixed number cannot describe both. With
    /// `FRINK_CPU_INT_DOT` off the activation stays f32 and only
    /// rounding separates the two. With it on, the activation is
    /// quantized to int8 at `d = amax / 127`, which is the flag both
    /// binaries turn on by default and the reason the Q5_K and Q6_K
    /// cases failed against a flat `1e-2`.
    ///
    /// The bound grows with the L2 norm of the row, NOT the L1. Each
    /// element carries an independent rounding of up to `d/2`, so the
    /// dot's error is a sum of independent terms whose standard
    /// deviation is `d/sqrt(12) * ||w||_2`. Bounding by the worst case
    /// `d/2 * ||w||_1` instead assumes every rounding aligns with its
    /// weight's sign, which on this fixture gives 0.347 against a dot
    /// of 2.77: 12% of the value, loose enough that injecting a 5%
    /// error still passed. Measured here, the real error is 1.8 sigma,
    /// so four sigma keeps better than 2x headroom while still failing
    /// that 5% injection.
    fn fused_dot_tolerance(weights: &[f32], x: &[f32], exact_bound: f32) -> f32 {
        if !frink_core::weight_matrix::cpu_int_dot_for(
            frink_core::weight_matrix::IntDotShape::Matvec,
        ) {
            return exact_bound;
        }
        let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let l2 = weights.iter().map(|w| w * w).sum::<f32>().sqrt();
        4.0 * (amax / 127.0) / 12f32.sqrt() * l2 + exact_bound
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q5_k_tensor_end_to_end() {
        let tmp =
            std::env::temp_dir().join(format!("frink_test_q5k_tensor_{}.gguf", std::process::id()));
        std::fs::write(&tmp, build_single_q5_k_tensor_gguf()).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real Q5_K GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q5_K tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q5K);
                assert!(
                    data.is_mapped(),
                    "Q5_K tensors should take the zero-copy mmap path, same as Q8_0/Q4_0"
                );
            }
            _ => panic!("expected a Quantized matrix for a Q5_K tensor"),
        }

        let expected = frink_quant::dequant_q5_k(&Q5_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < fused_dot_tolerance(&expected, &x, 1e-2),
            "end-to-end loaded+applied Q5_K matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    /// The same Q6_K block bytes cross-validated against an independent
    /// Python reference in `frink-quant`'s own tests; reused here to
    /// prove the *full*
    /// path -- real on-disk GGUF bytes, parsed by `frink-gguf`, read
    /// through `GgufFile::tensor_mapped_range`, dispatched by
    /// `WeightMatrix::apply` to `frink_quant::dot_q6_k_f32` -- produces
    /// the same result as directly dequantizing those bytes, not just
    /// that the isolated kernel is correct in unit-test isolation.
    #[rustfmt::skip]
    const Q6_K_TEST_BLOCK: [u8; 210] = [
        0xe0, 0xa5, 0x40, 0x5c, 0x8d, 0x3a, 0x0a, 0x26, 0xfb, 0x4b, 0x6e, 0x9a, 0xdf, 0x3e, 0xa3,
        0xc4, 0xf8, 0x2b, 0x1d, 0x95, 0x76, 0x7d, 0x3b, 0xcd, 0xfd, 0xef, 0xc2, 0x0b, 0x07, 0x63,
        0x29, 0xfb, 0x81, 0x57, 0xbe, 0xbe, 0x06, 0xf7, 0x3a, 0x92, 0xc4, 0x43, 0xff, 0xad, 0xac,
        0x7e, 0x0f, 0x00, 0x2a, 0x4f, 0xf0, 0xf8, 0xa9, 0xfa, 0x3c, 0x90, 0x6d, 0x73, 0x2d, 0x5a,
        0xe6, 0xc6, 0x46, 0xf2, 0x0d, 0x55, 0x4c, 0x25, 0x38, 0x71, 0x2b, 0x35, 0x38, 0x82, 0x16,
        0x37, 0x5f, 0x32, 0x61, 0x02, 0xdd, 0x2f, 0x6f, 0x7b, 0x1f, 0xb4, 0x1a, 0x1b, 0x3e, 0x4f,
        0x11, 0xa3, 0x17, 0x40, 0x5a, 0x5f, 0x76, 0xcd, 0x19, 0x27, 0x9b, 0xc7, 0xc8, 0xf7, 0xf7,
        0xee, 0xf4, 0x86, 0xd9, 0xfd, 0xa7, 0xfe, 0x9e, 0xac, 0x70, 0x53, 0x5b, 0x76, 0xfb, 0x39,
        0xf8, 0x4b, 0x98, 0xfe, 0xd0, 0x06, 0x21, 0x4c, 0x4d, 0xbe, 0x10, 0x2b, 0x06, 0x65, 0xc9,
        0x5e, 0xf9, 0x95, 0x72, 0xae, 0x99, 0xd9, 0x7e, 0x15, 0xbd, 0x5e, 0x6d, 0xe8, 0x25, 0x8a,
        0xd5, 0x99, 0xc6, 0x6b, 0x69, 0xc7, 0x84, 0xc6, 0xa4, 0xf7, 0xb9, 0x6d, 0x68, 0x45, 0x0e,
        0x65, 0x69, 0xeb, 0xe6, 0xeb, 0xe9, 0x28, 0xa6, 0xb9, 0x96, 0xf2, 0xe8, 0xa7, 0x9b, 0x6e,
        0x79, 0x8a, 0x68, 0x65, 0x59, 0x98, 0x8b, 0x44, 0x41, 0x98, 0x9a, 0x56, 0x01, 0x01, 0x01,
        0x02, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x02, 0x02, 0x01, 0x01, 0x01, 0x02, 0x1f, 0x25,
    ];

    fn build_single_q6_k_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-q6k-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q6_K block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(14).unwrap(); // dtype tag: Q6_K
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&Q6_K_TEST_BLOCK);
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q6_k_tensor_end_to_end() {
        let tmp =
            std::env::temp_dir().join(format!("frink_test_q6k_tensor_{}.gguf", std::process::id()));
        std::fs::write(&tmp, build_single_q6_k_tensor_gguf()).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real Q6_K GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q6_K tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q6K);
                assert!(
                    data.is_mapped(),
                    "Q6_K tensors should take the zero-copy mmap path, same as Q8_0/Q4_0"
                );
            }
            _ => panic!("expected a Quantized matrix for a Q6_K tensor"),
        }

        let expected = frink_quant::dequant_q6_k(&Q6_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < fused_dot_tolerance(&expected, &x, 1e-2),
            "end-to-end loaded+applied Q6_K matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    fn build_single_bf16_tensor_gguf(rows: u64, cols: u64, values: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-bf16-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(cols).unwrap();
        buf.write_u64::<LittleEndian>(rows).unwrap();
        buf.write_u32::<LittleEndian>(30).unwrap(); // dtype tag: BF16
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for &v in values {
            // Real bf16 truncation (round-toward-zero, matching a real
            // writer closely enough for round-trip test purposes): top
            // 16 bits of the f32 bit pattern.
            let bf16_bits = (v.to_bits() >> 16) as u16;
            buf.extend_from_slice(&bf16_bits.to_le_bytes());
        }
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_bf16_tensor_end_to_end() {
        // Values with zero low-mantissa bits, so f32->bf16 truncation
        // is lossless and this is an exact-equality check.
        let values: Vec<f32> = vec![1.0, -2.5, 0.0, 4.0, -8.0, 16.0];
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_bf16_tensor_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&tmp, build_single_bf16_tensor_gguf(2, 3, &values)).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real BF16 GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("BF16 tensor must load");
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        match &matrix {
            WeightMatrix::F32(tensor) => {
                assert_eq!(tensor.data, values, "BF16 must widen to f32 exactly");
            }
            _ => panic!("expected an F32 matrix for a BF16 tensor (no fused dot kernel for it)"),
        }
    }

    fn build_single_f16_tensor_gguf(rows: u64, cols: u64, values: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-f16-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
        buf.write_u64::<LittleEndian>(cols).unwrap();
        buf.write_u64::<LittleEndian>(rows).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap(); // dtype tag: F16
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for &v in values {
            buf.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        buf
    }

    /// `GgmlType::F16` was parsed and sized but had no dequant arm in any
    /// of the seven loaders, so every `*-f16.gguf` was a hard
    /// `UnsupportedDtype`. Values are exactly representable in f16, so
    /// this is an exact-equality check.
    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_f16_tensor_end_to_end() {
        let values: Vec<f32> = vec![1.0, -2.5, 0.0, 4.0, -8.0, 16.0];
        let tmp =
            std::env::temp_dir().join(format!("frink_test_f16_tensor_{}.gguf", std::process::id()));
        std::fs::write(&tmp, build_single_f16_tensor_gguf(2, 3, &values)).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real F16 GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("F16 tensor must load");
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        match &matrix {
            WeightMatrix::F32(tensor) => {
                assert_eq!(tensor.data, values, "F16 must widen to f32 exactly");
            }
            _ => panic!("expected an F32 matrix for an F16 tensor (no fused dot kernel for it)"),
        }

        // The same tensor read as a plain vector (norm weights, biases and
        // the router all take this path, not `load_weight_matrix`).
        let tmp =
            std::env::temp_dir().join(format!("frink_test_f16_vec_{}.gguf", std::process::id()));
        std::fs::write(&tmp, build_single_f16_tensor_gguf(2, 3, &values)).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real F16 GGUF file must parse");
        std::fs::remove_file(&tmp).ok();
        assert_eq!(load_f32_vec(&file, "test.weight").unwrap(), values);
    }

    fn build_single_q5_1_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-q5-1-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(32).unwrap(); // cols (1 Q5_1 block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(7).unwrap(); // dtype tag: Q5_1
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        // d=0.25 (f16 0x3400), m=1.5 (f16 0x3E00) -- both exact in f16,
        // hand-verified bit patterns to avoid pulling in the `half`
        // crate just for two test constants. qh varied, qs a real
        // (non-degenerate) pattern.
        buf.extend_from_slice(&0x3400u16.to_le_bytes());
        buf.extend_from_slice(&0x3E00u16.to_le_bytes());
        buf.extend_from_slice(&[0x9au8, 0x3c, 0xf0, 0x0f]);
        buf.extend_from_slice(&(0..16u8).map(|i| i | ((15 - i) << 4)).collect::<Vec<u8>>());
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q5_1_tensor_end_to_end() {
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_q5_1_tensor_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&tmp, build_single_q5_1_tensor_gguf()).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real Q5_1 GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q5_1 tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 32);
        let raw = file.tensor_bytes("test.weight").unwrap();
        let expected = frink_quant::dequant_q5_1(raw).unwrap();
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q5_1);
                assert!(data.is_mapped());
            }
            _ => panic!("expected a Quantized matrix for a Q5_1 tensor"),
        }

        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.017).cos()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < 1e-2,
            "end-to-end loaded+applied Q5_1 matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    // Same bytes as frink-quant's own Q3_K_TEST_BLOCK (Python-cross-
    // validated there); duplicated here to build a real on-disk GGUF
    // file, matching this file's existing per-format test convention
    // (see Q6_K_TEST_BLOCK above).
    const Q3_K_TEST_BLOCK: [u8; 110] = [
        0x56, 0xf2, 0xb4, 0x2b, 0xd5, 0x6f, 0x51, 0x71, 0x3c, 0x0a, 0xb9, 0x1d, 0xd0, 0xb9, 0x3b,
        0xb3, 0x0f, 0xff, 0x8c, 0xb2, 0x83, 0x3a, 0x3d, 0x24, 0xb1, 0x12, 0x56, 0xe3, 0x23, 0x54,
        0xf2, 0xfa, 0x7f, 0xdf, 0x31, 0xe1, 0x18, 0x26, 0x6e, 0xcd, 0x5b, 0x38, 0xee, 0xbd, 0x9f,
        0x8c, 0x57, 0x47, 0x0b, 0x11, 0xcb, 0xfb, 0xb4, 0x83, 0xa0, 0x4e, 0x0b, 0xd4, 0xa7, 0x85,
        0xe0, 0x60, 0xf3, 0xb3, 0xe3, 0x95, 0x43, 0xc6, 0x05, 0x05, 0x77, 0x53, 0xed, 0x23, 0xcc,
        0x6a, 0x0e, 0x89, 0xa1, 0x79, 0x85, 0xf6, 0x6e, 0x5a, 0x23, 0x63, 0xbe, 0x53, 0xfa, 0xa2,
        0x2b, 0xe9, 0xcd, 0xce, 0xf8, 0x3d, 0x6f, 0xd0, 0x42, 0x6e, 0x3b, 0x7f, 0x23, 0x26, 0xd3,
        0xb9, 0x18, 0xbf, 0xa4, 0x34,
    ];

    fn build_single_q3_k_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-q3k-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q3_K block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(11).unwrap(); // dtype tag: Q3_K
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&Q3_K_TEST_BLOCK);
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q3_k_tensor_end_to_end() {
        let tmp =
            std::env::temp_dir().join(format!("frink_test_q3k_tensor_{}.gguf", std::process::id()));
        std::fs::write(&tmp, build_single_q3_k_tensor_gguf()).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real Q3_K GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q3_K tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q3K);
                assert!(data.is_mapped());
            }
            _ => panic!("expected a Quantized matrix for a Q3_K tensor"),
        }

        let expected = frink_quant::dequant_q3_k(&Q3_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < fused_dot_tolerance(&expected, &x, 1e-1),
            "end-to-end loaded+applied Q3_K matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    // Same bytes as frink-quant's own IQ4_XS_TEST_BLOCK (Python-cross-
    // validated there); duplicated here to build a real on-disk GGUF
    // file, matching this file's existing per-format test convention.
    const IQ4_XS_TEST_BLOCK: [u8; 136] = [
        0x5c, 0x33, 0xb4, 0x39, 0xd1, 0x64, 0x97, 0x82, 0xcb, 0xbd, 0x88, 0x95, 0xf3, 0x60, 0x2a,
        0xb5, 0xe7, 0x24, 0xd3, 0xee, 0xfe, 0x71, 0x13, 0xbe, 0x70, 0x84, 0x48, 0x79, 0x7b, 0x3e,
        0xf0, 0x55, 0xdc, 0xb2, 0xb2, 0xde, 0x32, 0xa1, 0x5b, 0x02, 0x01, 0xdc, 0x2a, 0xbb, 0xf7,
        0x0b, 0x8a, 0x88, 0xdd, 0x0b, 0x02, 0x7e, 0x5e, 0x76, 0x87, 0x30, 0x1e, 0x1c, 0xcf, 0x48,
        0xd7, 0x61, 0xf3, 0x51, 0x52, 0x17, 0x98, 0x0a, 0x87, 0xcf, 0x02, 0x91, 0xc8, 0xee, 0xc0,
        0x91, 0x69, 0x2a, 0x4f, 0x64, 0x68, 0xa7, 0xb2, 0xe6, 0x98, 0x21, 0x81, 0x75, 0x53, 0x2a,
        0x8d, 0x12, 0xae, 0xe0, 0xea, 0x0c, 0x75, 0xff, 0x22, 0x5e, 0x25, 0x19, 0xda, 0x2e, 0x51,
        0x4e, 0x81, 0xdc, 0x0e, 0x78, 0x86, 0xd7, 0x58, 0xb5, 0xb7, 0xf6, 0x45, 0xa9, 0x0a, 0x83,
        0xfd, 0x2a, 0x12, 0x7d, 0xf0, 0x12, 0x97, 0xe2, 0xfe, 0xf4, 0xd0, 0xa2, 0x11, 0x14, 0x78,
        0xdb,
    ];

    fn build_single_iq4_xs_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-iq4xs-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 IQ4_XS block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(23).unwrap(); // dtype tag: IQ4_XS
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&IQ4_XS_TEST_BLOCK);
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_iq4_xs_tensor_end_to_end() {
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_iq4xs_tensor_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&tmp, build_single_iq4_xs_tensor_gguf()).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real IQ4_XS GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("IQ4_XS tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::IQ4XS);
                assert!(data.is_mapped());
            }
            _ => panic!("expected a Quantized matrix for an IQ4_XS tensor"),
        }

        let expected = frink_quant::dequant_iq4_xs(&IQ4_XS_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < 1e-1,
            "end-to-end loaded+applied IQ4_XS matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    // Same bytes as frink-quant's own IQ low-bit test blocks
    // (Python-cross-validated there against the real compiled ggml
    // implementation), duplicated as literals for the same reason as
    // IQ4_XS_TEST_BLOCK above.
    const IQ1_S_TEST_BLOCK: [u8; 50] = [
        0x0a, 0x2f, 0xfa, 0x06, 0x1e, 0x37, 0x6f, 0xe3, 0x62, 0xd0, 0xb6, 0xa4, 0x25, 0xae, 0x76,
        0x14, 0x72, 0x5b, 0xfa, 0x05, 0xd1, 0xf1, 0x2a, 0x4c, 0xad, 0x29, 0xae, 0xf4, 0xcf, 0x0c,
        0x96, 0x51, 0x58, 0x03, 0x6d, 0xd3, 0x10, 0x92, 0x70, 0xff, 0x61, 0x58, 0xc8, 0x30, 0x25,
        0x64, 0x49, 0x85, 0xc0, 0x24,
    ];
    const IQ2_XXS_TEST_BLOCK: [u8; 66] = [
        0x29, 0x30, 0xd9, 0x33, 0x95, 0x4c, 0x08, 0x1e, 0xad, 0x79, 0x49, 0xf2, 0x8d, 0x5f, 0x93,
        0xea, 0x78, 0x18, 0x98, 0xb9, 0x94, 0x14, 0xad, 0xce, 0xca, 0x1d, 0xab, 0x81, 0x53, 0x4a,
        0x68, 0xd0, 0x59, 0x96, 0x36, 0x5d, 0xbe, 0x20, 0xc4, 0xff, 0xe4, 0x2c, 0xcd, 0x2f, 0x4f,
        0x4f, 0x67, 0x53, 0xc6, 0xd5, 0xa2, 0xfb, 0xc7, 0xf3, 0xe2, 0x6b, 0xf1, 0x99, 0x23, 0x1e,
        0x2d, 0x5e, 0x8c, 0x78, 0xc2, 0x31,
    ];
    const IQ3_XXS_TEST_BLOCK: [u8; 98] = [
        0x71, 0x31, 0x16, 0x0a, 0x79, 0x04, 0x5d, 0x87, 0xae, 0x2a, 0x4a, 0x43, 0xfd, 0x02, 0xba,
        0x6c, 0x10, 0x42, 0x80, 0xe5, 0x1d, 0x08, 0x22, 0xcb, 0x21, 0x54, 0xf9, 0xaa, 0x8e, 0xc2,
        0xf2, 0x34, 0x66, 0x1e, 0x2a, 0xef, 0x19, 0xae, 0x48, 0x47, 0x29, 0xa0, 0x72, 0xd1, 0x31,
        0xc0, 0x65, 0x49, 0xde, 0x79, 0x32, 0xe6, 0x4d, 0xb6, 0x55, 0x3f, 0x4d, 0xf1, 0x18, 0xbb,
        0x18, 0x59, 0x4c, 0x31, 0xa3, 0xb2, 0x34, 0xdd, 0xf6, 0x4a, 0x91, 0x51, 0x3f, 0x3e, 0x40,
        0x69, 0xad, 0xbf, 0x1a, 0xd0, 0x05, 0xfb, 0xbe, 0x8b, 0x0b, 0xdd, 0xdf, 0x7d, 0x94, 0x74,
        0x92, 0x3e, 0xff, 0x04, 0x2a, 0xc4, 0xea, 0xc9,
    ];

    #[rustfmt::skip]
    const MXFP4_GGUF_TEST_BLOCKS: [u8; 68] = [0x79, 0xb4, 0x8d, 0xe2, 0x62, 0x5d, 0xbb, 0x9d, 0x54, 0xe6, 0xdb, 0x94, 0x59, 0x7d, 0x28, 0xf9, 0x79, 0x7a, 0xfc, 0xc1, 0xfa, 0x1e, 0x53, 0x5b, 0x0e, 0xc2, 0x5a, 0x2f, 0x0c, 0x82, 0x4d, 0xcb, 0x11, 0x28, 0x7b, 0x7c, 0xb6, 0x45, 0xe0, 0xb0, 0x52, 0x40, 0x51, 0xec, 0x30, 0x1a, 0xd2, 0x17, 0xf3, 0xbb, 0xfc, 0x7c, 0x8f, 0xf0, 0x67, 0x83, 0x88, 0x9d, 0x79, 0xdb, 0xf4, 0x45, 0x29, 0x78, 0xe6, 0xf4, 0x99, 0xea];

    /// A live ggml type this build has no kernel for must be REFUSED BY
    /// NAME at execution, having been sized correctly at parse.
    ///
    /// Before `TQ2_0` was recognized, tag 35 was `Other(35)`, which had
    /// no block layout: the tensor's size was unknown, so `tensor_bytes`
    /// could not even hand back the row, and the error named a number.
    /// Now the file parses, the tensor measures 66 bytes per 256
    /// elements, and the stop happens where it belongs -- at the point
    /// something wants to multiply by it -- naming `TQ2_0`.
    #[test]
    fn a_recognized_but_unimplemented_ggml_type_refuses_by_name_after_sizing_correctly() {
        // 256 elements of TQ2_0 = one 66-byte block.
        let block = pseudo_iq_block(66, 0x0720_5eed);
        let tmp =
            std::env::temp_dir().join(format!("frink_test_tq2_0_{}.gguf", std::process::id()));
        std::fs::write(
            &tmp,
            build_single_iq_lowbit_tensor_gguf("tq2test", 35, 256, &block),
        )
        .unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("a TQ2_0 file must still parse");
        std::fs::remove_file(&tmp).ok();

        // Sized, not zero: the size estimate is right even though the
        // kernel is missing.
        let info = file.find_tensor("test.weight").expect("tensor present");
        assert_eq!(info.dtype, GgmlType::TQ2_0);
        assert_eq!(info.byte_len(), Some(66));
        assert_eq!(
            file.tensor_bytes("test.weight").map(<[u8]>::len).ok(),
            Some(66)
        );

        match load_weight_matrix(&file, "test.weight") {
            Err(LoadError::UnsupportedDtype(name, GgmlType::TQ2_0)) => {
                assert_eq!(name, "test.weight");
            }
            Err(other) => panic!("TQ2_0 must be refused by name, got {other:?}"),
            Ok(_) => panic!("TQ2_0 must be refused, not loaded as some other kind"),
        }
    }

    /// An MXFP4 norm/bias must widen, not be refused.
    ///
    /// `load_weight_matrix` accepts MXFP4 as a 2-D weight and
    /// `load_moe_expert_matrices` accepts it as an expert tensor, and
    /// `WeightMatrix::dequant` calls `dequant_mxfp4_gguf` on both. One
    /// missing arm in `widen_plain_float` made the *1-D* tensors of the
    /// exact same dtype a hard `UnsupportedDtype` -- the split that
    /// turns a supported format into a load failure on the one
    /// checkpoint that uses it.
    #[test]
    fn an_mxfp4_one_dimensional_tensor_widens_instead_of_being_refused() {
        let expected = frink_quant::dequant_mxfp4_gguf(&MXFP4_GGUF_TEST_BLOCKS)
            .expect("the fixture blocks must dequantize");
        let cols = expected.len();
        let tmp =
            std::env::temp_dir().join(format!("frink_test_mxfp4_norm_{}.gguf", std::process::id()));
        std::fs::write(
            &tmp,
            build_single_iq_lowbit_tensor_gguf(
                "mxfp4norm",
                39,
                cols as u64,
                &MXFP4_GGUF_TEST_BLOCKS,
            ),
        )
        .unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("file must parse");
        std::fs::remove_file(&tmp).ok();

        let got = load_f32_vec(&file, "test.weight")
            .expect("an MXFP4 norm must load, not report an unsupported dtype");
        assert_eq!(got, expected);

        // Same arm, reached directly: `widen_plain_float` is the shared
        // helper the six architecture loaders call, so its table is the
        // one that has to know MXFP4.
        let direct = widen_plain_float(GgmlType::MXFP4, &MXFP4_GGUF_TEST_BLOCKS, "test.weight")
            .expect("widen_plain_float must widen MXFP4");
        assert_eq!(direct, expected);

        // And the refusal still works for a dtype that genuinely has no
        // widening path, so this test cannot pass by making everything
        // succeed.
        match widen_plain_float(GgmlType::TQ2_0, &MXFP4_GGUF_TEST_BLOCKS, "test.weight") {
            Err(LoadError::UnsupportedDtype(name, GgmlType::TQ2_0)) => {
                assert_eq!(name, "test.weight");
            }
            other => panic!("TQ2_0 must be refused by name, got {other:?}"),
        }
    }

    fn build_single_iq_lowbit_tensor_gguf(
        arch: &str,
        tag: u32,
        cols: u64,
        block: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count
        write_kv_str(&mut buf, "general.architecture", arch);
        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
        buf.write_u64::<LittleEndian>(cols).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(tag).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(block);
        buf
    }

    /// A structurally valid block of `len` bytes for any of the
    /// codebook-grid formats: every bit pattern is a legal code in all
    /// of them (the grid indices are bounded by their own bit widths),
    /// so a deterministic byte fill is a real block, not a fixture that
    /// happens to avoid the interesting paths. Only the f16 scale needs
    /// pinning, and only so the comparison below can't be NaN-vs-NaN.
    fn pseudo_iq_block(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            out.push((s >> 24) as u8);
        }
        out
    }

    /// End-to-end load+apply for the codebook-grid low-bit formats the
    /// published Dynamic GGUFs are built from: a real on-disk tensor of
    /// each type must load zero-copy as the right `QuantKind` and
    /// produce the same matvec result as dequantizing the block
    /// directly. That is the property this test exists for -- the
    /// *values* are pinned against real ggml in `frink-quant`; what
    /// can only break here is the tag -> kind -> block-stride chain,
    /// and a wrong stride silently reads the neighbouring row.
    /// Dtype tags (19/29/16/17/22/18/21/39) verified against ggml.h's
    /// enum ggml_type.
    #[test]
    fn load_weight_matrix_handles_real_on_disk_iq_lowbit_tensors_end_to_end() {
        type DequantFn = fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError>;
        // IQ1_M carries no f16 scale field; its scale is reassembled
        // from the four scale words' top nibbles, and the top nibble of
        // the last one supplies the f16 sign + high exponent bits.
        // Pinning it to 0x2 keeps the exponent out of the all-ones
        // NaN/Inf pattern whatever the rest of the fill does. The other
        // three do carry a leading f16 `d`, pinned for the same reason.
        let mut iq1m = pseudo_iq_block(frink_quant::IQ1_M_BLOCK_BYTES, 0x2907_31A0);
        iq1m[55] = (iq1m[55] & 0x0F) | 0x20;
        let mut iq2xs = pseudo_iq_block(frink_quant::IQ2_XS_BLOCK_BYTES, 0x2107_31A1);
        let mut iq2s = pseudo_iq_block(frink_quant::IQ2_S_BLOCK_BYTES, 0x2207_31A2);
        let mut iq3s = pseudo_iq_block(frink_quant::IQ3_S_BLOCK_BYTES, 0x2307_31A3);
        for blk in [&mut iq2xs, &mut iq2s, &mut iq3s] {
            blk[0..2].copy_from_slice(&half::f16::from_f32(0.115).to_le_bytes());
        }
        let cases: [(&str, u32, &[u8], QuantKind, DequantFn); 8] = [
            (
                "iq1s",
                19,
                &IQ1_S_TEST_BLOCK,
                QuantKind::IQ1S,
                frink_quant::dequant_iq1_s,
            ),
            (
                "iq1m",
                29,
                &iq1m,
                QuantKind::IQ1M,
                frink_quant::dequant_iq1_m,
            ),
            (
                "iq2xxs",
                16,
                &IQ2_XXS_TEST_BLOCK,
                QuantKind::IQ2XXS,
                frink_quant::dequant_iq2_xxs,
            ),
            (
                "iq2xs",
                17,
                &iq2xs,
                QuantKind::IQ2XS,
                frink_quant::dequant_iq2_xs,
            ),
            (
                "iq2s",
                22,
                &iq2s,
                QuantKind::IQ2S,
                frink_quant::dequant_iq2_s,
            ),
            (
                "iq3xxs",
                18,
                &IQ3_XXS_TEST_BLOCK,
                QuantKind::IQ3XXS,
                frink_quant::dequant_iq3_xxs,
            ),
            (
                "iq3s",
                21,
                &iq3s,
                QuantKind::IQ3S,
                frink_quant::dequant_iq3_s,
            ),
            (
                "mxfp4_gguf",
                39,
                &MXFP4_GGUF_TEST_BLOCKS,
                QuantKind::Mxfp4Gguf,
                frink_quant::dequant_mxfp4_gguf,
            ),
        ];
        for (name, tag, block, kind, dequant) in cases {
            let expected = dequant(block).unwrap();
            let cols = expected.len();
            let tmp = std::env::temp_dir().join(format!("frink_test_{name}_tensor.gguf"));
            std::fs::write(
                &tmp,
                build_single_iq_lowbit_tensor_gguf(name, tag, cols as u64, block),
            )
            .unwrap();
            let file = frink_gguf::GgufFile::open(&tmp).expect("file must parse");
            std::fs::remove_file(&tmp).ok();

            let matrix =
                load_weight_matrix(&file, "test.weight").expect("low-bit tensor must load");
            assert_eq!((matrix.rows(), matrix.cols()), (1, cols), "{name}");
            match &matrix {
                WeightMatrix::Quantized { kind: k, data, .. } => {
                    assert_eq!(*k, kind, "{name}");
                    assert!(data.is_mapped(), "{name} must load zero-copy");
                }
                _ => panic!("expected a Quantized matrix for {name}"),
            }

            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin()).collect();
            let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let got = matrix.apply(&x);
            assert!(
                (got[0] - expected_dot).abs() < 1e-1,
                "{name}: loaded+applied diverged from direct dequant: got={} expected={}",
                got[0],
                expected_dot
            );
        }
    }

    #[test]
    fn qwen2moe_disables_topk_renorm() {
        assert!(
            NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&"qwen2moe"),
            "qwen2moe must have norm_topk_prob=false (llama.cpp build_moe_ffn norm_w=false)"
        );
    }

    /// The `LLAMA_ROPE_TYPE_NONE` group used to be refused by name here;
    /// every row is served now, positioned the way its graph positions
    /// (`crate::position_embd`, `crate::alibi`), and what this pins is
    /// that not one of them reaches a rotation: the rule is
    /// `RopeLayers::Never` for each, at any depth it takes.
    #[test]
    fn an_architecture_with_no_rope_rotates_nothing() {
        for (arch, n_layers) in [
            ("gpt2", 12),
            ("mpt", 32),
            ("refact", 32),
            ("bloom", 30),
            ("jais", 40),
            ("baichuan", 40),
        ] {
            assert_eq!(
                crate::rope_layers::rope_layers(arch, n_layers, false, 0),
                crate::rope_layers::RopeLayers::Never,
                "{arch} positions without RoPE and must rotate nothing"
            );
            assert!(crate::capability::is_audited_generic(arch), "{arch}");
        }
    }

    /// A per-layer sliding-window ARRAY on an architecture whose graph
    /// reads the key as a scalar is IGNORED and the seeded period
    /// stands, exactly as llama.cpp does; a scalar still overrides the
    /// period.
    ///
    /// Three generations of this gate. `capability::
    /// unsupported_feature_keys` refused the key outright with the
    /// reason "not implemented in the generic decoder", which was
    /// false. Then the loader refused the ARRAY form for every
    /// architecture, on the reasoning that honouring it as a period was
    /// impossible and ignoring it would substitute the seed for the
    /// file's layout -- which is TRUE and is ALSO what llama.cpp does:
    /// `get_key_or_arr(kid, swa_period, false)` returns false on an
    /// array (`llama-model-loader.cpp:502-507`) and `plamo3.cpp:9-11`
    /// keeps its 8. Every real EXAONE-4 32B, EXAONE-MoE and Olmo-3
    /// export carries the array (`conversion/exaone.py:84`,
    /// `olmo.py:59-66`) and was refused over a value upstream never
    /// reads. `crate::swa_layers` carries which graphs read which form;
    /// the array-HONOURED mode has its own fixture in
    /// `tests/window_array_graphs.rs`.
    #[test]
    fn an_array_valued_sliding_window_pattern_is_ignored_where_llama_cpp_ignores_it() {
        // Disagrees with plamo3's seeded last-dense 8 on layers 0..3,
        // so honouring it would be visible.
        let pattern: [u32; 4] = [0, 0, 0, 0];
        let kvs: Vec<(&str, Kv)> = vec![
            ("general.architecture", Kv::Str("plamo3")),
            ("plamo3.block_count", Kv::U32(4)),
            ("plamo3.embedding_length", Kv::U32(64)),
            ("plamo3.attention.head_count", Kv::U32(1)),
            ("plamo3.attention.head_count_kv", Kv::U32(1)),
            ("plamo3.attention.key_length", Kv::U32(64)),
            ("plamo3.rope.freq_base", Kv::F32(10_000.0)),
            ("plamo3.attention.sliding_window", Kv::U32(3)),
            (
                "plamo3.attention.sliding_window_pattern",
                Kv::Arr32(&pattern),
            ),
        ];
        let file = open_metadata_gguf("swa_pattern_array", &kvs);
        let config = ModelConfig::from_gguf(&file).expect("the array is not a refusal");
        assert_eq!(
            config.swa_layers,
            crate::swa_layers::SwaLayers::period(8, false),
            "plamo3.cpp:9-11 seeds 8 and the scalar overload ignores an array"
        );
        assert_eq!(config.layer_sliding_window(0), Some(3));
        assert_eq!(config.layer_sliding_window(3), Some(3));

        // And the scalar spelling of the same key overrides the seed.
        let mut scalar = kvs;
        scalar.pop();
        scalar.push(("plamo3.attention.sliding_window_pattern", Kv::U32(2)));
        let file = open_metadata_gguf("swa_pattern_scalar", &scalar);
        let config = ModelConfig::from_gguf(&file).expect("a scalar period must load");
        assert_eq!(
            config.swa_layers,
            crate::swa_layers::SwaLayers::period(2, false)
        );
        assert_eq!(config.layer_sliding_window(0), Some(3));
        assert_eq!(config.layer_sliding_window(1), None);
    }

    /// Baichuan is one `general.architecture` string covering two
    /// positional schemes, and llama.cpp picks between them on the layer
    /// count alone (`src/models/baichuan.cpp:11-14`, with its own "TODO:
    /// become GGUF KV parameter"). The 13B used to be refused HERE; it is
    /// served now, and what this pins is that the two schemes are still
    /// told apart by the count, on both tables that must agree about it
    /// (`crate::alibi`, `crate::rope_layers`).
    #[test]
    fn baichuan_13b_positions_by_alibi_and_the_7b_rotates() {
        assert_eq!(
            crate::alibi::max_alibi_bias("baichuan", 40, None),
            Some(8.0)
        );
        assert_eq!(
            crate::rope_layers::rope_layers("baichuan", 40, false, 0),
            crate::rope_layers::RopeLayers::Never
        );
        assert_eq!(crate::alibi::max_alibi_bias("baichuan", 32, None), None);
        assert_eq!(
            crate::rope_layers::rope_layers("baichuan", 32, false, 0),
            crate::rope_layers::RopeLayers::All
        );
        // Both sizes pass the header stage and fail on the next missing
        // hparam, which is what proves neither is gated here any more.
        for (name, n) in [("baichuan13b", 40u32), ("baichuan7b", 32)] {
            let file = open_metadata_gguf(
                name,
                &[
                    ("general.architecture", Kv::Str("baichuan")),
                    ("baichuan.block_count", Kv::U32(n)),
                ],
            );
            match ModelConfig::from_gguf(&file) {
                Err(LoadError::MissingHparam(key)) => assert_eq!(key, "baichuan.embedding_length"),
                other => panic!("{name} must pass the header stage, got {other:?}"),
            }
        }
    }

    /// EXAONE-4 is ONE architecture string over TWO graphs, and
    /// llama.cpp picks between them off the LAYER COUNT with no GGUF key
    /// involved. It used to be refused for it; both sizes run now, and
    /// this is the test that says they run DIFFERENTLY.
    ///
    /// `exaone4.cpp:4-9` wraps the entire SWA setup in
    /// `if (hparams.n_layer() == 64)`, and :116 then gates rotation on
    /// it -- `use_rope = is_swa(il) || swa_type == NONE`. So:
    ///
    /// * 64 layers: a window, `set_swa_pattern(4)` last-dense, and the
    ///   FULL-ATTENTION layer of every period gets no rotation at all.
    /// * 30 layers: no window whatever the file declares, and every
    ///   layer rotates.
    ///
    /// Both halves are here because the gate is a layer-count EQUALITY.
    /// A one-sided version would pass while windowing the 1.2B off a key
    /// llama.cpp never reaches, which is the divergence
    /// `capability::swa_disabled_by_arch` was extended to stop -- and it
    /// would then rope three layers in four of the 1.2B not at all.
    #[test]
    fn the_two_exaone4_sizes_get_different_windows_and_different_rotation() {
        // `Kv` is not `Clone`, so the shared header is a builder rather
        // than a value; both sizes must read from one list or the test
        // compares two transcriptions.
        let base = |n_layers: u32| -> Vec<(&str, Kv)> {
            vec![
                ("general.architecture", Kv::Str("exaone4")),
                ("exaone4.block_count", Kv::U32(n_layers)),
                ("exaone4.embedding_length", Kv::U32(32)),
                ("exaone4.attention.head_count", Kv::U32(4)),
                ("exaone4.attention.head_count_kv", Kv::U32(2)),
                ("exaone4.attention.key_length", Kv::U32(8)),
                ("exaone4.attention.value_length", Kv::U32(8)),
                ("exaone4.rope.freq_base", Kv::F32(10_000.0)),
                // The SAME declared window for both sizes: that is the
                // whole point. Only the layer count may change the
                // answer.
                ("exaone4.attention.sliding_window", Kv::U32(4096)),
            ]
        };

        let file = open_metadata_gguf("exaone4_32b", &base(64));
        let cfg = ModelConfig::from_gguf(&file).expect("EXAONE-4 32B loads");
        assert_eq!(cfg.sliding_window, Some(4096));
        assert_eq!(
            cfg.swa_layers,
            crate::swa_layers::SwaLayers::period(4, false),
            "exaone4.cpp:7-9, and set_swa_pattern's default phase"
        );
        for il in 0..64 {
            assert_eq!(
                cfg.layer_rotates(il),
                il % 4 != 3,
                "layer {il} of EXAONE-4 32B: only the sliding layers rotate"
            );
        }

        let file = open_metadata_gguf("exaone4_1_2b", &base(30));
        let cfg = ModelConfig::from_gguf(&file).expect("EXAONE-4 1.2B loads");
        assert_eq!(
            cfg.sliding_window, None,
            "exaone4.cpp:4 never reaches set_swa_pattern below 64 layers, \
             so the declared window is dead metadata"
        );
        for il in 0..30 {
            assert!(cfg.layer_rotates(il), "layer {il} of EXAONE-4 1.2B");
        }
    }

    /// NextN/MTP blocks are inside `block_count` and llama.cpp skips
    /// them (`n_layer = n_layer_all - n_layer_nextn`, llama-hparams.cpp
    /// :280-282). For a graph that reads the key (`exaone-moe.cpp:23`)
    /// the trunk is what loads; the key is written as `0` by
    /// `conversion/exaone.py:146` for every EXAONE-MoE export without an
    /// MTP head, so zero must be the whole file. A nonzero value on a
    /// graph that does NOT read the key stays refused
    /// (`mtp_blocks::tests`).
    ///
    /// The second half pins the ORDER of two reads in `exaone4.cpp`:
    /// `:4` tests `n_layer() == 64` before `:18` reads the key, so it
    /// sees `block_count`. A 64-trunk file with one MTP block appended
    /// is 65 there and gets NO window in llama.cpp; frink feeds
    /// `block_count` to the same gate and gets the same answer.
    #[test]
    fn nextn_predict_layers_subtracts_the_trunk_for_a_reader_and_zero_is_the_whole_file() {
        let base = |nextn: u32| -> Vec<(&str, Kv)> {
            vec![
                ("general.architecture", Kv::Str("exaone-moe")),
                ("exaone-moe.block_count", Kv::U32(5)),
                ("exaone-moe.nextn_predict_layers", Kv::U32(nextn)),
                ("exaone-moe.embedding_length", Kv::U32(32)),
                ("exaone-moe.attention.head_count", Kv::U32(4)),
                ("exaone-moe.attention.head_count_kv", Kv::U32(2)),
                ("exaone-moe.attention.key_length", Kv::U32(8)),
                ("exaone-moe.attention.value_length", Kv::U32(8)),
                ("exaone-moe.rope.freq_base", Kv::F32(10_000.0)),
                ("exaone-moe.attention.sliding_window", Kv::U32(128)),
                ("exaone-moe.expert_count", Kv::U32(4)),
                ("exaone-moe.expert_used_count", Kv::U32(2)),
                ("exaone-moe.expert_gating_func", Kv::U32(2)),
            ]
        };
        let cfg = ModelConfig::from_gguf(&open_metadata_gguf("exaone_moe_mtp", &base(1)))
            .expect("a reader with an MTP block loads its trunk");
        assert_eq!((cfg.n_layers, cfg.n_mtp_blocks), (4, 1));
        let cfg = ModelConfig::from_gguf(&open_metadata_gguf("exaone_moe_no_mtp", &base(0)))
            .expect("zero is the whole file");
        assert_eq!((cfg.n_layers, cfg.n_mtp_blocks), (5, 0));

        // `exaone4.cpp:4` before `:18`: 64 trunk layers plus one MTP
        // block is NOT the 32B to llama.cpp.
        let exaone4 = |block_count: u32, nextn: u32| -> Vec<(&str, Kv)> {
            vec![
                ("general.architecture", Kv::Str("exaone4")),
                ("exaone4.block_count", Kv::U32(block_count)),
                ("exaone4.nextn_predict_layers", Kv::U32(nextn)),
                ("exaone4.embedding_length", Kv::U32(32)),
                ("exaone4.attention.head_count", Kv::U32(4)),
                ("exaone4.attention.head_count_kv", Kv::U32(2)),
                ("exaone4.attention.key_length", Kv::U32(8)),
                ("exaone4.attention.value_length", Kv::U32(8)),
                ("exaone4.rope.freq_base", Kv::F32(10_000.0)),
                ("exaone4.attention.sliding_window", Kv::U32(4096)),
            ]
        };
        let with_mtp =
            ModelConfig::from_gguf(&open_metadata_gguf("exaone4_65", &exaone4(65, 1))).unwrap();
        assert_eq!((with_mtp.n_layers, with_mtp.n_mtp_blocks), (64, 1));
        assert_eq!(
            with_mtp.sliding_window, None,
            "exaone4.cpp:4 sees n_layer_all = 65 and never reaches set_swa_pattern"
        );
        let without =
            ModelConfig::from_gguf(&open_metadata_gguf("exaone4_64", &exaone4(64, 0))).unwrap();
        assert_eq!(
            without.sliding_window,
            Some(4096),
            "the same trunk without the block is the 32B"
        );
    }

    /// `expert_used_count` is scalar-or-array upstream, and the array
    /// spelling used to fall through to a DEFAULT of 2 here.
    ///
    /// `llama-model.cpp:1266` reads the key with `get_key_or_arr` in
    /// the common loader -- every architecture, `n_layer_all` entries
    /// -- and `conversion/nemotron.py:574` writes a list for
    /// Nemotron-H Puzzle, whose architecture (`nemotron_h`) frink
    /// serves. Before this test, `metadata_u64` answered `None` for an
    /// array value, the `unwrap_or_else` below it pushed a best-effort
    /// note, and the model routed top-2 on every layer whatever the
    /// file declared: the silent-wrong class, not a refusal.
    ///
    /// Both arms are pinned, because a reader that honoured the
    /// uniform case and silently averaged the varying one would pass
    /// half of this.
    #[test]
    fn a_per_layer_expert_used_count_is_honoured_when_uniform_and_refused_when_not() {
        fn file(used: Kv<'_>) -> Vec<(&'static str, Kv<'_>)> {
            vec![
                ("general.architecture", Kv::Str("llama")),
                ("llama.block_count", Kv::U32(2)),
                ("llama.embedding_length", Kv::U32(32)),
                ("llama.attention.head_count", Kv::U32(4)),
                ("llama.attention.head_count_kv", Kv::U32(2)),
                ("llama.attention.key_length", Kv::U32(8)),
                ("llama.attention.value_length", Kv::U32(8)),
                ("llama.rope.freq_base", Kv::F32(10_000.0)),
                ("llama.expert_count", Kv::U32(8)),
                ("llama.expert_used_count", used),
            ]
        }
        let scalar = ModelConfig::from_gguf(&open_metadata_gguf(
            "experts_used_scalar",
            &file(Kv::U32(3)),
        ))
        .expect("the scalar spelling loads");
        assert_eq!(scalar.moe.n_experts_active, 3);

        let uniform = ModelConfig::from_gguf(&open_metadata_gguf(
            "experts_used_uniform",
            &file(Kv::Arr32(&[3, 3])),
        ))
        .expect("a uniform array is that one value");
        assert_eq!(
            uniform.moe.n_experts_active, 3,
            "an array of one repeated value is the scalar, not the default of 2"
        );

        let err = ModelConfig::from_gguf(&open_metadata_gguf(
            "experts_used_varying",
            &file(Kv::Arr32(&[3, 5])),
        ))
        .expect_err("a varying array has no single top-k and must stop");
        let msg = format!("{err}");
        assert!(
            msg.contains("expert_used_count") && msg.contains("PER-LAYER"),
            "the refusal must name the key and what is wrong with it: {msg}"
        );
    }

    /// An `olmo2` file carrying BOTH a sliding window and a RoPE
    /// scaling ropes its two kinds of layer differently, and frink
    /// carries one scaling for the whole model.
    ///
    /// `olmo2.cpp:120-134` runs the sliding layers with the scaling
    /// switched off -- `freq_scale = 1`, `ext_factor = 0`,
    /// `attn_factor = 1`, and the comment above it says so in as many
    /// words -- while :136-146 gives the full-attention layers the
    /// model's own. Rotating half the layers at a magnitude the
    /// checkpoint never trained at is the ALiBi class of divergence and
    /// runs fluently.
    ///
    /// Both negative halves are here because the gate is a CONJUNCTION
    /// and a gate that fires on either half alone would refuse every
    /// OLMo-2 checkpoint ever published.
    #[test]
    fn olmo2_is_refused_only_when_it_has_a_window_and_a_rope_scaling_together() {
        // `Kv` is not `Clone`, so the shared header is a builder
        // rather than a value -- which also keeps the three cases
        // reading from one list instead of three transcriptions.
        let base = || -> Vec<(&str, Kv)> {
            vec![
                ("general.architecture", Kv::Str("olmo2")),
                ("olmo2.block_count", Kv::U32(2)),
                ("olmo2.embedding_length", Kv::U32(24)),
                ("olmo2.attention.head_count", Kv::U32(4)),
                ("olmo2.attention.head_count_kv", Kv::U32(2)),
                ("olmo2.attention.key_length", Kv::U32(6)),
                ("olmo2.attention.value_length", Kv::U32(6)),
                ("olmo2.rope.freq_base", Kv::F32(10_000.0)),
            ]
        };

        let mut both = base();
        both.push(("olmo2.attention.sliding_window", Kv::U32(3)));
        both.push(("olmo2.rope.scaling.type", Kv::Str("yarn")));
        both.push(("olmo2.rope.scaling.factor", Kv::F32(4.0)));
        let file = open_metadata_gguf("olmo2_swa_yarn", &both);
        match ModelConfig::from_gguf(&file) {
            Err(LoadError::UnsupportedFeature(arch, msg)) => {
                assert_eq!(arch, "olmo2");
                assert!(msg.contains("sliding window"), "{msg}");
                assert!(msg.contains("yarn"), "{msg}");
            }
            other => panic!("olmo2 with a window AND yarn must refuse, got {other:?}"),
        }

        // A window with no scaling: both of llama.cpp's RoPE branches
        // reduce to the same plain rotation, and the difference is
        // masking alone, which frink implements.
        let mut window_only = base();
        window_only.push(("olmo2.attention.sliding_window", Kv::U32(3)));
        let file = open_metadata_gguf("olmo2_swa_only", &window_only);
        let config = ModelConfig::from_gguf(&file).expect("a window with no scaling must load");
        assert_eq!(config.sliding_window, Some(3));
        // olmo2.cpp:9-11: the period defaults to 4 and `set_swa_pattern`
        // leaves `dense_first` false.
        assert_eq!(
            config.swa_layers,
            crate::swa_layers::SwaLayers::period(4, false)
        );

        // Scaling with no window: one RoPE for the whole model, which is
        // what frink carries.
        let mut scaling_only = base();
        scaling_only.push(("olmo2.rope.scaling.type", Kv::Str("yarn")));
        scaling_only.push(("olmo2.rope.scaling.factor", Kv::F32(4.0)));
        let file = open_metadata_gguf("olmo2_yarn_only", &scaling_only);
        let config = ModelConfig::from_gguf(&file).expect("scaling with no window must load");
        assert_eq!(config.sliding_window, None);
    }

    /// The hyper-parameters a real Gemma-3 GGUF header carries for one
    /// size. `block_count` is the field llama.cpp's `LLM_TYPE_27B`
    /// switch reads (`gemma3.cpp:20-28`), so it is never a free
    /// parameter here.
    ///
    /// `linear_factor` adds the pair `conversion/base.py:1222-1230`
    /// writes from `rope_parameters["full_attention"]` -- and only from
    /// there: its own comment is "TODO: Handle sliding_attention
    /// similarly when models start implementing it", so the sliding
    /// layers get no scaling key at all.
    fn gemma3_config(
        tag: &str,
        n_layers: u32,
        hidden_dim: u32,
        n_heads: u32,
        head_dim: u32,
        linear_factor: Option<f32>,
    ) -> ModelConfig {
        let mut kvs: Vec<(&str, Kv)> = vec![
            ("general.architecture", Kv::Str("gemma3")),
            ("gemma3.block_count", Kv::U32(n_layers)),
            ("gemma3.embedding_length", Kv::U32(hidden_dim)),
            ("gemma3.attention.head_count", Kv::U32(n_heads)),
            ("gemma3.attention.head_count_kv", Kv::U32(n_heads)),
            ("gemma3.attention.key_length", Kv::U32(head_dim)),
            ("gemma3.attention.value_length", Kv::U32(head_dim)),
            // Global layers rotate at 1e6; the sliding ones fall back to
            // llama.cpp's `rope_freq_base_train_swa` default of 10000,
            // because `gemma3.cpp:11` reads only the BASE key.
            ("gemma3.rope.freq_base", Kv::F32(1_000_000.0)),
            ("gemma3.attention.sliding_window", Kv::U32(1024)),
            ("gemma3.attention.sliding_window_pattern", Kv::U32(6)),
        ];
        if let Some(factor) = linear_factor {
            kvs.push(("gemma3.rope.scaling.type", Kv::Str("linear")));
            kvs.push(("gemma3.rope.scaling.factor", Kv::F32(factor)));
        }
        ModelConfig::from_gguf(&open_metadata_gguf(tag, &kvs)).expect("gemma3 fixture must load")
    }

    /// The 27B attention scale reaches `ModelConfig`, and no other
    /// Gemma-3 size acquires one.
    ///
    /// `capability::attention_scale_override` is where the arithmetic is
    /// checked; this pins that the LOADER calls it with this file's own
    /// numbers. That step is the one that shipped broken: the function
    /// did not exist and `attention_scale` was the literal `None`, under
    /// a comment naming the exception. A helper nobody calls looks
    /// exactly like a fix.
    #[test]
    fn a_gemma3_27b_header_sets_the_attention_scale_and_no_smaller_size_does() {
        // Gemma-3-27B: 62 layers, n_embd 5376, 32 heads, head_dim 128.
        let big = gemma3_config("g3_27b_scale", 62, 5376, 32, 128, Some(8.0));
        let want = 1.0f32 / (5376.0f32 / 32.0).sqrt();
        let got = big
            .attention_scale
            .expect("Gemma-3-27B is llama.cpp's LLM_TYPE_27B");
        assert!(
            (got - want).abs() < 1e-7,
            "want 1/sqrt(168) = {want}, got {got}"
        );
        // The bug's magnitude: scores were sqrt(168/128) = 1.146x too
        // large without this.
        let kernel = 1.0f32 / 128.0f32.sqrt();
        assert!((kernel / got - (168.0f32 / 128.0).sqrt()).abs() < 1e-5);

        // Gemma-3-1B and -4B take llama.cpp's other branch, which is the
        // scale the attention kernels already apply. A `Some` here would
        // double-scale them.
        for (tag, n_layers, hidden, heads) in
            [("g3_1b_scale", 26, 1152, 4), ("g3_4b_scale", 34, 2560, 8)]
        {
            let cfg = gemma3_config(tag, n_layers, hidden, heads, 256, None);
            assert_eq!(
                cfg.attention_scale, None,
                "{tag} must keep the kernels' own 1/sqrt(head_dim)"
            );
        }
    }

    /// Gemma-3's declared linear scaling reaches the FULL-ATTENTION
    /// layers only, and the sliding ones rope unscaled.
    ///
    /// `gemma3.cpp:11` reads `LLM_KV_ROPE_FREQ_BASE_SWA` and nothing
    /// else, so `rope_freq_scale_train_swa` keeps its `1.0f` default
    /// (`src/llama-hparams.h:129`) while `get_rope_freq_scale`
    /// (`llama-model.cpp:2033-2035`) hands the trained scale to the full
    /// layers. The converter agrees: `conversion/base.py:1222-1230`
    /// takes the factor from `rope_parameters["full_attention"]` and
    /// writes nothing for the sliding half.
    ///
    /// frink folded the factor into ONE global `rope_freqs` vector, so
    /// Gemma-3-4B/12B/27B rotated five layers in six at `p/8`.
    #[test]
    fn gemma3_linear_scaling_reaches_the_full_layers_and_not_the_sliding_ones() {
        // Gemma-3-4B: 34 layers, head_dim 256, `rope_scaling: linear 8`.
        let cfg = gemma3_config("g3_4b_rope", 34, 2560, 8, 256, Some(8.0));
        let freqs = cfg
            .rope_freqs
            .as_ref()
            .expect("declared linear scaling must produce per-band divisors");
        assert!(
            freqs.full.iter().all(|f| (*f - 8.0).abs() < 1e-6),
            "full-attention layers divide every band by the trained factor: {:?}",
            freqs.full
        );
        let swa = freqs
            .swa
            .as_ref()
            .expect("gemma3 does not assign rope_freq_scale_train_swa, so 1.0 applies");
        assert!(
            swa.iter().all(|f| (*f - 1.0).abs() < 1e-6),
            "sliding layers rope at the raw position: {swa:?}"
        );
        assert_eq!(swa.len(), freqs.full.len(), "one divisor per rotated pair");

        // Period 6, last-dense (`capability::default_swa_layout`), so
        // layer 5 is the full-attention one and 0..=4 slide. The phase
        // matters: getting it wrong swaps which five-sixths are wrong.
        assert!(cfg.layer_sliding_window(0).is_some());
        assert!(cfg.layer_sliding_window(5).is_none());
        assert_eq!(
            cfg.layer_rope(0),
            Some(crate::config::LayerRopeParams {
                theta: 10_000.0,
                freq_factors: Some(&[1.0f32; 128][..]),
                rot_dim: None,
            })
        );
        assert_eq!(
            cfg.layer_rope(5),
            Some(crate::config::LayerRopeParams {
                theta: 1_000_000.0,
                freq_factors: Some(&[8.0f32; 128][..]),
                rot_dim: None,
            })
        );
        assert!(
            cfg.rope_freqs_vary_by_layer(),
            "the fused Metal stacks take one divisor slice for a whole run \
             and so must refuse this model"
        );

        // Gemma-3-1B declares no scaling at all -- the audited fixture,
        // and the reason this was invisible. Nothing to split, so no
        // per-layer set and no Metal refusal.
        let plain = gemma3_config("g3_1b_rope", 26, 1152, 4, 256, None);
        assert!(plain.rope_freqs.is_none());
        assert!(!plain.rope_freqs_vary_by_layer());
    }

    /// Gemma-2 is the counter-case, and it is why the SWA scale needs
    /// its own table rather than reusing `swa_rope_base_follows_model`.
    ///
    /// `gemma2.cpp:10-11` assigns BOTH `rope_freq_base_train_swa` and
    /// `rope_freq_scale_train_swa` from the model's trained values, so
    /// its sliding layers keep the declared scaling. Splitting them here
    /// would be the same bug pointed the other way.
    #[test]
    fn gemma2_sliding_layers_inherit_the_trained_rope_scale() {
        let cfg = ModelConfig::from_gguf(&open_metadata_gguf(
            "g2_rope",
            &[
                ("general.architecture", Kv::Str("gemma2")),
                ("gemma2.block_count", Kv::U32(26)),
                ("gemma2.embedding_length", Kv::U32(2304)),
                ("gemma2.attention.head_count", Kv::U32(8)),
                ("gemma2.attention.head_count_kv", Kv::U32(4)),
                ("gemma2.attention.key_length", Kv::U32(256)),
                ("gemma2.attention.value_length", Kv::U32(256)),
                ("gemma2.rope.freq_base", Kv::F32(10_000.0)),
                ("gemma2.attention.sliding_window", Kv::U32(4096)),
                ("gemma2.rope.scaling.type", Kv::Str("linear")),
                ("gemma2.rope.scaling.factor", Kv::F32(8.0)),
            ],
        ))
        .expect("gemma2 fixture must load");

        let freqs = cfg.rope_freqs.as_ref().expect("linear scaling declared");
        assert_eq!(
            freqs.swa, None,
            "gemma2.cpp:11 assigns rope_freq_scale_train_swa from the trained scale"
        );
        assert!(!cfg.rope_freqs_vary_by_layer());
        // Period 2, last-dense: layer 0 slides, layer 1 does not, and
        // both get the same divisors.
        assert!(cfg.layer_sliding_window(0).is_some());
        assert!(cfg.layer_sliding_window(1).is_none());
        assert_eq!(cfg.layer_rope(0), cfg.layer_rope(1));

        // The two tables really are different: this is the pair that
        // must not be collapsed into one.
        assert!(crate::capability::swa_rope_scale_follows_model("gemma2"));
        assert!(!crate::capability::swa_rope_scale_follows_model("gemma3"));
        for arch in ["olmo2", "laguna"] {
            assert!(
                crate::capability::swa_rope_base_follows_model(arch),
                "{arch} seeds the SWA base from the model"
            );
            assert!(
                !crate::capability::swa_rope_scale_follows_model(arch),
                "{arch} pins the SWA scale to 1.0 (olmo2.cpp:14, laguna.cpp:48)"
            );
        }
    }
}
