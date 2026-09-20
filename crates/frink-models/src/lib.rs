//! frink-models: GGUF decoder, architecture registry, and structural
//! presets for frontier stacks (GLM / DeepSeek V4 / Kimi).
//!
//! Unconfirmed preset fields go in `best_effort_fields` and must be
//! overwritten from real `config.json` / GGUF metadata. Status of what
//! actually runs: `docs/MODELS.md`. Presets `glm_5_2` / `deepseek_v4_pro`
//! / `kimi_k3` are sketches for synthetic tests — not real-checkpoint
//! support. Dedicated primitives live in `glm52_*`, `deepseek_v4_*`,
//! `kimi_*` modules.

pub mod act_layers;
pub mod alibi;
pub mod attn_gate;
pub mod attn_temperature;
pub mod attn_value_scale;
pub mod bert_encoder;
pub mod bert_gguf_loader;
pub mod block_residual;
pub mod capability;
pub mod chat_template;
pub mod chunked_swa;
pub mod clamp_kqv;
pub mod config;
pub mod decoder;
pub mod deepseek_v4_budget;
pub mod deepseek_v4_decoder;
pub mod device_budget;
pub mod draft_model;
pub mod dry;
pub mod embedding_model;
pub mod encoder;
pub mod engine;
pub mod engine_factory;
pub mod execution_plan;
pub mod gdn;

/// When a whole decoder layer can run in ONE Metal submission, and the
/// weights it takes. The refusals are exhaustive by construction.
#[cfg(feature = "metal")]
pub mod fused_layer;

/// The `-ctk` value vocabulary, which is llama.cpp's.
pub mod ctk;
/// What `--list-devices` prints, shared by both front ends.
pub mod devices;
pub mod gemma4_engine;
pub mod gemma4_gguf_loader;
pub mod glm52_decoder;
pub mod glm52_gguf_loader;
pub mod glm_dsa;
pub mod grammar;
pub mod grammar_sampler;
pub mod hadamard_fold;
pub mod hf_pull;
pub mod hrm;
#[cfg(feature = "hub")]
pub mod hub;
pub mod hybrid_engine;
pub mod hyper_connections;
pub mod kda;
pub mod kimi_decoder;
pub mod kimi_generate;
pub mod kimi_gguf_loader;
pub mod kimi_loader;
pub mod kimi_tokenizer;
pub mod kimi_validate;
pub mod kv_budget;
pub mod kv_head_dims;
pub mod latent_moe;
pub mod layer_loops;
pub mod layer_shapes;
pub mod loader;
pub mod lora;
pub mod lora_attach;
pub mod mamba1;
pub mod mamba2;
/// Metal launch descriptions built from a `WeightMatrix`, in one place
/// because the decoder and the recurrent branch both read them.
#[cfg(feature = "metal")]
pub(crate) mod metal_launch;
pub mod minimax_engine;
pub mod mla;
pub mod mla_arch;
pub mod mla_gguf_loader;
pub mod mla_q_proj;
pub mod mla_yarn;
pub mod mmproj;
pub mod moe_interleave;
pub mod mrope;
pub mod mtp_blocks;
pub mod norm;
pub mod norm_sites;
pub mod output_projection;
pub mod parallel_dense_ffn;
pub mod parallel_residual;
pub mod penalty_window;
pub mod plamo2_ssm;
pub mod pooling;
pub mod position_embd;
pub mod prefix_cache;
pub mod proj_bias;
pub mod qk_layer_norm;
pub(crate) mod qkv_fused;
pub mod rank_head;
pub mod recurrent_engine;
pub mod rerank_pooler;
pub mod residency_report;
pub mod rope_finetuned;
pub mod rope_layers;
pub mod rope_ntk_alpha;
pub mod routed_weight_site;
pub mod router_input;
pub mod safetensors_f32;
pub(crate) mod sampler_chain;
pub mod sampler_order;
pub mod sampling;
pub mod scalar_multipliers;
pub mod shortconv;
pub mod skip_stream;
pub mod speculative;
pub mod ssm_block;
pub mod sub_norms;
pub mod swa_geometry;
pub mod swa_layers;
pub mod t5_engine;
pub mod tensor_role;
#[cfg(test)]
pub(crate) mod test_source;
pub mod tokenizer;
pub mod unread_tensors;
pub mod vision;
pub mod vl_engine;
pub mod weight_scales;
pub mod weightless_qk_norm;
pub mod yarn_magnitude;

pub use bert_encoder::{BertEncoder, BertHparams, BertLayer};
pub use bert_gguf_loader::{load_bert_encoder_from_path, read_bert_hparams, BERT_ARCH};
pub use capability::{
    architecture_catalog, coverage_report_markdown, resolve_architecture, resolve_profile,
    ArchPath, ArchProfile, ArchScope, DecoderFamily, MemoryKind, QkNormStyle,
};
pub use config::{deepseek_v4_pro, glm_5_2, kimi_k3, FfnActivation, ModelConfig, RopeLayout};
pub use decoder::{Decoder, MultiSeqKv};
pub use device_budget::{BudgetBackend, DeviceBudget};
pub use draft_model::{DraftModelSpeculator, VocabMismatch};
pub use embedding_model::{is_embedding_arch, EmbedError, EmbeddingModel};
pub use encoder::{EncodeError, PairSequence, TextEncoder};
pub use engine::{
    DeepseekV4Engine, Engine, Glm52Engine, KimiEngine, MlaDenseFfn, MlaEngine, MlaLayerFfn,
    MlaLayerWeights, MlaMoeFfn, MlaMoeRuntime, TextTokenizer,
};
pub use engine_factory::{
    ensure_generic_decoder, load_gemma4_engine_from_path, load_glm52_engine_from_path,
    load_mla_engine_from_path, select_engine_kind, EngineSelectError, SelectedEngineKind,
    ServedEngine,
};
pub use execution_plan::{ExecutionPlan, FusedOpCaps, MemoryPlan, PlanGeometry};
pub use gemma4_engine::{Gemma4Engine, Gemma4Hparams, GEMMA4_ARCHES};
pub use kv_budget::{
    Ceiling, ContextCap, ContextFit, KvBudget, KvBudgetError, KvElem, KvLayout, KvResidency,
    KvShape, CTX_AUTO_GRANULARITY,
};
pub use loader::LoadError;
pub use norm::NormOp;
pub use output_projection::grouped_output_projection;
pub use penalty_window::PenaltyWindow;
pub use pooling::{l2_normalize, pool, PoolingError, PoolingType};
pub use prefix_cache::{PrefixCache, PrefixCacheStats, PrefixMatch};
pub use rank_head::{load_rank_head, RankHead};
pub use rerank_pooler::{splice_pooler, SpliceError, SplicedPooler};
pub use sampler_order::{ChainStep, SamplerName, SamplerOrder, SamplerOrderError};
pub use sampling::{sampling_distribution, Sampler, SamplingParams};
pub use speculative::{
    accept_or_resample, speculative_decode, speculative_decode_with, DraftBlock, DraftDist,
    Drafter, PromptLookupSpeculator, SpeculativeDecodeResult, SpeculativeOptions,
};
pub use tensor_role::TensorRole;
pub use tokenizer::{
    ByteTokenizer, GgufBpeTokenizer, GgufPlamo2Tokenizer, GgufSpmTokenizer, GgufUnigramTokenizer,
    GgufWordPieceTokenizer, NormalizerOptions, TokenizerLoadError,
};

#[cfg(feature = "metal")]
pub use frink_metal::attn::{metal_greedy_argmax_active, set_metal_greedy_argmax};
