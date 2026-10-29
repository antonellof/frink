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

pub(crate) mod arch_tables;
mod decoder_from_gguf;
mod expert_source;
mod expert_store_build;
mod layer_parts;
mod model_config;
#[cfg(feature = "metal")]
mod moe_planes;
mod rope_scaling;
mod sampling_metadata;
mod tensor_audit;
mod tensors;

pub use expert_source::{GgufExpertSource, StoredExpertLayout, StoredMatrixSpec};
pub(crate) use rope_scaling::yarn_scaling_from_gguf;
pub use tensor_audit::assert_every_tensor_consumed;
pub(crate) use tensors::{
    find_info, load_f32_vec, load_f32_vec_optional, load_weight_matrix, slice_quantized_rows,
    split_expert_tensor, widen_plain_float,
};

use frink_gguf::{GgmlType, GgufError};
use thiserror::Error;

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

// The norm-slot lists (`PRE_FFN_NORM_IS_POST_ATTENTION_NORM` and its
// siblings) live in `crate::norm_sites`, beside the table that reads
// them; the tests in `loader/tests/` still walk them.

#[cfg(test)]
mod tests;
