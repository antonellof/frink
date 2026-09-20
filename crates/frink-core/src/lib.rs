//! frink-core: tensor primitives, quantized matmul, RMSNorm, RoPE, and
//! grouped-query causal attention with a simple KV cache.
//!
//! CPU reference implementation. The op set and naming (RMSNorm, RoPE,
//! GQA, KV cache) follow the now-standard vocabulary of the GGUF and
//! transformer-inference ecosystem; the actual Rust code below is
//! written independently. See docs/THIRD_PARTY_NOTICES.md for design credit.
//!
//! The MoE expert-residency stack -- [`expert_store`] (the byte budget
//! and the SSD tier), [`expert_cache`] (which experts stay resident and
//! the copy plans that make them so), [`expert_slots`] (the bounded
//! slot pool behind the [`expert_slots::SlotDevice`] seam),
//! [`expert_pool`] (the CUDA side of that seam), [`expert_budget`]
//! (bytes in, expert slot count out), [`residency`],
//! [`placement`] and [`qstar`] -- lives together in one crate on
//! purpose: on unified memory two independent expert budgets are the
//! same physical RAM counted twice. [`expert_store`] is the budget
//! holder. The policy half is ported from FreeToken (Apache-2.0); see
//! docs/THIRD_PARTY_NOTICES.md.

pub mod activation_tap;
pub mod alibi;
pub mod attention;
pub mod bench_profile;
pub mod block_sparse;
pub mod cache;
// The two halves of issue #27's CPU scheduling change: `cpu_pool` is
// the persistent worker pool, `par` is the one seam every CPU parallel
// region in this crate goes through and the switch between them.
pub mod cpu_pool;
pub mod csa_hca_compress;
pub mod deepseek_v4_attention;
pub mod expert_budget;
pub mod expert_cache;
// Not feature-gated: `expert_pool::split_pair` is the one part of the
// slot-copy path a compiler cannot check and a GPU-less host can, so it
// is compiled and tested by the ordinary `cargo test` run. Everything
// touching cudarc inside it carries its own `cuda` gate.
pub mod expert_pool;
pub mod expert_slots;
pub mod expert_store;
pub mod gdn;
pub mod lightning;

/// The delta rule over a chunk of tokens: one state pass per chunk.
pub mod gdn_chunk;
pub mod host_memory;
pub mod instance;
pub mod kernel_registry;
pub mod kv_block;
pub mod kv_disk;
pub mod kv_signature;
pub mod kv_swa;
pub mod mamba2;
pub mod matmul;
pub mod mla_absorbed;
pub mod par;
pub mod placement;
pub mod qstar;
pub mod recurrent_state;
pub mod residency;
pub mod summary_stats;
pub mod tensor;
pub mod threads;
pub mod vexp;
pub mod weight_matrix;

pub use attention::{
    apply_rope_back, apply_rope_interleaved, apply_rope_interleaved_back,
    apply_rope_interleaved_with_freq_factors, apply_rope_with_freq_factors, causal_gqa_attention,
    causal_gqa_attention_paged, causal_gqa_attention_paged_sinks, causal_gqa_attention_prefill,
    causal_gqa_attention_prefill_shared_kv, causal_gqa_attention_prefill_shared_kv_windowed,
    causal_gqa_attention_sinks, causal_gqa_attention_softcap, causal_gqa_attention_windowed,
    causal_gqa_attention_windowed_softcap, lightning_indexer_topk,
};
pub use cache::{
    KvBlockPool, KvCache, KvPoolExhausted, PagedKvCache, PagedKvStore, PagedStoreExhausted,
    SharedPagedKv,
};
pub use csa_hca_compress::{channel_gated_pool, compress_block};
pub use deepseek_v4_attention::{csa_attention, hca_attention};
pub use kernel_registry::Registry as KernelRegistry;
pub use kv_block::{full_blocks, BlockHash, BlockHasher};
pub use kv_disk::{
    decode_block, encode_block, encoded_len, BlockFormatError, DiskConfig, DiskKvStore, DiskStats,
    ReadHandle, ReadOutcome, StoreError,
};
pub use kv_signature::{
    CacheSignature, KvBlock, KvDtype, SignatureError, UnverifiedBlock, BLOCK_FORMAT_VERSION,
    READABLE_FORMAT_VERSIONS,
};
pub use kv_swa::{aligned_block_size, BlockLayout, BlockLayoutError};
pub use matmul::{
    geglu, gelu, matmul_f32, rms_norm, rms_norm_per_head, silu, situ_and_mul, softcap_inplace,
    swiglu,
};
pub use tensor::Tensor;
#[cfg(feature = "cuda")]
pub use weight_matrix::cuda_dense_enabled;
#[cfg(feature = "metal")]
pub use weight_matrix::metal_dense_enabled;
pub use weight_matrix::{
    active_backend, cpu_int_dot_kind_supported, cuda_matvec_kind_supported, metal_matvec_kind_name,
    metal_mul_mm_kind_supported, BatchActs, QuantKind, WeightMatrix,
};
