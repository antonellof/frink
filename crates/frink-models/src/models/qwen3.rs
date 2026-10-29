//! `qwen3`: `src/models/qwen3.cpp` at the pinned commit.
//!
//! The generic body with per-head Q/K RMSNorm before RoPE and an
//! explicit head width (`head_dim * n_head != n_embd`), both of which
//! the catalog row's `QkNormStyle` and the file's
//! `attention.key_length` carry. No legacy table has a row for it, so
//! every fact is the default. `cls_out` (`:29`) is the reranker head,
//! `crate::rank_head`'s.
//!
//! Evidence: the Qwen3-0.6B row in `benchmarks/suite.json`.

use super::facts::ArchFacts;
use super::ModelDef;

pub const DEF: ModelDef = ModelDef {
    names: &["qwen3"],
    source: "src/models/qwen3.cpp",
    facts: ArchFacts::DEFAULT,
    evidence: Some("benchmarks/suite.json: Qwen3-0.6B against llama.cpp"),
};
