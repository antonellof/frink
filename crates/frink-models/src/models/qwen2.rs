//! `qwen2` (Qwen2, Qwen2.5): `src/models/qwen2.cpp` at the pinned commit.
//!
//! Llama's body with split Q/K/V biases (loaded by presence,
//! `crate::qkv_fused`) and one row of its own: an OPTIONAL `output.bias`
//! on the LM head (`:27`, `TENSOR_NOT_REQUIRED`), added after the output
//! matmul (`crate::proj_bias`).
//!
//! Evidence: the Qwen2.5-0.5B row in `benchmarks/suite.json`, measured
//! against llama.cpp on the same host and file.

use super::facts::{ArchFacts, TensorFacts};
use super::ModelDef;
use crate::proj_bias::Presence;

pub const FACTS: ArchFacts = ArchFacts {
    tensors: TensorFacts {
        output_bias: Some(Presence::Optional),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["qwen2"],
    source: "src/models/qwen2.cpp",
    facts: FACTS,
    evidence: Some("benchmarks/suite.json: Qwen2.5-0.5B against llama.cpp"),
};
