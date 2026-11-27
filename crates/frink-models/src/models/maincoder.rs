//! `maincoder`: `src/models/maincoder.cpp` at the pinned commit.
//!
//! Llama's pre-norm body with one difference: **the per-head Q/K norm
//! runs AFTER RoPE**. `:78-90` rope Q and K, then `:92` and `:95` norm
//! them. There is no GGUF key for the order; llama.cpp writes it into
//! each hand-written graph, so the only place it can come from is the
//! architecture string. Getting it wrong changes every layer's attention
//! scores without changing a single tensor shape. The audited majority
//! is the other way round (`qwen3moe.cpp:99,108`,
//! `bailingmoe2.cpp:123-135` norm first), which is why the decoder's
//! default is "before" and this row is the exception.
//!
//! Evidence: `tests/one_match_arm_graphs.rs`, against libllama.

use super::facts::{ArchFacts, TensorFacts};
use super::ModelDef;

pub const FACTS: ArchFacts = ArchFacts {
    tensors: TensorFacts {
        qk_norm_after_rope: true,
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["maincoder"],
    source: "src/models/maincoder.cpp",
    facts: FACTS,
    evidence: Some(
        "tests/one_match_arm_graphs.rs: QK norm after RoPE, a fixture against libllama's logits",
    ),
};
