//! `hunyuan-moe`: `src/models/hunyuan-moe.cpp` at the pinned commit.
//!
//! The one difference from the generic path is **the per-head Q/K norm
//! order**: `:93,104` rope Q and K, then `:110,115` norm them. There is
//! no GGUF key for the order, so it is a fact of the architecture
//! string (see [`super::maincoder`] for the same row and why the
//! default is "before").
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
    names: &["hunyuan-moe"],
    source: "src/models/hunyuan-moe.cpp",
    facts: FACTS,
    evidence: Some(
        "tests/one_match_arm_graphs.rs: QK norm after RoPE, a fixture against libllama's logits",
    ),
};
