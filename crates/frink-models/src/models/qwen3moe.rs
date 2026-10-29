//! `qwen3moe`: `src/models/qwen3moe.cpp` at the pinned commit.
//!
//! `qwen3`'s attention with a routed FFN: SOFTMAX gating with
//! renormalised top-k (`norm_w = true`, `:137-146`), which is the
//! default, and `n_ff != n_ff_exp`. No legacy table has a row for it.
//!
//! Evidence: `tests/qwen3moe_graph.rs`, a two-layer fixture against
//! libllama on all three forward paths.

use super::facts::ArchFacts;
use super::ModelDef;

pub const DEF: ModelDef = ModelDef {
    names: &["qwen3moe"],
    source: "src/models/qwen3moe.cpp",
    facts: ArchFacts::DEFAULT,
    evidence: Some("tests/qwen3moe_graph.rs: a fixture against libllama's logits"),
};
