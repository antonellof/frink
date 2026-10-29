//! `qwen` (Qwen-1, `QWenLMHeadModel`): `src/models/qwen.cpp` at the
//! pinned commit. Not `qwen2` and not `qwen3`.
//!
//! - **`feed_forward_length` counts gate and up together.** HF's
//!   `QWenMLP` sets `ff_dim_in = config.intermediate_size // 2`;
//!   `conversion/qwen.py`'s `QwenModel` writes `intermediate_size`
//!   through unchanged (`conversion/base.py:1206`); and `qwen.cpp:33-35`
//!   therefore creates `ffn_gate`, `ffn_up` and `ffn_down` at `n_ff / 2`.
//!   frink loads the dense FFN by tensor name, so the forward pass never
//!   depended on this; what it fixes is `expert_ffn_dim`, which every
//!   memory estimate and `frink inspect-plan` row prices the FFN from
//!   (`the_declared_ffn_width_matches_the_matrices_that_load`,
//!   `tests/one_match_arm_graphs.rs`).
//! - The fused `attn_qkv` carries a bias created REQUIRED (`:28`, flag
//!   `0`); frink loads it optional, as for every fused spelling (#447).
//!
//! Evidence: `tests/one_match_arm_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts};
use super::ModelDef;

pub const FACTS: ArchFacts = ArchFacts {
    ffn: FfnFacts {
        length_counts_gate_and_up: true,
        ..ArchFacts::DEFAULT.ffn
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["qwen"],
    source: "src/models/qwen.cpp",
    facts: FACTS,
    evidence: Some("tests/one_match_arm_graphs.rs: a fixture against libllama's logits"),
};
