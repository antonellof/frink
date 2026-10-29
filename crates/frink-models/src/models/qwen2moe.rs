//! `qwen2moe` (Qwen1.5-MoE, Qwen2-MoE): `src/models/qwen2moe.cpp` at the
//! pinned commit.
//!
//! - **No top-k renormalisation.** `build_moe_ffn` is called with
//!   `norm_w = false` and the SOFTMAX literal (`:132-141`), and the
//!   converter writes no `expert_weights_norm`, so the file cannot say
//!   otherwise. Renormalising made Qwen1.5-MoE's greedy decode emit
//!   garbage with every tensor loaded.
//! - The shared expert is gated and sized by
//!   `expert_shared_feed_forward_length` (`:5`); frink infers it from
//!   the `_shexp` tensors when an export omits the key.
//!
//! Evidence: the Qwen1.5-MoE-A2.7B row in `benchmarks/suite.json`, and
//! `tests/qwen2moe_paris.rs`.

use super::facts::{ArchFacts, MoeFacts};
use super::ModelDef;

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        no_topk_renormalize: true,
        ..ArchFacts::DEFAULT.moe
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["qwen2moe"],
    source: "src/models/qwen2moe.cpp",
    facts: FACTS,
    evidence: Some(
        "benchmarks/suite.json: Qwen1.5-MoE-A2.7B against llama.cpp; tests/qwen2moe_paris.rs",
    ),
};
