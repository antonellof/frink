//! `gpt-oss`: `src/models/openai-moe.cpp` at the pinned commit.
//!
//! - **Standard SWA, period 2**, the full-attention layer last
//!   (`:6-9`), with the sliding layers' RoPE base AND scale seeded from
//!   the model's (`:11-13`).
//! - **The pre-FFN norm is stored as `post_attention_norm`**, and there
//!   is no post-attention norm: the post-attention residual is normed
//!   with it before the FFN. A separate fact from the five extra
//!   per-layer tensors (`gpt_oss_layers`: sinks, biases, the SwiGLU
//!   clamp), so `seed_oss`, which shares the norm slot, is not handed
//!   attention sinks.
//! - **A REQUIRED `attn_output.bias`.**
//! - **`attention.sliding_window` is REQUIRED** (`:6`, no `false`): a
//!   file without it does not load upstream, and is refused here (#458).
//! - The window pattern through `load_swa_pattern(ml, 2)` (`:9`): an
//!   array in the file wins over the seeded period
//!   (`crate::swa_layers::LOAD_SWA_PATTERN_CALLERS`, #420).
//!
//! Evidence: `tests/gptoss_graph.rs`, against libllama.

use super::facts::{ArchFacts, SwaFacts, TensorFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::norm_sites::{NormSites, StoredNorm};
use crate::proj_bias::Presence;

pub const FACTS: ArchFacts = ArchFacts {
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 2,
            dense_first: false,
        }),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        window_required: Some("openai-moe.cpp:6"),
        ..ArchFacts::DEFAULT.swa
    },
    tensors: TensorFacts {
        gpt_oss_layers: true,
        norm_sites: NormSites {
            ffn: Some(StoredNorm::required(&["post_attention_norm"])),
            post_attn: None,
            ..NormSites::GENERIC
        },
        attn_out_bias: Some(Presence::Required),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["gpt-oss"],
    source: "src/models/openai-moe.cpp",
    facts: FACTS,
    evidence: Some("tests/gptoss_graph.rs: a fixture against libllama's logits"),
};
