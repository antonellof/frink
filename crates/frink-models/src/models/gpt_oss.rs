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
//!
//! `:6` reads `attention.sliding_window` as REQUIRED and frink does not
//! refuse a file without it (#458).
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
