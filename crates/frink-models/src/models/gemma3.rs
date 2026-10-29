//! `gemma3` (Gemma-3): `src/models/gemma3.cpp` at the pinned commit.
//!
//! - **Optional standard SWA, period 6**, the full layer last (`:4-7`).
//!   The sliding layers' RoPE base is the key or llama.cpp's 10000
//!   default, NOT the model's, and their scale stays 1.
//! - **The 27B's attention scale** is `1 / sqrt(n_embd / n_head)`
//!   (`case 62`, [`attention_scale_override`]).
//! - The softcap, GeGLU and the embedding scale come from the Gemma
//!   family in the catalog row.
//!
//! Evidence: the Gemma-3-1B row in `benchmarks/suite.json`.

use super::facts::{ArchFacts, HparamRules, ScalingFacts, SwaFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::swa_layers::PatternKeyRead;

/// `gemma3.cpp`'s `case 62: LLM_TYPE_27B`.
pub fn attention_scale_override(
    _arch: &str,
    n_layers: usize,
    hidden_dim: usize,
    n_heads: usize,
    head_dim: usize,
) -> Option<f32> {
    crate::capability::gemma_27b_attention_scale(62, n_layers, hidden_dim, n_heads, head_dim)
}

pub const FACTS: ArchFacts = ArchFacts {
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 6,
            dense_first: false,
        }),
        pattern_key_read: PatternKeyRead::ArrayThenScalar,
        ..ArchFacts::DEFAULT.swa
    },
    scaling: ScalingFacts {
        // The Gemma family scales the embeddings by `sqrt(n_embd)`.
        embeddings_scaled_by_sqrt_n_embd: true,
        ..ArchFacts::DEFAULT.scaling
    },
    rules: HparamRules {
        attention_scale_override,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["gemma3"],
    source: "src/models/gemma3.cpp",
    facts: FACTS,
    evidence: Some("benchmarks/suite.json: Gemma-3-1B against llama.cpp"),
};
