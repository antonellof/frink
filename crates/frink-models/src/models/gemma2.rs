//! `gemma2` (Gemma-2): `src/models/gemma2.cpp` at the pinned commit.
//!
//! - **Standard SWA, period 2**, the full layer last (`:4-6`), with the
//!   sliding layers' RoPE base and scale seeded from the model's
//!   (`:8-9`) and the pattern key read array-then-scalar.
//! - **The 27B's attention scale** is `1 / sqrt(n_embd / n_head)`, not
//!   the kernels' `1 / sqrt(head_dim)` (`case 46`, [`attention_scale_override`]).
//! - The softcaps, GeGLU and the `sqrt(n_embd)` embedding scale come
//!   from the Gemma family in the catalog row.
//!
//! `:5` seeds `n_swa = 4096` before the optional window key; frink
//! takes no window from a file without the key (#470).
//!
//! Evidence: the Gemma-2-2B row in `benchmarks/suite.json`.

use super::facts::{ArchFacts, HparamRules, ScalingFacts, SwaFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::swa_layers::PatternKeyRead;

/// `gemma2.cpp`'s `case 46: LLM_TYPE_27B`.
pub fn attention_scale_override(
    _arch: &str,
    n_layers: usize,
    hidden_dim: usize,
    n_heads: usize,
    head_dim: usize,
) -> Option<f32> {
    crate::capability::gemma_27b_attention_scale(46, n_layers, hidden_dim, n_heads, head_dim)
}

pub const FACTS: ArchFacts = ArchFacts {
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 2,
            dense_first: false,
        }),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
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
    names: &["gemma2"],
    source: "src/models/gemma2.cpp",
    facts: FACTS,
    evidence: Some("benchmarks/suite.json: Gemma-2-2B against llama.cpp"),
};
