//! `muse-glimmer`: `src/models/muse-glimmer.cpp` at the pinned commit.
//!
//! - **Standard SWA, period 4** (`:13`), the window REQUIRED (`:4`).
//! - **RoPE on the sliding layers only, NoPE on the full ones** (`:88`,
//!   [`rope_layers`]).
//! - **A weightless RMS on the token embeddings** (`:69`) and a LITERAL
//!   `1e-8` epsilon at the post-norm sites (`:63`).
//! - **A per-element sigmoid gate on the attention output** (`:46,
//!   100-135`, [`crate::attn_gate::MUSE_GLIMMER`]).
//! - **`logit_scale` REQUIRED and multiplied** (`:8,186`) and the two
//!   logit softcaps.
//!
//! `:7` seeds the sliding layers' RoPE base from the model's, which the
//! legacy table did not say and frink does not do, and the window is
//! REQUIRED and not refused when absent. Both are divergences from the
//! pin, carried over unchanged.
//!
//! Evidence: `tests/muse_glimmer_graphs.rs`, against libllama.

use super::facts::{ArchFacts, HparamRules, NormFacts, ScalingFacts, SwaFacts, TensorFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::rope_layers::RopeLayers;
use crate::scalar_multipliers::MultiplierSupport;
use crate::swa_layers::PatternKeyRead;

/// `muse-glimmer.cpp:88`: `use_rope = hparams.is_swa(il)`.
pub fn rope_layers(_arch: &str, _n_layers: usize, _window: bool, _dense: usize) -> RopeLayers {
    RopeLayers::SlidingOnly
}

pub const FACTS: ArchFacts = ArchFacts {
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 4,
            dense_first: false,
        }),
        pattern_key_read: PatternKeyRead::ArrayThenScalar,
        ..ArchFacts::DEFAULT.swa
    },
    norm: NormFacts {
        post_norm_eps_literal: Some(1e-8),
        ..ArchFacts::DEFAULT.norm
    },
    scaling: ScalingFacts {
        multipliers: MultiplierSupport::COHERE2,
        logit_softcap: true,
        ..ArchFacts::DEFAULT.scaling
    },
    tensors: TensorFacts {
        attn_gate: Some(crate::attn_gate::MUSE_GLIMMER),
        weightless_embedding_norm: true,
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        rope_layers,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["muse-glimmer"],
    source: "src/models/muse-glimmer.cpp",
    facts: FACTS,
    evidence: Some("tests/muse_glimmer_graphs.rs: a fixture against libllama's logits"),
};
