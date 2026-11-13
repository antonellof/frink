//! `afmoe`: `src/models/afmoe.cpp` at the pinned commit.
//!
//! - **Optional standard SWA, period 4** (`:15-21`), the sliding layers'
//!   base and scale seeded from the model's.
//! - **No RoPE on the last layer of each period** (`:137-138`, the
//!   default step 4, [`rope_layers`]).
//! - **A per-element sigmoid gate on the attention output**
//!   ([`crate::attn_gate::AFMOE`]).
//! - **`sqrt(n_embd)` embedding scaling** as arithmetic (`:120`, "MuP
//!   scaling"), the one graph outside the Gemma family that does.
//! - **Routing read from the file**: `expert_weights_scale` and
//!   `expert_weights_norm`, and SIGMOID when the gating key is absent
//!   (`:29-30`).
//!
//! - **`expert_shared_count` REQUIRED** (`:7`): a file without it is
//!   refused by name (#620), not given one shared expert.
//!
//! Evidence: `tests/gated_attention_graphs.rs`, against libllama.

use super::facts::{ArchFacts, HparamRules, MoeFacts, ScalingFacts, SwaFacts, TensorFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::rope_layers::{NoRopePhase, RopeLayers};
use crate::swa_layers::PatternKeyRead;

/// `afmoe.cpp:137-138`: the default step, the last layer of each four.
pub fn rope_layers(_arch: &str, _n_layers: usize, _window: bool, _dense: usize) -> RopeLayers {
    crate::rope_layers::default_no_rope_every(NoRopePhase::LastOfPeriod)
}

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        sigmoid_by_default: true,
        reads_expert_weights_norm: true,
        reads_expert_weights_scale: true,
        ..ArchFacts::DEFAULT.moe
    },
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 4,
            dense_first: false,
        }),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        pattern_key_read: PatternKeyRead::ArrayThenScalar,
        ..ArchFacts::DEFAULT.swa
    },
    scaling: ScalingFacts {
        embeddings_scaled_by_sqrt_n_embd: true,
        ..ArchFacts::DEFAULT.scaling
    },
    tensors: TensorFacts {
        attn_gate: Some(crate::attn_gate::AFMOE),
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        rope_layers,
        required_keys: &[("expert_shared_count", "afmoe.cpp:7")],
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["afmoe"],
    source: "src/models/afmoe.cpp",
    facts: FACTS,
    evidence: Some("tests/gated_attention_graphs.rs: a fixture against libllama's logits"),
};
