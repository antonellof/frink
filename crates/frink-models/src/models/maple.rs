//! `maple`: `src/models/maple.cpp` at the pinned commit.
//!
//! - **Standard SWA from a per-layer array** (`:4,10`, `get_arr`), the
//!   window REQUIRED (`:7`), the sliding layers' base and scale seeded
//!   from the model's (`:12-14`).
//! - **RoPE on the sliding layers only** (`:88`, [`rope_layers`]).
//! - **The clamped SwiGLU, clamp BEFORE the SiLU** on the routed
//!   experts (`:17`, `llama-graph.cpp:2228`).
//! - **Per-layer expert widths** read as an array (`:6`).
//!
//! The window is REQUIRED here and not refused when absent (#470).
//!
//! Evidence: `tests/no_rope_layer_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, HparamRules, SwaFacts, TopologyFacts};
use super::ModelDef;
use crate::rope_layers::RopeLayers;
use crate::swa_layers::PatternKeyRead;

/// `maple.cpp:88`: `if (hparams.is_swa(il))` around both rotations.
pub fn rope_layers(_arch: &str, _n_layers: usize, _window: bool, _dense: usize) -> RopeLayers {
    RopeLayers::SlidingOnly
}

pub const FACTS: ArchFacts = ArchFacts {
    ffn: FfnFacts {
        reads_swiglu_clamps: true,
        clamp_before_silu: true,
        ..ArchFacts::DEFAULT.ffn
    },
    swa: SwaFacts {
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        pattern_key_read: PatternKeyRead::PerLayerBool,
        ..ArchFacts::DEFAULT.swa
    },
    topology: TopologyFacts {
        per_layer_shapes: Some(crate::layer_shapes::MAPLE_SHAPES),
        ..ArchFacts::DEFAULT.topology
    },
    rules: HparamRules {
        rope_layers,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["maple"],
    source: "src/models/maple.cpp",
    facts: FACTS,
    evidence: Some("tests/no_rope_layer_graphs.rs: a fixture against libllama's logits"),
};
