//! `exaone-moe`: `src/models/exaone-moe.cpp` at the pinned commit.
//!
//! - **Standard SWA, period 4, unconditionally** (`:4-6`), the window
//!   REQUIRED (`:13`), the sliding layers' base and scale seeded from
//!   the model's.
//! - **RoPE on the sliding layers only** (`:136,155`): `exaone4`'s rule
//!   (`rope_layers::sliding_layers_rotate`), and since the window is
//!   required, always the sliding branch.
//! - **Routing read from the file**: `expert_weights_scale`,
//!   `expert_weights_norm`; NextN blocks (`:47`).
//!
//! `expert_gating_func` is read REQUIRED (`:16`) and defaulted here
//! (#475).
//!
//! Evidence: `tests/no_rope_layer_graphs.rs`, against libllama.

use super::facts::{ArchFacts, HparamRules, MoeFacts, SwaFacts, TopologyFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::rope_layers::RopeLayers;
use crate::swa_layers::PatternKeyRead;

/// `exaone-moe.cpp:136,155`: the sliding layers rotate, the full ones
/// do not.
pub fn rope_layers(_arch: &str, _n_layers: usize, has_window: bool, _dense: usize) -> RopeLayers {
    crate::rope_layers::sliding_layers_rotate(has_window)
}

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        reads_expert_weights_norm: true,
        reads_expert_weights_scale: true,
        ..ArchFacts::DEFAULT.moe
    },
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 4,
            dense_first: false,
        }),
        window_required: Some("exaone-moe.cpp:13"),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        pattern_key_read: PatternKeyRead::ArrayThenScalar,
        ..ArchFacts::DEFAULT.swa
    },
    topology: TopologyFacts {
        reads_nextn: true,
        ..ArchFacts::DEFAULT.topology
    },
    rules: HparamRules {
        rope_layers,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["exaone-moe"],
    source: "src/models/exaone-moe.cpp",
    facts: FACTS,
    evidence: Some("tests/no_rope_layer_graphs.rs: a fixture against libllama's logits"),
};
