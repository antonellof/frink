//! `mellum`: `src/models/mellum.cpp` at the pinned commit.
//!
//! - **Optional standard SWA, period 4** (`:6-16`), the sliding layers'
//!   base and scale seeded from the model's, the pattern key read
//!   array-then-scalar; the sliding layers rope with the scaling
//!   switched off (`:128-142`, `crate::swa_geometry`).
//! - **`expert_weights_norm` read from the file** (`:8-11`, seeded
//!   `true`) and the SOFTMAX gating hparam.
//! - **NextN blocks** (`:79`), subtracted from the layer count.
//!
//! Evidence: `tests/gated_attention_graphs.rs` and
//! `tests/window_array_graphs.rs`, against libllama.

use super::facts::{ArchFacts, MoeFacts, SwaFacts, TopologyFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::swa_layers::PatternKeyRead;

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        reads_expert_weights_norm: true,
        ..ArchFacts::DEFAULT.moe
    },
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 4,
            dense_first: false,
        }),
        layers_unscaled_rope: Some("mellum.cpp:128-142"),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        pattern_key_read: PatternKeyRead::ArrayThenScalar,
        ..ArchFacts::DEFAULT.swa
    },
    topology: TopologyFacts {
        reads_nextn: true,
        ..ArchFacts::DEFAULT.topology
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["mellum"],
    source: "src/models/mellum.cpp",
    facts: FACTS,
    evidence: Some("tests/gated_attention_graphs.rs: a fixture against libllama's logits"),
};
