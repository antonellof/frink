//! `plamo3` (PLaMo-3): `src/models/plamo3.cpp` at the pinned commit.
//!
//! - **Optional standard SWA, period 8**, the full layer last (`:5-9`),
//!   the pattern key read array-then-scalar; the sliding layers' base
//!   is the key or llama.cpp's default.
//! - **Per-layer head and FFN counts** (`:39-44,110-111`).
//! - **A fused `attn_qkv` with no bias** (`:45`).
//!
//! Evidence: `tests/fixture_away_graphs.rs`, against libllama.

use super::facts::{ArchFacts, SwaFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::qkv_fused::QkvTensors;
use crate::swa_layers::PatternKeyRead;

pub const FACTS: ArchFacts = ArchFacts {
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 8,
            dense_first: false,
        }),
        pattern_key_read: PatternKeyRead::ArrayThenScalar,
        ..ArchFacts::DEFAULT.swa
    },
    topology: TopologyFacts {
        per_layer_shapes: Some(crate::layer_shapes::PLAMO3_SHAPES),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        qkv_tensors: QkvTensors::Fused(None),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["plamo3"],
    source: "src/models/plamo3.cpp",
    facts: FACTS,
    evidence: Some("tests/fixture_away_graphs.rs: a fixture against libllama's logits"),
};
