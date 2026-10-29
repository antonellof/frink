//! `granite_swa` (Granite with interleaved SWA):
//! `src/models/granite-swa.cpp` at the pinned commit.
//!
//! - **Granite's four multipliers** (`:7-10`, logit REQUIRED),
//!   [`MultiplierSupport::GRANITE`].
//! - **Standard SWA from a per-layer array** (`:17-19`, `get_arr`), the
//!   window REQUIRED.
//! - **The per-layer rotate array**, `attention.rope_pattern` (`:43`):
//!   the one graph that reads it back (`crate::rope_layers`).
//! - **OPTIONAL biases** on `attn_output` and the dense FFN's three.
//!
//! The window is REQUIRED here and not refused when absent (#470).
//!
//! Evidence: `tests/granite_swa_graphs.rs`, against libllama.

use super::facts::{ArchFacts, ScalingFacts, SwaFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::proj_bias::Presence;
use crate::scalar_multipliers::MultiplierSupport;
use crate::swa_layers::PatternKeyRead;

pub const FACTS: ArchFacts = ArchFacts {
    swa: SwaFacts {
        pattern_key_read: PatternKeyRead::PerLayerBool,
        ..ArchFacts::DEFAULT.swa
    },
    scaling: ScalingFacts {
        multipliers: MultiplierSupport::GRANITE,
        ..ArchFacts::DEFAULT.scaling
    },
    topology: TopologyFacts {
        reads_rope_pattern: true,
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        attn_out_bias: Some(Presence::Optional),
        ffn_bias: Some((Presence::Optional, true)),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["granite_swa"],
    source: "src/models/granite-swa.cpp",
    facts: FACTS,
    evidence: Some("tests/granite_swa_graphs.rs: a fixture against libllama's logits"),
};
