//! `spark2_5` (Spark-2.5): `src/models/spark2-5.cpp` at the pinned
//! commit.
//!
//! - **Standard SWA from a per-layer array** (`:7-8`, `get_arr`), the
//!   window REQUIRED (`:5`).
//! - **A per-head sigmoid gate on the attention output**, REQUIRED
//!   (`:41,97-105`, [`crate::attn_gate::SPARK2_5`]).
//! - **GeGLU**: `LLM_FFN_GELU` gated (`:124`), where the family rule
//!   would give SwiGLU.
//! - **Per-layer head counts** (`:33-37,76-77`).
//!
//! `:10-12` seed the sliding layers' base and scale from the model's,
//! which the legacy table did not say and frink does not do; the window
//! is REQUIRED and not refused when absent (#470). Carried over
//! unchanged.
//!
//! Evidence: `tests/gated_attention_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, SwaFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::swa_layers::PatternKeyRead;

pub const FACTS: ArchFacts = ArchFacts {
    ffn: FfnFacts {
        geglu: true,
        ..ArchFacts::DEFAULT.ffn
    },
    swa: SwaFacts {
        pattern_key_read: PatternKeyRead::PerLayerBool,
        ..ArchFacts::DEFAULT.swa
    },
    topology: TopologyFacts {
        per_layer_shapes: Some(crate::layer_shapes::SPARK2_5_SHAPES),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        attn_gate: Some(crate::attn_gate::SPARK2_5),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["spark2_5"],
    source: "src/models/spark2-5.cpp",
    facts: FACTS,
    evidence: Some("tests/gated_attention_graphs.rs: a fixture against libllama's logits"),
};
