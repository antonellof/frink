//! `step35`: `src/models/step35.cpp` at the pinned commit.
//!
//! - **Standard SWA from a per-layer array** (`:26`, `get_arr`), with
//!   full-attention layers rotating half the file's width (`:9`,
//!   `n_rot_full / 2`) and sliding layers dropping `rope_freqs.weight`
//!   (`:247`).
//! - **Per-layer heads and KV widths** (`:76-78,208-209`) and
//!   **per-layer SwiGLU clamp arrays** (`:28-29`, clamp after SiLU).
//! - **An optional per-head sigmoid gate on the attention output**
//!   (`:96,268-284`, [`crate::attn_gate::STEP35`]).
//! - **Routing read from the file**: `expert_weights_scale`,
//!   `expert_weights_norm`, and SIGMOID when the gating key is absent
//!   (`:19-20`).
//! - **NextN MTP blocks** created past the trunk (`:170`).
//!
//! Evidence: `tests/clamped_swiglu_graphs.rs` and
//! `tests/gated_attention_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, MoeFacts, SwaFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::swa_layers::PatternKeyRead;

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        sigmoid_by_default: true,
        reads_expert_weights_norm: true,
        reads_expert_weights_scale: true,
        ..ArchFacts::DEFAULT.moe
    },
    swa: SwaFacts {
        full_layers_rotate_half: Some("step35.cpp:9"),
        layers_drop_rope_factors: Some("step35.cpp:247"),
        pattern_key_read: PatternKeyRead::PerLayerBool,
        ..ArchFacts::DEFAULT.swa
    },
    ffn: FfnFacts {
        reads_swiglu_clamps: true,
        ..ArchFacts::DEFAULT.ffn
    },
    topology: TopologyFacts {
        per_layer_shapes: Some(crate::layer_shapes::STEP35_SHAPES),
        reads_nextn: true,
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        attn_gate: Some(crate::attn_gate::STEP35),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["step35"],
    source: "src/models/step35.cpp",
    facts: FACTS,
    evidence: Some("tests/clamped_swiglu_graphs.rs: fixtures against libllama's logits"),
};
