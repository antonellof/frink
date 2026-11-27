//! `mimo2` (MiMo-V2-Flash / V2.5): `src/models/mimo2.cpp` at the pinned
//! commit.
//!
//! - **Standard SWA from a per-layer array** (`:12`, `get_arr`) and
//!   **per-layer head counts** (`:47-49,111-112`).
//! - **Split K/V head widths** (`:47-48,52,132-140,152-154`,
//!   `conversion/mimo.py:154`): `attention.value_length` may differ from
//!   `attention.key_length`.
//! - **Attention output value scale** (`:14-17,180-183`): multiplies the
//!   attention branch after `wo` when `attention.value_scale` is neither
//!   `0` nor `1`.
//! - **Hardcoded SIGMOID gating** (`:227`): the literal wins over any
//!   `expert_gating_func` key, and `expert_weights_scale` is dead
//!   metadata upstream.
//! - **NextN MTP blocks** created past the trunk (`:38`).
//!
//! Evidence: `tests/split_kv_head_dim_graphs.rs`, against libllama.

use frink_moe::GatingFunction;

use super::facts::{ArchFacts, MoeFacts, ScalingFacts, SwaFacts, TopologyFacts};
use super::ModelDef;
use crate::swa_layers::PatternKeyRead;

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        gating_literal: Some(GatingFunction::Sigmoid),
        ..ArchFacts::DEFAULT.moe
    },
    swa: SwaFacts {
        pattern_key_read: PatternKeyRead::PerLayerBool,
        ..ArchFacts::DEFAULT.swa
    },
    scaling: ScalingFacts {
        reads_attn_value_scale: true,
        ..ArchFacts::DEFAULT.scaling
    },
    topology: TopologyFacts {
        per_layer_shapes: Some(crate::layer_shapes::MIMO2_SHAPES),
        reads_nextn: true,
        split_kv_head_dims: true,
        ..ArchFacts::DEFAULT.topology
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["mimo2"],
    source: "src/models/mimo2.cpp",
    facts: FACTS,
    evidence: Some("tests/split_kv_head_dim_graphs.rs: three fixtures against libllama's logits"),
};
