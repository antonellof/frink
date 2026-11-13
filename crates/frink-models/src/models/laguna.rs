//! `laguna` (Laguna-XS.2): `src/models/laguna.cpp` at the pinned commit.
//!
//! - **Optional standard SWA, period 4, the full layer FIRST**
//!   (`:38-48`), the sliding layers' base seeded from the model's and
//!   their scale pinned to 1; the sliding layers rope with the scaling
//!   switched off (`:48,181-193`).
//! - **A per-head softplus gate on the attention output**
//!   ([`crate::attn_gate::LAGUNA`]) and **per-layer head counts**
//!   (`:86-88,176-177`).
//! - **Routing read from the file**, SIGMOID when the gating key is
//!   absent (`:55-56`); YaRN's beta keys read (`:10`).
//!
//! - **`leading_dense_block_count` REQUIRED** (`:14`): a file without
//!   it is refused by name (#620), not given zero dense layers.
//!
//! Evidence: `tests/gated_attention_graphs.rs`, against libllama.

use super::facts::{
    ArchFacts, HparamRules, MoeFacts, ScalingFacts, SwaFacts, TensorFacts, TopologyFacts,
};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::swa_layers::PatternKeyRead;

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
            dense_first: true,
        }),
        layers_unscaled_rope: Some("laguna.cpp:48,181-193"),
        rope_base_follows_model: true,
        pattern_key_read: PatternKeyRead::ArrayThenScalar,
        ..ArchFacts::DEFAULT.swa
    },
    scaling: ScalingFacts {
        reads_yarn_beta: true,
        ..ArchFacts::DEFAULT.scaling
    },
    topology: TopologyFacts {
        per_layer_shapes: Some(crate::layer_shapes::LAGUNA_SHAPES),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        attn_gate: Some(crate::attn_gate::LAGUNA),
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        required_keys: &[("leading_dense_block_count", "laguna.cpp:14")],
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["laguna"],
    source: "src/models/laguna.cpp",
    facts: FACTS,
    evidence: Some("tests/gated_attention_graphs.rs: a fixture against libllama's logits"),
};
