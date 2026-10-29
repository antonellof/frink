//! `cohere2moe` (Command A MoE): `src/models/cohere2moe.cpp` at the
//! pinned commit. `cohere2`'s graph with a routed FFN:
//!
//! - **The norm function from the file** (`:4-11,166`): RMSNorm when
//!   `attention.layer_norm_rms_epsilon` is present and nonzero, else the
//!   weighted LayerNorm.
//! - **Standard SWA, period 4, the full layer FIRST**
//!   (`load_swa_pattern(ml, 4, true)`, `:28`), the window REQUIRED
//!   (`:13`), base and scale seeded from the model's (`:30-32`).
//! - **RoPE on the sliding layers and the leading dense ones**
//!   (`:177-179,192`, [`rope_layers`]).
//! - **Routing read from the file**: `expert_weights_norm`,
//!   `expert_weights_scale` and `expert_gating_func` (`:19-21`), SIGMOID
//!   when the gating key is absent (`:23-25`); the shared expert's sum
//!   scaled by `0.5` (`:248-260`).
//! - **NextN blocks** (`:131`), subtracted from the layer count.
//! - The shared-norm parallel residual (`:222-266`) and `logit_scale`
//!   REQUIRED and multiplied (`:14`).
//!
//! Two keys this file reads REQUIRED are defaulted here rather than
//! refused (#464).
//!
//! Evidence: `tests/cohere2moe_graphs.rs`, against libllama.

use super::facts::{
    ArchFacts, HparamRules, MoeFacts, NormFacts, ScalingFacts, SwaFacts, TensorFacts, TopologyFacts,
};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;
use crate::rope_layers::RopeLayers;
use crate::scalar_multipliers::MultiplierSupport;
use crate::swa_layers::PatternKeyRead;

/// `cohere2moe.cpp:192`: `is_swa(il) || il < n_layer_dense_lead`. The
/// window is REQUIRED, so there always is one.
pub fn rope_layers(
    _arch: &str,
    _n_layers: usize,
    _has_window: bool,
    n_dense_lead: usize,
) -> RopeLayers {
    RopeLayers::SlidingOrLeadingDense { n_dense_lead }
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
            dense_first: true,
        }),
        window_required: Some("cohere2moe.cpp:13"),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        pattern_key_read: PatternKeyRead::ArrayThenScalar,
        ..ArchFacts::DEFAULT.swa
    },
    norm: NormFacts {
        norm_by_rms_eps_key: true,
        ..ArchFacts::DEFAULT.norm
    },
    scaling: ScalingFacts {
        multipliers: MultiplierSupport::COHERE2,
        ..ArchFacts::DEFAULT.scaling
    },
    topology: TopologyFacts {
        parallel_residual: Some(&crate::parallel_residual::COHERE2MOE),
        reads_nextn: true,
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNorm,
            ..NormSites::GENERIC
        },
        shared_expert_sum_scale: Some(0.5),
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        rope_layers,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["cohere2moe"],
    source: "src/models/cohere2moe.cpp",
    facts: FACTS,
    evidence: Some("tests/cohere2moe_graphs.rs: fixtures against libllama's logits"),
};
