//! `cohere2` (Command-R7B): `src/models/cohere2.cpp` at the pinned
//! commit. `command-r`'s graph with a window:
//!
//! - **Standard SWA, period 4**, the full layer last (`:4-5`), the
//!   sliding layers' RoPE base and scale seeded from the model's
//!   (`:7-10`), and `attention.sliding_window` REQUIRED (`:13`).
//! - **RoPE on the sliding layers only** (`:72,91`, `if (is_swa)`): the
//!   full-attention layer of each period rotates nothing
//!   ([`rope_layers`]).
//! - **A weighted LayerNorm** and the shared-norm parallel residual
//!   (`:120-134`, [`crate::parallel_residual::COHERE2`]).
//! - **`logit_scale` REQUIRED and multiplied** (`:14`),
//!   [`MultiplierSupport::COHERE2`].
//!
//! Evidence: `tests/cohere2_graphs.rs`, against libllama.

use super::facts::{ArchFacts, HparamRules, ScalingFacts, SwaFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;
use crate::rope_layers::RopeLayers;
use crate::scalar_multipliers::MultiplierSupport;
use crate::swa_layers::PatternKeyRead;

/// `cohere2.cpp:72,91`: the window is REQUIRED, so the rule is
/// unconditional.
pub fn rope_layers(_arch: &str, _n_layers: usize, _has_window: bool, _dense: usize) -> RopeLayers {
    RopeLayers::SlidingOnly
}

pub const FACTS: ArchFacts = ArchFacts {
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 4,
            dense_first: false,
        }),
        window_required: Some("cohere2.cpp:13"),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        pattern_key_read: PatternKeyRead::ArrayThenScalar,
        ..ArchFacts::DEFAULT.swa
    },
    scaling: ScalingFacts {
        multipliers: MultiplierSupport::COHERE2,
        ..ArchFacts::DEFAULT.scaling
    },
    topology: TopologyFacts {
        parallel_residual: Some(&crate::parallel_residual::COHERE2),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNorm,
            ..NormSites::GENERIC
        },
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        rope_layers,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["cohere2"],
    source: "src/models/cohere2.cpp",
    facts: FACTS,
    evidence: Some("tests/cohere2_graphs.rs: a fixture against libllama's logits"),
};
