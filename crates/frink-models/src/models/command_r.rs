//! `command-r` (Command-R 35B, Command-R+): `src/models/command-r.cpp`
//! at the pinned commit.
//!
//! - **A weighted LayerNorm** (`LLM_NORM` with a weight and no bias,
//!   `:68,127`).
//! - **A shared-norm parallel residual on every layer** (`:68,106-119`):
//!   attention and FFN read the same `attn_norm(x)`
//!   ([`crate::parallel_residual::COMMAND_R`]).
//! - **`logit_scale` MULTIPLIES** the logits (`:4`, optional),
//!   [`MultiplierSupport::COMMAND_R`].
//! - **The per-head Q/K LayerNorm**, REQUIRED where present
//!   (`:28-31,80,87`, Command-R+), refused by name
//!   (`crate::qk_layer_norm`).
//!
//! Evidence: `tests/command_r_graphs.rs`, against libllama.

use super::facts::{ArchFacts, ScalingFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;
use crate::scalar_multipliers::MultiplierSupport;

pub const FACTS: ArchFacts = ArchFacts {
    scaling: ScalingFacts {
        multipliers: MultiplierSupport::COMMAND_R,
        ..ArchFacts::DEFAULT.scaling
    },
    topology: TopologyFacts {
        parallel_residual: Some(&crate::parallel_residual::COMMAND_R),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNorm,
            ..NormSites::GENERIC
        },
        per_head_layer_norm_qk: Some("src/models/command-r.cpp:28-31,80,87"),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["command-r"],
    source: "src/models/command-r.cpp",
    facts: FACTS,
    evidence: Some("tests/command_r_graphs.rs: a fixture against libllama's logits"),
};
