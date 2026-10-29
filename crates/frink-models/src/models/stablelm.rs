//! `stablelm` (StableLM-2, StableLM-Zephyr): `src/models/stablelm.cpp`
//! at the pinned commit.
//!
//! - **A LayerNorm with weight and bias** at every site
//!   ([`NormFunction::LayerNormBias`]).
//! - **A parallel residual where `ffn_norm` is absent** (`:38-39`
//!   optional, `:129-138,147`): attention and FFN read the same
//!   `attn_norm(x)` ([`crate::parallel_residual::STABLELM`]).
//! - **The per-head Q/K LayerNorm** (`:34-35,84-97`, OPTIONAL), which
//!   frink refuses by name when present (`crate::qk_layer_norm`).
//!
//! Evidence: `tests/stablelm_graphs.rs` and
//! `tests/parallel_residual_graphs.rs`, against libllama.

use super::facts::{ArchFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;

pub const FACTS: ArchFacts = ArchFacts {
    topology: TopologyFacts {
        parallel_residual: Some(&crate::parallel_residual::STABLELM),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNormBias,
            ..NormSites::GENERIC
        },
        per_head_layer_norm_qk: Some("src/models/stablelm.cpp:34-35,84-97"),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["stablelm"],
    source: "src/models/stablelm.cpp",
    facts: FACTS,
    evidence: Some("tests/stablelm_graphs.rs: fixtures against libllama's logits"),
};
