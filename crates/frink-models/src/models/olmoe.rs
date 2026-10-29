//! `olmoe` (OLMoE): `src/models/olmoe.cpp` at the pinned commit.
//!
//! Llama's attention (with Q/K norm) over a routed FFN called with the
//! SOFTMAX literal and `norm_w = false` (`:136-145`): the selected
//! experts' weights are NOT renormalised, and no converter writes
//! `expert_weights_norm` to say so. Renormalising them made OLMoE emit
//! garbage with every tensor loaded, which is why the row exists.
//!
//! Evidence: the OLMoE-1B-7B row in `benchmarks/suite.json`.

use super::facts::{ArchFacts, MoeFacts};
use super::ModelDef;

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        no_topk_renormalize: true,
        ..ArchFacts::DEFAULT.moe
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["olmoe"],
    source: "src/models/olmoe.cpp",
    facts: FACTS,
    evidence: Some("benchmarks/suite.json: OLMoE-1B-7B against llama.cpp"),
};
