//! `plamo` (PLaMo-13B): `src/models/plamo.cpp` at the pinned commit.
//!
//! Llama's body with a parallel residual on every layer
//! (`:59-64,97-98,111-112`): attention and FFN read the same
//! `attn_norm(x)` ([`crate::parallel_residual::PLAMO`]). No other row.
//!
//! Evidence: `tests/parallel_residual_graphs.rs`, against libllama.

use super::facts::{ArchFacts, TopologyFacts};
use super::ModelDef;

pub const FACTS: ArchFacts = ArchFacts {
    topology: TopologyFacts {
        parallel_residual: Some(&crate::parallel_residual::PLAMO),
        ..ArchFacts::DEFAULT.topology
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["plamo"],
    source: "src/models/plamo.cpp",
    facts: FACTS,
    evidence: Some("tests/parallel_residual_graphs.rs: a fixture against libllama's logits"),
};
