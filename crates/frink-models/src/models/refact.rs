//! `refact` (Refact-1.6B): `src/models/refact.cpp` at the pinned commit.
//!
//! Llama's body (RMSNorm, split Q/K/V, SwiGLU, multi-query) positioned
//! by **ALiBi with the literal bias 8** (`:12`), no key read, and no
//! RoPE ([`crate::alibi::alibi_rope_layers`]).
//!
//! Evidence: `tests/alibi_graphs.rs`, against libllama.

use super::facts::{ArchFacts, HparamRules, TopologyFacts};
use super::ModelDef;
use crate::alibi::MaxAlibiBias;
use crate::rope_layers::RopeLayers;

const ALIBI: MaxAlibiBias = MaxAlibiBias::Literal;

/// No rotation where the bias positions the tokens (`crate::alibi`).
pub fn rope_layers(_arch: &str, n_layers: usize, _window: bool, _dense: usize) -> RopeLayers {
    crate::alibi::alibi_rope_layers(ALIBI, n_layers)
}

pub const FACTS: ArchFacts = ArchFacts {
    topology: TopologyFacts {
        alibi: Some(ALIBI),
        ..ArchFacts::DEFAULT.topology
    },
    rules: HparamRules {
        rope_layers,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["refact"],
    source: "src/models/refact.cpp",
    facts: FACTS,
    evidence: Some("tests/alibi_graphs.rs: a fixture against libllama's logits"),
};
