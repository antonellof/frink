//! `baichuan` (Baichuan): `src/models/baichuan.cpp` at the pinned
//! commit.
//!
//! One architecture string, two positional schemes decided by layer
//! count with no key (`:11-14`): Baichuan-13B (40 layers) is positioned
//! by **ALiBi with the literal bias 8** and rotates nothing;
//! Baichuan-7B (32) rotates and has no bias. One row,
//! [`MaxAlibiBias::LiteralAtLayers`], decides both, so the bias and the
//! absence of rotation cannot disagree about the count
//! ([`crate::alibi::alibi_rope_layers`]).
//!
//! Evidence: `tests/alibi_graphs.rs` (the 13B shape) and
//! `tests/no_rope_layer_graphs.rs`, against libllama.

use super::facts::{ArchFacts, HparamRules, TopologyFacts};
use super::ModelDef;
use crate::alibi::MaxAlibiBias;
use crate::rope_layers::RopeLayers;

const ALIBI: MaxAlibiBias = MaxAlibiBias::LiteralAtLayers(40);

/// `baichuan.cpp:11-14`: no rotation at 40 layers, every layer otherwise.
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
    names: &["baichuan"],
    source: "src/models/baichuan.cpp",
    facts: FACTS,
    evidence: Some("tests/alibi_graphs.rs: the 13B shape against libllama's logits"),
};
