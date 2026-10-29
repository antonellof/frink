//! `dots1` (dots.llm1): `src/models/dots1.cpp` at the pinned commit.
//!
//! A leading-dense MoE whose router reads its knobs from the file:
//! `expert_weights_scale` and `expert_weights_norm` (`:8-9`, optional)
//! and `expert_gating_func` (`:10`), all passed to `build_moe_ffn` as
//! hparams (`:145-155`) with the optional `exp_probs_b` selection bias
//! (`:48`). The two `expert_weights_*` rows say the keys are read here;
//! on every graph without them they are dead metadata.
//!
//! Evidence: `tests/moe_routing_bias.rs`, a fixture against libllama.

use super::facts::{ArchFacts, MoeFacts};
use super::ModelDef;

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        reads_expert_weights_norm: true,
        reads_expert_weights_scale: true,
        ..ArchFacts::DEFAULT.moe
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["dots1"],
    source: "src/models/dots1.cpp",
    facts: FACTS,
    evidence: Some("tests/moe_routing_bias.rs: a fixture against libllama's logits"),
};
