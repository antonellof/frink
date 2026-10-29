//! `granite`: `src/models/granite.cpp` at the pinned commit.
//!
//! Llama's body with four metadata multipliers and a RoPE switch:
//!
//! - `load_arch_hparams` reads `logit_scale` (required),
//!   `residual_scale`, `embedding_scale` and `attention.scale`
//!   (optional, `granite.cpp:7-10`), applied at `:180` (`1 /
//!   f_logit_scale`), `:225` (`kq_scale`, `0.0f` meaning unset),
//!   `:235-238,288-292` (both branch outputs) and the shared
//!   `build_inp_embd`. One [`MultiplierSupport::GRANITE`] constant.
//! - `:33-37` read `rope.scaling.finetuned` as a switch for RoPE itself,
//!   default true: a file declaring false rotates nothing.
//! - The same OPTIONAL biases as `llama`: `attn_output.bias` (`:82`)
//!   and the dense FFN's three (`:100-102`).
//!
//! Evidence: `tests/granite_family_graphs.rs`, against libllama.

use super::facts::{ArchFacts, ScalingFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::proj_bias::Presence;
use crate::scalar_multipliers::MultiplierSupport;

pub const FACTS: ArchFacts = ArchFacts {
    scaling: ScalingFacts {
        multipliers: MultiplierSupport::GRANITE,
        ..ArchFacts::DEFAULT.scaling
    },
    topology: TopologyFacts {
        rope_gated_on_finetuned: true,
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        attn_out_bias: Some(Presence::Optional),
        ffn_bias: Some((Presence::Optional, true)),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["granite"],
    source: "src/models/granite.cpp",
    facts: FACTS,
    evidence: Some("tests/granite_family_graphs.rs: a dense fixture against libllama's logits"),
};
