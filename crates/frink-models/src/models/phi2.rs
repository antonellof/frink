//! `phi2` (Phi-2): `src/models/phi2.cpp` at the pinned commit.
//!
//! - **A LayerNorm with weight and bias** at every site, and the
//!   ungated GELU FFN.
//! - **A parallel residual on every layer** (`:67,108,116-117`):
//!   attention and FFN read the same `attn_norm(x)`
//!   ([`crate::parallel_residual::PHI2`]).
//! - **REQUIRED biases** on `attn_output`, on the dense FFN's up and
//!   down, and on the LM head (`output.bias`, `:22,136`).
//!
//! Evidence: `tests/phi2_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;
use crate::proj_bias::Presence;

pub const FACTS: ArchFacts = ArchFacts {
    ffn: FfnFacts {
        gelu_ungated: true,
        ..ArchFacts::DEFAULT.ffn
    },
    topology: TopologyFacts {
        parallel_residual: Some(&crate::parallel_residual::PHI2),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNormBias,
            ..NormSites::GENERIC
        },
        attn_out_bias: Some(Presence::Required),
        ffn_bias: Some((Presence::Required, false)),
        output_bias: Some(Presence::Required),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["phi2"],
    source: "src/models/phi2.cpp",
    facts: FACTS,
    evidence: Some("tests/phi2_graphs.rs: a fixture against libllama's logits"),
};
