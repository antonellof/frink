//! `gptneox` (GPT-NeoX, Pythia): `src/models/gptneox.cpp` at the pinned
//! commit.
//!
//! - **A LayerNorm with weight and bias** at every site.
//! - **A parallel residual when the file says so** (`:5` reads
//!   `use_parallel_residual`, `:143-166`): two norms, attention over
//!   `attn_norm(x)` and the FFN over `ffn_norm(x)`
//!   ([`crate::parallel_residual::GPTNEOX`]).
//! - **REQUIRED biases** on the fused `attn_qkv` (`:66-67`), on
//!   `attn_output` and on the dense FFN's up and down (no gate: the FFN
//!   is the ungated GELU).
//!
//! Evidence: `tests/parallel_residual_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;
use crate::proj_bias::Presence;
use crate::qkv_fused::QkvTensors;

pub const FACTS: ArchFacts = ArchFacts {
    ffn: FfnFacts {
        gelu_ungated: true,
        ..ArchFacts::DEFAULT.ffn
    },
    topology: TopologyFacts {
        parallel_residual: Some(&crate::parallel_residual::GPTNEOX),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNormBias,
            ..NormSites::GENERIC
        },
        attn_out_bias: Some(Presence::Required),
        ffn_bias: Some((Presence::Required, false)),
        qkv_tensors: QkvTensors::Fused(Some(Presence::Required)),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["gptneox"],
    source: "src/models/gptneox.cpp",
    facts: FACTS,
    evidence: Some("tests/parallel_residual_graphs.rs: a fixture against libllama's logits"),
};
