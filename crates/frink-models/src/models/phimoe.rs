//! `phimoe` (Phi-3.5-MoE): `src/models/phimoe.cpp` at the pinned commit.
//!
//! - **An RMSNorm with a learned weight AND bias** at every site, all
//!   REQUIRED (`:20-21,28-29,35-36`), on `phi3`'s graph, which passes
//!   the bias to `build_norm(.., LLM_NORM_RMS, ..)`. The one such graph:
//!   `grep -B3 LLM_NORM_RMS src/models/*.cpp | grep norm_b` is `phi3`
//!   (this graph), `chameleon` (passes NULL), and graphs on other
//!   engines. [`crate::norm::NormOp::RmsBias`].
//! - **REQUIRED biases** on `attn_output` (`:33`) and on the LM head,
//!   `output.bias` (`:23`).
//! - **No window**, whatever the file says. `:3-10` read no window key,
//!   so `swa_type` stays NONE and the key `conversion/phi.py:171` writes
//!   for every export is dead metadata: libllama reports `n_swa = 0` for
//!   a file declaring one ([`swa_window_override`]).
//!
//! Evidence: `tests/phimoe_graphs.rs`, against libllama.

use super::facts::{ArchFacts, HparamRules, TensorFacts};
use super::ModelDef;
use crate::capability::SwaWindowOverride;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;
use crate::proj_bias::Presence;

/// `phimoe.cpp:3-10`: the file's window is never read.
pub fn swa_window_override(_arch: &str, _block_count: usize) -> SwaWindowOverride {
    SwaWindowOverride::Drop
}

pub const FACTS: ArchFacts = ArchFacts {
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::RmsBias,
            ..NormSites::GENERIC
        },
        attn_out_bias: Some(Presence::Required),
        output_bias: Some(Presence::Required),
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        swa_window_override,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["phimoe"],
    source: "src/models/phimoe.cpp",
    facts: FACTS,
    evidence: Some("tests/phimoe_graphs.rs: a fixture against libllama's logits"),
};
