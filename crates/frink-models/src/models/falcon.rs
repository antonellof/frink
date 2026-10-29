//! `falcon` (Falcon-7B / 40B / 180B): `src/models/falcon.cpp` at the
//! pinned commit.
//!
//! - **A LayerNorm with weight and bias** at every site, and the
//!   ungated GELU FFN.
//! - **A parallel residual on every layer**, with a per-layer second
//!   norm: `:35-36` create `attn_norm_2` OPTIONAL (40B and 180B have
//!   it, 7B does not), `:79-85` norm the layer input with it for
//!   attention when present, and `:124` hand the FFN `attn_norm(x)`
//!   either way. So a 40B layer is the two-norm parallel residual with
//!   the tensor names crossed relative to `gptneox`
//!   ([`crate::parallel_residual::FALCON`],
//!   `TensorFacts::attn_norm_2_feeds_attention`).
//! - **No Q/K/V bias**: the fused `attn_qkv` is created without one
//!   (`:38`).
//!
//! Evidence: `tests/falcon_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;
use crate::qkv_fused::QkvBias;

pub const FACTS: ArchFacts = ArchFacts {
    ffn: FfnFacts {
        gelu_ungated: true,
        ..ArchFacts::DEFAULT.ffn
    },
    topology: TopologyFacts {
        parallel_residual: Some(&crate::parallel_residual::FALCON),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNormBias,
            ..NormSites::GENERIC
        },
        qkv_bias: QkvBias::NotCreated,
        attn_norm_2_feeds_attention: true,
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["falcon"],
    source: "src/models/falcon.cpp",
    facts: FACTS,
    evidence: Some("tests/falcon_graphs.rs: the 7B and 40B shapes against libllama's logits"),
};
