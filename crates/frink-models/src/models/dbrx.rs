//! `dbrx` (DBRX): `src/models/dbrx.cpp` at the pinned commit.
//!
//! - **A weighted LayerNorm** (`LLM_NORM` with a weight and no bias) at
//!   every site (`:71,112,142`).
//! - **The pre-FFN norm is stored as `attn_output_norm`.** `:34` creates
//!   `attn_out_norm` and no `ffn_norm`, and `:110-113` norm `ffn_inp`
//!   (the post-attention residual) with it: the slot `gpt-oss` keeps
//!   under the other name.
//! - **`attention.clamp_kqv` is REQUIRED** (`:5`, no default) and clamps
//!   Q, K and V inside the shared `build_qkv` (`crate::clamp_kqv`).
//! - Routed FFN with the SOFTMAX literal and `norm_w = true`
//!   (`:115-124`), the defaults.
//!
//! Evidence: `tests/dbrx_graphs.rs`, against libllama.

use super::facts::{ArchFacts, ScalingFacts, TensorFacts};
use super::ModelDef;
use crate::norm::NormFunction;
use crate::norm_sites::{NormSites, StoredNorm};

pub const FACTS: ArchFacts = ArchFacts {
    scaling: ScalingFacts {
        clamp_kqv: Some(true),
        ..ArchFacts::DEFAULT.scaling
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNorm,
            ffn: Some(StoredNorm::required(&["attn_output_norm"])),
            ..NormSites::GENERIC
        },
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["dbrx"],
    source: "src/models/dbrx.cpp",
    facts: FACTS,
    evidence: Some("tests/dbrx_graphs.rs: a fixture against libllama's logits"),
};
