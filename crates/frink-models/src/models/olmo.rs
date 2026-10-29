//! `olmo` (OLMo-1): `src/models/olmo.cpp` at the pinned commit.
//!
//! Llama's pre-norm body with two differences, both stated below:
//!
//! - **A non-parametric LayerNorm at every site**: subtract the mean,
//!   divide by the standard deviation, no learned weight and no bias.
//!   The tensor loader creates Q/K/V, `attn_output` and gate/up/down and
//!   NOT ONE norm tensor, and the graph is `build_norm(x, NULL, NULL,
//!   LLM_NORM, il)` at `:65` (pre-attention), `:104` (pre-FFN) and
//!   `:128` (final). It is pre-norm like `llama`, so this is the norm
//!   FUNCTION and not the residual wiring (`olmo2`'s post-norm-only
//!   topology is the other one). Every `build_norm` call in llama.cpp's
//!   graphs was scanned for a null weight: three pass one to `LLM_NORM`,
//!   and all three are here; `talkie.cpp`'s five go to `LLM_NORM_RMS`,
//!   a different function.
//! - **`attention.clamp_kqv`**, read OPTIONAL at `:5` and applied to Q,
//!   K and V inside the shared `build_qkv` (`crate::clamp_kqv`).
//!
//! Evidence: `tests/olmo_graphs.rs`, against libllama.

use super::facts::{ArchFacts, ScalingFacts, TensorFacts};
use super::ModelDef;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;

pub const FACTS: ArchFacts = ArchFacts {
    scaling: ScalingFacts {
        clamp_kqv: Some(false),
        ..ArchFacts::DEFAULT.scaling
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNormNoParams,
            ..NormSites::GENERIC
        },
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["olmo"],
    source: "src/models/olmo.cpp",
    facts: FACTS,
    evidence: Some(
        "tests/olmo_graphs.rs: the non-parametric LayerNorm and the clamp, each against \
         libllama's logits",
    ),
};
