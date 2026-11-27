//! `talkie`: `src/models/talkie.cpp` at the pinned commit.
//!
//! Llama's body with four things no other graph has together:
//!
//! - **Weightless norms** (`:50,68,90,110,137`): every `build_norm` is
//!   `(x, nullptr, nullptr, LLM_NORM_RMS)`, so no site loads a norm
//!   tensor ([`NormFunction::RmsNoParams`]).
//! - **A per-head scalar Q gain** (`:26`, `attn_q_norm` is `{1, n_head}`)
//!   applied after RoPE, with a weightless per-head K norm beside it
//!   (`:82-91`, no `attn_k_norm` tensor). Decided by architecture
//!   because the weight's length is ambiguous with `head_dim`.
//! - **The embedding skip stream** (`:50-52,123-126`): the embeddings
//!   are normed before layer 0 and added into every layer's output
//!   times `blk.N.layer_output_scale.weight` (`crate::skip_stream`).
//! - **A REQUIRED `logit_scale`, multiplied onto the logits**
//!   (`:5,141`, [`MultiplierSupport::TALKIE`], the `grok` use).
//!
//! The converter also writes two `{1}` projection gains
//! (`conversion/talkie.py:26-31`), served by `crate::weight_scales`.
//!
//! Evidence: `tests/skip_stream_graphs.rs`, two fixtures against
//! libllama.

use super::facts::{ArchFacts, ScalingFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;
use crate::scalar_multipliers::MultiplierSupport;

pub const FACTS: ArchFacts = ArchFacts {
    scaling: ScalingFacts {
        multipliers: MultiplierSupport::TALKIE,
        ..ArchFacts::DEFAULT.scaling
    },
    topology: TopologyFacts {
        skip_stream: true,
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::RmsNoParams,
            ..NormSites::GENERIC
        },
        per_head_scalar_qk_gain: true,
        qk_norm_after_rope: true,
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["talkie"],
    source: "src/models/talkie.cpp",
    facts: FACTS,
    evidence: Some(
        "tests/skip_stream_graphs.rs: two fixtures (with and without the converter's gains) \
         against libllama's logits",
    ),
};
