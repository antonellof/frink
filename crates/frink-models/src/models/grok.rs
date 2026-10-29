//! `grok` (Grok-1, Grok-2): `src/models/grok.cpp` at the pinned commit.
//!
//! - **Multipliers and softcaps with literal defaults.** `:5-12` seed
//!   `attn_logit_softcapping = 30`, `yarn_beta_fast = 8` and the rest,
//!   then `:14-27` read the keys optionally over them: `logit_scale`,
//!   `embedding_scale`, `attention.output_scale` (applied inside the
//!   tanh softcap, `llama-graph.cpp:2572-2582`), the attention and final
//!   softcaps (`:214-218`), and YaRN's `beta_fast` / `beta_slow` (dead
//!   metadata on every graph that does not read them).
//!   [`MultiplierSupport::GROK`]. `router_logit_softcapping` (`:20`) is
//!   read and never applied upstream.
//! - **GeGLU**: `LLM_FFN_GELU` gated, where the family rule would give
//!   SwiGLU.
//! - **Post norms under Grok's names**, both REQUIRED: `:62` creates
//!   `attn_out_norm` and `:143-146` apply it to the attention output
//!   before the residual add (the post-attention slot, not `dbrx`'s
//!   pre-FFN one; grok keeps its own `ffn_norm`, `:64`); `:75-78` create
//!   the post-FFN norm from `LAYER_OUT_NORM` (Grok-1's `rms_norm_3`) or
//!   `FFN_POST_NORM` (Grok-2), applied at `:185-188`.
//! - **A dense FFN summed with the experts** on Grok-2's routed layers,
//!   OPTIONAL, scaled by `1/sqrt(2)` (`:66-68,171-184`).
//!
//! Evidence: `tests/grok_graphs.rs` and
//! `tests/parallel_dense_ffn_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, ScalingFacts, TensorFacts};
use super::ModelDef;
use crate::norm_sites::{NormSites, StoredNorm};
use crate::scalar_multipliers::MultiplierSupport;

pub const FACTS: ArchFacts = ArchFacts {
    ffn: FfnFacts {
        geglu: true,
        ..ArchFacts::DEFAULT.ffn
    },
    scaling: ScalingFacts {
        multipliers: MultiplierSupport::GROK,
        logit_softcap: true,
        reads_yarn_beta: true,
        ..ArchFacts::DEFAULT.scaling
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            post_attn: Some(StoredNorm::required(&["attn_output_norm"])),
            post_ffn: Some(StoredNorm::required(&[
                "layer_output_norm",
                "post_ffw_norm",
            ])),
            ..NormSites::GENERIC
        },
        parallel_dense: Some(&crate::parallel_dense_ffn::GROK),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["grok"],
    source: "src/models/grok.cpp",
    facts: FACTS,
    evidence: Some(
        "tests/grok_graphs.rs and tests/parallel_dense_ffn_graphs.rs: fixtures against \
         libllama's logits",
    ),
};
