//! `olmo2` (OLMo-2, Olmo-3): `src/models/olmo2.cpp` at the pinned commit.
//!
//! - **Post-norm only.** The tensor loader creates only `attn_q_norm`,
//!   `attn_k_norm`, `attn_post_norm` and `ffn_post_norm` per layer, and
//!   the graph reads the raw residual at both sublayers, norming each
//!   branch's OUTPUT before its residual add (`:157`, `:174`):
//!
//!   ```text
//!   ffn_inp = x       + post_attn_norm(attn(x))
//!   out     = ffn_inp + post_ffn_norm(ffn(ffn_inp))
//!   ```
//!
//!   The same statement for statement as `exaone4.cpp`, so both state
//!   [`NormSites::POST_NORM_ONLY`].
//! - **The optional window (Olmo-3).** `:6-13`: a nonzero
//!   `attention.sliding_window` turns on STANDARD SWA with a period-4
//!   seed, seeds the sliding layers' RoPE base from the model's and pins
//!   their scale to `1.0`.
//! - **Unscaled RoPE on the sliding layers** (`:115-142`): freq_scale 1,
//!   ext_factor 0, attn_factor 1. The loader refuses a file with both a
//!   window and a RoPE scaling (`crate::swa_geometry`).
//!
//! Evidence: `tests/post_norm_only_graphs.rs`, against libllama.

use super::facts::{ArchFacts, SwaFacts, TensorFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::norm_sites::NormSites;

pub const FACTS: ArchFacts = ArchFacts {
    swa: SwaFacts {
        pattern_key_read: crate::swa_layers::PatternKeyRead::ArrayThenScalar,
        default_layout: Some(SwaPattern {
            period: 4,
            dense_first: false,
        }),
        layers_unscaled_rope: Some("olmo2.cpp:120-134"),
        rope_base_follows_model: true,
        ..ArchFacts::DEFAULT.swa
    },
    tensors: TensorFacts {
        norm_sites: NormSites::POST_NORM_ONLY,
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["olmo2"],
    source: "src/models/olmo2.cpp",
    facts: FACTS,
    evidence: Some("tests/post_norm_only_graphs.rs: a fixture against libllama's logits"),
};
