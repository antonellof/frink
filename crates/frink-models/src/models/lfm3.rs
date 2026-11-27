//! `lfm3`: `src/models/lfm3.cpp` at the pinned commit.
//!
//! - **Hybrid short-conv + attention stack** (`:8-10,61,84-86,165-171`,
//!   `lfm-base.cpp:9-112`): layers with `head_count_kv == 0` run a short
//!   convolution block, and attention layers carry per-layer head widths
//!   (`:105-107`).
//! - **Standard SWA from a per-layer array** (`:13`, `get_arr`), with
//!   both `attention.sliding_window` (`:18`) and `rope.freq_base_swa`
//!   (`:20`) REQUIRED, and sliding layers dropping `rope_freqs.weight`
//!   (`:113`).
//! - **A required per-head sigmoid gate on the attention output**
//!   (`:77,132-139`, [`crate::attn_gate::LFM3`]).
//! - **Optional sandwich post norms, both or neither** (`:53-55`), and
//!   conv-layer outputs are post-normed with `attn_post_norm` before the
//!   residual add (`:178-181`).
//!
//! Evidence: `tests/lfm3_graphs.rs`, against libllama.

use super::facts::{ArchFacts, SwaFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::swa_layers::PatternKeyRead;

pub const FACTS: ArchFacts = ArchFacts {
    swa: SwaFacts {
        window_required: Some("lfm3.cpp:18"),
        rope_base_required: Some("lfm3.cpp:20"),
        layers_drop_rope_factors: Some("lfm3.cpp:113"),
        pattern_key_read: PatternKeyRead::PerLayerBool,
        ..ArchFacts::DEFAULT.swa
    },
    topology: TopologyFacts {
        per_layer_shapes: Some(crate::layer_shapes::LFM3_SHAPES),
        shortconv: true,
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        post_norms_both_or_neither: Some("lfm3.cpp:53-55"),
        attn_gate: Some(crate::attn_gate::LFM3),
        conv_output_post_normed: true,
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["lfm3"],
    source: "src/models/lfm3.cpp",
    facts: FACTS,
    evidence: Some("tests/lfm3_graphs.rs: two fixtures against libllama's logits"),
};
