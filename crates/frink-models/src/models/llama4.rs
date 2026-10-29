//! `llama4` (Llama 4 Scout / Maverick): `src/models/llama4.cpp` at the
//! pinned commit.
//!
//! - **A CHUNKED window with literals** (`:8-24`). Unless the file
//!   declares a window of zero, `swa_type = CHUNKED`, `n_swa = 8192`
//!   whatever the file says, a period-4 pattern (3 chunked, 1 full),
//!   the sliding layers' RoPE base and scale seeded from the model's,
//!   and a LITERAL attention temperature (scale 0.1, floor 8192, offset
//!   1) on the unrotated layers (`:15-17,175-176`). A declared zero
//!   takes the branch libllama aborts on, refused by name
//!   (`crate::chunked_swa`).
//! - **No RoPE on the full layer of each period** (`:145-146`, the
//!   default step 4), on the chunked branch ([`rope_layers`]).
//! - **Interleaved MoE**: `interleave_moe_layer_step` is REQUIRED
//!   (`:6`) and the loader creates experts on `(i + 1) % step == 0`
//!   only (`:49-51,64`), which frink honours.
//! - **Routing**: the SIGMOID literal with `norm_w = false`
//!   (`:228-230`), and the routing weight multiplied into the expert's
//!   INPUT (`llama-graph.cpp:1947,2086-2090`).
//! - **Weightless Q/K norm** on every model but the 128-expert one
//!   (`:43`, [`weightless_qk_norm`]).
//!
//! Evidence: `tests/llama4_graphs.rs`, against libllama.

use super::facts::{ArchFacts, HparamRules, MoeFacts, ScalingFacts, SwaFacts};
use super::ModelDef;
use crate::capability::SwaPattern;
use crate::rope_layers::{NoRopePhase, RopeLayers};
use frink_moe::GatingFunction;

/// `llama4.cpp:145-146`: with the chunked window, the default step's
/// last layer of each period rotates nothing; without one (the zero
/// branch, refused), every layer rotates.
pub fn rope_layers(_arch: &str, _n_layers: usize, has_window: bool, _dense: usize) -> RopeLayers {
    if has_window {
        crate::rope_layers::default_no_rope_every(NoRopePhase::LastOfPeriod)
    } else {
        RopeLayers::All
    }
}

/// `llama4.cpp:43`: every Llama 4 but Maverick (128 experts).
pub fn weightless_qk_norm(_arch: &str, n_experts: usize) -> bool {
    n_experts != 128
}

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        gating_literal: Some(GatingFunction::Sigmoid),
        no_topk_renormalize: true,
        routed_weight_before_ffn: true,
        interleave_step_honoured: Some("src/models/llama4.cpp:6,49-51,64"),
        ..ArchFacts::DEFAULT.moe
    },
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 4,
            dense_first: false,
        }),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        chunked: Some((8192, "src/models/llama4.cpp:5-21")),
        ..ArchFacts::DEFAULT.swa
    },
    scaling: ScalingFacts {
        attn_temperature_literal: Some(crate::attn_temperature::LLAMA4),
        ..ArchFacts::DEFAULT.scaling
    },
    rules: HparamRules {
        rope_layers,
        weightless_qk_norm,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["llama4"],
    source: "src/models/llama4.cpp",
    facts: FACTS,
    evidence: Some("tests/llama4_graphs.rs: fixtures against libllama's logits"),
};
