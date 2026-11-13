//! `smallthinker` (SmallThinker): `src/models/smallthinker.cpp` at the
//! pinned commit.
//!
//! - **SWA when the file declares a window, at 4096 whatever it says**
//!   (`:4-8`: the value read is a flag, then overwritten), period 4,
//!   the full layer FIRST, base and scale seeded from the model's
//!   ([`swa_window_override`]).
//! - **No RoPE on the first layer of each four, with a window only**
//!   (`:16-18,108-109`): without one the step is `n_layer`, i.e. every
//!   layer rotates ([`rope_layers`]).
//! - **The router reads the raw layer input** (`:111,151-161`) and the
//!   FFN is the gated ReLU (`:158`, `ggml_reglu_split`).
//!
//! Evidence: `tests/router_input_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, HparamRules, MoeFacts, SwaFacts};
use super::ModelDef;
use crate::capability::{SwaPattern, SwaWindowOverride, SMALLTHINKER_PINNED_WINDOW};
use crate::rope_layers::{NoRopePhase, RopeLayers, LLAMA_CPP_DEFAULT_NO_ROPE_STEP};
use crate::router_input::RouterInput;
use crate::swa_layers::PatternKeyRead;

/// `smallthinker.cpp:8`: a declared window is replaced by 4096.
pub fn swa_window_override(_arch: &str, _block_count: usize) -> SwaWindowOverride {
    SwaWindowOverride::Pin(SMALLTHINKER_PINNED_WINDOW)
}

/// `smallthinker.cpp:16-18,108-109`: with a window, the default step
/// skips `il % 4 == 0`; without one (or with exactly four layers, where
/// the step and `n_layer` coincide) every layer rotates.
pub fn rope_layers(_arch: &str, n_layers: usize, has_window: bool, _dense: usize) -> RopeLayers {
    if has_window && n_layers != LLAMA_CPP_DEFAULT_NO_ROPE_STEP {
        crate::rope_layers::default_no_rope_every(NoRopePhase::FirstOfPeriod)
    } else {
        RopeLayers::All
    }
}

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        router_input: RouterInput::RawLayerInput,
        ..ArchFacts::DEFAULT.moe
    },
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 4,
            dense_first: true,
        }),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        pattern_key_read: PatternKeyRead::ArrayThenScalar,
        ..ArchFacts::DEFAULT.swa
    },
    ffn: FfnFacts {
        reglu: true,
        ..ArchFacts::DEFAULT.ffn
    },
    rules: HparamRules {
        swa_window_override,
        rope_layers,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["smallthinker"],
    source: "src/models/smallthinker.cpp",
    facts: FACTS,
    evidence: Some("tests/router_input_graphs.rs: fixtures against libllama's logits"),
};
