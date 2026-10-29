//! `exaone4` (EXAONE-4.0): `src/models/exaone4.cpp` at the pinned commit.
//!
//! - **Post-norm only**, the same topology as `olmo2`
//!   ([`NormSites::POST_NORM_ONLY`]).
//! - **SWA by layer count.** `:4-11` switch STANDARD SWA on only when
//!   `n_layer() == 64` (the 32B), seed `n_swa = 4096` and a period-4
//!   pattern, and seed the sliding layers' RoPE base AND scale from the
//!   model's; `:14` then reads `attention.sliding_window` as OPTIONAL,
//!   so a 64-layer file without the key slides at 4096 (#440). The test
//!   is equality, so a 63- or 65-layer file gets no window. `n_layer()`
//!   is the TRUNK: the common loader reads `nextn_predict_layers` before
//!   `load_arch_hparams` (`llama-model.cpp:1411,1571`), so a 64-layer
//!   trunk with an MTP block appended is the 32B graph (#441), and the
//!   loader hands [`swa_window_override`] the trunk.
//! - **RoPE on the sliding layers only** (`:111`, `use_rope =
//!   is_swa(il) || swa_type == NONE`), so the 32B's full-attention
//!   layers rotate nothing and the 1.2B rotates everything
//!   ([`rope_layers`]).
//! - **NextN blocks** are tensors of the trailing `nextn` layers,
//!   created with `TENSOR_SKIP` (`:38-68`) and subtracted from the
//!   layer count.
//!
//! Evidence: `tests/post_norm_only_graphs.rs` (1.2B shape) and
//! `tests/no_rope_layer_graphs.rs` (the 64-layer shape), against
//! libllama.

use super::facts::{ArchFacts, HparamRules, SwaFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::capability::{SwaPattern, SwaWindowOverride};
use crate::norm_sites::NormSites;
use crate::rope_layers::RopeLayers;

/// `exaone4.cpp:4-14`: on the 64-layer trunk, 4096 unless the file
/// declares a window; below or above it, no window whatever the file
/// says.
pub fn swa_window_override(_arch: &str, n_layers: usize) -> SwaWindowOverride {
    if n_layers == 64 {
        SwaWindowOverride::Seed(SEEDED_WINDOW)
    } else {
        SwaWindowOverride::Drop
    }
}

/// `exaone4.cpp:6`.
pub const SEEDED_WINDOW: usize = 4096;

/// `exaone4.cpp:111`.
pub fn rope_layers(_arch: &str, _n_layers: usize, has_window: bool, _dense: usize) -> RopeLayers {
    crate::rope_layers::sliding_layers_rotate(has_window)
}

pub const FACTS: ArchFacts = ArchFacts {
    swa: SwaFacts {
        default_layout: Some(SwaPattern {
            period: 4,
            dense_first: false,
        }),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        ..ArchFacts::DEFAULT.swa
    },
    topology: TopologyFacts {
        reads_nextn: true,
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites::POST_NORM_ONLY,
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        swa_window_override,
        rope_layers,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["exaone4"],
    source: "src/models/exaone4.cpp",
    facts: FACTS,
    evidence: Some(
        "tests/post_norm_only_graphs.rs and tests/no_rope_layer_graphs.rs: the 1.2B and the \
         64-layer shapes against libllama's logits, the latter also without its window key \
         and with an MTP block appended",
    ),
};
