//! `exaone4` (EXAONE-4.0): `src/models/exaone4.cpp` at the pinned commit.
//!
//! - **Post-norm only**, the same topology as `olmo2`
//!   ([`NormSites::POST_NORM_ONLY`]).
//! - **SWA by layer count.** `:4-12` switch STANDARD SWA on only when
//!   `n_layer() == 64` (the 32B), with a period-4 seed, and seed the
//!   sliding layers' RoPE base AND scale from the model's. The test is
//!   equality, so a 63- or 65-layer file gets no window either; and it
//!   runs before the NextN blocks are subtracted, which is why the
//!   loader hands it `block_count` ([`swa_window_override`]).
//! - **RoPE on the sliding layers only** (`:111`, `use_rope =
//!   is_swa(il) || swa_type == NONE`), so the 32B's full-attention
//!   layers rotate nothing and the 1.2B rotates everything
//!   ([`rope_layers`]).
//! - **NextN blocks** are tensors of the trailing `nextn` layers
//!   (`:39-68`), subtracted from the layer count.
//!
//! Two divergences from the pin are carried over unchanged and filed:
//! `:6` pins `n_swa = 4096` before the optional key, so a 64-layer file
//! WITHOUT the key slides upstream and not here (#440); and the NextN
//! count is now read for every architecture (`llama-model.cpp:1261`,
//! #441).
//!
//! Evidence: `tests/post_norm_only_graphs.rs` (1.2B shape) and
//! `tests/no_rope_layer_graphs.rs` (the 64-layer shape), against
//! libllama.

use super::facts::{ArchFacts, HparamRules, SwaFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::capability::{SwaPattern, SwaWindowOverride};
use crate::norm_sites::NormSites;
use crate::rope_layers::RopeLayers;

/// `exaone4.cpp:4`: the file's window only on the 64-layer model.
pub fn swa_window_override(_arch: &str, block_count: usize) -> SwaWindowOverride {
    if block_count == 64 {
        SwaWindowOverride::Honour
    } else {
        SwaWindowOverride::Drop
    }
}

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
         64-layer shapes against libllama's logits",
    ),
};
