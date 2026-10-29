//! `mpt` (MPT-7B / 30B): `src/models/mpt.cpp` at the pinned commit.
//!
//! - **ALiBi from `attention.max_alibi_bias`** (`:6`, optional, default
//!   0): a file whose key is absent or zero has no position encoding at
//!   all, upstream and here. No RoPE either way.
//! - **`attention.clamp_kqv`** read optional (`:5`, `crate::clamp_kqv`).
//! - **An optional learned position table** (`:19,80-84`), added when
//!   present (`crate::position_embd`).
//! - **A weighted LayerNorm** (biases optional; MPT has none), the
//!   ungated GELU FFN, and OPTIONAL biases on the fused `attn_qkv`
//!   (`:36-37`), `attn_output` and the dense FFN's up and down.
//! - **A LayerNorm over the whole Q/K projection** (`:52-56,106-115`,
//!   optional), refused by name when present (`crate::qk_layer_norm`).
//!
//! Evidence: `tests/alibi_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, HparamRules, ScalingFacts, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::alibi::MaxAlibiBias;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;
use crate::proj_bias::Presence;
use crate::qkv_fused::QkvTensors;
use crate::rope_layers::RopeLayers;

const ALIBI: MaxAlibiBias = MaxAlibiBias::Key;

/// No rotation where the bias positions the tokens (`crate::alibi`).
pub fn rope_layers(_arch: &str, n_layers: usize, _window: bool, _dense: usize) -> RopeLayers {
    crate::alibi::alibi_rope_layers(ALIBI, n_layers)
}

pub const FACTS: ArchFacts = ArchFacts {
    ffn: FfnFacts {
        gelu_ungated: true,
        ..ArchFacts::DEFAULT.ffn
    },
    scaling: ScalingFacts {
        clamp_kqv: Some(false),
        ..ArchFacts::DEFAULT.scaling
    },
    topology: TopologyFacts {
        alibi: Some(ALIBI),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNorm,
            ..NormSites::GENERIC
        },
        per_head_layer_norm_qk: Some("src/models/mpt.cpp:52-56,106-115"),
        attn_out_bias: Some(Presence::Optional),
        ffn_bias: Some((Presence::Optional, false)),
        qkv_tensors: QkvTensors::Fused(Some(Presence::Optional)),
        position_embd: Some((Presence::Optional, "src/models/mpt.cpp:19,80-84")),
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        rope_layers,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["mpt"],
    source: "src/models/mpt.cpp",
    facts: FACTS,
    evidence: Some("tests/alibi_graphs.rs: fixtures against libllama's logits"),
};
