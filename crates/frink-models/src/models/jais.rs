//! `jais` (Jais-13B / 30B): `src/models/jais.cpp` at the pinned commit.
//!
//! - **ALiBi from `attention.max_alibi_bias`** (`:5`, optional), no RoPE.
//! - **The attention scale is `1 / head_dim`**, not `1 / sqrt(head_dim)`
//!   ([`attention_scale_override`]).
//! - **A LayerNorm with weight and bias** at every site, SwiGLU, and
//!   REQUIRED biases on the fused `attn_qkv` (`:31-32`), `attn_output`
//!   and every dense FFN matrix including `ffn_gate`.
//!
//! Evidence: `tests/alibi_graphs.rs`, against libllama.

use super::facts::{ArchFacts, HparamRules, TensorFacts, TopologyFacts};
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

/// `1 / head_dim` (Jais' `scale_qk_dot_by_d`), whatever the shape.
pub fn attention_scale_override(
    _arch: &str,
    _n_layers: usize,
    _hidden_dim: usize,
    _n_heads: usize,
    head_dim: usize,
) -> Option<f32> {
    (head_dim != 0).then(|| 1.0 / head_dim as f32)
}

pub const FACTS: ArchFacts = ArchFacts {
    topology: TopologyFacts {
        alibi: Some(ALIBI),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNormBias,
            ..NormSites::GENERIC
        },
        attn_out_bias: Some(Presence::Required),
        ffn_bias: Some((Presence::Required, true)),
        qkv_tensors: QkvTensors::Fused(Some(Presence::Required)),
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        rope_layers,
        attention_scale_override,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["jais"],
    source: "src/models/jais.cpp",
    facts: FACTS,
    evidence: Some("tests/alibi_graphs.rs: a fixture against libllama's logits"),
};
