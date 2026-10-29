//! `bloom` (BLOOM): `src/models/bloom.cpp` at the pinned commit.
//!
//! - **ALiBi with the literal bias 8** (`:18`), no key read, and no
//!   RoPE ([`crate::alibi::alibi_rope_layers`]).
//! - **A biased LayerNorm on the token embeddings** before layer 0:
//!   `:25-26` create `tok_norm` / `tok_norm_b` REQUIRED and `:77-80`
//!   norm `inpL` with them, the one decoder on the generic path that
//!   does.
//! - **A LayerNorm with weight and bias** at every site, the ungated
//!   GELU FFN, and REQUIRED biases on the fused `attn_qkv`
//!   (`:44-45`), `attn_output` and the dense FFN's up and down.
//!
//! Evidence: `tests/alibi_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, HparamRules, TensorFacts, TopologyFacts};
use super::ModelDef;
use crate::alibi::MaxAlibiBias;
use crate::norm::NormFunction;
use crate::norm_sites::{NormSites, StoredNorm};
use crate::proj_bias::Presence;
use crate::qkv_fused::QkvTensors;
use crate::rope_layers::RopeLayers;

const ALIBI: MaxAlibiBias = MaxAlibiBias::Literal;

/// No rotation where the bias positions the tokens (`crate::alibi`).
pub fn rope_layers(_arch: &str, n_layers: usize, _window: bool, _dense: usize) -> RopeLayers {
    crate::alibi::alibi_rope_layers(ALIBI, n_layers)
}

pub const FACTS: ArchFacts = ArchFacts {
    ffn: FfnFacts {
        gelu_ungated: true,
        ..ArchFacts::DEFAULT.ffn
    },
    topology: TopologyFacts {
        alibi: Some(ALIBI),
        ..ArchFacts::DEFAULT.topology
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNormBias,
            embedding: Some(StoredNorm::required(&["token_embd_norm"])),
            ..NormSites::GENERIC
        },
        attn_out_bias: Some(Presence::Required),
        ffn_bias: Some((Presence::Required, false)),
        qkv_tensors: QkvTensors::Fused(Some(Presence::Required)),
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        rope_layers,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["bloom"],
    source: "src/models/bloom.cpp",
    facts: FACTS,
    evidence: Some("tests/alibi_graphs.rs: a fixture against libllama's logits"),
};
