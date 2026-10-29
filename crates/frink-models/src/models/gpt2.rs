//! `gpt2` (GPT-2): `src/models/gpt2.cpp` at the pinned commit.
//!
//! - **A learned position table instead of RoPE**: `position_embd`
//!   REQUIRED (`:19,74-77`), added to the embeddings, and no
//!   `ggml_rope` call ([`crate::rope_layers::never`]).
//! - **A LayerNorm with weight and bias** at every site, the ungated
//!   GELU FFN, and REQUIRED biases on the fused `attn_qkv` (`:37-38`),
//!   `attn_output` and the dense FFN's up and down.
//!
//! Evidence: `tests/position_embd_graphs.rs`, against libllama.

use super::facts::{ArchFacts, FfnFacts, HparamRules, TensorFacts};
use super::ModelDef;
use crate::norm::NormFunction;
use crate::norm_sites::NormSites;
use crate::proj_bias::Presence;
use crate::qkv_fused::QkvTensors;

pub const FACTS: ArchFacts = ArchFacts {
    ffn: FfnFacts {
        gelu_ungated: true,
        ..ArchFacts::DEFAULT.ffn
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            function: NormFunction::LayerNormBias,
            ..NormSites::GENERIC
        },
        attn_out_bias: Some(Presence::Required),
        ffn_bias: Some((Presence::Required, false)),
        qkv_tensors: QkvTensors::Fused(Some(Presence::Required)),
        position_embd: Some((Presence::Required, "src/models/gpt2.cpp:19,74-77")),
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        rope_layers: crate::rope_layers::never,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["gpt2"],
    source: "src/models/gpt2.cpp",
    facts: FACTS,
    evidence: Some("tests/position_embd_graphs.rs: a fixture against libllama's logits"),
};
