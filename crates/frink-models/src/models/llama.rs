//! `llama`: `src/models/llama.cpp` at the pinned commit.
//!
//! The generic decoder IS this graph, so almost every fact is the
//! default. What the file adds is the two OPTIONAL bias groups its
//! tensor loader creates: `attn_output.bias` (`llama.cpp:57`) and the
//! dense FFN's `ffn_{gate,down,up}.bias` (`:75-77`), applied by
//! `build_attn` (`:170`) and `build_ffn` (`:190-192`).
//!
//! `llama-embed` is not here: it has its own catalog row, and
//! `capability::canonical_architecture` folds it to `llama` for the
//! decoder's tensor rows, so it reads this definition there.

use super::facts::{ArchFacts, TensorFacts};
use super::ModelDef;
use crate::proj_bias::Presence;

pub const FACTS: ArchFacts = ArchFacts {
    tensors: TensorFacts {
        attn_out_bias: Some(Presence::Optional),
        ffn_bias: Some((Presence::Optional, true)),
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["llama"],
    source: "src/models/llama.cpp",
    facts: FACTS,
    evidence: Some(
        "benchmarks/suite.json rows measured against llama.cpp on the same host and file \
         (TinyLlama, Mistral, Mixtral, SmolLM2, Llama-3.x all tag llama)",
    ),
};
