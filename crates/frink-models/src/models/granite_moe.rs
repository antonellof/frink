//! `granitemoe`: `src/models/granite-moe.cpp` at the pinned commit.
//!
//! No graph of its own: `models.h:1909` is `using graph =
//! llama_model_granite::graph`, so it runs [`super::granite`]'s graph
//! and takes its `ffn_gate_inp != nullptr` branch. Its hparam loader
//! reads the same four multipliers (`granite-moe.cpp:5-8`) and its
//! tensor loader creates the same optional biases (`:44,61-63`), so the
//! facts are granite's but one.
//!
//! One fact is NOT granite's: `granite-moe.cpp`'s `load_arch_hparams`
//! does not read `rope.scaling.finetuned` (granite's `:33-37` has no
//! counterpart here), so `rope_pattern` keeps its all-ones seed
//! (`llama-model.cpp:1464`) and a `granitemoe` file declaring `false`
//! still rotates every layer. No converter writes the key for this
//! architecture (`conversion/granite.py:181-231`, the `GraniteMoeModel`;
//! only `GraniteHybridModel` writes it, `:519`), so no real file reaches
//! it; `granitemoe_norope_tiny.gguf` does, against libllama (#439).
//!
//! `granite-moe` is a frink-only alias (`llama-arch.cpp:104` spells it
//! `granitemoe`), evidenced by a byte-identical fixture asserted against
//! `granitemoe`'s libllama golden.

use super::facts::{ArchFacts, TopologyFacts};
use super::ModelDef;

pub const FACTS: ArchFacts = ArchFacts {
    topology: TopologyFacts {
        rope_gated_on_finetuned: false,
        ..super::granite::FACTS.topology
    },
    ..super::granite::FACTS
};

pub const DEF: ModelDef = ModelDef {
    names: &["granitemoe", "granite-moe"],
    source: "src/models/granite-moe.cpp",
    facts: FACTS,
    evidence: Some(
        "tests/granite_family_graphs.rs: a MoE fixture against libllama's logits, the \
         `granite-moe` alias fixture against the same golden, and one declaring \
         `rope.scaling.finetuned = false` that still rotates",
    ),
};
