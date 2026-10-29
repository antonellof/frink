//! `granitemoe`: `src/models/granite-moe.cpp` at the pinned commit.
//!
//! No graph of its own: `models.h:1673-1678` is `using graph =
//! llama_model_granite::graph`, so it runs [`super::granite`]'s graph
//! and takes its `ffn_gate_inp != nullptr` branch. Its hparam loader
//! reads the same four multipliers (`granite-moe.cpp:5-8`) and its
//! tensor loader creates the same optional biases (`:44,61-63`), so the
//! facts are granite's.
//!
//! One divergence is carried over unchanged from the legacy table and
//! is NOT upstream's: `granite-moe.cpp`'s `load_arch_hparams` does not
//! read `rope.scaling.finetuned` (granite's `:33-37` has no counterpart
//! here), so llama.cpp keeps `rope_pattern` at its all-ones seed
//! (`llama-model.cpp:1314`) and rotates a `granitemoe` file that
//! declares `false`, where frink does not. No converter writes the key
//! for this architecture (`conversion/granite.py:181-231`, the
//! `GraniteMoeModel`; only `GraniteHybridModel` writes it, `:519`), so
//! no real file reaches it. This move is a refactor; the fix needs
//! its own fixture and golden.
//!
//! `granite-moe` is a frink-only alias (`llama-arch.cpp:104` spells it
//! `granitemoe`), evidenced by a byte-identical fixture asserted against
//! `granitemoe`'s libllama golden.

use super::facts::ArchFacts;
use super::ModelDef;

pub const FACTS: ArchFacts = super::granite::FACTS;

pub const DEF: ModelDef = ModelDef {
    names: &["granitemoe", "granite-moe"],
    source: "src/models/granite-moe.cpp",
    facts: FACTS,
    evidence: Some(
        "tests/granite_family_graphs.rs: a MoE fixture against libllama's logits, and the \
         `granite-moe` alias fixture against the same golden",
    ),
};
