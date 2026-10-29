//! Per-architecture definitions (ADR 0003,
//! `docs/plans/arch-registry-migration.md`).
//!
//! [`facts::ArchFacts`] is what the loaders read about an architecture.
//! [`facts::ArchFacts::of`] answers from the [`REGISTRY`] entry that
//! names it, and from the legacy name-keyed tables
//! ([`facts::ArchFacts::from_tables`]) for every architecture that has
//! not moved yet. One file per llama.cpp `src/models/<arch>.cpp`.

pub mod baichuan;
pub mod bloom;
pub mod cohere2;
pub mod cohere2moe;
pub mod command_r;
pub mod dbrx;
pub mod dots1;
pub mod exaone4;
pub mod facts;
pub mod falcon;
pub mod gemma2;
pub mod gemma3;
pub mod gpt2;
pub mod gpt_oss;
pub mod gptneox;
pub mod granite;
pub mod granite_moe;
pub mod granite_swa;
pub mod grok;
pub mod jais;
pub mod kolibri1;
pub mod llama;
pub mod llama4;
pub mod maple;
pub mod mellum;
pub mod mpt;
pub mod olmo;
pub mod olmo2;
pub mod olmoe;
pub mod phi2;
pub mod phimoe;
pub mod plamo;
pub mod plamo3;
pub mod qwen;
pub mod qwen2;
pub mod qwen2moe;
pub mod qwen3;
pub mod qwen3moe;
pub mod refact;
mod refused_keys;
pub mod spark2_5;
pub mod stablelm;
pub mod starcoder;

#[cfg(test)]
mod tests;

pub use facts::{ArchFacts, HparamRules};

/// One architecture as llama.cpp defines it.
#[derive(Debug, Clone, Copy)]
pub struct ModelDef {
    /// Every `general.architecture` string this definition answers for.
    /// The first is the llama.cpp spelling; any other is a frink alias
    /// that runs the same graph with the same rows.
    pub names: &'static [&'static str],
    /// The llama.cpp file the facts were read from.
    pub source: &'static str,
    /// Every table row, stated here instead of in the legacy tables.
    /// `profile` and `audited` stay `None` / `false`: the catalog and
    /// `capability::AUDITED_GENERIC_GQA` still own them (plan steps 10
    /// and 11), and [`ArchFacts::of`] fills them in.
    pub facts: ArchFacts,
    /// Why the generic path is trusted for these names, or `None` for
    /// an unaudited row. A test holds it to `AUDITED_GENERIC_GQA`.
    pub evidence: Option<&'static str>,
}

/// Every architecture that has moved out of the legacy tables.
pub const REGISTRY: &[ModelDef] = &[
    llama::DEF,
    granite::DEF,
    granite_moe::DEF,
    olmo::DEF,
    olmo2::DEF,
    exaone4::DEF,
    qwen::DEF,
    qwen2::DEF,
    qwen2moe::DEF,
    qwen3::DEF,
    qwen3moe::DEF,
    kolibri1::DEF,
    olmoe::DEF,
    dots1::DEF,
    phimoe::DEF,
    dbrx::DEF,
    grok::DEF,
    gpt_oss::DEF,
    llama4::DEF,
    stablelm::DEF,
    gptneox::DEF,
    phi2::DEF,
    falcon::DEF,
    plamo::DEF,
    command_r::DEF,
    cohere2::DEF,
    cohere2moe::DEF,
    bloom::DEF,
    refact::DEF,
    baichuan::DEF,
    mpt::DEF,
    jais::DEF,
    gpt2::DEF,
    starcoder::DEF,
    gemma2::DEF,
    gemma3::DEF,
    mellum::DEF,
    plamo3::DEF,
    maple::DEF,
    spark2_5::DEF,
    granite_swa::DEF,
];

/// The registry entry naming `arch`.
pub fn find(arch: &str) -> Option<&'static ModelDef> {
    REGISTRY.iter().find(|def| def.names.contains(&arch))
}
