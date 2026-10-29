//! Per-architecture definitions (ADR 0003,
//! `docs/plans/arch-registry-migration.md`).
//!
//! [`facts::ArchFacts`] is what the loaders read about an architecture.
//! [`facts::ArchFacts::of`] answers from the [`REGISTRY`] entry that
//! names it, and from the legacy name-keyed tables
//! ([`facts::ArchFacts::from_tables`]) for every architecture that has
//! not moved yet. One file per llama.cpp `src/models/<arch>.cpp`.

pub mod exaone4;
pub mod facts;
pub mod granite;
pub mod granite_moe;
pub mod llama;
pub mod olmo;
pub mod olmo2;
pub mod qwen;
pub mod qwen2;
pub mod qwen2moe;
pub mod qwen3;
pub mod qwen3moe;
mod refused_keys;

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
];

/// The registry entry naming `arch`.
pub fn find(arch: &str) -> Option<&'static ModelDef> {
    REGISTRY.iter().find(|def| def.names.contains(&arch))
}
