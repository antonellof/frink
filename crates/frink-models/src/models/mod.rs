//! Per-architecture definitions (ADR 0003,
//! `docs/plans/arch-registry-migration.md`).
//!
//! Step 1 of the migration: [`facts::ArchFacts`] is what the loaders
//! read about an architecture, built today from the legacy name-keyed
//! tables by [`facts::ArchFacts::from_tables`]. Later steps add one
//! `models/<arch>.rs` per family whose facts are written there instead.

pub mod facts;
mod refused_keys;

pub use facts::{ArchFacts, HparamRules};
