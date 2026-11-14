//! The host-side *policy* of serving: the parts that decide rather than
//! compute.
//!
//! A Rust port of the serving half of
//! [FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0;
//! see `docs/THIRD_PARTY_NOTICES.md`). FreeToken's thesis, from
//! *Efficient Edge-Native MoE Serving with Bandwidth-Adaptive
//! Execution* (arXiv:2608.16157), is that a personal machine is not a
//! small datacenter GPU but a heterogeneous, elastic pool of resources
//! -- VRAM, host RAM, PCIe, CPU cores -- whose balance differs per
//! machine, and that the serving stack should treat the *link between
//! them* as a scheduling signal rather than a fixed bottleneck.
//!
//! Everything here follows from that, and everything here is
//! deliberately tensor-free: each module takes measured numbers
//! (bandwidths, byte costs, pool sizes, token counts) and returns a
//! decision, so every policy below is testable on any host, with no GPU
//! and no model. That property is why these modules carry the test
//! density they do, and it is worth preserving as they are wired.
//!
//! | module | decides |
//! |---|---|
//! | [`radix`] | which prefix of a new prompt is already computed, and what may be evicted |
//! | [`anchor`] | which position an agentic turn will come back to, and what to keep for it |
//! | [`pool_budget`] | how VRAM is split between the expert cache and the KV pools, and how it is re-split live |
//! | [`parser`] | where a model's reasoning ends and its answer begins, and which tool it called |
//! | [`effort`] | which reasoning-effort dialect this checkpoint speaks |
//! | [`detokenize`] | what text is safe to stream after one more token |
//! | [`maintenance`] | whether a request, a rebuild, or a stop may proceed right now |
//! | [`rebuild`] | how a live pool re-split commits, rolls back, or latches |
//! | [`outbox`] | what a stop reports, once, however many times it is retried |
//! | [`footprint`] | what this process is actually costing in RSS or PSS |
//!
//! Where frink already had a mechanism the port would have duplicated
//! -- content-addressed KV blocks ([`frink_core::kv_block`]),
//! continuous batching ([`crate::serving::batch`]) -- these modules
//! plug into it instead of shadowing it. Each module's docs say which.
//!
//! The MoE expert-residency half of the same port lives in
//! `frink-core` beside `expert_store`: `expert_cache`, `expert_slots`,
//! `residency`, `placement`, `qstar` and `bench_profile`. It is there
//! and not here because it governs *device memory*, and because two
//! expert byte budgets that do not know about each other are, on
//! unified memory, the same RAM counted twice.

// WHY THE `allow(dead_code)` BELOW, AND WHAT WOULD REMOVE EACH ONE.
//
// Every module here has live callers. Some also carry the UNWIRED
// HALF of an issue that is still open, and the issue is named on each.
// That is the only reason an allow is acceptable in this crate:
// `grep -n "allow(dead_code)" policy/mod.rs` is meant to be the list of
// what still owes a caller. An item that cannot name an open issue
// does not belong under one -- delete it instead.

/// Wired: `decode_slide`, `AnchorState`, `SlidingRequest`,
/// `WindowPolicy`, `resolve_anchor_token`.
/// Unwired: `prefill_slide` and `SlideDecision::released`, the prompt
/// half of the same rule. Closes with #61's prompt-region eviction.
#[allow(dead_code)]
pub(crate) mod anchor;

pub(crate) mod detokenize;

/// Wired: effort probing and the thinking-mode resolution behind
/// `chat_template`. Unwired: `Effort::kwargs_for`, the round-trip a
/// user-supplied effort override would go through. Closes with #404.
#[allow(dead_code)]
pub(crate) mod effort;

// Not dead: `parse_smaps_rollup_pss` and `parse_status_rss` are called
// from `cache_admin`'s `#[cfg(target_os = "linux")]` probe. Everywhere
// else the caller is compiled out and the lint cannot see it, so the
// allow is conditional: on Linux these still have to earn their keep,
// which is exactly how `sum_footprints` and `ProbeCache::last` were
// caught after the supervisor that used them was deleted.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) mod footprint;

pub(crate) mod maintenance;

pub(crate) mod outbox;

/// Wired: both parsers, their formats and `infer`.
/// Unwired: `ReasoningFormat`/`ToolCallFormat` `parse`, the name
/// round-trip an explicit `--reasoning-format` override needs, the
/// `format()` accessors and `ReasoningDelta::is_empty`. Closes with
/// #404.
#[allow(dead_code)]
pub(crate) mod parser;

pub(crate) mod pool_budget;
pub(crate) mod radix;
pub(crate) mod rebuild;

/// The differential turn-boundary analyzer, beside `effort` because it
/// is the same mechanism: learn a template's behaviour by rendering it
/// rather than by matching its source.
pub(crate) mod turn;
