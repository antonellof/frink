//! Radix prefix caches: what part of a new prompt is already computed.
//!
//! Three caches, one tree. They differ in how many *currencies* a node
//! holds, and that difference reaches all the way into eviction:
//!
//! | cache | currencies | an eviction frees |
//! |---|---|---|
//! | [`RadixCache`] | full KV pages | pages |
//! | [`SwaRadixCache`] | full KV pages + sliding-window KV | either, independently |
//! | [`HybridRadixCache`] | full KV pages + a recurrent-state snapshot | either, independently |
//!
//! The two-currency caches are separate types rather than one
//! parameterized cache because *what may be evicted* differs at every
//! step: a window pass may tombstone an interior node in place, which
//! the plain cache has no concept of, and a recurrent snapshot lives at
//! a node's end boundary rather than across its span.
//!
//! `frink`'s existing `frink-models::prefix_cache` is a flat list of
//! whole-conversation snapshots with a linear longest-prefix scan;
//! these share *nodes* between prompts, so a thousand requests off one
//! system prompt hold one copy of it and the KV pages under it are
//! reference-counted rather than cloned.

pub mod plain;
pub mod salted;
pub mod tree;

pub use plain::RadixCache;
pub use salted::{Handle, SaltedRadix};
pub use tree::{align_ceil, align_down, NodeId};
