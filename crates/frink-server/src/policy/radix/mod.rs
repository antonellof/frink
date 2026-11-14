//! Radix prefix caches: what part of a new prompt is already computed.
//!
//! One tree ([`tree`]), one cache over it ([`RadixCache`], full KV
//! pages with LRU eviction of unlocked leaves), and one namespace map
//! over those ([`SaltedRadix`], a cache per caller salt sharing one page
//! pool). The sliding-window and recurrent-state variants FreeToken
//! also ships were ported and then deleted with the page ledger that
//! was their only caller; `docs/THIRD_PARTY_NOTICES.md` records them.
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
pub use tree::{align_down, NodeId};
