//! Serving telemetry: what the server may honestly claim about what it
//! has done.
//!
//! This used to be two module trees that never disagreed about their
//! job, only about where they came from: `stats.rs` held the
//! `/admin/stats` ring, and `policy::serving_stats` held the primitives
//! that ring was already built on plus the `/v1/stats` rate counters.
//! One was written here, the other ported; nothing else separated them.
//! They are one module now, split by what each piece answers rather
//! than by provenance:
//!
//! | module | answers |
//! |---|---|
//! | [`ring`] | which finished requests a poller has not seen yet, and how many it missed |
//! | [`rate`] | what the recent throughput is |
//! | [`serving`] | what is in flight right now, and what completed or aborted |
//! | [`requests`] | what `/admin/stats` and `/v1/requests` report about finished requests |
//!
//! Everything here is arithmetic, not measurement. Every clock reading
//! is a parameter: nothing below reads the wall clock, so every rule is
//! testable without waiting for one.
//!
//! [`ring`] and [`rate`] are ported from FreeToken's
//! `server/request_ring.py` and `server/stats.py` (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`. One deliberate departure, in both
//! directions from the same rule: where upstream returns `0` for "no
//! request had one", this returns `None`. Zero is a measurement;
//! absence is not.

/// What the recent throughput is, behind the two `/v1/stats` rates.
pub(crate) mod rate;

/// A Prometheus histogram, behind `/metrics`' latency families.
pub(crate) mod histogram;
pub(crate) mod requests;
pub(crate) mod ring;

/// The live counters and the feed both decode paths write them through.
pub(crate) mod serving;

#[cfg(test)]
mod http_tests;

pub(crate) use requests::{entry, Record, Stats, MAX_PAGE};
pub(crate) use serving::ServingFeed;
