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
//! | [`rate`] | what the recent throughput is, and how to report a value nothing has measured |
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

/// Wired: `RateWindow`, behind the two `/v1/stats` rates.
/// Unwired: `LastKnown`, the "report nothing rather than zero" rule for
/// a pool occupancy that only some replies carry. It has no reporter
/// yet because the pool geometry it would hold is itself groundwork.
/// Closes with #632.
#[allow(dead_code)]
pub(crate) mod rate;

/// A Prometheus histogram, behind `/metrics`' latency families.
pub(crate) mod histogram;
pub(crate) mod requests;

/// Wired: `new`, `push`, `since`, `rows` and `recorded_total`, all from
/// [`requests::Stats`].
/// Unwired: `len`, `is_empty`, `capacity` -- the ring's own size, which
/// `/admin/stats` does not report and its tests reach for directly.
/// These became visible only when this module stopped sharing one
/// crate-wide allow with the counters below; they are three accessors,
/// not a roadmap item, and the honest closure is to delete them if
/// nothing claims them.
#[allow(dead_code)]
pub(crate) mod ring;

/// Wired: `ServingStats` and the two sliding-window rates `/v1/stats`
/// reports. Unwired, and worth stating plainly: the counters
/// (`inflight`, `completed`, `aborted`, `prompt_tokens_total`,
/// `completion_tokens_total`) and `record` have NO caller, so the
/// server holds a counter block it never increments. `/v1/stats`
/// reports the rates only. Closes by wiring `record` into the request
/// path, or by deleting the counters.
#[allow(dead_code)]
pub(crate) mod serving;

pub(crate) use requests::{entry, Record, Stats, MAX_PAGE};
pub(crate) use serving::ServingStats;
