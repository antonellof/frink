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

// `rate` and `serving` carry one unwired half between them, and it is a
// live defect rather than spare capacity: nothing on any request path
// calls `ServingStats::on_prefill`, `on_decode`, `on_admitted`,
// `on_abort` or `on_finished`, so `RateWindow::record` has no caller,
// the counter block is never incremented, and the two `/v1/stats`
// throughput rates read 0.0 however busy the server is (`docs/API.md`
// says so). Both allows go when the two decode paths report tokens as
// they are produced, or when the rates and the counters are deleted.

/// Wired: `RateWindow::new` and `tokens_per_second`, behind the two
/// `/v1/stats` rates. Unwired: `record`; see above.
#[allow(dead_code)]
pub(crate) mod rate;

/// A Prometheus histogram, behind `/metrics`' latency families.
pub(crate) mod histogram;
pub(crate) mod requests;
pub(crate) mod ring;

/// Wired: `ServingStats::default` and the two rate reads `/v1/stats`
/// makes. Unwired: every feeder and every counter; see above.
#[allow(dead_code)]
pub(crate) mod serving;

pub(crate) use requests::{entry, Record, Stats, MAX_PAGE};
pub(crate) use serving::ServingStats;
