//! Rolling-window transfer rate and ETA for long-running jobs
//! (downloads, conversions, model loads).
//!
//! Written from a behavioural description, not from any reference
//! implementation -- see `bc5074aa:docs/plans/on-hold/frink-ui.md`.
//!
//! The whole point is what it *refuses* to say. Rate is
//! `bytes_delta / time_delta`, and on the very first tick `time_delta`
//! is a millisecond or two of a buffered write, which divides out to
//! "123 GB/s" and flashes it at the user before settling. So:
//!
//! - a rate is reported only once the window holds at least
//!   [`MIN_SAMPLES`] samples spanning at least [`MIN_SPAN_MS`]; before
//!   that the report is `stable == false` and carries no number at all,
//!   which makes the flash structurally impossible rather than merely
//!   unlikely;
//! - a byte counter that goes *backwards* (a resumed or restarted
//!   transfer) clears the window instead of producing a negative rate;
//! - ETA is clamped at zero, because a total that is smaller than the
//!   bytes already seen is a metadata bug, not a negative remaining
//!   time.
//!
//! Time is passed in as milliseconds rather than read from a clock, so
//! the behaviour is testable without sleeping.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Samples required before a rate is trusted.
pub const MIN_SAMPLES: usize = 3;
/// Milliseconds the window must span before a rate is trusted.
pub const MIN_SPAN_MS: u64 = 3_000;
/// Samples older than this are dropped, so a rate reflects the recent
/// past rather than the average since the job started.
pub const WINDOW_MS: u64 = 30_000;
/// Hard cap on retained samples, for a caller that observes at a high
/// rate. Overflow drops from the *middle* of the window, never the
/// oldest sample -- dropping the front would shrink the measured span
/// and could keep a fast-ticking job permanently "warming up".
const MAX_SAMPLES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sample {
    at_ms: u64,
    bytes: u64,
}

/// What the UI may display. `bytes_per_second` and `eta_seconds` are
/// `Some` only when `stable` is true.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RateReport {
    /// True once the window is long enough to divide with confidence.
    pub stable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_per_second: Option<f64>,
    /// Remaining seconds, when a total is known and the rate is stable
    /// and positive. Never negative.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<f64>,
    /// Samples currently in the window, so a UI can show "measuring…"
    /// with some idea of progress toward a first number.
    pub samples: usize,
}

impl RateReport {
    fn warming(samples: usize) -> Self {
        RateReport {
            stable: false,
            bytes_per_second: None,
            eta_seconds: None,
            samples,
        }
    }
}

/// Rolling window of `(timestamp, cumulative bytes)` observations.
#[derive(Debug, Default, Clone)]
pub struct RateEstimator {
    window: VecDeque<Sample>,
}

impl RateEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a cumulative byte count seen at `at_ms` (any monotonic
    /// millisecond clock; only differences matter).
    ///
    /// Two inputs are rejected rather than trusted: a sample older than
    /// the newest one (a non-monotonic clock would otherwise produce a
    /// negative span), and a byte count below the newest one, which
    /// means the transfer restarted -- averaging across that
    /// discontinuity would report a rate that never happened, so the
    /// window is cleared and measurement starts over.
    pub fn observe(&mut self, at_ms: u64, bytes: u64) {
        if let Some(last) = self.window.back() {
            if at_ms < last.at_ms {
                return;
            }
            if bytes < last.bytes {
                self.window.clear();
            }
        }
        self.window.push_back(Sample { at_ms, bytes });

        let cutoff = at_ms.saturating_sub(WINDOW_MS);
        while self.window.len() > 1 && self.window.front().is_some_and(|s| s.at_ms < cutoff) {
            self.window.pop_front();
        }
        while self.window.len() > MAX_SAMPLES {
            self.window.remove(1);
        }
    }

    /// Clears the window; the next report is `warming` again. For a job
    /// that pauses, where the elapsed idle time would otherwise be
    /// charged against the rate.
    pub fn reset(&mut self) {
        self.window.clear();
    }

    /// Latest observed cumulative byte count, if any.
    pub fn bytes_done(&self) -> Option<u64> {
        self.window.back().map(|s| s.bytes)
    }

    /// `total_bytes` is optional because plenty of real downloads have
    /// no `Content-Length`; without it there is a rate but no ETA, and
    /// the UI should show exactly that rather than a fabricated one.
    pub fn report(&self, total_bytes: Option<u64>) -> RateReport {
        let (Some(first), Some(last)) = (self.window.front(), self.window.back()) else {
            return RateReport::warming(0);
        };
        let span_ms = last.at_ms - first.at_ms;
        if self.window.len() < MIN_SAMPLES || span_ms < MIN_SPAN_MS {
            return RateReport::warming(self.window.len());
        }

        let bytes = last.bytes.saturating_sub(first.bytes) as f64;
        let rate = bytes / (span_ms as f64 / 1000.0);
        let eta = total_bytes.filter(|_| rate > 0.0).map(|total| {
            // saturating_sub is the clamp: a total below the bytes
            // already transferred means bad metadata, and "0s left" is
            // the only honest reading of it.
            total.saturating_sub(last.bytes) as f64 / rate
        });
        RateReport {
            stable: true,
            bytes_per_second: Some(rate),
            eta_seconds: eta,
            samples: self.window.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_tick_reports_no_rate_at_all() {
        let mut est = RateEstimator::new();
        est.observe(0, 0);
        est.observe(2, 8 * 1024 * 1024); // 8 MiB in 2ms == "4 GB/s"
        let report = est.report(Some(1 << 30));
        assert!(!report.stable);
        assert_eq!(report.bytes_per_second, None);
        assert_eq!(report.eta_seconds, None);
    }

    #[test]
    fn three_samples_are_not_enough_without_three_seconds() {
        let mut est = RateEstimator::new();
        for i in 0..5 {
            est.observe(i * 100, i * 1_000_000);
        }
        assert!(!est.report(None).stable);
    }

    #[test]
    fn three_seconds_are_not_enough_without_three_samples() {
        let mut est = RateEstimator::new();
        est.observe(0, 0);
        est.observe(5_000, 5_000_000);
        assert!(!est.report(None).stable);
    }

    #[test]
    fn reports_a_stable_rate_and_eta_once_the_window_qualifies() {
        let mut est = RateEstimator::new();
        // 1 MB/s for four seconds.
        for i in 0..=4u64 {
            est.observe(i * 1000, i * 1_000_000);
        }
        let report = est.report(Some(10_000_000));
        assert!(report.stable);
        assert_eq!(report.bytes_per_second, Some(1_000_000.0));
        // 6 MB left at 1 MB/s.
        assert_eq!(report.eta_seconds, Some(6.0));
    }

    #[test]
    fn a_restarted_transfer_clears_the_window_instead_of_going_negative() {
        let mut est = RateEstimator::new();
        for i in 0..=4u64 {
            est.observe(i * 1000, i * 1_000_000);
        }
        assert!(est.report(None).stable);
        est.observe(5_000, 0); // resumed from scratch
        let report = est.report(None);
        assert!(!report.stable);
        assert_eq!(report.samples, 1);
        assert_eq!(est.bytes_done(), Some(0));
    }

    #[test]
    fn eta_is_clamped_at_zero_when_the_total_is_wrong() {
        let mut est = RateEstimator::new();
        for i in 0..=4u64 {
            est.observe(i * 1000, i * 1_000_000);
        }
        // Server advertised 1 MB but sent 4 MB.
        assert_eq!(est.report(Some(1_000_000)).eta_seconds, Some(0.0));
    }

    #[test]
    fn no_total_means_a_rate_but_no_eta() {
        let mut est = RateEstimator::new();
        for i in 0..=4u64 {
            est.observe(i * 1000, i * 1_000_000);
        }
        let report = est.report(None);
        assert!(report.stable);
        assert!(report.bytes_per_second.is_some());
        assert_eq!(report.eta_seconds, None);
    }

    #[test]
    fn a_stalled_transfer_reports_zero_rather_than_an_eta() {
        let mut est = RateEstimator::new();
        for i in 0..=4u64 {
            est.observe(i * 1000, 1_000_000);
        }
        let report = est.report(Some(2_000_000));
        assert_eq!(report.bytes_per_second, Some(0.0));
        // Dividing by a zero rate is an infinite ETA; report none.
        assert_eq!(report.eta_seconds, None);
    }

    #[test]
    fn samples_older_than_the_window_are_dropped() {
        let mut est = RateEstimator::new();
        est.observe(0, 0);
        for i in 0..=4u64 {
            est.observe(WINDOW_MS + i * 1000, 1_000_000 + i * 1_000_000);
        }
        // The ancient first sample must not drag the average down.
        assert_eq!(est.report(None).bytes_per_second, Some(1_000_000.0));
    }

    #[test]
    fn a_fast_ticking_job_still_becomes_stable() {
        // 100 Hz for ten seconds: far more samples than MAX_SAMPLES, so
        // this only works if overflow drops from the middle.
        let mut est = RateEstimator::new();
        for i in 0..=1000u64 {
            est.observe(i * 10, i * 10_000);
        }
        let report = est.report(None);
        assert!(report.stable, "{report:?}");
        assert_eq!(report.bytes_per_second, Some(1_000_000.0));
    }

    #[test]
    fn a_backwards_clock_sample_is_ignored() {
        let mut est = RateEstimator::new();
        for i in 0..=4u64 {
            est.observe(i * 1000, i * 1_000_000);
        }
        est.observe(500, 9_000_000);
        assert_eq!(est.bytes_done(), Some(4_000_000));
        assert_eq!(est.report(None).bytes_per_second, Some(1_000_000.0));
    }

    #[test]
    fn reset_returns_to_warming() {
        let mut est = RateEstimator::new();
        for i in 0..=4u64 {
            est.observe(i * 1000, i * 1_000_000);
        }
        est.reset();
        let report = est.report(None);
        assert!(!report.stable);
        assert_eq!(report.samples, 0);
    }

    #[test]
    fn warming_reports_omit_the_absent_numbers_rather_than_nulling_them() {
        let json = serde_json::to_string(&RateReport::warming(1)).unwrap();
        assert_eq!(json, "{\"stable\":false,\"samples\":1}");
    }
}
