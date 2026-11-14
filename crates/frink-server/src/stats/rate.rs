//! Rates over a sliding window.
//!
//! [`RateWindow`] divides by the observed span, not by the window. A
//! server running at 50 tok/s, polled one second into a five-second
//! window, has produced 50 tokens; dividing by the window reports
//! 10 tok/s and the number is simply wrong. Dividing by
//! `now - oldest_retained_sample` reports 50, and still decays to zero
//! as the samples age out -- which is the property that matters, because
//! a *cumulative* average would have an idle server reporting the rate
//! it managed an hour ago forever.

use std::collections::VecDeque;

/// Samples retained per rate window.
///
/// Eviction is poll-driven -- only [`RateWindow::tokens_per_second`]
/// trims by age -- so a deployment whose clients never read `/v1/stats`
/// would otherwise grow these without bound. Generous against the
/// window's span at any realistic reply rate.
pub const RATE_SAMPLE_CAPACITY: usize = 4096;

/// A sliding-window rate: tokens per second over the recent past.
///
/// Samples older than the window are dropped on every read, so the rate
/// decays to zero by wall clock when nothing is happening. The
/// denominator is the span from the oldest retained sample to `now`, not
/// the window itself -- see the module doc for why the difference is not
/// cosmetic.
#[derive(Debug, Clone)]
pub struct RateWindow {
    window_ms: u64,
    samples: VecDeque<(u64, u64)>,
}

impl RateWindow {
    /// `window_ms` is clamped to at least 1ms so the span floor is never
    /// larger than the window.
    pub fn new(window_ms: u64) -> Self {
        RateWindow {
            window_ms: window_ms.max(1),
            samples: VecDeque::new(),
        }
    }

    /// Records `tokens` produced at `at_ms`, in any order.
    ///
    /// Bounded by [`RATE_SAMPLE_CAPACITY`] independently of the window,
    /// because age-based eviction only happens on a read.
    pub fn record(&mut self, at_ms: u64, tokens: u64) {
        if tokens == 0 {
            return;
        }
        if self.samples.len() == RATE_SAMPLE_CAPACITY {
            self.samples.pop_front();
        }
        // Kept in time order, which eviction and the span both read off
        // the front. Concurrent requests stamp a prefill at its START
        // and record it at its end, so a sample can arrive older than
        // the newest one held; it is almost always near the back.
        let at = self
            .samples
            .iter()
            .rposition(|(t, _)| *t <= at_ms)
            .map_or(0, |i| i + 1);
        self.samples.insert(at, (at_ms, tokens));
    }

    /// Tokens per second over the window ending at `now_ms`.
    ///
    /// `0.0` when nothing is in the window, which is the true throughput
    /// of an idle server.
    pub fn tokens_per_second(&mut self, now_ms: u64) -> f64 {
        self.evict_before(now_ms.saturating_sub(self.window_ms));
        let Some((oldest, _)) = self.samples.front().copied() else {
            return 0.0;
        };
        let tokens: u64 = self.samples.iter().map(|(_, t)| *t).sum();
        // A whole-millisecond clock can report a span of zero for
        // samples that really were separated in time; one millisecond is
        // the smallest span it can honestly claim to have measured.
        let span_ms = now_ms.saturating_sub(oldest).max(1);
        tokens as f64 * 1000.0 / span_ms as f64
    }

    /// Tokens held, without evicting: what a test checks a path fed.
    #[cfg(test)]
    pub(crate) fn tokens(&self) -> u64 {
        self.samples.iter().map(|(_, t)| *t).sum()
    }

    fn evict_before(&mut self, cutoff: u64) {
        while self.samples.front().is_some_and(|(at, _)| *at < cutoff) {
            self.samples.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rate has to be right *early*, when only part of the window
    /// has elapsed. Dividing by the window instead of the observed span
    /// reports a fifth of the truth one second into a five-second
    /// window, and a UI reads that as a slow model.
    #[test]
    fn the_rate_is_correct_before_the_window_has_filled() {
        let mut window = RateWindow::new(5_000);
        for step in 0..10u64 {
            window.record(step * 100, 5);
        }
        // 50 tokens over the 1000ms actually observed.
        assert_eq!(window.tokens_per_second(1_000), 50.0);
    }

    /// And the reason it is a window at all: an idle server reports 0,
    /// not the rate it managed while it was busy.
    #[test]
    fn an_idle_server_reports_zero_rather_than_its_busiest_minute() {
        let mut window = RateWindow::new(1_000);
        for step in 0..10u64 {
            window.record(step * 100, 10);
        }
        assert!(window.tokens_per_second(900) > 0.0);
        assert_eq!(window.tokens_per_second(3_600_000), 0.0);
    }

    #[test]
    fn a_rate_window_holds_a_bounded_number_of_samples_even_unpolled() {
        let mut window = RateWindow::new(1_000);
        for step in 0..(RATE_SAMPLE_CAPACITY as u64 + 500) {
            window.record(step, 1);
        }
        assert_eq!(window.samples.len(), RATE_SAMPLE_CAPACITY);
    }

    /// A sample older than the newest still counts from its own time:
    /// the span is measured from the oldest sample, wherever it was
    /// recorded, and it ages out when its time does.
    #[test]
    fn a_sample_recorded_out_of_order_is_held_in_time_order() {
        let mut window = RateWindow::new(1_000);
        window.record(500, 10);
        window.record(100, 10);
        assert_eq!(window.tokens_per_second(600), 40.0);
        assert_eq!(window.tokens_per_second(1_200), 10.0 * 1000.0 / 700.0);
    }
}
