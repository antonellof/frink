//! A Prometheus histogram over fixed bucket bounds, updated without a
//! lock.
//!
//! Each observation lands in exactly one bucket: the first whose upper
//! bound it does not exceed, or the overflow past the last. The
//! exposition is cumulative, as Prometheus wants, and `+Inf` and
//! `_count` are both the sum of every bucket, so a scrape that races an
//! observation can never see them disagree with each other. `_sum` is
//! kept in microseconds and may trail by the observation in flight.

use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) struct Histogram {
    bounds: &'static [f64],
    /// One counter per bound, then the overflow. Not cumulative.
    buckets: Box<[AtomicU64]>,
    sum_micros: AtomicU64,
}

impl Histogram {
    /// `bounds` must be finite and strictly increasing.
    pub(crate) fn new(bounds: &'static [f64]) -> Self {
        debug_assert!(
            bounds.windows(2).all(|w| w[0] < w[1]) && bounds.iter().all(|b| b.is_finite()),
            "histogram bounds must be finite and increasing: {bounds:?}"
        );
        Histogram {
            bounds,
            buckets: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_micros: AtomicU64::new(0),
        }
    }

    /// Records one value in seconds. A negative or non-finite value is
    /// a timing bug upstream and is dropped rather than recorded.
    pub(crate) fn observe(&self, seconds: f64) {
        if !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        let at = self
            .bounds
            .iter()
            .position(|&b| seconds <= b)
            .unwrap_or(self.bounds.len());
        self.buckets[at].fetch_add(1, Ordering::Relaxed);
        self.sum_micros
            .fetch_add((seconds * 1e6).round() as u64, Ordering::Relaxed);
    }

    /// The text exposition, `# HELP` and `# TYPE` included.
    pub(crate) fn render(&self, name: &str, help: &str) -> String {
        let mut out = format!("# HELP {name} {help}\n# TYPE {name} histogram\n");
        let mut cumulative = 0u64;
        for (bound, bucket) in self.bounds.iter().zip(self.buckets.iter()) {
            cumulative += bucket.load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}\n"));
        }
        cumulative += self.buckets[self.bounds.len()].load(Ordering::Relaxed);
        let sum = self.sum_micros.load(Ordering::Relaxed) as f64 / 1e6;
        out.push_str(&format!(
            "{name}_bucket{{le=\"+Inf\"}} {cumulative}\n{name}_sum {sum}\n{name}_count {cumulative}\n"
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value on a bound counts in that bucket, buckets accumulate,
    /// the overflow reaches only `+Inf`, and a bad value is dropped.
    #[test]
    fn buckets_are_cumulative_and_inclusive_of_their_bound() {
        static BOUNDS: [f64; 3] = [0.1, 0.5, 1.0];
        let h = Histogram::new(&BOUNDS);
        for v in [0.05, 0.1, 0.3, 2.0, f64::NAN, -1.0] {
            h.observe(v);
        }
        assert_eq!(
            h.render("t_seconds", "A test."),
            "# HELP t_seconds A test.\n# TYPE t_seconds histogram\n\
             t_seconds_bucket{le=\"0.1\"} 2\n\
             t_seconds_bucket{le=\"0.5\"} 3\n\
             t_seconds_bucket{le=\"1\"} 3\n\
             t_seconds_bucket{le=\"+Inf\"} 4\n\
             t_seconds_sum 2.45\n\
             t_seconds_count 4\n"
        );
    }
}
