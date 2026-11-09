//! Where a CUDA step's wall time goes on the host, by named phase.
//!
//! The kernel profile (`nsys`) shows the GPU's time and not the host's:
//! a 0.59.0 Llama-3.1-8B `pp512` step took 165 ms of wall time on an
//! L40S against about 110 ms of kernels, and nothing said where the
//! rest went. With `FRINK_CUDA_TIME=1` every prefill step prints its
//! phases' wall times, and decode prints their mean every
//! [`DECODE_WINDOW`] tokens. Off, a [`phase`] reads one cached flag and
//! records nothing.
//!
//! Always compiled, like [`crate::counters`], so its arithmetic is
//! tested on a host with no GPU.

use crate::counters::StepKind;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Decode tokens averaged into one printed line.
pub const DECODE_WINDOW: u32 = 64;

/// `FRINK_CUDA_TIME=1|true|on`. Read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("FRINK_CUDA_TIME").as_deref(),
            Ok("1" | "true" | "on")
        )
    })
}

/// The phases of the step under way, in the order first seen, and the
/// decode window's running sums.
#[derive(Default)]
struct Book {
    step: Vec<(&'static str, Duration)>,
    decode: Vec<(&'static str, Duration)>,
    decode_tokens: u32,
    prefill_steps: u32,
    /// When the last decode token ended: a token's whole wall time is
    /// the gap to the next end, sampling and the caller included.
    last_decode_end: Option<Instant>,
}

impl Book {
    fn add(list: &mut Vec<(&'static str, Duration)>, name: &'static str, d: Duration) {
        match list.iter_mut().find(|e| e.0 == name) {
            Some(e) => e.1 += d,
            None => list.push((name, d)),
        }
    }

    /// The line a step ends with, if it prints one, and the book reset
    /// for the next.
    fn end(&mut self, kind: StepKind) -> Option<String> {
        let step = std::mem::take(&mut self.step);
        match kind {
            StepKind::Prefill => {
                let n = self.prefill_steps;
                self.prefill_steps += 1;
                Some(format!(
                    "frink: cuda prefill step {n} time: {}",
                    fmt(&step, 1)
                ))
            }
            StepKind::Decode => {
                for (name, d) in step {
                    Self::add(&mut self.decode, name, d);
                }
                let now = Instant::now();
                if let Some(prev) = self.last_decode_end.replace(now) {
                    Self::add(&mut self.decode, "token", now - prev);
                }
                self.decode_tokens += 1;
                if self.decode_tokens < DECODE_WINDOW {
                    return None;
                }
                let line = format!(
                    "frink: cuda decode time, mean of {} tokens: {}",
                    self.decode_tokens,
                    fmt(&self.decode, self.decode_tokens)
                );
                self.decode.clear();
                self.decode_tokens = 0;
                Some(line)
            }
        }
    }
}

fn fmt(list: &[(&'static str, Duration)], n: u32) -> String {
    list.iter()
        .map(|(name, d)| {
            format!(
                "{name}={:.2}ms",
                d.as_secs_f64() * 1e3 / f64::from(n.max(1))
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

static BOOK: Mutex<Option<Book>> = Mutex::new(None);

fn record(name: &'static str, d: Duration) {
    let mut g = BOOK.lock().unwrap_or_else(|p| p.into_inner());
    Book::add(&mut g.get_or_insert_with(Book::default).step, name, d);
}

/// Times the scope it lives in as phase `name` of the step under way.
/// Phases may nest; each is its own wall time.
#[must_use = "a phase is timed until it is dropped"]
pub struct Phase {
    name: &'static str,
    start: Option<Instant>,
}

/// A [`Phase`] for `name`, timing only when [`enabled`].
pub fn phase(name: &'static str) -> Phase {
    Phase {
        name,
        start: enabled().then(Instant::now),
    }
}

impl Drop for Phase {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            record(self.name, start.elapsed());
        }
    }
}

/// Ends a step; a no-op unless [`enabled`]. Prints a prefill step's
/// phases, and decode's means once a window fills.
pub fn step_end(kind: StepKind) {
    if !enabled() {
        return;
    }
    let mut g = BOOK.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(line) = g.get_or_insert_with(Book::default).end(kind) {
        eprintln!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// A prefill step prints its phases in the order first seen, a
    /// phase seen twice summed; decode prints nothing until its window
    /// fills and then the per-token mean.
    #[test]
    fn a_step_prints_its_phases_and_decode_prints_a_window_mean() {
        let mut b = Book::default();
        Book::add(&mut b.step, "launch", ms(10));
        Book::add(&mut b.step, "push", ms(3));
        Book::add(&mut b.step, "launch", ms(5));
        let line = b.end(StepKind::Prefill).expect("a prefill step prints");
        assert_eq!(
            line,
            "frink: cuda prefill step 0 time: launch=15.00ms push=3.00ms"
        );
        assert!(b.step.is_empty(), "the next step starts empty");
        for t in 0..DECODE_WINDOW {
            Book::add(&mut b.step, "step", ms(2 + u64::from(t % 2) * 2));
            let printed = b.end(StepKind::Decode);
            assert_eq!(printed.is_some(), t + 1 == DECODE_WINDOW, "token {t}");
            if let Some(l) = printed {
                assert!(l.contains(" step=3.00ms"), "{l}");
                assert!(l.contains(" token="), "the gap between tokens: {l}");
            }
        }
        assert_eq!(b.decode_tokens, 0, "the window restarts");
    }
}
