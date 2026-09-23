//! The one clock a batched row is timed by.
//!
//! Continuous batching used to answer with a `Usage` carrying token
//! counts and nothing else: no `prompt_per_second`, no
//! `predicted_per_second`, no `time_to_first_token_ms`. The private
//! `generate` loop set all three, so whether a client saw a rate
//! depended on which decode path served it -- and on Metal, where
//! batching is the default, it never did. Frink Studio reads every
//! number under an answer out of `usage`, so the columns were simply
//! blank.
//!
//! The durable half of that fix is this type. It owns both the
//! timestamps AND the construction of the row's `Usage`, so a caller
//! cannot build one and forget the rates: there is no path from a
//! finished row to a `Usage` that does not go through
//! [`RowClock::usage`].
//!
//! The same shape recurred once, which is why `cached_tokens` is an
//! ARGUMENT here rather than a field somebody sets: the batched path
//! reused a prefix correctly -- measured, 894 ms cold against 409 ms
//! warm on the same prompt -- and reported `cached_tokens: 0` for
//! every request, because the private `generate` loop set that field
//! in `request_tail` and the batcher never reached it. A caller that
//! forgets it now fails to compile.

use std::time::Instant;

use frink_api::Usage;

/// Timestamps for one row, from the start of its prefill to the token
/// that ended it.
///
/// A row is timed by wall clock, which under batching includes the
/// steps taken for OTHER rows in the same batch. That is deliberate and
/// matches llama-server: the rate a caller cares about is the rate its
/// own request was served at, not a rate the row would have reached
/// alone.
pub(super) struct RowClock {
    /// Start of this row's prefill, set when the job is admitted to the
    /// worker rather than when it was enqueued: queue wait belongs to
    /// the request's duration, not to the prefill rate.
    started: Instant,
    /// Set once, when the prompt is through and the row becomes a
    /// decode slot.
    prefill_done: Option<Instant>,
    /// Set once, by the FIRST generated token. A later token must not
    /// move it, which is the whole reason this is not a plain field
    /// assignment at the push site.
    first_token: Option<Instant>,
}

impl RowClock {
    /// Starts the clock at the beginning of a row's prefill.
    pub(super) fn start() -> Self {
        Self {
            started: Instant::now(),
            prefill_done: None,
            first_token: None,
        }
    }

    /// Marks the end of prefill. Called once, on the handover from
    /// `Prefill` to `Slot`.
    pub(super) fn prefill_finished(&mut self) {
        self.prefill_done = Some(Instant::now());
    }

    /// Records a generated token. Only the first one moves the clock;
    /// every later call is a no-op, so the push site does not have to
    /// know which token it is holding.
    pub(super) fn token(&mut self) {
        self.first_token.get_or_insert_with(Instant::now);
    }

    /// The row's `Usage`, rates and prefix reuse included.
    ///
    /// A row that failed during prefill has no `prefill_done`, and its
    /// decode phase is empty rather than negative: `Usage::with_timings`
    /// leaves a rate unset for a zero-length phase, so an unfinished
    /// row reports durations without inventing a throughput.
    ///
    /// `cached_tokens` carries the same distinction the private loop
    /// makes: `Some(0)` is a prefix cache that was consulted and
    /// missed, `None` is no prefix cache to consult. A caller cannot
    /// pass neither.
    pub(super) fn usage(
        &self,
        prompt_tokens: usize,
        completion_tokens: usize,
        cached_tokens: Option<usize>,
    ) -> Usage {
        let ended = Instant::now();
        let prefill_end = self.prefill_done.unwrap_or(ended);
        let prefill_secs = prefill_end.duration_since(self.started).as_secs_f64();
        let decode_secs = ended.duration_since(prefill_end).as_secs_f64();
        let mut usage =
            Usage::new(prompt_tokens, completion_tokens).with_timings(prefill_secs, decode_secs);
        if let Some(cached) = cached_tokens {
            usage = usage.with_cached_tokens(cached);
        }
        match self.first_token {
            Some(first) => usage.with_ttft(first.duration_since(self.started).as_secs_f64()),
            None => usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape the bug had: a finished row must report both rates,
    /// not just its token counts.
    #[test]
    fn a_finished_row_reports_both_rates() {
        // Each phase is given a measurable duration on purpose.
        // `with_timings` leaves a rate unset for a ZERO-length phase
        // rather than dividing into infinity, so a test that finished
        // both phases inside one clock tick would fail against correct
        // code -- it did, once the machine was quiet enough.
        let mut clock = RowClock::start();
        std::thread::sleep(std::time::Duration::from_millis(2));
        clock.prefill_finished();
        clock.token();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let usage = clock.usage(41, 32, None);
        assert!(
            usage.prompt_per_second.is_some(),
            "prefill rate missing: {usage:?}"
        );
        assert!(
            usage.predicted_per_second.is_some(),
            "decode rate missing: {usage:?}"
        );
        assert!(
            usage.time_to_first_token_ms.is_some(),
            "ttft missing: {usage:?}"
        );
    }

    /// The first token owns the TTFT. A row whose later tokens moved it
    /// would report the time to its LAST token, which is not a
    /// latency anybody asked for.
    #[test]
    fn only_the_first_token_sets_ttft() {
        let mut clock = RowClock::start();
        clock.prefill_finished();
        clock.token();
        let first = clock.first_token.expect("first token recorded");
        std::thread::sleep(std::time::Duration::from_millis(2));
        clock.token();
        assert_eq!(
            clock.first_token.expect("still recorded"),
            first,
            "a later token moved the TTFT"
        );
    }

    /// A row that ended during prefill has no decode phase. It reports
    /// durations, and no decode rate, rather than a negative duration
    /// or a fabricated rate.
    #[test]
    fn a_row_that_never_decoded_has_no_decode_rate() {
        let clock = RowClock::start();
        let usage = clock.usage(41, 0, None);
        assert_eq!(usage.predicted_per_second, None);
        assert_eq!(usage.time_to_first_token_ms, None);
        assert!(usage.generation_duration_ms.is_some_and(|ms| ms >= 0.0));
    }
}
