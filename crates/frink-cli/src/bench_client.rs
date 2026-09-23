//! The serving-benchmark methodology, with no socket in it.
//!
//! `frink bench` is deliberately single-stream and HTTP-free: it
//! measures kernels against `llama-bench`, which is the right tool for
//! that question and the wrong one for "what does `frink-server` do
//! under sixteen concurrent clients". This module is the second
//! question's arithmetic -- how a request is pinned so it does exactly
//! the requested work, how a stream's tics become TTFT and TPOT, how
//! those become percentiles, and what throughput means over a run.
//!
//! Everything here takes time as a parameter and returns numbers. There
//! is no clock, no HTTP client and no tokenizer, which is what lets the
//! rules below be asserted on any host rather than inferred from a live
//! server that might have been slow for an unrelated reason.
//!
//! # The four rules that decide whether the numbers mean anything
//!
//! 1. **Every request does exactly the requested work.** Temperature 0,
//!    top-k 1, EOS ignored, and an exact output length -- see
//!    [`BenchSampling`]. A benchmark whose requests stop at their own
//!    EOS is measuring the model's opinion about when to stop, pooled
//!    with the server's speed.
//! 2. **The TTFT/TPOT split is POSITIONAL.** The first tic is
//!    time-to-first-token; every later one is an inter-token sample.
//!    See [`RequestTiming`] for why splitting on content instead makes
//!    two identically-behaving servers report different numbers.
//! 3. **Percentiles are index-based nearest-rank**, over samples
//!    *pooled across requests* rather than averaged per request and then
//!    averaged again. A p99 of per-request means is not a p99 of
//!    anything.
//! 4. **Throughput is total tokens over the run's whole span**, not the
//!    sum of per-request rates. See [`BenchReport::output_throughput`].
//!
//! Ported from FreeToken's `benchmark/client.py`; see
//! `docs/THIRD_PARTY_NOTICES.md`.

use frink_core::summary_stats::percentile;

/// The sampling every benchmark request is pinned to.
///
/// Not a default a caller may override: a benchmark whose requests
/// sample differently from each other is measuring the sampler, and one
/// whose requests stop at their own EOS is measuring the model's
/// opinion about length. Both are legitimate questions and neither is
/// this one.
///
/// `ignore_eos` is the load-bearing field. Without it a run's requests
/// finish at different lengths, so the slowest percentile is whichever
/// request happened to be asked for the most tokens -- a fact about the
/// prompts, reported as a fact about the server.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchSampling {
    pub temperature: f32,
    pub top_k: usize,
    pub ignore_eos: bool,
    /// Tokens every request must produce, exactly.
    pub output_len: usize,
}

impl BenchSampling {
    pub fn new(output_len: usize) -> Self {
        BenchSampling {
            temperature: 0.0,
            top_k: 1,
            ignore_eos: true,
            output_len,
        }
    }
}

/// One request's stream, as a list of arrival times in seconds.
///
/// # Why the split is positional
///
/// A tic is recorded per streamed chunk, and the FIRST tic is TTFT
/// whatever that chunk carried. The tempting alternative -- "TTFT is
/// the first chunk with non-empty content" -- makes the number depend
/// on whether a server opens its stream with a role-only chunk. Two
/// servers that produce their first token at the identical moment then
/// report different TTFTs, and the difference is a wire convention
/// rather than a speed.
///
/// # What is not a tic
///
/// A keepalive is not a token and must not be timed as one. A server
/// quiet through a long prefill sends them precisely during the window
/// TTFT is measuring, so counting one would report a TTFT of about
/// fifteen seconds for a request whose first token had not arrived --
/// the single most misleading number this module could produce. Nor is
/// the terminal chunk, which closes the stream and carries nothing.
/// See [`is_token_chunk`], and record only what it accepts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestTiming {
    started_s: f64,
    tics_s: Vec<f64>,
    reported_tokens: Option<usize>,
    /// The server's own `usage.prompt_tokens` and `usage.cached_tokens`
    /// for this request, when it reported them.
    ///
    /// Both or neither: a reuse fraction needs a denominator from the
    /// same request, and reading the two from different terminal
    /// frames is how a 97% figure gets computed against somebody
    /// else's prompt.
    prefix: Option<PrefixUse>,
}

/// One request's prompt size and how much of it the server did not
/// have to prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixUse {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
}

impl RequestTiming {
    /// A request dispatched at `started_s`.
    pub fn started(started_s: f64) -> Self {
        RequestTiming {
            started_s,
            tics_s: Vec::new(),
            reported_tokens: None,
            prefix: None,
        }
    }

    /// Records the prompt size and prefix reuse the server stated.
    ///
    /// `cached_tokens` is clamped to the prompt: a reuse figure above
    /// 100% is a server bug, and reporting it as one is more useful
    /// than propagating it into a mean that then reads as plausible.
    pub fn report_prefix(&mut self, prompt_tokens: usize, cached_tokens: usize) {
        self.prefix = Some(PrefixUse {
            prompt_tokens,
            cached_tokens: cached_tokens.min(prompt_tokens),
        });
    }

    pub fn prefix(&self) -> Option<PrefixUse> {
        self.prefix
    }

    /// The token count the SERVER stated, from the terminal chunk's
    /// `usage.completion_tokens`.
    ///
    /// Recorded because chunks and tokens are not the same thing. A
    /// server that streams incrementally sends one chunk per token and
    /// the two agree; one that buffers -- frink's synthetic-weights
    /// path, a continuous-batching row with no incremental stream to
    /// ride on -- delivers N tokens in a single chunk. Counting chunks
    /// there reports one token for a request that produced a hundred,
    /// and the throughput figure is then wrong by that factor.
    ///
    /// Timing still comes from the tics: what a buffered server cannot
    /// offer is inter-token detail, which is reported as absent rather
    /// than invented.
    pub fn report_tokens(&mut self, tokens: usize) {
        self.reported_tokens = Some(tokens);
    }

    /// Records one streamed chunk's arrival. Keepalives must be
    /// filtered out before this is called.
    pub fn tic(&mut self, at_s: f64) {
        self.tics_s.push(at_s);
    }

    /// Tokens this request produced: the server's own count when it
    /// stated one, and the chunk count otherwise.
    ///
    /// See [`report_tokens`](Self::report_tokens) for why the two are
    /// not interchangeable.
    pub fn tokens(&self) -> usize {
        self.reported_tokens.unwrap_or(self.tics_s.len())
    }

    /// Streamed chunks recorded, which is what the timing is built
    /// from.
    pub fn tics(&self) -> usize {
        self.tics_s.len()
    }

    /// Time to first token, or `None` for a request that produced
    /// nothing.
    ///
    /// `None` and not zero: a request that never answered has no TTFT,
    /// and a zero would be the best sample in the set -- so a run in
    /// which half the requests failed would report a *better* p50 than
    /// one in which they all succeeded.
    pub fn ttft(&self) -> Option<f64> {
        self.tics_s.first().map(|first| first - self.started_s)
    }

    /// Inter-token times, one per chunk after the first.
    ///
    /// Empty for a request that produced one token or none: there is no
    /// interval between a token and nothing.
    pub fn tpot_samples(&self) -> Vec<f64> {
        self.tics_s.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Dispatch to last chunk. `None` for a request that produced
    /// nothing, for the same reason [`ttft`](Self::ttft) is.
    pub fn end_to_end(&self) -> Option<f64> {
        self.tics_s.last().map(|last| last - self.started_s)
    }

    /// When this request's last chunk arrived, for the run span.
    pub fn last_tic_s(&self) -> Option<f64> {
        self.tics_s.last().copied()
    }

    pub fn started_s(&self) -> f64 {
        self.started_s
    }
}

/// Whether a streamed chat chunk carries a token, and should therefore
/// be timed.
///
/// The test is **the delta is non-empty**, and the two things it
/// excludes are the two that would each corrupt a different number:
///
/// - A **keepalive** (`delta: {}`, no finish reason). A server quiet
///   through a long prefill sends them during exactly the window TTFT
///   measures, so timing one reports a TTFT of about the keepalive
///   interval for a request whose first token had not arrived -- the
///   single most misleading number this module could produce.
/// - The **terminal chunk** (`delta: {}`, a finish reason). It closes
///   the stream and carries no token. Timing it adds one to every
///   request's token count and one spurious inter-token sample per
///   request, taken over the gap between the last real token and the
///   server writing its usage block.
///
/// What it deliberately INCLUDES is a role-only opening chunk, whose
/// delta carries `role` and no content. That is a real event in the
/// stream's positional sequence, and skipping it would make a server
/// that sends one report a different TTFT from a server that folds the
/// role into its first content chunk -- a wire convention reported as a
/// speed. See [`RequestTiming`].
pub fn is_token_chunk(chunk: &serde_json::Value) -> bool {
    let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) else {
        // No `choices` at all is the usage-only or replay-keepalive
        // shape. Not a token either way.
        return false;
    };
    choices.iter().any(|choice| {
        choice
            .get("delta")
            .and_then(|d| d.as_object())
            .is_some_and(|d| !d.is_empty())
    })
}

/// A latency figure at the three percentiles a serving run is read at,
/// plus the mean.
///
/// `None` throughout when there were no samples, rather than zeros: a
/// run in which nothing answered has no latency, and a table of zeros
/// reads as a run that was instantaneous.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Latency {
    pub mean: Option<f64>,
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub p99: Option<f64>,
}

impl Latency {
    /// Nearest-rank at 50 / 90 / 99, over the samples as given.
    ///
    /// The samples must already be POOLED across requests. Per-request
    /// means, percentiled, answer a different question and flatter the
    /// tail: one request that stalled for a second in the middle of two
    /// hundred fast tokens disappears into its own mean before the p99
    /// ever sees it.
    pub fn of(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Latency::default();
        }
        Latency {
            mean: Some(samples.iter().sum::<f64>() / samples.len() as f64),
            p50: percentile(samples, 50.0),
            p90: percentile(samples, 90.0),
            p99: percentile(samples, 99.0),
        }
    }
}

/// Sums the per-request prefix figures, or `None` when no request
/// reported one.
fn prefix_totals(timings: &[RequestTiming]) -> Option<PrefixTotals> {
    let uses: Vec<PrefixUse> = timings.iter().filter_map(RequestTiming::prefix).collect();
    if uses.is_empty() {
        return None;
    }
    Some(PrefixTotals {
        prompt_tokens: uses.iter().map(|u| u.prompt_tokens).sum(),
        cached_tokens: uses.iter().map(|u| u.cached_tokens).sum(),
        requests_with_reuse: uses.iter().filter(|u| u.cached_tokens > 0).count(),
        requests_reporting: uses.len(),
    })
}

/// What a whole run measured.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchReport {
    /// Requests that produced at least one token.
    pub completed: usize,
    /// Requests that produced nothing. Reported rather than dropped:
    /// a run where a third of the requests failed and the rest were
    /// fast is not a fast run.
    pub failed: usize,
    pub output_tokens: usize,
    /// Wall-clock seconds from the first dispatch to the last chunk.
    pub duration_s: f64,
    pub ttft: Latency,
    pub tpot: Latency,
    pub end_to_end: Latency,
    /// Prompt positions across the run, and how many the prefix cache
    /// covered. `None` when no request reported the pair, which is
    /// what a server with no prefix cache configured looks like --
    /// distinct from one that was consulted and missed, which reports
    /// a zero numerator.
    pub prefix: Option<PrefixTotals>,
}

/// Prefix reuse across a whole run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixTotals {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    /// Requests that reused at least one position.
    pub requests_with_reuse: usize,
    /// Requests that reported the pair at all.
    pub requests_reporting: usize,
}

impl PrefixTotals {
    /// Share of prompt positions the cache covered, 0.0 to 1.0.
    ///
    /// Over the run's TOTAL prompt tokens rather than the mean of
    /// per-request fractions: a run of one long cold prompt and nine
    /// short warm ones is mostly cold work, and averaging fractions
    /// would call it 90% reused.
    pub fn reuse_fraction(&self) -> Option<f64> {
        (self.prompt_tokens > 0).then(|| self.cached_tokens as f64 / self.prompt_tokens as f64)
    }
}

impl BenchReport {
    /// Summarize a finished run.
    ///
    /// The span runs from the EARLIEST dispatch to the LATEST chunk, so
    /// a staggered arrival pattern's idle head is inside it. That is
    /// what makes the throughput below comparable between a run that
    /// fired everything at once and one that replayed a trace.
    pub fn of(timings: &[RequestTiming]) -> Self {
        // Completion is about the STREAM, not the stated count: a
        // request that answered nothing has no timing to contribute
        // whatever a usage block might have claimed.
        let completed = timings.iter().filter(|t| t.tics() > 0).count();
        let ttfts: Vec<f64> = timings.iter().filter_map(RequestTiming::ttft).collect();
        let tpots: Vec<f64> = timings
            .iter()
            .flat_map(RequestTiming::tpot_samples)
            .collect();
        let e2es: Vec<f64> = timings
            .iter()
            .filter_map(RequestTiming::end_to_end)
            .collect();

        let first_start = timings
            .iter()
            .map(RequestTiming::started_s)
            .fold(f64::INFINITY, f64::min);
        let last_tic = timings
            .iter()
            .filter_map(RequestTiming::last_tic_s)
            .fold(f64::NEG_INFINITY, f64::max);
        let duration_s = if completed == 0 {
            0.0
        } else {
            (last_tic - first_start).max(0.0)
        };

        BenchReport {
            completed,
            failed: timings.len() - completed,
            output_tokens: timings.iter().map(RequestTiming::tokens).sum(),
            duration_s,
            ttft: Latency::of(&ttfts),
            tpot: Latency::of(&tpots),
            end_to_end: Latency::of(&e2es),
            prefix: prefix_totals(timings),
        }
    }

    /// Tokens per second across the whole run.
    ///
    /// Total tokens over the total span, and **not** the sum or mean of
    /// per-request rates. Those measure how fast one request goes while
    /// the others wait for it, which on a saturated server is a number
    /// that gets *better* the worse the queueing is: each request is
    /// individually fast once it finally runs. Only the whole-run figure
    /// answers "how much did this server actually deliver".
    pub fn output_throughput(&self) -> Option<f64> {
        (self.duration_s > 0.0).then(|| self.output_tokens as f64 / self.duration_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_benchmark_request_is_pinned_so_it_does_exactly_the_requested_work() {
        let sampling = BenchSampling::new(128);
        assert_eq!(sampling.temperature, 0.0);
        assert_eq!(sampling.top_k, 1);
        assert!(
            sampling.ignore_eos,
            "a request that stops at its own EOS measures the model's \
             opinion about length, not the server's speed"
        );
        assert_eq!(sampling.output_len, 128);
    }

    /// The positional rule. The first tic is TTFT whatever it carried,
    /// and every later one is an inter-token sample -- so a server that
    /// opens with a role-only chunk and one that does not report the
    /// same TTFT for the same first token.
    #[test]
    fn the_first_tic_is_the_time_to_first_token_whatever_it_carried() {
        let mut t = RequestTiming::started(10.0);
        t.tic(10.5);
        t.tic(10.6);
        t.tic(10.8);

        assert_eq!(t.ttft(), Some(0.5));
        assert_eq!(t.tokens(), 3);
        let tpot = t.tpot_samples();
        assert_eq!(tpot.len(), 2, "one interval per chunk after the first");
        assert!((tpot[0] - 0.1).abs() < 1e-9);
        assert!((tpot[1] - 0.2).abs() < 1e-9);
        assert!((t.end_to_end().unwrap() - 0.8).abs() < 1e-9);
    }

    /// A request that answered nothing has no latency, and reporting a
    /// zero would make it the best sample in the set -- so a run in
    /// which half the requests failed would show a better p50 than one
    /// in which they all succeeded.
    #[test]
    fn a_request_that_produced_nothing_has_no_latency_rather_than_zero() {
        let t = RequestTiming::started(1.0);
        assert_eq!(t.ttft(), None);
        assert_eq!(t.end_to_end(), None);
        assert!(t.tpot_samples().is_empty());

        let mut one = RequestTiming::started(1.0);
        one.tic(1.25);
        assert_eq!(one.ttft(), Some(0.25));
        assert!(
            one.tpot_samples().is_empty(),
            "there is no interval between a token and nothing"
        );
    }

    /// The two exclusions, each of which corrupts a different number.
    ///
    /// A keepalive arrives during exactly the window TTFT measures, so
    /// timing one reports a TTFT of about the keepalive interval for a
    /// request whose first token had not arrived. The terminal chunk
    /// carries no token, so timing it adds one to every request's token
    /// count and one spurious inter-token sample taken over the gap
    /// between the last real token and the usage block.
    #[test]
    fn only_a_chunk_that_carries_something_is_timed_as_a_token() {
        let keepalive = json!({
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
        });
        assert!(!is_token_chunk(&keepalive));

        let terminal = json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        });
        assert!(
            !is_token_chunk(&terminal),
            "the terminal frame is not a token"
        );

        let token = json!({
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}],
        });
        assert!(is_token_chunk(&token));

        // A role-only opening chunk IS a real event in the positional
        // sequence: skipping it would make a server that sends one
        // report a different TTFT from one that folds the role into its
        // first content chunk.
        let opening = json!({
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
        });
        assert!(is_token_chunk(&opening));

        // Reasoning is output too, and a reasoning model's first
        // reasoning delta is its time-to-first-token.
        let thought = json!({
            "choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}],
        });
        assert!(is_token_chunk(&thought));

        // The replay keepalive and the usage-only chunk carry no
        // choices at all.
        assert!(!is_token_chunk(&json!({"choices": []})));
        assert!(!is_token_chunk(&json!({"object": "chat.completion.chunk"})));
    }

    /// TPOT samples are POOLED across requests before the percentile,
    /// never averaged per request and percentiled after. One request
    /// that stalled in the middle of many fast tokens must reach the
    /// p99; inside its own mean it never does.
    #[test]
    fn inter_token_samples_are_pooled_across_requests_not_averaged_twice() {
        // Two requests of 50 tokens: 49 inter-token samples each, 98
        // pooled. One of them is a 1-second stall in the middle of an
        // otherwise fast request.
        let mut stalled = RequestTiming::started(0.0);
        let mut at = 0.0;
        stalled.tic(at);
        for i in 0..49 {
            at += if i == 25 { 1.0 } else { 0.001 };
            stalled.tic(at);
        }
        let mut fast = RequestTiming::started(0.0);
        let mut at = 0.0;
        fast.tic(at);
        for _ in 0..49 {
            at += 0.001;
            fast.tic(at);
        }

        let report = BenchReport::of(&[stalled.clone(), fast]);
        let p99 = report.tpot.p99.expect("samples exist");
        assert!(
            p99 > 0.5,
            "the stall must survive to the p99, got {p99} -- pooling is the \
             whole point"
        );

        // What percentiling per-request means would have done to it.
        let samples = stalled.tpot_samples();
        let per_request_means = [samples.iter().sum::<f64>() / samples.len() as f64, 0.001];
        let flattened = Latency::of(&per_request_means).p99.unwrap();
        assert!(
            flattened < p99,
            "averaging first hides the tail, which is why it is not done"
        );
    }

    /// Throughput is total tokens over the whole span, and the span
    /// includes the idle head of a staggered run. Summing per-request
    /// rates instead gets BETTER the worse the queueing is, because
    /// each request is individually fast once it finally runs.
    #[test]
    fn throughput_is_the_whole_runs_tokens_over_the_whole_runs_span() {
        let mut early = RequestTiming::started(0.0);
        early.tic(1.0);
        early.tic(2.0);
        // Dispatched late; the span must still start at 0.
        let mut late = RequestTiming::started(8.0);
        late.tic(9.0);
        late.tic(10.0);

        let report = BenchReport::of(&[early, late]);
        assert_eq!(report.output_tokens, 4);
        assert_eq!(report.duration_s, 10.0, "first dispatch to last chunk");
        assert_eq!(report.output_throughput(), Some(0.4));
    }

    /// A run where requests failed is not a fast run, so the failures
    /// are counted rather than dropped -- and a run where NOTHING
    /// answered has no duration and no throughput rather than a
    /// division by zero.
    #[test]
    fn failed_requests_are_counted_and_an_empty_run_has_no_throughput() {
        let report = BenchReport::of(&[RequestTiming::started(0.0), RequestTiming::started(1.0)]);
        assert_eq!(report.completed, 0);
        assert_eq!(report.failed, 2);
        assert_eq!(report.duration_s, 0.0);
        assert_eq!(report.output_throughput(), None);
        assert_eq!(report.ttft, Latency::default());
    }

    /// Chunks and tokens are not the same thing. A server that streams
    /// incrementally sends one chunk per token; one that buffers
    /// delivers many tokens in a single chunk, and counting chunks
    /// there reports one token for a request that produced a hundred --
    /// so throughput comes out wrong by that factor.
    #[test]
    fn a_buffered_server_is_credited_with_the_tokens_it_says_it_produced() {
        let mut buffered = RequestTiming::started(0.0);
        buffered.tic(2.0);
        assert_eq!(buffered.tokens(), 1, "with nothing else to go on");

        buffered.report_tokens(100);
        assert_eq!(buffered.tokens(), 100, "the server's own count wins");
        assert_eq!(buffered.tics(), 1, "the timing still comes from the stream");

        let report = BenchReport::of(&[buffered]);
        assert_eq!(report.output_tokens, 100);
        assert_eq!(report.output_throughput(), Some(50.0));
        assert_eq!(
            report.tpot,
            Latency::default(),
            "a buffered stream has no inter-token detail, and it is \
             reported as absent rather than invented"
        );
    }

    fn with_prefix(prompt: usize, cached: usize) -> RequestTiming {
        let mut t = RequestTiming::started(0.0);
        t.tic(0.1);
        t.tic(0.2);
        t.report_prefix(prompt, cached);
        t
    }

    /// **Reuse is a share of the run's TOTAL prompt tokens, not the
    /// mean of per-request shares.**
    ///
    /// One long cold prompt beside nine short warm ones is mostly cold
    /// work. Averaging fractions calls that 90% reused; the honest
    /// number here is 31%.
    #[test]
    fn reuse_is_weighted_by_prompt_size_and_not_by_request() {
        let mut timings = vec![with_prefix(1000, 0)];
        timings.extend((0..9).map(|_| with_prefix(50, 50)));
        let totals = BenchReport::of(&timings).prefix.expect("reported");

        assert_eq!((totals.prompt_tokens, totals.cached_tokens), (1450, 450));
        let f = totals.reuse_fraction().expect("a denominator");
        assert!(
            (f - 450.0 / 1450.0).abs() < 1e-9,
            "weighted by request instead of by token: {f}"
        );
        assert!(
            f < 0.32,
            "the mean of per-request fractions would be 0.9 here; got {f}"
        );
    }

    /// A server that reports nothing has no prefix cache to speak of,
    /// which is not the same as one that was consulted and missed.
    #[test]
    fn no_request_reporting_is_absent_rather_than_zero() {
        let mut t = RequestTiming::started(0.0);
        t.tic(0.1);
        assert_eq!(BenchReport::of(&[t]).prefix, None);

        let missed = BenchReport::of(&[with_prefix(100, 0)])
            .prefix
            .expect("consulted");
        assert_eq!(missed.cached_tokens, 0);
        assert_eq!(missed.requests_with_reuse, 0);
        assert_eq!(missed.reuse_fraction(), Some(0.0));
    }

    /// Reuse above 100% is a server bug. Clamping keeps it from
    /// becoming a plausible-looking mean that hides one.
    #[test]
    fn reuse_cannot_exceed_the_prompt_it_is_a_share_of() {
        let totals = BenchReport::of(&[with_prefix(100, 400)])
            .prefix
            .expect("reported");
        assert_eq!(totals.cached_tokens, 100);
        assert_eq!(totals.reuse_fraction(), Some(1.0));
    }
}
