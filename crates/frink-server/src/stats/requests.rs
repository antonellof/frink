//! The ring buffer behind `/admin/stats`.
//!
//! Keyed by the `request_id` the server already assigns and already
//! states on the wire, so a UI joins a log row to the message that
//! produced it by equality rather than by the claiming heuristic
//! `bc5074aa:docs/plans/on-hold/frink-ui.md` describes one reference product resorting
//! to when concurrent chats stole each other's numbers.
//!
//! `duration_ms` and `decode_ms` stay separate here and all the way out
//! to the wire. `duration_ms` is the whole server-side request: queue
//! wait, prefill and decode. `decode_ms` is the decode loop alone.
//! Dividing completion tokens by the former reports a 50 tok/s model as
//! 5 whenever the prompt is long, and every number computed from that
//! is then wrong in the same direction -- which is exactly why the plan
//! calls conflating them out by name.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use frink_api::RecentRequest;
use frink_core::summary_stats::{mean_of_present, percentile};

use super::histogram::Histogram;
use super::ring::RequestRing;

use crate::attribution::Attribution;

/// Requests remembered. The contract says the last 200.
pub(crate) const RING_CAPACITY: usize = 200;

/// The most rows one `/v1/requests` poll will return, however large a
/// `limit` it asks for. A page bigger than the ring cannot exist, and a
/// caller who wants more history than the ring holds needs a longer
/// ring, not a longer page.
pub(crate) const MAX_PAGE: usize = RING_CAPACITY;

/// Seconds to the first token: a short prompt on a warm cache is tens of
/// milliseconds, a long cold prefill is seconds.
const TTFT_BOUNDS: [f64; 11] = [0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0];
/// Seconds per output token, decode alone: a small model on Metal is a
/// few milliseconds, a large one on a CPU is a fraction of a second.
const PER_TOKEN_BOUNDS: [f64; 11] = [
    0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0,
];
/// Seconds per whole request: queue wait, prefill and decode.
const DURATION_BOUNDS: [f64; 11] = [
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

/// Everything `/admin/stats` reports that is not already an
/// `AppState` counter.
pub(crate) struct Stats {
    recent: Mutex<RequestRing<RecentRequest>>,
    tokens_prompt_total: AtomicU64,
    tokens_generated_total: AtomicU64,
    /// Prompt tokens the prefix cache served, beside
    /// `tokens_prompt_total`: their ratio is the cache's hit rate in
    /// tokens.
    tokens_cached_total: AtomicU64,
    /// Over every request, where the ring keeps only the last few:
    /// `/metrics` is scraped as a rate, so it has to count from start.
    ttft: Histogram,
    per_token: Histogram,
    duration: Histogram,
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}

impl Stats {
    pub(crate) fn new() -> Self {
        Stats {
            recent: Mutex::new(RequestRing::new(RING_CAPACITY)),
            tokens_prompt_total: AtomicU64::new(0),
            tokens_generated_total: AtomicU64::new(0),
            tokens_cached_total: AtomicU64::new(0),
            ttft: Histogram::new(&TTFT_BOUNDS),
            per_token: Histogram::new(&PER_TOKEN_BOUNDS),
            duration: Histogram::new(&DURATION_BOUNDS),
        }
    }

    /// Records one finished request. Called after the response has been
    /// produced, from whichever path knows the real token counts -- the
    /// streaming path records when its generation task ends, not when
    /// the SSE handler returns, because the handler returns before a
    /// single token exists.
    pub(crate) fn record(&self, entry: RecentRequest) {
        self.tokens_prompt_total
            .fetch_add(entry.prompt_tokens as u64, Ordering::Relaxed);
        self.tokens_generated_total
            .fetch_add(entry.completion_tokens as u64, Ordering::Relaxed);
        if let Some(cached) = entry.cached_tokens {
            self.tokens_cached_total
                .fetch_add(cached as u64, Ordering::Relaxed);
        }
        // Each from the field that measured it, and none derived from
        // another: a request that did not time its first token or its
        // decode loop adds nothing to that histogram.
        if let Some(ttft) = entry.ttft_ms {
            self.ttft.observe(ttft / 1e3);
        }
        if let (Some(decode), true) = (entry.decode_ms, entry.completion_tokens > 0) {
            self.per_token
                .observe(decode / 1e3 / entry.completion_tokens as f64);
        }
        self.duration.observe(entry.duration_ms as f64 / 1e3);
        self.recent
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(entry);
    }

    pub(crate) fn recent(&self) -> Vec<RecentRequest> {
        self.recent
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .rows()
            .cloned()
            .collect()
    }

    /// One incremental page for `/v1/requests`, plus the cursor to poll
    /// with next and how many rows fell out of the ring before this
    /// poll could see them.
    pub(crate) fn page(&self, since: u64, limit: usize) -> (Vec<RecentRequest>, u64, u64) {
        let ring = self.recent.lock().unwrap_or_else(|p| p.into_inner());
        let page = ring.since(since, limit.clamp(1, MAX_PAGE));
        (
            page.rows.into_iter().cloned().collect(),
            page.cursor,
            page.missed,
        )
    }

    /// How many requests this process has recorded in total, retained
    /// in the ring or long since evicted.
    pub(crate) fn recorded_total(&self) -> u64 {
        self.recent
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .recorded_total()
    }

    /// The 95th-percentile whole-request latency over the ring,
    /// nearest-rank -- so it names a request that really took that
    /// long. `None` before anything has been served.
    pub(crate) fn p95_duration_ms(&self) -> Option<f64> {
        let ring = self.recent.lock().unwrap_or_else(|p| p.into_inner());
        let durations: Vec<f64> = ring.rows().map(|r| r.duration_ms as f64).collect();
        percentile(&durations, 95.0)
    }

    /// Mean time-to-first-token over the rows that HAVE one.
    ///
    /// A non-streamed request has no TTFT; counting those as zero would
    /// make the server look faster the fewer clients stream.
    pub(crate) fn ttft_mean_ms(&self) -> Option<f64> {
        let ring = self.recent.lock().unwrap_or_else(|p| p.into_inner());
        let ttfts: Vec<Option<f64>> = ring.rows().map(|r| r.ttft_ms).collect();
        mean_of_present(ttfts)
    }

    pub(crate) fn tokens_prompt_total(&self) -> u64 {
        self.tokens_prompt_total.load(Ordering::Relaxed)
    }

    pub(crate) fn tokens_generated_total(&self) -> u64 {
        self.tokens_generated_total.load(Ordering::Relaxed)
    }

    /// The `/metrics` lines these counters and histograms make, on every
    /// decode path: the batcher's own counters are only part of it.
    pub(crate) fn render_metrics(&self) -> String {
        let mut out = format!(
            "# HELP frink_prompt_tokens_total Prompt tokens of every finished request.\n\
             # TYPE frink_prompt_tokens_total counter\n\
             frink_prompt_tokens_total {}\n\
             # HELP frink_prompt_tokens_cached_total Prompt tokens the prefix cache served instead of a prefill.\n\
             # TYPE frink_prompt_tokens_cached_total counter\n\
             frink_prompt_tokens_cached_total {}\n\
             # HELP frink_generation_tokens_total Completion tokens of every finished request.\n\
             # TYPE frink_generation_tokens_total counter\n\
             frink_generation_tokens_total {}\n",
            self.tokens_prompt_total(),
            self.tokens_cached_total.load(Ordering::Relaxed),
            self.tokens_generated_total(),
        );
        out.push_str(&self.ttft.render(
            "frink_time_to_first_token_seconds",
            "Time from the request to its first token, for requests that timed it.",
        ));
        out.push_str(&self.per_token.render(
            "frink_time_per_output_token_seconds",
            "Decode time per completion token, the decode loop alone.",
        ));
        out.push_str(&self.duration.render(
            "frink_request_duration_seconds",
            "Whole server-side request time: queue wait, prefill and decode.",
        ));
        out
    }
}

/// Everything a finished request knows about itself, named at the call
/// site.
///
/// A struct rather than a parameter list because the two durations and
/// the two attribution fields are individually easy to swap by accident
/// and impossible to catch by type -- `duration_ms` and a decode time
/// are both `u64`-ish, `via_api_key` and `client` are both
/// `Option<String>`. Naming them at every call site is the cheapest
/// guard there is.
pub(crate) struct Record<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) route: &'a str,
    /// The model that served it, not the one the request named. See
    /// [`frink_api::RecentRequest::model`].
    pub(crate) model: Option<String>,
    pub(crate) status: u16,
    pub(crate) stream: bool,
    /// Whole server-side wall time: queue wait, prefill and decode.
    pub(crate) duration_ms: u64,
    /// `None` from a path that cannot time itself. Never a stand-in
    /// built out of `duration_ms`.
    pub(crate) usage: Option<&'a frink_api::Usage>,
    pub(crate) attribution: &'a Attribution,
}

/// Builds one ring-buffer entry from what a finished request knows.
///
/// Keeps the two durations separate and never derives one from the
/// other; a caller that cannot time the decode loop passes `usage:
/// None` rather than reusing `duration_ms`.
pub(crate) fn entry(record: Record<'_>) -> RecentRequest {
    let usage = record.usage;
    RecentRequest {
        request_id: record.request_id.to_string(),
        at_ms: crate::tasks::now_ms(),
        route: record.route.to_string(),
        model: record.model,
        status: record.status,
        prompt_tokens: usage.map(|u| u.prompt_tokens).unwrap_or(0),
        completion_tokens: usage.map(|u| u.completion_tokens).unwrap_or(0),
        ttft_ms: usage.and_then(|u| u.time_to_first_token_ms),
        duration_ms: record.duration_ms,
        decode_ms: usage.and_then(|u| u.generation_duration_ms),
        cached_tokens: usage.and_then(|u| u.cached_tokens),
        stream: record.stream,
        acceptance_length: usage.and_then(|u| u.acceptance_length),
        draft_accept_rate_per_position: usage
            .and_then(|u| u.draft_accept_rate_per_position.clone()),
        via_api_key: record.attribution.via_api_key.clone(),
        client: record.attribution.client.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage() -> frink_api::Usage {
        frink_api::Usage::new(100, 10)
            .with_timings(1.0, 0.1)
            .with_ttft(0.9)
    }

    #[test]
    fn an_entry_keeps_the_two_durations_apart() {
        let e = entry(Record {
            request_id: "chatcmpl-1",
            route: frink_api::routes::V1_CHAT_COMPLETIONS,
            model: Some("served-model".to_string()),
            status: 200,
            stream: true,
            duration_ms: 1_100,
            usage: Some(&usage()),
            attribution: &Attribution::default(),
        });
        assert_eq!(e.duration_ms, 1_100);
        assert_eq!(e.decode_ms, Some(100.0));
        assert_eq!(e.ttft_ms, Some(900.0));
        assert_eq!(e.prompt_tokens, 100);
        assert_eq!(e.completion_tokens, 10);
    }

    /// One finished request reaches every family `/metrics` renders:
    /// its tokens, the cached share of its prompt, and one observation
    /// in each latency histogram, in the bucket its own field puts it.
    /// A request that timed nothing adds to the counts and the duration
    /// only.
    #[test]
    fn a_finished_request_reaches_the_metrics() {
        let stats = Stats::new();
        let timed = frink_api::Usage::new(100, 10)
            .with_timings(1.0, 0.1)
            .with_ttft(0.9)
            .with_cached_tokens(60);
        for (usage, duration_ms) in [(Some(&timed), 1_100), (None, 50)] {
            stats.record(entry(Record {
                request_id: "chatcmpl-m",
                route: frink_api::routes::V1_CHAT_COMPLETIONS,
                model: None,
                status: 200,
                stream: true,
                duration_ms,
                usage,
                attribution: &Attribution::default(),
            }));
        }
        let body = stats.render_metrics();
        for line in [
            "frink_prompt_tokens_total 100\n",
            "frink_prompt_tokens_cached_total 60\n",
            "frink_generation_tokens_total 10\n",
            // 0.9 s to the first token: above 0.5, inside 1.
            "frink_time_to_first_token_seconds_bucket{le=\"0.5\"} 0\n",
            "frink_time_to_first_token_seconds_bucket{le=\"1\"} 1\n",
            "frink_time_to_first_token_seconds_count 1\n",
            // 100 ms of decode over 10 tokens is 10 ms each.
            "frink_time_per_output_token_seconds_bucket{le=\"0.005\"} 0\n",
            "frink_time_per_output_token_seconds_bucket{le=\"0.01\"} 1\n",
            "frink_time_per_output_token_seconds_count 1\n",
            // Both requests have a duration: 0.05 s and 1.1 s.
            "frink_request_duration_seconds_bucket{le=\"0.1\"} 1\n",
            "frink_request_duration_seconds_bucket{le=\"1\"} 1\n",
            "frink_request_duration_seconds_bucket{le=\"2.5\"} 2\n",
            "frink_request_duration_seconds_count 2\n",
        ] {
            assert!(
                body.contains(line),
                "missing `{}` in:\n{body}",
                line.trim_end()
            );
        }
    }

    #[test]
    fn an_untimed_request_reports_null_rather_than_reusing_the_total() {
        let e = entry(Record {
            request_id: "chatcmpl-2",
            route: "/v1/completions",
            model: None,
            status: 200,
            stream: false,
            duration_ms: 42,
            usage: None,
            attribution: &Attribution::default(),
        });
        assert_eq!(e.decode_ms, None);
        assert_eq!(e.ttft_ms, None);
        assert_eq!(e.duration_ms, 42);
        assert_eq!(e.prompt_tokens, 0);
    }

    #[test]
    fn speculation_metrics_reach_the_admin_ring() {
        // /admin/stats is where a drafter's behaviour over many
        // requests is visible; per-position accept rates only mean
        // something in aggregate, so they have to survive the hop from
        // the response body into the ring.
        let usage = frink_api::Usage::new(100, 12)
            .with_timings(1.0, 0.1)
            .with_speculation(5, 7, 10, vec![0.95, 0.7, 0.4]);
        let e = entry(Record {
            request_id: "chatcmpl-3",
            route: frink_api::routes::V1_CHAT_COMPLETIONS,
            model: None,
            status: 200,
            stream: false,
            duration_ms: 1_100,
            usage: Some(&usage),
            attribution: &Attribution::default(),
        });
        assert_eq!(e.acceptance_length, Some(2.4));
        assert_eq!(
            e.draft_accept_rate_per_position,
            Some(vec![0.95, 0.7, 0.4]),
            "the per-position curve is the only way suffix decay shows up"
        );
    }

    #[test]
    fn a_non_speculative_request_leaves_the_acceptance_columns_empty() {
        let e = entry(Record {
            request_id: "chatcmpl-4",
            route: frink_api::routes::V1_CHAT_COMPLETIONS,
            model: None,
            status: 200,
            stream: false,
            duration_ms: 42,
            usage: Some(&usage()),
            attribution: &Attribution::default(),
        });
        assert_eq!(e.acceptance_length, None);
        assert_eq!(e.draft_accept_rate_per_position, None);
    }

    #[test]
    fn token_totals_accumulate_across_requests() {
        let stats = Stats::new();
        for _ in 0..3 {
            stats.record(entry(Record {
                request_id: "id",
                route: "/r",
                model: None,
                status: 200,
                stream: false,
                duration_ms: 1,
                usage: Some(&usage()),
                attribution: &Attribution::default(),
            }));
        }
        assert_eq!(stats.tokens_prompt_total(), 300);
        assert_eq!(stats.tokens_generated_total(), 30);
    }

    #[test]
    fn the_ring_keeps_the_newest_entries_and_drops_the_oldest() {
        let stats = Stats::new();
        for i in 0..RING_CAPACITY + 25 {
            stats.record(entry(Record {
                request_id: &format!("id-{i}"),
                route: "/r",
                model: None,
                status: 200,
                stream: false,
                duration_ms: 1,
                usage: None,
                attribution: &Attribution::default(),
            }));
        }
        let recent = stats.recent();
        assert_eq!(recent.len(), RING_CAPACITY);
        assert_eq!(recent[0].request_id, "id-25");
        assert_eq!(
            recent[RING_CAPACITY - 1].request_id,
            format!("id-{}", RING_CAPACITY + 24)
        );
    }

    #[test]
    fn an_error_response_is_recorded_with_its_status() {
        let stats = Stats::new();
        stats.record(entry(Record {
            request_id: "id-e",
            route: "/v1/chat/completions",
            model: None,
            status: 503,
            stream: false,
            duration_ms: 3,
            usage: None,
            attribution: &Attribution::default(),
        }));
        assert_eq!(stats.recent()[0].status, 503);
    }

    /// Attribution is carried, not re-derived: the entry states the key
    /// fingerprint and the self-declared label the request arrived
    /// with, and nothing else.
    #[test]
    fn an_entry_carries_the_attribution_it_was_given() {
        let attribution = Attribution {
            via_api_key: Some("key-deadbeef".to_string()),
            client: Some("frink-studio".to_string()),
        };
        let e = entry(Record {
            request_id: "id",
            route: "/r",
            model: None,
            status: 200,
            stream: false,
            duration_ms: 1,
            usage: None,
            attribution: &attribution,
        });
        assert_eq!(e.via_api_key.as_deref(), Some("key-deadbeef"));
        assert_eq!(e.client.as_deref(), Some("frink-studio"));

        let anonymous = entry(Record {
            request_id: "id",
            route: "/r",
            model: None,
            status: 200,
            stream: false,
            duration_ms: 1,
            usage: None,
            attribution: &Attribution::default(),
        });
        assert_eq!(anonymous.via_api_key, None);
        assert_eq!(anonymous.client, None);
    }
}
