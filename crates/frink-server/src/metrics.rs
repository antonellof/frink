//! The `/metrics` route: every counter the server keeps, in one
//! Prometheus text body.
//!
//! Moved out of `lib.rs` unchanged before a counter was added to it,
//! because that file is the largest in the repo and nothing shrinks on
//! its own.

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};

use crate::{lock_cache, AppState};

/// Prometheus text-exposition format (`# HELP`/`# TYPE` plus
/// `name value` lines), so this endpoint can be scraped directly by a
/// Prometheus server or anything compatible with that format without
/// frink needing to speak any particular metrics client library.
pub(crate) async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    use std::sync::atomic::Ordering;

    let cache_stats = lock_cache(&state.response_cache).stats();
    let active = state.active();
    let requests_total = state.requests_total.load(Ordering::Relaxed);
    let errors_total = state.request_errors_total.load(Ordering::Relaxed);
    let uptime = state.started_at.elapsed().as_secs_f64();

    let body = format!(
        "# HELP frink_requests_total Total chat completion requests received.\n\
         # TYPE frink_requests_total counter\n\
         frink_requests_total {requests_total}\n\
         # HELP frink_request_errors_total Total chat completion requests that returned an error.\n\
         # TYPE frink_request_errors_total counter\n\
         frink_request_errors_total {errors_total}\n\
         # HELP frink_cache_hits_total Whole-response cache hits.\n\
         # TYPE frink_cache_hits_total counter\n\
         frink_cache_hits_total {}\n\
         # HELP frink_cache_misses_total Whole-response cache misses.\n\
         # TYPE frink_cache_misses_total counter\n\
         frink_cache_misses_total {}\n\
         # HELP frink_cache_entries Current whole-response cache entry count.\n\
         # TYPE frink_cache_entries gauge\n\
         frink_cache_entries {}\n\
         # HELP frink_synthetic_weights 1 if serving synthetic random weights instead of a real checkpoint.\n\
         # TYPE frink_synthetic_weights gauge\n\
         frink_synthetic_weights {}\n\
         # HELP frink_uptime_seconds Seconds since this server process started.\n\
         # TYPE frink_uptime_seconds gauge\n\
         frink_uptime_seconds {uptime}\n",
        cache_stats.hits,
        cache_stats.misses,
        cache_stats.entries,
        // With nothing loaded there are no weights at all, synthetic or
        // otherwise; 0 is the reading that keeps the gauge meaning
        // "serving noise" rather than "serving nothing".
        active
            .as_ref()
            .map(|a| a.is_synthetic() as u8)
            .unwrap_or(0),
    );

    // Expert-store counters, present only when the model streams
    // routed experts through the bounded cache
    // (FRINK_EXPERT_CACHE_BYTES).
    let body = match active
        .as_ref()
        .and_then(|a| a.expert_store_stats())
    {
        Some(es) => format!(
            "{body}\
             # HELP frink_expert_cache_hits_total Expert-store cache hits.\n\
             # TYPE frink_expert_cache_hits_total counter\n\
             frink_expert_cache_hits_total {}\n\
             # HELP frink_expert_cache_misses_total Expert-store cache misses (source reads).\n\
             # TYPE frink_expert_cache_misses_total counter\n\
             frink_expert_cache_misses_total {}\n\
             # HELP frink_expert_cache_evictions_total Expert-store LRU evictions.\n\
             # TYPE frink_expert_cache_evictions_total counter\n\
             frink_expert_cache_evictions_total {}\n\
             # HELP frink_expert_cache_pass_throughs_total Acquires served uncached (entry could not fit the budget).\n\
             # TYPE frink_expert_cache_pass_throughs_total counter\n\
             frink_expert_cache_pass_throughs_total {}\n\
             # HELP frink_expert_cache_bytes_read_total Bytes read from the checkpoint for expert misses.\n\
             # TYPE frink_expert_cache_bytes_read_total counter\n\
             frink_expert_cache_bytes_read_total {}\n\
             # HELP frink_expert_cache_resident_bytes Current expert-cache footprint in bytes.\n\
             # TYPE frink_expert_cache_resident_bytes gauge\n\
             frink_expert_cache_resident_bytes {}\n",
            es.hits, es.misses, es.evictions, es.pass_throughs, es.bytes_read, es.resident_bytes,
        ),
        None => body,
    };

    // Scheduler counters, present only under continuous batching
    // (FRINK_CONTINUOUS_BATCHING=1). `prefill_chunks` next to
    // `prefill_tokens` is what makes chunked prefill observable: their
    // ratio is the effective chunk size the worker actually ran.
    let body = match active.as_ref().and_then(|a| a.batcher.as_ref()) {
        Some(batcher) => {
            let sched = batcher.stats();
            format!(
                "{body}\
                 # HELP frink_prefill_chunks_total Bounded prefill chunks the batch scheduler has run.\n\
                 # TYPE frink_prefill_chunks_total counter\n\
                 frink_prefill_chunks_total {}\n\
                 # HELP frink_prefill_tokens_total Prompt tokens run through chunked prefill.\n\
                 # TYPE frink_prefill_tokens_total counter\n\
                 frink_prefill_tokens_total {}\n\
                 # HELP frink_decode_steps_total Batched decode steps the batch scheduler has run.\n\
                 # TYPE frink_decode_steps_total counter\n\
                 frink_decode_steps_total {}\n\
                 # HELP frink_scheduler_queue_depth Requests waiting for admission to the batch scheduler.\n\
                 # TYPE frink_scheduler_queue_depth gauge\n\
                 frink_scheduler_queue_depth {}\n\
                 # HELP frink_scheduler_queue_rejected_total Requests refused with 503 because the admission queue was full.\n\
                 # TYPE frink_scheduler_queue_rejected_total counter\n\
                 frink_scheduler_queue_rejected_total {}\n\
                 # HELP frink_kv_blocks_total KV blocks in the scheduler's admission budget (0 when unconfigured).\n\
                 # TYPE frink_kv_blocks_total gauge\n\
                 frink_kv_blocks_total {}\n\
                 # HELP frink_kv_blocks_free KV blocks not reserved by an in-flight request.\n\
                 # TYPE frink_kv_blocks_free gauge\n\
                 frink_kv_blocks_free {}\n\
                 # HELP frink_kv_block_size Token positions per KV block.\n\
                 # TYPE frink_kv_block_size gauge\n\
                 frink_kv_block_size {}\n\
                 # HELP frink_kv_rejected_too_large_total Requests refused with 400 because they exceed the whole KV block budget.\n\
                 # TYPE frink_kv_rejected_too_large_total counter\n\
                 frink_kv_rejected_too_large_total {}\n\
                 # HELP frink_kv_rejected_context_length_total Requests refused with 400 for exceeding the per-request context ceiling.\n\
                 # TYPE frink_kv_rejected_context_length_total counter\n\
                 frink_kv_rejected_context_length_total {}\n\
                 # HELP frink_scheduler_aborted_total Requests the batch scheduler stopped because they were cancelled.\n\
                 # TYPE frink_scheduler_aborted_total counter\n\
                 frink_scheduler_aborted_total {}\n\
                 # HELP frink_scheduler_preempted_total Decoding rows preempted back to the queue to admit a waiting request (FRINK_CB_PREEMPT_AFTER_TICKS).\n\
                 # TYPE frink_scheduler_preempted_total counter\n\
                 frink_scheduler_preempted_total {preempted}\n\
                 # HELP frink_scheduler_max_seqs Cap on in-flight sequences (-np / FRINK_CB_MAX_SEQS); 0 when unlimited.\n\
                 # TYPE frink_scheduler_max_seqs gauge\n\
                 frink_scheduler_max_seqs {}\n\
                 # HELP frink_scheduler_prefill_chunk Prompt tokens per prefill chunk (-b / -ub / FRINK_CB_PREFILL_CHUNK).\n\
                 # TYPE frink_scheduler_prefill_chunk gauge\n\
                 frink_scheduler_prefill_chunk {}\n",
                sched.prefill_chunks,
                sched.prefill_tokens,
                sched.decode_steps,
                sched.queue_depth,
                sched.queue_rejected,
                sched.kv_blocks_total,
                sched.kv_blocks_free,
                sched.kv_block_size,
                sched.kv_rejected_too_large,
                sched.kv_rejected_context_length,
                sched.aborted,
                sched.max_seqs,
                sched.prefill_chunk,
                // Named, not positional: a counter inserted into the
                // middle of the list above shifts every positional
                // argument after it onto the wrong line.
                preempted = sched.preempted,
            )
        }
        None => body,
    };

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    )
        .into_response()
}
