//! Continuous-batching decode scheduler: many in-flight sequences share
//! one `Decoder::forward_multi_seq` step per tick instead of each
//! request owning a private `forward_token` loop.
//!
//! Opt-in via `FRINK_CONTINUOUS_BATCHING=1`; mutually exclusive with
//! the KV pool and prefix cache (those paths keep the private-loop
//! `generate`). Stop sequences go through the same
//! [`StopMatcher`](crate::stop::StopMatcher) as
//! `generate::sample_until_stop`: a token-level match on the id before
//! the token is detokenized, plus output-suffix buffering that
//! withholds any tail that could still complete a stop string. One
//! implementation, so a row in a batch and a row on its own cannot
//! disagree about where an answer ends.
//!
//! **Chunked prefill.** A prompt is not prefilled in one uninterruptible
//! `forward_token` loop. Each accepted job first becomes a
//! [`PrefillState`] -- a resumable state machine over (`caches`,
//! `tokens_processed`, `tokens_remaining`) whose `step_chunk` runs at
//! most `prefill_chunk` tokens and reports whether the prompt is
//! finished. That is what makes a prompt a *bounded* unit of work: the
//! worker interleaves **one** prefill chunk with **one** batched decode
//! step per tick, so a long prompt joining the batch delays in-flight
//! decodes by one chunk rather than by its whole length. Chunks are
//! taken round-robin from the waiting prompts, so N concurrent long
//! prompts still cost decode one chunk per tick, not N.
//!
//! Chunking is a *scheduling* boundary, not a numerical one: a chunk is
//! still the same per-token `forward_token` sequence at the same
//! positions, so chunk size never changes the logits or the sampled
//! tokens (asserted by `prefill_chunking_does_not_change_logits`).
//!
//! **Keyed row state.** In-flight rows live in a `HashMap<Uid, Slot>`
//! with a separate admission-ordered `Vec<Uid>`, never in a `Vec<Slot>`
//! addressed by batch position. Batch membership changes on almost
//! every tick -- a row finishes on EOS, on a stop string, on its token
//! budget -- and a positional table renumbers its survivors when that
//! happens (`swap_remove` moves the last row into the removed slot). A
//! `Uid` captured before a removal still names its own row afterwards,
//! or nothing at all; a batch index captured before a removal quietly
//! names a *different request*, whose sampler, stop strings and reply
//! channel are not the ones the caller asked for. That is the bug class
//! oMLX had to monkey-patch around, and it is silent: no panic, no
//! error, just the wrong constraints applied to the wrong row.
//!
//! Per-request sampler state (`Slot::sampler`, seeded per request) lives
//! in the row for the same reason -- a shared or global RNG would make
//! one request's output depend on how many others were in flight.
//!
//! **Knobs.** `FRINK_CB_MAX_SEQS`: cap on concurrent in-flight
//! sequences, counting prompts still prefilling (default: unlimited).
//! At the cap, new jobs stay queued in the channel until a slot frees;
//! only a completely idle worker blocks on `recv`.
//! `FRINK_CB_PREFILL_CHUNK`: prompt tokens per prefill chunk (default
//! [`DEFAULT_PREFILL_CHUNK`]). `FRINK_CB_MAX_QUEUE`: how many jobs may
//! wait for admission before new ones are refused with
//! [`DecodeError::QueueFull`] (default [`DEFAULT_MAX_QUEUE`]).
//!
//! **Block admission.** A sequence count is not a memory budget: 8
//! concurrent 200-token chats and 8 concurrent 100k-token documents
//! are both "8 sequences" and are three orders of magnitude apart in
//! KV. So admission is on an *integer block budget* --
//! `blocks_needed <= blocks_free`, where a block is a fixed run of
//! `FRINK_CB_KV_BLOCK_SIZE` token positions and a request needs
//! `ceil((prompt + max_tokens) / block_size)` of them, reserved for its
//! whole lifetime at admission and released when it finishes.
//!
//! Integer blocks rather than a byte watermark, and this is the one
//! place the source study is deliberately *not* copied: oMLX charges
//! admission against sampled process footprint because MLX's allocator
//! will not tell it what a cache costs. frink reads its own KV layout
//! out of the GGUF header, so the exact question -- will this fit --
//! has an exact integer answer, and an exact answer does not need a
//! watermark, a hysteresis band, or a background pressure enforcer.
//!
//! Refusals are typed and split, because they send an operator to
//! different knobs -- an OOM, or a single "server busy", sends them to
//! the wrong one:
//!
//! - **Longer than any one request may be.** `FRINK_CB_MAX_CONTEXT`
//!   positions, if set. `context_length_exceeded` -> 400. The
//!   request's own size is the problem.
//! - **Bigger than the whole KV budget.** `blocks_needed >
//!   blocks_total`: an idle server refuses it identically.
//!   `device_memory_budget_exceeded` -> 400.
//! - **Does not fit right now.** Not an error at all: the job waits at
//!   the head of the admission queue until enough blocks come back.
//! - **Too many already waiting.** [`DecodeError::QueueFull`] -> 503
//!   with `Retry-After`, because that one really does clear.
//!
//! Both 400s name `estimated_bytes` and `limit_bytes` -- the KV cost of
//! the request and of the ceiling it hit, priced from the model's own
//! `KvShape`, so the arithmetic is checkable rather than asserted --
//! and carry separate counters, so "one request too big" and "too many
//! requests" are never read as the same event.
//!
//! Admission is strict FIFO. A skip-ahead policy ("admit the next job
//! that fits") raises utilization and can starve a large request
//! indefinitely behind a stream of small ones -- a queue that reorders
//! by size systematically punishes exactly the requests that already
//! wait longest. FIFO cannot starve, so FIFO it is until there is a
//! measured reason to change it.
//!
//! `FRINK_CB_KV_BLOCKS` unset means no block budget at all, which is
//! today's behaviour: `max_seqs` alone. The two caps compose -- a job
//! must satisfy both.
//!
//! **Deferred abort.** A cancellation never mutates the batch from the
//! canceller's thread. `CancelToken::on_cancel` gives the scheduler a
//! callback that does one thing -- push an [`AbortId`] into
//! [`AbortInbox`] -- and the worker drains that inbox at the top of a
//! tick, *between* forward passes, and does the removal itself.
//!
//! The indirection is the point. Dropping a sequence means dropping the
//! KV buffers a forward pass may be mid-way through reading, and the
//! `std::mem::take` that lifts a row's caches into the batch leaves the
//! row holding empty vectors for the duration of the step -- so a
//! removal landing inside that window would either free live buffers or
//! scatter results back into a row that is no longer there. frink-metal
//! has the same hazard in its own terms: its kernels
//! `waitUntilCompleted` per dispatch, so no command buffer outlives a
//! `forward_multi_seq` call, and the step boundary really is a point
//! where nothing is in flight -- but only for the thread that owns the
//! step. That is why the mutation is deferred onto it rather than done
//! wherever the cancel arrived.
//!
//! A cancel is honoured wherever the request has got to: still queued
//! (never prefilled at all), mid-prefill (the remaining chunks are
//! never run), or decoding (it stops at the next step). All three reply
//! [`FinishReason::Cancelled`] with the tokens produced so far, because
//! a cancelled generation has partial output and throwing it away
//! serves nobody.
//!
//! **Queue cap.** The job channel is unbounded, so without a cap a
//! client retry storm turns straight into unbounded memory: every
//! retry parks another prompt (and its reply channel) in the queue,
//! and the server's only signal that it is drowning is the RSS graph.
//! [`QueueGate`] bounds the *waiting* jobs -- in-flight sequences are
//! `FRINK_CB_MAX_SEQS`'s business -- and a refusal is a fast, cheap
//! 503 with `Retry-After` rather than a slow, expensive timeout. A job
//! holds its queue slot until it is *admitted*, not until the worker
//! pulls it off the channel: with a block budget a job can sit in the
//! worker's own admission queue for a while, and a cap that stopped
//! counting it there would stop bounding anything.

mod batcher;
mod block_budget;
pub(super) mod cache_aware;
mod clock;
mod config;
mod counters;
mod prefill;
mod queue;
mod row;
/// Wired: `BatchStatus`, `PoolUsage`, `PrefillSnapshot`,
/// `StatusReporter` and the decode log interval, all from [`worker`].
/// Unwired: `PrefillSnapshot::from_chunks` and
/// `StatusReporter::decode_log_interval`, which report on
/// [`super::admission`]'s chunks rather than this batcher's. Closes
/// with `sched-time-debt` (roadmap `c3-serving-and-kv`).
#[allow(dead_code)]
mod status;
pub(super) mod step_budget;
mod worker;

#[cfg(test)]
mod tests;

pub(crate) use batcher::ContinuousBatcher;
pub(crate) use config::BatcherConfig;
pub(crate) use status::PoolUsage;
