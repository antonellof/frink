//! The worker tick: drain, admit, apply cancellations, run one prefill
//! chunk and one batched decode step, then flush whatever finished.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use frink_core::cache::{KvCache, PagedKvCache};
use frink_models::Decoder;
use frink_models::MultiSeqKv;

use crate::generate::{acquire_paged_caches, DecodeError, FinishReason, PagedKvConfig, Usage};
use crate::stop::StopStep;

use super::block_budget::BlockBudget;
use super::clock::RowClock;
use super::config::{decode_log_interval_from_env, send_finished, BatcherConfig, DecodeFn};
use super::counters::Counters;
use super::prefill::{Prefill, PrefillState};
use super::queue::{AbortId, AbortInbox, QueueGate};
use super::row::{Job, RowKv, Rows, Slot, Uid};
use super::status::{BatchStatus, PoolUsage, PrefillSnapshot, StatusReporter};

/// Moves everything currently on the channel into the worker's own
/// admission queue. Both are "waiting for admission" as far as
/// [`QueueGate`] is concerned, so nothing is released here.
pub(super) fn drain_channel(rx: &Receiver<Job>, waiting: &mut VecDeque<Job>) {
    while let Ok(job) = rx.try_recv() {
        waiting.push_back(job);
    }
}

/// Admits as many waiting jobs as *both* caps allow, turning each into
/// a `Prefill`.
///
/// The sequence cap counts prompts that are still prefilling as well as
/// rows already decoding: a prefilling prompt holds a full set of KV
/// caches, so not counting it would let the worker exceed `max_seqs` by
/// however many prompts happen to be in flight.
///
/// The block cap is the real memory statement:
/// `blocks_needed <= blocks_free`, reserved here for the request's
/// whole lifetime and released in `Rows::flush_finished`.
///
/// Strict FIFO: a head job that does not fit stops the line rather than
/// being skipped over. See the module note on why the skip-ahead
/// alternative is a starvation bug, not an optimization.
#[allow(clippy::too_many_arguments)] // one per thing admission needs;
                                     // bundling them would only move the
                                     // same list behind a struct.
pub(super) fn admit(
    decoder: &Arc<Decoder>,
    waiting: &mut VecDeque<Job>,
    prefills: &mut VecDeque<Prefill>,
    decoding: usize,
    config: &BatcherConfig,
    queue: &QueueGate,
    budget: &BlockBudget,
    paged: Option<&PagedKvConfig>,
) -> PrefillSnapshot {
    // Counted HERE, as each prompt is accepted, and not read back off
    // the live requests afterwards. By the time this tick's forward has
    // run, every admitted prompt's processed-token count has advanced
    // to its device length -- so a line built from the live rows would
    // report `#new-token` as one per request and `#cached-token` as the
    // whole prompt, which is the upstream test's planted-wrong-values
    // case exactly.
    let mut snapshot = PrefillSnapshot::default();
    while let Some(job) = waiting.front() {
        if decoding + prefills.len() >= config.max_seqs {
            break;
        }
        let blocks = job.blocks;
        if !budget.try_reserve(blocks) {
            break;
        }
        let job = waiting.pop_front().expect("front() just succeeded");
        // Admitted: the job has stopped waiting, so its queue slot goes
        // back now rather than when it was pulled off the channel.
        queue.release();
        match accept(decoder, job, config.prefill_chunk, paged) {
            Some(prefill) => {
                snapshot.new_seqs += 1;
                snapshot.new_tokens += prefill.prompt_tokens;
                prefills.push_back(prefill);
            }
            // Rejected at validation (a token outside the vocabulary):
            // it never becomes a row, so nothing else would ever give
            // its reservation back.
            None => budget.release(blocks),
        }
    }
    snapshot
}

/// Applies every pending cancellation, at a step boundary, on the
/// worker thread.
///
/// `carried` holds ids that arrived before their job did -- a cancel
/// can beat its own submission through the channel -- so they are kept
/// and retried rather than dropped, which would leave a request running
/// after the client asked it to stop.
///
/// The three states a cancelled request can be in are handled where
/// they live, because they cost different things to abandon:
///
/// - **waiting**: no KV, no reservation, no work done. Replied to and
///   dropped; its queue slot goes back.
/// - **prefilling**: holds KV and a reservation but has produced no
///   tokens. Dropped, reservation released.
/// - **decoding**: marked finished, and left to leave through
///   `flush_finished` like any other completed row, so there is exactly
///   one path that releases blocks and replies.
pub(super) fn apply_aborts(
    inbox: &AbortInbox,
    carried: &mut HashSet<AbortId>,
    waiting: &mut VecDeque<Job>,
    prefills: &mut VecDeque<Prefill>,
    rows: &mut Rows,
    queue: &QueueGate,
    budget: &BlockBudget,
) {
    carried.extend(inbox.drain());
    if carried.is_empty() {
        return;
    }

    let mut stopped = 0u64;

    // Queued, never started.
    let mut still_waiting = VecDeque::with_capacity(waiting.len());
    while let Some(job) = waiting.pop_front() {
        if carried.remove(&job.abort) {
            queue.release();
            send_finished(
                &job.reply,
                Ok((
                    FinishReason::Cancelled,
                    Vec::new(),
                    String::new(),
                    Usage::new(job.prompt_tokens.len(), 0),
                )),
            );
            stopped += 1;
        } else {
            still_waiting.push_back(job);
        }
    }
    *waiting = still_waiting;

    // Mid-prefill: the remaining chunks are never run.
    let mut still_prefilling = VecDeque::with_capacity(prefills.len());
    while let Some(prefill) = prefills.pop_front() {
        if carried.remove(&prefill.abort) {
            budget.release(prefill.blocks);
            send_finished(
                &prefill.reply,
                Ok((
                    FinishReason::Cancelled,
                    Vec::new(),
                    String::new(),
                    Usage::new(prefill.prompt_tokens, 0),
                )),
            );
            stopped += 1;
        } else {
            still_prefilling.push_back(prefill);
        }
    }
    *prefills = still_prefilling;

    // Decoding: marked here, removed by `flush_finished` below.
    for id in rows.mark_cancelled(carried) {
        carried.remove(&id);
        stopped += 1;
    }

    if stopped > 0 {
        inbox.aborted.fetch_add(stopped, Ordering::Relaxed);
    }
}

/// The gauges both status lines carry.
///
/// This is the block model's admission policy, and the only one on the
/// serving path. `crate::serving::admission` holds the WINDOW and
/// RECURRENT models' -- page-aligned chunks, window reclaim, recurrent
/// slots -- which are precisely the cases this batcher excludes below.
/// The two are not one policy written twice; see that module's docs
/// before merging them.
/// `window` and `recurrent` are `None` and not zero: frink serves no
/// windowed or recurrent model through this batcher, and the reporter
/// omits a pool it is given `None` for entirely rather than printing a
/// row of zeros an operator would size against.
///
/// With no block budget configured there is no pool to report, so the
/// KV gauge is an empty one -- `used` and `total` both zero, which
/// `ratio()` reads as an idle pool rather than a full one.
pub(super) fn batch_status(
    rows: &Rows,
    prefills: &VecDeque<Prefill>,
    waiting: &VecDeque<Job>,
    budget: &BlockBudget,
) -> BatchStatus {
    let kv_pages = match budget.total {
        Some(total) => PoolUsage::from_available(total, budget.free.load(Ordering::Relaxed)),
        None => PoolUsage::default(),
    };
    BatchStatus {
        running_reqs: rows.len() + prefills.len(),
        queue_reqs: waiting.len(),
        kv_pages,
        page_size: budget.block_size,
        window: None,
        recurrent: None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn worker_loop(
    decoder: Arc<Decoder>,
    decode: DecodeFn,
    rx: Receiver<Job>,
    config: BatcherConfig,
    counters: Arc<Counters>,
    queue: Arc<QueueGate>,
    budget: Arc<BlockBudget>,
    aborts: Arc<AbortInbox>,
    paged: Option<PagedKvConfig>,
) {
    let mut rows = Rows::default();
    let mut prefills: VecDeque<Prefill> = VecDeque::new();
    let mut waiting: VecDeque<Job> = VecDeque::new();
    // Cancellations that arrived before the job they name.
    let mut carried_aborts: HashSet<AbortId> = HashSet::new();
    // The operator-facing batch log. `frink-edge` holds no clock, so
    // the clock is here: one `Instant` per worker, read as seconds,
    // which is all the reporter's throughput arithmetic needs.
    let started = std::time::Instant::now();
    let mut reporter = StatusReporter::new(
        decode_log_interval_from_env(),
        started.elapsed().as_secs_f64(),
    );
    loop {
        // Only a completely idle worker blocks: with a prompt still
        // chunking, or a job waiting for capacity that an in-flight row
        // will return, there is always work to do on the next tick.
        if rows.is_empty() && prefills.is_empty() && waiting.is_empty() {
            match rx.recv() {
                Ok(job) => waiting.push_back(job),
                Err(_) => break,
            }
        }
        drain_channel(&rx, &mut waiting);
        // Before anything else this tick, and before any forward pass:
        // the one window in which mutating the batch is safe.
        apply_aborts(
            &aborts,
            &mut carried_aborts,
            &mut waiting,
            &mut prefills,
            &mut rows,
            &queue,
            &budget,
        );
        rows.flush_finished(&budget);
        let admitted = admit(
            &decoder,
            &mut waiting,
            &mut prefills,
            rows.len(),
            &config,
            &queue,
            &budget,
            paged.as_ref(),
        );
        if admitted.new_seqs > 0 {
            let status = batch_status(&rows, &prefills, &waiting, &budget);
            tracing::info!(
                "{}",
                reporter.report_prefill(started.elapsed().as_secs_f64(), &admitted, &status)
            );
        }
        let held = rows.blocks_held() + prefills.iter().map(|p| p.blocks).sum::<usize>();
        counters.peak_blocks.fetch_max(held, Ordering::Relaxed);
        // With nothing in flight the whole budget is free, and a job
        // that could not fit an empty server was refused at submission
        // -- so an idle worker with a non-empty queue would be a spin,
        // and this is where it would show up.
        debug_assert!(
            !(rows.is_empty() && prefills.is_empty() && !waiting.is_empty()),
            "idle worker cannot admit its queue: {} jobs stuck",
            waiting.len()
        );

        // One bounded prefill chunk per tick, round-robin across the
        // waiting prompts. Round-robin rather than "finish the head
        // first" so a long prompt cannot starve a short one behind it;
        // one chunk rather than "advance every pending prefill" so N
        // concurrent long prompts cost decode one chunk per tick, not N.
        if let Some(mut prefill) = prefills.pop_front() {
            let before = prefill.state.tokens_processed();
            let done = prefill.state.step_chunk();
            counters.prefill_chunks.fetch_add(1, Ordering::Relaxed);
            counters.prefill_tokens.fetch_add(
                (prefill.state.tokens_processed() - before) as u64,
                Ordering::Relaxed,
            );
            if done {
                rows.insert(prefill.into_slot());
            } else {
                prefills.push_back(prefill);
            }
        }

        if rows.is_empty() {
            continue;
        }

        let ready = rows.ready();
        if ready.is_empty() {
            rows.flush_finished(&budget);
            continue;
        }

        // Sample one token per ready row. A row that finishes here (EOS
        // or a stop match) simply does not join `active`.
        let mut active: Vec<Uid> = Vec::with_capacity(ready.len());
        for uid in ready {
            let Some(slot) = rows.get_mut(uid) else {
                continue;
            };
            // Through `sample_step`, the same call the private decode
            // loop makes. Calling `Sampler` directly here is what
            // dropped `response_format: {"type": "json_object"}` for
            // every batched request: an env var the caller cannot see
            // (`FRINK_CONTINUOUS_BATCHING`) decided a per-request
            // feature, and the only visible symptom was the final
            // `validate_json_object_output` turning into a 400.
            let next = match crate::sample_step::sample_next(
                &mut slot.sample,
                &slot.logits,
                &slot.params,
                // The row already keeps its prompt, for the radix
                // publish. The penalties window is the tail of
                // `prompt ++ generated`, so it needs the same slice.
                &slot.prompt_ids,
                &slot.generated_ids,
                &slot.stop_tokens,
                // Text, for the grammar and single-token stop
                // decisions. Lossy per token, which is what those
                // callers have always had; the EMITTED text below is
                // what #124 was about and it goes through the row's
                // own UTF-8 buffer.
                &|id| String::from_utf8_lossy(&decode(&[id])).into_owned(),
            ) {
                Ok(crate::sample_step::Step::Token { id, .. }) => id,
                // A complete grammar with nothing legal after it: this
                // row is finished, exactly as the private loop treats
                // it, and for the same reason.
                Ok(crate::sample_step::Step::GrammarComplete) => {
                    slot.finish = Some(FinishReason::Stop);
                    continue;
                }
                // A constraint this row cannot satisfy. It is this
                // ROW's failure and nobody else's, so it is answered
                // with an error and the batch carries on -- the same
                // statement `sample_until_stop` makes by returning
                // `Err`, which on the private path ends the one
                // generation it is running.
                Err(e) => {
                    slot.fail(e);
                    continue;
                }
            };
            // Three ways this token ends the answer, and they compose:
            // the model's own end-of-generation set (not `eos_id`
            // alone -- gemma-2 ends on `<end_of_turn>`), and a
            // single-token stop the caller asked for. All of them mean
            // the token is a terminator and not part of the output.
            //
            // `ignore_eos` suppresses the MODEL's set and only that, so
            // both decode paths agree: a benchmark asking to run past
            // EOS is not withdrawing its own fence, and the private
            // `generate` loop makes the same distinction.
            let model_eos = !slot.params.ignore_eos && slot.stop_tokens.contains(next);
            if model_eos || slot.stops.is_stop_token(next) {
                slot.finish = Some(FinishReason::Stop);
                continue;
            }
            slot.generated_ids.push(next);
            slot.clock.token();
            let piece = slot.utf8.push(&decode(&[next]));
            // An empty piece means the token was the middle of a
            // character. Nothing to show, and nothing a stop string
            // could match yet.
            if !piece.is_empty() && apply_stop_buffer(slot, &piece) {
                continue;
            }
            active.push(uid);
        }

        if !active.is_empty() {
            // `active[j]` names the row that owns `logits_batch[j]`.
            // The kernel takes slices, so the batch itself is
            // positional -- but the position maps to a *uid*, so the
            // scatter below cannot land on the wrong request even if
            // the table changes shape between steps.
            let tokens: Vec<usize> = active
                .iter()
                .map(|&uid| *rows.get(uid).unwrap().generated_ids.last().unwrap())
                .collect();
            let positions: Vec<usize> = active
                .iter()
                .map(|&uid| rows.get(uid).unwrap().pos)
                .collect();
            // Taken out of the rows for the call and put straight
            // back below, so one call sees the whole batch. For a paged
            // row this moves only the per-layer caches; the LEASE stays
            // in the row, so the page groups stay accounted even though
            // the caches are momentarily elsewhere.
            let paged = matches!(rows.get(active[0]).map(|s| &s.kv), Some(RowKv::Paged(_)));
            let mut contiguous_refs: Vec<Vec<KvCache>> = Vec::new();
            let mut paged_refs: Vec<Vec<PagedKvCache>> = Vec::new();
            for &uid in &active {
                let slot = rows.get_mut(uid).expect("active row exists");
                let pos = slot.pos;
                let token = *slot
                    .generated_ids
                    .last()
                    .expect("an active row has a token");
                match &mut slot.kv {
                    RowKv::Contiguous(c) => contiguous_refs.push(std::mem::take(c)),
                    RowKv::Paged(lease) => {
                        // Before the caches leave the lease: the slide
                        // moves page groups between the lease's own
                        // books and its block tables, so it has to see
                        // both. A no-op unless every layer slides by the
                        // same window.
                        lease.observe_sampled(token, pos + 1, false);
                        lease.before_step(pos);
                        paged_refs.push(std::mem::take(lease.caches_mut()));
                    }
                }
            }
            // `j` below indexes `active`, and only one of the two
            // vectors is populated, so it indexes that one. That holds
            // because the backing is a server-wide choice and every row
            // shares it -- asserted rather than assumed, since a mixed
            // batch would silently pair row `j` with another row's KV.
            debug_assert!(
                contiguous_refs.len() == active.len() || paged_refs.len() == active.len(),
                "every row in a batch must share one KV backing"
            );
            let logits_batch = if paged {
                let store = Arc::clone(match &rows.get(active[0]).expect("active row exists").kv {
                    RowKv::Paged(lease) => lease.store(),
                    RowKv::Contiguous(_) => unreachable!("checked above"),
                });
                decoder.forward_multi_seq_kv(
                    &tokens,
                    &positions,
                    &mut MultiSeqKv::Paged {
                        caches: &mut paged_refs,
                        stores: &store,
                    },
                )
            } else {
                decoder.forward_multi_seq(&tokens, &positions, &mut contiguous_refs)
            };
            counters.decode_steps.fetch_add(1, Ordering::Relaxed);
            // Every forward counts; only every Nth emits. The silent
            // ones are exactly what the emitted line's throughput is
            // measuring, and it measures the gap since the LAST EMITTED
            // line rather than the whole run -- what an operator reads
            // a decode log for is a change, and a lifetime mean is the
            // one statistic that cannot show one.
            if let Some(line) = reporter.report_decode(
                started.elapsed().as_secs_f64(),
                active.len(),
                &batch_status(&rows, &prefills, &waiting, &budget),
            ) {
                tracing::info!("{line}");
            }
            for (j, &uid) in active.iter().enumerate() {
                let slot = rows
                    .get_mut(uid)
                    .expect("an active row cannot vanish mid-step");
                match &mut slot.kv {
                    RowKv::Contiguous(c) => *c = std::mem::take(&mut contiguous_refs[j]),
                    RowKv::Paged(lease) => *lease.caches_mut() = std::mem::take(&mut paged_refs[j]),
                }
                slot.logits = logits_batch[j].clone();
                slot.pos += 1;
                if slot.generated_ids.len() >= slot.max_tokens {
                    slot.finish = Some(FinishReason::Length);
                }
            }
        }

        rows.flush_finished(&budget);
    }
}

/// Feeds `piece` through this row's stop matcher. Returns true when a
/// stop matched and the row should leave the active batch.
pub(super) fn apply_stop_buffer(slot: &mut Slot, piece: &str) -> bool {
    match slot.stops.push(piece) {
        StopStep::Emit(text) => {
            if !text.is_empty() {
                slot.visible.push_str(&text);
                let _ = slot.reply.send(super::config::BatcherEvent::Chunk(text));
            }
            false
        }
        StopStep::Matched { text, stop } => {
            if !text.is_empty() {
                slot.visible.push_str(&text);
                let _ = slot.reply.send(super::config::BatcherEvent::Chunk(text));
            }
            slot.finish = Some(FinishReason::StopSequence(stop));
            true
        }
    }
}

/// Validates a job and turns it into a waiting `Prefill`. No model work
/// happens here -- every prompt token is run by `step_chunk` on the
/// scheduler's own tick, which is the whole point of chunked prefill.
/// Returns `None` (having replied with the error) for a prompt this
/// model cannot accept at all.
pub(super) fn accept(
    decoder: &Arc<Decoder>,
    job: Job,
    chunk_size: usize,
    paged: Option<&PagedKvConfig>,
) -> Option<Prefill> {
    let vocab_size = decoder.config.vocab_size;
    if let Some(&bad) = job.prompt_tokens.iter().find(|&&t| t >= vocab_size) {
        send_finished(
            &job.reply,
            Err(DecodeError::TokenOutOfVocab {
                token: bad,
                vocab_size,
            }),
        );
        return None;
    }

    let state = match paged {
        Some(config) => {
            // Worst case for this row: its prompt plus everything it
            // may generate.
            let max_seq_len = job.prompt_tokens.len() + job.params.max_tokens;
            match acquire_paged_caches(decoder, config, &job.prompt_tokens, max_seq_len) {
                Ok(lease) => PrefillState::new_paged(
                    Arc::clone(decoder),
                    &job.prompt_tokens,
                    chunk_size,
                    lease,
                ),
                Err(_) => {
                    // The block budget said yes and the store said no.
                    // Refuse this row rather than admit one with no
                    // pages: the two counts are meant to agree, and the
                    // store is the one holding real memory.
                    send_finished(&job.reply, Err(DecodeError::KvPoolExhausted));
                    return None;
                }
            }
        }
        None => PrefillState::new(Arc::clone(decoder), &job.prompt_tokens, chunk_size),
    };
    Some(Prefill {
        state,
        clock: RowClock::start(),
        prompt_tokens: job.prompt_tokens.len(),
        params: job.params,
        stop_tokens: job.stop_tokens,
        reply: job.reply,
        abort: job.abort,
        blocks: job.blocks,
    })
}
