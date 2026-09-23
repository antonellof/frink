//! The handle callers hold: submits a job, owns the worker thread, and
//! reports what the worker has done.

use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use frink_models::tokenizer::StopTokens;
use frink_models::Decoder;

use crate::budget::ContextCeiling;
use crate::generate::{
    paged_hold_positions, paged_window_policy, DecodeError, FinishReason, GenerationParams,
    PagedKvConfig, Usage,
};
use crate::policy::anchor::WindowPolicy;

use super::block_budget::BlockBudget;
use super::config::{BatcherConfig, BatcherEvent, DecodeFn};
use super::counters::{BatcherStats, Counters};
use super::queue::{AbortInbox, QueueGate};
use super::row::Job;
use super::worker::worker_loop;

/// Owns a dedicated worker thread that batches decode steps. Cheap to
/// clone (`Sender` only); the worker stays alive as long as any clone
/// (or the original) exists.
#[derive(Clone)]
pub struct ContinuousBatcher {
    tx: Sender<Job>,
    counters: Arc<Counters>,
    queue: Arc<QueueGate>,
    budget: Arc<BlockBudget>,
    aborts: Arc<AbortInbox>,
    /// The window a paged row slides by, when the store is paged and
    /// every layer of this model slides by the same window.
    ///
    /// Held here so the block budget prices a request at what it will
    /// actually hold. Without it the budget would keep pricing a
    /// windowed request at its whole context and refuse admissions the
    /// paged store would happily serve -- two components disagreeing
    /// about the same server.
    paged_window: Option<WindowPolicy>,
    /// The knobs this worker was spawned with, kept so `stats()` can
    /// report them. The worker owns its own copy (`BatcherConfig` is
    /// `Copy`); this one exists to be *read back*, which is what makes
    /// `-np` and `-ub` verifiable from outside the process rather than
    /// only assertable from inside it.
    config: BatcherConfig,
}

pub(super) struct WorkerGuard {
    pub(super) _join: JoinHandle<()>,
}

impl ContinuousBatcher {
    /// Spawns the worker. Holds `decoder` and a detokenize callback for
    /// the worker's lifetime. Returns the shareable handle; the worker
    /// exits when the last `ContinuousBatcher` clone is dropped.
    /// Spawns the worker with the scheduler knobs passed in rather than
    /// read from the environment, building the ceiling from
    /// `config.max_context` and the decoder's own shape.
    ///
    /// Tests use this: two tests setting `FRINK_CB_*` in one process
    /// would race each other. Production goes through
    /// [`Self::spawn_with_ceiling`], so the ceiling the batched path
    /// enforces is the same *object* the private path enforces rather
    /// than a second copy of the same arithmetic.
    #[cfg(test)]
    pub fn spawn_with_config(
        decoder: Arc<Decoder>,
        decode: DecodeFn,
        config: BatcherConfig,
    ) -> Self {
        Self::spawn_with_config_paged(decoder, decode, config, None)
    }

    /// `spawn_with_config` with a paged store, for the tests that check
    /// batching and paging compose.
    #[cfg(test)]
    pub fn spawn_with_config_paged(
        decoder: Arc<Decoder>,
        decode: DecodeFn,
        config: BatcherConfig,
        paged: Option<PagedKvConfig>,
    ) -> Self {
        let ceiling = Arc::new(ContextCeiling::new(
            config.max_context,
            // Prefill runs one token at a time here, so a sliding layer
            // needs `window + 1 - 1` positions live: chunk = 1.
            frink_models::KvShape::from_config(&decoder.config, frink_models::KvElem::F32),
        ));
        Self::spawn_with_ceiling(decoder, decode, config, ceiling, paged)
    }

    /// `spawn_with_config` with the per-request context ceiling handed
    /// in rather than rebuilt from `config.max_context`.
    ///
    /// This is the production entry point. The point of passing the
    /// `Arc` is that `crate::main` builds exactly one `ContextCeiling`
    /// for the loaded model and gives it to both the batcher and the
    /// private `generate` path: a request refused by one is refused by
    /// the other, priced identically, counted once.
    pub fn spawn_with_ceiling(
        decoder: Arc<Decoder>,
        decode: DecodeFn,
        config: BatcherConfig,
        ceiling: Arc<ContextCeiling>,
        paged: Option<PagedKvConfig>,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let counters = Arc::new(Counters::default());
        let queue = Arc::new(QueueGate::new(config.max_queue));
        let budget = Arc::new(BlockBudget::new(
            config.kv_block_size,
            config.kv_blocks,
            ceiling,
        ));
        let aborts = Arc::new(AbortInbox::default());
        let paged_window = paged
            .as_ref()
            .and_then(|p| paged_window_policy(&decoder, p));
        let worker_counters = Arc::clone(&counters);
        let worker_queue = Arc::clone(&queue);
        let worker_budget = Arc::clone(&budget);
        let worker_aborts = Arc::clone(&aborts);
        let worker_paged = paged.clone();
        let _join = thread::Builder::new()
            .name("frink-continuous-batch".into())
            .spawn(move || {
                worker_loop(
                    decoder,
                    decode,
                    rx,
                    config,
                    worker_counters,
                    worker_queue,
                    worker_budget,
                    worker_aborts,
                    worker_paged,
                )
            })
            .expect("spawn continuous-batch worker");
        // Detach join handle intentionally: dropping the last Sender
        // closes `rx` and ends the loop. Keep a process-lifetime leak
        // of the JoinHandle via Box::leak so dropping batcher clones
        // does not join (callers may still be mid-generate).
        let _guard: &'static WorkerGuard = Box::leak(Box::new(WorkerGuard { _join }));
        ContinuousBatcher {
            tx,
            counters,
            queue,
            budget,
            aborts,
            paged_window,
            config,
        }
    }

    /// Live scheduler counters. Cheap (a handful of relaxed atomic
    /// loads) and safe to call from any thread while the worker runs.
    pub fn stats(&self) -> BatcherStats {
        BatcherStats {
            prefill_chunks: self.counters.prefill_chunks.load(Ordering::Relaxed),
            prefill_tokens: self.counters.prefill_tokens.load(Ordering::Relaxed),
            decode_steps: self.counters.decode_steps.load(Ordering::Relaxed),
            queue_depth: self.queue.depth(),
            queue_rejected: self.queue.rejected(),
            kv_blocks_total: self.budget.total.unwrap_or(0),
            kv_blocks_free: self.budget.free(),
            kv_block_size: self.budget.block_size,
            kv_rejected_too_large: self.budget.rejected_too_large.load(Ordering::Relaxed),
            kv_rejected_context_length: self.budget.ceiling.refused(),
            kv_blocks_peak: self.counters.peak_blocks.load(Ordering::Relaxed),
            aborted: self.aborts.aborted(),
            // `usize::MAX` is `BatcherConfig`'s "unlimited"; a gauge
            // reporting 18446744073709551615 would read as a number
            // somebody chose.
            max_seqs: if self.config.max_seqs == usize::MAX {
                0
            } else {
                self.config.max_seqs
            },
            prefill_chunk: self.config.prefill_chunk,
        }
    }

    /// Submit one generation job and block until it finishes. Safe to
    /// call from many `spawn_blocking` tasks concurrently -- they
    /// serialize only on the shared decode worker, which is the point.
    pub fn generate(
        &self,
        prompt_tokens: Vec<usize>,
        params: GenerationParams,
        stop_tokens: StopTokens,
    ) -> Result<(FinishReason, Vec<usize>, String, Usage), DecodeError> {
        self.generate_streaming(prompt_tokens, params, stop_tokens, None::<fn(&str)>)
    }

    /// Like [`Self::generate`], but invokes `on_chunk` on the worker
    /// thread for each detokenized piece as it becomes visible.
    pub fn generate_streaming(
        &self,
        prompt_tokens: Vec<usize>,
        params: GenerationParams,
        stop_tokens: StopTokens,
        mut on_chunk: Option<impl FnMut(&str)>,
    ) -> Result<(FinishReason, Vec<usize>, String, Usage), DecodeError> {
        // A request that could never fit is refused first, and refused
        // with the ceiling named: queueing it would only make it wait
        // for capacity that will never be enough. This is the
        // immovable half of the rejection split -- 400, not 503.
        //
        // `fit` first, the same call the private loop makes: a prompt
        // that fits with room to spare is SERVED with whatever output
        // budget remains, rather than refused over a `max_tokens` the
        // caller never sent (`ContextCeiling::fit`). Only the prompt's
        // own size and an unrepresentable sum refuse here.
        let mut params = params;
        params.max_tokens = self
            .budget
            .ceiling
            .fit(prompt_tokens.len(), params.max_tokens)?;
        let max_seq_len = prompt_tokens.len().saturating_add(params.max_tokens);
        // What this request will HOLD, which is its whole length unless
        // a sliding window gives the tail back as it goes. The ceiling
        // is still checked against the full length below: a window
        // bounds the memory, not what the request asked for, and a
        // deployment that caps context at 8k means 8k either way.
        let positions = paged_hold_positions(
            max_seq_len,
            prompt_tokens.len(),
            self.budget.block_size,
            self.paged_window.as_ref(),
        );
        let blocks = self.budget.blocks_for(positions);
        if let Some(refusal) = self.budget.immovable_refusal(max_seq_len, positions) {
            return Err(refusal);
        }
        // Refuse before allocating a queue slot for the prompt, so a
        // retry storm costs a rejection rather than memory.
        self.queue
            .try_reserve()
            .map_err(|queued| DecodeError::QueueFull {
                queued,
                cap: self.queue.cap,
            })?;
        let (reply_tx, reply_rx) = mpsc::channel::<BatcherEvent>();
        let abort = self.aborts.next_id();
        let cancel = params.cancel.clone();
        if self
            .tx
            .send(Job {
                prompt_tokens,
                params,
                stop_tokens,
                reply: reply_tx,
                abort,
                blocks,
                skips: 0,
            })
            .is_err()
        {
            // The worker is gone, so nothing will ever dequeue this
            // reservation: release it here or the gate leaks a slot.
            self.queue.release();
            return Err(DecodeError::KvPoolExhausted);
        }
        // Registered after the send, so an id only ever reaches the
        // inbox for a job the worker will actually see. `on_cancel`
        // fires immediately if the token is already cancelled, so the
        // window between the send and this line loses nothing.
        if let Some(token) = cancel {
            let inbox = Arc::clone(&self.aborts);
            token.on_cancel(move || inbox.enqueue(abort));
        }
        loop {
            match reply_rx.recv() {
                Ok(BatcherEvent::Chunk(chunk)) => {
                    if let Some(ref mut f) = on_chunk {
                        f(&chunk);
                    }
                }
                Ok(BatcherEvent::Finished(result)) => return *result,
                Err(_) => return Err(DecodeError::KvPoolExhausted),
            }
        }
    }
}
