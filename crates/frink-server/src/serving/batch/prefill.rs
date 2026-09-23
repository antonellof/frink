//! Chunked prefill: a prompt as a resumable state machine rather than
//! one uninterruptible `forward_token` loop.
//!
//! Chunking is a *scheduling* boundary, not a numerical one. Each chunk
//! runs the same prompt slice at the same positions, so chunk size never
//! changes the final logits or sampled tokens (asserted by
//! `prefill_chunking_does_not_change_logits`).

use std::sync::mpsc::Sender;

use frink_models::tokenizer::StopTokens;
use frink_models::Decoder;
use std::sync::Arc;

use crate::generate::{GenerationParams, PagedLease};

use super::clock::RowClock;
use super::config::BatcherEvent;
use super::queue::AbortId;
use super::row::{RowKv, Slot};
use crate::sample_step::SampleState;
use crate::stop::StopMatcher;

/// One request's prefill as a resumable state machine.
///
/// Holds the KV `caches` being built, how many prompt tokens have been
/// processed, and the logits produced by the most recent token.
/// `step_chunk` advances by at most `chunk_size` tokens and reports
/// whether the prompt is finished, which converts an unbounded prefill
/// into a bounded unit of work the scheduler can interleave with
/// decode.
pub struct PrefillState {
    decoder: Arc<Decoder>,
    kv: RowKv,
    /// The tokens this prefill must run. An *empty* prompt is stored as
    /// a single token 0, matching the private `generate` loop: one
    /// forward pass is still required to produce the logits the first
    /// sampled token comes from.
    tokens: Vec<usize>,
    /// How far through `tokens` this prefill has got. Never assigned a
    /// literal: it is seeded from, and re-checked against,
    /// [`RowKv::positions_written`] -- see [`PrefillState::over`].
    tokens_processed: usize,
    logits: Vec<f32>,
    chunk_size: usize,
}

impl PrefillState {
    pub fn new(decoder: Arc<Decoder>, prompt_tokens: &[usize], chunk_size: usize) -> Self {
        let kv = RowKv::Contiguous(decoder.config.new_kv_caches());
        PrefillState::over(decoder, kv, prompt_tokens, chunk_size)
    }

    /// A prefill whose KV lives in a shared paged store, with its
    /// pages already reserved.
    ///
    /// Reserved for `prompt + max_tokens` at admission for the same
    /// reason the private path does it: the decode step has nowhere to
    /// report a store that ran dry mid-answer.
    ///
    /// The lease may arrive with a radix prefix already installed. That
    /// is not this constructor's business to know: [`Self::over`] reads
    /// the starting position off the KV, so a pre-seeded lease and a
    /// cold one take the same path.
    pub fn new_paged(
        decoder: Arc<Decoder>,
        prompt_tokens: &[usize],
        chunk_size: usize,
        lease: PagedLease,
    ) -> Self {
        PrefillState::over(decoder, RowKv::Paged(lease), prompt_tokens, chunk_size)
    }

    /// The ONE place a prefill's starting position is decided, for
    /// every kind of KV a row can have.
    ///
    /// It is read off the KV rather than assumed, because the KV is the
    /// only thing that knows: `push` writes at its cache's `seq_len`
    /// whatever position the caller names. A second constructor that
    /// wrote `tokens_processed: 0` beside this one is exactly how issue
    /// #37 happened -- a paged lease that had adopted a cached prefix
    /// re-ran the whole prompt on top of it -- so there is no second
    /// constructor, and no literal to get wrong.
    fn over(
        decoder: Arc<Decoder>,
        mut kv: RowKv,
        prompt_tokens: &[usize],
        chunk_size: usize,
    ) -> Self {
        assert!(chunk_size > 0, "prefill chunk size must be positive");
        let tokens = if prompt_tokens.is_empty() {
            vec![0]
        } else {
            prompt_tokens.to_vec()
        };
        let tokens_processed = kv.positions_written();
        // `acquire_paged_caches` backs the adopted prefix off by a page
        // so that at least one token is left to run: a prefill with
        // nothing to run produces no logits, and the first sampled
        // token has nowhere to come from. This is the assertion that
        // holds that back-off and this loop to the same story.
        debug_assert!(
            tokens_processed < tokens.len(),
            "a prefill must have at least one token left to run, \
             got {tokens_processed} already done of {} -- the adopted \
             prefix was not backed off",
            tokens.len()
        );
        PrefillState {
            decoder,
            kv,
            tokens,
            tokens_processed,
            logits: Vec::new(),
            chunk_size,
        }
    }

    /// Prompt positions already in this row's KV. Includes any prefix
    /// the lease adopted from the radix tree: those positions are
    /// computed and resident, they were just computed by an earlier
    /// request.
    pub fn tokens_processed(&self) -> usize {
        self.tokens_processed
    }

    /// Prompt tokens still to run.
    pub fn tokens_remaining(&self) -> usize {
        self.tokens.len() - self.tokens_processed
    }

    pub fn is_done(&self) -> bool {
        self.tokens_remaining() == 0
    }

    /// The ceiling this prompt was admitted under (`prefill_chunk`).
    ///
    /// Kept on the state so the worker does not have to thread the
    /// config through to every prefill it advances.
    pub fn chunk_ceiling(&self) -> usize {
        self.chunk_size
    }

    /// Runs at most `budget` further prompt tokens. Returns `true`
    /// once the whole prompt has been processed. Calling it again after
    /// that is a no-op that still returns `true`.
    ///
    /// The KV position of each token is the row the KV will write it
    /// into, so resuming across chunk boundaries is exactly the
    /// sequential `forward_token` loop it replaces, split at different
    /// points.
    ///
    /// The budget is an ARGUMENT rather than the `chunk_size` field it
    /// used to read, because how much prompt a tick may run depends on
    /// how many rows are decoding in the same tick -- a fact this type
    /// does not have and the worker does. See
    /// [`super::step_budget`].
    pub fn step_chunk(&mut self, budget: usize) -> bool {
        debug_assert!(budget > 0, "a zero prefill budget cannot make progress");
        let end = (self.tokens_processed + budget).min(self.tokens.len());
        if self.tokens_processed >= end {
            return self.is_done();
        }
        // The position comes from the KV every chunk, never from a
        // counter kept alongside it.
        let pos = self.kv.positions_written();
        debug_assert_eq!(
            pos, self.tokens_processed,
            "the KV write cursor and the prompt cursor diverged"
        );
        let chunk = &self.tokens[self.tokens_processed..end];
        self.logits = match &mut self.kv {
            // Batched decode reads host K/V via `forward_multi_seq`. Metal
            // `forward_token` leaves host rows length-advanced but zero-filled;
            // `sync_metal_attn_kv_to_host` cannot repair that once lengths
            // match. Same contract as `forward_prompt_batch(..., host_kv: true)`.
            RowKv::Contiguous(caches) => {
                self.decoder.forward_batch_last_host_kv(chunk, pos, caches)
            }
            RowKv::Paged(lease) => {
                let store = Arc::clone(lease.store());
                self.decoder
                    .forward_batch_last_paged(chunk, pos, lease.caches_mut(), &store)
                    .expect("the row's pages were reserved at admission")
            }
        };
        self.tokens_processed = end;
        self.is_done()
    }

    /// Consumes a finished prefill into the pieces a decode row needs:
    /// KV caches, the logits the first token is sampled from, and the
    /// position the first generated token occupies.
    /// Hands the row over to decode, returning the prompt ids too.
    ///
    /// The ids used to be dropped here, which is why the batched path
    /// could ADOPT a radix prefix and never PUBLISH one: publishing
    /// needs the whole token sequence and the row no longer had it. The
    /// private generate loop kept them and published; the batcher did
    /// not, so prefix sharing under `FRINK_CONTINUOUS_BATCHING=1` was
    /// adopt-only against a tree nothing filled.
    pub(super) fn into_decode_start(self) -> (RowKv, Vec<f32>, usize, Vec<usize>) {
        debug_assert!(self.is_done(), "prefill must finish before decoding");
        (self.kv, self.logits, self.tokens_processed, self.tokens)
    }
}

/// An accepted job whose prompt is still being prefilled: the
/// resumable `PrefillState` plus everything the decode row will need
/// once the prompt is through.
pub(super) struct Prefill {
    pub(super) state: PrefillState,
    /// The *real* prompt length, for `Usage`. Deliberately not
    /// `state.tokens.len()`, which is 1 for an empty prompt.
    pub(super) prompt_tokens: usize,
    pub(super) params: GenerationParams,
    pub(super) stop_tokens: StopTokens,
    pub(super) reply: Sender<BatcherEvent>,
    pub(super) abort: AbortId,
    /// Blocks reserved at admission; carried into the `Slot` so the
    /// reservation survives the prefill-to-decode handover.
    pub(super) blocks: usize,
    /// Started when this row was admitted, and carried into the `Slot`
    /// so one clock spans prefill AND decode. Two clocks handed over at
    /// this seam would be two things that must agree about when prefill
    /// ended.
    pub(super) clock: RowClock,
    /// Prompt positions adopted from the radix tree, captured where
    /// the lease was taken. Carried through to the `Slot` so the one
    /// `Usage` constructor can report it.
    pub(super) cached_tokens: Option<usize>,
}

impl Prefill {
    pub(super) fn into_slot(self) -> Slot {
        let Prefill {
            state,
            prompt_tokens,
            params,
            stop_tokens,
            reply,
            abort,
            blocks,
            mut clock,
            cached_tokens,
        } = self;
        let (kv, logits, pos, prompt_ids) = state.into_decode_start();
        // The handover IS the end of prefill, so it is marked here
        // rather than at the call site: every path from a prompt to a
        // decode row goes through this function.
        clock.prefill_finished();
        Slot {
            kv,
            pos,
            logits,
            sample: SampleState::new(params.seed),
            generated_ids: Vec::with_capacity(params.max_tokens),
            prompt_ids,
            visible: String::new(),
            stops: StopMatcher::new(&params.stop, &params.stop_token_ids),
            prompt_tokens,
            max_tokens: params.max_tokens,
            stop_tokens,
            params,
            reply,
            finish: None,
            error: None,
            abort,
            blocks,
            clock,
            cached_tokens,
            utf8: Default::default(),
        }
    }
}
