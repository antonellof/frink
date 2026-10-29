//! One in-flight request's state, and the keyed table it lives in.
//!
//! Rows are addressed by [`Uid`], never by batch position. Batch
//! membership changes on almost every tick, and a positional table
//! renumbers its survivors when that happens; a `Uid` captured before a
//! removal still names its own row afterwards, or nothing at all. See
//! the keyed-row-state note in [`super`] for the bug class that avoids.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Sender;

use frink_core::cache::KvCache;
use frink_models::tokenizer::StopTokens;

use crate::generate::{DecodeError, FinishReason, GenerationParams, PagedLease};

use super::answer::Answer;
use super::block_budget::BlockBudget;
use super::config::BatcherEvent;
use super::queue::AbortId;

/// Where one batched row keeps its KV.
///
/// A paged row holds a whole [`PagedLease`], not just its caches: the
/// lease is what returns the row's page groups when the row ends, and
/// every way a row can end -- finished, cancelled, evicted, refused at
/// validation -- goes through dropping it. Splitting the caches from
/// the lease would mean one more path that has to remember to free.
pub(super) enum RowKv {
    Contiguous(Vec<KvCache>),
    Paged(PagedLease),
}

impl RowKv {
    /// Positions already written into this row's KV, and therefore the
    /// only position a forward pass over it may run at next.
    ///
    /// This is a READ of the cursor, not a second derivation of it.
    /// `KvCache::push` and `PagedKvCache::push` both write at their own
    /// `seq_len` and IGNORE the position their caller passes alongside,
    /// so the KV's length is the one true answer and everybody else's is
    /// a guess that has to be kept in step.
    ///
    /// A paged row does not start at zero. `acquire_paged_caches` seeds
    /// its lease with whatever prefix the radix tree already held
    /// (`PagedKvCache::adopt_blocks` installs those blocks and sets
    /// `seq_len` to the prefix length), so a prefill that assumed zero
    /// re-ran the whole prompt ON TOP of the adopted prefix: the tokens
    /// landed in rows past it while carrying RoPE positions `0..n`, and
    /// the answer came back wrong with a 200 on it. That was issue #37,
    /// and it existed because the batched prefill kept its own copy of
    /// this number while the private generate loop derived one.
    ///
    /// `&mut self` only because [`PagedLease`] exposes its caches
    /// mutably; nothing here is modified.
    pub(super) fn positions_written(&mut self) -> usize {
        match self {
            RowKv::Contiguous(caches) => caches.first().map_or(0, |c| c.positions()),
            RowKv::Paged(lease) => lease.caches_mut().first().map_or(0, |c| c.seq_len()),
        }
    }
}

pub(super) struct Job {
    pub(super) prompt_tokens: Vec<usize>,
    pub(super) params: GenerationParams,
    pub(super) stop_tokens: StopTokens,
    pub(super) reply: Sender<BatcherEvent>,
    /// Cancellation handle for this job, from submission onwards.
    pub(super) abort: AbortId,
    /// KV blocks this job will need for its whole lifetime, computed
    /// once by the submitter (which already knows the prompt length and
    /// `max_tokens`) so the worker's admission check is a comparison
    /// rather than arithmetic.
    pub(super) blocks: usize,
    /// How many times admission has passed this job over
    /// (`super::cache_aware`). Zero for every job that has never
    /// waited behind another.
    ///
    /// On the JOB rather than in a structure beside the queue, because
    /// the queue is mutated in four places and a parallel deque would
    /// have to agree with all four.
    pub(super) skips: u32,
    /// Admission passes this job has waited through, for the
    /// preemption bound (`super::preempt`). Zero whenever the job is
    /// put back in the queue.
    pub(super) waited: u32,
    /// `Some` for a request that was admitted, produced output, and was
    /// preempted back into the queue: what it had produced, carried
    /// whole so re-admission continues it rather than restarting it.
    /// Its `prompt_tokens` is then the prompt AS RUN (`Slot::prompt_ids`),
    /// and re-admission prefills that followed by the answer's tokens.
    ///
    /// Boxed: a queue of fresh jobs should not pay for a sampler each.
    pub(super) answer: Option<Box<Answer>>,
}

/// Stable identity for one in-flight request, handed out once at
/// admission and never reused. Unlike a batch index it does not move
/// when another row leaves the batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct Uid(u64);

/// The in-flight rows: state keyed by [`Uid`], plus the admission order
/// the batch is built in. Removing a row cannot renumber another one --
/// see the keyed-row-state note in the module docs for what that
/// prevents.
#[derive(Default)]
pub(super) struct Rows {
    pub(super) state: HashMap<Uid, Slot>,
    /// Admission order. Kept explicit so batch composition is
    /// deterministic; `HashMap` iteration order is not.
    pub(super) order: Vec<Uid>,
    pub(super) next_uid: u64,
}

impl Rows {
    pub(super) fn insert(&mut self, slot: Slot) -> Uid {
        let uid = Uid(self.next_uid);
        self.next_uid += 1;
        self.state.insert(uid, slot);
        self.order.push(uid);
        uid
    }

    pub(super) fn len(&self) -> usize {
        self.order.len()
    }

    /// KV blocks the rows in this table are holding right now.
    pub(super) fn blocks_held(&self) -> usize {
        self.state.values().map(|slot| slot.blocks).sum()
    }

    /// Marks any row whose abort id is in `ids` as cancelled, and
    /// reports which ids were consumed.
    ///
    /// Marking, not removing: the row leaves through the same
    /// `flush_finished` path as every other finished row, so its blocks
    /// are released and its caller is replied to exactly once. A second
    /// removal path is a second place to forget one of those.
    pub(super) fn mark_cancelled(&mut self, ids: &HashSet<AbortId>) -> Vec<AbortId> {
        let mut consumed = Vec::new();
        for uid in &self.order {
            if let Some(slot) = self.state.get_mut(uid) {
                if ids.contains(&slot.abort) && slot.finish.is_none() {
                    slot.finish = Some(FinishReason::Cancelled);
                    consumed.push(slot.abort);
                }
            }
        }
        consumed
    }

    pub(super) fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// `None` for a uid that has already left the batch. A stale uid
    /// resolves to nothing -- never to whichever row happens to sit
    /// where it used to.
    pub(super) fn get(&self, uid: Uid) -> Option<&Slot> {
        self.state.get(&uid)
    }

    pub(super) fn get_mut(&mut self, uid: Uid) -> Option<&mut Slot> {
        self.state.get_mut(&uid)
    }

    pub(super) fn remove(&mut self, uid: Uid) -> Option<Slot> {
        self.order.retain(|&u| u != uid);
        self.state.remove(&uid)
    }

    /// Rows that should take a decode step this tick, in admission
    /// order.
    pub(super) fn ready(&self) -> Vec<Uid> {
        self.order
            .iter()
            .copied()
            .filter(|uid| {
                self.state.get(uid).is_some_and(|s| {
                    s.finish.is_none() && s.answer.generated_ids.len() < s.max_tokens
                })
            })
            .collect()
    }

    /// Replies to and removes every row that has finished, returning
    /// each row's KV blocks to the budget as it goes. Release happens
    /// here, on the one path every finished row takes, so a row cannot
    /// leave the table without giving its capacity back.
    pub(super) fn flush_finished(&mut self, budget: &BlockBudget) {
        let finished: Vec<Uid> = self
            .order
            .iter()
            .copied()
            .filter(|uid| self.state.get(uid).is_some_and(|s| s.finish.is_some()))
            .collect();
        for uid in finished {
            if let Some(mut slot) = self.remove(uid) {
                budget.release(slot.blocks);
                // Publish before the lease drops, so the next request
                // with this prefix can adopt it.
                //
                // The batched path used to ADOPT from the radix tree and
                // never contribute to it, so under continuous batching
                // prefix sharing ran against a tree nothing filled: the
                // first request paid full prefill and so did every one
                // after it. The private generate loop published all
                // along; only the batcher did not, because the prompt
                // ids were dropped at the prefill-to-decode handover.
                publish_row(&mut slot.kv, &slot.prompt_ids, &slot.answer.generated_ids);
                reply_finished(slot);
            }
        }
    }
}

pub(super) struct Slot {
    pub(super) kv: RowKv,
    pub(super) pos: usize,
    pub(super) logits: Vec<f32>,
    /// Everything this row has produced that cannot be recomputed from
    /// its tokens: sampler, grammar, stop hold-back, UTF-8 tail, text
    /// already streamed, clock. One value so a preempted row carries it
    /// whole; see [`super::answer`].
    pub(super) answer: Answer,
    /// The prompt this row ran, kept so a finished paged row can
    /// PUBLISH its prefix to the radix tree, and so a preempted row can
    /// be prefilled again. Without it the batched path adopted prefixes
    /// and never contributed one.
    pub(super) prompt_ids: Vec<usize>,
    pub(super) max_tokens: usize,
    pub(super) stop_tokens: StopTokens,
    pub(super) params: GenerationParams,
    pub(super) reply: Sender<BatcherEvent>,
    pub(super) finish: Option<FinishReason>,
    /// Set instead of an answer when this row cannot be served -- today,
    /// only a grammar it cannot continue. Kept beside `finish` rather
    /// than replacing it so a failed row leaves through the SAME
    /// `flush_finished` path as every other ended row: its blocks are
    /// released and its caller replied to exactly once, which a second
    /// removal path is exactly how to get wrong.
    pub(super) error: Option<DecodeError>,
    pub(super) abort: AbortId,
    /// KV blocks this row reserved at admission, returned when it ends.
    pub(super) blocks: usize,
}

/// A row taken apart at the seam preemption cuts along.
///
/// `kv` is the only part that is thrown away; `job` is the request as
/// the queue holds it, carrying the row's [`Answer`]; `finish` and
/// `error` say whether it had already ended.
pub(super) struct SlotParts {
    pub(super) kv: RowKv,
    pub(super) job: Job,
    pub(super) finish: Option<FinishReason>,
    pub(super) error: Option<DecodeError>,
}

impl Slot {
    /// End this row with an error rather than an answer.
    ///
    /// One row's refusal, not the batch's: every other row in flight is
    /// unaffected, which is the whole reason a per-row error exists
    /// instead of a worker that stops.
    pub(super) fn fail(&mut self, error: DecodeError) {
        // `finish` is what `Rows::ready` and `Rows::flush_finished`
        // read, so it is set here too -- a row with an error and no
        // finish would keep taking decode steps forever.
        self.finish = Some(FinishReason::Stop);
        self.error = Some(error);
    }

    /// Splits this row into what is recomputed and what is carried.
    ///
    /// The destructure has no `..` ON PURPOSE. Every field of `Slot` is
    /// named here and either carried into the [`Job`] or dropped with a
    /// reason, so a field added to `Slot` does not compile until
    /// somebody decides whether a preempted row keeps it. A preemption
    /// path that copied row state field by field would lose the new one
    /// silently, which is the bug shape this repo keeps shipping.
    pub(super) fn into_parts(self) -> SlotParts {
        let Slot {
            kv,
            // Both recomputed by the re-prefill: the position is read
            // back off the KV, and the logits are the last prefill
            // token's.
            pos: _,
            logits: _,
            answer,
            prompt_ids,
            // Derived from `params.max_tokens` in `Prefill::into_slot`.
            max_tokens: _,
            stop_tokens,
            params,
            reply,
            finish,
            error,
            abort,
            blocks,
        } = self;
        SlotParts {
            kv,
            job: Job {
                prompt_tokens: prompt_ids,
                params,
                stop_tokens,
                reply,
                abort,
                blocks,
                skips: 0,
                waited: 0,
                answer: Some(Box::new(answer)),
            },
            finish,
            error,
        }
    }
}

/// Publishes a paged row's computed prefix to the radix tree, so the
/// next request sharing it -- or this one, prefilled again after a
/// preemption -- adopts it instead of recomputing.
///
/// One function for both callers. It must run before the lease drops:
/// the lease is what holds the pages until the tree has retained them.
pub(super) fn publish_row(kv: &mut RowKv, prompt_ids: &[usize], generated_ids: &[usize]) {
    if let RowKv::Paged(lease) = kv {
        let mut seq = Vec::with_capacity(prompt_ids.len() + generated_ids.len());
        seq.extend_from_slice(prompt_ids);
        seq.extend_from_slice(generated_ids);
        let bs = lease.block_size();
        crate::generate::publish_to_radix(lease, &seq, bs);
    }
}

/// Sends one finished row's result to its own waiting caller. Takes the
/// `Slot` by value, so a row's reply channel travels with its state and
/// cannot be paired with another row's output.
pub(super) fn reply_finished(slot: Slot) {
    let SlotParts {
        job, finish, error, ..
    } = slot.into_parts();
    let finish = finish.expect("only a finished row is replied to");
    let answer = job
        .answer
        .expect("an admitted row always carries its answer");
    answer.reply(&job.reply, finish, error);
}
