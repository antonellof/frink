//! Recompute-style preemption: evict a decoding row to admit a job that
//! has waited too long, and continue the evicted row later from the
//! tokens it had produced.
//!
//! **Why the trigger is a wait bound and not a block fault.** The usual
//! trigger is "a running row needs a new KV block and none is free".
//! That cannot happen here: admission reserves a request's WHOLE
//! lifetime, `prompt + max_tokens`, before it runs (`block_budget`), so
//! a running row never asks for a block. What can happen is a job that
//! does not fit waiting behind rows that hold large reservations for
//! answers that may run for a long time. So the trigger is the job
//! admission would take next having waited `preempt_after_ticks`
//! admission passes (`FRINK_CB_PREEMPT_AFTER_TICKS`, off by default).
//!
//! **What is evicted.** The most recently admitted decoding rows, as
//! few as make the job fit, or none at all if even every eligible row
//! would not. A row still prefilling is never evicted: it has produced
//! nothing a client has seen, and evicting it throws away work without
//! freeing anything a decode row would not.
//!
//! **What survives.** Everything in the row's [`Answer`]: the sampler's
//! RNG position, the grammar parse, the reasoning budget, the stop
//! matcher's withheld tail, the UTF-8 tail, the text already streamed,
//! the clock. The KV, position and logits are dropped and recomputed by
//! prefilling `prompt ++ generated` on re-admission, which the paged
//! path makes cheap by publishing the row's pages to the radix tree on
//! the way out. A client sees one stream with no duplicated and no
//! missing deltas, and at temperature 0 the same tokens.
//!
//! **Where it goes.** The front of the queue, behind the job it made
//! room for. Not in front of it: the evicted row fits in exactly the
//! blocks it just gave up, so placing it first would re-admit it into
//! them and evict nothing.
//!
//! **Why it terminates.** A row is preempted at most
//! [`MAX_PREEMPTIONS`] times. Past that it runs to completion, and with
//! no preemption left the queue is the strict FIFO that already cannot
//! starve. Without the bound two rows that cannot both fit evict each
//! other in turn, each paying a re-prefill per swap, and only make
//! progress by however many tokens fit between swaps.
//!
//! **What is refused.** A row on a paged lease with a sliding window.
//! Its KV is shaped by when each slide ran and by the anchor state on
//! the lease, and a recompute from tokens reproduces neither, so it
//! would continue from a different state than it left. It is not
//! evicted; the job waits, as it did before preemption existed.

use super::answer::Answer;
use super::block_budget::BlockBudget;
use super::config::BatcherConfig;
use super::row::{publish_row, Job, RowKv, Rows, Slot, SlotParts};

/// How many times one request may be preempted. The termination bound;
/// see the module note.
pub(super) const MAX_PREEMPTIONS: u32 = 4;

impl Slot {
    /// Whether this row may be evicted at all.
    fn preemptible(&self) -> bool {
        self.finish.is_none()
            && self.error.is_none()
            && self.answer.preemptions < MAX_PREEMPTIONS
            && !matches!(&self.kv, RowKv::Paged(lease) if lease.slides())
    }

    /// Evicts this row: gives its blocks back, publishes its pages for
    /// its own re-admission to adopt, drops its KV, and returns the job
    /// that continues it.
    pub(super) fn preempt(self, budget: &BlockBudget) -> Job {
        // Every token this row generated has been through a forward, so
        // its KV covers exactly `prompt ++ generated`. That is what lets
        // the re-prefill end on the logits this row would have sampled
        // from next.
        debug_assert_eq!(
            self.pos,
            self.prompt_ids.len() + self.answer.generated_ids.len(),
            "a preempted row's KV must cover its prompt and every token it generated"
        );
        let SlotParts {
            mut kv,
            mut job,
            finish,
            error,
        } = self.into_parts();
        debug_assert!(
            finish.is_none() && error.is_none(),
            "an ended row leaves through flush_finished, not preemption"
        );
        budget.release(job.blocks);
        let answer: &mut Answer = job
            .answer
            .as_mut()
            .expect("an admitted row always carries its answer");
        publish_row(&mut kv, &job.prompt_tokens, &answer.generated_ids);
        // The lease returns its pages here, after the tree has retained
        // the ones it kept.
        drop(kv);
        answer.preemptions += 1;
        job
    }
}

/// Evicts the most recently admitted decoding rows until `head` fits
/// under both caps, and returns the jobs that continue them, newest
/// first. Evicts nothing, and returns an empty list, when preemption is
/// off, when `head` has not waited long enough, or when every eligible
/// row together would still not make room.
///
/// `prefilling` is the number of prompts in flight that are not in
/// `rows`: they hold a sequence slot and cannot be evicted.
pub(super) fn make_room(
    rows: &mut Rows,
    prefilling: usize,
    head: &Job,
    budget: &BlockBudget,
    config: &BatcherConfig,
) -> Vec<Job> {
    let Some(after) = config.preempt_after_ticks else {
        return Vec::new();
    };
    if head.waited < after {
        return Vec::new();
    }
    let fits = |free: usize, seats: usize| {
        seats < config.max_seqs && (budget.total.is_none() || head.blocks <= free)
    };
    let mut free = budget.free();
    let mut seats = rows.len() + prefilling;
    let mut victims = Vec::new();
    for &uid in rows.order.iter().rev() {
        if fits(free, seats) {
            break;
        }
        let Some(slot) = rows.get(uid) else {
            continue;
        };
        if !slot.preemptible() {
            continue;
        }
        victims.push(uid);
        free += slot.blocks;
        seats -= 1;
    }
    // All or nothing: evicting rows that still do not make room would
    // cost each of them a re-prefill and admit nobody.
    if victims.is_empty() || !fits(free, seats) {
        return Vec::new();
    }
    victims
        .into_iter()
        .map(|uid| {
            rows.remove(uid)
                .expect("a victim was read from the table above")
                .preempt(budget)
        })
        .collect()
}
