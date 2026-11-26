//! The `priority` scheduling hint (#541).
//!
//! Priority decides WHICH WAITING JOB GETS THE SEAT first (admission),
//! and WHICH RUNNING ROW IS EVICTED to make room (preemption). It is
//! a scheduling hint, never part of the answer: no token, text or
//! timing is decided by it. These tests pin both halves and the
//! defaults that keep the arrival order.

use std::collections::VecDeque;
use std::sync::mpsc;

use super::super::preempt;
use super::super::queue::QueueGate;
use super::super::row::Job;
use super::super::worker::admit;
use super::*;

/// A job with an explicit `priority` (mirrors the radix test's `paged_job`).
fn priority_job(
    prompt: Vec<usize>,
    max_tokens: usize,
    priority: Option<i32>,
) -> (Job, mpsc::Receiver<BatcherEvent>) {
    let (tx, rx) = mpsc::channel();
    let mut params = greedy_params(max_tokens, 7);
    params.priority = priority;
    (
        Job {
            prompt_tokens: prompt,
            params,
            stop_tokens: StopTokens::default(),
            reply: tx,
            abort: AbortId(0),
            blocks: 1,
            skips: 0,
            waited: 0,
            answer: None,
        },
        rx,
    )
}

/// Two jobs in the waiting queue with `max_seqs` sized to take one.
fn priority_admit(waiting: VecDeque<Job>) -> (VecDeque<Prefill>, VecDeque<Job>) {
    let decoder = tiny_decoder();
    let queue = QueueGate::new(512);
    queue.try_reserve().expect("cap 512");
    queue.try_reserve().expect("cap 512");
    let budget = budget(4, None);
    let batcher_config = BatcherConfig {
        max_seqs: 1,
        prefill_chunk: 64,
        ..BatcherConfig::default()
    };
    let mut prefills: VecDeque<Prefill> = VecDeque::new();
    let mut waiting = waiting;
    admit(
        &decoder,
        &mut waiting,
        &mut prefills,
        &mut Rows::default(),
        &batcher_config,
        &queue,
        &budget,
        None,
        &Counters::default(),
    );
    (prefills, waiting)
}

/// **A high-priority request is admitted ahead of a waiting low-priority
/// one**, whatever the order they arrived in.
///
/// The front of the queue is the LOW-priority job; the back carries a
/// radix-less prompt with a higher priority. Both fit the same
/// single seat; only the back job is admitted.
///
/// Driven through `admit` itself, not the policy helper: a test that
/// called the helper directly passed with the wiring sabotaged to
/// always take the front of the queue.
#[test]
fn a_high_priority_job_is_admitted_ahead_of_a_waiting_low_priority_one() {
    let cold = vec![10usize, 11, 12];
    let warm = vec![20usize, 21, 22];
    let (low_job, _rx_low) = priority_job(cold.clone(), 2, Some(1));
    let (high_job, _rx_high) = priority_job(warm.clone(), 2, Some(5));

    let mut waiting: VecDeque<Job> = VecDeque::new();
    waiting.push_back(low_job);
    waiting.push_back(high_job);

    let (prefills, waiting) = priority_admit(waiting);

    assert_eq!(
        prefills.len(),
        1,
        "max_seqs is 1, so exactly one is admitted"
    );
    assert_eq!(
        prefills[0].answer.prompt_tokens,
        warm.len(),
        "the higher-priority job must be admitted first, not the front of the queue"
    );
    assert_eq!(waiting.len(), 1, "the low-priority job is still waiting");
    assert_eq!(
        waiting[0].skips, 1,
        "the job that was jumped must be one skip older, or the bound never bites"
    );
    assert_eq!(
        waiting[0].params.priority,
        Some(1),
        "the low-priority job stayed in the queue, carrying its own params"
    );
}

/// **The default is today's arrival order**: two unprioritised jobs
/// admit in the order they arrived, whatever else is true of them.
///
/// This is the regression guard for the whole grouping: a priority
/// implementation that reordered the queue would have to break this
/// one exactly this way, and it does.
#[test]
fn unprioritised_jobs_admit_in_arrival_order() {
    let first = vec![10usize, 11, 12];
    let second = vec![20usize, 21, 22];
    let (first_job, _rx_first) = priority_job(first.clone(), 2, None);
    let (second_job, _rx_second) = priority_job(second.clone(), 2, None);

    let mut waiting: VecDeque<Job> = VecDeque::new();
    waiting.push_back(first_job);
    waiting.push_back(second_job);

    let (prefills, _waiting) = priority_admit(waiting);

    assert_eq!(
        prefills.len(),
        1,
        "max_seqs is 1, so exactly one is admitted"
    );
    assert_eq!(
        prefills[0].answer.prompt_tokens,
        first.len(),
        "arrival order is the default: the front of the queue goes first"
    );
}

/// **Preemption prefers evicting the row that asked for the least, and
/// never evicts a row that asked for more than the waiting job.**
///
/// Hand-built rows (the helper's contiguous KV and empty logits are
/// enough for `make_room`, which reads only blocks and priority). The
/// numbers are spelled out so a policy that silently ignores priority
/// -- evicting the newest row instead -- fails here.
#[test]
fn make_room_evicts_the_lowest_priority_row_first() {
    let mut rows = Rows::default();
    // Two decoding rows, lowest priority oldest, then newest.
    let (low_row, _rr) = test_slot(4, 1);
    let (high_row, _hr) = test_slot(4, 2);
    let low_uid = rows.insert({
        let mut r = low_row;
        r.params.priority = Some(1);
        r
    });
    let high_uid = rows.insert({
        let mut r = high_row;
        r.params.priority = Some(3);
        r
    });
    let head = priority_head(Some(5));
    let config = BatcherConfig {
        // The head must have waited past the bound.
        preempt_after_ticks: Some(1),
        ..budget_config(4, 2)
    };

    // The budget holds two blocks; the two rows hold one each, so the
    // head (one block) fits only after a victim is evicted.
    let budget = budget(4, Some(2));
    assert!(
        budget.try_reserve(1),
        "one block reserved for the low-priority row"
    );
    assert!(
        budget.try_reserve(1),
        "one block reserved for the high-priority row"
    );
    let victims = preempt::make_room(&mut rows, 0, &head, &budget, &config);

    assert_eq!(victims.len(), 1, "one row makes exactly enough room");
    let evicted = &victims[0];
    assert_eq!(
        evicted.params.priority,
        Some(1),
        "the lowest-priority row must be the one evicted"
    );
    assert_eq!(rows.order, vec![high_uid], "the higher-priority row stays");
    assert!(
        rows.get(low_uid).is_none(),
        "the evicted row left the table"
    );
    assert!(rows.get(high_uid).is_some(), "the untouched row stayed");
}

/// **A request is never served by evicting a row that asked for more
/// than it.** The mirror of the above: a low-priority head and a
/// high-priority row cannot squeeze each other out.
#[test]
fn make_room_never_evicts_a_row_with_more_priority() {
    let mut rows = Rows::default();
    let (row, _r) = test_slot(4, 1);
    let uid = rows.insert({
        let mut r = row;
        r.params.priority = Some(3);
        r
    });
    let head = priority_head(Some(1));
    let config = BatcherConfig {
        preempt_after_ticks: Some(1),
        ..budget_config(4, 1)
    };
    let budget = budget(4, Some(1));
    assert!(budget.try_reserve(1), "one block held by the row");

    let victims = preempt::make_room(&mut rows, 0, &head, &budget, &config);

    assert!(
        victims.is_empty(),
        "the higher-priority row must be left alone"
    );
    assert_eq!(rows.order, vec![uid], "the row stayed in the table");
}

/// The head the preemption tests start from: a fresh job with the
/// requested priority, one block, and past the wait bound.
fn priority_head(priority: Option<i32>) -> Job {
    let (job, _rx) = priority_job(vec![10usize, 11], 2, priority);
    Job {
        // A resumed job carries its answer; a fresh one does not.
        skips: 0,
        waited: 2,
        ..job
    }
}
