//! Preemption (`super::super::preempt`): a decoding row evicted to admit
//! a job that has waited too long, and continued from what it had
//! produced.
//!
//! Each test runs two requests on a server that cannot hold both, with
//! the worker held inside the first request's first token until the
//! second is queued, so the first is DECODING when the second arrives. Every test also
//! asserts that a preemption actually happened: a run that never
//! evicted anything would pass every output check trivially.

use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;

use super::super::counters::BatcherStats;
use super::super::preempt::MAX_PREEMPTIONS;
use super::*;

/// One finished request as its client saw it.
struct Seen {
    finish: FinishReason,
    ids: Vec<usize>,
    text: String,
    /// Every streamed delta, in order.
    deltas: Vec<String>,
    cached_tokens: Option<usize>,
}

/// Long enough that a deadlock is not mistaken for a slow debug build,
/// short enough that one fails the run rather than hanging it.
const DEADLINE: Duration = Duration::from_secs(60);

/// Holds the batcher's worker inside its first token-to-text call.
///
/// The decode function runs on the worker thread, so while it is held
/// no row advances. Releasing it only once the second request is queued
/// makes "the first row is decoding when the second arrives" a fact
/// rather than a race: on a tiny model the first request's whole
/// output took microseconds, and a second thread woken by its first
/// streamed delta could arrive after it finished, so nothing was
/// preempted.
#[derive(Clone, Default)]
struct Gate(Arc<(std::sync::Mutex<GateState>, std::sync::Condvar)>);

#[derive(Default)]
struct GateState {
    entered: bool,
    open: bool,
}

impl Gate {
    /// `identity_decode`, except that its first call waits for [`Gate::open`].
    fn decode(&self) -> DecodeFn {
        let gate = self.clone();
        let inner = identity_decode();
        Arc::new(move |ids: &[usize]| {
            let (lock, cv) = &*gate.0;
            let mut st = lock.lock().unwrap();
            st.entered = true;
            cv.notify_all();
            while !st.open {
                st = cv.wait(st).unwrap();
            }
            drop(st);
            inner(ids)
        })
    }

    fn wait_entered(&self) {
        let (lock, cv) = &*self.0;
        let st = lock.lock().unwrap();
        let (st, timeout) = cv
            .wait_timeout_while(st, DEADLINE, |st| !st.entered)
            .unwrap();
        assert!(
            !timeout.timed_out() && st.entered,
            "the first request never produced a token"
        );
    }

    fn open(&self) {
        let (lock, cv) = &*self.0;
        lock.lock().unwrap().open = true;
        cv.notify_all();
    }
}

/// Runs `first`, holds the worker at its first token (`gate`, which
/// must be the one `batcher` was spawned with), queues `second`, then
/// lets both run, and returns both as their clients saw them.
///
/// Fails the test, rather than hanging it, if either does not finish
/// within [`DEADLINE`]: a preemption policy that livelocks or leaks a
/// reservation shows up here as a timeout.
fn run_overlapping(
    batcher: &ContinuousBatcher,
    gate: &Gate,
    first: (Vec<usize>, GenerationParams),
    second: (Vec<usize>, GenerationParams),
) -> [Seen; 2] {
    let (done_tx, done_rx) = mpsc::channel::<(usize, Result<Seen, DecodeError>)>();
    let spawn = |i: usize, (prompt, params): (Vec<usize>, GenerationParams)| {
        let batcher = batcher.clone();
        let done = done_tx.clone();
        thread::spawn(move || {
            let mut deltas = Vec::new();
            let out = batcher
                .generate_streaming(
                    prompt,
                    params,
                    StopTokens::default(),
                    Some(|chunk: &str| deltas.push(chunk.to_string())),
                )
                .map(|(finish, ids, text, usage)| Seen {
                    finish,
                    ids,
                    text,
                    deltas,
                    cached_tokens: usage.cached_tokens,
                });
            let _ = done.send((i, out));
        })
    };
    let a = spawn(0, first);
    gate.wait_entered();
    let b = spawn(1, second);
    // The worker frees a queue slot only when it admits the job, so a
    // depth of one is the second request waiting. The slot is taken
    // just before the job is sent; the pause covers that last step.
    let deadline = std::time::Instant::now() + DEADLINE;
    while batcher.stats().queue_depth == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the second request never queued"
        );
        thread::yield_now();
    }
    thread::sleep(Duration::from_millis(20));
    gate.open();
    let mut seen: [Option<Seen>; 2] = [None, None];
    for _ in 0..2 {
        match done_rx.recv_timeout(DEADLINE) {
            Ok((i, out)) => seen[i] = Some(out.expect("a preempted request must still be served")),
            Err(RecvTimeoutError::Timeout) => {
                panic!("two requests that cannot both fit did not both finish: deadlock")
            }
            Err(RecvTimeoutError::Disconnected) => panic!("a request thread died"),
        }
    }
    a.join().unwrap();
    b.join().unwrap();
    let [x, y] = seen;
    [x.expect("first replied"), y.expect("second replied")]
}

/// A server whose block budget holds exactly one of these requests.
///
/// Each request is `prompt + max_tokens` positions, so with a block of
/// 4 a 3-token prompt and 21 tokens of output is 6 blocks; a budget of
/// 6 admits one and makes the other wait, which is the pressure
/// preemption exists for.
fn one_at_a_time(preempt_after_ticks: Option<u32>) -> BatcherConfig {
    BatcherConfig {
        preempt_after_ticks,
        ..budget_config(4, 6)
    }
}

fn assert_preempted(stats: &BatcherStats) {
    assert!(
        stats.preempted > 0,
        "no row was preempted, so this run proves nothing about preemption"
    );
}

/// **A preempted row produces exactly the tokens it would have produced
/// unpreempted.**
///
/// The re-prefill has to land the KV where eviction found it: the
/// prompt AS RUN followed by every generated token, at the same
/// positions. Re-prefilling the prompt alone, or the prompt plus all
/// but the last token, leaves the row continuing from a different
/// state, and it goes on producing fluent tokens that are not these.
///
/// Both rows are checked against the private sequential decode, which
/// is the reference `continuous_batch_matches_sequential_generate_token_ids`
/// already holds the unpreempted batcher to.
///
/// Confirmed to FAIL when `accept` re-prefills only the prompt (the
/// generated tokens dropped from `run`).
#[test]
fn a_preempted_row_produces_the_tokens_an_unpreempted_one_does() {
    let decoder = tiny_decoder();
    let jobs = [(vec![1usize, 2, 3], 21usize, 5u64), (vec![7, 8, 9], 21, 9)];
    let reference: Vec<Vec<usize>> = jobs
        .iter()
        .map(|(p, n, seed)| sequential_ids(&decoder, p, &greedy_params(*n, *seed)))
        .collect();

    let gate = Gate::default();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        gate.decode(),
        one_at_a_time(Some(0)),
    );
    let [a, b] = run_overlapping(
        &batcher,
        &gate,
        (jobs[0].0.clone(), greedy_params(jobs[0].1, jobs[0].2)),
        (jobs[1].0.clone(), greedy_params(jobs[1].1, jobs[1].2)),
    );
    assert_preempted(&batcher.stats());
    assert_eq!(
        a.ids, reference[0],
        "the first row was changed by preemption"
    );
    assert_eq!(
        b.ids, reference[1],
        "the second row was changed by preemption"
    );
    assert_eq!(a.finish, FinishReason::Length);
    assert_eq!(b.finish, FinishReason::Length);
}

/// **A seeded sampler continues from where it was, not from its seed.**
///
/// At temperature above zero the sampled token depends on the RNG's
/// position. A preempted row that rebuilt its sampler from the seed
/// would replay the first draws against later logits: the request
/// still answers, and it is no longer the answer that seed gives.
///
/// The reference is the same two requests on the same budget with
/// preemption OFF, so the only thing that differs is the eviction.
#[test]
fn a_preempted_seeded_row_keeps_its_rng_position() {
    let decoder = tiny_decoder();
    let sampled = |seed: u64| {
        let mut p = greedy_params(21, seed);
        p.sampling.temperature = 1.5;
        p
    };
    let jobs = || {
        (
            (vec![1usize, 2, 3], sampled(31)),
            (vec![7usize, 8, 9], sampled(32)),
        )
    };

    let gate = Gate::default();
    let plain = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        gate.decode(),
        one_at_a_time(None),
    );
    let (first, second) = jobs();
    let [want_a, want_b] = run_overlapping(&plain, &gate, first, second);
    assert_eq!(plain.stats().preempted, 0, "preemption is off by default");

    let gate = Gate::default();
    let preempting = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        gate.decode(),
        one_at_a_time(Some(0)),
    );
    let (first, second) = jobs();
    let [a, b] = run_overlapping(&preempting, &gate, first, second);
    assert_preempted(&preempting.stats());
    assert_eq!(a.ids, want_a.ids, "the first row's draws moved");
    assert_eq!(b.ids, want_b.ids, "the second row's draws moved");
}

/// **A client of a preempted row sees one stream: no delta twice, none
/// missing, and the withheld tail still withheld.**
///
/// What a streaming client has is the deltas, not the final text, so
/// the deltas are what is checked: they concatenate to the reply's text,
/// and that text is the unpreempted one.
///
/// The stop string is chosen so the matcher is WITHHOLDING text when
/// the row is evicted: it starts with a letter the answer contains and
/// ends with one it never produces, so every occurrence of that letter
/// is held back until the next token proves it is not a stop. A
/// preemption that rebuilt the stop matcher would drop the held letter,
/// and one that re-sent `visible` would duplicate everything before it.
///
/// Confirmed to FAIL when `Slot::preempt` replaces the answer's
/// `StopMatcher` with a fresh one.
#[test]
fn a_preempted_rows_streamed_deltas_concatenate_to_the_unpreempted_text() {
    let decoder = tiny_decoder();
    let prompts = [vec![1usize, 2, 3], vec![7usize, 8, 9]];
    let reference: Vec<Vec<usize>> = prompts
        .iter()
        .map(|p| sequential_ids(&decoder, p, &greedy_params(21, 3)))
        .collect();
    // `identity_decode` only ever produces A-Z, so `#` never arrives
    // and the stop never matches. Every letter of the answer is a
    // candidate; the most frequent one is held most often.
    let held = |ids: &[usize]| {
        let text = identity_decode_text(ids);
        let mut counts = [0usize; 26];
        for c in text.bytes() {
            counts[(c - b'A') as usize] += 1;
        }
        let most = (0..26).max_by_key(|&i| counts[i]).expect("26 letters");
        format!("{}#", (b'A' + most as u8) as char)
    };
    let params = |i: usize| {
        let mut p = greedy_params(21, 3);
        p.stop = vec![held(&reference[i])];
        p
    };

    let gate = Gate::default();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        gate.decode(),
        one_at_a_time(Some(0)),
    );
    let [a, b] = run_overlapping(
        &batcher,
        &gate,
        (prompts[0].clone(), params(0)),
        (prompts[1].clone(), params(1)),
    );
    assert_preempted(&batcher.stats());
    for (i, seen) in [a, b].iter().enumerate() {
        let want = identity_decode_text(&reference[i]);
        assert_eq!(seen.ids, reference[i], "row {i}: tokens changed");
        assert_eq!(
            seen.deltas.concat(),
            seen.text,
            "row {i}: the streamed deltas are not the answer's text"
        );
        assert_eq!(
            seen.text, want,
            "row {i}: the text differs from the unpreempted answer"
        );
    }
}

/// **Two rows that cannot both fit both finish, and the swapping
/// between them is bounded.**
///
/// The budget holds one of them. With the bound at zero each one
/// evicts the other as soon as it is waiting and the other is decoding,
/// which is the worst case: what ends it is [`MAX_PREEMPTIONS`], after
/// which a row runs to completion and the queue is plain FIFO. A
/// preemption that kept its reservation would leave the evicted row
/// waiting for blocks nobody holds, and the test times out instead of
/// passing.
///
/// Run against both caps, because they are separate reasons for a job
/// not to fit and `make_room` has to relieve whichever one binds.
///
/// Confirmed to FAIL (timeout) when `Slot::preempt` does not release
/// its blocks, and (count) when `MAX_PREEMPTIONS` is not checked.
#[test]
fn two_rows_that_cannot_both_fit_both_finish_with_bounded_preemption() {
    let decoder = tiny_decoder();
    let seat_capped = BatcherConfig {
        prefill_chunk: 1,
        max_seqs: 1,
        preempt_after_ticks: Some(0),
        ..BatcherConfig::default()
    };
    for (label, config) in [
        ("block budget", one_at_a_time(Some(0))),
        ("sequence cap", seat_capped),
    ] {
        let gate = Gate::default();
        let batcher =
            ContinuousBatcher::spawn_with_config(Arc::clone(&decoder), gate.decode(), config);
        let [a, b] = run_overlapping(
            &batcher,
            &gate,
            (vec![1, 2, 3], greedy_params(21, 1)),
            (vec![4, 5, 6], greedy_params(21, 2)),
        );
        let stats = batcher.stats();
        assert_preempted(&stats);
        assert!(
            stats.preempted <= 2 * u64::from(MAX_PREEMPTIONS),
            "{label}: {} preemptions for two rows, past the bound of {MAX_PREEMPTIONS} each",
            stats.preempted
        );
        assert_eq!(a.ids.len(), 21, "{label}: the first row did not finish");
        assert_eq!(b.ids.len(), 21, "{label}: the second row did not finish");
        assert_eq!(
            stats.queue_depth, 0,
            "{label}: the queue gate leaked a slot"
        );
        if config.kv_blocks.is_some() {
            assert_eq!(
                stats.kv_blocks_free, 6,
                "{label}: a preempted row's blocks were not given back"
            );
        }
    }
}

/// **On a paged store with a prefix tree, a preempted row comes back
/// with the unpreempted tokens and reports the reuse it was first
/// served with.**
///
/// The eviction publishes the row's prompt and generated tokens to the
/// tree before its lease drops, and re-admission may adopt whatever of
/// that the tree still holds. Adopted pages at the wrong positions, or
/// another row's, would still produce fluent tokens; the reference
/// check is what catches it.
///
/// `cached_tokens` is the first admission's figure. Read off the
/// re-admitted lease it would be this request's own publish counted as
/// reuse, which is the exact mistake the field's doc already warns of.
#[test]
fn a_preempted_paged_row_resumes_through_its_own_prefix() {
    let decoder = tiny_decoder();
    let radix = Arc::new(Mutex::new(crate::policy::radix::SaltedRadix::new(4)));
    let paged = PagedKvConfig {
        store: Arc::new(frink_core::cache::SharedPagedKv::new(
            decoder.layers.len(),
            /* block_size = */ 4,
            /* blocks_per_layer = */ 64,
            decoder.config.n_kv_heads,
            decoder.config.head_dim,
        )),
        queue_wait: Duration::ZERO,
        radix: Some(Arc::clone(&radix)),
        anchor_token: None,
        slide_interval: crate::policy::pool_budget::DEFAULT_SWA_EVICTION_INTERVAL,
    };
    let prompts = [vec![1usize, 2, 3], vec![7usize, 8, 9]];
    let reference: Vec<Vec<usize>> = prompts
        .iter()
        .map(|p| sequential_ids(&decoder, p, &greedy_params(21, 6)))
        .collect();

    let gate = Gate::default();
    let batcher = ContinuousBatcher::spawn_with_config_paged(
        Arc::clone(&decoder),
        gate.decode(),
        one_at_a_time(Some(0)),
        Some(paged),
    );
    let [a, b] = run_overlapping(
        &batcher,
        &gate,
        (prompts[0].clone(), greedy_params(21, 6)),
        (prompts[1].clone(), greedy_params(21, 6)),
    );
    assert_preempted(&batcher.stats());
    assert_eq!(a.ids, reference[0], "the first paged row changed");
    assert_eq!(b.ids, reference[1], "the second paged row changed");
    assert_eq!(
        a.cached_tokens,
        Some(0),
        "a cold request reports no reuse, whatever it adopted from itself later"
    );
    assert_eq!(b.cached_tokens, Some(0));
}
