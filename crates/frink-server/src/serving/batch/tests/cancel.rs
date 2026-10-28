use super::*;

// ----------------------------------------------------------------
// Deferred abort (`sched-deferred-abort`)
// ----------------------------------------------------------------

/// The end-to-end fact the item exists for, and the gap the
/// `cancel` module used to state as unfixable: a request decoding
/// on the shared batcher thread stops when the client cancels it.
///
/// Confirmed to FAIL (finishes with `Length` after all 4000 tokens)
/// when the `apply_aborts` call is removed from the worker loop.
#[test]
fn a_decoding_request_stops_when_it_is_cancelled() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    let (params, token) = cancellable_params(4000, 9);
    let worker = {
        let batcher = batcher.clone();
        thread::spawn(move || batcher.generate(vec![1, 2, 3], params, StopTokens::from_eos(None)))
    };

    // Wait until it is genuinely decoding, so the cancel exercises
    // the decode path rather than racing the prefill.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while batcher.stats().decode_steps < 3 {
        assert!(
            std::time::Instant::now() < deadline,
            "never started decoding"
        );
        thread::sleep(std::time::Duration::from_millis(1));
    }
    token.cancel();

    let (finish, ids, _text, usage) = worker.join().expect("no panic").expect("generate");
    assert_eq!(finish, FinishReason::Cancelled);
    assert!(
        ids.len() < 4000,
        "cancelling did not shorten the decode: {} tokens",
        ids.len()
    );
    assert!(
        !ids.is_empty(),
        "the tokens produced before the cancel must survive it"
    );
    assert_eq!(usage.completion_tokens, ids.len());
    assert_eq!(batcher.stats().aborted, 1);
}

/// A cancelled request must leave the batch through the same exit
/// every finished row uses -- marked at the step boundary, removed
/// by `flush_finished` -- so its blocks are released once and its
/// caller is replied to once.
///
/// Removing the row inside `apply_aborts` instead would be a second
/// exit path, and the reply this asserts arrives exactly once would
/// arrive twice.
#[test]
fn a_cancelled_row_leaves_through_the_one_exit_every_row_uses() {
    let mut rows = Rows::default();
    let budget = budget(4, Some(4));
    assert!(budget.try_reserve(2));

    let (mut a, ra) = test_slot(9, 11);
    a.abort = AbortId(7);
    a.blocks = 2;
    a.answer.generated_ids.push(3);
    a.answer.visible.push_str("hi");
    let (b, rb) = test_slot(9, 22);
    let a_uid = rows.insert(a);
    let b_uid = rows.insert(b);

    let consumed = rows.mark_cancelled(&HashSet::from([AbortId(7)]));
    assert_eq!(consumed, vec![AbortId(7)]);
    assert!(
        rows.get(a_uid).is_some(),
        "the row must still be in the table until the flush: its KV \
             buffers are what the batch is built from"
    );
    assert_eq!(
        budget.free(),
        2,
        "marking must not release blocks -- the flush does"
    );
    assert!(ra.try_recv().is_err(), "no reply until the row leaves");

    rows.flush_finished(&budget);
    assert!(rows.get(a_uid).is_none());
    assert!(rows.get(b_uid).is_some(), "only the cancelled row left");
    assert_eq!(budget.free(), 4, "blocks came back exactly once");

    let (finish, ids, text, _usage) =
        finished_result(ra.try_recv().expect("one reply")).expect("ok");
    assert_eq!(finish, FinishReason::Cancelled);
    assert_eq!(ids, vec![3], "partial output survives the cancel");
    assert_eq!(text, "hi");
    assert!(
        ra.try_recv().is_err(),
        "a cancelled row must be replied to exactly once"
    );
    assert!(rb.try_recv().is_err(), "the other row is untouched");
}

/// A cancel can beat its own job through the channel. Dropping the
/// id because nothing matches it yet would leave the request
/// running after the client asked it to stop.
///
/// Confirmed to FAIL when `apply_aborts` drops unmatched ids
/// instead of carrying them.
#[test]
fn a_cancel_that_arrives_before_its_job_is_not_lost() {
    let config = budget_config(4, 4);
    let inbox = AbortInbox::default();
    let queue = QueueGate::new(config.max_queue);
    let budget = BlockBudget::new(
        config.kv_block_size,
        config.kv_blocks,
        Arc::new(ContextCeiling::new(config.max_context, test_shape())),
    );
    let mut carried = HashSet::new();
    let mut waiting = VecDeque::new();
    let mut prefills = VecDeque::new();
    let mut rows = Rows::default();

    // The cancel lands first, naming nothing.
    inbox.enqueue(AbortId(42));
    apply_aborts(
        &inbox,
        &mut carried,
        &mut waiting,
        &mut prefills,
        &mut rows,
        &queue,
        &budget,
    );
    assert_eq!(inbox.aborted(), 0, "nothing to stop yet");

    // Now the job it names shows up.
    let (job, rx) = abortable_job(AbortId(42), vec![1, 2]);
    waiting.push_back(job);
    queue.try_reserve().expect("cap");
    apply_aborts(
        &inbox,
        &mut carried,
        &mut waiting,
        &mut prefills,
        &mut rows,
        &queue,
        &budget,
    );

    assert!(waiting.is_empty(), "the late job must still be cancelled");
    assert_eq!(inbox.aborted(), 1);
    assert_eq!(queue.depth(), 0, "its queue slot came back");
    let (finish, ids, _, usage) = finished_result(rx.try_recv().expect("reply")).expect("ok");
    assert_eq!(finish, FinishReason::Cancelled);
    assert!(ids.is_empty(), "it never ran a token");
    assert_eq!(usage.prompt_tokens, 2);
}

/// A cancelled prompt must not be prefilled: that is the cheapest
/// possible moment to stop, and the whole point of checking before
/// the chunk rather than after it.
#[test]
fn a_cancelled_prefill_is_abandoned_and_gives_its_blocks_back() {
    let decoder = tiny_decoder();
    let config = budget_config(4, 4);
    let inbox = AbortInbox::default();
    let queue = QueueGate::new(config.max_queue);
    let budget = BlockBudget::new(
        config.kv_block_size,
        config.kv_blocks,
        Arc::new(ContextCeiling::new(config.max_context, test_shape())),
    );
    let mut carried = HashSet::new();
    let mut waiting = VecDeque::new();
    let mut prefills = VecDeque::new();
    let mut rows = Rows::default();

    let (job, rx) = abortable_job(AbortId(5), vec![1, 2, 3, 4]);
    waiting.push_back(job);
    queue.try_reserve().expect("cap");
    admit(
        &decoder,
        &mut waiting,
        &mut prefills,
        &mut Rows::default(),
        &config,
        &queue,
        &budget,
        None,
        &Counters::default(),
    );
    assert_eq!(prefills.len(), 1);
    assert_eq!(budget.free(), 3, "the prefill holds its reservation");

    inbox.enqueue(AbortId(5));
    apply_aborts(
        &inbox,
        &mut carried,
        &mut waiting,
        &mut prefills,
        &mut rows,
        &queue,
        &budget,
    );
    assert!(prefills.is_empty(), "the remaining chunks never run");
    assert_eq!(budget.free(), 4, "an abandoned prefill releases its blocks");
    assert_eq!(inbox.aborted(), 1);
    assert_eq!(
        finished_result(rx.try_recv().expect("reply"))
            .expect("ok")
            .0,
        FinishReason::Cancelled
    );
}

/// Cancelling one request must not disturb any other -- the failure
/// mode a positional row table would make easy.
#[test]
fn cancelling_one_request_leaves_its_neighbours_running() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    let expected = sequential_ids(&decoder, &[4, 5], &greedy_params(6, 3));

    let (doomed_params, token) = cancellable_params(4000, 9);
    let doomed = {
        let batcher = batcher.clone();
        thread::spawn(move || {
            batcher.generate(vec![1, 2, 3], doomed_params, StopTokens::from_eos(None))
        })
    };
    let survivor = {
        let batcher = batcher.clone();
        thread::spawn(move || {
            batcher.generate(vec![4, 5], greedy_params(6, 3), StopTokens::from_eos(None))
        })
    };

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while batcher.stats().decode_steps < 3 {
        assert!(
            std::time::Instant::now() < deadline,
            "never started decoding"
        );
        thread::sleep(std::time::Duration::from_millis(1));
    }
    token.cancel();

    let (finish, _, _, _) = doomed.join().expect("no panic").expect("generate");
    assert_eq!(finish, FinishReason::Cancelled);
    let (finish, ids, _, _) = survivor.join().expect("no panic").expect("generate");
    assert_eq!(finish, FinishReason::Length);
    assert_eq!(
        ids, expected,
        "an uncancelled request must produce exactly what it would have alone"
    );
}

// ----------------------------------------------------------------
// Stop sequences (`sched-stop-buffering`)
// ----------------------------------------------------------------

/// Layer 1 in the batched path. A batched row and a row decoding on
/// its own must agree about where an answer ends, so both go
/// through the same `StopMatcher`.
///
/// Confirmed to FAIL (runs to all 8 tokens) when the
/// `stops.is_stop_token` check is removed from the worker's
/// sampling step.
#[test]
fn a_token_level_stop_ends_a_batched_row() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    let baseline = sequential_ids(&decoder, &[1, 2, 3], &greedy_params(8, 4));
    assert!(baseline.len() > 1, "need something to stop before the end");
    let stop_token = baseline[1];

    let (finish, ids, _text, usage) = batcher
        .generate(
            vec![1, 2, 3],
            GenerationParams {
                stop_token_ids: vec![stop_token],
                ..greedy_params(8, 4)
            },
            StopTokens::from_eos(None),
        )
        .expect("generate");

    assert_eq!(finish, FinishReason::Stop);
    assert_eq!(
        ids,
        baseline[..1].to_vec(),
        "the stop token itself is not part of the answer"
    );
    assert_eq!(usage.completion_tokens, 1);
}

/// Layer 2 in the batched path: a stop string spanning several
/// tokens is matched, and nothing past it is reported.
#[test]
fn a_multi_token_stop_string_truncates_a_batched_row() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    let baseline = sequential_ids(&decoder, &[1, 2, 3], &greedy_params(8, 4));
    let full = identity_decode_text(&baseline);
    if full.chars().count() < 4 {
        return;
    }
    // Two characters out of the middle, so the match spans a token
    // boundary and the first character is a partial for a while.
    let cut: Vec<char> = full.chars().collect();
    let stop_str: String = cut[1..3].iter().collect();
    let expected: String = cut[..1].iter().collect();

    let (finish, _ids, text, _usage) = batcher
        .generate(
            vec![1, 2, 3],
            GenerationParams {
                stop: vec![stop_str.clone()],
                ..greedy_params(8, 4)
            },
            StopTokens::from_eos(None),
        )
        .expect("generate");

    assert_eq!(finish, FinishReason::StopSequence(stop_str.clone()));
    assert_eq!(
        text, expected,
        "a stop spanning two tokens must cut where it starts"
    );
    assert!(!text.contains(&stop_str));
}

/// Text withheld against a stop that never arrives is ordinary
/// output. Dropping it would truncate every answer whose tail looks
/// like the start of a stop string.
#[test]
fn a_batched_row_that_never_matches_loses_no_output() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    let baseline = sequential_ids(&decoder, &[1, 2, 3], &greedy_params(8, 4));
    let expected = identity_decode_text(&baseline);

    let (finish, ids, text, _usage) = batcher
        .generate(
            vec![1, 2, 3],
            GenerationParams {
                stop: vec!["ZZ_NEVER_MATCHES_ZZ".to_string()],
                ..greedy_params(8, 4)
            },
            StopTokens::from_eos(None),
        )
        .expect("generate");
    assert_eq!(finish, FinishReason::Length);
    assert_eq!(ids, baseline);
    assert_eq!(
        text, expected,
        "buffering is about when text is released, never whether"
    );
}

/// A request nobody cancels must be untouched by the machinery that
/// exists for the ones that are.
#[test]
fn an_uncancelled_request_is_unaffected_by_the_abort_path() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    let (params, _token) = cancellable_params(6, 3);
    let (finish, ids, _, _) = batcher
        .generate(vec![4, 5], params, StopTokens::from_eos(None))
        .expect("generate");
    assert_eq!(finish, FinishReason::Length);
    assert_eq!(ids, sequential_ids(&decoder, &[4, 5], &greedy_params(6, 3)));
    assert_eq!(batcher.stats().aborted, 0);
}
