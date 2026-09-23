use super::*;

// ----------------------------------------------------------------
// Block admission (`sched-block-admission`)
// ----------------------------------------------------------------

#[test]
fn blocks_are_counted_in_positions_and_always_round_up() {
    let budget = budget(4, Some(10));
    // A single-token request still holds a block.
    assert_eq!(budget.blocks_for(0), 1);
    assert_eq!(budget.blocks_for(1), 1);
    assert_eq!(budget.blocks_for(4), 1);
    // Rounding up, not down: 5 positions do not fit in one block.
    assert_eq!(budget.blocks_for(5), 2);
    assert_eq!(budget.blocks_for(8), 2);
    assert_eq!(budget.blocks_for(9), 3);
}

#[test]
fn an_unconfigured_budget_admits_everything() {
    let budget = budget(4, None);
    assert!(budget.immovable_refusal(usize::MAX, usize::MAX).is_none());
    assert!(budget.try_reserve(1_000_000));
    budget.release(1_000_000);
}

/// The whole point of an integer budget: a request that needs more
/// blocks than the server owns is refused as a *request* problem,
/// before it is allowed to occupy a queue slot. Answering 503 here
/// would send a client into a retry loop that cannot ever succeed.
#[test]
fn a_request_larger_than_the_whole_budget_is_refused_rather_than_queued() {
    let decoder = tiny_decoder();
    // 2 blocks of 4 positions = 8 positions in the entire server.
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        budget_config(4, 2),
    );

    let err = batcher
        .generate(
            vec![1, 2, 3, 4, 5, 6],
            greedy_params(8, 1),
            StopTokens::from_eos(None),
        )
        .expect_err("14 positions cannot fit an 8-position server");
    let shape = test_shape();
    match &err {
        DecodeError::KvBudgetExceeded {
            binding,
            estimated_bytes,
            limit_bytes,
            positions,
            positions_limit,
            detail,
        } => {
            assert_eq!(*binding, "device_memory_budget_exceeded");
            assert_eq!(*positions, 14);
            assert_eq!(*positions_limit, 8, "2 blocks x 4 positions");
            // The bytes are the model's real KV cost, not a
            // restatement of the block count: an operator has to be
            // able to check the arithmetic against `inspect-plan`.
            assert_eq!(*estimated_bytes, shape.kv_bytes_for_tokens(14));
            assert_eq!(*limit_bytes, shape.kv_bytes_for_tokens(8));
            assert!(estimated_bytes > limit_bytes);
            assert!(detail.contains("14"), "{detail}");
        }
        other => panic!("expected KvBudgetExceeded, got {other:?}"),
    }
    // Retrying it is pointless, and the error says so.
    assert_eq!(err.retry_after_secs(), None);

    let stats = batcher.stats();
    assert_eq!(stats.kv_rejected_too_large, 1);
    assert_eq!(
        stats.kv_rejected_context_length, 0,
        "no per-request context ceiling is configured here"
    );
    assert_eq!(
        stats.queue_rejected, 0,
        "too-big and under-pressure are different counters"
    );
    assert_eq!(
        stats.queue_depth, 0,
        "an impossible request must not occupy a queue slot"
    );
    assert_eq!(stats.kv_blocks_free, 2, "nothing was reserved");

    // A request that does fit still works on the same server.
    batcher
        .generate(vec![1, 2], greedy_params(2, 1), StopTokens::from_eos(None))
        .expect("4 positions fit");
}

/// The other immovable ceiling, and the reason there are two: this
/// request is not too big for the machine, it is too big for what
/// any one request is allowed to be. An operator reading
/// `device_memory_budget_exceeded` here would go looking for a
/// bigger box for a problem a shorter prompt solves.
///
/// Confirmed to FAIL when the `prompt_refusal` call is removed from
/// `ContextCeiling::fit` (the request is admitted and runs).
///
/// The refusing case is the PROMPT's own size, not `prompt +
/// max_tokens`: a prompt with room left over is clamped and served,
/// which is what the test below pins.
#[test]
fn a_prompt_longer_than_the_context_ceiling_names_that_ceiling() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        ceiling_config(6),
    );

    let err = batcher
        .generate(
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            greedy_params(4, 1),
            StopTokens::from_eos(None),
        )
        .expect_err("an 8-token prompt against a 6-position ceiling");
    let shape = test_shape();
    match &err {
        DecodeError::KvBudgetExceeded {
            binding,
            estimated_bytes,
            limit_bytes,
            positions,
            positions_limit,
            detail,
        } => {
            assert_eq!(*binding, "context_length_exceeded");
            assert_eq!(*positions, 8);
            assert_eq!(*positions_limit, 6);
            assert_eq!(*estimated_bytes, shape.kv_bytes_for_tokens(8));
            assert_eq!(*limit_bytes, shape.kv_bytes_for_tokens(6));
            assert!(detail.contains("prompt is too long"), "{detail}");
        }
        other => panic!("expected KvBudgetExceeded, got {other:?}"),
    }
    assert_eq!(err.retry_after_secs(), None);

    let stats = batcher.stats();
    assert_eq!(stats.kv_rejected_context_length, 1);
    assert_eq!(
        stats.kv_rejected_too_large, 0,
        "the machine's budget was never the binding ceiling"
    );
    assert_eq!(stats.queue_rejected, 0);

    // Exactly at the ceiling is admitted: the check is `>`, not
    // `>=`, or the advertised limit would be off by one.
    batcher
        .generate(
            vec![1, 2, 3, 4],
            greedy_params(2, 1),
            StopTokens::from_eos(None),
        )
        .expect("6 positions is 6 positions");
}

/// A prompt that FITS is served with its output budget clamped to
/// what remains, exactly as the private decode loop serves it. This is
/// what makes a large default `max_tokens` safe, and the batcher used
/// to refuse it instead: a server started `-c 16384` answered every
/// chat request that omitted `max_tokens` with a 400 naming a number
/// the caller never sent, because `DEFAULT_CHAT_MAX_TOKENS` is 32768.
///
/// Confirmed to FAIL when `generate_streaming` stops calling
/// `ContextCeiling::fit` (the request is refused).
#[test]
fn an_output_budget_past_the_ceiling_is_clamped_rather_than_refused() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        ceiling_config(6),
    );

    let (_finish, tokens, _text, _usage) = batcher
        .generate(
            vec![1, 2, 3, 4],
            greedy_params(usize::MAX - 4, 1),
            StopTokens::from_eos(None),
        )
        .expect("a 4-token prompt fits a 6-position ceiling with 2 to spare");
    assert_eq!(tokens.len(), 2, "clamped to the 2 positions that remain");

    let stats = batcher.stats();
    assert_eq!(
        stats.kv_rejected_context_length, 0,
        "a clamp is not a refusal, and must not be counted as one"
    );
}

/// A per-request context ceiling with a block budget generous enough
/// that the ceiling is the only thing that can bind.
fn ceiling_config(max_context: usize) -> BatcherConfig {
    BatcherConfig {
        prefill_chunk: 1,
        max_context: Some(max_context),
        kv_block_size: 4,
        kv_blocks: Some(1024),
        ..BatcherConfig::default()
    }
}

/// With both ceilings configured and both exceeded, the request's
/// own size is reported -- it is the one the client can act on.
#[test]
fn the_context_ceiling_is_reported_before_the_device_ceiling() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            max_context: Some(6),
            kv_block_size: 4,
            kv_blocks: Some(2),
            ..BatcherConfig::default()
        },
    );
    let err = batcher
        .generate(
            vec![1, 2, 3, 4, 5, 6],
            greedy_params(8, 1),
            StopTokens::from_eos(None),
        )
        .expect_err("14 positions breaks both ceilings");
    assert!(
        matches!(
            &err,
            DecodeError::KvBudgetExceeded { binding, .. }
                if *binding == "context_length_exceeded"
        ),
        "got {err:?}"
    );
    let stats = batcher.stats();
    assert_eq!(stats.kv_rejected_context_length, 1);
    assert_eq!(stats.kv_rejected_too_large, 0);
}

/// No ceilings configured is the default, and it must refuse
/// nothing.
#[test]
fn without_ceilings_nothing_is_refused_as_too_large() {
    let budget = BlockBudget::new(4, None, Arc::new(ContextCeiling::new(None, test_shape())));
    assert!(budget.immovable_refusal(1_000_000, 1_000_000).is_none());
    assert_eq!(budget.rejected_too_large.load(Ordering::Relaxed), 0);
    assert_eq!(budget.ceiling.refused(), 0);
}

/// The invariant, under real contention: however many requests are
/// in flight, the blocks they hold together never exceed the
/// budget. Six requests each needing two blocks against a
/// four-block server can be at most two at a time.
///
/// Confirmed to FAIL when `admit` reserves unconditionally (peak
/// climbs to 12 -- every request admitted at once).
#[test]
fn concurrent_requests_never_hold_more_blocks_than_the_budget() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        // 4 positions per block, 4 blocks. Each request below is
        // 3 prompt + 5 generated = 8 positions = 2 blocks.
        budget_config(4, 4),
    );

    let start = Arc::new(Barrier::new(6));
    let handles: Vec<_> = (0..6)
        .map(|i| {
            let batcher = batcher.clone();
            let start = Arc::clone(&start);
            thread::spawn(move || {
                start.wait();
                batcher
                    .generate(
                        vec![1, 2, 3],
                        greedy_params(5, i as u64),
                        StopTokens::from_eos(None),
                    )
                    .expect("every request fits the budget on its own")
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("no submitter panicked");
    }

    let stats = batcher.stats();
    assert!(
        stats.kv_blocks_peak <= stats.kv_blocks_total,
        "admission handed out {} blocks from a budget of {}",
        stats.kv_blocks_peak,
        stats.kv_blocks_total
    );
    assert_eq!(stats.kv_rejected_too_large, 0, "all six fit individually");
}

/// Every reservation comes back. A row that leaves without
/// releasing is a leak that shows up as a server which slowly stops
/// admitting anything, with no error anywhere -- so the ledger must
/// be exactly restored once the work is done.
///
/// Confirmed to FAIL when the `budget.release` in
/// `Rows::flush_finished` is removed.
#[test]
fn every_admitted_request_gives_its_blocks_back() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        budget_config(4, 4),
    );
    for i in 0..8 {
        batcher
            .generate(
                vec![1, 2, 3],
                greedy_params(4, i),
                StopTokens::from_eos(None),
            )
            .expect("generate");
    }
    // The last reply is sent before the row is removed, so give the
    // worker its moment to finish the release.
    for _ in 0..200 {
        if batcher.stats().kv_blocks_free == 4 {
            break;
        }
        thread::sleep(std::time::Duration::from_millis(5));
    }
    let stats = batcher.stats();
    assert_eq!(
        stats.kv_blocks_free, stats.kv_blocks_total,
        "an idle server must own its whole budget again"
    );
    assert!(stats.kv_blocks_peak > 0, "something was actually reserved");
}

/// A rejected job must not leak its reservation either: `accept`
/// refuses a prompt this model cannot tokenize, and that job never
/// becomes a row, so nothing downstream would ever release it.
#[test]
fn a_job_rejected_at_validation_gives_its_blocks_back() {
    let decoder = tiny_decoder();
    let vocab = decoder.config.vocab_size;
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        budget_config(4, 4),
    );
    assert!(matches!(
        batcher.generate(
            vec![vocab + 1],
            greedy_params(2, 1),
            StopTokens::from_eos(None)
        ),
        Err(DecodeError::TokenOutOfVocab { .. })
    ));
    for _ in 0..200 {
        if batcher.stats().kv_blocks_free == 4 {
            break;
        }
        thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(batcher.stats().kv_blocks_free, 4);
    // And the server still works afterwards.
    batcher
        .generate(vec![1, 2], greedy_params(2, 1), StopTokens::from_eos(None))
        .expect("a bad request must not poison the budget");
}

/// Admission is strict FIFO: a head job that does not fit stops the
/// line rather than being skipped over by a smaller one behind it.
/// Skip-ahead would raise utilization and let a stream of small
/// requests starve a large one indefinitely.
#[test]
fn a_head_job_that_does_not_fit_holds_the_line() {
    let decoder = tiny_decoder();
    let config = budget_config(4, 4);
    let budget = BlockBudget::new(
        config.kv_block_size,
        config.kv_blocks,
        Arc::new(ContextCeiling::new(config.max_context, test_shape())),
    );
    let queue = QueueGate::new(config.max_queue);
    // Two blocks already out to an in-flight request.
    assert!(budget.try_reserve(2));

    let mut waiting: VecDeque<Job> = VecDeque::new();
    let (big_tx, _big_rx) = mpsc::channel();
    let (small_tx, _small_rx) = mpsc::channel();
    waiting.push_back(Job {
        prompt_tokens: vec![1, 2, 3],
        params: greedy_params(2, 1),
        stop_tokens: StopTokens::from_eos(None),
        reply: big_tx,
        abort: AbortId(0),
        blocks: 3,
        skips: 0,
    });
    waiting.push_back(Job {
        prompt_tokens: vec![1],
        params: greedy_params(2, 2),
        stop_tokens: StopTokens::from_eos(None),
        reply: small_tx,
        abort: AbortId(1),
        blocks: 1,
        skips: 0,
    });
    // The gate is counting both of them.
    queue.try_reserve().expect("cap 512");
    queue.try_reserve().expect("cap 512");

    let mut prefills: VecDeque<Prefill> = VecDeque::new();
    admit(
        &decoder,
        &mut waiting,
        &mut prefills,
        0,
        &config,
        &queue,
        &budget,
        None,
    );

    assert!(
        prefills.is_empty(),
        "the 1-block job must not jump the 3-block job that cannot fit"
    );
    assert_eq!(waiting.len(), 2);
    assert_eq!(queue.depth(), 2, "neither job has stopped waiting");

    // Once the in-flight request finishes, the line moves -- both,
    // in order.
    budget.release(2);
    admit(
        &decoder,
        &mut waiting,
        &mut prefills,
        0,
        &config,
        &queue,
        &budget,
        None,
    );
    assert_eq!(prefills.len(), 2);
    assert!(waiting.is_empty());
    assert_eq!(queue.depth(), 0);
    assert_eq!(budget.free(), 0, "3 + 1 blocks are now out");
}

/// The sequence cap and the block cap are separate statements and
/// both must hold. A budget with room for four requests does not
/// override `max_seqs = 1`.
#[test]
fn the_sequence_cap_and_the_block_cap_compose() {
    let decoder = tiny_decoder();
    let config = BatcherConfig {
        max_seqs: 1,
        ..budget_config(4, 8)
    };
    let budget = BlockBudget::new(
        config.kv_block_size,
        config.kv_blocks,
        Arc::new(ContextCeiling::new(config.max_context, test_shape())),
    );
    let queue = QueueGate::new(config.max_queue);
    let mut waiting: VecDeque<Job> = VecDeque::new();
    // The receivers stay alive for the test: a dropped receiver
    // would make the reply channel closed, which is a different
    // situation from the one under test.
    let mut receivers = Vec::new();
    for i in 0..3 {
        let (tx, rx) = mpsc::channel();
        receivers.push(rx);
        waiting.push_back(Job {
            prompt_tokens: vec![1, 2],
            params: greedy_params(2, i),
            stop_tokens: StopTokens::from_eos(None),
            reply: tx,
            abort: AbortId(i),
            blocks: 1,
            skips: 0,
        });
        queue.try_reserve().expect("cap 512");
    }
    let mut prefills: VecDeque<Prefill> = VecDeque::new();
    admit(
        &decoder,
        &mut waiting,
        &mut prefills,
        0,
        &config,
        &queue,
        &budget,
        None,
    );
    assert_eq!(prefills.len(), 1, "max_seqs still binds");
    assert_eq!(budget.free(), 7, "only the admitted job reserved");
    assert_eq!(receivers.len(), 3);
}
