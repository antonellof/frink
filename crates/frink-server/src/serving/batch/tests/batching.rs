use super::*;

/// Continuous batching and paged KV COMPOSE: two concurrent jobs
/// over a shared paged store produce the same ids as two sequential
/// private-loop generates.
///
/// This is what the old exclusivity forbade. The batcher refused to
/// run alongside a KV pool or prefix cache because a batched row
/// could do neither pool acquisition nor prefix restore; a row that
/// holds a `PagedLease` gets both, so the refusal had nothing left
/// to protect. Token-for-token, not merely "it runs": the point is
/// that turning two independent switches on does not change what
/// the model says.
#[test]
fn continuous_batching_composes_with_paged_kv() {
    let decoder = tiny_decoder();
    let prompts: [Vec<usize>; 2] = [vec![1, 2, 3], vec![4, 5]];
    let params = [greedy_params(6, 7), greedy_params(4, 11)];
    let sequential: Vec<Vec<usize>> = prompts
        .iter()
        .zip(params.iter())
        .map(|(p, par)| sequential_ids(&decoder, p, par))
        .collect();

    let paged = PagedKvConfig {
        store: Arc::new(frink_core::cache::SharedPagedKv::new(
            decoder.layers.len(),
            /* block_size = */ 4,
            /* blocks_per_layer = */ 256,
            decoder.config.n_kv_heads,
            decoder.config.head_dim,
        )),
        queue_wait: std::time::Duration::ZERO,
        radix: None,
        anchor_token: None,
        slide_interval: crate::policy::pool_budget::DEFAULT_SWA_EVICTION_INTERVAL,
    };
    let store = Arc::clone(&paged.store);
    let free_before = store.free_groups();

    let batcher = ContinuousBatcher::spawn_with_config_paged(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
        Some(paged),
    );
    let barrier = Arc::new(Barrier::new(3));
    let results = Arc::new(Mutex::new(vec![None, None]));
    let mut threads = Vec::new();
    for i in 0..2 {
        let batcher = batcher.clone();
        let barrier = Arc::clone(&barrier);
        let results = Arc::clone(&results);
        let prompt = prompts[i].clone();
        let par = GenerationParams {
            cache_salt: None,
            prompt_logprobs: None,
            wants_logprobs: false,
            n: 1,
            interleave_choices: false,
            keep_special_tokens: false,
            truncate_prompt_tokens: None,
            token_mask: crate::token_mask::TokenMask::default(),
            reasoning: None,
            max_tokens: params[i].max_tokens,
            // Cloned rather than field-by-field: a hand-written copy
            // of `SamplingParams` is this repo's dominant defect at its
            // smallest, and this one already omitted every sampler
            // added after it was written.
            sampling: params[i].sampling.clone(),
            seed: params[i].seed,
            stop: vec![],
            stop_token_ids: Vec::new(),
            json_object: params[i].json_object,
            grammar: None,
            cancel: params[i].cancel.clone(),
            ignore_eos: false,
            reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
            lora: None,
        };
        threads.push(thread::spawn(move || {
            barrier.wait();
            let out = batcher
                .generate(prompt, par, StopTokens::default())
                .expect("a paged batched row must serve");
            results.lock().unwrap()[i] = Some(out.1);
        }));
    }
    barrier.wait();
    for t in threads {
        t.join().unwrap();
    }
    let got = results.lock().unwrap().clone();
    for (i, want) in sequential.iter().enumerate() {
        assert_eq!(
            got[i].as_ref().expect("both rows replied"),
            want,
            "row {i}: batching over paged KV changed the ids"
        );
    }

    // And every page came back once both rows ended.
    drop(batcher);
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert_eq!(
        store.free_groups(),
        free_before,
        "a finished batched row must return its pages"
    );
}

/// A window model slides ON THE BATCHER, and the batcher's own
/// budget prices it at what it holds.
///
/// Both halves are needed and they fail differently. Without the
/// slide in the decode step the paged store runs out and a row is
/// refused; without the window in the budget the row never gets that
/// far, because the budget refuses it at submission for a context it
/// was never going to hold. Both ceilings are set between the two
/// answers here, so either alone breaks the test.
///
/// Token-for-token against the sequential private loop, because a
/// slide that dropped a page one step early would still produce
/// fluent output -- just not this output.
#[test]
fn a_window_model_slides_while_continuously_batched() {
    let window = 8;
    let block_size = 4;
    let mut cfg = test_dense_fixture();
    cfg.sliding_window = Some(window);
    cfg.swa_layers = frink_models::swa_layers::SwaLayers::All;
    let vocab = cfg.vocab_size;
    let decoder = Arc::new(Decoder::new_random_small(cfg, 2, vocab));
    assert_eq!(decoder.config.uniform_sliding_window(), Some(window));

    let max_tokens = 400;
    let prompts: [Vec<usize>; 2] = [vec![1, 2, 3], vec![4, 5, 6]];
    let long = |seed: u64| GenerationParams {
        ignore_eos: true,
        ..greedy_params(max_tokens, seed)
    };
    let params = [long(7), long(11)];
    let sequential: Vec<Vec<usize>> = prompts
        .iter()
        .zip(params.iter())
        .map(|(p, par)| sequential_ids(&decoder, p, par))
        .collect();

    // 48 page groups per windowed row against 101 without the
    // window, and two rows to serve.
    let paged = PagedKvConfig {
        store: Arc::new(frink_core::cache::SharedPagedKv::new(
            decoder.layers.len(),
            block_size,
            /* blocks_per_layer = */ 120,
            decoder.config.n_kv_heads,
            decoder.config.head_dim,
        )),
        queue_wait: std::time::Duration::from_secs(5),
        radix: None,
        anchor_token: None,
        slide_interval: crate::policy::pool_budget::DEFAULT_SWA_EVICTION_INTERVAL,
    };
    let store = Arc::clone(&paged.store);
    let free_before = store.free_groups();

    let batcher = ContinuousBatcher::spawn_with_config_paged(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 8,
            kv_block_size: block_size,
            // 48 blocks per windowed row; 101 without the window,
            // which does not fit even once.
            kv_blocks: Some(100),
            ..BatcherConfig::default()
        },
        Some(paged),
    );

    let barrier = Arc::new(Barrier::new(3));
    let results = Arc::new(Mutex::new(vec![None, None]));
    let mut threads = Vec::new();
    for i in 0..2 {
        let batcher = batcher.clone();
        let barrier = Arc::clone(&barrier);
        let results = Arc::clone(&results);
        let prompt = prompts[i].clone();
        let par = params[i].clone();
        threads.push(thread::spawn(move || {
            barrier.wait();
            let out = batcher
                .generate(prompt, par, StopTokens::default())
                .expect("a windowed batched row must serve");
            results.lock().unwrap()[i] = Some(out.1);
        }));
    }
    barrier.wait();
    for t in threads {
        t.join().unwrap();
    }
    let got = results.lock().unwrap().clone();
    for (i, want) in sequential.iter().enumerate() {
        let ids = got[i].as_ref().expect("both rows replied");
        assert_eq!(ids.len(), max_tokens, "row {i} stopped early");
        assert_eq!(ids, want, "row {i}: the window slide changed the ids");
    }

    drop(batcher);
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert_eq!(
        store.free_groups(),
        free_before,
        "a finished slid row must return its recycled pages too"
    );
}

/// Two concurrent jobs through the batcher must match two sequential
/// private-loop generates token-for-token.
#[test]
fn continuous_batch_matches_sequential_generate_token_ids() {
    let decoder = tiny_decoder();
    let prompts: [Vec<usize>; 2] = [vec![1, 2, 3], vec![4, 5]];
    let params = [greedy_params(8, 7), greedy_params(5, 11)];
    let sequential: Vec<Vec<usize>> = prompts
        .iter()
        .zip(params.iter())
        .map(|(p, par)| sequential_ids(&decoder, p, par))
        .collect();

    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        // Chunk 1: every prompt token is its own scheduling unit, the
        // most aggressive split, and the sampled ids must not move.
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    let barrier = Arc::new(Barrier::new(3));
    let results = Arc::new(Mutex::new(vec![None, None]));
    let mut threads = Vec::new();
    for i in 0..2 {
        let batcher = batcher.clone();
        let barrier = Arc::clone(&barrier);
        let results = Arc::clone(&results);
        let prompt = prompts[i].clone();
        let par = GenerationParams {
            cache_salt: None,
            prompt_logprobs: None,
            wants_logprobs: false,
            n: 1,
            interleave_choices: false,
            keep_special_tokens: false,
            truncate_prompt_tokens: None,
            token_mask: crate::token_mask::TokenMask::default(),
            reasoning: None,
            max_tokens: params[i].max_tokens,
            // Cloned rather than field-by-field: a hand-written copy
            // of `SamplingParams` is this repo's dominant defect at its
            // smallest, and this one already omitted every sampler
            // added after it was written.
            sampling: params[i].sampling.clone(),
            seed: params[i].seed,
            stop: vec![],
            stop_token_ids: Vec::new(),
            json_object: params[i].json_object,
            grammar: None,
            cancel: params[i].cancel.clone(),
            ignore_eos: false,
            reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
            lora: None,
        };
        threads.push(thread::spawn(move || {
            barrier.wait();
            let out = batcher
                .generate(prompt, par, StopTokens::default())
                .expect("batch generate");
            results.lock().unwrap()[i] = Some(out.1);
        }));
    }
    barrier.wait();
    for t in threads {
        t.join().unwrap();
    }
    let got = results.lock().unwrap();
    assert_eq!(got[0].as_ref().unwrap(), &sequential[0]);
    assert_eq!(got[1].as_ref().unwrap(), &sequential[1]);
}

#[test]
fn continuous_batch_honors_stop_sequence_in_decoded_text() {
    let decoder = tiny_decoder();
    // Map every token id to a fixed letter so a stop string is easy
    // to force once we know the first few sequential ids.
    let decode: DecodeFn = Arc::new(|ids: &[usize]| {
        ids.iter()
            .map(|id| match id % 3 {
                0 => b'X',
                1 => b'Y',
                _ => b'Z',
            })
            .collect()
    });
    let prompt = vec![1usize, 2, 3];
    let mut params = greedy_params(32, 3);
    // First generate without stop to learn the decoded stream.
    let ids = sequential_ids(&decoder, &prompt, &params);
    let full: String = ids
        .iter()
        .map(|id| match id % 3 {
            0 => 'X',
            1 => 'Y',
            _ => 'Z',
        })
        .collect();
    // Pick a two-char substring that appears mid-stream when long enough.
    assert!(
        full.len() >= 4,
        "need enough tokens to place a mid-stream stop"
    );
    let stop = full[2..4].to_string();
    params.stop = vec![stop.clone()];

    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        decode,
        BatcherConfig {
            prefill_chunk: 2,
            ..BatcherConfig::default()
        },
    );
    let (finish, _ids, text, _usage) = batcher
        .generate(prompt, params, StopTokens::default())
        .expect("batch generate");
    assert_eq!(
        finish,
        FinishReason::StopSequence(stop.clone()),
        "a batched row must name the stop it hit, like an unbatched one"
    );
    assert!(
        !text.contains(&stop),
        "stop string must be trimmed from visible text: text={text:?} stop={stop:?}"
    );
    assert_eq!(&full[..full.find(&stop).unwrap()], text);
}
/// The continuous batcher carried the same single `eos_id` every
/// other server decode loop did, so a Llama-3 or gemma checkpoint
/// served through it ran past its own turn ender to `max_tokens`.
/// Here the stop set holds the third token this prompt would
/// otherwise generate and nothing else: a loop honouring the set
/// stops with exactly two tokens, one comparing against a lone
/// metadata EOS runs all 32.
#[test]
fn continuous_batch_stops_on_any_member_of_the_stop_set() {
    let decoder = tiny_decoder();
    let decode: DecodeFn = Arc::new(|_: &[usize]| Vec::new());
    let prompt = vec![1usize, 2, 3];
    let params = greedy_params(32, 3);
    let ids = sequential_ids(&decoder, &prompt, &params);
    assert!(ids.len() > 3, "need a mid-stream token to stop on");
    let turn_ender = ids[2];

    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        decode,
        BatcherConfig {
            prefill_chunk: 2,
            ..BatcherConfig::default()
        },
    );
    let (finish, got, _text, usage) = batcher
        .generate(prompt, params, StopTokens::from_eos(Some(turn_ender)))
        .expect("batch generate");
    assert_eq!(finish, FinishReason::Stop);
    assert_eq!(got, ids[..2].to_vec());
    assert_eq!(usage.completion_tokens, 2);
}

/// Under continuous batching, callers can observe tokens as they are
/// sampled rather than only in the final reply.
#[test]
fn batched_generate_streams_incremental_chunks() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    let mut streamed = Vec::new();
    let (finish, _ids, text, _usage) = batcher
        .generate_streaming(
            vec![1, 2, 3],
            greedy_params(8, 4),
            StopTokens::default(),
            Some(|chunk: &str| streamed.push(chunk.to_string())),
        )
        .expect("generate");
    assert!(matches!(finish, FinishReason::Length | FinishReason::Stop));
    assert!(!streamed.is_empty(), "expected at least one streamed chunk");
    assert_eq!(streamed.concat(), text);
}

/// The state machine itself: each `step_chunk` is bounded by the
/// chunk size, is resumable, and reports done exactly once the
/// prompt is exhausted. This is the property the whole scheduler
/// rests on -- an unbounded prefill has no safe interleaving point.
#[test]
fn prefill_step_chunk_is_bounded_and_resumable() {
    let decoder = tiny_decoder();
    let prompt: Vec<usize> = (1..=7).collect();
    let mut state = PrefillState::new(Arc::clone(&decoder), &prompt, 3);
    assert_eq!(state.tokens_remaining(), 7);
    assert_eq!(state.tokens_processed(), 0);

    assert!(!state.step_chunk());
    assert_eq!(state.tokens_processed(), 3, "a chunk may not overrun");
    assert_eq!(state.tokens_remaining(), 4);

    assert!(!state.step_chunk());
    assert_eq!(state.tokens_processed(), 6);

    assert!(state.step_chunk(), "final short chunk finishes the prompt");
    assert_eq!(state.tokens_processed(), 7);
    assert_eq!(state.tokens_remaining(), 0);
    assert!(state.is_done());
    assert!(state.step_chunk(), "stepping a finished prefill is a no-op");
    assert_eq!(state.tokens_processed(), 7);
}

/// An empty prompt still needs one forward pass to have logits to
/// sample from -- the case the pre-chunking `admit` special-cased.
#[test]
fn empty_prompt_prefills_one_stand_in_token() {
    let decoder = tiny_decoder();
    let mut state = PrefillState::new(Arc::clone(&decoder), &[], 4);
    assert_eq!(state.tokens_remaining(), 1);
    assert!(state.step_chunk());
    let (_caches, logits, pos, _ids) = state.into_decode_start();
    assert_eq!(pos, 1);
    assert_eq!(logits.len(), decoder.config.vocab_size);
}

/// Chunking is a scheduling boundary, not a numerical one: whatever
/// the chunk size, the prompt runs through the same prefill at the
/// same positions, so the final logits match the sequential reference.
/// Prefill uses `forward_batch_last_host_kv` so host K/V stays
/// authoritative for batched Metal decode; tiny float drift vs
/// per-token `forward_token` on CPU is expected and harmless.
#[test]
fn prefill_chunking_does_not_change_logits() {
    let decoder = tiny_decoder();
    let prompt: Vec<usize> = (0..11).map(|i| (i * 3 + 1) % 16).collect();

    let mut sequential: Vec<f32> = Vec::new();
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    for (pos, &tok) in prompt.iter().enumerate() {
        sequential = decoder.forward_token(tok, pos, &mut caches);
    }

    for chunk in [1usize, 2, 5, 11, 64] {
        let mut state = PrefillState::new(Arc::clone(&decoder), &prompt, chunk);
        while !state.step_chunk() {}
        let (_caches, logits, pos, _ids) = state.into_decode_start();
        assert_eq!(pos, prompt.len());
        assert_eq!(logits.len(), sequential.len());
        for (i, (&got, &want)) in logits.iter().zip(sequential.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "chunk size {chunk}, logit {i}: got={got} want={want}"
            );
        }
    }
}

/// The scheduling property chunking exists for, in two claims that
/// both fail under an unbounded prefill:
///
/// 1. A long prompt is *observable in partial states* -- it is a
///    sequence of bounded units, not one uninterruptible call. The
///    pre-chunking scheduler ran the whole prompt inside `admit`,
///    where `prefill_tokens` could only ever jump 0 -> len.
/// 2. Decode keeps stepping while those partial states go by. A
///    prompt joining the batch costs an in-flight decode one chunk,
///    not the whole prompt.
#[test]
fn long_prefill_does_not_freeze_an_in_flight_decode() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );

    // A long-running decode: enough tokens that it is still
    // generating while the second job's prompt is chunked through.
    let decode_job = {
        let batcher = batcher.clone();
        thread::spawn(move || {
            batcher.generate(vec![1, 2], greedy_params(90, 5), StopTokens::default())
        })
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while batcher.stats().decode_steps < 2 {
        assert!(std::time::Instant::now() < deadline, "decode never started");
        thread::yield_now();
    }

    let long_prompt: Vec<usize> = (0..40).map(|i| (i % 16) + 1).collect();
    let total = long_prompt.len() as u64;
    let prefill_at_submit = batcher.stats().prefill_tokens;
    let prefill_job = {
        let batcher = batcher.clone();
        thread::spawn(move || {
            batcher.generate(long_prompt, greedy_params(1, 9), StopTokens::default())
        })
    };

    // Claim 1: catch the long prompt mid-prefill. An unbounded
    // prefill is never observable here -- it goes straight to done.
    let decode_before = loop {
        assert!(
            std::time::Instant::now() < deadline,
            "never observed the long prompt mid-prefill"
        );
        let st = batcher.stats();
        let progressed = st.prefill_tokens - prefill_at_submit;
        assert!(
            progressed < total,
            "the whole prompt was prefilled without ever being observed \
                 partially done: prefill ran as one unbounded unit of work"
        );
        if progressed > 0 {
            break st.decode_steps;
        }
        thread::yield_now();
    };

    // Claim 2: decode advances before that prefill finishes.
    loop {
        assert!(
            std::time::Instant::now() < deadline,
            "decode stalled while a long prompt prefilled"
        );
        let st = batcher.stats();
        if st.decode_steps > decode_before {
            break;
        }
        assert!(
            st.prefill_tokens - prefill_at_submit < total,
            "the prompt finished prefilling before the in-flight decode \
                 took a single step: prefill froze decode"
        );
        thread::yield_now();
    }

    let (_finish, ids, _text, _usage) = prefill_job.join().unwrap().expect("prefill job");
    assert_eq!(ids.len(), 1);
    let (_finish, ids, _text, _usage) = decode_job.join().unwrap().expect("decode job");
    assert_eq!(ids.len(), 90);
}

/// The in-flight cap counts prompts that are still prefilling, not
/// just rows already decoding -- a prefilling prompt holds a full
/// set of KV caches. Two jobs under `max_seqs: 1` must both still
/// complete correctly (the second waits in the channel).
#[test]
fn max_seqs_cap_counts_prefilling_prompts_and_still_serves_both() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            max_seqs: 1,
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    let expected: Vec<Vec<usize>> = [(vec![1usize, 2, 3], 6u64), (vec![4usize, 5], 6)]
        .iter()
        .map(|(p, seed)| sequential_ids(&decoder, p, &greedy_params(6, *seed)))
        .collect();

    let handles: Vec<_> = [(vec![1usize, 2, 3], 6u64), (vec![4usize, 5], 6)]
        .into_iter()
        .map(|(prompt, seed)| {
            let batcher = batcher.clone();
            thread::spawn(move || {
                batcher
                    .generate(prompt, greedy_params(6, seed), StopTokens::default())
                    .expect("generate")
                    .1
            })
        })
        .collect();
    let got: Vec<Vec<usize>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(got[0], expected[0]);
    assert_eq!(got[1], expected[1]);
}

/// The other constrained-decoding feature, on the other decode loop.
///
/// A grammar rides on `GenerationParams` exactly as `json_object` does,
/// so the way it gets dropped here is the same way JSON mode did: the
/// batcher sampling for itself instead of through `sample_step`. The
/// detokenizer makes every EVEN id render as `b` and every odd id as
/// `a`, and `root ::= "a"+` admits only the odd ones -- decidable from
/// the ids alone, with no model behaviour assumed.
///
/// Both halves are asserted. The unconstrained run must reach an even
/// id, or the constrained run below proves nothing.
#[test]
fn a_grammar_constrains_a_batched_row_as_it_does_a_private_one() {
    let decoder = tiny_decoder();
    let decode: DecodeFn = Arc::new(|ids: &[usize]| {
        ids.iter()
            .map(|id| if id % 2 == 0 { b'b' } else { b'a' })
            .collect()
    });
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        decode,
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );

    let prompt = vec![1usize, 2, 3];
    let run = |grammar: Option<&str>| {
        let mut params = greedy_params(12, 5);
        params.grammar = grammar.map(|src| {
            Arc::new(
                frink_models::grammar::Grammar::from_str_with_root(src, "root")
                    .expect("test grammar parses"),
            )
        });
        batcher
            .generate(prompt.clone(), params, StopTokens::default())
            .expect("generate")
            .1
    };

    let unconstrained = run(None);
    assert!(
        unconstrained.iter().any(|id| id % 2 == 0),
        "the unconstrained run reached no forbidden token, so the \
         constrained run below would prove nothing"
    );

    let constrained = run(Some(r#"root ::= "a"+"#));
    assert!(
        !constrained.is_empty(),
        "a grammar-constrained batched row must still produce tokens"
    );
    assert!(
        constrained.iter().all(|id| id % 2 == 1),
        "a batched row sampled a token that renders as `b`: the grammar \
         was not applied ({constrained:?})"
    );
}

/// A batched row whose grammar cannot be satisfied fails THAT ROW, and
/// says why. The alternative -- a 200 carrying text the grammar forbids
/// -- is the failure this whole path exists to prevent, and a worker
/// that panicked or hung would take every other row down with it.
#[test]
fn a_batched_row_whose_grammar_dead_ends_is_refused_by_itself() {
    let decoder = tiny_decoder();
    let decode: DecodeFn = Arc::new(|ids: &[usize]| {
        ids.iter()
            .map(|id| if id % 2 == 0 { b'b' } else { b'a' })
            .collect()
    });
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        decode,
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );

    let mut params = greedy_params(8, 5);
    params.grammar = Some(Arc::new(
        // No token in this vocabulary renders as "z".
        frink_models::grammar::Grammar::from_str_with_root(r#"root ::= "z""#, "root")
            .expect("test grammar parses"),
    ));
    let err = batcher
        .generate(vec![1usize, 2, 3], params, StopTokens::default())
        .expect_err("this grammar cannot be served");
    assert!(
        matches!(err, DecodeError::GrammarConstraint { .. }),
        "{err}"
    );

    // The batcher is still serving: the refusal was one row's, not the
    // worker's.
    let after = batcher
        .generate(
            vec![1usize, 2, 3],
            greedy_params(4, 5),
            StopTokens::default(),
        )
        .expect("the worker survives a row it had to refuse");
    assert_eq!(after.1.len(), 4);
}

/// A per-request feature must not depend on a server-side env var.
///
/// `response_format: {"type": "json_object"}` was honoured by the
/// private decode loop and dropped here, so the same body got a
/// constrained answer or an unconstrained one depending on whether
/// `FRINK_CONTINUOUS_BATCHING=1` was set -- which the caller cannot
/// see. The only visible symptom was the final
/// `validate_json_object_output` turning into a 400 for output the
/// server itself had failed to constrain.
///
/// The detokenizer here makes every EVEN token id render as `<`, which
/// no JSON document may contain, and every odd id render as `a`, which
/// any of them may. So "the mask was applied" is decidable from the ids
/// alone, with no model behaviour assumed.
///
/// Both halves are asserted, and the second is what stops this passing
/// vacuously: the SAME prompt and seed without `json_object` must reach
/// at least one even id, or the constrained run proves nothing.
#[test]
fn json_object_mode_constrains_a_batched_row_as_it_does_a_private_one() {
    let decoder = tiny_decoder();
    let decode: DecodeFn = Arc::new(|ids: &[usize]| {
        ids.iter()
            .map(|id| if id % 2 == 0 { b'<' } else { b'a' })
            .collect()
    });
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        decode,
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );

    let prompt = vec![1usize, 2, 3];
    let run = |json_object: bool| {
        let mut params = greedy_params(12, 5);
        params.json_object = json_object;
        batcher
            .generate(prompt.clone(), params, StopTokens::default())
            .expect("generate")
            .1
    };

    let unconstrained = run(false);
    assert!(
        unconstrained.iter().any(|id| id % 2 == 0),
        "the unconstrained run reached no forbidden token, so the \
         constrained run below would prove nothing"
    );

    let constrained = run(true);
    assert!(
        !constrained.is_empty(),
        "a json-mode batched row must still produce tokens"
    );
    assert!(
        constrained.iter().all(|id| id % 2 == 1),
        "a batched json-mode row sampled a token that renders as `<`: \
         the logit mask was not applied ({constrained:?})"
    );
}
