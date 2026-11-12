use super::*;

use crate::server_draft::{DraftKind, ServerDraft};

/// One layer of `tiny_decoder`'s two, same seed: right often enough to
/// be accepted and wrong often enough to be cut.
fn partial_draft() -> ServerDraft {
    let cfg = test_dense_fixture();
    let vocab = cfg.vocab_size;
    ServerDraft {
        kind: DraftKind::Model(frink_models::DraftWeights::new(Decoder::new_random_small(
            cfg, 1, vocab,
        ))),
        max: 4,
        p_min: 0.0,
    }
}

fn paged_store(decoder: &Decoder) -> PagedKvConfig {
    PagedKvConfig {
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
    }
}

fn sampled_params(max_tokens: usize, seed: u64) -> GenerationParams {
    let mut p = greedy_params(max_tokens, seed);
    p.sampling.temperature = 0.8;
    p
}

/// Every row of a continuous batch speculates, and the answer is the
/// one each request gets decoded alone: greedy and seeded sampling,
/// with a draft model and with prompt lookup, on contiguous and on
/// paged KV. Three requests at once, so blocks of different lengths
/// share one verify forward.
#[test]
fn a_speculating_batch_answers_as_each_request_alone() {
    let decoder = tiny_decoder();
    let jobs: Vec<(Vec<usize>, GenerationParams)> = vec![
        (vec![1, 2, 3, 1, 2, 3], greedy_params(24, 7)),
        (vec![4, 5], sampled_params(20, 11)),
        (vec![9, 8, 9, 8, 9], greedy_params(16, 3)),
    ];
    let reference: Vec<Vec<usize>> = jobs
        .iter()
        .map(|(p, par)| sequential_ids(&decoder, p, par))
        .collect();

    let mut cut = false;
    for paged in [false, true] {
        for draft in [Some(partial_draft()), None] {
            let store = paged.then(|| paged_store(&decoder));
            let free_before = store.as_ref().map(|s| s.store.free_groups());
            let config = BatcherConfig {
                prefill_chunk: 1,
                ..BatcherConfig::default()
            };
            let ceiling = Arc::new(ContextCeiling::new(
                config.max_context,
                frink_models::KvShape::host(&decoder.config),
            ));
            let batcher = ContinuousBatcher::spawn_with_ceiling(
                Arc::clone(&decoder),
                identity_decode(),
                config,
                ceiling,
                store.clone(),
                draft,
            );
            let barrier = Arc::new(Barrier::new(jobs.len()));
            let threads: Vec<_> = jobs
                .iter()
                .cloned()
                .map(|(prompt, par)| {
                    let batcher = batcher.clone();
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        batcher
                            .generate(prompt, par, StopTokens::default())
                            .expect("batch generate")
                    })
                })
                .collect();
            for (i, t) in threads.into_iter().enumerate() {
                let (finish, ids, _text, usage) = t.join().unwrap();
                assert_eq!(
                    ids, reference[i],
                    "request {i} (paged: {paged}) changed by speculation"
                );
                assert_eq!(finish, FinishReason::Length);
                let drafted = usage.draft_tokens.unwrap_or(0);
                let accepted = usage.accepted_draft_tokens.unwrap_or(0);
                cut |= accepted > 0 && drafted > accepted;
            }
            if let (Some(store), Some(before)) = (store, free_before) {
                // The rows are flushed after their replies; give the
                // worker its tick.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while store.store.free_groups() != before {
                    assert!(std::time::Instant::now() < deadline, "a page leaked");
                    thread::yield_now();
                }
            }
        }
    }
    assert!(
        cut,
        "no request both accepted and rejected a draft, so settling a block was not tested"
    );
}

/// A row that drafts reports it in `usage`, as on the private loop, and
/// one that never drafted reports nothing: a zero would read as "the
/// drafter was useless".
#[test]
fn a_batched_row_reports_its_speculation_only_when_it_drafted() {
    let decoder = tiny_decoder();
    let config = BatcherConfig {
        prefill_chunk: 1,
        ..BatcherConfig::default()
    };
    let ceiling = Arc::new(ContextCeiling::new(
        config.max_context,
        frink_models::KvShape::host(&decoder.config),
    ));
    let batcher = ContinuousBatcher::spawn_with_ceiling(
        Arc::clone(&decoder),
        identity_decode(),
        config,
        ceiling,
        None,
        Some(partial_draft()),
    );
    let (_, ids, _, usage) = batcher
        .generate(vec![1, 2, 3], greedy_params(24, 7), StopTokens::default())
        .expect("generates");
    assert!(usage.draft_tokens.is_some_and(|d| d > 0), "{usage:?}");
    assert!(usage.acceptance_length.is_some());

    // One token: no room to draft.
    let (_, _, _, usage) = batcher
        .generate(vec![1, 2, 3], greedy_params(1, 7), StopTokens::default())
        .expect("generates");
    assert_eq!(usage.draft_tokens, None, "{usage:?}");
    assert_eq!(ids.len(), 24);
}
