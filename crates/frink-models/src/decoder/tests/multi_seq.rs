//! The multi-sequence step against independent single-sequence decodes,
//! over contiguous and paged KV.

use super::*;

/// `forward_multi_seq`'s core correctness property: batching N
/// independent sequences (different token histories, different
/// current positions, different KV caches) together must produce
/// EXACTLY the same per-sequence output as running each sequence
/// through `forward_token` alone, one step at a time. This is what
/// makes continuous batching safe -- no sequence's attention may
/// ever be perturbed by another sequence sharing its batched
/// matmul step.
/// The PAGED batch step must equal the contiguous one, bit for bit.
///
/// Continuous batching and paging are independent choices, so a
/// deployment can have either, both or neither; if they disagree,
/// the answer depends on two switches nobody thinks of as changing
/// the model. Every sequence here is at a different position with a
/// different length, which is the case the batched path exists for
/// and the one where a shared-KV mistake would surface.
#[test]
fn a_paged_multi_seq_step_is_bit_identical_to_the_contiguous_one() {
    for cfg in [
        tiny_test_config(),
        {
            let mut c = tiny_test_config();
            c.sliding_window = Some(2);
            c.swa_layers = crate::swa_layers::SwaLayers::All;
            c
        },
        {
            let mut c = tiny_test_config();
            c.embedding_scale = Some(7.5);
            c.final_logit_softcap = Some(0.05);
            c.attn_logit_softcap = Some(0.05);
            c
        },
    ] {
        let n_layers = 2;
        let decoder = Decoder::new_random_small(cfg, n_layers, 10);
        let histories: [&[usize]; 3] = [&[1, 3, 5], &[2, 7], &[4, 4, 4, 6]];
        let next = [6usize, 1, 2];

        // Contiguous: build each sequence's history, then one step.
        let mut contiguous: Vec<Vec<KvCache>> = histories
            .iter()
            .map(|h| {
                let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
                for (pos, &tok) in h.iter().enumerate() {
                    decoder.forward_token(tok, pos, &mut caches);
                }
                caches
            })
            .collect();
        let positions: Vec<usize> = histories.iter().map(|h| h.len()).collect();
        let want = decoder.forward_multi_seq(&next, &positions, &mut contiguous);

        // Paged: same histories through the paged decode path, then
        // one batched step over the shared store.
        let stores = SharedPagedKv::new(
            n_layers,
            /* block_size = */ 2,
            /* blocks_per_layer = */ 64,
            decoder.config.n_kv_heads,
            decoder.config.head_dim,
        );
        let mut paged: Vec<Vec<PagedKvCache>> = histories
            .iter()
            .map(|h| {
                let mut caches: Vec<PagedKvCache> =
                    (0..n_layers).map(|_| PagedKvCache::new()).collect();
                for (pos, &tok) in h.iter().enumerate() {
                    decoder
                        .forward_token_paged(tok, pos, &mut caches, &stores)
                        .expect("sized generously");
                }
                caches
            })
            .collect();
        let got = decoder.forward_multi_seq_kv(
            &next,
            &positions,
            &mut MultiSeqKv::Paged {
                caches: &mut paged,
                stores: &stores,
            },
        );

        assert_eq!(want.len(), got.len());
        for (s, (a, b)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "sequence {s}: logit count");
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "sequence {s}: paged batching changed the answer"
                );
            }
        }
    }
}

#[test]
fn forward_multi_seq_matches_independent_forward_token_per_sequence() {
    let cfg = tiny_test_config();
    let vocab = 8;
    // 3 independent sequences, deliberately different lengths/
    // histories/current tokens, so no two sequences are at the
    // same position when batched together.
    let seq_histories: [&[usize]; 3] = [&[1, 3, 5], &[2, 7], &[4, 4, 4, 6]];

    let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
    let mut independent_logits: Vec<Vec<f32>> = Vec::new();
    for history in seq_histories.iter() {
        let mut caches: Vec<KvCache> = decoder_a.config.new_kv_caches();
        let mut logits = Vec::new();
        for (pos, &tok) in history.iter().enumerate() {
            logits = decoder_a.forward_token(tok, pos, &mut caches);
        }
        independent_logits.push(logits);
    }

    // Same seed -> identical weights, fresh caches for a fair
    // comparison (mirrors forward_batch_matches_sequential_forward_token_exactly).
    let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
    let mut per_seq_caches: Vec<Vec<KvCache>> = seq_histories
        .iter()
        .map(|_| decoder_b.config.new_kv_caches())
        .collect();

    // Feed every sequence's prefix (all but its last token)
    // through forward_multi_seq one shared step at a time, then
    // do a final batched step for the last token of every
    // sequence so all three arrive at their final position in
    // the same batched call -- exercising genuinely different
    // per-sequence positions/histories within one batch, not just
    // parallel identical-length sequences.
    let max_len = seq_histories.iter().map(|h| h.len()).max().unwrap();
    let mut batched_logits: Vec<Vec<f32>> = vec![Vec::new(); seq_histories.len()];
    for step in 0..max_len {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut active: Vec<usize> = Vec::new();
        for (s, history) in seq_histories.iter().enumerate() {
            if step < history.len() {
                tokens.push(history[step]);
                positions.push(step);
                active.push(s);
            }
        }
        if tokens.is_empty() {
            continue;
        }
        let mut active_caches: Vec<Vec<KvCache>> = active
            .iter()
            .map(|&s| std::mem::take(&mut per_seq_caches[s]))
            .collect();
        let step_logits = decoder_b.forward_multi_seq(&tokens, &positions, &mut active_caches);
        for ((&s, caches), logits) in active.iter().zip(active_caches).zip(step_logits) {
            per_seq_caches[s] = caches;
            batched_logits[s] = logits;
        }
    }

    assert_eq!(batched_logits.len(), independent_logits.len());
    for (s, (seq_logits, batch_logits)) in independent_logits
        .iter()
        .zip(batched_logits.iter())
        .enumerate()
    {
        assert_eq!(seq_logits.len(), batch_logits.len());
        for (i, (a, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "sequence {s}, logit {i}: independent={a} batched={b}"
            );
        }
    }
}

/// [`the_batched_path_applies_attention_scale_like_the_contiguous_one`]
/// for the fourth host body.
///
/// `forward_multi_seq_kv` did not apply `attention_scale` either, so
/// a served request answered differently the moment it was batched
/// with another request -- the same weights, the same position, a
/// different temperature, decided by how busy the server was.
#[test]
fn the_multi_seq_path_applies_attention_scale_like_the_contiguous_one() {
    let vocab = 8;
    let histories: [&[usize]; 3] = [&[1, 3, 5], &[2, 7], &[4, 4, 4, 6]];
    let next = [6usize, 1, 2];
    let n_layers = 2;
    let scaled = || {
        let mut cfg = tiny_test_config();
        // See the batched twin: a scalar near 1 does not move this
        // model's logits far enough for a tolerance to see it.
        cfg.attention_scale = Some(8.0);
        cfg
    };

    // Builds every sequence's history with `forward_token`, then
    // takes the next step either per sequence or as one batch.
    let run = |cfg: ModelConfig, batched: bool| -> Vec<Vec<f32>> {
        let decoder = Decoder::new_random_small(cfg, n_layers, vocab);
        let mut per_seq: Vec<Vec<KvCache>> = histories
            .iter()
            .map(|h| {
                let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
                for (pos, &tok) in h.iter().enumerate() {
                    decoder.forward_token(tok, pos, &mut caches);
                }
                caches
            })
            .collect();
        let positions: Vec<usize> = histories.iter().map(|h| h.len()).collect();
        if batched {
            decoder.forward_multi_seq(&next, &positions, &mut per_seq)
        } else {
            next.iter()
                .zip(positions.iter())
                .zip(per_seq.iter_mut())
                .map(|((&tok, &pos), caches)| decoder.forward_token(tok, pos, caches))
                .collect()
        }
    };

    let want = run(scaled(), false);
    let got = run(scaled(), true);
    let unscaled = run(tiny_test_config(), true);

    assert!(
        got.iter()
            .zip(unscaled.iter())
            .any(|(g, u)| g.iter().zip(u.iter()).any(|(a, b)| (a - b).abs() > 1e-3)),
        "attention_scale must change the multi-seq answer, or this test cannot fail"
    );

    assert_eq!(want.len(), got.len());
    for (s, (a, b)) in want.iter().zip(got.iter()).enumerate() {
        assert_eq!(a.len(), b.len(), "sequence {s}: logit count");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "sequence {s}, logit {i}: independent={x} batched={y}"
            );
        }
    }
}
