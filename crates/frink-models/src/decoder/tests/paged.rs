//! The paged-KV paths against their contiguous twins: decode, prefill,
//! shared stores, and the refusal when the stores cannot hold a step.

use super::*;

/// `forward_token_paged` must produce bit-identical output to
/// `forward_token` across a multi-step decode (each layer's paged
/// store sized generously so no layer ever exhausts its blocks) --
/// the block-table indirection is a storage-layout detail, not a
/// math change.
#[test]
fn forward_token_paged_matches_forward_token_bit_identical() {
    paged_matches_contiguous(tiny_test_config());
}

/// Every arm of the attention dispatch, not just the plain one.
///
/// The paged path used to implement only full causal attention, and
/// `forward_token_paged` asserted rather than run gpt-oss, because a
/// missing sink term would have changed the distribution silently.
/// Now that it mirrors all three arms, each one has to be held to
/// the same bar the plain arm always was: BIT-identical, not close.
///
/// A sliding window and a softcap are both driven from the config
/// here, so a future edit that wires one arm and forgets another
/// fails on the arm it forgot rather than on a model nobody tests.
#[test]
fn every_paged_attention_arm_is_bit_identical_to_its_contiguous_twin() {
    let windowed = || {
        let mut cfg = tiny_test_config();
        // Smaller than the decode length below, so the window really
        // drops positions rather than degenerating to full causal.
        cfg.sliding_window = Some(2);
        cfg.swa_layers = crate::swa_layers::SwaLayers::All;
        cfg
    };
    let softcapped = || {
        let mut cfg = tiny_test_config();
        // Small enough that `sc * tanh(s / sc)` actually compresses.
        // A realistic 30.0 is numerically indistinguishable from no
        // cap at these tiny weights, so a test using it would pass
        // whether or not the arm was wired -- checked by breaking
        // the arm on purpose and watching it still pass.
        cfg.attn_logit_softcap = Some(0.05);
        cfg
    };
    let both = || {
        let mut cfg = windowed();
        cfg.attn_logit_softcap = Some(0.05);
        cfg
    };
    // Alternating window/full layers: the per-layer arm choice has
    // to be honoured per layer, not decided once for the model.
    let alternating = || {
        let mut cfg = tiny_test_config();
        cfg.sliding_window = Some(2);
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(2, false);
        cfg
    };

    for cfg in [windowed(), softcapped(), both(), alternating()] {
        paged_matches_contiguous(cfg);
    }
}

/// Five MORE model features the paged path had lost the same way
/// the first five went: by being a copy of the contiguous loop that
/// nothing forced to stay in step.
///
/// Found by running Gemma-2-2B through paged KV and watching it
/// answer differently from the same model on the same backend with
/// a contiguous cache -- on CPU, with no GPU involved at all. None
/// of the arm tests above could see it, because `tiny_test_config`
/// sets none of these and `new_random_small` builds every layer
/// without the two sandwich norms.
///
/// - `attention_scale`: Gemma scales Q itself and asks the kernel
///   for a score scale of 1.0, so the built-in `1/sqrt(head_dim)`
///   has to be compensated for. Missing, the model answers at a
///   different temperature.
/// - `post_attn_norm` / `post_ffn_norm`: Gemma-2's sandwich norms,
///   applied to each branch before it rejoins the residual.
/// - gpt-oss's `o_bias`, and its own FFN (`gpt_oss_ffn`, which
///   biases the router and runs the clamped OAI SwiGLU) instead of
///   the generic one.
///
/// Every one of them produces a plausible distribution rather than
/// an error, which is exactly why they are pinned rather than
/// trusted. Values are chosen so each really bites: a scale of 1.0
/// or an all-ones norm would let this pass either way.
#[test]
fn the_paged_path_keeps_every_per_layer_feature_the_contiguous_one_applies() {
    // Gemma's query pre-attention scalar, well away from the
    // kernel's own 1/sqrt(head_dim).
    let mut scaled = tiny_test_config();
    scaled.attention_scale = Some(0.37);
    paged_matches_contiguous_with(scaled, |_| {});

    // Sandwich norms, one at a time and then together, so a wired
    // half is not covered for by the other.
    for (attn, ffn) in [(true, false), (false, true), (true, true)] {
        paged_matches_contiguous_with(tiny_test_config(), with_sandwich_norms(attn, ffn));
    }

    // gpt-oss: the O bias and the OAI FFN, which the paged path was
    // substituting the generic router+SwiGLU for.
    paged_matches_contiguous_with(tiny_test_config(), with_gpt_oss_graph);
}

/// The two rules that live OUTSIDE the layer loop, which the arm
/// test above cannot reach.
///
/// The paged path had drifted from the contiguous one at both ends
/// of the stack, and neither drift was visible to any existing test
/// because `tiny_test_config` sets neither field:
///
/// - it called `embedding.dequant_row` directly instead of scaling
///   the row by `embedding_scale`, so every Gemma token entered the
///   stack `sqrt(hidden_dim)` times too small;
/// - it returned `output_head.apply(..)` raw instead of applying
///   `final_logit_softcap`, so Gemma-2's 30.0 cap never ran.
///
/// Both produce a plausible distribution rather than an error, which
/// is the whole reason to pin them: a wrong answer that still looks
/// like an answer is what a parity test is for. Values here are
/// chosen so each one actually bites -- a scale of 1.0 or a cap far
/// above the logit range would let this pass either way.
#[test]
fn the_paged_path_scales_embeddings_and_softcaps_logits_like_the_contiguous_one() {
    let scaled = || {
        let mut cfg = tiny_test_config();
        cfg.embedding_scale = Some(7.5);
        cfg
    };
    let capped = || {
        let mut cfg = tiny_test_config();
        // Small enough that `sc * tanh(x / sc)` really compresses at
        // this model's logit magnitudes, on the same reasoning as
        // the attention softcap above.
        cfg.final_logit_softcap = Some(0.05);
        cfg
    };
    let both = || {
        let mut cfg = scaled();
        cfg.final_logit_softcap = Some(0.05);
        cfg
    };

    for cfg in [scaled(), capped(), both()] {
        paged_matches_contiguous(cfg);
    }
}

/// Paged prefill must agree with contiguous prefill, and must leave
/// the KV in a state a paged DECODE can continue from.
///
/// The second half is the one worth having. `forward_batch_last`
/// returns only the last row's logits, so a gather/scatter that
/// mangled the KV -- wrote the rows in the wrong order, dropped the
/// part-full tail block, mis-sized a copy -- could still return the
/// right logits for THIS call and only surface on the next token.
/// Decoding four more tokens after the prefill is what makes the
/// stored KV observable, so both paths are compared over the whole
/// continuation rather than at the seam.
///
/// A block size of 2 against a 5-token prompt is deliberate: it
/// leaves the tail block part-full, which is the case
/// `blocks_needed_for` exists for and the one a `n / block_size`
/// reservation would get wrong.
fn paged_prefill_matches_contiguous(config: ModelConfig) {
    let n_layers = 2;
    let mut decoder = host_decoder(config, n_layers, 10);
    // Evicting, whatever the shell says: the paged prefill gathers into a
    // contiguous scratch copy and scatters it back by offset, so a
    // windowed arm here is what proves the scratch never evicts (#61).
    // The contiguous reference evicts too, which is the default run.
    decoder.kv_window = crate::decoder::KvWindowPolicy::on();
    let prompt = [3usize, 1, 4, 1, 5];
    let continuation = [9usize, 2, 6, 5];

    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    let mut plain = vec![decoder.forward_batch_last(&prompt, 0, &mut caches)];
    for (i, &tok) in continuation.iter().enumerate() {
        plain.push(decoder.forward_token(tok, prompt.len() + i, &mut caches));
    }

    let mut paged_caches: Vec<PagedKvCache> = (0..n_layers).map(|_| PagedKvCache::new()).collect();
    let stores = SharedPagedKv::from_stores(
        (0..n_layers)
            .map(|_| {
                PagedKvStore::new(
                    /* block_size = */ 2,
                    /* total_blocks = */ 16,
                    decoder.config.n_kv_heads,
                    decoder.config.head_dim,
                )
            })
            .collect(),
    );
    let mut paged = vec![decoder
        .forward_batch_last_paged(&prompt, 0, &mut paged_caches, &stores)
        .expect("store sized generously, must not exhaust")];
    for (i, &tok) in continuation.iter().enumerate() {
        paged.push(
            decoder
                .forward_token_paged(tok, prompt.len() + i, &mut paged_caches, &stores)
                .expect("store sized generously, must not exhaust"),
        );
    }

    assert_eq!(
        paged_caches[0].seq_len(),
        prompt.len() + continuation.len(),
        "paged prefill must advance seq_len by exactly the batch size"
    );
    assert_eq!(plain.len(), paged.len());
    for (step, (a, b)) in plain.iter().zip(paged.iter()).enumerate() {
        assert_eq!(a.len(), b.len(), "step {step}: logit count");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "step {step}: paged prefill + decode must be bit-identical to contiguous"
            );
        }
    }
}

/// Every arm again, this time through the prefill entry point. The
/// gather is shared, but the kernel the gathered buffer reaches is
/// the BLOCKED prefill one rather than the per-query decode one, so
/// arm coverage here is not implied by the decode tests above.
#[test]
fn paged_prefill_is_bit_identical_across_every_arm() {
    let windowed = || {
        let mut cfg = tiny_test_config();
        cfg.sliding_window = Some(2);
        cfg.swa_layers = crate::swa_layers::SwaLayers::All;
        cfg
    };
    let scaled_and_capped = || {
        let mut cfg = tiny_test_config();
        cfg.embedding_scale = Some(7.5);
        cfg.final_logit_softcap = Some(0.05);
        cfg.attn_logit_softcap = Some(0.05);
        cfg
    };
    let alternating = || {
        let mut cfg = tiny_test_config();
        cfg.sliding_window = Some(2);
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(2, false);
        cfg
    };

    for cfg in [
        tiny_test_config(),
        windowed(),
        scaled_and_capped(),
        alternating(),
    ] {
        paged_prefill_matches_contiguous(cfg);
    }
}

/// A prefill the stores cannot hold refuses having written NOTHING
/// -- checked on the case that actually needs the up-front loop.
///
/// Each layer owns its own store, so layer 0 having room says
/// nothing about layer 1. `append_contiguous` already refuses
/// rather than half-writing a single layer, so a test whose layers
/// are sized alike passes with the cross-layer reservation deleted
/// -- it would be asserting a property it never exercises. Here
/// layer 0 has room for the whole prompt and layer 1 does not, so
/// without the up-front check layer 0 is written, layer 1 refuses,
/// and the sequence ends up with its layers at DIFFERENT lengths.
/// No caller can recover from that, and nothing downstream would
/// report it: the next decode step simply attends over a shorter
/// history in one layer than the others.
///
/// Verified by deleting the reservation loop and watching this fail
/// on `layer 1 must be untouched`.
/// Three requests sharing one set of per-layer stores must get
/// exactly what they would get alone.
///
/// This is the property the RwLock exists for, and it cannot be
/// asserted single-threaded. Every request writes only blocks it
/// owns, so sharing changes where rows live and nothing else --
/// bit-identical, not close. A store that let one request's rows
/// land in another's blocks shows up here and nowhere else.
#[test]
fn concurrent_decodes_against_one_shared_store_match_running_them_alone() {
    use std::sync::Arc;

    let decoder = Arc::new(host_decoder(tiny_test_config(), 2, 10));
    let prompts: [&[usize]; 3] = [&[3, 1, 4], &[1, 5, 9], &[2, 6, 5]];
    let continuation = [7usize, 8, 3];

    // Each request run alone, against its own store, is the answer
    // sharing must not change.
    let solo: Vec<Vec<Vec<f32>>> = prompts
        .iter()
        .map(|prompt| {
            let stores =
                SharedPagedKv::new(2, 4, 32, decoder.config.n_kv_heads, decoder.config.head_dim);
            let mut caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
            run_one(&decoder, prompt, &continuation, &mut caches, &stores)
        })
        .collect();

    // The same three, concurrently, sharing ONE set of per-layer
    // stores. Every request writes only blocks it owns, so the
    // answers must be identical -- not close, identical. A store
    // that let one request's rows land in another's blocks would
    // show up here and nowhere else.
    let shared = Arc::new(SharedPagedKv::new(
        2,
        4,
        96,
        decoder.config.n_kv_heads,
        decoder.config.head_dim,
    ));
    let together: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = prompts
            .iter()
            .map(|prompt| {
                let decoder = Arc::clone(&decoder);
                let shared = Arc::clone(&shared);
                scope.spawn(move || {
                    let mut caches: Vec<PagedKvCache> =
                        (0..2).map(|_| PagedKvCache::new()).collect();
                    run_one(&decoder, prompt, &continuation, &mut caches, &shared)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    for (r, (alone, concurrent)) in solo.iter().zip(together.iter()).enumerate() {
        assert_eq!(alone.len(), concurrent.len(), "request {r}: step count");
        for (step, (a, b)) in alone.iter().zip(concurrent.iter()).enumerate() {
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "request {r} step {step}: sharing a store changed the answer"
                );
            }
        }
    }
}

/// Prefill then decode, returning every step's logits.
fn run_one(
    decoder: &Decoder,
    prompt: &[usize],
    continuation: &[usize],
    caches: &mut [PagedKvCache],
    stores: &SharedPagedKv,
) -> Vec<Vec<f32>> {
    let mut out = vec![decoder
        .forward_batch_last_paged(prompt, 0, caches, stores)
        .expect("sized generously")];
    for (i, &tok) in continuation.iter().enumerate() {
        out.push(
            decoder
                .forward_token_paged(tok, prompt.len() + i, caches, stores)
                .expect("sized generously"),
        );
    }
    out
}

/// A decode step the stores cannot hold advances NO layer.
///
/// This was a real defect until the reservation moved into
/// `forward_token_paged`: it pushed per layer with `?`, so a store
/// exhausting at layer 1 of 2 left layer 0 holding a position layer
/// 1 did not. Nothing downstream reports that -- the next step just
/// attends over a shorter history in the tail layers -- and the
/// prefill path had the guard while decode never did.
///
/// Layer 0 is given room and layer 1 none, so the bug is reachable:
/// with the reservation removed, layer 0 advances and layer 1
/// refuses.
#[test]
fn a_decode_step_the_stores_cannot_hold_advances_no_layer() {
    let decoder = host_decoder(tiny_test_config(), 2, 10);
    let mut caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
    // Block size 1 so "one more position" always needs a block.
    // Layer 0 gets two, layer 1 exactly one: the prompt fills layer
    // 1 completely, so the decode step below cannot fit there.
    let stores = SharedPagedKv::from_stores(
        [2usize, 1]
            .into_iter()
            .map(|blocks| {
                PagedKvStore::new(
                    1,
                    blocks,
                    decoder.config.n_kv_heads,
                    decoder.config.head_dim,
                )
            })
            .collect(),
    );

    decoder
        .forward_batch_last_paged(&[1usize], 0, &mut caches, &stores)
        .expect("one position fits in both layers");
    assert_eq!(caches[0].seq_len(), 1);
    assert_eq!(caches[1].seq_len(), 1);

    let result = decoder.forward_token_paged(2, 1, &mut caches, &stores);
    assert!(result.is_err(), "layer 1 has no block left");
    assert_eq!(
        caches[0].seq_len(),
        1,
        "layer 0 must not advance past a layer that could not"
    );
    assert_eq!(caches[1].seq_len(), 1);
}

#[test]
fn a_prefill_the_stores_cannot_hold_refuses_before_writing_any_layer() {
    let decoder = host_decoder(tiny_test_config(), 2, 10);
    let prompt = [1usize, 2, 3, 4, 5, 6];
    let mut paged_caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
    // Layer 0 fits the prompt with room to spare; layer 1's two
    // blocks of 2 hold 4 positions against a prompt of 6.
    let stores = SharedPagedKv::from_stores(
        [8usize, 2]
            .into_iter()
            .map(|blocks| {
                PagedKvStore::new(
                    2,
                    blocks,
                    decoder.config.n_kv_heads,
                    decoder.config.head_dim,
                )
            })
            .collect(),
    );

    let result = decoder.forward_batch_last_paged(&prompt, 0, &mut paged_caches, &stores);
    assert!(result.is_err(), "layer 1's store cannot hold the prompt");
    for (i, cache) in paged_caches.iter().enumerate() {
        assert_eq!(cache.seq_len(), 0, "layer {i} must be untouched");
        assert!(cache.block_table().is_empty(), "layer {i} holds no block");
    }
    for (i, expected) in [8usize, 2].into_iter().enumerate() {
        assert_eq!(stores.free_blocks(i), expected, "layer {i} leaked no block");
    }
}

/// Chunked prefill: two calls appending into the same sequence must
/// equal one call over the concatenation.
///
/// This is the case the part-full tail block breaks if
/// `to_contiguous` or the reservation is wrong, and it is how the
/// serving path actually prefills long prompts.
#[test]
fn two_paged_prefill_chunks_equal_one_call_over_the_whole_prompt() {
    let decoder = host_decoder(tiny_test_config(), 2, 10);
    let prompt = [3usize, 1, 4, 1, 5, 9, 2];
    let split = 3;

    let run = |chunks: &[&[usize]]| {
        let mut caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
        let stores = SharedPagedKv::from_stores(
            (0..2)
                .map(|_| {
                    PagedKvStore::new(2, 16, decoder.config.n_kv_heads, decoder.config.head_dim)
                })
                .collect(),
        );
        let mut pos = 0;
        let mut last = Vec::new();
        for chunk in chunks {
            last = decoder
                .forward_batch_last_paged(chunk, pos, &mut caches, &stores)
                .expect("sized generously");
            pos += chunk.len();
        }
        last
    };

    let whole = run(&[&prompt]);
    let chunked = run(&[&prompt[..split], &prompt[split..]]);
    assert_eq!(whole.len(), chunked.len());
    for (x, y) in whole.iter().zip(chunked.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "a chunked prefill must equal one call over the same tokens"
        );
    }
}

fn paged_matches_contiguous(config: ModelConfig) {
    paged_matches_contiguous_with(config, |_| {});
}

/// [`paged_matches_contiguous`] for the features that live on the
/// WEIGHTS rather than in the config, and so cannot be switched on
/// by handing a different `ModelConfig` in.
fn paged_matches_contiguous_with(config: ModelConfig, prepare: impl FnOnce(&mut Decoder)) {
    let n_layers = 2;
    let mut decoder = host_decoder(config, n_layers, 10);
    prepare(&mut decoder);
    let decoder = decoder;

    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    let steps = [3usize, 5, 7, 2, 9, 1];
    let mut plain_logits = Vec::new();
    for (pos, &tok) in steps.iter().enumerate() {
        plain_logits.push(decoder.forward_token(tok, pos, &mut caches));
    }

    let block_size = 2;
    let mut paged_caches: Vec<PagedKvCache> = (0..n_layers).map(|_| PagedKvCache::new()).collect();
    let stores = SharedPagedKv::from_stores(
        (0..n_layers)
            .map(|_| {
                PagedKvStore::new(
                    block_size,
                    /* total_blocks = */ 16,
                    decoder.config.n_kv_heads,
                    decoder.config.head_dim,
                )
            })
            .collect(),
    );
    let mut paged_logits = Vec::new();
    for (pos, &tok) in steps.iter().enumerate() {
        paged_logits.push(
            decoder
                .forward_token_paged(tok, pos, &mut paged_caches, &stores)
                .expect("store sized generously, must not exhaust"),
        );
    }

    assert_eq!(plain_logits.len(), paged_logits.len());
    for (a, b) in plain_logits.iter().zip(paged_logits.iter()) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "paged decode must be bit-identical to contiguous decode"
            );
        }
    }
}
