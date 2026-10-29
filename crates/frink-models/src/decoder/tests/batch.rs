//! The batched prefill path against the token path: every per-layer
//! feature, the attention scale, QK norm and QKV bias, and the KV state
//! it leaves behind.

use super::*;

/// The same feature list as
/// [`the_paged_path_keeps_every_per_layer_feature_the_contiguous_one_applies`],
/// checked against `forward_hidden_batch_inner` instead.
///
/// Necessary because `forward_token` and `forward_token_paged` now
/// share ONE body (`Decoder::attn_block`) that differs only in its
/// `KvStep`, so the paged test can no longer see a decoration
/// dropped from that body -- deleting `post_attn_norm` or gpt-oss's
/// `o_bias` from it leaves the whole suite green, which was measured
/// rather than assumed. `forward_hidden_batch_inner` is deliberately
/// NOT collapsed into the same body, so it is the independent
/// ground truth that keeps these features pinned.
#[test]
fn the_batched_path_keeps_every_per_layer_feature_the_token_path_applies() {
    for (attn, ffn) in [(true, false), (false, true), (true, true)] {
        batched_matches_contiguous_with(tiny_test_config(), with_sandwich_norms(attn, ffn));
    }
    batched_matches_contiguous_with(tiny_test_config(), with_gpt_oss_graph);
}

/// [`paged_matches_contiguous_with`] for `forward_batch` against
/// sequential `forward_token`.
///
/// Not bit-identity: batched prefill runs the blocked three-pass
/// softmax while decode keeps the online accumulator, so the two
/// agree to a tolerance rather than to the bit -- the same reason
/// `decoder_via_engine_trait_matches_forward_batch_ground_truth`
/// gives. 1e-5 is four orders below the ~1e-1 a dropped decoration
/// moves these logits by.
fn batched_matches_contiguous_with(config: ModelConfig, prepare: impl Fn(&mut Decoder)) {
    let n_layers = 2;
    let vocab = 10;
    let tokens = [3usize, 5, 7, 2, 9, 1];

    let mut seq_decoder = host_decoder(config.clone(), n_layers, vocab);
    prepare(&mut seq_decoder);
    let mut seq_caches: Vec<KvCache> = seq_decoder.config.new_kv_caches();
    let sequential: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &t)| seq_decoder.forward_token(t, pos, &mut seq_caches))
        .collect();

    // Same seed -> identical weights before `prepare`, and `prepare`
    // is deterministic, so this is a like-for-like comparison.
    let mut batch_decoder = host_decoder(config, n_layers, vocab);
    prepare(&mut batch_decoder);
    let mut batch_caches: Vec<KvCache> = batch_decoder.config.new_kv_caches();
    let batched = batch_decoder.forward_batch(&tokens, 0, &mut batch_caches);

    assert_eq!(sequential.len(), batched.len());
    for (pos, (a, b)) in sequential.iter().zip(batched.iter()).enumerate() {
        assert_eq!(a.len(), b.len(), "position {pos}: logit count");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "position {pos}, logit {i}: token path={x} batched={y}"
            );
        }
    }
}

/// Batched and sequential agree to about one f32 ulp, NOT exactly.
///
/// The name used to say "exactly" and the assertion allowed 1e-3,
/// which was wrong in both directions at once: the two paths are
/// not bit-identical (the sibling tests that really are say
/// `bit_identical`, and mean it), and the real difference is five
/// orders tighter than the bound that was nominally checking it.
/// Measured over this fixture: the largest deviation is 1.5e-8,
/// single-ulp on values around 0.05, arising from the reduction
/// order of a batched matmul against a per-token one.
///
/// The bound is 1e-6 now: loose enough for that ulp and for a
/// different machine's reduction order, tight enough that an
/// actual drift in the batched path cannot hide under it. This
/// matters beyond hygiene -- the server verifies speculated tokens
/// against logits from THIS function, so a near-tie inside this
/// margin is the one case where a speculated token can disagree
/// with what sequential decoding would have drawn.
#[test]
fn forward_batch_matches_sequential_forward_token_to_one_ulp() {
    let cfg = tiny_test_config();
    let vocab = 8;
    let tokens = [1usize, 3, 5, 2, 7];

    let decoder_a = host_decoder(cfg.clone(), 2, vocab);
    let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
    let sequential: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
        .collect();

    // A second decoder built with the same seed produces identical
    // weights (Decoder::new_random_small is deterministic), so
    // this is a fair like-for-like comparison against a fresh
    // cache rather than reusing decoder_a's now-mutated cache.
    let decoder_b = host_decoder(cfg, 2, vocab);
    let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
    let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

    assert_eq!(batched.len(), sequential.len());
    for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
        assert_eq!(seq_logits.len(), batch_logits.len());
        for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
            assert!(
                (s - b).abs() < 1e-6,
                "position {pos}, logit {i}: sequential={s} batched={b}"
            );
        }
    }
}

/// `forward_batch_last` exists to skip the vocabulary projection for
/// every position but the last, so the one thing that must hold is
/// that the row it *does* produce is the same row `forward_batch`
/// would have produced. It must also leave the KV cache in the same
/// state -- prefill's whole purpose -- which is checked by decoding
/// one more token from each cache and comparing.
#[test]
fn forward_batch_last_matches_the_final_row_of_forward_batch() {
    let cfg = tiny_test_config();
    let vocab = 16;
    let tokens = vec![1usize, 4, 7, 2, 9];

    let decoder_a = host_decoder(cfg.clone(), 2, vocab);
    let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
    let all_rows = decoder_a.forward_batch(&tokens, 0, &mut caches_a);

    let decoder_b = host_decoder(cfg, 2, vocab);
    let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
    let last = decoder_b.forward_batch_last(&tokens, 0, &mut caches_b);

    let expected = all_rows.last().expect("one row per prompt token");
    assert_eq!(last.len(), expected.len());
    for (i, (a, b)) in expected.iter().zip(last.iter()).enumerate() {
        assert!(
            a == b,
            "logit {i}: forward_batch={a} forward_batch_last={b}"
        );
    }

    // Same KV state: the next token's logits must agree too.
    let next_a = decoder_a.forward_token(3, tokens.len(), &mut caches_a);
    let next_b = decoder_b.forward_token(3, tokens.len(), &mut caches_b);
    for (i, (a, b)) in next_a.iter().zip(next_b.iter()).enumerate() {
        assert!(a == b, "post-prefill decode logit {i}: {a} vs {b}");
    }

    // Empty prompt is the degenerate case both paths must survive.
    let mut caches_c: Vec<KvCache> = decoder_b.config.new_kv_caches();
    assert!(decoder_b
        .forward_batch_last(&[], 0, &mut caches_c)
        .is_empty());
}

#[test]
fn the_batched_path_applies_attention_scale_like_the_contiguous_one() {
    let vocab = 8;
    let tokens = [1usize, 3, 5, 2, 7];
    let scaled = || {
        let mut cfg = tiny_test_config();
        // Far from the kernel's own 1/sqrt(head_dim) on purpose:
        // at this model's scale a scalar near 1 moves the logits by
        // ~2e-4, which is below the noise a tolerance test can see.
        cfg.attention_scale = Some(8.0);
        cfg
    };
    let fresh_caches = |d: &Decoder| -> Vec<KvCache> { d.config.new_kv_caches() };

    // Same seed -> identical weights, so the only difference between
    // these three decoders is the config field under test.
    let seq_decoder = host_decoder(scaled(), 2, vocab);
    let mut seq_caches = fresh_caches(&seq_decoder);
    let sequential: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &t)| seq_decoder.forward_token(t, pos, &mut seq_caches))
        .collect();

    let batch_decoder = host_decoder(scaled(), 2, vocab);
    let mut batch_caches = fresh_caches(&batch_decoder);
    let batched = batch_decoder.forward_batch(&tokens, 0, &mut batch_caches);

    let plain_decoder = host_decoder(tiny_test_config(), 2, vocab);
    let mut plain_caches = fresh_caches(&plain_decoder);
    let unscaled = plain_decoder.forward_batch(&tokens, 0, &mut plain_caches);
    assert!(
        batched
            .iter()
            .zip(unscaled.iter())
            .any(|(s, u)| s.iter().zip(u.iter()).any(|(a, b)| (a - b).abs() > 1e-3)),
        "attention_scale must change the batched answer, or this test cannot fail"
    );

    assert_eq!(batched.len(), sequential.len());
    for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
        assert_eq!(seq_logits.len(), batch_logits.len());
        for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
            assert!(
                (s - b).abs() < 1e-5,
                "position {pos}, logit {i}: sequential={s} batched={b}"
            );
        }
    }
}

/// OLMoE-style QK-norm (`attn_q_norm`/`attn_k_norm`, see `AttnWeights`'
/// doc comment): with both set, `forward_batch` must still match
/// sequential `forward_token` calls exactly -- the same consistency
/// property `forward_batch_matches_sequential_forward_token_exactly`
/// checks for the no-QK-norm path, now exercising the norm-applied
/// per-row slicing (`q_batch.chunks_mut(q_width)`,
/// `k_batch.chunks_mut(kv_width)`) instead of trusting it by
/// inspection.
#[test]
fn forward_batch_matches_forward_token_with_qk_norm_present() {
    let cfg = tiny_test_config();
    let vocab = 8;
    let tokens = [1usize, 3, 5, 2, 7];
    let q_width = cfg.n_heads * cfg.head_dim;
    let kv_width = cfg.n_kv_heads * cfg.head_dim;

    let mut decoder_a = host_decoder(cfg.clone(), 2, vocab);
    for layer in &mut decoder_a.layers {
        layer.attn.q_norm = Some((0..q_width).map(|i| 1.0 + i as f32 * 0.1).collect());
        layer.attn.k_norm = Some((0..kv_width).map(|i| 0.5 + i as f32 * 0.05).collect());
    }
    let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
    let sequential: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
        .collect();

    let mut decoder_b = host_decoder(cfg, 2, vocab);
    for layer in &mut decoder_b.layers {
        layer.attn.q_norm = Some((0..q_width).map(|i| 1.0 + i as f32 * 0.1).collect());
        layer.attn.k_norm = Some((0..kv_width).map(|i| 0.5 + i as f32 * 0.05).collect());
    }
    let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
    let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

    assert_eq!(batched.len(), sequential.len());
    for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
        for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
            assert!(
                (s - b).abs() < BATCH_VS_SEQUENTIAL_ULP,
                "position {pos}, logit {i}: sequential={s} batched={b}"
            );
        }
    }
}

/// QK-norm being present must actually change the output -- otherwise
/// the `Some(...)` branches in `forward_token`/`forward_batch` could
/// silently be dead code and this feature would ship unverified. Must
/// decode at least 2 positions: at position 0 with a fresh cache,
/// causal softmax has exactly one candidate (the token attending to
/// itself) and always evaluates to weight 1.0 regardless of the Q*K
/// dot product -- so the attention output there is Q/K-invariant by
/// construction, and a single-position version of this test would
/// pass even with `q_norm`/`k_norm` silently never applied.
#[test]
fn qk_norm_present_changes_output_versus_absent() {
    let cfg = tiny_test_config();
    let vocab = 8;
    let q_width = cfg.n_heads * cfg.head_dim;
    let kv_width = cfg.n_kv_heads * cfg.head_dim;
    let tokens = [3usize, 5];

    let without_norm = host_decoder(cfg.clone(), 1, vocab);
    let mut with_norm = host_decoder(cfg, 1, vocab);
    for layer in &mut with_norm.layers {
        layer.attn.q_norm = Some(vec![2.0; q_width]);
        layer.attn.k_norm = Some(vec![2.0; kv_width]);
    }

    let mut caches_a: Vec<KvCache> = without_norm.config.new_kv_caches();
    let mut caches_b: Vec<KvCache> = with_norm.config.new_kv_caches();

    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        out_a = without_norm.forward_token(t, pos, &mut caches_a);
        out_b = with_norm.forward_token(t, pos, &mut caches_b);
    }

    let differs = out_a
        .iter()
        .zip(out_b.iter())
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(
        differs,
        "QK-norm weights changed nothing -- forward_token likely isn't applying q_norm/k_norm"
    );
}

/// Qwen2/Qwen2-MoE-family QKV attention bias (`AttnWeights::q_bias`/
/// `k_bias`/`v_bias`): a real, previously-unhandled gap found by
/// running frink's generic GGUF loader against a real downloaded
/// Qwen1.5-MoE-A2.7B-Chat checkpoint, which produced fluent-but-wrong
/// output because these real `attn_{q,k,v}.bias` tensors were
/// silently never added anywhere. Same two real properties checked
/// as the QK-norm tests above: (1) `forward_batch` must match
/// sequential `forward_token` exactly with bias present (batched
/// per-row broadcast must be correct, not just the single-token
/// path), and (2) bias must actually change the output at position
/// 0 or later (not silently dead code) -- checked at position 1
/// specifically, since position 0's causal softmax has exactly one
/// candidate and is Q/K-invariant regardless of any additive bias
/// shifting Q/K, for the same reason the QK-norm test above needs
/// >=2 positions.
#[test]
fn forward_batch_matches_forward_token_with_qkv_bias_present() {
    let cfg = tiny_test_config();
    let vocab = 8;
    let tokens = [1usize, 3, 5, 2, 7];
    let q_width = cfg.n_heads * cfg.head_dim;
    let kv_width = cfg.n_kv_heads * cfg.head_dim;

    let mut decoder_a = host_decoder(cfg.clone(), 2, vocab);
    for layer in &mut decoder_a.layers {
        layer.attn.q_bias = Some((0..q_width).map(|i| 0.3 + i as f32 * 0.02).collect());
        layer.attn.k_bias = Some((0..kv_width).map(|i| -0.2 + i as f32 * 0.03).collect());
        layer.attn.v_bias = Some((0..kv_width).map(|i| 0.1 - i as f32 * 0.01).collect());
    }
    let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
    let sequential: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
        .collect();

    let mut decoder_b = host_decoder(cfg, 2, vocab);
    for layer in &mut decoder_b.layers {
        layer.attn.q_bias = Some((0..q_width).map(|i| 0.3 + i as f32 * 0.02).collect());
        layer.attn.k_bias = Some((0..kv_width).map(|i| -0.2 + i as f32 * 0.03).collect());
        layer.attn.v_bias = Some((0..kv_width).map(|i| 0.1 - i as f32 * 0.01).collect());
    }
    let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
    let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

    assert_eq!(batched.len(), sequential.len());
    for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
        for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
            assert!(
                (s - b).abs() < BATCH_VS_SEQUENTIAL_ULP,
                "position {pos}, logit {i}: sequential={s} batched={b}"
            );
        }
    }
}

#[test]
fn qkv_bias_present_changes_output_versus_absent() {
    let cfg = tiny_test_config();
    let vocab = 8;
    let q_width = cfg.n_heads * cfg.head_dim;
    let kv_width = cfg.n_kv_heads * cfg.head_dim;
    let tokens = [3usize, 5];

    let without_bias = host_decoder(cfg.clone(), 1, vocab);
    let mut with_bias = host_decoder(cfg, 1, vocab);
    for layer in &mut with_bias.layers {
        layer.attn.q_bias = Some(vec![0.5; q_width]);
        layer.attn.k_bias = Some(vec![0.5; kv_width]);
        layer.attn.v_bias = Some(vec![0.5; kv_width]);
    }

    let mut caches_a: Vec<KvCache> = without_bias.config.new_kv_caches();
    let mut caches_b: Vec<KvCache> = with_bias.config.new_kv_caches();

    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        out_a = without_bias.forward_token(t, pos, &mut caches_a);
        out_b = with_bias.forward_token(t, pos, &mut caches_b);
    }

    let differs = out_a
        .iter()
        .zip(out_b.iter())
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(
        differs,
        "QKV bias changed nothing -- forward_token likely isn't applying q_bias/k_bias/v_bias"
    );
}

#[test]
fn forward_batch_and_forward_token_leave_kv_caches_in_the_same_state() {
    let cfg = tiny_test_config();
    let tokens = [2usize, 4, 6];

    let decoder_a = host_decoder(cfg.clone(), 2, 8);
    let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
    for (pos, &t) in tokens.iter().enumerate() {
        decoder_a.forward_token(t, pos, &mut caches_a);
    }

    let decoder_b = host_decoder(cfg, 2, 8);
    let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
    decoder_b.forward_batch(&tokens, 0, &mut caches_b);

    for (ca, cb) in caches_a.iter().zip(caches_b.iter()) {
        assert_eq!(ca.positions(), cb.positions());
        assert_eq!(ca.k.len(), cb.k.len());
        for (a, b) in ca.k.to_f32_vec().iter().zip(cb.k.to_f32_vec().iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}

#[test]
fn dense_layer_forward_batch_matches_sequential_forward_token_to_one_ulp() {
    let cfg = tiny_dense_test_config();
    let vocab = 8;
    let tokens = [1usize, 3, 5, 2, 7];

    let decoder_a = host_decoder(cfg.clone(), 2, vocab);
    let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
    let sequential: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
        .collect();

    let decoder_b = host_decoder(cfg, 2, vocab);
    let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
    let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

    assert_eq!(batched.len(), sequential.len());
    for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
        for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
            assert!(
                (s - b).abs() < BATCH_VS_SEQUENTIAL_ULP,
                "position {pos}, logit {i}: sequential={s} batched={b}"
            );
        }
    }
}

#[test]
fn forward_batch_with_empty_tokens_returns_empty() {
    let decoder = host_decoder(tiny_test_config(), 2, 8);
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    let out = decoder.forward_batch(&[], 0, &mut caches);
    assert!(out.is_empty());
}

#[test]
fn forward_batch_continues_correctly_after_prior_forward_token_calls() {
    // Realistic usage pattern: some tokens processed one at a time
    // (e.g. the first generated token), then a batch verifying
    // several draft tokens at once, continuing from the same
    // cache. The batch's positions must be numbered starting from
    // wherever the cache left off, not from zero.
    let cfg = tiny_test_config();

    let decoder_a = host_decoder(cfg.clone(), 2, 8);
    let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
    decoder_a.forward_token(1, 0, &mut caches_a);
    decoder_a.forward_token(3, 1, &mut caches_a);
    let seq_next = decoder_a.forward_token(5, 2, &mut caches_a);

    let decoder_b = host_decoder(cfg, 2, 8);
    let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
    decoder_b.forward_token(1, 0, &mut caches_b);
    let batch_next = decoder_b.forward_batch(&[3, 5], 1, &mut caches_b);

    for (s, b) in seq_next.iter().zip(batch_next[1].iter()) {
        assert!(
            (s - b).abs() < BATCH_VS_SEQUENTIAL_ULP,
            "sequential={s} batched={b}"
        );
    }
}
