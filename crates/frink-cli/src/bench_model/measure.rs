//! `frink bench -m`'s timed loops: one `pp<N>` and one `tg<N>` row per
//! engine, every repetition checked by `bench_guard` before and after.
//! Moved out of `bench_model.rs` unchanged.

use super::{BenchArgs, Row};
use crate::bench_guard::{self, CacheProbe, WorkloadDigest};
use frink_core::cache::KvCache;
use frink_models::engine::Engine;
use frink_models::{Decoder, Gemma4Engine};
use std::time::Instant;

/// `pp<N>` / `tg<N>` for the generic [`Decoder`].
pub(super) fn measure_decoder(decoder: &Decoder, args: &BenchArgs) -> anyhow::Result<Vec<Row>> {
    let mut rows: Vec<Row> = Vec::new();
    if args.n_prompt > 0 {
        rows.push(bench_prefill(decoder, args.n_prompt, args.reps)?);
    }
    if args.n_gen > 0 {
        rows.push(bench_decode(decoder, args.n_gen, args.reps)?);
    }
    for row in &rows {
        bench_guard::check_timed_samples(&row.test, args.reps, row.samples.len())?;
        bench_guard::check_sample_rates(&row.test, &row.samples)?;
    }
    Ok(rows)
}

/// `pp<N>` / `tg<N>` for [`Gemma4Engine`] (sequential tokens only).
pub(super) fn measure_gemma4(engine: &Gemma4Engine, args: &BenchArgs) -> anyhow::Result<Vec<Row>> {
    let mut rows: Vec<Row> = Vec::new();
    if args.n_prompt > 0 {
        rows.push(bench_prefill_gemma4(engine, args.n_prompt, args.reps)?);
    }
    if args.n_gen > 0 {
        rows.push(bench_decode_gemma4(engine, args.n_gen, args.reps)?);
    }
    for row in &rows {
        bench_guard::check_timed_samples(&row.test, args.reps, row.samples.len())?;
        bench_guard::check_sample_rates(&row.test, &row.samples)?;
    }
    Ok(rows)
}

/// The engine-side half of the determinism assertion, threaded through
/// one row's repetitions.
///
/// Also where an empty logit vector is caught: a forward pass that
/// returned nothing cannot be shown to have computed anything, so the
/// duration it took is not a throughput.
pub(crate) fn check_result(
    test: &str,
    rep: usize,
    logits: &[f32],
    first: &mut Option<(usize, f32)>,
) -> anyhow::Result<()> {
    let pick = bench_guard::greedy_pick(logits)?.ok_or_else(|| {
        anyhow::anyhow!(
            "{test} rep {rep}: the forward pass returned no logits, so it cannot be \
             shown to have computed anything and its duration is not a throughput"
        )
    })?;
    let seen = *first.get_or_insert(pick);
    bench_guard::check_same_result(test, rep, seen, pick)
}

/// Snapshots every layer's KV cache for the guards in `bench_guard`.
pub(crate) fn probe(caches: &[KvCache]) -> Vec<CacheProbe> {
    caches.iter().map(probe_one).collect()
}

/// One layer's probe. An attention layer reports its resident ROWS,
/// beside the two buffer lengths they are derived from. A recurrent
/// layer (a Mamba, delta-net or short-conv block: K width zero, so
/// `rows()` is zero forever) has no rows to derive from, and its
/// position counter is the one length the block advances; reading rows
/// there refused every hybrid model's bench as "layer 0 consumed 0 of
/// 128 prompt tokens".
///
/// A windowed layer that evicts (#61, the default) is the other case
/// where rows stop being the length: past `window + slack` positions it
/// holds fewer rows than it consumed, and reading rows there would
/// refuse any row whose prompt passes the window as a run that skipped
/// work. Its position counter is what it consumed.
fn probe_one(c: &KvCache) -> CacheProbe {
    CacheProbe {
        seq_len: if c.k_width() == 0 || c.window().is_some() {
            c.positions()
        } else {
            c.rows()
        },
        k_len: c.k.len(),
        v_len: c.v.len(),
    }
}

/// Same, for Gemma-4's sparse `Vec<Option<KvCache>>` (shared-KV layers
/// leave holes, which are not caches and are not checked).
fn probe_gemma4(state: &frink_models::gemma4_engine::Gemma4DecodeState) -> Vec<CacheProbe> {
    state.kv.iter().flatten().map(probe_one).collect()
}

/// `pp<N>`: one batched forward over N tokens into fresh KV caches.
/// Reported as prompt tokens per second, as `llama-bench` does.
fn bench_prefill(decoder: &Decoder, n_prompt: usize, reps: usize) -> anyhow::Result<Row> {
    let test = format!("pp{n_prompt}");
    let tokens = synthetic_tokens(decoder.config.vocab_size, n_prompt, 0);
    // Asserted BEFORE the run: the stream about to be fed is exactly as
    // long as the row's label promises, and every id is in vocabulary.
    bench_guard::check_prompt_before(&test, n_prompt, &tokens, decoder.config.vocab_size)?;
    let mut out = Vec::with_capacity(reps);
    let mut first_digest: Option<WorkloadDigest> = None;
    let mut first_result: Option<(usize, f32)> = None;
    for rep in 0..reps + bench_guard::WARMUP_REPS {
        let mut caches = fresh_caches(decoder);
        // Every rep starts from an empty KV: a prefill that reused a
        // warm cache would report the engine's speed at work it did
        // not do. Asserted rather than assumed, so a later change to
        // `fresh_caches` cannot silently turn this into a cache-hit
        // benchmark.
        bench_guard::check_caches_cold(&test, rep, &probe(&caches))?;
        let mut digest = WorkloadDigest::new();
        digest.feed_all(&tokens);
        let t = Instant::now();
        // `forward_batch_last`, not `forward_batch`: llama-bench's
        // `pp512` asks for logits at the final position only, so
        // projecting all 512 rows through the vocabulary would be work
        // the reference engine never does.
        let logits = decoder.forward_batch_last(&tokens, 0, &mut caches);
        let dt = t.elapsed().as_secs_f64();
        // The prompt length is asserted AFTER the run too, not only
        // before: an engine that silently truncated the batch would
        // otherwise divide the full token count by a partial run's
        // time and report a speedup for doing less work.
        bench_guard::check_prefill_after(&test, n_prompt, &probe(&caches))?;
        let first = *first_digest.get_or_insert(digest);
        bench_guard::check_same_workload(&test, rep, first, digest)?;
        check_result(&test, rep, &logits, &mut first_result)?;
        // Rep 0 is the warmup: first touch of every weight page, and on
        // Metal the first pipeline compile. Timing it would measure the
        // OS and the shader cache, not the engine.
        if rep >= bench_guard::WARMUP_REPS {
            out.push(n_prompt as f64 / dt);
        }
    }
    Ok(Row {
        test,
        samples: out,
        digest: first_digest.unwrap_or_default(),
    })
}

/// Gemma-4 `pp<N>`: sequential `forward_token` over N synthetic ids.
/// Keeps only the final logits (same work llama-bench reports), but
/// without batched matmuls — honest for today's engine.
fn bench_prefill_gemma4(
    engine: &Gemma4Engine,
    n_prompt: usize,
    reps: usize,
) -> anyhow::Result<Row> {
    let test = format!("pp{n_prompt}");
    let vocab = Engine::vocab_size(engine);
    let tokens = synthetic_tokens(vocab, n_prompt, 0);
    bench_guard::check_prompt_before(&test, n_prompt, &tokens, vocab)?;
    let mut out = Vec::with_capacity(reps);
    let mut first_digest: Option<WorkloadDigest> = None;
    let mut first_result: Option<(usize, f32)> = None;
    for rep in 0..reps + bench_guard::WARMUP_REPS {
        let mut state = Engine::new_state(engine);
        bench_guard::check_caches_cold(&test, rep, &probe_gemma4(&state))?;
        let mut digest = WorkloadDigest::new();
        let t = Instant::now();
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            digest.feed(tok);
            logits = Engine::forward_token(engine, tok, i, &mut state);
        }
        let dt = t.elapsed().as_secs_f64();
        bench_guard::check_prefill_after(&test, n_prompt, &probe_gemma4(&state))?;
        let first = *first_digest.get_or_insert(digest);
        bench_guard::check_same_workload(&test, rep, first, digest)?;
        check_result(&test, rep, &logits, &mut first_result)?;
        if rep >= bench_guard::WARMUP_REPS {
            out.push(n_prompt as f64 / dt);
        }
    }
    Ok(Row {
        test,
        samples: out,
        digest: first_digest.unwrap_or_default(),
    })
}

/// `tg<N>`: N single-token decode steps after a one-token prime, so KV
/// length grows exactly as it does in real generation.
/// Whether the host-side `KvCache` is where the KV actually lands.
///
/// With GPU offload it is not: the device holds the KV and the host
/// struct stays empty unless `FRINK_CPU_KV_OFFLOAD=1` syncs it back.
/// Any assertion that counts host cache positions has to know that, or
/// it refuses every GPU run for doing nothing wrong.
pub(crate) fn host_kv_is_the_record() -> bool {
    let off = |k: &str| std::env::var(k).map(|v| v == "0").unwrap_or(false);
    let synced = std::env::var("FRINK_CPU_KV_OFFLOAD").as_deref() == Ok("1");
    synced || (off("FRINK_METAL") && off("FRINK_CUDA"))
}

fn bench_decode(decoder: &Decoder, n_gen: usize, reps: usize) -> anyhow::Result<Row> {
    let test = format!("tg{n_gen}");
    let vocab = decoder.config.vocab_size;
    let tokens = decode_tokens(vocab, n_gen, 0);
    // The decode stream gets the same BEFORE check the prompt does. It
    // is built up front for exactly that reason: a stream generated
    // inside the timed loop can only be checked once it is too late to
    // refuse.
    bench_guard::check_prompt_before(&test, n_gen, &tokens, vocab)?;
    let mut out = Vec::with_capacity(reps);
    let mut first_digest: Option<WorkloadDigest> = None;
    let mut first_result: Option<(usize, f32)> = None;
    for rep in 0..reps + bench_guard::WARMUP_REPS {
        let mut caches = fresh_caches(decoder);
        bench_guard::check_caches_cold(&test, rep, &probe(&caches))?;
        let _ = decoder.forward_token(0, 0, &mut caches);
        let mut digest = WorkloadDigest::new();
        let t = Instant::now();
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            digest.feed(tok);
            logits = decoder.forward_token(tok, i + 1, &mut caches);
        }
        let dt = t.elapsed().as_secs_f64();
        // One priming token plus n_gen decode steps. A short cache
        // means steps were skipped, which would inflate the rate.
        let kv_checked = bench_guard::check_decode_after(
            &test,
            1,
            n_gen,
            &probe(&caches),
            host_kv_is_the_record(),
        )?;
        let _ = kv_checked;
        let first = *first_digest.get_or_insert(digest);
        bench_guard::check_same_workload(&test, rep, first, digest)?;
        check_result(&test, rep, &logits, &mut first_result)?;
        if rep >= bench_guard::WARMUP_REPS {
            out.push(n_gen as f64 / dt);
        }
    }
    Ok(Row {
        test,
        samples: out,
        digest: first_digest.unwrap_or_default(),
    })
}

fn bench_decode_gemma4(engine: &Gemma4Engine, n_gen: usize, reps: usize) -> anyhow::Result<Row> {
    let test = format!("tg{n_gen}");
    let vocab = Engine::vocab_size(engine);
    let tokens = decode_tokens(vocab, n_gen, 0);
    bench_guard::check_prompt_before(&test, n_gen, &tokens, vocab)?;
    let mut out = Vec::with_capacity(reps);
    let mut first_digest: Option<WorkloadDigest> = None;
    let mut first_result: Option<(usize, f32)> = None;
    for rep in 0..reps + bench_guard::WARMUP_REPS {
        let mut state = Engine::new_state(engine);
        bench_guard::check_caches_cold(&test, rep, &probe_gemma4(&state))?;
        let _ = Engine::forward_token(engine, 0, 0, &mut state);
        let mut digest = WorkloadDigest::new();
        let t = Instant::now();
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            digest.feed(tok);
            logits = Engine::forward_token(engine, tok, i + 1, &mut state);
        }
        let dt = t.elapsed().as_secs_f64();
        let kv_checked = bench_guard::check_decode_after(
            &test,
            1,
            n_gen,
            &probe_gemma4(&state),
            host_kv_is_the_record(),
        )?;
        let _ = kv_checked;
        let first = *first_digest.get_or_insert(digest);
        bench_guard::check_same_workload(&test, rep, first, digest)?;
        check_result(&test, rep, &logits, &mut first_result)?;
        if rep >= bench_guard::WARMUP_REPS {
            out.push(n_gen as f64 / dt);
        }
    }
    Ok(Row {
        test,
        samples: out,
        digest: first_digest.unwrap_or_default(),
    })
}

pub(crate) fn fresh_caches(decoder: &Decoder) -> Vec<KvCache> {
    decoder.config.new_kv_caches()
}

/// Token ids that exist in the vocabulary but carry no linguistic
/// meaning: the point is to move the same bytes through the same
/// kernels, not to generate text.
///
/// `seq` offsets the stream so that parallel sequences in a batched
/// row do not carry identical prompts. Sequence 0 is the stream
/// `frink bench` has always fed, so its published digests still
/// describe the same work.
pub(crate) fn synthetic_tokens(vocab: usize, n: usize, seq: usize) -> Vec<usize> {
    let vocab = vocab.max(1);
    (0..n)
        .map(|i| (i * 7 + 1 + seq * SEQ_STRIDE) % vocab)
        .collect()
}

/// The token stream a `tg<N>` row feeds after its priming token, built
/// before the clock starts so its length can be asserted BEFORE the run
/// rather than only reconstructed from the KV cache afterwards.
///
/// `seq` as in [`synthetic_tokens`]: sequence 0 is the historical
/// stream, other sequences are shifted copies of it.
pub(crate) fn decode_tokens(vocab: usize, n_gen: usize, seq: usize) -> Vec<usize> {
    let vocab = vocab.max(1);
    (0..n_gen)
        .map(|i| (i + 1 + seq * SEQ_STRIDE) % vocab)
        .collect()
}

/// How far apart two parallel sequences' token streams start. Any
/// non-zero value distinguishes them; this one is co-prime with the
/// prompt stride of 7 so no two sequences' prompts coincide either.
const SEQ_STRIDE: usize = 1013;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_token_streams_satisfy_the_before_check_they_are_handed_to() {
        // The generators and the guard have to agree, or the BEFORE
        // check would be a refusal of every run rather than of a bad
        // one. `vocab` deliberately includes values smaller than the
        // stream, where the modulo wraps.
        for &(vocab, n) in &[(49152usize, 512usize), (7, 512), (2, 4), (49152, 1)] {
            for seq in [0, 1, 31] {
                bench_guard::check_prompt_before("pp", n, &synthetic_tokens(vocab, n, seq), vocab)
                    .unwrap();
                bench_guard::check_prompt_before("tg", n, &decode_tokens(vocab, n, seq), vocab)
                    .unwrap();
            }
        }
    }

    /// Sequence 0 is the stream every published `frink bench` digest
    /// was computed over; adding the `seq` parameter must not have
    /// moved it. The other sequences must differ from it, or a batched
    /// row would prefill the same prompt B times.
    #[test]
    fn sequence_zero_is_the_historical_stream_and_other_sequences_differ() {
        let vocab = 32000;
        let historical: Vec<usize> = (0..64).map(|i| (i * 7 + 1) % vocab).collect();
        assert_eq!(synthetic_tokens(vocab, 64, 0), historical);
        assert_ne!(
            synthetic_tokens(vocab, 64, 1),
            synthetic_tokens(vocab, 64, 0)
        );
        assert_ne!(decode_tokens(vocab, 64, 1), decode_tokens(vocab, 64, 0));
        assert_ne!(
            synthetic_tokens(vocab, 64, 2),
            synthetic_tokens(vocab, 64, 1)
        );
    }

    #[test]
    fn the_decode_stream_is_the_one_the_loop_used_to_generate_inline() {
        // Hoisting it out of the timed loop must not have changed the
        // workload: same ids, same order, so digests published before
        // and after this change still describe the same work.
        let vocab = 32000;
        let inline: Vec<usize> = (0..128).map(|i| (i + 1) % vocab).collect();
        assert_eq!(decode_tokens(vocab, 128, 0), inline);
    }

    #[test]
    fn a_repetition_that_returned_no_logits_is_refused_rather_than_timed() {
        let mut first = None;
        let err = check_result("pp512", 1, &[], &mut first)
            .unwrap_err()
            .to_string();
        assert!(err.contains("returned no logits"), "{err}");
    }

    #[test]
    fn the_first_repetitions_answer_becomes_the_one_the_rest_must_match() {
        let mut first = None;
        check_result("pp512", 0, &[0.1, 0.9, 0.2], &mut first).unwrap();
        assert_eq!(first, Some((1, 0.9)));
        // Same answer, different magnitudes: still the same work.
        check_result("pp512", 1, &[0.2, 0.7, 0.1], &mut first).unwrap();
        // A different answer from the same token stream is not.
        assert!(check_result("pp512", 2, &[0.9, 0.1, 0.2], &mut first).is_err());
    }
}
