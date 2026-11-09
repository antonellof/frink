//! The checks that stop `frink bench` from reporting a real number
//! for the wrong work.
//!
//! Every guard here exists because a benchmark can lie quietly, and a
//! quiet lie is worse than a crash: it gets published. A run whose
//! prompt was silently truncated reports an honest tok/s for a prompt
//! that was never processed. A run served partly from a warm KV cache
//! reports prefill throughput for a prefill that never happened. A run
//! whose first repetition paid for shader compilation reports the
//! Metal driver's startup cost as the engine's speed. None of those
//! produce an error on their own; they produce a plausible number.
//!
//! So each guard is a hard failure whose message names what it caught,
//! not a warning nobody reads. They are free functions over plain data
//! rather than methods on the engine types, so every one of them is
//! directly testable without loading a model -- which matters, because
//! a guard that has never been seen to fire is not a guard.

/// Repetitions run before timing starts, discarded. One is enough to
/// fault in every weight page, compile every Metal pipeline and warm
/// every lazily-built kernel table; the point is that it is *not* zero
/// and not silently variable.
pub const WARMUP_REPS: usize = 1;

/// What a KV cache looked like at some point in a repetition. Copied
/// out of `KvCache` so the guards stay independent of the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheProbe {
    pub seq_len: usize,
    /// Elements actually stored in `k` / `v`. Checked alongside
    /// `seq_len` because a cache whose length was reset to zero while
    /// keeping its contents is exactly what a prefix-cache reuse looks
    /// like from the outside.
    pub k_len: usize,
    pub v_len: usize,
}

impl CacheProbe {
    pub fn is_cold(&self) -> bool {
        self.seq_len == 0 && self.k_len == 0 && self.v_len == 0
    }
}

/// A timed row needs at least one timed repetition. `-r 0` used to run
/// the warmup and nothing else, leaving a row with zero samples whose
/// mean printed as `0.00` -- a fabricated number in the same column
/// as the measured ones.
pub fn check_repetitions(reps: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        reps >= 1,
        "-r {reps} leaves no timed repetitions: the warmup would run and the row \
         would report a mean over an empty sample set. Pass -r 1 or more."
    );
    Ok(())
}

/// Warmup accounting, asserted after the fact: a row must carry exactly
/// `reps` samples, meaning `WARMUP_REPS` repetitions ran and were
/// thrown away. Fires if a later edit changes the loop bound or the
/// discard condition, which is how a shader-compile timing quietly
/// gets back into a published mean.
pub fn check_timed_samples(test: &str, reps: usize, samples: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        samples == reps,
        "{test}: {samples} timed samples for -r {reps}. Exactly {WARMUP_REPS} warmup \
         repetition must run before timing and be discarded; a mismatch means either \
         an untimed repetition leaked into the mean or the warmup did not run, and \
         the first repetition pays for page faults and shader compilation."
    );
    Ok(())
}

/// Every timed sample has to be a usable rate.
///
/// A repetition whose duration rounded to zero yields `inf` tok/s, and a
/// single `inf` makes the mean `inf` (and the stddev NaN). It would
/// print as `inf` in the same column as measured rows and render into
/// the ledger as a win.
pub fn check_sample_rates(test: &str, samples: &[f64]) -> anyhow::Result<()> {
    if let Some((i, &v)) = samples
        .iter()
        .enumerate()
        .find(|(_, &v)| !v.is_finite() || v <= 0.0)
    {
        anyhow::bail!(
            "{test} rep {i} produced a rate of {v}: the timer measured no elapsed \
             time, so this is not a throughput. It would still enter the \
             mean and be published as one."
        );
    }
    Ok(())
}

/// The prompt length asserted BEFORE the run: the token stream about to
/// be fed must be exactly as long as the `pp<N>` (or `tg<N>`) label
/// promises, and every id must exist in the vocabulary. An id past the
/// end of the embedding table is either a panic or a garbage row --
/// neither is the workload the row claims to measure.
pub fn check_prompt_before(
    test: &str,
    n_prompt: usize,
    tokens: &[usize],
    vocab: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        n_prompt > 0,
        "{test}: refusing to time a zero-token prompt; the rate would divide zero \
         work by a real duration"
    );
    anyhow::ensure!(
        tokens.len() == n_prompt,
        "{test}: built {} tokens for a {n_prompt}-token prompt -- the row would \
         report {n_prompt} tokens per second for {} tokens of work",
        tokens.len(),
        tokens.len()
    );
    anyhow::ensure!(vocab > 0, "{test}: model reports a zero-sized vocabulary");
    if let Some((i, &t)) = tokens.iter().enumerate().find(|(_, &t)| t >= vocab) {
        anyhow::bail!("{test}: synthetic token {i} is id {t}, outside a vocabulary of {vocab}");
    }
    Ok(())
}

/// Zero cache hits, asserted rather than assumed. Every repetition must
/// start from a genuinely empty KV cache: a prefill measured against a
/// warm cache reports the engine's speed at work it did not do.
pub fn check_caches_cold(test: &str, rep: usize, caches: &[CacheProbe]) -> anyhow::Result<()> {
    anyhow::ensure!(
        !caches.is_empty(),
        "{test}: no KV caches to check -- a model with no layers is not a benchmark"
    );
    if let Some((layer, c)) = caches.iter().enumerate().find(|(_, c)| !c.is_cold()) {
        anyhow::bail!(
            "{test} rep {rep} started with a warm KV cache at layer {layer} \
             (seq_len {}, {} k / {} v elements retained): this repetition would be \
             served partly from cached attention state and report prefill \
             throughput for a prefill that never happened",
            c.seq_len,
            c.k_len,
            c.v_len
        );
    }
    Ok(())
}

/// The prompt length asserted AFTER the run, not only before. An engine
/// that silently truncated the batch would otherwise divide the full
/// token count by a partial run's duration and report a speedup for
/// doing less work.
pub fn check_prefill_after(
    test: &str,
    n_prompt: usize,
    caches: &[CacheProbe],
) -> anyhow::Result<()> {
    if let Some((layer, c)) = caches
        .iter()
        .enumerate()
        .find(|(_, c)| c.seq_len != n_prompt)
    {
        anyhow::bail!(
            "{test}: layer {layer} consumed {} of {n_prompt} prompt tokens -- the \
             reported rate would be {n_prompt} tok/s worth of credit for work the \
             engine skipped",
            c.seq_len
        );
    }
    Ok(())
}

/// The decode equivalent: `n_primed` positions already in the cache
/// (one priming token for a `tg<N>` row, the whole prompt for a
/// batched-bench row) plus `n_gen` steps must leave exactly
/// `n_primed + n_gen` positions in every layer's cache. A short cache
/// means steps were skipped, which inflates the rate.
///
/// Only meaningful when the HOST cache is the record. With GPU offload
/// the KV lives in device memory and the host `KvCache` is never
/// written unless `FRINK_CPU_KV_OFFLOAD=1` syncs it back, so every
/// layer reads 0 and this refuses a run that did nothing wrong. That
/// is not hypothetical: it failed every Metal row of a full suite
/// while the CPU rows passed, which is what a backend-blind assertion
/// looks like from the outside.
///
/// Returns whether the check actually ran, so the caller can record
/// that in the receipt. "Not checked" and "checked and passed" are
/// different facts and a receipt that conflates them is worth less.
pub fn check_decode_after(
    test: &str,
    n_primed: usize,
    n_gen: usize,
    caches: &[CacheProbe],
    host_kv_is_the_record: bool,
) -> anyhow::Result<bool> {
    if !host_kv_is_the_record {
        // The device holds the KV. There is nothing here to count.
        return Ok(false);
    }
    let expected = n_primed + n_gen;
    if let Some((layer, c)) = caches
        .iter()
        .enumerate()
        .find(|(_, c)| c.seq_len != expected)
    {
        anyhow::bail!(
            "{test}: layer {layer} advanced the KV cache to {} positions, expected \
             {expected} ({n_primed} primed + {n_gen} decode steps) -- decode steps \
             were skipped and the rate would count them anyway",
            c.seq_len
        );
    }
    Ok(true)
}

#[cfg(test)]
mod decode_guard_tests {
    use super::*;

    /// The whole point of the backend flag: a GPU run has an empty host
    /// cache and must not be refused for it. Before this, every Metal
    /// row of a full suite failed while the CPU rows passed.
    #[test]
    fn a_device_resident_cache_is_not_counted_and_says_so() {
        let empty = vec![
            CacheProbe {
                seq_len: 0,
                k_len: 0,
                v_len: 0
            };
            4
        ];
        let ran = check_decode_after("tg128", 1, 128, &empty, false)
            .expect("a device-resident cache must not refuse the run");
        assert!(
            !ran,
            "it must report that it did not check, not that it passed"
        );
    }

    /// And the check still bites where the host cache IS the record,
    /// or turning it off for GPU would have removed it everywhere.
    #[test]
    fn a_short_host_cache_still_fails() {
        let short = vec![CacheProbe {
            seq_len: 3,
            k_len: 3,
            v_len: 3,
        }];
        let err = check_decode_after("tg128", 1, 128, &short, true)
            .expect_err("a short host cache means skipped decode steps");
        assert!(err.to_string().contains("expected 129"));
    }

    #[test]
    fn a_correct_host_cache_reports_that_it_actually_checked() {
        let good = vec![CacheProbe {
            seq_len: 129,
            k_len: 129,
            v_len: 129,
        }];
        assert!(check_decode_after("tg128", 1, 128, &good, true).expect("must pass"));
    }
}

/// Running digest of the exact token stream fed inside the timed
/// region. FNV-1a over the ids: a few nanoseconds per token against
/// milliseconds per forward pass, so it cannot move the number it
/// guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkloadDigest(u64);

impl Default for WorkloadDigest {
    fn default() -> Self {
        WorkloadDigest(0xcbf2_9ce4_8422_2325)
    }
}

impl WorkloadDigest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, token: usize) {
        for b in (token as u64).to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub fn feed_all(&mut self, tokens: &[usize]) {
        for &t in tokens {
            self.feed(t);
        }
    }

    pub fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// The temperature-zero assertion, in the only form that can actually
/// be checked from outside a sampler: every timed repetition must feed
/// the engine the identical token stream.
///
/// `frink bench` drives synthetic ids and never samples, so nominally
/// temperature does not apply. That is exactly the claim worth
/// enforcing -- if a future edit fed generated tokens back into the
/// decode loop, any temperature above zero would make repetition 2 do
/// different work from repetition 1, and the mean across them would
/// stop being a mean of anything. Identical digests are what
/// "temperature 0" means for a benchmark; divergent digests catch its
/// absence whatever the cause.
pub fn check_same_workload(
    test: &str,
    rep: usize,
    first: WorkloadDigest,
    current: WorkloadDigest,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        first == current,
        "{test} rep {rep} fed a different token stream than the first repetition \
         ({} vs {}): the repetitions are not repetitions of the same work, so their \
         mean is not a measurement. A timed run must be deterministic -- \
         temperature 0, no sampling, no data-dependent token feedback.",
        current.hex(),
        first.hex()
    );
    Ok(())
}

/// The greedy (`temp 0`) next-token pick of a repetition's final
/// logits: `(id, logit)`.
///
/// `None` for an empty slice, so "the engine returned nothing" is
/// reported as its own failure rather than as a disagreement. A NaN
/// logit is an error, because it makes "the argmax" a property of the
/// comparison order rather than of the model.
pub fn greedy_pick(logits: &[f32]) -> anyhow::Result<Option<(usize, f32)>> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        anyhow::ensure!(
            !v.is_nan(),
            "logit {i} is NaN: the forward pass did not produce a usable result, \
             so the time it took is not a throughput"
        );
        match best {
            Some((_, b)) if v <= b => {}
            _ => best = Some((i, v)),
        }
    }
    Ok(best)
}

/// The other half of the determinism assertion: identical *input*
/// leaving the engine is not the same claim as identical *output*
/// coming back.
///
/// [`check_same_workload`] catches a sampler or a data-dependent feed
/// putting different tokens into repetition 2. It cannot catch the
/// engine itself computing something different from the same tokens --
/// a routing decision that depends on a warmed expert cache, a
/// reduction whose order varies with how work happened to be split, a
/// kernel that races. Those change what the repetition *did* while
/// leaving the digest identical, and the mean once again averages
/// unlike things.
///
/// Compared on the greedy token id rather than on exact logit bits:
/// last-bit float differences between two runs of the same kernel are
/// not the failure being hunted, a different answer is.
pub fn check_same_result(
    test: &str,
    rep: usize,
    first: (usize, f32),
    current: (usize, f32),
) -> anyhow::Result<()> {
    anyhow::ensure!(
        first.0 == current.0,
        "{test} rep {rep} computed a different answer than the first repetition \
         from the identical token stream: greedy token {} (logit {}) vs {} (logit \
         {}). The engine is not deterministic here, so these repetitions measured \
         different work and their mean is not a measurement of either.",
        current.0,
        current.1,
        first.0,
        first.1
    );
    Ok(())
}

/// Environment variables that `frink bench` sets for itself as part of
/// selecting a backend and thread count. They carry no information for
/// an auditor beyond what the receipt already records.
const SELF_SET_ENV: &[&str] = &[
    "FRINK_METAL",
    "FRINK_METAL_ATTN",
    "FRINK_CUDA",
    "FRINK_CPU_THREADS",
];

/// Every `FRINK_*` variable in effect that the bench did not set
/// itself, as `(name, value)` pairs sorted by name.
///
/// This is recorded, not refused. Some of these knobs pick a device or a
/// budget and say nothing an auditor cannot already read off the receipt,
/// and refusing on them would break the workflow the ledger depends on.
/// But others -- `FRINK_ALLOW_UNKNOWN_TENSORS` disables the fail-closed
/// loader gate, `FRINK_STRICT_KERNELS` is the gate a benchmark host is
/// supposed to be running under -- change how much work the engine does
/// or how much of it is checked, and a published row taken under one of
/// them is not comparable to a row taken without it. Putting them in the
/// receipt means the difference is discoverable later instead of lost
/// with the shell history.
pub fn nondefault_engine_env<I, K, V>(vars: I) -> Vec<(String, String)>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut out: Vec<(String, String)> = vars
        .into_iter()
        .filter_map(|(k, v)| {
            let k = k.as_ref();
            (k.starts_with("FRINK_") && !SELF_SET_ENV.contains(&k))
                .then(|| (k.to_string(), v.as_ref().to_string()))
        })
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cold(n: usize) -> Vec<CacheProbe> {
        vec![
            CacheProbe {
                seq_len: 0,
                k_len: 0,
                v_len: 0
            };
            n
        ]
    }

    fn filled(n: usize, seq_len: usize, elems_per_pos: usize) -> Vec<CacheProbe> {
        vec![
            CacheProbe {
                seq_len,
                k_len: seq_len * elems_per_pos,
                v_len: seq_len * elems_per_pos,
            };
            n
        ]
    }

    #[test]
    fn a_run_with_no_timed_repetitions_is_refused() {
        let err = check_repetitions(0).unwrap_err().to_string();
        assert!(err.contains("no timed repetitions"), "{err}");
        assert!(check_repetitions(1).is_ok());
    }

    #[test]
    fn a_row_must_carry_exactly_one_discarded_warmup() {
        assert!(check_timed_samples("pp512", 3, 3).is_ok());
        // The warmup leaked into the mean.
        let err = check_timed_samples("pp512", 3, 4).unwrap_err().to_string();
        assert!(err.contains("shader compilation"), "{err}");
        // A repetition vanished.
        assert!(check_timed_samples("pp512", 3, 2).is_err());
    }

    #[test]
    fn the_warmup_count_is_one_and_the_message_says_so() {
        assert_eq!(WARMUP_REPS, 1);
        let err = check_timed_samples("tg128", 2, 3).unwrap_err().to_string();
        assert!(err.contains("Exactly 1 warmup"), "{err}");
    }

    #[test]
    fn a_short_token_stream_cannot_be_reported_as_a_full_prompt() {
        let tokens: Vec<usize> = (0..500).collect();
        let err = check_prompt_before("pp512", 512, &tokens, 32000)
            .unwrap_err()
            .to_string();
        assert!(err.contains("built 500 tokens"), "{err}");
        assert!(err.contains("512 tokens per second for 500"), "{err}");
    }

    #[test]
    fn a_token_outside_the_vocabulary_is_caught_before_the_run() {
        let tokens = vec![1, 2, 99_999, 4];
        let err = check_prompt_before("pp4", 4, &tokens, 32000)
            .unwrap_err()
            .to_string();
        assert!(err.contains("token 2 is id 99999"), "{err}");
        assert!(check_prompt_before("pp4", 4, &[1, 2, 3, 4], 32000).is_ok());
    }

    #[test]
    fn a_zero_length_prompt_is_never_timed() {
        assert!(check_prompt_before("pp0", 0, &[], 32000).is_err());
        assert!(check_prompt_before("pp1", 1, &[0], 0).is_err());
    }

    #[test]
    fn a_warm_cache_at_any_layer_stops_the_repetition() {
        assert!(check_caches_cold("pp512", 0, &cold(8)).is_ok());
        let mut caches = cold(8);
        caches[5] = CacheProbe {
            seq_len: 128,
            k_len: 128 * 64,
            v_len: 128 * 64,
        };
        let err = check_caches_cold("pp512", 2, &caches)
            .unwrap_err()
            .to_string();
        assert!(err.contains("layer 5"), "{err}");
        assert!(err.contains("a prefill that never happened"), "{err}");
    }

    #[test]
    fn a_cache_reset_to_zero_length_but_still_holding_data_is_not_cold() {
        // What reusing a stored prefix looks like from outside: the
        // length says empty, the contents say otherwise.
        let sneaky = [CacheProbe {
            seq_len: 0,
            k_len: 128 * 64,
            v_len: 128 * 64,
        }];
        assert!(!sneaky[0].is_cold());
        let err = check_caches_cold("pp512", 1, &sneaky)
            .unwrap_err()
            .to_string();
        assert!(err.contains("elements retained"), "{err}");
    }

    #[test]
    fn a_model_with_no_caches_is_not_silently_accepted() {
        assert!(check_caches_cold("pp512", 0, &[]).is_err());
    }

    #[test]
    fn a_truncated_prefill_cannot_report_the_full_prompt_length() {
        assert!(check_prefill_after("pp512", 512, &filled(4, 512, 64)).is_ok());
        let mut caches = filled(4, 512, 64);
        caches[3].seq_len = 256;
        let err = check_prefill_after("pp512", 512, &caches)
            .unwrap_err()
            .to_string();
        assert!(err.contains("consumed 256 of 512"), "{err}");
        assert!(err.contains("layer 3"), "{err}");
    }

    #[test]
    fn a_prefill_that_ran_long_is_caught_too() {
        // Over-consumption is equally wrong: the divisor no longer
        // matches the work.
        let caches = filled(2, 600, 64);
        assert!(check_prefill_after("pp512", 512, &caches).is_err());
    }

    #[test]
    fn skipped_decode_steps_are_caught_by_the_final_cache_length() {
        assert!(check_decode_after("tg128", 1, 128, &filled(4, 129, 64), true).is_ok());
        let err = check_decode_after("tg128", 1, 128, &filled(4, 65, 64), true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 129"), "{err}");
        assert!(err.contains("decode steps were skipped"), "{err}");
    }

    /// A batched-bench row primes with the whole prompt, not one
    /// token, so the expected length is `pp + tg`. Pinned so the prime
    /// count cannot quietly go back to being a constant.
    #[test]
    fn a_prompt_primed_cache_is_expected_to_hold_prompt_plus_decode_positions() {
        assert!(check_decode_after("pp128 tg64", 128, 64, &filled(4, 192, 64), true).is_ok());
        let err = check_decode_after("pp128 tg64", 128, 64, &filled(4, 129, 64), true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 192"), "{err}");
        assert!(err.contains("128 primed + 64"), "{err}");
    }

    #[test]
    fn identical_token_streams_digest_identically_and_different_ones_do_not() {
        let mut a = WorkloadDigest::new();
        a.feed_all(&[1, 2, 3, 4]);
        let mut b = WorkloadDigest::new();
        b.feed_all(&[1, 2, 3, 4]);
        assert_eq!(a, b);
        assert!(check_same_workload("tg128", 1, a, b).is_ok());

        let mut c = WorkloadDigest::new();
        c.feed_all(&[1, 2, 3, 5]);
        assert_ne!(a, c);
        let err = check_same_workload("tg128", 2, a, c)
            .unwrap_err()
            .to_string();
        assert!(err.contains("different token stream"), "{err}");
        assert!(err.contains("temperature 0"), "{err}");
    }

    #[test]
    fn the_digest_is_order_sensitive_and_length_sensitive() {
        // A sampler that produced the same multiset in a different
        // order is still doing different work per repetition.
        let mut a = WorkloadDigest::new();
        a.feed_all(&[1, 2, 3]);
        let mut b = WorkloadDigest::new();
        b.feed_all(&[3, 2, 1]);
        assert_ne!(a, b, "digest must not be order-insensitive");

        let mut short = WorkloadDigest::new();
        short.feed_all(&[1, 2]);
        assert_ne!(
            a, short,
            "a truncated stream must not digest as the full one"
        );

        // A stream of zeros must not collide with an empty stream.
        let mut zeros = WorkloadDigest::new();
        zeros.feed_all(&[0, 0, 0]);
        assert_ne!(zeros, WorkloadDigest::new());
    }

    #[test]
    fn the_digest_is_stable_across_runs_so_two_receipts_can_be_compared() {
        // Pinned: if this changes, previously published digests stop
        // being comparable to new ones, which is the only thing the
        // receipt field is for.
        let mut d = WorkloadDigest::new();
        d.feed_all(&[1, 8, 15, 22]);
        assert_eq!(d.hex(), d.hex());
        let mut again = WorkloadDigest::new();
        again.feed_all(&[1, 8, 15, 22]);
        assert_eq!(d.hex(), again.hex());
        assert_eq!(d.hex().len(), 16);
    }

    #[test]
    fn an_infinite_rate_is_refused_because_it_would_enter_the_mean() {
        assert!(check_sample_rates("pp512", &[10.0, 11.0, 12.0]).is_ok());
        // dt rounded to zero. `inf` is not caught by any length or
        // cache check -- the work really was done, the clock just did
        // not see it -- so it needs its own guard.
        let err = check_sample_rates("pp512", &[10.0, f64::INFINITY])
            .unwrap_err()
            .to_string();
        assert!(err.contains("rep 1"), "{err}");
        assert!(err.contains("no elapsed time"), "{err}");
        assert!(check_sample_rates("pp512", &[10.0, f64::NAN]).is_err());
        assert!(check_sample_rates("pp512", &[0.0]).is_err());
        assert!(check_sample_rates("pp512", &[-1.0]).is_err());
    }

    #[test]
    fn an_empty_sample_set_is_left_to_the_warmup_accounting() {
        // check_timed_samples is the guard that owns "a row must have
        // samples"; this one must not duplicate a worse message for it.
        assert!(check_sample_rates("pp512", &[]).is_ok());
        assert!(check_timed_samples("pp512", 3, 0).is_err());
    }

    #[test]
    fn greedy_pick_is_the_first_maximum_and_refuses_nan() {
        assert_eq!(greedy_pick(&[0.1, 0.9, 0.3]).unwrap(), Some((1, 0.9)));
        assert_eq!(
            greedy_pick(&[0.9, 0.9]).unwrap(),
            Some((0, 0.9)),
            "a tie resolves to the lowest id, as greedy sampling does"
        );
        assert_eq!(greedy_pick(&[]).unwrap(), None);
        let err = greedy_pick(&[0.1, f32::NAN]).unwrap_err().to_string();
        assert!(err.contains("NaN"), "{err}");
    }

    #[test]
    fn an_engine_that_answered_differently_from_the_same_tokens_is_refused() {
        // The case check_same_workload structurally cannot see: the
        // token stream is identical, so the digests match, and the
        // engine still computed something else.
        let mut a = WorkloadDigest::new();
        a.feed_all(&[1, 2, 3]);
        let mut b = WorkloadDigest::new();
        b.feed_all(&[1, 2, 3]);
        assert!(check_same_workload("pp512", 2, a, b).is_ok());

        let first = greedy_pick(&[0.1, 0.9, 0.2]).unwrap().unwrap();
        let second = greedy_pick(&[0.9, 0.1, 0.2]).unwrap().unwrap();
        let err = check_same_result("pp512", 2, first, second)
            .unwrap_err()
            .to_string();
        assert!(err.contains("identical token stream"), "{err}");
        assert!(err.contains("greedy token 0"), "{err}");
        assert!(err.contains("not deterministic"), "{err}");
    }

    #[test]
    fn last_bit_logit_noise_is_not_treated_as_nondeterminism() {
        // Two runs of the same kernel differing in the last mantissa
        // bit are not the failure being hunted; a different answer is.
        // If this ever starts failing, the check has become a source of
        // false refusals on GPU backends.
        let first = greedy_pick(&[0.1, 0.9]).unwrap().unwrap();
        let jittered = greedy_pick(&[0.1, 0.9 + f32::EPSILON]).unwrap().unwrap();
        assert!(check_same_result("pp512", 2, first, jittered).is_ok());
    }

    #[test]
    fn engine_env_recording_keeps_the_knobs_and_drops_the_bench_own_settings() {
        let got = nondefault_engine_env([
            ("PATH", "/usr/bin"),
            ("FRINK_METAL", "auto"),
            ("FRINK_METAL_ATTN", "1"),
            ("FRINK_CUDA", "auto"),
            ("FRINK_CPU_THREADS", "8"),
            ("FRINK_CTK", "q8_0"),
            ("FRINK_ALLOW_UNKNOWN_TENSORS", "1"),
            ("HOME", "/Users/x"),
        ]);
        assert_eq!(
            got,
            vec![
                ("FRINK_ALLOW_UNKNOWN_TENSORS".to_string(), "1".to_string()),
                ("FRINK_CTK".to_string(), "q8_0".to_string()),
            ]
        );
    }

    #[test]
    fn a_clean_environment_records_nothing_rather_than_a_placeholder() {
        assert!(nondefault_engine_env([("PATH", "/usr/bin")]).is_empty());
    }
}
