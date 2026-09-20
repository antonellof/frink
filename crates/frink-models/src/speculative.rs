//! Speculative decoding: propose several candidate next tokens with
//! something cheap, then verify them all in a single
//! `Decoder::forward_batch` call instead of one `forward_token` call
//! per token.
//!
//! Two halves, deliberately separated:
//!
//! * **Drafting** is the [`Drafter`] trait. The only implementation in
//!   the tree is [`PromptLookupSpeculator`], an n-gram match over the
//!   history with no model at all (prompt-lookup drafting), chosen
//!   because it needs no GPU, no second set
//!   of weights and no checkpoint to be useful. A model-based drafter
//!   (MTP head, EAGLE, dFlash) is a second impl of the same trait.
//! * **Verification** is [`speculative_decode_with`], and it does not
//!   know or care which drafter proposed the block.
//!
//! # Losslessness is the property that matters most here
//!
//! Speculative decoding is only worth having if it produces exactly the
//! same *distribution* the target model would have produced on its own,
//! just faster. This module implements the speculative-sampling
//! rejection rule (Leviathan et al. 2023 / Chen et al. 2023): a draft
//! token `x` proposed with draft probability `q(x)` is accepted with
//! probability `min(1, p(x)/q(x))`, and on rejection the position is
//! resampled from the normalised residual `max(0, p - q)`. That rule is
//! lossless at *every* temperature.
//!
//! It is worth being precise about what the previous accept test --
//! `argmax(target_logits[i]) == guess` -- actually guaranteed, because
//! it looks like the same thing and is not. Argmax matching is exactly
//! the special case of the rule above at `temperature = 0`, where `p`
//! is a point mass: `p(x)` is 1 when the guess is the argmax and 0
//! otherwise, so acceptance is certain or impossible and the residual
//! collapses back onto the argmax. Above temperature 0 it is a
//! different algorithm with a different output distribution -- it
//! silently biases generation toward the target's argmax, because a
//! draft token only survives if it happens to be the most likely one.
//! [`accept_or_resample`] is therefore not an optimisation; it is the
//! difference between "lossless" being true and being a claim.
//!
//! The invariant is tested directly, not assumed:
//! `resampling_reproduces_the_target_distribution` pushes two hundred
//! thousand tokens through the accept/reject rule with deliberately bad
//! draft distributions and asserts the empirical output matches the
//! target distribution;
//! `speculative_decode_at_temperature_matches_plain_sampling` compares
//! a real decode at temperature 1.0 against the target's own exactly
//! enumerated per-position marginals; and
//! `speculative_decode_matches_greedy_token_for_token` asserts
//! token-for-token identity with a plain `forward_token` loop at
//! temperature 0.

use crate::decoder::Decoder;
use crate::penalty_window::PenaltyWindow;
use crate::sampling::{sampling_distribution, Sampler, SamplingParams};
use frink_core::cache::KvCache;

/// The distribution one drafted position was sampled from, as its
/// complete support: `(token id, probability)` pairs summing to 1.
///
/// Sparse rather than a dense vocabulary-length vector because the
/// distributions drafters actually produce are sparse: a prompt-lookup
/// drafter's support is a single token, and a real drafter's is a
/// top-k, not 150k floats per drafted position per step.
///
/// # Contract
///
/// The support must be the distribution the draft token was *actually*
/// sampled from. Losslessness does not require the drafter to be good,
/// or even sane -- the rejection rule corrects any `q` -- but it does
/// require `q` to be honest. A drafter that truncates its own softmax
/// to a top-k before sampling must report the truncated, renormalised
/// distribution, not the full softmax it started from.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DraftDist {
    support: Vec<(usize, f32)>,
}

impl DraftDist {
    /// A drafter that is certain: all probability on one token. This is
    /// what a lookup-table drafter honestly reports -- it did not
    /// sample, it asserted.
    pub fn deterministic(token: usize) -> Self {
        DraftDist {
            support: vec![(token, 1.0)],
        }
    }

    /// The nonzero entries of a dense probability vector.
    pub fn from_dense(probs: &[f32]) -> Self {
        DraftDist {
            support: probs
                .iter()
                .enumerate()
                .filter(|&(_, &p)| p > 0.0)
                .map(|(i, &p)| (i, p))
                .collect(),
        }
    }

    /// Builds from explicit `(token, probability)` pairs.
    pub fn from_support(support: Vec<(usize, f32)>) -> Self {
        DraftDist { support }
    }

    pub fn support(&self) -> &[(usize, f32)] {
        &self.support
    }

    /// `q(token)`, or 0.0 for a token outside the support.
    pub fn prob(&self, token: usize) -> f32 {
        self.support
            .iter()
            .find(|&&(t, _)| t == token)
            .map(|&(_, p)| p)
            .unwrap_or(0.0)
    }
}

/// A block of drafted tokens plus, per position, the distribution that
/// position was drawn from.
///
/// `tokens` and `dists` are the same length by construction: the
/// verification rule needs `q` for every token it might have to reject,
/// so a block that carried tokens without distributions could not be
/// verified losslessly at all.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DraftBlock {
    tokens: Vec<usize>,
    dists: Vec<DraftDist>,
}

impl DraftBlock {
    pub fn empty() -> Self {
        DraftBlock::default()
    }

    /// Panics if the two vectors disagree in length -- an unverifiable
    /// block is a programming error in the drafter, not a runtime
    /// condition to degrade around.
    pub fn new(tokens: Vec<usize>, dists: Vec<DraftDist>) -> Self {
        assert_eq!(
            tokens.len(),
            dists.len(),
            "a draft block needs one draft distribution per drafted token"
        );
        DraftBlock { tokens, dists }
    }

    /// A block from a drafter that has no distribution to offer: every
    /// position is reported as certain. Correct (and lossless) for a
    /// lookup drafter; wrong for a model-based one, which must report
    /// its real softmax.
    pub fn deterministic(tokens: Vec<usize>) -> Self {
        let dists = tokens
            .iter()
            .map(|&t| DraftDist::deterministic(t))
            .collect();
        DraftBlock { tokens, dists }
    }

    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    pub fn dists(&self) -> &[DraftDist] {
        &self.dists
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Drops everything past `len` positions, keeping tokens and
    /// distributions in step.
    pub fn truncate(&mut self, len: usize) {
        self.tokens.truncate(len);
        self.dists.truncate(len);
    }
}

/// Proposes a block of candidate continuation tokens.
///
/// The signature carries two things a plain `fn(&[usize]) -> Vec<usize>`
/// cannot express, and both are load-bearing:
///
/// * **Per-position draft probabilities**, without which
///   [`accept_or_resample`] cannot run and speculation is only lossless
///   at temperature 0.
/// * **The target model's hidden state** for the last position whose KV
///   is committed. Every model-based drafter worth having (EAGLE, MTP,
///   dFlash) conditions on it, and `Decoder::forward_batch_with_hidden`
///   already computes it as a by-product of verification, so a drafter
///   that wanted it would otherwise have to run the target twice.
///   Drafters that do not need it, like [`PromptLookupSpeculator`],
///   ignore the argument.
pub trait Drafter {
    /// Proposes at most `max_len` tokens to follow `history`.
    /// `target_hidden` is the target model's final-layer hidden state
    /// for `history`'s last token, or empty when none is available yet
    /// (which a drafter that needs it must handle by proposing
    /// nothing).
    /// Takes `&mut self` because a drafter that is itself a model
    /// carries KV state across calls, and hiding that behind interior
    /// mutability would put a runtime borrow check on the hot path to
    /// buy nothing. A stateless drafter simply ignores it.
    fn propose(&mut self, history: &[usize], target_hidden: &[f32], max_len: usize) -> DraftBlock;
}

/// Proposes candidate continuation tokens by looking for the longest
/// available match of the most recent `ngram_size` tokens earlier in
/// `history`, and returning up to `max_draft_len` tokens that followed
/// that earlier occurrence. Returns an empty block if no match is found
/// or `history` is too short to contain one.
///
/// This is deliberately simple (last-match-wins, not best-match or a
/// frequency-weighted choice): the whole point of prompt-lookup
/// decoding is that it's nearly free to compute, since a wrong guess
/// costs nothing but a rejected batch position, not a correctness bug.
#[derive(Debug, Clone, Copy)]
pub struct PromptLookupSpeculator {
    pub ngram_size: usize,
    pub max_draft_len: usize,
}

impl PromptLookupSpeculator {
    pub fn new(ngram_size: usize, max_draft_len: usize) -> Self {
        assert!(ngram_size >= 1, "ngram_size must be at least 1");
        assert!(max_draft_len >= 1, "max_draft_len must be at least 1");
        PromptLookupSpeculator {
            ngram_size,
            max_draft_len,
        }
    }

    /// Looks for the most recent earlier occurrence of `history`'s
    /// last `ngram_size` tokens, scanning from the end backwards so
    /// the *most recent* match wins (most likely to reflect current
    /// context, e.g. a loop the model is currently in). Returns the
    /// tokens that followed that occurrence, truncated to
    /// `max_draft_len`.
    pub fn propose_tokens(&self, history: &[usize]) -> Vec<usize> {
        if history.len() < self.ngram_size + 1 {
            return Vec::new();
        }
        let needle = &history[history.len() - self.ngram_size..];

        // Search every earlier start position, latest first. The last
        // possible start that still leaves room for the needle without
        // overlapping into the needle itself is history.len() -
        // ngram_size - 1 (exclusive of the needle's own occurrence).
        let last_possible_start = history.len() - self.ngram_size - 1;
        for start in (0..=last_possible_start).rev() {
            if &history[start..start + self.ngram_size] == needle {
                let continuation_start = start + self.ngram_size;
                let available = history.len() - continuation_start;
                let take = available.min(self.max_draft_len);
                return history[continuation_start..continuation_start + take].to_vec();
            }
        }
        Vec::new()
    }
}

impl Drafter for PromptLookupSpeculator {
    fn propose(&mut self, history: &[usize], _target_hidden: &[f32], max_len: usize) -> DraftBlock {
        let mut tokens = self.propose_tokens(history);
        tokens.truncate(max_len.min(self.max_draft_len));
        DraftBlock::deterministic(tokens)
    }
}

/// The speculative-sampling accept/reject decision for one drafted
/// position.
///
/// `target` is the target model's *final* sampling distribution for
/// this position -- what [`sampling_distribution`] returns, i.e. after
/// penalties, temperature, top-k and top-p, because that is the
/// distribution the non-speculative path would have drawn from.
/// `draft` is the distribution `token` was drawn from.
///
/// Returns `None` when the draft token is accepted, and
/// `Some(replacement)` when it is rejected -- the replacement is drawn
/// from the normalised residual `max(0, target - draft)`, which is what
/// makes the combined procedure's output distribution equal to
/// `target` exactly rather than approximately.
///
/// A token with `draft(token) == 0.0` violates the [`DraftDist`]
/// contract (it could not have been sampled from `draft`); it is
/// accepted, matching the `p/q -> infinity` limit, rather than
/// silently biasing the result.
pub fn accept_or_resample(
    target: &[f32],
    draft: &DraftDist,
    token: usize,
    rng: &mut Sampler,
) -> Option<usize> {
    let p = target.get(token).copied().unwrap_or(0.0);
    let q = draft.prob(token);
    if q <= 0.0 || p >= q {
        return None;
    }
    // p < q, so the accept probability p/q is a real coin flip.
    if rng.uniform() < p / q {
        return None;
    }

    // Rejected: draw the replacement from the normalised residual.
    let mut residual = target.to_vec();
    for &(t, qt) in draft.support() {
        if let Some(r) = residual.get_mut(t) {
            *r = (*r - qt).max(0.0);
        }
    }
    let total: f32 = residual.iter().sum();
    if total <= 0.0 {
        // Only reachable when target and draft are the same
        // distribution, in which case acceptance was certain and we
        // cannot be here -- but sampling from nothing is not an option,
        // so fall back to the target itself.
        return Some(rng.sample_from(target));
    }
    for r in residual.iter_mut() {
        *r /= total;
    }
    Some(rng.sample_from(&residual))
}

/// Everything `speculative_decode_with` needs beyond the model, the
/// prompt and the drafter.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeOptions {
    pub max_new_tokens: usize,
    /// Absolute position of the first `prompt_tokens` token in the KV
    /// cache: 0 for a fresh cache, `cache.seq_len` when resuming a
    /// warm one (a prefix-cache hit, or a second call continuing the
    /// first). Every position and every rollback length inside the
    /// decode loop is absolute, so a non-zero base is not a special
    /// case -- see `rolls_back_to_absolute_positions_on_a_warm_cache`.
    pub start_pos: usize,
    /// The sampling configuration the *target* model would have used
    /// without speculation. Verification is lossless with respect to
    /// exactly these parameters (see [`accept_or_resample`]).
    pub sampling: SamplingParams,
    pub seed: u64,
}

/// Result of a speculative decode run, with the counters that make its
/// actual savings observable rather than just assumed.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeDecodeResult {
    pub generated_tokens: Vec<usize>,
    /// Number of `Decoder::forward_batch` calls made (prefill counts as
    /// one call, each subsequent accept/reject round counts as one
    /// more, regardless of how many tokens that round produced).
    pub forward_calls: usize,
    /// Total tokens produced across all rounds -- always equal to
    /// `generated_tokens.len()`, kept as a separate field so the ratio
    /// `tokens_generated / forward_calls` (the actual speedup metric)
    /// is easy to read directly off this struct.
    pub tokens_generated: usize,
    /// Verification rounds: `forward_calls` minus the prefill call.
    /// This is the denominator of the published *acceptance length*
    /// metric.
    pub verification_steps: usize,
    /// Draft tokens the target actually evaluated. Positions past a
    /// rejection are never evaluated, so they are not counted here --
    /// counting them would deflate the accept rate by the drafter's
    /// block size rather than by its accuracy.
    pub drafted_tokens: usize,
    /// Draft tokens accepted.
    pub accepted_tokens: usize,
    /// Per drafted position (0 = first token after the anchor), how
    /// many times that position was *evaluated*, i.e. reached without
    /// an earlier rejection ending the round.
    pub evaluated_at_position: Vec<usize>,
    /// Per drafted position, how many times it was accepted.
    pub accepted_at_position: Vec<usize>,
}

impl SpeculativeDecodeResult {
    /// Average tokens produced per `forward_batch` call. 1.0 means
    /// speculation never helped (every round produced exactly the
    /// anchor token); higher means draft tokens were accepted.
    pub fn tokens_per_call(&self) -> f64 {
        if self.forward_calls == 0 {
            0.0
        } else {
            self.tokens_generated as f64 / self.forward_calls as f64
        }
    }

    /// The published metric: completion tokens per verification step.
    /// `None` when nothing was verified (an empty run), because a zero
    /// there would read as "speculation made things worse" rather than
    /// "speculation did not run".
    ///
    /// Deliberately not the same number as [`Self::tokens_per_call`],
    /// which charges the one-off prefill call against the average and
    /// so understates a short run.
    pub fn acceptance_length(&self) -> Option<f64> {
        if self.verification_steps == 0 {
            None
        } else {
            Some(self.tokens_generated as f64 / self.verification_steps as f64)
        }
    }

    /// Fraction of drafted positions accepted, over all positions.
    pub fn accept_rate(&self) -> Option<f64> {
        if self.drafted_tokens == 0 {
            None
        } else {
            Some(self.accepted_tokens as f64 / self.drafted_tokens as f64)
        }
    }

    /// Accept rate at each drafted position, conditional on that
    /// position having been reached.
    ///
    /// A single mean cannot distinguish a drafter that is uniformly
    /// mediocre from one that is excellent at position 0 and useless by
    /// position 7, and the two want opposite responses (raise the block
    /// size, or lower it). The published motivation for dFlash2's
    /// two-tap convolution is exactly this curve falling from 99.5% to
    /// 87.8% across a block, so it has to be visible per position or
    /// the diagnosis is not testable here.
    pub fn accept_rate_per_position(&self) -> Vec<f64> {
        self.evaluated_at_position
            .iter()
            .zip(self.accepted_at_position.iter())
            .map(|(&seen, &ok)| {
                if seen == 0 {
                    0.0
                } else {
                    ok as f64 / seen as f64
                }
            })
            .collect()
    }

    fn record_position(&mut self, position: usize, accepted: bool) {
        if self.evaluated_at_position.len() <= position {
            self.evaluated_at_position.resize(position + 1, 0);
            self.accepted_at_position.resize(position + 1, 0);
        }
        self.evaluated_at_position[position] += 1;
        self.drafted_tokens += 1;
        if accepted {
            self.accepted_at_position[position] += 1;
            self.accepted_tokens += 1;
        }
    }
}

/// Greedy speculative decode over a **fresh** KV cache, with
/// prompt-lookup drafting. Thin wrapper over
/// [`speculative_decode_with`], kept for callers that want the original
/// no-options shape.
pub fn speculative_decode<D: Drafter + ?Sized>(
    decoder: &Decoder,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
    kv_caches: &mut [KvCache],
    drafter: &mut D,
) -> SpeculativeDecodeResult {
    speculative_decode_observed(
        decoder,
        prompt_tokens,
        kv_caches,
        drafter,
        &mut |_| true,
        &SpeculativeOptions {
            max_new_tokens,
            ..SpeculativeOptions::default()
        },
    )
}

/// Decodes `options.max_new_tokens` tokens, using `drafter` to propose
/// candidate continuations and verifying each block in a single batched
/// call.
///
/// `prompt_tokens` is processed as one prefill batch (one
/// `forward_batch` call for the whole prompt, not one per prompt token
/// -- itself a real saving independent of speculation).
///
/// # Cache state
///
/// `kv_caches` may be warm. `options.start_pos` states where
/// `prompt_tokens` begins, and must equal every cache's current
/// `seq_len` -- the caches hold exactly the context preceding the
/// prompt, and this function appends to them. On return they hold that
/// context plus the prompt plus every generated token *except* the last
/// (whose KV is not computed until it is fed, which the next call does
/// for free by passing it as the anchor).
///
/// # Output distribution
///
/// Identical to plain token-at-a-time sampling from `decoder` with
/// `options.sampling`, at any temperature. See the module docs.
pub fn speculative_decode_with<D: Drafter + ?Sized>(
    decoder: &Decoder,
    prompt_tokens: &[usize],
    kv_caches: &mut [KvCache],
    drafter: &mut D,
    options: &SpeculativeOptions,
) -> SpeculativeDecodeResult {
    speculative_decode_observed(
        decoder,
        prompt_tokens,
        kv_caches,
        drafter,
        &mut |_| true,
        options,
    )
}

/// [`speculative_decode_with`], plus an observer called once per
/// committed token, in order, which can end the run by returning
/// `false`.
///
/// This exists so a caller that streams output, or that stops on an EOS
/// or a stop string, does not have to write a second copy of the
/// verification loop. Copying that loop to vary it is how this project
/// lost five model features from one duplicated decode path, and the
/// rejection rule is the last code in the tree that should be
/// duplicated: a subtly different copy is still lossless-looking.
///
/// The observer sees a token only once it is COMMITTED, so it never
/// sees a draft that was rejected. Returning `false` ends generation
/// after the current verification block finishes, which keeps the KV
/// caches in the single consistent state this function documents;
/// `generated_tokens` is truncated at the token that said stop, so the
/// caller's output and the returned tokens agree.
pub fn speculative_decode_observed<D: Drafter + ?Sized>(
    decoder: &Decoder,
    prompt_tokens: &[usize],
    kv_caches: &mut [KvCache],
    drafter: &mut D,
    on_token: &mut dyn FnMut(usize) -> bool,
    options: &SpeculativeOptions,
) -> SpeculativeDecodeResult {
    assert!(!prompt_tokens.is_empty(), "prompt must not be empty");
    // A rejected draft rolls every cache back to the last accepted
    // position, which a Mamba layer's state cannot do
    // (`frink_core::recurrent_state`); the CLI refuses `--model-draft`
    // on such a model before reaching here, and this is the backstop.
    assert!(
        !decoder.config.has_recurrent_layers(),
        "speculative decoding rolls the KV caches back on a rejected draft, and a recurrent \
         layer's state cannot be rolled back to a middle position"
    );
    for cache in kv_caches.iter() {
        assert_eq!(
            cache.positions(),
            options.start_pos,
            "start_pos must be the caches' current length: they hold exactly the \
             context preceding the prompt"
        );
    }

    let mut result = SpeculativeDecodeResult::default();
    if options.max_new_tokens == 0 {
        return result;
    }

    let mut rng = Sampler::new(options.seed);
    let mut history: Vec<usize> = prompt_tokens.to_vec();
    let mut generated: Vec<usize> = Vec::with_capacity(options.max_new_tokens);

    // Prefill: one batched call over the whole prompt.
    let (prefill_logits, prefill_hidden) =
        decoder.forward_batch_with_hidden(prompt_tokens, options.start_pos, kv_caches);
    result.forward_calls += 1;
    let last = prefill_logits
        .last()
        .expect("prompt_tokens is non-empty, so forward_batch returns at least one logits vector");
    let mut target_hidden = prefill_hidden.last().cloned().unwrap_or_default();

    // `pending` is decided but its KV is not in the cache yet: it is
    // fed as the anchor of the next batch, which is what lets one
    // forward call both commit it and verify a block after it.
    let mut pending = {
        // Split ONE structure rather than pairing `history` with the
        // separate `generated` vector: the two would then have to agree
        // about every push, which is exactly the shape this fix exists
        // to remove.
        let (seen_prompt, seen_generated) = history.split_at(prompt_tokens.len());
        let xtc_roll = rng.xtc_roll(&options.sampling);
        let probs = sampling_distribution(
            last,
            &options.sampling,
            PenaltyWindow::new(seen_prompt, seen_generated),
            xtc_roll,
        );
        rng.sample_from(&probs)
    };
    let mut pos = options.start_pos + prompt_tokens.len();

    // Set when the observer asks to stop. The current block still runs
    // to completion so the caches end in the one state this function
    // documents, and `generated` is cut back to this length afterwards.
    let mut stop_at: Option<usize> = None;

    loop {
        generated.push(pending);
        history.push(pending);
        if !on_token(pending) {
            stop_at = Some(generated.len());
            break;
        }
        if generated.len() == options.max_new_tokens {
            break;
        }
        // One short of the remaining budget on purpose: the last token
        // of the run is always committed as an anchor at the top of the
        // loop, never as an accepted draft. That keeps the cache in
        // exactly one state on return (see the doc comment) instead of
        // one state when the budget runs out on an anchor and another
        // when it runs out mid-block -- and it costs nothing, because
        // the drafted position it gives up is one whose KV would have
        // had to be discarded anyway.
        let draft_budget = options.max_new_tokens - generated.len() - 1;

        // The drafter is asked to continue a history that *includes*
        // `pending`, because the first drafted token lands at pos + 1.
        let mut draft = drafter.propose(&history, &target_hidden, draft_budget);
        draft.truncate(draft_budget);

        let mut batch = Vec::with_capacity(1 + draft.len());
        batch.push(pending);
        batch.extend_from_slice(draft.tokens());

        let (batch_logits, batch_hidden) =
            decoder.forward_batch_with_hidden(&batch, pos, kv_caches);
        result.forward_calls += 1;
        result.verification_steps += 1;

        // batch_logits[i] is the target's distribution for the position
        // right after batch[i], i.e. the distribution draft token i
        // should be judged against.
        let mut accepted = 0usize;
        let mut replacement: Option<usize> = None;
        for (i, (&token, dist)) in draft.tokens().iter().zip(draft.dists()).enumerate() {
            let (seen_prompt, seen_generated) = history.split_at(prompt_tokens.len());
            let xtc_roll = rng.xtc_roll(&options.sampling);
            let target = sampling_distribution(
                &batch_logits[i],
                &options.sampling,
                PenaltyWindow::new(seen_prompt, seen_generated),
                xtc_roll,
            );
            match accept_or_resample(&target, dist, token, &mut rng) {
                None => {
                    result.record_position(i, true);
                    accepted += 1;
                    history.push(token);
                    generated.push(token);
                    if stop_at.is_none() && !on_token(token) {
                        // Keep verifying the rest of the block: the
                        // loop below truncates the caches to exactly
                        // what was committed, and leaving early here
                        // would skip that.
                        stop_at = Some(generated.len());
                    }
                }
                Some(resampled) => {
                    result.record_position(i, false);
                    replacement = Some(resampled);
                    break;
                }
            }
        }

        // Every position past `accepted` was computed from a token that
        // is not going to be committed, so its KV is wrong. Lengths are
        // absolute, which is what makes a warm cache work: `pos` is
        // already offset by start_pos.
        let committed_len = pos + 1 + accepted;
        if accepted < draft.len() {
            for cache in kv_caches.iter_mut() {
                cache.truncate(committed_len);
            }
        }
        // POSITIONS: `committed_len` is absolute, offset by start_pos.
        debug_assert!(kv_caches.iter().all(|c| c.positions() == committed_len));

        target_hidden = batch_hidden[accepted].clone();
        pending = match replacement {
            // A rejected position was resampled from the residual; that
            // token is committed and becomes the next anchor.
            Some(tok) => tok,
            // Every draft token was accepted, so the last row of the
            // batch predicts a genuinely new position -- the free bonus
            // token that makes a fully-accepted block worth `k + 1`.
            None => {
                let (seen_prompt, seen_generated) = history.split_at(prompt_tokens.len());
                let xtc_roll = rng.xtc_roll(&options.sampling);
                let probs = sampling_distribution(
                    &batch_logits[accepted],
                    &options.sampling,
                    PenaltyWindow::new(seen_prompt, seen_generated),
                    xtc_roll,
                );
                rng.sample_from(&probs)
            }
        };
        pos = committed_len;
        if stop_at.is_some() {
            break;
        }
        debug_assert!(generated.len() < options.max_new_tokens);
    }

    if let Some(len) = stop_at {
        // The observer said stop at this token. Everything after it was
        // produced by a block that had already been dispatched, and the
        // caller never saw it.
        generated.truncate(len);
    } else {
        debug_assert_eq!(generated.len(), options.max_new_tokens);
    }
    result.tokens_generated = generated.len();
    result.generated_tokens = generated;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::glm_5_2;
    use crate::ModelConfig;
    use std::cell::RefCell;

    fn tiny_test_config() -> ModelConfig {
        let mut cfg = glm_5_2();
        cfg.hidden_dim = 16;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 4;
        cfg.moe.hidden_dim = 16;
        cfg.moe.n_experts = 6;
        cfg.moe.n_experts_active = 2;
        cfg.moe.n_shared_experts = 1;
        cfg.moe.expert_ffn_dim = 8;
        cfg
    }

    fn caches(decoder: &Decoder) -> Vec<KvCache> {
        decoder.config.new_kv_caches()
    }

    fn argmax(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// A drafter that always proposes the same block, with a
    /// configurable draft distribution, and records the history it was
    /// asked about.
    struct FixedDrafter {
        block: DraftBlock,
        seen_history: RefCell<Vec<Vec<usize>>>,
        seen_hidden_len: RefCell<Vec<usize>>,
    }

    impl FixedDrafter {
        fn new(block: DraftBlock) -> Self {
            FixedDrafter {
                block,
                seen_history: RefCell::new(Vec::new()),
                seen_hidden_len: RefCell::new(Vec::new()),
            }
        }
    }

    impl Drafter for FixedDrafter {
        fn propose(
            &mut self,
            history: &[usize],
            target_hidden: &[f32],
            max_len: usize,
        ) -> DraftBlock {
            self.seen_history.borrow_mut().push(history.to_vec());
            self.seen_hidden_len.borrow_mut().push(target_hidden.len());
            let mut block = self.block.clone();
            block.truncate(max_len);
            block
        }
    }

    // ---- PromptLookupSpeculator tests ----

    #[test]
    fn proposes_the_continuation_after_a_real_repeat() {
        let mut spec = PromptLookupSpeculator::new(2, 4);
        // "...1 2 3 4 5 9 9 9 1 2" -> earlier "1 2" occurs at the very
        // start (indices 0-1); the 4 tokens that followed it are
        // "3 4 5 9" (capped at max_draft_len=4).
        let history = vec![1, 2, 3, 4, 5, 9, 9, 9, 1, 2];
        assert_eq!(spec.propose_tokens(&history), vec![3, 4, 5, 9]);
        assert_eq!(
            spec.propose(&history, &[], 8),
            DraftBlock::deterministic(vec![3, 4, 5, 9])
        );
    }

    #[test]
    fn respects_max_draft_len() {
        let spec = PromptLookupSpeculator::new(2, 2);
        let history = vec![1, 2, 3, 4, 5, 6, 7, 1, 2];
        assert_eq!(spec.propose_tokens(&history), vec![3, 4]);
    }

    #[test]
    fn returns_empty_when_no_earlier_match_exists() {
        let spec = PromptLookupSpeculator::new(2, 4);
        let history = vec![1, 2, 3, 4, 5];
        assert_eq!(spec.propose_tokens(&history), Vec::<usize>::new());
    }

    #[test]
    fn returns_empty_when_history_too_short() {
        let spec = PromptLookupSpeculator::new(3, 4);
        let history = vec![1, 2, 3];
        assert_eq!(spec.propose_tokens(&history), Vec::<usize>::new());
    }

    #[test]
    fn finds_the_most_recent_match_when_several_exist() {
        let spec = PromptLookupSpeculator::new(1, 3);
        // needle = [9]. Earlier occurrences at index 0 (-> [8,7,6]) and
        // index 4 (-> [5,4,9]); most recent (index 4) should win.
        let history = vec![9, 8, 7, 6, 9, 5, 4, 9];
        assert_eq!(spec.propose_tokens(&history), vec![5, 4, 9]);
    }

    #[test]
    fn the_trait_caps_a_block_at_the_callers_budget() {
        let mut spec = PromptLookupSpeculator::new(2, 4);
        let history = vec![1, 2, 3, 4, 5, 9, 9, 9, 1, 2];
        let block = spec.propose(&history, &[], 2);
        assert_eq!(block.tokens(), &[3, 4]);
        assert_eq!(block.dists().len(), 2);
    }

    // ---- the rejection rule ----

    #[test]
    fn a_draft_at_least_as_likely_under_the_target_is_always_accepted() {
        let target = vec![0.6f32, 0.3, 0.1];
        let draft = DraftDist::from_dense(&[0.5, 0.4, 0.1]);
        let mut rng = Sampler::new(1);
        for _ in 0..100 {
            // p(0)=0.6 >= q(0)=0.5, so token 0 is never rejected.
            assert_eq!(accept_or_resample(&target, &draft, 0, &mut rng), None);
        }
    }

    #[test]
    fn a_draft_the_target_rules_out_is_always_rejected() {
        let target = vec![0.5f32, 0.5, 0.0];
        let draft = DraftDist::deterministic(2);
        let mut rng = Sampler::new(2);
        for _ in 0..50 {
            let replacement = accept_or_resample(&target, &draft, 2, &mut rng);
            let tok = replacement.expect("p(2) = 0 means token 2 can never be accepted");
            assert!(tok < 2, "residual must never resample the rejected token");
        }
    }

    #[test]
    fn resampling_reproduces_the_target_distribution() {
        // THE invariant. Draw a token from the draft distribution, run
        // it through the accept/reject rule, and the result must be
        // distributed as the TARGET, no matter how bad the draft is.
        //
        // A test that only checked "it runs" would pass on the old
        // argmax rule, which concentrates mass on the target's argmax
        // and is not the target distribution at all.
        let target = vec![0.30f32, 0.25, 0.20, 0.15, 0.07, 0.03];
        let drafts = [
            // A drafter that is simply wrong about which token is likely.
            DraftDist::from_dense(&[0.02, 0.03, 0.05, 0.10, 0.30, 0.50]),
            // A deterministic drafter, i.e. prompt lookup.
            DraftDist::deterministic(3),
            // A drafter whose support misses most of the target's.
            DraftDist::from_support(vec![(0, 0.5), (5, 0.5)]),
            // A perfect drafter.
            DraftDist::from_dense(&target),
        ];
        let draws = 200_000;
        for (d, draft) in drafts.iter().enumerate() {
            let mut rng = Sampler::new(0xA11CE + d as u64);
            let mut counts = vec![0usize; target.len()];
            for _ in 0..draws {
                // Sample the draft token from the draft distribution --
                // the rule is only lossless when q is honest about
                // where the token came from.
                let dense = {
                    let mut v = vec![0.0f32; target.len()];
                    for &(t, p) in draft.support() {
                        v[t] = p;
                    }
                    v
                };
                let x = rng.sample_from(&dense);
                let out = accept_or_resample(&target, draft, x, &mut rng).unwrap_or(x);
                counts[out] += 1;
            }
            let tv: f64 = counts
                .iter()
                .enumerate()
                .map(|(i, &c)| (c as f64 / draws as f64 - target[i] as f64).abs())
                .sum::<f64>()
                / 2.0;
            assert!(
                tv < 0.01,
                "draft {d}: speculative output distribution differs from the target \
                 (total variation {tv:.4}); counts={counts:?}"
            );
        }
    }

    // ---- speculative_decode correctness tests ----

    #[test]
    fn speculative_decode_matches_greedy_token_for_token() {
        // Quality-neutrality at temperature 0: token-for-token identity
        // against a plain sequential forward_token loop on a
        // separately constructed but identically-seeded decoder.
        let cfg = tiny_test_config();
        let vocab = 8;
        let prompt = vec![1usize, 2, 3, 4, 1, 2];
        let max_new = 6;

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut caches_a = caches(&decoder_a);
        let mut speculator = PromptLookupSpeculator::new(2, 3);
        let result =
            speculative_decode(&decoder_a, &prompt, max_new, &mut caches_a, &mut speculator);

        let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        let mut caches_b = caches(&decoder_b);
        let mut pending = decoder_b
            .forward_batch(&prompt, 0, &mut caches_b)
            .pop()
            .unwrap();
        let mut greedy = Vec::with_capacity(max_new);
        for pos in (prompt.len()..).take(max_new) {
            let tok = argmax(&pending);
            greedy.push(tok);
            pending = decoder_b.forward_token(tok, pos, &mut caches_b);
        }

        assert_eq!(
            result.generated_tokens, greedy,
            "speculative decode must produce exactly the same tokens as plain greedy decode"
        );
    }

    /// The exact per-position marginal distributions of plain
    /// token-at-a-time sampling from `decoder`, by enumerating every
    /// prefix rather than sampling them. Only tractable because the
    /// test model has a 6-token vocabulary and the horizon is 3, but
    /// worth it: the speculative sampler is then compared against the
    /// truth, not against a second noisy estimate of it.
    fn exact_marginals(
        decoder: &Decoder,
        prompt: &[usize],
        params: &SamplingParams,
        depth: usize,
        vocab: usize,
    ) -> Vec<Vec<f64>> {
        #[allow(clippy::too_many_arguments)]
        fn walk(
            decoder: &Decoder,
            kv: &[KvCache],
            logits: &[f32],
            history: &mut Vec<usize>,
            weight: f64,
            level: usize,
            depth: usize,
            pos: usize,
            params: &SamplingParams,
            marginals: &mut [Vec<f64>],
        ) {
            // `history` is already prompt-then-generated, and the
            // window only ever reads the tail of the two halves
            // together, so the whole sequence goes in the first one.
            // This helper enumerates the exact marginal over every
            // continuation, so it must not depend on a random draw:
            // `None` is correct here and only here, because the params
            // it is called with never enable XTC.
            debug_assert!(
                !params.xtc_can_fire(),
                "the exact marginal is not defined under XTC"
            );
            let probs =
                sampling_distribution(logits, params, PenaltyWindow::new(history, &[]), None);
            for (token, &p) in probs.iter().enumerate() {
                if p <= 0.0 {
                    continue;
                }
                marginals[level][token] += weight * p as f64;
                if level + 1 == depth {
                    continue;
                }
                let mut branch: Vec<KvCache> = kv.to_vec();
                let next = decoder.forward_token(token, pos, &mut branch);
                history.push(token);
                walk(
                    decoder,
                    &branch,
                    &next,
                    history,
                    weight * p as f64,
                    level + 1,
                    depth,
                    pos + 1,
                    params,
                    marginals,
                );
                history.pop();
            }
        }

        let mut marginals = vec![vec![0.0f64; vocab]; depth];
        let mut kv = caches(decoder);
        let logits = decoder.forward_batch(prompt, 0, &mut kv).pop().unwrap();
        let mut history = prompt.to_vec();
        walk(
            decoder,
            &kv,
            &logits,
            &mut history,
            1.0,
            0,
            depth,
            prompt.len(),
            params,
            &mut marginals,
        );
        marginals
    }

    /// Speculation and plain token-at-a-time decoding must agree about
    /// WHICH tokens the penalties look back over, including the prompt.
    ///
    /// This is issue #55's other half. `SpeculativeOptions` used to
    /// carry a `penalty_history_start` knob whose only job was to let a
    /// caller line the two paths up by hand, which meant nothing failed
    /// when they drifted -- and `--model-draft` shipped setting it to
    /// `prompt.len()` because the plain loop penalised the generated
    /// tokens alone. Both now go through `PenaltyWindow`, and this test
    /// is what notices if one of them stops.
    ///
    /// Greedy on purpose: the assertion is token-for-token equality, so
    /// a one-token disagreement in the window is a hard failure rather
    /// than a shift in a sampled distribution. The prompt repeats
    /// tokens 1 and 2, so the penalty has something to bite on from the
    /// very first generated position.
    #[test]
    fn speculation_and_plain_decoding_penalise_the_same_window() {
        let cfg = tiny_test_config();
        let vocab = 6;
        let prompt = vec![0usize, 1, 2, 3, 1];
        let max_new = 6;
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 3.0,
            penalty_last_n: 8,
            ..SamplingParams::default()
        };

        let decoder = Decoder::new_random_small(cfg, 2, vocab);

        // Plain token-at-a-time decoding, penalising over the same
        // window `Sampler::sample` would use on any decode loop.
        let mut kv = caches(&decoder);
        let mut pos = 0usize;
        let mut logits = Vec::new();
        for &tok in &prompt {
            logits = decoder.forward_token(tok, pos, &mut kv);
            pos += 1;
        }
        let mut sampler = Sampler::new(7);
        let mut plain: Vec<usize> = Vec::new();
        for _ in 0..max_new {
            let next = sampler.sample(&logits, &params, PenaltyWindow::new(&prompt, &plain));
            plain.push(next);
            logits = decoder.forward_token(next, pos, &mut kv);
            pos += 1;
        }

        let mut speculator = PromptLookupSpeculator::new(2, 3);
        let mut spec_kv = caches(&decoder);
        let out = speculative_decode_with(
            &decoder,
            &prompt,
            &mut spec_kv,
            &mut speculator,
            &SpeculativeOptions {
                max_new_tokens: max_new,
                sampling: params.clone(),
                seed: 7,
                ..SpeculativeOptions::default()
            },
        );
        assert_eq!(
            out.generated_tokens, plain,
            "speculation changed the text at --repeat-penalty {}",
            params.repetition_penalty
        );

        // And the penalty is doing something here, or the equality
        // above is satisfied by a window nobody reads.
        let mut off = params.clone();
        off.repetition_penalty = 1.0;
        let mut kv = caches(&decoder);
        let mut pos = 0usize;
        let mut logits = Vec::new();
        for &tok in &prompt {
            logits = decoder.forward_token(tok, pos, &mut kv);
            pos += 1;
        }
        let mut sampler = Sampler::new(7);
        let mut unpenalised: Vec<usize> = Vec::new();
        for _ in 0..max_new {
            let next = sampler.sample(&logits, &off, PenaltyWindow::new(&prompt, &unpenalised));
            unpenalised.push(next);
            logits = decoder.forward_token(next, pos, &mut kv);
            pos += 1;
        }
        assert_ne!(
            unpenalised, plain,
            "the penalty must change this generation, or the agreement above proves nothing"
        );
    }

    #[test]
    fn speculative_decode_at_temperature_matches_plain_sampling() {
        // The end-to-end half of the losslessness claim, and the one
        // the old argmax accept test fails: at temperature > 0 the
        // per-position output distribution of speculative decoding must
        // equal that of plain token-at-a-time sampling from the same
        // target. Argmax matching passes every other test in this file
        // and fails this one, because accepting a draft only when it is
        // the target's most likely token pushes mass onto the argmax.
        let cfg = tiny_test_config();
        let vocab = 6;
        let prompt = vec![1usize, 2, 3, 1, 2];
        let max_new = 3;
        let params = SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let seeds = 4_000u64;

        let decoder = Decoder::new_random_small(cfg, 1, vocab);
        let mut speculator = PromptLookupSpeculator::new(2, 3);
        let exact = exact_marginals(&decoder, &prompt, &params, max_new, vocab);

        let mut spec_counts = vec![vec![0usize; vocab]; max_new];
        for seed in 0..seeds {
            let mut kv = caches(&decoder);
            let out = speculative_decode_with(
                &decoder,
                &prompt,
                &mut kv,
                &mut speculator,
                &SpeculativeOptions {
                    max_new_tokens: max_new,
                    sampling: params.clone(),
                    seed,
                    ..SpeculativeOptions::default()
                },
            );
            for (i, &t) in out.generated_tokens.iter().enumerate() {
                spec_counts[i][t] += 1;
            }
        }

        for i in 0..max_new {
            let tv: f64 = (0..vocab)
                .map(|t| (spec_counts[i][t] as f64 / seeds as f64 - exact[i][t]).abs())
                .sum::<f64>()
                / 2.0;
            assert!(
                tv < 0.03,
                "position {i}: speculative sampling drifted from the target's own \
                 distribution (total variation {tv:.4})\n  speculative = {:?}\n  exact = {:?}",
                spec_counts[i]
                    .iter()
                    .map(|&c| c as f64 / seeds as f64)
                    .collect::<Vec<_>>(),
                exact[i]
            );
        }
    }

    #[test]
    fn speculative_decode_saves_real_calls_when_drafts_hit() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let prompt = vec![1usize, 2, 3, 1, 2];
        let max_new = 8;

        let decoder = Decoder::new_random_small(cfg, 2, vocab);
        let mut kv = caches(&decoder);
        let mut speculator = PromptLookupSpeculator::new(2, 4);
        let result = speculative_decode(&decoder, &prompt, max_new, &mut kv, &mut speculator);

        assert_eq!(result.tokens_generated, max_new);
        // Plain sequential decode needs exactly `max_new` calls here:
        // one prefill plus one per token except the last, whose KV is
        // never needed. Speculation must never need more.
        assert!(
            result.forward_calls <= max_new,
            "speculative decode must never need MORE forward_batch calls than plain \
             sequential decode would (calls={}, tokens={})",
            result.forward_calls,
            max_new
        );
    }

    #[test]
    fn speculative_decode_with_no_repeats_falls_back_to_one_token_per_call() {
        // A prompt with no internal repeats at all must still work
        // correctly, just without any speedup.
        let cfg = tiny_test_config();
        let vocab = 8;
        let prompt = vec![1usize, 2, 3];
        let max_new = 5;

        let decoder = Decoder::new_random_small(cfg, 2, vocab);
        let mut kv = caches(&decoder);
        let mut speculator = PromptLookupSpeculator::new(10, 4); // ngram far longer than any possible history
        let result = speculative_decode(&decoder, &prompt, max_new, &mut kv, &mut speculator);

        assert_eq!(result.tokens_generated, max_new);
        assert_eq!(
            result.forward_calls,
            1 + max_new - 1,
            "prefill (1 call) + one call per token, minus the last token, whose KV is \
             never needed because generation stopped"
        );
        assert_eq!(result.drafted_tokens, 0);
        assert_eq!(result.accept_rate(), None);
    }

    // ---- drafter trait plumbing ----

    #[test]
    fn the_drafter_is_asked_to_continue_the_anchor_token() {
        // The block the drafter proposes lands *after* the pending
        // token, so the history it sees must already contain it.
        // Drafting from a history that stopped one token short would
        // shift every proposal by one position and quietly halve the
        // accept rate without breaking any output-correctness test.
        let cfg = tiny_test_config();
        let decoder = Decoder::new_random_small(cfg, 2, 8);
        let mut kv = caches(&decoder);
        let prompt = vec![1usize, 2, 3];
        let mut drafter = FixedDrafter::new(DraftBlock::deterministic(vec![5, 6]));

        let result = speculative_decode(&decoder, &prompt, 4, &mut kv, &mut drafter);

        let seen = drafter.seen_history.borrow();
        assert!(!seen.is_empty(), "the drafter must actually be consulted");
        for (round, history) in seen.iter().enumerate() {
            assert_eq!(
                history.len(),
                prompt.len() + round + 1,
                "round {round}: history must grow by the committed tokens"
            );
            assert_eq!(
                history[..prompt.len()],
                prompt[..],
                "the prompt must stay at the front of the drafter's history"
            );
        }
        assert_eq!(seen[0][prompt.len()], result.generated_tokens[0]);
    }

    #[test]
    fn the_drafter_receives_the_targets_hidden_state() {
        // The conditioning tensor dFlash/EAGLE need. It is already
        // computed by verification; the trait exists so it stops being
        // discarded.
        let cfg = tiny_test_config();
        let hidden_dim = cfg.hidden_dim;
        let decoder = Decoder::new_random_small(cfg, 2, 8);
        let mut kv = caches(&decoder);
        let mut drafter = FixedDrafter::new(DraftBlock::deterministic(vec![5, 6]));

        speculative_decode(&decoder, &[1usize, 2, 3], 4, &mut kv, &mut drafter);

        let lens = drafter.seen_hidden_len.borrow();
        assert!(!lens.is_empty());
        for len in lens.iter() {
            assert_eq!(
                *len, hidden_dim,
                "every round must pass a full target hidden state, not an empty slice"
            );
        }
    }

    // ---- cache resume + rollback arithmetic ----

    #[test]
    fn resuming_a_warm_cache_gives_the_same_tokens_as_one_fresh_run() {
        // The serving shape: a prefix cache hands the decode loop a
        // cache that already holds part of the context.
        let cfg = tiny_test_config();
        let vocab = 8;
        let decoder = Decoder::new_random_small(cfg, 2, vocab);
        let mut speculator = PromptLookupSpeculator::new(2, 3);
        let full_prompt = vec![1usize, 2, 3, 4, 1, 2];
        let max_new = 6;

        let mut fresh = caches(&decoder);
        let cold = speculative_decode(&decoder, &full_prompt, max_new, &mut fresh, &mut speculator);

        // Warm: feed the first 4 prompt tokens through the decoder
        // first, then resume speculative decoding from position 4.
        let split = 4;
        let mut warm = caches(&decoder);
        decoder.forward_batch(&full_prompt[..split], 0, &mut warm);
        let resumed = speculative_decode_with(
            &decoder,
            &full_prompt[split..],
            &mut warm,
            &mut speculator,
            &SpeculativeOptions {
                max_new_tokens: max_new,
                start_pos: split,
                ..SpeculativeOptions::default()
            },
        );

        assert_eq!(
            resumed.generated_tokens, cold.generated_tokens,
            "resuming a warm cache must not change the output"
        );
    }

    #[test]
    fn rolls_back_to_absolute_positions_on_a_warm_cache() {
        // Rollback lengths are absolute cache lengths, not offsets from
        // the start of this call. With a warm cache the two differ by
        // start_pos, and a rollback that used the offset would truncate
        // into the caller's context. Forced rejections every round make
        // the rollback path run every round.
        let cfg = tiny_test_config();
        let decoder = Decoder::new_random_small(cfg, 2, 8);
        // Token 7 is a fixed guess; whether it is accepted is up to the
        // model, but the invariant below holds either way.
        let mut drafter = FixedDrafter::new(DraftBlock::deterministic(vec![7, 7, 7]));
        let context = vec![1usize, 2, 3, 4];
        let prompt = vec![5usize, 6];
        let max_new = 6;

        let mut kv = caches(&decoder);
        decoder.forward_batch(&context, 0, &mut kv);
        assert_eq!(kv[0].positions(), context.len());

        let result = speculative_decode_with(
            &decoder,
            &prompt,
            &mut kv,
            &mut drafter,
            &SpeculativeOptions {
                max_new_tokens: max_new,
                start_pos: context.len(),
                ..SpeculativeOptions::default()
            },
        );

        assert_eq!(result.tokens_generated, max_new);
        // Exact invariant: the cache holds the context, the prompt and
        // every generated token except the last (whose KV is not
        // computed until it is fed).
        let expected = context.len() + prompt.len() + result.tokens_generated - 1;
        for cache in kv.iter() {
            assert_eq!(
                cache.positions(),
                expected,
                "cache length must be absolute: context {} + prompt {} + generated {} - 1",
                context.len(),
                prompt.len(),
                result.tokens_generated
            );
        }
    }

    #[test]
    fn a_resumed_run_continues_a_previous_one() {
        // Two back-to-back calls on the same caches must equal one long
        // call: this is what "not a demo" means for the serving path.
        let cfg = tiny_test_config();
        let decoder = Decoder::new_random_small(cfg, 2, 8);
        let mut speculator = PromptLookupSpeculator::new(2, 3);
        let prompt = vec![1usize, 2, 3, 4, 1, 2];

        let mut one = caches(&decoder);
        let long = speculative_decode(&decoder, &prompt, 8, &mut one, &mut speculator);

        let mut kv = caches(&decoder);
        let first = speculative_decode(&decoder, &prompt, 4, &mut kv, &mut speculator);
        // The last generated token's KV is not in the cache yet, so it
        // is the first token of the continuation's "prompt".
        let resume_prompt = vec![*first.generated_tokens.last().unwrap()];
        let start = prompt.len() + first.tokens_generated - 1;
        let second = speculative_decode_with(
            &decoder,
            &resume_prompt,
            &mut kv,
            &mut speculator,
            &SpeculativeOptions {
                max_new_tokens: 5,
                start_pos: start,
                ..SpeculativeOptions::default()
            },
        );

        let mut stitched = first.generated_tokens.clone();
        stitched.pop(); // re-fed as the continuation's prompt
        stitched.extend_from_slice(&second.generated_tokens);
        assert_eq!(
            &stitched[..8],
            &long.generated_tokens[..],
            "a decode split across two calls must equal the same decode in one"
        );
    }

    #[test]
    #[should_panic(expected = "start_pos must be the caches' current length")]
    fn a_mismatched_start_pos_is_refused_rather_than_silently_wrong() {
        let cfg = tiny_test_config();
        let decoder = Decoder::new_random_small(cfg, 2, 8);
        let mut kv = caches(&decoder);
        decoder.forward_batch(&[1usize, 2, 3], 0, &mut kv);
        let mut speculator = PromptLookupSpeculator::new(2, 2);
        speculative_decode(&decoder, &[4usize, 5], 2, &mut kv, &mut speculator);
    }

    // ---- acceptance metrics ----

    #[test]
    fn per_position_accept_rates_expose_suffix_decay() {
        // A drafter whose first guess is always right and whose later
        // guesses are always wrong has the same mean accept rate as one
        // that is uniformly mediocre. Only the per-position curve tells
        // them apart, which is the whole reason it exists.
        let mut result = SpeculativeDecodeResult::default();
        for _ in 0..100 {
            result.record_position(0, true);
            result.record_position(1, false);
        }
        result.verification_steps = 100;
        result.tokens_generated = 200;

        assert_eq!(result.accept_rate(), Some(0.5));
        assert_eq!(result.accept_rate_per_position(), vec![1.0, 0.0]);
        assert_eq!(result.acceptance_length(), Some(2.0));
    }

    #[test]
    fn positions_after_a_rejection_are_not_counted_as_drafted() {
        // A round that rejects at position 0 never evaluates positions
        // 1..k. Counting them would report an accept rate that falls
        // with the block size rather than with the drafter's accuracy.
        let cfg = tiny_test_config();
        let decoder = Decoder::new_random_small(cfg, 2, 8);
        let mut kv = caches(&decoder);
        // Token 7 against a random model: whatever happens, every
        // counted position must have been reachable.
        let mut drafter = FixedDrafter::new(DraftBlock::deterministic(vec![7, 7, 7, 7]));
        let result = speculative_decode(&decoder, &[1usize, 2, 3], 6, &mut kv, &mut drafter);

        let evaluated = &result.evaluated_at_position;
        let accepted = &result.accepted_at_position;
        assert!(
            result.drafted_tokens > result.accepted_tokens,
            "the scenario is pointless unless something was actually rejected \
             (drafted {}, accepted {})",
            result.drafted_tokens,
            result.accepted_tokens
        );
        // The sharp invariant: position i+1 is only reached when
        // position i was accepted, so it can never have been evaluated
        // more often. Merely checking that the counts are
        // non-increasing is not enough -- crediting every position of
        // every proposed block, rejected or not, keeps them
        // non-increasing (it makes them equal) while reporting an
        // accept rate that decays with the block size rather than with
        // the drafter.
        for (i, &seen) in evaluated.iter().enumerate().skip(1) {
            assert!(
                seen <= accepted[i - 1],
                "position {i} was evaluated {seen} times but position {} was only \
                 accepted {} times: evaluated={evaluated:?} accepted={accepted:?}",
                i - 1,
                accepted[i - 1]
            );
        }
        assert_eq!(
            result.drafted_tokens,
            evaluated.iter().sum::<usize>(),
            "drafted_tokens must be the per-position counts' total"
        );
        assert_eq!(
            result.accepted_tokens,
            result.accepted_at_position.iter().sum::<usize>()
        );
        assert!(result.accepted_tokens <= result.drafted_tokens);
    }

    #[test]
    fn acceptance_length_is_reported_per_verification_step_not_per_call() {
        // The published metric divides by verification steps; charging
        // the one-off prefill against it understates short runs.
        let cfg = tiny_test_config();
        let decoder = Decoder::new_random_small(cfg, 2, 8);
        let mut kv = caches(&decoder);
        let mut speculator = PromptLookupSpeculator::new(2, 3);
        let result =
            speculative_decode(&decoder, &[1usize, 2, 3, 1, 2], 6, &mut kv, &mut speculator);

        assert_eq!(result.verification_steps, result.forward_calls - 1);
        let length = result.acceptance_length().unwrap();
        assert!(length >= 1.0, "every verification step commits >= 1 token");
        assert!(
            length > result.tokens_per_call(),
            "acceptance length must not be diluted by the prefill call"
        );
    }

    #[test]
    fn an_empty_run_reports_no_acceptance_length_rather_than_zero() {
        let result = SpeculativeDecodeResult::default();
        assert_eq!(result.acceptance_length(), None);
        assert_eq!(result.accept_rate(), None);
        assert_eq!(result.tokens_per_call(), 0.0);
    }

    /// The observer sees exactly the committed tokens, in order, and
    /// stopping through it truncates the result to the token that said
    /// so.
    ///
    /// This is what lets `frink run` stream and stop on an EOS without
    /// a second copy of the verification loop. A copy is how this
    /// project lost five model features from one duplicated decode
    /// path, and the rejection rule is the last code in the tree that
    /// should be duplicated: a subtly wrong copy still looks lossless.
    #[test]
    fn the_observer_sees_every_committed_token_and_can_end_the_run() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let prompt = vec![1usize, 2, 3, 4, 1, 2];

        let decoder = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut kv = caches(&decoder);
        let mut spec = PromptLookupSpeculator::new(2, 3);

        let mut seen = Vec::new();
        let result = speculative_decode_observed(
            &decoder,
            &prompt,
            &mut kv,
            &mut spec,
            &mut |t| {
                seen.push(t);
                true
            },
            &SpeculativeOptions {
                max_new_tokens: 6,
                start_pos: 0,
                sampling: SamplingParams::default(),
                seed: 0,
            },
        );
        assert_eq!(
            seen, result.generated_tokens,
            "the observer must see exactly what the run returns, in order"
        );

        // Now stop after three tokens.
        let decoder = Decoder::new_random_small(cfg, 2, vocab);
        let mut kv = caches(&decoder);
        let mut spec = PromptLookupSpeculator::new(2, 3);
        let mut count = 0usize;
        let stopped = speculative_decode_observed(
            &decoder,
            &prompt,
            &mut kv,
            &mut spec,
            &mut |_| {
                count += 1;
                count < 3
            },
            &SpeculativeOptions {
                max_new_tokens: 6,
                start_pos: 0,
                sampling: SamplingParams::default(),
                seed: 0,
            },
        );
        assert_eq!(
            stopped.generated_tokens.len(),
            3,
            "the run must end at the token that said stop, not at the end of its block"
        );
        assert_eq!(
            stopped.generated_tokens,
            result.generated_tokens[..3],
            "and the tokens up to the stop must be the ones an unstopped run produced"
        );
    }
}
