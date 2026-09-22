//! The seeded generator every draw in a generation comes off, and the
//! entry points that use it.
//!
//! Split out of `sampling.rs` so the chain runner beside it stays about
//! the chain. The RNG is one concept and a small one, but it is the
//! concept the whole run's reproducibility rests on: a request that
//! passed `seed: 42` must draw the same tokens on every machine, so
//! every draw -- the token, speculative decoding's accept coin, and
//! XTC's -- has to come off this one stream in this one order.

use super::penalties::apply_history_penalties;
use super::{argmax, filtered_distribution, greedy_choice, SamplingParams};
use crate::penalty_window::PenaltyWindow;

/// Sets logits a caller wants to forbid to `-inf`, in place, before the
/// sampler looks at them.
///
/// Two callers today, and they COMPOSE rather than exclude each other --
/// a masked logit stays masked, so the order they run in cannot matter:
/// JSON-object mode's character-class filter
/// (`frink_server::json_mode`), and grammar-constrained decoding
/// ([`crate::grammar_sampler::GrammarSampler::mask_logits`]).
///
/// The signature returns nothing because the callback runs from inside
/// the sampler, which has no error to return one through. A mask that
/// CAN fail -- a grammar that dead-ends leaves every logit at `-inf`,
/// and sampling from that is how an "impossible" request becomes
/// arbitrary text with a 200 -- records its refusal in the closure's own
/// captured state, and the decode loop reads it after the sample and
/// throws the token away. `frink_server::sample_step::sample_next` is
/// the one place that pairing lives.
pub type LogitMask<'a> = &'a mut dyn FnMut(&mut [f32]);

/// A small, seedable xorshift64* generator. Not cryptographically
/// secure -- sampling doesn't need that -- but reproducible given a
/// seed, which greedy argmax already was for free.
pub struct Sampler {
    state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        // xorshift64* requires a nonzero seed.
        Sampler {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        // The `*` in xorshift64*. Without it this is plain xorshift64,
        // whose state IS its output, and a small seed's first output is
        // therefore still small: for every seed below ~4000 the first
        // draw landed in the bottom eighth of [0, 1), so a request that
        // asked for `seed: 42` always got its first token from the
        // bottom of the CDF. The multiply is what decorrelates the
        // output from a low-entropy state; see
        // `low_seeds_do_not_bias_the_first_draw`.
        self.state.wrapping_mul(0x2545F491_4F6CDD1D)
    }

    /// Uniform float in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// The one uniform draw XTC needs for this token, or `None` when XTC
    /// cannot fire for these parameters.
    ///
    /// XTC is the only sampler in the chain that is itself stochastic
    /// (`llama_sample_xtc_apply` draws from its own `std::mt19937`,
    /// `src/llama-sampler.cpp:2146`), which is why the chain below takes
    /// the roll as an argument instead of owning an RNG: the chain is
    /// also what `sampling_distribution` runs for speculative
    /// verification, and a filter that drew its own randomness there
    /// would make "the distribution the sampler draws from" a different
    /// distribution every time it was asked.
    ///
    /// **The draw is skipped when XTC cannot fire**, and that is not an
    /// optimisation. Every draw advances the seeded stream, so drawing
    /// unconditionally would shift every subsequent token of every
    /// existing seeded generation, on every run that never asked for
    /// XTC. [`SamplingParams::xtc_can_fire`] is the single predicate
    /// this and [`Candidates::xtc`] share.
    pub fn xtc_roll(&mut self, params: &SamplingParams) -> Option<f32> {
        if params.xtc_can_fire() {
            Some(self.next_f32())
        } else {
            None
        }
    }

    /// Samples one token id from `logits`, given `params` and the
    /// [`PenaltyWindow`] the penalties look back over. Falls back to
    /// plain greedy argmax when `params.temperature <= 0.0`.
    ///
    /// `history` is a window and not a slice on purpose: it carries the
    /// PROMPT as well as the generated tokens, which is what llama.cpp
    /// penalises over. See [`crate::penalty_window`].
    ///
    /// A length-1 `logits` vector is treated as a precomputed greedy token
    /// id (`logits[0] as usize`) — used by the Metal dense-stack path that
    /// returns GPU argmax instead of downloading the full vocab.
    pub fn sample(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        history: PenaltyWindow<'_>,
    ) -> usize {
        self.sample_with_mask(logits, params, history, None)
    }

    /// Like [`Self::sample`], but optionally zeroes disallowed logits via
    /// `mask` before argmax / nucleus sampling (used for JSON-object mode).
    pub fn sample_with_mask(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        history: PenaltyWindow<'_>,
        mask: Option<LogitMask<'_>>,
    ) -> usize {
        self.sample_inner(logits, params, history, mask, false).0
    }

    /// The token AND the distribution it was drawn from, normalised to
    /// sum to 1 over the whole vocabulary.
    ///
    /// This is what `logprobs` has to report: not the raw logits, but
    /// the distribution the sampler actually drew from, with the
    /// penalties applied over the `penalty_last_n` window and
    /// llama.cpp's chain run in `params.sampler_order`. A filtered-out
    /// candidate is a zero, which is what "this token could not have
    /// been chosen" means.
    ///
    /// `None` for a vocabulary this sampler never saw: a backend that
    /// folded `lm_head` and `argmax` onto the device hands back a
    /// one-element vector holding the chosen id, and there is no
    /// distribution to report for it. A caller that needs one must ask
    /// for the vocabulary (`GenerationParams::needs_vocab_logits`)
    /// rather than be given a fabricated single-candidate answer.
    ///
    /// It is the SAME vector [`Self::sample_with_mask`] draws from --
    /// both go through `sample_inner` -- so a reported logprob cannot
    /// describe a distribution other than the one that was sampled.
    ///
    /// **Not `sampling_distribution`**, and the difference is the
    /// point. That function recomputes the pipeline from the logits
    /// WITHOUT drawing, which is right for speculative verification
    /// (it needs `p_target(x)` for a token someone else proposed) and
    /// wrong here for two reasons: it takes `xtc_roll` as an argument,
    /// so a caller who passed a fresh roll would report a distribution
    /// the draw never saw; and for a greedy request it returns a
    /// ONE-HOT, which as a logprob would claim the model was certain
    /// when nobody asked it. This reports the real distribution in
    /// both cases, because "how confident was the model" is a question
    /// a greedy caller is entitled to ask.
    pub fn sample_reporting(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        history: PenaltyWindow<'_>,
        mask: Option<LogitMask<'_>>,
    ) -> (usize, Option<Vec<f32>>) {
        self.sample_inner(logits, params, history, mask, true)
    }

    /// One pipeline, parameterised by whether the caller wants the
    /// distribution back.
    ///
    /// `want_probs` costs the greedy fast path: with it set, even a
    /// chain that keeps the argmax builds the full distribution,
    /// because there is nothing to report otherwise. Unset, every path
    /// is exactly what it was.
    fn sample_inner(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        history: PenaltyWindow<'_>,
        mut mask: Option<LogitMask<'_>>,
        want_probs: bool,
    ) -> (usize, Option<Vec<f32>>) {
        let xtc_roll = self.xtc_roll(params);
        // A device-folded argmax: one element holding the chosen id,
        // no vocabulary behind it.
        //
        // Gated on GREEDY, and that gate is load-bearing rather than
        // incidental: only the greedy device fold produces this shape,
        // and a SAMPLED request with a one-token vocabulary is a real
        // distribution whose only candidate is token 0. Hoisting this
        // check above the temperature test made `sample(&[42.0])` at
        // temperature 0.8 answer 42 instead of 0, which
        // `temperature_zero_accepts_precomputed_argmax_singleton`
        // catches.
        if params.temperature <= 0.0 && mask.is_none() && logits.len() == 1 {
            return (logits[0] as usize, None);
        }

        let mut scores: Vec<f32> = logits.to_vec();
        apply_history_penalties(&mut scores, params, history);

        if let Some(m) = mask.as_mut() {
            m(&mut scores);
        }

        if params.temperature <= 0.0 {
            if scores.len() == 1 {
                return (scores[0] as usize, None);
            }
            if !want_probs {
                return (greedy_choice(scores, params, history, xtc_roll), None);
            }
            // The greedy answer read off the distribution it is the
            // argmax OF, so the reported probabilities and the chosen
            // token cannot disagree.
            let probs = filtered_distribution(scores, params, history, xtc_roll);
            return (argmax(&probs), Some(probs));
        }

        let probs = filtered_distribution(scores, params, history, xtc_roll);
        let chosen = self.sample_from(&probs);
        (chosen, want_probs.then_some(probs))
    }

    /// A uniform draw in `[0.0, 1.0)`.
    ///
    /// Exposed because speculative decoding's accept test is a coin
    /// flip against `p_target(x) / p_draft(x)` rather than a draw from
    /// a distribution, and it must come off the same seeded stream as
    /// every other draw in the run or a "reproducible given a seed"
    /// generation stops being reproducible.
    pub fn uniform(&mut self) -> f32 {
        self.next_f32()
    }

    /// Draws one index from an already-normalised distribution.
    ///
    /// Split out of [`Self::sample_with_mask`] so speculative decoding
    /// can sample from a distribution it had to compute anyway (the
    /// rejection rule needs `p_target` itself, not just a draw from it)
    /// and still go through *exactly* the same draw as ordinary
    /// sampling. Two separate copies of this loop would be two chances
    /// to be subtly non-lossless.
    pub fn sample_from(&mut self, probs: &[f32]) -> usize {
        let draw = self.next_f32();
        let mut cumulative = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if draw < cumulative {
                return i;
            }
        }
        // Floating-point rounding may leave `draw` fractionally above
        // the final cumulative sum; the last nonzero-probability token
        // is the correct fallback, not index 0.
        probs
            .iter()
            .enumerate()
            .rev()
            .find(|&(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{sampling_distribution, spread_logits};

    ///
    /// This is the reason [`Sampler::xtc_roll`] is conditional rather
    /// than unconditional. Delete the `xtc_can_fire` guard there and
    /// this goes red on the first token.
    #[test]
    fn a_chain_without_xtc_does_not_consume_a_draw_for_it() {
        let params = SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let logits = spread_logits(32);
        assert!(
            Sampler::new(99).xtc_roll(&params).is_none(),
            "the guard must refuse the draw, not merely ignore it"
        );

        // ONE draw per token, taken by hand off a generator that never
        // heard of XTC. An unconditional roll makes `sample` consume
        // two values per token, so the very first token comes off the
        // SECOND draw and this diverges immediately.
        let mut sampled_by_chain = Sampler::new(99);
        let mut by_hand = Sampler::new(99);
        for step in 0..16 {
            let sampled = sampled_by_chain.sample(&logits, &params, PenaltyWindow::new(&[], &[]));
            let probs = sampling_distribution(&logits, &params, PenaltyWindow::new(&[], &[]), None);
            assert_eq!(
                sampled,
                by_hand.sample_from(&probs),
                "token {step} came off a different position in the stream"
            );
        }
        // And the two generators are still in lockstep afterwards.
        assert_eq!(sampled_by_chain.uniform(), by_hand.uniform());
    }

    /// **The flag does something.** A chain that runs the temperature
    /// before top-p keeps a different candidate set than the default,
    /// which is the whole reason the order is worth exposing -- and the
    /// reason getting it wrong is a silent quality regression rather
    /// than an error.
    ///
    /// A hot temperature flattens the distribution, so a top-p applied
    /// after it sums smaller probabilities and reaches `p` later,
    /// keeping MORE candidates.
    ///
    /// The reported distribution must be the one that was DRAWN from,
    /// not a second opinion computed beside it. Both go through
    /// `sample_inner`, and this pins the consequence: the same seed
    /// gives the same token whether or not the caller asked to see the
    /// probabilities, and the token always has nonzero probability in
    /// what is reported.
    #[test]
    fn the_reported_distribution_is_the_one_that_was_sampled() {
        let logits: Vec<f32> = (0..64).map(|i| ((i * 7) % 13) as f32 * 0.4).collect();
        let params = SamplingParams {
            temperature: 0.9,
            top_p: 0.95,
            ..SamplingParams::default()
        };

        let quiet = Sampler::new(7).sample(&logits, &params, PenaltyWindow::new(&[], &[]));
        let (loud, probs) =
            Sampler::new(7).sample_reporting(&logits, &params, PenaltyWindow::new(&[], &[]), None);
        assert_eq!(
            quiet, loud,
            "asking for the probabilities changed which token was drawn"
        );

        let probs = probs.expect("a real vocabulary reports a distribution");
        assert_eq!(probs.len(), logits.len(), "one entry per vocabulary slot");
        let total: f32 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "must be normalised, got {total}"
        );
        assert!(
            probs[loud] > 0.0,
            "the chosen token has zero probability in the distribution it came from"
        );
        // A filtered-out candidate is a zero, which is what "could not
        // have been chosen" means, so top-p really did remove some.
        assert!(
            probs.contains(&0.0),
            "top_p 0.95 kept every candidate, so this proves nothing"
        );
    }

    /// Greedy reports too, and the token it reports is the argmax OF
    /// the reported distribution -- read off the same vector rather
    /// than decided separately, so the two cannot disagree.
    #[test]
    fn greedy_reports_the_distribution_its_answer_is_the_argmax_of() {
        let logits = vec![0.1f32, 3.0, 0.2, 2.9];
        let params = SamplingParams::default();
        assert!(params.temperature <= 0.0, "default is greedy");

        let (chosen, probs) =
            Sampler::new(1).sample_reporting(&logits, &params, PenaltyWindow::new(&[], &[]), None);
        let probs = probs.expect("a real vocabulary reports a distribution");
        assert_eq!(chosen, 1, "the largest logit wins");
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(chosen, best, "the answer is not the argmax of the report");
        // And it agrees with the plain entry point.
        assert_eq!(
            Sampler::new(1).sample(&logits, &params, PenaltyWindow::new(&[], &[])),
            chosen
        );
    }

    /// A device-folded argmax has no vocabulary behind it, so there is
    /// nothing to report. `None` rather than a fabricated
    /// single-candidate distribution, which would read as "the model
    /// was certain" when nobody asked the model.
    #[test]
    fn a_device_folded_argmax_reports_no_distribution() {
        let params = SamplingParams::default();
        let (chosen, probs) =
            Sampler::new(1).sample_reporting(&[42.0], &params, PenaltyWindow::new(&[], &[]), None);
        assert_eq!(chosen, 42, "the singleton is the chosen id");
        assert!(
            probs.is_none(),
            "a folded argmax must not fabricate a distribution"
        );
    }

    #[test]
    fn temperature_zero_accepts_precomputed_argmax_singleton() {
        let mut sampler = Sampler::new(1);
        let params = SamplingParams::default();
        assert_eq!(
            sampler.sample(&[42.0], &params, PenaltyWindow::new(&[], &[])),
            42
        );
        // Non-greedy must not treat a singleton as a token id.
        let sampled = SamplingParams {
            temperature: 0.8,
            ..SamplingParams::default()
        };
        // Softmax of a single logit → only token 0 is eligible.
        assert_eq!(
            sampler.sample(&[42.0], &sampled, PenaltyWindow::new(&[], &[])),
            0
        );
    }

    #[test]
    fn temperature_zero_is_deterministic_greedy_argmax() {
        let logits = vec![0.1, 0.9, 0.3, -0.2];
        let params = SamplingParams::default();
        let mut sampler = Sampler::new(42);
        assert_eq!(
            sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[])),
            1
        );
        // Must be deterministic regardless of RNG state advancing.
        assert_eq!(
            sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[])),
            1
        );
    }

    #[test]
    fn high_temperature_can_pick_a_non_argmax_token_over_many_draws() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(7);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[])));
        }
        assert!(
            seen.len() > 1,
            "uniform logits at temperature=1.0 must produce more than one distinct token across 200 draws"
        );
    }

    #[test]
    fn top_k_one_is_equivalent_to_greedy() {
        let logits = vec![0.1, 0.9, 0.3, -0.2];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(123);
        for _ in 0..20 {
            assert_eq!(
                sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[])),
                1
            );
        }
    }

    #[test]
    fn top_p_near_zero_is_equivalent_to_greedy() {
        let logits = vec![0.1, 5.0, 0.3, -0.2];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.001,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(9);
        for _ in 0..20 {
            assert_eq!(
                sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[])),
                1
            );
        }
    }

    #[test]
    fn presence_and_frequency_penalties_reduce_seen_token_logits() {
        let logits = vec![0.0, 5.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            presence_penalty: 10.0,
            frequency_penalty: 0.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(1);
        let mut counts = [0usize; 3];
        for _ in 0..500 {
            counts[sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[1]))] += 1;
        }
        assert!(
            counts[1] < 250,
            "presence_penalty should discourage token 1; counts={counts:?}"
        );

        let params = SamplingParams {
            temperature: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 10.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(2);
        counts = [0; 3];
        for _ in 0..500 {
            counts[sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[1, 1, 1]))] += 1;
        }
        assert!(
            counts[1] < 250,
            "frequency_penalty should discourage repeated token 1; counts={counts:?}"
        );
    }

    #[test]
    fn repetition_penalty_reduces_probability_of_recently_seen_token() {
        let logits = vec![0.0, 5.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            repetition_penalty: 1000.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(3);
        let mut counts = [0usize; 3];
        for _ in 0..500 {
            counts[sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[1]))] += 1;
        }
        assert!(
            counts[1] < 250,
            "heavily penalizing token 1 (already in history) should make it far less likely than its raw logit alone would suggest; got counts={counts:?}"
        );
    }

    #[test]
    fn low_seeds_do_not_bias_the_first_draw() {
        // Every generation seeds a fresh `Sampler` (the server does it
        // per request, from the caller's `seed`), so the FIRST draw off
        // a freshly seeded generator is the one users actually see.
        // Plain xorshift64 returns its own state, so seeds 1..4000 all
        // produced a first draw in the bottom eighth of [0, 1) -- the
        // first sampled token of every seeded request came off the
        // bottom of the CDF.
        let vocab = 8;
        let logits = vec![0.0f32; vocab];
        let params = SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let seeds = 4_000u64;
        let mut counts = vec![0usize; vocab];
        for seed in 1..=seeds {
            counts[Sampler::new(seed).sample(&logits, &params, PenaltyWindow::new(&[], &[]))] += 1;
        }
        let expected = seeds as f64 / vocab as f64;
        for (token, &c) in counts.iter().enumerate() {
            assert!(
                (c as f64 - expected).abs() < expected * 0.25,
                "uniform logits: token {token} came up {c} times across {seeds} seeds, \
                 expected about {expected:.0} (counts={counts:?})"
            );
        }
    }

    #[test]
    fn the_published_distribution_is_the_one_sample_actually_draws_from() {
        // `sampling_distribution` is load-bearing for lossless
        // speculative verification: if it disagreed with what `sample`
        // draws from, every accept/reject decision would be measured
        // against the wrong target. Check them against each other
        // empirically, with filters on so the two code paths have
        // something to disagree about.
        let logits = vec![0.4, 2.0, -1.0, 1.2, 0.9, -0.3];
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 4,
            repetition_penalty: 1.3,
            ..SamplingParams::default()
        };
        let history = [1usize, 4];
        let claimed =
            sampling_distribution(&logits, &params, PenaltyWindow::new(&[], &history), None);
        assert!((claimed.iter().sum::<f32>() - 1.0).abs() < 1e-5);

        let draws = 100_000;
        let mut counts = vec![0usize; logits.len()];
        let mut sampler = Sampler::new(0xC0FFEE);
        for _ in 0..draws {
            counts[sampler.sample(&logits, &params, PenaltyWindow::new(&[], &history))] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            let empirical = c as f64 / draws as f64;
            assert!(
                (empirical - claimed[i] as f64).abs() < 0.01,
                "token {i}: sample() draws it {empirical:.4} of the time but \
                 sampling_distribution claims {:.4}",
                claimed[i]
            );
        }
    }

    #[test]
    fn degenerate_all_zero_probability_falls_back_to_greedy() {
        // top_k=1 combined with a top_p that would exclude even that
        // one surviving token is a contradictory/degenerate
        // configuration; must not panic or sample index 0 blindly.
        let logits = vec![0.1, 0.9, 0.3, -0.2];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(1);
        assert_eq!(
            sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[])),
            1
        );
    }
}
