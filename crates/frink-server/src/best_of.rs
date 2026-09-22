//! `best_of`: generate `k` completions and return the `n` best.
//!
//! The scoring rule is the one thing this field needs that `n` does
//! not, and it is why the row waited: "best" has to mean something,
//! and the only defensible meaning is the one upstream uses -- the
//! SUMMED LOG-PROBABILITY of the generated tokens, under the
//! distribution each was actually drawn from.
//!
//! That became available when the sampler learned to publish the
//! distribution it drew from (`frink_models::Sampler::sample_reporting`).
//! Before it, there was nothing to rank by except length.
//!
//! # The sum is LENGTH-SENSITIVE, and that is upstream's rule
//!
//! Every log-probability is negative, so a longer completion has a
//! more negative sum: **the rule systematically favours short
//! answers.** That is not a defect being papered over, it is what
//! `best_of` means upstream, and a caller can reproduce the ranking
//! exactly from the `logprobs` this server will hand them.
//!
//! The alternative -- a per-token MEAN -- ranks differently and is
//! deliberately not used. The two disagree whenever a long, slightly
//! less certain completion meets a short, less certain one: one token
//! at `p = 0.5` sums to `-0.69` while ten at `p = 0.9` sum to `-1.05`,
//! so the sum picks the single token and the mean picks the ten.
//! Choosing the mean here would give a different answer from every
//! other engine for the same request, which is worse than a rule with
//! a known bias.
//!
//! Two completions with the same score keep their generation order, so
//! a tie is resolved by the seed rather than by a hash: `best_of` with
//! a fixed seed returns the same choice every time.
//!
//! # What it costs, stated plainly
//!
//! `best_of: k` decodes `k` completions and throws `k - n` of them
//! away. The prompt is still prefilled ONCE -- that is `n`'s fork
//! doing the work -- but the decode is `k` times the tokens, and the
//! caller is billed for all of them. `usage.completion_tokens` counts
//! every token generated, including the discarded ones, because that
//! is what the machine did.

use crate::generate::GeneratedChoice;

/// One completion's score: the sum of `ln p` over its generated
/// tokens.
///
/// An empty completion scores `0.0`, which is the identity for a sum
/// and ranks it above every non-empty one -- every real logprob is
/// negative. That is deliberate rather than an oversight: a
/// zero-length completion is the model declining to say anything,
/// and if one is produced alongside longer ones the caller asked a
/// question whose best answer really was silence. It cannot happen
/// from `max_tokens > 0` without a stop condition firing immediately.
pub(crate) fn score(choice: &GeneratedChoice) -> f64 {
    choice
        .logprobs
        .iter()
        .map(|(id, probs)| {
            let p = probs.get(*id).copied().unwrap_or(0.0) as f64;
            p.ln()
        })
        .sum()
}

/// The `n` best of `choices`, in descending score, re-indexed from 0.
///
/// Stable: equal scores keep their generation order, so a seeded
/// request returns the same choice on every run.
pub(crate) fn take_best(mut choices: Vec<GeneratedChoice>, n: usize) -> Vec<GeneratedChoice> {
    // `sort_by` is stable in std, which is the tie rule above.
    choices.sort_by(|a, b| score(b).total_cmp(&score(a)));
    choices.truncate(n);
    choices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::FinishReason;

    fn choice(text: &str, probs: &[f32]) -> GeneratedChoice {
        // One token per probability, each the chosen one, in a
        // two-token vocabulary so the untaken slot carries the rest.
        let logprobs = probs.iter().map(|p| (0usize, vec![*p, 1.0 - *p])).collect();
        GeneratedChoice {
            finish: FinishReason::Stop,
            text: text.to_string(),
            logprobs,
        }
    }

    #[test]
    fn the_score_is_the_sum_of_the_log_probabilities() {
        let c = choice("x", &[0.5, 0.25]);
        let want = 0.5f64.ln() + 0.25f64.ln();
        assert!((score(&c) - want).abs() < 1e-9, "{}", score(&c));
    }

    /// **The rule is the SUM, and the sum favours short answers.**
    ///
    /// Pinned with the case where a sum and a mean genuinely disagree,
    /// because an example where they agree proves nothing about which
    /// one is implemented. One token at `p = 0.5` sums to `-0.69`; ten
    /// at `p = 0.9` sum to `-1.05`. The sum picks the single token,
    /// the mean picks the ten.
    ///
    /// This test exists because the first version of this module's
    /// documentation claimed the opposite -- that a sum prefers the
    /// longer good answer -- and the arithmetic says otherwise.
    #[test]
    fn the_rule_is_the_sum_which_a_mean_would_rank_differently() {
        let short_and_unsure = choice("one", &[0.5]);
        let long_and_confident = choice("ten of them", &[0.9; 10]);

        let sum_short = 0.5f64.ln();
        let sum_long = 10.0 * 0.9f64.ln();
        assert!(sum_short > sum_long, "the premise: the sum picks short");
        assert!(sum_short < 0.9f64.ln(), "the premise: the mean picks long");

        let best = take_best(vec![long_and_confident, short_and_unsure], 1);
        assert_eq!(
            best[0].text, "one",
            "ranked by something other than the summed logprob"
        );
    }

    #[test]
    fn take_best_returns_n_in_descending_score() {
        let best = take_best(
            vec![
                choice("worst", &[0.1]),
                choice("best", &[0.9]),
                choice("middle", &[0.5]),
            ],
            2,
        );
        assert_eq!(
            best.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
            vec!["best", "middle"]
        );
    }

    /// Equal scores keep generation order, so a seeded `best_of`
    /// returns the same completion every run rather than whichever the
    /// sort happened to touch last.
    #[test]
    fn ties_keep_generation_order() {
        let best = take_best(
            vec![
                choice("first", &[0.5]),
                choice("second", &[0.5]),
                choice("third", &[0.5]),
            ],
            2,
        );
        assert_eq!(
            best.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
            vec!["first", "second"]
        );
    }

    /// Asking for more than were generated returns what there is
    /// rather than panicking on a truncate past the end.
    #[test]
    fn asking_for_more_than_exist_returns_them_all() {
        let best = take_best(vec![choice("only", &[0.5])], 4);
        assert_eq!(best.len(), 1);
    }
}
