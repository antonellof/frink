//! `allowed_token_ids` and `bad_words`: two fields, one mask.
//!
//! Both steer the draw rather than ending it, which is what separates
//! them from `stop`: a stop string ENDS a generation once it has been
//! produced, and these two make sure it never is.
//!
//! They are one type because they are one operation at one seam -- the
//! mask closure `crate::sample_step` already hands the sampler, beside
//! the grammar, JSON mode and the reasoning budget. Order does not
//! matter and must not: no mask here ever clears a `-f32::INFINITY`,
//! so the result is the intersection whichever runs first.
//!
//! # `bad_words` is not a string filter
//!
//! A bad word is TOKENIZED, and what is forbidden is its LAST token,
//! and only when the tokens before it are exactly what has just been
//! generated. That is upstream's rule (`NoBadWordsLogitsProcessor`),
//! and the alternative -- masking every token of the word
//! unconditionally -- would forbid every word that merely starts the
//! same way. A one-token bad word has an empty prefix, so it is
//! forbidden at every position, which is the common case and the one
//! that reads as "this string never appears".
//!
//! The rule is about TOKENS, so it is exact only for the tokenization
//! the model would have produced. A bad word the model spells across a
//! different token boundary can still come out, and no
//! logit-processor implementation of this field avoids that; saying so
//! is better than implying a guarantee the mechanism cannot give.

/// Forbidden and permitted token sets for one request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub(crate) struct TokenMask {
    /// Sampling is restricted to these ids. Sorted and deduplicated at
    /// construction so the membership test is a binary search rather
    /// than a scan of a caller-supplied list per token.
    ///
    /// `None` is "no restriction". An EMPTY set is a caller asking for
    /// a draw from nothing, which is refused where it is parsed rather
    /// than turned into a generation that cannot produce a token.
    allowed: Option<Vec<usize>>,
    /// Tokenized bad words. Each is non-empty; the last element is
    /// what gets masked and the ones before it are the prefix that
    /// must match the tail of what has been generated.
    bad_words: Vec<Vec<usize>>,
    /// Bad words as the caller SPELLED them, before a tokenizer has
    /// seen them.
    ///
    /// The same two-stage shape `stop` / `stop_token_ids` already has,
    /// and for the same reason: the route that parses the request has
    /// no tokenizer, and the layer that has one is the last place that
    /// also has the request. [`Self::resolve`] is the move between
    /// them, and [`Self::unresolved`] is how the sampler can tell a
    /// mask that was never resolved from one that had nothing to
    /// resolve -- the distinction `ReasoningBudget::Requested` already
    /// draws, because running on would serve an unfiltered answer as a
    /// filtered one.
    pending: Vec<String>,
}

impl TokenMask {
    pub(crate) fn new(allowed: Option<Vec<usize>>, bad_words: Vec<Vec<usize>>) -> Self {
        let allowed = allowed.map(|mut ids| {
            ids.sort_unstable();
            ids.dedup();
            ids
        });
        TokenMask {
            allowed,
            bad_words: bad_words.into_iter().filter(|w| !w.is_empty()).collect(),
            pending: Vec::new(),
        }
    }

    /// A mask whose bad words are still strings.
    pub(crate) fn requested(allowed: Option<Vec<usize>>, words: Vec<String>) -> Self {
        TokenMask {
            pending: words.into_iter().filter(|w| !w.is_empty()).collect(),
            ..TokenMask::new(allowed, Vec::new())
        }
    }

    /// Turns the caller's strings into token sequences.
    ///
    /// A word that encodes to nothing is dropped: there is no token to
    /// forbid, and keeping an empty sequence would forbid every draw
    /// (an empty prefix matches, and there is no last token to take).
    pub(crate) fn resolve(&mut self, encode: impl Fn(&str) -> Vec<usize>) {
        for word in self.pending.drain(..) {
            let ids = encode(&word);
            if !ids.is_empty() {
                self.bad_words.push(ids);
            }
        }
    }

    /// Bad words that reached here without a tokenizer having seen
    /// them. A generation must STOP on this rather than answer
    /// unfiltered.
    pub(crate) fn unresolved(&self) -> &[String] {
        &self.pending
    }

    /// Whether this request has anything to mask.
    ///
    /// Read by `GenerationParams::needs_vocab_logits`: a backend that
    /// folded `lm_head + argmax` onto the device returns a token id
    /// rather than a vocabulary, and there would be nothing left to
    /// mask by the time it got here.
    pub(crate) fn is_empty(&self) -> bool {
        self.allowed.is_none() && self.bad_words.is_empty() && self.pending.is_empty()
    }

    /// Applies both rules to one row of logits.
    ///
    /// `history` is what this completion has generated so far, which
    /// only `bad_words` reads: a multi-token word is forbidden at its
    /// last token and only when the tokens before it are what just
    /// came out.
    pub(crate) fn mask(&self, scores: &mut [f32], history: &[usize]) {
        if let Some(allowed) = &self.allowed {
            for (id, score) in scores.iter_mut().enumerate() {
                if allowed.binary_search(&id).is_err() {
                    *score = f32::NEG_INFINITY;
                }
            }
        }
        for word in &self.bad_words {
            let (prefix, last) = word.split_at(word.len() - 1);
            if !history.ends_with(prefix) {
                continue;
            }
            if let Some(score) = scores.get_mut(last[0]) {
                *score = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_ids_are_the_only_ones_left_finite() {
        let mask = TokenMask::new(Some(vec![3, 1]), Vec::new());
        let mut scores = vec![0.0f32; 5];
        mask.mask(&mut scores, &[]);
        assert_eq!(
            scores,
            vec![
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY
            ]
        );
    }

    /// **A one-token bad word is forbidden at every position.**
    ///
    /// The common case, and the one that reads as "this string never
    /// appears". Its prefix is empty, and every history ends with an
    /// empty slice.
    #[test]
    fn a_single_token_bad_word_is_always_masked() {
        let mask = TokenMask::new(None, vec![vec![2]]);
        for history in [vec![], vec![9usize], vec![2, 7]] {
            let mut scores = vec![0.0f32; 4];
            mask.mask(&mut scores, &history);
            assert_eq!(scores[2], f32::NEG_INFINITY, "history {history:?}");
            assert_eq!(scores[0], 0.0, "history {history:?}: masked too much");
        }
    }

    /// **A multi-token bad word is forbidden only after its prefix.**
    ///
    /// The rule this field actually has. Masking every token of the
    /// word instead would forbid every word that merely starts the
    /// same way, which is a different and much larger promise.
    #[test]
    fn a_multi_token_bad_word_waits_for_its_prefix() {
        let mask = TokenMask::new(None, vec![vec![1, 2, 3]]);

        let mut scores = vec![0.0f32; 4];
        mask.mask(&mut scores, &[9, 9]);
        assert_eq!(scores[3], 0.0, "masked without the prefix having been seen");

        let mut scores = vec![0.0f32; 4];
        mask.mask(&mut scores, &[9, 1, 2]);
        assert_eq!(scores[3], f32::NEG_INFINITY, "the prefix matched");
        // And only the last token: the first two are ordinary tokens
        // that happen to start a forbidden word.
        assert_eq!(scores[1], 0.0);
        assert_eq!(scores[2], 0.0);
    }

    /// The two rules intersect, and neither clears the other's
    /// `-inf` -- which is what lets them run in either order beside
    /// the grammar and the reasoning budget.
    #[test]
    fn the_two_rules_intersect() {
        let mask = TokenMask::new(Some(vec![1, 2]), vec![vec![2]]);
        let mut scores = vec![0.0f32; 4];
        mask.mask(&mut scores, &[]);
        assert_eq!(scores[0], f32::NEG_INFINITY, "not allowed");
        assert_eq!(scores[1], 0.0, "allowed and not forbidden");
        assert_eq!(scores[2], f32::NEG_INFINITY, "allowed but forbidden");
        assert_eq!(scores[3], f32::NEG_INFINITY, "not allowed");
    }

    #[test]
    fn an_empty_mask_changes_nothing() {
        let mask = TokenMask::default();
        assert!(mask.is_empty());
        let mut scores = vec![0.0f32, 1.0, 2.0];
        mask.mask(&mut scores, &[1]);
        assert_eq!(scores, vec![0.0, 1.0, 2.0]);
    }

    /// A bad word whose last id is past the vocabulary is ignored
    /// rather than panicking: the ids come from a tokenizer and the
    /// logits from a checkpoint, and a mismatch is a refusal's job,
    /// not an index's.
    #[test]
    fn an_out_of_range_bad_word_does_not_panic() {
        let mask = TokenMask::new(None, vec![vec![99]]);
        let mut scores = vec![0.0f32; 4];
        mask.mask(&mut scores, &[]);
        assert_eq!(scores, vec![0.0; 4]);
    }
}
