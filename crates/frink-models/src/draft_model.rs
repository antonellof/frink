//! A second, smaller GGUF used as the drafter for speculative decoding.
//!
//! [`crate::speculative`] already had the half that is hard to get
//! right: the rejection rule, which makes speculation lossless at every
//! temperature rather than only at `--temp 0`. What it did not have was
//! a drafter worth running. The only implementation in the tree is
//! [`crate::speculative::PromptLookupSpeculator`], an n-gram match over
//! the history with no model at all. It is free, and it helps on
//! repetitive text, and it cannot carry a coding workload.
//!
//! # Why this is the item that moves the ceiling
//!
//! Decode reads every weight in the model to emit one token, so
//!
//! ```text
//! tokens/sec <= memory bandwidth / model bytes
//! ```
//!
//! is arithmetic, not engineering. A 17 GB checkpoint on a 960 GB/s
//! card cannot pass about 56 tok/s however good the kernels are. Better
//! kernels move an engine toward that number; they cannot move it past.
//!
//! A draft model changes what is read per token instead of how fast it
//! is read. A 2 GB drafter proposes `k` tokens, the target checks all
//! `k` in ONE pass over its 17 GB, good guesses are kept and bad ones
//! discarded, and the text is exactly what the target would have
//! written alone.
//!
//! # The two things this has to get right
//!
//! **The draft KV must roll back.** While proposing, the drafter
//! advances its own cache over tokens the target has not accepted and
//! may never accept. If those rows are left in place, the drafter's
//! context silently diverges from the target's. Nothing errors: the
//! accept rate just decays, which reads as "this drafter is bad" rather
//! than "this drafter is desynchronised". [`DraftModelSpeculator`]
//! therefore truncates to `synced` at the top of every `propose`, and
//! `synced` only ever counts tokens the caller's history actually
//! contains.
//!
//! This is the repo's dominant bug shape in its usual dress: two
//! structures that must agree about one thing, here the target's
//! history and the drafter's cache, with nothing enforcing it. What
//! enforces it is that `synced` is derived from the history passed in
//! on every call rather than remembered independently, so the drafter
//! cannot hold an opinion about the history that the history disagrees
//! with.
//!
//! **The vocabularies must match.** See [`VocabMismatch`].

use std::sync::Arc;

use crate::config::ModelConfig;
use crate::decoder::Decoder;
use crate::penalty_window::PenaltyWindow;
use crate::sampling::{sampling_distribution, Sampler, SamplingParams};
use crate::speculative::{DraftBlock, DraftDist, Drafter};
use frink_core::cache::KvCache;

/// The draft and target checkpoints do not agree about token ids.
///
/// This is the failure that costs a day, because it does not look like
/// a failure. The rejection rule compares the drafter's `q(x)` with the
/// target's `p(x)` at the same index `x`. If the two checkpoints number
/// their vocabularies differently, those are probabilities of different
/// tokens, the rule is comparing unrelated numbers, and the output is
/// no longer the target's distribution. What comes out is fluent text,
/// with no error and a plausible-looking accept rate.
///
/// So it is refused at construction, which per this repo's rule is
/// coverage rather than a defect: a partly-implemented thing must stop
/// and say what is missing instead of computing something else.
///
/// Vocabulary size is checked first because it is cheap and catches
/// most real mismatches (a 32000-token Llama drafter against a
/// 152064-token Qwen target). Equal sizes do not imply equal
/// vocabularies, though, so the caller that has both tokenizers should
/// also compare them; `vocab_size` is what a `Decoder` alone can see.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VocabMismatch {
    #[error(
        "draft and target checkpoints disagree about vocabulary size ({draft} vs {target}), so a \
         draft token id does not name the same token in both. Speculative decoding compares the \
         drafter's probability for a token id against the target's probability for that same id, \
         which would be comparing unrelated tokens: the result would not be the target's \
         distribution, and it would look exactly like text that is. Use a draft model from the \
         same family and tokenizer as the target"
    )]
    Size { draft: usize, target: usize },
}

/// A draft checkpoint, loaded once and shared by every drafter that
/// runs it: a server holds one and gives each request a
/// [`DraftModelSpeculator`] with KV of its own.
///
/// Built only by [`Self::new`] or [`load`], both of which turn the
/// decoder's KV-window policy off, so no drafter can be handed a
/// windowed cache it would have to roll back past (see
/// [`DraftModelSpeculator::new`]).
#[derive(Clone)]
pub struct DraftWeights(Arc<Decoder>);

impl DraftWeights {
    pub fn new(mut decoder: Decoder) -> Self {
        decoder.kv_window = crate::decoder::KvWindowPolicy::off();
        DraftWeights(Arc::new(decoder))
    }

    pub fn config(&self) -> &ModelConfig {
        &self.0.config
    }
}

/// Opens `path` as a drafter for `target`, refusing a recurrent draft:
/// the draft's own cache is rolled back after every verification block,
/// and llama.cpp gives a draft context no recurrent checkpoints either
/// (`n_rs_seq = 0`, `common/speculative.cpp:2726` at the pin). A
/// recurrent TARGET is served: its caches are checkpointed around each
/// verify batch (`frink_core::cache::KvCheckpoint`). The draft's KV
/// takes the target's dtype, one `--ctk` for the run. `frink run -md`
/// and `frink-server -md` both load through here.
pub fn load(path: &str, target: &ModelConfig) -> anyhow::Result<DraftWeights> {
    let mut config = ModelConfig::from_gguf(&frink_gguf::ShardedGguf::open(path)?)?;
    config.kv_dtype = target.kv_dtype;
    if config.has_recurrent_layers() {
        anyhow::bail!(
            "--model-draft cannot be a model with recurrent (Mamba) layers: the draft cache is \
             rolled back after every verification block"
        );
    }
    Ok(DraftWeights::new(Decoder::from_gguf(path, config)?))
}

/// A [`Drafter`] backed by a second [`Decoder`].
///
/// Owns its own KV caches, entirely separate from the target's. The two
/// models run over the same token sequence but have different layer
/// counts, head counts and head dimensions, so nothing about the two
/// caches is shared.
pub struct DraftModelSpeculator {
    decoder: Arc<Decoder>,
    kv_caches: Vec<KvCache>,
    /// How many tokens of the caller's history this drafter's KV holds.
    ///
    /// Never an independent record of "what I have seen": it is
    /// recomputed against the history handed to `propose`, so a
    /// rejected block cannot leave it overstating what is committed.
    synced: usize,
    sampling: SamplingParams,
    rng: Sampler,
    /// Positions whose draft probability fell below this stop the
    /// block. A drafter that is guessing is worse than no drafter: the
    /// target pays for the position either way, and a rejection also
    /// throws away every position after it.
    min_prob: f32,
    max_draft: usize,
}

impl DraftModelSpeculator {
    /// True when this drafter's KV really lives in the host caches it
    /// owns.
    ///
    /// A backend that keeps KV on the device leaves these at zero, and
    /// a drafter cannot roll back rows it cannot see. Callers check
    /// this after one warm-up rather than discovering it as a wrong
    /// accept rate.
    pub fn keeps_host_kv(&self) -> bool {
        // ROWS: the question is whether K/V actually landed in the
        // host buffer, which a device-resident backend leaves empty.
        self.kv_caches.first().is_some_and(|c| c.rows() > 0)
    }

    /// How many tokens of history this drafter's KV currently holds.
    /// Exposed so a caller can assert the drafter kept up.
    pub fn synced_len(&self) -> usize {
        self.synced
    }

    /// Fails when the two checkpoints cannot be compared token for
    /// token. See [`VocabMismatch`].
    /// `decoder` is taken by value and has its KV-window policy turned
    /// OFF (#61). [`Self::sync`] rolls the draft cache back to an
    /// arbitrary committed length -- potentially the whole history,
    /// after a long run of rejections -- and a windowed cache cannot
    /// represent a rollback past the rows it kept. The drafter is a
    /// small model whose KV is a small fraction of the target's, so
    /// this gives up almost none of the saving.
    pub fn new(
        decoder: Decoder,
        target_config: &ModelConfig,
        sampling: SamplingParams,
        seed: u64,
        max_draft: usize,
        min_prob: f32,
    ) -> Result<Self, VocabMismatch> {
        Self::from_weights(
            &DraftWeights::new(decoder),
            target_config,
            sampling,
            seed,
            max_draft,
            min_prob,
        )
    }

    /// As [`Self::new`], over weights loaded once and shared: this
    /// drafter's KV is its own, the weights are not copied.
    pub fn from_weights(
        weights: &DraftWeights,
        target_config: &ModelConfig,
        sampling: SamplingParams,
        seed: u64,
        max_draft: usize,
        min_prob: f32,
    ) -> Result<Self, VocabMismatch> {
        let decoder = Arc::clone(&weights.0);
        let draft_vocab = decoder.config.vocab_size;
        let target_vocab = target_config.vocab_size;
        if draft_vocab != target_vocab {
            return Err(VocabMismatch::Size {
                draft: draft_vocab,
                target: target_vocab,
            });
        }
        let kv_caches = decoder.config.new_kv_caches();
        Ok(DraftModelSpeculator {
            decoder,
            kv_caches,
            synced: 0,
            sampling,
            rng: Sampler::new(seed),
            min_prob,
            max_draft,
        })
    }

    /// Drops every cached row past `len`, on every layer.
    fn truncate_to(&mut self, len: usize) {
        for cache in &mut self.kv_caches {
            cache.truncate(len);
        }
    }

    /// Brings the draft cache up to `history` and returns the logits
    /// that follow its last token.
    ///
    /// Feeds only the tokens the cache does not already hold, which is
    /// what makes drafting cheap across a long conversation: the first
    /// call pays for the prompt and every later call pays for the
    /// handful of tokens the target committed since.
    ///
    /// The cache is rolled back to at most `history.len() - 1` rather
    /// than `history.len()`, and that off-by-one is load-bearing. The
    /// logits a block is drafted from are the ones that follow the last
    /// committed token, and they exist only as the return value of the
    /// forward pass that consumed it. Truncating to the full history
    /// would leave nothing to feed, so there would be no logits to
    /// draft from and every call after a rollback would propose
    /// nothing: speculation would quietly stop happening while
    /// remaining perfectly correct, which is the kind of failure that
    /// shows up as a benchmark result months later.
    ///
    /// The cost is re-feeding exactly one token per call. That is one
    /// step of the small model, against a block of them saved.
    fn sync(&mut self, history: &[usize]) -> Vec<f32> {
        debug_assert!(
            !history.is_empty(),
            "callers return early on an empty history"
        );
        // Everything past what the caller's history contains was
        // drafted and not accepted. It is not context, it is a guess
        // the target threw away.
        // Derived from the CACHE, not only from `self.synced`. The
        // cache is the authority on how many rows exist, and a backend
        // that keeps its KV somewhere other than this host `KvCache`
        // leaves it at zero however many tokens were fed. Trusting the
        // counter there truncated to 7 rows of a cache holding 0 and
        // panicked on a real Metal run. That is the same lesson as the
        // batched prefill: read the cursor, do not keep a copy of it.
        // ROWS, for the same reason: this bounds what can be kept by
        // what is really there, not by what the counter believes.
        let held = self.kv_caches.first().map_or(0, |c| c.rows());
        let keep = self.synced.min(history.len() - 1).min(held);
        self.truncate_to(keep);
        self.synced = keep;

        let mut logits = Vec::new();
        while self.synced < history.len() {
            let pos = self.synced;
            logits = self
                .decoder
                .forward_token(history[pos], pos, &mut self.kv_caches);
            self.synced += 1;
        }
        logits
    }
}

impl Drafter for DraftModelSpeculator {
    fn propose(&mut self, history: &[usize], _target_hidden: &[f32], max_len: usize) -> DraftBlock {
        let budget = max_len.min(self.max_draft);
        if budget == 0 || history.is_empty() {
            return DraftBlock::empty();
        }

        let mut logits = self.sync(history);

        let mut tokens = Vec::with_capacity(budget);
        let mut dists = Vec::with_capacity(budget);

        for _ in 0..budget {
            // `history` is everything the target has committed --
            // prompt included -- and `tokens` is what this block has
            // proposed on top of it, so the drafter's penalties see the
            // same sequence the target's would. The block used to clone
            // `history` per call to concatenate the two; the window
            // borrows both halves instead.
            // The XTC roll comes off THIS drafter's own stream, so the
            // distribution reported as `q` below is the one the token
            // was actually drawn from even when XTC is configured.
            let xtc_roll = self.rng.xtc_roll(&self.sampling);
            let probs = sampling_distribution(
                &logits,
                &self.sampling,
                PenaltyWindow::new(history, &tokens),
                xtc_roll,
            );
            let token = self.rng.sample_from(&probs);

            // `q` MUST be the distribution this token was actually
            // sampled from, truncation and all, or the rejection rule
            // is corrected against a lie. `sampling_distribution`
            // returns exactly that, so it is what gets reported.
            let dist = DraftDist::from_dense(&probs);
            let q = dist.prob(token);
            if q < self.min_prob {
                // Stop before committing this token, so the cache is
                // not advanced over a position nobody drafted.
                break;
            }

            tokens.push(token);
            dists.push(dist);

            let pos = self.synced;
            logits = self.decoder.forward_token(token, pos, &mut self.kv_caches);
            self.synced += 1;
        }

        DraftBlock::new(tokens, dists)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_dense_fixture;

    fn drafter(vocab: usize, max_draft: usize, min_prob: f32) -> DraftModelSpeculator {
        let target = {
            let mut c = test_dense_fixture();
            c.vocab_size = vocab;
            c
        };
        let decoder = Decoder::new_random_small(test_dense_fixture(), 2, vocab);
        DraftModelSpeculator::new(
            decoder,
            &target,
            SamplingParams::default(),
            7,
            max_draft,
            min_prob,
        )
        .expect("matching vocabularies")
    }

    /// A draft model whose vocabulary differs from the target's is
    /// refused at construction, not accepted and corrected later.
    ///
    /// There is nothing to correct. The rejection rule compares the
    /// drafter's probability for token id `x` against the target's
    /// probability for token id `x`; if the two checkpoints number
    /// their vocabularies differently those are different tokens, and
    /// the output is no longer the target's distribution while looking
    /// exactly like text that is. Fluent, plausible accept rate, no
    /// error. That is the one failure this engine refuses to serve.
    #[test]
    fn a_draft_model_with_a_different_vocabulary_is_refused_by_name() {
        let mut target = test_dense_fixture();
        target.vocab_size = 64;
        let decoder = Decoder::new_random_small(test_dense_fixture(), 2, 32);

        let err = DraftModelSpeculator::new(decoder, &target, SamplingParams::default(), 0, 4, 0.0)
            .err()
            .expect("32 != 64");

        assert_eq!(
            err,
            VocabMismatch::Size {
                draft: 32,
                target: 64
            }
        );
        let msg = err.to_string();
        // The message has to say both numbers and why it matters, or
        // the next person reads it as an arbitrary compatibility rule
        // and looks for a flag to turn it off.
        assert!(msg.contains("32") && msg.contains("64"), "{msg}");
        assert!(msg.contains("same family and tokenizer"), "{msg}");
    }

    /// The drafter proposes a block and reports one distribution per
    /// token, which is what the rejection rule needs to run at all.
    #[test]
    fn a_block_carries_one_honest_distribution_per_drafted_token() {
        let mut d = drafter(32, 4, 0.0);
        let block = d.propose(&[1, 2, 3], &[], 4);

        assert_eq!(block.len(), 4, "the whole budget was drafted");
        assert_eq!(block.tokens().len(), block.dists().len());
        for (token, dist) in block.tokens().iter().zip(block.dists()) {
            // `q(x)` for the token actually sampled must be nonzero:
            // the rule divides by it.
            assert!(
                dist.prob(*token) > 0.0,
                "a drafter must report the distribution it sampled from"
            );
        }
    }

    /// **The rollback.** After a block is proposed, the drafter's cache
    /// holds rows for tokens the target has not accepted. The next call
    /// arrives with a history that does not contain them, and those
    /// rows must be gone before anything else is fed.
    ///
    /// Left in place, the drafter's context silently diverges from the
    /// target's: every later proposal is conditioned on tokens that
    /// were thrown away. Nothing errors. The accept rate decays, which
    /// reads as "this drafter is bad" rather than "this drafter is
    /// desynchronised", and that is why this is asserted on the cache
    /// length rather than on output quality.
    #[test]
    fn the_draft_cache_rolls_back_the_positions_the_target_did_not_accept() {
        let mut d = drafter(32, 4, 0.0);

        let block = d.propose(&[1, 2, 3], &[], 4);
        assert_eq!(block.len(), 4);
        assert_eq!(
            d.synced, 7,
            "3 of history plus 4 drafted are in the cache after proposing"
        );

        // The target accepted exactly one of them, so the caller's
        // history grew by one, not by four.
        d.propose(&[1, 2, 3, block.tokens()[0]], &[], 4);

        assert_eq!(
            d.kv_caches[0].positions(),
            d.synced,
            "every layer's cache agrees with the drafter's own count"
        );
        assert_eq!(
            d.synced, 8,
            "4 committed tokens plus 4 freshly drafted, NOT 7 stale rows plus more"
        );
    }

    /// A history shorter than what the cache holds is a rollback too,
    /// and the arithmetic must not underflow into a huge truncate.
    #[test]
    fn a_history_shorter_than_the_cache_truncates_rather_than_underflowing() {
        let mut d = drafter(32, 4, 0.0);
        d.propose(&[1, 2, 3, 4, 5], &[], 4);
        assert_eq!(d.synced, 9);

        d.propose(&[1, 2], &[], 1);
        assert_eq!(d.synced, 3, "2 of history plus 1 drafted");
        assert_eq!(d.kv_caches[0].positions(), 3);
    }

    /// Drafting stops when the drafter's own probability for the token
    /// it just sampled falls below the floor.
    ///
    /// A guessing drafter is worse than none: the target pays for the
    /// position either way, and a rejection also discards every
    /// position after it. With the floor above 1.0 nothing can clear
    /// it, so the block is empty and the caller falls back to one
    /// ordinary decode step.
    #[test]
    fn a_drafter_below_the_probability_floor_proposes_nothing() {
        let mut d = drafter(32, 4, 1.01);
        let block = d.propose(&[1, 2, 3], &[], 4);
        assert!(block.is_empty(), "nothing clears a floor above 1.0");
        assert_eq!(
            d.synced, 3,
            "and the cache holds the history only, no abandoned draft rows"
        );
    }

    /// `max_draft` is a ceiling the caller's budget cannot raise.
    #[test]
    fn the_configured_maximum_bounds_the_callers_budget() {
        let mut d = drafter(32, 2, 0.0);
        assert_eq!(d.propose(&[1, 2, 3], &[], 8).len(), 2);
    }

    /// An empty history has nothing to condition on, and a zero budget
    /// asked for nothing. Both propose nothing rather than panicking.
    #[test]
    fn an_empty_history_or_a_zero_budget_proposes_nothing() {
        let mut d = drafter(32, 4, 0.0);
        assert!(d.propose(&[], &[], 4).is_empty());
        assert!(d.propose(&[1, 2], &[], 0).is_empty());
    }

    /// **The property the whole feature exists to preserve.**
    ///
    /// A draft model is only worth having if the text is exactly what
    /// the target would have written alone. At temperature 0 that is
    /// checkable exactly: token for token against a plain
    /// `forward_token` loop over an identically seeded target.
    ///
    /// This is the test that catches a desynchronised draft cache, a
    /// dishonest `q`, or an off-by-one in the block, because all three
    /// change the output rather than announcing themselves. A drafter
    /// is allowed to be bad; it is not allowed to be consulted in a way
    /// that changes the answer.
    ///
    /// The drafter here is a genuinely different model from the target
    /// (a different random seed, half the layers), so it is wrong
    /// often, which is exactly the case where rejection and rollback
    /// have to work.
    #[test]
    fn a_draft_model_does_not_change_what_the_target_writes() {
        use crate::speculative::speculative_decode;

        let cfg = test_dense_fixture();
        let vocab = 32;
        let prompt = vec![1usize, 2, 3, 4, 1, 2];
        let max_new = 8;

        let target = Decoder::new_random_small(cfg.clone(), 4, vocab);
        let mut caches: Vec<KvCache> = target.config.new_kv_caches();

        // A different model, not a copy of the target: two layers
        // rather than four, so it disagrees constantly.
        let draft = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut drafter =
            DraftModelSpeculator::new(draft, &target.config, SamplingParams::default(), 11, 4, 0.0)
                .expect("matching vocabularies");

        let result = speculative_decode(&target, &prompt, max_new, &mut caches, &mut drafter);

        // The same target, decoded the ordinary way.
        let plain = Decoder::new_random_small(cfg, 4, vocab);
        let mut plain_caches: Vec<KvCache> = plain.config.new_kv_caches();
        let mut pending = plain
            .forward_batch(&prompt, 0, &mut plain_caches)
            .pop()
            .expect("a non-empty prompt returns logits");
        let mut greedy = Vec::with_capacity(max_new);
        for pos in (prompt.len()..).take(max_new) {
            let tok = pending
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are finite"))
                .map(|(i, _)| i)
                .expect("a non-empty vocabulary");
            greedy.push(tok);
            pending = plain.forward_token(tok, pos, &mut plain_caches);
        }

        assert_eq!(
            result.generated_tokens, greedy,
            "a draft model may make decoding faster and may not make it different"
        );
    }

    /// **`q` must be the distribution the token was really sampled
    /// from**, at every temperature.
    ///
    /// The rejection rule accepts with probability `min(1, p(x)/q(x))`
    /// and resamples from `max(0, p - q)`. A drafter that overstates
    /// its own confidence, reporting a point mass for a token it
    /// actually drew from a spread distribution, makes `p/q` too small:
    /// tokens get rejected that should have been accepted, and the
    /// residual it resamples from is not the right residual. The output
    /// stops being the target's distribution.
    ///
    /// This cannot be caught at temperature 0, where the true
    /// distribution IS a point mass and a dishonest report is
    /// accidentally correct. That is exactly why this test sets a
    /// temperature: a suite that only checks greedy decoding will pass
    /// a drafter that lies.
    #[test]
    fn the_reported_distribution_is_the_one_sampled_from_at_temperature() {
        let target = {
            let mut c = test_dense_fixture();
            c.vocab_size = 32;
            c
        };
        let decoder = Decoder::new_random_small(test_dense_fixture(), 2, 32);
        let sampling = SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let mut d = DraftModelSpeculator::new(decoder, &target, sampling.clone(), 3, 4, 0.0)
            .expect("matching vocabularies");

        let block = d.propose(&[1, 2, 3], &[], 4);
        assert_eq!(block.len(), 4);

        let spread = block.dists().iter().any(|dist| dist.support().len() > 1);
        assert!(
            spread,
            "at temperature 1.0 a real model's draft distribution is not a point mass;              if it were, this test could not tell an honest report from a lie"
        );

        for (token, dist) in block.tokens().iter().zip(block.dists()) {
            let q = dist.prob(*token);
            assert!(q > 0.0, "the sampled token must be in its own support");
            assert!(
                q < 1.0,
                "a spread distribution reported as certainty is the lie this test exists for"
            );
            let total: f32 = dist.support().iter().map(|&(_, p)| p).sum();
            assert!(
                (total - 1.0).abs() < 1e-4,
                "a reported distribution must be normalised, got {total}"
            );
        }
    }
}
