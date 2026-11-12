//! llama.cpp's sampler chain, as the shrinking candidate list it
//! actually is.
//!
//! Every sampler in `src/llama-sampler.cpp` takes one
//! `llama_token_data_array` -- a list of `(token id, logit, p)` -- and
//! **removes entries from it**. The next sampler in the chain then sees
//! only the survivors, and any sampler that needs probabilities calls
//! `llama_sampler_softmax_impl` (`:293`), which renormalises **over the
//! survivors only**.
//!
//! That renormalisation is the part a "just zero the logits you don't
//! want" implementation silently gets wrong, and frink's did. With
//! `--top-k 40 --top-p 0.95`, llama.cpp's top-p sums probabilities that
//! were divided by the mass of the top 40; frink summed probabilities
//! divided by the mass of the **whole vocabulary**, which is larger, so
//! its running sum crossed 0.95 later and it kept MORE tokens than
//! llama.cpp for the same two flags. Same idea as the temperature-order
//! bug: not a reordering of independent steps, a different candidate
//! set.
//!
//! So this module models the candidate list rather than a keep-mask,
//! and each filter here is a transcription of the corresponding
//! `_apply` in llama.cpp, cited at its definition.
//!
//! **`min_keep`.** Several of llama.cpp's filters take a `min_keep`
//! floor and guard their cutoff with `i + 1 >= min_keep`. frink has no
//! such parameter (llama.cpp's own CLI does not expose one either --
//! `common/arg.cpp` has no `--min-keep`; only the server's JSON body
//! carries it), and `common_params_sampling::min_keep` defaults to `0`
//! (`common/common.h:228`), which makes every one of those guards
//! vacuously true. They are therefore folded out below rather than
//! carried as a field nothing sets.

/// The candidate list a chain of samplers narrows.
///
/// Held **sorted by descending logit** at all times. llama.cpp tracks a
/// `sorted` flag instead and lets each sampler sort lazily, but every
/// filter that cares either sorts first or only reads the maximum, so
/// maintaining the invariant eagerly reaches the same sets with less
/// state to get wrong.
pub(crate) struct Candidates {
    /// Token id of each live candidate.
    ids: Vec<usize>,
    /// Logit of each live candidate, parallel to `ids`.
    logits: Vec<f32>,
    /// Probability of each live candidate, parallel to `ids`. Only
    /// meaningful immediately after [`Self::softmax`]; the filters that
    /// need it call that themselves, exactly as llama.cpp's do.
    probs: Vec<f32>,
}

impl Candidates {
    /// The whole vocabulary as one candidate list, sorted by descending
    /// logit.
    ///
    /// Ties break on the lower token id so that a run is reproducible
    /// given a seed even when a model emits exactly equal logits, which
    /// `sort_unstable_by` on the logit alone does not guarantee.
    pub(crate) fn new(logits: &[f32]) -> Self {
        let mut ids: Vec<usize> = (0..logits.len()).collect();
        ids.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        let ordered: Vec<f32> = ids.iter().map(|&i| logits[i]).collect();
        let probs = vec![0.0f32; ordered.len()];
        Candidates {
            ids,
            logits: ordered,
            probs,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// Keep only the first `n` candidates.
    fn truncate(&mut self, n: usize) {
        self.ids.truncate(n);
        self.logits.truncate(n);
        self.probs.truncate(n);
    }

    /// `llama_sampler_softmax_impl` (`src/llama-sampler.cpp:293`):
    /// softmax over the **live candidates only**, so the normaliser is
    /// the mass of whatever survived so far.
    fn softmax(&mut self) {
        if self.logits.is_empty() {
            return;
        }
        // The list is sorted, so `data[0]` is the maximum -- the same
        // shortcut llama.cpp takes when `cur_p->sorted`.
        let max = self.logits[0];
        let mut sum = 0.0f32;
        for (p, &l) in self.probs.iter_mut().zip(self.logits.iter()) {
            *p = (l - max).exp();
            sum += *p;
        }
        if sum <= 0.0 || !sum.is_finite() {
            let uniform = 1.0 / self.probs.len() as f32;
            self.probs.fill(uniform);
            return;
        }
        for p in self.probs.iter_mut() {
            *p /= sum;
        }
    }

    /// `llama_sampler_top_k_impl` (`src/llama-sampler.cpp:321`): keep
    /// the `k` highest logits; `k <= 0` disables the filter.
    pub(crate) fn top_k(&mut self, k: usize) {
        if k == 0 {
            return;
        }
        self.truncate(k.min(self.len()));
    }

    /// `llama_sampler_top_p_apply` (`src/llama-sampler.cpp:1360`).
    ///
    /// The cutoff test is `cum_sum >= p`, and the candidate that crosses
    /// it is **included** (`last_idx = i + 1`). `p >= 1.0` disables.
    pub(crate) fn top_p(&mut self, p: f32) {
        if p >= 1.0 {
            return;
        }
        self.softmax();
        let mut cum_sum = 0.0f32;
        let mut last_idx = self.len();
        for (i, &prob) in self.probs[..self.len()].iter().enumerate() {
            cum_sum += prob;
            if cum_sum >= p {
                last_idx = i + 1;
                break;
            }
        }
        self.truncate(last_idx);
    }

    /// `llama_sampler_min_p_apply` (`src/llama-sampler.cpp:1556`).
    ///
    /// Keeps every candidate whose probability is at least `p` times the
    /// **top** candidate's, which llama.cpp expresses on the logits
    /// rather than the probabilities:
    ///
    /// ```text
    /// min_logit = data[0].logit + logf(p)   // p_i >= p * p_max
    /// ```
    ///
    /// -- the softmax normaliser cancels out of the ratio, so no
    /// `softmax` call is needed and the answer does not depend on which
    /// filters ran before this one.
    ///
    /// It does depend, absolutely, on the **temperature not having run
    /// yet**: temperature scales every logit, so it scales the gap
    /// `logit_i - logit_max` that is being compared against the fixed
    /// `ln(p)`. Running min-p after temperature would make the surviving
    /// set a function of `--temp`, which in llama.cpp it is not
    /// (`common/common.h:259-269` puts `MIN_P` before `TEMPERATURE`).
    ///
    /// `i` starts at 1 because "the first token always matches": the
    /// list is never emptied, even by a `p > 1.0` that no candidate can
    /// satisfy.
    pub(crate) fn min_p(&mut self, p: f32) {
        if p <= 0.0 || self.logits.is_empty() {
            return;
        }
        let min_logit = self.logits[0] + p.ln();
        let mut i = 1;
        while i < self.len() && self.logits[i] >= min_logit {
            i += 1;
        }
        self.truncate(i);
    }

    /// `llama_sampler_typical_apply`
    /// (`src/llama-sampler.cpp:1713-1769`): locally typical sampling.
    ///
    /// Not a truncation of the TOP of the distribution. It computes the
    /// distribution's entropy `H`, scores each candidate by how far its
    /// surprisal `-ln p` is from `H`, and keeps the smallest set of
    /// candidates -- ordered by that distance, so from the MIDDLE of the
    /// distribution outward -- whose probabilities exceed `p`. A token
    /// that is far more likely than typical is dropped just as a token
    /// far less likely is.
    ///
    /// Two details that separate this from top-p, both upstream's:
    ///
    /// * the cutoff is `cum_sum > p`, strictly, where top-p uses `>=`;
    /// * the surviving set is written back in DISTANCE order and
    ///   upstream marks the array unsorted (`cur_p->sorted = false`).
    ///   This module holds the sorted invariant instead, so the
    ///   survivors are re-sorted by logit. The SET is what the rest of
    ///   the chain reads, and the set is identical.
    ///
    /// `p >= 1.0` disables it, which is llama.cpp's default
    /// (`typ_p = 1.00f`, `common/common.h:230`).
    pub(crate) fn typical_p(&mut self, p: f32) {
        if p >= 1.0 || self.logits.is_empty() {
            return;
        }
        self.softmax();
        let entropy: f32 = self.probs.iter().map(|&q| -q * q.ln()).sum();
        // The absolute difference between surprisal and entropy, which
        // is what "typical" means here.
        let shifted: Vec<f32> = self
            .probs
            .iter()
            .map(|&q| (-q.ln() - entropy).abs())
            .collect();
        let mut order: Vec<usize> = (0..self.len()).collect();
        // Ties broken on the index, where upstream's `std::sort` leaves
        // them unspecified: a generation reproducible from a seed must
        // not depend on a sort's tie-breaking.
        order.sort_by(|&a, &b| shifted[a].total_cmp(&shifted[b]).then(a.cmp(&b)));

        let mut cum_sum = 0.0f32;
        let mut last_idx = order.len();
        for (i, &idx) in order.iter().enumerate() {
            cum_sum += self.probs[idx];
            if cum_sum > p {
                last_idx = i + 1;
                break;
            }
        }
        let mut keep: Vec<usize> = order[..last_idx].to_vec();
        keep.sort_unstable();
        self.retain_positions(&keep);
    }

    /// `llama_sampler_top_n_sigma_apply`
    /// (`src/llama-sampler.cpp:3002-3039`): mask every candidate more
    /// than `n` standard deviations of the LOGITS below the maximum.
    ///
    /// The statistic is over the raw logits, not the probabilities, and
    /// `-inf` entries (a candidate an earlier step already masked) are
    /// excluded from the mean and the deviation but still masked.
    ///
    /// Upstream sets the losing logits to `-inf` rather than removing
    /// them, and this does the same: the candidates stay in the list
    /// with zero probability, so a later `top_k` still counts them, as
    /// it does upstream. Because the list is held sorted by descending
    /// logit and the cut is a threshold, the masked candidates are
    /// always a suffix, so the invariant survives untouched.
    ///
    /// `n <= 0.0` disables it, which is llama.cpp's default
    /// (`top_n_sigma = -1.00f`, `common/common.h:250`). Note that `0.0`
    /// is a no-op and NOT greedy decoding, as of upstream PR 13345.
    pub(crate) fn top_n_sigma(&mut self, n: f32) {
        if n <= 0.0 || self.len() <= 1 {
            return;
        }
        let mut max = self.logits[0];
        let mut logits_sum = 0.0f32;
        let mut valid_count = 0usize;
        for &l in self.logits.iter() {
            if l != f32::NEG_INFINITY {
                max = max.max(l);
                logits_sum += l;
                valid_count += 1;
            }
        }
        let mean = if valid_count > 0 {
            logits_sum / valid_count as f32
        } else {
            0.0
        };
        // Upstream accumulates a float from a double `pow`, so each
        // addition rounds to f32 while the square itself does not.
        let mut acc = 0.0f32;
        for &l in self.logits.iter() {
            if l != f32::NEG_INFINITY {
                acc = (acc as f64 + ((l - mean) as f64).powi(2)) as f32;
            }
        }
        let std = if valid_count > 0 {
            (acc as f64 / valid_count as f64).sqrt() as f32
        } else {
            0.0
        };
        let threshold = max - n * std;
        for l in self.logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// `llama_sample_xtc_apply` (`src/llama-sampler.cpp:2137-2168`):
    /// "exclude top choices".
    ///
    /// The only sampler here that removes candidates from the TOP. With
    /// probability `probability` it drops every candidate whose
    /// probability is at or above `threshold` EXCEPT the least likely of
    /// them, on the theory that the most obvious continuation is the
    /// boring one. `chance` is the draw that decides, and it comes from
    /// the generation's own seeded stream -- see
    /// [`crate::sampling::Sampler::xtc_roll`] for why it is a parameter
    /// here rather than an RNG this module owns.
    ///
    /// Three guards, all upstream's: a non-positive probability, a
    /// threshold above 0.5 (above which more than one candidate can
    /// never clear it and the sampler is meaningless), and a list with
    /// fewer than two candidates. `min_keep` is folded out at 0, as
    /// everywhere else in this module.
    ///
    /// `pos_last > 0` is what keeps the last-remaining candidate: when
    /// only the top candidate clears the threshold, nothing is removed.
    pub(crate) fn xtc(&mut self, probability: f32, threshold: f32, chance: f32) {
        if probability <= 0.0 || threshold > 0.5 || self.len() < 2 {
            return;
        }
        if chance > probability {
            return;
        }
        self.softmax();
        let mut pos_last = 0usize;
        for (i, &prob) in self.probs[..self.len()].iter().enumerate() {
            if prob >= threshold {
                pos_last = i;
            } else {
                break;
            }
        }
        if pos_last > 0 {
            self.drop_front(pos_last);
        }
    }

    /// `llama_sampler_dry_apply`'s step 4 (`src/llama-sampler.cpp:3320`):
    /// subtract the DRY penalty from each candidate that has one.
    ///
    /// The penalties themselves are [`crate::dry::DryParams::penalties`],
    /// which is where the Z-algorithm and the sequence breakers live;
    /// this is only the subtraction, and the re-sort that keeps this
    /// module's invariant after logits have moved by different amounts.
    /// Upstream marks the array unsorted here for the same reason.
    pub(crate) fn dry(&mut self, penalties: &std::collections::HashMap<usize, f32>) {
        if penalties.is_empty() {
            return;
        }
        let mut moved = false;
        for (id, logit) in self.ids.iter().zip(self.logits.iter_mut()) {
            if let Some(&penalty) = penalties.get(id) {
                *logit -= penalty;
                moved = true;
            }
        }
        if moved {
            self.resort();
        }
    }

    /// Keep the candidates at these positions, which must be ascending
    /// and in range. The order is preserved, so the sorted invariant
    /// survives a filter that only ever removes.
    fn retain_positions(&mut self, keep: &[usize]) {
        let ids: Vec<usize> = keep.iter().map(|&i| self.ids[i]).collect();
        let logits: Vec<f32> = keep.iter().map(|&i| self.logits[i]).collect();
        self.probs.truncate(ids.len());
        self.ids = ids;
        self.logits = logits;
    }

    /// Drop the `n` most likely candidates. Only XTC does this.
    fn drop_front(&mut self, n: usize) {
        self.ids.drain(..n);
        self.logits.drain(..n);
        self.probs.truncate(self.ids.len());
    }

    /// Restore the descending-logit invariant after logits have been
    /// changed by different amounts. Ties break on the token id, for the
    /// same reproducibility reason [`Self::new`] does.
    fn resort(&mut self) {
        let mut order: Vec<usize> = (0..self.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            self.logits[b]
                .total_cmp(&self.logits[a])
                .then(self.ids[a].cmp(&self.ids[b]))
        });
        self.ids = order.iter().map(|&i| self.ids[i]).collect();
        self.logits = order.iter().map(|&i| self.logits[i]).collect();
    }

    /// `llama_sampler_temp_impl` (`src/llama-sampler.cpp:265`) for
    /// `temp > 0`: divide every surviving logit by the temperature.
    ///
    /// LAST in llama.cpp's default chain, after every truncation filter
    /// -- see [`Self::min_p`] and `sampling::filtered_distribution` for
    /// why that is a specification and not a convenience.
    pub(crate) fn temperature(&mut self, temp: f32) {
        if temp <= 0.0 {
            return;
        }
        for l in self.logits.iter_mut() {
            *l /= temp;
        }
    }

    /// The surviving candidates as a full-vocabulary distribution:
    /// softmaxed over the survivors, scattered back to token ids, zero
    /// everywhere the chain filtered out.
    pub(crate) fn into_distribution(mut self, vocab: usize) -> Vec<f32> {
        self.softmax();
        let mut out = vec![0.0f32; vocab];
        for (&id, &p) in self.ids.iter().zip(self.probs.iter()) {
            if let Some(slot) = out.get_mut(id) {
                *slot = p;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// min-p's threshold is llama.cpp's, on the logits:
    /// `data[0].logit + ln(p)`.
    ///
    /// Checked against arithmetic done by hand rather than against the
    /// implementation: logits `[4, 3, 2, 1]` with `p = 0.2` give
    /// `ln(0.2) = -1.6094`, so the threshold is `2.3905` and exactly the
    /// candidates at 4 and 3 clear it.
    #[test]
    fn min_p_keeps_candidates_within_ln_p_of_the_top_logit() {
        let mut c = Candidates::new(&[4.0, 3.0, 2.0, 1.0]);
        c.min_p(0.2);
        assert_eq!(c.ids, vec![0, 1], "threshold is 4 + ln(0.2) = 2.3905");

        // Equality is inclusive (`>=` in llama.cpp's loop guard).
        let mut c = Candidates::new(&[0.0, (0.5f32).ln(), -5.0]);
        c.min_p(0.5);
        assert_eq!(c.ids, vec![0, 1], "p_i == p * p_max must be kept");

        // The first token always matches, so the list is never emptied.
        let mut c = Candidates::new(&[1.0, 0.9, 0.8]);
        c.min_p(2.0);
        assert_eq!(c.ids, vec![0]);

        // 0.0 disables.
        let mut c = Candidates::new(&[4.0, 3.0, 2.0, 1.0]);
        c.min_p(0.0);
        assert_eq!(c.len(), 4);
    }

    /// top-p sums probabilities renormalised over the **survivors of the
    /// earlier filters**, not over the whole vocabulary.
    ///
    /// This is the divergence a keep-mask implementation cannot express.
    /// Logits `[3, 2, 1, 0, 0, 0, 0, 0]`: the top-2 mass is
    /// `e^3 + e^2` and within it token 0 already holds `e^3 / (e^3 +
    /// e^2) = 0.731`, so `--top-k 2 --top-p 0.72` keeps ONE token. Over
    /// the full-vocabulary softmax token 0 holds only `0.578`, so the
    /// same two flags would keep two.
    #[test]
    fn top_p_renormalises_over_what_top_k_left() {
        let logits = vec![3.0f32, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mut c = Candidates::new(&logits);
        c.top_k(2);
        c.top_p(0.72);
        assert_eq!(
            c.ids,
            vec![0],
            "renormalised over the top 2, token 0 already holds 0.731"
        );

        // Sanity on the other half of the claim: without the top-k the
        // full-vocabulary mass of token 0 is below 0.72, so the same
        // top-p keeps a second candidate. If this ever equals the case
        // above, the test above no longer proves renormalisation.
        let mut c = Candidates::new(&logits);
        c.top_p(0.72);
        assert_eq!(c.ids, vec![0, 1]);
    }

    /// The candidate that crosses the top-p threshold is kept, not
    /// dropped (`last_idx = i + 1`).
    #[test]
    fn top_p_includes_the_candidate_that_crosses_the_threshold() {
        // Two candidates at p = 0.5 each; `cum_sum >= 0.5` fires on the
        // first, which is therefore the last one kept.
        let mut c = Candidates::new(&[1.0f32, 1.0]);
        c.top_p(0.5);
        assert_eq!(c.ids, vec![0]);

        let mut c = Candidates::new(&[1.0f32, 1.0]);
        c.top_p(0.6);
        assert_eq!(c.ids, vec![0, 1]);
    }

    /// Equal logits sort by ascending token id, so a run stays
    /// reproducible given a seed. `sort_unstable_by` on the logit alone
    /// leaves ties in an unspecified order.
    #[test]
    fn equal_logits_break_ties_on_the_token_id() {
        let c = Candidates::new(&[1.0f32; 6]);
        assert_eq!(c.ids, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn the_published_distribution_is_normalised_and_zero_outside_the_survivors() {
        let mut c = Candidates::new(&[3.0f32, 2.0, 1.0, 0.0]);
        c.top_k(2);
        c.temperature(0.5);
        let probs = c.into_distribution(4);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(probs[2], 0.0);
        assert_eq!(probs[3], 0.0);
        // temp 0.5 doubles the logit gap: e^2 / (e^2 + 1) = 0.8808.
        assert!((probs[0] - 0.880_797).abs() < 1e-5, "got {}", probs[0]);
    }

    /// Logits whose softmax is exactly `probs`, so a case lifted from
    /// llama.cpp's `tests/test-sampling.cpp` (which builds its candidate
    /// array as `logit = logf(p)`) can be run here unchanged.
    fn from_probs(probs: &[f32]) -> Vec<f32> {
        probs.iter().map(|p| p.ln()).collect()
    }

    /// Locally typical sampling keeps the candidates CLOSEST to the
    /// distribution's entropy, which is not the same set as the most
    /// likely ones -- and on a flat-ish distribution it drops the leader.
    ///
    /// Both rows are llama.cpp's own
    /// (`tests/test-sampling.cpp:345-346`), re-run against `libllama`
    /// (b7650) to read the surviving TOKEN IDS rather than only the
    /// probabilities, because upstream writes the survivors back in
    /// distance order and this module re-sorts them:
    ///
    /// * `{0.97, 0.01, 0.01, 0.01}` at `p = 0.5` keeps `{0}`;
    /// * `{0.4, 0.2, 0.2, 0.2}` at `p = 0.5` keeps `{1, 2, 3}` -- the
    ///   0.4 leader is the ATYPICAL one and is dropped.
    ///
    /// A top-p implementation given the same two rows keeps `{0}` in
    /// both, so the second row is what makes this a test of typical-p.
    #[test]
    fn typical_p_keeps_the_candidates_nearest_the_entropy_and_can_drop_the_leader() {
        let mut c = Candidates::new(&from_probs(&[0.97, 0.01, 0.01, 0.01]));
        c.typical_p(0.5);
        assert_eq!(c.ids, vec![0]);

        let mut c = Candidates::new(&from_probs(&[0.4, 0.2, 0.2, 0.2]));
        c.typical_p(0.5);
        assert_eq!(c.ids, vec![1, 2, 3], "the 0.4 leader is atypical here");

        // And the contrast that names what this is not: top-p at the
        // same 0.5 KEEPS the leader (and one more), where typical-p
        // dropped it. No truncation-from-the-top filter can produce the
        // set above.
        let mut c = Candidates::new(&from_probs(&[0.4, 0.2, 0.2, 0.2]));
        c.top_p(0.5);
        assert_eq!(c.ids, vec![0, 1]);
    }

    /// The cutoff is `cum_sum > p` STRICTLY, where top-p's is `>=`, and
    /// the candidate that crosses it is kept.
    ///
    /// Logits `[3, 2, 1, 0]`, hand-computed: the softmax is
    /// `[0.6439, 0.2369, 0.0871, 0.0321]`, the entropy is `0.9477`, and
    /// the distances `|-ln p - H|` are `[0.5074, 0.4925, 1.4927,
    /// 2.4919]`, so the visiting order is `1, 0, 2, 3` and the running
    /// sums are `0.2369, 0.8808, 0.9679`. At `p = 0.5` the second sum
    /// already exceeds it, so two survive; at `p = 0.9` the third does,
    /// so three do. libllama returns exactly `[1, 0]` and `[1, 0, 2]`.
    #[test]
    fn typical_p_cuts_on_a_strict_running_sum_over_the_distance_order() {
        let mut c = Candidates::new(&[3.0f32, 2.0, 1.0, 0.0]);
        c.typical_p(0.5);
        assert_eq!(c.ids, vec![0, 1], "0.2369 then 0.8808 crosses 0.5");

        let mut c = Candidates::new(&[3.0f32, 2.0, 1.0, 0.0]);
        c.typical_p(0.9);
        assert_eq!(c.ids, vec![0, 1, 2], "0.8808 is not > 0.9; 0.9679 is");

        // 1.0 disables it, which is llama.cpp's default and is what
        // keeps the nine-step default chain a no-op.
        let mut c = Candidates::new(&[3.0f32, 2.0, 1.0, 0.0]);
        c.typical_p(1.0);
        assert_eq!(c.ids, vec![0, 1, 2, 3]);
    }

    /// The `>` is STRICT, and llama.cpp's `top_p` beside it is not.
    ///
    /// Four equal logits give four probabilities of exactly 0.25, whose
    /// partial sums (0.25, 0.5, 0.75) are exact in an f32 -- so this is
    /// the one shape where strict and inclusive give different answers
    /// without depending on rounding. libllama (b7650) returns 2, 3 and
    /// 4 survivors at p = 0.25, 0.5 and 0.75; the inclusive reading
    /// would return 1, 2 and 3.
    ///
    /// Only the COUNT is asserted. With every distance tied, upstream's
    /// `std::sort` leaves the survivors in an unspecified order and
    /// libllama actually returns `2, 0` for the first row; this module
    /// breaks ties on the token id instead, because a generation
    /// reproducible from a seed must not depend on a sort's internals.
    #[test]
    fn typical_ps_running_sum_is_strict_where_top_ps_is_inclusive() {
        for (p, expected) in [(0.25f32, 2usize), (0.5, 3), (0.75, 4)] {
            let mut c = Candidates::new(&[0.0f32; 4]);
            c.typical_p(p);
            assert_eq!(c.len(), expected, "typical_p({p})");
        }
        // The contrast, on the same exact sums: top-p's `>=` keeps one
        // fewer at each threshold.
        for (p, expected) in [(0.25f32, 1usize), (0.5, 2), (0.75, 3)] {
            let mut c = Candidates::new(&[0.0f32; 4]);
            c.top_p(p);
            assert_eq!(c.len(), expected, "top_p({p})");
        }
    }

    /// top-n-sigma masks on the standard deviation of the LOGITS, and
    /// masks to `-inf` rather than removing.
    ///
    /// Hand-computed for `probs {0.1, 0.2, 0.3, 0.4}`, whose logits are
    /// `[-2.3026, -1.6094, -1.2040, -0.9163]`: the mean is `-1.5081`,
    /// the deviations square to `1.0840` in total, so `sigma = 0.5206`
    /// and the threshold at `n = 1` is `-0.9163 - 0.5206 = -1.4369`.
    /// The two smallest logits fall below it. libllama returns exactly
    /// that: ids 0 and 1 at `-inf`, ids 2 and 3 untouched.
    #[test]
    fn top_n_sigma_masks_everything_more_than_n_sigma_below_the_top_logit() {
        let mut c = Candidates::new(&from_probs(&[0.1, 0.2, 0.3, 0.4]));
        c.top_n_sigma(1.0);
        // Sorted descending, so the surviving order is 3, 2, 1, 0.
        assert_eq!(c.ids, vec![3, 2, 1, 0], "masking must not remove");
        assert!(c.logits[0].is_finite() && c.logits[1].is_finite());
        assert_eq!(c.logits[2], f32::NEG_INFINITY);
        assert_eq!(c.logits[3], f32::NEG_INFINITY);
        // The mask is what the rest of the chain sees: zero probability.
        let probs = c.into_distribution(4);
        assert_eq!(probs[0], 0.0);
        assert_eq!(probs[1], 0.0);
        assert!(probs[2] > 0.0 && probs[3] > 0.0);
    }

    /// `n` is a knob, not a switch: tightening it masks more.
    ///
    /// Logits `[4, 3, 2, 1, 0]` have mean 2 and `sigma = sqrt(2) =
    /// 1.4142`. At `n = 1` the threshold is `4 - 1.4142 = 2.5858`, so
    /// the 4 and the 3 survive; at `n = 0.5` it is `3.2929`, so only the
    /// 4 does. libllama returns exactly those two masks.
    ///
    /// `n <= 0` disables it, and `0.0` is a NO-OP rather than greedy
    /// decoding as of upstream PR 13345 -- a reading worth pinning,
    /// because the obvious one (zero sigmas means keep only the max) is
    /// what llama.cpp used to do and no longer does.
    #[test]
    fn a_smaller_n_masks_more_and_a_non_positive_n_masks_nothing() {
        let masked = |n: f32| {
            let mut c = Candidates::new(&[4.0f32, 3.0, 2.0, 1.0, 0.0]);
            c.top_n_sigma(n);
            c.logits.iter().filter(|l| l.is_finite()).count()
        };
        assert_eq!(masked(1.0), 2, "threshold 4 - sqrt(2) = 2.5858");
        assert_eq!(masked(0.5), 1, "threshold 4 - 0.7071 = 3.2929");
        assert_eq!(masked(3.0), 5, "threshold 4 - 4.2426 keeps everything");
        assert_eq!(masked(0.0), 5, "0.0 is a no-op, not greedy");
        assert_eq!(masked(-1.0), 5, "the default, disabled");
    }

    /// XTC removes the top candidates ABOVE the threshold, keeping the
    /// least likely of them -- the only filter here that cuts from the
    /// top.
    ///
    /// Every row is llama.cpp's own
    /// (`tests/test-sampling.cpp:338-343`) re-run against `libllama`,
    /// on `probs {0.4, 0.3, 0.2, 0.1}`:
    ///
    /// | threshold | survivors |
    /// |---|---|
    /// | 0.09 | `{3}` |
    /// | 0.19 | `{2, 3}` |
    /// | 0.29 | `{1, 2, 3}` |
    /// | 0.39 | all four |
    ///
    /// The last row is the guard that matters: only the 0.4 candidate
    /// clears 0.39, so `pos_last` is 0 and XTC removes NOTHING. An
    /// implementation that dropped every candidate at or above the
    /// threshold would empty the list on that row.
    #[test]
    fn xtc_removes_every_candidate_above_the_threshold_except_the_least_likely() {
        for (threshold, expected) in [
            (0.09f32, vec![3usize]),
            (0.19, vec![2, 3]),
            (0.29, vec![1, 2, 3]),
            (0.39, vec![0, 1, 2, 3]),
        ] {
            let mut c = Candidates::new(&from_probs(&[0.4, 0.3, 0.2, 0.1]));
            // A chance below the probability, so the draw always fires.
            c.xtc(0.99, threshold, 0.0);
            assert_eq!(c.ids, expected, "threshold {threshold}");
        }
    }

    /// XTC is a coin flip, and the coin is the caller's.
    ///
    /// `chance > probability` skips, so a draw above the configured
    /// probability leaves the list alone. This is the whole reason
    /// `Sampler::xtc_roll` exists: pass a roll the chain never made and
    /// the filter silently never runs.
    #[test]
    fn xtc_only_fires_when_the_draw_falls_under_the_probability() {
        let fired = |probability: f32, chance: f32| {
            let mut c = Candidates::new(&from_probs(&[0.4, 0.3, 0.2, 0.1]));
            c.xtc(probability, 0.09, chance);
            c.ids.len() < 4
        };
        assert!(fired(0.5, 0.49));
        assert!(fired(0.5, 0.5), "the test is `chance > probability`");
        assert!(!fired(0.5, 0.51));
        // And both of upstream's disabling guards.
        assert!(!fired(0.0, 0.0), "probability 0.0 disables");
        let mut above_half = Candidates::new(&from_probs(&[0.4, 0.3, 0.2, 0.1]));
        above_half.xtc(1.0, 0.51, 0.0);
        assert_eq!(above_half.ids.len(), 4, "a threshold above 0.5 disables");
    }

    /// DRY subtracts per-token penalties and the list stays sorted.
    ///
    /// The sorted invariant is not decoration: every filter after this
    /// one reads `logits[0]` as the maximum, so a DRY penalty that
    /// pushed the leader below its neighbour without a re-sort would
    /// give min-p and top-p the wrong reference point. Upstream marks
    /// its array unsorted here for the same reason.
    #[test]
    fn dry_subtracts_its_penalty_and_leaves_the_list_sorted() {
        let mut penalties = std::collections::HashMap::new();
        penalties.insert(0usize, 2.5f32);
        let mut c = Candidates::new(&[3.0f32, 2.0, 1.0]);
        c.dry(&penalties);
        assert_eq!(c.ids, vec![1, 2, 0], "3.0 - 2.5 = 0.5 now sorts last");
        assert_eq!(c.logits, vec![2.0, 1.0, 0.5]);
        // An empty penalty map is a no-op, which is what keeps `dry` in
        // the default chain free.
        let mut c = Candidates::new(&[3.0f32, 2.0, 1.0]);
        c.dry(&std::collections::HashMap::new());
        assert_eq!(c.ids, vec![0, 1, 2]);
        assert_eq!(c.logits, vec![3.0, 2.0, 1.0]);
    }
}
