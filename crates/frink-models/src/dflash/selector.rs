//! Turning a noise block's output into draft tokens: the DFlash2
//! candidate selector (`build_dflash2_selector`, `dflash.cpp:558-648`,
//! and the greedy lattice walk in `speculative.cpp:1346-1420`), and
//! the plain DFlash read-out (`speculative.cpp:1430-1460`).

use frink_core::weight_matrix::WeightMatrix;

use super::forward::BlockOutput;

/// DFlash2's selector: a pairwise score between a candidate and its
/// predecessor, `<prev[p] * gate(h), next[c]> + logit[c]`.
pub(super) struct Selector {
    /// `{n_embd, rank}`: projects a position's hidden to the gate.
    pub(super) hidden: WeightMatrix,
    /// `{rank, vocab}` codebooks, one row per token.
    pub(super) prev: WeightMatrix,
    pub(super) next: WeightMatrix,
    pub(super) top_k: usize,
}

/// The indices of the `k` largest values (`ggml_top_k`), largest
/// first. The order only matters on an exact tie in the walk below,
/// where llama.cpp's is unspecified.
pub(super) fn top_k(row: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    let k = k.min(row.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
        row[b].total_cmp(&row[a]).then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    idx
}

/// The first maximum (`std::max_element`).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `softmax(scores)` at the argmax: `1 / sum(exp(s_k - s_max))`.
fn top_prob(scores: &[f32], best: usize) -> f32 {
    let sum: f32 = scores.iter().map(|&s| (s - scores[best]).exp()).sum();
    1.0 / sum
}

impl Selector {
    /// The greedy walk over positions `1..n_block`: each position's
    /// candidates are scored against the one chosen before it (the
    /// anchor, at position 1), and the walk stops early where the
    /// chosen score's softmax over the candidates is below `p_min`.
    ///
    /// Only the scores for the predecessor actually chosen are
    /// computed: the lattice llama.cpp packs holds every predecessor's
    /// row, and the walk reads one of them per position.
    ///
    /// Returns each drafted token with the score it was chosen on.
    pub(super) fn walk(&self, out: &BlockOutput, anchor: usize, p_min: f32) -> Vec<(usize, f32)> {
        let n_block = out.logits.len();
        let mut drafted = Vec::with_capacity(n_block.saturating_sub(1));
        let mut predecessor = anchor;
        for i in 1..n_block {
            let cands = top_k(&out.logits[i], self.top_k);
            let gate = self.hidden.apply(&out.hidden[i]);
            let cond: Vec<f32> = self
                .prev
                .dequant_row(predecessor)
                .iter()
                .zip(&gate)
                .map(|(p, g)| p * g)
                .collect();
            let scores: Vec<f32> = cands
                .iter()
                .map(|&c| {
                    let next = self.next.dequant_row(c);
                    let pair: f32 = next.iter().zip(&cond).map(|(a, b)| a * b).sum();
                    pair + out.logits[i][c]
                })
                .collect();
            let best = argmax(&scores);
            if p_min > 0.0 && top_prob(&scores, best) < p_min {
                break;
            }
            predecessor = cands[best];
            drafted.push((predecessor, scores[best]));
        }
        drafted
    }
}

/// Plain DFlash: each mask position's argmax, stopping where its
/// probability among the top 10 falls below `p_min` (the draft's
/// `top_k = 10` sampler, `speculative.cpp:1114-1121`), with the logit
/// each was chosen on.
pub(super) fn read_out(out: &BlockOutput, p_min: f32) -> Vec<(usize, f32)> {
    let mut drafted = Vec::new();
    for row in &out.logits[1..] {
        let cands = top_k(row, 10);
        let best = cands[0];
        if p_min > 0.0 {
            let vals: Vec<f32> = cands.iter().map(|&c| row[c]).collect();
            if top_prob(&vals, 0) < p_min {
                break;
            }
        }
        drafted.push((best, row[best]));
    }
    drafted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_returns_the_largest_indices_largest_first() {
        let row = [0.1, 3.0, -1.0, 2.0, 2.5];
        assert_eq!(top_k(&row, 3), vec![1, 4, 3]);
        assert_eq!(top_k(&row, 9).len(), 5);
    }

    #[test]
    fn the_top_probability_is_the_argmax_softmax() {
        let s = [0.0f32, (2.0f32).ln()];
        // exp(0 - ln2) + exp(0) = 1.5, so p = 2/3.
        assert!((top_prob(&s, 1) - 2.0 / 3.0).abs() < 1e-6);
    }
}
