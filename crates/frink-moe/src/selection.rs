//! Where the expert-selection bias (`exp_probs_b`) enters the top-k.
//!
//! llama.cpp has two rules, and they pick different experts whenever the
//! bias is not the same for every expert:
//!
//! - `build_moe_ffn` adds the bias to the GATED scores,
//!   `selection_probs = probs + exp_probs_b` (`src/llama-graph.cpp:2276-2280`,
//!   the add at `:2279`), DeepSeek-V3's aux-loss-free rule. Every graph
//!   that hands `build_moe_ffn` its bias takes it.
//! - `src/models/kolibri1.cpp:135-141` adds it to the RAW logits,
//!   `selection = logits + exp_probs_b`, ranks that with
//!   `ggml_argsort_top_k`, and hands the ids to `build_moe_ffn` as
//!   `selected_experts_in` with `exp_probs_b = nullptr` (`:149,155`).
//!   `build_moe_ffn` then skips its own selection
//!   (`llama-graph.cpp:2320-2324`), group mask included, and gathers the
//!   weights from `gating(logits)` at those ids (`:2336`).
//!
//! In both the weights are the UNBIASED gated scores; only the ranking
//! differs. Under sigmoid gating `sigmoid` squashes the logits into
//! `(0, 1)`, so a bias of a given size reorders far more experts after
//! the gating than before it.

/// Which ranking a biased router's top-k reads. An enum rather than a
/// `bool`, so a call site names the rule it means and a new rule is a
/// compile error at every `match`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionRule {
    /// `top_k(gating(logits) + bias)`: `llama-graph.cpp:2279`, the rule
    /// `build_moe_ffn` applies when it is handed `exp_probs_b`.
    BiasAfterGating,
    /// `top_k(logits + bias)`: `kolibri1.cpp:138`. The group filter is
    /// not applied, because the ids bypass `build_moe_ffn`'s selection.
    BiasBeforeGating,
}

impl SelectionRule {
    /// The scores the top-k ranks. `probs` is `gating(logits)`; with no
    /// bias both rules rank `probs`, which orders the experts as the raw
    /// logits do for every gating function (each is monotonic, and
    /// softmax shares one denominator).
    pub(crate) fn selection_scores(
        self,
        logits: &[f32],
        probs: &[f32],
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let Some(bias) = bias else {
            return probs.to_vec();
        };
        assert_eq!(
            bias.len(),
            probs.len(),
            "exp_probs_b must have one entry per expert"
        );
        let ranked = match self {
            Self::BiasAfterGating => probs,
            Self::BiasBeforeGating => logits,
        };
        ranked.iter().zip(bias).map(|(s, b)| s + b).collect()
    }

    /// Whether the expert-group filter runs on the selection scores
    /// (`llama-graph.cpp:2289-2316`). Under `BiasBeforeGating` the graph
    /// supplies the ids, and the masked scores are never read.
    pub(crate) fn masks_groups(self) -> bool {
        match self {
            Self::BiasAfterGating => true,
            Self::BiasBeforeGating => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SelectionRule::{BiasAfterGating, BiasBeforeGating};
    use crate::{route_top_k_biased, route_top_k_grouped_biased, GatingFunction};

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Expert 0 has the largest logit; expert 2 the largest bias. After
    /// the sigmoid the logit gap 3.0 vs -0.5 shrinks to 0.95 vs 0.38,
    /// so a bias of 2 lifts expert 2 to the top; on the raw logits the
    /// same bias leaves it at 1.5, under expert 0's 3.0.
    const LOGITS: [f32; 4] = [3.0, 0.0, -0.5, 0.5];
    const BIAS: [f32; 4] = [0.0, 0.0, 2.0, 0.25];

    #[test]
    fn the_two_rules_pick_different_experts() {
        let route = |rule| {
            route_top_k_biased(
                &LOGITS,
                Some(&BIAS),
                rule,
                1,
                GatingFunction::Sigmoid,
                false,
                1.0,
            )
        };
        let after = route(BiasAfterGating);
        let before = route(BiasBeforeGating);
        assert_eq!(after.expert_ids, vec![2], "sigmoid(l) + b");
        assert_eq!(before.expert_ids, vec![0], "l + b");
        // Both weigh by the UNBIASED sigmoid at the chosen id.
        assert_eq!(after.weights, vec![sigmoid(LOGITS[2])]);
        assert_eq!(before.weights, vec![sigmoid(LOGITS[0])]);
    }

    /// k = 2 with renormalisation, the shape `kolibri1` runs: the sets
    /// differ, and each set's weights sum to one over its own members.
    #[test]
    fn the_rules_disagree_on_a_top_two_and_both_renormalise() {
        let route = |rule| {
            route_top_k_biased(
                &LOGITS,
                Some(&BIAS),
                rule,
                2,
                GatingFunction::Sigmoid,
                true,
                1.0,
            )
        };
        let after = route(BiasAfterGating);
        let before = route(BiasBeforeGating);
        assert_eq!(after.expert_ids, vec![2, 0]);
        assert_eq!(before.expert_ids, vec![0, 2]);
        let mut a = after.expert_ids.clone();
        let mut b = before.expert_ids.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "this pair agrees as a SET; the next one does not");
        let after3 = route_top_k_biased(
            &LOGITS,
            Some(&[0.0, 0.0, 2.0, 0.9]),
            BiasAfterGating,
            2,
            GatingFunction::Sigmoid,
            true,
            1.0,
        );
        let before3 = route_top_k_biased(
            &LOGITS,
            Some(&[0.0, 0.0, 2.0, 0.9]),
            BiasBeforeGating,
            2,
            GatingFunction::Sigmoid,
            true,
            1.0,
        );
        // sigmoid + b: [0.95, 0.5, 2.38, 1.52] -> {2, 3};
        // l + b:       [3.0, 0.0, 1.5, 1.4]    -> {0, 2}.
        assert_eq!(after3.expert_ids, vec![2, 3]);
        assert_eq!(before3.expert_ids, vec![0, 2]);
        for d in [&after, &before, &after3, &before3] {
            let sum: f32 = d.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "{d:?}");
        }
    }

    /// Without a bias there is nothing for the rules to disagree about.
    #[test]
    fn without_a_bias_the_rules_agree() {
        for gating in [
            GatingFunction::Softmax,
            GatingFunction::Sigmoid,
            GatingFunction::SqrtSoftplus,
        ] {
            let a = route_top_k_biased(&LOGITS, None, BiasAfterGating, 2, gating, true, 1.0);
            let b = route_top_k_biased(&LOGITS, None, BiasBeforeGating, 2, gating, true, 1.0);
            assert_eq!(a.expert_ids, b.expert_ids, "{gating:?}");
            assert_eq!(a.weights, b.weights, "{gating:?}");
        }
    }

    /// `kolibri1`'s ids bypass the group block, so two groups of two
    /// with one group kept must not confine a `BiasBeforeGating` pick.
    #[test]
    fn bias_before_gating_ignores_the_group_filter() {
        let route = |rule| {
            route_top_k_grouped_biased(
                &LOGITS,
                Some(&[0.0; 4]),
                rule,
                2,
                1,
                2,
                GatingFunction::Sigmoid,
                false,
                1.0,
            )
        };
        // Groups {0, 1} and {2, 3}; the first wins on its top two, so
        // the grouped rule takes both of its experts.
        assert_eq!(route(BiasAfterGating).expert_ids, vec![0, 1]);
        assert_eq!(route(BiasBeforeGating).expert_ids, vec![0, 3]);
    }
}
