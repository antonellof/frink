//! Host MoE: the expert residency plan, routing, and the per-slot
//! expert runs (parallel and serial).

use std::sync::atomic::Ordering;

use frink_core::weight_matrix::WeightMatrix;
use frink_moe::{
    route_top_k, run_expert_placed, ExpertPlacement, ExpertWeights, GluAct, PlacementPlan,
};

use crate::config::ModelConfig;

use super::{Decoder, ExpertBacking, LayerWeights};

impl Decoder {
    /// The shared expert store's live counters, when this model runs
    /// with store-backed (streamed) routed experts -- `None` for fully
    /// resident models. Every store-backed layer shares one store, so
    /// the first one found speaks for the whole model.
    pub fn expert_store_stats(&self) -> Option<frink_core::expert_store::ExpertStoreStats> {
        self.layers.iter().find_map(|l| match &l.moe.experts {
            ExpertBacking::Stored { store, .. } => Some(store.stats()),
            ExpertBacking::Resident(_) => None,
        })
    }

    /// Builds one global device-residency plan across ALL layers'
    /// routed experts against the single configured VRAM budget --
    /// every `(layer, expert)` candidate competes in one hotness-
    /// ordered pass and the running byte total is shared, so the
    /// budget cannot be re-spent per layer (the accounting bug the
    /// earlier per-layer `placement_plan` calls had: N layers would
    /// plan N x the configured bytes). Dense layers contribute no
    /// candidates (their sole expert always runs on CPU). Rebuilt per
    /// forward call so it tracks observed hotness; not yet
    /// performance-tuned, a disclosed limit.
    pub(super) fn residency_plan(&self, vram_budget_bytes: u64) -> frink_moe::ResidencyPlan {
        let mut sizes_per_layer: Vec<Vec<usize>> = Vec::with_capacity(self.layers.len());
        let mut counts_per_layer: Vec<Vec<u64>> = Vec::with_capacity(self.layers.len());
        let mut any_observed = false;
        for layer in &self.layers {
            if Self::is_dense_layer(layer) {
                sizes_per_layer.push(Vec::new());
                counts_per_layer.push(Vec::new());
                continue;
            }
            sizes_per_layer.push(
                (0..layer.moe.n_experts())
                    .map(|e| layer.moe.expert_bytes(e))
                    .collect(),
            );
            let counts: Vec<u64> = layer
                .moe
                .activation_counts
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect();
            any_observed |= counts.iter().any(|&c| c > 0);
            counts_per_layer.push(counts);
        }
        PlacementPlan::plan_layers_against_global_budget(
            &sizes_per_layer,
            any_observed.then_some(counts_per_layer.as_slice()),
            vram_budget_bytes,
        )
    }

    /// True if this layer has nothing to route: exactly one expert and
    /// no shared experts, the shape every non-MoE model (and every
    /// DeepSeek-style "leading dense layer") loads as. Top-1 selection
    /// out of one expert always picks it, and its weight is always
    /// exactly 1.0 regardless of gating function (softmax over one
    /// logit is trivially 1.0; sigmoid-then-renormalize divides the
    /// selected score by itself) -- so skipping the router matmul,
    /// `route_top_k`'s sort/exp/renormalize work, and
    /// `combine_expert_outputs`'s Vec-wrapping for this case is not an
    /// approximation, it produces the exact same result.
    pub(super) fn is_dense_layer(layer: &LayerWeights) -> bool {
        layer.moe.n_experts() == 1 && layer.moe.shared_experts.is_empty()
    }

    /// llama.cpp `mul_mat_id` style: shared Q8 act + one flat parallel
    /// region over `(slot, row_pair)` for gate∥up (2-row SDOT), then
    /// SwiGLU, then per-slot down. Three regions, none nested, all
    /// through `frink_core::par` so they follow whichever scheduler
    /// `FRINK_CPU_POOL` selected.
    ///
    /// `experts[slot]` is the expert `decision.expert_ids[slot]`, lent by
    /// `MoeWeights::with_experts` from either backing.
    pub(super) fn cpu_moe_topk_parallel_slots(
        experts: &[&ExpertWeights],
        normed2: &[f32],
        decision: &frink_moe::RoutingDecision,
        hidden_dim: usize,
        act: GluAct,
    ) -> Option<Vec<(Vec<f32>, f32)>> {
        if !frink_core::weight_matrix::cpu_int_dot_for(
            frink_core::weight_matrix::IntDotShape::Matvec,
        ) || !normed2.len().is_multiple_of(32)
        {
            return None;
        }
        let n_slots = decision.expert_ids.len();
        if n_slots == 0 {
            return Some(Vec::new());
        }
        assert_eq!(experts.len(), n_slots, "one lent expert per routed slot");
        for ex in experts {
            if ex.gate.rows() == 0
                || ex.up.rows() != ex.gate.rows()
                || ex.down.rows() != hidden_dim
                || ex.gate.cols() != normed2.len()
                || ex.up.cols() != normed2.len()
                || ex.down.cols() != ex.gate.rows()
            {
                return None;
            }
            if !matches!(
                &ex.gate,
                WeightMatrix::Quantized {
                    kind: frink_core::QuantKind::Q4_0 | frink_core::QuantKind::Q8_0,
                    ..
                }
            ) || !matches!(
                &ex.up,
                WeightMatrix::Quantized {
                    kind: frink_core::QuantKind::Q4_0 | frink_core::QuantKind::Q8_0,
                    ..
                }
            ) {
                return None;
            }
        }
        let ffn_rows = experts[0].gate.rows();
        // Even ffn_rows: par_chunks_mut(2) never crosses a slot boundary.
        if !ffn_rows.is_multiple_of(2) {
            return None;
        }
        let q8 = frink_quant::quantize_activations_q8(normed2);
        let mut gate = vec![0f32; n_slots * ffn_rows];
        let mut up = vec![0f32; n_slots * ffn_rows];
        frink_core::par::chunks_mut2(&mut gate, &mut up, 2, 1, |p, gc, uc| {
            let row0 = p * 2;
            let slot = row0 / ffn_rows;
            let r = row0 % ffn_rows;
            let ex = experts[slot];
            if let (Some((g0, g1)), Some((u0, u1))) = (
                ex.gate.dot_pair_cpu_q8(r, &q8),
                ex.up.dot_pair_cpu_q8(r, &q8),
            ) {
                gc[0] = g0;
                gc[1] = g1;
                uc[0] = u0;
                uc[1] = u1;
            } else {
                gc[0] = ex.gate.dot_row_cpu_q8(r, &q8).unwrap_or(0.0);
                gc[1] = ex.gate.dot_row_cpu_q8(r + 1, &q8).unwrap_or(0.0);
                uc[0] = ex.up.dot_row_cpu_q8(r, &q8).unwrap_or(0.0);
                uc[1] = ex.up.dot_row_cpu_q8(r + 1, &q8).unwrap_or(0.0);
            }
        });
        let mut activated = vec![0f32; n_slots * ffn_rows];
        // The combine here is always parallel (decode's `n_slots *
        // ffn_rows` sits under `frink_core::matmul`'s own fork
        // threshold), which is why this does not just call `act.apply`.
        // `GluAct::combine` is that function one element at a time --
        // this used to match the variant here and multiply `f(gate) *
        // up` itself, a second spelling of the activation that a
        // parameterised variant (xIELU reads `up` alone) could not fit.
        frink_core::par::items_mut(&mut activated, 1, |idx, a| {
            *a = act.combine(gate[idx], up[idx])
        });
        let mut outs: Vec<(Vec<f32>, f32)> = decision
            .weights
            .iter()
            .map(|&w| (vec![0f32; hidden_dim], w))
            .collect();
        frink_core::par::items_mut(&mut outs, 1, |slot, (out, _)| {
            let ex = experts[slot];
            let act_slot = &activated[slot * ffn_rows..(slot + 1) * ffn_rows];
            if act_slot.len().is_multiple_of(32) {
                let down_q8 = frink_quant::quantize_activations_q8(act_slot);
                if let Some(d) = ex.down.apply_cpu_q8(&down_q8) {
                    *out = d;
                    return;
                }
            }
            *out = ex.down.apply(act_slot);
        });
        Some(outs)
    }

    /// Fallback: serial top-k with shared Q8 act (pre-mul_mat_id path).
    ///
    /// `weight_before_ffn` (`crate::routed_weight_site`): each slot
    /// reads its own scaled input and the shared activation is not
    /// built. `experts[slot]` is the expert `decision.expert_ids[slot]`,
    /// as for `cpu_moe_topk_parallel_slots`.
    pub(super) fn cpu_moe_serial_experts(
        experts: &[&ExpertWeights],
        normed2: &[f32],
        decision: &frink_moe::RoutingDecision,
        plan: Option<&PlacementPlan>,
        act: GluAct,
        weight_before_ffn: bool,
    ) -> Vec<(Vec<f32>, f32)> {
        let shared_act = if !weight_before_ffn
            && frink_core::weight_matrix::cpu_int_dot_for(
                frink_core::weight_matrix::IntDotShape::Matvec,
            )
            && normed2.len().is_multiple_of(32)
            && plan
                .map(|p| {
                    decision
                        .expert_ids
                        .iter()
                        .all(|&eid| matches!(p.placement_for(eid), ExpertPlacement::Cpu))
                })
                .unwrap_or(true)
        {
            Some(frink_quant::quantize_activations_q8(normed2))
        } else {
            None
        };
        decision
            .expert_ids
            .iter()
            .zip(decision.weights.iter())
            .zip(experts.iter())
            .map(|((&eid, &w), ex)| {
                let placement = plan
                    .map(|p| p.placement_for(eid))
                    .unwrap_or(ExpertPlacement::Cpu);
                let (input, w) =
                    crate::routed_weight_site::routed_slot(normed2, w, weight_before_ffn);
                if let Some(ref q8) = shared_act {
                    if let (Some(gate), Some(up)) =
                        (ex.gate.apply_cpu_q8(q8), ex.up.apply_cpu_q8(q8))
                    {
                        let activated = act.apply(&gate, &up);
                        return (ex.down.apply(&activated), w);
                    }
                }
                (run_expert_placed(&input, ex, placement, act), w)
            })
            .collect()
    }

    /// Runs one position's normalized hidden state through this
    /// layer's MoE FFN block, given already-computed router logits for
    /// that position, returning the combined output to add back into
    /// the residual stream. Shared by `forward_token` (router computed
    /// via a single `apply` call, since there's only one position) and
    /// `forward_batch`'s per-position loop (router computed via one
    /// batched `apply_batch` call up front, sliced per position here --
    /// see `forward_batch`'s doc comment for why that batching matters
    /// and must not be lost by calling this per position instead).
    /// `gpu_vram_budget_bytes`: see `Decoder::gpu_vram_budget_bytes`'s
    /// doc comment -- `None` dispatches every routed expert through
    /// `run_expert_placed` with `ExpertPlacement::Cpu`, which is
    /// exactly `run_expert`'s own behavior, so this is a real
    /// zero-behavior-change default, not just "probably fine."
    /// One token's routing decision for one MoE layer.
    ///
    /// Three shapes, in the order llama.cpp's `build_moe_ffn` decides
    /// them: grouped selection when the checkpoint declares expert
    /// groups; the biased/scaled port when the layer carries
    /// `exp_probs_b` or the model carries a non-unit
    /// `expert_weights_scale`; otherwise the plain top-k this decoder has
    /// always used. The last arm is kept rather than folded into
    /// `route_top_k_biased` so that every checkpoint without those two
    /// features routes through byte-identical code to before.
    ///
    /// `exp_probs_b` together with expert groups is refused at load
    /// (`loader.rs`), so that combination cannot reach here.
    pub(super) fn route_for_layer(
        layer: &LayerWeights,
        router_logits: &[f32],
        config: &ModelConfig,
    ) -> frink_moe::RoutingDecision {
        match (
            config.moe.expert_group_count,
            config.moe.expert_group_used_count,
        ) {
            (Some(n_groups), Some(k_per_group)) if n_groups > 1 && k_per_group > 0 => {
                frink_moe::route_top_k_grouped(
                    router_logits,
                    n_groups,
                    k_per_group,
                    config.moe.n_experts_active,
                    config.moe.gating,
                    config.moe.norm_topk_prob,
                )
            }
            _ if layer.moe.exp_probs_bias.is_some() || config.moe.expert_weights_scale != 1.0 => {
                frink_moe::route_top_k_biased(
                    router_logits,
                    layer.moe.exp_probs_bias.as_deref(),
                    config.moe.n_experts_active,
                    config.moe.gating,
                    config.moe.norm_topk_prob,
                    config.moe.expert_weights_scale,
                )
            }
            _ => route_top_k(
                router_logits,
                config.moe.n_experts_active,
                config.moe.gating,
                config.moe.norm_topk_prob,
            ),
        }
    }
}
