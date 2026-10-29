//! The Metal MoE launches: the on-device top-k decode, the batched
//! prefill experts, and the resident-expert decode.

use frink_moe::{route_top_k, ExpertWeights};

use crate::config::ModelConfig;

use super::{Decoder, ExpertBacking, LayerWeights};

impl Decoder {
    /// One Metal CB for all top-k routed experts (weighted sum). Returns
    /// `None` if any expert lacks a Metal launch (caller falls back).
    #[cfg(feature = "metal")]
    pub(super) fn try_metal_moe_topk(
        layer: &LayerWeights,
        normed2: &[f32],
        decision: &frink_moe::RoutingDecision,
    ) -> Option<Vec<f32>> {
        if decision.expert_ids.is_empty() {
            return Some(vec![0f32; normed2.len()]);
        }
        // A `MatvecLaunch` borrows the expert's bytes, so every expert
        // in the batch has to stay alive until the command buffer is
        // encoded and has completed (`launch_moe_topk_swiglu` waits).
        // Resident experts live in a `Vec` and are indexed. Streamed ones
        // are materialised by `MoeWeights::stored_experts`: one
        // `acquire_many` pins the whole top-k, and each owned view's
        // `WeightBytes::Shared` carries its pin, so the vector that
        // outlives the launches is what keeps the store from reusing a
        // slot the GPU is reading. With a Metal slot slab those views are
        // slices of one Shared `MTLBuffer` and bind it in place.
        //
        // A `?` on any expert without a launch drops the batch and falls
        // back. There used to be a validation pass here that lent each
        // expert once before this one lent them again: for a streamed
        // layer that was a second store acquire per expert, a read on a
        // miss, for nothing.
        let streamed: Vec<ExpertWeights>;
        let experts: Vec<&ExpertWeights> = match &layer.moe.experts {
            ExpertBacking::Resident(experts) => decision
                .expert_ids
                .iter()
                .map(|&eid| &experts[eid])
                .collect(),
            ExpertBacking::Stored { .. } => {
                // A read failure is not fatal here: the host path reads
                // the same bytes and reports it.
                streamed = layer.moe.stored_experts(&decision.expert_ids).ok()?;
                streamed.iter().collect()
            }
        };
        let mut launches = Vec::with_capacity(experts.len());
        for (ex, &weight) in experts.iter().zip(decision.weights.iter()) {
            launches.push(frink_metal::gpu::MoeExpertLaunch {
                gate: Self::metal_matvec_launch(&ex.gate)?,
                up: Self::metal_matvec_launch(&ex.up)?,
                down: Self::metal_matvec_launch(&ex.down)?,
                weight,
            });
        }

        match frink_metal::gpu::launch_moe_topk_swiglu(normed2, &launches) {
            Ok(out) => Some(out),
            Err(e) => {
                eprintln!("frink: Metal MoE top-k fuse failed, falling back: {e}");
                None
            }
        }
    }

    /// Prefill MoE FFN on Metal: host route over T, then one packed-id CB
    /// (`launch_moe_prefill_q4_0`). Shared experts (if any) run as dense
    /// batch FFN on the host/GPU path afterwards — not through `mul_mm_id`.
    /// Returns FFN outs `[T, H]` or `None`.
    #[cfg(feature = "metal")]
    pub(super) fn try_metal_moe_prefill_batch(
        layer_idx: usize,
        layer: &LayerWeights,
        normed2_batch: &[f32],
        router_logits_batch: &[f32],
        batch_size: usize,
        hidden_dim: usize,
        config: &ModelConfig,
    ) -> Option<Vec<f32>> {
        let acts = config.layer_ffn_acts(layer_idx);
        if batch_size == 0
            || !frink_core::metal_dense_enabled()
            || !acts.routed.is_swiglu()
            // The GPU router kernels take router weights and nothing
            // else: no `exp_probs_b` input, no `expert_weights_scale`
            // uniform, no groups. See `gpu_router_matches_host_routing`.
            || !Self::gpu_router_matches_host_routing(layer, config)
        {
            return None;
        }
        let ExpertBacking::Resident(_) = &layer.moe.experts else {
            return None;
        };
        let packed = Self::moe_packed_id(&layer.moe)?;
        let top_k = config.moe.n_experts_active;
        if top_k == 0 || top_k > 8 || packed.hidden_rows != hidden_dim {
            return None;
        }
        let n_experts = layer.moe.n_experts().max(1);
        let mut ids = Vec::with_capacity(batch_size * top_k);
        let mut route = Vec::with_capacity(batch_size * top_k);
        for b in 0..batch_size {
            let logits = &router_logits_batch[b * n_experts..(b + 1) * n_experts];
            let decision = route_top_k(logits, top_k, config.moe.gating, config.moe.norm_topk_prob);
            layer.moe.record_activations(&decision.expert_ids);
            if decision.expert_ids.len() != top_k {
                return None;
            }
            for (&eid, &w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
                ids.push(eid as i32);
                route.push(w);
            }
        }
        let mut out = match frink_metal::gpu::launch_moe_prefill_q4_0(
            normed2_batch,
            batch_size,
            &packed,
            &ids,
            &route,
            top_k,
        ) {
            Ok(out) => out,
            Err(e) => {
                eprintln!("frink: Metal MoE prefill failed, CPU fallback: {e}");
                return None;
            }
        };
        Self::accumulate_shared_experts_batch(
            layer,
            normed2_batch,
            batch_size,
            hidden_dim,
            &mut out,
            // The shared experts are `build_ffn`'s site, not the routed
            // experts' the fence above checked; on the host either way.
            acts.dense,
        );
        Some(out)
    }

    /// Phase-2 of resident MoE decode: experts on GPU `x2`, add into GPU `h`.
    #[cfg(feature = "metal")]
    pub(super) fn try_metal_moe_experts_resident(
        layer: &LayerWeights,
        decision: &frink_moe::RoutingDecision,
    ) -> Option<()> {
        if decision.expert_ids.is_empty() {
            return Some(());
        }
        let pending: Vec<(usize, f32)> = decision
            .expert_ids
            .iter()
            .zip(decision.weights.iter())
            .map(|(&eid, &w)| (eid, w))
            .collect();
        // Bail on the backing BEFORE validating the experts. The
        // validation loop below calls `with_expert`, which for
        // `Stored` backing acquires a lease and so can read from the
        // checkpoint file. Refusing afterwards meant a streamed layer
        // paid one read per top-k expert and then threw all of them
        // away, before `try_metal_moe_topk` read the same experts
        // again. This path stays resident-only for now, but it must
        // decline for free.
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        for &(eid, _) in &pending {
            let ex = &experts[eid];
            if Self::metal_matvec_launch(&ex.gate).is_none()
                || Self::metal_matvec_launch(&ex.up).is_none()
                || Self::metal_matvec_launch(&ex.down).is_none()
            {
                return None;
            }
        }
        let mut launches = Vec::with_capacity(pending.len());
        for &(eid, weight) in &pending {
            let ex = &experts[eid];
            launches.push(frink_metal::gpu::MoeExpertLaunch {
                gate: Self::metal_matvec_launch(&ex.gate)?,
                up: Self::metal_matvec_launch(&ex.up)?,
                down: Self::metal_matvec_launch(&ex.down)?,
                weight,
            });
        }
        match frink_metal::attn::launch_moe_decode_experts(&launches) {
            Ok(()) => Some(()),
            Err(e) => {
                eprintln!("frink: Metal MoE experts failed, falling back: {e}");
                None
            }
        }
    }
}
