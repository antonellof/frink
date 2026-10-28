//! The FFN bodies over a batch of rows and one row's combine: dense,
//! routed with shared experts, gpt-oss, and the dispatch between them.

use frink_core::matmul::rms_norm;
use frink_core::weight_matrix::WeightMatrix;
use frink_moe::{combine_expert_outputs, run_expert, ExpertPlacement, GluAct, PlacementPlan};

use crate::config::ModelConfig;

use super::ffn_block;
use super::{Decoder, GptOssLayer, LayerWeights};

impl Decoder {
    /// Shared expert as dense batch FFN (llama qwen2moe: not through
    /// `mul_mat_id`). Optional sigmoid gate scales per token.
    pub(super) fn accumulate_shared_experts_batch(
        layer: &LayerWeights,
        normed2_batch: &[f32],
        batch_size: usize,
        hidden_dim: usize,
        acc: &mut [f32],
        act: GluAct,
    ) {
        for shex in &layer.moe.shared_experts {
            // Prefer one Metal FFN CB (gate∥up→SiLU→down) over three
            // `apply_batch` round-trips — Qwen shexp is 4× routed width.
            #[cfg(feature = "metal")]
            let down = if frink_core::metal_dense_enabled() && batch_size >= 4 {
                match (
                    shex.gate.mul_mm_sg_launch(),
                    shex.up.mul_mm_sg_launch(),
                    shex.down.mul_mm_sg_launch(),
                    // The launch's last argument selects GELU over
                    // SiLU inside the kernel; hardcoding `false` here
                    // ran a shared expert as SwiGLU on a GeGLU model,
                    // and `!is_swiglu()` would run a third activation
                    // as GELU. `None` keeps the host path.
                    act.fused_kernel_gelu_flag(),
                ) {
                    (Some(g), Some(u), Some(d), Some(gelu)) => {
                        frink_metal::gpu::launch_dense_ffn_swiglu_batch(
                            &g,
                            &u,
                            &d,
                            normed2_batch,
                            batch_size,
                            gelu,
                        )
                        .ok()
                    }
                    _ => None,
                }
            } else {
                None
            };
            #[cfg(not(feature = "metal"))]
            let down: Option<Vec<f32>> = None;
            // Without `metal` the binding above is a literal `None`; the
            // fallback is the only arm and clippy flags the unwrap.
            #[cfg_attr(not(feature = "metal"), allow(clippy::unnecessary_literal_unwrap))]
            let down = down.unwrap_or_else(|| {
                let ffn_acts = shex.gate.quantize_batch_acts(normed2_batch, batch_size);
                // One rotation for the pair when they share a fold
                // (`apply_batch_pair_with_acts`); two independent calls
                // otherwise, which is every unfolded model.
                let (gate, up) = WeightMatrix::apply_batch_pair_with_acts(
                    &shex.gate,
                    &shex.up,
                    normed2_batch,
                    batch_size,
                    ffn_acts.as_ref(),
                );
                let activated = act.apply(&gate, &up);
                shex.down.apply_batch(&activated, batch_size)
            });
            if let Some(gate_w) = &layer.moe.shared_expert_gate {
                for b in 0..batch_size {
                    let x = &normed2_batch[b * hidden_dim..(b + 1) * hidden_dim];
                    let logit: f32 = gate_w.iter().zip(x.iter()).map(|(g, v)| g * v).sum();
                    let scale = 1.0 / (1.0 + (-logit).exp());
                    let out = &down[b * hidden_dim..(b + 1) * hidden_dim];
                    let row = &mut acc[b * hidden_dim..(b + 1) * hidden_dim];
                    for (a, &o) in row.iter_mut().zip(out.iter()) {
                        *a += scale * o;
                    }
                }
            } else {
                for (a, &o) in acc.iter_mut().zip(down.iter()) {
                    *a += o;
                }
            }
        }
    }

    /// `normed2` is what the DENSE half (the shared experts) reads;
    /// `routed_input` is what the routed experts read -- the same slice
    /// for every architecture but Arctic, whose routed branch reads
    /// `ffn_norm_exps(inpSA)` (`crate::router_input`). Two arguments
    /// rather than one so a caller cannot hand the experts the wrong
    /// vector without saying so.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn combine_ffn_outputs_for_position(
        layer_idx: usize,
        layer: &LayerWeights,
        normed2: &[f32],
        routed_input: &[f32],
        router_logits: &[f32],
        config: &ModelConfig,
        hidden_dim: usize,
        plan: Option<&PlacementPlan>,
    ) -> Vec<f32> {
        let decision = Self::route_for_layer(layer, router_logits, config);
        let acts = config.layer_ffn_acts(layer_idx);
        let act = acts.routed;
        layer.moe.record_activations(&decision.expert_ids);
        // No prefetch here: the two consumers of a stored layer's top-k
        // (`try_metal_moe_topk` and `MoeWeights::with_experts`) each ask
        // for the whole batch before reading any of it.

        // Metal: fuse all top-k experts into one CB (one wait) when every
        // routed expert has Metal matvec launches. Shared experts (rare
        // for OLMoE) still run on the host after.
        // `launch_moe_topk_swiglu` is SwiGLU-only, so a GeGLU MoE layer
        // keeps the host path rather than taking a kernel that computes
        // a different activation.
        #[cfg(feature = "metal")]
        if frink_core::metal_dense_enabled()
            && act.is_swiglu()
            && layer.moe.shared_experts.is_empty()
        {
            if let Some(fused) = Self::try_metal_moe_topk(layer, routed_input, &decision) {
                return fused;
            }
        }

        let routed_outputs: Vec<(Vec<f32>, f32)> = {
            // llama.cpp mul_mat_id: one shared Q8 act + flat (slot,row)
            // parallel over all top-k experts (not serial expert loops each
            // with their own rayon fork-join).
            let all_cpu = plan
                .map(|p| {
                    decision
                        .expert_ids
                        .iter()
                        .all(|&eid| matches!(p.placement_for(eid), ExpertPlacement::Cpu))
                })
                .unwrap_or(true);
            // The parallel-slot kernel quantises ONE input for every
            // slot; a model that weights the input per slot
            // (`crate::routed_weight_site`) takes the serial site.
            let before = config.moe.routed_weight_before_ffn;
            // ONE lend of all top-k for either backing, so a streamed
            // layer runs exactly the kernels a resident one does
            // (`MoeWeights::with_experts`). This used to match
            // `ExpertBacking::Resident` here and send a streamed layer
            // to the serial body unconditionally.
            layer.moe.with_experts(&decision.expert_ids, |experts| {
                if all_cpu && !before {
                    if let Some(outs) = Self::cpu_moe_topk_parallel_slots(
                        experts,
                        routed_input,
                        &decision,
                        hidden_dim,
                        act,
                    ) {
                        return outs;
                    }
                }
                Self::cpu_moe_serial_experts(experts, routed_input, &decision, plan, act, before)
            })
        };
        // Shared experts fire on every token regardless of routing, so
        // there's no offload decision to make for them the way there
        // is for routed experts -- always CPU, matching `run_expert`.
        let mut shared_outputs: Vec<Vec<f32>> = layer
            .moe
            .shared_experts
            .iter()
            .map(|e| run_expert(normed2, e, acts.dense))
            .collect();
        // Qwen2-MoE-specific: see `MoeWeights::shared_expert_gate`'s doc
        // comment. Scaling here (before `combine_expert_outputs`, which
        // is architecture-agnostic and knows nothing about this gate)
        // keeps the gate a decoder-level detail, not a frink-moe API
        // change.
        if let Some(gate) = &layer.moe.shared_expert_gate {
            let gate_logit: f32 = gate.iter().zip(normed2.iter()).map(|(g, x)| g * x).sum();
            let gate_value = 1.0 / (1.0 + (-gate_logit).exp());
            for out in shared_outputs.iter_mut() {
                for x in out.iter_mut() {
                    *x *= gate_value;
                }
            }
        }

        combine_expert_outputs(&routed_outputs, &shared_outputs, hidden_dim)
    }

    /// The dense FFN for a whole batch of positions in three batched
    /// matmuls (gate, up, down) instead of three per position.
    ///
    /// This is the counterpart of what `forward_hidden_batch` already
    /// did for Q/K/V and the router, and it is where a dense model's
    /// prefill time actually goes: `WeightMatrix::apply_batch` reads
    /// each weight row once and dots it against every position, rather
    /// than re-reading the whole FFN for each one.
    ///
    /// `None` for anything that is not a plain dense layer -- MoE
    /// routing is per position by construction, so those keep the
    /// sequential path.
    ///
    /// On a GPU backend the per-position alternative is one *fused*
    /// gate+up+SiLU+down launch (`apply_gpu_dense_ffn_swiglu`), so this
    /// used to be gated off there: three separate batched launches lost
    /// to it while `apply_batch` was still a batched *matvec*.
    ///
    /// That stopped being true once the simdgroup GEMM landed, and the
    /// old gate turned out to be the dominant cost of Metal prefill --
    /// a 512-token prompt ran the FFN one position at a time, 512 x
    /// n_layers fused launches, which a profile put at 90% of prefill
    /// while the GEMM it bypassed accounted for 21%.
    ///
    /// Decode (`batch_size == 1`) still takes the fused per-position
    /// launch, which is the right shape there.
    pub(super) fn dense_ffn_batch(
        layer_idx: usize,
        layer: &LayerWeights,
        normed2_batch: &[f32],
        batch_size: usize,
        config: &ModelConfig,
    ) -> Option<Vec<f32>> {
        let act = config.layer_ffn_acts(layer_idx).dense;
        // Match the GPU `mul_mm` threshold: below it the per-call launch
        // overhead outweighs the weight reuse.
        if !Self::is_dense_layer(layer) || batch_size < 4 {
            return None;
        }
        // On a GPU backend this only wins when the weights have a real
        // batched GEMM; otherwise `apply_batch` is a batched matvec and
        // loses to the fused per-position launch.
        #[cfg(any(feature = "metal", feature = "cuda"))]
        {
            #[cfg(feature = "metal")]
            let gpu_dense = frink_core::weight_matrix::metal_dense_enabled();
            #[cfg(not(feature = "metal"))]
            let gpu_dense = false;
            #[cfg(feature = "cuda")]
            let gpu_dense = gpu_dense || frink_core::weight_matrix::cuda_dense_enabled();
            if gpu_dense {
                let all_gemm = layer.moe.with_expert(0, |ex| {
                    ex.gate.prefers_gpu_batch()
                        && ex.up.prefers_gpu_batch()
                        && ex.down.prefers_gpu_batch()
                });
                if !all_gemm {
                    return None;
                }
            }
        }
        layer.moe.record_activations(&[0]);
        // One command buffer for the whole FFN when every matrix has a
        // simdgroup GEMM: gate and up feed the activation and the down
        // projection without the intermediates ever touching the host.
        // Three separate launches cost three round trips per layer plus
        // four copies of a `batch x ffn_dim` tensor. Not for a layer
        // with a norm between the activation and `down`
        // (`ffn_sub_norm`): the kernel has no such site.
        // ...nor a bias at any of its three sites (`crate::proj_bias`).
        #[cfg(feature = "metal")]
        if let (true, Some(gelu), None, None) = (
            frink_core::weight_matrix::metal_dense_enabled(),
            act.fused_kernel_gelu_flag(),
            layer.moe.ffn_sub_norm.as_ref(),
            layer.moe.dense_bias.as_ref(),
        ) {
            let fused = layer.moe.with_expert(0, |ex| {
                let (g, u, d) = (
                    ex.gate.mul_mm_sg_launch()?,
                    ex.up.mul_mm_sg_launch()?,
                    ex.down.mul_mm_sg_launch()?,
                );
                frink_metal::gpu::launch_dense_ffn_swiglu_batch(
                    &g,
                    &u,
                    &d,
                    normed2_batch,
                    batch_size,
                    gelu,
                )
                .ok()
            });
            if let Some(out) = fused {
                return Some(out);
            }
        }
        Some(layer.moe.with_expert(0, |ex| {
            let ffn_acts = ex.gate.quantize_batch_acts(normed2_batch, batch_size);
            let (mut gate, mut up) = WeightMatrix::apply_batch_pair_with_acts(
                &ex.gate,
                &ex.up,
                normed2_batch,
                batch_size,
                ffn_acts.as_ref(),
            );
            // `build_ffn`: `up_b` / `gate_b` before the activation
            // (`crate::proj_bias`). On an ungated layer `gate` is an
            // alias of `up` and only `up_b` exists; `act.apply` reads
            // `up` alone for it, so the gate copy going unbiased is not
            // read.
            if let Some(bias) = &layer.moe.dense_bias {
                bias.add_pre_activation(&mut gate, &mut up, batch_size);
            }
            let mut activated = act.apply(&gate, &up);
            // bitnet.cpp:135-140, per row, on the same vector the row
            // body norms in `frink_moe::run_expert_sub_normed`.
            if let Some(w) = &layer.moe.ffn_sub_norm {
                let width = w.len();
                activated = activated
                    .chunks(width)
                    .flat_map(|row| rms_norm(row, w, config.rms_norm_eps))
                    .collect();
            }
            let mut down = ex.down.apply_batch(&activated, batch_size);
            if let Some(bias) = &layer.moe.dense_bias {
                bias.add_post_down(&mut down, batch_size);
            }
            down
        }))
    }

    /// CPU MoE prefill: bucket tokens by expert, then one
    /// `apply_batch` per expert with tokens instead of per-token
    /// `combine_ffn_outputs_for_position`. Shared experts append via
    /// [`Self::accumulate_shared_experts_batch`]. `None` when gates fail
    /// (small batch, dense, Metal preferred, non-resident, or any
    /// GPU-placed expert). Both gated activations are served here --
    /// the combine goes through [`GluAct`], so GeGLU no longer falls out
    /// to the per-position path.
    /// `routed_batch` is what the experts read, `normed2_batch` what the
    /// shared experts read: the same rows for every architecture but
    /// Arctic (see `combine_ffn_outputs_for_position`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn moe_ffn_batch(
        layer_idx: usize,
        layer: &LayerWeights,
        normed2_batch: &[f32],
        routed_batch: &[f32],
        router_logits_batch: &[f32],
        batch_size: usize,
        config: &ModelConfig,
        plan: Option<&PlacementPlan>,
    ) -> Option<Vec<f32>> {
        let hidden_dim = config.hidden_dim;
        if batch_size < 32 || Self::is_dense_layer(layer) {
            return None;
        }
        // Metal prefill owns MoE when dense Metal is on
        // (`try_metal_moe_prefill_batch`); do not steal the path.
        #[cfg(feature = "metal")]
        if frink_core::metal_dense_enabled() {
            return None;
        }
        let acts = config.layer_ffn_acts(layer_idx);
        let act = acts.routed;
        // Either backing: each bucket's expert is lent for its three
        // GEMMs (`MoeWeights::with_expert`). This was resident-only, so
        // a streamed prefill of 32+ tokens ran per-position matvecs
        // instead and answered differently in the last bits.
        let n_experts = layer.moe.n_experts();
        let all_cpu = plan
            .map(|p| (0..n_experts).all(|eid| matches!(p.placement_for(eid), ExpertPlacement::Cpu)))
            .unwrap_or(true);
        if !all_cpu || n_experts == 0 {
            return None;
        }

        let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
        for b in 0..batch_size {
            let logits = &router_logits_batch[b * n_experts..(b + 1) * n_experts];
            let decision = Self::route_for_layer(layer, logits, config);
            layer.moe.record_activations(&decision.expert_ids);
            for (&eid, &w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
                buckets[eid].push((b, w));
            }
        }

        let mut acc = vec![0f32; batch_size * hidden_dim];
        for (eid, toks) in buckets.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let mut gathered = vec![0f32; n * hidden_dim];
            // Each gathered row is one (token, slot): the slot's input
            // and the weight its output carries come from the one
            // helper the row body uses (`crate::routed_weight_site`).
            let mut out_weights = Vec::with_capacity(n);
            for (i, &(tok, w)) in toks.iter().enumerate() {
                let (input, w) = crate::routed_weight_site::routed_slot(
                    &routed_batch[tok * hidden_dim..(tok + 1) * hidden_dim],
                    w,
                    config.moe.routed_weight_before_ffn,
                );
                gathered[i * hidden_dim..(i + 1) * hidden_dim].copy_from_slice(&input);
                out_weights.push(w);
            }
            let down = layer.moe.with_expert(eid, |ex| {
                let ffn_acts = ex.gate.quantize_batch_acts(&gathered, n);
                let gate = ex
                    .gate
                    .apply_batch_with_acts(&gathered, n, ffn_acts.as_ref());
                let up = ex.up.apply_batch_with_acts(&gathered, n, ffn_acts.as_ref());
                let activated = act.apply(&gate, &up);
                ex.down.apply_batch(&activated, n)
            });
            for (i, (&(tok, _), &w)) in toks.iter().zip(out_weights.iter()).enumerate() {
                let out = &down[i * hidden_dim..(i + 1) * hidden_dim];
                let row = &mut acc[tok * hidden_dim..(tok + 1) * hidden_dim];
                for (a, &o) in row.iter_mut().zip(out.iter()) {
                    *a += w * o;
                }
            }
        }

        Self::accumulate_shared_experts_batch(
            layer,
            normed2_batch,
            batch_size,
            hidden_dim,
            &mut acc,
            acts.dense,
        );
        Some(acc)
    }

    /// gpt-oss's MoE FFN for one position.
    ///
    /// A separate function rather than another branch inside
    /// `combine_ffn_outputs_for_position` on purpose: that path carries
    /// expert-store prefetch, residency placement, a Metal top-k fusion
    /// and a batched parallel-slot kernel, and every one of them would
    /// need its own gpt-oss variant to stay honest. This is the whole
    /// gpt-oss FFN in one readable block, checked end-to-end against
    /// llama.cpp, and slow — routed experts run serially. It is the
    /// correct-first shape; making it fast is a separate change with its
    /// own A/B, not something to smuggle in under a correctness fix.
    ///
    /// Ported from `llama-graph.cpp::build_moe_ffn` with
    /// `gating_op = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT`,
    /// `type_op = LLM_FFN_SWIGLU_OAI_MOE`, `norm_w = false`,
    /// `w_scale = 1`, all four bias tensors present.
    pub(super) fn gpt_oss_ffn(
        layer: &LayerWeights,
        oai: &GptOssLayer,
        normed2: &[f32],
        config: &ModelConfig,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let mut router_logits = layer.moe.router.apply(normed2);
        for (x, b) in router_logits.iter_mut().zip(oai.router_bias.iter()) {
            *x += b;
        }
        // Selection on the raw biased logits, softmax over the winners
        // only -- see `route_top_k_softmax_weight`.
        let decision =
            frink_moe::route_top_k_softmax_weight(&router_logits, config.moe.n_experts_active);
        layer.moe.record_activations(&decision.expert_ids);

        let mut out = vec![0f32; hidden_dim];
        for (slot, &eid) in decision.expert_ids.iter().enumerate() {
            let w = decision.weights[slot];
            let expert_out = layer.moe.with_expert(eid, |ex| {
                frink_moe::run_expert_oai(
                    normed2,
                    ex,
                    &oai.expert_bias[eid],
                    frink_moe::SWIGLU_OAI_ALPHA,
                    frink_moe::SWIGLU_OAI_LIMIT,
                )
            });
            for (o, e) in out.iter_mut().zip(expert_out.iter()) {
                *o += w * e;
            }
        }
        out
    }

    /// `forward_token`'s MoE FFN block for one position: the dense
    /// fast path (see `is_dense_layer`) or the full router+combine path
    /// with the router computed inline via a single-position `apply`.
    pub(super) fn run_ffn_block(
        layer_idx: usize,
        layer: &LayerWeights,
        normed2: &[f32],
        config: &ModelConfig,
        hidden_dim: usize,
        plan: Option<&PlacementPlan>,
        operand: ffn_block::RouterOperand,
    ) -> Vec<f32> {
        if Self::is_dense_layer(layer) {
            layer.moe.record_activations(&[0]);
            // One expert, run exactly the way a routed one is. The GeGLU
            // arm used to be spelled out here and nowhere else, which is
            // precisely how the routed paths ended up SwiGLU-only.
            let act = config.layer_ffn_acts(layer_idx).dense;
            return Self::run_dense_expert(layer, normed2, act, config.rms_norm_eps);
        }
        // What the router reads is the caller's fact (`crate::router_input`):
        // the normed FFN input here, logits computed before attention, or
        // Arctic's normed layer input, which its experts read too.
        let (router_logits, routed_input): (Vec<f32>, &[f32]) = match &operand {
            ffn_block::RouterOperand::FfnInput => (layer.moe.router.apply(normed2), normed2),
            ffn_block::RouterOperand::Precomputed(logits) => (logits.clone(), normed2),
            ffn_block::RouterOperand::BranchInput(x) => (layer.moe.router.apply(x), x.as_slice()),
        };
        Self::combine_ffn_outputs_for_position(
            layer_idx,
            layer,
            normed2,
            routed_input,
            &router_logits,
            config,
            hidden_dim,
            plan,
        )
    }
}
