//! The fused Metal prefill stack for runs of dense (and resident-MoE)
//! layers: which layers a run may cover, and the launch that encodes
//! it.

use frink_core::cache::KvCache;
use frink_moe::GluAct;

use crate::config::ModelConfig;

use super::{Decoder, ExpertBacking, LayerWeights};

impl Decoder {
    /// Dense layer eligible for the one-CB `mul_mm_sg` prefill stack.
    /// QKV bias / QK-norm are applied on-GPU via [`AttnExtras`] (same as
    /// decode); SWA fit is checked separately.
    #[cfg(feature = "metal")]
    pub(super) fn metal_prefill_dense_layer_eligible(
        layer: &LayerWeights,
        config: &ModelConfig,
        lora_attached: bool,
    ) -> bool {
        Self::fused_prefill_dense_layer_eligible(layer, config, lora_attached)
    }

    #[cfg(feature = "metal")]
    pub(super) fn metal_prefill_dense_swa_fits(
        &self,
        layer_idx: usize,
        start_pos: usize,
        batch_size: usize,
    ) -> bool {
        match self.config.layer_sliding_window(layer_idx) {
            Some(window) => start_pos + batch_size <= window,
            None => true,
        }
    }

    /// Routed-expert FFN for the fused prefill stack, or `None` when this
    /// layer must keep the host-routed path (`launch_moe_prefill_q4_0`).
    ///
    /// Note: routing happens on the GPU here, so prefill no longer feeds
    /// `record_activations`. Expert hotness for `inspect-plan` comes from
    /// decode, which still routes on the host.
    #[cfg(feature = "metal")]
    pub(super) fn metal_prefill_moe<'a>(
        layer: &'a LayerWeights,
        config: &ModelConfig,
        lora_attached: bool,
    ) -> Option<frink_metal::gpu::PrefillMoeMetal<'a>> {
        if !Self::metal_can_serve_model(config, lora_attached)
            || Self::is_dense_layer(layer)
            || !layer.moe.shared_experts.is_empty()
            || !config.model_ffn_act().is_some_and(GluAct::is_swiglu)
            // See `gpu_router_matches_host_routing`: the GPU router
            // takes router weights and nothing else.
            || !Self::gpu_router_matches_host_routing(layer, config)
        {
            return None;
        }
        let frink_core::weight_matrix::WeightMatrix::F32(router) = &layer.moe.router else {
            return None;
        };
        let packed = Self::moe_packed_q4(&layer.moe)?;
        let moe = frink_metal::gpu::PrefillMoeMetal {
            router_w: &router.data,
            top_k: config.moe.n_experts_active,
            renormalize: config.moe.norm_topk_prob,
            packed,
        };
        moe.is_supported().then_some(moe)
    }

    /// FFN half of a fused-prefill-stack layer: dense `mul_mm_sg` launches
    /// or (MoE) the routed-expert description.
    #[cfg(feature = "metal")]
    pub(super) fn metal_prefill_ffn<'a>(
        layer: &'a LayerWeights,
        config: &ModelConfig,
        lora_attached: bool,
    ) -> Option<frink_metal::attn::PrefillFfnMetal<'a>> {
        if let Some(moe) = Self::metal_prefill_moe(layer, config, lora_attached) {
            return Some(frink_metal::attn::PrefillFfnMetal::Moe(moe));
        }
        if !Self::is_dense_layer(layer) {
            return None;
        }
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        let ex = experts.first()?;
        Some(frink_metal::attn::PrefillFfnMetal::Dense {
            gate: ex.gate.mul_mm_sg_launch()?,
            up: ex.up.mul_mm_sg_launch()?,
            down: ex.down.mul_mm_sg_launch()?,
        })
    }

    /// Length of a consecutive run of Metal prefill-stack layers from
    /// `start`, or `None` when fewer than two layers qualify.
    #[cfg(feature = "metal")]
    pub(super) fn metal_prefill_dense_stack_run_len(
        &self,
        start: usize,
        start_pos: usize,
        batch_size: usize,
        kv_caches: &[KvCache],
        metal_kvs: Option<&[frink_metal::attn::MetalKvBuffers]>,
    ) -> Option<usize> {
        // See `layer_supports_metal_attn`: gpt-oss stays on CPU.
        if self.gpt_oss.is_some() {
            return None;
        }
        let metal_kvs = metal_kvs?;
        let mut run = 0usize;
        for li in start..self.layers.len() {
            let layer = &self.layers[li];
            let cache = &kv_caches[li];
            if !self.metal_prefill_dense_swa_fits(li, start_pos, batch_size) {
                break;
            }
            // POSITIONS: compared against `start_pos`, and against
            // Metal's own count of the same sequence.
            if metal_kvs[li].seq_len != cache.positions() || start_pos != cache.positions() {
                break;
            }
            let ok = layer.attn.q_proj.mul_mm_sg_launch().is_some()
                && layer.attn.k_proj.mul_mm_sg_launch().is_some()
                && layer.attn.v_proj.mul_mm_sg_launch().is_some()
                && layer.attn.o_proj.mul_mm_sg_launch().is_some()
                && Self::metal_prefill_ffn(layer, &self.config, self.lora_attached()).is_some();
            if !ok {
                break;
            }
            run += 1;
        }
        (run >= 2).then_some(run)
    }

    /// Try [`frink_metal::attn::launch_prefill_dense_stack`] for
    /// `run_len` layers starting at `start`. Advances host + Metal KV
    /// on success.
    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_metal_prefill_dense_stack(
        &self,
        start: usize,
        run_len: usize,
        hidden_batch: &[f32],
        start_pos: usize,
        batch_size: usize,
        n_heads: usize,
        metal_kvs: &mut [frink_metal::attn::MetalKvBuffers],
        kv_caches: &mut [KvCache],
        host_kv_authoritative: bool,
    ) -> Option<Vec<f32>> {
        // `None` for an activation no fused kernel implements: the
        // stack does not launch, rather than running it as GELU.
        let gelu = self
            .config
            .model_ffn_act()
            .and_then(GluAct::fused_kernel_gelu_flag)?;
        let mut prefill_layers = Vec::with_capacity(run_len);
        for li in start..start + run_len {
            let layer = &self.layers[li];
            let ffn = Self::metal_prefill_ffn(layer, &self.config, self.lora_attached())?;
            if matches!(ffn, frink_metal::attn::PrefillFfnMetal::Dense { .. }) {
                layer.moe.record_activations(&[0]);
            }
            let (q, k, v, o) = (
                layer.attn.q_proj.mul_mm_sg_launch()?,
                layer.attn.k_proj.mul_mm_sg_launch()?,
                layer.attn.v_proj.mul_mm_sg_launch()?,
                layer.attn.o_proj.mul_mm_sg_launch()?,
            );
            prefill_layers.push(frink_metal::attn::PrefillDenseLayerMetal {
                // `?`, not a `&`: the kernel applies the RMSNorm itself
                // and the post-norm-only topology has no weight to give
                // it, so the stack declines and the host body runs. See
                // `crate::norm`.
                attn_norm_w: layer.attn.norm_weight.rms_weights()?,
                ffn_norm_w: layer.moe.norm_weight.rms_weights()?,
                q,
                k,
                v,
                o,
                ffn,
                post_attn_norm: layer.attn.post_attn_norm.as_deref(),
                post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
                extras: self.metal_attn_extras(layer),
                rope: self.metal_layer_rope(li),
                layer_idx: li as u32,
            });
        }
        let kvs = &mut metal_kvs[start..start + run_len];
        let h_out = frink_metal::attn::launch_prefill_dense_stack(
            hidden_batch,
            &prefill_layers,
            kvs,
            n_heads,
            batch_size,
            self.metal_rope(),
            start_pos,
            self.config.rms_norm_eps,
            gelu,
            self.config.attn_logit_softcap,
        )
        .ok()?;
        for (mkv, cache) in kvs.iter().zip(&mut kv_caches[start..start + run_len]) {
            Self::advance_host_kv_after_metal_prefill(
                mkv,
                cache,
                batch_size,
                host_kv_authoritative,
            );
        }
        Some(h_out)
    }
}
