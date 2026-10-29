//! Which fused GPU launches may serve this model and each of its
//! layers: the model-wide predicate every fused stack shares, the per-
//! layer attention / FFN / MoE checks, and the views and RoPE
//! descriptions those launches are handed.

#[cfg(feature = "metal")]
use frink_core::weight_matrix::WeightMatrix;
#[cfg(any(feature = "metal", feature = "cuda"))]
use frink_moe::GluAct;

use crate::config::ModelConfig;

#[cfg(feature = "metal")]
use super::fused_view;
use super::{Decoder, LayerWeights};
#[cfg(feature = "metal")]
use super::{ExpertBacking, MoePackedQ4Planes, MoeWeights};

/// Whether the fused Metal attention block is switched on in this
/// process: dense Metal AND `FRINK_METAL_ATTN`. ONE spelling for every
/// site that asks. `KvWindowPolicy::from_env` once read the second half
/// alone, so `FRINK_METAL=0 FRINK_METAL_ATTN=1` (or a host-pinned test)
/// turned the host's KV window eviction off for a device path that
/// could not run.
#[cfg(feature = "metal")]
pub(crate) fn metal_attn_switched_on() -> bool {
    frink_core::metal_dense_enabled() && frink_metal::attn::metal_attn_enabled()
}

impl Decoder {
    /// Metal's launch description for `m`, or `None` when no kernel
    /// serves its storage. Delegates to `crate::metal_launch`, which is
    /// where both this file and `crate::gdn` read it from.
    #[cfg(feature = "metal")]
    pub(super) fn metal_matvec_launch<'a>(
        m: &'a WeightMatrix,
    ) -> Option<frink_metal::gpu::MatvecLaunch<'a>> {
        crate::metal_launch::matvec(m)
    }

    /// The per-model facts no fused Metal kernel implements, as ONE
    /// predicate the four Metal eligibility checks share.
    ///
    /// Five today. `residual_scale`: every fused launch that folds a
    /// residual add in -- the dense decode stack, the resident MoE
    /// decode stack, and both prefill stacks -- adds the branch output
    /// to the stream on device, with no uniform for a multiplier, so a
    /// Granite layer served by any of them would be scaled by the host
    /// bodies and not by the GPU. `clamp_kqv`: the fused launches apply
    /// the QKV bias inside their kernels (`AttnExtras`) and clamp
    /// nothing, so a DBRX or clamped-OLMo layer served by any of them
    /// would run unclamped projections while the host bodies clamp
    /// (`decoder/qkv_bias.rs`). An FFN activation the kernels cannot
    /// spell: every fused dense launch takes a `gelu: bool`, so the
    /// ungated ReLU-squared FFN (`GluAct::Reglu`, arcee) has no value to
    /// pass and every site that asks `fused_kernel_gelu_flag` gets
    /// `None` -- this predicate keeps the whole model off the stacks so
    /// that a layer is never half-served. And per-layer shapes: the
    /// stacks size Q/K/V and the Metal KV plane from ONE head count
    /// (`n_heads` is a launch argument, `MetalKvBuffers` one geometry),
    /// so a model whose layers disagree (`crate::layer_shapes`, deci /
    /// openelm) stays on the host bodies, which read each layer's own.
    /// And the per-position attention temperature
    /// (`crate::attn_temperature`): no fused launch takes a per-token Q
    /// scale, so a Ministral-3 layer served by any of them would attend
    /// at temperature 1 while the host bodies step it with position.
    /// Either way it is the same weights answering differently
    /// depending on which backend took the token, the exact failure
    /// `attention_scale` is fenced off for next door.
    ///
    /// It is one function rather than four spellings because the GPU
    /// router's eligibility check has already drifted four ways in this
    /// file -- prefill tested three conditions, fused decode two, and
    /// the whole-stack decode NONE.
    ///
    /// `lora_attached` is the fifth fact, and the first that is not a
    /// property of the file: a LoRA delta lives inside a
    /// `WeightMatrix::Adapted` and is served by that type's methods,
    /// while every fused stack takes its weights as raw bytes through
    /// `metal_matvec_launch` / `mul_mm_sg_launch` (both answer `None`
    /// for an adapted matrix). Fencing the whole model here, too, is
    /// what keeps a model with an adapter on one layer from being served
    /// half by a stack and half by the host.
    ///
    /// Named for the backend that had fused stacks first. The CUDA
    /// resident prefill layer (`decoder/cuda_prefill.rs`, #259) asks the
    /// same question through `fused_view`, because the facts are
    /// properties of the model and not of the kernel language.
    #[cfg(any(feature = "metal", feature = "cuda"))]
    pub(super) fn metal_can_serve_model(config: &ModelConfig, lora_attached: bool) -> bool {
        !lora_attached
            && config.residual_scale.is_none()
            // `minimax-01.cpp:249,428`: the residual a fused kernel
            // adds its branch to is the PRE-NORM's output, which never
            // leaves the device in any of these stacks
            // (`crate::normed_residual`).
            && config.normed_residual_scale.is_none()
            && config.clamp_kqv.is_none()
            && config.attn_temperature.is_none()
            // BitNet's two inner norms (`crate::sub_norms`): no fused
            // kernel norms between attention and `wo`, or between the
            // activation and `down`.
            && !config.block_sub_norms
            // Every fused launch bakes the pre-FFN norm over the
            // POST-ATTENTION residual into its kernel; a parallel layer
            // norms the layer INPUT (`crate::parallel_residual`).
            && !config.parallel_residual
            // The GPU embedding gather has no add and no stack sees
            // `pos` for it (`crate::position_embd`).
            && !config.learned_positions
            // Every fused launch runs attention alone on its layer; a
            // parallel Mamba-2 block (`crate::mamba2`) has no kernel.
            && !config.parallel_ssm
            // Every fused launch takes ONE sliding window per layer; a
            // chunked layer's window is per query
            // (`crate::chunked_swa`), and no fused kernel norms Q and K
            // without a weight after RoPE (`crate::weightless_qk_norm`).
            && !config.swa_chunked
            && !config.weightless_qk_norm
            // No fused kernel adds a per-key bias (`crate::alibi`).
            && config.alibi_max_bias.is_none()
            // Every fused launch takes ONE epsilon for the whole layer;
            // `muse-glimmer.cpp:63` runs its post-norms at a literal
            // 1e-8 and its pre-norms at the model's
            // (`crate::norm::POST_NORM_EPS_LITERAL`), so a model whose
            // two epsilons differ stays on the host rather than having
            // its post-norms silently run at the wrong one.
            && config.post_norm_eps() == config.rms_norm_eps
            // Every fused launch takes ONE head width for K and V (the
            // KV buffers, the attention tile, the `wo` fold); MiMo-V2's
            // split widths stay on the host (`crate::kv_head_dims`).
            && !config.kv_head_dims_split()
            // No fused kernel scales the branch after its `wo` fold
            // (`crate::attn_value_scale`).
            && config.attn_value_scale.is_none()
            // Every fused launch indexes `layers[l]` and `metal_kvs[l]`
            // with ONE `l`; a looped model's logical layers outnumber
            // its weights (`crate::layer_loops`).
            && config.layer_loops.is_none()
            // Talkie: an embedding norm no gather kernel applies, a
            // second residual no stack carries, a per-head SCALAR Q gain
            // and a weightless K norm no `AttnExtras` spells
            // (`crate::skip_stream`, `QkNormStyle::PerHeadScalar`).
            && !config.skip_stream
            && config.qk_norm_style != crate::capability::QkNormStyle::PerHeadScalar
            // `model_ffn_act` is `None` for an activation that varies
            // by layer (xIELU's parameters), which no fused kernel
            // takes; `fused_kernel_gelu_flag` is `None` for one no
            // kernel spells.
            && config
                .model_ffn_act()
                .and_then(GluAct::fused_kernel_gelu_flag)
                .is_some()
            && config.layer_shapes.is_uniform()
            // Each fused launch takes ONE rotary width (`MetalRope::
            // rot_dim`); a model whose sliding layers rotate a
            // different width from its full ones (`rope_dim_swa`,
            // Step-3.5 and Laguna-XS.2) stays on the host bodies, which
            // read each layer's own through `layer_rope`.
            && !config.rope_dim_varies_by_layer()
    }

    /// True when this layer can use the fused Metal attention block
    /// (Norm or NeoX RoPE, quantized projections; QKV bias + QK-norm
    /// via [`frink_metal::attn::AttnExtras`]).
    #[cfg(feature = "metal")]
    pub(super) fn layer_supports_metal_attn(&self, layer: &LayerWeights) -> bool {
        // The backend-neutral questions are `fused_view`'s; what is
        // left is whether Metal has a launch for every projection.
        self.layer_supports_fused_attn(layer)
            && Self::metal_matvec_launch(&layer.attn.q_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.k_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.v_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.o_proj).is_some()
    }

    /// Layer features only the fused dense stack implements — the
    /// per-layer Metal launches would silently skip them (wrong output).
    #[cfg(feature = "metal")]
    pub(super) fn layer_needs_metal_stack(&self, layer: &LayerWeights, layer_idx: usize) -> bool {
        layer.attn.post_attn_norm.is_some()
            || layer.attn.post_ffn_norm.is_some()
            || self.config.layer_sliding_window(layer_idx).is_some()
            || !self.config.layer_ffn_acts(layer_idx).all_swiglu()
            // `Some(base)` when this layer rotates at the model's own
            // base, so this ONE comparison covers both stack-only RoPE
            // facts: a per-layer base (Gemma-3's sliding layers) and a
            // layer that does not rotate at all (`crate::rope_layers`,
            // EXAONE-4 32B's full-attention layers). A second predicate
            // beside it is exactly the drift this file keeps paying for.
            || self.config.layer_rope_theta(layer_idx) != Some(self.config.rope_theta)
    }

    /// BOTH halves of layer `il`'s RoPE, in the shape the Metal
    /// launches take.
    ///
    /// One conversion, every Metal call site, because
    /// `ModelConfig::layer_rope` returning a pair and the launches
    /// taking a pair is only worth anything while nothing in between
    /// gets to take one half and default the other. That is exactly
    /// what the fused stacks used to do: a per-layer `rope_theta` beside
    /// ONE `freq_factors` slice for the whole run, which refused
    /// Gemma-3 4B/12B/27B off the fused path entirely rather than rope
    /// five layers in six at the wrong scale.
    #[cfg(feature = "metal")]
    pub(super) fn metal_layer_rope(
        &self,
        layer_idx: usize,
    ) -> Option<frink_metal::attn::LayerRope<'_>> {
        // Exhaustive on purpose: the third field is the rotary width,
        // which the Metal kernels take as ONE uniform for every layer
        // (`MetalRope::rot_dim`, from `metal_rope`). A layer whose width
        // differs from the model's must never reach a launch, and
        // `metal_can_serve_model` refuses such a model up front; this
        // is the check that the fence held, not a second fence.
        let crate::config::LayerRopeParams {
            theta,
            freq_factors,
            rot_dim,
        } = self.config.layer_rope(layer_idx)?;
        assert_eq!(
            rot_dim,
            self.config.rope_dim.filter(|w| *w < self.config.head_dim),
            "layer {layer_idx} rotates a different width from the model's; \
             `metal_can_serve_model` must have refused this model"
        );
        Some(frink_metal::attn::LayerRope {
            theta,
            freq_factors,
        })
    }

    /// [`Self::fused_attn_extras`] -- the ONE exhaustive destructure of
    /// `AttnWeights`, in `fused_view` -- in Metal's spelling.
    #[cfg(feature = "metal")]
    pub(super) fn metal_attn_view<'a>(
        &self,
        layer: &'a LayerWeights,
    ) -> Option<frink_metal::attn::AttnExtras<'a>> {
        let fused_view::FusedAttnExtras {
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
        } = Self::fused_attn_extras(layer)?;
        Some(frink_metal::attn::AttnExtras {
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
            attn_logit_softcap: self.config.attn_logit_softcap,
        })
    }

    /// Optional QKV bias / QK-norm ops for the Metal attn paths, for a
    /// layer [`Self::layer_supports_metal_attn`] has admitted. The
    /// `expect` can only fire when a launch site runs without asking
    /// that predicate, which is the drift this file's eligibility
    /// checks exist to stop, and a panic there beats a silent answer
    /// without the gate or the sinks.
    #[cfg(feature = "metal")]
    pub(super) fn metal_attn_extras<'a>(
        &self,
        layer: &'a LayerWeights,
    ) -> frink_metal::attn::AttnExtras<'a> {
        self.metal_attn_view(layer)
            .expect("layer_supports_metal_attn admits this layer, so its weights have a Metal view")
    }

    /// GPU expert residency only when Metal attention stays on-device
    /// (when the Metal dense+attn path is active). Avoids CPU-attention
    /// ↔ GPU-expert activation ping-pong on Metal MoE.
    #[cfg(feature = "metal")]
    pub(super) fn expert_residency_plan(
        &self,
        use_metal_attn: bool,
    ) -> Option<frink_moe::ResidencyPlan> {
        if metal_attn_switched_on() && !use_metal_attn {
            return None;
        }
        self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b))
    }

    #[cfg(not(feature = "metal"))]
    pub(super) fn expert_residency_plan(
        &self,
        _use_metal_attn: bool,
    ) -> Option<frink_moe::ResidencyPlan> {
        self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b))
    }

    /// Map this checkpoint's RoPE onto the Metal kernel uniforms:
    /// pairing convention, rotary width (`n_rot`), and ggml `rope_yarn`'s
    /// `mscale`. The last two are what
    /// [`Decoder::apply_rope_head_theta`] and
    /// [`Decoder::apply_rope_attn_factor`] do on the CPU side, so the
    /// two backends stay one graph.
    #[cfg(feature = "metal")]
    pub(super) fn metal_rope(&self) -> frink_metal::attn::MetalRope {
        use crate::config::RopeLayout;
        let layout = match self.config.rope_layout {
            RopeLayout::Norm => frink_metal::attn::MetalRopeLayout::Norm,
            RopeLayout::Neox => frink_metal::attn::MetalRopeLayout::Neox,
        };
        frink_metal::attn::MetalRope {
            layout,
            rot_dim: self
                .config
                .rope_dim
                .filter(|rot| *rot < self.config.head_dim),
            attn_factor: self.config.rope_attn_factor,
        }
    }

    /// Dense FFN (single expert) with Metal-capable gate/up/down.
    #[cfg(feature = "metal")]
    pub(super) fn layer_supports_metal_dense_ffn(layer: &LayerWeights) -> bool {
        Self::is_dense_layer(layer)
            // No fused FFN kernel scales after `down` (`crate::weight_scales`)
            // or adds a bias anywhere (`crate::proj_bias`).
            && layer.moe.down_scale.is_none()
            && layer.moe.dense_bias.is_none()
            && layer.moe.with_expert(0, |ex| {
                Self::metal_matvec_launch(&ex.gate).is_some()
                    && Self::metal_matvec_launch(&ex.up).is_some()
                    && Self::metal_matvec_launch(&ex.down).is_some()
            })
    }

    /// True when a plain top-k softmax over the raw router logits picks
    /// the SAME experts with the SAME weights that
    /// [`Self::route_for_layer`] would.
    ///
    /// Every Metal MoE path either routes on the GPU (which implements
    /// exactly that plain top-k and takes no other input) or, in
    /// `launch_moe_decode_pre`'s case, used to re-implement it on the
    /// host. `route_for_layer` has three arms this does not: grouped
    /// routing, a per-expert router bias (`exp_probs_bias`), and
    /// `expert_weights_scale`. A checkpoint carrying any of them routes
    /// to DIFFERENT experts with DIFFERENT weights depending on which
    /// backend served the token -- not an error, a different model.
    ///
    /// One predicate rather than the four hand-copied `.is_none()` lists
    /// this used to be, because those lists had already drifted three
    /// ways: the prefill sites checked all three conditions, the fused
    /// decode layer checked two of them, and the whole-stack decode
    /// checked none. Mirror `route_for_layer` arm for arm when either
    /// changes.
    // Read by the three Metal MoE eligibility predicates, and by
    // `the_gpu_router_predicate_admits_only_routing_it_reproduces`. A
    // CPU-only build has no Metal path to gate, so it is dead there.
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    pub(super) fn gpu_router_matches_host_routing(
        layer: &LayerWeights,
        config: &ModelConfig,
    ) -> bool {
        matches!(config.moe.gating, frink_moe::GatingFunction::Softmax)
            // Every GPU router reads `normed2`; a model whose router
            // reads the raw layer input (`crate::router_input`) would
            // route on the wrong tensor at full speed.
            && config.router_input == crate::router_input::RouterInput::NormedFfnInput
            // Conservative on purpose: `route_for_layer` only takes its
            // grouped arm for `n_groups > 1`, but a checkpoint that
            // declares the key at all is one this kernel was never
            // checked against.
            && config.moe.expert_group_count.is_none()
            && layer.moe.exp_probs_bias.is_none()
            && config.moe.expert_weights_scale == 1.0
    }

    /// MoE layer eligible for resident Metal decode (attn+router+experts
    /// without host residual ping-pong). Requires SwiGLU, no shared
    /// experts, Resident expert backing, Metal router/QKV/O, and a
    /// routing decision the GPU router reproduces exactly.
    #[cfg(feature = "metal")]
    pub(super) fn layer_supports_metal_moe_resident(
        layer: &LayerWeights,
        config: &ModelConfig,
        lora_attached: bool,
    ) -> bool {
        Self::metal_can_serve_model(config, lora_attached)
            && !Self::is_dense_layer(layer)
            && layer.moe.shared_experts.is_empty()
            && Self::gpu_router_matches_host_routing(layer, config)
            && config.model_ffn_act().is_some_and(GluAct::is_swiglu)
            // Streamed experts are eligible too. They were excluded
            // while the fused launch could not hold all of top-k at
            // once; it can now, by materialising each expert into an
            // owned view that carries its own pin on the store entry.
            && matches!(
                layer.moe.experts,
                ExpertBacking::Resident(_) | ExpertBacking::Stored { .. }
            )
            && Self::metal_matvec_launch(&layer.moe.router).is_some()
            && Self::metal_matvec_launch(&layer.attn.q_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.k_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.v_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.o_proj).is_some()
    }

    /// Contiguous Q4_0 expert planes for llama-style `mul_mv_id` MoE.
    #[cfg(feature = "metal")]
    pub(super) fn moe_packed_q4(moe: &MoeWeights) -> Option<frink_metal::gpu::MoePackedQ4<'_>> {
        moe.packed_q4.as_ref().map(MoePackedQ4Planes::view)
    }

    /// [`Self::moe_packed_q4`], only when every plane has a `mul_mv_id`
    /// kernel. The one fence the host-routed prefill and both fused
    /// decode launches take: a layer packed in a kind with no id kernel
    /// runs one matvec per expert instead (#448). The fused prefill
    /// stack fences on `mul_mm_id` itself (`PrefillMoeMetal::is_supported`).
    #[cfg(feature = "metal")]
    pub(super) fn moe_packed_id(moe: &MoeWeights) -> Option<frink_metal::gpu::MoePackedQ4<'_>> {
        Self::moe_packed_q4(moe).filter(|p| {
            frink_metal::gpu::moe_packed_mul_mv_id_supported(p.gate_kind, p.up_kind, p.down_kind)
        })
    }
}
