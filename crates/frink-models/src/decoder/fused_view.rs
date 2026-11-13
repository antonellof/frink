//! What a fused GPU launch may take from one layer, decided ONCE for
//! every backend that has such launches.
//!
//! Metal has had fused dense stacks since the beginning and CUDA gained
//! its resident prefill layer on 2026-09-17 (#259). The decisions about
//! which layers they may serve -- which `AttnWeights` fields a fused
//! attention kernel can apply, which model-wide facts fence a model off
//! every fused launch, what a dense layer must not carry -- are the same
//! questions for both, so they are answered here and each backend maps
//! the answer onto its own launch types. Two copies of the exhaustive
//! destructure would be the drift this file exists to stop: a field
//! added to `AttnWeights` fails to compile here until it is named, and
//! then it is named for both backends at once.

use super::{AttnWeights, Decoder, LayerWeights};
use crate::config::ModelConfig;

/// The optional per-layer attention ops every fused launch applies
/// between the QKV projections and RoPE, backend-neutral. Metal wraps
/// it as `frink_metal::attn::AttnExtras`, CUDA as
/// `frink_cuda::prefill::AttnExtrasCuda`.
#[derive(Clone, Copy)]
pub(crate) struct FusedAttnExtras<'a> {
    pub q_bias: Option<&'a [f32]>,
    pub k_bias: Option<&'a [f32]>,
    pub v_bias: Option<&'a [f32]>,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// Only under caps that apply them ([`FusedAttnCaps`]); `None`
    /// otherwise, because a layer carrying one was refused. Read by the
    /// CUDA stack alone.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub sinks: Option<&'a [f32]>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub o_bias: Option<&'a [f32]>,
}

/// What a backend's fused attention applies beyond the extras every
/// backend takes: gpt-oss's attention sinks (one logit per head in the
/// softmax, with no value) and a bias on `wo`. CUDA's resident stack
/// applies both. Metal applies neither, so a layer carrying either
/// takes the host body there.
#[derive(Clone, Copy)]
pub(crate) struct FusedAttnCaps {
    pub sinks: bool,
    pub o_bias: bool,
}

impl FusedAttnCaps {
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    pub(crate) const METAL: Self = Self {
        sinks: false,
        o_bias: false,
    };
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) const CUDA: Self = Self {
        sinks: true,
        o_bias: true,
    };
    /// The Vulkan decode stack applies neither (`vulkan_stack`).
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub(crate) const VULKAN: Self = Self {
        sinks: false,
        o_bias: false,
    };
}

impl Decoder {
    /// What the fused attention launches can take from one layer's
    /// attention weights, or `None` when the layer carries something
    /// none of them applies.
    ///
    /// An EXHAUSTIVE destructure with no `..`, on purpose: every field
    /// of [`AttnWeights`] is named here and either handed out in
    /// [`FusedAttnExtras`], consumed by the launch some other way (the
    /// four projections, the pre-norm, the post-norms), or the reason
    /// for the `None`. A field added to `AttnWeights` therefore fails
    /// to compile until this function says which of the three it is.
    /// The output gate is one no launch serves, and the attention sinks
    /// and `wo`'s bias are served only where `caps` says so: all three
    /// sit between the softmax and the residual, which the kernels fuse
    /// with no host round-trip, so a launch that ignored one would
    /// answer differently from the host bodies for the same weights --
    /// the fifth and sixth things found written into the Metal stacks
    /// unconditionally, after the final norm, the rotation, the
    /// residual scale and the activation.
    pub(crate) fn fused_attn_extras<'a>(
        layer: &'a LayerWeights,
        caps: FusedAttnCaps,
    ) -> Option<FusedAttnExtras<'a>> {
        Self::attn_extras(&layer.attn, caps)
    }

    /// [`Self::fused_attn_extras`] over the attention weights alone.
    fn attn_extras(attn: &AttnWeights, caps: FusedAttnCaps) -> Option<FusedAttnExtras<'_>> {
        let AttnWeights {
            q_proj: _,
            k_proj: _,
            v_proj: _,
            o_proj: _,
            norm_weight: _,
            q_norm,
            k_norm,
            q_bias,
            k_bias,
            v_bias,
            post_attn_norm: _,
            post_ffn_norm: _,
            output_gate,
            sinks,
            attn_sub_norm,
            o_scale,
            o_bias,
            shortconv,
            ssm,
            q_gate_interleaved,
        } = attn;
        // No fused attention kernel gates, norms between the V sum and
        // `wo`, scales after it, or runs a convolution or a state space
        // in its place, and only the backends whose caps say so apply
        // sinks or a bias on `wo`; a layer with any other runs on the
        // host.
        if output_gate.is_some()
            || (sinks.is_some() && !caps.sinks)
            || attn_sub_norm.is_some()
            || o_scale.is_some()
            || (o_bias.is_some() && !caps.o_bias)
            || shortconv.is_some()
            || ssm.is_some()
            || *q_gate_interleaved
        {
            return None;
        }
        Some(FusedAttnExtras {
            q_bias: q_bias.as_deref(),
            k_bias: k_bias.as_deref(),
            v_bias: v_bias.as_deref(),
            q_norm: q_norm.as_deref(),
            k_norm: k_norm.as_deref(),
            sinks: sinks.as_deref(),
            o_bias: o_bias.as_deref(),
        })
    }

    /// The backend-neutral half of "may a fused attention launch take
    /// this layer": the model-wide fence, the gpt-oss family, the
    /// `AttnWeights` fields, the RoPE layout and width, the QK-norm
    /// shape and order, `attention_scale`, the head width. Each backend
    /// adds the one question only it can answer -- whether its GEMM
    /// serves every projection -- on top of this.
    pub(crate) fn layer_supports_fused_attn(
        &self,
        layer: &LayerWeights,
        caps: FusedAttnCaps,
    ) -> bool {
        use crate::config::RopeLayout;
        if !Self::metal_can_serve_model(&self.config, self.lora_attached()) {
            return false;
        }
        // gpt-oss: a backend without the sinks and `wo`'s bias would
        // compute a *different* attention than the CPU path for the same
        // weights, so the family stays off it (Metal) rather than letting
        // the two disagree. The tensors are refused one line down too;
        // the name is the fence for whatever else the family carries.
        // Its FFN (the biased router, expert biases, the clamped SwiGLU)
        // is the FFN eligibility's question, not this one. See
        // `Decoder::gpt_oss`.
        if self.gpt_oss.is_some() && !(caps.sinks && caps.o_bias) {
            return false;
        }
        if Self::fused_attn_extras(layer, caps).is_none() {
            return false;
        }
        if !matches!(self.config.rope_layout, RopeLayout::Norm | RopeLayout::Neox) {
            return false;
        }
        // QKV bias (Qwen2) and QK-norm -- per-head (Qwen3/Gemma-3) or
        // whole-vector (OLMoE) -- run on the device through the extras.
        // NOT the per-head norm with a DISTINCT row per head (PLaMo-2):
        // its weight is exactly as long as a whole-vector one, and every
        // fused kernel would read it as one norm over the projection.
        if self.config.qk_norm_style == crate::capability::QkNormStyle::PerHeadDistinct {
            return false;
        }
        let q_len = self.config.n_heads * self.config.head_dim;
        let k_len = self.config.n_kv_heads * self.config.head_dim;
        let qk_norm_ok = |w: Option<&Vec<f32>>, vec_len: usize| -> bool {
            match w {
                None => true,
                Some(w) if w.len() == self.config.head_dim => true,
                Some(w) if w.len() == vec_len => true,
                _ => false,
            }
        };
        if !qk_norm_ok(layer.attn.q_norm.as_ref(), q_len)
            || !qk_norm_ok(layer.attn.k_norm.as_ref(), k_len)
        {
            return false;
        }
        // NOT the QK-norm ORDER. The extras hand the norm weights to
        // kernels that apply them before their own RoPE, so a
        // `maincoder` / `hunyuan-moe` layer would be normed on the wrong
        // side of the rotation by every fused launch while the host
        // bodies got it right -- the same weights answering differently
        // depending on which backend served the token. Same fence, same
        // reason, as `attention_scale` below.
        if self.qk_norm_after_rope {
            return false;
        }
        // Softcaps: final logit softcap is applied on the host after
        // lm_head. Attention softcap runs inside every fused attention
        // kernel (decode + prefill, both backends).
        //
        // NOT attention_scale, and this is a refusal rather than a
        // comment. The extras have no field for it and no fused kernel
        // applies it, so a checkpoint carrying one would be scaled by
        // the host bodies and not by any fused launch -- the same
        // weights answering at two different temperatures depending on
        // which backend served the token.
        //
        // LIVE, not latent: `capability::attention_scale_override` sets
        // it for Gemma-2-27B and Gemma-3-27B, so those two checkpoints
        // take the host path here and are scaled exactly once.
        if self.config.attention_scale.is_some() {
            return false;
        }
        if self.config.head_dim > 256 {
            return false;
        }
        // Partial rotary (`n_rot < head_dim`) and LongRoPE's `mscale`
        // ride the RoPE kernels as their `rot_dim` / `mscale` inputs, so
        // Phi-3/Phi-4 are admitted here. `n_rot` must still be even --
        // ggml's `ggml_rope_impl` asserts it, and an odd width would
        // leave one channel's pairing undefined rather than merely
        // unrotated.
        !self
            .config
            .rope_dim
            .is_some_and(|rot| rot == 0 || rot % 2 != 0 || rot > self.config.head_dim)
    }

    /// A dense layer a fused prefill stack may take, on either backend:
    /// one expert and no shared experts, nothing scaled after `down`
    /// (`crate::weight_scales`) or added at any FFN site
    /// (`crate::proj_bias`), on a model the fence admits.
    pub(crate) fn fused_prefill_dense_layer_eligible(
        layer: &LayerWeights,
        config: &ModelConfig,
        lora_attached: bool,
    ) -> bool {
        Self::is_dense_layer(layer)
            && layer.moe.down_scale.is_none()
            && layer.moe.dense_bias.is_none()
            && Self::metal_can_serve_model(config, lora_attached)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::norm::NormOp;
    use frink_core::{Tensor, WeightMatrix};

    fn attn(sinks: bool, o_bias: bool) -> AttnWeights {
        let m = |r: usize, c: usize| WeightMatrix::F32(Tensor::new(vec![0.0; r * c], vec![r, c]));
        AttnWeights {
            q_proj: m(8, 8),
            k_proj: m(8, 8),
            v_proj: m(8, 8),
            o_proj: m(8, 8),
            norm_weight: NormOp::None,
            q_norm: None,
            k_norm: None,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            post_attn_norm: None,
            post_ffn_norm: None,
            output_gate: None,
            sinks: sinks.then(|| vec![0.5; 2]),
            attn_sub_norm: None,
            o_scale: None,
            o_bias: o_bias.then(|| vec![0.25; 8]),
            shortconv: None,
            ssm: None,
            q_gate_interleaved: false,
        }
    }

    /// gpt-oss's sinks and `wo` bias reach a backend whose caps apply
    /// them (CUDA), and refuse the layer on one whose caps do not
    /// (Metal), each of the two on its own.
    #[test]
    fn sinks_and_the_o_bias_reach_only_a_backend_that_applies_them() {
        let gpt_oss = attn(true, true);
        assert!(Decoder::attn_extras(&gpt_oss, FusedAttnCaps::METAL).is_none());
        let cuda = Decoder::attn_extras(&gpt_oss, FusedAttnCaps::CUDA).expect("CUDA applies both");
        assert_eq!(cuda.sinks, Some(&[0.5f32; 2][..]));
        assert_eq!(cuda.o_bias, Some(&[0.25f32; 8][..]));
        for (sinks, o_bias) in [(true, false), (false, true)] {
            let one = attn(sinks, o_bias);
            assert!(Decoder::attn_extras(&one, FusedAttnCaps::METAL).is_none());
            assert!(Decoder::attn_extras(&one, FusedAttnCaps::CUDA).is_some());
        }
        let plain = attn(false, false);
        let extras = Decoder::attn_extras(&plain, FusedAttnCaps::METAL).unwrap();
        assert!(extras.sinks.is_none() && extras.o_bias.is_none());
    }
}
