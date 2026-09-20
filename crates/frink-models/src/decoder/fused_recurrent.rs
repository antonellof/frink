//! A recurrent decoder layer as ONE Metal submission, or nothing.
//!
//! [`crate::fused_layer`] says what the layer's WEIGHTS have to be.
//! This says what the MODEL has to be, and then runs it. The two halves
//! are separate because they fail for different reasons and a caller
//! has to answer both at one point, which is
//! [`Decoder::fused_recurrent_layer`].
//!
//! Why it exists at all is arithmetic, not a hunch: a Bonsai decode
//! token submits 192 command buffers and pays 0.15 ms of OS wake-up for
//! each, 35 ms of a 137 ms token, against GPU work that is already
//! faster than the reference's whole token. Three submissions a layer
//! become one here.

use super::{Decoder, LayerWeights};
use crate::fused_layer::LayerFfnParts;
use crate::layer_shapes::AttnShape;
use crate::ssm_block::SsmBlock;
use frink_core::recurrent_state::RecurrentState;

/// The Q/K/V a recurrent run encoded for the attention layer that
/// follows it, and which layer they belong to.
///
/// Carried one loop iteration by the decode body, so that layer does
/// not project them again in a submission of its own
/// (`frink_metal::gdn_branch::GdnRun::attn_head`).
pub(crate) type PendingQkv = Option<(usize, (Vec<f32>, Vec<f32>, Vec<f32>))>;

impl Decoder {
    /// Layer `l` end to end in one submission, or `None` when either
    /// half refuses and the host bodies run it.
    ///
    /// `normed` is `attn_norm(hidden)`, `hidden` the residual stream as
    /// it entered the layer. The returned vector is the stream as it
    /// leaves: the caller assigns it and moves to the next layer
    /// without an attention block or an FFN block of its own.
    #[cfg(feature = "metal")]
    pub(crate) fn fused_recurrent_layer(
        &self,
        l: usize,
        layer: &LayerWeights,
        normed: &[f32],
        hidden: &[f32],
        slot: &mut Option<RecurrentState>,
    ) -> Option<Vec<f32>> {
        // The MODEL's half. Each of these is applied by a host body and
        // by no kernel in the fused launch, so a model that has one
        // takes the host path -- not a launch that silently drops it.
        let shape = self.config.layer_shape(l);
        if !matches!(shape.attention, AttnShape::Gdn)
            // An FFN-free block keeps or discards its output by
            // architecture (`crate::layer_shapes`); neither is this.
            || shape.ffn_dim == 0
            // Granite's multipliers, applied to both branch outputs.
            || self.config.residual_scale.is_some()
            || self.config.normed_residual_scale.is_some()
            // Talkie's normed embedding added into every layer output.
            || self.config.skip_stream
            // gpt-oss's side table is a different FFN entirely.
            || self.gpt_oss.is_some()
            // Anything but SwiGLU has no kernel in this launch.
            || !self.config.layer_ffn_acts(l).all_swiglu()
        {
            return None;
        }
        let SsmBlock::Gdn(gdn) = layer.attn.ssm.as_ref()? else {
            return None;
        };
        let ffn = LayerFfnParts::for_layer(layer, self.config.rms_norm_eps, true)?;
        let state = slot.get_or_insert_with(|| gdn.zero_state());
        // The layer's own `attn_norm`, so the fused launch can compute
        // `normed` on device and the host copy above goes unused.
        let attn_norm = layer.attn.norm_weight.rms_weights()?;
        let out = gdn.fused_layer(
            attn_norm,
            normed,
            state,
            self.config.rms_norm_eps,
            &ffn,
            hidden,
        )?;
        // The host body records expert 0 for a dense layer every token,
        // so this does too, and only once the layer has actually run
        // here rather than fallen back.
        layer.moe.record_activations_dense();
        Some(out)
    }
}

impl Decoder {
    /// A RUN of consecutive recurrent layers from `l`, submitted back to
    /// back against one residual buffer and waited for ONCE. Returns
    /// the layer index the caller should resume at, or `None` when
    /// there is no run here worth making.
    ///
    /// # Why a run and not a layer
    ///
    /// With a layer down to one command buffer the ledger says a Bonsai
    /// decode token is GPU time plus the OS wake-up from
    /// `waitUntilCompleted`, 0.17 ms a submission and essentially no
    /// host time. Nothing requires the host to wait per layer:
    /// consecutive recurrent layers hand each other a residual stream it
    /// never looks at. One queue is ordered, so committing them back to
    /// back and waiting for the last waits for all of them.
    ///
    /// Qwen3.5 puts a full-attention layer every fourth, so the runs
    /// here are three layers long and three waits become one.
    #[cfg(feature = "metal")]
    pub(crate) fn fused_recurrent_run(
        &self,
        start: usize,
        hidden: &mut Vec<f32>,
        kv_caches: &mut [frink_core::KvCache],
        pending: &mut PendingQkv,
    ) -> Option<usize> {
        // How far the run reaches: consecutive layers this path serves
        // whole. Asked BEFORE anything is submitted, because a run that
        // discovered a refusal halfway would have to undo state the GPU
        // has already advanced.
        let mut end = start;
        while end < kv_caches.len() && self.fused_layer_parts(end).is_some() {
            end += 1;
        }
        // One layer is the single-submission path already, and it does
        // not have to hold its scratch across a wait.
        if end - start < 2 {
            return None;
        }
        let mut run = frink_metal::gdn_branch::GdnRun::start(hidden).ok()?;
        self.run_layers(&mut run, start, end, kv_caches)?;
        let head = self.encode_next_attn_head(&mut run, end, kv_caches.len());
        let (out, qkv) = run.finish_with_head().ok()?;
        *hidden = out;
        *pending = head.then_some(qkv).flatten().map(|q| (end, q));
        Some(end)
    }

    /// The same run, begun with an ATTENTION layer's tail.
    ///
    /// A hybrid alternates three recurrent layers and one attention
    /// layer, and the attention layer's tail feeds the next three: its
    /// output is the residual stream they read, which the host never
    /// looks at. So it goes in their command buffer and shares their
    /// single wait, which is sixteen fewer a token on Bonsai.
    #[cfg(feature = "metal")]
    pub(crate) fn fused_attention_tail_then_run(
        &self,
        l: usize,
        layer: &LayerWeights,
        branch: &[f32],
        hidden: &mut Vec<f32>,
        kv_caches: &mut [frink_core::KvCache],
        pending: &mut PendingQkv,
    ) -> Option<usize> {
        let (out_proj, fold_branch, ffn, launches) = self.attn_tail_launch(l, layer, branch)?;
        let mut end = l + 1;
        while end < kv_caches.len() && self.fused_layer_parts(end).is_some() {
            end += 1;
        }
        // With no recurrent layer behind it there is nothing to share a
        // wait with, and `launch_attn_tail` is the cheaper shape.
        if end == l + 1 {
            return None;
        }
        let mut run = frink_metal::gdn_branch::GdnRun::start(hidden).ok()?;
        run.attn_tail(
            &out_proj,
            fold_branch.as_ref(),
            &launches.as_metal(),
            branch,
        )
        .ok()?;
        let _ = ffn;
        layer.moe.record_activations_dense();
        self.run_layers(&mut run, l + 1, end, kv_caches)?;
        let head = self.encode_next_attn_head(&mut run, end, kv_caches.len());
        let (out, qkv) = run.finish_with_head().ok()?;
        *hidden = out;
        *pending = head.then_some(qkv).flatten().map(|q| (end, q));
        Some(end)
    }

    /// The NEXT layer's Q/K/V projections at the end of `run`, when
    /// that layer is an attention layer this path serves. `false` when
    /// there is nothing to encode, and then `finish` returns no head.
    #[cfg(feature = "metal")]
    pub(crate) fn encode_next_attn_head(
        &self,
        run: &mut frink_metal::gdn_branch::GdnRun,
        next: usize,
        n_layers: usize,
    ) -> bool {
        if next >= n_layers {
            return false;
        }
        let layer = self.layer_for(next);
        let Some((norm, q, k, v, fold_x)) = self.attn_head_launch(next, layer) else {
            return false;
        };
        run.attn_head(norm, self.config.rms_norm_eps, &q, &k, &v, fold_x.as_ref())
            .is_ok()
    }

    /// Appends layers `start..end` to `run`. The ONE place a recurrent
    /// layer joins a run, whatever began it.
    #[cfg(feature = "metal")]
    pub(crate) fn run_layers(
        &self,
        run: &mut frink_metal::gdn_branch::GdnRun,
        start: usize,
        end: usize,
        kv_caches: &mut [frink_core::KvCache],
    ) -> Option<()> {
        for (l, cache) in (start..end).zip(kv_caches[start..end].iter_mut()) {
            let layer = self.layer_for(l);
            let (gdn, attn_norm, ffn) = self.fused_layer_parts(l)?;
            let state = cache.recurrent.get_or_insert_with(|| gdn.zero_state());
            // SAFETY: each layer's state lives in its own cache element,
            // which nothing else touches until the run finishes; the
            // run holds the scratch its command buffers read.
            unsafe { gdn.run_layer(run, attn_norm, self.config.rms_norm_eps, &ffn, state) }?;
            cache
                .advance_len(1)
                .expect("unbounded/planned KvCache growth is infallible");
            layer.moe.record_activations_dense();
        }
        Some(())
    }

    /// Layer `l`'s block, `attn_norm` and dense FFN when every half of
    /// the fused path serves it. The ONE predicate both the single-layer
    /// and the run entry ask, so they cannot disagree about which layers
    /// are fusable.
    #[cfg(feature = "metal")]
    pub(crate) fn fused_layer_parts(
        &self,
        l: usize,
    ) -> Option<(&crate::gdn::Gdn, &[f32], LayerFfnParts<'_>)> {
        let layer = self.layer_for(l);
        let shape = self.config.layer_shape(l);
        if !matches!(shape.attention, AttnShape::Gdn)
            || shape.ffn_dim == 0
            || self.config.residual_scale.is_some()
            || self.config.normed_residual_scale.is_some()
            || self.config.skip_stream
            || self.gpt_oss.is_some()
            || !self.config.layer_ffn_acts(l).all_swiglu()
        {
            return None;
        }
        let SsmBlock::Gdn(gdn) = layer.attn.ssm.as_ref()? else {
            return None;
        };
        let attn_norm = layer.attn.norm_weight.rms_weights()?;
        let ffn = LayerFfnParts::for_layer(layer, self.config.rms_norm_eps, true)?;
        Some((gdn, attn_norm, ffn))
    }
}
