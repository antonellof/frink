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
        pos: usize,
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
        self.close_run(run, end, pos, hidden, kv_caches, pending)
    }

    /// The same run, begun with an ATTENTION layer's tail.
    ///
    /// A hybrid alternates three recurrent layers and one attention
    /// layer, and the attention layer's tail feeds the next three: its
    /// output is the residual stream they read, which the host never
    /// looks at. So it goes in their command buffer and shares their
    /// single wait, which is sixteen fewer a token on Bonsai.
    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fused_attention_tail_then_run(
        &self,
        l: usize,
        pos: usize,
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
        Self::record_fused_ffn(layer);
        self.run_layers(&mut run, l + 1, end, kv_caches)?;
        self.close_run(run, end, pos, hidden, kv_caches, pending)
    }

    /// Carries `run` on from layer `end` to the end of the stack where
    /// it can, then waits for it ONCE.
    ///
    /// Each attention layer [`Self::resident_attn_parts`] admits joins
    /// the run whole (`GdnRun::attn_layer_resident`), and the recurrent
    /// layers after it follow, so a hybrid token that qualifies end to
    /// end is one wait instead of one per attention layer (#540). Where
    /// the run stops short of the stack, the next attention layer's
    /// Q/K/V are encoded at its end as before and come back in
    /// `pending`. The rows each joined attention layer appended to its
    /// mirror are copied to the host cache after the wait, so the host
    /// stays the authority.
    #[cfg(feature = "metal")]
    pub(crate) fn close_run(
        &self,
        mut run: frink_metal::gdn_branch::GdnRun,
        mut end: usize,
        pos: usize,
        hidden: &mut Vec<f32>,
        kv_caches: &mut [frink_core::KvCache],
        pending: &mut PendingQkv,
    ) -> Option<usize> {
        let n_layers = kv_caches.len();
        let mut joined = Vec::new();
        while end < n_layers && self.decode_attn_layer_in_run(&mut run, end, pos, kv_caches) {
            joined.push(end);
            let mut next = end + 1;
            while next < n_layers && self.fused_layer_parts(next).is_some() {
                next += 1;
            }
            self.run_layers(&mut run, end + 1, next, kv_caches)?;
            end = next;
        }
        let head = self.encode_next_attn_head(&mut run, end, n_layers);
        let frink_metal::gdn_branch::RunOut {
            hidden: out,
            head: qkv,
            routed,
        } = run.finish_with_head().ok()?;
        // The experts each routed layer chose, recorded as the host
        // body records them.
        for (l, ids) in routed {
            self.layer_for(l).moe.record_activations(&ids);
        }
        for l in joined {
            let cache = &mut kv_caches[l];
            let prior = cache.rows();
            let (k, v) = cache.metal_attn.as_ref()?.tokens_host(prior, 1);
            cache
                .push(&k, &v)
                .expect("unbounded/planned KvCache growth is infallible");
        }
        *hidden = out;
        *pending = head.then_some(qkv).flatten().map(|q| (end, q));
        Some(end)
    }

    /// Attention layer `l` appended to `run` whole, at position `pos`,
    /// or `false` with nothing encoded and nothing advanced.
    #[cfg(feature = "metal")]
    fn decode_attn_layer_in_run(
        &self,
        run: &mut frink_metal::gdn_branch::GdnRun,
        l: usize,
        pos: usize,
        kv_caches: &mut [frink_core::KvCache],
    ) -> bool {
        let cache = &mut kv_caches[l];
        let Some(p) = self.resident_attn_parts(l, pos, cache) else {
            return false;
        };
        let mv = crate::metal_launch::matvec;
        let (Some(q), Some(k), Some(v), Some(o), Some(ffn)) =
            (mv(p.q), mv(p.k), mv(p.v), mv(p.o), p.ffn.launches())
        else {
            return false;
        };
        let head_dim = self.config.head_dim;
        let rows = cache.rows();
        let width = rows * p.n_kv_heads * head_dim;
        if Self::ensure_attn_mirror(cache, p.n_kv_heads, head_dim, rows, width, 1).is_none() {
            return false;
        }
        let Some(mirror) = cache.metal_attn.as_mut() else {
            return false;
        };
        let ffn = ffn.as_metal();
        let w = frink_metal::gdn_decode_attn::DecodeAttnLayer {
            attn_norm: p.attn_norm,
            eps: p.eps,
            fold_x: p.fold_x.as_ref(),
            q: &q,
            k: &k,
            v: &v,
            n_heads: p.n_heads,
            q_gate: p.q_gate,
            extras: &p.extras,
            rope: p.rope,
            layer_rope: p.layer_rope,
            o_proj: &o,
            fold_branch: p.fold_branch.as_ref(),
            ffn: &ffn,
        };
        // SAFETY: the mirror lives in this layer's cache element, which
        // nothing else touches until the run is waited for in
        // `close_run`.
        if unsafe { run.attn_layer_resident(&w, mirror, pos) }.is_err() {
            return false;
        }
        Self::record_fused_ffn(self.layer_for(l));
        true
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
            Self::record_fused_ffn(layer);
        }
        Some(())
    }

    /// The hotness counters for a layer a run carried, as the host body
    /// keeps them: expert 0 for a dense layer here, and a routed
    /// layer's chosen experts once the run hands them back
    /// ([`Self::close_run`]).
    #[cfg(feature = "metal")]
    pub(crate) fn record_fused_ffn(layer: &LayerWeights) {
        if layer.moe.n_experts() == 1 {
            layer.moe.record_activations_dense();
        }
    }

    /// Layer `l`'s FFN as a decode run takes it: the dense FFN, or a
    /// routed one with its single shared expert when the device router
    /// picks the experts the host would
    /// ([`Self::gpu_router_matches_host_routing`]) and the input is not
    /// weighted per expert. The ONE answer every run entry asks.
    #[cfg(feature = "metal")]
    pub(crate) fn run_ffn_parts<'a>(
        &self,
        l: usize,
        layer: &'a LayerWeights,
    ) -> Option<LayerFfnParts<'a>> {
        let eps = self.config.rms_norm_eps;
        if let Some(dense) = LayerFfnParts::for_layer(layer, eps, true) {
            return Some(dense);
        }
        if !Self::gpu_router_matches_host_routing(layer, &self.config)
            || self.config.moe.routed_weight_before_ffn
        {
            return None;
        }
        let routing = crate::fused_layer::Routing {
            top_k: self.config.moe.n_experts_active,
            renormalize: self.config.moe.norm_topk_prob,
            tag: l,
        };
        LayerFfnParts::for_routed_layer(layer, eps, true, routing)
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
        let ffn = self.run_ffn_parts(l, layer)?;
        Some((gdn, attn_norm, ffn))
    }
}
