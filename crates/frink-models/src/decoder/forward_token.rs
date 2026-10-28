//! The contiguous-KV single-token decode body,
//! `Decoder::forward_token_on_worker`.
//!
//! One function, kept whole: its Metal arms share locals with the host
//! layer loop, and splitting it would be a rewrite rather than a move.

use crate::skip_stream::SkipStream;
use frink_core::cache::KvCache;
#[cfg(feature = "metal")]
use frink_moe::GluAct;

#[cfg(feature = "metal")]
use crate::norm::NormOp;
use crate::scalar_multipliers::residual_add;

use super::attn_block;
#[cfg(feature = "cuda")]
use super::cuda_gqa_enabled;
#[cfg(feature = "metal")]
use super::lm_head::FoldedLmHead;
#[cfg(feature = "metal")]
use super::ExpertBacking;
use super::{Decoder, KvStep};

impl Decoder {
    /// The body of [`Self::forward_token`], already running on a
    /// CPU-pool worker. See `entry.rs` for why the split exists.
    pub(super) fn forward_token_on_worker(
        &self,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<f32> {
        // Clear stale dense-stack activation TLS. MoE scratch buffers are
        // reused across tokens (re-seeded); cleared after lm_head below.
        #[cfg(feature = "metal")]
        frink_metal::gpu::clear_resident_activation();

        assert_eq!(kv_caches.len(), self.config.n_layers);
        // Read only by the Metal arms below: the host layer body moved
        // into `attn_block` / `ffn_block_row`, which read the geometry
        // off `self.config` themselves.
        #[cfg(feature = "metal")]
        let hidden_dim = self.config.hidden_dim;
        #[cfg(feature = "metal")]
        let head_dim = self.config.head_dim;
        #[cfg(feature = "metal")]
        let n_heads = self.config.n_heads;
        #[cfg(feature = "metal")]
        let n_kv_heads = self.config.n_kv_heads;

        #[cfg(feature = "metal")]
        let metal_embd_kind = {
            let metal_path = frink_core::metal_dense_enabled()
                && frink_metal::attn::metal_attn_enabled()
                && self
                    .layers
                    .iter()
                    .all(|l| self.layer_supports_metal_attn(l))
                && self.layers.iter().all(Self::layer_supports_metal_dense_ffn);
            // Gemma scales the embedding row (`embedding_scale`) — the GPU
            // gather has no scale op, so dequant + scale on the host; nor
            // has it a norm (`embedding_norm`) or a position table (the
            // latter already fenced through `metal_can_serve_model`).
            if metal_path
                && self.config.embedding_scale.is_none()
                && matches!(self.embedding_norm, NormOp::None)
            {
                Self::metal_matvec_launch(&self.embedding)
                    .and_then(|l| frink_metal::embd::EmbdKind::from_fn_name(l.fn_name))
            } else {
                None
            }
        };
        // `metal_embd_kind` is only `Some` when `embedding_scale` is
        // `None` (the GPU gather has no scale op), so the empty vector
        // this leaves behind is one the scale would not have touched.
        #[cfg(feature = "metal")]
        let mut hidden = if metal_embd_kind.is_some() {
            Vec::new()
        } else {
            self.embed_token(token_id, pos)
        };
        #[cfg(not(feature = "metal"))]
        let mut hidden = self.embed_token(token_id, pos);
        #[cfg(feature = "cuda")]
        if cuda_gqa_enabled()
            && self.config.layer_shapes.is_uniform()
            && self.alibi_slopes.is_none()
        {
            // Fixed capacity so ensure_layer_kv does not recreate (and
            // wipe) mid-sequence as pos grows. ONE geometry for every
            // layer, which is why a per-layer-shape model never seeds it
            // (`gqa_attention` skips the resident hook on the same
            // predicate).
            const CUDA_KV_CAP: usize = 4096;
            if let Err(e) = frink_cuda::attn::ensure_layer_kv(
                self.layers.len(),
                self.config.n_kv_heads,
                self.config.head_dim,
                CUDA_KV_CAP,
            ) {
                eprintln!("frink: CUDA KV residency init failed: {e}");
            }
            if pos == 0 {
                frink_cuda::attn::clear_layer_kv();
            }
        }

        #[cfg(feature = "metal")]
        let use_metal_attn = frink_core::metal_dense_enabled()
            && frink_metal::attn::metal_attn_enabled()
            && self
                .layers
                .iter()
                .all(|l| self.layer_supports_metal_attn(l));

        #[cfg(not(feature = "metal"))]
        let use_metal_attn = false;

        let residency = self.expert_residency_plan(use_metal_attn);

        #[cfg(feature = "metal")]
        let mut metal_kv_guard: Option<
            std::sync::MutexGuard<'_, Option<Vec<frink_metal::attn::MetalKvBuffers>>>,
        > = if use_metal_attn {
            Some(Self::lock_metal_attn_kv(&self.metal_attn_kv))
        } else {
            None
        };

        #[cfg(feature = "metal")]
        if let Some(guard) = metal_kv_guard.as_mut() {
            let need = self.layers.len();
            let cap = kv_caches
                .iter()
                // POSITIONS: sized against `pos`, which is a position.
                .map(|c| c.positions().max(pos + 1).saturating_add(256))
                .max()
                .unwrap_or(512)
                .max(512)
                .max(pos + 1);
            let reset = match guard.as_ref() {
                None => true,
                Some(v) => {
                    if v.len() != need || v.iter().any(|m| m.capacity() < pos + 1) {
                        // Growing / reshaping: preserve Metal-ahead tokens on host first.
                        if v.len() == need {
                            for (m, c) in v.iter().zip(kv_caches.iter_mut()) {
                                Self::catch_up_host_kv_from_metal(m, c);
                            }
                        }
                        true
                    } else if v.iter().all(|m| m.seq_len == pos) {
                        // Metal already holds tokens [0, pos). Host may lag
                        // after dense-stack decode — do not re-upload from host.
                        false
                    } else {
                        // Stale Metal (new request / prefix restore): rebuild from host.
                        true
                    }
                }
            };
            if reset {
                let mut bufs = Vec::with_capacity(need);
                for _ in 0..need {
                    match frink_metal::attn::MetalKvBuffers::with_capacity(
                        n_kv_heads, head_dim, cap,
                    ) {
                        Ok(b) => bufs.push(b),
                        Err(_) => {
                            **guard = None;
                            break;
                        }
                    }
                }
                if bufs.len() == need {
                    // Sync from host after CPU prefill / prefix restore / capacity grow.
                    let mut ok = true;
                    for (m, c) in bufs.iter_mut().zip(kv_caches.iter()) {
                        // ROWS: this uploads `c.k` / `c.v` themselves, so the
                        // count must describe those buffers.
                        if c.rows() > 0
                            && m.upload_from_host(&c.k.to_f32_vec(), &c.v.to_f32_vec(), c.rows())
                                .is_err()
                        {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        **guard = Some(bufs);
                    } else {
                        **guard = None;
                    }
                } else {
                    **guard = None;
                }
            }
        }

        #[cfg(feature = "metal")]
        let mut metal_stack_done = false;
        #[cfg(feature = "metal")]
        let mut final_norm_done_in_stack = false;
        // OLMoE: all MoE layers in one CB (llama graph style).
        #[cfg(feature = "metal")]
        if use_metal_attn
            && self.layers.iter().enumerate().all(|(i, l)| {
                // Both halves are required. `layer_supports_metal_moe_resident`
                // answers "is this an MoE layer the GPU router can serve",
                // and says nothing about the four features
                // `MoeLayerMetal` has no fields for: a per-layer
                // `rope_theta`, a sliding `window`, `post_attn_norm`
                // and `post_ffn_norm`. `DenseLayerMetal` carries all
                // four and `launch_decode_dense_stack` implements
                // them; the MoE stack does neither, and nothing here
                // refused, so a windowed or sandwich-normed MoE
                // checkpoint would have answered as a different
                // model with no error.
                //
                // The per-layer path already pairs these two checks
                // (see the `metal_moe_resident` branch in the decode
                // loop). Only the whole-stack path was missing it.
                // Latent today because OLMoE and Qwen3-MoE ship none
                // of the four, which is exactly how `attention_scale`
                // stayed latent.
                Self::layer_supports_metal_moe_resident(l, &self.config, self.lora_attached())
                    && !self.layer_needs_metal_stack(l, i)
            })
            && !self.layers.iter().all(Self::layer_supports_metal_dense_ffn)
        {
            if let Some(guard) = metal_kv_guard.as_mut() {
                if let Some(metal_kvs) = guard.as_mut() {
                    if metal_kvs.iter().all(|m| m.seq_len == pos) {
                        let mut moe_layers = Vec::with_capacity(self.layers.len());
                        let mut ok = true;
                        for layer in &self.layers {
                            let ExpertBacking::Resident(_) = &layer.moe.experts else {
                                ok = false;
                                break;
                            };
                            let Some(packed) = Self::moe_packed_q4(&layer.moe) else {
                                ok = false;
                                break;
                            };
                            let (Some(q), Some(k), Some(v), Some(o), Some(r), Some(an), Some(fnw)) = (
                                Self::metal_matvec_launch(&layer.attn.q_proj),
                                Self::metal_matvec_launch(&layer.attn.k_proj),
                                Self::metal_matvec_launch(&layer.attn.v_proj),
                                Self::metal_matvec_launch(&layer.attn.o_proj),
                                Self::metal_matvec_launch(&layer.moe.router),
                                // The kernel bakes both RMSNorms in; the
                                // post-norm-only topology has neither.
                                layer.attn.norm_weight.rms_weights(),
                                layer.moe.norm_weight.rms_weights(),
                            ) else {
                                ok = false;
                                break;
                            };
                            moe_layers.push(frink_metal::attn::MoeLayerMetal {
                                attn_norm_w: an,
                                ffn_norm_w: fnw,
                                q,
                                k,
                                v,
                                o,
                                router: r,
                                packed,
                                extras: self.metal_attn_extras(layer),
                            });
                        }
                        if ok {
                            // Greedy: fold lm_head+argmax into the stack like the
                            // dense path does, and download one u32 instead of a
                            // hidden vector.
                            let greedy_gpu = frink_metal::attn::metal_greedy_argmax_active();
                            let lm_head_gpu_launch = Self::metal_matvec_launch(&self.output_head);
                            // One value carries both "lm_head runs in the
                            // stack" and "the stack returns an argmax id",
                            // so the second cannot drift off the first.
                            // See `decoder::lm_head`.
                            let folded = FoldedLmHead::permit(
                                greedy_gpu,
                                &self.final_norm,
                                self.output_bias.as_deref(),
                                lm_head_gpu_launch,
                            );
                            // The MoE stack used to be handed
                            // `Some(&self.final_norm)` unconditionally,
                            // which is unrepresentable now: a model whose
                            // final norm has no RMS weights leaves the
                            // norm to the host body below, and
                            // `final_norm_done_in_stack` is read off the
                            // SAME value rather than hardcoded `true`.
                            let final_norm_w = self.final_norm.rms_weights();
                            let embd_launch = Self::metal_matvec_launch(&self.embedding);
                            // Gemma scales embd on host; GPU gather has no scale.
                            let embd_gather = if self.config.embedding_scale.is_some() {
                                None
                            } else {
                                match (metal_embd_kind, embd_launch.as_ref()) {
                                    (Some(kind), Some(launch)) => {
                                        Some(frink_metal::attn::EmbdGatherMetal {
                                            kind,
                                            weights: launch.weights,
                                            rows: launch.rows,
                                            row_bytes: launch.row_bytes,
                                            n_cols: hidden_dim,
                                            token_id,
                                        })
                                    }
                                    _ => None,
                                }
                            };
                            if embd_gather.is_none() && hidden.is_empty() {
                                hidden = self.embedding.dequant_row(token_id);
                                if let Some(scale) = self.config.embedding_scale {
                                    for v in hidden.iter_mut() {
                                        *v *= scale;
                                    }
                                }
                            }
                            let seed = if embd_gather.is_some() {
                                frink_metal::attn::moe_decode_ensure(hidden_dim)
                            } else {
                                frink_metal::attn::moe_decode_seed(&hidden)
                            };
                            let hidden_ref: &[f32] =
                                if embd_gather.is_some() { &[] } else { &hidden };
                            match seed.and_then(|_| {
                                // A stack whose layer 0 does not rotate
                                // cannot ride this launch; refuse rather
                                // than rotate, and the CPU body serves
                                // the token.
                                let stack_rope = self
                                    .metal_layer_rope(0)
                                    .ok_or(frink_metal::gpu::MetalError::CommandFailed)?;
                                frink_metal::attn::launch_moe_decode_stack(
                                    hidden_ref,
                                    &moe_layers,
                                    metal_kvs,
                                    self.config.moe.n_experts_active,
                                    self.config.moe.norm_topk_prob,
                                    n_heads,
                                    self.metal_rope(),
                                    // ONE `LayerRope` for the whole
                                    // stack, sound because no layer here
                                    // `layer_needs_metal_stack`: none
                                    // slides, none has a base of its
                                    // own, and none is unrotated -- that
                                    // predicate now covers all three
                                    // through the same accessor, so the
                                    // stack-wide answer and the
                                    // per-layer one cannot disagree.
                                    stack_rope,
                                    pos,
                                    self.config.rms_norm_eps,
                                    final_norm_w,
                                    folded.as_ref().map(FoldedLmHead::launch),
                                    folded.as_ref().is_some_and(FoldedLmHead::argmax_only),
                                    true,
                                    embd_gather.as_ref(),
                                )
                            }) {
                                Ok((out, per_layer_ids)) => {
                                    for (layer, ids) in self.layers.iter().zip(per_layer_ids.iter())
                                    {
                                        if !ids.is_empty() {
                                            layer.moe.record_activations(ids);
                                        }
                                    }
                                    if let Some(folded) = folded.as_ref() {
                                        #[cfg(feature = "metal")]
                                        frink_metal::gpu::clear_resident_activation();
                                        // Softcaps anything vocabulary-shaped;
                                        // passes a 1-element argmax id through.
                                        return folded.interpret(
                                            out,
                                            self.output_head.rows(),
                                            self.config.final_logit_softcap,
                                            self.config.logit_multiplier,
                                        );
                                    }
                                    hidden = out;
                                    final_norm_done_in_stack = final_norm_w.is_some();
                                    metal_stack_done = true;
                                }
                                Err(e) => {
                                    eprintln!(
                                        "frink: Metal MoE stack failed, per-layer fallback: {e}"
                                    );
                                    if hidden.is_empty() {
                                        hidden = self.embedding.dequant_row(token_id);
                                        if let Some(scale) = self.config.embedding_scale {
                                            for v in hidden.iter_mut() {
                                                *v *= scale;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        #[cfg(feature = "metal")]
        if !metal_stack_done
            && use_metal_attn
            && self.layers.iter().all(Self::layer_supports_metal_dense_ffn)
        {
            if let Some(guard) = metal_kv_guard.as_mut() {
                let mut clear_metal_after_stack = false;
                if let Some(metal_kvs) = guard.as_mut() {
                    let seq_ok = metal_kvs.iter().all(|m| m.seq_len == pos);
                    if seq_ok {
                        // Build launches only for resident dense experts (Llama path).
                        let mut dense_layers = Vec::with_capacity(self.layers.len());
                        // The stack's activation uniform, or no stack
                        // at all for an activation it cannot spell.
                        let stack_gelu = self
                            .config
                            .model_ffn_act()
                            .and_then(GluAct::fused_kernel_gelu_flag);
                        let mut ok = stack_gelu.is_some();
                        for (li, layer) in self.layers.iter().enumerate() {
                            let ExpertBacking::Resident(experts) = &layer.moe.experts else {
                                ok = false;
                                break;
                            };
                            let ex = &experts[0];
                            let (
                                Some(q),
                                Some(k),
                                Some(v),
                                Some(o),
                                Some(g),
                                Some(u),
                                Some(d),
                                Some(an),
                                Some(fnw),
                            ) = (
                                Self::metal_matvec_launch(&layer.attn.q_proj),
                                Self::metal_matvec_launch(&layer.attn.k_proj),
                                Self::metal_matvec_launch(&layer.attn.v_proj),
                                Self::metal_matvec_launch(&layer.attn.o_proj),
                                Self::metal_matvec_launch(&ex.gate),
                                Self::metal_matvec_launch(&ex.up),
                                Self::metal_matvec_launch(&ex.down),
                                // The kernel bakes both RMSNorms in; the
                                // post-norm-only topology has neither.
                                layer.attn.norm_weight.rms_weights(),
                                layer.moe.norm_weight.rms_weights(),
                            )
                            else {
                                ok = false;
                                break;
                            };
                            dense_layers.push(frink_metal::attn::DenseLayerMetal {
                                attn_norm_w: an,
                                ffn_norm_w: fnw,
                                q,
                                k,
                                v,
                                o,
                                gate: g,
                                up: u,
                                down: d,
                                extras: self.metal_attn_extras(layer),
                                rope: self.metal_layer_rope(li),
                                window: self.config.layer_sliding_window(li),
                                post_attn_norm: layer.attn.post_attn_norm.as_deref(),
                                post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
                            });
                        }
                        if ok {
                            // Greedy GPU argmax-in-stack (1×u32 download) when
                            // generate marked this thread for temperature<=0.
                            // Otherwise host lm_head after the hidden download,
                            // which measured ~2x the tok/s of a full-vocab one.
                            let greedy_gpu = frink_metal::attn::metal_greedy_argmax_active();
                            let lm_head_gpu_launch = Self::metal_matvec_launch(&self.output_head);
                            // See `decoder::lm_head`: folding lm_head into
                            // the stack and the stack returning an argmax id
                            // are one decision, held in one value.
                            let folded = FoldedLmHead::permit(
                                greedy_gpu,
                                &self.final_norm,
                                self.output_bias.as_deref(),
                                lm_head_gpu_launch,
                            );
                            // Pass final_norm_w when: (1) lm_head runs in stack (folded),
                            // OR (2) lm_head will route to GPU after stack (lm_head_gpu_launch
                            // but no fold) so we can skip download→reupload via TLS.
                            //
                            // `rms_weights()` is what makes the second
                            // case safe for a non-RMS final norm: it is
                            // `None` there, so the stack does not norm
                            // and the host body does.
                            let final_norm_w = if folded.is_some() || lm_head_gpu_launch.is_some() {
                                self.final_norm.rms_weights()
                            } else {
                                None
                            };
                            let embd_launch = Self::metal_matvec_launch(&self.embedding);
                            // Gemma scales the embedding row on the host
                            // (`hidden` already carries sqrt(hidden_dim));
                            // the GPU gather has no scale op — skip it.
                            let embd_gather = if self.config.embedding_scale.is_some() {
                                None
                            } else {
                                match (metal_embd_kind, embd_launch.as_ref()) {
                                    (Some(kind), Some(launch)) => {
                                        Some(frink_metal::attn::EmbdGatherMetal {
                                            kind,
                                            weights: launch.weights,
                                            rows: launch.rows,
                                            row_bytes: launch.row_bytes,
                                            n_cols: hidden_dim,
                                            token_id,
                                        })
                                    }
                                    _ => None,
                                }
                            };
                            let hidden_ref: &[f32] =
                                if embd_gather.is_some() { &[] } else { &hidden };
                            match frink_metal::attn::launch_decode_dense_stack(
                                hidden_ref,
                                &dense_layers,
                                metal_kvs,
                                n_heads,
                                self.metal_rope(),
                                pos,
                                self.config.rms_norm_eps,
                                final_norm_w,
                                folded.as_ref().map(FoldedLmHead::launch),
                                folded.as_ref().is_some_and(FoldedLmHead::argmax_only),
                                embd_gather.as_ref(),
                                stack_gelu.expect("`ok` is false without it"),
                            ) {
                                Ok(out) => {
                                    // Metal KV advanced in-place. Skip host
                                    // last_token_host+push — host may lag until
                                    // sync_metal_attn_kv_to_host / CPU fallback.
                                    // Dense stack has no MoE routing; skip
                                    // per-layer hotness atomics on the hot path.
                                    if let Some(folded) = folded.as_ref() {
                                        // Skip host final_norm/lm_head. Clear TLS.
                                        #[cfg(feature = "metal")]
                                        frink_metal::gpu::clear_resident_activation();
                                        // `interpret` is what keeps
                                        // `final_logit_softcap` applied: the id
                                        // shape passes through, anything
                                        // vocabulary-shaped gets capped.
                                        return folded.interpret(
                                            out,
                                            self.output_head.rows(),
                                            self.config.final_logit_softcap,
                                            self.config.logit_multiplier,
                                        );
                                    }
                                    // Stack downloaded hidden (possibly normalized if
                                    // final_norm_w was Some). Track whether host should
                                    // skip final_norm.
                                    final_norm_done_in_stack = final_norm_w.is_some();
                                    hidden = out;
                                    metal_stack_done = true;
                                }
                                Err(e) => {
                                    eprintln!(
                                        "frink: Metal dense stack failed, per-layer fallback: {e}"
                                    );
                                    if hidden.is_empty() {
                                        hidden = self.embedding.dequant_row(token_id);
                                        if let Some(scale) = self.config.embedding_scale {
                                            for v in hidden.iter_mut() {
                                                *v *= scale;
                                            }
                                        }
                                    }
                                    // Preserve any prior Metal-ahead tokens on host
                                    // before dropping the device buffers.
                                    for (m, c) in metal_kvs.iter().zip(kv_caches.iter_mut()) {
                                        Self::catch_up_host_kv_from_metal(m, c);
                                    }
                                    clear_metal_after_stack = true;
                                }
                            }
                        }
                    }
                }
                if clear_metal_after_stack {
                    **guard = None;
                }
            }
        }

        #[cfg(feature = "metal")]
        let run_cpu_layers = !metal_stack_done;
        #[cfg(not(feature = "metal"))]
        let run_cpu_layers = true;

        // When true, residual lives in Metal MoE scratch — host `hidden` is stale.
        #[cfg(feature = "metal")]
        let mut metal_moe_resident = false;

        #[cfg(feature = "metal")]
        if run_cpu_layers && hidden.is_empty() && !metal_moe_resident {
            // GPU embedding gather or a skipped Metal dense stack can leave
            // `hidden` empty; CPU fallback must not call rms_norm on it.
            hidden = self.embed_token(token_id, pos);
        }

        // The skip source: the (normed) embedding, before any layer
        // touches `hidden` (`crate::skip_stream`). Captured here and
        // not at `embed_token`, because the Metal arms above leave
        // `hidden` empty on purpose -- and a skip-stream model never
        // reaches them (`metal_can_serve_model`).
        let skip_rows = self.config.skip_stream.then(|| hidden.clone());
        let mut hrm = self.hrm_streams(&hidden);
        if run_cpu_layers {
            // Indexed rather than iterated, because a RUN of consecutive
            // recurrent layers is submitted together
            // (`frink_metal::gdn_branch::GdnRun`) and needs several
            // caches at once. `fused_through` is how many layers that
            // consumed; every `continue` below is a `for`, so it cannot
            // spin.
            // `fused_through` is only ever written under `metal` or `cuda`; the
            // binding is unconditional so the loop reads the same either
            // way.
            #[allow(unused_mut, unused_assignments)]
            let mut fused_through = 0usize;
            // The Q/K/V a recurrent run encoded for the attention layer
            // that follows it, carried one iteration so that layer does
            // not project them again in a submission of its own.
            #[cfg(feature = "metal")]
            let mut pending_qkv: crate::decoder::fused_recurrent::PendingQkv = None;
            #[allow(clippy::needless_range_loop)]
            for l in 0..kv_caches.len() {
                self.hrm_stack_input(hrm.as_ref(), l, &mut hidden);
                if l < fused_through {
                    continue;
                }
                // Consecutive layers the fused launch serves whole pass
                // their residual stream to each other on the device and
                // wait ONCE, which is the submission count the decode
                // gap is made of (`docs/plans/gdn-resident-state.md`).
                #[cfg(feature = "metal")]
                if let Some(end) =
                    self.fused_recurrent_run(l, &mut hidden, kv_caches, &mut pending_qkv)
                {
                    fused_through = end;
                    continue;
                }
                // A run of dense layers on the device with the residual
                // resident across them and ONE wait at the end, instead
                // of five round trips per layer (#133). The same stack
                // the batched prefill takes, at one row; it declines
                // before touching any cache, and the body below then
                // runs the layer. `FRINK_CUDA_DECODE=host` is the A/B.
                #[cfg(feature = "cuda")]
                if let Some((h_out, run_len)) =
                    self.try_cuda_dense_stack(l, &hidden, pos, 1, kv_caches)
                {
                    hidden = h_out;
                    fused_through = l + run_len;
                    continue;
                }
                let cache = &mut kv_caches[l];
                let layer = self.layer_for(l);
                // --- attention block ---
                #[cfg(feature = "metal")]
                if metal_moe_resident
                    && (!Self::layer_supports_metal_moe_resident(
                        layer,
                        &self.config,
                        self.lora_attached(),
                    ) || self.layer_needs_metal_stack(layer, l))
                {
                    if let Some(h) = frink_metal::attn::moe_decode_take_hidden() {
                        hidden = h;
                    }
                    metal_moe_resident = false;
                }

                // The router's operand, captured where llama.cpp reads it:
                // BEFORE attention (`smallthinker.cpp:111`). A resident
                // Metal stack never serves that shape
                // (`gpu_router_matches_host_routing`), so `hidden` is not
                // stale for the one architecture that reads it here.
                let inputs = self.branch_inputs(layer, &hidden, 1);
                #[cfg(feature = "metal")]
                let normed = if metal_moe_resident {
                    // Residual is on-device; host rms_norm would use stale hidden.
                    Vec::new()
                } else {
                    self.pre_norm_residual(&layer.attn.norm_weight, &mut hidden, 1)
                };
                #[cfg(not(feature = "metal"))]
                let normed = self.pre_norm_residual(&layer.attn.norm_weight, &mut hidden, 1);

                #[cfg(feature = "metal")]
                {
                    let mut did_metal_attn = false;
                    let mut did_metal_dense = false;
                    let mut did_metal_moe = false;
                    let mut clear_metal_kv = false;
                    if let Some(guard) = metal_kv_guard.as_mut() {
                        if let Some(metal_kvs) = guard.as_mut() {
                            // Metal-authoritative: host may lag after dense-stack skip.
                            // Stack-only features (SWA / sandwich norms / GeGLU /
                            // per-layer theta) are NOT encoded by the per-layer
                            // launches — those layers must go to CPU here.
                            if metal_kvs[l].seq_len == pos
                                && !self.layer_needs_metal_stack(layer, l)
                            {
                                // The two pre-norm weights join the four
                                // projection launches in ONE gate, because every
                                // fused launch below bakes the RMSNorm into its
                                // kernel and there is no weight to bake for the
                                // post-norm-only topology (`olmo2` / `exaone4`).
                                // `NormOp::rms_weights` returning `None` sends
                                // the whole layer to the host body, which reads
                                // the raw residual the way llama.cpp does.
                                // `metal_layer_rope` joins the six because
                                // these four launches all rope
                                // unconditionally, and a layer llama.cpp
                                // does not rotate must reach the host
                                // body instead. It is the same `None`
                                // `layer_needs_metal_stack` reads two
                                // lines up, from the same accessor, so
                                // the two cannot disagree about which
                                // layers those are.
                                if let (
                                    Some(q_l),
                                    Some(k_l),
                                    Some(v_l),
                                    Some(o_l),
                                    Some(attn_norm_w),
                                    Some(ffn_norm_w),
                                    Some(layer_rope),
                                ) = (
                                    Self::metal_matvec_launch(&layer.attn.q_proj),
                                    Self::metal_matvec_launch(&layer.attn.k_proj),
                                    Self::metal_matvec_launch(&layer.attn.v_proj),
                                    Self::metal_matvec_launch(&layer.attn.o_proj),
                                    layer.attn.norm_weight.rms_weights(),
                                    layer.moe.norm_weight.rms_weights(),
                                    self.metal_layer_rope(l),
                                ) {
                                    // Full dense layer on one CB when FFN is Metal-capable.
                                    if Self::layer_supports_metal_dense_ffn(layer) {
                                        let dense_ok = layer.moe.with_expert(0, |ex| {
                                        let (Some(g_l), Some(u_l), Some(d_l)) = (
                                            Self::metal_matvec_launch(&ex.gate),
                                            Self::metal_matvec_launch(&ex.up),
                                            Self::metal_matvec_launch(&ex.down),
                                        ) else {
                                            return false;
                                        };
                                        match frink_metal::attn::launch_decode_dense_layer(
                                            &hidden,
                                            attn_norm_w,
                                            &q_l,
                                            &k_l,
                                            &v_l,
                                            &o_l,
                                            &mut metal_kvs[l],
                                            ffn_norm_w,
                                            &g_l,
                                            &u_l,
                                            &d_l,
                                            n_heads,
                                            self.metal_rope(),
                                            layer_rope,
                                            pos,
                                            self.config.rms_norm_eps,
                                            &self.metal_attn_extras(layer),
                                        ) {
                                            Ok(new_h) => {
                                                // Catch up any dense-stack lag + this token.
                                                Self::catch_up_host_kv_from_metal(
                                                    &metal_kvs[l],
                                                    cache,
                                                );
                                                layer.moe.record_activations(&[0]);
                                                hidden = new_h;
                                                true
                                            }
                                            Err(e) => {
                                                eprintln!(
                                                    "frink: Metal dense layer failed, CPU fallback: {e}"
                                                );
                                                false
                                            }
                                        }
                                    });
                                        if dense_ok {
                                            did_metal_dense = true;
                                            did_metal_attn = true;
                                        } else if metal_kvs[l].seq_len != cache.rows() {
                                            // Dense path may have advanced Metal KV before failing.
                                            Self::catch_up_host_kv_from_metal(&metal_kvs[l], cache);
                                            clear_metal_kv = true;
                                        }
                                    }

                                    // Resident MoE: attn+router on GPU, host top-k only,
                                    // then batched experts — no hidden download/upload.
                                    if !did_metal_dense
                                        && !clear_metal_kv
                                        && Self::layer_supports_metal_moe_resident(
                                            layer,
                                            &self.config,
                                            self.lora_attached(),
                                        )
                                    {
                                        if let Some(router_l) =
                                            Self::metal_matvec_launch(&layer.moe.router)
                                        {
                                            let seed_ok = if metal_moe_resident {
                                                true
                                            } else {
                                                match frink_metal::attn::moe_decode_seed(&hidden) {
                                                    Ok(()) => {
                                                        metal_moe_resident = true;
                                                        true
                                                    }
                                                    Err(e) => {
                                                        eprintln!(
                                                            "frink: Metal MoE seed failed: {e}"
                                                        );
                                                        false
                                                    }
                                                }
                                            };
                                            if seed_ok {
                                                // Prefer one-CB fused path (GPU top-k + packed experts).
                                                // See
                                                // `layer_supports_metal_moe_resident`:
                                                // the fused decode kernel
                                                // routes on the GPU and
                                                // has no `exp_probs_b` /
                                                // `expert_weights_scale`
                                                // input either.
                                                let fused_ok = match &layer.moe.experts {
                                                    ExpertBacking::Resident(_) => {
                                                        if let Some(packed) =
                                                            Self::moe_packed_q4(&layer.moe)
                                                        {
                                                            match frink_metal::attn::launch_moe_decode_layer_fused(
                                                                attn_norm_w,
                                                                &q_l,
                                                                &k_l,
                                                                &v_l,
                                                                &o_l,
                                                                &mut metal_kvs[l],
                                                                ffn_norm_w,
                                                                &router_l,
                                                                &packed,
                                                                self.config.moe.n_experts_active,
                                                                self.config.moe.norm_topk_prob,
                                                                n_heads,
                                                                self.metal_rope(),
                                                                layer_rope,
                                                                pos,
                                                                self.config.rms_norm_eps,
                                                                &self.metal_attn_extras(layer),
                                                            ) {
                                                                Ok(ids) => {
                                                                    layer.moe.record_activations(&ids);
                                                                    did_metal_moe = true;
                                                                    did_metal_attn = true;
                                                                    true
                                                                }
                                                                Err(e) => {
                                                                    eprintln!(
                                                                        "frink: Metal MoE fused layer failed: {e}"
                                                                    );
                                                                    false
                                                                }
                                                            }
                                                        } else {
                                                            false
                                                        }
                                                    }
                                                    _ => false,
                                                };

                                                if !fused_ok {
                                                    match frink_metal::attn::launch_moe_decode_pre(
                                                        attn_norm_w,
                                                        &q_l,
                                                        &k_l,
                                                        &v_l,
                                                        &o_l,
                                                        &mut metal_kvs[l],
                                                        ffn_norm_w,
                                                        &router_l,
                                                        n_heads,
                                                        self.metal_rope(),
                                                        layer_rope,
                                                        pos,
                                                        self.config.rms_norm_eps,
                                                        &self.metal_attn_extras(layer),
                                                    ) {
                                                        Ok(logits) => {
                                                            // Routing happens HERE, on the host,
                                                            // so there is no kernel limitation to
                                                            // excuse a second router: call the
                                                            // one every other host path calls.
                                                            let decision = Self::route_for_layer(
                                                                layer,
                                                                &logits,
                                                                &self.config,
                                                            );
                                                            layer.moe.record_activations(
                                                                &decision.expert_ids,
                                                            );
                                                            if let Some(()) = Self::try_metal_moe_experts_resident(
                                                            layer,
                                                            &decision,
                                                        ) {
                                                            did_metal_moe = true;
                                                            did_metal_attn = true;
                                                        } else if let Some(h) =
                                                            frink_metal::attn::moe_decode_take_hidden()
                                                        {
                                                            hidden = h;
                                                            metal_moe_resident = false;
                                                            // KV already advanced; finish FFN on host.
                                                            let normed2 = self
                                                                .pre_norm_residual(
                                                                &layer.moe.norm_weight,
                                                                &mut hidden,
                                                                1,
                                                            );
                                                            let ffn_out = Self::combine_ffn_outputs_for_position(
                                                                l,
                                                                layer,
                                                                &normed2,
                                                                &normed2,
                                                                &logits,
                                                                &self.config,
                                                                hidden_dim,
                                                                residency.as_ref().map(|p| p.layer_plan(self.physical_index(l))),
                                                            );
                                                            residual_add(
                                                                &mut hidden,
                                                                &ffn_out,
                                                                self.config.residual_scale,
                                                            );
                                                            did_metal_attn = true;
                                                            did_metal_moe = true; // skip second FFN
                                                        }
                                                        }
                                                        Err(e) => {
                                                            eprintln!(
                                                            "frink: Metal MoE pre failed, fallback: {e}"
                                                        );
                                                            if let Some(h) =
                                                            frink_metal::attn::moe_decode_take_hidden()
                                                        {
                                                            hidden = h;
                                                        }
                                                            metal_moe_resident = false;
                                                            if metal_kvs[l].seq_len != cache.rows()
                                                            {
                                                                Self::catch_up_host_kv_from_metal(
                                                                    &metal_kvs[l],
                                                                    cache,
                                                                );
                                                                clear_metal_kv = true;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if !did_metal_dense && !did_metal_moe && !clear_metal_kv {
                                        match frink_metal::attn::launch_decode_attn_block(
                                            &normed,
                                            &q_l,
                                            &k_l,
                                            &v_l,
                                            &o_l,
                                            &mut metal_kvs[l],
                                            n_heads,
                                            self.metal_rope(),
                                            layer_rope,
                                            pos,
                                            &self.metal_attn_extras(layer),
                                            self.config.rms_norm_eps,
                                        ) {
                                            Ok(projected) => {
                                                // Keep Metal KV authoritative — skip per-layer
                                                // host catch-up (dense-stack style). Host is
                                                // flushed on CPU fallback / prefix sync.
                                                residual_add(
                                                    &mut hidden,
                                                    &projected,
                                                    self.config.residual_scale,
                                                );
                                                did_metal_attn = true;
                                            }
                                            Err(e) => {
                                                eprintln!(
                                                "frink: Metal attn block failed, CPU fallback: {e}"
                                            );
                                                Self::catch_up_host_kv_from_metal(
                                                    &metal_kvs[l],
                                                    cache,
                                                );
                                                clear_metal_kv = true;
                                            }
                                        }
                                    }
                                }
                            } else if metal_kvs[l].seq_len > cache.rows() {
                                // Leaving Metal path: host must see full KV for CPU attn.
                                Self::catch_up_host_kv_from_metal(&metal_kvs[l], cache);
                            }
                        }
                        if clear_metal_kv {
                            **guard = None;
                        }
                    }
                    if did_metal_attn {
                        if !did_metal_dense && !did_metal_moe {
                            self.ffn_block_row(
                                l,
                                layer,
                                &mut hidden,
                                None,
                                residency
                                    .as_ref()
                                    .map(|p| p.layer_plan(self.physical_index(l))),
                                inputs,
                                skip_rows.as_deref().map(|rows| SkipStream { rows }),
                            );
                        }
                        continue;
                    }
                }

                let oai = self
                    .gpt_oss
                    .as_ref()
                    .map(|g| &g.layers[self.physical_index(l)]);
                // A recurrent layer whose whole shape the fused Metal
                // launch serves runs END TO END in one submission --
                // branch, residual, norm, FFN, residual -- and this
                // loop moves to the next layer. Everything it does not
                // serve it refuses (`crate::fused_layer`,
                // `decoder::fused_recurrent`), and the two bodies below
                // run exactly as before.
                #[cfg(feature = "metal")]
                if let Some(out) =
                    self.fused_recurrent_layer(l, layer, &normed, &hidden, &mut cache.recurrent)
                {
                    hidden = out;
                    cache
                        .advance_len(1)
                        .expect("unbounded/planned KvCache growth is infallible");
                    continue;
                }
                // An ATTENTION layer whose tail the fused launch serves
                // hands `wo` over to it, so `wo`, the residual add, the
                // FFN norm and the FFN are ONE command buffer instead
                // of two (`crate::decoder::fused_attention`). The
                // attention itself still runs on the host: the KV lives
                // there.
                #[cfg(feature = "metal")]
                let mut deferred: Option<Vec<f32>> = None;
                #[cfg(feature = "metal")]
                let mut ready: Option<attn_block::AttnReady> = None;
                #[cfg(feature = "metal")]
                let residual_in = hidden.clone();
                #[cfg(feature = "metal")]
                let tail = if self.fused_attention_tail_eligible(l, layer) {
                    attn_block::AttnTail::Defer {
                        branch: &mut deferred,
                        // The attention itself goes to the device when
                        // this layer's shape allows it; otherwise only
                        // the tail is fused and the host attends.
                        ready: self
                            .device_attention_shape_ok(l, layer)
                            .then_some(&mut ready),
                    }
                } else {
                    attn_block::AttnTail::apply()
                };
                #[cfg(not(feature = "metal"))]
                let tail = attn_block::AttnTail::apply();
                // The projections the preceding run encoded for THIS
                // layer, if it did: then nothing here projects again.
                #[cfg(feature = "metal")]
                let precomputed = match pending_qkv.take() {
                    Some((at, qkv)) if at == l => Some(qkv),
                    // A run encoded a head for a different layer than
                    // the one we reached, which only a refusal in
                    // between can cause: drop it rather than use it.
                    _ => None,
                };
                #[cfg(not(feature = "metal"))]
                let precomputed = None;
                let projected = self.attn_block_tail(
                    l,
                    layer,
                    &normed,
                    pos,
                    KvStep::Decode(&mut *cache),
                    tail,
                    precomputed,
                );
                // The attention layer's inputs came back instead of its
                // output: run the whole layer on the device, at the head
                // of the next recurrent run when there is one.
                #[cfg(feature = "metal")]
                if let Some(r) = ready {
                    let gate = r.gate.as_deref();
                    if let Some(end) = self.device_attention_layer_then_run(
                        l,
                        layer,
                        &r.q,
                        &r.k,
                        &r.v,
                        gate,
                        &mut hidden,
                        kv_caches,
                        &mut pending_qkv,
                    ) {
                        fused_through = end;
                        continue;
                    }
                    let cache = &mut kv_caches[l];
                    if let Some(out) = self.device_attention_layer(
                        l,
                        layer,
                        cache,
                        &r.q,
                        &r.k,
                        &r.v,
                        gate,
                        &residual_in,
                    ) {
                        cache
                            .push(&r.k, &r.v)
                            .expect("unbounded/planned KvCache growth is infallible");
                        hidden = out;
                        continue;
                    }
                    // Both device shapes refused after the layer had
                    // already been prepared, so finish it on the host
                    // from the very same inputs.
                    let mut attn_out = self.push_and_attend_row(
                        KvStep::Decode(&mut kv_caches[l]),
                        l,
                        layer,
                        &r.k,
                        &r.v,
                        &r.q,
                    );
                    if let Some(g) = gate {
                        crate::attn_gate::apply_interleaved_gate(&mut attn_out, g);
                    }
                    let projected = self.project_attn_rows(layer, &attn_out, 1);
                    residual_add(&mut hidden, &projected, self.config.residual_scale);
                    self.ffn_block_row(
                        l,
                        layer,
                        &mut hidden,
                        oai,
                        residency
                            .as_ref()
                            .map(|p| p.layer_plan(self.physical_index(l))),
                        inputs,
                        skip_rows.as_deref().map(|rows| SkipStream { rows }),
                    );
                    continue;
                }
                #[cfg(feature = "metal")]
                if let Some(branch) = deferred {
                    // The tail feeds the recurrent layers after it, so
                    // it rides in THEIR command buffer when there are
                    // any: one wait for the four layers instead of two.
                    if let Some(end) = self.fused_attention_tail_then_run(
                        l,
                        layer,
                        &branch,
                        &mut hidden,
                        kv_caches,
                        &mut pending_qkv,
                    ) {
                        fused_through = end;
                        continue;
                    }
                    if let Some(out) = self.fused_attention_tail(l, layer, &branch, &hidden) {
                        hidden = out;
                        continue;
                    }
                    // The launch refused after the predicate said yes,
                    // which only a device error does: finish the layer
                    // on the host from the branch it handed back.
                    let p = self.project_attn_rows(layer, &branch, 1);
                    residual_add(&mut hidden, &p, self.config.residual_scale);
                }
                if let Some(projected) = projected {
                    residual_add(&mut hidden, &projected, self.config.residual_scale);
                }
                self.ffn_block_row(
                    l,
                    layer,
                    &mut hidden,
                    oai,
                    residency
                        .as_ref()
                        .map(|p| p.layer_plan(self.physical_index(l))),
                    inputs,
                    skip_rows.as_deref().map(|rows| SkipStream { rows }),
                );
                self.hrm_store(hrm.as_mut(), l, &hidden);
            }
        } // run_cpu_layers

        #[cfg(feature = "metal")]
        if metal_moe_resident {
            if let Some(h) = frink_metal::attn::moe_decode_take_hidden() {
                hidden = h;
            }
        }

        // If Metal stack already ran final_norm, hidden is normalized; else
        // normalize here.
        #[cfg(feature = "metal")]
        let final_normed = if final_norm_done_in_stack {
            hidden.clone()
        } else {
            self.final_norm.apply(&hidden, self.config.rms_norm_eps)
        };
        #[cfg(not(feature = "metal"))]
        let final_normed = self.final_norm.apply(&hidden, self.config.rms_norm_eps);

        let logits = self.logits_from_normed(&final_normed);
        // `FRINK_CUDA_COUNT=1`: this token's syncs and copies.
        #[cfg(feature = "cuda")]
        frink_cuda::counters::step_end(frink_cuda::counters::StepKind::Decode);
        // Clear dense-stack activation TLS after lm_head (may have consumed it).
        // Keep MoE scratch buffers alive across tokens — `moe_decode_seed`
        // overwrites `h` each token; clearing here forced full realloc.
        #[cfg(feature = "metal")]
        frink_metal::gpu::clear_resident_activation();
        logits
    }
}
