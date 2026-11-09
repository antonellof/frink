//! The CUDA resident dense stack, as the two host bodies reach it: the
//! batched prefill (#259, `docs/plans/cpu-cuda-parity.md` step 2b) and
//! the single-token decode (#133).
//!
//! A run of consecutive dense layers per call: the hidden rows go up
//! once and come back after the last layer, everything between the
//! seven matmuls of each layer is a kernel, and each layer's K/V rows
//! come back in the same single download for the host `KvCache`.
//! Against that, the host prefill body spends 111 MB and seven
//! synchronous round trips per Llama-3.2-3B layer at pp512, and the
//! host decode body five round trips per layer with the norms, RoPE,
//! the attention and the residual adds on the host between them.
//!
//! ONE entry point for both, and the batch is what differs: a prefill
//! batch multiplies by the GEMM and uploads each layer's prefix for its
//! attention; a decode token multiplies by the matvec and attends over
//! the sequence's device mirror (`KvCache::cuda_attn`), which the stack
//! validates against the host rows and rebuilds when stale. The host
//! cache stays authoritative either way: every row is pushed here.
//! Which layers this may take is `fused_view`'s decision, shared with
//! the Metal stack; what this file adds is the one question only CUDA
//! can answer (does a kernel serve every projection at this batch) and
//! the mapping onto the launch's types.

use frink_core::cache::KvCache;
use frink_core::kv_store::KvBuf;
use frink_cuda::prefill::{
    launch_stack, resident_decode_enabled, AttnExtrasCuda, CudaKvMirror, ExpertProjCuda, GluCuda,
    LayerRopeCuda, MulMmWeights, PrefillDenseLayerCuda, PrefillParams, ProjWeights, RopeLayoutCuda,
    RouterCuda, StackFfn, StackInput, StackKv,
};

use rayon::prelude::*;

use super::fused_view::FusedAttnExtras;
use super::{Decoder, ExpertBacking, LayerWeights};

/// What the launch needs to know about one layer's cache, borrowed
/// apart from its device mirror so the two can be handed over at once.
struct CacheRows<'a> {
    positions: usize,
    rows: usize,
    n_kv_heads: usize,
    head_dim: usize,
    v_head_dim: usize,
    k: &'a KvBuf,
    v: &'a KvBuf,
}

impl Decoder {
    /// The embedding table as the device gather reads it
    /// (`frink_cuda::get_rows`), or `None` when the first hidden state
    /// is not the bare table row and the host must embed: an embedding
    /// scale (Gemma), an embedding norm (BLOOM), a learned position
    /// table, the skip stream's weightless norm, the HRM streams, or a
    /// table that is not plain quantized bytes of a kind CUDA gathers
    /// (an adapter, a Hadamard fold, a float table). Every one of those
    /// is something `embed_token` does to the row after `dequant_row`.
    pub(crate) fn cuda_embedding_gather(&self) -> Option<MulMmWeights<'_>> {
        if !frink_core::weight_matrix::cuda_dense_enabled()
            || !resident_decode_enabled()
            || self.config.embedding_scale.is_some()
            || !matches!(self.embedding_norm, crate::norm::NormOp::None)
            || self.position_embd.is_some()
            || self.config.skip_stream
            || self.hrm_z_l_init.is_some()
        {
            return None;
        }
        let view = self.embedding.cuda_mul_mm_view()?;
        frink_cuda::get_rows::kind_by_name(view.kind.name)?;
        Some(view)
    }

    /// Runs the longest run of eligible dense layers starting at `l`
    /// over `hidden_rows` (`batch_size` rows) on the device, pushes each
    /// layer's K/V rows into its cache, and answers the new hidden rows
    /// with the run length. `None` -- BEFORE touching the device or any
    /// cache -- when layer `l` itself is not one the launch serves, so
    /// the host body runs it. A launch that fails after admission is
    /// reported once and also answers `None`, with every host cache
    /// untouched.
    ///
    /// `batch_size == 1` is a decode token and takes the mirrors;
    /// `FRINK_CUDA_DECODE=host` declines it so the per-op path runs
    /// (the A/B). Two and three rows decline: below four the GEMM tile
    /// is mostly empty and the per-row host body has always served them.
    pub(crate) fn try_cuda_dense_stack(
        &self,
        l: usize,
        input: StackInput<'_>,
        start_pos: usize,
        batch_size: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<(Vec<f32>, usize)> {
        if !frink_core::weight_matrix::cuda_dense_enabled() {
            return None;
        }
        let decode = match batch_size {
            1 if resident_decode_enabled() => true,
            b if b >= 4 => false,
            _ => return None,
        };
        // The gated activations the launch has a kernel for: SwiGLU,
        // and GeGLU (`Some(true)`, Gemma). Every other one is the host's.
        let glu = cuda_glu(
            self.config
                .model_ffn_act()
                .and_then(frink_moe::GluAct::fused_kernel_gelu_flag),
        )?;
        // One `rms_eps` uniform for the whole layer, so a model whose
        // post-norms run at a different epsilon than its pre-norms
        // (`frink_models::norm::POST_NORM_EPS_LITERAL`) stays on the
        // host.
        if self.config.post_norm_eps() != self.config.rms_norm_eps {
            return None;
        }
        let kv_width = self.config.n_kv_heads * self.config.head_dim;

        // Describe the run first, borrowing each cache's rows for the
        // prefix and its mirror slot for the decode; both borrows end
        // before the pushes below.
        let mut described = Vec::new();
        let mut slots: Vec<&mut Option<CudaKvMirror>> = Vec::new();
        for (li, cache) in kv_caches.iter_mut().enumerate().skip(l) {
            let layer = self.layer_for(li);
            let (positions, rows) = (cache.positions(), cache.rows());
            let KvCache {
                n_kv_heads,
                head_dim,
                v_head_dim,
                k,
                v,
                cuda_attn,
                ..
            } = cache;
            let view = CacheRows {
                positions,
                rows,
                n_kv_heads: *n_kv_heads,
                head_dim: *head_dim,
                v_head_dim: *v_head_dim,
                k,
                v,
            };
            let Some(desc) = self.cuda_stack_layer(li, layer, view, start_pos, batch_size, glu)
            else {
                break;
            };
            described.push(desc);
            slots.push(cuda_attn);
        }
        if described.is_empty() {
            return None;
        }
        let run_len = described.len();
        let launch: Vec<(&PrefillDenseLayerCuda<'_>, &PrefillParams<'_>)> =
            described.iter().map(|(d, p)| (d, p)).collect();
        // A prefill step writes its rows into the same device mirrors a
        // decode token attends over, so the first token after a prompt
        // does not upload the prompt's whole history again, and a
        // chunked prefill does not upload its prefix per chunk.
        let kv = StackKv::Mirrors(&mut slots);
        // A routed layer's logits come back mid-stack; the host routes
        // them as it routes everywhere (`route_for_layer`) and lends the
        // chosen experts' views.
        let mut route =
            |i: usize,
             logits: &[f32]|
             -> Result<Vec<(ExpertProjCuda<'_>, f32)>, frink_cuda::gpu::CudaError> {
                let layer = self.layer_for(l + i);
                let decision = Self::route_for_layer(layer, logits, &self.config);
                layer.moe.record_activations(&decision.expert_ids);
                let ExpertBacking::Resident(experts) = &layer.moe.experts else {
                    return Err(frink_cuda::gpu::CudaError::Unsupported(
                        "routed layer without resident experts".to_string(),
                    ));
                };
                decision
                    .expert_ids
                    .iter()
                    .zip(&decision.weights)
                    .map(|(&eid, &w)| {
                        let ex = experts.get(eid).ok_or_else(|| {
                            frink_cuda::gpu::CudaError::Unsupported(format!(
                                "expert {eid} of {}",
                                experts.len()
                            ))
                        })?;
                        let view = |m| {
                            proj_view(m, 1).ok_or_else(|| {
                                frink_cuda::gpu::CudaError::Unsupported("expert view".to_string())
                            })
                        };
                        Ok((
                            ExpertProjCuda {
                                gate: view(&ex.gate)?,
                                up: view(&ex.up)?,
                                down: view(&ex.down)?,
                            },
                            w,
                        ))
                    })
                    .collect()
            };
        let out = match launch_stack(input, &launch, batch_size, kv, Some(&mut route)) {
            Ok(out) => out,
            Err(e) => {
                let what = if decode { "decode" } else { "prefill" };
                warn_once(&format!(
                    "frink: CUDA {what} stack declined, host body runs it: {e}"
                ));
                return None;
            }
        };
        drop(launch);
        drop(described);
        drop(slots);
        debug_assert_eq!(out.layers(), run_len);
        // Each layer's rows into its own cache, the layers in parallel:
        // a Llama-3.1-8B pp512 step pushes 134 MB of K/V into freshly
        // grown cache memory, and one thread doing it took 68 ms on an
        // L40S host, half again the GPU's whole step. The caches are
        // disjoint, and what else a push touches is an atomic counter
        // and the config.
        let rows: Vec<(&[f32], &[f32])> = (0..run_len).map(|i| out.kv(i)).collect();
        kv_caches[l..l + run_len]
            .par_iter_mut()
            .zip(rows.par_iter())
            .enumerate()
            .for_each(|(i, (cache, &(k_rows, v_rows)))| {
                let li = l + i;
                // A routed layer recorded its experts as it routed.
                if Self::is_dense_layer(self.layer_for(li)) {
                    self.layer_for(li).moe.record_activations(&[0]);
                }
                for b in 0..batch_size {
                    cache
                        .push(
                            &k_rows[b * kv_width..(b + 1) * kv_width],
                            &v_rows[b * kv_width..(b + 1) * kv_width],
                        )
                        .expect("unbounded/planned KvCache growth is infallible");
                }
                // After the push, as both host bodies do (`attn_block.rs`,
                // `batch.rs`): a no-op unless `FRINK_KV_WINDOW` is on and
                // the layer is windowed. Once it drops rows the cache's
                // rows stop being its positions and the next call declines
                // this layer, so the host body reads the window it kept.
                self.evict_layer_kv(li, cache);
            });
        drop(rows);
        Some((out.hidden, run_len))
    }

    /// A routed layer the stack serves, with its router's view and its
    /// experts' activation, or `None` for one the host body keeps.
    ///
    /// The routing itself is the host's (`route_for_layer`, called from
    /// the stack on the logits the device computed), so every gating
    /// function routes as it does everywhere else. What the stack does
    /// not reproduce is refused: shared experts and their gate, a weight
    /// applied before the experts, the gpt-oss FFN (biases, its clamped
    /// SwiGLU), a sub-norm, scales on the branch, a router that reads
    /// anything but the normed FFN input, and an expert whose kinds have
    /// no matvec. A prefill batch also needs the tensor-core GEMM, which
    /// the stack checks before it touches the device.
    fn cuda_moe_layer_eligible<'a>(
        &'a self,
        l: usize,
        layer: &'a LayerWeights,
    ) -> Option<(RouterCuda<'a>, GluCuda)> {
        let moe = &layer.moe;
        if self.gpt_oss.is_some()
            || !moe.shared_experts.is_empty()
            || moe.shared_expert_gate.is_some()
            || moe.ffn_sub_norm.is_some()
            || moe.down_scale.is_some()
            || moe.dense_bias.is_some()
            || moe.parallel_sum_scale.is_some()
            || self.config.moe.routed_weight_before_ffn
            || self.config.router_input != crate::router_input::RouterInput::NormedFfnInput
            || !Self::metal_can_serve_model(&self.config, self.lora_attached())
        {
            return None;
        }
        let glu = cuda_glu(
            self.config
                .layer_ffn_acts(l)
                .routed
                .fused_kernel_gelu_flag(),
        )?;
        let ExpertBacking::Resident(experts) = &moe.experts else {
            return None;
        };
        // Every expert, once: a kind with no matvec is refused here, at
        // description, rather than at the first token that routes to it.
        if !experts.iter().all(|ex| {
            [&ex.gate, &ex.up, &ex.down]
                .iter()
                .all(|m| proj_view(m, 1).is_some())
        }) {
            return None;
        }
        let router = match &moe.router {
            frink_core::WeightMatrix::F32(t) if t.cols() == self.config.hidden_dim => {
                RouterCuda::F32 {
                    data: &t.data,
                    rows: t.rows(),
                }
            }
            m => RouterCuda::Quant(proj_view(m, 1)?),
        };
        Some((router, glu))
    }

    /// One layer as the launch takes it, or `None` for a layer or a
    /// cache state the launch does not serve at this batch.
    fn cuda_stack_layer<'a>(
        &'a self,
        l: usize,
        layer: &'a LayerWeights,
        cache: CacheRows<'a>,
        start_pos: usize,
        batch_size: usize,
        glu: GluCuda,
    ) -> Option<(PrefillDenseLayerCuda<'a>, PrefillParams<'a>)> {
        // A routed layer rides the stack at a decode token and in a
        // prefill batch (`cuda_moe_layer_eligible`); every other layer is
        // the dense body's.
        let routed = !Self::is_dense_layer(layer);
        let eligible = if routed {
            (batch_size == 1 || frink_cuda::prefill::routed_batch_supported())
                && self.cuda_moe_layer_eligible(l, layer).is_some()
        } else {
            Self::fused_prefill_dense_layer_eligible(layer, &self.config, self.lora_attached())
        };
        if !eligible || !self.layer_supports_fused_attn(layer) {
            return None;
        }
        // POSITIONS and ROWS both: the prefix the kernel attends over
        // is the cache's rows, and they are only the positions when
        // nothing has been evicted. A windowed cache that has evicted
        // takes the host body, which reads the window it kept.
        if cache.positions != start_pos || cache.rows != start_pos {
            return None;
        }
        let head_dim = self.config.head_dim;
        let (n_heads, n_kv_heads) = (self.config.n_heads, self.config.n_kv_heads);
        let kv_width = n_kv_heads * head_dim;
        if cache.n_kv_heads != n_kv_heads
            || cache.head_dim != head_dim
            || cache.v_head_dim != head_dim
        {
            return None;
        }
        let FusedAttnExtras {
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
        } = Self::fused_attn_extras(layer)?;
        let rope = self.config.layer_rope(l).map(|r| LayerRopeCuda {
            theta: r.theta,
            freq_factors: r.freq_factors,
            rot_dim: r.rot_dim.unwrap_or(head_dim),
            layout: match self.config.rope_layout {
                crate::config::RopeLayout::Norm => RopeLayoutCuda::Norm,
                crate::config::RopeLayout::Neox => RopeLayoutCuda::Neox,
            },
            // `apply_rope_attn_factor` on the host: the rotated channels
            // of Q and K, on a layer that rotates.
            mscale: self.config.rope_attn_factor,
        });
        // A layer's experts are resident (`ExpertBacking::Stored` is the
        // out-of-core MoE); anything else takes the host.
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        let view = |m: &'a frink_core::WeightMatrix| proj_view(m, batch_size);
        let (ffn, glu) = if routed {
            let (router, glu) = self.cuda_moe_layer_eligible(l, layer)?;
            let ffn_dim = experts.first()?.gate.rows();
            (StackFfn::Moe { router, ffn_dim }, glu)
        } else {
            let ex = experts.first()?;
            (
                StackFfn::Dense(ExpertProjCuda {
                    gate: view(&ex.gate)?,
                    up: view(&ex.up)?,
                    down: view(&ex.down)?,
                }),
                glu,
            )
        };
        let desc = PrefillDenseLayerCuda {
            attn_norm_w: layer.attn.norm_weight.rms_weights()?,
            ffn_norm_w: layer.moe.norm_weight.rms_weights()?,
            q: view(&layer.attn.q_proj)?,
            k: view(&layer.attn.k_proj)?,
            v: view(&layer.attn.v_proj)?,
            o: view(&layer.attn.o_proj)?,
            ffn,
            glu,
            post_attn_norm: layer.attn.post_attn_norm.as_deref(),
            post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
            extras: AttnExtrasCuda {
                q_bias,
                k_bias,
                v_bias,
                q_norm,
                k_norm,
            },
            rope,
        };
        let params = PrefillParams {
            n_heads,
            n_kv_heads,
            head_dim,
            rms_eps: self.config.rms_norm_eps,
            // The host body's scale (`causal_gqa_attention_row`);
            // `attention_scale` is fenced off by
            // `layer_supports_fused_attn`.
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            attn_softcap: self.config.attn_logit_softcap,
            window: self.config.layer_sliding_window(l),
            // The kernel reads f32 rows. CUDA keeps an f32 host store
            // (`kv_budget::host_kv_dtype`); an f16 one, which a CUDA
            // build can still be handed, takes the host body rather than
            // being reinterpreted.
            prefix_k: &cache.k.as_f32()?[..start_pos * kv_width],
            prefix_v: &cache.v.as_f32()?[..start_pos * kv_width],
            start_pos,
        };
        Some((desc, params))
    }
}

/// Both views of a projection; the stack checks they describe the same
/// bytes and refuses a one-row step whose kind has no matvec, so a kind
/// added to one CUDA table and not the other takes the host body rather
/// than the wrong kernel.
fn proj_view(m: &frink_core::WeightMatrix, batch_size: usize) -> Option<ProjWeights<'_>> {
    let gemm = m.cuda_mul_mm_view()?;
    let matvec = m.cuda_matvec_launch();
    if ProjWeights::uses_matvec(batch_size) && matvec.is_none() {
        return None;
    }
    Some(ProjWeights { gemm, matvec })
}

/// The launch's activation for `GluAct::fused_kernel_gelu_flag`'s
/// answer: `Some(false)` is SwiGLU, `Some(true)` GeGLU, and `None` (an
/// activation no fused kernel computes) declines the layer.
fn cuda_glu(gelu: Option<bool>) -> Option<GluCuda> {
    match gelu? {
        false => Some(GluCuda::Silu),
        true => Some(GluCuda::Gelu),
    }
}

/// One line per distinct message per process: a launch that declines
/// every step would otherwise say so once per step.
fn warn_once(msg: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
    if guard
        .get_or_insert_with(HashSet::new)
        .insert(msg.to_string())
    {
        eprintln!("{msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every activation a fused kernel computes reaches the CUDA stack
    /// as its own kernel, and one no fused kernel computes declines the
    /// layer rather than running as SwiGLU.
    #[test]
    fn each_fused_activation_maps_to_its_cuda_kernel() {
        use frink_moe::GluAct;
        assert_eq!(
            cuda_glu(GluAct::Swiglu.fused_kernel_gelu_flag()),
            Some(GluCuda::Silu)
        );
        assert_eq!(
            cuda_glu(GluAct::Geglu.fused_kernel_gelu_flag()),
            Some(GluCuda::Gelu)
        );
        assert_eq!(cuda_glu(GluAct::Reglu.fused_kernel_gelu_flag()), None);
        assert_eq!(cuda_glu(GluAct::ReluSqr.fused_kernel_gelu_flag()), None);
    }
}
