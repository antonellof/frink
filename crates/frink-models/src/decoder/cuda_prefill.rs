//! The CUDA resident prefill stack, as the batched host body reaches
//! it (#259, `docs/plans/cpu-cuda-parity.md` step 2b).
//!
//! A run of consecutive dense layers per call: the hidden batch goes
//! up once and comes back after the last layer, everything between the
//! seven matmuls of each layer is a kernel, and each layer's K/V rows
//! come back for the host `KvCache` -- against the 111 MB and seven
//! synchronous round trips per Llama-3.2-3B layer the host body spends
//! at pp512. The host cache stays authoritative: the rows are pushed
//! here, and a prefix already in the cache is uploaded per layer for
//! the attention kernel. Which layers this may take is `fused_view`'s
//! decision, shared with the Metal stack; what this file adds is the
//! one question only CUDA can answer (does the GEMM serve every
//! projection) and the mapping onto the launch's types.

use frink_core::cache::KvCache;
use frink_cuda::prefill::{
    launch_prefill_dense_stack, AttnExtrasCuda, LayerRopeCuda, MulMmWeights, PrefillDenseLayerCuda,
    PrefillParams, RopeLayoutCuda,
};

use super::fused_view::FusedAttnExtras;
use super::{Decoder, ExpertBacking, LayerWeights};

impl Decoder {
    /// Runs the longest run of eligible dense layers starting at `l`
    /// over `hidden_batch` on the device, pushes each layer's K/V rows
    /// into its cache, and answers the new hidden batch with the run
    /// length. `None` -- BEFORE touching the device or any cache -- when
    /// layer `l` itself is not one the launch serves, so the host body
    /// runs it. A launch that fails after admission is reported once
    /// and also answers `None`, with every cache untouched.
    pub(crate) fn try_cuda_prefill_dense_stack(
        &self,
        l: usize,
        hidden_batch: &[f32],
        start_pos: usize,
        batch_size: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<(Vec<f32>, usize)> {
        if batch_size < 4 || !frink_core::weight_matrix::cuda_dense_enabled() {
            return None;
        }
        // The activation the fused body runs is SwiGLU; `Some(true)` is
        // GELU, which this launch has no kernel for yet.
        if self
            .config
            .model_ffn_act()
            .and_then(frink_moe::GluAct::fused_kernel_gelu_flag)
            != Some(false)
        {
            return None;
        }
        // One `rms_eps` uniform for the whole layer (`:182`), so a
        // model whose post-norms run at a different epsilon than its
        // pre-norms (`frink_models::norm::POST_NORM_EPS_LITERAL`)
        // stays on the host.
        if self.config.post_norm_eps() != self.config.rms_norm_eps {
            return None;
        }
        let kv_width = self.config.n_kv_heads * self.config.head_dim;

        // Describe the run first, borrowing the caches for their
        // prefixes; the borrow ends before the pushes below.
        let mut described = Vec::new();
        for (li, cache) in kv_caches.iter().enumerate().skip(l) {
            let layer = self.layer_for(li);
            let Some(desc) = self.cuda_prefill_layer(li, layer, cache, start_pos) else {
                break;
            };
            described.push(desc);
        }
        if described.is_empty() {
            return None;
        }
        let run_len = described.len();
        let launch: Vec<(&PrefillDenseLayerCuda<'_>, &PrefillParams<'_>)> =
            described.iter().map(|(d, p)| (d, p)).collect();
        let out = match launch_prefill_dense_stack(hidden_batch, &launch, batch_size) {
            Ok(out) => out,
            Err(e) => {
                warn_once(&format!(
                    "frink: CUDA prefill stack declined, host body runs it: {e}"
                ));
                return None;
            }
        };
        drop(launch);
        drop(described);
        debug_assert_eq!(out.kv_rows.len(), run_len);
        for (li, (k_rows, v_rows)) in (l..l + run_len).zip(&out.kv_rows) {
            self.layer_for(li).moe.record_activations(&[0]);
            let cache = &mut kv_caches[li];
            for b in 0..batch_size {
                cache
                    .push(
                        &k_rows[b * kv_width..(b + 1) * kv_width],
                        &v_rows[b * kv_width..(b + 1) * kv_width],
                    )
                    .expect("unbounded/planned KvCache growth is infallible");
            }
        }
        Some((out.hidden, run_len))
    }

    /// One layer as the launch takes it, or `None` for a layer or a
    /// cache state the launch does not serve.
    fn cuda_prefill_layer<'a>(
        &'a self,
        l: usize,
        layer: &'a LayerWeights,
        cache: &'a KvCache,
        start_pos: usize,
    ) -> Option<(PrefillDenseLayerCuda<'a>, PrefillParams<'a>)> {
        if !Self::fused_prefill_dense_layer_eligible(layer, &self.config, self.lora_attached())
            || !self.layer_supports_fused_attn(layer)
        {
            return None;
        }
        // POSITIONS and ROWS both: the prefix the kernel attends over
        // is the cache's rows, and they are only the positions when
        // nothing has been evicted. A windowed cache that has evicted
        // takes the host body, which reads the window it kept.
        if cache.positions() != start_pos || cache.rows() != start_pos {
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
        // A dense layer's one expert is resident (`ExpertBacking::
        // Stored` is the out-of-core MoE); anything else takes the host.
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        let ex = experts.first()?;
        fn view(m: &frink_core::WeightMatrix) -> Option<MulMmWeights<'_>> {
            m.cuda_mul_mm_view()
        }
        let desc = PrefillDenseLayerCuda {
            attn_norm_w: layer.attn.norm_weight.rms_weights()?,
            ffn_norm_w: layer.moe.norm_weight.rms_weights()?,
            q: view(&layer.attn.q_proj)?,
            k: view(&layer.attn.k_proj)?,
            v: view(&layer.attn.v_proj)?,
            o: view(&layer.attn.o_proj)?,
            gate: view(&ex.gate)?,
            up: view(&ex.up)?,
            down: view(&ex.down)?,
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

/// One line per distinct message per process: a launch that declines
/// every prefill would otherwise say so once per step.
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
