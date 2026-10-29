//! The contiguous-KV batched prefill body,
//! `Decoder::forward_hidden_batch_inner`, and its last-row wrapper.
//!
//! One function, kept whole: its labelled blocks share locals, and
//! splitting it would be a rewrite rather than a move.

use crate::skip_stream::SkipStream;
use frink_core::attention::causal_gqa_attention_row;
use frink_core::cache::KvCache;
#[cfg(feature = "metal")]
use frink_core::matmul::rms_norm;
#[cfg(feature = "metal")]
use frink_moe::GluAct;
use rayon::prelude::*;

use crate::scalar_multipliers::residual_add;

use super::ffn_block;
use super::{Decoder, KvStep};

impl Decoder {
    /// [`Self::forward_batch_last`], plus the choice of whether the host
    /// caches have to hold the real K/V when it returns. See
    /// [`Self::advance_host_kv_after_metal_prefill`] for why that is a
    /// choice at all.
    pub(super) fn forward_batch_last_inner(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
        host_kv_authoritative: bool,
    ) -> Vec<f32> {
        let hiddens =
            self.forward_hidden_batch_inner(tokens, start_pos, kv_caches, host_kv_authoritative);
        let Some(last) = hiddens.last() else {
            return Vec::new();
        };
        self.logits_from_normed(last)
    }

    /// [`Self::forward_hidden_batch`] with one extra promise the public
    /// signature cannot express.
    ///
    /// `host_kv_authoritative` says whether the caller will READ
    /// `kv_caches` afterwards. Metal prefill normally leaves K/V on the
    /// device and fills the host rows with a `advance_len` placeholder,
    /// which is correct only because the contiguous decode path then
    /// reads the device buffers too. `forward_batch_last_paged` reads
    /// the host rows -- it copies them into the page store -- so it
    /// passes `true` and pays for the download.
    pub(super) fn forward_hidden_batch_inner(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
        host_kv_authoritative: bool,
    ) -> Vec<Vec<f32>> {
        // Read only by the Metal arms below; a CPU-only build fills the
        // host cache with real rows on every path and has nothing to
        // choose between.
        let _ = host_kv_authoritative;
        assert_eq!(kv_caches.len(), self.config.n_layers);
        let batch_size = tokens.len();
        if batch_size == 0 {
            return Vec::new();
        }

        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let v_head_dim = self.config.v_head_dim();
        // The Metal KV plane holds ONE geometry, and `use_metal_attn`
        // below is false for a per-layer-shape model
        // (`metal_can_serve_model`), so the widest layer's count is
        // every layer's wherever this is read.
        #[cfg(feature = "metal")]
        let n_kv_heads = self.config.n_kv_heads;

        // [batch, hidden], flattened row-major.
        let mut hidden_batch: Vec<f32> = self.embed_tokens(tokens, |b| start_pos + b);
        let skip_rows = self.config.skip_stream.then(|| hidden_batch.clone());

        #[cfg(feature = "metal")]
        let use_metal_attn = crate::decoder::metal_eligibility::metal_attn_switched_on()
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
            let need_cap = start_pos
                .saturating_add(batch_size)
                .saturating_add(256)
                .max(512);
            let reset = match guard.as_ref() {
                None => true,
                Some(v) => {
                    v.len() != need
                        || v.iter().any(|m| m.capacity() < need_cap)
                        || v.iter()
                            .zip(kv_caches.iter())
                            // ROWS: Metal holds rows, and this asks whether the
                            // host buffer matches them.
                            .any(|(m, c)| m.seq_len != c.rows())
                }
            };
            if reset {
                let mut bufs = Vec::with_capacity(need);
                for _ in 0..need {
                    match frink_metal::attn::MetalKvBuffers::with_capacity(
                        n_kv_heads, head_dim, need_cap,
                    ) {
                        Ok(b) => bufs.push(b),
                        Err(_) => {
                            **guard = None;
                            break;
                        }
                    }
                }
                if bufs.len() == need {
                    let mut ok = true;
                    for (m, c) in bufs.iter_mut().zip(kv_caches.iter()) {
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

        let n_layers = self.config.n_layers;
        let mut hrm = self.hrm_streams(&hidden_batch);
        let mut l = 0usize;
        // Labelled for the Metal arm inside the `'attention` block below,
        // whose `continue` must name the loop it leaves.
        #[allow(unused_labels)]
        'layers: while l < n_layers {
            self.hrm_stack_input(hrm.as_ref(), l, &mut hidden_batch);
            let layer = self.layer_for(l);
            // THIS layer's head counts. Zero for the two attention-less
            // shapes, which leave the loop below before a width is used;
            // the Metal arms only run on a uniform model
            // (`metal_can_serve_model`), where every layer's counts are
            // the config's.
            let shape = self.config.layer_shape(l);
            let (n_heads, n_kv_heads) = (shape.attention.n_heads(), shape.attention.n_kv_heads());
            let q_width = n_heads * head_dim;
            let kv_width = n_kv_heads * head_dim;
            let v_width = n_kv_heads * v_head_dim;
            let out_width = n_heads * v_head_dim;

            // Multi-layer dense prefill: one CB, activations stay on GPU.
            #[cfg(feature = "metal")]
            if use_metal_attn && batch_size >= 4 {
                if let Some(guard) = metal_kv_guard.as_mut() {
                    if let Some(metal_kvs) = guard.as_mut() {
                        if let Some(run_len) = self.metal_prefill_dense_stack_run_len(
                            l,
                            start_pos,
                            batch_size,
                            kv_caches,
                            Some(metal_kvs.as_slice()),
                        ) {
                            if let Some(h_out) = self.try_metal_prefill_dense_stack(
                                l,
                                run_len,
                                &hidden_batch,
                                start_pos,
                                batch_size,
                                n_heads,
                                metal_kvs,
                                kv_caches,
                                host_kv_authoritative,
                            ) {
                                hidden_batch = h_out;
                                l += run_len;
                                continue;
                            }
                        }
                    }
                }
            }

            // A run of dense layers on the device, the hidden batch
            // resident across them, instead of seven round trips per
            // layer (#259). Declines before touching any cache; the
            // host body below then runs the layer.
            #[cfg(feature = "cuda")]
            if let Some((h_out, run_len)) = self.try_cuda_prefill_dense_stack(
                l,
                &hidden_batch,
                start_pos,
                batch_size,
                kv_caches,
            ) {
                hidden_batch = h_out;
                l += run_len;
                continue;
            }

            let cache = &mut kv_caches[l];

            // One-CB dense prefill (RMSNorm→QKV GEMM→attn→O→FFN) when every
            // projection has mul_mm_sg and the layer has no QKV bias / QK-norm.
            #[cfg(feature = "metal")]
            if use_metal_attn
                && batch_size >= 4
                && Self::metal_prefill_dense_layer_eligible(
                    layer,
                    &self.config,
                    self.lora_attached(),
                )
            {
                let swa_fits = self.metal_prefill_dense_swa_fits(l, start_pos, batch_size);
                if swa_fits {
                    if let Some(guard) = metal_kv_guard.as_mut() {
                        if let Some(metal_kvs) = guard.as_mut() {
                            // POSITIONS: compared against `start_pos`.
                            if metal_kvs[l].seq_len == cache.positions()
                                && start_pos == cache.positions()
                            {
                                layer.moe.record_activations(&[0]);
                                let fused = layer.moe.with_expert(0, |ex| {
                                    let (q, k, v, o) = (
                                        layer.attn.q_proj.mul_mm_sg_launch()?,
                                        layer.attn.k_proj.mul_mm_sg_launch()?,
                                        layer.attn.v_proj.mul_mm_sg_launch()?,
                                        layer.attn.o_proj.mul_mm_sg_launch()?,
                                    );
                                    let ffn = frink_metal::attn::PrefillFfnMetal::Dense {
                                        gate: ex.gate.mul_mm_sg_launch()?,
                                        up: ex.up.mul_mm_sg_launch()?,
                                        down: ex.down.mul_mm_sg_launch()?,
                                    };
                                    let gelu = self
                                        .config
                                        .model_ffn_act()
                                        .and_then(GluAct::fused_kernel_gelu_flag)?;
                                    let prefill_layer = frink_metal::attn::PrefillDenseLayerMetal {
                                        // See `try_metal_prefill_dense_stack`:
                                        // no weight, no fused launch.
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
                                        rope: self.metal_layer_rope(l),
                                        layer_idx: l as u32,
                                    };
                                    frink_metal::attn::launch_prefill_dense_layer(
                                        &hidden_batch,
                                        &prefill_layer,
                                        &mut metal_kvs[l],
                                        n_heads,
                                        batch_size,
                                        self.metal_rope(),
                                        start_pos,
                                        self.config.rms_norm_eps,
                                        gelu,
                                        self.config.attn_logit_softcap,
                                    )
                                    .ok()
                                });
                                if let Some(h_out) = fused {
                                    Self::advance_host_kv_after_metal_prefill(
                                        &metal_kvs[l],
                                        cache,
                                        batch_size,
                                        host_kv_authoritative,
                                    );
                                    hidden_batch = h_out;
                                    l += 1;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            // --- attention block ---
            let inputs = self.branch_inputs(layer, &hidden_batch, batch_size);
            let normed_batch: Vec<f32> =
                self.pre_norm_residual(&layer.attn.norm_weight, &mut hidden_batch, batch_size);
            let oai = self
                .gpt_oss
                .as_ref()
                .map(|g| &g.layers[self.physical_index(l)]);

            // The GQA body, or the two shapes that have none of it
            // (`crate::layer_shapes::AttnShape`): a labelled block so the
            // FFN half below is reached by all three without a copy of
            // it, and so the Metal arms inside keep their `continue`s.
            'attention: {
                match shape.attention {
                    // deci.cpp:107-109: the residual passes straight through.
                    crate::layer_shapes::AttnShape::Absent => break 'attention,
                    // deci.cpp:115-118: `attn_norm` then `wo`, nothing else.
                    crate::layer_shapes::AttnShape::Linear => {
                        let projected = layer.attn.o_proj.apply_batch(&normed_batch, batch_size);
                        residual_add(&mut hidden_batch, &projected, self.config.residual_scale);
                        break 'attention;
                    }
                    // lfm2.cpp:197 / granite-hybrid.cpp:163: the rows are
                    // consecutive positions of one sequence, on this
                    // layer's cache.
                    crate::layer_shapes::AttnShape::ShortConv
                    | crate::layer_shapes::AttnShape::Mamba2
                    | crate::layer_shapes::AttnShape::Mamba1
                    | crate::layer_shapes::AttnShape::Plamo2Ssm
                    | crate::layer_shapes::AttnShape::Gdn
                    | crate::layer_shapes::AttnShape::Lightning => {
                        let out = self.recurrent_block(
                            l,
                            layer,
                            &normed_batch,
                            batch_size,
                            KvStep::Batched(cache),
                        );
                        residual_add(&mut hidden_batch, &out, self.config.residual_scale);
                        break 'attention;
                    }
                    crate::layer_shapes::AttnShape::Gqa { .. } => {}
                }

                // falcon-h1.cpp:156-160: the parallel Mamba-2 block over the
                // same normed rows, on this layer's cache, before the push.
                let parallel_ssm = self.parallel_ssm_rows(
                    l,
                    layer,
                    &normed_batch,
                    batch_size,
                    &mut cache.recurrent,
                );
                // One shared activation-quant pass for q/k/v (plan 1e): the
                // three projections read the same normed batch, so quantize it
                // once instead of once per projection. A kind mismatch inside
                // the group just re-quantizes locally.
                let qkv_acts = layer
                    .attn
                    .q_proj
                    .quantize_batch_acts(&normed_batch, batch_size);
                let mut q_batch = layer.attn.q_proj.apply_batch_with_acts(
                    &normed_batch,
                    batch_size,
                    qkv_acts.as_ref(),
                );
                // qwen35.cpp:191-199: the gate rides in `wq`; split it
                // off before anything reads a Q width.
                let q_gate = layer.attn.q_gate_interleaved.then(|| {
                    let (q, gate) = crate::attn_gate::split_interleaved_q_gate(
                        &q_batch, batch_size, n_heads, head_dim,
                    );
                    q_batch = q;
                    gate
                });
                let mut k_batch = layer.attn.k_proj.apply_batch_with_acts(
                    &normed_batch,
                    batch_size,
                    qkv_acts.as_ref(),
                );
                let mut v_batch = layer.attn.v_proj.apply_batch_with_acts(
                    &normed_batch,
                    batch_size,
                    qkv_acts.as_ref(),
                );
                drop(qkv_acts);

                self.apply_qkv_bias_and_clamp(
                    layer,
                    &mut q_batch,
                    &mut k_batch,
                    &mut v_batch,
                    q_width,
                    kv_width,
                    v_width,
                );

                self.apply_qk_norms_pre_rope(layer, &mut q_batch, &mut k_batch, q_width, kv_width);
                // Host-side `mscale`, applied before either backend ropes.
                // The Metal branch below therefore hands its kernels
                // `attn_factor_applied_by_caller()` — folding it into cos/sin
                // there as well would square it.
                self.apply_rope_attn_factor(&mut q_batch, &mut k_batch, l);

                #[cfg(feature = "metal")]
                {
                    let mut did_metal_prefill = false;
                    // The Metal prefill kernel is full-causal: only safe on a
                    // SWA layer while every causal position is still inside
                    // the window. Longer prompts fall back to CPU attention.
                    let swa_fits = match self.config.layer_sliding_window(l) {
                        Some(window) => start_pos + batch_size <= window,
                        None => true,
                    };
                    // Metal prefill applies attn softcap in FA-vec / legacy GQA.
                    if let Some(guard) = metal_kv_guard.as_mut() {
                        if let Some(metal_kvs) = guard.as_mut() {
                            // POSITIONS: compared against `start_pos`.
                            // `layer_rope` is `None` where llama.cpp does
                            // not rotate this layer at all
                            // (`crate::rope_layers`); the Metal prefill
                            // block always ropes, so such a layer takes the
                            // CPU body below rather than a rotation the
                            // checkpoint never trained.
                            if let (true, Some(layer_rope)) = (
                                metal_kvs[l].seq_len == cache.positions()
                                    && start_pos == cache.positions()
                                    && swa_fits,
                                self.metal_layer_rope(l),
                            ) {
                                let prefill_res = {
                                    frink_metal::attn::launch_prefill_attn_block(
                                        &q_batch,
                                        &k_batch,
                                        &v_batch,
                                        &mut metal_kvs[l],
                                        n_heads,
                                        batch_size,
                                        self.metal_rope().attn_factor_applied_by_caller(),
                                        layer_rope,
                                        start_pos,
                                        self.config.attn_logit_softcap,
                                        false,
                                    )
                                    .map(
                                        |(attn_out_batch, _, _)| {
                                            Self::advance_host_kv_after_metal_prefill(
                                                &metal_kvs[l],
                                                cache,
                                                batch_size,
                                                host_kv_authoritative,
                                            );
                                            let projected_batch = layer
                                                .attn
                                                .o_proj
                                                .apply_batch(&attn_out_batch, batch_size);
                                            let projected_batch =
                                                if let Some(post) = &layer.attn.post_attn_norm {
                                                    projected_batch
                                                        .chunks(hidden_dim)
                                                        .flat_map(|row| {
                                                            rms_norm(
                                                                row,
                                                                post,
                                                                self.config.rms_norm_eps,
                                                            )
                                                        })
                                                        .collect::<Vec<_>>()
                                                } else {
                                                    projected_batch
                                                };
                                            residual_add(
                                                &mut hidden_batch,
                                                &projected_batch,
                                                self.config.residual_scale,
                                            );
                                            true
                                        },
                                    )
                                };
                                match prefill_res {
                                    Ok(true) => {
                                        did_metal_prefill = true;
                                    }
                                    Ok(false) => {}
                                    Err(e) => {
                                        eprintln!(
                                            "frink: Metal prefill attn failed, CPU fallback: {e}"
                                        );
                                        **guard = None;
                                    }
                                }
                            }
                        }
                    }
                    if did_metal_prefill {
                        self.ffn_block_batch(
                            l,
                            layer,
                            &mut hidden_batch,
                            batch_size,
                            oai,
                            residency
                                .as_ref()
                                .map(|p| p.layer_plan(self.physical_index(l))),
                            inputs,
                            ffn_block::BatchedFfnKernels::Prefill,
                            skip_rows.as_deref().map(|rows| SkipStream { rows }),
                        );
                        l += 1;
                        continue 'layers;
                    }
                }

                // RoPE per token is independent; parallelize for CPU pp512.
                q_batch
                    .par_chunks_mut(q_width)
                    .zip(k_batch.par_chunks_mut(kv_width))
                    .enumerate()
                    .for_each(|(b, (q_row, k_row))| {
                        let pos = start_pos + b;
                        for h in 0..n_heads {
                            self.apply_rope_head_layer(
                                &mut q_row[h * head_dim..(h + 1) * head_dim],
                                pos,
                                l,
                            );
                        }
                        for h in 0..n_kv_heads {
                            self.apply_rope_head_layer(
                                &mut k_row[h * head_dim..(h + 1) * head_dim],
                                pos,
                                l,
                            );
                        }
                    });
                // `maincoder` / `hunyuan-moe` norm HERE instead. Reachable
                // only on the host path, which is why
                // `layer_supports_metal_attn` refuses the layer outright
                // rather than letting the Metal arms above consume a batch
                // that has not been normed yet.
                self.apply_qk_norms_post_rope(
                    layer,
                    l,
                    &mut q_batch,
                    &mut k_batch,
                    q_width,
                    kv_width,
                );
                // Elementwise, so the whole Q batch in one call. Like the
                // multi-sequence path, this body did not apply it at all
                // until the decoration audit. It is placed AFTER the Metal
                // arms above deliberately: none of the seven fused launches
                // has an `attention_scale` uniform, and Q never returns to
                // the host inside `launch_prefill_dense_layer` /
                // `launch_prefill_dense_stack` for it to be scaled. The
                // refusal that keeps those arms out of reach when
                // `attention_scale` is set is in `layer_supports_metal_attn`.
                self.apply_attention_scale(&mut q_batch);
                // Same placement and same fence, per row: row `b` of this
                // batch is position `start_pos + b`, which is what the
                // RoPE loop above used for it.
                self.apply_attn_temperature(l, &mut q_batch, q_width, |b| start_pos + b);

                // ROWS, not positions: it is added to `b + 1` below to give
                // each query in the batch the length of the KV it attends
                // over, which is a count of resident rows.
                let base_seq_len = cache.rows();
                for b in 0..batch_size {
                    cache
                        .push(
                            &k_batch[b * kv_width..(b + 1) * kv_width],
                            &v_batch[b * v_width..(b + 1) * v_width],
                        )
                        .expect("unbounded/planned KvCache growth is infallible");
                }

                // Prefill attention over the just-written KV prefix. Parallel
                // over query positions — the serial loop was a dominant CPU
                // pp512 bottleneck (each query still attends only its causal
                // prefix; K/V slices are immutable after the pushes above).
                // f32 for the blocked kernel's many reads of each row;
                // borrowed for an f32 store, widened once for f16.
                let (cache_k_wide, cache_v_wide) =
                    (cache.k_rows().to_f32_cow(), cache.v_rows().to_f32_cow());
                let (cache_k, cache_v) = (&*cache_k_wide, &*cache_v_wide);
                let softcap = self.config.attn_logit_softcap;
                let batch_window = self.config.batch_window(l, start_pos, batch_size);
                // A layer with sinks takes the per-query path, windowed
                // or not: the blocked kernel has no sink term. So does a
                // chunked layer whose queries straddle a chunk boundary
                // (`crate::chunked_swa`): the blocked kernel takes ONE
                // window. Everything
                // else goes through the blocked kernel, which is Rayon over
                // `[query-block x head]` against one shared KV buffer,
                // windowed or not. SWA layers used to take a per-query
                // `causal_gqa_attention_windowed_softcap` instead, which is
                // `online_attn_accumulate`: two scalar `exp` and a
                // head_dim-wide rescale per KV position, with the head axis
                // serial inside each task. On Gemma-3-1B (22 of 26 layers
                // are SWA) that arm was 19.6% of non-idle CPU `pp512`
                // samples while doing the *same* KV work as this one - at
                // `pp512` the 512-wide window covers the whole prompt.
                let sinks = layer.attn.sinks.as_deref();
                let blocked = match (batch_window, sinks) {
                    (crate::config::BatchWindow::Uniform(window), None) => Some(window),
                    (crate::config::BatchWindow::Uniform(_), Some(_))
                    | (crate::config::BatchWindow::PerQuery, _) => None,
                };
                let mut attn_out_batch = match blocked {
                    Some(window) => self.prefill_attention_blocked(
                        &q_batch,
                        cache_k,
                        cache_v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        v_head_dim,
                        batch_size,
                        base_seq_len,
                        softcap,
                        window,
                    ),
                    None => {
                        let mut out = vec![0f32; batch_size * out_width];
                        out.par_chunks_mut(out_width)
                            .enumerate()
                            .for_each(|(b, dest)| {
                                let seq_len_b = base_seq_len + b + 1;
                                let attn_out = causal_gqa_attention_row(
                                    &q_batch[b * q_width..(b + 1) * q_width],
                                    &cache_k[..seq_len_b * kv_width],
                                    &cache_v[..seq_len_b * v_width],
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    v_head_dim,
                                    seq_len_b,
                                    // Row `b` is position `start_pos + b`,
                                    // as the RoPE loop above had it.
                                    self.config.layer_window_for_query(l, start_pos + b),
                                    sinks,
                                    // The sink arm carries no softcap, as
                                    // `push_and_attend_row` has it.
                                    if sinks.is_some() { None } else { softcap },
                                    self.alibi_slopes.as_deref(),
                                );
                                dest.copy_from_slice(&attn_out);
                            });
                        out
                    }
                };

                // Every query in this batch has now been answered, so the
                // rows behind the window are rows nothing will read again
                // (#61). This is why eviction is not inside `KvCache::push`:
                // `base_seq_len` above was captured BEFORE the batch's
                // pushes and every query's KV length is derived from it, so
                // a drop between the push loop and here would attend the
                // whole prompt over shifted keys.
                //
                // Per layer rather than after the stack, and that is where
                // most of the prefill saving is: a windowed layer hands its
                // prompt rows back before the next layer allocates its own,
                // so a 32k prompt holds ONE layer's full history at a time
                // instead of every windowed layer's at once.
                self.evict_layer_kv(l, cache);

                let mut projected_batch = self.attn_out_to_residual_rows(
                    layer,
                    &normed_batch,
                    &mut attn_out_batch,
                    batch_size,
                    q_gate.as_deref(),
                );
                Self::add_parallel_ssm(&mut projected_batch, parallel_ssm);
                residual_add(
                    &mut hidden_batch,
                    &projected_batch,
                    self.config.residual_scale,
                );
            } // 'attention

            // --- FFN block ---
            self.ffn_block_batch(
                l,
                layer,
                &mut hidden_batch,
                batch_size,
                oai,
                residency
                    .as_ref()
                    .map(|p| p.layer_plan(self.physical_index(l))),
                inputs,
                ffn_block::BatchedFfnKernels::Prefill,
                skip_rows.as_deref().map(|rows| SkipStream { rows }),
            );
            self.hrm_store(hrm.as_mut(), l, &hidden_batch);
            l += 1;
        }

        hidden_batch
            .chunks(hidden_dim)
            .map(|h| self.final_norm.apply(h, self.config.rms_norm_eps))
            .collect()
    }
}
