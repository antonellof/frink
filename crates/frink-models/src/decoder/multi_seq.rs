//! The multi-sequence decode step: one token for each of several
//! sequences, against contiguous or paged KV.

use crate::skip_stream::SkipStream;
use frink_core::cache::{KvCache, PagedKvCache, SharedPagedKv};
use rayon::prelude::*;

use crate::scalar_multipliers::residual_add;

use super::ffn_block;
use super::{Decoder, KvStep, LayerWeights};

/// Where a batch of independent sequences keeps its KV.
///
/// `forward_multi_seq` batches the projections across sequences but
/// must attend per sequence, because each has its own length and its
/// own history. That per-sequence step is the ONLY place the batched
/// path touches a cache, which is why paging it is a parameter here
/// rather than a second copy of a 300-line function -- the lesson the
/// paged decode path taught by losing five model features one at a
/// time to exactly that kind of copy.
pub enum MultiSeqKv<'a> {
    Contiguous(&'a mut [Vec<KvCache>]),
    Paged {
        caches: &'a mut [Vec<PagedKvCache>],
        stores: &'a SharedPagedKv,
    },
}

impl MultiSeqKv<'_> {
    /// Sequences in the batch.
    pub fn len(&self) -> usize {
        match self {
            MultiSeqKv::Contiguous(c) => c.len(),
            MultiSeqKv::Paged { caches, .. } => caches.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Layers each sequence carries, for the shape assertion.
    pub(super) fn layers_per_seq(&self, seq: usize) -> usize {
        match self {
            MultiSeqKv::Contiguous(c) => c[seq].len(),
            MultiSeqKv::Paged { caches, .. } => caches[seq].len(),
        }
    }

    /// Sequence `b`'s layer-`l` cache as the one-row step the attention
    /// and short-conv bodies take. The only place this enum's arms are
    /// matched for a cache, so paging changes where rows live and
    /// nothing else.
    pub(super) fn step(&mut self, b: usize, l: usize) -> KvStep<'_> {
        match self {
            // `Batched`, not `Decode`: the CUDA resident per-layer KV
            // holds ONE sequence's history, and this path never seeds
            // it. See `KvStep::Batched`.
            MultiSeqKv::Contiguous(caches) => KvStep::Batched(&mut caches[b][l]),
            MultiSeqKv::Paged { caches, stores } => KvStep::Paged {
                cache: &mut caches[b][l],
                stores,
            },
        }
    }
}

impl Decoder {
    /// Appends one position to sequence `b`'s layer-`l` KV, then
    /// attends over everything that sequence holds.
    ///
    /// The only place `forward_multi_seq_kv` touches a cache, and so
    /// the only place the backing matters.
    ///
    /// Selects sequence `b`'s layer-`l` cache and hands it to
    /// [`Decoder::push_and_attend_row`], the one attend body the whole
    /// crate shares. This used to spell that body out a second time; the
    /// contiguous arm of the copy differed from `forward_token`'s by
    /// exactly one call (the CUDA resident hook), which is the kind of
    /// difference nobody notices until it is a wrong answer.
    #[allow(clippy::too_many_arguments)] // one per thing the step needs
    pub(super) fn push_and_attend(
        &self,
        kv: &mut MultiSeqKv<'_>,
        b: usize,
        l: usize,
        layer: &LayerWeights,
        k: &[f32],
        v: &[f32],
        q: &[f32],
    ) -> Vec<f32> {
        self.push_and_attend_row(kv.step(b, l), l, layer, k, v, q)
    }

    /// The body of [`Self::forward_multi_seq_kv`], already running on a
    /// CPU-pool worker. See `entry.rs` for why the split exists.
    pub(super) fn forward_multi_seq_kv_on_worker(
        &self,
        tokens: &[usize],
        positions: &[usize],
        kv: &mut MultiSeqKv<'_>,
    ) -> Vec<Vec<f32>> {
        assert_eq!(tokens.len(), positions.len());
        assert_eq!(tokens.len(), kv.len());
        let batch_size = tokens.len();
        if batch_size == 0 {
            return Vec::new();
        }
        for seq in 0..batch_size {
            assert_eq!(kv.layers_per_seq(seq), self.config.n_layers);
        }

        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let v_head_dim = self.config.v_head_dim();

        // [batch, hidden], flattened row-major.
        let mut hidden_batch: Vec<f32> = self.embed_tokens(tokens, |b| positions[b]);
        let skip_rows = self.config.skip_stream.then(|| hidden_batch.clone());
        let mut hrm = self.hrm_streams(&hidden_batch);

        let residency = self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b));

        for l in 0..self.config.n_layers {
            self.hrm_stack_input(hrm.as_ref(), l, &mut hidden_batch);
            let layer = self.layer_for(l);
            // THIS layer's head counts; see the prefill body.
            let shape = self.config.layer_shape(l);
            let (n_heads, n_kv_heads) = (shape.attention.n_heads(), shape.attention.n_kv_heads());
            // --- attention block ---
            let inputs = self.branch_inputs(layer, &hidden_batch, batch_size);
            let normed_batch: Vec<f32> =
                self.pre_norm_residual(&layer.attn.norm_weight, &mut hidden_batch, batch_size);
            let oai = self
                .gpt_oss
                .as_ref()
                .map(|g| &g.layers[self.physical_index(l)]);

            'attention: {
                match shape.attention {
                    crate::layer_shapes::AttnShape::Absent => break 'attention,
                    crate::layer_shapes::AttnShape::Linear => {
                        let projected = layer.attn.o_proj.apply_batch(&normed_batch, batch_size);
                        residual_add(&mut hidden_batch, &projected, self.config.residual_scale);
                        break 'attention;
                    }
                    // lfm2.cpp:197 / granite-hybrid.cpp:163: each row is
                    // ONE position of its own sequence, so each runs on
                    // its own cache.
                    crate::layer_shapes::AttnShape::ShortConv
                    | crate::layer_shapes::AttnShape::Mamba2
                    | crate::layer_shapes::AttnShape::Mamba1
                    | crate::layer_shapes::AttnShape::Plamo2Ssm
                    | crate::layer_shapes::AttnShape::Gdn
                    | crate::layer_shapes::AttnShape::Lightning => {
                        let mut out = Vec::with_capacity(batch_size * hidden_dim);
                        for b in 0..batch_size {
                            let step = kv.step(b, l);
                            out.extend(self.recurrent_block(
                                l,
                                layer,
                                &normed_batch[b * hidden_dim..(b + 1) * hidden_dim],
                                1,
                                step,
                            ));
                        }
                        residual_add(&mut hidden_batch, &out, self.config.residual_scale);
                        break 'attention;
                    }
                    crate::layer_shapes::AttnShape::Gqa { .. } => {}
                }

                // falcon-h1.cpp:156-160: the parallel Mamba-2 block, each
                // row on its own sequence's cache, before the push.
                let parallel_ssm = layer.attn.ssm.as_ref().map(|_| {
                    let mut out = Vec::with_capacity(batch_size * hidden_dim);
                    for b in 0..batch_size {
                        let mut step = kv.step(b, l);
                        out.extend(
                            self.parallel_ssm_rows(
                                l,
                                layer,
                                &normed_batch[b * hidden_dim..(b + 1) * hidden_dim],
                                1,
                                step.recurrent_slot(),
                            )
                            .expect("the layer has the block"),
                        );
                    }
                    out
                });
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

                let q_width = n_heads * head_dim;
                let kv_width = n_kv_heads * head_dim;
                let v_width = n_kv_heads * v_head_dim;
                let out_width = n_heads * v_head_dim;

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
                self.apply_rope_attn_factor(&mut q_batch, &mut k_batch, l);

                for b in 0..batch_size {
                    let pos = positions[b];
                    let q_row = &mut q_batch[b * q_width..(b + 1) * q_width];
                    for h in 0..n_heads {
                        self.apply_rope_head_layer(
                            &mut q_row[h * head_dim..(h + 1) * head_dim],
                            pos,
                            l,
                        );
                    }
                    let k_row = &mut k_batch[b * kv_width..(b + 1) * kv_width];
                    for h in 0..n_kv_heads {
                        self.apply_rope_head_layer(
                            &mut k_row[h * head_dim..(h + 1) * head_dim],
                            pos,
                            l,
                        );
                    }
                }
                self.apply_qk_norms_post_rope(
                    layer,
                    l,
                    &mut q_batch,
                    &mut k_batch,
                    q_width,
                    kv_width,
                );
                // Applied to the whole Q batch at once because it is
                // elementwise. This path did not apply it at all until the
                // decoration audit: `attention_scale` reached only
                // `forward_token`'s CPU arm and `forward_token_paged`, so a
                // checkpoint carrying one answered at one temperature when
                // decoded alone and another when batched with its neighbours.
                self.apply_attention_scale(&mut q_batch);
                self.apply_attn_temperature(l, &mut q_batch, q_width, |b| positions[b]);

                let mut attn_out_batch = vec![0f32; batch_size * out_width];
                for b in 0..batch_size {
                    let attn_out = self.push_and_attend(
                        kv,
                        b,
                        l,
                        layer,
                        &k_batch[b * kv_width..(b + 1) * kv_width],
                        &v_batch[b * v_width..(b + 1) * v_width],
                        &q_batch[b * q_width..(b + 1) * q_width],
                    );
                    attn_out_batch[b * out_width..(b + 1) * out_width].copy_from_slice(&attn_out);
                }

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

            // --- FFN block --- per row, as this body has always run it;
            // see `BatchedFfnKernels::PerRow`.
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
                ffn_block::BatchedFfnKernels::PerRow,
                skip_rows.as_deref().map(|rows| SkipStream { rows }),
            );
            self.hrm_store(hrm.as_mut(), l, &hidden_batch);
        }

        let final_normed_batch: Vec<f32> = hidden_batch
            .par_chunks(hidden_dim)
            .map(|h| self.final_norm.apply(h, self.config.rms_norm_eps))
            .flatten()
            .collect();
        self.logits_from_flat_hidden(final_normed_batch, batch_size)
    }
}
