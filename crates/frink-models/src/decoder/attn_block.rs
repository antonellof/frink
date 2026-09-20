//! The attention half of one decoder layer, written once.
//!
//! Everything from the QKV projection to `post_attn_norm` used to be
//! spelled out longhand in `forward_token`'s CPU arm and again in
//! `forward_token_paged`, with a third copy of just the push-and-attend
//! step in `forward_multi_seq_kv`. That is how five model features went
//! missing from the paged path one at a time, and how a sixth
//! (`attention_scale`) reached only two of the four host bodies: a copy
//! diverges from its original and nothing notices.
//!
//! So the decorations live here, in one body, and the ONE thing that
//! genuinely differs between the callers -- where this row's K and V are
//! written and read -- is a parameter, [`KvStep`]. That is the same
//! answer [`super::MultiSeqKv`] already reached for the batched path and
//! the same one llama.cpp reached by overloading `build_attn` on its
//! memory-input type.

use frink_core::attention::causal_gqa_attention_row;
use frink_core::cache::{KvCache, PagedKvCache, SharedPagedKv};
use frink_core::matmul::rms_norm;
use frink_core::recurrent_state::RecurrentState;

use super::{Decoder, LayerWeights};
use crate::layer_shapes::AttnShape;

/// Where one row's K/V is written, and what that implies for the kernel
/// that reads it back.
///
/// A named variant per backing rather than a `paged: bool`, for the
/// reason `MultiSeqKv`'s doc comment gives: a tenth caller can silently
/// forget a flag, and cannot silently forget to name a variant.
pub(crate) enum KvStep<'a> {
    /// Single-sequence contiguous decode (`forward_token`).
    ///
    /// The only variant allowed to reach the CUDA resident per-layer KV
    /// in [`Decoder::gqa_attention`]: that buffer holds ONE sequence's
    /// history, seeded by `forward_token` at `pos == 0`.
    Decode(&'a mut KvCache),
    /// One sequence of a multi-sequence batch, contiguous
    /// (`forward_multi_seq`).
    ///
    /// Identical math to [`KvStep::Decode`] minus the CUDA resident
    /// hook. Taking that hook here would answer sequence `b` out of
    /// sequence 0's history, silently -- the resident buffer is never
    /// populated by the batched path.
    Batched(&'a mut KvCache),
    /// Block-table-indexed KV, shared across sequences
    /// (`forward_token_paged`, `forward_multi_seq_kv`'s paged arm).
    Paged {
        cache: &'a mut PagedKvCache,
        stores: &'a SharedPagedKv,
    },
}

impl KvStep<'_> {
    /// The sequence's recurrent-state slot for this layer, whichever
    /// backing holds it (`frink_core::recurrent_state`).
    pub(crate) fn recurrent_slot(&mut self) -> &mut Option<RecurrentState> {
        match self {
            KvStep::Decode(cache) | KvStep::Batched(cache) => &mut cache.recurrent,
            KvStep::Paged { cache, .. } => &mut cache.recurrent,
        }
    }
}

/// Who applies `wo` and the transforms after it.
///
/// `Defer` hands the caller the attention output BEFORE `wo` through
/// the slot and returns `None`, which is what lets a fused Metal tail
/// put `wo`, the residual add, the FFN norm and the FFN in one command
/// buffer (`crate::decoder::fused_attention`).
pub(crate) enum AttnTail<'a> {
    Apply(std::marker::PhantomData<&'a ()>),
    /// Hands the caller the attention output BEFORE `wo`. Only the
    /// Metal decode path defers -- without that backend there is
    /// nothing that could encode `wo` into somebody else's command
    /// buffer -- so the variant is gated and `Apply` carries the
    /// lifetime for both.
    #[cfg(feature = "metal")]
    Defer {
        /// The attention output BEFORE `wo`, when only the tail is
        /// fused.
        branch: &'a mut Option<Vec<f32>>,
        /// Where to leave Q/K/V and the gate when the caller means to
        /// run the ATTENTION on the device as well
        /// (`crate::decoder::device_attention`). `None` asks for the
        /// branch instead, attended on the host.
        ///
        /// Nothing is pushed to the KV cache on this path: the caller
        /// pushes, because only it knows whether the device launch
        /// took the row.
        ready: Option<&'a mut Option<AttnReady>>,
    },
}

/// One attention layer's inputs, after the projections, the biases, the
/// QK norms and RoPE -- exactly what the attention itself reads.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) struct AttnReady {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub gate: Option<Vec<f32>>,
}

impl AttnTail<'_> {
    /// The projection stays where it has always been.
    pub(crate) fn apply() -> Self {
        AttnTail::Apply(std::marker::PhantomData)
    }
}

impl Decoder {
    /// The three projections, on whichever backend serves them.
    ///
    /// A separate function because a recurrent RUN can encode them at
    /// its end for the attention layer that follows
    /// (`GdnRun::attn_head`), and then this is not called at all --
    /// but when it is, it must be the same arithmetic.
    fn project_qkv(layer: &LayerWeights, normed: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            if let Some(mut outs) = frink_core::WeightMatrix::apply_gpu_multi(
                &[&layer.attn.q_proj, &layer.attn.k_proj, &layer.attn.v_proj],
                normed,
            ) {
                let v = outs.pop().unwrap();
                let k = outs.pop().unwrap();
                let q = outs.pop().unwrap();
                (q, k, v)
            } else {
                frink_core::weight_matrix::WeightMatrix::apply_three(
                    &layer.attn.q_proj,
                    &layer.attn.k_proj,
                    &layer.attn.v_proj,
                    normed,
                )
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            frink_core::weight_matrix::WeightMatrix::apply_three(
                &layer.attn.q_proj,
                &layer.attn.k_proj,
                &layer.attn.v_proj,
                normed,
            )
        }
    }

    /// One layer's attention block for ONE row: QKV projection, the
    /// three QKV biases, the two QK norms, RoPE's `mscale`, per-head
    /// RoPE, `attention_scale`, the KV push and attend (with the
    /// layer's sinks, if it has any), the output gate, `o_proj`,
    /// gpt-oss's `o_bias`, and `post_attn_norm`.
    ///
    /// Takes `normed` rather than computing it: `forward_token`'s Metal
    /// arm needs the normed vector before it knows whether the block
    /// will run on the host at all.
    ///
    /// Returns the attention branch's contribution to the residual --
    /// the caller adds it -- or `None` for a layer that HAS no
    /// attention branch (`AttnShape::Absent`, deci.cpp:107-109), where
    /// the residual passes straight through. `Option` rather than an
    /// all-zero vector so a caller cannot add a branch that does not
    /// exist without saying so.
    ///
    /// The head counts are THIS layer's (`ModelConfig::layer_shape`),
    /// which is what makes deci's and openelm's per-layer widths one
    /// body with everyone else's.
    pub(crate) fn attn_block(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        pos: usize,
        kv: KvStep<'_>,
    ) -> Option<Vec<f32>> {
        self.attn_block_tail(layer_idx, layer, normed, pos, kv, AttnTail::apply(), None)
    }

    /// [`Self::attn_block`] with the choice of who applies `wo`.
    ///
    /// One body rather than two, because the decode path that defers
    /// the projection needs every other thing this does -- the QKV
    /// projections, the biases, the two QK norms, RoPE, the scale, the
    /// temperature, the KV push, the attend, the gates -- identically.
    /// A second copy of it is how this file lost eight model features.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_block_tail(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        pos: usize,
        mut kv: KvStep<'_>,
        #[cfg_attr(not(feature = "metal"), allow(unused_variables, unused_mut))] mut tail: AttnTail<
            '_,
        >,
        precomputed: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    ) -> Option<Vec<f32>> {
        let head_dim = self.config.head_dim;
        let (n_heads, n_kv_heads) = match self.config.layer_shape(layer_idx).attention {
            AttnShape::Gqa {
                n_heads,
                n_kv_heads,
            } => (n_heads, n_kv_heads),
            // deci.cpp:115-118: `attn_norm` then `wo`, nothing else.
            AttnShape::Linear => return Some(layer.attn.o_proj.apply(normed)),
            AttnShape::Absent => return None,
            // lfm2.cpp:197 / granite-hybrid.cpp:163: the recurrent block,
            // on this row's cache.
            AttnShape::ShortConv
            | AttnShape::Mamba2
            | AttnShape::Mamba1
            | AttnShape::Plamo2Ssm
            | AttnShape::Gdn
            | AttnShape::Lightning => {
                return Some(self.recurrent_block(layer_idx, layer, normed, 1, kv))
            }
        };

        let (mut q, mut k, mut v) = match precomputed {
            // Already projected, at the end of the recurrent run that
            // wrote the residual this layer norms
            // (`GdnRun::attn_head`): one wait covered both.
            Some(qkv) => qkv,
            None => Self::project_qkv(layer, normed),
        };
        // qwen35.cpp:191-199: the gate rides in `wq`; split it off
        // before anything reads a Q width.
        let q_gate = layer.attn.q_gate_interleaved.then(|| {
            let (qq, gate) = crate::attn_gate::split_interleaved_q_gate(&q, 1, n_heads, head_dim);
            q = qq;
            gate
        });
        // Whole rows here: one token's Q and K. See
        // `Decoder::qk_norm_after_rope` for why the norm has two homes.
        let (q_width, kv_width, v_width) = (q.len(), k.len(), v.len());
        self.apply_qkv_bias_and_clamp(layer, &mut q, &mut k, &mut v, q_width, kv_width, v_width);
        self.apply_qk_norms_pre_rope(layer, &mut q, &mut k, q_width, kv_width);
        self.apply_rope_attn_factor(&mut q, &mut k, layer_idx);

        for h in 0..n_heads {
            self.apply_rope_head_layer(&mut q[h * head_dim..(h + 1) * head_dim], pos, layer_idx);
        }
        for h in 0..n_kv_heads {
            self.apply_rope_head_layer(&mut k[h * head_dim..(h + 1) * head_dim], pos, layer_idx);
        }
        self.apply_qk_norms_post_rope(layer, layer_idx, &mut q, &mut k, q_width, kv_width);
        self.apply_attention_scale(&mut q);
        self.apply_attn_temperature(layer_idx, &mut q, q_width, |_| pos);

        // falcon-h1.cpp:156-160: the parallel Mamba-2 block on the same
        // normed row, summed into the attention branch.
        let ssm = self.parallel_ssm_rows(layer_idx, layer, normed, 1, kv.recurrent_slot());
        // The whole layer on the device, when the sequence's KV mirror
        // can carry it: attention, gate, `wo`, residual, norm, FFN,
        // residual, in one submission
        // (`crate::decoder::device_attention`). Tried BEFORE the host
        // push, so the mirror and the cache are at the same length and
        // nothing has to be re-uploaded; the push below then brings the
        // authority level with it.
        // The caller means to attend on the device: hand it the inputs
        // and let it decide how -- at the head of a run, on its own, or
        // not at all. Nothing is pushed here, because only the caller
        // knows which of those happened.
        #[cfg(feature = "metal")]
        if let AttnTail::Defer {
            ready: Some(slot), ..
        } = &mut tail
        {
            debug_assert!(
                ssm.is_none(),
                "a parallel SSM branch cannot defer its attention"
            );
            **slot = Some(AttnReady {
                q,
                k,
                v,
                gate: q_gate,
            });
            return None;
        }
        let mut attn_out = self.push_and_attend_row(kv, layer_idx, layer, &k, &v, &q);
        let branch = self.attn_branch_rows(layer, normed, &mut attn_out, 1, q_gate.as_deref());
        #[cfg(feature = "metal")]
        if let AttnTail::Defer { branch: slot, .. } = tail {
            // `wo` and everything after it is the caller's, because it
            // is going to encode them into one command buffer with the
            // FFN (`crate::decoder::fused_attention`). A layer with a
            // parallel Mamba-2 branch cannot take that path, and the
            // caller's predicate refuses it -- this is the check that
            // the fence held.
            debug_assert!(ssm.is_none(), "a parallel SSM branch cannot defer its tail");
            *slot = Some(branch);
            return None;
        }
        let mut projected = self.project_attn_rows(layer, &branch, 1);
        Self::add_parallel_ssm(&mut projected, ssm);
        Some(projected)
    }

    /// Everything between the softmax-weighted V sum and the residual
    /// add, for `rows` rows at once: the output gate, `o_proj`, its
    /// scale and bias, and `post_attn_norm`.
    ///
    /// ONE body for the row path (`rows == 1`) and the two batched
    /// host bodies. Before the gate existed each of the three spelled
    /// the `o_proj` / `o_bias` / `post_attn_norm` tail itself, and the
    /// gate would have been a fourth decoration to add to three
    /// places; it is added to one. `normed` is the SAME vector the
    /// Q/K/V projections read, which is what every gating graph
    /// projects the gate from (`crate::attn_gate`).
    ///
    /// `q_gate` is the gate the three bodies split off a double-width
    /// `wq` (`AttnWeights::q_gate_interleaved`), `rows * n_heads *
    /// head_dim` wide, or `None`; `qwen35.cpp:229-230` multiplies its
    /// sigmoid in here, before `wo`.
    pub(crate) fn attn_out_to_residual_rows(
        &self,
        layer: &LayerWeights,
        normed: &[f32],
        attn_out: &mut [f32],
        rows: usize,
        q_gate: Option<&[f32]>,
    ) -> Vec<f32> {
        let branch = self.attn_branch_rows(layer, normed, attn_out, rows, q_gate);
        self.project_attn_rows(layer, &branch, rows)
    }

    /// Everything the attention tail does BEFORE `wo`: the interleaved
    /// Q gate, the output gate, the sub-norm.
    ///
    /// Split from the projection because a fused Metal layer tail does
    /// `wo` itself, inside the command buffer that then adds the
    /// residual and runs the FFN
    /// (`crate::decoder::fused_attention`). Two halves of one function
    /// rather than two functions that must agree: the composition above
    /// is the only other caller.
    pub(crate) fn attn_branch_rows(
        &self,
        layer: &LayerWeights,
        normed: &[f32],
        attn_out: &mut [f32],
        rows: usize,
        q_gate: Option<&[f32]>,
    ) -> Vec<f32> {
        if let Some(gate) = q_gate {
            crate::attn_gate::apply_interleaved_gate(attn_out, gate);
        }
        if let Some(gate) = &layer.attn.output_gate {
            gate.apply_rows(normed, attn_out, rows, self.config.head_dim);
        }
        // bitnet.cpp:101-106: RMS over the concatenated heads, BEFORE
        // `wo`. After the gate only by convention -- no graph has both
        // (`crate::sub_norms`, `crate::attn_gate`) -- and per row,
        // because the norm is over one token's heads.
        match &layer.attn.attn_sub_norm {
            None => attn_out.to_vec(),
            Some(w) => {
                let width = w.len();
                attn_out
                    .chunks(width)
                    .flat_map(|row| rms_norm(row, w, self.config.rms_norm_eps))
                    .collect::<Vec<f32>>()
            }
        }
    }

    /// `wo` and everything after it: the `{1}` scale companion, the
    /// bias, the value scale, the post-attention norm.
    pub(crate) fn project_attn_rows(
        &self,
        layer: &LayerWeights,
        attn_out: &[f32],
        rows: usize,
    ) -> Vec<f32> {
        let mut projected = if rows == 1 {
            layer.attn.o_proj.apply(attn_out)
        } else {
            layer.attn.o_proj.apply_batch(attn_out, rows)
        };
        // `build_lora_mm(wo, cur, wo_s)`: the `{1}` companion multiplied
        // onto the projection's output (`crate::weight_scales`)...
        if let Some(scale) = layer.attn.o_scale {
            for x in projected.iter_mut() {
                *x *= scale;
            }
        }
        // ...and THEN `wo_b`, the order `build_attn` has
        // (`crate::proj_bias`; gpt-oss's, starcoder2's, every graph
        // that creates the tensor).
        if let Some(b) = &layer.attn.o_bias {
            let hidden = b.len();
            for row in projected.chunks_mut(hidden) {
                for (x, b) in row.iter_mut().zip(b.iter()) {
                    *x += b;
                }
            }
        }
        // mimo2.cpp:180-183: the branch scaled AFTER `wo`, before the
        // residual (`crate::attn_value_scale`).
        if let Some(scale) = self.config.attn_value_scale {
            for x in projected.iter_mut() {
                *x *= scale;
            }
        }
        if let Some(post) = &layer.attn.post_attn_norm {
            let hidden = post.len();
            projected = projected
                .chunks(hidden)
                .flat_map(|row| rms_norm(row, post, self.config.post_norm_eps()))
                .collect();
        }
        projected
    }

    /// Appends one row's K/V to whichever backing `kv` names, then
    /// attends over everything that sequence holds.
    ///
    /// The ONLY place the backing shows through, which is the whole
    /// point: paging changes where rows live and nothing else, so an arm
    /// one backing reproduced and the other did not would be a model
    /// that answers differently depending on whether a KV pool happened
    /// to be configured. `causal_gqa_attention_paged_sinks` covers all
    /// three contiguous arms in one entry point and is bit-identical to
    /// each by construction.
    pub(crate) fn push_and_attend_row(
        &self,
        kv: KvStep<'_>,
        layer_idx: usize,
        layer: &LayerWeights,
        k: &[f32],
        v: &[f32],
        q: &[f32],
    ) -> Vec<f32> {
        // The tensor decides, not the architecture: see
        // `AttnWeights::sinks`.
        let sinks = layer.attn.sinks.as_deref();
        // Only a GQA layer pushes; the other two shapes returned before
        // projecting anything. `n_heads()` is zero for them, and zero
        // heads is not a kernel argument this body may be handed.
        let shape = self.config.layer_shape(layer_idx).attention;
        let (n_heads, n_kv_heads) = (shape.n_heads(), shape.n_kv_heads());
        assert!(
            matches!(shape, AttnShape::Gqa { .. }),
            "layer {layer_idx} has no KV to push ({shape:?})"
        );
        let head_dim = self.config.head_dim;
        let v_head_dim = self.config.v_head_dim();
        // The query's own position, BEFORE the push: a chunked layer's
        // window is a function of it (`crate::chunked_swa`).
        let query_pos = match &kv {
            KvStep::Decode(cache) | KvStep::Batched(cache) => cache.positions(),
            KvStep::Paged { cache, .. } => cache.seq_len(),
        };
        let window = self.config.layer_window_for_query(layer_idx, query_pos);
        // The sink arm carries no softcap, matching llama.cpp's.
        let softcap = if sinks.is_some() {
            None
        } else {
            self.config.attn_logit_softcap
        };
        // Derived from the variant rather than passed as a flag; see
        // `KvStep::Batched`. The CUDA resident hook serves the plain
        // full-attention arm only, at one head width: a windowed layer,
        // a layer with sinks, or a model whose V width differs
        // (`crate::kv_head_dims`) takes the host kernel.
        let cuda_resident_layer = match &kv {
            KvStep::Decode(_)
                if window.is_none()
                    && sinks.is_none()
                    && v_head_dim == head_dim
                    && self.alibi_slopes.is_none() =>
            {
                Some(layer_idx)
            }
            KvStep::Decode(_) | KvStep::Batched(_) | KvStep::Paged { .. } => None,
        };
        match kv {
            KvStep::Decode(cache) | KvStep::Batched(cache) => {
                cache
                    .push(k, v)
                    .expect("unbounded/planned KvCache growth is infallible");
                let out = match cuda_resident_layer {
                    Some(l) => self.gqa_attention(
                        l,
                        q,
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cache.rows(),
                    ),
                    // ONE kernel for plain, windowed, softcapped and
                    // sink-bearing layers, at the cache's own two widths.
                    None => causal_gqa_attention_row(
                        q,
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        v_head_dim,
                        cache.rows(),
                        window,
                        sinks,
                        softcap,
                        self.alibi_slopes.as_deref(),
                    ),
                };
                // AFTER the read, never inside `push`: the rows this
                // drops are rows every kernel above has finished with
                // (#61). A no-op unless `FRINK_KV_WINDOW` is on and
                // this layer is windowed -- and note that it is the same
                // `window` the kernels just used, taken from the same
                // `ModelConfig`, because keeping fewer rows than the
                // kernel reads would answer out of a truncated history.
                self.evict_layer_kv(layer_idx, cache);
                out
            }
            KvStep::Paged { cache, stores } => {
                // Write guard for the push alone, then a read guard for
                // the attention: the rule `SharedPagedKv` documents.
                // Holding the write guard across attention would
                // serialise the expensive half and give back a global
                // lock.
                {
                    let mut store = stores.write(layer_idx);
                    cache
                        .push(&mut store, k, v)
                        .expect("every caller reserves this row's pages before the stack runs");
                }
                let store = stores.read(layer_idx);
                frink_core::causal_gqa_attention_paged_sinks(
                    q,
                    &store,
                    cache.block_table(),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    cache.seq_len(),
                    window,
                    sinks,
                    softcap,
                    self.alibi_slopes.as_deref(),
                )
            }
        }
    }
}
