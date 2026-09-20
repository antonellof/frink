//! The FFN half of one decoder layer, written once per SHAPE: one body
//! for a row, one for a batch of rows.
//!
//! `forward_token`'s CPU arm, its Metal-attention arm and
//! `forward_token_paged` each spelled out the same six lines -- norm,
//! gpt-oss or generic FFN, `post_ffn_norm`, residual add -- and the
//! per-layer seam (`crate::layer_shapes`) needed a seventh fact in all
//! three: an FFN-free layer (`deci.cpp:147-149`) runs none of it. Three
//! copies of a branch is how features go missing from one path; this is
//! the one body, and the callers pass the one thing that differs.
//!
//! The batched half had THREE copies too -- the Metal-prefill arm and
//! the host arm of `forward_hidden_batch_inner`, and
//! `forward_multi_seq_kv_on_worker` -- and the router-operand seam
//! (`crate::router_input`) needed an eighth fact in all of them: WHICH
//! tensor the router reads. Read side by side, the first two differed
//! by the gpt-oss branch and the FFN-free check (both unreachable on
//! the Metal arm, both present on the host arm) and the third by
//! having none of the batched fast paths. [`Decoder::ffn_block_batch`]
//! is the one body; the multi-sequence caller passes
//! `BatchedFfnKernels::PerRow` to keep exactly the behaviour it had.
//!
//! # What is captured before attention
//!
//! Two things the FFN body needs are facts about the hidden state AS
//! IT ENTERS THE LAYER, which attention has mutated in place by the
//! time the body runs: what the router reads ([`RouterOperand`];
//! `smallthinker.cpp:111` reads `inpL`, `arctic.cpp:136` norms `inpSA`)
//! and, on a PARALLEL-residual layer, what the FFN itself reads
//! ([`FfnInput`]; `gptneox.cpp:149` norms `inpL` with `ffn_norm`,
//! `plamo.cpp:97` and `stablelm.cpp:137` hand the FFN the vector
//! attention read, `crate::parallel_residual`). They travel together
//! as [`BranchInputs`], and [`Decoder::branch_inputs`] is the ONE
//! constructor, called where `attn_norm` is applied, before attention.
//! For every sequential architecture with a router on the FFN input it
//! answers the two defaults without reading its argument, and the body
//! computes `ffn_norm(h)` and `router · normed2` as it always did.
//! There is no `Default` and no second constructor, so a body cannot
//! be reached with either fact unstated, and a body that took one and
//! forgot the other cannot be written.
//!
//! # The parallel dense FFN
//!
//! A layer whose dense FFN is summed with its experts
//! (`crate::parallel_dense_ffn`: Grok-2, Arctic) has that FFN in the
//! shared-expert slot and, when the row scales the sum, a
//! `parallel_sum_scale` on the layer; [`Decoder::apply_parallel_sum_scale`]
//! multiplies the WHOLE branch output by it right after the combine and
//! before `down_scale` and the post-FFN norm, where `grok.cpp:180-186`
//! put it. Beside `apply_down_scale` at every site, so the two cannot
//! drift.

use frink_core::matmul::rms_norm;

use super::{Decoder, GptOssLayer, LayerWeights};
use crate::norm::NormOp;
use crate::router_input::RouterInput;
use crate::scalar_multipliers::residual_add;
use crate::skip_stream::SkipStream;

/// What the MoE router multiplies, for one layer of one forward pass.
///
/// See the module doc. `Precomputed` holds LOGITS, not the operand:
/// `smallthinker.cpp:111` computes them before attention, and
/// computing them at the same point keeps the arithmetic order
/// llama.cpp's and makes the operand impossible to confuse with the
/// post-attention residual, which is the same `Vec` mutated in place.
#[derive(Debug)]
pub(crate) enum RouterOperand {
    /// `router · ffn_norm(ffn_inp)`, computed inside the FFN body --
    /// llama.cpp's `build_moe_ffn` default and every generic-path graph
    /// but one.
    FfnInput,
    /// `router · inpL`, already computed: `[batch, n_experts]`.
    Precomputed(Vec<f32>),
    /// Arctic: the routed branch's INPUT, `ffn_norm_exps(inpSA)`,
    /// `[batch, hidden_dim]`. The body computes `router · x` from it
    /// and runs the routed experts on it; the dense half still reads
    /// `ffn_norm(ffn_inp)` (`crate::router_input::RouterInput::
    /// NormedLayerInput`).
    BranchInput(Vec<f32>),
}

/// What the FFN reads: the ordinary pre-FFN norm of the post-attention
/// residual, computed inside the body, or -- on a parallel-residual
/// layer -- a norm of the LAYER INPUT, captured before attention
/// (`crate::parallel_residual`). `[batch, hidden_dim]`, already normed.
#[derive(Debug)]
pub(crate) enum FfnInput {
    PostAttnResidual,
    LayerInput(Vec<f32>),
}

/// The two pre-attention facts a layer's FFN body needs, built by
/// [`Decoder::branch_inputs`] and nothing else.
#[derive(Debug)]
pub(crate) struct BranchInputs {
    pub(crate) router: RouterOperand,
    pub(crate) ffn: FfnInput,
}

/// Whether the batched FFN body may take its batched kernels
/// (`dense_ffn_batch`, `moe_ffn_batch`, the Metal MoE prefill) or must
/// run every row through the per-position bodies.
///
/// `PerRow` is what the multi-sequence body has always done: its rows
/// are one token from each of N sequences, the batched kernels'
/// thresholds (4 and 32 rows) were tuned for prefill, and switching
/// continuous batching onto them is a measurable change with its own
/// A/B, not something to smuggle in under a seam. A parameter rather
/// than a second body, so the two cannot drift about anything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchedFfnKernels {
    Prefill,
    PerRow,
}

impl Decoder {
    /// The weights logical layer `l` runs: `layers[l]` for every model
    /// but a looped one, where it is `layers[l % n_phys]`
    /// (`crate::layer_loops`). THE mapping; the three host bodies
    /// iterate `0..config.n_layers` and ask it, so a body cannot index
    /// `layers` with a logical index by mistake.
    pub fn layer_for(&self, l: usize) -> &LayerWeights {
        &self.layers[self.physical_index(l)]
    }

    /// The physical index behind logical layer `l`: the index into
    /// `layers`, the gpt-oss side table and the residency plan, all of
    /// which are sized per PHYSICAL layer.
    pub(crate) fn physical_index(&self, l: usize) -> usize {
        match self.config.layer_loops {
            Some(loops) => loops.physical(l),
            None => l,
        }
    }

    /// A dense layer's single expert on ONE row: `run_expert`, or its
    /// sub-normed twin when the layer carries BitNet's `ffn_sub_norm`
    /// (`bitnet.cpp:135-140`, `crate::sub_norms`).
    ///
    /// The one place the dense row body decides between the two, so
    /// that a fused kernel reachable from `run_expert` (the on-device
    /// SwiGLU, which has no norm between the activation and `down`)
    /// cannot be reached for a layer that needs the norm.
    pub(crate) fn run_dense_expert(
        layer: &LayerWeights,
        normed2: &[f32],
        act: frink_moe::GluAct,
        eps: f32,
    ) -> Vec<f32> {
        layer.moe.with_expert(0, |ex| {
            match (&layer.moe.ffn_sub_norm, &layer.moe.dense_bias) {
                (None, None) => frink_moe::run_expert(normed2, ex, act),
                (Some(w), None) => frink_moe::run_expert_sub_normed(normed2, ex, act, w, eps),
                // `crate::proj_bias`: the biases before the activation
                // and after `down`. No graph has both a bias and an
                // inner norm (`bitnet` has neither bias), so the pair
                // is a loader-refused shape rather than a fourth body.
                (None, Some(bias)) => frink_moe::run_expert_biased(normed2, ex, act, bias),
                (Some(_), Some(_)) => {
                    unreachable!(
                        "a dense layer with both an inner norm and biases is refused at load"
                    )
                }
            }
        })
    }

    /// THE constructor for [`BranchInputs`]. `hidden_before_attn` is
    /// `[batch_size, hidden_dim]`, the residual stream as it enters the
    /// layer.
    pub(crate) fn branch_inputs(
        &self,
        layer: &LayerWeights,
        hidden_before_attn: &[f32],
        batch_size: usize,
    ) -> BranchInputs {
        BranchInputs {
            router: self.router_operand(layer, hidden_before_attn, batch_size),
            ffn: self.ffn_input(layer, hidden_before_attn, batch_size),
        }
    }

    /// What the FFN reads on a parallel layer: `attn_norm(x)` -- the
    /// same function on the same vector attention took, so the two
    /// cannot disagree -- or `ffn_norm(x)`; and the body's own
    /// `ffn_norm(h)` on a sequential one.
    fn ffn_input(
        &self,
        layer: &LayerWeights,
        hidden_before_attn: &[f32],
        batch_size: usize,
    ) -> FfnInput {
        use crate::parallel_residual::ParallelNorm;
        let norm = match layer.moe.parallel {
            None => return FfnInput::PostAttnResidual,
            Some(ParallelNorm::SharedNorm) => {
                debug_assert!(
                    matches!(layer.moe.norm_weight, NormOp::None),
                    "a shared-norm parallel layer has no pre-FFN tensor"
                );
                &layer.attn.norm_weight
            }
            Some(ParallelNorm::TwoNorms) => &layer.moe.norm_weight,
        };
        debug_assert_eq!(
            hidden_before_attn.len(),
            batch_size * self.config.hidden_dim
        );
        let eps = self.config.rms_norm_eps;
        FfnInput::LayerInput(
            hidden_before_attn
                .chunks(self.config.hidden_dim)
                .flat_map(|row| norm.apply(row, eps))
                .collect(),
        )
    }

    /// The router half of [`Self::branch_inputs`].
    ///
    /// Answers `FfnInput` for a dense layer (nothing to route) and for
    /// gpt-oss (`gpt_oss_ffn` computes its own biased logits from the
    /// normed input and is the only reader of `router_bias`; no
    /// gpt-oss graph routes on the layer input, and the debug assert
    /// pins that the table agrees).
    fn router_operand(
        &self,
        layer: &LayerWeights,
        hidden_before_attn: &[f32],
        batch_size: usize,
    ) -> RouterOperand {
        match self.config.router_input {
            RouterInput::NormedFfnInput => RouterOperand::FfnInput,
            RouterInput::NormedLayerInput => {
                if Self::is_dense_layer(layer) {
                    return RouterOperand::FfnInput;
                }
                debug_assert_eq!(
                    hidden_before_attn.len(),
                    batch_size * self.config.hidden_dim
                );
                // REQUIRED at load for this operand (`arctic.cpp:45`);
                // a layer without it is a loader defect, not a file's.
                let w = layer
                    .moe
                    .exps_norm
                    .as_deref()
                    .expect("NormedLayerInput layer loaded without ffn_norm_exps");
                let eps = self.config.rms_norm_eps;
                RouterOperand::BranchInput(
                    hidden_before_attn
                        .chunks(self.config.hidden_dim)
                        .flat_map(|row| rms_norm(row, w, eps))
                        .collect(),
                )
            }
            RouterInput::RawLayerInput => {
                debug_assert!(
                    self.gpt_oss.is_none(),
                    "gpt-oss routes on the normed input; a RawLayerInput gpt-oss model is not a \
                     shape llama.cpp has"
                );
                if Self::is_dense_layer(layer) {
                    return RouterOperand::FfnInput;
                }
                debug_assert_eq!(
                    hidden_before_attn.len(),
                    batch_size * self.config.hidden_dim
                );
                RouterOperand::Precomputed(if batch_size == 1 {
                    layer.moe.router.apply(hidden_before_attn)
                } else {
                    layer.moe.router.apply_batch(hidden_before_attn, batch_size)
                })
            }
        }
    }

    /// Runs layer `layer_idx`'s FFN on `hidden` and adds it back, or
    /// does nothing for a layer whose shape has no FFN.
    ///
    /// `hidden` is the post-attention residual; on return it is the
    /// layer's output.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ffn_block_row(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        hidden: &mut [f32],
        oai: Option<&GptOssLayer>,
        plan: Option<&frink_moe::PlacementPlan>,
        inputs: BranchInputs,
        skip: Option<SkipStream<'_>>,
    ) {
        if self.config.layer_shape(layer_idx).ffn_dim == 0 {
            return;
        }
        let hidden_dim = self.config.hidden_dim;
        let BranchInputs {
            router: operand,
            ffn,
        } = inputs;
        let normed2 = match ffn {
            FfnInput::PostAttnResidual => self.pre_norm_residual(&layer.moe.norm_weight, hidden, 1),
            // A parallel layer's FFN reads a norm of the LAYER input,
            // which `Decoder::branch_inputs` took before attention ran;
            // nothing there is the residual stream, so there is nothing
            // for `crate::normed_residual` to adopt. No architecture
            // has both (`normed_residual::NORMED_RESIDUAL_ARCHITECTURES`
            // is one row and `minimax-01.cpp:434-440` is sequential).
            FfnInput::LayerInput(x) => x,
        };
        let mut ffn_out = match oai {
            Some(oai) => Self::gpt_oss_ffn(layer, oai, &normed2, &self.config, hidden_dim),
            None => Self::run_ffn_block(
                layer_idx,
                layer,
                &normed2,
                &self.config,
                hidden_dim,
                plan,
                operand,
            ),
        };
        Self::apply_parallel_sum_scale(layer, &mut ffn_out);
        Self::apply_down_scale(layer, &mut ffn_out);
        if let Some(post) = &layer.attn.post_ffn_norm {
            ffn_out = rms_norm(&ffn_out, post, self.config.post_norm_eps());
        }
        residual_add(hidden, &ffn_out, self.config.residual_scale);
        Self::apply_skip_stream(layer, hidden, skip, 1, hidden_dim);
        self.apply_loop_norm(layer_idx, hidden, 1);
    }

    /// `ggml_scale(ffn_out + moe_out, s)` (`grok.cpp:180`): the factor
    /// on the whole branch of a layer whose dense FFN is summed with
    /// its experts (`crate::parallel_dense_ffn`). Elementwise, so one
    /// row and a batch of rows are the same call; a no-op for every
    /// layer without the row's scale.
    fn apply_parallel_sum_scale(layer: &LayerWeights, ffn_out: &mut [f32]) {
        if let Some(scale) = layer.moe.parallel_sum_scale {
            for x in ffn_out.iter_mut() {
                *x *= scale;
            }
        }
    }

    /// `build_ffn(..., down, down_b, down_s, ...)`: the `{1}` companion
    /// multiplied onto the FFN output right after `down`, before any
    /// post-norm (`crate::weight_scales`). Elementwise, so one row and a
    /// batch of rows are the same call.
    fn apply_down_scale(layer: &LayerWeights, ffn_out: &mut [f32]) {
        if let Some(scale) = layer.moe.down_scale {
            for x in ffn_out.iter_mut() {
                *x *= scale;
            }
        }
    }

    /// Talkie's second residual (`talkie.cpp:123-126`,
    /// `crate::skip_stream`): `hidden += skip * out_scale`, row by row,
    /// after the FFN residual add. A model without the stream passes
    /// `None` and carries no `out_scale`; the two agree because both
    /// come from one `ModelConfig::skip_stream`, and the assert says so.
    fn apply_skip_stream(
        layer: &LayerWeights,
        hidden: &mut [f32],
        skip: Option<SkipStream<'_>>,
        rows: usize,
        hidden_dim: usize,
    ) {
        match (skip, layer.out_scale) {
            (Some(skip), Some(scale)) => {
                debug_assert_eq!(skip.rows.len(), rows * hidden_dim);
                debug_assert_eq!(hidden.len(), rows * hidden_dim);
                for (h, s) in hidden.iter_mut().zip(skip.rows.iter()) {
                    *h += s * scale;
                }
            }
            (None, None) => {}
            (skip, scale) => unreachable!(
                "the skip stream and the layer's out_scale come from one config fact; \
                 got skip={} out_scale={scale:?}",
                skip.is_some()
            ),
        }
    }

    /// The norm a pass boundary applies (`crate::layer_loops`). Here,
    /// at the end of BOTH FFN bodies, so every caller of either gets
    /// it; a model that does not loop never enters the branch.
    ///
    /// Two shapes: `nanbeige.cpp:167-175` norms with the model's own
    /// `output_norm` after every pass but the last, and
    /// `hrm-text.cpp:162` closes every stack with a WEIGHTLESS RMS --
    /// which for that architecture is the only final norm there is.
    fn apply_loop_norm(&self, layer_idx: usize, hidden: &mut [f32], rows: usize) {
        let Some(loops) = self.config.layer_loops else {
            return;
        };
        let Some(kind) = loops.loop_norm_after(layer_idx) else {
            return;
        };
        let width = self.config.hidden_dim;
        debug_assert_eq!(hidden.len(), rows * width);
        for row in hidden.chunks_mut(width) {
            let normed = match kind {
                crate::layer_loops::LoopNorm::Output => {
                    self.final_norm.apply(row, self.config.rms_norm_eps)
                }
                crate::layer_loops::LoopNorm::Weightless => {
                    crate::norm::NormOp::RmsNoParams.apply(row, self.config.rms_norm_eps)
                }
            };
            row.copy_from_slice(&normed);
        }
    }

    /// The batched twin of [`Self::ffn_block_row`]: runs layer
    /// `layer_idx`'s FFN on every row of `hidden_batch`
    /// (`[batch_size, hidden_dim]`, the post-attention residuals) and
    /// adds it back, or does nothing for a layer whose shape has no
    /// FFN.
    ///
    /// The fast paths are tried in the order the prefill body always
    /// tried them -- Metal MoE prefill, the batched dense FFN, the
    /// bucketed batched MoE -- and each answers `None` for a layer or a
    /// batch it does not serve, so the per-row bodies are the floor
    /// every row can reach. gpt-oss runs one position at a time through
    /// its single validated FFN; none of the batched paths knows about
    /// its router bias, expert bias or `swiglu_oai`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ffn_block_batch(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        hidden_batch: &mut [f32],
        batch_size: usize,
        oai: Option<&GptOssLayer>,
        plan: Option<&frink_moe::PlacementPlan>,
        inputs: BranchInputs,
        kernels: BatchedFfnKernels,
        skip: Option<SkipStream<'_>>,
    ) {
        if self.config.layer_shape(layer_idx).ffn_dim == 0 {
            return;
        }
        let hidden_dim = self.config.hidden_dim;
        let config = &self.config;
        let BranchInputs {
            router: operand,
            ffn,
        } = inputs;
        let normed2_batch: Vec<f32> = match ffn {
            FfnInput::PostAttnResidual => {
                self.pre_norm_residual(&layer.moe.norm_weight, hidden_batch, batch_size)
            }
            FfnInput::LayerInput(x) => {
                debug_assert_eq!(x.len(), batch_size * hidden_dim);
                x
            }
        };

        if let Some(oai) = oai {
            for b in 0..batch_size {
                let normed2 = &normed2_batch[b * hidden_dim..(b + 1) * hidden_dim];
                let ffn_out = Self::gpt_oss_ffn(layer, oai, normed2, config, hidden_dim);
                let hidden_row = &mut hidden_batch[b * hidden_dim..(b + 1) * hidden_dim];
                residual_add(hidden_row, &ffn_out, config.residual_scale);
            }
            Self::apply_skip_stream(layer, hidden_batch, skip, batch_size, hidden_dim);
            self.apply_loop_norm(layer_idx, hidden_batch, batch_size);
            return;
        }

        let dense = Self::is_dense_layer(layer);
        // Skip the batched router matmul entirely for a dense layer --
        // there is nothing to route (see `is_dense_layer`'s doc
        // comment), so computing it here just to ignore it below would
        // waste the one matmul this fast path exists to avoid.
        let (router_logits_batch, routed_batch): (Vec<f32>, &[f32]) = match &operand {
            _ if dense => (Vec::new(), normed2_batch.as_slice()),
            RouterOperand::FfnInput => (
                layer.moe.router.apply_batch(&normed2_batch, batch_size),
                normed2_batch.as_slice(),
            ),
            RouterOperand::Precomputed(logits) => (logits.clone(), normed2_batch.as_slice()),
            RouterOperand::BranchInput(x) => {
                (layer.moe.router.apply_batch(x, batch_size), x.as_slice())
            }
        };

        let batched: Option<Vec<f32>> = match kernels {
            BatchedFfnKernels::PerRow => None,
            BatchedFfnKernels::Prefill => {
                #[cfg(feature = "metal")]
                let metal_ffn = if !dense {
                    Self::try_metal_moe_prefill_batch(
                        layer_idx,
                        layer,
                        &normed2_batch,
                        &router_logits_batch,
                        batch_size,
                        hidden_dim,
                        config,
                    )
                } else {
                    None
                };
                #[cfg(not(feature = "metal"))]
                let metal_ffn: Option<Vec<f32>> = None;
                metal_ffn
                    // Dense FFN, batched. Without this the FFN -- the
                    // majority of a dense model's prefill work -- ran one
                    // position at a time while Q/K/V and the router were
                    // already batched, which is why `pp512` measured
                    // about the same as `tg128`.
                    .or_else(|| {
                        Self::dense_ffn_batch(layer_idx, layer, &normed2_batch, batch_size, config)
                    })
                    .or_else(|| {
                        Self::moe_ffn_batch(
                            layer_idx,
                            layer,
                            &normed2_batch,
                            routed_batch,
                            &router_logits_batch,
                            batch_size,
                            config,
                            plan,
                        )
                    })
            }
        };

        if let Some(mut ffn_batch) = batched {
            Self::apply_parallel_sum_scale(layer, &mut ffn_batch);
            Self::apply_down_scale(layer, &mut ffn_batch);
            if let Some(post) = &layer.attn.post_ffn_norm {
                ffn_batch = ffn_batch
                    .chunks(hidden_dim)
                    .flat_map(|row| rms_norm(row, post, config.post_norm_eps()))
                    .collect();
            }
            residual_add(hidden_batch, &ffn_batch, config.residual_scale);
            Self::apply_skip_stream(layer, hidden_batch, skip, batch_size, hidden_dim);
            self.apply_loop_norm(layer_idx, hidden_batch, batch_size);
            return;
        }

        let n_experts = layer.moe.n_experts().max(1);
        for b in 0..batch_size {
            let normed2 = &normed2_batch[b * hidden_dim..(b + 1) * hidden_dim];
            let mut ffn_out = if dense {
                Self::run_ffn_block(
                    layer_idx,
                    layer,
                    normed2,
                    config,
                    hidden_dim,
                    plan,
                    RouterOperand::FfnInput,
                )
            } else {
                let router_logits = &router_logits_batch[b * n_experts..(b + 1) * n_experts];
                Self::combine_ffn_outputs_for_position(
                    layer_idx,
                    layer,
                    normed2,
                    &routed_batch[b * hidden_dim..(b + 1) * hidden_dim],
                    router_logits,
                    config,
                    hidden_dim,
                    plan,
                )
            };
            Self::apply_parallel_sum_scale(layer, &mut ffn_out);
            Self::apply_down_scale(layer, &mut ffn_out);
            if let Some(post) = &layer.attn.post_ffn_norm {
                ffn_out = rms_norm(&ffn_out, post, config.post_norm_eps());
            }
            let hidden_row = &mut hidden_batch[b * hidden_dim..(b + 1) * hidden_dim];
            residual_add(hidden_row, &ffn_out, config.residual_scale);
        }
        Self::apply_skip_stream(layer, hidden_batch, skip, batch_size, hidden_dim);
        self.apply_loop_norm(layer_idx, hidden_batch, batch_size);
    }
}
