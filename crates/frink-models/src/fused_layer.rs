//! When a whole decoder layer can run in ONE Metal submission, and the
//! weights it takes.
//!
//! # Why
//!
//! `1ebda4fc:docs/plans/gdn-resident-state.md` prices a Bonsai decode token: GPU
//! 79 ms, submission latency 35 ms, host about 20 ms. The GPU work is
//! already faster than the reference's whole token and the 0.15 ms a
//! submission costs is the OS wake-up, so the gap is the COUNT: 192 a
//! token, three per layer. `crate::gdn`'s fused branch removed the host
//! step inside a recurrent layer without changing that count, because
//! it rides in a submission that already existed. This is the step that
//! changes the count, and by the same arithmetic it is the last one:
//! one submission a layer is 9.6 ms of latency against today's 35.
//!
//! # What it refuses, and why the list is written out
//!
//! The fused path implements exactly `x + ffn(rms_norm(x + branch))`
//! with a SwiGLU FFN of one dense expert. Everything else a layer can
//! carry -- a routed FFN, a shared expert, an inner norm, projection
//! biases, a post-attention or post-FFN norm, a residual scale, a skip
//! stream, a parallel residual, a down scale, a gpt-oss block, an
//! expert placement plan -- is applied by the host bodies and by no
//! kernel here.
//!
//! So [`LayerFfnParts::for_layer`] DESTRUCTURES `MoeWeights`
//! exhaustively, with no `..`. A field added to that struct does not
//! compile until somebody says whether this path serves it, which is
//! the only thing that reliably stops a fused path from quietly
//! dropping a model feature: this file's own history is eight of them,
//! one at a time.

use frink_core::WeightMatrix;

use crate::decoder::{ExpertBacking, LayerWeights, MoeWeights};
use crate::norm::NormOp;

/// The dense half of a layer, as weights rather than as launches.
///
/// Built once per layer per token from [`Self::for_layer`], which is
/// the only constructor, so the refusals cannot be bypassed by
/// assembling one by hand.
pub struct LayerFfnParts<'a> {
    norm: &'a [f32],
    norm_eps: f32,
    gate: &'a WeightMatrix,
    up: &'a WeightMatrix,
    down: &'a WeightMatrix,
}

/// The same, as Metal launches plus the rotations their inputs need.
///
/// Separate because a launch borrows the matrix it describes, so the
/// launches have to outlive the call that encodes them while the parts
/// above are cheap to build and throw away.
#[cfg(feature = "metal")]
pub struct LayerFfnLaunches<'a> {
    norm: &'a [f32],
    norm_eps: f32,
    gate: frink_metal::gpu::MatvecLaunch<'a>,
    up: frink_metal::gpu::MatvecLaunch<'a>,
    down: frink_metal::gpu::MatvecLaunch<'a>,
    fold_x: Option<frink_metal::hadamard::FoldPlan<'a>>,
    fold_act: Option<frink_metal::hadamard::FoldPlan<'a>>,
}

#[cfg(feature = "metal")]
impl<'a> LayerFfnLaunches<'a> {
    pub fn as_metal(&'a self) -> frink_metal::gdn_branch::LayerFfn<'a> {
        frink_metal::gdn_branch::LayerFfn {
            norm: self.norm,
            norm_eps: self.norm_eps,
            gate: &self.gate,
            up: &self.up,
            down: &self.down,
            fold_x: self.fold_x.as_ref(),
            fold_act: self.fold_act.as_ref(),
        }
    }
}

/// The head of a layer as weights: the input norm and the four
/// projections the recurrent branch reads.
pub struct LayerHeadParts<'a> {
    norm: &'a [f32],
    norm_eps: f32,
    qkv: &'a WeightMatrix,
    z: &'a WeightMatrix,
    beta: &'a WeightMatrix,
    alpha: &'a WeightMatrix,
}

/// The same, as Metal launches plus the one rotation `qkv` and `z`
/// share.
#[cfg(feature = "metal")]
pub struct LayerHeadLaunches<'a> {
    norm: &'a [f32],
    norm_eps: f32,
    qkv: frink_metal::gpu::MatvecLaunch<'a>,
    z: frink_metal::gpu::MatvecLaunch<'a>,
    beta: frink_metal::gpu::MatvecLaunch<'a>,
    alpha: frink_metal::gpu::MatvecLaunch<'a>,
    fold_x: Option<frink_metal::hadamard::FoldPlan<'a>>,
}

#[cfg(feature = "metal")]
impl<'a> LayerHeadLaunches<'a> {
    pub fn as_metal(&'a self) -> frink_metal::gdn_branch::LayerHeadIn<'a> {
        frink_metal::gdn_branch::LayerHeadIn {
            norm: self.norm,
            norm_eps: self.norm_eps,
            qkv: &self.qkv,
            z: &self.z,
            beta: &self.beta,
            alpha: &self.alpha,
            fold_x: self.fold_x.as_ref(),
        }
    }
}

impl<'a> LayerHeadParts<'a> {
    /// The head of a recurrent layer, given the layer's `attn_norm` and
    /// the block's four projections.
    ///
    /// `None` is not expressible here -- every field is required -- so
    /// the refusals all live in [`Self::launches`], where they are
    /// questions about storage and basis rather than about shape.
    #[allow(clippy::too_many_arguments)]
    pub fn for_block(
        norm: &'a [f32],
        norm_eps: f32,
        qkv: &'a WeightMatrix,
        z: &'a WeightMatrix,
        beta: &'a WeightMatrix,
        alpha: &'a WeightMatrix,
    ) -> Option<Self> {
        Some(Self {
            norm,
            norm_eps,
            qkv,
            z,
            beta,
            alpha,
        })
    }

    /// The Metal launches, or `None` when any matrix has no kernel for
    /// its storage or the four disagree about their input basis.
    #[cfg(feature = "metal")]
    pub fn launches(&self) -> Option<LayerHeadLaunches<'a>> {
        let hidden = self.norm.len();
        let (qkv_base, qkv_fold) = self.qkv.launch_parts();
        let (z_base, z_fold) = self.z.launch_parts();
        let (beta_base, beta_fold) = self.beta.launch_parts();
        let (alpha_base, alpha_fold) = self.alpha.launch_parts();
        // The kernel applies ONE rotation, to the buffer `qkv` and `z`
        // read, AFTER the two gate projections have read it unrotated.
        // So the gates must be unfolded and the pair must share a fold,
        // and a checkpoint that is not laid out that way takes the host
        // path rather than being reordered here.
        if beta_fold.is_some() || alpha_fold.is_some() {
            return None;
        }
        let fold_x = match (qkv_fold, z_fold) {
            (None, None) => None,
            (Some(a), Some(b)) if std::sync::Arc::ptr_eq(a, b) => Some(a.metal_plan(hidden)?),
            _ => return None,
        };
        Some(LayerHeadLaunches {
            norm: self.norm,
            norm_eps: self.norm_eps,
            qkv: crate::metal_launch::matvec(qkv_base)?,
            z: crate::metal_launch::matvec(z_base)?,
            beta: crate::metal_launch::matvec(beta_base)?,
            alpha: crate::metal_launch::matvec(alpha_base)?,
            fold_x,
        })
    }
}

impl<'a> LayerFfnParts<'a> {
    /// This layer's dense FFN, or `None` when the layer carries
    /// anything the fused path does not implement.
    ///
    /// `config_is_plain` is the caller's half of the question -- a
    /// residual scale, a skip stream or a non-SwiGLU activation are
    /// model facts rather than layer weights -- and is passed in so
    /// both halves are answered at one call.
    pub fn for_layer(layer: &'a LayerWeights, rms_eps: f32, config_is_plain: bool) -> Option<Self> {
        if !config_is_plain {
            return None;
        }
        // No `..`: a field added here has to be answered before this
        // compiles again.
        let MoeWeights {
            router: _,
            experts,
            shared_experts,
            shared_expert_gate,
            norm_weight,
            exp_probs_bias,
            ffn_sub_norm,
            down_scale,
            dense_bias,
            exps_norm,
            parallel_sum_scale,
            parallel,
            // Telemetry, and the fused path keeps it: the host body
            // records expert 0 for a dense layer every token, so a
            // fused layer that did not would make the hotness counters
            // depend on which backend ran.
            activation_counts: _,
            // The packed routed planes. A dense layer has none, and a
            // layer that has them is routed, which this refuses below
            // through its expert count anyway; naming it here is what
            // makes that an answer rather than an omission.
            #[cfg(feature = "metal")]
            packed_q4,
        } = &layer.moe;
        #[cfg(feature = "metal")]
        if packed_q4.is_some() {
            return None;
        }
        if !shared_experts.is_empty()
            || shared_expert_gate.is_some()
            || exp_probs_bias.is_some()
            || ffn_sub_norm.is_some()
            || down_scale.is_some()
            || dense_bias.is_some()
            || exps_norm.is_some()
            || parallel_sum_scale.is_some()
            || parallel.is_some()
            || layer.attn.post_attn_norm.is_some()
            || layer.attn.post_ffn_norm.is_some()
        {
            return None;
        }
        // The pre-FFN norm has to be a plain weighted RMS: the kernel
        // is `encode_rms_norm` and nothing else.
        let norm = match norm_weight {
            NormOp::Rms(w) => w.as_slice(),
            _ => return None,
        };
        // Exactly one resident expert, which is what `is_dense_layer`
        // means; a stored backing is the out-of-core path and has no
        // business in a per-token fused launch.
        let ex = match experts {
            ExpertBacking::Resident(v) if v.len() == 1 => &v[0],
            _ => return None,
        };
        // NOT recorded here: this is a predicate, and a layer that
        // passes it can still fall back when a launch fails, which
        // would then record the expert twice. The caller records once,
        // after the fused layer has actually run.
        Some(Self {
            norm,
            norm_eps: rms_eps,
            gate: &ex.gate,
            up: &ex.up,
            down: &ex.down,
        })
    }

    /// The same parts assembled directly, for the test that pins the
    /// fused layer against the host bodies.
    ///
    /// Test-only on purpose: [`Self::for_layer`] is the ONE production
    /// constructor, and it is where the refusals live. A second
    /// production path into this struct would be a second place that
    /// has to remember them.
    #[cfg(test)]
    pub(crate) fn from_parts(
        norm: &'a [f32],
        norm_eps: f32,
        gate: &'a WeightMatrix,
        up: &'a WeightMatrix,
        down: &'a WeightMatrix,
    ) -> Self {
        Self {
            norm,
            norm_eps,
            gate,
            up,
            down,
        }
    }

    /// The Metal launches for these weights, or `None` when any of the
    /// three has no kernel for its storage.
    #[cfg(feature = "metal")]
    pub fn launches(&self) -> Option<LayerFfnLaunches<'a>> {
        let (gate, up, down, fold_x, fold_act) =
            self.launches_with(crate::metal_launch::matvec, |l| l.rows)?;
        Some(LayerFfnLaunches {
            norm: self.norm,
            norm_eps: self.norm_eps,
            gate,
            up,
            down,
            fold_x,
            fold_act,
        })
    }

    /// The same three as GEMMs, for a prefill batch
    /// (`frink_metal::gdn_prefill`). The norm epsilon is the layer's,
    /// which the prefill run takes from the layer it belongs to.
    #[cfg(feature = "metal")]
    pub fn prefill_launches(&self) -> Option<frink_metal::gdn_prefill::PrefillFfn<'a>> {
        let (gate, up, down, fold_x, fold_act) =
            self.launches_with(WeightMatrix::mul_mm_sg_launch, |l| l.rows)?;
        Some(frink_metal::gdn_prefill::PrefillFfn {
            norm: self.norm,
            gate,
            up,
            down,
            fold_x,
            fold_act,
        })
    }

    /// Gate, up and down as `launch` builds them, with the rotations
    /// their inputs need. ONE body for the matvec and the GEMM forms,
    /// so the two cannot disagree about which folds a layer may have.
    #[cfg(feature = "metal")]
    #[allow(clippy::type_complexity)]
    fn launches_with<L>(
        &self,
        launch: impl Fn(&'a WeightMatrix) -> Option<L>,
        rows: impl Fn(&L) -> usize,
    ) -> Option<(
        L,
        L,
        L,
        Option<frink_metal::hadamard::FoldPlan<'a>>,
        Option<frink_metal::hadamard::FoldPlan<'a>>,
    )> {
        let hidden = self.norm.len();
        let one = |m: &'a WeightMatrix, width: usize| {
            let (base, fold) = m.launch_parts();
            let launched = launch(base)?;
            let plan = match fold {
                None => None,
                Some(f) => Some(f.metal_plan(width)?),
            };
            Some((launched, plan))
        };
        let (gate, fold_x) = one(self.gate, hidden)?;
        let (up, up_fold) = one(self.up, hidden)?;
        let (down, fold_act) = one(self.down, rows(&gate))?;
        // Gate and up read the SAME vector, so one rotation serves both
        // and two different ones cannot be expressed: a checkpoint
        // whose gate and up disagree about their input basis is not
        // this shape.
        let same = match (&fold_x, &up_fold) {
            (None, None) => true,
            (Some(a), Some(b)) => a.block == b.block && std::ptr::eq(a.signs?, b.signs?),
            _ => false,
        };
        if !same {
            return None;
        }
        Some((gate, up, down, fold_x, fold_act))
    }
}
