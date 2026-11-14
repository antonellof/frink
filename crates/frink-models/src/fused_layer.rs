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
    /// A routed layer's experts, when `gate`, `up` and `down` are its
    /// one SHARED expert ([`Self::for_routed_layer`]).
    #[cfg(feature = "metal")]
    routed: Option<RoutedParts<'a>>,
}

/// How a routed layer picks its experts, which the MODEL decides: the
/// caller has checked that the device router reproduces the host's
/// (`Decoder::gpu_router_matches_host_routing`).
#[derive(Clone, Copy, Debug)]
pub struct Routing {
    pub top_k: usize,
    /// `norm_topk_prob`.
    pub renormalize: bool,
    /// The caller's name for the layer, handed back with the experts
    /// the device chose (`frink_metal::gdn_branch::RunOut::routed`).
    pub tag: usize,
}

/// The routed half of a [`LayerFfnParts`].
#[cfg(feature = "metal")]
struct RoutedParts<'a> {
    router: &'a WeightMatrix,
    packed: frink_metal::gpu::MoePackedQ4<'a>,
    routing: Routing,
    shared_gate: Option<&'a [f32]>,
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
    routed: Option<RoutedLaunch<'a>>,
}

/// The routed half as launches: the router's matvec and the planes.
#[cfg(feature = "metal")]
struct RoutedLaunch<'a> {
    router: frink_metal::gpu::MatvecLaunch<'a>,
    packed: frink_metal::gpu::MoePackedQ4<'a>,
    routing: Routing,
    shared_gate: Option<&'a [f32]>,
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
            routed: self
                .routed
                .as_ref()
                .map(|r| frink_metal::gdn_moe::RoutedExperts {
                    router: &r.router,
                    packed: &r.packed,
                    top_k: r.routing.top_k,
                    renormalize: r.routing.renormalize,
                    shared_gate: r.shared_gate,
                    tag: r.routing.tag,
                }),
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
        Self::from_layer(layer, rms_eps, config_is_plain, None)
    }

    /// A ROUTED layer with one shared expert (Qwen3-Next, `qwen35moe`):
    /// `gate`, `up` and `down` are the shared expert, and the routed
    /// experts go with them on their packed planes. `None` for every
    /// refusal [`Self::for_layer`] makes, and for routed experts the
    /// device cannot serve: stored rather than resident, no planes, a
    /// plane with no `mul_mv_id` kernel, a router that is not f32.
    #[cfg(feature = "metal")]
    pub fn for_routed_layer(
        layer: &'a LayerWeights,
        rms_eps: f32,
        config_is_plain: bool,
        routing: Routing,
    ) -> Option<Self> {
        Self::from_layer(layer, rms_eps, config_is_plain, Some(routing))
    }

    /// The ONE body both constructors share, so a refusal added for one
    /// is a refusal for the other.
    fn from_layer(
        layer: &'a LayerWeights,
        rms_eps: f32,
        config_is_plain: bool,
        routing: Option<Routing>,
    ) -> Option<Self> {
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
            // The packed routed planes: what a routed layer's experts
            // run from, and nothing a dense layer has.
            #[cfg(feature = "metal")]
            packed_q4,
        } = &layer.moe;
        if exp_probs_bias.is_some()
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
        // A stored backing is the out-of-core path and has no business
        // in a per-token fused launch, routed or not.
        let ExpertBacking::Resident(resident) = experts else {
            return None;
        };
        // NOT recorded here: this is a predicate, and a layer that
        // passes it can still fall back when a launch fails, which
        // would then record the expert twice. The caller records once,
        // after the fused layer has actually run.
        let Some(routing) = routing else {
            // Exactly one resident expert, which is what
            // `is_dense_layer` means, and nothing routed beside it.
            #[cfg(feature = "metal")]
            if packed_q4.is_some() {
                return None;
            }
            if resident.len() != 1 || !shared_experts.is_empty() || shared_expert_gate.is_some() {
                return None;
            }
            let ex = &resident[0];
            return Some(Self {
                norm,
                norm_eps: rms_eps,
                gate: &ex.gate,
                up: &ex.up,
                down: &ex.down,
                #[cfg(feature = "metal")]
                routed: None,
            });
        };
        #[cfg(not(feature = "metal"))]
        {
            let _ = (routing, resident, shared_experts, shared_expert_gate);
            None
        }
        #[cfg(feature = "metal")]
        {
            // Routed experts and exactly ONE shared expert: the dense
            // tail runs the shared one, the planes the rest.
            let [shared] = shared_experts.as_slice() else {
                return None;
            };
            let packed = packed_q4.as_ref()?.view();
            if resident.len() < 2
                || routing.top_k == 0
                || routing.top_k > resident.len()
                || !frink_metal::gpu::moe_packed_mul_mv_id_supported(
                    packed.gate_kind,
                    packed.up_kind,
                    packed.down_kind,
                )
                || !matches!(layer.moe.router, WeightMatrix::F32(_))
            {
                return None;
            }
            Some(Self {
                norm,
                norm_eps: rms_eps,
                gate: &shared.gate,
                up: &shared.up,
                down: &shared.down,
                routed: Some(RoutedParts {
                    router: &layer.moe.router,
                    packed,
                    routing,
                    shared_gate: shared_expert_gate.as_deref(),
                }),
            })
        }
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
            #[cfg(feature = "metal")]
            routed: None,
        }
    }

    /// [`Self::from_parts`] with routed experts beside the shared one,
    /// for the test that pins the routed tail against the host MoE.
    #[cfg(all(test, feature = "metal"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts_routed(
        norm: &'a [f32],
        norm_eps: f32,
        gate: &'a WeightMatrix,
        up: &'a WeightMatrix,
        down: &'a WeightMatrix,
        router: &'a WeightMatrix,
        packed: frink_metal::gpu::MoePackedQ4<'a>,
        routing: Routing,
        shared_gate: Option<&'a [f32]>,
    ) -> Self {
        Self {
            norm,
            norm_eps,
            gate,
            up,
            down,
            routed: Some(RoutedParts {
                router,
                packed,
                routing,
                shared_gate,
            }),
        }
    }

    /// The Metal launches for these weights, or `None` when any of the
    /// three has no kernel for its storage.
    #[cfg(feature = "metal")]
    pub fn launches(&self) -> Option<LayerFfnLaunches<'a>> {
        let (gate, up, down, fold_x, fold_act) =
            self.launches_with(crate::metal_launch::matvec, |l| l.rows)?;
        let routed = match &self.routed {
            None => None,
            Some(r) => Some(RoutedLaunch {
                router: crate::metal_launch::matvec(r.router)?,
                packed: frink_metal::gpu::MoePackedQ4 { ..r.packed },
                routing: r.routing,
                shared_gate: r.shared_gate,
            }),
        };
        Some(LayerFfnLaunches {
            norm: self.norm,
            norm_eps: self.norm_eps,
            gate,
            up,
            down,
            fold_x,
            fold_act,
            routed,
        })
    }

    /// The same three as GEMMs, for a prefill batch
    /// (`frink_metal::gdn_prefill`). The norm epsilon is the layer's,
    /// which the prefill run takes from the layer it belongs to.
    #[cfg(feature = "metal")]
    pub fn prefill_launches(&self) -> Option<frink_metal::gdn_prefill::PrefillFfn<'a>> {
        let (gate, up, down, fold_x, fold_act) =
            self.launches_with(WeightMatrix::mul_mm_sg_launch, |l| l.rows)?;
        let routed = match &self.routed {
            None => None,
            Some(r) => {
                let WeightMatrix::F32(router) = r.router else {
                    return None;
                };
                let moe = frink_metal::gpu::PrefillMoeMetal {
                    router_w: &router.data,
                    top_k: r.routing.top_k,
                    renormalize: r.routing.renormalize,
                    packed: frink_metal::gpu::MoePackedQ4 { ..r.packed },
                    oai: None,
                };
                if !moe.is_supported() {
                    return None;
                }
                Some(frink_metal::gdn_prefill::PrefillRouted {
                    moe,
                    shared_gate: r.shared_gate,
                })
            }
        };
        Some(frink_metal::gdn_prefill::PrefillFfn {
            norm: self.norm,
            gate,
            up,
            down,
            fold_x,
            fold_act,
            routed,
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

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::{LayerFfnParts, Routing};
    use frink_core::weight_matrix::{QuantKind, WeightBytes};
    use frink_core::{Tensor, WeightMatrix};
    use frink_moe::{ExpertWeights, GluAct};

    type Expert = (WeightMatrix, WeightMatrix, WeightMatrix);

    /// One routed layer with a shared expert, and what the host MoE
    /// makes of `rows` tokens through `wo` and it.
    struct Case {
        hidden: usize,
        n_exp: usize,
        ffn: usize,
        top_k: usize,
        eps: f32,
        wo: WeightMatrix,
        router: WeightMatrix,
        ffn_norm: Vec<f32>,
        shared_gate: Vec<f32>,
        shared: Expert,
        routed: Vec<Expert>,
        branch: Vec<f32>,
        residual: Vec<f32>,
        want: Vec<f32>,
        /// The host's experts per row, sorted.
        ids: Vec<Vec<usize>>,
    }

    /// The host body: `wo`, the residual, the FFN norm, softmax top-k
    /// over the router, the renormalised weighted sum of the chosen
    /// experts, and the shared expert scaled by `sigmoid(gate . x)`
    /// (`qwen3next.cpp:521-539`). Experts dot against the dequantized
    /// Q8_0 rows, the arithmetic the device kernels do.
    fn case(rows: usize) -> Case {
        let (hidden, branch_dim, ffn, n_exp, top_k) = (256usize, 128usize, 64usize, 6usize, 4usize);
        let eps = 1e-6f32;
        let mut seed = 31_337u32;
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5) * scale
                })
                .collect()
        };
        let f32m = |rows: usize, cols: usize, v: Vec<f32>| {
            WeightMatrix::F32(Tensor::new(v, vec![rows, cols]))
        };
        let q8 = |rows: usize, cols: usize, v: &[f32]| WeightMatrix::Quantized {
            data: WeightBytes::Owned(frink_quant::quantize_q8_0(v)),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        let deq = |m: &WeightMatrix| {
            let (r, c) = (m.rows(), m.cols());
            f32m(r, c, (0..r).flat_map(|i| m.dequant_row(i)).collect())
        };
        // Quantized, so the prefill run has a GEMM for it.
        let wo = q8(hidden, branch_dim, &rnd(hidden * branch_dim, 0.2));
        let wo_host = deq(&wo);
        let router = f32m(n_exp, hidden, rnd(n_exp * hidden, 2.0));
        let ffn_norm: Vec<f32> = rnd(hidden, 0.5).iter().map(|x| 1.0 + x).collect();
        let shared_gate = rnd(hidden, 0.5);
        let mut expert = || -> Expert {
            (
                q8(ffn, hidden, &rnd(ffn * hidden, 0.5)),
                q8(ffn, hidden, &rnd(ffn * hidden, 0.5)),
                q8(hidden, ffn, &rnd(hidden * ffn, 0.5)),
            )
        };
        let shared = expert();
        let routed: Vec<Expert> = (0..n_exp).map(|_| expert()).collect();
        let branch = rnd(rows * branch_dim, 1.0);
        let residual = rnd(rows * hidden, 1.0);

        let run = |e: &Expert, x: &[f32]| {
            frink_moe::run_expert(
                x,
                &ExpertWeights {
                    gate: deq(&e.0),
                    up: deq(&e.1),
                    down: deq(&e.2),
                },
                GluAct::Swiglu,
            )
        };
        let mut want = Vec::with_capacity(rows * hidden);
        let mut ids = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut h: Vec<f32> = residual[r * hidden..(r + 1) * hidden]
                .iter()
                .zip(wo_host.apply(&branch[r * branch_dim..(r + 1) * branch_dim]))
                .map(|(a, b)| a + b)
                .collect();
            let normed = frink_core::rms_norm(&h, &ffn_norm, eps);
            let decision = frink_moe::route_top_k(
                &router.apply(&normed),
                top_k,
                frink_moe::GatingFunction::Softmax,
                true,
            );
            let logit: f32 = shared_gate.iter().zip(&normed).map(|(g, x)| g * x).sum();
            let scale = 1.0 / (1.0 + (-logit).exp());
            for (w, s) in h.iter_mut().zip(run(&shared, &normed)) {
                *w += scale * s;
            }
            for (&id, &wt) in decision.expert_ids.iter().zip(&decision.weights) {
                for (w, o) in h.iter_mut().zip(run(&routed[id], &normed)) {
                    *w += wt * o;
                }
            }
            want.extend(h);
            let mut chosen = decision.expert_ids;
            chosen.sort_unstable();
            ids.push(chosen);
        }
        Case {
            hidden,
            n_exp,
            ffn,
            top_k,
            eps,
            wo,
            router,
            ffn_norm,
            shared_gate,
            shared,
            routed,
            branch,
            residual,
            want,
            ids,
        }
    }

    /// The experts' three planes, contiguous, as the loader packs them.
    fn planes(c: &Case) -> [Vec<u8>; 3] {
        let plane = |pick: fn(&Expert) -> &WeightMatrix| -> Vec<u8> {
            c.routed
                .iter()
                .flat_map(|e| match pick(e) {
                    WeightMatrix::Quantized { data, .. } => data.as_slice().to_vec(),
                    _ => unreachable!("quantized above"),
                })
                .collect()
        };
        [plane(|e| &e.0), plane(|e| &e.1), plane(|e| &e.2)]
    }

    fn parts<'a>(c: &'a Case, p: &'a [Vec<u8>; 3]) -> LayerFfnParts<'a> {
        let packed = frink_metal::gpu::MoePackedQ4 {
            gate: &p[0],
            up: &p[1],
            down: &p[2],
            gate_stride: p[0].len() / c.n_exp,
            up_stride: p[1].len() / c.n_exp,
            down_stride: p[2].len() / c.n_exp,
            n_experts: c.n_exp,
            ffn_rows: c.ffn,
            hidden_rows: c.hidden,
            gate_row_bytes: p[0].len() / c.n_exp / c.ffn,
            down_row_bytes: p[2].len() / c.n_exp / c.hidden,
            gate_kind: "Q8_0",
            up_kind: "Q8_0",
            down_kind: "Q8_0",
        };
        LayerFfnParts::from_parts_routed(
            &c.ffn_norm,
            c.eps,
            &c.shared.0,
            &c.shared.1,
            &c.shared.2,
            &c.router,
            packed,
            Routing {
                top_k: c.top_k,
                renormalize: true,
                tag: 7,
            },
            Some(&c.shared_gate),
        )
    }

    fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1.0);
        for (i, (d, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (d - w).abs() <= tol * scale,
                "{what}[{i}]: device {d} host {w}"
            );
        }
    }

    /// A routed layer's tail in a DECODE run is the host MoE, and the
    /// run hands back the experts it chose, which are the host's.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn a_routed_tail_in_a_run_matches_the_host_moe() {
        let c = case(1);
        let p = planes(&c);
        let parts = parts(&c, &p);
        let launches = parts.launches().expect("every plane has a kernel");
        let out_proj = crate::metal_launch::matvec(&c.wo).expect("Q8_0 has a matvec");
        let mut run = frink_metal::gdn_branch::GdnRun::start(&c.residual).expect("a run");
        run.attn_tail(&out_proj, None, &launches.as_metal(), &c.branch)
            .expect("the tail encodes");
        let out = run.finish_with_head().expect("the run completes");
        let (tag, mut chosen) = out.routed.first().expect("one routed tail").clone();
        chosen.sort_unstable();
        assert_eq!(tag, 7, "the tag comes back");
        assert_eq!(chosen, c.ids[0], "the device chose the host's experts");
        close("out", &out.hidden, &c.want, 2e-4);
    }

    /// The same layer as a PREFILL run takes it: the router GEMM, the
    /// batched top-k, `mul_mm_id` over f16 activations, the per-row
    /// shared gate. Forty rows, so several experts gather tokens.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn a_routed_tail_in_a_prefill_run_matches_the_host_moe() {
        let rows = 40;
        let c = case(rows);
        let p = planes(&c);
        let parts = parts(&c, &p);
        let tail = frink_metal::gdn_prefill::PrefillAttnTail {
            branch: &c.branch,
            o_proj: c.wo.mul_mm_sg_launch().expect("Q8_0 has a GEMM"),
            fold: None,
            ffn: parts.prefill_launches().expect("every plane has a kernel"),
            eps: c.eps,
        };
        // SAFETY: no recurrent layer, so no state pointer is handed over.
        let got = unsafe {
            frink_metal::gdn_prefill::launch_gdn_prefill_run(
                &c.residual,
                rows,
                0,
                Some(&tail),
                &mut [],
            )
        }
        .expect("the run launches");
        // The GEMMs round their activation tiles to half.
        close("out", &got, &c.want, 3e-3);
    }
}
