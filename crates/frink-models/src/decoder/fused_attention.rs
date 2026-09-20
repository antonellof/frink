//! An ATTENTION layer's tail in one Metal submission.
//!
//! The recurrent layers of a hybrid are one submission each
//! (`crate::decoder::fused_recurrent`); the attention layers between
//! them still cost three, because their attention runs on the host.
//! Two of those three are `wo` and the FFN, with nothing but a vector
//! add and a norm in between, so they are one.
//!
//! On Bonsai that is sixteen layers of sixty-four, and the ledger
//! prices a wait at about 0.17 ms, so it is sixteen waits a token.
//! `docs/plans/gdn-resident-state.md` carries the table this is a row
//! of, and the floor it stops at.
//!
//! What it refuses is everything the host tail applies that no kernel
//! here does: the `{1}` scale companion on `wo`'s output, the output
//! bias, the value scale, the post-attention norm, and the parallel
//! Mamba-2 branch that would be summed into the projection. The FFN
//! half is `crate::fused_layer`'s, which destructures `MoeWeights`
//! exhaustively.

use super::{Decoder, LayerWeights};
use crate::fused_layer::LayerFfnParts;
use crate::layer_shapes::AttnShape;

impl Decoder {
    /// Whether layer `l`'s tail can be fused, asked BEFORE the
    /// attention runs so the block knows whether to apply `wo`.
    ///
    /// The same questions [`Self::fused_attention_tail`] asks, minus
    /// the ones that need the branch vector, so the two cannot
    /// disagree about which layers defer: a layer that deferred and
    /// then found itself refused would have to project on the host
    /// anyway, which the caller does.
    #[cfg(feature = "metal")]
    pub(crate) fn fused_attention_tail_eligible(&self, l: usize, layer: &LayerWeights) -> bool {
        self.fused_attention_refusals(l, layer).is_some()
    }

    /// `Some(())` when nothing refuses. One body, two callers.
    #[cfg(feature = "metal")]
    fn fused_attention_refusals(&self, l: usize, layer: &LayerWeights) -> Option<()> {
        if !frink_core::weight_matrix::metal_dense_enabled() {
            return None;
        }
        let shape = self.config.layer_shape(l);
        if !matches!(shape.attention, AttnShape::Gqa { .. })
            || shape.ffn_dim == 0
            || self.config.residual_scale.is_some()
            || self.config.normed_residual_scale.is_some()
            || self.config.skip_stream
            || self.gpt_oss.is_some()
            || !self.config.layer_ffn_acts(l).all_swiglu()
            // Everything the host tail does AFTER `wo`, which this
            // launch does not: `crate::weight_scales`,
            // `crate::proj_bias`, `crate::attn_value_scale`, the
            // post-attention norm, and `crate::mamba2`'s parallel
            // branch summed into the projection.
            || layer.attn.o_scale.is_some()
            || layer.attn.o_bias.is_some()
            || self.config.attn_value_scale.is_some()
            || layer.attn.post_attn_norm.is_some()
            || layer.attn.ssm.is_some()
        {
            return None;
        }
        LayerFfnParts::for_layer(layer, self.config.rms_norm_eps, true).map(|_| ())
    }

    /// The Q/K/V launches for an attention layer, when all three have
    /// Metal kernels and agree about their input basis.
    ///
    /// Used to encode the layer's projections at the END of the
    /// preceding recurrent run, where they cost no wait of their own:
    /// the layer reads `attn_norm(hidden)` and the run has just
    /// finished writing `hidden`.
    #[cfg(feature = "metal")]
    #[allow(clippy::type_complexity)]
    pub(crate) fn attn_head_launch<'a>(
        &self,
        l: usize,
        layer: &'a LayerWeights,
    ) -> Option<(
        &'a [f32],
        frink_metal::gpu::MatvecLaunch<'a>,
        frink_metal::gpu::MatvecLaunch<'a>,
        frink_metal::gpu::MatvecLaunch<'a>,
        Option<frink_metal::hadamard::FoldPlan<'a>>,
    )> {
        let shape = self.config.layer_shape(l);
        if !matches!(shape.attention, AttnShape::Gqa { .. })
            // The three projections' biases are applied by the host
            // AFTER this, so they are fine; what is not is a layer
            // whose pre-norm is anything but a plain weighted RMS,
            // which is the only norm this encodes.
            || layer.attn.q_bias.is_some() && false
        {
            return None;
        }
        let norm = layer.attn.norm_weight.rms_weights()?;
        let hidden = norm.len();
        let one = |m: &'a frink_core::WeightMatrix| {
            let (base, fold) = m.launch_parts();
            Some((crate::metal_launch::matvec(base)?, fold))
        };
        let (q, q_fold) = one(&layer.attn.q_proj)?;
        let (k, k_fold) = one(&layer.attn.k_proj)?;
        let (v, v_fold) = one(&layer.attn.v_proj)?;
        // One rotation for all three, since they read the same vector
        // and the kernel applies it once.
        let same = |a: Option<
            &std::sync::Arc<frink_core::weight_matrix::hadamard::HadamardFold>,
        >,
                    b: Option<
            &std::sync::Arc<frink_core::weight_matrix::hadamard::HadamardFold>,
        >| {
            match (a, b) {
                (None, None) => true,
                (Some(x), Some(y)) => std::sync::Arc::ptr_eq(x, y),
                _ => false,
            }
        };
        if !same(q_fold, k_fold) || !same(q_fold, v_fold) {
            return None;
        }
        let fold_x = match q_fold {
            None => None,
            Some(f) => Some(f.metal_plan(hidden)?),
        };
        Some((norm, q, k, v, fold_x))
    }

    /// The launch description for this layer's tail, or `None` when
    /// anything refuses.
    ///
    /// ONE builder, because the tail runs two ways -- in a command
    /// buffer of its own ([`Self::fused_attention_tail`]) and at the
    /// head of a recurrent run
    /// ([`Self::fused_attention_tail_then_run`]) -- and a second place
    /// assembling it would be a second place that has to remember the
    /// fold width and the refusals.
    #[cfg(feature = "metal")]
    #[allow(clippy::type_complexity)]
    pub(crate) fn attn_tail_launch<'a>(
        &self,
        l: usize,
        layer: &'a LayerWeights,
        branch: &[f32],
    ) -> Option<(
        frink_metal::gpu::MatvecLaunch<'a>,
        Option<frink_metal::hadamard::FoldPlan<'a>>,
        LayerFfnParts<'a>,
        crate::fused_layer::LayerFfnLaunches<'a>,
    )> {
        self.fused_attention_refusals(l, layer)?;
        let (base, fold) = layer.attn.o_proj.launch_parts();
        let out_proj = crate::metal_launch::matvec(base)?;
        let fold_branch = match fold {
            None => None,
            Some(f) => Some(f.metal_plan(branch.len())?),
        };
        let ffn = LayerFfnParts::for_layer(layer, self.config.rms_norm_eps, true)?;
        let launches = ffn.launches()?;
        Some((out_proj, fold_branch, ffn, launches))
    }

    /// `residual + ffn(rms_norm(residual + wo(branch)))` in one
    /// submission, or `None` when this layer is not that shape.
    ///
    /// `branch` is the attention output BEFORE `wo`
    /// (`Decoder::attn_branch_rows`).
    #[cfg(feature = "metal")]
    pub(crate) fn fused_attention_tail(
        &self,
        l: usize,
        layer: &LayerWeights,
        branch: &[f32],
        residual: &[f32],
    ) -> Option<Vec<f32>> {
        let (out_proj, fold_branch, _ffn, launches) = self.attn_tail_launch(l, layer, branch)?;
        let out = frink_metal::gdn_branch::launch_attn_tail(
            &out_proj,
            fold_branch.as_ref(),
            &launches.as_metal(),
            branch,
            residual,
        )
        .ok()?;
        // As the host body records it for a dense layer, and only once
        // this has actually run rather than fallen back.
        layer.moe.record_activations_dense();
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use frink_core::{Tensor, WeightMatrix};

    /// The fused attention tail IS the host tail: `wo`, the residual
    /// add, the FFN norm, the SwiGLU FFN and the second residual add.
    ///
    /// The oracle is those five composed here, because the production
    /// path takes the launch and so cannot be its own reference. The
    /// projection carries a real fold, since `wo` is folded on every
    /// Bonsai layer and a rotation that is skipped is invisible
    /// without one.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn the_fused_attention_tail_matches_the_host_tail() {
        use crate::fused_layer::LayerFfnParts;
        use frink_core::weight_matrix::hadamard::{FoldSite, HadamardFold};
        use frink_moe::{ExpertWeights, GluAct};

        let (hidden, branch_dim, ffn_dim) = (64usize, 96usize, 80usize);
        let mut seed = 6_180_339u32;
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5) * scale
                })
                .collect()
        };
        let mat = |rows: usize, cols: usize, v: Vec<f32>| {
            WeightMatrix::F32(Tensor::new(v, vec![rows, cols]))
        };
        let (wo_v, gate_v, up_v, down_v) = (
            rnd(hidden * branch_dim, 1.0),
            rnd(ffn_dim * hidden, 1.0),
            rnd(ffn_dim * hidden, 1.0),
            rnd(hidden * ffn_dim, 1.0),
        );
        let ffn_norm: Vec<f32> = rnd(hidden, 0.5).iter().map(|x| 1.0 + x).collect();
        let branch = rnd(branch_dim, 1.0);
        let residual = rnd(hidden, 1.0);
        let eps = 1e-6f32;

        // `wo` folded, as every Bonsai layer has it.
        let signs: std::sync::Arc<[f32]> = rnd(branch_dim, 2.0)
            .iter()
            .map(|x| if *x < 0.0 { -1.0f32 } else { 1.0 })
            .collect::<Vec<f32>>()
            .into();
        let fold = std::sync::Arc::new(HadamardFold {
            block: 16,
            signs: Some(signs),
            perm: None,
            site: FoldSite::Input,
        });
        let mut wo = mat(hidden, branch_dim, wo_v);
        wo.fold_hadamard(fold.clone());

        // The host tail.
        let projected = wo.apply(&branch);
        let mut host: Vec<f32> = residual
            .iter()
            .zip(projected.iter())
            .map(|(a, b)| a + b)
            .collect();
        let normed = frink_core::rms_norm(&host, &ffn_norm, eps);
        let ffn_out = frink_moe::run_expert(
            &normed,
            &ExpertWeights {
                gate: mat(ffn_dim, hidden, gate_v.clone()),
                up: mat(ffn_dim, hidden, up_v.clone()),
                down: mat(hidden, ffn_dim, down_v.clone()),
            },
            GluAct::Swiglu,
        );
        for (x, f) in host.iter_mut().zip(ffn_out.iter()) {
            *x += f;
        }

        // The fused tail, through the path the decoder takes.
        let (base, wo_fold) = wo.launch_parts();
        let out_proj = crate::metal_launch::matvec(base).expect("F32 has a Metal matvec");
        let plan = wo_fold
            .expect("folded above")
            .metal_plan(branch_dim)
            .expect("the plan fits this width");
        let (g, u, d) = (
            mat(ffn_dim, hidden, gate_v),
            mat(ffn_dim, hidden, up_v),
            mat(hidden, ffn_dim, down_v),
        );
        let parts = LayerFfnParts::from_parts(&ffn_norm, eps, &g, &u, &d);
        let launches = parts.launches().expect("all three have Metal matvecs");
        let device = frink_metal::gdn_branch::launch_attn_tail(
            &out_proj,
            Some(&plan),
            &launches.as_metal(),
            &branch,
            &residual,
        )
        .expect("this shape is one the kernels serve");

        let tol = 2e-4;
        for (i, (x, y)) in device.iter().zip(host.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol * y.abs().max(1.0),
                "out[{i}]: device={x} host={y}"
            );
        }
    }
}
