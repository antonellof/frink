//! A gated delta-net layer as the Metal prefill run takes it
//! (`frink_metal::gdn_prefill`), or `None` and the host bodies run.
//!
//! Beside `crate::gdn` rather than in it because that file is past the
//! size this repo splits at; the refusals are the decode path's
//! (`Gdn::fused_layer`), with the projections as GEMMs.

use frink_core::gdn::HeadMap;
use frink_core::WeightMatrix;

use crate::fused_layer::LayerFfnParts;
use crate::gdn::{BetaAlpha, Gdn};

impl Gdn {
    /// This layer's whole prefill body as one description:
    /// `attn_norm`, the four projections, the branch and the FFN.
    ///
    /// `None` for the shapes the kernels do not serve, each named: the
    /// FUSED beta/alpha spelling (its per-group interleave is host
    /// arithmetic), gate projections that are not plain f32 or that
    /// carry a fold, `qkv` and `z` folded differently (they share one
    /// input), and any projection with no GEMM for its storage.
    pub fn prefill_layer<'a>(
        &'a self,
        attn_norm: &'a [f32],
        rms_eps: f32,
        ffn: &LayerFfnParts<'a>,
    ) -> Option<frink_metal::gdn_prefill::PrefillGdnLayer<'a>> {
        let BetaAlpha::Split { beta, alpha } = &self.beta_alpha else {
            return None;
        };
        let f32_rows = |m: &'a WeightMatrix| match m {
            WeightMatrix::F32(t) => Some(t.data.as_slice()),
            _ => None,
        };
        let h = self.h;
        let hidden = attn_norm.len();
        let (qkv_base, qkv_fold) = self.qkv.launch_parts();
        let (z_base, z_fold) = self.z_proj.launch_parts();
        let fold_x = match (qkv_fold, z_fold) {
            (None, None) => None,
            (Some(a), Some(b)) if std::sync::Arc::ptr_eq(a, b) => Some(a.metal_plan(hidden)?),
            _ => return None,
        };
        let (out_base, out_fold) = self.out_proj.launch_parts();
        let fold_y = match out_fold {
            None => None,
            Some(f) => Some(f.metal_plan(h.value_dim())?),
        };
        let layer = frink_metal::gdn_prefill::PrefillGdnLayer {
            shape: frink_metal::gdn::DeltaShape {
                n_k_heads: h.n_k_heads,
                n_v_heads: h.n_v_heads,
                head_dim: h.head_dim,
                map: match h.map {
                    HeadMap::Tiled => frink_metal::gdn::HeadMapKind::Tiled,
                    HeadMap::Grouped => frink_metal::gdn::HeadMapKind::Grouped,
                },
            },
            head: frink_metal::gdn_head::HeadShape {
                n_k_heads: h.n_k_heads,
                n_v_heads: h.n_v_heads,
                head_dim: h.head_dim,
                d_conv: h.d_conv,
            },
            attn_norm,
            eps: rms_eps,
            qkv: qkv_base.mul_mm_sg_launch()?,
            z: z_base.mul_mm_sg_launch()?,
            fold_x,
            // Unfolded and unquantized, or no launch: the gates read
            // `attn_norm(x)` in the primal basis, at f32.
            beta: f32_rows(beta)?,
            alpha: f32_rows(alpha)?,
            conv1d: &self.conv1d,
            dt_bias: &self.dt_bias,
            a: &self.a,
            ssm_norm: &self.norm,
            out_proj: out_base.mul_mm_sg_launch()?,
            fold_y,
            ffn: ffn.prefill_launches()?,
        };
        layer.is_supported().then_some(layer)
    }
}

#[cfg(test)]
mod tests {
    use crate::fused_layer::LayerFfnParts;
    use crate::gdn::{BetaAlpha, Gdn, GdnHparams};
    use frink_core::gdn::HeadMap;
    use frink_core::matmul::rms_norm;
    use frink_core::recurrent_state::{AlignedF32, RecurrentState};
    use frink_core::weight_matrix::hadamard::{FoldSite, HadamardFold, HeadPerm};
    use frink_core::weight_matrix::{QuantKind, WeightBytes};
    use frink_core::{Tensor, WeightMatrix};
    use std::sync::Arc;

    /// An attention layer's tail and then a recurrent layer, as ONE
    /// prefill run, against the host bodies the decoder runs otherwise:
    /// `wo` and the FFN for the tail, `Gdn::forward_rows` and the FFN for
    /// the layer. Every fold Bonsai carries is here, `ssm_out`'s head
    /// permutation included, and 40 rows end the chunked recurrence on
    /// a short chunk. The run is repeated from the same state and must
    /// give the same bits: a first version did not.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn the_prefill_run_matches_the_host_layers() {
        let h = GdnHparams {
            d_conv: 4,
            head_dim: 128,
            n_k_heads: 2,
            n_v_heads: 4,
            map: HeadMap::Grouped,
        };
        let (n_embd, ffn_dim, branch_w, rows) = (256usize, 512usize, 384usize, 40usize);
        let eps = 1e-6f32;
        let mut seed = 4_242u32;
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5) * scale
                })
                .collect()
        };
        let q8 = |rows: usize, cols: usize, v: Vec<f32>| WeightMatrix::Quantized {
            data: WeightBytes::Owned(frink_quant::quantize_q8_0(&v)),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        let f32m = |rows: usize, cols: usize, v: Vec<f32>| {
            WeightMatrix::F32(Tensor::new(v, vec![rows, cols]))
        };
        let mut sign_vec = |n: usize| -> Arc<[f32]> {
            rnd(n, 2.0)
                .iter()
                .map(|x| if *x < 0.0 { -1.0f32 } else { 1.0 })
                .collect::<Vec<f32>>()
                .into()
        };
        let fold = |block: usize, signs: Arc<[f32]>, perm: Option<HeadPerm>| {
            Arc::new(HadamardFold {
                block,
                signs: Some(signs),
                perm,
                site: FoldSite::Input,
            })
        };
        let fold_x = fold(64, sign_vec(n_embd), None);
        let fold_y = fold(
            256,
            sign_vec(h.value_dim()),
            Some(HeadPerm {
                hd: 128,
                nk: 2,
                rep: 2,
            }),
        );
        let fold_ffn = fold(64, sign_vec(n_embd), None);
        let fold_act = fold(128, sign_vec(ffn_dim), None);
        let fold_branch = fold(128, sign_vec(branch_w), None);

        let (value_dim, conv_dim) = (h.value_dim(), h.conv_dim());
        let mut m = Gdn {
            h,
            qkv: q8(conv_dim, n_embd, rnd(conv_dim * n_embd, 0.2)),
            z_proj: q8(value_dim, n_embd, rnd(value_dim * n_embd, 0.2)),
            conv1d: rnd(h.d_conv * conv_dim, 1.0),
            dt_bias: rnd(h.n_v_heads, 1.0),
            // Stored negative, as every export stores it (`ssm_a`): a
            // positive one grows the state without bound.
            a: rnd(h.n_v_heads, 1.0)
                .iter()
                .map(|x| -(x.abs() + 0.1))
                .collect(),
            beta_alpha: BetaAlpha::Split {
                beta: f32m(h.n_v_heads, n_embd, rnd(h.n_v_heads * n_embd, 0.2)),
                alpha: f32m(h.n_v_heads, n_embd, rnd(h.n_v_heads * n_embd, 0.2)),
            },
            norm: rnd(h.head_dim, 1.0),
            out_proj: q8(n_embd, value_dim, rnd(n_embd * value_dim, 0.2)),
        };
        m.qkv.fold_hadamard(fold_x.clone());
        m.z_proj.fold_hadamard(fold_x);
        m.out_proj.fold_hadamard(fold_y);
        let mut gate = q8(ffn_dim, n_embd, rnd(ffn_dim * n_embd, 0.2));
        let mut up = q8(ffn_dim, n_embd, rnd(ffn_dim * n_embd, 0.2));
        let mut down = q8(n_embd, ffn_dim, rnd(n_embd * ffn_dim, 0.2));
        gate.fold_hadamard(fold_ffn.clone());
        up.fold_hadamard(fold_ffn);
        down.fold_hadamard(fold_act);
        let mut o_proj = q8(n_embd, branch_w, rnd(n_embd * branch_w, 0.2));
        o_proj.fold_hadamard(fold_branch.clone());
        let attn_norm: Vec<f32> = rnd(n_embd, 0.5).iter().map(|x| 1.0 + x).collect();
        let ffn_norm: Vec<f32> = rnd(n_embd, 0.5).iter().map(|x| 1.0 + x).collect();
        let (conv_len, ssm_len) = h.state_floats();
        let (conv0, ssm0) = (rnd(conv_len, 1.0), rnd(ssm_len, 0.5));
        let state0 = || {
            let mut conv = AlignedF32::zeros(conv_len);
            conv.copy_from_slice(&conv0);
            let mut ssm = AlignedF32::zeros(ssm_len);
            ssm.copy_from_slice(&ssm0);
            RecurrentState { conv, ssm }
        };
        let hidden = rnd(rows * n_embd, 1.0);
        let branch = rnd(rows * branch_w, 1.0);

        // The host bodies, in the decoder's order, batched as the
        // prefill FFN batches them (the GEMM rounds its activation tiles
        // to half, which a per-row matvec does not).
        let ffn_host = |x: &mut [f32]| {
            let n: Vec<f32> = x
                .chunks(n_embd)
                .flat_map(|row| rms_norm(row, &ffn_norm, eps))
                .collect();
            let g = gate.apply_batch(&n, rows);
            let u = up.apply_batch(&n, rows);
            let act: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(a, b)| a / (1.0 + (-a).exp()) * b)
                .collect();
            for (r, d) in x.iter_mut().zip(down.apply_batch(&act, rows)) {
                *r += d;
            }
        };
        let mut want = hidden.clone();
        for (r, p) in want.iter_mut().zip(o_proj.apply_batch(&branch, rows)) {
            *r += p;
        }
        ffn_host(&mut want);
        let normed: Vec<f32> = want
            .chunks(n_embd)
            .flat_map(|row| rms_norm(row, &attn_norm, eps))
            .collect();
        let mut host_state = state0();
        let out = m.forward_rows(&normed, rows, &mut host_state, eps);
        for (r, o) in want.iter_mut().zip(out) {
            *r += o;
        }
        ffn_host(&mut want);

        // The run, twice from the same state.
        let ffn = LayerFfnParts::from_parts(&ffn_norm, eps, &gate, &up, &down);
        let (o_base, _) = o_proj.launch_parts();
        let tail = frink_metal::gdn_prefill::PrefillAttnTail {
            branch: &branch,
            o_proj: o_base.mul_mm_sg_launch().expect("Q8_0 has a GEMM"),
            fold: fold_branch.metal_plan(branch_w),
            ffn: ffn.prefill_launches().expect("Q8_0 has a GEMM"),
            eps,
        };
        let run = |state: &mut RecurrentState| {
            let layer = m
                .prefill_layer(&attn_norm, eps, &ffn)
                .expect("this shape is one the kernels serve");
            let st = frink_metal::gdn_prefill::PrefillGdnState {
                ssm_ptr: state.ssm.as_ptr(),
                ssm_bytes: state.ssm.alloc_bytes(),
                conv_ptr: state.conv.as_ptr(),
                conv_bytes: state.conv.alloc_bytes(),
                conv_len: state.conv.len(),
            };
            // SAFETY: both states are page-aligned `AlignedF32`s
            // borrowed mutably for the call, which waits for the GPU.
            unsafe {
                frink_metal::gdn_prefill::launch_gdn_prefill_run(
                    &hidden,
                    rows,
                    Some(&tail),
                    &[(layer, st)],
                )
            }
            .expect("the run launches")
        };
        let mut s1 = state0();
        let got = run(&mut s1);
        let mut s2 = state0();
        let again = run(&mut s2);
        assert!(got == again, "the same run gave different bits");
        assert!(
            s1.ssm[..] == s2.ssm[..],
            "the same run left different states"
        );

        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1.0);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 2e-3 * scale,
                "out[{i}]: device {g} host {w}"
            );
        }
        for (what, d, w) in [
            ("ssm", &s1.ssm[..], &host_state.ssm[..]),
            ("conv", &s1.conv[..], &host_state.conv[..]),
        ] {
            let scale = w.iter().fold(0f32, |a, v| a.max(v.abs())).max(1.0);
            for (i, (x, y)) in d.iter().zip(w).enumerate() {
                assert!(
                    (x - y).abs() <= 2e-3 * scale,
                    "{what} state[{i}]: device {x} host {y}"
                );
            }
        }
    }
}
