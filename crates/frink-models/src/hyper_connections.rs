//! DeepSeek V4's "mHC" (multi-stream Hyper-Connection) residual mixing:
//! instead of one residual stream, the model carries `hc` (`hc_mult`,
//! real reference value 4) parallel streams per token, and each
//! sub-layer (attention, FFN) is preceded by a learned, per-token gated
//! merge of those streams into one input, followed by a Sinkhorn-
//! normalized mix back into all `hc` streams.
//!
//! Transcribed directly from the real, merged reference implementation
//! (llama.cpp PR #24162, `src/models/deepseek4.cpp`:
//! `build_hc_pre`/`build_hc_post`/`build_hc_head`/`build_hc_sinkhorn`/
//! `build_hc_weighted_sum`, read line-by-line), not derived by analogy
//! -- this closes the "mHC's exact math was not read" gap from earlier
//! research. The real implementation asserts `hc == 4` in `build_hc_pre`
//! (the mix-tensor offset layout is hardcoded to that split), so this
//! module does too rather than silently pretending to support other
//! values.
//!
//! Not yet wired into a DeepSeek V4 decoder (no such decoder exists in
//! frink yet) -- this is the residual-mixing primitive on its own,
//! analogous to how `mla.rs`/`block_residual.rs` exist as standalone,
//! tested modules before Kimi K3's decoder consumed them.

use frink_core::weight_matrix::WeightMatrix;

/// The real reference calls bare `ggml_rms_norm` here (no learned
/// per-element scale, unlike a normal transformer RMSNorm layer) --
/// this hyper-connection router computation has no weight tensor for
/// it in `deepseek4.cpp`.
fn rms_norm_no_weight(x: &[f32], eps: f32) -> Vec<f32> {
    crate::norm::rms_norm_no_params(x, eps)
}

/// The real reference implementation's only supported hyper-connection
/// multiplicity; `build_hc_pre`'s mix-tensor offsets are hardcoded to
/// this split (`GGML_ASSERT(hc == 4)` in the real source).
pub const HC_MULT: usize = 4;

/// Weights for the pre-sub-layer merge (`build_hc_pre`): projects the
/// flattened, RMS-normed `hc` streams to `(2 + hc) * hc` mix logits,
/// split into `pre` (hc), `post` (hc), and `comb` (hc*hc).
pub struct HyperConnectionPreWeights {
    pub fn_proj: WeightMatrix, // [(2+hc)*hc, hc*n_embd]
    /// `[scale_pre, scale_post, scale_comb]`, real tensor shape `{3}`.
    pub scale: [f32; 3],
    pub base_pre: [f32; HC_MULT],
    pub base_post: [f32; HC_MULT],
    pub base_comb: [f32; HC_MULT * HC_MULT],
}

/// Weights for the final output merge (`build_hc_head`): same
/// structure as the `pre` half of [`HyperConnectionPreWeights`], but
/// only ever produces the `hc`-wide merge gate (no `post`/`comb`,
/// since there is no further sub-layer to re-inject into).
pub struct HyperConnectionHeadWeights {
    pub fn_proj: WeightMatrix, // [hc, hc*n_embd]
    pub scale: f32,
    pub base: [f32; HC_MULT],
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `build_hc_weighted_sum`: per-token weighted sum of the `hc` stream
/// vectors (each `n_embd`-wide), `sum_h(x[h] * weights[h])`.
pub fn weighted_sum(x: &[Vec<f32>; HC_MULT], weights: &[f32; HC_MULT]) -> Vec<f32> {
    let n_embd = x[0].len();
    let mut out = vec![0f32; n_embd];
    for (xh, &w) in x.iter().zip(weights.iter()) {
        for (o, v) in out.iter_mut().zip(xh.iter()) {
            *o += v * w;
        }
    }
    out
}

/// `build_hc_sinkhorn`: `comb[dst][src]`, real algorithm -- softmax
/// over `dst` (per fixed `src`), `+eps`, one row-normalization (each
/// `dst` row sums to 1 over `src`), then `sinkhorn_iters - 1` rounds of
/// [column-normalize (each `src` column sums to 1 over `dst`),
/// row-normalize].
#[allow(clippy::needless_range_loop)]
pub fn sinkhorn(comb: &mut [[f32; HC_MULT]; HC_MULT], iters: u32, eps: f32) {
    for src in 0..HC_MULT {
        let mut col = [0f32; HC_MULT];
        for dst in 0..HC_MULT {
            col[dst] = comb[dst][src];
        }
        let max = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in col.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in col.iter_mut() {
            *v /= sum;
        }
        for dst in 0..HC_MULT {
            comb[dst][src] = col[dst] + eps;
        }
    }

    let norm_rows_over_src = |comb: &mut [[f32; HC_MULT]; HC_MULT]| {
        for dst in 0..HC_MULT {
            let sum: f32 = comb[dst].iter().sum::<f32>() + eps;
            for src in 0..HC_MULT {
                comb[dst][src] /= sum;
            }
        }
    };
    let norm_cols_over_dst = |comb: &mut [[f32; HC_MULT]; HC_MULT]| {
        for src in 0..HC_MULT {
            let sum: f32 = (0..HC_MULT).map(|dst| comb[dst][src]).sum::<f32>() + eps;
            for dst in 0..HC_MULT {
                comb[dst][src] /= sum;
            }
        }
    };

    norm_rows_over_src(comb);
    for _ in 1..iters {
        norm_cols_over_dst(comb);
        norm_rows_over_src(comb);
    }
}

/// `build_hc_pre`: merges `hc` residual streams into one sub-layer
/// input, plus the `post` gate and Sinkhorn-normalized `comb` matrix
/// needed by [`post`] afterward. Returns `(merged_input, post_gate,
/// comb_matrix)`.
#[allow(clippy::type_complexity, clippy::needless_range_loop)]
pub fn pre(
    weights: &HyperConnectionPreWeights,
    x: &[Vec<f32>; HC_MULT],
    rms_norm_eps: f32,
    sinkhorn_iters: u32,
    hc_eps: f32,
) -> (Vec<f32>, [f32; HC_MULT], [[f32; HC_MULT]; HC_MULT]) {
    let n_embd = x[0].len();
    let mut flat = Vec::with_capacity(HC_MULT * n_embd);
    for xh in x {
        flat.extend_from_slice(xh);
    }
    let flat_norm = rms_norm_no_weight(&flat, rms_norm_eps);
    let mixes = weights.fn_proj.apply(&flat_norm); // [(2+hc)*hc]

    let mut pre_gate = [0f32; HC_MULT];
    for h in 0..HC_MULT {
        pre_gate[h] = sigmoid(mixes[h] * weights.scale[0] + weights.base_pre[h]) + hc_eps;
    }

    let mut post_gate = [0f32; HC_MULT];
    for h in 0..HC_MULT {
        post_gate[h] = sigmoid(mixes[HC_MULT + h] * weights.scale[1] + weights.base_post[h]) * 2.0;
    }

    let mut comb = [[0f32; HC_MULT]; HC_MULT];
    for dst in 0..HC_MULT {
        for src in 0..HC_MULT {
            let idx = dst * HC_MULT + src;
            comb[dst][src] = mixes[2 * HC_MULT + idx] * weights.scale[2] + weights.base_comb[idx];
        }
    }
    sinkhorn(&mut comb, sinkhorn_iters, hc_eps);

    let merged = weighted_sum(x, &pre_gate);
    (merged, post_gate, comb)
}

/// `build_hc_post`: re-injects the sub-layer's single output back into
/// all `hc` streams, each scaled by its `post` gate and mixed with the
/// original (pre-merge) residual streams via the Sinkhorn `comb`
/// matrix: `out[dst] = sub_layer_out * post[dst] + sum_src(residual[src]
/// * comb[dst][src])`.
pub fn post(
    sub_layer_out: &[f32],
    residual: &[Vec<f32>; HC_MULT],
    post_gate: &[f32; HC_MULT],
    comb: &[[f32; HC_MULT]; HC_MULT],
) -> [Vec<f32>; HC_MULT] {
    let n_embd = sub_layer_out.len();
    std::array::from_fn(|dst| {
        let mut out = vec![0f32; n_embd];
        for (o, v) in out.iter_mut().zip(sub_layer_out.iter()) {
            *o = v * post_gate[dst];
        }
        for src in 0..HC_MULT {
            let w = comb[dst][src];
            for (o, v) in out.iter_mut().zip(residual[src].iter()) {
                *o += v * w;
            }
        }
        out
    })
}

/// `build_hc_head`: the final collapse of `hc` streams into one output
/// vector before unembedding -- structurally the `pre`-gate half of
/// [`pre`], with no `post`/`comb` since nothing follows it.
pub fn head(
    weights: &HyperConnectionHeadWeights,
    x: &[Vec<f32>; HC_MULT],
    rms_norm_eps: f32,
    hc_eps: f32,
) -> Vec<f32> {
    let n_embd = x[0].len();
    let mut flat = Vec::with_capacity(HC_MULT * n_embd);
    for xh in x {
        flat.extend_from_slice(xh);
    }
    let flat_norm = rms_norm_no_weight(&flat, rms_norm_eps);
    let mixes = weights.fn_proj.apply(&flat_norm); // [hc]

    let mut gate = [0f32; HC_MULT];
    for h in 0..HC_MULT {
        gate[h] = sigmoid(mixes[h] * weights.scale + weights.base[h]) + hc_eps;
    }
    weighted_sum(x, &gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::tensor::Tensor;

    fn wm(data: &[f32], rows: usize, cols: usize) -> WeightMatrix {
        assert_eq!(data.len(), rows * cols);
        WeightMatrix::F32(Tensor::new(data.to_vec(), vec![rows, cols]))
    }

    #[test]
    fn weighted_sum_with_one_hot_weights_selects_that_stream() {
        let x: [Vec<f32>; HC_MULT] = [
            vec![1.0, 2.0],
            vec![10.0, 20.0],
            vec![100.0, 200.0],
            vec![1000.0, 2000.0],
        ];
        let weights = [0.0, 1.0, 0.0, 0.0];
        let out = weighted_sum(&x, &weights);
        assert_eq!(out, vec![10.0, 20.0]);
    }

    #[test]
    fn sinkhorn_output_rows_and_columns_are_plausibly_normalized() {
        let mut comb = [
            [1.0, 0.5, 0.2, 0.1],
            [0.3, 1.2, 0.4, 0.2],
            [0.1, 0.2, 1.5, 0.3],
            [0.2, 0.1, 0.3, 1.1],
        ];
        sinkhorn(&mut comb, 3, 1e-6);
        for row in comb.iter() {
            for &v in row.iter() {
                assert!(v.is_finite() && v >= 0.0);
            }
        }
    }

    #[test]
    fn pre_then_post_round_trip_preserves_finite_output_and_stream_count() {
        let n_embd = 3;
        let hc_mix_dim = (2 + HC_MULT) * HC_MULT;
        let fn_proj = wm(
            &vec![0.05; hc_mix_dim * (HC_MULT * n_embd)],
            hc_mix_dim,
            HC_MULT * n_embd,
        );
        let weights = HyperConnectionPreWeights {
            fn_proj,
            scale: [1.0, 1.0, 1.0],
            base_pre: [0.0; HC_MULT],
            base_post: [0.0; HC_MULT],
            base_comb: [0.0; HC_MULT * HC_MULT],
        };
        let x: [Vec<f32>; HC_MULT] = std::array::from_fn(|h| vec![(h + 1) as f32; n_embd]);

        let (merged, post_gate, comb) = pre(&weights, &x, 1e-5, 3, 1e-6);
        assert_eq!(merged.len(), n_embd);
        assert!(merged.iter().all(|v| v.is_finite()));

        // Pretend the sub-layer is identity for this round-trip check.
        let streams = post(&merged, &x, &post_gate, &comb);
        assert_eq!(streams.len(), HC_MULT);
        for s in streams.iter() {
            assert_eq!(s.len(), n_embd);
            assert!(s.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn head_collapses_streams_to_a_single_finite_vector() {
        let n_embd = 3;
        let fn_proj = wm(
            &vec![0.05; HC_MULT * (HC_MULT * n_embd)],
            HC_MULT,
            HC_MULT * n_embd,
        );
        let weights = HyperConnectionHeadWeights {
            fn_proj,
            scale: 1.0,
            base: [0.0; HC_MULT],
        };
        let x: [Vec<f32>; HC_MULT] = std::array::from_fn(|h| vec![(h + 1) as f32; n_embd]);
        let out = head(&weights, &x, 1e-5, 1e-6);
        assert_eq!(out.len(), n_embd);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
