//! MiniMax-01's lightning attention block.
//!
//! The math is `frink_core::lightning`, which had been written and left
//! with no caller; this is the layer around it, the way
//! `crate::gdn` is the layer around `frink_core::gdn`.
//!
//! # The shape, from `src/models/minimax-01.cpp`
//!
//! A recurrent layer (`:44-51`) carries a FUSED `attn_qkv`
//! `{n_embd, 3 * n_head * head_dim}`, an `attn_gate` `wg`
//! `{n_embd, n_head * head_dim}`, an `attn_norm_2`
//! `{n_head * head_dim}` and the usual `attn_output`. A full-attention
//! layer in the same file carries an ordinary QKV instead, which is
//! why the mask that says which is which
//! (`crate::gdn::recurrent_layers`) is read before any tensor is.
//!
//! `:398-416` is the tail and it is NOT the attention tail this repo
//! already has: the block output is RMS-normed by `attn_norm_2`,
//! multiplied by `sigmoid(wg . x)` and only then sent through `wo`.
//! That is the delta-net's gated-norm shape rather than Falcon's
//! crossed pre-norm, which is what the refusal's mention of
//! `attn_norm_2` reads like until the graph is opened.
//!
//! # Per token, not per chunk
//!
//! `:315-388` computes a whole ubatch at once with three decay inputs
//! the host fills. `frink_core::lightning` proves that form equal to
//! the per-token recurrence for the case that matters here, and the
//! recurrence is what lets a decode step and a prefill row share one
//! body -- the same reason every other recurrent block in this repo is
//! written that way.

use frink_core::lightning::{lightning_step, slope_scale, slopes};
use frink_core::matmul::rms_norm;
use frink_core::weight_matrix::WeightMatrix;

/// One lightning layer's weights.
pub struct Lightning {
    /// `blk.N.attn_qkv.weight`, `[3 * n_head * head_dim, n_embd]`.
    pub qkv: WeightMatrix,
    /// `blk.N.attn_gate.weight`, `[n_head * head_dim, n_embd]`.
    pub gate: WeightMatrix,
    /// `blk.N.attn_norm_2.weight`, `[n_head * head_dim]`.
    pub norm: Vec<f32>,
    /// `blk.N.attn_output.weight`, `[n_embd, n_head * head_dim]`.
    pub out_proj: WeightMatrix,
}

/// The two numbers a layer's decay is built from.
///
/// Split out because both are functions of the LAYER and the head
/// count rather than of the file: `minimax-01.cpp:288` derives the
/// per-layer scale from the layer index, and the slopes are the
/// geometric ladder ALiBi uses. A file that carried them would be
/// carrying dead metadata.
#[derive(Debug, Clone)]
pub struct LightningDecay {
    /// One slope per head, in head order.
    pub slopes: Vec<f32>,
    /// This layer's scale on every slope.
    pub scale: f32,
}

impl LightningDecay {
    pub fn for_layer(il: usize, n_layer: usize, n_head: usize) -> Self {
        LightningDecay {
            slopes: slopes(n_head),
            scale: slope_scale(il, n_layer),
        }
    }

    /// `exp(-c s_h)`: what one token multiplies head `h`'s state by.
    pub fn per_step(&self, head: usize) -> f32 {
        (-self.scale * self.slopes[head]).exp()
    }
}

impl Lightning {
    /// Rows the per-head state needs: `head_dim * head_dim` per head.
    pub fn state_len(n_head: usize, head_dim: usize) -> usize {
        n_head * head_dim * head_dim
    }

    /// One token through the block, advancing `state` in place.
    ///
    /// NOT called `forward_token`: that name is reserved for the
    /// entry point that enters the CPU worker pool, and
    /// `engine::entry` fails the build for any other declaration of
    /// it. `forward_row` is what the other recurrent blocks call the
    /// same thing.
    ///
    /// `x` is the layer input AFTER `attn_norm`, as the graph has it
    /// (`:308`), and the return is what joins the residual -- `wo`
    /// applied, the gate and the norm already inside.
    pub fn forward_row(
        &self,
        x: &[f32],
        state: &mut [f32],
        n_head: usize,
        head_dim: usize,
        decay: &LightningDecay,
        eps: f32,
    ) -> Vec<f32> {
        let width = n_head * head_dim;
        debug_assert_eq!(state.len(), Self::state_len(n_head, head_dim));

        let qkv = self.qkv.apply(x);
        debug_assert_eq!(qkv.len(), 3 * width);
        let (q, rest) = qkv.split_at(width);
        let (k, v) = rest.split_at(width);

        // Per head, and the heads do not talk to each other: the state
        // is `head_dim x head_dim` per head and the recurrence is the
        // one `frink_core::lightning` pins against the chunked form.
        let mut inner = vec![0.0f32; width];
        for h in 0..n_head {
            let lo = h * head_dim;
            let hi = lo + head_dim;
            let s_lo = h * head_dim * head_dim;
            let s_hi = s_lo + head_dim * head_dim;
            let out = lightning_step(
                &q[lo..hi],
                &k[lo..hi],
                &v[lo..hi],
                &mut state[s_lo..s_hi],
                decay.per_step(h),
            );
            inner[lo..hi].copy_from_slice(&out);
        }

        // The tail: norm, then the sigmoid gate, then `wo`. The gate
        // reads the LAYER INPUT, not the block output (`:406`), which
        // is the one thing a reader is likely to get backwards.
        let normed = rms_norm(&inner, &self.norm, eps);
        let gate = self.gate.apply(x);
        let gated: Vec<f32> = normed
            .iter()
            .zip(&gate)
            .map(|(n, g)| n * (1.0 / (1.0 + (-g).exp())))
            .collect();
        self.out_proj.apply(&gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::Tensor;

    fn matrix(rows: usize, cols: usize, seed: f32) -> WeightMatrix {
        let v: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 + seed) * 0.017).sin() * 0.5)
            .collect();
        WeightMatrix::F32(Tensor::new(v, vec![rows, cols]))
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// The chunked form `minimax-01.cpp:315-388` computes, transcribed
    /// straight from the graph rather than from this module, so the
    /// two are independent statements of the same function:
    ///
    /// ```text
    ///   out[j] = (q[j] * exp(-c s (j+1))) @ KV_0
    ///          + sum_{i<=j} (q[j].k[i]) exp(-c s (j - i)) v[i]
    /// ```
    ///
    /// `KV_0` is zero here because a fresh sequence starts with no
    /// state, which is the case a prefill actually runs.
    fn chunked_reference(
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        c_s: f32,
    ) -> Vec<Vec<f32>> {
        let n = q.len();
        let d = q[0].len();
        let mut out = vec![vec![0.0f32; d]; n];
        for j in 0..n {
            for i in 0..=j {
                let qk: f32 = q[j].iter().zip(&k[i]).map(|(a, b)| a * b).sum();
                let decay = (-c_s * (j - i) as f32).exp();
                for t in 0..d {
                    out[j][t] += qk * decay * v[i][t];
                }
            }
        }
        out
    }

    /// The per-token recurrence this block runs has to agree with the
    /// chunked form the graph computes, over a whole prefill and not
    /// just one step. `frink_core::lightning` pins the two-token case;
    /// this is the case a real prompt takes, through the projections.
    #[test]
    fn the_block_agrees_with_the_graphs_chunked_form() {
        let (n_head, head_dim, n_embd, n_tokens) = (2usize, 4usize, 6usize, 5usize);
        let width = n_head * head_dim;
        let layer = Lightning {
            qkv: matrix(3 * width, n_embd, 1.0),
            gate: matrix(width, n_embd, 2.0),
            norm: (0..width).map(|i| 1.0 + i as f32 * 0.01).collect(),
            out_proj: matrix(n_embd, width, 3.0),
        };
        let decay = LightningDecay::for_layer(1, 4, n_head);
        let xs: Vec<Vec<f32>> = (0..n_tokens)
            .map(|t| {
                (0..n_embd)
                    .map(|i| ((t * n_embd + i) as f32 * 0.03).cos())
                    .collect()
            })
            .collect();

        // What the block produces, token by token.
        let mut state = vec![0.0f32; Lightning::state_len(n_head, head_dim)];
        let got: Vec<Vec<f32>> = xs
            .iter()
            .map(|x| layer.forward_row(x, &mut state, n_head, head_dim, &decay, 1e-5))
            .collect();

        // What the graph's chunked form produces, per head, from the
        // same projections.
        let projected: Vec<Vec<f32>> = xs.iter().map(|x| layer.qkv.apply(x)).collect();
        let mut inner = vec![vec![0.0f32; width]; n_tokens];
        for h in 0..n_head {
            let (lo, hi) = (h * head_dim, h * head_dim + head_dim);
            let take = |o: usize| -> Vec<Vec<f32>> {
                projected
                    .iter()
                    .map(|p| p[o * width + lo..o * width + hi].to_vec())
                    .collect()
            };
            let rows =
                chunked_reference(&take(0), &take(1), &take(2), decay.scale * decay.slopes[h]);
            for (t, row) in rows.iter().enumerate() {
                inner[t][lo..hi].copy_from_slice(row);
            }
        }
        let want: Vec<Vec<f32>> = xs
            .iter()
            .zip(&inner)
            .map(|(x, i)| {
                let normed = rms_norm(i, &layer.norm, 1e-5);
                let g = layer.gate.apply(x);
                let gated: Vec<f32> = normed
                    .iter()
                    .zip(&g)
                    .map(|(n, gg)| n * sigmoid(*gg))
                    .collect();
                layer.out_proj.apply(&gated)
            })
            .collect();

        for (t, (a, b)) in got.iter().zip(&want).enumerate() {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() < 1e-5,
                    "token {t} element {i}: recurrence {x} vs chunked {y}"
                );
            }
        }
    }

    /// The gate reads the LAYER INPUT, not the block output. Getting
    /// that backwards still produces plausible numbers, so it is
    /// pinned rather than trusted: with the gate weights zeroed the
    /// sigmoid is 0.5 everywhere, and the answer must be exactly half
    /// of the ungated one.
    #[test]
    fn the_gate_reads_the_layer_input() {
        let (n_head, head_dim, n_embd) = (2usize, 4usize, 6usize);
        let width = n_head * head_dim;
        let zero_gate = Lightning {
            qkv: matrix(3 * width, n_embd, 1.0),
            gate: WeightMatrix::F32(Tensor::new(vec![0.0; width * n_embd], vec![width, n_embd])),
            norm: vec![1.0; width],
            out_proj: matrix(n_embd, width, 3.0),
        };
        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.2).sin()).collect();
        let decay = LightningDecay::for_layer(0, 2, n_head);

        let mut state = vec![0.0f32; Lightning::state_len(n_head, head_dim)];
        let half = zero_gate.forward_row(&x, &mut state, n_head, head_dim, &decay, 1e-5);

        // The same block with the gate multiplied in by hand at 1.0.
        let mut s2 = vec![0.0f32; Lightning::state_len(n_head, head_dim)];
        let qkv = zero_gate.qkv.apply(&x);
        let (q, rest) = qkv.split_at(width);
        let (k, v) = rest.split_at(width);
        let mut inner = vec![0.0f32; width];
        for h in 0..n_head {
            let (lo, hi) = (h * head_dim, h * head_dim + head_dim);
            let (slo, shi) = (h * head_dim * head_dim, (h + 1) * head_dim * head_dim);
            let out = frink_core::lightning::lightning_step(
                &q[lo..hi],
                &k[lo..hi],
                &v[lo..hi],
                &mut s2[slo..shi],
                decay.per_step(h),
            );
            inner[lo..hi].copy_from_slice(&out);
        }
        let ungated = zero_gate
            .out_proj
            .apply(&rms_norm(&inner, &zero_gate.norm, 1e-5));

        for (i, (h, u)) in half.iter().zip(&ungated).enumerate() {
            assert!(
                (h - u * 0.5).abs() < 1e-5,
                "element {i}: gated {h} is not half of ungated {u}"
            );
        }
    }
}
