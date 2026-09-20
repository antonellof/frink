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
//! # Two facts a reader of the tensor shapes alone gets wrong
//!
//! Both were wrong here for one PR, and neither shows up as a panic --
//! the shapes agree either way and the numbers stay plausible, which
//! is why they are pinned by the libllama golden rather than by
//! inspection.
//!
//! **The projection runs through SiLU before it is split.**
//! `:303` is `QKVcur = ggml_silu(ctx0, QKVcur)` on the whole
//! `3 * n_head * head_dim` output, so Q, K and V are each
//! `silu(W x)` and not `W x`. No other fused QKV in this repo does
//! that.
//!
//! **The fused projection is HEAD-major, not Q-then-K-then-V.**
//! `:305` reshapes it to `{3 * head_dim, n_head, ...}` and `:307-309`
//! view Q, K and V at element offsets `0`, `head_dim` and
//! `2 * head_dim` INSIDE that first axis, so head `h` owns the
//! contiguous run `[q_h | k_h | v_h]` at `h * 3 * head_dim`. Reading
//! it as three `n_head * head_dim` blocks is a permutation of the
//! rows, which for a one-head fixture is the identity -- so the
//! fixture has two.
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
use frink_core::recurrent_state::RecurrentState;
use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::TensorSource;

use crate::loader::{load_f32_vec, load_weight_matrix, LoadError};

/// One lightning layer's weights, and the two counts that size them.
pub struct Lightning {
    /// Query heads. Uniform across the file: `minimax-01.cpp:44-51`
    /// sizes every recurrent layer from `n_head` and `n_embd_head_k`.
    pub n_head: usize,
    /// `n_embd_head_k`, which `llama-hparams.cpp:249-253` also uses as
    /// `n_embd_head_la` to size the recurrent state.
    pub head_dim: usize,
    /// This layer's decay, a function of the layer index and the head
    /// count alone.
    pub decay: LightningDecay,
    /// `blk.N.attn_qkv.weight`, `[3 * n_head * head_dim, n_embd]`,
    /// HEAD-major (see the module docs).
    pub qkv: WeightMatrix,
    /// `blk.N.attn_gate.weight`, `[n_head * head_dim, n_embd]`.
    pub gate: WeightMatrix,
    /// `blk.N.attn_norm_2.weight`, `[n_head * head_dim]`.
    pub norm: Vec<f32>,
    /// `blk.N.attn_output.weight`, `[n_embd, n_head * head_dim]`. The
    /// same tensor NAME a full-attention layer of the same file uses
    /// (`:53`), which is why it is loaded here rather than left on
    /// `AttnWeights::o_proj`: the tail applies it after the gate, not
    /// where the attention tail would.
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
    ///
    /// `llama-hparams.cpp:253` spells the same product
    /// `n_embd_head_la * n_embd_head_la * n_head()`.
    pub fn state_len(n_head: usize, head_dim: usize) -> usize {
        n_head * head_dim * head_dim
    }

    /// Loads layer `layer`'s four tensors and checks them against
    /// `minimax-01.cpp:44-53`'s shapes.
    ///
    /// `n_layer` is the LOGICAL layer count the graph loops over,
    /// because the decay scale is `1 - il / (n_layer - 1)` and nothing
    /// in the file carries it.
    pub fn load(
        file: &impl TensorSource,
        layer: usize,
        n_layer: usize,
        n_head: usize,
        head_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self, LoadError> {
        let width = n_head * head_dim;
        let name = |t: &str| format!("blk.{layer}.{t}");
        let matrix = |t: &str, rows: usize, cols: usize| -> Result<WeightMatrix, LoadError> {
            let m = load_weight_matrix(file, &name(t))?;
            if m.rows() != rows || m.cols() != cols {
                return Err(LoadError::UnsupportedFeature(
                    name(t),
                    format!(
                        "{}x{}; minimax-01.cpp:44-53 sizes it {rows}x{cols}",
                        m.rows(),
                        m.cols()
                    ),
                ));
            }
            Ok(m)
        };
        let norm = load_f32_vec(file, &name("attn_norm_2.weight"))?;
        if norm.len() != width {
            return Err(LoadError::UnsupportedFeature(
                name("attn_norm_2.weight"),
                format!(
                    "{} entries; minimax-01.cpp:48 sizes it n_head * head_dim = {width}",
                    norm.len()
                ),
            ));
        }
        Ok(Lightning {
            n_head,
            head_dim,
            decay: LightningDecay::for_layer(layer, n_layer, n_head),
            qkv: matrix("attn_qkv.weight", 3 * width, hidden_dim)?,
            gate: matrix("attn_gate.weight", width, hidden_dim)?,
            norm,
            out_proj: matrix("attn_output.weight", hidden_dim, width)?,
        })
    }

    /// A fresh sequence's state: one `head_dim x head_dim` KV per head
    /// and no convolution window.
    pub fn zero_state(&self) -> RecurrentState {
        RecurrentState::zeros(0, Self::state_len(self.n_head, self.head_dim))
    }

    /// `rows` consecutive tokens of ONE sequence (`normed` is
    /// `[rows][n_embd]`) through the block, advancing `state` in place.
    pub fn forward_rows(
        &self,
        normed: &[f32],
        rows: usize,
        state: &mut RecurrentState,
        rms_eps: f32,
    ) -> Vec<f32> {
        let hidden = self.out_proj.rows();
        assert_eq!(normed.len(), rows * hidden);
        assert_eq!(
            state.ssm.len(),
            Self::state_len(self.n_head, self.head_dim),
            "lightning state sized by these weights"
        );
        let mut out = Vec::with_capacity(rows * hidden);
        for row in normed.chunks(hidden) {
            out.extend(self.forward_row(row, &mut state.ssm, rms_eps));
        }
        out
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
    pub fn forward_row(&self, x: &[f32], state: &mut [f32], eps: f32) -> Vec<f32> {
        let (n_head, head_dim) = (self.n_head, self.head_dim);
        let width = n_head * head_dim;
        debug_assert_eq!(state.len(), Self::state_len(n_head, head_dim));

        // `:303`: the WHOLE projection through SiLU, before the split.
        let mut qkv = self.qkv.apply(x);
        debug_assert_eq!(qkv.len(), 3 * width);
        for v in &mut qkv {
            *v = *v / (1.0 + (-*v).exp());
        }

        // Per head, and the heads do not talk to each other: the state
        // is `head_dim x head_dim` per head and the recurrence is the
        // one `frink_core::lightning` pins against the chunked form.
        // Head `h` owns `[q | k | v]` at `h * 3 * head_dim` (`:305-309`).
        let mut inner = vec![0.0f32; width];
        for h in 0..n_head {
            let base = h * 3 * head_dim;
            let (q, k, v) = (
                &qkv[base..base + head_dim],
                &qkv[base + head_dim..base + 2 * head_dim],
                &qkv[base + 2 * head_dim..base + 3 * head_dim],
            );
            let s_lo = h * head_dim * head_dim;
            let s_hi = s_lo + head_dim * head_dim;
            let out = lightning_step(q, k, v, &mut state[s_lo..s_hi], self.decay.per_step(h));
            inner[h * head_dim..(h + 1) * head_dim].copy_from_slice(&out);
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

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn block(
        n_head: usize,
        head_dim: usize,
        n_embd: usize,
        il: usize,
        n_layer: usize,
    ) -> Lightning {
        let width = n_head * head_dim;
        Lightning {
            n_head,
            head_dim,
            decay: LightningDecay::for_layer(il, n_layer, n_head),
            qkv: matrix(3 * width, n_embd, 1.0),
            gate: matrix(width, n_embd, 2.0),
            norm: (0..width).map(|i| 1.0 + i as f32 * 0.01).collect(),
            out_proj: matrix(n_embd, width, 3.0),
        }
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
    ///
    /// TWO heads, because the head-major split of the fused projection
    /// (`:305-309`) is the identity at one head and a permutation at
    /// two.
    #[test]
    fn the_block_agrees_with_the_graphs_chunked_form() {
        let (n_head, head_dim, n_embd, n_tokens) = (2usize, 4usize, 6usize, 5usize);
        let width = n_head * head_dim;
        let layer = block(n_head, head_dim, n_embd, 1, 4);
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
            .map(|x| layer.forward_row(x, &mut state, 1e-5))
            .collect();

        // What the graph's chunked form produces, per head, from the
        // same projections -- SiLU applied and split head-major, as
        // `:303-309` do.
        let projected: Vec<Vec<f32>> = xs
            .iter()
            .map(|x| layer.qkv.apply(x).iter().map(|v| silu(*v)).collect())
            .collect();
        let mut inner = vec![vec![0.0f32; width]; n_tokens];
        for h in 0..n_head {
            let base = h * 3 * head_dim;
            let take = |o: usize| -> Vec<Vec<f32>> {
                projected
                    .iter()
                    .map(|p| p[base + o * head_dim..base + (o + 1) * head_dim].to_vec())
                    .collect()
            };
            let rows = chunked_reference(
                &take(0),
                &take(1),
                &take(2),
                layer.decay.scale * layer.decay.slopes[h],
            );
            for (t, row) in rows.iter().enumerate() {
                inner[t][h * head_dim..(h + 1) * head_dim].copy_from_slice(row);
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

    /// `forward_rows` over a whole prompt is `forward_row` in a loop:
    /// the batched host body and the decode step share ONE recurrence,
    /// so a prefill and the decode that follows it cannot disagree
    /// about the state.
    #[test]
    fn many_rows_are_the_same_as_one_row_at_a_time() {
        let (n_head, head_dim, n_embd, n_tokens) = (2usize, 3usize, 5usize, 4usize);
        let layer = block(n_head, head_dim, n_embd, 0, 3);
        let xs: Vec<f32> = (0..n_tokens * n_embd)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();

        let mut batched = layer.zero_state();
        let got = layer.forward_rows(&xs, n_tokens, &mut batched, 1e-5);

        let mut one = layer.zero_state();
        let mut want = Vec::new();
        for row in xs.chunks(n_embd) {
            want.extend(layer.forward_row(row, &mut one.ssm, 1e-5));
        }
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-6, "element {i}: {a} vs {b}");
        }
        assert_eq!(batched.ssm.as_ref(), one.ssm.as_ref());
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
        let mut zero_gate = block(n_head, head_dim, n_embd, 0, 2);
        zero_gate.gate =
            WeightMatrix::F32(Tensor::new(vec![0.0; width * n_embd], vec![width, n_embd]));
        zero_gate.norm = vec![1.0; width];
        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.2).sin()).collect();

        let mut state = vec![0.0f32; Lightning::state_len(n_head, head_dim)];
        let half = zero_gate.forward_row(&x, &mut state, 1e-5);

        // The same block with the gate multiplied in by hand at 1.0.
        let mut s2 = vec![0.0f32; Lightning::state_len(n_head, head_dim)];
        let qkv: Vec<f32> = zero_gate.qkv.apply(&x).iter().map(|v| silu(*v)).collect();
        let mut inner = vec![0.0f32; width];
        for h in 0..n_head {
            let base = h * 3 * head_dim;
            let (slo, shi) = (h * head_dim * head_dim, (h + 1) * head_dim * head_dim);
            let out = frink_core::lightning::lightning_step(
                &qkv[base..base + head_dim],
                &qkv[base + head_dim..base + 2 * head_dim],
                &qkv[base + 2 * head_dim..base + 3 * head_dim],
                &mut s2[slo..shi],
                zero_gate.decay.per_step(h),
            );
            inner[h * head_dim..(h + 1) * head_dim].copy_from_slice(&out);
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
