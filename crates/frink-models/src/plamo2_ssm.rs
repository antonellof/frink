//! PLaMo-2's state-space block (`build_plamo2_mamba_layer`,
//! `plamo2.cpp:218-343`): Mamba-1's dt / B / C path feeding Mamba-2's
//! per-head scan, in one spelling that neither [`crate::mamba1`] nor
//! [`crate::mamba2`] has, at the site attention occupies on a zero-KV
//! layer ([`crate::layer_shapes::AttnShape::Plamo2Ssm`]).
//!
//! Read against `mamba-base.cpp` (the two bodies the other modules
//! spell), these are the differences, each with its line:
//!
//! ```text
//! zx      = ssm_in(normed)                         {2 d_inner}            :245
//! z, x    = zx split PER HEAD: head h is [z_h | x_h], head_dim each     :248-258
//! x       = silu(conv(x))                          NO conv bias           :262-277
//! bcdt    = ssm_x(x)                               {2 d_state + dt_dim}   :283
//! B, C, dt = bcdt split IN THAT ORDER              :286-292 (Mamba-1: dt, B, C)
//! B, C, dt = rms_norm(.) * w                       REQUIRED weights, :295-297
//! dt      = ssm_dt(dt) + dt_b                      {n_heads}: one dt PER HEAD, :300-301
//! y       = ssm_scan(state, x, softplus(dt), A {1, n_heads}, B, C)   :304-321
//! y       = y + x * D                              D {1, n_heads}, per head, :333-334
//! y       = silu(z) * y                            :335
//! out     = ssm_out(y)                             :339
//! ```
//!
//! and the widths: `d_inner = n_heads * head_dim` where `n_heads` is
//! what the converter writes into `ssm.time_step_rank`
//! (`conversion/plamo.py:108`, `mamba_num_heads`), and `dt_dim =
//! max(64, n_embd / 16)` is a LITERAL of the graph (`:38`, `:285`),
//! not a key. `ssm.group_count` is written as 0 and asserted 0
//! (`:233`), so B and C are one group shared by every head.
//!
//! The scan is [`frink_core::mamba2::scan_step`] with `n_head =
//! n_heads`, `head_dim = d_inner / n_heads`, `n_group = 1` and
//! [`Decay::PerHead`], which is Mamba-2's arm of the kernel; the state
//! is a [`RecurrentState`] on the sequence's layer cache exactly as the
//! other two blocks': the conv window `[d_conv - 1][d_inner]`
//! (`n_embd_r` with `n_group = 0`) and the SSM state `[n_heads][head_dim]
//! [d_state]`.
//!
//! The converter adds `1.0 / 5` to `post_mixer_norm` and `1 / 5^1.5`
//! to `post_mlp_norm` (`conversion/plamo.py:135-140`) so the file's
//! post-norm weights are already the graph's; nothing here scales
//! them.

use frink_core::mamba2::{conv_step, scan_step, Decay, ScanDims};
use frink_core::matmul::rms_norm;
use frink_core::recurrent_state::RecurrentState;
use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::TensorSource;

use crate::loader::{load_f32_vec, load_weight_matrix, LoadError};

/// The graph's `dt_dim` (`plamo2.cpp:38,285`): a literal, not a key.
pub fn dt_dim(n_embd: usize) -> usize {
    (n_embd / 16).max(64)
}

/// The `ssm.*` hparams the graph reads (`plamo2.cpp:4-9`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plamo2SsmHparams {
    pub d_conv: usize,
    pub d_inner: usize,
    pub d_state: usize,
    /// `ssm.time_step_rank`, which PLaMo-2's converter fills with the
    /// HEAD count (`conversion/plamo.py:108`).
    pub n_heads: usize,
    pub dt_dim: usize,
}

impl Plamo2SsmHparams {
    pub fn read(file: &impl TensorSource, arch: &str, n_embd: usize) -> Result<Self, LoadError> {
        let key = |k: &str| format!("{arch}.ssm.{k}");
        let read = |k: &str| {
            file.metadata_u64(&key(k))
                .map(|v| v as usize)
                .ok_or_else(|| LoadError::MissingHparam(key(k)))
        };
        let h = Self {
            d_conv: read("conv_kernel")?,
            d_inner: read("inner_size")?,
            d_state: read("state_size")?,
            n_heads: read("time_step_rank")?,
            dt_dim: dt_dim(n_embd),
        };
        let n_group = file.metadata_u64(&key("group_count")).unwrap_or(0);
        if n_group != 0 {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!("ssm.group_count {n_group}: plamo2.cpp:233 asserts it is 0"),
            ));
        }
        if h.d_conv < 2 || h.d_inner == 0 || h.d_state == 0 || h.n_heads == 0 {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!("ssm.* hparams {h:?}: every one must be positive, conv_kernel at least 2"),
            ));
        }
        if !h.d_inner.is_multiple_of(h.n_heads) {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!(
                    "ssm.inner_size {} is not a multiple of the head count {} \
                     (plamo2.cpp:232 asserts d_inner % n_heads == 0)",
                    h.d_inner, h.n_heads
                ),
            ));
        }
        Ok(h)
    }

    pub fn head_dim(self) -> usize {
        self.d_inner / self.n_heads
    }

    pub fn scan_dims(self) -> ScanDims {
        ScanDims {
            n_head: self.n_heads,
            head_dim: self.head_dim(),
            d_state: self.d_state,
            n_group: 1,
        }
    }

    /// Floats of state per sequence per layer: `n_embd_r + n_embd_s`.
    pub fn state_floats(self) -> (usize, usize) {
        (
            (self.d_conv - 1) * self.d_inner,
            self.d_inner * self.d_state,
        )
    }
}

/// One PLaMo-2 SSM layer's weights.
pub struct Plamo2Ssm {
    pub h: Plamo2SsmHparams,
    /// `blk.N.ssm_in.weight`, `[2 d_inner, n_embd]`, rows per head
    /// `[z_h | x_h]`.
    pub in_proj: WeightMatrix,
    /// `blk.N.ssm_conv1d.weight`, `[d_inner][d_conv]`; no bias (`:277`).
    pub conv1d: Vec<f32>,
    /// `blk.N.ssm_x.weight`, `[2 d_state + dt_dim, d_inner]`.
    pub x_proj: WeightMatrix,
    /// `blk.N.ssm_b_norm` / `ssm_c_norm` `[d_state]`, `ssm_dt_norm`
    /// `[dt_dim]`, spelled with no `.weight` by the converter (see
    /// `load`); all REQUIRED (`:77-79`).
    pub b_norm: Vec<f32>,
    pub c_norm: Vec<f32>,
    pub dt_norm: Vec<f32>,
    /// `blk.N.ssm_dt.weight`, `[n_heads, dt_dim]`.
    pub dt_proj: WeightMatrix,
    /// `blk.N.ssm_dt.bias`, `[n_heads]`.
    pub dt_bias: Vec<f32>,
    /// `blk.N.ssm_a`, `[n_heads]`, stored negative (`-exp(A_log)`).
    pub a: Vec<f32>,
    /// `blk.N.ssm_d`, `[n_heads]`.
    pub d: Vec<f32>,
    /// `blk.N.ssm_out.weight`, `[n_embd, d_inner]`.
    pub out_proj: WeightMatrix,
}

impl Plamo2Ssm {
    /// Loads layer `layer`'s tensors and checks them against
    /// `plamo2.cpp:68-79`'s shapes.
    pub fn load(
        file: &impl TensorSource,
        arch: &str,
        layer: usize,
        hidden_dim: usize,
    ) -> Result<Self, LoadError> {
        let h = Plamo2SsmHparams::read(file, arch, hidden_dim)?;
        let name = |t: &str| format!("blk.{layer}.{t}");
        let check = |what: String, got: usize, want: usize| -> Result<(), LoadError> {
            if got == want {
                Ok(())
            } else {
                Err(LoadError::UnsupportedFeature(
                    what,
                    format!("{got} elements; the ssm.* hparams size it at {want}"),
                ))
            }
        };
        let matrix = |t: &str, rows: usize, cols: usize| -> Result<WeightMatrix, LoadError> {
            let m = load_weight_matrix(file, &name(t))?;
            if m.rows() != rows || m.cols() != cols {
                return Err(LoadError::UnsupportedFeature(
                    name(t),
                    format!(
                        "{}x{}; plamo2.cpp:68-79 sizes it {rows}x{cols}",
                        m.rows(),
                        m.cols()
                    ),
                ));
            }
            Ok(m)
        };
        let vec = |t: &str, want: usize| -> Result<Vec<f32>, LoadError> {
            let v = load_f32_vec(file, &name(t))?;
            check(name(t), v.len(), want)?;
            Ok(v)
        };
        // The three norms are `tn(LLM_TENSOR_SSM_*_NORM, i)` -- the
        // two-argument overload, NO `.weight` (`plamo2.cpp:77-79`), and
        // the converter agrees because its map entries end in `.weight`
        // and match exactly (`tensor_mapping.py:838,854,860`), so a real
        // file spells them `blk.N.ssm_dt_norm`. Both spellings are read,
        // as `crate::norm_sites` reads the post-norms, because nothing in
        // the metadata says which a file uses.
        let norm_vec = |t: &str, want: usize| -> Result<Vec<f32>, LoadError> {
            let bare = name(t);
            let suffixed = format!("{bare}.weight");
            let which = if file.find_tensor(&bare).is_some() {
                bare
            } else {
                suffixed
            };
            let v = load_f32_vec(file, &which)?;
            check(which, v.len(), want)?;
            Ok(v)
        };
        let in_proj = matrix("ssm_in.weight", 2 * h.d_inner, hidden_dim)?;
        let conv1d = vec("ssm_conv1d.weight", h.d_conv * h.d_inner)?;
        if file.find_tensor(&name("ssm_conv1d.bias")).is_some() {
            return Err(LoadError::UnsupportedFeature(
                name("ssm_conv1d.bias"),
                "plamo2.cpp:69 creates no conv bias and :277 adds none; a file carrying one \
                 describes a graph this architecture does not compute"
                    .to_string(),
            ));
        }
        let x_proj = matrix("ssm_x.weight", 2 * h.d_state + h.dt_dim, h.d_inner)?;
        let dt_norm = norm_vec("ssm_dt_norm", h.dt_dim)?;
        let b_norm = norm_vec("ssm_b_norm", h.d_state)?;
        let c_norm = norm_vec("ssm_c_norm", h.d_state)?;
        let dt_proj = matrix("ssm_dt.weight", h.n_heads, h.dt_dim)?;
        let dt_bias = vec("ssm_dt.bias", h.n_heads)?;
        let a = vec("ssm_a", h.n_heads)?;
        let d = vec("ssm_d", h.n_heads)?;
        let out_proj = matrix("ssm_out.weight", hidden_dim, h.d_inner)?;
        Ok(Self {
            h,
            in_proj,
            conv1d,
            x_proj,
            b_norm,
            c_norm,
            dt_norm,
            dt_proj,
            dt_bias,
            a,
            d,
            out_proj,
        })
    }

    /// A fresh sequence's state for this layer.
    pub fn zero_state(&self) -> RecurrentState {
        let (conv, ssm) = self.h.state_floats();
        RecurrentState::zeros(conv, ssm)
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
        let h = self.h;
        let n_embd = self.out_proj.rows();
        assert_eq!(normed.len(), rows * n_embd);
        let (conv_len, ssm_len) = h.state_floats();
        assert_eq!(
            state.conv.len(),
            conv_len,
            "conv state sized by these weights"
        );
        assert_eq!(state.ssm.len(), ssm_len, "ssm state sized by these weights");
        let zx = if rows == 1 {
            self.in_proj.apply(normed)
        } else {
            self.in_proj.apply_batch(normed, rows)
        };
        let (d_inner, d_state, dt_dim, head_dim) = (h.d_inner, h.d_state, h.dt_dim, h.head_dim());
        let dims = h.scan_dims();
        let mut ys = vec![0.0f32; rows * d_inner];
        let mut x = vec![0.0f32; d_inner];
        let mut z = vec![0.0f32; d_inner];
        let mut xc = vec![0.0f32; d_inner];
        let mut y = vec![0.0f32; d_inner];
        for r in 0..rows {
            let row = &zx[r * 2 * d_inner..(r + 1) * 2 * d_inner];
            // :248-258: head `hh` of the projection is `[z_hh | x_hh]`.
            for hh in 0..h.n_heads {
                let pair = &row[hh * 2 * head_dim..(hh + 1) * 2 * head_dim];
                z[hh * head_dim..(hh + 1) * head_dim].copy_from_slice(&pair[..head_dim]);
                x[hh * head_dim..(hh + 1) * head_dim].copy_from_slice(&pair[head_dim..]);
            }
            // :262-277: the conv over this token and the state, SiLU, no bias.
            conv_step(&mut state.conv, &self.conv1d, h.d_conv, &x, &mut xc);
            for v in xc.iter_mut() {
                *v = silu(*v);
            }
            // :283-297: B, C, dt from one projection, each RMS-normed.
            let bcdt = self.x_proj.apply(&xc);
            let (b, rest) = bcdt.split_at(d_state);
            let (c, dt_low) = rest.split_at(d_state);
            debug_assert_eq!(dt_low.len(), dt_dim);
            let b = rms_norm(b, &self.b_norm, rms_eps);
            let c = rms_norm(c, &self.c_norm, rms_eps);
            let dt_low = rms_norm(dt_low, &self.dt_norm, rms_eps);
            // :300-301: one dt per head, plus the bias; softplus is the scan's.
            let mut dt = self.dt_proj.apply(&dt_low);
            for (t, bias) in dt.iter_mut().zip(&self.dt_bias) {
                *t += bias;
            }
            scan_step(
                dims,
                &mut state.ssm,
                &xc,
                &dt,
                Decay::PerHead(&self.a),
                &b,
                &c,
                &mut y,
            );
            // :333-335: y + x * D (D per head), then silu(z) * y.
            for (hh, &d) in self.d.iter().enumerate().take(h.n_heads) {
                for i in hh * head_dim..(hh + 1) * head_dim {
                    y[i] = (y[i] + xc[i] * d) * silu(z[i]);
                }
            }
            ys[r * d_inner..(r + 1) * d_inner].copy_from_slice(&y);
        }
        if rows == 1 {
            self.out_proj.apply(&ys)
        } else {
            self.out_proj.apply_batch(&ys, rows)
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::Tensor;

    fn hp() -> Plamo2SsmHparams {
        Plamo2SsmHparams {
            d_conv: 3,
            d_inner: 8,
            d_state: 2,
            n_heads: 2,
            dt_dim: 64,
        }
    }

    #[test]
    fn dt_dim_is_the_graph_s_literal() {
        assert_eq!(dt_dim(4096), 256);
        assert_eq!(dt_dim(2048), 128);
        assert_eq!(dt_dim(512), 64, "the floor of 64 (plamo2.cpp:38)");
    }

    #[test]
    fn the_widths_are_the_graph_s() {
        let h = hp();
        assert_eq!(h.head_dim(), 4);
        assert_eq!(h.scan_dims().n_head, 2);
        assert_eq!(h.scan_dims().n_group, 1);
        assert_eq!(h.state_floats(), (2 * 8, 8 * 2));
    }

    fn block(h: Plamo2SsmHparams, n_embd: usize, seed: &mut u32) -> Plamo2Ssm {
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((*seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
                })
                .collect()
        };
        let mat = |rows: usize, cols: usize, v: Vec<f32>| {
            WeightMatrix::F32(Tensor::new(v, vec![rows, cols]))
        };
        Plamo2Ssm {
            h,
            in_proj: mat(2 * h.d_inner, n_embd, rnd(2 * h.d_inner * n_embd)),
            conv1d: rnd(h.d_conv * h.d_inner),
            x_proj: mat(
                2 * h.d_state + h.dt_dim,
                h.d_inner,
                rnd((2 * h.d_state + h.dt_dim) * h.d_inner),
            ),
            b_norm: rnd(h.d_state).iter().map(|v| 1.0 + v).collect(),
            c_norm: rnd(h.d_state).iter().map(|v| 1.0 + v).collect(),
            dt_norm: rnd(h.dt_dim).iter().map(|v| 1.0 + v).collect(),
            dt_proj: mat(h.n_heads, h.dt_dim, rnd(h.n_heads * h.dt_dim)),
            dt_bias: rnd(h.n_heads),
            a: rnd(h.n_heads).iter().map(|v| -(v.abs() + 0.5)).collect(),
            d: rnd(h.n_heads),
            out_proj: mat(n_embd, h.d_inner, rnd(n_embd * h.d_inner)),
        }
    }

    /// Batched rows and one-at-a-time rows agree and leave the same
    /// state, and the state carries: a second pass answers differently.
    #[test]
    fn rows_and_one_at_a_time_agree_and_leave_the_same_state() {
        let h = hp();
        let n_embd = 3;
        let mut seed = 7u32;
        let m = block(h, n_embd, &mut seed);
        let tokens: Vec<Vec<f32>> = (0..4)
            .map(|t| {
                (0..n_embd)
                    .map(|i| ((t * 3 + i) as f32 * 0.37).sin())
                    .collect()
            })
            .collect();
        let flat: Vec<f32> = tokens.concat();
        let mut s_batch = m.zero_state();
        let batched = m.forward_rows(&flat, 4, &mut s_batch, 1e-5);
        let mut s_seq = m.zero_state();
        let mut seq = Vec::new();
        for t in &tokens {
            seq.extend(m.forward_rows(t, 1, &mut s_seq, 1e-5));
        }
        for (a, b) in batched.iter().zip(&seq) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        assert_eq!(s_batch, s_seq);
        let again = m.forward_rows(&flat, 4, &mut s_batch, 1e-5);
        assert!(again
            .iter()
            .zip(&batched)
            .any(|(a, b)| (a - b).abs() > 1e-6));
    }

    /// The per-head interleave of `ssm_in`'s output is the one thing a
    /// Mamba-1 reading would get wrong silently: swapping z and x within
    /// each head changes the answer.
    #[test]
    fn z_and_x_are_interleaved_per_head() {
        let h = hp();
        let n_embd = 3;
        let mut seed = 9u32;
        let m = block(h, n_embd, &mut seed);
        let tok: Vec<f32> = vec![0.3, -0.7, 0.2];
        let mut s = m.zero_state();
        let want = m.forward_rows(&tok, 1, &mut s, 1e-5);
        // Swap the z and x halves of every head in `ssm_in`'s rows.
        let WeightMatrix::F32(t) = &m.in_proj else {
            unreachable!()
        };
        let mut rows: Vec<f32> = t.data.clone();
        let hd = h.head_dim();
        for hh in 0..h.n_heads {
            for i in 0..hd {
                for c in 0..n_embd {
                    let z_at = (hh * 2 * hd + i) * n_embd + c;
                    let x_at = (hh * 2 * hd + hd + i) * n_embd + c;
                    rows.swap(z_at, x_at);
                }
            }
        }
        let swapped = Plamo2Ssm {
            in_proj: WeightMatrix::F32(Tensor::new(rows, vec![2 * h.d_inner, n_embd])),
            ..block(h, n_embd, &mut 9u32)
        };
        let mut s2 = swapped.zero_state();
        let got = swapped.forward_rows(&tok, 1, &mut s2, 1e-5);
        assert!(want.iter().zip(&got).any(|(a, b)| (a - b).abs() > 1e-4));
    }
}
