//! The Mamba-1 block (`build_mamba_layer`, `mamba-base.cpp:4-148`): the
//! selective-scan block of Mamba, FalconMamba and Jamba, at the site
//! attention occupies on a zero-KV layer
//! ([`crate::layer_shapes::AttnShape::Mamba1`]) or on every layer of a
//! pure Mamba model.
//!
//! Called by three graphs of 155 (measured: `grep -n build_mamba_layer
//! src/models/*.cpp` is `jamba.cpp:128`, `mamba.cpp:106` and, in its
//! own spelling, `plamo2.cpp`), so this is ONE body:
//!
//! ```text
//! xz     = ssm_in(normed)                        {2 d_inner}               :34
//! x, z   = xz split                              {d_inner} each            :37-39
//! x      = silu(conv(x) + conv1d_b)              causal, width d_conv      :43-70
//! x_db   = ssm_x(x)                              {dt_rank + 2 d_state}     :74
//! dt, B, C = x_db split                          {dt_rank}, {d_state} x 2  :76-82
//! dt, B, C = rms_norm(.) [* w]   if dt_b_c_rms or the three norms exist    :84-88
//! dt     = ssm_dt(dt) + dt_b                     {d_inner}                 :90-91
//! y      = ssm_scan(state, x, softplus(dt), A, B, C)   A {d_inner, d_state}  :104-117
//! y      = y + x * D                             D {d_inner}               :121
//! y      = silu(z) * y                                                     :122
//! out    = ssm_out(y)                                                      :124
//! ```
//!
//! The scan is [`frink_core::mamba2::scan_step`] with `n_head =
//! d_inner`, `head_dim = 1`, `n_group = 1` and [`Decay::PerState`]
//! (`:18-19`; the kernel's `src3->ne[0] != 1` arm). The state is a
//! [`RecurrentState`] on the sequence's layer cache, exactly as the
//! Mamba-2 block's (`crate::mamba2`): the conv window
//! `[d_conv - 1][d_inner]` and the SSM state `[d_inner][d_state]`.
//!
//! # Norms on dt, B and C
//!
//! `:84-88`: when `{arch}.ssm.dt_b_c_rms` is true (FalconMamba; `mamba.
//! cpp:7`) OR the layer carries all three of `ssm_dt_norm`, `ssm_b_norm`,
//! `ssm_c_norm` (Jamba; `jamba.cpp:49,52-53`, REQUIRED there), each is
//! RMS-normed with that weight, or with no weight when the key set it
//! and the tensors are absent (`build_norm(x, NULL, NULL, LLM_NORM_RMS)`).
//! One of the three tensors without the other two is refused: the graph
//! would norm none of them, silently.

use frink_core::mamba2::{conv_step, scan_step, Decay, ScanDims};
use frink_core::matmul::rms_norm;
use frink_core::recurrent_state::RecurrentState;
use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::TensorSource;

use crate::loader::{load_f32_vec, load_f32_vec_optional, load_weight_matrix, LoadError};

/// The four `ssm.*` hparams a Mamba-1 graph reads (`mamba.cpp:3-6`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mamba1Hparams {
    pub d_conv: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub dt_rank: usize,
}

impl Mamba1Hparams {
    pub fn read(file: &impl TensorSource, arch: &str) -> Result<Self, LoadError> {
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
            dt_rank: read("time_step_rank")?,
        };
        if h.d_conv < 2 || h.d_inner == 0 || h.d_state == 0 || h.dt_rank == 0 {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!("ssm.* hparams {h:?}: every one must be positive, conv_kernel at least 2"),
            ));
        }
        Ok(h)
    }

    pub fn scan_dims(self) -> ScanDims {
        ScanDims {
            n_head: self.d_inner,
            head_dim: 1,
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

/// The RMS norms on dt, B and C (`mamba-base.cpp:84-88`).
pub enum DtBcNorm {
    /// Neither the key nor the tensors: `:84` is false.
    None,
    /// `ssm.dt_b_c_rms` with no tensors: a weightless RMSNorm on each.
    Weightless,
    /// The three weights (`jamba.cpp:49,52-53`).
    Weighted {
        dt: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
    },
}

/// One Mamba-1 layer's weights.
pub struct Mamba1 {
    pub h: Mamba1Hparams,
    /// `blk.N.ssm_in.weight`, `[2 d_inner, n_embd]`.
    pub in_proj: WeightMatrix,
    /// `blk.N.ssm_conv1d.weight`, `[d_inner][d_conv]`.
    pub conv1d: Vec<f32>,
    /// `blk.N.ssm_conv1d.bias`, `[d_inner]` (REQUIRED, `mamba.cpp:57`).
    pub conv1d_bias: Vec<f32>,
    /// `blk.N.ssm_x.weight`, `[dt_rank + 2 d_state, d_inner]`.
    pub x_proj: WeightMatrix,
    pub dt_bc_norm: DtBcNorm,
    /// `blk.N.ssm_dt.weight`, `[d_inner, dt_rank]`.
    pub dt_proj: WeightMatrix,
    /// `blk.N.ssm_dt.bias`, `[d_inner]`.
    pub dt_bias: Vec<f32>,
    /// `blk.N.ssm_a`, `[d_inner][d_state]`, stored negative.
    pub a: Vec<f32>,
    /// `blk.N.ssm_d`, `[d_inner]`.
    pub d: Vec<f32>,
    /// `blk.N.ssm_out.weight`, `[n_embd, d_inner]`.
    pub out_proj: WeightMatrix,
}

impl Mamba1 {
    /// Loads layer `layer`'s tensors and checks them against
    /// `mamba.cpp:55-65`'s shapes.
    pub fn load(
        file: &impl TensorSource,
        arch: &str,
        layer: usize,
        hidden_dim: usize,
    ) -> Result<Self, LoadError> {
        let h = Mamba1Hparams::read(file, arch)?;
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
                        "{}x{}; mamba.cpp:55-65 sizes it {rows}x{cols}",
                        m.rows(),
                        m.cols()
                    ),
                ));
            }
            Ok(m)
        };
        let in_proj = matrix("ssm_in.weight", 2 * h.d_inner, hidden_dim)?;
        let conv1d = load_f32_vec(file, &name("ssm_conv1d.weight"))?;
        check(
            name("ssm_conv1d.weight"),
            conv1d.len(),
            h.d_conv * h.d_inner,
        )?;
        let conv1d_bias = load_f32_vec(file, &name("ssm_conv1d.bias"))?;
        check(name("ssm_conv1d.bias"), conv1d_bias.len(), h.d_inner)?;
        let x_proj = matrix("ssm_x.weight", h.dt_rank + 2 * h.d_state, h.d_inner)?;
        let dt_norm = load_f32_vec_optional(file, &name("ssm_dt_norm.weight"))?;
        let b_norm = load_f32_vec_optional(file, &name("ssm_b_norm.weight"))?;
        let c_norm = load_f32_vec_optional(file, &name("ssm_c_norm.weight"))?;
        let dt_b_c_rms = file
            .metadata_bool(&format!("{arch}.ssm.dt_b_c_rms"))
            .unwrap_or(false);
        let dt_bc_norm = match (dt_norm, b_norm, c_norm) {
            (Some(dt), Some(b), Some(c)) => {
                check(name("ssm_dt_norm.weight"), dt.len(), h.dt_rank)?;
                check(name("ssm_b_norm.weight"), b.len(), h.d_state)?;
                check(name("ssm_c_norm.weight"), c.len(), h.d_state)?;
                DtBcNorm::Weighted { dt, b, c }
            }
            (None, None, None) if dt_b_c_rms => DtBcNorm::Weightless,
            (None, None, None) => DtBcNorm::None,
            (dt, b, c) => {
                return Err(LoadError::UnsupportedFeature(
                    arch.to_string(),
                    format!(
                        "blk.{layer}: ssm_dt_norm / ssm_b_norm / ssm_c_norm present {} / {} / {}; \
                         mamba-base.cpp:84 norms dt, B and C only when all three exist (or \
                         ssm.dt_b_c_rms is set), so a partial set would be loaded and ignored",
                        dt.is_some(),
                        b.is_some(),
                        c.is_some()
                    ),
                ));
            }
        };
        let dt_proj = matrix("ssm_dt.weight", h.d_inner, h.dt_rank)?;
        let dt_bias = load_f32_vec(file, &name("ssm_dt.bias"))?;
        check(name("ssm_dt.bias"), dt_bias.len(), h.d_inner)?;
        let a = load_f32_vec(file, &name("ssm_a"))?;
        if a.len() != h.d_inner * h.d_state {
            return Err(LoadError::UnsupportedFeature(
                name("ssm_a"),
                format!(
                    "{} elements; Mamba-1's A is {{d_state {}, d_inner {}}} (mamba.cpp:62), one \
                     decay per state element; a {{1, n_head}} A is Mamba-2's (`crate::mamba2`)",
                    a.len(),
                    h.d_state,
                    h.d_inner
                ),
            ));
        }
        let d = load_f32_vec(file, &name("ssm_d"))?;
        check(name("ssm_d"), d.len(), h.d_inner)?;
        let out_proj = matrix("ssm_out.weight", hidden_dim, h.d_inner)?;
        Ok(Self {
            h,
            in_proj,
            conv1d,
            conv1d_bias,
            x_proj,
            dt_bc_norm,
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
        let xz = if rows == 1 {
            self.in_proj.apply(normed)
        } else {
            self.in_proj.apply_batch(normed, rows)
        };
        let (d_inner, d_state, dt_rank) = (h.d_inner, h.d_state, h.dt_rank);
        let dims = h.scan_dims();
        let mut ys = vec![0.0f32; rows * d_inner];
        let mut xc = vec![0.0f32; d_inner];
        let mut y = vec![0.0f32; d_inner];
        for r in 0..rows {
            let row = &xz[r * 2 * d_inner..(r + 1) * 2 * d_inner];
            let (x, z) = row.split_at(d_inner);
            // :43-70: the conv over this token and the state, bias, SiLU.
            conv_step(&mut state.conv, &self.conv1d, h.d_conv, x, &mut xc);
            for (v, b) in xc.iter_mut().zip(&self.conv1d_bias) {
                *v = silu(*v + b);
            }
            // :74-82: dt, B, C from one projection of the conv output.
            let x_db = self.x_proj.apply(&xc);
            let (dt_low, bc) = x_db.split_at(dt_rank);
            let (b, c) = bc.split_at(d_state);
            let (dt_low, b, c) = match &self.dt_bc_norm {
                DtBcNorm::None => (dt_low.to_vec(), b.to_vec(), c.to_vec()),
                DtBcNorm::Weightless => (
                    rms_norm_weightless(dt_low, rms_eps),
                    rms_norm_weightless(b, rms_eps),
                    rms_norm_weightless(c, rms_eps),
                ),
                DtBcNorm::Weighted { dt, b: wb, c: wc } => (
                    rms_norm(dt_low, dt, rms_eps),
                    rms_norm(b, wb, rms_eps),
                    rms_norm(c, wc, rms_eps),
                ),
            };
            // :90-91: dt up to d_inner, plus the bias; softplus is the scan's.
            let mut dt = self.dt_proj.apply(&dt_low);
            for (t, bias) in dt.iter_mut().zip(&self.dt_bias) {
                *t += bias;
            }
            scan_step(
                dims,
                &mut state.ssm,
                &xc,
                &dt,
                Decay::PerState(&self.a),
                &b,
                &c,
                &mut y,
            );
            // :121-122: y + x * D, then silu(z) * y.
            for i in 0..d_inner {
                y[i] = (y[i] + xc[i] * self.d[i]) * silu(z[i]);
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

/// `build_norm(x, NULL, NULL, LLM_NORM_RMS)`: an RMSNorm with no weight.
fn rms_norm_weightless(x: &[f32], eps: f32) -> Vec<f32> {
    crate::norm::rms_norm_no_params(x, eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::Tensor;

    fn hp() -> Mamba1Hparams {
        Mamba1Hparams {
            d_conv: 3,
            d_inner: 4,
            d_state: 2,
            dt_rank: 2,
        }
    }

    #[test]
    fn the_widths_are_the_graph_s() {
        let h = hp();
        assert_eq!(h.scan_dims().n_head, 4);
        assert_eq!(h.scan_dims().head_dim, 1);
        assert_eq!(h.state_floats(), (2 * 4, 4 * 2));
    }

    /// Batched rows and one-at-a-time rows agree and leave the same
    /// state.
    #[test]
    fn rows_and_one_at_a_time_agree_and_leave_the_same_state() {
        let h = hp();
        let n_embd = 3;
        let mut seed = 11u32;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
                })
                .collect()
        };
        let mat = |rows: usize, cols: usize, v: Vec<f32>| {
            WeightMatrix::F32(Tensor::new(v, vec![rows, cols]))
        };
        let m = Mamba1 {
            h,
            in_proj: mat(2 * h.d_inner, n_embd, rnd(2 * h.d_inner * n_embd)),
            conv1d: rnd(h.d_conv * h.d_inner),
            conv1d_bias: rnd(h.d_inner),
            x_proj: mat(
                h.dt_rank + 2 * h.d_state,
                h.d_inner,
                rnd((h.dt_rank + 2 * h.d_state) * h.d_inner),
            ),
            dt_bc_norm: DtBcNorm::Weighted {
                dt: vec![1.1, 0.9],
                b: vec![1.2, 0.8],
                c: vec![0.7, 1.3],
            },
            dt_proj: mat(h.d_inner, h.dt_rank, rnd(h.d_inner * h.dt_rank)),
            dt_bias: rnd(h.d_inner),
            a: rnd(h.d_inner * h.d_state)
                .iter()
                .map(|v| -(v.abs() + 0.5))
                .collect(),
            d: rnd(h.d_inner),
            out_proj: mat(n_embd, h.d_inner, rnd(n_embd * h.d_inner)),
        };
        let tokens: Vec<Vec<f32>> = (0..4).map(|_| rnd(n_embd)).collect();
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
}
