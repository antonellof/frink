//! The Mamba-2 block: the recurrent layer of the Mamba-2 hybrids
//! (`granite-hybrid`, `nemotron-h`, `falcon-h1`) and of `mamba2` itself,
//! at the site attention occupies on a layer whose `head_count_kv` is
//! zero ([`crate::layer_shapes::AttnShape::Mamba2`]).
//!
//! `build_mamba2_layer` (`mamba-base.cpp:149-288`) is one body every
//! graph that has the block calls (measured: `grep -n build_mamba2_layer
//! src/models/*.cpp` is `granite-hybrid.cpp:163`, `falcon-h1.cpp:161`,
//! `nemotron-h.cpp:148`, `mamba.cpp:104`), so this is ONE body too:
//!
//! ```text
//! zxBCdt = ssm_in(normed)                       {2 d_inner + 2 n_group d_state + n_head}   :180
//! z      = zxBCdt[.. d_inner]                   {n_head, head_dim}                         :184
//! xBC    = zxBCdt[d_inner .. d_inner + w]       w = d_inner + 2 n_group d_state            :186
//! dt     = zxBCdt[d_inner + w ..]               {n_head}                                   :188
//! xBC    = silu(conv(xBC) + conv1d_b)           causal, width d_conv, state = last d_conv-1 :195-226
//! x, B, C = xBC split                           {n_head, head_dim}, {n_group, d_state} x 2  :231-236
//! dt     = dt + dt_b                                                                        :239
//! y      = ssm_scan(state, x, softplus(dt), A, B, C)   frink_core::mamba2::scan_rows      :244-258
//! y      = y + x * D                             D {n_head}, per head                       :268
//! y      = silu(z) * y                           ggml_swiglu_split                          :269
//! y      = rms_norm(y per group) * ssm_norm      {d_inner / n_group, n_group}, if present   :271-274
//! out    = ssm_out(y)                                                                       :278
//! ```
//!
//! The state is [`frink_core::recurrent_state::RecurrentState`] on the
//! sequence's layer cache: the conv window `[d_conv - 1][w]` and the
//! SSM state `[n_head][head_dim][d_state]`, zero for a new sequence,
//! created here on first use at the size these weights name. The KV
//! cache also counts the sequence's positions for the layer (one empty
//! push per token), because `positions()` of a layer's cache is what
//! the server reads as "how far this row has got".
//!
//! # What is served and what is not
//!
//! The four hparams `ssm.conv_kernel`, `ssm.inner_size`,
//! `ssm.state_size`, `ssm.time_step_rank` (the head count) and
//! `ssm.group_count`, as every Mamba-2 converter writes them. `ssm_a` is
//! `{1, n_head}` (one scalar per head, `granite-hybrid.cpp:67`); a
//! `{d_state, n_head}` A is Mamba-1's (`jamba`, `plamo2`, `mamba`) and
//! is refused by name here until `build_mamba_layer` has a body.
//! `ssm_conv1d.bias` is `TENSOR_NOT_REQUIRED` at `granite-hybrid.cpp:63`
//! and then `ggml_add`ed unconditionally at `mamba-base.cpp:222`:
//! libllama SEGFAULTS on a file without it (measured, `scripts/
//! make_granite_hybrid_fixture.py --no-conv-bias`), so frink REQUIRES
//! it rather than run a file its reference cannot. `ssm_norm` is
//! required by `granite-hybrid.cpp:68` and optional in
//! `mamba-base.cpp:271`, so it is loaded when present and applied when
//! loaded.

use frink_core::mamba2::{conv_silu_rows, scan_rows_state, Decay, ScanDims, ScanRows};

/// Architectures that run the Mamba-2 block IN PARALLEL with attention
/// on every layer, both reading the same `attn_norm` output, the two
/// outputs summed before the residual add: `falcon-h1.cpp:137-161`
/// (`:12` marks every layer recurrent AND every layer has heads). One
/// graph of 155 (measured: the other three `build_mamba2_layer` callers
/// branch on the layer kind). Such a layer is `AttnShape::Gqa` with
/// `AttnWeights::mamba2` set, and the block keeps its state on the
/// same cache the attention rows live in without counting positions
/// (attention counts them).
pub const PARALLEL_WITH_ATTENTION: &[&str] = &["falcon-h1"];

/// True when every attention layer of `arch` also runs the block.
pub fn parallel_with_attention(arch: &str) -> bool {
    PARALLEL_WITH_ATTENTION.contains(&arch)
}
use frink_core::recurrent_state::RecurrentState;
use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::TensorSource;

use crate::loader::{load_f32_vec, load_f32_vec_optional, load_weight_matrix, LoadError};

/// The five `ssm.*` hparams, as one value so a loader cannot read four.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mamba2Hparams {
    pub d_conv: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub n_head: usize,
    pub n_group: usize,
}

impl Mamba2Hparams {
    /// Reads `{arch}.ssm.*`; every key required, as `granite-hybrid.cpp:
    /// 8-12` reads them.
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
            n_head: read("time_step_rank")?,
            n_group: read("group_count")?,
        };
        if h.n_head == 0
            || !h.d_inner.is_multiple_of(h.n_head)
            || h.n_group == 0
            || !h.d_inner.is_multiple_of(h.n_group)
        {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!(
                    "ssm.inner_size {} must be a multiple of ssm.time_step_rank {} and of \
                     ssm.group_count {} (mamba-base.cpp:167-168)",
                    h.d_inner, h.n_head, h.n_group
                ),
            ));
        }
        if h.d_conv < 2 {
            return Err(LoadError::UnsupportedFeature(
                key("conv_kernel"),
                format!("{}: the conv state is d_conv - 1 rows", h.d_conv),
            ));
        }
        Ok(h)
    }

    pub fn head_dim(self) -> usize {
        self.d_inner / self.n_head
    }

    /// The conv's channel count, `d_inner + 2 n_group d_state`.
    pub fn conv_width(self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }

    /// `ssm_in`'s rows: `2 d_inner + 2 n_group d_state + n_head`.
    pub fn in_proj_rows(self) -> usize {
        2 * self.d_inner + 2 * self.n_group * self.d_state + self.n_head
    }

    pub fn scan_dims(self) -> ScanDims {
        ScanDims {
            n_head: self.n_head,
            head_dim: self.head_dim(),
            d_state: self.d_state,
            n_group: self.n_group,
        }
    }

    /// Floats of state per sequence per layer: `n_embd_r + n_embd_s`.
    pub fn state_floats(self) -> (usize, usize) {
        (
            (self.d_conv - 1) * self.conv_width(),
            self.scan_dims().state_len(),
        )
    }
}

/// One Mamba-2 layer's weights.
pub struct Mamba2 {
    pub h: Mamba2Hparams,
    /// `blk.N.ssm_in.weight`, `[in_proj_rows, n_embd]`.
    pub in_proj: WeightMatrix,
    /// `blk.N.ssm_conv1d.weight`, `[conv_width][d_conv]`.
    pub conv1d: Vec<f32>,
    /// `blk.N.ssm_conv1d.bias`, `[conv_width]`. Required: see the
    /// module doc.
    pub conv1d_bias: Vec<f32>,
    /// `blk.N.ssm_dt.bias`, `[n_head]`.
    pub dt_bias: Vec<f32>,
    /// `blk.N.ssm_a`, `[n_head]`, stored negative (`-exp(A_log)`).
    pub a: Vec<f32>,
    /// `blk.N.ssm_d`, `[n_head]`.
    pub d: Vec<f32>,
    /// `blk.N.ssm_norm.weight`, `[d_inner]` read as `[n_group][d_inner / n_group]`.
    pub norm: Option<Vec<f32>>,
    /// `blk.N.ssm_out.weight`, `[n_embd, d_inner]`.
    pub out_proj: WeightMatrix,
}

impl Mamba2 {
    /// Loads layer `layer`'s tensors and checks them against
    /// `granite-hybrid.cpp:60-69`'s shapes.
    pub fn load(
        file: &impl TensorSource,
        arch: &str,
        layer: usize,
        hidden_dim: usize,
    ) -> Result<Self, LoadError> {
        let h = Mamba2Hparams::read(file, arch)?;
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
        let in_proj = load_weight_matrix(file, &name("ssm_in.weight"))?;
        if in_proj.rows() != h.in_proj_rows() || in_proj.cols() != hidden_dim {
            return Err(LoadError::UnsupportedFeature(
                name("ssm_in.weight"),
                format!(
                    "{}x{}; granite-hybrid.cpp:61 sizes it {}x{hidden_dim}",
                    in_proj.rows(),
                    in_proj.cols(),
                    h.in_proj_rows()
                ),
            ));
        }
        let conv1d = load_f32_vec(file, &name("ssm_conv1d.weight"))?;
        check(
            name("ssm_conv1d.weight"),
            conv1d.len(),
            h.d_conv * h.conv_width(),
        )?;
        let conv1d_bias = match load_f32_vec_optional(file, &name("ssm_conv1d.bias"))? {
            Some(b) => b,
            None => {
                return Err(LoadError::UnsupportedFeature(
                    name("ssm_conv1d.bias"),
                    "absent. granite-hybrid.cpp:63 creates it TENSOR_NOT_REQUIRED and \
                     mamba-base.cpp:222 adds it unconditionally: libllama segfaults on a file \
                     without it (measured), so frink requires it"
                        .to_string(),
                ));
            }
        };
        check(name("ssm_conv1d.bias"), conv1d_bias.len(), h.conv_width())?;
        let dt_bias = load_f32_vec(file, &name("ssm_dt.bias"))?;
        check(name("ssm_dt.bias"), dt_bias.len(), h.n_head)?;
        let a = load_f32_vec(file, &name("ssm_a"))?;
        if a.len() != h.n_head {
            return Err(LoadError::UnsupportedFeature(
                name("ssm_a"),
                format!(
                    "{} elements for {} heads: Mamba-2's A is one scalar per head \
                     (granite-hybrid.cpp:66, `{{1, n_head}}`); a `{{d_state, n_head}}` A is \
                     Mamba-1's (`build_mamba_layer`), which frink has no body for",
                    a.len(),
                    h.n_head
                ),
            ));
        }
        let d = load_f32_vec(file, &name("ssm_d"))?;
        check(name("ssm_d"), d.len(), h.n_head)?;
        let norm = load_f32_vec_optional(file, &name("ssm_norm.weight"))?;
        if let Some(n) = &norm {
            check(name("ssm_norm.weight"), n.len(), h.d_inner)?;
        }
        let out_proj = load_weight_matrix(file, &name("ssm_out.weight"))?;
        if out_proj.rows() != hidden_dim || out_proj.cols() != h.d_inner {
            return Err(LoadError::UnsupportedFeature(
                name("ssm_out.weight"),
                format!(
                    "{}x{}; granite-hybrid.cpp:69 sizes it {hidden_dim}x{}",
                    out_proj.rows(),
                    out_proj.cols(),
                    h.d_inner
                ),
            ));
        }
        Ok(Self {
            h,
            in_proj,
            conv1d,
            conv1d_bias,
            dt_bias,
            a,
            d,
            norm,
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
    /// Returns the branch's contribution to the residual.
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
        let zxbcdt = if rows == 1 {
            self.in_proj.apply(normed)
        } else {
            self.in_proj.apply_batch(normed, rows)
        };
        let (d_inner, w, n_head, n_group) = (h.d_inner, h.conv_width(), h.n_head, h.n_group);
        let stride = h.in_proj_rows();
        let dims = h.scan_dims();
        let head_dim = h.head_dim();
        let mut ys = vec![0.0f32; rows * d_inner];
        // :195-226: the conv over every row and the state, its bias,
        // SiLU, parallel over channels.
        let mut xbc_all = vec![0.0f32; rows * w];
        conv_silu_rows(
            &mut state.conv,
            &self.conv1d,
            &self.conv1d_bias,
            h.d_conv,
            &zxbcdt[d_inner..],
            stride,
            rows,
            &mut xbc_all,
        );
        // :239: dt + dt_b; the softplus is the scan's.
        let mut dt_all = vec![0.0f32; rows * n_head];
        for (r, dt) in dt_all.chunks_exact_mut(n_head).enumerate() {
            let row = &zxbcdt[r * stride + d_inner + w..(r + 1) * stride];
            for (o, (t, bias)) in dt.iter_mut().zip(row.iter().zip(&self.dt_bias)) {
                *o = t + bias;
            }
        }
        // :244-258: the whole batch's scan in one parallel region. The
        // conv output is already `[x | B | C]`, the order it reads.
        scan_rows_state(
            dims,
            &mut state.ssm,
            ScanRows {
                xbc: &xbc_all,
                stride: w,
                dt: &dt_all,
                rows,
            },
            Decay::PerHead(&self.a),
            &mut ys,
        );
        // Rows are independent from here to `ssm_out`.
        frink_core::par::chunks_mut(&mut ys, d_inner, 1, |r, y| {
            let z = &zxbcdt[r * stride..r * stride + d_inner];
            let x = &xbc_all[r * w..r * w + d_inner];
            // :268-269: y + x * D per head, then silu(z) * y.
            for (hd, dd) in self.d.iter().enumerate() {
                for i in hd * head_dim..(hd + 1) * head_dim {
                    y[i] += x[i] * dd;
                    y[i] *= silu(z[i]);
                }
            }
            // :271-274: an RMSNorm over each group's d_inner / n_group
            // channels, times the group's slice of ssm_norm.
            if let Some(nw) = &self.norm {
                let gw = d_inner / n_group;
                for g in 0..n_group {
                    let seg = &mut y[g * gw..(g + 1) * gw];
                    let scale = frink_core::matmul::rms_scale(seg, rms_eps);
                    for (v, wv) in seg.iter_mut().zip(&nw[g * gw..(g + 1) * gw]) {
                        *v = *v * scale * wv;
                    }
                }
            }
        });
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

    fn hp() -> Mamba2Hparams {
        Mamba2Hparams {
            d_conv: 3,
            d_inner: 4,
            d_state: 2,
            n_head: 2,
            n_group: 1,
        }
    }

    #[test]
    fn the_widths_are_the_graph_s() {
        let h = hp();
        assert_eq!(h.head_dim(), 2);
        assert_eq!(h.conv_width(), 4 + 2 * 2);
        assert_eq!(h.in_proj_rows(), 8 + 4 + 2);
        assert_eq!(h.state_floats(), (2 * 8, 2 * 2 * 2));
    }

    /// The block one token at a time agrees with itself batched, and
    /// the state after both is the same: the batched body is the row
    /// body in a loop, not a second implementation.
    #[test]
    fn rows_and_one_at_a_time_agree_and_leave_the_same_state() {
        let h = hp();
        let n_embd = 3;
        let mut seed = 7u32;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
                })
                .collect()
        };
        let m = Mamba2 {
            h,
            in_proj: WeightMatrix::F32(Tensor::new(
                rnd(h.in_proj_rows() * n_embd),
                vec![h.in_proj_rows(), n_embd],
            )),
            conv1d: rnd(h.d_conv * h.conv_width()),
            conv1d_bias: rnd(h.conv_width()),
            dt_bias: rnd(h.n_head),
            a: vec![-0.5, -1.5],
            d: rnd(h.n_head),
            norm: Some(rnd(h.d_inner).iter().map(|v| v + 1.0).collect()),
            out_proj: WeightMatrix::F32(Tensor::new(
                rnd(n_embd * h.d_inner),
                vec![n_embd, h.d_inner],
            )),
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
        // The state moved: a second pass over the same tokens differs.
        let again = m.forward_rows(&flat, 4, &mut s_batch, 1e-5);
        assert!(again
            .iter()
            .zip(&batched)
            .any(|(a, b)| (a - b).abs() > 1e-6));
    }
}
