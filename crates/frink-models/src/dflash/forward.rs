//! The draft's weights and its two passes (`dflash.cpp:813-1116`):
//! injecting the target's features into the draft KV, and the
//! noise-block forward over `[anchor, mask, ...]`.

use std::collections::HashSet;

use frink_core::attention::{apply_rope, causal_gqa_attention_row};
use frink_core::matmul::{rms_norm, rms_norm_per_head};
use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::{GgufFile, TensorSource};

use super::config::{DFlashConfig, DFlashError};
use super::selector::Selector;
use crate::decoder::Decoder;
use crate::loader::{load_f32_vec, load_weight_matrix};
use crate::norm::rms_norm_no_params;

/// One sublayer's two-tap convolution weights: `base` is
/// `{n_embd, kernel, 2}` (side 0 before the sublayer, side 1 after) and
/// `proj` predicts the per-group correction from the sublayer's input.
pub(super) struct Conv {
    base: Vec<f32>,
    proj: WeightMatrix,
}

pub(super) struct Layer {
    attn_norm: Vec<f32>,
    wq: WeightMatrix,
    wk: WeightMatrix,
    /// `None`: V is the RMS-normed raw K (`dflash.cpp:866-871`).
    wv: Option<WeightMatrix>,
    wo: WeightMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    gate: WeightMatrix,
    up: WeightMatrix,
    down: WeightMatrix,
    post_ffn_norm: Option<Vec<f32>>,
    attn_conv: Option<Conv>,
    ffn_conv: Option<Conv>,
}

/// A loaded DFlash draft. The token table and LM head are the
/// target's unless the draft ships its own (`dflash.cpp:132`, `:1056`).
pub struct DFlashDraft {
    pub(super) cfg: DFlashConfig,
    fc: WeightMatrix,
    enc_norm: Vec<f32>,
    output_norm: Vec<f32>,
    layers: Vec<Layer>,
    tok_embd: Option<WeightMatrix>,
    output: Option<WeightMatrix>,
    pub(super) selector: Option<Selector>,
}

/// One layer's injected context: K and V rows of every committed
/// position, `n_head_kv * head_dim` values a row.
#[derive(Default, Clone)]
pub(super) struct ContextKv {
    pub(super) k: Vec<f32>,
    pub(super) v: Vec<f32>,
}

/// What the noise-block forward hands the walk.
pub(super) struct BlockOutput {
    /// `[n_block, vocab]` logits, row 0 the anchor's.
    pub(super) logits: Vec<Vec<f32>>,
    /// `[n_block, n_embd]` final-normed hidden (`t_embd`).
    pub(super) hidden: Vec<Vec<f32>>,
}

struct Loader<'f, F: TensorSource> {
    file: &'f F,
    used: HashSet<String>,
}

impl<F: TensorSource> Loader<'_, F> {
    fn has(&self, name: &str) -> bool {
        self.file.find_tensor(name).is_some()
    }
    fn matrix(&mut self, name: &str) -> Result<WeightMatrix, DFlashError> {
        if !self.has(name) {
            return Err(DFlashError::MissingTensor(name.into()));
        }
        self.used.insert(name.into());
        Ok(load_weight_matrix(self.file, name)?)
    }
    fn opt_matrix(&mut self, name: &str) -> Result<Option<WeightMatrix>, DFlashError> {
        if self.has(name) {
            self.matrix(name).map(Some)
        } else {
            Ok(None)
        }
    }
    fn vec(&mut self, name: &str) -> Result<Vec<f32>, DFlashError> {
        if !self.has(name) {
            return Err(DFlashError::MissingTensor(name.into()));
        }
        self.used.insert(name.into());
        Ok(load_f32_vec(self.file, name)?)
    }
    fn opt_vec(&mut self, name: &str) -> Result<Option<Vec<f32>>, DFlashError> {
        if self.has(name) {
            self.vec(name).map(Some)
        } else {
            Ok(None)
        }
    }
}

/// Tensors that make `dflash.cpp` build a graph this file does not.
const REFUSED_TENSORS: &[(&str, &str)] = &[
    ("markov_w1.weight", "a DSpark draft (Markov head)"),
    ("d2t", "a draft with a reduced vocabulary (d2t)"),
    ("rope_freqs.weight", "a draft with rope frequency factors"),
];
const REFUSED_LAYER_TENSORS: &[(&str, &str)] = &[
    ("attn_sinks.weight", "a draft with attention sinks"),
    (
        "layer_output_scale.weight",
        "a draft with per-layer output scales",
    ),
];

impl DFlashDraft {
    /// Loads a DFlash draft GGUF. Every tensor it carries is read or
    /// refused: one left over would be part of a graph frink is not
    /// running.
    pub fn load(file: &GgufFile) -> Result<Self, DFlashError> {
        let cfg = DFlashConfig::from_gguf(file)?;
        for (name, what) in REFUSED_TENSORS {
            if file.find_tensor(name).is_some() {
                return Err(DFlashError::Unsupported((*what).into()));
            }
        }
        let mut ld = Loader {
            file,
            used: HashSet::new(),
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            for (suffix, what) in REFUSED_LAYER_TENSORS {
                if ld.has(&format!("blk.{i}.{suffix}")) {
                    return Err(DFlashError::Unsupported((*what).into()));
                }
            }
            let b = |s: &str| format!("blk.{i}.{s}");
            let conv = |ld: &mut Loader<'_, _>, base: &str, proj: &str| -> Result<_, DFlashError> {
                if cfg.conv_kernel == 0 {
                    return Ok(None);
                }
                Ok(Some(Conv {
                    base: ld.vec(&b(base))?,
                    proj: ld.matrix(&b(proj))?,
                }))
            };
            layers.push(Layer {
                attn_norm: ld.vec(&b("attn_norm.weight"))?,
                wq: ld.matrix(&b("attn_q.weight"))?,
                wk: ld.matrix(&b("attn_k.weight"))?,
                wv: ld.opt_matrix(&b("attn_v.weight"))?,
                wo: ld.matrix(&b("attn_output.weight"))?,
                q_norm: ld.vec(&b("attn_q_norm.weight"))?,
                k_norm: ld.vec(&b("attn_k_norm.weight"))?,
                post_attn_norm: ld.opt_vec(&b("post_attention_norm.weight"))?,
                ffn_norm: ld.vec(&b("ffn_norm.weight"))?,
                gate: ld.matrix(&b("ffn_gate.weight"))?,
                up: ld.matrix(&b("ffn_up.weight"))?,
                down: ld.matrix(&b("ffn_down.weight"))?,
                post_ffn_norm: ld.opt_vec(&b("post_ffw_norm.weight"))?,
                attn_conv: conv(&mut ld, "attn_conv_base", "attn_conv_proj.weight")?,
                ffn_conv: conv(&mut ld, "ffn_conv_base", "ffn_conv_proj.weight")?,
            });
        }
        let fc = ld.matrix("fc.weight")?;
        let enc_norm = ld.vec("enc.output_norm.weight")?;
        let output_norm = ld.vec("output_norm.weight")?;
        let tok_embd = ld.opt_matrix("token_embd.weight")?;
        let output = ld.opt_matrix("output.weight")?;

        // dflash.cpp:158-182: the selector is keyed off its tensor.
        let selector = if ld.has("selector_hidden.weight") {
            if !cfg.has_selector() || cfg.conv_kernel == 0 {
                return Err(DFlashError::Unsupported(
                    "a DFlash2 draft missing its conv/selector metadata".into(),
                ));
            }
            // dflash.cpp:172: the lattice is packed into a hidden row.
            if cfg.n_embd < cfg.selector_top_k * (cfg.selector_top_k + 1) {
                return Err(DFlashError::Unsupported(
                    "a DFlash2 draft too narrow for its selector lattice".into(),
                ));
            }
            Some(Selector {
                hidden: ld.matrix("selector_hidden.weight")?,
                prev: ld.matrix("selector_predecessor.weight")?,
                next: ld.matrix("selector_successor.weight")?,
                top_k: cfg.selector_top_k,
            })
        } else {
            if cfg.has_selector() {
                return Err(DFlashError::MissingTensor("selector_hidden.weight".into()));
            }
            None
        };

        let leftover: Vec<&str> = file
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .filter(|n| !ld.used.contains(*n))
            .collect();
        if !leftover.is_empty() {
            return Err(DFlashError::Unsupported(format!(
                "a draft with tensors frink does not read ({})",
                leftover.join(", ")
            )));
        }
        let draft = Self {
            cfg,
            fc,
            enc_norm,
            output_norm,
            layers,
            tok_embd,
            output,
            selector,
        };
        draft.check_shapes()?;
        Ok(draft)
    }

    fn check_shapes(&self) -> Result<(), DFlashError> {
        let c = &self.cfg;
        let bad = |what: String| Err(DFlashError::Mismatch(what));
        let kv = c.n_head_kv * c.head_dim;
        for (i, l) in self.layers.iter().enumerate() {
            let ok = l.wq.rows() == c.n_head * c.head_dim
                && l.wq.cols() == c.n_embd
                && l.wk.rows() == kv
                && l.wv.as_ref().is_none_or(|v| v.rows() == kv)
                && l.wo.cols() == c.n_head * c.head_dim
                && l.wo.rows() == c.n_embd
                && l.q_norm.len() == c.head_dim
                && l.k_norm.len() == c.head_dim
                && l.gate.rows() == c.n_ff
                && l.down.rows() == c.n_embd;
            if !ok {
                return bad(format!("layer {i}'s projections do not match the metadata"));
            }
            for conv in [&l.attn_conv, &l.ffn_conv].into_iter().flatten() {
                let groups = c.n_embd / c.conv_group;
                if conv.base.len() != c.n_embd * c.conv_kernel * 2
                    || conv.proj.rows() != 2 * c.conv_kernel * groups
                {
                    return bad(format!(
                        "layer {i}'s convolution does not match the metadata"
                    ));
                }
            }
        }
        if self.fc.rows() != c.n_embd || !self.fc.cols().is_multiple_of(c.target_layers.len()) {
            return bad("the fc projection does not match the target layer count".into());
        }
        Ok(())
    }

    pub fn config(&self) -> &DFlashConfig {
        &self.cfg
    }

    /// Checks the draft against the target it will draft for, before
    /// anything runs: the widths it shares and the layers it reads.
    pub fn check_target(&self, target: &Decoder) -> Result<(), DFlashError> {
        let c = &self.cfg;
        let th = target.config.hidden_dim;
        if self.fc.cols() != c.target_features(th) {
            return Err(DFlashError::Mismatch(format!(
                "it reads {} values per position and the target's {} layers of width {th} give {}",
                self.fc.cols(),
                c.target_layers.len(),
                c.target_features(th)
            )));
        }
        let embd = self.tok_embd.as_ref().unwrap_or(&target.embedding);
        let head = self.output.as_ref().unwrap_or(&target.output_head);
        if embd.cols() != c.n_embd || head.cols() != c.n_embd {
            return Err(DFlashError::Mismatch(format!(
                "its hidden width {} is not the width of the token table and LM head it reads ({} \
                 and {})",
                c.n_embd,
                embd.cols(),
                head.cols()
            )));
        }
        if c.mask_token >= embd.rows() {
            return Err(DFlashError::Mismatch(format!(
                "its mask token {} is outside the {}-token table",
                c.mask_token,
                embd.rows()
            )));
        }
        if let Some(sel) = &self.selector {
            if sel.prev.rows() != head.rows() || sel.next.rows() != head.rows() {
                return Err(DFlashError::Mismatch(
                    "its selector codebooks do not cover the target vocabulary".into(),
                ));
            }
        }
        target.check_layer_taps(&c.target_layers)?;
        Ok(())
    }

    /// `dflash.cpp:850-910`: the target features of `rows` positions
    /// starting at `start_pos`, through `fc` and the encoder norm, then
    /// each layer's K/V, appended to `ctx`.
    pub(super) fn inject(&self, rows: &[Vec<f32>], start_pos: usize, ctx: &mut [ContextKv]) {
        let c = &self.cfg;
        let n = rows.len();
        if n == 0 {
            return;
        }
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let g = self.fc.apply_batch(&flat, n);
        let g: Vec<f32> = g
            .chunks(c.n_embd)
            .flat_map(|r| rms_norm(r, &self.enc_norm, c.rms_eps))
            .collect();
        for (layer, kv) in self.layers.iter().zip(ctx.iter_mut()) {
            let k_raw = layer.wk.apply_batch(&g, n);
            let v_raw = layer.wv.as_ref().map(|wv| wv.apply_batch(&g, n));
            self.push_kv(layer, &k_raw, v_raw.as_deref(), n, start_pos, kv);
        }
    }

    /// K normed and rotated, V projected or (shared) RMS-normed raw K,
    /// for `n` rows at `start_pos..`.
    fn kv_rows(
        &self,
        layer: &Layer,
        k_raw: &[f32],
        v_raw: Option<&[f32]>,
        n: usize,
        start_pos: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let w = c.n_head_kv * c.head_dim;
        let mut k = Vec::with_capacity(n * w);
        let mut v = Vec::with_capacity(n * w);
        for b in 0..n {
            let raw = &k_raw[b * w..(b + 1) * w];
            let mut kr = rms_norm_per_head(raw, &layer.k_norm, c.head_dim, c.rms_eps);
            for head in kr.chunks_mut(c.head_dim) {
                apply_rope(&mut head[..c.n_rot], start_pos + b, c.rope_theta);
            }
            k.extend_from_slice(&kr);
            match v_raw {
                Some(vr) => v.extend_from_slice(&vr[b * w..(b + 1) * w]),
                None => {
                    for head in raw.chunks(c.head_dim) {
                        v.extend(rms_norm_no_params(head, c.rms_eps));
                    }
                }
            }
        }
        (k, v)
    }

    fn push_kv(
        &self,
        layer: &Layer,
        k_raw: &[f32],
        v_raw: Option<&[f32]>,
        n: usize,
        start_pos: usize,
        kv: &mut ContextKv,
    ) {
        let (k, v) = self.kv_rows(layer, k_raw, v_raw, n, start_pos);
        kv.k.extend_from_slice(&k);
        kv.v.extend_from_slice(&v);
    }

    fn token_row(&self, target: &Decoder, token: usize) -> Vec<f32> {
        let mut row = self
            .tok_embd
            .as_ref()
            .unwrap_or(&target.embedding)
            .dequant_row(token);
        if let Some(s) = self.cfg.embedding_scale {
            row.iter_mut().for_each(|x| *x *= s);
        }
        row
    }

    /// `dflash.cpp:913-1110`: the block `[anchor, mask * (n_block-1)]`
    /// at positions `pos0..`, attending over the injected context of
    /// positions `0..pos0` and over the whole block, non-causally
    /// unless the draft says otherwise. The block's own K/V is not
    /// kept: `common_speculative` removes it before the target's
    /// verified features are injected over the same positions.
    pub(super) fn block_forward(
        &self,
        target: &Decoder,
        anchor: usize,
        n_block: usize,
        pos0: usize,
        ctx: &[ContextKv],
    ) -> BlockOutput {
        let c = &self.cfg;
        let e = c.n_embd;
        let kvw = c.n_head_kv * c.head_dim;
        let qw = c.n_head * c.head_dim;
        let mut x: Vec<f32> = (0..n_block)
            .flat_map(|i| self.token_row(target, if i == 0 { anchor } else { c.mask_token }))
            .collect();

        for (il, layer) in self.layers.iter().enumerate() {
            let norm_rows = |src: &[f32], w: &[f32]| -> Vec<f32> {
                src.chunks(e)
                    .flat_map(|r| rms_norm(r, w, c.rms_eps))
                    .collect()
            };
            let mut xn = norm_rows(&x, &layer.attn_norm);
            let attn_dyn = layer.attn_conv.as_ref().map(|cv| {
                let d = cv.proj.apply_batch(&xn, n_block);
                xn = conv(&xn, &d, &cv.base, 0, c, n_block);
                d
            });

            let q_raw = layer.wq.apply_batch(&xn, n_block);
            let k_raw = layer.wk.apply_batch(&xn, n_block);
            let v_raw = layer.wv.as_ref().map(|wv| wv.apply_batch(&xn, n_block));
            let (bk, bv) = self.kv_rows(layer, &k_raw, v_raw.as_deref(), n_block, pos0);
            let mut q = Vec::with_capacity(n_block * qw);
            for b in 0..n_block {
                let mut qr = rms_norm_per_head(
                    &q_raw[b * qw..(b + 1) * qw],
                    &layer.q_norm,
                    c.head_dim,
                    c.rms_eps,
                );
                for head in qr.chunks_mut(c.head_dim) {
                    apply_rope(&mut head[..c.n_rot], pos0 + b, c.rope_theta);
                }
                // `causal_gqa_attention_row` scales by 1/sqrt(head_dim);
                // a draft with its own scale is folded into Q.
                let fold = c.kq_scale * (c.head_dim as f32).sqrt();
                if fold != 1.0 {
                    qr.iter_mut().for_each(|v| *v *= fold);
                }
                q.extend_from_slice(&qr);
            }

            // The context before the anchor. Rows past it, which a
            // caller whose target has already committed the anchor
            // holds, are not read.
            let ctx_kv = &ctx[il];
            assert!(
                ctx_kv.k.len() >= pos0 * kvw,
                "the draft context must hold every position before the block"
            );
            let ctx_len = pos0;
            let window = c.sliding_window.filter(|_| c.swa_layers[il]);
            let mut attn = Vec::with_capacity(n_block * qw);
            for b in 0..n_block {
                let qpos = pos0 + b;
                // `is_masked_swa` (standard): a key at p0 is masked when
                // qpos - p0 >= window; later keys never are.
                let first = window
                    .map_or(0, |w| (qpos + 1).saturating_sub(w))
                    .min(ctx_len);
                let visible = if c.causal { b + 1 } else { n_block };
                let mut keys = Vec::with_capacity((ctx_len - first + visible) * kvw);
                let mut vals = Vec::with_capacity(keys.capacity());
                keys.extend_from_slice(&ctx_kv.k[first * kvw..ctx_len * kvw]);
                vals.extend_from_slice(&ctx_kv.v[first * kvw..ctx_len * kvw]);
                // A block key behind the window is masked too.
                let block_first = window.map_or(0, |w| (b + 1).saturating_sub(w).min(visible));
                keys.extend_from_slice(&bk[block_first * kvw..visible * kvw]);
                vals.extend_from_slice(&bv[block_first * kvw..visible * kvw]);
                let seq = keys.len() / kvw;
                attn.extend(causal_gqa_attention_row(
                    &q[b * qw..(b + 1) * qw],
                    &keys[..],
                    &vals[..],
                    c.n_head,
                    c.n_head_kv,
                    c.head_dim,
                    c.head_dim,
                    seq,
                    None,
                    None,
                    None,
                    None,
                ));
            }
            let mut cur = layer.wo.apply_batch(&attn, n_block);
            if let Some(s) = c.value_scale {
                cur.iter_mut().for_each(|v| *v *= s);
            }
            if let (Some(cv), Some(d)) = (&layer.attn_conv, &attn_dyn) {
                cur = conv(&cur, d, &cv.base, 1, c, n_block);
            }
            if let Some(w) = &layer.post_attn_norm {
                cur = norm_rows(&cur, w);
            }
            let ffn_inp: Vec<f32> = cur.iter().zip(&x).map(|(a, b)| a + b).collect();

            let mut h = norm_rows(&ffn_inp, &layer.ffn_norm);
            let ffn_dyn = layer.ffn_conv.as_ref().map(|cv| {
                let d = cv.proj.apply_batch(&h, n_block);
                h = conv(&h, &d, &cv.base, 0, c, n_block);
                d
            });
            let gate = layer.gate.apply_batch(&h, n_block);
            let up = layer.up.apply_batch(&h, n_block);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let mut out = layer.down.apply_batch(&act, n_block);
            if let (Some(cv), Some(d)) = (&layer.ffn_conv, &ffn_dyn) {
                out = conv(&out, d, &cv.base, 1, c, n_block);
            }
            if let Some(w) = &layer.post_ffn_norm {
                out = norm_rows(&out, w);
            }
            x = out.iter().zip(&ffn_inp).map(|(a, b)| a + b).collect();
        }

        let hidden: Vec<Vec<f32>> = x
            .chunks(e)
            .map(|r| rms_norm(r, &self.output_norm, c.rms_eps))
            .collect();
        let flat: Vec<f32> = hidden.iter().flatten().copied().collect();
        let head = self.output.as_ref().unwrap_or(&target.output_head);
        let mut logits = head.apply_batch(&flat, n_block);
        // dflash.cpp:1075-1086: only DFlash2 feeds logits to the
        // selector, so only it applies the output transforms.
        if self.selector.is_some() {
            for v in logits.iter_mut() {
                if let Some(s) = c.logit_scale {
                    *v *= s;
                }
                if let Some(cap) = c.final_logit_softcap {
                    *v = (*v / cap).tanh() * cap;
                }
            }
        }
        let vocab = head.rows();
        BlockOutput {
            logits: logits.chunks(vocab).map(<[f32]>::to_vec).collect(),
            hidden,
        }
    }
}

/// `build_dflash2_conv` (`dflash.cpp:493-556`) for one block of
/// `n_block` rows:
///
/// `out[i, c] = sum_t (base[c, t, side] + dyn[i, g(c), t, side]) * x[i - t, c]`
///
/// with `x[i - t] = 0` before the block's first row, `g(c) = c /
/// group`, `dyn` laid out `[n_groups, kernel, 2]` per row and `base`
/// `[n_embd, kernel, 2]`.
pub(super) fn conv(
    x: &[f32],
    dynamic: &[f32],
    base: &[f32],
    side: usize,
    c: &DFlashConfig,
    n_block: usize,
) -> Vec<f32> {
    let e = c.n_embd;
    let k = c.conv_kernel;
    let groups = e / c.conv_group;
    let d = 2 * k * groups;
    let mut out = vec![0.0f32; n_block * e];
    // Taps at or past the block only read the left padding.
    for t in 0..k.min(n_block) {
        for i in t..n_block {
            let coeff = &dynamic[i * d + groups * (t + k * side)..][..groups];
            let base_t = &base[e * (t + k * side)..][..e];
            let src = &x[(i - t) * e..(i - t + 1) * e];
            let dst = &mut out[i * e..(i + 1) * e];
            for ch in 0..e {
                dst[ch] += (base_t[ch] + coeff[ch / c.conv_group]) * src[ch];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(e: usize, k: usize, group: usize) -> DFlashConfig {
        DFlashConfig {
            n_embd: e,
            n_layer: 1,
            n_head: 1,
            n_head_kv: 1,
            head_dim: e,
            n_ff: e,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            n_rot: e,
            block_size: 4,
            conv_kernel: k,
            conv_group: group,
            selector_rank: 0,
            selector_top_k: 0,
            target_layers: vec![1],
            sliding_window: None,
            swa_layers: vec![false],
            causal: false,
            mask_token: 0,
            kq_scale: 1.0,
            embedding_scale: None,
            value_scale: None,
            logit_scale: None,
            final_logit_softcap: None,
        }
    }

    /// The formula written out by hand for a 3-row block, two taps, two
    /// groups of two channels, side 1: every index the layout has to
    /// get right (side, tap, group, the left padding) is distinct.
    #[test]
    fn conv_matches_the_formula_index_by_index() {
        let c = cfg(4, 2, 2);
        let (n, e, k, groups) = (3usize, 4usize, 2usize, 2usize);
        let x: Vec<f32> = (0..n * e).map(|i| 1.0 + i as f32).collect();
        let d = 2 * k * groups;
        let dynamic: Vec<f32> = (0..n * d).map(|i| 0.01 * i as f32).collect();
        let base: Vec<f32> = (0..e * k * 2).map(|i| 0.1 * i as f32 - 0.5).collect();
        let side = 1;
        let got = conv(&x, &dynamic, &base, side, &c, n);
        for i in 0..n {
            for ch in 0..e {
                let mut want = 0.0f32;
                for t in 0..k {
                    if i < t {
                        continue;
                    }
                    let w = base[ch + e * (t + k * side)]
                        + dynamic[i * d + ch / 2 + groups * (t + k * side)];
                    want += w * x[(i - t) * e + ch];
                }
                assert!(
                    (got[i * e + ch] - want).abs() < 1e-5,
                    "row {i} channel {ch}"
                );
            }
        }
    }

    /// Row 0 has no predecessor in its block: only tap 0 reaches it.
    #[test]
    fn the_first_row_reads_no_predecessor() {
        let c = cfg(2, 2, 1);
        let x = vec![1.0, 1.0, 100.0, 100.0];
        let dynamic = vec![0.0; 2 * 8];
        let base = vec![1.0; 2 * 2 * 2];
        let out = conv(&x, &dynamic, &base, 0, &c, 2);
        assert_eq!(&out[..2], &[1.0, 1.0]);
        assert_eq!(&out[2..], &[101.0, 101.0]);
    }
}
