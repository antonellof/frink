//! Gemma-4 on the fused Metal dense stacks (#626).
//!
//! The host body in `gemma4_engine.rs` is the reference. This file only
//! DESCRIBES a Gemma-4 layer to the two stacks every dense model already
//! runs on (`frink_metal::attn::launch_decode_dense_stack` and
//! `launch_prefill_dense_stack`), which were parameterised for what
//! Gemma-4 adds rather than copied (ADR 0004):
//!
//! | Gemma-4 (`gemma4.cpp`)                    | stack field                         |
//! |-------------------------------------------|-------------------------------------|
//! | shared-KV layers (`shared_kv_layers`)     | `LayerKv::Shared { slot }`          |
//! | SWA 256-wide / full 512-wide heads        | per-slot `MetalKvBuffers` geometry  |
//! | sliding window (`is_masked_swa` STANDARD) | `window`, decode and prefill        |
//! | `rope_freqs` on full layers only          | per-layer `LayerRope`               |
//! | `n_rot(il)` (`rope.dimension_count[_swa]`)| `MetalRope::rot_dim`, one per run   |
//! | `f_attention_scale = 1.0`                 | folded into the Q norm (below)      |
//! | weightless RMSNorm on V                   | `AttnExtras::v_norm` of ones        |
//! | per-layer embeddings and their gate       | `PleInputMetal`, `PerLayerGateMetal`|
//! | `layer_output_scale`                      | `out_scale`                         |
//! | sandwich norms, GELU FFN, logit softcap   | already there                       |
//!
//! The attention scale: every Metal attention kernel scores with
//! `1/sqrt(head_dim)`, and Gemma-4 wants `1.0`. The host body multiplies
//! the roped Q by `attention_scale * sqrt(head_dim)`; RoPE is linear, so
//! multiplying the Q norm's weight by the same factor before it is the
//! same arithmetic, and it costs no dispatch.
//!
//! The whole prompt is batched: the prefill attention kernels take the
//! sliding window, so a chunk past the window needs no per-token
//! fallback. What stays off the device: the token embedding and its
//! per-layer row are gathered on the host, as llama.cpp does for the
//! per-layer table.

use frink_core::matmul::{rms_norm, softcap_inplace};
use frink_core::weight_matrix::WeightMatrix;
use frink_metal::attn::{
    AttnExtras, AttnMixer, DenseLayerMetal, LayerKv, MetalKvBuffers, PerLayerGateMetal,
    PleInputMetal, PrefillDenseLayerMetal, PrefillFfnMetal, PrefillMatMetal, TokenMixer,
};
use frink_metal::rope::{LayerRope, MetalRope, MetalRopeLayout};

use crate::gemma4_engine::{Gemma4DecodeState, Gemma4Engine};

/// Largest prompt chunk one prefill command buffer takes.
const PREFILL_CHUNK: usize = 512;

/// Smallest chunk the prefill stack accepts (`launch_prefill_dense_stack`).
const PREFILL_MIN: usize = 4;

/// Engine constants the stacks read; see [`Gemma4Engine::device_aux`].
pub(crate) struct DeviceAux {
    /// Per layer: `q_norm * attention_scale * sqrt(head_dim)`.
    q_norm: Vec<Vec<f32>>,
    /// `head_dim` ones per layer, V's weightless norm.
    ones: Vec<Vec<f32>>,
}

/// The device half of a [`Gemma4DecodeState`].
#[derive(Default)]
pub(crate) struct Gemma4MetalState {
    kv: Option<DeviceKv>,
}

struct DeviceKv {
    /// One per KV-owning layer, indexed by layer id (owners are the
    /// first `n_layer_kv_from_start` layers).
    /// Each one's `host_rows_real` says how many leading rows of its
    /// host cache are real; the rest are placeholders for positions only
    /// the device holds (#630's bookkeeping, the same field the generic
    /// decoder reads).
    kvs: Vec<MetalKvBuffers>,
}

/// A batched launch for `m`: the quantized GEMM, or f32 rows (which
/// the GEMM takes 32 columns at a time; any other width stays on the
/// host).
fn prefill_mat(m: &WeightMatrix) -> Option<PrefillMatMetal<'_>> {
    match m {
        WeightMatrix::F32(t) => {
            let (rows, cols) = (t.shape[0], t.shape[1]);
            frink_metal::gpu::f32_mul_mm_sg_launch(&t.data, rows, cols)?;
            Some(PrefillMatMetal::F32 {
                weights: &t.data,
                rows,
                cols,
            })
        }
        _ => m.mul_mm_sg_launch().map(PrefillMatMetal::Quant),
    }
}

impl Gemma4Engine {
    fn device_aux(&self) -> &DeviceAux {
        self.device_aux.get_or_init(|| {
            let hp = &self.hp;
            let (q_norm, ones) = self
                .weights
                .layers
                .iter()
                .enumerate()
                .map(|(il, l)| {
                    let hd = hp.head_dim(il);
                    // The host body's `compensate`, from the same
                    // expression.
                    let compensate = hp.attention_scale * (hd as f32).sqrt();
                    let q = l.attn.q_norm.iter().map(|w| w * compensate).collect();
                    (q, vec![1.0f32; hd])
                })
                .unzip();
            DeviceAux { q_norm, ones }
        })
    }

    /// Whether Metal serves Gemma-4 tokens in this process: the same
    /// switch every dense model's fused stacks read.
    fn metal_switched_on() -> bool {
        crate::decoder::metal_attn_switched_on()
    }

    /// The run's RoPE, or `None` when the stacks cannot rotate this
    /// file as llama.cpp does. `MetalRope::rot_dim` is one width for
    /// every layer, while llama.cpp rotates `n_rot(il)` channels per
    /// layer (`gemma4.cpp`, `n_rot_l`): whole heads everywhere is
    /// `None` (each layer its own head), one partial width shared by
    /// every layer is that width, and anything else (a partial full
    /// width beside a different sliding one) stays on the host body
    /// rather than rotating some layer at the wrong width.
    fn metal_rope(&self) -> Option<MetalRope> {
        let hp = &self.hp;
        let base = MetalRope::new(MetalRopeLayout::Neox);
        let layers = 0..hp.n_layer;
        if layers.clone().all(|il| hp.rot_dim(il) == hp.head_dim(il)) {
            return Some(base);
        }
        let rot = hp.rot_dim(0);
        layers
            .clone()
            .all(|il| hp.rot_dim(il) == rot)
            .then_some(MetalRope {
                rot_dim: Some(rot),
                ..base
            })
    }

    /// The divisors are the first `n_rot/2` of `rope_freqs`, which is
    /// what ggml reads and the width `MetalRope::rot_dim` asks for.
    fn layer_rope(&self, il: usize) -> LayerRope<'_> {
        LayerRope {
            theta: self.hp.rope_theta_for(il),
            freq_factors: if self.hp.is_swa_layer(il) {
                None
            } else {
                let half = self.hp.rot_dim(il) / 2;
                self.weights.rope_freqs.as_deref().map(|f| &f[..half])
            },
        }
    }

    /// Q norm (scaled), and for a layer that owns its KV, K norm and V's
    /// ones.
    fn layer_extras(&self, il: usize) -> Option<AttnExtras<'_>> {
        let aux = self.device_aux();
        let attn = &self.weights.layers[il].attn;
        let own = self.hp.has_kv(il);
        Some(AttnExtras {
            q_norm: Some(&aux.q_norm[il]),
            k_norm: if own {
                Some(attn.k_norm.as_deref()?)
            } else {
                None
            },
            v_norm: own.then(|| aux.ones[il].as_slice()),
            ..AttnExtras::default()
        })
    }

    /// The KV wiring of layer `il`, built from `own` launches.
    fn layer_kv<L>(&self, il: usize, own: impl FnOnce() -> Option<(L, L)>) -> Option<LayerKv<L>> {
        if self.hp.has_kv(il) {
            let (k, v) = own()?;
            Some(LayerKv::Own { k, v, slot: il })
        } else {
            Some(LayerKv::Shared {
                slot: self.hp.kv_reuse_layer(il),
            })
        }
    }

    /// Every layer as the decode stack takes it, or `None` when some
    /// projection has no Metal matvec (the host body then serves).
    fn decode_layers(&self) -> Option<Vec<DenseLayerMetal<'_>>> {
        let mv = crate::metal_launch::matvec;
        let mut out = Vec::with_capacity(self.hp.n_layer);
        for (il, l) in self.weights.layers.iter().enumerate() {
            let a = &l.attn;
            // Without `attn_v` llama.cpp uses the raw K as V, and so
            // does the host body; the stacks have no arm for it, so
            // neither claims such a file.
            let kv = self.layer_kv(il, || {
                Some((mv(a.k_proj.as_ref()?)?, mv(a.v_proj.as_ref()?)?))
            })?;
            out.push(DenseLayerMetal {
                attn_norm_w: &l.attn_norm,
                ffn_norm_w: &l.ffn_norm,
                mixer: TokenMixer::Attention(AttnMixer {
                    q: mv(&a.q_proj)?,
                    kv,
                    o: mv(&a.o_proj)?,
                    extras: self.layer_extras(il)?,
                    rope: Some(self.layer_rope(il)),
                    window: self.hp.is_swa_layer(il).then_some(self.hp.sliding_window),
                }),
                gate: mv(&l.ffn_gate)?,
                up: mv(&l.ffn_up)?,
                down: mv(&l.ffn_down)?,
                post_attn_norm: Some(&a.post_attn_norm),
                post_ffn_norm: Some(&l.ffn_post_norm),
                per_layer: match (
                    &l.per_layer_inp_gate,
                    &l.per_layer_proj,
                    &l.per_layer_post_norm,
                ) {
                    (Some(g), Some(p), Some(n)) => Some(PerLayerGateMetal {
                        gate: mv(g)?,
                        proj: mv(p)?,
                        post_norm: n,
                        slot: il,
                    }),
                    _ => None,
                },
                out_scale: l.out_scale,
            });
        }
        Some(out)
    }

    /// Every layer as the prefill stack takes it.
    fn prefill_layers(&self) -> Option<Vec<PrefillDenseLayerMetal<'_>>> {
        fn mm(m: &WeightMatrix) -> Option<frink_metal::gpu::MulMmSgLaunch<'_>> {
            m.mul_mm_sg_launch()
        }
        let mut out = Vec::with_capacity(self.hp.n_layer);
        for (il, l) in self.weights.layers.iter().enumerate() {
            let a = &l.attn;
            let kv = self.layer_kv(il, || {
                Some((mm(a.k_proj.as_ref()?)?, mm(a.v_proj.as_ref()?)?))
            })?;
            out.push(PrefillDenseLayerMetal {
                attn_norm_w: &l.attn_norm,
                ffn_norm_w: &l.ffn_norm,
                q: mm(&a.q_proj)?,
                kv,
                o: mm(&a.o_proj)?,
                ffn: PrefillFfnMetal::Dense {
                    gate: mm(&l.ffn_gate)?,
                    up: mm(&l.ffn_up)?,
                    down: mm(&l.ffn_down)?,
                },
                post_attn_norm: Some(&a.post_attn_norm),
                post_ffn_norm: Some(&l.ffn_post_norm),
                extras: self.layer_extras(il)?,
                rope: Some(self.layer_rope(il)),
                layer_idx: il as u32,
                window: self.hp.is_swa_layer(il).then_some(self.hp.sliding_window),
                per_layer: match (
                    &l.per_layer_inp_gate,
                    &l.per_layer_proj,
                    &l.per_layer_post_norm,
                ) {
                    (Some(g), Some(p), Some(n)) => Some(PerLayerGateMetal {
                        gate: prefill_mat(g)?,
                        proj: prefill_mat(p)?,
                        post_norm: n,
                        slot: il,
                    }),
                    _ => None,
                },
                out_scale: l.out_scale,
            });
        }
        Some(out)
    }

    /// The PLE prologue's description, `None` for a file without
    /// per-layer embeddings (`Some(None)`) or when the projection has no
    /// launch (`None`).
    fn ple_input<'a, L>(
        &'a self,
        model_proj: impl FnOnce(&'a WeightMatrix) -> Option<L>,
        tok_rows: &'a [f32],
    ) -> Option<Option<PleInputMetal<'a, L>>> {
        let w = &self.weights;
        let (Some(proj), Some(norm)) = (&w.per_layer_model_proj, &w.per_layer_proj_norm) else {
            return Some(None);
        };
        let scales = self.ple_scales();
        Some(Some(PleInputMetal {
            model_proj: model_proj(proj)?,
            proj_norm_w: norm,
            tok_rows,
            n_embd: self.hp.embd_per_layer,
            n_layer: self.hp.n_layer,
            proj_scale: scales.proj,
            input_scale: scales.input,
        }))
    }

    /// Does this engine run on the Metal stacks at all? For the kernel
    /// registry and the tests; the forward paths ask the same questions
    /// each call and fall back per call.
    pub(crate) fn metal_stacks_cover(&self) -> bool {
        let tok: [f32; 0] = [];
        self.metal_rope().is_some()
            && self.decode_layers().is_some()
            && self.prefill_layers().is_some()
            && crate::metal_launch::matvec(&self.weights.output_head).is_some()
            && self.ple_input(crate::metal_launch::matvec, &tok).is_some()
            && self.ple_input(prefill_mat, &tok).is_some()
    }

    /// Make the device KV hold exactly positions `[0, pos)` with room for
    /// `n` more, building it from the host rows the first time. `None`
    /// when it cannot, after which the host body serves.
    fn ensure_device_kv<'s>(
        &self,
        state: &'s mut Gemma4DecodeState,
        pos: usize,
        n: usize,
    ) -> Option<&'s mut DeviceKv> {
        let hp = &self.hp;
        let n_own = hp.n_layer_kv_from_start;
        let owners = || state.kv[..n_own].iter().map(|c| c.as_ref());
        if let Some(dev) = state.metal.kv.as_mut() {
            let seq = dev.kvs[0].seq_len;
            if pos > seq {
                // Positions nobody computed: not ours to invent.
                Self::drop_device_kv(state);
                return None;
            }
            if pos < seq {
                // A rewind (a reused state, a rolled-back draft): the
                // device rows past `pos` are simply no longer read.
                for m in dev.kvs.iter_mut() {
                    m.seq_len = pos;
                    m.host_rows_real = m.host_rows_real.min(pos);
                }
                for c in state.kv[..n_own].iter_mut().flatten() {
                    c.truncate(pos);
                }
            }
            if pos + n > dev.kvs[0].capacity() {
                let cap = (pos + n).next_power_of_two();
                for (il, m) in dev.kvs.iter_mut().enumerate() {
                    let mut grown =
                        MetalKvBuffers::with_capacity(hp.n_kv_heads, hp.head_dim(il), cap).ok()?;
                    if m.seq_len > 0 {
                        let (k, v) = m.tokens_host(0, m.seq_len);
                        grown.upload_from_host(&k, &v, m.seq_len).ok()?;
                    }
                    // A device-to-device copy: the host is no more real
                    // than it was.
                    grown.host_rows_real = m.host_rows_real;
                    *m = grown;
                }
            }
            return state.metal.kv.as_mut();
        }
        // First use: the host rows are the record.
        if owners().any(|c| c.is_none_or(|c| c.positions() != pos || c.rows() != pos)) {
            return None;
        }
        let cap = (pos + n).next_power_of_two().max(4096);
        let mut kvs = Vec::with_capacity(n_own);
        for (il, c) in owners().enumerate() {
            let c = c.expect("checked above");
            let mut m = MetalKvBuffers::with_capacity(hp.n_kv_heads, hp.head_dim(il), cap).ok()?;
            if pos > 0 {
                m.upload_from_host(&c.k.to_f32_vec(), &c.v.to_f32_vec(), pos)
                    .ok()?;
            }
            kvs.push(m);
        }
        state.metal.kv = Some(DeviceKv { kvs });
        state.metal.kv.as_mut()
    }

    /// Hand the KV back to the host: replace the placeholder rows with
    /// the device's, and forget the device copy.
    fn drop_device_kv(state: &mut Gemma4DecodeState) {
        let Some(dev) = state.metal.kv.take() else {
            return;
        };
        for (m, c) in dev.kvs.iter().zip(state.kv.iter_mut()) {
            let Some(c) = c.as_mut() else { continue };
            let real = m.host_rows_real.min(m.seq_len);
            c.truncate(real);
            let n = m.seq_len - real;
            if n == 0 {
                continue;
            }
            let (k, v) = m.tokens_host(real, n);
            let per = c.n_kv_heads * c.head_dim;
            for i in 0..n {
                let off = i * per;
                c.push(&k[off..off + per], &v[off..off + per])
                    .expect("unbounded KvCache growth is infallible");
            }
        }
    }

    /// Advance every owning host cache by `n` placeholder rows, so its
    /// position count matches the device's.
    fn advance_host(&self, state: &mut Gemma4DecodeState, n: usize) {
        for c in state.kv[..self.hp.n_layer_kv_from_start]
            .iter_mut()
            .flatten()
        {
            c.advance_len(n)
                .expect("unbounded KvCache growth is infallible");
        }
    }

    /// One token through the decode stack: every layer, the final norm
    /// and the LM head in one command buffer. `None` = the host body
    /// serves this token (and every later one, until a state is new).
    pub(crate) fn forward_token_metal(
        &self,
        token_id: usize,
        pos: usize,
        state: &mut Gemma4DecodeState,
    ) -> Option<Vec<f32>> {
        if !Self::metal_switched_on() {
            if state.metal.kv.is_some() {
                Self::drop_device_kv(state);
            }
            return None;
        }
        self.decode_on_metal(token_id, pos, state)
    }

    /// [`Self::forward_token_metal`] without the process switch, so a
    /// test can hold the host body and this one side by side.
    pub(crate) fn decode_on_metal(
        &self,
        token_id: usize,
        pos: usize,
        state: &mut Gemma4DecodeState,
    ) -> Option<Vec<f32>> {
        let rope = self.metal_rope()?;
        let layers = self.decode_layers()?;
        let output = crate::metal_launch::matvec(&self.weights.output_head)?;
        let tok_rows = self.per_layer_token_row(token_id).unwrap_or_default();
        let ple = self.ple_input(crate::metal_launch::matvec, &tok_rows)?;
        let hidden = self.embed_scaled(token_id);
        let argmax_only = frink_metal::greedy_fold::metal_greedy_argmax_active();
        let dev = self.ensure_device_kv(state, pos, 1)?;
        let result = frink_metal::attn::launch_decode_dense_stack(
            &hidden,
            &layers,
            &mut dev.kvs,
            self.hp.n_heads,
            rope,
            pos,
            self.hp.rms_norm_eps,
            Some(&self.weights.output_norm),
            Some(&output),
            argmax_only,
            None,
            true,
            ple.as_ref(),
        );
        match result {
            Ok(mut out) => {
                self.advance_host(state, 1);
                // A one-element answer is the argmax id the generation
                // loop asked for; the softcap does not move an argmax.
                if !(argmax_only && out.len() == 1) {
                    if let Some(cap) = self.hp.final_logit_softcap {
                        softcap_inplace(&mut out, cap);
                    }
                }
                Some(out)
            }
            Err(e) => {
                eprintln!("frink: Gemma-4 Metal decode stack failed, host fallback: {e}");
                Self::drop_device_kv(state);
                None
            }
        }
    }

    /// The batched part of a prompt: as many leading tokens as the
    /// prefill stack can take, in chunks of one command buffer each.
    /// Returns how many tokens it consumed and, when that was all of
    /// them, the last one's logits.
    pub(crate) fn prefill_metal(
        &self,
        tokens: &[usize],
        start_pos: usize,
        state: &mut Gemma4DecodeState,
    ) -> (usize, Vec<f32>) {
        if !Self::metal_switched_on() {
            return (0, Vec::new());
        }
        self.prefill_on_metal(tokens, start_pos, state)
    }

    /// [`Self::prefill_metal`] without the process switch.
    pub(crate) fn prefill_on_metal(
        &self,
        tokens: &[usize],
        start_pos: usize,
        state: &mut Gemma4DecodeState,
    ) -> (usize, Vec<f32>) {
        if tokens.len() < PREFILL_MIN {
            return (0, Vec::new());
        }
        let Some(layers) = self.prefill_layers() else {
            return (0, Vec::new());
        };
        let hp = &self.hp;
        let mut done = 0usize;
        let mut last_hidden: Option<Vec<f32>> = None;
        while done < tokens.len() {
            let pos = start_pos + done;
            let n = (tokens.len() - done).min(PREFILL_CHUNK);
            if n < PREFILL_MIN {
                break;
            }
            match self.prefill_chunk(&layers, &tokens[done..done + n], pos, state) {
                Some(h) => {
                    done += n;
                    last_hidden = Some(h);
                }
                None => break,
            }
        }
        if done < tokens.len() {
            return (done, Vec::new());
        }
        let Some(h) = last_hidden else {
            return (0, Vec::new());
        };
        let normed = rms_norm(&h, &self.weights.output_norm, hp.rms_norm_eps);
        let mut logits = self.weights.output_head.apply(&normed);
        if let Some(cap) = hp.final_logit_softcap {
            softcap_inplace(&mut logits, cap);
        }
        (done, logits)
    }

    /// One prefill command buffer; the last token's hidden state.
    fn prefill_chunk(
        &self,
        layers: &[PrefillDenseLayerMetal<'_>],
        chunk: &[usize],
        pos: usize,
        state: &mut Gemma4DecodeState,
    ) -> Option<Vec<f32>> {
        let hp = &self.hp;
        let d = hp.hidden_dim;
        let mut hidden = Vec::with_capacity(chunk.len() * d);
        let mut tok_rows = Vec::new();
        for &t in chunk {
            hidden.extend_from_slice(&self.embed_scaled(t));
            if let Some(row) = self.per_layer_token_row(t) {
                tok_rows.extend_from_slice(&row);
            }
        }
        let ple = self.ple_input(prefill_mat, &tok_rows)?;
        let rope = self.metal_rope()?;
        let dev = self.ensure_device_kv(state, pos, chunk.len())?;
        let result = frink_metal::attn::launch_prefill_dense_stack(
            &hidden,
            layers,
            &mut dev.kvs,
            hp.n_heads,
            chunk.len(),
            rope,
            pos,
            hp.rms_norm_eps,
            true,
            None,
            ple.as_ref(),
        );
        match result {
            Ok(h) => {
                self.advance_host(state, chunk.len());
                Some(h[(chunk.len() - 1) * d..].to_vec())
            }
            Err(e) => {
                eprintln!("frink: Gemma-4 Metal prefill stack failed, decode fallback: {e}");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests;
