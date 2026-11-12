//! The Vulkan decode stack: a run of dense layers for one token in ONE
//! submission (#409, step 5).
//!
//! The per-op path sends Q/K/V, `wo` and the FFN to the device and runs
//! the norms, RoPE, attention and the residual adds on the host between
//! them: about three queue round trips per layer. Here every op of a
//! run of layers is recorded into one `frink_vulkan` batch (norm, Q/K/V,
//! RoPE, the f16 KV append, attention, `wo`, residual, norm, gate and
//! up, SwiGLU, down, residual) and the token waits once. The hidden
//! state and each layer's rotated K and V rows come back after that wait.
//!
//! The host cache stays the authority, as on CUDA (`cuda_prefill`): each
//! layer's rows are pushed into its `KvCache` after the run, and the
//! device KV is a mirror (`frink_core::vulkan_kv`) that the next token
//! checks against the host rows and rebuilds when they disagree. So the
//! prefill, which runs on the host, and every host fallback see a
//! complete cache.
//!
//! Which layers it may take is `fused_view`'s decision, shared with the
//! Metal and CUDA stacks. The stack declines what its ops do not
//! compute, so a layer it cannot serve runs on the host body unchanged:
//! QKV biases, QK-norms, post-norms, sinks, `wo` bias, a sliding window,
//! an attention softcap, RoPE `mscale`, routed experts, a GeGLU FFN, and
//! a host cache that is not f16 (`--ctk f32`).
//!
//! One submission holds [`MAX_RUN`] layers, so a deeper model waits once
//! per that many.

use frink_core::cache::KvCache;
use frink_core::vulkan_kv::VulkanKvMirror;
use frink_core::WeightMatrix;
use frink_vulkan::runtime::ops::{AttnShape, Batch, DeviceBuf};
use frink_vulkan::runtime::{MatvecJob, Runtime, MAX_DISPATCHES};

use super::fused_view::{FusedAttnCaps, FusedAttnExtras};
use super::{Decoder, ExpertBacking, LayerWeights};

/// Dispatches one layer records in `run_vulkan_stack`: norm, Q, K, V,
/// two RoPEs, two KV appends, attention (two), `wo`, add, norm, gate,
/// up, SwiGLU, down, add.
const OPS_PER_LAYER: usize = 19;

/// Layers one submission holds.
const MAX_RUN: usize = MAX_DISPATCHES / OPS_PER_LAYER;

/// One layer as the stack runs it: borrowed weights and its RoPE.
struct StackLayer<'a> {
    attn_norm: &'a [f32],
    ffn_norm: &'a [f32],
    q: MatvecJob<'a>,
    k: MatvecJob<'a>,
    v: MatvecJob<'a>,
    o: MatvecJob<'a>,
    gate: MatvecJob<'a>,
    up: MatvecJob<'a>,
    down: MatvecJob<'a>,
    /// `(cos, sin)` pairs for this token, and the rotated width; `None`
    /// for a layer llama.cpp does not rotate.
    rope: Option<(Vec<f32>, usize)>,
}

/// The run's new hidden state, and each layer's rotated K row and V row
/// for the host cache.
type StackOut = (Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>);

fn job(m: &WeightMatrix) -> Option<MatvecJob<'_>> {
    m.vulkan_stack_job()
}

impl Decoder {
    /// Layer `li` as the stack takes it at position `pos`, or `None` for
    /// a layer or a cache state it does not serve.
    fn vulkan_stack_layer<'a>(
        &'a self,
        li: usize,
        layer: &'a LayerWeights,
        cache: &KvCache,
        pos: usize,
    ) -> Option<StackLayer<'a>> {
        let c = &self.config;
        if !Self::fused_prefill_dense_layer_eligible(layer, c, self.lora_attached())
            || !self.layer_supports_fused_attn(layer, FusedAttnCaps::VULKAN)
        {
            return None;
        }
        // Everything the extras carry is arithmetic the stack's ops do
        // not do yet; `None` in each is the plain llama layer.
        let FusedAttnExtras {
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
            sinks,
            o_bias,
        } = Self::fused_attn_extras(layer, FusedAttnCaps::VULKAN)?;
        if q_bias.is_some()
            || k_bias.is_some()
            || v_bias.is_some()
            || q_norm.is_some()
            || k_norm.is_some()
            || sinks.is_some()
            || o_bias.is_some()
            || layer.attn.post_attn_norm.is_some()
            || layer.attn.post_ffn_norm.is_some()
        {
            return None;
        }
        // SwiGLU is the one activation the stack's FFN has.
        if c.model_ffn_act()
            .and_then(frink_moe::GluAct::fused_kernel_gelu_flag)
            != Some(false)
        {
            return None;
        }
        if c.attn_logit_softcap.is_some()
            || c.layer_sliding_window(li).is_some()
            || c.rope_attn_factor != 1.0
            || self.hrm_z_l_init.is_some()
            || !c.head_dim.is_multiple_of(2)
        {
            return None;
        }
        // The prefix the device attends over is the cache's rows, and
        // they are its positions only while nothing has been evicted.
        // The device KV is f16, as llama.cpp's default is. A run that
        // asked for `--ctk f32` keeps f32 rows on the host, and attending
        // over their f16 rounding would not be what it asked for.
        if cache.dtype() != frink_core::kv_store::KvDtype::F16
            || cache.positions() != pos
            || cache.rows() != pos
            || cache.n_kv_heads != c.n_kv_heads
            || cache.head_dim != c.head_dim
            || cache.v_head_dim != c.head_dim
        {
            return None;
        }
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        let ex = experts.first()?;
        let rope = c.layer_rope(li).map(|r| {
            let rot = r.rot_dim.unwrap_or(c.head_dim);
            (
                frink_core::rope_table::rope_cos_sin(pos, r.theta, rot, r.freq_factors),
                rot,
            )
        });
        Some(StackLayer {
            attn_norm: layer.attn.norm_weight.rms_weights()?,
            ffn_norm: layer.moe.norm_weight.rms_weights()?,
            q: job(&layer.attn.q_proj)?,
            k: job(&layer.attn.k_proj)?,
            v: job(&layer.attn.v_proj)?,
            o: job(&layer.attn.o_proj)?,
            gate: job(&ex.gate)?,
            up: job(&ex.up)?,
            down: job(&ex.down)?,
            rope,
        })
    }

    /// Runs the longest run of layers the stack serves, starting at `l`,
    /// for the token at `pos` whose hidden state is `hidden`. Pushes each
    /// layer's K/V into its cache and answers the new hidden state with
    /// the run length. `None` -- before touching any cache -- when layer
    /// `l` is not one it serves, so the host body runs it; a device error
    /// after admission is reported once and also answers `None`, with
    /// every host cache untouched.
    pub(crate) fn try_vulkan_dense_stack(
        &self,
        l: usize,
        hidden: &[f32],
        pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<(Vec<f32>, usize)> {
        if !frink_core::weight_matrix::vulkan_dense_enabled() || hidden.is_empty() {
            return None;
        }
        let mut layers = Vec::new();
        for (li, cache) in kv_caches.iter().enumerate().skip(l).take(MAX_RUN) {
            match self.vulkan_stack_layer(li, self.layer_for(li), cache, pos) {
                Some(s) => layers.push(s),
                None => break,
            }
        }
        if layers.is_empty() {
            return None;
        }
        let run = layers.len();
        let rt = frink_core::weight_matrix::gpu_backend::vulkan_runtime()?;
        let mut rt = rt.lock().ok()?;
        let out = match self.run_vulkan_stack(&mut rt, &layers, l, hidden, pos, kv_caches) {
            Ok(out) => out,
            Err(e) => {
                warn_once(&format!(
                    "frink: Vulkan decode stack declined, host body runs it: {e}"
                ));
                return None;
            }
        };
        drop(rt);
        let (new_hidden, rows) = out;
        for (i, (k_row, v_row)) in rows.iter().enumerate() {
            let li = l + i;
            let cache = &mut kv_caches[li];
            cache
                .push(k_row, v_row)
                .expect("unbounded/planned KvCache growth is infallible");
            if let Some(m) = cache.vulkan_attn.as_mut() {
                m.record_push(k_row);
            }
            // The residency counts the host FFN records for a dense layer.
            self.layer_for(li).moe.record_activations(&[0]);
            self.evict_layer_kv(li, cache);
        }
        Some((new_hidden, run))
    }

    /// The device half: sync the mirrors, record, submit, read back.
    fn run_vulkan_stack(
        &self,
        rt: &mut Runtime,
        layers: &[StackLayer<'_>],
        l: usize,
        hidden: &[f32],
        pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Result<StackOut, frink_vulkan::device::VulkanError> {
        let c = &self.config;
        let (n_heads, n_kv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (dim, q_dim, kv_dim) = (c.hidden_dim, n_heads * hd, n_kv * hd);
        let ffn_dim = layers[0].gate.rows;
        let seq = pos + 1;

        // Activations, shared by every layer of the run: the residual
        // stream ping-pongs between `h` and `h_mid`.
        let h = rt.scratch("h", 0, dim)?;
        let h_mid = rt.scratch("h_mid", 0, dim)?;
        let normed = rt.scratch("normed", 0, dim)?;
        let q = rt.scratch("q", 0, q_dim)?;
        let qr = rt.scratch("qr", 0, q_dim)?;
        let attn = rt.scratch("attn", 0, q_dim)?;
        let proj = rt.scratch("proj", 0, dim)?;
        let gate = rt.scratch("gate", 0, ffn_dim)?;
        let up = rt.scratch("up", 0, ffn_dim)?;
        let act = rt.scratch("act", 0, ffn_dim)?;
        let scores = rt.scratch("scores", 0, n_heads * seq)?;
        rt.write_f32(h, 0, hidden)?;

        // Per layer: the norms' weights, its RoPE table, its K before
        // and after rotation and its V (read back for the host cache),
        // and its mirror.
        struct Bound {
            attn_norm: DeviceBuf,
            ffn_norm: DeviceBuf,
            tab: Option<(DeviceBuf, usize)>,
            k: DeviceBuf,
            kr: DeviceBuf,
            v: DeviceBuf,
            cache: DeviceBuf,
            cache_words: usize,
            k_word: usize,
            v_word: usize,
            v_off: usize,
        }
        let mut bound = Vec::with_capacity(layers.len());
        for (i, s) in layers.iter().enumerate() {
            let li = l + i;
            let attn_norm = rt.constant_f32(s.attn_norm)?;
            let ffn_norm = rt.constant_f32(s.ffn_norm)?;
            let tab = match &s.rope {
                Some((t, rot)) => {
                    let b = rt.scratch("rope", i, t.len())?;
                    rt.write_f32(b, 0, t)?;
                    Some((b, *rot))
                }
                None => None,
            };
            let k = rt.scratch("k", i, kv_dim)?;
            let kr = rt.scratch("kr", i, kv_dim)?;
            let v = rt.scratch("v", i, kv_dim)?;
            let cache = &mut kv_caches[li];
            let rows = cache.rows();
            let KvCache {
                k: hk,
                v: hv,
                vulkan_attn,
                ..
            } = cache;
            let m = VulkanKvMirror::sync(vulkan_attn, rt, hk, hv, rows, kv_dim, seq)?;
            bound.push(Bound {
                attn_norm,
                ffn_norm,
                tab,
                k,
                kr,
                v,
                cache: m.buf(),
                cache_words: m.words(),
                k_word: m.k_word(pos),
                v_word: m.v_word(pos),
                v_off: m.v_off(),
            });
        }

        let eps = c.rms_norm_eps;
        let mut batch = Batch::new();
        for (s, b) in layers.iter().zip(&bound) {
            // --- attention ---
            batch.rms_norm(b.attn_norm, h, normed, dim, 1, eps);
            batch.matvec(s.q, (normed, 0), (q, 0));
            batch.matvec(s.k, (normed, 0), (b.k, 0));
            batch.matvec(s.v, (normed, 0), (b.v, 0));
            let (q_in, k_in) = match b.tab {
                Some((tab, rot)) => {
                    batch.rope(tab, q, qr, n_heads, hd, rot, self.rope_is_neox());
                    batch.rope(tab, b.k, b.kr, n_kv, hd, rot, self.rope_is_neox());
                    (qr, b.kr)
                }
                // An unrotated layer's K is its rotated K.
                None => (q, b.k),
            };
            batch.kv_store(k_in, b.cache, kv_dim, b.k_word, b.cache_words);
            batch.kv_store(b.v, b.cache, kv_dim, b.v_word, b.cache_words);
            batch.attention(
                b.cache,
                b.cache_words,
                q_in,
                scores,
                attn,
                AttnShape {
                    n_heads,
                    n_kv_heads: n_kv,
                    head_dim: hd,
                    seq_len: seq,
                    k_off: 0,
                    v_off: b.v_off,
                    scale: 1.0 / (hd as f32).sqrt(),
                },
            );
            batch.matvec(s.o, (attn, 0), (proj, 0));
            batch.add(h, proj, h_mid, dim);
            // --- FFN ---
            batch.rms_norm(b.ffn_norm, h_mid, normed, dim, 1, eps);
            batch.matvec(s.gate, (normed, 0), (gate, 0));
            batch.matvec(s.up, (normed, 0), (up, 0));
            batch.swiglu(gate, up, act, ffn_dim);
            batch.matvec(s.down, (act, 0), (proj, 0));
            batch.add(h_mid, proj, h, dim);
        }
        assert!(batch.len() <= OPS_PER_LAYER * layers.len());
        rt.submit_batch(&batch)?;

        let new_hidden = rt.read_f32(h, 0, dim)?;
        let mut rows = Vec::with_capacity(layers.len());
        for (s, b) in layers.iter().zip(&bound) {
            let k_src = if s.rope.is_some() { b.kr } else { b.k };
            rows.push((rt.read_f32(k_src, 0, kv_dim)?, rt.read_f32(b.v, 0, kv_dim)?));
        }
        Ok((new_hidden, rows))
    }

    fn rope_is_neox(&self) -> bool {
        matches!(self.config.rope_layout, crate::config::RopeLayout::Neox)
    }
}

fn warn_once(msg: &str) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| eprintln!("{msg}"));
}
