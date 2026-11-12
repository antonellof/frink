//! The Vulkan stack: a run of dense layers in ONE submission, for one
//! decode token or for a prefill batch (#409, steps 5 and 6).
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
//! sinks, `wo` bias, a sliding window,
//! an attention softcap, RoPE `mscale`, routed experts, a GeGLU FFN, and
//! a host cache that is not f16 (`--ctk f32`).
//!
//! One submission holds 26 plain layers (`MAX_DISPATCHES`, at 19
//! dispatches each, a few fewer with a bias or a QK-norm), so a deeper
//! model waits once per that many.
//!
//! A prefill batch is the same recording over `batch` positions
//! (`Batch::for_positions`): every buffer holds a row per position, the
//! batch's K and V are stored before its attention, and position `b`
//! attends over the prefix and the batch up to itself. A batch whose
//! score scratch would pass [`MAX_SCORE_BYTES`] is the host's.

use frink_core::cache::KvCache;
use frink_core::vulkan_kv::VulkanKvMirror;
use frink_core::WeightMatrix;
use frink_vulkan::runtime::ops::{AttnShape, Batch, DeviceBuf, NormKind};
use frink_vulkan::runtime::{MatvecJob, Runtime, MAX_DISPATCHES};

use super::fused_view::{FusedAttnCaps, FusedAttnExtras};
use super::{Decoder, ExpertBacking, LayerWeights};

/// Dispatches a plain layer records in `run_vulkan_stack`: norm, Q, K,
/// V, two RoPEs, two KV appends, attention (two), `wo`, add, norm, gate,
/// up, SwiGLU, down, add. A QKV bias, a QK-norm or a post-norm is one
/// more each, and a layer with no pre-norm one fewer
/// ([`StackLayer::ops`]).
const OPS_PER_LAYER: usize = 19;

/// The most score scratch one prefill batch may ask for: `batch *
/// n_heads * (start_pos + batch)` f32. Above it the host body runs the
/// batch, which is a refusal by size and not a different answer.
const MAX_SCORE_BYTES: usize = 256 << 20;

/// One layer as the stack runs it: borrowed weights and its RoPE.
struct StackLayer<'a> {
    /// The pre-norms; `None` for a layer with no norm before the block
    /// (the post-norm architectures).
    attn_norm: Option<PreNorm<'a>>,
    ffn_norm: Option<PreNorm<'a>>,
    /// RMS weights applied to the block's output before the residual
    /// add, at `post_norm_eps`.
    post_attn_norm: Option<&'a [f32]>,
    post_ffn_norm: Option<&'a [f32]>,
    q: MatvecJob<'a>,
    k: MatvecJob<'a>,
    v: MatvecJob<'a>,
    o: MatvecJob<'a>,
    gate: MatvecJob<'a>,
    up: MatvecJob<'a>,
    down: MatvecJob<'a>,
    /// `(cos, sin)` pairs for each position of the batch, one table
    /// after another, and the rotated width; `None` for a layer
    /// llama.cpp does not rotate.
    rope: Option<(Vec<f32>, usize)>,
    /// Added to the projection for every position (Qwen2's QKV bias).
    q_bias: Option<&'a [f32]>,
    k_bias: Option<&'a [f32]>,
    v_bias: Option<&'a [f32]>,
    /// QK-norm weights, applied after the bias and before RoPE: one
    /// `head_dim` weight shared by every head, or one over the whole
    /// projection (`layer_supports_fused_attn` admits only those two).
    q_norm: Option<&'a [f32]>,
    k_norm: Option<&'a [f32]>,
}

/// A pre-norm's kind, weight and bias once on the device.
type BoundNorm = (NormKind, Option<DeviceBuf>, Option<DeviceBuf>);

/// A pre-norm as the device runs it: the norm op, times a weight when
/// there is one, then a bias when there is one (`NormOp::apply`'s order).
#[derive(Clone, Copy)]
struct PreNorm<'a> {
    kind: NormKind,
    weight: Option<&'a [f32]>,
    bias: Option<&'a [f32]>,
}

impl<'a> PreNorm<'a> {
    /// `None` for [`NormOp::None`]: the block reads the residual stream.
    /// Exhaustive, so a new norm has to say how the device runs it.
    fn of(n: &'a crate::norm::NormOp) -> Option<Self> {
        use crate::norm::NormOp;
        let (kind, weight, bias) = match n {
            NormOp::None => return None,
            NormOp::Rms(w) => (NormKind::Rms, Some(&w[..]), None),
            NormOp::RmsNoParams => (NormKind::Rms, None, None),
            NormOp::RmsBias { weight, bias } => (NormKind::Rms, Some(&weight[..]), Some(&bias[..])),
            NormOp::LayerNormNoParams => (NormKind::Layer, None, None),
            NormOp::LayerNorm(w) => (NormKind::Layer, Some(&w[..]), None),
            NormOp::LayerNormBias { weight, bias } => {
                (NormKind::Layer, Some(&weight[..]), Some(&bias[..]))
            }
        };
        Some(Self { kind, weight, bias })
    }

    /// The dispatches it records: the norm, and the bias add.
    fn ops(self) -> usize {
        1 + self.bias.is_some() as usize
    }
}

impl StackLayer<'_> {
    /// The dispatches this layer records.
    fn ops(&self) -> usize {
        // The 19 count both pre-norms as one dispatch each.
        OPS_PER_LAYER - 2
            + [self.attn_norm, self.ffn_norm]
                .iter()
                .flatten()
                .map(|n| n.ops())
                .sum::<usize>()
            + [
                self.post_attn_norm,
                self.post_ffn_norm,
                self.q_bias,
                self.k_bias,
                self.v_bias,
                self.q_norm,
                self.k_norm,
            ]
            .iter()
            .flatten()
            .count()
    }
}

/// What the first layer of a run starts from.
#[derive(Clone, Copy)]
pub(crate) enum StackInput<'a> {
    /// The hidden rows, a row per position.
    Hidden(&'a [f32]),
    /// One decode token whose embedding row IS the first hidden state:
    /// the run's first op gathers it on the device
    /// ([`Decoder::vulkan_embedding_gather`]).
    Token(usize),
}

/// The run's new hidden rows, and each layer's rotated K rows and V
/// rows (a row per position) for the host cache.
type StackOut = (Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>);

fn job(m: &WeightMatrix) -> Option<MatvecJob<'_>> {
    m.vulkan_stack_job()
}

impl Decoder {
    /// The embedding table as the device gather reads it, or `None` when
    /// the first hidden state is not the bare table row and the host
    /// must embed: an embedding scale, an embedding norm, a learned
    /// position table, the skip stream, the HRM streams, or a table that
    /// is not plain bytes of a kind with a shader. The list
    /// `cuda_embedding_gather` keeps, for the same reason: each is
    /// something `embed_token` does to the row after `dequant_row`.
    pub(crate) fn vulkan_embedding_gather(&self) -> Option<MatvecJob<'_>> {
        if !frink_core::weight_matrix::vulkan_dense_enabled()
            || self.config.embedding_scale.is_some()
            || !matches!(self.embedding_norm, crate::norm::NormOp::None)
            || self.position_embd.is_some()
            || self.config.skip_stream
            || self.hrm_z_l_init.is_some()
        {
            return None;
        }
        job(&self.embedding)
    }

    /// Layer `li` as the stack takes it for `batch` positions from
    /// `pos`, or `None` for a layer or a cache state it does not serve.
    fn vulkan_stack_layer<'a>(
        &'a self,
        li: usize,
        layer: &'a LayerWeights,
        cache: &KvCache,
        pos: usize,
        batch: usize,
    ) -> Option<StackLayer<'a>> {
        let c = &self.config;
        if !Self::fused_prefill_dense_layer_eligible(layer, c, self.lora_attached())
            || !self.layer_supports_fused_attn(layer, FusedAttnCaps::VULKAN)
        {
            return None;
        }
        // Sinks and a bias on `wo` are arithmetic the stack's ops do
        // not do; the QKV biases and the QK-norms are.
        let FusedAttnExtras {
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
            sinks,
            o_bias,
        } = Self::fused_attn_extras(layer, FusedAttnCaps::VULKAN)?;
        if sinks.is_some() || o_bias.is_some() {
            return None;
        }
        let attn_norm = PreNorm::of(&layer.attn.norm_weight);
        let ffn_norm = PreNorm::of(&layer.moe.norm_weight);
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
            let tables = (pos..pos + batch)
                .flat_map(|p| frink_core::rope_table::rope_cos_sin(p, r.theta, rot, r.freq_factors))
                .collect();
            (tables, rot)
        });
        Some(StackLayer {
            attn_norm,
            ffn_norm,
            post_attn_norm: layer.attn.post_attn_norm.as_deref(),
            post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
            q: job(&layer.attn.q_proj)?,
            k: job(&layer.attn.k_proj)?,
            v: job(&layer.attn.v_proj)?,
            o: job(&layer.attn.o_proj)?,
            gate: job(&ex.gate)?,
            up: job(&ex.up)?,
            down: job(&ex.down)?,
            rope,
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
        })
    }

    /// Runs the longest run of layers the stack serves, starting at `l`,
    /// over `hidden_rows` (`batch` rows, positions `pos..pos + batch`).
    /// Pushes each layer's K/V rows into its cache and answers the new
    /// hidden rows with the run length. `None` -- before touching any
    /// cache -- when layer `l` is not one it serves, so the host body
    /// runs it; a device error after admission is reported once and also
    /// answers `None`, with every host cache untouched.
    pub(crate) fn try_vulkan_dense_stack(
        &self,
        l: usize,
        input: StackInput<'_>,
        pos: usize,
        batch: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<(Vec<f32>, usize)> {
        let fits = match input {
            StackInput::Hidden(rows) => rows.len() == batch * self.config.hidden_dim,
            // A gathered row is the first layer's input for one token.
            StackInput::Token(t) => {
                l == 0
                    && batch == 1
                    && self
                        .vulkan_embedding_gather()
                        .is_some_and(|table| t < table.rows)
            }
        };
        if !frink_core::weight_matrix::vulkan_dense_enabled()
            || batch == 0
            || !fits
            // `frink parity --dump-tensors` reads the host body's
            // intermediates.
            || crate::tensor_dump::active().is_some()
        {
            return None;
        }
        let scores = batch
            .checked_mul(self.config.n_heads)?
            .checked_mul(pos + batch)?
            .checked_mul(4)?;
        if scores > MAX_SCORE_BYTES {
            return None;
        }
        let mut layers = Vec::new();
        // One dispatch is kept for the gather.
        let mut budget = MAX_DISPATCHES - 1;
        for (li, cache) in kv_caches.iter().enumerate().skip(l) {
            match self.vulkan_stack_layer(li, self.layer_for(li), cache, pos, batch) {
                Some(s) if s.ops() <= budget => {
                    budget -= s.ops();
                    layers.push(s);
                }
                _ => break,
            }
        }
        if layers.is_empty() {
            return None;
        }
        let run = layers.len();
        let rt = frink_core::weight_matrix::gpu_backend::vulkan_runtime()?;
        let mut rt = rt.lock().ok()?;
        let out = match self.run_vulkan_stack(&mut rt, &layers, l, input, pos, batch, kv_caches) {
            Ok(out) => out,
            Err(e) => {
                warn_once(&format!(
                    "frink: Vulkan stack declined, host body runs it: {e}"
                ));
                return None;
            }
        };
        drop(rt);
        let (new_hidden, rows) = out;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;
        for (i, (k_rows, v_rows)) in rows.iter().enumerate() {
            let li = l + i;
            let cache = &mut kv_caches[li];
            for (k_row, v_row) in k_rows.chunks_exact(kv_dim).zip(v_rows.chunks_exact(kv_dim)) {
                // The residency counts the host FFN records for a dense
                // layer, a position at a time.
                self.layer_for(li).moe.record_activations(&[0]);
                cache
                    .push(k_row, v_row)
                    .expect("unbounded/planned KvCache growth is infallible");
                if let Some(m) = cache.vulkan_attn.as_mut() {
                    m.record_push(k_row);
                }
            }
            // After the pushes, as the host bodies do: a prefill batch
            // read its whole prefix.
            self.evict_layer_kv(li, cache);
        }
        Some((new_hidden, run))
    }

    /// The device half: sync the mirrors, record, submit, read back.
    #[allow(clippy::too_many_arguments)]
    fn run_vulkan_stack(
        &self,
        rt: &mut Runtime,
        layers: &[StackLayer<'_>],
        l: usize,
        input: StackInput<'_>,
        pos: usize,
        batch: usize,
        kv_caches: &mut [KvCache],
    ) -> Result<StackOut, frink_vulkan::device::VulkanError> {
        let c = &self.config;
        let (n_heads, n_kv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (dim, q_dim, kv_dim) = (c.hidden_dim, n_heads * hd, n_kv * hd);
        let ffn_dim = layers[0].gate.rows;
        // What the batch's first position attends over, and the rows the
        // mirror must hold once the batch is stored.
        let seq = pos + 1;
        let need = pos + batch;

        // Activations, shared by every layer of the run: the residual
        // stream ping-pongs between `h` and `h_mid`.
        let h = rt.scratch("h", 0, batch * dim)?;
        let h_mid = rt.scratch("h_mid", 0, batch * dim)?;
        let normed = rt.scratch("normed", 0, batch * dim)?;
        let normed_biased = rt.scratch("normed_biased", 0, batch * dim)?;
        let q = rt.scratch("q", 0, batch * q_dim)?;
        let qr = rt.scratch("qr", 0, batch * q_dim)?;
        let attn = rt.scratch("attn", 0, batch * q_dim)?;
        let proj = rt.scratch("proj", 0, batch * dim)?;
        let post = rt.scratch("post", 0, batch * dim)?;
        let gate = rt.scratch("gate", 0, batch * ffn_dim)?;
        let up = rt.scratch("up", 0, batch * ffn_dim)?;
        let act = rt.scratch("act", 0, batch * ffn_dim)?;
        let scores = rt.scratch("scores", 0, batch * n_heads * (seq + batch - 1))?;
        if let StackInput::Hidden(rows) = input {
            rt.write_f32(h, 0, rows)?;
        }

        // Per layer: the norms' weights, its RoPE table, its K before
        // and after rotation and its V (read back for the host cache),
        // and its mirror.
        struct Bound {
            /// The pre-attention and pre-FFN norms: kind, weight, bias.
            pre: [Option<BoundNorm>; 2],
            /// The post-attention and post-FFN norm weights.
            post: [Option<DeviceBuf>; 2],
            tab: Option<(DeviceBuf, usize)>,
            /// K and V each move between two buffers, an op at a time.
            k: [DeviceBuf; 2],
            v: [DeviceBuf; 2],
            /// The QKV biases and QK-norm weights on the device.
            bias: [Option<DeviceBuf>; 3],
            qk_norm: [Option<DeviceBuf>; 2],
            cache: DeviceBuf,
            cache_words: usize,
            k_word: usize,
            v_word: usize,
            v_off: usize,
        }
        let mut bound = Vec::with_capacity(layers.len());
        for (i, s) in layers.iter().enumerate() {
            let li = l + i;
            let mut norm = |w: Option<&[f32]>| w.map(|w| rt.constant_f32(w)).transpose();
            let mut pre_of = |n: Option<PreNorm<'_>>| {
                n.map(|n| {
                    Ok::<_, frink_vulkan::device::VulkanError>((
                        n.kind,
                        norm(n.weight)?,
                        norm(n.bias)?,
                    ))
                })
                .transpose()
            };
            let pre = [pre_of(s.attn_norm)?, pre_of(s.ffn_norm)?];
            let post = [norm(s.post_attn_norm)?, norm(s.post_ffn_norm)?];
            let tab = match &s.rope {
                Some((t, rot)) => {
                    let b = rt.scratch("rope", i, t.len())?;
                    rt.write_f32(b, 0, t)?;
                    Some((b, *rot))
                }
                None => None,
            };
            let k = [
                rt.scratch("k", i, batch * kv_dim)?,
                rt.scratch("kr", i, batch * kv_dim)?,
            ];
            let v = [
                rt.scratch("v", i, batch * kv_dim)?,
                rt.scratch("v2", i, batch * kv_dim)?,
            ];
            let mut constant = |w: Option<&[f32]>| w.map(|w| rt.constant_f32(w)).transpose();
            let bias = [
                constant(s.q_bias)?,
                constant(s.k_bias)?,
                constant(s.v_bias)?,
            ];
            let qk_norm = [constant(s.q_norm)?, constant(s.k_norm)?];
            let cache = &mut kv_caches[li];
            let rows = cache.rows();
            let KvCache {
                k: hk,
                v: hv,
                vulkan_attn,
                ..
            } = cache;
            let m = VulkanKvMirror::sync(vulkan_attn, rt, hk, hv, rows, kv_dim, need)?;
            bound.push(Bound {
                pre,
                post,
                tab,
                k,
                v,
                bias,
                qk_norm,
                cache: m.buf(),
                cache_words: m.words(),
                k_word: m.k_word(pos),
                v_word: m.v_word(pos),
                v_off: m.v_off(),
            });
        }

        let (eps, post_eps) = (c.rms_norm_eps, c.post_norm_eps());
        let mut ops = Batch::for_positions(batch);
        if let StackInput::Token(t) = input {
            let table = self
                .vulkan_embedding_gather()
                .expect("checked at admission");
            ops.gather(table, t, h);
        }
        let mut finals = Vec::with_capacity(layers.len());
        for (s, b) in layers.iter().zip(&bound) {
            // --- attention ---
            let ([pre_attn, pre_ffn], [post_attn, post_ffn]) = (b.pre, b.post);
            // The block's input: normed (and biased), or the residual
            // stream itself.
            let pre_norm = |ops: &mut Batch<'_>, n: Option<BoundNorm>, src| match n {
                Some((kind, w, bias)) => {
                    ops.norm(kind, w, src, normed, dim, 1, eps);
                    match bias {
                        Some(bias) => {
                            ops.add_bias(normed, bias, normed_biased, dim);
                            normed_biased
                        }
                        None => normed,
                    }
                }
                None => src,
            };
            let x_attn = pre_norm(&mut ops, pre_attn, h);
            // Q, K and V each start in the first buffer of a pair, and
            // every op that follows writes the other one.
            let (mut qs, mut ks, mut vs) = ([q, qr], b.k, b.v);
            ops.matvec(s.q, (x_attn, 0), (qs[0], 0));
            ops.matvec(s.k, (x_attn, 0), (ks[0], 0));
            ops.matvec(s.v, (x_attn, 0), (vs[0], 0));
            for (pair, bias, width) in [
                (&mut qs, b.bias[0], q_dim),
                (&mut ks, b.bias[1], kv_dim),
                (&mut vs, b.bias[2], kv_dim),
            ] {
                if let Some(bias) = bias {
                    ops.add_bias(pair[0], bias, pair[1], width);
                    pair.swap(0, 1);
                }
            }
            for (pair, w, heads, norm) in [
                (&mut qs, b.qk_norm[0], n_heads, s.q_norm),
                (&mut ks, b.qk_norm[1], n_kv, s.k_norm),
            ] {
                if let (Some(w), Some(norm)) = (w, norm) {
                    // One `head_dim` weight per head, or one weight over
                    // the whole projection.
                    let (n, rows) = if norm.len() == hd {
                        (hd, heads)
                    } else {
                        (heads * hd, 1)
                    };
                    ops.rms_norm(w, pair[0], pair[1], n, rows, eps);
                    pair.swap(0, 1);
                }
            }
            if let Some((tab, rot)) = b.tab {
                ops.rope(tab, qs[0], qs[1], n_heads, hd, rot, self.rope_is_neox());
                ops.rope(tab, ks[0], ks[1], n_kv, hd, rot, self.rope_is_neox());
                qs.swap(0, 1);
                ks.swap(0, 1);
            }
            // What the host cache takes back: K as attention reads it.
            let (q_in, k_in, v_in) = (qs[0], ks[0], vs[0]);
            finals.push((k_in, v_in));
            ops.kv_store(k_in, b.cache, kv_dim, b.k_word, b.cache_words);
            ops.kv_store(v_in, b.cache, kv_dim, b.v_word, b.cache_words);
            ops.attention(
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
            ops.matvec(s.o, (attn, 0), (proj, 0));
            // A post-norm is on the block's output, before the add.
            let attn_out = match post_attn {
                Some(w) => {
                    ops.rms_norm(w, proj, post, dim, 1, post_eps);
                    post
                }
                None => proj,
            };
            ops.add(h, attn_out, h_mid, dim);
            // --- FFN ---
            let x_ffn = pre_norm(&mut ops, pre_ffn, h_mid);
            ops.matvec(s.gate, (x_ffn, 0), (gate, 0));
            ops.matvec(s.up, (x_ffn, 0), (up, 0));
            ops.swiglu(gate, up, act, ffn_dim);
            ops.matvec(s.down, (act, 0), (proj, 0));
            let ffn_out = match post_ffn {
                Some(w) => {
                    ops.rms_norm(w, proj, post, dim, 1, post_eps);
                    post
                }
                None => proj,
            };
            ops.add(h_mid, ffn_out, h, dim);
        }
        assert!(ops.len() <= layers.iter().map(StackLayer::ops).sum::<usize>() + 1);
        rt.submit_batch(&ops)?;

        let new_hidden = rt.read_f32(h, 0, batch * dim)?;
        let mut rows = Vec::with_capacity(layers.len());
        for (k, v) in finals {
            rows.push((
                rt.read_f32(k, 0, batch * kv_dim)?,
                rt.read_f32(v, 0, batch * kv_dim)?,
            ));
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
