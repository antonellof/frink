//! The dense B=1 decode stack: every layer of a dense model in one
//! command buffer, hidden state resident on the GPU across layers.
//!
//! Split out of `attn.rs` along the seam GitHub issue #149 touches: the
//! host-side cost of encoding this stack is the whole remaining Metal
//! decode gap, so the encode loop needs to live somewhere it can be read
//! and changed in full. Per-token GPU/host accounting is in
//! `crate::timing::gpu_timing_note` and `crate::mem_ranges`.

use crate::attn::{
    assert_freq_factors_len, borrow_decode_scratch, copy_f32_into, encode_attn_extras,
    encode_gqa_with_kv, encode_kv_store_append, encode_rope, resident_attn_extras, LayerRope,
    MetalKvBuffers, MetalRope, RopeTarget, ScratchCaps,
};
use crate::elem::{encode_argmax, encode_gelu_mul, encode_silu_mul, encode_vec_add};
use crate::embd::{encode_get_rows, EmbdKind};
use crate::gpu::{
    compute_encoder_concurrent, encode_matvec, resident_f32_buffer, resident_weight_buffer,
    shared_metal, MatvecLaunch, MetalError,
};
use crate::kernel_timing::SpanClock;
use crate::mem_ranges::MemRanges;
use crate::norm::{encode_add_rms_norm, encode_rms_norm};
use crate::ple::{PerLayerGateMetal, PleInputMetal};
use crate::timing::{commit_wait_note, SubmitClock};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

/// Optional attention epilogue ops applied between the QKV matvecs and
/// RoPE, in CPU-path order: bias add (Qwen2-family `qkv_bias`), then
/// QK-RMSNorm — per-head (Qwen3 / Gemma-3, `weight.len() == head_dim`)
/// or whole-vector (OLMoE, `weight.len() == n_heads|n_kv_heads * head_dim`).
/// `attn_logit_softcap` is applied inside GQA after score scaling
/// (Gemma-2); when set, FA-vec is skipped in favour of the legacy kernel
/// unless the FA-vec softcap path is enabled.
#[derive(Default)]
pub struct AttnExtras<'a> {
    pub q_bias: Option<&'a [f32]>,
    pub k_bias: Option<&'a [f32]>,
    pub v_bias: Option<&'a [f32]>,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// Per-head RMSNorm of V (`head_dim` weights). Gemma-4 norms V with
    /// no weight at all (`ggml_rms_norm` on `Vcur`); its caller passes
    /// ones, which is the same arithmetic.
    pub v_norm: Option<&'a [f32]>,
    pub attn_logit_softcap: Option<f32>,
    /// gpt-oss's attention sinks, one logit per Q head: a key with no
    /// value in every head's softmax. Read inside the attention kernels
    /// (`crate::attn::AttnMask`), so it encodes nothing here.
    pub sinks: Option<&'a [f32]>,
    /// The bias on `wo`, added to the projected branch before the
    /// residual (gpt-oss). One `vec_add_rows` after the `o` projection.
    pub o_bias: Option<&'a [f32]>,
}

impl AttnExtras<'_> {
    /// True when `encode_attn_extras` would encode at least one dispatch.
    ///
    /// `attn_logit_softcap` is deliberately NOT part of this: it is a
    /// scalar consumed inside the GQA kernel, so a Gemma-2 layer that
    /// sets only the softcap encodes nothing here.
    ///
    /// The caller uses this to skip the hazard check entirely. That
    /// matters more than it looks: a `begin_op` around zero dispatches
    /// still emits a full `memoryBarrierWithResources`, and on
    /// Llama-3.2-1B -- no biases, no QK-norm -- that was 16 of the 176
    /// barriers a decode token encoded, ordering nothing against
    /// nothing (GitHub issue #149).
    ///
    /// `attn_extras_predicate_lists_every_field_that_encodes` destructures
    /// the struct exhaustively, so a new field cannot be added without
    /// deciding which side of this predicate it falls on.
    pub fn encodes_anything(&self) -> bool {
        self.q_bias.is_some()
            || self.k_bias.is_some()
            || self.v_bias.is_some()
            || self.q_norm.is_some()
            || self.k_norm.is_some()
            || self.v_norm.is_some()
    }

    /// The two fields only the fused stacks apply (the decode stacks and
    /// the prefill stack). A per-layer launch refuses a layer carrying
    /// either rather than attending without the sinks or projecting
    /// without the bias.
    pub fn stack_only(&self) -> bool {
        self.sinks.is_some() || self.o_bias.is_some()
    }

    /// `Err` for a layer [`Self::stack_only`] keeps off a per-layer launch.
    pub(crate) fn refuse_stack_only(&self) -> Result<(), MetalError> {
        if self.stack_only() {
            return Err(MetalError::Unsupported(
                "attention sinks / wo bias run only on the fused stacks",
            ));
        }
        Ok(())
    }

    /// True when any field touches K or V. A layer that reads another
    /// layer's cache ([`LayerKv::Shared`]) projects neither, so it must
    /// carry none of these.
    pub fn touches_kv(&self) -> bool {
        self.k_bias.is_some()
            || self.v_bias.is_some()
            || self.k_norm.is_some()
            || self.v_norm.is_some()
    }
}

/// Where one layer's attention gets its keys and values.
///
/// `slot` indexes the `kvs` slice the stack is handed. Every model but
/// Gemma-4 gives each layer its own slot (`Own`, slot = its position in
/// the stack); Gemma-4's last `shared_kv_layers` layers project no K or
/// V at all and attend over an earlier layer's cache as that layer left
/// it for this very token (`Shared`). One enum rather than optional K/V
/// launches beside a separate index, so a layer cannot project a K that
/// nothing appends, or append to a slot another layer owns.
#[derive(Clone, Copy)]
pub enum LayerKv<L> {
    Own { k: L, v: L, slot: usize },
    Shared { slot: usize },
}

impl<L> LayerKv<L> {
    pub fn slot(&self) -> usize {
        match self {
            Self::Own { slot, .. } | Self::Shared { slot } => *slot,
        }
    }

    /// The K and V projections, when this layer has its own.
    pub fn own(&self) -> Option<(&L, &L)> {
        match self {
            Self::Own { k, v, .. } => Some((k, v)),
            Self::Shared { .. } => None,
        }
    }
}

/// Checks a stack's KV wiring before anything is encoded: each slot is
/// owned by exactly one layer, every shared read comes after its owner,
/// and a sharing layer carries no K/V epilogue. Returns, per slot,
/// whether anything owns it.
pub(crate) fn check_layer_kv<L>(
    kv: impl Iterator<Item = (LayerKv<L>, bool)>,
    n_slots: usize,
) -> Vec<bool> {
    let mut owned = vec![false; n_slots];
    for (i, (kv, touches_kv)) in kv.enumerate() {
        let slot = kv.slot();
        assert!(slot < n_slots, "layer {i}: KV slot {slot} of {n_slots}");
        match kv {
            LayerKv::Own { .. } => {
                assert!(!owned[slot], "layer {i}: KV slot {slot} owned twice");
                owned[slot] = true;
            }
            LayerKv::Shared { .. } => {
                assert!(
                    owned[slot],
                    "layer {i}: reads KV slot {slot} before any layer writes it"
                );
                assert!(
                    !touches_kv,
                    "layer {i}: a shared-KV layer has no K/V to bias or norm"
                );
            }
        }
    }
    owned
}

/// Which buffer [`launch_decode_dense_stack`] hands back. Named, not
/// inferred from the length: a vocabulary as wide as the hidden state
/// (the synthetic fixtures' 512 and 512) used to download the hidden
/// state in place of the logits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Download {
    ArgmaxId,
    Logits,
    NormedHidden,
    Hidden,
}

/// Per-layer launches + norms for [`launch_decode_dense_stack`].
pub struct DenseLayerMetal<'a> {
    pub attn_norm_w: &'a [f32],
    pub ffn_norm_w: &'a [f32],
    pub q: MatvecLaunch<'a>,
    pub kv: LayerKv<MatvecLaunch<'a>>,
    pub o: MatvecLaunch<'a>,
    pub gate: MatvecLaunch<'a>,
    pub up: MatvecLaunch<'a>,
    pub down: MatvecLaunch<'a>,
    pub extras: AttnExtras<'a>,
    /// This layer's RoPE base AND divisors. Both halves, always: a
    /// Gemma-3 SWA layer differs from its full-attention neighbours in
    /// both (`rope_theta_swa`, and no linear scale folded into the
    /// divisors). See [`LayerRope`].
    ///
    /// `None` = this layer does not rotate at all (llama.cpp's
    /// per-layer `use_rope`, `frink_models::rope_layers`).
    pub rope: Option<LayerRope<'a>>,
    /// Sliding-window size for this layer (`None` = full causal).
    pub window: Option<usize>,
    /// Gemma post-attention / post-FFN sandwich norms, applied to the
    /// block output *before* the residual add.
    pub post_attn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
    /// Gemma-4's per-layer embedding gate, after the FFN residual; reads
    /// the stack's `ple` plane.
    pub per_layer: Option<PerLayerGateMetal<'a, MatvecLaunch<'a>>>,
    /// Gemma-4's `layer_output_scale`: the whole residual stream is
    /// multiplied by it at the end of the layer.
    pub out_scale: Option<f32>,
}

impl DenseLayerMetal<'_> {
    /// Does this layer need every residual applied eagerly, so `h` holds
    /// the full stream at the points where it is normed, gated or
    /// scaled? A post-norm does (see the sandwich comment in the stack),
    /// and so does anything that reads or scales `h` after the FFN.
    fn needs_eager_residual(&self) -> bool {
        self.post_attn_norm.is_some()
            || self.post_ffn_norm.is_some()
            || self.per_layer.is_some()
            || self.out_scale.is_some()
    }
}

/// Optional on-GPU embedding gather at the start of
/// [`launch_decode_dense_stack`] (skips host `dequant_row` + upload).
pub struct EmbdGatherMetal<'a> {
    pub kind: EmbdKind,
    pub weights: &'a [u8],
    pub rows: usize,
    pub row_bytes: usize,
    pub n_cols: usize,
    pub token_id: usize,
}

/// All dense layers in **one** command buffer (one wait). Hidden stays on
/// GPU across layers — Crane-style residency for B=1 decode.
/// When `embd` is `Some`, gathers that token row into scratch `h` on-GPU
/// instead of copying a host-provided `hidden` slice.
/// When `final_norm_w` + `output` are provided, also runs final RMSNorm +
/// lm_head on-GPU. With `argmax_only`, runs argmax and returns a
/// **1-element** `vec![token_id as f32]`; otherwise downloads vocab logits.
///
/// Chunked multi-CB early-commit (llama `n_main` style) was tried on Host B
/// and regressed decode tok/s — see `…_multicb*` receipts; kept single CB.
#[allow(clippy::too_many_arguments)]
pub fn launch_decode_dense_stack(
    hidden: &[f32],
    layers: &[DenseLayerMetal<'_>],
    kvs: &mut [MetalKvBuffers],
    n_heads: usize,
    rope_layout: MetalRope,
    pos: usize,
    rms_eps: f32,
    final_norm_w: Option<&[f32]>,
    output: Option<&MatvecLaunch<'_>>,
    argmax_only: bool,
    embd: Option<&EmbdGatherMetal<'_>>,
    gelu_ffn: bool,
    ple: Option<&PleInputMetal<'_, MatvecLaunch<'_>>>,
) -> Result<Vec<f32>, MetalError> {
    assert!(!layers.is_empty());
    let hidden_dim = match embd {
        Some(e) => e.n_cols,
        None => hidden.len(),
    };
    assert!(hidden_dim > 0);
    // Head width and KV head count are the SLOT's, per layer: Gemma-4's
    // sliding and full layers differ in both widths.
    let owned = check_layer_kv(
        layers.iter().map(|l| (l.kv, l.extras.touches_kv())),
        kvs.len(),
    );
    for (kv, owned) in kvs.iter().zip(&owned) {
        assert!(*owned, "a KV slot no layer of this stack writes");
        assert_eq!(pos, kv.seq_len);
        if kv.seq_len >= kv.capacity {
            return Err(MetalError::CommandFailed);
        }
    }
    for layer in layers.iter() {
        let kv = &kvs[layer.kv.slot()];
        assert_eq!(layer.q.rows, n_heads * kv.head_dim);
        assert_freq_factors_len(
            layer.rope.and_then(|r| r.freq_factors),
            rope_layout,
            kv.head_dim,
        );
        if layer.per_layer.is_some() {
            assert!(ple.is_some(), "a per-layer gate with no PLE plane");
        }
    }
    if let Some(p) = ple {
        p.check(1, p.model_proj.rows);
    }

    let max_q = layers.iter().map(|l| l.q.rows).max().unwrap();
    let max_kv = layers
        .iter()
        .filter_map(|l| l.kv.own().map(|(k, v)| k.rows.max(v.rows)))
        .max()
        .unwrap_or(1);
    let max_gate = layers
        .iter()
        .map(|l| {
            l.gate
                .rows
                .max(l.per_layer.as_ref().map_or(0, |p| p.gate.rows))
        })
        .max()
        .unwrap();
    let attn_elems = n_heads * kvs.iter().map(|kv| kv.head_dim).max().unwrap();
    let logits_rows = output.map(|o| o.rows);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let mut scratch_guard = borrow_decode_scratch(
        device,
        ScratchCaps {
            hidden: hidden_dim,
            max_q,
            max_kv,
            attn: attn_elems,
            max_gate,
            logits: logits_rows.unwrap_or(0),
        },
    )?;
    let scratch = scratch_guard.as_ref().expect("scratch just ensured");
    let h_buf = &scratch.h;
    if let Some(e) = embd {
        assert_eq!(e.n_cols, hidden_dim);
        assert!(e.token_id < e.rows);
        assert_eq!(e.weights.len(), e.rows * e.row_bytes);
        // Gather runs in the same CB below (after encoder create).
    } else {
        assert_eq!(hidden.len(), hidden_dim);
        copy_f32_into(h_buf, hidden);
    }
    let x_buf = &scratch.x;
    let x2_buf = &scratch.x2;
    let q_buf = &scratch.q;
    let k_buf = &scratch.k;
    let v_buf = &scratch.v;
    let attn_buf = &scratch.attn;
    let o_buf = &scratch.o;
    let gate_buf = &scratch.gate;
    let up_buf = &scratch.up;
    let act_buf = &scratch.act;
    let down_buf = &scratch.down;
    let logits_buf = scratch.logits.as_ref();
    let argmax_idx_buf = &scratch.argmax_idx;

    // One resident buffer PER LAYER. `resident_f32_buffer` keys its
    // cache on (pointer, len), so the at-most-two distinct divisor sets
    // an alternating-SWA model has are uploaded once each and every
    // layer past the first two is a cache hit -- no per-layer upload,
    // and no table of "which set does layer i use" for a call site to
    // get out of step with.
    let ff_resident = layers
        .iter()
        .map(|l| match l.rope.and_then(|r| r.freq_factors) {
            Some(ff) => resident_f32_buffer(device, ff).map(Some),
            None => Ok(None),
        })
        .collect::<Result<Vec<_>, MetalError>>()?;

    // Held to the end of the call: the plane is read by every layer.
    let ple_guard = match ple {
        Some(p) => Some(crate::ple::borrow_ple_scratch(device, p.plane_elems(1))?),
        None => None,
    };
    let ple_scratch = ple_guard.as_ref().map(|g| g.as_ref().expect("ple scratch"));
    if let (Some(p), Some(sc)) = (ple, ple_scratch) {
        p.upload_tok(sc);
    }

    let clock = SubmitClock::start();
    let mut cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let early = crate::timing::early_commit_layer(layers.len());
    // Gemma-style post-norms ("sandwich"), and Gemma-4's per-layer gate
    // and output scale: these layers take the EAGER residual path below,
    // which is what makes concurrent encode safe here.
    let sandwich = layers.iter().any(DenseLayerMetal::needs_eager_residual);
    // Every model encodes concurrently: gate∥up and Q∥K∥V overlap, and the
    // hazard tracker emits a barrier only where a dispatch reads or
    // overwrites something still in flight, narrowed to those resources.
    //
    // Sandwich models (Gemma post-norms) used to be forced onto the serial
    // encoder, because concurrent dispatch with in-place RMSNorm and
    // DEFERRED residuals diverged from CPU on Gemma-2 B=1 decode. That fix
    // landed two changes at once -- serial encode AND eager residuals -- and
    // the eager residuals are the half that mattered: with them, every op in
    // this function declares its own reads and writes (`encode_gqa_with_kv`
    // self-tracks, including the f16 dequant scratch no caller can name), so
    // concurrency is safe by construction rather than by scheduling luck.
    //
    // Measured on an M2 Pro, interleaved, GPU-clock: Gemma-2-2B Q4_K_M decode
    // 13.23 -> 12.20 ms/token, and greedy output stays byte-identical to the
    // serial encoder across Gemma-2 and Gemma-3 on every prompt tried.
    let (mut encoder, mut mrs) = (compute_encoder_concurrent(&cmd_buf)?, MemRanges::new());
    // `FRINK_METAL_KERNEL_TIMING=1`: one sampled encoder per op group,
    // so the report says which KIND of dispatch the GPU time went to.
    // A no-op otherwise. 18 groups per layer is the ceiling below.
    let mut spans = SpanClock::begin(device, 18 * layers.len() + 8);
    // Calibration: an encoder with nothing in it measures the boundary
    // cost the instrument adds to every span below.
    spans.span(&cmd_buf, &mut encoder, "(empty encoder)")?;

    let embd_resident = if let Some(e) = embd {
        let w = resident_weight_buffer(device, e.weights)?;
        spans.span(&cmd_buf, &mut encoder, "embd_gather")?;
        mrs.begin_op(&encoder, &[], &[h_buf]);
        encode_get_rows(
            &encoder,
            device,
            e.kind,
            &w,
            h_buf,
            e.row_bytes as u32,
            e.n_cols as u32,
            e.token_id as u32,
        )?;
        mrs.end_op(&[], &[h_buf]);
        Some(w)
    } else {
        None
    };
    let _embd_resident = embd_resident;

    if let (Some(p), Some(sc)) = (ple, ple_scratch) {
        spans.span(&cmd_buf, &mut encoder, "ple_inputs")?;
        p.encode_decode(&encoder, &mut mrs, device, sc, h_buf, rms_eps)?;
    }

    // Gemma sandwich (post_attn / post_ffn) must apply residuals eagerly —
    // same shape as the working prefill stack / CPU path. Deferred
    // `h += down` fused into the next layer's attn_norm matches SmolLM2
    // (no post-norms) but diverges for Gemma-2 Metal greedy (BOS loops /
    // `*` spam) even when GQA unit tests pass.
    // `sandwich` was computed above (also selects serial encoder).

    for (layer_idx, layer) in layers.iter().enumerate() {
        if early == Some(layer_idx) {
            // Hand the first layers to the GPU and keep encoding. A new
            // command buffer on the same queue orders every tracked
            // buffer against the previous one, so the tracker restarts.
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
            encoder = compute_encoder_concurrent(&cmd_buf)?;
            mrs = MemRanges::new();
        }
        assert_eq!(layer.attn_norm_w.len(), hidden_dim);
        assert_eq!(layer.ffn_norm_w.len(), hidden_dim);
        assert_eq!(layer.o.rows, hidden_dim);
        assert_eq!(layer.down.rows, hidden_dim);
        assert_eq!(layer.gate.rows, layer.up.rows);
        let kv = &kvs[layer.kv.slot()];
        let (head_dim, n_kv_heads) = (kv.head_dim, kv.n_kv_heads);
        let own_kv = layer.kv.own();

        let attn_nw = resident_f32_buffer(device, layer.attn_norm_w)?;
        let ffn_nw = resident_f32_buffer(device, layer.ffn_norm_w)?;
        let q_w = resident_weight_buffer(device, layer.q.weights)?;
        let kv_w = match own_kv {
            Some((k, v)) => Some((
                resident_weight_buffer(device, k.weights)?,
                resident_weight_buffer(device, v.weights)?,
            )),
            None => None,
        };
        let o_w = resident_weight_buffer(device, layer.o.weights)?;
        let gate_w = resident_weight_buffer(device, layer.gate.weights)?;
        let up_w = resident_weight_buffer(device, layer.up.weights)?;
        let down_w = resident_weight_buffer(device, layer.down.weights)?;
        let kv_k = kv.k.as_ref();
        let kv_v = kv.v.as_ref();
        // The sinks and `wo`'s bias (gpt-oss), resident; a layer with
        // neither looks nothing up.
        let tail = resident_attn_extras(device, &layer.extras)?;

        // Pre-LN: layer 0 norms raw hidden; later layers either fuse the
        // previous FFN residual into attn_norm (non-sandwich) or just
        // RMSNorm (sandwich already applied `h += down` eagerly).
        if layer_idx == 0 || sandwich {
            spans.span(&cmd_buf, &mut encoder, "attn_norm")?;
            mrs.begin_op(&encoder, &[h_buf], &[x_buf]);
            encode_rms_norm(
                &encoder,
                device,
                h_buf,
                &attn_nw.buffer,
                x_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
            mrs.end_op(&[h_buf], &[x_buf]);
        } else {
            spans.span(&cmd_buf, &mut encoder, "add_attn_norm")?;
            mrs.begin_op(&encoder, &[h_buf, down_buf], &[h_buf, x_buf]);
            encode_add_rms_norm(
                &encoder,
                device,
                h_buf,
                down_buf,
                &attn_nw.buffer,
                x_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
            mrs.end_op(&[h_buf, down_buf], &[h_buf, x_buf]);
        }
        // Q∥K∥V (Q alone on a layer that reads another layer's KV).
        spans.span(&cmd_buf, &mut encoder, "qkv")?;
        if let (Some((k, v)), Some((k_w, v_w))) = (own_kv, kv_w.as_ref()) {
            mrs.begin_op(&encoder, &[x_buf], &[q_buf, k_buf, v_buf]);
            encode_matvec(&encoder, device, &layer.q, &q_w, x_buf, q_buf)?;
            encode_matvec(&encoder, device, k, k_w, x_buf, k_buf)?;
            encode_matvec(&encoder, device, v, v_w, x_buf, v_buf)?;
            mrs.end_op(&[x_buf], &[q_buf, k_buf, v_buf]);
        } else {
            mrs.begin_op(&encoder, &[x_buf], &[q_buf]);
            encode_matvec(&encoder, device, &layer.q, &q_w, x_buf, q_buf)?;
            mrs.end_op(&[x_buf], &[q_buf]);
        }

        // Skipped entirely when the layer has no biases and no QK-norm:
        // the hazard check around zero dispatches is still a real
        // barrier. See `AttnExtras::encodes_anything`.
        if layer.extras.encodes_anything() {
            spans.span(&cmd_buf, &mut encoder, "attn_extras")?;
            mrs.begin_op(&encoder, &[q_buf, k_buf, v_buf], &[q_buf, k_buf, v_buf]);
            encode_attn_extras(
                &encoder,
                device,
                &layer.extras,
                q_buf,
                k_buf,
                v_buf,
                layer.q.rows,
                own_kv.map_or(0, |(k, _)| k.rows),
                own_kv.map_or(0, |(_, v)| v.rows),
                n_heads,
                n_kv_heads,
                head_dim,
                rms_eps,
            )?;
            mrs.end_op(&[q_buf, k_buf, v_buf], &[q_buf, k_buf, v_buf]);
        }

        // `None` = this layer does not rotate, and then there is no
        // dispatch at all: q and k reach the KV store and the attention
        // exactly as the projections left them, which is what ggml does
        // when the graph never builds the `ggml_rope_ext` node. Writing
        // the RoPE in unconditionally is how the fused stacks lost the
        // final norm for OLMo-1, one layer of indirection away.
        if let Some(layer_rope) = layer.rope {
            let ff_buf = ff_resident[layer_idx].as_ref().map(|b| b.buffer.as_ref());
            // Q and K in one dispatch: same theta, position and freq
            // factors, different buffers, and RoPE is per-head independent.
            spans.span(&cmd_buf, &mut encoder, "rope")?;
            // A shared-KV layer rotates Q only: its K was rotated, at
            // this position, by the layer that wrote the slot.
            let qk: &[&ProtocolObject<dyn MTLBuffer>] = if own_kv.is_some() {
                &[q_buf, k_buf]
            } else {
                &[q_buf]
            };
            mrs.begin_op(&encoder, qk, qk);
            encode_rope(
                &encoder,
                device,
                rope_layout,
                RopeTarget {
                    vecs: q_buf,
                    n_heads: n_heads as u32,
                },
                own_kv.map(|_| RopeTarget {
                    vecs: k_buf,
                    n_heads: n_kv_heads as u32,
                }),
                head_dim as u32,
                layer_rope.theta,
                pos as u32,
                ff_buf,
            )?;
            mrs.end_op(qk, qk);
        }

        if own_kv.is_some() {
            let token_elems = (n_kv_heads * head_dim) as u32;
            let offset = (pos * n_kv_heads * head_dim) as u32;
            spans.span(&cmd_buf, &mut encoder, "kv_append")?;
            mrs.begin_op(&encoder, &[k_buf, v_buf], &[kv_k, kv_v]);
            // K and V in one dispatch: same offset, same length, disjoint
            // destinations.
            encode_kv_store_append(&encoder, device, k_buf, v_buf, kv, offset, token_elems)?;
            mrs.end_op(&[k_buf, v_buf], &[kv_k, kv_v]);
        }

        let new_seq = (pos + 1) as u32;
        // Sliding window: only the last `window` positions (incl. current)
        // are visible, matching `causal_gqa_attention_windowed`.
        let kv_start = match layer.window {
            Some(w) => (pos + 1).saturating_sub(w) as u32,
            None => 0,
        };
        spans.span(&cmd_buf, &mut encoder, "gqa")?;
        // Self-tracking: with a quantized KV cache this also writes a shared
        // f16 dequant scratch that no caller can name.
        encode_gqa_with_kv(
            &encoder,
            &mut mrs,
            device,
            q_buf,
            kv,
            attn_buf,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            new_seq,
            kv_start,
            layer.extras.attn_logit_softcap,
            tail.sinks(),
        )?;

        spans.span(&cmd_buf, &mut encoder, "o_proj")?;
        mrs.begin_op(&encoder, &[attn_buf], &[o_buf]);
        encode_matvec(&encoder, device, &layer.o, &o_w, attn_buf, o_buf)?;
        mrs.end_op(&[attn_buf], &[o_buf]);
        tail.encode_o_bias(&encoder, &mut mrs, device, o_buf, hidden_dim, 1)?;

        // Gemma sandwich norm: normalize the attn block output *before*
        // the residual add (in-place: each thread reads x[i] only after
        // the barriered reduction, so out == x is safe).
        if let Some(post) = layer.post_attn_norm {
            assert_eq!(post.len(), hidden_dim);
            let pw = resident_f32_buffer(device, post)?;
            spans.span(&cmd_buf, &mut encoder, "post_attn_norm")?;
            mrs.begin_op(&encoder, &[o_buf], &[o_buf]);
            encode_rms_norm(
                &encoder,
                device,
                o_buf,
                &pw.buffer,
                o_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
            mrs.end_op(&[o_buf], &[o_buf]);
        }
        // Attn residual + ffn_norm in one dispatch, for every model.
        //
        // Sandwich layers used to split this into `vec_add` then `rms_norm`,
        // which is the same arithmetic in two dispatches: `post_attn_norm`
        // has already been applied to `o_buf` in place above, so both paths
        // compute `h += o` then `x2 = rms_norm(h)`. The split was a leftover
        // from the serial-encoder era -- `encode_add_rms_norm` writes `h`
        // itself, so the residual is just as eager as the two-dispatch form.
        spans.span(&cmd_buf, &mut encoder, "add_ffn_norm")?;
        mrs.begin_op(&encoder, &[h_buf, o_buf], &[h_buf, x2_buf]);
        encode_add_rms_norm(
            &encoder,
            device,
            h_buf,
            o_buf,
            &ffn_nw.buffer,
            x2_buf,
            hidden_dim as u32,
            rms_eps,
        )?;
        mrs.end_op(&[h_buf, o_buf], &[h_buf, x2_buf]);
        // gate ∥ up (llama concurrent)
        spans.span(&cmd_buf, &mut encoder, "gate_up")?;
        mrs.begin_op(&encoder, &[x2_buf], &[gate_buf, up_buf]);
        encode_matvec(&encoder, device, &layer.gate, &gate_w, x2_buf, gate_buf)?;
        encode_matvec(&encoder, device, &layer.up, &up_w, x2_buf, up_buf)?;
        mrs.end_op(&[x2_buf], &[gate_buf, up_buf]);

        spans.span(&cmd_buf, &mut encoder, "glu")?;
        mrs.begin_op(&encoder, &[gate_buf, up_buf], &[act_buf]);
        if gelu_ffn {
            encode_gelu_mul(
                &encoder,
                device,
                gate_buf,
                up_buf,
                act_buf,
                layer.gate.rows as u32,
            )?;
        } else {
            encode_silu_mul(
                &encoder,
                device,
                gate_buf,
                up_buf,
                act_buf,
                layer.gate.rows as u32,
            )?;
        }
        mrs.end_op(&[gate_buf, up_buf], &[act_buf]);

        spans.span(&cmd_buf, &mut encoder, "down")?;
        mrs.begin_op(&encoder, &[act_buf], &[down_buf]);
        encode_matvec(&encoder, device, &layer.down, &down_w, act_buf, down_buf)?;
        mrs.end_op(&[act_buf], &[down_buf]);

        if let Some(post) = layer.post_ffn_norm {
            assert_eq!(post.len(), hidden_dim);
            let pw = resident_f32_buffer(device, post)?;
            spans.span(&cmd_buf, &mut encoder, "post_ffn_norm")?;
            mrs.begin_op(&encoder, &[down_buf], &[down_buf]);
            encode_rms_norm(
                &encoder,
                device,
                down_buf,
                &pw.buffer,
                down_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
            mrs.end_op(&[down_buf], &[down_buf]);
        }
        if sandwich {
            spans.span(&cmd_buf, &mut encoder, "residual_add")?;
            // Eager FFN residual — next layer attn_norm is plain RMSNorm.
            mrs.begin_op(&encoder, &[h_buf, down_buf], &[h_buf]);
            encode_vec_add(&encoder, device, h_buf, down_buf, hidden_dim as u32)?;
            mrs.end_op(&[h_buf, down_buf], &[h_buf]);
        }
        if let Some(gate) = layer.per_layer.as_ref() {
            let (p, sc) = (ple.expect("checked above"), ple_scratch.expect("with ple"));
            spans.span(&cmd_buf, &mut encoder, "per_layer_gate")?;
            gate.encode_decode(
                &encoder,
                &mut mrs,
                device,
                sc,
                p.n_layer,
                &crate::ple::GateScratch {
                    h: h_buf,
                    gate: gate_buf,
                    act: act_buf,
                    out: down_buf,
                    half: None,
                },
                hidden_dim,
                rms_eps,
            )?;
        }
        if let Some(scale) = layer.out_scale {
            spans.span(&cmd_buf, &mut encoder, "out_scale")?;
            crate::ple::encode_out_scale(&encoder, &mut mrs, device, h_buf, scale, hidden_dim)?;
        }
        // Non-sandwich: defer `h += down` until the next layer's attn_norm
        // (or final_norm) so it fuses with that RMSNorm. `down_buf` stays in
        // the tracker's dst set, so the next layer's fused norm barriers
        // against it exactly once. Last layer handled below.
    }

    // Final norm / lm_head. Sandwich already applied every FFN residual;
    // non-sandwich still has a deferred last-layer `down` to fold in.
    let (download_n, download) = if let Some(fnw) = final_norm_w {
        assert_eq!(fnw.len(), hidden_dim);
        let fn_buf = resident_f32_buffer(device, fnw)?;
        if sandwich {
            spans.span(&cmd_buf, &mut encoder, "final_norm")?;
            mrs.begin_op(&encoder, &[h_buf], &[x_buf]);
            encode_rms_norm(
                &encoder,
                device,
                h_buf,
                &fn_buf.buffer,
                x_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
            mrs.end_op(&[h_buf], &[x_buf]);
        } else {
            spans.span(&cmd_buf, &mut encoder, "add_final_norm")?;
            mrs.begin_op(&encoder, &[h_buf, down_buf], &[h_buf, x_buf]);
            encode_add_rms_norm(
                &encoder,
                device,
                h_buf,
                down_buf,
                &fn_buf.buffer,
                x_buf,
                hidden_dim as u32,
                rms_eps,
            )?;
            mrs.end_op(&[h_buf, down_buf], &[h_buf, x_buf]);
        }
        if let (Some(out_l), Some(logits)) = (output, logits_buf) {
            assert_eq!(out_l.rows, logits_rows.unwrap());
            // RAW: lm_head reads `x_buf` written by the norm above. Without
            // ordering here Metal may overlap the matvec with the norm on
            // small hiddens (SmolLM2 h=576) and produce garbage logits /
            // greedy tokens while host lm_head after wait looks fine — the
            // tracker sees `x_buf` as src-after-dst and barriers.
            spans.span(&cmd_buf, &mut encoder, "lm_head")?;
            mrs.begin_op(&encoder, &[x_buf], &[logits.as_ref()]);
            let out_w = resident_weight_buffer(device, out_l.weights)?;
            encode_matvec(&encoder, device, out_l, &out_w, x_buf, logits)?;
            mrs.end_op(&[x_buf], &[logits.as_ref()]);
            if argmax_only {
                spans.span(&cmd_buf, &mut encoder, "argmax")?;
                mrs.begin_op(&encoder, &[logits.as_ref()], &[argmax_idx_buf]);
                encode_argmax(&encoder, device, logits, argmax_idx_buf, out_l.rows as u32)?;
                mrs.end_op(&[logits.as_ref()], &[argmax_idx_buf]);
                (1, Download::ArgmaxId)
            } else {
                (out_l.rows, Download::Logits)
            }
        } else {
            // final_norm ran but no lm_head — download normalized hidden
            // and mark x_buf resident for the next apply_gpu.
            (hidden_dim, Download::NormedHidden)
        }
    } else if sandwich {
        (hidden_dim, Download::Hidden)
    } else {
        spans.span(&cmd_buf, &mut encoder, "residual_add")?;
        // No final_norm: still apply the deferred last-layer FFN residual.
        mrs.begin_op(&encoder, &[h_buf, down_buf], &[h_buf]);
        encode_vec_add(&encoder, device, h_buf, down_buf, hidden_dim as u32)?;
        mrs.end_op(&[h_buf, down_buf], &[h_buf]);
        (hidden_dim, Download::Hidden)
    };

    encoder.endEncoding();
    commit_wait_note(&cmd_buf, "dense-decode/tok", 32, clock);
    spans.finish(&cmd_buf, "dense-decode/tok", 32);

    for kv in kvs.iter_mut() {
        kv.seq_len = pos + 1;
    }
    drop(ple_guard);

    let norm_resident = download == Download::NormedHidden;
    let src: &ProtocolObject<dyn MTLBuffer> = match download {
        Download::ArgmaxId => {
            let ptr = argmax_idx_buf.contents();
            // SAFETY: `argmax_idx` is a StorageModeShared buffer of at least
            // one u32, written by the argmax dispatch of the command buffer
            // waited on above.
            let idx = unsafe { *(ptr.as_ptr() as *const u32) as usize };
            return Ok(vec![idx as f32]);
        }
        Download::NormedHidden => x_buf,
        Download::Hidden => h_buf,
        Download::Logits => logits_buf.expect("logits buffer when downloading logits"),
    };
    let out_ptr = src.contents();
    // SAFETY: `src` is a StorageModeShared buffer allocated for at least
    // `download_n` f32 and written by the command buffer waited on above.
    let out =
        unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, download_n).to_vec() };

    // final_norm ran but no lm_head: `scratch.x` still holds exactly the
    // vector being returned, so say so and let the caller's next matvec
    // bind it instead of uploading a copy. Published UNDER THE GUARD, on
    // the vector's own address, so nothing but that vector can claim it.
    // See `crate::resident_act` for the version of this that used a raw
    // pointer and a length.
    if norm_resident {
        let scratch = scratch_guard
            .as_mut()
            .expect("scratch ensured at entry and held throughout");
        crate::resident_act::publish(scratch, &out);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{check_layer_kv, AttnExtras, LayerKv};

    fn own(slot: usize) -> (LayerKv<()>, bool) {
        (LayerKv::Own { k: (), v: (), slot }, true)
    }

    fn shared(slot: usize) -> (LayerKv<()>, bool) {
        (LayerKv::Shared { slot }, false)
    }

    /// Gemma-4 E2B's wiring in miniature: the last layers attend over
    /// the last sliding and the last full layer's caches. Every slot is
    /// owned exactly once, and nothing reads a slot before it is
    /// written in the same token.
    #[test]
    fn layer_kv_accepts_gemma4_shared_kv_wiring() {
        let wiring = [
            own(0),
            own(1),
            own(2),
            own(3),
            shared(2),
            shared(3),
            shared(2),
        ];
        assert_eq!(check_layer_kv(wiring.into_iter(), 4), vec![true; 4]);
        // A slot nobody owns is reported, not invented: the stacks refuse it.
        assert_eq!(
            check_layer_kv([own(0), shared(0)].into_iter(), 2),
            vec![true, false]
        );
    }

    #[test]
    #[should_panic(expected = "before any layer writes it")]
    fn layer_kv_refuses_a_read_before_its_owner() {
        check_layer_kv([shared(1), own(0), own(1)].into_iter(), 2);
    }

    #[test]
    #[should_panic(expected = "owned twice")]
    fn layer_kv_refuses_two_owners_of_one_slot() {
        check_layer_kv([own(0), own(0)].into_iter(), 1);
    }

    /// A sharing layer projects no K or V, so a K norm or V bias on it
    /// would be silently dropped by the stack rather than applied.
    #[test]
    #[should_panic(expected = "no K/V to bias or norm")]
    fn layer_kv_refuses_a_kv_epilogue_on_a_sharing_layer() {
        check_layer_kv([own(0), (LayerKv::Shared { slot: 0 }, true)].into_iter(), 1);
    }

    #[test]
    fn touches_kv_lists_every_field_on_k_or_v() {
        let w = [1.0f32];
        let none = AttnExtras {
            q_bias: Some(&w),
            q_norm: Some(&w),
            attn_logit_softcap: Some(30.0),
            ..AttnExtras::default()
        };
        assert!(
            !none.touches_kv(),
            "Q-side fields are a sharing layer's own"
        );
        for e in [
            AttnExtras {
                k_bias: Some(&w),
                ..AttnExtras::default()
            },
            AttnExtras {
                v_bias: Some(&w),
                ..AttnExtras::default()
            },
            AttnExtras {
                k_norm: Some(&w),
                ..AttnExtras::default()
            },
            AttnExtras {
                v_norm: Some(&w),
                ..AttnExtras::default()
            },
        ] {
            assert!(e.touches_kv());
        }
    }

    /// `encodes_anything` decides whether the decode stack pays a barrier
    /// for this layer's attention epilogue, so a field it forgets is a
    /// silently skipped dispatch -- the repo's dominant defect shape,
    /// two structures that must agree with nothing enforcing it.
    ///
    /// The exhaustive destructure below has no `..`, so adding a field to
    /// `AttnExtras` fails to COMPILE here until somebody decides whether
    /// it encodes a dispatch.
    #[test]
    fn attn_extras_predicate_lists_every_field_that_encodes() {
        let w = [1.0f32];
        let base = AttnExtras::default();
        let AttnExtras {
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
            v_norm,
            attn_logit_softcap,
            sinks,
            o_bias,
        } = &base;
        assert!(sinks.is_none() && o_bias.is_none());
        assert!(q_bias.is_none());
        assert!(k_bias.is_none());
        assert!(v_bias.is_none());
        assert!(q_norm.is_none());
        assert!(k_norm.is_none());
        assert!(v_norm.is_none());
        assert!(attn_logit_softcap.is_none());
        assert!(
            !base.encodes_anything(),
            "an empty epilogue encodes nothing, so it must not take a barrier"
        );

        // Every field that DOES encode a dispatch, one at a time.
        let encoders: [AttnExtras<'_>; 6] = [
            AttnExtras {
                q_bias: Some(&w),
                ..AttnExtras::default()
            },
            AttnExtras {
                k_bias: Some(&w),
                ..AttnExtras::default()
            },
            AttnExtras {
                v_bias: Some(&w),
                ..AttnExtras::default()
            },
            AttnExtras {
                q_norm: Some(&w),
                ..AttnExtras::default()
            },
            AttnExtras {
                k_norm: Some(&w),
                ..AttnExtras::default()
            },
            AttnExtras {
                v_norm: Some(&w),
                ..AttnExtras::default()
            },
        ];
        for (i, e) in encoders.iter().enumerate() {
            assert!(e.encodes_anything(), "field {i} encodes but is not listed");
        }

        // And the ones that do not: the softcap and the sinks are read
        // inside the GQA kernel, and `wo`'s bias is added after `wo`,
        // never encoded by `encode_attn_extras`.
        for e in [
            AttnExtras {
                attn_logit_softcap: Some(30.0),
                ..AttnExtras::default()
            },
            AttnExtras {
                sinks: Some(&w),
                ..AttnExtras::default()
            },
            AttnExtras {
                o_bias: Some(&w),
                ..AttnExtras::default()
            },
        ] {
            assert!(
                !e.encodes_anything(),
                "a field read by the attention kernel or after `wo` encodes \
                 no dispatch here, so it must not force a barrier around zero work"
            );
            assert_eq!(
                e.stack_only(),
                e.attn_logit_softcap.is_none(),
                "sinks and wo's bias are the stacks' alone"
            );
        }
    }
}
