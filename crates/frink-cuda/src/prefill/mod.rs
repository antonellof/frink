//! A run of dense layers on the device, end to end: a prefill batch or
//! one decode token.
//!
//! Before this module a batched prefill ran a dense layer as seven
//! `launch_mul_mm` calls, each a synchronous round trip (a pageable
//! upload of the activation, the launch, a synchronous download of the
//! result) with the norms, RoPE, attention, SwiGLU and residual adds
//! on the host between them: 111 MB over PCIe per Llama-3.2-3B layer at
//! pp512, 3.1 GB and 196 syncs per step, and a GPU idle two thirds of
//! the time (#259, `docs/plans/cpu-cuda-parity.md` step 2b). This is
//! the Metal shape (`frink_metal::attn::launch_prefill_dense_layer`)
//! for CUDA: the hidden batch goes up once, every op between the
//! matmuls is a kernel, and what comes back, in ONE download at the
//! end, is the new hidden batch and every layer's K and V rows for the
//! host `KvCache`, which stays authoritative.
//!
//! Decode is the same stack at `batch == 1` (#133), and only two
//! things differ, each a parameter rather than a second body (ADR
//! 0004): the projections multiply by the matvec instead of the GEMM
//! ([`proj::ProjWeights::uses_matvec`]), and the attention reads the
//! sequence's device mirror of its history ([`StackKv::Mirrors`],
//! [`kv_mirror`]) instead of uploading the host prefix, through the
//! one-row split-key kernel ([`crate::decode_attn`]). Before this the
//! CUDA decode token made five synchronous round trips per layer, with
//! the norms, RoPE, the attention and the residual adds on the host.
//!
//! What it serves is exactly what the host bodies do for a layer
//! `Decoder::fused_prefill_dense_layer_eligible` admits; every decision
//! about WHICH layers is the decoder's, made through the same
//! predicates the Metal stack asks, so the two backends cannot admit
//! different sets. The refusals here are the shapes the kernels cannot
//! take (`head_dim > 256`, a GELU FFN, an odd rotary width, a one-row
//! step on a kind with no matvec), and each is `CudaError::Unsupported`
//! so the caller falls back to the host body rather than computing
//! something else.

pub mod enqueue;
pub mod kernels;
pub mod kv_mirror;
pub mod proj;

use crate::gpu::{resident_cuda_f32, shared_device, silu_mul_device, CudaError};
use crate::mul_mm::{validate_shape, MulMmKind};
use crate::xfer;
use cudarc::driver::{CudaDevice, CudaSlice};
use enqueue::{
    enqueue_add_bias_rows, enqueue_add_rows, enqueue_causal_gqa_prefill, enqueue_decode_attn,
    enqueue_rmsnorm_rows, enqueue_rope_rows, AttnArgs, RopeArgs,
};
pub use kv_mirror::CudaKvMirror;
pub use proj::ProjWeights;
use std::sync::Arc;

/// One quantized projection as the GEMM takes it: the GGUF bytes, the
/// kernel row for their kind, and the shape. `WeightMatrix::
/// cuda_mul_mm_view` is the one constructor.
#[derive(Clone, Copy)]
pub struct MulMmWeights<'a> {
    pub kind: &'static MulMmKind,
    pub data: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
}

impl MulMmWeights<'_> {
    pub(crate) fn validate(&self, batch: usize) -> Result<(), CudaError> {
        validate_shape(
            self.kind,
            self.data.len(),
            batch * self.cols,
            self.rows,
            self.cols,
            batch,
            self.row_bytes,
        )
        .map_err(|e| CudaError::Unsupported(e.to_string()))
    }
}

/// Pairing convention of the rotation (`frink_models::config::
/// RopeLayout`): `Norm` rotates adjacent pairs, `Neox` split halves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeLayoutCuda {
    Norm,
    Neox,
}

/// This layer's rotation, or `None` for a layer llama.cpp does not
/// rotate (`frink_models::rope_layers`): the kernel then does not run.
#[derive(Clone, Copy)]
pub struct LayerRopeCuda<'a> {
    pub theta: f32,
    pub freq_factors: Option<&'a [f32]>,
    /// Channels of each head that rotate; `head_dim` for a full rotation.
    pub rot_dim: usize,
    pub layout: RopeLayoutCuda,
    /// YaRN's magnitude term on the rotated channels of Q and K
    /// (`ModelConfig::rope_attn_factor`); `1.0` when there is none.
    pub mscale: f32,
}

/// The optional per-layer attention ops the host body applies between
/// the QKV projections and RoPE: the biases (Qwen2), the QK norms
/// (Qwen3 / Gemma-3 per head, OLMoE whole vector; the weight's length
/// says which). Built by `Decoder::fused_attn_extras`, the ONE
/// exhaustive destructure of `AttnWeights`, so a field that arrives
/// here has been accounted for there.
#[derive(Clone, Copy, Default)]
pub struct AttnExtrasCuda<'a> {
    pub q_bias: Option<&'a [f32]>,
    pub k_bias: Option<&'a [f32]>,
    pub v_bias: Option<&'a [f32]>,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
}

/// One dense layer as the launch takes it.
pub struct PrefillDenseLayerCuda<'a> {
    pub attn_norm_w: &'a [f32],
    pub ffn_norm_w: &'a [f32],
    pub q: ProjWeights<'a>,
    pub k: ProjWeights<'a>,
    pub v: ProjWeights<'a>,
    pub o: ProjWeights<'a>,
    pub gate: ProjWeights<'a>,
    pub up: ProjWeights<'a>,
    pub down: ProjWeights<'a>,
    /// Gemma-2's norms on each branch output before its residual add.
    pub post_attn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
    pub extras: AttnExtrasCuda<'a>,
    pub rope: Option<LayerRopeCuda<'a>>,
}

/// The per-model facts the layer launch takes beside the weights.
pub struct PrefillParams<'a> {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    /// `1 / sqrt(head_dim)` unless the architecture overrides it; the
    /// decoder passes the number it uses on the host.
    pub attn_scale: f32,
    pub attn_softcap: Option<f32>,
    /// This layer's sliding window, `None` for full attention.
    pub window: Option<usize>,
    /// K and V rows already in the host cache for this layer, laid out
    /// `[start_pos, n_kv_heads, head_dim]` each (post-RoPE K, as the
    /// cache holds it). Empty when `start_pos == 0`. Uploaded for the
    /// attention under [`StackKv::UploadPrefix`]; under
    /// [`StackKv::Mirrors`] only compared with the mirror, and uploaded
    /// only when the mirror does not hold them.
    pub prefix_k: &'a [f32],
    pub prefix_v: &'a [f32],
    pub start_pos: usize,
}

/// What comes back: the new hidden batch, and the batch's K (post-RoPE)
/// and V rows in the host cache's layout, one row per position.
pub struct PrefillLayerOut {
    pub hidden: Vec<f32>,
    pub k_rows: Vec<f32>,
    pub v_rows: Vec<f32>,
}

/// What the first layer of a stack reads.
#[derive(Clone, Copy)]
pub enum StackInput<'a> {
    /// The hidden rows, `batch * hidden_dim`, computed on the host and
    /// uploaded once.
    Hidden(&'a [f32]),
    /// The rows `ids` of an embedding table, gathered on the device
    /// ([`crate::get_rows`]) instead of dequantized on the host and
    /// uploaded: one id per batch row. Only for a model whose embedding
    /// row IS the first hidden state (no scale, norm or position
    /// table); that is the caller's to decide.
    Gather {
        table: MulMmWeights<'a>,
        ids: &'a [u32],
    },
}

impl StackInput<'_> {
    /// The shape agreement with the first layer, before any device work.
    fn check(&self, batch: usize, hidden_dim: usize) -> Result<(), CudaError> {
        let rows = match self {
            StackInput::Hidden(h) if h.len() == batch * hidden_dim => batch,
            StackInput::Gather { table, ids } if table.cols == hidden_dim => {
                crate::get_rows::validate(
                    table.kind,
                    table.data.len(),
                    table.rows,
                    ids,
                    table.cols,
                    table.row_bytes,
                )
                .map_err(|e| CudaError::Unsupported(e.to_string()))?;
                ids.len()
            }
            _ => usize::MAX,
        };
        if batch == 0 || rows != batch {
            return Err(CudaError::Unsupported(
                "prefill dense stack: the input does not match the first layer's width and batch"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Where each layer's attention reads its history from.
pub enum StackKv<'m, 'c> {
    /// Upload each layer's host prefix ([`PrefillParams::prefix_k`])
    /// for its attention: right for a prefill step, which reads the
    /// prefix once.
    UploadPrefix,
    /// Each layer's own device mirror, one slot per layer of the run,
    /// in order: right for a decode token, which would otherwise upload
    /// the whole history per layer. A slot that is empty or stale is
    /// rebuilt from the host prefix first (see [`kv_mirror`] for the
    /// check), and every mirror is appended to on the device.
    Mirrors(&'m mut [&'c mut Option<CudaKvMirror>]),
}

/// Whether a decode token may take the resident stack. On unless
/// `FRINK_CUDA_DECODE=host`, which keeps every decode token on the
/// per-op path (five synchronous round trips per layer, host norms,
/// RoPE and attention) for the A/B. Read once (`docs/CONFIG.md`).
pub fn resident_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FRINK_CUDA_DECODE").as_deref(),
            Ok("host" | "off" | "0")
        )
    })
}

/// Runs one dense layer over `hidden` (`batch * hidden_dim`, row-major)
/// on the device: pre-norm, Q/K/V projections, biases, QK norms, RoPE,
/// causal GQA over prefix + batch, `wo`, post-norm, residual, pre-norm,
/// gate/up projections, SwiGLU, down projection, post-norm, residual.
/// One download at the end and no other sync.
///
/// Refuses (`Unsupported`) rather than approximates: a GEMM shape the
/// kernel cannot tile, a `head_dim` above 256, a rotary width that is
/// odd or wider than the head. The activation is SwiGLU: the decoder
/// only builds a layer for a model whose `fused_kernel_gelu_flag` is
/// `Some(false)`.
pub fn launch_prefill_dense_layer(
    hidden: &[f32],
    layer: &PrefillDenseLayerCuda<'_>,
    params: &PrefillParams<'_>,
    batch: usize,
) -> Result<PrefillLayerOut, CudaError> {
    let mut out = launch_prefill_dense_stack(
        StackInput::Hidden(hidden),
        &[(layer, params)],
        batch,
        StackKv::UploadPrefix,
    )?;
    let (k_rows, v_rows) = out.kv_rows.pop().expect("one layer in, one layer out");
    Ok(PrefillLayerOut {
        hidden: out.hidden,
        k_rows,
        v_rows,
    })
}

/// What a stack returns: the hidden batch after the last layer, and
/// each layer's K (post-RoPE) and V rows in the host cache's layout.
pub struct PrefillStackOut {
    pub hidden: Vec<f32>,
    pub kv_rows: Vec<(Vec<f32>, Vec<f32>)>,
}

/// A run of consecutive dense layers with the hidden batch resident
/// across them: one upload at the start and ONE download at the end,
/// which carries the hidden batch and every layer's K and V rows
/// together. This is what the decoder calls, for a prefill batch and
/// for a decode token alike; [`launch_prefill_dense_layer`] is the
/// one-layer case of it. Every layer is shape-checked before the first
/// touches the device, so a refusal leaves no partial state anywhere
/// (a mirror rebuilt from the host before a later failure holds the
/// host's rows, which is a valid state).
pub fn launch_prefill_dense_stack(
    input: StackInput<'_>,
    layers: &[(&PrefillDenseLayerCuda<'_>, &PrefillParams<'_>)],
    batch: usize,
    mut kv: StackKv<'_, '_>,
) -> Result<PrefillStackOut, CudaError> {
    let Some((first, first_params)) = layers.first() else {
        return Err(CudaError::Unsupported(
            "prefill dense stack: no layers".to_string(),
        ));
    };
    let hidden_dim = first.attn_norm_w.len();
    input.check(batch, hidden_dim)?;
    for (layer, params) in layers {
        check_layer_shapes(layer, params, batch, hidden_dim)?;
        if params.start_pos != first_params.start_pos {
            return Err(CudaError::Unsupported(
                "prefill dense stack: layers of one run at different positions".to_string(),
            ));
        }
    }
    let start_pos = first_params.start_pos;
    let dev = shared_device()?;

    // The mirrors, validated TOGETHER: layer 0's newest K row depends
    // only on (token, position), so two histories can agree there and
    // nowhere else. One stale layer rebuilds all of them.
    if let StackKv::Mirrors(slots) = &mut kv {
        if slots.len() != layers.len() {
            return Err(CudaError::Unsupported(format!(
                "resident stack: {} KV mirrors for {} layers",
                slots.len(),
                layers.len()
            )));
        }
        let need = start_pos + batch;
        let fresh = slots.iter().zip(layers).all(|(slot, (_, p))| {
            slot.as_ref()
                .is_some_and(|m| m.holds(p.prefix_k, p.prefix_v, p.n_kv_heads * p.head_dim))
        });
        for (slot, (_, p)) in slots.iter_mut().zip(layers) {
            let width = p.n_kv_heads * p.head_dim;
            match slot.as_mut() {
                Some(m) if fresh => m.reserve(&dev, need)?,
                _ => {
                    **slot = Some(CudaKvMirror::from_host(
                        &dev, p.prefix_k, p.prefix_v, width, need,
                    )?)
                }
            }
        }
    }

    // ONE buffer for everything the host gets back: the hidden batch,
    // then per layer the batch's K rows and V rows.
    let mut kv_offsets = Vec::with_capacity(layers.len());
    let mut staged_len = batch * hidden_dim;
    for (_, p) in layers {
        let n = batch * p.n_kv_heads * p.head_dim;
        kv_offsets.push((staged_len, n));
        staged_len += 2 * n;
    }
    let mut staged = dev
        .alloc_zeros::<f32>(staged_len)
        .map_err(|e| CudaError::Launch(format!("stack staging alloc: {e:?}")))?;

    let mut held = Held::default();
    let mut h = match input {
        StackInput::Hidden(hidden) => xfer::htod(&dev, "hidden upload", hidden)?,
        StackInput::Gather { table, ids } => {
            let d_ids = xfer::htod(&dev, "gather ids upload", ids)?;
            let (h, w) = crate::get_rows_launch::enqueue_get_rows(
                &dev,
                table.kind,
                table.data,
                &d_ids,
                table.cols,
                table.row_bytes,
            )?;
            held.weights.push(w);
            h
        }
    };
    for (i, (layer, params)) in layers.iter().enumerate() {
        let mirror = match &mut kv {
            StackKv::UploadPrefix => None,
            StackKv::Mirrors(slots) => Some(
                slots[i]
                    .as_mut()
                    .expect("every mirror slot was filled above"),
            ),
        };
        let step = LayerStep {
            batch,
            mirror,
            stage_at: kv_offsets[i],
        };
        run_layer_resident(&dev, &mut h, layer, params, step, &mut staged, &mut held)?;
    }
    xfer::dtod(
        &dev,
        "hidden stage",
        &h,
        &mut staged.slice_mut(..batch * hidden_dim),
    )?;
    // The stack's one host wait: everything above was enqueued.
    let mut all = xfer::dtoh(&dev, "stack download", &staged)?;
    drop(held);

    let kv_rows: Vec<(Vec<f32>, Vec<f32>)> = kv_offsets
        .iter()
        .map(|&(at, n)| (all[at..at + n].to_vec(), all[at + n..at + 2 * n].to_vec()))
        .collect();
    if let StackKv::Mirrors(slots) = &mut kv {
        for ((slot, (_, p)), (k_rows, _)) in slots.iter_mut().zip(layers).zip(&kv_rows) {
            let width = p.n_kv_heads * p.head_dim;
            let m = slot.as_mut().expect("every mirror slot was filled above");
            m.commit(start_pos + batch, &k_rows[k_rows.len() - width..]);
        }
    }
    all.truncate(batch * hidden_dim);
    Ok(PrefillStackOut {
        hidden: all,
        kv_rows,
    })
}

/// The host-side shape agreement, before any device work: a mismatch
/// is a named error and never a kernel reading past a buffer.
fn check_layer_shapes(
    layer: &PrefillDenseLayerCuda<'_>,
    params: &PrefillParams<'_>,
    batch: usize,
    hidden_dim: usize,
) -> Result<(), CudaError> {
    let PrefillParams {
        n_heads,
        n_kv_heads,
        head_dim,
        prefix_k,
        prefix_v,
        start_pos,
        ..
    } = *params;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let ffn_dim = layer.gate.rows();
    if layer.attn_norm_w.len() != hidden_dim
        || layer.ffn_norm_w.len() != hidden_dim
        || layer.q.rows() != q_width
        || layer.k.rows() != kv_width
        || layer.v.rows() != kv_width
        || layer.o.rows() != hidden_dim
        || layer.o.cols() != q_width
        || layer.up.rows() != ffn_dim
        || layer.down.rows() != hidden_dim
        || layer.down.cols() != ffn_dim
        || [layer.q, layer.k, layer.v, layer.gate, layer.up]
            .iter()
            .any(|m| m.cols() != hidden_dim)
        || prefix_k.len() != start_pos * kv_width
        || prefix_v.len() != start_pos * kv_width
    {
        return Err(CudaError::Unsupported(
            "prefill dense layer: projection shapes do not agree with the layer geometry"
                .to_string(),
        ));
    }
    for m in [
        &layer.q,
        &layer.k,
        &layer.v,
        &layer.o,
        &layer.gate,
        &layer.up,
        &layer.down,
    ] {
        m.validate(batch)?;
    }
    if batch == 1 && !crate::decode_attn::head_dim_ok(head_dim) {
        return Err(CudaError::Unsupported(format!(
            "resident decode: head_dim {head_dim} (the one-row attention takes a multiple of 4 \
             up to 256)"
        )));
    }
    if let Some(rope) = &layer.rope {
        if rope.rot_dim == 0 || !rope.rot_dim.is_multiple_of(2) || rope.rot_dim > head_dim {
            return Err(CudaError::Unsupported(format!(
                "prefill dense layer: rotary width {} on a head of {head_dim}",
                rope.rot_dim
            )));
        }
        if let Some(ff) = rope.freq_factors {
            if ff.len() != rope.rot_dim / 2 {
                return Err(CudaError::Unsupported(format!(
                    "prefill dense layer: {} freq_factors for {} rotation bands",
                    ff.len(),
                    rope.rot_dim / 2
                )));
            }
        }
    }
    norm_rows(layer.extras.q_norm, q_width)?;
    norm_rows(layer.extras.k_norm, kv_width)?;
    for (bias, width) in [
        (layer.extras.q_bias, q_width),
        (layer.extras.k_bias, kv_width),
        (layer.extras.v_bias, kv_width),
    ] {
        if let Some(bias) = bias {
            if bias.len() != width {
                return Err(CudaError::Unsupported(format!(
                    "prefill dense layer: a bias of {} on a projection of {width}",
                    bias.len()
                )));
            }
        }
    }
    Ok(())
}

/// A QK norm weight as `(n, rows per position)`: per head (`head_dim`
/// long) or whole vector, which is how the loader's length rule reads
/// them.
fn norm_rows(w: Option<&[f32]>, width: usize) -> Result<Option<(usize, usize)>, CudaError> {
    match w {
        None => Ok(None),
        Some(w) if w.is_empty() || !width.is_multiple_of(w.len()) => {
            Err(CudaError::Unsupported(format!(
                "prefill dense layer: a QK norm of {} over a width of {width}",
                w.len()
            )))
        }
        Some(w) => Ok(Some((w.len(), width / w.len()))),
    }
}

/// Every resident buffer a launch reads, held until the stack's one
/// download has synchronised the stream: the kernels read them
/// asynchronously, and a buffer must not be released under a kernel
/// that is still reading it.
#[derive(Default)]
struct Held {
    weights: Vec<Arc<crate::gpu::ResidentCudaWeights>>,
    vectors: Vec<Arc<crate::resident::ResidentCuda<f32>>>,
}

impl Held {
    /// A long-lived f32 vector (norm weight, bias, RoPE divisors) on
    /// the device, held: uploaded on first use, never per call.
    fn vector(
        &mut self,
        dev: &Arc<CudaDevice>,
        v: &[f32],
    ) -> Result<Arc<crate::resident::ResidentCuda<f32>>, CudaError> {
        let r = resident_cuda_f32(dev, v)?;
        self.vectors.push(r.clone());
        Ok(r)
    }

    /// One projection on the device, its weights held.
    fn proj(
        &mut self,
        dev: &Arc<CudaDevice>,
        m: &ProjWeights<'_>,
        x: &CudaSlice<f32>,
        batch: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let (out, w) = m.enqueue(dev, x, batch)?;
        self.weights.push(w);
        Ok(out)
    }
}

/// What one layer of a run varies by beside its weights: the batch,
/// where its history lives, and where its K/V rows are staged.
struct LayerStep<'m> {
    batch: usize,
    /// The sequence's device history under [`StackKv::Mirrors`].
    mirror: Option<&'m mut CudaKvMirror>,
    /// `(offset, len)` of this layer's K rows in the staging buffer;
    /// its V rows follow them.
    stage_at: (usize, usize),
}

/// One layer over the resident hidden batch `h`, in place. Stages the
/// batch's K/V rows into `staged`. Shapes were checked by
/// `check_layer_shapes`.
fn run_layer_resident(
    dev: &Arc<CudaDevice>,
    h: &mut CudaSlice<f32>,
    layer: &PrefillDenseLayerCuda<'_>,
    params: &PrefillParams<'_>,
    step: LayerStep<'_>,
    staged: &mut CudaSlice<f32>,
    held: &mut Held,
) -> Result<(), CudaError> {
    let LayerStep {
        batch,
        mirror,
        stage_at: (at, n_kv),
    } = step;
    let hidden_dim = layer.attn_norm_w.len();
    let PrefillParams {
        n_heads,
        n_kv_heads,
        head_dim,
        rms_eps,
        attn_scale,
        attn_softcap,
        window,
        prefix_k,
        prefix_v,
        start_pos,
    } = *params;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let ffn_dim = layer.gate.rows();
    let q_norm_shape = norm_rows(layer.extras.q_norm, q_width)?;
    let k_norm_shape = norm_rows(layer.extras.k_norm, kv_width)?;
    debug_assert_eq!(n_kv, batch * kv_width);

    let attn_norm_w = held.vector(dev, layer.attn_norm_w)?;
    let ffn_norm_w = held.vector(dev, layer.ffn_norm_w)?;

    // --- attention ---
    let normed = enqueue_rmsnorm_rows(dev, h, &attn_norm_w.slice, batch, hidden_dim, rms_eps)?;
    let mut q = held.proj(dev, &layer.q, &normed, batch)?;
    let mut k = held.proj(dev, &layer.k, &normed, batch)?;
    let mut v = held.proj(dev, &layer.v, &normed, batch)?;
    drop(normed);

    for (x, bias, width) in [
        (&mut q, layer.extras.q_bias, q_width),
        (&mut k, layer.extras.k_bias, kv_width),
        (&mut v, layer.extras.v_bias, kv_width),
    ] {
        if let Some(bias) = bias {
            if bias.len() != width {
                return Err(CudaError::Unsupported(format!(
                    "prefill dense layer: a bias of {} on a projection of {width}",
                    bias.len()
                )));
            }
            let d_bias = held.vector(dev, bias)?;
            enqueue_add_bias_rows(dev, x, &d_bias.slice, batch, width)?;
        }
    }
    if let (Some(w), Some((n, per))) = (layer.extras.q_norm, q_norm_shape) {
        let d_w = held.vector(dev, w)?;
        q = enqueue_rmsnorm_rows(dev, &q, &d_w.slice, batch * per, n, rms_eps)?;
    }
    if let (Some(w), Some((n, per))) = (layer.extras.k_norm, k_norm_shape) {
        let d_w = held.vector(dev, w)?;
        k = enqueue_rmsnorm_rows(dev, &k, &d_w.slice, batch * per, n, rms_eps)?;
    }
    if let Some(rope) = &layer.rope {
        let d_ff = match rope.freq_factors {
            Some(ff) => Some(held.vector(dev, ff)?),
            None => None,
        };
        let args = RopeArgs {
            theta: rope.theta,
            freq_factors: d_ff.as_ref().map(|r| &r.slice),
            rot_dim: rope.rot_dim,
            neox: rope.layout == RopeLayoutCuda::Neox,
            mscale: rope.mscale,
        };
        enqueue_rope_rows(dev, &mut q, batch, n_heads, head_dim, start_pos, &args)?;
        enqueue_rope_rows(dev, &mut k, batch, n_kv_heads, head_dim, start_pos, &args)?;
    }

    // The batch's K (post-RoPE) and V rows are final here: stage them
    // for the stack's one download.
    xfer::dtod(
        dev,
        "K rows stage",
        &k,
        &mut staged.slice_mut(at..at + n_kv),
    )?;
    xfer::dtod(
        dev,
        "V rows stage",
        &v,
        &mut staged.slice_mut(at + n_kv..at + 2 * n_kv),
    )?;

    // K/V over prefix + batch, in the cache's `[pos, kv_head, dim]`
    // layout: the batch's rows are already in it (row `b` of the
    // projection IS position `start_pos + b`).
    let n_prefix = start_pos * kv_width;
    let total = n_prefix + batch * kv_width;
    let uploaded;
    let (k_all, v_all): (&CudaSlice<f32>, &CudaSlice<f32>) = match mirror {
        // The sequence's own device history: append on the device.
        Some(m) => {
            debug_assert!(m.rows() == start_pos && m.cap_rows() >= start_pos + batch);
            xfer::dtod(
                dev,
                "K mirror append",
                &k,
                &mut m.k.slice_mut(n_prefix..total),
            )?;
            xfer::dtod(
                dev,
                "V mirror append",
                &v,
                &mut m.v.slice_mut(n_prefix..total),
            )?;
            (&m.k, &m.v)
        }
        None if start_pos == 0 => (&k, &v),
        // The prefix is uploaded and prepended by a device copy.
        None => {
            let mut k_all = dev
                .alloc_zeros::<f32>(total)
                .map_err(|e| CudaError::Launch(format!("k_all alloc: {e:?}")))?;
            let mut v_all = dev
                .alloc_zeros::<f32>(total)
                .map_err(|e| CudaError::Launch(format!("v_all alloc: {e:?}")))?;
            let d_pk = xfer::htod(dev, "prefix K upload", prefix_k)?;
            let d_pv = xfer::htod(dev, "prefix V upload", prefix_v)?;
            xfer::dtod(
                dev,
                "prefix K copy",
                &d_pk,
                &mut k_all.slice_mut(..n_prefix),
            )?;
            xfer::dtod(
                dev,
                "prefix V copy",
                &d_pv,
                &mut v_all.slice_mut(..n_prefix),
            )?;
            xfer::dtod(dev, "batch K copy", &k, &mut k_all.slice_mut(n_prefix..))?;
            xfer::dtod(dev, "batch V copy", &v, &mut v_all.slice_mut(n_prefix..))?;
            uploaded = (k_all, v_all);
            (&uploaded.0, &uploaded.1)
        }
    };

    let args = AttnArgs {
        n_heads,
        n_kv_heads,
        head_dim,
        start_pos,
        window,
        scale: attn_scale,
        softcap: attn_softcap,
    };
    // One query row is the decode kernel's shape, split across the
    // keys; a batch is the prefill kernel's, one warp per row.
    let attn = if batch == 1 {
        enqueue_decode_attn(dev, &q, k_all, v_all, &args)?
    } else {
        enqueue_causal_gqa_prefill(dev, &q, k_all, v_all, batch, &args)?
    };
    drop(q);
    let mut o = held.proj(dev, &layer.o, &attn, batch)?;
    drop(attn);
    if let Some(post) = layer.post_attn_norm {
        let d_w = held.vector(dev, post)?;
        o = enqueue_rmsnorm_rows(dev, &o, &d_w.slice, batch, hidden_dim, rms_eps)?;
    }
    enqueue_add_rows(dev, h, &o, batch * hidden_dim)?;
    drop(o);

    // --- FFN ---
    let normed2 = enqueue_rmsnorm_rows(dev, h, &ffn_norm_w.slice, batch, hidden_dim, rms_eps)?;
    let gate = held.proj(dev, &layer.gate, &normed2, batch)?;
    let up = held.proj(dev, &layer.up, &normed2, batch)?;
    drop(normed2);
    let act = silu_mul_device(dev, &gate, &up, batch * ffn_dim)?;
    drop((gate, up));
    let mut down = held.proj(dev, &layer.down, &act, batch)?;
    drop(act);
    if let Some(post) = layer.post_ffn_norm {
        let d_w = held.vector(dev, post)?;
        down = enqueue_rmsnorm_rows(dev, &down, &d_w.slice, batch, hidden_dim, rms_eps)?;
    }
    enqueue_add_rows(dev, h, &down, batch * hidden_dim)?;
    drop(down);
    Ok(())
}

#[cfg(test)]
mod gather_tests;
#[cfg(test)]
mod tests;
