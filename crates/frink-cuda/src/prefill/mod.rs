//! One dense prefill layer on the device, end to end.
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
//! matmuls is a kernel, and what comes back is the new hidden batch and
//! the batch's K and V rows for the host `KvCache`, which stays
//! authoritative -- there is no CUDA KV arena on this path yet, so a
//! prefix already in the cache is uploaded per layer for the attention
//! kernel to read.
//!
//! What it serves is exactly what the host batched body does for a
//! layer `Decoder::fused_prefill_dense_layer_eligible` admits; every
//! decision about WHICH layers is the decoder's, made through the same
//! predicates the Metal stack asks, so the two backends cannot admit
//! different sets. The refusals here are the shapes the kernels cannot
//! take (`head_dim > 256`, a GELU FFN, an odd rotary width), and each
//! is `CudaError::Unsupported` so the caller falls back to the host
//! body rather than computing something else.

pub mod enqueue;
pub mod kernels;

use crate::gpu::{resident_f32, shared_device, silu_mul_device, CudaError, ResidentF32};
use crate::mul_mm::{validate_shape, MulMmKind};
use crate::mul_mm_launch::enqueue_mul_mm;
use cudarc::driver::CudaSlice;
use enqueue::{
    enqueue_add_bias_rows, enqueue_add_rows, enqueue_causal_gqa_prefill, enqueue_rmsnorm_rows,
    enqueue_rope_rows, AttnArgs, RopeArgs,
};

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
    fn validate(&self, batch: usize) -> Result<(), CudaError> {
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
    pub q: MulMmWeights<'a>,
    pub k: MulMmWeights<'a>,
    pub v: MulMmWeights<'a>,
    pub o: MulMmWeights<'a>,
    pub gate: MulMmWeights<'a>,
    pub up: MulMmWeights<'a>,
    pub down: MulMmWeights<'a>,
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
    /// cache holds it). Empty when `start_pos == 0`.
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

fn upload(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    what: &str,
    x: &[f32],
) -> Result<CudaSlice<f32>, CudaError> {
    dev.htod_copy(x.to_vec())
        .map_err(|e| CudaError::Launch(format!("{what} upload: {e:?}")))
}

fn download<S: cudarc::driver::DevicePtr<f32>>(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    what: &str,
    d: &S,
) -> Result<Vec<f32>, CudaError> {
    dev.dtoh_sync_copy(d)
        .map_err(|e| CudaError::Launch(format!("{what} download: {e:?}")))
}

/// Runs one dense layer over `hidden` (`batch * hidden_dim`, row-major)
/// on the device: pre-norm, Q/K/V GEMMs, biases, QK norms, RoPE,
/// causal GQA over prefix + batch, `wo`, post-norm, residual, pre-norm,
/// gate/up GEMMs, SwiGLU, down GEMM, post-norm, residual. Three
/// downloads at the end (hidden, K rows, V rows) and no other sync.
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
    let mut out = launch_prefill_dense_stack(hidden, &[(layer, params)], batch)?;
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
/// across them: one upload at the start, one download of it at the
/// end, and two small downloads (K rows, V rows) per layer. This is
/// what the decoder calls; [`launch_prefill_dense_layer`] is the
/// one-layer case of it. Every layer is shape-checked before the first
/// touches the device, so a refusal leaves no partial state anywhere.
pub fn launch_prefill_dense_stack(
    hidden: &[f32],
    layers: &[(&PrefillDenseLayerCuda<'_>, &PrefillParams<'_>)],
    batch: usize,
) -> Result<PrefillStackOut, CudaError> {
    let Some((first, _)) = layers.first() else {
        return Err(CudaError::Unsupported(
            "prefill dense stack: no layers".to_string(),
        ));
    };
    let hidden_dim = first.attn_norm_w.len();
    if hidden.len() != batch * hidden_dim {
        return Err(CudaError::Unsupported(
            "prefill dense stack: hidden batch does not match the first layer's width".to_string(),
        ));
    }
    for (layer, params) in layers {
        check_layer_shapes(layer, params, batch, hidden_dim)?;
    }
    let dev = shared_device()?;
    let mut h = upload(&dev, "hidden", hidden)?;
    // Every layer is enqueued before anything is downloaded: the first
    // version synchronised twice per layer (its K and V rows), and the
    // GPU sat idle after each sync while the host pushed rows and
    // uploaded the next layer's small vectors -- a third of the pp512
    // step on an RTX 3090 with the kernels already at 0.17 s of the
    // 0.26 s. The per-layer K/V buffers stay alive here (4 MB a layer
    // on Llama-3.2-3B) until the one sync at the end.
    let mut pending = Vec::with_capacity(layers.len());
    for (layer, params) in layers {
        pending.push(run_layer_resident(&dev, &mut h, layer, params, batch)?);
    }
    let hidden = download(&dev, "hidden", &h)?;
    let mut kv_rows = Vec::with_capacity(layers.len());
    for LayerKv {
        k_all,
        v_all,
        n_prefix,
        held: _,
    } in pending
    {
        let k_rows = download(&dev, "K rows", &k_all.slice(n_prefix..))?;
        let v_rows = download(&dev, "V rows", &v_all.slice(n_prefix..))?;
        kv_rows.push((k_rows, v_rows));
    }
    Ok(PrefillStackOut { hidden, kv_rows })
}

/// What one enqueued layer leaves for the downloads at the end of the
/// stack: its K/V over prefix and batch, where the batch starts, and
/// the resident buffers its kernels still read.
struct LayerKv {
    k_all: CudaSlice<f32>,
    v_all: CudaSlice<f32>,
    n_prefix: usize,
    /// Kept alive, never read: dropped after the sync.
    #[allow(dead_code)]
    held: Vec<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
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
    let ffn_dim = layer.gate.rows;
    if layer.attn_norm_w.len() != hidden_dim
        || layer.ffn_norm_w.len() != hidden_dim
        || layer.q.rows != q_width
        || layer.k.rows != kv_width
        || layer.v.rows != kv_width
        || layer.o.rows != hidden_dim
        || layer.o.cols != q_width
        || layer.up.rows != ffn_dim
        || layer.down.rows != hidden_dim
        || layer.down.cols != ffn_dim
        || [layer.q, layer.k, layer.v, layer.gate, layer.up]
            .iter()
            .any(|m| m.cols != hidden_dim)
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

/// One layer over the resident hidden batch `h`, in place. Returns
/// the batch's K/V rows. Shapes were checked by `check_layer_shapes`.
fn run_layer_resident(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    h: &mut CudaSlice<f32>,
    layer: &PrefillDenseLayerCuda<'_>,
    params: &PrefillParams<'_>,
    batch: usize,
) -> Result<LayerKv, CudaError> {
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
    let ffn_dim = layer.gate.rows;
    let q_norm_shape = norm_rows(layer.extras.q_norm, q_width)?;
    let k_norm_shape = norm_rows(layer.extras.k_norm, kv_width)?;

    // Every resident buffer a kernel reads asynchronously is held here
    // until the stack's one sync: the GEMM weights, and the model's
    // small vectors, which are resident too (`resident_f32`) rather
    // than uploaded per layer per step.
    let mut held: Vec<std::sync::Arc<dyn std::any::Any + Send + Sync>> = Vec::with_capacity(16);
    let vec_of = |what: &str, v: &[f32]| -> Result<std::sync::Arc<ResidentF32>, CudaError> {
        resident_f32(dev, v).map_err(|e| CudaError::Launch(format!("{what}: {e}")))
    };

    let attn_norm_w = vec_of("attn_norm", layer.attn_norm_w)?;
    let ffn_norm_w = vec_of("ffn_norm", layer.ffn_norm_w)?;

    // --- attention ---
    let normed = enqueue_rmsnorm_rows(dev, h, &attn_norm_w.slice, batch, hidden_dim, rms_eps)?;
    let gemm = |m: &MulMmWeights<'_>, x: &CudaSlice<f32>, held: &mut Vec<_>| {
        let (out, w) = enqueue_mul_mm(dev, m.kind, m.data, x, m.rows, m.cols, batch, m.row_bytes)?;
        held.push(w as std::sync::Arc<dyn std::any::Any + Send + Sync>);
        Ok::<_, CudaError>(out)
    };
    let mut q = gemm(&layer.q, &normed, &mut held)?;
    let mut k = gemm(&layer.k, &normed, &mut held)?;
    let mut v = gemm(&layer.v, &normed, &mut held)?;
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
            let d_bias = vec_of("qkv bias", bias)?;
            enqueue_add_bias_rows(dev, x, &d_bias.slice, batch, width)?;
            held.push(d_bias);
        }
    }
    if let (Some(w), Some((n, per))) = (layer.extras.q_norm, q_norm_shape) {
        let d_w = vec_of("q_norm", w)?;
        q = enqueue_rmsnorm_rows(dev, &q, &d_w.slice, batch * per, n, rms_eps)?;
        held.push(d_w);
    }
    if let (Some(w), Some((n, per))) = (layer.extras.k_norm, k_norm_shape) {
        let d_w = vec_of("k_norm", w)?;
        k = enqueue_rmsnorm_rows(dev, &k, &d_w.slice, batch * per, n, rms_eps)?;
        held.push(d_w);
    }
    if let Some(rope) = &layer.rope {
        let d_ff = match rope.freq_factors {
            Some(ff) => Some(vec_of("freq_factors", ff)?),
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
        if let Some(d_ff) = d_ff {
            held.push(d_ff);
        }
    }

    // K/V over prefix + batch, in the cache's `[pos, kv_head, dim]`
    // layout: the batch's rows are already in it (row `b` of the
    // projection IS position `start_pos + b`), so the prefix is
    // prepended by a device copy rather than a host round trip.
    let (k_all, v_all) = if start_pos == 0 {
        (k, v)
    } else {
        let total = (start_pos + batch) * kv_width;
        let mut k_all = dev
            .alloc_zeros::<f32>(total)
            .map_err(|e| CudaError::Launch(format!("k_all alloc: {e:?}")))?;
        let mut v_all = dev
            .alloc_zeros::<f32>(total)
            .map_err(|e| CudaError::Launch(format!("v_all alloc: {e:?}")))?;
        let d_pk = upload(dev, "prefix K", prefix_k)?;
        let d_pv = upload(dev, "prefix V", prefix_v)?;
        let n_prefix = start_pos * kv_width;
        dev.dtod_copy(&d_pk, &mut k_all.slice_mut(..n_prefix))
            .map_err(|e| CudaError::Launch(format!("prefix K copy: {e:?}")))?;
        dev.dtod_copy(&d_pv, &mut v_all.slice_mut(..n_prefix))
            .map_err(|e| CudaError::Launch(format!("prefix V copy: {e:?}")))?;
        dev.dtod_copy(&k, &mut k_all.slice_mut(n_prefix..))
            .map_err(|e| CudaError::Launch(format!("batch K copy: {e:?}")))?;
        dev.dtod_copy(&v, &mut v_all.slice_mut(n_prefix..))
            .map_err(|e| CudaError::Launch(format!("batch V copy: {e:?}")))?;
        (k_all, v_all)
    };

    let attn = enqueue_causal_gqa_prefill(
        dev,
        &q,
        &k_all,
        &v_all,
        batch,
        &AttnArgs {
            n_heads,
            n_kv_heads,
            head_dim,
            start_pos,
            window,
            scale: attn_scale,
            softcap: attn_softcap,
        },
    )?;
    drop(q);
    let mut o = gemm(&layer.o, &attn, &mut held)?;
    drop(attn);
    if let Some(post) = layer.post_attn_norm {
        let d_w = vec_of("post_attn_norm", post)?;
        o = enqueue_rmsnorm_rows(dev, &o, &d_w.slice, batch, hidden_dim, rms_eps)?;
        held.push(d_w);
    }
    enqueue_add_rows(dev, h, &o, batch * hidden_dim)?;
    drop(o);

    // --- FFN ---
    let normed2 = enqueue_rmsnorm_rows(dev, h, &ffn_norm_w.slice, batch, hidden_dim, rms_eps)?;
    let gate = gemm(&layer.gate, &normed2, &mut held)?;
    let up = gemm(&layer.up, &normed2, &mut held)?;
    drop(normed2);
    let act = silu_mul_device(dev, &gate, &up, batch * ffn_dim)?;
    drop((gate, up));
    let mut down = gemm(&layer.down, &act, &mut held)?;
    drop(act);
    if let Some(post) = layer.post_ffn_norm {
        let d_w = vec_of("post_ffn_norm", post)?;
        down = enqueue_rmsnorm_rows(dev, &down, &d_w.slice, batch, hidden_dim, rms_eps)?;
        held.push(d_w);
    }
    enqueue_add_rows(dev, h, &down, batch * hidden_dim)?;
    drop(down);

    // The batch's rows of K/V are the tail of `k_all` / `v_all`; the
    // stack downloads them after every layer is enqueued. Nothing here
    // synchronises.
    held.push(attn_norm_w);
    held.push(ffn_norm_w);
    Ok(LayerKv {
        k_all,
        v_all,
        n_prefix: start_pos * kv_width,
        held,
    })
}

#[cfg(test)]
mod tests;
