//! One layer of a resident stack over the hidden rows on the device,
//! moved out of `prefill/mod.rs` (ADR 0008).

use super::enqueue::{
    enqueue_add_bias_rows, enqueue_add_rows, enqueue_causal_gqa_prefill, enqueue_decode_attn,
    enqueue_kv_append, enqueue_rmsnorm_rows, enqueue_rope_rows, AttnArgs, RopeArgs, StepPos,
};
use super::held::Held;
use super::shapes::norm_rows;
use super::{
    moe, CudaKvMirror, ExpertProjCuda, PrefillDenseLayerCuda, PrefillParams, RopeLayoutCuda,
    RouteFn, StackFfn,
};
use crate::gpu::{glu_mul_device, CudaError, GluCuda, MatvecInput};
use crate::xfer;
use cudarc::driver::{CudaDevice, CudaSlice, CudaViewMut};
use std::sync::Arc;

/// What one layer of a run varies by beside its weights: the batch,
/// where its history lives, and where its K/V rows are staged.
pub(super) struct LayerStep<'m, 'r, 'e> {
    pub(super) batch: usize,
    /// Where a decode token's device routing writes its expert ids.
    pub(super) ids: Option<CudaViewMut<'m, i32>>,
    /// This layer's index in the run, which the routing is asked by.
    pub(super) index: usize,
    /// The caller's routing, for an MoE layer.
    pub(super) route: Option<&'m mut RouteFn<'r, 'e>>,
    /// The sequence's device history under [`super::StackKv::Mirrors`].
    pub(super) mirror: Option<&'m mut CudaKvMirror>,
    /// `(offset, len)` of this layer's K rows in the staging buffer;
    /// its V rows follow them.
    pub(super) stage_at: (usize, usize),
    /// The device word holding the step's position, for a decode token
    /// captured into a graph (`super::decode_graph`); `None` takes the
    /// position from the layer's params, as a step enqueued as it runs.
    pub(super) pos_word: Option<&'m CudaSlice<i32>>,
}

/// One layer over the resident hidden batch `h`, in place, answering
/// whether its experts were routed on the device. Stages the batch's
/// K/V rows into `staged`. Shapes were checked by
/// `check_layer_shapes`.
pub(super) fn run_layer_resident<'e>(
    dev: &Arc<CudaDevice>,
    h: &mut CudaSlice<f32>,
    layer: &PrefillDenseLayerCuda<'_>,
    params: &PrefillParams<'_>,
    step: LayerStep<'_, '_, 'e>,
    staged: &mut CudaSlice<f32>,
    held: &mut Held,
) -> Result<bool, CudaError> {
    let LayerStep {
        batch,
        ids,
        index,
        mirror,
        stage_at: (at, n_kv),
        route,
        pos_word,
    } = step;
    let mut routed_on_device = false;
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
    let q_norm_shape = norm_rows(layer.extras.q_norm, q_width)?;
    let k_norm_shape = norm_rows(layer.extras.k_norm, kv_width)?;
    debug_assert_eq!(n_kv, batch * kv_width);
    // A device position reads the row a replay writes, and serves only
    // what a graph captures: one row over the sequence's mirror.
    let pos = match (pos_word, mirror.as_deref()) {
        (None, _) => StepPos::Host(start_pos),
        (Some(word), Some(m)) if batch == 1 => StepPos::Device {
            pos: word,
            cap_rows: m.cap_rows(),
        },
        _ => {
            return Err(CudaError::Unsupported(
                "resident stack: a device position outside a one-row step over a KV mirror"
                    .to_string(),
            ))
        }
    };

    let attn_norm_w = held.vector(dev, layer.attn_norm_w)?;
    let ffn_norm_w = held.vector(dev, layer.ffn_norm_w)?;

    // --- attention ---
    let normed = enqueue_rmsnorm_rows(dev, h, &attn_norm_w.slice, batch, hidden_dim, rms_eps)?;
    // Q, K and V read one input: under dp4a, one q8_1 copy for all three.
    let attn_x = MatvecInput::new(&normed);
    let mut q = held.proj(dev, &layer.q, &attn_x, batch)?;
    let mut k = held.proj(dev, &layer.k, &attn_x, batch)?;
    let mut v = held.proj(dev, &layer.v, &attn_x, batch)?;
    drop(attn_x);
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
        enqueue_rope_rows(dev, &mut q, batch, n_heads, head_dim, pos, &args)?;
        enqueue_rope_rows(dev, &mut k, batch, n_kv_heads, head_dim, pos, &args)?;
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
        // The sequence's own device history: append on the device, by
        // a copy at the row's offset, or for a captured token by a
        // kernel at the row its replay writes.
        Some(m) => {
            debug_assert!(m.rows() == start_pos && m.cap_rows() >= start_pos + batch);
            match pos {
                StepPos::Host(_) => {
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
                }
                StepPos::Device { pos, cap_rows } => {
                    enqueue_kv_append(dev, &k, &v, &mut m.k, &mut m.v, kv_width, cap_rows, pos)?
                }
            }
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
    let sinks = match layer.extras.sinks {
        Some(s) => Some(held.vector(dev, s)?),
        None => None,
    };
    let sinks = sinks.as_ref().map(|s| &s.slice);
    let attn = if batch == 1 {
        enqueue_decode_attn(dev, &q, k_all, v_all, &args, sinks, pos)?
    } else {
        enqueue_causal_gqa_prefill(dev, &q, k_all, v_all, batch, &args, sinks)?
    };
    drop(q);
    let mut o = held.proj(dev, &layer.o, &MatvecInput::new(&attn), batch)?;
    drop(attn);
    // `wo`'s bias, on the projection before any norm of it, where
    // `build_attn` adds `wo_b`.
    if let Some(b) = layer.extras.o_bias {
        let d_b = held.vector(dev, b)?;
        enqueue_add_bias_rows(dev, &mut o, &d_b.slice, batch, hidden_dim)?;
    }
    if let Some(post) = layer.post_attn_norm {
        let d_w = held.vector(dev, post)?;
        o = enqueue_rmsnorm_rows(dev, &o, &d_w.slice, batch, hidden_dim, rms_eps)?;
    }
    enqueue_add_rows(dev, h, &o, batch * hidden_dim)?;
    drop(o);

    // --- FFN ---
    let normed2 = enqueue_rmsnorm_rows(dev, h, &ffn_norm_w.slice, batch, hidden_dim, rms_eps)?;
    // Gate and up (an MoE layer's router and experts too) read one input.
    let ffn_x = MatvecInput::new(&normed2);
    let mut down = match layer.ffn {
        StackFfn::Dense(ex) => {
            let ffn_dim = ex.gate.rows();
            run_expert(dev, held, &ex, layer.glu, &ffn_x, batch, ffn_dim)?
        }
        StackFfn::Moe {
            router,
            ffn_dim,
            device,
        } => {
            let route = route.ok_or_else(|| {
                CudaError::Unsupported("resident stack: an MoE layer with no routing".to_string())
            })?;
            let device = device.zip(ids);
            let (out, on_device) = moe::run_routed(
                dev, held, router, layer.glu, &ffn_x, batch, hidden_dim, ffn_dim, index, route,
                device,
            )?;
            routed_on_device = on_device;
            out
        }
    };
    drop(ffn_x);
    drop(normed2);
    if let Some(post) = layer.post_ffn_norm {
        let d_w = held.vector(dev, post)?;
        down = enqueue_rmsnorm_rows(dev, &down, &d_w.slice, batch, hidden_dim, rms_eps)?;
    }
    enqueue_add_rows(dev, h, &down, batch * hidden_dim)?;
    drop(down);
    Ok(routed_on_device)
}

/// One gated FFN over `x` (`batch` rows): gate and up, the gated
/// activation, down. The weights are held until the stack's wait.
pub(super) fn run_expert(
    dev: &Arc<CudaDevice>,
    held: &mut Held,
    ex: &ExpertProjCuda<'_>,
    glu: GluCuda,
    x: &MatvecInput<'_>,
    batch: usize,
    ffn_dim: usize,
) -> Result<CudaSlice<f32>, CudaError> {
    let mut gate = held.proj(dev, &ex.gate, x, batch)?;
    let mut up = held.proj(dev, &ex.up, x, batch)?;
    if let Some(b) = ex.bias {
        let d_g = held.vector(dev, b.gate)?;
        enqueue_add_bias_rows(dev, &mut gate, &d_g.slice, batch, ffn_dim)?;
        let d_u = held.vector(dev, b.up)?;
        enqueue_add_bias_rows(dev, &mut up, &d_u.slice, batch, ffn_dim)?;
    }
    let act = glu_mul_device(dev, glu, &gate, &up, batch * ffn_dim)?;
    drop((gate, up));
    let mut down = held.proj(dev, &ex.down, &MatvecInput::new(&act), batch)?;
    if let Some(b) = ex.bias {
        let d_d = held.vector(dev, b.down)?;
        enqueue_add_bias_rows(dev, &mut down, &d_d.slice, batch, ex.down.rows())?;
    }
    Ok(down)
}
