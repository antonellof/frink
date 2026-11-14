//! An ATTENTION layer whole inside a prefill run
//! (`crate::gdn_prefill`): `attn_norm`, the shared input rotation, the
//! Q/K/V GEMMs, the interleaved Q gate split, the QK norms, RoPE, the
//! append to the sequence's device KV mirror, the causal attention, the
//! gate, `wo` and the FFN, with the residual stream resident.
//!
//! # Why
//!
//! Issue #540. Once the recurrent layers ran as one device run each,
//! a Bonsai-2-27B `pp512` still spent about a quarter of its time on
//! the sixteen attention layers between those runs: three projection
//! submissions each, the gate split, the QK norms, RoPE and the
//! attention on the host, and a run boundary per attention layer. With
//! this the whole stack is one command buffer per prompt batch.
//!
//! The GEMMs read f32 activations, as the per-launch projections did,
//! so Q/K/V round as before; the attention is the device prefill kernel
//! over the f16 mirror, the store llama.cpp's own `-ctk f16` attends
//! over.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};
use std::ptr::NonNull;

use crate::attn::{AttnExtras, LayerRope, MetalKvBuffers, MetalKvDtype, MetalRope};
use crate::gdn_prefill::{
    cols_of, encode_fold, encode_gemm, encode_wo_tail, fold_ok, slot, wo_tail_supported, Enc,
    PrefillFfn,
};
use crate::gpu::{ensure_pipeline, resident_f32_buffer, MetalError, MulMmSgLaunch};
use crate::hadamard::FoldPlan;

/// `qwen35.cpp:191-199`: each head of the double-width `wq` output is
/// `[q (head_dim) | gate (head_dim)]`. One thread per element of Q;
/// `head` counts heads across rows, so a row needs no index of its own.
const Q_GATE_SPLIT_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q_gate_split_rows(
    device const float* fused [[buffer(0)]],
    device float* q [[buffer(1)]],
    device float* g [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    const uint head_dim = dims.x;
    if (i >= dims.y) {
        return;
    }
    const uint head = i / head_dim;
    const uint src = head * 2u * head_dim + (i - head * head_dim);
    q[i] = fused[src];
    g[i] = fused[src + head_dim];
}
"#;

/// One attention layer as the prefill run takes it. Built by the caller
/// once per layer and checked by [`Self::is_supported`] before anything
/// is encoded.
pub struct PrefillAttnLayer<'a> {
    /// `blk.N.attn_norm.weight`; the model's RMS epsilon is `eps`.
    pub attn_norm: &'a [f32],
    pub eps: f32,
    /// The rotation `wq`, `wk` and `wv` share on their input.
    pub fold_x: Option<FoldPlan<'a>>,
    pub q: MulMmSgLaunch<'a>,
    pub k: MulMmSgLaunch<'a>,
    pub v: MulMmSgLaunch<'a>,
    pub n_heads: usize,
    /// `wq` is double width, a gate interleaved per head, applied as
    /// `attn *= sigmoid(gate)` before `wo`.
    pub q_gate: bool,
    /// The QKV biases and the per-head QK norms, applied before RoPE,
    /// and the attention softcap. No sinks, no `wo` bias, no V norm:
    /// [`Self::is_supported`] refuses them.
    pub extras: AttnExtras<'a>,
    pub rope: MetalRope,
    /// `None` for a layer that does not rotate.
    pub layer_rope: Option<LayerRope<'a>>,
    pub o_proj: MulMmSgLaunch<'a>,
    pub fold_branch: Option<FoldPlan<'a>>,
    pub ffn: PrefillFfn<'a>,
}

impl PrefillAttnLayer<'_> {
    /// Whether every width agrees with every other, with the mirror and
    /// with what the kernels serve, for a batch of `rows` at
    /// `start_pos`.
    pub fn is_supported(&self, kv: &MetalKvBuffers, rows: usize, start_pos: usize) -> bool {
        let hidden = self.attn_norm.len();
        let (hd, n_kv) = (kv.head_dim, kv.n_kv_heads);
        let q_width = self.n_heads * hd;
        let kv_width = n_kv * hd;
        let e = &self.extras;
        let norm_ok = |w: Option<&[f32]>| w.is_none_or(|w| w.len() == hd);
        let bias_ok = |b: Option<&[f32]>, n: usize| b.is_none_or(|b| b.len() == n);
        let rot = self.rope.rot_dim.unwrap_or(hd);
        hidden > 0
            && hd > 0
            && hd <= 256
            && n_kv > 0
            && self.n_heads.is_multiple_of(n_kv)
            // f16, the store llama.cpp attends over at its default
            // `-ctk`; a quantized mirror attends through a process-wide
            // scratch this run does not order.
            && kv.dtype == MetalKvDtype::F16
            && kv.seq_len == start_pos
            && kv.seq_len + rows <= kv.capacity
            && cols_of(&self.q) == hidden
            && cols_of(&self.k) == hidden
            && cols_of(&self.v) == hidden
            && self.q.rows == q_width * (1 + usize::from(self.q_gate))
            && self.k.rows == kv_width
            && self.v.rows == kv_width
            && cols_of(&self.o_proj) == q_width
            && fold_ok(self.fold_x.as_ref(), hidden)
            && fold_ok(self.fold_branch.as_ref(), q_width)
            && wo_tail_supported(&self.o_proj, self.fold_branch.as_ref(), &self.ffn, hidden)
            && norm_ok(e.q_norm)
            && norm_ok(e.k_norm)
            && e.v_norm.is_none()
            && bias_ok(e.q_bias, q_width)
            && bias_ok(e.k_bias, kv_width)
            && bias_ok(e.v_bias, kv_width)
            && e.sinks.is_none()
            && e.o_bias.is_none()
            && rot > 0
            && rot.is_multiple_of(2)
            && rot <= hd
            && self
                .layer_rope
                .and_then(|r| r.freq_factors)
                .is_none_or(|ff| ff.len() == rot / 2)
    }
}

pub(crate) fn encode_q_gate_split(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    fused: &ProtocolObject<dyn MTLBuffer>,
    q: &ProtocolObject<dyn MTLBuffer>,
    g: &ProtocolObject<dyn MTLBuffer>,
    head_dim: usize,
    n: usize,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, Q_GATE_SPLIT_SRC, "q_gate_split_rows")?;
    encoder.setComputePipelineState(&pipe.0);
    // SAFETY: three buffers the run's scratch owns and an 8-byte
    // constant that outlives the call.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(fused), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(q), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(g), 0, 2);
        let mut dims: [u32; 2] = [head_dim as u32, n as u32];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 8, 3);
    }
    crate::dispatch::dispatch_counted(
        encoder,
        MTLSize {
            width: n.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// One attention layer into the run, reading and writing the residual
/// stream in `slot::H` and appending `rows` rows to `kv` (whose
/// `seq_len` the run advances once it has waited).
pub(crate) fn encode_attn_layer(
    enc: &mut Enc<'_>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    w: &PrefillAttnLayer<'_>,
    kv: &MetalKvBuffers,
    sc: &crate::scratch_pool::Scratch,
    rows: usize,
    start_pos: usize,
) -> Result<(), MetalError> {
    let b = |i: usize| sc.buf(i);
    let hidden = w.attn_norm.len();
    let (hd, n_kv) = (kv.head_dim, kv.n_kv_heads);
    let (q_width, kv_width) = (w.n_heads * hd, n_kv * hd);
    let attn_norm = resident_f32_buffer(device, w.attn_norm)?;

    let encoder = enc.span("pf_attn_norm")?;
    crate::norm::encode_rms_norm_batch(
        encoder,
        device,
        b(slot::H),
        &attn_norm.buffer,
        b(slot::XN),
        hidden as u32,
        rows as u32,
        w.eps,
    )?;
    let encoder = enc.span("pf_fold_x")?;
    encode_fold(
        encoder,
        device,
        b(slot::XN),
        hidden,
        rows,
        w.fold_x.as_ref(),
    )?;
    let encoder = enc.span("pf_qkv")?;
    let q_out = if w.q_gate { slot::AQ } else { slot::AQR };
    encode_gemm(encoder, device, &w.q, b(slot::XN), b(q_out), rows)?;
    encode_gemm(encoder, device, &w.k, b(slot::XN), b(slot::AK), rows)?;
    encode_gemm(encoder, device, &w.v, b(slot::XN), b(slot::AV), rows)?;
    if w.q_gate {
        let encoder = enc.span("pf_q_gate_split")?;
        encode_q_gate_split(
            encoder,
            device,
            b(slot::AQ),
            b(slot::AQR),
            b(slot::AG),
            hd,
            rows * q_width,
        )?;
    }
    if w.extras.encodes_anything() {
        let resident = crate::attn::resident_attn_extras(device, &w.extras)?;
        let encoder = enc.span("pf_attn_extras")?;
        crate::attn::encode_attn_extras_batch(
            encoder,
            device,
            &w.extras,
            b(slot::AQR),
            b(slot::AK),
            b(slot::AV),
            q_width,
            kv_width,
            kv_width,
            w.n_heads,
            n_kv,
            hd,
            rows,
            w.eps,
            &resident,
        )?;
    }
    if let Some(r) = w.layer_rope {
        let ff = match r.freq_factors {
            Some(ff) => Some(resident_f32_buffer(device, ff)?),
            None => None,
        };
        let ff = ff.as_ref().map(|f| &*f.buffer);
        let encoder = enc.span("pf_rope")?;
        for (buf, heads) in [(slot::AQR, w.n_heads), (slot::AK, n_kv)] {
            crate::rope::encode_rope_batch(
                encoder,
                device,
                w.rope,
                b(buf),
                heads as u32,
                hd as u32,
                r.theta,
                start_pos as u32,
                rows as u32,
                ff,
            )?;
        }
    }
    let encoder = enc.span("pf_kv_append")?;
    crate::attn::encode_kv_store_append(
        encoder,
        device,
        b(slot::AK),
        b(slot::AV),
        kv,
        (kv.seq_len * kv_width) as u32,
        (rows * kv_width) as u32,
    )?;
    let encoder = enc.span("pf_attention")?;
    crate::attn::encode_gqa_prefill_with_kv(
        encoder,
        &mut crate::mem_ranges::MemRanges::new(),
        device,
        b(slot::AQR),
        kv,
        b(slot::AATT),
        w.n_heads as u32,
        n_kv as u32,
        hd as u32,
        rows as u32,
        kv.seq_len as u32,
        w.extras.attn_logit_softcap,
        crate::attn::AttnMask::default(),
    )?;
    if w.q_gate {
        let encoder = enc.span("pf_attn_gate")?;
        crate::elem::encode_sigmoid_mul(
            encoder,
            device,
            b(slot::AATT),
            b(slot::AG),
            (rows * q_width) as u32,
        )?;
    }
    encode_wo_tail(
        enc,
        device,
        &w.o_proj,
        w.fold_branch.as_ref(),
        &w.ffn,
        w.eps,
        b(slot::AATT),
        sc,
        rows,
    )
}
