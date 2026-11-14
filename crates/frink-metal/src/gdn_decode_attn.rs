//! A hybrid's ATTENTION layer whole inside a decode run
//! (`crate::gdn_branch::GdnRun`): `attn_norm`, the shared input
//! rotation, the Q/K/V projections, the interleaved Q gate split, the
//! QK norms and biases, RoPE at the token's position, the append to the
//! sequence's device KV mirror, the attention, the gate, `wo` and the
//! FFN, with the residual stream resident.
//!
//! # Why
//!
//! Issue #540. Before this a Bonsai or Qwen3.5 decode token came back
//! to the host once per attention layer: the run ended with the next
//! layer's Q/K/V, the host split the gate, normed and rotated them, and
//! the next run began with the attention. Sixteen waits a token on
//! Bonsai-2-27B, each an OS wake-up plus the host work between, where
//! llama.cpp's graph waits once. With this the whole stack is one run
//! and the token waits once.
//!
//! The decode twin of `crate::gdn_prefill_attn`: the same kernels after
//! the projections, at one row, and matvecs instead of GEMMs.
//!
//! The run's last step is here too: the final norm and the output
//! projection ([`GdnRun::output_head`]), so a token whose every layer
//! rode the run downloads its logits from the same wait rather than
//! paying a submission and a wake-up for the head.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use crate::attn::{AttnExtras, LayerRope, MetalKvBuffers, MetalKvDtype, MetalRope};
use crate::gdn_branch::{encode_attn_tail, GdnRun, LayerFfn, ScratchSet};
use crate::gpu::{
    encode_matvec, resident_f32_buffer, resident_weight_buffer, shared_metal, MatvecLaunch,
    MetalError,
};
use crate::hadamard::FoldPlan;

/// One attention layer as a decode run takes it. Built by the caller
/// per token and checked by [`Self::is_supported`] before anything is
/// encoded.
pub struct DecodeAttnLayer<'a> {
    /// `blk.N.attn_norm.weight`; the model's RMS epsilon is `eps`.
    pub attn_norm: &'a [f32],
    pub eps: f32,
    /// The rotation `wq`, `wk` and `wv` share on their input.
    pub fold_x: Option<&'a FoldPlan<'a>>,
    pub q: &'a MatvecLaunch<'a>,
    pub k: &'a MatvecLaunch<'a>,
    pub v: &'a MatvecLaunch<'a>,
    pub n_heads: usize,
    /// `wq` is double width, a gate interleaved per head, applied as
    /// `attn *= sigmoid(gate)` before `wo`.
    pub q_gate: bool,
    /// The QKV biases, the per-head QK norms and the softcap. No sinks,
    /// no `wo` bias, no V norm: [`Self::is_supported`] refuses them.
    pub extras: &'a AttnExtras<'a>,
    pub rope: MetalRope,
    /// `None` for a layer that does not rotate.
    pub layer_rope: Option<LayerRope<'a>>,
    pub o_proj: &'a MatvecLaunch<'a>,
    pub fold_branch: Option<&'a FoldPlan<'a>>,
    pub ffn: &'a LayerFfn<'a>,
}

fn cols(m: &MatvecLaunch<'_>) -> usize {
    m.row_bytes / m.block_bytes * m.block_elems
}

impl DecodeAttnLayer<'_> {
    /// Whether every width agrees with every other, with the mirror and
    /// with what the kernels serve, for one token at `pos`.
    pub fn is_supported(&self, kv: &MetalKvBuffers, pos: usize) -> bool {
        let hidden = self.attn_norm.len();
        let (hd, n_kv) = (kv.head_dim, kv.n_kv_heads);
        let q_width = self.n_heads * hd;
        let kv_width = n_kv * hd;
        let e = self.extras;
        let norm_ok = |w: Option<&[f32]>| w.is_none_or(|w| w.len() == hd);
        let bias_ok = |b: Option<&[f32]>, n: usize| b.is_none_or(|b| b.len() == n);
        let fold_ok = |p: Option<&FoldPlan<'_>>, w: usize| p.is_none_or(|p| p.check(w).is_ok());
        let rot = self.rope.rot_dim.unwrap_or(hd);
        let f = self.ffn;
        hidden > 0
            && hd > 0
            && hd <= 256
            && n_kv > 0
            && self.n_heads.is_multiple_of(n_kv)
            // f16, the store llama.cpp attends over at its default
            // `-ctk`, as the prefill run requires: a quantized mirror
            // attends through a process-wide scratch.
            && kv.dtype == MetalKvDtype::F16
            && kv.seq_len == pos
            && kv.seq_len < kv.capacity()
            && cols(self.q) == hidden
            && cols(self.k) == hidden
            && cols(self.v) == hidden
            && self.q.rows == q_width * (1 + usize::from(self.q_gate))
            && self.k.rows == kv_width
            && self.v.rows == kv_width
            && self.o_proj.rows == hidden
            && cols(self.o_proj) == q_width
            && fold_ok(self.fold_x, hidden)
            && fold_ok(self.fold_branch, q_width)
            && f.fits(hidden)
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

/// `attn_norm(hidden)`, the shared input rotation and the three
/// projections, encoded into the caller's command buffer.
///
/// The ONE attention head: [`GdnRun::attn_head`] ends a run with it and
/// hands Q/K/V to the host, and [`GdnRun::attn_layer_resident`] keeps
/// them on the device.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attn_head(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    device: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    hidden_buf: &ProtocolObject<dyn MTLBuffer>,
    norm: &[f32],
    norm_eps: f32,
    q: &MatvecLaunch<'_>,
    k: &MatvecLaunch<'_>,
    v: &MatvecLaunch<'_>,
    fold_x: Option<&FoldPlan<'_>>,
    normed: &ProtocolObject<dyn MTLBuffer>,
    out: [&ProtocolObject<dyn MTLBuffer>; 3],
) -> Result<(), MetalError> {
    encode_normed_projections(
        encoder,
        device,
        hidden_buf,
        norm,
        norm_eps,
        fold_x,
        normed,
        &[(q, out[0]), (k, out[1]), (v, out[2])],
    )
}

/// `rms_norm(hidden)`, its rotation when the projections are folded,
/// and each projection of the result: the attention head's three and
/// the output head's one.
#[allow(clippy::too_many_arguments)]
fn encode_normed_projections(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    device: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    hidden_buf: &ProtocolObject<dyn MTLBuffer>,
    norm: &[f32],
    norm_eps: f32,
    fold_x: Option<&FoldPlan<'_>>,
    normed: &ProtocolObject<dyn MTLBuffer>,
    projections: &[(&MatvecLaunch<'_>, &ProtocolObject<dyn MTLBuffer>)],
) -> Result<(), MetalError> {
    let hidden = norm.len();
    let norm_w = resident_f32_buffer(device, norm)?;
    let signs = match fold_x.and_then(|p| p.signs) {
        None => None,
        Some(sg) => {
            // SAFETY: a `&[f32]` viewed as its own bytes, read only.
            let bytes = unsafe {
                std::slice::from_raw_parts(sg.as_ptr() as *const u8, std::mem::size_of_val(sg))
            };
            Some(resident_weight_buffer(device, bytes)?)
        }
    };
    crate::norm::encode_rms_norm(
        encoder,
        device,
        hidden_buf,
        &norm_w.buffer,
        normed,
        hidden as u32,
        norm_eps,
    )?;
    if let Some(plan) = fold_x {
        plan.check(hidden)?;
        crate::hadamard::encode_fold(
            encoder,
            device,
            normed,
            hidden,
            plan,
            signs.as_ref().map(|b| &*b.buffer),
        )?;
    }
    for (m, out) in projections {
        let w = resident_weight_buffer(device, m.weights)?;
        encode_matvec(encoder, device, m, &w, normed, out)?;
    }
    Ok(())
}

/// Where in `main` the attention reads its inputs and writes its
/// output, so [`encode_attn_core`] can take the scratch by value and
/// still be told which of its buffers are which.
pub(crate) struct AttnSlots {
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub gate: Option<usize>,
    /// `n_heads * head_dim` floats, the attention output and `wo`'s
    /// input.
    pub attn: usize,
}

/// Append this token's K/V, attend over everything including it, gate,
/// then `wo` and the FFN tail, encoded into the caller's encoder.
///
/// The ONE decode attention: [`crate::gdn_branch`]'s host-fed layer
/// writes Q/K/V into `main` from the host, and
/// [`GdnRun::attn_layer_resident`] has its own kernels write them.
/// `main` is held in `keep` until the command buffer completes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attn_core(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    device: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    kv: &mut MetalKvBuffers,
    main: crate::scratch_pool::Scratch,
    slots: AttnSlots,
    n_heads: usize,
    softcap: Option<f32>,
    out_proj: &MatvecLaunch<'_>,
    fold_branch: Option<&FoldPlan<'_>>,
    ffn: &LayerFfn<'_>,
    residual: &[f32],
    h_ext: Option<&ProtocolObject<dyn MTLBuffer>>,
    keep: &mut Vec<ScratchSet>,
) -> Result<(), MetalError> {
    let hidden = out_proj.rows;
    let per_token = kv.elems_per_token();
    let q_len = n_heads * kv.head_dim;
    let branch_cols = cols(out_proj);
    if (h_ext.is_none() && residual.len() != hidden)
        || branch_cols != q_len
        || kv.seq_len >= kv.capacity()
    {
        return Err(MetalError::CommandFailed);
    }
    let tail = crate::scratch_pool::Scratch::take(device, &ffn.tail_lens(hidden))
        .ok_or(MetalError::BufferAllocFailed)?;
    let proj = crate::scratch_pool::Scratch::take(device, &[hidden])
        .ok_or(MetalError::BufferAllocFailed)?;
    let out_w = resident_weight_buffer(device, out_proj.weights)?;
    let signs = match fold_branch.and_then(|p| p.signs) {
        None => None,
        Some(sg) => {
            // SAFETY: a `&[f32]` viewed as its own bytes, read only.
            let bytes = unsafe {
                std::slice::from_raw_parts(sg.as_ptr() as *const u8, std::mem::size_of_val(sg))
            };
            Some(resident_weight_buffer(device, bytes)?)
        }
    };
    let (q_buf, k_buf, v_buf) = (main.buf(slots.q), main.buf(slots.k), main.buf(slots.v));
    let attn_buf = main.buf(slots.attn);

    let mut mrs = crate::mem_ranges::MemRanges::new();
    let at = (kv.seq_len * per_token) as u32;
    crate::attn::encode_kv_store_append(encoder, device, k_buf, v_buf, kv, at, per_token as u32)?;
    let seq_len = kv.seq_len + 1;
    crate::attn::encode_gqa_with_kv(
        encoder,
        &mut mrs,
        device,
        q_buf,
        kv,
        attn_buf,
        n_heads as u32,
        kv.n_kv_heads as u32,
        kv.head_dim as u32,
        seq_len as u32,
        0,
        softcap,
        None,
    )?;
    if let Some(g) = slots.gate {
        crate::elem::encode_sigmoid_mul(encoder, device, attn_buf, main.buf(g), q_len as u32)?;
    }
    // `wo` reads the gated attention output directly: the fold rewrites
    // that buffer in place and nothing reads it afterwards.
    encode_attn_tail(
        encoder,
        device,
        out_proj,
        &out_w,
        fold_branch,
        signs.as_ref().map(|b| &*b.buffer),
        ffn,
        attn_buf,
        branch_cols,
        proj.buf(0),
        &tail,
        residual,
        h_ext,
        hidden,
    )?;
    kv.seq_len = seq_len;
    keep.push(ScratchSet {
        main,
        head: Some(proj),
        ffn: Some(tail),
    });
    Ok(())
}

impl GdnRun {
    /// The output head at the END of this run: `final_norm(hidden)`, its
    /// rotation when `output` is folded, and `output` itself. The logits
    /// come back from [`Self::finish_with_head`] with the residual
    /// stream, unscaled and uncapped: the caller applies the head's
    /// bias, multiplier and softcap as it does to a host projection.
    pub fn output_head(
        &mut self,
        norm: &[f32],
        norm_eps: f32,
        output: &MatvecLaunch<'_>,
        fold: Option<&FoldPlan<'_>>,
    ) -> Result<(), MetalError> {
        let hidden = self.rows;
        if norm.len() != hidden
            || cols(output) != hidden
            || self.head.is_some()
            || self.logits.is_some()
        {
            return Err(MetalError::CommandFailed);
        }
        let shared = shared_metal()?;
        let device = &shared.device;
        // normed, then the logits.
        let sc = crate::scratch_pool::Scratch::take(device, &[hidden, output.rows])
            .ok_or(MetalError::BufferAllocFailed)?;
        let cmd_buf = shared
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandFailed)?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?;
        encode_normed_projections(
            &encoder,
            device,
            self.hidden.buf(0),
            norm,
            norm_eps,
            fold,
            sc.buf(0),
            &[(output, sc.buf(1))],
        )?;
        encoder.endEncoding();
        cmd_buf.commit();
        self.committed(cmd_buf);
        self.logits = Some((sc, output.rows));
        Ok(())
    }

    /// A whole ATTENTION layer in this run, from the residual stream the
    /// run holds to the one it leaves: nothing returns to the host.
    ///
    /// `kv` is the layer's device mirror, which must hold exactly the
    /// `pos` rows before this token ([`DecodeAttnLayer::is_supported`]).
    /// Its `seq_len` advances here; the caller copies the new row back
    /// to the host cache once [`Self::finish`] has waited.
    ///
    /// # Safety
    ///
    /// As [`Self::layer`]: the mirror this borrows must stay the
    /// caller's until [`Self::finish`] returns.
    pub unsafe fn attn_layer_resident(
        &mut self,
        w: &DecodeAttnLayer<'_>,
        kv: &mut MetalKvBuffers,
        pos: usize,
    ) -> Result<(), MetalError> {
        if self.rows != w.attn_norm.len() || !w.is_supported(kv, pos) {
            return Err(MetalError::CommandFailed);
        }
        let shared = shared_metal()?;
        let device = &shared.device;
        let hd = kv.head_dim;
        let (q_width, kv_width) = (w.n_heads * hd, kv.n_kv_heads * hd);
        // normed, q (fused when gated), k, v, q, gate, attention out.
        let main = crate::scratch_pool::Scratch::take(
            device,
            &[
                self.rows, w.q.rows, kv_width, kv_width, q_width, q_width, q_width,
            ],
        )
        .ok_or(MetalError::BufferAllocFailed)?;
        let (q_slot, gate) = if w.q_gate { (4, Some(5)) } else { (1, None) };
        let resident = if w.extras.encodes_anything() {
            Some(crate::attn::resident_attn_extras(device, w.extras)?)
        } else {
            None
        };
        let ff = match w.layer_rope.and_then(|r| r.freq_factors) {
            Some(ff) => Some(resident_f32_buffer(device, ff)?),
            None => None,
        };
        let cmd_buf = shared
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandFailed)?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?;
        encode_attn_head(
            &encoder,
            device,
            self.hidden.buf(0),
            w.attn_norm,
            w.eps,
            w.q,
            w.k,
            w.v,
            w.fold_x,
            main.buf(0),
            [main.buf(1), main.buf(2), main.buf(3)],
        )?;
        if w.q_gate {
            crate::gdn_prefill_attn::encode_q_gate_split(
                &encoder,
                device,
                main.buf(1),
                main.buf(4),
                main.buf(5),
                hd,
                q_width,
            )?;
        }
        if let Some(resident) = &resident {
            crate::attn::encode_attn_extras_batch(
                &encoder,
                device,
                w.extras,
                main.buf(q_slot),
                main.buf(2),
                main.buf(3),
                q_width,
                kv_width,
                kv_width,
                w.n_heads,
                kv.n_kv_heads,
                hd,
                1,
                w.eps,
                resident,
            )?;
        }
        if let Some(r) = w.layer_rope {
            for (slot, heads) in [(q_slot, w.n_heads), (2, kv.n_kv_heads)] {
                crate::rope::encode_rope_batch(
                    &encoder,
                    device,
                    w.rope,
                    main.buf(slot),
                    heads as u32,
                    hd as u32,
                    r.theta,
                    pos as u32,
                    1,
                    ff.as_ref().map(|f| &*f.buffer),
                )?;
            }
        }
        let hidden_buf = self.hidden.buf(0);
        encode_attn_core(
            &encoder,
            device,
            kv,
            main,
            AttnSlots {
                q: q_slot,
                k: 2,
                v: 3,
                gate,
                attn: 6,
            },
            w.n_heads,
            w.extras.attn_logit_softcap,
            w.o_proj,
            w.fold_branch,
            w.ffn,
            &[],
            Some(hidden_buf),
            &mut self.keep,
        )?;
        self.note_routed(w.ffn);
        encoder.endEncoding();
        cmd_buf.commit();
        self.committed(cmd_buf);
        Ok(())
    }
}
