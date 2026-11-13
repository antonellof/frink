//! A recurrent layer's whole BRANCH in one command buffer: the two
//! gates, the causal convolution with its SiLU, the per-head l2 norms,
//! the delta rule, the gated output norm, the folded rotation and the
//! output projection.
//!
//! # Why this shape and not another
//!
//! `1ebda4fc:docs/plans/gdn-resident-state.md` prices a Bonsai decode token: 192
//! command buffers, 71.6 ms of GPU, 29.9 ms of latency beyond it, and
//! about 35 ms of host of which two thirds is the recurrence. The GPU
//! work is already faster than the reference's whole token, the 0.15 ms
//! a submission costs is the OS wake-up, and every way around that has
//! been measured and lost. So the only thing left to remove is host
//! time inside a layer, and this is where nearly all of it is.
//!
//! The projections that FEED this stay where they are. `attn_qkv` and
//! `attn_gate` are one launch already (`WeightMatrix::apply_pair`), and
//! the two gate projections are tiny unquantized matrices whose own
//! measurement said sending them with the pair was neutral. So a
//! recurrent layer keeps the two submissions it has and loses the host
//! step between them, rather than gaining a third.
//!
//! # Why it can win where three earlier attempts lost
//!
//! Those three moved the state once per row through a kernel that read
//! it UNCOALESCED, and measured the kernel against six CPU cores while
//! still paying a submission for it. Two of those three facts have
//! changed: `crate::gdn`'s step reads the state coalesced now, and the
//! submission this rides in already existed. The third has not, and is
//! why the host body stays as the fallback and the caller measures.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use crate::attn::MetalKvBuffers;
use crate::gdn::{buffer_no_copy, encode_delta_step_at, encode_gated_norm, DeltaShape};
use crate::gdn_head::{encode_conv_silu, encode_gates, encode_l2_norm_heads, HeadShape};
use crate::gpu::{encode_matvec, resident_weight_buffer, shared_metal, MatvecLaunch, MetalError};
use crate::hadamard::FoldPlan;

/// Everything about one recurrent layer that does not change between
/// tokens, so the launch takes ONE argument and cannot be handed the
/// convolution of one layer with the output projection of another.
pub struct BranchWeights<'a> {
    pub shape: DeltaShape,
    pub head: HeadShape,
    /// `blk.N.ssm_conv1d.weight`, `[conv_dim][d_conv]`.
    pub conv1d: &'a [f32],
    /// `blk.N.ssm_dt.bias` and `blk.N.ssm_a`, `[n_v_heads]` each.
    pub dt_bias: &'a [f32],
    pub a: &'a [f32],
    /// `blk.N.ssm_norm.weight`, `[head_dim]`.
    pub ssm_norm: &'a [f32],
    /// The model's RMS epsilon, which the l2 norms clamp by and the
    /// gated norm uses.
    pub eps: f32,
    /// `blk.N.ssm_out.weight`, and the rotation its input needs.
    pub out_proj: &'a MatvecLaunch<'a>,
    pub fold_y: Option<&'a FoldPlan<'a>>,
    /// The layer's HEAD, when the caller owns it too: the input norm
    /// and the four projections that feed the branch.
    ///
    /// `Some` removes the second submission a recurrent layer costs,
    /// and with `ffn` set as well the layer is ONE.
    pub head_in: Option<LayerHeadIn<'a>>,
    /// The rest of the layer, when the caller owns it too.
    ///
    /// `Some` turns three submissions a layer into ONE: the branch's
    /// output is added to the residual, normed and run through the FFN
    /// without the hidden state coming back to the host in between,
    /// which is the only thing left between this engine and the
    /// reference's decode rate
    /// (`1ebda4fc:docs/plans/gdn-resident-state.md`). `None` stops at
    /// `ssm_out` and the caller finishes the layer itself.
    pub ffn: Option<LayerFfn<'a>>,
}

/// The head of a layer: `attn_norm(x)`, then the four projections the
/// branch reads.
///
/// `beta` and `alpha` are the SPLIT spelling only; the fused one's
/// per-group interleave is host arithmetic with no kernel here, and a
/// caller carrying it passes `None` for the whole head.
pub struct LayerHeadIn<'a> {
    pub norm: &'a [f32],
    pub norm_eps: f32,
    pub qkv: &'a MatvecLaunch<'a>,
    pub z: &'a MatvecLaunch<'a>,
    pub beta: &'a MatvecLaunch<'a>,
    pub alpha: &'a MatvecLaunch<'a>,
    /// The rotation the projections' shared input needs. One plan, not
    /// four: they all read `attn_norm(x)`, and a checkpoint whose four
    /// disagree about their input basis is not this shape.
    pub fold_x: Option<&'a FoldPlan<'a>>,
}

/// The dense half of a layer: `x + ffn(norm(x + branch))`.
///
/// One shape only, and a caller whose layer is not that shape passes
/// `None` rather than having this approximate it. What that excludes is
/// named at the call site, field by field.
pub struct LayerFfn<'a> {
    /// `blk.N.ffn_norm.weight` and the model's RMS epsilon.
    pub norm: &'a [f32],
    pub norm_eps: f32,
    pub gate: &'a MatvecLaunch<'a>,
    pub up: &'a MatvecLaunch<'a>,
    pub down: &'a MatvecLaunch<'a>,
    /// The rotations the FFN's input and its activation need.
    pub fold_x: Option<&'a FoldPlan<'a>>,
    pub fold_act: Option<&'a FoldPlan<'a>>,
}

impl BranchWeights<'_> {
    /// Whether the shapes agree with each other and with what the
    /// kernels serve. A caller checks this once per layer rather than
    /// discovering it per token.
    pub fn is_supported(&self) -> bool {
        let h = self.head;
        h.n_k_heads > 0
            && h.n_v_heads > 0
            && h.head_dim.is_power_of_two()
            && h.head_dim <= crate::gdn::MAX_HEAD_DIM
            && h.d_conv >= 2
            && h.d_conv <= 8
            && h.n_v_heads.is_multiple_of(h.n_k_heads)
            && self.shape.n_k_heads == h.n_k_heads
            && self.shape.n_v_heads == h.n_v_heads
            && self.shape.head_dim == h.head_dim
            && self.conv1d.len() == h.conv_dim() * h.d_conv
            && self.dt_bias.len() == h.n_v_heads
            && self.a.len() == h.n_v_heads
            && self.ssm_norm.len() == h.head_dim
            && self.out_proj.row_bytes / self.out_proj.block_bytes * self.out_proj.block_elems
                == h.value_dim()
            && match &self.head_in {
                None => true,
                Some(hd) => {
                    let hidden = self.out_proj.rows;
                    hd.norm.len() == hidden
                        && hd.qkv.rows == h.conv_dim()
                        && hd.z.rows == h.value_dim()
                        && hd.beta.rows == h.n_v_heads
                        && hd.alpha.rows == h.n_v_heads
                        && [hd.qkv, hd.z, hd.beta, hd.alpha]
                            .iter()
                            .all(|m| m.row_bytes / m.block_bytes * m.block_elems == hidden)
                }
            }
            && match &self.ffn {
                None => true,
                Some(f) => {
                    let hidden = self.out_proj.rows;
                    f.norm.len() == hidden
                        && f.gate.rows == f.up.rows
                        && f.down.rows == hidden
                        && f.gate.row_bytes / f.gate.block_bytes * f.gate.block_elems == hidden
                        && f.up.row_bytes / f.up.block_bytes * f.up.block_elems == hidden
                        && f.down.row_bytes / f.down.block_bytes * f.down.block_elems == f.gate.rows
                }
            }
    }
}

/// `h += branch`, then `h += ffn(rms_norm(h))`, encoded into the
/// caller's command buffer.
///
/// The ONE FFN tail: a fused recurrent layer ends with it
/// ([`encode_layer`]) and so does a fused ATTENTION layer
/// ([`launch_attn_tail`]), which has the same shape after `wo` and
/// nothing else in common. `h_ext` is the residual stream when the
/// caller already holds it on the device (a run of layers), else the
/// stream is filled from `res` into the scratch's own buffer.
///
/// Returns the buffer holding the layer's output.
#[allow(clippy::too_many_arguments)]
fn encode_ffn_tail<'b>(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    device: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    f: &LayerFfn<'_>,
    sc: &'b crate::scratch_pool::Scratch,
    branch: &ProtocolObject<dyn MTLBuffer>,
    res: &[f32],
    h_ext: Option<&'b ProtocolObject<dyn MTLBuffer>>,
    hidden: usize,
) -> Result<&'b ProtocolObject<dyn MTLBuffer>, MetalError> {
    // The layer's residual stream: the caller's buffer when
    // there is one (a RUN of layers shares it and it never
    // comes back to the host between them), else this layer's
    // own, filled from the host.
    let h_buf = match h_ext {
        Some(b) => b,
        None => sc.write(0, res).ok_or(MetalError::CommandFailed)?,
    };
    let (normed, gate_buf) = (sc.buf(1), sc.buf(2));
    let (up_buf, act_buf, ffn_out) = (sc.buf(3), sc.buf(4), sc.buf(5));
    let ffn_signs = |plan: Option<&FoldPlan<'_>>| match plan.and_then(|p| p.signs) {
        None => Ok(None),
        Some(signs) => {
            // SAFETY: a `&[f32]` viewed as its own bytes, read only.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    signs.as_ptr() as *const u8,
                    std::mem::size_of_val(signs),
                )
            };
            resident_weight_buffer(device, bytes).map(Some)
        }
    };
    let x_signs = ffn_signs(f.fold_x)?;
    let act_signs = ffn_signs(f.fold_act)?;
    let norm_w = crate::gpu::resident_f32_buffer(device, f.norm)?;
    let gate_w = resident_weight_buffer(device, f.gate.weights)?;
    let up_w = resident_weight_buffer(device, f.up.weights)?;
    let down_w = resident_weight_buffer(device, f.down.weights)?;

    // `h += branch` and then `rms_norm(h)` are ONE kernel, the
    // same one the dense decode stack uses. Two dispatches here
    // cost about 8 microseconds of fixed overhead each, which
    // across 48 layers is where a measurable part of this
    // token's GPU time goes (`1ebda4fc:docs/plans/gdn-resident-state.md`
    // prices the branch's twelve dispatches at ~4.8 ms).
    crate::norm::encode_add_rms_norm(
        encoder,
        device,
        h_buf,
        branch,
        &norm_w.buffer,
        normed,
        hidden as u32,
        f.norm_eps,
    )?;
    if let Some(plan) = f.fold_x {
        plan.check(hidden)?;
        crate::hadamard::encode_fold(
            encoder,
            device,
            normed,
            hidden,
            plan,
            x_signs.as_ref().map(|b| &*b.buffer),
        )?;
    }
    encode_matvec(encoder, device, f.gate, &gate_w, normed, gate_buf)?;
    encode_matvec(encoder, device, f.up, &up_w, normed, up_buf)?;
    crate::elem::encode_silu_mul(
        encoder,
        device,
        gate_buf,
        up_buf,
        act_buf,
        f.gate.rows as u32,
    )?;
    if let Some(plan) = f.fold_act {
        plan.check(f.gate.rows)?;
        crate::hadamard::encode_fold(
            encoder,
            device,
            act_buf,
            f.gate.rows,
            plan,
            act_signs.as_ref().map(|b| &*b.buffer),
        )?;
    }
    encode_matvec(encoder, device, f.down, &down_w, act_buf, ffn_out)?;
    crate::elem::encode_vec_add(encoder, device, h_buf, ffn_out, hidden as u32)?;
    Ok(h_buf)
}

/// One layer encoded into a caller's command buffer, with its scratch
/// handed back so the caller can hold it until that buffer completes.
///
/// The ONE encoder: [`launch_gdn_branch`] wraps it in a command buffer
/// of its own and waits, and [`GdnRun`] gives a whole run of layers one
/// command buffer each and waits ONCE at the end. A second copy of this
/// sequence is the thing this repo keeps paying for.
///
/// # Safety
///
/// As [`launch_gdn_branch`]: the two state pointers must address that
/// many readable-writable, page-aligned bytes, held exclusively by the
/// caller until the command buffer this encodes into has completed.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_layer(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    w: &BranchWeights<'_>,
    ssm_ptr: *mut f32,
    ssm_bytes: usize,
    conv_ptr: *mut f32,
    conv_bytes: usize,
    conv_len: usize,
    qkv: Option<&[f32]>,
    z: Option<&[f32]>,
    beta_in: Option<&[f32]>,
    alpha_in: Option<&[f32]>,
    residual: Option<&[f32]>,
    h_ext: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<(EncodedLayer, ScratchSet), MetalError> {
    let h = w.head;
    let (key_dim, value_dim, conv_dim) = (h.key_dim(), h.value_dim(), h.conv_dim());
    if !w.is_supported()
        || conv_len != h.conv_state_len()
        // The four projections arrive from the host exactly when the
        // head is NOT on the device, so a caller cannot half-fuse.
        || w.head_in.is_some() == qkv.is_some()
        || qkv.is_some_and(|x| x.len() != conv_dim)
        || z.is_some_and(|x| x.len() != value_dim)
        || beta_in.is_some_and(|x| x.len() != h.n_v_heads)
        || alpha_in.is_some_and(|x| x.len() != h.n_v_heads)
        || qkv.is_some() != z.is_some()
        || qkv.is_some() != beta_in.is_some()
        || qkv.is_some() != alpha_in.is_some()
        // The residual stream is needed by the FFN, and by the head as
        // the vector it norms. It comes from ONE of two places -- the
        // host (`residual`) or a buffer the caller already holds
        // (`h_ext`, which is how a RUN passes it from layer to layer)
        // -- so a caller cannot ask for a fused layer and forget what
        // to add the branch to, and the length is this call's business
        // only when the bytes travel.
        || ((w.ffn.is_some() || w.head_in.is_some())
            && residual.is_none()
            && h_ext.is_none())
        || (h_ext.is_none() && residual.is_some_and(|r| r.len() != w.out_proj.rows))
    {
        return Err(MetalError::CommandFailed);
    }

    let shared = shared_metal()?;
    let device = &shared.device;

    // SAFETY: the caller's contract, forwarded.
    let ssm_buf = unsafe { buffer_no_copy(device, ssm_ptr, ssm_bytes) }
        .ok_or(MetalError::BufferAllocFailed)?;
    // The layer's CONSTANTS are resident, cached by address: `conv1d`
    // alone is `d_conv * conv_dim` floats, 164 KB on Bonsai, and
    // uploading it per layer per token is 7.9 MB a token of pure copy
    // for bytes that never change. The first version of this launch
    // did exactly that for five operands and measured SLOWER than the
    // host body it replaced, which is what the ledger's GPU time going
    // UP by 3.1 ms was.
    let taps_buf = crate::gpu::resident_f32_buffer(device, w.conv1d)?;
    let dt_buf = crate::gpu::resident_f32_buffer(device, w.dt_bias)?;
    let a_buf = crate::gpu::resident_f32_buffer(device, w.a)?;
    let norm_buf = crate::gpu::resident_f32_buffer(device, w.ssm_norm)?;
    // SAFETY: the caller's contract, forwarded.
    let conv_buf = unsafe { buffer_no_copy(device, conv_ptr, conv_bytes) }
        .ok_or(MetalError::BufferAllocFailed)?;
    // Scratch from the pool rather than six fresh allocations: `[q | k
    // | v]` in one buffer, which is the layout the convolution writes
    // and the layout the delta step reads by offset, so nothing is
    // copied between those two either. Returned when `scratch` drops,
    // which is after the wait below.
    let scratch = crate::scratch_pool::Scratch::take(
        device,
        &[
            conv_dim,
            h.n_v_heads,
            h.n_v_heads,
            value_dim,
            value_dim,
            w.out_proj.rows,
            // The per-token inputs, which really do have to travel: 66
            // KB on Bonsai, against the 123 KB of convolution window
            // that does not. They travel as a memcpy into a pooled
            // buffer rather than as four allocations.
            conv_dim,
            value_dim,
            h.n_v_heads,
            h.n_v_heads,
        ],
    )
    .ok_or(MetalError::BufferAllocFailed)?;
    let hidden = w.out_proj.rows;
    // The head's own scratch: one buffer for `attn_norm(x)`, which the
    // two gate projections read BEFORE it is rotated in place and the
    // other two read after.
    let head_scratch = match &w.head_in {
        None => None,
        // Two: the residual as it arrives, and the norm of it. Not
        // one in place -- an RMS norm reduces the whole vector before
        // it writes any of it, and a kernel reading a buffer it is
        // writing is a hazard nothing here would report.
        Some(_) => Some(
            crate::scratch_pool::Scratch::take(device, &[hidden, hidden])
                .ok_or(MetalError::BufferAllocFailed)?,
        ),
    };
    // The FFN's own scratch, taken only when there is an FFN: the
    // residual (which becomes the layer's output), the normed copy, the
    // two projections, the activation and the FFN output.
    let ffn_scratch = match &w.ffn {
        None => None,
        Some(f) => Some(
            crate::scratch_pool::Scratch::take(
                device,
                &[hidden, hidden, f.gate.rows, f.up.rows, f.gate.rows, hidden],
            )
            .ok_or(MetalError::BufferAllocFailed)?,
        ),
    };
    let conv_out = scratch.buf(0);
    let (beta_buf, g_buf) = (scratch.buf(1), scratch.buf(2));
    let o_buf = scratch.buf(3);
    let y_buf = scratch.buf(4);
    let out_buf = scratch.buf(5);
    // The four projection OUTPUTS: filled from the host when the head
    // is not fused, written by matvecs below when it is.
    let (qkv_buf, z_buf, beta_in_buf, alpha_in_buf) = match (qkv, z, beta_in, alpha_in) {
        (Some(a), Some(b), Some(c), Some(d)) => {
            let fill = |i: usize, xs: &[f32]| scratch.write(i, xs).ok_or(MetalError::CommandFailed);
            (fill(6, a)?, fill(7, b)?, fill(8, c)?, fill(9, d)?)
        }
        _ => (
            scratch.buf(6),
            scratch.buf(7),
            scratch.buf(8),
            scratch.buf(9),
        ),
    };
    let weights_buf = resident_weight_buffer(device, w.out_proj.weights)?;
    let signs_buf = match w.fold_y.and_then(|p| p.signs) {
        None => None,
        Some(signs) => {
            // SAFETY: a `&[f32]` viewed as its own bytes, for a read.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    signs.as_ptr() as *const u8,
                    std::mem::size_of_val(signs),
                )
            };
            Some(resident_weight_buffer(device, bytes)?)
        }
    };

    // The layer's HEAD, when the caller owns it: the norm and the four
    // projections, which is the second of a recurrent layer's three
    // submissions.
    if let (Some(hd), Some(sc), Some(res)) = (&w.head_in, &head_scratch, residual) {
        let x_buf = match h_ext {
            Some(b) => b,
            None => sc.write(0, res).ok_or(MetalError::CommandFailed)?,
        };
        let normed = sc.buf(1);
        let norm_w = crate::gpu::resident_f32_buffer(device, hd.norm)?;
        crate::norm::encode_rms_norm(
            encoder,
            device,
            x_buf,
            &norm_w.buffer,
            normed,
            hidden as u32,
            hd.norm_eps,
        )?;
        // BEFORE the rotation: the two gate projections are stored
        // unfolded, so they read `attn_norm(x)` in the primal basis,
        // and the fold below rewrites that buffer in place.
        let beta_w = resident_weight_buffer(device, hd.beta.weights)?;
        let alpha_w = resident_weight_buffer(device, hd.alpha.weights)?;
        encode_matvec(encoder, device, hd.beta, &beta_w, normed, beta_in_buf)?;
        encode_matvec(encoder, device, hd.alpha, &alpha_w, normed, alpha_in_buf)?;
        if let Some(plan) = hd.fold_x {
            plan.check(hidden)?;
            let signs = match plan.signs {
                None => None,
                Some(sg) => {
                    // SAFETY: a `&[f32]` viewed as its own bytes, read only.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            sg.as_ptr() as *const u8,
                            std::mem::size_of_val(sg),
                        )
                    };
                    Some(resident_weight_buffer(device, bytes)?)
                }
            };
            crate::hadamard::encode_fold(
                encoder,
                device,
                normed,
                hidden,
                plan,
                signs.as_ref().map(|b| &*b.buffer),
            )?;
        }
        let qkv_w = resident_weight_buffer(device, hd.qkv.weights)?;
        let z_w = resident_weight_buffer(device, hd.z.weights)?;
        encode_matvec(encoder, device, hd.qkv, &qkv_w, normed, qkv_buf)?;
        encode_matvec(encoder, device, hd.z, &z_w, normed, z_buf)?;
    }
    encode_gates(
        encoder,
        device,
        h.n_v_heads,
        beta_in_buf,
        alpha_in_buf,
        &dt_buf.buffer,
        &a_buf.buffer,
        beta_buf,
        g_buf,
    )?;
    encode_conv_silu(
        encoder,
        device,
        h,
        &conv_buf,
        &taps_buf.buffer,
        qkv_buf,
        conv_out,
    )?;
    // Q and K only, and in ONE dispatch: they are the first
    // `2 * n_k_heads` heads of the `[q | k | v]` buffer the convolution
    // wrote, contiguous, and V is that output as it stands. Two
    // dispatches over disjoint halves of one buffer also read as a
    // conflict to any barrier scheme that is per-buffer, so this
    // removes a dispatch AND a false dependency.
    encode_l2_norm_heads(
        encoder,
        device,
        conv_out,
        0,
        h.head_dim,
        2 * h.n_k_heads,
        w.eps,
    )?;
    encode_delta_step_at(
        encoder,
        device,
        w.shape,
        &ssm_buf,
        (conv_out, 0),
        (conv_out, key_dim),
        (conv_out, 2 * key_dim),
        (g_buf, 0),
        (beta_buf, 0),
        (o_buf, 0),
    )?;
    encode_gated_norm(
        encoder,
        device,
        h.n_v_heads,
        h.head_dim,
        w.eps,
        o_buf,
        z_buf,
        &norm_buf.buffer,
        y_buf,
    )?;
    if let Some(plan) = w.fold_y {
        plan.check(value_dim)?;
        crate::hadamard::encode_fold(
            encoder,
            device,
            y_buf,
            value_dim,
            plan,
            signs_buf.as_ref().map(|b| &*b.buffer),
        )?;
    }
    encode_matvec(encoder, device, w.out_proj, &weights_buf, y_buf, out_buf)?;

    // The rest of the layer, when the caller owns it: the whole point
    // of the file, because none of this needs the host and each piece
    // of it used to cost a submission.
    // The value of this match is the encoding; which buffer holds the
    // result is reported through `EncodedLayer` instead, because a run
    // of layers keeps that buffer and does not read it here.
    let _ = match (&w.ffn, &ffn_scratch, residual) {
        (Some(f), Some(sc), Some(res)) => {
            encode_ffn_tail(encoder, device, f, sc, out_buf, res, h_ext, hidden)?
        }
        _ => out_buf,
    };
    Ok((
        EncodedLayer {
            result: match (&w.ffn, h_ext.is_some()) {
                // A full layer's output IS the residual stream, and a
                // RUN passes that buffer in: then there is nothing here
                // to read and the run reads it once at the end.
                (Some(_), true) => ResultBuf::Caller,
                (Some(_), false) => ResultBuf::Ffn,
                (None, _) => ResultBuf::OutProj,
            },
            rows: if w.ffn.is_some() {
                hidden
            } else {
                w.out_proj.rows
            },
        },
        ScratchSet {
            main: scratch,
            head: head_scratch,
            ffn: ffn_scratch,
        },
    ))
}

/// What [`encode_layer`] wrote, and where.
struct EncodedLayer {
    result: ResultBuf,
    rows: usize,
}

/// Which buffer holds the layer's output.
enum ResultBuf {
    /// The residual stream the caller supplied; nothing to read here.
    Caller,
    /// This layer's own residual buffer, a full layer with no caller
    /// stream.
    Ffn,
    /// `ssm_out`'s output: a branch without an FFN.
    OutProj,
}

/// The pooled buffers one encoded layer is still using. Held by the
/// caller until the command buffer completes, and returned to the pool
/// when dropped -- which is what makes the pool safe with a command
/// buffer that has not been waited for yet.
struct ScratchSet {
    main: crate::scratch_pool::Scratch,
    #[allow(dead_code)]
    head: Option<crate::scratch_pool::Scratch>,
    ffn: Option<crate::scratch_pool::Scratch>,
}

impl ScratchSet {
    /// The buffer an encoded layer's output is in, or `None` when it is
    /// the caller's own stream.
    fn result<'b>(&'b self, which: &ResultBuf) -> Option<&'b ProtocolObject<dyn MTLBuffer>> {
        match which {
            ResultBuf::Caller => None,
            ResultBuf::Ffn => self.ffn.as_ref().map(|s| s.buf(0)),
            ResultBuf::OutProj => Some(self.main.buf(5)),
        }
    }
}

/// One token through the whole branch, in one submission.
///
/// BOTH states are page-aligned host allocations wrapped in place and
/// never copied. `qkv` / `z` / `beta_in` / `alpha_in` arrive from the
/// host exactly when `head_in` is `None`.
///
/// # Safety
///
/// Each pointer must address that many readable-writable, page-aligned
/// bytes which outlive the call and which the caller holds exclusively
/// across it. The GPU writes those bytes rather than a copy of them,
/// and this call waits for it.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gdn_branch(
    w: &BranchWeights<'_>,
    ssm_ptr: *mut f32,
    ssm_bytes: usize,
    conv_ptr: *mut f32,
    conv_bytes: usize,
    conv_len: usize,
    qkv: Option<&[f32]>,
    z: Option<&[f32]>,
    beta_in: Option<&[f32]>,
    alpha_in: Option<&[f32]>,
    residual: Option<&[f32]>,
    h_ext: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<Vec<f32>, MetalError> {
    let shared = shared_metal()?;
    let queue = &shared.queue;
    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    // SAFETY: the caller's contract, forwarded unchanged.
    let (done, keep) = unsafe {
        encode_layer(
            &encoder, w, ssm_ptr, ssm_bytes, conv_ptr, conv_bytes, conv_len, qkv, z, beta_in,
            alpha_in, residual, h_ext,
        )
    }?;
    encoder.endEncoding();
    let clock = crate::timing::SubmitClock::start();
    let label = match (w.head_in.is_some(), w.ffn.is_some()) {
        (true, true) => "gdn-layer-full",
        (false, true) => "gdn-layer",
        _ => "gdn-branch",
    };
    crate::timing::commit_wait_note(&cmd_buf, label, 32, clock);

    let result_buf = keep.result(&done.result).ok_or(MetalError::CommandFailed)?;
    // SAFETY: shared storage of exactly this many floats, written by
    // kernels this call has waited for. The convolution window needs no
    // read-back: the kernel wrote the caller's own bytes.
    unsafe {
        Ok(
            std::slice::from_raw_parts(result_buf.contents().as_ptr() as *const f32, done.rows)
                .to_vec(),
        )
    }
}

/// A RUN of consecutive full recurrent layers that shares one residual
/// buffer and waits ONCE.
///
/// # Why
///
/// With a layer down to a single command buffer, the ledger says the
/// token is GPU time plus the OS wake-up from `waitUntilCompleted`,
/// about 0.17 ms per submission and nothing else. Nothing requires the
/// host to wait per layer: consecutive recurrent layers pass their
/// residual stream to each other and the host does not look at it, so
/// the buffers can be committed back to back and waited for once. The
/// GPU runs them in order because one queue is ordered.
///
/// What that costs is care with the pool: a command buffer that has not
/// been waited for is still going to READ its scratch, so every layer's
/// scratch is held here until [`Self::finish`] returns. Dropping it
/// earlier would hand a live buffer to the next layer.
pub struct GdnRun {
    hidden: crate::scratch_pool::Scratch,
    keep: Vec<ScratchSet>,
    last: Option<objc2::rc::Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    rows: usize,
    submissions: usize,
    /// The NEXT attention layer's projections, when they were encoded
    /// at the end of this run: the scratch holding `q`, `k` and `v`,
    /// and their lengths. `finish` reads them back with the hidden
    /// state, so the host gets all four for one wait.
    head: Option<(crate::scratch_pool::Scratch, usize, usize)>,
}

impl GdnRun {
    /// Starts a run with `hidden` as the residual stream.
    pub fn start(hidden: &[f32]) -> Result<Self, MetalError> {
        let shared = shared_metal()?;
        let scratch = crate::scratch_pool::Scratch::take(&shared.device, &[hidden.len()])
            .ok_or(MetalError::BufferAllocFailed)?;
        scratch.write(0, hidden).ok_or(MetalError::CommandFailed)?;
        Ok(Self {
            hidden: scratch,
            keep: Vec::new(),
            last: None,
            rows: hidden.len(),
            submissions: 0,
            head: None,
        })
    }

    /// One more layer, committed but NOT waited for.
    ///
    /// # Safety
    ///
    /// As [`launch_gdn_branch`], and for longer: the two state pointers
    /// must stay valid and exclusively the caller's until
    /// [`Self::finish`] returns, because the GPU has not necessarily
    /// touched them yet when this returns.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn layer(
        &mut self,
        w: &BranchWeights<'_>,
        ssm_ptr: *mut f32,
        ssm_bytes: usize,
        conv_ptr: *mut f32,
        conv_bytes: usize,
        conv_len: usize,
    ) -> Result<(), MetalError> {
        // A run only carries layers that are whole: the residual stream
        // is what one hands the next, and a branch without an FFN does
        // not produce one.
        if w.ffn.is_none() || w.head_in.is_none() {
            return Err(MetalError::CommandFailed);
        }
        let shared = shared_metal()?;
        let cmd_buf = shared
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandFailed)?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?;
        // SAFETY: the caller's contract, forwarded; the scratch this
        // returns is held in `self.keep` until `finish`.
        let (_, keep) = unsafe {
            encode_layer(
                &encoder,
                w,
                ssm_ptr,
                ssm_bytes,
                conv_ptr,
                conv_bytes,
                conv_len,
                None,
                None,
                None,
                None,
                // The residual is already in the run's own buffer, so
                // nothing travels from the host: `h_ext` is where both
                // the head and the FFN read and write it.
                Some(&[]),
                Some(self.hidden.buf(0)),
            )
        }?;
        encoder.endEncoding();
        cmd_buf.commit();
        self.keep.push(keep);
        self.last = Some(cmd_buf);
        self.submissions += 1;
        Ok(())
    }

    /// An ATTENTION layer's tail at the head of this run: `wo`, the
    /// residual add, the FFN norm, the FFN and the second residual add,
    /// committed but NOT waited for.
    ///
    /// A hybrid alternates three recurrent layers and one attention
    /// layer, and the attention layer's tail feeds the next three. Its
    /// output is the residual stream they read, which the host never
    /// looks at, so it belongs in their command buffer rather than one
    /// of its own: sixteen waits a token.
    ///
    /// `residual` is ignored when the run already holds the stream --
    /// which it does from [`Self::start`] -- and the tail writes back
    /// into that same buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_tail(
        &mut self,
        out_proj: &MatvecLaunch<'_>,
        fold_branch: Option<&FoldPlan<'_>>,
        ffn: &LayerFfn<'_>,
        branch: &[f32],
    ) -> Result<(), MetalError> {
        let hidden = self.rows;
        let branch_cols = out_proj.row_bytes / out_proj.block_bytes * out_proj.block_elems;
        if out_proj.rows != hidden
            || branch.len() != branch_cols
            || ffn.norm.len() != hidden
            || ffn.gate.rows != ffn.up.rows
            || ffn.down.rows != hidden
        {
            return Err(MetalError::CommandFailed);
        }
        let shared = shared_metal()?;
        let device = &shared.device;
        let scratch = crate::scratch_pool::Scratch::take(device, &[branch_cols, hidden])
            .ok_or(MetalError::BufferAllocFailed)?;
        let branch_buf = scratch.write(0, branch).ok_or(MetalError::CommandFailed)?;
        let proj_buf = scratch.buf(1);
        let tail = crate::scratch_pool::Scratch::take(
            device,
            &[
                hidden,
                hidden,
                ffn.gate.rows,
                ffn.up.rows,
                ffn.gate.rows,
                hidden,
            ],
        )
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
        let cmd_buf = shared
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandFailed)?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?;
        encode_attn_tail(
            &encoder,
            device,
            out_proj,
            &out_w,
            fold_branch,
            signs.as_ref().map(|b| &*b.buffer),
            ffn,
            branch_buf,
            branch_cols,
            proj_buf,
            &tail,
            &[],
            // The run's own stream: the tail reads it as the residual
            // and writes the layer's output back into it.
            Some(self.hidden.buf(0)),
            hidden,
        )?;
        encoder.endEncoding();
        cmd_buf.commit();
        self.keep.push(ScratchSet {
            main: scratch,
            head: None,
            ffn: Some(tail),
        });
        self.last = Some(cmd_buf);
        self.submissions += 1;
        Ok(())
    }

    /// The NEXT layer's `attn_norm` and its Q/K/V projections, at the
    /// END of this run.
    ///
    /// An attention layer reads `attn_norm(hidden)` and projects it,
    /// and `hidden` is what this run just finished writing. Doing it
    /// here costs no wait of its own: the host gets `q`, `k` and `v`
    /// back from [`Self::finish`] alongside the residual stream, and
    /// its attention can start immediately.
    ///
    /// The Q projection is whatever width the caller's launch says --
    /// a gated Q is double width and the caller splits it.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_head(
        &mut self,
        norm: &[f32],
        norm_eps: f32,
        q: &MatvecLaunch<'_>,
        k: &MatvecLaunch<'_>,
        v: &MatvecLaunch<'_>,
        fold_x: Option<&FoldPlan<'_>>,
    ) -> Result<(), MetalError> {
        let hidden = self.rows;
        if norm.len() != hidden || self.head.is_some() {
            return Err(MetalError::CommandFailed);
        }
        let shared = shared_metal()?;
        let device = &shared.device;
        // normed, then q, k, v.
        let sc = crate::scratch_pool::Scratch::take(device, &[hidden, q.rows, k.rows, v.rows])
            .ok_or(MetalError::BufferAllocFailed)?;
        let normed = sc.buf(0);
        let norm_w = crate::gpu::resident_f32_buffer(device, norm)?;
        let (q_w, k_w, v_w) = (
            resident_weight_buffer(device, q.weights)?,
            resident_weight_buffer(device, k.weights)?,
            resident_weight_buffer(device, v.weights)?,
        );
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
        let cmd_buf = shared
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandFailed)?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?;
        crate::norm::encode_rms_norm(
            &encoder,
            device,
            self.hidden.buf(0),
            &norm_w.buffer,
            normed,
            hidden as u32,
            norm_eps,
        )?;
        if let Some(plan) = fold_x {
            plan.check(hidden)?;
            crate::hadamard::encode_fold(
                &encoder,
                device,
                normed,
                hidden,
                plan,
                signs.as_ref().map(|b| &*b.buffer),
            )?;
        }
        encode_matvec(&encoder, device, q, &q_w, normed, sc.buf(1))?;
        encode_matvec(&encoder, device, k, &k_w, normed, sc.buf(2))?;
        encode_matvec(&encoder, device, v, &v_w, normed, sc.buf(3))?;
        encoder.endEncoding();
        cmd_buf.commit();
        self.last = Some(cmd_buf);
        self.submissions += 1;
        self.head = Some((sc, q.rows, k.rows));
        Ok(())
    }

    /// A whole ATTENTION layer at the head of this run: the KV append,
    /// the attention, the gate, `wo`, the residual, the FFN norm, the
    /// FFN and the second residual, committed but NOT waited for.
    ///
    /// With this, a hybrid's four-layer group -- one attention layer
    /// and three recurrent -- is ONE wait, and nothing in it returns to
    /// the host.
    ///
    /// # Safety
    ///
    /// As [`Self::layer`]: the states this borrows must stay the
    /// caller's until [`Self::finish`] returns.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn_layer(
        &mut self,
        kv: &mut MetalKvBuffers,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: Option<&[f32]>,
        n_heads: usize,
        softcap: Option<f32>,
        out_proj: &MatvecLaunch<'_>,
        fold_branch: Option<&FoldPlan<'_>>,
        ffn: &LayerFfn<'_>,
    ) -> Result<(), MetalError> {
        let shared = shared_metal()?;
        let cmd_buf = shared
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandFailed)?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?;
        let hidden_buf = self.hidden.buf(0);
        encode_attn_layer(
            &encoder,
            kv,
            q,
            k,
            v,
            gate,
            n_heads,
            softcap,
            out_proj,
            fold_branch,
            ffn,
            // The stream is already in the run's buffer.
            &[],
            Some(hidden_buf),
            &mut self.keep,
        )?;
        encoder.endEncoding();
        cmd_buf.commit();
        self.last = Some(cmd_buf);
        self.submissions += 1;
        Ok(())
    }

    /// Waits for every layer committed so far and returns the residual
    /// stream.
    pub fn finish(self) -> Result<Vec<f32>, MetalError> {
        Ok(self.finish_with_head()?.0)
    }

    /// [`Self::finish`], also returning the Q/K/V that
    /// [`Self::attn_head`] encoded, when it did.
    #[allow(clippy::type_complexity)]
    pub fn finish_with_head(
        self,
    ) -> Result<(Vec<f32>, Option<(Vec<f32>, Vec<f32>, Vec<f32>)>), MetalError> {
        let read = |buf: &ProtocolObject<dyn MTLBuffer>, n: usize| -> Vec<f32> {
            // SAFETY: shared storage of exactly `n` floats, written by
            // kernels the wait below has completed.
            unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, n).to_vec() }
        };
        let Some(last) = &self.last else {
            // SAFETY: shared storage of exactly `rows` floats, written
            // by `start` and by nothing since.
            return Ok((read(self.hidden.buf(0), self.rows), None));
        };
        // One queue is ordered, so waiting for the LAST buffer waits
        // for every buffer before it.
        let clock = crate::timing::SubmitClock::start();
        crate::timing::note_wait(last, "gdn-run", self.submissions, 32, clock);
        let hidden = read(self.hidden.buf(0), self.rows);
        let head = self.head.as_ref().map(|(sc, q_rows, k_rows)| {
            (
                read(sc.buf(1), *q_rows),
                read(sc.buf(2), *k_rows),
                read(sc.buf(3), *k_rows),
            )
        });
        Ok((hidden, head))
    }
}

/// `wo` on the attention branch, then the FFN tail, encoded into the
/// caller's command buffer.
///
/// The ONE attention tail: [`launch_attn_tail`] wraps it in a command
/// buffer of its own and waits, and [`GdnRun::attn_tail`] puts it at
/// the head of a RUN, where it shares the single wait with the
/// recurrent layers that follow it.
#[allow(clippy::too_many_arguments)]
fn encode_attn_tail<'b>(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    device: &objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    out_proj: &MatvecLaunch<'_>,
    out_w: &crate::gpu::ResidentWeightBuffer,
    fold_branch: Option<&FoldPlan<'_>>,
    signs: Option<&ProtocolObject<dyn MTLBuffer>>,
    ffn: &LayerFfn<'_>,
    branch_buf: &ProtocolObject<dyn MTLBuffer>,
    branch_cols: usize,
    proj_buf: &ProtocolObject<dyn MTLBuffer>,
    tail: &'b crate::scratch_pool::Scratch,
    residual: &[f32],
    h_ext: Option<&'b ProtocolObject<dyn MTLBuffer>>,
    hidden: usize,
) -> Result<&'b ProtocolObject<dyn MTLBuffer>, MetalError> {
    if let Some(plan) = fold_branch {
        plan.check(branch_cols)?;
        crate::hadamard::encode_fold(encoder, device, branch_buf, branch_cols, plan, signs)?;
    }
    encode_matvec(encoder, device, out_proj, out_w, branch_buf, proj_buf)?;
    encode_ffn_tail(
        encoder, device, ffn, tail, proj_buf, residual, h_ext, hidden,
    )
}

/// An ATTENTION layer's tail in one command buffer: `wo`, the residual
/// add, the FFN norm, the FFN and the second residual add.
///
/// The recurrent layers are one submission each
/// ([`encode_layer`]); the attention layers between them still cost
/// three, because their attention runs on the host. Two of those three
/// are this: `wo` was its own `matvec-fused` submission and the FFN its
/// own `dense-ffn`, with nothing but a vector add and a norm in
/// between. It reuses [`encode_ffn_tail`], so the two layer kinds
/// cannot drift about what a layer tail is.
///
/// `branch` is the attention output BEFORE `wo`
/// (`Decoder::attn_branch_rows`), `residual` the stream as it entered
/// the layer.
pub fn launch_attn_tail(
    out_proj: &MatvecLaunch<'_>,
    fold_branch: Option<&FoldPlan<'_>>,
    ffn: &LayerFfn<'_>,
    branch: &[f32],
    residual: &[f32],
) -> Result<Vec<f32>, MetalError> {
    let hidden = residual.len();
    let branch_cols = out_proj.row_bytes / out_proj.block_bytes * out_proj.block_elems;
    if out_proj.rows != hidden
        || branch.len() != branch_cols
        || ffn.norm.len() != hidden
        || ffn.gate.rows != ffn.up.rows
        || ffn.down.rows != hidden
    {
        return Err(MetalError::CommandFailed);
    }
    let shared = shared_metal()?;
    let device = &shared.device;
    let scratch = crate::scratch_pool::Scratch::take(
        device,
        &[
            // the branch, `wo`'s output, and the FFN tail's six
            branch_cols,
            hidden,
            hidden,
            hidden,
            ffn.gate.rows,
            ffn.up.rows,
            ffn.gate.rows,
            hidden,
        ],
    )
    .ok_or(MetalError::BufferAllocFailed)?;
    let branch_buf = scratch.write(0, branch).ok_or(MetalError::CommandFailed)?;
    let proj_buf = scratch.buf(1);
    // The FFN tail indexes its own scratch from 0, so it gets a view
    // starting where its six buffers do.
    let tail = crate::scratch_pool::Scratch::take(
        device,
        &[
            hidden,
            hidden,
            ffn.gate.rows,
            ffn.up.rows,
            ffn.gate.rows,
            hidden,
        ],
    )
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

    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = &*encoder;
    let result = encode_attn_tail(
        encoder,
        device,
        out_proj,
        &out_w,
        fold_branch,
        signs.as_ref().map(|b| &*b.buffer),
        ffn,
        branch_buf,
        branch_cols,
        proj_buf,
        &tail,
        residual,
        None,
        hidden,
    )?;
    encoder.endEncoding();
    let clock = crate::timing::SubmitClock::start();
    crate::timing::commit_wait_note(&cmd_buf, "attn-tail", 32, clock);

    // SAFETY: shared storage of exactly `hidden` floats, written by
    // kernels this call has waited for.
    unsafe {
        Ok(std::slice::from_raw_parts(result.contents().as_ptr() as *const f32, hidden).to_vec())
    }
}

/// Everything an ATTENTION layer does after its projections, in ONE
/// command buffer: this token's K/V appended to the device KV, the
/// attention over it, the sigmoid gate, `wo`, the residual add, the FFN
/// norm, the FFN and the second residual add.
///
/// # Why
///
/// With the projections riding in the previous run
/// ([`GdnRun::attn_head`]) and the tail in the next
/// ([`GdnRun::attn_tail`]), the only thing an attention layer still
/// came back to the host for was the attention itself, because the KV
/// lived there. It does not have to: `kv` is the sequence's own device
/// mirror, appended one row a token and re-uploaded whenever the host
/// cache says it has drifted.
///
/// That removes the last host step in a decode token AND the last
/// growing one: the reference's decode is flat from 32 to 300 tokens
/// and this engine's was not, because attention on the host grows with
/// the context while a kernel over resident KV does not.
///
/// `gate` is the half of a double-width `wq` the caller split off; it
/// is `None` for a layer that does not gate.
#[allow(clippy::too_many_arguments)]
fn encode_attn_layer(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    kv: &mut MetalKvBuffers,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: Option<&[f32]>,
    n_heads: usize,
    softcap: Option<f32>,
    out_proj: &MatvecLaunch<'_>,
    fold_branch: Option<&FoldPlan<'_>>,
    ffn: &LayerFfn<'_>,
    residual: &[f32],
    h_ext: Option<&ProtocolObject<dyn MTLBuffer>>,
    keep: &mut Vec<ScratchSet>,
) -> Result<(), MetalError> {
    // `wo`'s row count IS the hidden width; `residual` is empty when
    // the run already holds the stream in `h_ext`.
    let hidden = out_proj.rows;
    let per_token = kv.elems_per_token();
    let branch_cols = out_proj.row_bytes / out_proj.block_bytes * out_proj.block_elems;
    if (h_ext.is_none() && residual.len() != hidden)
        || k.len() != per_token
        || v.len() != per_token
        || q.len() != n_heads * kv.head_dim
        || branch_cols != n_heads * kv.head_dim
        || gate.is_some_and(|g| g.len() != q.len())
        || kv.seq_len >= kv.capacity()
    {
        return Err(MetalError::CommandFailed);
    }
    let shared = shared_metal()?;
    let device = &shared.device;
    let scratch = crate::scratch_pool::Scratch::take(
        device,
        &[q.len(), per_token, per_token, q.len().max(1), hidden],
    )
    .ok_or(MetalError::BufferAllocFailed)?;
    let q_buf = scratch.write(0, q).ok_or(MetalError::CommandFailed)?;
    let k_buf = scratch.write(1, k).ok_or(MetalError::CommandFailed)?;
    let v_buf = scratch.write(2, v).ok_or(MetalError::CommandFailed)?;
    let gate_buf = match gate {
        None => None,
        Some(g) => Some(scratch.write(3, g).ok_or(MetalError::CommandFailed)?),
    };
    // The attention output, which is also `wo`'s input.
    let attn_buf = scratch.buf(4);
    let tail = crate::scratch_pool::Scratch::take(
        device,
        &[
            hidden,
            hidden,
            ffn.gate.rows,
            ffn.up.rows,
            ffn.gate.rows,
            hidden,
        ],
    )
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

    let mut mrs = crate::mem_ranges::MemRanges::new();
    // Append this token, then attend over everything including it.
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
    if let Some(g) = gate_buf {
        crate::elem::encode_sigmoid_mul(encoder, device, attn_buf, g, q.len() as u32)?;
    }
    // `wo` reads the gated attention output directly: the fold rewrites
    // that buffer in place and nothing reads it afterwards, so there is
    // nothing to copy.
    let branch_buf = attn_buf;
    encode_attn_tail(
        encoder,
        device,
        out_proj,
        &out_w,
        fold_branch,
        signs.as_ref().map(|b| &*b.buffer),
        ffn,
        branch_buf,
        branch_cols,
        proj.buf(0),
        &tail,
        residual,
        h_ext,
        hidden,
    )?;
    kv.seq_len = seq_len;
    keep.push(ScratchSet {
        main: scratch,
        head: Some(proj),
        ffn: Some(tail),
    });
    Ok(())
}

/// The whole attention layer in a command buffer of its own.
///
/// [`GdnRun::attn_layer`] puts the same encoding at the head of a run
/// instead, where it shares the run's single wait; this is the shape
/// for a layer with no recurrent layers behind it.
#[allow(clippy::too_many_arguments)]
pub fn launch_attn_layer(
    kv: &mut MetalKvBuffers,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: Option<&[f32]>,
    n_heads: usize,
    softcap: Option<f32>,
    out_proj: &MatvecLaunch<'_>,
    fold_branch: Option<&FoldPlan<'_>>,
    ffn: &LayerFfn<'_>,
    residual: &[f32],
) -> Result<Vec<f32>, MetalError> {
    let hidden = residual.len();
    let shared = shared_metal()?;
    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    // SERIAL, not concurrent: `encode_attn_tail` was written for an
    // encoder that orders its own dispatches, and on a concurrent one
    // its fold, matvecs, norm and adds run unordered. The first version
    // of this launch used a concurrent encoder and generated fluent
    // nonsense while `frink parity` stayed MATCH -- parity reads the
    // FIRST token, which is prefill, and this path is decode.
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    let mut keep = Vec::new();
    encode_attn_layer(
        &encoder,
        kv,
        q,
        k,
        v,
        gate,
        n_heads,
        softcap,
        out_proj,
        fold_branch,
        ffn,
        residual,
        None,
        &mut keep,
    )?;
    encoder.endEncoding();
    let clock = crate::timing::SubmitClock::start();
    crate::timing::commit_wait_note(&cmd_buf, "attn-layer", 32, clock);
    let out = keep
        .last()
        .and_then(|s| s.ffn.as_ref())
        .ok_or(MetalError::CommandFailed)?
        .buf(0);
    // SAFETY: shared storage of exactly `hidden` floats, written by
    // kernels this call has waited for.
    unsafe {
        Ok(std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, hidden).to_vec())
    }
}
