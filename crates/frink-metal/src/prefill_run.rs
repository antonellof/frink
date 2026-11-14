//! A prefill RUN: consecutive layers of one sequence's prompt batch in
//! ONE command buffer, the hidden batch uploaded once and read back
//! once. Every recurrent kind rides in it (the gated delta-net,
//! `crate::gdn_prefill`; Mamba-1, Mamba-2 and LFM2's short convolution,
//! `crate::ssm_prefill`) and so does the attention between them
//! (`crate::gdn_prefill_attn`), Falcon-H1's with its Mamba-2 block
//! beside it.

use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use crate::attn::MetalKvBuffers;
use crate::gdn_prefill::{
    encode_layer, encode_wo_tail, slot, Enc, PrefillAttnTail, PrefillFfn, PrefillGdnLayer,
    PrefillGdnState, RunOptions,
};
use crate::gdn_prefill_attn::PrefillAttnLayer;
use crate::gpu::{shared_metal, MatvecLaunch, MetalError};
use crate::ssm_prefill::{
    PrefillConvLayer, PrefillConvState, PrefillMamba1, PrefillMamba2, PrefillMambaLayer, SsmNeed,
};

/// One layer of a run: a recurrent layer with its state, or an
/// attention layer whole with the sequence's device KV mirror, which the
/// run appends this batch to.
pub enum PrefillRunLayer<'a, 'k> {
    Gdn(PrefillGdnLayer<'a>, PrefillGdnState),
    Attn(PrefillAttnLayer<'a>, &'k mut MetalKvBuffers),
    /// LFM2's short convolution, its history in and out.
    Conv(PrefillConvLayer<'a>, PrefillConvState<'k>),
    /// A Mamba-1 block where attention would be. The block holds its
    /// state (`crate::ssm_decode::Mamba1Mixer::new`).
    Mamba1(PrefillMambaLayer<'a, PrefillMamba1<'a>>),
    Mamba2(PrefillMambaLayer<'a, PrefillMamba2<'a>>),
    /// Attention AND a Mamba-2 block over the same normed rows, summed
    /// before the residual (`falcon-h1.cpp:137-161`).
    Parallel(
        PrefillAttnLayer<'a>,
        Box<PrefillMamba2<'a>>,
        &'k mut MetalKvBuffers,
    ),
}

impl PrefillRunLayer<'_, '_> {
    fn hidden(&self) -> usize {
        match self {
            Self::Gdn(l, _) => l.attn_norm.len(),
            Self::Attn(l, _) | Self::Parallel(l, ..) => l.attn_norm.len(),
            Self::Conv(l, _) => l.attn_norm.len(),
            Self::Mamba1(l) => l.attn_norm.len(),
            Self::Mamba2(l) => l.attn_norm.len(),
        }
    }

    fn is_supported(&self, rows: usize, start_pos: usize) -> bool {
        match self {
            Self::Gdn(l, _) => l.is_supported(),
            Self::Attn(l, kv) => l.is_supported(kv, rows, start_pos),
            Self::Parallel(l, m, kv) => {
                let n = l.attn_norm.len();
                // The block reads the normed rows before any rotation.
                l.fold_x.is_none()
                    && m.in_cols() == n
                    && m.out_rows() == n
                    && l.is_supported(kv, rows, start_pos)
            }
            Self::Conv(l, s) => l.is_supported(s),
            Self::Mamba1(l) => l.is_supported(),
            Self::Mamba2(l) => l.is_supported(),
        }
    }

    fn ffn(&self) -> Option<&PrefillFfn<'_>> {
        match self {
            Self::Gdn(g, _) => Some(&g.ffn),
            Self::Attn(a, _) | Self::Parallel(a, ..) => Some(&a.ffn),
            Self::Conv(c, _) => c.ffn.as_ref(),
            Self::Mamba1(m) => m.ffn.as_ref(),
            Self::Mamba2(m) => m.ffn.as_ref(),
        }
    }

    fn ssm_need(&self) -> SsmNeed {
        match self {
            Self::Conv(c, _) => SsmNeed::of_conv(c),
            Self::Mamba1(m) => SsmNeed::of_mamba1(&m.block),
            Self::Mamba2(m) => SsmNeed::of_mamba2(&m.block),
            Self::Parallel(_, m, _) => SsmNeed::of_mamba2(m),
            Self::Gdn(..) | Self::Attn(..) => SsmNeed::default(),
        }
    }
}

/// The model's output head on the batch's LAST row, when the run ends
/// the stack: `rms_norm(h) * norm`, then `lm_head`, in the run's own
/// command buffer. A prefill only samples from the last row, and as a
/// second submission this cost LFM2-1.2B's 512-token prompt 0.5 ms of
/// its 33 for 0.2 ms of GPU work. Raw logits: the caller applies what
/// its host head would after the matvec (`decoder::lm_head`). With a
/// head the run returns the logits INSTEAD of the hidden batch, which a
/// caller sampling the last row has no use for (4 MB read back there).
pub struct PrefillOutputHead<'a> {
    pub norm: &'a std::sync::Arc<[f32]>,
    pub eps: f32,
    pub lm_head: MatvecLaunch<'a>,
}

impl PrefillOutputHead<'_> {
    fn is_supported(&self, hidden: usize) -> bool {
        self.norm.len() == hidden
            && self.lm_head.rows > 0
            && self.lm_head.row_bytes / self.lm_head.block_bytes * self.lm_head.block_elems
                == hidden
    }
}

/// [`launch_prefill_run_with_head`] without an output head.
///
/// # Safety
///
/// As [`launch_prefill_run_with_head`]'s.
pub unsafe fn launch_prefill_run(
    hidden_batch: &[f32],
    rows: usize,
    start_pos: usize,
    head: Option<&PrefillAttnTail<'_>>,
    layers: &mut [PrefillRunLayer<'_, '_>],
    opts: RunOptions,
) -> Result<Vec<f32>, MetalError> {
    // SAFETY: the caller's contract, forwarded.
    unsafe { launch_prefill_run_with_head(hidden_batch, rows, start_pos, head, layers, opts, None) }
        .map(|(h, _)| h)
}

/// `rows` tokens of one sequence, at positions `start_pos..`, through
/// `layers`, in order, in ONE command buffer, the hidden batch uploaded
/// once and read back once. Returns the batch as it leaves the last
/// layer, or with `output` the last row's raw logits and no batch; each
/// attention layer's mirror has then grown by `rows`, and each conv
/// layer's `bx_out` has been handed its new history rows.
///
/// Every layer is checked before anything is encoded, so an `Err` has
/// touched no state.
///
/// # Safety
///
/// Each recurrent state's two pointers must address that many
/// readable-writable, page-aligned bytes, held exclusively by the
/// caller across the call, which waits for the GPU: the GPU writes
/// those very bytes.
pub unsafe fn launch_prefill_run_with_head(
    hidden_batch: &[f32],
    rows: usize,
    start_pos: usize,
    head: Option<&PrefillAttnTail<'_>>,
    layers: &mut [PrefillRunLayer<'_, '_>],
    opts: RunOptions,
    output: Option<&PrefillOutputHead<'_>>,
) -> Result<(Vec<f32>, Option<Vec<f32>>), MetalError> {
    let hidden = match (head, layers.first()) {
        (Some(t), _) => t.o_proj.rows,
        (None, Some(first)) => first.hidden(),
        (None, None) => return Err(MetalError::CommandFailed),
    };
    if rows == 0
        || hidden_batch.len() != rows * hidden
        || head.is_some_and(|t| !t.is_supported(rows, hidden))
        || output.is_some_and(|o| !o.is_supported(hidden))
        || layers
            .iter()
            .any(|l| !l.is_supported(rows, start_pos) || l.hidden() != hidden)
    {
        return Err(MetalError::CommandFailed);
    }
    let gdn = |f: fn(&PrefillGdnLayer<'_>) -> usize| {
        layers
            .iter()
            .filter_map(|l| match l {
                PrefillRunLayer::Gdn(g, _) => Some(f(g)),
                _ => None,
            })
            .max()
            .unwrap_or(0)
            .max(1)
            * rows
    };
    let attn = |f: &dyn Fn(&PrefillAttnLayer<'_>, &MetalKvBuffers) -> usize| {
        layers
            .iter()
            .filter_map(|l| match l {
                PrefillRunLayer::Attn(a, kv) | PrefillRunLayer::Parallel(a, _, kv) => {
                    Some(f(a, kv))
                }
                _ => None,
            })
            .max()
            .unwrap_or(0)
            .max(1)
            * rows
    };
    let mut lens = [0usize; slot::COUNT];
    lens[slot::H] = rows * hidden;
    lens[slot::XN] = rows * hidden;
    lens[slot::QKV] = gdn(|l| l.head.conv_dim());
    lens[slot::Z] = gdn(|l| l.head.value_dim());
    lens[slot::BETA_IN] = gdn(|l| l.head.n_v_heads);
    lens[slot::ALPHA_IN] = lens[slot::BETA_IN];
    lens[slot::BETA] = lens[slot::BETA_IN];
    lens[slot::G] = lens[slot::BETA_IN];
    lens[slot::CONV] = lens[slot::QKV];
    lens[slot::O] = lens[slot::Z];
    lens[slot::Y] = lens[slot::Z];
    lens[slot::BRANCH] = head.map_or(1, |t| t.branch.len());
    lens[slot::OUT] = rows * hidden;
    let ffn_rows = layers
        .iter()
        .filter_map(|l| l.ffn().map(|f| f.gate.rows))
        .chain(head.map(|t| t.ffn.gate.rows))
        .max()
        .unwrap_or(1);
    lens[slot::GATE] = ffn_rows * rows;
    lens[slot::UP] = lens[slot::GATE];
    lens[slot::AQ] = attn(&|a, _| a.q.rows);
    lens[slot::AQR] = attn(&|a, kv| a.n_heads * kv.head_dim);
    lens[slot::AG] = lens[slot::AQR];
    lens[slot::AATT] = lens[slot::AQR];
    lens[slot::AK] = attn(&|_, kv| kv.elems_per_token());
    lens[slot::AV] = lens[slot::AK];
    let routed = layers
        .iter()
        .filter_map(|l| l.ffn())
        .chain(head.map(|t| &t.ffn))
        .any(|f| f.routed.is_some());
    // Half the floats would hold the f16 copy; a whole width keeps the
    // pool's sizes few.
    lens[slot::XN16] = if routed { rows * hidden } else { 1 };
    lens[slot::MOE_OUT] = lens[slot::XN16];
    let ssm = layers
        .iter()
        .map(PrefillRunLayer::ssm_need)
        .fold(SsmNeed::default(), SsmNeed::max);
    for (i, n) in ssm.lens(rows) {
        lens[i] = n;
    }
    let parallel = layers
        .iter()
        .any(|l| matches!(l, PrefillRunLayer::Parallel(..)));
    lens[slot::SSM_OUT] = if parallel { rows * hidden } else { 1 };
    // Each conv layer's history region, `[l_cache - 1 + rows][n]`, in
    // layer order.
    let hist_at: Vec<usize> = layers
        .iter()
        .scan(0usize, |at, l| {
            let here = *at;
            if let PrefillRunLayer::Conv(c, _) = l {
                *at += c.hist_len(rows);
            }
            Some(here)
        })
        .collect();
    lens[slot::CONV_HIST] = layers
        .iter()
        .map(|l| match l {
            PrefillRunLayer::Conv(c, _) => c.hist_len(rows),
            _ => 0,
        })
        .sum::<usize>()
        .max(1);
    lens[slot::HEAD_X] = output.map_or(1, |_| hidden);
    lens[slot::LOGITS] = output.map_or(1, |o| o.lm_head.rows);

    let shared = shared_metal()?;
    let device = &shared.device;
    let sc =
        crate::scratch_pool::Scratch::take(device, &lens).ok_or(MetalError::BufferAllocFailed)?;
    sc.write(slot::H, hidden_batch)
        .ok_or(MetalError::CommandFailed)?;
    // Each conv layer's window ahead of the rows the run appends.
    let hist_ptr = sc.buf(slot::CONV_HIST).contents().as_ptr() as *mut f32;
    for (l, &at) in layers.iter().zip(&hist_at) {
        if let PrefillRunLayer::Conv(_, s) = l {
            // SAFETY: shared storage of `lens[CONV_HIST]` floats, which
            // holds this layer's `hist_len(rows)` floats from `at`, and
            // no command buffer that uses it has been committed yet.
            unsafe {
                std::ptr::copy_nonoverlapping(s.window.as_ptr(), hist_ptr.add(at), s.window.len())
            };
        }
    }
    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let mut enc = Enc {
        encoder: cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::CommandFailed)?,
        spans: crate::kernel_timing::SpanClock::begin(device, 26 * (layers.len() + 1)),
        cmd: &cmd_buf,
        residual_scale: opts.residual_scale,
    };
    let encoded = (|| {
        if let Some(t) = head {
            let branch = sc
                .write(slot::BRANCH, t.branch)
                .ok_or(MetalError::CommandFailed)?;
            encode_wo_tail(
                &mut enc,
                device,
                &t.o_proj,
                t.fold.as_ref(),
                &t.ffn,
                t.eps,
                branch,
                None,
                &sc,
                rows,
            )?;
        }
        for (layer, &at) in layers.iter().zip(&hist_at) {
            match layer {
                // SAFETY: the caller's contract, forwarded per layer.
                PrefillRunLayer::Gdn(l, state) => {
                    unsafe { encode_layer(&mut enc, device, l, state, &sc, rows) }?
                }
                PrefillRunLayer::Attn(l, kv) => crate::gdn_prefill_attn::encode_attn_layer(
                    &mut enc, device, l, None, kv, &sc, rows, start_pos,
                )?,
                PrefillRunLayer::Parallel(l, m, kv) => crate::gdn_prefill_attn::encode_attn_layer(
                    &mut enc,
                    device,
                    l,
                    Some(m.as_ref()),
                    kv,
                    &sc,
                    rows,
                    start_pos,
                )?,
                PrefillRunLayer::Conv(l, _) => {
                    crate::ssm_prefill::encode_conv_layer(&mut enc, device, l, at, &sc, rows)?
                }
                PrefillRunLayer::Mamba1(l) => {
                    crate::ssm_prefill::encode_mamba1_layer(&mut enc, device, l, &sc, rows)?
                }
                PrefillRunLayer::Mamba2(l) => {
                    crate::ssm_prefill::encode_mamba2_layer(&mut enc, device, l, &sc, rows)?
                }
            }
        }
        if let Some(o) = output {
            encode_output_head(&mut enc, device, o, &sc, rows, hidden)?;
        }
        Ok(())
    })();
    // Ended either way; on an error nothing is committed and the buffer
    // is dropped unsubmitted, so no state has moved.
    enc.encoder.endEncoding();
    encoded?;
    let clock = crate::timing::SubmitClock::start();
    crate::timing::commit_wait_note(&cmd_buf, "gdn-prefill", 16, clock);
    enc.spans.finish(&cmd_buf, "gdn-prefill", 4);
    if cmd_buf.status() != objc2_metal::MTLCommandBufferStatus::Completed {
        return Err(MetalError::CommandFailed);
    }
    for (layer, &at) in layers.iter_mut().zip(&hist_at) {
        match layer {
            PrefillRunLayer::Attn(_, kv) | PrefillRunLayer::Parallel(_, _, kv) => {
                kv.seq_len += rows
            }
            PrefillRunLayer::Conv(c, s) => {
                let n = c.attn_norm.len();
                let from = at + (c.l_cache - 1) * n;
                // SAFETY: this layer's region of the shared history slot,
                // written by the command buffer waited for above.
                let bx = unsafe { std::slice::from_raw_parts(hist_ptr.add(from), rows * n) };
                (s.bx_out)(bx);
            }
            _ => {}
        }
    }
    // SAFETY: shared storage of `rows * hidden` floats (and of the
    // head's `lm_head.rows`), written by the kernels this call waited for.
    unsafe {
        let read = |i: usize, n: usize| {
            std::slice::from_raw_parts(sc.buf(i).contents().as_ptr() as *const f32, n).to_vec()
        };
        Ok(match output {
            // The caller asked for the logits, not the stream.
            Some(o) => (Vec::new(), Some(read(slot::LOGITS, o.lm_head.rows))),
            None => (read(slot::H, rows * hidden), None),
        })
    }
}

/// The last row of the residual stream normed into `HEAD_X`, and
/// `lm_head` of it into `LOGITS`.
fn encode_output_head(
    enc: &mut Enc<'_>,
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    o: &PrefillOutputHead<'_>,
    sc: &crate::scratch_pool::Scratch,
    rows: usize,
    hidden: usize,
) -> Result<(), MetalError> {
    let norm = crate::gpu::resident_f32_shared(device, o.norm)?;
    let w = crate::gpu::resident_weight_buffer(device, o.lm_head.weights)?;
    let encoder = enc.span("pf_output_norm")?;
    crate::norm::encode_rms_norm_at(
        encoder,
        device,
        sc.buf(slot::H),
        (rows - 1) * hidden * 4,
        &norm.buffer,
        sc.buf(slot::HEAD_X),
        0,
        hidden as u32,
        o.eps,
    )?;
    let encoder = enc.span("pf_lm_head")?;
    crate::gpu::encode_matvec(
        encoder,
        device,
        &o.lm_head,
        &w,
        sc.buf(slot::HEAD_X),
        sc.buf(slot::LOGITS),
    )
}
