//! The Mamba and short-convolution layers of a prefill RUN
//! (`crate::gdn_prefill`): every row of a prompt batch through
//! `attn_norm`, the block and the residual (and the FFN, where the layer
//! has one) with the hidden batch resident, so LFM2, Granite-4.0-H,
//! Falcon-H1, Nemotron-H and the pure Mamba models put a prompt on the
//! GPU in one command buffer, attention layers included.
//!
//! # Why
//!
//! Before this a hybrid's prompt went layer by layer through the host
//! bodies: each projection its own submission with the activations
//! uploaded and read back, the conv and the gates on the host, and a
//! Mamba layer's scan a submission of its own. On an M5 Max that was
//! 3-4x behind llama.cpp's `pp512` on all four models measured, with the
//! GPU idle most of the prompt.
//!
//! The blocks are the decode stack's own (`crate::ssm_decode`'s
//! `encode_mamba1` / `encode_mamba2` over [`SsmProj`], and
//! `crate::shortconv`'s conv), run over `rows` rows with GEMMs where the
//! token has matvecs, so the two cannot drift. The GEMMs read f32
//! activations, as the per-launch path's do, so the projections round as
//! they did before.
//!
//! # State
//!
//! The HOST owns every state before and after the run. A Mamba layer's
//! `RecurrentState` is wrapped in place and advanced by the scan and the
//! conv; a conv layer's history is the layer's `KvCache`, whose last
//! `l_cache - 1` rows the caller hands in ([`PrefillConvState::window`])
//! and whose new rows (`b * x`, one per token) are handed to
//! [`PrefillConvState::bx_out`] straight from the run's buffer, for the
//! caller to push into that cache: one copy, not one into a `Vec` and a
//! second into the cache (LFM2-1.2B's ten conv layers hand back 40 MB
//! for a 512-token prompt).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};
use std::sync::Arc;

use crate::gdn_prefill::{cols_of, encode_gemm, slot, Enc, PrefillFfn};
use crate::gpu::{resident_f32_shared, MetalError, MulMmSgLaunch};
use crate::mem_ranges::MemRanges;
use crate::ssm_decode::{
    encode_mamba1, encode_mamba2, Mamba1Bufs, Mamba1Mixer, Mamba2Bufs, Mamba2Mixer, ScratchNeed,
};

/// A Mamba-1 block with a GEMM per projection.
pub type PrefillMamba1<'a> = Mamba1Mixer<'a, MulMmSgLaunch<'a>>;
/// A Mamba-2 block with a GEMM per projection.
pub type PrefillMamba2<'a> = Mamba2Mixer<'a, MulMmSgLaunch<'a>>;

/// LFM2's short-convolution layer (`lfm2.cpp:139-208`).
pub struct PrefillConvLayer<'a> {
    pub attn_norm: &'a Arc<[f32]>,
    pub eps: f32,
    /// `[3 n_embd, n_embd]`, `[b | c | x]`.
    pub in_proj: MulMmSgLaunch<'a>,
    /// `[n_embd][l_cache]`, tap `l_cache - 1` on the newest input.
    pub taps: &'a Arc<[f32]>,
    pub l_cache: usize,
    pub out_proj: MulMmSgLaunch<'a>,
    pub ffn: Option<PrefillFfn<'a>>,
}

/// Where a conv layer's new history rows go: `[rows][n_embd]`, borrowed
/// from the run's buffer for the call.
pub type HistorySink<'k> = Box<dyn FnMut(&[f32]) + 'k>;

/// A conv layer's history in and out (see the module doc).
pub struct PrefillConvState<'k> {
    /// `[l_cache - 1][n_embd]`, the inputs before this batch, oldest
    /// first, zero rows before the sequence started.
    pub window: &'k [f32],
    /// Called once the run waited, with `[rows][n_embd]`, each row's
    /// `b * x`, borrowed from the run's buffer for the call.
    pub bx_out: HistorySink<'k>,
}

impl PrefillConvLayer<'_> {
    pub(crate) fn is_supported(&self, state: &PrefillConvState<'_>) -> bool {
        let n = self.attn_norm.len();
        n > 0
            && (2..=crate::shortconv::MAX_L_CACHE).contains(&self.l_cache)
            && self.in_proj.rows == 3 * n
            && cols_of(&self.in_proj) == n
            && self.out_proj.rows == n
            && cols_of(&self.out_proj) == n
            && self.taps.len() == n * self.l_cache
            && state.window.len() == (self.l_cache - 1) * n
            && self.ffn.as_ref().is_none_or(|f| f.fits(n))
    }

    /// Floats of the run's history slot this layer takes.
    pub(crate) fn hist_len(&self, rows: usize) -> usize {
        (self.l_cache - 1 + rows) * self.attn_norm.len()
    }
}

/// A Mamba layer: `attn_norm`, the block, the residual, and the FFN a
/// hybrid's layer has after it (none on a pure Mamba model).
pub struct PrefillMambaLayer<'a, M> {
    pub attn_norm: &'a Arc<[f32]>,
    pub eps: f32,
    pub block: M,
    pub ffn: Option<PrefillFfn<'a>>,
}

impl<'a> PrefillMambaLayer<'a, PrefillMamba1<'a>> {
    pub(crate) fn is_supported(&self) -> bool {
        let n = self.attn_norm.len();
        self.block.in_cols() == n
            && self.block.out_rows() == n
            && self.ffn.as_ref().is_none_or(|f| f.fits(n))
    }
}

impl<'a> PrefillMambaLayer<'a, PrefillMamba2<'a>> {
    pub(crate) fn is_supported(&self) -> bool {
        let n = self.attn_norm.len();
        self.block.in_cols() == n
            && self.block.out_rows() == n
            && self.ffn.as_ref().is_none_or(|f| f.fits(n))
    }
}

/// The per-row widths of the run's SSM slots, for one layer.
#[derive(Clone, Copy, Default)]
pub(crate) struct SsmNeed {
    pub block: ScratchNeed,
    pub dt_low: usize,
    /// The short conv's `in_proj` output and gated output.
    pub conv: usize,
}

impl SsmNeed {
    pub(crate) fn max(self, o: Self) -> Self {
        Self {
            block: self.block.max(o.block),
            dt_low: self.dt_low.max(o.dt_low),
            conv: self.conv.max(o.conv),
        }
    }

    pub(crate) fn of_mamba1(m: &PrefillMamba1<'_>) -> Self {
        Self {
            block: m.widths(),
            dt_low: m.dt_rank(),
            conv: 0,
        }
    }

    pub(crate) fn of_mamba2(m: &PrefillMamba2<'_>) -> Self {
        Self {
            block: m.widths(),
            ..Self::default()
        }
    }

    pub(crate) fn of_conv(c: &PrefillConvLayer<'_>) -> Self {
        Self {
            conv: c.attn_norm.len(),
            ..Self::default()
        }
    }

    /// Floats per slot for `rows` rows, `(slot, len)`.
    pub(crate) fn lens(self, rows: usize) -> [(usize, usize); 6] {
        let b = self.block;
        let r = |w: usize| w.max(1) * rows;
        [
            (slot::SSM_IN, r(b.q.max(3 * self.conv))),
            (slot::SSM_CONV, r(b.attn)),
            (slot::SSM_XDB, r(b.kv)),
            (slot::SSM_DT, r(b.kv)),
            (slot::SSM_DTLOW, r(self.dt_low)),
            (slot::SSM_Y, r(b.gate.max(self.conv))),
        ]
    }
}

fn attn_norm(
    enc: &mut Enc<'_>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    w: &Arc<[f32]>,
    eps: f32,
    sc: &crate::scratch_pool::Scratch,
    rows: usize,
) -> Result<(), MetalError> {
    let norm = resident_f32_shared(device, w)?;
    let encoder = enc.span("pf_attn_norm")?;
    crate::norm::encode_rms_norm_batch(
        encoder,
        device,
        sc.buf(slot::H),
        &norm.buffer,
        sc.buf(slot::XN),
        w.len() as u32,
        rows as u32,
        eps,
    )
}

/// A conv layer into the run. Its history region of `slot::CONV_HIST`
/// starts `hist_at` floats in and already holds the window.
pub(crate) fn encode_conv_layer(
    enc: &mut Enc<'_>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    w: &PrefillConvLayer<'_>,
    hist_at: usize,
    sc: &crate::scratch_pool::Scratch,
    rows: usize,
) -> Result<(), MetalError> {
    let b = |i: usize| sc.buf(i);
    let n = w.attn_norm.len();
    let taps = resident_f32_shared(device, w.taps)?;
    attn_norm(enc, device, w.attn_norm, w.eps, sc, rows)?;
    let encoder = enc.span("pf_conv_in_proj")?;
    encode_gemm(
        encoder,
        device,
        &w.in_proj,
        b(slot::XN),
        b(slot::SSM_IN),
        rows,
    )?;
    let encoder = enc.span("pf_shortconv")?;
    let hist = b(slot::CONV_HIST);
    crate::shortconv::encode_shortconv_rows(
        encoder,
        device,
        crate::shortconv::ConvRowsIo {
            hist: (hist, hist_at * 4),
            bx_out: (hist, (hist_at + (w.l_cache - 1) * n) * 4),
            taps: &taps.buffer,
            bcx: b(slot::SSM_IN),
            y: b(slot::SSM_Y),
            n,
            l_cache: w.l_cache,
            rows,
        },
    )?;
    let encoder = enc.span("pf_conv_out_proj")?;
    encode_gemm(
        encoder,
        device,
        &w.out_proj,
        b(slot::SSM_Y),
        b(slot::OUT),
        rows,
    )?;
    crate::gdn_prefill::encode_tail(enc, device, w.ffn.as_ref(), w.eps, sc, rows, n)
}

/// A Mamba-1 layer into the run.
pub(crate) fn encode_mamba1_layer(
    enc: &mut Enc<'_>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    w: &PrefillMambaLayer<'_, PrefillMamba1<'_>>,
    sc: &crate::scratch_pool::Scratch,
    rows: usize,
) -> Result<(), MetalError> {
    let b = |i: usize| sc.buf(i);
    attn_norm(enc, device, w.attn_norm, w.eps, sc, rows)?;
    let encoder = enc.span("pf_mamba1")?;
    encode_mamba1(
        encoder,
        &mut MemRanges::new(),
        device,
        &w.block,
        &Mamba1Bufs {
            x: b(slot::XN),
            xz: b(slot::SSM_IN),
            xc: b(slot::SSM_CONV),
            x_db: b(slot::SSM_XDB),
            dt: b(slot::SSM_DT),
            dt_low: Some(b(slot::SSM_DTLOW)),
            y: b(slot::SSM_Y),
            out: b(slot::OUT),
        },
        rows,
    )?;
    let hidden = w.attn_norm.len();
    crate::gdn_prefill::encode_tail(enc, device, w.ffn.as_ref(), w.eps, sc, rows, hidden)
}

/// The Mamba-2 block over `attn_norm(h)` in `slot::XN`, into `out`: a
/// Mamba-2 layer's whole branch, or Falcon-H1's beside its attention.
pub(crate) fn encode_mamba2_block(
    enc: &mut Enc<'_>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    m: &PrefillMamba2<'_>,
    sc: &crate::scratch_pool::Scratch,
    out: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
) -> Result<(), MetalError> {
    let b = |i: usize| sc.buf(i);
    let encoder = enc.span("pf_mamba2")?;
    encode_mamba2(
        encoder,
        &mut MemRanges::new(),
        device,
        m,
        &Mamba2Bufs {
            x: b(slot::XN),
            zxbcdt: b(slot::SSM_IN),
            xbc: b(slot::SSM_CONV),
            y: b(slot::SSM_Y),
            out,
        },
        rows,
    )
}

/// A Mamba-2 layer into the run.
pub(crate) fn encode_mamba2_layer(
    enc: &mut Enc<'_>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    w: &PrefillMambaLayer<'_, PrefillMamba2<'_>>,
    sc: &crate::scratch_pool::Scratch,
    rows: usize,
) -> Result<(), MetalError> {
    attn_norm(enc, device, w.attn_norm, w.eps, sc, rows)?;
    encode_mamba2_block(enc, device, &w.block, sc, sc.buf(slot::OUT), rows)?;
    let hidden = w.attn_norm.len();
    crate::gdn_prefill::encode_tail(enc, device, w.ffn.as_ref(), w.eps, sc, rows, hidden)
}
