//! [`DFlashDrafter`]: a DFlash draft behind [`crate::speculative::Drafter`].

use crate::decoder::Decoder;
use crate::speculative::{DraftBlock, Drafter};

use super::config::DFlashError;
use super::forward::{ContextKv, DFlashDraft};
use super::selector::read_out;

/// Drafts for one sequence of `target`.
///
/// The draft's KV holds the target's features for every committed
/// position, fed by [`Drafter::observe_target`] after each target
/// forward. A block is drafted at the position after them, from the
/// anchor (the last committed token, whose features the target has not
/// computed yet), exactly as `common_speculative_impl_draft_dflash`
/// drafts at `pos0` from `id_last`.
pub struct DFlashDrafter<'t> {
    draft: DFlashDraft,
    target: &'t Decoder,
    ctx: Vec<ContextKv>,
    /// Positions the context holds: `0..ctx_len`.
    ctx_len: usize,
    /// Set when the target reported positions past a gap in the
    /// context; the drafter then proposes nothing rather than draft
    /// from a context with a hole in it.
    gapped: bool,
    n_max: usize,
    p_min: f32,
}

impl<'t> DFlashDrafter<'t> {
    /// Pairs `draft` with `target`, refusing a pair that does not fit
    /// (widths, vocabulary, the layers it reads). `n_max` is clamped to
    /// the trained block, as `common_speculative` clamps it
    /// (`speculative.cpp:1099-1105`).
    pub fn new(
        draft: DFlashDraft,
        target: &'t Decoder,
        n_max: usize,
        p_min: f32,
    ) -> Result<Self, DFlashError> {
        draft.check_target(target)?;
        let n_max = n_max.min(draft.config().max_draft());
        let layers = draft.config().n_layer;
        Ok(Self {
            draft,
            target,
            ctx: vec![ContextKv::default(); layers],
            ctx_len: 0,
            gapped: false,
            n_max,
            p_min,
        })
    }

    pub fn config(&self) -> &super::DFlashConfig {
        self.draft.config()
    }

    /// The most tokens one block drafts.
    pub fn n_max(&self) -> usize {
        self.n_max
    }

    fn truncate(&mut self, len: usize) {
        let kvw = self.draft.config().n_head_kv * self.draft.config().head_dim;
        for kv in &mut self.ctx {
            kv.k.truncate(len * kvw);
            kv.v.truncate(len * kvw);
        }
        self.ctx_len = len;
    }

    /// Drafts after `anchor` at the position after the context, with
    /// up to `max_len` tokens, each with the score the walk chose it on
    /// (the lattice score for DFlash2, the logit for DFlash). Public for
    /// the reference test, which pins both against libllama's.
    pub fn draft_block(&self, anchor: usize, max_len: usize) -> Vec<(usize, f32)> {
        let n_draft = self.n_max.min(max_len);
        if n_draft == 0 || self.gapped || self.ctx_len == 0 {
            return Vec::new();
        }
        let out =
            self.draft
                .block_forward(self.target, anchor, n_draft + 1, self.ctx_len, &self.ctx);
        match &self.draft.selector {
            Some(sel) => sel.walk(&out, anchor, self.p_min),
            None => read_out(&out, self.p_min),
        }
    }
}

impl Drafter for DFlashDrafter<'_> {
    fn propose(&mut self, history: &[usize], _target_hidden: &[f32], max_len: usize) -> DraftBlock {
        let Some(&anchor) = history.last() else {
            return DraftBlock::default();
        };
        // The walk is an argmax, so the token drawn at each position is
        // certain given the block: a point mass is the draft's honest
        // distribution, and `accept_or_resample` stays lossless at any
        // temperature.
        let tokens = self
            .draft_block(anchor, max_len)
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        DraftBlock::deterministic(tokens)
    }

    fn target_layers(&self) -> &[usize] {
        &self.draft.config().target_layers
    }

    fn observe_target(&mut self, start_pos: usize, features: &[Vec<f32>]) {
        if start_pos > self.ctx_len {
            self.gapped = true;
            return;
        }
        // Positions at and after `start_pos` are being recomputed by
        // the target: a rejected block's rows go, the verified ones
        // come back.
        self.truncate(start_pos);
        self.draft.inject(features, start_pos, &mut self.ctx);
        self.ctx_len = start_pos + features.len();
    }
}
