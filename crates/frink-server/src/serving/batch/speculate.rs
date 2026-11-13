//! Speculative decoding inside the continuous batch (#533).
//!
//! A row samples its next token as it always has, then drafts a block
//! after it, and every row's `[token, draft...]` goes through ONE
//! forward (`Decoder::forward_multi_seq_lanes`), each row attending over
//! its own sequence. The rows are then settled one lane at a time by
//! agreement, as the private loop's `verify_block` settles them: the
//! token drawn from each verify row is committed, and the block stops
//! at the first draw that differs from the draft.
//!
//! Every committed token is drawn ONCE, from the logits a plain decode
//! would have drawn it from, in the same order: the last draw of a
//! block (the one that disagreed, or the bonus after a fully accepted
//! block) is committed but not yet fed, and the next tick feeds it
//! instead of sampling (`Slot::pending`). So the answer, seeded
//! sampling included, is the answer without a drafter; a drafter only
//! saves forwards.

use std::sync::Arc;

use frink_models::dflash::DFlashDrafter;
use frink_models::speculative::{Drafter, PromptLookupSpeculator};
use frink_models::{Decoder, DraftModelSpeculator};

use crate::generate::GenerationParams;
use crate::sampling_loop::{DEFAULT_DRAFT_MAX, DRAFT_NGRAM};
use crate::server_draft::ServerDraft;

use super::config::DecodeFn;
use super::row::{RowKv, Slot};
use super::worker::{commit_token, draw};

/// What drafts for one row.
pub(super) enum RowDrafter {
    /// An n-gram match over the row's own output, as on the private
    /// loop when no `--model-draft` is set.
    Lookup(PromptLookupSpeculator),
    /// The server's `--model-draft`, with KV of this row's own.
    Model(Box<DraftModelSpeculator>, usize),
    /// A DFlash `--model-draft`, fed the target's layer inputs at every
    /// position the row writes: the prompt's at the prefill handover,
    /// then every verify forward's.
    DFlash(Box<DFlashDrafter<Arc<Decoder>>>),
}

impl RowDrafter {
    /// The drafter for a row, or `None` where the row must not draft:
    /// a grammar or `json_object` request (as on the private loop), a
    /// recurrent target (whose batched state has no checkpoint).
    pub(super) fn for_row(
        draft: Option<&ServerDraft>,
        decoder: &Arc<Decoder>,
        params: &GenerationParams,
    ) -> Option<Self> {
        if params.grammar.is_some() || params.json_object || decoder.config.has_recurrent_layers() {
            return None;
        }
        if let Some(d) = draft.and_then(|d| d.dflash(Arc::clone(decoder))) {
            return Some(RowDrafter::DFlash(Box::new(d)));
        }
        match draft {
            Some(d) => d
                .speculator(&decoder.config, params.sampling.clone(), params.seed)
                .map(|s| RowDrafter::Model(Box::new(s), d.max)),
            None => Some(RowDrafter::Lookup(PromptLookupSpeculator::new(
                DRAFT_NGRAM,
                DEFAULT_DRAFT_MAX,
            ))),
        }
    }

    fn max_len(&self) -> usize {
        match self {
            RowDrafter::Lookup(_) => DEFAULT_DRAFT_MAX,
            RowDrafter::Model(_, max) => *max,
            RowDrafter::DFlash(d) => d.n_max(),
        }
    }

    /// The target layers whose inputs this drafter reads, for a DFlash
    /// draft; `None` for every drafter that reads only tokens.
    pub(super) fn target_layers(&self) -> Option<&[usize]> {
        match self {
            RowDrafter::DFlash(d) => Some(d.target_layers()),
            RowDrafter::Lookup(_) | RowDrafter::Model(..) => None,
        }
    }

    /// The prompt's layer inputs, from the tapped prefill.
    pub(super) fn observe_prompt(&mut self, features: &[Vec<f32>]) {
        if let RowDrafter::DFlash(d) = self {
            d.observe_target(0, features);
        }
    }

    /// `anchor_pos` is where the row's newest token goes: it is not fed
    /// yet, and a DFlash draft reads the context before it, as
    /// llama.cpp's loop drafts from `id_last` at `pos0`.
    fn propose(
        &mut self,
        prompt: &[usize],
        generated: &[usize],
        anchor_pos: usize,
        max: usize,
    ) -> Vec<usize> {
        let mut block = match self {
            RowDrafter::DFlash(d) => match generated.last() {
                Some(&anchor) => d
                    .draft_after(anchor, anchor_pos, max)
                    .into_iter()
                    .map(|(t, _)| t)
                    .collect(),
                None => Vec::new(),
            },
            RowDrafter::Lookup(d) => d.propose_tokens(generated),
            // The draft model continues the whole sequence and rolls its
            // own KV back to it each round.
            RowDrafter::Model(d, _) => {
                let mut sequence = Vec::with_capacity(prompt.len() + generated.len());
                sequence.extend_from_slice(prompt);
                sequence.extend_from_slice(generated);
                d.propose(&sequence, &[], max).tokens().to_vec()
            }
        };
        block.truncate(max);
        block
    }
}

/// The block to verify after the row's newest token, empty when the row
/// does not draft this tick.
///
/// No longer than the budget allows: the newest token and the drafts
/// are committed now, and the draw after them commits one more, so a
/// block may run to `max_tokens - generated - 1`. And no longer than
/// the row's KV can roll back: a windowed layer keeps only so many rows
/// of headroom, and a paged lease under a sliding window none.
pub(super) fn draft_block(slot: &mut Slot, decoder: &Decoder) -> Vec<usize> {
    let Some(drafter) = slot.drafter.as_mut() else {
        return Vec::new();
    };
    let room = slot
        .max_tokens
        .saturating_sub(slot.answer.generated_ids.len() + 1);
    let undoable = match &slot.kv {
        RowKv::Contiguous(caches) => caches
            .iter()
            .map(|c| c.rollback_limit())
            .min()
            .unwrap_or(usize::MAX),
        RowKv::Paged(lease) if lease.can_roll_back(decoder) => usize::MAX,
        RowKv::Paged(_) => 0,
    };
    let max = drafter.max_len().min(room).min(undoable);
    if max == 0 {
        return Vec::new();
    }
    drafter.propose(&slot.prompt_ids, &slot.answer.generated_ids, slot.pos, max)
}

/// Hands a DFlash row's drafter the layer inputs of the rows this
/// forward wrote at `pos`. A no-op for every other drafter.
pub(super) fn observe_verify(slot: &mut Slot, pos: usize, features: &[Vec<f32>]) {
    if let Some(RowDrafter::DFlash(d)) = slot.drafter.as_mut() {
        d.observe_target(pos, features);
    }
}

/// Once the row has settled, forgets the layer inputs of rows a
/// rejection undid. A no-op for every other drafter.
pub(super) fn forget_rejected(slot: &mut Slot) {
    if let Some(RowDrafter::DFlash(d)) = slot.drafter.as_mut() {
        d.rollback(slot.pos);
    }
}

/// Settles a row after the forward that fed its newest token and
/// `draft` at `slot.pos`: `rows[i]` are the logits after the newest
/// token and the first `i` drafts.
pub(super) fn settle(slot: &mut Slot, rows: &[Vec<f32>], draft: &[usize], decode: &DecodeFn) {
    debug_assert_eq!(rows.len(), draft.len() + 1, "one row per fed token");
    // The newest token, fed by this forward.
    slot.pos += 1;
    slot.answer.forwards += 1;
    slot.answer.drafted += draft.len();
    for (i, logits) in rows.iter().enumerate() {
        let Some(token) = draw(slot, logits, decode) else {
            break;
        };
        let agreed = draft.get(i) == Some(&token);
        let before = slot.answer.generated_ids.len();
        let live = commit_token(slot, token, decode);
        if agreed {
            slot.answer.accepted += 1;
            // Its row is in the KV, written by this forward from the
            // same token.
            if slot.answer.generated_ids.len() > before {
                slot.pos += 1;
            }
        }
        if !live {
            break;
        }
        if !agreed {
            // The draw that disagreed with the draft, or the one after
            // a fully accepted block: committed, and fed next tick.
            slot.pending = Some(token);
            break;
        }
    }
    // Rows past the committed tokens describe drafts that never
    // happened; a published prefix must not contain them.
    truncate(&mut slot.kv, slot.pos);
}

/// Drops every row past `len` positions.
fn truncate(kv: &mut RowKv, len: usize) {
    match kv {
        RowKv::Contiguous(caches) => {
            for cache in caches {
                cache.truncate(len);
            }
        }
        RowKv::Paged(lease) => {
            for cache in lease.caches_mut() {
                cache.truncate(len);
            }
        }
    }
}
