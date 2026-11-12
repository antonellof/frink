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
}

impl RowDrafter {
    /// The drafter for a row, or `None` where the row must not draft:
    /// a grammar or `json_object` request (as on the private loop), a
    /// recurrent target (whose batched state has no checkpoint), and a
    /// draft this path does not serve yet (DFlash, #600).
    pub(super) fn for_row(
        draft: Option<&ServerDraft>,
        decoder: &Decoder,
        params: &GenerationParams,
    ) -> Option<Self> {
        if params.grammar.is_some() || params.json_object || decoder.config.has_recurrent_layers() {
            return None;
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
        }
    }

    fn propose(&mut self, prompt: &[usize], generated: &[usize], max: usize) -> Vec<usize> {
        let mut block = match self {
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
    drafter.propose(&slot.prompt_ids, &slot.answer.generated_ids, max)
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
