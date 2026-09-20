//! Block-size alignment for sliding-window attention (SWA).
//!
//! The block cache slices a sequence into fixed runs of `block_size`
//! token positions and treats each run as an independently storable,
//! evictable, restorable unit. That works without further thought for
//! full causal attention: block `b` covers positions
//! `[b*B, (b+1)*B)`, restoring blocks `0..m` reproduces positions
//! `[0, m*B)`, and nothing about position `p`'s KV state depends on how
//! the range was cut.
//!
//! **SWA breaks that, and it breaks it silently.** A sliding layer only
//! ever needs -- and, once the KV is capped rather than kept whole,
//! only ever *holds* -- the most recent `window` positions. Dropping
//! stale positions has to happen in whole blocks, because a block is
//! the eviction unit. So the resident set is a whole number of blocks,
//! and the only way a whole number of blocks can be exactly the last
//! `window` positions is:
//!
//! ```text
//! window % block_size == 0
//! ```
//!
//! When it does not divide, the boundary between "still inside the
//! window" and "safe to drop" falls in the middle of a block, and there
//! are only two things an implementation can do with that block: keep
//! it (so the window is silently *wider* than the model's, changing the
//! attention mask) or drop it (so positions the model must attend to
//! are silently *missing*). Neither errors. Both produce confident
//! wrong tokens, which is the same failure class
//! [`kv_signature`](crate::kv_signature) exists to prevent -- so it is
//! prevented the same way: refuse, naming both numbers.
//!
//! > Wording note, for anyone holding `docs/plans/serving-and-tiered-kv.md`
//! > open: the plan states this as "block size must be a multiple of the
//! > sliding-window size". That is the same constraint with the operands
//! > the other way round, and this direction is the one that is
//! > implementable -- forcing `block_size` up to a multiple of a 128-token
//! > gpt-oss window would make every block at least a whole window, which
//! > defeats the point of blocks. The constraint is the usual
//! > `sliding_window % block_size == 0`, and this module holds it.
//!
//! # Why this is live, not hypothetical
//!
//! frink runs a real alternating-SWA gpt-oss graph (window 128, every
//! other layer) and Gemma-3 SWA prefill (window 512, every 6th layer
//! full-attention). An alternating model does not weaken the rule: the
//! full-attention layers impose no constraint, and one mis-aligned
//! sliding layer is enough to corrupt the answer. And the disk tier
//! ([`kv_disk`](crate::kv_disk)) makes a mis-aligned block *durable* --
//! it would outlive the process that created it and be handed to the
//! next one. That is why [`BlockLayout`] is carried in the cache
//! signature rather than checked once at startup: a block written under
//! one window is refused by a build expecting another, instead of being
//! read back as if the window had never changed.

/// Why a block layout was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockLayoutError {
    /// A zero-token block is not a unit of anything.
    ZeroBlockSize,
    /// `Some(0)` is not "no window": a zero window would mean a query
    /// attends to nothing, which no model does. A model with no
    /// sliding-window attention must say `None`.
    ZeroWindow,
    /// The eviction boundary would fall inside a block. See the module
    /// note for what the two possible responses to that both corrupt.
    Misaligned {
        window: usize,
        block_size: usize,
        remainder: usize,
    },
}

impl std::fmt::Display for BlockLayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockLayoutError::ZeroBlockSize => write!(f, "KV block size must be positive"),
            BlockLayoutError::ZeroWindow => write!(
                f,
                "sliding window must be positive; a model without SWA has no window at all"
            ),
            BlockLayoutError::Misaligned {
                window,
                block_size,
                remainder,
            } => write!(
                f,
                "KV block size {block_size} does not divide the sliding window {window} \
                 ({window} % {block_size} = {remainder}); a block would straddle the \
                 window boundary and be either kept too long or dropped too early"
            ),
        }
    }
}

impl std::error::Error for BlockLayoutError {}

/// How a sequence is cut into cache blocks, and the sliding window (if
/// any) those blocks must line up with.
///
/// Only constructible through [`BlockLayout::new`], so a `BlockLayout`
/// value is itself the proof that the alignment rule holds -- there is
/// no path that produces a mis-aligned one to be checked later and
/// forgotten.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    block_size: usize,
    sliding_window: Option<usize>,
}

impl BlockLayout {
    /// Validates and builds a layout. `sliding_window` is `None` for a
    /// full-causal model and `Some(w)` for one whose sliding layers use
    /// window `w` -- including alternating models, where some layers
    /// are full-attention: those layers are unconstrained, so the
    /// model's constraint is the window of the layers that have one.
    pub fn new(block_size: usize, sliding_window: Option<usize>) -> Result<Self, BlockLayoutError> {
        if block_size == 0 {
            return Err(BlockLayoutError::ZeroBlockSize);
        }
        match sliding_window {
            None => Ok(BlockLayout {
                block_size,
                sliding_window: None,
            }),
            Some(0) => Err(BlockLayoutError::ZeroWindow),
            Some(window) => {
                let remainder = window % block_size;
                if remainder != 0 {
                    return Err(BlockLayoutError::Misaligned {
                        window,
                        block_size,
                        remainder,
                    });
                }
                Ok(BlockLayout {
                    block_size,
                    sliding_window: Some(window),
                })
            }
        }
    }

    /// A full-causal model's layout. Cannot fail except on a zero block
    /// size.
    pub fn full_attention(block_size: usize) -> Result<Self, BlockLayoutError> {
        Self::new(block_size, None)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// How many whole blocks a sliding layer keeps resident. `None` for
    /// a full-causal model, which keeps all of them.
    ///
    /// Exact by construction: the layout would not exist if the window
    /// were not a whole number of blocks.
    pub fn blocks_per_window(&self) -> Option<usize> {
        self.sliding_window.map(|w| w / self.block_size)
    }
}

/// How many rows a *contiguous* windowed KV cache keeps resident, and
/// when it drops the ones behind the window.
///
/// [`BlockLayout`] above is the same question for the block/paged tier,
/// where the eviction unit is a block. This is the answer for
/// `frink_core::cache::KvCache`, where the eviction unit is a row and
/// the only thing stopping a per-token drain is arithmetic.
///
/// # Why there is slack
///
/// Dropping exactly one row per push moves every remaining row down by
/// one every single token: `window * n_kv_heads * head_dim` floats per
/// layer per token, about a megabyte per token per layer on Gemma-3-4B.
/// So the cache is allowed to run `slack` rows past the window and then
/// drops `slack + 1` at once, which amortises the move to roughly
/// `window / (slack + 1)` rows per token.
///
/// # Why this is a type and not two lines in `push`
///
/// The store keeps these rows and the budget
/// (`frink_models::kv_budget`) has to price exactly what the store
/// keeps. #33 is the record of what happens when those two are separate
/// statements of the same rule: the budget capped a sliding layer that
/// no store ever capped, `-c auto` approved a context that did not fit,
/// and the failure arrived as an OOM. [`KvWindow::rows_after`] is the
/// single rule; the store calls it to decide and the budget calls it to
/// price, so there is nothing left for them to disagree about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvWindow {
    window: usize,
    slack: usize,
}

impl KvWindow {
    /// `None` for a zero window, for [`BlockLayoutError::ZeroWindow`]'s
    /// reason: a query that attends to nothing is not a model.
    pub fn new(window: usize, slack: usize) -> Option<Self> {
        (window > 0).then_some(KvWindow { window, slack })
    }

    /// The slack a caller gets when it has no opinion: half a window,
    /// so the cache peaks at 1.5x the window and moves roughly two rows
    /// per token instead of `window` of them.
    pub fn with_default_slack(window: usize) -> Option<Self> {
        Self::new(window, (window / 2).max(1))
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn slack(&self) -> usize {
        self.slack
    }

    /// The most rows this window ever leaves resident.
    pub fn max_rows(&self) -> usize {
        self.window + self.slack
    }

    /// **The rule.** Rows resident once the sequence has consumed
    /// `positions` positions and the holder has evicted at every step.
    ///
    /// Grows one per position up to `window + slack`, then drops back to
    /// `window` and climbs again, so the resident count cycles through
    /// `[window, window + slack]` with period `slack + 1`.
    ///
    /// The invariant every reader depends on is
    /// `rows_after(p) >= min(p, window)`: the rows kept are always at
    /// least the last `window` positions, which is exactly the set a
    /// windowed attention kernel reads. Everything above that is slack
    /// the kernel skips, so evicting can only ever change *where* a row
    /// sits, never *whether* it is read. That is why turning eviction on
    /// is token-identical rather than approximately so, and it is
    /// asserted in this module's tests rather than argued here.
    pub fn rows_after(&self, positions: usize) -> usize {
        let peak = self.window + self.slack;
        if positions <= peak {
            return positions;
        }
        self.window + (positions - peak - 1) % (self.slack + 1)
    }
}

/// The largest block size no greater than `desired` that satisfies the
/// rule for `window`.
///
/// This is what a *configuration* layer should call: an operator asking
/// for 256-token blocks on a 128-token-window gpt-oss should get 128,
/// not an error and not silent corruption. It never rounds *up*, since
/// a block larger than asked for costs more memory per eviction step
/// than the operator budgeted for; and it never returns 0.
///
/// `window == None` returns `desired` unchanged. `desired == 0` is
/// treated as 1, because there is no smaller honest answer.
///
/// The search walks down from `desired`; block sizes are small (tens to
/// low hundreds of tokens), so this is cheaper than factoring the
/// window and is called once per model load.
pub fn aligned_block_size(desired: usize, window: Option<usize>) -> usize {
    let desired = desired.max(1);
    let Some(window) = window.filter(|w| *w > 0) else {
        return desired;
    };
    (1..=desired.min(window))
        .rev()
        .find(|candidate| window.is_multiple_of(*candidate))
        // 1 divides every positive window, so the iterator is never
        // empty -- but expressing the fallback beats an unwrap.
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the module. gpt-oss's real window is 128; a
    /// 48-token block does not divide it, and the two things an
    /// implementation could do with the straddling block are both
    /// wrong-answer bugs, not misses.
    #[test]
    fn a_block_size_that_does_not_divide_the_window_is_refused() {
        let err = BlockLayout::new(48, Some(128)).expect_err("48 does not divide 128");
        assert_eq!(
            err,
            BlockLayoutError::Misaligned {
                window: 128,
                block_size: 48,
                remainder: 32,
            }
        );
        // The operator has to be able to fix it from the message alone.
        let text = err.to_string();
        assert!(text.contains("48"), "{text}");
        assert!(text.contains("128"), "{text}");
    }

    #[test]
    fn a_block_size_that_divides_the_window_is_accepted() {
        let layout = BlockLayout::new(32, Some(128)).expect("32 divides 128");
        assert_eq!(layout.block_size(), 32);
        assert_eq!(layout.sliding_window(), Some(128));
        assert_eq!(layout.blocks_per_window(), Some(4));
    }

    /// A block larger than the whole window is the most obvious form of
    /// the bug and must not slip through as "well, it covers it".
    #[test]
    fn a_block_larger_than_the_window_is_refused() {
        assert!(matches!(
            BlockLayout::new(256, Some(128)),
            Err(BlockLayoutError::Misaligned { .. })
        ));
        // ... unless it is exactly the window, which is aligned.
        assert!(BlockLayout::new(128, Some(128)).is_ok());
    }

    #[test]
    fn a_full_causal_model_constrains_nothing() {
        let layout = BlockLayout::full_attention(48).expect("no window, no constraint");
        assert_eq!(layout.sliding_window(), None);
        assert_eq!(layout.blocks_per_window(), None);
    }

    /// `Some(0)` and `None` are different claims and only one of them
    /// is "this model has no SWA".
    #[test]
    fn a_zero_window_is_not_the_same_as_no_window() {
        assert_eq!(
            BlockLayout::new(16, Some(0)),
            Err(BlockLayoutError::ZeroWindow)
        );
        assert!(BlockLayout::new(16, None).is_ok());
    }

    #[test]
    fn a_zero_block_size_is_refused_with_or_without_a_window() {
        assert_eq!(
            BlockLayout::new(0, None),
            Err(BlockLayoutError::ZeroBlockSize)
        );
        assert_eq!(
            BlockLayout::new(0, Some(128)),
            Err(BlockLayoutError::ZeroBlockSize)
        );
    }

    /// Whatever `aligned_block_size` returns must be constructible --
    /// otherwise the config layer hands the cache a size the cache
    /// rejects, which is a startup crash rather than a fix.
    #[test]
    fn the_aligned_size_is_always_a_size_the_layout_accepts() {
        for window in [1usize, 2, 3, 128, 512, 1024, 4096, 4099] {
            for desired in [1usize, 7, 16, 31, 32, 100, 128, 256, 5000] {
                let size = aligned_block_size(desired, Some(window));
                assert!(size > 0 && size <= desired, "{desired}/{window} -> {size}");
                BlockLayout::new(size, Some(window)).unwrap_or_else(|e| {
                    panic!("aligned_block_size({desired}, {window}) = {size} is not valid: {e}")
                });
            }
        }
    }

    #[test]
    fn the_aligned_size_rounds_down_never_up() {
        // gpt-oss: a 256-token request on a 128 window becomes 128, not
        // 256 and not 384.
        assert_eq!(aligned_block_size(256, Some(128)), 128);
        // Gemma-3: 512-token window, 100 requested -> 64, the largest
        // divisor at or below 100.
        assert_eq!(aligned_block_size(100, Some(512)), 64);
        // Already aligned: untouched.
        assert_eq!(aligned_block_size(64, Some(512)), 64);
        // A prime window leaves only 1 below itself.
        assert_eq!(aligned_block_size(100, Some(4099)), 1);
    }

    #[test]
    fn no_window_leaves_the_desired_size_alone() {
        assert_eq!(aligned_block_size(48, None), 48);
        assert_eq!(aligned_block_size(0, None), 1);
    }

    /// A window of zero would mean a query attends to nothing.
    #[test]
    fn a_zero_window_is_not_a_window() {
        assert!(KvWindow::new(0, 4).is_none());
        assert!(KvWindow::with_default_slack(0).is_none());
        assert!(KvWindow::new(1, 0).is_some());
    }

    /// The closed form must agree with the loop it stands for.
    ///
    /// `rows_after` is a formula and the store is a loop that pushes one
    /// row and drops back to the formula's answer. Two statements of one
    /// rule is this repo's dominant bug shape, so the formula is checked
    /// against a simulation of the loop rather than against hand-written
    /// numbers that were themselves derived from the formula.
    #[test]
    fn the_closed_form_matches_a_step_by_step_simulation() {
        for window in 1..=9usize {
            for slack in 0..=7usize {
                let w = KvWindow::new(window, slack).expect("positive window");
                let mut rows = 0usize;
                for positions in 1..=200usize {
                    // What the store does: append one row, then drop
                    // back to whatever the rule allows.
                    rows += 1;
                    rows = rows.min(w.rows_after(positions));
                    assert_eq!(
                        rows,
                        w.rows_after(positions),
                        "window {window} slack {slack} at {positions} positions"
                    );
                }
            }
        }
    }

    /// The property attention depends on, and the only reason evicting
    /// is allowed to be token-identical: whatever else it drops, a
    /// windowed cache still holds the last `min(positions, window)`
    /// positions.
    #[test]
    fn the_last_window_positions_are_always_still_resident() {
        for window in 1..=9usize {
            for slack in 0..=7usize {
                let w = KvWindow::new(window, slack).expect("positive window");
                for positions in 0..=200usize {
                    let rows = w.rows_after(positions);
                    assert!(
                        rows >= positions.min(window),
                        "window {window} slack {slack} at {positions}: kept {rows} rows, \
                         which is fewer than the {} the kernel reads",
                        positions.min(window)
                    );
                    assert!(rows <= positions, "cannot keep rows that were never pushed");
                    assert!(rows <= w.max_rows(), "resident rows must stay bounded");
                }
            }
        }
    }

    /// The point of the whole exercise: resident rows STOP GROWING.
    #[test]
    fn a_windowed_layer_stops_growing_while_positions_do_not() {
        let w = KvWindow::with_default_slack(1024).expect("positive window");
        assert_eq!(w.rows_after(512), 512);
        assert!(w.rows_after(32_768) <= w.max_rows());
        assert_eq!(w.max_rows(), 1024 + 512);
        // 21x fewer rows than positions at a 32k context.
        assert!(w.rows_after(32_768) * 20 < 32_768);
    }

    /// Slack is what makes the drain a block move instead of a per-token
    /// one: with `slack` rows of headroom the cache only actually drops
    /// rows once every `slack + 1` positions.
    #[test]
    fn rows_are_dropped_once_per_slack_plus_one_positions() {
        let w = KvWindow::new(8, 3).expect("positive window");
        let drops = (1..=400usize)
            .filter(|p| w.rows_after(*p) < w.rows_after(p - 1) + 1)
            .count();
        // The first drop is at position 12 (the first that would take
        // the cache past `window + slack` = 11), then one every
        // `slack + 1` = 4 positions.
        assert_eq!(drops, (400 - 12) / 4 + 1);
        // The same statement the other way round: 400 positions cost 98
        // block moves, not 400 per-token ones.
        assert!(drops * 4 <= 400);
    }
}
