//! Semantic anchor checkpoints: keeping the state an agent is about to
//! come back to.
//!
//! # The workload this exists for
//!
//! An agentic turn is not one generation. The model reasons, calls a
//! tool, the harness runs it, and the *same conversation* comes back
//! with the tool's result spliced in. What comes back is not an
//! extension of what went out -- the harness routinely edits the
//! context at that boundary: it drops the thinking block, rewrites the
//! call into a canonical form, appends a result. So the next request's
//! prompt agrees with the last one up to **the tool call** and diverges
//! after it.
//!
//! That makes the position where the tool call began the single most
//! valuable position in the sequence: it is where the next request will
//! rejoin. Ordinary caching does not know that. A window model slides
//! its window forward as it decodes and, by the time the turn ends, the
//! state around the anchor is long gone -- so the next turn recomputes
//! a prefix that was live thirty tokens earlier.
//!
//! An anchor is one integer that says "this position is worth keeping",
//! and two rules that spend it:
//!
//! - the **window slide** stops short of `anchor - window - retain gap`
//!   instead of following the cursor ([`decode_slide`]);
//! - a **recurrent model** freezes its state at the anchor into its idle
//!   snapshot slot ([`snapshot_at_anchor`]), because recurrent state is
//!   a point rather than a span and cannot be reconstructed from
//!   neighbours.
//!
//! # Why the anchor is dropped rather than held forever
//!
//! Holding a position live costs pool. If decode runs a long way past
//! the anchor, the request is holding two windows -- one at the cursor,
//! one at the anchor -- and the gap between them is unbounded, so the
//! pool floor would have to be unbounded too. [`decode_slide`] drops
//! the anchor once the cursor has drifted more than one window plus the
//! gap past it: at that distance the turn is clearly not about to end,
//! and a prefix that far back is cheaper to recompute than to hold.
//!
//! That bound is what makes a per-request window cost finite at all:
//! without the drop rule, any pool sized for one would be a fiction.
//!
//! Ported 1:1 from FreeToken's `scheduler/scheduler.py` (anchor
//! detection), `scheduler/cache.py` (`maybe_free_swa_out_of_window`,
//! `snapshot_toolcall_anchor`) and `utils/hf.py`
//! (`load_toolcall_anchor_id`) (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`.

use crate::policy::pool_budget::SWA_RETAIN_GAP;
use crate::policy::radix::align_down;

/// Resolve a tool-call opener to the single token that announces it.
///
/// `None` unless the opener encodes to **exactly one** token, and that
/// is not a limitation to work around -- it is what makes the detection
/// free. A one-token opener can be recognized from the sampled id
/// alone, before anything is detokenized, on every step, for nothing. A
/// multi-token opener would need the decoded text and a partial-match
/// buffer on the hot path, to place a hint. Not worth it: a checkpoint
/// whose opener is not a single token simply gets no anchors, and
/// caching degrades to ordinary prefix reuse.
///
/// `encode` is the served tokenizer's encoder, without special-token
/// handling -- the opener is being looked up as literal text.
pub fn resolve_anchor_token(
    opener: Option<&str>,
    encode: impl Fn(&str) -> Vec<u32>,
) -> Option<u32> {
    let opener = opener?;
    if opener.is_empty() {
        return None;
    }
    let ids = encode(opener);
    match ids.as_slice() {
        [single] => Some(*single),
        _ => None,
    }
}

/// One request's anchor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnchorState {
    anchor_len: Option<usize>,
}

impl AnchorState {
    pub fn new() -> Self {
        AnchorState::default()
    }

    /// The sequence length at which this request opened a tool call, if
    /// it has.
    pub fn anchor_len(&self) -> Option<usize> {
        self.anchor_len
    }

    pub fn clear(&mut self) {
        self.anchor_len = None;
    }

    /// Offer one sampled token.
    ///
    /// The **first** tool call of a turn is the anchor and later ones
    /// are ignored: a turn that calls three tools rejoins at the first,
    /// because everything from there on is what the harness rewrites. A
    /// token that ends the generation is not an anchor either -- there
    /// is no continuation to rejoin.
    ///
    /// `position` is the sequence length *including* this token.
    pub fn observe(
        &mut self,
        token: u32,
        anchor_token: Option<u32>,
        position: usize,
        finished: bool,
    ) -> bool {
        if finished || self.anchor_len.is_some() || anchor_token != Some(token) {
            return false;
        }
        self.anchor_len = Some(position);
        true
    }
}

/// The geometry a window slide reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowPolicy {
    pub sliding_window: usize,
    pub page_size: usize,
    /// How many decode steps between slides. Sliding every step would
    /// cost a pool operation per token; the pool floor pays for the
    /// tokens that accumulate in between.
    pub eviction_interval: usize,
}

impl WindowPolicy {
    pub fn new(sliding_window: usize, page_size: usize) -> Self {
        WindowPolicy {
            sliding_window,
            page_size,
            eviction_interval: crate::policy::pool_budget::DEFAULT_SWA_EVICTION_INTERVAL,
        }
    }

    pub fn with_eviction_interval(mut self, interval: usize) -> Self {
        // A zero interval would slide on every step *and* divide by
        // zero on the cadence check.
        self.eviction_interval = interval.max(1);
        self
    }
}

/// One request's position, as a slide sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlidingRequest {
    /// The sequence length the request has reached.
    pub position: usize,
    /// How far its window state has already been released.
    pub already_released: usize,
    /// The prefix the prefix cache owns. Never released by the
    /// request: those pages are shared, and freeing them would take
    /// them out from under every other request holding the same prefix.
    pub locked_prefix: usize,
    /// Decode steps this request has taken. Step 0 is skipped: its
    /// state may still be in flight from the prefill that produced it.
    pub decode_step: usize,
}

/// What a slide decided.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlideDecision {
    /// Positions whose window state to release, `[free_from,
    /// free_to)`. Empty when the two are equal.
    pub free_from: usize,
    pub free_to: usize,
    /// The cursor has drifted too far past the anchor to keep holding
    /// it; the caller should clear its [`AnchorState`].
    pub drop_anchor: bool,
}

impl SlideDecision {
    pub fn frees_nothing(&self) -> bool {
        self.free_to <= self.free_from
    }

    pub fn released(&self) -> usize {
        self.free_to.saturating_sub(self.free_from)
    }
}

/// The window slide, run during decode.
///
/// `None` means "not this step" -- either the cadence has not come
/// round or the request is on its first decode step. Both are cheap
/// checks the caller would otherwise have to remember to make.
///
/// The anchor, when there is one, only ever makes the slide *less*
/// aggressive: `min` with the anchor's cap, never `max`. An anchor
/// cannot cause state to be freed that would otherwise have been kept.
pub fn decode_slide(
    request: &SlidingRequest,
    anchor: Option<usize>,
    policy: &WindowPolicy,
    forward_iter: usize,
) -> Option<SlideDecision> {
    if !forward_iter.is_multiple_of(policy.eviction_interval) {
        return None;
    }
    if request.decode_step < 1 {
        return None;
    }
    // Signed throughout: an early request's threshold is legitimately
    // negative (it has not produced a window's worth of tokens yet),
    // and clamping it to zero before the anchor comparison would make
    // the drop rule fire on requests that have barely started.
    let window = policy.sliding_window as i64;
    let gap = SWA_RETAIN_GAP as i64;
    let page = policy.page_size as i64;
    let mut threshold = (request.position as i64 - 1) - window - page;
    let mut drop_anchor = false;

    if let Some(anchor_len) = anchor {
        let cap = anchor_len as i64 - window - gap;
        if threshold - cap > window + gap {
            // The cursor is more than a window past what the anchor
            // wants held. Holding both would need unbounded pool.
            drop_anchor = true;
        } else {
            threshold = threshold.min(cap);
        }
    }

    Some(release_span(threshold, request, page, drop_anchor))
}

/// The prefill sibling of [`decode_slide`].
///
/// Runs on **every** prefill batch rather than on a cadence: a chunked
/// prefill can add thousands of tokens in one step, so waiting for a
/// step count would let a long prompt hold its whole history's window
/// state. The threshold is measured from what the request has computed
/// rather than from its cursor, and the anchor plays no part -- a
/// prompt being prefilled has not called a tool yet.
pub fn prefill_slide(request: &SlidingRequest, policy: &WindowPolicy) -> SlideDecision {
    let threshold =
        request.position as i64 - policy.sliding_window as i64 - policy.page_size as i64;
    release_span(threshold, request, policy.page_size as i64, false)
}

fn release_span(
    threshold: i64,
    request: &SlidingRequest,
    page: i64,
    drop_anchor: bool,
) -> SlideDecision {
    // A negative threshold aligns further negative, so it can never
    // exceed `start` and nothing is released -- which is the answer for
    // a request that has not yet produced a window's worth of tokens.
    let aligned = if threshold < 0 {
        threshold - (page + threshold % page) % page
    } else {
        align_down(threshold as usize, page as usize) as i64
    };
    let start = request.already_released.max(request.locked_prefix);
    let free_to = if aligned > start as i64 {
        aligned as usize
    } else {
        start
    };
    SlideDecision {
        free_from: start,
        free_to,
        drop_anchor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::parser::ToolCallFormat;

    const WINDOW: usize = 512;
    const PAGE: usize = 64;

    fn policy() -> WindowPolicy {
        WindowPolicy::new(WINDOW, PAGE)
    }

    fn request(position: usize) -> SlidingRequest {
        SlidingRequest {
            position,
            already_released: 0,
            locked_prefix: 0,
            decode_step: 1,
        }
    }

    /// The opener is looked up as literal text and only counts when the
    /// tokenizer spells it with one token -- that is what makes
    /// detection an integer compare on the hot path.
    #[test]
    fn an_anchor_token_is_only_resolved_when_it_is_a_single_token() {
        let single = resolve_anchor_token(Some("<tool_call>"), |_| vec![151657]);
        assert_eq!(single, Some(151657));

        let split = resolve_anchor_token(Some("<tool_call>"), |_| vec![27, 14172, 13]);
        assert_eq!(split, None, "a multi-token opener gets no anchors");
        assert_eq!(resolve_anchor_token(None, |_| vec![1]), None);
        assert_eq!(resolve_anchor_token(Some(""), |_| vec![1]), None);
    }

    /// Every family that has an opener can be asked for one; the
    /// harmony format has none, and must not invent one.
    #[test]
    fn every_format_with_an_opener_can_resolve_an_anchor() {
        for format in [
            ToolCallFormat::Qwen25,
            ToolCallFormat::Qwen3Coder,
            ToolCallFormat::Glm47,
            ToolCallFormat::Llama3,
            ToolCallFormat::Mistral,
            ToolCallFormat::DeepSeekV32,
            ToolCallFormat::MiniMax,
            ToolCallFormat::Gemma4,
        ] {
            assert_eq!(
                resolve_anchor_token(format.opener(), |_| vec![7]),
                Some(7),
                "{format:?}"
            );
        }
        assert_eq!(
            resolve_anchor_token(ToolCallFormat::GptOss.opener(), |_| vec![7]),
            None,
            "a channel header opens ordinary messages too"
        );
    }

    #[test]
    fn the_first_tool_call_of_a_turn_is_the_anchor() {
        let mut state = AnchorState::new();
        assert!(!state.observe(5, Some(9), 100, false), "not the opener");
        assert!(state.observe(9, Some(9), 101, false));
        assert_eq!(state.anchor_len(), Some(101));

        // A later call does not move it: the harness rewrites from the
        // first one on.
        assert!(!state.observe(9, Some(9), 140, false));
        assert_eq!(state.anchor_len(), Some(101));
    }

    /// A token that ends the generation has no continuation to rejoin,
    /// so it is not an anchor.
    #[test]
    fn a_terminal_token_is_not_an_anchor() {
        let mut state = AnchorState::new();
        assert!(!state.observe(9, Some(9), 101, true));
        assert_eq!(state.anchor_len(), None);
    }

    #[test]
    fn a_checkpoint_with_no_anchor_token_never_anchors() {
        let mut state = AnchorState::new();
        assert!(!state.observe(9, None, 101, false));
        assert_eq!(state.anchor_len(), None);
    }

    /// The cadence is not an optimization detail: sliding every step
    /// would cost a pool operation per token.
    #[test]
    fn the_slide_runs_on_a_cadence_and_never_on_the_first_step() {
        let policy = policy();
        let request = request(4096);
        assert!(decode_slide(&request, None, &policy, 1).is_none());
        assert!(decode_slide(&request, None, &policy, 128).is_some());

        let first_step = SlidingRequest {
            decode_step: 0,
            ..request
        };
        assert!(
            decode_slide(&first_step, None, &policy, 128).is_none(),
            "step 0's state may still be in flight from its prefill"
        );
    }

    #[test]
    fn a_long_generation_releases_everything_outside_its_window() {
        let policy = policy();
        let decision = decode_slide(&request(4096), None, &policy, 128).expect("a slide is due");
        // (4096 - 1) - 512 - 64 = 3519, aligned down to a 64-page.
        assert_eq!(decision.free_from, 0);
        assert_eq!(decision.free_to, 3456);
        assert!(!decision.drop_anchor);
    }

    /// A request that has not yet produced a window's worth of tokens
    /// releases nothing, rather than underflowing into a huge span.
    #[test]
    fn a_short_generation_releases_nothing() {
        let policy = policy();
        for position in [0usize, 1, 63, 100, 500] {
            let decision =
                decode_slide(&request(position), None, &policy, 128).expect("a slide is due");
            assert!(
                decision.frees_nothing(),
                "position {position} released {}",
                decision.released()
            );
        }
    }

    /// Shared pages are never released by one request: the prefix the
    /// cache owns is held on behalf of everyone using it.
    #[test]
    fn the_cache_owned_prefix_is_never_released() {
        let policy = policy();
        let request = SlidingRequest {
            locked_prefix: 3000,
            ..request(4096)
        };
        let decision = decode_slide(&request, None, &policy, 128).expect("a slide is due");
        assert_eq!(decision.free_from, 3000);
        assert_eq!(decision.free_to, 3456);
    }

    #[test]
    fn a_slide_never_re_releases_what_it_already_released() {
        let policy = policy();
        let request = SlidingRequest {
            already_released: 3456,
            ..request(4096)
        };
        let decision = decode_slide(&request, None, &policy, 128).expect("a slide is due");
        assert!(decision.frees_nothing());
    }

    /// The point of the whole module: with an anchor, the slide stops
    /// short of it so the next turn can rejoin there.
    #[test]
    fn an_anchor_holds_the_state_the_next_turn_will_rejoin_at() {
        let policy = policy();
        let anchor = 3000;
        // Still inside the drift bound; see the drop test below for
        // where it ends.
        let unanchored = decode_slide(&request(3500), None, &policy, 128).unwrap();
        let anchored = decode_slide(&request(3500), Some(anchor), &policy, 128).unwrap();

        assert!(
            anchored.free_to < unanchored.free_to,
            "the anchor must hold state the plain slide would have taken"
        );
        // The window before the anchor stays live.
        assert!(anchored.free_to <= anchor - WINDOW - SWA_RETAIN_GAP);
        assert!(!anchored.drop_anchor);
    }

    /// An anchor may only ever hold *more* state, never free state a
    /// plain slide would have kept.
    #[test]
    fn an_anchor_never_makes_the_slide_more_aggressive() {
        let policy = policy();
        for position in (600..6000).step_by(97) {
            for anchor in (100..position).step_by(211) {
                let plain = decode_slide(&request(position), None, &policy, 128).unwrap();
                let anchored =
                    decode_slide(&request(position), Some(anchor), &policy, 128).unwrap();
                assert!(
                    anchored.free_to <= plain.free_to,
                    "position {position} anchor {anchor}: anchored freed further"
                );
            }
        }
    }

    /// And once the cursor has drifted more than a window past what the
    /// anchor wants held, the anchor is dropped -- otherwise the
    /// request holds two windows with an unbounded gap between them,
    /// and the pool floor that pays for this feature becomes a fiction.
    #[test]
    fn a_drifted_anchor_is_dropped_rather_than_held_forever() {
        let policy = policy();
        let anchor = 3000;
        // cap = 3000 - 512 - 16 = 2472. Drop when threshold - cap >
        // 528, i.e. threshold > 3000, i.e. position - 577 > 3000.
        let held = decode_slide(&request(3576), Some(anchor), &policy, 128).unwrap();
        assert!(!held.drop_anchor, "just inside the bound");

        let dropped = decode_slide(&request(3600), Some(anchor), &policy, 128).unwrap();
        assert!(dropped.drop_anchor, "just outside it");
        // Dropping it also stops holding for it: the slide that drops
        // the anchor is the one that reclaims its state.
        let plain = decode_slide(&request(3600), None, &policy, 128).unwrap();
        assert_eq!(dropped.free_to, plain.free_to);
    }

    /// Held state stays bounded however long the generation runs --
    /// the property the `anchor_checkpoints` pool term is sized for.
    #[test]
    fn held_state_stays_bounded_over_a_long_generation() {
        let policy = policy();
        let anchor_at = 3000usize;
        let mut anchor = None;
        let mut released = 0usize;
        let bound = 2 * (WINDOW + SWA_RETAIN_GAP) + policy.eviction_interval + 2 * PAGE;

        // 100k, the length the acceptance names. The property is a
        // CEILING checked at every step, so a longer run is strictly
        // more evidence: a leak that only shows past 60k would pass the
        // shorter loop and still be a leak.
        for position in (128..100_000).step_by(128) {
            if position >= anchor_at && anchor.is_none() {
                anchor = Some(anchor_at);
            }
            let request = SlidingRequest {
                position,
                already_released: released,
                locked_prefix: 0,
                decode_step: 1,
            };
            let decision = decode_slide(&request, anchor, &policy, 128).expect("a slide is due");
            if decision.drop_anchor {
                anchor = None;
            }
            released = released.max(decision.free_to);
            assert!(
                position - released <= bound,
                "position {position} holds {} tokens, over the {bound} the pool is sized for",
                position - released
            );
        }
        assert_eq!(anchor, None, "the anchor was dropped as the cursor ran on");
    }

    /// Prefill slides on every batch, not on a cadence: one chunked
    /// prefill step can add thousands of tokens.
    #[test]
    fn prefill_slides_on_every_batch() {
        let policy = policy();
        let decision = prefill_slide(&request(4096), &policy);
        // 4096 - 512 - 64 = 3520, already page-aligned.
        assert_eq!(decision.free_to, 3520);
        assert!(!decision.drop_anchor);
        assert!(prefill_slide(&request(100), &policy).frees_nothing());
    }
}
