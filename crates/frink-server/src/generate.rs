//! Shared token-generation loop for both the non-streaming and SSE
//! streaming `/v1/chat/completions` paths: sampling
//! (temperature/top-p/top-k/repetition penalty) with a greedy-argmax
//! path at `temperature<=0.0`,
//! plus stop-sequence handling that's correct even when a stop string
//! spans more than one generated token.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::policy::anchor::{decode_slide, AnchorState, SlidingRequest, WindowPolicy};
use crate::policy::pool_budget::SWA_RETAIN_GAP;
use crate::policy::radix::{align_down, NodeId, RadixCache};
use frink_core::cache::{
    KvBlockPool, KvCache, KvPoolExhausted as CacheKvPoolExhausted, PageGroup, PagedKvCache,
    PagedStoreExhausted, SharedPagedKv,
};
use frink_models::sampling::SamplingParams;
use frink_models::tokenizer::{prepend_bos, SpecialTokens, StopTokens};
use frink_models::{Ceiling, Decoder, Engine, KvElem, KvShape, PrefixCache, TextTokenizer};

use crate::budget::ContextCeiling;

use crate::model::ServerTokenizer;

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("prompt encoded to token id {token}, which is outside this model's vocabulary of {vocab_size} (its tokenizer does not match this checkpoint)")]
    TokenOutOfVocab { token: usize, vocab_size: usize },
    #[error("server is at capacity: the shared KV cache block pool has no free blocks for a new request; retry shortly")]
    KvPoolExhausted,
    /// A well-formed request this deployment cannot serve, named rather
    /// than approximated. The one case today is `n` > 1 on the paged
    /// store, whose block lists have no copy-on-write, so the forks the
    /// feature is FOR cannot be taken.
    #[error("{0}")]
    Unsupported(String),
    /// The batch scheduler's admission queue is full. Distinct from
    /// `KvPoolExhausted`: nothing is exhausted, the server is simply
    /// further behind than it is willing to queue. Naming the depth and
    /// the cap keeps a retry storm diagnosable -- an operator can tell
    /// "too many clients" from "one request too big".
    #[error("server is at capacity: {queued} requests are already queued for the batch scheduler (limit {cap}); retry shortly")]
    QueueFull { queued: usize, cap: usize },
    /// The request cannot fit a ceiling that will not move: a typed
    /// refusal rather than an out-of-memory kill somewhere downstream.
    ///
    /// `binding` names *which* ceiling, because the two send an
    /// operator to different knobs -- `context_length_exceeded` is the
    /// request's size against what this deployment admits per request,
    /// `device_memory_budget_exceeded` is the whole server's KV budget.
    /// `estimated_bytes` and `limit_bytes` are the KV cost of the
    /// request and of the ceiling, so the arithmetic is checkable
    /// rather than asserted.
    ///
    /// Deliberately not a 503: an idle server refuses this identically,
    /// so "retry shortly" would be a lie.
    #[error("{binding}: {detail}")]
    KvBudgetExceeded {
        /// Machine-readable ceiling code from
        /// [`frink_models::Ceiling::code`].
        binding: &'static str,
        /// KV bytes this request would cost at its full length.
        estimated_bytes: u64,
        /// KV bytes the binding ceiling allows.
        limit_bytes: u64,
        /// Token positions the request asked for (prompt + max_tokens).
        positions: usize,
        /// Token positions the binding ceiling allows.
        positions_limit: usize,
        detail: String,
    },
    /// Grammar-constrained decoding could not continue.
    ///
    /// Two live causes, both properties of the *request* rather than of
    /// the server: a grammar this model's vocabulary cannot spell, so
    /// every logit was masked; and a sampled token the grammar refused,
    /// which means the mask and the accept disagreed. Either way the
    /// only alternative is to emit a token the caller's grammar forbids
    /// and report it as constrained output, so this stops instead. See
    /// [`frink_models::grammar_sampler::ConstraintError`], whose text
    /// is carried in `detail`.
    #[error("grammar-constrained decoding stopped: {detail}")]
    GrammarConstraint { detail: String },
    /// A reasoning budget reached the sampler that it could not
    /// enforce: unresolved (a path skipped the tokenizer seam that
    /// turns the caller's number into token sequences) or asked of a
    /// family whose closer cannot be forced. Either way the only
    /// alternative is an unbounded thought served as a bounded one, so
    /// this stops instead. Both causes are meant to be caught earlier
    /// -- the route refuses the family with a 501 -- which is why this
    /// is an error and not a fallback.
    #[error("reasoning budget could not be enforced: {detail}")]
    ReasoningBudget { detail: String },
}

impl DecodeError {
    /// Seconds to advise a client to wait before retrying, or `None`
    /// for an error retrying cannot fix.
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            DecodeError::TokenOutOfVocab { .. } => None,
            // A feature this deployment does not have does not appear
            // on a retry.
            DecodeError::Unsupported(_) => None,
            // The same grammar against the same vocabulary fails the
            // same way on every retry.
            DecodeError::GrammarConstraint { .. } => None,
            DecodeError::ReasoningBudget { .. } => None,
            // Retrying an over-budget request changes nothing: the
            // ceiling it hit is the whole server, not the current load.
            DecodeError::KvBudgetExceeded { .. } => None,
            DecodeError::KvPoolExhausted | DecodeError::QueueFull { .. } => Some(1),
        }
    }
}

/// A shared `KvBlockPool` plus this server's admission-control wait
/// policy: how long a request is willing to retry acquiring its
/// per-layer caches before giving up, when the pool is momentarily
/// exhausted. `queue_wait: Duration::ZERO` (the default) means "try
/// once, reject immediately" -- the original reject-only behavior.
#[derive(Clone)]
pub struct KvPoolConfig {
    pub pool: Arc<Mutex<KvBlockPool>>,
    pub queue_wait: Duration,
}

/// The paged counterpart to [`KvPoolConfig`]: per-layer
/// [`SharedPagedKv`] storage every request draws pages from, plus the
/// same admission wait policy.
///
/// Mutually exclusive with `KvPoolConfig` -- they are two answers to
/// the same question, and a deployment picks one.
#[derive(Clone)]
pub struct PagedKvConfig {
    pub store: Arc<SharedPagedKv>,
    pub queue_wait: Duration,
    /// `Some` when prefix sharing is on: a radix tree from token
    /// prefixes to the page groups holding their KV.
    ///
    /// This is what `frink-models::prefix_cache` could not be. That
    /// one CLONES a `Vec<KvCache>` per entry, so N conversations off
    /// one system prompt hold N copies of its KV. The radix tree stores
    /// page groups and reference-counts them, so they hold one.
    pub radix: Option<Arc<Mutex<RadixCache>>>,
    /// The single token that opens a tool call for this checkpoint, when
    /// its family has one and it encodes to exactly one token.
    ///
    /// `None` costs nothing but the anchor: the slide still runs, it
    /// just follows the cursor instead of stopping short of the position
    /// the next agentic turn will rejoin at.
    pub anchor_token: Option<u32>,
    /// Decode steps between window slides.
    ///
    /// Sliding every step would cost a page operation per token for a
    /// page's worth of pages every `block_size` tokens; the admission
    /// bound pays for what accumulates in between, which is why this
    /// number appears in [`slide_hold_bound`] as well as here. Zero is
    /// clamped to one by [`WindowPolicy::with_eviction_interval`], since
    /// it would otherwise divide by zero on the cadence check.
    pub slide_interval: usize,
}

/// The sliding-window state one paged request carries.
///
/// Present only when [`ModelConfig::uniform_sliding_window`] said every
/// layer slides. A page group holds one block in every layer, so a model
/// with even one full-attention layer cannot give a group back -- see
/// that method for why the narrowest window is the wrong answer there.
///
/// [`ModelConfig::uniform_sliding_window`]: frink_models::ModelConfig::uniform_sliding_window
struct WindowSlide {
    policy: WindowPolicy,
    /// Positions whose pages this request has already recycled. Behind
    /// the window, so nothing reads them again.
    released: usize,
    /// The prefix the radix tree owns, which is shared with every other
    /// request holding it and so is never recycled however far behind
    /// the window it falls.
    locked_prefix: usize,
    /// Decode steps taken. `decode_slide` skips step 0, whose state may
    /// still be in flight from the prefill that produced it.
    decode_step: usize,
    anchor: AnchorState,
    anchor_token: Option<u32>,
}

/// Positions a sliding request may hold beyond its prompt.
///
/// This is the ceiling `frink-edge`'s own bound test asserts at every
/// step of a 100_000-token run, and it is what makes admission by the
/// window sound rather than hopeful. Each term is a real reason the
/// slide lags the cursor:
///
/// - `window` is what attention still reads;
/// - a second `window + gap` is the anchor's, which caps the threshold
///   at `anchor - window - gap` until the cursor drifts a whole window
///   past it and the anchor is dropped;
/// - `eviction_interval` is the cadence -- positions accumulate between
///   slides, which is the point of not sliding every step;
/// - two pages cover the `- page` in the threshold and the alignment
///   down to a page boundary.
fn slide_hold_bound(window: usize, policy: &WindowPolicy) -> usize {
    2 * (window + SWA_RETAIN_GAP) + policy.eviction_interval + 2 * policy.page_size
}

/// One request's paged KV, which returns its pages when dropped.
///
/// `PagedKvCache` has no `Drop` of its own -- releasing needs a
/// `&mut PagedKvStore` it does not hold a reference to -- so without
/// this, every refusal path and every `?` between admission and the end
/// of generation leaks the whole request's pages until the process
/// exits. `KvCache::with_pool` gets this for free from its own `Drop`;
/// the paged side has to be given it.
pub struct PagedLease {
    caches: Vec<PagedKvCache>,
    store: Arc<SharedPagedKv>,
    /// This sequence's page groups, in position order: `groups[i]`
    /// holds positions `[i * block_size, (i+1) * block_size)`.
    ///
    /// Held as groups rather than per-layer block ids because that is
    /// the unit the radix tree shares and reference-counts. Every entry
    /// here is one the lease must release exactly once.
    /// `None` where a sliding window has recycled the group away: that
    /// index is behind the window, nothing reads it, and its group is
    /// waiting in `spare` to back a later position.
    groups: Vec<Option<PageGroup>>,
    /// Groups this request recycled, kept PRIVATE rather than handed
    /// back to the store.
    ///
    /// Handing them back would be the generous thing and it would also
    /// make `push` fallible again: another request could take the page
    /// this one is about to need, mid-answer, with nowhere to report it.
    /// The generosity happens once, at admission, where a sliding
    /// request asks for a window's worth instead of a whole context's --
    /// which is the larger saving anyway, and one that a refusal can
    /// still be returned from.
    spare: Vec<PageGroup>,
    /// How many leading groups came from the radix tree rather than
    /// from a fresh allocation, and the node they were matched at.
    ///
    /// The node stays locked against eviction for as long as this lease
    /// lives, because those pages are being attended over.
    adopted: Option<(usize, NodeId)>,
    radix: Option<Arc<Mutex<RadixCache>>>,
    window: Option<WindowSlide>,
}

impl Drop for PagedLease {
    fn drop(&mut self) {
        // Unlock first: while the node is locked its pages are
        // protected from eviction, and releasing our own hold before
        // unlocking would let the tree believe a page is free while
        // this lease still names it.
        if let (Some((_, node)), Some(radix)) = (self.adopted, self.radix.as_ref()) {
            radix.lock().unwrap_or_else(|p| p.into_inner()).unlock(node);
        }
        // Every group this sequence held, adopted or fresh. A group the
        // tree also holds survives this, because its refcount does not
        // reach zero.
        //
        // `flatten` rather than `unwrap`: a slid request left holes
        // where it recycled, and those groups are in `spare`. Both
        // halves are drained, and a group is in exactly one of them, so
        // each is still released exactly once.
        for group in self.groups.drain(..).flatten() {
            self.store.release_group(group);
        }
        for group in self.spare.drain(..) {
            self.store.release_group(group);
        }
    }
}

impl PagedLease {
    /// This request's per-layer paged caches, for a caller that drives
    /// the forward itself.
    ///
    /// The lease keeps owning the page GROUPS, so taking the caches out
    /// with `mem::take` and putting them back -- which the batched
    /// decode step does, to hand the whole batch to one call -- does
    /// not disturb the accounting. Only `Drop` releases groups.
    pub fn caches_mut(&mut self) -> &mut Vec<PagedKvCache> {
        &mut self.caches
    }

    pub fn store(&self) -> &Arc<SharedPagedKv> {
        &self.store
    }

    /// The store's page size, which every caller needs to turn groups
    /// into positions.
    pub fn block_size(&self) -> usize {
        self.store.read(0).block_size()
    }

    /// Positions this request did not have to compute.
    pub fn adopted_positions(&self, block_size: usize) -> usize {
        self.adopted
            .map(|(groups, _)| groups * block_size)
            .unwrap_or(0)
    }

    /// Whether this request's window has taken any page away.
    ///
    /// Load-bearing at publish time: a slid sequence cannot be published
    /// to the radix tree. The tree keys on a PREFIX and this sequence's
    /// prefix is exactly the part that is gone -- what it still holds is
    /// a suffix. Publishing anyway would hand the next request a page
    /// whose contents belong to a position a thousand tokens later.
    pub fn has_slid(&self) -> bool {
        self.window.as_ref().is_some_and(|w| w.released > 0)
    }

    /// Offer one sampled token to the anchor detector.
    ///
    /// Separate from [`Self::before_step`] because the two happen at
    /// different moments: a token is observed once it has been sampled,
    /// and the slide runs before the forward that consumes it.
    pub fn observe_sampled(&mut self, token: usize, position: usize, finished: bool) {
        if let Some(w) = self.window.as_mut() {
            w.anchor
                .observe(token as u32, w.anchor_token, position, finished);
        }
    }

    /// Run one decode step's window slide, then make sure the page
    /// `position` will be written into exists.
    ///
    /// Both halves here, in this order, because they are two ends of one
    /// mechanism: the slide is what produces the spare page that the
    /// extension then installs. Splitting them would let a caller do the
    /// second without the first and quietly fall back to taking a page
    /// from the store, which is the failure this whole design removes.
    pub fn before_step(&mut self, position: usize) {
        if self.window.is_none() {
            return;
        }
        let block_size = self.block_size();
        self.slide(position, block_size);
        self.extend_to(position, block_size);
        // `debug_assert!` still TYPE-CHECKS its argument in release, so
        // calling a `#[cfg(debug_assertions)]` method from inside one
        // does not compile without debug assertions. The cfg has to be
        // on the call, not only on the definition.
        #[cfg(debug_assertions)]
        debug_assert!(
            self.tables_match_groups(),
            "a sliding lease must own every block its tables name"
        );
    }

    fn slide(&mut self, position: usize, block_size: usize) {
        let Some(w) = self.window.as_mut() else {
            return;
        };
        w.decode_step += 1;
        let request = SlidingRequest {
            position,
            already_released: w.released,
            locked_prefix: w.locked_prefix,
            decode_step: w.decode_step,
        };
        // `forward_iter` and `decode_step` are the same counter here:
        // one request's cadence is its own step count. A batched engine
        // that wanted every row to slide on the same iteration would
        // pass the batcher's counter instead, and the policy would
        // still be this one.
        let Some(decision) =
            decode_slide(&request, w.anchor.anchor_len(), &w.policy, w.decode_step)
        else {
            return;
        };
        if decision.drop_anchor {
            w.anchor.clear();
        }
        if decision.frees_nothing() {
            return;
        }
        // Both ends are page-aligned -- `free_from` is the previous
        // `free_to` or the locked prefix, `free_to` is aligned down --
        // so this divides exactly and never half-frees a page.
        let (from, to) = (
            decision.free_from / block_size,
            decision.free_to / block_size,
        );
        for slot in &mut self.groups[from..to] {
            if let Some(group) = slot.take() {
                self.spare.push(group);
            }
        }
        w.released = decision.free_to;
    }

    /// Installs a page at the index `position` belongs to, if the block
    /// tables do not reach that far yet.
    ///
    /// Recycled spare first, a fresh group from the store second. The
    /// order matters more than the fallback ever firing: taking the
    /// spare is what keeps a long generation's footprint flat, and the
    /// store acquire is there so that `groups` stays the *only* owner of
    /// every block in the tables. Letting `PagedKvCache::push` grow a
    /// table by itself would acquire a block this lease never records,
    /// and `Drop` releases what `groups` names -- so that block would be
    /// gone until the process exits.
    ///
    /// Extending nothing when the store is empty too is the one clean
    /// answer left: the tables are unchanged, and the caller's `reserve`
    /// refuses having taken nothing.
    fn extend_to(&mut self, position: usize, block_size: usize) {
        let index = position / block_size;
        while self.groups.len() <= index {
            let Some(group) = self.spare.pop().or_else(|| self.store.acquire_group()) else {
                return;
            };
            // A recycled group's physical blocks now back two table
            // indices: the stale one the slide emptied, and this one.
            // Safe precisely because the stale index is behind the
            // window and the kernel never reads it -- see
            // `PagedKvCache::append_block`.
            let blocks = self.store.group_blocks(group);
            for (cache, &block) in self.caches.iter_mut().zip(&blocks) {
                cache.append_block(block);
            }
            self.groups.push(Some(group));
        }
    }

    /// Every cache's block table names exactly the groups this lease
    /// holds, in order.
    ///
    /// The property the whole recycling scheme rests on: `Drop` releases
    /// `groups`, so a table entry that no group accounts for is a leaked
    /// page and a group no table names is a page nothing can read.
    #[cfg(debug_assertions)]
    fn tables_match_groups(&self) -> bool {
        self.caches
            .iter()
            .all(|c| c.block_table().len() == self.groups.len())
    }
}

/// How many page groups a request must hold to run to `max_seq_len`
/// without ever asking the store for another one.
///
/// Without a window that is the whole sequence, and reserving it all up
/// front is not an optimisation -- it is what makes the decode loop's
/// signature honest. `sample_until_stop` takes a closure returning
/// `Vec<f32>`, with nowhere to report a store that ran dry at token 300
/// of 400, the same reason `acquire_pooled_caches` sizes for
/// `max_seq_len` rather than growing (a real panic in live testing).
///
/// WITH a window the answer is the prompt plus a bound, because the
/// slide gives pages back to this request faster than decode consumes
/// them. The prompt term is not a window's worth: prefill materialises
/// positions `0..prompt_len` to reuse the one prefill kernel
/// (`PagedKvCache::to_contiguous`), so every prompt page must still be
/// there while it runs. What the window removes is the *generation*
/// term -- a 4k-window model answering 100k tokens holds its prompt and
/// a window, not a prompt and 100k.
///
/// `prompt_len + bound` is safe at every position, in two cases. Below
/// `prompt_len + bound` it is trivially safe, since nothing is ever
/// released. Above it, `frink-edge`'s own ceiling applies -- the one
/// its bound test asserts at every step of a 100_000-token run -- and
/// the live span is at most `bound` on its own.
fn paged_groups_needed(
    max_seq_len: usize,
    prompt_len: usize,
    block_size: usize,
    window: Option<&WindowPolicy>,
) -> usize {
    paged_hold_positions(max_seq_len, prompt_len, block_size, window)
        .div_ceil(block_size)
        .max(1)
}

/// The same answer in POSITIONS, which is the unit the batch
/// scheduler's block budget speaks.
///
/// One function for both because they are one decision. The budget
/// bounds how many requests the server admits at once and the store
/// bounds whether each of them can run; a budget that priced a windowed
/// request at its whole context would keep refusing admissions the
/// store would happily serve, and the two would disagree about the same
/// server.
///
/// The extra page is slack over the bound, and the `min` is what stops
/// a short request from being made *more* expensive by being windowed.
pub(crate) fn paged_hold_positions(
    max_seq_len: usize,
    prompt_len: usize,
    block_size: usize,
    window: Option<&WindowPolicy>,
) -> usize {
    let Some(policy) = window else {
        return max_seq_len;
    };
    (prompt_len + slide_hold_bound(policy.sliding_window, policy) + block_size).min(max_seq_len)
}

/// The window policy a paged request runs under on this model, or
/// `None` when it may not slide at all.
pub(crate) fn paged_window_policy(
    decoder: &Decoder,
    config: &PagedKvConfig,
) -> Option<WindowPolicy> {
    let block_size = config.store.read(0).block_size();
    decoder
        .config
        .uniform_sliding_window()
        .map(|w| WindowPolicy::new(w, block_size).with_eviction_interval(config.slide_interval))
}

/// Reserves everything this request can need up front, retrying until
/// `config.queue_wait` elapses.
///
/// "Everything it can need" is [`paged_groups_needed`], which is the
/// whole sequence for a full-attention model and prompt-plus-a-window
/// for a sliding one. A request that cannot fit is refused here, before
/// any work, rather than dying halfway through an answer.
pub(crate) fn acquire_paged_caches(
    decoder: &Decoder,
    config: &PagedKvConfig,
    tokens: &[usize],
    max_seq_len: usize,
) -> Result<PagedLease, PagedStoreExhausted> {
    let block_size = config.store.read(0).block_size();
    let deadline = Instant::now() + config.queue_wait;

    // Consult the tree ONCE, before the retry loop. A match locks the
    // node, so re-matching per attempt would take a second lock on the
    // same node and the unlock on drop would balance only one of them,
    // leaving the prefix pinned forever.
    let adopted = match config.radix.as_ref() {
        Some(radix) => {
            let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let mut tree = radix.lock().unwrap_or_else(|p| p.into_inner());
            let m = tree.match_prefix(&ids);
            // Never adopt the WHOLE prompt. Prefill has to run over at
            // least one token to produce the logits that predict the
            // next one, and a fully-adopted prompt leaves nothing to
            // run. Backing off one page is the cheap answer; the old
            // contiguous prefix cache hit the same wall and backed off
            // one POSITION, which paging cannot do because a partly
            // shared page cannot be written.
            let cap = align_down(ids.len().saturating_sub(1), block_size);
            let cached_len = m.cached_len.min(cap);
            if cached_len == 0 {
                None
            } else {
                tree.lock(m.node);
                // One index per TOKEN, so consecutive tokens in a page
                // repeat its group. Step by `block_size` to get each
                // group once, in position order.
                let per_token = tree.matched_indices(m.node);
                let groups: Vec<PageGroup> = per_token[..cached_len]
                    .iter()
                    .step_by(block_size)
                    .map(|&g| PageGroup(g))
                    .collect();
                Some((cached_len, m.node, groups))
            }
        }
        None => None,
    };
    // Every adopted group gains a holder for the life of this lease.
    if let Some((_, _, groups)) = adopted.as_ref() {
        for &g in groups {
            config.store.retain_group(g);
        }
    }
    let (cached_len, node, adopted_groups) = match adopted {
        Some((len, node, groups)) => (len, Some(node), groups),
        None => (0, None, Vec::new()),
    };

    // A model whose layers do not all slide by the same window cannot
    // give a page group back at all: the group holds a block in every
    // layer, and a full-attention layer still reads position 0.
    let policy = paged_window_policy(decoder, config);

    // Only what the adopted prefix does not already cover.
    let total_groups = paged_groups_needed(max_seq_len, tokens.len(), block_size, policy.as_ref());
    let need = total_groups.saturating_sub(adopted_groups.len());
    let make_window = || {
        policy.map(|policy| WindowSlide {
            policy,
            released: 0,
            // The adopted prefix is the tree's, shared with every other
            // request holding it, so the slide floors here rather than
            // at zero.
            locked_prefix: cached_len,
            decode_step: 0,
            anchor: AnchorState::new(),
            anchor_token: config.anchor_token,
        })
    };

    loop {
        let mut fresh: Vec<PageGroup> = Vec::with_capacity(need);
        while fresh.len() < need {
            match config.store.acquire_group() {
                Some(g) => fresh.push(g),
                None => break,
            }
        }
        if fresh.len() == need {
            let mut groups = adopted_groups;
            groups.extend(fresh);
            let caches = seed_caches(decoder, config, &groups, cached_len, block_size);
            return Ok(PagedLease {
                caches,
                store: Arc::clone(&config.store),
                groups: groups.into_iter().map(Some).collect(),
                spare: Vec::new(),
                adopted: node.map(|n| (cached_len / block_size, n)),
                radix: config.radix.clone(),
                window: make_window(),
            });
        }
        // Give back what this attempt took before waiting, or a
        // request that never fits holds pages the requests that would
        // fit are waiting for.
        let short = need - fresh.len();
        for g in fresh {
            config.store.release_group(g);
        }
        // THEN reclaim from the tree, which is the only thing that ever
        // gives these pages back.
        //
        // `publish_to_radix` retains a group for every page it hands
        // the tree, and nothing released them, so the pool shrank
        // monotonically: a long-running server ended up refusing
        // requests that fit, while the tree sat on pages no request was
        // reading. `evict` was written for exactly this call (its own
        // doc says asking for more than `evictable_size` means "the
        // admission arithmetic promised memory the cache never had")
        // and had no caller at all.
        //
        // Only unlocked nodes are evictable, so a prefix some live
        // lease adopted cannot be taken out from under it. The adopted
        // node above is locked before this point for that reason.
        let mut reclaimed = 0usize;
        if let Some(radix) = config.radix.as_ref() {
            let freed = {
                let mut tree = radix.lock().unwrap_or_else(|p| p.into_inner());
                // The tree counts TOKENS and the shortfall is in
                // GROUPS, one block per group. Never ask for more than
                // it holds unlocked: `evict` panics on that, by design,
                // because it means the caller's arithmetic was wrong.
                let want = (short * block_size).min(tree.evictable_size());
                // One index per token, so a page repeats across its
                // block. Release each group ONCE or the store's
                // refcount underflows.
                let per_token = tree.evict(want);
                per_token.into_iter().collect::<BTreeSet<u32>>()
            };
            for g in freed {
                config.store.release_group(PageGroup(g));
                reclaimed += 1;
            }
        }
        // Reclaiming is PROGRESS, not waiting, so retry immediately
        // rather than charging it against the deadline. A caller with
        // `queue_wait = 0` would otherwise evict and then give up
        // without ever trying the pages it just freed, which is a
        // refusal with the memory sitting right there.
        //
        // This terminates: every pass either frees at least one group
        // or falls through to the deadline, and `evictable_size` only
        // shrinks.
        if reclaimed > 0 {
            continue;
        }
        let now = Instant::now();
        if now >= deadline {
            // The adopted groups and the tree lock go back through the
            // lease's own Drop, which is why they are handed to one
            // here rather than released by hand: one release path, not
            // two that must agree.
            drop(PagedLease {
                caches: Vec::new(),
                store: Arc::clone(&config.store),
                groups: adopted_groups.into_iter().map(Some).collect(),
                spare: Vec::new(),
                adopted: node.map(|n| (cached_len / block_size, n)),
                radix: config.radix.clone(),
                window: make_window(),
            });
            return Err(PagedStoreExhausted);
        }
        std::thread::sleep(Duration::from_millis(10).min(deadline - now));
    }
}

/// Installs the per-layer block tables for `groups`, with `cached_len`
/// positions already computed.
fn seed_caches(
    decoder: &Decoder,
    config: &PagedKvConfig,
    groups: &[PageGroup],
    cached_len: usize,
    block_size: usize,
) -> Vec<PagedKvCache> {
    // A group holds one block per layer, so layer `l`'s table is the
    // `l`th block of each group, in order.
    let per_group: Vec<Vec<usize>> = groups
        .iter()
        .map(|&g| config.store.group_blocks(g))
        .collect();
    (0..decoder.layers.len())
        .map(|layer| {
            let table: Vec<usize> = per_group.iter().map(|blocks| blocks[layer]).collect();
            let mut cache = PagedKvCache::new();
            cache.adopt_blocks(table, cached_len, block_size);
            cache
        })
        .collect()
}

/// Publishes this request's pages under its full token sequence, so the
/// next request sharing the prefix adopts them instead of recomputing.
///
/// The duplicate count `insert_prefix` returns is the load-bearing
/// part: it is not "how much I stored", it is "how much you must free".
/// Another request published the same prefix while this one was
/// generating, the tree kept ITS pages, and ours for that span are now
/// unreferenced by the tree. Dropping them on the floor is the classic
/// leak in this shape of cache.
pub(crate) fn publish_to_radix(lease: &mut PagedLease, tokens: &[usize], block_size: usize) {
    let Some(radix) = lease.radix.clone() else {
        return;
    };
    // A sequence whose window slid has given its prefix away, and a
    // prefix is exactly what the tree keys on. What it still holds is a
    // suffix at the cursor; publishing it would hand the next request
    // pages whose contents belong to positions far past the ones it
    // matched on.
    if lease.has_slid() {
        return;
    }
    let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    // One index per token: each position names the group holding it.
    let mut per_token: Vec<u32> = Vec::with_capacity(ids.len());
    for (i, group) in lease.groups.iter().enumerate() {
        let group = group.expect("a lease that has not slid holds every group it names");
        let covered = block_size.min(ids.len().saturating_sub(i * block_size));
        for _ in 0..covered {
            per_token.push(group.0);
        }
    }
    if per_token.len() < ids.len() {
        // More tokens than pages held: nothing coherent to publish.
        return;
    }
    let result = {
        let mut tree = radix.lock().unwrap_or_else(|p| p.into_inner());
        tree.insert_prefix(&ids, &per_token[..ids.len()])
    };
    // The tree now holds a reference to every group it kept, so those
    // survive this lease's own release.
    let kept = result.cached_len / block_size..result.inserted_len / block_size;
    for i in kept {
        if let Some(Some(g)) = lease.groups.get(i) {
            lease.store.retain_group(*g);
        }
    }
}

/// Which KV representation this request is running on.
///
/// One enum rather than a second `generate`: the decode loop already
/// takes an "advance one token" closure, so paging changes three
/// expressions inside `generate` and nothing else. A parallel function
/// would duplicate the sampling, the stop matching and the usage
/// accounting, which is how the paged DECODER path lost five model
/// features one at a time.
pub(crate) enum Kv {
    Contiguous(Vec<KvCache>),
    Paged(PagedLease),
}

impl Kv {
    /// Prefill, returning the last position's logits.
    ///
    /// `host_kv` only reaches the contiguous arm: the paged arm always
    /// needs real host rows, because scattering them into the page
    /// store IS reading them back.
    /// Prefill AND score every prompt position, for
    /// `prompt_logprobs`.
    ///
    /// A separate entry point rather than a flag on [`Self::prefill`],
    /// because the two do measurably different work:
    /// `forward_batch_last` projects the lm_head ONCE, at the final
    /// position, while this projects it at every one. On a 6000-token
    /// prompt that is 6000 vocabulary-wide matmuls instead of one, and
    /// a request that did not ask for prompt logprobs must not pay for
    /// them.
    ///
    /// `None` for the paged store: its prefill skips whatever the
    /// radix tree already computed, so the positions it returns are
    /// the ones it recomputed rather than the whole prompt, and
    /// scoring a prefix nobody ran would be an invention. Refused by
    /// name at the route instead.
    fn prefill_scored(&mut self, decoder: &Decoder, tokens: &[usize]) -> Option<Vec<Vec<f32>>> {
        match self {
            Kv::Contiguous(caches) => Some(decoder.forward_batch(tokens, 0, caches)),
            Kv::Paged(_) => None,
        }
    }

    fn prefill(&mut self, decoder: &Decoder, tokens: &[usize], host_kv: bool) -> Vec<f32> {
        match self {
            Kv::Contiguous(caches) => forward_prompt_batch(decoder, tokens, 0, caches, host_kv),
            Kv::Paged(lease) => {
                // Skip whatever the radix tree already computed. Those
                // positions are already in the block table with their
                // KV written, so prefill starts where they end.
                let done = lease.adopted_positions(lease.block_size());
                decoder
                    .forward_batch_last_paged(
                        &tokens[done..],
                        done,
                        &mut lease.caches,
                        &lease.store,
                    )
                    .expect("the whole request's pages were reserved at admission")
            }
        }
    }

    /// One decode step.
    /// Feed a whole block and return one logit row per token, which
    /// is what verifying `k` drafted tokens costs instead of `k`
    /// forwards ([`Decoder::forward_batch`]).
    ///
    /// `None` means this store cannot serve speculation, and the
    /// caller takes the ordinary path rather than guessing. Two
    /// reasons, both real:
    ///
    /// * **Paged.** Rejecting a draft means releasing the positions it
    ///   wrote, and a paged store hands those out through a block
    ///   table that the radix cache may already have published a
    ///   prefix of. Undoing that is its own row
    ///   (`docs/plans/server-speculative-decoding.md`); refusing is the
    ///   honest answer until it lands.
    /// * **Recurrent.** A Mamba-style state is a REDUCTION over the
    ///   whole prefix rather than a row per position, so there is
    ///   nothing to drop. `KvCache::can_truncate_to` says so, and
    ///   [`Self::truncate_to`] asks it rather than assuming.
    #[allow(
        dead_code,
        reason = "called by the speculative loop; see sampling_loop::verify_block"
    )]
    fn step_batch(
        &mut self,
        decoder: &Decoder,
        tokens: &[usize],
        pos: usize,
    ) -> Option<Vec<Vec<f32>>> {
        match self {
            Kv::Contiguous(caches) => {
                // Asked BEFORE the forward, not after: a store that
                // cannot roll back must not be written speculatively
                // in the first place.
                if !caches.iter().all(|c| c.can_truncate_to(pos)) {
                    return None;
                }
                Some(decoder.forward_batch(tokens, pos, caches))
            }
            Kv::Paged(_) => None,
        }
    }

    /// Drop every position past `pos`, which is how a rejected draft is
    /// undone. `false` when this store cannot, and the caller must then
    /// never have speculated -- [`Self::step_batch`] is the gate.
    #[allow(
        dead_code,
        reason = "called by the speculative loop; see sampling_loop::verify_block"
    )]
    fn truncate_to(&mut self, pos: usize) -> bool {
        match self {
            Kv::Contiguous(caches) => {
                if !caches.iter().all(|c| c.can_truncate_to(pos)) {
                    return false;
                }
                for c in caches.iter_mut() {
                    c.truncate(pos);
                }
                true
            }
            Kv::Paged(_) => false,
        }
    }

    fn step(&mut self, decoder: &Decoder, token: usize, pos: usize) -> Vec<f32> {
        match self {
            Kv::Contiguous(caches) => decoder.forward_token(token, pos, caches),
            Kv::Paged(lease) => {
                // Slides the window and installs the page this position
                // writes into, before the forward that writes it. A
                // no-op for a model without a uniform window.
                lease.before_step(pos);
                decoder
                    .forward_token_paged(token, pos, &mut lease.caches, &lease.store)
                    .expect("admission reserved the prompt plus this request's window bound")
            }
        }
    }

    /// The contiguous caches, when there are any.
    ///
    /// `prefix_cache` stores `Vec<KvCache>` snapshots, so a paged
    /// request has nothing to hand it. That is why the two are refused
    /// together at startup rather than silently producing a cache that
    /// never hits -- and it is what `wire-radix-prefix-cache` removes.
    pub(crate) fn contiguous_mut(&mut self) -> Option<&mut Vec<KvCache>> {
        match self {
            Kv::Contiguous(caches) => Some(caches),
            Kv::Paged(_) => None,
        }
    }

    pub(crate) fn into_contiguous(self) -> Option<Vec<KvCache>> {
        match self {
            Kv::Contiguous(caches) => Some(caches),
            Kv::Paged(_) => None,
        }
    }
}

/// Retries acquiring one `KvCache` per layer from `config.pool` until
/// either all of them succeed or `config.queue_wait` has elapsed since
/// the first attempt. Sleeping between attempts happens on whichever
/// thread calls this -- fine here since generation already runs on
/// tokio's blocking-thread pool (`spawn_blocking`), not an async
/// reactor thread that a `std::thread::sleep` would otherwise stall.
///
/// `max_seq_len` (this request's real worst-case sequence length --
/// prompt length plus `max_tokens`) is passed straight through to
/// `KvCache::with_pool`, so each layer's cache reserves enough blocks
/// for the *whole* request up front rather than growing mid-decode.
/// This isn't just an optimization: `Decoder::forward_token`/
/// `forward_batch` treat `KvCache::push` as infallible, so a pooled
/// cache that under-reserves at construction and then fails to
/// acquire another block later (because some other request took the
/// pool's remaining capacity in the meantime) would panic mid-decode
/// instead of failing this request cleanly at admission time -- caught
/// by a real panic during live testing before this fix.
/// The typed refusal for a request the KV pool can *never* satisfy, or
/// `None` when the pool could serve it once enough blocks come back.
///
/// A request holds one `KvCache` per layer and each reserves
/// `ceil(max_seq_len / block_size)` blocks up front (see
/// [`acquire_pooled_caches`]), so its whole-pool cost is
/// `n_layers * blocks_per_layer`. When that exceeds
/// `KvBlockPool::total_blocks` the request is refused by arithmetic
/// rather than by exhaustion: an *empty* pool would refuse it
/// identically, which is precisely the test for whether "retry shortly"
/// is a true statement. Before this existed, such a request slept
/// through `FRINK_KV_POOL_QUEUE_TIMEOUT_MS` and then got a 503 with a
/// `Retry-After` that could never come good.
///
/// Priced in real KV bytes through `ceiling`'s `KvShape` when one is
/// available, so the refusal names bytes rather than an opaque block
/// count. Positions, not bytes, decide it: the pool's own ledger is in
/// positions and this must agree with the acquisition it is predicting.
fn pool_immovable_refusal(
    decoder: &Decoder,
    config: &KvPoolConfig,
    max_seq_len: usize,
) -> Option<DecodeError> {
    let (block_size, total_blocks) = {
        let pool = config.pool.lock().unwrap_or_else(|p| p.into_inner());
        (pool.block_size(), pool.total_blocks())
    };
    if block_size == 0 || decoder.layers.is_empty() {
        return None;
    }
    let blocks_per_layer = max_seq_len.div_ceil(block_size).max(1);
    let needed = blocks_per_layer.saturating_mul(decoder.layers.len());
    if needed <= total_blocks {
        return None;
    }
    // The largest per-layer reservation the whole pool could cover, and
    // therefore the longest sequence it can ever hold. Floors to zero
    // for a pool too small for even one block per layer, which is an
    // honest answer: such a pool serves nothing.
    let blocks_per_layer_limit = total_blocks / decoder.layers.len();
    let positions_limit = blocks_per_layer_limit * block_size;
    let shape = KvShape::from_config(&decoder.config, KvElem::F32);
    Some(DecodeError::KvBudgetExceeded {
        binding: Ceiling::DeviceMemory.code(),
        estimated_bytes: shape.kv_bytes_for_tokens(max_seq_len),
        limit_bytes: shape.kv_bytes_for_tokens(positions_limit),
        positions: max_seq_len,
        positions_limit,
        detail: format!(
            "request needs {needed} KV pool blocks ({max_seq_len} token positions at \
             {block_size} per block, across {} layers) but the whole pool is {total_blocks} \
             blocks; an idle server would refuse it identically",
            decoder.layers.len()
        ),
    })
}

fn acquire_pooled_caches(
    decoder: &Decoder,
    config: &KvPoolConfig,
    max_seq_len: usize,
) -> Result<Vec<KvCache>, CacheKvPoolExhausted> {
    let deadline = Instant::now() + config.queue_wait;
    loop {
        let attempt: Result<Vec<KvCache>, CacheKvPoolExhausted> = decoder
            .config
            .new_kv_caches_with_pool(&config.pool, max_seq_len);
        let now = Instant::now();
        if attempt.is_ok() || now >= deadline {
            return attempt;
        }
        std::thread::sleep(Duration::from_millis(10).min(deadline - now));
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    /// A caller-supplied stop STRING matched, and this is the one that
    /// did.
    ///
    /// Separate from [`Stop`](FinishReason::Stop), which is the model
    /// ending its own turn, because two protocols ask which it was:
    /// Anthropic reports `stop_reason: "stop_sequence"` with the text
    /// beside it, and an agent branches on that to tell "I hit the
    /// fence I put up" from "the model was done". Both still answer
    /// `"stop"` on the OpenAI surface, which has no field for the
    /// distinction.
    ///
    /// The string is the stop as the CALLER spelled it, not the text
    /// that matched it -- they are the same today and the caller's
    /// spelling is the one it can compare against.
    StopSequence(String),
    Length,
    /// A canceller asked for this generation to stop; see the `cancel`
    /// module. Deliberately not folded into `Stop`: the tokens that did
    /// arrive are a partial answer, and a client that cannot tell a
    /// completed answer from an interrupted one will show the second as
    /// the first.
    Cancelled,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            // A caller's stop string is still "stop" here: OpenAI's
            // vocabulary has no other value for it, and inventing one
            // would make a completed answer look like a failure to a
            // client that checks against the documented set. The
            // surfaces that CAN say more read the variant instead.
            FinishReason::Stop | FinishReason::StopSequence(_) => "stop",
            FinishReason::Length => "length",
            // Not an OpenAI-defined value -- OpenAI has no cancel
            // endpoint to produce one. A client that does not know the
            // string still sees a terminated stream with *a* finish
            // reason, which is what stops it being read as truncation.
            FinishReason::Cancelled => "cancelled",
        }
    }

    /// The caller-supplied stop string this generation ran into, if it
    /// ran into one.
    ///
    /// `None` covers every other ending, a stop TOKEN included: a
    /// control token in the caller's stop set is not a string the
    /// caller can compare its own `stop` list against, so naming one
    /// here would report a match the caller never asked for.
    pub fn matched_stop(&self) -> Option<&str> {
        match self {
            FinishReason::StopSequence(stop) => Some(stop),
            _ => None,
        }
    }

    /// Proof that this generation produced a WHOLE answer, or `None`
    /// if it did not.
    ///
    /// The match is exhaustive with no `_` arm on purpose: it is the
    /// one place in the server that decides what "the answer is
    /// finished" means, and a variant added to this enum later must
    /// stop this crate compiling here rather than defaulting to
    /// whichever side of the question its author never considered.
    ///
    /// Anything that stores an answer for a LATER caller needs this,
    /// because storing a partial answer republishes it as a finished
    /// one. Today that is the whole-response cache; see
    /// [`crate::response_cache::CachedCompletion::cacheable`], which is
    /// the only holder of a [`Completed`] outside this module and
    /// cannot mint one itself.
    pub fn completed(&self) -> Option<Completed> {
        match self {
            // Every one of these is the generation reaching an end the
            // request asked for: the model's own turn end, a stop
            // string the caller supplied, or the caller's own token
            // budget (`max_tokens` is part of the cache key, so
            // replaying a `Length` answer replays it under the same
            // budget that produced it).
            FinishReason::Stop | FinishReason::StopSequence(_) | FinishReason::Length => {
                Some(Completed(()))
            }
            // A canceller cut this short. The tokens that arrived are
            // the honest answer to THIS request and a truncation of
            // every other one.
            FinishReason::Cancelled => None,
        }
    }
}

/// Evidence that a generation ran to an end of its own, produced only
/// by [`FinishReason::completed`].
///
/// The unit field is private to this module, so no other module can
/// build one, clone one out of thin air, or forget to obtain one: a
/// function that requires a `Completed` is a function a partial answer
/// cannot be passed to. That is the difference between this and a
/// `bool` beside the data, which the next caller does not have to look
/// at (#57).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completed(());

/// OpenAI-convention token accounting, reported in the response's
/// `usage` field. Counted from the exact token ids the generation loop
/// processed (prompt after BOS insertion, and every generated id), not
/// re-tokenized after the fact.
///
/// Defined in `frink-api` rather than here because the shape is part
/// of the public wire contract: the UI, `frink chat` and any external
/// client read these fields, so exactly one definition may exist (see
/// that crate's module docs for why the prefill and decode phases stay
/// separate).
pub use frink_api::Usage;

#[derive(Clone)]
pub struct GenerationParams {
    /// The reasoning split this request's checkpoint implies, resolved
    /// once by whoever holds the model name
    /// (`ReasoningFormat::infer`). `None` when the checkpoint has no
    /// reasoning format at all, which is what makes
    /// `usage.completion_tokens_details` ABSENT rather than a zero the
    /// model did not earn.
    ///
    /// Whether the prompt already opened the block is not carried here:
    /// it is a property of the rendered prompt, which `generate` has,
    /// so deriving it there keeps one source instead of two.
    pub reasoning: Option<crate::policy::parser::ReasoningFormat>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub seed: u64,
    /// OpenAI's `n`: how many completions to return for ONE prompt.
    ///
    /// `1` is every request that says nothing, and is the path this
    /// server has always taken. Above 1 the prompt is prefilled ONCE
    /// and the KV cache forked per choice, which is the only reason
    /// the field exists -- a caller who wanted `k` independent
    /// generations could already send `k` requests
    /// (`docs/plans/several-completions-per-request.md`).
    ///
    /// Choice `i` samples from `seed + i`, derived rather than drawn,
    /// so a seeded request is reproducible and choice 0 of `n = 4` is
    /// byte-identical to the single answer of `n = 1`.
    pub n: usize,
    /// Whether this request will READ the per-token distribution.
    ///
    /// A flag rather than "report always", because reporting costs the
    /// greedy fast path: `Sampler::sample_reporting` has to build the
    /// full filtered distribution even for a chain that would have
    /// answered with an argmax. A request that will not read it should
    /// not pay for it.
    pub wants_logprobs: bool,
    /// How many alternatives to report per PROMPT position, or `None`
    /// for a request that did not ask.
    ///
    /// Separate from `wants_logprobs` because the costs are different
    /// and so are the refusals: scoring the prompt projects the
    /// lm_head at every prompt position rather than once, and the
    /// paged store cannot do it at all (its prefill skips what the
    /// radix tree already holds, so it has no rows for those
    /// positions).
    pub prompt_logprobs: Option<usize>,
    /// vLLM's `cache_salt`: which caller's namespace this request's
    /// prefix cache entries belong to, hashed to a `u64`.
    ///
    /// `None` is the shared namespace, which is what every request got
    /// before this field existed and what every request that names no
    /// salt still gets. A prefix cache is shared state keyed by token
    /// ids, so WITHOUT a salt one caller's prompt can be answered from
    /// another's cached prefix -- that is not a performance question
    /// but an isolation one, and naming a salt is how a caller asks
    /// for their own.
    pub cache_salt: Option<u64>,
    pub stop: Vec<String>,
    /// Stop strings that are exactly one token in this model's
    /// vocabulary, resolved once by whoever holds the tokenizer.
    ///
    /// Layer 1 of [`crate::stop`]: matched on the id, before the token
    /// is detokenized, so a control token stops the answer whatever it
    /// renders as. Empty when no stop string is a single token, or when
    /// the caller has no tokenizer to resolve them with.
    pub stop_token_ids: Vec<usize>,
    /// When true, constrain sampling toward JSON-safe token pieces and
    /// validate the emitted text is a JSON object (best-effort; see
    /// `json_mode` module).
    pub json_object: bool,
    /// A GBNF grammar every sampled token must keep parseable.
    ///
    /// The *initial* machine, shared: a request's live parse state is
    /// per-generation and lives in `sample_step::SampleState`, which
    /// clones this on its first constrained step. An `Arc` because
    /// `GenerationParams` is cloned per request and a compiled grammar
    /// is a rule table, not a flag.
    ///
    /// This is the real constraint; `json_object` above is the
    /// stateless character-class approximation that predates it. They
    /// compose (both masks run, neither can unmask), so a request may
    /// set both, and a `json_object` request whose grammar is
    /// `json.gbnf` is simply the strict version of itself.
    pub grammar: Option<std::sync::Arc<frink_models::grammar::Grammar>>,
    /// Cooperative stop flag, polled once per decoded token.
    ///
    /// It rides on the params rather than being a separate argument
    /// because every path that already threads params through -- the
    /// streaming and non-streaming chat handlers, `/v1/completions`,
    /// the Anthropic surface -- then gets cancellation without a new
    /// parameter each, and a caller that has no cancellation to offer
    /// leaves it `None`. See the `cancel` module for the two tiers that
    /// set it.
    pub cancel: Option<crate::cancel::CancelToken>,
    /// Run past the model's own end-of-generation tokens.
    ///
    /// For benchmarking, and for nothing else. A serving run needs every
    /// request to produce EXACTLY `max_tokens`, or the slowest
    /// percentile is whichever request happened to be asked for the
    /// most tokens -- a fact about the prompts, reported as a fact about
    /// the server.
    ///
    /// Suppresses the MODEL's set only. A stop string or stop token the
    /// caller supplied still ends the answer: the caller asking to
    /// ignore the model's opinion about length is not the caller
    /// withdrawing their own fence.
    pub ignore_eos: bool,
    /// llama.cpp's `reasoning_budget_tokens`: a token budget for the
    /// chain of thought, enforced in the sampler by forcing the closer
    /// once it is spent. `Unrestricted` (llama.cpp's `-1`) is no
    /// machine at all. A `Requested` number becomes an `Armed` plan at
    /// the same seam that resolves `stop_token_ids`, because both need
    /// the tokenizer -- and the sampler refuses a `Requested` budget
    /// rather than run without it. See [`crate::reasoning_budget`].
    pub reasoning_budget: crate::reasoning_budget::ReasoningBudget,
    /// A per-request override of every LoRA adapter's scale, by id --
    /// the request's `lora: [{id, scale}]` field resolved against the
    /// loaded adapters (`crate::lora::resolve_request`). `None` runs
    /// with the scales `POST /lora-adapters` (or the command line) set;
    /// `Some` makes the generation exclusive for its duration.
    pub lora: Option<Vec<f32>>,
}

impl GenerationParams {
    /// Whether a canceller has asked this generation to stop.
    ///
    /// `None` -- no cancellation wired up -- is never cancelled, so a
    /// caller that does not care pays one branch and no atomic.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.is_cancelled())
    }

    /// Whether anything about this request has to LOOK at the logits of
    /// the whole vocabulary before a token is chosen.
    ///
    /// The question exists because a decode backend is allowed to skip
    /// producing a vocabulary-shaped vector at all: the Metal dense and
    /// MoE stacks fold `final_norm + lm_head + argmax` onto the device
    /// and hand `forward_token` back a ONE-element vector holding the
    /// chosen id (`frink_models::decoder`, guarded by
    /// `FoldedLmHead::permit`). That is a large win, and it is only
    /// sound when nothing downstream needs the vocabulary.
    ///
    /// JSON-object mode needs it: [`crate::json_mode::mask_logits_for_json`]
    /// scores every vocabulary entry. Handed a one-element vector it
    /// masks one number that is not a logit, the constraint silently
    /// does nothing, and the caller gets a 200 carrying unstructured
    /// text. That was the live bug -- and `temperature: 0` with
    /// `response_format: {"type":"json_object"}` is the ORDINARY way to
    /// ask for structured output, so it was the common case rather than
    /// a corner.
    ///
    /// It is a predicate rather than a second clause on the fold's own
    /// condition so that the next thing to need the vocabulary -- a
    /// grammar, `logit_bias`, a min-p that must see the full
    /// distribution -- is one arm added here and is then true everywhere
    /// the question is asked. Both askers are named in the doc of
    /// [`greedy_gpu_fold_allowed`].
    ///
    /// The grammar arm is that next thing, and it needs the vocabulary
    /// even harder than JSON mode does: a grammar's mask is the ONLY
    /// reason its output parses, so a folded argmax under a grammar is
    /// unconstrained text served against a `response_format` the caller
    /// was told was honoured.
    ///
    /// The sampler arm is the third such thing, and it arrived with
    /// llama.cpp's `xtc` and `typ_p`. Both of those can remove the
    /// MAXIMUM from the candidate list, and `dry` can move which logit
    /// the maximum is, so at `temperature <= 0` the answer is no longer
    /// the argmax of the raw logits. A device that folded the argmax
    /// away would hand the sampler one precomputed id with no
    /// vocabulary left for XTC to remove anything from, and XTC would
    /// silently not run. `SamplingParams::greedy_equals_raw_argmax` is
    /// the single predicate deciding that, shared with
    /// `frink_cli::run`'s copy of this gate.
    ///
    /// The repetition / presence / frequency penalties are the FOURTH,
    /// and they were missing from that predicate until GitHub issue
    /// #170. They move logits over the whole vocabulary before the
    /// candidate list exists, so a device argmax over raw logits skips
    /// them entirely -- a wrong token, not a wrong distribution. A
    /// request carrying any of `repeat_penalty`, `presence_penalty` or
    /// `frequency_penalty` therefore needs the vocabulary even at
    /// `temperature: 0`.
    pub(crate) fn needs_vocab_logits(&self) -> bool {
        self.json_object
            || self.grammar.is_some()
            || self.reasoning_budget.needs_vocab_logits()
            || !self.sampling.greedy_equals_raw_argmax()
    }
}

/// Whether this request may let a backend fold `lm_head + argmax` into
/// its decode stack and return a token id instead of logits.
///
/// Greedy decoding is a necessary condition -- an argmax computed on
/// device cannot be re-sampled at temperature -- but it is NOT a
/// sufficient one, and treating it as sufficient is what broke JSON mode
/// at `temperature: 0`. The fold is sound only when the answer to
/// [`GenerationParams::needs_vocab_logits`] is also no.
///
/// A free function, taking the params, so it can be asserted in the
/// default (non-Metal) build the gates actually run: the fold itself is
/// `#[cfg(feature = "metal")]`, and a condition that only type-checks
/// under a feature flag is a condition nothing tests. Same `cfg` shape,
/// and for the same reason, as `FoldedLmHead` on the models side.
#[cfg(any(feature = "metal", test))]
pub(crate) fn greedy_gpu_fold_allowed(params: &GenerationParams) -> bool {
    params.sampling.temperature <= 0.0 && !params.needs_vocab_logits()
}

fn chunked_prefill_tokens() -> Option<usize> {
    std::env::var(
        crate::prefill_batch::PREFILL_CHUNK_ENV_KEYS[crate::prefill_batch::PRIVATE_PATH_KEY],
    )
    .ok()
    .and_then(|v| v.parse().ok())
    .filter(|&n| n > 0)
}

#[cfg(feature = "metal")]
fn cpu_kv_offload_enabled() -> bool {
    matches!(
        std::env::var("FRINK_CPU_KV_OFFLOAD").ok().as_deref(),
        Some("1")
    )
}

/// Batched prompt prefill, optionally split into `FRINK_CHUNKED_PREFILL`-sized
/// chunks that append into the same KV caches.
/// `host_kv`: whether these caches will be READ afterwards rather than
/// only decoded from. A Metal prefill otherwise leaves the real K/V on
/// the device and the host rows zero-filled, and
/// `sync_metal_attn_kv_to_host` cannot repair that -- it appends past
/// `seq_len`, which the zero fill has already advanced. The prefix
/// cache reads them, so a stored snapshot was all zeros and the next
/// request restoring it answered fluent nonsense. Paid for only when a
/// prefix cache is configured, because it costs one KV download per
/// layer and nothing else in this path reads the rows back.
pub(crate) fn forward_prompt_batch(
    decoder: &Decoder,
    tokens: &[usize],
    start_pos: usize,
    caches: &mut [KvCache],
    host_kv: bool,
) -> Vec<f32> {
    let run = |part: &[usize], pos: usize, caches: &mut [KvCache]| {
        if host_kv {
            decoder.forward_batch_last_host_kv(part, pos, caches)
        } else {
            decoder.forward_batch_last(part, pos, caches)
        }
    };
    if let Some(chunk) = chunked_prefill_tokens() {
        let mut pos = start_pos;
        let mut last = Vec::new();
        for part in tokens.chunks(chunk) {
            last = run(part, pos, caches);
            pos += part.len();
        }
        last
    } else {
        run(tokens, start_pos, caches)
    }
}

/// The server's decode engine: one forward, and speculation when the
/// store and the request allow it.
///
/// It owns the `&mut Kv` because a speculative round batches and rolls
/// back the same store a step writes to, and two borrows of it cannot
/// coexist (`sampling_loop::DecodeEngine` says why).
///
/// The drafter is `PromptLookupSpeculator`: an n-gram match over the
/// history with no second checkpoint, so it costs no memory, no load
/// time and no second tokenizer, and it is useful exactly where a
/// coding agent lives -- output that quotes its input. A model-based
/// drafter is another `Drafter` impl, not another engine.
struct ServerEngine<'a> {
    decoder: &'a Decoder,
    kv: &'a mut Kv,
    first_token_at: &'a mut Option<std::time::Instant>,
    #[cfg(feature = "metal")]
    kv_offload: bool,
    drafter: Option<frink_models::speculative::PromptLookupSpeculator>,
    accepted: usize,
    drafted: usize,
    /// FORWARD PASSES, not speculative rounds.
    ///
    /// `acceptance_length` divides the answer by this, and a round
    /// that drafted nothing still costs a forward. Counting only the
    /// speculative rounds reported 24 tokens over 3 rounds as 8.0
    /// tokens per step while the run had actually taken more forwards
    /// than that -- a number that flatters the drafter. Tokens per
    /// forward is both honest and the speedup itself: 1.0 means
    /// speculation bought nothing.
    forwards: usize,
}

impl ServerEngine<'_> {
    fn sync_after_step(&mut self) {
        #[cfg(feature = "metal")]
        if self.kv_offload {
            if let Some(caches) = self.kv.contiguous_mut() {
                self.decoder.sync_metal_attn_kv_to_host(caches);
            }
        }
    }
}

impl crate::sampling_loop::DecodeEngine for ServerEngine<'_> {
    fn step(&mut self, next: usize, pos: usize) -> Vec<f32> {
        if self.first_token_at.is_none() {
            *self.first_token_at = Some(std::time::Instant::now());
        }
        // The anchor is offered the token that is about to be fed
        // forward, at the length that includes it. A token that ended
        // the generation never reaches here, which is the `finished`
        // case `observe` refuses -- there would be no continuation to
        // rejoin at.
        if let Kv::Paged(lease) = &mut *self.kv {
            lease.observe_sampled(next, pos + 1, false);
        }
        self.forwards += 1;
        let l = self.kv.step(self.decoder, next, pos);
        self.sync_after_step();
        l
    }

    fn draft(&mut self, _prompt: &[usize], history: &[usize], max: usize) -> Vec<usize> {
        let Some(d) = self.drafter.as_ref() else {
            return Vec::new();
        };
        let mut proposal = d.propose_tokens(history);
        proposal.truncate(max);
        proposal
    }

    fn batch(&mut self, tokens: &[usize], pos: usize) -> Option<Vec<Vec<f32>>> {
        let rows = self.kv.step_batch(self.decoder, tokens, pos)?;
        self.forwards += 1;
        self.sync_after_step();
        Some(rows)
    }

    fn truncate(&mut self, pos: usize) {
        // `step_batch` refused the stores that cannot do this, so a
        // failure here would mean the two have drifted apart.
        debug_assert!(
            self.kv.truncate_to(pos),
            "a store that accepted a speculative batch must be able to roll it back"
        );
        self.sync_after_step();
    }

    fn observe(&mut self, accepted: usize, drafted: usize) {
        self.accepted += accepted;
        self.drafted += drafted;
    }
}

/// Runs the prompt through `decoder`, then generates up to
/// `params.max_tokens` new tokens, calling `emit` with each newly-safe-
/// to-flush chunk of decoded text as it becomes available (see the
/// stop-sequence buffering note below). Returns the reason generation
/// stopped.
///
/// Stop-sequence correctness: a stop string can span more than one
/// generated token (e.g. stop=" END" while the tokenizer emits " "
/// and "END" as separate pieces), so text can't simply be flushed
/// token-by-token as soon as it's decoded -- the tail end of the
/// buffer might still turn into part of a stop match once the next
/// token arrives. This holds back the last `longest_stop_len - 1`
/// bytes of decoded text (respecting UTF-8 char boundaries) until
/// they're confirmed clean, the same buffering approach real
/// inference servers use for this exact reason.
/// What a generation hands back before a wire shapes it: one
/// `(finish reason, per-token distributions)` per choice, the prompt's
/// logit rows and the ids they came from, and the usage.
///
/// A named type because clippy is right that the tuple had become
/// unreadable, and because it is the fourth thing to be added to it in
/// a day -- see `Generated`, which is this plus the text the emit
/// callback collected.
pub(crate) type GenerationOutput = (
    Vec<(FinishReason, crate::sampling_loop::PerTokenProbs)>,
    Vec<Vec<f32>>,
    Vec<usize>,
    Usage,
);

/// One completion of a request, with everything a wire may render for
/// it.
///
/// A struct rather than a tuple because this grew three times in one
/// day -- `FinishReason`, then `(FinishReason, String)`, then a third
/// member for the distributions -- and each widening was a mechanical
/// edit across every caller. A named field is added once and read by
/// the wires that want it.
/// Everything one request produced.
///
/// A struct because the tuple it replaces grew FOUR times in one day
/// -- `FinishReason`, then with the text, then with the per-choice
/// distributions, then with the prompt's. Each widening was a
/// mechanical edit across every caller, and the fourth is the one that
/// should have been the first.
///
/// `prompt_rows` sits here rather than on a choice because it belongs
/// to the REQUEST: the prompt was prefilled once and scored once,
/// whatever `n` was.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Generated {
    pub(crate) choices: Vec<GeneratedChoice>,
    /// One logit row per prompt position, EMPTY unless the request
    /// asked to score the prompt.
    pub(crate) prompt_rows: Vec<Vec<f32>>,
    /// The prompt ids those rows were computed from, INCLUDING the BOS
    /// the tokenizer prepended.
    ///
    /// Carried beside the rows rather than re-derived at the route,
    /// for the same reason `PerTokenProbs` carries its ids: a renderer
    /// that re-tokenized the prompt could line up a different sequence
    /// with these rows and report a score for a token the model never
    /// saw.
    pub(crate) prompt_ids: Vec<usize>,
    pub(crate) usage: Usage,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GeneratedChoice {
    pub(crate) finish: FinishReason,
    pub(crate) text: String,
    /// Per-token `(id, distribution it was drawn from)`, EMPTY unless
    /// the request set `GenerationParams::wants_logprobs`.
    pub(crate) logprobs: crate::sampling_loop::PerTokenProbs,
}

#[allow(clippy::too_many_arguments)]
// one clear parameter per concern; a
// bundling struct here would just be GenerationParams's fields plus
// decoder/tokenizer/stop_tokens/bos_id/prompt/kv_pool/prefix_cache/emit
// re-wrapped for no real benefit at this call depth (two call sites,
// both in this crate).
pub fn generate(
    decoder: &Decoder,
    tokenizer: &ServerTokenizer,
    stop_tokens: &StopTokens,
    bos_id: Option<usize>,
    prompt: &str,
    params: &GenerationParams,
    kv_pool: Option<&KvPoolConfig>,
    paged_kv: Option<&PagedKvConfig>,
    prefix_cache: Option<&Mutex<PrefixCache>>,
    ceiling: Option<&ContextCeiling>,
    mut emit: impl FnMut(usize, &str),
) -> Result<GenerationOutput, DecodeError> {
    let vocab_size = decoder.config.vocab_size;

    // Metal greedy GPU argmax: fold final_norm+lm_head+argmax into the
    // dense-stack CB and download one token id instead of hidden/vocab.
    // Thread-local so concurrent Arc<Decoder> requests do not race.
    //
    // `greedy_gpu_fold_allowed` and not `temperature <= 0.0`: the fold
    // returns a token id where a caller-supplied logit constraint
    // expects a vocabulary, and the constraint would then apply to
    // nothing at all. See `GenerationParams::needs_vocab_logits`.
    #[cfg(feature = "metal")]
    let _metal_greedy_guard = {
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                frink_models::set_metal_greedy_argmax(false);
            }
        }
        if greedy_gpu_fold_allowed(params) {
            frink_models::set_metal_greedy_argmax(true);
            Some(Guard)
        } else {
            None
        }
    };

    // `Parse`: the prompt is what a chat template rendered, or a raw
    // completion, and llama.cpp's server tokenizes both with
    // `parse_special = true`. See `Model::encode`.
    let mut tokens = tokenizer.encode(prompt, SpecialTokens::Parse);
    prepend_bos(&mut tokens, bos_id);
    let prompt_tokens = tokens.len();
    if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab_size) {
        return Err(DecodeError::TokenOutOfVocab {
            token: bad,
            vocab_size,
        });
    }

    // With a shared pool configured, admission control happens here:
    // each layer's cache reserves enough blocks up front for this
    // request's real worst case (prompt length + max_tokens), not just
    // one block -- see `acquire_pooled_caches`'s doc comment for why
    // under-reserving here would let a request panic mid-decode
    // instead of failing cleanly at admission time. If any layer can't
    // get that many blocks, `acquire_pooled_caches` retries (bounded by
    // `config.queue_wait`) before the request is rejected. Each failed
    // attempt's partial `Vec` is dropped immediately (via `collect`),
    // releasing any blocks it did acquire through `KvCache`'s `Drop`
    // impl before the next retry, so a request that ultimately gives up
    // leaves the pool exactly as it found it.
    //
    // Prefix-cache restoration only applies when there's no shared KV
    // block pool: a restored cache is a plain, unpooled clone (see
    // `KvCache`'s `Clone` doc comment), so combining it with pool-based
    // admission control would let a request's real memory usage
    // silently bypass the pool's bounded-memory guarantee. Not
    // supported together yet.
    // The per-request context ceiling, checked before any KV is
    // acquired and before a single forward pass runs. This path used to
    // have no context ceiling at all: an oversized request either ran
    // until something else killed it, or -- with a pool configured --
    // waited out `queue_wait` and left with a 503 "retry shortly" about
    // a request an idle server would refuse identically. The ceiling is
    // the same `ContextCeiling` the batch scheduler admits on (see
    // `crate::budget`), so the two paths cannot disagree.
    //
    // It has TWO outcomes, not one. A prompt at or past the ceiling has
    // no output budget left to give it and is refused. A prompt that
    // fits is SERVED, with `max_tokens` clamped down to what remains --
    // refusing that case instead turns a servable 100k-prompt request
    // into a 400 over a `max_tokens` the caller very likely never set.
    // The clamp is what makes a large default output budget safe.
    let clamped;
    let params = match ceiling {
        // `ContextCeiling::fit` IS the two outcomes above, and it is
        // shared with the continuous batcher's admission so the two
        // decode paths cannot answer the same request differently.
        Some(ceiling) => match ceiling.fit(prompt_tokens, params.max_tokens)? {
            fitted if fitted == params.max_tokens => params,
            fitted => {
                let mut p = params.clone();
                tracing::debug!(
                    "max_tokens clamped from {} to {fitted} by the context ceiling",
                    p.max_tokens
                );
                p.max_tokens = fitted;
                clamped = p;
                &clamped
            }
        },
        None => params,
    };
    // `params` is the clamped copy by now, so this cannot exceed the
    // ceiling when one exists. `checked_add` covers the case where none
    // does: an unbounded deployment must still not wrap into a small
    // number and pass every check below it.
    let Some(max_seq_len) = prompt_tokens.checked_add(params.max_tokens) else {
        return Err(DecodeError::KvBudgetExceeded {
            binding: frink_models::Ceiling::ContextLength.code(),
            estimated_bytes: 0,
            limit_bytes: 0,
            positions: usize::MAX,
            positions_limit: usize::MAX,
            detail: format!(
                "prompt of {prompt_tokens} tokens plus max_tokens of {} overflows the position \
                 counter, so this request cannot be served by any deployment",
                params.max_tokens
            ),
        });
    };

    // A request whose worst case exceeds the *whole* pool is not a
    // request to retry: no amount of waiting frees blocks that do not
    // exist. Separated from `KvPoolExhausted` (503) for exactly the
    // reason the batch scheduler separates `kv_rejected_too_large` from
    // `queue_rejected` -- one says "come back later", the other says
    // "this will never work", and an operator sent to the wrong one
    // tunes the wrong knob.
    if let Some(config) = kv_pool {
        if let Some(err) = pool_immovable_refusal(decoder, config, max_seq_len) {
            return Err(err);
        }
    }

    let restored = if kv_pool.is_none() {
        prefix_cache.and_then(|pc| {
            let m = pc
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .find_longest_prefix_salted(&tokens, params.cache_salt);
            (m.matched_len > 0).then_some(m)
        })
    } else {
        None
    };

    // Prompt tokens this request will *not* have to recompute. Reported
    // as `usage.cached_tokens` -- `Some(0)` when a prefix cache exists
    // and missed, `None` when there is no prefix cache to consult, so a
    // client can tell "no hit" from "no cache".
    let cached_tokens = restored
        .as_ref()
        .map(|m| m.matched_len)
        .or_else(|| (prefix_cache.is_some() && kv_pool.is_none()).then_some(0));

    // One logit row per prompt position, only when the request asked
    // to score the prompt.
    let mut prompt_rows: Vec<Vec<f32>> = Vec::new();
    // Kept because the tail consumes `tokens`.
    let mut tokens_for_scoring: Vec<usize> = Vec::new();
    let prefill_start = std::time::Instant::now();
    let mut pos;
    let mut logits: Vec<f32>;
    let mut kv: Kv;
    if let Some(m) = restored {
        // Only the contiguous path reaches here: a prefix cache and a
        // paged store are refused together at startup, because
        // `PrefixCache` stores `Vec<KvCache>` snapshots a paged request
        // cannot produce.
        kv = Kv::Contiguous(
            m.kv_caches
                .expect("matched_len > 0 always carries kv_caches"),
        );
        let caches = &mut *kv
            .contiguous_mut()
            .expect("a restored prefix is contiguous by construction");
        let suffix = &tokens[m.matched_len..];
        if suffix.is_empty() {
            if let Some(pl) = m.pending_logits {
                // The whole query was already processed and stored
                // verbatim before (e.g. an exact-repeat prompt with
                // unseeded sampling, so the whole-response cache
                // couldn't serve it) -- zero forward passes needed.
                pos = m.matched_len;
                logits = pl;
            } else {
                // Rare: our query exactly matches a strict PREFIX of a
                // longer stored entry (someone else's conversation
                // continued past this point), so there's no stored
                // "what comes next" for our shorter query. Back the
                // restored cache off by one position and reprocess
                // just that last matched token to get real logits,
                // rather than guessing.
                let back_to = m.matched_len - 1;
                for c in caches.iter_mut() {
                    c.truncate(back_to);
                }
                pos = back_to;
                logits = decoder.forward_token(tokens[back_to], pos, caches);
                pos += 1;
            }
        } else {
            pos = m.matched_len;
            let mut l = Vec::new();
            for &tok in suffix {
                l = decoder.forward_token(tok, pos, caches);
                pos += 1;
            }
            logits = l;
        }
    } else {
        kv = match (paged_kv, kv_pool) {
            (Some(config), _) => Kv::Paged(
                acquire_paged_caches(decoder, config, &tokens, max_seq_len)
                    .map_err(|_| DecodeError::KvPoolExhausted)?,
            ),
            (None, Some(config)) => Kv::Contiguous(
                acquire_pooled_caches(decoder, config, max_seq_len)
                    .map_err(|_| DecodeError::KvPoolExhausted)?,
            ),
            (None, None) => Kv::Contiguous(decoder.config.new_kv_caches()),
        };
        // Process the prompt once, capturing the *last* call's logits
        // (which already predict the first generated token) instead of
        // discarding them. A prompt of zero tokens (bos_id unset and an
        // empty-string prompt encoding to nothing) has no real token to
        // seed from, so a synthetic id 0 bootstraps decoding.
        pos = 0;
        logits = if tokens.is_empty() {
            let l = kv.step(decoder, 0, pos);
            pos += 1;
            l
        } else {
            // Batched prefill: one pass over the prompt (shared weight
            // traffic on CPU; fewer per-token bookkeeping costs). Last
            // row's logits predict the first generated token — same as
            // the sequential loop this replaces (see unit test below).
            // When `FRINK_CHUNKED_PREFILL` is set, split long prompts
            // into chunks that reuse the same KV caches.
            pos = tokens.len();
            // Scoring the prompt needs a logit row per POSITION, which
            // is a separate entry point because it projects the
            // lm_head at every one instead of once
            // (`Kv::prefill_scored`). A request that did not ask keeps
            // the cheap path exactly as it was.
            if params.cache_salt.is_some() && matches!(kv, Kv::Paged(_)) {
                // The radix tree walks ONE tree keyed by token ids and
                // its nodes hold block indices the whole deployment
                // shares; there is no namespace to scope a lookup to.
                // Serving this would be the worst of the three
                // options: a caller who asked for isolation, told they
                // got it, and sharing anyway.
                return Err(DecodeError::Unsupported(
                    "`cache_salt` is not implemented for the paged KV store: its prefix tree has \
                     no per-caller namespace, so the isolation the field asks for would not \
                     hold. Serve without `--paged-kv`."
                        .to_string(),
                ));
            }
            if params.prompt_logprobs.is_some() {
                let Some(rows) = kv.prefill_scored(decoder, &tokens) else {
                    // The paged store's prefill SKIPS whatever the
                    // radix tree already holds, so it has no rows for
                    // those positions and scoring them would be an
                    // invention. Refused by name rather than reported
                    // with holes.
                    return Err(DecodeError::Unsupported(
                        "`prompt_logprobs` needs a logit row for every prompt position, and a \
                         paged request's prefill skips the positions the prefix tree already \
                         holds. Serve it without `--paged-kv`."
                            .to_string(),
                    ));
                };
                let last = rows.last().cloned().unwrap_or_default();
                tokens_for_scoring = tokens.clone();
                prompt_rows = rows;
                last
            } else {
                // The prefix cache is the only thing here that reads
                // these caches back; without one, the download is pure
                // cost.
                kv.prefill(decoder, &tokens, prefix_cache.is_some())
            }
        };
    }

    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    let decode_start = std::time::Instant::now();
    #[cfg(feature = "metal")]
    let kv_offload = cpu_kv_offload_enabled();
    let decode_token = |id: usize| tokenizer.decode(&[id]);

    // `logits` becomes the prediction for the position after each
    // generated token; every generated token gets exactly one
    // corresponding cache entry via the `step` closure below, matching
    // `prefix_cache`'s `pending_logits` expectations regardless of
    // whether the loop goes on to hit a stop sequence or max_tokens.
    // Time-to-first-token is stamped inside the `step` closure rather
    // than approximated as "prefill time": `step` is called immediately
    // after the first token is sampled, so this is the real moment the
    // user could have seen something. Threading it through the closure
    // instead of `sample_until_stop`'s signature keeps that function's
    // (already long) argument list unchanged, and the closure's mutable
    // borrow ends when it is dropped at the call's return.
    let mut first_token_at: Option<std::time::Instant> = None;
    // Prompt-lookup drafting, when the request and the store allow it.
    //
    // No second checkpoint, so it costs no memory and no load time;
    // `Kv::step_batch` refuses the stores that cannot roll a rejected
    // draft back, which leaves this decision about the REQUEST alone.
    //
    // A grammar is refused here rather than inside the loop: the
    // verification draws through the same grammar machine, and a
    // rejected block would leave it advanced over tokens that were
    // never emitted. That is the one sampler feature the verify-by-
    // agreement rule does not get for free, and refusing is honest
    // where a silent fallback would not be.
    let draft_max = if params.grammar.is_some() || params.json_object {
        0
    } else {
        crate::sampling_loop::DEFAULT_DRAFT_MAX
    };
    // `n` completions of ONE prompt, which is the whole reason the
    // field exists: the prefill above ran once and the forks below
    // start from it (`docs/plans/several-completions-per-request.md`).
    let n = params.n.max(1);
    // The forks are taken from the POST-PREFILL state, before choice 0
    // decodes into `kv` and mutates it. Cloning after would give
    // choices 1.. a cache that already holds choice 0's tokens, which
    // is a different prompt.
    let mut forks: Vec<Kv> = Vec::new();
    if n > 1 {
        let Some(caches) = kv.contiguous_mut() else {
            // Refused rather than re-prefilled per choice. A caller who
            // asked for four and silently got four prefills paid four
            // times for the thing the field exists to avoid, with
            // nothing in the response saying so.
            return Err(DecodeError::Unsupported(
                "`n` > 1 needs a forkable KV cache; this request is on the paged store, whose \
                 block lists have no copy-on-write yet. Serve it without `--paged-kv`, or send \
                 the request n times."
                    .to_string(),
            ));
        };
        forks = (1..n).map(|_| Kv::Contiguous(caches.clone())).collect();
    }

    let mut finishes: Vec<FinishReason> = Vec::with_capacity(n);
    // One entry per choice, each the per-token distributions that
    // choice was drawn from. EMPTY vectors when the request did not
    // ask for logprobs, which is every request until a route wires
    // `wants_logprobs`.
    let mut choice_logprobs: Vec<crate::sampling_loop::PerTokenProbs> = Vec::with_capacity(n);
    let mut completion_tokens = 0usize;
    let mut speculation = (0usize, 0usize, 0usize);
    // Choice 0's ids and final logits are what the request tail
    // publishes: the prefix cache stores ONE continuation, and choice 0
    // is the one a later `n = 1` with the same seed reproduces.
    let mut first_generated_ids: Vec<usize> = Vec::new();
    let mut first_logits: Vec<f32> = Vec::new();

    for choice in 0..n {
        // Derived, not drawn: `n: 4, seed: 7` is stable across runs and
        // across `n`, so choice 0 of four is byte-identical to the
        // single answer of one.
        let mut choice_params = params.clone();
        choice_params.seed = params.seed.wrapping_add(choice as u64);
        let choice_params = if choice == 0 { params } else { &choice_params };

        let choice_kv: &mut Kv = if choice == 0 {
            &mut kv
        } else {
            &mut forks[choice - 1]
        };
        let mut engine = ServerEngine {
            decoder,
            kv: choice_kv,
            first_token_at: &mut first_token_at,
            #[cfg(feature = "metal")]
            kv_offload,
            drafter: (draft_max > 0).then(|| {
                frink_models::speculative::PromptLookupSpeculator::new(
                    crate::sampling_loop::DRAFT_NGRAM,
                    draft_max,
                )
            }),
            accepted: 0,
            drafted: 0,
            forwards: 0,
        };
        let (finish, generated_ids, final_logits, choice_probs) =
            crate::sampling_loop::sample_until_stop(
                logits.clone(),
                pos,
                &tokens,
                stop_tokens,
                choice_params,
                |ids| tokenizer.decode_bytes(ids),
                &mut engine,
                &mut |text: &str| emit(choice, text),
                &decode_token,
                draft_max,
            )?;
        // Read the counters out BEFORE the borrow ends, because this is
        // the engine's last use and it borrows `kv` and
        // `first_token_at` mutably.
        speculation.0 += engine.forwards;
        speculation.1 += engine.accepted;
        speculation.2 += engine.drafted;
        completion_tokens += generated_ids.len();
        finishes.push(finish);
        choice_logprobs.push(choice_probs);
        if choice == 0 {
            first_generated_ids = generated_ids;
            first_logits = final_logits;
        }
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();
    let generated_ids = first_generated_ids;
    logits = first_logits;
    // Everything after the last token -- the usage block and the three
    // places this request's KV may be published -- is
    // `crate::request_tail`. It is lifted out unchanged, and the seam
    // is not arbitrary: it is exactly the part that runs ONCE per
    // request rather than once per completion, which is what
    // `docs/plans/several-completions-per-request.md` needs next.
    let usage = crate::request_tail::RequestTail {
        decoder,
        tokenizer,
        params,
        prompt,
        tokens,
        generated_ids,
        completion_tokens,
        logits,
        kv,
        prompt_tokens,
        vocab_size,
        prefill_secs,
        decode_secs,
        prefill_start,
        first_token_at,
        cached_tokens,
        speculation,
        kv_pool_configured: kv_pool.is_some(),
        radix_enabled: paged_kv.is_some_and(|c| c.radix.is_some()),
        prefix_cache,
        cache_salt: params.cache_salt,
    }
    .finish();

    let scored_ids = if prompt_rows.is_empty() {
        Vec::new()
    } else {
        tokens_for_scoring
    };
    Ok((
        finishes.into_iter().zip(choice_logprobs).collect(),
        prompt_rows,
        scored_ids,
        usage,
    ))
}

/// Shared sampling + stop-sequence-aware emission loop, given already-
/// primed `logits`/`pos` (the prompt has already been processed by the
/// caller) and a `step` closure that advances one position and returns
/// the new logits. Used by both `generate` (the GGUF path, whose own
/// KV-pool/prefix-cache-aware prompt priming stays separate above) and
/// `generate_engine` (any other `Engine`, with simpler priming and no
/// pooling/restoration) so the actual sampling/stop-sequence
/// correctness -- the part most worth not duplicating -- lives in
/// exactly one place. Returns the finish reason, the generated token
/// ids, and the final logits (the prediction for whatever would come
/// next), since `generate`'s prefix-cache storage needs both.
///
/// The generic counterpart to `generate`, for any `Engine` other than
/// `Decoder` (in practice, Kimi K3's `KimiEngine`). Deliberately
/// simpler: no KV block pool, no `PrefixCache` restoration -- Kimi's
/// KDA state is a fixed-size recurrent matrix that collapses history
/// irreversibly, so it can't support the truncate/restore operations
/// those features need (see `frink_models::engine`'s module docs). Every
/// request processes its full prompt from scratch against fresh
/// engine state.
pub fn generate_engine<E: Engine, T: TextTokenizer>(
    engine: &E,
    tokenizer: &T,
    stop_tokens: &StopTokens,
    bos_id: Option<usize>,
    prompt: &str,
    params: &GenerationParams,
    mut emit: impl FnMut(&str),
    // A `Vec` for the same reason `generate` returns one: two shapes
    // for "what this request produced" is one more chance for a caller
    // to handle the simple case and forget the other. These engines
    // hold a recurrent state that collapses history irreversibly, so
    // they cannot fork a prefix and always answer with exactly one.
) -> Result<GenerationOutput, DecodeError> {
    // These engines cannot fork: a KDA / MLA recurrent state is a
    // reduction over the whole prefix, so there is no cache to clone
    // into a second choice. Refused by name rather than served as one.
    if params.n > 1 {
        return Err(DecodeError::Unsupported(
            "`n` > 1 needs a forkable KV cache, and this checkpoint runs on an engine whose \
             state collapses history irreversibly. Send the request n times."
                .to_string(),
        ));
    }

    let vocab_size = engine.vocab_size();
    // `Parse`, as the GGUF path above. See `Model::encode`.
    let mut tokens = tokenizer.encode(prompt, SpecialTokens::Parse);
    prepend_bos(&mut tokens, bos_id);
    let prompt_tokens = tokens.len();
    if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab_size) {
        return Err(DecodeError::TokenOutOfVocab {
            token: bad,
            vocab_size,
        });
    }

    let mut state = engine.new_state();
    let mut pos = 0;
    let prefill_start = std::time::Instant::now();
    let logits = if tokens.is_empty() {
        let l = engine.forward_token(0, pos, &mut state);
        pos += 1;
        l
    } else {
        let mut l = Vec::new();
        for &tok in tokens.iter() {
            l = engine.forward_token(tok, pos, &mut state);
            pos += 1;
        }
        l
    };
    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    let decode_start = std::time::Instant::now();

    // Timed for the same reason the `Decoder` path is: a UI that has to
    // wall-clock these engines instead cannot separate prefill from
    // decode, and Kimi/MLA prefill is sequential (one forward per prompt
    // token), so the two phases differ by more here, not less.
    let mut first_token_at: Option<std::time::Instant> = None;
    let (finish, generated_ids, _final_logits, _probs) = crate::sampling_loop::sample_until_stop(
        logits,
        pos,
        &tokens,
        stop_tokens,
        params,
        |ids| tokenizer.decode_bytes(ids),
        &mut |next: usize, pos: usize| {
            if first_token_at.is_none() {
                first_token_at = Some(std::time::Instant::now());
            }
            engine.forward_token(next, pos, &mut state)
        },
        &mut emit,
        &|id: usize| tokenizer.decode(&[id]),
        0,
    )?;
    let decode_secs = decode_start.elapsed().as_secs_f64();

    let mut usage =
        Usage::new(prompt_tokens, generated_ids.len()).with_timings(prefill_secs, decode_secs);
    if let Some(at) = first_token_at {
        usage = usage.with_ttft(at.duration_since(prefill_start).as_secs_f64());
    }
    // No distributions: these engines refuse `n` > 1 above and their
    // logprobs row is not wired either.
    Ok((vec![(finish, Vec::new())], Vec::new(), Vec::new(), usage))
}

#[cfg(test)]
mod tests {

    use super::*;
    use frink_models::config::test_dense_fixture;

    /// Which endings may be replayed to a LATER caller, enumerated so
    /// the answer is on the record rather than inferred from whichever
    /// handler happens to hold a cancel token (#57).
    ///
    /// `Length` is on the completed side deliberately: `max_tokens` is
    /// part of the response cache's key, so an answer truncated at the
    /// budget is only ever replayed under the same budget that produced
    /// it. `Cancelled` is the whole of the other side -- it kept the
    /// tokens that had arrived, which answers the cancelled request and
    /// truncates every other one.
    #[test]
    fn only_a_generation_that_reached_an_end_of_its_own_counts_as_completed() {
        for reason in [
            FinishReason::Stop,
            FinishReason::StopSequence("<END>".to_string()),
            FinishReason::Length,
        ] {
            assert!(
                reason.completed().is_some(),
                "{reason:?} produced the whole answer the request asked for"
            );
        }
        assert!(
            FinishReason::Cancelled.completed().is_none(),
            "a cancelled generation is a partial answer and may not be stored \
             for anybody else"
        );
    }

    fn small_decoder() -> Decoder {
        Decoder::new_random_small(test_dense_fixture(), 2, 256)
    }

    /// The server really speculates, and says so in `usage`.
    ///
    /// What this test does NOT do is compare the speculative answer
    /// against an unspeculated one, and the reason is worth stating:
    /// there is no way to turn drafting off for one request without
    /// changing something else about it. The first version used
    /// `json_object` as the control and was wrong -- that turns on a
    /// grammar, so it compared two different requests and the text
    /// differed for that reason rather than because speculation had
    /// changed anything.
    ///
    /// The equivalence claim is proven where it can be proven cleanly:
    /// `sampling_loop::speculation_changes_the_forward_count_and_not_
    /// the_answer` runs the same scripted engine with three drafters
    /// and demands identical ids and text, and the KV seam's own test
    /// pins a batched forward against a per-token one at 1e-6. This
    /// test covers what those cannot: that the wiring reaches a real
    /// request at all.
    /// The derived seeds do two things, and both need pinning: the
    /// other choices must DIFFER from choice 0 (otherwise `n` returns
    /// the same answer `k` times and buys nothing), and the whole set
    /// must be REPRODUCIBLE (otherwise a seeded request is not seeded).
    ///
    /// This is the test that catches a change to `seed + i`.
    #[test]
    fn the_derived_seeds_make_the_other_choices_differ_reproducibly() {
        let decoder = small_decoder();
        let prompt = "abcabcabcabcabcabc";
        let mut params = greedy_params(8);
        params.sampling.temperature = 1.0;
        params.n = 4;

        let run = || {
            let mut per_choice = vec![String::new(); 4];
            generate(
                &decoder,
                &ServerTokenizer::Byte,
                &StopTokens::default(),
                None,
                prompt,
                &params,
                None,
                None,
                None,
                None,
                |choice, s| per_choice[choice].push_str(s),
            )
            .expect("n = 4");
            per_choice
        };

        let first = run();
        let second = run();
        assert_eq!(
            first, second,
            "the same seeded request gave two different sets of choices"
        );
        assert!(
            first[1..].iter().any(|t| *t != first[0]),
            "every choice matched choice 0, so the derived seeds did nothing: {first:?}"
        );
    }

    /// **`n` > 1 prefills once and forks.** The acceptance number is
    /// not wall clock, it is `prompt_tokens`: four choices of one
    /// prompt must be billed for ONE prompt, because that is the only
    /// thing the field buys. If this reads four times the `n = 1`
    /// figure, the fork did not happen and the feature is a loop
    /// (`docs/plans/several-completions-per-request.md`).
    #[test]
    fn several_choices_prefill_the_prompt_once() {
        let decoder = small_decoder();
        let prompt = "abcabcabcabcabcabc";

        let mut one_text = String::new();
        let (one_finish, _rows, _ids, one_usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            prompt,
            &greedy_params(6),
            None,
            None,
            None,
            None,
            |_, s| one_text.push_str(s),
        )
        .expect("n = 1");
        assert_eq!(one_finish.len(), 1);

        let mut four = greedy_params(6);
        four.n = 4;
        let mut per_choice = vec![String::new(); 4];
        let (finishes, _rows, _ids, usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            prompt,
            &four,
            None,
            None,
            None,
            None,
            |choice, s| per_choice[choice].push_str(s),
        )
        .expect("n = 4");

        assert_eq!(finishes.len(), 4, "one finish reason per choice");
        // THE acceptance test: the prompt was prefilled once, so it is
        // billed once.
        assert_eq!(
            usage.prompt_tokens, one_usage.prompt_tokens,
            "n = 4 billed the prompt more than once, so the KV was not forked"
        );
        // And the completion side is the sum, not one choice's.
        assert_eq!(usage.completion_tokens, one_usage.completion_tokens * 4);
        for (i, text) in per_choice.iter().enumerate() {
            assert!(!text.is_empty(), "choice {i} produced nothing");
        }
    }

    /// Choice 0 is byte-identical to the single answer at the same
    /// seed. That is what makes the derived seeds reproducible rather
    /// than merely different: a caller can raise `n` without the answer
    /// they already had changing underneath them.
    ///
    /// **Sampled, not greedy, on purpose.** At temperature 0 the seed
    /// cannot change anything.
    ///
    /// This pins choice 0 ONLY, and deliberately: choice 0 runs on the
    /// request's own `params`, so no change to the derivation can move
    /// it -- confirmed by sabotage, `seed + i` shifted to `seed + i + 1`
    /// leaves this green. The derivation itself is pinned by
    /// [`the_derived_seeds_make_the_other_choices_differ_reproducibly`],
    /// which is the test that goes red for that mutation.
    #[test]
    fn choice_zero_is_the_answer_n_equals_one_would_have_given() {
        let decoder = small_decoder();
        let prompt = "abcabcabcabcabcabc";
        let sampled = |max_tokens: usize| {
            let mut p = greedy_params(max_tokens);
            p.sampling.temperature = 1.0;
            p
        };

        let mut one = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            prompt,
            &sampled(6),
            None,
            None,
            None,
            None,
            |_, s| one.push_str(s),
        )
        .expect("n = 1");

        let mut three = sampled(6);
        three.n = 3;
        let mut per_choice = vec![String::new(); 3];
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            prompt,
            &three,
            None,
            None,
            None,
            None,
            |choice, s| per_choice[choice].push_str(s),
        )
        .expect("n = 3");

        assert_eq!(
            per_choice[0], one,
            "choice 0 must be the answer n = 1 gives at the same seed"
        );
    }

    #[test]
    fn the_server_speculates_and_reports_it() {
        let decoder = small_decoder();
        let params = greedy_params(24);

        // Repetition is what a prompt-lookup drafter matches on, and
        // the byte tokenizer makes the text the token ids.
        let mut text = String::new();
        let (finish, _rows, _ids, usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            "abcabcabcabcabcabcabcabcabcabcabcabc",
            &params,
            None,
            None,
            None,
            None,
            |_, s| text.push_str(s),
        )
        .expect("speculative");

        assert_eq!(finish[0].0, FinishReason::Length);
        assert_eq!(usage.completion_tokens, 24);
        assert!(
            usage.draft_tokens.is_some_and(|d| d > 0),
            "a repetitive prompt drafted nothing, so this test proves nothing"
        );
        assert!(
            usage.accepted_draft_tokens.is_some_and(|a| a > 0),
            "nothing was accepted, so no forward was saved"
        );
        // Tokens per forward: 1.0 is "speculation bought nothing".
        let speedup = usage.acceptance_length.expect("acceptance length");
        assert!(
            speedup > 1.0,
            "speculation saved no forwards (tokens per forward {speedup})"
        );

        // A request with no room to draft reports NOTHING rather than
        // zeros: an absent field reads as "this request did not
        // speculate", where a zero reads as "the drafter was useless".
        //
        // One token is the case where the budget guard forbids
        // drafting outright -- a block always commits its accepted
        // drafts PLUS one, so a round needs at least two tokens of
        // room. That guard is also what keeps `max_tokens` honest, and
        // it is the bug this row's loop test caught.
        //
        // Note the prompt is repetitive here too: the drafter would
        // happily propose from it, and does not get the chance.
        let single = greedy_params(1);
        let mut one_text = String::new();
        let (_, _rows, _ids, one_usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            "abcabcabcabcabcabcabcabcabcabcabcabc",
            &single,
            None,
            None,
            None,
            None,
            |_, s| one_text.push_str(s),
        )
        .expect("single token");
        assert_eq!(one_usage.completion_tokens, 1);
        assert!(
            one_usage.draft_tokens.is_none(),
            "a request with no room to draft must not report speculation"
        );
    }

    /// A batch of `k` tokens leaves the cache exactly where feeding
    /// them one at a time would, and hands back one row per token.
    /// Both halves matter: the rows are what verification reads, and
    /// the cache state is what the next step continues from.
    #[test]
    fn step_batch_matches_stepping_one_token_at_a_time() {
        let decoder = small_decoder();
        let tokens = [1usize, 5, 9, 2];

        let mut batched = Kv::Contiguous(decoder.config.new_kv_caches());
        let rows = batched
            .step_batch(&decoder, &tokens, 0)
            .expect("a contiguous store speculates");
        assert_eq!(rows.len(), tokens.len(), "one row per token");

        let mut stepped = Kv::Contiguous(decoder.config.new_kv_caches());
        let one_at_a_time: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| stepped.step(&decoder, t, pos))
            .collect();

        for (pos, (b, o)) in rows.iter().zip(one_at_a_time.iter()).enumerate() {
            assert_eq!(b.len(), o.len(), "row {pos} width");
            for (i, (x, y)) in b.iter().zip(o.iter()).enumerate() {
                // The measured batched-vs-sequential ulp, the same
                // bound `frink-models` pins that difference at.
                assert!(
                    (x - y).abs() < 1e-6,
                    "row {pos} logit {i}: batched={x} sequential={y}"
                );
            }
        }
    }

    /// Rejecting a draft means the rows it wrote are gone, and the
    /// store continues as if they never happened. Checked by walking
    /// the same tokens again after the rollback and demanding the same
    /// logits: a stale row left behind would change them.
    #[test]
    fn truncate_to_undoes_the_rows_a_rejected_draft_wrote() {
        let decoder = small_decoder();
        let committed = [1usize, 5];
        let draft = [9usize, 2];

        let mut kv = Kv::Contiguous(decoder.config.new_kv_caches());
        kv.step_batch(&decoder, &committed, 0).expect("committed");
        let after_commit = kv
            .step_batch(&decoder, &draft, committed.len())
            .expect("draft");

        // Reject the whole draft.
        assert!(
            kv.truncate_to(committed.len()),
            "a contiguous store rolls back"
        );

        // Feeding the same draft again must land on the same rows.
        let again = kv
            .step_batch(&decoder, &draft, committed.len())
            .expect("draft again");
        for (pos, (a, b)) in after_commit.iter().zip(again.iter()).enumerate() {
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 1e-6,
                    "row {pos} logit {i} after rollback: {x} vs {y}"
                );
            }
        }
    }

    /// The gate is asked BEFORE anything is written. A store that
    /// cannot roll back must not be speculated into at all, which is
    /// why `step_batch` returns `None` rather than `truncate_to`
    /// failing after the damage is done.
    #[test]
    fn a_paged_store_refuses_to_speculate_rather_than_failing_to_undo() {
        // Constructed without a pool: the point is the arm, not the
        // pages, and `step_batch` decides on the variant alone.
        let decoder = small_decoder();
        let mut kv = Kv::Contiguous(decoder.config.new_kv_caches());
        assert!(
            kv.step_batch(&decoder, &[1], 0).is_some(),
            "the contiguous arm serves speculation"
        );
        assert!(kv.truncate_to(0), "and can undo it");
    }

    fn greedy_params(max_tokens: usize) -> GenerationParams {
        GenerationParams {
            cache_salt: None,
            prompt_logprobs: None,
            wants_logprobs: false,
            reasoning: None,
            max_tokens,
            sampling: SamplingParams::default(),
            seed: 1,
            n: 1,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            json_object: false,
            grammar: None,
            cancel: None,
            ignore_eos: false,
            reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
            lora: None,
        }
    }

    /// Regression test for a real bug caught by close reading, not by
    /// any earlier test (the earlier tests only checked `generate`
    /// against a test helper that replicated the same buggy pattern,
    /// so they never could have caught it): `generate`'s original
    /// prompt-processing loop pushed every prompt token into the KV
    /// cache once (correct), then its first generation-loop iteration
    /// re-processed the *last* prompt token a second time via another
    /// `forward_token` call at the wrong position (`tokens.len()`
    /// instead of its real position `tokens.len() - 1`) just to obtain
    /// logits -- silently duplicating that token in the cache with a
    /// different RoPE rotation applied, corrupting every subsequent
    /// position's attention. Fixed by capturing the prompt loop's own
    /// last-iteration logits instead of discarding and re-deriving
    /// them. This locks in the fixed pattern (which `generate` uses
    /// internally) against `forward_batch`'s independent ground truth.
    #[test]
    fn prompt_processing_matches_forward_batch_ground_truth_with_no_duplicate_position() {
        let decoder = small_decoder();
        let tokens = vec![1usize, 2, 3, 4];

        let mut fresh_caches: Vec<KvCache> = decoder.config.new_kv_caches();
        let batch_logits = decoder.forward_batch(&tokens, 0, &mut fresh_caches);
        let ground_truth_next_logits = batch_logits.last().unwrap().clone();

        // The exact pattern `generate` now uses: one forward_token call
        // per prompt token, keeping the last call's logits.
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
        let mut logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            logits = decoder.forward_token(tok, pos, &mut caches);
        }

        assert_eq!(
            caches[0].positions(),
            fresh_caches[0].positions(),
            "must not push any position beyond the real prompt length"
        );
        // Tolerance, not bit equality. `forward_token` goes through
        // `WeightMatrix::apply` (one activation) and `forward_batch`
        // through `apply_batch`, and since the CPU batch GEMM kernels
        // landed those are different kernels with different f32
        // accumulation orders -- on aarch64 the batched path uses the
        // i8mm interleave-8 GEMM. So they agree to last-ulp, not
        // bit-for-bit. llama.cpp has the same batch-vs-sequential
        // property. The engine-side twin of this test was relaxed for
        // the same reason; this one only shows up on a host where the
        // aarch64 kernels actually run.
        assert_eq!(logits.len(), ground_truth_next_logits.len());
        for (i, (a, b)) in logits.iter().zip(&ground_truth_next_logits).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * a.abs().max(1.0),
                "logit {i} predicting the first generated token: sequential {a} vs forward_batch {b}"
            );
        }
    }

    /// End-to-end version of the same property: `generate`'s full
    /// greedy decode loop (prompt processing + iterative generation)
    /// must produce exactly the token sequence an independent
    /// step-by-step computation (via `forward_batch` for the prompt,
    /// then `forward_token` once per new position, argmax at each
    /// step) would produce. Restricted to ASCII byte values so
    /// `ServerTokenizer::Byte`'s `decode` is lossless in both
    /// directions and the generated text can be compared back to
    /// token ids exactly.
    #[test]
    fn generate_greedy_output_matches_independent_step_by_step_computation() {
        let decoder = small_decoder();
        let prompt_ids = vec![1usize, 2, 3];
        let prompt = String::from_utf8(prompt_ids.iter().map(|&b| b as u8).collect()).unwrap();
        let max_tokens = 8;

        // Independent computation: forward_batch over the prompt, then
        // one forward_token + argmax per new position, decoding each
        // generated id one at a time and concatenating -- exactly
        // `generate`'s own decode granularity (`ServerTokenizer::Byte`
        // is lossy per non-ASCII byte, so decoding token-by-token vs.
        // decoding the whole sequence at once are not equivalent; this
        // must replicate the real call pattern, not just the ids).
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
        let mut logits = decoder
            .forward_batch(&prompt_ids, 0, &mut caches)
            .pop()
            .unwrap();
        let mut expected_text = String::new();
        for pos in (prompt_ids.len()..).take(max_tokens) {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            expected_text.push_str(&ServerTokenizer::Byte.decode(&[next]));
            logits = decoder.forward_token(next, pos, &mut caches);
        }

        let mut actual_text = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(max_tokens),
            None,
            None,
            None,
            None,
            |_, s| actual_text.push_str(s),
        )
        .unwrap();

        assert_eq!(actual_text, expected_text);
    }

    #[test]
    fn rejects_out_of_vocab_prompt_tokens() {
        // ByteTokenizer emits raw bytes; vocab 32 makes ASCII letters OOV.
        let decoder = Decoder::new_random_small(test_dense_fixture(), 2, 32);
        let result = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            "hello",
            &greedy_params(4),
            None,
            None,
            None,
            None,
            |_, _| {},
        );
        assert!(matches!(result, Err(DecodeError::TokenOutOfVocab { .. })));
    }

    #[test]
    fn greedy_generation_hits_length_without_eos() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let mut chunks = String::new();
        let (finish, _rows, _ids, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            None,
            None,
            None,
            None,
            |_, s| chunks.push_str(s),
        )
        .unwrap();
        assert_eq!(finish[0].0, FinishReason::Length);
    }

    /// A cancel raised while the loop is running must actually stop it,
    /// keep whatever was already decoded, and say `Cancelled` -- not
    /// `Stop`, which a client would render as a finished answer.
    #[test]
    fn a_cancelled_generation_stops_early_and_keeps_its_tokens() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let cancel = crate::cancel::CancelToken::new();

        let mut params = greedy_params(200);
        params.cancel = Some(cancel.clone());

        let mut chunks = String::new();
        let mut emitted = 0usize;
        let (finish, _rows, _ids, usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &params,
            None,
            None,
            None,
            None,
            |_, s| {
                chunks.push_str(s);
                emitted += 1;
                // Stands in for the socket dropping (or `/v1/cancel`
                // arriving) a few tokens into the answer.
                if emitted == 3 {
                    cancel.cancel();
                }
            },
        )
        .unwrap();

        assert_eq!(finish[0].0, FinishReason::Cancelled);
        assert!(
            usage.completion_tokens < 200,
            "cancelling did not shorten the decode: {} tokens",
            usage.completion_tokens
        );
        assert!(
            !chunks.is_empty(),
            "the tokens decoded before the cancel must survive it"
        );
    }

    /// A generation nobody cancelled must be untouched by the machinery
    /// -- the flag is polled every token, so a bug here would shorten
    /// every answer on the server.
    #[test]
    fn an_uncancelled_generation_runs_to_its_normal_end() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let mut params = greedy_params(5);
        params.cancel = Some(crate::cancel::CancelToken::new());

        let (finish, _rows, _ids, usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &params,
            None,
            None,
            None,
            None,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(finish[0].0, FinishReason::Length);
        assert_eq!(usage.completion_tokens, 5);
    }

    /// Discovers the real greedy-argmax next-token id after `prompt_ids`
    /// via `forward_batch` (ground truth: its last returned row predicts
    /// the token immediately after the full prompt, exactly what
    /// `generate` computes as its first generation-loop `logits` value)
    /// -- not by decoding it to text and reading bytes back, which is
    /// lossy for `ByteTokenizer`: a standalone byte >= 128 is not valid
    /// UTF-8 on its own, so `String::from_utf8_lossy` replaces it with
    /// the 3-byte U+FFFD replacement character, and reading
    /// `s.bytes().next()` off that recovers 0xEF (239), not the
    /// original token id.
    fn greedy_next_token_after(decoder: &Decoder, prompt_ids: &[usize]) -> usize {
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
        let logits = decoder
            .forward_batch(prompt_ids, 0, &mut caches)
            .pop()
            .unwrap();
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn eos_token_stops_generation_before_max_tokens() {
        let decoder = small_decoder();
        let prompt_ids = vec![1usize, 2];
        let prompt = String::from_utf8(prompt_ids.iter().map(|&b| b as u8).collect()).unwrap();

        // ByteTokenizer::encode is a lossless direct byte->id mapping
        // (only decode is lossy, see greedy_next_token_after's doc
        // comment), so `generate`'s internal prompt replay reaches
        // exactly the same state as this direct computation.
        let eos = greedy_next_token_after(&decoder, &prompt_ids);

        let (finish, _rows, _ids, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::from_eos(Some(eos)),
            None,
            &prompt,
            &greedy_params(50),
            None,
            None,
            None,
            None,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(
            finish[0].0,
            FinishReason::Stop,
            "generation must stop as soon as the greedy-chosen token matches eos_id, not run to max_tokens"
        );
    }

    /// The bug this replaced: every server decode loop carried a single
    /// `eos_id`, so a Llama-3 checkpoint (whose `eos_token_id` is
    /// `<|end_of_text|>` while turns end with `<|eot_id|>`) or a gemma-2
    /// one (`<end_of_turn>`) ran past the end of its own turn to
    /// `max_tokens` over HTTP even after `eog_token_ids` landed for the
    /// CLI. Here the metadata EOS is deliberately a token the model will
    /// never pick, and the *turn ender* is the greedy next token: only a
    /// loop that consults the whole set stops.
    #[test]
    fn a_turn_ender_that_is_not_the_metadata_eos_still_stops_generation() {
        let decoder = small_decoder();
        let prompt_ids = vec![1usize, 2];
        let prompt = String::from_utf8(prompt_ids.iter().map(|&b| b as u8).collect()).unwrap();
        let turn_ender = greedy_next_token_after(&decoder, &prompt_ids);
        let never_sampled = (turn_ender + 1) % decoder.config.vocab_size;

        let stop = StopTokens::from_eos(Some(never_sampled)).with_id(Some(turn_ender));
        let (finish, _rows, _ids, usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &stop,
            None,
            &prompt,
            &greedy_params(50),
            None,
            None,
            None,
            None,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(finish[0].0, FinishReason::Stop);
        assert_eq!(
            usage.completion_tokens, 0,
            "the very first sampled token was the turn ender"
        );
    }

    #[test]
    fn a_stop_sequence_that_never_matches_does_not_drop_any_generated_content() {
        // The hold-back buffering (needed so a stop sequence spanning
        // more than one token is never partially flushed) must not
        // silently swallow output when no stop sequence ever matches:
        // the final emitted text must be byte-for-byte identical to an
        // otherwise-identical run with no stop sequences configured at
        // all, since the buffering is purely about *when* text is
        // flushed, never *whether* it is.
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8]).unwrap();

        let mut baseline = String::new();
        let (baseline_finish, _rows, _ids, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(20),
            None,
            None,
            None,
            None,
            |_, s| baseline.push_str(s),
        )
        .unwrap();

        let mut with_unmatchable_stop = String::new();
        let (stop_finish, _rows, _ids, _usage2) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &GenerationParams {
                cache_salt: None,
                prompt_logprobs: None,
                wants_logprobs: false,
                reasoning: None,
                max_tokens: 20,
                sampling: SamplingParams::default(),
                seed: 1,
                n: 1,
                stop: vec!["ZZ_NEVER_MATCHES_ZZ".to_string()],
                stop_token_ids: Vec::new(),
                json_object: false,
                grammar: None,
                cancel: None,
                ignore_eos: false,
                reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
                lora: None,
            },
            None,
            None,
            None,
            None,
            |_, s| with_unmatchable_stop.push_str(s),
        )
        .unwrap();

        assert_eq!(baseline_finish[0].0, FinishReason::Length);
        assert_eq!(stop_finish[0].0, FinishReason::Length);
        assert_eq!(with_unmatchable_stop, baseline);
    }

    /// A decode loop whose next token is scripted, so the two stop
    /// layers can be exercised on exactly the token sequence they are
    /// meant to react to rather than on whatever a random decoder
    /// happens to emit.
    ///
    /// `script` is the token id produced at each step; `render` is how
    /// each id detokenizes.
    fn run_scripted(
        script: &[usize],
        render: impl Fn(usize) -> String,
        params: &GenerationParams,
    ) -> (FinishReason, Vec<usize>, Vec<String>) {
        run_scripted_with_stops(script, render, params, StopTokens::from_eos(None))
    }

    fn run_scripted_with_stops(
        script: &[usize],
        render: impl Fn(usize) -> String,
        params: &GenerationParams,
        stop_tokens: StopTokens,
    ) -> (FinishReason, Vec<usize>, Vec<String>) {
        try_run_scripted_with_stops(script, render, params, stop_tokens)
            .expect("an unconstrained script cannot fail to decode")
    }

    /// As [`run_scripted_with_stops`], for the tests that are ABOUT a
    /// generation that stops with an error -- a grammar with no legal
    /// continuation. Everything else goes through the unwrapping
    /// version, so a decode that starts failing is a test failure
    /// rather than a quietly different return value.
    fn try_run_scripted_with_stops(
        script: &[usize],
        render: impl Fn(usize) -> String,
        params: &GenerationParams,
        stop_tokens: StopTokens,
    ) -> Result<(FinishReason, Vec<usize>, Vec<String>), DecodeError> {
        let vocab = script.iter().copied().max().unwrap_or(0) + 2;
        let logits_for = |id: usize| {
            let mut v = vec![0.0f32; vocab];
            v[id] = 10.0;
            v
        };
        let mut next = 0usize;
        let mut take = || {
            let id = script
                .get(next)
                .copied()
                .unwrap_or(script[script.len() - 1]);
            next += 1;
            id
        };
        let first = logits_for(take());
        let mut chunks: Vec<String> = Vec::new();
        let (finish, ids, _, _probs) = crate::sampling_loop::sample_until_stop(
            first,
            0,
            // A scripted-logits harness with no real prompt: the empty
            // half is the truth here, not an omission.
            &[],
            &stop_tokens,
            params,
            |ids| {
                ids.iter()
                    .copied()
                    .map(&render)
                    .collect::<String>()
                    .into_bytes()
            },
            &mut |_tok: usize, _pos: usize| logits_for(take()),
            |chunk| chunks.push(chunk.to_string()),
            &render,
            0,
        )?;
        Ok((finish, ids, chunks))
    }

    /// The wiring #124 was about, end to end through the decode loop.
    ///
    /// The unit tests in `crate::utf8_stream` prove the buffer works.
    /// They cannot prove `sample_until_stop` USES it, and that call is
    /// the whole fix -- the same gap that let a batched row ship with
    /// no timings. So this drives the real loop with a tokenizer whose
    /// tokens are single bytes, which is exactly what a byte-fallback
    /// vocabulary does to an emoji.
    ///
    /// It cannot go through `run_scripted`: that helper's `render`
    /// returns `String`, and half a character has no `String`.
    #[test]
    fn a_character_split_across_tokens_is_emitted_whole_by_the_decode_loop() {
        let smiley = "😊".as_bytes().to_vec();
        assert_eq!(smiley.len(), 4, "the point of this test");
        let vocab = smiley.len() + 2;
        let logits_for = |id: usize| {
            let mut v = vec![0.0f32; vocab];
            v[id] = 10.0;
            v
        };
        let len = smiley.len();
        let mut next = 0usize;
        let mut take = move || {
            let id = next.min(len - 1);
            next += 1;
            id
        };
        let first = logits_for(take());
        let bytes = smiley;
        let mut chunks: Vec<String> = Vec::new();
        let (_finish, ids, _, _probs) = crate::sampling_loop::sample_until_stop(
            first,
            0,
            &[],
            &StopTokens::default(),
            &scripted_params(4),
            // One byte per token: each of the middle ones is invalid
            // UTF-8 on its own, which is the whole problem.
            |ids| ids.iter().map(|&id| bytes[id]).collect(),
            &mut |_tok: usize, _pos: usize| logits_for(take()),
            |chunk| chunks.push(chunk.to_string()),
            &|_id| String::new(),
            0,
        )
        .expect("decode");

        assert_eq!(ids.len(), 4, "four tokens, one per byte");
        assert_eq!(
            chunks.concat(),
            "😊",
            "a character split across tokens must not be decoded per token"
        );
        assert!(
            !chunks.concat().contains(char::REPLACEMENT_CHARACTER),
            "the bytes were valid UTF-8 together; only the split made them look invalid"
        );
    }

    fn scripted_params(max_tokens: usize) -> GenerationParams {
        GenerationParams {
            cache_salt: None,
            prompt_logprobs: None,
            wants_logprobs: false,
            reasoning: None,
            max_tokens,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            seed: 1,
            n: 1,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            json_object: false,
            grammar: None,
            cancel: None,
            ignore_eos: false,
            reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
            lora: None,
        }
    }

    /// A grammar reaches the PRIVATE decode loop -- the one `generate`
    /// runs -- and both of its halves do.
    ///
    /// The script wants token 2 ("c") at every step, and `root ::= "ab"`
    /// makes that illegal at every step. The mask alone would give
    /// `[0, 0]`, because without the accept the grammar keeps answering
    /// "what may the FIRST token be?"; both halves give `[0, 1]`, which
    /// is the only string this grammar admits.
    ///
    /// The vacuity check is the unconstrained run below it: the same
    /// script with no grammar must produce the token the grammar had to
    /// take away, or this proves nothing.
    #[test]
    fn a_grammar_constrains_the_private_decode_loop() {
        let render = |id: usize| char::from(b'a' + id as u8).to_string();
        let script = [2usize, 2, 2, 2];

        let unconstrained = run_scripted(&script, render, &scripted_params(2));
        assert_eq!(
            unconstrained.1,
            vec![2, 2],
            "the model wants \"cc\", so a grammar forbidding it has work to do"
        );

        let mut params = scripted_params(2);
        params.grammar = Some(std::sync::Arc::new(
            frink_models::grammar::Grammar::from_str_with_root(r#"root ::= "ab""#, "root")
                .expect("test grammar parses"),
        ));
        let (finish, ids, chunks) = run_scripted(&script, render, &params);
        assert_eq!(
            ids,
            vec![0, 1],
            "the grammar was not applied token by token"
        );
        assert_eq!(chunks.concat(), "ab");
        assert_eq!(finish, FinishReason::Length);
    }

    /// The same loop, where the grammar finishes before `max_tokens`
    /// does and this vocabulary has no end-of-generation token to end
    /// on: a COMPLETE answer, reported as `Stop` rather than as an
    /// error or as 8 tokens of whatever came next.
    #[test]
    fn a_completed_grammar_ends_the_private_decode_loop() {
        let render = |id: usize| char::from(b'a' + id as u8).to_string();
        let mut params = scripted_params(8);
        params.grammar = Some(std::sync::Arc::new(
            frink_models::grammar::Grammar::from_str_with_root(r#"root ::= "ab""#, "root")
                .expect("test grammar parses"),
        ));
        let (finish, ids, chunks) = run_scripted(&[2usize; 8], render, &params);
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(chunks.concat(), "ab");
        assert_eq!(finish, FinishReason::Stop);
    }

    /// And a grammar this vocabulary cannot spell STOPS, with an error
    /// naming the constraint -- rather than serving text that does not
    /// satisfy the grammar the caller was told was applied.
    #[test]
    fn a_grammar_the_vocabulary_cannot_spell_fails_the_generation() {
        let render = |id: usize| char::from(b'a' + id as u8).to_string();
        let mut params = scripted_params(4);
        params.grammar = Some(std::sync::Arc::new(
            frink_models::grammar::Grammar::from_str_with_root(r#"root ::= "z""#, "root")
                .expect("test grammar parses"),
        ));
        let err =
            try_run_scripted_with_stops(&[2usize; 4], render, &params, StopTokens::from_eos(None))
                .expect_err("no token in this vocabulary renders as \"z\"");
        assert!(
            matches!(err, DecodeError::GrammarConstraint { .. }),
            "{err}"
        );
    }

    /// Layer 1: a stop *token* ends the answer, and the token itself
    /// never appears in it -- the same treatment EOS already gets,
    /// which is the point. A control token the client named is no more
    /// part of the output than the end-of-sequence token is.
    ///
    /// Confirmed to FAIL (runs to all 6 tokens, emitting "aabaab") when
    /// the `is_stop_token` check is removed from `sample_until_stop`.
    #[test]
    fn a_token_level_stop_ends_generation_and_never_reaches_the_output() {
        let render = |id: usize| char::from(b'a' + id as u8).to_string();
        let script = [0usize, 0, 1, 0, 0, 1];

        let (finish, ids, chunks) = run_scripted(&script, render, &scripted_params(6));
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(ids, script.to_vec());
        assert_eq!(chunks.concat(), "aabaab");

        let (finish, ids, chunks) = run_scripted(
            &script,
            render,
            &GenerationParams {
                stop_token_ids: vec![1],
                ..scripted_params(6)
            },
        );
        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(ids, vec![0, 0], "the stop token is not part of the answer");
        assert_eq!(
            chunks.concat(),
            "aa",
            "the stop token must not be rendered into the output"
        );
    }

    /// The case layer 2 provably cannot cover: a control token that
    /// renders as the empty string. The text scan has nothing to match
    /// on, so only matching the id ends the answer.
    #[test]
    fn a_stop_token_that_renders_as_nothing_is_still_a_stop() {
        // Token 1 detokenizes to "", as real special tokens can.
        let render = |id: usize| {
            if id == 1 {
                String::new()
            } else {
                char::from(b'a' + id as u8).to_string()
            }
        };
        let script = [0usize, 1, 0, 0];

        // The text layer alone: the stop string never appears, so
        // generation runs to its limit.
        let (finish, _, chunks) = run_scripted(
            &script,
            render,
            &GenerationParams {
                stop: vec!["<|end|>".to_string()],
                ..scripted_params(4)
            },
        );
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(chunks.concat(), "aaa");

        // With the id resolved, it stops where it should.
        let (finish, ids, chunks) = run_scripted(
            &script,
            render,
            &GenerationParams {
                stop: vec!["<|end|>".to_string()],
                stop_token_ids: vec![1],
                ..scripted_params(4)
            },
        );
        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(ids, vec![0]);
        assert_eq!(chunks.concat(), "a");
    }

    /// Layer 2: nothing that turns out to be part of the stop string
    /// is ever emitted. `emit` is append-only -- SSE has no way to take
    /// a chunk back -- so over-emitting is not a display glitch, it is
    /// the stop sequence failing to do the one thing it promises.
    ///
    /// The script spells "ab" (a partial match that is disproved) and
    /// then "abc" (the real one), so the buffer has to hold, release,
    /// and hold again.
    #[test]
    fn nothing_that_becomes_part_of_the_stop_is_ever_emitted() {
        let render = |id: usize| char::from(b'a' + id as u8).to_string();
        let script = [0usize, 1, 0, 1, 2, 0];
        let params = GenerationParams {
            stop: vec!["abc".to_string()],
            ..scripted_params(6)
        };

        let (finish, _, chunks) = run_scripted(&script, render, &params);
        assert_eq!(
            finish,
            FinishReason::StopSequence("abc".to_string()),
            "the reason names the stop that fired, not merely that one did"
        );
        assert_eq!(
            chunks.concat(),
            "ab",
            "the answer is everything before the stop, and nothing after it"
        );

        // Append-only means every intermediate state must already be a
        // prefix of that: emitting one character too many can never be
        // undone.
        let mut seen = String::new();
        for chunk in &chunks {
            seen.push_str(chunk);
            assert!(
                "ab".starts_with(&seen),
                "the stream ran ahead of the answer: {seen:?} (chunks: {chunks:?})"
            );
        }
    }

    /// A partial match that is disproved is released with the very
    /// token that disproves it, not carried to the end of the answer.
    ///
    /// The conservative alternative -- always withhold
    /// `longest_stop - 1` bytes -- is equally safe and permanently
    /// leaves the stream that many bytes behind the model, for a match
    /// that in most chunks is not even beginning.
    ///
    /// Confirmed to FAIL (first chunk is "a", not "abd") when
    /// `partial_suffix_len` is replaced by that fixed hold-back.
    #[test]
    fn a_disproved_partial_is_released_by_the_token_that_disproves_it() {
        let render = |id: usize| char::from(b'a' + id as u8).to_string();
        // "a", "b" are held as a possible "abc"; "d" settles it.
        let script = [0usize, 1, 3, 0];
        let (_, _, chunks) = run_scripted(
            &script,
            render,
            &GenerationParams {
                stop: vec!["abc".to_string()],
                ..scripted_params(4)
            },
        );
        assert_eq!(chunks.concat(), "abda", "no output is lost");
        assert_eq!(
            chunks.first().map(String::as_str),
            Some("abd"),
            "the whole disproved partial goes out at once: {chunks:?}"
        );
    }

    #[test]
    fn a_stop_sequence_that_does_match_truncates_output_before_it() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8]).unwrap();

        // Discover what greedy decode actually produces, then use a
        // substring of it (starting after the first character, so at
        // least one character of real output precedes the match) as a
        // stop sequence guaranteed to match.
        let mut baseline = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(20),
            None,
            None,
            None,
            None,
            |_, s| baseline.push_str(s),
        )
        .unwrap();
        let Some((cut, _)) = baseline.char_indices().nth(1) else {
            // Degenerate case for this decoder/seed: fewer than 2 chars
            // generated: nothing meaningful to truncate, skip.
            return;
        };
        // This decoder emits arbitrary bytes through `ServerTokenizer::
        // Byte`, so the tail can end in a U+FFFD that `Utf8Stream::flush`
        // produced -- a character whose remaining bytes never arrived
        // because generation hit `max_tokens` mid-sequence.
        //
        // That one is NOT matchable, and correctly so: while the loop is
        // running, those bytes may still be completed by the next token,
        // so the replacement does not exist yet. It is only knowable
        // once generation has ended, which is after every stop decision
        // has been made. Trimming it keeps this test about what it says
        // it is about -- a stop sequence that DOES match.
        let stop_str = baseline[cut..]
            .trim_end_matches(char::REPLACEMENT_CHARACTER)
            .to_string();
        if stop_str.is_empty() {
            return;
        }

        let mut truncated = String::new();
        let (finish, _rows, _ids, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &GenerationParams {
                cache_salt: None,
                prompt_logprobs: None,
                wants_logprobs: false,
                reasoning: None,
                max_tokens: 20,
                sampling: SamplingParams::default(),
                seed: 1,
                n: 1,
                stop: vec![stop_str.clone()],
                stop_token_ids: Vec::new(),
                json_object: false,
                grammar: None,
                cancel: None,
                ignore_eos: false,
                reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
                lora: None,
            },
            None,
            None,
            None,
            None,
            |_, s| truncated.push_str(s),
        )
        .unwrap();

        assert_eq!(finish[0].0, FinishReason::StopSequence(stop_str));
        assert_eq!(truncated, baseline[..cut]);
    }

    #[test]
    fn usage_reports_both_phases_and_a_time_to_first_token() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let (_finish, _rows, _ids, usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            None,
            None,
            None,
            None,
            |_, _| {},
        )
        .unwrap();

        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(usage.completion_tokens, 5);
        let prefill = usage.prompt_eval_duration_ms.expect("prefill timed");
        let decode = usage.generation_duration_ms.expect("decode timed");
        let ttft = usage.time_to_first_token_ms.expect("first token timed");
        // TTFT is measured from the start of prefill, so it can never be
        // shorter than prefill, and the first of five tokens must land
        // before the decode loop finishes all five.
        assert!(ttft >= prefill, "ttft {ttft} < prefill {prefill}");
        assert!(
            ttft <= prefill + decode + 1.0,
            "ttft {ttft} exceeds the whole request"
        );
    }

    #[test]
    fn cached_tokens_distinguishes_a_miss_from_an_absent_prefix_cache() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();

        let (_f, _rows, _ids, no_cache) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(2),
            None,
            None,
            None,
            None,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(no_cache.cached_tokens, None, "no prefix cache configured");

        let pc = Mutex::new(PrefixCache::new(4));
        let (_f, _rows, _ids, miss) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(2),
            None,
            None,
            Some(&pc),
            None,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(miss.cached_tokens, Some(0), "cache consulted, missed");

        // Second turn extends the first: three prompt tokens are reused.
        let longer = String::from_utf8(vec![1u8, 2, 3, 9]).unwrap();
        let (_f, _rows, _ids, hit) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &longer,
            &greedy_params(2),
            None,
            None,
            Some(&pc),
            None,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(hit.cached_tokens, Some(3));
    }

    #[test]
    fn prefix_cache_reuses_a_shared_prefix_and_produces_the_same_output_as_a_fresh_run() {
        let decoder = small_decoder();
        let prompt1 = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let pc = Mutex::new(PrefixCache::new(4));

        let mut out1 = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt1,
            &greedy_params(5),
            None,
            None,
            Some(&pc),
            None,
            |_, s| out1.push_str(s),
        )
        .unwrap();
        assert_eq!(pc.lock().unwrap().stats().misses, 1);

        // prompt2's tokens (raw bytes, via ByteTokenizer) start with
        // prompt1's exact bytes -- the common multi-turn-chat shape.
        let prompt2 = String::from_utf8(vec![1u8, 2, 3, 9, 9]).unwrap();

        let mut out2_with_cache = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt2,
            &greedy_params(5),
            None,
            None,
            Some(&pc),
            None,
            |_, s| out2_with_cache.push_str(s),
        )
        .unwrap();
        let stats = pc.lock().unwrap().stats();
        assert_eq!(stats.hits, 1, "prompt2 must hit the stored prompt1 entry");
        assert_eq!(stats.total_positions_reused, 3);

        let mut out2_fresh = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt2,
            &greedy_params(5),
            None,
            None,
            None,
            None,
            |_, s| out2_fresh.push_str(s),
        )
        .unwrap();

        assert_eq!(
            out2_with_cache, out2_fresh,
            "restoring from the prefix cache must produce identical output to processing the whole prompt from scratch"
        );
    }

    #[test]
    fn prefix_cache_exact_repeat_skips_prompt_processing_via_pending_logits() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let pc = Mutex::new(PrefixCache::new(4));

        let mut out1 = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            None,
            None,
            Some(&pc),
            None,
            |_, s| out1.push_str(s),
        )
        .unwrap();

        // The exact same prompt again: the stored entry's `tokens` (the
        // full prompt+completion from the first call) starts with this
        // exact prompt, so this only matches the prompt-length prefix
        // of a longer stored entry, not an exact full-entry match --
        // covering the "no pending_logits available" fallback path,
        // not the zero-forward-pass shortcut. Still must produce
        // identical output to a from-scratch run.
        let mut out2_with_cache = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            None,
            None,
            Some(&pc),
            None,
            |_, s| out2_with_cache.push_str(s),
        )
        .unwrap();

        let mut out2_fresh = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            None,
            None,
            None,
            None,
            |_, s| out2_fresh.push_str(s),
        )
        .unwrap();

        assert_eq!(out2_with_cache, out2_fresh);
    }

    #[test]
    fn prefix_cache_is_not_consulted_when_a_kv_pool_is_configured() {
        let decoder = small_decoder(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let pc = Mutex::new(PrefixCache::new(4));
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 2)));
        let config = pool_config(pool, Duration::ZERO);

        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            Some(&pc),
            None,
            |_, _| {},
        )
        .unwrap();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            Some(&pc),
            None,
            |_, _| {},
        )
        .unwrap();

        let stats = pc.lock().unwrap().stats();
        assert_eq!(
            stats.hits + stats.misses,
            0,
            "prefix cache must never be consulted while a KV pool is configured"
        );
    }

    fn pool_config(pool: Arc<Mutex<KvBlockPool>>, queue_wait: Duration) -> KvPoolConfig {
        KvPoolConfig { pool, queue_wait }
    }

    /// Published prefixes must come BACK, or a long-running server
    /// refuses requests that fit.
    ///
    /// `publish_to_radix` retains a page group for every page it hands
    /// the tree, and nothing released them: `RadixCache::evict` had no
    /// caller anywhere. So the pool shrank monotonically, and a server
    /// that had been up long enough started answering
    /// `PagedStoreExhausted` while the tree sat on pages no request was
    /// reading.
    ///
    /// This asserts the POOL recovers, not that `evict` was called. A
    /// test that only checks the call happened proves nothing about a
    /// leak, which is how this one survived having 33 tests around it.
    #[test]
    fn published_prefixes_are_reclaimed_under_pressure() {
        let decoder = small_decoder();
        let block_size = 4;
        // Small on purpose: enough for a few prompts at once, so the
        // pool must be recycled rather than merely large.
        let config = paged_config_with_radix(&decoder, block_size, 24, true);
        let free_before = config.store.free_groups();
        assert!(free_before > 0, "the fixture must start with free pages");

        // Each prompt is distinct, so every one publishes a NEW prefix
        // and none of them can be adopted from an earlier one. Without
        // eviction the tree accumulates all of them and the pool runs
        // dry.
        for round in 0..40u32 {
            let tokens: Vec<usize> = (0..12).map(|i| (round * 100 + i) as usize).collect();
            let mut lease = acquire_paged_caches(&decoder, &config, &tokens, tokens.len() + 8)
                .expect("admission must keep succeeding once the tree can be evicted");
            publish_to_radix(&mut lease, &tokens, block_size);
            drop(lease);
        }

        // The pool is whole again: every page either sits in the tree
        // as evictable or is free, and nothing has leaked.
        let tree_holds = {
            let tree = config
                .radix
                .as_ref()
                .expect("configured with a tree")
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            tree.total_size().div_ceil(block_size)
        };
        assert_eq!(
            config.store.free_groups() + tree_holds,
            free_before,
            "every page must be free or accounted for in the tree: \
             free={} tree={} started={free_before}",
            config.store.free_groups(),
            tree_holds
        );
    }

    fn paged_config(decoder: &Decoder, block_size: usize, blocks: usize) -> PagedKvConfig {
        paged_config_with_radix(decoder, block_size, blocks, false)
    }

    fn paged_config_with_radix(
        decoder: &Decoder,
        block_size: usize,
        blocks: usize,
        share_prefixes: bool,
    ) -> PagedKvConfig {
        PagedKvConfig {
            store: Arc::new(decoder.config.new_paged_kv(block_size, blocks)),
            queue_wait: Duration::ZERO,
            radix: share_prefixes.then(|| {
                Arc::new(Mutex::new(crate::policy::radix::RadixCache::new(
                    block_size,
                )))
            }),
            anchor_token: None,
            slide_interval: crate::policy::pool_budget::DEFAULT_SWA_EVICTION_INTERVAL,
        }
    }

    /// Serving on paged KV must produce the SAME TEXT as serving on
    /// contiguous KV.
    ///
    /// This is the property the whole paged path exists to preserve,
    /// and the one no lower-level test can state: the decoder tests pin
    /// bit-identity of logits, but a caller only ever sees tokens, and
    /// between the two sit admission, prefill, the decode loop and
    /// sampling. If any of those dispatched differently, the logits
    /// could match and the answer still change.
    ///
    /// Greedy sampling makes the comparison exact rather than
    /// distributional.
    #[test]
    fn a_paged_request_generates_the_same_text_as_a_contiguous_one() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();

        let mut contiguous = String::new();
        let (finish_a, _rows, _ids, usage_a) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(6),
            None,
            None,
            None,
            None,
            |_, s| contiguous.push_str(s),
        )
        .unwrap();

        let config = paged_config(&decoder, /* block_size = */ 4, /* blocks = */ 64);
        let mut paged = String::new();
        let (finish_b, _rows, _ids, usage_b) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(6),
            None,
            Some(&config),
            None,
            None,
            |_, s| paged.push_str(s),
        )
        .unwrap();

        assert_eq!(paged, contiguous, "paged serving changed the answer");
        assert_eq!(finish_b, finish_a);
        assert_eq!(usage_b.completion_tokens, usage_a.completion_tokens);
        assert_eq!(usage_b.prompt_tokens, usage_a.prompt_tokens);
    }

    /// Every page comes back when the request ends.
    ///
    /// `PagedKvCache` has no `Drop`, so releasing is `PagedLease`'s job
    /// alone. Without it the store bleeds a whole request's pages per
    /// request and a long-running server stops admitting anything, with
    /// nothing in the logs to say why. Checked per layer, since the
    /// lease releases in a loop and a bound that skipped the last layer
    /// would still look right on layer 0.
    #[test]
    fn a_finished_paged_request_returns_every_page_it_held() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let config = paged_config(&decoder, 4, 64);
        let before: Vec<usize> = (0..decoder.layers.len())
            .map(|l| config.store.free_blocks(l))
            .collect();

        for _ in 0..3 {
            let mut out = String::new();
            generate(
                &decoder,
                &ServerTokenizer::Byte,
                &StopTokens::default(),
                None,
                &prompt,
                &greedy_params(5),
                None,
                Some(&config),
                None,
                None,
                |_, s| out.push_str(s),
            )
            .unwrap();
        }

        for (l, expected) in before.iter().enumerate() {
            assert_eq!(
                config.store.free_blocks(l),
                *expected,
                "layer {l} leaked pages across repeated requests"
            );
        }
    }

    /// A model where EVERY layer slides by the same window, which is
    /// what makes a page group releasable: the group holds one block in
    /// every layer, so one full-attention layer would still be reading
    /// the block the slide gave away.
    fn windowed_decoder(window: usize) -> Decoder {
        let mut cfg = test_dense_fixture();
        cfg.sliding_window = Some(window);
        cfg.swa_layers = frink_models::swa_layers::SwaLayers::All;
        Decoder::new_random_small(cfg, 2, 256)
    }

    fn run_to_completion(max_tokens: usize) -> GenerationParams {
        GenerationParams {
            // Every one of these tokens must actually be generated, or
            // a run that stopped at token 5 would "pass" a test about
            // what happens after 200.
            ignore_eos: true,
            ..greedy_params(max_tokens)
        }
    }

    /// The arithmetic both admission paths share, on its own.
    ///
    /// A unit test because the two callers price the same request in
    /// different units -- the store in page groups, the batch
    /// scheduler's budget in positions -- and a formula that drifted
    /// between them would have the two components refusing and
    /// admitting the same request.
    #[test]
    fn a_window_prices_a_request_at_its_prompt_plus_a_bound() {
        let block_size = 4;
        let policy = WindowPolicy::new(8, block_size);
        let bound = slide_hold_bound(8, &policy);
        assert_eq!(bound, 2 * (8 + SWA_RETAIN_GAP) + 128 + 2 * block_size);

        // Full attention: the whole sequence, however long.
        assert_eq!(paged_hold_positions(10_000, 3, block_size, None), 10_000);
        assert_eq!(paged_groups_needed(10_000, 3, block_size, None), 2_500);

        // Windowed: prompt plus the bound plus a page of slack, and
        // flat in `max_seq_len` -- ten times the generation costs the
        // same pages, which IS the feature.
        let held = paged_hold_positions(10_000, 3, block_size, Some(&policy));
        assert_eq!(held, 3 + bound + block_size);
        assert_eq!(
            paged_hold_positions(100_000, 3, block_size, Some(&policy)),
            held,
            "a longer generation must not cost more pages"
        );
        // But a longer PROMPT does, because prefill materialises
        // positions 0..prompt_len to reuse the one prefill kernel.
        assert!(paged_hold_positions(10_000, 900, block_size, Some(&policy)) > held);

        // And a request shorter than the bound is not made more
        // expensive by being windowed.
        assert_eq!(paged_hold_positions(20, 3, block_size, Some(&policy)), 20);
        assert_eq!(
            paged_groups_needed(20, 3, block_size, Some(&policy)),
            paged_groups_needed(20, 3, block_size, None)
        );
    }

    /// THE ACCEPTANCE PROPERTY of this feature: a window model holds its
    /// prompt and a window, not its whole context.
    ///
    /// Stated as the only thing an operator can actually observe --
    /// whether the request is served. One store, one prompt, one length,
    /// two models: the windowed one runs, the full-attention one is
    /// refused by the same store. The window is the whole difference.
    #[test]
    fn a_window_model_runs_on_a_store_too_small_for_its_whole_context() {
        let window = 8;
        let windowed = windowed_decoder(window);
        assert_eq!(
            windowed.config.uniform_sliding_window(),
            Some(window),
            "the fixture must be uniformly windowed or this test proves nothing"
        );
        let full = small_decoder();
        assert_eq!(full.config.uniform_sliding_window(), None);

        let block_size = 4;
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let max_tokens = 400;
        // Between the two answers: 48 groups for the windowed request,
        // 101 for the same request without a window.
        let blocks = 60;
        let params = run_to_completion(max_tokens);

        let win_config = paged_config(&windowed, block_size, blocks);
        let mut out = String::new();
        let (_, _rows, _ids, usage) = generate(
            &windowed,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &params,
            None,
            Some(&win_config),
            None,
            None,
            |_, s| out.push_str(s),
        )
        .expect("a window model must fit a store sized for its window");
        assert_eq!(usage.completion_tokens, max_tokens);

        // Full attention, and ALTERNATING attention, are both refused by
        // the same store. The alternating case is the one worth spelling
        // out: its `kv_block_window` is `Some(8)`, so a slide keyed on
        // that question would have admitted it and then freed pages its
        // full-attention layers were still reading -- a wrong answer
        // rather than a refusal.
        let mut alternating_cfg = test_dense_fixture();
        alternating_cfg.sliding_window = Some(window);
        alternating_cfg.swa_layers = frink_models::swa_layers::SwaLayers::period(2, false);
        let alternating = Decoder::new_random_small(alternating_cfg, 2, 256);
        assert_eq!(alternating.config.kv_block_window(), Some(window));
        assert_eq!(alternating.config.uniform_sliding_window(), None);

        for (name, model) in [("full attention", &full), ("alternating", &alternating)] {
            let config = paged_config(model, block_size, blocks);
            let err = generate(
                model,
                &ServerTokenizer::Byte,
                &StopTokens::default(),
                None,
                &prompt,
                &params,
                None,
                Some(&config),
                None,
                None,
                |_, _| {},
            )
            .unwrap_err();
            assert!(matches!(err, DecodeError::KvPoolExhausted), "{name}: {err}");
        }
    }

    /// Sliding must not change what the model says.
    ///
    /// The sharp end of the whole design: a recycled page is overwritten
    /// by a later position, so a page freed one token too early does not
    /// fail -- it answers with another position's keys. Greedy sampling
    /// against the contiguous path, which applies the same window in
    /// attention and frees nothing, makes that visible as different
    /// text.
    ///
    /// Long enough to slide many times over: the default cadence is 128
    /// steps, so a six-token test would exercise none of this.
    #[test]
    fn a_sliding_paged_request_says_the_same_thing_as_a_contiguous_one() {
        let decoder = windowed_decoder(8);
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let params = run_to_completion(400);

        let mut contiguous = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &params,
            None,
            None,
            None,
            None,
            |_, s| contiguous.push_str(s),
        )
        .unwrap();

        let config = paged_config(&decoder, /* block_size = */ 4, /* blocks = */ 60);
        let mut paged = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &params,
            None,
            Some(&config),
            None,
            None,
            |_, s| paged.push_str(s),
        )
        .unwrap();

        assert_eq!(paged, contiguous, "the window slide changed the answer");
    }

    /// Once the window is sliding, a request stops asking the store for
    /// pages: it reuses its own.
    ///
    /// This is what keeps the footprint flat, and it is measured as the
    /// STORE's free-group count rather than as anything the lease
    /// reports about itself. A request that quietly kept acquiring would
    /// still answer correctly and would still bring a busy server down.
    #[test]
    fn a_long_windowed_generation_stops_taking_pages_from_the_store() {
        let decoder = windowed_decoder(8);
        let block_size = 4;
        let config = paged_config(&decoder, block_size, /* blocks = */ 200);
        let tokens: Vec<usize> = vec![1, 2, 3];

        let mut lease = acquire_paged_caches(&decoder, &config, &tokens, tokens.len() + 4_000)
            .expect("the store holds a window's worth");
        let after_admission = config.store.free_groups();
        let held = lease.groups.len();

        for pos in tokens.len()..tokens.len() + 4_000 {
            lease.before_step(pos);
            assert_eq!(
                config.store.free_groups(),
                after_admission,
                "position {pos} took a page from the store instead of recycling"
            );
        }
        assert!(
            lease.window.as_ref().unwrap().released > 0,
            "4000 positions at a window of 8 must have slid"
        );
        // Live pages plus spares is what admission reserved: recycling
        // moves groups between the two, it does not create or lose them.
        assert_eq!(
            lease.groups.iter().flatten().count() + lease.spare.len(),
            held
        );
    }

    /// A tool call holds the window back at the position the next turn
    /// will rejoin at.
    ///
    /// An agentic turn does not end the conversation: the harness runs
    /// the tool and comes back with the same context up to the call and
    /// a different one after it. So the position where the call opened
    /// is where the next request rejoins, and a window that followed the
    /// cursor would have thrown it away by then.
    ///
    /// Two identical runs, one with the checkpoint's anchor token
    /// configured and one without, differing only in that. The anchored
    /// one must hold strictly more, and must then let go once the cursor
    /// has drifted a whole window past -- because holding the cursor and
    /// an unbounded-distance anchor is what the pool sizing cannot pay
    /// for.
    #[test]
    fn a_tool_call_anchor_holds_the_window_back_and_then_lets_go() {
        let anchor_token = 77;
        let block_size = 4;
        let decoder = windowed_decoder(8);
        let tokens: Vec<usize> = vec![1, 2, 3];

        // Released positions at `check`, with the anchor token offered
        // at each of `at`, when `armed`.
        let released_at = |armed: bool, at: &[usize], check: usize| -> usize {
            let mut config = paged_config(&decoder, block_size, 400);
            // Every four steps rather than every 128: the cadence is
            // what makes the anchor's effect observable at a chosen
            // position rather than at the next multiple of the default.
            config.slide_interval = 4;
            config.anchor_token = armed.then_some(anchor_token as u32);
            let mut lease = acquire_paged_caches(&decoder, &config, &tokens, tokens.len() + 1_000)
                .expect("the store is large enough");
            for pos in tokens.len()..check {
                if at.contains(&(pos + 1)) {
                    lease.observe_sampled(anchor_token, pos + 1, false);
                }
                lease.before_step(pos);
            }
            lease.window.as_ref().unwrap().released
        };

        // Just after the call: the anchored run has held on to the
        // pages around it, the unanchored one has moved past them.
        let first = 200;
        assert!(
            released_at(true, &[first], 208) < released_at(false, &[first], 208),
            "the anchor did not hold the window back"
        );

        // Far past it: the anchored run has caught up, because the turn
        // is clearly not about to end and holding two windows an
        // unbounded distance apart needs an unbounded pool.
        assert_eq!(
            released_at(true, &[first], 600),
            released_at(false, &[first], 600),
            "the anchor was never dropped, so the hold is unbounded"
        );

        // And a LATER call anchors again. Only the first call of a turn
        // is the anchor, so a request that had one and dropped it must
        // be able to take another -- otherwise one early tool call
        // spends the anchor for the whole rest of the conversation.
        let second = 600;
        assert!(
            released_at(true, &[first, second], 610) < released_at(false, &[], 610),
            "a dropped anchor left the request unable to take another"
        );
    }

    /// A slid request returns its recycled pages too.
    ///
    /// `Drop` releases what `groups` names, and a slide takes groups OUT
    /// of `groups` -- so a lease that forgot its spare list would give
    /// back only the live window and leak everything the slide had
    /// recycled, which on a window model is most of what it held.
    /// Checked per layer, since a bound that skipped the last layer
    /// would still look right on layer 0.
    #[test]
    fn a_slid_request_returns_the_pages_it_recycled_as_well_as_the_ones_it_held() {
        let decoder = windowed_decoder(8);
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let config = paged_config(&decoder, /* block_size = */ 4, /* blocks = */ 60);
        let before: Vec<usize> = (0..decoder.layers.len())
            .map(|l| config.store.free_blocks(l))
            .collect();

        // Three runs on a store that could not serve even one of them
        // twice over: a leak shows as the second run being refused.
        for run in 0..3 {
            let mut out = String::new();
            generate(
                &decoder,
                &ServerTokenizer::Byte,
                &StopTokens::default(),
                None,
                &prompt,
                &run_to_completion(400),
                None,
                Some(&config),
                None,
                None,
                |_, s| out.push_str(s),
            )
            .unwrap_or_else(|e| panic!("run {run} was refused: {e}"));
        }

        for (l, expected) in before.iter().enumerate() {
            assert_eq!(
                config.store.free_blocks(l),
                *expected,
                "layer {l} kept the pages a slid request recycled"
            );
        }
    }

    /// THE SAFETY BOUNDARY: every page the kernel still reads is one the
    /// lease still owns.
    ///
    /// The paged attention kernel indexes `block_table[t / block_size]`
    /// for `t` from `seq_len - window`, so an index in that range whose
    /// group has been recycled is a page some LATER position has been
    /// writing into. That does not fail -- it answers with another
    /// position's keys.
    ///
    /// Checked as a property at every step rather than at the end,
    /// because a page freed one step too early is back to being safe a
    /// few steps later once the window has moved past it. And checked
    /// here rather than through generated text, which cannot see it: the
    /// reserve slack means a recycled page is not physically reused
    /// until long after the window has left it, so a slide freeing a
    /// whole window too much still produces the right answer on this
    /// fixture. Robust, but not evidence -- so the invariant is stated
    /// where it is true rather than where it happens to show.
    #[test]
    fn every_page_the_kernel_still_reads_is_one_the_lease_still_owns() {
        let window = 8;
        let block_size = 4;
        let decoder = windowed_decoder(window);
        let config = paged_config(&decoder, block_size, /* blocks = */ 200);
        let tokens: Vec<usize> = vec![1, 2, 3];

        let mut lease = acquire_paged_caches(&decoder, &config, &tokens, tokens.len() + 4_000)
            .expect("the store holds a window's worth");

        for pos in tokens.len()..tokens.len() + 4_000 {
            lease.before_step(pos);
            // What attention sees once this position is written.
            let seq_len = pos + 1;
            let first = seq_len.saturating_sub(window) / block_size;
            for i in first..=pos / block_size {
                assert!(
                    lease.groups[i].is_some(),
                    "at position {pos} the window reaches page {i}, which was recycled"
                );
            }
        }
        assert!(
            lease.window.as_ref().unwrap().released > 0,
            "nothing was recycled, so this proved nothing"
        );
    }

    /// A slid request never gives away the prefix the tree owns.
    ///
    /// Those pages are shared: another request is attending over them
    /// right now, and a third will adopt them tomorrow. The slide floors
    /// at the locked prefix rather than at zero for exactly that reason,
    /// and this checks the floor by looking at which page groups are
    /// still there rather than at what the policy returned.
    #[test]
    fn a_slid_request_never_recycles_the_prefix_the_tree_owns() {
        let decoder = windowed_decoder(8);
        let block_size = 4;
        let config = paged_config_with_radix(&decoder, block_size, 400, true);

        // Publish a prefix: a short request, which does not slide, so
        // its pages reach the tree.
        let shared: Vec<u8> = (1u8..=16).collect();
        let prompt = String::from_utf8(shared.clone()).unwrap();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(2),
            None,
            Some(&config),
            None,
            None,
            |_, _| {},
        )
        .unwrap();

        // A second request off that prefix, driven long past the window.
        let tokens: Vec<usize> = shared.iter().map(|&b| b as usize).collect();
        let mut lease = acquire_paged_caches(&decoder, &config, &tokens, tokens.len() + 3_000)
            .expect("the store is large enough");
        let locked = lease.adopted_positions(block_size);
        assert!(locked > 0, "this test needs a real prefix match");

        for pos in tokens.len()..tokens.len() + 3_000 {
            lease.before_step(pos);
        }
        let released = lease.window.as_ref().unwrap().released;
        assert!(released >= locked, "the slide must have passed the prefix");
        for i in 0..locked / block_size {
            assert!(
                lease.groups[i].is_some(),
                "page {i} of the shared prefix was recycled out from under the tree"
            );
        }
    }

    /// A sequence whose window slid is NOT published to the tree.
    ///
    /// The tree keys on a prefix, and a slid sequence's prefix is
    /// precisely the part it gave away -- what it still holds is a
    /// suffix at the cursor. Publishing anyway would hand the next
    /// request pages whose contents belong a thousand positions later,
    /// and it would match on them, and the answer would be wrong rather
    /// than slow.
    ///
    /// The control is the same prompt on a request too short to slide,
    /// which does publish. Without it this test would pass against a
    /// tree that never publishes anything.
    #[test]
    fn a_slid_sequence_is_not_published_to_the_tree() {
        let decoder = windowed_decoder(8);
        let block_size = 4;
        let prompt = String::from_utf8((1u8..=16).collect::<Vec<u8>>()).unwrap();

        let cached_after = |first: GenerationParams| -> usize {
            let config = paged_config_with_radix(&decoder, block_size, 400, true);
            let mut out = String::new();
            generate(
                &decoder,
                &ServerTokenizer::Byte,
                &StopTokens::default(),
                None,
                &prompt,
                &first,
                None,
                Some(&config),
                None,
                None,
                |_, s| out.push_str(s),
            )
            .unwrap();
            // What the SECOND request adopted, which on a fresh tree is
            // only ever what the first published.
            let mut probe = String::new();
            let (_, _rows, _ids, usage) = generate(
                &decoder,
                &ServerTokenizer::Byte,
                &StopTokens::default(),
                None,
                &prompt,
                &greedy_params(2),
                None,
                Some(&config),
                None,
                None,
                |_, s| probe.push_str(s),
            )
            .unwrap();
            usage.cached_tokens.unwrap_or(0)
        };

        assert!(
            cached_after(greedy_params(2)) > 0,
            "a request that never slid must publish its pages"
        );
        assert_eq!(
            cached_after(run_to_completion(600)),
            0,
            "a slid sequence published a prefix it no longer holds"
        );
    }

    /// THE ACCEPTANCE PROPERTY: two prompts sharing a prefix hold ONE
    /// copy of its pages, not two.
    ///
    /// This is what `frink-models::prefix_cache` structurally cannot
    /// do. That one clones a `Vec<KvCache>` per entry, so the second
    /// conversation off a shared system prompt holds its own copy of
    /// that prompt's KV. Here the second adopts the first's page
    /// groups and the refcount goes to two, so the pages consumed by
    /// two requests are strictly fewer than twice one request's.
    ///
    /// Measured as free groups, which is the store's own count rather
    /// than a number this test computes.
    #[test]
    fn two_prompts_sharing_a_prefix_hold_one_copy_of_it() {
        let decoder = small_decoder();
        // A shared prefix of 8 bytes, then one differing byte each.
        let shared: Vec<u8> = (1u8..=8).collect();
        let mut a = shared.clone();
        a.push(40);
        let mut b = shared.clone();
        b.push(50);
        let prompt_a = String::from_utf8(a).unwrap();
        let prompt_b = String::from_utf8(b).unwrap();

        let run = |config: &PagedKvConfig, prompt: &str| -> String {
            let mut out = String::new();
            generate(
                &decoder,
                &ServerTokenizer::Byte,
                &StopTokens::default(),
                None,
                prompt,
                &greedy_params(2),
                None,
                Some(config),
                None,
                None,
                |_, s| out.push_str(s),
            )
            .unwrap();
            out
        };

        // Measured as what each request COSTS the store, not what is
        // free afterwards: without a tree every group is released at
        // the end, so free-afterwards is identical either way and
        // measures nothing. With a tree, pages it keeps stay held, so
        // the drop in free groups is what each request added.
        let cfg = paged_config_with_radix(&decoder, 4, 64, true);
        let start = cfg.store.free_groups();
        let text_a = run(&cfg, &prompt_a);
        let after_a = cfg.store.free_groups();
        let text_b = run(&cfg, &prompt_b);
        let after_b = cfg.store.free_groups();

        let cost_a = start - after_a;
        let cost_b = after_a - after_b;
        assert!(cost_a > 0, "the first request must publish something");
        assert!(
            cost_b < cost_a,
            "the second request shares A's prefix and must cost less \
             (first {cost_a} groups, second {cost_b})"
        );

        // And the saving is REPORTED, not merely real: a caller sees
        // the adopted positions as `cached_tokens`, the same field the
        // contiguous prefix cache uses for the same meaning.
        let cfg2 = paged_config_with_radix(&decoder, 4, 64, true);
        let mut sink = String::new();
        let (_f, _rows, _ids, first_usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt_a,
            &greedy_params(2),
            None,
            Some(&cfg2),
            None,
            None,
            |_, s| sink.push_str(s),
        )
        .unwrap();
        assert_eq!(
            first_usage.cached_tokens,
            Some(0),
            "a cold tree reuses nothing"
        );
        let (_f, _rows, _ids, second_usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt_b,
            &greedy_params(2),
            None,
            Some(&cfg2),
            None,
            None,
            |_, s| sink.push_str(s),
        )
        .unwrap();
        let reused = second_usage.cached_tokens.expect("a tree is configured");
        assert!(
            reused >= 8,
            "the 8-token shared prefix must be reported as reused, got {reused}"
        );

        // And the answers are unchanged: adopting a prefix must not
        // change what the model says. A cache that is fast and wrong is
        // worse than no cache.
        let plain = paged_config(&decoder, 4, 64);
        assert_eq!(text_a, run(&plain, &prompt_a), "prompt A changed");
        assert_eq!(text_b, run(&plain, &prompt_b), "prompt B changed");
    }

    /// Repeated requests off one prefix do not exhaust the store.
    ///
    /// The leak this guards is specific: `insert_prefix` reports how
    /// much of the span the tree ALREADY had, and those pages of ours
    /// are the ones the tree did not take. Retaining the wrong range
    /// either leaks them (retained but never released) or frees pages
    /// the tree still points at.
    #[test]
    fn many_requests_off_one_prefix_neither_leak_nor_free_the_trees_pages() {
        let decoder = small_decoder();
        let config = paged_config_with_radix(&decoder, 4, 64, true);
        let shared: Vec<u8> = (1u8..=8).collect();

        let mut lows = Vec::new();
        for suffix in 0..6u8 {
            let mut p = shared.clone();
            p.push(60 + suffix);
            let prompt = String::from_utf8(p).unwrap();
            let mut out = String::new();
            generate(
                &decoder,
                &ServerTokenizer::Byte,
                &StopTokens::default(),
                None,
                &prompt,
                &greedy_params(2),
                None,
                Some(&config),
                None,
                None,
                |_, s| out.push_str(s),
            )
            .expect("the store is sized for many of these");
            lows.push(config.store.free_groups());
        }

        // The tree keeps some pages forever, so free groups settle to a
        // floor rather than returning to the start. What must NOT
        // happen is a monotone slide toward zero: after the prefix is
        // published once, later requests off it cost only their own
        // suffix, so the last two runs must leave the same amount free.
        assert_eq!(
            lows[lows.len() - 1],
            lows[lows.len() - 2],
            "steady state expected once the shared prefix is published; \
             free groups per run were {lows:?}"
        );
        assert!(
            lows[lows.len() - 1] > 0,
            "the store must not have been consumed: {lows:?}"
        );
    }

    /// A request too big for the store is refused at admission, having
    /// emitted nothing and taken no page.
    ///
    /// The refusal must happen BEFORE any work: `sample_until_stop`
    /// takes a closure returning `Vec<f32>` with nowhere to report a
    /// store that ran dry at token 300 of 400, which is why
    /// `acquire_paged_caches` reserves the whole worst-case length up
    /// front. The same reasoning `acquire_pooled_caches` records having
    /// learned from a live panic.
    #[test]
    fn a_paged_request_too_big_for_the_store_is_refused_before_emitting_anything() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3, 4]).unwrap();
        // One block of 2 positions per layer against a prompt of 4 plus
        // 8 more tokens.
        let config = paged_config(&decoder, /* block_size = */ 2, /* blocks = */ 1);

        let result = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(8),
            None,
            Some(&config),
            None,
            None,
            |_, _| panic!("a refused request must not emit"),
        );
        assert!(
            matches!(result, Err(DecodeError::KvPoolExhausted)),
            "expected a typed refusal, got {result:?}"
        );
        for l in 0..decoder.layers.len() {
            assert_eq!(
                config.store.free_blocks(l),
                1,
                "layer {l} must keep every page after a refusal"
            );
        }
    }

    #[test]
    fn generate_succeeds_with_a_pool_that_has_enough_blocks() {
        let decoder = small_decoder(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 2)));
        let config = pool_config(pool.clone(), Duration::ZERO);

        let mut out = String::new();
        let (finish, _rows, _ids, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            None,
            None,
            |_, s| out.push_str(s),
        )
        .unwrap();
        assert_eq!(finish[0].0, FinishReason::Length);
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            2,
            "every acquired block must be released once the request finishes"
        );
    }

    /// Regression test for a real bug caught by live testing (not by
    /// any unit test): with a small `block_size`, a request whose
    /// prompt + max_tokens exceeds one block used to reserve only one
    /// block per layer at admission time, then panic deep inside
    /// `Decoder::forward_token` once decode outgrew that block and the
    /// pool had nothing left to grow into (`KvCache::push` returning
    /// `Err` where `forward_token` assumes it can't fail). Fixed by
    /// having `acquire_pooled_caches` reserve blocks for the whole
    /// worst-case sequence length up front. This test must not panic --
    /// it must either succeed cleanly or fail at admission with
    /// `KvPoolExhausted`, never partway through decode.
    #[test]
    fn generate_reserves_enough_blocks_up_front_for_a_sequence_spanning_multiple_blocks() {
        let decoder = small_decoder(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap(); // 2 tokens via ByteTokenizer
        let max_tokens = 10;
        let block_size = 2;
        // prompt (2) + max_tokens (10) = 12 positions -> 6 blocks/layer * 2 layers = 12 blocks.
        let pool = Arc::new(Mutex::new(KvBlockPool::new(block_size, 12)));
        let config = pool_config(pool.clone(), Duration::ZERO);

        let (finish, _rows, _ids, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(max_tokens),
            Some(&config),
            None,
            None,
            None,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(finish[0].0, FinishReason::Length);
        assert_eq!(pool.lock().unwrap().free_blocks(), 12);
    }

    /// One block short of the worst case means the pool can *never*
    /// serve this request, so the refusal is the immovable one -- a 400
    /// naming `device_memory_budget_exceeded`, not a 503 inviting a
    /// retry that an empty pool would refuse identically.
    ///
    /// Confirmed to FAIL when `pool_immovable_refusal` is removed from
    /// `generate` (the request falls through to the acquisition and
    /// comes back as the retryable `KvPoolExhausted`), and when its
    /// `needed <= total_blocks` comparison is loosened to `<`.
    #[test]
    fn generate_fails_at_admission_not_mid_decode_when_the_pool_cannot_cover_the_worst_case() {
        let decoder = small_decoder(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let max_tokens = 10;
        let block_size = 2;
        // One block short of the 12 the worst case (see the test
        // above) actually needs.
        let pool = Arc::new(Mutex::new(KvBlockPool::new(block_size, 11)));
        let config = pool_config(pool.clone(), Duration::ZERO);

        let result = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(max_tokens),
            Some(&config),
            None,
            None,
            None,
            |_, _| {},
        );
        let err = result.expect_err("11 blocks cannot cover a 12-block worst case");
        assert!(
            matches!(
                &err,
                DecodeError::KvBudgetExceeded { binding, positions, .. }
                    if *binding == frink_models::Ceiling::DeviceMemory.code()
                        && *positions == 12
            ),
            "expected an immovable device-memory refusal, got {err:?}"
        );
        assert_eq!(
            err.retry_after_secs(),
            None,
            "no wait frees blocks that do not exist"
        );
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            11,
            "a rejected request must leave the pool exactly as it found it"
        );
    }

    /// A refusal's two halves must describe ONE reservation.
    ///
    /// `pool_immovable_refusal` reports `estimated_bytes` from
    /// `KvShape::kv_bytes_for_tokens` beside a block count of
    /// `max_seq_len.div_ceil(block_size) * n_layers`. The block count
    /// has never had a window in it -- `KvCache::with_pool` reserves
    /// `max_seq_len` positions for every layer -- while the byte figure
    /// used to discount the sliding layers. For gpt-oss at 8192
    /// positions the message therefore said ~390 MiB next to a ~768 MiB
    /// reservation the same function had just rejected (#33).
    ///
    /// Two decoders, same shape, one with a window: the refusal must
    /// price them identically, because the pool reserves for them
    /// identically.
    #[test]
    fn the_pool_refusals_byte_figure_does_not_discount_a_window_the_pool_still_reserves() {
        let full = small_decoder();
        let mut alternating_cfg = test_dense_fixture();
        alternating_cfg.sliding_window = Some(4);
        alternating_cfg.swa_layers = frink_models::swa_layers::SwaLayers::period(2, false);
        let alternating = Decoder::new_random_small(alternating_cfg, 2, 256);
        assert_eq!(
            alternating.config.uniform_sliding_window(),
            None,
            "an alternating model is the case no store may recycle"
        );

        // A pool that cannot cover one block per layer, so both models
        // reach the immovable refusal.
        let pool = Arc::new(Mutex::new(KvBlockPool::new(8, 1)));
        let config = pool_config(pool, Duration::ZERO);
        let max_seq_len = 64;

        let bytes_of =
            |decoder: &Decoder| match pool_immovable_refusal(decoder, &config, max_seq_len) {
                Some(DecodeError::KvBudgetExceeded {
                    estimated_bytes, ..
                }) => estimated_bytes,
                other => panic!("expected an immovable refusal, got {other:?}"),
            };

        let windowed_bytes = bytes_of(&alternating);
        assert_eq!(
            windowed_bytes,
            bytes_of(&full),
            "the window discounts nothing the pool reserves"
        );
        // And that figure is the whole reservation the refusal names:
        // every layer, every position.
        assert_eq!(
            windowed_bytes,
            KvShape::from_config(&full.config, KvElem::F32).per_token_kv_bytes()
                * max_seq_len as u64
        );
    }

    /// A one-block pool cannot hold a two-layer model's caches under any
    /// schedule, so this is the immovable refusal too.
    #[test]
    fn generate_rejects_the_request_without_leaking_blocks_when_the_pool_is_too_small() {
        let decoder = small_decoder(); // 2 layers -> needs 2 blocks, one per layer's cache
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 1)));
        let config = pool_config(pool.clone(), Duration::ZERO);

        let result = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            None,
            None,
            |_, _| {},
        );
        let err = result.expect_err("one block cannot hold two layers' caches");
        assert!(
            matches!(
                &err,
                DecodeError::KvBudgetExceeded { binding, .. }
                    if *binding == frink_models::Ceiling::DeviceMemory.code()
            ),
            "expected an immovable device-memory refusal, got {err:?}"
        );
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            1,
            "a rejected request must leave the pool exactly as it found it"
        );
    }

    #[test]
    fn generate_releases_blocks_so_back_to_back_requests_do_not_starve_the_pool() {
        let decoder = small_decoder(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        // Just enough for one request's caches at a time -- a second,
        // concurrent request would be rejected, but a *sequential*
        // second request must succeed once the first has returned its
        // blocks.
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 2)));
        let config = pool_config(pool.clone(), Duration::ZERO);

        for _ in 0..3 {
            let (finish, _rows, _ids, _usage) = generate(
                &decoder,
                &ServerTokenizer::Byte,
                &StopTokens::default(),
                None,
                &prompt,
                &greedy_params(5),
                Some(&config),
                None,
                None,
                None,
                |_, _| {},
            )
            .unwrap();
            assert_eq!(finish[0].0, FinishReason::Length);
        }
        assert_eq!(pool.lock().unwrap().free_blocks(), 2);
    }

    /// `queue_wait = 0` must reject on the first failed attempt rather
    /// than retry.
    ///
    /// The pool here is big enough for the request and *momentarily*
    /// held by someone else, which is the only situation in which
    /// `KvPoolExhausted` is an honest answer: a pool permanently too
    /// small is refused earlier, by `pool_immovable_refusal`, and would
    /// make the timing assertion below vacuous.
    #[test]
    fn generate_with_zero_queue_wait_rejects_immediately() {
        let decoder = small_decoder(); // 2 layers -> needs 2 blocks
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 2)));

        // Another in-flight request holding both blocks for longer than
        // this request is willing to wait for them.
        let holder_pool = pool.clone();
        let holder = std::thread::spawn(move || {
            let mut held = KvCache::with_pool(1, 1, holder_pool, 0).unwrap();
            held.push(&[0.0], &[0.0]).unwrap(); // crosses into the second block
            std::thread::sleep(Duration::from_millis(200));
            drop(held);
        });
        std::thread::sleep(Duration::from_millis(15));

        let config = pool_config(pool, Duration::ZERO);
        let started = Instant::now();
        let result = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            None,
            None,
            |_, _| {},
        );
        assert!(
            matches!(result, Err(DecodeError::KvPoolExhausted)),
            "a pool that could serve this request once its holder lets go is momentary \
             exhaustion, which is retryable"
        );
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "queue_wait=0 must reject on the first attempt, not retry: took {:?}",
            started.elapsed()
        );
        holder.join().unwrap();
    }

    /// The private path's context ceiling, and the property that makes
    /// it worth having: the refusal happens before any KV is acquired
    /// and before a single forward pass runs.
    ///
    /// The refusal is for a prompt that does not fit BY ITSELF. That is
    /// the only case with no output budget left to clamp to; see the
    /// test below for the other outcome.
    ///
    /// Confirmed to FAIL when the `prompt_refusal` block is removed
    /// from `generate` -- the request then runs to completion and
    /// returns `Ok`.
    #[test]
    fn a_prompt_past_the_context_ceiling_is_refused_before_any_kv_is_acquired() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3, 4, 5]).unwrap();
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 64)));
        let config = pool_config(pool.clone(), Duration::ZERO);
        let shape = KvShape::from_config(&decoder.config, KvElem::F32);
        // The prompt alone is 5 tokens against a ceiling of 4, so no
        // output budget exists that would make it servable.
        let ceiling = ContextCeiling::new(Some(4), shape);

        let err = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            None,
            Some(&ceiling),
            |_, _| panic!("no token may be emitted by a refused request"),
        )
        .expect_err("a 5-token prompt must not be admitted under a 4-position ceiling");
        match &err {
            DecodeError::KvBudgetExceeded {
                binding,
                positions,
                positions_limit,
                detail,
                ..
            } => {
                assert_eq!(*binding, frink_models::Ceiling::ContextLength.code());
                assert_eq!(*positions, 5);
                assert_eq!(*positions_limit, 4);
                // Load-bearing wording: Claude Code and OpenClaw match
                // on this text to recognise a blown context window,
                // because the Anthropic wire carries no error code.
                assert_eq!(detail, "prompt is too long: 5 tokens > 4 maximum");
            }
            other => panic!("expected a context-length refusal, got {other:?}"),
        }
        assert_eq!(err.retry_after_secs(), None, "a 400, not a retryable 503");
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            64,
            "the refusal must land before any block is taken"
        );
        assert_eq!(ceiling.refused(), 1);
    }

    /// The other outcome, and the one this test used to get wrong: a
    /// prompt that FITS is served with `max_tokens` clamped to what
    /// remains, not refused.
    ///
    /// This test previously asserted the refusal. Refusing here turns a
    /// servable request into a 400 over a `max_tokens` the caller very
    /// likely never set -- which is exactly what a large default output
    /// budget would make happen on every long prompt.
    #[test]
    fn a_prompt_that_fits_is_served_with_its_budget_clamped_rather_than_refused() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let shape = KvShape::from_config(&decoder.config, KvElem::F32);
        // 2 prompt tokens under a 4-position ceiling leaves room for 2,
        // and the request asks for 5.
        let ceiling = ContextCeiling::new(Some(4), shape);

        let mut emitted = 0usize;
        let (finish, _rows, _ids, usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            None,
            None,
            None,
            Some(&ceiling),
            |_, _| emitted += 1,
        )
        .expect("a prompt that fits must be served");
        assert_eq!(usage.completion_tokens, 2, "clamped to the room left");
        assert_eq!(finish[0].0, FinishReason::Length);
        assert_eq!(ceiling.refused(), 0, "a clamp is not a refusal");
        assert!(emitted > 0);
    }

    /// A request that fits the ceiling is untouched by it: the same
    /// request that runs without a ceiling runs with one.
    ///
    /// Without this, a ceiling that refused everything would still pass
    /// the test above.
    #[test]
    fn a_request_inside_the_ceiling_is_admitted_unchanged() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let shape = KvShape::from_config(&decoder.config, KvElem::F32);
        let ceiling = ContextCeiling::new(Some(7), shape);

        let mut with = String::new();
        let (finish, _rows, _ids, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            None,
            None,
            None,
            Some(&ceiling),
            |_, s| with.push_str(s),
        )
        .expect("7 positions fits a 7-position ceiling exactly");
        assert_eq!(finish[0].0, FinishReason::Length);

        let mut without = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            None,
            None,
            None,
            None,
            |_, s| without.push_str(s),
        )
        .unwrap();
        assert_eq!(with, without, "an unbinding ceiling must change nothing");
        assert_eq!(ceiling.refused(), 0);
    }

    #[test]
    fn generate_with_a_queue_wait_succeeds_once_another_holder_releases_its_blocks() {
        let decoder = small_decoder(); // 2 layers, needs 2 blocks
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 2)));

        // Hold both blocks on another thread for a short while, then
        // release them -- simulating another in-flight request that's
        // about to finish.
        let holder_pool = pool.clone();
        let holder = std::thread::spawn(move || {
            let mut held = KvCache::with_pool(1, 1, holder_pool.clone(), 0).unwrap();
            held.push(&[0.0], &[0.0]).unwrap(); // crosses into needing the second block
            std::thread::sleep(Duration::from_millis(80));
            drop(held); // returns both blocks to the pool
        });
        // Give the holder a moment to actually acquire before we try.
        std::thread::sleep(Duration::from_millis(15));

        let config = pool_config(pool.clone(), Duration::from_millis(500));
        let (finish, _rows, _ids, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            &StopTokens::default(),
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            None,
            None,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(
            finish[0].0,
            FinishReason::Length,
            "a sufficiently long queue_wait must let the request succeed once the holder releases"
        );
        holder.join().unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 2);
    }

    /// `ignore_eos` is what makes a serving benchmark's requests do the
    /// same amount of work as each other. Without it they finish at
    /// different lengths and the slowest percentile is whichever
    /// request happened to be asked for the most tokens -- a fact about
    /// the prompts, reported as a fact about the server.
    #[test]
    fn ignore_eos_runs_a_request_out_to_its_full_budget() {
        let render = |id: usize| char::from(b'a' + id as u8).to_string();
        // The model tries to end its turn on its third token.
        let script = [0usize, 1, 7, 2, 3, 4];
        let eos = StopTokens::from_eos(Some(7));

        let stops_early =
            run_scripted_with_stops(&script, render, &scripted_params(6), eos.clone());
        assert_eq!(stops_early.0, FinishReason::Stop);
        assert_eq!(stops_early.1.len(), 2, "the model ended its own turn");

        let runs_on = run_scripted_with_stops(
            &script,
            render,
            &GenerationParams {
                ignore_eos: true,
                ..scripted_params(6)
            },
            eos,
        );
        assert_eq!(runs_on.0, FinishReason::Length);
        assert_eq!(
            runs_on.1.len(),
            6,
            "exactly the budget, which is the whole point"
        );
    }

    /// `ignore_eos` suppresses the MODEL's set and only that. A caller
    /// asking to run past the model's opinion about length is not a
    /// caller withdrawing their own fence, and a benchmark that could
    /// not be stopped by its own sentinel would be a footgun rather
    /// than a knob.
    #[test]
    fn ignore_eos_does_not_withdraw_the_callers_own_stop() {
        let render = |id: usize| char::from(b'a' + id as u8).to_string();
        let script = [0usize, 1, 2, 3, 4, 5];

        let (finish, ids, _) = run_scripted_with_stops(
            &script,
            render,
            &GenerationParams {
                ignore_eos: true,
                stop_token_ids: vec![2],
                ..scripted_params(6)
            },
            StopTokens::from_eos(Some(7)),
        );
        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(ids.len(), 2, "the caller's stop token still ends it");
    }
}
