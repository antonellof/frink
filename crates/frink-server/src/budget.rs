//! One answer to "does this request fit", and one place that prices the
//! loaded model against the machine it landed on.
//!
//! Two things live here, and they are separate on purpose.
//!
//! # [`ContextCeiling`] -- the shared per-request ceiling
//!
//! Before this module the context ceiling existed only inside the
//! continuous batcher's `BlockBudget`, so a deployment running the
//! private `generate` path (the default: continuous batching is opt-in,
//! and is switched off entirely whenever a KV pool or prefix cache is
//! configured) had **no** context ceiling at all. An oversized request
//! there did not get a typed 400 naming what bound it -- it drained the
//! KV pool's admission wait and left with a 503 "retry shortly", which
//! is a lie about a request an idle server would refuse identically.
//!
//! `ContextCeiling` is the whole ceiling: the position limit, the
//! [`KvShape`] that prices a refusal in real bytes, and the rejection
//! counter. Both decode paths hold the *same* `Arc<ContextCeiling>`, so
//! the two cannot drift the way two copies of the same arithmetic
//! would -- the same discipline `crate::stop` applies to stop
//! sequences.
//!
//! # [`derive_limits`] -- pricing the model at load
//!
//! `frink` (the CLI) already prices a checkpoint before it loads it:
//! `--ctx-size auto`, a pre-load `KvBudget::check`, and a
//! `Ceiling::DeviceMemory` refusal. `frink-server` did not. It admitted
//! requests against ceilings an operator had *configured*
//! (`FRINK_CB_MAX_CONTEXT`, `FRINK_CB_KV_BLOCKS`) or, far more often,
//! against no ceiling at all, and discovered the real one as an OOM
//! kill. [`derive_limits`] closes that: weights plus `n_ctx *
//! per_token_kv` plus headroom against the device budget, every term
//! exact from the GGUF header, evaluated once at load.
//!
//! ## Where derivation deliberately stops
//!
//! - **An explicit setting always wins.** A derived number never
//!   overrides `FRINK_CB_MAX_CONTEXT` or `FRINK_CB_KV_BLOCKS`; it only
//!   ever fills an *absent* ceiling, where the status quo is "unbounded
//!   until the kernel intervenes".
//! - **An unknown budget derives nothing.** No probe, no ceiling: the
//!   same rule the CLI's `resolve_ctx_size` follows. Refusing on the
//!   strength of a number we do not have is worse than not refusing.
//! - **A model that does not fit at all derives nothing either**, and
//!   this is the one place where fail-closed is the *wrong* reading. A
//!   fit of zero tokens would mean a ceiling of zero positions, which is
//!   not a ceiling -- it is a refusal to serve the model, decided by an
//!   estimate. `frink_models::kv_budget`'s own module doc is explicit
//!   that `weights_bytes` is the checkpoint's byte count over *mmap'd*
//!   pages, an upper bound on residency rather than a measurement, and
//!   that "a model can exceed this budget and still run". Refusing one
//!   oversized request against a working estimate is a different claim
//!   from refusing every request because the estimate says the model
//!   should not have loaded -- and it *did* load. So this logs loudly,
//!   names `FRINK_DEVICE_BUDGET_BYTES`, and leaves the ceiling absent.

use std::sync::atomic::{AtomicU64, Ordering};

use frink_models::{Ceiling, ContextFit, KvBudget, KvElem, KvShape};

use crate::generate::DecodeError;

/// The per-request context ceiling, shared by every decode path.
///
/// `positions` throughout is prompt + `max_tokens`: the worst-case
/// sequence length the request will hold KV for, which is the number
/// both the KV pool and the block ledger reserve against.
#[derive(Debug)]
pub struct ContextCeiling {
    /// Positions any one request may ask for. `None` = no ceiling,
    /// which is what an unpriced deployment has.
    limit: Option<usize>,
    /// This model's real KV geometry, so a refusal states bytes rather
    /// than an opaque position count.
    shape: KvShape,
    /// Requests refused for exceeding `limit`. Counted here rather than
    /// at each call site so `/metrics` reports one number no matter
    /// which path did the refusing.
    refused: AtomicU64,
}

impl ContextCeiling {
    pub fn new(limit: Option<usize>, shape: KvShape) -> Self {
        ContextCeiling {
            limit,
            shape,
            refused: AtomicU64::new(0),
        }
    }

    /// KV bytes `positions` of context costs, every layer at every
    /// position.
    ///
    /// **A reporting number, not the admission number.** What decides
    /// whether a request is admitted is `limit`, in POSITIONS, and that
    /// limit came from [`KvBudget::max_context`], which does know about
    /// eviction (#61). This only fills the byte fields of a refusal
    /// whose stated reason is the position ceiling, so under
    /// `FRINK_KV_WINDOW` it over-states the bytes a windowed model
    /// would really have held -- in the direction that makes a refusal
    /// look more justified rather than less, and never in a decision.
    /// Giving this the residency too would mean threading it through
    /// eighteen constructor call sites to change a message.
    pub fn bytes_for(&self, positions: usize) -> u64 {
        self.shape.kv_bytes_for_tokens(positions)
    }

    pub fn refused(&self) -> u64 {
        self.refused.load(Ordering::Relaxed)
    }

    /// The per-request position ceiling, when this deployment has one.
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// The refusal for a prompt that does not fit the ceiling *by
    /// itself*, which is a different answer from one whose prompt plus
    /// budget does not.
    ///
    /// A prompt shorter than the ceiling is servable -- with a smaller
    /// output budget -- so it is clamped rather than refused (see
    /// `generate`). A prompt at or past the ceiling has no budget left
    /// to clamp to, so it is the one case that must fail.
    ///
    /// The wording is load-bearing and copied verbatim: Claude Code and
    /// OpenClaw match on this text to recognise a blown context window,
    /// because the Anthropic wire carries no error code for it.
    pub fn prompt_refusal(&self, prompt_tokens: usize) -> Option<DecodeError> {
        let limit = self.limit?;
        if prompt_tokens < limit {
            return None;
        }
        self.refused.fetch_add(1, Ordering::Relaxed);
        Some(DecodeError::KvBudgetExceeded {
            binding: Ceiling::ContextLength.code(),
            estimated_bytes: self.bytes_for(prompt_tokens),
            limit_bytes: self.bytes_for(limit),
            positions: prompt_tokens,
            positions_limit: limit,
            detail: format!("prompt is too long: {prompt_tokens} tokens > {limit} maximum"),
        })
    }

    /// The typed refusal for a `prompt + max_tokens` that cannot be
    /// represented at all, or `None` when the sum is a real number.
    ///
    /// The wrap was the whole bug in #36. `prompt_tokens +
    /// params.max_tokens` is a `usize` addition on a value read
    /// straight off the wire, so `max_tokens: 18446744073709551615`
    /// produced `prompt_tokens - 1`, which is BELOW any ceiling. The
    /// clamp that existed to bound this was therefore skipped by
    /// exactly the requests that needed it most, and the value went on
    /// to size a `Vec::with_capacity`.
    ///
    /// A wrapping sum is refused even when there is NO ceiling
    /// configured, because it is not a request any deployment could
    /// serve: no machine holds `usize::MAX` positions. That is the
    /// derived invariant doing the work rather than a chosen constant.
    ///
    /// `saturating_add` would have been the wrong tool. It silently
    /// clamps a nonsense value into a plausible one and serves it,
    /// which answers a question the caller did not ask.
    pub fn overflow_refusal(&self, prompt_tokens: usize, max_tokens: usize) -> Option<DecodeError> {
        match prompt_tokens.checked_add(max_tokens) {
            // It fits in a `usize`, so it is a real request. Whether it
            // fits the CEILING is the clamp's business, and answering
            // it here would count a clamp as a refusal.
            Some(_) => None,
            None => {
                self.refused.fetch_add(1, Ordering::Relaxed);
                Some(DecodeError::KvBudgetExceeded {
                    binding: Ceiling::ContextLength.code(),
                    estimated_bytes: 0,
                    limit_bytes: 0,
                    positions: usize::MAX,
                    positions_limit: self.limit.unwrap_or(usize::MAX),
                    detail: format!(
                        "prompt of {prompt_tokens} tokens plus max_tokens of {max_tokens} \
                         overflows the position counter, so this request cannot be served by \
                         any deployment. Send a max_tokens that fits the model's context"
                    ),
                })
            }
        }
    }

    /// The `max_tokens` this request actually gets: the caller's, or
    /// what the ceiling leaves after the prompt, or a refusal when the
    /// prompt has no room at all.
    ///
    /// ONE function for the decision, because the two decode paths used
    /// to answer it differently. `generate` clamped (the private loop,
    /// and the reason `DEFAULT_CHAT_MAX_TOKENS` is allowed to be 32k at
    /// all), while the continuous batcher called [`Self::refusal`] with
    /// `prompt + max_tokens` and rejected. Since the batcher became the
    /// default decode path, EVERY chat request that omitted `max_tokens`
    /// against a deployment with a ceiling under 32k was a 400 naming a
    /// number the caller never sent -- a server started `-c 16384`
    /// answered `hi` with `context_length_exceeded`, measured.
    ///
    /// The refusing cases stay refusals, and both are the request's own
    /// shape rather than the budget's: a prompt at or past the ceiling
    /// ([`Self::prompt_refusal`]) has nothing to clamp to, and a sum
    /// that does not fit a `usize` ([`Self::overflow_refusal`]) is not
    /// a request any deployment could serve.
    pub fn fit(&self, prompt_tokens: usize, max_tokens: usize) -> Result<usize, DecodeError> {
        if let Some(err) = self.prompt_refusal(prompt_tokens) {
            return Err(err);
        }
        // Before the clamp's own comparison, which is where the wrap
        // used to walk past the guard.
        if let Some(err) = self.overflow_refusal(prompt_tokens, max_tokens) {
            return Err(err);
        }
        match self.limit {
            // `prompt_refusal` has already established `prompt < limit`.
            Some(limit) if prompt_tokens + max_tokens > limit => Ok(limit - prompt_tokens),
            _ => Ok(max_tokens),
        }
    }

    /// The typed refusal for a request of `positions` positions, or
    /// `None` when it fits.
    ///
    /// Deliberately a 400 (`retry_after_secs() == None`): the ceiling is
    /// a property of the deployment, so an idle server refuses this
    /// request identically and "retry shortly" would be a lie.
    pub fn refusal(&self, positions: usize) -> Option<DecodeError> {
        let limit = self.limit?;
        if positions <= limit {
            return None;
        }
        self.refused.fetch_add(1, Ordering::Relaxed);
        Some(DecodeError::KvBudgetExceeded {
            binding: Ceiling::ContextLength.code(),
            estimated_bytes: self.bytes_for(positions),
            limit_bytes: self.bytes_for(limit),
            positions,
            positions_limit: limit,
            detail: format!(
                "request asks for {positions} token positions (prompt + max_tokens) but this \
                 deployment admits {limit} per request; shorten the prompt or lower max_tokens"
            ),
        })
    }
}

/// What [`derive_limits`] concluded, with the arithmetic that produced
/// it so an operator can check the division by hand.
#[derive(Debug, Clone, Copy)]
pub struct DerivedLimits {
    /// Positions any one request may hold: the largest context that
    /// fits this machine, capped at the model's own trained context.
    pub max_context: usize,
    /// Blocks for the whole-server KV ledger, at `block_size` positions
    /// each. Floored, never rounded up: a block the machine cannot hold
    /// is not a block to promise.
    pub kv_blocks: usize,
    /// The priced fit this came from; `Display`s as the full inequality.
    pub fit: ContextFit,
}

/// Turns a priced [`KvBudget`] into the two ceilings the server admits
/// on, or `None` when no honest ceiling follows.
///
/// `gguf_ctx` is the model's own trained context length -- the cap, so a
/// machine with room to spare derives the same number llama.cpp would
/// default to rather than a larger one the model was never trained for.
///
/// `None` means "derive nothing" and has exactly one cause: the priced
/// fit is zero tokens, i.e. weights plus headroom already fill the
/// budget. See this module's doc comment for why that is logged rather
/// than turned into a ceiling of zero.
pub fn derive_limits(
    budget: &KvBudget,
    gguf_ctx: usize,
    block_size: usize,
) -> Option<DerivedLimits> {
    assert!(block_size > 0, "kv block size must be positive");
    let cap = gguf_ctx.max(1);
    let fit = budget.max_context(cap, frink_models::CTX_AUTO_GRANULARITY);
    if fit.tokens == 0 {
        return None;
    }
    Some(DerivedLimits {
        max_context: fit.tokens,
        // Floor: the ledger's job is to hand out capacity that exists.
        // Rounding a partial block up would promise `block_size`
        // positions the fit says are not there.
        kv_blocks: fit.tokens / block_size,
        fit,
    })
}

/// Fills the ceilings an operator did not set from `derived`, and
/// leaves the ones they did set exactly alone.
///
/// Split out as a pure function because the precedence *is* the
/// contract: an operator who names a number has information this
/// arithmetic does not, so a derived value may only ever occupy an
/// empty slot. Returns what changed, so the caller logs the numbers it
/// actually adopted rather than the ones it computed.
pub fn apply_derived(
    config: &mut crate::serving::batch::BatcherConfig,
    derived: &DerivedLimits,
) -> Adopted {
    let mut adopted = Adopted::default();
    if config.max_context.is_none() {
        config.max_context = Some(derived.max_context);
        adopted.max_context = true;
    }
    // A zero-block ledger would refuse every request, which is the
    // "ceiling of zero" this module refuses to invent -- see the module
    // doc. Leave the ledger absent and let the context ceiling, which
    // is a real number here, do the refusing.
    if config.kv_blocks.is_none() && derived.kv_blocks > 0 {
        config.kv_blocks = Some(derived.kv_blocks);
        adopted.kv_blocks = true;
    }
    // The two ceilings must agree about one thing: the longest request
    // this deployment can ever admit. A per-request ceiling above what
    // the WHOLE ledger holds is a promise nothing can keep -- the
    // clamp lets `max_tokens` grow up to it and the block budget then
    // refuses the result as immovable, which is the same request
    // refused by the ceiling that did not advertise itself. Measured:
    // `serve -c 65536` on a machine whose derived ledger was 99 blocks
    // of 256 answered every default-budget chat request with
    // `device_memory_budget_exceeded`.
    //
    // Narrowing only, and only against a ledger: an operator who names
    // a smaller context keeps it, and a deployment with no ledger has
    // nothing to narrow against.
    if let (Some(blocks), Some(ctx)) = (config.kv_blocks, config.max_context) {
        let ledger_positions = blocks.saturating_mul(config.kv_block_size);
        if ledger_positions > 0 && ctx > ledger_positions {
            config.max_context = Some(ledger_positions);
            adopted.max_context_narrowed = Some(ledger_positions);
        }
    }
    adopted
}

/// Which ceilings [`apply_derived`] actually filled in.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Adopted {
    pub max_context: bool,
    pub kv_blocks: bool,
    /// The per-request ceiling was narrowed to what the whole KV ledger
    /// holds, and this is the number it became. `None` when the two
    /// already agreed. Reported separately from `max_context` because
    /// it can happen to a ceiling the OPERATOR set, which
    /// [`apply_derived`]'s precedence rule otherwise leaves alone.
    pub max_context_narrowed: Option<usize>,
}

/// Prices the GGUF at `path` against the machine, best effort.
///
/// Returns `None` -- and logs why -- for every reason the CLI treats as
/// "do not check": no device-budget probe, or a header this planner
/// cannot read (the MLA/Gemma4/GLM stacks carry their own hparams and
/// never build a `ModelConfig`). A model this server cannot price is
/// served exactly as it was before this module existed.
///
/// The `String` is the probe's own description, so the startup log says
/// where the budget number came from rather than asserting it.
pub fn price_gguf(
    path: &str,
    kv_elem: KvElem,
    concurrent_requests: usize,
) -> Option<(KvBudget, usize, String)> {
    use frink_models::residency_report::{ResidencyAssumptions, ResidencyReport};
    use frink_models::{BudgetBackend, DeviceBudget};

    let backend = if cfg!(feature = "metal") {
        BudgetBackend::Metal
    } else if cfg!(feature = "cuda") {
        BudgetBackend::Cuda
    } else {
        BudgetBackend::Cpu
    };
    let device = DeviceBudget::detect(backend);
    if device.is_unknown() {
        tracing::info!("{device}; serving with no derived context ceiling");
        return None;
    }
    let gguf_ctx = gguf_context_length(path)?;
    let assumptions = ResidencyAssumptions {
        context_tokens: gguf_ctx,
        concurrent_requests: concurrent_requests.max(1),
        kv_elem,
        // **The server prices the UNWINDOWED number even when
        // `FRINK_KV_WINDOW` is on, and that is deliberate.**
        //
        // Sliding-window eviction (#61 step 2) is a property of the
        // contiguous host `KvCache`, and it is not the only store this
        // server allocates. `FRINK_KV_POOL_BLOCKS` reserves
        // `max_seq_len` positions per layer from a shared pool up front
        // and never gives a block back mid-sequence, and
        // `FRINK_PAGED_KV_BLOCKS` runs the paged store, whose decode
        // arm never calls `evict_layer_kv` at all (#61 step 4). Either
        // holds the whole context however narrow the window is.
        //
        // This runs before either is chosen, so pricing the window here
        // would hand a pooled or paged deployment a ceiling smaller
        // than the memory it goes on to reserve -- which is exactly #33,
        // admitted and then OOM, in the direction that hurts. An
        // over-estimate only costs context.
        //
        // Lifting this needs the store the request will really use to
        // be known at this point, which is #61 steps 4 and 5.
        kv_window: frink_models::decoder::KvWindowPolicy::off(),
        ..ResidencyAssumptions::default()
    };
    match ResidencyReport::from_gguf(path, assumptions, device.usable_bytes) {
        Ok(report) => Some((report.kv_budget(), gguf_ctx, device.to_string())),
        Err(e) => {
            tracing::info!(
                "KV budget not computed for this checkpoint ({e}); serving with no derived \
                 context ceiling"
            );
            None
        }
    }
}

/// The model's trained context length from its own header, or the same
/// 4096 fallback `frink run` uses when the key is absent.
fn gguf_context_length(path: &str) -> Option<usize> {
    let file = frink_gguf::ShardedGguf::open(path).ok()?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    Some(
        file.metadata_u64(&format!("{arch}.context_length"))
            .map(|v| v as usize)
            .unwrap_or(4096),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_models::KvLayout;

    /// A shape whose arithmetic is easy to do by hand: 2 layers, 1 kv
    /// head, head_dim 4, f32 -> 2 layers * 4 elems * 2 (K and V) * 4
    /// bytes = 64 bytes per token.
    fn shape() -> KvShape {
        KvShape {
            n_layers: 2,
            kv_layers: 2,
            layout: KvLayout::Gqa {
                n_kv_heads: 1,
                head_dim: 4,
                v_head_dim: 4,
            },
            elem: KvElem::F32,
        }
    }

    fn budget(device_bytes: u64, weights_bytes: u64) -> KvBudget {
        KvBudget {
            weights_bytes,
            activation_headroom_bytes: 0,
            device_budget_bytes: device_bytes,
            shape: shape(),
            residency: frink_models::KvResidency::keeps_everything(shape().n_layers),
            concurrent_requests: 1,
        }
    }

    #[test]
    fn per_token_bytes_are_what_the_hand_computation_says() {
        assert_eq!(shape().per_token_kv_bytes(), 64);
    }

    /// The whole point of the ceiling: a request past it is refused
    /// with the *context* code, priced in real bytes, and is not a
    /// retryable error.
    ///
    /// Confirmed to FAIL when `refusal` returns `None` unconditionally,
    /// and when the `positions <= limit` comparison is flipped.
    #[test]
    fn a_request_past_the_ceiling_is_refused_in_bytes_and_is_not_retryable() {
        let ceiling = ContextCeiling::new(Some(100), shape());
        assert!(ceiling.refusal(100).is_none(), "exactly at the limit fits");
        let err = ceiling.refusal(101).expect("101 > 100 must be refused");
        match &err {
            DecodeError::KvBudgetExceeded {
                binding,
                estimated_bytes,
                limit_bytes,
                positions,
                positions_limit,
                ..
            } => {
                assert_eq!(*binding, Ceiling::ContextLength.code());
                assert_eq!(*estimated_bytes, 101 * 64);
                assert_eq!(*limit_bytes, 100 * 64);
                assert_eq!(*positions, 101);
                assert_eq!(*positions_limit, 100);
            }
            other => panic!("expected KvBudgetExceeded, got {other:?}"),
        }
        assert_eq!(
            err.retry_after_secs(),
            None,
            "an idle server refuses this identically, so 'retry shortly' would be a lie"
        );
        assert_eq!(ceiling.refused(), 1, "the refusal must be counted");
    }

    /// An absent ceiling is not a ceiling of zero. A server that could
    /// not price its model must serve exactly as it did before.
    #[test]
    fn no_ceiling_refuses_nothing() {
        let ceiling = ContextCeiling::new(None, shape());
        assert!(ceiling.refusal(usize::MAX / 2).is_none());
        assert_eq!(ceiling.refused(), 0);
    }

    /// 4096 bytes of KV room / 64 bytes per token = 64 tokens, which is
    /// under one `CTX_AUTO_GRANULARITY` step, so the fit reports the
    /// exact number rather than rounding it away to nothing.
    #[test]
    fn the_derived_context_is_the_room_left_after_weights_divided_by_the_per_token_cost() {
        let derived = derive_limits(&budget(8192, 4096), 100_000, 16)
            .expect("4096 bytes of KV room fits some context");
        assert_eq!(derived.max_context, 64);
        assert_eq!(derived.kv_blocks, 4, "64 positions / 16 per block");
    }

    /// The model's own trained context is the cap: a machine with room
    /// to spare must not derive a context the model was never trained
    /// for.
    ///
    /// Confirmed to FAIL when `derive_limits` passes `usize::MAX` as the
    /// cap instead of `gguf_ctx`.
    #[test]
    fn a_roomy_machine_is_still_capped_at_the_models_trained_context() {
        let derived = derive_limits(&budget(1 << 40, 0), 4096, 256)
            .expect("a terabyte of room fits the model's whole context");
        assert_eq!(derived.max_context, 4096);
        assert_eq!(derived.kv_blocks, 16);
    }

    /// A partial block is not a block. Rounding up here would promise
    /// `block_size` positions the priced fit says are not there.
    ///
    /// Confirmed to FAIL when `kv_blocks` uses `div_ceil`.
    #[test]
    fn a_partial_block_is_floored_away_rather_than_promised() {
        // 6400 bytes of KV room -> 100 tokens, block size 64 -> one
        // whole block and 36 positions left over.
        let derived = derive_limits(&budget(6400, 0), 100_000, 64).expect("100 tokens fit");
        assert_eq!(derived.max_context, 100);
        assert_eq!(derived.kv_blocks, 1);
    }

    /// Weights alone fill the budget: no context fits. This derives
    /// *nothing* rather than a ceiling of zero -- see the module doc for
    /// why a refusal to serve is not the same claim as a refusal of one
    /// oversized request.
    #[test]
    fn a_model_that_leaves_no_room_derives_no_ceiling_at_all() {
        assert!(derive_limits(&budget(4096, 4096), 100_000, 16).is_none());
        assert!(derive_limits(&budget(4096, 8192), 100_000, 16).is_none());
    }

    fn derived(max_context: usize, kv_blocks: usize) -> DerivedLimits {
        DerivedLimits {
            max_context,
            kv_blocks,
            fit: budget(1 << 30, 0).max_context(max_context.max(1), 1),
        }
    }

    fn cfg(
        max_context: Option<usize>,
        kv_blocks: Option<usize>,
        block: usize,
    ) -> crate::serving::batch::BatcherConfig {
        crate::serving::batch::BatcherConfig {
            max_context,
            kv_blocks,
            kv_block_size: block,
            ..Default::default()
        }
    }

    /// A per-request ceiling above what the whole ledger holds is a
    /// promise nothing can keep: `fit` would clamp `max_tokens` up to
    /// it and the block budget would then refuse the result. `serve -c
    /// 65536` against a 99-block ledger of 256 did exactly that, and
    /// every default-budget chat request got
    /// `device_memory_budget_exceeded`.
    #[test]
    fn a_ceiling_above_the_whole_ledger_is_narrowed_to_it() {
        let mut config = cfg(Some(65_536), Some(99), 256);
        let adopted = apply_derived(&mut config, &derived(4096, 16));
        assert_eq!(config.max_context, Some(99 * 256));
        assert_eq!(adopted.max_context_narrowed, Some(99 * 256));
        assert!(
            !adopted.max_context,
            "the operator's number was narrowed, not filled in"
        );
    }

    /// Narrowing only. An operator asking for less than the ledger
    /// holds keeps their number, and a deployment with no ledger has
    /// nothing to narrow against.
    #[test]
    fn a_ceiling_the_ledger_can_hold_is_left_alone() {
        let mut config = cfg(Some(4096), Some(99), 256);
        let adopted = apply_derived(&mut config, &derived(4096, 16));
        assert_eq!(config.max_context, Some(4096));
        assert_eq!(adopted.max_context_narrowed, None);

        let mut no_ledger = cfg(Some(65_536), None, 256);
        let adopted = apply_derived(&mut no_ledger, &derived(100, 0));
        assert_eq!(no_ledger.max_context, Some(65_536));
        assert_eq!(adopted.max_context_narrowed, None);
    }

    /// The precedence contract: an operator who set a number keeps it.
    ///
    /// Confirmed to FAIL when `apply_derived` assigns unconditionally
    /// instead of only into an empty slot.
    #[test]
    fn a_configured_ceiling_is_never_overridden_by_a_derived_one() {
        let mut config = crate::serving::batch::BatcherConfig {
            max_context: Some(999),
            kv_blocks: Some(7),
            ..Default::default()
        };
        let adopted = apply_derived(&mut config, &derived(4096, 16));
        assert_eq!(config.max_context, Some(999));
        assert_eq!(config.kv_blocks, Some(7));
        assert_eq!(adopted, Adopted::default(), "nothing was adopted");
    }

    /// An absent ceiling is the slot derivation exists to fill, and the
    /// two slots are independent: setting one by hand must not suppress
    /// the other's derivation.
    #[test]
    fn an_absent_ceiling_is_filled_and_the_two_slots_are_independent() {
        let mut both = crate::serving::batch::BatcherConfig {
            max_context: None,
            kv_blocks: None,
            ..Default::default()
        };
        let adopted = apply_derived(&mut both, &derived(4096, 16));
        assert_eq!(both.max_context, Some(4096));
        assert_eq!(both.kv_blocks, Some(16));
        assert_eq!(
            adopted,
            Adopted {
                max_context: true,
                kv_blocks: true,
                max_context_narrowed: None
            }
        );

        let mut half = crate::serving::batch::BatcherConfig {
            max_context: Some(512),
            kv_blocks: None,
            ..Default::default()
        };
        let adopted = apply_derived(&mut half, &derived(4096, 16));
        assert_eq!(half.max_context, Some(512), "the set one survives");
        assert_eq!(half.kv_blocks, Some(16), "the unset one is still derived");
        assert_eq!(
            adopted,
            Adopted {
                max_context: false,
                kv_blocks: true,
                max_context_narrowed: None
            }
        );
    }

    /// A fit that yields no whole block leaves the ledger absent rather
    /// than installing a zero-block budget that refuses everything --
    /// the context ceiling, which is a real number here, does the
    /// refusing instead.
    ///
    /// Confirmed to FAIL when the `derived.kv_blocks > 0` guard is
    /// dropped: `kv_blocks` becomes `Some(0)`.
    #[test]
    fn a_fit_smaller_than_one_block_leaves_the_ledger_absent() {
        let mut config = crate::serving::batch::BatcherConfig {
            max_context: None,
            kv_blocks: None,
            ..Default::default()
        };
        apply_derived(&mut config, &derived(100, 0));
        assert_eq!(config.max_context, Some(100));
        assert_eq!(config.kv_blocks, None);
    }

    /// A sliding-window model derives a ceiling from memory like any
    /// other model. It used to derive the model's whole trained context
    /// however small the budget was: `KvShape` subtracted the sliding
    /// layers out of the divisor, a fully-windowed model divided by zero
    /// bytes per token, and no KV store ever gave those bytes back
    /// (#33). The server then admitted a context it had to allocate in
    /// full.
    ///
    /// The window is taken from a real `ModelConfig` rather than
    /// hand-written into the shape, because the shape has no window
    /// field to write it into any more -- which is the fix.
    #[test]
    fn a_sliding_model_derives_its_ceiling_from_memory_like_any_other() {
        let mut cfg = frink_models::config::test_dense_fixture();
        cfg.n_layers = 2;
        cfg.n_kv_heads = 1;
        cfg.head_dim = 4;
        cfg.sliding_window = Some(8);
        cfg.swa_layers = frink_models::swa_layers::SwaLayers::All; // every layer slides: the old zero divisor
        let windowed = KvShape::from_config(&cfg, KvElem::F32);
        assert_eq!(windowed, shape(), "64 bytes/token, window or no window");

        // Room for 1024 tokens against a model that would like 8192.
        let b = KvBudget {
            shape: windowed,
            ..budget(64 * 1024, 0)
        };
        let derived = derive_limits(&b, 8192, 256).expect("a fit of 1024 tokens is a real fit");
        assert_eq!(derived.max_context, 1024);
    }

    /// **The wrap that skipped the clamp.** `max_tokens` arrives as a
    /// `usize` straight off an HTTP body, and `prompt + max_tokens` was
    /// an unchecked addition. At `usize::MAX` it wrapped to
    /// `prompt - 1`, which is BELOW any ceiling, so the guard that
    /// existed to bound the value was skipped by exactly the requests
    /// that needed bounding. The value then went on to size a
    /// `Vec::with_capacity`.
    #[test]
    fn a_max_tokens_that_wraps_the_position_sum_is_refused() {
        let ceiling = ContextCeiling::new(Some(100), shape());
        let err = ceiling
            .overflow_refusal(10, usize::MAX)
            .expect("usize::MAX must be refused");
        assert!(
            format!("{err}").contains("max_tokens"),
            "the refusal must name the field the caller sent"
        );

        // This is the comparison that used to let it through: below the
        // limit, and not a real request.
        assert!(
            10usize.wrapping_add(usize::MAX) < 100,
            "the wrap this refusal exists to catch"
        );
    }

    /// A deployment with NO ceiling is the other half of the hole. It
    /// cannot clamp, so it must still refuse a sum that cannot exist,
    /// rather than treating "no ceiling" as "unbounded".
    #[test]
    fn a_wrapping_sum_is_refused_even_with_no_ceiling_configured() {
        let ceiling = ContextCeiling::new(None, shape());
        assert!(
            ceiling.overflow_refusal(10, usize::MAX).is_some(),
            "no ceiling is not a licence to accept a request no machine could serve"
        );
        assert!(ceiling.overflow_refusal(10, 100).is_none());
    }

    /// This guard refuses the IMPOSSIBLE, not the merely oversized.
    ///
    /// A request past the ceiling is servable and is clamped by the
    /// caller. Refusing it here would turn a servable long-prompt
    /// request into a 400 over a `max_tokens` the caller very likely
    /// never set, and would count a clamp as a refusal in the stats.
    #[test]
    fn an_ordinary_request_past_the_ceiling_is_left_for_the_clamp() {
        let ceiling = ContextCeiling::new(Some(100), shape());
        assert!(ceiling.overflow_refusal(10, 50).is_none(), "10 + 50 fits");
        assert!(
            ceiling.overflow_refusal(10, 200).is_none(),
            "over the ceiling but representable: the clamp handles it"
        );
        assert_eq!(ceiling.refused(), 0, "a clamp is not a refusal");
    }

    /// **A `max_tokens` that does NOT wrap still reached an overflow**,
    /// one layer down, while computing the byte count for the refusal
    /// message. `u64::MAX / 64` positions multiplied past `u64::MAX` in
    /// `KvShape::kv_bytes_for_tokens` and panicked the request thread.
    ///
    /// Sized past the lazy-allocation threshold on purpose: a test at
    /// `500_000_000` would pass whatever the code does, because a large
    /// `Vec::with_capacity` is reserved lazily on macOS and the loop
    /// fails on the first read. `u64::MAX / 64` fails deterministically.
    #[test]
    fn a_huge_but_representable_max_tokens_reports_bytes_instead_of_panicking() {
        let ceiling = ContextCeiling::new(Some(100), shape());
        // 10 + u64::MAX/64, which is what a 10-token prompt with
        // `max_tokens: u64::MAX / 64` really produces. Exactly
        // `u64::MAX / 64` lands just UNDER the overflow and would make
        // this test pass whatever the code does, which is the same trap
        // as sizing an allocation test below the lazy-reserve
        // threshold: the number has to be chosen to fail.
        let positions = 10 + (u64::MAX / 64) as usize;

        // The byte count saturates rather than overflowing. It is a
        // reporting number, and "astronomically large" is all the
        // message needs to convey.
        // Asserted as "astronomically large" rather than as an exact
        // number: whether it saturates or merely lands just under
        // `u64::MAX` depends on the shape, and the property that matters
        // is that it is produced at all.
        assert!(ceiling.bytes_for(positions) > u64::MAX / 2);

        // And the refusal it feeds is produced, not panicked through.
        let err = ceiling
            .refusal(positions)
            .expect("far past a 100-position ceiling");
        assert!(format!("{err}").contains("100"));
    }
}
