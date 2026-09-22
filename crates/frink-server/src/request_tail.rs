//! Everything a generation does AFTER its last token: the usage block,
//! and the three places its KV may be published for reuse.
//!
//! Lifted out of `generate::generate` unchanged. Two reasons, and the
//! second is the one that matters:
//!
//! 1. `generate.rs` is 4784 lines, and this repo's most expensive
//!    lesson is that a change which would grow a file past roughly a
//!    thousand splits the file first. The reason is written down: the
//!    same decode layer was once spelled out eleven times across two
//!    big files and lost eight model features one at a time, each
//!    silently.
//!
//! 2. **It is exactly the part that must run ONCE per request when a
//!    request grows several completions** (`docs/plans/
//!    several-completions-per-request.md`). `prompt_tokens` is counted
//!    once because the prompt was prefilled once; the prefix cache
//!    stores one continuation and cannot represent `k`; the radix
//!    publish names one sequence. Inside a long function those are
//!    three `if`s a reader has to notice. As a module they are a
//!    boundary: what is in here happens per REQUEST, what is outside
//!    happens per CHOICE.
//!
//! Nothing here decides anything new. Every comment that explained a
//! subtlety came with it, because the subtleties did not move.

use std::sync::Mutex;

use frink_api::usage::Usage;
use frink_models::Decoder;

use crate::generate::{GenerationParams, Kv};
use crate::ServerTokenizer;
use frink_models::PrefixCache;

/// What the request produced, and everything the tail needs to price
/// and publish it.
///
/// A struct rather than nineteen arguments: the list is long because
/// the tail genuinely touches that much, and a caller that has to name
/// each field cannot pass two of them in the wrong order.
pub(crate) struct RequestTail<'a> {
    pub(crate) decoder: &'a Decoder,
    pub(crate) tokenizer: &'a ServerTokenizer,
    pub(crate) params: &'a GenerationParams,
    /// The rendered prompt, read only to ask whether it already opened
    /// a reasoning block.
    pub(crate) prompt: &'a str,
    /// The prompt's ids. CONSUMED: the prefix-cache store extends them
    /// with the generated ids and keeps the whole sequence.
    pub(crate) tokens: Vec<usize>,
    /// CHOICE 0's ids. The prefix cache stores one continuation and
    /// the reasoning split describes one answer, so both read this.
    pub(crate) generated_ids: Vec<usize>,
    /// Tokens generated across EVERY choice. Separate from
    /// `generated_ids.len()` because a request may have produced `n` of
    /// them, and the bill is the sum while the stored continuation is
    /// choice 0 (`docs/plans/several-completions-per-request.md`).
    pub(crate) completion_tokens: usize,
    /// The prediction for whatever would come next, which is what a
    /// later prefix restore needs alongside the rows.
    pub(crate) logits: Vec<f32>,
    pub(crate) kv: Kv,
    pub(crate) prompt_tokens: usize,
    pub(crate) vocab_size: usize,
    pub(crate) prefill_secs: f64,
    pub(crate) decode_secs: f64,
    pub(crate) prefill_start: std::time::Instant,
    pub(crate) first_token_at: Option<std::time::Instant>,
    /// Prompt positions a contiguous prefix restore saved.
    pub(crate) cached_tokens: Option<usize>,
    /// `(forwards, accepted, drafted)` from the speculative loop.
    pub(crate) speculation: (usize, usize, usize),
    pub(crate) kv_pool_configured: bool,
    pub(crate) radix_enabled: bool,
    pub(crate) prefix_cache: Option<&'a Mutex<PrefixCache>>,
}

impl RequestTail<'_> {
    /// Price the request and publish its KV, consuming both.
    pub(crate) fn finish(self) -> Usage {
        let RequestTail {
            // Only the Metal build reads it, to flush a device-resident
            // KV back to the host before the prefix cache stores it.
            #[cfg_attr(not(feature = "metal"), allow(unused_variables))]
            decoder,
            tokenizer,
            params,
            prompt,
            mut tokens,
            generated_ids,
            completion_tokens,
            logits,
            mut kv,
            prompt_tokens,
            vocab_size,
            prefill_secs,
            decode_secs,
            prefill_start,
            first_token_at,
            cached_tokens,
            speculation: (spec_forwards, spec_accepted, spec_drafted),
            kv_pool_configured,
            radix_enabled,
            prefix_cache,
        } = self;

        let mut usage =
            Usage::new(prompt_tokens, completion_tokens).with_timings(prefill_secs, decode_secs);
        // The producer this metric never had. Reported only when a round
        // actually ran, so the fields stay ABSENT for a request that did
        // not speculate rather than reporting a zero that reads as "the
        // drafter was useless".
        if spec_drafted > 0 {
            usage = usage.with_speculation(spec_forwards, spec_accepted, spec_drafted, Vec::new());
        }
        if let Some(at) = first_token_at {
            usage = usage.with_ttft(at.duration_since(prefill_start).as_secs_f64());
        }
        if let Some(cached) = cached_tokens {
            usage = usage.with_cached_tokens(cached);
        }
        // How much of the answer was thinking. `None` when this checkpoint
        // has no reasoning format, which leaves `completion_tokens_details`
        // ABSENT -- a zero there reads as "this model did not think" rather
        // than "nobody counted", and `/v1/responses` shipped exactly that
        // confusion (#120).
        //
        // Whether the prompt already opened the block is read off the
        // rendered prompt, not guessed from the family: a template asked to
        // think opens it itself, and then the first generated token is
        // already reasoning with no marker to find.
        if let Some(reasoning) = crate::reasoning_tokens::count(
            params.reasoning,
            params
                .reasoning
                .is_some_and(|f| f.prompt_opens_reasoning(prompt)),
            &generated_ids,
            |ids| tokenizer.decode(ids),
        ) {
            usage = usage.with_reasoning_tokens(reasoning);
        }
        // A paged request's reuse is the radix tree's, not the contiguous
        // prefix cache's, so it is counted here instead. Reported through
        // the same field because it means the same thing to a caller:
        // prompt positions this request did not have to compute.
        if let Kv::Paged(lease) = &kv {
            let adopted = lease.adopted_positions(lease.block_size());
            if radix_enabled {
                usage = usage.with_cached_tokens(adopted);
            }
        }

        // Store the full sequence this request actually processed (prompt
        // plus everything generated) so a future request sharing this
        // prefix -- the common multi-turn-chat case, where each turn's
        // prompt is the previous turn's full prompt+reply plus a little
        // more -- can skip recomputing it. `caches`/`logits` are exactly
        // in the right state for this: `logits` predicts whatever would
        // come after this sequence, and every token in
        // `tokens`/`generated_ids` has exactly one corresponding cache
        // entry. Skipped whenever a KV pool is configured.
        //
        // Before the contiguous store below, which consumes `kv`,
        // `tokens` and `generated_ids`. Independent of the pool, which a
        // paged request never has: it is the paged store that holds this
        // request's KV, and the tree that decides whether the next request
        // can reuse it. The two are mutually exclusive at startup, so only
        // one of these ever runs.
        if let Kv::Paged(lease) = &mut kv {
            // Publish under the sequence actually processed, prompt plus
            // everything generated, so the next request sharing that prefix
            // adopts the pages rather than recomputing them. The lease
            // keeps holding them either way; what changes is that the tree
            // now holds them too, so they outlive this request.
            let mut full = tokens.clone();
            full.extend(generated_ids.iter().copied());
            let block_size = lease.block_size();
            crate::generate::publish_to_radix(lease, &full, block_size);
        }

        if !kv_pool_configured {
            if let Some(pc) = prefix_cache {
                // Greedy Metal argmax returns a 1-element "logits" vec; that is
                // not a full pending distribution and must not be stored for
                // later (possibly non-greedy) prefix restores.
                if logits.len() == vocab_size {
                    #[cfg(feature = "metal")]
                    if let Some(caches) = kv.contiguous_mut() {
                        decoder.sync_metal_attn_kv_to_host(caches);
                    }
                    // `None` only for a paged request, which is refused
                    // alongside a prefix cache at startup; storing nothing
                    // is the honest answer either way.
                    if let Some(caches) = kv.into_contiguous() {
                        tokens.extend(generated_ids);
                        pc.lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .store(tokens, caches, logits);
                    }
                }
            }
        }
        usage
    }
}
