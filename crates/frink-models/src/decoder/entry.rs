//! The public forward entry points of [`Decoder`], and the one place
//! each of them enters the CPU worker pool.
//!
//! # Why these ten functions live in their own file
//!
//! Every one of them is a wrapper of the same shape:
//! [`frink_core::par::on_workers`] around a body that lives in
//! `decoder.rs`. That shape is the fix for the cost measured in #27 and
//! #128, and it is worth exactly one file so that the rule is visible
//! rather than repeated across a six-thousand-line module: **a forward
//! pass enters the pool once, at its outermost boundary.**
//!
//! # What the wrapper buys
//!
//! `rayon::join` and the `par_iter` bridges cost very different things
//! depending on who calls them. From a rayon worker the caller runs one
//! half itself and waits on a spin latch; from any other thread the job
//! is injected and the caller blocks on a pthread condvar, contributing
//! no arithmetic while it sleeps. A decode step opens roughly five
//! parallel regions per layer, so a 30-layer model was paying ~150 of
//! the second kind per token.
//!
//! Sampled with `sample` on an M2 Pro over SmolLM2-135M Q8_0 `tg128`,
//! the driving thread held **74% of the token** inside `__psynch_cvwait`
//! under rayon's `LockLatch`, while the matvec kernel accounted for
//! about a tenth of the same window across every thread in the process.
//! Wrapping the step turns those ~150 cold entries into one, and
//! [`frink_core::par::cold_regions`] is how a test says so without a
//! stopwatch.
//!
//! # The invariant this file exists to hold
//!
//! `decoder.rs` declares no `pub fn forward_*` of its own; the bodies
//! there are private and named `*_on_worker`, or are the `*_inner`
//! helpers those call. `a_public_forward_may_not_be_declared_outside_
//! this_file` is the test that keeps it that way, because an entry point
//! added next door would silently reintroduce the per-region cost while
//! every other entry point still read as fixed.

use frink_core::cache::{KvCache, PagedKvCache, PagedStoreExhausted, SharedPagedKv};
use frink_core::par;

use super::{Decoder, MultiSeqKv};

impl Decoder {
    /// Runs one decode step for `token_id` at position `pos`, updating
    /// `kv_caches` (one per layer) in place, and returns the logits over
    /// the (test-scale) vocabulary.
    ///
    /// Enters the CPU worker pool once for the whole call; see the
    /// module docs for what that is worth.
    pub fn forward_token(
        &self,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<f32> {
        par::on_workers(move || self.forward_token_on_worker(token_id, pos, kv_caches))
    }

    /// Same computation as `forward_token`, but each layer's K/V cache
    /// is a `PagedKvCache` (block-table-indexed into a per-layer
    /// `PagedKvStore`) instead of a `KvCache`'s contiguous buffer --
    /// exercises the paged attention kernel in a real decode loop
    /// instead of only in isolation. `kv_caches`/`stores` are parallel
    /// per-layer arrays, mirroring `forward_token`'s `kv_caches: &mut
    /// [KvCache]`. Must produce bit-identical output to `forward_token`
    /// given stores sized so no layer ever exhausts its blocks --
    /// pinned by
    /// `forward_token_paged_matches_forward_token_bit_identical` and,
    /// per attention arm, by
    /// `every_paged_attention_arm_is_bit_identical_to_its_contiguous_twin`.
    ///
    /// This used to refuse gpt-oss outright, because the paged kernel
    /// had no attention-sink term and no sliding-window arm and would
    /// have answered differently from the contiguous path without
    /// saying so. It now mirrors all three arms of that dispatch, so
    /// the refusal is gone rather than merely relaxed.
    ///
    /// Enters the CPU worker pool once for the whole call; see the
    /// module docs for what that is worth.
    pub fn forward_token_paged(
        &self,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
    ) -> Result<Vec<f32>, PagedStoreExhausted> {
        par::on_workers(move || {
            self.forward_token_paged_on_worker(token_id, pos, kv_caches, stores)
        })
    }

    /// Processes multiple new positions in one call instead of calling
    /// `forward_token` once per position. `tokens[i]` is the token at
    /// absolute position `start_pos + i`; all positions attend
    /// causally (position `i` sees positions `0..=i` of this batch
    /// plus everything already in `kv_caches`, nothing later).
    ///
    /// The attention block's Q/K/V/O projections and the MoE router
    /// are computed as batched matmuls (`WeightMatrix::apply_batch`),
    /// which for quantized weights means each weight row is read from
    /// memory once and dotted against every position in the batch,
    /// not once per position -- see `apply_batch`'s doc comment for
    /// why that's a real memory-bandwidth saving, not just fewer
    /// function calls. The expert FFN stage is *not* batched: which
    /// expert(s) a position routes to is data-dependent per position,
    /// so positions routed to different experts can't share a single
    /// matmul the way the shared Q/K/V/router projections can. RoPE
    /// and attention itself (causal masking, softmax) are also
    /// per-position, since they're cheap relative to the matmuls and
    /// batching them would add complexity for little benefit.
    ///
    /// This is what makes prompt-lookup speculative decoding
    /// (`speculative` module) actually save work rather than just
    /// reshuffle it: verifying `k` draft tokens costs one batched call
    /// here, not `k` calls to `forward_token`.
    ///
    /// Thin wrapper over [`Self::forward_hidden_batch`] + `output_head`.
    ///
    /// Enters the CPU worker pool once for the whole call; see the
    /// module docs for what that is worth.
    pub fn forward_batch(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<Vec<f32>> {
        par::on_workers(move || {
            let hiddens = self.forward_hidden_batch(tokens, start_pos, kv_caches);
            if hiddens.is_empty() {
                return Vec::new();
            }
            let batch_size = hiddens.len();
            let flat: Vec<f32> = hiddens.into_iter().flatten().collect();
            self.logits_from_flat_hidden(flat, batch_size)
        })
    }

    /// [`Self::forward_batch`] that also hands back the final-layer
    /// hidden state for every position instead of dropping it.
    ///
    /// `forward_batch` computes these and throws them away; a
    /// hidden-state-conditioned drafter (EAGLE, MTP, dFlash) needs
    /// exactly the vector for the last *verified* position, so
    /// recomputing it would mean running the target model twice for
    /// something the first pass already had in hand. The extra cost
    /// here is one copy of `[batch x hidden]`, which is why
    /// `forward_batch` keeps its move-only path for the prefill case
    /// that does not want them.
    ///
    /// Returns `(logits_per_position, hidden_per_position)`, both
    /// indexed by position in `tokens`.
    ///
    /// Enters the CPU worker pool once for the whole call; see the
    /// module docs for what that is worth.
    pub fn forward_batch_with_hidden(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        par::on_workers(move || {
            let hiddens = self.forward_hidden_batch(tokens, start_pos, kv_caches);
            if hiddens.is_empty() {
                return (Vec::new(), Vec::new());
            }
            let batch_size = hiddens.len();
            let flat: Vec<f32> = hiddens.iter().flatten().copied().collect();
            (self.logits_from_flat_hidden(flat, batch_size), hiddens)
        })
    }

    /// [`Self::forward_batch`] for the common case where only the final
    /// position's logits are wanted: prefill a prompt, then sample the
    /// next token. Runs `output_head` on **one** row instead of all
    /// `batch_size` of them.
    ///
    /// The KV cache and every hidden state are identical either way —
    /// only the vocabulary projection is skipped, and only for rows
    /// whose logits the caller was going to drop. That projection is not
    /// a rounding error: it is `[batch x hidden] x [hidden x vocab]`,
    /// which for a large-vocabulary model with a small body is a large
    /// share of prefill. `V*H / (V*H + L*P_layer)` comes to 30% on
    /// Gemma-3-1B, 21% on Llama-3.2-1B and SmolLM2, 23% on Gemma-2-2B.
    /// llama.cpp does not do this work at all during `pp512` —
    /// `llama_batch_get_one` leaves `logits` unset, so `inp_out_ids`
    /// selects a single row.
    ///
    /// [`Self::forward_batch`] stays for the callers that genuinely need
    /// every row: speculative verification checks each draft position,
    /// and `/v1/embeddings` pools over all of them.
    ///
    /// Enters the CPU worker pool once for the whole call; see the
    /// module docs for what that is worth.
    pub fn forward_batch_last(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<f32> {
        par::on_workers(move || self.forward_batch_last_inner(tokens, start_pos, kv_caches, false))
    }

    /// [`Self::forward_batch_last`] for a caller that will READ the
    /// caches afterwards rather than only decode from them.
    ///
    /// A Metal prefill otherwise leaves K/V on the device and the host
    /// rows zero-filled, which is invisible to a caller that keeps
    /// decoding (the device buffers stay authoritative) and fatal to
    /// one that copies the rows somewhere else. Two callers do copy
    /// them: `forward_batch_last_paged`, into the page store, and
    /// `frink-server`'s prefix cache, into a snapshot a later request
    /// restores from. Both used to get zeros, and both answered fluent
    /// nonsense from a prompt the model never attended over.
    ///
    /// Costs one KV download per layer. Use [`Self::forward_batch_last`]
    /// when nothing will read the caches back.
    ///
    /// Enters the CPU worker pool once for the whole call; see the
    /// module docs for what that is worth.
    pub fn forward_batch_last_host_kv(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<f32> {
        par::on_workers(move || self.forward_batch_last_inner(tokens, start_pos, kv_caches, true))
    }

    /// [`Self::forward_batch_last`] over paged KV: the prefill twin of
    /// [`Self::forward_token_paged`].
    ///
    /// # Why this gathers instead of paging the kernel
    ///
    /// `forward_hidden_batch`'s fast arm hands `cache.k` / `cache.v` to
    /// `causal_gqa_attention_prefill_shared_kv_windowed`, which is Rayon
    /// over `[query-block x head]` against one flat KV buffer. That
    /// blocking is why CPU prefill is not the per-query path, and a
    /// block table cannot be handed to it as a slice.
    ///
    /// The alternative was a second blocked kernel that reads through
    /// the table. This file has just finished paying for what a second
    /// copy of a rule costs: the paged decode path silently lost the
    /// window arm, the sink term, the attention softcap, the embedding
    /// scale and the final logit softcap, one at a time, because it was
    /// a copy. A prefill kernel is a much larger surface to keep in
    /// step than any of those. So the pages are materialised, the ONE
    /// prefill implementation every other path uses runs against them,
    /// and the new rows go back.
    ///
    /// Bit-identity is therefore by construction rather than by
    /// agreement between two kernels: this calls the same function with
    /// the same values. What the tests pin is that the gather and the
    /// scatter are faithful, not that two implementations of attention
    /// happen to match.
    ///
    /// The cost is one KV-sized copy per layer per call, against the
    /// matmuls that dominate prefill. Decode is untouched: it still
    /// reads through the block table and copies nothing, which is where
    /// page sharing pays.
    ///
    /// # Failure is checked before anything is written
    ///
    /// Every layer's blocks are reserved up front, so a store too small
    /// for the batch refuses with `PagedStoreExhausted` having mutated
    /// no layer. A partial append would leave some layers longer than
    /// others, and no caller can recover from that.
    ///
    /// Enters the CPU worker pool once for the whole call; see the
    /// module docs for what that is worth.
    pub fn forward_batch_last_paged(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
    ) -> Result<Vec<f32>, PagedStoreExhausted> {
        par::on_workers(move || {
            Ok(self
                .forward_batch_paged_on_worker(tokens, start_pos, kv_caches, stores, false)?
                .pop()
                .unwrap_or_default())
        })
    }

    /// [`Self::forward_batch_last_paged`] returning one logit row per
    /// POSITION, for `prompt_logprobs`.
    ///
    /// The paged twin of [`Self::forward_batch`], and the same
    /// function underneath: the gather into contiguous scratch, the
    /// up-front reservation and the scatter back into the pages are
    /// identical, and only the lm_head projection differs. A request
    /// that did not ask to score its prompt takes the call above and
    /// pays for one projection rather than `tokens.len()` of them.
    pub fn forward_batch_paged(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
    ) -> Result<Vec<Vec<f32>>, PagedStoreExhausted> {
        par::on_workers(move || {
            self.forward_batch_paged_on_worker(tokens, start_pos, kv_caches, stores, true)
        })
    }

    /// Like [`Self::forward_batch`], but returns final RMS-normed hidden
    /// states (pre-`output_head`) — one `hidden_dim` vector per input
    /// token. Used by `/v1/embeddings` pooling (mean / last).
    ///
    /// Enters the CPU worker pool once for the whole call; see the
    /// module docs for what that is worth.
    pub fn forward_hidden_batch(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<Vec<f32>> {
        par::on_workers(move || {
            self.forward_hidden_batch_inner(tokens, start_pos, kv_caches, false)
        })
    }

    /// Continuous-batching primitive: one decode step across N
    /// independent *sequences*, each contributing exactly one new
    /// token at its own current position, sharing every layer's
    /// projection/router matmuls the same way `forward_batch` shares
    /// them across positions of a single sequence -- but each
    /// sequence keeps its own `KvCache`, independent `seq_len`, and
    /// independent position, so sequences admitted/evicted at
    /// different times can still share one batched matmul per step
    /// (this is what "continuous" batching means: the batch
    /// membership can change every step, unlike `forward_batch`'s
    /// fixed-size prompt-processing batch). `kv_caches[s][l]` is
    /// sequence `s`'s layer-`l` cache; `tokens[s]`/`positions[s]` is
    /// that sequence's next token and its position within its own
    /// history. Returns one logits vector per sequence, same order as
    /// `tokens`.
    ///
    /// Must produce bit-identical output to calling `forward_token`
    /// once per sequence with that sequence's own cache/position --
    /// batching independent sequences together is a scheduling detail,
    /// not a math change (no sequence's attention ever reads another
    /// sequence's cache).
    ///
    /// Enters the CPU worker pool once for the whole call; see the
    /// module docs for what that is worth.
    pub fn forward_multi_seq(
        &self,
        tokens: &[usize],
        positions: &[usize],
        kv_caches: &mut [Vec<KvCache>],
    ) -> Vec<Vec<f32>> {
        par::on_workers(move || {
            self.forward_multi_seq_kv(tokens, positions, &mut MultiSeqKv::Contiguous(kv_caches))
        })
    }

    /// [`Self::forward_multi_seq`] over either KV backing.
    ///
    /// One body for both: the batched projections are identical, and
    /// the per-sequence attention step is the only place the backing
    /// shows through.
    ///
    /// Enters the CPU worker pool once for the whole call; see the
    /// module docs for what that is worth.
    pub fn forward_multi_seq_kv(
        &self,
        tokens: &[usize],
        positions: &[usize],
        kv: &mut MultiSeqKv<'_>,
    ) -> Vec<Vec<f32>> {
        par::on_workers(move || self.forward_multi_seq_kv_on_worker(tokens, positions, kv))
    }
}

#[cfg(test)]
mod tests {
    /// Every public forward entry point belongs in this file, because
    /// this file is where the pool is entered. One declared next door
    /// would open a parallel region per matvec again, and nothing but
    /// a throughput measurement on a quiet host would notice.
    ///
    /// Sabotage: move any wrapper below back into `decoder.rs` with its
    /// `pub` intact and this goes red naming it.
    #[test]
    fn a_public_forward_may_not_be_declared_outside_this_file() {
        let body = include_str!("../decoder.rs");
        let stray: Vec<&str> = body
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("pub fn forward"))
            .collect();
        assert!(
            stray.is_empty(),
            "these forward entry points bypass the pool wrapper in entry.rs: {stray:?}"
        );
    }
}
