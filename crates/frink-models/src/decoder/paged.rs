//! The paged-KV bodies: one decode token and one prefill batch against
//! a block table.

use crate::skip_stream::SkipStream;
use frink_core::cache::{KvCache, PagedKvCache, PagedStoreExhausted, SharedPagedKv};

use crate::scalar_multipliers::residual_add;

use super::{Decoder, KvStep};

impl Decoder {
    /// The body of [`Self::forward_token_paged`], already running on a
    /// CPU-pool worker. See `entry.rs` for why the split exists.
    pub(super) fn forward_token_paged_on_worker(
        &self,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
    ) -> Result<Vec<f32>, PagedStoreExhausted> {
        assert_eq!(kv_caches.len(), self.config.n_layers);
        assert_eq!(stores.layer_count(), self.config.n_layers);
        // All layers advance or none do. Pushing per layer with `?` and
        // failing at layer 3 of 4 leaves layers 0..2 holding a position
        // the rest do not, and nothing downstream reports it: the next
        // step simply attends over a shorter history in the tail
        // layers. Reserving one position everywhere first turns that
        // into a clean refusal.
        //
        // The guards span the check AND the push for the same reason
        // the prefill path holds them: otherwise another request takes
        // the blocks in between.
        {
            let mut guards = stores.write_all();
            for (cache, store) in kv_caches.iter().zip(guards.iter()) {
                if cache.blocks_needed_for(store, 1) > store.free_block_count() {
                    return Err(PagedStoreExhausted);
                }
            }
            // Reserve by taking the blocks now, so the per-layer pushes
            // below cannot fail. `PagedKvCache::reserve` grows the block
            // table without advancing `seq_len`, leaving each push a
            // pure write into a block this sequence already owns.
            for (cache, store) in kv_caches.iter_mut().zip(guards.iter_mut()) {
                cache
                    .reserve(store, 1)
                    .expect("checked against free_block_count under this same guard");
            }
        }

        let mut hidden = self.embed_token(token_id, pos);
        let skip_rows = self.config.skip_stream.then(|| hidden.clone());
        let mut hrm = self.hrm_streams(&hidden);
        let residency = self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b));

        for (l, cache) in kv_caches.iter_mut().enumerate() {
            self.hrm_stack_input(hrm.as_ref(), l, &mut hidden);
            let layer = self.layer_for(l);
            // --- attention block ---
            let inputs = self.branch_inputs(layer, &hidden, 1);
            let normed = self.pre_norm_residual(&layer.attn.norm_weight, &mut hidden, 1);

            // The same body the contiguous path runs, with the paged
            // backing as its one parameter. It used to be a copy, and
            // the copy had silently dropped `attention_scale`,
            // `post_attn_norm`, `post_ffn_norm`, gpt-oss's `o_bias` and
            // `gpt_oss_ffn` -- five features that each produce a
            // plausible distribution rather than an error.
            let oai = self
                .gpt_oss
                .as_ref()
                .map(|g| &g.layers[self.physical_index(l)]);
            if let Some(projected) = self.attn_block(
                l,
                layer,
                &normed,
                pos,
                KvStep::Paged {
                    cache: &mut *cache,
                    stores,
                },
            ) {
                residual_add(&mut hidden, &projected, self.config.residual_scale);
            }
            self.ffn_block_row(
                l,
                layer,
                &mut hidden,
                oai,
                residency
                    .as_ref()
                    .map(|p| p.layer_plan(self.physical_index(l))),
                inputs,
                skip_rows.as_deref().map(|rows| SkipStream { rows }),
            );
            self.hrm_store(hrm.as_mut(), l, &hidden);
        }

        let final_normed = self.final_norm.apply(&hidden, self.config.rms_norm_eps);
        Ok(self.logits_from_normed(&final_normed))
    }

    /// The body of [`Self::forward_batch_last_paged`], already running on a
    /// CPU-pool worker. See `entry.rs` for why the split exists.
    ///
    /// `every_row` is the ONE difference between prefilling a paged
    /// prompt and SCORING one: the gather, the reservation and the
    /// scatter are identical, and what changes is whether the lm_head
    /// is projected once or at every position. A second copy of this
    /// function to vary that is how this repo has lost model features
    /// before, so it is a parameter.
    pub(super) fn forward_batch_paged_on_worker(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
        every_row: bool,
    ) -> Result<Vec<Vec<f32>>, PagedStoreExhausted> {
        assert_eq!(kv_caches.len(), self.config.n_layers);
        assert_eq!(stores.layer_count(), self.config.n_layers);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Reserve every layer up front, under guards spanning the check
        // AND the take. Each layer has its own store, so one having
        // room says nothing about the next -- and under concurrency,
        // checking and then taking as separate steps lets another
        // request slip in between and leave this one half-written.
        //
        // Reserving before the forward rather than after also means a
        // request that cannot fit is refused before it burns a prefill.
        {
            let mut guards = stores.write_all();
            for (cache, store) in kv_caches.iter().zip(guards.iter()) {
                if cache.blocks_needed_for(store, tokens.len()) > store.free_block_count() {
                    return Err(PagedStoreExhausted);
                }
            }
            for (cache, store) in kv_caches.iter_mut().zip(guards.iter_mut()) {
                cache
                    .reserve(store, tokens.len())
                    .expect("checked against free_block_count under this same guard");
            }
        }

        // Gather under read guards, one layer at a time: the forward
        // below is the expensive part and holds nothing.
        let mut scratch: Vec<KvCache> = kv_caches
            .iter()
            .enumerate()
            .map(|(l, cache)| cache.to_contiguous(&stores.read(l)))
            .collect();

        // `host_kv_authoritative`: the scatter below READS these caches,
        // and a Metal prefill otherwise leaves them holding
        // `advance_len` placeholders while the real K/V sits on the
        // device. Copying those placeholders into the page store is
        // what made paged KV on Metal answer fluent nonsense from a
        // prompt the model never attended over.
        let logits = if every_row {
            // One logit row per POSITION, which is what scoring a
            // prompt needs and what prefilling one must not pay for:
            // on a 6000-token prompt this is 6000 vocabulary-wide
            // matmuls instead of one.
            let hiddens = self.forward_hidden_batch_inner(tokens, start_pos, &mut scratch, true);
            let batch_size = hiddens.len();
            let flat: Vec<f32> = hiddens.into_iter().flatten().collect();
            self.logits_from_flat_hidden(flat, batch_size)
        } else {
            vec![self.forward_batch_last_inner(tokens, start_pos, &mut scratch, true)]
        };

        // Scatter into blocks this sequence already owns. Nothing here
        // can fail, which is the point of reserving above.
        for (l, (cache, gathered)) in kv_caches.iter_mut().zip(&scratch).enumerate() {
            let mut store = stores.write(l);
            let width = store.n_kv_heads() * store.head_dim();
            let base = cache.seq_len() * width;
            cache
                .append_contiguous(
                    &mut store,
                    gathered.k_rows().slice(base..gathered.k.len()),
                    gathered.v_rows().slice(base..gathered.v.len()),
                    tokens.len(),
                )
                .expect("blocks reserved above are still held by this sequence");
            // A recurrent layer's state moved in the scratch copy and
            // is not rows (`frink_core::recurrent_state`).
            cache.recurrent = gathered.recurrent.clone();
        }
        Ok(logits)
    }
}
