//! The recurrent half of a hybrid layer, at the site attention occupies,
//! on the three cache backings.
//!
//! Two blocks stand where attention stands on a zero-KV layer
//! (`crate::layer_shapes::AttnShape`): LFM2's short convolution, whose
//! state is a window of its inputs and lives as the layer's KV history
//! (`crate::shortconv`), and the Mamba-2 block, whose state is a
//! reduction and lives as a `RecurrentState` beside the cache
//! (`crate::mamba2`, `frink_core::recurrent_state`). Each body takes
//! its state as a closure or a `&mut`; this file is the ONE place the
//! backing is matched for either, so a backing cannot pad, index, push
//! or carry the state differently from the others -- the same reason
//! [`super::attn_block::KvStep`] exists for attention.

use frink_core::cache::KvCache;
use frink_core::recurrent_state::RecurrentState;

use super::attn_block::KvStep;
use super::{Decoder, LayerWeights};
use crate::layer_shapes::AttnShape;
use crate::shortconv::window_from_history;

impl Decoder {
    /// `rows` consecutive positions of ONE sequence through layer
    /// `layer_idx`'s recurrent block. `normed` is `[rows][n_embd]`, the
    /// `attn_norm` output; the result is the branch's contribution to
    /// the residual, which the caller adds (the contract `attn_block`
    /// has). Dispatches on the layer's SHAPE, so a layer whose weights
    /// and shape disagree panics here rather than running the wrong
    /// block.
    pub(crate) fn recurrent_block(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        rows: usize,
        kv: KvStep<'_>,
    ) -> Vec<f32> {
        let mut out = match self.config.layer_shape(layer_idx).attention {
            AttnShape::ShortConv => self.shortconv_block(layer_idx, layer, normed, rows, kv),
            AttnShape::Mamba1
            | AttnShape::Mamba2
            | AttnShape::Plamo2Ssm
            | AttnShape::Gdn
            | AttnShape::Lightning => self.ssm_block(layer_idx, layer, normed, rows, kv),
            other => unreachable!("layer {layer_idx} is {other:?}, not a recurrent block"),
        };
        // `plamo2.cpp:150`: the block's output under `attn_post_norm`
        // before the residual add, the same site the attention tail
        // applies it at (`attn_out_to_residual_rows`). Loaded only for
        // the shape whose graph creates it (`layer_shapes::
        // load_non_gqa_attention`), so this is a no-op elsewhere.
        if let Some(post) = &layer.attn.post_attn_norm {
            let hidden = post.len();
            out = out
                .chunks(hidden)
                .flat_map(|row| {
                    frink_core::matmul::rms_norm(row, post, self.config.post_norm_eps())
                })
                .collect();
        }
        out
    }

    /// LFM2's short convolution. Each row's `bx` is pushed to the
    /// sequence's layer cache as its one "K" row
    /// (`AttnShape::cache_geometry`) before the window is read, so the
    /// window's newest entry is this row and the state after the call
    /// is the history llama.cpp would carry forward.
    fn shortconv_block(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        rows: usize,
        kv: KvStep<'_>,
    ) -> Vec<f32> {
        let conv =
            layer.attn.shortconv.as_ref().unwrap_or_else(|| {
                panic!("layer {layer_idx} is ShortConv-shaped but has no weights")
            });
        let (l_cache, n_embd) = (conv.l_cache, conv.hidden_dim());
        match kv {
            KvStep::Decode(cache) | KvStep::Batched(cache) => {
                conv.forward_rows(normed, rows, |bx| {
                    contiguous_step(cache, bx, l_cache, n_embd)
                })
            }
            KvStep::Paged { cache, stores } => conv.forward_rows(normed, rows, |bx| {
                {
                    let mut store = stores.write(layer_idx);
                    cache
                        .push(&mut store, bx, &[])
                        .expect("every caller reserves this row's pages before the stack runs");
                }
                let store = stores.read(layer_idx);
                let table = cache.block_table();
                let block = store.block_size();
                window_from_history(l_cache, n_embd, cache.seq_len(), |i| {
                    store.k_row(table[i / block], i % block)
                })
            }),
        }
    }

    /// The state-space block where attention would be. The state is the
    /// cache's `recurrent` slot ([`Self::ssm_state_step`]); after the
    /// rows run, the cache is advanced by `rows` EMPTY positions so its
    /// `positions()` / `seq_len()` still says how far the sequence has
    /// got, which is what every consumer of a per-layer cache reads.
    fn ssm_block(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        rows: usize,
        mut kv: KvStep<'_>,
    ) -> Vec<f32> {
        let out = self.ssm_state_step(layer_idx, layer, normed, rows, kv.recurrent_slot());
        match kv {
            KvStep::Decode(cache) | KvStep::Batched(cache) => {
                cache
                    .advance_len(rows)
                    .expect("unbounded/planned KvCache growth is infallible");
            }
            KvStep::Paged { cache, stores } => {
                let mut store = stores.write(layer_idx);
                for _ in 0..rows {
                    cache
                        .push(&mut store, &[], &[])
                        .expect("every caller reserves this row's pages before the stack runs");
                }
            }
        }
        out
    }

    /// The block's arithmetic (`crate::ssm_block`, either generation)
    /// over `rows` rows of ONE sequence, on the
    /// state in `slot`, created at this layer's size on the sequence's
    /// first token (zeros, as `build_rs` zeroes a new sequence's). Counts
    /// no positions: the zero-KV arm above does that, and the parallel
    /// arm ([`Self::add_parallel_ssm`]) leaves it to attention.
    pub(crate) fn ssm_state_step(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        rows: usize,
        slot: &mut Option<RecurrentState>,
    ) -> Vec<f32> {
        let block =
            layer.attn.ssm.as_ref().unwrap_or_else(|| {
                panic!("layer {layer_idx} runs a Mamba-2 block but has no weights")
            });
        let state = slot.get_or_insert_with(|| block.zero_state());
        block.forward_rows(normed, rows, state, self.config.rms_norm_eps)
    }

    /// `falcon-h1.cpp:156-158`: on an attention layer that ALSO runs the
    /// Mamba-2 block (`crate::mamba2::PARALLEL_WITH_ATTENTION`), the
    /// block's output over the same `normed` rows attention reads, or
    /// `None` for a layer without the block. Run BEFORE the attention
    /// push so the two never disagree about which token the state saw;
    /// [`Self::add_parallel_ssm`] sums it into the attention branch.
    /// ONE pair for the row body and both batched bodies.
    pub(crate) fn parallel_ssm_rows(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        rows: usize,
        slot: &mut Option<RecurrentState>,
    ) -> Option<Vec<f32>> {
        if layer.attn.ssm.is_none() || self.config.layer_shape(layer_idx).attention.is_recurrent() {
            return None;
        }
        Some(self.ssm_state_step(layer_idx, layer, normed, rows, slot))
    }

    /// `falcon-h1.cpp:160`: `attn_out + ssm_out`, before the one residual
    /// add. A no-op for `None`.
    pub(crate) fn add_parallel_ssm(projected: &mut [f32], ssm: Option<Vec<f32>>) {
        if let Some(ssm) = ssm {
            assert_eq!(ssm.len(), projected.len());
            for (p, s) in projected.iter_mut().zip(&ssm) {
                *p += s;
            }
        }
    }
}

/// Push one `bx` row to a contiguous cache and read the window back.
fn contiguous_step(cache: &mut KvCache, bx: &[f32], l_cache: usize, n_embd: usize) -> Vec<f32> {
    cache
        .push(bx, &[])
        .expect("unbounded/planned KvCache growth is infallible");
    let rows = cache.rows();
    window_from_history(l_cache, n_embd, rows, |i| {
        &cache.k[i * n_embd..(i + 1) * n_embd]
    })
}
