//! A run of recurrent PREFILL layers in one Metal submission, or
//! nothing (issue #540).
//!
//! The decode path's [`Decoder::fused_layer_parts`] says which layers
//! qualify, and [`Decoder::fused_attention_tail_eligible`] which
//! attention tails may lead a run, so prefill and decode cannot
//! disagree about what a fusable layer is; `frink_metal::gdn_prefill`
//! runs them with the hidden batch resident across the run.

use super::{Decoder, LayerWeights};
use crate::fused_layer::LayerFfnParts;
use frink_core::KvCache;

impl Decoder {
    /// Layers `start..` of a `rows`-token batch of one sequence, as far
    /// as consecutive recurrent layers qualify and no further than
    /// `run_end`. Returns where the caller resumes, with `hidden`
    /// replaced by the batch as it leaves the run, or `None` when
    /// nothing ran and the host bodies take layer `start`.
    pub(crate) fn metal_recurrent_prefill_run(
        &self,
        start: usize,
        run_end: usize,
        hidden: &mut Vec<f32>,
        rows: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<usize> {
        self.metal_prefill_run(start, run_end, None, hidden, rows, kv_caches)
    }

    /// Attention layer `l`'s tail -- `wo` over `branch` (the attention
    /// output as [`Self::attn_branch_rows`] leaves it), the residual and
    /// the FFN -- and then the recurrent layers after it, in the same
    /// submission. `hidden` is the residual stream as it entered layer
    /// `l`. `None` when the tail is not that shape, and then nothing
    /// has run.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn metal_attn_tail_prefill_run(
        &self,
        l: usize,
        layer: &LayerWeights,
        branch: &[f32],
        run_end: usize,
        hidden: &mut Vec<f32>,
        rows: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<usize> {
        if !self.prefill_rows_take_device(rows) || !self.fused_attention_tail_eligible(l, layer) {
            return None;
        }
        let ffn = LayerFfnParts::for_layer(layer, self.config.rms_norm_eps, true)?;
        let (base, fold) = layer.attn.o_proj.launch_parts();
        let o_proj = base.mul_mm_sg_launch()?;
        let width = branch.len() / rows;
        let fold = match fold {
            None => None,
            Some(f) => Some(f.metal_plan(width)?),
        };
        let tail = frink_metal::gdn_prefill::PrefillAttnTail {
            branch,
            o_proj,
            fold,
            ffn: ffn.prefill_launches()?,
            eps: self.config.rms_norm_eps,
        };
        let end = self.metal_prefill_run(l + 1, run_end, Some(&tail), hidden, rows, kv_caches)?;
        layer.moe.record_activations_dense();
        Some(end)
    }

    /// Below the chunked recurrence's own device crossover the host
    /// body is the faster one, and it is where `delta_chunk_rows` would
    /// have sent these rows anyway.
    fn prefill_rows_take_device(&self, rows: usize) -> bool {
        rows >= frink_core::gdn_chunk::DEVICE_ROWS
            && frink_core::weight_matrix::metal_dense_enabled()
    }

    /// The ONE body both entries share: `head`, if any, then the
    /// qualifying recurrent layers from `start`.
    fn metal_prefill_run(
        &self,
        start: usize,
        run_end: usize,
        head: Option<&frink_metal::gdn_prefill::PrefillAttnTail<'_>>,
        hidden: &mut Vec<f32>,
        rows: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<usize> {
        if !self.prefill_rows_take_device(rows) {
            return None;
        }
        let eps = self.config.rms_norm_eps;
        let mut parts = Vec::new();
        let mut end = start;
        while end < run_end.min(kv_caches.len()) {
            let Some(p) = self.fused_layer_parts(end) else {
                break;
            };
            parts.push(p);
            end += 1;
        }
        if parts.is_empty() && head.is_none() {
            return None;
        }
        let mut entries = Vec::with_capacity(parts.len());
        for ((gdn, attn_norm, ffn), cache) in parts.iter().zip(kv_caches[start..end].iter_mut()) {
            let layer = gdn.prefill_layer(attn_norm, eps, ffn)?;
            let state = cache.recurrent.get_or_insert_with(|| gdn.zero_state());
            entries.push((
                layer,
                frink_metal::gdn_prefill::PrefillGdnState {
                    ssm_ptr: state.ssm.as_ptr(),
                    ssm_bytes: state.ssm.alloc_bytes(),
                    conv_ptr: state.conv.as_ptr(),
                    conv_bytes: state.conv.alloc_bytes(),
                    conv_len: state.conv.len(),
                },
            ));
        }
        // SAFETY: each state is page-aligned (`AlignedF32`) and lives in
        // its own cache element, borrowed mutably through `kv_caches`
        // for this whole call and touched by nothing else; the launch
        // waits for the GPU before returning.
        let out = unsafe {
            frink_metal::gdn_prefill::launch_gdn_prefill_run(hidden, rows, head, &entries)
        }
        .ok()?;
        drop(entries);
        for (l, cache) in (start..end).zip(kv_caches[start..end].iter_mut()) {
            cache
                .advance_len(rows)
                .expect("unbounded/planned KvCache growth is infallible");
            self.layer_for(l).moe.record_activations_dense();
        }
        *hidden = out;
        Some(end)
    }
}
