//! Per-layer routed-expert id log for the fused MoE decode stack.
//!
//! Why this module exists. [`crate::attn::launch_moe_decode_stack`] folds
//! every MoE layer into ONE command buffer with ONE
//! `waitUntilCompleted`, and the top-k router kernel wrote each layer's
//! selection into a single `top_k`-wide scratch buffer. Layer *n+1*
//! overwrote layer *n*, so after the wait the host could not recover any
//! layer's choice but the last one -- and the stack simply returned empty
//! id vectors with the note "skip expert-id host download on the hot path
//! (sync tax)". The consequence reached all the way out of this crate:
//! `MoeWeights::activation_counts` stayed identically zero on Metal, so
//! the expert-residency/eviction policy that ranks experts by observed
//! hotness ranked every expert equal-at-zero on the one backend this
//! project's development machine runs.
//!
//! The fix keeps the property that comment was protecting. The log is one
//! wide buffer of `layers x stride` `u32`; layer *l* binds it at
//! [`slot_bytes`]; the host reads it ONCE, after the command buffer the
//! stack already waits on. That means:
//!
//! * no extra dispatch, and no extra command buffer;
//! * no extra sync point -- the read happens after an existing wait;
//! * no extra barrier. [`crate::mem_ranges::MemRanges`] tracks whole
//!   `MTLBuffer` identity, so one wide buffer conflicts exactly where the
//!   one narrow buffer conflicted and the barrier sequence is unchanged.
//!
//! The counts this produces are exact, not sampled: every layer of every
//! decoded token contributes its real top-k. (A device-side atomic
//! histogram, the other option on the table, would have needed a shader
//! change and a per-selection atomic, and would still have needed a
//! readback -- for a strictly weaker signal, since a histogram cannot say
//! *which* layer voted.)

use crate::gpu::MetalError;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// First `u32` of layer `layer_idx`'s slot.
pub(crate) const fn slot_start(layer_idx: usize, stride: usize) -> usize {
    layer_idx * stride
}

/// Byte offset to bind layer `layer_idx`'s slot at (`setBuffer:offset:`).
///
/// This is the single definition the GPU writer and the host reader
/// share; they cannot drift apart into naming different slots.
pub(crate) const fn slot_bytes(layer_idx: usize, stride: usize) -> usize {
    slot_start(layer_idx, stride) * std::mem::size_of::<u32>()
}

/// Split a harvested log into one expert-id list per layer.
///
/// Only the first `top_k` ids of each `stride`-wide slot are live; the
/// tail is whatever a previous token (or a wider `top_k`) left there and
/// must never reach the activation counts.
pub(crate) fn split(raw: &[u32], n_layers: usize, top_k: usize, stride: usize) -> Vec<Vec<usize>> {
    (0..n_layers)
        .map(|layer| {
            let at = slot_start(layer, stride);
            raw.get(at..at + top_k)
                .map(|slot| slot.iter().map(|&e| e as usize).collect())
                .unwrap_or_default()
        })
        .collect()
}

/// One layer's slice of a [`MoeIdsLog`], as a kernel argument.
///
/// Carrying the offset beside the buffer (rather than passing the buffer
/// alone and binding at 0) is what stops a caller from silently reading
/// layer 0's slot for every layer -- the bug this module was written to
/// end.
#[derive(Clone, Copy)]
pub(crate) struct IdsBinding<'a> {
    pub buf: &'a ProtocolObject<dyn MTLBuffer>,
    /// Byte offset of the slot inside `buf`.
    pub offset: usize,
}

impl<'a> IdsBinding<'a> {
    /// A whole buffer used as one slot (prefill, and the single-layer
    /// launches: they own an id buffer each, so their offset is 0).
    pub(crate) fn whole(buf: &'a ProtocolObject<dyn MTLBuffer>) -> Self {
        Self { buf, offset: 0 }
    }
}

fn alloc_u32(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    device
        .newBufferWithLength_options(
            n * std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or(MetalError::BufferAllocFailed)
}

/// `layers x stride` routed-expert ids, one slot per decode-stack layer.
pub(crate) struct MoeIdsLog {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    stride: usize,
    layers: usize,
}

impl MoeIdsLog {
    pub(crate) fn new(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        stride: usize,
        layers: usize,
    ) -> Result<Self, MetalError> {
        let stride = stride.max(1);
        let layers = layers.max(1);
        Ok(Self {
            buf: alloc_u32(device, stride * layers)?,
            stride,
            layers,
        })
    }

    /// Grow (never shrink) to hold `n_layers` slots of `top_k` ids. Only
    /// grows, so a one-layer fallback launch after a full-stack launch
    /// does not throw the stack's slots away and re-allocate next token.
    pub(crate) fn ensure(
        &mut self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        top_k: usize,
        n_layers: usize,
    ) -> Result<(), MetalError> {
        if top_k <= self.stride && n_layers <= self.layers {
            return Ok(());
        }
        let stride = self.stride.max(top_k);
        let layers = self.layers.max(n_layers);
        self.buf = alloc_u32(device, stride * layers)?;
        self.stride = stride;
        self.layers = layers;
        Ok(())
    }

    /// The whole buffer, for [`crate::mem_ranges::MemRanges`] hazard
    /// tracking (which is per-buffer, not per-slot).
    pub(crate) fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buf
    }

    /// Layer `layer_idx`'s slot as a kernel argument. `ensure` must have
    /// been called for this many layers; the clamp is there so a caller
    /// that forgot mis-attributes one layer's hotness rather than binding
    /// a kernel past the end of the buffer, and the `debug_assert` turns
    /// that into a test failure instead of a silent one.
    pub(crate) fn binding(&self, layer_idx: usize) -> IdsBinding<'_> {
        debug_assert!(layer_idx < self.layers, "ids log slot out of range");
        IdsBinding {
            buf: &self.buf,
            offset: slot_bytes(layer_idx.min(self.layers - 1), self.stride),
        }
    }

    /// Read every layer's selection back. Call only after the command
    /// buffer that wrote the log has completed.
    pub(crate) fn harvest(&self, n_layers: usize, top_k: usize) -> Vec<Vec<usize>> {
        let n_layers = n_layers.min(self.layers);
        let top_k = top_k.min(self.stride);
        // SAFETY: `self.buf` is a StorageModeShared buffer this type
        // allocated with exactly `self.layers * self.stride` u32s, so the
        // slice is in bounds and correctly aligned (Metal buffer bases are
        // page-aligned). Shared storage means the CPU mapping is the same
        // memory the GPU wrote, and every caller reaches here only after
        // `waitUntilCompleted` on the command buffer that wrote it, so no
        // GPU write races this read.
        let raw = unsafe {
            std::slice::from_raw_parts(
                self.buf.contents().as_ptr() as *const u32,
                self.layers * self.stride,
            )
        };
        split(raw, n_layers, top_k, self.stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The writer's byte offset and the reader's slice must name the same
    /// slot, for every layer. Written the way the GPU writes it -- through
    /// [`slot_bytes`] -- and read the way the host reads it.
    #[test]
    fn split_recovers_each_layers_own_selection() {
        let (stride, top_k, n_layers) = (8usize, 3usize, 4usize);
        let picks = [[5u32, 1, 9], [0, 2, 3], [7, 7, 7], [4, 6, 8]];
        let mut raw = vec![u32::MAX; n_layers * stride];
        for (layer, sel) in picks.iter().enumerate() {
            let at = slot_bytes(layer, stride) / std::mem::size_of::<u32>();
            raw[at..at + top_k].copy_from_slice(sel);
        }

        let got = split(&raw, n_layers, top_k, stride);

        assert_eq!(got.len(), n_layers);
        for (layer, sel) in picks.iter().enumerate() {
            let want: Vec<usize> = sel.iter().map(|&e| e as usize).collect();
            assert_eq!(got[layer], want, "layer {layer} read another layer's slot");
        }
    }

    /// The exact shape of the bug this module ends: with one shared slot
    /// every layer reported the same ids. Distinct per-layer selections
    /// must come back distinct.
    #[test]
    fn layers_do_not_alias_one_anothers_slots() {
        let (stride, top_k, n_layers) = (8usize, 2usize, 3usize);
        let mut raw = vec![0u32; n_layers * stride];
        for layer in 0..n_layers {
            let at = slot_bytes(layer, stride) / std::mem::size_of::<u32>();
            raw[at] = layer as u32 * 10;
            raw[at + 1] = layer as u32 * 10 + 1;
        }

        let got = split(&raw, n_layers, top_k, stride);

        assert_eq!(got, vec![vec![0, 1], vec![10, 11], vec![20, 21]]);
    }

    /// `stride` is the *capacity* of a slot, `top_k` is how much of it is
    /// live. Stale tail entries must not be counted as activations.
    #[test]
    fn split_ignores_the_stale_tail_of_each_slot() {
        let (stride, top_k, n_layers) = (8usize, 2usize, 2usize);
        let mut raw = vec![77u32; n_layers * stride];
        for (layer, sel) in [[1u32, 2], [3, 4]].iter().enumerate() {
            let at = slot_bytes(layer, stride) / std::mem::size_of::<u32>();
            raw[at..at + top_k].copy_from_slice(sel);
        }

        let got = split(&raw, n_layers, top_k, stride);

        assert_eq!(got, vec![vec![1, 2], vec![3, 4]]);
    }

    /// A truncated log yields an empty list for the missing layers rather
    /// than panicking or inventing expert 0 -- an empty list is what
    /// `Decoder`'s Metal arm skips (`if !ids.is_empty()`).
    #[test]
    fn split_returns_empty_for_slots_past_the_end() {
        let raw = vec![1u32, 2, 3, 4];
        let got = split(&raw, 4, 2, 2);
        assert_eq!(got, vec![vec![1, 2], vec![3, 4], Vec::new(), Vec::new()]);
    }

    /// Real Metal: the ids the router kernel writes into a layer's slot
    /// are that layer's routing decision, for the layer's own logits.
    /// Run with `cargo test -p frink-metal --features metal -- --ignored`.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn gpu_router_writes_each_layers_decision_into_its_own_slot() {
        use crate::gpu::{encode_moe_topk_softmax_batch, shared_metal};
        use objc2_metal::{
            MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLResourceOptions,
        };

        let shared = shared_metal().expect("metal device");
        let device = &shared.device;
        let (n_experts, top_k, n_layers) = (16usize, 4usize, 3usize);

        // Distinct, tie-free logits per layer, so each layer's top-k is a
        // different set of experts and a slot mix-up cannot go unnoticed.
        let logits: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                (0..n_experts)
                    .map(|e| ((e * 7 + l * 5) % n_experts) as f32 * 0.5 - 3.0)
                    .collect()
            })
            .collect();

        let mut log = MoeIdsLog::new(device, top_k, n_layers).expect("ids log");
        log.ensure(device, top_k, n_layers).expect("ensure");
        let weights = device
            .newBufferWithLength_options(top_k * 4, MTLResourceOptions::StorageModeShared)
            .expect("weights buffer");
        let logit_bufs: Vec<_> = logits
            .iter()
            .map(|l| {
                let buf = device
                    .newBufferWithLength_options(l.len() * 4, MTLResourceOptions::StorageModeShared)
                    .expect("logits buffer");
                // SAFETY: shared-storage buffer of exactly `l.len()` f32s,
                // written before any GPU work is committed against it.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        l.as_ptr(),
                        buf.contents().as_ptr() as *mut f32,
                        l.len(),
                    );
                }
                buf
            })
            .collect();

        let cmd_buf = shared.queue.commandBuffer().expect("command buffer");
        let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
        for (layer, logit_buf) in logit_bufs.iter().enumerate() {
            encode_moe_topk_softmax_batch(
                &encoder,
                device,
                logit_buf,
                log.binding(layer),
                &weights,
                n_experts as u32,
                top_k as u32,
                true,
                1,
                None,
            )
            .expect("encode topk");
        }
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let got = log.harvest(n_layers, top_k);
        assert_eq!(got.len(), n_layers);
        for (layer, l) in logits.iter().enumerate() {
            let mut order: Vec<usize> = (0..n_experts).collect();
            order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
            let mut want: Vec<usize> = order[..top_k].to_vec();
            want.sort_unstable();
            let mut have = got[layer].clone();
            have.sort_unstable();
            assert_eq!(have, want, "layer {layer} routed to the wrong experts");
        }
    }
}
