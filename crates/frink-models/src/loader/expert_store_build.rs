//! The one expert store a streamed GGUF model shares across its layers:
//! which slot memory it lives in, how many slots the budget buys, and
//! the hotness it ranks eviction by.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use frink_core::expert_slab::{HostSlotSlab, SlotMemory};
use frink_core::expert_store::{ExpertHotness, ExpertKey, ExpertStore};

use super::expert_source::{GgufExpertSource, StoredExpertLayout};

/// The decoder's own per-layer activation counters, read by the store's
/// eviction policy. `counts[l]` is layer `l`'s
/// `MoeWeights::activation_counts`, the same `Arc` the decoder bumps at
/// routing time, so the store ranks by exactly what the router chose and
/// keeps no second tally that could disagree with it.
pub(crate) struct LayerActivationCounts {
    counts: Vec<Option<Arc<[AtomicU64]>>>,
}

impl LayerActivationCounts {
    pub(crate) fn new(counts: Vec<Option<Arc<[AtomicU64]>>>) -> Self {
        LayerActivationCounts { counts }
    }
}

impl ExpertHotness for LayerActivationCounts {
    fn selections(&self, key: ExpertKey) -> u64 {
        self.counts
            .get(key.layer as usize)
            .and_then(Option::as_ref)
            .and_then(|c| c.get(key.expert as usize))
            .map_or(0, |c| c.load(Ordering::Relaxed))
    }
}

/// How many `slot_bytes` slots `budget` buys. The budget is a ceiling,
/// so the slab is the largest whole number of slots under it.
pub(crate) fn slots_for_budget(budget: usize, slot_bytes: usize) -> usize {
    budget.checked_div(slot_bytes).unwrap_or(0)
}

/// The slot memory for `slots` slots: one Shared `MTLBuffer` when Metal
/// serves this process (the GPU then reads streamed experts in place,
/// and unified memory holds each one once), host RAM otherwise or if
/// the device refuses the allocation.
fn slot_memory(
    num_layers: usize,
    slots: usize,
    slot_bytes: usize,
) -> Result<Box<dyn SlotMemory>, String> {
    #[cfg(feature = "metal")]
    if frink_core::metal_dense_enabled() {
        match frink_core::expert_slab::MetalSlotSlab::metal(num_layers, slots, slot_bytes) {
            Ok(slab) => return Ok(Box::new(slab)),
            Err(e) => eprintln!(
                "frink: Metal expert slab refused ({e}); using host memory, so Metal \
                 launches will copy streamed experts"
            ),
        }
    }
    Ok(Box::new(HostSlotSlab::host(num_layers, slots, slot_bytes)?))
}

/// Builds the store: slot mode sized from `budget`, ranked by `hotness`.
///
/// One slot size for every layer, the largest expert's, so any expert
/// fits any slot; a mixed-quant checkpoint wastes the difference on its
/// smaller layers rather than splitting the pool. A budget smaller than
/// one slot keeps heap mode, whose every acquire is then an uncached
/// pass-through: correct, and as slow as asking for no cache deserves.
pub(crate) fn build_expert_store(
    source: GgufExpertSource,
    layouts: &[&[StoredExpertLayout]],
    num_layers: usize,
    budget: usize,
    hotness: LayerActivationCounts,
) -> ExpertStore<GgufExpertSource> {
    let slot_bytes = layouts
        .iter()
        .flat_map(|l| l.iter())
        .map(StoredExpertLayout::total_bytes)
        .max()
        .unwrap_or(0);
    let total: usize = layouts.iter().map(|l| l.len()).sum();
    // A budget above the expert total buys no more than one slot each.
    let slots = slots_for_budget(budget, slot_bytes).min(total);
    let hotness: Arc<dyn ExpertHotness> = Arc::new(hotness);
    if slots == 0 {
        return ExpertStore::new(source, budget).with_hotness(hotness);
    }
    match slot_memory(num_layers, slots, slot_bytes) {
        Ok(memory) => {
            eprintln!(
                "frink: expert store: {slots} of {total} routed experts resident \
                 ({slots} slots x {slot_bytes} bytes, {} memory)",
                memory.kind()
            );
            ExpertStore::with_slots(source, memory).with_hotness(hotness)
        }
        Err(e) => {
            eprintln!("frink: expert slot pool unavailable ({e}); using per-expert buffers");
            ExpertStore::new(source, budget).with_hotness(hotness)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_budget_buys_whole_slots_and_never_more() {
        assert_eq!(slots_for_budget(1000, 300), 3);
        assert_eq!(slots_for_budget(299, 300), 0);
        assert_eq!(slots_for_budget(1000, 0), 0);
    }

    /// The store must read the decoder's counters, not a copy taken at
    /// load: a selection recorded after the store was built has to move
    /// its ranking, or hotness would be frozen at zero.
    #[test]
    fn hotness_reads_the_live_counters_it_was_given() {
        let layer1: Arc<[AtomicU64]> = (0..4).map(|_| AtomicU64::new(0)).collect();
        let h = LayerActivationCounts::new(vec![None, Some(Arc::clone(&layer1))]);
        let k = ExpertKey {
            layer: 1,
            expert: 2,
        };
        assert_eq!(h.selections(k), 0);
        layer1[2].fetch_add(3, Ordering::Relaxed);
        assert_eq!(h.selections(k), 3);
        assert_eq!(
            h.selections(ExpertKey {
                layer: 0,
                expert: 0
            }),
            0
        );
        assert_eq!(
            h.selections(ExpertKey {
                layer: 9,
                expert: 0
            }),
            0
        );
    }
}
