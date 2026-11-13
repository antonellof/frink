//! Where a stack's output lands on the device before its one download:
//! the layout, which is pure, and the buffers, which a stack enqueued
//! as it runs allocates per call and a captured decode token
//! (`super::decode_graph`) allocates once and keeps, so every replay
//! writes the addresses its graph froze.

use crate::gpu::CudaError;
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// Most experts a device-routed token picks (`moe_topk` takes up to 16):
/// a layer's slice of the routed ids holds this many a row.
pub(crate) const IDS_PER_ROW: usize = 16;

/// The staged download, the hidden rows first and then each layer's K
/// rows and V rows, and the routed ids, a slice a layer.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct StageLayout {
    /// Floats of hidden rows at the start.
    pub(crate) hidden: usize,
    /// Per layer, `(offset, len)` of its K rows; its V rows follow.
    pub(crate) kv: Vec<(usize, usize)>,
    /// Floats staged in all.
    pub(crate) len: usize,
    /// Routed ids in a layer's slice.
    pub(crate) ids_per_layer: usize,
    /// Per layer, the experts a token picks when the device routes it.
    pub(crate) device_k: Vec<Option<usize>>,
}

impl StageLayout {
    /// The layout for `batch` rows of `hidden_dim` through layers whose
    /// K and V rows are `kv_widths` wide, the device routing picking
    /// `device_k` experts a token where it routes.
    pub(crate) fn of(
        batch: usize,
        hidden_dim: usize,
        kv_widths: impl IntoIterator<Item = usize>,
        device_k: Vec<Option<usize>>,
    ) -> Self {
        let hidden = batch * hidden_dim;
        let mut len = hidden;
        let kv = kv_widths
            .into_iter()
            .map(|w| {
                let n = batch * w;
                let at = len;
                len += 2 * n;
                (at, n)
            })
            .collect();
        Self {
            hidden,
            kv,
            len,
            ids_per_layer: batch * IDS_PER_ROW,
            device_k,
        }
    }

    /// Whether any layer routes on the device, and so writes ids.
    pub(crate) fn routed(&self) -> bool {
        self.device_k.iter().any(Option::is_some)
    }
}

/// The device buffers a stack writes for the host: the staged download
/// and, when a layer routes on the device, its ids.
pub(crate) struct StackOutBufs {
    pub(crate) staged: CudaSlice<f32>,
    pub(crate) ids: Option<CudaSlice<i32>>,
}

impl StackOutBufs {
    /// Zeroed buffers for `layout`, on `dev`'s stream.
    pub(crate) fn alloc(dev: &Arc<CudaDevice>, layout: &StageLayout) -> Result<Self, CudaError> {
        let staged = dev
            .alloc_zeros::<f32>(layout.len)
            .map_err(|e| CudaError::Launch(format!("stack staging alloc: {e:?}")))?;
        let ids = if layout.routed() {
            Some(
                dev.alloc_zeros::<i32>(layout.device_k.len() * layout.ids_per_layer)
                    .map_err(|e| CudaError::Launch(format!("routed ids alloc: {e:?}")))?,
            )
        } else {
            None
        };
        Ok(Self { staged, ids })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hidden rows first, then each layer's K rows and V rows back to
    /// back at its own width, with nothing between them and nothing
    /// after; ids sized by the batch.
    #[test]
    fn the_staged_download_packs_the_hidden_rows_then_each_layer_s_k_and_v() {
        let l = StageLayout::of(3, 10, [4, 6], vec![None, Some(2)]);
        assert_eq!(l.hidden, 30);
        assert_eq!(l.kv, vec![(30, 12), (54, 18)]);
        assert_eq!(l.len, 30 + 2 * 12 + 2 * 18);
        assert_eq!(l.ids_per_layer, 3 * IDS_PER_ROW);
        assert!(l.routed());
        let one = StageLayout::of(1, 64, [32], vec![None]);
        assert_eq!(one.kv, vec![(64, 32)]);
        assert_eq!(one.len, 128);
        assert!(!one.routed());
    }
}
