//! What a lease points at: an expert's own heap buffer, or one slot of
//! a [`SlotMemory`] pool.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use crate::expert_slab::SlotMemory;

/// A slot pool the store owns: the device for writes, and each slot's
/// address cached so a read never takes the device lock.
pub(super) struct SlotPool {
    pub(super) device: Mutex<Box<dyn SlotMemory>>,
    slot_addrs: Vec<NonNull<u8>>,
    pub(super) slot_bytes: usize,
    pub(super) kind: &'static str,
}

// SAFETY: every address in `slot_addrs` points into memory `device`
// owns, which lives exactly as long as this pool, and never moves
// (`SlotMemory::slot_ptr`'s contract). A slot is written only through
// `device`, under its mutex, and only while no `ExpertBytes` for that
// slot exists (it is on the free list, being claimed); reads go through
// an `ExpertBytes`, whose existence keeps the slot off the free list.
// So no read races a write.
unsafe impl Send for SlotPool {}
unsafe impl Sync for SlotPool {}

impl SlotPool {
    pub(super) fn new(device: Box<dyn SlotMemory>) -> Self {
        let geometry = device.geometry().clone();
        let kind = device.kind();
        let slot_addrs = (0..geometry.slots as u32)
            .map(|s| device.slot_ptr(s))
            .collect();
        SlotPool {
            device: Mutex::new(device),
            slot_addrs,
            slot_bytes: geometry.row_bytes[0],
            kind,
        }
    }

    pub(super) fn slots(&self) -> usize {
        self.slot_addrs.len()
    }
}

/// One expert's bytes, wherever they live. Dereferences to the exact
/// expert length, never the slot's, so a slot's zero padding is never
/// part of a weight view.
pub struct ExpertBytes {
    repr: Repr,
}

enum Repr {
    Heap(Vec<u8>),
    Slot {
        pool: Arc<SlotPool>,
        slot: u32,
        len: usize,
    },
}

impl ExpertBytes {
    pub(super) fn slot(pool: Arc<SlotPool>, slot: u32, len: usize) -> Self {
        debug_assert!(len <= pool.slot_bytes);
        ExpertBytes {
            repr: Repr::Slot { pool, slot, len },
        }
    }

    /// The slot these bytes live in, or `None` for a heap buffer (a
    /// heap-mode entry or an uncached pass-through).
    pub fn slot_index(&self) -> Option<u32> {
        match &self.repr {
            Repr::Heap(_) => None,
            Repr::Slot { slot, .. } => Some(*slot),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        match &self.repr {
            Repr::Heap(v) => v,
            Repr::Slot { pool, slot, len } => {
                let base = pool.slot_addrs[*slot as usize];
                // SAFETY: `base` is valid for `slot_bytes >= len` bytes
                // while `pool` lives, and this value holds the pool. The
                // slot is not rewritten while this value exists: the
                // store only reclaims a slot whose entry's `Arc` it holds
                // alone, and this value is reachable only through such
                // an `Arc` (see `SlotPool`'s SAFETY note).
                unsafe { std::slice::from_raw_parts(base.as_ptr(), *len) }
            }
        }
    }
}

impl From<Vec<u8>> for ExpertBytes {
    fn from(v: Vec<u8>) -> Self {
        ExpertBytes {
            repr: Repr::Heap(v),
        }
    }
}

impl std::ops::Deref for ExpertBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}
