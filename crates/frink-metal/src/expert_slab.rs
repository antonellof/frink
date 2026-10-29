//! One Shared `MTLBuffer` that holds a pool of routed-expert slots.
//!
//! # Why this exists
//!
//! A streamed expert used to reach the GPU as a host `Vec<u8>` the
//! expert store owned, which the weight cache then COPIED into a second
//! `MTLBuffer` on first use. On Apple silicon those are the same
//! physical RAM, so every hot expert was held twice, once under the
//! expert budget and once under `FRINK_METAL_WEIGHT_CACHE_BYTES`, on
//! exactly the machine whose problem is that the model does not fit.
//!
//! This buffer is the single copy. It is allocated by Metal with
//! `StorageModeShared`, the CPU writes experts straight into its
//! contents, the CPU kernels read them from there, and its contents are
//! registered with the weight cache's alias table so a GPU launch over
//! a slice inside it binds this buffer at an offset instead of
//! uploading anything (`gpu::register_owned_region`). The expert store
//! counts the bytes; the weight cache charges zero for an alias.
//!
//! What it is NOT: a policy. Which expert lives in which slot, and when
//! a slot may be rewritten, is `frink_core::expert_store`'s business.
//! The rule this buffer depends on is that a slot is never rewritten
//! while a command buffer reading it is in flight; the store enforces it
//! by refusing to evict a slot any live lease still points into, and
//! every launch that reads a streamed expert waits for completion while
//! the lease is held.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::gpu::{register_owned_region, shared_metal, unregister_owned_region, MetalError};

static SLAB_BINDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Called by the weight cache each time it builds an alias into a slab.
pub(crate) fn note_slab_bind() {
    SLAB_BINDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Weight-cache entries built as aliases of a slab, process-wide. A
/// test reads it to prove a streamed expert reached a GPU launch in
/// place: a run that silently fell back to the host, or copied, leaves
/// it where it was.
pub fn slab_binds() -> u64 {
    SLAB_BINDS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A Shared `MTLBuffer` of `len` bytes, host-addressable, aliased by the
/// weight cache for as long as it lives.
pub struct MetalSlab {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    base: NonNull<u8>,
    len: usize,
}

// SAFETY: the buffer is created once and never resized; `base` points
// into its contents and stays valid for as long as `buffer` is retained,
// which is the lifetime of this value. Concurrent access to the bytes
// is governed by the owner (the expert store writes a slot only when no
// lease points into it), so sharing the handle across threads adds no
// access the owner does not already serialise.
unsafe impl Send for MetalSlab {}
unsafe impl Sync for MetalSlab {}

impl MetalSlab {
    /// Allocates `len` bytes of Shared storage. Refuses zero, which
    /// Metal would refuse too, with a less useful error.
    pub fn new(len: usize) -> Result<Self, MetalError> {
        if len == 0 {
            return Err(MetalError::BufferAllocFailed);
        }
        let shared = shared_metal()?;
        let buffer = shared
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocFailed)?;
        let base = buffer.contents().cast::<u8>();
        register_owned_region(buffer.clone(), base.as_ptr() as usize, len);
        Ok(MetalSlab { buffer, base, len })
    }

    /// First byte of the contents. Valid and writable for
    /// [`len`](Self::len) bytes for as long as this value lives.
    pub fn base(&self) -> NonNull<u8> {
        self.base
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True when a weight slice inside this slab binds the slab itself
    /// rather than a copy: the property that keeps unified memory
    /// counted once. For tests and for the load-time report.
    pub fn aliases(&self, inside: &[u8]) -> Result<bool, MetalError> {
        let shared = shared_metal()?;
        let resident = crate::gpu::resident_weight_buffer(&shared.device, inside)?;
        Ok(std::ptr::eq(
            Retained::as_ptr(&resident.buffer),
            Retained::as_ptr(&self.buffer),
        ))
    }
}

impl Drop for MetalSlab {
    fn drop(&mut self) {
        unregister_owned_region(self.base.as_ptr() as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slice inside the slab must bind the slab buffer, not a copy,
    /// and must see bytes written after the alias was first built: the
    /// slots are REUSED, so an alias that captured the contents at
    /// first use would serve the previous occupant.
    ///
    /// Sabotage: register nothing in `MetalSlab::new` and the first
    /// assertion goes red (the weight cache copies).
    #[test]
    #[ignore = "needs a Metal device"]
    fn a_slice_inside_the_slab_aliases_it_and_sees_later_writes() {
        let slab = MetalSlab::new(1 << 16).expect("slab");
        let at = 4096usize;
        let n = 256usize;
        // SAFETY: `at + n` is inside the slab's `len`, and nothing else
        // reads or writes these bytes during the test.
        let slot = unsafe { std::slice::from_raw_parts_mut(slab.base().as_ptr().add(at), n) };
        slot.fill(7);
        assert!(slab.aliases(slot).unwrap(), "slab slice must alias");
        slot.fill(9);
        let shared = shared_metal().unwrap();
        let bound = crate::gpu::resident_weight_buffer(&shared.device, slot).unwrap();
        // SAFETY: the bound buffer is this slab's Shared buffer, whose
        // contents are `len` bytes; `weight_offset + n` is inside it.
        let seen = unsafe {
            std::slice::from_raw_parts(
                (bound.buffer.contents().as_ptr() as *const u8).add(bound.weight_offset),
                n,
            )
        };
        assert!(seen.iter().all(|&b| b == 9), "alias must see the rewrite");
    }
}
