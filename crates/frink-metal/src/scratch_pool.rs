//! Reusable shared-storage buffers, keyed by length.
//!
//! # Why
//!
//! A launch that allocates its own scratch pays Metal for the
//! allocation every time it runs. That is invisible in the
//! `FRINK_METAL_GPU_TIMING` ledger, which times `commit` to
//! completion, so it reads as neither GPU time nor submission latency:
//! it is host time inside the launch.
//!
//! It was measured on the fused recurrent branch
//! (`crate::gdn_branch`). That launch is correct and removes about 17
//! ms of host recurrence from a Bonsai decode token, and its first
//! version ran 7.22 to 6.68 tok/s -- SLOWER -- against a ledger that
//! accounted for only 10 ms of the 19 ms it had lost. The missing part
//! was eleven fresh buffers per layer per token, some 500 allocations a
//! token. Making the layer's constants resident and wrapping the
//! convolution window in place recovered 6.68 to 6.98; this recovers
//! the rest.
//!
//! # What makes it sound
//!
//! Buffers are handed out and returned by the SAME call, which waits
//! for its command buffer before returning them, so a buffer is never
//! in two command buffers at once and the GPU is never reading one that
//! has been handed to somebody else. A caller that returned them before
//! waiting would be the bug this design cannot express: [`Scratch`]
//! owns them and gives them back on drop.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;
use std::sync::Mutex;

struct Pooled(Retained<ProtocolObject<dyn MTLBuffer>>);

// SAFETY: a buffer is only ever inside the `POOL` mutex or owned by one
// `Scratch`, which is not `Send` and lives on the thread that made it.
unsafe impl Send for Pooled {}

static POOL: Mutex<Option<HashMap<usize, Vec<Pooled>>>> = Mutex::new(None);

/// How many returned buffers of `floats` floats the pool keeps.
///
/// A cap, so a run that touches many distinct widths does not hold
/// every one of them forever, and one that SCALES with the width,
/// because a decode run holds its scratch across a whole token: a
/// Bonsai-2-27B token is some 300 live buffers of the hidden width
/// alone. A flat eight sent all but eight of them back to Metal at the
/// end of every token and allocated them again during the next, which
/// was a sixth of the token's host time (#540). Small buffers are kept
/// up to [`KEEP_BYTES`] per width; a large one still eight deep.
fn keep_at_most(floats: usize) -> usize {
    (KEEP_BYTES / (floats * 4).max(1)).max(8)
}

/// What the pool may keep per width beyond the eight-deep floor.
const KEEP_BYTES: usize = 64 << 20;

/// Buffers borrowed from the pool for one launch, returned on drop.
pub(crate) struct Scratch {
    bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    lens: Vec<usize>,
}

impl Scratch {
    /// `lens.len()` shared-storage buffers of the given FLOAT lengths,
    /// reused when the pool has them.
    pub(crate) fn take(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        lens: &[usize],
    ) -> Option<Self> {
        let mut guard = POOL.lock().ok()?;
        let pool = guard.get_or_insert_with(HashMap::new);
        let mut bufs = Vec::with_capacity(lens.len());
        for &n in lens {
            let buf = match pool.get_mut(&n).and_then(|v| v.pop()) {
                Some(p) => p.0,
                None => device
                    .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)?,
            };
            bufs.push(buf);
        }
        Some(Self {
            bufs,
            lens: lens.to_vec(),
        })
    }

    /// Buffer `i` filled with `data`, which must be exactly the length
    /// it was asked for.
    ///
    /// This is what an upload becomes once the buffer is pooled: a
    /// memcpy into shared storage rather than a fresh allocation whose
    /// bytes are copied by Metal. The same copy, without the allocator.
    pub(crate) fn write(&self, i: usize, data: &[f32]) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        if data.len() != self.lens[i] {
            return None;
        }
        let buf = &self.bufs[i];
        // SAFETY: shared storage of exactly `lens[i]` floats, which
        // this `Scratch` owns and no command buffer is reading: the
        // caller's launch has not been committed yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buf.contents().as_ptr() as *mut f32,
                data.len(),
            );
        }
        Some(buf)
    }

    /// Buffer `i`, in the order the lengths were asked for.
    pub(crate) fn buf(&self, i: usize) -> &ProtocolObject<dyn MTLBuffer> {
        &self.bufs[i]
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let Ok(mut guard) = POOL.lock() else { return };
        let pool = guard.get_or_insert_with(HashMap::new);
        for (buf, n) in self.bufs.drain(..).zip(self.lens.iter()) {
            let slot = pool.entry(*n).or_default();
            if slot.len() < keep_at_most(*n) {
                slot.push(Pooled(buf));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A buffer handed back comes out again rather than being
    /// reallocated, which is the whole point; and two live `Scratch`
    /// never share one, which is what makes it safe.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn a_returned_buffer_is_reused_and_a_live_one_is_not() {
        let shared = crate::gpu::shared_metal().expect("a device");
        let device = &shared.device;
        let first = {
            let s = Scratch::take(device, &[777]).expect("a buffer");
            s.buf(0) as *const _
        };
        let second = {
            let s = Scratch::take(device, &[777]).expect("a buffer");
            s.buf(0) as *const _
        };
        assert_eq!(first, second, "the same buffer comes back out");

        let held = Scratch::take(device, &[777]).expect("a buffer");
        let other = Scratch::take(device, &[777]).expect("a buffer");
        assert_ne!(
            held.buf(0) as *const _,
            other.buf(0) as *const _,
            "a buffer still held is not handed to a second caller"
        );
    }

    /// A whole decode run's worth of one width comes back out of the
    /// pool rather than out of Metal: 300 buffers of a hidden width,
    /// held at once as a run holds them, then taken again.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn a_run_held_at_once_is_reused_whole() {
        let shared = crate::gpu::shared_metal().expect("a device");
        let device = &shared.device;
        let take = || -> Vec<Scratch> {
            (0..300)
                .map(|_| Scratch::take(device, &[5_121]).expect("a buffer"))
                .collect()
        };
        let ptrs = |held: &[Scratch]| -> std::collections::HashSet<usize> {
            held.iter()
                .map(|s| s.buf(0) as *const _ as *const u8 as usize)
                .collect()
        };
        let first = ptrs(&take());
        let second = ptrs(&take());
        assert_eq!(first.len(), 300);
        assert_eq!(
            first.intersection(&second).count(),
            300,
            "every buffer of the run came back out"
        );
    }
}
