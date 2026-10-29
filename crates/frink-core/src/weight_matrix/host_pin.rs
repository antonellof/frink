//! A process-wide "the host, from here on", for a caller that has to
//! know which engine computed its numbers.
//!
//! Every backend's `dense_enabled` is a `OnceLock` over the environment,
//! so the FIRST matvec in a process decides the backend for all of them.
//! That is right for `frink run`, which sets `FRINK_METAL` before it
//! loads anything. It is wrong for a test binary, where tests run in
//! parallel in no fixed order: a host golden that set `FRINK_METAL=0`
//! got the CPU only if no other test had touched a weight first, and on
//! a `--features metal` build with the variable unset the probe said
//! "Metal" and 163 host goldens compared the GPU against a CPU line.
//!
//! [`pin_host_backend`] is checked BEFORE those cells, so it holds
//! whatever was decided earlier. It only ever turns devices off: a pin
//! that could turn one on would be a second spelling of `FRINK_METAL`,
//! and two spellings of one switch is how this repo loses a flag.

use std::sync::atomic::{AtomicBool, Ordering};

static HOST_PINNED: AtomicBool = AtomicBool::new(false);

/// From now on, in this process, no dense backend reports itself
/// enabled: every [`super::metal_dense_enabled`] (and the CUDA and Vulkan
/// twins) answers `false`, so every device path that gates on them,
/// which is every one, takes the host body. Irreversible by design.
pub fn pin_host_backend() {
    HOST_PINNED.store(true, Ordering::SeqCst);
}

/// Whether [`pin_host_backend`] has run in this process.
#[inline]
pub fn host_backend_pinned() -> bool {
    HOST_PINNED.load(Ordering::Relaxed)
}
