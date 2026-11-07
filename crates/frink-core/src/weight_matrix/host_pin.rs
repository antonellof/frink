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
//!
//! Never call it from `frink-core`'s own unit tests. Those share one
//! binary with tests that need the device, so a pin there would turn
//! Metal off for whichever of them happened to run later. A unit test
//! that asserts on a host path takes `ForceHost` instead, which holds
//! for its own thread and leaves its neighbours on the device.

use std::sync::atomic::{AtomicBool, Ordering};

static HOST_PINNED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
thread_local! {
    static HOST_THIS_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The calling thread's projections take the host body until this
/// drops, on every backend, because it answers through the one check
/// they all read. libtest gives each test its own thread, and reuses
/// one under `--test-threads=1`, which is why this restores on drop.
///
/// Only gates read on the calling thread see it, so a path that picks
/// its backend from inside a rayon task is not held by this.
#[cfg(test)]
pub(crate) struct ForceHost {
    previous: bool,
    // The flag belongs to this thread, so the guard cannot leave it.
    _not_send: std::marker::PhantomData<*const ()>,
}

#[cfg(test)]
impl ForceHost {
    pub(crate) fn new() -> Self {
        Self {
            previous: HOST_THIS_THREAD.replace(true),
            _not_send: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
impl Drop for ForceHost {
    fn drop(&mut self) {
        HOST_THIS_THREAD.set(self.previous);
    }
}

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
    #[cfg(test)]
    if HOST_THIS_THREAD.get() {
        return true;
    }
    HOST_PINNED.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard holds for its own thread, is gone once it drops, and
    /// is invisible elsewhere. The last half is the point: it is what
    /// lets a host-path test share a binary with a device test.
    ///
    /// The first assertion also reads red if a unit test in this crate
    /// ever calls [`pin_host_backend`], which the module doc forbids.
    #[test]
    fn force_host_holds_this_thread_until_it_drops_and_no_other() {
        assert!(!host_backend_pinned(), "a unit test pinned the process");
        {
            let _host = ForceHost::new();
            assert!(host_backend_pinned());
            let elsewhere = std::thread::spawn(host_backend_pinned).join().unwrap();
            assert!(!elsewhere, "the guard reached another thread");
            {
                let _nested = ForceHost::new();
            }
            assert!(host_backend_pinned(), "a nested guard cleared the outer");
        }
        assert!(!host_backend_pinned(), "the guard outlived its scope");
    }
}
