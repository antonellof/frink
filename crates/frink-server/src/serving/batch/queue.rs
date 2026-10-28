//! The two bounded inboxes between a caller and the worker thread: how
//! many jobs may wait for admission, and which of them have been
//! cancelled.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Bounds the number of jobs waiting for admission.
///
/// The reservation is a compare-and-swap loop, not a load followed by a
/// fetch_add: with N threads submitting at once, "read the depth, then
/// increment it" admits every thread that read a value below the cap,
/// which is precisely the retry storm the cap exists to stop.
pub(super) struct QueueGate {
    pub(super) depth: AtomicUsize,
    pub(super) cap: usize,
    pub(super) rejected: AtomicU64,
}

impl QueueGate {
    pub(super) fn new(cap: usize) -> Self {
        QueueGate {
            depth: AtomicUsize::new(0),
            cap,
            rejected: AtomicU64::new(0),
        }
    }

    /// Claims one queue slot, or reports the depth that refused it.
    pub(super) fn try_reserve(&self) -> Result<(), usize> {
        let mut current = self.depth.load(Ordering::Acquire);
        loop {
            if current >= self.cap {
                self.rejected.fetch_add(1, Ordering::Relaxed);
                return Err(current);
            }
            match self.depth.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => current = actual,
            }
        }
    }

    /// Frees a slot: the worker has taken the job off the channel, or
    /// the send failed and the job never joined the queue at all.
    pub(super) fn release(&self) {
        let previous = self.depth.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "queue depth underflow");
    }

    /// Claims a slot for a job going BACK into the queue: a decoding row
    /// preempted to make room for another. Not capped: the job held a
    /// slot before it was admitted, it has a client streaming from it,
    /// and refusing it here would drop an answer half-way through. Its
    /// admission releases the slot again, so the depth stays the count
    /// of jobs in the worker's queue.
    pub(super) fn requeue(&self) {
        self.depth.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn depth(&self) -> usize {
        self.depth.load(Ordering::Relaxed)
    }

    pub(super) fn rejected(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }
}

/// Identifies one submitted job for cancellation. Handed out by
/// [`ContinuousBatcher::generate`] before the job is sent, so a cancel
/// racing the submission has something to name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct AbortId(pub(super) u64);

/// Ids whose requests have been asked to stop, waiting for the worker
/// to act on them at a step boundary.
///
/// A set rather than a channel: cancelling twice is the same fact
/// stated twice, and the worker should do the work once.
#[derive(Default)]
pub(super) struct AbortInbox {
    pub(super) pending: Mutex<HashSet<AbortId>>,
    pub(super) next_id: AtomicU64,
    /// Requests actually stopped by a cancellation.
    pub(super) aborted: AtomicU64,
}

impl AbortInbox {
    pub(super) fn next_id(&self) -> AbortId {
        AbortId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Called from whichever thread cancelled -- an HTTP handler,
    /// usually. Deliberately does nothing but record the id.
    pub(super) fn enqueue(&self, id: AbortId) {
        self.pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id);
    }

    /// Takes everything pending. Worker thread, at a step boundary.
    pub(super) fn drain(&self) -> HashSet<AbortId> {
        std::mem::take(&mut *self.pending.lock().unwrap_or_else(|p| p.into_inner()))
    }

    pub(super) fn aborted(&self) -> u64 {
        self.aborted.load(Ordering::Relaxed)
    }
}
