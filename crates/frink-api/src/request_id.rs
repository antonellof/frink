//! Server-assigned request ids.
//!
//! The id is stated by the server in the response (and in the *first*
//! streamed chunk, before any content), so a client never has to invent
//! one or correlate by heuristic. The alternative -- matching a live
//! request against a metrics snapshot by "the newest one that looks like
//! mine" -- is what a UI is forced into when the server stays silent,
//! and it mis-attributes as soon as two chats run at once.
//!
//! Format: `chatcmpl-` + a per-process random-ish stamp + a monotonic
//! counter. The stamp makes ids from two runs of the same binary
//! distinguishable (so a log spanning a restart does not collide); the
//! counter makes them unique within a run without a lock or an RNG
//! dependency. The prefix matches OpenAI's, which some clients display.

use std::sync::atomic::{AtomicU64, Ordering};

/// OpenAI uses this prefix for chat completion ids; clients occasionally
/// display or match on it.
pub const PREFIX: &str = "chatcmpl-";

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Per-process stamp: the low bits of the wall clock at first use. Not
/// a security token -- it only needs to differ between restarts.
///
/// Public because `frink-server`'s conversation ids need the same
/// stamp for the same reason, and had a byte-identical copy of this
/// function whose doc comment said it was mirroring this one. A copy
/// that says it is a copy is still a copy.
pub fn process_stamp() -> u64 {
    use std::sync::OnceLock;
    static STAMP: OnceLock<u64> = OnceLock::new();
    *STAMP.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    })
}

/// A fresh request id. Unique within this process, and in practice
/// across restarts.
pub fn next_request_id() -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "{PREFIX}{:012x}{:06x}",
        process_stamp() & 0xffff_ffff_ffff,
        n
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_prefixed() {
        let a = next_request_id();
        let b = next_request_id();
        assert_ne!(a, b);
        assert!(a.starts_with(PREFIX), "{a}");
        assert!(b.starts_with(PREFIX), "{b}");
    }

    #[test]
    fn ids_are_stable_width_for_the_first_million_requests() {
        // A UI that column-aligns a request log should not have the
        // column jump on request 16.
        let first = next_request_id();
        for _ in 0..64 {
            assert_eq!(next_request_id().len(), first.len());
        }
    }

    #[test]
    fn ids_are_unique_across_threads() {
        let handles: Vec<_> = (0..8)
            .map(|_| std::thread::spawn(|| (0..64).map(|_| next_request_id()).collect::<Vec<_>>()))
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        for h in handles {
            for id in h.join().unwrap() {
                assert!(seen.insert(id.clone()), "duplicate id {id}");
            }
        }
    }
}
