//! Live serving counters: what is admitted right now, what became of
//! everything that is not, and how fast the decode paths are going.
//!
//! [`ServingFeed`] is the one handle both decode paths write through:
//! the private loop (`generate::generate`, `generate_engine`) and the
//! continuous batch worker (`serving::batch`). Admission, abort and the
//! terminal reply are counted once per generation, at
//! `crate::run_generation_emit`, which every generating route passes
//! through; prefilled and decoded tokens are counted where they are
//! computed, so the two rates are true on both paths.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use super::rate::RateWindow;

/// Live serving counters: what is in flight, what completed, and at what
/// rate.
///
/// The `aborting` set is the part worth reading twice. A request that
/// has been sent an abort stays *active* until its terminal reply
/// actually arrives, and then completes the count as an abort rather
/// than as a completion. Dropping it from the in-flight set at dispatch
/// would let `requests.active` read 0 while a request was still
/// producing tokens, and counting it as completed would make a
/// cancelled request look like a clean one.
#[derive(Debug, Clone)]
pub struct ServingStats {
    inflight: BTreeSet<u64>,
    aborting: BTreeSet<u64>,
    completed: u64,
    aborted: u64,
    decode: RateWindow,
    prefill: RateWindow,
}

impl Default for ServingStats {
    fn default() -> Self {
        Self::new(5_000)
    }
}

impl ServingStats {
    pub fn new(window_ms: u64) -> Self {
        ServingStats {
            inflight: BTreeSet::new(),
            aborting: BTreeSet::new(),
            completed: 0,
            aborted: 0,
            decode: RateWindow::new(window_ms),
            prefill: RateWindow::new(window_ms),
        }
    }

    /// One request admitted. Re-admitting an id clears any abort left
    /// over from a previous life of that slot.
    pub fn on_admitted(&mut self, uid: u64) {
        self.inflight.insert(uid);
        self.aborting.remove(&uid);
    }

    /// An abort was dispatched. The request stays active until its
    /// terminal reply arrives.
    pub fn on_abort(&mut self, uid: u64) {
        if self.inflight.contains(&uid) {
            self.aborting.insert(uid);
        }
    }

    /// `tokens` prompt tokens were computed, the work having started at
    /// `at_ms`.
    pub fn on_prefill(&mut self, at_ms: u64, tokens: u64) {
        self.prefill.record(at_ms, tokens);
    }

    /// `tokens` completion tokens were committed at `at_ms`.
    pub fn on_decode(&mut self, at_ms: u64, tokens: u64) {
        self.decode.record(at_ms, tokens);
    }

    /// A terminal reply arrived for `uid`.
    pub fn on_finished(&mut self, uid: u64) {
        if !self.inflight.remove(&uid) {
            return;
        }
        if self.aborting.remove(&uid) {
            self.aborted += 1;
        } else {
            self.completed += 1;
        }
    }

    /// `uid` ended in an error, with no reply to count. It leaves the
    /// in-flight set and is neither a completion nor an abort: a
    /// refused or failed generation counted as completed would make a
    /// failing server look like a busy one.
    pub fn on_failed(&mut self, uid: u64) {
        self.inflight.remove(&uid);
        self.aborting.remove(&uid);
    }

    pub fn active(&self) -> usize {
        self.inflight.len()
    }

    pub fn completed(&self) -> u64 {
        self.completed
    }

    pub fn aborted(&self) -> u64 {
        self.aborted
    }

    pub fn decode_tokens_per_second(&mut self, now_ms: u64) -> f64 {
        self.decode.tokens_per_second(now_ms)
    }

    pub fn prefill_tokens_per_second(&mut self, now_ms: u64) -> f64 {
        self.prefill.tokens_per_second(now_ms)
    }

    /// Tokens in the (prefill, decode) windows, read without evicting.
    #[cfg(test)]
    pub(crate) fn window_tokens(&self) -> (u64, u64) {
        (self.prefill.tokens(), self.decode.tokens())
    }
}

/// The server's handle on its [`ServingStats`], with the clock every
/// sample is stamped by and every read is taken at.
///
/// One clock for both, owned here: a rate whose samples and reads came
/// from two `Instant`s would be off by the gap between them. Cheap to
/// clone (two `Arc`s and an `Instant`), so it rides on
/// `GenerationParams` into whichever decode path serves the request,
/// the batch worker's thread included.
#[derive(Debug, Clone)]
pub(crate) struct ServingFeed {
    stats: Arc<Mutex<ServingStats>>,
    origin: Instant,
    next_uid: Arc<AtomicU64>,
}

impl Default for ServingFeed {
    fn default() -> Self {
        ServingFeed {
            stats: Arc::new(Mutex::new(ServingStats::default())),
            origin: Instant::now(),
            next_uid: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl ServingFeed {
    /// The counters, for a reader. Poison is recovered: these are
    /// counters, and a panicked writer leaves them no less true.
    pub(crate) fn lock(&self) -> MutexGuard<'_, ServingStats> {
        self.stats.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Milliseconds on this feed's clock at `at`.
    fn ms(&self, at: Instant) -> u64 {
        at.saturating_duration_since(self.origin)
            .as_millis()
            .min(u64::MAX as u128) as u64
    }

    /// Milliseconds on this feed's clock now: what a read of the rates
    /// passes as `now_ms`.
    pub(crate) fn now_ms(&self) -> u64 {
        self.ms(Instant::now())
    }

    /// One generation accepted by a decode path. A cancel of `cancel`
    /// is recorded as an abort when it is dispatched (or at once, if it
    /// already was); the request stays active until the returned
    /// admission is finished or dropped.
    pub(crate) fn admit(&self, cancel: Option<&crate::cancel::CancelToken>) -> Admission {
        let uid = self.next_uid.fetch_add(1, Ordering::Relaxed);
        self.lock().on_admitted(uid);
        if let Some(token) = cancel {
            let feed = self.clone();
            token.on_cancel(move || feed.lock().on_abort(uid));
        }
        Admission {
            feed: self.clone(),
            uid,
            answered: false,
        }
    }

    /// `tokens` prompt tokens computed, the work having started at
    /// `started`. Stamped at the START so a poll right after one prefill
    /// divides by the time it took, not by the time since it ended.
    pub(crate) fn prefilled(&self, started: Instant, tokens: usize) {
        let at = self.ms(started);
        self.lock().on_prefill(at, tokens as u64);
    }

    /// `tokens` completion tokens committed just now.
    pub(crate) fn decoded(&self, tokens: usize) {
        let at = self.now_ms();
        self.lock().on_decode(at, tokens as u64);
    }
}

/// One admitted generation. [`Admission::answered`] records its terminal
/// reply; dropping it unanswered (an error, or a panic unwinding the
/// decode) records a failure, so no path can leave it in flight.
pub(crate) struct Admission {
    feed: ServingFeed,
    uid: u64,
    answered: bool,
}

impl Admission {
    /// The generation produced its reply: a completion, or an abort if
    /// one was dispatched before it.
    pub(crate) fn answered(mut self) {
        self.answered = true;
        self.feed.lock().on_finished(self.uid);
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        if !self.answered {
            self.feed.lock().on_failed(self.uid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_is_counted_once_it_reaches_a_terminal_reply() {
        let mut stats = ServingStats::new(1_000);
        stats.on_admitted(1);
        stats.on_admitted(2);
        assert_eq!(stats.active(), 2);
        stats.on_finished(1);
        assert_eq!(stats.completed(), 1);
        assert_eq!(stats.active(), 1);
    }

    /// The abort rule, which is what `requests.active` rests on:
    /// dispatching an abort does not make a request inactive, and when
    /// it does end it is not a completion.
    #[test]
    fn an_aborted_request_stays_active_until_its_terminal_reply() {
        let mut stats = ServingStats::new(1_000);
        stats.on_admitted(7);
        stats.on_abort(7);
        assert_eq!(stats.active(), 1, "an abort is dispatched, not applied");
        assert_eq!(stats.completed(), 0);

        // Tokens racing the abort still count toward the rate.
        stats.on_decode(10, 3);
        assert_eq!(stats.decode_tokens_per_second(11), 3000.0);
        stats.on_finished(7);
        assert_eq!(stats.active(), 0);
        assert_eq!(stats.completed(), 0);
        assert_eq!(stats.aborted(), 1);
    }

    #[test]
    fn a_terminal_reply_for_an_unknown_request_changes_nothing() {
        let mut stats = ServingStats::new(1_000);
        stats.on_admitted(1);
        stats.on_finished(1);
        stats.on_finished(1);
        assert_eq!(stats.completed(), 1);
        assert_eq!(stats.active(), 0);
    }

    /// A generation that errors leaves the in-flight set and is counted
    /// as nothing, and dropping an admission unanswered is that error:
    /// a decode that panics or returns `Err` cannot leave a request
    /// active forever.
    #[test]
    fn an_admission_dropped_unanswered_is_neither_completed_nor_aborted() {
        let feed = ServingFeed::default();
        let failed = feed.admit(None);
        assert_eq!(feed.lock().active(), 1);
        drop(failed);
        let answered = feed.admit(None);
        answered.answered();
        let stats = feed.lock();
        assert_eq!(stats.active(), 0);
        assert_eq!(stats.completed(), 1);
        assert_eq!(stats.aborted(), 0);
    }

    /// A cancel is an abort from the moment it is dispatched, whether
    /// that was before the admission or during it.
    #[test]
    fn a_cancel_before_or_during_a_generation_counts_it_as_aborted() {
        let feed = ServingFeed::default();
        let early = crate::cancel::CancelToken::new();
        early.cancel();
        feed.admit(Some(&early)).answered();
        let late = crate::cancel::CancelToken::new();
        let admission = feed.admit(Some(&late));
        late.cancel();
        assert_eq!(feed.lock().active(), 1, "still producing its reply");
        admission.answered();
        let untouched = crate::cancel::CancelToken::new();
        feed.admit(Some(&untouched)).answered();
        let stats = feed.lock();
        assert_eq!((stats.completed(), stats.aborted()), (1, 2));
    }
}
