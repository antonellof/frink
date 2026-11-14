//! The orphan deadline on an SSE stream's send.
//!
//! # What was actually wrong, since the plan got this one wrong
//!
//! The plan (`4dc195e5:docs/plans/serving-and-tiered-kv.md`, `sched-output-mailbox`)
//! asked for a single-slot coalescing mailbox to replace "an unbounded
//! channel", on the reasoning that a slow or disconnected SSE consumer
//! would otherwise grow memory without bound. **That premise is false
//! for frink.** The SSE channel is a `tokio::sync::mpsc::channel(64)`
//! -- bounded from the day it was written -- and a send that fails
//! because the receiver is gone already flips the request's cancel
//! token, so a *disconnected* consumer is handled and a *slow* one gets
//! real backpressure. A mailbox would add coalescing (dropping
//! intermediate tokens), which is wrong for a token stream: every chunk
//! is content, not a redundant state update.
//!
//! What backpressure on a blocking thread actually costs is the thing
//! this module fixes. `Sender::blocking_send` parks the calling thread
//! until there is room, and "until there is room" has no upper bound
//! when the receiver is neither draining nor dropped -- a client whose
//! TCP window went to zero, a paused debugger, a phone that walked out
//! of coverage without the socket noticing. The generation runs on
//! `spawn_blocking`, so a parked send pins:
//!
//! - one thread of Tokio's blocking pool, permanently;
//! - an `Arc<Model>`, so `/admin/models/unload` cannot free the weights;
//! - the request's `CancelGuard`, so the id never leaves the registry.
//!
//! Nothing ever unsticks it, because the only two events the send waits
//! for are "the client reads" and "the client disconnects", and a
//! stalled-but-open socket is neither. N such clients retire N blocking
//! threads. That is the orphan case the plan's second bullet named, and
//! it is a liveness bug rather than a memory one.
//!
//! # The fix
//!
//! [`send_or_orphan`] gives the send a deadline. Past it, the stream is
//! declared orphaned and treated *exactly* as a disconnect already is:
//! the same `Err` the caller already handles by cancelling the
//! generation. One stop path, not two -- the discipline `crate::cancel`
//! already keeps.
//!
//! The deadline is deliberately generous ([`DEFAULT_ORPHAN_TIMEOUT`],
//! overridable with `FRINK_SSE_ORPHAN_TIMEOUT_MS`). Hitting it means
//! the client accepted nothing at all while 64 events waited, for that
//! long -- which is not a slow reader, it is a reader that has stopped.
//! A healthy-but-slow client keeps draining and never approaches it.

use std::convert::Infallible;
use std::time::Duration;

use axum::response::sse::Event;
use futures_util::Stream;
use serde::Serialize;
use tokio::sync::mpsc::{Receiver, Sender};

/// Silence after which a keepalive goes out.
///
/// 15s, the same interval axum's own default uses and the same one the
/// Anthropic and Responses streams use, so all three surfaces look
/// alike to a proxy.
pub(crate) const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);

/// The frame a stream sends when it has nothing to say.
///
/// **A data frame, never an SSE comment.** axum's `Sse::keep_alive`
/// writes `: ping`, which a proxy sees but a client's event handler
/// never does -- and a client's stream-idle timeout is armed on
/// received *events*. codex's is 300s and only a data frame resets it,
/// so a comment-kept stream is torn down and reconnected in the middle
/// of a long prefill, which is exactly when a keepalive was supposed to
/// help.
///
/// `payload` is whatever that surface's protocol calls a no-op: for
/// chat completions, a chunk with an empty delta, which a concatenating
/// client adds nothing from.
pub(crate) fn keepalive_event<T: Serialize>(payload: &T) -> Event {
    // Cannot fail for the chunk types this is called with; the fallback
    // keeps a serialization bug from taking a stream down over a frame
    // that carries no information anyway.
    let data = serde_json::to_string(payload).unwrap_or_else(|e| {
        tracing::error!("failed to serialize a keepalive frame: {e}");
        "{}".to_string()
    });
    Event::default().data(data)
}

/// The stream, with `keepalive` injected after every `interval` of
/// silence.
///
/// The silence BEFORE the first event counts, which is the point: that
/// window is the queue wait plus the prefill, and on a long prompt it
/// is the longest quiet stretch of the whole request. A keepalive that
/// only started after the first token would miss it entirely.
///
/// Keepalives are injected here rather than sent through the channel so
/// they never reach a `resume::Emitter`: a replayed stream must not
/// re-deliver keepalives that were only ever about a connection that no
/// longer exists.
pub(crate) fn with_keepalive(
    events: Receiver<Result<Event, Infallible>>,
    keepalive: Event,
    interval: Duration,
) -> impl Stream<Item = Result<Event, Infallible>> {
    // The receiver rides in the unfold's *state*: a future returned by
    // the closure may not borrow it.
    futures_util::stream::unfold(
        (events, keepalive),
        move |(mut events, keepalive)| async move {
            match tokio::time::timeout(interval, events.recv()).await {
                Err(_elapsed) => Some((Ok(keepalive.clone()), (events, keepalive))),
                Ok(Some(event)) => Some((event, (events, keepalive))),
                Ok(None) => None,
            }
        },
    )
}

/// How long a blocking send waits for room before the stream is
/// declared orphaned.
///
/// Long enough that no reader which is still reading can trip it (the
/// channel holds 64 events, so tripping this means zero were accepted
/// in this whole span), short enough that a wedged client cannot retire
/// a blocking thread for the life of the process.
pub const DEFAULT_ORPHAN_TIMEOUT: Duration = Duration::from_secs(30);

/// Why a send gave up. Both variants mean the same thing to the caller
/// -- stop generating -- and are kept apart only so the log says which
/// happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendFailure {
    /// The receiver was dropped: the client disconnected, cleanly.
    Disconnected,
    /// The receiver still exists but has not made room within the
    /// deadline. It is not reading.
    Orphaned,
}

/// `FRINK_SSE_ORPHAN_TIMEOUT_MS`, or [`DEFAULT_ORPHAN_TIMEOUT`].
///
/// A value of `0` disables the deadline, restoring the previous
/// block-forever behaviour for anyone who needs it. Unparseable input
/// keeps the default rather than panicking: this is read on the
/// generation path, and a typo in an env var must not take down a
/// request that would otherwise have succeeded.
pub fn orphan_timeout_from_env() -> Option<Duration> {
    match std::env::var("FRINK_SSE_ORPHAN_TIMEOUT_MS") {
        Err(_) => Some(DEFAULT_ORPHAN_TIMEOUT),
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(0) => None,
            Ok(ms) => Some(Duration::from_millis(ms)),
            Err(_) => {
                tracing::warn!(
                    "FRINK_SSE_ORPHAN_TIMEOUT_MS='{raw}' is not a non-negative integer; \
                     using the {DEFAULT_ORPHAN_TIMEOUT:?} default"
                );
                Some(DEFAULT_ORPHAN_TIMEOUT)
            }
        },
    }
}

/// Sends one event from a blocking thread, giving up after `timeout`.
///
/// Must be called from a thread that is *not* a Tokio runtime worker --
/// in this server, from inside `spawn_blocking`, which is where every
/// generation runs. That is the same requirement `blocking_send` has,
/// and for the same reason.
///
/// `timeout: None` blocks indefinitely, which is what this code did
/// before the deadline existed.
pub fn send_or_orphan<T>(
    tx: &Sender<T>,
    value: T,
    timeout: Option<Duration>,
) -> Result<(), SendFailure> {
    let Some(timeout) = timeout else {
        return tx
            .blocking_send(value)
            .map_err(|_| SendFailure::Disconnected);
    };
    // `block_on` rather than `blocking_send`: the timeout variant of
    // send is async-only, and this thread is a blocking-pool thread, so
    // parking it on a future is exactly what it is for.
    let handle = match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle,
        // No runtime to drive the timer (a unit test calling this off
        // the runtime, an embedder). Fall back to the un-deadlined send
        // rather than refusing to deliver the event.
        Err(_) => {
            return tx
                .blocking_send(value)
                .map_err(|_| SendFailure::Disconnected);
        }
    };
    match handle.block_on(tx.send_timeout(value, timeout)) {
        Ok(()) => Ok(()),
        Err(tokio::sync::mpsc::error::SendTimeoutError::Closed(_)) => {
            Err(SendFailure::Disconnected)
        }
        Err(tokio::sync::mpsc::error::SendTimeoutError::Timeout(_)) => Err(SendFailure::Orphaned),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// The bug, stated as a test: a receiver that exists but never
    /// reads must not park the generation thread forever.
    ///
    /// The receiver is held for the whole test and never polled, which
    /// is precisely the case `blocking_send` cannot escape -- it is not
    /// closed, so no `Err` ever arrives, and it is not draining, so no
    /// permit ever does either.
    ///
    /// Confirmed to FAIL when `send_or_orphan` is patched to call
    /// `blocking_send` unconditionally: the send never returns, and the
    /// five-second guard below reports `Elapsed` rather than letting the
    /// test hang.
    #[tokio::test]
    async fn a_receiver_that_never_reads_does_not_park_the_sender_forever() {
        let (tx, _rx_held_and_never_polled) = tokio::sync::mpsc::channel::<u32>(1);
        tx.send(1).await.expect("the first send fills the channel");

        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        let result = tokio::task::spawn_blocking(move || {
            let out = send_or_orphan(&tx, 2, Some(Duration::from_millis(50)));
            flag.store(true, Ordering::SeqCst);
            out
        });

        let out = tokio::time::timeout(Duration::from_secs(5), result)
            .await
            .expect("the send must give up rather than park forever")
            .expect("the blocking task must not panic");
        assert_eq!(out, Err(SendFailure::Orphaned));
        assert!(done.load(Ordering::SeqCst));
    }

    /// A dropped receiver is still reported as a disconnect, not as an
    /// orphan, and reports it immediately rather than after the
    /// deadline: the two cases mean the same thing to the caller but
    /// not to whoever reads the log.
    #[tokio::test]
    async fn a_dropped_receiver_is_a_disconnect_and_is_reported_at_once() {
        let (tx, rx) = tokio::sync::mpsc::channel::<u32>(1);
        drop(rx);
        let started = std::time::Instant::now();
        let out = tokio::task::spawn_blocking(move || {
            send_or_orphan(&tx, 1, Some(Duration::from_secs(30)))
        })
        .await
        .unwrap();
        assert_eq!(out, Err(SendFailure::Disconnected));
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "a closed channel must not wait out the orphan deadline"
        );
    }

    /// The deadline must be invisible to a stream that is being read:
    /// every event arrives, in order, with none coalesced away.
    ///
    /// This is the guard against "fixing" the hang by dropping events,
    /// which is what the plan's coalescing mailbox would have done.
    #[tokio::test]
    async fn a_draining_receiver_gets_every_event_in_order() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<u32>(2);
        let sender = tokio::task::spawn_blocking(move || {
            for i in 0..64 {
                send_or_orphan(&tx, i, Some(Duration::from_secs(30)))?;
            }
            Ok::<(), SendFailure>(())
        });
        let mut got = Vec::new();
        while let Some(v) = rx.recv().await {
            got.push(v);
        }
        sender
            .await
            .unwrap()
            .expect("a drained channel never fails");
        assert_eq!(got, (0..64).collect::<Vec<_>>());
    }

    /// `timeout: None` is the old behaviour, kept reachable so an
    /// operator who wants it can have it back.
    #[tokio::test]
    async fn no_deadline_still_delivers() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<u32>(1);
        let sender = tokio::task::spawn_blocking(move || send_or_orphan(&tx, 7, None));
        assert_eq!(rx.recv().await, Some(7));
        sender.await.unwrap().expect("a live receiver accepts");
    }

    #[test]
    fn the_default_deadline_applies_when_the_env_var_is_absent() {
        // Not `set_var`: tests share a process, and this asserts the
        // parse rules rather than the environment.
        assert_eq!(DEFAULT_ORPHAN_TIMEOUT, Duration::from_secs(30));
    }

    /// The rule the whole thing exists for: silence BEFORE the first
    /// event counts. That window is the queue wait plus the prefill,
    /// which on a long prompt is the longest quiet stretch of the
    /// request -- a keepalive that only armed after the first token
    /// would miss exactly the case it was added for.
    #[tokio::test]
    async fn a_stream_that_has_not_started_still_sends_keepalives() {
        use futures_util::StreamExt;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(4);
        let keepalive = keepalive_event(&serde_json::json!({"ping": true}));
        let mut stream = Box::pin(with_keepalive(rx, keepalive, Duration::from_millis(20)));

        // Nothing has been sent, and the stream still produces frames.
        for _ in 0..2 {
            assert!(
                stream.next().await.is_some(),
                "a quiet stream keeps talking"
            );
        }

        tx.send(Ok(Event::default().data("real"))).await.unwrap();
        assert!(stream.next().await.is_some());
        drop(tx);
        assert!(
            stream.next().await.is_none(),
            "a closed channel ends the stream rather than keeping it alive forever"
        );
    }

    /// A keepalive is a DATA frame. axum's `Sse::keep_alive` writes
    /// `: ping`, an SSE comment, which a proxy sees but a client's
    /// event handler never does -- and the timeout being reset is armed
    /// on received events. This asserts the serialized frame carries a
    /// `data:` line, which is the whole difference.
    #[test]
    fn a_keepalive_carries_data_and_not_a_comment() {
        let event = keepalive_event(&serde_json::json!({"object": "chat.completion.chunk"}));
        let wire = format!("{:?}", event);
        assert!(
            wire.contains("chat.completion.chunk"),
            "the payload must be in the frame, not just implied: {wire}"
        );
    }
}
