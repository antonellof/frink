//! Resumable streams: the replay buffer behind `id:` / `retry:` /
//! `Last-Event-ID`, and the polling fallback that reads the same buffer.
//!
//! # Why this is opt-in
//!
//! `bc5074aa:docs/plans/on-hold/frink-ui.md` asks for SSE replay *and* records why it
//! could not simply be added: two-tier cancellation made a dropped
//! socket **stop** the generation, so by the time a client reconnected
//! there was nothing left running to resume into. Emitting an `id:`
//! anyway would have been a promise the server could not keep.
//!
//! That tension is a design call, and the call made here is: **the
//! client decides, per request.** `stream_resumable: true` on
//! `/v1/chat/completions` means "keep going if I disappear, I may come
//! back"; anything else keeps today's behaviour exactly, socket-drop
//! cancellation included. Neither answer is right for both callers -- a
//! browser tab that navigates away wants the CPU back, and a browser
//! tab whose proxy dropped a 90-second generation wants the answer it
//! already paid for -- so the request says which one it is.
//!
//! A resumable request is not left without a stop path: `POST
//! /v1/cancel` still ends it, and that is the tier a page-unload
//! `keepalive` POST can actually reach.
//!
//! # What replay buys, and what bounds it
//!
//! Every event of a resumable stream is stored here as the exact `data:`
//! payload that went out, numbered from zero. A reconnect names the last
//! id it saw and gets everything after it, then continues live. The
//! polling fallback reads the same buffer over plain JSON, which is the
//! answer to the failure this item is named for: a reverse proxy that
//! buffers `text/event-stream` cannot buffer a short JSON response.
//!
//! Two bounds, and both fail closed rather than quietly:
//!
//! - **The window is finite** ([`REPLAY_BYTES`]). Past it the oldest
//!   events are dropped and `first_index` moves up. A reader that asks
//!   for something evicted gets `410 Gone` -- never a stream that
//!   silently skips the middle of an answer, which the client could not
//!   detect.
//! - **Retention is finite** ([`RETAIN_AFTER_FINISH`], [`MAX_SLOTS`]).
//!   A finished stream is kept briefly for a late reconnect and then
//!   dropped, so a long-lived server does not accumulate every answer it
//!   ever produced in memory.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::AppState;

/// Replay window per stream. An answer longer than this is still
/// delivered live; only its *beginning* stops being replayable.
const REPLAY_BYTES: usize = 1 << 20;

/// How long a finished stream stays reconnectable. Long enough for a
/// proxy hiccup and a browser's retry, short enough that a busy server
/// is not holding yesterday's answers.
const RETAIN_AFTER_FINISH: Duration = Duration::from_secs(120);

/// Hard cap on remembered streams, so a burst of resumable requests
/// cannot grow this without bound between sweeps.
const MAX_SLOTS: usize = 64;

/// The `retry:` value stated once, on the first event of a resumable
/// stream: how long a client should wait before reconnecting.
pub(crate) const RECONNECT_DELAY: Duration = Duration::from_millis(1500);

/// Longest a poll request parks waiting for the next event.
///
/// The fallback is a poll, but a poll that returns instantly with
/// nothing would either burn requests or lag the answer by its own
/// interval. Parking briefly makes it read like a stream while staying
/// a plain short-lived JSON response -- which is exactly what a
/// buffering proxy passes through unharmed.
const POLL_WAIT: Duration = Duration::from_secs(10);

/// Longest a resumed SSE connection waits for the next event before
/// yielding to the keep-alive.
const RESUME_WAIT: Duration = Duration::from_secs(5);

/// Empty polls before a replay stream sends a keepalive.
///
/// Three of them at [`RESUME_WAIT`] is the same 15s of silence
/// [`crate::sse::KEEPALIVE_INTERVAL`] allows on a live stream, counted
/// in polls rather than off a clock so it is deterministic and
/// testable.
const RESUME_KEEPALIVE_POLLS: u32 = 3;

/// One stored event: the `data:` payload exactly as it went out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct StoredEvent {
    pub(crate) index: u64,
    pub(crate) data: String,
}

#[derive(Default)]
struct SlotState {
    events: VecDeque<StoredEvent>,
    /// Index of the oldest event still held. Anything below this has
    /// been evicted and can never be served again.
    first_index: u64,
    next_index: u64,
    bytes: usize,
    finished: bool,
    finished_at: Option<Instant>,
}

/// The replay buffer for one generation.
pub(crate) struct StreamSlot {
    request_id: String,
    created_at: Instant,
    state: Mutex<SlotState>,
    /// Woken on every append and on finish. Readers register interest
    /// *before* inspecting the buffer, so an event that lands in
    /// between is never missed.
    notify: tokio::sync::Notify,
}

/// What one read of the buffer found.
#[derive(Debug, PartialEq)]
pub(crate) struct ReadResult {
    pub(crate) events: Vec<StoredEvent>,
    pub(crate) next_index: u64,
    pub(crate) finished: bool,
    /// The requested position had already been evicted. The caller must
    /// refuse rather than serve the events it does have: a stream
    /// missing its middle is worse than one that stopped, because
    /// nothing downstream can tell.
    pub(crate) lost: bool,
}

impl StreamSlot {
    fn new(request_id: &str) -> Self {
        StreamSlot {
            request_id: request_id.to_string(),
            created_at: Instant::now(),
            state: Mutex::new(SlotState::default()),
            notify: tokio::sync::Notify::new(),
        }
    }

    pub(crate) fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Appends one event and returns the index it was given.
    ///
    /// Called from the blocking generation thread, so it must not
    /// await: `Notify::notify_waiters` is sync and that is the whole
    /// interaction with the async side.
    pub(crate) fn push(&self, data: String) -> u64 {
        let index = {
            let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let index = st.next_index;
            st.next_index += 1;
            st.bytes += data.len();
            st.events.push_back(StoredEvent { index, data });
            while st.bytes > REPLAY_BYTES && st.events.len() > 1 {
                if let Some(dropped) = st.events.pop_front() {
                    st.bytes -= dropped.data.len();
                    st.first_index = dropped.index + 1;
                }
            }
            index
        };
        self.notify.notify_waiters();
        index
    }

    /// Marks the generation over. Readers that catch up now stop rather
    /// than waiting for an event that will never come.
    pub(crate) fn finish(&self) {
        {
            let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
            st.finished = true;
            st.finished_at = Some(Instant::now());
        }
        self.notify.notify_waiters();
    }

    /// Non-blocking look at what is available from `cursor`.
    fn read_now(&self, cursor: u64) -> ReadResult {
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if cursor < st.first_index {
            return ReadResult {
                events: Vec::new(),
                next_index: st.first_index,
                finished: st.finished,
                lost: true,
            };
        }
        let events: Vec<StoredEvent> = st
            .events
            .iter()
            .filter(|e| e.index >= cursor)
            .cloned()
            .collect();
        ReadResult {
            next_index: events.last().map(|e| e.index + 1).unwrap_or(cursor),
            events,
            finished: st.finished,
            lost: false,
        }
    }

    /// Events from `cursor` on, waiting up to `wait` for the first one.
    ///
    /// Returns immediately when anything is already buffered, when the
    /// position is lost, or when the stream has finished.
    pub(crate) async fn read_from(&self, cursor: u64, wait: Duration) -> ReadResult {
        let deadline = Instant::now() + wait;
        loop {
            // Registered before the buffer is inspected. The other order
            // is a lost wakeup: an event appended between the read and
            // the await would leave this reader parked until the
            // timeout, which on a fast stream is every reader.
            let notified = self.notify.notified();
            let result = self.read_now(cursor);
            if result.lost || result.finished || !result.events.is_empty() {
                return result;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return result;
            }
            if tokio::time::timeout(remaining, notified).await.is_err() {
                return self.read_now(cursor);
            }
        }
    }

    fn droppable_at(&self, now: Instant) -> bool {
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        st.finished_at
            .is_some_and(|at| now.duration_since(at) >= RETAIN_AFTER_FINISH)
    }
}

/// Every resumable stream this process is holding open or remembering.
#[derive(Default)]
pub(crate) struct StreamRegistry {
    slots: Mutex<HashMap<String, Arc<StreamSlot>>>,
}

impl StreamRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Starts remembering one generation, and sweeps what is stale.
    ///
    /// Sweeping here rather than on a timer keeps this module free of a
    /// background task: a registry nobody is adding to is a registry
    /// nobody is reading either, and it holds at most [`MAX_SLOTS`]
    /// finished streams until the next request.
    pub(crate) fn register(&self, request_id: &str) -> Arc<StreamSlot> {
        let slot = Arc::new(StreamSlot::new(request_id));
        let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        let now = Instant::now();
        slots.retain(|_, s| !s.droppable_at(now));
        while slots.len() >= MAX_SLOTS {
            // Oldest first, and only ones that have finished -- evicting
            // a live stream would break the connection currently reading
            // it, which is worse than holding one extra buffer.
            let victim = slots
                .values()
                .filter(|s| s.state.lock().unwrap_or_else(|p| p.into_inner()).finished)
                .min_by_key(|s| s.created_at)
                .map(|s| s.request_id.clone());
            match victim {
                Some(id) => {
                    slots.remove(&id);
                }
                None => break,
            }
        }
        slots.insert(request_id.to_string(), Arc::clone(&slot));
        slot
    }

    pub(crate) fn get(&self, request_id: &str) -> Option<Arc<StreamSlot>> {
        self.slots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(request_id)
            .cloned()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.slots.lock().unwrap_or_else(|p| p.into_inner()).len()
    }
}

/// Stamps every event of one generation, and mirrors it into the
/// replay buffer when there is one.
///
/// One object rather than a flag threaded through five send sites: the
/// rule "an `id:` only where a buffer exists" then holds by
/// construction instead of by five correct calls.
pub(crate) struct Emitter {
    slot: Option<Arc<StreamSlot>>,
}

impl Emitter {
    pub(crate) fn new(slot: Option<Arc<StreamSlot>>) -> Self {
        Emitter { slot }
    }

    /// Whether a dropped receiver should be allowed to stop the work.
    /// See this module's doc comment.
    pub(crate) fn is_resumable(&self) -> bool {
        self.slot.is_some()
    }

    fn emit(&self, data: String) -> Event {
        match &self.slot {
            None => sse_event(data, None, false),
            Some(slot) => {
                let index = slot.push(data.clone());
                sse_event(data, Some(event_id(slot.request_id(), index)), index == 0)
            }
        }
    }

    /// One JSON chunk, serialized once and used for both the wire and
    /// the buffer -- so a replayed event is byte-identical to the one
    /// the client missed, rather than a re-serialization of it.
    pub(crate) fn event<T: Serialize>(&self, payload: &T) -> Event {
        // `to_string` on a struct that derives Serialize and holds no
        // map with non-string keys cannot fail; the fallback keeps a
        // serialization bug from taking the whole stream down with a
        // panic on the generation thread.
        let data = serde_json::to_string(payload).unwrap_or_else(|e| {
            tracing::error!("failed to serialize a stream chunk: {e}");
            "{}".to_string()
        });
        self.emit(data)
    }

    /// The `[DONE]` sentinel, buffered like any other event so a
    /// resumed or polled reader sees the same end of stream.
    pub(crate) fn done(&self) -> Event {
        self.emit("[DONE]".to_string())
    }

    /// Closes the replay buffer. A no-op for a non-resumable stream.
    fn finish(&self) {
        if let Some(slot) = &self.slot {
            slot.finish();
        }
    }
}

/// The buffer is closed by dropping the emitter, not by remembering to
/// call `finish()`.
///
/// A generation thread that panics would otherwise leave a slot marked
/// live forever: readers would park on it until their timeouts, and the
/// registry only ever evicts *finished* slots, so it could never be
/// reclaimed. Same reasoning as the `TaskGuard` in `tasks` and the
/// cancel guard in `cancel` -- nothing awaits a `spawn_blocking` handle,
/// so a panic has to be survivable by construction.
impl Drop for Emitter {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Stamps one SSE event with the id a reconnect will name.
///
/// `id:` and `retry:` are emitted **only** for a resumable stream. An
/// id on a stream with no replay buffer behind it tells a client it may
/// reconnect into something that no longer exists, which is exactly the
/// promise this module was written rather than make.
pub(crate) fn sse_event(data: String, id: Option<String>, first: bool) -> Event {
    let mut event = Event::default();
    if let Some(id) = id {
        event = event.id(id);
        if first {
            event = event.retry(RECONNECT_DELAY);
        }
    }
    event.data(data)
}

/// The id one event is named by on the wire.
///
/// Qualified by the request so a `Last-Event-ID` from some *other*
/// stream cannot be mistaken for a position in this one -- browsers
/// resend the last id they saw, and a bare `7` would be a valid
/// position in every stream at once.
pub(crate) fn event_id(request_id: &str, index: u64) -> String {
    format!("{request_id}:{index}")
}

/// Reads a `Last-Event-ID` back, insisting it names *this* stream.
///
/// Returns the index of the next event wanted. `None` means the id does
/// not belong here, which is refused rather than rounded down to zero:
/// replaying an entire other answer would be a silent, confident lie.
pub(crate) fn cursor_from_event_id(request_id: &str, last_event_id: &str) -> Option<u64> {
    let (id, index) = last_event_id.rsplit_once(':')?;
    if id != request_id {
        return None;
    }
    index.parse::<u64>().ok().map(|i| i + 1)
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResumeQuery {
    /// Same value as the `Last-Event-ID` header, for clients that
    /// cannot set headers (an `EventSource`, most obviously). The
    /// header wins when both are present.
    #[serde(default)]
    last_event_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PollQuery {
    /// First event index wanted. Absent means "from the beginning",
    /// which is what a fallback that never saw the stream needs.
    #[serde(default)]
    from: Option<u64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PollResponse {
    pub(crate) request_id: String,
    pub(crate) events: Vec<StoredEvent>,
    /// Where to ask from next. Stated by the server so the client never
    /// has to derive it from the last event it happened to render.
    pub(crate) next_index: u64,
    /// `true` once the generation is over *and* every event has been
    /// handed out, so a client stops polling on this alone.
    pub(crate) done: bool,
}

fn unknown_stream(request_id: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": {
                "message": format!(
                    "no resumable stream with request_id '{request_id}'. It was never \
                     started with stream_resumable, or it finished long enough ago to \
                     have been forgotten."
                ),
                "type": "invalid_request_error",
                "code": "stream_not_found",
            }
        })),
    )
        .into_response()
}

fn replay_window_lost(request_id: &str, first_index: u64) -> Response {
    (
        StatusCode::GONE,
        Json(serde_json::json!({
            "error": {
                "message": format!(
                    "the replay window for '{request_id}' has moved past the requested \
                     position; the earliest event still held is {first_index}. Resuming \
                     from here would skip part of the answer without either end being \
                     able to tell."
                ),
                "type": "invalid_request_error",
                "code": "replay_window_lost",
            }
        })),
    )
        .into_response()
}

/// `GET /v1/stream/{request_id}` -- reconnect into a running or
/// just-finished resumable stream.
pub(crate) async fn resume(
    State(state): State<Arc<AppState>>,
    AxumPath(request_id): AxumPath<String>,
    headers: HeaderMap,
    Query(query): Query<ResumeQuery>,
) -> Response {
    let Some(slot) = state.streams.get(&request_id) else {
        return unknown_stream(&request_id);
    };

    let last_event_id = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or(query.last_event_id);
    let cursor = match last_event_id {
        None => 0,
        Some(id) => match cursor_from_event_id(&request_id, &id) {
            Some(cursor) => cursor,
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": {
                            "message": format!(
                                "Last-Event-ID '{id}' does not name a position in \
                                 '{request_id}'"
                            ),
                            "type": "invalid_request_error",
                            "code": "bad_last_event_id",
                        }
                    })),
                )
                    .into_response();
            }
        },
    };

    // Checked before a single byte of the response goes out, because
    // after that the only way to report it is an abrupt end -- which the
    // client would read as a truncation with no reason attached.
    let first = slot.read_now(cursor);
    if first.lost {
        return replay_window_lost(&request_id, first.next_index);
    }

    let keepalive_id = request_id.clone();
    let stream = futures_util::stream::unfold(
        (slot, cursor, true, 0u32),
        move |(slot, cursor, is_first, quiet)| {
            let keepalive_id = keepalive_id.clone();
            async move {
                let result = slot.read_from(cursor, RESUME_WAIT).await;
                if result.lost {
                    // Ends the response with no `[DONE]` and no finish
                    // reason. The client's truncation rule is what surfaces
                    // it -- the same rule that catches a proxy cutting the
                    // connection, and for the same reason: the answer is
                    // incomplete and nothing may pretend otherwise.
                    return None;
                }
                if result.events.is_empty() {
                    if result.finished {
                        return None;
                    }
                    // Nothing yet, still running. Keep the connection, and
                    // after three quiet polls send a DATA frame rather than
                    // nothing at all: a client's stream-idle timeout is
                    // armed on received events, and an empty batch is not
                    // one -- see `crate::sse::with_keepalive` for why an
                    // SSE comment does not do here either.
                    //
                    // No `id:`, so it does not enter the replay sequence a
                    // `Last-Event-ID` resumes from. `choices: []` is the
                    // usage-only chunk's own shape and asserts nothing
                    // about an answer that has not started.
                    let quiet = quiet + 1;
                    let events = if quiet % RESUME_KEEPALIVE_POLLS == 0 {
                        vec![Ok(crate::sse::keepalive_event(&serde_json::json!({
                            "id": keepalive_id,
                            "object": "chat.completion.chunk",
                            "choices": [],
                        })))]
                    } else {
                        Vec::new()
                    };
                    return Some((events, (slot, cursor, is_first, quiet)));
                }
                let next = result.next_index;
                let request_id = slot.request_id().to_string();
                let events: Vec<Result<Event, std::convert::Infallible>> = result
                    .events
                    .into_iter()
                    .enumerate()
                    .map(|(i, e)| {
                        Ok(sse_event(
                            e.data,
                            Some(event_id(&request_id, e.index)),
                            is_first && i == 0,
                        ))
                    })
                    .collect();
                Some((events, (slot, next, false, 0)))
            }
        },
    );

    (
        [(
            axum::http::HeaderName::from_static("x-accel-buffering"),
            axum::http::HeaderValue::from_static("no"),
        )],
        Sse::new(futures_util::StreamExt::flat_map(
            stream,
            futures_util::stream::iter,
        )),
    )
        .into_response()
}

/// `GET /v1/stream/{request_id}/poll` -- the same replay buffer over
/// plain JSON.
///
/// The fallback the plan asks for beside every stream. A reverse proxy
/// that buffers `text/event-stream` -- nginx's default, and the default
/// of everything that copied it -- turns a token-by-token SSE response
/// into one long silence followed by the whole answer, which from the
/// browser is indistinguishable from a hung backend. It cannot do that
/// to a short JSON response that has already ended.
pub(crate) async fn poll(
    State(state): State<Arc<AppState>>,
    AxumPath(request_id): AxumPath<String>,
    Query(query): Query<PollQuery>,
) -> Response {
    let Some(slot) = state.streams.get(&request_id) else {
        return unknown_stream(&request_id);
    };
    let cursor = query.from.unwrap_or(0);
    let result = slot.read_from(cursor, POLL_WAIT).await;
    if result.lost {
        return replay_window_lost(&request_id, result.next_index);
    }
    Json(PollResponse {
        request_id,
        // Done only once the buffer is drained as well as finished, so a
        // client that reads `done` never discards events it has not been
        // given yet.
        done: result.finished && result.events.is_empty(),
        next_index: result.next_index,
        events: result.events,
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot() -> StreamSlot {
        StreamSlot::new("chatcmpl-1")
    }

    #[tokio::test]
    async fn a_reader_from_zero_gets_everything_in_order() {
        let slot = slot();
        for i in 0..3 {
            assert_eq!(slot.push(format!("event-{i}")), i);
        }
        slot.finish();
        let result = slot.read_from(0, Duration::ZERO).await;
        assert_eq!(
            result
                .events
                .iter()
                .map(|e| e.data.as_str())
                .collect::<Vec<_>>(),
            vec!["event-0", "event-1", "event-2"]
        );
        assert_eq!(result.next_index, 3);
        assert!(result.finished);
        assert!(!result.lost);
    }

    /// The replay contract: a reconnect gets what it missed and not
    /// what it already rendered. Re-sending a token the client has is
    /// the failure mode that makes replay worse than restarting.
    #[tokio::test]
    async fn a_resume_gets_only_what_came_after_the_last_seen_id() {
        let slot = slot();
        for i in 0..5 {
            slot.push(format!("event-{i}"));
        }
        let cursor = cursor_from_event_id("chatcmpl-1", "chatcmpl-1:2").unwrap();
        assert_eq!(cursor, 3);
        let result = slot.read_from(cursor, Duration::ZERO).await;
        assert_eq!(
            result
                .events
                .iter()
                .map(|e| e.data.as_str())
                .collect::<Vec<_>>(),
            vec!["event-3", "event-4"]
        );
    }

    /// A `Last-Event-ID` from another stream must not be read as a
    /// position in this one. Browsers resend the last id they saw, and
    /// a bare index would be valid in every stream at once.
    #[test]
    fn an_event_id_from_another_stream_is_refused_not_rounded_down() {
        assert_eq!(cursor_from_event_id("chatcmpl-1", "chatcmpl-1:0"), Some(1));
        assert_eq!(cursor_from_event_id("chatcmpl-1", "chatcmpl-2:9"), None);
        assert_eq!(cursor_from_event_id("chatcmpl-1", "7"), None);
        assert_eq!(cursor_from_event_id("chatcmpl-1", "chatcmpl-1:x"), None);
        assert_eq!(event_id("chatcmpl-1", 4), "chatcmpl-1:4");
    }

    /// Eviction must be loud. Serving the events that survive would
    /// hand back an answer with a hole in it that neither end can see.
    #[tokio::test]
    async fn a_position_that_has_been_evicted_is_reported_lost() {
        let slot = slot();
        let big = "x".repeat(REPLAY_BYTES / 4);
        for _ in 0..8 {
            slot.push(big.clone());
        }
        let result = slot.read_from(0, Duration::ZERO).await;
        assert!(result.lost, "the window has moved past index 0");
        assert!(result.events.is_empty());
        assert!(
            result.next_index > 0,
            "the caller is told where it can start"
        );

        // A reader that is still inside the window is unaffected.
        let inside = slot.read_from(result.next_index, Duration::ZERO).await;
        assert!(!inside.lost);
        assert!(!inside.events.is_empty());
    }

    /// The newest event is never evicted, however large: a one-event
    /// buffer still delivers to a live reader.
    #[tokio::test]
    async fn the_window_never_evicts_the_only_event_it_holds() {
        let slot = slot();
        slot.push("y".repeat(REPLAY_BYTES * 2));
        let result = slot.read_from(0, Duration::ZERO).await;
        assert!(!result.lost);
        assert_eq!(result.events.len(), 1);
    }

    /// A reader parked on a live stream must be woken by the next
    /// event, not by the timeout -- otherwise every token costs a full
    /// wait and "resumed" reads as "hung".
    #[tokio::test]
    async fn a_waiting_reader_is_woken_by_the_next_event() {
        let slot = Arc::new(slot());
        let writer = Arc::clone(&slot);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            writer.push("late".to_string());
        });
        let started = Instant::now();
        let result = slot.read_from(0, Duration::from_secs(5)).await;
        assert_eq!(result.events.len(), 1);
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the reader waited for the timeout instead of the event"
        );
    }

    /// A finished stream ends its readers rather than parking them for
    /// an event that will never come.
    #[tokio::test]
    async fn a_finished_stream_stops_its_readers_immediately() {
        let slot = slot();
        slot.push("only".to_string());
        slot.finish();
        let drained = slot.read_from(1, Duration::from_secs(30)).await;
        assert!(drained.events.is_empty());
        assert!(drained.finished);
    }

    /// `id:` and `retry:` are promises about replay, so they may not
    /// appear on a stream with no buffer behind them -- that is the
    /// exact promise this module exists rather than make.
    #[test]
    fn a_non_resumable_emitter_buffers_nothing_and_stops_on_a_lost_receiver() {
        let emitter = Emitter::new(None);
        assert!(!emitter.is_resumable());
        // Nothing to finish, and finishing must not panic.
        emitter.finish();
    }

    #[test]
    fn a_resumable_emitter_numbers_its_events_from_zero_and_buffers_them() {
        let registry = StreamRegistry::new();
        let slot = registry.register("chatcmpl-e");
        let emitter = Emitter::new(Some(Arc::clone(&slot)));
        assert!(emitter.is_resumable());
        let _ = emitter.event(&serde_json::json!({"n": 1}));
        let _ = emitter.done();
        emitter.finish();

        let result = slot.read_now(0);
        assert_eq!(result.events.len(), 2);
        assert_eq!(result.events[0].index, 0);
        assert_eq!(result.events[0].data, r#"{"n":1}"#);
        assert_eq!(
            result.events[1].data, "[DONE]",
            "the end of stream is replayable too, or a resumed reader never stops"
        );
        assert!(result.finished);
    }

    /// A decode thread that panics must not leave a slot live forever:
    /// readers would park on it and the registry only evicts finished
    /// slots, so it could never be reclaimed.
    #[test]
    fn dropping_an_emitter_closes_its_buffer_even_when_nothing_finished_it() {
        let registry = StreamRegistry::new();
        let slot = registry.register("chatcmpl-panic");
        {
            let emitter = Emitter::new(Some(Arc::clone(&slot)));
            let _ = emitter.event(&serde_json::json!({"n": 1}));
            // No `finish()` call: the thread "panicked" here.
        }
        assert!(
            slot.read_now(0).finished,
            "a dropped emitter must close its buffer"
        );
    }

    #[test]
    fn a_registered_stream_is_findable_by_its_request_id() {
        let registry = StreamRegistry::new();
        let slot = registry.register("chatcmpl-a");
        slot.push("hi".to_string());
        assert!(registry.get("chatcmpl-a").is_some());
        assert!(registry.get("chatcmpl-b").is_none());
    }

    /// Retention is bounded, and bounded in a way that never drops a
    /// stream someone is reading.
    #[test]
    fn the_registry_evicts_finished_streams_before_live_ones() {
        let registry = StreamRegistry::new();
        for i in 0..MAX_SLOTS {
            let slot = registry.register(&format!("chatcmpl-{i}"));
            // Every one but the first is finished, so the first is the
            // oldest *and* the one eviction must not take.
            if i > 0 {
                slot.finish();
            }
        }
        assert_eq!(registry.len(), MAX_SLOTS);
        registry.register("chatcmpl-new");
        assert!(registry.len() <= MAX_SLOTS);
        assert!(
            registry.get("chatcmpl-0").is_some(),
            "a live stream was evicted out from under its reader"
        );
        assert!(registry.get("chatcmpl-new").is_some());
    }
}
