//! Wire shapes for the `/admin` control surface: the model inventory,
//! the one long-running-task contract, and the server's own counters.
//!
//! Two rules run through all of it.
//!
//! **Absent means absent.** Every field the UI reads is always present
//! in the JSON, and a value that could not be established cheaply is
//! `null` rather than a plausible-looking default. A `0` context length
//! and an unknown context length are different facts, and a UI that
//! cannot tell them apart will print the wrong one with confidence.
//! That is why the optional fields here are *not* `skip_serializing_if`
//! -- the key stays, the value goes to `null`.
//!
//! **Rates come from the estimator or not at all.** [`TaskProgress`] is
//! built from [`crate::progress::RateReport`], which refuses to divide
//! until its window is long enough. Nothing here may compute a rate on
//! the side; see [`TaskProgress::from_report`].

use serde::{Deserialize, Serialize};

use crate::progress::RateReport;

// ---------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------

/// What a model on disk is doing right now.
///
/// `available` is the resting state -- present, readable, not loaded.
/// `error` is sticky: it records that the *last* attempt to load this
/// model failed, so the UI can show why without the user having to
/// retry to find out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelState {
    Loaded,
    Loading,
    Available,
    Error,
}

/// One model the server can serve, described from its GGUF header
/// alone. Nothing here requires reading a single weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Stable within a server run: the file stem for a `.gguf`, the
    /// directory name for a checkpoint directory. This is what
    /// [`LoadModelRequest`] takes, and the only way to name a model --
    /// there is deliberately no "load this path" endpoint.
    pub id: String,
    /// Absolute path, for display. A client cannot ask the server to
    /// load an arbitrary one.
    pub path: String,
    /// On-disk size, summed across shards for a split checkpoint.
    pub size_bytes: u64,
    /// `general.architecture`, verbatim. `null` when the header does
    /// not carry it.
    pub arch: Option<String>,
    /// Quantization name (`Q4_K_M`, `F16`, ...) from `general.file_type`
    /// when it maps to a name this server knows, else the dominant
    /// tensor dtype, else `null`. Never guessed from the filename.
    pub quant: Option<String>,
    /// `{arch}.context_length` from the header.
    pub context_length: Option<u64>,
    /// `general.parameter_count` when present, else the summed element
    /// count of every tensor in the header.
    pub param_count: Option<u64>,
    pub state: ModelState,
    /// Why the last load attempt failed. `null` unless `state` is
    /// [`ModelState::Error`].
    pub error: Option<String>,
    /// Bytes actually resident for this model. `null` for anything not
    /// loaded, and `null` for a loaded model whose footprint the server
    /// cannot measure -- an mmap-resident checkpoint's true RSS is a
    /// property of the page cache, not of this process, and reporting
    /// the file size as "resident" would be a lie in both directions.
    pub resident_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelsResponse {
    /// The directory that was scanned. `null` when no model path is
    /// configured at all, which is also when `models` is empty for a
    /// reason the UI should explain rather than read as "none found".
    pub model_dir: Option<String>,
    /// Id of the loaded model, or `null` when nothing is loaded.
    pub active: Option<String>,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadModelRequest {
    pub id: String,
}

/// `202` body for anything that starts a background job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskAccepted {
    pub task_id: String,
}

/// The answer to `POST /sleep` and `POST /wake_up`.
///
/// `is_sleeping` is the state AFTER the call, so a caller does not
/// have to follow up with `GET /is_sleeping` to learn whether the
/// thing it asked for happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SleepResponse {
    pub ok: bool,
    pub is_sleeping: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]

pub struct UnloadResponse {
    pub ok: bool,
    /// Always `null` on success; stated rather than omitted so the UI
    /// can use one code path for "what is active now".
    pub active: Option<String>,
}

/// A Hub repo plus the file to take from it. `file` may be a literal
/// name or a `*` glob, which is resolved against the repo's file list.
/// Both are validated server-side: only `.gguf` targets, and nothing
/// that could name a path outside the model directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DownloadRequest {
    pub repo: String,
    pub file: String,
}

// ---------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskKind {
    Download,
    Load,
}

/// `queued`/`running` are live; `done`/`error`/`cancelled` are terminal
/// and never change again. A UI can stop polling a task the moment it
/// reads a terminal status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Queued,
    Running,
    Done,
    Error,
    Cancelled,
}

impl TaskStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskStatus::Done | TaskStatus::Error | TaskStatus::Cancelled
        )
    }
}

/// Whether the rate/ETA numbers may be shown at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProgressState {
    /// Not enough samples yet. `rate_bytes_per_s` and `eta_seconds` are
    /// `null` and the UI must show "measuring", not a number.
    Warming,
    Stable,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TaskProgress {
    /// `bytes_done / bytes_total`, clamped to `0.0..=1.0`. `null` when
    /// the total is unknown -- an indeterminate bar is honest, a bar
    /// pinned at 100% is not.
    pub fraction: Option<f64>,
    pub bytes_done: u64,
    pub bytes_total: Option<u64>,
    pub rate_bytes_per_s: Option<f64>,
    pub eta_seconds: Option<f64>,
    pub state: ProgressState,
}

impl TaskProgress {
    /// The only sanctioned way to build one.
    ///
    /// A warming report yields `null` rate and `null` ETA no matter
    /// what the caller believes it knows: the estimator's whole purpose
    /// is refusing to divide too early, and recomputing around it would
    /// reintroduce the "123 GB/s" flash it exists to prevent.
    pub fn from_report(report: RateReport, bytes_done: u64, bytes_total: Option<u64>) -> Self {
        let stable = report.stable;
        TaskProgress {
            fraction: bytes_total
                .filter(|t| *t > 0)
                .map(|total| (bytes_done as f64 / total as f64).clamp(0.0, 1.0)),
            bytes_done,
            bytes_total,
            rate_bytes_per_s: stable.then_some(report.bytes_per_second).flatten(),
            eta_seconds: stable.then_some(report.eta_seconds).flatten(),
            state: if stable {
                ProgressState::Stable
            } else {
                ProgressState::Warming
            },
        }
    }

    /// A job with nothing measurable in bytes (a model load): no
    /// fraction, no rate, no pretence of either.
    pub fn indeterminate() -> Self {
        TaskProgress {
            fraction: None,
            bytes_done: 0,
            bytes_total: None,
            rate_bytes_per_s: None,
            eta_seconds: None,
            state: ProgressState::Warming,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskView {
    pub task_id: String,
    pub kind: TaskKind,
    /// One human sentence naming what this job is doing, written by the
    /// server so the UI never has to assemble one from ids and paths.
    pub label: String,
    pub status: TaskStatus,
    pub error: Option<String>,
    /// Unix epoch milliseconds, from the server's clock. The plan is
    /// explicit that the browser's clock is not to be trusted for
    /// ordering, so both timestamps are stated rather than implied.
    pub started_at_ms: u64,
    pub updated_at_ms: u64,
    pub progress: TaskProgress,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TasksResponse {
    pub tasks: Vec<TaskView>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancelResponse {
    pub ok: bool,
}

// ---------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------

/// One finished request, as recorded in the ring buffer.
///
/// `duration_ms` and `decode_ms` are separate on purpose and must stay
/// that way: `duration_ms` carries queue wait plus prefill plus decode,
/// so dividing completion tokens by it reports a 50 tok/s model as 5
/// whenever the prompt is long. Everything downstream of that number is
/// then wrong in the same direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecentRequest {
    /// The same id the response carried, so a UI can join a log line to
    /// the message it produced without a claiming heuristic.
    pub request_id: String,
    /// Unix epoch milliseconds when the request finished.
    pub at_ms: u64,
    pub route: String,
    /// The model that **served** this request, as `/v1/models` names it.
    ///
    /// Not the `model` field the client sent. frink serves whatever is
    /// loaded and ignores that string, so echoing it back would make
    /// the log agree with the caller's belief rather than with what
    /// happened -- and after a model swap those are different answers.
    /// `null` when nothing was loaded, which is what a 503 row means.
    pub model: Option<String>,
    pub status: u16,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: Option<f64>,
    /// Total server-side wall time for the request.
    pub duration_ms: u64,
    /// Time inside the decode loop only. `null` when the engine did not
    /// time itself, or the answer came from cache.
    pub decode_ms: Option<f64>,
    pub stream: bool,
    /// Completion tokens per verification step when this request used
    /// speculative decoding; `null` when it did not. See
    /// [`crate::Usage::acceptance_length`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_length: Option<f64>,
    /// Accept rate at each position within the draft block. Kept in the
    /// ring rather than only in the response body because suffix decay
    /// is a property of the *drafter over time*, and a single request's
    /// numbers are too few to read it off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_accept_rate_per_position: Option<Vec<f64>>,
    /// Which bearer key served this request, as a short fingerprint --
    /// never the key itself, and never reversible into it.
    ///
    /// `null` means the request carried no `Authorization: Bearer`
    /// header at all, which on a server started without
    /// `FRINK_API_KEY` is every request. Two rows with the same
    /// fingerprint were authenticated with the same key; two rows with
    /// different fingerprints were not. That is the whole of what this
    /// field claims.
    ///
    /// The fingerprint is salted per process, so it is stable within
    /// one server run and deliberately meaningless across restarts: a
    /// captured `/admin/stats` payload cannot be used offline to test
    /// guesses at the key.
    pub via_api_key: Option<String>,
    /// The caller's self-declared label, from the `X-Frink-Client`
    /// request header, truncated and stripped of anything that is not a
    /// plain label character.
    ///
    /// **A claim, not proof.** Frink Studio sends `frink-studio`, and
    /// so could any other client; nothing here authenticates it. It is
    /// recorded because a self-declared label plus a key fingerprint is
    /// still the difference between "an editor is hammering this
    /// server" and "that was me in the other tab", and because
    /// inventing the distinction from timing would be worse. A UI that
    /// shows it must say it is self-declared.
    pub client: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatsResponse {
    pub uptime_seconds: u64,
    pub requests_total: u64,
    pub errors_total: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub tokens_prompt_total: u64,
    pub tokens_generated_total: u64,
    /// Seconds since the last request finished; `null` when none has.
    pub last_request_age_seconds: Option<f64>,
    /// Streamed generations decoding right now -- the ones that could
    /// be stopped by `POST /v1/cancel` at this instant.
    ///
    /// Not a queue depth: nothing is queued in front of a decode here,
    /// so this counts work in progress, not work waiting. Named for
    /// what it is so no one reads a backlog into it.
    pub generating_now: usize,
    /// Requests waiting for a decode slot, from the continuous-batching
    /// scheduler's own queue.
    ///
    /// `null` -- not `0` -- when continuous batching is off, because
    /// then there is no queue at all: every request goes straight onto
    /// its own blocking thread. A gauge reading `0` claims an empty
    /// queue was measured; `null` says there was nothing to measure,
    /// and a UI must be able to tell those apart.
    pub queue_depth: Option<usize>,
    /// Requests the queue turned away because it was full, since start.
    /// `null` under the same condition as [`Self::queue_depth`].
    pub queue_rejected_total: Option<u64>,
    /// Newest last, capped server-side. See [`RecentRequest`].
    pub recent: Vec<RecentRequest>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::RateEstimator;

    fn stable_estimator() -> RateEstimator {
        let mut est = RateEstimator::new();
        for i in 0..=4u64 {
            est.observe(i * 1000, i * 1_000_000);
        }
        est
    }

    #[test]
    fn a_warming_estimator_yields_no_rate_and_no_eta() {
        let mut est = RateEstimator::new();
        est.observe(0, 0);
        est.observe(2, 8 * 1024 * 1024);
        let progress =
            TaskProgress::from_report(est.report(Some(1 << 30)), 8 * 1024 * 1024, Some(1 << 30));
        assert_eq!(progress.state, ProgressState::Warming);
        assert_eq!(progress.rate_bytes_per_s, None);
        assert_eq!(progress.eta_seconds, None);
        // A fraction is still fine: it is a ratio of two counters, not
        // a derivative, so no window is needed to trust it.
        assert!(progress.fraction.is_some());
    }

    #[test]
    fn a_stable_estimator_passes_its_numbers_through_unchanged() {
        let est = stable_estimator();
        let report = est.report(Some(10_000_000));
        let progress = TaskProgress::from_report(report, 4_000_000, Some(10_000_000));
        assert_eq!(progress.state, ProgressState::Stable);
        assert_eq!(progress.rate_bytes_per_s, Some(1_000_000.0));
        assert_eq!(progress.eta_seconds, Some(6.0));
        assert_eq!(progress.fraction, Some(0.4));
    }

    #[test]
    fn an_unknown_total_means_no_fraction_rather_than_zero() {
        let est = stable_estimator();
        let progress = TaskProgress::from_report(est.report(None), 4_000_000, None);
        assert_eq!(progress.fraction, None);
        assert_eq!(progress.eta_seconds, None);
        assert_eq!(progress.rate_bytes_per_s, Some(1_000_000.0));
    }

    #[test]
    fn a_fraction_never_exceeds_one_even_with_bad_metadata() {
        let est = stable_estimator();
        let progress = TaskProgress::from_report(est.report(Some(1_000)), 4_000_000, Some(1_000));
        assert_eq!(progress.fraction, Some(1.0));
    }

    #[test]
    fn optional_model_fields_serialize_as_null_rather_than_vanishing() {
        let entry = ModelEntry {
            id: "m".into(),
            path: "/models/m.gguf".into(),
            size_bytes: 1,
            arch: None,
            quant: None,
            context_length: None,
            param_count: None,
            state: ModelState::Available,
            error: None,
            resident_bytes: None,
        };
        let json: serde_json::Value = serde_json::to_value(&entry).unwrap();
        for key in [
            "arch",
            "quant",
            "context_length",
            "param_count",
            "error",
            "resident_bytes",
        ] {
            assert!(json.get(key).is_some(), "{key} was omitted entirely");
            assert!(json[key].is_null(), "{key} was not null");
        }
        assert_eq!(json["state"], "available");
    }

    #[test]
    fn task_statuses_wire_as_the_lowercase_names_the_contract_names() {
        let view = TaskView {
            task_id: "t1".into(),
            kind: TaskKind::Download,
            label: "Downloading x.gguf".into(),
            status: TaskStatus::Running,
            error: None,
            started_at_ms: 1,
            updated_at_ms: 2,
            progress: TaskProgress::indeterminate(),
        };
        let json = serde_json::to_value(&view).unwrap();
        assert_eq!(json["kind"], "download");
        assert_eq!(json["status"], "running");
        assert_eq!(json["progress"]["state"], "warming");
        assert!(json["progress"]["bytes_total"].is_null());
        assert_eq!(json["progress"]["bytes_done"], 0);
    }

    #[test]
    fn terminal_statuses_are_exactly_the_three_that_stop_polling() {
        assert!(TaskStatus::Done.is_terminal());
        assert!(TaskStatus::Error.is_terminal());
        assert!(TaskStatus::Cancelled.is_terminal());
        assert!(!TaskStatus::Queued.is_terminal());
        assert!(!TaskStatus::Running.is_terminal());
    }

    #[test]
    fn recent_requests_keep_the_two_durations_apart() {
        let recent = RecentRequest {
            request_id: "chatcmpl-1".into(),
            at_ms: 10,
            route: "/v1/chat/completions".into(),
            model: Some("Qwen3-0.6B-Q4_K_M".into()),
            status: 200,
            prompt_tokens: 100,
            completion_tokens: 10,
            ttft_ms: Some(900.0),
            duration_ms: 1_100,
            decode_ms: Some(100.0),
            stream: true,
            acceptance_length: None,
            draft_accept_rate_per_position: None,
            via_api_key: None,
            client: None,
        };
        let json = serde_json::to_value(&recent).unwrap();
        assert_eq!(json["duration_ms"], 1_100);
        assert_eq!(json["decode_ms"], 100.0);
    }

    /// An absent attribution has to survive the wire as `null` rather
    /// than vanishing: "no key was presented" is a fact the monitor
    /// shows, and a missing key would read as a UI bug instead.
    #[test]
    fn absent_attribution_serializes_as_null_rather_than_vanishing() {
        let recent = RecentRequest {
            request_id: "chatcmpl-1".into(),
            at_ms: 10,
            route: "/v1/tokenize".into(),
            model: None,
            status: 200,
            prompt_tokens: 0,
            completion_tokens: 0,
            ttft_ms: None,
            duration_ms: 1,
            decode_ms: None,
            stream: false,
            via_api_key: None,
            client: None,
            acceptance_length: None,
            draft_accept_rate_per_position: None,
        };
        let json = serde_json::to_value(&recent).unwrap();
        for key in ["via_api_key", "client", "model"] {
            assert!(json.get(key).is_some(), "{key} was omitted entirely");
            assert!(json[key].is_null(), "{key} was not null");
        }
    }

    /// The queue gauge is `null` when there is no queue, and a UI must
    /// be able to tell that from a measured empty one.
    #[test]
    fn an_absent_queue_gauge_is_null_not_zero() {
        let stats = StatsResponse {
            uptime_seconds: 1,
            requests_total: 0,
            errors_total: 0,
            cache_hits: 0,
            cache_misses: 0,
            tokens_prompt_total: 0,
            tokens_generated_total: 0,
            last_request_age_seconds: None,
            generating_now: 0,
            queue_depth: None,
            queue_rejected_total: None,
            recent: Vec::new(),
        };
        let json = serde_json::to_value(&stats).unwrap();
        assert!(json["queue_depth"].is_null());
        assert!(json["queue_rejected_total"].is_null());
        assert_eq!(json["generating_now"], 0);
    }
}
