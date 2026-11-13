//! One long-running-job contract, shared by every background job the
//! server runs (today: a Hub download and a model load).
//!
//! The uniformity is the point. `bc5074aa:docs/plans/on-hold/frink-ui.md` observes that
//! both reference products reuse a single task shape across four job
//! types, and that this is what makes the UI cheap: one polling loop,
//! one progress component, one cancel button, regardless of what is
//! actually running.
//!
//! Three properties this module guarantees, each with a test:
//!
//! - **Terminal is terminal.** Once a task reports `done`, `error` or
//!   `cancelled`, nothing can move it again. A late progress update
//!   from a worker that has not noticed it was cancelled is dropped,
//!   not applied -- otherwise a cancelled download would flicker back
//!   to `running` and a UI that stopped polling would be wrong.
//! - **Rates come from [`frink_api::progress::RateEstimator`] only.**
//!   The registry holds the estimator and hands its report straight to
//!   [`frink_api::TaskProgress`]; there is no second path by which a
//!   number could reach the wire before the window is long enough.
//! - **Cancellation is cooperative and honest about it.** `cancel()`
//!   raises a flag; the worker decides when to notice. A download
//!   checks it between chunks and stops within one chunk. A model load
//!   cannot be interrupted mid-mmap at all, so it checks the flag on
//!   the way in and again before publishing, and discards a finished
//!   load rather than pretending it stopped early. The task only
//!   reaches `cancelled` when a worker actually acknowledges it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use frink_api::progress::RateEstimator;
use frink_api::{TaskKind, TaskProgress, TaskStatus, TaskView};

/// How many finished tasks are remembered. Old entries are evicted
/// oldest-first, but only if they are terminal -- a live task is never
/// dropped from the registry, because a UI polling it would then see it
/// vanish rather than finish.
const MAX_TASKS: usize = 64;

/// Milliseconds since the Unix epoch, from the server's clock.
///
/// Epoch rather than a monotonic instant because these timestamps go on
/// the wire and are compared against `/admin/stats` entries. The rate
/// estimator only ever looks at differences, so the (very small) risk
/// of a wall-clock step backwards is handled by the estimator itself,
/// which ignores a sample older than the newest one.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

struct TaskInner {
    status: TaskStatus,
    error: Option<String>,
    started_at_ms: u64,
    updated_at_ms: u64,
    bytes_done: u64,
    bytes_total: Option<u64>,
    estimator: RateEstimator,
}

/// One job. Handed to the worker as an `Arc`; the registry keeps
/// another so `GET /admin/tasks` can read it while the worker runs.
pub(crate) struct Task {
    pub(crate) task_id: String,
    pub(crate) kind: TaskKind,
    pub(crate) label: String,
    inner: Mutex<TaskInner>,
    cancel_requested: AtomicBool,
}

impl Task {
    fn new(task_id: String, kind: TaskKind, label: String) -> Self {
        let now = now_ms();
        Task {
            task_id,
            kind,
            label,
            inner: Mutex::new(TaskInner {
                status: TaskStatus::Queued,
                error: None,
                started_at_ms: now,
                updated_at_ms: now,
                bytes_done: 0,
                bytes_total: None,
                estimator: RateEstimator::new(),
            }),
            cancel_requested: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TaskInner> {
        // Same defence as the response cache: a panic while holding
        // this lock must not brick every later task.
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Moves `queued` -> `running`. A no-op once terminal.
    pub(crate) fn start(&self) {
        let mut inner = self.lock();
        if inner.status.is_terminal() {
            return;
        }
        inner.status = TaskStatus::Running;
        inner.updated_at_ms = now_ms();
    }

    /// Records a cumulative byte count. `total` may be `None` for a
    /// transfer with no `Content-Length`.
    pub(crate) fn observe(&self, bytes_done: u64, total: Option<u64>) {
        self.observe_at(now_ms(), bytes_done, total)
    }

    /// [`Task::observe`] with an explicit clock, for tests.
    pub(crate) fn observe_at(&self, at_ms: u64, bytes_done: u64, total: Option<u64>) {
        let mut inner = self.lock();
        if inner.status.is_terminal() {
            return;
        }
        inner.status = TaskStatus::Running;
        inner.bytes_done = bytes_done;
        inner.bytes_total = total;
        inner.updated_at_ms = at_ms;
        inner.estimator.observe(at_ms, bytes_done);
    }

    fn finish(&self, status: TaskStatus, error: Option<String>) {
        let mut inner = self.lock();
        if inner.status.is_terminal() {
            return;
        }
        inner.status = status;
        inner.error = error;
        inner.updated_at_ms = now_ms();
    }

    pub(crate) fn succeed(&self) {
        self.finish(TaskStatus::Done, None);
    }

    pub(crate) fn fail(&self, error: impl std::fmt::Display) {
        self.finish(TaskStatus::Error, Some(error.to_string()));
    }

    /// Acknowledges a cancellation request. Called by the *worker*, not
    /// by the HTTP handler: a task is only `cancelled` once something
    /// has actually stopped.
    pub(crate) fn acknowledge_cancel(&self) {
        self.finish(TaskStatus::Cancelled, None);
    }

    /// Raises the cancel flag. Returns false when the task had already
    /// finished, so the handler can say so instead of claiming success.
    pub(crate) fn request_cancel(&self) -> bool {
        if self.lock().status.is_terminal() {
            return false;
        }
        self.cancel_requested.store(true, Ordering::Relaxed);
        true
    }

    /// Polled by workers between units of work.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancel_requested.load(Ordering::Relaxed)
    }

    pub(crate) fn status(&self) -> TaskStatus {
        self.lock().status
    }

    pub(crate) fn view(&self) -> TaskView {
        let inner = self.lock();
        let report = inner.estimator.report(inner.bytes_total);
        TaskView {
            task_id: self.task_id.clone(),
            kind: self.kind,
            label: self.label.clone(),
            status: inner.status,
            error: inner.error.clone(),
            started_at_ms: inner.started_at_ms,
            updated_at_ms: inner.updated_at_ms,
            progress: TaskProgress::from_report(report, inner.bytes_done, inner.bytes_total),
        }
    }
}

/// Held by a worker for its whole life so a task cannot be abandoned
/// mid-flight.
///
/// A `spawn_blocking` closure that panics is not observed by anyone:
/// nothing awaits its `JoinHandle`, so without this the task would sit
/// at `running` with zero progress for the rest of the process's life,
/// and a UI would poll it forever. Dropping during unwind still records
/// a verdict, and the mutex is poison-tolerant, so a panic *while
/// holding the task lock* is handled too.
pub(crate) struct TaskGuard(Arc<Task>);

impl TaskGuard {
    pub(crate) fn new(task: Arc<Task>) -> Self {
        TaskGuard(task)
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        if self.0.status().is_terminal() {
            return;
        }
        if self.0.is_cancelled() {
            self.0.acknowledge_cancel();
        } else {
            self.0
                .fail("the worker ended without reporting a result (it panicked)");
        }
    }
}

/// Every task this process has run, newest last, bounded.
pub(crate) struct TaskRegistry {
    tasks: Mutex<VecDeque<Arc<Task>>>,
    next_id: AtomicU64,
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskRegistry {
    pub(crate) fn new() -> Self {
        TaskRegistry {
            tasks: Mutex::new(VecDeque::new()),
            next_id: AtomicU64::new(1),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<Arc<Task>>> {
        self.tasks.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Registers a new task in `queued` and returns it. The caller
    /// spawns the worker; the registry never runs anything itself.
    pub(crate) fn create(&self, kind: TaskKind, label: impl Into<String>) -> Arc<Task> {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        let prefix = match kind {
            TaskKind::Download => "dl",
            TaskKind::Load => "load",
        };
        let task = Arc::new(Task::new(format!("{prefix}-{n}"), kind, label.into()));
        let mut tasks = self.lock();
        tasks.push_back(Arc::clone(&task));
        // Evict the oldest *finished* entries only. A live task that
        // scrolled off the end would look to a polling UI like a job
        // that disappeared mid-flight.
        while tasks.len() > MAX_TASKS {
            let Some(pos) = tasks.iter().position(|t| t.status().is_terminal()) else {
                break;
            };
            tasks.remove(pos);
        }
        task
    }

    pub(crate) fn get(&self, task_id: &str) -> Option<Arc<Task>> {
        self.lock().iter().find(|t| t.task_id == task_id).cloned()
    }

    pub(crate) fn views(&self) -> Vec<TaskView> {
        self.lock().iter().map(|t| t.view()).collect()
    }

    /// True when a task of this kind and label is still live.
    ///
    /// Matched on the label rather than merely the kind because the
    /// thing that must not happen twice is not "a download" but *this*
    /// download: two workers on the same target would interleave writes
    /// into one `.part` file and produce a corrupt checkpoint that
    /// still has the right size.
    pub(crate) fn has_live(&self, kind: TaskKind, label: &str) -> bool {
        self.lock()
            .iter()
            .any(|t| t.kind == kind && t.label == label && !t.status().is_terminal())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_api::ProgressState;

    #[test]
    fn a_new_task_is_queued_with_no_progress_numbers() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Download, "Downloading x.gguf");
        let view = task.view();
        assert_eq!(view.status, TaskStatus::Queued);
        assert_eq!(view.progress.state, ProgressState::Warming);
        assert_eq!(view.progress.bytes_done, 0);
        assert_eq!(view.progress.rate_bytes_per_s, None);
        assert_eq!(view.progress.fraction, None);
        assert!(view.task_id.starts_with("dl-"));
    }

    #[test]
    fn progress_reports_no_rate_until_the_estimator_says_stable() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Download, "d");
        // Two samples 2ms apart: a naive rate here is gigabytes/second.
        task.observe_at(1_000, 0, Some(10_000_000));
        task.observe_at(1_002, 8_000_000, Some(10_000_000));
        let view = task.view();
        assert_eq!(view.status, TaskStatus::Running);
        assert_eq!(view.progress.state, ProgressState::Warming);
        assert_eq!(view.progress.rate_bytes_per_s, None);
        assert_eq!(view.progress.eta_seconds, None);
        // The fraction is a ratio of counters, not a derivative, so it
        // is available immediately.
        assert_eq!(view.progress.fraction, Some(0.8));
    }

    #[test]
    fn a_long_enough_window_produces_a_rate_and_an_eta() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Download, "d");
        for i in 0..=4u64 {
            task.observe_at(i * 1000, i * 1_000_000, Some(10_000_000));
        }
        let view = task.view();
        assert_eq!(view.progress.state, ProgressState::Stable);
        assert_eq!(view.progress.rate_bytes_per_s, Some(1_000_000.0));
        assert_eq!(view.progress.eta_seconds, Some(6.0));
        assert_eq!(view.progress.bytes_done, 4_000_000);
    }

    #[test]
    fn a_resumed_transfer_stops_reporting_a_rate_rather_than_a_wrong_one() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Download, "d");
        for i in 0..=4u64 {
            task.observe_at(i * 1000, i * 1_000_000, Some(10_000_000));
        }
        assert_eq!(task.view().progress.state, ProgressState::Stable);
        task.observe_at(5_000, 0, Some(10_000_000)); // restarted
        let view = task.view();
        assert_eq!(view.progress.state, ProgressState::Warming);
        assert_eq!(view.progress.rate_bytes_per_s, None);
    }

    #[test]
    fn done_is_terminal_and_later_progress_is_ignored() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Download, "d");
        task.observe_at(1_000, 10, Some(100));
        task.succeed();
        task.observe_at(2_000, 50, Some(100));
        task.fail("too late");
        let view = task.view();
        assert_eq!(view.status, TaskStatus::Done);
        assert_eq!(view.error, None);
        assert_eq!(view.progress.bytes_done, 10);
    }

    #[test]
    fn an_error_records_its_message_and_stays_put() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Load, "l");
        task.start();
        task.fail("no such file");
        task.succeed();
        let view = task.view();
        assert_eq!(view.status, TaskStatus::Error);
        assert_eq!(view.error.as_deref(), Some("no such file"));
    }

    #[test]
    fn cancel_raises_a_flag_and_only_the_worker_makes_it_terminal() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Download, "d");
        task.start();
        assert!(task.request_cancel());
        // The HTTP handler has returned, but nothing has stopped yet:
        // claiming `cancelled` here would be a lie about the worker.
        assert_eq!(task.status(), TaskStatus::Running);
        assert!(task.is_cancelled());
        task.acknowledge_cancel();
        assert_eq!(task.status(), TaskStatus::Cancelled);
    }

    #[test]
    fn cancelling_a_finished_task_reports_failure_rather_than_success() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Download, "d");
        task.succeed();
        assert!(!task.request_cancel());
        assert_eq!(task.status(), TaskStatus::Done);
    }

    #[test]
    fn a_cancelled_task_cannot_be_marked_done_afterwards() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Download, "d");
        task.request_cancel();
        task.acknowledge_cancel();
        task.succeed();
        assert_eq!(task.status(), TaskStatus::Cancelled);
    }

    #[test]
    fn the_registry_finds_tasks_by_id_and_lists_them_in_order() {
        let reg = TaskRegistry::new();
        let a = reg.create(TaskKind::Download, "a");
        let b = reg.create(TaskKind::Load, "b");
        assert_ne!(a.task_id, b.task_id);
        assert_eq!(reg.get(&b.task_id).unwrap().label, "b");
        assert!(reg.get("nope").is_none());
        let views = reg.views();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].task_id, a.task_id);
    }

    #[test]
    fn live_tasks_are_never_evicted_by_newer_ones() {
        let reg = TaskRegistry::new();
        let live = reg.create(TaskKind::Load, "live");
        live.start();
        for i in 0..MAX_TASKS * 2 {
            let t = reg.create(TaskKind::Download, format!("d{i}"));
            t.succeed();
        }
        assert!(reg.get(&live.task_id).is_some());
        assert!(reg.views().len() <= MAX_TASKS + 1);
    }

    /// Nothing awaits a worker's `JoinHandle`, so a panic there is
    /// invisible. Without the guard the task would poll as `running`
    /// with zero progress for the life of the process.
    #[test]
    fn a_panicking_worker_still_leaves_the_task_in_a_terminal_state() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Download, "d");
        let handle = {
            let task = Arc::clone(&task);
            std::thread::spawn(move || {
                let _guard = TaskGuard::new(task);
                panic!("worker exploded");
            })
        };
        assert!(handle.join().is_err());
        assert_eq!(task.status(), TaskStatus::Error);
        assert!(task.view().error.is_some());
    }

    /// A worker that was asked to stop and then died still reads as
    /// cancelled, not as a crash the user did not cause.
    #[test]
    fn a_cancelled_worker_that_dies_is_recorded_as_cancelled() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Download, "d");
        task.request_cancel();
        drop(TaskGuard::new(Arc::clone(&task)));
        assert_eq!(task.status(), TaskStatus::Cancelled);
    }

    /// The guard must not overwrite a verdict the worker already gave.
    #[test]
    fn the_guard_leaves_a_finished_task_alone() {
        let reg = TaskRegistry::new();
        let task = reg.create(TaskKind::Load, "l");
        task.succeed();
        drop(TaskGuard::new(Arc::clone(&task)));
        assert_eq!(task.status(), TaskStatus::Done);
        assert_eq!(task.view().error, None);
    }

    #[test]
    fn has_live_distinguishes_two_jobs_of_the_same_kind() {
        let reg = TaskRegistry::new();
        let t = reg.create(TaskKind::Download, "a.gguf");
        reg.create(TaskKind::Download, "b.gguf").succeed();
        assert!(reg.has_live(TaskKind::Download, "a.gguf"));
        // Same kind, finished: not a reason to reject a new one.
        assert!(!reg.has_live(TaskKind::Download, "b.gguf"));
        assert!(!reg.has_live(TaskKind::Load, "a.gguf"));
        t.succeed();
        assert!(!reg.has_live(TaskKind::Download, "a.gguf"));
    }
}
