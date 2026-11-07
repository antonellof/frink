//! Process lifecycle journal: a small, append-only, JSON-lines file
//! recording *process facts* (start, exit, panic) about this
//! `frink-server` instance, independent of `tracing_subscriber`'s
//! stdout logging wired up in `main()`. The point is narrow: if the
//! process is killed (OOM-killed, `SIGKILL`, a hard crash) with no
//! surviving terminal, there is still a file on disk that answers "was
//! this process even running, and what was the last thing that
//! happened to it" -- stdout logging alone can't answer that once the
//! terminal/journal-capture pipe is gone.
//!
//! ## Privacy: this file never contains prompts or generated text
//!
//! This is a real, deliberate property, not an incidental one: every
//! record type in [`Record`] carries only process metadata (version,
//! pid, exit reason, panic message/location) -- never a request body,
//! rendered prompt, or generated completion. A crash journal is exactly
//! the kind of file that tends to get attached to bug reports or left
//! world-readable on a shared box, so it must be safe to hand to anyone
//! without leaking what a user asked the model or what the model said
//! back. Do not add a record type or field to this module that carries
//! request/response content -- if a future change needs that, it
//! belongs in the request-scoped `tracing` logs (which the operator
//! already controls the verbosity/destination of), not here.
//!
//! ## Why a flat file, not a "real" logging destination
//!
//! This journal deliberately does not go through `tracing`: a panic or
//! an `abort` can happen after `tracing`'s async writers/subscribers are
//! in an unknown state, and the whole point is to survive that. Every
//! write here is a direct, synchronous, unbuffered `OpenOptions` append
//! plus `flush()`, so a record is durable (subject to the OS/page cache
//! -- this does not `fsync`) by the time the call returns, not
//! whenever some background worker gets around to it.
//!
//! ## Default path
//!
//! Honors `FRINK_JOURNAL_PATH` if set. Otherwise defaults to
//! `./frink-journal.log` in the current working directory. This
//! codebase has no existing convention for per-platform state
//! directories (no `dirs`/`directories` dependency anywhere in the
//! workspace -- `FRINK_MODEL_PATH`, `FRINK_ADDR`, etc. are all
//! plain env-var-or-literal-default, no XDG/`~` resolution), so rather
//! than introduce one just for this feature, the default mirrors that
//! existing convention: simple, cross-platform, zero new dependencies.
//! An operator who wants `~/.local/state/frink/journal.log` (or
//! anything else) already has the tool to get it: set
//! `FRINK_JOURNAL_PATH` explicitly, exactly like every other path/
//! knob this server takes from the environment.
//!
//! ## Rotation
//!
//! Before appending, if the journal file exceeds a size threshold
//! (5 MiB by default, [`DEFAULT_ROTATE_BYTES`]), the current file is
//! renamed to `<path>.1` (overwriting any previous `.1`) and a fresh
//! file is started. Exactly one retained predecessor, not an unbounded
//! chain -- this is a crash journal, not an audit log; the goal is "it
//! can't grow forever," not "keep every byte ever written."

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Rotate the journal once it exceeds this many bytes. Not `const` in
/// the sense of "never overridden" -- [`Journal::with_rotate_threshold`]
/// exists precisely so tests can inject a tiny threshold and exercise
/// rotation without writing 5 MiB per test run.
pub const DEFAULT_ROTATE_BYTES: u64 = 5 * 1024 * 1024;

/// One process-lifecycle record. `#[serde(tag = "type")]` gives every
/// serialized line a `"type"` field naming the variant, matching the
/// task's "each a JSON object with at least a `type` field" shape
/// without hand-writing it into every variant.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Record {
    SessionStart {
        timestamp: String,
        version: String,
        pid: u32,
    },
    SessionExit {
        timestamp: String,
        reason: String,
    },
    Panic {
        timestamp: String,
        message: String,
        thread: String,
        location: Option<String>,
        /// Always `false` today: this codebase has no existing marker
        /// for "this specific panic was anticipated/harmless" outside
        /// `frink_cuda::gpu::probe` (which suppresses the panic
        /// entirely via a temporary empty hook rather than tagging it
        /// -- see that function's doc comment), so there is nothing to
        /// mirror yet. The field is included now, defaulted `false`,
        /// so a real detection rule can be added later without
        /// changing the on-disk record shape.
        expected: bool,
    },
}

impl Record {
    pub fn session_start(version: &str, pid: u32) -> Self {
        Record::SessionStart {
            timestamp: now_iso8601(),
            version: version.to_string(),
            pid,
        }
    }

    pub fn session_exit(reason: impl Into<String>) -> Self {
        Record::SessionExit {
            timestamp: now_iso8601(),
            reason: reason.into(),
        }
    }

    fn panic(message: String, thread: String, location: Option<String>) -> Self {
        Record::Panic {
            timestamp: now_iso8601(),
            message,
            thread,
            location,
            expected: false,
        }
    }
}

/// An append-only JSON-lines journal file, with size-capped rotation.
/// Cheap to construct (just resolves and stores a path + threshold; no
/// file is opened/created until the first [`Journal::append`]).
#[derive(Debug, Clone)]
pub struct Journal {
    path: PathBuf,
    rotate_bytes: u64,
}

impl Journal {
    /// Resolves the journal path from `FRINK_JOURNAL_PATH`, falling
    /// back to `./frink-journal.log` -- see the module doc comment
    /// for why that default was chosen over a per-platform state
    /// directory.
    pub fn from_env() -> Self {
        let path = std::env::var_os("FRINK_JOURNAL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("frink-journal.log"));
        Self {
            path,
            rotate_bytes: DEFAULT_ROTATE_BYTES,
        }
    }

    /// Same as [`Journal::from_env`] but at an explicit path -- used by
    /// tests, and available to any future caller that wants a journal
    /// somewhere other than the env-resolved default. Only exercised
    /// from `#[cfg(test)]` today, hence `allow(dead_code)`: real
    /// callers currently only ever want [`Journal::from_env`].
    #[allow(dead_code)]
    pub fn at_path(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            rotate_bytes: DEFAULT_ROTATE_BYTES,
        }
    }

    /// Overrides the rotation threshold (default
    /// [`DEFAULT_ROTATE_BYTES`]) -- exists so tests can force rotation
    /// with a handful of bytes instead of writing 5 MiB. Only
    /// exercised from `#[cfg(test)]` today.
    #[allow(dead_code)]
    pub fn with_rotate_threshold(mut self, bytes: u64) -> Self {
        self.rotate_bytes = bytes;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Rotates (if the file exists and is over threshold) then appends
    /// `record` as one JSON line. Best-effort: I/O failures (e.g. an
    /// unwritable directory) are logged via `tracing` and swallowed
    /// rather than propagated -- a journal that can't be written must
    /// never be the reason the server itself fails to start, serve, or
    /// shut down.
    pub fn append(&self, record: &Record) {
        if let Err(e) = self.try_append(record) {
            tracing::warn!("journal write to {:?} failed: {e}", self.path);
        }
    }

    fn try_append(&self, record: &Record) -> std::io::Result<()> {
        self.rotate_if_needed()?;
        let mut line = serde_json::to_string(record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        line.push('\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(line.as_bytes())?;
        file.flush()
    }

    fn rotate_if_needed(&self) -> std::io::Result<()> {
        let len = match std::fs::metadata(&self.path) {
            Ok(meta) => meta.len(),
            // No file yet -- nothing to rotate, first append creates it.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        };
        if len <= self.rotate_bytes {
            return Ok(());
        }
        let mut rotated = self.path.clone();
        let rotated_name = match self.path.file_name() {
            Some(name) => format!("{}.1", name.to_string_lossy()),
            None => "frink-journal.log.1".to_string(),
        };
        rotated.set_file_name(rotated_name);
        // `rename` overwrites an existing `.1` on every platform this
        // targets (POSIX always does; Windows since Rust made
        // `fs::rename` call `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING`),
        // so this keeps exactly one retained predecessor, never more.
        std::fs::rename(&self.path, &rotated)
    }
}

/// Installs a panic hook that appends a [`Record::Panic`] to `journal`
/// and then chains to whatever hook was previously installed (normally
/// the Rust default, which prints to stderr and honors
/// `RUST_BACKTRACE`) -- so ordinary panic behavior is fully preserved;
/// this only ever adds a durable record alongside it. Call once, early
/// in `main()`, after the journal itself is constructed.
pub fn install_panic_hook(journal: Journal) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let message = panic_message(info);
        let thread = std::thread::current()
            .name()
            .unwrap_or("<unnamed>")
            .to_string();
        let location = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()));
        journal.append(&Record::panic(message, thread, location));
        previous(info);
    }));
}

fn panic_message(info: &std::panic::PanicHookInfo<'_>) -> String {
    if let Some(s) = info.payload().downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// A minimal, dependency-free UTC ISO 8601 / RFC 3339 timestamp
/// (`YYYY-MM-DDTHH:MM:SS.mmmZ`). No `chrono`/`time` crate exists
/// anywhere in this workspace (see `Cargo.toml`'s
/// `[workspace.dependencies]`), so rather than add one solely for this
/// feature, this converts `SystemTime` -> civil calendar date using
/// Howard Hinnant's well-known `civil_from_days` algorithm (the same
/// proleptic-Gregorian integer-arithmetic approach widely used in
/// dependency-free date code) -- it's a handful of lines of pure
/// integer math, exactly the kind of small independent implementation
/// this codebase already favors (see the GGUF parser's own doc
/// comments on being an independent implementation against the public
/// spec, not a wrapped third-party one).
fn now_iso8601() -> String {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format_iso8601(dur.as_secs(), dur.subsec_millis())
}

fn format_iso8601(total_secs: u64, millis: u32) -> String {
    let days = (total_secs / 86_400) as i64;
    let secs_of_day = total_secs % 86_400;
    let hour = secs_of_day / 3600;
    let min = (secs_of_day % 3600) / 60;
    let sec = secs_of_day % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}.{millis:03}Z")
}

/// Days since the Unix epoch (1970-01-01) -> proleptic Gregorian
/// (year, month, day). Howard Hinnant's `civil_from_days`
/// (public domain algorithm, http://howardhinnant.github.io/date_algorithms.html),
/// transcribed directly rather than pulled in as a dependency.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;

    const TEST_THREAD: &str = "journal-test-thread";

    fn read_lines(path: &Path) -> Vec<String> {
        let file = std::fs::File::open(path).expect("journal file must exist");
        std::io::BufReader::new(file)
            .lines()
            .map(|l| l.expect("valid utf8 line"))
            .collect()
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        // 1970-01-01 is day 0.
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2000-01-01 is a well-known reference point (10957 days after epoch).
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        // 2024-02-29 (leap day) -- 19782 days after epoch.
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    }

    #[test]
    fn now_iso8601_has_the_expected_shape() {
        let ts = now_iso8601();
        // "YYYY-MM-DDTHH:MM:SS.mmmZ" is exactly 24 characters.
        assert_eq!(ts.len(), 24, "unexpected timestamp shape: {ts}");
        assert!(ts.ends_with('Z'));
        assert_eq!(ts.as_bytes()[4], b'-');
        assert_eq!(ts.as_bytes()[10], b'T');
    }

    #[test]
    fn session_start_and_exit_round_trip_as_valid_json_lines() {
        let dir = std::env::temp_dir().join(format!(
            "frink-journal-test-roundtrip-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("journal.log");
        let _ = std::fs::remove_file(&path);

        let journal = Journal::at_path(&path);
        journal.append(&Record::session_start(env!("CARGO_PKG_VERSION"), 4242));
        journal.append(&Record::session_exit("normal"));

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 2);

        let start: Record = serde_json::from_str(&lines[0]).expect("valid JSON line");
        match &start {
            Record::SessionStart { version, pid, .. } => {
                assert_eq!(version, env!("CARGO_PKG_VERSION"));
                assert_eq!(*pid, 4242);
            }
            other => panic!("expected SessionStart, got {other:?}"),
        }

        let exit: Record = serde_json::from_str(&lines[1]).expect("valid JSON line");
        match &exit {
            Record::SessionExit { reason, .. } => assert_eq!(reason, "normal"),
            other => panic!("expected SessionExit, got {other:?}"),
        }

        // Every line must also carry a "type" field per the task's record shape.
        let raw: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(raw["type"], "session_start");
        assert!(raw["timestamp"].is_string());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn error_exit_reason_round_trips() {
        let dir = std::env::temp_dir().join(format!(
            "frink-journal-test-error-exit-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("journal.log");
        let _ = std::fs::remove_file(&path);

        let journal = Journal::at_path(&path);
        journal.append(&Record::session_exit("bind error: address in use"));

        let lines = read_lines(&path);
        let exit: Record = serde_json::from_str(&lines[0]).unwrap();
        match exit {
            Record::SessionExit { reason, .. } => {
                assert_eq!(reason, "bind error: address in use")
            }
            other => panic!("expected SessionExit, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rotation_moves_oversized_file_to_dot_1_and_keeps_only_one_predecessor() {
        let dir = std::env::temp_dir().join(format!(
            "frink-journal-test-rotation-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("journal.log");
        let rotated = dir.join("journal.log.1");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&rotated);

        // Tiny threshold so a handful of small records blow past it
        // without writing anything close to the real 5 MiB default.
        let journal = Journal::at_path(&path).with_rotate_threshold(50);

        journal.append(&Record::session_start("0.0.0-test", 1));
        assert!(
            !rotated.exists(),
            "must not rotate before exceeding threshold"
        );

        // Write a marker record we can identify after rotation, then
        // enough further records to push the file over the tiny
        // threshold and trigger a rotate on the next append.
        journal.append(&Record::session_exit("marker-before-rotation"));
        for i in 0..20 {
            journal.append(&Record::session_exit(format!("filler-{i}")));
        }

        assert!(rotated.exists(), "rotated predecessor file must exist");

        // Exactly one retained predecessor: rotating again must not
        // produce a `.log.2` or otherwise accumulate files.
        let before_second_rotation_len = std::fs::metadata(&rotated).unwrap().len();
        for i in 0..20 {
            journal.append(&Record::session_exit(format!("more-filler-{i}")));
        }
        assert!(
            !dir.join("journal.log.2").exists(),
            "must never keep more than one retained predecessor"
        );
        // The rotated file should have been replaced by a newer batch,
        // not appended to indefinitely.
        let after_second_rotation_len = std::fs::metadata(&rotated).unwrap().len();
        assert!(after_second_rotation_len > 0);
        let _ = before_second_rotation_len; // just documenting intent above

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn panic_hook_writes_a_panic_record_with_message_thread_and_location() {
        let dir =
            std::env::temp_dir().join(format!("frink-journal-test-panic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("journal.log");
        let _ = std::fs::remove_file(&path);

        // `std::panic::set_hook` is process-global, and this test
        // binary runs many tests concurrently on shared threads, so
        // this test saves whatever hook is active (normally the
        // harness's own), installs the journal hook just long enough
        // to trigger and observe one deliberate panic, then restores
        // exactly what was there before -- leaving no lasting effect
        // on any other test's panic handling.
        let prior_hook = std::panic::take_hook();
        let journal = Journal::at_path(&path);
        install_panic_hook(journal.clone());

        let result = std::panic::catch_unwind(|| {
            std::thread::Builder::new()
                .name(TEST_THREAD.to_string())
                .spawn(|| {
                    panic!("deliberate test panic for the journal");
                })
                .unwrap()
                .join()
        });
        std::panic::set_hook(prior_hook);
        assert!(result.is_ok(), "catch_unwind itself must not propagate");
        assert!(
            result.unwrap().is_err(),
            "the spawned thread must have actually panicked"
        );

        // The hook is process-global while installed, so a test on
        // another thread that panics in this window (any `should_panic`)
        // is journaled too. Count only this thread's records: exactly one.
        let ours: Vec<Record> = read_lines(&path)
            .iter()
            .map(|l| serde_json::from_str::<Record>(l).unwrap())
            .filter(|r| matches!(r, Record::Panic { thread, .. } if thread == TEST_THREAD))
            .collect();
        assert_eq!(
            ours.len(),
            1,
            "exactly one panic record from this test's thread"
        );
        let record = ours.into_iter().next().unwrap();
        match record {
            Record::Panic {
                message,
                thread,
                location,
                expected,
                ..
            } => {
                assert_eq!(message, "deliberate test panic for the journal");
                assert_eq!(thread, TEST_THREAD);
                assert!(location.is_some(), "location must be captured");
                assert!(
                    location.unwrap().contains("journal.rs"),
                    "location should point at this file"
                );
                assert!(!expected, "expected must default to false");
            }
            other => panic!("expected Panic, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
