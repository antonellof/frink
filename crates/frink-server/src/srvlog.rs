//! Startup lines in the shape `llama serve` prints them.
//!
//! Measured against llama.cpp 0.4.1 (`llama serve`, build b10964):
//!
//! ```text
//! 0.00.102.345 I srv    load_model: loading model 'models/…gguf'
//! 0.01.123.100 I cmn          init: llama threadpool init, n_threads = 6
//! 0.01.255.290 I srv  llama_server: listening on http://127.0.0.1:8399
//! ```
//!
//! Four fields: elapsed since process start, a one-letter level, a
//! three-letter subsystem, and a component right-aligned to twelve
//! columns before the colon. The alignment is what makes the messages
//! line up, so it is a constant here rather than a width repeated at
//! every call site.
//!
//! # What this does NOT replace
//!
//! The `frink.server.ready` JSON line on **stdout**. That is a parsed
//! contract -- it is what makes `--port 0` usable, because the
//! supervisor learns the bound port from the child rather than probing
//! for it. These lines go to stderr, where a human reads them, and the
//! two streams stay separate on purpose.

use std::fmt::Write as _;
use std::sync::OnceLock;
use std::time::Instant;

/// Process start, for the elapsed column.
///
/// Set on the first line rather than at `main`, because a library
/// cannot assume it owns `main` -- and the first line is close enough
/// to start that the difference is below the microsecond the format
/// prints.
fn started() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Severity, in llama.cpp's one-letter spelling.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    Info,
    Warn,
}

impl Level {
    fn letter(self) -> char {
        match self {
            Level::Info => 'I',
            Level::Warn => 'W',
        }
    }
}

/// Which half of the server is speaking. Three letters, as upstream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sub {
    /// The server proper.
    Srv,
    /// Shared/common setup: threads, params.
    Cmn,
}

impl Sub {
    fn tag(self) -> &'static str {
        match self {
            Sub::Srv => "srv",
            Sub::Cmn => "cmn",
        }
    }
}

/// Columns the component is right-aligned into, before the colon.
///
/// Twelve, because `llama_server` and `common_param` are twelve
/// characters and upstream's shorter names (`load_model`, `init`) are
/// padded out to meet them.
const COMPONENT_WIDTH: usize = 12;

/// `MM.SS.mmm.uuu` since process start.
///
/// Minutes, seconds, milliseconds, microseconds -- read off upstream's
/// own output, where `0.01.255.290` is the line printed 1.255290 s
/// after start.
fn elapsed(since: std::time::Duration) -> String {
    let total = since.as_micros();
    let us = (total % 1_000) as u64;
    let ms = ((total / 1_000) % 1_000) as u64;
    let secs = (total / 1_000_000) as u64;
    format!("{}.{:02}.{:03}.{:03}", secs / 60, secs % 60, ms, us)
}

/// One formatted line, without the trailing newline.
pub fn line(at: std::time::Duration, level: Level, sub: Sub, component: &str, msg: &str) -> String {
    let mut out = String::with_capacity(64 + msg.len());
    let _ = write!(
        out,
        "{} {} {}  {:>COMPONENT_WIDTH$}: {msg}",
        elapsed(at),
        level.letter(),
        sub.tag(),
        component
    );
    out
}

/// Prints one line to stderr.
pub fn log(level: Level, sub: Sub, component: &str, msg: &str) {
    eprintln!("{}", line(started().elapsed(), level, sub, component, msg));
}

/// `I srv` with the server's own component name, the common case.
pub fn srv(msg: &str) {
    log(Level::Info, Sub::Srv, "frink_server", msg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// **The columns are llama.cpp's**, measured from its own output.
    #[test]
    fn a_line_matches_llama_cpps_columns() {
        let got = line(
            Duration::from_micros(1_255_290),
            Level::Info,
            Sub::Srv,
            "frink_server",
            "listening on http://127.0.0.1:8383",
        );
        assert_eq!(
            got,
            "0.01.255.290 I srv  frink_server: listening on http://127.0.0.1:8383"
        );
    }

    /// The elapsed field is minutes.seconds.millis.micros, which is
    /// what `0.01.255.290` means upstream -- not seconds with a
    /// four-part fraction.
    #[test]
    fn elapsed_rolls_over_into_minutes() {
        assert_eq!(elapsed(Duration::from_micros(100_492)), "0.00.100.492");
        assert_eq!(elapsed(Duration::from_micros(1_255_290)), "0.01.255.290");
        // 61.5 s is one minute and 1.5 seconds, not "0.61...".
        assert_eq!(elapsed(Duration::from_micros(61_500_000)), "1.01.500.000");
    }

    /// **Shorter component names are padded, not left ragged.**
    ///
    /// This is the whole reason the width is a constant: upstream pads
    /// `load_model` and `init` out to meet `llama_server`, and a
    /// message that did not would break the column a reader scans.
    #[test]
    fn short_components_are_right_aligned() {
        let a = line(Duration::ZERO, Level::Info, Sub::Srv, "load_model", "x");
        let b = line(Duration::ZERO, Level::Info, Sub::Cmn, "init", "x");
        let c = line(Duration::ZERO, Level::Info, Sub::Srv, "frink_server", "x");
        for l in [&a, &b, &c] {
            assert_eq!(
                l.find(':').expect("a colon"),
                c.find(':').expect("a colon"),
                "the colons do not line up:\n{a}\n{b}\n{c}"
            );
        }
        assert!(a.contains("srv    load_model:"), "{a}");
        assert!(b.contains("cmn          init:"), "{b}");
    }

    /// Levels are one letter, as upstream.
    #[test]
    fn levels_are_one_letter() {
        for (lvl, ch) in [(Level::Info, 'I'), (Level::Warn, 'W')] {
            let l = line(Duration::ZERO, lvl, Sub::Srv, "x", "m");
            assert!(
                l.contains(&format!(" {ch} srv")),
                "level {lvl:?} should print {ch}: {l}"
            );
        }
    }
}
