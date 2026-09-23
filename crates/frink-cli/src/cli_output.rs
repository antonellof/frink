//! What `frink run` / `frink cli` print, and the warmup that makes the
//! prompt figure mean something.
//!
//! # The number that was wrong
//!
//! The run path reported `prompt 5 tokens, 12.64 t/s` where llama.cpp
//! reported 197.7 t/s for the same prompt and the same file. That
//! reads as a 16x loss and is not one: the prompt timer started before
//! the first forward pass, so the one-time backend warmup -- Metal
//! pipeline construction, buffer allocation, the first dispatch of
//! every kernel -- was inside it and then divided by the prompt
//! length.
//!
//! Measured: prefill takes about 0.40 s for a 5-token prompt and about
//! 0.43 s for a 482-token one on an M2 Pro. The marginal cost of 477
//! further tokens is roughly 0.03 s, so the prompt rate at 482 tokens
//! reads 1108 t/s and at 5 tokens reads 12.64. Same engine, same
//! speed; one number is nearly all fixed cost.
//!
//! llama.cpp does not have this because `common_init_from_params` runs
//! a warmup decode at load and clears the KV before any timer starts.
//! [`warm_up`] is that, and it runs on its OWN caches so a warmup pass
//! cannot leave a token in the sequence the prompt is about to use --
//! which is the failure mode of warming in place and forgetting to
//! reset.
//!
//! # Why the reporting lives here
//!
//! Four run bodies each printed their own copy of the timing line, so
//! a change to the format was four edits and a change to the
//! ARITHMETIC was four chances to get it wrong. They call
//! [`Timings::print`] now.

use std::io::Write;
use std::sync::OnceLock;

use frink_models::{Decoder, Engine};

/// The device line, stashed between backend selection and the banner.
///
/// Backend selection happens before the model is opened and the banner
/// is printed after it, so the line cannot simply be printed where it
/// is computed without landing above the wordmark -- which is where
/// llama.cpp prints nothing at all. Threading it through every
/// intervening signature would touch four run bodies to move one
/// string, so it waits here instead.
static DEVICE_LINE: OnceLock<String> = OnceLock::new();

/// Records the device line for the banner to print. First call wins.
pub fn set_device_line(line: String) {
    let _ = DEVICE_LINE.set(line);
}

fn device_line() -> Option<&'static str> {
    DEVICE_LINE.get().map(String::as_str)
}

/// Runs one throwaway forward pass so the caller's timer does not
/// include first-dispatch cost.
///
/// On its own caches, dropped on return: warming the real ones would
/// put a token at position 0 of every layer, and every prompt after it
/// would be computed against a sequence that silently began with
/// something nobody asked for.
///
/// Token 0 (`BOS` on every vocabulary this engine loads) rather than a
/// sampled one, because the point is to touch the kernels, not to
/// produce anything.
pub fn warm_up(decoder: &Decoder) {
    let mut scratch = decoder.config.new_kv_caches_with_capacity(1);
    let _ = decoder.forward_token(0, 0, &mut scratch);
}

/// The same, for the dedicated engines (`MLA`, `Gemma4`, `GLM-5.2`),
/// which carry their own state type.
///
/// Generic over [`Engine`] rather than one function per engine: the
/// trait already gives both halves this needs, `new_state` for a
/// scratch sequence and `forward_token` to touch the kernels, so a
/// fourth engine gets warmed the day it is added and not the day
/// somebody notices its prompt figure is wrong.
pub fn warm_up_engine<E: Engine>(engine: &E) {
    let mut scratch = engine.new_state();
    let _ = engine.forward_token(0, 0, &mut scratch);
}

/// What a finished run reports.
///
/// Carries counts and DURATIONS rather than rates, so the division
/// happens in one place: two of the four copies this replaced guarded
/// a zero denominator and two did not.
pub struct Timings {
    pub prompt_tokens: usize,
    pub prompt_secs: f64,
    pub predicted_tokens: usize,
    pub predicted_secs: f64,
}

impl Timings {
    fn rate(tokens: usize, secs: f64) -> Option<f64> {
        (secs > 0.0 && tokens > 0).then(|| tokens as f64 / secs)
    }

    pub fn prompt_per_second(&self) -> Option<f64> {
        Self::rate(self.prompt_tokens, self.prompt_secs)
    }

    pub fn predicted_per_second(&self) -> Option<f64> {
        Self::rate(self.predicted_tokens, self.predicted_secs)
    }

    /// llama.cpp's closing line, in its shape.
    ///
    /// `[ Prompt: 197.7 t/s | Generation: 152.2 t/s ]` is what
    /// `llama cli` prints after an answer, and matching it is the
    /// point: a user comparing the two engines should not have to
    /// translate between two spellings of the same two numbers.
    ///
    /// A rate that was never measured prints as `-`, never as `0.0`.
    /// Zero is the slowest possible speed and would read as a
    /// measurement rather than as its absence.
    pub fn line(&self) -> String {
        let fmt = |r: Option<f64>| match r {
            Some(v) => format!("{v:.1} t/s"),
            None => "-".to_string(),
        };
        format!(
            "[ Prompt: {} | Generation: {} ]",
            fmt(self.prompt_per_second()),
            fmt(self.predicted_per_second())
        )
    }

    pub fn print(&self, w: &mut impl Write) -> std::io::Result<()> {
        writeln!(w, "\n{}", self.line())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(pt: usize, ps: f64, gt: usize, gs: f64) -> Timings {
        Timings {
            prompt_tokens: pt,
            prompt_secs: ps,
            predicted_tokens: gt,
            predicted_secs: gs,
        }
    }

    /// The line is llama.cpp's, character for character in its fixed
    /// parts, so a user running both sees one format.
    #[test]
    fn the_line_matches_llama_cpps_shape() {
        let line = t(197, 1.0, 152, 1.0).line();
        assert_eq!(line, "[ Prompt: 197.0 t/s | Generation: 152.0 t/s ]");
    }

    /// **A rate nobody measured is a dash, not a zero.**
    ///
    /// An empty prompt has no prompt rate. Reporting `0.0 t/s` there
    /// says the engine managed zero tokens per second, which is the
    /// slowest number printable and the opposite of the truth.
    #[test]
    fn an_unmeasured_rate_is_a_dash() {
        assert_eq!(t(0, 0.0, 8, 1.0).prompt_per_second(), None);
        assert!(
            t(0, 0.0, 8, 1.0).line().contains("Prompt: -"),
            "an unmeasured prompt rate must not print as a number"
        );
        // And a zero-length interval, which a fast enough prompt on a
        // coarse clock really produces.
        assert_eq!(t(5, 0.0, 8, 1.0).prompt_per_second(), None);
    }

    /// One divide, one guard. Two of the four copies this replaced
    /// checked the denominator and two checked nothing.
    #[test]
    fn both_rates_come_from_the_same_guard() {
        let x = t(10, 2.0, 30, 3.0);
        assert_eq!(x.prompt_per_second(), Some(5.0));
        assert_eq!(x.predicted_per_second(), Some(10.0));
    }
}

/// `frink version`, in the shape llama.cpp's launcher prints.
///
/// `llama version` answers with `version: X (build N, commit SHA)`
/// and a line naming the compiler and target. This carries the same
/// facts under the same labels; the build NUMBER has no frink
/// equivalent and is left out rather than invented, because a made-up
/// build number is worse than a missing one for anyone reporting a
/// bug against it.
pub fn version_block() -> String {
    let version = env!("CARGO_PKG_VERSION");
    let head = match option_env!("FRINK_GIT_SHA") {
        Some(sha) => format!("version: {version} (commit {sha})"),
        // A tarball build has no git metadata. Saying so beats
        // printing something that reads like a commit.
        None => format!("version: {version}"),
    };
    format!(
        "{head}\nbuilt for {}\n",
        option_env!("FRINK_TARGET").unwrap_or("unknown")
    )
}

/// `frink licenses`, the third-party notices.
///
/// Embedded at compile time rather than read from disk: the notices
/// are a LICENCE OBLIGATION, and an installed binary that could only
/// satisfy it when run from a checkout would not satisfy it at all.
pub fn licenses_block() -> std::io::Result<String> {
    Ok(include_str!("../../../docs/THIRD_PARTY_NOTICES.md").to_string())
}

/// The block `llama cli` prints once a model is open.
///
/// Four labels, colon-aligned at column 12, in llama.cpp's order:
///
/// ```text
/// build      : b10964-b29c606e2
/// model      : models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
/// ftype      : Q4_K - Medium
/// modalities : text
/// ```
///
/// Same labels and same alignment, with frink's own values. The
/// alignment is part of the shape a user recognises, so it is a
/// constant here rather than four `format!` widths that can disagree.
pub struct Banner<'a> {
    pub model_path: &'a str,
    /// The GGUF file type as its name, e.g. `Q4_K - Medium`.
    pub ftype: &'a str,
    /// `text`, or `text, vision` once anything else is served.
    pub modalities: &'a str,
}

/// Where llama.cpp's labels put the colon.
const LABEL_WIDTH: usize = 10;

impl Banner<'_> {
    pub fn render(&self) -> String {
        let build = match option_env!("FRINK_GIT_SHA") {
            Some(sha) => format!("{}-{sha}", env!("CARGO_PKG_VERSION")),
            None => env!("CARGO_PKG_VERSION").to_string(),
        };
        let row = |k: &str, v: &str| format!("{k:<LABEL_WIDTH$} : {v}\n");
        // A fifth row llama.cpp does not have. Frink can run the same
        // file on three backends with a selectable KV dtype, and which
        // one it picked is the first thing anybody asks about a
        // number. It is a ROW rather than the loose line it used to be
        // so the block keeps its shape.
        let device = device_line()
            .map(|d| {
                let v = d
                    .trim_start_matches("frink: ")
                    .trim_start_matches("device=");
                row("device", v)
            })
            .unwrap_or_default();
        format!(
            "{}{}{}{}{}",
            row("build", &build),
            row("model", self.model_path),
            row("ftype", self.ftype),
            row("modalities", self.modalities),
            device,
        )
    }

    pub fn print(&self, w: &mut impl Write) -> std::io::Result<()> {
        write!(w, "\n{}\n", self.render())
    }
}

#[cfg(test)]
mod banner_tests {
    use super::*;

    fn banner() -> Banner<'static> {
        Banner {
            model_path: "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            ftype: "Q4_K - Medium",
            modalities: "text",
        }
    }

    /// The labels and their alignment are llama.cpp's, because that
    /// alignment is what a user recognises at a glance.
    #[test]
    fn the_banner_matches_llama_cpps_labels_and_alignment() {
        let out = banner().render();
        let lines: Vec<&str> = out.lines().collect();
        // Four fixed rows; `device` is present only once a backend has
        // been selected, which a unit test has not done.
        assert!(
            (4..=5).contains(&lines.len()),
            "four rows, plus device when known:\n{out}"
        );

        for (i, key) in ["build", "model", "ftype", "modalities"].iter().enumerate() {
            assert!(
                lines[i].starts_with(key),
                "row {i} should be `{key}`, is: {}",
                lines[i]
            );
        }
        // Every colon in the same column, which is the visual shape.
        let cols: Vec<usize> = lines
            .iter()
            .map(|l| l.find(':').expect("every row has a colon"))
            .collect();
        assert!(
            cols.windows(2).all(|w| w[0] == w[1]),
            "the colons are not aligned: {cols:?}\n{out}"
        );
        // `modalities` is the longest label and sets the width, so it
        // is the one that proves the column is wide enough rather than
        // merely consistent.
        assert_eq!(
            cols[3],
            LABEL_WIDTH + 1,
            "the longest label should sit flush against its colon"
        );
    }

    /// The values are the ones handed in, not re-derived.
    #[test]
    fn the_banner_prints_the_values_it_is_given() {
        let out = banner().render();
        assert!(out.contains("models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"));
        assert!(out.contains("Q4_K - Medium"));
        assert!(out.contains("modalities : text\n"));
    }
}

/// `general.file_type` as llama.cpp names it.
///
/// The values are `llama_ftype` (`llama.h`), and the strings are
/// `llama_model_ftype_name`'s (`llama-model.cpp`), so the banner reads
/// the same as llama.cpp's for the same file.
///
/// A value this table does not carry reports the NUMBER rather than a
/// guess: an unknown ftype printed as a plausible name is worse than
/// one printed as `ftype 38`, because only the second sends somebody
/// to look it up.
pub fn ftype_name(ftype: Option<u64>) -> String {
    let Some(v) = ftype else {
        return "unknown".to_string();
    };
    let name = match v {
        0 => "all F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K - Medium",
        11 => "Q3_K - Small",
        12 => "Q3_K - Medium",
        13 => "Q3_K - Large",
        14 => "Q4_K - Small",
        15 => "Q4_K - Medium",
        16 => "Q5_K - Small",
        17 => "Q5_K - Medium",
        18 => "Q6_K",
        19 => "IQ2_XXS - 2.0625 bpw",
        20 => "IQ2_XS - 2.3125 bpw",
        21 => "Q2_K - Small",
        22 => "IQ3_XS - 3.3 bpw",
        23 => "IQ3_XXS - 3.0625 bpw",
        24 => "IQ1_S - 1.5625 bpw",
        25 => "IQ4_NL - 4.5 bpw",
        26 => "IQ3_S - 3.4375 bpw",
        27 => "IQ3_M - 3.66 bpw",
        28 => "IQ2_S - 2.5 bpw",
        29 => "IQ2_M - 2.7 bpw",
        30 => "IQ4_XS - 4.25 bpw",
        31 => "IQ1_M - 1.75 bpw",
        32 => "BF16",
        36 => "TQ1_0 - 1.69 bpw ternary",
        37 => "TQ2_0 - 2.06 bpw ternary",
        38 => "MXFP4 MoE",
        _ => return format!("ftype {v}"),
    };
    name.to_string()
}

#[cfg(test)]
mod ftype_tests {
    use super::*;

    /// The file this project benchmarks against, by its own number.
    #[test]
    fn the_common_quants_read_as_llama_cpp_names_them() {
        assert_eq!(ftype_name(Some(15)), "Q4_K - Medium");
        assert_eq!(ftype_name(Some(7)), "Q8_0");
        assert_eq!(ftype_name(Some(1)), "F16");
    }

    /// **An unknown value is a number, never a name.**
    ///
    /// The table will go stale the next time llama.cpp adds a quant.
    /// When it does, the banner has to say so rather than pick the
    /// nearest row: `ftype 99` sends somebody to `llama.h`, and a
    /// wrong name sends them nowhere.
    #[test]
    fn an_unknown_ftype_reports_its_number() {
        assert_eq!(ftype_name(Some(99)), "ftype 99");
        assert_eq!(ftype_name(None), "unknown");
    }
}

/// The wordmark `llama cli` prints above its banner.
///
/// Same slot, same block-glyph style, frink's own letters -- this is
/// the one place where matching llama.cpp would mean printing their
/// name, which is the opposite of the point.
const LOGO: &str = "\
▄▄▄▄▄ ▄▄▄▄  ▄▄ ▄▄  ▄▄ ▄▄  ▄▄ ▄▄
██    ██  ▄ ██ ██  ██ ██▄ ██ ██ ▄▀
██▀▀  ██▀▀  ██ ██  ██ ██ ▀██ ██▀█
██    ██    ██ ██  ██ ██  ██ ██ ▀▄
██    ██    ██ ▀█▄▄█▀ ██  ██ ██  ▀\
";

/// Prints the wordmark, once, before the banner.
pub fn print_logo(w: &mut impl Write) -> std::io::Result<()> {
    writeln!(w, "\n{LOGO}")
}

/// Echoes the prompt the way `llama cli` does, so a transcript of
/// either engine reads the same.
pub fn print_prompt_echo(w: &mut impl Write, prompt: &str) -> std::io::Result<()> {
    // One line, whatever the prompt's own line count: llama.cpp shows
    // the turn, not the rendered template, and a 700-token system
    // prompt echoed in full would bury the answer.
    let first = prompt.lines().next().unwrap_or("").trim();
    let shown = if prompt.lines().count() > 1 || first.chars().count() > 120 {
        let cut: String = first.chars().take(117).collect();
        format!("{cut}...")
    } else {
        first.to_string()
    };
    writeln!(w, "\n> {shown}")
}

/// llama.cpp's closing word.
pub fn print_exiting(w: &mut impl Write) -> std::io::Result<()> {
    writeln!(w, "\nExiting...")
}

#[cfg(test)]
mod presentation_tests {
    use super::*;

    /// The wordmark says frink. Matching llama.cpp's OUTPUT does not
    /// extend to printing their name, and a test says so because this
    /// is the kind of thing a copy-paste gets wrong.
    #[test]
    fn the_logo_is_frinks_own() {
        let flat: String = LOGO.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(!flat.is_empty(), "the wordmark is blank");
        assert!(
            !LOGO.to_ascii_lowercase().contains("llama"),
            "the wordmark must not carry another project's name"
        );
        assert_eq!(LOGO.lines().count(), 5, "five rows, as llama.cpp's has");
    }

    /// A long or multi-line prompt is shortened, because the echo
    /// exists to show the turn and not to reprint the input.
    #[test]
    fn a_long_prompt_is_shortened_in_the_echo() {
        let mut out = Vec::new();
        let long = "x".repeat(400);
        print_prompt_echo(&mut out, &long).expect("write");
        let s = String::from_utf8(out).expect("utf8");
        assert!(s.contains("..."), "a long prompt must be elided: {s}");
        assert!(s.len() < 200, "the echo is still long: {} chars", s.len());

        let mut multi = Vec::new();
        print_prompt_echo(&mut multi, "first line\nsecond line").expect("write");
        let s = String::from_utf8(multi).expect("utf8");
        assert!(s.contains("first line"), "{s}");
        assert!(!s.contains("second line"), "only the first line: {s}");
    }

    /// A short single-line prompt is echoed whole, which is the common
    /// case and the one that should look exactly like llama.cpp's.
    #[test]
    fn a_short_prompt_is_echoed_whole() {
        let mut out = Vec::new();
        print_prompt_echo(&mut out, "Capital of France?").expect("write");
        assert_eq!(
            String::from_utf8(out).expect("utf8"),
            "\n> Capital of France?\n"
        );
    }
}
