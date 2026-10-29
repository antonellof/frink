//! The llama.cpp side of `frink bench --compare`: run `llama-bench`,
//! read its numbers AND the build that produced them, and carry that
//! build through the receipt into `frink bench --render`.
//!
//! Split out of `bench_model` (981 lines) because the reference is its
//! own concept, and because the writer and the reader of the receipt's
//! `llama_cpp` field have to agree. Both go through
//! [`LlamaIdentity::from_receipt_field`], so a field the renderer cannot
//! read is a field the writer refuses (ADR 0004).
//!
//! # Why the build is recorded
//!
//! ADR 0006: a gap is only meaningful against a named reference. Two
//! llama.cpp builds have been measured apart on the same file, so a
//! `llama_tps` with no commit beside it is half a measurement, and two
//! gaps taken against different builds are not one comparison.

use std::collections::BTreeMap;

/// The receipt key that holds the llama.cpp identity.
pub(crate) const RECEIPT_KEY: &str = "llama_cpp";

/// What `llama-bench` printed: per-test tok/s plus whatever it said
/// about its own build. Fields stay optional here because this is a
/// parse of someone else's output; [`LlamaRun::identity`] is where a
/// missing one becomes a refusal.
#[derive(Debug, Default)]
pub(crate) struct LlamaRun {
    /// `{test -> avg tok/s}`, keyed the way llama-bench's markdown names
    /// tests (`pp512`, `tg128`), which is how frink names its rows.
    pub tps: BTreeMap<String, f64>,
    build_commit: Option<String>,
    build_number: Option<i64>,
    backends: Option<String>,
    cpu_info: Option<String>,
    gpu_info: Option<String>,
    /// The argv frink ran, so the comparison can be re-run from the
    /// receipt.
    command: Vec<String>,
}

/// The llama.cpp build behind a receipt's `llama_tps`, as recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LlamaIdentity {
    /// `build_commit` as llama-bench reports it (llama.cpp's
    /// `git rev-parse --short HEAD` at build time).
    pub build_commit: String,
    /// `build_number`: commit count at build time. Recorded as reported.
    pub build_number: Option<i64>,
    /// `backends`: the ggml backends compiled in (`Metal,BLAS`,
    /// `CUDA`, `CPU`). llama-bench reports no compiler flags, so this is
    /// the build configuration it does report.
    pub backends: String,
    pub cpu_info: Option<String>,
    pub gpu_info: Option<String>,
    pub command: Vec<String>,
}

/// A build_commit llama.cpp writes when it was built outside git. It
/// names no commit, so it is treated as missing rather than recorded.
const UNKNOWN_COMMIT: &str = "unknown";

impl LlamaRun {
    /// The recorded identity, or the names of the fields llama-bench did
    /// not report.
    pub fn identity(&self) -> Result<LlamaIdentity, Vec<&'static str>> {
        let mut missing = Vec::new();
        let commit = self
            .build_commit
            .as_deref()
            .map(str::trim)
            .filter(|c| !c.is_empty() && *c != UNKNOWN_COMMIT);
        if commit.is_none() {
            missing.push("build_commit");
        }
        let backends = self
            .backends
            .as_deref()
            .map(str::trim)
            .filter(|b| !b.is_empty());
        if backends.is_none() {
            missing.push("backends");
        }
        match (commit, backends) {
            (Some(c), Some(b)) => Ok(LlamaIdentity {
                build_commit: c.to_string(),
                build_number: self.build_number,
                backends: b.to_string(),
                cpu_info: self.cpu_info.clone(),
                gpu_info: self.gpu_info.clone(),
                command: self.command.clone(),
            }),
            _ => Err(missing),
        }
    }

    /// The value written under [`RECEIPT_KEY`]: the identity when
    /// llama-bench reported one, `null` when it did not (or did not run).
    /// [`check_receipt`] decides whether `null` is acceptable.
    pub fn receipt_field(run: Option<&LlamaRun>) -> serde_json::Value {
        run.and_then(|r| r.identity().ok())
            .map(|id| id.to_json())
            .unwrap_or(serde_json::Value::Null)
    }

    /// Names what is missing, for the write-time refusal.
    fn missing(run: Option<&LlamaRun>) -> Vec<&'static str> {
        match run {
            None => vec!["build_commit", "backends"],
            Some(r) => r.identity().err().unwrap_or_default(),
        }
    }
}

impl LlamaIdentity {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "build_commit": self.build_commit,
            "build_number": self.build_number,
            "backends": self.backends,
            "cpu_info": self.cpu_info,
            "gpu_info": self.gpu_info,
            "command": self.command,
        })
    }

    /// Reads the field back. The only reader of [`RECEIPT_KEY`]; the
    /// write-time check calls it too, so a receipt that is written is a
    /// receipt that renders with its reference.
    pub fn from_receipt_field(v: &serde_json::Value) -> Option<LlamaIdentity> {
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
        let commit = s("build_commit").filter(|c| !c.is_empty() && c != UNKNOWN_COMMIT)?;
        let backends = s("backends").filter(|b| !b.is_empty())?;
        Some(LlamaIdentity {
            build_commit: commit,
            build_number: v.get("build_number").and_then(|x| x.as_i64()),
            backends,
            cpu_info: s("cpu_info"),
            gpu_info: s("gpu_info"),
            command: v
                .get("command")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
        })
    }
}

/// Refuses a receipt that carries a llama.cpp number but not the build
/// that produced it (ADR 0006). Reads the assembled `tests` and the
/// field exactly as they will be written, not the run they came from.
pub(crate) fn check_receipt(
    tests: &[serde_json::Value],
    field: &serde_json::Value,
    run: Option<&LlamaRun>,
) -> anyhow::Result<()> {
    let with_number: Vec<&str> = tests
        .iter()
        .filter(|t| t.get("llama_tps").is_some_and(|v| !v.is_null()))
        .map(|t| t.get("test").and_then(|v| v.as_str()).unwrap_or("?"))
        .collect();
    if with_number.is_empty() || LlamaIdentity::from_receipt_field(field).is_some() {
        return Ok(());
    }
    anyhow::bail!(
        "refusing to write a receipt with a llama.cpp number ({}) but no recorded \
         llama.cpp build: llama-bench did not report {}. A gap is only meaningful \
         against a named reference (ADR 0006); rebuild llama.cpp from a git checkout \
         so llama-bench reports its commit, or run without --compare.",
        with_number.join(", "),
        LlamaRun::missing(run).join(" or "),
    )
}

/// Which llama.cpp build a rendered row was compared against.
///
/// Ordered and comparable so `--render` can group by it: rows with
/// different references never share a table.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Reference {
    /// Recorded: `(build_commit, build_number, backends)`.
    Known(String, Option<i64>, String),
    /// The receipt records that there was no llama.cpp number (the
    /// field is `null`), so there is no reference to name.
    NoComparison,
    /// The receipt predates the field. The build is not known, and none
    /// is made up.
    Unknown,
}

impl Reference {
    pub fn from_receipt(receipt: &serde_json::Value) -> Reference {
        match receipt.get(RECEIPT_KEY) {
            None => Reference::Unknown,
            Some(serde_json::Value::Null) => Reference::NoComparison,
            Some(v) => match LlamaIdentity::from_receipt_field(v) {
                Some(id) => Reference::Known(id.build_commit, id.build_number, id.backends),
                None => Reference::Unknown,
            },
        }
    }

    /// A short cell for the summary table.
    pub fn short(&self) -> String {
        match self {
            Reference::Known(commit, _, _) => format!("`{commit}`"),
            Reference::NoComparison => "none".into(),
            Reference::Unknown => "unknown".into(),
        }
    }

    /// The sentence under a section heading.
    pub fn describe(&self) -> String {
        match self {
            Reference::Known(commit, number, backends) => {
                let number = number.map(|n| format!("build {n}, ")).unwrap_or_default();
                format!("llama.cpp `{commit}` ({number}{backends})")
            }
            Reference::NoComparison => {
                "no llama.cpp build (these receipts carry no llama.cpp number)".into()
            }
            Reference::Unknown => "llama.cpp build **unknown**: these receipts predate \
                                   recording it, so the build behind their llama.cpp \
                                   column is not known"
                .into(),
        }
    }
}

/// Runs `llama-bench` on the same GGUF with matching workload flags and
/// reads its JSON output: per-test tok/s and the build that ran.
///
/// Thread count is deliberately *not* forced. Each engine picking its
/// own default is the comparison that means something: llama.cpp
/// defaults to performance cores and degrades sharply above them, so
/// pinning both engines to the same oversubscribed count (as the old
/// suite did with `-t 10`) handicaps llama by 2-4x on Apple Silicon and
/// flatters frink.
pub(crate) fn run_llama_bench(
    model: &str,
    n_prompt: usize,
    n_gen: usize,
    reps: usize,
    ngl: usize,
) -> anyhow::Result<LlamaRun> {
    let args: Vec<String> = [
        "-m",
        model,
        "-p",
        &n_prompt.to_string(),
        "-n",
        &n_gen.to_string(),
        "-r",
        &reps.to_string(),
        "-ngl",
        &ngl.to_string(),
        "-o",
        "json",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let out = std::process::Command::new("llama-bench")
        .args(&args)
        .output()
        .map_err(|e| anyhow::anyhow!("could not run llama-bench (is it on PATH?): {e}"))?;
    if !out.status.success() {
        anyhow::bail!("llama-bench exited with {}", out.status);
    }
    let mut run = parse_llama_bench_json(&String::from_utf8_lossy(&out.stdout))?;
    run.command = std::iter::once("llama-bench".to_string())
        .chain(args)
        .collect();
    Ok(run)
}

/// The test name llama-bench's markdown printer gives an entry, so the
/// JSON rows key the same way frink's rows do.
fn test_name(n_prompt: i64, n_gen: i64, n_depth: i64) -> String {
    let mut name = if n_prompt > 0 && n_gen == 0 {
        format!("pp{n_prompt}")
    } else if n_gen > 0 && n_prompt == 0 {
        format!("tg{n_gen}")
    } else {
        format!("pp{n_prompt}+tg{n_gen}")
    };
    if n_depth > 0 {
        name.push_str(&format!(" @ d{n_depth}"));
    }
    name
}

/// Parses `llama-bench -o json`: an array of one object per test, each
/// carrying `build_commit`, `build_number`, `backends`, the workload and
/// `avg_ts`. Entries that disagree about the build are refused: one
/// binary cannot have produced them, and one receipt names one build.
fn parse_llama_bench_json(text: &str) -> anyhow::Result<LlamaRun> {
    let v: serde_json::Value = serde_json::from_str(text.trim())
        .map_err(|e| anyhow::anyhow!("llama-bench -o json printed something unparsable: {e}"))?;
    let entries = v
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("llama-bench -o json did not print an array"))?;
    let mut run = LlamaRun::default();
    for (i, e) in entries.iter().enumerate() {
        let s = |k: &str| e.get(k).and_then(|x| x.as_str()).map(str::to_string);
        let n = |k: &str| e.get(k).and_then(|x| x.as_i64()).unwrap_or(0);
        let reported = (
            s("build_commit"),
            e.get("build_number").and_then(|x| x.as_i64()),
        );
        if i == 0 {
            (run.build_commit, run.build_number) = reported;
            run.backends = s("backends");
            run.cpu_info = s("cpu_info");
            run.gpu_info = s("gpu_info");
        } else if reported != (run.build_commit.clone(), run.build_number)
            || s("backends") != run.backends
        {
            anyhow::bail!(
                "llama-bench entries disagree about their build ({:?} vs {:?}); \
                 one receipt names one llama.cpp build",
                run.build_commit,
                reported.0
            );
        }
        if let Some(ts) = e.get("avg_ts").and_then(|x| x.as_f64()) {
            run.tps
                .insert(test_name(n("n_prompt"), n("n_gen"), n("n_depth")), ts);
        }
    }
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `llama-bench -o json` at the pinned commit, trimmed to the fields
    /// its JSON printer always writes that this file reads, plus a few
    /// it ignores. Shape from `json_printer` in
    /// `tools/llama-bench/llama-bench.cpp`.
    fn captured(commit: &str, number: i64, backends: &str) -> String {
        let entry = |n_prompt: i64, n_gen: i64, ts: f64| {
            format!(
                r#"  {{
    "build_commit": "{commit}",
    "build_number": {number},
    "cpu_info": "Apple M2 Pro",
    "gpu_info": "Apple M2 Pro",
    "backends": "{backends}",
    "model_filename": "models/smollm2-135m-q8_0.gguf",
    "model_type": "llama ?B Q8_0",
    "n_threads": 6,
    "n_gpu_layers": 99,
    "type_k": "f16",
    "n_prompt": {n_prompt},
    "n_gen": {n_gen},
    "n_depth": 0,
    "test_time": "2026-10-28T10:00:00Z",
    "avg_ns": 123,
    "stddev_ns": 4,
    "avg_ts": {ts},
    "stddev_ts": 1.5,
    "samples_ns": [ 1, 2, 3 ],
    "samples_ts": [ {ts}, {ts}, {ts} ]
  }}"#
            )
        };
        format!(
            "[\n{},\n{}\n]\n",
            entry(512, 0, 2779.28),
            entry(0, 128, 140.5)
        )
    }

    /// The numbers AND the build come out of one captured run, keyed the
    /// way frink names its rows.
    ///
    /// Sabotage: read `build_number` from `"n_threads"`; the identity's
    /// number goes to 6 and this goes red.
    #[test]
    fn a_captured_llama_bench_run_yields_its_numbers_and_its_build() {
        let run = parse_llama_bench_json(&captured("b1a2c3d", 6543, "Metal,BLAS")).unwrap();
        assert_eq!(run.tps.get("pp512"), Some(&2779.28));
        assert_eq!(run.tps.get("tg128"), Some(&140.5));
        let id = run.identity().expect("identity");
        assert_eq!(id.build_commit, "b1a2c3d");
        assert_eq!(id.build_number, Some(6543));
        assert_eq!(id.backends, "Metal,BLAS");
        assert_eq!(id.gpu_info.as_deref(), Some("Apple M2 Pro"));
    }

    /// llama.cpp built outside git reports `unknown` and an empty
    /// backend list would name nothing: both are missing, by name.
    ///
    /// Sabotage: drop the `UNKNOWN_COMMIT` filter in `identity`; the
    /// commit is accepted and this goes red.
    #[test]
    fn an_unnamed_build_is_missing_rather_than_recorded() {
        let run = parse_llama_bench_json(&captured("unknown", 0, "")).unwrap();
        assert_eq!(
            run.identity().unwrap_err(),
            vec!["build_commit", "backends"]
        );
    }

    /// Two builds in one output cannot be one reference.
    #[test]
    fn entries_that_disagree_about_the_build_are_refused() {
        let a = captured("aaaaaaa", 1, "CPU");
        let b = captured("bbbbbbb", 2, "CPU");
        let mixed = format!(
            "{},{}",
            a.trim_end().trim_end_matches(']'),
            b.trim_start().trim_start_matches('[')
        );
        let err = parse_llama_bench_json(&mixed).unwrap_err().to_string();
        assert!(err.contains("disagree"), "{err}");
    }

    #[test]
    fn test_names_match_llama_benchs_markdown_printer() {
        assert_eq!(test_name(512, 0, 0), "pp512");
        assert_eq!(test_name(0, 128, 0), "tg128");
        assert_eq!(test_name(512, 128, 0), "pp512+tg128");
        assert_eq!(test_name(512, 0, 4096), "pp512 @ d4096");
    }

    fn tests_with(llama: Option<f64>) -> Vec<serde_json::Value> {
        vec![serde_json::json!({"test": "tg128", "frink_tps": 10.0, "llama_tps": llama})]
    }

    /// The write-time refusal (ADR 0006): a llama.cpp number with no
    /// build beside it is not written, and the message names what is
    /// missing.
    ///
    /// Sabotage: make `check_receipt` return `Ok(())` unconditionally;
    /// this goes red.
    #[test]
    fn a_llama_number_without_a_recorded_build_is_refused_by_name() {
        let run = parse_llama_bench_json(&captured("unknown", 0, "CUDA")).unwrap();
        let field = LlamaRun::receipt_field(Some(&run));
        assert!(field.is_null());
        let err = check_receipt(&tests_with(Some(20.0)), &field, Some(&run))
            .unwrap_err()
            .to_string();
        assert!(err.contains("build_commit"), "{err}");
        assert!(!err.contains("backends"), "backends was reported: {err}");
        assert!(err.contains("tg128"), "{err}");
    }

    /// The refusal only fires on a number: a receipt with no llama.cpp
    /// column (llama-bench missing, or an arch it lacks) still writes.
    #[test]
    fn a_receipt_without_llama_numbers_needs_no_build() {
        check_receipt(&tests_with(None), &serde_json::Value::Null, None).unwrap();
    }

    /// What is written is what is read back: the writer and the
    /// renderer agree on one field.
    ///
    /// Sabotage: write the commit under `"commit"` in `to_json`; the
    /// round trip fails and this goes red.
    #[test]
    fn the_written_field_reads_back_as_the_same_reference() {
        let run = parse_llama_bench_json(&captured("b1a2c3d", 6543, "CUDA")).unwrap();
        let field = LlamaRun::receipt_field(Some(&run));
        check_receipt(&tests_with(Some(20.0)), &field, Some(&run)).unwrap();
        let receipt = serde_json::json!({ RECEIPT_KEY: field });
        assert_eq!(
            Reference::from_receipt(&receipt),
            Reference::Known("b1a2c3d".into(), Some(6543), "CUDA".into())
        );
    }

    /// A receipt from before the field is `Unknown`, never a guessed
    /// build; a receipt that recorded "no comparison" says so.
    #[test]
    fn old_receipts_have_an_unknown_reference_and_invent_none() {
        let old = serde_json::json!({"tests": []});
        assert_eq!(Reference::from_receipt(&old), Reference::Unknown);
        assert!(Reference::Unknown.describe().contains("unknown"));
        let none = serde_json::json!({ RECEIPT_KEY: null });
        assert_eq!(Reference::from_receipt(&none), Reference::NoComparison);
    }
}
