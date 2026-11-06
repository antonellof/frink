//! llama-bench's `-o` / `-oe` printers (`csv`, `json`, `jsonl`, `md`,
//! `sql`) for `frink bench -m`.
//!
//! Field names, field types and the printers' shapes are llama-bench's
//! (`tools/llama-bench/llama-bench.cpp`: `test::get_fields`,
//! `get_field_type`, `csv_printer`, `json_printer`, `jsonl_printer`,
//! `sql_printer`), so a script that reads `llama-bench -o json` reads
//! this. The field SET is the subset frink can state truthfully: a
//! field that describes one of llama.cpp's own knobs (`cpu_mask`,
//! `poll`, `split_mode`, `tensor_split`, `n_batch`, ...) is left out
//! rather than filled with a default frink does not have. `docs/CLI.md`
//! lists the difference.
//!
//! The statistics are llama-bench's too: `avg_ts` is the MEAN of the
//! per-repetition rates and `stddev_ts` the SAMPLE stddev (n - 1), as
//! `llama-bench.cpp:102-118` computes them, whatever the markdown table
//! prints (#481).

use std::fmt::Write as _;

/// One of llama-bench's output formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Csv,
    Json,
    Jsonl,
    Md,
    Sql,
}

impl OutputFormat {
    /// llama-bench's `output_format_from_str`, plus `none` where it
    /// accepts one (`-oe`).
    pub fn parse(s: &str, allow_none: bool) -> anyhow::Result<Option<Self>> {
        Ok(Some(match s {
            "csv" => Self::Csv,
            "json" => Self::Json,
            "jsonl" => Self::Jsonl,
            "md" => Self::Md,
            "sql" => Self::Sql,
            "none" if allow_none => return Ok(None),
            other => anyhow::bail!(
                "unknown output format `{other}`: llama-bench takes csv, json, jsonl, md or sql{}",
                if allow_none { " (or none)" } else { "" }
            ),
        }))
    }
}

/// llama-bench's `field_type`, which decides how a value is quoted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Str,
    Int,
    Float,
}

/// One value, already in llama-bench's string form.
#[derive(Clone, Debug)]
struct Field {
    name: &'static str,
    kind: Kind,
    value: String,
}

/// What a run knows about itself, shared by every test row.
pub struct RunInfo {
    pub build_commit: String,
    pub cpu_info: String,
    pub gpu_info: String,
    pub backends: String,
    pub model_filename: String,
    pub model_type: String,
    pub model_size: u64,
    pub model_n_params: u64,
    pub n_threads: usize,
    pub type_k: String,
    pub type_v: String,
    pub n_gpu_layers: usize,
}

/// One test (`pp<N>` or `tg<N>`) in llama-bench's terms.
pub struct TestRecord {
    fields: Vec<Field>,
    samples_ns: Vec<u64>,
    samples_ts: Vec<f64>,
}

/// llama-bench's `avg` over `u64` or `f64`.
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// llama-bench's `stdev`: the sample form, `n - 1`, and 0 below two.
fn sample_stddev(v: &[f64]) -> f64 {
    let n = v.len();
    if n <= 1 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
}

impl TestRecord {
    /// `samples_ts` are the per-repetition rates `frink bench` timed;
    /// `samples_ns` are the durations they came from (`n_tokens / ts`).
    pub fn new(
        run: &RunInfo,
        n_prompt: usize,
        n_gen: usize,
        test_time: String,
        samples_ts: &[f64],
    ) -> Self {
        let n_tokens = (n_prompt + n_gen) as f64;
        let samples_ns: Vec<u64> = samples_ts
            .iter()
            .map(|ts| (1e9 * n_tokens / ts).round() as u64)
            .collect();
        let ns: Vec<f64> = samples_ns.iter().map(|&n| n as f64).collect();
        let s = |name, value: &str| Field {
            name,
            kind: Kind::Str,
            value: value.to_string(),
        };
        let i = |name, value: u64| Field {
            name,
            kind: Kind::Int,
            value: value.to_string(),
        };
        let f = |name, value: f64| Field {
            name,
            kind: Kind::Float,
            value: format!("{value:.6}"),
        };
        let fields = vec![
            s("build_commit", &run.build_commit),
            s("cpu_info", &run.cpu_info),
            s("gpu_info", &run.gpu_info),
            s("backends", &run.backends),
            s("model_filename", &run.model_filename),
            s("model_type", &run.model_type),
            i("model_size", run.model_size),
            i("model_n_params", run.model_n_params),
            i("n_threads", run.n_threads as u64),
            s("type_k", &run.type_k),
            s("type_v", &run.type_v),
            i("n_gpu_layers", run.n_gpu_layers as u64),
            i("n_prompt", n_prompt as u64),
            i("n_gen", n_gen as u64),
            i("n_depth", 0),
            s("test_time", &test_time),
            i("avg_ns", mean(&ns) as u64),
            i("stddev_ns", sample_stddev(&ns) as u64),
            f("avg_ts", mean(samples_ts)),
            f("stddev_ts", sample_stddev(samples_ts)),
        ];
        Self {
            fields,
            samples_ns,
            samples_ts: samples_ts.to_vec(),
        }
    }
}

/// llama-bench's `escape_json`.
fn escape_json(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for c in v.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) <= 0x1f => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn json_value(f: &Field) -> String {
    match f.kind {
        Kind::Str => format!("\"{}\"", escape_json(&f.value)),
        Kind::Int | Kind::Float => f.value.clone(),
    }
}

fn join<T: std::fmt::Display>(v: &[T]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Renders every test in `format`. `Md` is not rendered here: the
/// markdown table is `frink bench`'s own and is printed by the caller.
pub fn render(format: OutputFormat, tests: &[TestRecord]) -> String {
    let names: Vec<&str> = tests
        .first()
        .map(|t| t.fields.iter().map(|f| f.name).collect())
        .unwrap_or_default();
    let mut out = String::new();
    match format {
        OutputFormat::Md => {}
        OutputFormat::Csv => {
            let _ = writeln!(out, "{}", names.join(","));
            for t in tests {
                let row: Vec<String> = t
                    .fields
                    .iter()
                    .map(|f| format!("\"{}\"", f.value.replace('"', "\"\"")))
                    .collect();
                let _ = writeln!(out, "{}", row.join(","));
            }
        }
        OutputFormat::Json => {
            out.push_str("[\n");
            for (k, t) in tests.iter().enumerate() {
                if k > 0 {
                    out.push_str(",\n");
                }
                out.push_str("  {\n");
                for f in &t.fields {
                    let _ = writeln!(out, "    \"{}\": {},", f.name, json_value(f));
                }
                let _ = writeln!(out, "    \"samples_ns\": [ {} ],", join(&t.samples_ns));
                let _ = writeln!(out, "    \"samples_ts\": [ {} ]", join(&t.samples_ts));
                out.push_str("  }");
            }
            out.push_str("\n]\n");
        }
        OutputFormat::Jsonl => {
            for t in tests {
                out.push('{');
                for f in &t.fields {
                    let _ = write!(out, "\"{}\": {}, ", f.name, json_value(f));
                }
                let _ = write!(out, "\"samples_ns\": [ {} ],", join(&t.samples_ns));
                let _ = writeln!(out, "\"samples_ts\": [ {} ]}}", join(&t.samples_ts));
            }
        }
        OutputFormat::Sql => {
            out.push_str("CREATE TABLE IF NOT EXISTS llama_bench (\n");
            if let Some(t) = tests.first() {
                for (k, f) in t.fields.iter().enumerate() {
                    let ty = match f.kind {
                        Kind::Str => "TEXT",
                        Kind::Int => "INTEGER",
                        Kind::Float => "REAL",
                    };
                    let sep = if k + 1 < t.fields.len() { "," } else { "" };
                    let _ = writeln!(out, "  {} {ty}{sep}", f.name);
                }
            }
            out.push_str(");\n\n");
            for t in tests {
                let values: Vec<String> =
                    t.fields.iter().map(|f| format!("'{}'", f.value)).collect();
                let _ = writeln!(
                    out,
                    "INSERT INTO llama_bench ({}) VALUES ({});",
                    names.join(", "),
                    values.join(", ")
                );
            }
        }
    }
    out
}

/// `%FT%TZ` for a time in seconds since the Unix epoch, as llama-bench's
/// `get_test_time` writes it (`std::gmtime` + `strftime`). Days to a
/// civil date by Howard Hinnant's `civil_from_days`.
pub fn iso8601_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Now, as [`iso8601_utc`].
pub fn now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    iso8601_utc(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run() -> RunInfo {
        RunInfo {
            build_commit: "abc1234".into(),
            cpu_info: "Apple M2 Pro".into(),
            gpu_info: String::new(),
            backends: "CPU".into(),
            model_filename: "m \"q\".gguf".into(),
            model_type: "llama Q4_K_M".into(),
            model_size: 1000,
            model_n_params: 135_000_000,
            n_threads: 6,
            type_k: "f16".into(),
            type_v: "f16".into(),
            n_gpu_layers: 0,
        }
    }

    fn record() -> TestRecord {
        TestRecord::new(
            &run(),
            0,
            128,
            "2026-11-06T10:00:00Z".into(),
            &[100.0, 200.0],
        )
    }

    /// llama-bench's statistics, not the table's: mean 150, sample
    /// stddev sqrt(5000) (n - 1), not the population sqrt(2500).
    #[test]
    fn avg_ts_is_the_mean_and_stddev_ts_the_sample_form() {
        let t = record();
        let get = |n: &str| t.fields.iter().find(|f| f.name == n).unwrap().value.clone();
        assert_eq!(get("avg_ts"), "150.000000");
        assert_eq!(get("stddev_ts"), format!("{:.6}", 5000f64.sqrt()));
        // 128 tokens at 100 and 200 t/s: 1.28 s and 0.64 s.
        assert_eq!(t.samples_ns, vec![1_280_000_000, 640_000_000]);
        assert_eq!(get("avg_ns"), "960000000");
    }

    /// The json printer parses as JSON, carries llama-bench's types
    /// (strings quoted and escaped, numbers bare) and ends with the two
    /// sample arrays.
    #[test]
    fn json_and_jsonl_parse_with_llama_benchs_types() {
        let json = render(OutputFormat::Json, &[record(), record()]);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let first = &v[0];
        assert_eq!(first["model_filename"], "m \"q\".gguf");
        assert_eq!(first["n_gen"], 128);
        assert_eq!(first["avg_ts"], 150.0);
        assert_eq!(first["samples_ts"][1], 200.0);
        assert_eq!(v.as_array().unwrap().len(), 2);
        for line in render(OutputFormat::Jsonl, &[record(), record()]).lines() {
            let l: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(l["n_prompt"], 0);
        }
    }

    /// CSV quotes every value and doubles embedded quotes; SQL declares
    /// llama-bench's table and column types.
    #[test]
    fn csv_and_sql_follow_llama_benchs_printers() {
        let csv = render(OutputFormat::Csv, &[record()]);
        let mut lines = csv.lines();
        assert!(lines.next().unwrap().starts_with("build_commit,cpu_info,"));
        assert!(lines.next().unwrap().contains("\"m \"\"q\"\".gguf\""));
        let sql = render(OutputFormat::Sql, &[record()]);
        assert!(sql.starts_with("CREATE TABLE IF NOT EXISTS llama_bench (\n"));
        assert!(sql.contains("  avg_ts REAL,\n"));
        assert!(sql.contains("  model_size INTEGER,\n"));
        assert!(sql.contains("INSERT INTO llama_bench (build_commit, "));
    }

    #[test]
    fn formats_parse_as_llama_bench_spells_them() {
        assert_eq!(
            OutputFormat::parse("jsonl", false).unwrap(),
            Some(OutputFormat::Jsonl)
        );
        assert_eq!(OutputFormat::parse("none", true).unwrap(), None);
        assert!(OutputFormat::parse("none", false).is_err());
        assert!(OutputFormat::parse("yaml", true).is_err());
    }

    #[test]
    fn test_time_is_iso8601_utc() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        // 2026-11-06T10:00:00Z
        assert_eq!(iso8601_utc(1_793_959_200), "2026-11-06T10:00:00Z");
        // A leap day.
        assert_eq!(iso8601_utc(951_782_400), "2000-02-29T00:00:00Z");
    }
}
