//! `frink serve-bench`: what `frink-server` does under concurrency.
//!
//! `frink bench` measures kernels against `llama-bench` and is
//! deliberately single-stream and HTTP-free. That is the right tool for
//! "how fast is this matvec" and no tool at all for "what is the p99
//! time-to-first-token at sixteen concurrent clients", which is the
//! question an operator sizing a deployment is actually asking.
//!
//! Every rule that decides whether the numbers mean anything lives in
//! [`crate::bench_client`], with no socket in it, so each one is
//! asserted on any host rather than inferred from a live server that
//! might have been slow for an unrelated reason. This module is the
//! socket: it opens connections, dispatches on a schedule, tics per
//! streamed chunk, and hands the tics to that arithmetic.
//!
//! The HTTP client is hand-rolled on `TcpStream` for the same reason
//! `chat` is: the CLI links no HTTP stack, and a benchmark that pulled
//! in a TLS-capable one would measure the client's connection handling
//! as much as the server's.

use std::io::BufRead;
use std::sync::mpsc;
use std::time::Instant;

use crate::bench_client::{
    is_token_chunk, BenchReport, BenchSampling, Latency, PrefixTotals, RequestTiming,
};
use anyhow::Result;
use clap::Parser;
use serde_json::{json, Value};

use crate::http;

#[derive(Parser, Debug)]
pub struct ServeBenchArgs {
    /// Base URL of a running `frink-server`.
    #[arg(long, default_value = "http://127.0.0.1:8383")]
    pub url: String,
    /// Requests to send in total.
    #[arg(long, default_value_t = 32)]
    pub requests: usize,
    /// Requests in flight at once.
    #[arg(long, default_value_t = 8)]
    pub concurrency: usize,
    /// Tokens every request must produce, exactly.
    #[arg(long, default_value_t = 128)]
    pub output_len: usize,
    /// Prompt length, in characters of the generated filler.
    ///
    /// Characters and not tokens, deliberately. An exact TOKEN length
    /// needs the server's own tokenizer driven to a fixed point --
    /// decoding ids to text and encoding it back is not the identity,
    /// since the tokenizer merges adjacent pieces, so a prompt built
    /// from N ids re-encodes shorter -- and every step of that loop is
    /// a round trip against the server this command is about to
    /// benchmark. The reported prompt-token count is the server's, so a
    /// run says what it really sent rather than what it meant to.
    #[arg(long, default_value_t = 512)]
    pub prompt_chars: usize,
    /// A system prompt of this many characters, IDENTICAL across every
    /// request, so the run measures prefix reuse instead of avoiding
    /// it.
    ///
    /// Off by default, because `--prompt-chars` filler is deliberately
    /// built to defeat a prefix cache and the two would otherwise
    /// contradict each other. With this set, every request carries the
    /// same system message and its own filler, which is the shape a
    /// real deployment has: one agent prompt, many questions.
    ///
    /// The reported reuse is the SERVER's `usage.cached_tokens`, not
    /// an inference from latency, so a run says what was actually
    /// reused rather than what was probably reused.
    #[arg(long, default_value_t = 0)]
    pub shared_prefix: usize,
    /// Send this `cache_salt` with every request.
    ///
    /// A prefix cache namespace. Two runs under different salts share
    /// no pages even with identical prompts, which is the property
    /// worth checking on a multi-tenant deployment.
    #[arg(long)]
    pub cache_salt: Option<String>,
    /// Model name to send. Optional: the server serves whatever is
    /// loaded regardless.
    #[arg(long)]
    pub model: Option<String>,
    /// Emit the report as JSON instead of a table.
    #[arg(long)]
    pub json: bool,
}

pub fn run_serve_bench(args: ServeBenchArgs) -> Result<()> {
    if args.requests == 0 {
        anyhow::bail!("--requests must be at least 1");
    }
    if args.concurrency == 0 {
        anyhow::bail!("--concurrency must be at least 1");
    }
    if args.output_len == 0 {
        anyhow::bail!("--output-len must be at least 1: a request that generates nothing has no latency to measure");
    }
    let base = args.url.trim_end_matches('/').to_string();
    // Fail before the run rather than N times inside it: a connection
    // refused per worker buries the one useful line under noise.
    http::parse_url(&format!("{base}/v1/chat/completions"))?;

    let sampling = BenchSampling::new(args.output_len);
    // Built once and shared by reference: a per-request copy would be
    // the same text, but building it per request inside the timed
    // section would charge the client's allocator to the server.
    let shared = (args.shared_prefix > 0).then(|| filler_prompt(args.shared_prefix, 0));
    let started = Instant::now();

    // A shared work counter rather than a slice per worker: requests
    // are not equal-cost, so a static split leaves the fastest worker
    // idle while the slowest still has a queue -- which shows up as the
    // server being slower at the end of a run than the start.
    let (tx, rx) = mpsc::channel::<usize>();
    for i in 0..args.requests {
        tx.send(i)
            .expect("the receiver is alive until the scope ends");
    }
    drop(tx);
    let rx = std::sync::Mutex::new(rx);

    let timings: Vec<RequestTiming> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..args.concurrency.min(args.requests))
            .map(|_| {
                let base = &base;
                let prompt_chars = args.prompt_chars;
                let model = &args.model;
                let shared = shared.as_deref();
                let salt = args.cache_salt.as_deref();
                let rx = &rx;
                scope.spawn(move || {
                    let mut mine = Vec::new();
                    loop {
                        let Ok(index) = rx.lock().unwrap_or_else(|p| p.into_inner()).recv() else {
                            break;
                        };
                        // Per request, from the work item's own index:
                        // two requests must not share a prompt, or the
                        // second measures the first's pages.
                        let prompt = filler_prompt(prompt_chars, index + 1);
                        let body = request_body(model.as_deref(), shared, salt, &prompt, &sampling);
                        let dispatched = started.elapsed().as_secs_f64();
                        let mut timing = RequestTiming::started(dispatched);
                        // A failed request contributes a timing with no
                        // tics, which the report counts as failed rather
                        // than dropping: a run where a third of the
                        // requests failed and the rest were fast is not
                        // a fast run.
                        if let Err(e) = stream_request(base, &body, &started, &mut timing) {
                            eprintln!("request failed: {e:#}");
                        }
                        mine.push(timing);
                    }
                    mine
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect()
    });

    let report = BenchReport::of(&timings);
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report_json(&report, &args))?
        );
    } else {
        print_report(&report, &args);
    }
    if report.completed == 0 {
        anyhow::bail!("every request failed; nothing was measured");
    }
    Ok(())
}

/// The body every benchmark request sends.
///
/// `stream: true` is not a preference: TTFT is only observable on a
/// stream, and a buffered request can report nothing but end-to-end.
fn request_body(
    model: Option<&str>,
    shared_prefix: Option<&str>,
    cache_salt: Option<&str>,
    prompt: &str,
    sampling: &BenchSampling,
) -> Value {
    let mut messages = Vec::with_capacity(2);
    if let Some(shared) = shared_prefix {
        messages.push(json!({"role": "system", "content": shared}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    let mut body = json!({
        "model": model.unwrap_or("bench"),
        "messages": messages,
        "max_tokens": sampling.output_len,
        "temperature": sampling.temperature,
        "top_k": sampling.top_k,
        "ignore_eos": sampling.ignore_eos,
        "stream": true,
        // Without this the terminal frame carries no usage block, and
        // the prompt/cached pair the reuse figure needs is simply
        // absent -- which reads as "no prefix cache" rather than as
        // "nobody asked".
        "stream_options": {"include_usage": true},
    });
    if let Some(salt) = cache_salt {
        body["cache_salt"] = json!(salt);
    }
    body
}

/// Filler with no structure a template or a cache could shortcut,
/// DIFFERENT for every request in a run.
///
/// Deliberately varied rather than one repeated word: a prompt of
/// `"a a a a"` is the best case for any prefix cache and several
/// tokenizers, so a run built on one measures the cache rather than the
/// server.
///
/// `seed` is why this takes an argument. The first version varied
/// WITHIN a prompt and built the SAME prompt for every request, which
/// is the best case for a prefix cache ACROSS requests -- the very
/// thing the paragraph above says it avoids. Measured once the reuse
/// figure existed to show it: eight requests at concurrency four
/// reused 43.5% of their prompt tokens with no shared prefix asked
/// for, because the last four found the first four's pages. Every
/// throughput this command has ever printed was that much too
/// flattering, and nothing could see it until the server was asked
/// what it had reused.
fn filler_prompt(chars: usize, seed: usize) -> String {
    const WORDS: [&str; 12] = [
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
        "eiusmod",
        "tempor",
    ];
    let mut out = String::with_capacity(chars + 16);
    // The seed goes FIRST, so two prompts diverge at token one rather
    // than at the tail. A distinguishing suffix would leave the whole
    // head shareable and the run would still be measuring the cache.
    if chars > 0 {
        out.push_str(&format!("{seed} "));
    }
    let mut i = seed;
    while out.len() < chars {
        if !out.is_empty() {
            out.push(' ');
        }
        // A cheap deterministic walk: reproducible between runs, which
        // a benchmark needs, without pulling in an RNG.
        out.push_str(WORDS[(i * 7 + i * i) % WORDS.len()]);
        i += 1;
    }
    out.truncate(chars);
    out
}

/// Sends one streamed request and tics per token chunk.
fn stream_request(
    base: &str,
    body: &Value,
    started: &Instant,
    timing: &mut RequestTiming,
) -> Result<()> {
    let bytes = serde_json::to_vec(body)?;
    let (status, mut reader) =
        http::open("POST", &format!("{base}/v1/chat/completions"), Some(&bytes))?;
    if !(200..300).contains(&status) {
        let mut rest = String::new();
        use std::io::Read;
        reader.read_to_string(&mut rest)?;
        anyhow::bail!("HTTP {status}: {rest}");
    }

    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let trimmed = line.trim_end();
        if let Some(data) = trimmed.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                // The rule this whole command depends on: a keepalive
                // arrives during exactly the window TTFT measures, and
                // the terminal frame carries no token. See
                // `is_token_chunk` for what each would corrupt.
                if is_token_chunk(&chunk) {
                    timing.tic(started.elapsed().as_secs_f64());
                }
                // The terminal chunk carries the server's own token
                // count. Taken in preference to the chunk count,
                // because a buffered answer arrives as one chunk and
                // was still N tokens of work.
                if let Some(n) = chunk
                    .get("usage")
                    .and_then(|u| u.get("completion_tokens"))
                    .and_then(serde_json::Value::as_u64)
                {
                    timing.report_tokens(n as usize);
                }
                // The same terminal frame carries the prompt size and
                // the prefix hit. Read as a PAIR from one frame: a
                // fraction whose numerator and denominator came from
                // different requests is a plausible-looking number
                // with nothing behind it.
                if let Some(usage) = chunk.get("usage") {
                    let prompt = usage.get("prompt_tokens").and_then(Value::as_u64);
                    let cached = usage.get("cached_tokens").and_then(Value::as_u64);
                    if let (Some(p), Some(c)) = (prompt, cached) {
                        timing.report_prefix(p as usize, c as usize);
                    }
                }
            }
        }
        line.clear();
    }
    Ok(())
}

fn ms(seconds: Option<f64>) -> String {
    match seconds {
        Some(s) => format!("{:.1}", s * 1000.0),
        // A dash and not a zero: a run with no samples measured
        // nothing, and a zero reads as instantaneous.
        None => "-".to_string(),
    }
}

fn latency_json(l: &Latency) -> Value {
    json!({
        "mean_ms": l.mean.map(|v| v * 1000.0),
        "p50_ms": l.p50.map(|v| v * 1000.0),
        "p90_ms": l.p90.map(|v| v * 1000.0),
        "p99_ms": l.p99.map(|v| v * 1000.0),
    })
}

fn report_json(report: &BenchReport, args: &ServeBenchArgs) -> Value {
    json!({
        "requests": args.requests,
        "concurrency": args.concurrency,
        "output_len": args.output_len,
        "completed": report.completed,
        "failed": report.failed,
        "output_tokens": report.output_tokens,
        "duration_s": report.duration_s,
        "output_throughput_tps": report.output_throughput(),
        "ttft": latency_json(&report.ttft),
        "tpot": latency_json(&report.tpot),
        "end_to_end": latency_json(&report.end_to_end),
        "shared_prefix_chars": args.shared_prefix,
        "cache_salt": args.cache_salt,
        "prefix": report.prefix.map(prefix_json),
    })
}

fn prefix_json(p: PrefixTotals) -> Value {
    json!({
        "prompt_tokens": p.prompt_tokens,
        "cached_tokens": p.cached_tokens,
        "reuse_fraction": p.reuse_fraction(),
        "requests_with_reuse": p.requests_with_reuse,
        "requests_reporting": p.requests_reporting,
    })
}

/// The prefix-cache line, or the reason there is none.
///
/// Silence would be the wrong answer in every case: a run that asked
/// for a shared prefix and got no reuse is a finding, and a server with
/// no prefix cache is a different finding, and neither should look like
/// the other.
fn print_prefix(report: &BenchReport, args: &ServeBenchArgs) {
    let Some(p) = report.prefix else {
        if args.shared_prefix > 0 {
            println!(
                "prefix reuse: not reported by this server \
                 (no prefix cache configured, or usage omitted)"
            );
        }
        return;
    };
    let pct = p.reuse_fraction().map(|f| f * 100.0).unwrap_or(0.0);
    println!(
        "prefix reuse: {}/{} prompt tokens ({pct:.1}%), {} of {} requests reused something",
        p.cached_tokens, p.prompt_tokens, p.requests_with_reuse, p.requests_reporting
    );
}

fn print_report(report: &BenchReport, args: &ServeBenchArgs) {
    println!(
        "\nserve-bench  {} requests, concurrency {}, {} tokens each",
        args.requests, args.concurrency, args.output_len
    );
    println!(
        "completed {}  failed {}  tokens {}  span {:.2}s",
        report.completed, report.failed, report.output_tokens, report.duration_s
    );
    match report.output_throughput() {
        Some(tps) => println!("output throughput: {tps:.1} tok/s (whole run)"),
        None => println!("output throughput: - (nothing completed)"),
    }
    print_prefix(report, args);
    println!();
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>10}",
        "", "mean", "p50", "p90", "p99"
    );
    for (name, l) in [
        ("TTFT (ms)", &report.ttft),
        ("TPOT (ms)", &report.tpot),
        ("E2E (ms)", &report.end_to_end),
    ] {
        println!(
            "{:<12} {:>10} {:>10} {:>10} {:>10}",
            name,
            ms(l.mean),
            ms(l.p50),
            ms(l.p90),
            ms(l.p99)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every field the methodology depends on has to actually reach the
    /// wire. `ignore_eos` is the one worth a test of its own: without
    /// it the requests finish at different lengths and the slowest
    /// percentile is whichever prompt happened to run longest.
    #[test]
    fn a_bench_request_pins_everything_the_methodology_depends_on() {
        let body = request_body(Some("m"), None, None, "hello", &BenchSampling::new(64));
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["top_k"], 1);
        assert_eq!(body["ignore_eos"], true);
        assert_eq!(
            body["stream"], true,
            "TTFT is only observable on a stream; a buffered request can \
             report nothing but end-to-end"
        );
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(
            body["stream_options"]["include_usage"], true,
            "without it the terminal frame carries no usage block and the \
             prefix figures are absent rather than zero"
        );
        assert!(
            body.get("cache_salt").is_none(),
            "an unrequested salt must not be sent: it would put the run in \
             its own cache namespace and quietly change what it measures"
        );
    }

    /// The shared prefix is a SYSTEM message ahead of the per-request
    /// filler, and the salt rides beside it.
    #[test]
    fn a_shared_prefix_is_a_system_message_before_the_per_request_prompt() {
        let body = request_body(
            Some("m"),
            Some("shared"),
            Some("tenant-a"),
            "mine",
            &BenchSampling::new(8),
        );
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "shared");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "mine");
        assert_eq!(body["cache_salt"], "tenant-a");
    }

    /// Reproducible between runs, and not one repeated word: a prompt
    /// of `"a a a a"` is the best case for any prefix cache and several
    /// tokenizers, so a run built on one measures the cache.
    #[test]
    fn the_filler_prompt_is_reproducible_and_not_one_repeated_word() {
        let a = filler_prompt(200, 1);
        assert_eq!(a.len(), 200);
        assert_eq!(
            a,
            filler_prompt(200, 1),
            "the same run twice sends the same prompt"
        );

        let distinct: std::collections::HashSet<&str> = a.split_whitespace().collect();
        assert!(
            distinct.len() > 3,
            "a prompt of one repeated word measures the prefix cache: {a}"
        );

        assert!(filler_prompt(0, 1).is_empty());
    }

    /// **Two requests in a run must not share a prompt.**
    ///
    /// The defect this pins: every request sent the SAME filler, so the
    /// second half of a run found the first half's pages and the
    /// throughput figure included reuse the command was written to
    /// avoid. Measured at 43.5% of prompt tokens on eight requests at
    /// concurrency four.
    ///
    /// They must diverge at the FRONT: a distinguishing suffix leaves
    /// the whole head shareable and changes nothing.
    #[test]
    fn two_requests_in_a_run_do_not_share_a_prompt_prefix() {
        let a = filler_prompt(200, 1);
        let b = filler_prompt(200, 2);
        assert_ne!(a, b, "two requests sent the same prompt");

        let shared = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
        assert!(
            shared < 4,
            "two prompts share their first {shared} characters, so the \
             second request measures the first's pages"
        );
    }

    /// A missing figure prints as a dash, never as a zero: a zero is
    /// the best possible latency and would make a run in which nothing
    /// answered look like the fastest one.
    #[test]
    fn a_latency_that_was_never_measured_prints_as_a_dash() {
        assert_eq!(ms(None), "-");
        assert_eq!(ms(Some(0.0125)), "12.5");
    }
}
