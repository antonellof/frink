//! The serving audit cites file and line. Those citations have to be
//! real.
//!
//! `docs/plans/serving-parity-audit.md` says what continuous batching,
//! paged KV, prefix caching and cache-aware admission actually do, and
//! points at the line that decides each one. A citation is the most
//! perishable thing a document can carry: the code moves and the
//! number stays, and a reader who follows it lands somewhere
//! plausible and wrong.
//!
//! Two audits before this one shipped prose that contradicted the
//! code -- a leak described as live after it was fixed, a named
//! refusal called an absence. Both were caught by reading, once,
//! months late. This is the same claim class with a test under it.
//!
//! What it checks is that each cited line still contains the SYMBOL
//! the audit says is there. Not the wording of the audit, which is
//! free to change, and not the exact line of an uncited fact.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/frink-server has two ancestors")
        .to_path_buf()
}

fn line_at(rel: &str, one_based: usize) -> String {
    let p = repo_root().join(rel);
    let text =
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()));
    text.lines()
        .nth(one_based - 1)
        .unwrap_or_else(|| panic!("{rel} has no line {one_based}"))
        .to_string()
}

/// Every `file:line` the audit cites, with the symbol that must still
/// be on that line.
///
/// The symbol is what makes this a test rather than a line count: an
/// edit above the citation moves it and this fails, which is the
/// signal to re-read the audit, not to bump a number.
const CITATIONS: &[(&str, usize, &str)] = &[
    // Continuous batching.
    (
        "crates/frink-server/src/serving/batch/worker.rs",
        61,
        "fn admit",
    ),
    // Paged KV.
    ("crates/frink-core/src/cache.rs", 57, "fn new"),
    ("crates/frink-server/src/generate.rs", 415, "fn fork"),
    (
        "crates/frink-server/src/serving/batch/prefill.rs",
        144,
        "fn step_chunk",
    ),
    // Prefix caching.
    (
        "crates/frink-server/src/policy/radix/plain.rs",
        176,
        "fn peek_cached_len",
    ),
    (
        "crates/frink-server/src/policy/radix/plain.rs",
        201,
        "fn match_prefix",
    ),
    (
        "crates/frink-server/src/policy/radix/plain.rs",
        211,
        "fn insert_prefix",
    ),
    (
        "crates/frink-server/src/policy/radix/plain.rs",
        245,
        "fn lock",
    ),
    (
        "crates/frink-server/src/policy/radix/plain.rs",
        259,
        "fn unlock",
    ),
    (
        "crates/frink-server/src/policy/radix/plain.rs",
        287,
        "fn evict",
    ),
    ("crates/frink-server/src/generate.rs", 857, "tree.evict("),
];

/// **Every cited line still holds what the audit says it holds.**
#[test]
fn the_serving_audits_citations_still_point_at_their_subject() {
    let audit = std::fs::read_to_string(repo_root().join("docs/plans/serving-parity-audit.md"))
        .expect("the serving audit");

    for (file, line, symbol) in CITATIONS {
        let found = line_at(file, *line);
        assert!(
            found.contains(symbol),
            "{file}:{line} no longer contains {symbol:?} -- it is:\n  {found}\n\
             Re-read docs/plans/serving-parity-audit.md against the code \
             rather than only bumping the number."
        );

        // And the audit has to actually cite it, or this table is
        // pinning lines nobody claims -- coverage of nothing.
        let basename = file.rsplit('/').next().expect("a basename");
        assert!(
            audit.contains(&format!("{basename}:{line}"))
                || audit.contains(&format!("`{basename}`")),
            "{file}:{line} is pinned here but the audit cites neither it \
             nor {basename}"
        );
    }
}

/// **The audit's two bounds are the constants the code holds.**
///
/// The anti-starvation argument is only worth as much as its numbers.
/// If `WINDOW` or `MAX_SKIPS` moves and the prose does not, the
/// document is making a safety claim about a policy that no longer has
/// those bounds.
#[test]
fn the_audits_admission_bounds_are_the_ones_the_code_uses() {
    let src = std::fs::read_to_string(
        repo_root().join("crates/frink-server/src/serving/batch/cache_aware.rs"),
    )
    .expect("cache_aware.rs");
    let audit = std::fs::read_to_string(repo_root().join("docs/plans/serving-parity-audit.md"))
        .expect("the serving audit");

    for (name, quoted) in [("WINDOW", "WINDOW = 8"), ("MAX_SKIPS", "MAX_SKIPS = 4")] {
        let decl = src
            .lines()
            .find(|l| l.contains(&format!("const {name}")))
            .unwrap_or_else(|| panic!("no `const {name}` in cache_aware.rs"));
        let value = decl
            .rsplit('=')
            .next()
            .and_then(|v| v.trim().trim_end_matches(';').parse::<u32>().ok())
            .unwrap_or_else(|| panic!("cannot read {name}'s value from: {decl}"));
        let expected = quoted
            .rsplit(' ')
            .next()
            .and_then(|v| v.parse::<u32>().ok())
            .expect("the quoted value");
        assert_eq!(
            value, expected,
            "the audit quotes `{quoted}` and the code holds {name} = {value}"
        );
        assert!(
            audit.contains(quoted),
            "the audit no longer quotes `{quoted}`; the starvation bound it \
             argues from must stay visible"
        );
    }
}

/// **The one modelled number is labelled as modelled.**
///
/// Every other figure in that document came off a socket. The
/// cache-aware saving did not: it is a unit test with an eviction
/// model in it. A reader has to be able to tell, and the sentence that
/// tells them is load-bearing.
#[test]
fn the_modelled_figure_is_still_marked_as_modelled() {
    let audit = std::fs::read_to_string(repo_root().join("docs/plans/serving-parity-audit.md"))
        .expect("the serving audit");

    assert!(
        audit.contains("3000 prefill\ntokens against 1200")
            || audit.contains("3000 prefill tokens against 1200"),
        "the modelled figure is gone; if it moved, its label has to move with it"
    );
    assert!(
        audit.contains("not a server reading"),
        "the modelled figure has lost the sentence saying it is modelled, \
         which is the only thing separating it from the measured ones"
    );
}

/// The receipt the serving charts and prose are both drawn from.
fn serving_receipt() -> serde_json::Value {
    let p = repo_root().join("benchmarks/receipts/serving/serving_features_m2pro_0.49.0.json");
    let text =
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()));
    serde_json::from_str(&text).expect("the serving receipt is JSON")
}

/// A document with every run of whitespace collapsed to one space.
///
/// Markdown wraps. `736 of 757` is really `736 of\n  757` in the
/// source, and a raw `contains` reports it missing -- which is exactly
/// what happened here, and is the defect `documented_counts.rs`
/// already carries a `flatten` for. A test that fails on line breaks
/// teaches people to reflow prose to satisfy it.
fn doc(rel: &str) -> String {
    let text = std::fs::read_to_string(repo_root().join(rel))
        .unwrap_or_else(|e| panic!("reading {rel}: {e}"));
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// **The prose quotes the numbers the receipt holds.**
///
/// Three documents quote these figures -- the serving audit, the
/// benchmark ledger and the README -- and the charts are generated
/// from a fourth place. Typing a number into any of them by hand is
/// four structures that must agree with nothing enforcing it, which
/// is this repo's dominant bug shape with a chart attached. The
/// receipt is the one source; this says the prose still matches it.
#[test]
fn the_serving_prose_quotes_the_measured_receipt() {
    let r = serving_receipt();
    let audit = doc("docs/plans/serving-parity-audit.md");
    let ledger = doc("benchmarks/RESULTS.md");
    let readme = doc("README.md");

    let points = r["concurrency_scaling"]["points"]
        .as_array()
        .expect("scaling points");
    let lo = points[0]["tok_s"].as_f64().expect("first");
    let hi = points[points.len() - 1]["tok_s"].as_f64().expect("last");
    assert!(
        audit.contains(&format!("{lo:.1}")) && audit.contains(&format!("{hi:.1}")),
        "the audit no longer quotes the scaling endpoints {lo:.1} and {hi:.1}"
    );

    let off = r["continuous_batching_ab"]["off"]["tok_s"]
        .as_f64()
        .expect("off");
    let on = r["continuous_batching_ab"]["on"]["tok_s"]
        .as_f64()
        .expect("on");
    assert!(
        audit.contains(&format!("{off:.1}")) && audit.contains(&format!("{on:.1}")),
        "the audit no longer quotes the A/B pair {off:.1} / {on:.1}"
    );
    // The README quotes the RATIO rather than the two throughputs,
    // because a landing page states the result and not the method.
    let ratio = on / off;
    assert!(
        readme.contains(&format!("{ratio:.2}x")),
        "README quotes a batching speedup that is not {ratio:.2}x"
    );

    let prompt = r["prefix_reuse"]["prompt_tokens"].as_u64().expect("prompt");
    let cached = r["prefix_reuse"]["warm"]["cached_tokens"]
        .as_u64()
        .expect("cached");
    let reuse = format!("{cached} of {prompt}");
    for (name, text) in [
        ("the audit", &audit),
        ("the ledger", &ledger),
        ("README", &readme),
    ] {
        assert!(
            text.contains(&reuse) || text.contains(&format!("{cached}/{prompt}")),
            "{name} no longer quotes the measured reuse `{reuse}`"
        );
    }

    for (name, text) in [("the audit", &audit), ("the ledger", &ledger)] {
        let cold = r["prefix_reuse"]["cold"]["latency_ms"]
            .as_u64()
            .expect("cold");
        let warm = r["prefix_reuse"]["warm"]["latency_ms"]
            .as_u64()
            .expect("warm");
        assert!(
            text.contains(&format!("{cold} ms")) && text.contains(&format!("{warm} ms")),
            "{name} no longer quotes the cold/warm latencies {cold} ms / {warm} ms"
        );
    }
}

/// **The committed charts are the ones the generator produces from the
/// receipt.**
///
/// A committed generated file drifts the moment somebody edits the
/// receipt and does not re-run the script -- the same defect the
/// architecture manifest had, where two published rows described a
/// path the catalog had moved off.
///
/// Checked by looking for the receipt's own values inside the SVG
/// text rather than by re-running Python, which a Rust test should not
/// need.
#[test]
fn the_committed_charts_carry_the_receipts_numbers() {
    let r = serving_receipt();

    let scaling = doc("docs/assets/serving-scaling-light.svg");
    for p in r["concurrency_scaling"]["points"]
        .as_array()
        .expect("points")
    {
        let v = p["tok_s"].as_f64().expect("tok_s");
        assert!(
            scaling.contains(&format!("{v:.1}")),
            "the scaling chart does not plot {v:.1}; regenerate with \
             `python3 scripts/make_serving_charts.py`"
        );
    }

    let batching = doc("docs/assets/serving-batching-light.svg");
    for arm in ["off", "on"] {
        let v = r["continuous_batching_ab"][arm]["tok_s"]
            .as_f64()
            .expect("arm");
        assert!(
            batching.contains(&format!("{v}")),
            "the batching chart does not carry the {arm} arm's {v}; regenerate it"
        );
    }

    let prefix = doc("docs/assets/serving-prefix-light.svg");
    for arm in ["cold", "warm", "isolation"] {
        let v = r["prefix_reuse"][arm]["latency_ms"].as_u64().expect("arm");
        assert!(
            prefix.contains(&format!("{v} ms")),
            "the prefix chart does not carry the {arm} arm's {v} ms; regenerate it"
        );
    }

    // Both themes ship, because GitHub strips media queries out of an
    // embedded SVG and the <picture> element needs a file per scheme.
    for name in ["scaling", "batching", "prefix"] {
        for theme in ["light", "dark"] {
            let p = repo_root().join(format!("docs/assets/serving-{name}-{theme}.svg"));
            assert!(p.is_file(), "missing chart {}", p.display());
        }
    }
}
