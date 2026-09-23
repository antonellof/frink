//! The speed claims in the prose have to be the numbers the ledger
//! holds, or say plainly that they are not.
//!
//! The defect this was written against: `docs/FEATURES.md` said CUDA
//! prefill was "about 4x" off while `benchmarks/RESULTS.md` published
//! 22x to 43x for the same backend, and the rendered ledger carried no
//! version, so neither document could tell a reader that the two
//! describe different builds. Both numbers were true when taken. The
//! pair was not, and nothing compared them.
//!
//! Two structures that must agree about one thing, with nothing
//! enforcing it, pointed at the benchmark ledger this time.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<root>/crates/frink-cli`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/frink-cli has two ancestors")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let p = repo_root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
}

/// `text` with every run of whitespace collapsed to one space, so a
/// claim wrapped across two markdown source lines is still findable.
fn flatten(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every gap multiple the ledger publishes, as printed (`1.04`).
fn ledger_gaps(results: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in results.lines() {
        if !line.starts_with('|') {
            continue;
        }
        // Gap cells are the only `**N.NN×**` in the table.
        let mut rest = line;
        while let Some(i) = rest.find("**") {
            rest = &rest[i + 2..];
            let Some(j) = rest.find("×**") else { break };
            let cell = &rest[..j];
            if cell.chars().all(|c| c.is_ascii_digit() || c == '.') && !cell.is_empty() {
                out.push(cell.to_string());
            }
            rest = &rest[j + 3..];
        }
    }
    out
}

/// **Every backend the ledger marks stale must be called stale in the
/// prose.**
///
/// The ledger's staleness marker is derived from the receipts; this
/// says the documents that summarise it cannot quietly present an old
/// number as what the engine does now. `FEATURES.md` is the file a
/// reader reaches for, so it is the one held to it.
#[test]
fn a_stale_ledger_is_admitted_in_the_prose() {
    let results = read("benchmarks/RESULTS.md");
    let features = flatten(&read("docs/FEATURES.md"));

    if !results.contains("⚠️") {
        // Nothing stale: the claim this test exists for cannot be made
        // wrongly, and asserting the word "stale" is absent would pin
        // prose nobody has to write.
        return;
    }
    assert!(
        features.contains("stale"),
        "the ledger marks rows stale and docs/FEATURES.md does not say so anywhere"
    );
}

/// **A gap figure in the prose is either in the ledger or dated.**
///
/// Checked for CUDA, the backend the two disagreed on. The 4.2x is not
/// a receipt and is allowed to stand only because the sentence carrying
/// it says which build and which run produced it; the ledger's own
/// figures must still be reproducible from the table.
#[test]
fn the_ledgers_cuda_range_is_the_one_the_prose_quotes() {
    let results = read("benchmarks/RESULTS.md");
    let features = read("docs/FEATURES.md");
    let gaps = ledger_gaps(&results);
    assert!(!gaps.is_empty(), "no gap cells parsed out of the ledger");

    // A markdown table row is ONE source line, so this reads lines
    // rather than the flattened text: flattening first made an earlier
    // paragraph mentioning the same card look like the cell, and the
    // test failed against correct prose.
    let row = features
        .lines()
        .find(|l| l.starts_with('|') && l.contains("NVIDIA RTX"))
        .unwrap_or_else(|| panic!("no NVIDIA row in docs/FEATURES.md"))
        .to_string();
    let row = row.as_str();

    for quoted in ["22x", "43x"] {
        assert!(
            row.contains(quoted),
            "the NVIDIA row no longer quotes {quoted}; if the ledger moved, \
             re-derive the sentence from it rather than dropping the range"
        );
        let bare = quoted.trim_end_matches('x');
        assert!(
            gaps.iter().any(|g| g.starts_with(bare)),
            "docs/FEATURES.md quotes {quoted} for NVIDIA and no ledger gap \
             starts with {bare}: {gaps:?}"
        );
    }

    // The unreceipted figure has to carry its provenance, or it reads
    // as a ledger number that is simply missing from the ledger.
    assert!(
        row.contains("4.2x") && row.contains("1932") && row.contains("8,200"),
        "the post-fix 4.2x figure must keep the two throughputs it was \
         divided from, since no receipt backs it: {row}"
    );
}

/// The ledger's own summary has to name the build that produced each
/// group, so a gap column cannot hide its age.
#[test]
fn every_ledger_section_names_the_build_it_was_measured_on() {
    let results = read("benchmarks/RESULTS.md");
    let after = results
        .split("### At a glance")
        .nth(1)
        .unwrap_or_else(|| panic!("no summary table in benchmarks/RESULTS.md"));
    // Stop at the first per-machine heading: past it the detail tables
    // have their own `|` rows, which carry no version and never should.
    let summary = after.split("\n###").next().unwrap_or(after);
    assert!(
        summary.contains("Measured at"),
        "the summary table has no column naming the build each row came from"
    );
    for line in summary.lines().filter(|l| l.starts_with('|')) {
        if line.contains("---") || line.contains("Measured at") {
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        assert!(
            line.contains('`'),
            "a summary row carries no version: {line}"
        );
    }
}
