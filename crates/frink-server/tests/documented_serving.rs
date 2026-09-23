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
