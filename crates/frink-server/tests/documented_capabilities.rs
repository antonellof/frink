//! The capability table in `docs/FEATURES.md` has to match the code.
//!
//! That table is headed by a paragraph saying it is "checked against
//! the code rather than asserted", and for a long time nothing checked
//! it. Audited on 2026-09-23, two rows were wrong in opposite
//! directions: one said `RadixCache::evict` had no caller and the page
//! pool leaked, which had been fixed; and one said NVFP4 was not
//! parsed, when it is recognized, sized, and refused at execution --
//! which is coverage in this repo, not absence.
//!
//! Both are the dominant bug shape with a document on one side: two
//! structures that must agree about one thing, with nothing enforcing
//! it. These tests are the enforcement. They pin the STRUCTURAL claims
//! -- a symbol exists, a symbol has a caller, a symbol has none --
//! because those are the ones that go stale silently when somebody
//! wires or unwires a thing and does not read the prose.
//!
//! What they deliberately do NOT pin is the wording. A row may be
//! rewritten freely; what it may not do is contradict the code.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<root>/crates/frink-server`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/frink-server has two ancestors")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let p = repo_root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
}

/// Every `.rs` file under `crates/`, as (path, contents).
fn sources() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.filter_map(Result::ok) {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                if let Ok(t) = std::fs::read_to_string(&p) {
                    out.push((p, t));
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(&repo_root().join("crates"), &mut out);
    assert!(!out.is_empty(), "no sources found; the walk is broken");
    out
}

/// `text` with `//` line comments stripped.
///
/// A claim about callers must not be satisfied by a doc comment
/// DESCRIBING the caller -- which is exactly how the `evict` row
/// survived: the fix's own explanatory comment mentions the function
/// by name several times.
fn without_comments(text: &str) -> String {
    text.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The FEATURES row whose first cell contains `needle`.
fn capability_row(needle: &str) -> String {
    let features = read("docs/FEATURES.md");
    features
        .lines()
        .find(|l| l.starts_with('|') && l.contains(needle))
        .unwrap_or_else(|| panic!("no capability row mentioning {needle:?} in docs/FEATURES.md"))
        .to_string()
}

/// **The radix cache's eviction has a caller, and the row says so.**
///
/// The claim that went stale. `evict` reclaims pages from the prefix
/// tree when a request cannot find any; without a caller the pool
/// shrinks monotonically and a long-running server refuses requests
/// that fit. It is called now, and this fails if that is undone or if
/// the row goes back to claiming it is not.
#[test]
fn radix_eviction_is_called_and_the_table_agrees() {
    let callers: Vec<String> = sources()
        .into_iter()
        .filter(|(p, _)| !p.ends_with("documented_capabilities.rs"))
        // Its own module and the tests beside it do not count: the
        // claim is that PRODUCTION reclaims pages.
        .filter(|(p, _)| !p.to_string_lossy().contains("policy/radix/"))
        // The RADIX tree's evict, not any evict. `.evict(` alone was
        // satisfied by `weight_matrix/repack_cache.rs`, an unrelated
        // LRU with a method of the same name, so the first version of
        // this test stayed green with the real call site deleted.
        // Found by sabotage, which is the only way that shows.
        .filter(|(_, t)| {
            let code = without_comments(t);
            code.contains("radix") && code.contains("tree.evict(")
        })
        .map(|(p, _)| p.to_string_lossy().into_owned())
        .collect();

    assert!(
        !callers.is_empty(),
        "nothing outside policy/radix calls evict: the page pool leaks, \
         and docs/FEATURES.md must say so again"
    );

    let row = capability_row("Agentic context edits");
    assert!(
        !row.contains("has no caller"),
        "eviction has a caller ({callers:?}) and the row still says it has none: {row}"
    );
}

/// **A format that is recognized and refused is not a format that is
/// absent**, and the row has to tell them apart.
///
/// This repo treats a named refusal as coverage: a file carrying NVFP4
/// stops with its type named instead of computing something else. A
/// row calling that "not parsed" understates what the engine does and
/// would send a reader looking for a parser that exists.
#[test]
fn nvfp4_is_recognized_and_refused_rather_than_absent() {
    let gguf = read("crates/frink-gguf/src/lib.rs");
    assert!(
        gguf.contains("NVFP4"),
        "frink-gguf no longer knows NVFP4; the row must change with it"
    );
    assert!(
        gguf.contains("(40, GgmlType::NVFP4)"),
        "NVFP4 is no longer mapped from its ggml tag, so a file carrying \
         it would not be recognized at all"
    );

    let row = capability_row("NVFP4");
    assert!(
        !row.contains("Neither is parsed"),
        "NVFP4 is parsed and sized, and refused at execution: {row}"
    );
}

/// **`ExecutionPlan` is still read by nothing**, which is what its row
/// claims.
///
/// Pinned in the other direction from the two above: this one fails
/// when somebody WIRES it, which is the moment the row becomes wrong
/// and the moment nobody thinks to reread a capability table.
#[test]
fn the_execution_plan_row_is_still_true() {
    let readers: Vec<String> = sources()
        .into_iter()
        .filter(|(p, _)| {
            let s = p.to_string_lossy();
            !s.ends_with("documented_capabilities.rs") && !s.contains("execution_plan.rs")
        })
        .filter(|(_, t)| without_comments(t).contains("execution_plan."))
        .map(|(p, _)| p.to_string_lossy().into_owned())
        .collect();

    let row = capability_row("Graph-compatible execution");
    if readers.is_empty() {
        assert!(
            row.contains("read by nothing"),
            "nothing reads the plan and the row no longer says so: {row}"
        );
    } else {
        panic!(
            "{readers:?} now read the execution plan; \
             update the 'Graph-compatible execution' row, which still says \
             it is read by nothing"
        );
    }
}
