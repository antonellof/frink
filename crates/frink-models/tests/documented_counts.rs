//! The numbers the prose states have to be the numbers the code holds.
//!
//! `README.md` said 47 audited architectures, `CLAUDE.md` said 98, and
//! `capability::AUDITED_GENERIC_GQA` held 104. Three numbers for one
//! fact, with nothing enforcing that they agree, which is this repo's
//! dominant bug shape pointed at its own documentation: every row that
//! closed since those sentences were written made them more wrong, and
//! a reader had no way to tell which one to believe.
//!
//! The fix is not to correct them once. It is this test, so the next
//! row that moves the table and forgets the prose turns something red
//! rather than shipping a fourth number.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<root>/crates/frink-models`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/frink-models has two ancestors")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let p = repo_root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
}

/// Every "<n> architectures run with" claim, with its file and line so
/// a failure says where to edit.
///
/// The number is the one IMMEDIATELY before the phrase, not the last
/// one on the line: `CLAUDE.md`'s sentence continues ", 4 more have
/// dedicated engines", and a scan that took the last number on the
/// line read that 4 as the audited count.
fn claimed_counts(doc_name: &str, text: &str) -> Vec<(String, usize, usize)> {
    const PHRASE: &str = "architectures run with";
    // Whitespace-normalised, because markdown wraps: README carries
    // "47 architectures run\n  with a logit comparison", and matching
    // the raw text found nothing there while reporting success. A
    // check that cannot fire is worse than no check.
    let flat: String = {
        let mut f = String::with_capacity(text.len());
        let mut ws = false;
        for c in text.chars() {
            if c.is_whitespace() {
                ws = true;
            } else {
                if ws && !f.is_empty() {
                    f.push(' ');
                }
                ws = false;
                f.push(c);
            }
        }
        f
    };
    let text = flat.as_str();
    let mut out = Vec::new();
    for (i, _) in text.match_indices(PHRASE) {
        let before = &text[..i];
        // Walk back over the separator between the number and the noun
        // (a space, a newline, and markdown bold markers).
        let head = before.trim_end_matches(|c: char| c.is_whitespace() || c == '*');
        let digits: String = head
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        let n: String = digits.chars().rev().collect();
        if let Ok(v) = n.parse::<usize>() {
            // Line numbers are lost by the flattening; report the
            // claim's position in words instead, which still points a
            // reader at the sentence.
            let words_in = before.split_whitespace().count();
            out.push((doc_name.to_string(), words_in, v));
        }
    }
    out
}

#[test]
fn the_docs_state_the_audited_architecture_count_the_catalog_holds() {
    let actual = frink_models::capability::AUDITED_GENERIC_GQA.len();
    let mut claims = Vec::new();
    for doc in ["README.md", "CLAUDE.md"] {
        claims.extend(claimed_counts(doc, &read(doc)));
    }
    assert!(
        !claims.is_empty(),
        "no doc states the audited count any more; if the sentence moved, \
         this test has to move with it rather than be deleted"
    );
    for (doc, line, claimed) in &claims {
        assert_eq!(
            *claimed, actual,
            "{doc} (about word {line}) says {claimed} audited \
             architectures, capability::AUDITED_GENERIC_GQA holds {actual}"
        );
    }
}

/// The manifest is generated from the same catalog (`frink archs
/// --write`), so a stale one is the same defect in a different file.
#[test]
fn the_generated_manifest_lists_every_audited_architecture() {
    let manifest = read("docs/manifests/architecture_manifest.md");
    let missing: Vec<_> = frink_models::capability::AUDITED_GENERIC_GQA
        .iter()
        .filter(|a| !manifest.contains(&format!("`{a}`")))
        .collect();
    assert!(
        missing.is_empty(),
        "docs/manifests/architecture_manifest.md is stale, \
         regenerate with `frink archs --write`; missing: {missing:?}"
    );
}

/// Every relative markdown link in the docs has to resolve.
///
/// Four did not: two plans pointed at `../../CONFIG.md` and
/// `../../benchmarks/RESULTS.md` from a depth that no longer existed
/// after the files moved. Nothing checked, so they rotted quietly and
/// a reader following one got a 404 on GitHub.
#[test]
fn every_relative_doc_link_resolves() {
    // This project's own docs only: the repo-root markdown files
    // (not recursively, or the walk reaches `ui/node_modules`, whose
    // vendored READMEs have link rot of their own and are nobody's
    // business here) plus everything under `docs/`.
    let root = repo_root();
    let mut docs: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_file() && p.extension().is_some_and(|x| x == "md") {
                docs.push(p);
            }
        }
    }
    collect_markdown(&root.join("docs"), &mut docs);
    assert!(docs.len() > 10, "found only {} markdown files", docs.len());

    let mut broken = Vec::new();
    for doc in &docs {
        let text = std::fs::read_to_string(doc).unwrap_or_default();
        for target in relative_link_targets(&text) {
            let resolved = doc.parent().expect("a file has a parent").join(&target);
            if !resolved.exists() {
                broken.push(format!("{}: {target}", doc.display()));
            }
        }
    }
    assert!(
        broken.is_empty(),
        "broken relative links:\n  {}",
        broken.join("\n  ")
    );
}

fn collect_markdown(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_markdown(&p, out);
        } else if p.extension().is_some_and(|x| x == "md") {
            out.push(p);
        }
    }
}

/// `](path.md)` and `](path.md#anchor)`, skipping URLs.
fn relative_link_targets(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == ']' && bytes[i + 1] == '(' {
            let mut j = i + 2;
            let mut target = String::new();
            while j < bytes.len() && bytes[j] != ')' && bytes[j] != '#' {
                target.push(bytes[j]);
                j += 1;
            }
            if target.ends_with(".md") && !target.contains("://") && !target.starts_with('/') {
                out.push(target);
            }
            i = j;
        }
        i += 1;
    }
    out
}

/// `docs/CONFIG.md` has a "Removed" section saying those switches are
/// gone and setting them does nothing. Both halves of that claim are
/// checkable: a documented LIVE variable must be read somewhere, and a
/// REMOVED one must not be.
///
/// It passes today (90 live, all read; 18 removed, none read). It is
/// here because the removed half is the one that rots: re-adding a
/// knob under an old name would make the documentation actively wrong
/// rather than merely stale.
#[test]
fn documented_environment_variables_match_what_the_code_reads() {
    let text = read("docs/CONFIG.md");
    let removed_start = text
        .find("## Removed")
        .expect("CONFIG.md has a Removed section");
    let removed_end = text[removed_start + 5..]
        .find("\n## ")
        .map(|o| removed_start + 5 + o)
        .unwrap_or(text.len());

    let live: Vec<String> = frink_env_names(&text[..removed_start])
        .into_iter()
        .chain(frink_env_names(&text[removed_end..]))
        .collect();
    let removed = frink_env_names(&text[removed_start..removed_end]);
    assert!(live.len() > 50, "only {} live vars parsed", live.len());
    assert!(!removed.is_empty(), "the Removed section parsed empty");

    let mut sources = String::new();
    let mut files = Vec::new();
    collect_rust(&repo_root().join("crates"), &mut files);
    for f in files {
        sources.push_str(&std::fs::read_to_string(f).unwrap_or_default());
    }

    let unread: Vec<&String> = live
        .iter()
        .filter(|v| !sources.contains(&format!("\"{v}\"")))
        .collect();
    assert!(unread.is_empty(), "documented but never read: {unread:?}");

    let resurrected: Vec<&String> = removed
        .iter()
        .filter(|v| sources.contains(&format!("\"{v}\"")))
        .collect();
    assert!(
        resurrected.is_empty(),
        "listed under Removed and still read: {resurrected:?}"
    );
}

fn frink_env_names(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in text.split('`') {
        if part.starts_with("FRINK_")
            && part
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            && !out.contains(&part.to_string())
        {
            out.push(part.to_string());
        }
    }
    out
}

fn collect_rust(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            collect_rust(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The unaudited-triage distribution `docs/MODELS.md` prints has to be
/// the one the catalog holds.
///
/// It was not. The table read `new code 1 / unknown 1` while the
/// catalog held three NEW CODE rows and one UNKNOWN, so the document
/// that exists to say how far frink is from llama.cpp on models
/// UNDERSTATED the gap by half -- the same defect this file was
/// written for, on the one table a reader is most likely to quote.
///
/// Counted from the catalog rather than from a second list, so a row
/// that closes moves the number here and the prose has to follow.
#[test]
fn docs_state_the_unaudited_triage_distribution_the_catalog_holds() {
    use frink_models::capability::{
        architecture_catalog, ArchPath, TriageClass, AUDITED_GENERIC_GQA,
    };

    let mut counts = [0usize; 4];
    for p in architecture_catalog() {
        if !matches!(p.path, ArchPath::GenericGqa { .. }) {
            continue;
        }
        if AUDITED_GENERIC_GQA.contains(&p.gguf_name) {
            continue;
        }
        let Some(t) = p.triage else {
            // `TRIAGE_PENDING`'s own test covers this case; an
            // untriaged row is not part of the distribution.
            continue;
        };
        counts[match t.class {
            TriageClass::FixtureAway => 0,
            TriageClass::OneMatchArm => 1,
            TriageClass::NewCode => 2,
            TriageClass::Unknown => 3,
        }] += 1;
    }

    let doc = read("docs/MODELS.md");
    for (label, want) in [
        ("fixture-away", counts[0]),
        ("one match arm", counts[1]),
        ("new code", counts[2]),
        ("unknown", counts[3]),
    ] {
        let row = format!("| {label} | {want} |");
        assert!(
            doc.contains(&row),
            "docs/MODELS.md must carry the row `{row}` for the catalog's {want} {label} \
             architecture(s); it does not. The distribution there is stale."
        );
    }

    // The same number, stated in prose elsewhere. `docs/ROADMAP.md`
    // ranked the work as "close the 41 unaudited architectures" long
    // after the count reached single digits, which is the same drift
    // in the document a reader uses to decide what to work on.
    let total: usize = counts.iter().sum();
    for (doc_name, doc_text) in [("docs/ROADMAP.md", read("docs/ROADMAP.md"))] {
        for (line_no, line) in doc_text.lines().enumerate() {
            if let Some(at) = line.find(" unaudited architectures") {
                let claimed: String = line[..at]
                    .chars()
                    .rev()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                assert_eq!(
                    claimed.parse::<usize>().ok(),
                    Some(total),
                    "{doc_name}:{} claims `{claimed} unaudited architectures`; the catalog \
                     holds {total}",
                    line_no + 1
                );
            }
        }
    }
    assert!(
        total > 0,
        "no unaudited generic architecture is triaged, so this test proved nothing"
    );
    // The prose states the total in two places beside the table, and
    // both went stale with it.
    assert!(
        doc.contains(&format!("None of the {total} is a fixture")),
        "docs/MODELS.md must say `None of the {total} is a fixture ...`"
    );
    assert!(
        doc.contains(&format!("All {total} have now been read")),
        "docs/MODELS.md must say `All {total} have now been read ...`"
    );
}
