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
