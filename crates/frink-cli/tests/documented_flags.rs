//! Every CLI flag the binaries accept has to appear in the docs.
//!
//! `--model-draft` shipped with a real generation path, five distinct
//! refusals, and a dozen references from `docs/MODELS.md` saying which
//! models it will not serve -- while `docs/CLI.md` never named it. The
//! only speculative line there described `frink speculative`, a demo
//! with no draft model, so a reader would reasonably conclude the demo
//! was the whole feature.
//!
//! `--draft-max` and `--draft-p-min` were missing for the same reason:
//! nothing compared the flags that exist to the flags that are
//! written down.
//!
//! This is the comparison: a flag's long name has to occur in the
//! prose, followed by something that is not more flag. The boundary is
//! not fussiness. A plain substring search passes `--draft` on the
//! strength of `--draft-max` appearing, and it passed a deliberate
//! sabotage that mangled a documented flag into
//! `--draft-p-minSABOTAGE`, because the original is still a substring
//! of the mangled one. A check that cannot fail the sabotage aimed at
//! it is not a check.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
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

/// Every `long = "..."` in a clap derive, which is the flag as a user
/// types it.
fn long_flags(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (i, _) in source.match_indices("long = \"") {
        let rest = &source[i + "long = \"".len()..];
        if let Some(end) = rest.find('"') {
            let f = rest[..end].to_string();
            if !out.contains(&f) {
                out.push(f);
            }
        }
    }
    out
}

#[test]
fn every_run_flag_is_documented() {
    let flags = long_flags(&read("crates/frink-cli/src/run.rs"));
    assert!(flags.len() > 30, "only {} flags parsed", flags.len());
    let docs = read("docs/CLI.md");
    let missing: Vec<&String> = flags.iter().filter(|f| !documents(&docs, f)).collect();
    assert!(
        missing.is_empty(),
        "these `frink run` flags are accepted and undocumented in docs/CLI.md: {missing:?}"
    );
}

#[test]
fn every_server_flag_is_documented() {
    let flags = long_flags(&read("crates/frink-server/src/cli.rs"));
    assert!(flags.len() > 20, "only {} flags parsed", flags.len());
    // The server's surface is split across the CLI, API and config
    // pages; which page is a judgement call, being written down at all
    // is not.
    let docs = ["docs/CLI.md", "docs/API.md", "docs/CONFIG.md", "README.md"]
        .iter()
        .map(|p| read(p))
        .collect::<Vec<_>>()
        .join("\n");
    let missing: Vec<&String> = flags.iter().filter(|f| !documents(&docs, f)).collect();
    assert!(
        missing.is_empty(),
        "these `frink-server` flags are accepted and undocumented: {missing:?}"
    );
}

/// Whether `docs` names `--flag` as a whole flag.
///
/// The character after it must not continue the flag, or `--draft`
/// would be satisfied by `--draft-max` and a mangled entry would still
/// match the name it mangled.
fn documents(docs: &str, flag: &str) -> bool {
    let needle = format!("--{flag}");
    docs.match_indices(&needle).any(|(i, _)| {
        docs[i + needle.len()..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
    })
}

#[cfg(test)]
mod tests {
    use super::documents;

    /// The boundary rule, in both directions.
    #[test]
    fn a_longer_flag_does_not_document_a_shorter_one() {
        assert!(documents("| `--draft-max N` | ...", "draft-max"));
        assert!(
            !documents("| `--draft-max N` | ...", "draft"),
            "--draft-max must not stand in for --draft"
        );
        assert!(
            !documents("`--draft-p-minSABOTAGE`", "draft-p-min"),
            "a mangled entry must not match the name it mangled"
        );
        // The ordinary case: a flag followed by punctuation or a space.
        assert!(documents("`--ctk` selects", "ctk"));
        assert!(documents("use --jinja.", "jinja"));
    }
}
