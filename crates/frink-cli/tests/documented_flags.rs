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

/// Every `.rs` under `dir`, recursively.
fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
    out
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

/// Flags `docs/CLI.md` names on purpose and frink does NOT accept.
///
/// Every one is a llama.cpp flag the page mentions in order to say it
/// is absent, which is information a reader wants and a spelling they
/// must not copy. Listed by name so the reverse check below can be
/// exact rather than approximate: a blanket exemption for "anything
/// that looks like llama.cpp's" would have let `--sampler-seq`
/// through, which is the bug that produced this test.
const NAMED_AS_ABSENT: &[&str] = &[
    // `frink imatrix` has no way to combine earlier matrices.
    "in-file",
    // `frink perplexity` implements neither the stride mode nor the
    // multiple-choice tasks.
    "ppl-stride",
    // The libllama oracle binary `frink parity` shells out to has no
    // tokenize mode on older builds; the page says so and names the
    // flag.
    "tokenize",
    // A build profile, not a frink flag: `cargo build --release`.
    "release",
    // Written as `--split-` in prose about the `gguf-split` family.
    "split-",
];

/// **Every flag `docs/CLI.md` spells has to be one the CLI accepts.**
///
/// The other direction, and the one that had never been checked.
/// `--sampler-seq` was documented in TWO places as llama.cpp's other
/// spelling of `--samplers`, and clap rejected it with `unexpected
/// argument`: a reader copying the documented spelling got an error
/// from the page that told them to type it. It is an `alias` now.
///
/// A documented flag that does not exist is worse than an undocumented
/// one that does. The undocumented flag still works.
#[test]
fn every_documented_flag_is_one_the_cli_accepts() {
    // EVERY CLI and server source, not a hand-listed few: the first
    // version of this test named three files and reported thirty real
    // flags as missing because they are declared in `bench.rs`,
    // `chat.rs`, `quantize.rs` and the rest. A check whose input is a
    // list somebody has to remember to extend is the defect it is
    // looking for.
    let mut accepted: Vec<String> = Vec::new();
    let mut sources = 0usize;
    for dir in ["crates/frink-cli/src", "crates/frink-server/src"] {
        for entry in walk(&repo_root().join(dir)) {
            let text = std::fs::read_to_string(&entry).expect("read a source file");
            sources += 1;
            accepted.extend(long_flags(&text));
            // Clap derives a flag name from the FIELD when no
            // `long = "..."` is given, which is how `frink bench`
            // spells `--compare`, `--suite` and two dozen others. The
            // first version of this check read only the explicit
            // spellings and called all of them undocumented.
            accepted.extend(field_flags(&text));
            // `alias` and `visible_alias` are spellings a user may type
            // and clap accepts, so they count exactly as `long` does.
            // `alias`, `visible_alias` and the PLURAL forms are all
            // spellings clap accepts. Reading only the singular missed
            // `visible_aliases = ["gpu-layers", "ngl"]`, which is how
            // the server spells llama.cpp's `-ngl`.
            for key in [
                "alias = \"",
                "visible_alias = \"",
                "aliases = [",
                "visible_aliases = [",
            ] {
                for (i, _) in text.match_indices(key) {
                    let rest = &text[i + key.len()..];
                    let group = &rest[..rest
                        .find(']')
                        .unwrap_or(0)
                        .max(rest.find('"').map_or(0, |q| q + 1))];
                    for (j, _) in group.match_indices('"') {
                        let tail = &group[j + 1..];
                        if let Some(end) = tail.find('"') {
                            accepted.push(tail[..end].to_string());
                        }
                    }
                    if key.ends_with('"') {
                        if let Some(end) = rest.find('"') {
                            accepted.push(rest[..end].to_string());
                        }
                    }
                }
            }
        }
    }
    assert!(sources > 10, "only {sources} source files walked");
    assert!(
        accepted.len() > 50,
        "only {} flags parsed, so this check would pass vacuously",
        accepted.len()
    );

    let docs = read("docs/CLI.md");
    let mut unknown: Vec<String> = Vec::new();
    let mut rest = docs.as_str();
    // Only flags written in backticks: prose quotes llama.cpp's output
    // and its error strings, and a bare `--foo` inside a quoted error
    // is not a claim about frink's surface.
    while let Some(at) = rest.find("`--") {
        rest = &rest[at + 1..];
        let end = rest
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
            .unwrap_or(rest.len());
        let spelled = rest[2..end].to_string();
        if spelled.is_empty() {
            continue;
        }
        if accepted.contains(&spelled) || NAMED_AS_ABSENT.contains(&spelled.as_str()) {
            continue;
        }
        if !unknown.contains(&spelled) {
            unknown.push(spelled);
        }
    }
    assert!(
        unknown.is_empty(),
        "docs/CLI.md spells these flags and the CLI does not accept them: {unknown:?}. \
         Either add the spelling as a clap `alias`, or -- if the page names it to say frink \
         does NOT have it -- add it to NAMED_AS_ABSENT with the reason."
    );
}

/// Field names in a clap struct, kebab-cased: the flag clap derives
/// when the field carries no explicit `long`.
///
/// Over-accepting slightly is the right error here. A field that is
/// not a flag adds a name nobody documents, which costs nothing; a
/// flag this misses is reported as undocumented and sends a reader to
/// fix a page that was right.
fn field_flags(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in source.lines() {
        let t = line.trim();
        // `pub` is optional: a subcommand's inline fields are written
        // without it, which is how `frink bench` spells `--suite` and
        // `--render`.
        let rest = t.strip_prefix("pub ").unwrap_or(t);
        let Some(name) = rest.split(':').next() else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
            continue;
        }
        out.push(name.replace('_', "-"));
    }
    out
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
