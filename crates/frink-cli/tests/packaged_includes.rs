//! Every `include_str!` / `include_bytes!` in the workspace names a
//! file inside its own crate.
//!
//! `cargo publish` builds each crate from its packaged tarball, which
//! holds the crate directory and nothing above it. An include that
//! climbs out of the crate compiles in the workspace, so every CI job
//! stays green, and fails only at publish: v0.50.0 published eleven
//! crates and stopped at `frink-cli`, whose `frink licenses` read
//! `../../../docs/THIRD_PARTY_NOTICES.md`. The crate holds a symlink
//! to that file now, which `cargo package` stores as the file's
//! contents, so the notices still have one source.

use std::path::{Component, Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/frink-cli sits two levels below the repo root")
        .to_path_buf()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The string literal of every `include_str!(` / `include_bytes!(`,
/// whitespace and line breaks allowed before it, since `cargo fmt`
/// wraps long arguments onto the next line.
fn include_literals(src: &str) -> Vec<String> {
    let mut found = Vec::new();
    for mac in ["include_str!(", "include_bytes!("] {
        let mut rest = src;
        while let Some(i) = rest.find(mac) {
            rest = &rest[i + mac.len()..];
            let arg = rest.trim_start();
            if let Some(lit) = arg.strip_prefix('"') {
                if let Some(end) = lit.find('"') {
                    found.push(lit[..end].to_string());
                }
            }
        }
    }
    found
}

/// Resolves `rel` against `base` without touching the filesystem, so a
/// symlink inside the crate counts as inside: that is what `cargo
/// package` sees too.
fn lexical_join(base: &Path, rel: &str) -> PathBuf {
    let mut out = base.to_path_buf();
    for comp in Path::new(rel).components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

#[test]
fn every_include_stays_inside_its_crate() {
    let crates = repo_root().join("crates");
    let mut checked = 0;
    let mut escaping = Vec::new();
    for entry in std::fs::read_dir(&crates)
        .expect("crates/ exists")
        .flatten()
    {
        let crate_dir = entry.path();
        if !crate_dir.join("Cargo.toml").is_file() {
            continue;
        }
        let mut files = Vec::new();
        for sub in ["src", "tests", "benches", "examples"] {
            rust_files(&crate_dir.join(sub), &mut files);
        }
        for file in files {
            let src = std::fs::read_to_string(&file).unwrap();
            for lit in include_literals(&src) {
                checked += 1;
                let target = lexical_join(file.parent().unwrap(), &lit);
                if !target.starts_with(&crate_dir) {
                    escaping.push(format!(
                        "{}: include of {lit:?} leaves {}",
                        file.strip_prefix(repo_root()).unwrap().display(),
                        crate_dir.file_name().unwrap().to_string_lossy(),
                    ));
                }
            }
        }
    }
    // A scan that finds nothing passes for the wrong reason.
    assert!(checked > 0, "no include_str!/include_bytes! found at all");
    assert!(
        escaping.is_empty(),
        "these will not build from the published crate:\n{}",
        escaping.join("\n")
    );
}

#[test]
fn a_wrapped_include_is_still_read() {
    let src = "x(include_str!(\n        \"../../../docs/A.md\"\n    ))";
    assert_eq!(include_literals(src), ["../../../docs/A.md"]);
}
