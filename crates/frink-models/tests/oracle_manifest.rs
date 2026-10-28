//! The oracle's manifests agree with the tree they describe.
//!
//! `scripts/oracle.sh` (ADR 0011) regenerates every fixture and every
//! llama.cpp golden from libllama at the pin and fails CI when one
//! moves. It can only regenerate what `tools/oracle/` lists, so a golden
//! or fixture added without a row would be checked by nothing while the
//! job stayed green: a gate that cannot fire. These tests make the
//! omission a red test instead, with no llama.cpp needed.
//!
//! The pin is stated once, in `tools/llama_cpp_pin`. The two places
//! that name it in prose are held to that file here.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tests_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The first word of every non-comment line.
fn manifest_keys(name: &str) -> Vec<String> {
    read(&repo().join("tools/oracle").join(name))
        .lines()
        .map(|l| l.split('#').next().unwrap_or_default().trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.split_whitespace().next().unwrap().to_string())
        .collect()
}

fn pin() -> String {
    let pin = read(&repo().join("tools/llama_cpp_pin")).trim().to_string();
    assert!(
        pin.len() == 40 && pin.bytes().all(|b| b.is_ascii_hexdigit()),
        "tools/llama_cpp_pin must be one full commit SHA, got {pin:?}"
    );
    pin
}

#[test]
fn the_pin_is_stated_once_and_the_prose_agrees_with_it() {
    let pin = pin();
    let short = &pin[..8];
    // The CURRENT audit is the newest one by its dated name.
    let audit = std::fs::read_dir(repo().join("docs/plans"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("parity-audit-") && n.ends_with(".md"))
        .max()
        .expect("a parity audit under docs/plans");
    let title = read(&repo().join("docs/plans").join(&audit));
    let title = title.lines().next().unwrap_or_default();
    assert!(
        title.contains(short),
        "docs/plans/{audit} is the current audit and its title names a different pin than \
         tools/llama_cpp_pin ({short}): {title}"
    );
    let names = read(&tests_dir().join("llama_cpp_arch_names.txt"));
    let source = names
        .lines()
        .find(|l| l.starts_with("# Source:"))
        .expect("a `# Source:` line");
    assert!(
        source.contains(&format!("at {short}")),
        "llama_cpp_arch_names.txt was derived at a different commit than tools/llama_cpp_pin \
         ({short}): {source}"
    );
}

/// `file.rs:CONST` for every `const *GOLDEN*: [f32; N]` under tests/.
fn golden_consts() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for dir in [tests_dir(), tests_dir().join("common")] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "rs") || path.ends_with("oracle_manifest.rs") {
                continue;
            }
            let rel = path
                .strip_prefix(tests_dir())
                .unwrap()
                .to_string_lossy()
                .into_owned();
            for line in read(&path).lines() {
                let line = line.trim_start();
                let line = line.strip_prefix("pub ").unwrap_or(line);
                let Some(rest) = line.strip_prefix("const ") else {
                    continue;
                };
                let Some((name, ty)) = rest.split_once(':') else {
                    continue;
                };
                if name.contains("GOLDEN") && ty.trim_start().starts_with("[f32;") {
                    out.insert(format!("{rel}:{name}"));
                }
            }
        }
    }
    out
}

fn decode_goldens() -> BTreeSet<String> {
    std::fs::read_dir(tests_dir().join("fixtures/decode_steps"))
        .unwrap()
        .map(|e| {
            format!(
                "fixtures/decode_steps/{}",
                e.unwrap().file_name().to_string_lossy()
            )
        })
        .collect()
}

#[test]
fn every_golden_has_an_oracle_row_and_every_row_a_golden() {
    let mut tree = golden_consts();
    tree.extend(decode_goldens());
    assert!(
        tree.len() > 100,
        "the scan found {} goldens: it has stopped seeing them",
        tree.len()
    );
    let listed: BTreeSet<String> = manifest_keys("goldens.tsv").into_iter().collect();
    let unlisted: Vec<_> = tree.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&tree).collect();
    assert!(
        unlisted.is_empty(),
        "goldens no row in tools/oracle/goldens.tsv regenerates, so CI never checks them \
         against libllama: {unlisted:?}"
    );
    assert!(
        stale.is_empty(),
        "tools/oracle/goldens.tsv rows naming no golden: {stale:?}"
    );
}

#[test]
fn every_fixture_has_an_oracle_row_and_every_row_a_fixture() {
    let tree: BTreeSet<String> = std::fs::read_dir(tests_dir().join("fixtures"))
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().to_string_lossy().into_owned();
            n.strip_suffix(".gguf").map(str::to_string)
        })
        .collect();
    let listed: BTreeSet<String> = manifest_keys("fixtures.tsv").into_iter().collect();
    let unlisted: Vec<_> = tree.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&tree).collect();
    assert!(
        unlisted.is_empty(),
        "fixtures with no row in tools/oracle/fixtures.tsv, so CI neither regenerates them nor \
         runs `frink parity` on them: {unlisted:?}"
    );
    assert!(
        stale.is_empty(),
        "tools/oracle/fixtures.tsv rows naming no fixture: {stale:?}"
    );
}
