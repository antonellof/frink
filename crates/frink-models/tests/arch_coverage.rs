//! Every architecture llama.cpp can name, this catalog has an answer
//! for.
//!
//! "The catalog covers all of llama.cpp's architecture names" was a
//! claim made by reading the two lists side by side once, by hand.
//! Nothing held it afterwards, and the failure mode is silent in the
//! worst way: a name llama.cpp adds and frink never hears of does not
//! produce a refusal, it produces `resolve_architecture` returning
//! `None` for a real file somebody downloaded.
//!
//! The claim is true as of the `5b59b83` pin -- 153 of 153, measured
//! by this test rather than asserted -- and these keep it true.
//!
//! # Why the names are committed rather than read from a checkout
//!
//! CI has no llama.cpp source. A coverage test that skipped itself
//! when the source was absent would pass everywhere and check nothing
//! on the machine that matters, which is this repo's "a gate that
//! cannot fire" rule. So the list is DATA, committed beside this file
//! with the pin it came from, and
//! [`the_pinned_names_match_a_real_checkout`] is the `#[ignore]`d
//! re-measurement that regenerates it -- the same shape as every
//! golden here.
//!
//! It sits directly in `tests/` and not in a `tests/data/` beside it,
//! because `.gitignore` carries a blanket `data/` rule: the first
//! version of this file was invisible to git, which would have passed
//! here and failed in CI with the fixture simply absent. Cargo builds
//! only `.rs` files in `tests/` as targets, so a `.txt` here is inert.
//!
//! # What this does NOT claim
//!
//! That frink runs them. Most of the catalog is refusals and deferrals,
//! which is coverage in this repo, not a gap. What it claims is that
//! every name has a decided answer: run it, refuse it by name, or
//! defer it with a reason.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use frink_models::capability::{architecture_catalog, resolve_architecture};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/frink-models has two ancestors")
        .to_path_buf()
}

/// The pinned llama.cpp names, comments stripped.
fn pinned_names() -> BTreeSet<String> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/llama_cpp_arch_names.txt");
    let text =
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()));
    let names: BTreeSet<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    assert!(
        names.len() > 100,
        "the pinned name list has only {} entries; it is truncated or \
         the comment filter ate it",
        names.len()
    );
    names
}

fn catalog_names() -> BTreeSet<String> {
    architecture_catalog()
        .iter()
        .map(|p| p.gguf_name.to_string())
        .collect()
}

/// **Every name llama.cpp writes, the catalog answers for.**
///
/// The one that fails when the pin moves and nobody re-reads the
/// inventory. A missing name is not a refusal: it is
/// `resolve_architecture` returning `None` on a file a user really
/// has.
#[test]
fn the_catalog_answers_for_every_llama_cpp_architecture() {
    let pinned = pinned_names();
    let catalog = catalog_names();

    let missing: Vec<&String> = pinned.difference(&catalog).collect();
    assert!(
        missing.is_empty(),
        "{} llama.cpp architecture name(s) have no catalog row: {missing:?}\n\
         Each needs a row -- audited, dedicated, refused by name, or deferred \
         with a reason. A name with no row answers `None`, which is not a refusal.",
        missing.len()
    );

    // And the answer has to be reachable through the resolver, not
    // merely present in the table: a row the resolver cannot reach is
    // the `kimi_k3` underscore defect, where a refusal sat in the
    // catalog under a spelling no file carries.
    for name in &pinned {
        assert!(
            resolve_architecture(name).is_some(),
            "`{name}` has a catalog row but `resolve_architecture` \
             answers None for it"
        );
    }
}

/// **A catalog row that llama.cpp does not name has to be explained.**
///
/// The extras are deliberate: in-repo test fixtures, the two
/// hyphenated Granite aliases, `phi4`, and the four strings that are
/// refused precisely BECAUSE no converter writes them. Pinned by name
/// so a fifth one cannot be added without saying which kind it is --
/// an unexplained extra is a row somebody invented.
#[test]
fn every_catalog_row_llama_cpp_does_not_name_is_accounted_for() {
    const EXPLAINED: &[(&str, &str)] = &[
        ("ferroxtest", "in-repo GGUF test fixture"),
        ("ferroxtestmoe", "in-repo GGUF test fixture"),
        ("ferroxtestmixed", "in-repo GGUF test fixture"),
        ("granite-hybrid", "hyphenated alias of `granitehybrid`"),
        ("granite-moe", "hyphenated alias of `granitemoe`"),
        (
            "mistral",
            "refused: no converter writes it; files say `llama`",
        ),
        (
            "mixtral",
            "refused: no converter writes it; files say `llama`",
        ),
        ("yi", "refused: no converter writes it; files say `llama`"),
        ("yi-vl", "deferred multimodal; llama.cpp names it elsewhere"),
        ("phi4", "alias; real exports declare `phi3`"),
    ];

    let pinned = pinned_names();
    let catalog = catalog_names();
    let extras: BTreeSet<&str> = catalog
        .iter()
        .map(String::as_str)
        .filter(|n| !pinned.contains(*n))
        .collect();
    let explained: BTreeSet<&str> = EXPLAINED.iter().map(|(n, _)| *n).collect();

    let unexplained: Vec<&&str> = extras.difference(&explained).collect();
    assert!(
        unexplained.is_empty(),
        "catalog row(s) llama.cpp does not name and nothing explains: {unexplained:?}"
    );

    // The list must not rot in the other direction either: an entry
    // here for a row that no longer exists reads as coverage of
    // something absent.
    let stale: Vec<&&str> = explained.difference(&extras).collect();
    assert!(
        stale.is_empty(),
        "EXPLAINED names rows that are no longer in the catalog: {stale:?}"
    );
}

/// The published manifest is generated from the catalog, so it has to
/// still say what the catalog says -- every CELL of it.
///
/// `docs/manifests/architecture_manifest.md` is the artifact a reader
/// meets, and a committed generated file drifts the moment somebody
/// changes a row and does not regenerate.
///
/// **Compared row by row and not by name**, because the first version
/// of this test compared names only and passed over two rows that were
/// really wrong: `llama-embed` was published as
/// `DeferredEncoderEmbedding / deferred` after the PR that made it run
/// on the generic path, and `minimax-01` was published as
/// `StandardGqa / KvGqa` after it moved to `Hybrid`. A reader would
/// have been told the first was not served at all. A name-only
/// comparison cannot see either, which is the whole defect: the
/// interesting drift is in what a row SAYS, not in whether it exists.
#[test]
fn the_published_manifest_matches_the_catalog() {
    let p = repo_root().join("docs/manifests/architecture_manifest.md");
    let text =
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()));

    let published: BTreeSet<String> = text
        .lines()
        .filter(|l| l.starts_with("| `"))
        .map(|l| l.trim().to_string())
        .collect();
    let generated: BTreeSet<String> = frink_models::capability::coverage_report_markdown()
        .lines()
        .filter(|l| l.starts_with("| `"))
        .map(|l| l.trim().to_string())
        .collect();

    assert!(
        !generated.is_empty(),
        "the generator produced no rows; the comparison would be vacuous"
    );

    let stale: Vec<&String> = published.difference(&generated).collect();
    let fresh: Vec<&String> = generated.difference(&published).collect();
    assert!(
        stale.is_empty() && fresh.is_empty(),
        "the manifest has drifted from the catalog.\n  \
         published but no longer generated: {stale:#?}\n  \
         generated but not published: {fresh:#?}\n\
         Regenerate with `frink archs --write docs/manifests/architecture_manifest.md` \
         rather than editing it by hand."
    );
}

/// The re-measurement behind the pinned list.
///
/// `#[ignore]`d because it needs a llama.cpp checkout at the pin. Run
/// it when the pin moves: it re-derives `LLM_ARCH_NAMES` from source
/// and fails with the diff, which is the signal to update the data
/// file and then let the tests above say what the new names need.
#[test]
#[ignore = "needs a llama.cpp checkout under .scratch/"]
fn the_pinned_names_match_a_real_checkout() {
    let src = repo_root().join(".scratch/llama.cpp/src/llama-arch.cpp");
    let text =
        std::fs::read_to_string(&src).unwrap_or_else(|e| panic!("reading {}: {e}", src.display()));

    // `{ LLM_ARCH_FOO, "foo" }` -- the table's one shape.
    let mut derived: BTreeSet<String> = BTreeSet::new();
    for line in text.lines() {
        let Some(rest) = line.split("LLM_ARCH_").nth(1) else {
            continue;
        };
        let Some(open) = rest.find('"') else { continue };
        let after = &rest[open + 1..];
        let Some(close) = after.find('"') else {
            continue;
        };
        derived.insert(after[..close].to_string());
    }
    assert!(
        derived.len() > 100,
        "parsed only {} names out of llama-arch.cpp; the table's shape changed",
        derived.len()
    );

    let pinned = pinned_names();
    let added: Vec<&String> = derived.difference(&pinned).collect();
    let removed: Vec<&String> = pinned.difference(&derived).collect();
    assert!(
        added.is_empty() && removed.is_empty(),
        "the checkout disagrees with the pinned list.\n  \
         new upstream: {added:?}\n  gone upstream: {removed:?}\n\
         Update tests/llama_cpp_arch_names.txt, then let the coverage \
         test say what the new names need."
    );
}
