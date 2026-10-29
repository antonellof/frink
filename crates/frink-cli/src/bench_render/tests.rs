//! Tests for [`super::render`], split out of `bench_render.rs` to keep it
//! under ADR 0008's size bound.

use super::*;

/// Writes one engine receipt for `host` into `dir`, with whatever
/// tests the caller names.
fn receipt_with(dir: &Path, id: &str, host: &str, backend: &str, tests: &[(&str, f64, f64)]) {
    receipt_versioned(dir, id, host, backend, CURRENT_VERSION, tests);
}

fn receipt_versioned(
    dir: &Path,
    id: &str,
    host: &str,
    backend: &str,
    version: &str,
    tests: &[(&str, f64, f64)],
) {
    let tests: Vec<serde_json::Value> = tests
        .iter()
        .map(|(test, frink, llama)| {
            serde_json::json!({
                "test": test, "frink_tps": frink, "llama_tps": llama,
                "gap": llama / frink,
            })
        })
        .collect();
    let r = serde_json::json!({
        "schema": 2, "kind": "engine", "id": id,
        "backend": backend, "backend_active": backend,
        "host_spec": {"label": host},
        "frink_version": version,
        "tests": tests,
    });
    write(dir, &format!("{id}_{backend}_{version}"), &r);
}

/// A schema-3 receipt measured against llama.cpp `commit`.
fn receipt_against(dir: &Path, id: &str, host: &str, commit: &str, frink: f64, llama: f64) {
    let r = serde_json::json!({
        "schema": 3, "kind": "engine", "id": id,
        "backend": "cuda", "backend_active": "cuda",
        "host_spec": {"label": host},
        "frink_version": CURRENT_VERSION,
        "tests": [{"test": "tg128", "frink_tps": frink, "llama_tps": llama,
                   "gap": llama / frink}],
        "llama_cpp": {"build_commit": commit, "build_number": 7000,
                      "backends": "CUDA", "command": ["llama-bench"]},
    });
    write(dir, &format!("{id}_cuda_{commit}"), &r);
}

fn write(dir: &Path, stem: &str, r: &serde_json::Value) {
    std::fs::write(
        dir.join(format!("{stem}.json")),
        serde_json::to_string(r).expect("json"),
    )
    .expect("write receipt");
}

fn receipt(dir: &Path, id: &str, host: &str, backend: &str, frink: f64, llama: f64) {
    receipt_with(dir, id, host, backend, &[("tg128", frink, llama)]);
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "frink_{tag}_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("receipts").join("engine")).expect("mkdir");
    std::fs::write(
        dir.join("RESULTS.md"),
        format!("head\n{BEGIN}\nold\n{END}\ntail\n"),
    )
    .expect("seed");
    dir
}

/// A model's prefill and decode land on ONE row, and every number
/// from both receipts survives the pivot.
///
/// The table used to print one row per (model, test), so a model's
/// two halves sat ten rows apart and a reader comparing them had to
/// find both. The risk in collapsing them is dropping a cell
/// silently, so this asserts all four tok/s values and both gaps.
///
/// Sabotage: drop the `e.tg_frink = r.frink` assignment in
/// `pivot`; the decode column goes to `—` and this goes red.
#[test]
fn a_models_prefill_and_decode_share_one_row() {
    let dir = scratch("pivot");
    let engine = dir.join("receipts").join("engine");
    receipt_with(
        &engine,
        "m1",
        "Box One",
        "metal",
        &[("pp512", 100.0, 200.0), ("tg128", 50.0, 25.0)],
    );

    render(&dir).expect("render");
    let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");

    let row = out
        .lines()
        .find(|l| l.starts_with("| m1 |"))
        .unwrap_or_else(|| panic!("no row for the model:\n{out}"));
    // Prefill gap 2.00 (slower), decode gap 0.50 (faster), then the
    // four raw numbers in header order.
    assert!(row.contains("🔴 **2.00×**"), "prefill gap missing: {row}");
    assert!(row.contains("🟢 **0.50×**"), "decode gap missing: {row}");
    for cell in ["| 100 |", "| 200 |", "| 50.0 |", "| 25.0 |"] {
        assert!(row.contains(cell), "{cell} missing from {row}");
    }
    // One row, not two: the old shape printed the model twice.
    assert_eq!(
        out.lines().filter(|l| l.starts_with("| m1 |")).count(),
        1,
        "the model must appear on exactly one row:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Column headers name the workloads that were actually measured.
///
/// Hard-coding `pp512` / `tg128` would mislabel a suite run at any
/// other size, which is a wrong number in the ledger with no way to
/// tell from the file.
///
/// Sabotage: replace the `find` with a literal `"pp512"`.
#[test]
fn the_column_headers_name_the_workload_that_ran() {
    let dir = scratch("headers");
    let engine = dir.join("receipts").join("engine");
    receipt_with(
        &engine,
        "m1",
        "Box One",
        "metal",
        &[("pp2048", 10.0, 10.0), ("tg64", 10.0, 10.0)],
    );

    render(&dir).expect("render");
    let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");
    assert!(out.contains("frink pp2048"), "header not derived:\n{out}");
    assert!(out.contains("llama.cpp tg64"), "header not derived:\n{out}");
    assert!(!out.contains("pp512"), "a workload nobody ran:\n{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A model measured on only one of the two workloads still gets a
/// row, with the missing half marked rather than dropped.
///
/// Gemma-4-E2B is exactly this: `llama-bench` has no `gemma4` arch,
/// so its llama.cpp column is blank on both tests. A pivot that
/// required both halves would delete the row and the ledger would
/// silently stop mentioning a model frink runs.
///
/// Sabotage: make `pivot` skip a `ModelRow` whose `tg_gap` is None.
#[test]
fn a_model_measured_on_one_workload_is_not_dropped() {
    let dir = scratch("halfrow");
    let engine = dir.join("receipts").join("engine");
    receipt_with(
        &engine,
        "only_pp",
        "Box One",
        "metal",
        &[("pp512", 7.0, 9.0)],
    );

    render(&dir).expect("render");
    let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");
    let row = out
        .lines()
        .find(|l| l.starts_with("| only_pp |"))
        .unwrap_or_else(|| panic!("row dropped:\n{out}"));
    assert!(row.contains("—"), "the missing half must be marked: {row}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two machines render as two sections, and neither is dropped.
///
/// This used to be a hard refusal ("one table cannot describe
/// them"), which meant the ledger could only ever describe the
/// laptop that happened to run the suite: it claimed a CPU gap of
/// 1.41x to 5.06x while having no x86 or CUDA row at all. The
/// refusal's REASON was right, so rows are separated rather than
/// merged, and this pins that.
#[test]
fn two_hosts_render_as_two_sections_rather_than_an_error() {
    let dir = scratch("render");
    let engine = dir.join("receipts").join("engine");
    receipt(&engine, "m1", "Apple M2 Pro", "metal", 100.0, 90.0);
    receipt(&engine, "m1", "Rented Xeon", "cpu", 10.0, 20.0);

    render(&dir).expect("two hosts must render, not refuse");

    let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");
    assert!(out.contains("Apple M2 Pro"), "first host missing:\n{out}");
    assert!(out.contains("Rented Xeon"), "second host missing:\n{out}");
    assert!(
        out.contains("2 machines"),
        "the reader is not told there are two machines:\n{out}"
    );
    // The whole point: a row is never presented without its host.
    let xeon = out.find("Rented Xeon").expect("host heading");
    let cpu_row = out.find("| 10.0 |");
    if let Some(cpu_row) = cpu_row {
        assert!(
            cpu_row > xeon,
            "the Xeon's row appears before its host heading"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The summary is GENERATED, not written by hand.
///
/// It replaced a hand-written headline table, which is a thing that
/// drifts: the numbers above the fold stop matching the receipts
/// below it and nobody notices, because nothing compares them.
#[test]
fn the_summary_is_derived_from_the_same_rows_as_the_detail_tables() {
    let dir = scratch("summary");
    let engine = dir.join("receipts").join("engine");
    // Two gaps on one host+backend, so the summary must show a range.
    receipt(&engine, "a", "Box One", "cuda", 10.0, 20.0);
    receipt(&engine, "b", "Box One", "cuda", 10.0, 100.0);

    render(&dir).expect("render");
    let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");

    assert!(out.contains("### At a glance"), "no summary table:\n{out}");
    let summary = &out[out.find("### At a glance").expect("summary")..];
    let first_detail = summary.find("\n### ").unwrap_or(summary.len());
    let summary = &summary[..first_detail];
    assert!(
        summary.contains("2.00×") && summary.contains("10.00×"),
        "the summary must span the rows it describes:\n{summary}"
    );
    assert!(
        summary.contains("Box One"),
        "the summary must name the host:\n{summary}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gap_cell_colours_match_the_ledger_convention() {
    assert!(gap_cell(0.80).starts_with("🟢"));
    assert!(gap_cell(1.00).starts_with("⚪"));
    assert!(gap_cell(0.96).starts_with("⚪"));
    assert!(gap_cell(1.40).starts_with("🔴"));
}

#[test]
fn splice_replaces_an_existing_block_and_keeps_the_surrounding_text() {
    let doc = format!("before\n{BEGIN}\nold\n{END}\nafter\n");
    let out = splice(&doc, &format!("{BEGIN}\nnew\n{END}"));
    assert!(out.contains("before"), "text before the block must survive");
    assert!(out.contains("after"), "text after the block must survive");
    assert!(out.contains("new"));
    assert!(!out.contains("old"), "the old block must be gone");
}

#[test]
fn splice_appends_when_the_markers_are_missing() {
    let out = splice("just some prose\n", &format!("{BEGIN}\nfresh\n{END}"));
    assert!(out.starts_with("just some prose"));
    assert!(out.contains("fresh"));
}

/// A render with no receipts must refuse rather than publish an
/// empty table over a real one.
#[test]
fn rendering_nothing_refuses_instead_of_emptying_the_ledger() {
    let dir = std::env::temp_dir().join(format!("frink-render-guard-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("receipts").join("engine")).unwrap();
    let results = dir.join("RESULTS.md");
    let original = "# Results\n\nreal numbers live here\n";
    std::fs::write(&results, original).unwrap();

    let err = render(&dir).unwrap_err().to_string();
    assert!(
        err.contains("nothing to render"),
        "expected a refusal naming the empty receipt dir, got: {err}"
    );
    assert_eq!(
        std::fs::read_to_string(&results).unwrap(),
        original,
        "the existing ledger must survive a render that had no receipts"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn splice_does_not_duplicate_the_block_on_a_second_render() {
    let block = format!("{BEGIN}\nv1\n{END}");
    let once = splice("doc\n", &block);
    let twice = splice(&once, &format!("{BEGIN}\nv2\n{END}"));
    assert_eq!(twice.matches(BEGIN).count(), 1, "exactly one engine block");
    assert!(twice.contains("v2") && !twice.contains("v1"));
}
/// **A ledger row must say which build produced it.**
///
/// The defect this pins: every CUDA row published here was measured
/// before the K-quant GEMM landed and overstated the gap by about
/// six times, and the rendered table said nothing about its own
/// age. A number that was true when taken and is false when read is
/// the most expensive kind in this file, because a gap column
/// cannot show it.
#[test]
fn a_row_measured_on_an_older_build_is_marked_stale() {
    let dir = scratch("stale");
    let rdir = dir.join("receipts").join("engine");
    receipt_versioned(
        &rdir,
        "a",
        "Host A",
        "cuda",
        "0.17.1",
        &[("tg128", 10.0, 50.0)],
    );
    render(&dir).expect("render");
    let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");

    assert!(
        out.contains("`0.17.1`"),
        "the version that produced the row is not in the table:\n{out}"
    );
    // The marker has to be on the ROW, not only in the note below
    // the table. Asserting `out.contains` alone passed with the
    // column's marker removed, because the note carries the same
    // glyph -- the test was reading the wrong half of its own
    // subject.
    let row = out
        .lines()
        .find(|l| l.contains("0.17.1") && l.starts_with('|'))
        .unwrap_or_else(|| panic!("no summary row for the receipt:\n{out}"));
    assert!(
        row.contains("⚠️"),
        "the summary row carries no staleness marker: {row}"
    );
    assert!(
        out.contains(CURRENT_VERSION),
        "the note does not say which build is current:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The mark is not decoration: a ledger measured on THIS build
/// carries no warning, so the marker means something when it
/// appears.
#[test]
fn a_row_measured_on_this_build_is_not_marked() {
    let dir = scratch("current");
    let rdir = dir.join("receipts").join("engine");
    receipt(&rdir, "a", "Host A", "cuda", 10.0, 50.0);
    render(&dir).expect("render");
    let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");

    assert!(
        out.contains(&format!("`{CURRENT_VERSION}`")),
        "the current version is not in the table:\n{out}"
    );
    assert!(
        !out.contains("⚠️"),
        "a ledger measured on this build must carry no staleness \
         marker, or the marker says nothing when it appears:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two llama.cpp builds on ONE machine are two sections, each heading
/// naming its build, and neither table holds the other's row (ADR
/// 0006). The summary keeps them on separate rows too, rather than
/// printing one range across both builds.
///
/// Sabotage: filter `here` by host only in `render`; each section then
/// holds both rows and this goes red.
#[test]
fn two_llama_builds_on_one_host_render_as_two_sections() {
    let dir = scratch("two_refs");
    let engine = dir.join("receipts").join("engine");
    receipt_against(&engine, "m1", "Rented Box", "aaaaaaa", 10.0, 20.0);
    receipt_against(&engine, "m2", "Rented Box", "bbbbbbb", 10.0, 30.0);

    render(&dir).expect("render");
    let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");

    assert!(
        out.contains("2 llama.cpp references"),
        "the reader is not told there are two builds:\n{out}"
    );
    let a = out
        .find("### Rented Box, llama.cpp `aaaaaaa`")
        .unwrap_or_else(|| panic!("no section for the first build:\n{out}"));
    let b = out
        .find("### Rented Box, llama.cpp `bbbbbbb`")
        .unwrap_or_else(|| panic!("no section for the second build:\n{out}"));
    assert!(a < b, "sections out of order:\n{out}");
    let first = &out[a..b];
    let second = &out[b..];
    assert!(
        first.contains("| m1 |") && !first.contains("| m2 |"),
        "{first}"
    );
    assert!(
        second.contains("| m2 |") && !second.contains("| m1 |"),
        "{second}"
    );

    let summary = &out[out.find("### At a glance").expect("summary")..a];
    let summary_rows: Vec<&str> = summary
        .lines()
        .filter(|l| l.starts_with("| Rented Box"))
        .collect();
    assert_eq!(
        summary_rows.len(),
        2,
        "one summary row per build:\n{summary}"
    );
    assert!(summary_rows
        .iter()
        .any(|l| l.contains("`aaaaaaa`") && l.contains("2.00×")));
    assert!(summary_rows
        .iter()
        .any(|l| l.contains("`bbbbbbb`") && l.contains("3.00×")));
    let _ = std::fs::remove_dir_all(&dir);
}

/// One build on one machine stays one flat table, with the build named
/// once above it.
#[test]
fn one_llama_build_is_named_above_the_table() {
    let dir = scratch("one_ref");
    let engine = dir.join("receipts").join("engine");
    receipt_against(&engine, "m1", "Rented Box", "aaaaaaa", 10.0, 20.0);
    receipt_against(&engine, "m2", "Rented Box", "aaaaaaa", 10.0, 30.0);

    render(&dir).expect("render");
    let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");
    assert!(
        out.contains("against llama.cpp `aaaaaaa` (build 7000, CUDA)"),
        "the build is not named:\n{out}"
    );
    assert!(!out.contains("llama.cpp references"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Receipts written before the field still render, marked as having an
/// unknown reference, and no build is invented for them. Beside a
/// receipt that does name its build, they are a separate section.
///
/// Sabotage: map a missing `llama_cpp` key to `Reference::NoComparison`
/// in `Reference::from_receipt`; "unknown" disappears and this goes red.
#[test]
fn receipts_from_before_the_field_render_with_an_unknown_reference() {
    let dir = scratch("unknown_ref");
    let engine = dir.join("receipts").join("engine");
    receipt(&engine, "old", "Rented Box", "cuda", 10.0, 20.0);
    receipt_against(&engine, "new", "Rented Box", "aaaaaaa", 10.0, 30.0);

    render(&dir).expect("old receipts must still render");
    let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");

    let unknown = out
        .find("### Rented Box, llama.cpp unknown")
        .unwrap_or_else(|| panic!("no unknown-reference section:\n{out}"));
    let section = &out[unknown..];
    let section = &section[..section[4..].find("\n### ").map_or(section.len(), |i| i + 4)];
    assert!(section.contains("| old |"), "{section}");
    assert!(!section.contains("| new |"), "{section}");
    assert!(section.contains("**unknown**"), "{section}");
    assert!(
        !section.contains("aaaaaaa"),
        "a build was filled in for a receipt that never recorded one:\n{section}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
