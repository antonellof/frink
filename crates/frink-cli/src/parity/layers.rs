//! `frink parity --dump-tensors <dir>`: where in the forward pass the
//! two engines part, per layer and per op (#476).
//!
//! First-token KL is one number for the whole pass. When a change moves
//! it the wrong way, it cannot say whether the change is wrong or a
//! second deviation elsewhere was cancelling the first. Both engines
//! dump the prompt's intermediates under llama.cpp's tensor names
//! (`tools/llama_logits.c` with `LLAMA_LOGITS_DUMP_TENSORS`, and
//! `frink_models::tensor_dump`), and this prints, for every tensor both
//! wrote, the relative L2 distance of frink's from llama.cpp's.
//!
//! Read a row against the one before it. A tensor whose distance jumps
//! from its input's is the op that diverges; one that tracks its input
//! is inheriting the difference.
//!
//! Rows are compared from the END: llama.cpp computes the last layer's
//! tail for the output position only (`inp_out_ids`), so its `l_out` of
//! that layer and `result_norm` hold one row where frink's hold the
//! prompt.

use anyhow::Context;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One line of a `manifest.tsv`.
#[derive(Debug, Clone, PartialEq)]
struct Entry {
    name: String,
    width: usize,
    rows: usize,
}

/// One compared tensor.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Row {
    pub name: String,
    pub width: usize,
    pub rows: usize,
    /// `||frink - llama|| / ||llama||` over the rows both hold.
    pub rel_l2: f64,
    /// The largest per-row distance. The figure above is dominated by
    /// the row with the largest norm (the first token's, in most
    /// checkpoints), so a row that moved can hide under it; this cannot.
    pub rel_l2_worst: f64,
    /// The same over the last row only: the output position.
    pub rel_l2_last: f64,
}

/// The directory frink writes into; set before the frink prefill runs.
pub(super) fn frink_dir(root: &Path) -> PathBuf {
    root.join("frink")
}

/// Runs the primary dumper once more with the dump switched on, checks
/// that dumping did not change its answer, and prints the comparison.
pub(super) fn run(
    dumper: &Path,
    model: &str,
    tokens: &[u32],
    root: &Path,
    reference_logits: &[f32],
) -> anyhow::Result<()> {
    frink_models::tensor_dump::finish().context("writing frink's tensor dump")?;
    let llama_dir = root.join("llama");
    std::fs::create_dir_all(&llama_dir)
        .with_context(|| format!("creating {}", llama_dir.display()))?;
    let dumped = super::reference_logits(dumper, model, tokens, usize::MAX, Some(&llama_dir))
        .context("running the reference dumper with LLAMA_LOGITS_DUMP_TENSORS")?;
    if !llama_dir.join("manifest.tsv").exists() {
        anyhow::bail!(
            "{} wrote no tensor dump: a dumper built before LLAMA_LOGITS_DUMP_TENSORS ignores \
             it. Rebuild it from this tree (tools/build_llama_logits.sh)",
            dumper.display()
        );
    }
    // The eval callback splits llama.cpp's graph at every dumped node.
    // It should not change any node's arithmetic; if it did, the dump
    // describes a different pass from the one the verdict scored.
    let same = dumped.logits.len() == reference_logits.len()
        && dumped
            .logits
            .iter()
            .zip(reference_logits)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    println!();
    println!(
        "per-tensor distance, frink against llama.cpp ({})",
        root.display()
    );
    if !same {
        println!(
            "  WARNING: llama.cpp's logits with the dump on differ from its logits without it; \
             the tensors below are from the dumped pass"
        );
    }
    let rows = compare_dirs(&llama_dir, &frink_dir(root))?;
    print_rows(&rows.compared);
    if !rows.missing.is_empty() {
        println!(
            "  {} llama.cpp tensors have no frink counterpart: {}",
            rows.missing.len(),
            rows.missing.join(" ")
        );
    }
    Ok(())
}

pub(super) struct Comparison {
    pub compared: Vec<Row>,
    /// llama.cpp tensors frink did not write, or wrote at another width.
    pub missing: Vec<String>,
}

/// Every llama.cpp tensor, in its graph order, against frink's tensor of
/// the same name.
pub(super) fn compare_dirs(llama: &Path, frink: &Path) -> anyhow::Result<Comparison> {
    let llama_entries = read_manifest(llama)?;
    let frink_entries: HashMap<String, Entry> = read_manifest(frink)?
        .into_iter()
        .map(|e| (e.name.clone(), e))
        .collect();
    // Refused rather than printed as a table of nothing but misses: the
    // flag was asked for and could not be honoured (ADR 0005).
    if frink_entries.is_empty() {
        anyhow::bail!(
            "frink wrote no tensors: --dump-tensors covers the generic decoder's batched \
             prefill on the CPU, and this checkpoint ran another engine or backend"
        );
    }
    let mut compared = Vec::new();
    let mut missing = Vec::new();
    for l in &llama_entries {
        let Some(f) = frink_entries.get(&l.name).filter(|f| f.width == l.width) else {
            missing.push(l.name.clone());
            continue;
        };
        let a = read_tensor(llama, l)?;
        let b = read_tensor(frink, f)?;
        let rows = l.rows.min(f.rows);
        let n = rows * l.width;
        let (a, b) = (&a[a.len() - n..], &b[b.len() - n..]);
        let w = l.width;
        compared.push(Row {
            name: l.name.clone(),
            width: w,
            rows,
            rel_l2: rel_l2(a, b),
            rel_l2_worst: a
                .chunks(w)
                .zip(b.chunks(w))
                .map(|(x, y)| rel_l2(x, y))
                .fold(0.0, f64::max),
            rel_l2_last: rel_l2(&a[n - w..], &b[n - w..]),
        });
    }
    Ok(Comparison { compared, missing })
}

/// `||b - a|| / ||a||`, accumulated in f64. Zero when both are zero.
pub(super) fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = y as f64 - x as f64;
        num += d * d;
        den += x as f64 * x as f64;
    }
    if den == 0.0 {
        if num == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (num / den).sqrt()
    }
}

fn print_rows(rows: &[Row]) {
    println!(
        "  {:<18} {:>6} {:>6} {:>12} {:>12} {:>12}",
        "tensor", "width", "rows", "rel L2", "worst row", "last row"
    );
    for r in rows {
        println!(
            "  {:<18} {:>6} {:>6} {:>12.3e} {:>12.3e} {:>12.3e}",
            r.name, r.width, r.rows, r.rel_l2, r.rel_l2_worst, r.rel_l2_last
        );
    }
}

fn read_manifest(dir: &Path) -> anyhow::Result<Vec<Entry>> {
    let path = dir.join("manifest.tsv");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut f = line.split('\t');
            let (Some(name), Some(width), Some(rows), None) =
                (f.next(), f.next(), f.next(), f.next())
            else {
                anyhow::bail!("{}: `{line}` is not `name\\twidth\\trows`", path.display());
            };
            let width: usize = width
                .parse()
                .with_context(|| format!("width in `{line}`"))?;
            let rows: usize = rows.parse().with_context(|| format!("rows in `{line}`"))?;
            if width == 0 || rows == 0 {
                anyhow::bail!("{}: `{line}` describes an empty tensor", path.display());
            }
            Ok(Entry {
                name: name.to_string(),
                width,
                rows,
            })
        })
        .collect()
}

/// The tensor `e` names, held to the size its manifest line claims.
fn read_tensor(dir: &Path, e: &Entry) -> anyhow::Result<Vec<f32>> {
    let path = dir.join(format!("{}.f32", e.name));
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let want = e
        .width
        .checked_mul(e.rows)
        .and_then(|n| n.checked_mul(4))
        .with_context(|| format!("{}: width x rows overflows", e.name))?;
    if bytes.len() != want {
        anyhow::bail!(
            "{} holds {} bytes, its manifest line says {want}",
            path.display(),
            bytes.len()
        );
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, manifest: &str, tensors: &[(&str, &[f32])]) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("manifest.tsv"), manifest).unwrap();
        for (name, data) in tensors {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
            std::fs::write(dir.join(format!("{name}.f32")), bytes).unwrap();
        }
    }

    #[test]
    fn rel_l2_is_relative_to_the_reference() {
        assert_eq!(rel_l2(&[3.0, 4.0], &[3.0, 4.0]), 0.0);
        // ||(0.3, 0.4)|| / ||(3, 4)|| = 0.5 / 5.
        assert!((rel_l2(&[3.0, 4.0], &[3.3, 4.4]) - 0.1).abs() < 1e-6);
        assert_eq!(rel_l2(&[0.0], &[0.0]), 0.0);
        assert!(rel_l2(&[0.0], &[1.0]).is_infinite());
    }

    /// llama.cpp's last layer holds the output row only; the comparison
    /// must line that row up with frink's LAST row, not its first.
    #[test]
    fn rows_are_aligned_from_the_end() {
        let root = std::env::temp_dir().join(format!("frink-layers-{}", std::process::id()));
        let (llama, frink) = (root.join("llama"), root.join("frink"));
        write(
            &llama,
            "l_out-1\t2\t1\nQcur-0\t2\t2\n",
            &[("l_out-1", &[5.0, 6.0]), ("Qcur-0", &[1.0, 1.0, 2.0, 2.0])],
        );
        write(
            &frink,
            "Qcur-0\t2\t2\nl_out-1\t2\t3\n",
            &[
                ("Qcur-0", &[1.0, 1.0, 2.0, 2.2]),
                ("l_out-1", &[9.0, 9.0, 9.0, 9.0, 5.0, 6.0]),
            ],
        );
        let c = compare_dirs(&llama, &frink).unwrap();
        std::fs::remove_dir_all(&root).ok();
        assert!(c.missing.is_empty());
        assert_eq!(c.compared[0].name, "l_out-1");
        assert_eq!(c.compared[0].rows, 1);
        assert_eq!(c.compared[0].rel_l2, 0.0);
        assert_eq!(c.compared[1].name, "Qcur-0");
        assert!(c.compared[1].rel_l2 > 0.0);
        assert!(c.compared[1].rel_l2_last > c.compared[1].rel_l2);
        // Row 0 matches exactly, so the worst row is the last one.
        assert_eq!(c.compared[1].rel_l2_worst, c.compared[1].rel_l2_last);
    }

    #[test]
    fn a_tensor_frink_wrote_at_another_width_is_missing_not_compared() {
        let root = std::env::temp_dir().join(format!("frink-layers-w-{}", std::process::id()));
        let (llama, frink) = (root.join("llama"), root.join("frink"));
        write(
            &llama,
            "Qcur-0#2\t4\t1\nkqv_out-0\t2\t1\n",
            &[("Qcur-0#2", &[1.0; 4]), ("kqv_out-0", &[1.0; 2])],
        );
        write(&frink, "Qcur-0#2\t2\t2\n", &[("Qcur-0#2", &[1.0; 4])]);
        let c = compare_dirs(&llama, &frink).unwrap();
        std::fs::remove_dir_all(&root).ok();
        assert!(c.compared.is_empty());
        assert_eq!(c.missing, vec!["Qcur-0#2", "kqv_out-0"]);
    }

    #[test]
    fn a_frink_side_that_wrote_nothing_is_refused() {
        let root = std::env::temp_dir().join(format!("frink-layers-e-{}", std::process::id()));
        let (llama, frink) = (root.join("llama"), root.join("frink"));
        write(&llama, "l_out-0\t2\t1\n", &[("l_out-0", &[1.0; 2])]);
        write(&frink, "", &[]);
        let err = compare_dirs(&llama, &frink).err().unwrap();
        std::fs::remove_dir_all(&root).ok();
        assert!(
            format!("{err:#}").contains("frink wrote no tensors"),
            "{err:#}"
        );
    }

    #[test]
    fn a_tensor_shorter_than_its_manifest_line_is_refused() {
        let root = std::env::temp_dir().join(format!("frink-layers-s-{}", std::process::id()));
        let (llama, frink) = (root.join("llama"), root.join("frink"));
        write(&llama, "l_out-0\t2\t2\n", &[("l_out-0", &[1.0; 3])]);
        write(&frink, "l_out-0\t2\t2\n", &[("l_out-0", &[1.0; 4])]);
        let err = compare_dirs(&llama, &frink).err().unwrap();
        std::fs::remove_dir_all(&root).ok();
        assert!(
            format!("{err:#}").contains("manifest line says 16"),
            "{err:#}"
        );
    }
}
