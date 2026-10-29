//! The greedy multi-step decode half of `frink parity`.
//!
//! The first-token comparison reads the last PROMPT position, which is
//! prefill. A decode-side defect (a KV push skipped once the cache
//! grows, a RoPE position off by one in the single-token body, a
//! window evicted one row early, a recurrent state not carried) can
//! produce fluent nonsense while that half stays MATCH. This half asks
//! the dumper for llama.cpp's own greedy decode
//! (`LLAMA_LOGITS_DECODE_STEPS`, `tools/llama_logits.c`) and feeds
//! frink the SAME tokens one at a time, so every step compares two
//! distributions over one context.
//!
//! Teacher-forced rather than free-running: a near tie that flips at
//! step 7 is reported at step 7 and does not turn steps 8 to 32 into a
//! comparison of two different prefixes.

use super::calibration::KL_WRONG;
use super::{metrics, KL_NOISE};
use anyhow::Context;
use std::path::Path;
use std::process::Command;

/// The decode verdict: the WORST step decides it.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(super) enum StepVerdict {
    Match,
    Drift,
    TieFlip,
    Wrong,
}

impl StepVerdict {
    pub(super) fn label(self) -> &'static str {
        match self {
            StepVerdict::Match => "MATCH",
            StepVerdict::Drift => "DRIFT",
            StepVerdict::TieFlip => "TIE-FLIP",
            StepVerdict::Wrong => "WRONG",
        }
    }
}

pub(super) struct Step {
    pub(super) kl: f64,
    pub(super) top1_ref: usize,
    pub(super) top1_frink: usize,
    pub(super) verdict: StepVerdict,
}

/// One step, on the same ladder as the first-token verdict. The WRONG
/// line is the absolute floor [`KL_WRONG`]: the decode half runs on the
/// primary reference only, so it has no measured spread to widen it,
/// which is right for the F32 / F16 / Q8_0 files it is meant for.
pub(super) fn judge(reference: &[f32], frink: &[f32]) -> Step {
    let p = metrics::softmax(reference);
    let q = metrics::softmax(frink);
    let kl = metrics::kl(&p, &q);
    let (_, max_delta) = metrics::total_variation(&p, &q);
    let ref_order = metrics::order_desc(&p);
    let top1_ref = ref_order[0];
    let top1_frink = metrics::order_desc(&q)[0];
    let margin = ref_order.get(1).map_or(1.0, |&i2| p[top1_ref] - p[i2]);
    let verdict = if top1_ref == top1_frink {
        if kl < KL_NOISE {
            StepVerdict::Match
        } else if kl < KL_WRONG {
            StepVerdict::Drift
        } else {
            StepVerdict::Wrong
        }
    } else if kl < KL_WRONG && margin <= max_delta {
        StepVerdict::TieFlip
    } else {
        StepVerdict::Wrong
    };
    Step {
        kl,
        top1_ref,
        top1_frink,
        verdict,
    }
}

/// llama.cpp's greedy decode: `steps` rows of `n_vocab` logits, row 0
/// at the last prompt position.
fn reference_rows(
    dumper: &Path,
    model: &Path,
    prompt: &[u32],
    steps: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let out_path =
        std::env::temp_dir().join(format!("frink-parity-decode-{}.bin", std::process::id()));
    let out = Command::new(dumper)
        .env("LLAMA_LOGITS_DECODE_STEPS", steps.to_string())
        .arg(model)
        .arg(&out_path)
        .args(prompt.iter().map(u32::to_string))
        .output()
        .context("running the reference dumper for the decode")?;
    if !out.status.success() {
        anyhow::bail!(
            "reference decode failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("(no stderr)")
        );
    }
    let bytes = std::fs::read(&out_path)
        .with_context(|| format!("reading the reference decode from {}", out_path.display()))?;
    let _ = std::fs::remove_file(&out_path);
    let floats: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    if steps == 0
        || floats.is_empty()
        || !floats.len().is_multiple_of(steps)
        || !bytes.len().is_multiple_of(4)
    {
        anyhow::bail!(
            "the reference decode wrote {} bytes, not {steps} whole rows of f32 (a dumper \
             without LLAMA_LOGITS_DECODE_STEPS writes one row: rebuild it with \
             tools/build_llama_logits.sh)",
            bytes.len()
        );
    }
    let n_vocab = floats.len() / steps;
    Ok(floats.chunks(n_vocab).map(<[f32]>::to_vec).collect())
}

/// The first maximum, as the dumper's own argmax picks it: a tie must
/// resolve to the same token on both sides of the comparison.
fn first_argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

/// Runs the decode comparison and prints one line per step plus a
/// verdict line. Returns the worst step's verdict.
pub(super) fn run(
    dumper: &Path,
    model: &Path,
    prompt: &[u32],
    steps: usize,
) -> anyhow::Result<StepVerdict> {
    let reference = reference_rows(dumper, model, prompt, steps)?;
    let fed: Vec<usize> = reference[..steps - 1]
        .iter()
        .map(|r| first_argmax(r))
        .collect();
    let prompt: Vec<usize> = prompt.iter().map(|&t| t as usize).collect();
    let frink = crate::verify_engine::teacher_forced_logits(model, &prompt, &fed)
        .context("frink teacher-forced decode")?;
    if frink.len() != reference.len() || frink[0].len() != reference[0].len() {
        anyhow::bail!(
            "decode shapes disagree: llama.cpp {} x {}, frink {} x {}",
            reference.len(),
            reference[0].len(),
            frink.len(),
            frink[0].len()
        );
    }
    let judged: Vec<Step> = reference
        .iter()
        .zip(&frink)
        .map(|(r, f)| judge(r, f))
        .collect();
    let worst = judged
        .iter()
        .map(|s| s.verdict)
        .max()
        .unwrap_or(StepVerdict::Match);
    let worst_kl = judged.iter().map(|s| s.kl).fold(0.0, f64::max);
    println!(
        "decode: {} ({steps} greedy steps, frink fed llama.cpp's tokens; worst KL(llama||frink) \
         {worst_kl:.3e})",
        worst.label()
    );
    for (i, s) in judged.iter().enumerate() {
        if s.verdict != StepVerdict::Match {
            println!(
                "  step {i:>3}  {:8}  KL {:.3e}  top-1 llama {} / frink {}",
                s.verdict.label(),
                s.kl,
                s.top1_ref,
                s.top1_frink
            );
        }
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_row_is_a_match() {
        let r = [0.1f32, 4.0, -1.0];
        assert_eq!(judge(&r, &r).verdict, StepVerdict::Match);
    }

    #[test]
    fn a_decisive_different_token_is_wrong() {
        let a = [0.0f32, 8.0, 0.0];
        let b = [0.0f32, 0.0, 8.0];
        assert_eq!(judge(&a, &b).verdict, StepVerdict::Wrong);
    }

    #[test]
    fn the_worst_step_orders_above_the_rest() {
        assert!(StepVerdict::Wrong > StepVerdict::TieFlip);
        assert!(StepVerdict::TieFlip > StepVerdict::Drift);
        assert!(StepVerdict::Drift > StepVerdict::Match);
    }

    #[test]
    fn a_tie_resolves_to_the_first_maximum_like_the_dumper() {
        assert_eq!(first_argmax(&[1.0, 3.0, 3.0]), 1);
    }
}
