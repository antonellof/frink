//! A Q2_K and a Q3_K model decode on Metal as they do on the CPU (#405).
//!
//! The committed graph fixtures are all F32, so the Metal decode sweep
//! in `decode_step_graphs` never reaches a quantized kernel. This suite
//! writes the 512-wide synthetic llama (`common::synthetic_llama`) with
//! every weight matrix in Q2_K, then in Q3_K, and runs the serving
//! shape on both engines: a batched prefill (the simdgroup GEMM, 6
//! tokens) and then [`STEPS`] single-token decodes (the matvec). The
//! CPU runs in a child process, because a test process is pinned to
//! one backend (`common::backend`).
//!
//! The Metal run replays the CPU's greedy tokens, so every step compares
//! the same position and a near-tie at one step cannot hide the rest.
//! It also reads the kernel registry: a Metal miss for the kind under
//! test means the weights ran on the host and the comparison was the
//! CPU against itself.
//!
//! `frink_quant` has no Q2_K or Q3_K encoder (the formats are load-only
//! there), so the fixture's encoders are here: a plain min/max
//! quantizer per 16-weight sub-block. They only have to be a valid
//! packing of SOME weights; `the_fixture_encoders_round_trip_through_the_cpu_dequant`
//! holds them to `frink_quant`'s dequant, which is what defines the
//! model both engines run.

mod common;

use frink_gguf::{GgmlType, ShardedGguf};
use frink_models::{Decoder, ModelConfig};
use half::f16;
use std::path::Path;

use common::synthetic_llama;

/// Decode steps after the prefill.
const STEPS: usize = 16;

/// Set on the child: the fixture to print rows for.
const CHILD: &str = "FRINK_LOW_K_DECODE_CHILD";

/// One Q2_K block (256 weights, 84 bytes): per sub-block, the range
/// `[min(lo, 0), hi]` in four steps.
fn q2_k_block(v: &[f32], out: &mut Vec<u8>) {
    let mut step = [0f32; 16];
    let mut min = [0f32; 16];
    for s in 0..16 {
        let sub = &v[16 * s..16 * s + 16];
        let lo = sub.iter().copied().fold(0f32, f32::min);
        let hi = sub.iter().copied().fold(f32::MIN, f32::max);
        step[s] = (hi - lo) / 3.0;
        min[s] = -lo;
    }
    let d = f16::from_f32(step.iter().copied().fold(0f32, f32::max) / 15.0);
    let dmin = f16::from_f32(min.iter().copied().fold(0f32, f32::max) / 15.0);
    let (df, dminf) = (d.to_f32(), dmin.to_f32());
    let mut blk = [0u8; 84];
    for s in 0..16 {
        let sc = if df > 0.0 {
            (step[s] / df).round().clamp(0.0, 15.0)
        } else {
            0.0
        };
        let m = if dminf > 0.0 {
            (min[s] / dminf).round().clamp(0.0, 15.0)
        } else {
            0.0
        };
        blk[s] = sc as u8 | ((m as u8) << 4);
        let (dl, ml) = (df * sc, dminf * m);
        let (n, j, h) = (s / 8, (s / 2) % 4, s % 2);
        for l in 0..16 {
            let raw = if dl > 0.0 {
                ((v[16 * s + l] + ml) / dl).round().clamp(0.0, 3.0)
            } else {
                0.0
            };
            blk[16 + 32 * n + 16 * h + l] |= (raw as u8) << (2 * j);
        }
    }
    blk[80..82].copy_from_slice(&d.to_le_bytes());
    blk[82..84].copy_from_slice(&dmin.to_le_bytes());
    out.extend_from_slice(&blk);
}

/// One Q3_K block (256 weights, 110 bytes): per sub-block, a symmetric
/// scale putting `max |v|` at 4 steps, the value stored as `q + 4` in
/// three bits (two in `qs`, the high one in `hmask`).
fn q3_k_block(v: &[f32], out: &mut Vec<u8>) {
    let mut step = [0f32; 16];
    for s in 0..16 {
        step[s] = v[16 * s..16 * s + 16]
            .iter()
            .fold(0f32, |m, x| m.max(x.abs()))
            / 4.0;
    }
    let d = f16::from_f32(step.iter().copied().fold(0f32, f32::max) / 31.0);
    let df = d.to_f32();
    let mut blk = [0u8; 110];
    for s in 0..16 {
        let sc = if df > 0.0 {
            (step[s] / df).round().clamp(0.0, 31.0)
        } else {
            0.0
        };
        let sc6 = sc as u8 + 32;
        blk[96 + s % 8] |= (sc6 & 0xF) << (4 * (s / 8));
        blk[104 + s % 4] |= (sc6 >> 4) << (2 * (s / 4));
        let dl = df * sc;
        let (n, j, h) = (s / 8, (s / 2) % 4, s % 2);
        for l in 0..16 {
            let q = if dl > 0.0 {
                (v[16 * s + l] / dl).round().clamp(-4.0, 3.0)
            } else {
                0.0
            };
            let q = (q + 4.0) as u8;
            blk[32 + 32 * n + 16 * h + l] |= (q & 3) << (2 * j);
            blk[16 * h + l] |= (q >> 2) << (4 * n + j);
        }
    }
    blk[108..110].copy_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&blk);
}

fn encode(dtype: GgmlType, cols: usize, w: &[f32]) -> Vec<u8> {
    let block: fn(&[f32], &mut Vec<u8>) = match dtype {
        GgmlType::Q2K => q2_k_block,
        GgmlType::Q3K => q3_k_block,
        other => panic!("this suite encodes Q2_K and Q3_K, not {other:?}"),
    };
    assert_eq!(cols % 256, 0, "a K-quant row is whole 256-weight blocks");
    let mut out = Vec::new();
    for chunk in w.as_chunks::<256>().0 {
        block(chunk, &mut out);
    }
    out
}

/// The encoders are a packing of the weights `frink_quant` decodes back,
/// not of noise: a wrong field offset or bit position leaves the
/// dequantized row uncorrelated with the input.
#[test]
fn the_fixture_encoders_round_trip_through_the_cpu_dequant() {
    let v: Vec<f32> = (0..4096)
        .map(|i| ((i as f32) * 0.37).sin() * 0.05)
        .collect();
    type Dequant = fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError>;
    for (dtype, dequant, floor) in [
        (GgmlType::Q2K, frink_quant::dequant_q2_k as Dequant, 0.90f64),
        (GgmlType::Q3K, frink_quant::dequant_q3_k as Dequant, 0.97),
    ] {
        let back = dequant(&encode(dtype, 1024, &v)).expect("whole blocks");
        let dot: f64 = v.iter().zip(&back).map(|(a, b)| f64::from(a * b)).sum();
        let nv: f64 = v.iter().map(|a| f64::from(a * a)).sum::<f64>().sqrt();
        let nb: f64 = back.iter().map(|a| f64::from(a * a)).sum::<f64>().sqrt();
        let cos = dot / (nv * nb);
        assert!(
            cos > floor,
            "{dtype:?}: cosine {cos} to the input, want > {floor}"
        );
    }
}

/// Prefill, then [`STEPS`] decodes of `tokens` (greedy when `None`).
/// Returns the tokens fed and the logits before each.
fn run(d: &Decoder, tokens: Option<&[usize]>) -> (Vec<usize>, Vec<Vec<f32>>) {
    let prompt = synthetic_llama::PROMPT;
    let mut kv = d.config.new_kv_caches();
    let mut logits = d.forward_batch_last(&prompt, 0, &mut kv);
    let (mut fed, mut rows) = (Vec::new(), Vec::new());
    for step in 0..STEPS {
        let next = tokens.map_or_else(|| common::argmax(&logits), |t| t[step]);
        rows.push(std::mem::take(&mut logits));
        fed.push(next);
        logits = d.forward_token(next, prompt.len() + step, &mut kv);
    }
    rows.push(logits);
    (fed, rows)
}

fn load(path: &Path) -> Decoder {
    let file = ShardedGguf::open(path).expect("open the fixture");
    let config = ModelConfig::from_gguf(&file).expect("model config");
    common::load_decoder(path, config)
}

/// The child half: the CPU's greedy run, printed. Inert unless
/// [`CHILD`] is set.
#[test]
fn prints_cpu_rows_for_the_parent() {
    let Some(path) = std::env::var_os(CHILD) else {
        return;
    };
    common::pin_backend(common::Backend::Cpu);
    let (fed, rows) = run(&load(Path::new(&path)), None);
    let ids: Vec<String> = fed.iter().map(|t| t.to_string()).collect();
    println!("FRINK-ROWS tokens {}", ids.join(" "));
    for row in rows {
        let vals: Vec<String> = row.iter().map(|v| format!("{v:e}")).collect();
        println!("FRINK-ROWS logits {}", vals.join(" "));
    }
}

#[test]
#[ignore = "needs a real Metal-capable GPU; run with --features metal -- --ignored on Apple Silicon"]
fn metal_decode_of_q2_k_and_q3_k_matches_cpu() {
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    metal::run_both();
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;
    use frink_core::kernel_registry::{self, op, Backend as KBackend, Outcome};
    use frink_core::weight_matrix::QuantKind;

    /// The line |Metal - CPU| may reach on any logit at any step.
    ///
    /// The Metal prefill GEMM stages both operands as `half` (as
    /// llama.cpp's `mul_mm` does) and the Metal KV store is f16 where
    /// the CPU's is f32, so the two engines differ by rounding. Measured
    /// on an M2 Pro (2026-10-29): worst 1.46e-3 (Q2_K) and 1.33e-3
    /// (Q3_K) over the prefill and all steps. 1.5e-2 is ten times that;
    /// a wrong min nibble in the Q2_K GEMM functor measured 0.63 at the
    /// first row and a wrong high-bit mask in the Q3_K matvec 2.2.
    const TOL: f32 = 1.5e-2;

    fn cpu_rows(path: &Path) -> (Vec<usize>, Vec<Vec<f32>>) {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args(["--exact", "--nocapture", "prints_cpu_rows_for_the_parent"])
            .env(CHILD, path)
            .env("FRINK_GOLDEN_BACKEND", "cpu")
            .env("FRINK_METAL", "0")
            .env("FRINK_METAL_ATTN", "0")
            .output()
            .expect("spawn the CPU child");
        assert!(
            out.status.success(),
            "CPU child failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let mut tokens = Vec::new();
        let mut rows = Vec::new();
        for line in stdout.lines() {
            if let Some(t) = line.strip_prefix("FRINK-ROWS tokens ") {
                tokens = t.split_whitespace().map(|w| w.parse().unwrap()).collect();
            } else if let Some(l) = line.strip_prefix("FRINK-ROWS logits ") {
                rows.push(l.split_whitespace().map(|w| w.parse().unwrap()).collect());
            }
        }
        assert_eq!(tokens.len(), STEPS, "CPU child tokens:\n{stdout}");
        assert_eq!(rows.len(), STEPS + 1, "CPU child rows");
        (tokens, rows)
    }

    /// The registry rows for `kind` on Metal: a hit for both the decode
    /// matvec and the prefill GEMM, and no miss.
    fn assert_ran_on_metal(kind: QuantKind) {
        let entries = kernel_registry::global().entries();
        let mine: Vec<_> = entries
            .iter()
            .filter(|e| e.key.backend == KBackend::Metal && e.key.kind == Some(kind))
            .collect();
        for want in [op::MATVEC, op::GEMM_PREFILL] {
            assert!(
                mine.iter()
                    .any(|e| e.key.op == want && matches!(e.outcome, Outcome::Hit)),
                "{kind:?}: no Metal {want} hit in the registry: {mine:#?}"
            );
        }
        let misses: Vec<_> = mine.iter().filter(|e| e.outcome.is_miss()).collect();
        assert!(misses.is_empty(), "{kind:?} left Metal: {misses:#?}");
    }

    pub fn run_both() {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        common::pin_backend(common::Backend::Metal);
        frink_models::set_metal_greedy_argmax(false);
        for (dtype, kind) in [
            (GgmlType::Q2K, QuantKind::Q2K),
            (GgmlType::Q3K, QuantKind::Q3K),
        ] {
            let path = synthetic_llama::write("metal_low_k_decode", dtype, encode);
            let (tokens, cpu) = cpu_rows(&path);
            let before = frink_metal::metal_encode_stats().dispatches;
            let (_, gpu) = run(&load(&path), Some(&tokens));
            let dispatched = frink_metal::metal_encode_stats().dispatches - before;
            std::fs::remove_file(&path).ok();
            assert!(dispatched > 0, "{dtype:?}: nothing was dispatched on Metal");
            assert_ran_on_metal(kind);

            let mut distinct = tokens.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert!(
                distinct.len() >= 3,
                "{dtype:?}: the continuation {tokens:?} is too degenerate to compare engines on"
            );
            let mut worst = 0f32;
            for (step, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
                let delta = common::worst_vs(g, c);
                assert!(
                    delta <= TOL,
                    "{dtype:?} step {step}: max |metal - cpu| = {delta} > {TOL}"
                );
                worst = worst.max(delta);
                let (picked, want) = (common::argmax(g), common::argmax(c));
                let lead = c[want] - c[picked];
                assert!(
                    picked == want || lead <= 2.0 * TOL,
                    "{dtype:?} step {step}: Metal picks {picked}, the CPU {want} by {lead}"
                );
            }
            eprintln!(
                "metal-low-k {dtype:?}: {dispatched} dispatches, worst |metal - cpu| = {worst:.3e} \
                 over the prefill and {STEPS} steps"
            );
        }
    }
}
