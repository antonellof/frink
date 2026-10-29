//! Q2_K, Q3_K and IQ1/IQ2/IQ3 models decode on Metal as they do on the
//! CPU (#405).
//!
//! The committed graph fixtures are all F32, so the Metal decode sweep
//! in `decode_step_graphs` never reaches a quantized kernel. This suite
//! writes the 512-wide synthetic llama (`common::synthetic_llama`) with
//! every weight matrix in Q2_K, then in Q3_K, then in each of IQ2_XXS,
//! IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S and IQ1_M, and runs the serving
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
//! `frink_quant` has no Q2_K, Q3_K or IQ encoder (the formats are
//! load-only there), so the fixture's encoders are in
//! `common::quant_encode` (a plain min/max quantizer per 16-weight
//! sub-block) and `common::iq_encode` (each group snapped to a nearby
//! grid row). They only have to be a valid
//! packing of SOME weights; `the_fixture_encoders_round_trip_through_the_cpu_dequant`
//! holds them to `frink_quant`'s dequant, which is what defines the
//! model both engines run.

mod common;

use frink_gguf::{GgmlType, ShardedGguf};
use frink_models::{Decoder, ModelConfig};
use std::path::Path;

use common::synthetic_llama;

/// Decode steps after the prefill.
const STEPS: usize = 16;

/// Set on the child: the fixture to print rows for.
const CHILD: &str = "FRINK_LOW_K_DECODE_CHILD";

fn encode(dtype: GgmlType, cols: usize, w: &[f32]) -> Vec<u8> {
    match dtype {
        GgmlType::Q2K | GgmlType::Q3K => common::quant_encode::encode(dtype, cols, w),
        iq if common::iq_encode::KINDS.contains(&iq) => common::quant_encode::encode(iq, cols, w),
        other => panic!("this suite encodes Q2_K, Q3_K and the IQ kinds, not {other:?}"),
    }
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
    // The IQ floors are the measured cosines less about 0.015 (M2 Pro,
    // 2026-10-29: IQ2 0.933..0.946, IQ3 0.978 and 0.986, IQ1 0.908 and
    // 0.925); a wrong field offset measures near 0.
    let iq = common::iq_encode::KINDS.map(|k| {
        let floor = match k {
            GgmlType::IQ1S | GgmlType::IQ1M => 0.89,
            GgmlType::IQ3XXS | GgmlType::IQ3S => 0.96,
            _ => 0.92,
        };
        (k, common::iq_encode::dequant(k), floor)
    });
    for (dtype, dequant, floor) in [
        (GgmlType::Q2K, frink_quant::dequant_q2_k as Dequant, 0.90f64),
        (GgmlType::Q3K, frink_quant::dequant_q3_k as Dequant, 0.97),
    ]
    .into_iter()
    .chain(iq)
    {
        let back = dequant(&encode(dtype, 1024, &v)).expect("whole blocks");
        let dot: f64 = v.iter().zip(&back).map(|(a, b)| f64::from(a * b)).sum();
        let nv: f64 = v.iter().map(|a| f64::from(a * a)).sum::<f64>().sqrt();
        let nb: f64 = back.iter().map(|a| f64::from(a * a)).sum::<f64>().sqrt();
        let cos = dot / (nv * nb);
        eprintln!("{dtype:?}: cosine {cos:.4} to the input (floor {floor})");
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
    metal::run_kinds(&[GgmlType::Q2K, GgmlType::Q3K]);
}

#[test]
#[ignore = "needs a real Metal-capable GPU; run with --features metal -- --ignored on Apple Silicon"]
fn metal_decode_of_the_iq_kinds_matches_cpu() {
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    metal::run_kinds(&common::iq_encode::KINDS);
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

    pub fn run_kinds(kinds: &[GgmlType]) {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        common::pin_backend(common::Backend::Metal);
        frink_models::set_metal_greedy_argmax(false);
        for &dtype in kinds {
            let kind = frink_core::weight_matrix::quant_kind_for(dtype).expect("a quant kind");
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
