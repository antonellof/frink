//! A quantized token embedding is gathered on Metal, and the row it
//! gathers is the CPU's (#405).
//!
//! Metal's `get_rows` used to cover Q4_K and Q6_K only, so a model whose
//! `token_embd` was any other kind dequantized its row on the host every
//! token. This suite writes the synthetic llama (`common::synthetic_llama`)
//! with every matrix in a kind, once with an F32 embedding and once with
//! the embedding in the same kind, and decodes both on Metal:
//!
//! - the quantized-embedding decode must agree with a CPU-pinned child
//!   process's logits, step for step, on the CPU's greedy tokens; and
//! - it must encode exactly one more Metal dispatch per decode step than
//!   the F32-embedding decode. The F32 row has no gather and is copied
//!   in from the host, so that one dispatch is the gather. A kind whose
//!   gather fell to the host would encode the same count as F32.
//!
//! `cargo test -p frink-models --features metal --test metal_embd_gather -- --ignored --nocapture`

mod common;

use frink_gguf::{GgmlType, ShardedGguf};
use frink_models::{Decoder, ModelConfig};
use std::path::Path;

use common::synthetic_llama;

/// Decode steps after the prefill.
const STEPS: usize = 8;

/// Set on the child: the fixture to print rows for.
const CHILD: &str = "FRINK_EMBD_GATHER_CHILD";

/// `frink_quant`'s row encoders for the two kinds under test: Q8_0 (a
/// 32-element kind) and Q5_K (a 256-element one), neither of which the
/// gather covered before.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
fn encode(dtype: GgmlType, cols: usize, w: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for row in w.chunks_exact(cols) {
        match dtype {
            GgmlType::Q8_0 => frink_quant::encode_row_q8_0(row, &mut out),
            GgmlType::Q5K => frink_quant::encode_row_q5_k(row, None, &mut out),
            other => panic!("this suite encodes Q8_0 and Q5_K, not {other:?}"),
        }
        .expect("whole blocks");
    }
    out
}

/// Prefill, then [`STEPS`] decodes of `tokens` (greedy when `None`).
fn run(d: &Decoder, tokens: Option<&[usize]>) -> (Vec<usize>, Vec<Vec<f32>>) {
    let prompt = synthetic_llama::PROMPT;
    let mut kv = d.config.new_kv_caches();
    let mut logits = d.forward_batch_last(&prompt, 0, &mut kv);
    let (mut fed, mut rows) = (Vec::new(), Vec::new());
    for step in 0..STEPS {
        let next = tokens.map_or_else(|| common::argmax(&logits), |t| t[step]);
        fed.push(next);
        logits = d.forward_token(next, prompt.len() + step, &mut kv);
        rows.push(logits.clone());
    }
    (fed, rows)
}

fn load(path: &Path) -> Decoder {
    let file = ShardedGguf::open(path).expect("open the fixture");
    let config = ModelConfig::from_gguf(&file).expect("model config");
    common::load_decoder(path, config)
}

/// The child half: the CPU's greedy run, printed. Inert unless [`CHILD`]
/// is set.
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
fn a_quantized_embedding_is_gathered_on_metal_and_matches_cpu() {
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    metal::run_both();
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;

    /// |Metal - CPU| on any logit at any decode step. The gather itself
    /// is bit-exact against the CPU dequant
    /// (`frink_metal::embd::tests::every_gather_matches_the_cpu_dequant`);
    /// what is left is the Metal KV store's f16 against the CPU's f32
    /// and accumulation order, the same budget as
    /// `metal_low_k_quant_decode`.
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
        assert_eq!(rows.len(), STEPS, "CPU child rows");
        (tokens, rows)
    }

    /// Metal dispatches encoded by the [`STEPS`] decodes of `tokens`
    /// (the prefill is run first and not counted), and their logits.
    fn decode_dispatches(d: &Decoder, tokens: &[usize]) -> (u64, Vec<Vec<f32>>) {
        let prompt = synthetic_llama::PROMPT;
        let mut kv = d.config.new_kv_caches();
        d.forward_batch_last(&prompt, 0, &mut kv);
        let before = frink_metal::metal_encode_stats().dispatches;
        let rows: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(step, &t)| d.forward_token(t, prompt.len() + step, &mut kv))
            .collect();
        (frink_metal::metal_encode_stats().dispatches - before, rows)
    }

    pub fn run_both() {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        common::pin_backend(common::Backend::Metal);
        frink_models::set_metal_greedy_argmax(false);
        for dtype in [GgmlType::Q8_0, GgmlType::Q5K] {
            let quant = synthetic_llama::write_with_embd("metal_embd_gather", dtype, dtype, encode);
            let f32_embd =
                synthetic_llama::write_with_embd("metal_embd_gather", dtype, GgmlType::F32, encode);
            let (tokens, cpu) = cpu_rows(&quant);
            let (n_quant, gpu) = decode_dispatches(&load(&quant), &tokens);
            let (n_f32, _) = decode_dispatches(&load(&f32_embd), &tokens);
            std::fs::remove_file(&quant).ok();
            std::fs::remove_file(&f32_embd).ok();

            let mut worst = 0f32;
            for (step, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
                let delta = common::worst_vs(g, c);
                assert!(
                    delta <= TOL,
                    "{dtype:?} step {step}: max |metal - cpu| = {delta} > {TOL}"
                );
                worst = worst.max(delta);
            }
            eprintln!(
                "metal-embd-gather {dtype:?}: decode dispatches {n_quant} (F32 embedding \
                 {n_f32}), worst |metal - cpu| = {worst:.3e} over {STEPS} steps"
            );
            assert!(n_f32 > 0, "{dtype:?}: the decode never reached Metal");
            assert_eq!(
                n_quant,
                n_f32 + STEPS as u64,
                "{dtype:?}: a {dtype:?} embedding should add one gather dispatch per \
                 decode step over the F32 embedding's host copy"
            );
        }
    }
}
