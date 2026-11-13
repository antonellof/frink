//! gpt-oss-20b on Metal's fused stacks against the CPU, on the real
//! checkpoint (#627).
//!
//! The MoE decode stack and the prefill stack serve gpt-oss with every
//! piece of its graph on the device: attention sinks in the attention
//! kernels, the sliding window on every other layer (in the prefill
//! kernels too), the Q/K/V/O biases, the router bias, the per-expert
//! biases and the clamped SwiGLU. This suite runs the checkpoint twice,
//! in a CPU-pinned child and in this Metal process, on prompts that
//! include one longer than the 128-key window, and holds the Metal run
//! to the CPU's:
//!
//! - every step's argmax is the CPU's (the Metal run is fed the CPU's
//!   greedy tokens, so one near-tie cannot cascade into a different
//!   continuation and hide every later step), except where the CPU's
//!   own top two are within [`NEAR_TIE`] and Metal picks the second:
//!   the primes prompt's first token is one, 0.018 apart on the CPU,
//!   which f16 activations decide either way;
//! - the logits of the CPU's top candidates agree within [`TOL`];
//! - the Metal run took the stacks: a decode token is ONE command
//!   buffer's worth of work, a handful of dispatches per layer, not a
//!   per-projection submission each (the counter is the evidence).
//!
//! Skips without the file: `FRINK_TEST_MODELS_DIR` or `models/` must
//! hold `gpt-oss-20b-MXFP4.gguf`. The same checks run on every Metal
//! host against a synthetic gpt-oss (`common::synthetic_gptoss`), whose
//! window of 8 keys a 21-token prompt crosses in both prefill chunks
//! and in decode, at a tolerance the real file's MoE near-ties do not
//! allow.
//!
//! `cargo test --release -p frink-models --features metal --test gptoss_metal_stack -- --ignored --nocapture`

mod common;

use frink_gguf::ShardedGguf;
use frink_models::{Decoder, ModelConfig};
use std::path::PathBuf;

const FILE: &str = "gpt-oss-20b-MXFP4.gguf";
const CHILD: &str = "FRINK_GPTOSS_STACK_CHILD";
/// Set on the synthetic run's child: the fixture to print rows for.
const SYNTH_CHILD: &str = "FRINK_GPTOSS_SYNTH_CHILD";

/// The synthetic prompt, prefilled as two chunks (12 then 9), so the
/// second chunk attends over a prefix as well as itself, and both cross
/// the window.
const SYNTH_PROMPT: [usize; 21] = [
    3, 141, 59, 265, 358, 97, 9, 323, 84, 62, 433, 17, 250, 5, 77, 401, 8, 199, 300, 41, 2,
];
const SYNTH_SPLIT: usize = 12;
const SYNTH_STEPS: usize = 12;

/// Prefill `SYNTH_PROMPT` in its two chunks, then decode `tokens` (the
/// greedy pick when `None`). The rows after each chunk and each step.
fn synth_run(d: &Decoder, tokens: Option<&[usize]>) -> (Vec<usize>, Vec<Vec<f32>>) {
    let mut kv = d.config.new_kv_caches();
    let mut rows = vec![d.forward_batch_last(&SYNTH_PROMPT[..SYNTH_SPLIT], 0, &mut kv)];
    let mut logits = d.forward_batch_last(&SYNTH_PROMPT[SYNTH_SPLIT..], SYNTH_SPLIT, &mut kv);
    let mut fed = Vec::new();
    for step in 0..SYNTH_STEPS {
        let next = tokens.map_or_else(|| common::argmax(&logits), |t| t[step]);
        rows.push(std::mem::take(&mut logits));
        fed.push(next);
        logits = d.forward_token(next, SYNTH_PROMPT.len() + step, &mut kv);
    }
    rows.push(logits);
    (fed, rows)
}

fn synth_load(path: &std::path::Path) -> Decoder {
    let file = ShardedGguf::open(path).expect("open the fixture");
    let config = ModelConfig::from_gguf(&file).expect("config");
    common::load_decoder(path, config)
}

/// The synthetic run's child half. Inert unless [`SYNTH_CHILD`] is set.
#[test]
fn prints_synthetic_cpu_rows_for_the_parent() {
    let Some(path) = std::env::var_os(SYNTH_CHILD) else {
        return;
    };
    common::pin_backend(common::Backend::Cpu);
    let (fed, rows) = synth_run(&synth_load(std::path::Path::new(&path)), None);
    let ids: Vec<String> = fed.iter().map(|t| t.to_string()).collect();
    println!("FRINK-ROWS tokens {}", ids.join(" "));
    for row in rows {
        let vals: Vec<String> = row.iter().map(|v| format!("{v:e}")).collect();
        println!("FRINK-ROWS logits {}", vals.join(" "));
    }
}

#[test]
#[ignore = "needs a real Metal-capable GPU"]
fn a_synthetic_gpt_oss_runs_the_metal_stacks_and_matches_the_cpu() {
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    metal::run_synthetic();
}

/// Greedy steps after each prompt's prefill.
const STEPS: usize = 40;

/// Candidates per step whose logits are compared.
const TOP: usize = 5;

/// Prompts: a one-token one, two short ones, and one past the 128-key
/// window so the sliding layers drop keys during prefill AND decode.
fn prompts() -> Vec<String> {
    vec![
        "Hi".to_string(),
        "List the first twenty prime numbers, then explain why 1 is not prime.".to_string(),
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n".to_string(),
        "The history of the Roman Empire spans many centuries, beginning with the founding \
         of the city of Rome according to legend in 753 BC. Over time, Rome grew from a small \
         settlement on the Tiber River into a republic governed by elected magistrates and a \
         senate. The republic expanded through conquest and alliance across the Italian \
         peninsula, and later throughout the Mediterranean, defeating Carthage in the Punic \
         Wars and absorbing the Hellenistic kingdoms of the east. Internal strife, civil wars, \
         and the ambitions of powerful generals such as Marius, Sulla, Pompey and Julius \
         Caesar eventually brought the republic to an end. Caesar's adopted heir Octavian \
         defeated Mark Antony and Cleopatra at Actium in 31 BC and became Augustus, the first \
         emperor. Summarize the main causes of the fall of the Western Roman Empire in three \
         short points:"
            .to_string(),
    ]
}

fn path() -> Option<PathBuf> {
    let p = common::model_dir().join(FILE);
    p.exists().then_some(p)
}

fn tokens(file: &ShardedGguf, prompt: &str) -> Vec<usize> {
    use frink_models::tokenizer::{GgufBpeTokenizer, SpecialTokens};
    let mut t: Vec<usize> = GgufBpeTokenizer::from_gguf(file)
        .expect("bpe tokenizer")
        .encode(prompt, SpecialTokens::Parse)
        .into_iter()
        .map(|i| i as usize)
        .collect();
    if frink_models::tokenizer::should_add_bos_token(file) {
        if let Some(b) = file.metadata_u64("tokenizer.ggml.bos_token_id") {
            t.insert(0, b as usize);
        }
    }
    t
}

fn load() -> (Decoder, ShardedGguf) {
    let p = path().expect("checked by the caller");
    let file = ShardedGguf::open(&p).expect("open gpt-oss");
    let config = ModelConfig::from_gguf(&file).expect("config");
    (common::load_decoder(&p, config), file)
}

/// The `TOP` largest logits, best first.
fn top(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(TOP, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(TOP);
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.into_iter().map(|i| (i, logits[i])).collect()
}

/// The child half: the CPU's greedy run of every prompt. Inert unless
/// [`CHILD`] is set.
#[test]
fn prints_cpu_rows_for_the_parent() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    common::pin_backend(common::Backend::Cpu);
    let (d, file) = load();
    for (pi, prompt) in prompts().iter().enumerate() {
        let toks = tokens(&file, prompt);
        let mut kv = d.config.new_kv_caches();
        let mut logits = d.forward_batch_last(&toks, 0, &mut kv);
        for step in 0..STEPS {
            let best = top(&logits);
            let row: Vec<String> = best.iter().map(|(i, v)| format!("{i}:{v:e}")).collect();
            println!("FRINK-STEP {pi} {step} {}", row.join(" "));
            logits = d.forward_token(best[0].0, toks.len() + step, &mut kv);
        }
    }
}

#[test]
#[ignore = "needs Apple Metal and models/gpt-oss-20b-MXFP4.gguf"]
fn gpt_oss_runs_the_metal_stacks_and_matches_the_cpu() {
    if path().is_none() {
        eprintln!("skip: no {FILE} under {}", common::model_dir().display());
        return;
    }
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    metal::run();
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;
    use frink_gguf::GgmlType;

    /// |Metal - CPU| on any logit of the synthetic model: the f16
    /// staging of the prefill GEMMs and the f16 KV store, against f32.
    const SYNTH_TOL: f32 = 2e-2;

    fn synth_cpu(path: &std::path::Path) -> (Vec<usize>, Vec<Vec<f32>>) {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args([
                "--exact",
                "--nocapture",
                "prints_synthetic_cpu_rows_for_the_parent",
            ])
            .env(SYNTH_CHILD, path)
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
        let (mut tokens, mut rows) = (Vec::new(), Vec::new());
        for line in stdout.lines() {
            if let Some(t) = line.strip_prefix("FRINK-ROWS tokens ") {
                tokens = t.split_whitespace().map(|w| w.parse().unwrap()).collect();
            } else if let Some(l) = line.strip_prefix("FRINK-ROWS logits ") {
                rows.push(l.split_whitespace().map(|w| w.parse().unwrap()).collect());
            }
        }
        assert_eq!(tokens.len(), SYNTH_STEPS, "CPU child:\n{stdout}");
        (tokens, rows)
    }

    pub fn run_synthetic() {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        common::pin_backend(common::Backend::Metal);
        frink_models::set_metal_greedy_argmax(false);
        let path = common::synthetic_gptoss::write(
            "gptoss_metal_stack",
            GgmlType::MXFP4,
            common::quant_encode::encode,
        );
        let (tokens, cpu) = synth_cpu(&path);
        let d = synth_load(&path);
        assert!(d.gpt_oss.is_some(), "the fixture loads as gpt-oss");
        for (l, layer) in d.layers.iter().enumerate() {
            let v = layer.moe.packed_q4.as_ref().expect("experts packed").view();
            assert_eq!(v.gate_kind, "MXFP4", "layer {l}");
        }
        let stats = || frink_metal::metal_encode_stats().dispatches;
        let at = stats();
        let (_, gpu) = synth_run(&d, Some(&tokens));
        let total = stats() - at;
        assert_eq!(gpu.len(), cpu.len());
        let mut worst = 0f32;
        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            let delta = common::worst_vs(g, c);
            worst = worst.max(delta);
            assert!(
                delta <= SYNTH_TOL,
                "row {i}: max |metal - cpu| {delta} > {SYNTH_TOL}"
            );
            // Row 0 ends the first chunk; rows 1..=STEPS each chose a fed token.
            if (1..=SYNTH_STEPS).contains(&i) {
                assert_eq!(common::argmax(g), tokens[i - 1], "row {i}: greedy pick");
            }
        }
        // One decode token alone: the MoE stack's band (see `run`).
        // And the prefill: the stack's ~25 a layer for the whole batch,
        // where the host body would encode none.
        let mut kv = d.config.new_kv_caches();
        let at = stats();
        d.forward_batch_last(&SYNTH_PROMPT, 0, &mut kv);
        let prefill = (stats() - at) as usize;
        let at = stats();
        d.forward_token(tokens[0], SYNTH_PROMPT.len(), &mut kv);
        let one = (stats() - at) as usize;
        eprintln!(
            "synthetic gpt-oss: worst |metal - cpu| {worst:.3e}, {total} dispatches for the run, \
             {prefill} for a {}-token prefill, {one} for one decode token",
            SYNTH_PROMPT.len()
        );
        assert!(
            (15 * N_LAYERS_SYNTH..=40 * N_LAYERS_SYNTH).contains(&one),
            "{one} dispatches a decode token is not the MoE stack"
        );
        assert!(
            (15 * N_LAYERS_SYNTH..=60 * N_LAYERS_SYNTH).contains(&prefill),
            "{prefill} dispatches for the prefill is not the prefill stack"
        );
        std::fs::remove_file(&path).ok();
    }

    const N_LAYERS_SYNTH: usize = common::synthetic_gptoss::N_LAYER;

    /// |Metal - CPU| on a top candidate's logit. The prefill GEMMs stage
    /// activations as `half` and the KV store is f16, against an f32
    /// host; gpt-oss's logits run to about 30.
    ///
    /// Measured on an M5 Max: 0.002 over 40 decode steps when the prompt
    /// is one token (the decode stack alone), and 0.85 at worst after a
    /// 16-token prefill, at a step where all five candidates moved
    /// together by 0.71 to 0.85 -- one routed expert changing in one
    /// layer, a discontinuity of top-k routing. The same step with the
    /// prompt fed token by token through the decode stack lands within
    /// 0.001, and with the tensor-API GEMMs off (`FRINK_METAL_TENSOR=0`)
    /// the first such jump moves to a later step: f16 activations in the
    /// prefill GEMMs (as llama.cpp's `mul_mm` stages them) nudging a
    /// near-tied router, not a missing term.
    const TOL: f32 = 1.0;

    /// A CPU top-two gap under which f16 activations may order the two
    /// either way. Measured: 0.018 on the primes prompt's first token,
    /// where the Metal logits sat 0.021 from the CPU's.
    const NEAR_TIE: f32 = 0.05;

    type Steps = Vec<Vec<Vec<(usize, f32)>>>;

    fn cpu_steps() -> Steps {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args(["--exact", "--nocapture", "prints_cpu_rows_for_the_parent"])
            .env(CHILD, "1")
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
        let mut steps: Steps = vec![Vec::new(); prompts().len()];
        for line in stdout.lines() {
            let Some(rest) = line.strip_prefix("FRINK-STEP ") else {
                continue;
            };
            let mut w = rest.split_whitespace();
            let pi: usize = w.next().unwrap().parse().unwrap();
            let _step: usize = w.next().unwrap().parse().unwrap();
            steps[pi].push(
                w.map(|c| {
                    let (i, v) = c.split_once(':').unwrap();
                    (i.parse().unwrap(), v.parse().unwrap())
                })
                .collect(),
            );
        }
        for s in &steps {
            assert_eq!(s.len(), STEPS, "CPU child steps:\n{stdout}");
        }
        steps
    }

    pub fn run() {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        common::pin_backend(common::Backend::Metal);
        frink_models::set_metal_greedy_argmax(false);
        let cpu = cpu_steps();
        let (d, file) = load();
        let n_layers = d.layers.len();
        let stats = || frink_metal::metal_encode_stats().dispatches;
        for (pi, prompt) in prompts().iter().enumerate() {
            let mut worst = 0f32;
            let toks = tokens(&file, prompt);
            let mut kv = d.config.new_kv_caches();
            let mut logits = d.forward_batch_last(&toks, 0, &mut kv);
            let mut per_token = Vec::new();
            for (step, want) in cpu[pi].iter().enumerate() {
                let got = common::argmax(&logits);
                let gap = want[0].1 - want[1].1;
                let deltas: Vec<f32> = want.iter().map(|&(i, v)| (logits[i] - v).abs()).collect();
                let delta = deltas.iter().copied().fold(0f32, f32::max);
                worst = worst.max(delta);
                let tie = gap < NEAR_TIE && got == want[1].0;
                if tie {
                    eprintln!(
                        "prompt {pi} step {step}: near-tie, CPU gap {gap:.3}, Metal picks the \
                         CPU's second"
                    );
                }
                assert!(
                    got == want[0].0 || tie,
                    "prompt {pi} ({} tokens) step {step}: Metal picks {got}, the CPU {} \
                     (CPU top-2 gap {gap:.3}, worst top-{TOP} delta {delta:.3})",
                    toks.len(),
                    want[0].0
                );
                assert!(
                    delta <= TOL,
                    "prompt {pi} step {step}: |metal - cpu| {delta:.3} on a top-{TOP} logit"
                );
                let at = stats();
                logits = d.forward_token(want[0].0, toks.len() + step, &mut kv);
                per_token.push(stats() - at);
            }
            // The stack encodes about twenty dispatches a layer; the
            // per-projection path it replaced was a submission per
            // matvec, and the host body none at all. Either way off the
            // stack is outside this band.
            let max = *per_token.iter().max().unwrap() as usize;
            let min = *per_token.iter().min().unwrap() as usize;
            eprintln!(
                "gpt-oss prompt {pi} ({} tokens): decode {min}..{max} dispatches a token, \
                 worst |metal - cpu| {worst:.3}",
                toks.len()
            );
            assert!(
                min >= 15 * n_layers && max <= 40 * n_layers,
                "prompt {pi}: {min}..{max} dispatches a decode token is not the MoE stack"
            );
        }
    }
}
