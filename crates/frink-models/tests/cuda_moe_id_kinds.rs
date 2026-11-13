//! A MoE model's routed experts run through CUDA's id kernels
//! (`mul_mv_id` for decode and short batches, `mul_mm_id` once an
//! expert gathers eight rows), and answer as the CPU does; a pack in a
//! kind with no id kernel falls back to one fused FFN per expert
//! instead of failing (#405, Metal's #448 rule).
//!
//! The synthetic llama is written as a Mixtral-shaped MoE
//! (`common::synthetic_llama::write_moe`, every matrix and expert in the
//! kind), as in `metal_moe_id_kinds`. The host half runs on any
//! `--features cuda` build: every layer's experts are packed into the
//! planes CUDA's kernels index, for each kind, and a pack relabelled
//! Q8_1 (no CUDA kernel of any kind) is fenced. The device half is
//! `#[ignore]`d and holds a run to three things:
//!
//! - its logits agree with a CPU child process on the CPU's greedy
//!   tokens;
//! - each decode token makes exactly as many host waits as the same
//!   model in Q8_0 (the waits do not depend on the kind), and FEWER
//!   than the same model with its planes dropped, which runs one fused
//!   FFN per routed expert with a download each (`frink_cuda::counters`,
//!   the counted `xfer` door, can tell the two apart);
//! - the relabelled pack makes the per-expert run's waits and matches
//!   the CPU.
//!
//! `cargo test -p frink-models --features cuda --test cuda_moe_id_kinds -- --ignored --nocapture`

#![cfg(feature = "cuda")]

mod common;

use frink_gguf::{GgmlType, ShardedGguf};
use frink_models::{Decoder, ModelConfig};
use std::path::Path;

use common::quant_encode;
use common::synthetic_llama;

/// Decode steps after the prefill.
const STEPS: usize = 6;

/// Eleven distinct prompt tokens.
const PROMPT: [usize; 11] = [3, 141, 59, 265, 358, 97, 9, 323, 84, 62, 433];

/// Set on the child: the fixture to print rows for.
const CHILD: &str = "FRINK_CUDA_MOE_ID_CHILD";

/// The kinds the fixture encoders write, Q8_0 first (the baseline).
const KINDS: [GgmlType; 18] = [
    GgmlType::Q8_0,
    GgmlType::Q2K,
    GgmlType::Q3K,
    GgmlType::IQ4NL,
    GgmlType::MXFP4,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::IQ4XS,
    GgmlType::IQ2XXS,
    GgmlType::IQ2XS,
    GgmlType::IQ2S,
    GgmlType::IQ3XXS,
    GgmlType::IQ3S,
    GgmlType::IQ1S,
    GgmlType::IQ1M,
    GgmlType::Q4_1,
    GgmlType::Q5_1,
    GgmlType::TQ1_0,
];

fn load(path: &Path) -> Decoder {
    let file = ShardedGguf::open(path).expect("open the fixture");
    let config = ModelConfig::from_gguf(&file).expect("model config");
    assert_eq!(config.moe.n_experts, synthetic_llama::N_EXPERT, "a MoE");
    common::load_decoder(path, config)
}

fn kind_name(dtype: GgmlType) -> &'static str {
    frink_core::weight_matrix::quant_kind_for(dtype)
        .unwrap_or_else(|| panic!("{dtype:?}"))
        .name()
}

/// Every layer's experts are packed in `kind` and CUDA's fence admits
/// them (`cuda_view`), and the same planes relabelled Q8_1 are fenced.
/// Needs no device: the fence is the kernel table.
///
/// Sabotage: make `frink_cuda::moe_id::supported` answer `Some` for
/// Q8_1 (or drop the `?` on a role in `cuda_view`) and this goes red.
#[test]
fn every_kind_packs_for_the_cuda_id_kernels_and_q8_1_is_fenced() {
    for dtype in KINDS {
        let kind = kind_name(dtype);
        let path = synthetic_llama::write_moe("cuda_moe_id_host", dtype, quant_encode::encode);
        let mut d = load(&path);
        for (l, layer) in d.layers.iter_mut().enumerate() {
            let planes = layer.moe.packed_q4.take().unwrap_or_else(|| {
                panic!("{kind}: layer {l}'s experts were not packed for the id kernels")
            });
            let view = planes
                .cuda_view()
                .unwrap_or_else(|| panic!("{kind}: layer {l}'s pack was fenced"));
            assert_eq!(view.gate.kind.name, kind, "layer {l}");
            assert_eq!(view.down.rows, view.gate.cols, "layer {l}: down is hidden x ffn");
            assert!(
                planes.relabelled("Q8_1").cuda_view().is_none(),
                "{kind}: a Q8_1 pack was admitted"
            );
        }
        std::fs::remove_file(&path).ok();
    }
}

/// Prefill, then [`STEPS`] decodes of `tokens` (greedy when `None`).
fn run(d: &Decoder, tokens: Option<&[usize]>) -> (Vec<usize>, Vec<Vec<f32>>) {
    let mut kv = d.config.new_kv_caches();
    let mut logits = d.forward_batch_last(&PROMPT, 0, &mut kv);
    let (mut fed, mut rows) = (Vec::new(), Vec::new());
    for step in 0..STEPS {
        let next = tokens.map_or_else(|| common::argmax(&logits), |t| t[step]);
        rows.push(std::mem::take(&mut logits));
        fed.push(next);
        logits = d.forward_token(next, PROMPT.len() + step, &mut kv);
    }
    rows.push(logits);
    (fed, rows)
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
#[ignore = "requires real CUDA hardware; not yet run. The CUDA id kernels have NEVER executed on a GPU"]
fn moe_experts_of_each_kind_run_the_id_kernels_and_match_cpu() {
    device::run_all();
}

#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn a_pack_with_no_id_kernel_falls_back_per_expert() {
    device::run_fallback();
}

mod device {
    use super::*;

    /// |CUDA - CPU| on any logit at any step. The CUDA prefill GEMM is
    /// f32 SIMT or tensor-core f16 operands, and the decode KV mirror is
    /// f32; Metal's f16 bound is the loose end of that.
    const TOL: f32 = 1.5e-2;

    fn cpu_rows(path: &Path) -> (Vec<usize>, Vec<Vec<f32>>) {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args(["--exact", "--nocapture", "prints_cpu_rows_for_the_parent"])
            .env(CHILD, path)
            .env("FRINK_GOLDEN_BACKEND", "cpu")
            .env("FRINK_CUDA", "0")
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
        assert_eq!(tokens.len(), STEPS, "CPU child tokens:\n{stdout}");
        (tokens, rows)
    }

    /// Host waits per decode token, and the logits.
    fn counted(d: &Decoder, tokens: &[usize]) -> (u64, Vec<Vec<f32>>) {
        let mut kv = d.config.new_kv_caches();
        let mut rows = vec![d.forward_batch_last(&PROMPT, 0, &mut kv)];
        let at = frink_cuda::counters::snapshot();
        for (step, &t) in tokens.iter().enumerate() {
            rows.push(d.forward_token(t, PROMPT.len() + step, &mut kv));
        }
        let waits = frink_cuda::counters::snapshot().since(at).syncs;
        (waits / tokens.len() as u64, rows)
    }

    fn check(what: &str, gpu: &[Vec<f32>], cpu: &[Vec<f32>]) -> f32 {
        let mut worst = 0f32;
        for (step, (g, c)) in gpu.iter().zip(cpu).enumerate() {
            let delta = common::worst_vs(g, c);
            assert!(delta <= TOL, "{what} step {step}: max |cuda - cpu| = {delta} > {TOL}");
            worst = worst.max(delta);
        }
        worst
    }

    fn device_present() -> bool {
        let present = frink_cuda::HardwareProfile::detect().cuda_available;
        if !present {
            eprintln!("skip: no CUDA device detected");
        }
        present
    }

    fn unpacked(path: &Path) -> Decoder {
        let mut d = load(path);
        for layer in &mut d.layers {
            layer.moe.packed_q4 = None;
        }
        d
    }

    pub fn run_all() {
        if !device_present() {
            return;
        }
        let mut baseline_waits = None;
        // Every decoder lives to the end, so no two kinds' planes share
        // an address in the resident cache.
        let mut keep = Vec::new();
        for dtype in KINDS {
            let kind = kind_name(dtype);
            let path = synthetic_llama::write_moe("cuda_moe_id", dtype, quant_encode::encode);
            let (tokens, cpu) = cpu_rows(&path);
            let d = load(&path);
            let (waits, gpu) = counted(&d, &tokens);
            let per_expert = unpacked(&path);
            let (loop_waits, _) = counted(&per_expert, &tokens);
            let worst = check(kind, &gpu, &cpu);
            eprintln!(
                "cuda-moe-id {kind}: {waits} waits per decode token (per-expert {loop_waits}), \
                 worst |cuda - cpu| = {worst:.3e}"
            );
            assert!(
                waits < loop_waits,
                "{kind}: the counter cannot tell the id path from the per-expert loop"
            );
            let base = *baseline_waits.get_or_insert(waits);
            assert_eq!(waits, base, "{kind}: not the waits the Q8_0 id path makes");
            keep.push((d, per_expert, path));
        }
        for (_, _, path) in &keep {
            std::fs::remove_file(path).ok();
        }
    }

    pub fn run_fallback() {
        if !device_present() {
            return;
        }
        let path = synthetic_llama::write_moe("cuda_moe_no_id", GgmlType::Q8_0, quant_encode::encode);
        let (tokens, cpu) = cpu_rows(&path);
        let mut relabelled = load(&path);
        for layer in &mut relabelled.layers {
            layer.moe.packed_q4 = layer.moe.packed_q4.take().map(|p| p.relabelled("Q8_1"));
        }
        let (waits, gpu) = counted(&relabelled, &tokens);
        let (loop_waits, _) = counted(&unpacked(&path), &tokens);
        let (id_waits, _) = counted(&load(&path), &tokens);
        let worst = check("relabelled Q8_1", &gpu, &cpu);
        eprintln!(
            "cuda-moe-no-id: {waits} waits per decode token (per-expert {loop_waits}, \
             id {id_waits}), worst |cuda - cpu| = {worst:.3e}"
        );
        assert_eq!(waits, loop_waits, "the fenced pack did not take the per-expert path");
        assert!(id_waits < loop_waits);
        std::fs::remove_file(&path).ok();
    }
}
