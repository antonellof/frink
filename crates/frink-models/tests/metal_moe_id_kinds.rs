//! A MoE model with Q2_K, Q3_K, IQ4_NL, MXFP4, Q5_K, Q6_K or IQ4_XS
//! experts runs its experts through Metal's fused id kernels, and
//! answers as the CPU does (#405, #448). A pack in a kind with no id
//! kernel falls back to one matvec per expert instead of failing.
//!
//! The fused MoE kernels (`mul_mv_id` for decode and short prefills,
//! `mul_mm_id` once an expert gathers eight tokens) had entries for
//! Q4_0, Q5_0, Q4_K and Q8_0 only, and the loader packed no plane of
//! these four kinds. A plain-SwiGLU MoE with such experts therefore ran
//! one matvec per routed expert, three launches each, on every layer:
//! correct, and not the path llama.cpp takes.
//!
//! This suite writes the synthetic llama as a Mixtral-shaped MoE
//! (`common::synthetic_llama::write_moe`, every matrix and expert in
//! the kind) and holds the Metal run to three things:
//!
//! - every layer's experts were packed into the id planes;
//! - prefill and decode encode exactly the dispatches the same model
//!   with Q8_0 experts does (Q8_0 has had both id kernels throughout),
//!   and FEWER than the same model with its planes dropped, which is
//!   the one-matvec-per-expert path: the counter can tell the two
//!   apart, so equality with Q8_0 is the id path, not an assumption;
//! - the logits agree with a CPU-pinned child process on the CPU's
//!   greedy tokens.
//!
//! The prefill runs the fused prefill stack, whose routed experts are
//! the indexed GEMM `mul_mm_id` (f16 activations) at any prompt length;
//! decode runs the fused MoE decode stack, whose experts are
//! `mul_mv_id`. The f32-activation `mul_mm_id` the host-routed prefill
//! takes past eight tokens an expert is held to the CPU in
//! `frink_metal::moe_mv_id::tests`.
//!
//! `cargo test -p frink-models --features metal --test metal_moe_id_kinds -- --ignored --nocapture`

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
const CHILD: &str = "FRINK_MOE_ID_KINDS_CHILD";

/// The kinds under test.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
const KINDS: [GgmlType; 7] = [
    GgmlType::Q2K,
    GgmlType::Q3K,
    GgmlType::IQ4NL,
    GgmlType::MXFP4,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::IQ4XS,
];

/// The IQ4_NL, IQ4_XS and MXFP4 fixture encoders pack the weights
/// `frink_quant` decodes back (`metal_low_k_quant_decode` holds the Q2_K and Q3_K
/// ones to the same).
#[test]
fn the_codebook_encoders_round_trip_through_the_cpu_dequant() {
    let v: Vec<f32> = (0..4096)
        .map(|i| ((i as f32) * 0.37).sin() * 0.05)
        .collect();
    type Dequant = fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError>;
    for (dtype, dequant) in [
        (GgmlType::IQ4NL, frink_quant::dequant_iq4_nl as Dequant),
        (GgmlType::MXFP4, frink_quant::dequant_mxfp4_gguf as Dequant),
        (GgmlType::IQ4XS, frink_quant::dequant_iq4_xs as Dequant),
    ] {
        let back = dequant(&quant_encode::encode(dtype, 1024, &v)).expect("whole blocks");
        let dot: f64 = v.iter().zip(&back).map(|(a, b)| f64::from(a * b)).sum();
        let nv: f64 = v.iter().map(|a| f64::from(a * a)).sum::<f64>().sqrt();
        let nb: f64 = back.iter().map(|a| f64::from(a * a)).sum::<f64>().sqrt();
        let cos = dot / (nv * nb);
        assert!(cos > 0.97, "{dtype:?}: cosine {cos} to the input");
    }
}

/// Prefill, then [`STEPS`] decodes of `tokens` (greedy when `None`).
/// Returns the tokens fed and the logits after the prefill and after
/// each decode.
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

fn load(path: &Path) -> Decoder {
    let file = ShardedGguf::open(path).expect("open the fixture");
    let config = ModelConfig::from_gguf(&file).expect("model config");
    assert_eq!(config.moe.n_experts, synthetic_llama::N_EXPERT, "a MoE");
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
fn moe_experts_of_each_kind_run_the_id_kernels_and_match_cpu() {
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    metal::run_all();
}

/// A layer packed in a kind with no id kernel (#448). No loader builds
/// one, so the Q8_0 fixture's planes are relabelled Q4_1, which has no
/// Metal kernel of any kind. The fused decode stack used to encode it
/// and fail with `CommandFailed`; it must fence it out as the prefill
/// does, so the run is the one-matvec-per-expert run, dispatch for
/// dispatch, and matches the CPU.
#[test]
#[ignore = "needs a real Metal-capable GPU; run with --features metal -- --ignored on Apple Silicon"]
fn a_pack_with_no_id_kernel_falls_back_on_decode() {
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    metal::run_fallback();
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;

    /// |Metal - CPU| on any logit at any step. The prefill GEMMs and
    /// `mul_mm_id` stage both operands as `half` and the Metal KV store
    /// is f16, as in `metal_low_k_quant_decode`, whose measured worst
    /// this is ten times.
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

    /// Dispatches the prefill and the decodes of `tokens` encode, and
    /// the logits.
    fn counted(d: &Decoder, tokens: &[usize]) -> ((u64, u64), Vec<Vec<f32>>) {
        let stats = || frink_metal::metal_encode_stats().dispatches;
        let mut kv = d.config.new_kv_caches();
        let at = stats();
        let mut rows = vec![d.forward_batch_last(&PROMPT, 0, &mut kv)];
        let prefill = stats() - at;
        let at = stats();
        for (step, &t) in tokens.iter().enumerate() {
            rows.push(d.forward_token(t, PROMPT.len() + step, &mut kv));
        }
        ((prefill, stats() - at), rows)
    }

    /// Every layer's experts sit in id planes of `kind`.
    fn assert_packed(d: &Decoder, kind: &str) {
        for (l, layer) in d.layers.iter().enumerate() {
            let planes = layer.moe.packed_q4.as_ref().unwrap_or_else(|| {
                panic!("{kind}: layer {l}'s experts were not packed for the id kernels")
            });
            let v = planes.view();
            assert_eq!(
                (v.gate_kind, v.up_kind, v.down_kind),
                (kind, kind, kind),
                "layer {l}"
            );
        }
    }

    pub fn run_all() {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        common::pin_backend(common::Backend::Metal);
        frink_models::set_metal_greedy_argmax(false);

        let baseline_path =
            synthetic_llama::write_moe("metal_moe_id", GgmlType::Q8_0, quant_encode::encode);
        let baseline = load(&baseline_path);
        assert_packed(&baseline, "Q8_0");

        // Every decoder lives to the end, so no two kinds' planes share
        // an address and each count is its own model's.
        let mut keep = Vec::new();
        for dtype in KINDS {
            let path = synthetic_llama::write_moe("metal_moe_id", dtype, quant_encode::encode);
            let kind = match dtype {
                GgmlType::Q2K => "Q2_K",
                GgmlType::Q3K => "Q3_K",
                GgmlType::IQ4NL => "IQ4_NL",
                GgmlType::MXFP4 => "MXFP4",
                GgmlType::Q5K => "Q5_K",
                GgmlType::Q6K => "Q6_K",
                GgmlType::IQ4XS => "IQ4_XS",
                other => panic!("{other:?}"),
            };
            let (tokens, cpu) = cpu_rows(&path);
            let d = load(&path);
            assert_packed(&d, kind);
            let ((prefill, decode), gpu) = counted(&d, &tokens);
            let ((q8_prefill, q8_decode), _) = counted(&baseline, &tokens);

            // The same model with its planes dropped: one matvec per
            // routed expert.
            let mut unpacked = load(&path);
            for layer in &mut unpacked.layers {
                layer.moe.packed_q4 = None;
            }
            let ((loop_prefill, loop_decode), _) = counted(&unpacked, &tokens);
            keep.push((d, unpacked, path));

            let mut worst = 0f32;
            for (step, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
                let delta = common::worst_vs(g, c);
                assert!(
                    delta <= TOL,
                    "{kind} step {step}: max |metal - cpu| = {delta} > {TOL}"
                );
                worst = worst.max(delta);
            }
            eprintln!(
                "metal-moe-id {kind}: prefill {prefill} dispatches (Q8_0 {q8_prefill}, \
                 per-expert {loop_prefill}), decode {decode} (Q8_0 {q8_decode}, per-expert \
                 {loop_decode}), worst |metal - cpu| = {worst:.3e}"
            );
            assert!(
                decode < loop_decode && prefill < loop_prefill,
                "{kind}: the counter cannot tell the id path from the per-expert loop"
            );
            assert_eq!(
                (prefill, decode),
                (q8_prefill, q8_decode),
                "{kind}: the experts did not take the id kernels Q8_0 takes"
            );
        }
        for (_, _, path) in &keep {
            std::fs::remove_file(path).ok();
        }
        std::fs::remove_file(&baseline_path).ok();
    }

    pub fn run_fallback() {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        common::pin_backend(common::Backend::Metal);
        frink_models::set_metal_greedy_argmax(false);
        let path =
            synthetic_llama::write_moe("metal_moe_no_id", GgmlType::Q8_0, quant_encode::encode);
        let (tokens, cpu) = cpu_rows(&path);

        let mut relabelled = load(&path);
        for layer in &mut relabelled.layers {
            layer.moe.packed_q4 = layer.moe.packed_q4.take().map(|p| p.relabelled("Q4_1"));
        }
        assert_packed(&relabelled, "Q4_1");
        let ((prefill, decode), gpu) = counted(&relabelled, &tokens);

        let mut unpacked = load(&path);
        for layer in &mut unpacked.layers {
            layer.moe.packed_q4 = None;
        }
        let ((loop_prefill, loop_decode), _) = counted(&unpacked, &tokens);
        let ((id_prefill, id_decode), _) = counted(&load(&path), &tokens);

        let mut worst = 0f32;
        for (step, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            let delta = common::worst_vs(g, c);
            assert!(
                delta <= TOL,
                "relabelled Q4_1 step {step}: max |metal - cpu| = {delta} > {TOL}"
            );
            worst = worst.max(delta);
        }
        eprintln!(
            "metal-moe-no-id: prefill {prefill} (per-expert {loop_prefill}, id {id_prefill}), \
             decode {decode} (per-expert {loop_decode}, id {id_decode}), \
             worst |metal - cpu| = {worst:.3e}"
        );
        assert_ne!(
            (loop_prefill, loop_decode),
            (id_prefill, id_decode),
            "the counter cannot tell the per-expert path from the id path"
        );
        assert_eq!(
            (prefill, decode),
            (loop_prefill, loop_decode),
            "a pack with no id kernel did not take the per-expert path"
        );
        drop((relabelled, unpacked));
        std::fs::remove_file(&path).ok();
    }
}
