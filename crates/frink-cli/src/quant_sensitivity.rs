//! Per-tensor quantization sensitivity, measured on this checkpoint.
//!
//! `frink inspect-plan` prices a checkpoint from static type rules --
//! the same rules llama.cpp's quant mixes encode, which say things like
//! "keep `attn_v` and `ffn_down` a tier higher". Those rules are a good
//! prior and they are not a measurement: they were derived on other
//! models, and no rule knows whether *this* checkpoint's layer 14 cares.
//!
//! So measure it. For one tensor at a time:
//!
//! 1. dequantize it, quantize the result to a candidate format,
//!    dequantize back -- the round trip;
//! 2. score `relative_mse` per block against the values it started
//!    from, which is the weight-space damage;
//! 3. swap the round-tripped weight into the loaded model, run a real
//!    prefill, and measure how far the output distribution moved
//!    (KL, in nats) from the untouched run.
//!
//! **One tensor is perturbed at a time and every other weight stays
//! exactly as the checkpoint shipped it.** That is the point of the
//! design: a sweep that quantized layer by layer and kept going would
//! feed each layer an input already damaged by the layers above it, and
//! by layer 20 every tensor looks sensitive because the residual stream
//! is full of accumulated error. Propagating the clean values forward
//! costs one forward pass per tensor and makes the column mean what it
//! says: this tensor's own contribution.
//!
//! Weight-space `relative_mse` and output-space KL are both printed
//! because they disagree, and the disagreement is the finding. A tensor
//! can round-trip badly and barely move the logits (the model does not
//! use that subspace) or round-trip cleanly and move them a lot (a
//! narrow, load-bearing projection). Only the second column is a reason
//! to spend bits.
//!
//! Runs on CPU by construction: swapping a `WeightMatrix` invalidates
//! the Metal packed-expert planes built at load, and the question --
//! how much does this quantization hurt -- is not a backend question.

use anyhow::Context;
use frink_core::cache::KvCache;
use frink_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
use frink_models::decoder::{Decoder, ExpertBacking};
use half::f16;
use std::path::Path;

/// Fixed so runs are comparable across invocations and models.
const PROMPT: &str = "The capital of France is";

pub struct QuantSensitivityArgs {
    pub model: String,
    pub prompt: Option<String>,
    pub prompt_tokens: Option<usize>,
    /// Format to round-trip through: `q4_0` or `q8_0`.
    pub candidate: String,
    /// `START:END` (end exclusive) to restrict the sweep.
    pub layers: Option<String>,
    /// Routed experts to probe per MoE layer, from index 0.
    pub experts: usize,
    /// How many rows the table prints.
    pub top: usize,
}

/// The formats this tool can quantize *into*.
///
/// Deliberately short. Q8_0 and Q4_0 are the two legacy formats with a
/// simple, exactly-specified rounding rule, they bracket the useful
/// range (near-lossless vs. the cheapest 4-bit), and they are the two
/// kinds with the fastest CPU dot kernels -- so a tensor this tool says
/// is insensitive is a tensor that can be moved to a faster kernel, not
/// just a smaller one. The K-quants' block search is a different
/// algorithm and belongs in a quantizer, not in a diagnostic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Candidate {
    Q4_0,
    Q8_0,
}

impl Candidate {
    fn parse(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "q4_0" | "q4" => Ok(Candidate::Q4_0),
            "q8_0" | "q8" => Ok(Candidate::Q8_0),
            other => anyhow::bail!("--candidate expects q4_0 or q8_0, got `{other}`"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Candidate::Q4_0 => "q4_0",
            Candidate::Q8_0 => "q8_0",
        }
    }

    fn kind(self) -> QuantKind {
        match self {
            Candidate::Q4_0 => QuantKind::Q4_0,
            Candidate::Q8_0 => QuantKind::Q8_0,
        }
    }

    fn block_elems(self) -> usize {
        32
    }

    fn quantize(self, src: &[f32]) -> Vec<u8> {
        match self {
            Candidate::Q4_0 => quantize_q4_0(src),
            Candidate::Q8_0 => frink_quant::quantize_q8_0(src),
        }
    }

    fn dequantize(self, bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
        let out = match self {
            Candidate::Q4_0 => frink_quant::dequant_q4_0(bytes),
            Candidate::Q8_0 => frink_quant::dequant_q8_0(bytes),
        };
        out.map_err(|e| anyhow::anyhow!("{e:?}"))
    }
}

/// Q4_0, ggml's rounding rule.
///
/// The block's scale is taken from the element with the largest
/// magnitude divided by `-8`, not from `amax / 7.5` or any other
/// symmetric choice: the code range is `[-8, 7]`, so anchoring on the
/// negative end is what makes the extreme value representable. Codes
/// are stored `[0..16)` in the low nibbles and `[16..32)` in the high
/// nibbles of the same sixteen bytes, which is the layout
/// `frink_quant::dequant_q4_0` reads back.
fn quantize_q4_0(src: &[f32]) -> Vec<u8> {
    let block = frink_quant::Q4_0_BLOCK_ELEMS;
    let mut out = Vec::with_capacity(src.len().div_ceil(block) * frink_quant::Q4_0_BLOCK_BYTES);
    for chunk in src.chunks(block) {
        let mut amax = 0f32;
        let mut vmax = 0f32;
        for &v in chunk {
            if v.abs() > amax {
                amax = v.abs();
                vmax = v;
            }
        }
        let d = vmax / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for i in 0..block / 2 {
            let lo = code_q4_0(chunk.get(i).copied().unwrap_or(0.0), id);
            let hi = code_q4_0(chunk.get(i + block / 2).copied().unwrap_or(0.0), id);
            out.push(lo | (hi << 4));
        }
    }
    out
}

fn code_q4_0(v: f32, id: f32) -> u8 {
    let x = v * id + 8.5;
    (x as i32).clamp(0, 15) as u8
}

/// What one round trip did to one tensor.
struct Row {
    name: String,
    kind: String,
    /// Mean of the per-block `relative_mse`.
    rel_mse_mean: f64,
    /// 99th percentile block, because a mean over ten thousand blocks
    /// hides the handful that carry an outlier weight.
    rel_mse_p99: f64,
    /// KL(clean || perturbed) over the next-token distribution, nats.
    kl: f64,
    /// Whether the greedy token changed.
    top1_flipped: bool,
}

pub fn run(args: QuantSensitivityArgs) -> anyhow::Result<()> {
    let candidate = Candidate::parse(&args.candidate)?;
    guard_int_dot(std::env::var("FRINK_CPU_INT_DOT").ok().as_deref())?;
    // Force the CPU path before anything reads a weight: the backend
    // switches are process-lifetime `OnceLock`s, so this is the only
    // point at which the choice can still be made.
    // SAFETY: single-threaded, before the model is loaded or any worker
    // thread exists.
    unsafe { frink_models::offload::select(frink_models::offload::OffloadDevice::Cpu)? };

    let prompt = args.prompt.clone().unwrap_or_else(|| PROMPT.to_string());
    let path = crate::pull::resolve_model_path(&args.model)?;
    let (mut decoder, tokens, _eos) = crate::verify_engine::load_and_tokenize(
        Path::new(&path),
        &prompt,
        frink_models::tokenizer::SpecialTokens::Parse,
        args.prompt_tokens,
    )?;

    let n_layers = decoder.layers.len();
    let (from, to) = parse_layer_range(args.layers.as_deref(), n_layers)?;

    let clean = last_logits(&decoder, &tokens);
    let clean_probs = softmax(&clean);
    let clean_top1 = argmax(&clean);

    println!(
        "quant-sensitivity {}: round-trip through {}, {} prompt tokens, layers {from}..{to} of {n_layers}",
        short(&args.model),
        candidate.name(),
        tokens.len(),
    );
    println!("one tensor perturbed at a time; every other weight is the checkpoint's own");

    let mut rows: Vec<Row> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for l in from..to {
        for slot in slots_for(&decoder, l, args.experts) {
            let name = format!("blk.{l}.{}", slot.suffix());
            let Some(target) = slot_mut(&mut decoder, l, slot) else {
                skipped.push(format!("{name} (expert store: bytes are not resident)"));
                continue;
            };
            if !target.cols().is_multiple_of(candidate.block_elems()) {
                skipped.push(format!(
                    "{name} ({} columns is not a multiple of {})",
                    target.cols(),
                    candidate.block_elems()
                ));
                continue;
            }
            let kind = target
                .quant_kind()
                .map(|k| k.name().to_string())
                .unwrap_or_else(|| "f32".to_string());
            let (perturbed, blocks) = round_trip(target, candidate)?;
            let original = std::mem::replace(target, perturbed);

            let logits = last_logits(&decoder, &tokens);
            let kl = kl_nats(&clean_probs, &logits);
            let top1_flipped = argmax(&logits) != clean_top1;

            // Put the checkpoint's own weight back before the next
            // tensor is touched, or the sweep turns into the
            // accumulating one this design exists to avoid.
            if let Some(slot_back) = slot_mut(&mut decoder, l, slot) {
                *slot_back = original;
            }

            rows.push(Row {
                name,
                kind,
                rel_mse_mean: blocks.mean,
                rel_mse_p99: blocks.p99,
                kl,
                top1_flipped,
            });
        }
    }

    if rows.is_empty() {
        anyhow::bail!("no tensor in layers {from}..{to} could be round-tripped; nothing measured");
    }
    print_table(&rows, args.top, candidate);
    for s in &skipped {
        println!("skipped {s}");
    }
    Ok(())
}

/// Refuse to run with the integer dot path on.
///
/// Two reasons, and the first is a correctness trap rather than a
/// preference. The int-dot kernels read a repacked copy of the weight
/// bytes from a process-wide cache keyed by the buffer's *address*, so a
/// freshly allocated round-tripped tensor that lands on an address a
/// previously freed one used would silently be given the old tensor's
/// repacked bytes. Second, int-dot quantizes the activations too, and
/// mixing that error into a measurement of the weight's error is
/// exactly the confound this tool is for.
fn guard_int_dot(value: Option<&str>) -> anyhow::Result<()> {
    if matches!(value, Some("1" | "true" | "on")) {
        anyhow::bail!(
            "FRINK_CPU_INT_DOT is on. quant-sensitivity swaps weight buffers in and out of a \
             loaded model, and the int-dot repack cache is keyed by buffer address, so a reused \
             allocation would be served another tensor's repacked bytes. Unset it (the reference \
             float dot path is also the right one for measuring a weight's own error) and rerun."
        );
    }
    Ok(())
}

/// Which tensors of layer `l` to probe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Q,
    K,
    V,
    O,
    Router,
    Gate(usize),
    Up(usize),
    Down(usize),
}

impl Slot {
    fn suffix(self) -> String {
        match self {
            Slot::Q => "attn_q".into(),
            Slot::K => "attn_k".into(),
            Slot::V => "attn_v".into(),
            Slot::O => "attn_output".into(),
            Slot::Router => "ffn_gate_inp".into(),
            Slot::Gate(e) => format!("ffn_gate.{e}"),
            Slot::Up(e) => format!("ffn_up.{e}"),
            Slot::Down(e) => format!("ffn_down.{e}"),
        }
    }
}

fn slots_for(decoder: &Decoder, l: usize, experts: usize) -> Vec<Slot> {
    let mut out = vec![Slot::Q, Slot::K, Slot::V, Slot::O];
    let n_experts = decoder.layers[l].moe.experts.n_experts();
    if n_experts > 1 {
        out.push(Slot::Router);
    }
    for e in 0..experts.min(n_experts) {
        out.push(Slot::Gate(e));
        out.push(Slot::Up(e));
        out.push(Slot::Down(e));
    }
    out
}

fn slot_mut(decoder: &mut Decoder, l: usize, slot: Slot) -> Option<&mut WeightMatrix> {
    let layer = decoder.layers.get_mut(l)?;
    Some(match slot {
        Slot::Q => &mut layer.attn.q_proj,
        Slot::K => &mut layer.attn.k_proj,
        Slot::V => &mut layer.attn.v_proj,
        Slot::O => &mut layer.attn.o_proj,
        Slot::Router => &mut layer.moe.router,
        Slot::Gate(e) | Slot::Up(e) | Slot::Down(e) => {
            let ExpertBacking::Resident(experts) = &mut layer.moe.experts else {
                return None;
            };
            let expert = experts.get_mut(e)?;
            match slot {
                Slot::Gate(_) => &mut expert.gate,
                Slot::Up(_) => &mut expert.up,
                _ => &mut expert.down,
            }
        }
    })
}

struct BlockStats {
    mean: f64,
    p99: f64,
}

/// Dequantize, requantize to `candidate`, dequantize back, and score
/// every block against the values it started from.
fn round_trip(
    m: &WeightMatrix,
    candidate: Candidate,
) -> anyhow::Result<(WeightMatrix, BlockStats)> {
    let rows = m.rows();
    let cols = m.cols();
    let block = candidate.block_elems();
    let mut bytes: Vec<u8> = Vec::new();
    let mut per_block: Vec<f64> = Vec::with_capacity(rows * cols / block);
    for r in 0..rows {
        let original = m.dequant_row(r);
        let packed = candidate.quantize(&original);
        let back = candidate.dequantize(&packed)?;
        for (a, b) in original.chunks(block).zip(back.chunks(block)) {
            let energy: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let err: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = x as f64 - y as f64;
                    d * d
                })
                .sum();
            // An all-zero block round-trips exactly; scoring it as 0/0
            // would put NaN through the whole tensor's mean.
            per_block.push(if energy > 0.0 { err / energy } else { 0.0 });
        }
        bytes.extend_from_slice(&packed);
    }
    let stats = BlockStats {
        mean: mean(&per_block),
        p99: percentile(&mut per_block, 0.99),
    };
    Ok((
        WeightMatrix::Quantized {
            data: WeightBytes::Owned(bytes),
            rows,
            cols,
            kind: candidate.kind(),
        },
        stats,
    ))
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// `q`-quantile by selection, which is linear and does not need the
/// whole vector sorted.
fn percentile(xs: &mut [f64], q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let idx = (((xs.len() - 1) as f64) * q).round() as usize;
    let (_, nth, _) = xs.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    *nth
}

fn last_logits(decoder: &Decoder, tokens: &[usize]) -> Vec<f32> {
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    decoder.forward_batch_last(tokens, 0, &mut caches)
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&l| (l as f64 - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// KL(clean || perturbed), nats.
///
/// This direction, not the symmetric one: it weights each term by how
/// much probability the *undamaged* model put there, so mass the clean
/// model never used cannot inflate the score.
fn kl_nats(clean_probs: &[f64], perturbed_logits: &[f32]) -> f64 {
    let q = softmax(perturbed_logits);
    let mut acc = 0.0;
    for (p, q) in clean_probs.iter().zip(q.iter()) {
        if *p > 0.0 {
            acc += p * (p / q.max(f64::MIN_POSITIVE)).ln();
        }
    }
    acc.max(0.0)
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

fn print_table(rows: &[Row], top: usize, candidate: Candidate) {
    let mut sorted: Vec<&Row> = rows.iter().collect();
    sorted.sort_by(|a, b| b.kl.total_cmp(&a.kl));
    println!();
    println!(
        "{:<24} {:>6}  {:>12} {:>12}  {:>11}  top1",
        "tensor", "from", "relmse mean", "relmse p99", "dKL nats"
    );
    for r in sorted.iter().take(top) {
        println!(
            "{:<24} {:>6}  {:>12.3e} {:>12.3e}  {:>11.3e}  {}",
            r.name,
            r.kind,
            r.rel_mse_mean,
            r.rel_mse_p99,
            r.kl,
            if r.top1_flipped { "FLIPPED" } else { "kept" }
        );
    }

    let total: f64 = rows.iter().map(|r| r.kl).sum();
    println!();
    for (family, share, worst) in family_shares(rows) {
        println!(
            "{family:<14} {:>5.1}% of measured KL, worst single tensor {worst}",
            share * 100.0
        );
    }
    let flips = rows.iter().filter(|r| r.top1_flipped).count();
    println!(
        "total measured KL {total:.4e} nats across {} tensors at {}; {flips} of them flip the \
         greedy token on their own",
        rows.len(),
        candidate.name(),
    );
}

/// Share of the total measured KL each tensor family carries.
///
/// This is the column a static quant rule is guessing at. A rule that
/// keeps `ffn_down` a tier higher is right on a checkpoint where
/// `ffn_down` is at the top of this list and is spending bits for
/// nothing on one where it is not.
fn family_shares(rows: &[Row]) -> Vec<(String, f64, String)> {
    let total: f64 = rows.iter().map(|r| r.kl).sum();
    let mut by_family: std::collections::BTreeMap<String, (f64, f64, String)> =
        std::collections::BTreeMap::new();
    for r in rows {
        let family = family_of(&r.name);
        let e = by_family
            .entry(family)
            .or_insert((0.0, f64::NEG_INFINITY, String::new()));
        e.0 += r.kl;
        if r.kl > e.1 {
            e.1 = r.kl;
            e.2 = r.name.clone();
        }
    }
    let mut out: Vec<(String, f64, String)> = by_family
        .into_iter()
        .map(|(f, (sum, _, worst))| (f, if total > 0.0 { sum / total } else { 0.0 }, worst))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// `blk.14.ffn_down.0` -> `ffn_down`.
fn family_of(name: &str) -> String {
    let after_layer = name.splitn(3, '.').nth(2).unwrap_or(name);
    after_layer
        .split('.')
        .next()
        .unwrap_or(after_layer)
        .to_string()
}

/// `START:END`, end exclusive, either side optional.
fn parse_layer_range(spec: Option<&str>, n_layers: usize) -> anyhow::Result<(usize, usize)> {
    let Some(spec) = spec else {
        return Ok((0, n_layers));
    };
    let (a, b) = spec
        .split_once(':')
        .context("--layers expects START:END, for example 0:4 or 12:")?;
    let from = if a.is_empty() { 0 } else { a.parse()? };
    let to = if b.is_empty() { n_layers } else { b.parse()? };
    if from >= to {
        anyhow::bail!("--layers {spec} is empty ({from}..{to})");
    }
    if to > n_layers {
        anyhow::bail!("--layers {spec} runs past the model's {n_layers} layers");
    }
    Ok((from, to))
}

fn short(model: &str) -> String {
    Path::new(model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_round_trips_a_block_the_way_frink_quant_reads_it() {
        // The quantizer here and the dequantizer in frink-quant have to
        // agree on the nibble layout, or every number this tool prints
        // is measuring a packing bug.
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.25).collect();
        let bytes = quantize_q4_0(&src);
        assert_eq!(bytes.len(), frink_quant::Q4_0_BLOCK_BYTES);
        let back = frink_quant::dequant_q4_0(&bytes).unwrap();
        assert_eq!(back.len(), 32);
        let amax = src.iter().fold(0f32, |a, &b| a.max(b.abs()));
        for (a, b) in src.iter().zip(back.iter()) {
            // One code step is amax/8; rounding is within half of that.
            assert!((a - b).abs() <= amax / 8.0, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_0_keeps_the_extreme_value_representable() {
        // The scale is anchored on the largest-magnitude element over
        // -8, so that element must come back close, not clipped.
        let mut src = vec![0.01f32; 32];
        src[5] = -3.0;
        let back = frink_quant::dequant_q4_0(&quantize_q4_0(&src)).unwrap();
        assert!((back[5] + 3.0).abs() < 1e-3, "got {}", back[5]);
    }

    #[test]
    fn an_all_zero_block_round_trips_without_a_nan() {
        let back = frink_quant::dequant_q4_0(&quantize_q4_0(&[0.0f32; 32])).unwrap();
        assert!(back.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn kl_of_a_distribution_against_itself_is_zero() {
        let logits = vec![1.0f32, -2.0, 3.5, 0.0];
        let p = softmax(&logits);
        assert!(kl_nats(&p, &logits) < 1e-12);
    }

    #[test]
    fn kl_grows_with_the_damage() {
        let clean = vec![2.0f32, 1.0, 0.0];
        let p = softmax(&clean);
        let small = kl_nats(&p, &[2.05, 1.0, 0.0]);
        let large = kl_nats(&p, &[0.0, 1.0, 2.0]);
        assert!(small > 0.0);
        assert!(large > small * 10.0, "{small} vs {large}");
    }

    #[test]
    fn percentile_picks_the_upper_tail() {
        // Nearest-rank on 100 samples: the 99th percentile is the
        // second-largest, not the largest.
        let mut xs: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(percentile(&mut xs, 0.99), 98.0);
        let mut xs: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(percentile(&mut xs, 0.5), 50.0);
    }

    #[test]
    fn the_int_dot_path_is_refused_rather_than_silently_wrong() {
        assert!(guard_int_dot(Some("1")).is_err());
        assert!(guard_int_dot(Some("on")).is_err());
        assert!(guard_int_dot(Some("0")).is_ok());
        assert!(guard_int_dot(None).is_ok());
    }

    #[test]
    fn layer_ranges_are_end_exclusive_and_bounded() {
        assert_eq!(parse_layer_range(None, 16).unwrap(), (0, 16));
        assert_eq!(parse_layer_range(Some("0:4"), 16).unwrap(), (0, 4));
        assert_eq!(parse_layer_range(Some("12:"), 16).unwrap(), (12, 16));
        assert_eq!(parse_layer_range(Some(":3"), 16).unwrap(), (0, 3));
        assert!(parse_layer_range(Some("4:4"), 16).is_err());
        assert!(parse_layer_range(Some("0:99"), 16).is_err());
        assert!(parse_layer_range(Some("nonsense"), 16).is_err());
    }

    #[test]
    fn family_rollup_strips_the_layer_and_the_expert_index() {
        assert_eq!(family_of("blk.14.ffn_down.0"), "ffn_down");
        assert_eq!(family_of("blk.3.attn_v"), "attn_v");
    }

    #[test]
    fn family_shares_sum_to_one_and_name_the_worst_tensor() {
        let rows = vec![
            Row {
                name: "blk.0.attn_v".into(),
                kind: "q4_K".into(),
                rel_mse_mean: 0.0,
                rel_mse_p99: 0.0,
                kl: 1.0,
                top1_flipped: false,
            },
            Row {
                name: "blk.1.attn_v".into(),
                kind: "q4_K".into(),
                rel_mse_mean: 0.0,
                rel_mse_p99: 0.0,
                kl: 3.0,
                top1_flipped: false,
            },
            Row {
                name: "blk.0.ffn_down.0".into(),
                kind: "q6_K".into(),
                rel_mse_mean: 0.0,
                rel_mse_p99: 0.0,
                kl: 5.0,
                top1_flipped: true,
            },
        ];
        let shares = family_shares(&rows);
        assert_eq!(shares[0].0, "ffn_down");
        assert!((shares.iter().map(|s| s.1).sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(shares[1].2, "blk.1.attn_v");
    }

    #[test]
    fn candidate_names_round_trip() {
        assert_eq!(Candidate::parse("q4_0").unwrap(), Candidate::Q4_0);
        assert_eq!(Candidate::parse("Q8_0").unwrap(), Candidate::Q8_0);
        assert!(Candidate::parse("q4_K").is_err());
    }
}
