//! Per-layer divergence between two backends, read off the KV cache.
//!
//! `frink verify` answers one bit: do two backends produce the same
//! tokens. When the answer is no it says at which token, which is a
//! long way from which layer, and every debugging session since has
//! started by bisecting kernels by hand.
//!
//! This reads the per-layer state the engine already keeps. After one
//! prefill, every layer's `KvCache` holds that layer's K and V for
//! every position, laid out `[seq, n_kv_heads, head_dim]`, so the
//! per-head magnitudes are already there, one layer at a time, with no
//! instrumentation inside the decoder and no second forward pass.
//!
//! Two properties make this worth reading:
//!
//! * **Layer L's K/V are computed from layer L's input.** So the first
//!   layer whose magnitudes disagree bounds where the divergence
//!   entered: at layer L's norm/QKV projection, or in whatever produced
//!   layer L's input (layer L-1's attention output or FFN). Everything
//!   below that layer is exonerated, which is most of the model.
//! * **Per head, and the spread, not the mean.** A mean magnitude ratio
//!   is ~1.0 when a single head of 32 is computing garbage, because the
//!   other 31 hold it there. The standard deviation across heads is
//!   what moves, and a single bad head is the shape of every
//!   simdgroup/warp-indexing bug this project has hit. Both are
//!   printed, so the reader can see the mean failing to notice.
//!
//! What it does not see: layer L's own attention output and FFN, which
//! reach the cache only through layer L+1's K/V. Read a first
//! divergence at layer L as "at or immediately before layer L".
//!
//! MoE routing comes along free: `MoeWeights::activation_counts` is a
//! live per-expert selection histogram, so the same run says whether
//! the two backends routed tokens to the same experts at all. A
//! routing split and a wrong expert kernel look identical in the output
//! text and nothing alike here.
//!
//! Backend choice is a process-lifetime `OnceLock`, so this cannot
//! switch backends in-process: it re-invokes itself once per backend,
//! exactly as `verify` and `bench --suite` do.
//!
//! # The noise floor, measured
//!
//! CPU vs CPU is exactly 1.0 on every head, which is the self-test.
//! CPU vs Metal on a healthy model (Llama-3.2-1B Q4_K_M, 16 prompt
//! tokens, 2026-08-27) spreads 1.6e-5 to 1.3e-4, and OLMoE Q4_0 the
//! same. The default `--tol` of 1e-3 is roughly 8x the worst of that,
//! so it is a threshold with headroom rather than a guess.

use anyhow::Context;
use frink_core::cache::KvCache;
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;

/// Fixed so runs are comparable across invocations and models.
const PROMPT: &str = "The capital of France is";

/// Marker the child prints so the parent can find the payload even if
/// the engine writes other things to stdout.
const TAG: &str = "FRINK_DIVERGENCE ";

/// Below this the batched-prefill attention kernels (`n_q >= 8`) never
/// run, so a clean report covers decode only.
const PREFILL_MIN_TOKENS: usize = 8;

pub struct DivergenceArgs {
    pub model: String,
    /// Backend to compare against the CPU reference. `cpu` is legal and
    /// is the self-test: every ratio must be exactly 1.
    pub backend: String,
    pub prompt: Option<String>,
    pub prompt_tokens: Option<usize>,
    /// Per-head magnitude-ratio spread at or above which a layer is
    /// called diverged.
    pub tol: f64,
    /// `all` scores every prompt position, `last` only the final one.
    pub at: String,
    /// Internal: run one backend and print the payload.
    pub emit: bool,
    /// The global `--allow-multiple-instances` flag. Forwarded to the
    /// children, which are the processes that actually load a model:
    /// without this the flag would be accepted on a command that never
    /// registers and then ignored by the two that do.
    pub allow_multiple_instances: bool,
}

/// One layer's per-head magnitudes plus its routing histogram.
struct LayerProbe {
    /// Positions this layer's cache actually holds. Reported so the
    /// parent can tell "the backends agree" from "this backend handed
    /// back nothing to compare".
    seq: usize,
    /// L2 norm of this head's K over every prompt position.
    k_all: Vec<f64>,
    v_all: Vec<f64>,
    /// The same at the final position only, which is the one the next
    /// token is decoded from.
    k_last: Vec<f64>,
    v_last: Vec<f64>,
    /// How many times each routed expert was selected. Empty for a
    /// dense layer.
    experts: Vec<u64>,
}

pub fn run(args: DivergenceArgs) -> anyhow::Result<()> {
    let prompt = args.prompt.clone().unwrap_or_else(|| PROMPT.to_string());
    if args.emit {
        return emit(&args.model, &prompt, args.prompt_tokens);
    }
    let at = match args.at.as_str() {
        "all" | "last" => args.at.as_str(),
        other => anyhow::bail!("--at expects `all` or `last`, got `{other}`"),
    };

    let reference = child_probe(&args, "cpu", &prompt)?;
    let candidate = child_probe(&args, &args.backend, &prompt)?;
    report(&args, at, &reference, &candidate)
}

/// The comparison, once both children have answered.
fn report(
    args: &DivergenceArgs,
    at: &str,
    reference: &Value,
    candidate: &Value,
) -> anyhow::Result<()> {
    let ref_layers = layers_of(reference)?;
    let cand_layers = layers_of(candidate)?;
    if ref_layers.len() != cand_layers.len() {
        anyhow::bail!(
            "the two children disagree on layer count ({} vs {}); they did not load the same model",
            ref_layers.len(),
            cand_layers.len()
        );
    }
    let prompt_len = reference["prompt_len"].as_u64().unwrap_or(0) as usize;
    if let Some(l) = first_empty_layer(&ref_layers) {
        anyhow::bail!(empty_side_message("cpu", l));
    }
    if let Some(l) = first_empty_layer(&cand_layers) {
        anyhow::bail!(empty_side_message(&args.backend, l));
    }

    println!(
        "layer-divergence {}: cpu vs {}, {} prompt tokens{}, per-head magnitudes at={at}",
        short(&args.model),
        args.backend,
        prompt_len,
        if prompt_len >= PREFILL_MIN_TOKENS {
            ""
        } else {
            " (decode only: under 8 tokens the prefill attention kernels never run)"
        }
    );
    println!(
        "{:>5}  {:>9} {:>9} {:>16}  {:>9} {:>9} {:>16}  routing",
        "layer", "K mean", "K spread", "K worst head", "V mean", "V spread", "V worst head"
    );

    let mut first_bad: Option<(usize, &'static str, HeadStats)> = None;
    let mut routing_split = Vec::new();
    let mut no_counts = 0usize;
    for (l, (r, c)) in ref_layers.iter().zip(cand_layers.iter()).enumerate() {
        let k = compare(&norms(r, "k", at)?, &norms(c, "k", at)?);
        let v = compare(&norms(r, "v", at)?, &norms(c, "v", at)?);
        let route = routing_delta(r, c)?;
        println!(
            "{l:>5}  {:>9.6} {:>9.2e} {:>16}  {:>9.6} {:>9.2e} {:>16}  {}",
            k.mean,
            k.spread,
            k.worst_label(),
            v.mean,
            v.spread,
            v.worst_label(),
            match route {
                Routing::Dense => "-".to_string(),
                Routing::NoCounts => "no counts".to_string(),
                Routing::Delta(d) => format!("{d:.4} TV"),
            }
        );
        if first_bad.is_none() {
            if k.spread >= args.tol {
                first_bad = Some((l, "K", k));
            } else if v.spread >= args.tol {
                first_bad = Some((l, "V", v));
            }
        }
        match route {
            // Selection counts are integers, so any real split is far
            // above this; the guard is only against a division artifact.
            Routing::Delta(d) if d > 1e-9 => routing_split.push((l, d)),
            Routing::NoCounts => no_counts += 1,
            _ => {}
        }
    }

    for (l, d) in &routing_split {
        println!(
            "routing: layer {l} sent {:.2}% of its expert selections to different experts",
            d * 100.0
        );
    }
    if no_counts > 0 {
        println!(
            "routing: {no_counts} layers could not be compared because one side recorded no \
             expert selections. `MoeWeights::activation_counts` is what the placement plan calls \
             observed hotness, so a backend that never records leaves that plan guessing."
        );
    }

    match first_bad {
        None => {
            println!(
                "OK: every layer's per-head magnitude spread is under {:.1e} on both K and V",
                args.tol
            );
            Ok(())
        }
        Some((l, which, s)) => {
            println!(
                "DIVERGED at layer {l} ({which}): spread {:.3e} across {} heads, worst head {} at ratio {:.6} ({}/cpu)",
                s.spread,
                s.n_heads,
                s.worst_head,
                s.worst_ratio,
                args.backend
            );
            println!(
                "  layer {l}'s K/V come from layer {l}'s input, so the fault is in layer {l}'s \
                 norm/QKV projection or in whatever produced its input (layer {}'s attention \
                 output or FFN). Layers below {l} agree.",
                l.saturating_sub(1)
            );
            anyhow::bail!(
                "{} diverges from the CPU reference at layer {l}",
                args.backend
            )
        }
    }
}

/// The first layer that reported positions but no magnitude, or `None`.
///
/// This is the vacuous pass this tool is most exposed to. A backend
/// whose K and V are all zero produces a per-head ratio of zero on every
/// head, which has a spread of zero, which reads as perfect agreement.
/// Nothing about that is a comparison, so it has to be an error rather
/// than a row of `0.00e0`.
fn first_empty_layer(layers: &[Value]) -> Option<usize> {
    layers.iter().position(|l| {
        let seq = l["seq"].as_u64().unwrap_or(0);
        let k: f64 = l["k_all"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).sum())
            .unwrap_or(0.0);
        seq > 0 && k <= 0.0
    })
}

fn empty_side_message(backend: &str, layer: usize) -> String {
    format!(
        "the {backend} side reported layer {layer} as holding positions with zero magnitude, so \
         there is nothing to compare there. On Metal this means the fused prefill stack kept K \
         and V on the device and the readback did not reach them; rerun with \
         FRINK_METAL_ATTN=0 to compare the per-layer Metal matmuls against the CPU reference \
         with the KV on the host."
    )
}

/// Per-head magnitude ratios, summarised.
struct HeadStats {
    n_heads: usize,
    mean: f64,
    /// Population standard deviation of the per-head ratios. The number
    /// this tool exists for: the mean stays at 1.0 while one head of
    /// thirty-two is wrong.
    spread: f64,
    worst_head: usize,
    worst_ratio: f64,
}

impl HeadStats {
    fn worst_label(&self) -> String {
        format!("{} ({:.4})", self.worst_head, self.worst_ratio)
    }
}

/// Ratio of candidate to reference, per head.
///
/// The epsilon makes a head that is zero on both sides read as ratio 1
/// rather than NaN, and a head that is zero on only one side read as a
/// number far from 1 rather than an infinity that poisons the spread.
fn compare(reference: &[f64], candidate: &[f64]) -> HeadStats {
    const EPS: f64 = 1e-30;
    let n = reference.len().min(candidate.len());
    if n == 0 {
        return HeadStats {
            n_heads: 0,
            mean: 1.0,
            spread: 0.0,
            worst_head: 0,
            worst_ratio: 1.0,
        };
    }
    let ratios: Vec<f64> = (0..n)
        .map(|h| (candidate[h] + EPS) / (reference[h] + EPS))
        .collect();
    let mean = ratios.iter().sum::<f64>() / n as f64;
    let var = ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let (worst_head, worst_ratio) =
        ratios
            .iter()
            .enumerate()
            .fold((0usize, 1.0f64), |acc, (h, &r)| {
                if (r - 1.0).abs() > (acc.1 - 1.0).abs() {
                    (h, r)
                } else {
                    acc
                }
            });
    HeadStats {
        n_heads: n,
        mean,
        spread: var.sqrt(),
        worst_head,
        worst_ratio,
    }
}

/// What the two expert-selection histograms say about each other.
#[derive(Debug, PartialEq)]
enum Routing {
    /// Neither side has routed experts: a dense layer.
    Dense,
    /// One side recorded no selections at all. Not agreement, and not a
    /// split: that backend's MoE path does not maintain the counter, so
    /// there is nothing to compare. Worth its own variant because
    /// printing it as `0.0000 TV` would claim the routing matched.
    NoCounts,
    /// Total-variation distance between the two histograms.
    Delta(f64),
}

/// Compare the two expert-selection histograms.
///
/// Normalised by the selection count, so a run that decoded a different
/// number of positions cannot masquerade as a routing split.
fn routing_delta(reference: &Value, candidate: &Value) -> anyhow::Result<Routing> {
    let a = counts_of(reference)?;
    let b = counts_of(candidate)?;
    if a.is_empty() && b.is_empty() {
        return Ok(Routing::Dense);
    }
    if a.len() != b.len() {
        return Ok(Routing::NoCounts);
    }
    let sa: f64 = a.iter().sum::<u64>() as f64;
    let sb: f64 = b.iter().sum::<u64>() as f64;
    if sa == 0.0 || sb == 0.0 {
        return Ok(Routing::NoCounts);
    }
    let tv = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 / sa - *y as f64 / sb).abs())
        .sum::<f64>()
        / 2.0;
    Ok(Routing::Delta(tv))
}

fn counts_of(layer: &Value) -> anyhow::Result<Vec<u64>> {
    Ok(layer["experts"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default())
}

fn norms(layer: &Value, which: &str, at: &str) -> anyhow::Result<Vec<f64>> {
    let key = format!("{which}_{at}");
    layer[&key]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .with_context(|| format!("child payload has no `{key}` for a layer"))
}

fn layers_of(payload: &Value) -> anyhow::Result<Vec<Value>> {
    payload["layers"]
        .as_array()
        .cloned()
        .context("child payload has no `layers` array")
}

fn short(model: &str) -> String {
    Path::new(model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.to_string())
}

/// Runs one backend in a child and parses back its payload.
fn child_probe(args: &DivergenceArgs, backend: &str, prompt: &str) -> anyhow::Result<Value> {
    let exe = std::env::current_exe()?;
    let mut cmd = Command::new(&exe);
    cmd.arg("layer-divergence")
        .args(["-m", &args.model])
        .args(["--backend", backend])
        .args(["--prompt", prompt])
        .arg("--emit");
    if let Some(n) = args.prompt_tokens {
        cmd.args(["--prompt-tokens", &n.to_string()]);
    }
    if args.allow_multiple_instances {
        cmd.env("FRINK_ALLOW_MULTIPLE_INSTANCES", "1");
    }
    // Same reasoning as `verify`: `FRINK_METAL` alone turns on only the
    // Metal matvecs, and the fused attention block that owns the
    // device-resident KV this tool reads is behind `FRINK_METAL_ATTN`.
    // A run that leaves it unset probes a graph nobody runs.
    let metal_attn = std::env::var("FRINK_METAL_ATTN").unwrap_or_else(|_| "1".to_string());
    let out = cmd
        .env("FRINK_METAL", if backend == "cpu" { "0" } else { "1" })
        .env(
            "FRINK_METAL_ATTN",
            if backend == "cpu" { "0" } else { &metal_attn },
        )
        .env("FRINK_CUDA", if backend == "cuda" { "1" } else { "0" })
        .output()
        .with_context(|| format!("spawning layer-divergence child for {backend}"))?;
    if !out.status.success() {
        anyhow::bail!(
            "layer-divergence child for {backend} failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("(no stderr)")
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix(TAG))
        .with_context(|| format!("layer-divergence child for {backend} printed no payload"))?;
    serde_json::from_str(line)
        .with_context(|| format!("layer-divergence child for {backend} printed invalid JSON"))
}

/// Child side: one prefill, then read every layer's cache.
fn emit(model: &str, prompt: &str, prompt_tokens: Option<usize>) -> anyhow::Result<()> {
    let path = crate::pull::resolve_model_path(model)?;
    let (decoder, tokens, _eos) = crate::verify_engine::load_and_tokenize(
        Path::new(&path),
        prompt,
        frink_models::tokenizer::SpecialTokens::Parse,
        prompt_tokens,
    )?;
    let mut caches = fresh_caches(&decoder);
    let _ = decoder.forward_batch_last(&tokens, 0, &mut caches);

    // Metal's fused prefill stack keeps K and V device-resident and only
    // calls `advance_len` on the host cache, so the host copy comes back
    // the right LENGTH and full of zeros. `sync_metal_attn_kv_to_host`
    // cannot repair that in place -- it appends the device's suffix past
    // `cache.seq_len`, and `seq_len` is already caught up -- so the pull
    // goes into a second, empty set of caches, which the device fills
    // from position 0.
    #[cfg(feature = "metal")]
    let device_view = {
        let mut view = fresh_caches(&decoder);
        decoder.sync_metal_attn_kv_to_host(&mut view);
        Some(view)
    };
    #[cfg(not(feature = "metal"))]
    let device_view: Option<Vec<KvCache>> = None;

    let mut from_device = 0usize;
    let probes: Vec<LayerProbe> = (0..decoder.layers.len())
        .map(|l| {
            let host = &caches[l];
            // The host cache is the record whenever it holds anything
            // real; the device copy is read only where it does not.
            let cache = if energy(&host.k.to_f32_vec()) > 0.0 {
                host
            } else {
                match device_view.as_ref().map(|v| &v[l]) {
                    Some(d) if energy(&d.k.to_f32_vec()) > 0.0 => {
                        from_device += 1;
                        d
                    }
                    _ => host,
                }
            };
            let heads = cache.n_kv_heads;
            let dim = cache.head_dim;
            // ROWS: `last` indexes the final resident row of k/v below.
            let seq = cache.rows();
            let last = seq.saturating_sub(1);
            let (k, v) = (cache.k.to_f32_vec(), cache.v.to_f32_vec());
            LayerProbe {
                seq,
                k_all: head_norms(&k, heads, dim, 0, seq),
                v_all: head_norms(&v, heads, dim, 0, seq),
                k_last: head_norms(&k, heads, dim, last, seq),
                v_last: head_norms(&v, heads, dim, last, seq),
                experts: routed_counts(&decoder.layers[l]),
            }
        })
        .collect();

    let payload = json!({
        "prompt_len": tokens.len(),
        "from_device": from_device,
        "layers": probes
            .iter()
            .map(|p| json!({
                "seq": p.seq,
                "k_all": p.k_all,
                "v_all": p.v_all,
                "k_last": p.k_last,
                "v_last": p.v_last,
                "experts": p.experts,
            }))
            .collect::<Vec<_>>(),
    });
    println!("{TAG}{payload}");
    Ok(())
}

fn fresh_caches(decoder: &frink_models::decoder::Decoder) -> Vec<KvCache> {
    decoder.config.new_kv_caches()
}

/// Sum of squares, used only to ask whether a buffer holds anything at
/// all.
fn energy(buf: &[f32]) -> f64 {
    buf.iter().map(|&x| (x as f64) * (x as f64)).sum()
}

/// This layer's routed-expert selection histogram, or empty when the
/// layer is dense.
///
/// A dense FFN is carried as a single "expert", and a one-entry
/// histogram is always a perfect match no matter what the backends did
/// (printing `0.0000 TV` on every dense layer would be a column of
/// reassurance nobody measured).
fn routed_counts(layer: &frink_models::decoder::LayerWeights) -> Vec<u64> {
    if layer.moe.activation_counts.len() <= 1 {
        return Vec::new();
    }
    layer
        .moe
        .activation_counts
        .iter()
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .collect()
}

/// L2 norm of each head's slice over positions `[from, to)`.
///
/// `buf` is `[seq, n_heads, head_dim]` flattened, which is the layout
/// `KvCache` documents and `Decoder` writes.
fn head_norms(buf: &[f32], n_heads: usize, head_dim: usize, from: usize, to: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n_heads];
    if head_dim == 0 || n_heads == 0 {
        return out;
    }
    let per_pos = n_heads * head_dim;
    for p in from..to {
        let base = p * per_pos;
        if base + per_pos > buf.len() {
            break;
        }
        for (h, acc) in out.iter_mut().enumerate() {
            let s = &buf[base + h * head_dim..base + (h + 1) * head_dim];
            *acc += s.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        }
    }
    for acc in out.iter_mut() {
        *acc = acc.sqrt();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_norms_reads_the_seq_heads_dim_layout() {
        // 2 positions, 2 heads, 2 dims: head 0 is (3,4) then (0,0);
        // head 1 is (0,0) then (5,12).
        let buf = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0];
        let all = head_norms(&buf, 2, 2, 0, 2);
        assert!((all[0] - 5.0).abs() < 1e-9);
        assert!((all[1] - 13.0).abs() < 1e-9);
        // Last position only sees head 1.
        let last = head_norms(&buf, 2, 2, 1, 2);
        assert!((last[0] - 0.0).abs() < 1e-9);
        assert!((last[1] - 13.0).abs() < 1e-9);
    }

    #[test]
    fn one_wrong_head_moves_the_spread_and_not_the_mean() {
        // This is the whole argument for reporting the spread. Thirty-two
        // heads, one of them 40% high and another 40% low: the mean ratio
        // is exactly 1.0 and says the backends agree.
        let reference = vec![1.0f64; 32];
        let mut candidate = vec![1.0f64; 32];
        candidate[7] = 1.4;
        candidate[19] = 0.6;
        let s = compare(&reference, &candidate);
        assert!((s.mean - 1.0).abs() < 1e-12, "mean was {}", s.mean);
        assert!(s.spread > 0.09, "spread was {}", s.spread);
        assert!(s.worst_head == 7 || s.worst_head == 19);
    }

    #[test]
    fn identical_heads_are_exactly_one() {
        let a = vec![0.5, 2.0, 7.25];
        let s = compare(&a, &a);
        assert_eq!(s.spread, 0.0);
        assert!((s.mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_head_that_is_zero_on_both_sides_is_not_a_divergence() {
        // A padded or unused head must not read as NaN, and must not
        // read as an infinite ratio either -- both would make the
        // spread useless for every other head in the layer.
        let s = compare(&[0.0, 1.0], &[0.0, 1.0]);
        assert!(s.spread.is_finite());
        assert!(s.spread < 1e-9);
    }

    #[test]
    fn a_head_that_collapsed_to_zero_is_flagged() {
        let s = compare(&[1.0, 1.0], &[1.0, 0.0]);
        assert_eq!(s.worst_head, 1);
        assert!(s.worst_ratio < 1e-9);
        assert!(s.spread > 0.4);
    }

    #[test]
    fn a_side_with_positions_but_no_magnitude_is_an_error_not_a_pass() {
        // Metal's fused prefill stack advances the host cache's length
        // and leaves it zero-filled. Every per-head ratio is then 0/x,
        // whose spread is 0, which reads as perfect agreement -- the
        // loudest possible false OK. It has to be caught structurally.
        let zeroed = vec![json!({"seq": 16, "k_all": [0.0, 0.0], "v_all": [0.0, 0.0]})];
        assert_eq!(first_empty_layer(&zeroed), Some(0));
    }

    #[test]
    fn a_layer_that_holds_no_positions_at_all_is_not_flagged() {
        // A cache with seq 0 measured nothing and claims nothing; only
        // "length without content" is the trap.
        let empty = vec![json!({"seq": 0, "k_all": [0.0], "v_all": [0.0]})];
        assert_eq!(first_empty_layer(&empty), None);
        let real = vec![json!({"seq": 4, "k_all": [1.5], "v_all": [2.0]})];
        assert_eq!(first_empty_layer(&real), None);
    }

    #[test]
    fn routing_delta_is_zero_for_identical_histograms_and_scale_free() {
        let a = json!({"experts": [10u64, 30, 60]});
        // Same distribution, twice as many selections: not a split.
        let b = json!({"experts": [20u64, 60, 120]});
        assert_eq!(routing_delta(&a, &b).unwrap(), Routing::Delta(0.0));
    }

    #[test]
    fn routing_delta_sees_a_swapped_expert() {
        let a = json!({"experts": [100u64, 0]});
        let b = json!({"experts": [0u64, 100]});
        assert_eq!(routing_delta(&a, &b).unwrap(), Routing::Delta(1.0));
    }

    #[test]
    fn a_side_that_recorded_nothing_is_not_agreement() {
        // The Metal MoE prefill path leaves activation_counts at zero
        // while the CPU path fills it in. Scoring that as a perfect
        // match would announce that routing agreed when one side never
        // said where it routed.
        let recorded = json!({"experts": [10u64, 30, 60]});
        let silent = json!({"experts": [0u64, 0, 0]});
        assert_eq!(
            routing_delta(&recorded, &silent).unwrap(),
            Routing::NoCounts
        );
    }

    #[test]
    fn a_dense_layer_has_no_routing_number() {
        let dense = json!({"experts": []});
        assert_eq!(routing_delta(&dense, &dense).unwrap(), Routing::Dense);
    }
}
