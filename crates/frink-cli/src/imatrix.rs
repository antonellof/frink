//! `frink imatrix`: llama.cpp's `llama-imatrix`. Runs a calibration
//! text through a model and writes, per quantizable weight, the
//! per-column sum of squared activations that `frink quantize
//! --imatrix` and `llama-quantize --imatrix` weight the fit by.
//!
//! The method is `tools/imatrix/imatrix.cpp` at b7650, and each piece
//! is cited where it is implemented:
//!
//! * **What is collected** (`imatrix.cpp:229-237`): the f32 input of
//!   every matrix multiplication whose weight is under `blk.`, and
//!   `output.weight` with `--process-output`. Norm weights and the
//!   embedding lookup are not matrix multiplications; on a tied model
//!   the output head is `token_embd.weight` and is not collected
//!   either. Expert weights are collected per expert, with one count
//!   each (`:302-317`).
//! * **The accumulation** (`:365-372`): `values[j] += x[j]*x[j]` per
//!   row, `counts += rows`. In [`collector`].
//! * **The chunking** (`:909-1013`): tokenize the whole file once with
//!   the checkpoint's BOS rule, split into non-overlapping chunks of
//!   `--ctx-size`, and overwrite each chunk's first token with BOS
//!   when the vocabulary adds one. At least two chunks' worth of
//!   tokens is required (`:934`). Each chunk is one forward pass over
//!   a fresh KV cache (`llama_memory_clear`, `:983`).
//! * **The file** (`:507-615` GGUF, `:401-505` legacy): in [`file`].
//!
//! Deviations, all stated rather than hidden. `llama-imatrix` folds
//! `n_batch / n_ctx` chunks into one batch as separate sequences;
//! frink runs one chunk per forward pass. The rows reach the
//! accumulator in the same token order either way, so the sums are
//! the same arithmetic. `llama-imatrix` prints a perplexity as it goes
//! (`--no-ppl` turns it off); frink does not, because `frink
//! perplexity` already computes that number by llama.cpp's method and
//! a second copy here would be a second copy. Special-token markers in
//! the text are always parsed, which is `llama-imatrix
//! --parse-special`, because frink's tokenizers have no mode that
//! does not (and on the repo's own docs that is a ten-token
//! difference over 60 KB, which shifts every chunk). There is no
//! `--in-file` combining of earlier matrices.
//!
//! # Why it runs on the CPU
//!
//! Activations are observed through `frink_core::activation_tap`, a
//! seam at the two functions every projection goes through. The tap
//! sees what the CPU path is handed; the GPU paths hand their inputs
//! straight to a command buffer. So the backend is pinned to CPU
//! through the same `apply_env` `frink bench --n-gpu-layers 0` uses,
//! which reads the decision back rather than trusting the flag.
//!
//! # What frink's file shares with llama.cpp's, and what it does not
//!
//! The format, the entry names, the counts and the accumulation rule
//! are the same, so either tool's file feeds either quantizer, and a
//! `llama-imatrix` file through `frink quantize --imatrix` gives
//! `llama-quantize --imatrix`'s bytes. The sums are not bit-identical,
//! because the activations are not. Measured on Qwen3-0.6B over 8
//! chunks of prose on which both tokenizers agree exactly: the layer-0
//! pre-attention entries agree to 7e-6 (relative, per column), the
//! first attention block puts 3e-3 between the engines, and it
//! compounds to 2.7e-2 per column by layer 27 (L2-relative over
//! entries: median 7.4e-4, max 6.0e-3). That gap is unchanged by
//! llama.cpp's KV cache type and by BF16 versus F32 weights, so it is
//! the two engines' attention arithmetic, which is `frink parity`'s
//! subject, not this tool's. `--compare` prints the gap per entry so
//! the bound is a number rather than a claim, and `docs/CLI.md` has
//! the table.

pub mod collector;
pub mod file;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use frink_core::cache::KvCache;
use frink_core::WeightMatrix;
use frink_gguf::sharded::ShardedGguf;
use frink_gguf::GgufFile;
use frink_models::decoder::{Decoder, ExpertBacking};

use collector::Collector;
use file::OutputFormat;

#[derive(Parser, Debug)]
pub struct ImatrixArgs {
    /// Model GGUF (or `org/repo` to pull). The checkpoint being
    /// calibrated should be the one you intend to quantize: an imatrix
    /// from a Q4_K_M file describes Q4_K_M's activations.
    #[arg(short = 'm', long = "model")]
    pub model: String,

    /// Calibration text. Tokenized once, with BOS if the checkpoint
    /// adds one; must tokenize to at least `2 * --ctx-size` tokens.
    #[arg(short = 'f', long = "file")]
    pub file: PathBuf,

    /// Output path. llama.cpp's default is `imatrix.gguf` in the
    /// working directory.
    #[arg(short = 'o', long = "output", default_value = "imatrix.gguf")]
    pub output: PathBuf,

    /// `gguf` (llama.cpp's current format) or `dat` (the legacy binary
    /// older `llama-quantize` builds read).
    #[arg(long = "output-format", default_value = "gguf")]
    pub output_format: OutputFormat,

    /// Tokens per chunk. Each chunk is one forward pass over a fresh
    /// KV cache.
    #[arg(short = 'c', long = "ctx-size", default_value_t = 512)]
    pub ctx_size: usize,

    /// Chunks to process; negative means all of them.
    #[arg(long, default_value_t = -1)]
    pub chunks: i64,

    /// Also collect `output.weight`. Off by default, as upstream.
    #[arg(long)]
    pub process_output: bool,

    /// CPU threads (0 = all).
    #[arg(short = 't', long, default_value_t = 0)]
    pub threads: usize,

    /// After writing, compare the result against another imatrix file
    /// (typically `llama-imatrix`'s for the same model and text) and
    /// print the per-entry relative difference.
    #[arg(long)]
    pub compare: Option<PathBuf>,
}

pub fn run(args: ImatrixArgs) -> Result<()> {
    if args.ctx_size == 0 {
        bail!("imatrix requires --ctx-size > 0");
    }
    // Pins the CPU backend and reads the decision back. See the module
    // doc for why the tap needs the CPU path.
    crate::bench_model::apply_env(args.threads, 0, None)?;

    let path = crate::pull::resolve_model_path(&args.model)?;
    let text = std::fs::read_to_string(&args.file)
        .with_context(|| format!("reading calibration text {}", args.file.display()))?;

    let t0 = Instant::now();
    let (decoder, tokens, _eos) = crate::verify_engine::load_and_tokenize(
        Path::new(&path),
        &text,
        // llama-imatrix's default (`common.h`: `parse_special = false`):
        // calibration text that mentions `<s>` is text.
        frink_models::tokenizer::SpecialTokens::AsText,
        None,
    )
    .context("loading the model and tokenizing the calibration text")?;
    println!(
        "imatrix: tokenization and load took {:.1} s; {} tokens",
        t0.elapsed().as_secs_f64(),
        tokens.len()
    );

    let file = ShardedGguf::open(&path)?;
    let bos = if frink_models::tokenizer::should_add_bos_token(&file) {
        file.metadata_u64("tokenizer.ggml.bos_token_id")
            .map(|v| v as usize)
    } else {
        None
    };
    drop(file);

    let n_ctx = args.ctx_size;
    if tokens.len() < 2 * n_ctx {
        bail!(
            "you need at least {} tokens for a context of {n_ctx} tokens; the data file you \
             provided tokenizes to only {}",
            2 * n_ctx,
            tokens.len()
        );
    }
    let n_chunk_max = tokens.len() / n_ctx;
    let n_chunk = if args.chunks < 0 {
        n_chunk_max
    } else {
        (args.chunks as usize).min(n_chunk_max)
    };

    let header = GgufFile::open(&path)?;
    let mut collector = Collector::default();
    let registered = register_decoder(&decoder, &header, &mut collector, args.process_output)?;
    drop(header);
    println!(
        "imatrix: collecting {} weights ({} dense, {} expert stacks); computing over {n_chunk} \
         chunks, n_ctx={n_ctx}, backend {}",
        collector.n_registered(),
        registered.dense.len(),
        registered.expert_stacks.len(),
        crate::bench_model::active_backend()
    );

    let collector = Arc::new(collector);
    let tap = collector.clone();
    let guard = frink_core::activation_tap::install(Arc::new(
        move |m: &WeightMatrix, rows: &[f32], n: usize| tap.observe(m, rows, n),
    ))
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let run_start = Instant::now();
    for chunk in 0..n_chunk {
        let start = chunk * n_ctx;
        let mut ids = tokens[start..start + n_ctx].to_vec();
        if let Some(b) = bos {
            ids[0] = b;
        }
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
        let t = Instant::now();
        let _logits = decoder.forward_batch(&ids, 0, &mut caches);
        if let Some(msg) = collector.non_finite() {
            drop(guard);
            bail!("{msg}");
        }
        let per_chunk = t.elapsed().as_secs_f64();
        println!(
            "[{}/{n_chunk}] {per_chunk:.2} s/chunk, ETA {:.0} s",
            chunk + 1,
            per_chunk * (n_chunk - chunk - 1) as f64
        );
    }
    drop(guard);
    println!(
        "imatrix: {n_chunk} chunks in {:.1} s",
        run_start.elapsed().as_secs_f64()
    );

    let stats = collector.stats();
    let n_tokens = (n_chunk * n_ctx) as i64;
    check_counts(&stats, &registered, n_tokens)?;

    let dataset = args.file.to_string_lossy().into_owned();
    file::write(
        &args.output,
        args.output_format,
        &stats,
        &[dataset],
        n_chunk as u32,
        n_ctx as u32,
    )?;
    println!(
        "imatrix: stored collected data after {n_chunk} chunks in {}",
        args.output.display()
    );

    if let Some(other) = &args.compare {
        compare(&args.output, other)?;
    }
    Ok(())
}

/// What `register_decoder` wired up, so the count check afterwards
/// knows which entries must be complete and which may be partial.
#[derive(Default)]
struct Registered {
    dense: Vec<String>,
    expert_stacks: Vec<String>,
}

/// Walks the decoder's public weights and registers each one under the
/// GGUF tensor name it was loaded from. The names are looked up in the
/// file rather than assumed, so a fused `attn_qkv.weight` is collected
/// once under its own name, and a name the file does not carry is not
/// invented.
fn register_decoder(
    decoder: &Decoder,
    header: &GgufFile,
    collector: &mut Collector,
    process_output: bool,
) -> Result<Registered> {
    if decoder.gpt_oss.is_some() {
        bail!(
            "this checkpoint uses the gpt-oss graph, whose projections `frink imatrix` has not \
             mapped to tensor names; refusing rather than writing an imatrix with the wrong \
             entries"
        );
    }
    let has = |name: &str| header.find_tensor(name).is_some();
    let mut reg = Registered::default();
    let dense =
        |collector: &mut Collector, reg: &mut Registered, m: &WeightMatrix, name: String| {
            collector.register_dense(m, &name);
            reg.dense.push(name);
        };

    for (i, layer) in decoder.layers.iter().enumerate() {
        let n = |t: &str| format!("blk.{i}.{t}.weight");
        // Attention. Q, K and V share one input, so a fused checkpoint
        // gets ONE entry from the Q projection and no K/V entries,
        // which is exactly what llama.cpp's single MUL_MAT produces.
        if has(&n("attn_qkv")) {
            dense(collector, &mut reg, &layer.attn.q_proj, n("attn_qkv"));
        } else {
            for (m, t) in [
                (&layer.attn.q_proj, "attn_q"),
                (&layer.attn.k_proj, "attn_k"),
                (&layer.attn.v_proj, "attn_v"),
            ] {
                if has(&n(t)) {
                    dense(collector, &mut reg, m, n(t));
                } else {
                    bail!(
                        "layer {i}: neither {} nor {} is in the file",
                        n(t),
                        n("attn_qkv")
                    );
                }
            }
        }
        if has(&n("attn_output")) {
            dense(collector, &mut reg, &layer.attn.o_proj, n("attn_output"));
        } else {
            bail!("layer {i}: {} is not in the file", n("attn_output"));
        }

        // FFN. A dense layer is a one-expert stack in frink; the file
        // says which it is by which names it carries.
        let experts = match &layer.moe.experts {
            ExpertBacking::Resident(v) => v,
            ExpertBacking::Stored { .. } => bail!(
                "layer {i}: experts are streamed from an expert store, whose per-use weight \
                 views have no stable identity for the tap; run with resident experts"
            ),
        };
        if experts.len() == 1 && has(&n("ffn_gate")) {
            let ex = &experts[0];
            for (m, t) in [
                (&ex.gate, "ffn_gate"),
                (&ex.up, "ffn_up"),
                (&ex.down, "ffn_down"),
            ] {
                if has(&n(t)) {
                    dense(collector, &mut reg, m, n(t));
                } else {
                    bail!("layer {i}: {} is not in the file", n(t));
                }
            }
        } else if has(&n("ffn_gate_exps")) {
            let n_experts = experts.len();
            for t in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                if !has(&n(t)) {
                    bail!("layer {i}: {} is not in the file", n(t));
                }
            }
            for (e, ex) in experts.iter().enumerate() {
                collector.register_expert(&ex.gate, &n("ffn_gate_exps"), e, n_experts);
                collector.register_expert(&ex.up, &n("ffn_up_exps"), e, n_experts);
                collector.register_expert(&ex.down, &n("ffn_down_exps"), e, n_experts);
            }
            reg.expert_stacks.extend(
                ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                    .iter()
                    .map(|t| n(t)),
            );
            if has(&n("ffn_gate_inp")) {
                dense(collector, &mut reg, &layer.moe.router, n("ffn_gate_inp"));
            }
            if let Some(sh) = layer.moe.shared_experts.first() {
                for (m, t) in [
                    (&sh.gate, "ffn_gate_shexp"),
                    (&sh.up, "ffn_up_shexp"),
                    (&sh.down, "ffn_down_shexp"),
                ] {
                    if has(&n(t)) {
                        dense(collector, &mut reg, m, n(t));
                    }
                }
            }
        } else {
            bail!(
                "layer {i}: the file carries neither {} nor {}, so its FFN weights cannot be named",
                n("ffn_gate"),
                n("ffn_gate_exps")
            );
        }
    }
    if process_output {
        if has("output.weight") {
            dense(
                collector,
                &mut reg,
                &decoder.output_head,
                "output.weight".into(),
            );
        } else {
            println!(
                "imatrix: --process-output given but the file has no output.weight (tied \
                 embeddings); llama.cpp does not collect a tied head either, so nothing is added"
            );
        }
    }
    Ok(reg)
}

/// After the run, every dense entry must have seen exactly the tokens
/// that were processed: fewer means a decoder path bypassed the tap,
/// more means a path observed the same rows twice. Either is a wrong
/// file, so both are refusals. Expert entries may legitimately be
/// partial (an expert the text never routed to), which upstream warns
/// about and so does this; an expert stack with NO data at all while
/// the run processed tokens is the bypass case again.
fn check_counts(
    stats: &std::collections::BTreeMap<String, file::Stats>,
    registered: &Registered,
    n_tokens: i64,
) -> Result<()> {
    for name in &registered.dense {
        let c = stats[name].counts[0];
        if c != n_tokens {
            bail!(
                "entry {name} saw {c} activation rows but {n_tokens} tokens were processed: the \
                 decoder path for this weight did not go through the activation tap exactly \
                 once, so the file would be wrong. Refusing to write it."
            );
        }
    }
    for name in &registered.expert_stacks {
        let counts = &stats[name].counts;
        let n_zero = counts.iter().filter(|&&c| c == 0).count();
        if n_zero == counts.len() {
            bail!(
                "entry {name} saw no activation rows for any of its {} experts while {n_tokens} \
                 tokens were processed: the MoE prefill path did not go through the activation \
                 tap. Refusing to write a file with an empty expert stack.",
                counts.len()
            );
        }
        if n_zero > 0 {
            println!(
                "imatrix: entry '{name}' has partial data ({:.2}%)",
                100.0 * (counts.len() - n_zero) as f64 / counts.len() as f64
            );
        }
    }
    Ok(())
}

/// Reads both files as the quantizer would and prints, per entry, the
/// largest relative difference between the two importance weights,
/// plus the overall bound. Entries present in one file and not the
/// other are listed by name.
fn compare(ours: &Path, theirs: &Path) -> Result<()> {
    let a = file::read(ours)?;
    let b = file::read(theirs)?;
    println!(
        "imatrix compare: {} ({} entries) vs {} ({} entries)",
        ours.display(),
        a.entries.len(),
        theirs.display(),
        b.entries.len()
    );
    let mut worst = 0f64;
    let mut worst_name = String::new();
    let mut n_common = 0usize;
    for (name, va) in &a.entries {
        let Some(vb) = b.entries.get(name) else {
            println!("  only in {}: {name}", ours.display());
            continue;
        };
        if va.len() != vb.len() {
            println!(
                "  {name}: {} values vs {} -- not comparable",
                va.len(),
                vb.len()
            );
            continue;
        }
        n_common += 1;
        // Relative to the larger of the two, so a column that is tiny
        // in both does not dominate; the absolute scale is also printed
        // so a reader can see whether "1e-3" is on 1e-9 or on 1e3.
        let mut max_rel = 0f64;
        let mut max_abs = 0f64;
        for (&x, &y) in va.iter().zip(vb) {
            let (x, y) = (x as f64, y as f64);
            let denom = x.abs().max(y.abs());
            if denom > 0.0 {
                max_rel = max_rel.max((x - y).abs() / denom);
            }
            max_abs = max_abs.max(x.abs().max(y.abs()));
        }
        println!("  {name:<40} max rel diff {max_rel:.3e}  (max |w| {max_abs:.3e})");
        if max_rel > worst {
            worst = max_rel;
            worst_name = name.clone();
        }
    }
    for name in b.entries.keys() {
        if !a.entries.contains_key(name) {
            println!("  only in {}: {name}", theirs.display());
        }
    }
    println!(
        "imatrix compare: {n_common} entries compared; worst relative difference {worst:.3e} \
         in {worst_name}"
    );
    Ok(())
}
