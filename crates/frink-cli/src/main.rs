//! frink CLI, llama.cpp-style GGUF completion (`-m`/`-p`/`-n`/…) plus
//! inspect / presets / smoke / Kimi helpers. See `docs/CLI.md`.

mod batched_bench;
mod bench_bw;
mod bench_client;
mod bench_contract;
mod bench_guard;
mod bench_model;
mod bench_render;
mod bench_suite;
mod chat;
mod cli_output;
mod download;
mod gguf_split;
mod hf;
mod host_state;
mod http;
mod imatrix;
mod layer_divergence;
mod parity;
mod perplexity;
mod pull;
mod quant_sensitivity;
mod quantize;
mod run;
mod serve_bench;
mod splice_pooler;
mod verify;
mod verify_engine;

use clap::{Parser, Subcommand};

use frink_core::cache::KvCache;
use frink_gguf::ShardedGguf;
use frink_models::{
    config::test_dense_fixture, deepseek_v4_pro, glm_5_2, kimi_k3, Decoder, ModelConfig,
};

#[derive(Parser)]
#[command(name = "frink", version, about = "Pure-Rust MoE inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Start even though another frink process is already holding a
    /// model. Off by default: two models on one box do not share it,
    /// they thrash it, and every timing either reports becomes noise.
    /// `FRINK_ALLOW_MULTIPLE_INSTANCES=1` does the same.
    #[arg(long, global = true)]
    allow_multiple_instances: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// GGUF completion (llama.cpp-style `-m`/`-p`/`-n`/…).
    ///
    /// Also accepts top-level flags: `frink -m model.gguf -p "Hi" -n 64`.
    ///
    /// `cli` is an ALIAS, not a second command: llama.cpp's launcher
    /// spells this `llama cli`, and a user moving between the two
    /// should not have to learn that one calls it `run`. Both reach
    /// the same body.
    #[command(alias = "cli")]
    Run(run::InferArgs),
    /// Show version.
    ///
    /// `--version` already exists; this is the SUBCOMMAND spelling
    /// llama.cpp's launcher offers, and a user typing `frink version`
    /// should not be told it is not a command.
    Version,
    /// Show third-party licenses.
    Licenses,
    /// Multi-turn chat REPL against a running `frink-server` (HTTP).
    ///
    /// Reuses the server's chat-template + streaming path, start the
    /// server first (`FRINK_MODEL_PATH=… frink-server`).
    Chat(chat::ChatArgs),
    /// Serve the OpenAI-compatible HTTP API.
    ///
    /// Identical to the standalone `frink-server` binary: same flags,
    /// same routes, same `frink.server.ready` line on stdout, it links
    /// the same library rather than reimplementing it.
    #[cfg(feature = "serve")]
    Serve(frink_server::ServerArgs),
    /// Serve the OpenAI-compatible HTTP API. NOT BUILT INTO THIS BINARY.
    ///
    /// Present so the failure is a sentence instead of clap's
    /// "unrecognized subcommand", which reads like the feature does not
    /// exist rather than like it was compiled out.
    #[cfg(not(feature = "serve"))]
    Serve {
        /// Swallowed so `frink serve -m model.gguf` reaches the
        /// explanation below instead of dying on an unexpected flag.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
        args: Vec<String>,
    },
    /// Measure this host's CPU/PCIe bandwidths and write the profile
    /// `qstar` reads to decide the MoE fetch split.
    ///
    /// Without one, every deployment gets an unbenchmarked default.
    /// The PCIe half needs a CUDA build; without it the command
    /// measures the CPU side, says so, and writes nothing rather than
    /// half a profile.
    BenchBw(bench_bw::BenchBwArgs),
    /// Concurrency, TTFT and queueing numbers for a running
    /// `frink-server` (HTTP).
    ///
    /// Distinct from `frink bench`, which is single-stream and
    /// HTTP-free and measures kernels against `llama-bench`. This one
    /// answers what the SERVER does under load. Start the server first.
    ServeBench(serve_bench::ServeBenchArgs),
    /// Throughput as a function of batch size, like
    /// `llama-batched-bench`: for every `-npp` x `-ntg` x `-npl`
    /// combination, prompt speed, decode speed and the total, in
    /// llama.cpp's ten columns. Drives the continuous batcher's engine
    /// seams directly, no HTTP.
    #[command(name = "batched-bench")]
    BatchedBench(batched_bench::BatchedBenchArgs),
    /// Download a GGUF from Hugging Face Hub (`hf download`).
    Pull(pull::PullArgs),

    /// Download a GGUF from Hugging Face, same syntax as `hf download`.
    Download(download::DownloadArgs),
    /// Write a quantized copy of a GGUF, byte-identical to
    /// `llama-quantize`'s.
    ///
    /// frink READS every quant kind it runs and WRITES `Q8_0`, `Q4_K_S`,
    /// `Q4_K_M`, `Q5_K_S`, `Q5_K_M` and `Q6_K` -- the full llama.cpp
    /// mix for each, or `--pure` for the target's block format
    /// everywhere -- with or without `--imatrix`. Every other target
    /// (Q2_K/Q3_K, the IQ tiers, MXFP4, the legacy Q4_0 family) is
    /// refused BY NAME: each needs its own transcription of an
    /// iterative fit, and an approximation of one produces a file that
    /// loads and generates measurably worse text. Use `llama-quantize`
    /// for those; frink reads what it writes.
    Quantize(quantize::QuantizeArgs),
    /// Compute an importance matrix from a calibration text:
    /// llama.cpp's `llama-imatrix`, same file format in both
    /// directions. Feed the result to `frink quantize --imatrix` or to
    /// `llama-quantize --imatrix`.
    Imatrix(imatrix::ImatrixArgs),
    /// Split a GGUF into shards, or merge a shard set back into one
    /// file: llama.cpp's `llama-gguf-split`, same flags and same
    /// `<prefix>-NNNNN-of-MMMMM.gguf` names.
    #[command(name = "gguf-split")]
    GgufSplit(gguf_split::GgufSplitArgs),
    /// Write a reranker GGUF that carries the pooler llama.cpp's
    /// converter dropped (`bert.pooler.dense` -> `cls`), taken from
    /// the checkpoint's own safetensors, so `/v1/rerank` scores on the
    /// trained range instead of an uncalibrated one (issue #82).
    #[command(name = "splice-pooler")]
    SplicePooler(splice_pooler::SplicePoolerArgs),
    /// Print GGUF header metadata and tensor list for a model file.
    Inspect { path: String },
    /// Dry-run residency plan for a GGUF checkpoint: what it would
    /// cost to run (dense weights, routed experts resident vs.
    /// streamed, KV caches) against the selected backend's memory
    /// budget -- computed from the header alone, nothing loaded. Always
    /// reports the largest context that fits and the arithmetic behind
    /// it. --strict exits non-zero when the plan overcommits.
    InspectPlan {
        path: String,
        /// Context length each request's KV cache is sized for.
        /// Omit to price the model's own `{arch}.context_length`.
        #[arg(long)]
        context: Option<usize>,
        /// Concurrent requests to budget KV caches for.
        #[arg(long, default_value_t = 1)]
        concurrency: usize,
        /// Stream routed experts through a bounded cache of this many
        /// bytes instead of counting them fully resident.
        #[arg(long)]
        expert_cache_bytes: Option<u64>,
        /// Which backend's memory budget to plan against.
        #[arg(long, default_value = "cpu")]
        backend: frink_models::BudgetBackend,
        /// KV cache dtype the plan should price (`--ctk` analogue).
        /// Only the Metal path has a device KV whose dtype this
        /// selects; on CPU the host cache is always f32.
        #[arg(long, default_value = "f32")]
        ctk: String,
        /// Refuse (exit 1) when the plan exceeds the usable budget.
        #[arg(long)]
        strict: bool,
    },
    /// List the built-in architecture presets and their headline stats.
    Presets,
    /// Print the GGUF architecture coverage manifest (llama.cpp inventory
    /// classified into Frink families / scope). Pass `--write PATH` to
    /// regenerate `docs/manifests/architecture_manifest.md`.
    Archs {
        #[arg(long)]
        write: Option<String>,
    },
    /// Probe and print this host's hardware capabilities (CPU cores,
    /// RAM, SIMD flags, and CUDA device info if built with --features
    /// cuda). See frink-cuda's docs for exactly what is and isn't
    /// verified about the CUDA fields.
    Caps,
    /// Run a tiny synthetic forward-pass smoke test for a preset
    /// (random weights, small dims) to prove the pipeline executes.
    Smoke {
        /// One of: glm-5.2, deepseek-v4-pro, kimi-k3
        #[arg(default_value = "glm-5.2")]
        preset: String,
        #[arg(long, default_value_t = 8)]
        steps: usize,
    },
    /// Load REAL weights from a GGUF file (test-dense or test-moe
    /// fixture) and print logits for a single decode step, for
    /// cross-validation against the Python reference implementation.
    RunReal {
        path: String,
        #[arg(long, default_value_t = 0)]
        token: usize,
        #[arg(long, default_value_t = 0)]
        pos: usize,
        /// "dense" (single-expert test fixture) or "moe" (4-expert
        /// test fixture)
        #[arg(long, default_value = "dense")]
        fixture: String,
    },
    /// Benchmark: fused quantized matvec vs. dequant-then-matmul, at a
    /// realistic single-expert-FFN matrix size, measuring both wall
    /// time and resident memory. Also benchmarks end-to-end decode
    /// throughput (tokens/sec) for each preset at synthetic-but-full-
    /// scale-shape dimensions.
    /// Check that a GPU backend agrees with the CPU reference.
    ///
    /// Greedy-decodes the same prompt on both and compares token ids.
    /// The benchmark harness measures throughput and never inspects
    /// output, which has let wrong-kernel bugs sit behind green rows.
    Verify {
        /// GGUF to check.
        #[arg(short = 'm', long)]
        model: String,
        /// Backend to compare against the CPU reference.
        #[arg(long, default_value = "metal")]
        backend: String,
        /// Internal: print token ids for one backend and exit.
        #[arg(long, hide = true)]
        emit: bool,
        /// Stretch the prompt to this many tokens (repeating it) before
        /// prefill. Under 8 tokens the batched-prefill attention kernels
        /// never run, so the default prompt checks decode only.
        #[arg(long)]
        prompt_tokens: Option<usize>,
        /// Prompt to compare on. Defaults to a fixed short one so runs
        /// are comparable across models.
        #[arg(short = 'p', long)]
        prompt: Option<String>,
    },
    /// Check that frink agrees with llama.cpp on the first-token
    /// distribution.
    ///
    /// `verify` compares frink-CPU against frink-Metal, so both can be
    /// wrong together. This feeds the same token ids to llama.cpp's own
    /// library and compares the logit distributions, not greedy text,
    /// which cannot separate a wrong graph from two near-tied logits
    /// swapping.
    Parity {
        /// GGUF to check.
        #[arg(short = 'm', long)]
        model: String,
        /// Prompt to compare on. Defaults to a fixed short one so runs
        /// are comparable across models and sessions.
        #[arg(short = 'p', long)]
        prompt: Option<String>,
        /// Stretch the prompt to this many tokens before prefill, so the
        /// batched-prefill kernels are actually reached.
        #[arg(long)]
        prompt_tokens: Option<usize>,
        /// How many top tokens to intersect between the two engines.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// Compiled reference dumper (see tools/llama_logits.c).
        ///
        /// REPEATABLE, and repeating it is what makes a WRONG verdict
        /// believable. The first is the reference every printed number
        /// is about; each further one must be built against a DIFFERENT
        /// libllama (`LLAMA_CPP_PREFIX=… tools/build_llama_logits.sh`),
        /// and `parity` measures how far those builds are from each
        /// other on this checkpoint to decide what "the graphs
        /// disagree" is allowed to mean. Two builds have been measured
        /// 3.5e-2 apart in KL from an identical graph, above the
        /// constant that used to be the WRONG line, so with only one
        /// reference no K-quant checkpoint can be called WRONG at all.
        #[arg(long)]
        dumper: Vec<String>,
        /// Write every compared logit vector under this prefix
        /// (`<prefix>.llama.f32`, `<prefix>.llama2.f32`, …,
        /// `<prefix>.frink.f32`, `<prefix>.tokens.txt`), as raw
        /// little-endian f32.
        ///
        /// A parity verdict is a distance between two points and says
        /// where neither one is, so it cannot tell "frink moved" from
        /// "the reference moved". The `.llama*.f32` files compared to
        /// each other answer that, with no frink in the comparison.
        #[arg(long)]
        dump_logits: Option<String>,
    },
    /// Corpus perplexity, llama.cpp's `perplexity` tool.
    ///
    /// `parity` compares one distribution and `verify` compares text;
    /// neither says whether a quantized checkpoint is WORSE. This scores
    /// a whole corpus in non-overlapping `--ctx-size` windows, scoring
    /// only each window's second half, and reports `exp(mean nll)` with
    /// a standard error — upstream's method, so the number is
    /// comparable to a published one.
    Perplexity {
        /// GGUF to score.
        #[arg(short = 'm', long)]
        model: String,
        /// Plain-text corpus. `crates/frink-cli/tests/corpus/` has one.
        #[arg(short = 'f', long)]
        file: String,
        /// Window length. 512 is llama.cpp's `perplexity` default, and a
        /// perplexity at another context is not comparable to one at 512.
        #[arg(short = 'c', long = "ctx-size", default_value_t = perplexity::DEFAULT_CTX)]
        ctx_size: usize,
        /// Stop after this many windows (default: as many as fit).
        #[arg(long)]
        chunks: Option<usize>,
    },
    /// Find the layer where a GPU backend starts disagreeing with the
    /// CPU reference, rather than the token.
    ///
    /// One prefill per backend, then every layer's KV cache is read
    /// back and scored per head. Reports the SPREAD of the per-head
    /// magnitude ratios, not the mean: one wrong head in thirty-two
    /// leaves the mean at 1.0. MoE checkpoints also get a per-layer
    /// expert-routing comparison.
    LayerDivergence {
        /// GGUF to check.
        #[arg(short = 'm', long)]
        model: String,
        /// Backend to compare against the CPU reference. `cpu` is the
        /// self-test: every ratio must come back exactly 1.
        #[arg(long, default_value = "metal")]
        backend: String,
        /// Prompt to compare on. Defaults to a fixed short one so runs
        /// are comparable across models.
        #[arg(short = 'p', long)]
        prompt: Option<String>,
        /// Stretch the prompt to this many tokens before prefill, so
        /// the batched-prefill kernels are actually reached.
        #[arg(long)]
        prompt_tokens: Option<usize>,
        /// Per-head magnitude-ratio spread at or above which a layer is
        /// called diverged.
        #[arg(long, default_value_t = 1e-3)]
        tol: f64,
        /// Score every prompt position (`all`) or only the last one
        /// (`last`, the position the next token is decoded from).
        #[arg(long, default_value = "all")]
        at: String,
        /// Internal: probe one backend and print the payload.
        #[arg(long, hide = true)]
        emit: bool,
    },
    /// Measure what quantizing each tensor would cost THIS checkpoint,
    /// instead of applying a static quant-mix rule to it.
    ///
    /// Round-trips one tensor at a time through a candidate format,
    /// scores relative_mse per block, swaps it into the loaded model,
    /// and reports how far the next-token distribution moved. Every
    /// other weight stays as the checkpoint shipped it, so no tensor
    /// inherits the layers above it.
    QuantSensitivity {
        /// GGUF to measure.
        #[arg(short = 'm', long)]
        model: String,
        /// Prompt the sensitivity is measured on.
        #[arg(short = 'p', long)]
        prompt: Option<String>,
        /// Stretch the prompt to this many tokens before prefill.
        #[arg(long)]
        prompt_tokens: Option<usize>,
        /// Format to round-trip through: `q4_0` or `q8_0`.
        #[arg(long, default_value = "q4_0")]
        candidate: String,
        /// Restrict the sweep to `START:END` (end exclusive).
        #[arg(long)]
        layers: Option<String>,
        /// Routed experts to probe per MoE layer, from index 0. Each
        /// one costs three more forward passes per layer.
        #[arg(long, default_value_t = 1)]
        experts: usize,
        /// How many rows of the ranked table to print.
        #[arg(long, default_value_t = 20)]
        top: usize,
    },
    Bench {
        /// Real GGUF to benchmark. With this set, `bench` becomes a
        /// `llama-bench` work-alike (pp/tg on real weights) and the
        /// synthetic matvec microbenchmark below is skipped.
        #[arg(short = 'm', long)]
        model: Option<String>,
        /// Prompt tokens for the `pp<N>` batched-prefill row (`0` skips).
        #[arg(short = 'p', long = "n-prompt", default_value_t = 512)]
        n_prompt: usize,
        /// Decode steps for the `tg<N>` row (`0` skips).
        #[arg(short = 'n', long = "n-gen", default_value_t = 128)]
        n_gen: usize,
        /// Timed repetitions per row; one extra warmup is discarded.
        #[arg(short = 'r', long = "repetitions", default_value_t = 3)]
        reps: usize,
        /// CPU threads (`0` = performance-core default)
        #[arg(short = 't', long, default_value_t = 0)]
        threads: usize,
        /// GPU layers: `0` forces CPU, anything else offloads
        #[arg(long = "n-gpu-layers", default_value_t = 0)]
        n_gpu_layers: usize,
        /// Context size (`0` = GGUF default)
        #[arg(short = 'c', long, default_value_t = 0)]
        ctx_size: usize,
        #[arg(long, default_value_t = 4096)]
        hidden: usize,
        #[arg(long, default_value_t = 14336)]
        ffn_dim: usize,
        #[arg(long, default_value_t = 20)]
        iters: usize,
        /// Also run `llama-bench` on the same GGUF and print the gap.
        #[arg(long)]
        compare: bool,
        /// Run every entry in `benchmarks/suite.json` (each in a fresh
        /// child process), write engine receipts, then re-render the
        /// engine table in `benchmarks/RESULTS.md`.
        #[arg(long)]
        suite: bool,
        /// Re-render the engine table from existing receipts only.
        #[arg(long)]
        render: bool,
        /// Restrict `--suite` to one suite id.
        #[arg(long)]
        id: Option<String>,
        /// Restrict `--suite` to one backend.
        #[arg(long)]
        backend: Option<String>,
        /// Skip suite entries whose estimated RAM exceeds ~75% of host RAM.
        #[arg(long)]
        fit_host: bool,
        /// Skip suite entries whose GGUF is not present.
        #[arg(long)]
        skip_missing: bool,
        /// benchmarks/ directory.
        #[arg(long, default_value = "benchmarks")]
        bench_dir: String,
        /// Internal (`--suite` children): suite id to record in the receipt.
        #[arg(long)]
        suite_id: Option<String>,
        /// Internal (`--suite` children): backend label for the receipt.
        #[arg(long, default_value = "cpu")]
        backend_label: String,
        /// Write a JSON receipt to this path.
        #[arg(long)]
        receipt: Option<String>,
        /// Refuse to start a timed run when the host's 1-minute load
        /// average is at or above this. `0` disables the check and
        /// marks the receipt as not quiet-host.
        #[arg(long, default_value_t = host_state::DEFAULT_MAX_LOAD)]
        max_load: f64,
    },
    /// Demonstrate prompt-lookup speculative decoding on a repetitive
    /// prompt, reporting forward_batch call count vs. tokens produced.
    /// Uses synthetic (random) weights -- see the printed caveat about
    /// why hit rate is not representative of a real trained model.
    Speculative {
        #[arg(default_value = "glm-5.2")]
        preset: String,
        #[arg(
            long,
            default_value = "the cat sat on the mat. the cat sat on the mat. the cat sat on the"
        )]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 4)]
        ngram_size: usize,
        #[arg(long, default_value_t = 8)]
        max_draft_len: usize,
    },
    /// Load a real Kimi K3 checkpoint directory (a real
    /// `model.safetensors.index.json` + shards, plus a real
    /// `tiktoken.model` and optionally `tokenizer_config.json`) and
    /// generate real text from a prompt. This is the only command that
    /// runs Kimi K3's real, complete inference path end to end --
    /// loading (`kimi_loader::load_kimi_checkpoint`), the dedicated
    /// forward pass (`kimi_decoder`), the real tokenizer
    /// (`kimi_tokenizer`), and sampling all wired together. Not
    /// runnable against the actual published checkpoint in this
    /// project's development environment (96 shards, 1.56TB even
    /// MXFP4-compressed; see docs/MODELS.md) -- but real
    /// code, tested end to end against small synthetic checkpoints, for
    /// anyone with the real hardware/storage to point it at real
    /// weights.
    RunKimi {
        /// Directory containing model.safetensors.index.json, its
        /// shard files, tiktoken.model, and (optionally)
        /// tokenizer_config.json.
        checkpoint_dir: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        #[arg(long, default_value_t = 1.0)]
        repetition_penalty: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

fn preset_by_name(name: &str) -> anyhow::Result<ModelConfig> {
    match name {
        "glm-5.2" | "glm5.2" | "glm" => Ok(glm_5_2()),
        "deepseek-v4-pro" | "deepseek" | "dsv4" => Ok(deepseek_v4_pro()),
        "kimi-k3" | "kimi" => Ok(kimi_k3()),
        other => {
            anyhow::bail!("unknown preset '{other}'; expected glm-5.2, deepseek-v4-pro, or kimi-k3")
        }
    }
}

/// Seeds `RAYON_NUM_THREADS` for subcommands that never reach
/// `apply_backend_env` / `bench_model::apply_env` (both of which build
/// the pool explicitly via [`frink_core::threads::init_cpu_pool`] once
/// their `-t` is known). Subcommand flags parsed later still win, since
/// those paths overwrite the variable before the pool is built.
///
/// This used to be `available_parallelism() / 2`, which on a 6P+4E M2
/// Pro guessed 5 -- close enough to look right, wrong enough to make
/// every default-config CPU measurement run one core short.
fn init_rayon_threads() {
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        let n = frink_core::threads::resolve_cpu_threads();
        // SAFETY: single-threaded init before worker threads spawn.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", n.to_string()) };
    }
}

/// What `frink serve` says when the binary was built without the
/// `serve` feature.
///
/// Names the fix rather than the symptom: the subcommand is real, the
/// server is simply not linked into this build, and "unrecognized
/// subcommand" would send the reader looking for a version that has it.
#[cfg(not(feature = "serve"))]
const SERVE_FEATURE_MISSING: &str = "\
this frink was built without the `serve` feature, so it has no HTTP server.
Rebuild with it:  cargo install frink-cli   (serve is on by default)
Or run the standalone binary:  frink-server -m model.gguf
(This only happens in a build that passed --no-default-features.)";

/// Every subcommand `Commands` declares, as clap will name it.
///
/// This list is what tells [`rewrite_llama_style_argv`] that `frink
/// serve …` is a subcommand and not llama.cpp-style top-level flags.
/// A subcommand missing from here does not fail loudly -- it is silently
/// rewritten into `frink run …` and starts a *completion*. Adding a
/// variant to `Commands` therefore means adding it here too, which
/// `every_clap_subcommand_survives_the_argv_rewriter` enforces against
/// clap's own subcommand list.
const SUBCOMMANDS: &[&str] = &[
    "run",
    // llama.cpp's launcher spells `run` as `cli`, and the two
    // informational commands it offers beside it. All three have to be
    // here or the rewriter turns `frink version` into a completion
    // whose prompt is the word "version".
    "cli",
    "version",
    "licenses",
    "pull",
    "download",
    "chat",
    "serve",
    "serve-bench",
    "bench-bw",
    "inspect",
    "inspect-plan",
    "presets",
    "archs",
    "caps",
    "smoke",
    "run-real",
    "bench",
    "batched-bench",
    "verify",
    "layer-divergence",
    "quant-sensitivity",
    "quantize",
    "imatrix",
    "gguf-split",
    "splice-pooler",
    "parity",
    "perplexity",
    "speculative",
    "run-kimi",
    "help",
];

// A backend feature on `frink` has to reach the server it links, or one
// binary answers two different ways: `frink run --device metal` uses the
// GPU and `frink serve --device metal` refuses with "built without
// --features metal". The forwarding is one `frink-server?/metal` in
// Cargo.toml and dropping it still compiles, so it is asserted here
// instead of being rechecked by hand at release time.
#[cfg(all(feature = "serve", feature = "metal"))]
const _: () = assert!(
    frink_server::BUILT_WITH_METAL,
    "frink-cli's `metal` feature must forward to frink-server: \
     metal = [.., \"frink-server?/metal\"]"
);
#[cfg(all(feature = "serve", feature = "cuda"))]
const _: () = assert!(
    frink_server::BUILT_WITH_CUDA,
    "frink-cli's `cuda` feature must forward to frink-server: \
     cuda = [.., \"frink-server?/cuda\"]"
);

/// Root flags that take no value and may legally appear *before* a
/// subcommand, because clap declares them `global = true`.
const GLOBAL_FLAGS: &[&str] = &["--allow-multiple-instances"];

/// Rewrite `frink -m …` into `frink run -m …` so llama.cpp-style
/// top-level flags work without typing the `run` subcommand.
fn rewrite_llama_style_argv(args: Vec<String>) -> Vec<String> {
    let args: Vec<String> = args
        .into_iter()
        .map(|arg| match arg.as_str() {
            "-ngl" => "--n-gpu-layers".into(),
            "-dev" => "--device".into(),
            // llama.cpp's parser is hand-written, so `-hf` is ONE
            // token. clap reads it as `-h` plus `f` and prints help,
            // which is what `frink serve -hf repo:Q4_K_M` did: it
            // looked like the flag did not exist.
            "-hf" => "--hf-repo".into(),
            "-hff" => "--hf-file".into(),
            // `llama-batched-bench`'s own spellings, for `batched-bench`.
            "-npp" => "--n-pp".into(),
            "-ntg" => "--n-tg".into(),
            "-npl" => "--n-pl".into(),
            "-pps" => "--pp-shared".into(),
            "-tgs" => "--tg-separate".into(),
            "-ub" => "--ubatch-size".into(),
            "-kvu" => "--kv-unified".into(),
            "-fa" => "--flash-attn".into(),
            "-tb" => "--threads-batch".into(),
            // llama.cpp 0.4's `-st` / `--single-turn`. Like `-hf` it is
            // one token to its hand-written parser and two short flags
            // to clap, and `-s` is already `--seed` here, so it cannot
            // be a clap short at all.
            //
            // It does NOT mean `--no-cnv`: `-st` runs the conversation
            // for one turn with the chat template applied, where
            // `--no-cnv` skips the template entirely. An early version
            // of this row mapped them together and changed the answer.
            "-st" => "--single-turn".into(),
            _ => arg,
        })
        .collect();
    if args.len() < 2 {
        return args;
    }
    // A `global = true` flag is allowed to precede the subcommand it
    // applies to, so `frink --allow-multiple-instances serve` has to
    // find `serve` at position 2. Without this skip the rewriter sees a
    // flag, assumes llama.cpp style, and produces `frink run
    // --allow-multiple-instances serve …`, which dies on "unexpected
    // argument 'serve'". Only value-less root flags belong in this list
    // -- one that took a value would make position 2 its value, not a
    // subcommand.
    let first_word = args
        .iter()
        .skip(1)
        .position(|a| !GLOBAL_FLAGS.contains(&a.as_str()))
        .map(|offset| offset + 1)
        .unwrap_or(1);
    let first = args[first_word].as_str();
    if SUBCOMMANDS.contains(&first)
        || first == "-h"
        || first == "--help"
        || first == "-V"
        || first == "--version"
    {
        return args;
    }
    let mut out = Vec::with_capacity(args.len() + 1);
    out.push(args[0].clone());
    out.push("run".into());
    out.extend(args.into_iter().skip(1));
    out
}

/// Registers this process in the instance registry when the command is
/// about to load a model, and refuses to start if another live frink
/// already holds one.
///
/// Header-only commands (`inspect`, `inspect-plan`, `presets`, `archs`,
/// `caps`), the HTTP client (`chat`), the downloader (`pull`) and
/// `bench --suite` / `--render` are deliberately exempt: none of them
/// puts weights in memory, and `--suite` is a supervisor whose children
/// each register on their own.
/// How this command should identify itself in the instance registry, or
/// `None` when it must not register at all.
///
/// `serve` is the interesting `None`: it *does* load a model, but
/// `frink_server::run_server` registers itself as `"server"` before it
/// binds. Registering here too would put two guards on one pid, and
/// since both name the same registry file, the inner one's `Drop` would
/// delete the entry while the server was still holding the weights,
/// making a live server invisible to the next `frink run`.
fn instance_target(command: &Commands) -> Option<(&'static str, Option<String>)> {
    match command {
        Commands::Run(a) => Some(("run", a.model.clone())),
        // Only the `--emit` child loads weights; the parent spawns one
        // child per backend and compares what they print. A parent that
        // registered would be refused by its own first child, which is
        // exactly what `verify` did on an idle host until this line
        // grew the `emit` test.
        Commands::Verify { model, emit, .. } => emit.then(|| ("verify", Some(model.clone()))),
        // Only the `--emit` child of `layer-divergence` loads weights;
        // the parent is a supervisor, exactly like `bench --suite`. If
        // it registered, its own children would be refused by the
        // one-model-per-host rule it exists to run under.
        Commands::LayerDivergence { model, emit, .. } => {
            emit.then(|| ("layer-divergence", Some(model.clone())))
        }
        Commands::QuantSensitivity { model, .. } => {
            Some(("quant-sensitivity", Some(model.clone())))
        }
        // Registered for the same reason `quant-sensitivity` is: it holds
        // a whole checkpoint resident for as long as the corpus takes,
        // which is the case the one-model-per-host rule exists for. It is
        // not a timing, so `--allow-multiple-instances` costs nothing but
        // the words.
        Commands::Perplexity { model, .. } => Some(("perplexity", Some(model.clone()))),
        Commands::Bench {
            model,
            suite,
            render,
            ..
        } => {
            if *suite || *render || model.is_none() {
                return None;
            }
            Some(("bench", model.clone()))
        }
        Commands::BatchedBench(args) => Some(("batched-bench", Some(args.model.clone()))),
        Commands::Smoke { preset, .. } => Some(("smoke", Some(preset.clone()))),
        Commands::RunKimi { checkpoint_dir, .. } => {
            Some(("run-kimi", Some(checkpoint_dir.clone())))
        }
        _ => None,
    }
}

fn claim_instance(cli: &Cli) -> anyhow::Result<Option<frink_core::instance::InstanceGuard>> {
    use frink_core::instance::{register, InstancePolicy};
    let Some((command, model)) = instance_target(&cli.command) else {
        return Ok(None);
    };
    // The flag is an explicit opt-in, so it wins outright; the env var
    // only decides when the flag was not passed.
    let policy = if cli.allow_multiple_instances {
        InstancePolicy::Multi
    } else {
        InstancePolicy::from_env_or(InstancePolicy::Single)
    };
    match register(
        command,
        model.as_deref(),
        bench_model::active_backend(),
        policy,
    ) {
        Ok(guard) => Ok(Some(guard)),
        Err(conflict) => Err(anyhow::anyhow!("{conflict}")),
    }
}

fn main() -> anyhow::Result<()> {
    init_rayon_threads();
    tracing_subscriber::fmt::init();
    let cli = Cli::parse_from(rewrite_llama_style_argv(std::env::args().collect()));

    // BEFORE `claim_instance`, and that ordering is the whole point.
    //
    // The backend decision is cached in a `OnceLock` the first time
    // anything asks for it, and `claim_instance` asks: it records which
    // backend this process holds. So the old ordering froze the answer
    // while `FRINK_METAL` was still unset, and the `--n-gpu-layers 0`
    // that `bench` applies later could not change it. Every `cpu` row
    // in `benchmarks/RESULTS.md` was measured on Metal because of this,
    // and every one of those receipts recorded `backend_active:
    // "Metal"` next to `backend: "cpu"` without anything comparing the
    // two (#126).
    match &cli.command {
        Commands::Bench {
            threads,
            n_gpu_layers,
            suite,
            render,
            ..
        } => {
            // `--suite` and `--render` do not benchmark in THIS process:
            // the suite spawns a child per entry and render only reads
            // receipts, so applying a backend here would pin the parent
            // to one backend for children that each want their own.
            if !suite && !render {
                bench_model::apply_env(*threads, *n_gpu_layers)?;
            }
        }
        // Same ordering constraint, same function: the batched bench
        // writes the same kind of receipt and is refused the same way
        // when the label and the backend disagree.
        Commands::BatchedBench(args) => bench_model::apply_env(args.threads, args.n_gpu_layers)?,
        _ => {}
    }

    // Held for the whole run: dropping it deregisters this process.
    let _instance = claim_instance(&cli)?;
    // Read before `cli.command` is moved into the match. Subcommands
    // that re-invoke this binary have to hand the child the same
    // decision the parent was given, and a clap flag -- unlike an
    // environment variable -- is not inherited.
    let allow_multiple_instances = cli.allow_multiple_instances;

    match cli.command {
        Commands::Run(args) => run::run_infer(args)?,
        Commands::Version => print!("{}", crate::cli_output::version_block()),
        Commands::Licenses => print!("{}", crate::cli_output::licenses_block()?),
        Commands::Chat(args) => chat::run_chat(args)?,
        Commands::ServeBench(args) => serve_bench::run_serve_bench(args)?,
        Commands::BatchedBench(args) => batched_bench::run(args)?,
        Commands::BenchBw(args) => bench_bw::run_bench_bw(args)?,
        // Blocking, and it builds its own Tokio runtime: nothing above
        // this point has started one. It also claims the instance
        // registry itself (as `server`), which is why `instance_target`
        // deliberately leaves `serve` alone.
        #[cfg(feature = "serve")]
        Commands::Serve(args) => frink_server::run_server(args)?,
        #[cfg(not(feature = "serve"))]
        Commands::Serve { .. } => anyhow::bail!(SERVE_FEATURE_MISSING),
        Commands::Pull(args) => pull::run_pull(args)?,
        Commands::Download(args) => download::run(args)?,
        Commands::Quantize(args) => quantize::run(args)?,
        Commands::Imatrix(args) => imatrix::run(args)?,
        Commands::GgufSplit(args) => gguf_split::run(args)?,
        Commands::SplicePooler(args) => splice_pooler::run(args)?,
        Commands::Inspect { path } => {
            let file = ShardedGguf::open(&path)?;
            if file.shard_count() > 1 {
                println!("Split GGUF: {} shards", file.shard_count());
                for p in file.shard_paths() {
                    println!("  {}", p.display());
                }
            }
            if let Some(name) = file.metadata_str("general.name") {
                println!("Model name: {name}");
            }
            println!("Tensor count: {}", file.tensor_count());
            for (i, (_, t)) in file.tensors().enumerate() {
                if i >= 20 {
                    println!("  ... and {} more", file.tensor_count() - 20);
                    break;
                }
                println!("  {:<40} {:?} {:?}", t.name, t.shape, t.dtype);
            }
        }
        Commands::InspectPlan {
            path,
            context,
            concurrency,
            expert_cache_bytes,
            backend,
            ctk,
            strict,
        } => {
            let budget = frink_models::DeviceBudget::detect(backend);
            let ctx_cap = frink_gguf::ShardedGguf::open(&path)
                .ok()
                .and_then(|f| {
                    let arch = f.metadata_str("general.architecture")?.to_string();
                    f.metadata_u64(&format!("{arch}.context_length"))
                })
                .unwrap_or(4096) as usize;
            let assumptions = frink_models::residency_report::ResidencyAssumptions {
                // `auto` still needs *a* context to price the plan's KV
                // line; the model's own is the honest placeholder, and
                // the chosen number is reported separately below.
                context_tokens: context.unwrap_or(ctx_cap),
                concurrent_requests: concurrency,
                expert_cache_bytes,
                kv_elem: frink_models::KvElem::from_ctk(&ctk),
                ..Default::default()
            };
            let report = frink_models::residency_report::ResidencyReport::from_gguf(
                &path,
                assumptions,
                budget.usable_bytes,
            )?;
            println!("{budget}");
            println!("{report}");
            let fit = report.auto_context(ctx_cap);
            println!("  {fit}");
            println!("  {}", budget.caveat());
            if strict {
                if let Err(e) = report.check_strict() {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Presets => {
            for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
                println!("{}", cfg.name);
                println!(
                    "  layers={} hidden={} heads={}/{} (q/kv)",
                    cfg.n_layers, cfg.hidden_dim, cfg.n_heads, cfg.n_kv_heads
                );
                println!(
                    "  experts: {} total, {} active/token, {} shared",
                    cfg.moe.n_experts, cfg.moe.n_experts_active, cfg.moe.n_shared_experts
                );
                println!(
                    "  best-effort / unconfirmed fields: {:?}",
                    cfg.best_effort_fields
                );
                println!();
            }
        }
        Commands::Archs { write } => {
            let report = frink_models::coverage_report_markdown();
            if let Some(path) = write {
                std::fs::write(&path, &report)?;
                println!("wrote architecture coverage manifest to {path}");
            } else {
                print!("{report}");
            }
        }
        Commands::Caps => {
            let profile = frink_cuda::HardwareProfile::detect();
            println!("Hardware capabilities (detected, not assumed):");
            println!("  CPU logical cores : {}", profile.cpu_logical_cores);
            if profile.host_ram_total_bytes > 0 {
                println!(
                    "  Host RAM total    : {:.2} GiB",
                    profile.host_ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            } else {
                println!("  Host RAM total    : could not detect (non-Linux host?)");
            }
            println!("  SIMD              : {}", profile.simd.label());
            println!(
                "    avx2={} avx512f={} fma={} neon={}",
                profile.simd.avx2, profile.simd.avx512f, profile.simd.fma, profile.simd.neon
            );
            if profile.cuda_available {
                println!(
                    "  CUDA              : {} device(s), first = {:?}, {:.2} GiB VRAM",
                    profile.cuda_device_count,
                    profile.cuda_device_name,
                    profile.cuda_vram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            } else {
                println!("  CUDA              : not available (no device found, or built without --features cuda)");
            }
            let metal = frink_metal::MetalProfile::detect();
            if metal.available {
                println!("  Metal             : {:?}", metal.device_name);
            } else {
                println!("  Metal             : not available (no device found, or built without --features metal)");
            }
        }
        Commands::Smoke { preset, steps } => {
            let base_cfg = preset_by_name(&preset)?;
            println!(
                "Running smoke test for '{}' (small synthetic weights, {} steps)",
                base_cfg.name, steps
            );

            let mut cfg = base_cfg;
            cfg.hidden_dim = 32;
            cfg.n_heads = 4;
            cfg.n_kv_heads = 2;
            cfg.head_dim = 8;
            cfg.moe.hidden_dim = 32;
            cfg.moe.n_experts = cfg.moe.n_experts.min(16);
            cfg.moe.expert_ffn_dim = 16;

            let vocab = 32;
            let decoder = Decoder::new_random_small(cfg.clone(), 2, vocab);
            let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

            for pos in 0..steps {
                let token = pos % vocab;
                let logits = decoder.forward_token(token, pos, &mut caches);
                let argmax = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                let all_finite = logits.iter().all(|v| v.is_finite());
                println!("  step {pos}: token_in={token} argmax_out={argmax} finite={all_finite}");
                if !all_finite {
                    anyhow::bail!("non-finite logits at step {pos}");
                }
            }
            println!("Smoke test passed: {steps} decode steps, all logits finite.");

            // Real expert-placement plan, driven by each expert's actual
            // resident byte size and the per-expert activation counts
            // just observed above -- not a synthetic example. Falls back
            // to a documented placeholder budget when no CUDA VRAM was
            // detected (e.g. this host, or built without --features cuda),
            // so the feature is still visible without a GPU present.
            let profile = frink_cuda::HardwareProfile::detect();
            let (budget_bytes, budget_note) = if profile.cuda_vram_total_bytes > 0 {
                (
                    profile.cuda_vram_total_bytes,
                    "detected CUDA VRAM".to_string(),
                )
            } else {
                let placeholder = 512 * 1024 * 1024;
                (
                    placeholder,
                    "no CUDA VRAM detected; using a 512 MiB placeholder budget".to_string(),
                )
            };
            let plan = decoder.layers[0].moe.placement_plan(budget_bytes);
            let on_gpu = plan.overrides.len();
            println!(
                "Layer 0 expert placement plan ({budget_note}, {:.2} MiB budget): {on_gpu}/{} experts fit on GPU",
                budget_bytes as f64 / (1024.0 * 1024.0),
                decoder.layers[0].moe.n_experts()
            );
        }
        Commands::RunReal {
            path,
            token,
            pos,
            fixture,
        } => {
            let cfg = match fixture.as_str() {
                "dense" => test_dense_fixture(),
                "moe" => frink_models::config::test_moe_fixture(),
                "mixed" => frink_models::config::test_mixed_fixture(),
                other => {
                    anyhow::bail!("unknown fixture '{other}'; expected 'dense', 'moe', or 'mixed'")
                }
            };
            let decoder = Decoder::from_gguf(&path, cfg)?;
            let mut caches: Vec<frink_core::cache::KvCache> = decoder.config.new_kv_caches();
            let logits = decoder.forward_token(token, pos, &mut caches);
            println!("logits ({} values):", logits.len());
            for (i, v) in logits.iter().enumerate() {
                println!("  [{i}] {v:.6}");
            }
        }
        Commands::Verify {
            model,
            backend,
            emit,
            prompt_tokens,
            prompt,
        } => {
            return verify::run(verify::VerifyArgs {
                model,
                backend,
                emit,
                prompt_tokens,
                prompt,
                allow_multiple_instances,
            });
        }
        Commands::Parity {
            model,
            prompt,
            prompt_tokens,
            top_k,
            dumper,
            dump_logits,
        } => {
            return parity::run(parity::ParityArgs {
                model,
                prompt,
                prompt_tokens,
                top_k,
                dumper,
                dump_logits,
            });
        }
        Commands::Perplexity {
            model,
            file,
            ctx_size,
            chunks,
        } => {
            return perplexity::run(perplexity::PerplexityArgs {
                model,
                file,
                ctx_size,
                chunks,
            });
        }
        Commands::LayerDivergence {
            model,
            backend,
            prompt,
            prompt_tokens,
            tol,
            at,
            emit,
        } => {
            return layer_divergence::run(layer_divergence::DivergenceArgs {
                model,
                backend,
                prompt,
                prompt_tokens,
                tol,
                at,
                emit,
                allow_multiple_instances,
            });
        }
        Commands::QuantSensitivity {
            model,
            prompt,
            prompt_tokens,
            candidate,
            layers,
            experts,
            top,
        } => {
            return quant_sensitivity::run(quant_sensitivity::QuantSensitivityArgs {
                model,
                prompt,
                prompt_tokens,
                candidate,
                layers,
                experts,
                top,
            });
        }
        Commands::Bench {
            model,
            n_prompt,
            n_gen,
            reps,
            threads,
            n_gpu_layers,
            ctx_size,
            hidden,
            ffn_dim,
            iters,
            compare,
            suite,
            render,
            id,
            backend,
            fit_host,
            skip_missing,
            bench_dir,
            suite_id,
            backend_label,
            receipt,
            max_load,
        } => {
            if render {
                return bench_render::render(std::path::Path::new(&bench_dir));
            }
            if suite {
                return bench_suite::run_suite(bench_suite::SuiteArgs {
                    bench_dir: bench_dir.into(),
                    n_prompt,
                    n_gen,
                    reps,
                    only_id: id,
                    only_backend: backend,
                    fit_host,
                    skip_missing,
                    max_load,
                });
            }
            if let Some(model) = model {
                // Already applied above, before `claim_instance` could
                // freeze the backend. Kept as a no-op call rather than
                // deleted so a future caller of this arm cannot get an
                // unconfigured process.
                bench_model::apply_env(threads, n_gpu_layers)?;
                return bench_model::run(bench_model::BenchArgs {
                    model,
                    n_prompt,
                    n_gen,
                    reps,
                    ctx_size,
                    compare,
                    backend: backend_label,
                    receipt: receipt.map(Into::into),
                    id: suite_id,
                    max_load,
                });
            }
            use frink_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
            use std::time::Instant;

            println!(
                "=== matvec microbenchmark: [{ffn_dim} x {hidden}] weight, {iters} iterations ==="
            );

            let mut rng_state: u64 = 12345;
            let mut next = || {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                ((rng_state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            };
            let weights: Vec<f32> = (0..ffn_dim * hidden).map(|_| next() * 0.1).collect();
            let x: Vec<f32> = (0..hidden).map(|_| next() * 0.1).collect();

            let f32_matrix = WeightMatrix::F32(frink_core::tensor::Tensor::new(
                weights.clone(),
                vec![ffn_dim, hidden],
            ));
            let mut packed = Vec::new();
            for row in weights.chunks(hidden) {
                packed.extend(frink_quant::quantize_q8_0(row));
            }
            let packed_for_scalar_bench = packed.clone();
            let quant_matrix = WeightMatrix::Quantized {
                data: WeightBytes::Owned(packed),
                rows: ffn_dim,
                cols: hidden,
                kind: QuantKind::Q8_0,
            };

            // warm-up
            let _ = f32_matrix.apply(&x);
            let _ = quant_matrix.apply(&x);

            let t0 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(f32_matrix.apply(&x));
            }
            let f32_elapsed = t0.elapsed();

            let t1 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(quant_matrix.apply(&x));
            }
            let quant_elapsed = t1.elapsed();

            // This isolates the *scalar* kernel cost as a single-
            // threaded direct loop, to compare against "dispatched"
            // below (which goes through WeightMatrix::apply, i.e. both
            // rayon row-parallelism AND AVX2/FMA SIMD dispatch if the
            // host supports it). On an AVX2-capable x86_64 host, most
            // of that gap is genuinely SIMD; on an ARM host without a
            // NEON kernel for this format, the "dispatched" path
            // silently falls back to the same scalar kernel per row,
            // so the entire gap here is multi-core parallelism, not
            // SIMD -- confirmed by comparing this against the explicit
            // 1-thread-vs-all-cores section below, which isolates
            // parallelism on its own. Don't read this ratio as "SIMD
            // speedup" without checking which case you're in.
            let row_bytes =
                (hidden / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES;
            let scalar_rows: Vec<&[u8]> = packed_for_scalar_bench.chunks_exact(row_bytes).collect();
            let t2 = Instant::now();
            for _ in 0..iters {
                let mut acc = 0f32;
                for row in &scalar_rows {
                    acc += frink_quant::dot_q8_0_f32_scalar(row, &x);
                }
                std::hint::black_box(acc);
            }
            let scalar_elapsed = t2.elapsed();

            let f32_bytes = f32_matrix.resident_bytes();
            let quant_bytes = quant_matrix.resident_bytes();

            println!(
                "  f32 dequant-resident matmul : {:>8.3} ms/call  ({} bytes resident)",
                f32_elapsed.as_secs_f64() * 1000.0 / iters as f64,
                f32_bytes
            );
            println!(
                "  fused Q8_0 (scalar, no SIMD, single-threaded) : {:>8.3} ms/call",
                scalar_elapsed.as_secs_f64() * 1000.0 / iters as f64
            );
            println!(
                "  fused Q8_0 (dispatched, row-parallel)          : {:>8.3} ms/call  ({} bytes resident)  <- rayon row parallelism + AVX2/FMA (x86_64) or NEON (aarch64) if host supports it",
                quant_elapsed.as_secs_f64() * 1000.0 / iters as f64,
                quant_bytes
            );
            println!(
                "  dispatched is {:.2}x the single-threaded scalar path's time (see \"multi-core scaling\" below to separate the SIMD and parallelism contributions to this number)",
                quant_elapsed.as_secs_f64() / scalar_elapsed.as_secs_f64()
            );
            println!(
                "  memory reduction: {:.2}x smaller resident",
                f32_bytes as f64 / quant_bytes as f64
            );
            println!(
                "  speed ratio: fused Q8_0 (dispatched) is {:.2}x the f32 path's time (<1.0 = fused is faster; with AVX2+FMA this is usually both faster AND smaller, not a memory-for-speed tradeoff)",
                quant_elapsed.as_secs_f64() / f32_elapsed.as_secs_f64()
            );

            println!("\n=== Q4_0 matvec microbenchmark: [{ffn_dim} x {hidden}] weight, {iters} iterations ===");
            let q4_row_bytes =
                (hidden / frink_quant::Q4_0_BLOCK_ELEMS) * frink_quant::Q4_0_BLOCK_BYTES;
            let mut q4_packed = Vec::with_capacity(ffn_dim * q4_row_bytes);
            for row in weights.chunks(hidden) {
                for block in row.chunks(frink_quant::Q4_0_BLOCK_ELEMS) {
                    let amax = block.iter().fold(0f32, |a, &b| a.max(b.abs()));
                    let scale = if amax == 0.0 { 1.0 } else { amax / 7.0 };
                    q4_packed.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                    for i in 0..16 {
                        let lo = ((block.get(i).copied().unwrap_or(0.0) / scale)
                            .round()
                            .clamp(-8.0, 7.0) as i32
                            + 8) as u8;
                        let hi = ((block.get(i + 16).copied().unwrap_or(0.0) / scale)
                            .round()
                            .clamp(-8.0, 7.0) as i32
                            + 8) as u8;
                        q4_packed.push(lo | (hi << 4));
                    }
                }
            }
            let q4_rows: Vec<&[u8]> = q4_packed.chunks_exact(q4_row_bytes).collect();

            let t3 = Instant::now();
            for _ in 0..iters {
                let mut acc = 0f32;
                for row in &q4_rows {
                    acc += frink_quant::dot_q4_0_f32_scalar(row, &x);
                }
                std::hint::black_box(acc);
            }
            let q4_scalar_elapsed = t3.elapsed();

            // Route through WeightMatrix::apply, not a direct
            // dot_q4_0_f32 loop -- same rayon row-parallel dispatch the
            // Q8_0 "dispatched" number above uses. An earlier version of
            // this benchmark called dot_q4_0_f32 directly in a plain
            // sequential loop here, which meant it never exercised
            // multi-core parallelism at all and wasn't measuring the
            // same thing as the Q8_0 number above -- caught by actually
            // comparing RAYON_NUM_THREADS=1 vs default on real multi-
            // core hardware and finding the two numbers didn't move
            // together the way they should have.
            let q4_quant_matrix = WeightMatrix::Quantized {
                data: WeightBytes::Owned(q4_packed.clone()),
                rows: ffn_dim,
                cols: hidden,
                kind: QuantKind::Q4_0,
            };
            let _ = q4_quant_matrix.apply(&x); // warm-up
            let t4 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(q4_quant_matrix.apply(&x));
            }
            let q4_dispatched_elapsed = t4.elapsed();

            println!(
                "  Q4_0 (scalar, no SIMD, single-threaded) : {:>8.3} ms/call  ({} bytes resident)",
                q4_scalar_elapsed.as_secs_f64() * 1000.0 / iters as f64,
                q4_packed.len()
            );
            println!(
                "  Q4_0 (dispatched, row-parallel)          : {:>8.3} ms/call  <- rayon row parallelism + AVX2/FMA (x86_64) or NEON (aarch64) if host supports it",
                q4_dispatched_elapsed.as_secs_f64() * 1000.0 / iters as f64
            );
            println!(
                "  Q4_0 dispatched is {:.2}x the single-threaded scalar path's time",
                q4_dispatched_elapsed.as_secs_f64() / q4_scalar_elapsed.as_secs_f64()
            );
            println!(
                "  Q4_0 memory reduction vs f32: {:.2}x smaller resident",
                f32_bytes as f64 / q4_packed.len() as f64
            );

            println!("\n=== multi-core scaling: fused Q8_0 matvec, [{ffn_dim} x {hidden}], {iters} iterations ===");
            let available_cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let single_thread_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("building a 1-thread rayon pool must not fail");
            let t5 = Instant::now();
            single_thread_pool.install(|| {
                for _ in 0..iters {
                    std::hint::black_box(quant_matrix.apply(&x));
                }
            });
            let single_thread_elapsed = t5.elapsed();
            let single_thread_ms = single_thread_elapsed.as_secs_f64() * 1000.0 / iters as f64;
            let all_cores_ms = quant_elapsed.as_secs_f64() * 1000.0 / iters as f64;
            println!("  available cores (std::thread::available_parallelism): {available_cores}");
            println!("  1 thread            : {single_thread_ms:>8.3} ms/call");
            println!("  {available_cores} threads (default) : {all_cores_ms:>8.3} ms/call");
            println!(
                "  measured speedup: {:.2}x (ideal linear speedup would be {:.2}x for {} cores)",
                single_thread_ms / all_cores_ms,
                available_cores,
                available_cores
            );

            println!("\n=== end-to-end decode throughput (synthetic weights, capped layer count for sandbox runtime) ===");
            for (name, preset) in [
                ("glm-5.2", glm_5_2()),
                ("deepseek-v4-pro", deepseek_v4_pro()),
                ("kimi-k3", kimi_k3()),
            ] {
                let mut cfg = preset;
                cfg.hidden_dim = 512;
                cfg.n_heads = 8;
                cfg.n_kv_heads = 2;
                cfg.head_dim = 64;
                cfg.moe.hidden_dim = 512;
                cfg.moe.n_experts = cfg.moe.n_experts.min(32);
                cfg.moe.expert_ffn_dim = 512;
                let n_layers = 4;
                let vocab = 256;

                let decoder = Decoder::new_random_small(cfg, n_layers, vocab);
                let mut caches: Vec<frink_core::cache::KvCache> = decoder.config.new_kv_caches();

                let n_tokens = 32;
                let t0 = Instant::now();
                for pos in 0..n_tokens {
                    std::hint::black_box(decoder.forward_token(pos % vocab, pos, &mut caches));
                }
                let elapsed = t0.elapsed();
                let toks_per_sec = n_tokens as f64 / elapsed.as_secs_f64();
                println!(
                    "  {name:<16} {n_layers} layers, hidden=512, {} experts ({} active): {:>7.1} tok/s ({:.2} ms/token, CPU reference, single thread pool)",
                    decoder.config.moe.n_experts,
                    decoder.config.moe.n_experts_active,
                    toks_per_sec,
                    elapsed.as_secs_f64() * 1000.0 / n_tokens as f64
                );
            }
            println!(
                "\nNote: these are CPU-reference numbers on this sandbox's vCPUs, not GPU numbers."
            );
            println!("They demonstrate the pipeline's relative cost structure (attention vs MoE FFN, memory");
            println!("footprint of quantized vs f32 weights), not absolute performance on target hardware");
            println!("like a DGX Spark. See benchmarks/RESULTS.md for methodology and caveats.");
        }
        Commands::Speculative {
            preset,
            prompt,
            max_new_tokens,
            ngram_size,
            max_draft_len,
        } => {
            let base_cfg = preset_by_name(&preset)?;
            let mut cfg = base_cfg;
            cfg.hidden_dim = 64;
            cfg.n_heads = 8;
            cfg.n_kv_heads = 2;
            cfg.head_dim = 8;
            cfg.moe.hidden_dim = 64;
            cfg.moe.n_experts = cfg.moe.n_experts.min(16);
            cfg.moe.expert_ffn_dim = 32;

            let decoder = Decoder::new_random_small(cfg, 3, 256);
            let mut caches: Vec<frink_core::cache::KvCache> = decoder.config.new_kv_caches();

            let prompt_tokens: Vec<usize> = frink_models::ByteTokenizer::encode(&prompt)
                .into_iter()
                .map(|b| b as usize)
                .collect();

            println!("Prompt: {prompt:?} ({} tokens)", prompt_tokens.len());
            println!("Speculator: ngram_size={ngram_size}, max_draft_len={max_draft_len}");
            println!(
                "NOTE: this decoder uses random synthetic weights, not a trained model. A real",
            );
            println!(
                "trained model asked to continue an obviously repeating pattern like this prompt"
            );
            println!(
                "would very likely predict the repeat correctly (that's exactly the case prompt-"
            );
            println!(
                "lookup decoding targets) -- random weights will not reliably reproduce that, so a"
            );
            println!("low or zero accept rate below is expected and does not indicate a bug.\n");

            let mut speculator =
                frink_models::PromptLookupSpeculator::new(ngram_size, max_draft_len);
            let result = frink_models::speculative_decode(
                &decoder,
                &prompt_tokens,
                max_new_tokens,
                &mut caches,
                &mut speculator,
            );

            println!("Tokens generated : {}", result.tokens_generated);
            println!("forward_batch calls: {}", result.forward_calls);
            println!(
                "Tokens per call   : {:.2} (1.00 = no speedup; higher = drafts were accepted)",
                result.tokens_per_call()
            );
            println!(
                "Acceptance length : {} (completion tokens per verification step -- the \
                 published metric, prefill excluded)",
                result
                    .acceptance_length()
                    .map(|a| format!("{a:.2}"))
                    .unwrap_or_else(|| "n/a".to_string())
            );
            println!(
                "Draft accept rate : {} ({} accepted / {} evaluated)",
                result
                    .accept_rate()
                    .map(|r| format!("{:.1}%", r * 100.0))
                    .unwrap_or_else(|| "n/a".to_string()),
                result.accepted_tokens,
                result.drafted_tokens
            );
            let per_position = result.accept_rate_per_position();
            if !per_position.is_empty() {
                // A single mean cannot tell a uniformly mediocre drafter
                // from one that is right at position 0 and useless by
                // position k; those want opposite block sizes.
                let cells: Vec<String> = per_position
                    .iter()
                    .zip(result.evaluated_at_position.iter())
                    .enumerate()
                    .map(|(i, (rate, seen))| format!("  [{i}] {:.1}% of {seen}", rate * 100.0))
                    .collect();
                println!("Per-position accept rate (conditional on reaching the position):");
                for cell in cells {
                    println!("{cell}");
                }
            }
            println!(
                "Calls saved vs. sequential decode: {} (sequential would need exactly {} calls)",
                max_new_tokens as i64 - result.forward_calls as i64,
                max_new_tokens
            );
        }
        Commands::RunKimi {
            checkpoint_dir,
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            seed,
        } => {
            use std::path::Path;

            let dir = Path::new(&checkpoint_dir);
            let index_path = dir.join("model.safetensors.index.json");
            println!(
                "Opening real Kimi K3 checkpoint index: {}",
                index_path.display()
            );
            let shard = frink_safetensors::ShardedSafetensors::open_index(&index_path)?;

            let model_cfg = kimi_k3();
            let hp = frink_models::kimi_loader::KimiRealHparams::real();
            println!(
                "Loading all {} real layers (this eagerly touches every routed expert's mmap \
                 range, but never materializes a dequantized f32 copy -- see \
                 kimi_loader's module docs)...",
                model_cfg.n_layers
            );
            let weights = frink_models::kimi_loader::load_kimi_checkpoint(&shard, &model_cfg, &hp)?;
            println!("Loaded. Vocab size: {}", weights.output_head.rows());

            let vocab_path = dir.join("tiktoken.model");
            let vocab_text = std::fs::read_to_string(&vocab_path)?;
            let ranks = frink_models::kimi_tokenizer::parse_tiktoken_vocab(&vocab_text)?;
            let tokenizer_config_path = dir.join("tokenizer_config.json");
            let special_tokens = if tokenizer_config_path.exists() {
                let text = std::fs::read_to_string(&tokenizer_config_path)?;
                frink_models::kimi_tokenizer::parse_special_tokens(&text)?
            } else {
                std::collections::HashMap::new()
            };
            let eos_id = special_tokens.get("[EOS]").copied();
            let tokenizer =
                frink_models::kimi_tokenizer::KimiTokenizer::new(ranks, special_tokens)?;
            println!(
                "Loaded real tokenizer: {} base tokens, eos_id={eos_id:?}",
                tokenizer.vocab_size()
            );

            let frink_models::config::AttentionKind::KimiHybrid(hybrid) = &model_cfg.attention
            else {
                anyhow::bail!("kimi_k3() preset must use AttentionKind::KimiHybrid");
            };
            let decoder_cfg = frink_models::kimi_decoder::KimiDecoderConfig {
                attn_res_block_size: 12,
                rms_norm_eps: model_cfg.rms_norm_eps,
                situ_beta: 4.0,
                situ_linear_beta: 25.0,
                moe: frink_models::latent_moe::KimiMoeConfig {
                    n_experts_active: model_cfg.moe.n_experts_active,
                    moe_renormalize: true,
                    routed_scaling_factor: 1.0,
                    situ_beta: 4.0,
                    situ_linear_beta: 25.0,
                    rms_norm_eps: model_cfg.rms_norm_eps,
                },
            };

            let sampling = frink_models::sampling::SamplingParams {
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                penalty_last_n: 64,
                ..Default::default()
            };

            println!("Generating (max {max_new_tokens} new tokens)...");
            let (text, ids) = frink_models::kimi_generate::kimi_generate(
                &weights,
                &decoder_cfg,
                &hybrid.mla,
                &hybrid.kda,
                &tokenizer,
                &prompt,
                max_new_tokens,
                &sampling,
                eos_id,
                seed,
            );
            println!("Generated {} tokens.", ids.len());
            println!("---");
            println!("{text}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::{rewrite_llama_style_argv, SUBCOMMANDS};

    fn rewrite(argv: &[&str]) -> Vec<String> {
        rewrite_llama_style_argv(argv.iter().copied().map(String::from).collect())
    }

    /// `try_parse_from`, not `parse_from`: clap's non-`try` parser exits
    /// the *process* on a parse error, which takes the whole test binary
    /// down with it and reports as "test exited abnormally" instead of
    /// naming the test that broke.
    fn parse_cli(argv: Vec<String>) -> super::Cli {
        use clap::Parser;
        super::Cli::try_parse_from(&argv)
            .unwrap_or_else(|e| panic!("`{}` did not parse: {e}", argv.join(" ")))
    }

    /// Every subcommand clap knows about has to be in `SUBCOMMANDS`, or
    /// the rewriter turns it into an implicit `run` and the user gets a
    /// completion instead of the command they typed. This is a silent
    /// failure -- no error, no clue -- so it is asserted structurally
    /// against clap's own list rather than against a second hand-written
    /// one that can drift the same way.
    #[test]
    fn every_clap_subcommand_survives_the_argv_rewriter() {
        use clap::CommandFactory;
        for sub in super::Cli::command().get_subcommands() {
            let name = sub.get_name().to_string();
            assert!(
                SUBCOMMANDS.contains(&name.as_str()),
                "subcommand `{name}` is missing from SUBCOMMANDS: `frink {name} …` would be \
                 rewritten into an implicit `run`"
            );
            let rewritten = rewrite(&["frink", &name, "-m", "model.gguf"]);
            assert_eq!(
                rewritten[1], name,
                "`frink {name}` must reach clap as `{name}`, not as `{}`",
                rewritten[1]
            );
        }
    }

    /// `frink quantize --help` and `Target::ALL` are two structures that
    /// must agree about which targets write, and for three PRs they did
    /// not: the help said "Q8_0, plus Q4_K_S / Q4_K_M with --pure" while
    /// the code wrote six targets, mixes included. So the help is checked
    /// against the table: every target the policy accepts is named in
    /// the subcommand's long help, and `--imatrix` is mentioned because
    /// the encoders take one.
    #[test]
    fn the_quantize_help_names_every_target_the_policy_accepts() {
        use clap::CommandFactory;
        let cmd = super::Cli::command();
        let quantize = cmd
            .get_subcommands()
            .find(|c| c.get_name() == "quantize")
            .expect("quantize subcommand");
        let help = quantize
            .get_long_about()
            .map(|s| s.to_string())
            .expect("quantize has a long help");
        for target in crate::quantize::policy::Target::ALL {
            assert!(
                help.contains(target.name()),
                "quantize's help does not mention {}, which `--type` accepts",
                target.name()
            );
        }
        assert!(
            help.contains("--imatrix"),
            "quantize's help does not mention --imatrix"
        );
        assert!(
            !help.contains("with `--pure`"),
            "quantize's help still ties the K-quants to --pure"
        );
    }

    /// The named case of the above, kept explicit because `serve` is the
    /// one that would start a *completion* while the user waited for an
    /// HTTP server: it prints tokens and exits 0, so a supervisor
    /// watching for the `frink.server.ready` line just hangs.
    #[test]
    fn serve_is_not_rewritten_into_an_implicit_run() {
        assert_eq!(
            rewrite(&["frink", "serve", "-m", "model.gguf", "--port", "0"]),
            ["frink", "serve", "-m", "model.gguf", "--port", "0"]
        );
    }

    /// The rewriter still has to translate llama.cpp's multi-character
    /// short options *inside* a `serve` invocation -- `frink-server`
    /// accepted `-ngl`/`-dev`, so `frink serve` has to as well.
    #[test]
    fn serve_keeps_the_llama_style_short_options() {
        assert_eq!(
            rewrite(&["frink", "serve", "-ngl", "all", "-dev", "metal"]),
            [
                "frink",
                "serve",
                "--n-gpu-layers",
                "all",
                "--device",
                "metal"
            ]
        );
    }

    /// `serve` must not claim a registry slot from here: the server
    /// claims its own as `"server"`, and a second guard on the same pid
    /// would deregister the live server when it dropped.
    #[test]
    fn serve_does_not_claim_the_instance_registry_from_the_cli_side() {
        let cli = parse_cli(rewrite(&["frink", "serve", "-m", "model.gguf"]));
        assert!(
            super::instance_target(&cli.command).is_none(),
            "`frink serve` must leave the registry to frink_server::run_server"
        );

        // The other half of the invariant: a completion run still
        // registers, so this test fails if `instance_target` were
        // "fixed" by making it return None for everything.
        let cli = parse_cli(rewrite(&["frink", "-m", "model.gguf"]));
        assert_eq!(
            super::instance_target(&cli.command).map(|(c, _)| c),
            Some("run")
        );
    }

    /// Flag parity with the standalone binary, asserted against
    /// `ServerArgs` itself rather than against the documented subset:
    /// `docs/CLI.md` lists eight of the eleven flags and reads as
    /// exhaustive, so a `serve` written from the docs would ship a
    /// downgrade nobody noticed.
    #[cfg(feature = "serve")]
    #[test]
    fn serve_exposes_every_flag_frink_server_has() {
        use clap::CommandFactory;
        let ids = |cmd: &clap::Command| -> std::collections::BTreeSet<String> {
            cmd.get_arguments()
                .map(|a| a.get_id().to_string())
                .filter(|id| id != "help" && id != "version")
                .collect()
        };
        let standalone = ids(&frink_server::ServerArgs::command());
        let root = super::Cli::command();
        let serve = ids(root.find_subcommand("serve").expect("serve subcommand"));
        assert!(
            standalone.is_subset(&serve),
            "`frink serve` is missing flags `frink-server` accepts: {:?}",
            standalone.difference(&serve).collect::<Vec<_>>()
        );
    }

    /// Parity of *values*, not just of flag names: every flag has to
    /// land in the same field with the same value it would have reached
    /// through `frink-server`'s own argv path. `--port 0` is in here on
    /// purpose -- it is a request for a kernel-assigned port, and a
    /// front end that dropped it would silently serve on 8383.
    #[cfg(feature = "serve")]
    #[test]
    fn serve_parses_the_same_command_line_as_frink_server() {
        use clap::Parser;
        let flags = [
            "-m",
            "model.gguf",
            "--host",
            "0.0.0.0",
            "--port",
            "0",
            "-t",
            "4",
            "--device",
            "metal",
            "--n-gpu-layers",
            "all",
            "--mcp-config",
            "mcp.json",
            "--exit-on-stdin-close",
            "--allow-multiple-instances",
        ];

        let standalone =
            frink_server::ServerArgs::try_parse_from(std::iter::once("frink-server").chain(flags))
                .expect("frink-server accepts this command line");

        let mut argv = vec!["frink".to_string(), "serve".to_string()];
        argv.extend(flags.iter().map(|f| f.to_string()));
        let cli = parse_cli(rewrite_llama_style_argv(argv));
        let super::Commands::Serve(via_cli) = cli.command else {
            panic!("`frink serve …` did not parse as the serve subcommand");
        };

        assert_eq!(via_cli, standalone);
    }

    /// `--allow-multiple-instances` is declared `global = true`, so both
    /// sides of the subcommand are legal places to type it and both have
    /// to reach the server -- if the leading form silently did nothing,
    /// the server would refuse to start on a host where the operator
    /// had already said a second instance was fine.
    ///
    /// Two mechanisms make it work, one of them not ours: the argv
    /// rewriter must not mistake the leading flag for llama.cpp-style
    /// argv (it used to, see the test below), and clap propagates the
    /// root global's *value* into the subcommand's own flag because both
    /// carry the id `allow_multiple_instances`. That second half is
    /// clap's behaviour, not a guarantee we wrote, which is exactly why
    /// it is pinned here: rename either side and this test fails while
    /// nothing else would.
    #[cfg(feature = "serve")]
    #[test]
    fn allow_multiple_instances_reaches_serve_from_either_side_of_the_subcommand() {
        use clap::Parser;
        let expected = frink_server::ServerArgs::try_parse_from([
            "frink-server",
            "-m",
            "model.gguf",
            "--allow-multiple-instances",
        ])
        .unwrap();

        for argv in [
            &[
                "frink",
                "serve",
                "-m",
                "model.gguf",
                "--allow-multiple-instances",
            ][..],
            &[
                "frink",
                "--allow-multiple-instances",
                "serve",
                "-m",
                "model.gguf",
            ][..],
        ] {
            let cli = parse_cli(rewrite(argv));
            assert!(
                cli.allow_multiple_instances,
                "`{}` did not set the root flag",
                argv.join(" ")
            );
            let super::Commands::Serve(args) = cli.command else {
                panic!("`{}` did not parse as the serve subcommand", argv.join(" "));
            };
            assert_eq!(
                args,
                expected,
                "`{}` did not reach the server as an opt-in",
                argv.join(" ")
            );
        }
    }

    /// The rewriter half of the above, stated on its own: a global flag
    /// before the subcommand must not turn the subcommand into an
    /// argument of an implicit `run`.
    #[test]
    fn a_global_flag_before_a_subcommand_does_not_trigger_the_run_rewrite() {
        assert_eq!(
            rewrite(&[
                "frink",
                "--allow-multiple-instances",
                "serve",
                "-m",
                "x.gguf"
            ]),
            [
                "frink",
                "--allow-multiple-instances",
                "serve",
                "-m",
                "x.gguf"
            ]
        );
        // …and llama.cpp-style argv behind the same flag still gets it.
        assert_eq!(
            rewrite(&["frink", "--allow-multiple-instances", "-m", "x.gguf"]),
            ["frink", "run", "--allow-multiple-instances", "-m", "x.gguf"]
        );
    }

    /// `--list-devices` is the flag that is easiest to lose, because it
    /// exits before anything is served and so no smoke test covers it.
    #[cfg(feature = "serve")]
    #[test]
    fn serve_accepts_list_devices_on_its_own() {
        use clap::Parser;
        let cli = parse_cli(rewrite(&["frink", "serve", "--list-devices"]));
        let super::Commands::Serve(args) = cli.command else {
            panic!("not the serve subcommand");
        };
        assert_eq!(
            args,
            frink_server::ServerArgs::try_parse_from(["frink-server", "--list-devices"]).unwrap()
        );
    }

    /// Without the feature the subcommand still exists and still
    /// swallows the server's flags, so the user gets the sentence that
    /// names the fix instead of clap's "unrecognized subcommand".
    #[cfg(not(feature = "serve"))]
    #[test]
    fn serve_without_the_feature_explains_itself_instead_of_erroring_out_of_clap() {
        let cli = parse_cli(rewrite(&["frink", "serve", "-m", "model.gguf"]));
        let super::Commands::Serve { args } = cli.command else {
            panic!("`frink serve …` did not parse as the serve subcommand");
        };
        assert_eq!(args, ["-m", "model.gguf"]);
        assert!(super::SERVE_FEATURE_MISSING.contains("serve is on by default"));
    }

    #[test]
    fn rewrites_llama_multi_character_short_options() {
        let args = ["frink", "-m", "model.gguf", "-dev", "none", "-ngl", "0"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            rewrite_llama_style_argv(args),
            [
                "frink",
                "run",
                "-m",
                "model.gguf",
                "--device",
                "none",
                "--n-gpu-layers",
                "0"
            ]
        );
    }

    /// **`-st` is not `--no-cnv`.**
    ///
    /// llama.cpp 0.4 dropped `--no-cnv` and offers `-st` /
    /// `--single-turn`, and the two are easy to read as synonyms.
    /// They are not: `-st` runs the CONVERSATION for one turn, with
    /// the chat template applied, where `--no-cnv` skips the template
    /// entirely.
    ///
    /// An early version of that rewriter row mapped them together,
    /// and it changed the answer -- measured on the same prompt and
    /// the same file, the templated reply is "The capital of France
    /// is Paris." and the raw one continues " Paris\nThe capital city
    /// of France is...". Nothing would have caught that but reading
    /// two outputs side by side, which is why it is a test now.
    #[test]
    fn single_turn_keeps_the_chat_template() {
        let out = rewrite(&["frink", "-m", "model.gguf", "-p", "hi", "-st"]);
        assert!(
            out.contains(&"--single-turn".to_string()),
            "-st must reach the single-turn flag: {out:?}"
        );
        assert!(
            !out.contains(&"--no-cnv".to_string()),
            "-st must NOT become --no-cnv: it keeps the chat template, \
             and mapping the two together changes the generated text: {out:?}"
        );
    }

    /// And the two flags stay distinct at the clap layer, so neither
    /// is an alias of the other.
    #[test]
    fn single_turn_and_no_cnv_are_separate_flags() {
        use clap::CommandFactory;
        let run = crate::Cli::command();
        let run = run
            .get_subcommands()
            .find(|c| c.get_name() == "run")
            .expect("run subcommand");
        let names: Vec<&str> = run
            .get_arguments()
            .filter_map(|a| a.get_long())
            .filter(|l| *l == "no-cnv" || *l == "single-turn")
            .collect();
        assert_eq!(
            names.len(),
            2,
            "both flags must exist independently, found {names:?}"
        );
    }
}
