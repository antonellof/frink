//! frink CLI, llama.cpp-style GGUF completion (`-m`/`-p`/`-n`/…) plus
//! inspect / presets / smoke / Kimi helpers. See `docs/CLI.md`.

mod argv;
mod batched_bench;
mod bench_bw;
mod bench_client;
mod bench_cmd;
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
#[cfg(test)]
mod llama_flags;
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
// Parsed once per process, so the size of the largest variant costs
// nothing worth a `Box`. Only the build without `serve` trips it, where
// the `Serve` stub is small beside `Run`.
#[allow(clippy::large_enum_variant)]
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
    /// Benchmark a GGUF like `llama-bench`, or the synthetic kernels.
    Bench(bench_cmd::BenchCmd),
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
        Commands::Bench(args) => {
            if args.suite || args.render || args.model.is_none() {
                return None;
            }
            Some(("bench", args.model.clone()))
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
    let argv = argv::rewrite_llama_style_argv(std::env::args().collect()).unwrap_or_else(|e| {
        argv::built_cli()
            .error(clap::error::ErrorKind::UnknownArgument, e)
            .exit()
    });
    let cli = Cli::parse_from(argv);

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
        Commands::Bench(args) => {
            // `--suite` and `--render` do not benchmark in THIS process:
            // the suite spawns a child per entry and render only reads
            // receipts, so applying a backend here would pin the parent
            // to one backend for children that each want their own.
            if !args.suite && !args.render {
                bench_model::apply_env(args.threads, args.n_gpu_layers, args.model.as_deref())?;
            }
        }
        // Same ordering constraint, same function: the batched bench
        // writes the same kind of receipt and is refused the same way
        // when the label and the backend disagree.
        Commands::BatchedBench(args) => {
            bench_model::apply_env(args.threads, args.n_gpu_layers, Some(&args.model))?
        }
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
        Commands::Bench(args) => return args.run(),
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
mod cli_tests;
