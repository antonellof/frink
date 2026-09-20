//! `frink-server`'s argument surface: the llama.cpp-shaped command line
//! both front ends parse, and the environment it lowers to.
//!
//! Split out of `lib.rs` under this repo's "a new file beats a new
//! section" rule, with no behaviour change: the struct, the
//! multi-character short-option rewrite (`-ngl`, `-np`, `-hf`), the
//! device/GPU-layer value types and `apply_cli_overrides` all moved
//! verbatim, along with the tests that cover them.
//!
//! One rule this module holds to, because the repo has already paid for
//! breaking it: **a flag that is accepted must reach the thing it
//! names.** `frink bench --n-gpu-layers 0` was documented as forcing
//! CPU and did not, because the backend was decided before the flag was
//! read. Every override here runs in `apply_cli_overrides`, *before*
//! the runtime and the model loader read the environment.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::str::FromStr;

use clap::{Parser, ValueEnum};

// `PartialEq` so frink-cli's serve tests can assert that both front
// ends parse a command line into the SAME arguments, rather than
// asserting field by field and missing whichever one is added next.
#[derive(Parser, Debug, PartialEq)]
// No `version` here on purpose. This struct is both `frink-server`'s
// own argv and the body of frink-cli's `serve` subcommand, and clap
// gives an embedded subcommand its own `--version` derived from the
// variant name: `frink serve --version` printed `frink-serve 0.10.0`,
// naming a binary nobody ships. The front end's own `--version` is the
// truth, and both report the same workspace version anyway.
#[command(
    name = "frink-server",
    about = "OpenAI-compatible Frink inference server"
)]
pub struct ServerArgs {
    /// Model path (GGUF file or Kimi checkpoint directory).
    #[arg(short = 'm', long = "model", value_name = "FILE")]
    model: Option<String>,

    /// Hugging Face repo to serve, `user/repo[:QUANT]`, llama.cpp's
    /// `-hf`.
    ///
    /// Downloads into the frink cache on first use and reuses it
    /// after, so `-hf TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF:Q4_K_M`
    /// is the whole command. The tag after the colon is a QUANT LABEL,
    /// not a git revision, and it matches without regard to case.
    #[arg(
        long = "hf-repo",
        visible_alias = "hf",
        value_name = "REPO[:QUANT]",
        conflicts_with = "model"
    )]
    hf_repo: Option<String>,

    /// Exact filename inside `--hf-repo`, llama.cpp's `-hff`.
    ///
    /// For a repo whose quant labels do not disambiguate, or a file
    /// whose name carries no quant at all.
    #[arg(long = "hf-file", value_name = "FILE", requires = "hf_repo")]
    hf_file: Option<String>,

    /// Context size, llama.cpp's `-c`. Sets `FRINK_CB_MAX_CONTEXT`.
    ///
    /// Unset means the ceiling is derived at load from the weights and
    /// the per-token KV against the device budget, capped at the
    /// model's trained context, which is usually what you want.
    #[arg(short = 'c', long = "ctx-size", value_name = "N")]
    ctx_size: Option<usize>,

    /// Require `Authorization: Bearer <key>`, llama.cpp's `--api-key`.
    /// Sets `FRINK_API_KEY`, which also gates `/admin`.
    #[arg(long = "api-key", value_name = "KEY")]
    api_key: Option<String>,

    /// Read the API key from a file, llama.cpp's `--api-key-file`.
    ///
    /// Preferred over `--api-key` on a shared host: an argument is
    /// visible in `ps` to every user on the machine.
    #[arg(long = "api-key-file", value_name = "PATH", conflicts_with = "api_key")]
    api_key_file: Option<std::path::PathBuf>,

    /// Name this model answers to in `/v1/models` and in responses,
    /// llama.cpp's `--alias`. Sets `FRINK_MODEL_NAME`.
    #[arg(long = "alias", visible_alias = "model-alias", value_name = "NAME")]
    alias: Option<String>,

    /// KV cache dtype, llama.cpp's `--cache-type-k`. Metal only; the
    /// CPU and CUDA KV cache is the host `Vec<f32>`.
    ///
    /// Validated against the same vocabulary `frink run` uses
    /// (`frink_models::ctk`): the two flags set the same variable, and
    /// an engine that refuses a value on one binary while silently
    /// serving f16 on the other is two answers to one question.
    #[arg(
        long = "ctk",
        visible_alias = "cache-type-k",
        value_name = "TYPE",
        value_parser = frink_models::ctk::parse_value
    )]
    ctk: Option<String>,

    /// Accepted and already the default: frink always compiles and
    /// evaluates the GGUF's own `tokenizer.chat_template`. llama.cpp
    /// needs `--jinja` to do that, so a command copied from there
    /// carries it, and dying on an unknown flag would be a worse answer
    /// than saying "yes, always".
    #[arg(long = "jinja", default_value_t = false)]
    jinja: bool,

    /// Refused rather than ignored: frink has no
    /// template-free/sniffing mode to fall back to. See `--jinja`.
    #[arg(long = "no-jinja", default_value_t = false)]
    no_jinja: bool,

    /// Accepted; frink does no warm-up pass, so there is none to skip.
    #[arg(long = "no-warmup", default_value_t = false)]
    no_warmup: bool,

    /// Accepted. Fused attention is a backend decision here, not a
    /// request-time one: it is on wherever the Metal kernels support
    /// the shape (`FRINK_METAL_ATTN`).
    #[arg(long = "flash-attn", visible_alias = "fa", value_name = "MODE", num_args = 0..=1, default_missing_value = "auto")]
    flash_attn: Option<String>,

    /// IP address to listen on.
    #[arg(long, value_name = "HOST")]
    host: Option<IpAddr>,

    /// Port to listen on. `0` asks the kernel for a free one; the
    /// actually-bound address is then announced on stdout (see
    /// [`announce_ready`]), which is how a supervising process is meant
    /// to learn it.
    #[arg(long, value_name = "PORT")]
    port: Option<u16>,

    /// CPU threads (sets FRINK_CPU_THREADS and RAYON_NUM_THREADS).
    #[arg(short = 't', long = "threads", value_name = "N")]
    threads: Option<usize>,

    /// Device used for offloading (`none` disables GPU use).
    #[arg(
        long = "device",
        visible_alias = "dev",
        value_name = "DEVICE",
        ignore_case = true
    )]
    device: Option<OffloadDevice>,

    /// Print available offload devices and exit.
    #[arg(long = "list-devices", default_value_t = false)]
    pub(crate) list_devices: bool,

    /// GPU layers: `0`, a positive number, `auto`, or `all`.
    ///
    /// Partial placement is not implemented yet; any value above zero
    /// currently enables all supported operations on the selected backend.
    #[arg(
        long = "n-gpu-layers",
        visible_aliases = ["gpu-layers", "ngl"],
        value_name = "N"
    )]
    n_gpu_layers: Option<GpuLayers>,

    /// MCP tool-server config JSON (stub: listed in `/v1/models` metadata).
    #[arg(long = "mcp-config", value_name = "PATH")]
    pub(crate) mcp_config: Option<PathBuf>,

    /// Exit when stdin reaches EOF (for a supervising parent process).
    ///
    /// Opt-in on purpose: a server started with stdin redirected from
    /// `/dev/null` -- systemd, cron, `nohup` -- sees EOF immediately,
    /// and making this the default would turn those into a server that
    /// exits the moment it starts. A parent that *wants* the guarantee
    /// (the desktop shell) passes the flag and keeps the pipe open.
    #[arg(long = "exit-on-stdin-close", default_value_t = false)]
    pub(crate) exit_on_stdin_close: bool,

    /// Share one batched decode worker across concurrent requests
    /// (llama.cpp `-cb`). Also sets `FRINK_CONTINUOUS_BATCHING=1`.
    #[arg(
        long = "cont-batching",
        visible_aliases = ["continuous-batching", "cb"],
        default_value_t = false
    )]
    cont_batching: bool,

    /// Disable auto continuous batching on Metal
    /// (`FRINK_CONTINUOUS_BATCHING=0`).
    #[arg(
        long = "no-cont-batching",
        default_value_t = false,
        conflicts_with = "cont_batching"
    )]
    no_cont_batching: bool,

    /// Max concurrent sequences under continuous batching (llama.cpp
    /// `-np`). Sets `FRINK_CB_MAX_SEQS`; implies `--cont-batching`
    /// unless `--no-cont-batching` is set.
    #[arg(long = "parallel", visible_alias = "np", value_name = "N")]
    parallel: Option<usize>,

    /// Logical maximum prompt tokens per forward pass, llama.cpp's
    /// `-b`. See [`crate::prefill_batch`] for how it and `-ub` resolve
    /// to the one number frink keeps.
    #[arg(long = "batch-size", visible_alias = "b", value_name = "N")]
    batch_size: Option<usize>,

    /// Physical maximum prompt tokens per forward pass, llama.cpp's
    /// `-ub`. Clamped to `--batch-size` when both are given.
    #[arg(long = "ubatch-size", visible_alias = "ub", value_name = "N")]
    ubatch_size: Option<usize>,

    /// Directory slot files are saved into and restored from,
    /// llama.cpp's `--slot-save-path`. Sets `FRINK_SLOT_SAVE_PATH`.
    ///
    /// `POST /slots/{id_slot}?action=save|restore` refuses with a 501
    /// naming this flag while it is unset, exactly as llama.cpp does
    /// (`tools/server/server-context.cpp:4538`): a server that would
    /// write KV state to disk should have been told where.
    #[arg(long = "slot-save-path", value_name = "DIR")]
    slot_save_path: Option<PathBuf>,

    /// LoRA adapter GGUF (llama.cpp's `--lora`), applied at scale 1.
    /// Repeatable; comma-separated values are accepted as upstream
    /// accepts them. Sets `FRINK_LORA` together with `--lora-scaled`,
    /// and every load -- the first and each `/admin/models/load` --
    /// attaches the same adapters or refuses the checkpoint by name.
    #[arg(long = "lora", value_name = "FILE", action = clap::ArgAction::Append)]
    lora: Vec<String>,

    /// LoRA adapter with a scale, `FILE:SCALE` (llama.cpp's
    /// `--lora-scaled`). Repeatable; adapters are numbered in the order
    /// given, every `--lora` before every `--lora-scaled`, and that
    /// number is the `id` `POST /lora-adapters` and a request's `lora`
    /// field address.
    #[arg(long = "lora-scaled", value_name = "FILE:SCALE", action = clap::ArgAction::Append)]
    lora_scaled: Vec<String>,

    /// Load the adapters but apply none of them until a
    /// `POST /lora-adapters` sets their scales (llama.cpp's
    /// `--lora-init-without-apply`). Sets `FRINK_LORA_INIT_WITHOUT_APPLY`.
    #[arg(long = "lora-init-without-apply", default_value_t = false)]
    lora_init_without_apply: bool,

    /// Start even though another frink process is already holding a
    /// model. Off by default: two models on one box do not share it,
    /// they thrash it, and both serve slower than either would alone.
    /// `FRINK_ALLOW_MULTIPLE_INSTANCES=1` does the same.
    #[arg(long = "allow-multiple-instances", default_value_t = false)]
    pub(crate) allow_multiple_instances: bool,

    /// Token budget for thinking, llama.cpp's `--reasoning-budget`: -1
    /// for unrestricted, 0 for immediate end, N>0 for a token budget
    /// (default: -1). The server default a request's
    /// `reasoning_budget_tokens` falls back to when it is absent or -1;
    /// sets `FRINK_REASONING_BUDGET`. Enforced in the sampler: once N
    /// tokens of thought have followed the opener, the closer is forced
    /// so the answer still arrives.
    #[arg(
        long = "reasoning-budget",
        value_name = "N",
        allow_hyphen_values = true
    )]
    reasoning_budget: Option<i64>,

    /// Continue a trailing assistant message instead of starting a new
    /// turn, llama.cpp's `--prefill-assistant` (the default). A request
    /// can still say `continue_final_message: false`.
    #[arg(long = "prefill-assistant", default_value_t = false)]
    prefill_assistant: bool,

    /// Treat a trailing assistant message as a complete turn,
    /// llama.cpp's `--no-prefill-assistant`. Sets
    /// `FRINK_PREFILL_ASSISTANT=0`; a request's own
    /// `continue_final_message` still wins.
    #[arg(
        long = "no-prefill-assistant",
        default_value_t = false,
        conflicts_with = "prefill_assistant"
    )]
    no_prefill_assistant: bool,
}

impl ServerArgs {
    /// Parses `frink-server`'s own argv, including the llama.cpp-style
    /// multi-character short options (`-ngl`, `-dev`) that clap cannot
    /// express and which are rewritten to their long forms first.
    ///
    /// Public because frink-cli's `serve` subcommand hands the same
    /// arguments to the same parser rather than reimplementing it.
    pub fn parse_llama_style<I>(argv: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        Self::parse_from(rewrite_llama_style_argv(argv.into_iter().collect()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OffloadDevice {
    Auto,
    None,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuLayers {
    Auto,
    All,
    Count(u32),
}

impl GpuLayers {
    fn offload_enabled(self) -> bool {
        !matches!(self, Self::Count(0))
    }
}

impl FromStr for GpuLayers {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "all" => Ok(Self::All),
            _ => value
                .parse::<u32>()
                .map(Self::Count)
                .map_err(|_| "expected 0, a positive integer, 'auto', or 'all'".into()),
        }
    }
}

impl fmt::Display for GpuLayers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::All => f.write_str("all"),
            Self::Count(value) => value.fmt(f),
        }
    }
}

/// Whether this build of the server has the Metal kernels compiled in.
///
/// Exists for the front ends that link this library: frink-cli's
/// `metal` feature has to forward into frink-server
/// (`frink-server?/metal`) or `frink serve --device metal` refuses on
/// a Metal host while `frink run` on the same binary uses it. That
/// mismatch is one Cargo manifest edit away and compiles cleanly, so
/// frink-cli asserts on this constant at compile time.
pub const BUILT_WITH_METAL: bool = cfg!(feature = "metal");

/// Whether this build of the server has the CUDA kernels compiled in.
/// See [`BUILT_WITH_METAL`].
pub const BUILT_WITH_CUDA: bool = cfg!(feature = "cuda");

fn rewrite_llama_style_argv(args: Vec<String>) -> Vec<String> {
    args.into_iter()
        .map(|arg| match arg.as_str() {
            "-ngl" => "--n-gpu-layers".into(),
            "-dev" => "--device".into(),
            "-cb" => "--cont-batching".into(),
            "-np" => "--parallel".into(),
            "-b" => "--batch-size".into(),
            "-ub" => "--ubatch-size".into(),
            // One token in llama.cpp's hand-written parser. clap sees
            // `-h` followed by `f` and prints help, which is what
            // `frink serve -hf repo:Q4_K_M` did: the flag looked
            // absent rather than mis-spelled.
            "-hf" => "--hf-repo".into(),
            "-hff" => "--hf-file".into(),
            _ => arg,
        })
        .collect()
}

fn cli_bind_addr(args: &ServerArgs, env_addr: Option<&str>) -> Option<String> {
    if args.host.is_none() && args.port.is_none() {
        return None;
    }

    let existing = env_addr.and_then(|value| value.parse::<SocketAddr>().ok());
    let host = args
        .host
        .or_else(|| existing.map(|addr| addr.ip()))
        .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
    let port = args
        .port
        .or_else(|| existing.map(|addr| addr.port()))
        .unwrap_or(8383);
    Some(SocketAddr::new(host, port).to_string())
}

/// Resolves a `-hf` reference to a local path, downloading it once.
///
/// Progress goes to STDERR, not stdout: stdout carries the
/// `frink.server.ready` line a supervising process parses, and a
/// progress bar in the middle of it would break that contract.
fn resolve_hf_repo(spec: &str, file: Option<&str>) -> anyhow::Result<String> {
    let mut hf = frink_models::hub::HfRef::parse(spec);
    if let Some(f) = file {
        hf.file = Some(f.to_string());
    }
    eprintln!(
        "frink: resolving {} on the Hub{}",
        hf.repo,
        hf.quant
            .as_deref()
            .map(|q| format!(" ({q})"))
            .unwrap_or_default()
    );

    let mut last = std::time::Instant::now();
    let mut draw = move |done: u64, total: Option<u64>| {
        if last.elapsed() < std::time::Duration::from_millis(200) {
            return;
        }
        last = std::time::Instant::now();
        let mib = done as f64 / 1024.0 / 1024.0;
        match total {
            Some(t) if t > 0 => {
                eprint!(
                    "\r  {mib:>9.1} MiB  {:5.1}%",
                    (done as f64 / t as f64) * 100.0
                )
            }
            _ => eprint!("\r  {mib:>9.1} MiB"),
        }
    };

    let (path, downloaded) = hf
        .ensure_local(&mut draw)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if downloaded {
        eprintln!();
        eprintln!("frink: downloaded {}", path.display());
    } else {
        eprintln!("frink: using cached {}", path.display());
    }
    Ok(path.to_string_lossy().into_owned())
}

pub(crate) fn apply_cli_overrides(args: &ServerArgs) -> anyhow::Result<()> {
    if let Some(model) = &args.model {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_MODEL_PATH", model) };
    }
    if let Some(spec) = &args.hf_repo {
        let path = resolve_hf_repo(spec, args.hf_file.as_deref())?;
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_MODEL_PATH", &path) };
    }
    if let Some(n) = args.ctx_size {
        if n == 0 {
            anyhow::bail!("--ctx-size must be greater than zero");
        }
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_CB_MAX_CONTEXT", n.to_string()) };
    }
    if let Some(key) = &args.api_key {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_API_KEY", key) };
    }
    if let Some(path) = &args.api_key_file {
        let key = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading --api-key-file {}: {e}", path.display()))?;
        let key = key.trim();
        if key.is_empty() {
            anyhow::bail!(
                "--api-key-file {} is empty: an empty key would leave every route open, \
                 which is the opposite of what passing the flag asked for",
                path.display()
            );
        }
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_API_KEY", key) };
    }
    if let Some(alias) = &args.alias {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_MODEL_NAME", alias) };
    }
    if let Some(ctk) = &args.ctk {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_CTK", ctk.trim()) };
    }
    // Refused by NAME rather than ignored. A prompt framed by a
    // hand-written guess instead of the checkpoint's own template is
    // the kind of wrong answer that reads as a model quality problem,
    // so "frink cannot do that" is the honest reply.
    if args.no_jinja {
        anyhow::bail!(
            "--no-jinja: frink has no template-free mode. It compiles and evaluates the GGUF's \
             own tokenizer.chat_template, which is what llama.cpp's --jinja turns on, and there \
             is no sniffing fallback to switch to. Use --no-cnv on `frink run` for a raw \
             completion"
        );
    }
    if let Some(mode) = &args.flash_attn {
        let mode = mode.trim().to_ascii_lowercase();
        if mode == "off" || mode == "disabled" || mode == "0" {
            anyhow::bail!(
                "--flash-attn off: fused attention is a backend property here, not a per-run \
                 switch. Set FRINK_METAL_ATTN=0 to take the unfused Metal path, or --device cpu"
            );
        }
    }

    if let Some(addr) = cli_bind_addr(args, std::env::var("FRINK_ADDR").ok().as_deref()) {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_ADDR", addr) };
    }

    if let Some(threads) = args.threads {
        if threads == 0 {
            anyhow::bail!("--threads must be greater than zero");
        }
        // SAFETY: called before the runtime starts worker threads.
        unsafe {
            std::env::set_var("FRINK_CPU_THREADS", threads.to_string());
            std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
        }
    }

    if args.device.is_none() && args.n_gpu_layers.is_none() {
        // device overrides skipped
    } else {
        let layers = args.n_gpu_layers.unwrap_or(GpuLayers::Auto);
        let device = if layers.offload_enabled() {
            args.device.unwrap_or(OffloadDevice::Auto)
        } else {
            OffloadDevice::None
        };

        match device {
            OffloadDevice::None | OffloadDevice::Cpu => unsafe {
                std::env::set_var("FRINK_METAL", "0");
                std::env::set_var("FRINK_METAL_ATTN", "0");
                std::env::set_var("FRINK_CUDA", "0");
            },
            OffloadDevice::Auto => unsafe {
                std::env::set_var("FRINK_METAL", "auto");
                std::env::set_var("FRINK_CUDA", "auto");
                if std::env::var_os("FRINK_METAL_ATTN").is_none() {
                    std::env::set_var("FRINK_METAL_ATTN", "1");
                }
            },
            OffloadDevice::Metal => {
                #[cfg(not(feature = "metal"))]
                {
                    anyhow::bail!(
                        "Metal requested but this binary was built without --features metal"
                    );
                }
                #[cfg(feature = "metal")]
                {
                    if !frink_metal::MetalProfile::detect().available {
                        anyhow::bail!("Metal requested but no Metal device is available");
                    }
                    unsafe {
                        std::env::set_var("FRINK_METAL", "1");
                        if std::env::var_os("FRINK_METAL_ATTN").is_none() {
                            std::env::set_var("FRINK_METAL_ATTN", "1");
                        }
                        std::env::set_var("FRINK_CUDA", "0");
                    }
                }
            }
            OffloadDevice::Cuda => {
                #[cfg(not(feature = "cuda"))]
                {
                    anyhow::bail!(
                        "CUDA requested but this binary was built without --features cuda"
                    );
                }
                #[cfg(feature = "cuda")]
                {
                    if !frink_cuda::HardwareProfile::detect().cuda_available {
                        anyhow::bail!("CUDA requested but no CUDA device is available");
                    }
                    unsafe {
                        std::env::set_var("FRINK_CUDA", "1");
                        std::env::set_var("FRINK_METAL", "0");
                        std::env::set_var("FRINK_METAL_ATTN", "0");
                    }
                }
            }
        }
    }

    if let Some(n) = args.parallel {
        if n == 0 {
            anyhow::bail!("--parallel must be greater than zero");
        }
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_CB_MAX_SEQS", n.to_string()) };
    }

    let lora_specs = frink_models::lora_attach::LoraSpec::from_flags(&args.lora, &args.lora_scaled)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if !lora_specs.is_empty() {
        for spec in &lora_specs {
            if !spec.path.is_file() {
                anyhow::bail!(
                    "--lora {}: no such file. An adapter that cannot be opened would be \
                     discovered at model load rather than at startup",
                    spec.path.display()
                );
            }
        }
        let value = lora_specs
            .iter()
            .map(|s| format!("{}:{}", s.path.display(), s.scale))
            .collect::<Vec<_>>()
            .join(",");
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var(crate::lora::ENV_SPECS, value) };
    }
    if args.lora_init_without_apply {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var(crate::lora::ENV_INIT_WITHOUT_APPLY, "1") };
    }

    if let Some(dir) = &args.slot_save_path {
        if !dir.is_dir() {
            anyhow::bail!(
                "--slot-save-path {} is not a directory. Slots are written into it by name, so \
                 a path that does not exist would be discovered on the first save rather than \
                 at startup",
                dir.display()
            );
        }
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_SLOT_SAVE_PATH", dir) };
    }

    for (flag, value) in [
        ("--batch-size", args.batch_size),
        ("--ubatch-size", args.ubatch_size),
    ] {
        if value == Some(0) {
            anyhow::bail!("{flag} must be greater than zero");
        }
    }
    if let Some(chunk) = crate::prefill_batch::effective_chunk(args.batch_size, args.ubatch_size) {
        // Both spellings, from one number and one array of names: the
        // private decode loop and the batch scheduler each read their
        // own variable, and an operator who names `-ub` must not have
        // to know which path this server happens to be serving on.
        for key in crate::prefill_batch::PREFILL_CHUNK_ENV_KEYS {
            // SAFETY: called before the runtime starts worker threads.
            unsafe { std::env::set_var(key, chunk.to_string()) };
        }
    }

    if let Some(budget) = args.reasoning_budget {
        // The same range check the request field applies, so the flag
        // and the field cannot admit different values.
        crate::reasoning_budget::BudgetTokens::parse(budget)
            .map_err(|why| anyhow::anyhow!("--reasoning-budget: {why}"))?;
        // SAFETY: called before the runtime starts worker threads.
        unsafe {
            std::env::set_var(
                crate::reasoning_budget::SERVER_DEFAULT_ENV,
                budget.to_string(),
            )
        };
    }
    if args.prefill_assistant {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var(crate::continuation::PREFILL_ASSISTANT_ENV, "1") };
    } else if args.no_prefill_assistant {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var(crate::continuation::PREFILL_ASSISTANT_ENV, "0") };
    }

    if args.cont_batching {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_CONTINUOUS_BATCHING", "1") };
    } else if args.no_cont_batching {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_CONTINUOUS_BATCHING", "0") };
    } else if args.parallel.is_some() {
        // llama.cpp `-np` is only meaningful with continuous batching.
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_CONTINUOUS_BATCHING", "1") };
    }

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The server and `frink run` set the same `FRINK_CTK` variable, so
    /// they have to agree about which values exist. The server used to
    /// accept anything and pass it through, which meant
    /// `frink-server --ctk nonsense` served f16 without a word while
    /// `frink run --ctk nonsense` refused.
    #[test]
    fn the_server_accepts_exactly_the_cache_types_the_cli_does() {
        for good in ["f16", "q8_0", "q4_0", "fp8", "q5_1"] {
            let a = ServerArgs::try_parse_from(["frink-server", "--ctk", good]);
            assert!(a.is_ok(), "{good} was refused");
        }
        for bad in ["nonsense", "turbo4", "q3_k"] {
            let e = ServerArgs::try_parse_from(["frink-server", "--ctk", bad])
                .expect_err(&format!("{bad} was accepted"));
            let msg = e.to_string();
            assert!(msg.contains("unsupported cache type"), "{msg}");
        }
    }

    #[test]
    fn parses_llama_server_style_options() {
        let argv = [
            "frink-server",
            "-m",
            "model.gguf",
            "--host",
            "::1",
            "--port",
            "9000",
            "-t",
            "4",
            "-dev",
            "Metal",
            "-ngl",
            "all",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let args = ServerArgs::try_parse_from(rewrite_llama_style_argv(argv)).unwrap();

        assert_eq!(args.model.as_deref(), Some("model.gguf"));
        assert_eq!(args.host, Some(IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)));
        assert_eq!(args.port, Some(9000));
        assert_eq!(args.threads, Some(4));
        assert_eq!(args.device, Some(OffloadDevice::Metal));
        assert_eq!(args.n_gpu_layers, Some(GpuLayers::All));
        assert_eq!(
            cli_bind_addr(&args, Some("127.0.0.1:8383")).as_deref(),
            Some("[::1]:9000")
        );
    }

    #[test]
    fn port_zero_survives_argument_parsing_as_a_real_request() {
        // `--port 0` must reach the bind call intact: it is a request
        // for a kernel-assigned port, not a missing value to default to
        // 8383. The address it produces is deliberately provisional --
        // the ready line reports what was actually bound.
        let argv = ["frink-server", "--port", "0"]
            .into_iter()
            .map(String::from)
            .collect();
        let args = ServerArgs::try_parse_from(rewrite_llama_style_argv(argv)).unwrap();
        assert_eq!(args.port, Some(0));
        assert_eq!(
            cli_bind_addr(&args, Some("127.0.0.1:8383")).as_deref(),
            Some("127.0.0.1:0")
        );
    }

    #[test]
    fn parallel_flag_parses_and_rewrites_np() {
        let argv = ["frink-server", "-np", "4"]
            .into_iter()
            .map(String::from)
            .collect();
        let args = ServerArgs::try_parse_from(rewrite_llama_style_argv(argv)).unwrap();
        assert_eq!(args.parallel, Some(4));
    }

    /// `-b` and `-ub` are two tokens in llama.cpp's hand-written
    /// parser and one token to clap, which sees `-b` as a short option
    /// it has never heard of. The rewrite is what makes a copied
    /// `llama-server ... -b 2048 -ub 512` command run here at all.
    #[test]
    fn batch_flags_parse_and_rewrite_their_llama_cpp_short_forms() {
        let argv = ["frink-server", "-b", "2048", "-ub", "512"]
            .into_iter()
            .map(String::from)
            .collect();
        let args = ServerArgs::try_parse_from(rewrite_llama_style_argv(argv)).unwrap();
        assert_eq!(args.batch_size, Some(2048));
        assert_eq!(args.ubatch_size, Some(512));
    }

    /// Zero is not a batch size, and accepting it would make
    /// `env_positive` panic the server later with a message naming an
    /// environment variable the operator never set.
    #[test]
    fn a_zero_batch_size_is_refused_by_name_rather_than_lowered_to_the_environment() {
        for flag in ["--batch-size", "--ubatch-size"] {
            let args = ServerArgs::try_parse_from(
                ["frink-server", flag, "0"].into_iter().map(String::from),
            )
            .unwrap();
            let err = apply_cli_overrides(&args).unwrap_err().to_string();
            assert!(err.contains(flag), "{flag}: {err}");
        }
    }

    /// A path that is not a directory is refused at startup rather
    /// than on the first save. The repo's rule about gates applies to
    /// flags too: a `--slot-save-path` pointing at nothing looks
    /// configured until somebody tries to use it, hours later.
    #[test]
    fn a_slot_save_path_that_is_not_a_directory_is_refused_at_startup() {
        let args = ServerArgs::try_parse_from(
            ["frink-server", "--slot-save-path", "/definitely/not/here"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        let err = apply_cli_overrides(&args).unwrap_err().to_string();
        assert!(err.contains("--slot-save-path"), "{err}");
        assert!(
            std::env::var("FRINK_SLOT_SAVE_PATH").is_err(),
            "a refused path must not have been lowered to the environment first"
        );
    }

    /// llama.cpp's spelling and range (`common/arg.cpp:3608-3614`):
    /// `-1`, `0` and `N` parse -- `-1` needs `allow_hyphen_values`, or
    /// clap reads it as a flag -- and anything below `-1` is refused by
    /// name before it is lowered to the environment.
    #[test]
    fn reasoning_budget_parses_llama_cpps_range_and_refuses_the_rest() {
        for (value, expect) in [("-1", -1), ("0", 0), ("2000", 2000)] {
            let args = ServerArgs::try_parse_from(
                ["frink-server", "--reasoning-budget", value]
                    .into_iter()
                    .map(String::from),
            )
            .unwrap();
            assert_eq!(args.reasoning_budget, Some(expect), "{value}");
        }
        let args = ServerArgs::try_parse_from(
            ["frink-server", "--reasoning-budget", "-2"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        let err = apply_cli_overrides(&args).unwrap_err().to_string();
        assert!(err.contains("--reasoning-budget"), "{err}");
    }

    /// Both spellings of llama.cpp's prefill switch parse, and they
    /// conflict rather than letting the last one win silently.
    #[test]
    fn prefill_assistant_has_both_of_llama_cpps_spellings() {
        let on = ServerArgs::try_parse_from(
            ["frink-server", "--prefill-assistant"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(on.prefill_assistant && !on.no_prefill_assistant);
        let off = ServerArgs::try_parse_from(
            ["frink-server", "--no-prefill-assistant"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(off.no_prefill_assistant && !off.prefill_assistant);
        assert!(ServerArgs::try_parse_from(
            [
                "frink-server",
                "--prefill-assistant",
                "--no-prefill-assistant"
            ]
            .into_iter()
            .map(String::from),
        )
        .is_err());
    }

    #[test]
    fn stdin_close_exit_is_opt_in() {
        // Default off: a server whose stdin is /dev/null (systemd, cron,
        // nohup) would otherwise exit the instant it started.
        let args =
            ServerArgs::try_parse_from(["frink-server"].into_iter().map(String::from)).unwrap();
        assert!(!args.exit_on_stdin_close);
        let args = ServerArgs::try_parse_from(
            ["frink-server", "--exit-on-stdin-close"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(args.exit_on_stdin_close);
    }
}
