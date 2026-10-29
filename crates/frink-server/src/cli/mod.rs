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

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use clap::{CommandFactory, FromArgMatches, Parser, ValueEnum};
use frink_models::gpu_layers::GpuLayers;

// `PartialEq` so frink-cli's serve tests can assert that both front
// ends parse a command line into the SAME arguments, rather than
// asserting field by field and missing whichever one is added next.
#[derive(Parser, Debug, PartialEq)]
// `version` with an explicit `display_name`. This struct is both
// `frink-server`'s own argv and the body of frink-cli's `serve`
// subcommand, and clap names an embedded subcommand after its variant:
// `frink serve --version` once printed `frink-serve 0.10.0`, naming a
// binary nobody ships. Both spellings print `frink-server` now, the
// binary this is, at the one workspace version. llama-server takes
// `--version`, so going without it was a gap (ADR 0005).
#[command(
    name = "frink-server",
    display_name = "frink-server",
    version,
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
        visible_aliases = ["hf", "hfr"],
        value_name = "REPO[:QUANT]",
        conflicts_with = "model"
    )]
    hf_repo: Option<String>,

    /// Exact filename inside `--hf-repo`, llama.cpp's `-hff`.
    ///
    /// For a repo whose quant labels do not disambiguate, or a file
    /// whose name carries no quant at all.
    #[arg(
        long = "hf-file",
        visible_alias = "hff",
        value_name = "FILE",
        requires = "hf_repo"
    )]
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
    /// llama.cpp's `-a` / `--alias`. Sets `FRINK_MODEL_NAME`.
    ///
    /// ONE name: llama.cpp takes a comma-separated list, and a list is
    /// refused here rather than served as one name with commas in it.
    #[arg(
        short = 'a',
        long = "alias",
        visible_alias = "model-alias",
        value_name = "NAME"
    )]
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

    /// Hugging Face access token for `-hf`, llama.cpp's `-hft` /
    /// `--hf-token`. Sets `HF_TOKEN`, the variable the downloader
    /// already reads and the one llama.cpp declares on this flag.
    #[arg(long = "hf-token", visible_alias = "hft", value_name = "TOKEN")]
    hf_token: Option<String>,

    /// PEM private key to serve HTTPS with, llama.cpp's
    /// `--ssl-key-file`. Sets `FRINK_TLS_KEY`; needs `--ssl-cert-file`.
    #[arg(long = "ssl-key-file", value_name = "FNAME")]
    ssl_key_file: Option<PathBuf>,

    /// PEM certificate to serve HTTPS with, llama.cpp's
    /// `--ssl-cert-file`. Sets `FRINK_TLS_CERT`; needs `--ssl-key-file`.
    #[arg(long = "ssl-cert-file", value_name = "FNAME")]
    ssl_cert_file: Option<PathBuf>,

    /// Comma-separated origins allowed by CORS, llama.cpp's
    /// `--cors-origins`. Sets `FRINK_CORS_ORIGINS`.
    ///
    /// Exact origins only. llama.cpp's default `*` and its special value
    /// `localhost` (reflect the Origin header when it is local) are
    /// refused by name: frink has no wildcard by design, and `localhost`
    /// would otherwise be taken as the literal origin `localhost`, which
    /// no browser sends.
    #[arg(long = "cors-origins", value_name = "ORIGINS")]
    cors_origins: Option<String>,

    /// Threads that process HTTP requests, llama.cpp's
    /// `--threads-http`. Sets `FRINK_TOKIO_WORKERS`. A count below 1
    /// (llama.cpp's default is `-1`) keeps frink's own default, as it
    /// keeps llama.cpp's; generation does not run on these threads in
    /// either engine.
    #[arg(long = "threads-http", value_name = "N", allow_hyphen_values = true)]
    threads_http: Option<i64>,

    /// Accepted: `/metrics` (Prometheus text format) is always served,
    /// so llama.cpp's switch to turn it on has nothing left to turn on.
    #[arg(long = "metrics", default_value_t = false)]
    metrics: bool,

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

    /// GPU layers: `0`, `auto`, `all`, or a count at or above the
    /// model's layer count.
    ///
    /// Partial placement is not implemented, so a PARTIAL count is
    /// refused at startup against the model's own layer count, as
    /// `frink run` refuses it. This used to say "any value above zero
    /// enables all supported operations", which is `-ngl 10` silently
    /// offloading every layer of a model the user was trying to fit.
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
        visible_alias = "nocb",
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
        let mut cmd = Self::built_command();
        let argv = llama_style_argv(&cmd, argv.into_iter().collect(), |k| std::env::var(k).ok())
            .unwrap_or_else(|e| cmd.error(clap::error::ErrorKind::UnknownArgument, e).exit());
        let matches = cmd
            .try_get_matches_from_mut(argv)
            .unwrap_or_else(|e| e.exit());
        Self::from_arg_matches(&matches).unwrap_or_else(|e| e.exit())
    }

    /// The command [`Self::parse_llama_style`] parses with, built, with
    /// llama.cpp's `--usage` spelling of help added before clap indexes
    /// the spellings ([`frink_core::argv::with_usage_alias`]).
    pub fn built_command() -> clap::Command {
        let mut cmd = frink_core::argv::with_usage_alias(Self::command());
        cmd.build();
        cmd
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

/// `argv` as clap should see it: llama.cpp's single-dash names
/// rewritten from the command's own spellings ([`frink_core::argv`]),
/// then every set `LLAMA_ARG_*` variable appended as its flag
/// ([`crate::llama_env`]). `cmd` must be built.
fn llama_style_argv(
    cmd: &clap::Command,
    argv: Vec<String>,
    env: impl Fn(&str) -> Option<String>,
) -> Result<Vec<String>, String> {
    let mut argv = argv.into_iter();
    let program = argv.next().unwrap_or_else(|| "frink-server".to_string());
    let tail = frink_core::argv::expand_single_dash_names(cmd, argv.collect())?;
    let tail = crate::llama_env::append_llama_arg_env(cmd, tail, env)?;
    Ok(std::iter::once(program).chain(tail).collect())
}

/// The layer count `-ngl` is checked against, read from the GGUF header
/// alone. `None` for a checkpoint directory or an unreadable file.
fn model_layer_count(path: &str) -> Option<usize> {
    let file = frink_gguf::ShardedGguf::open(path).ok()?;
    let arch = file.metadata_str("general.architecture")?;
    file.metadata_u64(&format!("{arch}.block_count"))
        .map(|n| n as usize)
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

/// `--cors-origins` values llama.cpp gives a meaning frink does not
/// have, refused by name before `FRINK_CORS_ORIGINS` would read them as
/// something else (`security::parse_cors_origins` refuses `*` too, but
/// only once the router is built, after the model has loaded).
fn check_cors_origins(spec: &str) -> anyhow::Result<()> {
    for origin in spec.split(',').map(str::trim) {
        match origin {
            "*" => anyhow::bail!(
                "--cors-origins *: frink allows exact origins only, by design. List the \
                 origins to allow"
            ),
            "localhost" => anyhow::bail!(
                "--cors-origins localhost: llama.cpp's special value (reflect the Origin header \
                 when it is local) is not implemented. List the exact origins, e.g. \
                 http://localhost:5173"
            ),
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn apply_cli_overrides(args: &ServerArgs) -> anyhow::Result<()> {
    if let Some(token) = &args.hf_token {
        // Before `-hf` is resolved below, which is what reads it.
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("HF_TOKEN", token) };
    }
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
        if alias.contains(',') {
            anyhow::bail!(
                "--alias {alias}: frink serves one model name, and llama.cpp's comma-separated \
                 list of aliases is not implemented. Pass one name"
            );
        }
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
        frink_core::argv::check_flash_attn(mode).map_err(|e| anyhow::anyhow!(e))?;
    }
    match (&args.ssl_key_file, &args.ssl_cert_file) {
        (Some(key), Some(cert)) => {
            // SAFETY: called before the runtime starts worker threads.
            unsafe {
                std::env::set_var("FRINK_TLS_KEY", key);
                std::env::set_var("FRINK_TLS_CERT", cert);
            }
        }
        (None, None) => {}
        _ => anyhow::bail!(
            "--ssl-key-file and --ssl-cert-file must be given together: HTTPS needs both the \
             key and the certificate, and serving plain HTTP instead would ignore the one given"
        ),
    }
    if let Some(origins) = &args.cors_origins {
        check_cors_origins(origins)?;
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_CORS_ORIGINS", origins) };
    }
    // llama.cpp reads any count below 1 as "pick the default"
    // (`server-http.cpp`: `if (n_threads_http < 1)`), so `0` and `-1`
    // both keep frink's own default here too.
    if let Some(n) = args.threads_http.filter(|n| *n > 0) {
        // SAFETY: called before the runtime starts worker threads.
        unsafe { std::env::set_var("FRINK_TOKIO_WORKERS", n.to_string()) };
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

    if let Some(layers @ GpuLayers::Count(n)) = args.n_gpu_layers {
        if n > 0 {
            let path = std::env::var("FRINK_MODEL_PATH").ok();
            let n_layers = path.as_deref().and_then(model_layer_count).ok_or_else(|| {
                anyhow::anyhow!(
                    "--n-gpu-layers {n}: the model's layer count could not be read, so whether \
                     {n} is all of them cannot be checked, and a partial offload is not \
                     implemented. Use `-ngl all` or `-ngl 0`"
                )
            })?;
            layers.check_supported(n_layers)?;
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
mod tests;
