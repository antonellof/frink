//! `frink bench -m model.gguf` — a `llama-bench` work-alike.
//!
//! Same workload definition as `llama-bench` (`pp<N>` = batched prefill
//! of N synthetic tokens, `tg<N>` = N single-token decode steps after a
//! one-token prime), same reporting (mean ± sample stddev over `-r`
//! repetitions, after one discarded warmup, as llama-bench's markdown
//! cell prints them), same flag names.
//! Output is directly comparable to
//! `llama-bench -m <same gguf> -p <N> -n <N> -t <T> -ngl <L>`.
//!
//! Deliberately *not* here: HTTP, chat template, real tokenizer,
//! sampling. Synthetic token ids exercise the same weights and the same
//! KV growth without making the number depend on a tokenizer's behavior.
//!
//! Engines: generic [`Decoder`] (batched `forward_batch_last` for `pp*`)
//! and dedicated [`Gemma4Engine`] (sequential `forward_token` for both
//! `pp*` and `tg*` until a batched Gemma-4 prefill lands).

use crate::bench_contract::{self, HostState};
use crate::bench_guard::{self, WorkloadDigest};
use crate::bench_llama;
use anyhow::Context;
use frink_models::{
    load_gemma4_engine_from_path, select_engine_kind, Decoder, ModelConfig, SelectedEngineKind,
    ServedEngine,
};
use std::path::Path;
use std::time::Instant;

mod measure;
mod output;
mod report;

pub use output::OutputFormat;

pub(crate) use measure::{check_result, decode_tokens, fresh_caches, probe, synthetic_tokens};
use measure::{measure_decoder, measure_gemma4};
use report::{
    estimate_gemma4_params, estimate_params, human_bytes, human_params, quant_label, truncate,
    write_receipt,
};

/// One `llama-bench` row: a named workload and its per-rep tok/s.
struct Row {
    test: String,
    samples: Vec<f64>,
    /// Digest of the token stream every timed repetition fed. Written
    /// into the receipt so two rows for the same workload can be shown
    /// to have measured the same work, across sessions.
    digest: WorkloadDigest,
    /// llama-bench's `n_prompt` / `n_gen` for this test: one of them is
    /// the row's size and the other is 0.
    n_prompt: usize,
    n_gen: usize,
    /// llama-bench's `n_depth`: context in the KV before the timed part.
    n_depth: usize,
    /// When the test started (`%FT%TZ`), llama-bench's `test_time`.
    test_time: String,
}

impl Row {
    /// The mean of the per-repetition rates: llama-bench's `avg_ts`, the
    /// number before `±` in its markdown cell (`llama-bench.cpp:102-108`).
    /// The table, the `--compare` gap, the receipt and `-o json` all
    /// report it, so the gap divides llama.cpp's `avg_ts` by its like.
    fn mean(&self) -> f64 {
        output::mean(&self.samples)
    }

    /// The sample stddev (`n - 1`): llama-bench's `stdev_ts`, the number
    /// after `±` (`llama-bench.cpp:110-118`).
    fn stddev(&self) -> f64 {
        output::sample_stddev(&self.samples)
    }
}

/// Applies the same backend/threading env `frink run` does, before any
/// worker thread exists. Kept separate from [`run`] so the caller can
/// order it ahead of everything else in `main`.
///
/// # Safety
/// Must be called while the process is still single-threaded.
/// Applies the backend/threading env, then CHECKS that it took.
///
/// The check is the point. Setting `FRINK_METAL=0` is not the same as
/// running on the CPU: the backend decision is cached in a `OnceLock`
/// the first time anything reads it, so a caller that ran too late set
/// a variable nobody would ever consult again. That is exactly what
/// happened to every `cpu` row in `benchmarks/RESULTS.md` (#126) --
/// each receipt recorded `backend: "cpu"` beside `backend_active:
/// "Metal"`, and nothing compared them.
///
/// So this asks `active_backend()` immediately, which both forces the
/// cache while the env is still fresh and reveals a disagreement, and
/// it fails loudly rather than measuring the wrong device.
///
/// `model` is the GGUF the run will load, when there is one: a PARTIAL
/// `-ngl` (above zero, below its layer count) is refused against it.
/// llama-bench's `-ngl 10` offloads ten layers; frink has no partial
/// placement, and offloading all of them while the receipt says 10 is
/// a row measuring something other than what it is labelled.
pub fn apply_env(threads: usize, n_gpu_layers: usize, model: Option<&str>) -> anyhow::Result<()> {
    if let (Some(path), true) = (model, n_gpu_layers > 0) {
        if let Ok(file) = frink_gguf::ShardedGguf::open(path) {
            let arch = file
                .metadata_str("general.architecture")
                .unwrap_or("unknown");
            if let Some(n_layers) = file.metadata_u64(&format!("{arch}.block_count")) {
                let layers = u32::try_from(n_gpu_layers).unwrap_or(u32::MAX);
                frink_models::gpu_layers::GpuLayers::Count(layers)
                    .check_supported(n_layers as usize)?;
            }
        }
    }
    // SAFETY: called from `main` before rayon/Metal workers spawn.
    unsafe {
        if threads > 0 {
            std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
            std::env::set_var("FRINK_CPU_THREADS", threads.to_string());
        }
        frink_models::offload::select(frink_models::offload::OffloadDevice::with_offload(
            None,
            n_gpu_layers != 0,
        ))?;
        frink_core::weight_matrix::default_cpu_int_dot_on();
    }
    frink_core::threads::init_cpu_pool();

    // Reads (and therefore fixes) the cached backend while the env
    // above is still the most recent word on it.
    let active = active_backend();
    let wanted_cpu = n_gpu_layers == 0;
    if wanted_cpu && active != "CPU" {
        anyhow::bail!(
            "--n-gpu-layers 0 asks for CPU but this process resolved to {active}. \
             The backend is decided once per process and something fixed it before \
             this call, so the run would measure {active} and label the receipt `cpu`. \
             Refusing rather than publishing a mislabelled row (#126)."
        );
    }
    Ok(())
}

/// How each test runs: llama-bench's `--no-warmup`, `--delay` and
/// `--progress`.
#[derive(Clone, Copy, Debug)]
pub struct RunOpts {
    /// Untimed repetitions before the timed ones:
    /// [`bench_guard::WARMUP_REPS`], or 0 under `--no-warmup`.
    pub warmup: usize,
    /// Seconds slept before each test.
    pub delay_s: u64,
    /// Print llama-bench's progress lines to stderr.
    pub progress: bool,
}

/// llama.cpp's `set_process_priority` (`common/common.cpp`): `--prio`
/// `-1..=3` as `nice` 5, 0, -5, -10, -20 for this process. `0` changes
/// nothing. Called before any worker thread is spawned, so the threads
/// inherit it (on Linux `nice` is per thread). A priority the OS refuses
/// (a negative `nice` without the privilege) is an error, as
/// llama-bench exits on it.
pub fn set_process_priority(prio: i32) -> anyhow::Result<()> {
    let nice = match prio {
        0 => return Ok(()),
        -1 => 5,
        1 => -5,
        2 => -10,
        3 => -20,
        other => anyhow::bail!("--prio {other}: llama-bench takes -1, 0, 1, 2 or 3"),
    };
    #[cfg(unix)]
    {
        // SAFETY: `setpriority` reads its three integer arguments and
        // touches no memory of ours.
        let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, nice) };
        if rc != 0 {
            anyhow::bail!(
                "--prio {prio}: failed to set process priority (nice {nice}): {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = nice;
        anyhow::bail!("--prio {prio}: process priority is only implemented on Unix hosts")
    }
}

pub struct BenchArgs {
    pub model: String,
    pub n_prompt: usize,
    pub n_gen: usize,
    /// llama-bench's `-d`: context tokens put in the KV, untimed, before
    /// every repetition.
    pub depth: usize,
    /// llama-bench's `-pg pp,tg`: one combined test, if asked for.
    pub pg: Option<(usize, usize)>,
    /// llama-bench's `-b` / `-ub`, resolved by
    /// `frink_core::argv::effective_ubatch`: prompt tokens per forward
    /// pass, `None` for the whole prompt at once.
    pub chunk: Option<usize>,
    pub reps: usize,
    pub ctx_size: usize,
    /// Also run `llama-bench` on the same GGUF with matching flags and
    /// report the gap, instead of leaving the comparison to the reader.
    pub compare: bool,
    /// Backend label recorded in the receipt (`cpu` / `metal` / `cuda`).
    pub backend: String,
    /// Where to write the JSON receipt, if anywhere.
    pub receipt: Option<std::path::PathBuf>,
    /// Suite id this run belongs to, for the receipt.
    pub id: Option<String>,
    /// 1-minute load average above which the run refuses to start.
    /// `0.0` disables the check. See `host_state`.
    pub max_load: f64,
    /// Warmup, delay and progress.
    pub run: RunOpts,
    /// llama-bench's `-ngl` as given, for the structured output.
    pub n_gpu_layers: usize,
    /// llama-bench's `-o`: what stdout carries.
    pub output: OutputFormat,
    /// llama-bench's `-oe`: a second copy on stderr, if any.
    pub output_err: Option<OutputFormat>,
    /// llama-bench's `-ctk` as given (already in `FRINK_CTK`), checked
    /// against what the active backend's store holds before anything
    /// is timed.
    pub cache_type_k: Option<String>,
}

pub fn run(args: BenchArgs) -> anyhow::Result<()> {
    // Before anything is loaded: a timed run on a busy or hot host
    // produces a number that looks like a measurement and is not one.
    bench_guard::check_repetitions(args.reps)?;
    if args.run.warmup < bench_guard::WARMUP_REPS && args.receipt.is_some() {
        anyhow::bail!(
            "--no-warmup with --receipt: a receipt's measurement contract times only \
             after {} discarded warmup repetition(s), and a row taken without it would \
             carry first-touch page faults and pipeline compiles into its rate",
            bench_guard::WARMUP_REPS
        );
    }
    let (load_start, thermal_start) = bench_contract::preflight_host(args.max_load)?;
    if let Some(want) = &args.cache_type_k {
        let held = frink_models::kv_budget::host_kv_dtype(active_budget_backend(), want).name();
        if held != want {
            anyhow::bail!(
                "--cache-type-k {want}: the {} KV store would hold {held}, so the row would \
                 measure {held} and be labelled {want}. frink bench takes the types this \
                 backend stores",
                active_backend()
            );
        }
    }
    let model = crate::pull::resolve_model_path(&args.model)?;
    let path = Path::new(&model);
    if !path.exists() {
        anyhow::bail!("model not found: {model}");
    }
    bench_contract::ensure_weights_fit(args.max_load, path, 0.0)?;

    let file = frink_gguf::ShardedGguf::open(path)?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    let kind = select_engine_kind(&arch).map_err(|e| anyhow::anyhow!("{e}"))?;

    let ctx_needed = args.depth
        + (args.n_prompt.max(1) + args.n_gen).max(args.pg.map_or(0, |(pp, tg)| pp + tg))
        + 2;
    if args.ctx_size > 0 && ctx_needed > args.ctx_size {
        anyhow::bail!(
            "workload needs {ctx_needed} positions but -c is {}",
            args.ctx_size
        );
    }

    let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let (params_b, load_s, rows) = match kind {
        SelectedEngineKind::GenericDecoder => {
            let mut config = ModelConfig::from_gguf(&file)
                .with_context(|| format!("reading model config for arch {arch}"))?;
            // `llama-bench` measures at f16 K/V unless `-ctk` says
            // otherwise, so the row it is compared against keeps the
            // store `frink run` would.
            config.kv_dtype = frink_models::host_kv_dtype_from_env(active_budget_backend());
            let params_b = estimate_params(&config);
            let load_t = Instant::now();
            let decoder = Decoder::from_gguf(path, config)?;
            let load_s = load_t.elapsed().as_secs_f64();
            (params_b, load_s, measure_decoder(&decoder, &args)?)
        }
        SelectedEngineKind::Gemma4 => {
            let load_t = Instant::now();
            let served = load_gemma4_engine_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
            let ServedEngine::Gemma4(engine) = served else {
                anyhow::bail!("expected ServedEngine::Gemma4 for arch {arch}");
            };
            let engine = *engine;
            let load_s = load_t.elapsed().as_secs_f64();
            let params_b = estimate_gemma4_params(&engine);
            // Gemma-4 has no batched prefill yet: pp* is sequential
            // forward_token (same as `frink run`). llama-bench still
            // batches, so the pp gap is partly that asymmetry.
            eprintln!("frink bench: gemma4 uses sequential prefill (no forward_batch_last yet)");
            (params_b, load_s, measure_gemma4(&engine, &args)?)
        }
        other => anyhow::bail!(
            "`frink bench` does not cover {other:?} yet (arch {arch}); \
             use the dedicated engine path via `frink run` for inference"
        ),
    };

    let backend = active_backend();
    let threads = frink_core::threads::resolve_cpu_threads();
    let mut table = String::new();
    {
        use std::fmt::Write as _;
        let _ = writeln!(
            table,
            "| {:<30} | {:>10} | {:>10} | {:<10} | {:>7} | {:>15} | {:>20} |",
            "model", "size", "params", "backend", "threads", "test", "t/s"
        );
        let _ = writeln!(
            table,
            "| {:-<30} | {:->10} | {:->10} | {:-<10} | {:->7} | {:->15} | {:->20} |",
            "", "", "", "", "", "", ""
        );
        for row in &rows {
            let _ = writeln!(
                table,
                "| {:<30} | {:>10} | {:>10} | {:<10} | {:>7} | {:>15} | {:>13.2} ± {:>4.2} |",
                truncate(&format!("{arch} {}", quant_label(&file)), 30),
                human_bytes(size_bytes),
                human_params(params_b),
                backend,
                threads,
                row.test,
                row.mean(),
                row.stddev(),
            );
        }
    }
    let structured = |format: OutputFormat| -> String {
        if format == OutputFormat::Md {
            return table.clone();
        }
        let info = output::RunInfo {
            build_commit: option_env!("FRINK_GIT_SHA")
                .unwrap_or("unknown")
                .to_string(),
            cpu_info: crate::host_state::host_spec().cpu.unwrap_or_default(),
            gpu_info: bench_contract::accelerator_name(backend).unwrap_or_default(),
            backends: backend.to_string(),
            model_filename: model.clone(),
            model_type: format!("{arch} {}", quant_label(&file)),
            model_size: size_bytes,
            // llama.cpp's `llama_model_n_params` counts every tensor's
            // elements; the table's `params` is an estimate.
            model_n_params: file
                .tensors()
                .map(|(_, t)| t.shape.iter().product::<u64>())
                .sum(),
            n_threads: threads,
            type_k: frink_models::host_kv_dtype_from_env(active_budget_backend())
                .name()
                .to_string(),
            type_v: frink_models::host_kv_dtype_from_env(active_budget_backend())
                .name()
                .to_string(),
            n_gpu_layers: args.n_gpu_layers,
        };
        let tests: Vec<output::TestRecord> = rows
            .iter()
            .map(|r| {
                output::TestRecord::new(
                    &info,
                    r.n_prompt,
                    r.n_gen,
                    r.n_depth,
                    r.test_time.clone(),
                    &r.samples,
                )
            })
            .collect();
        output::render(format, &tests)
    };
    print!("{}", structured(args.output));
    if let Some(err_format) = args.output_err {
        eprint!("{}", structured(err_format));
    }
    // Anything else frink prints must not land in a stdout a script is
    // parsing as csv, json, jsonl or sql.
    let side_table = args.output != OutputFormat::Md;
    eprintln!(
        "\nfrink bench: load {load_s:.2}s, {} reps + {} discarded warmup each",
        args.reps, args.run.warmup
    );
    let bench_contract::HostAfter {
        load_end,
        thermal_end,
        engine_env,
    } = bench_contract::report_host_after("frink bench", load_start, &thermal_start);

    let ngl = if backend == "CPU" { 0 } else { 99 };
    let llama = if args.compare {
        match bench_llama::run_llama_bench(
            &model,
            args.n_prompt,
            args.n_gen,
            args.depth,
            args.pg,
            args.reps,
            ngl,
        ) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("frink bench: llama-bench comparison unavailable: {e}");
                None
            }
        }
    } else {
        eprintln!(
            "compare: llama-bench -m {model} -p {} -n {}{} -ngl {ngl}   (or pass --compare)",
            args.n_prompt,
            args.n_gen,
            if args.depth > 0 {
                format!(" -d {}", args.depth)
            } else {
                String::new()
            },
        );
        None
    };

    macro_rules! side_println {
        ($($t:tt)*) => {
            if side_table {
                eprintln!($($t)*)
            } else {
                println!($($t)*)
            }
        };
    }
    if let Some(llama) = &llama {
        side_println!();
        side_println!(
            "| {:<15} | {:>12} | {:>12} | {:>8} |",
            "test",
            "frink",
            "llama.cpp",
            "gap"
        );
        side_println!("| {:-<15} | {:->12} | {:->12} | {:->8} |", "", "", "", "");
        for row in &rows {
            let l = llama.tps.get(&row.test).copied();
            let gap = l.map(|l| l / row.mean());
            side_println!(
                "| {:<15} | {:>12.2} | {:>12} | {:>8} |",
                row.test,
                row.mean(),
                l.map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into()),
                gap.map(|g| format!("{g:.2}×"))
                    .unwrap_or_else(|| "—".into()),
            );
        }
        eprintln!("gap = llama / frink; <1 means frink is faster");
        match llama.identity() {
            Ok(id) => eprintln!(
                "llama.cpp: build {} ({}), backends {}",
                id.build_commit,
                id.build_number
                    .map_or_else(|| "?".into(), |n| n.to_string()),
                id.backends
            ),
            Err(missing) => eprintln!(
                "llama.cpp: llama-bench did not report {}; a receipt of this run is refused",
                missing.join(" or ")
            ),
        }
    }

    if let Some(dest) = &args.receipt {
        write_receipt(
            dest,
            &args,
            &model,
            &arch,
            &file,
            size_bytes,
            params_b,
            threads,
            backend,
            &rows,
            llama.as_ref(),
            load_s,
            HostState {
                load_start,
                load_end,
                thermal_start,
                thermal_end,
            },
            &engine_env,
        )?;
        eprintln!("frink bench: receipt written to {}", dest.display());
    }
    Ok(())
}

pub fn active_backend() -> &'static str {
    #[cfg(feature = "metal")]
    {
        if frink_core::weight_matrix::metal_dense_enabled() {
            return "Metal";
        }
    }
    #[cfg(feature = "cuda")]
    {
        if frink_core::weight_matrix::cuda_dense_enabled() {
            return "CUDA";
        }
    }
    "CPU"
}

/// [`active_backend`] as the budget names it.
pub fn active_budget_backend() -> frink_models::BudgetBackend {
    match active_backend() {
        "Metal" => frink_models::BudgetBackend::Metal,
        "CUDA" => frink_models::BudgetBackend::Cuda,
        _ => frink_models::BudgetBackend::Cpu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(samples: &[f64]) -> Row {
        Row {
            test: "tg128".to_string(),
            samples: samples.to_vec(),
            digest: WorkloadDigest::new(),
            n_prompt: 0,
            n_gen: 128,
            n_depth: 0,
            test_time: String::new(),
        }
    }

    /// The centre is llama-bench's mean, not the median: a skewed set
    /// tells them apart.
    #[test]
    fn the_centre_is_the_mean_that_llama_bench_prints() {
        assert_eq!(row(&[10.0, 20.0, 60.0]).mean(), 30.0);
        assert_eq!(row(&[]).mean(), 0.0);
    }

    #[test]
    fn stddev_is_zero_for_identical_samples_and_undefined_counts_report_zero() {
        assert_eq!(row(&[5.0, 5.0, 5.0]).stddev(), 0.0);
        assert_eq!(row(&[5.0]).stddev(), 0.0);
        assert_eq!(row(&[]).stddev(), 0.0);
    }

    #[test]
    fn stddev_is_the_sample_form_that_llama_bench_prints() {
        // mean 20, deviations -10/0/+10 -> sqrt(200/2) = 10, llama-bench's
        // `stdev` (n - 1), not the population form sqrt(200/3).
        let got = row(&[10.0, 20.0, 30.0]).stddev();
        assert!(
            (got - 10.0).abs() < 1e-9,
            "sample stddev expected, got {got}"
        );
    }
}
