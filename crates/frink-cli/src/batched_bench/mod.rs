//! `frink batched-bench` -- a `llama-batched-bench` work-alike.
//!
//! Throughput as a function of batch size: for every combination of
//! prompt length (`-npp`), generation length (`-ntg`) and parallel
//! sequences (`-npl`), one row reporting prompt speed, decode speed
//! and the total, in the same ten columns `llama-batched-bench`
//! prints so the two tables can be read side by side. The workload is
//! `tools/batched-bench/batched-bench.cpp:132-235` and the row format
//! is `:245`; `workload.rs` and `report.rs` cite the lines they mirror.
//!
//! # What it drives
//!
//! Not HTTP (`frink serve-bench` does that) and not a third decode
//! path. Prefill goes through `Decoder::forward_batch_last_host_kv`,
//! one sequence at a time, which is what the continuous batcher does
//! for a row it admits; decode goes through `Decoder::forward_multi_seq`,
//! which is the batcher's per-tick step (`frink-server/src/serving/
//! batch/worker.rs:470`). So the numbers are the engine seams the
//! server runs on, measured without the server around them.
//!
//! # What it reuses from `frink bench`
//!
//! The measurement contract (`bench_contract`): quiet, cool, not-full
//! host before the clock; host state on both sides of the run; a
//! receipt refused at write time when its label disagrees with the
//! backend that ran. And the guards (`bench_guard`): token streams
//! asserted before the run, cold caches asserted per repetition, the
//! prompt and decode lengths re-read from the KV afterwards, and the
//! determinism pair -- identical input (workload digest) and identical
//! output (greedy pick) between the discarded warmup and the timed
//! run. Every row runs twice for exactly that reason: a single timed
//! pass has nothing to be compared against.
//!
//! # Flags that refuse
//!
//! `-b` (`n_batch`), `-kvu`, `-fa` and `-tb` are accepted by the
//! parser and refused by name, because frink has nothing behind them
//! and a flag that is accepted and ignored has burned this repo before.
//! [`refuse_unhonoured_flags`] is an exhaustive destructure, so a new
//! flag cannot be added without deciding which side it is on.

mod report;
mod workload;

use crate::bench_contract;
use crate::bench_model::active_backend;
use crate::host_state;
use anyhow::Context;
use clap::{Parser, ValueEnum};
use frink_models::{select_engine_kind, Decoder, ModelConfig, SelectedEngineKind};
use std::path::{Path, PathBuf};
use std::time::Instant;

use workload::{Combo, Shape};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// The markdown table `llama-batched-bench` prints by default.
    Md,
    /// One JSON object per row, `--output-format jsonl` upstream.
    Jsonl,
}

/// `llama-batched-bench`'s flags. Its single-dash spellings (`-npp`,
/// `-ntg`, `-npl`, `-pps`, `-tgs`, `-ub`, `-ngl`, `-kvu`, `-fa`, `-tb`)
/// are the aliases below, which `crate::argv` reads them through.
#[derive(Parser, Debug)]
pub struct BatchedBenchArgs {
    /// GGUF to benchmark.
    #[arg(short = 'm', long)]
    pub model: String,
    /// Prompt lengths to sweep (`-npp 128,256,512`).
    #[arg(
        long = "n-pp",
        visible_alias = "npp",
        value_delimiter = ',',
        required = true
    )]
    pub n_pp: Vec<usize>,
    /// Generation lengths to sweep (`-ntg 128,256`).
    #[arg(
        long = "n-tg",
        visible_alias = "ntg",
        value_delimiter = ',',
        required = true
    )]
    pub n_tg: Vec<usize>,
    /// Parallel sequence counts to sweep (`-npl 1,2,4,8`).
    #[arg(
        long = "n-pl",
        visible_alias = "npl",
        value_delimiter = ',',
        required = true
    )]
    pub n_pl: Vec<usize>,
    /// `-pps`: one prompt shared by every sequence (prefilled once,
    /// then copied) instead of one prompt per sequence.
    #[arg(long = "pp-shared", visible_alias = "pps")]
    pub pp_shared: bool,
    /// `-tgs`: decode each sequence to completion in turn
    /// (`0 0 0 ... 1 1 1 ...`) instead of one step across all of them
    /// per call (`0123 0123 ...`).
    #[arg(long = "tg-separate", visible_alias = "tgs")]
    pub tg_separate: bool,
    /// `n_kv_max`: combinations needing more positions than this are
    /// skipped, as upstream does. `0` = the GGUF's own context length.
    #[arg(short = 'c', long, default_value_t = 0)]
    pub ctx_size: usize,
    /// `-ub`: prompt tokens per forward call. Each sequence's prompt
    /// is fed in chunks of this size, the physical batch upstream.
    #[arg(long = "ubatch-size", visible_alias = "ub", default_value_t = 512)]
    pub ubatch_size: usize,
    /// CPU threads (`0` = performance-core default).
    #[arg(short = 't', long, default_value_t = 0)]
    pub threads: usize,
    /// GPU layers: `0` forces CPU, the model's layer count or more
    /// offloads all of them, and a partial count is refused.
    #[arg(long = "n-gpu-layers", visible_alias = "ngl", default_value_t = 0)]
    pub n_gpu_layers: usize,
    #[arg(long = "output-format", value_enum, default_value_t = OutputFormat::Md)]
    pub output_format: OutputFormat,
    /// Backend label for the receipt (`cpu` / `metal` / `cuda`). Must
    /// match the backend that runs, or the receipt is refused.
    #[arg(long, default_value = "cpu")]
    pub backend_label: String,
    /// Write a JSON receipt of every row to this path.
    #[arg(long)]
    pub receipt: Option<PathBuf>,
    /// Refuse to start when the host's 1-minute load average is at or
    /// above this. `0` disables the check and marks the receipt as
    /// not quiet-host.
    #[arg(long, default_value_t = host_state::DEFAULT_MAX_LOAD)]
    pub max_load: f64,

    // ---- accepted so they can be refused by name ----
    /// REFUSED. `n_batch` is upstream's logical batch; frink has no
    /// logical/physical split, `-ub` is the chunk it actually feeds.
    #[arg(short = 'b', long = "batch-size", hide_short_help = true)]
    pub batch_size: Option<usize>,
    /// REFUSED. `-kvu` shares one KV buffer across sequences; this
    /// tool gives every sequence its own cache, so `N_KV` is always
    /// `B*(PP+TG)`.
    #[arg(long = "kv-unified", alias = "kvu", hide_short_help = true)]
    pub kv_unified: bool,
    /// REFUSED. frink has no flash-attention switch to honour.
    #[arg(long = "flash-attn", alias = "fa", hide_short_help = true)]
    pub flash_attn: Option<String>,
    /// REFUSED. frink has one thread pool for prefill and decode.
    #[arg(long = "threads-batch", alias = "tb", hide_short_help = true)]
    pub threads_batch: Option<usize>,
}

/// Every flag the parser accepts is either honoured or refused here,
/// and the destructure has no `..`, so adding a field without
/// deciding which does not compile.
fn refuse_unhonoured_flags(args: &BatchedBenchArgs) -> anyhow::Result<()> {
    let BatchedBenchArgs {
        model: _,
        n_pp: _,
        n_tg: _,
        n_pl: _,
        pp_shared: _,
        tg_separate: _,
        ctx_size: _,
        ubatch_size: _,
        threads: _,
        n_gpu_layers: _,
        output_format: _,
        backend_label: _,
        receipt: _,
        max_load: _,
        batch_size,
        kv_unified,
        flash_attn,
        threads_batch,
    } = args;
    if let Some(b) = batch_size {
        anyhow::bail!(
            "-b {b} (n_batch) is refused: frink has no logical batch distinct from the \
             physical one. `-ub` is the number of prompt tokens fed per forward call and \
             is the only batch size this tool honours; drop -b."
        );
    }
    if *kv_unified {
        anyhow::bail!(
            "-kvu is refused: this tool gives every sequence its own KV cache, so a shared \
             prompt is COPIED into each one and N_KV is B*(PP+TG) whether or not -pps is \
             set. A unified buffer would report a smaller N_KV for work this engine does \
             not do that way; drop -kvu."
        );
    }
    if let Some(v) = flash_attn {
        anyhow::bail!(
            "-fa {v} is refused: frink has no flash-attention switch, so the flag would be \
             accepted and change nothing. Drop it; the row is measured with the attention \
             kernel the engine always uses."
        );
    }
    if let Some(t) = threads_batch {
        anyhow::bail!(
            "-tb {t} is refused: frink has one thread pool for prefill and decode (-t). \
             A separate batch thread count would be accepted and ignored; drop -tb."
        );
    }
    Ok(())
}

/// A sweep value of zero is a row that would print NaN (`0/0`) in the
/// same column as measured ones; upstream prints it, this refuses it.
fn ensure_positive(flag: &str, values: &[usize]) -> anyhow::Result<()> {
    anyhow::ensure!(!values.is_empty(), "{flag} needs at least one value");
    if let Some(z) = values.iter().position(|&v| v == 0) {
        anyhow::bail!(
            "{flag} value {z} is 0: a zero-length prompt or generation divides zero work \
             by a duration and prints NaN beside the measured rows"
        );
    }
    Ok(())
}

pub fn run(args: BatchedBenchArgs) -> anyhow::Result<()> {
    refuse_unhonoured_flags(&args)?;
    ensure_positive("-npp", &args.n_pp)?;
    ensure_positive("-ntg", &args.n_tg)?;
    ensure_positive("-npl", &args.n_pl)?;
    anyhow::ensure!(args.ubatch_size > 0, "-ub 0 would feed no tokens per call");
    // The label only matters if a receipt is written, and then it is
    // checked twice by the same predicate: here, so a sweep is not run
    // for a receipt that will be refused, and at write time.
    if args.receipt.is_some() {
        bench_contract::ensure_label_matches_backend(&args.backend_label, active_backend())?;
    }

    let (load_start, thermal_start) = bench_contract::preflight_host(args.max_load)?;
    let model = crate::pull::resolve_model_path(&args.model)?;
    let path = Path::new(&model);
    if !path.exists() {
        anyhow::bail!("model not found: {model}");
    }

    let file = frink_gguf::ShardedGguf::open(path)?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    let kind = select_engine_kind(&arch).map_err(|e| anyhow::anyhow!("{e}"))?;
    // `forward_multi_seq` is a `Decoder` method; the dedicated engines
    // have no multi-sequence step, and a per-sequence `forward_token`
    // loop would measure something the batcher never runs.
    if kind != SelectedEngineKind::GenericDecoder {
        anyhow::bail!(
            "`frink batched-bench` covers the generic decoder only; arch {arch} selects \
             {kind:?}, which has no multi-sequence decode step to measure"
        );
    }
    let mut config = ModelConfig::from_gguf(&file)
        .with_context(|| format!("reading model config for arch {arch}"))?;
    config.kv_dtype =
        frink_models::host_kv_dtype_from_env(crate::bench_model::active_budget_backend());

    let n_kv_max = if args.ctx_size > 0 {
        args.ctx_size
    } else {
        file.metadata_u64(&format!("{arch}.context_length"))
            .map(|v| v as usize)
            .with_context(|| {
                format!("{arch}.context_length is not in the GGUF; pass -c <n_kv_max>")
            })?
    };

    let shape = Shape {
        pp_shared: args.pp_shared,
        tg_separate: args.tg_separate,
        ubatch: args.ubatch_size,
    };
    let combos = plan_combos(
        &args.n_pp,
        &args.n_tg,
        &args.n_pl,
        shape.pp_shared,
        n_kv_max,
    );
    anyhow::ensure!(
        !combos.is_empty(),
        "every combination needs more than n_kv_max = {n_kv_max} positions; raise -c or \
         shrink the sweep"
    );

    // The KV for the largest row, on top of the weights, before either
    // is allocated: a run that pages is timing the page file.
    let largest_kv = combos
        .iter()
        .map(|c| c.n_kv(shape.pp_shared))
        .max()
        .unwrap_or(0);
    let kv_gb = (largest_kv * kv_bytes_per_position(&config)) as f64 / 1024.0 / 1024.0 / 1024.0;
    bench_contract::ensure_weights_fit(args.max_load, path, kv_gb)?;

    let load_t = Instant::now();
    let decoder = Decoder::from_gguf(path, config)?;
    let load_s = load_t.elapsed().as_secs_f64();

    let backend = active_backend();
    let threads = frink_core::threads::resolve_cpu_threads();
    let header = report::Header {
        n_kv_max,
        n_ubatch: shape.ubatch,
        is_pp_shared: shape.pp_shared,
        is_tg_separate: shape.tg_separate,
        n_gpu_layers: args.n_gpu_layers,
        n_threads: threads,
        backend,
    };
    if args.output_format == OutputFormat::Md {
        println!();
        println!("{}", report::header_line(&header));
        println!();
        for line in report::TABLE_HEADER {
            println!("{line}");
        }
    }

    let mut rows = Vec::with_capacity(combos.len());
    for combo in combos {
        let measured = workload::measure(&decoder, combo, &shape)?;
        match args.output_format {
            OutputFormat::Md => println!("{}", report::md_row(&measured)),
            OutputFormat::Jsonl => println!("{}", report::jsonl_row(&header, &measured)),
        }
        rows.push(measured);
    }

    eprintln!(
        "\nfrink batched-bench: load {load_s:.2}s, every row timed once after {} discarded \
         warmup pass",
        crate::bench_guard::WARMUP_REPS
    );
    let bench_contract::HostAfter {
        load_end,
        thermal_end,
        engine_env,
    } = bench_contract::report_host_after("frink batched-bench", load_start, &thermal_start);

    if let Some(dest) = &args.receipt {
        let host = bench_contract::HostState {
            load_start,
            load_end,
            thermal_start,
            thermal_end,
        };
        let common = bench_contract::receipt_common(
            &args.backend_label,
            backend,
            threads,
            load_s,
            host,
            &engine_env,
        )?;
        report::write_receipt(dest, common, &header, &model, &arch, &rows)?;
        eprintln!("frink batched-bench: receipt written to {}", dest.display());
    }
    Ok(())
}

/// The sweep in upstream's order (`batched-bench.cpp:132-143`): `pp`
/// outermost, then `tg`, then `pl`, and a combination whose KV would
/// not fit `n_kv_max` is left out -- upstream `continue`s silently,
/// this says which rows it dropped and why.
fn plan_combos(
    n_pp: &[usize],
    n_tg: &[usize],
    n_pl: &[usize],
    pp_shared: bool,
    n_kv_max: usize,
) -> Vec<Combo> {
    let mut out = Vec::new();
    for &pp in n_pp {
        for &tg in n_tg {
            for &pl in n_pl {
                let combo = Combo { pp, tg, pl };
                let n_kv = combo.n_kv(pp_shared);
                if n_kv > n_kv_max {
                    eprintln!(
                        "frink batched-bench: skipping {} -- needs {n_kv} KV positions, \
                         n_kv_max is {n_kv_max}",
                        combo.label()
                    );
                    continue;
                }
                out.push(combo);
            }
        }
    }
    out
}

/// Host-side KV bytes per position: K and V, `f32`, every layer.
fn kv_bytes_per_position(config: &ModelConfig) -> usize {
    config.n_layers * config.n_kv_heads * config.head_dim * 2 * std::mem::size_of::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> BatchedBenchArgs {
        BatchedBenchArgs::try_parse_from(
            std::iter::once("batched-bench").chain(argv.iter().copied()),
        )
        .expect("parses")
    }

    const BASE: &[&str] = &[
        "-m", "x.gguf", "--n-pp", "8", "--n-tg", "4", "--n-pl", "1,2",
    ];

    /// Each refused flag must reach its refusal: a gate keyed on a flag
    /// the parser does not accept would never fire.
    #[test]
    fn every_unhonoured_flag_is_refused_by_name() {
        for (extra, needle) in [
            (vec!["-b", "2048"], "-b 2048 (n_batch) is refused"),
            (vec!["--kv-unified"], "-kvu is refused"),
            (vec!["--flash-attn", "on"], "-fa on is refused"),
            (vec!["--threads-batch", "4"], "-tb 4 is refused"),
        ] {
            let argv: Vec<&str> = BASE.iter().copied().chain(extra).collect();
            let err = refuse_unhonoured_flags(&parse(&argv))
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{err}");
        }
        assert!(refuse_unhonoured_flags(&parse(BASE)).is_ok());
    }

    #[test]
    fn the_sweep_lists_are_comma_separated_and_required() {
        let a = parse(BASE);
        assert_eq!(a.n_pl, vec![1, 2]);
        assert_eq!(a.ubatch_size, 512, "upstream's n_ubatch default");
        assert!(BatchedBenchArgs::try_parse_from(["batched-bench", "-m", "x.gguf"]).is_err());
    }

    /// A zero in any list is a NaN row; upstream prints it, this
    /// refuses before loading anything.
    #[test]
    fn a_zero_sweep_value_is_refused_before_the_model_loads() {
        let err = ensure_positive("-ntg", &[128, 0]).unwrap_err().to_string();
        assert!(err.contains("-ntg value 1 is 0"), "{err}");
        assert!(ensure_positive("-npp", &[]).is_err());
        assert!(ensure_positive("-npp", &[1]).is_ok());
    }

    /// `batched-bench.cpp:132-143`: pp outermost, then tg, then pl,
    /// and the rows that do not fit are dropped from the sweep rather
    /// than run against a context they exceed.
    #[test]
    fn combos_come_in_upstream_order_and_oversized_rows_are_dropped() {
        let got = plan_combos(&[8, 16], &[4], &[1, 2], false, 40);
        let labels: Vec<String> = got.iter().map(Combo::label).collect();
        // 16*4... pp16 tg4 pl2 needs 2*(16+4)=40, fits exactly; pp16 pl2
        // with tg4 is the largest. Nothing exceeds 40 here.
        assert_eq!(
            labels,
            ["pp8 tg4 pl1", "pp8 tg4 pl2", "pp16 tg4 pl1", "pp16 tg4 pl2"]
        );
        let got = plan_combos(&[8, 16], &[4], &[1, 2], false, 39);
        let labels: Vec<String> = got.iter().map(Combo::label).collect();
        assert_eq!(labels, ["pp8 tg4 pl1", "pp8 tg4 pl2", "pp16 tg4 pl1"]);
    }

    #[test]
    fn kv_bytes_count_k_and_v_in_f32_for_every_layer() {
        let mut config = frink_models::config::test_dense_fixture();
        config.n_layers = 3;
        config.n_kv_heads = 2;
        config.head_dim = 8;
        assert_eq!(kv_bytes_per_position(&config), 3 * 2 * 8 * 2 * 4);
    }
}
