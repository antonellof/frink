//! `frink bench`: the flag surface, and the synthetic kernel benchmark
//! that runs when no model is named.
//!
//! Moved out of `main.rs` verbatim (a pure move) before `frink bench`
//! grew llama-bench's flags, so they land in a file of their own rather
//! than in a 1500-line one.

use clap::Args;

use frink_models::{deepseek_v4_pro, glm_5_2, kimi_k3, Decoder};

use crate::{bench_model, bench_render, bench_suite, host_state};

/// `frink bench`'s flags (llama-bench's `-m`/`-p`/`-n`/`-r`/...).
#[derive(Args)]
pub struct BenchCmd {
    /// Real GGUF to benchmark. With this set, `bench` becomes a
    /// `llama-bench` work-alike (pp/tg on real weights) and the
    /// synthetic matvec microbenchmark below is skipped.
    #[arg(short = 'm', long)]
    pub model: Option<String>,
    /// Prompt tokens for the `pp<N>` batched-prefill row (`0` skips).
    #[arg(short = 'p', long = "n-prompt", default_value_t = 512)]
    pub n_prompt: usize,
    /// Decode steps for the `tg<N>` row (`0` skips).
    #[arg(short = 'n', long = "n-gen", default_value_t = 128)]
    pub n_gen: usize,
    /// Timed repetitions per row; one extra warmup is discarded.
    #[arg(short = 'r', long = "repetitions", default_value_t = 3)]
    pub reps: usize,
    /// CPU threads (`0` = performance-core default)
    #[arg(short = 't', long, default_value_t = 0)]
    pub threads: usize,
    /// GPU layers, llama-bench's `-ngl`: `0` forces CPU, the model's
    /// layer count or more offloads all of them, and a partial count
    /// in between is refused (frink has no partial placement).
    #[arg(long = "n-gpu-layers", visible_alias = "ngl", default_value_t = 0)]
    pub n_gpu_layers: usize,
    /// Context size (`0` = GGUF default)
    #[arg(short = 'c', long, default_value_t = 0)]
    pub ctx_size: usize,
    #[arg(long, default_value_t = 4096)]
    pub hidden: usize,
    #[arg(long, default_value_t = 14336)]
    pub ffn_dim: usize,
    #[arg(long, default_value_t = 20)]
    pub iters: usize,
    /// Also run `llama-bench` on the same GGUF and print the gap.
    #[arg(long)]
    pub compare: bool,
    /// Run every entry in `benchmarks/suite.json` (each in a fresh
    /// child process), write engine receipts, then re-render the
    /// engine table in `benchmarks/RESULTS.md`.
    #[arg(long)]
    pub suite: bool,
    /// Re-render the engine table from existing receipts only.
    #[arg(long)]
    pub render: bool,
    /// Restrict `--suite` to one suite id.
    #[arg(long)]
    pub id: Option<String>,
    /// Restrict `--suite` to one backend.
    #[arg(long)]
    pub backend: Option<String>,
    /// Skip suite entries whose estimated RAM exceeds ~75% of host RAM.
    #[arg(long)]
    pub fit_host: bool,
    /// Skip suite entries whose GGUF is not present.
    #[arg(long)]
    pub skip_missing: bool,
    /// benchmarks/ directory.
    #[arg(long, default_value = "benchmarks")]
    pub bench_dir: String,
    /// Internal (`--suite` children): suite id to record in the receipt.
    #[arg(long)]
    pub suite_id: Option<String>,
    /// Internal (`--suite` children): backend label for the receipt.
    #[arg(long, default_value = "cpu")]
    pub backend_label: String,
    /// Write a JSON receipt to this path.
    #[arg(long)]
    pub receipt: Option<String>,
    /// Refuse to start a timed run when the host's 1-minute load
    /// average is at or above this. `0` disables the check and
    /// marks the receipt as not quiet-host.
    #[arg(long, default_value_t = host_state::DEFAULT_MAX_LOAD)]
    pub max_load: f64,
    /// Flash attention, llama-bench's `-fa` (`on`, `off`, `auto`; the
    /// default is `auto` on both). `on` and `auto` are what frink does
    /// wherever the backend has a fused kernel; `off` is refused, and so
    /// is llama-bench's comma-separated sweep, since one `frink bench`
    /// run measures one configuration.
    #[arg(long = "flash-attn", visible_alias = "fa", value_name = "MODE")]
    pub flash_attn: Option<String>,
    /// Print the devices this build can run on and exit, llama-bench's
    /// `--list-devices`.
    #[arg(long = "list-devices", default_value_t = false)]
    pub list_devices: bool,
}

/// `frink bench -fa VALUE`: one value through the shared
/// [`frink_core::argv::check_flash_attn`], with llama-bench's sweep
/// (`-fa 0,1`) refused by name rather than read as an unknown value.
fn check_bench_flash_attn(value: &str) -> anyhow::Result<()> {
    if value.contains(',') {
        anyhow::bail!(
            "--flash-attn {value}: frink bench measures one configuration per run;              llama-bench's comma-separated sweep is not implemented. Run once per value"
        );
    }
    frink_core::argv::check_flash_attn(value).map_err(|e| anyhow::anyhow!(e))
}

impl BenchCmd {
    pub fn run(self) -> anyhow::Result<()> {
        let Self {
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
            flash_attn,
            list_devices,
        } = self;
        if list_devices {
            frink_models::devices::print_available_devices();
            return Ok(());
        }
        if let Some(mode) = &flash_attn {
            check_bench_flash_attn(mode)?;
        }
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
            bench_model::apply_env(threads, n_gpu_layers, Some(&model))?;
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
        synthetic(hidden, ffn_dim, iters);
        Ok(())
    }
}

/// With no `-m`: fused quantized matvec vs. dequant-then-matmul, at a
/// realistic single-expert-FFN matrix size, measuring both wall
/// time and resident memory. Also benchmarks end-to-end decode
/// throughput (tokens/sec) for each preset at synthetic-but-full-
/// scale-shape dimensions.
fn synthetic(hidden: usize, ffn_dim: usize, iters: usize) {
    use frink_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
    use std::time::Instant;

    println!("=== matvec microbenchmark: [{ffn_dim} x {hidden}] weight, {iters} iterations ===");

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
    let row_bytes = (hidden / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES;
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

    println!(
        "\n=== Q4_0 matvec microbenchmark: [{ffn_dim} x {hidden}] weight, {iters} iterations ==="
    );
    let q4_row_bytes = (hidden / frink_quant::Q4_0_BLOCK_ELEMS) * frink_quant::Q4_0_BLOCK_BYTES;
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
    println!("\nNote: these are CPU-reference numbers on this sandbox's vCPUs, not GPU numbers.");
    println!(
        "They demonstrate the pipeline's relative cost structure (attention vs MoE FFN, memory"
    );
    println!("footprint of quantized vs f32 weights), not absolute performance on target hardware");
    println!("like a DGX Spark. See benchmarks/RESULTS.md for methodology and caveats.");
}

#[cfg(test)]
mod tests {
    use super::check_bench_flash_attn;

    #[test]
    fn bench_flash_attn_takes_one_llama_value_and_refuses_off_and_sweeps() {
        for ok in ["on", "auto", "1", "enabled", "-1"] {
            check_bench_flash_attn(ok).unwrap_or_else(|e| panic!("{ok}: {e}"));
        }
        for (bad, says) in [
            ("off", "backend property"),
            ("0", "backend property"),
            ("0,1", "sweep"),
            ("maybe", "expected on"),
        ] {
            let err = check_bench_flash_attn(bad).unwrap_err().to_string();
            assert!(err.contains(says), "{bad}: {err}");
        }
    }
}
