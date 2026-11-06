//! `frink bench -m`'s report side: the display helpers and the JSON
//! receipt. Moved out of `bench_model.rs` unchanged.

use super::{BenchArgs, Row};
use crate::bench_contract::{self, HostState};
use crate::bench_llama::{self, LlamaRun};
use frink_models::engine::Engine;
use frink_models::{Gemma4Engine, ModelConfig};
use std::path::Path;

/// Rough total for the report column only — embeddings plus the
/// per-layer estimate `ModelConfig` already computes. Not a substitute
/// for the GGUF's own tensor accounting, and not used in any tok/s math.
pub(super) fn estimate_params(config: &ModelConfig) -> u64 {
    let embeddings = 2 * config.vocab_size as u64 * config.hidden_dim as u64;
    embeddings + config.approx_active_params_per_token() as u64
}

/// Display-only param estimate for Gemma-4 (emb + head + per-layer Q/O/FFN).
pub(super) fn estimate_gemma4_params(engine: &Gemma4Engine) -> u64 {
    let hp = &engine.hp;
    let v = Engine::vocab_size(engine) as u64;
    let h = hp.hidden_dim as u64;
    let mut n = 2 * v * h;
    for (il, &ffn) in hp.ffn_dims.iter().enumerate() {
        let hd = hp.head_dim(il) as u64;
        let nh = hp.n_heads as u64;
        let nkv = hp.n_kv_heads as u64;
        n += nh * hd * h; // q
        if hp.has_kv(il) {
            n += 2 * nkv * hd * h; // k, v
        }
        n += nh * hd * h; // o
        let f = ffn as u64;
        n += 3 * f * h; // gate, up, down
    }
    n
}

pub(super) fn quant_label(file: &frink_gguf::ShardedGguf) -> String {
    file.metadata_str("general.file_type")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "quantized".to_string())
}

pub(super) fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

pub(super) fn human_bytes(b: u64) -> String {
    let mib = b as f64 / (1024.0 * 1024.0);
    if mib >= 1024.0 {
        format!("{:.2} GiB", mib / 1024.0)
    } else {
        format!("{mib:.2} MiB")
    }
}

pub(super) fn human_params(p: u64) -> String {
    let b = p as f64;
    if b >= 1e9 {
        format!("{:.2} B", b / 1e9)
    } else {
        format!("{:.2} M", b / 1e6)
    }
}

#[allow(clippy::too_many_arguments)] // one call site; every field is reported
pub(super) fn write_receipt(
    dest: &Path,
    args: &BenchArgs,
    model: &str,
    arch: &str,
    file: &frink_gguf::ShardedGguf,
    size_bytes: u64,
    params: u64,
    threads: usize,
    backend: &str,
    rows: &[Row],
    llama: Option<&LlamaRun>,
    load_s: f64,
    host: HostState,
    engine_env: &[(String, String)],
) -> anyhow::Result<()> {
    let mut tests = Vec::new();
    for row in rows {
        let l = llama.and_then(|m| m.tps.get(&row.test)).copied();
        tests.push(serde_json::json!({
            "test": row.test,
            "frink_tps": row.median(),
            "frink_stddev": row.stddev(),
            "frink_samples": row.samples,
            "llama_tps": l,
            "gap": l.map(|l| l / row.median()),
            // Same workload, same digest -- across models, sessions and
            // machines. Two rows that claim to measure `pp512` on the
            // same checkpoint and disagree here did not measure the
            // same work.
            "workload_digest": row.digest.hex(),
        }));
    }
    // ADR 0006: a llama.cpp number names the build that produced it,
    // or the receipt is not written.
    let llama_cpp = LlamaRun::receipt_field(llama);
    bench_llama::check_receipt(&tests, &llama_cpp, llama)?;
    // The shared envelope carries both write-time refusals: a GPU row
    // that cannot name its card, and a label that disagrees with the
    // backend that ran (#126).
    let mut receipt =
        bench_contract::receipt_common(&args.backend, backend, threads, load_s, host, engine_env)?;
    let serde_json::Value::Object(own) = serde_json::json!({
        // 3: adds `llama_cpp`. Schema-2 receipts render with an
        // unknown llama.cpp reference.
        "schema": 3,
        "kind": "engine",
        "id": args.id,
        "model_path": model,
        "arch": arch,
        "quant": quant_label(file),
        "size_bytes": size_bytes,
        "approx_params": params,
        "reps": args.reps,
        "tests": tests,
    }) else {
        unreachable!("json! with braces is an object");
    };
    receipt.extend(own);
    receipt.insert(bench_llama::RECEIPT_KEY.to_string(), llama_cpp);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(dest, serde_json::to_string_pretty(&receipt)? + "\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_readable_sizes_switch_units_at_a_gibibyte() {
        assert_eq!(human_bytes(512 * 1024 * 1024), "512.00 MiB");
        assert_eq!(human_bytes(2 * 1024 * 1024 * 1024), "2.00 GiB");
    }

    #[test]
    fn human_readable_params_switch_units_at_a_billion() {
        assert_eq!(human_params(135_000_000), "135.00 M");
        assert_eq!(human_params(8_000_000_000), "8.00 B");
    }

    #[test]
    fn truncate_leaves_short_strings_alone_and_clips_long_ones() {
        assert_eq!(truncate("llama Q8_0", 30), "llama Q8_0");
        assert_eq!(truncate(&"x".repeat(40), 30), "x".repeat(30));
    }
}
