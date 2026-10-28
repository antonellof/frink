//! The two CPU schedulers must produce the same tokens.
//!
//! Issue #27 replaces rayon's per-operation fork-join with a persistent
//! worker pool (`frink_core::cpu_pool`, selected by `FRINK_CPU_POOL`).
//! The reason that change exists is speed, and **an agent may not
//! benchmark**: measurement needs a quiet host and a loaded one reads
//! 25-45% low. So this suite asserts the other half, the half that can
//! be settled without a stopwatch -- that the new path computes exactly
//! what the old one did, greedily, token for token.
//!
//! It is not a smoke test of the pool (`frink_core::cpu_pool`'s unit
//! tests cover deadlock, dropped work, panics and lifetime). It is the
//! end-to-end statement: every rewritten matvec, every K-quant GEMV
//! tail, prefill and decode, agree bit-for-bit across the switch.
//!
//! # Why it writes its own checkpoints
//!
//! It used to run only over SmolLM2-135M from the ignored `models/`
//! directory and print `skip:` without it, which is every CI run and
//! every fresh worktree. A parity guard that skips wherever the code is
//! being changed reads as coverage and is none. The committed
//! `tests/fixtures/*.gguf` cannot replace it: they are 32 wide, every
//! matvec is under `frink_core::threads::should_parallelize`'s floor,
//! and both arms would run the same serial loop and agree by
//! construction.
//!
//! So [`synthetic`] writes a 2-layer llama per run, 512 wide with a
//! 1536 FFN, in Q8_0 and in Q4_K. Its narrowest projection (K/V, 256 x
//! 512) is over that floor, so under `FRINK_CPU_POOL=spin` every weight
//! matrix in the step goes through `frink_core::cpu_pool`. The real
//! checkpoints still run when they are present.
//!
//! # Why subprocesses
//!
//! `FRINK_CPU_POOL` is read once and cached for the process, which is
//! deliberate -- a scheduler that could change mid-run would make a
//! before/after measurement meaningless. So one process cannot exercise
//! both arms, and the comparison runs each in a child of this test
//! binary. Both children are spawned the same way, so neither arm gets
//! the advantage of being the one that ran in-process.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use frink_core::cache::KvCache;
use frink_gguf::{GgmlType, GgufValue, GgufWriter, ShardedGguf, TensorPlan};
use frink_models::config::ModelConfig;
use frink_models::decoder::Decoder;
use frink_models::tokenizer::{GgufBpeTokenizer, SpecialTokens};

const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 12;

/// Set on a child to make it actually run the generation.
const CHILD: &str = "FRINK_TEST_CPU_POOL_CHILD";
/// The checkpoint a child should generate from.
const CHILD_GGUF: &str = "FRINK_TEST_CPU_POOL_GGUF";
/// Set on a child to feed it [`SYNTHETIC_PROMPT`] as raw token ids: a
/// synthetic checkpoint carries no vocabulary to encode [`PROMPT`] with.
const CHILD_RAW_PROMPT: &str = "FRINK_TEST_CPU_POOL_RAW_PROMPT";
/// The line a child prints its token ids on.
const MARKER: &str = "FRINK_TOKENS";

/// `FRINK_TEST_MODELS_DIR`, else the workspace's `models/`. The
/// override exists because a git worktree does not carry the (ignored)
/// checkpoint directory.
fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("FRINK_TEST_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

/// The real checkpoints this also runs over, when they are present.
///
/// Two quantizations rather than one on purpose: Q8_0 and Q4_K_M take
/// *different* rewritten kernels (`gemv_q8_0x4_group` and the
/// interleaved `Q4_KX8` group plus its scalar tail), and a chunking bug
/// in one would not show up in the other.
fn real_checkpoints() -> Vec<PathBuf> {
    let dir = models_dir();
    [
        "hf_test/SmolLM2-135M-Instruct-Q8_0.gguf",
        "hf_test/SmolLM2-135M-Instruct-Q4_K_M.gguf",
    ]
    .iter()
    .map(|name| dir.join(name))
    .filter(|path| path.exists())
    .collect()
}

/// Token ids a synthetic checkpoint is prompted with. Distinct, and not
/// a run, so a wrong embedding row is a wrong prefill.
const SYNTHETIC_PROMPT: [usize; 6] = [3, 141, 59, 265, 358, 97];

/// The synthetic llama that makes this suite run everywhere.
mod synthetic {
    use super::*;

    pub const N_EMBD: usize = 512;
    pub const N_FF: usize = 1536;
    pub const N_HEAD: usize = 8;
    pub const N_HEAD_KV: usize = 4;
    pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
    pub const N_LAYER: usize = 2;
    pub const N_VOCAB: usize = 512;

    /// A deterministic, sign-balanced fill. A 64-bit LCG is enough: the
    /// weights only have to be fixed, not good.
    fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((x >> 40) as f32) / ((1u64 << 24) as f32);
                (unit * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn encode(dtype: GgmlType, cols: usize, w: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in w.chunks(cols) {
            let ok = match dtype {
                GgmlType::Q8_0 => frink_quant::encode_row_q8_0(row, &mut out),
                GgmlType::Q4K => frink_quant::encode_row_q4_k(row, None, &mut out),
                other => panic!("the fixture does not encode {other:?}"),
            };
            ok.expect("every fixture row length is a whole number of blocks");
        }
        out
    }

    /// One tensor: its name, output rows, input columns, dtype, values.
    /// GGUF's shape is `[cols, rows]`, fastest dimension first.
    struct Tensor {
        name: String,
        rows: usize,
        cols: usize,
        dtype: GgmlType,
        values: Vec<f32>,
    }

    fn metadata() -> BTreeMap<String, GgufValue> {
        let u = |n: usize| GgufValue::U32(n as u32);
        [
            (
                "general.architecture",
                GgufValue::String("llama".to_string()),
            ),
            ("llama.block_count", u(N_LAYER)),
            ("llama.embedding_length", u(N_EMBD)),
            ("llama.feed_forward_length", u(N_FF)),
            ("llama.attention.head_count", u(N_HEAD)),
            ("llama.attention.head_count_kv", u(N_HEAD_KV)),
            ("llama.attention.key_length", u(HEAD_DIM)),
            ("llama.vocab_size", u(N_VOCAB)),
            ("llama.rope.freq_base", GgufValue::F32(10_000.0)),
            (
                "llama.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-5),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    fn tensors(dtype: GgmlType) -> Vec<Tensor> {
        let mut seed = 0u64;
        let mut matrix = |name: String, rows: usize, cols: usize, dtype: GgmlType| {
            seed += 1;
            let scale = 1.0 / (cols as f32).sqrt();
            Tensor {
                name,
                rows,
                cols,
                dtype,
                values: fill(seed, rows * cols, scale),
            }
        };
        let norm = |name: String, seed: u64| Tensor {
            name,
            rows: 1,
            cols: N_EMBD,
            dtype: GgmlType::F32,
            values: fill(1000 + seed, N_EMBD, 0.2)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect(),
        };

        let kv = N_HEAD_KV * HEAD_DIM;
        let mut out = vec![matrix(
            "token_embd.weight".to_string(),
            N_VOCAB,
            N_EMBD,
            GgmlType::F32,
        )];
        for l in 0..N_LAYER {
            out.push(norm(format!("blk.{l}.attn_norm.weight"), 2 * l as u64));
            for (name, rows, cols) in [
                ("attn_q", N_EMBD, N_EMBD),
                ("attn_k", kv, N_EMBD),
                ("attn_v", kv, N_EMBD),
                ("attn_output", N_EMBD, N_EMBD),
            ] {
                out.push(matrix(format!("blk.{l}.{name}.weight"), rows, cols, dtype));
            }
            out.push(norm(format!("blk.{l}.ffn_norm.weight"), 2 * l as u64 + 1));
            for (name, rows, cols) in [
                ("ffn_gate", N_FF, N_EMBD),
                ("ffn_up", N_FF, N_EMBD),
                ("ffn_down", N_EMBD, N_FF),
            ] {
                out.push(matrix(format!("blk.{l}.{name}.weight"), rows, cols, dtype));
            }
        }
        out.push(norm("output_norm.weight".to_string(), 99));
        out.push(matrix("output.weight".to_string(), N_VOCAB, N_EMBD, dtype));
        out
    }

    /// Writes the checkpoint with every weight matrix in `dtype` and
    /// returns its path.
    pub fn write(dtype: GgmlType) -> PathBuf {
        let encoded: Vec<(TensorPlan, Vec<u8>)> = tensors(dtype)
            .into_iter()
            .map(|t| {
                let shape = if t.rows == 1 {
                    vec![t.cols as u64]
                } else {
                    vec![t.cols as u64, t.rows as u64]
                };
                let bytes = if t.dtype == GgmlType::F32 {
                    t.values.iter().flat_map(|v| v.to_le_bytes()).collect()
                } else {
                    encode(t.dtype, t.cols, &t.values)
                };
                let plan = TensorPlan {
                    name: t.name,
                    shape,
                    dtype: t.dtype,
                    byte_len: bytes.len(),
                };
                (plan, bytes)
            })
            .collect();
        let plan = encoded.iter().map(|(p, _)| p.clone()).collect();

        // `target/`, so a leftover is swept by `cargo clean`; the pid
        // keeps two concurrent runs apart.
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        std::fs::create_dir_all(&dir).expect("create the fixture directory");
        let path = dir.join(format!(
            "cpu_pool_parity_{dtype:?}-{}.gguf",
            std::process::id()
        ));
        let file = std::fs::File::create(&path).expect("create the fixture");
        let mut w = GgufWriter::create(file, &metadata(), plan).expect("write the header");
        for (p, bytes) in &encoded {
            w.write_tensor(&p.name, bytes).expect("write a tensor");
        }
        w.finish().expect("finish the fixture");
        path
    }
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("logits are finite"))
        .map(|(i, _)| i)
        .expect("non-empty logits")
}

/// Greedy continuation: prefill through `forward_batch`, then
/// `forward_token` per step, exactly the shape decode has.
fn greedy_tokens(path: &Path, raw_prompt: bool) -> Vec<usize> {
    let file = ShardedGguf::open(path).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("model config");
    let decoder = Decoder::from_gguf(path, config.clone()).expect("load decoder");

    let prompt: Vec<usize> = if raw_prompt {
        SYNTHETIC_PROMPT.to_vec()
    } else {
        GgufBpeTokenizer::from_gguf(&file)
            .expect("tokenizer")
            .encode(PROMPT, SpecialTokens::Parse)
            .into_iter()
            .map(|t| t as usize)
            .collect()
    };
    let mut caches: Vec<KvCache> = config.new_kv_caches();

    let logits = decoder.forward_batch(&prompt, 0, &mut caches);
    let mut last = logits.last().expect("non-empty prompt").clone();

    let mut generated = Vec::with_capacity(MAX_NEW_TOKENS);
    for step in 0..MAX_NEW_TOKENS {
        let next = argmax(&last);
        generated.push(next);
        last = decoder.forward_token(next, prompt.len() + step, &mut caches);
    }
    generated
}

/// Run this test binary again with `backend` selected, and read back the
/// tokens it generated.
fn tokens_from_child(path: &Path, raw_prompt: bool, backend: &str) -> Vec<usize> {
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = Command::new(exe);
    if raw_prompt {
        cmd.env(CHILD_RAW_PROMPT, "1");
    }
    let out = cmd
        .args(["--exact", "generates_tokens_for_the_parent", "--nocapture"])
        .env(CHILD, "1")
        .env(CHILD_GGUF, path)
        .env("FRINK_CPU_POOL", backend)
        // Both arms run the integer `vec_dot` kernels, which is what the
        // product ships and where all the rewritten GEMVs live. Without
        // this the library default (reference-exact f32 dequant-dot)
        // would leave the interleaved paths untested.
        .env("FRINK_CPU_INT_DOT", "1")
        .env("FRINK_TEST_MODELS_DIR", models_dir())
        .output()
        .expect("spawn child test process");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child ({backend}) failed: {}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let line = stdout
        .lines()
        .find_map(|l| l.strip_prefix(MARKER))
        .unwrap_or_else(|| panic!("child ({backend}) printed no {MARKER} line:\n{stdout}"));
    line.split_whitespace()
        .map(|t| t.parse::<usize>().expect("token id"))
        .collect()
}

/// The child half. Inert unless [`CHILD`] is set, so an ordinary
/// `cargo test` run does not load a model twice for nothing.
#[test]
fn generates_tokens_for_the_parent() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let path = PathBuf::from(std::env::var_os(CHILD_GGUF).expect("child needs a GGUF path"));
    let raw_prompt = std::env::var_os(CHILD_RAW_PROMPT).is_some();
    let tokens = greedy_tokens(&path, raw_prompt);
    let ids: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    println!("{MARKER} {}", ids.join(" "));
}

/// Both arms over one checkpoint, asserting the greedy continuations
/// are identical.
fn assert_schedulers_agree(path: &Path, raw_prompt: bool) {
    let rayon = tokens_from_child(path, raw_prompt, "rayon");
    let spin = tokens_from_child(path, raw_prompt, "spin");
    assert_eq!(
        rayon.len(),
        MAX_NEW_TOKENS,
        "{}: the rayon arm generated nothing to compare",
        path.display()
    );
    // A continuation stuck on one token agrees across almost any bug,
    // because a perturbed matvec rarely moves a saturated argmax.
    let mut distinct = rayon.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() >= 3,
        "{}: the continuation {rayon:?} is too degenerate to tell two schedulers apart",
        path.display()
    );
    assert_eq!(
        rayon,
        spin,
        "{}: the persistent pool and rayon disagree about the greedy continuation",
        path.display()
    );
}

/// **The assertion the pool rests on**, on checkpoints this test writes,
/// so it runs in CI and in every worktree rather than skipping.
///
/// Sabotage, run 2026-10-28: append `.saturating_sub(1)` to the spin
/// arm's `let hi = ...` in `frink_core::par::indices`, `indices_init`
/// or `chunks_mut`, one at a time, and each goes red with a different
/// continuation, because a matvec that skips rows moves the argmax.
/// The same edit in `items_mut`, `chunks_mut2_by` and `chunks_mut_init`
/// stays green: a dense llama decode step does not reach them, so this
/// suite says nothing about their spin arms.
#[test]
fn the_two_cpu_schedulers_produce_token_identical_output() {
    for dtype in [GgmlType::Q8_0, GgmlType::Q4K] {
        let path = synthetic::write(dtype);
        assert_schedulers_agree(&path, true);
        std::fs::remove_file(&path).ok();
    }
}

/// The same assertion on real checkpoints, where real activations
/// exercise the kernels' value range. Needs `models/`, so it says what
/// it skipped rather than passing silently.
#[test]
fn the_two_cpu_schedulers_agree_on_real_checkpoints() {
    let models = real_checkpoints();
    if models.is_empty() {
        eprintln!(
            "skip: no checkpoint under {} (the synthetic suite still ran)",
            models_dir().display()
        );
        return;
    }
    for path in models {
        assert_schedulers_agree(&path, false);
    }
}
