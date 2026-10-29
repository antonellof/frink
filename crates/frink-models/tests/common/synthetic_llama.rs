//! The synthetic llama several suites write per run: a 2-layer llama,
//! 512 wide with a 1536 FFN, every weight matrix in one quant kind.
//!
//! 512 wide so its narrowest projection (K/V, 256 x 512) is over
//! `frink_core::threads::should_parallelize`'s floor and over a Metal
//! GEMM tile, which the committed 32-wide `tests/fixtures/*.gguf` are
//! not. The caller supplies the encoder, so a kind with no
//! `frink_quant` encoder can still be written by the suite that needs
//! it.

use std::collections::BTreeMap;
use std::path::PathBuf;

use frink_gguf::{GgmlType, GgufValue, GgufWriter, TensorPlan};

/// Encodes `values`, a whole number of rows `cols` wide, as `dtype`.
pub type Encode = fn(GgmlType, usize, &[f32]) -> Vec<u8>;

/// Token ids a synthetic checkpoint is prompted with. Distinct, and not
/// a run, so a wrong embedding row is a wrong prefill.
pub const PROMPT: [usize; 6] = [3, 141, 59, 265, 358, 97];

pub const N_EMBD: usize = 512;
pub const N_FF: usize = 1536;
pub const N_HEAD: usize = 8;
pub const N_HEAD_KV: usize = 4;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
pub const N_LAYER: usize = 2;
pub const N_VOCAB: usize = 512;

/// The MoE variant ([`write_moe`]): Mixtral's shape (the llama graph
/// with `{arch}.expert_count`, softmax routing, top-k renormalized),
/// plain SwiGLU experts and no shared expert. llama.cpp sizes a llama
/// expert by `feed_forward_length` (`llama-model.cpp`, `ffn_gate_exps
/// {n_embd, n_ff, n_expert}`), so the expert width is [`N_FF`].
pub const N_EXPERT: usize = 4;
pub const N_EXPERT_USED: usize = 2;
pub const N_FF_EXP: usize = N_FF;

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

/// One tensor: its name, output rows, input columns, dtype, values.
/// GGUF's shape is `[cols, rows]`, fastest dimension first.
struct Tensor {
    name: String,
    rows: usize,
    cols: usize,
    /// Stacked experts (a 3-D `[cols, rows, n_expert]` tensor), else 1.
    n_expert: usize,
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

fn tensors(dtype: GgmlType, embd: GgmlType, moe: bool) -> Vec<Tensor> {
    let mut seed = 0u64;
    let mut matrix = |name: String, rows: usize, cols: usize, dtype: GgmlType| {
        seed += 1;
        let scale = 1.0 / (cols as f32).sqrt();
        Tensor {
            name,
            rows,
            cols,
            n_expert: 1,
            dtype,
            values: fill(seed, rows * cols, scale),
        }
    };
    let norm = |name: String, seed: u64| Tensor {
        name,
        rows: 1,
        cols: N_EMBD,
        n_expert: 1,
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
        embd,
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
        if moe {
            // The router stays F32, as converters store it.
            out.push(matrix(
                format!("blk.{l}.ffn_gate_inp.weight"),
                N_EXPERT,
                N_EMBD,
                GgmlType::F32,
            ));
            for (name, rows, cols) in [
                ("ffn_gate_exps", N_FF_EXP, N_EMBD),
                ("ffn_up_exps", N_FF_EXP, N_EMBD),
                ("ffn_down_exps", N_EMBD, N_FF_EXP),
            ] {
                let mut t = matrix(
                    format!("blk.{l}.{name}.weight"),
                    rows * N_EXPERT,
                    cols,
                    dtype,
                );
                t.rows = rows;
                t.n_expert = N_EXPERT;
                out.push(t);
            }
            continue;
        }
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

/// Writes the checkpoint with every weight matrix in `dtype`, encoded
/// by `encode`, and returns its path. `tag` names the file, so two
/// suites never share one.
pub fn write(tag: &str, dtype: GgmlType, encode: Encode) -> PathBuf {
    write_variant(tag, dtype, GgmlType::F32, false, encode)
}

/// [`write`] with the token embedding stored as `embd` rather than F32,
/// for the suites that exercise a backend's embedding gather.
pub fn write_with_embd(tag: &str, dtype: GgmlType, embd: GgmlType, encode: Encode) -> PathBuf {
    write_variant(tag, dtype, embd, false, encode)
}

/// [`write`] with every FFN a routed MoE of [`N_EXPERT`] experts of
/// [`N_FF_EXP`], top [`N_EXPERT_USED`], the experts stored as `dtype`.
pub fn write_moe(tag: &str, dtype: GgmlType, encode: Encode) -> PathBuf {
    write_variant(tag, dtype, GgmlType::F32, true, encode)
}

fn write_variant(tag: &str, dtype: GgmlType, embd: GgmlType, moe: bool, encode: Encode) -> PathBuf {
    let mut meta = metadata();
    if moe {
        meta.insert("llama.expert_count".into(), GgufValue::U32(N_EXPERT as u32));
        meta.insert(
            "llama.expert_used_count".into(),
            GgufValue::U32(N_EXPERT_USED as u32),
        );
    }
    let encoded: Vec<(TensorPlan, Vec<u8>)> = tensors(dtype, embd, moe)
        .into_iter()
        .map(|t| {
            let shape = if t.n_expert > 1 {
                vec![t.cols as u64, t.rows as u64, t.n_expert as u64]
            } else if t.rows == 1 {
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
        "{tag}_{dtype:?}_{embd:?}-{}.gguf",
        std::process::id()
    ));
    let file = std::fs::File::create(&path).expect("create the fixture");
    let mut w = GgufWriter::create(file, &meta, plan).expect("write the header");
    for (p, bytes) in &encoded {
        w.write_tensor(&p.name, bytes).expect("write a tensor");
    }
    w.finish().expect("finish the fixture");
    path
}
