//! A llama checkpoint written per run, at whatever width a suite needs.
//!
//! Two suites need a real-shaped dense decode step without a checkpoint
//! on disk: `cpu_pool_token_parity` (the two CPU schedulers agree token
//! for token) and `decode_scheduler_choice` (one decode step does not
//! alternate between them). They used to differ only in the constants,
//! so the constants are a [`Shape`] and the writer is this one file.
//!
//! Each suite compiles this as its own module and uses part of it, as
//! with `common/mod.rs`.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use frink_gguf::{GgmlType, GgufValue, GgufWriter, TensorPlan};

/// The dimensions of one synthetic llama.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_layer: usize,
    pub n_vocab: usize,
}

impl Shape {
    /// 512 wide with a 1536 FFN, two layers. Its narrowest projection
    /// (K/V, 256 x 512) is over `frink_core::threads::should_parallelize`'s
    /// floor, so every weight matrix in the step really runs in parallel.
    pub const PARITY: Self = Self {
        n_embd: 512,
        n_ff: 1536,
        n_head: 8,
        n_head_kv: 4,
        n_layer: 2,
        n_vocab: 512,
    };

    pub fn head_dim(self) -> usize {
        self.n_embd / self.n_head
    }
}

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

fn metadata(s: Shape) -> BTreeMap<String, GgufValue> {
    let u = |n: usize| GgufValue::U32(n as u32);
    [
        (
            "general.architecture",
            GgufValue::String("llama".to_string()),
        ),
        ("llama.block_count", u(s.n_layer)),
        ("llama.embedding_length", u(s.n_embd)),
        ("llama.feed_forward_length", u(s.n_ff)),
        ("llama.attention.head_count", u(s.n_head)),
        ("llama.attention.head_count_kv", u(s.n_head_kv)),
        ("llama.attention.key_length", u(s.head_dim())),
        ("llama.vocab_size", u(s.n_vocab)),
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

fn tensors(s: Shape, dtype: GgmlType) -> Vec<Tensor> {
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
        cols: s.n_embd,
        dtype: GgmlType::F32,
        values: fill(1000 + seed, s.n_embd, 0.2)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect(),
    };

    let kv = s.n_head_kv * s.head_dim();
    let mut out = vec![matrix(
        "token_embd.weight".to_string(),
        s.n_vocab,
        s.n_embd,
        GgmlType::F32,
    )];
    for l in 0..s.n_layer {
        out.push(norm(format!("blk.{l}.attn_norm.weight"), 2 * l as u64));
        for (name, rows, cols) in [
            ("attn_q", s.n_embd, s.n_embd),
            ("attn_k", kv, s.n_embd),
            ("attn_v", kv, s.n_embd),
            ("attn_output", s.n_embd, s.n_embd),
        ] {
            out.push(matrix(format!("blk.{l}.{name}.weight"), rows, cols, dtype));
        }
        out.push(norm(format!("blk.{l}.ffn_norm.weight"), 2 * l as u64 + 1));
        for (name, rows, cols) in [
            ("ffn_gate", s.n_ff, s.n_embd),
            ("ffn_up", s.n_ff, s.n_embd),
            ("ffn_down", s.n_embd, s.n_ff),
        ] {
            out.push(matrix(format!("blk.{l}.{name}.weight"), rows, cols, dtype));
        }
    }
    out.push(norm("output_norm.weight".to_string(), 99));
    out.push(matrix(
        "output.weight".to_string(),
        s.n_vocab,
        s.n_embd,
        dtype,
    ));
    out
}

/// Writes the checkpoint with every weight matrix in `dtype` and returns
/// its path. `tag` keeps two suites (and two shapes) from sharing a file.
pub fn write(shape: Shape, dtype: GgmlType, tag: &str) -> PathBuf {
    let encoded: Vec<(TensorPlan, Vec<u8>)> = tensors(shape, dtype)
        .into_iter()
        .map(|t| {
            let dims = if t.rows == 1 {
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
                shape: dims,
                dtype: t.dtype,
                byte_len: bytes.len(),
            };
            (plan, bytes)
        })
        .collect();
    let plan = encoded.iter().map(|(p, _)| p.clone()).collect();

    // `target/`, so a leftover is swept by `cargo clean`; the pid keeps
    // two concurrent runs apart.
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    let path = dir.join(format!(
        "synthetic_llama_{tag}_{dtype:?}-{}.gguf",
        std::process::id()
    ));
    let file = std::fs::File::create(&path).expect("create the fixture");
    let mut w = GgufWriter::create(file, &metadata(shape), plan).expect("write the header");
    for (p, bytes) in &encoded {
        w.write_tensor(&p.name, bytes).expect("write a tensor");
    }
    w.finish().expect("finish the fixture");
    path
}
