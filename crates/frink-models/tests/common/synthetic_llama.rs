//! The synthetic llama several suites write per run: a 2-layer llama,
//! 512 wide with a 1536 FFN, every weight matrix in one quant kind.
//!
//! 512 wide so its narrowest projection (K/V, 256 x 512) is over
//! `frink_core::threads::should_parallelize`'s floor and over a Metal
//! GEMM tile, which the committed 32-wide `tests/fixtures/*.gguf` are
//! not. The caller supplies the encoder, so a kind with no
//! `frink_quant` encoder can still be written by the suite that needs
//! it.
//!
//! The width is the default, not a fixture of the file: [`Shape`] holds
//! it, [`Shape::DEFAULT`] is the 512-wide llama above, and [`write_shaped`]
//! writes any other, for a suite whose assertion depends on where a
//! layer's matvecs sit against a size threshold (`decode_scheduler_choice`).

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

/// The dimensions of one synthetic llama. The constants above are
/// [`Shape::DEFAULT`]'s fields, so the two cannot disagree.
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
    pub const DEFAULT: Self = Self {
        n_embd: N_EMBD,
        n_ff: N_FF,
        n_head: N_HEAD,
        n_head_kv: N_HEAD_KV,
        n_layer: N_LAYER,
        n_vocab: N_VOCAB,
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

fn tensors(s: Shape, dtype: GgmlType, embd: GgmlType, moe: bool) -> Vec<Tensor> {
    let Shape {
        n_embd,
        n_ff,
        n_head_kv,
        n_layer,
        n_vocab,
        ..
    } = s;
    let head_dim = s.head_dim();
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
        cols: n_embd,
        n_expert: 1,
        dtype: GgmlType::F32,
        values: fill(1000 + seed, n_embd, 0.2)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect(),
    };

    let kv = n_head_kv * head_dim;
    let mut out = vec![matrix(
        "token_embd.weight".to_string(),
        n_vocab,
        n_embd,
        embd,
    )];
    for l in 0..n_layer {
        out.push(norm(format!("blk.{l}.attn_norm.weight"), 2 * l as u64));
        for (name, rows, cols) in [
            ("attn_q", n_embd, n_embd),
            ("attn_k", kv, n_embd),
            ("attn_v", kv, n_embd),
            ("attn_output", n_embd, n_embd),
        ] {
            out.push(matrix(format!("blk.{l}.{name}.weight"), rows, cols, dtype));
        }
        out.push(norm(format!("blk.{l}.ffn_norm.weight"), 2 * l as u64 + 1));
        if moe {
            // The router stays F32, as converters store it.
            out.push(matrix(
                format!("blk.{l}.ffn_gate_inp.weight"),
                N_EXPERT,
                n_embd,
                GgmlType::F32,
            ));
            for (name, rows, cols) in [
                ("ffn_gate_exps", N_FF_EXP, n_embd),
                ("ffn_up_exps", N_FF_EXP, n_embd),
                ("ffn_down_exps", n_embd, N_FF_EXP),
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
            ("ffn_gate", n_ff, n_embd),
            ("ffn_up", n_ff, n_embd),
            ("ffn_down", n_embd, n_ff),
        ] {
            out.push(matrix(format!("blk.{l}.{name}.weight"), rows, cols, dtype));
        }
    }
    out.push(norm("output_norm.weight".to_string(), 99));
    out.push(matrix("output.weight".to_string(), n_vocab, n_embd, dtype));
    out
}

/// Writes the checkpoint with every weight matrix in `dtype`, encoded
/// by `encode`, and returns its path. `tag` names the file, so two
/// suites never share one.
pub fn write(tag: &str, dtype: GgmlType, encode: Encode) -> PathBuf {
    write_variant(tag, Shape::DEFAULT, dtype, GgmlType::F32, false, encode)
}

/// [`write`] at `shape` instead of [`Shape::DEFAULT`].
pub fn write_shaped(tag: &str, shape: Shape, dtype: GgmlType, encode: Encode) -> PathBuf {
    write_variant(tag, shape, dtype, GgmlType::F32, false, encode)
}

/// [`write`] with the token embedding stored as `embd` rather than F32,
/// for the suites that exercise a backend's embedding gather.
pub fn write_with_embd(tag: &str, dtype: GgmlType, embd: GgmlType, encode: Encode) -> PathBuf {
    write_variant(tag, Shape::DEFAULT, dtype, embd, false, encode)
}

/// [`write`] with every FFN a routed MoE of [`N_EXPERT`] experts of
/// [`N_FF_EXP`], top [`N_EXPERT_USED`], the experts stored as `dtype`.
pub fn write_moe(tag: &str, dtype: GgmlType, encode: Encode) -> PathBuf {
    write_variant(tag, Shape::DEFAULT, dtype, GgmlType::F32, true, encode)
}

fn write_variant(
    tag: &str,
    shape: Shape,
    dtype: GgmlType,
    embd: GgmlType,
    moe: bool,
    encode: Encode,
) -> PathBuf {
    assert!(
        !moe || matches!(
            shape,
            Shape {
                n_embd: N_EMBD,
                n_ff: N_FF,
                ..
            }
        ),
        "the MoE variant sizes its experts by N_FF_EXP and is written at Shape::DEFAULT only"
    );
    let mut meta = metadata(shape);
    if moe {
        meta.insert("llama.expert_count".into(), GgufValue::U32(N_EXPERT as u32));
        meta.insert(
            "llama.expert_used_count".into(),
            GgufValue::U32(N_EXPERT_USED as u32),
        );
    }
    let encoded: Vec<(TensorPlan, Vec<u8>)> = tensors(shape, dtype, embd, moe)
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
