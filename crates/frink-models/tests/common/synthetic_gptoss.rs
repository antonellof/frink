//! A synthetic gpt-oss written per run: every tensor of the
//! `openai-moe` graph, at the real checkpoint's head width (64) so the
//! Metal attention kernels it serves are the ones gpt-oss-20b takes,
//! with a window of [`WINDOW`] keys so a short prompt slides it.
//!
//! The attention projections are Q8_0 (they have both a Metal matvec
//! and a `mul_mm_sg` GEMM), the routed experts MXFP4 as the real file
//! ships them, everything else F32. The values are chosen so that each
//! gpt-oss addition binds: sinks in the score range, biases a sizeable
//! fraction of the activations, expert weights scaled so the SwiGLU
//! clamp (7.0) cuts a share of the gate and up rows.

use std::collections::BTreeMap;
use std::path::PathBuf;

use frink_gguf::{GgmlType, GgufValue, GgufWriter, TensorPlan};

use super::synthetic_llama::fill;

pub const N_EMBD: usize = 256;
pub const N_HEAD: usize = 4;
pub const N_HEAD_KV: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const N_LAYER: usize = 4;
pub const N_VOCAB: usize = 512;
pub const N_EXPERT: usize = 4;
pub const N_EXPERT_USED: usize = 2;
pub const N_FF_EXP: usize = 256;
/// Sliding window of the even layers (gpt-oss's period-2 pattern).
pub const WINDOW: usize = 8;

struct Tensor {
    name: String,
    /// GGUF shape, fastest dimension first.
    shape: Vec<u64>,
    /// Row width the encoder packs.
    cols: usize,
    dtype: GgmlType,
    values: Vec<f32>,
}

fn metadata() -> BTreeMap<String, GgufValue> {
    let u = |n: usize| GgufValue::U32(n as u32);
    let a = "gpt-oss";
    [
        (
            "general.architecture".to_string(),
            GgufValue::String(a.into()),
        ),
        (format!("{a}.block_count"), u(N_LAYER)),
        (format!("{a}.context_length"), u(4096)),
        (format!("{a}.embedding_length"), u(N_EMBD)),
        (format!("{a}.feed_forward_length"), u(N_FF_EXP)),
        (format!("{a}.attention.head_count"), u(N_HEAD)),
        (format!("{a}.attention.head_count_kv"), u(N_HEAD_KV)),
        (format!("{a}.attention.key_length"), u(HEAD_DIM)),
        (format!("{a}.attention.value_length"), u(HEAD_DIM)),
        (
            format!("{a}.attention.layer_norm_rms_epsilon"),
            GgufValue::F32(1e-5),
        ),
        (format!("{a}.rope.freq_base"), GgufValue::F32(10_000.0)),
        (format!("{a}.rope.dimension_count"), u(HEAD_DIM)),
        (format!("{a}.expert_count"), u(N_EXPERT)),
        (format!("{a}.expert_used_count"), u(N_EXPERT_USED)),
        (format!("{a}.expert_feed_forward_length"), u(N_FF_EXP)),
        (format!("{a}.attention.sliding_window"), u(WINDOW)),
    ]
    .into_iter()
    .collect()
}

fn tensors(experts: GgmlType) -> Vec<Tensor> {
    let mut seed = 0u64;
    let mut t = |name: String, shape: Vec<u64>, dtype: GgmlType, scale: f32, offset: f32| {
        seed += 1;
        let n: u64 = shape.iter().product();
        Tensor {
            name,
            cols: shape[0] as usize,
            shape,
            dtype,
            values: fill(seed, n as usize, scale)
                .into_iter()
                .map(|v| v + offset)
                .collect(),
        }
    };
    let (e, h, ff) = (N_EXPERT as u64, N_EMBD as u64, N_FF_EXP as u64);
    let (q, kv) = ((N_HEAD * HEAD_DIM) as u64, (N_HEAD_KV * HEAD_DIM) as u64);
    let w = |cols: u64| 1.0 / (cols as f32).sqrt();
    let mut out = vec![t(
        "token_embd.weight".into(),
        vec![h, N_VOCAB as u64],
        GgmlType::F32,
        1.0,
        0.0,
    )];
    for l in 0..N_LAYER {
        let b = |s: &str| format!("blk.{l}.{s}");
        out.push(t(b("attn_norm.weight"), vec![h], GgmlType::F32, 0.2, 1.0));
        out.push(t(b("attn_q.weight"), vec![h, q], GgmlType::Q8_0, w(h), 0.0));
        out.push(t(
            b("attn_k.weight"),
            vec![h, kv],
            GgmlType::Q8_0,
            w(h),
            0.0,
        ));
        out.push(t(
            b("attn_v.weight"),
            vec![h, kv],
            GgmlType::Q8_0,
            w(h),
            0.0,
        ));
        out.push(t(b("attn_q.bias"), vec![q], GgmlType::F32, 0.5, 0.0));
        out.push(t(b("attn_k.bias"), vec![kv], GgmlType::F32, 0.5, 0.0));
        out.push(t(b("attn_v.bias"), vec![kv], GgmlType::F32, 0.5, 0.0));
        out.push(t(
            b("attn_output.weight"),
            vec![q, h],
            GgmlType::Q8_0,
            w(q),
            0.0,
        ));
        out.push(t(b("attn_output.bias"), vec![h], GgmlType::F32, 0.5, 0.0));
        // Sinks around the scores' own range, so each takes a real share.
        out.push(t(
            b("attn_sinks.weight"),
            vec![N_HEAD as u64],
            GgmlType::F32,
            1.0,
            1.0,
        ));
        out.push(t(
            b("post_attention_norm.weight"),
            vec![h],
            GgmlType::F32,
            0.2,
            1.0,
        ));
        out.push(t(
            b("ffn_gate_inp.weight"),
            vec![h, e],
            GgmlType::F32,
            4.0 * w(h),
            0.0,
        ));
        out.push(t(b("ffn_gate_inp.bias"), vec![e], GgmlType::F32, 1.0, 0.0));
        // Eight times the usual scale, so |gate| and |up| often pass the
        // SwiGLU clamp of 7.
        for (name, rows, cols) in [
            ("ffn_gate_exps", ff, h),
            ("ffn_up_exps", ff, h),
            ("ffn_down_exps", h, ff),
        ] {
            let s = if name == "ffn_down_exps" {
                w(cols)
            } else {
                8.0 * w(cols)
            };
            out.push(t(
                b(&format!("{name}.weight")),
                vec![cols, rows, e],
                experts,
                s,
                0.0,
            ));
            out.push(t(
                b(&format!("{name}.bias")),
                vec![rows, e],
                GgmlType::F32,
                1.0,
                0.0,
            ));
        }
    }
    out.push(t(
        "output_norm.weight".into(),
        vec![h],
        GgmlType::F32,
        0.2,
        1.0,
    ));
    out.push(t(
        "output.weight".into(),
        vec![h, N_VOCAB as u64],
        GgmlType::Q8_0,
        w(h),
        0.0,
    ));
    out
}

/// Writes the checkpoint with its routed experts as `experts`, encoded
/// by `encode`, and returns its path.
pub fn write(tag: &str, experts: GgmlType, encode: super::synthetic_llama::Encode) -> PathBuf {
    let encoded: Vec<(TensorPlan, Vec<u8>)> = tensors(experts)
        .into_iter()
        .map(|t| {
            let bytes = if t.dtype == GgmlType::F32 {
                t.values.iter().flat_map(|v| v.to_le_bytes()).collect()
            } else {
                encode(t.dtype, t.cols, &t.values)
            };
            let plan = TensorPlan {
                name: t.name,
                shape: t.shape,
                dtype: t.dtype,
                byte_len: bytes.len(),
            };
            (plan, bytes)
        })
        .collect();
    let plan = encoded.iter().map(|(p, _)| p.clone()).collect();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    let path = dir.join(format!("{tag}_{experts:?}-{}.gguf", std::process::id()));
    let file = std::fs::File::create(&path).expect("create the fixture");
    let mut w = GgufWriter::create(file, &metadata(), plan).expect("write the header");
    for (p, bytes) in &encoded {
        w.write_tensor(&p.name, bytes).expect("write a tensor");
    }
    w.finish().expect("finish the fixture");
    path
}
