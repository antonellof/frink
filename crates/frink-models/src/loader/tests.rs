//! The loader's unit tests, by what they load. This file holds the
//! hand-written GGUF builders the submodules share.

use super::arch_tables::{
    DEDICATED_OWNS_ITS_BEHAVIOUR, EXPERT_WEIGHTS_NORM_READERS, EXPERT_WEIGHTS_SCALE_READERS,
    GATING_LITERAL_ARCHITECTURES, GATING_LITERAL_NAMES, LEADING_DENSE_KEY_IS_INERT,
    NO_TOPK_RENORMALIZE_ARCHITECTURES, QK_NORM_AFTER_ROPE_ARCHITECTURES,
    SIGMOID_GATING_ARCHITECTURES,
};
use super::rope_scaling::YARN_BETA_READERS;
use super::*;
use crate::config::ModelConfig;
use byteorder::{LittleEndian, WriteBytesExt};
use frink_core::weight_matrix::{QuantKind, WeightMatrix};
use frink_gguf::{TensorInfo, TensorSource};
use frink_moe::GatingFunction;
use std::io::Write;

mod config;
mod layers;
mod rope;
mod sampling;
mod tables;
mod tensors;

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.write_all(s.as_bytes()).unwrap();
}

fn write_kv_str(buf: &mut Vec<u8>, key: &str, val: &str) {
    write_string(buf, key);
    buf.write_u32::<LittleEndian>(8).unwrap(); // type = string
    write_string(buf, val);
}

/// A minimal, tensor-free GGUF byte buffer declaring only
/// `general.architecture` (no `{arch}.block_count` or any other
/// hparam key) -- the shape a stripped-down or malformed file might
/// take, and the exact case `ModelConfig::from_gguf` must reject
/// loudly rather than silently default around.
fn build_arch_only_gguf(arch: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count
    write_kv_str(&mut buf, "general.architecture", arch);
    buf
}

fn write_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
    write_string(buf, key);
    buf.write_u32::<LittleEndian>(6).unwrap(); // type = float32
    buf.write_f32::<LittleEndian>(val).unwrap();
}

/// `arch` plus one f32 hparam, so a metadata-only feature gate can be
/// exercised without building a whole checkpoint.
fn build_arch_plus_f32_gguf(arch: &str, key: &str, val: f32) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(2).unwrap(); // kv_count
    write_kv_str(&mut buf, "general.architecture", arch);
    write_kv_f32(&mut buf, key, val);
    buf
}

fn config_error_for(arch: &str, key: &str, val: f32, tag: &str) -> LoadError {
    let tmp = std::env::temp_dir().join(format!("frink_test_scale_{tag}.gguf"));
    std::fs::write(&tmp, build_arch_plus_f32_gguf(arch, key, val)).unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
    std::fs::remove_file(&tmp).ok();
    ModelConfig::from_gguf(&file).expect_err("must not succeed")
}

/// One GGUF metadata value, in the three types these header-only
/// fixtures need.
enum Kv<'a> {
    Str(&'a str),
    U32(u32),
    F32(f32),
    /// A uint32 ARRAY. Only one gate needs it -- the sliding-window
    /// pattern, which llama.cpp reads with `ml.get_key_or_arr` --
    /// and without it that gate could only be tested through a
    /// value of some other type, which is not the case it exists
    /// for.
    Arr32(&'a [u32]),
}

/// A tensor-free GGUF carrying exactly `kvs` -- enough for
/// `ModelConfig::from_gguf` to run without a single weight on disk.
fn build_metadata_gguf(kvs: &[(&str, Kv)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(kvs.len() as u64).unwrap();
    for (k, v) in kvs {
        match v {
            Kv::Str(s) => write_kv_str(&mut buf, k, s),
            Kv::U32(n) => {
                write_string(&mut buf, k);
                buf.write_u32::<LittleEndian>(4).unwrap(); // type = uint32
                buf.write_u32::<LittleEndian>(*n).unwrap();
            }
            Kv::F32(f) => write_kv_f32(&mut buf, k, *f),
            Kv::Arr32(values) => {
                write_string(&mut buf, k);
                buf.write_u32::<LittleEndian>(9).unwrap(); // type = array
                buf.write_u32::<LittleEndian>(4).unwrap(); // element type = uint32
                buf.write_u64::<LittleEndian>(values.len() as u64).unwrap();
                for v in *values {
                    buf.write_u32::<LittleEndian>(*v).unwrap();
                }
            }
        }
    }
    buf
}

fn open_metadata_gguf(tag: &str, kvs: &[(&str, Kv)]) -> frink_gguf::GgufFile {
    let tmp = std::env::temp_dir().join(format!("frink_test_meta_{tag}.gguf"));
    std::fs::write(&tmp, build_metadata_gguf(kvs)).unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("header-only file must parse");
    std::fs::remove_file(&tmp).ok();
    file
}

/// A minimal `llama` hparam set (64-wide single head, base 10000)
/// plus whatever RoPE-scaling keys a test wants to add.
fn llama_config_with(tag: &str, extra: &[(&str, Kv)]) -> ModelConfig {
    let mut kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("llama")),
        ("llama.block_count", Kv::U32(1)),
        ("llama.embedding_length", Kv::U32(64)),
        ("llama.attention.head_count", Kv::U32(1)),
        ("llama.attention.head_count_kv", Kv::U32(1)),
        ("llama.attention.key_length", Kv::U32(64)),
        ("llama.rope.freq_base", Kv::F32(10_000.0)),
    ];
    for (k, v) in extra {
        kvs.push((
            k,
            match v {
                Kv::Str(s) => Kv::Str(s),
                Kv::U32(n) => Kv::U32(*n),
                Kv::F32(f) => Kv::F32(*f),
                Kv::Arr32(a) => Kv::Arr32(a),
            },
        ));
    }
    ModelConfig::from_gguf(&open_metadata_gguf(tag, &kvs)).expect("fixture must load")
}

fn config_for_arch(arch: &'static str) -> Result<ModelConfig, LoadError> {
    // The per-arch hyperparameter keys are looked up by the arch's
    // own prefix, so they have to be built for the arch under test.
    let keys: Vec<String> = [
        "block_count",
        "embedding_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
    ]
    .iter()
    .map(|k| format!("{arch}.{k}"))
    .collect();
    let theta = format!("{arch}.rope.freq_base");
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str(arch)),
        (keys[0].as_str(), Kv::U32(1)),
        (keys[1].as_str(), Kv::U32(64)),
        (keys[2].as_str(), Kv::U32(1)),
        (keys[3].as_str(), Kv::U32(1)),
        (keys[4].as_str(), Kv::U32(64)),
        (theta.as_str(), Kv::F32(10_000.0)),
    ];
    ModelConfig::from_gguf(&open_metadata_gguf(arch, &kvs))
}
