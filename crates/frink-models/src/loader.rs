//! Loads a real `Decoder` from an on-disk GGUF file, using the
//! llama.cpp-style tensor naming convention
//! (`token_embd.weight`, `blk.N.attn_q.weight`, `blk.N.ffn_gate.weight`
//! or, for MoE, `blk.N.ffn_gate_exps.weight`, `output_norm.weight`,
//! `output.weight`). Until this module existed, frink could only run
//! correctly-shaped *random* weights.
//!
//! Quantized tensors (Q8_0 / Q4_0) are loaded as `WeightMatrix::Quantized`
//! backed by `WeightBytes::Mapped` -- a zero-copy view into the same
//! mmap `GgufFile` already holds, with no intermediate heap copy of the
//! tensor's bytes at all. So a checkpoint's resident memory is the
//! mmap page cache, not the mmap plus a second in-process copy of every
//! weight. `WeightMatrix::apply` dispatches to frink-quant's fused
//! dequant+dot kernels directly against those mapped bytes at inference
//! time. F32 tensors (norms, embeddings, and any weight not natively
//! quantized) still copy into an owned `Tensor`, since they're small
//! relative to the quantized weight matrices and need per-element
//! access patterns a raw byte view doesn't support as cleanly.
//!
//! Verified end to end (see `crates/frink-models/tests/gguf_roundtrip.rs`)
//! against a genuinely Q8_0-quantized, generated on-disk GGUF fixture
//! for the dense (single-expert) case, and against real OLMoE / Qwen2-MoE
//! checkpoints for the multi-expert 3D-packed-tensor path.

mod arch_tables;
mod decoder_from_gguf;
mod expert_source;
mod layer_parts;
mod model_config;
#[cfg(feature = "metal")]
mod moe_planes;
mod rope_scaling;
mod sampling_metadata;
mod tensor_audit;
mod tensors;

pub use expert_source::{GgufExpertSource, StoredExpertLayout, StoredMatrixSpec};
pub(crate) use rope_scaling::yarn_scaling_from_gguf;
pub use tensor_audit::assert_every_tensor_consumed;
pub(crate) use tensors::{
    find_info, load_f32_vec, load_f32_vec_optional, load_weight_matrix, slice_quantized_rows,
    split_expert_tensor, widen_plain_float,
};

use frink_gguf::{GgmlType, GgufError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Shard(#[from] frink_gguf::ShardError),
    #[error("tensor '{0}' has unsupported dtype {1:?}")]
    UnsupportedDtype(String, GgmlType),
    #[error(
        "MoE tensor '{0}' is not 3D or its expert count {1} does not match config n_experts {2}"
    )]
    ExpertCountMismatch(String, usize, usize),
    #[error("GGUF file is missing required hparam metadata key '{0}'")]
    MissingHparam(String),
    /// `general.architecture` is not in the capability registry -- refuse
    /// to guess RoPE/gating rather than emit fluent-but-wrong logits.
    #[error(
        "unsupported GGUF architecture '{0}': not in frink's capability registry \
         (unknown required features fail closed; see frink_models::capability)"
    )]
    UnsupportedArchitecture(String),
    /// Architecture exists but must not use the generic GQA decoder.
    #[error("architecture '{0}' cannot use the generic Decoder: {1}")]
    DedicatedArchitectureRequired(String, &'static str),
    /// Metadata advertises a feature the generic decoder does not implement.
    #[error("architecture '{0}' requires unimplemented feature: {1}")]
    UnsupportedFeature(String, String),
    #[error(
        "architecture '{0}' has never been verified against llama.cpp. It would run on \
         frink's shared generic-GQA path, which ASSUMES plain GQA with {1:?} RoPE and no \
         ALiBi, no learned position embeddings and no per-layer rope skipping. That \
         assumption has already been wrong for gpt2, mpt, refact, bloom and jais, each of \
         which loaded clean and answered as a different model. {2} Set \
         FRINK_ALLOW_UNAUDITED_ARCH=1 to run it anyway and compare the output against \
         llama.cpp yourself"
    )]
    UnauditedArchitecture(String, crate::config::RopeLayout, String),
    /// The checkpoint carries per-block tensors this build never reads,
    /// i.e. weights that contribute to the real graph and would simply
    /// be missing from ours. See [`assert_every_tensor_consumed`].
    #[error(
        "checkpoint carries {0} tensor(s) this build never reads, so its graph is not the one \
         frink would run: {1}. This is a missing feature, not a corrupt file. Override with \
         FRINK_ALLOW_UNKNOWN_TENSORS=1 to load anyway and accept wrong output."
    )]
    UnconsumedTensors(usize, String),
    /// `FRINK_STRICT_KERNELS=1` and the model has weights with no
    /// kernel on the selected accelerator, i.e. it would run, correctly,
    /// on a silently slower path. Refusing is the point: a benchmark or
    /// CI run must not be able to publish a number taken off the
    /// backend it claims. See [`frink_core::kernel_registry`].
    #[error("{0}")]
    StrictKernels(String),
}

// The norm-slot lists (`PRE_FFN_NORM_IS_POST_ATTENTION_NORM` and its
// siblings) live in `crate::norm_sites`, beside the table that reads
// them; the tests below still walk them.

#[cfg(test)]
mod tests {

    /// A quantized 1-D tensor loads through the shared helper.
    ///
    /// This used to be six copies of `load_f32_vec`, and they had
    /// drifted badly: this one decoded twenty dtypes while the five
    /// architecture loaders decoded three (F32/F16/BF16). A quantizer
    /// that emits a Q8_0 norm or bias -- ordinary for aggressive
    /// quants -- loaded on the generic path and was rejected with
    /// `UnsupportedDtype` on GLM-5.2, Kimi, DeepSeek-MLA, Gemma-4 and
    /// the hybrid stack.
    ///
    /// This file's own comment predicted exactly that, about the same
    /// split one level down: "a dtype frink can decode should never be
    /// rejected here just because the *other* dispatch table below
    /// knows it -- that split is how a supported format turns into a
    /// load failure on the one checkpoint that uses it."
    #[test]
    fn a_quantized_one_dimensional_tensor_widens_through_the_shared_helper() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.25).collect();
        let quantized = frink_quant::quantize_q8_0(&values);

        struct OneTensor {
            info: TensorInfo,
            bytes: Vec<u8>,
        }
        impl TensorSource for OneTensor {
            fn metadata(&self, _key: &str) -> Option<&frink_gguf::GgufValue> {
                None
            }
            fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
                (name == self.info.name).then_some(&self.info)
            }
            fn tensor_bytes(&self, _name: &str) -> Result<&[u8], GgufError> {
                Ok(&self.bytes)
            }
            fn tensor_mapped_range(
                &self,
                name: &str,
            ) -> Result<
                (
                    std::sync::Arc<frink_gguf::MmapHandle>,
                    std::ops::Range<usize>,
                ),
                GgufError,
            > {
                // Never reached: `load_f32_vec` widens from bytes.
                Err(GgufError::TensorNotFound(name.to_string()))
            }
        }

        let source = OneTensor {
            info: TensorInfo {
                name: "blk.0.attn_norm.weight".to_string(),
                shape: vec![64],
                dtype: GgmlType::Q8_0,
                offset: 0,
            },
            bytes: quantized,
        };

        let widened = load_f32_vec(&source, "blk.0.attn_norm.weight")
            .expect("a Q8_0 norm must load, not report an unsupported dtype");
        assert_eq!(widened.len(), values.len());
        for (got, want) in widened.iter().zip(values.iter()) {
            assert!(
                (got - want).abs() < 0.05,
                "q8_0 round trip: got {got}, want {want}"
            );
        }
    }
    use super::arch_tables::{
        DEDICATED_OWNS_ITS_BEHAVIOUR, EXPERT_WEIGHTS_NORM_READERS, EXPERT_WEIGHTS_SCALE_READERS,
        GATING_LITERAL_ARCHITECTURES, GATING_LITERAL_NAMES, LEADING_DENSE_KEY_IS_INERT,
        NO_TOPK_RENORMALIZE_ARCHITECTURES, QK_NORM_AFTER_ROPE_ARCHITECTURES,
        SIGMOID_GATING_ARCHITECTURES,
    };
    use super::rope_scaling::YARN_BETA_READERS;
    use super::*;
    use crate::config::ModelConfig;
    use frink_core::weight_matrix::{QuantKind, WeightMatrix};
    use frink_gguf::{TensorInfo, TensorSource};
    use frink_moe::GatingFunction;

    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

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

    #[test]
    fn model_config_from_gguf_fails_loudly_when_required_hparams_are_missing() {
        let tmp =
            std::env::temp_dir().join(format!("frink_test_arch_only_{}.gguf", std::process::id()));
        // Use a registered architecture so the failure is MissingHparam,
        // not UnsupportedArchitecture.
        std::fs::write(&tmp, build_arch_only_gguf("llama")).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
        std::fs::remove_file(&tmp).ok();

        match ModelConfig::from_gguf(&file) {
            Err(LoadError::MissingHparam(key)) => {
                assert_eq!(key, "llama.block_count");
            }
            other => panic!(
                "expected LoadError::MissingHparam for a file with no hparam keys, got {other:?}"
            ),
        }
    }

    #[test]
    fn model_config_from_gguf_fails_closed_on_unknown_architecture() {
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_unknown_arch_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&tmp, build_arch_only_gguf("bogus-arch-with-no-hparams")).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
        std::fs::remove_file(&tmp).ok();

        match ModelConfig::from_gguf(&file) {
            Err(LoadError::UnsupportedArchitecture(arch)) => {
                assert_eq!(arch, "bogus-arch-with-no-hparams");
            }
            other => panic!(
                "expected LoadError::UnsupportedArchitecture for an unregistered arch, got {other:?}"
            ),
        }
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

    /// Granite / MiniCPM / Command-R multipliers are hparams, not
    /// tensors, so `assert_every_tensor_consumed` cannot see them: a
    /// checkpoint declaring one loads, runs at full speed, and computes
    /// a differently-scaled graph than it was trained as. An
    /// architecture whose reference graph does not apply one must refuse
    /// it by name.
    ///
    /// Driven on `llama` rather than on `granite`, and that swap is the
    /// point: `granite` APPLIES all four now
    /// (`crate::scalar_multipliers`), so leaving the case here would
    /// have turned this test into a test of nothing the day the feature
    /// landed. llama.cpp's llama graph reads none of the four keys, so a
    /// `llama` checkpoint declaring one is exactly the silent divergence
    /// the gate exists for.
    #[test]
    fn a_declared_multiplier_this_decoder_does_not_apply_is_refused_by_name() {
        for (key, val) in [
            ("llama.logit_scale", 6.0f32),
            ("llama.residual_scale", 0.22),
            ("llama.embedding_scale", 12.0),
            ("llama.attention.scale", 0.015_625),
        ] {
            let tag = key.replace('.', "_");
            match config_error_for("llama", key, val, &tag) {
                LoadError::UnsupportedFeature(arch, msg) => {
                    assert_eq!(arch, "llama");
                    assert!(msg.contains(key), "error must name the key: {msg}");
                }
                other => panic!("expected UnsupportedFeature for {key}, got {other:?}"),
            }
        }
    }

    /// The complement, and the half that would otherwise have gone
    /// missing: `granite` must NOT be refused for the keys its graph
    /// applies.
    ///
    /// The refusal list and the implementation are two views of ONE
    /// table (`scalar_multipliers::multiplier_support`), so this test
    /// and the one above cannot both pass while they disagree -- which
    /// is the whole value of deriving the list rather than restating it.
    #[test]
    fn granite_is_not_refused_for_the_multipliers_it_applies() {
        for (key, val) in [
            ("granite.logit_scale", 8.0f32),
            ("granite.residual_scale", 0.22),
            ("granite.embedding_scale", 12.0),
            ("granite.attention.scale", 0.015_625),
        ] {
            let tag = format!("granite_ok_{}", key.replace('.', "_"));
            // The file carries no `block_count`, so the load still fails
            // -- but on the *missing hparam*, having passed this gate.
            match config_error_for("granite", key, val, &tag) {
                LoadError::MissingHparam(k) => assert_eq!(k, "granite.block_count"),
                other => panic!("{key}={val} must pass the scaling gate, got {other:?}"),
            }
        }
    }

    /// The gate must not fire on a multiplier that is a no-op. A file
    /// writing `residual_scale = 1.0` describes the graph frink already
    /// computes, and refusing it would be a false alarm. llama.cpp's
    /// `f_attention_scale` uses `0.0` rather than `1.0` as its "unset"
    /// sentinel, so the two are checked against their own no-op values.
    #[test]
    fn a_multiplier_that_is_a_no_op_is_not_refused() {
        for (key, val) in [
            ("llama.logit_scale", 1.0f32),
            ("llama.residual_scale", 1.0),
            ("llama.embedding_scale", 1.0),
            ("llama.attention.scale", 0.0),
        ] {
            let tag = format!("noop_{}", key.replace('.', "_"));
            // The file carries no `block_count`, so the load still fails
            // -- but on the *missing hparam*, having passed this gate.
            match config_error_for("llama", key, val, &tag) {
                LoadError::MissingHparam(k) => assert_eq!(k, "llama.block_count"),
                other => panic!("no-op {key}={val} must pass the scaling gate, got {other:?}"),
            }
        }
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

    /// Builds a config for an arbitrary architecture tag, returning the
    /// error rather than unwrapping it.
    /// llama.cpp chooses the FFN gate activation PER ARCHITECTURE;
    /// frink chose it per family. Those are different partitions, and
    /// `grok` is where they disagree: `src/models/grok.cpp:165` passes
    /// `LLM_FFN_GELU` to `build_moe_ffn`, while `grok` is
    /// `DecoderFamily::StandardGqa` and so was handed SwiGLU -- a
    /// different FFN on every layer.
    ///
    /// It was pinned here while `grok` still refused, because the
    /// failure mode is that auditing it later makes it silently wrong,
    /// and an audit is exactly when nobody thinks to re-check the
    /// activation. `grok` is audited now (tests/grok_graphs.rs), and the
    /// fixture's GELU experts are what that suite compares.
    #[test]
    fn the_ffn_activation_follows_the_architecture_not_the_family() {
        use crate::capability::uses_geglu;
        use crate::config::FfnActivation;

        assert!(uses_geglu("grok"), "grok's MoE FFN gate is GELU upstream");
        // Same family, SiLU upstream (`src/models/dbrx.cpp:122`), so the
        // family rule alone cannot be what selects grok.
        assert!(!uses_geglu("dbrx"));
        assert!(!uses_geglu("llama"));

        // The Gemma lineage keeps its GELU through the FAMILY rule, so
        // the new per-architecture arm must not have displaced it.
        // gemma2/gemma3 only: `gemma` v1 is unaudited and refuses, so
        // it cannot be loaded to check its activation.
        for gemma in ["gemma2", "gemma3"] {
            assert!(
                !uses_geglu(gemma),
                "{gemma} is GELU via GemmaFamily; listing it here too \
                 would hide a later regression in the family rule"
            );
            assert_eq!(
                config_for_arch(gemma).expect("gemma loads").ffn_activation,
                FfnActivation::Gelu,
                "{gemma}"
            );
        }

        // And a plain SwiGLU architecture stays SwiGLU.
        assert_eq!(
            config_for_arch("llama")
                .expect("llama loads")
                .ffn_activation,
            FfnActivation::Swiglu
        );

        // The ungated ReLU-squared row, and the four that share its FFN
        // and refuse for something else (`capability::uses_relu_sqr`).
        assert_eq!(
            config_for_arch("arcee")
                .expect("arcee loads")
                .ffn_activation,
            FfnActivation::ReluSqr
        );
        for shared in ["plm", "nemotron", "jais2", "nemotron_h"] {
            assert!(crate::capability::uses_relu_sqr(shared), "{shared}");
        }
        assert!(!crate::capability::uses_relu_sqr("llama"));
    }

    /// A per-layer array whose entries differ, on an architecture whose
    /// llama.cpp graph reads layer 0, is refused naming the table; the
    /// same arrays with equal entries are the uniform model, for any
    /// architecture, because a converter may spell a scalar as a list.
    ///
    /// Reachability, not only `LayerShapes::resolve`'s own unit test:
    /// this goes through `from_gguf` on a header-only file, which is
    /// where `openelm` used to die on `MissingHparam` for a key its
    /// file carried.
    #[test]
    fn a_varying_per_layer_array_is_refused_on_a_layer_zero_architecture_and_equal_ones_are_uniform(
    ) {
        let kvs = [
            ("general.architecture", Kv::Str("llama")),
            ("llama.block_count", Kv::U32(2)),
            ("llama.embedding_length", Kv::U32(64)),
            ("llama.attention.head_count", Kv::Arr32(&[2, 2])),
            ("llama.attention.head_count_kv", Kv::Arr32(&[2, 1])),
            ("llama.attention.key_length", Kv::U32(32)),
            ("llama.rope.freq_base", Kv::F32(10_000.0)),
        ];
        let err = ModelConfig::from_gguf(&open_metadata_gguf("layer_shapes_vary", &kvs))
            .expect_err("llama takes layer 0 upstream");
        let msg = err.to_string();
        assert!(msg.contains("PER_LAYER_SHAPE_ARCHS"), "{msg}");
        assert!(msg.contains("LLAMA_LOAD_LOCALS"), "{msg}");

        let kvs = [
            ("general.architecture", Kv::Str("llama")),
            ("llama.block_count", Kv::U32(2)),
            ("llama.embedding_length", Kv::U32(64)),
            ("llama.attention.head_count", Kv::Arr32(&[2, 2])),
            ("llama.attention.head_count_kv", Kv::Arr32(&[1, 1])),
            ("llama.attention.key_length", Kv::U32(32)),
            ("llama.rope.freq_base", Kv::F32(10_000.0)),
        ];
        let cfg = ModelConfig::from_gguf(&open_metadata_gguf("layer_shapes_equal", &kvs))
            .expect("equal arrays are the uniform model");
        assert!(cfg.layer_shapes.is_uniform());
        assert_eq!((cfg.n_heads, cfg.n_kv_heads), (2, 1));

        // An array of the wrong length is refused as llama.cpp refuses
        // it (`key has wrong array length`).
        let kvs = [
            ("general.architecture", Kv::Str("llama")),
            ("llama.block_count", Kv::U32(2)),
            ("llama.embedding_length", Kv::U32(64)),
            ("llama.attention.head_count", Kv::Arr32(&[2, 2, 2])),
            ("llama.attention.key_length", Kv::U32(32)),
            ("llama.rope.freq_base", Kv::F32(10_000.0)),
        ];
        let err = ModelConfig::from_gguf(&open_metadata_gguf("layer_shapes_len", &kvs))
            .expect_err("three entries for two layers");
        assert!(err.to_string().contains("wrong array length"), "{err}");
    }

    /// The no-renormalise list is keyed on what llama.cpp's GRAPH does,
    /// not on what a GGUF says, because for these architectures the
    /// GGUF says nothing.
    ///
    /// `expert_weights_norm` is only written by converters that set it.
    /// `deepseek.cpp:145` passes `norm_w=false`, and
    /// `conversion/deepseek.py`'s `DeepseekModel` never writes the key
    /// -- only `DeepseekV2Model` does. So a real `deepseek` checkpoint
    /// carries no key at all and frink fell through to its default,
    /// renormalising the selected experts' softmax weights where
    /// llama.cpp leaves them alone.
    ///
    /// The same mistake made OLMoE emit garbage, which is why that list
    /// exists. This pins the membership so a later edit cannot quietly
    /// drop a name back into the renormalising default.
    #[test]
    fn the_architectures_llama_cpp_does_not_renormalise_are_pinned() {
        for arch in ["deepseek", "olmoe", "qwen2moe"] {
            assert!(
                NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&arch),
                "{arch} passes norm_w=false in llama.cpp and must not be renormalised"
            );
        }
        // `deepseek2` is a DIFFERENT architecture whose converter DOES
        // write the key, so it must not be on this list -- it gets its
        // answer from the file.
        assert!(!NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&"deepseek2"));
        assert!(!NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&"qwen3moe"));
    }

    /// Every architecture llama.cpp defaults to SIGMOID gating must be
    /// on the list, because for these the GGUF carries no key to say so.
    ///
    /// Each of these reads `LLM_KV_EXPERT_GATING_FUNC` as optional and
    /// then sets SIGMOID when it is absent, so a converted checkpoint
    /// has nothing in it that would correct frink's softmax default.
    /// Same shape as the `deepseek` top-k renormalisation bug, and as
    /// `phi3`'s sliding window: the file is silent and the architecture
    /// decides.
    /// The literal table and its name list are two spellings of one
    /// fact; this is what keeps them one.
    #[test]
    fn the_gating_literal_names_are_the_gating_literal_table() {
        let from_table: Vec<&str> = GATING_LITERAL_ARCHITECTURES
            .iter()
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(from_table, GATING_LITERAL_NAMES);
        // `mimo2.cpp:227` passes the SIGMOID literal, so the key is
        // never read there; a hand-written SOFTMAX key must not turn it.
        assert!(matches!(
            GATING_LITERAL_ARCHITECTURES
                .iter()
                .find(|(n, _)| *n == "mimo2")
                .map(|(_, g)| *g),
            Some(GatingFunction::Sigmoid)
        ));
    }

    #[test]
    fn the_architectures_llama_cpp_defaults_to_sigmoid_gating_are_pinned() {
        for arch in ["afmoe", "deepseek2", "glm4moe", "laguna", "step35"] {
            assert!(
                SIGMOID_GATING_ARCHITECTURES.contains(&arch),
                "{arch} sets SIGMOID when the gating key is absent"
            );
        }
        // Architectures that HARDCODE softmax must stay off it, or the
        // fix becomes the opposite bug: `ernie4-5-moe.cpp:90` and
        // `qwen3moe` both gate with softmax unconditionally.
        for softmax in ["ernie4_5-moe", "qwen3moe", "olmoe", "llama"] {
            assert!(
                !SIGMOID_GATING_ARCHITECTURES.contains(&softmax),
                "{softmax} does not default to sigmoid"
            );
        }
    }

    /// Every name in every architecture-keyed behaviour table is a name
    /// the catalog actually resolves, on the generic-GQA path.
    ///
    /// These five tables are the repo's dominant bug shape in its purest
    /// form: five lists of strings that have to agree with a sixth
    /// structure (`capability::architecture_catalog`) about what an
    /// architecture is called, with nothing checking it. A typo, a
    /// hyphen where the GGUF has an underscore, or a name that later
    /// moves to a dedicated stack all produce the same thing -- an entry
    /// that reads as coverage and can never fire. This repo has shipped
    /// exactly that once already, in `unsupported_feature_keys`, keyed
    /// on a GGUF spelling no converter writes.
    ///
    /// The generic-path check is the second half and the sharper one: a
    /// behaviour flag on an architecture that is `DedicatedOnly` or
    /// `Deferred` never reaches this loader, so it is dead text.
    ///
    /// Sabotage to confirm: add `"seedoss"` to any list below.
    #[test]
    fn every_architecture_keyed_behaviour_table_names_a_real_generic_row() {
        let tables: &[(&str, &[&str])] = &[
            ("SIGMOID_GATING_ARCHITECTURES", SIGMOID_GATING_ARCHITECTURES),
            ("GATING_LITERAL_ARCHITECTURES", GATING_LITERAL_NAMES),
            ("EXPERT_WEIGHTS_SCALE_READERS", EXPERT_WEIGHTS_SCALE_READERS),
            ("EXPERT_WEIGHTS_NORM_READERS", EXPERT_WEIGHTS_NORM_READERS),
            ("YARN_BETA_READERS", YARN_BETA_READERS),
            (
                "NO_TOPK_RENORMALIZE_ARCHITECTURES",
                NO_TOPK_RENORMALIZE_ARCHITECTURES,
            ),
            (
                "PRE_FFN_NORM_IS_POST_ATTENTION_NORM",
                crate::norm_sites::PRE_FFN_NORM_IS_POST_ATTENTION_NORM,
            ),
            (
                "PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM",
                crate::norm_sites::PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM,
            ),
            (
                "POST_NORMS_UNDER_GROK_NAMES",
                crate::norm_sites::POST_NORMS_UNDER_GROK_NAMES,
            ),
            (
                "ATTN_NORM_2_FEEDS_ATTENTION",
                crate::norm_sites::ATTN_NORM_2_FEEDS_ATTENTION,
            ),
            ("LEADING_DENSE_KEY_IS_INERT", LEADING_DENSE_KEY_IS_INERT),
            (
                "QK_NORM_AFTER_ROPE_ARCHITECTURES",
                QK_NORM_AFTER_ROPE_ARCHITECTURES,
            ),
        ];
        for (table, names) in tables {
            for arch in *names {
                let profile = crate::capability::resolve_profile(arch).unwrap_or_else(|| {
                    panic!("{table} names `{arch}`, which the catalog does not have")
                });
                if matches!(profile.path, crate::capability::ArchPath::GenericGqa { .. }) {
                    continue;
                }
                // Not a generic row, so the entry cannot fire HERE.
                // That is allowed only when something else is named as
                // applying the behaviour instead. An unexplained dead
                // entry still fails, which is the whole point.
                let owner = DEDICATED_OWNS_ITS_BEHAVIOUR
                    .iter()
                    .find(|(name, _)| name == arch)
                    .map(|(_, owner)| *owner);
                assert!(
                    owner.is_some(),
                    "{table} names `{arch}`, which resolves to {:?} and never reaches this \
                     loader, so the entry cannot fire. Either drop it, or add it to \
                     DEDICATED_OWNS_ITS_BEHAVIOUR naming what applies the behaviour instead",
                    profile.path
                );
            }
        }
    }

    /// The three tables that describe how a layer is BUILT, rather than
    /// how it is routed, only carry architectures that are audited.
    ///
    /// The distinction matters and is not pedantry. A routing default
    /// (`SIGMOID_GATING_ARCHITECTURES`, `NO_TOPK_RENORMALIZE_ARCHITECTURES`)
    /// is allowed to name an architecture that still refuses: it is
    /// written down ahead of time so a later admission inherits the
    /// right answer, and the tables say so. But the three below change
    /// which TENSOR a layer reads and in what order -- and each was
    /// added for exactly one architecture, whose fixture is the only
    /// thing proving the change is right. A fourth name appearing here
    /// without evidence would be a claim about a graph nobody read,
    /// carried by a list whose doc comment cites two.
    #[test]
    fn the_layer_shape_tables_only_name_audited_architectures() {
        for (table, names) in [
            (
                "PRE_FFN_NORM_IS_POST_ATTENTION_NORM",
                crate::norm_sites::PRE_FFN_NORM_IS_POST_ATTENTION_NORM,
            ),
            (
                "PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM",
                crate::norm_sites::PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM,
            ),
            (
                "POST_NORMS_UNDER_GROK_NAMES",
                crate::norm_sites::POST_NORMS_UNDER_GROK_NAMES,
            ),
            (
                "ATTN_NORM_2_FEEDS_ATTENTION",
                crate::norm_sites::ATTN_NORM_2_FEEDS_ATTENTION,
            ),
            ("LEADING_DENSE_KEY_IS_INERT", LEADING_DENSE_KEY_IS_INERT),
            (
                "QK_NORM_AFTER_ROPE_ARCHITECTURES",
                QK_NORM_AFTER_ROPE_ARCHITECTURES,
            ),
        ] {
            for arch in names {
                assert!(
                    crate::capability::is_audited_generic(arch),
                    "{table} names `{arch}`, which is not in AUDITED_GENERIC_GQA. Either it \
                     has a fixture proving the change is right -- audit it -- or the entry \
                     is a guess about a graph"
                );
            }
        }
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

    /// The generic path is OPT-IN, and this is what proves it.
    ///
    /// An architecture nobody has checked used to FALL ONTO generic GQA
    /// and run. Five did exactly that and computed the wrong thing for
    /// the life of the project. The refusal exists; nothing tested it,
    /// so a reordering or an unevidenced addition to
    /// `AUDITED_GENERIC_GQA` would have gone unnoticed.
    #[test]
    fn an_unaudited_generic_architecture_refuses_rather_than_guessing() {
        // `grovemoe` is on the generic path and is not in the audited
        // list. It is the sixth name to hold this slot: `starcoder` was
        // first, until an audit found it REQUIRES a fused
        // `attn_qkv.bias` and a learned `position_embd` the generic
        // decoder has no slot for, so it refuses for a stronger reason;
        // then `xverse`, until it was admitted with a libllama-golden
        // fixture (`tests/fixture_away_graphs.rs`); then `nanbeige`,
        // until the layer loop became `crate::layer_loops`; then
        // `talkie`, until `crate::skip_stream`; then `arctic`, until
        // `crate::parallel_dense_ffn`. `grovemoe` runs a SECOND expert
        // bank (`src/models/grovemoe.cpp:57-59,137-164`) whose upstream
        // graph diverges from the reference, and its blocker is
        // invisible in metadata, so nothing but this gate stops it.
        assert!(
            !crate::capability::is_audited_generic("grovemoe"),
            "this test needs an arch that is generic AND unaudited"
        );
        match config_for_arch("grovemoe") {
            Err(LoadError::UnauditedArchitecture(name, ..)) => assert_eq!(name, "grovemoe"),
            other => panic!("expected an unaudited refusal, got {other:?}"),
        }
    }

    /// An architecture with evidence still loads, or the inversion would
    /// have turned every model off.
    #[test]
    fn an_audited_architecture_still_loads() {
        assert!(crate::capability::is_audited_generic("llama"));
        assert!(config_for_arch("llama").is_ok());
    }

    /// A NAMED problem must outrank "unaudited".
    ///
    /// `grovemoe` is unaudited AND names its second expert bank; a
    /// `llama4` file declaring a window of zero names the branch
    /// libllama aborts on (`crate::chunked_swa`), and that is what its
    /// refusal should say. Reporting "unaudited" instead would be true
    /// and far less useful, and it is the ordering the loader's own
    /// comment claims. Nothing checked that claim. (`gpt2`, `bloom` and
    /// then a plain `llama4` were the example until each was served.)
    #[test]
    fn a_named_refusal_outranks_the_unaudited_one() {
        let kvs: Vec<(&str, Kv)> = vec![
            ("general.architecture", Kv::Str("llama4")),
            ("llama4.block_count", Kv::U32(1)),
            ("llama4.embedding_length", Kv::U32(64)),
            ("llama4.attention.head_count", Kv::U32(1)),
            ("llama4.attention.head_count_kv", Kv::U32(1)),
            ("llama4.attention.key_length", Kv::U32(64)),
            ("llama4.rope.freq_base", Kv::F32(10_000.0)),
            ("llama4.expert_count", Kv::U32(16)),
            ("llama4.interleave_moe_layer_step", Kv::U32(1)),
            ("llama4.attention.sliding_window", Kv::U32(0)),
        ];
        let err = ModelConfig::from_gguf(&open_metadata_gguf("llama4", &kvs))
            .expect_err("a zero window must refuse");
        assert!(
            !matches!(err, LoadError::UnauditedArchitecture(..)),
            "llama4 should report its own reason, not that nobody audited it: {err:?}"
        );
        assert!(err.to_string().contains("llama-graph.cpp:159"), "{err}");
    }

    /// A checkpoint that declares YaRN gets the per-band divisors the
    /// reference's `"yarn"` arm implies, folded into `rope_freqs` so the
    /// existing RoPE kernels apply them. Expected values are hand-derived
    /// from `_find_correction_dim` for this fixture (rotary width 64,
    /// base 10000, original context 131072): `low = 22`, `high = 35`.
    ///
    /// Before this, frink read neither `rope.scaling.type` nor
    /// `rope.scaling.factor`, so this file roped exactly like an
    /// unscaled one -- correct near position 0, progressively wrong
    /// further in.
    #[test]
    fn a_gguf_declaring_yarn_gets_its_rope_frequencies_rewritten() {
        let cfg = llama_config_with(
            "yarn",
            &[
                ("llama.rope.scaling.type", Kv::Str("yarn")),
                ("llama.rope.scaling.factor", Kv::F32(8.0)),
                (
                    "llama.rope.scaling.original_context_length",
                    Kv::U32(131_072),
                ),
            ],
        );
        let factors = cfg
            .rope_freqs
            .expect("a YaRN checkpoint must carry rewritten per-band frequencies")
            .full;
        assert_eq!(factors.len(), 32, "one divisor per rotation band");
        assert!(
            (factors[0] - 1.0).abs() < 1e-6,
            "the fastest band is left extrapolated, got {}",
            factors[0]
        );
        let ramp = (31.0 - 22.0) / (35.0 - 22.0);
        let want = 1.0 / (ramp / 8.0 + (1.0 - ramp));
        assert!(
            (factors[31] - want).abs() < 1e-4,
            "slowest band: got {}, reference {want}",
            factors[31]
        );
    }

    /// The rewrite must not fire on a file that did not ask for it. A
    /// scaling type frink does not implement (`linear`, `longrope`) is
    /// left exactly as it was rather than being roped as YaRN, which
    /// would be a new kind of wrong rather than the current known one.
    /// `rope.scaling.type = "linear"` must actually scale.
    ///
    /// Rotating position `p/s` is the same as rotating `p` with every
    /// band's frequency divided by `s`, and `rope_freqs` is exactly a
    /// per-band frequency divisor, so a uniform vector of `s` expresses
    /// linear scaling with no new code on the RoPE paths.
    ///
    /// Before this, the scaling type was compared against "yarn" and
    /// anything else returned None, so such a file loaded and roped at
    /// unscaled positions: a different model, no error.
    #[test]
    fn linear_scaling_is_applied_as_a_uniform_frequency_divisor() {
        let cfg = llama_config_with(
            "linear",
            &[
                ("llama.rope.scaling.type", Kv::Str("linear")),
                ("llama.rope.scaling.factor", Kv::F32(4.0)),
            ],
        );
        let freqs = &cfg
            .rope_freqs
            .as_ref()
            .expect("linear scaling must produce frequency factors")
            .full;
        assert_eq!(freqs.len(), cfg.head_dim / 2, "one factor per rotated pair");
        assert!(
            freqs.iter().all(|f| (*f - 4.0).abs() < 1e-6),
            "linear scaling is uniform across bands, unlike YaRN: got {freqs:?}"
        );
    }

    /// A factor that corrects nothing is not a correction.
    #[test]
    fn a_linear_factor_of_one_is_treated_as_absent() {
        assert!(llama_config_with(
            "linear_one",
            &[
                ("llama.rope.scaling.type", Kv::Str("linear")),
                ("llama.rope.scaling.factor", Kv::F32(1.0)),
            ],
        )
        .rope_freqs
        .is_none());
    }

    #[test]
    fn a_gguf_without_yarn_scaling_keeps_its_rope_frequencies_untouched() {
        assert!(llama_config_with("noscale", &[]).rope_freqs.is_none());
        // Linear scaling is NOT "no scaling". It used to land here,
        // asserted as `is_none()`, on the reasoning that leaving
        // positions alone beat roping them wrong in a new way. Both are
        // wrong output: llama.cpp divides the positions by the factor.
        // See `linear_scaling_is_applied_as_a_uniform_frequency_divisor`.
        // YaRN with a no-op factor is not a correction either.
        assert!(llama_config_with(
            "yarn_factor_one",
            &[
                ("llama.rope.scaling.type", Kv::Str("yarn")),
                ("llama.rope.scaling.factor", Kv::F32(1.0)),
                (
                    "llama.rope.scaling.original_context_length",
                    Kv::U32(131_072),
                ),
            ],
        )
        .rope_freqs
        .is_none());
    }

    /// The correction range is measured against the context the
    /// checkpoint was *trained* at, so a file that declares YaRN without
    /// `rope.scaling.original_context_length` leaves the rotation alone
    /// rather than inventing a trained length (`context_length` on such
    /// a file is the *extended* one, which would put the ramp in the
    /// wrong place at every band).
    #[test]
    fn yarn_without_an_original_context_length_is_not_guessed_at() {
        let cfg = llama_config_with(
            "yarn_noctx",
            &[
                ("llama.rope.scaling.type", Kv::Str("yarn")),
                ("llama.rope.scaling.factor", Kv::F32(8.0)),
            ],
        );
        assert!(cfg.rope_freqs.is_none());
    }

    /// `rope.scaling.yarn_beta_fast` / `_slow` move the correction ramp
    /// only on the graphs that read them (`YARN_BETA_READERS`, measured
    /// at `604cc430`); on every other architecture llama.cpp keeps its
    /// seeded 32 / 1 whatever the file says, and so must frink.
    ///
    /// Sabotage to confirm: drop the `reads_beta` early return and the
    /// `llama` arm reads 8.0 / 2.0.
    #[test]
    fn yarn_betas_are_read_only_where_the_graph_reads_them() {
        use frink_gguf::GgufValue as V;
        let file = |arch: &str| {
            crate::test_source::StubSource::with_tensors(&[])
                .with_key(
                    &format!("{arch}.rope.scaling.type"),
                    V::String("yarn".into()),
                )
                .with_key(&format!("{arch}.rope.scaling.factor"), V::F32(4.0))
                .with_key(&format!("{arch}.rope.scaling.yarn_beta_fast"), V::F32(8.0))
                .with_key(&format!("{arch}.rope.scaling.yarn_beta_slow"), V::F32(2.0))
        };
        for (arch, fast, slow) in [
            ("llama", 32.0, 1.0),
            ("qwen3", 32.0, 1.0),
            ("laguna", 8.0, 2.0),
            ("mistral3", 8.0, 2.0),
        ] {
            let y = yarn_scaling_from_gguf(&file(arch), arch, Some(4096))
                .unwrap_or_else(|| panic!("{arch}: yarn declared"));
            assert_eq!((y.beta_fast, y.beta_slow), (fast, slow), "{arch}");
        }
    }

    /// `general.sampling.*` is the checkpoint's own recommendation, and
    /// only the keys the file carries become one: a file naming just
    /// `top_k` must leave temperature and top_p to the server's
    /// defaults.
    #[test]
    fn gguf_sampling_metadata_is_read_as_the_checkpoints_recommendation() {
        use crate::sampling::RecommendedSampling;
        let full = RecommendedSampling::from_gguf(&open_metadata_gguf(
            "sampling_full",
            &[
                ("general.architecture", Kv::Str("llama")),
                ("general.sampling.temp", Kv::F32(1.0)),
                ("general.sampling.top_k", Kv::U32(20)),
                ("general.sampling.top_p", Kv::F32(0.95)),
            ],
        ));
        assert_eq!(
            full,
            RecommendedSampling {
                temperature: Some(1.0),
                top_p: Some(0.95),
                top_k: Some(20),
            }
        );

        let partial = RecommendedSampling::from_gguf(&open_metadata_gguf(
            "sampling_partial",
            &[
                ("general.architecture", Kv::Str("llama")),
                ("general.sampling.top_k", Kv::U32(40)),
            ],
        ));
        assert_eq!(partial.top_k, Some(40));
        assert_eq!(partial.temperature, None);
        assert_eq!(partial.top_p, None);
    }

    /// A converter that wrote `temp = 1` stores a GGUF integer, not a
    /// float. Dropping it would serve a checkpoint that asked for
    /// temperature 1.0 at the framework's greedy default -- the
    /// repetition-loop failure the recommendation exists to prevent.
    #[test]
    fn an_integer_valued_sampling_temp_is_still_a_recommendation() {
        let recommended = crate::sampling::RecommendedSampling::from_gguf(&open_metadata_gguf(
            "sampling_int_temp",
            &[
                ("general.architecture", Kv::Str("llama")),
                ("general.sampling.temp", Kv::U32(1)),
            ],
        ));
        assert_eq!(recommended.temperature, Some(1.0));
    }

    /// The overwhelming majority of checkpoints recommend nothing, and
    /// those must keep frink's existing defaults exactly.
    #[test]
    fn a_gguf_without_sampling_metadata_recommends_nothing() {
        let recommended = crate::sampling::RecommendedSampling::from_gguf(&open_metadata_gguf(
            "sampling_absent",
            &[("general.architecture", Kv::Str("llama"))],
        ));
        assert!(recommended.is_empty());
    }

    #[test]
    fn model_config_from_gguf_rejects_dedicated_architectures() {
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_dedicated_arch_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&tmp, build_arch_only_gguf("deepseek4")).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("minimal header must still parse");
        std::fs::remove_file(&tmp).ok();

        match ModelConfig::from_gguf(&file) {
            Err(LoadError::DedicatedArchitectureRequired(arch, _)) => {
                assert_eq!(arch, "deepseek4");
            }
            other => panic!(
                "expected LoadError::DedicatedArchitectureRequired for deepseek4, got {other:?}"
            ),
        }
    }

    /// The same Q5_K block bytes cross-validated against an independent
    /// Python reference in `frink-quant`'s own tests, reused here for
    /// the same full-path proof as the Q6_K test below.
    #[rustfmt::skip]
    const Q5_K_TEST_BLOCK: [u8; 176] = [
        0x66, 0x2a, 0x66, 0x2a, 0x01, 0x01, 0x01, 0x01, 0x4f, 0x4b, 0x10, 0x12, 0x41, 0xe2, 0xc1,
        0xb1, 0x72, 0x2f, 0x20, 0x07, 0x31, 0x0c, 0x38, 0xb3, 0x9c, 0xb8, 0xad, 0x2f, 0x9a, 0xea,
        0x17, 0xd0, 0xee, 0x93, 0x9e, 0x3e, 0x74, 0xbb, 0x28, 0x18, 0x39, 0x25, 0xb6, 0x09, 0x18,
        0x29, 0x1c, 0x1d, 0x29, 0x41, 0x40, 0x0a, 0x74, 0x7d, 0xfd, 0x21, 0xdd, 0x6d, 0x45, 0x73,
        0x0e, 0x1e, 0xc0, 0x4a, 0xfc, 0xf3, 0x8e, 0x24, 0x6b, 0x34, 0x7d, 0xbe, 0x94, 0xde, 0x59,
        0x7a, 0x35, 0x30, 0x36, 0x0a, 0xf9, 0x4a, 0x9b, 0xa2, 0x26, 0x21, 0xa2, 0xfa, 0xdf, 0x4b,
        0x29, 0x64, 0x6f, 0xbb, 0xca, 0x0f, 0x3c, 0xda, 0x20, 0xf4, 0x93, 0x86, 0xab, 0x6e, 0xb9,
        0xe5, 0xd5, 0xa0, 0x82, 0xd6, 0x41, 0xff, 0x12, 0xbc, 0x34, 0xbb, 0xab, 0xb8, 0x20, 0x2f,
        0xbb, 0x5f, 0x0c, 0x10, 0xcf, 0x49, 0xc5, 0x86, 0x5c, 0xdf, 0xff, 0x78, 0x44, 0x26, 0x3b,
        0xc2, 0x23, 0x3d, 0x2b, 0xe9, 0x00, 0x12, 0xf8, 0xea, 0xe2, 0x9e, 0x5e, 0x50, 0x20, 0x9f,
        0x9d, 0x8d, 0x7d, 0x7f, 0xcc, 0x1d, 0x0e, 0x13, 0xf8, 0xc2, 0xf1, 0x3d, 0x08, 0x2f, 0x23,
        0x13, 0xac, 0x0d, 0xa7, 0xe7, 0x20, 0xa3, 0x90, 0xb7, 0xc8, 0x28,
    ];

    fn build_single_q5_k_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-q5k-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols,
                                                   // rows] -- reversed from the semantic [rows, cols] this tensor
                                                   // represents (1 row, 256 cols / 1 Q5_K block).
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q5_K block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(13).unwrap(); // dtype tag: Q5_K
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&Q5_K_TEST_BLOCK);
        buf
    }

    /// How far a fused dot may sit from an exact dequantized dot.
    ///
    /// Two regimes, and one fixed number cannot describe both. With
    /// `FRINK_CPU_INT_DOT` off the activation stays f32 and only
    /// rounding separates the two. With it on, the activation is
    /// quantized to int8 at `d = amax / 127`, which is the flag both
    /// binaries turn on by default and the reason the Q5_K and Q6_K
    /// cases failed against a flat `1e-2`.
    ///
    /// The bound grows with the L2 norm of the row, NOT the L1. Each
    /// element carries an independent rounding of up to `d/2`, so the
    /// dot's error is a sum of independent terms whose standard
    /// deviation is `d/sqrt(12) * ||w||_2`. Bounding by the worst case
    /// `d/2 * ||w||_1` instead assumes every rounding aligns with its
    /// weight's sign, which on this fixture gives 0.347 against a dot
    /// of 2.77: 12% of the value, loose enough that injecting a 5%
    /// error still passed. Measured here, the real error is 1.8 sigma,
    /// so four sigma keeps better than 2x headroom while still failing
    /// that 5% injection.
    fn fused_dot_tolerance(weights: &[f32], x: &[f32], exact_bound: f32) -> f32 {
        if !frink_core::weight_matrix::cpu_int_dot_for(
            frink_core::weight_matrix::IntDotShape::Matvec,
        ) {
            return exact_bound;
        }
        let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let l2 = weights.iter().map(|w| w * w).sum::<f32>().sqrt();
        4.0 * (amax / 127.0) / 12f32.sqrt() * l2 + exact_bound
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q5_k_tensor_end_to_end() {
        let tmp =
            std::env::temp_dir().join(format!("frink_test_q5k_tensor_{}.gguf", std::process::id()));
        std::fs::write(&tmp, build_single_q5_k_tensor_gguf()).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real Q5_K GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q5_K tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q5K);
                assert!(
                    data.is_mapped(),
                    "Q5_K tensors should take the zero-copy mmap path, same as Q8_0/Q4_0"
                );
            }
            _ => panic!("expected a Quantized matrix for a Q5_K tensor"),
        }

        let expected = frink_quant::dequant_q5_k(&Q5_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < fused_dot_tolerance(&expected, &x, 1e-2),
            "end-to-end loaded+applied Q5_K matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    /// The same Q6_K block bytes cross-validated against an independent
    /// Python reference in `frink-quant`'s own tests; reused here to
    /// prove the *full*
    /// path -- real on-disk GGUF bytes, parsed by `frink-gguf`, read
    /// through `GgufFile::tensor_mapped_range`, dispatched by
    /// `WeightMatrix::apply` to `frink_quant::dot_q6_k_f32` -- produces
    /// the same result as directly dequantizing those bytes, not just
    /// that the isolated kernel is correct in unit-test isolation.
    #[rustfmt::skip]
    const Q6_K_TEST_BLOCK: [u8; 210] = [
        0xe0, 0xa5, 0x40, 0x5c, 0x8d, 0x3a, 0x0a, 0x26, 0xfb, 0x4b, 0x6e, 0x9a, 0xdf, 0x3e, 0xa3,
        0xc4, 0xf8, 0x2b, 0x1d, 0x95, 0x76, 0x7d, 0x3b, 0xcd, 0xfd, 0xef, 0xc2, 0x0b, 0x07, 0x63,
        0x29, 0xfb, 0x81, 0x57, 0xbe, 0xbe, 0x06, 0xf7, 0x3a, 0x92, 0xc4, 0x43, 0xff, 0xad, 0xac,
        0x7e, 0x0f, 0x00, 0x2a, 0x4f, 0xf0, 0xf8, 0xa9, 0xfa, 0x3c, 0x90, 0x6d, 0x73, 0x2d, 0x5a,
        0xe6, 0xc6, 0x46, 0xf2, 0x0d, 0x55, 0x4c, 0x25, 0x38, 0x71, 0x2b, 0x35, 0x38, 0x82, 0x16,
        0x37, 0x5f, 0x32, 0x61, 0x02, 0xdd, 0x2f, 0x6f, 0x7b, 0x1f, 0xb4, 0x1a, 0x1b, 0x3e, 0x4f,
        0x11, 0xa3, 0x17, 0x40, 0x5a, 0x5f, 0x76, 0xcd, 0x19, 0x27, 0x9b, 0xc7, 0xc8, 0xf7, 0xf7,
        0xee, 0xf4, 0x86, 0xd9, 0xfd, 0xa7, 0xfe, 0x9e, 0xac, 0x70, 0x53, 0x5b, 0x76, 0xfb, 0x39,
        0xf8, 0x4b, 0x98, 0xfe, 0xd0, 0x06, 0x21, 0x4c, 0x4d, 0xbe, 0x10, 0x2b, 0x06, 0x65, 0xc9,
        0x5e, 0xf9, 0x95, 0x72, 0xae, 0x99, 0xd9, 0x7e, 0x15, 0xbd, 0x5e, 0x6d, 0xe8, 0x25, 0x8a,
        0xd5, 0x99, 0xc6, 0x6b, 0x69, 0xc7, 0x84, 0xc6, 0xa4, 0xf7, 0xb9, 0x6d, 0x68, 0x45, 0x0e,
        0x65, 0x69, 0xeb, 0xe6, 0xeb, 0xe9, 0x28, 0xa6, 0xb9, 0x96, 0xf2, 0xe8, 0xa7, 0x9b, 0x6e,
        0x79, 0x8a, 0x68, 0x65, 0x59, 0x98, 0x8b, 0x44, 0x41, 0x98, 0x9a, 0x56, 0x01, 0x01, 0x01,
        0x02, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x02, 0x02, 0x01, 0x01, 0x01, 0x02, 0x1f, 0x25,
    ];

    fn build_single_q6_k_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-q6k-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q6_K block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(14).unwrap(); // dtype tag: Q6_K
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&Q6_K_TEST_BLOCK);
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q6_k_tensor_end_to_end() {
        let tmp =
            std::env::temp_dir().join(format!("frink_test_q6k_tensor_{}.gguf", std::process::id()));
        std::fs::write(&tmp, build_single_q6_k_tensor_gguf()).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real Q6_K GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q6_K tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q6K);
                assert!(
                    data.is_mapped(),
                    "Q6_K tensors should take the zero-copy mmap path, same as Q8_0/Q4_0"
                );
            }
            _ => panic!("expected a Quantized matrix for a Q6_K tensor"),
        }

        let expected = frink_quant::dequant_q6_k(&Q6_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < fused_dot_tolerance(&expected, &x, 1e-2),
            "end-to-end loaded+applied Q6_K matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    fn build_single_bf16_tensor_gguf(rows: u64, cols: u64, values: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-bf16-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(cols).unwrap();
        buf.write_u64::<LittleEndian>(rows).unwrap();
        buf.write_u32::<LittleEndian>(30).unwrap(); // dtype tag: BF16
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for &v in values {
            // Real bf16 truncation (round-toward-zero, matching a real
            // writer closely enough for round-trip test purposes): top
            // 16 bits of the f32 bit pattern.
            let bf16_bits = (v.to_bits() >> 16) as u16;
            buf.extend_from_slice(&bf16_bits.to_le_bytes());
        }
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_bf16_tensor_end_to_end() {
        // Values with zero low-mantissa bits, so f32->bf16 truncation
        // is lossless and this is an exact-equality check.
        let values: Vec<f32> = vec![1.0, -2.5, 0.0, 4.0, -8.0, 16.0];
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_bf16_tensor_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&tmp, build_single_bf16_tensor_gguf(2, 3, &values)).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real BF16 GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("BF16 tensor must load");
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        match &matrix {
            WeightMatrix::F32(tensor) => {
                assert_eq!(tensor.data, values, "BF16 must widen to f32 exactly");
            }
            _ => panic!("expected an F32 matrix for a BF16 tensor (no fused dot kernel for it)"),
        }
    }

    fn build_single_f16_tensor_gguf(rows: u64, cols: u64, values: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-f16-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
        buf.write_u64::<LittleEndian>(cols).unwrap();
        buf.write_u64::<LittleEndian>(rows).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap(); // dtype tag: F16
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for &v in values {
            buf.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        buf
    }

    /// `GgmlType::F16` was parsed and sized but had no dequant arm in any
    /// of the seven loaders, so every `*-f16.gguf` was a hard
    /// `UnsupportedDtype`. Values are exactly representable in f16, so
    /// this is an exact-equality check.
    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_f16_tensor_end_to_end() {
        let values: Vec<f32> = vec![1.0, -2.5, 0.0, 4.0, -8.0, 16.0];
        let tmp =
            std::env::temp_dir().join(format!("frink_test_f16_tensor_{}.gguf", std::process::id()));
        std::fs::write(&tmp, build_single_f16_tensor_gguf(2, 3, &values)).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real F16 GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("F16 tensor must load");
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        match &matrix {
            WeightMatrix::F32(tensor) => {
                assert_eq!(tensor.data, values, "F16 must widen to f32 exactly");
            }
            _ => panic!("expected an F32 matrix for an F16 tensor (no fused dot kernel for it)"),
        }

        // The same tensor read as a plain vector (norm weights, biases and
        // the router all take this path, not `load_weight_matrix`).
        let tmp =
            std::env::temp_dir().join(format!("frink_test_f16_vec_{}.gguf", std::process::id()));
        std::fs::write(&tmp, build_single_f16_tensor_gguf(2, 3, &values)).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real F16 GGUF file must parse");
        std::fs::remove_file(&tmp).ok();
        assert_eq!(load_f32_vec(&file, "test.weight").unwrap(), values);
    }

    fn build_single_q5_1_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-q5-1-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(32).unwrap(); // cols (1 Q5_1 block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(7).unwrap(); // dtype tag: Q5_1
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        // d=0.25 (f16 0x3400), m=1.5 (f16 0x3E00) -- both exact in f16,
        // hand-verified bit patterns to avoid pulling in the `half`
        // crate just for two test constants. qh varied, qs a real
        // (non-degenerate) pattern.
        buf.extend_from_slice(&0x3400u16.to_le_bytes());
        buf.extend_from_slice(&0x3E00u16.to_le_bytes());
        buf.extend_from_slice(&[0x9au8, 0x3c, 0xf0, 0x0f]);
        buf.extend_from_slice(&(0..16u8).map(|i| i | ((15 - i) << 4)).collect::<Vec<u8>>());
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q5_1_tensor_end_to_end() {
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_q5_1_tensor_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&tmp, build_single_q5_1_tensor_gguf()).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real Q5_1 GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q5_1 tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 32);
        let raw = file.tensor_bytes("test.weight").unwrap();
        let expected = frink_quant::dequant_q5_1(raw).unwrap();
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q5_1);
                assert!(data.is_mapped());
            }
            _ => panic!("expected a Quantized matrix for a Q5_1 tensor"),
        }

        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.017).cos()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < 1e-2,
            "end-to-end loaded+applied Q5_1 matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    // Same bytes as frink-quant's own Q3_K_TEST_BLOCK (Python-cross-
    // validated there); duplicated here to build a real on-disk GGUF
    // file, matching this file's existing per-format test convention
    // (see Q6_K_TEST_BLOCK above).
    const Q3_K_TEST_BLOCK: [u8; 110] = [
        0x56, 0xf2, 0xb4, 0x2b, 0xd5, 0x6f, 0x51, 0x71, 0x3c, 0x0a, 0xb9, 0x1d, 0xd0, 0xb9, 0x3b,
        0xb3, 0x0f, 0xff, 0x8c, 0xb2, 0x83, 0x3a, 0x3d, 0x24, 0xb1, 0x12, 0x56, 0xe3, 0x23, 0x54,
        0xf2, 0xfa, 0x7f, 0xdf, 0x31, 0xe1, 0x18, 0x26, 0x6e, 0xcd, 0x5b, 0x38, 0xee, 0xbd, 0x9f,
        0x8c, 0x57, 0x47, 0x0b, 0x11, 0xcb, 0xfb, 0xb4, 0x83, 0xa0, 0x4e, 0x0b, 0xd4, 0xa7, 0x85,
        0xe0, 0x60, 0xf3, 0xb3, 0xe3, 0x95, 0x43, 0xc6, 0x05, 0x05, 0x77, 0x53, 0xed, 0x23, 0xcc,
        0x6a, 0x0e, 0x89, 0xa1, 0x79, 0x85, 0xf6, 0x6e, 0x5a, 0x23, 0x63, 0xbe, 0x53, 0xfa, 0xa2,
        0x2b, 0xe9, 0xcd, 0xce, 0xf8, 0x3d, 0x6f, 0xd0, 0x42, 0x6e, 0x3b, 0x7f, 0x23, 0x26, 0xd3,
        0xb9, 0x18, 0xbf, 0xa4, 0x34,
    ];

    fn build_single_q3_k_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-q3k-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q3_K block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(11).unwrap(); // dtype tag: Q3_K
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&Q3_K_TEST_BLOCK);
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_q3_k_tensor_end_to_end() {
        let tmp =
            std::env::temp_dir().join(format!("frink_test_q3k_tensor_{}.gguf", std::process::id()));
        std::fs::write(&tmp, build_single_q3_k_tensor_gguf()).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real Q3_K GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("Q3_K tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::Q3K);
                assert!(data.is_mapped());
            }
            _ => panic!("expected a Quantized matrix for a Q3_K tensor"),
        }

        let expected = frink_quant::dequant_q3_k(&Q3_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < fused_dot_tolerance(&expected, &x, 1e-1),
            "end-to-end loaded+applied Q3_K matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    // Same bytes as frink-quant's own IQ4_XS_TEST_BLOCK (Python-cross-
    // validated there); duplicated here to build a real on-disk GGUF
    // file, matching this file's existing per-format test convention.
    const IQ4_XS_TEST_BLOCK: [u8; 136] = [
        0x5c, 0x33, 0xb4, 0x39, 0xd1, 0x64, 0x97, 0x82, 0xcb, 0xbd, 0x88, 0x95, 0xf3, 0x60, 0x2a,
        0xb5, 0xe7, 0x24, 0xd3, 0xee, 0xfe, 0x71, 0x13, 0xbe, 0x70, 0x84, 0x48, 0x79, 0x7b, 0x3e,
        0xf0, 0x55, 0xdc, 0xb2, 0xb2, 0xde, 0x32, 0xa1, 0x5b, 0x02, 0x01, 0xdc, 0x2a, 0xbb, 0xf7,
        0x0b, 0x8a, 0x88, 0xdd, 0x0b, 0x02, 0x7e, 0x5e, 0x76, 0x87, 0x30, 0x1e, 0x1c, 0xcf, 0x48,
        0xd7, 0x61, 0xf3, 0x51, 0x52, 0x17, 0x98, 0x0a, 0x87, 0xcf, 0x02, 0x91, 0xc8, 0xee, 0xc0,
        0x91, 0x69, 0x2a, 0x4f, 0x64, 0x68, 0xa7, 0xb2, 0xe6, 0x98, 0x21, 0x81, 0x75, 0x53, 0x2a,
        0x8d, 0x12, 0xae, 0xe0, 0xea, 0x0c, 0x75, 0xff, 0x22, 0x5e, 0x25, 0x19, 0xda, 0x2e, 0x51,
        0x4e, 0x81, 0xdc, 0x0e, 0x78, 0x86, 0xd7, 0x58, 0xb5, 0xb7, 0xf6, 0x45, 0xa9, 0x0a, 0x83,
        0xfd, 0x2a, 0x12, 0x7d, 0xf0, 0x12, 0x97, 0xe2, 0xfe, 0xf4, 0xd0, 0xa2, 0x11, 0x14, 0x78,
        0xdb,
    ];

    fn build_single_iq4_xs_tensor_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

        write_kv_str(&mut buf, "general.architecture", "frink-iq4xs-test");

        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                                   // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
        buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 IQ4_XS block)
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(23).unwrap(); // dtype tag: IQ4_XS
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&IQ4_XS_TEST_BLOCK);
        buf
    }

    #[test]
    fn load_weight_matrix_handles_a_real_on_disk_iq4_xs_tensor_end_to_end() {
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_iq4xs_tensor_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&tmp, build_single_iq4_xs_tensor_gguf()).unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("real IQ4_XS GGUF file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("IQ4_XS tensor must load");
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 256);
        match &matrix {
            WeightMatrix::Quantized { kind, data, .. } => {
                assert_eq!(*kind, QuantKind::IQ4XS);
                assert!(data.is_mapped());
            }
            _ => panic!("expected a Quantized matrix for an IQ4_XS tensor"),
        }

        let expected = frink_quant::dequant_iq4_xs(&IQ4_XS_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let got = matrix.apply(&x);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected_dot).abs() < 1e-1,
            "end-to-end loaded+applied IQ4_XS matrix diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }

    // Same bytes as frink-quant's own IQ low-bit test blocks
    // (Python-cross-validated there against the real compiled ggml
    // implementation), duplicated as literals for the same reason as
    // IQ4_XS_TEST_BLOCK above.
    const IQ1_S_TEST_BLOCK: [u8; 50] = [
        0x0a, 0x2f, 0xfa, 0x06, 0x1e, 0x37, 0x6f, 0xe3, 0x62, 0xd0, 0xb6, 0xa4, 0x25, 0xae, 0x76,
        0x14, 0x72, 0x5b, 0xfa, 0x05, 0xd1, 0xf1, 0x2a, 0x4c, 0xad, 0x29, 0xae, 0xf4, 0xcf, 0x0c,
        0x96, 0x51, 0x58, 0x03, 0x6d, 0xd3, 0x10, 0x92, 0x70, 0xff, 0x61, 0x58, 0xc8, 0x30, 0x25,
        0x64, 0x49, 0x85, 0xc0, 0x24,
    ];
    const IQ2_XXS_TEST_BLOCK: [u8; 66] = [
        0x29, 0x30, 0xd9, 0x33, 0x95, 0x4c, 0x08, 0x1e, 0xad, 0x79, 0x49, 0xf2, 0x8d, 0x5f, 0x93,
        0xea, 0x78, 0x18, 0x98, 0xb9, 0x94, 0x14, 0xad, 0xce, 0xca, 0x1d, 0xab, 0x81, 0x53, 0x4a,
        0x68, 0xd0, 0x59, 0x96, 0x36, 0x5d, 0xbe, 0x20, 0xc4, 0xff, 0xe4, 0x2c, 0xcd, 0x2f, 0x4f,
        0x4f, 0x67, 0x53, 0xc6, 0xd5, 0xa2, 0xfb, 0xc7, 0xf3, 0xe2, 0x6b, 0xf1, 0x99, 0x23, 0x1e,
        0x2d, 0x5e, 0x8c, 0x78, 0xc2, 0x31,
    ];
    const IQ3_XXS_TEST_BLOCK: [u8; 98] = [
        0x71, 0x31, 0x16, 0x0a, 0x79, 0x04, 0x5d, 0x87, 0xae, 0x2a, 0x4a, 0x43, 0xfd, 0x02, 0xba,
        0x6c, 0x10, 0x42, 0x80, 0xe5, 0x1d, 0x08, 0x22, 0xcb, 0x21, 0x54, 0xf9, 0xaa, 0x8e, 0xc2,
        0xf2, 0x34, 0x66, 0x1e, 0x2a, 0xef, 0x19, 0xae, 0x48, 0x47, 0x29, 0xa0, 0x72, 0xd1, 0x31,
        0xc0, 0x65, 0x49, 0xde, 0x79, 0x32, 0xe6, 0x4d, 0xb6, 0x55, 0x3f, 0x4d, 0xf1, 0x18, 0xbb,
        0x18, 0x59, 0x4c, 0x31, 0xa3, 0xb2, 0x34, 0xdd, 0xf6, 0x4a, 0x91, 0x51, 0x3f, 0x3e, 0x40,
        0x69, 0xad, 0xbf, 0x1a, 0xd0, 0x05, 0xfb, 0xbe, 0x8b, 0x0b, 0xdd, 0xdf, 0x7d, 0x94, 0x74,
        0x92, 0x3e, 0xff, 0x04, 0x2a, 0xc4, 0xea, 0xc9,
    ];

    #[rustfmt::skip]
    const MXFP4_GGUF_TEST_BLOCKS: [u8; 68] = [0x79, 0xb4, 0x8d, 0xe2, 0x62, 0x5d, 0xbb, 0x9d, 0x54, 0xe6, 0xdb, 0x94, 0x59, 0x7d, 0x28, 0xf9, 0x79, 0x7a, 0xfc, 0xc1, 0xfa, 0x1e, 0x53, 0x5b, 0x0e, 0xc2, 0x5a, 0x2f, 0x0c, 0x82, 0x4d, 0xcb, 0x11, 0x28, 0x7b, 0x7c, 0xb6, 0x45, 0xe0, 0xb0, 0x52, 0x40, 0x51, 0xec, 0x30, 0x1a, 0xd2, 0x17, 0xf3, 0xbb, 0xfc, 0x7c, 0x8f, 0xf0, 0x67, 0x83, 0x88, 0x9d, 0x79, 0xdb, 0xf4, 0x45, 0x29, 0x78, 0xe6, 0xf4, 0x99, 0xea];

    /// A live ggml type this build has no kernel for must be REFUSED BY
    /// NAME at execution, having been sized correctly at parse.
    ///
    /// Before `TQ2_0` was recognized, tag 35 was `Other(35)`, which had
    /// no block layout: the tensor's size was unknown, so `tensor_bytes`
    /// could not even hand back the row, and the error named a number.
    /// Now the file parses, the tensor measures 66 bytes per 256
    /// elements, and the stop happens where it belongs -- at the point
    /// something wants to multiply by it -- naming `TQ2_0`.
    #[test]
    fn a_recognized_but_unimplemented_ggml_type_refuses_by_name_after_sizing_correctly() {
        // 256 elements of TQ2_0 = one 66-byte block.
        let block = pseudo_iq_block(66, 0x0720_5eed);
        let tmp =
            std::env::temp_dir().join(format!("frink_test_tq2_0_{}.gguf", std::process::id()));
        std::fs::write(
            &tmp,
            build_single_iq_lowbit_tensor_gguf("tq2test", 35, 256, &block),
        )
        .unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("a TQ2_0 file must still parse");
        std::fs::remove_file(&tmp).ok();

        // Sized, not zero: the size estimate is right even though the
        // kernel is missing.
        let info = file.find_tensor("test.weight").expect("tensor present");
        assert_eq!(info.dtype, GgmlType::TQ2_0);
        assert_eq!(info.byte_len(), Some(66));
        assert_eq!(
            file.tensor_bytes("test.weight").map(<[u8]>::len).ok(),
            Some(66)
        );

        match load_weight_matrix(&file, "test.weight") {
            Err(LoadError::UnsupportedDtype(name, GgmlType::TQ2_0)) => {
                assert_eq!(name, "test.weight");
            }
            Err(other) => panic!("TQ2_0 must be refused by name, got {other:?}"),
            Ok(_) => panic!("TQ2_0 must be refused, not loaded as some other kind"),
        }
    }

    /// An MXFP4 norm/bias must widen, not be refused.
    ///
    /// `load_weight_matrix` accepts MXFP4 as a 2-D weight and
    /// `load_moe_expert_matrices` accepts it as an expert tensor, and
    /// `WeightMatrix::dequant` calls `dequant_mxfp4_gguf` on both. One
    /// missing arm in `widen_plain_float` made the *1-D* tensors of the
    /// exact same dtype a hard `UnsupportedDtype` -- the split that
    /// turns a supported format into a load failure on the one
    /// checkpoint that uses it.
    #[test]
    fn an_mxfp4_one_dimensional_tensor_widens_instead_of_being_refused() {
        let expected = frink_quant::dequant_mxfp4_gguf(&MXFP4_GGUF_TEST_BLOCKS)
            .expect("the fixture blocks must dequantize");
        let cols = expected.len();
        let tmp =
            std::env::temp_dir().join(format!("frink_test_mxfp4_norm_{}.gguf", std::process::id()));
        std::fs::write(
            &tmp,
            build_single_iq_lowbit_tensor_gguf(
                "mxfp4norm",
                39,
                cols as u64,
                &MXFP4_GGUF_TEST_BLOCKS,
            ),
        )
        .unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("file must parse");
        std::fs::remove_file(&tmp).ok();

        let got = load_f32_vec(&file, "test.weight")
            .expect("an MXFP4 norm must load, not report an unsupported dtype");
        assert_eq!(got, expected);

        // Same arm, reached directly: `widen_plain_float` is the shared
        // helper the six architecture loaders call, so its table is the
        // one that has to know MXFP4.
        let direct = widen_plain_float(GgmlType::MXFP4, &MXFP4_GGUF_TEST_BLOCKS, "test.weight")
            .expect("widen_plain_float must widen MXFP4");
        assert_eq!(direct, expected);

        // And the refusal still works for a dtype that genuinely has no
        // widening path, so this test cannot pass by making everything
        // succeed.
        match widen_plain_float(GgmlType::TQ2_0, &MXFP4_GGUF_TEST_BLOCKS, "test.weight") {
            Err(LoadError::UnsupportedDtype(name, GgmlType::TQ2_0)) => {
                assert_eq!(name, "test.weight");
            }
            other => panic!("TQ2_0 must be refused by name, got {other:?}"),
        }
    }

    fn build_single_iq_lowbit_tensor_gguf(
        arch: &str,
        tag: u32,
        cols: u64,
        block: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count
        write_kv_str(&mut buf, "general.architecture", arch);
        write_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
        buf.write_u64::<LittleEndian>(cols).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // rows
        buf.write_u32::<LittleEndian>(tag).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(block);
        buf
    }

    /// A structurally valid block of `len` bytes for any of the
    /// codebook-grid formats: every bit pattern is a legal code in all
    /// of them (the grid indices are bounded by their own bit widths),
    /// so a deterministic byte fill is a real block, not a fixture that
    /// happens to avoid the interesting paths. Only the f16 scale needs
    /// pinning, and only so the comparison below can't be NaN-vs-NaN.
    fn pseudo_iq_block(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            out.push((s >> 24) as u8);
        }
        out
    }

    /// End-to-end load+apply for the codebook-grid low-bit formats the
    /// published Dynamic GGUFs are built from: a real on-disk tensor of
    /// each type must load zero-copy as the right `QuantKind` and
    /// produce the same matvec result as dequantizing the block
    /// directly. That is the property this test exists for -- the
    /// *values* are pinned against real ggml in `frink-quant`; what
    /// can only break here is the tag -> kind -> block-stride chain,
    /// and a wrong stride silently reads the neighbouring row.
    /// Dtype tags (19/29/16/17/22/18/21/39) verified against ggml.h's
    /// enum ggml_type.
    #[test]
    fn load_weight_matrix_handles_real_on_disk_iq_lowbit_tensors_end_to_end() {
        type DequantFn = fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError>;
        // IQ1_M carries no f16 scale field; its scale is reassembled
        // from the four scale words' top nibbles, and the top nibble of
        // the last one supplies the f16 sign + high exponent bits.
        // Pinning it to 0x2 keeps the exponent out of the all-ones
        // NaN/Inf pattern whatever the rest of the fill does. The other
        // three do carry a leading f16 `d`, pinned for the same reason.
        let mut iq1m = pseudo_iq_block(frink_quant::IQ1_M_BLOCK_BYTES, 0x2907_31A0);
        iq1m[55] = (iq1m[55] & 0x0F) | 0x20;
        let mut iq2xs = pseudo_iq_block(frink_quant::IQ2_XS_BLOCK_BYTES, 0x2107_31A1);
        let mut iq2s = pseudo_iq_block(frink_quant::IQ2_S_BLOCK_BYTES, 0x2207_31A2);
        let mut iq3s = pseudo_iq_block(frink_quant::IQ3_S_BLOCK_BYTES, 0x2307_31A3);
        for blk in [&mut iq2xs, &mut iq2s, &mut iq3s] {
            blk[0..2].copy_from_slice(&half::f16::from_f32(0.115).to_le_bytes());
        }
        let cases: [(&str, u32, &[u8], QuantKind, DequantFn); 8] = [
            (
                "iq1s",
                19,
                &IQ1_S_TEST_BLOCK,
                QuantKind::IQ1S,
                frink_quant::dequant_iq1_s,
            ),
            (
                "iq1m",
                29,
                &iq1m,
                QuantKind::IQ1M,
                frink_quant::dequant_iq1_m,
            ),
            (
                "iq2xxs",
                16,
                &IQ2_XXS_TEST_BLOCK,
                QuantKind::IQ2XXS,
                frink_quant::dequant_iq2_xxs,
            ),
            (
                "iq2xs",
                17,
                &iq2xs,
                QuantKind::IQ2XS,
                frink_quant::dequant_iq2_xs,
            ),
            (
                "iq2s",
                22,
                &iq2s,
                QuantKind::IQ2S,
                frink_quant::dequant_iq2_s,
            ),
            (
                "iq3xxs",
                18,
                &IQ3_XXS_TEST_BLOCK,
                QuantKind::IQ3XXS,
                frink_quant::dequant_iq3_xxs,
            ),
            (
                "iq3s",
                21,
                &iq3s,
                QuantKind::IQ3S,
                frink_quant::dequant_iq3_s,
            ),
            (
                "mxfp4_gguf",
                39,
                &MXFP4_GGUF_TEST_BLOCKS,
                QuantKind::Mxfp4Gguf,
                frink_quant::dequant_mxfp4_gguf,
            ),
        ];
        for (name, tag, block, kind, dequant) in cases {
            let expected = dequant(block).unwrap();
            let cols = expected.len();
            let tmp = std::env::temp_dir().join(format!("frink_test_{name}_tensor.gguf"));
            std::fs::write(
                &tmp,
                build_single_iq_lowbit_tensor_gguf(name, tag, cols as u64, block),
            )
            .unwrap();
            let file = frink_gguf::GgufFile::open(&tmp).expect("file must parse");
            std::fs::remove_file(&tmp).ok();

            let matrix =
                load_weight_matrix(&file, "test.weight").expect("low-bit tensor must load");
            assert_eq!((matrix.rows(), matrix.cols()), (1, cols), "{name}");
            match &matrix {
                WeightMatrix::Quantized { kind: k, data, .. } => {
                    assert_eq!(*k, kind, "{name}");
                    assert!(data.is_mapped(), "{name} must load zero-copy");
                }
                _ => panic!("expected a Quantized matrix for {name}"),
            }

            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin()).collect();
            let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let got = matrix.apply(&x);
            assert!(
                (got[0] - expected_dot).abs() < 1e-1,
                "{name}: loaded+applied diverged from direct dequant: got={} expected={}",
                got[0],
                expected_dot
            );
        }
    }

    #[test]
    fn qwen2moe_disables_topk_renorm() {
        assert!(
            NO_TOPK_RENORMALIZE_ARCHITECTURES.contains(&"qwen2moe"),
            "qwen2moe must have norm_topk_prob=false (llama.cpp build_moe_ffn norm_w=false)"
        );
    }

    /// The `LLAMA_ROPE_TYPE_NONE` group used to be refused by name here;
    /// every row is served now, positioned the way its graph positions
    /// (`crate::position_embd`, `crate::alibi`), and what this pins is
    /// that not one of them reaches a rotation: the rule is
    /// `RopeLayers::Never` for each, at any depth it takes.
    #[test]
    fn an_architecture_with_no_rope_rotates_nothing() {
        for (arch, n_layers) in [
            ("gpt2", 12),
            ("mpt", 32),
            ("refact", 32),
            ("bloom", 30),
            ("jais", 40),
            ("baichuan", 40),
        ] {
            assert_eq!(
                crate::rope_layers::rope_layers(arch, n_layers, false, 0),
                crate::rope_layers::RopeLayers::Never,
                "{arch} positions without RoPE and must rotate nothing"
            );
            assert!(crate::capability::is_audited_generic(arch), "{arch}");
        }
    }

    /// A per-layer sliding-window ARRAY on an architecture whose graph
    /// reads the key as a scalar is IGNORED and the seeded period
    /// stands, exactly as llama.cpp does; a scalar still overrides the
    /// period.
    ///
    /// Three generations of this gate. `capability::
    /// unsupported_feature_keys` refused the key outright with the
    /// reason "not implemented in the generic decoder", which was
    /// false. Then the loader refused the ARRAY form for every
    /// architecture, on the reasoning that honouring it as a period was
    /// impossible and ignoring it would substitute the seed for the
    /// file's layout -- which is TRUE and is ALSO what llama.cpp does:
    /// `get_key_or_arr(kid, swa_period, false)` returns false on an
    /// array (`llama-model-loader.cpp:502-507`) and `plamo3.cpp:9-11`
    /// keeps its 8. Every real EXAONE-4 32B, EXAONE-MoE and Olmo-3
    /// export carries the array (`conversion/exaone.py:84`,
    /// `olmo.py:59-66`) and was refused over a value upstream never
    /// reads. `crate::swa_layers` carries which graphs read which form;
    /// the array-HONOURED mode has its own fixture in
    /// `tests/window_array_graphs.rs`.
    #[test]
    fn an_array_valued_sliding_window_pattern_is_ignored_where_llama_cpp_ignores_it() {
        // Disagrees with plamo3's seeded last-dense 8 on layers 0..3,
        // so honouring it would be visible.
        let pattern: [u32; 4] = [0, 0, 0, 0];
        let kvs: Vec<(&str, Kv)> = vec![
            ("general.architecture", Kv::Str("plamo3")),
            ("plamo3.block_count", Kv::U32(4)),
            ("plamo3.embedding_length", Kv::U32(64)),
            ("plamo3.attention.head_count", Kv::U32(1)),
            ("plamo3.attention.head_count_kv", Kv::U32(1)),
            ("plamo3.attention.key_length", Kv::U32(64)),
            ("plamo3.rope.freq_base", Kv::F32(10_000.0)),
            ("plamo3.attention.sliding_window", Kv::U32(3)),
            (
                "plamo3.attention.sliding_window_pattern",
                Kv::Arr32(&pattern),
            ),
        ];
        let file = open_metadata_gguf("swa_pattern_array", &kvs);
        let config = ModelConfig::from_gguf(&file).expect("the array is not a refusal");
        assert_eq!(
            config.swa_layers,
            crate::swa_layers::SwaLayers::period(8, false),
            "plamo3.cpp:9-11 seeds 8 and the scalar overload ignores an array"
        );
        assert_eq!(config.layer_sliding_window(0), Some(3));
        assert_eq!(config.layer_sliding_window(3), Some(3));

        // And the scalar spelling of the same key overrides the seed.
        let mut scalar = kvs;
        scalar.pop();
        scalar.push(("plamo3.attention.sliding_window_pattern", Kv::U32(2)));
        let file = open_metadata_gguf("swa_pattern_scalar", &scalar);
        let config = ModelConfig::from_gguf(&file).expect("a scalar period must load");
        assert_eq!(
            config.swa_layers,
            crate::swa_layers::SwaLayers::period(2, false)
        );
        assert_eq!(config.layer_sliding_window(0), Some(3));
        assert_eq!(config.layer_sliding_window(1), None);
    }

    /// Baichuan is one `general.architecture` string covering two
    /// positional schemes, and llama.cpp picks between them on the layer
    /// count alone (`src/models/baichuan.cpp:11-14`, with its own "TODO:
    /// become GGUF KV parameter"). The 13B used to be refused HERE; it is
    /// served now, and what this pins is that the two schemes are still
    /// told apart by the count, on both tables that must agree about it
    /// (`crate::alibi`, `crate::rope_layers`).
    #[test]
    fn baichuan_13b_positions_by_alibi_and_the_7b_rotates() {
        assert_eq!(
            crate::alibi::max_alibi_bias("baichuan", 40, None),
            Some(8.0)
        );
        assert_eq!(
            crate::rope_layers::rope_layers("baichuan", 40, false, 0),
            crate::rope_layers::RopeLayers::Never
        );
        assert_eq!(crate::alibi::max_alibi_bias("baichuan", 32, None), None);
        assert_eq!(
            crate::rope_layers::rope_layers("baichuan", 32, false, 0),
            crate::rope_layers::RopeLayers::All
        );
        // Both sizes pass the header stage and fail on the next missing
        // hparam, which is what proves neither is gated here any more.
        for (name, n) in [("baichuan13b", 40u32), ("baichuan7b", 32)] {
            let file = open_metadata_gguf(
                name,
                &[
                    ("general.architecture", Kv::Str("baichuan")),
                    ("baichuan.block_count", Kv::U32(n)),
                ],
            );
            match ModelConfig::from_gguf(&file) {
                Err(LoadError::MissingHparam(key)) => assert_eq!(key, "baichuan.embedding_length"),
                other => panic!("{name} must pass the header stage, got {other:?}"),
            }
        }
    }

    /// EXAONE-4 is ONE architecture string over TWO graphs, and
    /// llama.cpp picks between them off the LAYER COUNT with no GGUF key
    /// involved. It used to be refused for it; both sizes run now, and
    /// this is the test that says they run DIFFERENTLY.
    ///
    /// `exaone4.cpp:4-9` wraps the entire SWA setup in
    /// `if (hparams.n_layer() == 64)`, and :116 then gates rotation on
    /// it -- `use_rope = is_swa(il) || swa_type == NONE`. So:
    ///
    /// * 64 layers: a window, `set_swa_pattern(4)` last-dense, and the
    ///   FULL-ATTENTION layer of every period gets no rotation at all.
    /// * 30 layers: no window whatever the file declares, and every
    ///   layer rotates.
    ///
    /// Both halves are here because the gate is a layer-count EQUALITY.
    /// A one-sided version would pass while windowing the 1.2B off a key
    /// llama.cpp never reaches, which is the divergence
    /// `capability::swa_disabled_by_arch` was extended to stop -- and it
    /// would then rope three layers in four of the 1.2B not at all.
    #[test]
    fn the_two_exaone4_sizes_get_different_windows_and_different_rotation() {
        // `Kv` is not `Clone`, so the shared header is a builder rather
        // than a value; both sizes must read from one list or the test
        // compares two transcriptions.
        let base = |n_layers: u32| -> Vec<(&str, Kv)> {
            vec![
                ("general.architecture", Kv::Str("exaone4")),
                ("exaone4.block_count", Kv::U32(n_layers)),
                ("exaone4.embedding_length", Kv::U32(32)),
                ("exaone4.attention.head_count", Kv::U32(4)),
                ("exaone4.attention.head_count_kv", Kv::U32(2)),
                ("exaone4.attention.key_length", Kv::U32(8)),
                ("exaone4.attention.value_length", Kv::U32(8)),
                ("exaone4.rope.freq_base", Kv::F32(10_000.0)),
                // The SAME declared window for both sizes: that is the
                // whole point. Only the layer count may change the
                // answer.
                ("exaone4.attention.sliding_window", Kv::U32(4096)),
            ]
        };

        let file = open_metadata_gguf("exaone4_32b", &base(64));
        let cfg = ModelConfig::from_gguf(&file).expect("EXAONE-4 32B loads");
        assert_eq!(cfg.sliding_window, Some(4096));
        assert_eq!(
            cfg.swa_layers,
            crate::swa_layers::SwaLayers::period(4, false),
            "exaone4.cpp:7-9, and set_swa_pattern's default phase"
        );
        for il in 0..64 {
            assert_eq!(
                cfg.layer_rotates(il),
                il % 4 != 3,
                "layer {il} of EXAONE-4 32B: only the sliding layers rotate"
            );
        }

        let file = open_metadata_gguf("exaone4_1_2b", &base(30));
        let cfg = ModelConfig::from_gguf(&file).expect("EXAONE-4 1.2B loads");
        assert_eq!(
            cfg.sliding_window, None,
            "exaone4.cpp:4 never reaches set_swa_pattern below 64 layers, \
             so the declared window is dead metadata"
        );
        for il in 0..30 {
            assert!(cfg.layer_rotates(il), "layer {il} of EXAONE-4 1.2B");
        }
    }

    /// NextN/MTP blocks are inside `block_count` and llama.cpp skips
    /// them (`n_layer = n_layer_all - n_layer_nextn`, llama-hparams.cpp
    /// :280-282). For a graph that reads the key (`exaone-moe.cpp:23`)
    /// the trunk is what loads; the key is written as `0` by
    /// `conversion/exaone.py:146` for every EXAONE-MoE export without an
    /// MTP head, so zero must be the whole file. A nonzero value on a
    /// graph that does NOT read the key stays refused
    /// (`mtp_blocks::tests`).
    ///
    /// The second half pins the ORDER of two reads in `exaone4.cpp`:
    /// `:4` tests `n_layer() == 64` before `:18` reads the key, so it
    /// sees `block_count`. A 64-trunk file with one MTP block appended
    /// is 65 there and gets NO window in llama.cpp; frink feeds
    /// `block_count` to the same gate and gets the same answer.
    #[test]
    fn nextn_predict_layers_subtracts_the_trunk_for_a_reader_and_zero_is_the_whole_file() {
        let base = |nextn: u32| -> Vec<(&str, Kv)> {
            vec![
                ("general.architecture", Kv::Str("exaone-moe")),
                ("exaone-moe.block_count", Kv::U32(5)),
                ("exaone-moe.nextn_predict_layers", Kv::U32(nextn)),
                ("exaone-moe.embedding_length", Kv::U32(32)),
                ("exaone-moe.attention.head_count", Kv::U32(4)),
                ("exaone-moe.attention.head_count_kv", Kv::U32(2)),
                ("exaone-moe.attention.key_length", Kv::U32(8)),
                ("exaone-moe.attention.value_length", Kv::U32(8)),
                ("exaone-moe.rope.freq_base", Kv::F32(10_000.0)),
                ("exaone-moe.attention.sliding_window", Kv::U32(128)),
                ("exaone-moe.expert_count", Kv::U32(4)),
                ("exaone-moe.expert_used_count", Kv::U32(2)),
                ("exaone-moe.expert_gating_func", Kv::U32(2)),
            ]
        };
        let cfg = ModelConfig::from_gguf(&open_metadata_gguf("exaone_moe_mtp", &base(1)))
            .expect("a reader with an MTP block loads its trunk");
        assert_eq!((cfg.n_layers, cfg.n_mtp_blocks), (4, 1));
        let cfg = ModelConfig::from_gguf(&open_metadata_gguf("exaone_moe_no_mtp", &base(0)))
            .expect("zero is the whole file");
        assert_eq!((cfg.n_layers, cfg.n_mtp_blocks), (5, 0));

        // `exaone4.cpp:4` before `:18`: 64 trunk layers plus one MTP
        // block is NOT the 32B to llama.cpp.
        let exaone4 = |block_count: u32, nextn: u32| -> Vec<(&str, Kv)> {
            vec![
                ("general.architecture", Kv::Str("exaone4")),
                ("exaone4.block_count", Kv::U32(block_count)),
                ("exaone4.nextn_predict_layers", Kv::U32(nextn)),
                ("exaone4.embedding_length", Kv::U32(32)),
                ("exaone4.attention.head_count", Kv::U32(4)),
                ("exaone4.attention.head_count_kv", Kv::U32(2)),
                ("exaone4.attention.key_length", Kv::U32(8)),
                ("exaone4.attention.value_length", Kv::U32(8)),
                ("exaone4.rope.freq_base", Kv::F32(10_000.0)),
                ("exaone4.attention.sliding_window", Kv::U32(4096)),
            ]
        };
        let with_mtp =
            ModelConfig::from_gguf(&open_metadata_gguf("exaone4_65", &exaone4(65, 1))).unwrap();
        assert_eq!((with_mtp.n_layers, with_mtp.n_mtp_blocks), (64, 1));
        assert_eq!(
            with_mtp.sliding_window, None,
            "exaone4.cpp:4 sees n_layer_all = 65 and never reaches set_swa_pattern"
        );
        let without =
            ModelConfig::from_gguf(&open_metadata_gguf("exaone4_64", &exaone4(64, 0))).unwrap();
        assert_eq!(
            without.sliding_window,
            Some(4096),
            "the same trunk without the block is the 32B"
        );
    }

    /// `expert_used_count` is scalar-or-array upstream, and the array
    /// spelling used to fall through to a DEFAULT of 2 here.
    ///
    /// `llama-model.cpp:1266` reads the key with `get_key_or_arr` in
    /// the common loader -- every architecture, `n_layer_all` entries
    /// -- and `conversion/nemotron.py:574` writes a list for
    /// Nemotron-H Puzzle, whose architecture (`nemotron_h`) frink
    /// serves. Before this test, `metadata_u64` answered `None` for an
    /// array value, the `unwrap_or_else` below it pushed a best-effort
    /// note, and the model routed top-2 on every layer whatever the
    /// file declared: the silent-wrong class, not a refusal.
    ///
    /// Both arms are pinned, because a reader that honoured the
    /// uniform case and silently averaged the varying one would pass
    /// half of this.
    #[test]
    fn a_per_layer_expert_used_count_is_honoured_when_uniform_and_refused_when_not() {
        fn file(used: Kv<'_>) -> Vec<(&'static str, Kv<'_>)> {
            vec![
                ("general.architecture", Kv::Str("llama")),
                ("llama.block_count", Kv::U32(2)),
                ("llama.embedding_length", Kv::U32(32)),
                ("llama.attention.head_count", Kv::U32(4)),
                ("llama.attention.head_count_kv", Kv::U32(2)),
                ("llama.attention.key_length", Kv::U32(8)),
                ("llama.attention.value_length", Kv::U32(8)),
                ("llama.rope.freq_base", Kv::F32(10_000.0)),
                ("llama.expert_count", Kv::U32(8)),
                ("llama.expert_used_count", used),
            ]
        }
        let scalar = ModelConfig::from_gguf(&open_metadata_gguf(
            "experts_used_scalar",
            &file(Kv::U32(3)),
        ))
        .expect("the scalar spelling loads");
        assert_eq!(scalar.moe.n_experts_active, 3);

        let uniform = ModelConfig::from_gguf(&open_metadata_gguf(
            "experts_used_uniform",
            &file(Kv::Arr32(&[3, 3])),
        ))
        .expect("a uniform array is that one value");
        assert_eq!(
            uniform.moe.n_experts_active, 3,
            "an array of one repeated value is the scalar, not the default of 2"
        );

        let err = ModelConfig::from_gguf(&open_metadata_gguf(
            "experts_used_varying",
            &file(Kv::Arr32(&[3, 5])),
        ))
        .expect_err("a varying array has no single top-k and must stop");
        let msg = format!("{err}");
        assert!(
            msg.contains("expert_used_count") && msg.contains("PER-LAYER"),
            "the refusal must name the key and what is wrong with it: {msg}"
        );
    }

    /// An `olmo2` file carrying BOTH a sliding window and a RoPE
    /// scaling ropes its two kinds of layer differently, and frink
    /// carries one scaling for the whole model.
    ///
    /// `olmo2.cpp:120-134` runs the sliding layers with the scaling
    /// switched off -- `freq_scale = 1`, `ext_factor = 0`,
    /// `attn_factor = 1`, and the comment above it says so in as many
    /// words -- while :136-146 gives the full-attention layers the
    /// model's own. Rotating half the layers at a magnitude the
    /// checkpoint never trained at is the ALiBi class of divergence and
    /// runs fluently.
    ///
    /// Both negative halves are here because the gate is a CONJUNCTION
    /// and a gate that fires on either half alone would refuse every
    /// OLMo-2 checkpoint ever published.
    #[test]
    fn olmo2_is_refused_only_when_it_has_a_window_and_a_rope_scaling_together() {
        // `Kv` is not `Clone`, so the shared header is a builder
        // rather than a value -- which also keeps the three cases
        // reading from one list instead of three transcriptions.
        let base = || -> Vec<(&str, Kv)> {
            vec![
                ("general.architecture", Kv::Str("olmo2")),
                ("olmo2.block_count", Kv::U32(2)),
                ("olmo2.embedding_length", Kv::U32(24)),
                ("olmo2.attention.head_count", Kv::U32(4)),
                ("olmo2.attention.head_count_kv", Kv::U32(2)),
                ("olmo2.attention.key_length", Kv::U32(6)),
                ("olmo2.attention.value_length", Kv::U32(6)),
                ("olmo2.rope.freq_base", Kv::F32(10_000.0)),
            ]
        };

        let mut both = base();
        both.push(("olmo2.attention.sliding_window", Kv::U32(3)));
        both.push(("olmo2.rope.scaling.type", Kv::Str("yarn")));
        both.push(("olmo2.rope.scaling.factor", Kv::F32(4.0)));
        let file = open_metadata_gguf("olmo2_swa_yarn", &both);
        match ModelConfig::from_gguf(&file) {
            Err(LoadError::UnsupportedFeature(arch, msg)) => {
                assert_eq!(arch, "olmo2");
                assert!(msg.contains("sliding window"), "{msg}");
                assert!(msg.contains("yarn"), "{msg}");
            }
            other => panic!("olmo2 with a window AND yarn must refuse, got {other:?}"),
        }

        // A window with no scaling: both of llama.cpp's RoPE branches
        // reduce to the same plain rotation, and the difference is
        // masking alone, which frink implements.
        let mut window_only = base();
        window_only.push(("olmo2.attention.sliding_window", Kv::U32(3)));
        let file = open_metadata_gguf("olmo2_swa_only", &window_only);
        let config = ModelConfig::from_gguf(&file).expect("a window with no scaling must load");
        assert_eq!(config.sliding_window, Some(3));
        // olmo2.cpp:9-11: the period defaults to 4 and `set_swa_pattern`
        // leaves `dense_first` false.
        assert_eq!(
            config.swa_layers,
            crate::swa_layers::SwaLayers::period(4, false)
        );

        // Scaling with no window: one RoPE for the whole model, which is
        // what frink carries.
        let mut scaling_only = base();
        scaling_only.push(("olmo2.rope.scaling.type", Kv::Str("yarn")));
        scaling_only.push(("olmo2.rope.scaling.factor", Kv::F32(4.0)));
        let file = open_metadata_gguf("olmo2_yarn_only", &scaling_only);
        let config = ModelConfig::from_gguf(&file).expect("scaling with no window must load");
        assert_eq!(config.sliding_window, None);
    }

    /// The hyper-parameters a real Gemma-3 GGUF header carries for one
    /// size. `block_count` is the field llama.cpp's `LLM_TYPE_27B`
    /// switch reads (`gemma3.cpp:20-28`), so it is never a free
    /// parameter here.
    ///
    /// `linear_factor` adds the pair `conversion/base.py:1222-1230`
    /// writes from `rope_parameters["full_attention"]` -- and only from
    /// there: its own comment is "TODO: Handle sliding_attention
    /// similarly when models start implementing it", so the sliding
    /// layers get no scaling key at all.
    fn gemma3_config(
        tag: &str,
        n_layers: u32,
        hidden_dim: u32,
        n_heads: u32,
        head_dim: u32,
        linear_factor: Option<f32>,
    ) -> ModelConfig {
        let mut kvs: Vec<(&str, Kv)> = vec![
            ("general.architecture", Kv::Str("gemma3")),
            ("gemma3.block_count", Kv::U32(n_layers)),
            ("gemma3.embedding_length", Kv::U32(hidden_dim)),
            ("gemma3.attention.head_count", Kv::U32(n_heads)),
            ("gemma3.attention.head_count_kv", Kv::U32(n_heads)),
            ("gemma3.attention.key_length", Kv::U32(head_dim)),
            ("gemma3.attention.value_length", Kv::U32(head_dim)),
            // Global layers rotate at 1e6; the sliding ones fall back to
            // llama.cpp's `rope_freq_base_train_swa` default of 10000,
            // because `gemma3.cpp:11` reads only the BASE key.
            ("gemma3.rope.freq_base", Kv::F32(1_000_000.0)),
            ("gemma3.attention.sliding_window", Kv::U32(1024)),
            ("gemma3.attention.sliding_window_pattern", Kv::U32(6)),
        ];
        if let Some(factor) = linear_factor {
            kvs.push(("gemma3.rope.scaling.type", Kv::Str("linear")));
            kvs.push(("gemma3.rope.scaling.factor", Kv::F32(factor)));
        }
        ModelConfig::from_gguf(&open_metadata_gguf(tag, &kvs)).expect("gemma3 fixture must load")
    }

    /// The 27B attention scale reaches `ModelConfig`, and no other
    /// Gemma-3 size acquires one.
    ///
    /// `capability::attention_scale_override` is where the arithmetic is
    /// checked; this pins that the LOADER calls it with this file's own
    /// numbers. That step is the one that shipped broken: the function
    /// did not exist and `attention_scale` was the literal `None`, under
    /// a comment naming the exception. A helper nobody calls looks
    /// exactly like a fix.
    #[test]
    fn a_gemma3_27b_header_sets_the_attention_scale_and_no_smaller_size_does() {
        // Gemma-3-27B: 62 layers, n_embd 5376, 32 heads, head_dim 128.
        let big = gemma3_config("g3_27b_scale", 62, 5376, 32, 128, Some(8.0));
        let want = 1.0f32 / (5376.0f32 / 32.0).sqrt();
        let got = big
            .attention_scale
            .expect("Gemma-3-27B is llama.cpp's LLM_TYPE_27B");
        assert!(
            (got - want).abs() < 1e-7,
            "want 1/sqrt(168) = {want}, got {got}"
        );
        // The bug's magnitude: scores were sqrt(168/128) = 1.146x too
        // large without this.
        let kernel = 1.0f32 / 128.0f32.sqrt();
        assert!((kernel / got - (168.0f32 / 128.0).sqrt()).abs() < 1e-5);

        // Gemma-3-1B and -4B take llama.cpp's other branch, which is the
        // scale the attention kernels already apply. A `Some` here would
        // double-scale them.
        for (tag, n_layers, hidden, heads) in
            [("g3_1b_scale", 26, 1152, 4), ("g3_4b_scale", 34, 2560, 8)]
        {
            let cfg = gemma3_config(tag, n_layers, hidden, heads, 256, None);
            assert_eq!(
                cfg.attention_scale, None,
                "{tag} must keep the kernels' own 1/sqrt(head_dim)"
            );
        }
    }

    /// Gemma-3's declared linear scaling reaches the FULL-ATTENTION
    /// layers only, and the sliding ones rope unscaled.
    ///
    /// `gemma3.cpp:11` reads `LLM_KV_ROPE_FREQ_BASE_SWA` and nothing
    /// else, so `rope_freq_scale_train_swa` keeps its `1.0f` default
    /// (`src/llama-hparams.h:129`) while `get_rope_freq_scale`
    /// (`llama-model.cpp:2033-2035`) hands the trained scale to the full
    /// layers. The converter agrees: `conversion/base.py:1222-1230`
    /// takes the factor from `rope_parameters["full_attention"]` and
    /// writes nothing for the sliding half.
    ///
    /// frink folded the factor into ONE global `rope_freqs` vector, so
    /// Gemma-3-4B/12B/27B rotated five layers in six at `p/8`.
    #[test]
    fn gemma3_linear_scaling_reaches_the_full_layers_and_not_the_sliding_ones() {
        // Gemma-3-4B: 34 layers, head_dim 256, `rope_scaling: linear 8`.
        let cfg = gemma3_config("g3_4b_rope", 34, 2560, 8, 256, Some(8.0));
        let freqs = cfg
            .rope_freqs
            .as_ref()
            .expect("declared linear scaling must produce per-band divisors");
        assert!(
            freqs.full.iter().all(|f| (*f - 8.0).abs() < 1e-6),
            "full-attention layers divide every band by the trained factor: {:?}",
            freqs.full
        );
        let swa = freqs
            .swa
            .as_ref()
            .expect("gemma3 does not assign rope_freq_scale_train_swa, so 1.0 applies");
        assert!(
            swa.iter().all(|f| (*f - 1.0).abs() < 1e-6),
            "sliding layers rope at the raw position: {swa:?}"
        );
        assert_eq!(swa.len(), freqs.full.len(), "one divisor per rotated pair");

        // Period 6, last-dense (`capability::default_swa_layout`), so
        // layer 5 is the full-attention one and 0..=4 slide. The phase
        // matters: getting it wrong swaps which five-sixths are wrong.
        assert!(cfg.layer_sliding_window(0).is_some());
        assert!(cfg.layer_sliding_window(5).is_none());
        assert_eq!(
            cfg.layer_rope(0),
            Some(crate::config::LayerRopeParams {
                theta: 10_000.0,
                freq_factors: Some(&[1.0f32; 128][..]),
                rot_dim: None,
            })
        );
        assert_eq!(
            cfg.layer_rope(5),
            Some(crate::config::LayerRopeParams {
                theta: 1_000_000.0,
                freq_factors: Some(&[8.0f32; 128][..]),
                rot_dim: None,
            })
        );
        assert!(
            cfg.rope_freqs_vary_by_layer(),
            "the fused Metal stacks take one divisor slice for a whole run \
             and so must refuse this model"
        );

        // Gemma-3-1B declares no scaling at all -- the audited fixture,
        // and the reason this was invisible. Nothing to split, so no
        // per-layer set and no Metal refusal.
        let plain = gemma3_config("g3_1b_rope", 26, 1152, 4, 256, None);
        assert!(plain.rope_freqs.is_none());
        assert!(!plain.rope_freqs_vary_by_layer());
    }

    /// Gemma-2 is the counter-case, and it is why the SWA scale needs
    /// its own table rather than reusing `swa_rope_base_follows_model`.
    ///
    /// `gemma2.cpp:10-11` assigns BOTH `rope_freq_base_train_swa` and
    /// `rope_freq_scale_train_swa` from the model's trained values, so
    /// its sliding layers keep the declared scaling. Splitting them here
    /// would be the same bug pointed the other way.
    #[test]
    fn gemma2_sliding_layers_inherit_the_trained_rope_scale() {
        let cfg = ModelConfig::from_gguf(&open_metadata_gguf(
            "g2_rope",
            &[
                ("general.architecture", Kv::Str("gemma2")),
                ("gemma2.block_count", Kv::U32(26)),
                ("gemma2.embedding_length", Kv::U32(2304)),
                ("gemma2.attention.head_count", Kv::U32(8)),
                ("gemma2.attention.head_count_kv", Kv::U32(4)),
                ("gemma2.attention.key_length", Kv::U32(256)),
                ("gemma2.attention.value_length", Kv::U32(256)),
                ("gemma2.rope.freq_base", Kv::F32(10_000.0)),
                ("gemma2.attention.sliding_window", Kv::U32(4096)),
                ("gemma2.rope.scaling.type", Kv::Str("linear")),
                ("gemma2.rope.scaling.factor", Kv::F32(8.0)),
            ],
        ))
        .expect("gemma2 fixture must load");

        let freqs = cfg.rope_freqs.as_ref().expect("linear scaling declared");
        assert_eq!(
            freqs.swa, None,
            "gemma2.cpp:11 assigns rope_freq_scale_train_swa from the trained scale"
        );
        assert!(!cfg.rope_freqs_vary_by_layer());
        // Period 2, last-dense: layer 0 slides, layer 1 does not, and
        // both get the same divisors.
        assert!(cfg.layer_sliding_window(0).is_some());
        assert!(cfg.layer_sliding_window(1).is_none());
        assert_eq!(cfg.layer_rope(0), cfg.layer_rope(1));

        // The two tables really are different: this is the pair that
        // must not be collapsed into one.
        assert!(crate::capability::swa_rope_scale_follows_model("gemma2"));
        assert!(!crate::capability::swa_rope_scale_follows_model("gemma3"));
        for arch in ["olmo2", "laguna"] {
            assert!(
                crate::capability::swa_rope_base_follows_model(arch),
                "{arch} seeds the SWA base from the model"
            );
            assert!(
                !crate::capability::swa_rope_scale_follows_model(arch),
                "{arch} pins the SWA scale to 1.0 (olmo2.cpp:14, laguna.cpp:48)"
            );
        }
    }
}
