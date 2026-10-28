//! `ModelConfig::from_gguf`'s refusals: missing and unknown keys, the
//! multipliers this decoder does not apply, per-layer arrays, the
//! unaudited and dedicated architectures, and the FFN activation.

use super::*;

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
fn a_varying_per_layer_array_is_refused_on_a_layer_zero_architecture_and_equal_ones_are_uniform() {
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
        other => {
            panic!("expected LoadError::DedicatedArchitectureRequired for deepseek4, got {other:?}")
        }
    }
}
