//! RoPE as the config resolves it: YaRN and linear scaling, the
//! unrotated architectures, and Gemma's per-layer bases and scales.

use super::*;

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
        // `grok.cpp:26-27` read both over its own `beta_fast = 8` seed,
        // so the declared `beta_slow` is what tells the reader apart.
        ("grok", 8.0, 2.0),
    ] {
        let y = yarn_scaling_from_gguf(&file(arch), arch, Some(4096))
            .unwrap_or_else(|| panic!("{arch}: yarn declared"));
        assert_eq!((y.beta_fast, y.beta_slow), (fast, slow), "{arch}");
    }
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
