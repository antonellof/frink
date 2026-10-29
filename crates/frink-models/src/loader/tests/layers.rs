//! Per-layer facts the config resolves: which layers slide, the MTP
//! trunk, a per-layer expert count, and the window-with-scaling refusal.

use super::*;

/// An ARRAY-valued `sliding_window_pattern` is the layout on a graph
/// that calls `load_swa_pattern` (`llama-model.cpp:3952-3959`: the
/// array first, then the scalar, then the seed). `plamo3.cpp:9` is one
/// of them, so a file whose array disagrees with the seeded 8 runs the
/// array (#420), and the scalar spelling of the same key overrides the
/// seed as a period.
#[test]
fn an_array_valued_sliding_window_pattern_is_honoured_where_llama_cpp_honours_it() {
    // Disagrees with plamo3's seeded last-dense 8 on layers 0..3,
    // so honouring it is visible.
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
    let config = ModelConfig::from_gguf(&file).expect("the array loads");
    assert_eq!(
        config.swa_layers,
        crate::swa_layers::SwaLayers::PerLayer(vec![false; 4].into()),
        "load_swa_pattern takes the array over plamo3.cpp:9's seed"
    );
    assert_eq!(config.layer_sliding_window(0), None);

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
/// :348-350). For a graph whose tensor loader creates them
/// (`exaone-moe.cpp:47`) the trunk is what loads; the key is written as
/// `0` by `conversion/exaone.py:146` for every EXAONE-MoE export without
/// an MTP head, so zero must be the whole file. A nonzero value on a
/// graph whose loader does NOT create them stays refused
/// (`mtp_blocks::tests`).
///
/// The second half pins what `exaone4.cpp:4` sees: the common loader
/// has read the key before it (`llama-model.cpp:1411,1571`), so a
/// 64-trunk file with one MTP block appended is the 32B, and 65 blocks
/// with no MTP block is not (#441).
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

    // `exaone4.cpp:4` tests the trunk: 64 layers plus one MTP block IS
    // the 32B to llama.cpp, and 65 layers is not.
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
            ("exaone4.attention.sliding_window", Kv::U32(128)),
        ]
    };
    let with_mtp =
        ModelConfig::from_gguf(&open_metadata_gguf("exaone4_65", &exaone4(65, 1))).unwrap();
    assert_eq!((with_mtp.n_layers, with_mtp.n_mtp_blocks), (64, 1));
    assert_eq!(
        with_mtp.sliding_window,
        Some(128),
        "exaone4.cpp:4 sees n_layer() = 64, the trunk, and slides"
    );
    let deeper =
        ModelConfig::from_gguf(&open_metadata_gguf("exaone4_65_trunk", &exaone4(65, 0))).unwrap();
    assert_eq!(
        deeper.sliding_window, None,
        "a 65-layer trunk is not the 32B"
    );
}

/// EXAONE-4 32B with NO window key slides at 4096 (#440):
/// `exaone4.cpp:6` seeds `n_swa = 4096` before `:14` reads the key as
/// OPTIONAL. A declared ZERO overwrites the seed with a window that
/// masks every key, and is refused. Below 64 layers the absent key is
/// still no window.
#[test]
fn exaone4_32b_without_its_window_key_slides_at_the_seeded_4096() {
    let base = |n_layers: u32, window: Option<u32>| -> Vec<(&str, Kv)> {
        let mut v = vec![
            ("general.architecture", Kv::Str("exaone4")),
            ("exaone4.block_count", Kv::U32(n_layers)),
            ("exaone4.embedding_length", Kv::U32(32)),
            ("exaone4.attention.head_count", Kv::U32(4)),
            ("exaone4.attention.head_count_kv", Kv::U32(2)),
            ("exaone4.attention.key_length", Kv::U32(8)),
            ("exaone4.attention.value_length", Kv::U32(8)),
            ("exaone4.rope.freq_base", Kv::F32(10_000.0)),
        ];
        if let Some(w) = window {
            v.push(("exaone4.attention.sliding_window", Kv::U32(w)));
        }
        v
    };
    let cfg = ModelConfig::from_gguf(&open_metadata_gguf("exaone4_64_nowin", &base(64, None)))
        .expect("loads");
    assert_eq!(cfg.sliding_window, Some(4096));
    assert_eq!(
        cfg.swa_layers,
        crate::swa_layers::SwaLayers::period(4, false)
    );
    assert!(
        !cfg.layer_rotates(3),
        "the full-attention layers rotate nothing"
    );
    let cfg = ModelConfig::from_gguf(&open_metadata_gguf("exaone4_30_nowin", &base(30, None)))
        .expect("loads");
    assert_eq!(cfg.sliding_window, None);
    match ModelConfig::from_gguf(&open_metadata_gguf("exaone4_64_zero", &base(64, Some(0)))) {
        Err(crate::loader::LoadError::UnsupportedFeature(a, m)) => {
            assert_eq!(a, "exaone4");
            assert!(m.contains("sliding_window` is 0"), "{m}");
        }
        other => panic!("a declared zero window must be refused, got {other:?}"),
    }
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
