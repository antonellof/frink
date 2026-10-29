//! Apertus, checked against llama.cpp itself: an FFN activation whose
//! PARAMETERS VARY BY LAYER.
//!
//! `apertus` was triaged NEW CODE on `apertus.cpp:6-9`, which reads
//! `xielu.alpha_n`, `xielu.alpha_p`, `xielu.beta` and `xielu.eps` as
//! `n_layer`-long arrays (or one scalar broadcast, `get_key_or_arr`),
//! and `:132-138`, which hands layer `il`'s four to `ggml_xielu` over
//! that layer's `ffn_up` output. `FfnActivation` had no way to carry a
//! parameter, let alone one per layer, and every FFN body converted the
//! model-wide `ffn_activation` to a `GluAct` without knowing which
//! layer it was running.
//!
//! **The seam.** `frink_models::act_layers` reads the four keys the
//! way `get_key_or_arr` does and folds them the way `ggml_xielu` does
//! (`beta + softplus(alpha_n)`, `softplus(alpha_p)`);
//! `FfnActivation::Xielu` carries the table so the kind and the
//! parameters cannot disagree; `ModelConfig::layer_ffn_acts(il)` is the
//! ONE accessor, and `GluAct::from(ffn_activation)` no longer exists
//! because it could not be written for a variant that needs the layer.
//! The FFN is UNGATED and takes the same gate-to-up alias as `arcee`;
//! `GluAct::Xielu` reads the `up` operand alone.
//!
//! **Where the numbers come from.** Each `*_GOLDEN` was produced by
//! running llama.cpp's own graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! Regenerating:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_apertus_fixture.py \
//!     crates/frink-models/tests/fixtures/apertus_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_apertus_fixture.py \
//!     crates/frink-models/tests/fixtures/apertus_scalar_tiny.gguf --scalar
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_apertus_fixture.py \
//!     crates/frink-models/tests/fixtures/apertus_qknorm_bias_tiny.gguf --qk-norm-bias
//! /tmp/ref_logits crates/frink-models/tests/fixtures/apertus_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GRAPH_PROMPT,
};
use frink_models::act_layers::XieluLayers;
use frink_models::{FfnActivation, ModelConfig, RopeLayout};
use frink_moe::{GluAct, XieluParams};

const APERTUS: &str = "apertus";
/// The same weights, the four keys as ONE scalar each.
const APERTUS_SCALAR: &str = "apertus_scalar";
/// The same weights, plus `attn_q_norm.bias` / `attn_k_norm.bias`.
const APERTUS_QKNORM_BIAS: &str = "apertus_qknorm_bias";

/// llama.cpp's logits for `apertus_tiny.gguf` over [`GRAPH_PROMPT`].
const APERTUS_GOLDEN: [f32; 48] = [
    0.57540876,
    2.198359,
    -2.2023356,
    -0.6537913,
    -1.1609751,
    -0.38580835,
    -0.8722731,
    0.8882388,
    -0.6675779,
    -2.3784184,
    -0.88163614,
    -1.896804,
    0.9259517,
    -2.0676188,
    2.0615187,
    -1.5849015,
    -2.8056638,
    0.44233537,
    -1.1601524,
    2.4873514,
    0.3291728,
    -0.41446838,
    -0.5385244,
    0.8861667,
    -0.35670245,
    -0.7226599,
    -0.10094422,
    -3.3551378,
    2.3091342,
    3.217208,
    -0.49769974,
    0.9325632,
    -2.2671025,
    -2.2003722,
    -2.254445,
    -0.60457003,
    2.3115144,
    0.7015171,
    -0.48302782,
    -0.47727245,
    -2.7787547,
    -1.4279127,
    -0.8523816,
    -3.631866,
    0.24088003,
    -2.7047124,
    -1.6728499,
    -0.29418832,
];

/// llama.cpp's logits for `apertus_scalar_tiny.gguf`: layer 0's four
/// scalars broadcast to both layers.
const APERTUS_SCALAR_GOLDEN: [f32; 48] = [
    0.75349087,
    1.8001257,
    -2.4080544,
    -0.33062598,
    -0.8818627,
    -0.9946704,
    -0.7034973,
    0.36403027,
    -0.3950612,
    -2.424864,
    -0.48513192,
    -1.7490184,
    0.8557503,
    -2.030672,
    2.2449164,
    -1.241854,
    -2.708693,
    0.5537913,
    -0.92210644,
    2.6275105,
    0.2451546,
    -0.39819682,
    -0.22304824,
    1.0274572,
    -0.60309577,
    -0.9781857,
    -0.2712146,
    -3.1296043,
    1.872018,
    3.281887,
    0.0017762184,
    1.0447685,
    -2.221422,
    -2.1002908,
    -2.5670779,
    -1.1007462,
    2.2869432,
    0.962994,
    -0.15839267,
    -0.25386834,
    -2.5356014,
    -1.8349712,
    -0.52645236,
    -3.5101717,
    0.30765307,
    -2.1653757,
    -1.5971752,
    -0.46207526,
];

/// The row itself, on all three forward paths.
#[test]
fn apertus_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(APERTUS, &APERTUS_GOLDEN);
}

/// The scalar spelling of the same four keys: llama.cpp broadcasts one
/// value to every layer (`llama-model-loader.cpp:469-478`), so this
/// file runs layer 0's parameters on both layers, and the golden --
/// libllama's own -- says so.
#[test]
fn a_scalar_key_is_broadcast_to_every_layer_as_llama_cpp_broadcasts_it() {
    assert_all_three_paths_match(APERTUS_SCALAR, &APERTUS_SCALAR_GOLDEN);
    let d = load_graph_fixture(APERTUS_SCALAR);
    assert_eq!(d.config.layer_ffn_acts(0), d.config.layer_ffn_acts(1));
    // And the two files really are different graphs, so the test above
    // could not have passed by accident on the array file's numbers.
    assert!(
        worst_vs(&APERTUS_SCALAR_GOLDEN, &APERTUS_GOLDEN) > 1e-2,
        "the scalar and array fixtures must disagree, or broadcasting is invisible"
    );
}

/// The number in the report, so it can be regenerated rather than
/// trusted. Run with `--nocapture` to see it.
#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (APERTUS, &APERTUS_GOLDEN),
        (APERTUS_SCALAR, &APERTUS_SCALAR_GOLDEN),
    ] {
        let d = load_graph_fixture(name);
        let mut kv = graph_caches(&d);
        let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
        let kl = kl_vs_golden(&got, golden);
        let worst = worst_vs(&got, golden);
        println!("| `{name}` | {kl:.2e} | {worst:.2e} |");
        assert!(kl < 1e-8, "{name}: KL {kl}");
    }
}

/// The loaded shape: the activation is xIELU, the table has one entry
/// per layer, the two entries are the file's and differ, and on every
/// layer the gate IS the up matrix.
#[test]
fn each_layer_carries_its_own_folded_parameters_over_an_aliased_gate() {
    let d = load_graph_fixture(APERTUS);
    let FfnActivation::Xielu(layers) = &d.config.ffn_activation else {
        panic!(
            "apertus must load as xIELU, got {:?}",
            d.config.ffn_activation
        );
    };
    assert_eq!(layers.len(), 2);
    // scripts/make_apertus_fixture.py: XIELU = [(0.8, 0.8, 0.5, -1e-6),
    // (0.2, 1.5, 0.75, -0.3)], folded by `from_gguf`.
    assert_eq!(
        layers.layer(0),
        XieluParams::from_gguf(0.8, 0.8, 0.5, -1e-6)
    );
    assert_eq!(
        layers.layer(1),
        XieluParams::from_gguf(0.2, 1.5, 0.75, -0.3)
    );
    assert_ne!(layers.layer(0), layers.layer(1));
    assert_eq!(
        d.config.layer_ffn_acts(1).dense,
        GluAct::Xielu(layers.layer(1))
    );
    assert_eq!(
        d.config.model_ffn_act(),
        None,
        "no whole-model activation to hand a kernel"
    );
    assert!(d.config.ffn_is_ungated());
    for (il, layer) in d.layers.iter().enumerate() {
        layer.moe.with_expert(0, |ex| {
            assert_eq!(ex.gate.rows(), ex.up.rows(), "blk.{il}");
            for r in 0..ex.up.rows() {
                assert_eq!(
                    ex.gate.dequant_row(r),
                    ex.up.dequant_row(r),
                    "blk.{il} row {r}"
                );
            }
            assert_eq!(ex.up.rows(), 40, "blk.{il}: n_ff");
        });
    }
}

/// The sabotage that matters: reading the arrays and indexing them
/// wrong. Swapping the two layers' parameter sets moves the logits;
/// so does giving both layers layer 0's, which is what a model-wide
/// activation would have done.
#[test]
fn indexing_the_arrays_by_the_wrong_layer_diverges_from_llama_cpp() {
    let d = load_graph_fixture(APERTUS);
    let FfnActivation::Xielu(layers) = d.config.ffn_activation.clone() else {
        unreachable!()
    };
    let (p0, p1) = (layers.layer(0), layers.layer(1));
    for (what, table) in [("swapped", vec![p1, p0]), ("layer 0 on both", vec![p0, p0])] {
        let mut d = load_graph_fixture(APERTUS);
        d.config.ffn_activation = FfnActivation::Xielu(XieluLayers::new(table));
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &APERTUS_GOLDEN,
        );
        assert!(
            worst > 1e-2,
            "{what}: the output moved by only {worst}; the two layers' parameters must be too \
             alike for the suite to see which layer's are used"
        );
    }
}

/// Each of the four parameters is visible on its own: zeroing the
/// fold, the quadratic term, the linear term, or narrowing `eps` all
/// move the logits. A parameter the suite cannot see is a parameter it
/// cannot pin.
#[test]
fn every_one_of_the_four_parameters_is_visible_in_the_logits() {
    let d = load_graph_fixture(APERTUS);
    let FfnActivation::Xielu(layers) = d.config.ffn_activation.clone() else {
        unreachable!()
    };
    let base = [layers.layer(0), layers.layer(1)];
    let mutate = |f: fn(&mut XieluParams)| {
        let mut t = base;
        for p in &mut t {
            f(p);
        }
        t.to_vec()
    };
    for (what, table) in [
        ("alpha_n", mutate(|p| p.alpha_n = 0.0)),
        ("alpha_p", mutate(|p| p.alpha_p = 0.0)),
        ("beta", mutate(|p| p.beta = 0.0)),
        ("eps", mutate(|p| p.eps = -1e-6)),
    ] {
        let mut d = load_graph_fixture(APERTUS);
        d.config.ffn_activation = FfnActivation::Xielu(XieluLayers::new(table));
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &APERTUS_GOLDEN,
        );
        assert!(worst > 1e-3, "{what}: the output moved by only {worst}");
    }
}

/// SwiGLU on the aliased pair -- what the loader would compute if the
/// activation were dropped and the alias kept -- diverges.
#[test]
fn running_swiglu_on_the_aliased_pair_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture(APERTUS);
    d.config.ffn_activation = FfnActivation::Swiglu;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &APERTUS_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "SwiGLU on the aliased pair moved the output by only {worst}"
    );
}

/// A file WITH an `ffn_gate` is refused by name through the same
/// branch `arcee` takes, and the message names both ungated graphs.
///
/// No apertus fixture with a gate exists (no converter writes one);
/// the arcee one has the gate and the same tensor shapes, so it is
/// loaded under an xIELU config to prove the branch fires for the
/// second ungated activation and not only for `== ReluSqr`.
#[test]
fn a_file_with_a_gate_is_refused_naming_the_ungated_graph() {
    let gated = graph_fixture_path("arcee_gated");
    let file = frink_gguf::GgufFile::open(&gated).expect("fixture opens");
    let mut config = ModelConfig::from_gguf(&file).expect("the arcee header parses");
    config.ffn_activation = FfnActivation::Xielu(XieluLayers::new(vec![
        XieluParams::from_gguf(
            0.8, 0.8, 0.5, -1e-6
        );
        2
    ]));
    let msg = match frink_models::Decoder::from_gguf(&gated, config) {
        Ok(_) => panic!("a file with a gate must be refused"),
        Err(err) => format!("{err}"),
    };
    assert!(msg.contains("ffn_gate"), "{msg}");
    assert!(msg.contains("ungated"), "{msg}");
    assert!(msg.contains("apertus.cpp:129-142"), "{msg}");
}

/// A file whose arrays are the wrong length is refused naming the key,
/// as llama.cpp refuses it (`key has wrong array length`).
#[test]
fn an_array_of_the_wrong_length_is_refused_naming_the_key() {
    use frink_gguf::{GgufValue, TensorSource};
    struct Wrapped(frink_gguf::GgufFile, GgufValue);
    impl TensorSource for Wrapped {
        fn metadata(&self, key: &str) -> Option<&GgufValue> {
            if key == "xielu.beta" {
                Some(&self.1)
            } else {
                self.0.metadata(key)
            }
        }
        fn find_tensor(&self, name: &str) -> Option<&frink_gguf::TensorInfo> {
            self.0.find_tensor(name)
        }
        fn tensor_bytes(&self, name: &str) -> Result<&[u8], frink_gguf::GgufError> {
            self.0.tensor_bytes(name)
        }
        fn tensor_mapped_range(
            &self,
            name: &str,
        ) -> Result<
            (
                std::sync::Arc<frink_gguf::MmapHandle>,
                std::ops::Range<usize>,
            ),
            frink_gguf::GgufError,
        > {
            self.0.tensor_mapped_range(name)
        }
    }
    let file = frink_gguf::GgufFile::open(graph_fixture_path(APERTUS)).expect("opens");
    let wrapped = Wrapped(
        file,
        GgufValue::Array(vec![
            GgufValue::F32(0.5),
            GgufValue::F32(0.5),
            GgufValue::F32(0.5),
        ]),
    );
    let msg = match ModelConfig::from_gguf(&wrapped) {
        Ok(_) => panic!("a three-entry array on a two-layer file must be refused"),
        Err(err) => format!("{err}"),
    };
    assert!(msg.contains("xielu.beta"), "{msg}");
    assert!(msg.contains("3 entries"), "{msg}");
}

/// `attn_q_norm.bias` / `attn_k_norm.bias`: created by `apertus.cpp:50,52`
/// and never read (`:93,96` pass `NULL`). libllama's logits for the
/// file that carries them are BYTE-IDENTICAL to the base file's
/// (measured, `cmp` on the two reference dumps), so frink serves it
/// the same way: loaded, ignored, and matching the same golden.
#[test]
fn qk_norm_biases_llama_cpp_never_reads_are_ignored_not_applied_or_refused() {
    assert!(
        frink_gguf::GgufFile::open(graph_fixture_path(APERTUS_QKNORM_BIAS))
            .expect("opens")
            .find_tensor("blk.0.attn_q_norm.bias")
            .is_some(),
        "the fixture must carry the bias, or the test measures nothing"
    );
    let d = load_graph_fixture(APERTUS_QKNORM_BIAS);
    let mut kv = graph_caches(&d);
    common::assert_close(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &APERTUS_GOLDEN,
        common::GRAPH_TOL,
        "apertus with unread qk-norm biases",
    );
}

/// The rest of the graph, pinned against the C.
#[test]
fn apertus_is_a_neox_rope_llama_with_pre_rope_qk_norm_and_no_scalars() {
    let d = load_graph_fixture(APERTUS);
    assert_eq!(
        d.config.rope_layout,
        RopeLayout::Neox,
        "llama-model.cpp:2671"
    );
    assert_eq!(
        d.config.attention_scale, None,
        "apertus.cpp:74-75 with f_attention_scale unset"
    );
    assert_eq!(d.config.embedding_scale, None);
    assert_eq!(d.config.residual_scale, None);
    assert_eq!(d.config.sliding_window, None);
    assert!(d.layers[0].attn.q_norm.is_some() && d.layers[0].attn.k_norm.is_some());
    assert_eq!(
        d.layers[0].attn.q_norm.as_ref().map(Vec::len),
        Some(6),
        "per head, apertus.cpp:49"
    );
}

/// Metal: no fused kernel spells xIELU, and the model answers no
/// whole-model activation, so every fused launch is refused through
/// the predicate they share. With `--features metal` on the M2 Pro the
/// three-path test above is what proves the host fallback is taken.
#[test]
fn a_parameterised_activation_has_no_whole_model_answer_for_the_fused_kernels() {
    let d = load_graph_fixture(APERTUS);
    assert_eq!(d.config.model_ffn_act(), None);
    assert_eq!(
        d.config
            .model_ffn_act()
            .and_then(GluAct::fused_kernel_gelu_flag),
        None
    );
}

/// With Metal switched ON, an xIELU model must never reach a fused
/// Metal launch (no kernel spells it), and still match libllama from
/// the host bodies. The control -- the same file told it is plain
/// SwiGLU on the aliased pair, wrong but kernel-shaped -- must reach
/// one, or the switch was not live and the first half proved nothing.
///
/// `cargo test -p frink-models --features metal --test per_layer_activation_graphs -- --ignored`
#[test]
#[ignore = "needs Apple Metal GPU"]
fn a_parameterised_activation_stays_on_the_host_with_metal_switched_on() {
    #[cfg(feature = "metal")]
    common::pin_backend(common::Backend::Metal);
    #[cfg(not(feature = "metal"))]
    {
        eprintln!("skip: built without --features metal");
    }
    #[cfg(feature = "metal")]
    {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        let d = load_graph_fixture(APERTUS);
        let mut kv = graph_caches(&d);
        common::assert_close(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &APERTUS_GOLDEN,
            common::GRAPH_TOL,
            "apertus: prefill with Metal on",
        );
        let mut kv = graph_caches(&d);
        let mut out = Vec::new();
        for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
            out = d.forward_token(tok, pos, &mut kv);
        }
        common::assert_close(
            &out,
            &APERTUS_GOLDEN,
            common::GRAPH_TOL,
            "apertus: decode with Metal on",
        );
        assert!(
            !d.metal_attn_kv_allocated(),
            "an xIELU model must never reach a fused Metal attention launch"
        );
        let mut d = load_graph_fixture(APERTUS);
        d.config.ffn_activation = FfnActivation::Swiglu;
        let mut kv = graph_caches(&d);
        for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
            let _ = d.forward_token(tok, pos, &mut kv);
        }
        assert!(
            d.metal_attn_kv_allocated(),
            "the SwiGLU control did not reach a fused launch; the Metal switch is not live \
             in this process and the assertions above proved nothing"
        );
    }
}
