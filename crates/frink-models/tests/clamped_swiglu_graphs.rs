//! Step-3.5, checked against llama.cpp itself: PER-LAYER SwiGLU clamp
//! arrays read by SITE, and a rotary width that is TWO-VALUED by the
//! sliding-or-full fact.
//!
//! `step35` was triaged NEW CODE on two things after every other
//! blocker its verdict named had closed on an earlier seam:
//!
//! * `step35.cpp:28-29` reads `swiglu_clamp_exp` and
//!   `swiglu_clamp_shexp` as optional `n_layer`-long arrays, and
//!   llama.cpp's GENERIC FFN builders apply layer `il`'s entry when it
//!   is above `1e-6` -- `build_moe_ffn` (llama-graph.cpp:2146-2164)
//!   reads `_exp` for the routed experts, `build_ffn` (:1751-1768)
//!   reads `_shexp` for the shared experts AND the leading dense
//!   layers -- as `min(silu(gate), l) * clamp(up, -l, l)`.
//! * `step35.cpp:9` halves `n_rot_full` AFTER llama-model.cpp:1222
//!   seeded `n_rot_swa` from it, so `n_rot(il)` (llama-hparams.cpp:
//!   85-91) is the whole head on the sliding layers and half on the
//!   full ones. No key says so.
//!
//! **The seams.** The clamp is the SECOND body on the per-layer
//! activation plumbing that xIELU opened (`frink_models::act_layers`,
//! `tests/per_layer_activation_graphs.rs`): `FfnActivation::
//! SwigluClamped` carries both arrays, `GluAct::SwigluClamped` is the
//! arithmetic, and the one thing it needed that xIELU did not is the
//! SITE, so `ModelConfig::layer_ffn_acts(il)` answers a `routed` /
//! `dense` pair and every FFN body names the field it runs. The
//! rotary width is `ModelConfig::rope_dim_swa` (`frink_models::
//! swa_geometry`), handed out per layer through `layer_rope`; the
//! fused Metal launches take one width and are fenced off.
//!
//! **Where the numbers come from.** Each `*_GOLDEN` was produced by
//! running llama.cpp's own graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! Regenerating:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_step35_fixture.py \
//!     crates/frink-models/tests/fixtures/step35_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_step35_fixture.py \
//!     crates/frink-models/tests/fixtures/step35_noclamp_tiny.gguf --no-clamp
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_step35_fixture.py \
//!     crates/frink-models/tests/fixtures/step35_mtp_tiny.gguf --mtp
//! /tmp/ref_logits crates/frink-models/tests/fixtures/step35_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GRAPH_PROMPT,
};
use frink_models::act_layers::{LayerFfnActs, SwigluClamps};
use frink_models::{FfnActivation, ModelConfig, RopeLayout};
use frink_moe::{ClampForm, GluAct};

const STEP35: &str = "step35";
/// The same weights with neither clamp key.
const STEP35_NOCLAMP: &str = "step35_noclamp";
/// The same trunk plus one NextN block inside `block_count`.
const STEP35_MTP: &str = "step35_mtp";

/// llama.cpp's logits for `step35_tiny.gguf` over [`GRAPH_PROMPT`].
/// `step35_mtp_tiny.gguf` produces the SAME 48 numbers, byte for byte
/// (`cmp` on the two reference dumps): the block is skipped upstream.
const STEP35_GOLDEN: [f32; 48] = [
    -2.003458,
    -0.48528326,
    0.43615007,
    0.549622,
    -0.95694655,
    0.6806216,
    0.1855886,
    0.3616861,
    -1.8394105,
    3.7911568,
    -0.5240021,
    1.2821372,
    0.7942611,
    -0.10253036,
    0.7256867,
    -1.1815972,
    0.3106292,
    1.8533351,
    0.9554646,
    -0.3798321,
    1.4952782,
    -0.94514954,
    -2.1832008,
    0.3216235,
    -0.25822186,
    -0.6459647,
    0.43663573,
    1.4655704,
    2.0640159,
    -3.3604739,
    2.0372353,
    0.59971935,
    1.2716625,
    -0.850787,
    -0.83550465,
    -0.42506137,
    -0.8198667,
    -0.705837,
    0.78048927,
    -1.6572762,
    -0.1676774,
    -2.252253,
    -1.4420235,
    -0.2886601,
    -0.12879175,
    -1.2347265,
    1.4604833,
    1.933814,
];

/// llama.cpp's logits for `step35_noclamp_tiny.gguf`: the same
/// weights, plain SwiGLU everywhere.
const STEP35_NOCLAMP_GOLDEN: [f32; 48] = [
    1.2275026,
    0.06808083,
    -1.3012469,
    0.28708717,
    -0.17766884,
    0.46882653,
    -0.8462756,
    -2.1837316,
    -0.58917964,
    -0.9890877,
    0.6721114,
    0.16126971,
    1.9490702,
    -2.3880768,
    0.037116703,
    -0.60543114,
    0.33002222,
    -1.7687161,
    -0.5966573,
    -0.13531338,
    1.3225213,
    1.897458,
    0.84041476,
    1.5161291,
    -0.06241022,
    -0.28277737,
    1.9732094,
    1.7348502,
    0.93495405,
    -0.3264407,
    -0.58875275,
    0.9350789,
    -0.45100254,
    1.5473307,
    0.13078018,
    0.8129002,
    -0.8306872,
    1.2299483,
    1.8494061,
    0.6669048,
    -2.3530586,
    0.43955335,
    2.2108452,
    -0.3602634,
    1.7626252,
    -0.62505186,
    0.79487485,
    -0.18159232,
];

/// The row itself, on all three forward paths.
#[test]
fn step35_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(STEP35, &STEP35_GOLDEN);
}

/// A file with neither clamp key runs plain SwiGLU on every layer, as
/// the zero-filled arrays do upstream; the two goldens differ, so the
/// clamp is visible.
#[test]
fn a_file_with_no_clamp_keys_runs_plain_swiglu_as_llama_cpp_does() {
    assert_all_three_paths_match(STEP35_NOCLAMP, &STEP35_NOCLAMP_GOLDEN);
    let d = load_graph_fixture(STEP35_NOCLAMP);
    assert_eq!(d.config.ffn_activation, FfnActivation::Swiglu);
    assert!(
        worst_vs(&STEP35_NOCLAMP_GOLDEN, &STEP35_GOLDEN) > 1e-1,
        "the clamped and unclamped fixtures must disagree, or the clamp is invisible"
    );
}

/// One NextN block inside `block_count` with `nextn_predict_layers =
/// 1`, as `step3.py:222-223` writes it: skipped, and the trunk answers
/// the trunk's golden.
#[test]
fn a_nextn_block_is_skipped_as_llama_cpp_skips_it() {
    assert_all_three_paths_match(STEP35_MTP, &STEP35_GOLDEN);
    let d = load_graph_fixture(STEP35_MTP);
    assert_eq!(d.config.n_layers, 4);
    assert_eq!(d.config.n_mtp_blocks, 1);
    assert_eq!(d.layers.len(), 4);
    let FfnActivation::SwigluClamped(clamps) = &d.config.ffn_activation else {
        panic!("{:?}", d.config.ffn_activation)
    };
    assert_eq!(
        clamps.len(),
        4,
        "the arrays are block_count long and truncated to the trunk"
    );
}

/// The number in the report, so it can be regenerated rather than
/// trusted. Run with `--nocapture` to see it.
#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (STEP35, &STEP35_GOLDEN),
        (STEP35_NOCLAMP, &STEP35_NOCLAMP_GOLDEN),
        (STEP35_MTP, &STEP35_GOLDEN),
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

/// The loaded shape: each site reads its own array, a zero entry is
/// plain SwiGLU on that site alone, and the fixture's arrays are the
/// ones the script wrote.
#[test]
fn each_layer_s_two_sites_carry_their_own_clamp() {
    let d = load_graph_fixture(STEP35);
    // scripts/make_step35_fixture.py: CLAMP_EXP = [0, 1.5, 0, 2.5],
    // CLAMP_SHEXP = [2.0, 3.0, 1.0, 0].
    let want = [
        (
            GluAct::Swiglu,
            GluAct::SwigluClamped {
                limit: 2.0,
                form: ClampForm::AfterSilu,
            },
        ),
        (
            GluAct::SwigluClamped {
                limit: 1.5,
                form: ClampForm::AfterSilu,
            },
            GluAct::SwigluClamped {
                limit: 3.0,
                form: ClampForm::AfterSilu,
            },
        ),
        (
            GluAct::Swiglu,
            GluAct::SwigluClamped {
                limit: 1.0,
                form: ClampForm::AfterSilu,
            },
        ),
        (
            GluAct::SwigluClamped {
                limit: 2.5,
                form: ClampForm::AfterSilu,
            },
            GluAct::Swiglu,
        ),
    ];
    for (il, (routed, dense)) in want.into_iter().enumerate() {
        assert_eq!(
            d.config.layer_ffn_acts(il),
            LayerFfnActs { routed, dense },
            "layer {il}"
        );
    }
    assert_eq!(d.config.model_ffn_act(), None);
    assert!(!d.config.ffn_is_ungated());
    // Layer 0 is dense by TENSOR PRESENCE (`step35.cpp:304`), the
    // others MoE with a shared expert on each.
    assert_eq!(d.layers[0].moe.n_experts(), 1);
    assert!(d.layers[0].moe.shared_experts.is_empty());
    for il in 1..4 {
        assert_eq!(d.layers[il].moe.n_experts(), 4, "layer {il}");
        assert_eq!(d.layers[il].moe.shared_experts.len(), 1, "layer {il}");
    }
}

/// The sabotage that matters: reading the right array at the wrong
/// site, or the same array at both. Swapping `routed` and `dense`
/// moves the logits; so does giving both sites the routed array.
#[test]
fn reading_the_arrays_at_the_wrong_site_diverges_from_llama_cpp() {
    let routed = vec![0.0, 1.5, 0.0, 2.5];
    let dense = vec![2.0, 3.0, 1.0, 0.0];
    for (what, r, d) in [
        ("swapped", dense.clone(), routed.clone()),
        ("routed at both", routed.clone(), routed.clone()),
        ("dense at both", dense.clone(), dense.clone()),
    ] {
        let mut dec = load_graph_fixture(STEP35);
        dec.config.ffn_activation =
            FfnActivation::SwigluClamped(SwigluClamps::new(r, d, ClampForm::AfterSilu));
        let mut kv = graph_caches(&dec);
        let worst = worst_vs(
            &dec.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &STEP35_GOLDEN,
        );
        assert!(worst > 1e-2, "{what}: the output moved by only {worst}");
    }
}

/// Dropping the clamp arithmetic while keeping the arrays -- what a
/// decoder that read the keys and ran SwiGLU would do -- gives the
/// unclamped file's answer, not this file's.
#[test]
fn running_plain_swiglu_on_the_clamped_file_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture(STEP35);
    d.config.ffn_activation = FfnActivation::Swiglu;
    let mut kv = graph_caches(&d);
    let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    assert!(worst_vs(&got, &STEP35_GOLDEN) > 1e-1);
    common::assert_close(
        &got,
        &STEP35_NOCLAMP_GOLDEN,
        common::GRAPH_TOL,
        "the same weights unclamped are the unclamped fixture",
    );
}

/// The rotary width is two-valued: the file says 8 (the whole head),
/// `step35.cpp:9` halves it for the full layers, and the sliding
/// layers keep 8. Rotating every layer at one width -- either one --
/// diverges.
#[test]
fn the_full_layers_rotate_half_the_head_and_the_sliding_layers_all_of_it() {
    let d = load_graph_fixture(STEP35);
    assert_eq!(d.config.head_dim, 8);
    assert_eq!(
        d.config.rope_dim,
        Some(4),
        "n_rot_full, halved at step35.cpp:9"
    );
    assert_eq!(
        d.config.rope_dim_swa,
        Some(8),
        "n_rot_swa, seeded before the halving"
    );
    assert!(d.config.rope_dim_varies_by_layer());
    // SWA_PATTERN = [F, T, F, T].
    for (il, want) in [(0, Some(4)), (1, None), (2, Some(4)), (3, None)] {
        let rope = d.config.layer_rope(il).expect("every layer rotates");
        assert_eq!(rope.rot_dim, want, "layer {il}");
        assert_eq!(
            rope.theta,
            if want.is_none() { 5000.0 } else { 10000.0 },
            "layer {il}: the sliding layers' own base"
        );
    }
    for (what, full, swa) in [("all half", Some(4), None), ("all whole", None, None)] {
        let mut d = load_graph_fixture(STEP35);
        d.config.rope_dim = full;
        d.config.rope_dim_swa = swa;
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &STEP35_GOLDEN,
        );
        assert!(worst > 1e-2, "{what}: the output moved by only {worst}");
    }
}

/// The rest of the graph, pinned against the C: every seam the verdict
/// had already crossed off is carried by this fixture, not assumed.
#[test]
fn step35_carries_every_seam_its_verdict_crossed_off() {
    let d = load_graph_fixture(STEP35);
    assert_eq!(
        d.config.rope_layout,
        RopeLayout::Neox,
        "llama-model.cpp:2680"
    );
    // Per-layer head counts (crate::layer_shapes).
    assert!(!d.config.layer_shapes.is_uniform());
    let heads = |il: usize| match d.config.layer_shape(il).attention {
        frink_models::layer_shapes::AttnShape::Gqa {
            n_heads,
            n_kv_heads,
        } => (n_heads, n_kv_heads),
        other => panic!("layer {il}: {other:?}"),
    };
    assert_eq!(
        [heads(0), heads(1), heads(2), heads(3)],
        [(2, 1), (4, 2), (2, 1), (4, 2)]
    );
    // The window array, with the window narrower than the prompt.
    assert_eq!(d.config.sliding_window, Some(3));
    assert_eq!(
        (0..4)
            .map(|il| d.config.layer_sliding_window(il).is_some())
            .collect::<Vec<_>>(),
        [false, true, false, true]
    );
    // Sigmoid routing by default, the router bias, the scale, the norm.
    assert_eq!(
        d.config.moe.gating,
        frink_moe::GatingFunction::Sigmoid,
        "step35.cpp:19-21 with no key written"
    );
    assert!(d.layers[1].moe.exp_probs_bias.is_some());
    assert_eq!(d.config.moe.expert_weights_scale, 2.5);
    assert!(d.config.moe.norm_topk_prob);
    // The per-head sigmoid attention gate on every layer.
    for il in 0..4 {
        assert!(d.layers[il].attn.output_gate.is_some(), "layer {il}");
    }
    assert_eq!(d.config.attention_scale, None, "step35.cpp:262");
}

/// Metal: no fused kernel spells the clamp and none takes two rotary
/// widths, so the model answers no whole-model activation and the
/// shared fence refuses it. With `--features metal` on the M2 Pro the
/// three-path test above is what proves the host fallback is taken.
#[test]
fn a_clamped_two_width_model_has_no_whole_model_answer_for_the_fused_kernels() {
    let d = load_graph_fixture(STEP35);
    assert_eq!(d.config.model_ffn_act(), None);
    assert!(d.config.rope_dim_varies_by_layer());
    // And the unclamped file still varies its rotary width, so it is
    // refused by the width alone.
    let d = load_graph_fixture(STEP35_NOCLAMP);
    assert_eq!(d.config.model_ffn_act(), Some(GluAct::Swiglu));
    assert!(d.config.rope_dim_varies_by_layer());
}

/// A file whose clamp array has the wrong length is refused naming the
/// key, as llama.cpp refuses it (`key has wrong array length`).
#[test]
fn a_clamp_array_of_the_wrong_length_is_refused_naming_the_key() {
    use frink_gguf::{GgufValue, TensorSource};
    struct Wrapped(frink_gguf::GgufFile, GgufValue);
    impl TensorSource for Wrapped {
        fn metadata(&self, key: &str) -> Option<&GgufValue> {
            if key == "step35.swiglu_clamp_shexp" {
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
    let file = frink_gguf::GgufFile::open(graph_fixture_path(STEP35)).expect("opens");
    let wrapped = Wrapped(
        file,
        GgufValue::Array(vec![GgufValue::F32(1.0), GgufValue::F32(1.0)]),
    );
    let msg = match ModelConfig::from_gguf(&wrapped) {
        Ok(_) => panic!("a two-entry array on a four-layer file must be refused"),
        Err(err) => format!("{err}"),
    };
    assert!(msg.contains("swiglu_clamp_shexp"), "{msg}");
    assert!(msg.contains("2 entries"), "{msg}");
}

/// With Metal switched ON, the clamped, two-width model must never
/// reach a fused Metal launch (no kernel spells the clamp, none takes
/// two rotary widths), and still match libllama from the host bodies;
/// the unclamped file is held off by the width alone.
///
/// No control from this file can reach a launch: with both of its own
/// fences lifted it is still non-uniform in its head counts, and that
/// is a third fence (`crate::layer_shapes`). The laguna M.1 control in
/// `tests/gated_attention_graphs.rs` is what proves the switch is
/// live; what this suite pins is that lifting both of THIS model's
/// fences leaves exactly the shape fence standing.
///
/// `cargo test -p frink-models --features metal --test clamped_swiglu_graphs -- --ignored`
#[test]
#[ignore = "needs Apple Metal GPU"]
fn a_clamped_two_width_model_stays_on_the_host_with_metal_switched_on() {
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
        for (name, golden) in [
            (STEP35, &STEP35_GOLDEN),
            (STEP35_NOCLAMP, &STEP35_NOCLAMP_GOLDEN),
        ] {
            let d = load_graph_fixture(name);
            let mut kv = graph_caches(&d);
            common::assert_close(
                &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
                golden,
                common::GRAPH_TOL,
                &format!("{name}: prefill with Metal on"),
            );
            let mut kv = graph_caches(&d);
            let mut out = Vec::new();
            for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
                out = d.forward_token(tok, pos, &mut kv);
            }
            common::assert_close(
                &out,
                golden,
                common::GRAPH_TOL,
                &format!("{name}: decode with Metal on"),
            );
            assert!(
                !d.metal_attn_kv_allocated(),
                "{name}: a two-width model must never reach a fused Metal attention launch"
            );
        }
        let mut d = load_graph_fixture(STEP35_NOCLAMP);
        d.config.rope_dim_swa = None;
        assert!(!d.config.rope_dim_varies_by_layer());
        assert_eq!(d.config.model_ffn_act(), Some(GluAct::Swiglu));
        assert!(
            !d.config.layer_shapes.is_uniform(),
            "still per-layer heads, the third fence"
        );
    }
}
