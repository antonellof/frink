//! Per-layer heterogeneous shapes, checked against llama.cpp itself:
//! `deci` and `openelm` on ONE seam.
//!
//! Both were triaged NEW CODE on the same fact. llama.cpp reads
//! `{arch}.attention.head_count`, `.head_count_kv` and
//! `{arch}.feed_forward_length` as a scalar OR an `n_layer`-long array
//! for EVERY architecture (`get_key_or_arr`, llama-model.cpp:1149-1158)
//! and hands most graphs layer 0 through `LLAMA_LOAD_LOCALS`; these two
//! index the arrays per layer in both their tensor loader and their
//! graph (`deci.cpp:30-34,103-105`, `openelm.cpp:26-28,67-69`), and
//! `ModelConfig` carried all three as scalars that every host body read
//! once above its layer loop.
//!
//! **Measured before built.** All 155 `src/models/*.cpp` were scanned
//! for `n_head(i)`, `n_head_kv(i)`, `n_ff(i)`, `n_embd_k_gqa(i)`,
//! `n_embd_v_gqa(i)`, `n_rot(i)` and the `_arr` fields:
//! `layer_shapes::PER_LAYER_SHAPE_ARCHS` is the result, with what each
//! row still needs. Three findings from the scan and the fixtures:
//!
//! 1. `n_rot(il)` is NOT an array upstream: `llama-hparams.cpp:85-91`
//!    is `is_swa(il) ? n_rot_swa : n_rot_full`, so `step35`'s and
//!    `laguna`'s "per-layer rotary width" is a two-valued SWA/full
//!    split, a different and smaller seam than this one.
//! 2. `granite.cpp:204` reads `n_head(il)` in its graph but sizes its
//!    tensors from layer 0 (`:68`), so a heterogeneous Granite file
//!    fails in llama.cpp's own loader; it is not in the table.
//! 3. `deci.cpp:147-149` `continue`s an FFN-free layer BEFORE the
//!    residual add at `:150-153`. On a layer that also has attention
//!    the computed attention output is discarded: scaling that layer's
//!    attention weights by 3 leaves libllama's logits byte-identical
//!    (measured on `deci_attn_ffnfree_tiny.gguf`). And if such a layer
//!    is LAST, libllama aborts outright (`GGML_ASSERT(buffer)`,
//!    ggml-backend.cpp:194) because the `inp_out_ids` `get_rows` result
//!    never rejoins the graph. frink refuses the first by name and the
//!    fixtures avoid the second.
//!
//! **What the seam is.** `ModelConfig::layer_shape(il)` is the ONE
//! accessor for a layer's `AttnShape::{Gqa, Linear, Absent}` and
//! `ffn_dim`; the scalars are documented as the widest layer's, for
//! budgets. `ModelConfig::new_kv_caches` / `new_paged_kv` size each
//! layer's cache from its own shape, replacing ~90 hand-written
//! `KvCache::new(config.n_kv_heads, ..)` sites, and `KvCache::push`
//! asserts the row width so a cache built from the scalar panics on
//! the first token of a narrower layer. `metal_can_serve_model` keeps
//! a non-uniform model off every fused Metal launch (one `n_heads`
//! argument, one `MetalKvBuffers` geometry), the CUDA resident KV is
//! gated on the same predicate, and the slot-file writer refuses a
//! set of layers whose geometries differ.
//!
//! **Where the numbers come from.** Each golden was produced by running
//! llama.cpp's own graph over its fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! Regenerating:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_deci_fixture.py \
//!     crates/frink-models/tests/fixtures/deci_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_deci_fixture.py \
//!     crates/frink-models/tests/fixtures/deci_kv_only_tiny.gguf --kv-only
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_deci_fixture.py \
//!     crates/frink-models/tests/fixtures/deci_attn_ffnfree_tiny.gguf --attn-ffnfree
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_openelm_fixture.py \
//!     crates/frink-models/tests/fixtures/openelm_tiny.gguf
//! /tmp/ref_logits <fixture> 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GRAPH_PROMPT,
};
use frink_core::cache::KvCache;
use frink_models::capability::QkNormStyle;
use frink_models::layer_shapes::{AttnShape, LayerShape, LayerShapes};
use frink_models::{ModelConfig, NormOp, RopeLayout};

const DECI: &str = "deci";
const DECI_KV_ONLY: &str = "deci_kv_only";
const OPENELM: &str = "openelm";

const DECI_GOLDEN: [f32; 48] = [
    1.4331788,
    -2.71866,
    0.66528094,
    -3.5902803,
    -0.7970845,
    1.2758441,
    1.0586318,
    1.942835,
    0.5134384,
    -1.1962459,
    -0.056241095,
    -1.8373556,
    0.7332511,
    0.18208505,
    -2.9639764,
    0.6070172,
    2.620933,
    -1.8580642,
    0.28291583,
    -2.04144,
    0.0794121,
    0.43324864,
    -0.10005893,
    1.5135344,
    0.027186811,
    -2.1996922,
    -0.02839902,
    0.18154687,
    1.2254814,
    0.10891289,
    1.340531,
    0.4840635,
    0.66370267,
    0.17737979,
    0.33590952,
    -0.36287254,
    1.276825,
    2.2857864,
    0.0444739,
    1.0342226,
    0.2978132,
    0.6721678,
    -1.2298176,
    -1.9146186,
    -1.5663421,
    1.7788808,
    -0.22746408,
    1.1275123,
];

const DECI_KV_ONLY_GOLDEN: [f32; 48] = [
    1.8845997,
    -0.4751297,
    0.9248638,
    -2.1040947,
    0.7407677,
    1.0108163,
    -1.7061309,
    -0.36859614,
    1.3944849,
    -1.4286779,
    -0.1050726,
    2.1006773,
    1.5668745,
    1.9392642,
    0.067401096,
    -1.1968663,
    1.7726955,
    2.1108177,
    -0.61612785,
    -0.61165845,
    2.1078165,
    0.76744056,
    1.2139252,
    -1.2811415,
    3.0353029,
    0.96419525,
    0.13860774,
    -0.13149408,
    0.17039818,
    -3.0274034,
    1.2023331,
    2.1906447,
    0.375583,
    -1.7670578,
    -0.76667136,
    0.004958898,
    0.3659907,
    0.17856643,
    -2.0640225,
    0.07118863,
    0.9068029,
    -0.44177684,
    0.83832645,
    0.30876932,
    -0.02880472,
    2.2662604,
    -1.2199794,
    0.8183892,
];

const OPENELM_GOLDEN: [f32; 48] = [
    -0.9687437,
    -0.009876013,
    -1.0234756,
    1.4560915,
    -0.64805037,
    -0.072382405,
    0.078888044,
    3.4787698,
    0.18833429,
    -0.5133585,
    -0.89827335,
    1.2050449,
    1.0119542,
    -0.73539424,
    -1.3891572,
    -0.013600461,
    1.855889,
    -0.33100748,
    2.2477822,
    -0.13009682,
    -0.7881586,
    -0.33944038,
    0.7807396,
    -0.751681,
    -0.2279007,
    1.8154901,
    -1.2683353,
    -0.3075223,
    0.60595393,
    2.0340488,
    1.372062,
    0.47275484,
    0.6995953,
    0.37244004,
    -0.45028436,
    -0.47168425,
    -2.3655455,
    1.573791,
    0.28353986,
    0.31686136,
    -0.8692597,
    -0.2417105,
    -1.7027298,
    0.15012643,
    -0.722872,
    -0.5758084,
    -1.4561477,
    -0.9098055,
];

/// The three rows, on all three forward paths.
#[test]
fn deci_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(DECI, &DECI_GOLDEN);
}

#[test]
fn deci_kv_only_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(DECI_KV_ONLY, &DECI_KV_ONLY_GOLDEN);
}

#[test]
fn openelm_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(OPENELM, &OPENELM_GOLDEN);
}

/// The numbers in the report, so they can be regenerated rather than
/// trusted. Run with `--nocapture` to see them.
#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (DECI, &DECI_GOLDEN),
        (DECI_KV_ONLY, &DECI_KV_ONLY_GOLDEN),
        (OPENELM, &OPENELM_GOLDEN),
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

/// The loaded shapes of the Nemotron-shaped fixture: one layer of each
/// kind, in the order the file declares them, and the tensors each kind
/// does and does not carry.
///
/// Structural rather than numeric, so that a loader that read the
/// arrays back into scalars says which layer it got wrong before the
/// numeric test says "diverged".
#[test]
fn deci_loads_one_layer_of_each_kind_with_the_tensors_that_kind_has() {
    let d = load_graph_fixture(DECI);
    let want = vec![
        LayerShape {
            attention: AttnShape::Gqa {
                n_heads: 4,
                n_kv_heads: 2,
            },
            ffn_dim: 40,
        },
        LayerShape {
            attention: AttnShape::Absent,
            ffn_dim: 0,
        },
        LayerShape {
            attention: AttnShape::Linear,
            ffn_dim: 24,
        },
        LayerShape {
            attention: AttnShape::Absent,
            ffn_dim: 32,
        },
    ];
    assert_eq!(d.config.layer_shapes, LayerShapes::PerLayer(want.clone()));
    for (il, shape) in want.iter().enumerate() {
        assert_eq!(d.config.layer_shape(il), *shape, "blk.{il}");
    }
    // The scalars are the WIDEST layer's, for budgets.
    assert_eq!(d.config.n_heads, 4);
    assert_eq!(d.config.n_kv_heads, 2);
    assert_eq!(d.config.moe.expert_ffn_dim, 40);
    assert_eq!(
        d.config.kv_heads_all_layers(),
        2,
        "only blk.0 caches anything"
    );

    // blk.0: a plain GQA layer.
    assert_eq!(d.layers[0].attn.q_proj.rows(), 24);
    assert_eq!(d.layers[0].attn.k_proj.rows(), 12);
    // blk.1: the identity. No projections, no norms.
    assert_eq!(d.layers[1].attn.q_proj.rows(), 0);
    assert_eq!(d.layers[1].attn.o_proj.rows(), 0);
    assert_eq!(
        d.layers[1].attn.norm_weight,
        NormOp::None,
        "deci.cpp:107-109: no attn_norm"
    );
    assert_eq!(
        d.layers[1].moe.norm_weight,
        NormOp::None,
        "deci.cpp:52-54: no ffn_norm"
    );
    // blk.2: wo-only. A square attn_output, a norm, no Q/K/V.
    assert_eq!(d.layers[2].attn.q_proj.rows(), 0);
    assert_eq!(d.layers[2].attn.o_proj.rows(), 24);
    assert_eq!(
        d.layers[2].attn.o_proj.cols(),
        24,
        "deci.cpp:39: {{n_embd, n_embd}}"
    );
    assert!(matches!(d.layers[2].attn.norm_weight, NormOp::Rms(_)));
    // blk.3: FFN only.
    assert_eq!(d.layers[3].attn.o_proj.rows(), 0);
    assert_eq!(d.layers[3].attn.norm_weight, NormOp::None);
    d.layers[3]
        .moe
        .with_expert(0, |ex| assert_eq!(ex.up.rows(), 32));
}

/// The DeciLM-7B shape: only `head_count_kv` varies, every layer is
/// GQA, and the caches are sized per layer.
#[test]
fn deci_kv_only_varies_the_gqa_ratio_per_layer_and_nothing_else() {
    let d = load_graph_fixture(DECI_KV_ONLY);
    let shapes: Vec<AttnShape> = (0..3)
        .map(|il| d.config.layer_shape(il).attention)
        .collect();
    assert_eq!(
        shapes,
        [
            AttnShape::Gqa {
                n_heads: 4,
                n_kv_heads: 2
            },
            AttnShape::Gqa {
                n_heads: 4,
                n_kv_heads: 1
            },
            AttnShape::Gqa {
                n_heads: 4,
                n_kv_heads: 4
            },
        ]
    );
    assert!((0..3).all(|il| d.config.layer_shape(il).ffn_dim == 40));
    assert_eq!(d.config.n_kv_heads, 4, "the widest layer");
    assert_eq!(d.config.kv_heads_all_layers(), 7);
    let caches = d.config.new_kv_caches();
    assert_eq!(
        caches.iter().map(|c| c.n_kv_heads).collect::<Vec<_>>(),
        [2, 1, 4]
    );
    for (il, layer) in d.layers.iter().enumerate() {
        assert_eq!(
            layer.attn.k_proj.rows(),
            shapes[il].n_kv_heads() * 6,
            "blk.{il}"
        );
    }
}

/// openelm: three layers, three shapes, one fused QKV per layer split
/// by that layer's own counts, per-head QK-norm before RoPE, tied head.
#[test]
fn openelm_splits_each_layer_s_fused_qkv_by_its_own_counts() {
    let d = load_graph_fixture(OPENELM);
    let want = [(4, 2, 40), (2, 1, 24), (3, 3, 32)];
    for (il, (nh, nkv, nff)) in want.iter().enumerate() {
        let shape = d.config.layer_shape(il);
        assert_eq!(
            shape,
            LayerShape {
                attention: AttnShape::Gqa {
                    n_heads: *nh,
                    n_kv_heads: *nkv
                },
                ffn_dim: *nff
            },
            "blk.{il}"
        );
        let layer = &d.layers[il];
        assert_eq!(layer.attn.q_proj.rows(), nh * 6, "blk.{il} Q");
        assert_eq!(layer.attn.k_proj.rows(), nkv * 6, "blk.{il} K");
        assert_eq!(layer.attn.v_proj.rows(), nkv * 6, "blk.{il} V");
        assert_eq!(layer.attn.o_proj.cols(), nh * 6, "blk.{il} wo");
        layer
            .moe
            .with_expert(0, |ex| assert_eq!(ex.up.rows(), *nff, "blk.{il} n_ff"));
        assert_eq!(
            layer.attn.q_norm.as_ref().map(|w| w.len()),
            Some(6),
            "openelm.cpp:36: {{n_embd_head_k}}"
        );
    }
    assert_eq!(d.config.qk_norm_style, QkNormStyle::PerHead);
    assert_eq!(
        d.config.rope_layout,
        RopeLayout::Neox,
        "llama-model.cpp:2650"
    );
    assert_eq!(d.config.attention_scale, None, "openelm.cpp:117");
    assert_eq!(d.config.rms_norm_eps, 1e-6);
    assert_eq!(d.config.n_heads, 4);
    assert_eq!(d.config.n_kv_heads, 3);
    assert_eq!(d.config.kv_heads_all_layers(), 6);
}

/// The scalar is not the answer: a cache set built from it, the way
/// every call site used to build one, refuses the first token of a
/// narrower layer instead of storing a misaligned history.
///
/// This is the guard that turns a missed `new_kv_caches` call site
/// into a panic rather than a fluent wrong answer.
#[test]
fn a_cache_set_built_from_the_scalar_panics_on_the_first_narrower_layer() {
    let d = load_graph_fixture(OPENELM);
    let mut wrong: Vec<KvCache> = d
        .layers
        .iter()
        .map(|_| KvCache::new(d.config.n_kv_heads, d.config.head_dim))
        .collect();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        d.forward_token(GRAPH_PROMPT[0], 0, &mut wrong)
    }));
    assert!(
        res.is_err(),
        "blk.0 has 2 KV heads and the scalar says 3; push must refuse"
    );
    // And the right set runs.
    let mut right = d.config.new_kv_caches();
    let _ = d.forward_token(GRAPH_PROMPT[0], 0, &mut right);
    assert_eq!(right[0].rows(), 1);
}

/// Each of deci's non-GQA layer kinds is load-bearing: turning the
/// wo-only layer into an attention-free one, or running the FFN-free
/// layer's placeholders, diverges from the golden.
///
/// Each substitution on its own, so the suite cannot pass by getting
/// one kind right and the other wrong.
#[test]
fn each_of_deci_s_layer_kinds_is_visible_in_the_logits() {
    let as_shapes = |d: &frink_models::Decoder| match &d.config.layer_shapes {
        LayerShapes::PerLayer(v) => v.clone(),
        LayerShapes::Uniform => panic!("deci is per-layer"),
    };
    // 1. The wo-only layer treated as attention-free.
    let mut d = load_graph_fixture(DECI);
    let mut shapes = as_shapes(&d);
    shapes[2].attention = AttnShape::Absent;
    d.config.layer_shapes = LayerShapes::PerLayer(shapes);
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &DECI_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "dropping the wo-only branch moved the output by only {worst}; blk.2's wo must be \
         too small to see"
    );
    // 2. The FFN-only layer's FFN skipped.
    let mut d = load_graph_fixture(DECI);
    let mut shapes = as_shapes(&d);
    shapes[3].ffn_dim = 0;
    d.config.layer_shapes = LayerShapes::PerLayer(shapes);
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &DECI_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "skipping blk.3's FFN moved the output by only {worst}"
    );
}

/// The FFN-free layer WITH attention is refused by name, from a file
/// libllama loads and runs.
///
/// `deci.cpp:147-149` `continue`s before the residual add at
/// `:150-153`, so the attention output is discarded. MEASURED: scaling
/// `blk.1.attn_*` by 3 in this file leaves libllama's 48 logits
/// byte-identical. frink does not reproduce a dead branch as the
/// reference; the refusal names the line.
#[test]
fn an_ffn_free_layer_with_attention_is_refused_naming_the_dropped_branch() {
    let file =
        frink_gguf::GgufFile::open(graph_fixture_path("deci_attn_ffnfree")).expect("fixture opens");
    let err = ModelConfig::from_gguf(&file).expect_err("refused at the header");
    let msg = format!("{err}");
    assert!(msg.contains("deci.cpp:147-149"), "{msg}");
    assert!(msg.contains("blk.1"), "{msg}");
}

/// The per-layer paged store, sized like the contiguous caches, and
/// the paged decode path matching the contiguous one on a model whose
/// layers differ.
#[test]
fn the_paged_store_is_sized_per_layer_and_paged_decode_matches_contiguous() {
    let d = load_graph_fixture(DECI_KV_ONLY);
    let store = std::sync::Arc::new(d.config.new_paged_kv(4, 8));
    assert_eq!(store.layer_count(), 3);
    let mut paged: Vec<frink_core::cache::PagedKvCache> = (0..3)
        .map(|_| frink_core::cache::PagedKvCache::new())
        .collect();
    let mut contiguous = d.config.new_kv_caches();
    let mut want = Vec::new();
    let mut got = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        want = d.forward_token(tok, pos, &mut contiguous);
        got = d
            .forward_token_paged(tok, pos, &mut paged, &store)
            .expect("8 blocks of 4 hold 6 positions");
    }
    assert_eq!(
        got, want,
        "paged and contiguous decode must be bit-identical"
    );
    common::assert_close(
        &got,
        &DECI_KV_ONLY_GOLDEN,
        common::GRAPH_TOL,
        "paged decode",
    );
}
