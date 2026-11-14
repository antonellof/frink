//! Granite 4.0 (`granitehybrid`), checked against llama.cpp itself: the
//! first Mamba-2 row, on the seam `crate::mamba2` /
//! `layer_shapes::AttnShape::Mamba2` / `frink_core::recurrent_state`.
//!
//! `granite-hybrid.cpp:17-19` marks layer `il` recurrent when
//! `n_head_kv(il) == 0` (the converter's array, `conversion/granite.py:
//! 238-241`), and `:128-142` runs one residual topology for both kinds:
//! `attn_norm`, the Mamba-2 block (`mamba-base.cpp:149-288`) or GQA,
//! the residual add scaled by `residual_scale`, `ffn_norm`, the FFN
//! (dense, or MoE with the shared expert). `granite.cpp`'s four
//! multipliers apply. `conversion/granite.py:253-256` writes
//! `rope.scaling.finetuned = false` for every export with a Mamba
//! layer, so the attention layers rotate NOTHING (`crate::rope_finetuned`,
//! `RopeLayers::Never`); the `rope` fixture writes it true and rotates
//! NORM (Bamba's shape).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `granitehybrid` | see `report_kl_against_llama_cpp` | |
//! | `granitehybrid_rope` | (`rope.scaling.finetuned = true`) | |
//! | `granitehybrid_moe` | (4 experts, 2 used, softmax, shared expert) | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_granite_hybrid_fixture.py \
//!     crates/frink-models/tests/fixtures/granitehybrid_tiny.gguf [--rope | --moe]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/granitehybrid_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::RopeLayout;
use frink_models::layer_shapes::AttnShape;
use frink_models::rope_layers::RopeLayers;
use frink_models::Decoder;

const GH: &str = "granitehybrid";
const GH_ROPE: &str = "granitehybrid_rope";
const GH_MOE: &str = "granitehybrid_moe";

const GH_GOLDEN: [f32; 48] = [
    -0.25681278,
    -0.7425126,
    0.055023838,
    0.56638616,
    0.28316763,
    0.90965146,
    0.207424,
    -0.52295524,
    0.71875566,
    -0.06772013,
    0.5483881,
    0.40476808,
    -0.004471147,
    0.3710489,
    0.05327775,
    0.34012553,
    0.0020415545,
    -1.1538728,
    -0.5242459,
    -0.97796583,
    -0.16708654,
    -0.31937185,
    -0.4802385,
    0.7637499,
    0.52803004,
    -0.032963533,
    0.2594169,
    -0.1396129,
    0.7563423,
    -0.55339426,
    0.52414215,
    0.46639648,
    0.3505067,
    0.08360209,
    0.6702198,
    0.3676821,
    0.35507542,
    -0.8444435,
    -0.014685819,
    0.16831775,
    0.26453003,
    -0.44684267,
    -0.22521739,
    -0.1599689,
    0.17626347,
    -1.0095729,
    -0.5152101,
    -0.2762197,
];

const GH_ROPE_GOLDEN: [f32; 48] = [
    -0.22439031,
    -1.0733272,
    -0.4444368,
    1.0288013,
    0.34474707,
    0.3526938,
    -0.08550482,
    0.043829333,
    0.6395511,
    0.035275493,
    1.0108322,
    0.79399425,
    0.23057933,
    0.31712085,
    0.21554446,
    -0.10060896,
    -0.18234585,
    -1.3074349,
    -0.94492435,
    -0.9807326,
    0.02823083,
    -0.5843713,
    -0.16710411,
    0.47284552,
    0.65301096,
    -0.23122919,
    -0.15178685,
    -0.55047864,
    0.6703891,
    -0.61529094,
    -0.20810619,
    0.2274874,
    0.3126152,
    -0.1420259,
    0.43855247,
    -0.044344623,
    0.7016823,
    -0.6196639,
    -0.44224796,
    -0.42513004,
    0.17362884,
    0.1770005,
    0.18358174,
    0.63645375,
    0.14492461,
    -1.1255493,
    -0.5234267,
    -0.788428,
];

const GH_MOE_GOLDEN: [f32; 48] = [
    -0.18473203,
    -0.34870657,
    -0.21441992,
    -0.6917295,
    0.36449954,
    0.20870903,
    -0.4458898,
    0.3586417,
    0.193199,
    -1.0552315,
    -0.08141168,
    0.86915344,
    -0.2507509,
    0.38949296,
    0.552716,
    0.029973585,
    -0.7303184,
    0.47012982,
    -0.12361397,
    -0.3283039,
    0.00041364506,
    0.22869079,
    0.051446974,
    1.1107305,
    -0.5654611,
    1.0669935,
    0.7347078,
    0.048057042,
    -0.49675155,
    0.09484286,
    0.022498166,
    -0.39624125,
    -0.6801243,
    0.7124078,
    -0.57517105,
    0.0071977796,
    -0.6457335,
    -0.020423269,
    -0.1810356,
    -0.12844257,
    0.09584587,
    0.45437524,
    -0.36622167,
    -0.18849795,
    0.15465286,
    0.021770503,
    -0.4443926,
    -0.04162198,
];

fn decode(decoder: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    out
}

#[test]
fn granitehybrid_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(GH, &GH_GOLDEN);
}

/// On Metal a Granite-4.0-H decode token runs whole in the dense stack,
/// its Mamba-2 layers as token mixers and its `residual_scale` and
/// `attention.scale` as the stack's own uniforms
/// (`frink_metal::attn::DenseLayerMetal::residual_scale`,
/// `AttnMixer::q_scale`), and the answer is still llama.cpp's, NoPE and
/// rotated. The allocation check says the stack ran: the host bodies
/// would match the golden too.
///
/// `cargo test -p frink-models --features metal --test granite_hybrid_graphs -- --ignored`
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs Apple Metal GPU"]
fn metal_decode_runs_granite_hybrid_in_the_dense_stack() {
    if frink_metal::gpu::probe().is_none() {
        eprintln!("skip: no Metal GPU detected");
        return;
    }
    common::pin_backend(common::Backend::Metal);
    frink_models::set_metal_greedy_argmax(false);
    for (name, golden) in [(GH, &GH_GOLDEN), (GH_ROPE, &GH_ROPE_GOLDEN)] {
        let d = load_graph_fixture(name);
        assert!(d.config.attention_scale.is_some() && d.config.residual_scale.is_some());
        let got = common::decode_with_resident_planes(&d, name);
        assert!(
            d.metal_hybrid_allocated(),
            "{name}: the hybrid stack never ran"
        );
        let worst = worst_vs(&got, golden);
        assert!(
            worst < common::METAL_GOLDEN_TOL,
            "{name}: worst |metal - llama.cpp| {worst:.3e}"
        );
    }
}

/// `rope.scaling.finetuned = true`: the attention layer rotates (NORM).
/// Its golden differs from the NoPE file's, so the switch is measured
/// and not assumed.
#[test]
fn the_rotated_variant_matches_llama_cpp() {
    assert_all_three_paths_match(GH_ROPE, &GH_ROPE_GOLDEN);
    assert!(worst_vs(&GH_ROPE_GOLDEN, &GH_GOLDEN) > 1e-2);
}

#[test]
fn the_moe_variant_with_the_shared_expert_matches_llama_cpp() {
    assert_all_three_paths_match(GH_MOE, &GH_MOE_GOLDEN);
    let d = load_graph_fixture(GH_MOE);
    assert_eq!(d.config.moe.n_experts, 4);
    assert_eq!(d.config.moe.n_shared_experts, 1);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (GH, &GH_GOLDEN),
        (GH_ROPE, &GH_ROPE_GOLDEN),
        (GH_MOE, &GH_MOE_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: Mamba-2 where the array says 0, GQA where it
/// says 2, the five `ssm.*` hparams on the weights, NO rotation on the
/// NoPE file and the four multipliers read.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("granitehybrid"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Norm
        })
    ));
    let d = load_graph_fixture(GH);
    assert_eq!(d.config.rope_layers, RopeLayers::Never, "NoPE attention");
    assert_eq!(
        load_graph_fixture(GH_ROPE).config.rope_layers,
        RopeLayers::All
    );
    assert!(d.config.residual_scale.is_some() && d.config.embedding_scale.is_some());
    for (il, layer) in d.layers.iter().enumerate() {
        let shape = d.config.layer_shape(il).attention;
        if il == 1 {
            assert_eq!(
                shape,
                AttnShape::Gqa {
                    n_heads: 4,
                    n_kv_heads: 2
                }
            );
            assert!(layer.attn.ssm.is_none());
        } else {
            assert_eq!(shape, AttnShape::Mamba2, "blk.{il}");
            let m = layer
                .attn
                .ssm
                .as_ref()
                .and_then(|b| b.mamba2())
                .expect("Mamba-2 weights");
            assert_eq!(
                (
                    m.h.d_conv,
                    m.h.d_inner,
                    m.h.d_state,
                    m.h.n_head,
                    m.h.n_group
                ),
                (4, 48, 8, 4, 2)
            );
            assert_eq!(layer.attn.q_proj.rows(), 0, "no Q on a Mamba layer");
            assert_eq!(d.config.layer_cache_geometry(il), (0, 6, 6));
        }
    }
}

/// The state rides on the layer's cache: absent before the first token,
/// sized by the weights after it, cloned with the cache, cleared with
/// it, and every layer's cache counts positions whether or not it holds
/// rows.
#[test]
fn the_recurrent_state_lives_on_the_cache_and_the_cache_counts_positions() {
    let d = load_graph_fixture(GH);
    let mut kv = graph_caches(&d);
    assert!(kv[0].recurrent.is_none());
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        d.forward_token(tok, pos, &mut kv);
    }
    let m = d.layers[0].attn.ssm.as_ref().unwrap().mamba2().unwrap();
    let (conv, ssm) = m.h.state_floats();
    let state = kv[0]
        .recurrent
        .as_ref()
        .expect("state after the first token");
    assert_eq!((state.conv.len(), state.ssm.len()), (conv, ssm));
    for (il, c) in kv.iter().enumerate() {
        assert_eq!(
            c.positions(),
            GRAPH_PROMPT.len(),
            "blk.{il} counts positions"
        );
    }
    assert_eq!(kv[0].rows(), 0, "a Mamba layer holds no rows");
    assert_eq!(kv[1].rows(), GRAPH_PROMPT.len());
    let fork = kv[0].clone();
    assert_eq!(fork.recurrent, kv[0].recurrent);
    assert!(kv[0].can_truncate_to(GRAPH_PROMPT.len()) && kv[0].can_truncate_to(0));
    assert!(!kv[0].can_truncate_to(3), "a reduction has no middle");
    kv[0].clear();
    assert!(kv[0].recurrent.is_none());
}

/// The paged backing carries the state on the per-sequence cache (not
/// in the store), through both the token and the batched paged paths.
#[test]
fn paged_decode_and_paged_prefill_match_contiguous() {
    let d = load_graph_fixture(GH);
    let store = std::sync::Arc::new(d.config.new_paged_kv(4, 8));
    let mut paged: Vec<frink_core::cache::PagedKvCache> = (0..d.config.n_layers)
        .map(|_| frink_core::cache::PagedKvCache::new())
        .collect();
    let mut contiguous = graph_caches(&d);
    let mut want = Vec::new();
    let mut got = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        want = d.forward_token(tok, pos, &mut contiguous);
        got = d
            .forward_token_paged(tok, pos, &mut paged, &store)
            .expect("8 blocks of 4 hold 6 positions");
    }
    assert_eq!(got, want, "paged and contiguous states");
    assert!(worst_vs(&got, &GH_GOLDEN) < GRAPH_TOL);
    assert_eq!(paged[0].seq_len(), GRAPH_PROMPT.len());
    assert!(paged[0].recurrent.is_some());

    // The batched paged prefill gathers, runs and scatters: the state
    // must come back with the rows.
    let mut paged2: Vec<frink_core::cache::PagedKvCache> = (0..d.config.n_layers)
        .map(|_| frink_core::cache::PagedKvCache::new())
        .collect();
    let logits = d
        .forward_batch_last_paged(&GRAPH_PROMPT, 0, &mut paged2, &store)
        .expect("fits");
    assert!(worst_vs(&logits, &GH_GOLDEN) < GRAPH_TOL);
    assert_eq!(
        paged2[0].recurrent, paged[0].recurrent,
        "the prefill's state is the decode's"
    );
}

/// The block's pieces are visible: the SSM decay (A), the skip term
/// (D), the conv state.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(GH);
    assert_decoder_matches_on_all_three_paths(&d, &GH_GOLDEN, GRAPH_TOL, "baseline");

    let saved: Vec<std::sync::Arc<[f32]>> = d
        .layers
        .iter()
        .filter_map(|l| {
            l.attn
                .ssm
                .as_ref()
                .and_then(|b| b.mamba2())
                .map(|m| m.a.clone())
        })
        .collect();
    for l in d.layers.iter_mut() {
        if let Some(m) = l.attn.ssm.as_mut().and_then(|b| b.mamba2_mut()) {
            for a in std::sync::Arc::make_mut(&mut m.a).iter_mut() {
                *a = -30.0; // no memory at all
            }
        }
    }
    let worst = worst_vs(&decode(&d), &GH_GOLDEN);
    assert!(worst > 1e-2, "the SSM state not seen: {worst}");
    let mut it = saved.into_iter();
    for l in d.layers.iter_mut() {
        if let Some(m) = l.attn.ssm.as_mut().and_then(|b| b.mamba2_mut()) {
            m.a = it.next().unwrap();
        }
    }
    assert_decoder_matches_on_all_three_paths(&d, &GH_GOLDEN, GRAPH_TOL, "restored");

    for l in d.layers.iter_mut() {
        if let Some(m) = l.attn.ssm.as_mut().and_then(|b| b.mamba2_mut()) {
            for v in std::sync::Arc::make_mut(&mut m.d).iter_mut() {
                *v = 0.0;
            }
        }
    }
    let worst = worst_vs(&decode(&d), &GH_GOLDEN);
    assert!(worst > 1e-2, "the D skip not seen: {worst}");
}
