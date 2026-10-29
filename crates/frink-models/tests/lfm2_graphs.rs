//! LFM2, checked against llama.cpp itself: the first HYBRID row on the
//! generic path, on the seam `crate::shortconv` /
//! `layer_shapes::AttnShape::ShortConv`.
//!
//! `lfm2.cpp:9-11` marks layer `il` recurrent when `n_head_kv(il) == 0`
//! (the converter's array, `conversion/lfm2.py:37-40`), and `:192-208`
//! is ONE residual topology for both kinds: `attn_norm`, the short
//! convolution (`:139-189`) or GQA (`:110-137`; per-head RMS QK norm
//! `{n_embd_head_k}` at `:74-75`, NEOX RoPE, `wo` `{n_embd, n_embd}`),
//! the residual add, `ffn_norm`, SwiGLU. The final norm is stored as
//! `token_embd_norm.weight` (`LLM_TENSOR_OUTPUT_NORM_LFM2`,
//! llama-arch.cpp:384) and applied at `:212`; `output` is tied when
//! absent (`:37-41`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `lfm2` | see `report_kl_against_llama_cpp` | |
//! | `lfm2_fused` | (the converter's fused `attn_qkv`; libllama byte-identical to `lfm2`) | |
//! | `lfm2_output` | (a separate `output.weight`) | |
//! | `lfm2moe` | (`lfm2moe`: one leading dense layer, a sigmoid MoE with `exp_probs_b` on the rest) | |
//!
//! `lfm2_window` declares `attention.sliding_window 4`; libllama runs it
//! (its logits differ from `lfm2`'s) and frink refuses it by name.
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_lfm2_fixture.py \
//!     crates/frink-models/tests/fixtures/lfm2_tiny.gguf [--fused-qkv | --output | --window]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/lfm2_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    graph_fixture_path, kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath, QkNormStyle};
use frink_models::config::RopeLayout;
use frink_models::layer_shapes::AttnShape;
use frink_models::{Decoder, LoadError, ModelConfig};
use frink_moe::GatingFunction;

const LFM2: &str = "lfm2";
const LFM2_FUSED: &str = "lfm2_fused";
const LFM2_OUTPUT: &str = "lfm2_output";
const LFM2_WINDOW: &str = "lfm2_window";
const LFM2MOE: &str = "lfm2moe";

const LFM2_GOLDEN: [f32; 48] = [
    -0.7957208,
    -1.0949144,
    1.1510789,
    -0.7174512,
    0.6061578,
    2.035816,
    -1.0446837,
    -0.51440835,
    0.106185436,
    1.6726716,
    -1.1166134,
    -1.3961111,
    0.21587065,
    -0.7937374,
    -1.1573095,
    1.6552248,
    -0.61917067,
    0.1134741,
    -2.1890545,
    -0.13979566,
    1.1174879,
    -1.2235979,
    1.3523095,
    0.8779149,
    -0.81434476,
    0.21026042,
    -2.3469431,
    0.6506362,
    -1.873858,
    0.27796865,
    0.2762602,
    1.3069516,
    1.211674,
    0.46970546,
    -0.16166127,
    -0.79020214,
    -0.5541268,
    -0.98638,
    2.2717872,
    0.7920846,
    -0.7579817,
    2.543402,
    -1.1688484,
    -1.9257866,
    -1.478356,
    -1.1687315,
    0.34882605,
    0.6873579,
];

const LFM2_OUTPUT_GOLDEN: [f32; 48] = [
    0.24992129,
    -0.52726275,
    -0.35745943,
    -0.52423596,
    0.46511424,
    3.6101003,
    0.9844166,
    -0.39169163,
    -0.01883617,
    -1.7551847,
    1.1285117,
    0.53508604,
    0.19035327,
    0.40200436,
    1.0861784,
    0.82165813,
    0.32673264,
    -2.511534,
    0.48494774,
    -0.4221602,
    -0.85943276,
    -2.9800258,
    -2.172226,
    2.8774548,
    -0.60309225,
    -1.4671144,
    -0.28519714,
    1.2668669,
    -0.9228035,
    1.8385018,
    -1.3193693,
    1.4697948,
    -1.4408727,
    2.7617917,
    0.6703675,
    1.1152695,
    -0.7149642,
    -1.9254955,
    -0.7090295,
    3.5120373,
    0.26838714,
    1.7216109,
    -0.07181811,
    -1.2953314,
    1.9240181,
    0.2651766,
    -0.60123825,
    -1.464062,
];

const LFM2MOE_GOLDEN: [f32; 48] = [
    1.9647387,
    -1.4767911,
    -0.9754009,
    2.744078,
    0.6172535,
    2.8822598,
    0.95489186,
    0.33114186,
    0.89591736,
    1.7829331,
    0.80464506,
    0.019917876,
    0.19850309,
    -0.91026944,
    -0.7553232,
    2.294318,
    -1.7630544,
    -2.0266988,
    -0.3181442,
    -1.4291439,
    -0.6683065,
    -1.9095902,
    0.32190263,
    2.3132637,
    0.4962061,
    -1.968719,
    -0.119931296,
    2.4317455,
    -2.0145152,
    0.17097563,
    -1.2232091,
    0.9582149,
    3.6602554,
    -0.07174578,
    0.04873529,
    -0.5801943,
    0.14411034,
    -2.5924613,
    1.4491975,
    -1.0441153,
    -0.6576078,
    0.81238157,
    1.4386629,
    -2.3015327,
    0.12230853,
    -1.5466526,
    0.043979853,
    -0.018183798,
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
fn lfm2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(LFM2, &LFM2_GOLDEN);
}

/// `create_tensor_qkv` (`lfm2.cpp:78`) takes the fused spelling too;
/// libllama's logits for the two files are byte-identical.
#[test]
fn the_fused_qkv_spelling_matches_the_same_golden() {
    assert_all_three_paths_match(LFM2_FUSED, &LFM2_GOLDEN);
}

#[test]
fn a_separate_output_weight_matches_llama_cpp() {
    assert_all_three_paths_match(LFM2_OUTPUT, &LFM2_OUTPUT_GOLDEN);
}

/// `lfm2moe` is the same graph (`models.h:1899`): `leading_dense_block_
/// count` dense layers, then a sigmoid MoE with the REQUIRED router
/// bias (`lfm2moe.cpp:8,38-47`), `norm_w = true` (lfm2.cpp:118). Its
/// hparams read no `expert_weights_scale`; the fixture declares 2.5 and
/// libllama's golden is unscaled.
#[test]
fn lfm2moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(LFM2MOE, &LFM2MOE_GOLDEN);
    let d = load_graph_fixture(LFM2MOE);
    assert!(matches!(
        resolve_architecture("lfm2moe"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    assert_eq!(d.config.moe.gating, GatingFunction::Sigmoid);
    assert!(d.config.moe.norm_topk_prob);
    assert_eq!(d.config.moe.n_experts, 4);
    assert!(d.config.layer_is_dense(0) && !d.config.layer_is_dense(1));
    assert!(d.layers[0].moe.exp_probs_bias.is_none(), "blk.0 is dense");
    for il in 1..4 {
        assert_eq!(d.layers[il].moe.router.rows(), 4, "blk.{il} routes");
        assert!(
            d.layers[il].moe.exp_probs_bias.is_some(),
            "blk.{il} exp_probs_b"
        );
    }
    assert_eq!(d.config.layer_shape(0).attention, AttnShape::ShortConv);
    assert_eq!(d.config.layer_shape(2).attention, AttnShape::ShortConv);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (LFM2, &LFM2_GOLDEN),
        (LFM2_FUSED, &LFM2_GOLDEN),
        (LFM2_OUTPUT, &LFM2_OUTPUT_GOLDEN),
        (LFM2MOE, &LFM2MOE_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: conv layers where the array says 0, GQA
/// where it says 2, the per-head QK norm, the conv's width from the
/// key, the output norm from `token_embd_norm`, and a cache per conv
/// layer of one `n_embd`-wide row with no V.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("lfm2"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(LFM2);
    assert_eq!(d.config.qk_norm_style, QkNormStyle::PerHead);
    assert_eq!(d.config.n_layers, 4);
    for (il, layer) in d.layers.iter().enumerate() {
        let shape = d.config.layer_shape(il).attention;
        if il % 2 == 0 {
            assert_eq!(shape, AttnShape::ShortConv, "blk.{il}");
            let conv = layer.attn.shortconv.as_ref().expect("conv weights");
            assert_eq!(conv.l_cache, 3);
            assert_eq!(conv.conv.len(), 3 * d.config.hidden_dim);
            assert_eq!(layer.attn.q_proj.rows(), 0, "no Q on a conv layer");
            assert_eq!(
                d.config.layer_cache_geometry(il),
                (1, d.config.hidden_dim, 0)
            );
        } else {
            assert_eq!(
                shape,
                AttnShape::Gqa {
                    n_heads: 4,
                    n_kv_heads: 2
                },
                "blk.{il}"
            );
            assert!(layer.attn.shortconv.is_none());
            let q = layer.attn.q_norm.as_ref().expect("attn_q_norm");
            assert_eq!(q.len(), d.config.head_dim, "per head");
            assert_eq!(d.config.layer_cache_geometry(il), (2, 6, 6));
        }
    }
    // The caches the config builds have the geometry the shapes name.
    let kv = graph_caches(&d);
    assert_eq!(kv[0].k_width(), d.config.hidden_dim);
    assert_eq!(kv[0].v_width(), 0);
    assert_eq!(kv[1].k_width(), 2 * d.config.head_dim);
}

/// The paged backing reads the conv window through its block table
/// (blocks of 4 with a conv of width 3: the window straddles a block
/// boundary at positions 4 and 5), and agrees with the contiguous one.
#[test]
fn paged_decode_matches_contiguous_across_a_block_boundary() {
    let d = load_graph_fixture(LFM2);
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
    assert_eq!(got, want, "paged and contiguous conv histories");
    assert!(worst_vs(&got, &LFM2_GOLDEN) < GRAPH_TOL);
    // The conv layer's store holds one n_embd row per position.
    assert_eq!(store.read(0).k_width(), d.config.hidden_dim);
    assert_eq!(paged[0].seq_len(), GRAPH_PROMPT.len());
}

/// Truncating the history and continuing is what keeping the whole
/// conv history buys (`crate::shortconv`): after a truncate the state
/// is exactly what the shorter prefix would have left.
#[test]
fn the_conv_history_truncates_like_a_kv_cache() {
    let d = load_graph_fixture(LFM2);
    let mut kv = graph_caches(&d);
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        d.forward_token(tok, pos, &mut kv);
    }
    for c in kv.iter_mut() {
        c.truncate(3);
    }
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate().skip(3) {
        out = d.forward_token(tok, pos, &mut kv);
    }
    assert!(worst_vs(&out, &LFM2_GOLDEN) < GRAPH_TOL);
}

/// Each piece of the conv block is visible: the newest input on the
/// LAST tap (reversing the taps diverges), the `c` gate, and the state
/// (a conv that saw only the current token diverges).
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(LFM2);
    assert_decoder_matches_on_all_three_paths(&d, &LFM2_GOLDEN, GRAPH_TOL, "baseline");

    // Reverse every conv layer's taps: oldest input on the last tap.
    for layer in d.layers.iter_mut() {
        if let Some(conv) = layer.attn.shortconv.as_mut() {
            let l = conv.l_cache;
            for taps in conv.conv.chunks_mut(l) {
                taps.reverse();
            }
        }
    }
    let worst = worst_vs(&decode(&d), &LFM2_GOLDEN);
    assert!(worst > 1e-2, "tap order not seen: {worst}");
    for layer in d.layers.iter_mut() {
        if let Some(conv) = layer.attn.shortconv.as_mut() {
            let l = conv.l_cache;
            for taps in conv.conv.chunks_mut(l) {
                taps.reverse();
            }
        }
    }
    assert_decoder_matches_on_all_three_paths(&d, &LFM2_GOLDEN, GRAPH_TOL, "restored");

    // Zero every tap but the last: a conv with no state.
    for layer in d.layers.iter_mut() {
        if let Some(conv) = layer.attn.shortconv.as_mut() {
            let l = conv.l_cache;
            for taps in conv.conv.chunks_mut(l) {
                for t in &mut taps[..l - 1] {
                    *t = 0.0;
                }
            }
        }
    }
    let worst = worst_vs(&decode(&d), &LFM2_GOLDEN);
    assert!(worst > 1e-2, "the conv state not seen: {worst}");
}

/// A whole-vector reading of the same QK-norm bytes would be a
/// different graph: the loader takes the per-head one.
#[test]
fn the_qk_norm_is_per_head_and_visible() {
    let mut d = load_graph_fixture(LFM2);
    for layer in d.layers.iter_mut() {
        layer.attn.q_norm = None;
        layer.attn.k_norm = None;
    }
    let worst = worst_vs(&decode(&d), &LFM2_GOLDEN);
    assert!(worst > 1e-2, "the QK norm not seen: {worst}");
}

/// `attention.sliding_window` on this architecture is refused by name,
/// from a file libllama runs (its logits differ from `lfm2`'s).
#[test]
fn a_window_is_refused_by_name() {
    let path = graph_fixture_path(LFM2_WINDOW);
    let file = frink_gguf::GgufFile::open(&path).expect("fixture opens");
    let err = match ModelConfig::from_gguf(&file) {
        Err(e) => e,
        Ok(config) => Decoder::from_gguf(&path, config)
            .err()
            .expect("a windowed LFM2 file is refused"),
    };
    let msg = format!("{err}");
    assert!(matches!(err, LoadError::UnsupportedFeature(..)), "{msg}");
    assert!(msg.contains("lfm2.cpp:24-29"), "{msg}");
}
