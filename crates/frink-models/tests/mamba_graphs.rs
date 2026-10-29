//! Mamba-1, checked against llama.cpp itself: `jamba` (AI21 Jamba),
//! `mamba` (Mamba-130M to 2.8B, FalconMamba-7B) and, on the block
//! `granitehybrid` opened, pure `mamba2` (Mamba-Codestral-7B).
//!
//! `build_mamba_layer` (`mamba-base.cpp:4-148`, `crate::mamba1`) is the
//! selective scan with a per-STATE decay (`Decay::PerState`,
//! `ggml_ssm_scan`'s `src3->ne[0] != 1` arm), `dt`, `B` and `C` from one
//! projection of the conv output, an RMS norm on each when the file
//! carries the three weights (Jamba, `jamba.cpp:49,52-53`) or sets
//! `ssm.dt_b_c_rms` (FalconMamba, weightless), and `dt` projected up to
//! `d_inner` with a bias. Jamba runs it where `head_count_kv` is 0 and
//! RoPE-less attention elsewhere (`jamba.cpp:92-102`), with a dense or
//! MoE FFN per layer by whether the router tensor exists (`:89-101,152`;
//! `moe_interleave::DENSE_LAYER_BY_ROUTER_ABSENCE`). The pure models run
//! the block on every layer with no FFN (`mamba.cpp:73-88`;
//! `layer_shapes::PURE_RECURRENT`, head_dim 0).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `jamba` | see `report_kl_against_llama_cpp` | |
//! | `mamba` | | |
//! | `mamba_dtbcrms` | (`ssm.dt_b_c_rms`, the weightless norms) | |
//! | `mamba2` | (pure Mamba-2, a separate `output.weight`) | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mamba_fixture.py \
//!     crates/frink-models/tests/fixtures/jamba_tiny.gguf --arch jamba
//! /tmp/ref_logits crates/frink-models/tests/fixtures/jamba_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_all_three_paths_match_within,
    assert_decoder_matches_on_all_three_paths, graph_caches, kl_vs_golden, load_graph_fixture,
    worst_vs, GRAPH_PROMPT,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::layer_shapes::AttnShape;
use frink_models::mamba1::DtBcNorm;
use frink_models::rope_layers::RopeLayers;
use frink_models::Decoder;

const JAMBA: &str = "jamba";
const MAMBA: &str = "mamba";
const MAMBA_DTBCRMS: &str = "mamba_dtbcrms";
const MAMBA2: &str = "mamba2";

/// The Mamba-1 rows sit at 5e-6 to 1e-5 max delta against libllama
/// (KL 2e-12 to 7e-12): `d_inner` one-element "heads" each take their
/// own `exp` per state element (`Decay::PerState`), and the two
/// projections into and out of the scan accumulate in a different
/// order from ggml's. The `phimoe` / `orion` class, not the GELU
/// table's; the Mamba-2 rows stay at `GRAPH_TOL`.
const MAMBA1_TOL: f32 = 5e-5;

const JAMBA_GOLDEN: [f32; 48] = [
    -1.9663377,
    -0.9450677,
    0.018398374,
    0.012534678,
    -0.85945106,
    -0.17342913,
    -3.0206368,
    -1.0671022,
    -0.50083435,
    0.26782972,
    1.1286671,
    0.8811697,
    -0.7284383,
    -2.0960097,
    1.896606,
    -0.17723948,
    -0.33415544,
    0.8305094,
    0.066825986,
    -1.4256666,
    0.983124,
    0.2765033,
    -0.41844738,
    0.14667806,
    -0.3797737,
    -2.7263985,
    0.1123659,
    0.84684134,
    2.322513,
    -0.02341938,
    1.9303751,
    2.2754016,
    -0.8007066,
    0.92957854,
    -1.3109334,
    -1.3997297,
    -0.78418434,
    2.691886,
    0.31171322,
    0.6505099,
    -0.025861949,
    0.6779201,
    -0.97503513,
    -0.2859332,
    0.33320707,
    1.1956241,
    -0.008789077,
    2.665274,
];

const MAMBA_GOLDEN: [f32; 48] = [
    -0.1058676,
    -1.6378187,
    -2.029526,
    0.19989784,
    -2.4548154,
    1.9478612,
    1.307256,
    1.0701766,
    0.790402,
    -2.1815124,
    -1.6252074,
    -1.2159412,
    0.37233356,
    0.93621504,
    1.8565391,
    -2.4600816,
    1.0896114,
    0.28502992,
    -0.8815542,
    1.7243545,
    -1.0134375,
    0.9465151,
    0.7057977,
    -2.9073987,
    1.5885276,
    2.2003741,
    1.629475,
    0.2598338,
    1.874522,
    0.957419,
    2.2595205,
    -0.27894062,
    -2.400549,
    -2.7466054,
    -2.0931337,
    -0.80348605,
    0.93170685,
    2.4356923,
    0.71080893,
    -0.4748994,
    1.545821,
    1.0759708,
    0.27163735,
    1.2275194,
    0.99340826,
    1.1393255,
    1.490738,
    0.57912,
];

const MAMBA_DTBCRMS_GOLDEN: [f32; 48] = [
    -1.822084,
    0.12616134,
    -1.5878246,
    1.0719968,
    -0.7503692,
    2.4272466,
    -0.3197248,
    -0.18239248,
    -0.01762706,
    1.1021817,
    -1.0284183,
    0.20069918,
    -0.2572071,
    -0.80553794,
    2.3627486,
    -0.27430183,
    0.20085907,
    0.64136773,
    -0.2781433,
    0.58703625,
    0.020504743,
    3.251829,
    -1.107492,
    -1.1921735,
    1.3588881,
    -1.2033026,
    1.2715935,
    -0.222476,
    -0.01210846,
    -0.78934515,
    0.018862069,
    1.0044863,
    0.08810702,
    -0.88932884,
    0.7816452,
    0.7330321,
    -0.65727293,
    -0.46255445,
    -1.1590393,
    -0.35524815,
    1.3320609,
    0.72828,
    0.39567497,
    -1.175176,
    -0.4190097,
    -2.1000333,
    2.062297,
    -0.49739805,
];

const MAMBA2_GOLDEN: [f32; 48] = [
    -0.36720425,
    0.009850249,
    -0.8157603,
    1.4756937,
    -2.3100436,
    0.4652757,
    0.088963255,
    -1.3007338,
    0.4541139,
    -0.85279083,
    -0.6064899,
    -0.69239104,
    -1.0280735,
    -0.06524229,
    -1.9125013,
    -0.6425878,
    -0.13044384,
    0.038234226,
    -0.37951505,
    2.3439486,
    -0.7443007,
    -0.5721311,
    -0.06523728,
    -1.5632281,
    -0.8956878,
    0.16050434,
    -2.8503008,
    1.6673676,
    1.7402252,
    0.6817362,
    0.24076185,
    2.2760746,
    0.690343,
    1.3614969,
    -1.4577657,
    0.2430207,
    0.027143046,
    1.3331959,
    -0.09516125,
    2.2477865,
    1.0578831,
    1.29111,
    -0.35891712,
    -0.6394238,
    0.9700943,
    0.44679406,
    2.0071948,
    0.27376857,
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
fn jamba_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(JAMBA, &JAMBA_GOLDEN, MAMBA1_TOL);
}

#[test]
fn mamba_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(MAMBA, &MAMBA_GOLDEN, MAMBA1_TOL);
}

/// `ssm.dt_b_c_rms` (FalconMamba): the weightless norms on dt, B and
/// C, whose golden differs from the plain file's.
#[test]
fn the_weightless_dt_b_c_norms_match_llama_cpp() {
    assert_all_three_paths_match_within(MAMBA_DTBCRMS, &MAMBA_DTBCRMS_GOLDEN, MAMBA1_TOL);
    assert!(worst_vs(&MAMBA_DTBCRMS_GOLDEN, &MAMBA_GOLDEN) > 1e-2);
    let d = load_graph_fixture(MAMBA_DTBCRMS);
    let m = d.layers[0].attn.ssm.as_ref().unwrap().mamba1().unwrap();
    assert!(matches!(m.dt_bc_norm, DtBcNorm::Weightless));
}

#[test]
fn pure_mamba2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(MAMBA2, &MAMBA2_GOLDEN);
    let d = load_graph_fixture(MAMBA2);
    for il in 0..d.config.n_layers {
        assert_eq!(d.config.layer_shape(il).attention, AttnShape::Mamba2);
    }
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (JAMBA, &JAMBA_GOLDEN),
        (MAMBA, &MAMBA_GOLDEN),
        (MAMBA_DTBCRMS, &MAMBA_DTBCRMS_GOLDEN),
        (MAMBA2, &MAMBA2_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// Jamba as loaded: Mamba-1 where the array says 0 with the three norms,
/// RoPE-less attention on layer 1, MoE on layers 0 and 3 and dense on
/// 1 and 2 by the router's presence, no top-k renormalisation.
#[test]
fn the_loaded_jamba_is_the_graph() {
    for arch in ["jamba", "mamba", "mamba2"] {
        assert!(
            matches!(
                resolve_architecture(arch),
                Some(ArchPath::GenericGqa { .. })
            ),
            "{arch}"
        );
    }
    let d = load_graph_fixture(JAMBA);
    assert_eq!(d.config.rope_layers, RopeLayers::Never);
    assert!(!d.config.moe.norm_topk_prob);
    assert_eq!(d.config.moe.expert_weights_scale, 1.0);
    for il in [0, 2, 3] {
        assert_eq!(
            d.config.layer_shape(il).attention,
            AttnShape::Mamba1,
            "blk.{il}"
        );
        let m = d.layers[il].attn.ssm.as_ref().unwrap().mamba1().unwrap();
        assert!(
            matches!(m.dt_bc_norm, DtBcNorm::Weighted { .. }),
            "blk.{il}"
        );
        assert_eq!(
            (m.h.d_conv, m.h.d_inner, m.h.d_state, m.h.dt_rank),
            (4, 48, 8, 3)
        );
    }
    assert!(matches!(
        d.config.layer_shape(1).attention,
        AttnShape::Gqa {
            n_heads: 4,
            n_kv_heads: 2
        }
    ));
    for (il, moe) in [(0, true), (1, false), (2, false), (3, true)] {
        assert_eq!(d.layers[il].moe.n_experts() > 1, moe, "blk.{il} routes");
    }
}

/// The pure models: no heads, no head width, no FFN, every layer the
/// block, every cache counting positions with no rows.
#[test]
fn the_pure_models_have_no_attention_anywhere() {
    for (name, shape) in [(MAMBA, AttnShape::Mamba1), (MAMBA2, AttnShape::Mamba2)] {
        let d = load_graph_fixture(name);
        assert_eq!(d.config.head_dim, 0, "{name}");
        assert!(d.config.has_recurrent_layers());
        for il in 0..d.config.n_layers {
            let s = d.config.layer_shape(il);
            assert_eq!((s.attention, s.ffn_dim), (shape, 0), "{name} blk.{il}");
        }
        let mut kv = graph_caches(&d);
        for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
            d.forward_token(tok, pos, &mut kv);
        }
        for c in &kv {
            assert_eq!((c.rows(), c.positions()), (0, GRAPH_PROMPT.len()));
            assert!(c.recurrent.is_some());
        }
    }
}

/// The paged backing agrees, rows and state, for the hybrid and a pure
/// model.
#[test]
fn paged_decode_matches_contiguous() {
    for (name, golden) in [(JAMBA, &JAMBA_GOLDEN[..]), (MAMBA, &MAMBA_GOLDEN[..])] {
        let d = load_graph_fixture(name);
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
        assert_eq!(got, want, "{name}");
        assert!(worst_vs(&got, golden) < MAMBA1_TOL, "{name}");
    }
}

/// The Mamba-1 pieces are visible: the per-state decay (A), the D skip,
/// the dt/B/C norms.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(JAMBA);
    assert_decoder_matches_on_all_three_paths(&d, &JAMBA_GOLDEN, MAMBA1_TOL, "baseline");
    let m = d.layers[0].attn.ssm.as_mut().unwrap();
    let frink_models::ssm_block::SsmBlock::Mamba1(m) = m else {
        panic!("blk.0 is Mamba-1");
    };
    let saved = std::mem::replace(&mut m.dt_bc_norm, DtBcNorm::None);
    let worst = worst_vs(&decode(&d), &JAMBA_GOLDEN);
    assert!(worst > 1e-2, "the dt/B/C norms not seen: {worst}");
    let frink_models::ssm_block::SsmBlock::Mamba1(m) = d.layers[0].attn.ssm.as_mut().unwrap()
    else {
        unreachable!()
    };
    m.dt_bc_norm = saved;
    assert_decoder_matches_on_all_three_paths(&d, &JAMBA_GOLDEN, MAMBA1_TOL, "restored");
    let frink_models::ssm_block::SsmBlock::Mamba1(m) = d.layers[0].attn.ssm.as_mut().unwrap()
    else {
        unreachable!()
    };
    for a in m.a.iter_mut() {
        *a = -30.0;
    }
    let worst = worst_vs(&decode(&d), &JAMBA_GOLDEN);
    assert!(worst > 1e-2, "the SSM state not seen: {worst}");
}
