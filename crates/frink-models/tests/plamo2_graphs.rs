//! PLaMo-2 (`plamo2`), checked against llama.cpp itself: the first row
//! on `crate::plamo2_ssm` / `layer_shapes::AttnShape::Plamo2Ssm`, and
//! the first with `QkNormStyle::PerHeadDistinct`.
//!
//! `plamo2.cpp:19` marks layer `il` recurrent when `n_head_kv(il) == 0`
//! (the converter's arrays, `conversion/plamo.py:72-95`), and
//! `:132-183` runs one residual topology for both kinds: `attn_norm`,
//! PLaMo-2's SSM block (`:218-343`) or attention (`:190-216`: a fused
//! `attn_qkv`, the per-head QK RMSNorm with a distinct row per head,
//! NEOX RoPE, `kq_scale = 1/sqrt(v_dim)`), `attn_post_norm`, the
//! residual add, `ffn_norm`, the Phi-3 fused SwiGLU FFN,
//! `ffn_post_norm`, the residual add.
//!
//! The SSM block is Mamba-1's dt / B / C path (one projection of the
//! conv output, REQUIRED weighted norms) in the order B, C, dt, feeding
//! Mamba-2's per-head scan (dt, A, D per head, `head_dim = d_inner /
//! n_heads`), with z and x interleaved per head in `ssm_in`'s output
//! and no conv bias; `dt_dim = max(64, n_embd / 16)` is a literal of
//! the graph. Neither `crate::mamba1` nor `crate::mamba2` spells it.
//!
//! # RoPE, and the one place frink and libllama part
//!
//! The HF model rotates q and k over `qk_dim` after the QK norm
//! (`modeling_plamo.py`, `_rotary_pos_emb`), and `plamo2.cpp:239,245`
//! call `ggml_rope_ext` for it. But `llama-model.cpp:1189-1201` seed
//! `n_rot` from `n_head()` -- layer 0's head count -- and set it to 0
//! when that is 0, and the current converter writes `head_count` as an
//! ARRAY with 0 on every SSM layer (`conversion/plamo.py:87-88,95`),
//! layer 0 included. So libllama runs every current PLaMo-2 export with
//! `n_rot = 0` and rotates nothing (`print_info: n_rot = 0`, measured on
//! `plamo2_tiny`); `rope.dimension_count` cannot restore it, because that
//! read sits inside the same `n_head() > 0` branch. frink rotates, as
//! the model does and as the graph's authors wrote.
//!
//! So the golden is taken from the file where libllama rotates too:
//! `plamo2_scalar_heads_tiny`, the same weights with `head_count 4` as
//! a scalar and the KV array alone marking the SSM layers, which is an
//! equally valid llama.cpp file (`plamo2.cpp:19` reads only
//! `n_head_kv(i)`; `print_info: n_rot = 8`). frink must match it on
//! BOTH spellings. libllama's own logits for the array file are kept as
//! `LIBLLAMA_UNROTATED` and the test measures the distance, so the
//! deviation is a number and not a sentence.
//!
//! # Where the numbers come from
//!
//! The goldens were produced by running llama.cpp's own `plamo2` graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `plamo2_scalar_heads` (rotated) | see `report_kl_against_llama_cpp` | |
//! | `plamo2` (converter's arrays; libllama unrotated) | frink equals the rotated golden | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_plamo2_fixture.py \
//!     crates/frink-models/tests/fixtures/plamo2_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_plamo2_fixture.py \
//!     crates/frink-models/tests/fixtures/plamo2_scalar_heads_tiny.gguf --scalar-heads
//! /tmp/ref_logits crates/frink-models/tests/fixtures/plamo2_scalar_heads_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, kl_vs_golden, load_graph_fixture, worst_vs,
    GRAPH_PROMPT,
};
use frink_models::capability::{resolve_architecture, ArchPath, QkNormStyle};
use frink_models::config::RopeLayout;
use frink_models::layer_shapes::AttnShape;
use frink_models::rope_layers::RopeLayers;
use frink_models::Decoder;

const P2: &str = "plamo2";
const P2_SCALAR: &str = "plamo2_scalar_heads";

/// libllama over `plamo2_scalar_heads_tiny`: the rotated graph.
const P2_GOLDEN: [f32; 48] = [
    -1.0006586,
    -1.1626514,
    -0.1904155,
    0.2997827,
    0.3692283,
    0.33434415,
    0.42244124,
    0.2560249,
    -0.97931224,
    1.5760844,
    0.12655342,
    -0.6255829,
    -0.72727245,
    2.2330797,
    -2.2907815,
    0.6144577,
    1.1690793,
    1.207073,
    -0.5873551,
    0.89694905,
    0.64149916,
    0.5511484,
    -2.405486,
    2.1947997,
    -1.1409345,
    0.25560552,
    0.3137213,
    2.6148946,
    0.53161705,
    -0.25475848,
    1.7794983,
    1.9854268,
    2.12962,
    -1.5013224,
    -0.12669575,
    -0.3007954,
    -1.4708554,
    -1.9452466,
    -0.084961295,
    0.16923985,
    -1.43385,
    -0.9393533,
    0.08250879,
    0.07174915,
    1.127681,
    0.90926534,
    -3.7148025,
    -2.186347,
];

/// libllama over `plamo2_tiny` (the converter's arrays): `n_rot = 0`,
/// nothing rotated. Not a golden; the measurement of the deviation.
const LIBLLAMA_UNROTATED: [f32; 48] = [
    -0.033088923,
    -0.85981125,
    -1.0091588,
    0.5213228,
    0.69448304,
    0.050588787,
    0.9265797,
    -0.46792933,
    -2.3142657,
    1.6075093,
    0.25836045,
    0.053070873,
    0.03523445,
    1.5113193,
    -1.0694735,
    1.0769756,
    1.9119565,
    1.7774587,
    -0.34719983,
    0.37585226,
    0.23796377,
    0.3248368,
    -2.665509,
    1.3744824,
    -1.0871066,
    0.45666265,
    -0.19539738,
    2.3807333,
    0.09459239,
    -0.6893137,
    1.753377,
    2.5276866,
    1.2184323,
    -1.3524175,
    -0.34082907,
    -0.57303816,
    -1.5252805,
    -0.90970737,
    -0.49180174,
    1.2461139,
    -1.5758035,
    -0.54574144,
    -0.15837656,
    0.004574001,
    0.8401049,
    0.9666215,
    -3.8702016,
    -2.6500757,
];

fn decode(decoder: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    out
}

/// The scalar-heads file, where libllama rotates: frink matches on
/// every path.
#[test]
fn plamo2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(P2_SCALAR, &P2_GOLDEN);
}

/// The converter's spelling: frink answers the SAME rotated logits
/// (the head-count spelling changes nothing here), and libllama's own
/// answer for this file is the unrotated graph, measurably elsewhere.
#[test]
fn the_converter_s_array_spelling_is_rotated_here_and_not_in_libllama() {
    assert_all_three_paths_match(P2, &P2_GOLDEN);
    let gap = worst_vs(&LIBLLAMA_UNROTATED, &P2_GOLDEN);
    assert!(gap > 1e-1, "libllama's n_rot = 0 must be visible: {gap}");
}

#[test]
fn report_kl_against_llama_cpp() {
    for name in [P2_SCALAR, P2] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp rotated || frink) = {:.3e}, max |delta| = {:.3e}; \
             vs libllama unrotated max |delta| = {:.3e}",
            kl_vs_golden(&out, &P2_GOLDEN),
            worst_vs(&out, &P2_GOLDEN),
            worst_vs(&out, &LIBLLAMA_UNROTATED)
        );
    }
}

/// What the loader built: PLaMo-2's block where the array says 0, GQA
/// where it says 2, the block's widths from the `ssm.*` hparams with
/// `dt_dim` from the literal, the per-head-distinct QK norm, NEOX
/// rotation on every attention layer.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture(P2),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(P2);
    assert_eq!(d.config.rope_layers, RopeLayers::All);
    assert_eq!(d.config.qk_norm_style, QkNormStyle::PerHeadDistinct);
    for (il, layer) in d.layers.iter().enumerate() {
        let shape = d.config.layer_shape(il).attention;
        if il % 2 == 1 {
            assert_eq!(
                shape,
                AttnShape::Gqa {
                    n_heads: 4,
                    n_kv_heads: 2
                }
            );
            assert!(layer.attn.ssm.is_none());
            // One row per head: 4 x 8 for Q, 2 x 8 for K.
            assert_eq!(layer.attn.q_norm.as_ref().map(Vec::len), Some(32));
            assert_eq!(layer.attn.k_norm.as_ref().map(Vec::len), Some(16));
        } else {
            assert_eq!(shape, AttnShape::Plamo2Ssm, "blk.{il}");
            let m = match layer.attn.ssm.as_ref() {
                Some(frink_models::ssm_block::SsmBlock::Plamo2(m)) => m,
                _ => panic!("blk.{il}: PLaMo-2 block"),
            };
            assert_eq!(
                (
                    m.h.d_conv,
                    m.h.d_inner,
                    m.h.d_state,
                    m.h.n_heads,
                    m.h.dt_dim
                ),
                (4, 24, 8, 4, 64)
            );
            assert_eq!(layer.attn.q_proj.rows(), 0, "no Q on an SSM layer");
            assert_eq!(d.config.layer_cache_geometry(il), (0, 8, 8));
        }
        assert!(layer.attn.post_attn_norm.is_some() && layer.attn.post_ffn_norm.is_some());
    }
}

/// The state rides on the layer's cache and every layer's cache counts
/// positions whether or not it holds rows.
#[test]
fn the_recurrent_state_lives_on_the_cache_and_the_cache_counts_positions() {
    let d = load_graph_fixture(P2);
    let mut kv = graph_caches(&d);
    assert!(kv[0].recurrent.is_none());
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        d.forward_token(tok, pos, &mut kv);
    }
    let state = kv[0]
        .recurrent
        .as_ref()
        .expect("state after the first token");
    assert_eq!((state.conv.len(), state.ssm.len()), (3 * 24, 24 * 8));
    for (il, c) in kv.iter().enumerate() {
        assert_eq!(
            c.positions(),
            GRAPH_PROMPT.len(),
            "blk.{il} counts positions"
        );
    }
    assert_eq!(kv[0].rows(), 0, "an SSM layer holds no rows");
    assert_eq!(kv[1].rows(), GRAPH_PROMPT.len());
}
