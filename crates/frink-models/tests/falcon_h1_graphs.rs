//! Falcon-H1 (`falcon-h1`), checked against llama.cpp itself: attention
//! AND the Mamba-2 block on every layer, in parallel
//! (`crate::mamba2::PARALLEL_WITH_ATTENTION`, `ModelConfig::parallel_ssm`).
//!
//! `falcon-h1.cpp:137-161`: `attn_norm(x)` feeds both the rotated GQA
//! (`:141-154`) and `build_mamba2_layer` (`:158`), the two outputs are
//! summed (`:160`) and added to the residual once; then `ffn_norm` and
//! SwiGLU (`:167-174`). `:12` marks every layer recurrent, so the
//! layer's cache holds the attention rows AND the block's state.
//! `ssm_norm` is optional (`:70`); `attn_output.bias` is created and
//! never read (`:76`, `:154` passes NULL); `ffn_norm` is looked up as
//! `blk.N.ffn_norm` with no `.weight` (`:80`, the two-argument `LLM_TN`
//! overload) and libllama refuses the `.weight` spelling (measured:
//! `check_tensor_dims: tensor 'blk.0.ffn_norm' not found`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `falcon_h1` | see `report_kl_against_llama_cpp` | |
//! | `falcon_h1_nossmnorm` | (no `ssm_norm`) | |
//! | `falcon_h1_output` | (a separate `output.weight`) | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_falcon_h1_fixture.py \
//!     crates/frink-models/tests/fixtures/falcon_h1_tiny.gguf [--no-ssm-norm | --output]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/falcon_h1_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::RopeLayout;
use frink_models::layer_shapes::AttnShape;
use frink_models::Decoder;

const FH1: &str = "falcon_h1";
const FH1_NOSSMNORM: &str = "falcon_h1_nossmnorm";
const FH1_OUTPUT: &str = "falcon_h1_output";

const FH1_GOLDEN: [f32; 48] = [
    -0.6961461,
    1.3732871,
    0.062281482,
    1.3522964,
    -2.6813776,
    1.1528902,
    -0.62408954,
    2.1664424,
    -0.75196624,
    -0.9272255,
    -1.0419033,
    -0.5210315,
    -0.69825053,
    -2.181869,
    0.5292149,
    -0.13669077,
    1.0152586,
    -1.2236295,
    -0.19279458,
    -1.8496673,
    -0.35045373,
    -0.5213676,
    -0.056952618,
    0.8210954,
    -0.19537,
    1.2204739,
    2.7426062,
    0.68990123,
    1.9693447,
    -1.1470805,
    -1.7101523,
    -2.3020487,
    -0.15774225,
    -0.5011668,
    0.15089671,
    -0.42031997,
    -0.4905998,
    -1.923659,
    -0.65335613,
    -0.2645383,
    -1.5037391,
    0.5134231,
    0.5425779,
    -0.8949485,
    0.6149525,
    -0.30624473,
    1.7502667,
    0.21057636,
];

const FH1_NOSSMNORM_GOLDEN: [f32; 48] = [
    -0.029400885,
    -1.0776794,
    -0.9209957,
    0.5705744,
    -0.90420496,
    -3.1367679,
    -0.94462025,
    -0.21992198,
    0.7948678,
    -0.3833546,
    1.2466096,
    -0.76363736,
    -0.6734562,
    -1.0492152,
    0.13165063,
    -0.036705837,
    -0.62297034,
    -0.15333119,
    0.5725062,
    -1.2870284,
    -1.3356755,
    -0.7822225,
    -0.38964638,
    -1.3800931,
    -0.034176186,
    -1.4869628,
    1.6559966,
    -2.2824945,
    0.74753743,
    2.041632,
    -0.87142533,
    0.6389051,
    -1.1869912,
    -1.1611768,
    -0.4079123,
    -0.83129084,
    1.9874183,
    -0.1284315,
    1.3079523,
    -0.7054998,
    -0.5166384,
    -0.19621998,
    0.10233645,
    -0.75574243,
    2.9510076,
    2.6155496,
    1.5750726,
    -1.2750314,
];

const FH1_OUTPUT_GOLDEN: [f32; 48] = [
    0.12301362,
    2.1793213,
    -1.1627039,
    0.28780615,
    2.4085543,
    0.0984466,
    0.9337989,
    3.6491437,
    -1.56977,
    -2.0033088,
    0.01800615,
    1.7672086,
    0.96936387,
    -0.86011386,
    0.77681184,
    -1.1097353,
    1.3985933,
    0.6592895,
    0.14593829,
    -0.49333313,
    0.7880093,
    -0.016235001,
    -2.3994904,
    0.42428726,
    0.25034353,
    -2.7805326,
    -0.85465217,
    -1.3769777,
    -0.63508916,
    -0.5908395,
    -0.18158385,
    -1.5296507,
    0.11134828,
    -0.098544806,
    -1.2337172,
    0.57606184,
    -0.0065384433,
    0.44019797,
    -2.5737553,
    -0.73988295,
    -0.57101315,
    -1.6633399,
    0.4195699,
    -2.3385599,
    1.3690768,
    -0.40200132,
    -1.1428993,
    -0.61459655,
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
fn falcon_h1_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(FH1, &FH1_GOLDEN);
}

/// On Metal every Falcon-H1 layer runs whole in the dense stack, the
/// attention and the Mamba-2 block side by side
/// (`TokenMixer::Parallel`, `decoder::metal_hybrid`), and the answer is
/// still llama.cpp's, with and without `ssm_norm`. The allocation check
/// is what says the stack ran: the host bodies would match too.
///
/// `cargo test -p frink-models --features metal --test falcon_h1_graphs -- --ignored`
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs Apple Metal GPU"]
fn metal_decode_runs_both_branches_in_the_dense_stack() {
    if frink_metal::gpu::probe().is_none() {
        eprintln!("skip: no Metal GPU detected");
        return;
    }
    common::pin_backend(common::Backend::Metal);
    frink_models::set_metal_greedy_argmax(false);
    for (name, golden) in [(FH1, &FH1_GOLDEN), (FH1_NOSSMNORM, &FH1_NOSSMNORM_GOLDEN)] {
        let d = load_graph_fixture(name);
        let got = decode(&d);
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

#[test]
fn without_ssm_norm_matches_llama_cpp() {
    assert_all_three_paths_match(FH1_NOSSMNORM, &FH1_NOSSMNORM_GOLDEN);
    let d = load_graph_fixture(FH1_NOSSMNORM);
    assert!(d.layers[0]
        .attn
        .ssm
        .as_ref()
        .unwrap()
        .mamba2()
        .unwrap()
        .norm
        .is_none());
}

#[test]
fn a_separate_output_weight_matches_llama_cpp() {
    assert_all_three_paths_match(FH1_OUTPUT, &FH1_OUTPUT_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (FH1, &FH1_GOLDEN),
        (FH1_NOSSMNORM, &FH1_NOSSMNORM_GOLDEN),
        (FH1_OUTPUT, &FH1_OUTPUT_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// Every layer is a GQA layer WITH the block: uniform shapes, the
/// block's weights beside the projections, the state beside the rows.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("falcon-h1"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(FH1);
    assert!(d.config.parallel_ssm && d.config.has_recurrent_layers());
    assert!(d.config.layer_shapes.is_uniform());
    for (il, layer) in d.layers.iter().enumerate() {
        assert!(matches!(
            d.config.layer_shape(il).attention,
            AttnShape::Gqa {
                n_heads: 4,
                n_kv_heads: 2
            }
        ));
        assert!(layer.attn.ssm.is_some(), "blk.{il} has the block");
        assert_eq!(layer.attn.q_proj.rows(), 24, "blk.{il} has attention");
    }
    let mut kv = graph_caches(&d);
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        d.forward_token(tok, pos, &mut kv);
    }
    for (il, c) in kv.iter().enumerate() {
        assert_eq!(c.rows(), GRAPH_PROMPT.len(), "blk.{il} attention rows");
        assert_eq!(c.positions(), GRAPH_PROMPT.len(), "blk.{il} counted once");
        assert!(c.recurrent.is_some(), "blk.{il} state");
    }
}

/// The paged backing agrees with the contiguous one, rows and state.
#[test]
fn paged_decode_matches_contiguous() {
    let d = load_graph_fixture(FH1);
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
    assert_eq!(got, want);
    assert!(worst_vs(&got, &FH1_GOLDEN) < GRAPH_TOL);
    assert_eq!(paged[0].recurrent, contiguous[0].recurrent);
}

/// Both branches are visible: zeroing `ssm_out` or `wo` each moves the
/// logits, and the sum is a sum (neither branch is dropped).
#[test]
fn both_parallel_branches_reach_the_residual() {
    let mut d = load_graph_fixture(FH1);
    assert_decoder_matches_on_all_three_paths(&d, &FH1_GOLDEN, GRAPH_TOL, "baseline");
    let zero = |rows: usize, cols: usize| {
        frink_core::weight_matrix::WeightMatrix::F32(frink_core::Tensor::new(
            vec![0.0; rows * cols],
            vec![rows, cols],
        ))
    };
    let m = d.layers[1].attn.ssm.as_mut().unwrap().mamba2_mut().unwrap();
    let (rows, cols) = (m.out_proj.rows(), m.out_proj.cols());
    let saved = std::mem::replace(&mut m.out_proj, zero(rows, cols));
    let worst = worst_vs(&decode(&d), &FH1_GOLDEN);
    assert!(worst > 1e-2, "the Mamba-2 branch not seen: {worst}");
    d.layers[1]
        .attn
        .ssm
        .as_mut()
        .unwrap()
        .mamba2_mut()
        .unwrap()
        .out_proj = saved;
    assert_decoder_matches_on_all_three_paths(&d, &FH1_GOLDEN, GRAPH_TOL, "restored");
    let (rows, cols) = (
        d.layers[1].attn.o_proj.rows(),
        d.layers[1].attn.o_proj.cols(),
    );
    let saved = std::mem::replace(&mut d.layers[1].attn.o_proj, zero(rows, cols));
    let worst = worst_vs(&decode(&d), &FH1_GOLDEN);
    assert!(worst > 1e-2, "the attention branch not seen: {worst}");
    d.layers[1].attn.o_proj = saved;
}
