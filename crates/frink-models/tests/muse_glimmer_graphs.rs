//! `muse-glimmer` against libllama: two norm facts no other
//! architecture in llama.cpp has.
//!
//! Both are in `src/models/muse-glimmer.cpp` and both had to be built
//! for this golden to hold:
//!
//!   1. **A weightless RMS on the EMBEDDINGS** (`:69`,
//!      `build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1)`), with
//!      no `token_embd_norm` tensor in the file. `bloom`'s embedding
//!      norm -- the only other one of the 155 graphs -- has a weight,
//!      and every other weightless RMS upstream is a LAYER slot, so
//!      the pair (this site, no weight) did not exist here:
//!      `TensorFacts::weightless_embedding_norm` is the row and
//!      `NormOp::RmsNoParams` the function.
//!   2. **A post-norm epsilon that is a LITERAL** (`:63`, `const float
//!      post_norm_eps = 1e-8f`, with the comment "Different to
//!      f_norm_rms_eps for post-attn / post-FFN norms"), used at
//!      `:140-141,166-167` while `:90,153` use the model's.
//!      `NormFacts::post_norm_eps_literal` is the row and
//!      `ModelConfig::post_norm_eps()` the one accessor the three host
//!      post-norm sites read; the fused Metal launches and the CUDA
//!      prefill refuse a model whose two epsilons differ, because each
//!      bakes ONE epsilon into its kernel.
//!
//! The fixture declares `attention.layer_norm_rms_epsilon = 1e-3`,
//! FOUR ORDERS larger than the literal, so the two are not
//! interchangeable on this file: a post-norm run at the model's
//! epsilon moves the logits, which is what makes the golden evidence
//! rather than decoration.
//!
//! Everything else it carries was already served and each table gained
//! one name: the per-element sigmoid attention gate (`:46,100-135`,
//! whose own comment says "same as afmoe"), RoPE on the SLIDING layers
//! only (`:88`) with `rope.freq_base_swa`, the window pattern read
//! scalar-then-array (`:26`, `load_swa_pattern(ml, 4)`; this file
//! declares the scalar 2, so layers 0 and 2 slide), a per-head QK
//! RMSNorm, and `logit_scale` MULTIPLIED with the final tanh softcap
//! on top.
//!
//! **Where the numbers come from.** `scripts/gptoss_reference_logits.cpp`
//! against a libllama built from `.scratch/llama.cpp` at `5b59b83`.
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_muse_glimmer_fixture.py \
//!     crates/frink-models/tests/fixtures/muse_glimmer_tiny.gguf
//! /tmp/ref_logits crates/frink-models/tests/fixtures/muse_glimmer_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, kl_vs_golden, load_graph_fixture, worst_vs,
    GRAPH_PROMPT,
};
use frink_models::norm::NormOp;
use frink_models::rope_layers::RopeLayers;

const MUSE: &str = "muse_glimmer";

/// llama.cpp's logits for `muse_glimmer_tiny.gguf` over [`GRAPH_PROMPT`].
const MUSE_GOLDEN: [f32; 48] = [
    -0.033096004,
    -0.29820913,
    -0.48638144,
    -0.2888131,
    -0.19230951,
    1.9990773,
    -1.1890546,
    -1.2670343,
    -0.9608549,
    -1.3708407,
    0.33525068,
    -0.0034216342,
    -0.61640334,
    -0.1591583,
    0.25167435,
    0.5724019,
    -0.42089224,
    2.2864869,
    -1.4286789,
    0.35290223,
    0.83093256,
    0.6839304,
    -0.49484947,
    0.037300583,
    0.96179724,
    -1.3598571,
    -1.8076324,
    0.05996145,
    -0.5335364,
    0.51912546,
    -0.070294686,
    0.9753507,
    0.52498853,
    -0.75025743,
    0.60509604,
    0.78714335,
    -0.38158724,
    0.090586275,
    -0.032029904,
    -0.88534766,
    1.5811108,
    0.03751316,
    0.47593895,
    0.7957943,
    -0.8536158,
    -1.7511691,
    0.9947831,
    1.4545946,
];

#[test]
fn muse_glimmer_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(MUSE, &MUSE_GOLDEN);
}

/// The number in the report, so it can be regenerated rather than
/// trusted. Run with `--nocapture` to see it.
#[test]
fn report_kl_against_llama_cpp() {
    let d = load_graph_fixture(MUSE);
    let mut kv = graph_caches(&d);
    let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    let kl = kl_vs_golden(&got, &MUSE_GOLDEN);
    let worst = worst_vs(&got, &MUSE_GOLDEN);
    println!("| `muse-glimmer` | {kl:.2e} | {worst:.2e} |");
    assert!(kl < 1e-8, "muse-glimmer: KL {kl}");
}

/// The two norm facts reached the config, and they are two different
/// numbers on this file.
#[test]
fn the_embedding_norm_is_weightless_and_the_post_norms_have_their_own_eps() {
    let d = load_graph_fixture(MUSE);
    assert_eq!(
        d.embedding_norm,
        NormOp::RmsNoParams,
        "muse-glimmer.cpp:69 norms the embeddings with no weight"
    );
    assert_eq!(d.config.rms_norm_eps, 1e-3);
    assert_eq!(d.config.post_norm_eps(), 1e-8);
    for (il, layer) in d.layers.iter().enumerate() {
        assert!(layer.attn.post_attn_norm.is_some(), "blk.{il}");
        assert!(layer.attn.post_ffn_norm.is_some(), "blk.{il}");
    }
}

/// The window pattern came from the SCALAR spelling, and the rotation
/// follows it.
#[test]
fn the_scalar_pattern_decides_which_layers_slide_and_rotate() {
    let d = load_graph_fixture(MUSE);
    assert_eq!(d.config.rope_layers, RopeLayers::SlidingOnly);
    for il in 0..4 {
        assert_eq!(
            d.config.layer_rope(il).is_some(),
            d.config.layer_sliding_window(il).is_some(),
            "blk.{il}: muse-glimmer.cpp:88 rotates exactly the sliding layers"
        );
    }
    // `set_swa_pattern(2)`: half the layers slide, and which half is
    // what the golden pins.
    let sliding = (0..4)
        .filter(|il| d.config.layer_sliding_window(*il).is_some())
        .count();
    assert_eq!(sliding, 2, "a period of 2 over four layers");
}
