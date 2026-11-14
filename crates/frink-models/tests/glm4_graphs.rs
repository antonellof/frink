//! GLM-4-0414 (`glm4`: the 9B and 32B, GLM-Z1, GLM-OCR), checked against
//! llama.cpp itself on the generic path.
//!
//! `glm4` was dispatched to the GLM-5.2 MLA loader, which asks for
//! `attention.q_lora_rank` and three more keys `src/models/glm4.cpp:3-9`
//! never read, so a real GLM-4-9B-0414 download failed with "missing
//! hparam glm4.attention.q_lora_rank" -- the `glm4moe` defect a second
//! time, on the model family's dense members. What the graph is
//! (`glm4.cpp:97-176`): plain GQA with Q/K/V biases, NORM RoPE over the
//! first half of each head, Gemma-2's `post_attention_norm` and
//! `post_ffw_norm` in Gemma-2's slots beside the ordinary `attn_norm` /
//! `ffn_norm`, and a FUSED SwiGLU `ffn_up` of `{n_embd, 2 * n_ff}` with
//! no gate -- every piece of which the generic decoder already served.
//! No code changed for the row: its profile moved from `dedicated` to
//! `gqa_norm`, and the fixture matched on the first run.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `glm4` | the converter's text shape (`conversion/glm.py:21,50`): `rope.dimension_count = head_dim / 2`, no sections |
//! | `glm4_mrope` | a GLM-4.1V text tower's `rope.dimension_sections`: llama.cpp rotates it `LLAMA_ROPE_TYPE_MROPE` (`rope type = 8`) over weights the converter permuted to NEOX (`glm.py:53-85`), and its logits differ from the plain file's by 0.72 (measured), so frink REFUSES it (`frink_models::mrope`) |
//!
//! # Where the numbers come from
//!
//! `GLM4_GOLDEN` was produced by running llama.cpp's own `glm4` graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `glm4` | 9.67e-15 | 3.87e-07 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_glm4_fixture.py \
//!     crates/frink-models/tests/fixtures/glm4_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_glm4_fixture.py \
//!     crates/frink-models/tests/fixtures/glm4_mrope_tiny.gguf --mrope
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    graph_fixture_path, kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::RopeLayout;
use frink_models::loader::LoadError;
use frink_models::mrope::{declares_mrope, mrope_refusal};
use frink_models::{Decoder, ModelConfig};

const GLM4: &str = "glm4";
const MROPE: &str = "glm4_mrope";

const GLM4_GOLDEN: [f32; 48] = [
    0.07828741,
    0.5034288,
    0.046144612,
    0.08549856,
    0.01195875,
    0.5581027,
    -0.28935105,
    -0.39554062,
    -0.0053126365,
    0.34893978,
    -0.26434582,
    0.49873286,
    0.023306243,
    -0.21226008,
    -0.10752797,
    -0.023722157,
    -0.08046431,
    -0.28843424,
    -0.4327777,
    0.12573273,
    -0.16537665,
    -0.24213424,
    0.051234663,
    -0.13113153,
    0.20615202,
    0.30344272,
    -0.030324187,
    0.26124275,
    -0.5148146,
    0.22128172,
    -0.51012486,
    0.48977447,
    -0.20926204,
    0.09606652,
    -0.31659088,
    -0.5964767,
    0.10322812,
    0.34475273,
    -0.084027946,
    0.095903635,
    -0.36978218,
    -0.18283989,
    -0.34375945,
    -0.0799817,
    0.11921175,
    -0.6111267,
    -0.5834216,
    0.20025179,
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
fn glm4_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(GLM4, &GLM4_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    let out = decode(&load_graph_fixture(GLM4));
    println!(
        "glm4: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
        kl_vs_golden(&out, &GLM4_GOLDEN),
        worst_vs(&out, &GLM4_GOLDEN)
    );
}

/// The row, and what the loader built: NORM RoPE over half the head,
/// both post norms in Gemma's slots, the fused `ffn_up` split into a
/// gate and an up of `n_ff` rows each, biases on Q/K/V.
#[test]
fn the_loaded_layers_are_glm4_cpps() {
    assert!(matches!(
        resolve_architecture("glm4"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Norm
        })
    ));
    let d = load_graph_fixture(GLM4);
    assert_eq!(
        d.config.rope_dim,
        Some(8),
        "head_dim 16 * partial_rotary_factor 0.5"
    );
    for layer in &d.layers {
        assert!(
            layer.attn.post_attn_norm.is_some(),
            "Gemma's post-attention slot"
        );
        assert!(layer.attn.post_ffn_norm.is_some(), "Gemma's post-FFN slot");
        assert!(
            layer.attn.q_bias.is_some()
                && layer.attn.k_bias.is_some()
                && layer.attn.v_bias.is_some()
        );
        layer.moe.with_expert(0, |ex| {
            assert_eq!(ex.gate.rows(), 40, "the first half of the fused ffn_up");
            assert_eq!(ex.up.rows(), 40, "the second half");
        });
    }
    // The MLA loader refuses the architecture up front.
    let file = frink_gguf::GgufFile::open(graph_fixture_path(GLM4)).unwrap();
    match frink_models::glm52_gguf_loader::read_glm52_hparams(&file) {
        Err(LoadError::UnsupportedArchitecture(arch)) => assert_eq!(arch, "glm4"),
        Err(other) => panic!("expected the architecture refused up front, got {other:?}"),
        Ok(_) => panic!("the GLM-5.2 loader must not accept a glm4 file"),
    }
}

/// A vision export's text tower is refused, from a file that has the
/// sections; the plain file, which differs only by that key, loads.
#[test]
fn a_glm41v_text_tower_is_refused_by_name() {
    let file = frink_gguf::GgufFile::open(graph_fixture_path(MROPE)).expect("fixture opens");
    assert!(
        declares_mrope(&file, "glm4"),
        "the fixture must declare M-RoPE"
    );
    assert!(mrope_refusal(&file, "glm4").is_some());
    let err = ModelConfig::from_gguf(&file).expect_err("refused");
    let msg = err.to_string();
    assert!(
        msg.contains("rope.dimension_sections") && msg.contains("glm.py:53-85"),
        "{msg}"
    );
    let plain = frink_gguf::GgufFile::open(graph_fixture_path(GLM4)).unwrap();
    assert!(!declares_mrope(&plain, "glm4"));
    assert!(mrope_refusal(&plain, "glm4").is_none());
}

/// Each thing the golden checks, sabotaged on the loaded decoder: each
/// post norm flattened, the rotary width widened to the whole head, the
/// Q bias dropped.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(GLM4);
    assert_decoder_matches_on_all_three_paths(&d, &GLM4_GOLDEN, GRAPH_TOL, "baseline");

    let saved: Vec<_> = d
        .layers
        .iter_mut()
        .map(|l| l.attn.post_attn_norm.replace(vec![1.0; 32].into()))
        .collect();
    let worst = worst_vs(&decode(&d), &GLM4_GOLDEN);
    assert!(worst > 1e-2, "post_attention_norm not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.post_attn_norm = s;
    }

    let saved: Vec<_> = d
        .layers
        .iter_mut()
        .map(|l| l.attn.post_ffn_norm.replace(vec![1.0; 32].into()))
        .collect();
    let worst = worst_vs(&decode(&d), &GLM4_GOLDEN);
    assert!(worst > 1e-2, "post_ffw_norm not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.post_ffn_norm = s;
    }

    d.config.rope_dim = Some(16);
    let worst = worst_vs(&decode(&d), &GLM4_GOLDEN);
    assert!(worst > 1e-2, "partial rotary width not seen: {worst}");
    d.config.rope_dim = Some(8);

    let saved: Vec<_> = d.layers.iter_mut().map(|l| l.attn.q_bias.take()).collect();
    let worst = worst_vs(&decode(&d), &GLM4_GOLDEN);
    assert!(worst > 1e-2, "attn_q.bias not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.q_bias = s;
    }

    assert_decoder_matches_on_all_three_paths(&d, &GLM4_GOLDEN, GRAPH_TOL, "restored");
}
