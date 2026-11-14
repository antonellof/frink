//! GLM-4.5-MoE (`glm4moe`: GLM-4.5, GLM-4.5-Air, GLM-4.6), checked
//! against llama.cpp itself on the generic path.
//!
//! The row refused twice before it ran. First the refusal sent the
//! reader to `glm52_gguf_loader`, which asks for a `q_lora_rank` no
//! glm4moe file carries -- `glm4-moe.cpp`'s `load_arch_hparams` reads
//! none of the four MLA keys and its `load_arch_tensors` calls
//! `create_tensor_qkv` -- so a real GLM-4.5-Air download failed with
//! "missing hparam glm4moe.attention.q_lora_rank", a true statement
//! about a key the architecture is not supposed to have. Then it named
//! the ONE thing genuinely missing: the pre-FFN norm stored as
//! `blk.N.post_attention_norm.weight` with no `blk.N.ffn_norm`
//! (`glm4-moe.cpp:75`, applied to `ffn_inp` at `:215`, gpt-oss's slot).
//! That is one row in `norm_sites::PRE_FFN_NORM_IS_POST_ATTENTION_NORM`,
//! and with it the row matches libllama at 1e-15: everything else --
//! Q/K/V biases, the optional per-head Q/K norm before RoPE, NEOX
//! partial RoPE, a leading dense block, sigmoid routing with
//! `exp_probs_b`, `expert_weights_norm` and `expert_weights_scale` read
//! from the file, a shared expert summed with the routed output -- the
//! generic decoder already computed and `dots1` already pinned.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `glm4moe` | the 355B shape: per-head `attn_q_norm` / `attn_k_norm` present (`:68-71`) |
//! | `glm4moe_air` | the GLM-4.5-Air shape: no Q/K norms; its golden differs |
//! | `glm4moe_mrope` | a GLM-4.5V text tower's `rope.dimension_sections = [2, 1, 1, 0]`, under which llama.cpp rotates with `LLAMA_ROPE_TYPE_MROPE` (`rope type = 8`, llama-model.cpp:2700); on text positions its logits are BYTE-IDENTICAL to the plain file's (measured), because M-RoPE with equal position components is NEOX rotation band for band, so frink serves it as NEOX and this test pins the identity |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own
//! `glm4moe` graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `glm4moe` | 1.51e-15 | 1.19e-07 |
//! | `glm4moe_air` | 2.41e-15 | 1.79e-07 |
//! | `glm4moe_mrope` | the `glm4moe` golden, byte for byte | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_glm4moe_fixture.py \
//!     crates/frink-models/tests/fixtures/glm4moe_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_glm4moe_fixture.py \
//!     crates/frink-models/tests/fixtures/glm4moe_air_tiny.gguf --no-qk-norm
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_glm4moe_fixture.py \
//!     crates/frink-models/tests/fixtures/glm4moe_mrope_tiny.gguf --mrope
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    graph_fixture_path, kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_gguf::TensorSource;
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::RopeLayout;
use frink_models::loader::LoadError;
use frink_models::norm::NormOp;
use frink_models::Decoder;

const GLM4MOE: &str = "glm4moe";
const AIR: &str = "glm4moe_air";
const MROPE: &str = "glm4moe_mrope";

const GLM4MOE_GOLDEN: [f32; 48] = [
    0.012747437,
    0.19718257,
    0.3139101,
    0.31187516,
    0.0592269,
    -0.3197574,
    0.23505591,
    0.12870896,
    0.061043933,
    -0.12670158,
    -0.12723793,
    0.20605353,
    0.10945897,
    0.28252238,
    -0.34613985,
    0.41095144,
    0.4254516,
    0.12488929,
    -0.24611542,
    0.32800087,
    -0.415311,
    -0.19070686,
    0.19303957,
    0.21971852,
    0.50665843,
    0.03924465,
    0.106951,
    -0.14779708,
    -0.23050058,
    -0.080485485,
    0.2147947,
    0.41916752,
    0.4865568,
    -0.31319448,
    -0.2481955,
    0.06569699,
    0.25685608,
    -0.042138822,
    0.28081876,
    -0.0052292794,
    0.114929214,
    -0.28674406,
    -0.18028164,
    0.076168686,
    0.2702269,
    -0.026183885,
    0.21088699,
    0.12327288,
];

const GLM4MOE_AIR_GOLDEN: [f32; 48] = [
    0.022734322,
    -0.16619587,
    -0.45168376,
    -0.14376625,
    0.30490786,
    -0.36061755,
    0.15730397,
    -0.8146218,
    0.26460934,
    -0.22337492,
    -0.13377604,
    -0.18001673,
    -0.11210279,
    0.2518542,
    -0.0058252737,
    0.2034002,
    0.28530413,
    0.27295452,
    0.5432568,
    -0.30961668,
    -0.0201118,
    0.1823842,
    0.31615397,
    0.3635513,
    -0.16979967,
    -0.18807733,
    -0.002304852,
    0.03886442,
    0.050278023,
    0.19846027,
    0.4199319,
    -0.014283404,
    -0.39950824,
    0.045017812,
    -0.07330832,
    -0.06997451,
    -0.17255118,
    -0.48187476,
    0.33324328,
    0.1031757,
    -0.19696799,
    0.40305454,
    -0.18704185,
    0.030153781,
    0.040605515,
    0.6886872,
    -0.04690735,
    0.263619,
];

fn open(name: &str) -> frink_gguf::GgufFile {
    frink_gguf::GgufFile::open(graph_fixture_path(name)).expect("fixture opens")
}

fn decode(decoder: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    out
}

#[test]
fn glm4moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(GLM4MOE, &GLM4MOE_GOLDEN);
}

#[test]
fn the_air_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(AIR, &GLM4MOE_AIR_GOLDEN);
    // The two shapes really differ: a Q/K norm silently dropped would
    // land on the other golden, and the two are apart.
    assert!(worst_vs(&GLM4MOE_GOLDEN, &GLM4MOE_AIR_GOLDEN) > 1e-2);
}

/// `rope.dimension_sections` on a text tower: llama.cpp switches to
/// `LLAMA_ROPE_TYPE_MROPE` and produces the same logits (measured, byte
/// for byte); frink rotates NEOX and matches the same golden.
#[test]
fn a_glm45v_text_tower_rotates_as_neox_and_llama_cpp_agrees() {
    let file = open(MROPE);
    assert!(
        file.metadata("glm4moe.rope.dimension_sections").is_some(),
        "the fixture must declare the sections for this to prove anything"
    );
    assert_all_three_paths_match(MROPE, &GLM4MOE_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [(GLM4MOE, &GLM4MOE_GOLDEN), (AIR, &GLM4MOE_AIR_GOLDEN)] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// The fixture really is a `glm4moe` file with none of the MLA keys the
/// first refusal asked for, and the norm slot the second one named --
/// otherwise this suite tests a strawman rather than the architecture.
#[test]
fn a_valid_glm4moe_checkpoint_carries_no_mla_hparams_and_no_ffn_norm() {
    let file = open(GLM4MOE);
    assert_eq!(file.metadata_str("general.architecture"), Some("glm4moe"));
    for key in [
        "glm4moe.attention.q_lora_rank",
        "glm4moe.attention.kv_lora_rank",
        "glm4moe.attention.qk_nope_head_dim",
        "glm4moe.attention.qk_rope_head_dim",
    ] {
        assert!(
            file.metadata(key).is_none(),
            "{key} must not exist: glm4-moe.cpp's load_arch_hparams never reads it"
        );
    }
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
    ] {
        assert!(file.find_tensor(name).is_some(), "{name} must exist");
    }
    for l in 0..2 {
        assert!(
            file.find_tensor(&format!("blk.{l}.post_attention_norm.weight"))
                .is_some(),
            "glm4-moe.cpp:75 makes this REQUIRED"
        );
        assert!(
            file.find_tensor(&format!("blk.{l}.ffn_norm.weight"))
                .is_none(),
            "glm4-moe.cpp creates no ffn_norm"
        );
    }
    // And the GLM-5.2 loader refuses the ARCHITECTURE now, up front,
    // rather than failing on the first MLA key it cannot find.
    match frink_models::glm52_gguf_loader::read_glm52_hparams(&file) {
        Err(LoadError::UnsupportedArchitecture(arch)) => assert_eq!(arch, "glm4moe"),
        Err(other) => panic!("expected the architecture refused up front, got {other:?}"),
        Ok(_) => panic!("the GLM-5.2 loader must not accept a glm4moe file"),
    }
}

/// What the loader built: the pre-FFN slot holds `post_attention_norm`,
/// the Gemma slot is empty, the routing is the file's, the Air shape
/// has no Q/K norm.
#[test]
fn the_loaded_layers_are_glm4_moe_cpps() {
    assert!(matches!(
        resolve_architecture("glm4moe"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(GLM4MOE);
    for layer in &d.layers {
        assert!(
            matches!(layer.moe.norm_weight, NormOp::Rms(_)),
            "the pre-FFN slot is filled"
        );
        assert!(
            layer.attn.post_attn_norm.is_none(),
            "Gemma's slot stays empty"
        );
        assert!(layer.attn.q_norm.is_some() && layer.attn.k_norm.is_some());
    }
    assert!(
        d.layers[0].moe.shared_experts.is_empty(),
        "the leading dense block"
    );
    assert_eq!(d.layers[1].moe.shared_experts.len(), 1);
    assert!(d.layers[1].moe.exp_probs_bias.is_some());
    assert_eq!(d.config.moe.expert_weights_scale, 2.5);
    assert_eq!(d.config.moe.gating, frink_moe::GatingFunction::Sigmoid);
    let air = load_graph_fixture(AIR);
    for layer in &air.layers {
        assert!(layer.attn.q_norm.is_none() && layer.attn.k_norm.is_none());
    }
}

/// Each thing the golden checks, sabotaged on the loaded decoder: the
/// pre-FFN norm flattened, the shared expert dropped, the routing
/// scale reset, the Q norm flattened.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(GLM4MOE);
    assert_decoder_matches_on_all_three_paths(&d, &GLM4MOE_GOLDEN, GRAPH_TOL, "baseline");

    let saved: Vec<_> = d
        .layers
        .iter_mut()
        .map(|l| std::mem::replace(&mut l.moe.norm_weight, NormOp::Rms(vec![1.0; 32].into())))
        .collect();
    let worst = worst_vs(&decode(&d), &GLM4MOE_GOLDEN);
    assert!(
        worst > 1e-2,
        "post_attention_norm as the pre-FFN norm not seen: {worst}"
    );
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.moe.norm_weight = s;
    }

    let shared = std::mem::take(&mut d.layers[1].moe.shared_experts);
    let worst = worst_vs(&decode(&d), &GLM4MOE_GOLDEN);
    assert!(worst > 1e-2, "shared expert not seen: {worst}");
    d.layers[1].moe.shared_experts = shared;

    let scale = d.config.moe.expert_weights_scale;
    d.config.moe.expert_weights_scale = 1.0;
    let worst = worst_vs(&decode(&d), &GLM4MOE_GOLDEN);
    assert!(worst > 1e-2, "expert_weights_scale not seen: {worst}");
    d.config.moe.expert_weights_scale = scale;

    // The per-head Q norm flattened. (`exp_probs_b` is a SELECTION
    // bias: on this fixture's six tokens it does not change the top-2,
    // so dropping it leaves the logits alone -- `tests/moe_routing_bias.rs`
    // is where it is measured to matter.)
    let saved: Vec<_> = d
        .layers
        .iter_mut()
        .map(|l| l.attn.q_norm.replace(vec![1.0; 16].into()))
        .collect();
    let worst = worst_vs(&decode(&d), &GLM4MOE_GOLDEN);
    assert!(worst > 1e-2, "attn_q_norm not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.q_norm = s;
    }

    assert_decoder_matches_on_all_three_paths(&d, &GLM4MOE_GOLDEN, GRAPH_TOL, "restored");
}
