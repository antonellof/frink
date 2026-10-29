//! DBRX, checked against llama.cpp itself.
//!
//! `dbrx` was triaged NEW CODE on three blockers, and each turned out to
//! be one implementation that another row also needed -- the shape this
//! repo's honest position says the NEW CODE column moves on:
//!
//! 1. **LayerNorm with a learned weight and no bias.** `dbrx.cpp:4` reads
//!    `LLM_KV_ATTENTION_LAYERNORM_EPS` and the graph passes `LLM_NORM`
//!    with a weight and a null bias at all three sites (:69-71, :110-112,
//!    :140-142). `crate::norm::NormOp::LayerNorm` is the variant the
//!    OLMo-1 work deliberately left unwritten until a row called it. It
//!    reaches every site that norms, including the fused Metal stacks,
//!    because those ask `NormOp::rms_weights()` and fall back to the host
//!    body on `None`.
//! 2. **A REQUIRED QKV clamp.** `dbrx.cpp:5` reads
//!    `{arch}.attention.clamp_kqv` with no default, and
//!    `llama-graph.cpp:1611-1652` clamps the projection after the bias.
//!    `crate::clamp_kqv` implements it through the one helper every host
//!    body shares (`decoder/qkv_bias.rs`), which is what also closed the
//!    clip_qkv sub-refusal on `olmo`.
//! 3. **The pre-FFN norm under another name.** `dbrx.cpp:34` creates
//!    `attn_out_norm` and no `ffn_norm`, and `:110-113` norms `ffn_inp`
//!    with it. `crate::norm_sites` reads `blk.N.attn_output_norm` into the
//!    pre-FFN slot -- and reads the SAME tensor name into `grok`'s
//!    post-attention slot, which is why the table exists.
//!
//! **Where the numbers come from.** `DBRX_GOLDEN` was produced by
//! running llama.cpp's own graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! Regenerating:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_dbrx_fixture.py \
//!     crates/frink-models/tests/fixtures/dbrx_tiny.gguf
//! /tmp/ref_logits crates/frink-models/tests/fixtures/dbrx_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GRAPH_PROMPT,
};
use frink_models::{ModelConfig, NormOp, RopeLayout};

const DBRX: &str = "dbrx";

const DBRX_GOLDEN: [f32; 48] = [
    1.0858252,
    -1.8408885,
    0.3869276,
    1.1676722,
    1.0398256,
    1.543047,
    0.85116374,
    3.7879746,
    2.6799614,
    1.4918476,
    0.81070256,
    0.11361289,
    1.0644239,
    2.6292768,
    2.1081023,
    0.5227695,
    -1.6096022,
    0.87097824,
    0.9810619,
    -0.7173015,
    -1.4883262,
    0.3530448,
    0.94246244,
    -2.5826697,
    -0.8795967,
    -0.7761003,
    -0.17827803,
    -0.54424894,
    2.0584912,
    -1.2654483,
    0.44401312,
    0.2428726,
    -0.19837248,
    3.507348,
    -0.0046930313,
    -1.0208509,
    0.7653076,
    -0.3968808,
    3.238174,
    -1.6467134,
    0.5265689,
    -0.8837353,
    -0.14669013,
    -1.5937493,
    -0.3958586,
    -0.8994981,
    -2.6471696,
    -0.868943,
];

/// The row itself, on all three forward paths.
#[test]
fn dbrx_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(DBRX, &DBRX_GOLDEN);
}

/// The number in the report, so it can be regenerated rather than
/// trusted. Run with `--nocapture` to see it.
#[test]
fn report_kl_against_llama_cpp() {
    let d = load_graph_fixture(DBRX);
    let mut kv = graph_caches(&d);
    let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    let kl = kl_vs_golden(&got, &DBRX_GOLDEN);
    let worst = worst_vs(&got, &DBRX_GOLDEN);
    println!("| `dbrx` | {kl:.2e} | {worst:.2e} |");
    assert!(kl < 1e-8, "KL {kl}");
}

/// Every norm site is the weighted LayerNorm, and the pre-FFN one came
/// from `attn_output_norm`.
///
/// Structural rather than numeric: if `loader.rs` ever reads `dbrx`'s
/// tensors into `NormOp::Rms`, the numeric test above fails too, but
/// this one says WHICH site and WHY.
#[test]
fn every_norm_site_is_the_weighted_layer_norm() {
    let d = load_graph_fixture(DBRX);
    assert_eq!(d.layers.len(), 2, "layer count");
    for (il, layer) in d.layers.iter().enumerate() {
        assert!(
            matches!(layer.attn.norm_weight, NormOp::LayerNorm(_)),
            "blk.{il}: dbrx.cpp:69-71 is LLM_NORM with a weight; got {:?}",
            layer.attn.norm_weight
        );
        assert!(
            matches!(layer.moe.norm_weight, NormOp::LayerNorm(_)),
            "blk.{il}: dbrx.cpp:110-112 is LLM_NORM with attn_out_norm; got {:?}",
            layer.moe.norm_weight
        );
        // No post-norms: dbrx has no Gemma-2 sandwich, and the tensor
        // that COULD be misread into the post-attention slot went to
        // the pre-FFN one instead.
        assert!(
            layer.attn.post_attn_norm.is_none(),
            "blk.{il}: attn_output_norm must not land in the post-attention slot"
        );
        assert!(layer.attn.post_ffn_norm.is_none());
    }
    assert!(
        matches!(d.final_norm, NormOp::LayerNorm(_)),
        "dbrx.cpp:140-142; got {:?}",
        d.final_norm
    );
    assert_eq!(
        d.config.rms_norm_eps, 1e-5,
        "the epsilon comes from `dbrx.attention.layer_norm_epsilon`, not the RMS spelling"
    );
}

/// Substituting an RMSNorm with the SAME weight at any site diverges.
///
/// This is the sabotage that matters, because it is the shortcut
/// somebody will reach for: "dbrx's norm tensors have the same shape as
/// everyone else's, so load them into the RMS slot". Each site is
/// substituted on its own, so the suite cannot pass by getting two of
/// three right.
#[test]
fn substituting_an_rmsnorm_with_the_same_weight_at_any_site_diverges_from_llama_cpp() {
    let as_rms = |op: &NormOp| match op {
        NormOp::LayerNorm(w) => NormOp::Rms(w.clone()),
        other => panic!("expected a weighted LayerNorm, got {other:?}"),
    };
    for site in ["attn", "ffn", "final"] {
        let mut d = load_graph_fixture(DBRX);
        match site {
            "attn" => {
                for layer in d.layers.iter_mut() {
                    layer.attn.norm_weight = as_rms(&layer.attn.norm_weight);
                }
            }
            "ffn" => {
                for layer in d.layers.iter_mut() {
                    layer.moe.norm_weight = as_rms(&layer.moe.norm_weight);
                }
            }
            _ => d.final_norm = as_rms(&d.final_norm),
        }
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &DBRX_GOLDEN,
        );
        assert!(
            worst > 1e-2,
            "an RMSNorm at the {site} site moved the output by only {worst}; this fixture's \
             hidden states must be too close to centred for the mean subtraction to matter, \
             and it cannot see the norm function it exists to pin"
        );
    }
}

/// The clamp is read, applied, and visible: dropping it diverges.
///
/// `clamp_kqv = 8.0` is every DBRX checkpoint's value. The sabotage is
/// what makes the golden evidence FOR the clamp rather than merely
/// consistent with it: if no projection in this fixture crossed 8.0 the
/// clamp would be inert and the suite could not tell whether it ran.
#[test]
fn the_clamp_is_applied_and_dropping_it_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture(DBRX);
    assert_eq!(d.config.clamp_kqv, Some(8.0));
    d.config.clamp_kqv = None;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &DBRX_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "dropping the clamp moved the output by only {worst}; the fixture's projections \
         must be too small for the clamp to bite"
    );
}

/// A DBRX file WITHOUT the clamp key is refused, as llama.cpp refuses it.
///
/// `dbrx.cpp:5` reads the key with no default, and libllama's loader
/// fails on `dbrx_noclamp_tiny.gguf` with `key not found in model:
/// dbrx.attention.clamp_kqv` (measured). The file is the fixture above
/// minus that one key, so the refusal is reachable from a file shaped
/// exactly like the ones the converter writes.
#[test]
fn a_dbrx_file_without_the_clamp_key_is_refused_like_llama_cpp_refuses_it() {
    let file =
        frink_gguf::GgufFile::open(graph_fixture_path("dbrx_noclamp")).expect("fixture opens");
    let err = ModelConfig::from_gguf(&file).expect_err("REQUIRED key missing");
    let msg = format!("{err}");
    assert!(msg.contains("dbrx.attention.clamp_kqv"), "{msg}");
    assert!(msg.contains("dbrx.cpp:5"), "{msg}");
    // The file with the key is the one that loads, or the gate above
    // would be refusing the architecture rather than the key.
    let ok = frink_gguf::GgufFile::open(graph_fixture_path(DBRX)).expect("fixture opens");
    assert!(ModelConfig::from_gguf(&ok).is_ok());
}

/// The fused QKV, the RoPE variant and the attention scale, each pinned
/// against the C and each with the fixture able to see the other
/// answer where there is one.
#[test]
fn dbrx_fuses_its_qkv_ropes_neox_and_uses_the_kernels_own_attention_scale() {
    let d = load_graph_fixture(DBRX);
    assert_eq!(d.config.rope_layout, RopeLayout::Neox);
    assert_eq!(
        d.config.attention_scale, None,
        "dbrx.cpp:97 passes 1/sqrt(n_embd_head)"
    );
    assert_eq!(d.config.n_heads, 4);
    assert_eq!(d.config.n_kv_heads, 2, "the fixture exercises GQA");
    assert_eq!(d.config.head_dim, 6);
    assert_eq!(d.config.moe.n_experts, 4);
    assert_eq!(d.config.moe.n_experts_active, 2);
    assert!(
        d.config.moe.norm_topk_prob,
        "build_moe_ffn(..., norm_w = true) at dbrx.cpp:122"
    );
    assert!(
        d.layers[0].attn.q_bias.is_none() && d.layers[0].attn.v_bias.is_none(),
        "dbrx.cpp:31 creates the fused weight and no bias"
    );
    // No scalar multipliers: dbrx.cpp reads none of the keys.
    assert_eq!(d.config.embedding_scale, None);
    assert_eq!(d.config.residual_scale, None);
    assert_eq!(d.config.logit_multiplier, None);

    let mut d = load_graph_fixture(DBRX);
    d.config.rope_layout = RopeLayout::Norm;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &DBRX_GOLDEN,
    );
    assert!(
        worst > 1e-3,
        "rotating the NORM pairs moved the output by only {worst}; the attention in this \
         fixture is too flat to see a positional bug"
    );
}
