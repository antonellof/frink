//! OLMo-1, checked against llama.cpp itself.
//!
//! `olmo` is AI2's OLMo-1 and is NOT `olmo2`. It was triaged NEW CODE,
//! and the blocker turned out to be the norm FUNCTION rather than the
//! residual wiring: `src/models/olmo.cpp:15-36` creates Q/K/V,
//! `attn_output` and gate/up/down and **not one norm tensor**, and its
//! graph normalises at all three sites with a null weight and a null
//! bias.
//!
//! ```text
//! cur = build_norm(inpL,    NULL, NULL, LLM_NORM, il);  // :65-67
//! cur = build_norm(ffn_inp, NULL, NULL, LLM_NORM, il);  // :104-106
//! cur = build_norm(cur,     NULL, NULL, LLM_NORM, -1);  // :128-130
//! ```
//!
//! `LLM_NORM` is `ggml_norm`: subtract the mean, divide by the standard
//! deviation over the BIASED variance
//! (`ggml/src/ggml-cpu/ops.cpp:3716-3745`). So OLMo-1 is pre-norm like
//! `llama`, and what differs is the function --
//! `crate::norm::NormOp::LayerNormNoParams`.
//!
//! **The shared cause everyone hoped for is not there, and that is a
//! measurement.** Every `build_norm` call in all 155 of llama.cpp's
//! `src/models/*.cpp` graphs was scanned for a null weight argument.
//! Three calls pass one to `LLM_NORM`, and all three are `olmo.cpp`;
//! `talkie.cpp` passes one to `LLM_NORM_RMS` at five sites, which is a
//! different function. `openelm`, `bitnet`, `arcee`, `mellum`,
//! `nanbeige` and `deci` were checked by name and none of them
//! normalises without parameters. So this row closed ALONE, unlike
//! `olmo2`/`exaone4` and unlike the three Granite rows.
//!
//! The LayerNorm *function* is shared -- `dbrx` and the
//! `nemotron` / `orion` / `stablelm` / `codeshell` / `jais2` /
//! `starcoder` / `starcoder2` / `phimoe` bias group all use `LLM_NORM`
//! with a learned weight -- but every one of them is refused for more
//! than the norm, so a `LayerNorm(weight, bias)` variant would have had
//! no caller. `capability::NON_PARAMETRIC_LAYER_NORM` says all of this
//! where the next person will look.
//!
//! **The clamp was a refusal, and it is implemented now.** The `olmo`
//! triage verdict called `{arch}.attention.clamp_kqv` "an optional key
//! nothing here applies", which reads like an aside. It is not:
//! `llama-graph.cpp:1611-1652` clamps Q, K and V by it inside
//! `build_qkv`, `conversion/olmo.py:23-25` writes it for every
//! checkpoint whose HF config carries a `clip_qkv` (OLMo-7B-Twin-2T and
//! OLMo-1.7-7B do, at 8.0; the original OLMo-7B does not), and the
//! second fixture measures that llama.cpp's own logits MOVE when the
//! key is present. It stayed a refusal while frink's three host bodies
//! each applied the QKV bias in their own loop, because a clamp added
//! to some of them would have been wrong on the others; it closed when
//! `dbrx` needed the same clamp as a REQUIRED key and the three loops
//! collapsed onto one helper (`decoder/qkv_bias.rs`, `crate::clamp_kqv`).
//! `OLMO_CLAMPED_GOLDEN` is llama.cpp's answer for the clamped file, and
//! the sabotage that pins it is dropping the clamp.
//!
//! **Where the numbers come from.** `OLMO_GOLDEN` was produced by
//! running llama.cpp's own graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! Regenerating (both files together):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_olmo_fixture.py \
//!     crates/frink-models/tests/fixtures/olmo_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_olmo_fixture.py \
//!     crates/frink-models/tests/fixtures/olmo_clamped_tiny.gguf --clamp
//! /tmp/ref_logits crates/frink-models/tests/fixtures/olmo_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, load_graph_fixture, worst_vs, GRAPH_PROMPT,
};
use frink_models::{NormOp, RopeLayout};

const OLMO: &str = "olmo";

const OLMO_GOLDEN: [f32; 48] = [
    -1.6020501,
    0.50705075,
    -0.15139839,
    -0.5893539,
    -1.2324784,
    1.0751607,
    -1.1203105,
    -0.5548687,
    -1.9752331,
    -1.4798931,
    0.06502618,
    1.3066584,
    -0.7089759,
    0.21442454,
    0.011099964,
    -0.020552337,
    -0.628126,
    0.2954016,
    2.3210027,
    0.32131535,
    -2.238449,
    1.7778922,
    0.008124579,
    -0.41518313,
    0.8804916,
    -3.3213017,
    0.42046535,
    -1.2797751,
    -2.1666622,
    -1.4526423,
    1.0017449,
    1.2853384,
    -1.0446689,
    -0.61206055,
    2.1103303,
    -0.3623781,
    0.99392796,
    1.5957211,
    -0.30395782,
    1.9168491,
    0.63057876,
    1.120366,
    -0.034361586,
    -0.023620725,
    1.2378674,
    -0.7475845,
    1.0401692,
    0.78326005,
];

/// The same weights with `olmo.attention.clamp_kqv = 8.0`, from
/// llama.cpp. Differs from `OLMO_GOLDEN` by up to 0.18 in a logit,
/// which is the clamp and nothing else: the two files are
/// byte-identical apart from that one key.
const OLMO_CLAMPED_GOLDEN: [f32; 48] = [
    -1.5948613,
    0.5056803,
    -0.13761881,
    -0.41353196,
    -1.2346972,
    1.0697594,
    -1.1932158,
    -0.51069546,
    -1.9717276,
    -1.4389496,
    0.0059762,
    1.2528489,
    -0.7875987,
    0.31666917,
    0.12518159,
    -0.07026945,
    -0.5794481,
    0.35173202,
    2.3756158,
    0.3073057,
    -2.2741225,
    1.7606305,
    -0.06974431,
    -0.36501426,
    0.8467765,
    -3.2045987,
    0.42799103,
    -1.3141127,
    -2.1140578,
    -1.5678383,
    1.0730993,
    1.3464007,
    -1.0052842,
    -0.66942203,
    2.0486064,
    -0.34467196,
    1.0760533,
    1.6434469,
    -0.30358222,
    1.9849664,
    0.6845192,
    1.1937413,
    -0.018013388,
    0.05063148,
    1.2606807,
    -0.8459838,
    1.043538,
    0.8363968,
];

/// The row itself, on all three forward paths.
#[test]
fn olmo_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(OLMO, &OLMO_GOLDEN);
}

/// The topology: OLMo-1 norms BEFORE both sublayers, and has nothing to
/// norm with at any of the three sites.
///
/// Structural rather than numeric, and it is the assertion that
/// separates this row from the one it is most likely to be confused
/// with. `olmo2` has NO pre-norms and two post-norms; `olmo` has two
/// pre-norms, no post-norms, and no weights anywhere. If this ever
/// fails, `loader.rs` found a tensor OLMo-1 files do not contain.
#[test]
fn every_norm_site_is_the_non_parametric_layer_norm() {
    let d = load_graph_fixture(OLMO);
    assert_eq!(d.layers.len(), 2, "layer count");
    for (il, layer) in d.layers.iter().enumerate() {
        assert_eq!(
            layer.attn.norm_weight,
            NormOp::LayerNormNoParams,
            "blk.{il}: the attention branch norms the residual, with no weight"
        );
        assert_eq!(
            layer.moe.norm_weight,
            NormOp::LayerNormNoParams,
            "blk.{il}: the FFN branch norms the residual, with no weight"
        );
        // And NOT the olmo2 shape: no post-norms at all.
        assert!(
            layer.attn.post_attn_norm.is_none(),
            "blk.{il}: olmo.cpp creates no ATTN_POST_NORM"
        );
        assert!(
            layer.attn.post_ffn_norm.is_none(),
            "blk.{il}: olmo.cpp creates no FFN_POST_NORM"
        );
    }
    assert_eq!(
        d.final_norm,
        NormOp::LayerNormNoParams,
        "olmo.cpp:15-36 creates no `output_norm` and :128-130 norms with a null weight"
    );
}

/// Substituting an RMSNorm at the three sites diverges.
///
/// This is the sabotage that matters, because it is the shortcut
/// somebody will reach for: "OLMo-1 has no norm weights, so load a
/// vector of ones and keep the RMSNorm". An all-ones RMSNorm does not
/// subtract the mean, and this measures how far from the truth that
/// lands. Without it, `LayerNormNoParams` and `Rms(vec![1.0; n])` would
/// be indistinguishable to this suite and the variant would be
/// decoration.
///
/// Each site is substituted on its own, so the suite cannot pass by
/// getting two of three right.
#[test]
fn substituting_an_all_ones_rmsnorm_at_any_site_diverges_from_llama_cpp() {
    for site in ["attn", "ffn", "final"] {
        let mut d = load_graph_fixture(OLMO);
        let hidden = d.config.hidden_dim;
        match site {
            "attn" => {
                for layer in d.layers.iter_mut() {
                    layer.attn.norm_weight = NormOp::Rms(vec![1.0; hidden].into());
                }
            }
            "ffn" => {
                for layer in d.layers.iter_mut() {
                    layer.moe.norm_weight = NormOp::Rms(vec![1.0; hidden].into());
                }
            }
            _ => d.final_norm = NormOp::Rms(vec![1.0; hidden].into()),
        }
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &OLMO_GOLDEN,
        );
        assert!(
            worst > 1e-2,
            "an all-ones RMSNorm at the {site} site moved the output by only {worst}; this \
             fixture's hidden states must be too close to centred for the mean subtraction \
             to matter, and it cannot see the norm function it exists to pin"
        );
    }
}

/// Dropping the norm at any site diverges too.
///
/// The complementary half of the test above: `NormOp::None` is the
/// olmo2/exaone4 answer, and reading OLMo-1 as that topology -- "no
/// norm tensors, so no norm" -- is the other plausible misreading of
/// the same file.
#[test]
fn dropping_the_norm_at_any_site_diverges_from_llama_cpp() {
    for site in ["attn", "ffn", "final"] {
        let mut d = load_graph_fixture(OLMO);
        match site {
            "attn" => {
                for layer in d.layers.iter_mut() {
                    layer.attn.norm_weight = NormOp::None;
                }
            }
            "ffn" => {
                for layer in d.layers.iter_mut() {
                    layer.moe.norm_weight = NormOp::None;
                }
            }
            _ => d.final_norm = NormOp::None,
        }
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &OLMO_GOLDEN,
        );
        assert!(
            worst > 1e-2,
            "removing the {site} norm moved the output by only {worst}; the fixture cannot \
             tell OLMo-1 from the post-norm-only topology"
        );
    }
}

/// The lm_head is TIED, and the attention scale is the kernels' own.
///
/// `olmo.cpp:21-25` creates `output` as `TENSOR_NOT_REQUIRED` and falls
/// back to `token_embd`; the fixture ships no `output.weight`, so a
/// loader that required one could not open it. `:94` passes
/// `1/sqrtf(float(n_embd_head))` literally, so `attention_scale` must
/// stay `None` -- `Some` there means "pre-scale Q", which would scale
/// every score twice.
#[test]
fn olmo_ties_its_lm_head_and_uses_the_kernels_own_attention_scale() {
    let d = load_graph_fixture(OLMO);
    assert_eq!(d.config.attention_scale, None);
    assert_eq!(d.config.n_heads, 4);
    assert_eq!(d.config.n_kv_heads, 2, "the fixture exercises GQA");
    assert_eq!(d.config.head_dim, 6);
    assert_eq!(
        d.config.rms_norm_eps, 1e-5,
        "the epsilon comes from `olmo.attention.layer_norm_epsilon`, not the RMS spelling"
    );
    // No scalar multipliers: olmo.cpp reads none of the four keys.
    assert_eq!(d.config.embedding_scale, None);
    assert_eq!(d.config.residual_scale, None);
    assert_eq!(d.config.logit_multiplier, None);
}

/// OLMo-1's RoPE is the consecutive-pairs variant, and the fixture can
/// see the other one.
///
/// `LLM_ARCH_OLMO` is in `llama_model_rope_type`'s NORM group
/// (llama-model.cpp:2585), which is also why `conversion/olmo.py:33-36`
/// permutes `q_proj` and `k_proj` the way `LlamaModel` does. Nothing in
/// a GGUF says which variant an architecture uses.
#[test]
fn olmo_ropes_consecutive_pairs_and_the_fixture_can_see_the_other_variant() {
    let d = load_graph_fixture(OLMO);
    assert_eq!(d.config.rope_layout, RopeLayout::Norm);

    let mut d = load_graph_fixture(OLMO);
    d.config.rope_layout = RopeLayout::Neox;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &OLMO_GOLDEN,
    );
    assert!(
        worst > 1e-3,
        "rotating the NEOX pairs moved the output by only {worst}; the attention in this \
         fixture is too flat to see a positional bug"
    );
}

/// An `olmo` file declaring a positive `attention.clamp_kqv` matches
/// llama.cpp on all three paths, with the clamp applied.
///
/// `olmo_clamped_tiny.gguf` is byte-identical to the fixture above
/// except for that one key at 8.0 -- OLMo-1.7-7B's value -- so this is
/// the clamp's own evidence, not the row's.
#[test]
fn a_clamped_olmo_file_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("olmo_clamped", &OLMO_CLAMPED_GOLDEN);
}

/// The loader reads the clamp into the config for exactly this
/// architecture, and llama.cpp's own "no clamp" sentinel is honoured.
#[test]
fn the_clamp_is_read_into_the_config() {
    let clamped = load_graph_fixture("olmo_clamped");
    assert_eq!(clamped.config.clamp_kqv, Some(8.0));
    let plain = load_graph_fixture(OLMO);
    assert_eq!(
        plain.config.clamp_kqv, None,
        "no key: llama.cpp's 0.0 default, no clamp"
    );
}

/// Dropping the clamp diverges from llama.cpp, and applying one where
/// there is none diverges too.
///
/// The first half is the sabotage that makes `OLMO_CLAMPED_GOLDEN` a
/// test rather than a table: if the clamped file's projections never
/// crossed 8.0 the clamp would be inert and this suite could not see
/// whether it ran. The second half pins the direction: the unclamped
/// file must not be clamped, or the original OLMo-7B -- whose converted
/// file carries no key -- would run a graph llama.cpp does not.
#[test]
fn removing_or_inventing_the_clamp_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture("olmo_clamped");
    d.config.clamp_kqv = None;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &OLMO_CLAMPED_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "dropping the clamp moved the output by only {worst}; the clamped fixture's \
         projections must be too small for the clamp to bite, and it cannot see the \
         feature it exists to pin"
    );

    let mut d = load_graph_fixture(OLMO);
    d.config.clamp_kqv = Some(8.0);
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &OLMO_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "clamping the unclamped file moved the output by only {worst}"
    );
}
