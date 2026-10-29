//! openPangu-Embedded, checked against llama.cpp itself: a row that had
//! been filed as an EMBEDDING model from its name.
//!
//! `pangu-embedded` is openPangu-Embedded-1B / 7B (Huawei), a decoder
//! LLM (`PanguEmbeddedForCausalLM`; `conversion/pangu.py` is a
//! `TextModel` with an `lm_head`; "Embedded" means edge devices). frink
//! had it under `DeferredEncoderEmbedding` in the catalog and in
//! `embedding_model::NOT_YET`, so a real file was refused as an encoder
//! this crate does not have. `pangu-embed.cpp` is `llama.cpp`'s graph
//! with ONE difference: a REQUIRED `attn_output.bias` (`:37`, flag 0;
//! `proj_bias::ATTN_OUT_BIAS_CREATORS`). NEOX RoPE
//! (llama-model.cpp:2675), `n_rot == n_embd_head` (`:59`), fused or
//! split QKV (`:35`), SwiGLU (`:118-123`), `output` tied when absent
//! (`:22-27`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `pangu_embedded` | see `report_kl_against_llama_cpp` | |
//! | `pangu_embedded_fused` | (fused `attn_qkv`; libllama byte-identical to the split file) | |
//! | `pangu_embedded_output` | (a separate `output.weight`) | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_pangu_fixture.py \
//!     crates/frink-models/tests/fixtures/pangu_embedded_tiny.gguf [--fused-qkv | --output]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/pangu_embedded_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::RopeLayout;
use frink_models::Decoder;

const PANGU: &str = "pangu_embedded";
const PANGU_FUSED: &str = "pangu_embedded_fused";
const PANGU_OUTPUT: &str = "pangu_embedded_output";

const PANGU_GOLDEN: [f32; 48] = [
    0.36418727,
    -2.9886916,
    -0.6848898,
    -1.2053616,
    0.07544726,
    -2.762255,
    1.3927878,
    -2.9490185,
    -1.3029495,
    -0.53564036,
    -0.5718189,
    -0.3481601,
    -1.6766384,
    -3.0085373,
    1.0241673,
    0.37920648,
    1.3318384,
    -1.5700023,
    -0.1771574,
    -1.3311372,
    1.3281722,
    -1.4164968,
    -1.2125971,
    0.32548487,
    -1.0929202,
    -2.0692904,
    -0.9779012,
    -0.103432566,
    1.6959395,
    -2.1802597,
    2.8027835,
    -0.30698997,
    -1.6554761,
    -2.919616,
    -1.0220553,
    -0.5137596,
    0.90522486,
    -0.17770344,
    1.8490144,
    -0.026609242,
    2.5837765,
    2.4408107,
    -0.8441118,
    -0.30698708,
    -0.701975,
    2.077434,
    -2.497168,
    -0.09410006,
];

const PANGU_OUTPUT_GOLDEN: [f32; 48] = [
    1.1323619,
    -0.9568968,
    0.899494,
    1.029347,
    0.49796474,
    1.3224279,
    -1.5152233,
    -1.5602891,
    1.5065863,
    0.6214522,
    0.66266495,
    3.7042751,
    -0.58301985,
    -0.23464689,
    -0.67621076,
    0.3422795,
    1.5349938,
    -1.8877932,
    -1.0364965,
    -1.6657373,
    -0.45708525,
    2.256647,
    0.64403474,
    -1.0101988,
    -0.92882556,
    0.11811134,
    2.303063,
    -0.15895607,
    -1.3371646,
    0.12627265,
    -1.1193048,
    -1.9105799,
    -1.150451,
    -0.44306022,
    -1.6249063,
    -1.898076,
    0.20056656,
    -1.190696,
    0.6522414,
    0.89818555,
    -0.57478964,
    0.4048733,
    -0.85791385,
    -0.28360146,
    0.59549403,
    0.22130355,
    1.1331692,
    -0.23485816,
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
fn pangu_embedded_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(PANGU, &PANGU_GOLDEN);
}

#[test]
fn the_fused_qkv_spelling_matches_the_same_golden() {
    assert_all_three_paths_match(PANGU_FUSED, &PANGU_GOLDEN);
}

#[test]
fn a_separate_output_weight_matches_llama_cpp() {
    assert_all_three_paths_match(PANGU_OUTPUT, &PANGU_OUTPUT_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (PANGU, &PANGU_GOLDEN),
        (PANGU_FUSED, &PANGU_GOLDEN),
        (PANGU_OUTPUT, &PANGU_OUTPUT_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// A decoder, not an embedding model: on the generic NEOX path, with the
/// required bias loaded on every layer.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("pangu-embedded"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    assert!(!frink_models::embedding_model::is_embedding_arch(
        "pangu-embedded"
    ));
    let d = load_graph_fixture(PANGU);
    for (il, layer) in d.layers.iter().enumerate() {
        let b = layer.attn.o_bias.as_ref().expect("attn_output.bias");
        assert_eq!(b.len(), d.config.hidden_dim, "blk.{il}");
    }
}

/// The bias is the one thing the graph adds to `llama.cpp`'s, and it is
/// visible: dropped, the logits leave the golden.
#[test]
fn the_output_bias_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(PANGU);
    assert_decoder_matches_on_all_three_paths(&d, &PANGU_GOLDEN, GRAPH_TOL, "baseline");
    let saved: Vec<_> = d.layers.iter_mut().map(|l| l.attn.o_bias.take()).collect();
    let worst = worst_vs(&decode(&d), &PANGU_GOLDEN);
    assert!(worst > 1e-2, "attn_output.bias not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.o_bias = s;
    }
    assert_decoder_matches_on_all_three_paths(&d, &PANGU_GOLDEN, GRAPH_TOL, "restored");
}
