//! BitNet, checked against llama.cpp itself: the two norms INSIDE the
//! blocks.
//!
//! `bitnet` was triaged NEW CODE on two norm slots the generic decoder
//! did not have (`src/models/bitnet.cpp`): `attn_sub_norm` (`:24`,
//! `{n_embd}`) on the attention output BEFORE `wo` (`:101-106`), and
//! `ffn_sub_norm` (`:36`, `{n_ff}`) on `silu(gate) * up` BEFORE
//! `ffn_down` (`:127-141`). Neither is Gemma's post-norm, which sits on
//! the other side of the projection. The reach was MEASURED before a
//! line was written (`frink_models::sub_norms`): one graph of 155
//! creates either tensor, so the seam is one `bool` on `ModelConfig`
//! read by the loader (the pair is REQUIRED) and by the Metal
//! predicate (every fused launch refuses), and the arithmetic sits in
//! the ONE attention tail and the ONE dense FFN row body.
//!
//! # What each fixture is shaped to catch
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `bitnet` | both norms, weights drawn AWAY from one; no `output.weight`; `rope.scaling.type = linear` at 1.0 |
//! | `bitnet_scales` | the same file plus seven `2.0` per-projection `.scale` tensors: llama.cpp APPLIES them (its logits differ) and frink REFUSES the file by name |
//!
//! The norm weights are drawn around 1.5 with spread 0.5, so a norm
//! that is skipped, or applied with unit weights, moves the logits by
//! far more than the tolerance; the sabotage tests measure that rather
//! than assume it.
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own
//! `bitnet` graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`. Not by re-reading a
//! spec, and not by frink checking itself.
//!
//! Measured against that reference over `GRAPH_PROMPT`, the fixture
//! being F32 so no `vec_dot_type` question arises
//! (`report_kl_against_llama_cpp` prints this):
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `bitnet` | see the test's output | see the test's output |
//!
//! Regenerating (both halves must be redone together if the fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_bitnet_fixture.py \
//!     crates/frink-models/tests/fixtures/bitnet_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_bitnet_fixture.py \
//!     crates/frink-models/tests/fixtures/bitnet_scales_tiny.gguf --with-scales
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::config::RopeLayout;
use frink_models::{Decoder, ModelConfig};

const BITNET: &str = "bitnet";
const BITNET_SCALES: &str = "bitnet_scales";

const BITNET_GOLDEN: [f32; 48] = [
    -0.12547764,
    0.18523544,
    0.52988243,
    -0.26532736,
    -1.0190994,
    0.7036342,
    1.0287766,
    0.34641105,
    -0.32826522,
    0.42419302,
    0.5687842,
    -0.06531001,
    0.16017064,
    -0.30960256,
    0.24988714,
    -0.32860452,
    0.13365017,
    -0.65097713,
    0.24189074,
    0.057833016,
    -0.42345053,
    -0.14083058,
    -0.557377,
    0.14078963,
    0.22946452,
    -0.18131444,
    0.17040369,
    0.9604119,
    -0.068427846,
    0.888931,
    0.34001142,
    0.5274208,
    -0.13121608,
    -0.17464158,
    0.2620743,
    0.19921014,
    0.108441934,
    0.21665178,
    -0.21282876,
    -0.024257421,
    0.07280815,
    0.70018923,
    0.65501827,
    -0.8636498,
    0.48617867,
    0.16584647,
    0.27305567,
    0.19236019,
];

const BITNET_SCALES_GOLDEN_HEAD: [f32; 4] = [-0.37028125, 0.13126105, 0.6901369, -0.35633603];

#[test]
fn the_bitnet_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(BITNET, &BITNET_GOLDEN);
}

/// The measurement the module doc quotes.
#[test]
fn report_kl_against_llama_cpp() {
    let d = load_graph_fixture(BITNET);
    let mut kv = graph_caches(&d);
    let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    let kl = kl_vs_golden(&got, &BITNET_GOLDEN);
    let worst = worst_vs(&got, &BITNET_GOLDEN);
    println!("| `{BITNET}` | {kl:.2e} | {worst:.2e} |");
    assert!(kl < 1e-8, "{BITNET}: KL {kl}");
}

/// What the loader resolved, value by value: the fact, the two
/// tensors at their widths on every layer, the tied lm_head, the RoPE
/// layout, the no-op linear scaling.
#[test]
fn the_loader_resolves_the_row_the_way_llama_cpp_does() {
    let d = load_graph_fixture(BITNET);
    assert!(d.config.block_sub_norms, "the one fact (`sub_norms`)");
    assert_eq!(d.layers.len(), 3);
    for (il, layer) in d.layers.iter().enumerate() {
        let attn = layer
            .attn
            .attn_sub_norm
            .as_ref()
            .unwrap_or_else(|| panic!("layer {il} has no attn_sub_norm"));
        assert_eq!(
            attn.len(),
            d.config.hidden_dim,
            "layer {il}: bitnet.cpp:24 `{{n_embd}}`"
        );
        let ffn = layer
            .moe
            .ffn_sub_norm
            .as_ref()
            .unwrap_or_else(|| panic!("layer {il} has no ffn_sub_norm"));
        assert_eq!(
            ffn.len(),
            d.config.layer_shape(il).ffn_dim,
            "layer {il}: bitnet.cpp:36 `{{n_ff}}`"
        );
        // Drawn away from one on purpose; a fixture whose norm weights
        // were all one could not tell "applied" from "skipped with the
        // right magnitude".
        assert!(attn.iter().any(|w| (w - 1.0).abs() > 0.3), "layer {il}");
        assert!(ffn.iter().any(|w| (w - 1.0).abs() > 0.3), "layer {il}");
        // Not Gemma's slot: the post-norms are empty.
        assert!(layer.attn.post_attn_norm.is_none() && layer.attn.post_ffn_norm.is_none());
    }
    // bitnet.cpp:164: the LM head is tok_embd; the file has no output.weight.
    let file = frink_gguf::GgufFile::open(graph_fixture_path(BITNET)).expect("opens");
    assert!(file.find_tensor("output.weight").is_none());
    assert_eq!(d.output_head.rows(), d.config.vocab_size);
    // llama-model.cpp:2625: NEOX.
    assert_eq!(d.config.rope_layout, RopeLayout::Neox);
    // conversion/bitnet.py:19-20 writes linear at 1.0 on every export; a
    // factor of one resolves to no correction at all.
    assert!(d.config.rope_freqs.is_none(), "{:?}", d.config.rope_freqs);
}

/// Skipping either inner norm -- which is what every host body did
/// before the seam, and what every fused Metal kernel would still do
/// -- diverges from llama.cpp by orders of magnitude more than the
/// tolerance. Each norm separately, so that neither can hide behind
/// the other.
#[test]
fn skipping_either_inner_norm_diverges_from_llama_cpp() {
    let golden = &BITNET_GOLDEN;
    for which in ["attn", "ffn"] {
        let mut d = load_graph_fixture(BITNET);
        for layer in d.layers.iter_mut() {
            match which {
                "attn" => layer.attn.attn_sub_norm = None,
                _ => layer.moe.ffn_sub_norm = None,
            }
        }
        let worst = decode_worst(&d, golden);
        assert!(
            worst > 100.0 * GRAPH_TOL,
            "skipping {which}_sub_norm moved the worst logit by only {worst}; the fixture \
             cannot see the norm"
        );
    }
}

/// A norm applied with UNIT weights is the other cheap mistake (the
/// tensor read for its presence and not its values), and the weights
/// are drawn away from one so that it shows.
#[test]
fn applying_either_inner_norm_with_unit_weights_diverges_from_llama_cpp() {
    let golden = &BITNET_GOLDEN;
    for which in ["attn", "ffn"] {
        let mut d = load_graph_fixture(BITNET);
        for layer in d.layers.iter_mut() {
            match which {
                "attn" => {
                    let n = layer.attn.attn_sub_norm.as_ref().unwrap().len();
                    layer.attn.attn_sub_norm = Some(vec![1.0; n]);
                }
                _ => {
                    let n = layer.moe.ffn_sub_norm.as_ref().unwrap().len();
                    layer.moe.ffn_sub_norm = Some(vec![1.0; n]);
                }
            }
        }
        let worst = decode_worst(&d, golden);
        assert!(
            worst > 100.0 * GRAPH_TOL,
            "unit-weight {which}_sub_norm moved the worst logit by only {worst}"
        );
    }
}

/// Applying `attn_sub_norm` AFTER `wo` -- reading it as Gemma's
/// `post_attention_norm` -- is the confusion the two slots invite, and
/// on this fixture (`n_embd == n_heads * head_dim`, so the widths
/// agree and nothing would refuse it) it diverges.
#[test]
fn reading_attn_sub_norm_as_the_post_attention_norm_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture(BITNET);
    for layer in d.layers.iter_mut() {
        layer.attn.post_attn_norm = layer.attn.attn_sub_norm.take().map(Into::into);
    }
    let worst = decode_worst(&d, &BITNET_GOLDEN);
    assert!(worst > 100.0 * GRAPH_TOL, "{worst}");
}

/// The prefill body's batched dense FFN and the row body's expert
/// must apply the same norm: a batch of six positions through
/// `forward_batch_last` and six single tokens through `forward_token`
/// agree with each other and with llama.cpp (the three-path test
/// above holds the golden; this one pins that the batched arm did not
/// silently fall back to the row arm, by running a batch wide enough
/// to take it).
#[test]
fn the_batched_dense_ffn_applies_the_inner_norm() {
    let d = load_graph_fixture(BITNET);
    // Twelve positions: past `dense_ffn_batch`'s four-row threshold on
    // any backend.
    let prompt: Vec<usize> = GRAPH_PROMPT
        .iter()
        .chain(GRAPH_PROMPT.iter())
        .copied()
        .collect();
    let mut kv = graph_caches(&d);
    let batched = d.forward_batch_last(&prompt, 0, &mut kv);
    let mut kv = graph_caches(&d);
    let mut rowwise = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        rowwise = d.forward_token(tok, pos, &mut kv);
    }
    let worst = worst_vs(&batched, &rowwise);
    assert!(
        worst < GRAPH_TOL,
        "batched vs row-wise dense FFN differ by {worst}"
    );
}

/// A file carrying the optional per-projection `.scale` tensors
/// (`bitnet.cpp:27-43`) is refused BY NAME, before the unread-tensor
/// gate, because llama.cpp multiplies them in (`build_lora_mm`) and
/// frink does not: libllama's logits for this file differ from the
/// unscaled fixture's from the first value (measured, pinned below),
/// so running it as if unscaled would be wrong at every projection.
#[test]
fn a_file_carrying_per_tensor_weight_scales_is_refused_by_name() {
    // The premise: libllama honours the scales.
    assert!(
        BITNET_SCALES_GOLDEN_HEAD
            .iter()
            .zip(BITNET_GOLDEN.iter())
            .any(|(s, p)| (s - p).abs() > 1e-3),
        "the two goldens agree; the scales were not applied upstream and this test \
         measures nothing"
    );
    let path = graph_fixture_path(BITNET_SCALES);
    let file = frink_gguf::GgufFile::open(&path).expect("fixture opens");
    let config = ModelConfig::from_gguf(&file).expect("the header is a plain bitnet header");
    assert!(config.block_sub_norms);
    let msg = match Decoder::from_gguf(&path, config) {
        Ok(_) => panic!("a file with per-tensor scales loaded"),
        Err(err) => format!("{err}"),
    };
    assert!(msg.contains("per-tensor weight scales"), "{msg}");
    assert!(msg.contains("blk.0.attn_k.scale"), "{msg}");
    assert!(msg.contains("build_lora_mm"), "{msg}");
    // The refusal is a feature refusal, not the generic unread-tensor
    // error that `FRINK_ALLOW_UNKNOWN_TENSORS=1` can be talked past.
    assert!(!msg.contains("never read"), "{msg}");
}

fn decode_worst(d: &Decoder, golden: &[f32]) -> f32 {
    let mut kv = graph_caches(d);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = d.forward_token(tok, pos, &mut kv);
    }
    worst_vs(&out, golden)
}
