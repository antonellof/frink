//! `granite_swa` (Granite 4.1) against libllama: the per-layer RoPE
//! PATTERN the file carries, and the seams that had to hold with it.
//!
//! `granite_swa` landed upstream after the 2026-08-04 llama.cpp pin and
//! was triaged NEW CODE on 2026-09-19 for two small per-layer tables.
//! One of them is the reason this suite exists:
//! `{arch}.attention.rope_pattern` (`src/models/granite-swa.cpp:43`,
//! read back at `:212` through `llama_hparams::has_rope`,
//! `llama-hparams.cpp:333-343`) is one entry per layer, nonzero meaning
//! "this layer rotates" -- and it is the FIRST upstream graph that lets
//! the FILE decide that. Every other per-layer RoPE gate in
//! `frink_models::rope_layers` is a rule read out of a literal:
//! `exaone-moe`'s `is_swa(il)`, `smollm3`'s `(il + 1) % 4`, `cohere2`'s
//! window. `grep -rn LLM_KV_ATTENTION_ROPE_PATTERN src/models/*.cpp`
//! over the 155 graphs is that one line, so
//! `RopeLayers::FileMask` is built for this architecture and for
//! nothing else -- upstream seeds the array with 1 for every model
//! (`llama-model.cpp:1314`) and only this graph reads it back, so
//! honouring the key elsewhere would answer differently from llama.cpp
//! on a file that carries it as dead metadata.
//!
//! **The two arrays disagree on purpose.** The fixture's window array
//! is `[true, false, true, true]` and its rope pattern is
//! `[1, 1, 0, 1]`, so layer 1 is the full-attention layer and layer 2
//! is the unrotated one. A loader that read either array into the
//! other's field would rotate the wrong layer and mask the wrong one,
//! and the golden would catch it -- which a fixture where both arrays
//! agree could not.
//!
//! What else the fixture carries, each a seam that already existed and
//! each now with one more name in its table:
//!
//!   * Granite's four scalar multipliers (`:7-10`; `logit_scale` is
//!     REQUIRED and DIVIDED at `:192`, the residual scale is applied to
//!     BOTH branch outputs at `:247-249,308-310`,
//!     `frink_models::scalar_multipliers`),
//!   * the sliding-window BOOL ARRAY read with `get_arr` (`:17`,
//!     `frink_models::swa_layers`), with a window of 3 against a
//!     six-token prompt,
//!   * REQUIRED per-layer attention sinks `{n_head}` (`:81`), one extra
//!     logit per head in the softmax,
//!   * all four OPTIONAL projection biases (`:79,100-102`,
//!     `frink_models::proj_bias`),
//!   * a `kq_scale` taken from `attention.scale` rather than
//!     `1/sqrt(head_dim)` (`:231`).
//!
//! **Where the numbers come from.** The golden was produced by running
//! llama.cpp's own graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp`, linked against a libllama
//! built from `.scratch/llama.cpp` at the moved pin (`5b59b83`).
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_granite_swa_fixture.py \
//!     crates/frink-models/tests/fixtures/granite_swa_tiny.gguf
//! /tmp/ref_logits crates/frink-models/tests/fixtures/granite_swa_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, kl_vs_golden, load_graph_fixture, worst_vs,
    GRAPH_PROMPT,
};
use frink_models::rope_layers::RopeLayers;

const GRANITE_SWA: &str = "granite_swa";

/// llama.cpp's logits for `granite_swa_tiny.gguf` over [`GRAPH_PROMPT`].
const GRANITE_SWA_GOLDEN: [f32; 48] = [
    0.857627,
    0.79289556,
    1.4505575,
    0.49421772,
    1.8097901,
    -1.6516342,
    0.30792546,
    -0.1794219,
    -0.19733909,
    1.1200176,
    1.7344476,
    -0.6214245,
    -0.61528766,
    -1.3066138,
    1.7541556,
    1.1465441,
    -0.20200875,
    -1.8967904,
    -0.3088787,
    -0.649673,
    1.1068534,
    -0.76487637,
    0.5709899,
    0.1268,
    -1.402708,
    -0.49631813,
    0.8200806,
    -0.908173,
    0.20929827,
    -0.31179804,
    0.05162111,
    -0.37773842,
    -1.4159544,
    0.02587609,
    -1.0802233,
    0.08250445,
    0.2793896,
    -0.7148607,
    0.47267687,
    -0.57510126,
    0.53714514,
    0.41321534,
    -0.17540231,
    0.32323018,
    -0.25954777,
    -0.31498933,
    1.2101995,
    0.15957242,
];

#[test]
fn granite_swa_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(GRANITE_SWA, &GRANITE_SWA_GOLDEN);
}

/// The number in the report, so it can be regenerated rather than
/// trusted. Run with `--nocapture` to see it.
#[test]
fn report_kl_against_llama_cpp() {
    let d = load_graph_fixture(GRANITE_SWA);
    let mut kv = graph_caches(&d);
    let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    let kl = kl_vs_golden(&got, &GRANITE_SWA_GOLDEN);
    let worst = worst_vs(&got, &GRANITE_SWA_GOLDEN);
    println!("| `granite_swa` | {kl:.2e} | {worst:.2e} |");
    assert!(kl < 1e-8, "granite_swa: KL {kl}");
}

/// The file's rope pattern reaches the config as a MASK, and it is not
/// the window array.
///
/// Structural, so a loader that read one array into the other says
/// which before the numeric test says "diverged".
#[test]
fn the_rope_pattern_and_the_window_array_are_two_different_arrays() {
    let d = load_graph_fixture(GRANITE_SWA);
    assert_eq!(
        d.config.rope_layers,
        RopeLayers::FileMask(vec![true, true, false, true].into()),
        "granite-swa.cpp:43 reads the pattern the file carries"
    );
    // The window array is the OTHER one, and layer 1 is its exception.
    assert_eq!(d.config.layer_sliding_window(0), Some(3));
    assert_eq!(d.config.layer_sliding_window(1), None);
    assert_eq!(d.config.layer_sliding_window(2), Some(3));
    assert_eq!(d.config.layer_sliding_window(3), Some(3));
    // And the two answers compose: layer 2 slides AND does not rotate,
    // layer 1 rotates AND does not slide.
    assert!(d.config.layer_rope(0).is_some());
    assert!(d.config.layer_rope(1).is_some());
    assert!(
        d.config.layer_rope(2).is_none(),
        "the pattern's zero is the only thing that stops layer 2 rotating"
    );
    assert!(d.config.layer_rope(3).is_some());
}

/// Granite's multipliers reached the config, including the one that is
/// DIVIDED rather than multiplied.
#[test]
fn the_four_granite_multipliers_are_read() {
    let d = load_graph_fixture(GRANITE_SWA);
    assert_eq!(d.config.embedding_scale, Some(1.5));
    assert_eq!(d.config.residual_scale, Some(0.75));
    assert_eq!(d.config.attention_scale, Some(0.2));
    // `granite-swa.cpp:192` is `ggml_scale(cur, 1.0f / f_logit_scale)`,
    // and `ModelConfig::logit_multiplier` is always the MULTIPLY, so a
    // Granite row carries the reciprocal (`scalar_multipliers`).
    assert_eq!(d.config.logit_multiplier, Some(0.5));
}

/// Every layer's REQUIRED sink tensor loaded, one scalar per head.
#[test]
fn every_layer_carries_its_attention_sinks() {
    let d = load_graph_fixture(GRANITE_SWA);
    for (il, layer) in d.layers.iter().enumerate() {
        let sinks = layer
            .attn
            .sinks
            .as_ref()
            .unwrap_or_else(|| panic!("granite_swa blk.{il} has no sinks"));
        assert_eq!(sinks.len(), d.config.n_heads, "blk.{il}");
    }
}

/// Refused by name, as libllama refuses it: the file leaves out the
/// window key the graph reads REQUIRED (#470).
fn assert_window_refused(fixture: &str, line: &str) {
    let file = frink_gguf::GgufFile::open(common::graph_fixture_path(fixture)).expect("opens");
    match frink_models::ModelConfig::from_gguf(&file) {
        Err(frink_models::LoadError::UnsupportedFeature(_, msg)) => {
            assert!(msg.contains("attention.sliding_window"), "{msg}");
            assert!(msg.contains(line), "{msg}");
        }
        Err(other) => {
            panic!("{fixture}: expected the missing window refused by name, got {other:?}")
        }
        Ok(_) => panic!("{fixture}: a file without its REQUIRED window loaded"),
    }
}

/// Granite-SWA without its REQUIRED window (`granite-swa.cpp:18`).
#[test]
fn granite_swa_without_its_window_key_is_refused() {
    assert_window_refused("granite_swa_nowindow", "granite-swa.cpp:18");
}
