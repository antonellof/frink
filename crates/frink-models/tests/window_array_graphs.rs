//! The per-layer sliding-window ARRAY and the NextN/MTP blocks, checked
//! against llama.cpp itself: two seams (`frink_models::swa_layers`,
//! `frink_models::mtp_blocks`) that three refusals named and every
//! real EXAONE-4 32B, EXAONE-MoE, Olmo-3, MiMo-V2, Step-3.5 and Mellum
//! export carries.
//!
//! # The array
//!
//! At the pin (`604cc430`) `{arch}.attention.sliding_window_pattern` is
//! read in one of two ways (re-derived for #420; `crate::swa_layers`
//! holds the census):
//!
//! | mode | arrays are | scalars are | absent | rows |
//! |---|---|---|---|---|
//! | `load_swa_pattern` (`llama-model.cpp:3952-3959`) | the per-layer truth | the period | the seed | `exaone4`, `exaone-moe`, `olmo2`, `gemma3`, `mellum`, `cohere2moe`, ... (19) |
//! | `get_arr`, required | the per-layer truth, any length | refused | refused | `mimo2`, `step35`, `maple`, `gemma4`, ... |
//!
//! Four fixtures evidence the `load_swa_pattern` row:
//!
//! - `exaone_moe_array`: the base `exaone_moe` fixture plus the array
//!   `conversion/exaone.py:84` writes ([T, T, T, F], the seeded
//!   period). libllama's logits are the base's.
//! - `exaone_moe_array_flipped`: the same with the array INVERTED.
//!   libllama runs the array: its logits move off the base's by 0.70
//!   (until #420 frink ran the seeded period and read WRONG).
//! - `mellum`: its array [T, T, F, T] disagrees with the seeded period-4
//!   [T, T, T, F] on two layers; `load_tensors` reports `is_swa = 1, 1,
//!   0, 1`.
//! - `mellum_nopattern`: no key at all, the seeded period.
//!
//! # The MTP block
//!
//! `exaone_moe_mtp` is the base fixture with ONE NextN block appended
//! inside `block_count` (5 blocks, `nextn_predict_layers = 1`), the
//! shape of every K-EXAONE export (`exaone.py:132,146`). llama.cpp
//! reports `n_layer = 4, n_layer_all = 5` and "model has unused
//! tensor blk.4.*" fifteen times; its trunk is the same graph as the
//! base with the output head redrawn, and the golden is that. The
//! block's weights are drawn 8x wider than the trunk's, so a loader
//! that ran it as a fifth layer would not be a near miss -- and the
//! two sabotages below show that no config reaches that: counting it
//! as a layer asks it for experts a dense NextN block does not have,
//! and skipping it without saying so leaves fifteen tensors for the
//! consumption gate to name.
//!
//! **Where the numbers come from.** Each golden was produced by running
//! llama.cpp's own graph over its fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! Regenerating:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_exaone_moe_fixture.py \
//!     crates/frink-models/tests/fixtures/exaone_moe_array_tiny.gguf --window-array agree
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_exaone_moe_fixture.py \
//!     crates/frink-models/tests/fixtures/exaone_moe_array_flipped_tiny.gguf --window-array disagree
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_exaone_moe_fixture.py \
//!     crates/frink-models/tests/fixtures/exaone_moe_mtp_tiny.gguf --window-array agree --mtp
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mellum_fixture.py \
//!     crates/frink-models/tests/fixtures/mellum_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mellum_fixture.py \
//!     crates/frink-models/tests/fixtures/mellum_nopattern_tiny.gguf --no-pattern
//! /tmp/ref_logits <fixture> 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, EXAONE_MOE_GOLDEN, GRAPH_PROMPT,
};
use frink_models::loader::LoadError;
use frink_models::swa_layers::SwaLayers;
use frink_models::{Decoder, ModelConfig};

const EXAONE_MOE_MTP_GOLDEN: [f32; 48] = [
    -0.0399573,
    -0.26254788,
    0.112740405,
    -0.45957953,
    0.20378731,
    0.09908521,
    0.0650342,
    0.03963881,
    0.027113974,
    0.04426959,
    0.2971161,
    0.44927487,
    -0.08063339,
    -0.31896305,
    0.12336944,
    -0.26059663,
    0.23168293,
    -0.07957372,
    -0.041728646,
    -0.06822635,
    -0.012256693,
    -0.06147337,
    0.058683395,
    0.018150419,
    0.34285027,
    -0.17899069,
    0.35574993,
    0.005653359,
    0.21688741,
    0.18249817,
    0.015758194,
    0.19431275,
    -0.5292666,
    0.089496456,
    -0.27419174,
    0.6649489,
    -0.22339234,
    -0.28366813,
    0.23308644,
    0.13166983,
    -0.035332873,
    -0.45804775,
    0.030781724,
    0.18007353,
    0.16690017,
    -0.18101907,
    -0.04980211,
    -0.14753367,
];

const MELLUM_GOLDEN: [f32; 48] = [
    0.060756013,
    -0.023824722,
    0.20276277,
    -0.08673769,
    -0.17752253,
    0.16671993,
    -0.049065586,
    -0.14727482,
    -0.23216516,
    0.018855155,
    -0.15827064,
    -0.13220614,
    0.017678097,
    0.44898403,
    0.18785386,
    -0.060248606,
    0.18946445,
    0.09642051,
    0.009634852,
    -0.22297539,
    -0.1649667,
    0.3061224,
    -0.31932747,
    -0.063845895,
    -0.42996672,
    -0.00028830767,
    0.4683508,
    -0.33105162,
    0.21087095,
    -0.16849121,
    -0.39124924,
    0.124670506,
    0.019993221,
    -0.48215437,
    -0.43157175,
    -0.56464714,
    0.34288388,
    0.11505209,
    0.18613203,
    -0.30828017,
    0.1446941,
    -0.335675,
    -0.043562084,
    -0.14170834,
    0.27802056,
    0.03002052,
    0.1744021,
    -0.067254916,
];

/// `exaone_moe_array_flipped_tiny.gguf`: the array INVERTED against the
/// seeded period. `load_swa_pattern` takes the array (#420), so
/// libllama's logits move off the base file's by 0.70 max abs.
const EXAONE_MOE_FLIPPED_GOLDEN: [f32; 48] = [
    -0.36112043,
    -0.15549019,
    -0.2942359,
    -0.0548671,
    -0.08957204,
    0.086164206,
    0.25420028,
    0.049788203,
    0.20311397,
    -0.009974703,
    0.4299879,
    0.27177116,
    0.36194116,
    -0.3163033,
    -0.010676175,
    0.1885718,
    -0.120714575,
    0.08473517,
    0.0375147,
    0.4242869,
    0.26002222,
    0.33452454,
    -0.0062123537,
    -0.30934316,
    -0.5209664,
    -0.19133455,
    -0.103322566,
    -0.2931842,
    -0.13361463,
    -0.07746658,
    0.10823922,
    -0.13729525,
    0.22548118,
    -0.3418675,
    -0.10008506,
    -0.15268692,
    0.008641757,
    -0.094832055,
    -0.06721012,
    -0.5969528,
    0.043797053,
    -0.09765089,
    -0.20075879,
    1.4021993e-5,
    0.15751052,
    -0.02442417,
    -0.016382776,
    -0.6707088,
];

/// `mellum_nopattern_tiny.gguf`: no `sliding_window_pattern` at all.
/// `load_swa_pattern(ml, 4)` keeps the seed, [T, T, T, F] (#420).
const MELLUM_NOPATTERN_GOLDEN: [f32; 48] = [
    0.37298518,
    -0.18626924,
    0.33165437,
    -0.19307545,
    -0.17579763,
    0.12869173,
    0.26050994,
    0.05211313,
    -0.10542885,
    0.088689186,
    -0.23170376,
    -0.21285078,
    -0.32911098,
    0.1662871,
    0.14099197,
    0.20221432,
    0.51401716,
    -0.0034375787,
    -0.19342026,
    -0.030705646,
    -0.22674789,
    0.4210993,
    -0.13054617,
    -0.037026204,
    -0.4449697,
    0.23926044,
    0.47461843,
    -0.2816104,
    0.30103633,
    -0.16393946,
    -0.112568095,
    -0.017281383,
    -0.16355602,
    -0.437272,
    -0.22602874,
    -0.5332494,
    0.12737328,
    0.23687601,
    0.077030405,
    -0.17496257,
    -0.008741047,
    -0.3190671,
    0.04778391,
    0.26289815,
    0.07921851,
    0.12163802,
    0.41894308,
    0.32847905,
];

/// `mellum_mtp_tiny.gguf`: `mellum_tiny.gguf` with one MTP block
/// appended. libllama's answer is the trunk's, bit for bit (#441).
const MELLUM_MTP_GOLDEN: [f32; 48] = [
    0.060756013,
    -0.023824722,
    0.20276277,
    -0.08673769,
    -0.17752253,
    0.16671993,
    -0.049065586,
    -0.14727482,
    -0.23216516,
    0.018855155,
    -0.15827064,
    -0.13220614,
    0.017678097,
    0.44898403,
    0.18785386,
    -0.060248606,
    0.18946445,
    0.09642051,
    0.009634852,
    -0.22297539,
    -0.1649667,
    0.3061224,
    -0.31932747,
    -0.063845895,
    -0.42996672,
    -0.00028830767,
    0.4683508,
    -0.33105162,
    0.21087095,
    -0.16849121,
    -0.39124924,
    0.124670506,
    0.019993221,
    -0.48215437,
    -0.43157175,
    -0.56464714,
    0.34288388,
    0.11505209,
    0.18613203,
    -0.30828017,
    0.1446941,
    -0.335675,
    -0.043562084,
    -0.14170834,
    0.27802056,
    0.03002052,
    0.1744021,
    -0.067254916,
];

/// Every (fixture, golden) pair this suite audits, for the KL report.
const ROWS: [(&str, &[f32]); 5] = [
    ("exaone_moe_array", &EXAONE_MOE_GOLDEN),
    ("exaone_moe_array_flipped", &EXAONE_MOE_FLIPPED_GOLDEN),
    ("mellum_nopattern", &MELLUM_NOPATTERN_GOLDEN),
    ("exaone_moe_mtp", &EXAONE_MOE_MTP_GOLDEN),
    ("mellum", &MELLUM_GOLDEN),
];

fn prefill(decoder: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(decoder);
    decoder.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv)
}

// --- `load_swa_pattern`: the array, then the scalar, then the seed ---

/// The array a real converter writes AGREES with the seeded period, so
/// the base golden holds on all three paths whichever one is read.
#[test]
fn exaone_moe_with_the_converter_array_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("exaone_moe_array", &EXAONE_MOE_GOLDEN);
    let d = load_graph_fixture("exaone_moe_array");
    assert_eq!(
        d.config.swa_layers,
        SwaLayers::PerLayer(vec![true, true, true, false].into()),
        "load_swa_pattern takes the array first (llama-model.cpp:3953)"
    );
}

/// The INVERTED array is the layout libllama runs (#420): its logits
/// move off the base file's, and frink's follow them.
#[test]
fn exaone_moe_with_a_disagreeing_array_runs_the_array_as_llama_cpp_does() {
    assert_all_three_paths_match("exaone_moe_array_flipped", &EXAONE_MOE_FLIPPED_GOLDEN);
    assert!(worst_vs(&EXAONE_MOE_FLIPPED_GOLDEN, &EXAONE_MOE_GOLDEN) > 1e-1);
    let d = load_graph_fixture("exaone_moe_array_flipped");
    // The file says [F, F, F, T], and that is what runs.
    assert_eq!(d.config.layer_sliding_window(0), None);
    assert_eq!(
        d.config.layer_sliding_window(3),
        Some(d.config.sliding_window.unwrap())
    );
}

/// No key at all: `load_swa_pattern(ml, 4)` keeps the seed, where frink
/// used to refuse the file as missing a required key (#420).
#[test]
fn mellum_without_a_pattern_key_runs_the_seeded_period_as_llama_cpp_does() {
    assert_all_three_paths_match("mellum_nopattern", &MELLUM_NOPATTERN_GOLDEN);
    let d = load_graph_fixture("mellum_nopattern");
    assert_eq!(d.config.swa_layers, SwaLayers::period(4, false));
}

// --- the array, honoured where llama.cpp honours it ------------------

/// `mellum.cpp:16`: the array is the layout, and it is NOT the seeded
/// period.
#[test]
fn mellum_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("mellum", &MELLUM_GOLDEN);
    let d = load_graph_fixture("mellum");
    assert_eq!(
        d.config.swa_layers,
        SwaLayers::PerLayer(vec![true, true, false, true].into())
    );
    assert_eq!(d.config.sliding_window, Some(3));
    let windowed: Vec<bool> = (0..4)
        .map(|il| d.config.layer_sliding_window(il).is_some())
        .collect();
    assert_eq!(
        windowed,
        [true, true, false, true],
        "libllama's load_tensors reports is_swa = 1, 1, 0, 1"
    );
}

/// An MTP block appended to the same file is skipped, as libllama skips
/// it (#441): `mellum.cpp:79-89` create it `TENSOR_SKIP`, the common
/// loader has already subtracted it from `n_layer()`, and the logits
/// are the four-layer file's. frink refused this file until the census
/// was re-measured at the pin.
#[test]
fn mellum_with_an_mtp_block_matches_llama_cpp() {
    assert_eq!(MELLUM_MTP_GOLDEN, MELLUM_GOLDEN);
    assert_all_three_paths_match("mellum_mtp", &MELLUM_MTP_GOLDEN);
    let d = load_graph_fixture("mellum_mtp");
    assert_eq!((d.config.n_layers, d.config.n_mtp_blocks), (4, 1));
}

/// Sabotage: keeping the seeded period-4 layout in place of the file's
/// array -- what `exaone-moe` correctly does and `mellum` must not --
/// windows layer 2 and unwindows layer 3, and the logits see it.
#[test]
fn mellum_with_the_seeded_period_instead_of_its_array_diverges_from_llama_cpp() {
    let path = graph_fixture_path("mellum");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.swa_layers = SwaLayers::period(4, false);
    let d = common::load_decoder(&path, config);
    assert_eq!(d.config.layer_sliding_window(2), Some(3), "sabotage landed");
    assert_eq!(d.config.layer_sliding_window(3), None, "sabotage landed");
    let worst = worst_vs(&prefill(&d), &MELLUM_GOLDEN);
    assert!(
        worst > 1e-2,
        "the seeded period must be visible against a 3-token window: {worst}"
    );
}

/// Sabotage: windowing every layer -- the answer a missing array used
/// to fall through to -- diverges too.
#[test]
fn mellum_windowing_every_layer_diverges_from_llama_cpp() {
    let path = graph_fixture_path("mellum");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.swa_layers = SwaLayers::All;
    let d = common::load_decoder(&path, config);
    let worst = worst_vs(&prefill(&d), &MELLUM_GOLDEN);
    assert!(worst > 1e-2, "windowing layer 2 must be visible: {worst}");
}

// --- the MTP block -----------------------------------------------------

/// One NextN block inside `block_count`: the trunk runs, the block does
/// not, and the golden is llama.cpp's over the same file.
#[test]
fn exaone_moe_with_an_mtp_block_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("exaone_moe_mtp", &EXAONE_MOE_MTP_GOLDEN);
    let d = load_graph_fixture("exaone_moe_mtp");
    assert_eq!(
        (d.config.n_layers, d.config.n_mtp_blocks),
        (4, 1),
        "llama.cpp: n_layer = 4, n_layer_all = 5"
    );
    assert_eq!(d.layers.len(), 4, "the block is not a layer");
    // The trunk still has its window and its NoPE layer: the block
    // did not shift either rule.
    assert_eq!(d.config.layer_sliding_window(3), None);
    assert!(!d.config.layer_rotates(3));
}

/// Sabotage: a config that counts the block as a fifth layer cannot
/// load at all. A NextN block is DENSE upstream (`exaone-moe.cpp:73`
/// takes the dense branch for `i >= n_layer`) while the trunk's rule
/// makes every layer past the leading one MoE, so the loader asks the
/// block for expert tensors it does not have and stops there -- before
/// the consumption gate would have named the unread `nextn.*` head
/// (the test below shows that gate firing on its own). Running the
/// block is therefore not reachable by any config, which is stronger
/// than "diverges".
#[test]
fn counting_the_mtp_block_as_a_layer_cannot_load() {
    let path = graph_fixture_path("exaone_moe_mtp");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.n_layers = 5;
    config.n_mtp_blocks = 0;
    match Decoder::from_gguf(&path, config) {
        Err(LoadError::Gguf(frink_gguf::GgufError::TensorNotFound(name))) => {
            assert!(
                name.starts_with("blk.4.ffn_") && name.contains("_exps"),
                "the block has no experts: {name}"
            );
        }
        Err(other) => panic!("the block is not a layer, got {other:?}"),
        Ok(_) => panic!("the block is not a layer, yet it loaded as one"),
    }
}

/// Sabotage the other way: a config that skips the block but marks
/// nothing as skipped trips the same gate on all fifteen tensors. The
/// mark and the layer loop take their range from the same two config
/// fields, so this is the shape that CANNOT happen from the loader; it
/// is here to show the gate sees a skipped block that nobody declared.
#[test]
fn an_undeclared_skip_is_refused_by_the_consumption_gate() {
    let path = graph_fixture_path("exaone_moe_mtp");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.n_mtp_blocks = 0;
    match Decoder::from_gguf(&path, config) {
        Err(LoadError::UnconsumedTensors(n, _)) => assert_eq!(n, 15),
        Err(other) => panic!("fifteen unread tensors must refuse, got {other:?}"),
        Ok(_) => panic!("fifteen unread tensors must refuse, yet it loaded"),
    }
}

// --- the number this suite speaks in ------------------------------------

#[test]
fn report_kl_against_llama_cpp() {
    for (name, want) in ROWS {
        let d = load_graph_fixture(name);
        let got = prefill(&d);
        let kl = kl_vs_golden(&got, want);
        let worst = worst_vs(&got, want);
        eprintln!("{name}: KL(llama.cpp || frink) = {kl:.3e} nats, max |delta| = {worst:.3e}");
        assert!(kl < 1e-8, "{name}: KL {kl}");
    }
}
