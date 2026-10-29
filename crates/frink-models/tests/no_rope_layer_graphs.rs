//! The PER-LAYER-RoPE group, checked against llama.cpp itself.
//!
//! Three rows on ONE rule. llama.cpp gates rotation per layer in six
//! architectures (`frink_models::rope_layers` has the table and the
//! lines), and frink had no way to say so, which cost:
//!
//! * `exaone-moe` an UNAUDITED refusal, triaged NEW CODE on exactly this
//!   and nothing else (`exaone-moe.cpp:136,155-161`);
//! * `smollm3` an OUTRIGHT refusal, in the "No RoPE at all" group beside
//!   the ALiBi architectures (`smollm3.cpp:5,69`);
//! * EXAONE-4 32B a refusal BY NAME in `loader.rs`, on
//!   `block_count == 64` (`exaone4.cpp:4-9,116`).
//!
//! **`exaone-moe` and `exaone4` are one rule, not two that look alike.**
//! `exaone4.cpp:116` is `use_rope = is_swa(il) || swa_type == NONE`;
//! `exaone-moe.cpp:136` is `is_local_layer = is_swa(il)`, and
//! `exaone-moe.cpp:4` sets `swa_type = LLAMA_SWA_TYPE_STANDARD`
//! unconditionally, which nails `exaone4`'s second disjunct to false.
//! Read side by side, :155-161 and :132-138 are the same two
//! `ggml_rope_ext` calls under the same predicate; the only thing that
//! differs is that `exaone4` reaches that predicate solely at 64
//! layers. So `RopeLayers::SlidingOnly` is one variant and both rows
//! take it, and this suite is what proves the shared body is right for
//! both rather than right for one and plausible for the other.
//!
//! `smollm3` is a DIFFERENT variant of the same enum --
//! `RopeLayers::NoRopeEvery`, `(il + 1) % 4 != 0`, no window involved
//! -- and it is the row where the rule is the ONLY thing: its graph is
//! otherwise the plain pre-norm llama one. The other five gated
//! architectures each carry the rule plus something else, so this
//! fixture is the one that isolates it.
//!
//! # What each fixture is shaped to catch
//!
//! Every fixture is at least FOUR layers, because every rule here has a
//! period of 4 and a shorter file would rotate every layer and be
//! evidence about nothing. The 32B fixture is exactly 64, because
//! `exaone4.cpp:4` tests EQUALITY. Each carries a sliding window
//! NARROWER than the six-token prompt where the graph has one, so the
//! mask is exercised rather than being a no-op.
//!
//! | fixture | layers | unrotated layers | why |
//! |---|---|---|---|
//! | `exaone4_32b` | 64 | 3, 7, ..., 63 | `set_swa_pattern(4)` last-dense, then `is_swa(il)` |
//! | `exaone_moe` | 4 | 3 | the same, and layer 3 is also an MoE layer |
//! | `smollm3` | 4 | 3 | `(il + 1) % 4 == 0` |
//!
//! `exaone_moe` also carries every piece of MoE machinery its graph
//! uses -- leading dense, `exp_probs_b` drawn large enough to reorder
//! the top-k, a shared expert sized by
//! `expert_shared_feed_forward_length`, sigmoid gating from metadata,
//! `expert_weights_scale` and `_norm`, per-head QK-norm -- so "the MoE
//! half is fine" is measured here rather than asserted in a triage
//! verdict.
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! for that architecture over the same fixture file, through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`. Not by re-reading a spec,
//! and not by frink checking itself.
//!
//! Measured against that reference over `GRAPH_PROMPT`, all three
//! fixtures being F32 so no `vec_dot_type` question arises
//! (`report_kl_against_llama_cpp` prints this table):
//!
//! | arch | KL(llama.cpp \|\| frink) | max abs logit delta | top-1 |
//! |---|---|---|---|
//! | `exaone4_32b` | 2.05e-12 | 3.93e-06 | agrees |
//! | `exaone_moe` | 1.43e-14 | 3.87e-07 | agrees |
//! | `smollm3` | 5.29e-15 | 2.31e-07 | agrees |
//!
//! The 64-layer row's delta is ten times the others' and still 2500x
//! under the tolerance: sixty-four residual adds accumulate more
//! ordering noise than four.
//!
//! That is float32 accumulation noise -- ggml blocks its matmuls and
//! frink does not -- and it is orders of magnitude under every
//! sabotage below, each of which is required to move the logits by more
//! than 1e-2.
//!
//! Regenerating (both halves must be redone together if a fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_exaone4_fixture.py \
//!     crates/frink-models/tests/fixtures/exaone4_32b_tiny.gguf --layers 64
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_exaone_moe_fixture.py \
//!     crates/frink-models/tests/fixtures/exaone_moe_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_smollm3_fixture.py \
//!     crates/frink-models/tests/fixtures/smollm3_tiny.gguf
//! clang++ -std=c++17 -O2 scripts/gptoss_reference_logits.cpp \
//!     -I$LLAMA/include -I$LLAMA/ggml/include -L$BUILD/bin -lllama \
//!     -Wl,-rpath,$BUILD/bin -o /tmp/ref_logits
//! /tmp/ref_logits crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, kl_vs_golden, load_graph_fixture, worst_vs,
    EXAONE_MOE_GOLDEN, GRAPH_PROMPT,
};
use frink_models::rope_layers::{NoRopePhase, RopeLayers};
use std::num::NonZeroUsize;

/// All three rows, named once so no test below can quietly cover one
/// and claim the group.
const ROWS: [&str; 3] = ["exaone4_32b", "exaone_moe", "smollm3"];

const EXAONE4_32B_GOLDEN: [f32; 32] = [
    -0.31675017,
    -0.52429783,
    0.26734048,
    -0.102382086,
    -0.1522709,
    -0.7544714,
    -0.012922343,
    0.26474053,
    -0.048706904,
    0.2536888,
    0.12412921,
    -0.1337578,
    -0.16766791,
    0.25164938,
    -0.01992102,
    0.08800185,
    0.25482318,
    0.12509932,
    -0.3280951,
    0.15627429,
    0.19082192,
    0.062021248,
    0.31698078,
    0.0901043,
    0.025959905,
    0.1553717,
    0.6365426,
    -0.23589525,
    0.0011638589,
    -0.15762156,
    -0.037891954,
    0.027845688,
];

/// `exaone4_32b_nowindow_tiny.gguf`: the 64-layer file without
/// `attention.sliding_window`. libllama slides it at the seeded 4096
/// (`exaone4.cpp:6`), so the full-attention layers rotate nothing
/// (#440).
const EXAONE4_32B_NOWINDOW_GOLDEN: [f32; 32] = [
    -0.34208846,
    -0.3947661,
    -0.10962723,
    -0.042137608,
    0.0108428,
    -0.6239327,
    -0.13453189,
    0.10313081,
    -0.050256614,
    -0.12259874,
    0.19557089,
    -0.3169595,
    -0.037645638,
    0.28570563,
    0.13868013,
    0.36613095,
    0.21560583,
    -0.03157097,
    -0.17200866,
    0.0040836334,
    0.21910283,
    0.039009623,
    0.34845865,
    0.07631612,
    0.16707954,
    0.13410087,
    0.2729799,
    -0.2380636,
    -0.23302345,
    -0.14149782,
    -0.029318288,
    -0.029059485,
];

/// `exaone4_32b_mtp_tiny.gguf`: the 64-layer file with one MTP block
/// appended (`block_count` 65, `nextn_predict_layers` 1). libllama's
/// answer is the 64-layer file's, bit for bit: `exaone4.cpp:4` tests
/// the trunk and the block is `TENSOR_SKIP` (#441).
const EXAONE4_32B_MTP_GOLDEN: [f32; 32] = [
    -0.31675017,
    -0.52429783,
    0.26734048,
    -0.102382086,
    -0.1522709,
    -0.7544714,
    -0.012922343,
    0.26474053,
    -0.048706904,
    0.2536888,
    0.12412921,
    -0.1337578,
    -0.16766791,
    0.25164938,
    -0.01992102,
    0.08800185,
    0.25482318,
    0.12509932,
    -0.3280951,
    0.15627429,
    0.19082192,
    0.062021248,
    0.31698078,
    0.0901043,
    0.025959905,
    0.1553717,
    0.6365426,
    -0.23589525,
    0.0011638589,
    -0.15762156,
    -0.037891954,
    0.027845688,
];

const SMOLLM3_GOLDEN: [f32; 48] = [
    0.27115217,
    0.06371546,
    0.3299479,
    -0.2427842,
    0.23616026,
    -0.14877482,
    0.30543458,
    -0.22159739,
    0.41239274,
    0.12743974,
    -0.06658066,
    0.29521623,
    0.050334934,
    0.04756926,
    0.2542944,
    0.15043536,
    -0.39550003,
    0.4945417,
    0.15346187,
    0.11020486,
    0.019908935,
    0.0002733376,
    -0.46940517,
    0.37514725,
    -0.1582205,
    0.061323546,
    0.8541198,
    0.0029746816,
    0.19136414,
    0.29016852,
    0.45179823,
    0.10128763,
    0.07604365,
    0.01710502,
    0.26389623,
    -0.63600034,
    0.54424465,
    -0.07327777,
    0.027833104,
    0.44640344,
    -0.3069582,
    -0.059719637,
    0.33859453,
    -0.19505653,
    -0.19432792,
    -0.29664865,
    0.70958346,
    0.352763,
];

/// llama.cpp's logits for `maple_tiny.gguf` over [`GRAPH_PROMPT`].
const MAPLE_GOLDEN: [f32; 48] = [
    0.6486808,
    -0.23711741,
    -0.5294898,
    -0.46147358,
    -1.9501901,
    3.4603767,
    0.15342909,
    0.5967338,
    0.26974905,
    3.5528212,
    -1.273411,
    -1.2770839,
    -1.8373278,
    2.3348446,
    0.009929895,
    -0.6402428,
    2.4302635,
    0.2994001,
    -1.1233808,
    -0.85524726,
    -2.314069,
    0.15556979,
    0.22562456,
    0.4092492,
    -2.305313,
    1.2321382,
    -0.66026187,
    1.2131181,
    3.2547603,
    -1.1485307,
    0.2499813,
    0.66347164,
    -2.2737732,
    1.6876196,
    0.4008119,
    -0.59180975,
    0.7276684,
    0.29646862,
    0.77146876,
    1.9338744,
    4.3427477,
    1.137814,
    1.5413116,
    -0.009805858,
    -0.3871348,
    0.8032818,
    -1.3253019,
    1.0272561,
];

/// `maple_model_rope_tiny.gguf`: no `rope.freq_base_swa`, a model base
/// of 40000 and a linear factor of 2. `maple.cpp:12-13` seed the sliding
/// layers' base and scale from the model's (#446).
const MAPLE_MODEL_ROPE_GOLDEN: [f32; 48] = [
    0.9988805,
    -0.13319254,
    0.86428416,
    -0.60122246,
    -3.733892,
    2.8123028,
    0.06451139,
    0.32578102,
    0.53381896,
    2.6254642,
    -1.2790289,
    -0.8403716,
    -0.5325314,
    0.7440721,
    0.22820932,
    -0.5612092,
    2.4643466,
    1.1048443,
    -0.20520483,
    -1.860012,
    -2.1637416,
    0.31232756,
    1.1921091,
    0.45086825,
    -2.0796974,
    0.67866784,
    -1.0180643,
    -0.22611213,
    2.5526838,
    -1.5739808,
    0.058849514,
    1.5425493,
    -1.981108,
    1.6793767,
    0.26018664,
    -0.39497483,
    0.055113196,
    0.70387167,
    1.3647919,
    2.1699717,
    2.8602967,
    0.27044737,
    1.1593324,
    -0.91082084,
    -0.10111934,
    -0.07340336,
    -1.1552482,
    0.2756722,
];

fn golden(name: &str) -> &'static [f32] {
    match name {
        "exaone4_32b" => &EXAONE4_32B_GOLDEN,
        "exaone_moe" => &EXAONE_MOE_GOLDEN,
        "smollm3" => &SMOLLM3_GOLDEN,
        "maple" => &MAPLE_GOLDEN,
        other => panic!("no golden for {other}"),
    }
}

/// Which layers llama.cpp leaves unrotated in each fixture, from the
/// lines in the module docs. Written out rather than derived from the
/// rule under test, or the test would agree with any rule.
fn expected_unrotated(name: &str, n_layers: usize) -> Vec<usize> {
    match name {
        // exaone4.cpp:116 / exaone-moe.cpp:136 over set_swa_pattern(4)
        // last-dense: the dense layer of each period.
        "exaone4_32b" | "exaone_moe" => (0..n_layers).filter(|il| il % 4 == 3).collect(),
        // smollm3.cpp:69: (il + 1) % 4 == 0.
        "smollm3" => (0..n_layers).filter(|il| (il + 1) % 4 == 0).collect(),
        // maple.cpp:88 rotates only where `is_swa(il)`, and the
        // fixture's `attention.sliding_window_pattern` array says
        // [true, true, false, true] -- so layer 2, and only layer 2.
        "maple" => vec![2],
        other => panic!("no expectation for {other}"),
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

// --- the evidence ------------------------------------------------------

/// Maple, the SlidingOnly rule read off a per-layer ARRAY rather than a
/// period.
///
/// The two EXAONE rows and `cohere2` reach `SlidingOnly` with the
/// sliding layers decided by a seeded period; this fixture's pattern is
/// an array the FILE carries, so what it adds is that the two seams
/// compose -- `swa_layers` answers which layers slide, `rope_layers`
/// rotates exactly those, and neither restates the other's answer.
#[test]
fn maple_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("maple", &MAPLE_GOLDEN);
}

#[test]
fn exaone4_32b_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("exaone4_32b", &EXAONE4_32B_GOLDEN);
}

#[test]
fn exaone_moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("exaone_moe", &EXAONE_MOE_GOLDEN);
}

#[test]
fn smollm3_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("smollm3", &SMOLLM3_GOLDEN);
}

/// The numbers in the module-level table, so they can be regenerated
/// rather than trusted. Run with `--nocapture` to see them.
#[test]
fn report_kl_against_llama_cpp() {
    for name in ROWS {
        let d = load_graph_fixture(name);
        let mut kv = graph_caches(&d);
        let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
        let want = golden(name);
        let kl = kl_vs_golden(&got, want);
        let worst = worst_vs(&got, want);
        let top1 = if argmax(&got) == argmax(want) {
            "agrees"
        } else {
            "DIFFERS"
        };
        println!("| `{name}` | {kl:.2e} | {worst:.2e} | {top1} |");
        assert_eq!(argmax(&got), argmax(want), "{name}: top-1 must agree");
    }
}

// --- the rule, structurally ---------------------------------------------

/// The loader assigns the rule, and it names the right layers. Checked
/// against a hand-written expectation per row (not against
/// `RopeLayers::rotates`, which is the thing under test).
#[test]
fn the_loader_leaves_exactly_the_layers_llama_cpp_leaves_unrotated() {
    for name in ROWS {
        let d = load_graph_fixture(name);
        let n = d.config.n_layers;
        let unrotated: Vec<usize> = (0..n).filter(|&il| !d.config.layer_rotates(il)).collect();
        assert_eq!(unrotated, expected_unrotated(name, n), "{name}");
        // Not vacuous: every row here leaves at least one layer alone,
        // and rotates at least one.
        assert!(!unrotated.is_empty(), "{name}: the rule must fire");
        assert!(
            unrotated.len() < n,
            "{name}: the rule must not fire everywhere"
        );
        // And `layer_rope` -- the accessor every rotation site reads --
        // is `None` for exactly those layers, so the CPU and Metal
        // bodies cannot see a different answer from this one.
        for il in 0..n {
            assert_eq!(
                d.config.layer_rope(il).is_none(),
                unrotated.contains(&il),
                "{name} layer {il}: layer_rope and layer_rotates disagree"
            );
        }
    }
}

/// The two EXAONE rows resolve to the SAME variant, which is the claim
/// the whole group rests on; `smollm3` to a different one, which is why
/// it needs its own fixture.
#[test]
fn the_two_exaone_rows_share_one_variant_and_smollm3_has_its_own() {
    let exaone4 = load_graph_fixture("exaone4_32b").config.rope_layers;
    let exaone_moe = load_graph_fixture("exaone_moe").config.rope_layers;
    let smollm3 = load_graph_fixture("smollm3").config.rope_layers;
    assert_eq!(exaone4, RopeLayers::SlidingOnly);
    assert_eq!(exaone4, exaone_moe, "one rule, not two that look alike");
    assert_eq!(
        smollm3,
        RopeLayers::NoRopeEvery {
            step: NonZeroUsize::new(4).unwrap(),
            phase: NoRopePhase::LastOfPeriod,
        }
    );
}

/// EXAONE-4 32B's SWA is switched on by its layer count alone, and the
/// window is the file's: `exaone4.cpp:16` reads it after :4-9 has
/// already decided the pattern.
#[test]
fn exaone4_32b_gets_a_window_from_its_layer_count() {
    let d = load_graph_fixture("exaone4_32b");
    assert_eq!(d.config.n_layers, 64, "exaone4.cpp:4 tests equality");
    assert_eq!(d.config.sliding_window, Some(3));
    assert_eq!(
        d.config.swa_layers,
        frink_models::swa_layers::SwaLayers::period(4, false)
    );
}

/// Without the key the 32B still slides, at `exaone4.cpp:6`'s 4096,
/// and its full-attention layers still rotate nothing (#440).
///
/// Over this prompt a 4096 window masks nothing, so what the golden
/// sees is the rotation: no window would rotate every layer, which the
/// second assertion shows llama.cpp does not do.
#[test]
fn exaone4_32b_without_its_window_key_matches_llama_cpp() {
    assert_all_three_paths_match("exaone4_32b_nowindow", &EXAONE4_32B_NOWINDOW_GOLDEN);
    let d = load_graph_fixture("exaone4_32b_nowindow");
    assert_eq!(d.config.sliding_window, Some(4096));
    let mut every = load_graph_fixture("exaone4_32b_nowindow");
    every.config.rope_layers = RopeLayers::All;
    let mut kv = graph_caches(&every);
    let worst = worst_vs(
        &every.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &EXAONE4_32B_NOWINDOW_GOLDEN,
    );
    assert!(worst > 1e-2, "rotating every layer moved only {worst}");
}

/// One MTP block appended is still the 32B: `exaone4.cpp:4` sees the
/// trunk, and libllama's answer is the 64-layer file's (#441).
#[test]
fn exaone4_32b_with_an_mtp_block_matches_llama_cpp() {
    assert_eq!(EXAONE4_32B_MTP_GOLDEN, EXAONE4_32B_GOLDEN);
    assert_all_three_paths_match("exaone4_32b_mtp", &EXAONE4_32B_MTP_GOLDEN);
    let d = load_graph_fixture("exaone4_32b_mtp");
    assert_eq!((d.config.n_layers, d.config.n_mtp_blocks), (64, 1));
    assert_eq!(d.config.sliding_window, Some(3));
}

/// Maple's sliding layers rotate at the MODEL's base and scale when the
/// file has no `rope.freq_base_swa` (`maple.cpp:12-13`, #446), not at
/// `llama_hparams`' 10000 and 1.
#[test]
fn maple_without_a_swa_rope_base_matches_llama_cpp() {
    assert_all_three_paths_match("maple_model_rope", &MAPLE_MODEL_ROPE_GOLDEN);
    let d = load_graph_fixture("maple_model_rope");
    assert_eq!(d.config.rope_theta_swa, Some(40_000.0));
}

// --- sabotage: each of these must move the logits by more than 1e-2 ---

/// Rotating EVERY layer -- what frink did before this rule existed and
/// what every generic architecture still does -- diverges on all three
/// rows. This is the mutation that turns the rule back into the bug.
#[test]
fn rotating_every_layer_diverges_from_llama_cpp() {
    for name in ROWS {
        let mut d = load_graph_fixture(name);
        d.config.rope_layers = RopeLayers::All;
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            golden(name),
        );
        assert!(
            worst > 1e-2,
            "{name}: rotating the unrotated layers moved the output by only {worst}, \
             so the suite cannot see the rule at all"
        );
    }
}

/// The OTHER phase. `smollm3` skips the LAST layer of each period and
/// `smallthinker` the FIRST; on a four-layer file they disagree about
/// layers 0 and 3, and llama.cpp's logits know which.
#[test]
fn smollm3_with_smallthinkers_phase_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture("smollm3");
    d.config.rope_layers = RopeLayers::NoRopeEvery {
        step: NonZeroUsize::new(4).unwrap(),
        phase: NoRopePhase::FirstOfPeriod,
    };
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        golden("smollm3"),
    );
    assert!(
        worst > 1e-2,
        "the phase must be observable, moved only {worst}"
    );
}

/// Dropping the window on the two EXAONE rows is TWO errors at once --
/// the mask goes, and under `SlidingOnly` so does every rotation -- and
/// both are the answer llama.cpp gives a 30-layer EXAONE-4, so this is
/// the sabotage that says the 64-layer file is a different graph from
/// the 1.2B's rather than the same one with a bigger number.
#[test]
fn dropping_the_window_on_an_exaone_row_diverges_from_llama_cpp() {
    for name in ["exaone4_32b", "exaone_moe"] {
        let mut d = load_graph_fixture(name);
        d.config.sliding_window = None;
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            golden(name),
        );
        assert!(worst > 1e-2, "{name}: moved only {worst}");
    }
}

/// The window alone, with the rotation kept as llama.cpp has it: the
/// mask is exercised by these fixtures rather than being a no-op over
/// six tokens. Widening it past the prompt must diverge.
#[test]
fn widening_the_window_past_the_prompt_diverges_from_llama_cpp() {
    for name in ["exaone4_32b", "exaone_moe"] {
        let mut d = load_graph_fixture(name);
        d.config.sliding_window = Some(GRAPH_PROMPT.len() + 1);
        assert_eq!(
            d.config.rope_layers,
            RopeLayers::SlidingOnly,
            "rotation untouched"
        );
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            golden(name),
        );
        assert!(
            worst > 1e-2,
            "{name}: the window is not exercised, moved only {worst}"
        );
    }
}

/// `exaone_moe`'s MoE half is real: its selection bias reorders the
/// top-k, and zeroing it diverges. Without this the fixture could carry
/// a bias too small to matter and the verdict "the MoE half is fine"
/// would be as unmeasured as it was in the triage row.
#[test]
fn zeroing_exaone_moes_selection_bias_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture("exaone_moe");
    let mut touched = 0;
    for layer in d.layers.iter_mut() {
        if let Some(b) = layer.moe.exp_probs_bias.as_mut() {
            b.iter_mut().for_each(|v| *v = 0.0);
            touched += 1;
        }
    }
    assert_eq!(touched, 3, "three MoE layers behind one leading dense one");
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        golden("exaone_moe"),
    );
    assert!(worst > 1e-2, "moved only {worst}");
}
