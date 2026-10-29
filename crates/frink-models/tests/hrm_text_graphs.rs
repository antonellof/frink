//! `hrm_text` (DFM Mimir 1B) against libllama: TWO residual streams and
//! two stacks replayed over twelve slots.
//!
//! `src/models/hrm-text.cpp` is the first decoder here that is not a
//! walk down one residual stream:
//!
//! ```text
//! zH = embeddings * embedding_scale        // :174
//! zL = hrm.z_l_init                        // :182, one [n_embd] row
//! for h in 0..h_cycles {                   // :183
//!     for l in 0..l_cycles { zL = stack(zH + zL) }
//!     zH = stack(zH + zL)
//! }
//! logits = output * zH                     // :198, NO final norm
//! ```
//!
//! and its stacks are ALIASES: `:47-87` creates `2 * layers_per_stack`
//! blocks (LOW at `[0, lps)`, HIGH at `[lps, 2*lps)`) while
//! `block_count` is the expanded slot count `lps * h * (l + 1)`, which
//! `:22-23` asserts. Each slot keeps its own KV.
//!
//! `frink_models::layer_loops::LayerLoops::Hrm` is the schedule --
//! which physical block a slot runs, where a stack starts, which
//! stream a finished stack writes -- and `frink_models::hrm` is the
//! state the four host bodies carry, one type with two methods rather
//! than four copies of "hold two vectors and add them here".
//!
//! The fixture is `lps = 2, h = 2, l = 2`: twelve slots over four
//! blocks, passes LOW LOW HIGH LOW LOW HIGH. Both counts are above one
//! on purpose -- at `l_cycles = 1` the passes alternate and a schedule
//! that had the aliasing backwards would still touch both stacks in
//! the right order.
//!
//! **Where the numbers come from.** `scripts/gptoss_reference_logits.cpp`
//! against a libllama built from `.scratch/llama.cpp` at `5b59b83`.
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_hrm_text_fixture.py \\
//!     crates/frink-models/tests/fixtures/hrm_text_tiny.gguf
//! /tmp/ref_logits crates/frink-models/tests/fixtures/hrm_text_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match_within, graph_caches, kl_vs_golden, load_graph_fixture, worst_vs,
    GRAPH_PROMPT,
};
use frink_models::layer_loops::{HrmStream, LayerLoops, LoopNorm};
use frink_models::norm::NormOp;

const HRM: &str = "hrm_text";
const HRM_DEEP: &str = "hrm_text_deep";

/// The deep schedule's line, and where it comes from.
///
/// Every stack ends with a WEIGHTLESS RMS, so a renormalised residual
/// amplifies the ~1e-7 that two f32 reduction orders differ by, and the
/// amplification grows with the number of stacks. Measured across
/// three depths on arm64: two stacks exact at the suite default (3e-5),
/// three at 3.5e-5, six at 1.6e-4 -- and the six-stack file measured
/// 1.1e-3 on x86_64 CI, which is the same effect with another SIMD
/// width. So the deep fixture is THREE stacks, the shallowest schedule
/// that runs the LOW stack twice in a row, and the line is 5e-4: an
/// order over the arm64 number, comfortably over the x86 one at this
/// depth, and three orders under every sabotage below.
const HRM_DEEP_TOL: f32 = 5e-4;

/// The two-stack fixture's line: the same effect, one stack less.
///
/// It held the suite default (3e-5) against an arm64 NATIVE libllama's
/// golden. The golden is now CI's answer (ADR 0011: a portable AVX2
/// libllama on x86_64), and the two libllama builds alone differ by
/// 2.3e-5 on this fixture. frink then sits 4.0e-5 (arm64) and 4.2e-5
/// (x86_64 CI) from it, on the prefill path only, so the line is 1e-4:
/// over both, and under the deep fixture's 5e-4. Scaling `zL`'s init
/// row by 1.001 (in `frink_models::hrm`) moves this fixture by 7.1e-2.
const HRM_TOL: f32 = 1e-4;

/// llama.cpp's logits for `hrm_text_tiny.gguf` over [`GRAPH_PROMPT`].
const HRM_GOLDEN: [f32; 48] = [
    0.4885881,
    -0.027557671,
    -2.1162999,
    0.8374407,
    -0.40649465,
    -1.29326,
    -2.8178494,
    -0.86931276,
    -0.1435011,
    1.9715679,
    0.14475471,
    1.0473208,
    -2.6040013,
    -1.041673,
    -1.4675782,
    2.7864273,
    1.3373219,
    2.335105,
    -1.4018886,
    0.7970139,
    -0.3718974,
    -2.0812876,
    1.5884306,
    -3.7440865,
    1.1386176,
    -0.546362,
    -2.2635136,
    -1.2015758,
    -1.425969,
    0.044279397,
    -1.6493742,
    0.3337676,
    -1.6249663,
    0.2574932,
    -0.26018596,
    0.5918122,
    0.14686126,
    -0.61769986,
    3.3195477,
    0.047840834,
    0.56373036,
    0.08472836,
    -0.9055227,
    -1.2128551,
    0.53930724,
    2.7570639,
    0.09012517,
    1.6185118,
];

/// The same weights with TWO low passes per cycle.
const HRM_DEEP_GOLDEN: [f32; 48] = [
    0.9075689,
    0.22953193,
    -2.9290028,
    -0.91444284,
    0.34194714,
    -1.0681345,
    0.0669018,
    -1.3104153,
    -0.2717059,
    2.2454932,
    1.481487,
    -0.39871246,
    -0.95129514,
    -0.24828774,
    -0.76835823,
    2.8056111,
    0.2915573,
    -0.7311452,
    -0.98887086,
    -0.19203097,
    -0.10740459,
    -2.5545633,
    1.3897939,
    -2.4129481,
    -0.11540204,
    -0.44590685,
    -2.757229,
    -1.4002708,
    2.7963278,
    -1.2634618,
    0.12008256,
    -1.1157322,
    0.40243834,
    0.32885432,
    -1.8392376,
    -0.839645,
    0.12102103,
    0.79962355,
    2.2496767,
    1.5526245,
    -0.7898988,
    1.5447862,
    -0.9307158,
    -3.330858,
    -0.55150706,
    1.7405634,
    0.48067826,
    1.1693368,
];

#[test]
fn hrm_text_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(HRM, &HRM_GOLDEN, HRM_TOL);
}

/// The number in the report, so it can be regenerated rather than
/// trusted. Run with `--nocapture` to see it.
/// The deep schedule: two LOW passes in a row, so the LOW stack runs
/// twice against two different cache slots.
#[test]
fn the_deep_schedule_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(HRM_DEEP, &HRM_DEEP_GOLDEN, HRM_DEEP_TOL);
}

#[test]
fn report_kl_against_llama_cpp() {
    let d = load_graph_fixture(HRM);
    let mut kv = graph_caches(&d);
    let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    let kl = kl_vs_golden(&got, &HRM_GOLDEN);
    let worst = worst_vs(&got, &HRM_GOLDEN);
    println!("| `hrm_text` | {kl:.2e} | {worst:.2e} |");
    assert!(kl < 1e-8, "hrm_text: KL {kl}");
}

/// The file holds FOUR blocks and the model runs TWELVE slots, each
/// slot with its own KV and the weights aliased by the schedule.
///
/// Structural, so a mapping that ran the wrong stack says which before
/// the numeric test says "diverged".
#[test]
fn twelve_slots_alias_four_blocks_in_the_order_hrm_text_cpp_runs_them() {
    let d = load_graph_fixture(HRM);
    assert_eq!(d.layers.len(), 4, "the file holds 2 * layers_per_stack");
    assert_eq!(d.config.n_layers, 8, "block_count is the slot count");
    assert_eq!(
        d.config.layer_loops,
        Some(LayerLoops::Hrm {
            lps: 2,
            h_cycles: 2,
            l_cycles: 1,
        })
    );
    let physical: Vec<usize> = (0..8)
        .map(|l| {
            d.layers
                .iter()
                .position(|w| std::ptr::eq(w, d.layer_for(l)))
                .expect("every slot aliases one of the blocks")
        })
        .collect();
    assert_eq!(physical, [0, 1, 2, 3, 0, 1, 2, 3], "LOW HIGH LOW HIGH");
    // The deep file is the other order: two LOW passes, then a HIGH.
    let deep = load_graph_fixture(HRM_DEEP);
    assert_eq!(deep.layers.len(), 4);
    assert_eq!(deep.config.n_layers, 6);
    let deep_physical: Vec<usize> = (0..6)
        .map(|l| {
            deep.layers
                .iter()
                .position(|w| std::ptr::eq(w, deep.layer_for(l)))
                .expect("every slot aliases one of the blocks")
        })
        .collect();
    assert_eq!(deep_physical, [0, 1, 0, 1, 2, 3], "LOW LOW HIGH");
    // Every slot has its own cache: the KV is per SLOT even though the
    // weights are per block.
    assert_eq!(d.config.new_kv_caches().len(), 8);
}

/// The stream schedule, and the norms that close each stack.
#[test]
fn every_stack_ends_with_a_weightless_norm_and_writes_one_stream() {
    let d = load_graph_fixture(HRM);
    let loops = d.config.layer_loops.expect("an HRM schedule");
    assert_eq!(
        (0..8)
            .filter(|&l| loops.loop_norm_after(l) == Some(LoopNorm::Weightless))
            .collect::<Vec<_>>(),
        [1, 3, 5, 7],
        "hrm-text.cpp:162 norms at the end of EVERY stack"
    );
    assert_eq!(
        (0..8)
            .filter_map(|l| loops.stream_after(l))
            .collect::<Vec<_>>(),
        [
            HrmStream::Low,
            HrmStream::High,
            HrmStream::Low,
            HrmStream::High,
        ]
    );
    let deep = load_graph_fixture(HRM_DEEP)
        .config
        .layer_loops
        .expect("an HRM schedule");
    assert_eq!(
        (0..12)
            .filter_map(|l| deep.stream_after(l))
            .collect::<Vec<_>>(),
        [
            HrmStream::Low,
            HrmStream::Low,
            HrmStream::High,
            HrmStream::Low,
            HrmStream::Low,
            HrmStream::High,
        ]
    );
    // There is no `output_norm` tensor to read: the last stack's own
    // norm is the final one (`norm_sites::NO_OUTPUT_NORM`).
    assert_eq!(d.final_norm, NormOp::None);
    // And every layer norms without a weight.
    assert!(d.hrm_z_l_init.is_some());
}
