//! `Decoder::forward_multi_seq_lanes`: one batched forward in which a
//! sequence may own several rows, which is how the continuous batcher
//! verifies every lane's speculative block at once. Each lane's rows
//! must come out as that lane's block run alone through the
//! single-sequence batch body: the same logits, the same layer inputs,
//! and the same caches afterwards.

mod common;

use common::{graph_caches, load_graph_fixture, tolerance, GRAPH_PROMPT, GRAPH_TOL};
use frink_core::cache::KvCache;
use frink_models::{Decoder, LaneRows, MultiSeqKv};

fn assert_close(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: width");
    let delta = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    // The suite's own line on the CPU; on Metal the two bodies run
    // different kernels, held to the suite's Metal bound.
    let tol = tolerance(GRAPH_TOL);
    assert!(
        delta <= tol,
        "{what}: max |delta| {delta} (tolerance {tol})"
    );
}

/// A prompt prefilled into host rows, as the continuous batcher
/// prefills (`host_kv`): on Metal a plain prefill keeps K/V in the
/// decoder's one device arena and leaves the host rows as placeholders,
/// and the multi-sequence body attends over host rows.
fn prefilled(target: &Decoder, prompt: &[usize]) -> Vec<KvCache> {
    let mut caches = graph_caches(target);
    target
        .forward_batch_last_with_taps(prompt, 0, &mut caches, &[], true)
        .expect("no taps");
    caches
}

fn assert_lanes_match_one_sequence_at_a_time(name: &str) {
    let target = load_graph_fixture(name);
    let layers = [1usize];
    let prompts: [Vec<usize>; 2] = [GRAPH_PROMPT.to_vec(), vec![5, 3, 9, 2]];
    let blocks: [Vec<usize>; 2] = [vec![4, 8, 15], vec![16]];

    // Interleaved on purpose: lane 0, lane 1, lane 0, lane 0.
    let rows = [(0usize, 0usize), (1, 0), (0, 1), (0, 2)];
    let tokens: Vec<usize> = rows.iter().map(|&(l, i)| blocks[l][i]).collect();
    let positions: Vec<usize> = rows.iter().map(|&(l, i)| prompts[l].len() + i).collect();
    let lanes: Vec<usize> = rows.iter().map(|&(l, _)| l).collect();

    let mut batched: Vec<Vec<KvCache>> = prompts.iter().map(|p| prefilled(&target, p)).collect();
    let LaneRows { logits, features } = target
        .forward_multi_seq_lanes(
            &tokens,
            &positions,
            &lanes,
            &mut MultiSeqKv::Contiguous(&mut batched),
            &layers,
        )
        .expect("layer 1 is a valid tap");
    assert_eq!(logits.len(), rows.len());

    for lane in 0..2 {
        let mut alone = prefilled(&target, &prompts[lane]);
        let want = target
            .forward_batch_with_taps(&blocks[lane], prompts[lane].len(), &mut alone, &layers)
            .expect("layer 1 is a valid tap");
        for (r, &(l, i)) in rows.iter().enumerate().filter(|(_, &(l, _))| l == lane) {
            assert_close(
                &format!("{name} lane {l} row {i} logits"),
                &logits[r],
                &want.logits[i],
            );
            assert_close(
                &format!("{name} lane {l} row {i} features"),
                &features[r],
                &want.features[i],
            );
        }
        for (layer, (a, b)) in batched[lane].iter().zip(&alone).enumerate() {
            assert_eq!(
                a.positions(),
                b.positions(),
                "{name} lane {lane} layer {layer}"
            );
            if let (Some(x), Some(y)) = (&a.recurrent, &b.recurrent) {
                assert_close(
                    &format!("{name} lane {lane} layer {layer} state"),
                    &x.ssm,
                    &y.ssm,
                );
            }
        }
        // The caches mean the same thing: the next token over each gives
        // the same answer. Compared through a forward rather than row by
        // row because on Metal each body keeps its rows where its own
        // kernels read them.
        let pos = prompts[lane].len() + blocks[lane].len();
        let mut one_lane = vec![std::mem::take(&mut batched[lane])];
        let LaneRows { logits: next, .. } = target
            .forward_multi_seq_lanes(
                &[23],
                &[pos],
                &[0],
                &mut MultiSeqKv::Contiguous(&mut one_lane),
                &[],
            )
            .expect("no taps");
        let want_next = target
            .forward_batch_with_taps(&[23], pos, &mut alone, &[])
            .expect("no taps");
        assert_close(
            &format!("{name} lane {lane} next token"),
            &next[0],
            &want_next.logits[0],
        );
    }
}

#[test]
fn lanes_of_a_dense_target_match_one_sequence_at_a_time() {
    assert_lanes_match_one_sequence_at_a_time("dflash_target");
}

#[test]
fn lanes_of_a_gated_delta_net_target_match_one_sequence_at_a_time() {
    assert_lanes_match_one_sequence_at_a_time("qwen35");
}

#[test]
fn lanes_of_a_hybrid_mamba_target_match_one_sequence_at_a_time() {
    assert_lanes_match_one_sequence_at_a_time("jamba");
}

/// A sequence's rows out of order would attend a position before it is
/// written; refused rather than answered.
#[test]
#[should_panic(expected = "consecutive positions, in order")]
fn a_lane_whose_rows_skip_a_position_is_refused() {
    let target = load_graph_fixture("dflash_target");
    let mut kv = vec![prefilled(&target, &GRAPH_PROMPT)];
    let n = GRAPH_PROMPT.len();
    let _ = target.forward_multi_seq_lanes(
        &[4, 8],
        &[n, n + 2],
        &[0, 0],
        &mut MultiSeqKv::Contiguous(&mut kv),
        &[],
    );
}
