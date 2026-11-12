//! DFlash drafts against libllama (ADR 0011): the same greedy rounds
//! `scripts/dflash_reference.cpp` runs, replayed through frink's
//! target taps and `DFlashDrafter`, and the speculative loop with a
//! DFlash draft held to plain greedy decoding.
//!
//! The goldens are `tests/fixtures/dflash/*.txt`, written by libllama
//! (`tools/oracle/goldens.tsv`, kind `dflash`). Each round pins the
//! anchor, the drafted tokens and the score each was chosen on, so the
//! injection, the convolution, the block attention and the selector
//! are all inside what is compared.

mod common;

use common::{argmax, graph_caches, graph_fixture_path, load_graph_fixture, GRAPH_PROMPT};
use frink_models::dflash::{DFlashDraft, DFlashDrafter, DFlashError};
use frink_models::sampling::SamplingParams;
use frink_models::speculative::{speculative_decode_with, Drafter, SpeculativeOptions};
use frink_models::Decoder;

/// The lattice scores are sums of a dozen f32 products over a few
/// layers; libllama's and frink's reductions differ in order.
const SCORE_TOL: f32 = 2e-3;

struct Round {
    anchor: usize,
    draft: Vec<usize>,
    scores: Vec<f32>,
}

fn golden(name: &str) -> Vec<Round> {
    let path = format!(
        "{}/tests/fixtures/dflash/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("golden exists");
    text.lines()
        .filter(|l| l.starts_with("round"))
        .map(|l| {
            let (head, scores) = l.split_once(" scores").expect("scores");
            let (head, draft) = head.split_once(" draft").expect("draft");
            let anchor = head.rsplit(' ').next().unwrap().parse().unwrap();
            Round {
                anchor,
                draft: draft
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect(),
                scores: scores
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect(),
            }
        })
        .collect()
}

fn draft(name: &str) -> Result<DFlashDraft, DFlashError> {
    let file = frink_gguf::GgufFile::open(graph_fixture_path(name)).expect("fixture opens");
    DFlashDraft::load(&file)
}

/// `dflash_reference.cpp`'s loop, step for step.
fn replay(target: &Decoder, drafter: &mut DFlashDrafter<'_>, rounds: usize) -> Vec<Round> {
    let layers = drafter.target_layers().to_vec();
    let mut caches = graph_caches(target);
    let prompt = GRAPH_PROMPT.to_vec();
    let t = target
        .forward_batch_with_taps(&prompt, 0, &mut caches, &layers)
        .unwrap();
    drafter.observe_target(0, &t.features);
    let mut anchor = argmax(t.logits.last().unwrap());
    let mut committed = prompt.len();
    let mut out = Vec::new();
    for _ in 0..rounds {
        let block = drafter.draft_block(anchor, usize::MAX);
        let draft: Vec<usize> = block.iter().map(|&(t, _)| t).collect();
        out.push(Round {
            anchor,
            draft: draft.clone(),
            scores: block.iter().map(|&(_, s)| s).collect(),
        });
        let mut verify = vec![anchor];
        verify.extend_from_slice(&draft);
        let t = target
            .forward_batch_with_taps(&verify, committed, &mut caches, &layers)
            .unwrap();
        let accepted = draft
            .iter()
            .zip(&t.logits)
            .take_while(|(&d, l)| argmax(l) == d)
            .count();
        let next = argmax(&t.logits[accepted]);
        for c in caches.iter_mut() {
            c.truncate(committed + 1 + accepted);
        }
        drafter.observe_target(committed, &t.features[..=accepted]);
        committed += 1 + accepted;
        anchor = next;
    }
    out
}

fn assert_matches_libllama(draft_fixture: &str, golden_name: &str) {
    let target = load_graph_fixture("dflash_target");
    let mut drafter = DFlashDrafter::new(draft(draft_fixture).unwrap(), &target, 4, 0.0).unwrap();
    let want = golden(golden_name);
    let got = replay(&target, &mut drafter, want.len());
    for (r, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.anchor, w.anchor, "{golden_name} round {r}: anchor");
        assert_eq!(g.draft, w.draft, "{golden_name} round {r}: drafted tokens");
        for (i, (a, b)) in g.scores.iter().zip(&w.scores).enumerate() {
            assert!(
                (a - b).abs() <= SCORE_TOL,
                "{golden_name} round {r} position {i}: score {a} vs libllama {b}"
            );
        }
    }
}

#[test]
fn dflash2_rounds_match_libllama() {
    assert_matches_libllama("dflash2_draft", "dflash2_draft");
}

#[test]
fn dflash1_rounds_match_libllama() {
    assert_matches_libllama("dflash1_draft", "dflash1_draft");
}

#[test]
fn dflash2_with_a_sliding_window_matches_libllama() {
    assert_matches_libllama("dflash2_swa_draft", "dflash2_swa_draft");
}

/// Plain greedy decoding of `n` tokens after the graph prompt, and the
/// speculative loop with the DFlash2 draft over the same tokens.
fn plain_and_speculative(target: &Decoder, n: usize) -> (Vec<usize>, Vec<usize>, usize, usize) {
    let mut plain = Vec::new();
    let mut caches = graph_caches(target);
    let mut logits = target
        .forward_batch(&GRAPH_PROMPT, 0, &mut caches)
        .pop()
        .unwrap();
    for pos in (GRAPH_PROMPT.len()..).take(n) {
        let t = argmax(&logits);
        plain.push(t);
        logits = target.forward_token(t, pos, &mut caches);
    }

    let mut drafter = DFlashDrafter::new(draft("dflash2_draft").unwrap(), target, 4, 0.0).unwrap();
    let mut caches = graph_caches(target);
    let result = speculative_decode_with(
        target,
        &GRAPH_PROMPT,
        &mut caches,
        &mut drafter,
        &SpeculativeOptions {
            max_new_tokens: n,
            start_pos: 0,
            sampling: SamplingParams::default(),
            seed: 0,
        },
    );
    (
        plain,
        result.generated_tokens,
        result.drafted_tokens,
        result.verification_steps,
    )
}

/// The point of speculation: with a DFlash draft the target's greedy
/// output is token for token the output without one. Checked over
/// enough tokens that blocks are both accepted and rejected.
#[test]
fn speculative_decoding_with_a_dflash_draft_is_greedy_decoding() {
    let target = load_graph_fixture("dflash_target");
    let n = 40;
    let (plain, spec, drafted, steps) = plain_and_speculative(&target, n);
    assert_eq!(spec, plain);
    assert!(drafted > 0, "the draft proposed nothing");
    assert!(
        steps < n,
        "every block was rejected, so the loop never exercised an accepted draft"
    );
}

/// The same on Metal, where a rejected block must also rewind the
/// device KV arena (`Decoder::truncate_kv`): rolling back only the
/// host cache, whose rows Metal prefill leaves as placeholders, made
/// the next verify rebuild the arena from them, and the output left the
/// greedy path two tokens in.
///
/// `cargo test -p frink-models --features metal --test dflash_graphs -- --ignored --exact metal_speculation_rewinds_the_device_kv`
#[test]
#[ignore = "needs Apple Metal GPU"]
fn metal_speculation_rewinds_the_device_kv() {
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    {
        common::pin_backend(common::Backend::Metal);
        let target = load_graph_fixture("dflash_target");
        let (plain, spec, _, steps) = plain_and_speculative(&target, 48);
        assert_eq!(spec, plain);
        assert!(
            steps < 48,
            "no block was accepted, so nothing was rolled back past"
        );
    }
}

#[test]
fn a_draft_whose_width_differs_from_the_target_is_refused_by_name() {
    // The maincoder fixture is 24 wide; the draft reads 32-wide rows.
    let other = load_graph_fixture("maincoder");
    let err = DFlashDrafter::new(draft("dflash2_draft").unwrap(), &other, 4, 0.0)
        .err()
        .expect("refused");
    assert!(err.to_string().contains("does not fit its target"), "{err}");
}

#[test]
fn a_target_is_not_a_draft() {
    let err = draft("dflash_target").err().expect("refused");
    assert!(err.to_string().contains("not a DFlash draft"), "{err}");
}
