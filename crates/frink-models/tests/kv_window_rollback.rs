//! A cache that evicts behind its sliding window must survive the three
//! things that move a KV cache other than one push at a time: a
//! speculative block that is written and then partly rejected, a fork
//! for `n > 1` completions, and a decode that runs far past the window.
//!
//! `kv_window_eviction.rs` proves plain decoding is token-identical with
//! eviction on. This file is the rest of #61's interactions, on the same
//! kind of fixture: real random weights, a small window so eviction
//! fires within a few tokens, and a TEACHER-FORCED schedule so
//! consecutive keys differ and a one-row hole changes a logit (see that
//! file's `run` for why a free-running random model cannot show one).
//!
//! The window here is the per-layer ARRAY form (`SwaLayers::PerLayer`),
//! the one a file states row by row rather than by period, so the layer
//! a cache is armed for comes from the same lookup a real Gemma-3 or
//! Mellum file drives.
//!
//! Each arm runs the IDENTICAL schedule of batch writes, rejections and
//! single steps; only the window policy differs. Every logit row a
//! committed position produced must then be bit-for-bit equal. That is
//! the whole claim: eviction changes where a row sits, never whether a
//! query reads it.

use frink_core::cache::KvCache;
use frink_models::config::{test_dense_fixture, ModelConfig};
use frink_models::decoder::{Decoder, KvWindowPolicy};
use frink_models::swa_layers::SwaLayers;

const VOCAB: usize = 24;
/// Large enough that the default window keeps a real rollback reserve
/// (slack 6, reserve 3, so a block of 4 may be rejected whole), small
/// enough that 60 positions drain it many times over.
const WINDOW: usize = 12;
/// The file's per-layer answer: two full-attention layers among four
/// windowed ones, not on a period.
const SLIDES: [bool; 6] = [true, true, false, true, false, true];

fn array_swa_config() -> ModelConfig {
    let mut cfg = test_dense_fixture();
    cfg.n_layers = SLIDES.len();
    cfg.n_kv_heads = 2;
    cfg.n_heads = 4;
    cfg.head_dim = 8;
    cfg.hidden_dim = 32;
    cfg.sliding_window = Some(WINDOW);
    cfg.swa_layers = SwaLayers::PerLayer(SLIDES.to_vec().into());
    cfg
}

fn decoder(policy: KvWindowPolicy) -> Decoder {
    let cfg = array_swa_config();
    let mut d = Decoder::new_random_small(cfg, SLIDES.len(), VOCAB);
    d.kv_window = policy;
    d
}

/// A token nothing else in the run would produce at this step, so
/// consecutive keys differ.
fn token(step: usize, salt: usize) -> usize {
    (step * 7 + salt * 5 + 3) % VOCAB
}

/// The rows the next query at `positions` reads on a layer whose window
/// is `WINDOW`: the `WINDOW - 1` behind it. Taken from the CONFIG, not
/// from the cache's own armed window, so a cache that was armed for the
/// wrong window cannot vouch for itself.
fn assert_no_hole(caches: &[KvCache], cfg: &ModelConfig, at: &str) {
    for (l, c) in caches.iter().enumerate() {
        if let Some(w) = cfg.layer_sliding_window(l) {
            let needed = c.positions().min(w - 1);
            assert!(
                c.rows() >= needed,
                "{at}: layer {l} holds {} rows at {} positions, the next query reads {needed}",
                c.rows(),
                c.positions()
            );
        }
    }
}

/// What one speculative run committed, and the logit row behind each
/// committed position.
struct Speculated {
    committed: Vec<usize>,
    logits: Vec<Vec<f32>>,
    caches: Vec<KvCache>,
}

/// The server's and the CLI's speculative shape, driven from a script:
/// write `[anchor] + draft` in one batch, keep `accept` of the draft,
/// truncate the rest, and carry on. `block` is the draft length; the
/// rejection depth cycles through every value from "all rejected" to
/// "all kept" so each phase of the drain meets each depth.
fn speculate(policy: KvWindowPolicy, block: usize, rounds: usize) -> Speculated {
    let d = decoder(policy);
    let cfg = d.config.clone();
    let mut kv = cfg.new_kv_caches();
    let prompt: Vec<usize> = (0..9).map(|i| token(i, 1)).collect();
    let mut logits = vec![d
        .forward_batch(&prompt, 0, &mut kv)
        .pop()
        .expect("non-empty prompt")];
    let mut committed = prompt.clone();
    let mut pos = prompt.len();
    for round in 0..rounds {
        assert!(
            kv.iter().all(|c| c.rollback_limit() >= block),
            "round {round}: a {block}-token draft is longer than the caches can roll back"
        );
        let anchor = token(pos, 2);
        let draft: Vec<usize> = (0..block).map(|i| token(pos + 1 + i, round)).collect();
        let mut batch = vec![anchor];
        batch.extend(&draft);
        let rows = d.forward_batch(&batch, pos, &mut kv);
        let accept = round % (block + 1);
        let keep = pos + 1 + accept;
        for c in kv.iter_mut() {
            assert!(c.can_truncate_to(keep), "round {round}: refused a rollback");
            c.truncate(keep);
        }
        assert_no_hole(&kv, &cfg, &format!("round {round}"));
        committed.push(anchor);
        committed.extend(&draft[..accept]);
        logits.extend(rows.into_iter().take(1 + accept));
        pos = keep;
    }
    Speculated {
        committed,
        logits,
        caches: kv,
    }
}

/// The draft length every speculative caller would use against an
/// evicting decoder: what its caches say, asked the way the callers ask
/// (`KvCache::rollback_limit`). Both arms of a comparison run this one
/// schedule, so the only thing that differs between them is eviction.
fn evicting_rollback_limit() -> usize {
    let d = decoder(KvWindowPolicy::on());
    let mut kv = d.config.new_kv_caches();
    let prompt: Vec<usize> = (0..3 * WINDOW).map(|i| token(i, 1)).collect();
    d.forward_batch(&prompt, 0, &mut kv);
    kv.iter().map(|c| c.rollback_limit()).min().unwrap()
}

/// **Speculative rollback on an evicting cache is lossless.**
///
/// The same schedule of blocks and rejections, with eviction off and
/// on, must commit the same positions and produce bit-identical logits
/// for every one of them. A rollback that left the window one row short
/// shows up here as a logit that moved, because every key in the window
/// is distinct.
///
/// Sabotaged by arming each layer one row narrower than it attends
/// (`KvWindowPolicy::window_for` building `window - 1`), this goes red:
/// the narrower window's reserve lets a rejection leave `window - 2`
/// rows under a query that reads `window - 1`.
#[test]
fn a_rejected_draft_on_an_evicting_cache_reads_what_a_full_cache_reads() {
    let block = evicting_rollback_limit();
    assert!(block >= 2, "the fixture must reject multi-token drafts");
    let rounds = 40;
    let full = speculate(KvWindowPolicy::off(), block, rounds);
    let windowed = speculate(KvWindowPolicy::on(), block, rounds);
    assert_eq!(full.committed, windowed.committed);
    assert_eq!(full.logits.len(), windowed.logits.len());
    for (i, (a, b)) in full.logits.iter().zip(&windowed.logits).enumerate() {
        assert_eq!(a, b, "logit row {i} diverged");
    }
    // And it evicted, or the identity above is a tautology.
    let cfg = array_swa_config();
    let positions = windowed.caches[0].positions();
    assert!(
        positions > 4 * WINDOW,
        "the run must go well past the window"
    );
    for (l, c) in windowed.caches.iter().enumerate() {
        match cfg.layer_sliding_window(l) {
            Some(_) => {
                let w = c.window().expect("a windowed layer is armed");
                assert!(
                    c.rows() <= w.max_rows(),
                    "layer {l}: {} rows is past the bound {}",
                    c.rows(),
                    w.max_rows()
                );
            }
            None => assert_eq!(c.rows(), positions, "layer {l} is full attention"),
        }
    }
}

/// A draft longer than the reserve is the case the callers must not
/// write: the cache says so before the write (`rollback_limit`), and
/// after a write that drained, a rollback that deep is refused rather
/// than performed.
#[test]
fn a_draft_past_the_reserve_is_refused_by_the_cache_not_computed_across() {
    let d = decoder(KvWindowPolicy::on());
    let mut kv = d.config.new_kv_caches();
    let prompt: Vec<usize> = (0..40).map(|i| token(i, 1)).collect();
    d.forward_batch(&prompt, 0, &mut kv);
    let limit = kv.iter().map(|c| c.rollback_limit()).min().unwrap();
    assert!(limit < usize::MAX, "a windowed layer bounds the rollback");
    // Write a block twice the limit and try to reject all of it. The
    // drain after the write may or may not have reached the rows this
    // needs; whichever it is, the cache must answer, and a "yes" must
    // leave no hole.
    let block: Vec<usize> = (0..2 * limit).map(|i| token(40 + i, 3)).collect();
    d.forward_batch(&block, 40, &mut kv);
    let refused = kv.iter().any(|c| !c.can_truncate_to(40));
    assert!(
        refused,
        "a {}-position rollback on a {WINDOW}-window cache must be refused somewhere",
        2 * limit
    );
}

/// **`n > 1` forks of an evicting cache.** The server clones the
/// post-prefill caches once per extra completion and decodes each on
/// its own. A clone must carry its window and its resident rows, so
/// each fork drains independently and reads what a fork of a full
/// cache would.
#[test]
fn forks_of_an_evicting_cache_decode_exactly_like_forks_of_a_full_one() {
    let run = |policy: KvWindowPolicy| -> Vec<Vec<Vec<f32>>> {
        let d = decoder(policy);
        let mut kv = d.config.new_kv_caches();
        let prompt: Vec<usize> = (0..30).map(|i| token(i, 1)).collect();
        d.forward_batch(&prompt, 0, &mut kv);
        let forks: Vec<Vec<KvCache>> = (0..3).map(|_| kv.clone()).collect();
        forks
            .into_iter()
            .enumerate()
            .map(|(f, mut caches)| {
                (0..3 * WINDOW)
                    .map(|s| d.forward_token(token(s, 10 + f), prompt.len() + s, &mut caches))
                    .collect()
            })
            .collect()
    };
    let full = run(KvWindowPolicy::off());
    let windowed = run(KvWindowPolicy::on());
    assert_eq!(
        full, windowed,
        "a fork of an evicting cache read different rows"
    );
    assert_ne!(full[0], full[1], "the forks must decode different scripts");
}
