//! Turning KV-window eviction on must not change a single token.
//!
//! #61 step 2 lets a windowed layer's host `KvCache` drop rows that have
//! fallen behind its window. The argument that this is free is short --
//! a windowed kernel reads the last `window` rows and eviction never
//! drops one of those -- and short arguments about attention are exactly
//! the ones this repo has been wrong about before. So it is measured:
//! the same decoder, the same weights, the same prompt, greedily, with
//! the switch off and on, token for token.
//!
//! # Why a synthetic model and not gemma-2-2b
//!
//! The local alternating-SWA checkpoints have windows of 4096 (Gemma-2)
//! and 1024 (Gemma-3). Eviction cannot fire below that, so a test on one
//! of them would need a >4k-token prompt to prove anything and would
//! otherwise pass while exercising nothing -- the "the fixture has no
//! sliding layers" trap in a different costume. The window is a config
//! number, not a weights number, so this uses a real decoder with real
//! (random) weights and a 6-position window, where eviction fires 7
//! tokens in and every assertion below is about eviction rather than
//! about how long the test is willing to run.
//!
//! `kv_window_real_checkpoint.rs` is the same comparison through
//! gemma-2-2b's real Q4_K_M tensors, narrowing the window for the same
//! reason and skipping when the checkpoint is absent.

use frink_core::cache::KvCache;
use frink_models::config::{test_dense_fixture, ModelConfig};
use frink_models::decoder::{Decoder, KvWindowPolicy};

const N_LAYERS: usize = 6;
const VOCAB: usize = 24;
const WINDOW: usize = 6;
/// Every third layer is full attention, so the test covers both halves
/// of an alternating model: the layers that must evict and the layers
/// that must not.
const SWA_PERIOD: usize = 3;

fn alternating_swa_config() -> ModelConfig {
    let mut cfg = test_dense_fixture();
    cfg.n_layers = N_LAYERS;
    cfg.n_kv_heads = 2;
    cfg.n_heads = 4;
    cfg.head_dim = 8;
    cfg.hidden_dim = 32;
    cfg.sliding_window = Some(WINDOW);
    cfg.swa_layers = frink_models::swa_layers::SwaLayers::period(SWA_PERIOD, false);
    cfg
}

fn caches(cfg: &ModelConfig) -> Vec<KvCache> {
    cfg.new_kv_caches()
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("logits are finite"))
        .map(|(i, _)| i)
        .expect("non-empty logits")
}

/// One decode run: the argmax at every step, every logit vector behind
/// those argmaxes, and the caches it left behind.
struct Run {
    tokens: Vec<usize>,
    /// Every step's full logit vector, not just the argmax.
    ///
    /// The ids are the headline; the logits are the instrument. An
    /// argmax survives changes the distribution did not.
    logits: Vec<Vec<f32>>,
    caches: Vec<KvCache>,
}

/// Prefill through `forward_batch`, then `forward_token` per step --
/// exactly the shape the CLI drives.
///
/// The decode steps are TEACHER-FORCED from a fixed script rather than
/// fed their own argmax, and that is load-bearing rather than
/// convenient. A random-weight model collapses onto one repeated token
/// within a few steps, and once every row inside the window holds the
/// same token at nearly the same state, attending over five of them and
/// over six gives an answer too close to tell apart. A varied script
/// keeps consecutive keys distinguishable, which is the only condition
/// under which "one row short" is detectable at all.
///
/// Sabotaged to keep two rows fewer than the rule allows, this test goes
/// red. Sabotaged by ONE row it does not, and that is not a gap: because
/// eviction runs after the read, keeping `window - 1` rows and then
/// pushing this step's own row leaves exactly `window` -- the correct
/// set. `the_windowed_layers_stop_growing_and_the_dense_ones_do_not`
/// catches the one-row case, by asserting the store's own residency.
fn run(policy: KvWindowPolicy, prompt: &[usize], script: &[usize]) -> Run {
    let cfg = alternating_swa_config();
    let mut decoder = Decoder::new_random_small(cfg.clone(), N_LAYERS, VOCAB);
    decoder.kv_window = policy;
    let mut kv = caches(&decoder.config);

    let prefill = decoder.forward_batch(prompt, 0, &mut kv);
    let mut last = prefill.last().expect("non-empty prompt").clone();
    let mut logits = vec![last.clone()];
    let mut tokens = vec![argmax(&last)];
    for (step, fed) in script.iter().enumerate() {
        last = decoder.forward_token(*fed, prompt.len() + step, &mut kv);
        tokens.push(argmax(&last));
        logits.push(last.clone());
    }
    Run {
        tokens,
        logits,
        caches: kv,
    }
}

/// A varied, deterministic decode script: consecutive positions must
/// hold DIFFERENT keys, or a window that is one row short reads a row
/// identical to the one it should have read and nothing is detectable.
fn script(steps: usize) -> Vec<usize> {
    (0..steps).map(|i| (i * 7 + 5) % VOCAB).collect()
}

fn prompt() -> Vec<usize> {
    (0..11usize).map(|i| (i * 3) % VOCAB).collect()
}

/// **The test that matters.** Same weights, same prompt, temperature 0,
/// switch off versus on: the token ids must be equal.
///
/// Not "close" and not "same perplexity". Eviction either drops rows
/// nothing reads, in which case the arithmetic is untouched and the ids
/// are identical, or it drops a row something reads, in which case the
/// model answers out of a history with a hole in it and no softer
/// assertion would be honest about that.
#[test]
fn evicting_behind_the_window_does_not_change_a_single_token() {
    let prompt = prompt();
    let script = script(40);
    let without = run(KvWindowPolicy::off(), &prompt, &script);
    let with = run(KvWindowPolicy::on(), &prompt, &script);
    assert_eq!(
        without.tokens, with.tokens,
        "eviction changed the generated tokens: it dropped a row the kernel reads"
    );
    // And bit-for-bit on the distributions behind them. See `Run::logits`.
    assert_eq!(without.logits.len(), with.logits.len());
    for (step, (a, b)) in without.logits.iter().zip(with.logits.iter()).enumerate() {
        assert_eq!(a, b, "logits diverged at step {step}");
    }
}

/// **The test that stops the one above passing for the wrong reason.**
///
/// If nothing evicted, the identity above is a tautology. So: the
/// windowed layers' resident rows must have STOPPED GROWING while the
/// position counter kept going, and the full-attention layers must NOT
/// have stopped, because they still read position 0.
#[test]
fn the_windowed_layers_stop_growing_and_the_dense_ones_do_not() {
    let prompt = prompt();
    let steps = 40usize;
    let total = prompt.len() + steps;
    let kv = run(KvWindowPolicy::on(), &prompt, &script(steps)).caches;
    let cfg = alternating_swa_config();

    let mut windowed = 0usize;
    let mut dense = 0usize;
    for (l, cache) in kv.iter().enumerate() {
        assert_eq!(
            cache.positions(),
            total,
            "layer {l} lost track of the sequence length"
        );
        match cfg.layer_sliding_window(l) {
            Some(w) => {
                windowed += 1;
                assert_eq!(w, WINDOW);
                let window = cache.window().expect("a windowed layer must be armed");
                assert!(
                    cache.rows() <= window.max_rows(),
                    "layer {l} holds {} rows after {total} positions; it should have \
                     stopped at {}",
                    cache.rows(),
                    window.max_rows()
                );
                assert!(
                    cache.rows() >= WINDOW,
                    "layer {l} holds {} rows, fewer than the {WINDOW} the kernel reads",
                    cache.rows()
                );
            }
            None => {
                dense += 1;
                assert_eq!(
                    cache.rows(),
                    total,
                    "layer {l} attends over the whole context and must keep all of it"
                );
                assert!(cache.window().is_none());
            }
        }
    }
    assert!(
        windowed > 0 && dense > 0,
        "the fixture must alternate, or this test proves nothing \
         ({windowed} windowed, {dense} dense)"
    );
}

/// `FRINK_KV_WINDOW=0` is byte-for-byte the store this engine had
/// before #61.
#[test]
fn the_off_policy_keeps_every_row_in_every_layer() {
    let prompt = prompt();
    let steps = 40usize;
    let total = prompt.len() + steps;
    let kv = run(KvWindowPolicy::off(), &prompt, &script(steps)).caches;
    for (l, cache) in kv.iter().enumerate() {
        assert!(
            cache.window().is_none(),
            "layer {l} was armed with eviction off"
        );
        assert_eq!(
            cache.rows(),
            total,
            "layer {l} dropped a row with eviction off"
        );
        assert_eq!(cache.positions(), total);
    }
}

/// What the saving actually is, on this fixture: the windowed layers
/// cost a constant instead of a per-position term.
///
/// The number is read off the stores rather than computed, for the same
/// reason `kv_budget`'s acceptance test reads them: a restated rule is
/// how #33 happened.
#[test]
fn a_windowed_layer_costs_a_constant_and_a_dense_one_costs_the_context() {
    let prompt = prompt();
    let steps = 240usize;
    let evicting = run(KvWindowPolicy::on(), &prompt, &script(steps)).caches;
    let plain = run(KvWindowPolicy::off(), &prompt, &script(steps)).caches;

    let bytes = |kv: &[KvCache]| -> usize { kv.iter().map(|c| c.allocated_bytes()).sum() };
    let evicting_bytes = bytes(&evicting);
    let plain_bytes = bytes(&plain);
    assert!(
        evicting_bytes * 2 < plain_bytes,
        "eviction saved less than half: {evicting_bytes} vs {plain_bytes} bytes"
    );
}
