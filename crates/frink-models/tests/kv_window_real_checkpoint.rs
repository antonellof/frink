//! The eviction identity, on a real checkpoint's weights.
//!
//! `kv_window_eviction.rs` proves it on a synthetic decoder, where every
//! weight is `Lcg` output. This runs the same comparison through
//! gemma-2-2b's real Q4_K_M tensors, its real alternating-SWA layout,
//! its real logit softcap and its real fused dequant-dot kernels -- the
//! things a synthetic F32 decoder does not exercise.
//!
//! # The window is narrowed, on purpose, and this says so
//!
//! Gemma-2's window is 4096. Eviction cannot fire below that, so this
//! test at the checkpoint's own window would need a >4k-token prefill to
//! drop a single row, and would otherwise pass having exercised nothing
//! -- the "the fixture has no sliding layers" trap wearing a real
//! checkpoint as a disguise.
//!
//! The window is a `ModelConfig` number, not a weights number, so both
//! arms are loaded with `sliding_window` set to [`WINDOW`]. That makes
//! the model a different model from the one on disk, and it does NOT
//! make the comparison weaker: what is compared is the same narrowed
//! model with eviction off and on. A row dropped that the kernel reads
//! shows up here exactly as it would at 4096, and 64 positions in rather
//! than 4097.
//!
//! Skipped when the checkpoint is absent (a worktree does not carry the
//! ignored `models/` directory). `FRINK_TEST_GEMMA2_GGUF` overrides the
//! path, which is the same escape hatch `gemma2_metal_quality_gate.rs`
//! offers.
//!
//! # It used to take 21 minutes, and that was the build, not the test
//!
//! This ran for **1,266 seconds** on the M2 Pro and failed twice under
//! the parallel workspace suite while passing on its own. The cause
//! was not the checkpoint: `cargo test` compiled the quantized matmul
//! kernels at opt-level 0. With the numeric crates optimised in the
//! test profile (see the `[profile.test.package.*]` block in the
//! workspace `Cargo.toml`) the same test takes **19.9 seconds**, a 64x
//! difference, and it stays in the default suite rather than being
//! hidden behind `--ignored`.

use std::path::{Path, PathBuf};

use frink_core::cache::KvCache;
use frink_gguf::ShardedGguf;
use frink_models::config::ModelConfig;
use frink_models::decoder::{Decoder, KvWindowPolicy};

/// Small enough that eviction fires inside a short prompt, and still
/// several times the number of decode steps below, so the test is
/// exercising a saturated window rather than a growing one.
const WINDOW: usize = 64;
const PROMPT_LEN: usize = 96;
const STEPS: usize = 48;

fn gguf_path() -> PathBuf {
    if let Ok(p) = std::env::var("FRINK_TEST_GEMMA2_GGUF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-2-2b-it-Q4_K_M.gguf")
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("logits are finite"))
        .map(|(i, _)| i)
        .expect("non-empty logits")
}

/// Varied, deterministic token ids. Consecutive positions must hold
/// DIFFERENT keys or a window one row short reads a row identical to the
/// one it should have read, and nothing is detectable -- the same reason
/// `kv_window_eviction.rs` teacher-forces from a script.
fn ids(count: usize, seed: usize, vocab: usize) -> Vec<usize> {
    (0..count)
        .map(|i| (seed + i * 4099) % (vocab - 1) + 1)
        .collect()
}

struct Run {
    tokens: Vec<usize>,
    logits: Vec<Vec<f32>>,
    caches: Vec<KvCache>,
}

fn run(path: &Path, policy: KvWindowPolicy) -> Run {
    let file = ShardedGguf::open(path).expect("open GGUF");
    let mut config = ModelConfig::from_gguf(&file).expect("model config");
    assert!(
        config.sliding_window.is_some(),
        "this checkpoint has no sliding-window layers, so it proves nothing here"
    );
    config.sliding_window = Some(WINDOW);
    let mut decoder = Decoder::from_gguf(path, config.clone()).expect("load decoder");
    decoder.kv_window = policy;

    let prompt = ids(PROMPT_LEN, 7, config.vocab_size);
    let script = ids(STEPS, 1234, config.vocab_size);
    let mut caches: Vec<KvCache> = config.new_kv_caches();

    let mut last = decoder.forward_batch_last(&prompt, 0, &mut caches);
    let mut logits = vec![last.clone()];
    let mut tokens = vec![argmax(&last)];
    for (step, fed) in script.iter().enumerate() {
        last = decoder.forward_token(*fed, prompt.len() + step, &mut caches);
        tokens.push(argmax(&last));
        logits.push(last.clone());
    }
    Run {
        tokens,
        logits,
        caches,
    }
}

/// Same weights, same tokens, eviction off versus on: identical ids and
/// identical logits.
#[test]
fn evicting_behind_the_window_changes_no_token_on_a_real_checkpoint() {
    let path = gguf_path();
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let without = run(&path, KvWindowPolicy::off());
    let with = run(&path, KvWindowPolicy::on());

    assert_eq!(
        without.tokens, with.tokens,
        "eviction changed the tokens: it dropped a row the kernel reads"
    );
    for (step, (a, b)) in without.logits.iter().zip(with.logits.iter()).enumerate() {
        assert_eq!(a, b, "logits diverged at step {step}");
    }

    // ...and it actually evicted, or the assertions above are a
    // tautology. Gemma-2 alternates, so both halves must be present.
    let total = PROMPT_LEN + STEPS;
    let cfg = {
        let file = ShardedGguf::open(&path).expect("open GGUF");
        let mut c = ModelConfig::from_gguf(&file).expect("model config");
        c.sliding_window = Some(WINDOW);
        c
    };
    let (mut windowed, mut dense) = (0usize, 0usize);
    for (l, cache) in with.caches.iter().enumerate() {
        assert_eq!(cache.positions(), total);
        match cfg.layer_sliding_window(l) {
            Some(_) => {
                windowed += 1;
                let w = cache.window().expect("a windowed layer must be armed");
                assert!(cache.rows() <= w.max_rows(), "layer {l} kept growing");
                assert!(
                    cache.rows() >= WINDOW,
                    "layer {l} kept fewer rows than it reads"
                );
            }
            None => {
                dense += 1;
                assert_eq!(cache.rows(), total, "layer {l} must keep every position");
            }
        }
    }
    assert!(
        windowed > 0 && dense > 0,
        "Gemma-2 alternates; {windowed} windowed and {dense} dense means the \
         SWA pattern was not read"
    );

    // The bytes really went back.
    let evicting: usize = with.caches.iter().map(|c| c.allocated_bytes()).sum();
    let plain: usize = without.caches.iter().map(|c| c.allocated_bytes()).sum();
    assert!(
        evicting < plain,
        "eviction freed nothing: {evicting} vs {plain} bytes"
    );
}
