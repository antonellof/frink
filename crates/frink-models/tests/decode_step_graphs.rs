//! DECODE checked against llama.cpp, step by step: a greedy 32-token
//! generation whose every distribution is libllama's own.
//!
//! # The hole this closes
//!
//! Every other golden in this directory is the logit vector at the LAST
//! position of a six-token prompt. `assert_all_three_paths_match` does
//! drive `forward_token` over that prompt, but six positions is all it
//! ever sees, so a decode-side defect that only shows past position 5 --
//! a KV push skipped once the cache grows, a RoPE position off by one in
//! the single-token body, a window evicted one row early, a recurrent
//! state not carried -- is invisible to all of them, and to `frink
//! parity`, which reads one token and that token is prefill.
//!
//! That is not hypothetical. A Metal device-attention change once made a
//! hybrid model generate FLUENT NONSENSE while parity stayed MATCH; a
//! greedy 100-token generation against the same build with the device
//! path off is what caught it. This suite is that comparison, against
//! the reference rather than against frink itself.
//!
//! # Where the numbers come from
//!
//! `fixtures/decode_steps/<name>.txt` is llama.cpp's greedy decode of
//! the same `GRAPH_PROMPT` every other golden uses: the prompt as ONE
//! batch, then each argmax fed back as its OWN one-token `llama_decode`,
//! so libllama's single-token graph answers every step after the first.
//! F32 KV and flash attention off, as for every golden here.
//!
//! ```text
//! clang++ -std=c++17 -O2 scripts/gptoss_reference_logits.cpp \
//!     -I$LLAMA/include -I$LLAMA/ggml/include \
//!     -L$LLAMA/build/bin -lllama -Wl,-rpath,$LLAMA/build/bin -o /tmp/ref_logits
//! REF_DECODE_STEPS=32 /tmp/ref_logits \
//!     crates/frink-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5 \
//!     > crates/frink-models/tests/fixtures/decode_steps/<name>.txt
//! ```
//!
//! Generated against `libllama.0.4.1` built from `.scratch/llama.cpp` at
//! `5b59b83`; its non-decode output for every fixture below is
//! byte-identical to the last-position golden already committed for it.
//!
//! # What is covered
//!
//! Six fixtures, chosen for where decode can part from prefill: two
//! plain dense GQA graphs (`mistral3`, `glm4`), two sliding windows of 3
//! (`cohere2` dense, `smallthinker_swa2` MoE) so 32 steps cross the
//! window ten times over, and two hybrids whose recurrent state is a
//! reduction over the whole prefix (`qwen35`'s gated delta net,
//! `granitehybrid`'s Mamba-2). Four decode bodies: prefill then the CPU
//! row body (`forward_token`), the row body from position 0, the paged
//! body (`forward_token_paged`), and continuous batching
//! (`forward_multi_seq`) with two sequences at DIFFERENT positions in
//! every batch.
//!
//! Each body free-runs on its own argmax, so a wrong token is reported
//! at the step it happens and every later step inherits it. The
//! goldens' top-1 margins are pinned well above the tolerance
//! (`every_golden_has_a_decisive_argmax`), so a token mismatch is never
//! numeric drift flipping a near tie.
//!
//! `decode_equals_prefill_over_the_generated_sequence` needs no
//! reference: decoding token by token must give what prefilling the
//! whole sequence gives at every position. It runs over more fixtures
//! than the goldens, because it costs nothing to add one.
//!
//! The Metal half is `metal_decode_matches_cpu_and_llama_cpp`
//! (`--features metal -- --ignored`).

mod common;
use common::{argmax, graph_fixture_path, load_graph_fixture, GRAPH_TOL};
use frink_core::cache::PagedKvCache;
use frink_models::Decoder;

/// Greedy steps per golden. 6 + 32 = 38 positions, inside every
/// fixture's `context_length` of 64.
const STEPS: usize = 32;

/// The two hybrids' line. Measured worst over 32 steps and four bodies:
/// 8.0e-5 (`qwen35`), 8.6e-5 (`granitehybrid`), against ~2e-6 at the
/// first step. It is NOT a trend: the per-step worst is spiky and
/// non-monotone (`qwen35` 5.6e-5 at step 12, back to 5.0e-6 at 13), as
/// float32 reduction order amplified through a state that is a
/// reduction over the whole prefix is, and the reference has the same
/// noise in itself -- libllama's own PREFILL of the 37-token sequence
/// differs from its own decode by 1.3e-5 (`qwen35`) and 7.2e-6
/// (`granitehybrid`), measured, where `mistral3`'s differs by 1.5e-7.
/// frink's decode and prefill agree to `GRAPH_TOL` on both
/// (`decode_equals_prefill_over_the_generated_sequence`). 2e-4 is 2.5x
/// the measured worst and three orders under the decode sabotages.
const RECURRENT_DECODE_TOL: f32 = 2e-4;

/// The decode goldens and the line each is held to. The four attention
/// rows sit at `GRAPH_TOL` in their last-position suites and stay there
/// for all 32 steps (worst 1.2e-6); the per-body numbers are in
/// `report_decode_agreement`.
const GOLDENS: &[(&str, f32)] = &[
    ("mistral3", GRAPH_TOL),
    ("glm4", GRAPH_TOL),
    ("cohere2", GRAPH_TOL),
    ("smallthinker_swa2", GRAPH_TOL),
    ("qwen35", RECURRENT_DECODE_TOL),
    ("granitehybrid", RECURRENT_DECODE_TOL),
];

/// Paged block size. Does not divide the six-token prompt, so the
/// first decode step lands mid-block and later ones cross boundaries.
const BLOCK: usize = 4;

struct DecodeGolden {
    prompt: Vec<usize>,
    tokens: Vec<usize>,
    logits: Vec<Vec<f32>>,
}

fn read_golden(name: &str) -> DecodeGolden {
    let path = format!(
        "{}/tests/fixtures/decode_steps/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut prompt = None;
    let mut tokens = None;
    let mut logits = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut words = line.split_whitespace();
        let tag = words.next().unwrap_or_default();
        match tag {
            "prompt" => prompt = Some(words.map(|w| w.parse().unwrap()).collect()),
            "tokens" => tokens = Some(words.map(|w| w.parse().unwrap()).collect()),
            "logits" => logits.push(words.map(|w| w.parse().unwrap()).collect()),
            other => panic!("{path}: unknown line tag {other:?}"),
        }
    }
    let golden = DecodeGolden {
        prompt: prompt.unwrap_or_else(|| panic!("{path}: no prompt line")),
        tokens: tokens.unwrap_or_else(|| panic!("{path}: no tokens line")),
        logits,
    };
    assert_eq!(golden.tokens.len(), STEPS, "{path}: token count");
    assert_eq!(golden.logits.len(), STEPS, "{path}: logit row count");
    assert_eq!(golden.prompt, common::GRAPH_PROMPT, "{path}: prompt");
    golden
}

/// One decode body, free-running for [`STEPS`] greedy steps. Row `i` is
/// the distribution the body's `i`-th token is the argmax of.
type Body = fn(&Decoder, &[usize]) -> Vec<Vec<f32>>;

const BODIES: &[(&str, Body)] = &[
    ("prefill + forward_token", prefill_then_row),
    ("forward_token from position 0", row_only),
    ("forward_token_paged", paged),
    ("forward_multi_seq, two sequences", multi_seq),
];

/// The serving shape: one batched prefill, then the row body.
fn prefill_then_row(d: &Decoder, prompt: &[usize]) -> Vec<Vec<f32>> {
    let mut kv = d.config.new_kv_caches();
    let mut logits = d.forward_batch_last(prompt, 0, &mut kv);
    let mut out = Vec::with_capacity(STEPS);
    for pos in prompt.len()..prompt.len() + STEPS {
        let next = argmax(&logits);
        out.push(std::mem::take(&mut logits));
        if out.len() < STEPS {
            logits = d.forward_token(next, pos, &mut kv);
        }
    }
    out
}

/// The row body for every position, prompt included: no batched kernel
/// anywhere in the run.
fn row_only(d: &Decoder, prompt: &[usize]) -> Vec<Vec<f32>> {
    let mut kv = d.config.new_kv_caches();
    let mut logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        logits = d.forward_token(tok, pos, &mut kv);
    }
    let mut out = Vec::with_capacity(STEPS);
    for pos in prompt.len()..prompt.len() + STEPS {
        let next = argmax(&logits);
        out.push(std::mem::take(&mut logits));
        if out.len() < STEPS {
            logits = d.forward_token(next, pos, &mut kv);
        }
    }
    out
}

/// Paged prefill, then the paged row body.
fn paged(d: &Decoder, prompt: &[usize]) -> Vec<Vec<f32>> {
    let blocks = (prompt.len() + STEPS).div_ceil(BLOCK) + 2;
    let store = d.config.new_paged_kv(BLOCK, blocks);
    let mut kv: Vec<PagedKvCache> = (0..d.layers.len()).map(|_| PagedKvCache::new()).collect();
    let mut logits = d
        .forward_batch_last_paged(prompt, 0, &mut kv, &store)
        .expect("store sized for the whole run");
    let mut out = Vec::with_capacity(STEPS);
    for pos in prompt.len()..prompt.len() + STEPS {
        let next = argmax(&logits);
        out.push(std::mem::take(&mut logits));
        if out.len() < STEPS {
            logits = d
                .forward_token_paged(next, pos, &mut kv, &store)
                .expect("store sized for the whole run");
        }
    }
    out
}

/// How many steps the second sequence starts behind the first.
const LAG: usize = 3;

/// Continuous batching: the same generation twice in one batch, the
/// second sequence admitted [`LAG`] steps late, so every batch holds two
/// rows at DIFFERENT positions with caches of different lengths. Both
/// must reproduce the golden; the first one's rows are returned, and the
/// second's are checked against them here, bit for bit, because
/// batching independent sequences is a scheduling detail, not a math
/// change (`forward_multi_seq`'s contract).
fn multi_seq(d: &Decoder, prompt: &[usize]) -> Vec<Vec<f32>> {
    let mut kv = [d.config.new_kv_caches(), d.config.new_kv_caches()];
    let mut next: [Option<usize>; 2] = [None, None];
    let mut rows: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
    // Sequence 0 is prefilled now; sequence 1 at step LAG.
    let first = d.forward_batch_last(prompt, 0, &mut kv[0]);
    next[0] = Some(argmax(&first));
    rows[0].push(first);
    for step in 1..STEPS + LAG {
        if step == LAG {
            let first = d.forward_batch_last(prompt, 0, &mut kv[1]);
            next[1] = Some(argmax(&first));
            rows[1].push(first);
        }
        let live: Vec<usize> = (0..2)
            .filter(|&s| next[s].is_some() && rows[s].len() < STEPS)
            .collect();
        if live.is_empty() {
            break;
        }
        let tokens: Vec<usize> = live.iter().map(|&s| next[s].unwrap()).collect();
        let positions: Vec<usize> = live
            .iter()
            .map(|&s| prompt.len() + rows[s].len() - 1)
            .collect();
        // `forward_multi_seq` takes one cache set per row of the batch.
        let mut batch_kv: Vec<_> = live.iter().map(|&s| std::mem::take(&mut kv[s])).collect();
        let out = d.forward_multi_seq(&tokens, &positions, &mut batch_kv);
        for ((&s, caches), logits) in live.iter().zip(batch_kv).zip(out) {
            kv[s] = caches;
            next[s] = Some(argmax(&logits));
            rows[s].push(logits);
        }
    }
    assert_eq!(rows[1].len(), STEPS, "the lagged sequence did not finish");
    for (i, (a, b)) in rows[0].iter().zip(&rows[1]).enumerate() {
        assert_eq!(
            a, b,
            "multi-seq: step {i} differs between the two sequences of one batch"
        );
    }
    let [rows0, _] = rows;
    rows0
}

/// The first step at which `got` parts from `want`, or `None`: the
/// token first, because a wrong token is the defect a reader cares
/// about, then the whole distribution at `tol`.
///
/// Returns rather than panics so a test can report EVERY fixture and
/// body that diverges, not only the first one it reached: which bodies
/// a defect reaches is most of the diagnosis.
fn first_divergence(
    got: &[Vec<f32>],
    want: &[Vec<f32>],
    want_tokens: Option<&[usize]>,
    tol: f32,
) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("{} steps, expected {}", got.len(), want.len()));
    }
    for (i, (row, w)) in got.iter().zip(want).enumerate() {
        let pos = common::GRAPH_PROMPT.len() + i;
        if let Some(tokens) = want_tokens {
            let picked = argmax(row);
            if picked != tokens[i] {
                return Some(format!(
                    "step {i} (position {pos}): picked token {picked}, expected {}",
                    tokens[i]
                ));
            }
        }
        let worst = common::worst_vs(row, w);
        if worst.is_nan() || worst > tol {
            return Some(format!(
                "step {i} (position {pos}): max logit delta {worst:.3e} > {tol:.0e}"
            ));
        }
    }
    None
}

fn worst_over_steps(got: &[Vec<f32>], golden: &DecodeGolden) -> f32 {
    got.iter()
        .zip(&golden.logits)
        .map(|(a, b)| common::worst_vs(a, b))
        .fold(0f32, f32::max)
}

#[test]
fn greedy_decode_matches_llama_cpp_on_every_body() {
    let mut failures = Vec::new();
    for &(name, tol) in GOLDENS {
        let golden = read_golden(name);
        let d = load_graph_fixture(name);
        for &(body, run) in BODIES {
            let got = run(&d, &golden.prompt);
            if let Some(why) = first_divergence(&got, &golden.logits, Some(&golden.tokens), tol) {
                failures.push(format!("{name} / {body}: {why}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "greedy decode parted from llama.cpp's:\n  {}",
        failures.join("\n  ")
    );
}

/// The token assertion above is only sound if a tolerance-sized error
/// cannot flip an argmax. Every golden's top-1 must lead its runner-up
/// by far more than its tolerance at every step; a fixture whose
/// generation wanders into a near tie is the wrong fixture for a token
/// check, and this says so instead of letting the suite flake.
#[test]
fn every_golden_has_a_decisive_argmax() {
    for &(name, tol) in GOLDENS {
        let golden = read_golden(name);
        for (i, row) in golden.logits.iter().enumerate() {
            let top = argmax(row);
            assert_eq!(
                top, golden.tokens[i],
                "{name}: step {i}'s token is not its argmax"
            );
            let runner_up = row
                .iter()
                .enumerate()
                .filter(|&(t, _)| t != top)
                .map(|(_, &v)| v)
                .fold(f32::NEG_INFINITY, f32::max);
            let margin = row[top] - runner_up;
            assert!(
                margin > 10.0 * tol,
                "{name}: step {i}'s top-1 margin is {margin}, too close for a token check"
            );
        }
    }
}

/// The worst logit delta per fixture and body, over all 32 steps.
/// `cargo test --test decode_step_graphs report -- --nocapture`.
#[test]
fn report_decode_agreement() {
    for &(name, _) in GOLDENS {
        let golden = read_golden(name);
        let d = load_graph_fixture(name);
        for &(body, run) in BODIES {
            let got = run(&d, &golden.prompt);
            eprintln!(
                "{name:>18} / {body:<34} worst |frink - llama.cpp| over {STEPS} steps = {:.3e}",
                worst_over_steps(&got, &golden)
            );
        }
    }
}

/// Fixtures the self-consistency check runs on beyond the goldens: a
/// cross-section of the norm, residual, attention and FFN seams, each
/// one more place a decode body could drift from the batched one.
const CONSISTENCY_EXTRA: &[&str] = &[
    "mellum",
    "laguna_swa",
    "gptoss",
    "qwen3moe",
    "qwen3next",
    "granite",
    "gemma",
    "exaone4",
    "olmo2",
    "lfm2",
    "mamba2",
    "jamba",
    "falcon_h1",
    "nemotron_h",
    "deci",
    "llama4",
];

/// No reference needed: the row body decoding one token at a time must
/// agree with ONE batched prefill over the prompt plus everything it
/// generated, at every generated position. A KV-cache or state defect
/// on any fixture shows here whether or not it has a decode golden.
#[test]
fn decode_equals_prefill_over_the_generated_sequence() {
    let goldens = GOLDENS.iter().map(|&(name, _)| name);
    let mut failures = Vec::new();
    for name in goldens.chain(CONSISTENCY_EXTRA.iter().copied()) {
        assert!(
            std::path::Path::new(&graph_fixture_path(name)).exists(),
            "{name}: fixture missing"
        );
        let d = load_graph_fixture(name);
        let prompt = common::GRAPH_PROMPT;
        let decoded = row_only(&d, &prompt);
        let mut sequence = prompt.to_vec();
        sequence.extend(decoded.iter().take(STEPS - 1).map(|row| argmax(row)));
        let mut kv = d.config.new_kv_caches();
        let prefilled = d.forward_batch(&sequence, 0, &mut kv);
        assert_eq!(prefilled.len(), sequence.len(), "{name}: prefill row count");
        // Row `i` of the decode is the distribution at position
        // `prompt.len() - 1 + i`.
        let want = &prefilled[prompt.len() - 1..];
        if let Some(why) = first_divergence(&decoded, want, None, GRAPH_TOL) {
            failures.push(format!("{name}: {why}"));
        }
    }
    assert!(
        failures.is_empty(),
        "token-by-token decode parted from one prefill over the same sequence:\n  {}",
        failures.join("\n  ")
    );
}

/// Metal's decode against the CPU's and against llama.cpp, for 32
/// greedy steps.
///
/// Metal is decided once per process (`dense_enabled` caches the
/// switch), so the CPU half runs in a CHILD of this test binary with
/// `FRINK_METAL=0` and hands its rows back in the golden format; this
/// process runs with the device paths on. Both must reproduce
/// llama.cpp's tokens, and Metal must sit within [`METAL_TOL`] of the
/// CPU at every step.
///
/// `cargo test -p frink-models --features metal --test decode_step_graphs -- --ignored --nocapture`
///
/// Run it from a user terminal: a sandboxed or agent shell cannot reach
/// `MTLCompilerService`, so every kernel compile fails and the run
/// falls back to the host or stalls.
#[test]
#[ignore = "needs Apple Metal GPU"]
fn metal_decode_matches_cpu_and_llama_cpp() {
    if let Ok(name) = std::env::var("FRINK_DECODE_STEP_CHILD") {
        // The CPU child: print one body's rows and stop.
        let d = load_graph_fixture(&name);
        let rows = prefill_then_row(&d, &common::GRAPH_PROMPT);
        let tokens: Vec<String> = rows.iter().map(|r| argmax(r).to_string()).collect();
        println!("FRINK-ROWS tokens {}", tokens.join(" "));
        for row in rows {
            let vals: Vec<String> = row.iter().map(|v| format!("{v:e}")).collect();
            println!("FRINK-ROWS logits {}", vals.join(" "));
        }
        return;
    }
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    metal::run();
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;

    /// Device kernels accumulate in their own order and some keep
    /// activations in f16; this is the line that still sits orders of
    /// magnitude under the decode sabotage (> 1e-1).
    pub const METAL_TOL: f32 = 2e-3;

    fn cpu_rows_from_child(name: &str) -> Vec<Vec<f32>> {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args([
                "--ignored",
                "--exact",
                "--nocapture",
                "metal_decode_matches_cpu_and_llama_cpp",
            ])
            .env("FRINK_DECODE_STEP_CHILD", name)
            .env("FRINK_METAL", "0")
            .env("FRINK_METAL_ATTN", "0")
            .output()
            .expect("spawning the CPU child");
        assert!(
            out.status.success(),
            "{name}: CPU child failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.strip_prefix("FRINK-ROWS logits "))
            .map(|l| l.split_whitespace().map(|w| w.parse().unwrap()).collect())
            .collect()
    }

    pub fn run() {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        std::env::set_var("FRINK_METAL", "1");
        std::env::set_var("FRINK_METAL_ATTN", "1");
        // Host lm_head + host argmax: this measures the layer stack, and
        // the rows have to exist to be compared.
        frink_models::set_metal_greedy_argmax(false);
        let mut device_reached = Vec::new();
        let mut failures = Vec::new();
        for &(name, _) in GOLDENS {
            let golden = read_golden(name);
            let cpu = cpu_rows_from_child(name);
            assert_eq!(cpu.len(), STEPS, "{name}: CPU child row count");
            let d = load_graph_fixture(name);
            let mut kv = d.config.new_kv_caches();
            // The serving shape, inline so the caches can be probed after.
            let mut logits = d.forward_batch_last(&golden.prompt, 0, &mut kv);
            let mut gpu = Vec::with_capacity(STEPS);
            for pos in golden.prompt.len()..golden.prompt.len() + STEPS {
                let next = argmax(&logits);
                gpu.push(std::mem::take(&mut logits));
                if gpu.len() < STEPS {
                    logits = d.forward_token(next, pos, &mut kv);
                }
            }
            let reached = d.metal_attn_kv_allocated() || kv.iter().any(|c| c.metal_attn.is_some());
            let worst_cpu = gpu
                .iter()
                .zip(&cpu)
                .map(|(a, b)| common::worst_vs(a, b))
                .fold(0f32, f32::max);
            eprintln!(
                "{name:>18}: device path reached = {reached}, worst |metal - cpu| = {worst_cpu:.3e}, \
                 worst |metal - llama.cpp| = {:.3e}",
                worst_over_steps(&gpu, &golden)
            );
            if let Some(why) = first_divergence(&gpu, &cpu, Some(&golden.tokens), METAL_TOL) {
                failures.push(format!("{name}: Metal vs CPU: {why}"));
            }
            if reached {
                device_reached.push(name);
            }
        }
        assert!(
            failures.is_empty(),
            "Metal decode parted from the CPU's:\n  {}",
            failures.join("\n  ")
        );
        // Without this, a switch that never reached a kernel would make
        // every comparison above CPU against CPU.
        assert!(
            device_reached.contains(&"mistral3"),
            "the plain dense fixture never reached a fused Metal launch; the switch is not \
             live in this process and the comparisons above proved nothing \
             (reached: {device_reached:?})"
        );
    }
}
