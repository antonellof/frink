//! Helpers shared by the integration tests.
//!
//! Each of these had grown two copies. That matters more in a test than
//! in library code: a tolerance or a comparison that drifts between two
//! copies makes one suite quietly weaker than the other, and a test
//! that has stopped checking what it claims to check looks exactly like
//! a test that passes.
//!
//! Rust compiles every file under `tests/` as its own crate, so a
//! helper only one suite uses is dead code in the others. `#[allow]` on
//! the module rather than on each item, since which suite uses what is
//! not a property worth maintaining.

#![allow(dead_code)]

mod backend;
pub mod iq_encode;
pub mod quant_encode;
pub mod synthetic_llama;
#[allow(unused_imports)]
pub use backend::{backend, pin_backend, tolerance, Backend, METAL_GOLDEN_TOL};

use frink_core::cache::KvCache;
use frink_models::{Decoder, ModelConfig};
use std::path::{Path, PathBuf};

/// Fails with the worst absolute difference and the first few values of
/// each side.
///
/// The worst difference rather than the first: a reference mismatch is
/// almost never at index 0, and reporting the first divergence hides
/// how bad it gets. The sample of both sides is what turns "0.03 > 0.01"
/// into a diagnosis.
///
/// `tol` is the CPU line. The comparison runs on this process's pinned
/// [`backend`], which fails here if a decoder was built on a backend
/// nobody pinned; on Metal the line is never tighter than
/// [`METAL_GOLDEN_TOL`], and every comparison prints its worst so the
/// spread can be re-measured with `--nocapture`.
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    let on = backend();
    let tol = tolerance(tol);
    assert_eq!(got.len(), want.len(), "{what}: logit count");
    let worst = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    if on == Backend::Metal {
        eprintln!("metal-golden {worst:.3e} {what}");
    }
    assert!(
        worst <= tol,
        "{what} on {on:?}: max |frink - llama.cpp| = {worst} > {tol}\n  frink: {:?}\n  llama:  {:?}",
        &got[..8.min(got.len())],
        &want[..8.min(want.len())]
    );
}

/// Every `.gguf` under `dir`, recursively.
///
/// A missing or unreadable directory yields nothing rather than
/// failing: these suites scan an optional local model directory, and a
/// machine that has none should skip rather than error.
pub fn collect_gguf(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_gguf(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("gguf") {
            out.push(p);
        }
    }
}

// --- llama.cpp-golden graph fixtures --------------------------------
//
// The harness behind `one_match_arm_graphs.rs` and
// `fixture_away_graphs.rs`. Both suites admit architectures to
// `capability::AUDITED_GENERIC_GQA` on the same evidence -- a tiny
// synthetic GGUF whose golden values come from llama.cpp's own graph via
// libllama -- so they must compare on the same forward paths, at the
// same tolerance, from the same prompt. Two copies of that would be two
// standards, and the weaker one would be invisible.

/// The token ids every graph fixture is driven with. Explicit ids, never
/// tokenized text: the fixtures' vocabularies are placeholders.
pub const GRAPH_PROMPT: [usize; 6] = [3, 7, 11, 19, 23, 5];

/// Float32 accumulation order differs between the two engines (ggml
/// blocks its matmuls; frink does not), so this is a numeric-agreement
/// tolerance, not a bit-exactness claim. The measured worst case across
/// the fixtures was ~1e-6 while `matmul_f32` summed each row in order;
/// since it dots through the four-lane SIMD kernel (2026-09-18) the
/// worst is 1.5e-5, on `nemotron_h_moe`'s F32 router, so the line sits
/// at 3e-5. Every sabotage test moves the outputs by orders of
/// magnitude more.
pub const GRAPH_TOL: f32 = 3e-5;

/// The tolerance for a row whose FFN is **GeGLU**, where the reference
/// itself is the approximate side.
///
/// ggml defines `GGML_GELU_FP16` (`ggml/src/ggml-cpu/vec.h:46`), so
/// llama.cpp's CPU GELU is a 65536-entry **f16 lookup table**: the input
/// is rounded to f16 to index it and the stored value is f16 too
/// (`vec.h:1414-1425`). `GGML_SILU_FP16` is NOT defined, so SiLU is
/// exact SIMD -- which is exactly why every SwiGLU row in these suites
/// agrees to ~1e-6 and the one GeGLU row does not.
///
/// This is the same shape as the K-quant `vec_dot_type` divergence in
/// `docs/plans/llama-cpp-gap-inventory.md` §10: two defensible
/// implementations, one of them the reference's, and the gap is not a
/// frink bug. It was measured rather than assumed. An independent numpy
/// forward pass over the same fixture
/// (`scripts/gemma_fixture_numpy_ref.py`) reproduces llama.cpp's logits
/// to **1.19e-7** when its GELU goes through ggml's table and to
/// **3.93e-5** when its GELU is exact -- the second number being, to the
/// digit, what frink shows.
///
/// 2e-4 is therefore ~5x the measured gap and still orders of magnitude
/// under every sabotage in these suites, each of which moves the logits
/// by more than 1e-2.
pub const GELU_TABLE_TOL: f32 = 2e-4;

/// llama.cpp's logits for `exaone_moe_tiny.gguf` over [`GRAPH_PROMPT`],
/// produced by `scripts/gptoss_reference_logits.cpp` against libllama.
///
/// Shared by two suites rather than copied: `no_rope_layer_graphs`
/// audits the row on it, and `window_array_graphs` re-measures it
/// against two fixtures that differ from the base ONLY in carrying
/// `attention.sliding_window_pattern` as a per-layer array (one
/// agreeing with the seeded period, one inverted). libllama's logits
/// for all three are byte-identical -- `exaone-moe.cpp:7` reads the
/// key through the scalar overload of `get_key_or_arr`, which returns
/// false on an array -- so one constant IS the claim.
pub const EXAONE_MOE_GOLDEN: [f32; 48] = [
    -0.14592841,
    0.07052712,
    -0.18324699,
    -0.7073586,
    -0.08151308,
    0.6084596,
    0.39939362,
    -0.021832988,
    0.020600826,
    -0.13804454,
    0.36290932,
    -0.28110275,
    0.50166076,
    0.17682396,
    0.21900047,
    -0.03235077,
    -0.401246,
    0.6190969,
    0.20589752,
    -0.27477443,
    0.30851653,
    0.3272262,
    0.21583161,
    -0.46494246,
    -0.25845033,
    -0.14444366,
    -0.2289539,
    -0.38887924,
    -0.031140663,
    -0.26675346,
    -0.108890794,
    -0.13819581,
    -0.32289377,
    -0.12842968,
    0.387041,
    0.22891839,
    0.17846167,
    -0.24666634,
    0.09334628,
    -0.36998641,
    0.050943088,
    -0.42619628,
    0.08477548,
    0.22552544,
    0.028861046,
    -0.07179281,
    -0.07399839,
    -0.42189643,
];

pub fn graph_fixture_path(name: &str) -> String {
    format!(
        "{}/tests/fixtures/{name}_tiny.gguf",
        env!("CARGO_MANIFEST_DIR")
    )
}

pub fn load_graph_fixture(name: &str) -> Decoder {
    try_load_graph_fixture(name).unwrap_or_else(|e| panic!("{name}: fixture loads: {e}"))
}

/// Every `<name>_tiny.gguf` graph fixture, by name, sorted.
pub fn graph_fixture_names() -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .expect("tests/fixtures is readable")
        .filter_map(|e| {
            let file = e.ok()?.file_name().into_string().ok()?;
            file.strip_suffix("_tiny.gguf").map(str::to_string)
        })
        .collect();
    names.sort();
    names
}

/// [`load_graph_fixture`] for a sweep over every fixture, some of which
/// are refusals: the loader's error, as text, instead of a panic.
pub fn try_load_graph_fixture(name: &str) -> Result<Decoder, String> {
    let path = graph_fixture_path(name);
    let file = frink_gguf::GgufFile::open(&path).map_err(|e| e.to_string())?;
    let config = ModelConfig::from_gguf(&file).map_err(|e| e.to_string())?;
    backend();
    Decoder::from_gguf(&path, config).map_err(|e| e.to_string())
}

/// The one way a golden suite builds a decoder: on the backend this
/// process is pinned to ([`backend`]), so `--features metal` cannot
/// route a host golden to the GPU without saying so.
pub fn load_decoder(path: impl AsRef<Path>, config: ModelConfig) -> Decoder {
    backend();
    Decoder::from_gguf(path, config).expect("fixture loads")
}

pub fn graph_caches(decoder: &Decoder) -> Vec<KvCache> {
    decoder.config.new_kv_caches()
}

/// Prefill, decode and continuous batching are three separate bodies in
/// this repo, and the reason this helper runs all three on every
/// architecture is that they have diverged before: five model features
/// went missing from one of them while the others kept working.
pub fn assert_all_three_paths_match(name: &str, golden: &[f32]) {
    assert_all_three_paths_match_within(name, golden, GRAPH_TOL);
}

/// The same three paths at a stated tolerance, for the rows where the
/// REFERENCE is the approximate side.
///
/// Parameterised rather than copied: two bodies would be two standards,
/// and the weaker one would be invisible. Only [`GELU_TABLE_TOL`] is
/// ever passed here, and only by the GeGLU rows.
pub fn assert_all_three_paths_match_within(name: &str, golden: &[f32], tol: f32) {
    assert_decoder_matches_on_all_three_paths(&load_graph_fixture(name), golden, tol, name);
}

/// The three paths over a decoder the caller has already built -- one
/// it has attached a LoRA adapter to, say -- rather than one loaded by
/// fixture name. The two named helpers above are this one applied to
/// `load_graph_fixture`, so a fourth path added here reaches every
/// suite at once.
pub fn assert_decoder_matches_on_all_three_paths(
    decoder: &Decoder,
    golden: &[f32],
    tol: f32,
    name: &str,
) {
    let mut kv = graph_caches(decoder);
    assert_close(
        &decoder.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        golden,
        tol,
        &format!("{name}: prefill (forward_batch_last)"),
    );

    let mut kv = graph_caches(decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    assert_close(&out, golden, tol, &format!("{name}: decode"));

    let mut kv = vec![graph_caches(decoder)];
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        let batch = decoder.forward_multi_seq(&[tok], &[pos], &mut kv);
        out = batch.into_iter().next().unwrap();
    }
    assert_close(&out, golden, tol, &format!("{name}: multi-seq"));
}

/// The worst absolute difference from the golden values, for the
/// sabotage tests: a mutation that does not move this is a mutation the
/// suite cannot see.
pub fn worst_vs(got: &[f32], golden: &[f32]) -> f32 {
    got.iter()
        .zip(golden.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

/// `KL(llama.cpp || frink)` over the softmax of the two logit vectors,
/// in nats.
///
/// The max absolute logit difference is what
/// [`assert_all_three_paths_match`] gates on, because it is the sharper
/// instrument on a 48-wide synthetic vocabulary. KL is reported beside
/// it because it is the number `frink parity` speaks in and the one a
/// reader can compare against the K-quant drift in
/// `docs/plans/llama-cpp-gap-inventory.md` §10 -- a comparison that only
/// means anything if both sides are quoted in the same unit.
///
/// Here rather than in each suite: it had grown two copies
/// (`granite_family_graphs`, `no_rope_layer_graphs`), and a helper that
/// drifts between two suites makes one of them quietly report a
/// different number under the same name.
pub fn kl_vs_golden(got: &[f32], golden: &[f32]) -> f64 {
    let softmax = |v: &[f32]| -> Vec<f64> {
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exp: Vec<f64> = v.iter().map(|&x| (x as f64 - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        exp.into_iter().map(|e| e / sum).collect()
    };
    let p = softmax(golden);
    let q = softmax(got);
    p.iter()
        .zip(q.iter())
        .map(|(&p, &q)| if p > 0.0 { p * (p / q).ln() } else { 0.0 })
        .sum()
}

// ---------------------------------------------------------------------
// Real-checkpoint sweeps
//
// Shared by `paged_metal_parity` and `model_swap_isolation`, which both
// greedy-decode one prompt on a real GGUF and compare the token ids two
// runs produce. Each had grown its own copy of the tokenize/stretch
// pair, and a prompt or a BOS rule that drifted between them would have
// made one suite quietly measure something else.
// ---------------------------------------------------------------------

/// Root the real-GGUF suites scan.
///
/// `FRINK_TEST_MODELS_DIR` because a git worktree has no `models/` of
/// its own, and these checks are worth running from one.
pub fn model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("FRINK_TEST_MODELS_DIR") {
        return PathBuf::from(d);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

/// The greedy pick, which is what makes two runs comparable at all.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// Repeat until exactly `want` tokens, keeping at most one leading BOS
/// -- the same stretch `frink verify --prompt-tokens` performs, down
/// to only treating the first token as BOS when it really is one. A
/// checkpoint without BOS otherwise loses its first word, which makes a
/// suite run a different prompt from the one `verify` reports on.
fn stretch(mut tokens: Vec<usize>, want: usize, bos: Option<usize>) -> Vec<usize> {
    if tokens.len() >= want {
        tokens.truncate(want);
        return tokens;
    }
    let leading_bos = (bos.is_some() && tokens.first().copied() == bos).then(|| tokens[0]);
    let body: Vec<usize> = tokens[leading_bos.iter().count()..].to_vec();
    tokens.truncate(leading_bos.iter().count());
    while tokens.len() < want {
        let take = (want - tokens.len()).min(body.len());
        tokens.extend_from_slice(&body[..take]);
    }
    tokens
}

/// `prompt` under this checkpoint's own tokenizer, with its BOS policy
/// applied, stretched to exactly `want` tokens.
///
/// Panics rather than skipping for a tokenizer family it does not
/// cover: a suite that silently ran a different prompt would report a
/// pass it had not earned.
pub fn prompt_tokens(file: &frink_gguf::ShardedGguf, prompt: &str, want: usize) -> Vec<usize> {
    use frink_models::tokenizer::{GgufBpeTokenizer, GgufSpmTokenizer, SpecialTokens};
    let raw: Vec<usize> = match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => GgufBpeTokenizer::from_gguf(file)
            .expect("bpe tokenizer")
            .encode(prompt, SpecialTokens::Parse)
            .into_iter()
            .map(|i| i as usize)
            .collect(),
        Some("llama") => GgufSpmTokenizer::from_gguf(file)
            .expect("spm tokenizer")
            .encode(prompt, SpecialTokens::Parse)
            .into_iter()
            .map(|i| i as usize)
            .collect(),
        other => panic!("this suite does not cover tokenizer {other:?}"),
    };
    let bos = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);
    let mut tokens = raw;
    if frink_models::tokenizer::should_add_bos_token(file) {
        if let Some(b) = bos {
            if tokens.first() != Some(&b) {
                tokens.insert(0, b);
            }
        }
    }
    stretch(tokens, want, bos)
}

/// Streamed experts (`ExpertBacking::Stored`) must compute EXACTLY what
/// resident ones do: prefill as a batch, then decode, at a budget that
/// caches everything and at a 1-byte one where every acquire is an
/// uncached pass-through and the store's buffers are freed and
/// reallocated hardest. Same bytes, same kernels, so `assert_eq!` on
/// the f32 vectors, no tolerance.
///
/// Three MoE bodies, because the backing used to pick between them: a
/// SHORT prefill and decode run the per-position top-k body
/// (`cpu_moe_topk_parallel_slots`, which a streamed layer was denied),
/// and a prefill of 32+ tokens runs the bucketed batch GEMM
/// (`moe_ffn_batch`, resident-only until the same change).
///
/// ONE body for both int-dot arms. `FRINK_CPU_INT_DOT` is read once per
/// process, so the arm a suite tests is decided by which test binary
/// calls this: `gguf_roundtrip` runs the library default (int-dot off,
/// the f32 dequant-dot), `streaming_int_dot` the shipped default (on,
/// the repacked kernels and the repack cache). Only the second can see
/// a repack-cache defect, because the first never repacks anything.
pub fn assert_store_backed_matches_resident(fixture: &str, cfg: fn() -> ModelConfig) {
    // A host property: on Metal the resident experts take the fused MoE
    // kernels and the streamed ones the host, which is two engines.
    pin_backend(Backend::Cpu);
    const DECODE: [usize; 4] = [6, 8, 10, 4];
    let short: Vec<usize> = vec![2, 5, 9, 1, 7, 3];
    // Past `moe_ffn_batch`'s 32-row floor, every token id in range.
    let long: Vec<usize> = (0..40).map(|i| (i * 7 + 3) % 32).collect();
    let run = |d: &Decoder, prompt: &[usize]| -> Vec<Vec<f32>> {
        let vocab = d.config.vocab_size;
        assert!(prompt.iter().chain(DECODE.iter()).all(|&t| t < vocab));
        let mut caches: Vec<KvCache> = d.config.new_kv_caches();
        let mut out = d.forward_batch(prompt, 0, &mut caches);
        for (i, tok) in DECODE.iter().enumerate() {
            out.push(d.forward_token(*tok, prompt.len() + i, &mut caches));
        }
        out
    };

    let resident = Decoder::from_gguf(fixture, cfg()).expect("resident load");
    for prompt in [&short, &long] {
        assert_prompt_matches(&resident, fixture, cfg, prompt, &|d| run(d, prompt));
    }
}

fn assert_prompt_matches(
    resident: &Decoder,
    fixture: &str,
    cfg: fn() -> ModelConfig,
    prompt: &[usize],
    run: &dyn Fn(&Decoder) -> Vec<Vec<f32>>,
) {
    let want = run(resident);
    let n = prompt.len();
    // A quarter of the routed-expert bytes: small enough that the slot
    // pool must evict and REUSE slots, the address-reuse shape the
    // int-dot repack cache once got wrong (see `streaming_int_dot`).
    let expert_bytes: usize = resident
        .layers
        .iter()
        .map(|l| {
            (0..l.moe.n_experts())
                .map(|e| l.moe.expert_bytes(e))
                .sum::<usize>()
        })
        .sum();
    let quarter = (expert_bytes / 4) as u64;
    for budget in [64 * 1024 * 1024u64, quarter, 1u64] {
        let stored = Decoder::from_gguf_with_expert_cache(fixture, cfg(), Some(budget))
            .expect("store-backed load");
        let got = run(&stored);
        let stats = stored
            .expert_store_stats()
            .expect("a cache budget must make the experts store-backed");
        assert!(
            stats.misses > 0,
            "budget={budget}: the store was never read, so nothing was streamed"
        );
        if budget == 1 {
            assert!(
                stats.pass_throughs > 0,
                "budget=1: every acquire should be an uncached pass-through"
            );
        }
        if budget == quarter {
            assert!(
                stats.evictions > 0,
                "a quarter budget must evict, or slot reuse went untested: {stats:?}"
            );
            assert!(stats.peak_resident_bytes <= quarter, "{stats:?}");
        }
        assert_eq!(got.len(), want.len(), "budget={budget}: row count");
        for (row, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let phase = if row < n { "prefill" } else { "decode" };
            assert_eq!(
                g, w,
                "budget={budget}, {n}-token prompt: streamed experts must be \
                 bit-identical to resident at {phase} row {row}"
            );
        }
    }
}
