//! A MoE whose expert budget is a fraction of its expert bytes decodes
//! exactly what the fully resident model decodes (issue #406).
//!
//! The committed MoE fixture (`frink_real_moe_test.gguf`) cannot carry
//! this on its own: it has a shared expert, which keeps every routed
//! layer off the Metal top-k launch, so a Metal run of it never puts a
//! streamed expert on the GPU at all. So this writes its own checkpoint
//! per run: a Mixtral-shaped `llama` with 4 layers of 8 routed experts,
//! top-2, no shared expert, every matrix Q4_0, wide enough (128 x 256)
//! that the Metal launches take it.
//!
//! What each budget is checked for, beyond the logits:
//!
//! - the pool really evicted (a run where everything fit would pass the
//!   comparison while testing no reuse at all);
//! - its resident bytes never passed the budget, at the high-water mark,
//!   not just at the end;
//! - a decode token read at most `top_k` experts per routed layer.
//!
//! CPU: `cargo test -p frink-models --test out_of_core_moe`.
//! Metal: `cargo test -p frink-models --features metal --test
//! out_of_core_moe -- --ignored` (needs a device).

use std::collections::BTreeMap;
use std::path::PathBuf;

use frink_core::cache::KvCache;
use frink_gguf::{GgmlType, GgufValue, GgufWriter, ShardedGguf, TensorPlan};
use frink_models::config::ModelConfig;
use frink_models::decoder::Decoder;

const N_EMBD: usize = 128;
const N_FF: usize = 256;
const N_HEAD: usize = 4;
const N_HEAD_KV: usize = 2;
const HEAD_DIM: usize = N_EMBD / N_HEAD;
const N_LAYER: usize = 4;
const N_EXPERT: usize = 8;
const TOP_K: usize = 2;
const N_VOCAB: usize = 96;

/// Half, a quarter, and one slot's worth. The last is the one that
/// catches a pinned expert being evicted: with one slot, the first
/// expert of a top-k is the only entry when the second one misses, so
/// only the pin keeps the second out of the first's slot. At the larger
/// budgets the hotness policy almost never picks a pinned expert anyway
/// (it was just routed to), which is why they pass that sabotage.
const FRACTIONS: [f64; 3] = [0.5, 0.25, 1.0 / (N_LAYER * N_EXPERT) as f64];

const PROMPT: [usize; 5] = [3, 41, 59, 26, 88];
const DECODE_STEPS: usize = 24;

/// A deterministic, sign-balanced fill.
fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((x >> 40) as f32) / ((1u64 << 24) as f32);
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

/// ggml's `quantize_row_q4_0_ref`: per 32-value block, `d = max / -8`
/// where `max` is the signed value of largest magnitude, then nibbles
/// `min(15, x / d + 8.5)`, low halves first. Q4_0 because it is the
/// format the Metal launches batch, and frink-quant has no encoder for
/// it (nothing in the engine writes Q4_0).
fn q4_0(cols: usize, w: &[f32]) -> Vec<u8> {
    assert_eq!(cols % 32, 0);
    let mut out = Vec::with_capacity(w.len() / 32 * 18);
    for block in w.chunks(32) {
        let max = block
            .iter()
            .copied()
            .fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        let q = |v: f32| ((v * id + 8.5) as u8).min(15);
        for j in 0..16 {
            out.push(q(block[j]) | (q(block[j + 16]) << 4));
        }
    }
    out
}

fn metadata() -> BTreeMap<String, GgufValue> {
    let u = |n: usize| GgufValue::U32(n as u32);
    [
        (
            "general.architecture",
            GgufValue::String("llama".to_string()),
        ),
        ("llama.block_count", u(N_LAYER)),
        ("llama.embedding_length", u(N_EMBD)),
        ("llama.feed_forward_length", u(N_FF)),
        ("llama.attention.head_count", u(N_HEAD)),
        ("llama.attention.head_count_kv", u(N_HEAD_KV)),
        ("llama.attention.key_length", u(HEAD_DIM)),
        ("llama.vocab_size", u(N_VOCAB)),
        ("llama.expert_count", u(N_EXPERT)),
        ("llama.expert_used_count", u(TOP_K)),
        ("llama.rope.freq_base", GgufValue::F32(10_000.0)),
        (
            "llama.attention.layer_norm_rms_epsilon",
            GgufValue::F32(1e-5),
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

/// `(name, GGUF shape fastest-first, bytes, dtype)` for every tensor.
fn tensors() -> Vec<(String, Vec<u64>, Vec<u8>, GgmlType)> {
    let mut seed = 0u64;
    let mut next = || {
        seed += 1;
        seed
    };
    let f32_bytes = |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let norm = |seed: u64| -> Vec<u8> {
        f32_bytes(
            fill(1000 + seed, N_EMBD, 0.2)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect(),
        )
    };
    let mut out = Vec::new();
    let emb = fill(next(), N_VOCAB * N_EMBD, 1.0);
    out.push((
        "token_embd.weight".to_string(),
        vec![N_EMBD as u64, N_VOCAB as u64],
        f32_bytes(emb),
        GgmlType::F32,
    ));
    let kv = N_HEAD_KV * HEAD_DIM;
    for l in 0..N_LAYER {
        out.push((
            format!("blk.{l}.attn_norm.weight"),
            vec![N_EMBD as u64],
            norm(2 * l as u64),
            GgmlType::F32,
        ));
        for (name, rows, cols) in [
            ("attn_q", N_EMBD, N_EMBD),
            ("attn_k", kv, N_EMBD),
            ("attn_v", kv, N_EMBD),
            ("attn_output", N_EMBD, N_EMBD),
        ] {
            let w = fill(next(), rows * cols, 1.0 / (cols as f32).sqrt());
            out.push((
                format!("blk.{l}.{name}.weight"),
                vec![cols as u64, rows as u64],
                q4_0(cols, &w),
                GgmlType::Q4_0,
            ));
        }
        out.push((
            format!("blk.{l}.ffn_norm.weight"),
            vec![N_EMBD as u64],
            norm(2 * l as u64 + 1),
            GgmlType::F32,
        ));
        // A router with a real spread, so routing is not uniform and
        // some experts are hotter than others.
        let router = fill(next(), N_EXPERT * N_EMBD, 3.0 / (N_EMBD as f32).sqrt());
        out.push((
            format!("blk.{l}.ffn_gate_inp.weight"),
            vec![N_EMBD as u64, N_EXPERT as u64],
            f32_bytes(router),
            GgmlType::F32,
        ));
        for (name, rows, cols) in [
            ("ffn_gate_exps", N_FF, N_EMBD),
            ("ffn_up_exps", N_FF, N_EMBD),
            ("ffn_down_exps", N_EMBD, N_FF),
        ] {
            let w = fill(next(), N_EXPERT * rows * cols, 1.0 / (cols as f32).sqrt());
            out.push((
                format!("blk.{l}.{name}.weight"),
                vec![cols as u64, rows as u64, N_EXPERT as u64],
                q4_0(cols, &w),
                GgmlType::Q4_0,
            ));
        }
    }
    out.push((
        "output_norm.weight".to_string(),
        vec![N_EMBD as u64],
        norm(99),
        GgmlType::F32,
    ));
    let head = fill(next(), N_VOCAB * N_EMBD, 1.0 / (N_EMBD as f32).sqrt());
    out.push((
        "output.weight".to_string(),
        vec![N_EMBD as u64, N_VOCAB as u64],
        q4_0(N_EMBD, &head),
        GgmlType::Q4_0,
    ));
    out
}

fn write_fixture(tag: &str) -> PathBuf {
    let tensors = tensors();
    let plan: Vec<TensorPlan> = tensors
        .iter()
        .map(|(name, shape, bytes, dtype)| TensorPlan {
            name: name.clone(),
            shape: shape.clone(),
            dtype: *dtype,
            byte_len: bytes.len(),
        })
        .collect();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    let path = dir.join(format!("out_of_core_moe_{tag}-{}.gguf", std::process::id()));
    let file = std::fs::File::create(&path).expect("create the fixture");
    let mut w = GgufWriter::create(file, &metadata(), plan).expect("write the header");
    for (name, _, bytes, _) in &tensors {
        w.write_tensor(name, bytes).expect("write a tensor");
    }
    w.finish().expect("finish the fixture");
    path
}

fn config(path: &PathBuf) -> ModelConfig {
    ModelConfig::from_gguf(&ShardedGguf::open(path).expect("open")).expect("config")
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .expect("non-empty logits")
}

/// Greedy decode: prefill, then `DECODE_STEPS` single-token steps, each
/// fed the previous argmax. Returns the tokens, every step's logits,
/// and the store misses each decode step cost (empty when resident).
fn greedy(d: &Decoder) -> (Vec<usize>, Vec<Vec<f32>>, Vec<u64>) {
    let mut caches: Vec<KvCache> = d.config.new_kv_caches();
    let mut rows = d.forward_batch(&PROMPT, 0, &mut caches);
    let mut logits = vec![rows.pop().expect("prefill returns a row per token")];
    let mut tokens = vec![argmax(&logits[0])];
    let mut misses = Vec::new();
    for step in 0..DECODE_STEPS {
        let before = d.expert_store_stats().map(|s| s.misses);
        let out = d.forward_token(*tokens.last().unwrap(), PROMPT.len() + step, &mut caches);
        if let (Some(b), Some(after)) = (before, d.expert_store_stats()) {
            misses.push(after.misses - b);
        }
        tokens.push(argmax(&out));
        logits.push(out);
    }
    (tokens, logits, misses)
}

/// Every routed expert's bytes, summed: what a budget is a fraction of.
fn expert_bytes(d: &Decoder) -> usize {
    d.layers
        .iter()
        .map(|l| {
            (0..l.moe.n_experts())
                .map(|e| l.moe.expert_bytes(e))
                .sum::<usize>()
        })
        .sum()
}

/// Loads at `fraction` of the expert bytes and checks the run against
/// `want`. `exact` asks for bit-identical logits; a backend whose
/// resident path runs different kernels from its streamed one is held
/// to identical tokens instead.
fn assert_budget_matches(
    path: &PathBuf,
    want: &(Vec<usize>, Vec<Vec<f32>>, Vec<u64>),
    fraction: f64,
    exact: bool,
) -> Decoder {
    let total = {
        let resident = Decoder::from_gguf(path, config(path)).expect("resident load");
        expert_bytes(&resident)
    };
    let budget = (total as f64 * fraction) as u64;
    let d = Decoder::from_gguf_with_expert_cache(path, config(path), Some(budget))
        .expect("streamed load");
    let store = d
        .expert_store()
        .expect("a budget makes the experts store-backed");
    let slots = store.slot_count().expect("a GGUF store uses the slot pool");
    let n_experts = N_LAYER * N_EXPERT;
    assert!(
        slots < n_experts,
        "fraction {fraction}: {slots} slots hold every one of {n_experts} experts, so nothing \
         would be evicted"
    );

    let got = greedy(&d);
    assert_eq!(got.0, want.0, "fraction {fraction}: greedy tokens differ");
    if exact {
        for (step, (g, w)) in got.1.iter().zip(&want.1).enumerate() {
            assert_eq!(g, w, "fraction {fraction}: logits differ at step {step}");
        }
    }

    let st = d.expert_store_stats().unwrap();
    assert!(
        st.evictions > 0,
        "fraction {fraction}: nothing was evicted: {st:?}"
    );
    assert!(
        st.peak_resident_bytes <= budget,
        "fraction {fraction}: the pool held {} bytes against a budget of {budget}",
        st.peak_resident_bytes
    );
    let bound = (N_LAYER * TOP_K) as u64;
    for (step, m) in got.2.iter().enumerate() {
        assert!(
            *m <= bound,
            "fraction {fraction}: decode step {step} read {m} experts, more than \
             top_k x layers = {bound}"
        );
    }
    eprintln!(
        "fraction {fraction}: {slots}/{n_experts} slots ({}), hits {} misses {} evictions {} \
         refused {} peak {} / budget {budget}",
        store.slot_memory_kind().unwrap_or("heap"),
        st.hits,
        st.misses,
        st.evictions,
        st.refused_admissions,
        st.peak_resident_bytes
    );
    d
}

/// On the host the streamed and resident layers run the same kernels
/// over the same bytes, so the logits are equal bit for bit at a half
/// and at a quarter of the expert bytes.
///
/// Sabotage: in `ExpertStore::victim`, drop the
/// `Arc::strong_count(&e.data) == 1` filter, so a pinned expert (one
/// this very top-k is reading) can be evicted and its slot rewritten
/// by the next miss. The logits go wrong at step 0 of the one-slot
/// budget (checked: `fraction 0.03125: logits differ at step 0`).
#[test]
fn a_quarter_of_the_expert_bytes_decodes_the_resident_tokens_on_the_host() {
    let path = write_fixture("cpu");
    let resident = Decoder::from_gguf(&path, config(&path)).expect("resident load");
    let want = greedy(&resident);
    let distinct: std::collections::BTreeSet<_> = want.0.iter().collect();
    assert!(
        distinct.len() > 1,
        "a constant greedy run would compare nothing"
    );
    for fraction in FRACTIONS {
        assert_budget_matches(&path, &want, fraction, !cfg!(feature = "metal"));
    }
    std::fs::remove_file(&path).ok();
}

/// The same on Metal, where the streamed experts live in a Shared
/// `MTLBuffer` slot slab and the fused top-k launch binds it in place.
///
/// The resident model takes the packed `mul_mv_id` stack, which routes
/// on the GPU and runs different kernels from the per-expert top-k a
/// streamed layer takes, so resident-versus-streamed is held to the
/// same TOKENS. Streamed-versus-streamed is held to the same LOGITS:
/// a full budget and a quarter budget run identical kernels, so any
/// difference between them is the pool handing a kernel the wrong
/// bytes.
#[test]
#[ignore = "needs a Metal device"]
#[cfg(feature = "metal")]
fn a_quarter_of_the_expert_bytes_decodes_the_resident_tokens_on_metal() {
    // SAFETY: set before any decode in this test binary; the CPU test
    // above reads no Metal variable, and Metal's backend probe reads
    // these once.
    unsafe {
        std::env::set_var("FRINK_METAL", "1");
        std::env::set_var("FRINK_METAL_ATTN", "1");
    }
    let path = write_fixture("metal");
    let resident = Decoder::from_gguf(&path, config(&path)).expect("resident load");
    let want = greedy(&resident);

    let full = {
        let d = Decoder::from_gguf_with_expert_cache(&path, config(&path), Some(1 << 30))
            .expect("full-budget streamed load");
        assert_eq!(
            d.expert_store().and_then(|s| s.slot_memory_kind()),
            Some("metal"),
            "on Metal the slot pool must be the Shared MTLBuffer slab"
        );
        greedy(&d)
    };
    assert_eq!(
        full.0, want.0,
        "streamed tokens differ from resident on Metal"
    );

    for fraction in FRACTIONS {
        let binds = frink_metal::expert_slab::slab_binds();
        let d = assert_budget_matches(&path, &full, fraction, true);
        assert_eq!(
            d.expert_store().and_then(|s| s.slot_memory_kind()),
            Some("metal")
        );
        // The experts ran on the GPU, from the slab, without a copy: a
        // host fallback, or a launch that uploaded the bytes, binds
        // nothing here.
        assert!(
            frink_metal::expert_slab::slab_binds() > binds,
            "fraction {fraction}: no Metal launch bound the slot slab"
        );
    }
    std::fs::remove_file(&path).ok();
}
