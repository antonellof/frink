//! The BERT-family encoders against llama.cpp's own pooled embedding,
//! on synthetic fixtures: `nomic-bert` and `jina-bert-v3`.
//!
//! # What these rows are
//!
//! `src/models/bert.cpp`'s graph serves several architectures, and the
//! ones frink builds differ from `bert` in two lines of it and
//! nothing else. `bert_gguf_loader::ENCODER_ARCHS` is the table:
//!
//! | arch | rotation | FFN |
//! |---|---|---|
//! | `bert` | learned position table | ungated GELU, both biases |
//! | `nomic-bert` | NEOX RoPE on Q/K | gated SiLU, no biases |
//! | `jina-bert-v3` | NEOX RoPE on Q/K | ungated GELU, both biases |
//!
//! `nomic-bert` differs from `bert` in exactly two lines:
//!
//!   * **RoPE on Q and K** (`:126-133`, NEOX by
//!     `llama_model_rope_type`), where `bert` adds a learned position
//!     table to the embeddings instead (`:90`, gated on
//!     `arch == LLM_ARCH_BERT`). A rotating file carries NO
//!     `position_embd`: measured, libllama's load log never names it
//!     for this architecture, and `bert_gguf_loader` refuses a
//!     rotating file that carries one rather than ignoring it.
//!   * **A gated SiLU FFN** (`:195-201`, the final `else`): `ffn_up`
//!     and `ffn_gate` with no biases, where `bert`'s is an ungated
//!     GELU with both.
//!
//! `bert_encoder::BertFfn` and `BertHparams::rope_theta` are those two
//! facts, read from the architecture at load through
//! `bert_gguf_loader::ENCODER_ARCHS` -- not from tensor presence,
//! because a `bert` file that happens to carry a gate would then run
//! gated here and ungated in llama.cpp.
//!
//! # Why a synthetic fixture and a committed golden
//!
//! `tests/bert_llama_cpp_parity.rs` is the same comparison against a
//! real 36 MB checkpoint and is `#[ignore]`d for that reason. This one
//! runs everywhere: the fixture is 200 KB of F32 in the repo and the
//! golden is what `tools/llama_logits.c --embed` printed for it,
//! through llama.cpp's own MEAN pooling.
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nomic_bert_fixture.py \\
//!     crates/frink-models/tests/fixtures/nomic_bert_tiny.gguf
//! ./target/llama_logits --embed \\
//!     crates/frink-models/tests/fixtures/nomic_bert_tiny.gguf "hello world"
//! ```

use frink_models::EmbeddingModel;

/// llama.cpp's MEAN-pooled embedding for `"hello world"` on
/// `nomic_bert_tiny.gguf`, un-normalized.
const NOMIC_GOLDEN: [f32; 32] = [
    0.32149905,
    0.094289646,
    0.15053454,
    -2.6003315,
    -0.019941427,
    0.049918007,
    -0.075354025,
    1.1375569,
    -0.49217373,
    2.3775523,
    -1.7570381,
    -1.7811859,
    -0.22807966,
    0.95391023,
    0.9663884,
    0.96289575,
    -1.3318931,
    -0.00059386156,
    -0.2390738,
    -0.17903815,
    -0.022602148,
    0.053266555,
    1.3737991,
    0.747592,
    0.09730208,
    -1.1276442,
    -0.6946099,
    0.31933406,
    0.4832091,
    0.8345471,
    2.83744,
    -0.011955578,
];

/// llama.cpp's MEAN-pooled embedding for `"hello world"` on
/// `jina_bert_v3_tiny.gguf`, un-normalized.
const JINA_V3_GOLDEN: [f32; 32] = [
    -0.67560554,
    0.19488943,
    0.47437486,
    -2.4178503,
    -0.30169535,
    -0.31104457,
    0.12225819,
    -1.6278875,
    -0.6008126,
    0.21150663,
    2.5187652,
    -1.4892101,
    -0.47848803,
    -0.28579736,
    0.9464601,
    1.0431744,
    -0.37948397,
    0.18413225,
    1.6074077,
    -2.496784,
    -0.44237632,
    1.1085048,
    1.0128899,
    0.43405753,
    -0.16862528,
    0.31937033,
    -1.0025164,
    -0.060127847,
    1.1103966,
    -0.7716639,
    -0.3838148,
    0.96059936,
];

/// llama.cpp's MEAN-pooled embedding for `"hello world"` on
/// `jina_bert_v2_tiny.gguf`, un-normalized.
const JINA_V2_GOLDEN: [f32; 32] = [
    0.27334905,
    -0.38287723,
    -0.8778778,
    1.4703183,
    1.1156274,
    -0.31832957,
    -1.5785532,
    1.2068794,
    -0.8358101,
    -0.76573783,
    -0.2442192,
    -0.3482085,
    -0.011384547,
    -1.1329051,
    0.96807116,
    0.6340127,
    0.6995489,
    0.16578382,
    -1.0536755,
    0.7068157,
    -1.1811839,
    -1.0217535,
    -0.10069026,
    0.2300063,
    -2.3862777,
    -0.0014972091,
    0.24693376,
    -0.28318134,
    -0.8058681,
    0.21487659,
    -0.81692284,
    -1.3321261,
];

/// llama.cpp's MEAN-pooled embedding for `"hello world"` on
/// `neo_bert_tiny.gguf`, un-normalized.
const NEO_BERT_GOLDEN: [f32; 32] = [
    -0.13721451,
    -0.30953044,
    0.5746242,
    0.48224318,
    -0.5322671,
    -0.33387074,
    0.38777876,
    -0.1777301,
    -0.23012762,
    -0.0012531132,
    -0.9625808,
    -0.6623553,
    0.49542165,
    0.30453932,
    -0.41867602,
    1.64823,
    1.1925495,
    0.4239208,
    0.5657171,
    0.08625776,
    -0.18825471,
    -2.3315008,
    -0.22755218,
    -0.60745573,
    0.46385777,
    0.08840336,
    -0.78944945,
    -0.40569812,
    -0.61793786,
    -1.0149992,
    -0.93395233,
    -0.33015501,
];

/// llama.cpp's MEAN-pooled embedding for `"hello world"` on
/// `eurobert_tiny.gguf`, un-normalized.
const EUROBERT_GOLDEN: [f32; 32] = [
    0.13046771,
    -0.1842036,
    0.072864585,
    0.9002577,
    -1.0929451,
    0.9860673,
    0.17524076,
    -0.326321,
    0.79325575,
    0.37356985,
    -1.8769684,
    -1.020309,
    0.102970585,
    0.31528306,
    -0.14207721,
    1.311766,
    0.24592108,
    0.5444715,
    -0.18846819,
    0.17959613,
    -0.5218657,
    -0.85480857,
    1.7329481,
    2.719172,
    0.06770612,
    -0.5333273,
    -0.74348855,
    0.21054459,
    0.11979509,
    0.530933,
    -0.8243756,
    0.4151523,
];

fn fixture_named(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nomic_bert_tiny.gguf")
}

/// The whole row: RoPE on Q/K, the gated FFN, no position table, and
/// llama.cpp's pooling.
#[test]
fn nomic_bert_matches_llama_cpp_pooled_embedding() {
    let model = EmbeddingModel::from_gguf_path(fixture()).expect("load the nomic fixture");
    let ours = model.embed("hello world", false).expect("embed");
    assert_eq!(ours.len(), NOMIC_GOLDEN.len());
    let mut worst = 0.0f32;
    for (a, b) in ours.iter().zip(&NOMIC_GOLDEN) {
        worst = worst.max((a - b).abs());
    }
    // 5e-3, and the number comes from a bisection rather than from
    // widening until it passed. On this f32 fixture the two engines
    // agree EXACTLY (to six decimals, every element) when the
    // attention output is switched off -- `attn_output.weight = 0`,
    // which leaves the embedding path, both LayerNorms and the gated
    // FFN -- and differ by about 3e-4 per element when it is on. The
    // residual is therefore inside the attention block and not in
    // anything this row added; it survives setting `attn_q.weight = 0`
    // (a uniform softmax, so no Q.K product at all) and is unchanged
    // by rounding K and V to f16 the way a reference that casts them
    // would, so the two obvious explanations are measured and
    // eliminated rather than assumed. `tests/bert_llama_cpp_parity.rs`
    // sees the same magnitude on a real checkpoint and attributes it
    // to Q8_0 activation quantization; this fixture is f32, so that
    // explanation does not cover it and the question is recorded here
    // rather than answered.
    assert!(worst < 5e-3, "max |frink - llama.cpp| = {worst}");
}

/// `jina-bert-v3` is the third architecture on this graph, and it is
/// the OTHER combination: `nomic-bert`'s rotation with `bert`'s
/// ungated GELU FFN.
///
/// Its own tensor loader (`jina-bert-v3.cpp:25-43`) creates no
/// position table and no QK-norm tensors, so the two branches of the
/// shared graph that could have differed are settled by what the file
/// holds -- which is why the row cost one line of
/// `bert_gguf_loader::ENCODER_ARCHS` and a fixture.
#[test]
fn jina_bert_v3_matches_llama_cpp_pooled_embedding() {
    let model = EmbeddingModel::from_gguf_path(fixture_named("jina_bert_v3_tiny.gguf"))
        .expect("load the jina-bert-v3 fixture");
    let hp = model.hparams().expect("a BERT-family encoder");
    assert_eq!(hp.arch, "jina-bert-v3");
    assert_eq!(hp.rope_theta, Some(1000.0), "it rotates");
    assert_eq!(
        hp.ffn,
        frink_models::bert_encoder::BertFfn::GeluSeq,
        "bert.cpp:179-187 includes JINA_BERT_V3 in the ungated GELU arm"
    );
    let ours = model.embed("hello world", false).expect("embed");
    let mut worst = 0.0f32;
    for (a, b) in ours.iter().zip(&JINA_V3_GOLDEN) {
        worst = worst.max((a - b).abs());
    }
    // Wider than the gated row's line for a reason that is measured
    // elsewhere in this tree: this FFN is GELU, and llama.cpp computes
    // GELU from a 65536-entry f16 table
    // (`tests/proj_bias_graphs.rs` carries the same class at 1e-2).
    assert!(worst < 1e-2, "max |frink - llama.cpp| = {worst}");
}

/// `jina-bert-v2` is the row on this graph whose position is neither a
/// table nor a rotation.
///
/// `jina-bert-v2.cpp:5` sets `f_max_alibi_bias = 8.0f` as a LITERAL
/// and `bert.cpp:78-80` builds no `inp_pos` for it, so ALiBi is the
/// only place position enters -- and it is SYMMETRIC here, because
/// `llama-graph.cpp:442` fills a non-causal model's mask with
/// `-|p0 - p1|` where the decoder's is `p_key - p_query`.
///
/// The fixture also carries the two optional norms upstream creates
/// for this architecture and no other on the graph (the whole-
/// projection QK LayerNorm at `bert.cpp:109-123` and the second
/// attention norm at `:156-159`), and the FUSED GEGLU spelling: one
/// `2 * n_ff`-wide `ffn_up` whose first half is the gate.
#[test]
fn jina_bert_v2_matches_llama_cpp_pooled_embedding() {
    let model = EmbeddingModel::from_gguf_path(fixture_named("jina_bert_v2_tiny.gguf"))
        .expect("load the jina-bert-v2 fixture");
    let hp = model.hparams().expect("a BERT-family encoder");
    assert_eq!(hp.arch, "jina-bert-v2");
    assert_eq!(hp.rope_theta, None, "no rotation");
    assert_eq!(
        hp.ffn,
        frink_models::bert_encoder::BertFfn::GegluFusedUp,
        "no ffn_gate tensor, so the gate is fused into ffn_up"
    );
    let slopes = hp.alibi_slopes.as_ref().expect("alibi at a literal 8.0");
    assert_eq!(slopes.len(), 4);
    assert!(slopes[0] > slopes[3], "the slopes decrease per head");

    let ours = model.embed("hello world", false).expect("embed");
    let mut worst = 0.0f32;
    for (a, b) in ours.iter().zip(&JINA_V2_GOLDEN) {
        worst = worst.max((a - b).abs());
    }
    // GELU again, so the GELU-table line rather than the gated row's.
    assert!(worst < 1e-2, "max |frink - llama.cpp| = {worst}");
}

/// The PRE-NORM pair: `neo-bert` and `eurobert` are one topology with
/// three columns between them.
///
/// `neo-bert.cpp:59-118` and `eurobert.cpp:55-114` are the same shape
/// -- RMSNorm BEFORE each block, a bare residual after it, and one
/// final norm where the post-norm rows have one after every add. What
/// differs is the QKV spelling (fused for `neo-bert`, split for
/// `eurobert`), the FFN's (a `2 * n_ff`-wide `ffn_up` against a
/// separate `ffn_gate`), the rotation (NORM against NEOX) and the
/// tensor the final norm is stored under (`enc.output_norm` against
/// `output_norm`). `bert_gguf_loader::EncoderSpec` is those columns.
#[test]
fn the_pre_norm_encoders_match_llama_cpp() {
    for (file, arch, golden) in [
        ("neo_bert_tiny.gguf", "neo-bert", &NEO_BERT_GOLDEN),
        ("eurobert_tiny.gguf", "eurobert", &EUROBERT_GOLDEN),
    ] {
        let model = EmbeddingModel::from_gguf_path(fixture_named(file))
            .unwrap_or_else(|e| panic!("load {arch}: {e}"));
        let hp = model.hparams().expect("a BERT-family encoder");
        assert_eq!(hp.arch, arch);
        assert_eq!(
            hp.topology,
            frink_models::bert_encoder::BertTopology::PreNormRms
        );
        assert_eq!(
            hp.rope_interleaved,
            arch == "neo-bert",
            "llama_model_rope_type answers NORM for neo-bert and NEOX for eurobert"
        );
        let ours = model.embed("hello world", false).expect("embed");
        let mut worst = 0.0f32;
        for (a, b) in ours.iter().zip(golden.iter()) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 5e-3, "{arch}: max |frink - llama.cpp| = {worst}");
    }
}

/// The two facts reached the encoder, and they came from the
/// ARCHITECTURE rather than from which tensors happen to be present.
#[test]
fn the_rotation_and_the_gate_are_read_from_the_architecture() {
    let model = EmbeddingModel::from_gguf_path(fixture()).expect("load the nomic fixture");
    let hp = model.hparams().expect("a BERT-family encoder");
    assert_eq!(hp.arch, "nomic-bert");
    assert_eq!(hp.rope_theta, Some(1000.0), "bert.cpp:126-133 rotates");
    assert_eq!(hp.rope_dim, 8, "the whole head");
    assert_eq!(
        hp.ffn,
        frink_models::bert_encoder::BertFfn::SwigluPar,
        "bert.cpp:195-201 is the gated SiLU"
    );
}
