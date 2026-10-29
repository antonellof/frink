//! The FIXTURE-AWAY architectures, checked against llama.cpp itself.
//!
//! `capability.rs` triaged every unaudited refusal into four classes.
//! FIXTURE-AWAY is the class that says frink ALREADY builds this
//! architecture's graph and what is missing is EVIDENCE. That claim is
//! cheap to make and expensive to be wrong about, so this suite makes it
//! the same way `one_match_arm_graphs.rs` does: a tiny synthetic GGUF
//! whose golden values come from **llama.cpp's own graph** via a real
//! `libllama`, compared on all three forward paths.
//!
//! | arch | RoPE | what its fixture pins beyond the plain graph |
//! |---|---|---|
//! | `internlm2` | NORM | the optional q/k/v projection biases |
//! | `xverse` | NORM | nothing else -- it is llama under another name |
//! | `ernie4_5` | NORM | `head_dim != n_embd / n_head` |
//! | `baichuan` | NORM | 32 layers, because llama.cpp picks ALiBi off the layer count |
//! | `exaone` | NEOX | a tied lm_head, and `head_dim != n_embd / n_head` |
//! | `gemma` | NEOX | the sqrt(n_embd) embedding scale, GeGLU, a tied lm_head |
//! | `bailingmoe2` | NEOX | fused QKV, per-head QK norm before RoPE, sigmoid-gated MoE |
//! | `plamo3` | NEOX | sandwich norms, a real sliding window, and a tensor name nobody else spells |
//!
//! One of the seven verdicts was WRONG, and building its fixture is what
//! found that. `plamo3` is the only architecture upstream that creates
//! `ATTN_POST_NORM` / `FFN_POST_NORM` through the two-argument `LLM_TN`
//! overload (`plamo3.cpp:52,55`), so it asks for
//! `blk.N.post_attention_norm` with NO `.weight`, and gguf-py emits
//! exactly that for it. frink read only the suffixed spelling, so it
//! could never have loaded a real PLaMo-3 checkpoint -- fail-closed, but
//! not "a fixture away". `load_norm_vec_either_spelling` in `loader.rs`
//! is the arm; this suite is the evidence it is the right one.
//!
//! **Every row was checked against the C on the six facts this repo has
//! lost at least once**, and every one of the six is asserted below
//! rather than merely read:
//!
//! 1. **RoPE variant.** `llama_model_rope_type` (llama-model.cpp) puts
//!    `internlm2` (:2579), `xverse` (:2581), `baichuan` (:2577) and
//!    `ernie4_5` (:2602) in the NORM group and `exaone` (:2655) in the
//!    NEOX group. `the_rope_variant_each_architecture_uses_is_the_one_llama_cpp_uses`
//!    pins the resolved layout and
//!    `rotating_the_wrong_pairs_diverges_from_llama_cpp` proves each
//!    fixture can see a flip. Getting this wrong is the Llama-3.1-8B
//!    wrong-output bug.
//! 2. **SWA pattern and phase.** Six of the seven read
//!    `LLM_KV_ATTENTION_SLIDING_WINDOW` nowhere at all:
//!    `internlm2.cpp:3-11`, `xverse.cpp:3-12`, `baichuan.cpp:3-15`,
//!    `ernie4-5.cpp:3-21`, `exaone.cpp:3-10` and `bailingmoe2.cpp:3-21`
//!    are the complete `load_arch_hparams` bodies and none mentions a
//!    window, so `hparams.swa_type` stays `LLAMA_SWA_TYPE_NONE` and
//!    there is no phase to get wrong. `plamo3` is the exception and
//!    gets the full treatment: `plamo3.cpp:5-11` reads the window and a
//!    scalar `sliding_window_pattern` and calls `set_swa_pattern` with
//!    `dense_first = false`, its fixture drives a period of 2 out of the
//!    file over four layers with a window NARROWER than the prompt, and
//!    the period, the phase and the window each have their own sabotage
//!    test.
//! 3. **`attention_scale`.** All seven pass a literal
//!    `1.0f/sqrtf(float(n_embd_head))` to `build_attn`
//!    (internlm2.cpp:92, xverse.cpp:90, baichuan.cpp:103,
//!    ernie4-5.cpp:120, exaone.cpp:93, bailingmoe2.cpp:141,
//!    plamo3.cpp:140) and none reads `LLM_KV_ATTENTION_SCALE`, so
//!    `ModelConfig::attention_scale` must stay `None` and let the
//!    kernels' own scale stand.
//! 4. **`post_attn_norm`** and **5. `post_ffn_norm`.** Six of the seven
//!    create neither `LLM_TENSOR_ATTN_POST_NORM` nor
//!    `LLM_TENSOR_FFN_POST_NORM`; each layer has exactly `attn_norm` and
//!    `ffn_norm`, both applied BEFORE their branch on a sequential
//!    residual. `plamo3` has BOTH, applied to their branch's OUTPUT
//!    before the residual add (:152-155, :171-174) -- the Gemma-2
//!    placement, which is where frink applies them -- and dropping
//!    either one has its own sabotage test.
//! 6. **QK-norm and its order.** None of the five DENSE rows creates
//!    `attn_q_norm` / `attn_k_norm`, so there is no ordering question
//!    for them. `bailingmoe2` does: `bailingmoe2.cpp:52-53` creates
//!    per-head norms of width `n_embd_head_k` and :123-135 applies them
//!    BEFORE `ggml_rope_ext`, the opposite of `maincoder` and
//!    `hunyuan-moe` next door. Its fixture's norm weights are centred
//!    near 1.5 so the two orders are far apart, and
//!    `norming_bailingmoe2_after_rope_instead_of_before_diverges_from_llama_cpp`
//!    proves it.
//!
//! The MoE half of the checklist -- the gating function, and whether the
//! top-k weights are renormalised after selection -- arises for
//! `bailingmoe2` alone (`plamo3` is dense). Both are REQUIRED-or-read metadata keys there
//! (`bailingmoe2.cpp:10-11`), the fixture carries both, and
//! `bailingmoe2_reads_its_routing_out_of_the_file_rather_than_guessing`
//! asserts what they resolved to. It matters because frink's
//! architecture-name fallback would default this row to SOFTMAX. For
//! the five dense rows the question does not arise at all, and
//! `only_bailingmoe2_is_moe` pins that rather than leaving it implied.
//!
//! **Where the numbers come from.** Each `GOLDEN` array was produced by
//! running llama.cpp's own graph for that architecture over the same
//! fixture file, through `scripts/gptoss_reference_logits.cpp` linked
//! against a real `libllama`. Not by re-reading a spec, and not by
//! frink checking itself.
//!
//! Regenerating (both halves must be redone together if a fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_xverse_fixture.py \
//!     crates/frink-models/tests/fixtures/xverse_tiny.gguf
//! clang++ -std=c++17 -O2 scripts/gptoss_reference_logits.cpp \
//!     -I$LLAMA/include -I$LLAMA/ggml/include -L$BUILD/bin -lllama \
//!     -Wl,-rpath,$BUILD/bin -o /tmp/ref_logits
//! /tmp/ref_logits crates/frink-models/tests/fixtures/xverse_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_all_three_paths_match_within, graph_caches,
    graph_fixture_path, load_graph_fixture, worst_vs, GELU_TABLE_TOL, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::QkNormStyle;
use frink_models::swa_layers::SwaLayers;
use frink_models::{ModelConfig, RopeLayout};
use frink_moe::GatingFunction;

/// The rows whose llama.cpp graph has no QK-norm, no post-norms, no
/// window and no experts. Named once so the four checklist tests below
/// cannot drift apart about who is in the set.
const DENSE_ROWS: [&str; 6] = [
    "internlm2",
    "xverse",
    "ernie4_5",
    "baichuan",
    "exaone",
    "gemma",
];

/// Every row this suite admits.
const ALL_ROWS: [&str; 8] = [
    "internlm2",
    "xverse",
    "ernie4_5",
    "baichuan",
    "exaone",
    "gemma",
    "bailingmoe2",
    "plamo3",
];

/// The rows with no sliding window and no post-norms. `plamo3` has
/// both, so it is excluded here and pinned by its own tests instead.
const NO_WINDOW_NO_POST_NORM_ROWS: [&str; 7] = [
    "internlm2",
    "xverse",
    "ernie4_5",
    "baichuan",
    "exaone",
    "gemma",
    "bailingmoe2",
];

// --- internlm2 -----------------------------------------------------
//
// `src/models/internlm2.cpp` in full: `load_arch_hparams` (:3-11) reads
// only the RMS epsilon, `load_arch_tensors` (:13-35) creates attn_norm,
// split Q/K/V through `create_tensor_qkv`, attn_output, ffn_norm and
// gate/up/down, and the graph (:59-122) is `x + attn(rms(x))` then
// `y + ffn(rms(y))` with `LLM_FFN_SILU, LLM_FFN_PAR` SwiGLU. The
// fixture additionally carries the OPTIONAL q/k/v biases
// `create_tensor_qkv` allows (llama-model.cpp:2897-2899), which real
// InternLM2 exports ship and which frink loads generically.

const INTERNLM2_GOLDEN: [f32; 48] = [
    -0.38497463,
    -0.411406,
    -0.23690121,
    -0.079736516,
    -0.086446024,
    0.034069862,
    -0.18947236,
    0.31578812,
    -0.009138606,
    0.23619087,
    0.29375088,
    0.09236469,
    0.46713042,
    -0.38761902,
    0.8017329,
    -0.24398606,
    -0.7430083,
    -0.08430417,
    -0.11138179,
    0.09992017,
    0.2612124,
    -0.4456048,
    0.19681671,
    -0.09016396,
    0.044498637,
    -0.27157593,
    -0.62290406,
    -0.017267626,
    0.33660623,
    -0.19604379,
    0.28323877,
    -0.5600808,
    0.15880197,
    0.097796604,
    -0.31987217,
    -0.17884886,
    -0.11307311,
    -0.035523556,
    -0.37471962,
    -0.2398659,
    0.051130064,
    -0.28694645,
    0.17326327,
    -0.12214558,
    -0.06344788,
    -0.07479449,
    -0.2589051,
    0.2516519,
];

#[test]
fn internlm2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("internlm2", &INTERNLM2_GOLDEN);
}

/// The biases are in the file, are non-zero, and are loaded.
///
/// Without this the golden comparison would pass just as well against a
/// fixture that had none, and the claim that frink handles InternLM2's
/// real tensor set would rest on a file that did not have it.
#[test]
fn internlm2_loads_the_optional_qkv_biases_its_file_carries() {
    let d = load_graph_fixture("internlm2");
    for (il, layer) in d.layers.iter().enumerate() {
        for (what, bias) in [
            ("q", &layer.attn.q_bias),
            ("k", &layer.attn.k_bias),
            ("v", &layer.attn.v_bias),
        ] {
            let bias = bias
                .as_ref()
                .unwrap_or_else(|| panic!("blk.{il}: attn_{what}.bias must be loaded"));
            assert!(
                bias.iter().any(|b| *b != 0.0),
                "blk.{il}: attn_{what}.bias is all zeros; it could not fail"
            );
        }
    }
}

// --- xverse --------------------------------------------------------
//
// `src/models/xverse.cpp` is llama under a different name: :3-12 reads
// only the RMS epsilon, :14-35 is the plain tensor set, :59-121 is the
// sequential residual with SiLU SwiGLU. There is deliberately nothing
// extra in this fixture -- the row's whole content is that the graph is
// the generic one, and the evidence for that is the comparison itself.

const XVERSE_GOLDEN: [f32; 48] = [
    -0.06578407,
    0.35991532,
    0.09296507,
    0.0500691,
    0.013582745,
    0.50010747,
    -0.20849259,
    0.053211644,
    0.057681076,
    0.1302229,
    -0.409114,
    0.24452712,
    0.39229798,
    -0.24814555,
    -0.19971883,
    -0.10270727,
    0.36186606,
    0.24045944,
    0.1772546,
    0.38434425,
    0.116239354,
    -0.1490037,
    -0.16259417,
    0.07176717,
    0.27208632,
    -0.1671407,
    0.032589324,
    -0.13619582,
    0.006233629,
    0.29767945,
    0.16670844,
    -0.28568843,
    -0.07653904,
    0.18598174,
    -0.036290076,
    -0.3572907,
    -0.10366354,
    0.472095,
    -0.385141,
    0.06120664,
    0.05274259,
    -0.39883983,
    -0.22407724,
    0.3536656,
    0.004656989,
    -0.14421564,
    0.42171967,
    -0.18553367,
];

#[test]
fn xverse_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("xverse", &XVERSE_GOLDEN);
}

// --- ernie4_5 (dense) ----------------------------------------------
//
// `src/models/ernie4-5.cpp`'s dense branch (:64-68 for the tensors,
// :95-149 for the graph). Its Q projection and `attn_output` are sized
// from `n_embd_head_k * n_head` (:41-42) rather than from n_embd, so
// head_dim may disagree with `n_embd / n_head`, and this fixture makes
// it disagree (8 against 6).
//
// DELIBERATELY ABSENT from the fixture: the OPTIONAL `attn_output.bias`
// at :45. frink has no slot for one outside the gpt-oss path and
// refuses a checkpoint carrying it BY NAME through the unread-tensor
// gate, which is the correct behaviour and is why the fixture must not
// have one. `ernie4_5-moe` is a separate row and still refuses.

const ERNIE4_5_GOLDEN: [f32; 48] = [
    0.03883054,
    -0.22302249,
    -0.17906801,
    0.15218028,
    -0.02055931,
    -0.070127405,
    0.1165026,
    -0.034907892,
    -0.12018872,
    -0.048992664,
    0.016463999,
    0.08155338,
    -0.07699711,
    -0.12543437,
    0.28277662,
    0.38459194,
    0.13750306,
    -0.41823462,
    -0.043292582,
    -0.12194319,
    -0.09083214,
    0.071103305,
    0.01813967,
    0.2717319,
    0.031989187,
    -0.024477273,
    0.18046162,
    -0.09042182,
    -0.22470418,
    0.2138165,
    -0.22854619,
    0.008390563,
    -0.12621851,
    -0.4525338,
    0.47459865,
    -0.22544149,
    0.48321664,
    0.0048647877,
    -0.038895264,
    -0.11535422,
    -0.048806466,
    0.031679325,
    0.103515804,
    0.15141746,
    -0.08237043,
    0.10312017,
    -0.1163362,
    -0.32099646,
];

#[test]
fn ernie4_5_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("ernie4_5", &ERNIE4_5_GOLDEN);
}

/// head_dim comes from `attention.key_length`, not from n_embd/n_head.
#[test]
fn ernie4_5_reads_its_head_dim_from_the_file_rather_than_deriving_it() {
    let d = load_graph_fixture("ernie4_5");
    assert_eq!(d.config.hidden_dim, 24);
    assert_eq!(d.config.n_heads, 4);
    // 6 is what dividing would give.
    assert_eq!(d.config.head_dim, 8);
}

// --- baichuan (7B) -------------------------------------------------
//
// One architecture string, two models. `src/models/baichuan.cpp:5-14`
// picks the variant off the LAYER COUNT -- 32 is 7B, 40 is 13B and sets
// `f_max_alibi_bias = 8.0f` -- with its own comment "TODO: become GGUF
// KV parameter". The graph then builds positions only for 7B (:58) and
// applies `ggml_rope_ext` only for 7B (:77-95), so a 13B or an
// unrecognised layer count gets NO rotation at all.
//
// That is why this fixture has 32 layers and not 2: a 2-layer file is
// `LLM_TYPE_UNKNOWN`, falls into the no-RoPE arm at :91, and would be
// evidence about a graph no real checkpoint runs. The 13B stays refused
// by name in `loader.rs` on `block_count == 40`.

const BAICHUAN_GOLDEN: [f32; 48] = [
    0.056164384,
    0.05335981,
    0.034717236,
    -0.016637973,
    -0.1681825,
    -0.19657308,
    -0.093710594,
    0.12201875,
    0.202757,
    0.061601378,
    0.0852424,
    0.017296217,
    -0.13683479,
    0.019249436,
    -0.0067887288,
    0.030473292,
    -0.0056031533,
    0.019746449,
    0.040556442,
    -0.13934176,
    -0.00078091025,
    0.067291364,
    0.04678838,
    -0.04069045,
    -0.07528572,
    0.09840066,
    0.027413199,
    -0.012988582,
    -0.0071283206,
    0.10830741,
    0.046693385,
    0.23345774,
    0.10834331,
    -0.017043814,
    0.04738696,
    -0.09828674,
    -0.018817,
    0.037317395,
    0.10740893,
    -0.086651094,
    0.16450927,
    0.11803734,
    0.07888082,
    0.051779855,
    -0.087531194,
    -0.07406292,
    0.008599423,
    0.0709973,
];

#[test]
fn baichuan_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("baichuan", &BAICHUAN_GOLDEN);
}

/// The fixture is the 7B, and the file says so the only way llama.cpp
/// reads it.
///
/// A regenerated fixture with fewer layers would still load and still
/// compare -- against a reference that had silently stopped rotating.
/// This is the assertion that makes that impossible to miss.
#[test]
fn the_baichuan_fixture_has_the_32_layers_that_select_the_rotating_variant() {
    let d = load_graph_fixture("baichuan");
    assert_eq!(
        d.config.n_layers, 32,
        "baichuan.cpp:5-14 reads the variant off this number; 32 is the 7B, which is the \
         only one that RoPEs"
    );
}

// --- exaone (3.x) --------------------------------------------------
//
// `src/models/exaone.cpp`: :3-10 reads only the RMS epsilon, :12-40 is
// the plain tensor set sized from `n_embd_head_k * n_head`, :65-121 is
// the sequential residual with SiLU SwiGLU. Two things this fixture
// pins that the NORM rows above do not:
//
//   * NEOX RoPE, the opposite of the other four.
//   * A TIED lm_head: `output` is TENSOR_NOT_REQUIRED (:19) and falls
//     back to `token_embd` (:22-24), so the file ships no
//     `output.weight`.
//
// This is EXAONE 3.x. `exaone4` has no pre-attention and no pre-FFN
// norm, and `exaone-moe` skips RoPE entirely on its full-attention
// layers; both are different graphs and both still refuse.

const EXAONE_GOLDEN: [f32; 48] = [
    -0.2249429,
    0.34571093,
    0.028062485,
    -0.3856704,
    0.08937423,
    0.18980339,
    -0.052521326,
    0.014981142,
    0.036303222,
    -0.11778174,
    0.24545757,
    -0.19239902,
    0.14795516,
    0.0996446,
    0.0015762504,
    -0.054319553,
    0.40219936,
    -0.35769135,
    0.22361766,
    0.07902548,
    0.18331085,
    0.21348247,
    0.25453204,
    -0.16258636,
    0.42788702,
    0.2037906,
    -0.018451544,
    0.5553448,
    0.26972133,
    0.13789096,
    -0.30589244,
    0.06253724,
    -0.023459988,
    0.16139778,
    0.26396036,
    0.2515811,
    0.53262925,
    0.1445617,
    -0.4598862,
    0.12606835,
    -0.1691545,
    0.17235437,
    -0.09227791,
    -0.11130126,
    0.046939783,
    -0.2966211,
    -0.31041434,
    -0.3674787,
];

#[test]
fn exaone_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("exaone", &EXAONE_GOLDEN);
}

/// The fixture really is the tied case.
#[test]
fn the_exaone_fixture_ships_no_output_weight_and_ties_the_lm_head() {
    let path = graph_fixture_path("exaone");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    assert!(
        file.find_tensor("output.weight").is_none(),
        "the fixture must be the tied case or it pins nothing about exaone.cpp:22-24"
    );
    // It still produced output above, so the lm_head came from
    // `token_embd.weight`.
    assert!(file.find_tensor("token_embd.weight").is_some());
}

// --- gemma (Gemma-1) ------------------------------------------------
//
// The oldest row of the family, and the last one that ran without
// evidence. `src/models/gemma.cpp` in full: `load_arch_hparams` (:3-11)
// reads only the RMS epsilon and picks a size off the layer count that
// nothing branches on; `load_arch_tensors` (:13-34) creates attn_norm,
// split Q/K/V through `create_tensor_qkv`, an attn_output sized from
// `n_embd_head_k * n_head`, ffn_norm and gate/up/down, with no biases,
// no QK-norm and no post-norms; and the graph (:41-138) is the
// sequential residual.
//
// Three pieces are Gemma's and all three are in the fixture's path: the
// sqrt(n_embd) embedding scale (:49), GeGLU rather than SwiGLU (:112,
// `LLM_FFN_GELU, LLM_FFN_PAR`), and an attention scale llama.cpp
// reaches by scaling Q by `1/sqrtf(n_embd_head)` at :86 and then
// passing `kq_scale = 1.0f` at :91 -- the same number frink's kernels
// apply when `attention_scale` is None, which is why fact 3 below holds
// for this row too.
//
// The lm_head is TIED with no fallback: :20 duplicates
// `token_embd.weight` into `output` unconditionally, so the fixture
// ships no `output.weight` and the embedding scale is not cancelled by
// a separate head.

const GEMMA_GOLDEN: [f32; 48] = [
    0.29964927,
    0.6283499,
    0.22527266,
    -0.10582298,
    -0.4777732,
    -0.24892166,
    -0.28255585,
    -0.08312392,
    0.06489246,
    -0.56164,
    -0.16337517,
    0.24526599,
    0.119027674,
    -0.7749605,
    0.16473716,
    0.20174162,
    -0.15967777,
    0.19335732,
    -0.4492208,
    0.5128704,
    0.0019430518,
    0.1487342,
    0.16347827,
    -0.24232048,
    0.24746539,
    0.603195,
    0.1283208,
    -0.4600235,
    0.3777966,
    -0.022114657,
    -0.21029684,
    0.3769806,
    -0.21484213,
    0.13908015,
    -0.077077374,
    0.046033803,
    -0.3738503,
    0.7610342,
    -0.05985511,
    -0.02556756,
    -0.1984951,
    0.13185397,
    -0.19448894,
    0.16269659,
    -0.40522176,
    -0.4143316,
    0.21067318,
    -0.5611287,
];

/// The one row here compared at [`GELU_TABLE_TOL`] rather than
/// `GRAPH_TOL`, because llama.cpp's GELU is an f16 lookup table and
/// frink's is exact. See that constant for the measurement that settled
/// it: an independent numpy pass over this same fixture matches
/// llama.cpp to 1.19e-7 through the table and to 3.93e-5 without it,
/// and 3.93e-5 is what frink shows.
#[test]
fn gemma_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within("gemma", &GEMMA_GOLDEN, GELU_TABLE_TOL);
}

/// The loosened tolerance is loosened for a REASON, and the reason is
/// bounded.
///
/// A per-row tolerance is the easiest place in this suite to hide a real
/// error, so the gap is asserted from BOTH sides: it is genuinely larger
/// than `GRAPH_TOL` (otherwise the exemption is unnecessary and should
/// go) and still under half of `GELU_TABLE_TOL` (otherwise the
/// exemption has started covering something else). If frink ever
/// adopts ggml's table, the first assertion fails and says so.
#[test]
fn gemmas_gap_from_llama_cpp_is_the_gelu_table_and_stays_that_size() {
    let d = load_graph_fixture("gemma");
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &GEMMA_GOLDEN,
    );
    assert!(
        worst > GRAPH_TOL,
        "gemma now agrees to {worst}, within GRAPH_TOL; drop GELU_TABLE_TOL for this row"
    );
    assert!(
        worst < GELU_TABLE_TOL / 2.0,
        "gemma's gap grew to {worst}; GELU_TABLE_TOL was sized for the f16 GELU table \
         (measured 3.93e-5) and is now covering something else"
    );
}

/// Gemma-1's three pieces are the ones frink resolved, and the fixture
/// really is the tied case.
///
/// The embedding scale is asserted as a NUMBER rather than as
/// `is_some`: `sqrt(hidden_dim)` and `sqrt(head_dim)` are both
/// plausible readings of "scale the embeddings", they differ by a
/// factor of 1.7 here, and only one of them is `gemma.cpp:49`.
#[test]
fn gemma_resolves_its_embedding_scale_its_activation_and_its_tied_head() {
    let d = load_graph_fixture("gemma");
    assert_eq!(
        d.config.embedding_scale,
        Some((d.config.hidden_dim as f32).sqrt()),
        "gemma.cpp:49 scales the embeddings by sqrt(n_embd)"
    );
    assert_eq!(
        d.config.ffn_activation,
        frink_models::config::FfnActivation::Gelu,
        "gemma.cpp:112 passes LLM_FFN_GELU, not LLM_FFN_SILU"
    );
    // head_dim comes from `gemma.attention.key_length`; n_embd/n_head
    // would be 6.
    assert_eq!(d.config.head_dim, 8);
    assert_eq!(d.config.hidden_dim, 24);

    let path = graph_fixture_path("gemma");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    assert!(
        file.find_tensor("output.weight").is_none(),
        "gemma.cpp:20 ties the lm_head unconditionally; a fixture with its own head would \
         not exercise that, and llama.cpp would refuse the unread tensor"
    );
}

/// Gemma-1 declares no softcap and no sliding window, so the Gemma-2/3
/// machinery must resolve to INERT on this row.
///
/// `GemmaFamily` is exempted from `unsupported_feature_keys` on the
/// grounds that it implements the softcap and the sliding window. That
/// exemption is only honest while a Gemma checkpoint declaring none of
/// them ends up with none of them, and Gemma-1 is the row where the
/// whole family's machinery has nothing to do.
///
/// It is NOT exempted from `unsupported_scaling_keys` any more, and the
/// difference is worth stating: Gemma scales its embeddings and its 27B
/// attention scores, but reads NEITHER key -- both are arithmetic in
/// `load_arch_hparams`. The blanket exemption meant a hand-written
/// `gemma3.residual_scale` would have loaded and been ignored.
#[test]
fn gemma_one_gets_no_softcap_no_window_and_no_attention_scale_override() {
    let d = load_graph_fixture("gemma");
    assert!(d.config.sliding_window.is_none(), "gemma-1 declares none");
    assert!(d.config.attn_logit_softcap.is_none());
    assert!(d.config.final_logit_softcap.is_none());
    // Fact 3: the 27B branch of `attention_scale_override` is the only
    // Gemma size that overrides the kernels' own scale, and this is not
    // it.
    assert!(d.config.attention_scale.is_none());
}

/// Dropping the sqrt(n_embd) embedding scale is a large divergence.
///
/// This is what makes the golden comparison mean something. `gemma`'s
/// verdict claimed the scale was implemented; a fixture that could not
/// see it would have "proved" that claim just as well with the scale
/// removed.
#[test]
fn dropping_gemmas_embedding_scale_diverges_from_llama_cpp() {
    let path = graph_fixture_path("gemma");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.embedding_scale = None;
    let d = common::load_decoder(&path, config);
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &GEMMA_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "removing the embedding scale moved the logits by only {worst}; the fixture cannot \
         see gemma.cpp:49"
    );
}

/// Running Gemma's FFN as SwiGLU instead of GeGLU is a large
/// divergence.
///
/// gelu and silu agree to within a few percent near zero, so this only
/// holds because the fixture's `ffn_gate` is drawn wide enough to reach
/// the range where they part.
#[test]
fn running_gemmas_ffn_as_swiglu_instead_of_geglu_diverges_from_llama_cpp() {
    let path = graph_fixture_path("gemma");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.ffn_activation = frink_models::config::FfnActivation::Swiglu;
    let d = common::load_decoder(&path, config);
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &GEMMA_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "swapping GeGLU for SwiGLU moved the logits by only {worst}; the fixture cannot see \
         gemma.cpp:112"
    );
}

// --- bailingmoe2 (Ling-2.0) -----------------------------------------
//
// The one MoE row in this batch, and the only one with a QK-norm.
// `src/models/bailingmoe2.cpp`: a FUSED `attn_qkv` (:49) that
// `load_qkv_projections` splits by the same arithmetic
// `llm_graph_context::build_qkv` uses, per-head `attn_q_norm` /
// `attn_k_norm` of width `n_embd_head_k` (:52-53) applied BEFORE RoPE
// (:123-135), leading dense layers that :57 really does branch on,
// `exp_probs_b` (:61), a shared expert `n_ff_shexp * n_expert_shared`
// wide (:58), and `expert_weights_norm` / `expert_weights_scale` /
// `expert_gating_func` all read from METADATA (:9-11) rather than
// hardcoded. Sequential residual (:149, :191).
//
// It is NOT `bailingmoe`, which is a NORM-RoPE row admitted earlier for
// the opposite reason: that one reads `leading_dense_block_count` and
// then never branches on it.

const BAILINGMOE2_GOLDEN: [f32; 48] = [
    -0.09195735,
    0.124805875,
    0.02403112,
    0.33854562,
    -0.3279288,
    0.0605181,
    0.34822923,
    -0.00276571,
    0.106965765,
    -0.1772596,
    -0.25497413,
    0.32685977,
    0.37302768,
    -0.13560869,
    -0.096370846,
    -0.44192657,
    0.15269074,
    0.32532498,
    -0.11355774,
    -0.008084873,
    0.3309023,
    -0.3946235,
    0.018904123,
    -0.018048527,
    -0.1964149,
    -0.05105628,
    0.42756924,
    -0.18849637,
    0.40864584,
    -0.11983519,
    -0.103815354,
    0.21823177,
    0.26624107,
    0.18969972,
    0.38225642,
    -0.048707068,
    0.9169096,
    -0.49677095,
    -0.0034501795,
    0.261436,
    0.08845875,
    -0.449533,
    -0.19524488,
    0.17076917,
    -0.31562522,
    -0.54570425,
    0.36975017,
    -0.4732991,
];

#[test]
fn bailingmoe2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("bailingmoe2", &BAILINGMOE2_GOLDEN);
}

/// The routing facts come out of the FILE, not out of an
/// architecture-name guess.
///
/// This is the assertion that matters most on this row.
/// `bailingmoe2.cpp:11` reads `expert_gating_func` as REQUIRED and the
/// real models are sigmoid-gated, while frink's fallback
/// (`SIGMOID_GATING_ARCHITECTURES` in `loader.rs`) does NOT list
/// `bailingmoe2` and would default it to softmax. So the file's key is
/// the only thing standing between this row and the wrong scoring
/// function on every token -- the `deepseek` shape again, and worth
/// pinning by name rather than trusting the logit comparison to notice.
#[test]
fn bailingmoe2_reads_its_routing_out_of_the_file_rather_than_guessing() {
    let path = graph_fixture_path("bailingmoe2");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    // The keys really are in the file; otherwise this proves nothing
    // about reading them.
    assert!(
        file.metadata_u64("bailingmoe2.expert_gating_func")
            .is_some(),
        "the fixture must carry the key it is being read from"
    );
    let d = load_graph_fixture("bailingmoe2");
    assert_eq!(
        d.config.moe.gating,
        GatingFunction::Sigmoid,
        "bailingmoe2.cpp:11 reads this from the file; frink's name-based fallback would \
         have said softmax"
    );
    assert!(
        d.config.moe.norm_topk_prob,
        "expert_weights_norm is true in this file (bailingmoe2.cpp:10 reads it)"
    );
    assert_eq!(d.config.moe.expert_weights_scale, 2.5);
    assert_eq!(d.config.moe.n_experts, 6);
    assert_eq!(d.config.moe.n_experts_active, 2);
    assert_eq!(d.config.moe.n_shared_experts, 2);
    // Honoured here, unlike `bailingmoe`.
    assert!(d.config.layer_is_dense(0));
    assert!(!d.config.layer_is_dense(1));
    assert!(!d.config.layer_is_dense(2));
    // Fused QKV split, per-head QK norm, and the shared expert's width
    // is n_ff_shexp * n_expert_shared = 8 * 2, not 8.
    assert_eq!(d.config.qk_norm_style, QkNormStyle::PerHead);
    for (il, layer) in d.layers.iter().enumerate() {
        assert_eq!(
            layer.attn.q_norm.as_ref().map(Vec::len),
            Some(d.config.head_dim),
            "blk.{il}: per-head Q norm"
        );
        if il == 0 {
            continue;
        }
        assert_eq!(layer.moe.shared_experts.len(), 1, "blk.{il}");
        assert_eq!(layer.moe.shared_experts[0].gate.rows(), 16, "blk.{il}");
        assert!(layer.moe.exp_probs_bias.is_some(), "blk.{il}");
    }
}

/// Routing through the wrong scoring function is a visible error, not a
/// rounding one.
///
/// Without this the golden comparison could pass under either gating
/// function for a fixture whose router happened to pick the same two
/// experts with near-equal weights, and the fact the test above pins
/// would be untested rather than merely unasserted.
#[test]
fn routing_bailingmoe2_through_softmax_instead_of_sigmoid_diverges_from_llama_cpp() {
    let path = graph_fixture_path("bailingmoe2");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.moe.gating = GatingFunction::Softmax;
    let d = common::load_decoder(&path, config);
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &BAILINGMOE2_GOLDEN,
    );
    assert!(
        worst > 1e-3,
        "softmax gating changed the output by only {worst}; the fixture cannot see this"
    );
}

/// QK norm on the wrong side of RoPE is a large divergence here too.
///
/// `bailingmoe2` norms BEFORE rotating; `maincoder` and `hunyuan-moe`
/// norm after. One `Decoder` flag decides it for all three, so the
/// fixture has to be able to see the flip in this direction as well as
/// the other.
#[test]
fn norming_bailingmoe2_after_rope_instead_of_before_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture("bailingmoe2");
    assert!(
        !d.qk_norm_after_rope,
        "bailingmoe2.cpp:123-135 norms Q and K and only then rotates them"
    );
    d.qk_norm_after_rope = true;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &BAILINGMOE2_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "swapping the QK-norm order moved the output by only {worst}; \
         the fixture cannot see this arm"
    );
}

// --- plamo3 ---------------------------------------------------------
//
// The sandwich-norm row, the only one here with a sliding window, and
// the one whose FIXTURE-AWAY verdict turned out to be WRONG by one
// tensor name.
//
// `LLM_TN` appends `.weight` only when given a suffix
// (llama-arch.cpp:898-910), and every architecture that creates
// ATTN_POST_NORM / FFN_POST_NORM passes one -- except `plamo3`, which
// uses the two-argument overload (plamo3.cpp:52,55) and so asks for
// `blk.N.post_attention_norm` and `blk.N.post_ffw_norm` with NO suffix.
// gguf-py emits exactly those names for it, because its PLaMo mapping
// keys (tensor_mapping.py:368,434) already END in `.weight` and
// `get_type_and_name` (:2585-2594) tries an exact match before
// stripping a suffix. frink read only the suffixed spelling, so it
// could never have loaded a real PLaMo-3 checkpoint --
// `load_norm_vec_either_spelling` in `loader.rs` is the arm that fixes
// it, and this fixture uses the un-suffixed names because that is what
// a converter writes.
//
// The rest is slot for slot: attn_norm (:104), fused attn_qkv (:47) with
// head_dim decoupled from n_embd/n_head, per-head QK norm before RoPE
// (:49-50, :128-138), attn_post_norm on the attention output before its
// residual add (:152-155), ffn_norm (:160), a fused SwiGLU ffn_up of
// n_ff*2 with no ffn_gate (:57, :163-168), and ffn_post_norm on the FFN
// output before its add (:171-174). NEOX RoPE.

const PLAMO3_GOLDEN: [f32; 48] = [
    -0.13417868,
    0.5886667,
    0.43671453,
    0.47216994,
    0.0437923,
    0.6202583,
    -0.33917707,
    -0.15624414,
    -0.4340622,
    0.08551871,
    -0.28434193,
    -0.6546696,
    -0.33789676,
    0.06587186,
    0.3138425,
    0.7828187,
    -0.2447187,
    0.75487375,
    -0.6003437,
    0.070651375,
    -0.27455223,
    0.5503286,
    1.041208,
    -1.1511995,
    0.37486026,
    0.3923036,
    -0.086244255,
    -0.47600126,
    -0.25178447,
    0.8037309,
    0.19178456,
    -0.6201743,
    0.41236115,
    0.654441,
    -0.09785151,
    0.095623925,
    -0.27044287,
    -0.5235145,
    -0.57836926,
    0.32180697,
    -0.3484251,
    0.004385993,
    -0.17752947,
    -0.032321587,
    0.11161679,
    0.20395917,
    0.15744714,
    0.037524372,
];

#[test]
fn plamo3_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("plamo3", &PLAMO3_GOLDEN);
}

/// The fixture spells the post-norms the way a converter does, and
/// frink reads that spelling.
///
/// This is the assertion the whole row turns on. If the fixture were
/// regenerated with `.weight` names it would still load in frink and
/// would stop being evidence about PLaMo-3 -- and libllama would refuse
/// it outright, so the golden values could not be regenerated at all.
#[test]
fn the_plamo3_fixture_spells_its_post_norms_without_a_weight_suffix() {
    let path = graph_fixture_path("plamo3");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    for base in ["post_attention_norm", "post_ffw_norm"] {
        assert!(
            file.find_tensor(&format!("blk.0.{base}")).is_some(),
            "blk.0.{base} is the name plamo3.cpp:52,55 asks for"
        );
        assert!(
            file.find_tensor(&format!("blk.0.{base}.weight")).is_none(),
            "blk.0.{base}.weight is the spelling every OTHER architecture uses; carrying \
             both would make this fixture prove nothing about plamo3"
        );
    }
    let d = load_graph_fixture("plamo3");
    for (il, layer) in d.layers.iter().enumerate() {
        let post_attn = layer
            .attn
            .post_attn_norm
            .as_ref()
            .unwrap_or_else(|| panic!("blk.{il}: attn_post_norm must be loaded"));
        let post_ffn = layer
            .attn
            .post_ffn_norm
            .as_ref()
            .unwrap_or_else(|| panic!("blk.{il}: ffn_post_norm must be loaded"));
        assert!(post_attn.iter().any(|w| *w != 0.0), "blk.{il}");
        assert!(post_ffn.iter().any(|w| *w != 0.0), "blk.{il}");
    }
}

/// Dropping either sandwich norm is a large divergence.
///
/// `post_attn_norm` and `post_ffn_norm` are two of the eight model
/// features a copied decode path in this repo has silently lost, so a
/// fixture that carried them without being able to see them missing
/// would be the same trap one level up.
#[test]
fn dropping_either_of_plamo3s_post_norms_diverges_from_llama_cpp() {
    for which in ["attn", "ffn"] {
        let mut d = load_graph_fixture("plamo3");
        for layer in d.layers.iter_mut() {
            if which == "attn" {
                layer.attn.post_attn_norm = None;
            } else {
                layer.attn.post_ffn_norm = None;
            }
        }
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &PLAMO3_GOLDEN,
        );
        assert!(
            worst > 1e-2,
            "dropping post_{which}_norm moved the output by only {worst}; \
             the fixture cannot see this slot"
        );
    }
}

/// The sliding window, its period AND its phase, all read from the
/// file.
///
/// `plamo3.cpp:5-11` reads `attention.sliding_window` and a scalar
/// `attention.sliding_window_pattern` (defaulting to 8) and calls
/// `set_swa_pattern(period)` with `dense_first = false`, which is
/// `is_swa(il) = il % period < period - 1`. The fixture sets period 2
/// over four layers, so layers 0 and 2 slide and 1 and 3 do not. Getting
/// the phase inverted is invisible on a prompt shorter than the window,
/// which is why this fixture's window is 3 and its prompt is 6.
#[test]
fn plamo3_reads_its_window_period_and_phase_and_the_window_actually_bites() {
    let d = load_graph_fixture("plamo3");
    assert_eq!(d.config.sliding_window, Some(3));
    assert_eq!(
        d.config.swa_layers,
        SwaLayers::period(2, false),
        "set_swa_pattern's dense_first defaults to false and plamo3.cpp:11 does not pass it"
    );
    assert_eq!(d.config.layer_sliding_window(0), Some(3));
    assert_eq!(d.config.layer_sliding_window(1), None);
    assert_eq!(d.config.layer_sliding_window(2), Some(3));
    assert_eq!(d.config.layer_sliding_window(3), None);
    // The window is narrower than the prompt, so it masks something at
    // the position the comparison reads.
    assert!(GRAPH_PROMPT.len() > 3);
}

/// Widening the window past the prompt diverges, which is what proves
/// the mask is doing work.
///
/// A fixture whose window never bites compares equal with SWA switched
/// off entirely, and would be evidence about the tensor set rather than
/// about attention.
#[test]
fn removing_plamo3s_sliding_window_diverges_from_llama_cpp() {
    let path = graph_fixture_path("plamo3");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.sliding_window = None;
    let d = common::load_decoder(&path, config);
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &PLAMO3_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "dropping the window moved the output by only {worst}; it never masked anything"
    );
}

/// Inverting the SWA phase diverges too.
///
/// The period alone is not the pattern: `smallthinker` and `laguna`
/// windowed every layer here once because frink implemented only one
/// phase. Swapping which layers slide has to be visible, or the phase is
/// untested even though the period is not.
#[test]
fn inverting_plamo3s_swa_phase_diverges_from_llama_cpp() {
    let path = graph_fixture_path("plamo3");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.swa_layers = SwaLayers::period(2, true);
    let d = common::load_decoder(&path, config);
    assert_eq!(
        d.config.layer_sliding_window(0),
        None,
        "phase really flipped"
    );
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &PLAMO3_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "inverting the SWA phase moved the output by only {worst}; \
         the fixture cannot see which layers slide"
    );
}

/// The triage verdict asked for this one by name: llama.cpp carries
/// head_dim_q and head_dim_v separately (plamo3.cpp:25-26) and frink
/// has a single `head_dim`, so the fixture has to be a file where the
/// two agree -- and a file where they disagree is refused by name in
/// `loader.rs` rather than run on one of them.
#[test]
fn the_plamo3_fixture_has_equal_key_and_value_head_dims() {
    let path = graph_fixture_path("plamo3");
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    assert_eq!(file.metadata_u64("plamo3.attention.key_length"), Some(8));
    assert_eq!(file.metadata_u64("plamo3.attention.value_length"), Some(8));
    let d = load_graph_fixture("plamo3");
    assert_eq!(d.config.head_dim, 8);
    // 6 is what n_embd / n_head would give.
    assert_eq!(d.config.hidden_dim, 24);
    assert_eq!(d.config.n_heads, 4);
}

// --- the six facts, asserted rather than assumed --------------------

/// Fact 1: the RoPE variant each row resolves to is the group llama.cpp
/// puts it in.
#[test]
fn the_rope_variant_each_architecture_uses_is_the_one_llama_cpp_uses() {
    for (name, want) in [
        // llama-model.cpp `llama_model_rope_type`, NORM group.
        ("internlm2", RopeLayout::Norm),
        ("xverse", RopeLayout::Norm),
        ("ernie4_5", RopeLayout::Norm),
        ("baichuan", RopeLayout::Norm),
        // ... and the NEOX group.
        ("exaone", RopeLayout::Neox),
        // llama-model.cpp:2642, the same NEOX arm gemma2 and gemma3 are
        // in.
        ("gemma", RopeLayout::Neox),
        ("bailingmoe2", RopeLayout::Neox),
        ("plamo3", RopeLayout::Neox),
    ] {
        assert_eq!(
            load_graph_fixture(name).config.rope_layout,
            want,
            "{name}: rope layout"
        );
    }
}

/// Fact 1, the half that makes fact 1 worth asserting: each fixture can
/// SEE a flipped RoPE variant.
///
/// Rotating the wrong pairs of every Q/K head is the defect that
/// produced the Llama-3.1-8B wrong-output bug, and a fixture whose
/// positions happened not to matter would agree under either variant.
#[test]
fn rotating_the_wrong_pairs_diverges_from_llama_cpp() {
    for (name, golden) in [
        ("internlm2", &INTERNLM2_GOLDEN),
        ("xverse", &XVERSE_GOLDEN),
        ("ernie4_5", &ERNIE4_5_GOLDEN),
        ("baichuan", &BAICHUAN_GOLDEN),
        ("exaone", &EXAONE_GOLDEN),
        ("gemma", &GEMMA_GOLDEN),
        ("bailingmoe2", &BAILINGMOE2_GOLDEN),
        ("plamo3", &PLAMO3_GOLDEN),
    ] {
        let path = graph_fixture_path(name);
        let file = frink_gguf::GgufFile::open(&path).expect("opens");
        let mut config = ModelConfig::from_gguf(&file).expect("parses");
        config.rope_layout = match config.rope_layout {
            RopeLayout::Norm => RopeLayout::Neox,
            _ => RopeLayout::Norm,
        };
        let d = common::load_decoder(&path, config);
        let mut kv = graph_caches(&d);
        let worst = worst_vs(&d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv), golden);
        assert!(
            worst > 1e-2,
            "{name}: flipping the RoPE variant moved the output by only {worst}; \
             the fixture cannot see this"
        );
    }
}

/// Fact 3, on EVERY row: no attention-scale override.
///
/// All seven pass a literal `1.0f/sqrtf(float(n_embd_head))` to
/// `build_attn` and none reads `LLM_KV_ATTENTION_SCALE`, so
/// `ModelConfig::attention_scale` must stay `None` and let the kernels'
/// own scale stand. Getting this wrong is silent: the model still runs.
#[test]
fn no_row_here_overrides_the_attention_scale() {
    for name in ALL_ROWS {
        let d = load_graph_fixture(name);
        assert!(
            d.config.attention_scale.is_none(),
            "{name}: attention_scale must stay unset"
        );
    }
}

/// Facts 2, 4 and 5, for the six rows where they are ABSENCES.
///
/// `plamo3` is excluded because it genuinely has a sliding window and
/// both post-norms; its window, period, phase and two norm slots are
/// pinned by its own tests above. The membership lives in one const so
/// the two halves cannot drift apart about who is in the set.
///
/// Each of these has been lost at least once here by a copied decode
/// path, so the assertion is that frink resolved them to absent rather
/// than to something plausible.
#[test]
fn six_of_the_rows_have_no_window_and_no_post_norm() {
    for name in NO_WINDOW_NO_POST_NORM_ROWS {
        let d = load_graph_fixture(name);
        // Fact 2: none reads LLM_KV_ATTENTION_SLIDING_WINDOW, so there
        // is no window and therefore no pattern phase to get wrong.
        assert!(d.config.sliding_window.is_none(), "{name}: sliding window");
        for (il, layer) in d.layers.iter().enumerate() {
            // Facts 4 and 5.
            assert!(
                layer.attn.post_attn_norm.is_none(),
                "{name} blk.{il}: no LLM_TENSOR_ATTN_POST_NORM upstream"
            );
            assert!(
                layer.attn.post_ffn_norm.is_none(),
                "{name} blk.{il}: no LLM_TENSOR_FFN_POST_NORM upstream"
            );
        }
    }
}

/// Fact 6, for the five rows where it is an ABSENCE.
///
/// `bailingmoe2` is excluded because it genuinely has per-head QK norm,
/// and its ordering is pinned by its own two tests above. Splitting the
/// set this way rather than writing an inline name comparison is
/// deliberate: the membership lives in one const the checklist tests
/// share.
#[test]
fn the_dense_rows_have_no_qk_norm_and_therefore_no_ordering_question() {
    for name in DENSE_ROWS {
        let d = load_graph_fixture(name);
        for (il, layer) in d.layers.iter().enumerate() {
            assert!(
                layer.attn.q_norm.is_none() && layer.attn.k_norm.is_none(),
                "{name} blk.{il}: no attn_q_norm/attn_k_norm upstream, so no ordering \
                 question either"
            );
        }
    }
}

/// The MoE half of the checklist arises for exactly one row, and that
/// is a fact about the files rather than an omission.
///
/// Stated as an assertion because "we did not check the gating function"
/// and "there is no gating function" read identically in a commit
/// message.
#[test]
fn only_bailingmoe2_is_moe() {
    for name in DENSE_ROWS {
        let d = load_graph_fixture(name);
        assert_eq!(
            d.config.moe.n_experts, 1,
            "{name}: dense, so gating and top-k renormalisation do not arise"
        );
        assert_eq!(d.config.moe.n_shared_experts, 0, "{name}");
    }
    let plamo3 = load_graph_fixture("plamo3");
    assert_eq!(
        plamo3.config.moe.n_experts, 1,
        "plamo3 is dense too, it just is not in DENSE_ROWS because it has post-norms"
    );
    let moe = load_graph_fixture("bailingmoe2");
    assert!(
        moe.config.moe.n_experts > 1,
        "bailingmoe2 is the row the MoE questions are asked of"
    );
}
