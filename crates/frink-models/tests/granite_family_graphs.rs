//! The Granite family, checked against llama.cpp itself.
//!
//! `granite`, `granitemoe` and frink's `granite-moe` alias were all
//! triaged NEW CODE against ONE cause: four scalar MULTIPLIERS that
//! `src/models/granite.cpp` applies and the generic decoder did not.
//! They are hyper-parameters rather than tensors, so
//! `assert_every_tensor_consumed` cannot see them and a Granite
//! checkpoint would have loaded, run at full speed, and answered at a
//! scale it was never trained at. frink refused such a file by name
//! instead, and this suite is the evidence that replaced that refusal.
//!
//! | key | where llama.cpp applies it | frink |
//! |---|---|---|
//! | `{arch}.embedding_scale` | `llama-graph.cpp:2337-2342`, the shared `build_inp_embd` | `ModelConfig::embedding_scale` |
//! | `{arch}.attention.scale` | `granite.cpp:225`, as `kq_scale`, with a `0.0f` "unset" sentinel | `ModelConfig::attention_scale` |
//! | `{arch}.residual_scale` | `granite.cpp:235-238` and `:288-292`, on BOTH branch outputs | `ModelConfig::residual_scale` |
//! | `{arch}.logit_scale` | `granite.cpp:180`, as `1.0f / f_logit_scale` | `ModelConfig::logit_multiplier` |
//!
//! **One implementation, not one per architecture.**
//! `src/models/granite-moe.cpp` has NO graph: `models.h:1583-1591` is
//! `using graph = llama_model_granite::graph`, so the MoE row runs the
//! dense row's graph and takes its `ffn_gate_inp != nullptr` branch. The
//! two differ in the FFN and in nothing else. `crate::scalar_multipliers`
//! therefore carries one `MultiplierSupport` constant for all three
//! rows, and `capability::unsupported_scaling_keys` -- the list that
//! still refuses these keys everywhere else -- is DERIVED from that same
//! table rather than restated beside it.
//!
//! **The `granite-moe` alias.** `llama-arch.cpp:101` spells the
//! architecture `granitemoe`; no GGUF anywhere says `granite-moe`, so
//! there is no libllama golden for it and there never can be. Its
//! evidence is a second fixture, written from the same script with the
//! same seed and the same weights, differing only in the architecture
//! string and therefore in every key prefix, asserted against
//! `granitemoe`'s libllama golden. An alias nothing outside frink
//! exercises is exactly the row that drifts unnoticed, and one golden
//! read by both files is the only thing that stops it.
//!
//! **The six facts this repo has lost at least once**, checked against
//! the C for both rows and asserted below rather than merely read:
//!
//! 1. **RoPE variant.** `llama_model_rope_type` (llama-model.cpp:2594-2595)
//!    puts `LLM_ARCH_GRANITE` and `LLM_ARCH_GRANITE_MOE` in the
//!    consecutive-pairs NORM group.
//! 2. **SWA pattern and phase.** Neither `granite.cpp:5-45` nor
//!    `granite-moe.cpp:3-24` reads `LLM_KV_ATTENTION_SLIDING_WINDOW` at
//!    all, so `hparams.swa_type` stays `LLAMA_SWA_TYPE_NONE` and there
//!    is no phase to get wrong.
//! 3. **`attention_scale`.** This is the row where it is REAL rather
//!    than absent: `granite.cpp:225` passes `f_attention_scale` as
//!    `kq_scale` whenever it is non-zero. Both fixtures declare it, and
//!    it is 0.9 against the kernels' own `1/sqrt(6) = 0.408`.
//! 4. **`post_attn_norm`** and 5. **`post_ffn_norm`.** Neither row
//!    creates `LLM_TENSOR_ATTN_POST_NORM` or `LLM_TENSOR_FFN_POST_NORM`
//!    (`granite.cpp:48-107`); each layer has exactly `attn_norm` and
//!    `ffn_norm`, both applied BEFORE their branch.
//! 6. **QK-norm and its order.** Neither row creates `attn_q_norm` or
//!    `attn_k_norm`, so the ordering question does not arise.
//!
//! The MoE half of the checklist arises for `granitemoe` alone:
//! `granite.cpp:266-277` passes `norm_w = true` (the top-k weights ARE
//! renormalised), `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`, and
//! `hparams.expert_weights_scale` -- which neither Granite
//! `load_arch_hparams` reads, so it keeps its `0.0f` default and
//! `llama-graph.cpp:2070` treats that as no scaling. Its shared expert
//! (`:280-287`) is UNGATED, unlike qwen2moe's.
//!
//! **Where the numbers come from.** Each `GOLDEN` array was produced by
//! running llama.cpp's own graph for that architecture over the same
//! fixture file, through `scripts/gptoss_reference_logits.cpp` linked
//! against a real `libllama` built from `.scratch/llama.cpp`. Not by
//! re-reading a spec, and not by frink checking itself.
//!
//! Regenerating (both halves must be redone together if a fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_granite_fixture.py \
//!     crates/frink-models/tests/fixtures/granite_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_granitemoe_fixture.py \
//!     crates/frink-models/tests/fixtures/granitemoe_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_granitemoe_fixture.py \
//!     crates/frink-models/tests/fixtures/granite-moe_tiny.gguf --arch granite-moe
//! clang++ -std=c++17 -O2 scripts/gptoss_reference_logits.cpp \
//!     -I$LLAMA/include -I$LLAMA/ggml/include -L$BUILD/bin -lllama \
//!     -Wl,-rpath,$BUILD/bin -o /tmp/ref_logits
//! /tmp/ref_logits crates/frink-models/tests/fixtures/granite_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::{LoadError, ModelConfig, RopeLayout};
use frink_moe::GatingFunction;

/// The rows this suite admits. `granite-moe` is here because it must be
/// held to the same standard as the two llama.cpp spells, not because
/// llama.cpp has ever seen it.
const ALL_ROWS: [&str; 3] = ["granite", "granitemoe", "granite-moe"];

/// The four multipliers, at the values both fixtures declare. Named once
/// so a fixture regenerated with different numbers cannot leave the
/// assertions below quietly checking the old ones.
const LOGIT_SCALE: f32 = 2.5;
const RESIDUAL_SCALE: f32 = 0.6;
const EMBEDDING_SCALE: f32 = 2.0;
const ATTENTION_SCALE: f32 = 0.9;

// --- granite (dense) -----------------------------------------------

const GRANITE_GOLDEN: [f32; 48] = [
    0.21238104,
    0.04593788,
    0.11568701,
    0.04954892,
    -0.075302266,
    -0.14445536,
    -0.13070965,
    0.15606217,
    0.04042698,
    0.097806156,
    0.061000098,
    0.18766312,
    0.0122355595,
    0.26365158,
    -0.038631074,
    -0.016227191,
    0.22130993,
    -0.10674151,
    -0.107973576,
    -0.03487205,
    -0.11159064,
    -0.02601102,
    -0.21261115,
    -0.03686999,
    0.026109463,
    -0.13615204,
    -0.19500309,
    -0.31484917,
    -0.086113006,
    0.17768298,
    0.0023619116,
    0.10767656,
    0.14539307,
    -0.053612947,
    0.17392503,
    0.13508613,
    0.3039389,
    -0.013565193,
    -0.030599846,
    -0.16500527,
    -0.101739004,
    0.11029877,
    -0.31132218,
    -0.023155175,
    -0.1264253,
    0.17630778,
    0.11140199,
    0.029099492,
];

// --- granitemoe, and the granite-moe alias -------------------------

const GRANITEMOE_GOLDEN: [f32; 48] = [
    -0.18181351,
    0.036202192,
    -0.00089810713,
    -0.20761596,
    -0.0577685,
    0.03013968,
    -0.21684515,
    -0.008914097,
    0.13303457,
    -0.44153872,
    -0.1307926,
    0.16704307,
    0.09496286,
    -0.041438863,
    0.15549271,
    -0.20929816,
    -0.18375571,
    0.2420918,
    -0.01620405,
    0.042574324,
    0.26279458,
    0.049937963,
    0.2108588,
    -0.19250788,
    -0.046461165,
    0.15700065,
    0.21241191,
    -0.17370479,
    -0.06468272,
    -0.2718286,
    0.010240785,
    -0.30038166,
    -0.16452256,
    0.01389374,
    -0.19321552,
    -0.0070547187,
    0.05246648,
    -0.25962797,
    -0.2188357,
    0.1500749,
    -0.12813823,
    -0.12928225,
    0.105589524,
    -0.0389446,
    0.047136985,
    0.047886096,
    -0.04052823,
    0.3787539,
];

/// `granitemoe_norope_tiny.gguf`: `granitemoe_tiny.gguf` plus
/// `rope.scaling.finetuned = false`. libllama's answer is the rotated
/// one, bit for bit (#439): `granite-moe.cpp`'s `load_arch_hparams`
/// never reads the key, so the switch `granite.cpp:33-37` builds from it
/// stays at the all-ones seed.
const GRANITEMOE_NOROPE_GOLDEN: [f32; 48] = [
    -0.18181351,
    0.036202192,
    -0.00089810713,
    -0.20761596,
    -0.0577685,
    0.03013968,
    -0.21684515,
    -0.008914097,
    0.13303457,
    -0.44153872,
    -0.1307926,
    0.16704307,
    0.09496286,
    -0.041438863,
    0.15549271,
    -0.20929816,
    -0.18375571,
    0.2420918,
    -0.01620405,
    0.042574324,
    0.26279458,
    0.049937963,
    0.2108588,
    -0.19250788,
    -0.046461165,
    0.15700065,
    0.21241191,
    -0.17370479,
    -0.06468272,
    -0.2718286,
    0.010240785,
    -0.30038166,
    -0.16452256,
    0.01389374,
    -0.19321552,
    -0.0070547187,
    0.05246648,
    -0.25962797,
    -0.2188357,
    0.1500749,
    -0.12813823,
    -0.12928225,
    0.105589524,
    -0.0389446,
    0.047136985,
    0.047886096,
    -0.04052823,
    0.3787539,
];

const GRANITE_NOROPE_GOLDEN: [f32; 48] = [
    0.23052013,
    0.09288454,
    0.21783777,
    0.14887694,
    -0.11577002,
    -0.08765256,
    -0.1511468,
    0.043374788,
    0.0828368,
    0.008786253,
    0.04943922,
    0.2754695,
    0.019775443,
    0.09583954,
    -0.2172904,
    0.020783458,
    0.18882959,
    -0.20409067,
    -0.1668834,
    -0.124541715,
    -0.12507145,
    -0.01460816,
    -0.107605636,
    -0.07398244,
    0.023462543,
    -0.0828686,
    -0.16141032,
    -0.26306504,
    -0.13297711,
    0.059283186,
    -0.014935446,
    0.12299373,
    0.11021507,
    -0.043226868,
    0.32263643,
    0.13570234,
    0.32243794,
    -0.046138868,
    0.049874146,
    -0.2163964,
    -0.24558543,
    0.17022283,
    -0.3291752,
    0.056525428,
    0.04466207,
    0.25512132,
    0.19062524,
    -0.11793824,
];

/// The golden array for `arch`. The alias shares `granitemoe`'s, which
/// is the whole claim being made about it.
fn golden(arch: &str) -> &'static [f32] {
    match arch {
        "granite" => &GRANITE_GOLDEN,
        "granitemoe" | "granite-moe" => &GRANITEMOE_GOLDEN,
        other => panic!("no golden for `{other}`"),
    }
}

#[test]
fn granite_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("granite", &GRANITE_GOLDEN);
}

#[test]
fn granitemoe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("granitemoe", &GRANITEMOE_GOLDEN);
}

/// The alias carries `granitemoe`'s evidence, because it can carry no
/// evidence of its own.
///
/// `granite-moe` is a frink-only string. Nothing upstream will ever
/// produce a file spelling it that way, so no libllama run can ever
/// judge it, and the only way it can be wrong is by resolving to
/// different arithmetic from the row it aliases. The fixture is the same
/// weights under a different `general.architecture`, so equality with
/// `granitemoe`'s golden is exactly the property that matters.
#[test]
fn the_granite_moe_alias_reproduces_granitemoes_llama_cpp_logits() {
    assert_all_three_paths_match("granite-moe", &GRANITEMOE_GOLDEN);
}

/// The KL each row lands at, reported rather than merely bounded.
///
/// The bound is generous on purpose -- it is the max-abs comparison in
/// `assert_all_three_paths_match` that actually gates these rows, at
/// `GRAPH_TOL` -- but a KL printed only on failure is a number nobody
/// ever reads. `--nocapture` shows it.
#[test]
fn every_granite_row_agrees_with_llama_cpp_in_kl() {
    for arch in ALL_ROWS {
        let d = load_graph_fixture(arch);
        let mut kv = graph_caches(&d);
        let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
        let kl = kl_vs_golden(&got, golden(arch));
        let worst = worst_vs(&got, golden(arch));
        println!("{arch}: KL(llama.cpp || frink) = {kl:.3e}, max |dlogit| = {worst:.3e}");
        assert!(
            kl < 1e-9,
            "`{arch}` KL {kl} is far above float32 accumulation noise"
        );
    }
}

// --- what the fixtures pin about the loader ------------------------

/// All four multipliers arrive in the config, at the values the files
/// declare, on every row.
///
/// The golden comparison alone cannot say WHICH multiplier is doing the
/// work, and three of the four could be swapped for each other's slot
/// without the totals looking obviously wrong. `logit_multiplier` is the
/// one worth staring at: it is the RECIPROCAL, resolved at load time,
/// because `granite.cpp:180` divides.
#[test]
fn every_granite_row_reads_all_four_multipliers_out_of_its_file() {
    for arch in ALL_ROWS {
        let d = load_graph_fixture(arch);
        let c = &d.config;
        assert_eq!(c.embedding_scale, Some(EMBEDDING_SCALE), "{arch}");
        assert_eq!(c.residual_scale, Some(RESIDUAL_SCALE), "{arch}");
        assert_eq!(c.attention_scale, Some(ATTENTION_SCALE), "{arch}");
        let want = 1.0 / LOGIT_SCALE;
        assert_eq!(
            c.logit_multiplier,
            Some(want),
            "`{arch}` must carry 1/logit_scale, not logit_scale: granite.cpp:180 DIVIDES"
        );
    }
}

/// Fact 1: NORM RoPE, and every layer rotates.
///
/// `granite.cpp:206` gates RoPE on `rope_finetuned`, which defaults to
/// TRUE and which no Granite converter writes, so a real export always
/// rotates. Pinning the layout here is the same guard the 24-architecture
/// RoPE audit exists for: getting it wrong rotates the wrong pairs of
/// every Q and K head and answers fluently.
#[test]
fn the_granite_rows_use_the_norm_rope_llama_cpp_uses() {
    for arch in ALL_ROWS {
        let d = load_graph_fixture(arch);
        assert_eq!(
            d.config.rope_layout,
            RopeLayout::Norm,
            "`{arch}`: llama-model.cpp:2594-2595 puts it in the consecutive-pairs group"
        );
    }
}

/// Facts 2, 4, 5 and 6: no window, no post-norms, no QK-norm.
///
/// Asserted rather than assumed, because "the Gemma machinery is inert
/// here" is precisely the kind of claim that stops being true when a
/// default moves.
#[test]
fn the_granite_rows_have_no_window_no_post_norms_and_no_qk_norm() {
    for arch in ALL_ROWS {
        let d = load_graph_fixture(arch);
        assert!(d.config.sliding_window.is_none(), "{arch}");
        assert!(d.config.attn_logit_softcap.is_none(), "{arch}");
        assert!(d.config.final_logit_softcap.is_none(), "{arch}");
        for (il, layer) in d.layers.iter().enumerate() {
            assert!(layer.attn.post_attn_norm.is_none(), "{arch} blk.{il}");
            assert!(layer.attn.post_ffn_norm.is_none(), "{arch} blk.{il}");
            assert!(layer.attn.q_norm.is_none(), "{arch} blk.{il}");
            assert!(layer.attn.k_norm.is_none(), "{arch} blk.{il}");
        }
    }
}

/// The MoE half: softmax gating, renormalised top-k, no expert-weight
/// scale, and ONE ungated shared expert inferred from the tensors.
///
/// `expert_shared_count` is never written by the Granite converter, so
/// the shared expert has to be found from
/// `blk.0.ffn_gate_shexp.weight`; and `expert_weights_scale` is never
/// read by either Granite `load_arch_hparams`, so it must resolve to the
/// 1.0 that llama.cpp's `w_scale != 0 && w_scale != 1` guard makes
/// equivalent to its own `0.0f` default.
#[test]
fn granitemoe_routes_the_way_llama_cpps_build_moe_ffn_does() {
    for arch in ["granitemoe", "granite-moe"] {
        let d = load_graph_fixture(arch);
        assert_eq!(d.config.moe.gating, GatingFunction::Softmax, "{arch}");
        assert!(
            d.config.moe.norm_topk_prob,
            "`{arch}`: granite.cpp:273 passes norm_w = true"
        );
        assert_eq!(d.config.moe.expert_weights_scale, 1.0, "{arch}");
        assert_eq!(d.config.moe.n_experts_active, 2, "{arch}");
        for (il, layer) in d.layers.iter().enumerate() {
            assert_eq!(
                layer.moe.shared_experts.len(),
                1,
                "`{arch}` blk.{il}: one shared expert, inferred from the tensors"
            );
            assert!(
                layer.moe.shared_expert_gate.is_none(),
                "`{arch}` blk.{il}: granite.cpp:280-287 adds the shared expert UNGATED"
            );
        }
    }
}

/// The dense row really is dense, so the two fixtures are testing two
/// different FFN branches rather than the same one twice.
///
/// `granite.cpp:88-110` is one `if (n_expert == 0)`, and both fixtures
/// come out of it -- which is exactly why this has to be asserted. A
/// `granitemoe` fixture that had quietly loaded as dense would agree
/// with the golden on nothing and disagree loudly, but a `granite`
/// fixture that had somehow acquired experts would be a second copy of
/// the MoE test wearing the dense row's name.
#[test]
fn only_the_moe_rows_are_moe() {
    let dense = load_graph_fixture("granite");
    assert_eq!(
        dense.config.moe.n_experts, 1,
        "granite is the dense row; frink spells a dense FFN as one expert"
    );
    assert_eq!(dense.config.moe.n_shared_experts, 0, "granite");
    for arch in ["granitemoe", "granite-moe"] {
        assert_eq!(load_graph_fixture(arch).config.moe.n_experts, 4, "{arch}");
    }
}

// --- the refusal that became an implementation, and the one that did not

/// `{arch}.rope.scaling.finetuned = false` runs UNROTATED, as llama.cpp
/// runs it.
///
/// This was the half of the Granite verdict that stayed a refusal for
/// four days: `granite.cpp:33-35` reads the key as a switch for RoPE
/// ITSELF and `:130-133,:206-219` then build no positions and skip both
/// `ggml_rope_ext` calls. The reference prints `rope_finetuned =
/// unknown` for this fixture and returns logits that differ from
/// `granite_tiny.gguf`'s in the first digit (measured, and pinned
/// below). frink had no per-model way to express "no RoPE" until
/// `gpt2` closed on `RopeLayers::Never`, and Granite-4.0's converter
/// writes this key FALSE for every hybrid export
/// (`conversion/granite.py:253-256`), which is what turned the refusal
/// into `crate::rope_finetuned::unrotated`.
#[test]
fn a_granite_file_declaring_rope_finetuned_false_runs_unrotated_as_llama_cpp_does() {
    assert_all_three_paths_match("granite_norope", &GRANITE_NOROPE_GOLDEN);
    let d = load_graph_fixture("granite_norope");
    assert_eq!(
        d.config.rope_layers,
        frink_models::rope_layers::RopeLayers::Never
    );
    // The switch is visible: the rotated file's golden is elsewhere.
    assert!(worst_vs(&GRANITE_NOROPE_GOLDEN, &GRANITE_GOLDEN) > 1e-1);
}

/// ... and the same fixture WITHOUT that key loads, so the refusal is
/// keyed on the value and not on some other difference.
///
/// `granite_tiny.gguf` and `granite_norope_tiny.gguf` come from one
/// script and differ by exactly this key. Without this half, a
/// generator that had broken the variant file in some unrelated way
/// would still make the test above pass.
#[test]
fn the_same_fixture_without_that_key_loads_and_rotates() {
    let d = load_graph_fixture("granite");
    assert_eq!(d.config.rope_layout, RopeLayout::Norm);
}

/// ... but NOT for `granitemoe`, which runs granite's graph without
/// granite's hparam loader (#439).
///
/// `granite-moe.cpp:3-19` reads the four multipliers and the shared
/// expert width and nothing else, so `rope_pattern` keeps the all-ones
/// seed (`llama-model.cpp:1464`) and `granite.cpp:140,216`'s
/// `has_rope(il)` is true on every layer whatever the file declares.
/// libllama's logits for the file declaring `false` are bit-identical
/// to its logits for the file without the key (measured, and pinned by
/// the first assertion), and frink must rotate it the same way.
#[test]
fn a_granitemoe_file_declaring_rope_finetuned_false_still_rotates_as_llama_cpp_does() {
    assert_eq!(
        GRANITEMOE_NOROPE_GOLDEN, GRANITEMOE_GOLDEN,
        "libllama ignores the key on granitemoe"
    );
    assert_all_three_paths_match("granitemoe_norope", &GRANITEMOE_NOROPE_GOLDEN);
    let d = load_graph_fixture("granitemoe_norope");
    assert_ne!(
        d.config.rope_layers,
        frink_models::rope_layers::RopeLayers::Never
    );
}

/// `{arch}.logit_scale` is REQUIRED (`granite.cpp:7` reads it with no
/// default), so a Granite file that omits it is refused rather than
/// defaulted to 1.0.
///
/// llama.cpp cannot load such a file either -- measured: it dies in
/// `llama_model_loader::get_key` with `key not found in model:
/// granite.logit_scale`. Defaulting would mean frink answering,
/// fluently, on a file its own reference throws on, and the answer would
/// be off by whatever the checkpoint's real `logits_scaling` was, which
/// for Granite-3.0 is 8.
#[test]
fn a_granite_file_without_its_required_logit_scale_is_refused() {
    let path = graph_fixture_path("granite_nologit");
    let file = frink_gguf::GgufFile::open(&path).expect("the variant fixture parses");
    match ModelConfig::from_gguf(&file) {
        Err(LoadError::UnsupportedFeature(arch, msg)) => {
            assert_eq!(arch, "granite");
            assert!(msg.contains("granite.logit_scale"), "{msg}");
            assert!(msg.contains("REQUIRED"), "{msg}");
        }
        other => panic!("a Granite file with no logit_scale must be refused, got {other:?}"),
    }
}

// --- sabotage: each multiplier must be VISIBLE in the golden --------

/// Dropping the residual multiplier diverges from llama.cpp.
///
/// This is the one that reaches furthest into the engine:
/// `residual_scale` multiplies BOTH branch outputs of EVERY layer, which
/// in `decoder.rs` is eighteen separate residual adds across prefill,
/// decode, paged decode and continuous batching. If the fixture could
/// not see it, the claim that all eighteen agree would rest on reading
/// them.
#[test]
fn dropping_granites_residual_scale_diverges_from_llama_cpp() {
    for arch in ALL_ROWS {
        assert_sabotage_is_visible(arch, "residual_scale", |c| c.residual_scale = None);
    }
}

/// Dropping the logit multiplier diverges from llama.cpp.
///
/// A `logit_scale` of 2.5 means every logit is 2.5x too large without
/// it: a perfectly ordered distribution at the wrong temperature, which
/// changes nothing a greedy smoke test would notice and everything a
/// sampler would.
#[test]
fn dropping_granites_logit_multiplier_diverges_from_llama_cpp() {
    for arch in ALL_ROWS {
        assert_sabotage_is_visible(arch, "logit_multiplier", |c| c.logit_multiplier = None);
    }
}

/// INVERTING the logit multiplier diverges too.
///
/// The stronger half of the previous test. `granite.cpp:180` DIVIDES,
/// and frink stores the reciprocal; a reader of `lm_head.rs` cannot
/// tell which way round that should be, and multiplying by 2.5 instead
/// of by 0.4 is as ordinary-looking a distribution as the right one.
#[test]
fn inverting_granites_logit_multiplier_diverges_from_llama_cpp() {
    for arch in ALL_ROWS {
        assert_sabotage_is_visible(arch, "logit_multiplier inverted", |c| {
            c.logit_multiplier = Some(LOGIT_SCALE)
        });
    }
}

/// Dropping the embedding multiplier diverges from llama.cpp.
///
/// RMSNorm is scale-invariant, so this one is NOT visible through the
/// first `attn_norm`: it is visible because the residual stream carries
/// the scaled embedding while the branch outputs do not, so the ratio
/// between them -- and therefore everything after the final norm --
/// moves.
#[test]
fn dropping_granites_embedding_scale_diverges_from_llama_cpp() {
    for arch in ALL_ROWS {
        assert_sabotage_is_visible(arch, "embedding_scale", |c| c.embedding_scale = None);
    }
}

/// Dropping the attention scale diverges from llama.cpp.
///
/// Falling back to the kernels' own `1/sqrt(6) = 0.408` where the file
/// says 0.9 is a sharper softmax than the trained one -- the same
/// failure Gemma-2-27B and Gemma-3-27B shipped with.
#[test]
fn dropping_granites_attention_scale_diverges_from_llama_cpp() {
    for arch in ALL_ROWS {
        assert_sabotage_is_visible(arch, "attention_scale", |c| c.attention_scale = None);
    }
}

/// Rotating the wrong pairs diverges from llama.cpp.
///
/// The fixtures' Q and K are drawn wide precisely so this has a margin.
/// A fixture whose attention comes out near-uniform cannot see a
/// positional bug at all, and would "prove" the NORM/NEOX choice just as
/// well with it reversed.
#[test]
fn rotating_the_granite_rows_as_neox_diverges_from_llama_cpp() {
    for arch in ALL_ROWS {
        assert_sabotage_is_visible(arch, "NEOX RoPE", |c| c.rope_layout = RopeLayout::Neox);
    }
}

/// Load `arch`'s fixture with `mutate` applied to its config, and
/// require the result to be far from the golden.
///
/// `1e-2` is three orders of magnitude above `GRAPH_TOL`, so a sabotage
/// that only just clears it is reported rather than passing quietly.
fn assert_sabotage_is_visible(arch: &str, what: &str, mutate: impl FnOnce(&mut ModelConfig)) {
    let path = graph_fixture_path(arch);
    let file = frink_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    mutate(&mut config);
    let d = common::load_decoder(&path, config);
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        golden(arch),
    );
    assert!(
        worst > 1e-2,
        "`{arch}`: sabotaging {what} moved the logits by only {worst} (tolerance is \
         {GRAPH_TOL}); the fixture cannot see it"
    );
}
