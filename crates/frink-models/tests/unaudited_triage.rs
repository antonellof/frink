//! Triage of the `LoadError::UnauditedArchitecture` refusals.
//!
//! frink's generic GQA path is opt-in: an architecture that is not in
//! `capability::AUDITED_GENERIC_GQA` refuses rather than running on the
//! guess that it is plain GQA. That closed the "loads and computes
//! something else" class -- `gpt2`, `mpt`, `refact`, `bloom` and `jais`
//! all did exactly that -- but it left **47** architectures refusing with
//! one identical paragraph whose only content is "nobody has checked
//! this".
//!
//! That paragraph is useless for the decision a user actually has, which
//! is whether their model is one fixture away or needs an attention
//! implementation. `docs/plans/llama-cpp-gap-inventory.md` §1.3 shows
//! the 47 split at least three ways, and this suite pins the split for
//! the architectures that have been read on **both** sides:
//!
//! - **fixture-away** -- everything is implemented; evidence is missing.
//! - **one match arm** -- one small, nameable piece: an activation, a
//!   norm slot, a routing flag, an ordering.
//! - **new code** -- a different attention or residual structure.
//! - **UNKNOWN** -- reading did not settle it; the verdict says what
//!   would.
//!
//! **Why this suite exists rather than only the unit tests in
//! `capability.rs`:** the failure mode here is not a compile error, it
//! is a *confident wrong verdict*. This repo has now found four
//! architectures whose refusal named something that was not the real
//! blocker -- `glm4moe` was told it lacked an MLA hyper-parameter it must
//! not have, and `minimax-m2` was blamed on MTP weights no converter can
//! emit. Each assertion below therefore pins a specific claim about a
//! specific llama.cpp line, so that changing the verdict without
//! changing the reading fails.
//!
//! Every claim pinned here was read in
//! `.scratch/llama.cpp/src/models/*.cpp` against frink's generic
//! decoder, and the citation is in the verdict string itself. One test
//! is the exception and reads FILES rather than source --
//! `real_mistral_and_yi_checkpoints_declare_llama`, `#[ignore]`d
//! because it needs the checkpoints in `models/`. It is the measurement
//! that closed the last three UNKNOWN rows.

mod common;
use common::collect_gguf;
use frink_models::capability::{
    architecture_catalog, is_audited_generic, unaudited_refusal_detail, unaudited_triage, ArchPath,
    TriageClass, TRIAGE_PENDING,
};

/// Every architecture that reaches the unaudited refusal gets a
/// non-empty detail line -- triaged or not.
///
/// A blank detail would be the old refusal wearing a new field.
#[test]
fn every_unaudited_architecture_renders_a_detail_line() {
    let mut n = 0;
    for p in architecture_catalog() {
        if !matches!(p.path, ArchPath::GenericGqa { .. }) || is_audited_generic(p.gguf_name) {
            continue;
        }
        n += 1;
        let detail = unaudited_refusal_detail(p.gguf_name);
        assert!(
            detail.starts_with("TRIAGE"),
            "`{}` renders {detail:?}",
            p.gguf_name
        );
        assert!(detail.len() > 100, "`{}` renders {detail:?}", p.gguf_name);
    }
    assert_eq!(
        n, 4,
        "the unaudited count moved. It was 47 until the triage itself found `minicpm3` was \
         an MLA model sitting on the generic-GQA row and it was reclassified to \
         DedicatedOnly, 46 until `deepseek`, `bailingmoe`, `seed_oss`, `maincoder` and \
         `hunyuan-moe` were admitted with libllama-golden fixtures, 41 until \
         `internlm2`, `xverse`, `ernie4_5`, `baichuan`, `exaone`, `bailingmoe2` and \
         `plamo3` were \
         admitted with theirs (`tests/fixture_away_graphs.rs`), 34 until `gemma`, \
         `hunyuan-dense` and `ernie4_5-moe` were admitted with theirs, 31 until \
         `olmo2` and `exaone4` were -- the first two NEW CODE rows to close, and they \
         closed TOGETHER because they are one topology with one implementation \
         (`frink_models::norm`, `tests/post_norm_only_graphs.rs`) -- 29 until \
         `chatglm` was admitted with the fused-QKV-bias arm and its fixture, 28 until \
         `mistral`, `mixtral` and `yi` were found not to be architectures at all and \
         moved to DedicatedOnly, and 25 until `granite`, `granitemoe` and the \
         `granite-moe` alias closed together on ONE implementation of their four scalar \
         multipliers (`tests/granite_family_graphs.rs`), and 22 until `olmo` closed on \
         the non-parametric LayerNorm (`frink_models::norm`, `tests/olmo_graphs.rs`) -- \
         the FIRST NEW CODE row to close alone, and it closed alone because its cause \
         really is unshared: every `build_norm` call in llama.cpp's 155 graphs was \
         scanned for a null weight and all three hits are `olmo.cpp`, and 21 until \
         `exaone-moe` closed on the per-layer RoPE gate (`frink_models::rope_layers`, \
         `tests/no_rope_layer_graphs.rs`) -- ONE cause behind THREE refusals, of which \
         only this row was in this count: EXAONE-4 32B was refused BY NAME in loader.rs \
         and `smollm3` sat in the \"No RoPE at all\" DedicatedOnly group, so both closed \
         with it and neither moved this number, and 20 until `grok` and `dbrx` closed \
         together on seams landed the day before (`scalar_multipliers::MultiplierDefaults`, \
         `norm::NormOp::LayerNorm`, `clamp_kqv`, `norm_sites`; tests/grok_graphs.rs, \
         tests/dbrx_graphs.rs), the clamp also closing `olmo`'s clip_qkv refusal by name, \
         and 18 until `arcee` closed on the ungated ReLU-squared FFN \
         (`FfnActivation::ReluSqr`, tests/ungated_ffn_graphs.rs) and `deci` and `openelm` \
         closed together on the per-layer shape seam (`frink_models::layer_shapes`, \
         tests/per_layer_shape_graphs.rs), and 15 until `afmoe` and `laguna` closed \
         together on the gated attention (`frink_models::attn_gate`, \
         tests/gated_attention_graphs.rs) -- ONE cause behind THREE verdicts, read side by \
         side and found to be one op with two free parameters; `step35`, the third, still \
         needs its per-layer clamp arrays and window array and says so, and 13 until \
         `mellum` closed on the per-layer sliding-window ARRAY seam \
         (`frink_models::swa_layers`, tests/window_array_graphs.rs) -- the seam three \
         verdicts named, of which `mellum` is the only generic-path graph that HONOURS the \
         array; the same seam lifted an over-refusal on every real EXAONE-4 32B, \
         EXAONE-MoE and Olmo-3 export, whose arrays llama.cpp IGNORES, and \
         `frink_models::mtp_blocks` beside it skips the NextN blocks `mimo2` and `step35` \
         named, so both say so and lead with what is left, and 12 until `apertus` and \
         `step35` closed together on the per-layer ACTIVATION PARAMETER seam \
         (`frink_models::act_layers`, tests/per_layer_activation_graphs.rs, \
         tests/clamped_swiglu_graphs.rs) -- ONE plumbing question behind two verdicts, \
         `layer il runs its FFN activation with these scalars`, and TWO bodies, xIELU and \
         the clamped SwiGLU, with the site (routed versus dense) the one thing the second \
         needed of the plumbing that the first did not; `step35`'s half-width rotary \
         landed on `frink_models::swa_geometry` as the two-valued width llama.cpp's \
         `n_rot(il)` already was, and lifted Laguna-XS.2's `rope.dimension_count_swa` \
         refusal by name with it, and 10 until `mistral3` closed on the per-position \
         attention temperature (`frink_models::attn_temperature`, \
         tests/attn_temperature_graphs.rs) -- the reach measured first, three graphs of \
         140, the other two on their own engines: `llama4` seeds the constants from \
         literals and `deepseek2` / `mistral4` read the same key, which the MLA loader now \
         refuses by name where it dropped it; its `yarn_log_multiplier` half found YaRN's \
         magnitude term missing for every architecture (`frink_models::yarn_magnitude`), \
         and 9 until `smallthinker` closed on the router operand \
         (`frink_models::router_input`, tests/router_input_graphs.rs) -- the reach \
         measured first over all 59 `build_moe_ffn` call sites: four pass a precomputed \
         `probs_in`, and only this one on the generic path routes on something other \
         than the normed FFN input; its gated ReLU experts split `GluAct::ReluSqr` from \
         `GluAct::Reglu`, because the one variant that had served `arcee` by aliasing \
         answered `relu(up)^2` for a real gate, and its `n_swa = 4096` pin is a third \
         answer on the one table `swa_disabled_by_arch` is derived from, and 8 until \
         `bitnet` closed on the two norms INSIDE the blocks (`frink_models::sub_norms`, \
         tests/sub_norm_graphs.rs) -- one graph of 155 creates either tensor, measured, \
         so the seam is a `bool` on `ModelConfig` and the row closed alone; its optional \
         per-projection `.scale` tensors are a refusal by name now \
         (`frink_models::weight_scales`) from a fixture whose libllama logits differ \
         from the unscaled file's, and 7 until `mimo2` closed on the split K/V head \
         width (`frink_models::kv_head_dims`, tests/split_kv_head_dim_graphs.rs) -- \
         three converters of fourteen write `value_length` apart from `key_length`, one \
         on this engine; `KvCache`, `PagedKvStore`, the one row kernel the three \
         contiguous arms collapsed onto, the batched prefill kernel, the projection \
         check and the fused-QKV cut all took the V width, and building it found \
         `expert_weights_scale` honoured for every architecture where llama.cpp reads it \
         in twenty loaders (`EXPERT_WEIGHTS_SCALE_READERS`), and 6 until `nanbeige` \
         closed on the layer loop (`frink_models::layer_loops`, \
         tests/layer_loop_graphs.rs) -- one graph of 155 reads `num_loops`; the weights \
         are shared and the KV is not, so `Decoder::layers` stays physical, `n_layers` is \
         logical, and one mapping serves the three bodies, and 5 until `talkie` closed \
         on its four things at once (`frink_models::skip_stream`, `NormOp::RmsNoParams`, \
         `QkNormStyle::PerHeadScalar`, and the two `.scale` companions \
         `frink_models::weight_scales` now serves; tests/skip_stream_graphs.rs), and 4 \
         until `plm` closed on the MLA engine (`frink_models::mla_arch`, \
         `frink_models::mla_q_proj`, tests/plm_graphs.rs) -- the reach measured first: \
         six graphs of 155 create `attn_kv_a_mqa` and three have a direct `attn_q` \
         beside it, and on that engine the direct form is `plm` and every LITE \
         `deepseek2` (`deepseek2.cpp:8`, decided from the layer count), which the loader \
         had refused for a `q_lora_rank` llama.cpp never reads there; the fixture is the \
         MLA engine's FIRST libllama golden, and 3 until `arctic` closed on the parallel \
         dense + MoE layer (`frink_models::parallel_dense_ffn`, \
         `RouterInput::NormedLayerInput`, tests/parallel_dense_ffn_graphs.rs) -- the reach \
         measured first: two graphs of 155 SUM a dense FFN with their routed output, and \
         the other is Grok-2, refused by name until then from a fixture that has a golden \
         now; the branch operand is one graph of 155, a third variant of the seam \
         `smallthinker` opened, and the verdict that said the seam did not reach it \
         was read before it was believed \
         -- rows closing is the count going DOWN for the best reason. Either an \
         architecture was audited or reclassified (good -- update the count and the docs) \
         or one was added (check it was triaged). ON 2026-09-19 IT WENT UP, 2 to 10, for \
         the third reason: the llama.cpp PIN moved (2026-08-04 to `5b59b83`, 792 commits, \
         fifteen new graphs) and eight of the new architectures are generic-path \
         candidates that had to be read before they could be refused honestly -- \
         `granite_swa`, `graniteswitch`, `muse-glimmer`, `maple`, `spark2_5`, `hrm_text`, \
         `minimax-01`, `qwen4exp`. A parity count against a moving upstream goes UP when \
         the pin moves and DOWN when a row closes, and one that only ever went down would \
         mean nobody was reading upstream, and 10 to 4 as SIX of those eight closed: \
         `spark2_5` and `maple` on one match arm each, `granite_swa` and `muse-glimmer` \
         the same day, `hrm_text` the next, and `minimax-01` on 2026-09-20, whose \
         lightning-attention block went on the `AttnShape` seam the Mamba and Qwen3.5 rows \
         built and whose recurrent mask went on `gdn::recurrent_layers`, so what actually \
         needed new code was the RESIDUAL -- `src/models/minimax-01.cpp:249,428` make each \
         sublayer's pre-norm output, scaled, the stream its branch joins, which is ONE \
         graph of the 155 (`frink_models::normed_residual`)"
    );
}

/// Batch 1: the architectures people actually download, with the class
/// each was placed in and the llama.cpp fact that decides it.
///
/// The class is asserted together with a substring of the blocker on
/// purpose. Asserting the class alone would let somebody flip a verdict
/// and leave the (now-contradictory) reasoning in place, which is
/// exactly how `glm4moe` came to refuse for a reason it did not have.
#[test]
fn batch_one_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        // --- fixture-away: implemented, unevidenced ------------------
        //
        // THIS CLASS IS NOW EMPTY. `gemma` was the last row in it and
        // got its fixture (`tests/fixture_away_graphs.rs`), so every
        // architecture still refusing needs code, not evidence. That is
        // the honest headline and
        // `every_unaudited_row_is_triaged_and_the_distribution_is_pinned`
        // is what holds it.
        //
        // `internlm2`, `exaone` and `ernie4_5` were HERE, and so were
        // `xverse` and `baichuan` in batch three. All five got their
        // fixture (`tests/fixture_away_graphs.rs`), so they are audited
        // now and carry no verdict at all. So did `bailingmoe2`, the one
        // MoE row of the six --
        // `every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited`
        // is what stops a stale verdict outliving its refusal. Closing a
        // FIXTURE-AWAY row is the cheapest kind of progress there is and
        // the count going down here is what it looks like.
        //
        // --- one match arm: small and nameable -----------------------
        //
        // `seed_oss`, `deepseek` and `hunyuan-moe` were HERE. All three
        // arms landed, with libllama-golden fixtures
        // (`tests/one_match_arm_graphs.rs`), so they are audited now and
        // carry no verdict at all —
        // `every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited`
        // is what stops a stale verdict outliving its refusal.
        //
        // `ernie4_5-moe` was HERE, ONE MATCH ARM on
        // `interleave_moe_layer_step`. The arm landed as a REFUSAL
        // rather than an implementation -- llama.cpp's own tensor loader
        // (ernie4-5.cpp:49) has no step in it, so an interleaved
        // checkpoint cannot be loaded by llama.cpp either -- and the
        // step every real checkpoint carries is audited against libllama
        // (`tests/one_match_arm_graphs.rs`), so the row carries no
        // verdict at all.
        //
        // --- new code: a different graph -----------------------------
        //
        // `olmo2` and `exaone4` used to head this group -- no attn_norm
        // and no ffn_norm at all, Q/K/V off the raw residual. That
        // blocker was real and it is now IMPLEMENTED, once, for both
        // (`frink_models::norm`), so neither carries a verdict any
        // more, and `the_post_norm_group_is_three_topologies_and_only_two_of_them_closed` below
        // is now about which of the three shapes each of the group is.
        //
        // `granite`, `granitemoe` and the `granite-moe` alias were HERE
        // too, NEW CODE on the four scalar multipliers
        // (granite.cpp:5-10,180,225,235-238,288-292). All three closed
        // together, on ONE implementation of the multipliers
        // (`frink_models::scalar_multipliers`) rather than three, and
        // all three have libllama-golden fixtures
        // (`tests/granite_family_graphs.rs`) -- the alias by reading
        // `granitemoe`'s golden out of a file that differs only in its
        // architecture string, because no llama.cpp GGUF spells it that
        // way and none ever will. They carry no verdict now;
        // `every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited`
        // is what stops a stale one outliving its refusal.
        //
        // BOTH closures took more than one row at a time, and for the
        // same reason: each found ONE cause behind several refusals.
        // That is what the NEW CODE column moving looks like.
        //
        // The `rope_finetuned` half of the Granite verdict did NOT
        // become an implementation. granite.cpp:33-35 reads
        // `{arch}.rope.scaling.finetuned` as a switch for RoPE itself,
        // so a file declaring it false runs unrotated in llama.cpp and
        // frink refuses it by name (`frink_models::rope_finetuned`).
        // --- unknown: say so, and say what would settle it -----------
        //
        // `phi4` is not a llama.cpp architecture at all, so there is no
        // graph to diff against.
        ("phi4", TriageClass::Unknown, "WHAT WOULD SETTLE IT"),
    ];

    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// The "post-norm group" was never one group, and it is THREE norm
/// topologies, not two.
///
/// `docs/plans/llama-cpp-gap-inventory.md` §1.3 grouped `olmo2`,
/// `seed_oss` and `exaone4` together as "likely a fixture away, if the
/// loader wires the post-norm slots for non-Gemma families". The wiring
/// question had a yes answer, and it was the wrong question: the three
/// are three different residual shapes and were never one class.
///
/// * `seed_oss` HAS `attn_norm` and uses `attn_post_norm` as its
///   pre-FFN norm (`seed-oss.cpp:36-37,113-115`). Closed 2026-09-02.
/// * `olmo2` and `exaone4` have NEITHER pre-norm and read the raw
///   residual at both sublayers (`olmo2.cpp:45-52,92,169`,
///   `exaone4.cpp:60-67,118,159`). That is one topology across the two,
///   `frink_models::norm` is the one implementation, and
///   `tests/post_norm_only_graphs.rs` is the evidence for both.
/// * `olmo` (OLMo-1) is the third: it norms BEFORE both sublayers, so
///   it is pre-norm like llama, and what it lacks is the norm FUNCTION
///   -- non-parametric LayerNorm, all three `build_norm` calls with a
///   NULL weight (`olmo.cpp:65-67,104-106,128-130`). It still refuses,
///   and `norm` is no help to it.
///
/// Pinning the split stops the grouping being restored from the prose,
/// and it now also pins that closing two of the three did NOT sweep the
/// third along with them.
#[test]
fn the_post_norm_group_is_three_topologies_and_only_two_of_them_closed() {
    // seed_oss: has a pre-attention norm; its post_attention_norm IS
    // the pre-FFN norm.
    assert!(
        is_audited_generic("seed_oss") && unaudited_triage("seed_oss").is_none(),
        "seed_oss has attn_norm and olmo2 does not; they were never one class"
    );

    // olmo2 / exaone4: one topology, one implementation, both audited.
    for name in ["olmo2", "exaone4"] {
        assert!(
            is_audited_generic(name),
            "`{name}` closed with a libllama-golden fixture"
        );
        assert!(
            unaudited_triage(name).is_none(),
            "`{name}` is audited and must carry no verdict"
        );
        assert!(
            frink_models::capability::is_post_norm_only(name),
            "`{name}` is the post-norm-only topology"
        );
    }
    assert_eq!(
        frink_models::capability::POST_NORM_ONLY_ARCHITECTURES,
        &["olmo2", "exaone4"],
        "a third name here is a third llama.cpp graph somebody read"
    );

    // olmo (OLMo-1) is the THIRD topology and it closed too, on a
    // different variant of the same enum -- pre-norm, with a
    // non-parametric LayerNorm at all three sites
    // (`tests/olmo_graphs.rs`). What still has to hold is that it is
    // NOT on the post-norm-only list: a decoder that read OLMo-1 that
    // way would drop both its norms and answer fluently.
    assert!(is_audited_generic("olmo"));
    assert!(
        unaudited_triage("olmo").is_none(),
        "`olmo` is audited and must carry no verdict"
    );
    assert!(
        !frink_models::capability::is_post_norm_only("olmo"),
        "OLMo-1 norms before both sublayers; it is not post-norm-only"
    );
    assert!(
        frink_models::capability::uses_non_parametric_layer_norm("olmo"),
        "OLMo-1's norm has no parameters, which is the whole row"
    );
    assert_eq!(
        frink_models::capability::NON_PARAMETRIC_LAYER_NORM,
        &["olmo"],
        "a second name here would be a second llama.cpp graph with a null-weight \
         `LLM_NORM`, and the scan over all 155 found none"
    );
}

/// `ernie4_5-moe` is audited, and the sigmoid correction moved from its
/// verdict into a golden test rather than being dropped.
///
/// The inventory (§1.3) lists it beside `bailingmoe2` as "sigmoid-routed
/// MoE with `ffn_exp_probs_b` router bias". `bailingmoe2` reads its
/// gating function from metadata (`bailingmoe2.cpp:11`) so it can be
/// either; `ernie4-5-moe.cpp:90` **hardcodes**
/// `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`. That correction used to live
/// in the refusal string, which is gone now, so this pins the join: the
/// row really did move to the audited side, and the claim it used to
/// make in words is now made in logits by
/// `routing_ernie_moe_through_sigmoid_instead_of_softmax_diverges_from_llama_cpp`.
#[test]
fn ernie_moe_is_audited_and_carries_no_stale_sigmoid_verdict() {
    assert!(
        is_audited_generic("ernie4_5-moe"),
        "ernie4_5-moe was admitted with a libllama-golden fixture at step 1"
    );
    assert!(
        unaudited_triage("ernie4_5-moe").is_none(),
        "an audited row must carry no verdict; the interleave step it used to describe is \
         now a named refusal in `moe_interleave` for any step above 1"
    );
}

/// Every triaged architecture really does reach the unaudited refusal.
///
/// A verdict on a row that refuses for a *different*, named reason would
/// be dead text the user never sees -- the same shape as the `glm4moe`
/// bug, where the reason shown and the reason true were two different
/// strings.
#[test]
fn every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited() {
    for p in architecture_catalog() {
        let Some(t) = p.triage else { continue };
        assert!(
            matches!(p.path, ArchPath::GenericGqa { .. }),
            "`{}` carries a {:?} verdict but resolves to {:?}, which refuses elsewhere",
            p.gguf_name,
            t.class,
            p.path
        );
        assert!(
            !is_audited_generic(p.gguf_name),
            "`{}` is audited and runs; a triage verdict there is never rendered",
            p.gguf_name
        );
    }
}

/// The pending list is a to-do with a shrinking count, not a parking
/// bay.
///
/// If this number goes UP without the total moving, an architecture lost
/// its verdict.
#[test]
fn the_remaining_work_is_counted() {
    assert_eq!(
        TRIAGE_PENDING.len(),
        0,
        "all 47 unaudited architectures are triaged; a name reappearing here means a new \
         architecture reached the generic path without being read"
    );
    let triaged = architecture_catalog()
        .iter()
        .filter(|p| p.triage.is_some())
        .count();
    // 2 before the 2026-09-19 pin move (`grovemoe`, `phi4`), plus the
    // eight upstream architectures it brought in.
    assert_eq!(triaged + TRIAGE_PENDING.len(), 4);
}

/// `minicpm3` is refused as an MLA model, not as an unaudited one.
///
/// The triage found the catalog claimed `StandardGqa`/`KvGqa` for a
/// model whose every checkpoint carries `attn_q_a`/`attn_kv_a_mqa` and
/// no `attn_q.weight` (`src/models/minicpm3.cpp:41-46`), so the generic
/// path could never have loaded one. Being told "unaudited" for that is
/// telling the user the wrong thing about their model.
///
/// A message-quality fix rather than a correctness one -- the old
/// failure was already a clean missing-tensor error -- which is why the
/// reason has to name BOTH blockers, the MLA tensor set and the
/// hardcoded MiniCPM multipliers.
#[test]
fn minicpm3_is_refused_as_mla_not_as_unaudited() {
    assert!(
        unaudited_triage("minicpm3").is_none(),
        "minicpm3 left the unaudited generic set"
    );
    match frink_models::capability::resolve_architecture("minicpm3") {
        Some(ArchPath::DedicatedOnly { reason }) => {
            assert!(reason.contains("MLA"), "{reason}");
            assert!(
                reason.contains("scale_depth"),
                "the multipliers are the second blocker and must be named: {reason}"
            );
        }
        other => panic!("minicpm3 must be DedicatedOnly, got {other:?}"),
    }
}

/// The untriaged message still works, and still claims no class.
///
/// `TRIAGE_PENDING` is empty now that all 47 are read, so this exercises
/// the branch through a name the catalog does not carry. It is what a
/// NEW architecture added to the generic path would render until
/// somebody reads it, and it must not imply a class: "unaudited" and
/// "untriaged" are different claims, and reading a class into the
/// untriaged message is how a guess becomes a citation.
#[test]
fn the_untriaged_message_claims_no_class() {
    let d = unaudited_refusal_detail("an-architecture-nobody-has-read-yet");
    assert!(d.contains("not done for"), "{d}");
    for label in [
        TriageClass::FixtureAway.label(),
        TriageClass::OneMatchArm.label(),
        TriageClass::NewCode.label(),
    ] {
        assert!(
            !d.contains(label),
            "the untriaged message implies {label}: {d}"
        );
    }
}

/// Batch 2: the next six by download volume, plus the two the
/// activation audit turned up on the way.
///
/// Every one came out `NewCode`, which is itself the finding. Batch 1
/// mixed five fixture-away rows in; past the first dozen the unaudited
/// set is genuinely harder, and the refusal now says so per
/// architecture instead of implying a uniform distance.
#[test]
fn batch_two_verdicts_are_pinned_to_what_was_read() {
    // `grok` and `dbrx` were the first two rows here and are audited
    // now (tests/grok_graphs.rs, tests/dbrx_graphs.rs): the defaults
    // hook, the weighted LayerNorm, the QKV clamp and the norm-site
    // table each turned out to be one column on a seam that already
    // existed. Their absence from this list is asserted by
    // `grok_and_dbrx_are_audited_and_carry_no_stale_verdict` below.
    let cases: &[(&str, TriageClass, &str)] = &[
        // `smallthinker` was HERE, NEW CODE on the router operand, and
        // is audited now (tests/router_input_graphs.rs); its absence is
        // asserted by `smallthinker_is_audited_and_carries_no_stale_verdict`
        // below.
        // `bitnet` was HERE, NEW CODE on `attn_sub_norm` / `ffn_sub_norm`,
        // and is audited now (tests/sub_norm_graphs.rs); its absence is
        // asserted by `bitnet_is_audited_and_carries_no_stale_verdict`
        // below.
        // `minicpm3` was HERE, and the triage that produced this list
        // is what removed it: reading `minicpm3.cpp:5-6,41-46` showed an
        // MLA tensor set on a row the catalog called `StandardGqa`, so
        // it moved to `DedicatedOnly` rather than staying an unaudited
        // generic architecture. Its refusal is now asserted by
        // `minicpm3_is_refused_as_mla_not_as_unaudited` below.
        // `openelm` was HERE, NEW CODE on per-layer head counts, and is
        // audited now with `deci` on one seam
        // (`frink_models::layer_shapes`, tests/per_layer_shape_graphs.rs).
        // `arcee` was HERE too, NEW CODE on the ungated ReLU-squared
        // FFN, and is audited (tests/ungated_ffn_graphs.rs). It shared
        // its verdict constant with `plm`, and that constant was WRONG
        // about `plm` by an attention block: `plm.cpp:84-166` is
        // DeepSeek-2's MLA attention, which the FFN-only reading had
        // not seen. `plm` was HERE with that corrected verdict, and
        // closed on 2026-09-12 on the MLA engine (`DecoderFamily::Mla`,
        // tests/plm_graphs.rs), where its `ArchPath::DedicatedOnly`
        // profile carries no verdict; `plm_is_served_by_the_mla_engine`
        // below asserts the move.
    ];
    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// `plm` left the triage table for the MLA engine: no verdict, the
/// `Mla` family, a `DedicatedOnly` path that names the engine, and the
/// generic loader still refusing it by that path rather than as
/// unaudited.
#[test]
fn plm_is_served_by_the_mla_engine() {
    use frink_models::capability::{resolve_profile, ArchPath, DecoderFamily};
    assert!(
        unaudited_triage("plm").is_none(),
        "an MLA-engine row carries no triage verdict"
    );
    let p = resolve_profile("plm").expect("registered");
    assert_eq!(p.family, DecoderFamily::Mla);
    match p.path {
        ArchPath::DedicatedOnly { reason } => assert!(reason.contains("MLA"), "{reason}"),
        other => panic!("plm is DedicatedOnly, not {other:?}"),
    }
    assert!(frink_models::mla_arch::mla_arch("plm").is_some());
}

/// `smallthinker`'s verdict led with the routing input, then the
/// activation, then the window pin, and said the NoPE layers were no
/// longer a blocker. All three landed (`frink_models::router_input`,
/// `FfnActivation::Reglu`, `capability::swa_window_override`) and the
/// row is audited on three libllama-golden fixtures
/// (tests/router_input_graphs.rs), so it carries no verdict: a verdict
/// on an audited row is never rendered and would be dead text.
#[test]
fn smallthinker_is_audited_and_carries_no_stale_verdict() {
    assert!(is_audited_generic("smallthinker"));
    assert!(unaudited_triage("smallthinker").is_none());
    // The seam's own census agrees about which row it serves.
    assert!(frink_models::router_input::ROUTER_INPUT_TABLE
        .iter()
        .any(|(name, input, _)| *name == "smallthinker"
            && *input == frink_models::router_input::RouterInput::RawLayerInput));
    // `arctic`'s verdict used to say it shares `smallthinker`'s shape,
    // then that the seam "does not reach it"; it reaches it now as a
    // third variant, `NormedLayerInput`, and the row is audited
    // (tests/parallel_dense_ffn_graphs.rs).
    assert!(is_audited_generic("arctic"));
    assert!(unaudited_triage("arctic").is_none());
    assert!(frink_models::router_input::ROUTER_INPUT_TABLE
        .iter()
        .any(|(name, input, _)| *name == "arctic"
            && *input == frink_models::router_input::RouterInput::NormedLayerInput));
}

/// `bitnet`'s verdict named four things: the two inner norms, the
/// per-projection `.scale` tensors, and the missing `output` tensor.
/// The norms landed (`frink_models::sub_norms`), the scales are a
/// refusal by name (`frink_models::weight_scales`), the tied lm_head
/// was already served, and the row is audited on a libllama-golden
/// fixture (tests/sub_norm_graphs.rs), so it carries no verdict.
#[test]
fn bitnet_is_audited_and_carries_no_stale_verdict() {
    assert!(is_audited_generic("bitnet"));
    assert!(unaudited_triage("bitnet").is_none());
    // The seam's own census agrees about which row it serves.
    assert!(frink_models::sub_norms::block_sub_norms("bitnet"));
    assert!(frink_models::sub_norms::SUB_NORM_ARCHS
        .iter()
        .all(|(name, _)| is_audited_generic(name)));
}

/// `nanbeige`'s verdict named one thing, the layer loop, and said the
/// per-layer arrays were not the blocker; the loop landed
/// (`frink_models::layer_loops`), the arrays are replicated as
/// `nanbeige.cpp:24-26` replicates them, and the row is audited on
/// three libllama-golden fixtures (tests/layer_loop_graphs.rs), so it
/// carries no verdict.
#[test]
fn nanbeige_is_audited_and_carries_no_stale_verdict() {
    assert!(is_audited_generic("nanbeige"));
    assert!(unaudited_triage("nanbeige").is_none());
    assert!(frink_models::layer_loops::LOOP_READERS
        .iter()
        .all(|(name, _)| is_audited_generic(name)));
}

/// `talkie`'s verdict named four things and all four landed: the
/// weightless norms (`NormOp::RmsNoParams`), the per-head scalar Q gain
/// with the weightless K norm (`QkNormStyle::PerHeadScalar`), the
/// embedding skip stream (`frink_models::skip_stream`), and the two
/// projection gains its converter writes (`frink_models::weight_scales`
/// serves exactly those two). Audited on two libllama-golden fixtures
/// (tests/skip_stream_graphs.rs), so it carries no verdict.
#[test]
fn talkie_is_audited_and_carries_no_stale_verdict() {
    assert!(is_audited_generic("talkie"));
    assert!(unaudited_triage("talkie").is_none());
    assert!(frink_models::skip_stream::has_skip_stream("talkie"));
    assert!(frink_models::capability::uses_non_parametric_rms_norm(
        "talkie"
    ));
    assert!(frink_models::capability::uses_per_head_scalar_qk_gain(
        "talkie"
    ));
    assert_eq!(
        frink_models::weight_scales::SERVED_SCALE_TENSORS,
        ["attn_output.scale", "ffn_down.scale"]
    );
}

/// `grok` and `dbrx` are audited, and neither carries a verdict any
/// more.
///
/// A stale verdict on an admitted row is the `glm4moe` shape: a refusal
/// message nobody can reach that still reads as a claim. Both rows were
/// NEW CODE and both closed on seams landed the day before, which is the
/// outcome the honest position says the column moves on.
#[test]
fn grok_and_dbrx_are_audited_and_carry_no_stale_verdict() {
    for arch in ["grok", "dbrx"] {
        assert!(is_audited_generic(arch), "`{arch}` must be audited");
        assert!(
            unaudited_triage(arch).is_none(),
            "`{arch}` is audited and must not also carry a verdict"
        );
    }
}

/// Where the refusal a user actually sees is NOT this one, the verdict
/// says so rather than letting the reader assume the triage line is what
/// they got.
///
/// `openelm` was the row that pinned this rule: it died on a
/// missing-hparam error for keys its file does carry, before the
/// unaudited gate, because `GgufValue::as_u64` returned `None` for the
/// per-layer ARRAYS its converter writes. That is gone with the row --
/// `layer_shapes::read_u64_per_layer` reads both spellings and the
/// architecture is audited -- and `granite` had left this list the
/// same way before it, and `mimo2` -- refused by the loader's `split
/// K/V head dims` check before the unaudited gate, which its verdict
/// led with -- closed on `frink_models::kv_head_dims` after both.
/// With no live example the rule is pinned on the property it exists
/// for: a verdict for an architecture that is refused EARLIER, by
/// name, must say so, and every row that has carried the rule is
/// audited now, so the misleading messages they disclosed cannot be
/// produced any more.
#[test]
fn verdicts_disclose_when_an_earlier_refusal_fires_first() {
    for arch in ["openelm", "granite", "mimo2"] {
        assert!(unaudited_triage(arch).is_none(), "{arch}");
        assert!(frink_models::capability::is_audited_generic(arch), "{arch}");
    }
    // The gate that used to fire first for `mimo2` admits its real pair
    // now and still refuses it for everyone else, naming the assert.
    assert!(frink_models::kv_head_dims::resolve_v_head_dim("mimo2", 192, Some(128)).is_ok());
    let err = frink_models::kv_head_dims::resolve_v_head_dim("llama", 192, Some(128))
        .expect_err("llama asserts the widths equal");
    assert!(err.to_string().contains("split K/V head dims"));
}

/// Batch 3: the alias rows and the plain long-tail.
///
/// This is the batch that was expected to be cheap, and half of it was.
/// `xverse` and `baichuan` really were llama-shaped, are audited now
/// (`tests/fixture_away_graphs.rs`) and so carry no verdict any more.
/// `deci` and `olmo` are not llama-shaped, and neither is the alias
/// trio, for a reason that has nothing to do with their graphs.
#[test]
fn batch_three_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        // `chatglm` was FIXTURE-AWAY here, then ONE MATCH ARM once
        // somebody tried to build its fixture and read the converter.
        // The arm -- the fused `attn_qkv.bias` -- landed in
        // `qkv_fused`, so the row is audited and carries no verdict at
        // all. It was the LAST one-match-arm row anywhere; see
        // `tests/one_match_arm_graphs.rs`.
        //
        // `mistral`, `mixtral` and `yi` were HERE too, UNKNOWN on "what
        // would settle it: a real GGUF spelling one of these". The
        // answer came back NO -- libllama refuses all three strings --
        // so they are refused as strings now, not triaged as
        // architectures. See
        // `the_alias_rows_are_refused_as_strings_no_converter_writes`.
        // `deci` was HERE, NEW CODE on per-layer shapes with a three-way
        // branch on them, and is audited now with `openelm` on one seam
        // (`frink_models::layer_shapes`, tests/per_layer_shape_graphs.rs);
        // the branch combination llama.cpp handles by discarding a
        // computed attention output is refused by name from a fixture
        // that has it, and `the_per_layer_shape_seam_...` below is where
        // the claim about the reach of that seam lives.
        //
        // `olmo` was HERE, NEW CODE on "NO norm weights at all". It is
        // audited now (`tests/olmo_graphs.rs`) and carries no verdict;
        // `the_post_norm_group_is_three_topologies...` above is where
        // the claim about it lives.
    ];
    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
    // The batch is EMPTY now -- every row it held either closed or
    // turned out not to be an architecture -- so the loop above cannot
    // fail, and what this test pins instead is that none of them came
    // back: an audited row carrying a verdict would be a stale claim
    // presented as a current one.
    for arch in ["xverse", "baichuan", "chatglm", "deci", "olmo"] {
        assert!(
            is_audited_generic(arch) && unaudited_triage(arch).is_none(),
            "`{arch}` closed out of batch three and must stay audited with no verdict"
        );
    }
}

/// The per-layer shape seam (`frink_models::layer_shapes`) closed
/// `deci` and `openelm` and reaches three more rows, and each of those
/// three says so: the verdict names the per-layer half as done and the
/// remaining blocker as something else.
///
/// The reach was MEASURED before the seam was built -- all 155
/// `src/models/*.cpp` scanned for `n_head(i)` / `n_head_kv(i)` /
/// `n_ff(i)` in both the tensor loader and the graph -- and
/// `PER_LAYER_SHAPE_ARCHS` is the record. A verdict that still called
/// per-layer heads the blocker on a row the seam serves would be the
/// confident wrong verdict this suite exists to catch.
#[test]
fn the_per_layer_shape_seam_closed_two_rows_and_its_reach_is_recorded_on_the_rest() {
    use frink_models::layer_shapes::per_layer_shapes_read_by_llama_cpp;
    for arch in ["deci", "openelm"] {
        assert!(is_audited_generic(arch), "{arch}");
        assert!(unaudited_triage(arch).is_none(), "{arch}");
        assert!(per_layer_shapes_read_by_llama_cpp(arch), "{arch}");
    }
    // `laguna` was the third row here and closed the next day on the
    // gated attention; `step35` the fourth, on the per-layer activation
    // seam. Both read per-layer heads and are served.
    for arch in ["laguna", "step35"] {
        assert!(
            is_audited_generic(arch) && per_layer_shapes_read_by_llama_cpp(arch),
            "{arch}"
        );
    }
    // `mimo2` was the one row in the reach table still refusing, and
    // closed on the split K/V head width (`frink_models::kv_head_dims`);
    // its fixture carries the converter's `head_count_kv` array.
    assert!(
        is_audited_generic("mimo2")
            && unaudited_triage("mimo2").is_none()
            && per_layer_shapes_read_by_llama_cpp("mimo2")
    );
    // `nanbeige` reads the arrays too and replicates them per pass;
    // it closed on `frink_models::layer_loops`, whose `LayerShapes::
    // replicated` is the copy `nanbeige.cpp:24-26` makes.
    assert!(
        is_audited_generic("nanbeige")
            && unaudited_triage("nanbeige").is_none()
            && per_layer_shapes_read_by_llama_cpp("nanbeige")
    );
    // And `granite` reads `n_head(il)` in its graph alone
    // (`granite.cpp:204`) while sizing tensors from layer 0, so it is
    // deliberately absent from the table.
    assert!(!per_layer_shapes_read_by_llama_cpp("granite"));
}

/// The three alias rows are refused as STRINGS NOBODY WRITES, not
/// triaged as architectures.
///
/// This closes the UNKNOWN their old verdict opened. That verdict asked
/// for "a real GGUF whose general.architecture is literally one of
/// these three", and the answer is that no such file can be produced:
///
///   * `mistral`, `mixtral` and `yi` are in neither `LLM_ARCH_NAMES`
///     (`src/llama-arch.cpp` carries `mistral3` and `mistral4` and
///     nothing else under that prefix) nor gguf-py's
///     `MODEL_ARCH_NAMES`.
///   * libllama REFUSES a file declaring any of the three:
///     `llama_model_load: error loading model: unknown model
///     architecture: 'mistral'` -- measured on a synthetic llama-shaped
///     file written under each string, which is also why no golden
///     reference for these rows can ever exist.
///   * The two real checkpoints in `models/` both declare `llama`; see
///     `real_mistral_and_yi_checkpoints_declare_llama` below.
///
/// Leaving them on the generic path was a live hazard, and the reason
/// this had to move rather than merely be re-worded: they carried NEOX
/// while `llama` -- the graph they claim to be -- is in
/// `llama_model_rope_type`'s NORM group, and
/// `rope_layout_matches_llama_cpp` cannot see it, because a name absent
/// from llama.cpp's table is a `continue` there.
#[test]
fn the_alias_rows_are_refused_as_strings_no_converter_writes() {
    for arch in ["mistral", "mixtral", "yi"] {
        assert!(
            unaudited_triage(arch).is_none(),
            "`{arch}` still carries a triage verdict; it is not an unaudited architecture, \
             it is a string no converter writes"
        );
        match frink_models::capability::resolve_architecture(arch) {
            Some(ArchPath::DedicatedOnly { reason }) => {
                // The refusal has to carry the actionable half -- "your
                // file is spelled `llama`" -- and the measurement that
                // decided it. A refusal that only says no sends the user
                // back to the same question.
                for claim in [
                    "LLM_ARCH_NAMES",
                    "unknown model architecture",
                    "general.architecture = llama",
                    "convert_hf_to_gguf.py",
                ] {
                    assert!(
                        reason.contains(claim),
                        "`{arch}`'s refusal drops {claim:?}: {reason}"
                    );
                }
            }
            other => panic!("`{arch}` must be refused as an alias, got {other:?}"),
        }
    }
    // And the string they redirect to must actually run, or the advice
    // is wrong.
    assert!(is_audited_generic("llama"));
}

/// The measurement behind the row above, re-runnable on this machine.
///
/// `#[ignore]`d because it needs the real checkpoints in `models/`.
/// Every Mistral, Mixtral and Yi GGUF converts to `llama`, and this is
/// what says so from FILES rather than from a reading of
/// `conversion/*.py`. If a checkpoint ever turns up declaring one of
/// the three, this fails and the alias rows need re-reading with that
/// file in hand -- which is exactly the evidence their old UNKNOWN
/// verdict asked for and nobody could supply.
///
///     FRINK_TEST_MODELS_DIR=$PWD/models \
///       cargo test -p frink-models --test unaudited_triage -- --ignored
///
/// The env var is not optional in practice: `cargo test` runs with the
/// PACKAGE directory as its cwd, so the bare `models` default resolves
/// under `crates/frink-models/` and finds nothing. Same convention as
/// `tests/chat_template_real_gguf.rs`. Run once on 2026-09-10 over the
/// development host's 20 checkpoints: none declares one of the three
/// strings, and both Mistral/Yi files declare `llama`.
#[test]
#[ignore = "needs the real GGUF checkpoints in models/"]
fn real_mistral_and_yi_checkpoints_declare_llama() {
    let root = std::env::var("FRINK_TEST_MODELS_DIR").unwrap_or_else(|_| "models".to_string());
    let mut files = Vec::new();
    collect_gguf(std::path::Path::new(&root), &mut files);
    assert!(!files.is_empty(), "no GGUFs under {root}");

    let mut checked = 0;
    for path in &files {
        let Ok(file) = frink_gguf::GgufFile::open(path.to_str().unwrap()) else {
            continue;
        };
        let arch = frink_gguf::TensorSource::metadata_str(&file, "general.architecture")
            .unwrap_or_default()
            .to_string();
        assert!(
            !["mistral", "mixtral", "yi"].contains(&arch.as_str()),
            "{}: declares `{arch}`, which no converter was thought to write -- re-read \
             the alias rows with this file in hand",
            path.display()
        );
        let name = frink_gguf::TensorSource::metadata_str(&file, "general.name")
            .unwrap_or_default()
            .to_lowercase();
        if name.contains("mistral") || name.contains("mixtral") || name.contains("yi-") {
            checked += 1;
            assert_eq!(
                arch,
                "llama",
                "{}: a Mistral/Mixtral/Yi checkpoint that is not `llama`",
                path.display()
            );
        }
    }
    assert!(
        checked > 0,
        "no Mistral/Mixtral/Yi checkpoint under {root}, so this proved nothing"
    );
}

/// `baichuan` is one architecture string covering two different models,
/// and admitting it admitted only ONE of them.
///
/// This test used to read the triage verdict, which said in words which
/// model the refusal was about. The verdict is gone -- `baichuan` is
/// audited now -- so the same fact has to be held somewhere, and it is
/// held in two places that must agree: `loader.rs` refuses
/// `block_count == 40` by name before the audited list is ever
/// consulted (`baichuan_13b_is_refused_because_it_uses_alibi_and_the_7b_is_not`),
/// and the fixture behind the admission has 32 layers, not 2, because
/// `src/models/baichuan.cpp:5-14` reads the variant off the layer count
/// and any other value gets NO RoPE
/// (`the_baichuan_fixture_has_the_32_layers_that_select_the_rotating_variant`).
///
/// What this asserts is the join: that the name really did move to the
/// audited side, so a reader who finds the 13B refusal knows it is not
/// the whole story.
#[test]
fn baichuan_is_audited_as_the_7b_and_carries_no_stale_verdict() {
    assert!(
        is_audited_generic("baichuan"),
        "baichuan-7B was admitted with a libllama-golden fixture"
    );
    assert!(
        unaudited_triage("baichuan").is_none(),
        "an audited row must carry no verdict; the refusal it used to describe now lives \
         in loader.rs's block_count == 40 check, which is about the 13B alone"
    );
}

/// Batch 4 and batch 5: the remaining long tail.
#[test]
fn batches_four_and_five_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        // Batch 4. `maincoder` and `bailingmoe` were here and are now
        // audited; see `tests/one_match_arm_graphs.rs`.
        // `arctic` was HERE, NEW CODE on its PARALLEL dense + MoE
        // layer, and is audited (tests/parallel_dense_ffn_graphs.rs):
        // `frink_models::parallel_dense_ffn` and
        // `RouterInput::NormedLayerInput`.
        // `mistral3` was here on "attention temperature tuning" and is
        // audited; see `tests/attn_temperature_graphs.rs`.
        // `nanbeige` was HERE, NEW CODE on RUNNING THE SAME PHYSICAL
        // LAYERS MORE THAN ONCE, and is audited now (`frink_models::
        // layer_loops`, tests/layer_loop_graphs.rs); its absence is
        // asserted by `nanbeige_is_audited_and_carries_no_stale_verdict`
        // below.
        // `mellum` was HERE, NEW CODE on "two per-layer RoPE
        // variants" and, second, the sliding-window ARRAY. The array is
        // `frink_models::swa_layers` and the row is audited on a
        // libllama-golden fixture whose array disagrees with the seeded
        // period (`tests/window_array_graphs.rs`); the RoPE half is a
        // refusal by name (`swa_geometry`) for a file with both a window
        // and a scaling, which every real Mellum2 is.
        // `talkie` was HERE, NEW CODE on NO norm weights (and three
        // more things), and is audited now (`frink_models::skip_stream`,
        // `NormOp::RmsNoParams`, `QkNormStyle::PerHeadScalar`,
        // tests/skip_stream_graphs.rs); its absence is asserted by
        // `talkie_is_audited_and_carries_no_stale_verdict` below.
        // `mimo2` was HERE. Its leading blocker WAS "attention sinks",
        // then "NEXTN blocks and a window array", then "a V head width
        // that differs from the K head width", and each landed in turn:
        // sinks by tensor presence, `frink_models::mtp_blocks`,
        // `frink_models::swa_layers`, and `frink_models::kv_head_dims`
        // (tests/split_kv_head_dim_graphs.rs). Its absence is asserted
        // by `mimo2_is_audited_and_carries_no_stale_verdict` below.
        // Batch 5. `plamo3` was here, FIXTURE-AWAY. Building its
        // fixture found the verdict was wrong by one tensor name -- it
        // is the only architecture upstream that spells its two
        // post-norms without a `.weight` suffix -- so the arm landed in
        // `loader.rs` and it is audited now
        // (`tests/fixture_away_graphs.rs`).
        // `afmoe` and `laguna` were HERE, NEW CODE on the gated
        // attention. `frink_models::attn_gate` implements it for all
        // three graphs that have it, and both rows are audited on
        // libllama-golden fixtures (`tests/gated_attention_graphs.rs`);
        // `the_gated_attention_seam_closed_two_rows_and_narrowed_the_third`
        // below pins that.
        // `apertus` was HERE, NEW CODE on xIELU's four per-layer
        // parameter arrays. `frink_models::act_layers` reads them as
        // `get_key_or_arr` does and `frink_moe::GluAct::Xielu` carries
        // one layer's four; the row is audited on libllama-golden
        // fixtures (`tests/per_layer_activation_graphs.rs`), one with
        // the arrays and one with the scalar spelling llama.cpp
        // broadcasts. Its verdict's third sentence -- QK-norm BIASES
        // frink's norms cannot take -- was wrong: `apertus.cpp:93,96`
        // pass NULL for them, measured byte-identical with and without.
        // `exaone-moe` was HERE, NEW CODE on "GLOBAL layers get no
        // RoPE". That is `exaone4.cpp:116` with `swa_type` pinned to
        // STANDARD -- one rule, not two -- and `frink_models::
        // rope_layers` implements it for both, so the row is audited
        // and carries no verdict (`tests/no_rope_layer_graphs.rs`).
        // `the_per_layer_rope_gate_is_no_longer_anybody_s_leading_blocker`
        // below pins that, and pins that the two rows which still
        // mention NoPE now say it is NOT their blocker.
        ("grovemoe", TriageClass::NewCode, "SECOND bank of experts"),
        // `hunyuan-dense` was HERE, ONE MATCH ARM on the NTK-alpha RoPE
        // base rescale. The arm landed (`rope_ntk_alpha`) and is
        // evidenced against libllama, so the row is audited and carries
        // no verdict --
        // `the_qk_norm_ordering_arm_is_no_longer_anybody_s_leading_blocker`
        // below is what pins that.
        // `step35` was HERE, NEW CODE on its per-layer SwiGLU clamp
        // arrays and its half-width rotary on the full layers. The
        // clamp is the second body on the same seam as `apertus`'s
        // xIELU (`frink_models::act_layers`, `GluAct::SwigluClamped`,
        // read by SITE), the rotary width is `ModelConfig::rope_dim_swa`
        // (`frink_models::swa_geometry`), and the row is audited on
        // three libllama-golden fixtures (`tests/clamped_swiglu_graphs.rs`):
        // clamped, unclamped, and with a NextN block.
        // `the_per_layer_activation_seam_closed_two_rows` below pins it.
    ];
    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// The QK-norm ordering arm was wanted by three architectures, and all
/// three run on it now.
///
/// The cross-row check was the argument for adding a shared flag rather
/// than special-casing one architecture, and this is what it bought:
/// `hunyuan-dense` needed only to be added to
/// `QK_NORM_AFTER_ROPE_ARCHITECTURES`, leaving one real arm (the
/// NTK-alpha base rescale) rather than two. The test now runs the other
/// way round -- no row may still refuse for an ordering that is
/// implemented, which is the `glm4moe` defect exactly: the reason shown
/// and the reason true being two different strings.
#[test]
fn the_qk_norm_ordering_arm_is_no_longer_anybody_s_leading_blocker() {
    for arch in ["hunyuan-moe", "maincoder", "hunyuan-dense"] {
        assert!(
            is_audited_generic(arch) && unaudited_triage(arch).is_none(),
            "`{arch}` got the ordering arm and evidence; it must not still be refused"
        );
    }
    for p in architecture_catalog() {
        let Some(t) = p.triage else { continue };
        assert!(
            !t.blocker.contains("QK norm AFTER") && !t.blocker.contains("QK-norm AFTER"),
            "`{}` still leads with an ordering frink implements: {}",
            p.gguf_name,
            t.blocker
        );
    }
}

/// The per-layer RoPE gate was the LEADING blocker of one row
/// (`exaone-moe`) and a listed blocker of two more (`afmoe`,
/// `smallthinker`), and `frink_models::rope_layers` implements it for
/// all six architectures llama.cpp gates. So: `exaone-moe` is audited
/// and carries no verdict, and the two rows that still refuse for
/// other reasons must now say the gate is NOT what stops them -- a
/// verdict that keeps naming an implemented feature as a blocker is
/// the shape this suite exists to catch.
///
/// (`exaone-moe`'s old verdict also recorded that its hardcoded
/// `n_swa = 128` was checked and CLEAN -- `exaone-moe.cpp:13` reads the
/// window as a REQUIRED key -- and that finding now lives in the fixture
/// itself, which declares a window narrower than the prompt and matches
/// llama.cpp on it.)
#[test]
fn the_per_layer_rope_gate_is_no_longer_anybody_s_leading_blocker() {
    assert!(
        is_audited_generic("exaone-moe"),
        "exaone-moe closed on the per-layer RoPE gate"
    );
    assert!(unaudited_triage("exaone-moe").is_none());
    // `afmoe` was the second row here and closed on the gated attention
    // the day after; its fixture's layer 3 is the unrotated one.
    assert!(is_audited_generic("afmoe") && unaudited_triage("afmoe").is_none());
    // `smallthinker` was the third row here, said the gate was NO
    // LONGER its blocker, and closed on its router operand the day
    // after (tests/router_input_graphs.rs); its fixtures' layers 0 and
    // 4 are the unrotated ones, the OTHER phase from smollm3's.
    assert!(is_audited_generic("smallthinker") && unaudited_triage("smallthinker").is_none());
    // And the census in `rope_layers` names both, so the verdict text
    // and the table cannot drift apart about which rows carry the gate.
    for arch in ["afmoe", "smallthinker", "exaone-moe"] {
        assert!(
            frink_models::rope_layers::PER_LAYER_ROPE_GATES
                .iter()
                .any(|(name, _)| *name == arch),
            "`{arch}` is not in PER_LAYER_ROPE_GATES"
        );
    }
}

/// The gated attention was the LEADING blocker of two rows (`afmoe`,
/// `laguna`) and a listed blocker of a third (`step35`), and
/// `frink_models::attn_gate` implements it for all three graphs that
/// have it. So: all three are audited and carry no verdict -- `step35`
/// said the gate was not what stopped it for one day and then closed
/// on what did. `mimo2`'s leading blocker moved the same way: the
/// sinks are a tensor-presence fact now and its verdict leads with what
/// every real export actually carries.
#[test]
fn the_gated_attention_seam_closed_two_rows_and_narrowed_the_third() {
    use frink_models::attn_gate::{attn_gate_spec, GatePresence, ATTN_GATE_ARCHS};
    for arch in ["afmoe", "laguna", "step35"] {
        assert!(
            is_audited_generic(arch) && unaudited_triage(arch).is_none(),
            "`{arch}` closed and must carry no verdict"
        );
    }
    assert_eq!(
        attn_gate_spec("step35").map(|s| s.presence),
        Some(GatePresence::Optional),
        "step35.cpp:96 creates the gate TENSOR_NOT_REQUIRED"
    );
    // The census names the three, so the verdicts and the table cannot
    // drift apart about which rows carry the op.
    for arch in ["afmoe", "laguna", "step35"] {
        assert!(
            ATTN_GATE_ARCHS.iter().any(|(n, _)| *n == arch),
            "`{arch}` is not in ATTN_GATE_ARCHS"
        );
    }
    // `mimo2`'s sinks are a tensor-presence fact and the row closed on
    // its split K/V head width; its fixture carries sinks on every layer.
    assert!(is_audited_generic("mimo2") && unaudited_triage("mimo2").is_none());
}

/// `mimo2`'s verdict named a split K/V head width and a value scale and
/// said four earlier blockers were done; the two landed
/// (`frink_models::kv_head_dims`, `frink_models::attn_value_scale`),
/// the four were, and the row is audited on three libllama-golden
/// fixtures (tests/split_kv_head_dim_graphs.rs), so it carries no
/// verdict. Building it found `expert_weights_scale` honoured for every
/// architecture where llama.cpp reads it in twenty loaders and nowhere
/// else; the loader's `EXPERT_WEIGHTS_SCALE_READERS` is the measurement.
#[test]
fn mimo2_is_audited_and_carries_no_stale_verdict() {
    assert!(is_audited_generic("mimo2"));
    assert!(unaudited_triage("mimo2").is_none());
    assert!(frink_models::kv_head_dims::admits_split_kv_head_dims(
        "mimo2"
    ));
    assert!(frink_models::kv_head_dims::SPLIT_KV_HEAD_DIM_ARCHS
        .iter()
        .all(|(name, _)| is_audited_generic(name)));
    assert!(frink_models::attn_value_scale::VALUE_SCALE_READERS
        .iter()
        .all(|(name, _)| is_audited_generic(name)));
}

/// Every one of the 47 now carries a verdict, and the four classes are
/// all represented.
///
/// The distribution is the headline: `NewCode` dominates. That is the
/// honest answer to "how far is frink from llama.cpp on models", and it
/// is the number this whole item existed to produce.
#[test]
fn every_unaudited_row_is_triaged_and_the_distribution_is_pinned() {
    let mut fixture = 0;
    let mut arm = 0;
    let mut new_code = 0;
    let mut unknown = 0;
    for p in architecture_catalog() {
        if !matches!(p.path, ArchPath::GenericGqa { .. }) || is_audited_generic(p.gguf_name) {
            continue;
        }
        match p.triage.expect("every unaudited row is triaged").class {
            TriageClass::FixtureAway => fixture += 1,
            TriageClass::OneMatchArm => arm += 1,
            TriageClass::NewCode => new_code += 1,
            TriageClass::Unknown => unknown += 1,
        }
    }
    assert_eq!(
        (fixture, arm, new_code, unknown),
        (0, 0, 3, 1),
        "the triage distribution moved; if a verdict changed on evidence that is correct, \
         update this and docs/MODELS.md together. BOTH cheap classes were ZERO between \
         2026-09-12 and 2026-09-19 -- `gemma` was the last FIXTURE-AWAY row and `chatglm` \
         the last ONE MATCH ARM one -- and the 2026-09-19 pin move refilled the ONE MATCH \
         ARM column with two: `maple` needs one `frink_models::rope_layers` row \
         (`RopeLayers::SlidingOnly`, `src/models/maple.cpp:88`) and `spark2_5` needed one \
         `frink_models::attn_gate` row (sigmoid, per head, `src/models/spark2-5.cpp:41`), \
         which it got the same day, and `maple` closed with it -- so the column is EMPTY \
         again, and the two rows cost a fixture and an hour each. `maple` also found the \
         thing a verdict written from one file cannot: `llama-graph.cpp:2228` sends it \
         to `ggml_swiglu_clamp`, the gate clamped BEFORE the SiLU, where `step35` takes \
         the `else` branch that clamps the SiLU's output: the lesson \
         of `minimax-m2` is that a row whose verdict names its own closing evidence and \
         does not go and get it is a refusal that could have been a row. The NEW CODE \
         column went 1 to 7 the same day (`granite_swa`, `graniteswitch`, `muse-glimmer`, \
         `hrm_text`, `minimax-01`, `qwen4exp`, plus `grovemoe`). It is back to 3 (`graniteswitch`, `qwen4exp`, \
         `grovemoe`): `granite_swa` and `muse-glimmer` closed on norm facts nothing else \
         upstream has, `hrm_text` on its two-stack schedule, and `minimax-01` on \
         `frink_models::lightning` plus `frink_models::normed_residual`, where the BLOCK \
         was the cheap half because the seam that selects it already existed. \
         That column went 26 to 24 when `olmo2` and `exaone4` closed together -- \
         one topology, one implementation -- 24 to 21 when `granite`, `granitemoe` \
         and the `granite-moe` alias closed on ONE implementation of their four scalar \
         multipliers, 21 to 20 when `olmo` closed on the non-parametric LayerNorm, and \
         20 to 19 when `exaone-moe` closed on the per-layer RoPE gate \
         (`frink_models::rope_layers`) -- one cause behind three refusals, but only \
         one of the three was in this column (EXAONE-4 32B was refused by name and \
         `smollm3` was DedicatedOnly), which is why it reads like `olmo` and is not, and \
         19 to 17 when `grok` and `dbrx` closed on seams landed the day before: the \
         defaults hook and the norm-site table for `grok`, the weighted LayerNorm, the \
         QKV clamp and the same table for `dbrx` -- and the clamp closed `olmo`'s \
         clip_qkv refusal by name with it, which again moved the audited number and not \
         this one, and 17 to 14 when `arcee` closed on the ungated ReLU-squared FFN and \
         `deci` and `openelm` closed together on the per-layer shape seam \
         (`frink_models::layer_shapes`) -- one cause behind five refusals, of which two \
         were closable by it alone and three (`laguna`, `mimo2`, `step35`) say so and \
         name what else they need. `arcee` closed ALONE for the opposite reason to \
         `olmo`'s: its cause IS shared (five graphs pass LLM_FFN_RELU_SQR) but the constant \
         that shared it with `plm` had missed `plm`'s MLA attention, and 14 to 12 when \
         `afmoe` and `laguna` closed together on the gated attention \
         (`frink_models::attn_gate`) -- one op with two free parameters behind three \
         verdicts, read side by side first; `step35` keeps the other two things its \
         verdict names, and `mimo2`'s sinks moved off the gpt-oss name onto the tensor \
         without closing it, because every real export carries NEXTN blocks and a \
         per-layer window array, and 12 to 11 when `mellum` closed on the per-layer \
         sliding-window ARRAY (`frink_models::swa_layers`) -- the cause those two \
         verdicts named, and `mellum` is the one generic-path graph that honours the \
         array; `frink_models::mtp_blocks` landed beside it and skips the NextN blocks \
         both named, so `mimo2` now leads with its split K/V head width and `step35` \
         with its clamp arrays and half-width rotary, and 11 to 9 when `apertus` and \
         `step35` closed together on the per-layer ACTIVATION PARAMETER seam \
         (`frink_models::act_layers`) -- one plumbing question behind two verdicts and \
         two activation bodies, and the clamp's routed-versus-dense SITE the one thing the \
         second needed that the first did not; `step35`'s half-width rotary landed on \
         `frink_models::swa_geometry` as the two-valued width `n_rot(il)` already was \
         upstream, which lifted Laguna-XS.2's second-rotary-width refusal by name, and \
         9 to 8 when `mistral3` closed on the per-position attention temperature \
         (`frink_models::attn_temperature`) -- one cause behind three graphs, measured \
         first, and only this one on the generic path: `llama4` (literals, its own \
         engine) and `deepseek2` / `mistral4` (the same key, the MLA engine, which \
         refuses it by name now) say so. Its verdict's other half, \
         `yarn_log_multiplier`, turned out to adjust a YaRN magnitude term frink did \
         not apply for ANY architecture (`frink_models::yarn_magnitude`), and 8 to 7 \
         when `smallthinker` closed on the router operand \
         (`frink_models::router_input`) -- the reach measured first over every \
         `build_moe_ffn` call site: four graphs pass a precomputed `probs_in`, \
         `grovemoe` among them, and `grovemoe` shares the MECHANISM and not the cause \
         (it routes on the normed FFN input; its blocker is a second expert bank), so \
         this closed alone and the column says why. Its gated ReLU experts found the \
         one `GluAct` variant that had served `arcee` by aliasing answering \
         `relu(up)^2` for a REAL gate; it is two variants now. And 7 to 6 when `bitnet` \
         closed on the two norms INSIDE the blocks (`frink_models::sub_norms`): the \
         reach came back with one graph of 155, so the fact is a `bool` read by the \
         loader and by the Metal predicate, and the arithmetic landed in the one \
         attention tail and the one dense FFN row body that already existed. And 6 to \
         5 when `mimo2` closed on the split K/V head width (`frink_models::kv_head_dims`), \
         whose reach is one generic-path converter and the MLA engine, which has carried \
         the pair since it existed; the seam is one `Option<usize>` on `ModelConfig` and \
         a V width beside every K width in the cache, the kernels and the checks. And 5 \
         to 4 when `nanbeige` closed on the layer loop (`frink_models::layer_loops`): a \
         logical-to-physical mapping and a loop norm at the end of both FFN bodies. And \
         4 to 3 when `talkie` closed on four seams at once, each one graph of 155: a \
         weightless RMSNorm variant, a per-head scalar QK gain, the embedding skip stream \
         and the two projection gains. And 3 to 2 when `plm` moved to the MLA engine \
         (`frink_models::mla_arch`): its attention had been there since the engine \
         existed, and what the row needed was the direct Q form, the ungated dense FFN \
         and the tied lm_head as one table -- plus the engine's first libllama golden, \
         which is the evidence every other row in this file was held to. And 2 to 1 when \
         `arctic` closed on `frink_models::parallel_dense_ffn` (the dense FFN summed with \
         the experts: the shared-expert slot under the dense names plus the row's scale, \
         two graphs of 155, Grok-2's refusal by name lifted with it) and \
         `RouterInput::NormedLayerInput` (the routed branch reading the layer input under \
         a second norm, one graph of 155). What is left is `grovemoe`, whose upstream graph \
         diverges from its reference, and `phi4`. \
         The first two closures took several rows at once because each found ONE cause \
         behind several refusals; `olmo` is the first that did not, and the reason is \
         recorded rather than hoped over -- every `build_norm` call in llama.cpp's 140 \
         graphs was scanned for a null weight and all three hits are `olmo.cpp`, so \
         there was no second row to take. The \
         single UNKNOWN left is `phi4`; `mistral`, `mixtral` and `yi` were the other \
         three and turned out not to be architectures at all"
    );
    assert_eq!(fixture + arm + new_code + unknown, 4);
}

/// The per-layer activation-parameter seam closed two rows whose
/// verdicts named different activations -- xIELU and a clamped SwiGLU
/// -- because the thing missing was the same plumbing: a layer's FFN
/// activation carrying scalars read from the GGUF. Both are audited,
/// neither carries a verdict, and the two tables that decide which
/// architecture reads which keys are the measured lists.
#[test]
fn the_per_layer_activation_seam_closed_two_rows() {
    use frink_models::act_layers::{reads_swiglu_clamps, uses_xielu};
    for arch in ["apertus", "step35"] {
        assert!(
            is_audited_generic(arch) && unaudited_triage(arch).is_none(),
            "`{arch}` closed on the per-layer activation seam and must carry no verdict"
        );
    }
    assert!(uses_xielu("apertus") && !reads_swiglu_clamps("apertus"));
    assert!(reads_swiglu_clamps("step35") && !uses_xielu("step35"));
    // No row still refusing names either activation as its blocker.
    for p in architecture_catalog() {
        let Some(t) = p.triage else { continue };
        assert!(
            !t.blocker.contains("xIELU") && !t.blocker.contains("swiglu_clamp"),
            "`{}` names an implemented activation as a blocker: {}",
            p.gguf_name,
            t.blocker
        );
    }
}
