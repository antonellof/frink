//! The per-architecture tables the config resolution reads: which
//! graphs default to sigmoid gating or pass a gating literal, which
//! read the two `expert_weights_*` keys, which do not renormalise
//! top-k, and the smaller one-row facts beside them. Each row carries
//! the llama.cpp line that decides it.

use frink_moe::GatingFunction;

/// Architecture-family name strings (GGUF's `general.architecture` value)
/// known, from reading ik_llama.cpp's `llama-hparams.cpp`
/// (`LLM_ARCH_DEEPSEEK2`, `LLM_ARCH_GLM4_MOE` cases), to default to
/// sigmoid MoE gating with post-selection renormalization rather than
/// softmax. Every member's citation is inline here; `docs/MODELS.md`
/// carries none and the pointer that used to send readers there was
/// dangling.
/// `afmoe`, `laguna` and `step35` added 2026-09-01 by the
/// unaudited-refusal triage's gating sweep. Each reads
/// `LLM_KV_EXPERT_GATING_FUNC` as OPTIONAL and then, when the key is
/// absent, sets `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`
/// (`afmoe.cpp:29-30`, `laguna.cpp:55-56`, `step35.cpp:19-20`). Frink
/// fell back to softmax for all three.
///
/// This is the `deepseek` shape a third, fourth and fifth time: a
/// default that is right for most architectures and silently wrong for
/// one, where the GGUF carries no key to correct it. Nothing is live
/// today -- all three are `NewCode` for other reasons and refuse before
/// reaching here -- but the list is what a later admission would trust.
pub(crate) const SIGMOID_GATING_ARCHITECTURES: &[&str] = &[
    "afmoe",
    // cohere2moe.cpp:27-29: the key read optional, SIGMOID when absent.
    "cohere2moe",
    "deepseek2",
    "glm4moe",
    "laguna",
    "step35",
];

/// Architectures whose graph passes a gating LITERAL into
/// `build_moe_ffn`, so the file's `expert_gating_func` is never read:
/// the literal wins even over a key that says otherwise.
///
/// Measured 2026-09-12 by parsing every `build_moe_ffn(` call's
/// arguments in all 155 `src/models/*.cpp`: three graphs pass
/// `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` (`llama4.cpp`, `mimo2.cpp:227`,
/// `nemotron-h.cpp`), twenty-six pass `_SOFTMAX`, nineteen pass
/// `hparams.expert_gating_func`. Only `mimo2` of the three is on this
/// loader. The twenty-six softmax literals are not tabled: every
/// converter for them writes no key or writes SOFTMAX, so the key and
/// the literal agree on every real file, and a table of twenty-six
/// hand-copied rows would be a bigger risk than the hand-written file
/// it guards against. `conversion/mimo.py` writes SIGMOID from
/// `scoring_func`, so on a real MiMo file the two agree too; the row
/// exists because the literal is what llama.cpp runs.
///
/// Re-measured 2026-10-28 at the `604cc430` pin, counting call SITES
/// (MTP graphs included, so the numbers are not the ones above): 80
/// sites where `5b59b83` had 69 -- 4 SIGMOID literals at both pins, 33
/// SOFTMAX at both, the hparam 32 -> 43. Every new site is a new graph
/// except `mellum`, which moved from the SOFTMAX literal to the hparam
/// seeded SOFTMAX (`mellum.cpp:9,11,206`): on a file without the key
/// the answer is unchanged, and on one with it this loader already
/// read the key. No row is added or removed.
pub(crate) const GATING_LITERAL_ARCHITECTURES: &[(&str, GatingFunction)] = &[
    ("mimo2", GatingFunction::Sigmoid),
    // `nemotron-h.cpp:218`: the SIGMOID literal; the converter writes no
    // `expert_gating_func` (`conversion/nemotron.py:238-250`).
    ("nemotron_h_moe", GatingFunction::Sigmoid),
    // `llama4.cpp:230`: the SIGMOID literal, with `norm_w = false` at
    // `:228` (`NO_TOPK_RENORMALIZE_ARCHITECTURES`); the converter
    // writes no key (`conversion/llama.py:374-394`).
    ("llama4", GatingFunction::Sigmoid),
];

/// The names alone, for the cross-table test.
#[cfg(test)]
pub(super) const GATING_LITERAL_NAMES: &[&str] = &["mimo2", "nemotron_h_moe", "llama4"];

/// Architectures whose `load_arch_hparams` reads
/// `{arch}.expert_weights_scale` (`LLM_KV_EXPERT_WEIGHTS_SCALE`) -- and
/// so the only ones whose `build_moe_ffn` call sees a nonzero
/// `hparams.expert_weights_scale`. On every other architecture the key
/// is dead metadata upstream: the field stays 0 and the multiply is
/// skipped, whatever the file says.
///
/// Measured 2026-09-12: `grep -l LLM_KV_EXPERT_WEIGHTS_SCALE
/// src/models/*.cpp` is twenty graphs; these are the eight on this
/// loader (`deepseek2` / `deepseek32` / `deepseek2ocr` / `deepseek4` /
/// `glm-dsa` / `glm4-moe` / `kimi-linear` / `minimax-m3` / `dflash` /
/// `nemotron-h` / `hy-v3` are on other engines, refused, or unknown
/// here; `cohere2moe.cpp:20` joined on 2026-09-14). Found by `mimo2`'s fixture: `mimo2.cpp` reads the
/// key nowhere, libllama ran the fixture unscaled, and frink -- which
/// honoured the key for any architecture -- scaled it by 2.5.
///
/// Re-measured 2026-10-28 at the `604cc430` pin: the same grep is 29
/// files (24 at `5b59b83` with this spelling of the command), and the
/// five it gained are all NEW graphs (`beam`, `flint`, `glm5-next`,
/// `k2-horizon`, `muse-spark`), every one refused, so no row is added.
pub(crate) const EXPERT_WEIGHTS_SCALE_READERS: &[&str] = &[
    "afmoe",
    "bailingmoe",
    "bailingmoe2",
    // `cohere2moe.cpp:19-20` read both the norm and the scale.
    "cohere2moe",
    "deepseek",
    "dots1",
    "exaone-moe",
    // `glm4-moe.cpp:13-14` read both the scale and the norm.
    "glm4moe",
    "laguna",
    // `nemotron-h.cpp:19` (`routed_scaling_factor`, 2.5 on Nemotron-3 Nano).
    "nemotron_h_moe",
    "step35",
];

/// The same for `{arch}.expert_weights_norm` (`LLM_KV_EXPERT_WEIGHTS_NORM`):
/// eighteen graphs read it upstream, seven on this loader; every other
/// graph passes `norm_w` as a LITERAL into `build_moe_ffn`, and the
/// literal is what `NO_TOPK_RENORMALIZE_ARCHITECTURES` and its default
/// transcribe. `deepseek` reads the scale but not the norm
/// (`deepseek.cpp` passes `false`).
///
/// Re-measured 2026-10-28 at the `604cc430` pin: `grep -l
/// LLM_KV_EXPERT_WEIGHTS_NORM src/models/*.cpp` is 29 files where it was
/// 22, and ONE of the seven it gained is a graph frink SERVES: `mellum`
/// (`mellum.cpp:8-11`) now seeds `expert_weights_norm = true`, reads the
/// key, and passes the hparam (`:204`) where `5b59b83`'s `:180` passed
/// the literal `true`
/// (it reads `expert_gating_func` the same way, seeded SOFTMAX, which
/// is what this loader already did for it). Without the row a Mellum
/// file declaring `false` would have been renormalised here and not
/// upstream. The other six are new graphs, all refused.
pub(crate) const EXPERT_WEIGHTS_NORM_READERS: &[&str] = &[
    "afmoe",
    "bailingmoe",
    "bailingmoe2",
    "cohere2moe",
    "dots1",
    "exaone-moe",
    "glm4moe",
    "laguna",
    // `mellum.cpp:8-11` since the `604cc430` pin (seeded `true`).
    "mellum",
    // `nemotron-h.cpp:18` (`norm_topk_prob`).
    "nemotron_h_moe",
    "step35",
];

/// Names that appear in a behaviour table above but are `DedicatedOnly`
/// or `Deferred`, together with the module that actually applies the
/// behaviour for them.
///
/// Two true things were in conflict here, and deleting either would
/// have lost one. `SIGMOID_GATING_ARCHITECTURES` records a fact about
/// llama.cpp (these architectures default to sigmoid when the GGUF
/// carries no `expert_gating_func`), and a test pins it as such. The
/// cross-table test records a different fact: an entry for an
/// architecture that never reaches THIS loader cannot fire, and a gate
/// that cannot fire is worse than no gate because it reads as coverage.
///
/// Both hold. `deepseek2` is genuinely sigmoid-gated and genuinely
/// never arrives here. So the resolution is not to drop a name
/// from either place, it is to say out loud who owns it instead, and to
/// make an unexplained dead entry still fail.
///
/// Adding a name here is a claim that the named module applies the
/// behaviour. It is checked no further than that, so it is the one line
/// in this file to be suspicious of.
/// Test-only: it asserts a relationship rather than driving one, and a
/// production reader would have to be told that.
#[cfg(test)]
pub(super) const DEDICATED_OWNS_ITS_BEHAVIOUR: &[(&str, &str)] = &[
    // `mla_gguf_loader` reads `expert_gating_func` and falls back to
    // Sigmoid itself, so deepseek2's gating is decided there.
    ("deepseek2", "mla_gguf_loader"),
    // `glm4moe` was here while it was refused; it is a generic-path
    // row now (2026-09-12) and the sigmoid default is live in THIS
    // loader.
];

/// Architecture-family names whose real reference implementation skips
/// renormalizing top-k softmax routing weights after selection (GGUF
/// carries no metadata key for this -- it's hardcoded per-architecture in
/// both the real HF `transformers` model code and llama.cpp's
/// `build_moe_ffn` call sites, not read from the file). Confirmed for
/// `olmoe` against `OlmoeTopKRouter.forward` in
/// `transformers/models/olmoe/modeling_olmoe.py` (`config.norm_topk_prob`
/// is `false` in the real published config.json) and llama.cpp's
/// `src/models/olmoe.cpp` (`build_moe_ffn(..., false, ...,
/// LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, ...)`). See
/// `MoeLayerConfig::norm_topk_prob`'s doc comment for why this matters:
/// getting it wrong silently produces wrong generation output even
/// though the file loads and shape-validates fine.
// Architectures whose reference graphs pass `norm_w=false` to
// `build_moe_ffn` (llama.cpp) / `norm_topk_prob=false` in HF config.
// Qwen2-MoE: `.scratch/llama.cpp/src/models/qwen2moe.cpp` -- Softmax +
// `false` for the norm_topk slot. Renormalizing top-k weights made
// Qwen1.5-MoE greedy decode emit garbage despite shared-expert load.
// `deepseek` (V1) added 2026-09-01 by the unaudited-refusal triage.
// `src/models/deepseek.cpp:145-155` passes `norm_w=false`, and
// `conversion/deepseek.py`'s `DeepseekModel` never writes
// `{arch}.expert_weights_norm` -- only `DeepseekV2Model` does -- so no
// real `deepseek` GGUF carries the key to override the default with.
// Frink therefore renormalised where llama.cpp does not. Same class of
// bug as the OLMoE one above, and latent only because `deepseek` is
// unaudited and refuses first.
// `jamba.cpp:164` passes `norm_w = false` and its converter writes no
// `expert_weights_norm` (`conversion/jamba.py:24-54`).
// `llama4.cpp:228` passes `false` beside its SIGMOID literal
// (`GATING_LITERAL_ARCHITECTURES`): the top-k sigmoid scores weight
// the experts unrenormalised.
pub(crate) const NO_TOPK_RENORMALIZE_ARCHITECTURES: &[&str] =
    &["deepseek", "jamba", "llama4", "olmoe", "qwen2moe"];

/// Architectures whose `{arch}.feed_forward_length` counts the gate and
/// the up projection TOGETHER, so each FFN matrix is half as wide as the
/// key says.
///
/// Qwen-1 (`QWenLMHeadModel`, GGUF string `qwen` -- not `qwen2` and not
/// `qwen3`) is the only one. HF's `QWenMLP` sets
/// `ff_dim_in = config.intermediate_size // 2` and builds `w1` and `w2`
/// at that width; `conversion/qwen.py`'s `QwenModel` inherits the base
/// `set_gguf_parameters`, which writes `intermediate_size` through
/// unchanged (`conversion/base.py:1206`); and `src/models/qwen.cpp:33-35`
/// therefore creates `ffn_gate`, `ffn_up` and `ffn_down` at `n_ff / 2`.
///
/// **This costs no logits and is still worth fixing.** frink loads the
/// dense FFN by tensor NAME and uses each matrix's own shape, so the
/// forward pass was always right; what was wrong was `expert_ffn_dim`,
/// which is what every memory estimate and `frink inspect-plan` row
/// prices the FFN from. That is this repo's dominant bug shape --
/// `ModelConfig` and the weights disagreeing about one number with
/// nothing comparing them -- so
/// `the_declared_ffn_width_matches_the_matrices_that_load`
/// (tests/one_match_arm_graphs.rs) now compares them.
pub(crate) const FFN_LENGTH_COUNTS_GATE_AND_UP: &[&str] = &["qwen"];

/// Architectures whose checkpoints carry `{arch}.leading_dense_block_count`
/// while their reference graph never branches on it: **every** layer is
/// MoE regardless of what the key says.
///
/// `bailingmoe` is the case this list exists for.
/// `src/models/bailingmoe.cpp:5` reads
/// `LLM_KV_LEADING_DENSE_BLOCK_COUNT` into `n_layer_dense_lead` and then
/// `load_arch_tensors` creates `ffn_gate_inp`, the expert tensors and
/// the shared-expert tensors unconditionally for every layer (:39-54 --
/// there is no `if (i < n_layer_dense_lead)` anywhere in the file) and
/// the graph has no dense branch either (:119-152). Meanwhile
/// `conversion/bailingmoe.py:27` writes `first_k_dense_replace` into the
/// key verbatim, so real Ling checkpoints DO carry a nonzero value.
///
/// Frink's `ModelConfig::layer_is_dense` does branch on it, so without
/// this list frink looks for `blk.0.ffn_gate.weight` on a layer that
/// only ships experts and dies on a missing tensor. That is a load
/// failure rather than wrong logits, which is why it stayed latent.
///
/// Do not read this as "the key is meaningless": for `deepseek`,
/// `dots1`, `glm4moe` and every other leading-dense architecture the key
/// is load-bearing and must be honoured. Membership here is a statement
/// about ONE architecture's graph, checked in that graph.
pub(crate) const LEADING_DENSE_KEY_IS_INERT: &[&str] = &["bailingmoe"];

/// Architectures whose reference graph applies `attn_q_norm` /
/// `attn_k_norm` AFTER `ggml_rope_ext`, not before it.
///
/// There is no GGUF key for this. llama.cpp writes the order into each
/// hand-written graph, so the only place it can come from is the
/// architecture string, and getting it wrong changes every layer's
/// attention scores without changing a single tensor shape.
///
/// - `maincoder`: `src/models/maincoder.cpp:78-90` ropes Q and K, then
///   norms them at `:92` and `:95`.
/// - `hunyuan-moe`: `src/models/hunyuan-moe.cpp:93,104` rope, `:110,115`
///   norm.
///
/// - `hunyuan-dense`: it has no graph of its own --
///   `src/models/models.h:1830-1834` derives `llama_model_hunyuan_dense`
///   from `llama_model_hunyuan_vl` and reuses its graph -- so the lines
///   are `hunyuan-vl.cpp:56-66` (rope) then `:73-81` (norm). Its second
///   blocker, the NTK-alpha RoPE base rescale, is implemented too; see
///   [`crate::rope_ntk_alpha`].
///
/// The audited majority is the other way round -- `qwen3moe.cpp:99,108`
/// and `bailingmoe2.cpp:123-135` both norm first -- which is why the
/// decoder's default is "before" and this list is the exception.
pub(super) const QK_NORM_AFTER_ROPE_ARCHITECTURES: &[&str] =
    &["hunyuan-dense", "hunyuan-moe", "maincoder", "talkie"];
