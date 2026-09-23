# llama.cpp vs frink: a differential inventory

**This is not a plan. It is an inventory**, produced 2026-09-01 by
reading both trees. Its job is to be the first systematic, evidence-
backed answer to the question
[`north-star.md`](north-star.md) asks: for any GGUF a user can run under
llama.cpp, what happens under frink?

**The standard every row here is held to.** A row exists only when the
claim was read on *both* sides, and both citations are given. Where a
claim could not be verified it is marked **UNKNOWN** with a note saying
what would settle it. That rule is not decoration. This repo has three
recorded cases of a count that was a claim rather than a reading: the
architecture catalog read as a support matrix, two loaders demanding
GGUF keys that do not exist while their own fixtures supplied the
invented keys, and a quant table accurate on the lines it flagged and
wrong by omission on a whole SIMD tier. A confident wrong count is worse
than no count.

**Read the body as a SNAPSHOT dated 2026-09-01, not as current
state.** Rows are deliberately left as the reading found them, because
an audit that is edited in place stops being evidence of anything. What
has moved since is listed here and nowhere else, so a reader who checks
this header cannot be misled by a row below it:

| Row | Then | Now |
|---|---|---|
| `logit_bias` (§3.1, §E5) | Silently dropped on chat, refused on completions | **Served on all three generation wires**, keyed by the response cache, with the device argmax fold refused for a biased request (0.47.0) |
| `echo`, `logprobs`, `prompt_logprobs` (§7, §E-rows) | Refused on `/v1/completions` | **Served**, `echo` with the logprobs arrays covering the echoed span (0.43.0, 0.45.0) |
| `n`, `best_of` | Not in the inventory | **Served on both OpenAI routes** from one prefill, on both KV stores, interleaved when streamed (0.40.0, 0.41.0) |
| `response_format: json_schema` | Converter in the tree, route answered 501 | **Served**, through the same converter a forced `tool_choice` uses |
| `--repeat-last-n` (§632) | MISSING | Ships, at llama.cpp's default |
| `-md` / `--model-draft`, `--draft-max`, `--draft-min` (§652) | MISSING on the run path | Ship |
| K-quant logit drift (§10) | The reason frink's logits differ from llama.cpp's on K-quants | Unchanged, and still not a frink bug: llama.cpp quantizes activations to `Q8_K` before the dot product and frink keeps them in f32 |

The last row is why this file is linked from `README.md` and is the
part worth reading first.

Line numbers are from the llama.cpp checkout in `.scratch/llama.cpp` and
from this repo as of 2026-09-01.

---

## 0. What changed since the 2026-08-27 audit

Two of the headline numbers repeated in `north-star.md`,
`docs/plans/README.md` and `CLAUDE.md` are **stale in frink's favour**,
and one is stale against it. Reading the code rather than the doc:

| Claim in the docs | What the code says | Evidence |
|---|---|---|
| "68 architectures share one generic GQA path" and "63 remain unaudited" (silently running) | The inversion **already landed**. An architecture on the generic path that is not in `AUDITED_GENERIC_GQA` returns `LoadError::UnauditedArchitecture` and does not run at all, unless `FRINK_ALLOW_UNAUDITED_ARCH=1` | `crates/frink-models/src/loader.rs:680-690`; the list is `crates/frink-models/src/capability.rs:99-119` |
| "of 150 catalog rows ... a handful load and are WRONG" | The five named strings (`gpt2`, `mpt`, `refact`, `bloom`, `jais`) are now `DedicatedOnly` refusals, pinned by a test that they can never be re-listed as audited | `crates/frink-models/src/capability.rs:323-379`, `:1291-1300` |
| "47 dedicated paths" | There are **five** engine kinds reachable from a load: generic `Decoder`, `Kimi`, `Glm52`, `Mla`, `Gemma4` | `crates/frink-models/src/engine_factory.rs:114-122`, `:47-71` |
| "~12 run with evidence" | **11** GGUF architecture strings are on the audited list; a 12th and 13th (`deepseek2`/`mistral4` via `MlaEngine`, `gemma4` via `Gemma4Engine`) load but carry no cross-engine evidence and say so | `capability.rs:99-119`; `docs/MODELS.md:80` |

So the honest current position is **better structured and no broader**
than the docs say: the silent-wrong class was converted into refusals,
but the set of architectures that actually produce tokens did not grow.
`docs/plans/README.md:26-34` and `north-star.md:65-95` should be re-read
against this section; **fixing them is out of scope for this document,
which edits no existing file.**

---

## 1. Architectures

### 1.1 The shape of the gap

| | llama.cpp | frink | Evidence |
|---|---|---|---|
| Per-architecture graph files | **140** | n/a (one shared decoder + 4 dedicated engines) | `ls .scratch/llama.cpp/src/models/*.cpp` = 140; `crates/frink-models/src/decoder.rs` is 6702 lines |
| Architecture strings in the registry | 140 + `(unknown)` sentinel | **150 rows, and every one of llama.cpp's 140 is present** | `.scratch/llama.cpp/src/llama-arch.cpp` `LLM_ARCH_NAMES`; `capability.rs:230-982` |
| Strings that reach an executing path | 140 | **11** audited generic + 4 dedicated engines | `capability.rs:99-119`, `engine_factory.rs:114-122` |
| Strings on frink's generic path but **not** audited (refuse today) | n/a | **47** | 20 at `capability.rs:242-265`, 31 at `:266-306`, 7 explicit pushes at `:504-647`, minus the 11 audited |

**Name coverage is complete and is not the gap.** A set difference of
llama.cpp's `LLM_ARCH_NAMES` against `architecture_catalog()` leaves
exactly one llama.cpp entry unmatched, and that is the `(unknown)`
sentinel, which frink does carry under its literal spelling
(`capability.rs:953`). Eight frink rows have no llama.cpp counterpart
and are inert aliases or fixtures: `mistral`, `mixtral`, `yi`, `yi-vl`,
`phi4`, `granite-moe`, `granite-hybrid`, `kimi_k3`, plus the three
`ferroxtest*` fixtures. No real GGUF carries those strings (Mistral,
Mixtral and Yi checkpoints all tag `llama` or `qwen2`), so they are
dead rows, not coverage.

The gap is therefore **not "frink does not know about these
architectures"**. It is "frink knows about them and refuses 129 of
them". Under the bar in `north-star.md:51-59` that is a gap, and a
refusal is a gap rather than a defect.

### 1.2 The four outcomes, counted

| Outcome | Count | How to reproduce the count |
|---|---|---|
| **A. Runs, with named evidence** | 11 | `AUDITED_GENERIC_GQA`, `capability.rs:99-119`: `llama`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `olmoe`, `gemma2`, `gemma3`, `phi3`, `gpt-oss`, `dots1` |
| **B. Loads, no cross-engine evidence** | 4 engine kinds | `Mla` (`deepseek2`, `deepseek32`, `mistral4`), `Glm52` (`glm-dsa`, `glm4`), `Kimi` (`kimi-linear`, `kimi_k3`), `Gemma4` (`gemma4`, `gemma4-assistant`) -- `engine_factory.rs:57-64`. `docs/MODELS.md:80` says of Kimi/GLM-5.2/DeepSeek-V4 "nothing has been run end to end on a real checkpoint" |
| **C. Refuses, naming exactly what is missing** | ~82 | The `DedicatedOnly` and `Deferred` reason strings, `capability.rs:323-967` |
| **D. Refuses generically as unaudited** | 47 | `loader.rs:680-690`. Honest, but the message names no feature, so a user cannot tell a one-line fix from a six-month one |
| **E. Loads and computes something else** | see §1.4 | Two found; both conditional |

**Both D and E moved the same day.** D is now **46**, and the refusal is
no longer generic: every row carries a triage verdict naming the missing
piece (§8 item 6, landed). The count dropped by one because the triage
itself found `minicpm3` was never a generic-GQA model at all. It
requires `q_lora_rank`/`kv_lora_rank` and the DeepSeek-2
`attn_q_a`/`attn_q_b`/`attn_kv_a_mqa`/`attn_kv_b` tensor set
(`src/models/minicpm3.cpp:5-6,41-46`), so no MiniCPM3 checkpoint could
ever have loaded on the generic path, and it now refuses by name. Both
E rows are fixed; see §7.

Category D is the one this document would most like to shrink, and
§1.3 is why shrinking it is not free.

### 1.3 Category D is not one bucket: some of the 47 are one fixture away and some are new code

The refusal message for all 47 is the same, which hides a real
difference. Read against llama.cpp's graph, the 47 split at least three
ways. Every row below was read in llama.cpp's own source.

| Arch | What llama.cpp's graph does that the generic decoder does not | llama.cpp file:line | frink side | Verdict |
|---|---|---|---|---|
| `grok` | **Hardcodes** `f_attn_logit_softcapping = 30.0` and `f_router_logit_softcapping = 30.0` before letting an optional key override, and requires `attn_out_norm` + `ffn_post_norm` | `src/models/grok.cpp:9-11,19-21,62,75-77` | `unsupported_feature_keys` fires only on a **present** metadata key (`capability.rs:1156-1177`), so a Grok GGUF omitting the keys would pass that gate. Router logit softcapping has no frink concept at all | **New code.** Same hardcoded-literal shape as the `bloom`/`refact` ALiBi class |
| `dbrx` | LayerNorm (`LLM_NORM`, mean-subtracting) not RMSNorm, plus `f_clamp_kqv` and a required `attn_out_norm` | `src/models/dbrx.cpp:5,34,71,111-112,142` | The generic decoder is RMSNorm-only; `capability.rs:396-399` states this in the bias-refusal comment | **New code**, and it belongs in the existing "required LayerNorm bias" refusal group rather than in D |
| `apertus` | xIELU activation with four **per-layer arrays** of parameters | `src/models/apertus.cpp:6-9,132-139` | `FfnActivation` has three variants: `Swiglu`, `SwigluFused`, `Gelu` (`crates/frink-models/src/config.rs:304-312`) | **New code** |
| `bitnet` | A required `attn_sub_norm` inside the attention block, and `ffn_sub_norm` inside the FFN | `src/models/bitnet.cpp:24,36,101-106,135-140` | **CLOSED 2026-09-12.** `frink_models::sub_norms` (one `bool` on `ModelConfig`, two tensors on the layer, applied in the one attention tail and `frink_moe::run_expert_sub_normed`); the per-projection `.scale` tensors llama.cpp multiplies in for every architecture are refused by name (`frink_models::weight_scales`). libllama-golden fixture, `tests/sub_norm_graphs.rs` | ~~New code~~ Two slots, one graph of 140; real BitNet GGUFs still need `TQ1_0` / `TQ2_0` kernels |
| `grovemoe` | A SECOND expert bank (`*_chexps`) fed and weighted from the first bank's routing, then scaled by `expert_group_scale` | `src/models/grovemoe.cpp:57-59,137-167`; `llama-graph.cpp:1997,2035-2039` | frink's MoE layer holds one bank. **And, read against `modeling_grove_moe.py` on 2026-09-12, llama.cpp's graph disagrees with the reference twice**: the chunk experts read the routed experts' OUTPUT (`:148-152`) where the reference reads the same input, and their weights are gathered at the CHUNK index (`llama-graph.cpp:2035-2039`) where the reference gathers at the original expert's. Neither was discussed in upstream PR #15510 | **New code**, and there is no single graph to match until upstream settles it; refused by name with the finding in the verdict |
| `step35` | Per-layer SwiGLU clamp arrays (`swiglu_clamp_exp`, `swiglu_clamp_shexp`) | `src/models/step35.cpp:28-29` | frink has a `swiglu_oai` clamp for gpt-oss only (`decoder.rs:400-403`) | **Probably parameterisable** from the gpt-oss clamp |
| `smallthinker` | `LLM_FFN_RELU` experts, not SiLU; and the router reads `inpL`, the raw layer input | `src/models/smallthinker.cpp:111,151-161,158` | **CLOSED 2026-09-12.** `FfnActivation::Reglu` (gated `relu(gate) * up`, distinct from `arcee`'s aliased `ReluSqr`) and `frink_models::router_input` (`RouterInput::RawLayerInput`, captured before attention in `Decoder::router_operand`); `n_swa` pinned to 4096 as `smallthinker.cpp:8` does (`capability::swa_window_override`). Three libllama-golden fixtures, `tests/router_input_graphs.rs` | ~~New code, tiny~~ The activation was the small half; the router operand was the row's real blocker and no seam had touched it |
| `olmo2`, `seed_oss`, `exaone4` | Required `attn_post_norm` and (olmo2, exaone4) `ffn_post_norm` applied after the branch, before the residual | `src/models/olmo2.cpp:47,52,161-163,178-180`; `seed-oss.cpp:37,114-116`; `exaone4.cpp:60,67,152-153,166-167` | frink HAS `post_attn_norm` / `post_ffn_norm` (they are two of the five features `north-star.md:9` records the paged path having lost, so they exist on the contiguous path) | **Likely a fixture away**, if the loader wires the slots for non-Gemma families. UNKNOWN until someone reads the wiring; `grep -n post_attn_norm crates/frink-models/src/loader.rs` settles it |
| `bailingmoe2`, `ernie4_5-moe` | ~~Sigmoid-routed~~ MoE with `ffn_exp_probs_b` router bias, shared expert, `expert_weights_norm` | `src/models/bailingmoe2.cpp:61,169-171`; `ernie4-5-moe.cpp:86-88` | frink implements exactly this shape and `dots1` pins it (`capability.rs:280-284`) | **CLOSED, and the sigmoid half was wrong.** `ernie4-5-moe.cpp:90` HARDCODES `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`; only `bailingmoe2` reads its gating function from the file (`bailingmoe2.cpp:11`). Both rows are audited now, and `routing_ernie_moe_through_sigmoid_instead_of_softmax_diverges_from_llama_cpp` is what stops the claim coming back |
| `arctic` | A dense `{n_embd, n_embd}` FFN summed with the routed experts on every layer; the routed branch (router and experts) reading `ffn_norm_exps(inpSA)`, the layer input under a second norm | `src/models/arctic.cpp:38-48,118-154` | **CLOSED 2026-09-12.** `frink_models::parallel_dense_ffn` (two rows: `grok` Grok-2 with `sqrt(2)/2`, `arctic` required with no scale; the shared-expert slot under the dense names), `RouterInput::NormedLayerInput` (one row). libllama goldens for `arctic`, `arctic_wscale` (the key llama.cpp ignores) and `grok_dense_ffn`, `tests/parallel_dense_ffn_graphs.rs` | ~~New code~~ Two seams, one with reach two |
| `plm` | DeepSeek-2 MLA attention on a dense model: direct `attn_q`, `attn_kv_a_mqa` / `attn_kv_a_norm` / `attn_kv_b`, ungated ReLU-squared FFN, tied lm_head | `src/models/plm.cpp:23-24,32-40,84-166,181-187`; `conversion/plm.py:14-19` | **CLOSED 2026-09-12** on the MLA engine: `frink_models::mla_arch` (the three-column table), `frink_models::mla_q_proj` (direct or low-rank Q; the lite DeepSeek-V2 rule with it), `MlaDenseFfn::act`. libllama golden, KL 1.87e-13, `tests/plm_graphs.rs` -- the MLA engine's first. The engine refuses `rope.scaling.type` by name now | ~~New code~~ Three columns on an engine that had the attention |
| `talkie` | No norm weights, a `{1, n_head}` Q gain after RoPE, an embedding skip stream, `attn_output.scale` / `ffn_down.scale`, `logit_scale` | `src/models/talkie.cpp:5,26,50-52,82-91,117,123-126,141`; `conversion/talkie.py:26-31` | **CLOSED 2026-09-12.** `NormOp::RmsNoParams`, `QkNormStyle::PerHeadScalar`, `frink_models::skip_stream`, and `frink_models::weight_scales` serving the two companions; `MultiplierSupport::TALKIE`. Two libllama-golden fixtures, `tests/skip_stream_graphs.rs` | ~~New code~~ Four seams, each one graph of 140 |
| `nanbeige` | Runs the same physical layers `num_loops` times, each logical layer with its own KV, `output_norm` between passes | `src/models/nanbeige.cpp:6-31,69-73,167-175` | **CLOSED 2026-09-12.** `frink_models::layer_loops`: `Decoder::layers` stays physical, `ModelConfig::n_layers` is logical, `Decoder::layer_for` is the one mapping, the loop norm sits at the end of both FFN bodies; the fused Metal launches refuse a looped model. Three libllama-golden fixtures, `tests/layer_loop_graphs.rs` | ~~New code~~ A mapping, not a copy |
| `mimo2` | A V head width that differs from the K head width; `attention.value_scale` | `src/models/mimo2.cpp:47-48,52,132-140,152-154,180-183`; `conversion/mimo.py:154,163-165` | **CLOSED 2026-09-12.** `frink_models::kv_head_dims` (`ModelConfig::v_head_dim`, `KvCache::new_split`, `causal_gqa_attention_row`, the split prefill kernel) and `frink_models::attn_value_scale`; every fused Metal launch, the CUDA resident hook, the slot file and the KV block file refuse a split model. Three libllama-golden fixtures, `tests/split_kv_head_dim_graphs.rs` | ~~New code~~ Found `expert_weights_scale` honoured for every architecture where llama.cpp reads it in twenty loaders (`EXPERT_WEIGHTS_SCALE_READERS`) |
| `minimax-m2` | Plain GQA, whole-vector QK norm, partial NEOX RoPE, sigmoid MoE with `exp_probs_b` | `src/models/minimax-m2.cpp:26,30-31,96-106,131-141` | Already stated in the refusal reason: "UNAUDITED, not unimplemented" | **A fixture away**; already tracked in roadmap `b2-close-the-68` |

The finding that matters more than any individual row: **the refusal
text does not distinguish these.** A user hitting
`UnauditedArchitecture` on `bailingmoe2` (which needs a fixture) and on
`apertus` (which needs an activation function) reads the identical
sentence. That is a cheap, high-leverage fix and it is **not** in
`roadmap.md`.

### 1.4 Silently wrong today (jumps the queue)

These are bugs, not gaps: a real GGUF loads and computes something
llama.cpp does not.

#### E1. `phi3` sliding window is applied where llama.cpp explicitly disables it

- **llama.cpp:** on reading a nonzero `phi3.attention.sliding_window`,
  it logs a warning and then **turns SWA off entirely**:
  `hparams.swa_type = LLAMA_SWA_TYPE_NONE; hparams.n_swa = 0;
  hparams.set_swa_pattern(1);`
  (`.scratch/llama.cpp/src/models/phi3.cpp:13-24`). No layer is
  windowed.
- **The key is really written.** The converter emits it
  unconditionally, using `0` only to mark Phi-4:
  `.scratch/llama.cpp/conversion/phi.py:167-171`. So a Phi-3 GGUF
  carries a nonzero value and a Phi-4 GGUF carries `0`.
- **frink:** `loader.rs:417-419` reads the key and keeps any value
  `> 0`. `default_swa_layout("phi3")` returns `None`
  (`capability.rs:1062-1119` -- deliberately, and the comment at
  `:1039-1041` explains that `set_swa_pattern(1)` means "no layer
  sliding"), and `phi3` is `PhiFamily` so the Gemma fallback of period
  6 does not apply either (`loader.rs:431-446`). That leaves
  `swa_pattern = None`, and
  `ModelConfig::layer_sliding_window`
  (`crates/frink-models/src/config.rs:359-377`) maps
  `None => true`: **every layer windows.**
- **Consequence:** a Phi-3-mini-4k GGUF (HF `sliding_window: 2047`)
  answers from a 2047-token history under frink and from the full
  history under llama.cpp. Phi-3-mini-128k and Phi-3.5-mini declare
  262144 and are inert in practice. **Phi-4-mini writes `0` and is
  filtered out by `loader.rs:419`, which is exactly why the suite's
  Phi-4-mini bench row never saw this.**
- **Secondary effect, same root:** because `sliding_window.is_some()`,
  `rope_theta_swa` is set, and `swa_rope_base_follows_model("phi3")` is
  false (`capability.rs:1134-1151`), so frink rotates every layer at
  theta `10000` rather than the model's own base
  (`loader.rs:480-494`). Inert for the Phi-3 line, whose base is
  `10000`. UNKNOWN for any `phi3`-tagged checkpoint with a different
  base; a `frink inspect` of one would settle it.
- **Severity:** high (fluent wrong answer past 2047 tokens on a
  currently-*audited* architecture). **Size: S.** Either add `"phi3"`
  to a "declared window means no window" list, or filter the key for
  `PhiFamily` at `loader.rs:417`.
- **Not in `roadmap.md`.** Item `b1-silently-wrong-today` is marked
  `done` and its point (5) covered the `default_swa_pattern` table;
  this is the adjacent case where the right answer is not a period at
  all.

#### E2. `unsupported_feature_keys` cannot see a hardcoded softcap

- **llama.cpp:** `grok` seeds `f_attn_logit_softcapping = 30.0f` and
  `f_router_logit_softcapping = 30.0f` and only then lets an *optional*
  key override (`src/models/grok.cpp:9-11,19-21`). `minicpm` does the
  same for three multipliers, which frink already caught and refused
  (`capability.rs:605-625`).
- **frink:** `unsupported_feature_keys` (`capability.rs:1156-1177`)
  builds a refusal only for keys that are **present** in the file.
- **Why it is not live today:** `grok` is on the generic path and
  unaudited, so it refuses first. This is latent, and it is a
  standing trap for whoever audits `grok`: the fixture will pass and
  the checkpoint will be wrong.
- **Severity:** medium (latent). **Size: S** -- the fix is a per-arch
  hardcoded-default table, the same mechanism `default_swa_layout` and
  `swa_rope_base_follows_model` already are.

### 1.5 Refused for the wrong reason

Auditing the reason strings against llama.cpp found **no live instance**
of the class that has bitten this repo four times (`glm4moe` demanding
`q_lora_rank`, `deepseek2` demanding invented keys, `minimax` blaming
MTP). Those four are fixed and the reasons now read correctly against
`glm4-moe.cpp:75,215`, `minimax-m2.cpp:26-141` and `minimax-m3.cpp:76-82`.

One **documentation** instance survived: `docs/MODELS.md:78` still said
MiniMax "will not load" for "MiniMax 256-expert sigmoid MoE + MTP",
which is the reason `capability.rs:663-723` explicitly retired as false
in both clauses. Doc drift, not a code defect. **Fixed**: that row is
now two rows, one per architecture, carrying the reasons the code
actually gives.

### 1.6 The single highest-value architecture item

**Make the unaudited refusal say which of the three kinds of missing it
is.** Today `LoadError::UnauditedArchitecture` (`loader.rs:689`) says
the same sentence for 47 architectures, of which -- per §1.3, read in
llama.cpp's own source -- some need a kilobyte fixture and a parity run,
some need one match arm, and some need a new activation function or a
new norm placement. That distinction is the entire content of "how far
is frink from llama.cpp on models", and it is currently invisible to
the person deciding what to work on and to the user deciding whether to
file an issue.

It beats the bigger candidates on the project's own ranking rule
(`roadmap.md:5-7`, ship-small-or-do-not-start): it is a data table plus
a wider error enum, it lands in one file, and after step one a user can
read a refusal and know whether their model is a week away or a quarter
away. The model-layer split (`roadmap.md:33-35`) and the audit outward
(`b2-close-the-68`) both get easier once the 47 are triaged, because
the triage *is* the work queue those items need.

---

## 2. Backends

The question is not "does a backend exist" -- frink has CUDA and Metal.
It is **which operations and which quant types each covers**, and where
a GPU path silently drops to the CPU.

### 2.1 Which backends exist at all

llama.cpp's `ggml/src/` holds 18 backend directories. frink has two.

| llama.cpp backend (size) | frink counterpart | Severity | Size |
|---|---|---|---|
| `ggml-cpu/` (66 files, 90,589 lines) | `frink-quant` + `frink-core` (`lib.rs` 8230, `repack.rs` 6446) | -- | -- |
| `ggml-cuda/` (274 files, 43,277 lines) | `frink-cuda/src/`, **5 files, 2,601 lines** (`gpu.rs:1794`, `attn.rs:395`, `graph.rs:159`) | **critical** | XL |
| `ggml-metal/` (12 files, 24,979 lines) | `frink-metal/src/`, 8 files, 19,664 lines | high | L |
| **`ggml-vulkan/`** (145 files, 34,174 lines) | **none** | **critical** | XL |
| `ggml-sycl/` (157 files) | none | medium | XL |
| `ggml-hip/` (CMake shim over ggml-cuda) | none | high | M, and it falls out of a real CUDA backend |
| `ggml-opencl/` (172 files -- Adreno / Mali phones) | none | medium | XL |
| `ggml-blas/` (529 lines -- Accelerate / MKL / OpenBLAS bridge) | none | medium | S |
| `ggml-rpc/` (2,772 lines) | none | low | M |
| `ggml-cann`, `ggml-webgpu`, `ggml-zdnn`, `ggml-musa`, `ggml-openvino`, `ggml-hexagon`, `ggml-et`, `ggml-virtgpu`, `ggml-zendnn` | none | low | XL each |

Vulkan is the one that decides the north-star's "hardware people
actually own" clause: it is the only backend covering AMD, Intel Arc and
Android GPUs from one codebase. This is already `roadmap.md:61-63`,
theme D.

### 2.2 Quant type matrix

llama.cpp `enum ggml_type`: `ggml/include/ggml.h:390-433` (43 tags, 5
removed upstream). frink `GgmlType`: `crates/frink-gguf/src/lib.rs:114-186`.

Evidence anchors. llama.cpp: CUDA dequant `ggml-cuda/convert.cu:460-508`,
CUDA mmvq `ggml-cuda/mmvq.cu:10-36`, CUDA MMQ `ggml-cuda/mmq.cuh:62-95`,
Metal kernel instantiations in `ggml-metal/ggml-metal.metal`. frink:
CPU dtype gate `crates/frink-core/src/weight_matrix.rs:472-495`, Metal
matvec table `weight_matrix.rs:398-408` +
`crates/frink-metal/src/gpu.rs:4812-4823`, Metal GEMM `gpu.rs:3035-3054`,
CUDA `weight_matrix.rs:500-506` + `crates/frink-cuda/src/gpu.rs:140,208,306,403,496`.

| Type | llama CPU / CUDA / Metal | frink CPU | frink CUDA | frink Metal (matvec / GEMM) |
|---|---|---|---|---|
| F32 | Y Y Y | Y | via f32 path | Y `gpu.rs:4814` / -- |
| F16 | Y Y Y | dequant only `quant/lib.rs:224` | N | **N** |
| BF16 | Y Y Y | dequant only `quant/lib.rs:206` | N | **N** |
| Q4_0 | Y Y Y | Y | Y `cuda/gpu.rs:208` | Y / Y |
| Q4_1 | Y Y Y | Y `quant/lib.rs:4373` | N | **N / N** |
| Q5_0 | Y Y Y | Y `quant/lib.rs:4450` | N | **N** / Y (see §2.5) |
| Q5_1 | Y Y Y | Y `quant/lib.rs:4518` | N | **N / N** |
| Q8_0 | Y Y Y | Y | Y `cuda/gpu.rs:140` | Y / Y |
| Q2_K | Y Y Y | Y `quant/lib.rs:4670` | N | **N / N** |
| Q3_K | Y Y Y | Y `quant/lib.rs:4805` | N | **N / N** |
| Q4_K | Y Y Y | Y | Y `cuda/gpu.rs:306` | Y / Y |
| Q5_K | Y Y Y | Y | Y `cuda/gpu.rs:403` | Y / Y |
| Q6_K | Y Y Y | Y | Y `cuda/gpu.rs:496` | Y / Y |
| IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | Y Y Y | Y (scalar or AVX2; no NEON) | N | **N / N** |
| IQ4_NL | Y Y Y (+`mul_mv_ext` r1_2..5) | Y `quant/lib.rs:4898` | N | **N / N** |
| IQ4_XS | Y Y Y | Y `quant/lib.rs:4962` | N | Y `gpu.rs:4820` / Y |
| **MXFP4** | Y Y Y (`mmq.cuh:74`, `mmvq.cu:19`, `mul_mm_mxfp4_f32`) | Y `quant/lib.rs:5081` | N | **N / N** -- this is why gpt-oss decodes on CPU |
| NVFP4 | Y Y N | **N**, sized and refused `gguf/lib.rs:178-182` | N | N |
| TQ1_0 / TQ2_0 | Y (`ggml-cpu/quants.c:107,113`) N N | **N**, sized and refused `gguf/lib.rs:163-176` | N | N |
| Q1_0 / Q2_0 | Y Y Y (`mmq.cuh:62`, `mul_mv_q1_0_f32`) | **N**, sized and refused `gguf/lib.rs:183-186` | N | N |

**Totals.** Quantized matvec types: llama.cpp CUDA **24**, Metal **22**,
CPU **26**. frink CPU **21**, Metal **6** plus F32, CUDA **5**.

CPU repacking: llama.cpp `ggml-cpu/repack.cpp` repacks Q4_0, Q4_K, Q5_K,
Q6_K, Q8_0, Q8_K, Q2_K, IQ4_NL, MXFP4. frink
`crates/frink-quant/src/repack.rs` repacks Q4_0 (`:2273`), Q8_0
(`:443`), Q4_K (`:90`), Q5_K (`:1123`), Q6_K (`:1815`) -- the five that
matter, missing IQ4_NL, MXFP4 and Q2_K. The `Q4_0_4_4` family is removed
upstream (`ggml.h:421-428`) and is not a gap on either side.

### 2.3 CPU SIMD tiers

| Tier | llama.cpp | frink | Severity | Size |
|---|---|---|---|---|
| x86 AVX2 | `ggml-cpu/arch/x86/quants.c` | Y, 9 `target_feature(enable="avx2")` functions | -- | -- |
| **x86 AVX512 / AVX512-VNNI / AVX-VNNI** | `ggml-cpu/arch/x86/cpu-feats.cpp` | **none** (zero `avx512`/`vnni` hits in `frink-quant`) | high | L |
| Intel AMX int8 | `ggml-cpu/amx/mmq.cpp` | none | medium | XL |
| ARM NEON | `ggml-cpu/arch/arm/quants.c` | Y, 26 `enable="neon"` | -- | -- |
| ARM dotprod + i8mm | Y | Y (`repack.rs`, 32 `dotprod`, 92 `i8mm`) | -- | -- |
| ARM SVE / SME (+ `kleidiai/`) | Y | none | medium | L |
| riscv-v, s390 VXE, wasm SIMD, POWER VSX, LoongArch LSX | `ggml-cpu/arch/*` | none | low | XL |

AVX512 is already `roadmap.md:65-67` (`avx512-int-dot`, theme D2), and
that item's own framing -- "can be WRITTEN and compile-checked here" -- is
still the right one.

### 2.4 CUDA, op family by op family

`ggml-cuda/` holds 65 `.cu` op families (23,261 lines of `.cu` plus
about 20k of `.cuh`). **frink-cuda has 8 kernels total**, all NVRTC-
compiled from Rust string literals: no `.cu` file, no PTX, no build
script.

| llama.cpp | frink-cuda | Severity | Size |
|---|---|---|---|
| **`mmq.cu` (384) + `mmq.cuh`** -- int8 tensor-core / `dp4a` quantized GEMM, 24 types, `ldmatrix`/`mma` intrinsics (`mmq.cuh:689,709`) | **nothing. No GEMM of any kind.** `crates/frink-core/src/weight_matrix.rs:2028-2054` loops `apply_gpu` once per position | **critical** | L for a naive `mul_mm`, XL for MMQ |
| `mmvq.cu` (1297) -- q8_1-quantized-activation matvec, 24 types | 5 float matvecs, f32 activations, no `dp4a` | critical | L |
| `mmvf.cu` (869), `mmf.cu` -- f16/bf16/f32 matvec and matmul | none; F16/BF16 are dequantized to F32 host tensors (`weight_matrix.rs:683`) | high | M |
| `mmid.cu`, `topk-moe.cu` (438) -- `MUL_MAT_ID` fused MoE + top-k | **none.** frink-metal has `moe_mm_id_*` and `moe_topk_softmax_batch`; CUDA has zero MoE kernels | high | L |
| `fattn.cu` (589) + `fattn-mma-f16.cuh` + `fattn-vec.cuh` + `fattn-tile.cu`; head dims 40/64/72/80/96/112/128/192/256/320/512/576 (`fattn.cu:122-182,394-430`); sinks (`fattn-common.cuh:26,986,1216`), mask, ALiBi, GQA, quantized KV | `crates/frink-cuda/src/attn.rs:23` `gqa_decode` -- one warp per head, sequential over `seq_len`, f32 KV, **decode only**, no prefill FA, no sinks, no mask, no ALiBi, hardcoded `rsqrtf(head_dim)` (`attn.rs:41`) | critical | L |
| `quantize.cu` (697) -- on-device activation quantization to q8_1 | none; activations upload as f32 | high | M |
| `rope.cu` (672) | **none.** RoPE is host-side on the CUDA path | high | M |
| `norm.cu` (698) -- RMS / L2 / group norm | `cuda/gpu.rs:998` `fused_add_rmsnorm_f32` only | medium | S |
| `softmax.cu` (480), `argsort.cu` (292), `top-k.cu`, `argmax.cu` | none | medium | M |
| `cpy.cu` (617), `set-rows.cu` (398), `getrows.cu` (490), `concat`, `pad`, `acc`, `set`, `roll` | none | medium | M |
| `ssm-scan.cu` (845), `ssm-conv.cu`, `gated_delta_net.cu`, `wkv.cu`, `gla.cu` -- Mamba / RWKV / Qwen3-Next | none | high (blocks a whole model class) | XL |
| `im2col.cu`, `conv2d*.cu`, `pool2d.cu` -- vision towers | none | medium | XL |
| `binbcast.cu` (574), `unary.cu` (646, 20+ unaries), `scale`, `clamp`, `fill`, `sum*`, `cumsum`, `mean`, `diag*`, `tri`, `solve_tri`, `out-prod`, `arange`, `tsembd`, `upscale`, `fwht`, `snake`, `softcap`, `count-equal` | `cuda/gpu.rs:942` `silu_mul_f32` only | medium | XL |
| `allreduce.cu` (971), peer access and NCCL (`ggml-cuda.cu:402,607-613`), `tensor_split` (`:302,386`) | **none.** Single device, no split modes, no P2P | medium | L |
| CUDA graphs -- full capture and `cudaGraphExecUpdate` (`ggml-cuda.cu:2595-2613,3883-4069`) | `crates/frink-cuda/src/graph.rs:1-11`, raw `cuGraph*` FFI behind `FRINK_CUDA_GRAPH=1`; the module's own doc says the hardware receipt is pending | medium | M |
| Unified memory `cudaMallocManaged` (`ggml-cuda.cu:141-142,4659`) | none | low | S |
| `ggml_backend_cuda_device_supports_op`, 179 `GGML_OP_` arms (`ggml-cuda.cu:4733`) | frink has no graph or op abstraction; the decode graph is hand-written in `crates/frink-models/src/decoder.rs` | architectural | -- |
| `lightning-indexer.cu` (588), `dsv4-hc.cu` (294) -- DeepSeek-V3.2/V4 | none | low | L |
| `opt-step-adamw.cu`, `cross-entropy-loss.cu` -- training | none, out of scope | -- | -- |

**Net: frink-cuda executes 8 kernels** -- matvec for 5 types, SwiGLU,
add+RMSNorm, and decode attention. Everything else on a CUDA host runs
on the CPU.

### 2.5 Metal

`ggml-metal/ggml-metal-ops.cpp:185-497` dispatches roughly 70
`GGML_OP_*`. frink-metal has 62 distinct `kernel void` bodies across
`gpu.rs`, `attn.rs`, `elem.rs` and `embd.rs`.

| llama.cpp Metal | frink-metal | Severity | Size |
|---|---|---|---|
| `kernel_mul_mv_*` for 22 types, plus `mul_mv_ext_*_r1_{2..5}` multi-row variants, plus `mul_mv_id_*` for 20 types | 7 matvec kernels (`f32`, `q8_0`, `q4_0`, `q4_k`, `q5_k`, `q6_k`, `iq4_xs`; `gpu.rs:4812-4823`); `mul_mv_id` for `q4_0`/`q4_k`/`q8_0` only (`gpu.rs:3042-3047`) | high | L |
| `kernel_mul_mm_*` simdgroup GEMM for 22 types, f16 and f32 src1 | 7 `*_mul_mm_sg` (`gpu.rs:3035-3041`) plus f16 twins (`:3048-3054`); 17 `simdgroup_*` uses against llama's 34 | medium | M |
| **`GGML_OP_FLASH_ATTN_EXT`** `ggml-metal.metal:7069`, `_pad:6258`, `_blk:6330`; **sinks** via function constant `:6386,6992-6997`; `max_bias`/ALiBi `:1994,2099`; arbitrary mask; DK/DV 40 to 576 | `gqa_decode_fa_vec{,_d64,_d96,_d256}` and `gqa_prefill_fa_vec*` -- head_dim in {64, 96, 128, 256} only (`attn.rs:2742-2751`, `:3754-3771`), causal + SWA (`attn.rs:1187`, `:6010-6012`) and softcap; **no sinks** (zero `sink` hits in `frink-metal`), **no ALiBi**, no arbitrary mask | **high** for sinks, medium for head dims | M |
| `GGML_OP_SSM_CONV` / `SSM_SCAN` (`ggml-metal-ops.cpp:337,341`), `RWKV_WKV6/7` (`:345`), `GATED_DELTA_NET` (`:350`) | none | high | XL |
| `IM2COL`, `CONV_2D`, `CONV_2D_DW`, `CONV_3D`, `CONV_TRANSPOSE_1D/2D`, `COL2IM_1D`, `POOL_1D/2D`, `UPSCALE` (`:396-478`) | none | medium | XL |
| `ARGSORT` (`:448`), `TOP_K` (`:452`), `ARGMAX` (`:482`) | `argmax_f32` (`elem.rs`) and `moe_topk_softmax_batch` (routing-specific) only; no general argsort or top-k | medium | M |
| standalone `SOFT_MAX` (`:333`), `GET_ROWS` for all 22 types (`:366`), `SET_ROWS` (`:370`), `CPY`/`DUP`/`CONT` (`:468-470`), plus `CONCAT`, `PAD`, `ROLL`, `ARANGE`, `TIMESTEP_EMBEDDING`, `CUMSUM`, `DIAG`, `TRI`, `SOLVE_TRI`, `L2_NORM`, `GROUP_NORM`, `ACC`, `SET`, `COUNT_EQUAL`, `SUM`/`SUM_ROWS`/`MEAN`, 22 unary ops, 6 GLU ops (`ggml-metal-device.cpp:239-319`) | `get_rows_q4_k` / `get_rows_q6_k` (`embd.rs`); `rms_norm*`, `add_rms_norm*`, `silu_mul`, `gelu_mul`, `vec_add`, `axpy`, `f32_to_f16` (`elem.rs`) | medium | XL |
| KV cache dtypes f32/f16/bf16/q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl | f32/f16/q8_0/q4 (`crates/frink-models/src/kv_budget.rs:100-108`) | low | S |

Where frink is genuinely competitive: the Metal MoE stack is real and
mirrors llama's `mul_mm_id_map0` design -- `moe_mm_id_map0_ne20_{2,4,6,8}`,
`moe_gather_rows`, `moe_scatter_rows`, `moe_weighted_sum_residual`,
`q4_0`/`q4_k`/`q8_0` `moe_matvec_id`, and `moe_ids.rs`.

### 2.6 Silent CPU fallback

frink is better instrumented here than expected, and the
instrumentation has holes.

| Site | Behaviour | Disclosed? |
|---|---|---|
| `crates/frink-core/src/kernel_registry.rs:687-700` `seal()` | Prints every accelerator lookup that will take a slow path; `FRINK_STRICT_KERNELS=1` refuses; `FRINK_KERNEL_REGISTRY=1` prints the whole table | **Yes. This is the right design and it should be the model for the rest** |
| `crates/frink-models/src/loader.rs:2096-2097`, `gemma4_gguf_loader.rs:319-320` | probe + seal | Yes |
| `glm52_gguf_loader.rs`, `hybrid_gguf_loader.rs`, `kimi_gguf_loader.rs`, `kimi_loader.rs`, `mla_gguf_loader.rs` | **zero `probe_kernels()` / `seal_or_error` hits** | **No. Five of seven loaders never seal, so every DeepSeek-MLA, GLM, Kimi and hybrid model runs with no backend disclosure at all.** Medium, S |
| `weight_matrix.rs:398-408` Metal matvec table returns `None` for 15 of the 21 CPU-supported kinds | runs on CPU | Registered as a `miss` (`weight_matrix.rs:2010-2024`), so disclosed only when the loader seals |
| `weight_matrix.rs:500-506` CUDA matvec false for 16 kinds | CPU | Same |
| `weight_matrix.rs:2028-2054` CUDA batch arm | per-position matvec loop, not a GEMM; a per-position `None` falls to CPU with **no `miss` recorded** | **Silent** |
| `crates/frink-models/src/decoder.rs:848-853` gpt-oss (attention sinks) | whole family on CPU | **Silent** -- a source comment only, not routed through `kernel_registry` |
| `decoder.rs:884-889` `attention_scale.is_some()` | CPU | Silent (a correct refusal, undisclosed) |
| `decoder.rs:893` `head_dim > 256` | CPU; excludes MLA (576/512) entirely | Silent |
| `decoder.rs:856` non-Norm/NeoX RoPE layout | CPU | Silent |
| `decoder.rs:3969` "longer prompts fall back to CPU attention" | CPU | Silent |
| `weight_matrix.rs:418-455` `metal_mul_mm_kind_supported` excludes Q5_0 | Its own comment (audited 2026-08-31) records that the exclusion is stale: `mul_mm_sg_launch` and the loader's `mapped_sg` list both carry Q5_0 and bypass this table, so **a Q5_0 checkpoint prefills on Metal and decodes on CPU today** | **Silent split.** Low, M |
| llama.cpp baseline | `src/llama-model.cpp:1632,1637` prints `offloading N repeating layers to GPU` / `offloaded N/M layers to GPU` unconditionally | frink `run.rs` prints no equivalent line; `active_backend()` is used only by `bench_model.rs:190` and `parity.rs:115` |

### 2.7 The single highest-value backend item

**A batched quantized GEMM on CUDA** -- `mul_mm`, plus on-device q8_1
activation quantization, plus a `dp4a`-based `mmvq` -- landed as the
first real op family in a `frink-cuda` whose kernels live in `.cu`
files rather than in Rust string literals.

frink-cuda today is 2,601 lines with 8 kernels and **no
matrix-matrix product of any kind**. The repo records the consequence in
its own words at `crates/frink-core/src/weight_matrix.rs:2028-2033`:
without the per-position arm, "measured on an RTX 4090, SmolLM2 `pp512`
ran at 28 tok/s against llama.cpp's 57466", and the arm that replaced it
is still "the wrong shape". That is the prefill gap on the hardware most
llama.cpp users own, and it makes every other CUDA row moot: flash
attention, MoE `mul_mat_id`, CUDA graphs and multi-GPU all optimise a
decode path a prompt never survives long enough to reach.

It is also the only item in this section where frink already owns the
design on the other side of the fence:
`crates/frink-metal/src/gpu.rs:2306` `mul_mm_sg_impl` is a working,
tested tiled GEMM with a per-type `Dequant` trait seam, so the CUDA
version is a port of a proven shape rather than a new design. Scope it
as a naive tiled `mul_mm` first (L, unblocks prefill), then the `dp4a`
integer path (XL, closes the constant factor).

The caveat this project should state alongside it:
`roadmap.md:82` keeps CUDA at "must compile" because no CUDA hardware
has ever been measured. That measurement gap is the reason this item is
easy to under-prioritise, and it does not make the gap smaller.

---

## 3. Sampling and constrained decoding

All of frink's sampling is one 834-line file,
`crates/frink-models/src/sampling.rs`, with **no sampler-chain
abstraction**: `Sampler::sample_with_mask` (`:252`) is a fixed pipeline
of penalties, then temperature, then top-k, then top-p, plus one
optional logit-mask callback whose only caller is JSON-object mode.
llama.cpp's samplers are `src/llama-sampler.cpp` (4106 lines in this
checkout; note the file is `llama-sampler.cpp`, not
`llama-sampling.cpp`) chained by `common/sampling.cpp:345-417`.

### 3.1 Sampler coverage

**Seven rows of this table have moved since it was written**, all on
2026-09-01. `min_p`, the `penalty_last_n` window, GBNF grammar and
`logit_bias` are no longer missing; `top_p`, the repetition penalty and
temperature ordering no longer diverge. JSON-schema-to-grammar had a converter in
the tree and a route that answered 501; both have since shipped. See
the snapshot table at the top of this file for everything else that has
moved. The per-row detail
is in §7 and §8; the table is left as the reading found it.

| Sampler | llama.cpp | frink | Severity | Size |
|---|---|---|---|---|
| greedy | `src/llama-sampler.cpp:1017` | `sampling.rs:262` + `argmax` `:446` | -- | -- |
| dist (multinomial) | `src/llama-sampler.cpp:1239` | `sampling.rs:305` | -- | -- |
| temperature | `src/llama-sampler.cpp:1901` | `sampling.rs:375` | -- (but see §3.2) | -- |
| dynamic temperature (`temp_ext`) | `src/llama-sampler.cpp:2100`, apply `:1930-1975` | **MISSING** | low | S |
| top_k | `src/llama-sampler.cpp:1330` | `sampling.rs:379-385` | -- | -- |
| top_p | `src/llama-sampler.cpp:1526` | `sampling.rs:387-402` | **high, diverges** (§3.2) | S |
| **min_p** | `src/llama-sampler.cpp:1685` | **MISSING** | **high** | S |
| typical_p | `src/llama-sampler.cpp:1795` | **MISSING** | low | S |
| top_n_sigma | `src/llama-sampler.cpp:3063` | **MISSING** | low | S |
| XTC | `src/llama-sampler.cpp:2207` | **MISSING** | low | S |
| DRY | `src/llama-sampler.cpp:3403`; defaults `common/common.h:239-243` | **MISSING** | medium | M |
| repetition penalty | `src/llama-sampler.cpp:2968`, apply `:2745-2752` | `sampling.rs:421-431` | **high, WRONG** (§3.2) | S |
| frequency / presence penalty | `src/llama-sampler.cpp:2755` | `sampling.rs:432-443` | medium | S |
| `penalty_last_n` window (default 64) | `src/llama-sampler.cpp:2700-2717`, `common/common.h:236` | **MISSING**; frink penalises the whole unbounded generation history | medium | S |
| mirostat v1 / v2 | `src/llama-sampler.cpp:2326` / `:2430` | **MISSING** | low | M / S |
| logit_bias | `src/llama-sampler.cpp:3792` | **MISSING** as a sampler; the generic `LogitMask` hook (`sampling.rs:201`) has one caller | medium | S |
| **grammar / GBNF** | `src/llama-sampler.cpp:2610`; engine `src/llama-grammar.cpp` (1522 lines) | **MISSING entirely** | **high** | L |
| lazy grammar triggers | `src/llama-sampler.cpp:2628` | **MISSING** | high | M |
| **JSON schema to grammar** | `common/json-schema-to-grammar.cpp:1158`; CLI `common/arg.cpp:2229` | **MISSING**; `response_format: json_schema` refused 501 at `crates/frink-server/src/lib.rs:1371-1381` | **high** | L |
| infill | `src/llama-sampler.cpp:4035` | **MISSING** | low | M |
| adaptive_p | `src/llama-sampler.cpp:3623` | **MISSING** | low | M |
| seed / RNG | `src/llama-sampler.cpp:1239` | `sampling.rs:207-236` (xorshift64*), server `lib.rs:1493-1499` | -- (different RNG, so bit-identical cross-engine reproduction is impossible by construction, which is worth stating in `docs/API.md`) | -- |
| `min_keep` floor on every filter | `common/common.h:228` | **MISSING**; frink has a post-hoc all-zeroes fallback to greedy (`sampling.rs:404-416`) instead | low | S |

The nearest thing frink has to constrained decoding is
`crates/frink-server/src/json_mode.rs` (83 lines): a character-class
filter zeroing any token whose decoded piece contains a character
outside a hand-written JSON-safe set (`json_safe_char`, `:11-32`). It is
not a state machine and cannot enforce balanced braces or schema
structure. The file says so at line 3.

### 3.2 Silently wrong today (jumps the queue)

#### E3. Repetition penalty compounds per occurrence

- **llama.cpp** iterates over the **candidates** and applies the divide
  **once** per candidate, whatever its count:
  `src/llama-sampler.cpp:2745-2752` (`if (logit <= 0) logit *=
  penalty_repeat; else logit /= penalty_repeat;`), with the count used
  only for the *frequency* term on the next line.
- **frink** iterates over the **history** and re-applies the divide
  once per occurrence: `crates/frink-models/src/sampling.rs:421-431`.
  A token seen `n` times is scaled by `penalty^n`.
- At the CLI default `--repeat-penalty 1.1`
  (`crates/frink-cli/src/run.rs:74`), a token repeated ten times is
  penalised 2.59x instead of 1.1x.
- **Severity: high** (output-visible on default parameters).
  **Size: S**, roughly five lines: count first, penalise once.

#### E4. Temperature is applied before the truncation filters

- **llama.cpp** puts `COMMON_SAMPLER_TYPE_TEMPERATURE` **last** in the
  default chain, after `TOP_K`, `TYPICAL_P`, `TOP_P`, `MIN_P` and `XTC`:
  `common/common.h:259-269`, consumed by the switch at
  `common/sampling.cpp:349-397`. So the nucleus is chosen on the
  temperature-1.0 softmax and temperature then sharpens the survivors.
- **frink** divides by temperature first (`sampling.rs:375`), softmaxes
  (`:377`), and only then applies top-k (`:379`) and top-p (`:387`).
- `top_k` is unaffected, because temperature is monotone and ranks do
  not move. **`top_p` genuinely differs**: at `temp=0.7, top_p=0.95`
  frink keeps strictly fewer tokens than llama.cpp on identical
  parameters, and at `temp>1` strictly more.
- **Severity: high** (a user copying llama.cpp settings gets a different
  distribution, and the likely report is "frink is more repetitive").
  **Size: S**: filter on the temperature-1.0 softmax, then rescale.

#### E5. `logit_bias` is a silent no-op on `/v1/chat/completions`

- `ChatCompletionRequest` (`crates/frink-server/src/lib.rs:1040-1131`)
  has **no `logit_bias` field**, and the struct does not use
  `deny_unknown_fields`, so a client sending it gets a 200 and unbiased
  output.
- `/v1/completions` gets this right: it deserializes the field purely in
  order to refuse it (`crates/frink-server/src/openai_extra.rs:159`
  declared, `:202` refused).
- The same hole silently swallows `min_p`, `typical_p`, `mirostat*`,
  `top_n_sigma`, `xtc_*`, `dry_*` and `dynatemp_*` on every surface.
- **Severity: medium.** **Size: S**, and the fix pattern already exists
  one file over.

#### E6. JSON-object mode is dropped under continuous batching

- The non-batched decode path passes the JSON logit mask
  (`crates/frink-server/src/generate.rs:1518-1533`).
- The batched worker calls `sampler.sample(...)` with **no mask**
  (`crates/frink-server/src/serving/batch/worker.rs:327-329`) even
  though `slot.params.json_object` is carried on the row
  (`crates/frink-server/src/serving/batch/row.rs:37`).
- Under `FRINK_CONTINUOUS_BATCHING=1`
  (`crates/frink-server/src/lib.rs:574`) the constraint is therefore
  silently off. The final `validate_json_object_output` check still
  runs, so the visible symptom is a 400 rather than bad JSON.
- This is exactly the "a copy diverges from its original and nothing
  notices" failure `CLAUDE.md` names. **Severity: medium.
  Size: S.**

Also worth recording, though not a bug: the **penalty window** differs
in both bound and content. llama.cpp penalises the last
`penalty_last_n` (default 64) accepted tokens including the prompt
(`src/llama-sampler.cpp:2700-2717`); frink penalises all generated
tokens, unbounded, and never the prompt
(`crates/frink-server/src/generate.rs:1530`,
`crates/frink-server/src/serving/batch/worker.rs:329`, both pass
`&generated_ids`).

Defaults differ too: `frink run` ships `temp 0.8 / top_k 40 / top_p
0.95 / repeat_penalty 1.1` (`crates/frink-cli/src/run.rs:62-75`)
against llama.cpp's `temp 0.8 / top_k 40 / top_p 0.95 / min_p 0.05 /
penalty_repeat 1.0` (`common/common.h:230-238`). frink cannot
reproduce llama.cpp's own defaults without `min_p`, and applies a
repeat penalty llama.cpp leaves off.

One stale comment found while reading: `lib.rs:1116` introduces
`presence_penalty` / `frequency_penalty` as "OpenAI fields we explicitly
reject rather than silently ignore". Both are in fact honoured
(`lib.rs:1179-1180`) and `validate_supported_fields`
(`lib.rs:1324-1406`) never rejects them.

### 3.3 The single highest-value sampling item

**Grammar-constrained decoding**, meaning a real GBNF engine plus the
JSON-schema-to-grammar path, mirroring `src/llama-grammar.cpp` and
`common/json-schema-to-grammar.cpp`. It is the only missing sampler that
blocks whole *features* rather than shifting a distribution:
`tool_choice: "required"` is refused because of it
(`crates/frink-server/src/lib.rs:1396-1400`), `response_format:
json_schema` is refused because of it (`:1371`), tool calling is fenced
with a `</tool_call>` stop string instead of being constrained
(`:1414-1417`), and the one structured-output mode that does ship is an
83-line character filter.

It is also the largest item here, so the sequencing that maximises value
per line is: **fix E3, E4 and E5 first** (each S-sized, each silently
changing output on parameters callers already send), **add `min_p`**
(S; llama.cpp enables it by default, so frink cannot reproduce
llama.cpp's own defaults without it), then take the grammar engine as
the one L-sized project.

**Nothing in this section appears in `roadmap.md` at all.** Sampling is
not a theme there, and `north-star.md`'s `t2-same-commands` mentions
`--temp` / `--top-k` / `--top-p` / `--repeat-penalty` only as *flag
names to alias*, not as math to match.

---

## 4. Command shapes

"Same command shapes" is an explicit north-star goal
(`north-star.md:43-44`, and the audit list at `:21`), so a flag that is
missing is a real gap and a flag that means something *different* is
worse than missing.

Scale of the surface: llama.cpp defines roughly 320 options in one file,
`common/arg.cpp:1398-4587`. frink's entire completion surface is
`InferArgs`, 22 flags, `crates/frink-cli/src/run.rs:20-135`; the
server's is `ServerArgs`, 10 flags,
`crates/frink-server/src/lib.rs:111-176`.

### 4.1 Conflicts: the same flag, a different meaning

These are the rows that matter most, because nothing errors.

| Flag | llama.cpp | frink | Why it bites | Severity | Size |
|---|---|---|---|---|---|
| **`-ngl` / `--gpu-layers`** | Offloads exactly N layers (`common/arg.cpp:2699`) | Any `N > 0` means **all** layers. `GpuLayers::offload_enabled` is `!matches!(self, Self::Count(0))` (`crates/frink-cli/src/run.rs:151-154`) -- the count is never consulted past the zero test, and the doc comment at `:78-81` admits partial placement is unimplemented | `-ngl 20` on a 40-layer model is a deliberate half-offload in llama.cpp and a full offload in frink. The user asked to fit in VRAM and silently got a different memory plan | **critical** | S to refuse, L to implement |
| **`-e` / `--escape`** | Default **on**: `common/common.h:563` `bool escape = true;` (`arg.cpp:1799-1804` registers `--no-escape` as the opt-out) | Default **off**: `run.rs:115` `default_value_t = false` | `llama-cli -p "a\nb"` emits a real newline; `frink -p "a\nb"` emits a literal backslash-n. Same command, different prompt, no warning | **critical** | S |
| `--repeat-penalty` | Applies over a window of `--repeat-last-n`, default 64 (`arg.cpp:2025,2037`) | Applies over the entire generated history, unbounded, prompt excluded (`crates/frink-models/src/sampling.rs:420-434`; call sites `run.rs:909,1035,1161,1286`, `generate.rs:1530`) -- and compounds per occurrence, see §3.2/E3 | Three divergences in one flag | high | S |
| `-dev` / `--device` | A comma-separated device **list**, `CUDA0,CUDA1` (`arg.cpp:2654`) | A single-valued enum `auto`/`none`/`cpu`/`metal`/`cuda` (`run.rs:82-89`, `OffloadDevice` `:135-142`) | `-dev CUDA0` fails to parse. The flag name is borrowed and the value grammar is not | high | M |
| `--ctk` values | Actually changes the KV dtype (`arg.cpp:2384`) | Accepts `q8_0`/`fp8`/`q4` and applies them on Metal; on CPU and CUDA the KV cache is the host `Vec<f32>` and the flag is reported as ignored | The flag is honoured where a device KV store exists | medium | M |
| `-cnv` | Both `-cnv` (on) and `-no-cnv` (off) exist (`arg.cpp:1849-1850`) | Only `--no-cnv` (`run.rs:111`); `-cnv` is a hard parse error | A pasted command that explicitly asks for conversation mode fails outright | medium | S |
| `-sys` / `--system-prompt` | `arg.cpp:1724` | Spelled `--system` (`run.rs:107`); neither llama.cpp spelling parses | same | medium | S |
| `-ctk` short form | `arg.cpp:2384` accepts `-ctk` and `--cache-type-k` | Only `--ctk` (`run.rs:133`). `frink-server` does rewrite `-ngl`/`-dev` (`lib.rs:178-184`) but that rewriter covers neither `-ctk` nor `frink run` | medium | S |
| `-n` default | `-1`, meaning to EOS (`arg.cpp:1605`) | `128` (`run.rs:40`) | `frink -m x -p "…"` stops at 128 where llama.cpp runs to EOS. Documented, still a surprise | low | S |

### 4.2 Missing flags, by how much they hurt

| Flag | llama.cpp | frink | Severity | Size |
|---|---|---|---|---|
| `-b` / `--batch-size`, `-ub` / `--ubatch-size` | `arg.cpp:1616`, `:1623` | MISSING; only env `FRINK_CB_PREFILL_CHUNK` | high | M |
| `--min-p` | `arg.cpp:1987` | MISSING; no `min_p` on `SamplingParams` (`sampling.rs:21-40`). llama.cpp turns it on by default | high | S |
| `--repeat-last-n` | `arg.cpp:2025` | MISSING | high | S |
| `-fa` / `--flash-attn` | `arg.cpp:1701` | MISSING as a flag; Metal FA is env-only (`FRINK_METAL_FA_*`) | high | S |
| `-np` / `--parallel`, `-cb` / `--cont-batching` | `arg.cpp:2491,2502,2517` | MISSING; env only (`FRINK_CB_MAX_SEQS`, `FRINK_CONTINUOUS_BATCHING`) | high | S |
| `--chat-template` / `--chat-template-file` | `arg.cpp:3637,3649` | MISSING on **both** binaries; no way to override the GGUF's template from a command line | high | M |
| `-hf` / `--hf-repo` on the run path | `arg.cpp:2970` | Separate `pull`/`download` subcommands (`main.rs:90,93`); `frink -hf user/repo -p …` fails | high | M |
| `-i` / `--interactive` | `arg.cpp:1869` | MISSING. `frink chat` is HTTP-only (`chat.rs:15-44`) and needs a running server | high | L |
| `--grammar`, `--grammar-file`, `-j` / `--json-schema` | `arg.cpp:2215,2222,2229` | MISSING (see §3) | high | XL |
| `--no-mmap`, `--mlock` | `arg.cpp:2605,2596` | MISSING (the loader always mmaps) | medium | M |
| `-ctv` / `--cache-type-v` | `arg.cpp:2397` | MISSING; V dtype is not selectable at all | medium | M |
| `--rope-scaling`, `--rope-freq-base`, `--rope-freq-scale`, `--yarn-*` | `arg.cpp:2281-2340` | ALL MISSING | medium | M |
| `-ot` / `--override-tensor`, `-cmoe` / `--cpu-moe`, `-ncmoe` | `arg.cpp:2670,2676,2683` | MISSING. **Notable**: this is per-tensor CPU/GPU placement, which is exactly what frink's `frink-core` expert-residency stack was built to execute and which nothing currently drives | high | L |
| `-sm` / `--split-mode`, `-mg` / `--main-gpu`, `-ts` / `--tensor-split` | `arg.cpp:2717,2768,2741` | MISSING; frink has no multi-GPU concept | medium | XL |
| `--lora` / `--lora-scaled` | `arg.cpp:2865,2875` | **DONE** (`run.rs`, `frink-server/src/cli.rs`; the adapter is a `WeightMatrix::Adapted` decoration in `frink-core/src/weight_matrix/lora.rs`, attached by name in `frink-models/src/lora_attach.rs`). Routed-expert targets, aLoRA and the dedicated engines refuse by name | -- | -- |
| `-l` / `--logit-bias` | `arg.cpp:2193` | MISSING; the API refuses it by name on `/v1/completions` (`openai_extra.rs:198-207`) and silently drops it on chat (§3.2/E5) | medium | M |
| `--keep`, `-r` / `--reverse-prompt`, `-sp` / `--special`, `--in-prefix`/`--in-suffix` | `arg.cpp:1630,1835,1842,1898,1906` | ALL MISSING | medium | S-M |
| `--jinja` / `--no-jinja` | `arg.cpp:3571` | MISSING; frink always uses its own engine (`chat_template.rs`) | medium | S |
| `--api-key` / `--api-key-file` | `arg.cpp:3396,3407` | MISSING as a flag; env `FRINK_API_KEY` only (`lib.rs:4297`) | medium | S |
| `--typical`, `--top-nsigma`, `--xtc-*`, `--dry-*`, `--mirostat*`, `--dynatemp-*`, `--samplers` | `arg.cpp:1932-2168` | ALL MISSING (see §3) | low | M |
| `-tb` / `--threads-batch`, `--numa`, `--prio`, `--cpu-mask` family | `arg.cpp:1482,2639,1519,1492-1573` | ALL MISSING | low-medium | S-M |
| `--override-kv`, `--check-tensors` | `arg.cpp:2845,2838` | MISSING | medium/low | M/S |
| `-md` / `--model-draft`, `--draft-max`, `--draft-min` | `arg.cpp:4089,4235,4242` | MISSING on the run path; `frink speculative` (`main.rs:364-378`) is a preset-only n-gram demo | medium | L |

**The server flag gap is the sharper half.** `ServerArgs`
(`lib.rs:111-176`) is `-m`, `--host`, `--port`, `-t`, `--device`,
`--list-devices`, `--n-gpu-layers`, `--mcp-config`,
`--exit-on-stdin-close`, `--allow-multiple-instances`. Every flag a
`llama-server` user types routinely is absent: no `-c`, no `-np`, no
`-cb`, no `-fa`, no `-ctk`/`-ctv`, no `--api-key`, no `--jinja`, no
`--chat-template`, no `--slot-save-path`, no `--cache-reuse`. Their
equivalents live in the roughly 95 `FRINK_*` environment variables in
`docs/CONFIG.md`. Even `--ctk`, which `frink run` accepts, is not a
server flag.

### 4.3 Tools with no frink counterpart

frink ships two binaries. Subcommands: `crates/frink-cli/src/main.rs:43-393`.

| llama.cpp tool | frink | Severity | Size |
|---|---|---|---|
| **`tools/quantize`** (F32/BF16 GGUF to quantized GGUF) | **NONE.** `frink quant-sensitivity` (`main.rs:265`) only measures round-trip error and writes no file. A user must keep llama.cpp installed in order to make a GGUF | **critical** | L |
| **`tools/perplexity`** (ppl, hellaswag, winogrande, KL-divergence) | **NONE.** `verify` / `parity` / `layer-divergence` compare against a reference implementation, not a corpus. Nothing in frink can answer "did this quantization hurt the model" -- which `roadmap.md:82` already names as `tooling-quality-eval` | high | M |
| `tools/imatrix` | NONE | medium | L |
| `tools/gguf-split` | NONE. frink *reads* shards (`frink_gguf::ShardedGguf`) but cannot produce or merge them | medium | M |
| `tools/export-lora` | NONE. The adapter format and the merge arithmetic exist now (`frink_models::lora`), so this is a `quantize`-adjacent write path: dequantize each named tensor, add `scale * B A`, requantize, write. Not built with `--lora` because it did not fall out of it | medium | M |
| `tools/mtmd` (multimodal) | NONE | medium | XL |
| `tools/tokenize` | Partial: `frink parity tokenize` (`main.rs:202`) is a comparison harness, not a dump | low | S |
| `tools/rpc`, `tools/tts`, `tools/cvector-generator` | NONE | low | XL |
| `tools/batched-bench` | `frink batched-bench` (HTTP-free, drives `forward_multi_seq`; ported 2026-09-11); `frink serve-bench` covers the HTTP side | — | done |
| `tools/fit-params` | `frink inspect-plan` (`main.rs:102-127`) is a genuine equivalent, arguably richer | -- | -- |
| `tools/llama-bench` | `frink bench` (`main.rs:289`), an explicit work-alike with `--compare` | -- | -- |

Doc check: `docs/CLI.md`'s flag list matches `run.rs`. One ambiguity, not
a false claim: `docs/CLI.md:479` writes `--temperature`, which the
parser does not accept (`run.rs:62` registers `--temp` with no alias);
that row is describing an HTTP request field.

### 4.4 The single highest-value CLI item

**Make `-ngl N` mean N layers, or refuse.** This is the flag every
llama.cpp user types first and the one that decides whether a model fits
at all. Today `frink -ngl 20` and `frink -ngl 99` do the identical
thing (`run.rs:151-154`), nothing errors, and the user tuning offload
against VRAM gets an OOM instead of a signal. It is a *conflict*, not a
blank, which makes it strictly worse than the whole of §4.2 under this
project's own rule: a refusal is coverage, a wrong answer is not.

If partial placement is genuinely far off, the interim fix is an
afternoon: make `0 < N < n_layers` refuse by name. That converts a
critical into a documented limitation. It also gates the
`-ot`/`-cmoe`/`-ncmoe` family, which is the per-tensor placement control
the `frink-core` residency stack was built for and which nothing
drives.

Runner-up, and critical in its own right: `--escape` defaulting to
`false` against llama.cpp's `true` (`common/common.h:563` vs
`run.rs:115`), which silently changes the prompt of every pasted command
containing `\n`.

---

## 5. Server and API surface

llama.cpp's routes are one contiguous block,
`tools/server/server.cpp:226-290`. frink's is
`crates/frink-server/src/lib.rs:4236-4293`, with path constants in
`crates/frink-api/src/routes.rs`.

| Endpoint | llama.cpp | frink | Severity | Size |
|---|---|---|---|---|
| `GET /health` | `server.cpp:233` | `lib.rs:4236` | -- | -- |
| `GET /v1/health` | `:234` | MISSING | low | S |
| `GET /metrics` | `:235` | `lib.rs:4278`, but behind the API key (llama.cpp is public with `--metrics`) | low | S |
| **`GET /props`** | `:236` | **MISSING** | high | M |
| `POST /props` | `:237` | MISSING | medium | M |
| `GET /models` | `:238` | MISSING (only the `/v1/` spelling) | low | S |
| `GET /v1/models` | `:239` | `lib.rs:4239` | -- | -- |
| **`POST /completion`, `POST /completions`** | `:240-241` | **MISSING**. Only `/v1/completions` (`lib.rs:4273`) | high | M |
| `POST /v1/completions` | `:242` | `lib.rs:4273` → `openai_extra.rs:461`. Refuses `logprobs`, `echo`, `suffix`, `logit_bias` (`openai_extra.rs:198-207`) and token-id prompts (`:181`) | -- | -- |
| `POST /chat/completions` | `:243` | MISSING (only `/v1/`) | low | S |
| `POST /v1/chat/completions` | `:244` | `lib.rs:4257` | -- | -- |
| `POST /v1/responses` | `:246` | `lib.rs:4243`. Bare `/responses` MISSING | low | S |
| `POST /v1/messages` (Anthropic) | `:250` | `lib.rs:4268` (`anthropic.rs`) | -- | -- |
| **`POST /infill`** | `:251` | **MISSING**; nothing named infill exists in `frink-server` or `frink-api` | high | L |
| `POST /embedding`, `/embeddings` | `:252-253` | MISSING (only `/v1/embeddings`) | low | S |
| `POST /v1/embeddings` | `:254` | `lib.rs:4276` → `openai_extra.rs:331`. Mean/last pooling of a **decoder's** hidden states; no embedding-model path (`roadmap.md:69-71`) | -- | -- |
| `POST /rerank`, `/v1/rerank` | `:255-258` | MISSING; no `rerank` string in either crate | medium | L |
| **`POST /tokenize`** | `:259` | **PATH MISMATCH.** frink mounts `/v1/tokenize` (`lib.rs:4274`, `routes.rs:21`); llama.cpp has no `/v1/` spelling and frink has no bare one. Neither client works against the other | high | S |
| **`POST /detokenize`** | `:260` | **PATH MISMATCH**, same (`lib.rs:4275`, `routes.rs:22`) | high | S |
| `POST /apply-template` | `:261` | MISSING, despite an 860-line `chat_template.rs` | medium | S |
| `GET /lora-adapters`, `POST /lora-adapters` | `:269-270` | **DONE** (`frink-server/src/lora.rs`), with the per-request `lora` field on the three completion routes. Differences from upstream are named there: an unknown id is a 400 rather than ignored, and a scale change is exclusive against the generations in flight rather than a per-slot list | -- | -- |
| **`GET /slots`** | `:272` | **MISSING.** Nearest are frink-only and differently shaped: `/v1/stats` (`lib.rs:4252`), `/v1/requests` (`:4253`), `/v1/cache/status` (`:4254`) | high | M |
| `POST /slots/:id` (save/restore/erase) | `:273` | MISSING; no KV save/restore to disk | medium | L |
| `POST /models`, `/models/load`, `/models/unload` | `:226-228` | frink has `/admin/models{,/load,/unload}` (`lib.rs:4283-4285`) -- same capability, different paths | low | S |
| `POST /v1/audio/transcriptions` | `:248` | MISSING | low | XL |
| `POST /tools` (MCP proxy) | `:347` | MISSING. `--mcp-config` (`lib.rs:156`) is a stub; `docs/API.md:820` confirms nothing is invoked | low | L |

frink-only routes, for completeness: `/v1/cancel`,
`/v1/cache/rebuild`, `/v1/admin/prepare-stop`, `/cache/stats`,
`/admin/download`, `/admin/tasks`, `/admin/stats`, `/v1/conversations*`,
`/v1/stream/{id}` and `/v1/stream/{id}/poll`.

### 5.1 Per-request features

| Feature | llama.cpp | frink | Severity |
|---|---|---|---|
| SSE streaming | native `/completion` shape and OpenAI shape; writer at `tools/server/server-stream.cpp` | OpenAI + Anthropic chunk shapes only (`crates/frink-server/src/sse.rs`). The native `data: {"content":…,"stop":false}` frame does not exist, because the endpoint does not | high |
| **`logprobs` / `top_logprobs` / `n_probs`** | `tools/server/server-context.cpp:1818`, `populate_token_probs` `:1987-2040` | **NO.** Refused by name: `lib.rs:1357-1360` (chat), `openai_extra.rs:198-207` (completions). The fields deserialize purely to produce a named refusal | high |
| Multimodal image input | `tools/mtmd`, `--image` `arg.cpp:2556` | NO. `image_url` parts are detected (`lib.rs:885,905`) solely to refuse (`:1353`) | medium |
| Tool / function calling | `tools/server/server-tools.cpp` | **Partially yes.** `tools`/`tool_choice` are real fields (`lib.rs:1088`), gated at `:1195-1201`, with a 2938-line parser at `policy/parser/tool_call.rs`. `auto` and `none` work; `tool_choice: "required"` (`:1394-1399`) and a named choice (`:1400-1403`) are refused for want of constrained decoding | medium |
| Grammar / JSON-schema constrained decoding | `--grammar` `arg.cpp:2215`, `--json-schema` `:2229`, `tools/server/server-schema.cpp` | NO, anywhere in the engine. This one absence is the reason for three of the refusals above | high |
| Continuous batching | on by flag | yes (`serving/batch/`), env-gated not flag-gated | low |
| API key auth | `--api-key` `arg.cpp:3396` | env `FRINK_API_KEY` only (`lib.rs:4297`); accepts both `Authorization: Bearer` and `x-api-key` | medium |

**Doc check, and it is a good result.** `docs/API.md:9-38` lists exactly
the routes the router registers, and its claim at `:39-41` that every
path is one constant in `frink-api` holds against `routes.rs` and
`lib.rs:4236-4293`. The "Not yet" section (`:820-838`) matches the
refusal sites in code. **No API doc/code disagreement was found**, which
is worth recording because it is rarer than it should be. The one thing
the doc does not surface is that `/v1/tokenize` and `/v1/detokenize` are
frink-invented paths no llama.cpp client knows.

### 5.2 The single highest-value server item

**`POST /completion` and `/completions` in llama.cpp's native request
and SSE shape** (`server.cpp:240-241`). frink covers the OpenAI dialect
via `/v1/completions`, but the native endpoint is what llama.cpp's own
web UI, `llama.vim`, and a long tail of scripts actually speak,
including the `data: {"content":…,"stop":false}` streaming frame that
frink's SSE writer has no equivalent for. It is mostly re-shaping an
existing generation path rather than new engine work (M), and it turns
"frink cannot be dropped in behind my existing tooling" into "it can".

The cheap companion in the same change is aliasing `/tokenize` and
`/detokenize` onto the handlers already mounted at `/v1/tokenize` and
`/v1/detokenize` (`lib.rs:4274-4275`): an S-sized edit removing two
silent 404s for every llama.cpp client. The runner-up on severity though
not on cost is `logprobs` (`lib.rs:1357`), which evaluation harnesses
and speculative-decode debugging both require and which today is a named
refusal with no path forward that does not touch the sampler.

---

## 6. Other material findings

Things that did not fit the five categories, recorded rather than
forced.

| Finding | llama.cpp | frink | Severity | Size |
|---|---|---|---|---|
| **Vocabulary types** | 6 real types plus `NONE`: SPM, BPE, WPM, UGM, RWKV, PLAMO2 (`include/llama.h:73-79`) | **3**: BPE (`gpt2`, `gemma4`), SPM (`llama`), Unigram (`t5`). `bert`, `rwkv`, `none` refuse **by name**, everything else including `plamo2` refuses generically (`crates/frink-cli/src/run.rs:385-404`) | high | L |
| **State save / restore** | `llama_state_save_file`, `llama_state_seq_save_file`, `llama_state_get_size` (`include/llama.h:801,843,877`) -- this is what powers `POST /slots/:id` save/restore and `--prompt-cache` | **none.** No KV serialization anywhere; the radix prefix cache is in-process only | medium | L |
| **Chat templating** | A 6,298-line homegrown Jinja engine, `common/jinja/` (`value.cpp` 1586, `value.h` 759, plus lexer/parser/runtime/caps), used via `common/chat.cpp:334` | **Real parity, and this is worth recording as a positive.** frink evaluates the checkpoint's own template with `minijinja`; the last substring sniffer is gone (`crates/frink-server/src/chat_template.rs:1-56`). It goes further than llama.cpp in one place: the end-of-turn marker, the tool grammar and the effort vocabulary are all **probed by rendering**, not by pattern-matching template source | -- | -- |
| **Backend abstraction** | `ggml_backend_*` with `supports_op` per device (179 `GGML_OP_` arms for CUDA, `ggml-cuda.cu:4733`), so a new backend implements an interface | frink has **no backend trait at all**; the decode graph is hand-written per backend in `crates/frink-models/src/decoder.rs` and `crates/frink-metal/src/attn.rs` | high (it is the prerequisite for any third backend) | XL |
| **Offload disclosure** | `src/llama-model.cpp:1632,1637` unconditionally prints `offloading N repeating layers to GPU` / `offloaded N/M layers to GPU` | `frink run` prints no equivalent; `active_backend()` is read only by `bench_model.rs:190` and `parity.rs:115` | medium | S |

The backend-abstraction row is worth reading next to `roadmap.md:61-63`,
which already puts `backend-seam-refactor` before `vulkan-beachhead` for
exactly this reason. This inventory agrees with that ordering and adds
one argument to it: the CUDA gap in §2.4 is not 65 missing files, it is
one missing *interface* plus 65 kernels, and the interface is the part
that determines whether the 65 can be written incrementally.

---

## 7. Silently wrong today, all in one place

These are bugs, not gaps. Under the bar in `north-star.md:51-56` a
refusal is coverage and a wrong answer is not, so these jump the queue
regardless of size.

**ALL SEVEN ARE FIXED, later the same day.** The "Live today?" column is
kept as written, because it is what the reading found; the "Fixed by"
column is the record of what closed it. The estimate below the table was
right: it was about a day of work.

| # | What | Where | Live today? | Fixed by |
|---|---|---|---|---|
| **E3** | Repetition penalty compounds as `penalty^n` where llama.cpp applies it once | `crates/frink-models/src/sampling.rs:421-431` vs `src/llama-sampler.cpp:2745-2752` | **Yes**, on every model at the CLI default `--repeat-penalty 1.1` | `8356c74`. `apply_history_penalties` counts first and divides once per token |
| **E4** | Temperature is applied before top-p, where llama.cpp applies it last | `sampling.rs:375-402` vs `common/common.h:259-269` | **Yes**, whenever `temp != 1` and `top_p < 1` | `8356c74`. `filtered_distribution` runs top-k, top-p, min-p, then temperature, with the renormalisation between steps in `sampler_chain` |
| **E1** | `phi3` windows every layer where llama.cpp windows none | `loader.rs:417-419` + `config.rs:359-377` vs `src/models/phi3.cpp:13-24` | **Yes** beyond the declared window, on an architecture that is on the *audited* list | `2d36356`. See the correction below: this row's Phi-4 caveat was wrong |
| **E5** | `logit_bias` is silently dropped on `/v1/chat/completions` | `crates/frink-server/src/lib.rs:1040-1131` (no field, no `deny_unknown_fields`) vs `openai_extra.rs:159,202` which refuses it properly | **Yes** | `1b4ab74`. Both routes call one `refuse_logit_bias`; an empty `{}` is served rather than refused |
| **E6** | JSON-object mode is dropped under continuous batching | `serving/batch/worker.rs:327-329` vs `generate.rs:1518-1533` | Yes, under `FRINK_CONTINUOUS_BATCHING=1` | `1b4ab74`, and it was dropped three ways, not one: also by the Metal greedy lm_head fold at `temperature <= 0`, and by `generate_engine` passing no detokenizer at all |
| **E7** | Q5_0 prefills on Metal and decodes on CPU, from a stale exclusion table its own comment already flags | `crates/frink-core/src/weight_matrix.rs:418-455` | Yes | `e7a6036`. `Q5_0_MATVEC_KERNEL_SRC` exists and the two Metal tables now name the same seven kinds. Still unmeasured: no Q5_0 row in the bench suite |
| **E2** | `unsupported_feature_keys` cannot see a softcap llama.cpp hardcodes | `capability.rs:1156-1177` vs `src/models/grok.cpp:9-11` | **Latent** -- `grok` refuses as unaudited first. A trap for whoever audits it | `c7f9c63`, and the reading understated it. The refusal named a key **no converter writes**, so that arm had never matched anything for any non-Gemma architecture, ever. A gate that cannot fire is worse than no gate, because it reads as coverage |

### One row of this section was itself wrong

E1 above said Phi-4-mini "writes `0` and is filtered out ... which is
exactly why the suite's Phi-4-mini bench row never saw this", citing the
converter at `conversion/phi.py:167-171`. Reading the actual file showed
otherwise: `models/Phi-4-mini-instruct-Q4_K_M.gguf`, a row in the
benchmark suite, declares `phi3.attention.sliding_window = 262144`. The
bench row did not dodge the bug by writing zero; it was affected and the
window was simply larger than any prompt in `pp512`/`tg128`. Recorded
here because "read the converter" is weaker evidence than "read the
checkpoint", and this document's own standard says so.

---

## 8. The top ten, ranked against the north star

Ranked by the bar at `north-star.md:51-59`, in its own order: **(1) never
compute something else, (2) same tokens at temperature 0, (3) not
slower, (4) the command a user knows works.** "Tracked" means an
existing `roadmap.md` item already covers it; "new" means this inventory
is the first place it is written down.

**Items 1 through 8 all landed on 2026-09-01**, in roughly the order
this table ranks them; item 9's `/tokenize` and `/detokenize` aliases
were in flight as this was written, and item 10 (Vulkan) is open. The
`Landed` column below records what closed each, and item 7 carries the
caveat that matters most in this whole document: the CUDA GEMM **has
never executed on a GPU**.

| # | Item | Why here | Size | Landed |
|---|---|---|---|---|
| **1** | **Fix the repetition penalty (E3)** | Bar item 2, and it is live on *every* model at the CLI's own default. Five lines: count first, penalise once | S | **DONE** `8356c74` |
| **2** | **Apply temperature after the truncation filters (E4)** | Bar item 2. The same parameters produce a different nucleus than llama.cpp, and the user-visible symptom is "frink is more repetitive" | S | **DONE** `8356c74` |
| **3** | **`phi3` must not window when the key says it should (E1)** | Bar item 1, on an architecture frink *claims as audited*. This row's claim that the Phi-4 bench row "writes `0` and dodges the bug" was itself wrong: the file declares 262144. See the correction in §7 | S | **DONE** `2d36356` |
| **4** | **`-ngl N` offloads N layers, or refuses** | Bar item 4, and it is a *conflict* not a blank: the flag everyone types first silently means something else. If partial placement is far off, refusing `0 < N < n_layers` converts a critical into a documented limitation in an afternoon | S to refuse, L to implement | **DONE (the refusal half)** `7b3a3f5`. `GpuLayers::check_supported` refuses `0 < N < n_layers` by name; partial placement is still unimplemented |
| **5** | **`--escape` default must match llama.cpp's `true`** | Bar item 4. Every pasted command containing `\n` currently gets a different prompt, with no warning | S | **DONE** `7b3a3f5` |
| **6** | **Triage the 46 unaudited refusals** into fixture-away / one-match-arm / new-code, and say which in the error | Bar items 1 and 4. It is the only thing that makes "how far is frink from llama.cpp on models" legible, and §1.3 shows the three classes genuinely differ. It also produces the work queue that `b2-close-the-68` and the model-layer split both need | M | **DONE** `126252b`, `52a25b6`. 9 fixture-away / 7 one-match-arm / 26 new-code / 4 unknown, pinned by `tests/unaudited_triage.rs`; the count fell 47 to 46 when the triage found `minicpm3` was MLA |
| **7** | **A batched quantized GEMM on CUDA (`mul_mm`)** | Bar item 3, on the hardware most llama.cpp users own. `frink-cuda` has **no matrix-matrix product at all** (§2.4), and the repo's own comment records the consequence. It is a port of `frink-metal`'s proven `mul_mm_sg_impl`, not a new design | L | **WRITTEN AND WIRED, NEVER RUN ON A GPU** `5dd8202`, `75780d8`. Q8_0 and Q4_0 only. Evidence is a thread-by-thread scalar twin plus a host harness that compiles and executes the emitted CUDA against a barrier shim; the hardware test is `#[ignore]`d with "NEVER RUN" as its reason. This is not a performance claim and `docs/` makes none |
| **8** | **Grammar / GBNF and JSON-schema constrained decoding** | Bar item 4 at feature granularity: three separate 501s (`tool_choice: required`, named `tool_choice`, `response_format: json_schema`) all trace to this one absence, and the structured-output mode that does ship is an 83-line character filter | L | **DONE for GBNF** `7cfdab8`, `5bd3518`, `a674f95`. Parser + stack machine + `grammar` on chat and completions, reaching all three decode paths. A JSON-Schema-to-GBNF converter exists in `frink_models::grammar`, but `response_format: json_schema` still answers 501, and lazy grammars (`trigger_patterns`) are unported, so both `tool_choice` 501s stand |
| **9** | **Native `POST /completion` + alias `/tokenize` and `/detokenize`** | Bar item 4 for everything that is not an OpenAI client: llama.cpp's own web UI, `llama.vim`, and a long tail of wrappers. The alias half is an S-sized edit removing two silent 404s | M + S | **NEW** |
| **10** | **Vulkan** | The largest single hardware gap and the only backend covering AMD, Intel and Android from one codebase. Ranked tenth only because everything above it is smaller and lands sooner, which is `roadmap.md:5-7`'s own rule | XL | **Tracked** (`d-hardware-reach`) |

**Eight of the ten are new to this document.** The two that are tracked
(items 6 and 10) are the two largest, which is the pattern worth
noticing: the roadmap holds the big structural themes and has no entry
for the small, live, output-changing defects that items 1 through 5 are.
Items 1, 2, 3, 5 and the refusal half of 4 are together roughly one day
of work and close five of the seven rows in §7.

### Just below the line, and why

- **`min_p`** (S). llama.cpp enables it by default (`common/common.h:230-238`),
  so frink cannot reproduce llama.cpp's own defaults without it. It is
  under the line only because it changes a distribution rather than
  contradicting a stated parameter. **NEW.**
- **E5 and E6** (S each). Both are silent no-ops rather than wrong math,
  and both are in §7. **NEW.**
- **A `quantize` tool** (L). frink cannot produce a GGUF at all
  (`tools/quantize` has no counterpart; `frink quant-sensitivity` only
  measures). Under the line because llama.cpp stays a one-time
  dependency rather than a runtime one, but it is the largest single
  "you still need llama.cpp installed" item. **NEW.**
- **`logprobs` / `n_probs`** (M). Evaluation harnesses and
  speculative-decode debugging both need it; refused by name at
  `lib.rs:1357`. **NEW.**
- **Metal attention sinks** (M), which would take gpt-oss off CPU-only
  (`decoder.rs:848-853`; llama.cpp has them at
  `ggml-metal.metal:6386,6992-6997`). Honestly disclosed in
  `docs/MODELS.md:77`, which is why it is not higher. **NEW.**
- **Five of seven loaders never seal the kernel registry** (S), so every
  MLA, GLM, Kimi and hybrid model runs with no backend disclosure. The
  mechanism already exists and works
  (`crates/frink-core/src/kernel_registry.rs:687-700`); it is five call
  sites. **NEW.**
- **A perplexity / quality tool** (M). Already `roadmap.md:82`
  (`tooling-quality-eval`). **Tracked.**
- **Embedding and reranking models** (L). Already `roadmap.md:69-71`
  (`b3-the-embedding-model-gap`), and §5 confirms it: no `/rerank` route
  and `/v1/embeddings` pools a decoder. **Tracked.**

---

## 9. What this document did not reach

Stated plainly, because a gap in the inventory is not the same as an
absence of a gap.

- **Numerics.** Nothing here compares frink's kernels against ggml's
  for accuracy. `frink parity` exists for this
  (`north-star.md:33`) and running it across the suite would say more
  than any table above.
- **The remaining ~35 of the 47 unaudited architectures.** §1.3 reads
  eleven of them in llama.cpp's source. The rest are not claimed either
  way, which is what item 6 in §8 is for.
- **Tokenizer behaviour beyond the vocabulary-type count.** The
  pre-tokenizer work in `roadmap.md:21-23` closed a great deal and is
  reported there in detail; this document did not re-audit it.
- **CUDA and Vulkan claims are read, never run.** Nobody has executed
  frink on a CUDA host (`roadmap.md:82`), so every CUDA row here is a
  source reading. That is the correct standard for an inventory and it
  is not a substitute for a measurement.
- **`ggml`'s CPU op set** was compared at the SIMD-tier and quant-type
  level, not op by op. A per-op CPU diff would likely find more.

---

## 10. Measured 2026-09-01: K-quants diverge from llama.cpp, and it is the QUANT not the architecture

Everything above §9 is a source reading. This section is a measurement,
run on a quiet host with `frink parity` after the tokenizer half was
brought to MATCH.

**The tokenizer half is clean.** Nine of nine checkpoints libllama can
load tokenize identically to llama.cpp across the 19-case corpus, 3,839
tokens: Llama-3.2, Qwen2.5, DeepSeek-R1-Distill, OLMoE, TinyLlama,
gemma-2, Phi-4-mini, Yi-1.5, Mistral-7B. (`gemma-4` is skipped — the
installed libllama predates the architecture.)

**The logit half is not, and the controlled experiment localises it.**
Five quantizations of the SAME checkpoint, same prompt, frink CPU
against llama.cpp CPU so no backend difference is in play:

| `models/Llama-3.2-1B-Instruct-*.gguf` | verdict |
|---|---|
| `Q8_0` | **MATCH** |
| `IQ4_XS` | **MATCH** |
| `Q6_K` | DRIFT |
| `Q5_K_M` | DRIFT |
| `Q4_K_M` | DRIFT |

One model, one architecture, one prompt: the only variable is the quant
format, so this is not a graph bug. It reproduces across architectures —
Qwen2.5-1.5B `Q4_K_M` (KL 7.7e-3), gemma-2-2b `Q4_K_M` (KL 6.5e-3) and
Llama-3.2-1B `Q4_K_M` (KL 1.8e-3) all DRIFT, while TinyLlama `Q8_0`
MATCHes at KL 2.4e-4.

**Impact is bounded and should not be overstated: the top-1 token is
IDENTICAL on every row measured**, and top-10 overlap is 9 or 10 of 10.
This is a distribution difference, not a wrong answer, and it is
invisible to any greedy-text comparison — which is exactly why
`frink parity` compares distributions rather than text.

**Where it is NOT.** `dequant_q4_k`'s six-bit scale/min extraction is
byte-identical to llama.cpp's `get_scale_min_k4`, including the
awkward `j >= 4` branch that borrows bits across the array. Checked
line by line.

**The leading hypothesis, and the evidence against it.** llama.cpp does
not dot K-quants against f32 activations: `GGML_TYPE_Q4_K`, `Q5_K` and
`Q6_K` all declare `vec_dot_type = GGML_TYPE_Q8_K`, so the ACTIVATION is
quantized to 8 bits and the product is an integer MAC scaled at the end.
frink keeps activations in f32. Two different numeric paths, and
frink's is the more precise one — llama.cpp is buying int8 SIMD
throughput with accuracy. `Q8_0` uses `vec_dot_type = GGML_TYPE_Q8_0`,
a much closer path, and it MATCHes.

That theory does not survive contact with `IQ4_XS`, which ALSO declares
`vec_dot_type = GGML_TYPE_Q8_K` and MATCHES. So the activation
quantization cannot be the whole story, and the honest position is that
three of four data points fit and one does not. **Do not act on the
hypothesis until the IQ4_XS row is explained** — it is the row that
would falsify it.

### RESOLVED, same day: the IQ4_XS row was mislabelled by its own filename

`models/Llama-3.2-1B-Instruct-IQ4_XS.gguf` **contains no IQ4_XS tensors
at all.** Reading the tensor table directly:

| file | quantized tensors | dominant `vec_dot_type` | verdict |
|---|---|---|---|
| `Q8_0` | 113x `Q8_0` | `Q8_0` | **MATCH** |
| `IQ4_XS` | 96x **`IQ4_NL`** + 16x `Q5_K` + 1x `Q6_K` | `Q8_0` | **MATCH** |
| `Q6_K` | 113x `Q6_K` | `Q8_K` | DRIFT |
| `Q5_K_M` | 96x `Q5_K` + 17x `Q6_K` | `Q8_K` | DRIFT |
| `Q4_K_M` | 96x `Q4_K` + 17x `Q6_K` | `Q8_K` | DRIFT |

`IQ4_NL` declares `vec_dot_type = GGML_TYPE_Q8_0`, not `Q8_K`. So the
row that appeared to falsify the hypothesis was never a Q8_K file --
the name is the quantization RECIPE, and the recipe fell back to
`IQ4_NL` for all 96 per-layer weights.

**All five data points now fit, with no exception**: the verdict tracks
the `vec_dot_type` of the 96 per-layer tensors exactly. `Q8_0`
activations MATCH, `Q8_K` activations DRIFT.

Note the `IQ4_XS` file still carries 16 `Q5_K` tensors and MATCHes
anyway, which is consistent with a per-tensor accumulation difference
that scales with how many layers use it rather than a single wrong
constant.

**So this is not a frink defect.** llama.cpp quantizes the ACTIVATION
to 8 bits for K-quant dots and accumulates in integers; frink keeps
activations in f32. Both are defensible, and frink is on the more
accurate side of the trade -- llama.cpp is buying int8 SIMD throughput
with precision. The `DRIFT` verdict is `frink parity` reporting a real
difference, correctly, and mislabelling its cause: the threshold assumes
both engines took the same numeric path.

**What is still owed**, and it is small: `frink parity`'s verdict text
should say so, rather than telling the reader to go do a per-layer
divergence run on a difference that is expected. The decisive
confirmation, if anyone wants it, is to dot one K-quant row with a
Q8_K-quantized activation and check the gap matches the observed KL.

**What this does NOT excuse.** A KL of 7.7e-3 on Qwen2.5 with a top-2
margin of 2.5e-1 is comfortable; gemma-2's margin is 4.7e-2, which is
not. Being the more accurate engine is not the same as being
interchangeable, and any future claim of bit-identical output on
K-quants is false.

**Why this is not filed as a bug.** Nothing here shows frink computing
the wrong thing. It shows two engines making different, defensible
numeric choices, with frink on the more accurate side of the trade. It
belongs in the gap inventory because "same or better performance on the
same models" is the goal and a KL of 7.7e-3 is the sort of thing that
becomes a wrong answer at a longer context or a narrower top-2 margin —
gemma-2's margin here is 4.7e-2, which is not much headroom.

### 10.1 Measured 2026-09-12: on PLM's MLA, Q8_0 drifts too, and it is llama.cpp's loss

The section above says `Q8_0` activations MATCH. That held for every
generic-path checkpoint measured (Llama-3.2-1B Q8_0: 2.4e-4 with flash
attention off, 6.9e-4 with it on) and does NOT hold for the first real
MLA checkpoint put in front of `frink parity`: **PLM-1.8B-Instruct
Q8_0** reads `WRONG` at KL 3.54e-2 (top-1 agrees, top-10 overlap 9/10).

The arbiter is the dequantized file (`scripts/dequantize_gguf.py`
writes the same checkpoint as ALL_F32; 6.8 GiB), run through both
engines with `--dump-logits`, on the same five token ids:

| | KL |
|---|---|
| llama.cpp f32 vs frink f32 (the graph) | **4.53e-5** |
| llama.cpp f32 vs llama.cpp Q8_0 (llama.cpp's own quantization loss) | **3.72e-2** |
| llama.cpp f32 vs frink Q8_0 (frink's) | **4.52e-5** |
| frink f32 vs frink Q8_0 | 2.8e-9 |
| llama.cpp Q8_0 vs frink Q8_0 (what `parity` reports) | 3.54e-2 |

So the graph agrees to 4.5e-5 (the residual is llama.cpp's F16 KV
cache; the F32 fixtures agree to 1e-13), frink's Q8_0 inference is
2.8e-9 from its own f32 -- Q8_0 WEIGHTS are that lossless -- and the
entire `WRONG` is the reference's 8-bit ACTIVATION quantization, which
on this graph costs three orders of magnitude more than on a Llama.
Where it bites was bisected on a real-dims synthetic fixture, quantizing
one tensor at a time: `attn_kv_a_mqa` and `attn_kv_b` Q8_0 each move
llama.cpp 7e-2 from the f32 answer, `attn_q` 8e-3, `attn_output` 1e-4,
`ffn_up` 4e-8. The compressed latent (`kv_lora_rank = 512` wide,
RMS-normed) is quantized to Q8_0 in 32-wide blocks before `kv_b`
re-expands it into every head's K and V, and MLA latents carry the
outlier channels that per-block 8-bit quantization is worst at.

Two things follow. `frink parity`'s `WRONG` line for Q8_0 (1e-2
absolute, from llama.cpp's build-to-build spread) assumes the reference
is exact to within noise on Q8_0, and on MLA it is not; the dequantized
file is the way to tell, and the recipe is in `docs/CLI.md`. And the
number worth having about PLM is the other one: frink serves the Q8_0
file at 4.5e-5 from f32 where llama.cpp serves it at 3.7e-2.

The same run found that llama.cpp aborts on this file with flash
attention on (`ggml_set_rows: GGML_ASSERT(a->ne[0] == b->ne[0])` from
`build_attn`, K head 192 / V head 128; 1269cb1), so
`LLAMA_LOGITS_FLASH_ATTN=0` was added to the dumper; that is
llama.cpp's defect, not measured here beyond noting it.
