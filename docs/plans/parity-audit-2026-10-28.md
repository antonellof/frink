# Parity audit, 2026-10-28: the pin moves to `604cc430`

What this is: the llama.cpp pin moved from `5b59b83` (2026-09-19) to
`604cc430` (2026-10-28), and every census frink's capability tables are
derived from was re-run against the new tree in the same PR, before any
new model row. It follows
[`parity-audit-2026-09-19.md`](parity-audit-2026-09-19.md), whose
section 1.3 said why the two belong together: a census over a stale
tree is wrong in the direction that matters, a graph that STARTED
reading a key.

Method: every count below came from a command, and the command is
printed beside it. The two trees were compared as plain directories
(`old/` at `5b59b83`, `new/` at `604cc430`, both checked out from
`.scratch/llama.cpp`), so `diff` and `grep` are enough to re-run any of
it.

## 0. The one-line answer

913 commits, 13 new architecture names, 14 new files under
`src/models/` (13 graphs and one shared builder), nothing removed. Nine
of the thirteen are text generation; every one of the thirteen now has a
catalog row that refuses or defers with the line that decides it. Three
SERVED architectures changed behaviour upstream in a way frink would
have computed differently, and all three are fixed or refused by name
here (section 2).

## 1. What the pin brought

### 1.1 Counts

```
$ git -C .scratch/llama.cpp rev-list --count 5b59b83..604cc430      # 913
$ ls old/src/models/*.cpp | wc -l                                   # 155
$ ls new/src/models/*.cpp | wc -l                                   # 169
$ diff <(ls old/src/models) <(ls new/src/models)                   # 14 added, 0 removed
$ grep -oE '\{ *LLM_ARCH_[A-Z0-9_]+, *"[^"]+" *\}' {old,new}/src/llama-arch.cpp \
    | sed -E 's/.*"([^"]+)".*/\1/' | sort      # 153 names -> 166
$ comm -13 names_old.txt names_new.txt                              # the 13 below
```

`crates/frink-models/tests/llama_cpp_arch_names.txt` carries the 166,
and `tests/arch_coverage.rs` holds that each has a catalog row the
resolver reaches (166 of 166). The `#[ignore]`d re-derivation in the
same file reads `.scratch/llama.cpp` under the repo root, which a
worktree does not have, so its parser was applied to `604cc430`'s
`llama-arch.cpp` by hand (the same split on `LLM_ARCH_` and the first
quoted string): it yields the same 166 as the regex above.

### 1.2 The thirteen, ranked by how close each is to running

| arch | graph | catalog answer | what decides it |
|---|---|---|---|
| `kolibri1` | `kolibri1.cpp` (184) | AUDITED 2026-10-29 (`tests/kolibri1_graphs.rs`, a 32-step decode in `tests/decode_step_graphs.rs`; was NEW CODE, before that ONE MATCH ARM) | the router ranks `logits + exp_probs_b` BEFORE the sigmoid (`:138`) and hands the ids to `build_moe_ffn` as `selected_experts_in`, while the DeepSeek-V3 rule frink has adds the bias after the gating (`llama-graph.cpp:2279`); found when wiring it. The rest was read as table rows: RoPE on sliding layers only (`:98`, `RopeLayers::SlidingOnly`), the window array (`:17`), Gemma-2 sandwich norms (`:42,51`), per-head QK RMS, sigmoid routing with `exp_probs_b` precomputed into `selected_experts_in` (`:133-155`), one shared expert |
| `lfm3` | `lfm3.cpp` (217) | AUDITED (`tests/lfm3_graphs.rs`, a 32-step decode in `tests/decode_step_graphs.rs`) | LFM2's layers plus a per-head sigmoid attention gate (`:77,132-138`), the window array (`:13-18`), `rope_freqs` on full layers only (`:79-82,113`, `step35`'s rule), optional sandwich norms. Its d1 variant is non-causal and refused |
| `flint` | `flint.cpp` (194) | NEW CODE (tokenizer) | the graph is served seams (`attention.rope_pattern`, the second reader of 169; window array; REQUIRED sinks; shared expert). Its `flint` pre-tokenizer is a sequence of two regexes (`llama-vocab.cpp:285-290`), which `tokenizer::pretokenize` cannot express |
| `beam` | `beam.cpp` (261) | NEW CODE | the residual is `h / sqrt(i) + r * sqrt((i - 1) / i)` (`:109-113`); a POWER-form attention temperature (`attn_temp_pow`, `:51`; `llama-graph.cpp:221-227`) on its NoPE layers |
| `maion-coder` | `maion-coder.cpp` (149) | NEW CODE | a latent MoE (`:7,54-58,122-134`), the shape frink refuses by name for Nemotron-3 Super |
| `k2-horizon` | `k2-horizon.cpp` (360) | NEW CODE | a grouped RMS norm on every layer (`:7-10,241`), a base-2 softplus attention gate (`:283-291`), value experts (MoVA, `:26-34,137-190`) |
| `muse-spark` | `muse-spark.cpp` (238) | NEW CODE | a norm-preserving residual with per-channel alpha/beta tensors (`:53-59,167-168,209-210`), a latent MoE with a pre- and a post-expert norm (`:66-74`; the new `post_exp_norm` argument of `build_moe_ffn`, `llama-graph.cpp:2556-2560`) |
| `deepseek41` | `deepseek41.cpp` (933) | dedicated | `deepseek4`'s CSA/HCA + mHC plus per-layer KV / top-k source layers (`:31-66`) and an Engram n-gram memory (`:84-149`) |
| `glm5-next` | `glm5-next.cpp` (1106) | dedicated | KDA + nope-MLA with a k-pool DSA indexer and mHC (`:4-5,15-49`) |
| `sensenova_u1` | `sensenova-u1.cpp` (190) | deferred multimodal | vision-language; text positions are `[t, 0, 0, 0]` across the M-RoPE sections (`:3-25`), so not NEOX band for band even on text |
| `gemma-embedding2` | `gemma-embedding2.cpp` (232) | deferred encoder | `causal_attn = false`, symmetric SWA (`:4-8`) |
| `clef` | `clef.cpp` (647) | deferred (scoring) | a decision model: `qwen35`'s layers under a scorer, one score per token, no LM head (`:8-24`) |
| `vllm-decision2` | `vllm-decision2.cpp` (179) | deferred (scoring) | the same on `qwen3` (`:7-15`) |

The fourteenth file, `lfm-base.cpp`, is the short-convolution block and
the two LFM FFN bodies moved out of `lfm2.cpp` so `lfm3.cpp` can share
them. Diffing `lfm2.cpp`'s causal graph against the old one shows a pure
move (section 2.4).

The triage distribution went (0, 0, 3, 1) to (0, 2, 8, 1)
(fixture-away, one match arm, new code, unknown) and the triaged count 4
to 11. `docs/MODELS.md` carries the table; `tests/documented_counts.rs`
pins it.

## 2. Served architectures whose upstream behaviour changed

This is the part of a pin move that matters more than the new names: a
file of an architecture frink already runs, computed differently from
the reference from today on.

### 2.1 The generic builders, read first

```
$ git log --oneline 5b59b83..604cc430 -- src/llama-graph.cpp src/llama-model.cpp \
    src/llama-hparams.cpp src/llama-hparams.h src/llama-model-loader.cpp src/llama-arch.cpp
                                                                      # 81 commits
$ diff -U2 old/src/llama-graph.cpp new/src/llama-graph.cpp | wc -l    # 1253
$ diff -U3 old/src/llama-model.cpp new/src/llama-model.cpp | wc -l    # 1605
```

Every behavioural change in them, and who it reaches:

| builder change | reaches | frink |
|---|---|---|
| `build_amax_src1_f16` around `down_exps` for `mistral4`, `mellum`, `flint`, `laguna` (`llama-graph.cpp:2532`, #31147) | `mellum`, `laguna` (served) | none needed: it divides by a power of two and multiplies back, an identity in f32. Precision only |
| `prism.hadamard.version 2` / `tied_output` (`llama-model.cpp:2071-2232`) | `llama`, `qwen3*`, `qwen35*` | already refused by name: `hadamard_fold.rs` accepts version 1 only |
| `prec_policy` (W4A4, `general.tensor_extra.prec_a4`) | any | precision only |
| `LLM_FFN_OP_TYPES_FROM_STRING["gelu"]` is now the exact erf GELU | `modern-bert`, `dflash` (the only callers) | deferred / enum-only |
| `post_exp_norm` argument of `build_moe_ffn` | `muse-spark` only | refused |
| `attn_temp_pow` | `beam` only | refused |
| `LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY` | `lfm3` d1 only | refused |
| `rope.section_order` key, M-RoPE input order | MiniCPM-V 4.7 | text positions unchanged |
| `build_inp_embd(tok_embd, tok_scale)` | the four Gemma graphs | same arithmetic on token input: `sqrt(n_embd)` on token rows, 1 on embedding rows |
| no-cache attention padded to 256 KV rows | encoders | padding masked to `-inf`, identity |
| `RANK` pooling takes the LAST token for any causal model, not `qwen3` alone (`llama-graph.cpp:368-371`) | causal rerankers | a serving behaviour to check against `/v1/rerank`, not an architecture row; recorded, not changed here |
| `nextn_flags()` (#30097) shared by 17 graphs | MTP blocks | tensor requiredness only |
| `weights_sum + 1e-20` in place of the F16 clamp, `deepseek41` only (`llama-graph.cpp:2353-2358`) | `deepseek41` | refused |

### 2.2 `mellum` now READS `expert_weights_norm` -- fixed

```
$ grep -n 'EXPERT_WEIGHTS_NORM\|EXPERT_GATING' new/src/models/mellum.cpp
8:    hparams.expert_weights_norm = true;      (seeded)
10:   ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, ...,  false);
11:   ml.get_key(LLM_KV_EXPERT_GATING_FUNC,  ...,  false);
```

At `5b59b83`, `mellum.cpp:180` passed the literal `true` and
`LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`; at `604cc430` it passes both
hparams (`:204-206`). frink's loader reads `expert_weights_norm` only
for the architectures in `EXPERT_WEIGHTS_NORM_READERS`, and `mellum` was
not one, so a Mellum file declaring `false` would have been renormalised
here and not upstream. **`mellum` is in the table now.** The gating key
needed nothing: frink already read it with SOFTMAX as the default, which
is exactly what the new seed is.

### 2.3 Nemotron gained a pre-tokenizer frink did not have -- refused by name

`0693dbfb` (#23343) added a BPE tokenizer to `nemotron`, an architecture
frink serves, under `tokenizer.ggml.pre = domyn-small`
(`llama-vocab.cpp:2653-2656`, SPM-style BPE over raw UTF-8). frink's
`pretokenize::regex_for` sends an unknown `pre` to the GPT-2 pattern,
where llama.cpp throws `unknown pre-tokenizer type` (`:2760`), so that
file would have tokenized fluently and wrongly.

```
$ diff <(grep -oE 'tokenizer_pre == "[^"]+"' old/src/llama-vocab.cpp | sort -u) \
       <(grep -oE 'tokenizer_pre == "[^"]+"' new/src/llama-vocab.cpp | sort -u)
> domyn-small  > flint  > glm5  > k2-horizon  > mmbert  > nemotron-3.5-embed  > soprano
```

`glm5` joined `glm4` on the `CHATGLM4` arm, whose pattern
(`:414-417`) is frink's `LLAMA3` arm byte for byte, so it is SERVED now.
The other six each need a regex sequence or a raw-UTF-8 mode, and
`pretokenize::UNSERVED_PRE` refuses them by name at
`GgufBpeTokenizer::from_gguf`. The new `plamo3` tokenizer MODEL needed
nothing: an unknown `tokenizer.ggml.model` already refuses in both the
CLI and the server.

### 2.4 LFM2 gained a non-causal decision head -- refused by name

`lfm2.cpp` now reads `{arch}.decision.block_count`: the file's last
blocks are a biased-LayerNorm scorer over a BIDIRECTIONAL trunk that
emits three scores per token and no logits, with no KV memory
(`llama-model.cpp:2808`). `conversion/lfm2.py:258` writes the key for
every LFM2-d1-omni export. frink's decoder never reads
`attention.causal`, so only the unread-tensor gate stood between such a
file and a causal run of its head blocks.
`capability::unsupported_feature_keys` refuses the key by name now.
`decision.type` alone is not refused: the `lfm2-d1` / `openjev` kinds
read the ordinary LM logits of a causal model.

The causal LFM2 graph itself is a pure move: the short convolution and
the two FFN bodies went to `lfm-base.cpp` unchanged (`diff` of the
`graph<iswa>` bodies is the three call sites and nothing else).

### 2.5 `laguna` now reads the YaRN betas -- and frink read them everywhere

```
$ grep -ln YARN_BETA_FAST {old,new}/src/models/*.cpp
old: grok.cpp mistral3.cpp
new: grok.cpp mistral3.cpp laguna.cpp k2-horizon.cpp
```

`laguna.cpp:10-11` read `rope.scaling.yarn_beta_fast` / `_slow`, which
frink already honoured for Laguna, so for the row that moved nothing
changes. Measuring it found the opposite gap: frink read both keys for
EVERY architecture, while upstream reads them in those four graphs and
nowhere else, and `conversion/base.py:1653-1656` writes them for any HF
config whose YaRN block carries `beta_fast`. That is the
`expert_weights_scale` shape again (a key honoured where upstream
ignores it). It is reachable on real files: `conversion/mistral.py:33-37,
94,103-104,114,175` writes the two betas from a Mistral-format
checkpoint it emits as `llama` or as `deepseek2`, and neither graph reads them.
`loader::YARN_BETA_READERS` is the table now, with a test that a
`llama` / `qwen3` file declaring `8 / 2` keeps 32 / 1. The MLA engine's
YaRN goes through the same reader (`mla_yarn.rs`), so `deepseek2` /
`mistral4` keep the seeded pair too, as `deepseek2.cpp` does.

### 2.6 Behaviour that changed and needs nothing here

- **`nemotron` non-standard GQA** (#23343, `nemotron.cpp:27-33`): `wq`
  is now `{n_embd, n_head * head_dim}` rather than square. The generic
  path sizes Q from `n_head x head_dim` already (it is how Qwen3 loads),
  so such a file should load; no golden evidences it yet.
- **`nemotron_h` per-layer expert counts** (`nemotron-h.cpp:111-131`,
  read off the router tensor's shape): frink takes one model-wide
  `expert_count` and refuses a stored expert tensor of any other count
  (`loader.rs`, "MoE tensor ... does not match config n_experts"), so a
  Puzzle file with fewer experts on some layer stops loudly.
- **`mistral3`'s optional ada RMS norm** (Voxtral realtime,
  `mistral3.cpp:58-63,226-236`), **`qwen3`'s optional `dense_2_out`**
  and **`qwen35`'s optional `cls_out`** (embedding projections): each is
  decided by tensor presence, so the unread-tensor gate refuses a file
  that carries one.
- **`hrm_text`'s `z_l_init`** moved buffer placement only.
- **`mimo2`, `step35`, `cohere2moe`, `glm4moe`, `deepseek2`, `glm-dsa`,
  `qwen35*`, `qwen3next`, `openai-moe`, `nemotron_h*`**: the MTP flags
  moved into the shared `nextn_flags()` and the output-row cropping into
  `crop_before_nextn` / `crop_after_nextn`; neither changes a logit.

## 3. Every census, re-run

Each row is a census some frink table is derived from, run over both
trees with the pattern beside it. "Existing graphs" are graphs present
at `5b59b83`; a hit gained there is the silent-wrong class, and every
one is accounted for in section 2. The rest of each column is the new
graphs, all refused.

```
$ python3 census.py   # one grep -l per pattern over {old,new}/src/models/*.cpp
```

| census (table it feeds) | pattern | old -> new | existing graphs gained | action |
|---|---|---|---|---|
| NextN readers (`mtp_blocks::NEXTN_READERS`) | `n_layer_nextn\|NEXTN_PREDICT_LAYERS` | 17 -> 19 | `mellum`, `qwen4exp` | none: `mellum` creates its MTP block upstream now, frink still refuses a nonzero key on it (an over-refusal, the safe direction, recorded in the `mtp_blocks` module doc as before) |
| `EXPERT_WEIGHTS_SCALE_READERS` | `LLM_KV_EXPERT_WEIGHTS_SCALE` | 24 -> 29 | none | none |
| `EXPERT_WEIGHTS_NORM_READERS` | `LLM_KV_EXPERT_WEIGHTS_NORM` | 22 -> 29 | `mellum` | **row added** (2.2) |
| gating key readers | `LLM_KV_EXPERT_GATING_FUNC` | 24 -> 31 | `mellum` | none (2.2) |
| `GATING_LITERAL_ARCHITECTURES` | call sites of `build_moe_ffn(` by gating argument | 69 -> 80 sites; SIGMOID 4 -> 4, SOFTMAX 33 -> 33, hparam 32 -> 43 | `mellum` SOFTMAX -> hparam | none |
| `rope_layers::ROPE_PATTERN_READERS` | `LLM_KV_ATTENTION_ROPE_PATTERN` | 1 -> 2 | none | none (`flint`, refused) |
| `rope_layers` (is_swa gates the rotation) | `if (is_swa(il))\|use_rope\|has_rope(il)` | 11 -> 15 | none | none |
| `swa_layers` (window array readers) | `LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN` | 9 -> 15 | none | none |
| `YARN_BETA_READERS` (new) | `YARN_BETA_FAST` | 2 -> 4 | `laguna` | **table added** (2.5) |
| `yarn_magnitude` | `rope_yarn_log_mul\|YARN_LOG_MUL` | 4 -> 4 | none | none |
| `attn_temperature` | `attn_temp\|temperature_scale\|build_inp_attn_scale` | 4 -> 5 | none | none (`beam`) |
| `attn_gate` | `LLM_TENSOR_ATTN_GATE` | 15 -> 19 | none | none |
| sinks | `ATTN_SINKS` | 6 -> 8 | none | none |
| `norm::POST_NORM_EPS_LITERAL` | `post_norm_eps\|POST_NORM_EPS` | 1 -> 2 | none | none: `muse-spark` reads it as a KEY (`:5`), a second form for whoever closes it |
| `norm_sites::WEIGHTLESS_EMBEDDING_NORM` / `NON_PARAMETRIC_*` | `build_norm(x, nullptr\|NULL, nullptr\|NULL` | 4 -> 5 | none | none |
| sandwich norms | `ATTN_POST_NORM` / `FFN_POST_NORM` | 20 -> 26 / 14 -> 21 | `dflash` | none (enum-only) |
| `attn_value_scale` | `ATTENTION_VALUE_SCALE` | 1 -> 2 | `dflash` | none (enum-only) |
| `act_layers` clamp arrays | `LLM_KV_SWIGLU_CLAMP` | 6 -> 7 | none | none (`glm5-next`) |
| clamp form (`frink_moe::ClampForm`, `llama-graph.cpp` branches) | `ggml_swiglu_clamp` arch list | `maple`, `deepseek4`, `hy_v4`, `dflash` -> + `deepseek41`, `glm5-next` | none | none |
| `alibi` | `f_max_alibi_bias` | 8 -> 8 | none | none |
| `position_embd` | `POS_EMBD` | 4 -> 4 | none | none |
| `mamba1` / `mamba2` | `build_mamba_layer(` / `build_mamba2_layer(` | 3 -> 3 / 5 -> 5 | none | none |
| `shortconv` | `shortconv` | 2 -> 4 | none | none (`lfm-base`, `lfm3`) |
| `gdn` | `build_layer_attn_linear` | 4 -> 5 | none | none (`clef`) |
| `mrope::MROPE_READERS` | `ROPE_DIMENSION_SECTIONS` | 13 -> 14 | `bailingmoe3` | none (dedicated) |
| `mla` | `ATTN_KV_A_MQA` | 10 -> 11 | none | none (`glm5-next`) |
| `moe_interleave` | `n_moe_layer_step` | 3 -> 4 | none | none (`maion-coder`) |
| latent MoE refusal | `MOE_LATENT_SIZE` | 1 -> 3 | none | none (`maion-coder`, `muse-spark`) |
| `proj_bias` | `ATTN_OUT, "bias"` / `FFN_UP, "bias"` | 34 -> 37 / 27 -> 31 | `lfm2`, `modern-bert`, `dflash` | `lfm2`'s are in the decision head, refused (2.4) |
| LayerNorm users | `LLM_NORM,` | 30 -> 34 | `lfm2` | the decision head, refused |
| ungated / gated ReLU | `LLM_FFN_RELU[^_]` | 2 -> 4 | `lfm2`, `modern-bert` | the decision head, refused |
| exact GELU | `LLM_FFN_GELU_ERF` | 0 -> 2 | `mistral3`, `modern-bert` | `mistral3`'s is the ada norm, refused by its tensors |
| `layer_loops`, `sub_norms`, `router_input` (`FFN_NORM_EXPS`), `parallel_residual`, `chunked_swa`, `xielu`, output bias, ungated GELU, `1/d` scale, residual scale | as in the module docs | unchanged | none | none |

Two numbers the module docs carried at 155 are now 169 graphs; the
module text that says "one graph of 155" was a measurement at
`5b59b83` and stays dated as such. Where a census grew, it grew by new
graphs only, except for the rows in section 2.

## 4. Ranked, against the north star

1. **Done in this PR**: the three served-row fixes (`mellum`'s norm key,
   the `domyn-small` / five other pre-tokenizers, LFM2's decision head),
   the YaRN-beta reader table, `glm5` on the `LLAMA3` arm, and thirteen
   catalog rows.
2. **`kolibri1` and `lfm3`**, the two ONE MATCH ARM rows. `lfm3` closed
   the same day on exactly its rows plus four refusals: one post norm
   without the other and no sliding layer at all (both throw upstream),
   `attention.causal = false` (the d1 graph), and a conv layer marked
   sliding (ignored upstream, but here it would evict the conv history). `kolibri1` did not: its selection adds
   the bias to the RAW logits (`kolibri1.cpp:138`), which no frink
   router does, so it is NEW CODE, one selection operand in
   `frink_moe`. It closed on 2026-10-29 on exactly that
   (`frink_moe::SelectionRule::BiasBeforeGating`), as the first new
   file in `crate::models` (`models/kolibri1.rs`).
3. **The pre-tokenizer sequence.** `flint`, `soprano`, and the older
   `deepseek-*`, `superbpe`, `minerva` / `mellum2`, `tiny_aya`, `poro`,
   `viking` arms all need `regex_for` to return a SEQUENCE of patterns.
   One change closes the `flint` row and corrects three DeepSeek arms
   that `pretokenize.rs` already says are wrong.
4. **The latent MoE**, now three graphs (`nemotron-h` Super,
   `maion-coder`, `muse-spark`): one seam with three callers, measured
   before it is built.
5. **`RANK` pooling on causal rerankers** (2.1): check `/v1/rerank`
   against the reference's new "last token for any causal model" rule.
6. The rest of the new code: `beam`'s scaled residual and power-form
   temperature, `k2-horizon`'s grouped norm and value experts,
   `muse-spark`'s norm-preserving residual, then the two dedicated
   engines.
