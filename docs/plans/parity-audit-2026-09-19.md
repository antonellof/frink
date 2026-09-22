# Parity audit, 2026-09-19: architectures and serving features

What this is: a re-measurement of the two engines frink is read
against, done on the day 0.25.0 shipped, so the next work is chosen
from numbers rather than from the last audit's memory.

Method, and the reason for it: every count below came from a command,
and the command is printed beside it. `docs/plans/llama-cpp-gap-
inventory.md` (2026-09-01) was written the same way and one of its own
rows turned out to be wrong; a count nobody can re-run is a claim, not
a measurement.

## Update, same day: four of the twelve new rows closed

`spark2_5`, `maple`, `granite_swa` and `muse-glimmer` are audited
against libllama built from the moved pin (`tests/
gated_attention_graphs.rs`, `no_rope_layer_graphs.rs`,
`granite_swa_graphs.rs`, `muse_glimmer_graphs.rs`). Three of the four
cost a table row and a fixture, which is what their verdicts said they
would; the fourth, `maple`, cost one more thing its own graph file does
not show (`llama-graph.cpp:2228` sends it to `ggml_swiglu_clamp`, which
clamps the gate BEFORE the SiLU).

The seams that landed with them, each a census of one or two graphs:
`RopeLayers::FileMask` (the first upstream graph that lets the FILE say
which layers rotate), `frink_moe::ClampForm`,
`norm_sites::WEIGHTLESS_EMBEDDING_NORM`,
`norm::POST_NORM_EPS_LITERAL`.

Two defects came out of the same work, both in the class this repo
calls silent-wrong: `expert_used_count` and
`expert_feed_forward_length` are read with `get_key_or_arr` upstream
for EVERY architecture, and frink read each as a scalar -- so the
array spelling that `conversion/nemotron.py` writes for Nemotron-H
Puzzle, an architecture frink serves, fell into a default (top-2, and
`feed_forward_length / n_experts_used`). Both are honoured when uniform
and refused by name when they vary.

**What is left of the twelve**: `graniteswitch` (a per-token adapter
selection), `hrm_text` (two streams and a cycle schedule -- `zH`/`zL`
recombined at every stack boundary, `hrm-text.cpp:183-196`),
`minimax-01` (lightning attention as a recurrent block), `bailingmoe3`
/ `kimi-k3` (MLA + KDA hybrids), `dots3note` / `hy_v4` (DSA and
hyper-connections), `qwen4exp` (delta-net over a hybrid memory index),
and the two TTS rows. Each needs a block that does not exist here yet,
which is the honest reason the cheap ones went first.

## Update 2: the encoder family, and a correction to this document

Two more rows closed the same day, and they are the cheapest of the
eight because they are not on the decoder at all: `nomic-bert`
(nomic-embed-text v1 / v1.5) and `jina-bert-v3` (jina-embeddings-v3)
embed, on the BERT encoder frink has had since before this audit.

**This document's section 1.2 was imprecise about them.** It counted
eleven "deferred encoder/embedding" rows as gaps; `bert` was already
SERVED for embeddings with its own libllama parity test, and the
catalog row says so in as many words -- it is deferred from the
DECODER path, which is a different claim. Reading the catalog's own
comment rather than its scope column is what found that, which is the
same lesson `pangu-embedded` taught on 2026-09-14.

What the two rows cost: one line each of
`bert_gguf_loader::ENCODER_ARCHS`, plus `BertFfn` and
`BertHparams::rope_theta` for the two facts that differ across the
family, plus a fixture each. `bert.cpp`'s graph serves several
architectures and the ones frink builds differ from `bert` in two
lines of it and nothing else:

| arch | rotation | FFN |
|---|---|---|
| `bert` | learned position table | ungated GELU, both biases |
| `nomic-bert` | NEOX RoPE on Q/K | gated SiLU, no biases |
| `jina-bert-v3` | NEOX RoPE on Q/K | ungated GELU, both biases |

`jina-bert-v3`'s refusal had named "RoPE and per-projection QK norm",
and the second half does not exist: its own tensor loader creates no
`attn_q_norm`, so that branch of the shared graph is dead for it. A
verdict read from a graph's BRANCHES rather than from the
architecture's own loader names blockers it does not have.

And one open question, recorded rather than answered: on an F32
fixture frink and libllama agree EXACTLY with the attention output
switched off and differ by ~3e-4 with it on. That is not the Q8_0
activation-quantization story `tests/bert_llama_cpp_parity.rs` tells
about the real checkpoint, and two obvious explanations are measured
and eliminated (a uniform softmax still differs; f16 K/V does not move
it). `tests/bert_family_graphs.rs` carries the bisection.

**What is left in the family**: `jina-bert-v2` (ALiBi, an optional
whole-projection QK LayerNorm, a second attention norm and a fused
gate+up), `neo-bert` (its own graph), `modern-bert` (alternating
local/global attention), `eurobert` (its own graph), `nomic-bert-moe`
(a second FFN shape on its MoE layers), `t5encoder`, and the two
decoder-embedding rows. `jina-bert-v2` is the next cheapest and needs
ALiBi on the encoder's attention, which `frink_core::alibi` already
computes for the decoder.

## 0. The one-line answer

Against the llama.cpp this repo PINS, frink serves **every text
generation architecture that has a graph**. Against llama.cpp's
`master` as of today it serves 116 of 130, because the pin is six
weeks and **792 commits** stale and fourteen architectures landed in
that window. On the serving side the architecture question is the wrong one
-- the overlap is high and the naming spaces differ -- and the gap is
in SERVING features, where one of them is already half-built in this
tree.

## 1. llama.cpp

### 1.1 The pin is stale, and that is now the headline

```
$ cd .scratch/llama.cpp && git log -1 --format=%ci      # 2026-08-04
$ git rev-list --count HEAD..origin/master              # 792
$ git ls-tree -r --name-only origin/master src/models | wc -l   # 156
```

The pinned tree has 140 graphs and `master` has 155 plus `clip.cpp`.
Fifteen new files, fourteen of them new architectures:

```
$ git show origin/master:src/llama-arch.cpp | grep -o 'LLM_ARCH_[A-Z0-9_]*,\s*"[^"]*"' \
    | sed 's/.*"\(.*\)"/\1/' | sort > /tmp/up.txt      # 153 names
$ git show HEAD:src/llama-arch.cpp | ... > /tmp/pin.txt  # 139 names
$ comm -13 /tmp/pin.txt /tmp/up.txt
```

| arch | graph | lines | what it is, and what frink would need |
|---|---|---|---|
| `spark2_5` | `spark2-5.cpp` | 146 | **CLOSED 2026-09-19**: one `attn_gate` row (sigmoid, per head) and three tables that each gained a name |
| `maple` | `maple.cpp` | 150 | **CLOSED 2026-09-19**: one `rope_layers` row (`SlidingOnly`) plus `ClampForm::BeforeSilu`, which `maple.cpp` does not show -- `llama-graph.cpp:2228` decides it |
| `granite_swa` | `granite-swa.cpp` | 319 | Granite's four multipliers (served) on an iSWA pattern array (served) with an optional per-layer `expert_used_count` ARRAY. Closest to a fixture-away row of the fourteen |
| `graniteswitch` | `granite-switch.cpp` | 427 | the same multipliers plus an `adapter_ids` argument threaded through the layer: a per-token expert-adapter selection with no counterpart here |
| `muse-glimmer` | `muse-glimmer.cpp` | 203 | window + `logit_scale` + final logit softcap (all served) with an attention GATE (`attn_gate`, served since `afmoe`) |
| `hrm_text` | `hrm-text.cpp` | 213 | two transformer stacks alternating over one token stream under H/L cycle counts. `crate::layer_loops` (nanbeige) is the same IDEA -- weights replayed, KV logical -- with a different schedule |
| `dots3note` | `dots3note.cpp` | 476 | DSA indexer + absorbed MLA (the `glm-dsa` engine's shape) with step35's head-wise output gate |
| `hy_v4` | `hy-v4.cpp` | 601 | independent hyper-connections: several residual streams reduced and redistributed per layer, plus a DSA cache |
| `minimax-01` | `minimax-01.cpp` | 484 | lightning attention as a RECURRENT layer under `attention.recurrent_layers` / `full_attention_interval` -- the hybrid seam `qwen35` built, with a different block |
| `bailingmoe3` | `bailingmoe3.cpp` | 540 | MLA + KDA (Kimi delta attention) hybrid with a conv kernel and a safe-gate flag |
| `kimi-k3` | `kimi-k3.cpp` | 618 | KDA + MLA hybrid, cross-layer residual attention, latent MoE, "situ" activation, an MLA output gate |
| `qwen4exp` | `qwen4exp.cpp` | 1297 | the largest new graph: hybrid memory index, delta-net, MoE |
| `pockettts` | `pockettts.cpp` | 146 | TTS. Out of scope until an audio scope exists |
| `qwen3tts` | `qwen3tts.cpp` | 3 | TTS shim |

Twelve of the fourteen are text generation. Two of those twelve
(`granite_swa`, `maple`) read only keys and ops frink already serves,
which is the cheapest class this repo has -- and the lesson from
`minimax-m2` is that a row in that class costs a fixture and an hour,
so it should not sit in a table for a week.

Three (`minimax-01`, `bailingmoe3`, `kimi-k3`) are hybrid recurrent
rows on the seam `granitehybrid` / `qwen35` built, which is the seam
that has closed nine rows in two weeks.

### 1.2 Against the pin, the text-generation gap is zero

```
$ ./target/release/frink archs | awk -F'|' 'NF>4{print $6}' | sort | uniq -c
   95 generic-gqa   21 dedicated   31 deferred   3 test-fixture
$ ... | awk -F'|' 'NF>4{print $3}' | sort | uniq -c
  119 TextGeneration  11 DeferredEncoderEmbedding  10 DeferredMultimodal
    5 EnumOnly         4 DeferredDiffusion          1 DeferredAudio
```

Every deferred row is an encoder/embedding, a multimodal, a diffusion
or an audio model, or an `EnumOnly` name llama.cpp itself has no graph
for (`gptj`, `eagle3`, `dflash`, `clip`, `(unknown)`). **No text
generation architecture of the pinned llama.cpp is refused.**

So the four remaining llama.cpp scopes are, in the order their user
population justifies:

1. **encoder / embedding (11)**: `bert`, `nomic-bert`, `nomic-bert-moe`,
   `jina-bert-v2`, `jina-bert-v3`, `modern-bert`, `neo-bert`,
   `eurobert`, `gemma-embedding`, `llama-embed`, `t5encoder`. frink
   already serves `/v1/embeddings` and `/v1/rerank` -- from a decoder.
   These are the models people actually embed with, and `bert` alone
   is most of that population.
2. **multimodal (10)**, which needs an image encoder and a projector
   before any of the ten matters.
3. **diffusion text (4)** and **audio (1)**.

### 1.3 What this means for the pin

Updating the pin is not a chore here, it is the measurement: every
capability table in `frink-models` is derived from a census over
`src/models/*.cpp`, and a census over a six-week-old tree can be
wrong in the direction that matters (a graph that started reading a
key). The pin bump and the census re-run belong in one PR, before any
of the fourteen rows.

## 2. Serving features

### 2.1 A model count is not the comparison

A GGUF architecture string is not a Hugging Face `*ForCausalLM` class
name, and several of the latter map onto one of the former
(`LlamaForCausalLM`, `MistralForCausalLM` and `YiForCausalLM` are all
`llama` in GGUF, which this repo learned the hard way on 2026-09-10).
Counting one against the other compares two different things. The
scopes frink actually defers are three: multimodal,
pooling/embedding models, and encoder-decoder.

### 2.2 Feature parity, measured against this tree

The serving features an OpenAI-compatible engine is expected to have,
each checked against frink by grep rather than by memory:

| Serving feature | frink | evidence |
|---|---|---|
| chunked prefill (CP) | **yes** | `frink-server/src/generate.rs` |
| automatic prefix caching (APC) | **yes** | `policy/radix`, over paged KV |
| LoRA, per request | **yes** | `frink-server/src/lora.rs`, with a reader/writer gate llama.cpp does not have |
| speculative decoding (SD) | **yes, since 0.28.0** | verified by agreement with the server's own sampler, so lossless at any temperature; `frink-server/src/sampling_loop.rs` |
| structured outputs | **yes** | `grammar_request.rs`, `json_mode.rs`, `tool_grammar/`, `frink_models::grammar` |
| tool calling | **yes** | and 0.25.0 fixed the format being chosen by the served NAME |
| reasoning outputs | **yes** | `reasoning_tokens.rs`, `reasoning_budget.rs` |
| pooling / embeddings | **partial** | `/v1/embeddings`, `/v1/rerank` from a decoder; no BERT-family encoder |
| logprobs / top_logprobs | **yes** on both OpenAI routes | the sampler publishes the distribution it drew from |
| prompt logprobs | **yes** on `/v1/completions` | plain softmax, not the sampler's chain |
| `n` > 1 / best-of | **yes** on both OpenAI routes, one shared prefill | beam search stays refused |
| prompt embeds as input | **no, refused by name** | was a silent 200 until 2026-09-22 |
| encoder-decoder | **no** | `t5` and friends are deferred |
| multimodal | **no** | 10 deferred rows |
| CUDA graph capture | **no** | and on Metal the equivalent -- one encoded graph per token -- is exactly what 0.25.0's submission collapsing approximates by hand |
| tensor / pipeline parallel | **no** | single process, single device |
| disaggregated prefill, KV connectors, KV offload | **no** | `docs/plans/out-of-core-moe.md` is the nearest thing and is groundwork |
| sleep mode | **no** | |
| per-request metrics | **yes** | `stats/` |
| quantized KV cache | **Metal only** | `--ctk` reaches `frink-metal`'s device store; the host cache is f32 (see 2.4) |
| per-caller cache isolation (`cache_salt`) | **yes on the contiguous caches**, refused by name on the paged store | see 2.5 |

### 2.3 The finding worth acting on

**Speculative decoding is built and unreachable from the server**, and
it is this repo's dominant bug shape wearing a different hat: two
structures that must agree, with nothing enforcing it. The evidence is
not an opinion --

```
$ grep -rn 'speculat' crates/frink-server/src --include=*.rs -l
crates/frink-server/src/stats/requests.rs
$ grep -rn 'with_speculation' crates/frink-server/src | grep -v test
(nothing)
```

-- the server has an acceptance-rate metric, a test that the metric
reaches the admin ring, and NO producer for it. A metrics column that
no code path can fill reads as coverage, which is the same defect
class as a gate that cannot fire.

`frink_models::speculative` is lossless by construction (the
Leviathan / Chen rejection rule, with `accept_or_resample` pinned by
tests), `PromptLookupSpeculator` needs no second checkpoint and no
GPU, and `draft_model.rs` already exists for the `--model-draft` case.
So both the n-gram and the draft-model arms are one wiring job away
over the API, and a multi-token-prediction arm is one `Drafter` impl
away for the seventeen architectures whose MTP blocks frink already
SKIPS by name (`crate::mtp_blocks::NEXTN_READERS`).

### 2.4 The host KV store is f32 where llama.cpp's is f16

Found on 2026-09-22 from a real deployment, not from reading. A
CPU-only host running Ternary-Bonsai-2-27B refused a 6415-token prompt
at a derived ceiling of 4096.

The arithmetic is honest -- `kv_elem_for` prices the CPU backend as
`KvElem::F32` and the store really is `Vec<f32>`
(`frink-core/src/cache.rs:138`) -- but the WIDTH is twice the
reference's. `llama_context_default_params` sets `type_k` and `type_v`
to `GGML_TYPE_F16` (`llama-context.cpp:3672-3673`), so llama.cpp holds
2 bytes per element where frink holds 4.

For that checkpoint the KV is 524288 bytes/token (64 layers x 2 x 4
kv-heads x 256 head-dim x f32), so the 4096 ceiling is exactly 2.0 GiB
of KV. At llama.cpp's default width the same budget buys 8192.

`--ctk` does not help: `run.rs:815` selects
`KvElem::from_ctk` for `BudgetBackend::Metal` only, and pins CPU and
CUDA to `F32`. So the flag parses, is documented as llama.cpp's
`-ctk`, and cannot change the host store. It is not a misprice -- the
budget agrees with the store -- but it is a flag that does not do what
its name says on two of the three backends.

**This is a llama.cpp parity gap of the largest practical kind**: it
halves the context of every CPU deployment against the reference, and
it ranks above every row in section 2.2 that no llama.cpp user can ask
for.

### 2.5 `cache_salt` names an isolation property, not a knob

**Served since 2026-09-22**, and building it confirmed the reading:
the field needed BOTH shared caches scoped, not one, and a partial
overlap across salts had to be no match rather than a shorter one. The
paged store is refused by name, because its radix tree has no
namespace to scope a lookup to. The original note is kept below
because it is what made the row the right shape.

Refused by name since 2026-09-22 along with the rest of the
unimplemented surface -- except it is not in that table, deliberately.
The field selects which cached prefixes a request may reuse. A server
that IGNORES it can serve one caller from another caller's cached
prefix, and frink's radix cache is keyed by token ids and shared
across requests (`policy::radix`). So the honest statement is not
"frink lacks a knob" but "frink offers no per-caller cache isolation",
which is a design row rather than a field to wire.

## 3. Ranked, against the north star

The north star is "the Rust alternative to llama.cpp: same models,
same command shapes, same or better performance". A serving feature no
llama.cpp user can ask for ranks below a llama.cpp gap of the same
size.

0. **An f16 host KV store** (2.4). Found 2026-09-22 from a real CPU
   deployment. llama.cpp's default KV width is f16 and frink's host
   cache is f32, so every CPU deployment gets HALF the context of the
   reference on the same box, and `--ctk` cannot reach that store. It
   goes above everything below it because it is a llama.cpp gap that
   a llama.cpp user hits on their first long prompt.

1. **Update the llama.cpp pin and re-run every census.** Everything
   below is measured against a tree that is 792 commits old, and two
   of this repo's tables are derived from a grep over it.
2. **Speculative decoding in the server.** Built, tested, lossless,
   unreachable; the metric for it already exists. llama.cpp's server
   has it over HTTP and frink does not.
3. ~~**`granite_swa` and `maple`**~~ -- `maple` closed on 2026-09-19
   with `spark2_5`; `granite_swa` is left and needs two small per-layer
   tables (an `expert_used_count` array, which the loader now reads,
   and `attention.rope_pattern`, the first upstream graph that lets the
   FILE decide which layers rotate).
4. **`bert` and the encoder/embedding family.** Eleven llama.cpp rows
   and the whole pooling scope in one seam, and frink already has the
   two routes that would serve them.
5. **`minimax-01`**, the new hybrid recurrent row, on the seam that has
   closed nine rows in two weeks.
6. The MLA/DSA and hyper-connection rows (`dots3note`, `hy_v4`,
   `bailingmoe3`, `kimi-k3`, `qwen4exp`), each of which needs a block
   that does not exist here yet.
7. `n` > 1 / best-of / prompt logprobs -- small, API-shaped, and
   nobody has asked.

Multimodal is deliberately below all of these: it is an image encoder,
a projector and a preprocessing pipeline, and it would be the largest
single thing in the tree.
