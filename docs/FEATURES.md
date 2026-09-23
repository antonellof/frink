# Features

Frink is a pure-Rust GGUF inference engine for dense and MoE models.
Weights stay quantized when the file is mmapped, and dequantization
happens inside the matvec. Backends: CPU, Apple Metal, and CUDA.

## Models

**99 architectures run** with a benchmark row, a pinned logit
comparison against real `libllama`, or a fixture behind each. Four more
stop with an error that names what is missing. `frink archs` prints the
current list; [`MODELS.md`](MODELS.md) says what runs, what refuses and
why; [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) has the speed
against llama.cpp on the same host and file.

Where the speed stands, re-measured on a quiet M2 Pro: **all 16
comparable Metal `tg128` rows are faster than llama.cpp**, 0.60x
(SmolLM2-135M, Qwen2.5-0.5B) to 0.96x (OLMoE), with dense Metal
prefill 0.99x to 1.09x. **CPU is behind on every row**: prefill 6.3x
to 10.1x, decode 1.06x to 1.92x on x86.

What the decoder can express, rather than which checkpoint closed when:

| Shape | Examples |
|---|---|
| Dense GQA, MoE | Llama 3.x, Qwen2.5/3, Gemma-2/3, Mistral, Phi-4-mini, SmolLM2, OLMoE, Mixtral |
| MLA, with and without the absorption optimization | DeepSeek-V2/V3, Mistral-Large-3, PLM |
| Dedicated engines | Gemma-4 (per-layer embeddings, shared KV), GLM-5.2, Kimi, DeepSeek-V4 |
| Sliding-window attention: per layer, per-layer arrays, chunked | Gemma-2/3, gpt-oss, Llama 4, Mistral, Cohere2 |
| Recurrent blocks at the attention site | Mamba-1 and Mamba-2 (Jamba, FalconMamba, Granite 4.0, Nemotron-H, Falcon-H1), short convolutions (LFM2), gated delta nets (Qwen3.5, Qwen3-Next), lightning attention (MiniMax-01), PLaMo-2 |
| Position encodings other than NEOX RoPE | NORM RoPE, partial rotary, two rotary widths, per-layer RoPE gates, YaRN and LongRoPE, M-RoPE on text positions, learned absolute tables (GPT-2, StarCoder), ALiBi (BLOOM, MPT, Refact, Jais, Baichuan-13B), no rotation at all |
| Norms | RMS and LayerNorm, each with and without weights and biases, pre/post/both, per-head and per-head-distinct QK norms, norms inside the sublayers (BitNet), weightless embedding norms |
| Residual topologies | Sequential, parallel (`x + attn(norm x) + ffn(norm x)`), normed-input (MiniMax-01), skip streams, the same physical layers run more than once (Nanbeige) |
| FFN activations | SwiGLU and its clamped forms, GeGLU, ReGLU, ungated GELU and ReLU-squared, xIELU, per-layer activation parameters |
| MoE routing | Softmax and sigmoid, router biases, expert-weight scales and norms, shared experts, a dense FFN summed with the routed output (Arctic, Grok-2), routing on the raw layer input (SmallThinker), the weight on the expert's input (Llama 4), interleaved experts |
| Attention extras | Sinks, gates (sigmoid and softplus), per-position temperature, logit softcaps, split K/V head widths, MTP blocks skipped inside `block_count` |
| Encoders | BERT, nomic-bert, jina-bert-v3 for `/v1/embeddings`; a cross-encoder rank head for `/v1/rerank` and `/v1/score` |
| Quantization beyond llama.cpp's | PrismML `PTQ1_0` with a folded Hadamard rotation, verified on the real Ternary-Bonsai-2-27B checkpoint |

Backend limits worth knowing before you pick a model: **gpt-oss and
Llama 4 are CPU only** (no Metal kernel for attention sinks; Llama 4's
window is chunked and the fused Metal launches take one window per
layer), and **no recurrent model supports prefix-cache reuse or
`--model-draft`**, because a recurrent state cannot be rolled back to a
middle position.


## Backends

| Backend | Capabilities |
|---|---|
| **CPU** | Dense and MoE. int8×int8 matvec on by default (`FRINK_CPU_INT_DOT=0` opts out), interleaved Q4_Kx8 / Q8_0x4 GEMV, Q8_0x4 batch GEMM for prefill, Q5/Q6 int-dot, pool sized to performance cores |
| **Metal** | FA-vec attention (decode d=64/96/128/256, prefill d=128/256), concurrent FFN/QKV encode, MoE Concurrent with fused groups, `MemRanges`, `mul_mm_id` prefill, quantized KV (`q8_0` / `fp8` / `q4_0`, the last with a Hadamard rotation on K) |
| **CUDA** | Matvec, resident weights, FFN fuse (`--features cuda`), batched GEMMs for `Q8_0`, `Q4_0`, `Q5_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q2_K`, `Q3_K`, `IQ4_NL`, `IQ4_XS` and `MXFP4` (verified on an RTX 3090, 2026-09-15), and a resident dense prefill stack (norms, QKV bias, QK norm, RoPE, causal GQA, SwiGLU, residuals on the device; K/V rows back to the host cache) |
| **Vulkan** | `Q8_0` matvec only, no GEMM (`--features vulkan`). A beachhead, not a backend: see below |

**Vulkan is one kernel, and calling it a backend would be generous.**
`--features vulkan` gives a `Q8_0` matvec and nothing else. It reports
no GEMM for any kind, so a prefill genuinely lands on the host, and it
claims no other quantization. It did run on real hardware (an M2 Pro
through MoltenVK) against a scalar twin, which is what earned it a place
in the dispatch table at all, but there is no measured number for it and
zero-copy residency from mmap is unproven. It exists so that AMD and
Intel have a path at all, and because the seam it needed is the seam a
real backend needs. `docs/plans/vulkan-beachhead-verdict.md` has the
sizing: a full Vulkan backend is 15 to 25k lines.

**The CUDA GEMM has run, and it is now the limit.** Its hardware test
passes every kind and shape on an RTX 3090 and `frink verify
--backend cuda` is token-identical to the CPU on Q4_K_M, Q5_K_M, Q6_K,
Q8_0 and IQ4_XS checkpoints (2026-09-15). It also keeps the
thread-by-thread scalar twin held against `frink-quant`'s independent
dequantize-then-GEMM, and the host harness that executes the emitted
CUDA C against a barrier shim
(`crates/frink-cuda/tools/mul_mm_host_check/run.sh`), zero mismatches
across **11 kinds, 33 shapes and 75,042 compared positions**. Below
the width threshold a single token stays on the matvec kernels.

**CUDA prefill is resident and on the tensor cores, and still 4x
off.** A dense layer used to be seven synchronous round trips with
everything else on the host: 3.1 GB over PCIe per Llama-3.2-3B pp512
step, two thirds of the step by `nsys`. `frink_cuda::prefill` runs a
run of dense layers on the device with one upload and one download of
the hidden batch (#259), and `mul_mm_tc` puts the GEMM on `mma.sync`
with f16 operands on `sm_80` and up (#261): pp512 on that model went
305 to 1932 tok/s on an RTX 3090 against llama.cpp's ~8,200. What is
left is a host third of the step between launches, the GEMM's
remaining distance to int8 `mmq`, and a K/V-tiled attention kernel.
Decode is 2.2x to 5.0x behind on the same cards (#133). So a Windows or Linux
install runs, answers correctly, and should not be chosen for speed
yet. `/health` reports the same thing per capability, with a reason
string, instead of quietly greying a control out.

## Quantization

Parsed and executable on CPU: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`,
`Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`,
`IQ4_NL`, `IQ4_XS`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S`, `MXFP4`, `TQ1_0`, `PTQ1_0`.

"Executable" is not one speed. What a format gets, read off the kernel
tables rather than from intent:

| Tier | Formats | CPU SIMD | GPU |
|---|---|---|---|
| Full | `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` | AVX2 + NEON, plus the int-dot path | Metal matvec + GEMM, CUDA matvec + GEMM |
| Metal only | `IQ4_XS`, `Q5_0` | AVX2 + NEON | Metal matvec + GEMM |
| Metal only, scalar CPU | `PTQ1_0` (PrismML ternary) | scalar | Metal matvec + GEMM |
| CPU-vectorized | `Q4_1`, `Q5_1`, `Q8_1`, `Q2_K`, `Q3_K`, `IQ4_NL`, safetensors two-buffer `MXFP4` | AVX2 + NEON | none |
| AVX2 only | `IQ1_S`, `IQ2_XXS`, `IQ3_XXS` | AVX2; **scalar on ARM** | none |
| Scalar only | `IQ2_XS`, `IQ2_S`, `IQ3_S`, `IQ1_M`, GGUF-block `MXFP4` | none | none |

Metal's MoE indexed GEMM (`mul_mm_id`) is narrower still: `Q4_0`,
`Q8_0` and `Q4_K` only.

Three things worth knowing before choosing a quant:

- **The bottom two tiers are correct and slow, on purpose.** They were
  added for coverage: before them those tags could not be decoded at
  all, which ruled out 5 of the 16 published Unsloth `UD-*` variants. A
  vectorized path was left out rather than written without a golden
  vector that could tell it apart from the scalar one.
- **On an Apple machine the "AVX2 only" row IS the scalar row.**
  `IQ1_S`, `IQ2_XXS` and `IQ3_XXS` have x86 kernels and no NEON ones.
- **`I32`, `TQ2_0`, `NVFP4`, `Q1_0`, `Q2_0` and `PQ2_0` are recognized
  and sized, and nothing executes them.** They parse, `frink inspect`
  reports their real footprint, and a checkpoint needing one stops with
  an error naming the format rather than being silently mis-measured.

`IQ2_XS`, `IQ2_S`, `IQ3_S` and `IQ1_M` were validated bit-exact against
llama.cpp's own `dequantize_row_*` by linking `ggml-quants.c`, not by
re-reading the spec. They have not been validated end to end on a
published `UD-*` checkpoint.

## CLI

llama.cpp-style completion flags (`-m`, `-p`, `-n`, `-ngl`, `--ctk`, …),
plus `frink chat`, `frink pull` (Hugging Face Hub), `inspect`, `archs`,
and `presets`. See [`CLI.md`](CLI.md).

Constrained decoding is on the CLI too: `--grammar`, `--grammar-file`
and `-j` / `--json-schema`, the same spellings llama.cpp uses, reaching
the same stack machine the HTTP `grammar` field does. `--ctk` selects a
KV dtype on Metal only; the CPU and CUDA KV cache is the host `Vec<f32>`
and the startup banner says so when the flag is being ignored.

`--lora` / `--lora-scaled` load a LoRA adapter GGUF
(`convert_lora_to_gguf.py`'s format) on the CLI and the server, applied
inside every projection it names as llama.cpp's `build_lora_mm` applies
it and checked against libllama with the same adapter (KL under 5e-13 on
the fixture, 5.2e-4 on Llama-3.2-1B Q8_0 with a rank-8 adapter). The
server lists and rescales them through `GET`/`POST /lora-adapters` and
the per-request `lora` field. Routed-expert adapters, activated LoRAs
and the dedicated engines refuse by name; on Metal an adapted model runs
on the per-matrix path rather than the fused stacks. See
[`CLI.md`](CLI.md#lora-adapters) and [`API.md`](API.md#lora-adapters).

`frink perplexity` is the quality axis: corpus evaluation using
llama.cpp's method, agreeing with `llama-perplexity` to within a fifth
of one standard error on five checkpoints. Where the two differ, the gap
is monotone in the quant and has the sign the documented `vec_dot_type`
difference predicts. `frink quantize` writes **`Q8_0`, `Q4_K_S`,
`Q4_K_M`, `Q5_K_S`, `Q5_K_M` and `Q6_K` byte-identically** to
`llama_model_quantize()`, with or without an importance matrix
(`--imatrix`, 311 of 311 tensors identical to `llama-quantize
--imatrix` on a BF16 Qwen3-0.6B for all five targets tried), and
refuses every other target by name. `frink imatrix` is
`llama-imatrix`: same file format in both directions, so either tool's
matrix feeds either quantizer.
**The claim that Q4_K could never be byte-identical was wrong**, and it
was wrong for an instructive reason: llama.cpp's `sumlx += w*x[i]*l`
is contracted by its compiler into a single fused multiply-add, and
Rust does not contract, so a strict transcription of the C was the
defect. One unit in the last place flips a comparison and rewrites a
whole super-block. With the fusion spelled out as `mul_add`, Q4_K went
from 1.15% of super-blocks differing to zero, across all 147 tensors of
a real model. See [`CLI.md`](CLI.md).

The sampler flags carry llama.cpp's own defaults on `--temp` (0.8),
`--top-k` (40), `--top-p` (0.95), `--min-p` (0.05) and `--repeat-last-n`
(64). **One default still differs on purpose**: `--repeat-penalty` is
1.1 here and 1.0 (off) in llama.cpp
(`common/common.h:239`), so a run left entirely to defaults is not
token-identical. `-e`/`--escape` is on by default as it is there, and a
*partial* `-ngl N` is refused rather than silently offloading every
layer.

That non-default default has a **Metal decode cost**, and it is
deliberate. At `--temp 0` the Metal stack can fold
`final_norm + lm_head + argmax` into its own command buffer and hand
back one token id instead of `vocab_size` floats. A device argmax over
raw logits cannot apply the penalties, so the fold is now refused
whenever any of `--repeat-penalty`, `--presence-penalty` or
`--frequency-penalty` is live -- which, at `--repeat-penalty 1.1`, is
every plain greedy run. It used to fire anyway and return a token the
host sampler would not have chosen (GitHub issue #170).
`--repeat-penalty 1.0` or `--repeat-last-n 0` gets the fold back and
is also what makes a run token-identical to llama.cpp's defaults.

`frink bench -m model.gguf` works like `llama-bench`: the same `pp512`
and `tg128` workloads, reported as a median with a population stddev.
Add `--compare` to run `llama-bench` alongside it and print the gap.
`--suite` drives every entry in
[`benchmarks/suite.json`](../benchmarks/suite.json) and regenerates
[`RESULTS.md`](../benchmarks/RESULTS.md). See
[`benchmarks/README.md`](../benchmarks/README.md).

## Server

Two ways to start the same server. `frink serve` is a subcommand of the
main binary behind an optional `serve` feature, off by default for
`cargo install` because it pulls in 98 crates a completion-only user
does not need. `frink-server` is that same server as its own
executable, and both parse identical arguments through identical code.
The prebuilt release binary is built with `serve`, so the downloaded
`frink` does both.

OpenAI-compatible HTTP API:

- Chat completions (SSE, optionally resumable: `id:` + `retry:` +
  `Last-Event-ID` replay and a JSON polling fallback), completions,
  tokenize / detokenize
- `POST /v1/cancel` stops a running generation by request id, which is
  the stop path a resumable stream needs, since closing its socket no
  longer ends it
- Embeddings. A real encoder checkpoint (BGE / E5 / GTE class, anything
  whose `tokenizer.ggml.model` is `bert`) can be the loaded model:
  `FRINK_MODEL_PATH=bge-small-en-v1.5-q8_0.gguf` serves `/v1/embeddings`
  and the six generating routes answer **501 naming the model**, not a
  missing tensor. Pooling comes from the checkpoint's own
  `pooling_type` (NONE / MEAN / CLS / LAST; RANK refuses, it is a
  classification head rather than a pooling rule, and a rank-head
  checkpoint belongs on `/v1/rerank` instead). A decoder GGUF still
  pools its hidden states (mean/last) as before, and
  `FRINK_EMBEDDING_MODEL_PATH` runs an encoder side-by-side with a
  generative model in one process
- Reranking. A `bert` checkpoint carrying a rank head (`cls`,
  `cls.output`, `cls.norm`, `classifier.output_labels`) is served by
  `POST /v1/rerank`, scoring `[CLS] query [SEP] document [SEP]` through
  the head itself rather than through the cosine of two embeddings. Such
  a checkpoint could not load at all before: `assert_every_tensor_
  consumed` rejected the `cls.*` tensors nobody read. Verified end to
  end against `ms-marco-MiniLM-L6-v2`: the order matches a HuggingFace
  reference, which needed the document to be segment 1 rather than
  llama.cpp's all-zero token types. The SCALE matches too, once the
  pooler is back: llama.cpp's converter deletes `bert.pooler.dense`
  from every BERT reranker, so the converter's output scores on about
  plus or minus 0.2 where the checkpoint was trained to produce about
  plus or minus 11 (#82). `frink splice-pooler` writes a GGUF that
  carries the pooler, tied to the checkpoint by the classifier both
  files hold rather than by a name; on the spliced file every score
  is within 0.051 of HuggingFace across four query sets. A file
  without the pooler still serves, with `frink_score_head:
  classifier(cls)` on every response, so a client can tell which range
  it is reading
- Anthropic Messages: `POST /v1/messages` streaming and buffered
  (thinking and tool blocks, protocol-native `ping` keepalive) plus
  `POST /v1/messages/count_tokens`
- `POST /v1/responses`, the surface `codex` speaks, streaming and
  buffered. This server keeps no responses, so the two lookups by
  response id answer 404
- Sampling matched to llama.cpp's own chain, all nine steps of it in
  upstream's order: `penalties`, `dry`, `top_n_sigma`, `top_k`,
  `typ_p`, `top_p`, `min_p`, `xtc`, `temperature`
  (`common/common.h:259-269`). Temperature runs **last**, after the
  truncation filters, and the repetition penalty is applied once per
  candidate rather than once per occurrence. DRY's sequence breakers are
  tokenised against the loaded model's own vocabulary; a checkpoint with
  none refuses DRY rather than running it breaker-less. `mirostat` and
  `infill` are the two upstream samplers still refused, each by name.
  Every route reads the same knobs through one `SamplingKnobs::resolve`
- **Several completions per request.** `n` and `best_of` prefill the
  prompt ONCE and fork the KV per choice, so `prompt_tokens` counts it
  once while `completion_tokens` sums. Choice `i` samples from
  `seed + i`, so choice 0 of four is byte-identical to the single
  answer at the same seed. With `stream`, the choices are INTERLEAVED
  a token at a time, each chunk carrying its own `choices[].index` and
  its own reasoning and tool-call parser state. On the paged store the
  fork is copy-on-write: full pages are shared and only the
  part-written tail is copied
- **Per-token steering.** `logit_bias` (upstream's `-100..100`,
  outside it a 400 rather than a clamp), `allowed_token_ids` and
  `bad_words`, all applied in the same mask the grammar, JSON mode and
  the reasoning budget share. A bias cannot lift a token a constraint
  forbade: a bias is finite and a mask is `-inf`
- **Logprobs.** `logprobs` on completions (parallel arrays with
  `text_offset`) and on chat (OpenAI's `content[]` shape), over the
  distribution the sampler actually drew from, with candidates the
  chain REMOVED omitted rather than reported as `null`.
  `prompt_logprobs` scores the prompt from the PLAIN softmax, because
  a prompt token was supplied rather than drawn
- **Prompt controls.** `echo` returns the prompt and the completion as
  one string with the logprobs arrays covering both;
  `truncate_prompt_tokens` keeps the last `k` tokens, applied before
  anything is prefilled so the KV, the usage and `echo` all see the
  prompt that was ANSWERED; `cache_salt` names a caller's prefix-cache
  namespace, honoured by the contiguous cache, the response cache and
  the paged store's radix tree
- **Sleep and wake** (`POST /sleep`, `POST /wake_up`,
  `GET /is_sleeping`): an unload that REMEMBERS, so the server can
  bring the model back itself. Frees the KV pool, the paged store, the
  repack and expert caches and any device buffers. One level, not two:
  frink mmaps its weights, so discarding them is what dropping the
  handle already does
- **Sentence-pair scoring** (`POST /v1/score`, `POST /score`): a
  cross-encoder answers with its head, a bi-encoder with the cosine of
  two embeddings, and `frink_score_kind` says which. `/v1/rerank`
  requires the head and refuses to substitute a cosine, because a
  rerank promises the model's own ranking
- Grammar-constrained decoding, in every spelling: llama.cpp's own
  `grammar` field, OpenAI's `response_format: json_schema`, llama.cpp's
  bare `json_schema` field on `/completion`, and a forced `tool_choice`.
  A schema is compiled to GBNF first, so all of them end at one stack
  machine that masks every token which cannot continue a valid string,
  on chat and completions and all three decode paths. Two constraints in
  one request are refused rather than ranked. `response_format:
  json_object` is still the best-effort character mask, and composes
- **Parallel serving (Metal).** Multiple concurrent streaming clients
  share one batched decode worker (llama.cpp slots model). Continuous
  batching is on by default when compatible; streaming emits tokens
  incrementally under CB (0.15.2). Metal CB prefill keeps host K/V
  authoritative for batched decode (0.15.3). Host B receipt on
  Llama-3.2-3B Q4_K_M: 16/16 OK at concurrency 8, ~24 aggregate tok/s,
  ~118 ms mean TTFT sequential. CLI: `-cb`, `-np` / `--parallel N`,
  `--no-cont-batching` for the serialized private path. See
  [`plans/metal-parallel-concurrency.md`](plans/metal-parallel-concurrency.md)
- Chunked prefill (same scheduler as continuous batching). `-b` /
  `-ub` set the chunk on both decode paths from one number
- **Slot save/restore** (llama.cpp's `POST /slots/{id}?action=save|restore`):
  a prompt prefix's KV written to `--slot-save-path` and restored into
  the prefix cache after a restart, so a long system prompt is prefilled
  once per file rather than once per process. The file carries a
  checkpoint fingerprint, and a restore under another model or another
  quantisation is refused by name. See [`API.md`](API.md#slot-save-and-restore)
- Paged KV: shared page storage many requests read through a block
  table, with a radix tree over reference-counted page groups so
  conversations off one system prompt share its KV rather than each
  holding a copy. Off unless `FRINK_PAGED_KV_BLOCKS` is set. It used to
  be refused on a GPU backend, where it returned fluent wrong tokens; the
  cause was a Metal prefill leaving K/V on the device and filling the
  host cache with placeholders that the paged prefill then copied into
  the page store, and that refusal is lifted. See
  [`CONFIG.md`](CONFIG.md)
- **On the paged store**, a model whose layers *all* slide by the same
  window slides during decode, so a request holds its prompt and a
  window rather than its whole context , and admission prices it
  that way, so a store too small for the whole context still serves
  it. A tool call anchors the slide at the position the next agentic
  turn will rejoin at, and the anchor is dropped once the cursor
  drifts a window past it. An alternating-SWA model (gpt-oss,
  Gemma-3) does not slide *there*: a page group holds one block in
  every layer, and the full-attention layers still read position 0
- **On the contiguous host store**, eviction is PER LAYER, so the
  alternating models do get it: each layer's `KvCache` carries its own
  window from `attention.sliding_window` /
  `attention.sliding_window_pattern` and drops the rows behind it, while
  the full-attention layers keep everything. Off unless
  `FRINK_KV_WINDOW` is set, and it turns itself off under Metal
  attention, on a draft model, and beside the prefix cache. Output is
  token-identical with it on or off, asserted on logits as well as token
  ids against gemma-2-2b-it-Q4_K_M. `--ctx-size auto` and the pre-load
  admission check are priced against the same per-layer residency the
  stores evict with, so the saving is context a user is actually offered
  rather than memory nothing spends. See [`CONFIG.md`](CONFIG.md)
- `frink serve-bench`: concurrency, TTFT, TPOT and queueing numbers
  for a live server, with the methodology (positional split, pooled
  nearest-rank percentiles, whole-run throughput) tested socket-free.
  Host B receipts for Metal CB at 0.15.3:
  [`benchmarks/receipts/serving/`](../benchmarks/receipts/serving/)
- Live serving telemetry (`GET /v1/stats`, `GET /v1/requests`) and an
  elastic KV/expert split that can be reported and re-sized without a
  restart (`GET /v1/cache/status`, `POST /v1/cache/rebuild`). A request
  that arrives mid-rebuild is turned away with an error rather than
  parked in a queue behind it
- `reasoning_content`: a reasoning model's chain of thought is split
  out of `content`, streamed as it arrives rather than at the end
- Tool calls in eleven wire formats, not one: the format the served
  checkpoint's family emits, then the prompt-engineered one, and every
  call in a response rather than the first. Five of the eleven stream
  their arguments as deltas
- Prompts rendered by *evaluating* the checkpoint's own
  `tokenizer.chat_template`, with `chat_template_kwargs` and
  `reasoning_effort` passed through (the effort quantized onto what that
  checkpoint's template really grades)

## Edge-native MoE serving: what is real here

FreeToken describes an edge-native MoE serving engine, and frink ports
its host-side policy (Apache-2.0, see
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)). That policy now lives in
the crates that use it rather than in a crate of its own: the expert
residency stack in `frink-core` beside `expert_store`, and the serving
policy in `frink-server::policy`.

This table is what frink actually does against that description,
checked against the code rather than asserted. The gap is the roadmap.

| Capability | In frink today |
|---|---|
| Bandwidth-adaptive CPU/GPU co-execution (`q*`) | **Partial.** `qstar::BandwidthProfile` is in `frink-core` and used by `frink bench-bw`, a measurement tool. The serving path does not consult it. |
| Full-layer double-buffered prefill streaming | **Built, not wired.** Kept for the out-of-core work, which names it. |
| Global LRU expert caching | **Yes, and now singular.** `expert_store` is wired into both decode paths and proven bit-identical to resident at a 1-byte budget. The second, competing cache and its separate byte budget were folded in beside it. |
| Graph-compatible execution | **No.** Execution is eager. `ExecutionPlan` is built and read by nothing. |
| FTW fast weight format | **No.** GGUF only. |
| Semantic anchor checkpoints for KV | **Yes.** `anchor::decode_slide` and `WindowPolicy` are wired into `generate.rs` and the batch scheduler. |
| Agentic context edits without recompute | **Partial, and currently leaking.** The radix prefix cache shares pages and reports `cached_tokens`, but `RadixCache::evict` has no caller, so the page pool shrinks until admission refuses. |
| Elastic VRAM re-allocation without restart | **Partial.** `POST /v1/cache/rebuild` re-splits KV pool geometry at runtime. Moving bytes between an expert cache and KV is not implemented. |
| MXFP4 / BF16 | **Yes**, executable. MXFP4 is CPU-only. |
| NVFP4 / FP8 | **No.** Neither is parsed. |
| DeepSeek-V4-Flash, GLM-5.2, Kimi K3 | **Loaders and primitives only.** Nothing has run end to end on a real checkpoint. |
| OpenAI + Anthropic compatible APIs | **Yes**, both, plus Responses. Tool calls parsed in eleven wire formats. |
| NVIDIA RTX 30/40/50 | **Runs, measured, behind.** Receipts on a GTX 1080, an RTX 3060 and an RTX 3090; correct by `frink verify`; prefill about 4x and decode 2x to 5x off llama.cpp. No GPU in CI. |

Two honest notes. Frink runs on Apple Metal, which that description
does not cover, and Metal is where it is fastest: every `pp512` row is
0.99x to 1.09x against llama.cpp and **every one of the 16** comparable
`tg128` rows is faster. And the single largest gap is not on this table:
running a model that does not fit in memory works as policy and not as
execution.

## Serving policy

A Rust port of the host-side decision logic in
[FreeToken](https://github.com/FlashML-org/FreeToken)
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)): the parts of an
edge-native MoE engine that *decide* rather than compute. Tensor-free
and testable without a GPU. Each module takes measured numbers and
returns a decision.

It lives in the crates that use it: the serving half in
`frink-server::policy`, the MoE expert-residency half in `frink-core`
beside `expert_store`, which is the single holder of the expert byte
budget.

### Driving something today

| Module | Decides | Where it runs |
|---|---|---|
| `parser` | where reasoning ends and the answer begins, and which tool was called in which format | `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, streaming and buffered |
| `detokenize` | what text is safe to stream after one more token | the stop-string withhold rule, which `frink-server`'s `StopMatcher` delegates to so there is one implementation |
| `radix` | which prefix of a new prompt is already computed, page-keyed and node-sharing | the paged-KV serving path, where it shares KV pages between prompts by reference count |
| `anchor` | how far a window may slide, and where a tool call pins it so the next agentic turn rejoins rather than recomputes | the paged-KV serving path, on both the private generate loop and the continuous batcher |
| `scheduler` | admission, chunked-prefill sizing, and what a chunk reserves | the continuous batcher's status and pool accounting |
| `effort` | which reasoning-effort dialect a checkpoint speaks | probed once per checkpoint at load, then applied to every request's `chat_template_kwargs`, and advertised on `/v1/models` |
| `serving_stats` | what a server may honestly claim about its own throughput and latency | `/v1/stats`, `/v1/requests`, `/admin/stats` |
| `maintenance` | whether a request, a cache rebuild or a stop may proceed right now | `POST /v1/cache/rebuild` and `POST /v1/admin/prepare-stop` |
| `pool` | how VRAM splits between the expert cache and KV, and how it is re-split live | the target geometry `POST /v1/cache/rebuild` validates against |
| `rebuild` · `outbox` · `footprint` | whether a re-split rolls back, what a stop receipt is worth, what this process really occupies | the same two admin endpoints |
| `deepseek_v4_budget` | per-layer KV tier sizing, and which compressor each layer runs (none / CSA / HCA) | the DeepSeek-V4 decoder |
| `bench_profile` · `bench_client` | when a measured bandwidth profile may be trusted, and what a serving benchmark may report | `frink bench-bw` and `frink serve-bench` |

### Complete, tested, and waiting for a consumer

`qstar` (the `q*` bandwidth split), `expert_cache`, `expert_slots`,
`expert_budget`, `placement` and `residency`, all in `frink-core`.
Each is covered by unit tests and none of them is on a serving path.
Do not read a benchmark as evidence for any of them.
[`plans/out-of-core-moe.md`](plans/out-of-core-moe.md) is what they are
waiting on: running a model larger than memory, which is the single
largest thing they would buy.

`expert_slots` sits closest to real memory: it executes the expert
cache's copy plans against a bounded slot pool, and a warm decode step
copies zero bytes on a host pool. `frink-core`'s `CudaExpertPool`
implements its `SlotDevice` trait under `--features cuda`, and that pool
is compile-verified with its hardware test left `#[ignore]`d, so on a
real card the property is written down and not yet measured. A host
`SlotDevice` (`HostSlotMemory`) also exists; a Metal one does not, and
that is the concrete gap.

Inside `frink-server::policy`, the modules carrying an unwired half
name the roadmap item that would close it, at their declaration in
`policy/mod.rs`. `grep -n "allow(dead_code)" crates/frink-server/src/policy/mod.rs`
is the list of what still owes a caller.

Frink Studio, the web UI in [`ui/`](../ui), is a separate app that
talks to this API over HTTP. `frink-server` does not serve it, and
`GET /` on it is a 404.

See [`API.md`](API.md) and [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).
