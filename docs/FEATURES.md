# Features

Frink is a pure-Rust GGUF inference engine for dense and MoE models.
Weights stay quantized in the mmapped file and are dequantized inside
the matvec. Backends: CPU, Apple Metal and CUDA, plus an opt-in Vulkan
backend.

## Models

Every architecture in `capability::AUDITED_GENERIC_GQA` runs with a
benchmark row, a pinned logit comparison against `libllama`, or a
fixture behind it (the README states the count, which a test pins).
Architectures read but not audited stop with an error naming what is
missing. `frink archs` prints the list; [`MODELS.md`](MODELS.md) says
what runs, what refuses and why.

Speed against llama.cpp is in
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md), one row per host,
backend and checkpoint. Most rows are marked stale (measured on an
older build). On them Metal is level on prefill and ahead on decode,
CPU is behind on prefill, and CUDA is behind on both.

| Shape | Examples |
|---|---|
| Dense GQA, MoE | Llama 3.x, Qwen2.5/3, Gemma-2/3, Mistral, Phi-4-mini, SmolLM2, OLMoE, Mixtral |
| MLA, with and without the absorption optimization | DeepSeek-V2/V3, Mistral-Large-3, PLM |
| Dedicated engines | Gemma-4 (per-layer embeddings, shared KV) |
| Speculative drafts | An autoregressive draft GGUF, prompt lookup, and DFlash / DFlash2 block drafts (llama.cpp's `dflash` architecture) |
| Sliding-window attention: per layer, per-layer arrays, chunked | Gemma-2/3, gpt-oss, Llama 4, Mistral, Cohere2 |
| Recurrent blocks at the attention site | Mamba-1 and Mamba-2 (Jamba, FalconMamba, Granite 4.0, Nemotron-H, Falcon-H1), short convolutions (LFM2), gated delta nets (Qwen3.5, Qwen3-Next), lightning attention (MiniMax-01), PLaMo-2 |
| Position encodings other than NEOX RoPE | NORM RoPE, partial rotary, two rotary widths, per-layer RoPE gates, YaRN and LongRoPE, M-RoPE on text positions, learned absolute tables (GPT-2, StarCoder), ALiBi (BLOOM, MPT, Refact, Jais, Baichuan-13B), no rotation at all |
| Norms | RMS and LayerNorm, each with and without weights and biases, pre/post/both, per-head and per-head-distinct QK norms, norms inside the sublayers (BitNet), weightless embedding norms |
| Residual topologies | Sequential, parallel (`x + attn(norm x) + ffn(norm x)`), normed-input (MiniMax-01), skip streams, the same physical layers run more than once (Nanbeige) |
| FFN activations | SwiGLU and its clamped forms, GeGLU, ReGLU, ungated GELU and ReLU-squared, xIELU, per-layer activation parameters |
| MoE routing | Softmax and sigmoid, router biases, expert-weight scales and norms, shared experts, a dense FFN summed with the routed output (Arctic, Grok-2), routing on the raw layer input (SmallThinker), the weight on the expert's input (Llama 4), interleaved experts |
| Attention extras | Sinks, gates (sigmoid and softplus), per-position temperature, logit softcaps, split K/V head widths, MTP blocks skipped inside `block_count` |
| Encoders | BERT, nomic-bert, jina-bert-v2 (ALiBi), jina-bert-v3, NeoBERT and EuroBERT (pre-norm RMS) for `/v1/embeddings`; a cross-encoder rank head for `/v1/rerank` and `/v1/score` |
| Quantization beyond llama.cpp's | PrismML `PTQ1_0` with a folded Hadamard rotation, verified on the real Ternary-Bonsai-2-27B checkpoint |

GLM-5.2, Kimi and DeepSeek-V4 have loaders and primitives but have not
run end to end on a real checkpoint, so they are not in the table.

Limits that affect model choice:

- On Metal, gpt-oss and Llama 4 run on the host path (no Metal kernel
  for attention sinks; Llama 4's chunked window does not fit the fused
  launches). CUDA's resident stack runs gpt-oss.
- A recurrent model gets no prefix-cache reuse, since its state cannot
  be truncated mid-sequence. It can be a speculation target (state
  checkpointed before each verify, replayed after a rejection) on the
  CLI and the server's private loop, but not in the continuous batch or
  on the paged store, and it cannot be a draft.

## Backends

| Backend | Capabilities |
|---|---|
| **CPU** | Dense and MoE. An integer dot against quantized activations for every weight kind (ggml's `vec_dot_type`, ADR 0009), with AVX2 and NEON bodies; on by default where it measured faster (aarch64), `FRINK_CPU_INT_DOT` overrides. Interleaved Q4_Kx8 / Q8_0x4 GEMV, a Q8_0x4 batch GEMM for prefill, and a thread pool sized to the performance cores |
| **Metal** | FA-vec attention (decode d=64/96/128/256, prefill d=128/256), concurrent FFN/QKV encode, fused MoE groups, `mul_mm_id` prefill, quantized KV (`q8_0` / `fp8` / `q4_0`, the last with a Hadamard rotation on K). A server decodes on the private loop, with attention on the device, unless continuous batching is turned on |
| **CUDA** | Resident weights and a resident stack for prefill and decode (`--features cuda`): norms, QKV bias, QK norm, RoPE, attention, SwiGLU or GeGLU and residuals on the device, K/V rows back to the host cache through page-locked memory. Prefill multiplies `Q8_0` and `Q4_K` on the int8 tensor cores and the other kinds on the f16 ones, and a prefill batch's attention runs on the tensor cores at head widths 64, 128 and 256 (`sm_80` and up). Decode matvecs for `Q4_K`, `Q6_K`, `Q8_0` and `MXFP4` dot a q8_1 activation with `dp4a`, as llama.cpp does. Routed MoE layers route on the device for plain softmax and gpt-oss routing, and a prefill batch's experts run as one grouped GEMM per projection. gpt-oss runs with its sinks, biases and clamped SwiGLU on the device. `FRINK_CUDA_GRAPH=1` replays a decode token from a CUDA graph; it is off by default and has not run on a GPU |
| **Vulkan** | Matvec, prefill GEMM and embedding gather for every weight kind, resident weights, and a decode and prefill stack (`--features vulkan`, opt-in with `--device vulkan` or `FRINK_VULKAN=1`). Correct, not yet faster than the CPU: see below |

[`KERNELS.md`](KERNELS.md) shows which quant kinds each backend runs per
op (matvec, GEMM, `get_rows`) and which fall back to the host. It is
generated by `frink caps --kernels` and a test fails when it drifts. A
fallback prints a `NO KERNEL` line at load, and `frink inspect-plan`
lists the tensors of one GGUF that would take one.

### Vulkan

GLSL shaders in `crates/frink-vulkan/shaders`, compiled SPIR-V
committed beside them. Each kind is tested against its `frink_quant`
scalar twin on a real device (M2 Pro through MoltenVK) and on lavapipe
in CI. Weights stay resident.

The stack runs a run of layers per submission (one per 26 layers on a
deeper model), for decode or prefill: pre-norm (RMS, LayerNorm,
weightless or biased), QKV bias, QK norm, RoPE, an f16 KV mirror,
attention with or without a sliding window, `wo`, post-norm, SwiGLU or
GeGLU, and residuals. K and V rows go back to the host cache, which
stays the authority. A layer it does not serve (attention sinks, a `wo`
bias, a chunked window, an attention softcap, RoPE `mscale`, routed
experts, or anything under `--ctk f32`) runs on the host with its
matvecs offloaded.

It is held to the CPU and llama.cpp's decode goldens on lavapipe
(`decode_step_graphs::vulkan_decode_matches_cpu_and_llama_cpp`), and
`frink parity` is MATCH on Llama-3.2-3B Q4_K_M at a 300-token prompt.
It stays opt-in until measured against the CPU on rented hardware (ADR
0006). It gives AMD and Intel GPUs a path, including ones ROCm does not
support (#409).

### CUDA

The GEMM's hardware test passes every kind and shape on an RTX 3090,
and `frink verify --backend cuda` is token-identical to the CPU on
Q4_K_M, Q5_K_M, Q6_K, Q8_0 and IQ4_XS checkpoints. On rented L40S, A10G
and RTX PRO 6000 cards every dense model checked was token-identical to
the CPU and at parity MATCH with llama.cpp; the tensor-core attention
and `dp4a` matvecs are token-identical to the CPU on an A100. The
`Q4_1`, `Q5_1`, `TQ1_0`, `PTQ1_0` and seven IQ1/IQ2/IQ3 kernels, the
`get_rows` gather and the CUDA graph replay have not yet run on
hardware (#405, #537).

CUDA is correct but behind llama.cpp on speed; the gap per card is in
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md). `/health` reports
each capability with a reason string.

## Quantization

| Status | Kinds |
|---|---|
| Parsed and executable | `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, `IQ4_XS`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `MXFP4`, `TQ1_0`, `PTQ1_0` |
| Recognized and sized, refused at execution by name | `I32`, `TQ2_0`, `NVFP4`, `Q1_0`, `Q2_0`, `PQ2_0` |

Every weight kind has a matvec and a GEMM kernel on CPU, Metal, CUDA
and Vulkan ([`KERNELS.md`](KERNELS.md)); `Q8_1` is ggml's activation
format and marked `n/a`. Some CUDA kernels have not run on a GPU (#405).

Metal's MoE id kernels: `mul_mv_id` (decode, and prefill below
llama.cpp's batch threshold) covers every kind with a Metal kernel but
`PTQ1_0`, whose experts run one matvec each. `mul_mm_id` (prefill)
covers `Q4_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`,
`IQ4_NL`, `IQ4_XS` and `MXFP4`; other expert kinds prefill through
`mul_mv_id`.

`IQ2_XS`, `IQ2_S`, `IQ3_S` and `IQ1_M` are bit-exact against
llama.cpp's `dequantize_row_*` (linked from `ggml-quants.c`), but not
yet validated end to end on a published `UD-*` checkpoint.

## CLI

llama.cpp-style flags (`-m`, `-p`, `-n`, `-ngl`, `--ctk`, …), plus
`frink chat`, `pull`, `inspect`, `inspect-plan`, `archs`, `presets`,
`gguf-split`, `batched-bench`, `verify` (a GPU backend against the CPU),
`parity` (frink against llama.cpp) and `layer-divergence`. Everything
below is detailed in [`CLI.md`](CLI.md).

| Feature | Notes |
|---|---|
| Constrained decoding | `--grammar`, `--grammar-file`, `-j` / `--json-schema`, the same stack machine as the HTTP `grammar` field |
| KV precision | Host KV is `f16` by default, `f32` under `--ctk f32`, as in llama.cpp. `--ctk` also selects Metal's quantized device KV; the startup banner names what each backend keeps |
| LoRA | `--lora` / `--lora-scaled` load a `convert_lora_to_gguf.py` adapter on the CLI and server, applied as llama.cpp's `build_lora_mm` (KL under 5e-13 on the fixture, 5.2e-4 on Llama-3.2-1B Q8_0 rank 8). The server rescales through `GET`/`POST /lora-adapters` and the `lora` field. Routed-expert adapters, activated LoRAs and the dedicated engines refuse; on Metal an adapted model takes the per-matrix path. See [`CLI.md`](CLI.md#lora-adapters), [`API.md`](API.md#lora-adapters) |
| Speculative decoding | `-md` takes an autoregressive draft of the same family or a DFlash / DFlash2 block draft (`--spec-type draft-dflash`, picked from the file), which reads the target's residual stream and proposes a block per pass. DFlash drafts and lattice scores match libllama on the fixtures (`tests/dflash_graphs.rs`), and greedy output is unchanged by it on CPU and Metal. The server takes `--model-draft` |
| `frink perplexity` | llama.cpp's method; within a fifth of one standard error of `llama-perplexity` on five checkpoints, with remaining gaps in the sign the `vec_dot_type` difference predicts |
| `frink quantize` | `Q8_0`, `Q4_K_S`, `Q4_K_M`, `Q5_K_S`, `Q5_K_M`, `Q6_K` byte-identical to `llama_model_quantize()`, with or without `--imatrix` (311 of 311 tensors identical on a BF16 Qwen3-0.6B for all five targets tried); other targets refuse |
| `frink imatrix` | `llama-imatrix`'s file format both ways, so either matrix feeds either quantizer |
| `frink bench` | llama-bench's `pp512` / `tg128`, mean and sample stddev; `--compare` runs `llama-bench` beside it; `--suite` runs [`suite.json`](../benchmarks/suite.json) and regenerates [`RESULTS.md`](../benchmarks/RESULTS.md). See [`benchmarks/README.md`](../benchmarks/README.md) |

Sampler defaults match llama.cpp on `--temp` (0.8), `--top-k` (40),
`--top-p` (0.95), `--min-p` (0.05) and `--repeat-last-n` (64), with one
deliberate difference: `--repeat-penalty` is 1.1 here and 1.0 (off) in
llama.cpp (`common/common.h:239`), so an all-defaults run is not
token-identical. `-e`/`--escape` is on by default, and a partial
`-ngl N` is refused rather than offloading every layer.

That penalty costs Metal decode: at `--temp 0` the stack can fold
`final_norm + lm_head + argmax` on the device and return one token id,
but not while any of `--repeat-penalty`, `--presence-penalty` or
`--frequency-penalty` is live. `--repeat-penalty 1.0` or
`--repeat-last-n 0` restores the fold and token identity with
llama.cpp's defaults.

## Server

`frink serve` (the `serve` feature, on by default and in the prebuilt;
`--no-default-features` leaves it out) and `frink-server` are the same
server, parsing identical arguments through identical code. Routes and
fields are in [`API.md`](API.md), switches in [`CONFIG.md`](CONFIG.md).

### APIs

| Surface | Notes |
|---|---|
| Chat completions, completions, tokenize / detokenize | SSE, optionally resumable (`id:` + `retry:` + `Last-Event-ID` replay, JSON polling fallback). `POST /v1/cancel` stops a generation by request id, since closing a resumable stream's socket does not |
| Anthropic Messages | `POST /v1/messages` streaming and buffered (thinking and tool blocks, `ping` keepalive), plus `/v1/messages/count_tokens` |
| Responses | `POST /v1/responses`, the surface `codex` speaks. No responses are stored, so the two lookups by id answer 404 |
| Embeddings | A `bert`-tokenizer encoder (BGE / E5 / GTE class) can be the loaded model; the six generating routes then answer 501 naming the model. Pooling follows the file's `pooling_type` (NONE / MEAN / CLS / LAST; RANK refuses and belongs on `/v1/rerank`). A decoder GGUF pools mean/last (`llama-embed` refuses by name until checked). `FRINK_EMBEDDING_MODEL_PATH` runs an encoder beside a generative model |
| Reranking | A `bert` checkpoint with a rank head (`cls`, `cls.output`, `cls.norm`, `classifier.output_labels`) serves `POST /v1/rerank` through the head, never a cosine. Order matches a HuggingFace reference on `ms-marco-MiniLM-L6-v2` (the document is segment 1). llama.cpp's converter drops `bert.pooler.dense`, so scores land near ±0.2 instead of ±11 (#82); `frink splice-pooler` restores it (every score within 0.051 of HuggingFace on four query sets). A file without the pooler still serves, tagged `frink_score_head: classifier(cls)` |
| Scoring | `POST /v1/score` and `/score`: a cross-encoder answers with its head, a bi-encoder with a cosine, and `frink_score_kind` says which |
| llama-server routes | `/props`, `GET /slots`, slot erase, `/infill`, `/apply-template`, `/metrics`, slot save/restore (`POST /slots/{id}?action=save\|restore` to `--slot-save-path`, fingerprinted so another model or quantisation is refused; see [`API.md`](API.md#slot-save-and-restore)) |
| Sleep and wake | `POST /sleep`, `POST /wake_up`, `GET /is_sleeping`: an unload that remembers the model, freeing the KV pool, paged store, repack and expert caches and device buffers. One level, since weights are mmapped |
| Telemetry and elastic cache | `GET /v1/stats`, `GET /v1/requests`, `GET /v1/cache/status`, `POST /v1/cache/rebuild`. A request that arrives mid-rebuild is turned away, not queued |

Frink Studio ([`ui/`](../ui)) is a separate app over HTTP;
`frink-server` does not serve it, and `GET /` is a 404.

### Sampling and output control

| Feature | Behaviour |
|---|---|
| Sampler chain | llama.cpp's nine steps in upstream order: `penalties`, `dry`, `top_n_sigma`, `top_k`, `typ_p`, `top_p`, `min_p`, `xtc`, `temperature` (`common/common.h:259-269`). The repetition penalty applies once per candidate. DRY's breakers are tokenised against the loaded vocabulary; none means DRY refuses. `mirostat` and `infill` refuse by name. One `SamplingKnobs::resolve` serves every route |
| `n`, `best_of` | The prompt is prefilled once and the KV forked per choice (copy-on-write on the paged store). Choice `i` uses `seed + i`, so choice 0 equals the single answer. Streamed choices interleave a token at a time, each with its own parser state |
| `logit_bias`, `allowed_token_ids`, `bad_words` | One mask shared with grammar, JSON mode and the reasoning budget. `logit_bias` outside `-100..100` is a 400. A finite bias cannot lift a `-inf` mask |
| Logprobs | On completions (parallel arrays with `text_offset`) and chat (`content[]`), over the distribution sampled from; removed candidates are omitted. `prompt_logprobs` uses the plain softmax. `return_tokens_as_token_ids` spells tokens by id |
| `skip_special_tokens: false` | Keeps the end-of-generation token in text and `usage`; a `stop_token_ids` entry is excluded as a stop string is |
| `echo`, `truncate_prompt_tokens`, `cache_salt` | `echo` returns prompt plus completion with logprobs over both. Truncation keeps the last `k` tokens before prefill. `cache_salt` namespaces the contiguous cache, the response cache and the radix tree |
| Grammar | llama.cpp's `grammar`, OpenAI's `response_format: json_schema`, the bare `json_schema` on `/completion`, and a forced `tool_choice` all compile to one GBNF stack machine, on chat and completions and all three decode paths. Two constraints in one request are refused. `json_object` stays a best-effort character mask |
| Reasoning | `reasoning_content` streams separately. `reasoning_budget_tokens` / `thinking_budget_tokens` (`thinking.budget_tokens`, `--reasoning-budget N`) force the closing tag after N tokens; matches llama.cpp token for token on DeepSeek-R1-Distill at temperature 0 |
| `continue_final_message` | A trailing assistant message is continued, its thought re-opened. On by default as in llama.cpp; `false` or `--no-prefill-assistant` turns it off |
| Tool calls | Eleven wire formats: the family's own, then the prompt-engineered one, every call in a response. Five stream arguments as deltas |
| Chat templates | The checkpoint's `tokenizer.chat_template` is evaluated, with `chat_template_kwargs` and `reasoning_effort` (quantized onto what that template grades) |

### Scheduling and KV

| Feature | Behaviour |
|---|---|
| Private loop | The default: one client at a time. On Metal it attends on the device and is the faster path for one client |
| Continuous batching | Opt-in on every backend (`-cb`, `-np` / `--parallel N`, `FRINK_CONTINUOUS_BATCHING=1`): clients share one batched decode worker, llama.cpp's slots model. Throughput per concurrency is in [`RESULTS.md`](../benchmarks/RESULTS.md#serving-features-measured) |
| Chunked prefill | `-b` / `-ub` set the chunk on both paths. Each tick has one token budget for prefill and decode (`FRINK_CB_MAX_BATCH_TOKENS`, default 256) with a 16-token prefill floor. Latency effect not yet measured |
| Cache-aware admission | With a radix cache, a waiting job whose prompt is computed may go first. Only the first eight waiters are considered and none is passed over more than four times; ties keep arrival order, and a picked job that does not fit stops the line |
| Prefix cache | `FRINK_PREFIX_CACHE_ENTRIES`, private loop: LRU KV snapshots, so a follow-up turn prefills only the new message. Stores greedy Metal requests too; refuses a recurrent model's state |
| Paged KV | `FRINK_PAGED_KV_BLOCKS`: shared pages through block tables, with a radix tree over reference-counted page groups so conversations sharing a system prompt share its KV. A model whose layers all slide by one window slides here, and admission prices prompt plus window; a tool call anchors the slide where the next turn rejoins. Alternating-SWA models (gpt-oss, Gemma-3) do not slide on this store |
| Per-layer eviction | On the contiguous host store, on by default: each layer keeps `window` rows plus slack from `attention.sliding_window` / `sliding_window_pattern` or the per-layer array (Gemma-3-4B at 32k keeps under a fifth of its KV). Speculation, the prefix cache and `n > 1` forks still work. `FRINK_KV_WINDOW=0` keeps every row; off under `FRINK_METAL_ATTN` and on a draft model; the paged store and KV pool do not evict. Token-identical on or off. `--ctx-size auto` and admission price the same residency |
| Speculative decoding | Prompt lookup or the `-md` draft (autoregressive or DFlash), verified in one batched forward; a token is kept only if it equals the sampler's draw. Runs on the private loop, the paged store and the continuous batch. Grammar, `json_object` and paged leases under a sliding window do not speculate. Recurrent targets speculate on the private loop only, with state checkpointed and replayed. DFlash is refused beside the prefix cache and paged store. `usage` reports the fields only when it ran |
| `frink serve-bench` | Concurrency, TTFT, TPOT and queueing for a live server; `--shared-prefix` reports reuse from `cached_tokens` (`--cache-salt` isolates the run). Receipts: [`benchmarks/receipts/serving/`](../benchmarks/receipts/serving/) |

## Edge-native MoE serving: what is real here

Frink ports FreeToken's host-side policy (Apache-2.0, see
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)): the expert residency
stack lives in `frink-core` beside `expert_store`, the serving policy in
`frink-server::policy`. The table compares FreeToken's description with
what frink does; the gaps are the roadmap. Its structural claims are
pinned by `crates/frink-server/tests/documented_capabilities.rs`.

| Capability | In frink today |
|---|---|
| Bandwidth-adaptive CPU/GPU co-execution (`q*`) | **Partial.** `qstar::BandwidthProfile` is in `frink-core` and used by `frink bench-bw`, a measurement tool. The serving path does not consult it. |
| Full-layer double-buffered prefill streaming | **Built, not wired.** Kept for the out-of-core work, which names it. |
| Global LRU expert caching | **Yes, singular, and ranked by routing counts.** `expert_store` is wired into both decode paths; for a GGUF model its entries live in the slots of a `SlotDevice` (host RAM, or on Metal one Shared buffer the GPU reads in place), eviction ranks by the decoder's activation counts, and decode is proven identical to resident at half, a quarter and one slot of the expert bytes. |
| Graph-compatible execution | **Off by default.** Execution is eager. `FRINK_CUDA_GRAPH=1` replays a CUDA decode token of the resident stack from a captured graph; it has not run on a GPU. `ExecutionPlan` is built and read by nothing. |
| FTW fast weight format | **No.** GGUF only. |
| Semantic anchor checkpoints for KV | **Yes.** `anchor::decode_slide` and `WindowPolicy` are wired into `generate.rs` and the batch scheduler. |
| Agentic context edits without recompute | **Partial.** The radix prefix cache shares pages and reports `cached_tokens` on the private and batched paths. `acquire_paged_caches` calls `evict` when a request cannot find pages, freeing what no live lease has locked, and retries at once. A changed middle still forces a recompute from that point, because the cache is a prefix tree. |
| Elastic VRAM re-allocation without restart | **Partial.** `POST /v1/cache/rebuild` re-splits KV pool geometry at runtime. Moving bytes between an expert cache and KV is not implemented. |
| MXFP4 / BF16 | **Yes**, executable. GGUF-block MXFP4 has Metal and CUDA matvec and GEMM kernels; the safetensors two-buffer form is CPU-only. |
| NVFP4 / FP8 | **Refused by name, not executed.** NVFP4 is recognized and sized by `frink-gguf` (tag 40) and has no execution path. FP8 is only a KV cache wire (`--ctk fp8`), not a weight format. |
| DeepSeek-V4-Flash, GLM-5.2, Kimi K3 | **Loaders and primitives only.** Nothing has run end to end on a real checkpoint. |
| OpenAI + Anthropic compatible APIs | **Yes**, both, plus Responses. Tool calls parsed in eleven wire formats. |
| NVIDIA RTX 30/40/50 | **Runs, correct, behind, and the published receipts are stale.** Receipts on an RTX 3060 and an RTX 3090 (0.17.1 and 0.21.0) and an L40S (0.59.0); correct by `frink verify` and `frink parity`. The RTX receipts read 22x to 43x on prefill and predate the resident prefill and the tensor-core GEMM; the ledger marks every CUDA row stale. The last RTX 3090 prefill number measured after those landed is 4.2x (1932 tok/s against ~8,200), not a receipt. No GPU in CI, and no receipt on this build. |

Frink also runs on Apple Metal, where it is fastest. The largest gap is
out-of-core: with `FRINK_EXPERT_CACHE_BYTES` below the expert bytes the
residency policy runs and has been checked on models that fit, but no
model that really exceeds RAM has run, and streamed layers decode
several times slower than resident ones even with every expert cached
([`plans/out-of-core-moe.md`](plans/out-of-core-moe.md)).

## Serving policy

A Rust port of the host-side decision logic in
[FreeToken](https://github.com/FlashML-org/FreeToken)
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)): tensor-free
modules that take measured numbers and return a decision, testable
without a GPU. `expert_store` is the single holder of the expert byte
budget.

### Driving something today

| Module | Decides | Where it runs |
|---|---|---|
| `parser` | where reasoning ends and the answer begins, and which tool was called in which format | `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, streaming and buffered |
| `detokenize` | what text is safe to stream after one more token | the stop-string withhold rule, which `frink-server`'s `StopMatcher` delegates to so there is one implementation |
| `radix` | which prefix of a new prompt is already computed, page-keyed and node-sharing | the paged-KV serving path, where it shares KV pages between prompts by reference count; its occupancy is `pools.prefix_cache` on `/v1/stats` |
| `anchor` | how far a window may slide, and where a tool call pins it so the next agentic turn rejoins rather than recomputes | the paged-KV serving path, on both the private generate loop and the continuous batcher |
| `effort` | which reasoning-effort dialect a checkpoint speaks | probed once per checkpoint at load, then applied to every request's `chat_template_kwargs`, and advertised on `/v1/models` |
| `stats` | what a server may honestly claim about its own throughput and latency | `/v1/stats`, `/v1/requests`, `/admin/stats`. Both decode paths feed the `/v1/stats` rates and request counts |
| `maintenance` | whether a request, a cache rebuild or a stop may proceed right now | `POST /v1/cache/rebuild` and `POST /v1/admin/prepare-stop` |
| `pool_budget` | whether a live re-split fits its floors and byte budget, before anything is freed | `POST /v1/cache/rebuild`, which refuses a KV target over `FRINK_KV_BYTE_BUDGET` |
| `rebuild` · `outbox` · `footprint` | whether a re-split rolls back, what a stop receipt is worth, what this process really occupies | the same two admin endpoints |
| `deepseek_v4_budget` | per-layer KV tier sizing, and which compressor each layer runs (none / CSA / HCA) | the DeepSeek-V4 decoder |
| `bench_profile` · `bench_client` | when a measured bandwidth profile may be trusted, and what a serving benchmark may report | `frink bench-bw` and `frink serve-bench` |

### Complete, tested, and waiting for a consumer

`qstar` (the `q*` bandwidth split), `expert_cache`, `expert_budget`,
`placement` and `residency`, all in `frink-core`, are unit-tested and on
no serving path, so no benchmark is evidence for them. They wait on
[`plans/out-of-core-moe.md`](plans/out-of-core-moe.md).

`expert_slots` is on the decode path: with `FRINK_EXPERT_CACHE_BYTES`
set, the expert store keeps entries in a `SlotDevice`
(`frink-core/src/expert_slab.rs`), an anonymous host mapping on CPU or
one Metal Shared buffer the kernels read in place, built by
`frink-models/src/loader/expert_store_build.rs`. `CudaExpertPool`
implements the same trait under `--features cuda`, compile-verified
with its hardware test `#[ignore]`d.

In `frink-server::policy`, each module with an unwired half names the
roadmap item that closes it at its declaration;
`grep -n "allow(dead_code)" crates/frink-server/src/policy/mod.rs` lists
what still owes a caller.

See [`API.md`](API.md) and [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).
