# Features

Frink is a pure-Rust GGUF inference engine for dense and MoE models.
Weights stay quantized when the file is mmapped, and dequantization
happens inside the matvec. Backends: CPU, Apple Metal and CUDA, plus an
opt-in Vulkan backend.

## Models

Every architecture in `capability::AUDITED_GENERIC_GQA` runs with a
benchmark row, a pinned logit comparison against real `libllama`, or a
fixture behind it (the README states the count, which a test pins). The
architectures on the same path that have been read but not audited stop
with an error that names what is missing. `frink archs` prints the
current list; [`MODELS.md`](MODELS.md) says what runs, what refuses and
why.

Speed against llama.cpp is in
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md), one row per host,
backend and checkpoint, each with the build it was measured on. Most of
its rows are marked stale (measured on an older build). On those rows
Metal is level with llama.cpp on prefill and ahead on decode, CPU is
behind on prefill, and CUDA is behind on both.

The model shapes the decoder can express:

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

GLM-5.2, Kimi and DeepSeek-V4 have loaders and primitives in the tree,
and nothing has run end to end on a real checkpoint of any of them, so
they are not in the table.

Two backend limits affect model choice. On Metal, gpt-oss and Llama 4
run on the host path (no Metal kernel for attention sinks; Llama 4's
window is chunked and the fused Metal launches take one window per
layer); CUDA's resident stack runs gpt-oss. A recurrent model gets no
prefix-cache reuse, because its state cannot be truncated to a middle
position. It can be a speculation target (its state is checkpointed
before each verify batch and replayed after a rejection) on the CLI and
the server's private loop, but not in the continuous batch or on the
paged store, and it cannot be a draft.


## Backends

| Backend | Capabilities |
|---|---|
| **CPU** | Dense and MoE. An integer dot against quantized activations for every weight kind (ggml's `vec_dot_type`, ADR 0009), with AVX2 and NEON bodies; on by default where it measured faster (aarch64), `FRINK_CPU_INT_DOT` overrides. Interleaved Q4_Kx8 / Q8_0x4 GEMV, a Q8_0x4 batch GEMM for prefill, and a thread pool sized to the performance cores |
| **Metal** | FA-vec attention (decode d=64/96/128/256, prefill d=128/256), concurrent FFN/QKV encode, fused MoE groups, `mul_mm_id` prefill, quantized KV (`q8_0` / `fp8` / `q4_0`, the last with a Hadamard rotation on K). A server decodes on the private loop, with attention on the device, unless continuous batching is turned on |
| **CUDA** | Resident weights and a resident stack for prefill and decode (`--features cuda`): norms, QKV bias, QK norm, RoPE, attention, SwiGLU or GeGLU and residuals on the device, K/V rows back to the host cache through page-locked memory. Prefill multiplies `Q8_0` and `Q4_K` on the int8 tensor cores and the other kinds on the f16 ones, and a prefill batch's attention runs on the tensor cores at head widths 64, 128 and 256 (`sm_80` and up). Decode matvecs for `Q4_K`, `Q6_K`, `Q8_0` and `MXFP4` dot a q8_1 activation with `dp4a`, as llama.cpp does. Routed MoE layers route on the device for plain softmax and gpt-oss routing, and a prefill batch's experts run as one grouped GEMM per projection. gpt-oss runs with its sinks, biases and clamped SwiGLU on the device. `FRINK_CUDA_GRAPH=1` replays a decode token from a CUDA graph; it is off by default and has not run on a GPU |
| **Vulkan** | Matvec, prefill GEMM and embedding gather for every weight kind, resident weights, and a decode and prefill stack (`--features vulkan`, opt-in with `--device vulkan` or `FRINK_VULKAN=1`). Correct, not yet faster than the CPU: see below |

Which quant kinds each backend runs, per op (matvec for decode, GEMM
for prefill, `get_rows` for the embedding), and which fall back to the
host, is [`KERNELS.md`](KERNELS.md). It is generated from the dispatch
tables by `frink caps --kernels` and a test fails when it drifts. A
fallback also prints a `NO KERNEL` line at load, and `frink inspect-plan`
lists the tensors of one GGUF that would take one.

### Vulkan

The shaders are GLSL in `crates/frink-vulkan/shaders`, with the
compiled SPIR-V committed beside them. Each kind is run against its
`frink_quant` scalar twin on a real device (an M2 Pro through MoltenVK)
and on Mesa's lavapipe in CI. Weights are uploaded once and stay
resident.

The stack runs a run of layers in one submission (one per 26 layers on
a deeper model), for a decode token or a prefill batch: pre-norm
(RMS, LayerNorm, weightless or biased), QKV bias, QK norm, RoPE, an f16
KV mirror, attention with or without a sliding window, `wo`, post-norm,
SwiGLU or GeGLU, and the residuals. Each layer's K and V rows go back
to the host cache, which stays the authority. A layer the stack does
not serve (attention sinks, a `wo` bias, a chunked window, an attention
softcap, RoPE `mscale`, routed experts, or any layer under `--ctk f32`,
since the device KV is f16) runs on the host with its matvecs
offloaded.

The stack is held to the CPU and to llama.cpp's decode goldens over the
graph fixtures in CI on lavapipe
(`decode_step_graphs::vulkan_decode_matches_cpu_and_llama_cpp`), and
`frink parity` is MATCH with it on Llama-3.2-3B Q4_K_M at a 300-token
prompt. It stays opt-in until it is measured against the CPU on rented
hardware (ADR 0006). It exists to give every AMD and Intel GPU a path,
including the ones ROCm does not support (#409).

### CUDA

The GEMM's hardware test passes every kind and shape on an RTX 3090,
and `frink verify --backend cuda` is token-identical to the CPU on
Q4_K_M, Q5_K_M, Q6_K, Q8_0 and IQ4_XS checkpoints. On rented L40S, A10G
and RTX PRO 6000 cards every dense model checked was token-identical to
the CPU and at parity MATCH with llama.cpp, and the tensor-core
attention and `dp4a` matvecs are token-identical to the CPU on an A100. The
`Q4_1`, `Q5_1`, `TQ1_0`, `PTQ1_0` and seven IQ1/IQ2/IQ3 kernels, the
`get_rows` gather and the CUDA graph replay have not yet run on
hardware (#405, #537).

CUDA is behind llama.cpp on speed; the gap per card and model is in
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md). A Windows or Linux
install runs and answers correctly, but is not yet the choice for
speed. `/health` reports each capability with a reason string.

## Quantization

Parsed and executable: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`,
`Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`,
`IQ4_NL`, `IQ4_XS`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S`, `MXFP4`, `TQ1_0`, `PTQ1_0`.

Every weight kind has a matvec and a GEMM kernel on CPU, Metal, CUDA
and Vulkan; [`KERNELS.md`](KERNELS.md) shows it per op and backend.
`Q8_1` is ggml's activation format, with no weight path to match, which
`KERNELS.md` marks `n/a`. Some of the CUDA kernels have not yet run on
a GPU (#405).

Metal's MoE id kernels: `mul_mv_id` (decode, and prefill below
llama.cpp's batch threshold) covers every kind with a Metal kernel but
`PTQ1_0`, whose experts run one matvec each. `mul_mm_id` (prefill)
covers `Q4_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`,
`IQ4_NL`, `IQ4_XS` and `MXFP4`; an expert pack of any other kind
prefills through `mul_mv_id`.

`I32`, `TQ2_0`, `NVFP4`, `Q1_0`, `Q2_0` and `PQ2_0` are recognized and
sized, and nothing executes them. They parse, `frink inspect` reports
their real footprint, and a checkpoint needing one stops with an error
naming the format.

`IQ2_XS`, `IQ2_S`, `IQ3_S` and `IQ1_M` were validated bit-exact against
llama.cpp's own `dequantize_row_*` by linking `ggml-quants.c`. They
have not been validated end to end on a published `UD-*` checkpoint.

## CLI

llama.cpp-style completion flags (`-m`, `-p`, `-n`, `-ngl`, `--ctk`, …),
plus `frink chat`, `pull` (Hugging Face Hub), `inspect`, `inspect-plan`,
`archs`, `presets`, `gguf-split`, `batched-bench`, `verify` (a GPU
backend against the CPU), `parity` (frink against llama.cpp) and
`layer-divergence`. See [`CLI.md`](CLI.md).

Constrained decoding is on the CLI too: `--grammar`, `--grammar-file`
and `-j` / `--json-schema`, the same spellings llama.cpp uses, reaching
the same stack machine the HTTP `grammar` field does. The host KV cache
keeps `f16` by default and `f32` under `--ctk f32`, llama.cpp's default
and opt-out, so a CPU run holds the same bytes per token as llama.cpp;
`--ctk` also selects the Metal device store's quantized wires, and the
startup banner names what each backend really keeps.

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

Speculative decoding on the CLI takes a draft model with `-md`: an
autoregressive GGUF of the same family, or a DFlash / DFlash2 block
draft (llama.cpp's `dflash` architecture, picked from the file as
`--spec-type draft-dflash`). A DFlash draft reads the target's residual
stream at the layers it names and proposes a whole block in one pass;
its drafted tokens and lattice scores match libllama on the fixtures
(`tests/dflash_graphs.rs`), and the target's greedy output with it is
token for token the output without it, on the CPU and on Metal. The
server takes the same draft with `--model-draft`, on its private loop
and in the continuous batch. See [`CLI.md`](CLI.md) and
[`CONFIG.md`](CONFIG.md).

`frink perplexity` is the quality axis: corpus evaluation using
llama.cpp's method, agreeing with `llama-perplexity` to within a fifth
of one standard error on five checkpoints. Where the two differ, the gap
is monotone in the quant and has the sign the documented `vec_dot_type`
difference predicts. `frink quantize` writes `Q8_0`, `Q4_K_S`,
`Q4_K_M`, `Q5_K_S`, `Q5_K_M` and `Q6_K` byte-identically to
`llama_model_quantize()`, with or without an importance matrix
(`--imatrix`, 311 of 311 tensors identical to `llama-quantize
--imatrix` on a BF16 Qwen3-0.6B for all five targets tried), and
refuses every other target by name. `frink imatrix` is
`llama-imatrix`: same file format in both directions, so either tool's
matrix feeds either quantizer. See [`CLI.md`](CLI.md).

The sampler flags carry llama.cpp's own defaults on `--temp` (0.8),
`--top-k` (40), `--top-p` (0.95), `--min-p` (0.05) and `--repeat-last-n`
(64). One default differs on purpose: `--repeat-penalty` is
1.1 here and 1.0 (off) in llama.cpp
(`common/common.h:239`), so a run left entirely to defaults is not
token-identical. `-e`/`--escape` is on by default as it is there, and a
*partial* `-ngl N` is refused rather than silently offloading every
layer.

That default has a Metal decode cost. At `--temp 0` the Metal stack can
fold `final_norm + lm_head + argmax` into its own command buffer and
hand back one token id instead of `vocab_size` floats. A device argmax
cannot apply the penalties, so the fold is refused whenever any of
`--repeat-penalty`, `--presence-penalty` or `--frequency-penalty` is
live, which at `--repeat-penalty 1.1` is every plain greedy run.
`--repeat-penalty 1.0` or `--repeat-last-n 0` gets the fold back and
is also what makes a run token-identical to llama.cpp's defaults.

`frink bench -m model.gguf` works like `llama-bench`: the same `pp512`
and `tg128` workloads, reported as llama-bench's mean and sample stddev.
Add `--compare` to run `llama-bench` alongside it and print the gap.
`--suite` drives every entry in
[`benchmarks/suite.json`](../benchmarks/suite.json) and regenerates
[`RESULTS.md`](../benchmarks/RESULTS.md). See
[`benchmarks/README.md`](../benchmarks/README.md).

## Server

Two ways to start the same server. `frink serve` is a subcommand of the
main binary, built in by default (the `serve` feature, which
`--no-default-features` leaves out for a completion-only build).
`frink-server` is that same server as its own executable, and both
parse identical arguments through identical code.
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
  and the six generating routes answer with a 501 that names the
  model rather than a missing tensor. Pooling comes from the checkpoint's own
  `pooling_type` (NONE / MEAN / CLS / LAST; RANK refuses, it is a
  classification head rather than a pooling rule, and a rank-head
  checkpoint belongs on `/v1/rerank` instead). A decoder GGUF still
  pools its hidden states (mean/last) as before (`llama-embed`, which
  llama.cpp runs with no output head, refuses by name until its pooled
  embedding is checked against llama.cpp's), and
  `FRINK_EMBEDDING_MODEL_PATH` runs an encoder side-by-side with a
  generative model in one process
- Reranking. A `bert` checkpoint carrying a rank head (`cls`,
  `cls.output`, `cls.norm`, `classifier.output_labels`) is served by
  `POST /v1/rerank`, scoring `[CLS] query [SEP] document [SEP]` through
  the head itself rather than through the cosine of two embeddings.
  Verified end to end against `ms-marco-MiniLM-L6-v2`: the order matches a HuggingFace
  reference, which needed the document to be segment 1 rather than
  llama.cpp's all-zero token types. The scale matches too, once the
  pooler is restored: llama.cpp's converter deletes `bert.pooler.dense`
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
- Several completions per request. `n` and `best_of` prefill the
  prompt once and fork the KV per choice, so `prompt_tokens` counts it
  once while `completion_tokens` sums. Choice `i` samples from
  `seed + i`, so choice 0 of four is byte-identical to the single
  answer at the same seed. With `stream`, the choices are interleaved
  a token at a time, each chunk carrying its own `choices[].index` and
  its own reasoning and tool-call parser state. On the paged store the
  fork is copy-on-write: full pages are shared and only the
  part-written tail is copied
- Per-token steering. `logit_bias` (upstream's `-100..100`,
  outside it a 400 rather than a clamp), `allowed_token_ids` and
  `bad_words`, all applied in the same mask the grammar, JSON mode and
  the reasoning budget share. A bias cannot lift a token a constraint
  forbade, because a bias is finite and a mask is `-inf`
- Logprobs. `logprobs` on completions (parallel arrays with
  `text_offset`) and on chat (OpenAI's `content[]` shape), over the
  distribution the sampler drew from. Candidates the chain removed
  are omitted, not reported as `null`. `prompt_logprobs` scores the
  prompt from the plain softmax, because a prompt token is supplied,
  not drawn. `return_tokens_as_token_ids` spells each reported token by
  its id, since two ids can detokenize to the same text
- `skip_special_tokens: false` keeps the end-of-generation token in the
  returned text and in `usage`. It still ends the answer; a
  `stop_token_ids` entry is excluded as a stop string is
- Prompt controls. `echo` returns the prompt and the completion as
  one string with the logprobs arrays covering both;
  `truncate_prompt_tokens` keeps the last `k` tokens, applied before
  anything is prefilled so the KV, the usage and `echo` all see the
  prompt that was answered; `cache_salt` names a caller's prefix-cache
  namespace, honoured by the contiguous cache, the response cache and
  the paged store's radix tree
- Sleep and wake (`POST /sleep`, `POST /wake_up`,
  `GET /is_sleeping`): an unload that remembers the model, so the
  server can bring it back itself. It frees the KV pool, the paged
  store, the repack and expert caches and any device buffers. There is
  one sleep level: frink mmaps its weights, so dropping the handle
  already discards them
- Sentence-pair scoring (`POST /v1/score`, `POST /score`): a
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
- Continuous batching: concurrent clients share one batched decode
  worker (llama.cpp's slots model), streaming tokens as they come. It is
  opt-in on every backend (`-cb`, `-np` / `--parallel N` or
  `FRINK_CONTINUOUS_BATCHING=1`); without it concurrent requests
  serialize on the private loop. On Metal the private loop is also the
  faster one for a single client, since it attends on the device and the
  batch attends on the host. The measured throughput at each
  concurrency is in
  [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md#serving-features-measured)
- Prefix cache (`FRINK_PREFIX_CACHE_ENTRIES`, private loop): whole KV
  snapshots in an LRU list, so a follow-up chat turn prefills only the
  new message. It stores a greedy Metal request too, whose decode keeps
  only the token id, and refuses a recurrent model's state. See
  [`CONFIG.md`](CONFIG.md)
- Cache-aware admission. When a radix prefix cache is configured,
  a waiting job whose prompt is already computed may be admitted ahead
  of one that is not. Two bounds keep it from starving any job: only
  the first eight waiting jobs are considered, and no job may be
  passed over more than four times, after which nothing overtakes it.
  It has no effect without a prefix tree or on a cold one, because a
  tie keeps arrival order. The capacity rule is still strict FIFO with
  respect to the job it picked: if that one does not fit, the line
  stops instead of skipping ahead by size
- Chunked prefill (same scheduler as continuous batching). `-b` /
  `-ub` set the chunk on both decode paths from one number. Each worker
  tick has one token budget shared by prefill and decode
  (`FRINK_CB_MAX_BATCH_TOKENS`, default 256): the prefill chunk takes
  what the decoding rows leave, with a floor of 16 prompt tokens, so a
  growing batch shortens the chunk instead of delaying every row's next
  token. The latency effect has not been measured yet
- Speculative decoding on the server. A request drafts tokens by
  prompt lookup (an n-gram match over its own history), or with the
  draft model `-md` names (autoregressive or DFlash), and verifies them
  in one batched forward. A drafted token is kept only if it equals the
  sampler's own draw, so the output does not change. It runs on the
  private loop, on the paged store (a rejected block rewinds its
  tables) and in the continuous batch, where every row's block is
  verified in one multi-sequence forward. Grammar and `json_object`
  requests and a paged lease under a sliding window do not speculate,
  and `usage` reports the speculation fields only when it ran. A target
  with recurrent layers (Mamba, GDN, Lightning) speculates on the
  private loop only: its state is checkpointed before each verify batch,
  and a rejection restores it and replays the committed tokens, as
  llama.cpp's server does. A DFlash draft is refused beside the prefix
  cache and the paged store
- Slot save/restore (llama.cpp's `POST /slots/{id}?action=save|restore`):
  a prompt prefix's KV written to `--slot-save-path` and restored into
  the prefix cache after a restart, so a long system prompt is prefilled
  once per file rather than once per process. The file carries a
  checkpoint fingerprint, and a restore under another model or another
  quantisation is refused by name. See [`API.md`](API.md#slot-save-and-restore)
- llama-server's own routes: `/props`, `GET /slots`, slot erase,
  `/infill`, `/apply-template` and `/metrics`. See [`API.md`](API.md)
- Paged KV: shared page storage many requests read through a block
  table, with a radix tree over reference-counted page groups so
  conversations off one system prompt share its KV rather than each
  holding a copy. Off unless `FRINK_PAGED_KV_BLOCKS` is set. See
  [`CONFIG.md`](CONFIG.md)
- On the paged store, a model whose layers *all* slide by the same
  window slides during decode, so a request holds its prompt and a
  window rather than its whole context, and admission prices it
  that way, so a store too small for the whole context still serves
  it. A tool call anchors the slide at the position the next agentic
  turn will rejoin at, and the anchor is dropped once the cursor
  drifts a window past it. An alternating-SWA model (gpt-oss,
  Gemma-3) does not slide *there*: a page group holds one block in
  every layer, and the full-attention layers still read position 0
- On the contiguous host store, eviction is per layer and on by
  default, so the alternating models get it: each layer's `KvCache`
  carries its own window from `attention.sliding_window` and
  `attention.sliding_window_pattern` (or the file's per-layer array) and
  keeps only `window` rows plus slack, while the full-attention layers
  keep everything (Gemma-3-4B at a 32k context keeps under a fifth of
  its KV). Speculative decoding, the prefix cache and
  `n > 1` forks all work on an evicting cache: a drain keeps a rollback
  reserve and drafts are bounded by it, and a prefix-cache match deeper
  than the cache can truncate to is a miss rather than a history with a
  hole in it. `FRINK_KV_WINDOW=0` keeps every row. It turns itself off
  under `FRINK_METAL_ATTN` and on a draft model, and the paged store and
  the KV pool do not evict. Output is token-identical with it on or off.
  `--ctx-size auto` and the pre-load admission check are priced against
  the same per-layer residency. See [`CONFIG.md`](CONFIG.md)
- `frink serve-bench`: concurrency, TTFT, TPOT and queueing numbers
  for a live server, with the methodology (positional split, pooled
  nearest-rank percentiles, whole-run throughput) tested socket-free.
  `--shared-prefix` puts one system message ahead of every request and
  reports prefix-cache reuse from the server's own `cached_tokens`, as
  a share of all prompt tokens in the run (`--cache-salt` gives the run
  its own namespace).
  The serving receipts are in
  [`benchmarks/receipts/serving/`](../benchmarks/receipts/serving/)
- Live serving telemetry (`GET /v1/stats`, `GET /v1/requests`) and an
  elastic KV/expert split that can be reported and re-sized without a
  restart (`GET /v1/cache/status`, `POST /v1/cache/rebuild`). A request
  that arrives mid-rebuild is turned away with an error rather than
  parked in a queue behind it
- `reasoning_content`: a reasoning model's chain of thought is split
  out of `content`, streamed as it arrives rather than at the end
- Reasoning budget: `reasoning_budget_tokens` / `thinking_budget_tokens`
  (`thinking.budget_tokens` on `/v1/messages`, `--reasoning-budget N` as
  the server default) forces the closing tag once N tokens of thought
  are spent, so the answer still arrives with `finish_reason: "stop"`.
  It matches llama.cpp token for token on DeepSeek-R1-Distill at
  temperature 0
- `continue_final_message` (llama.cpp's field and values): a trailing
  assistant message is continued as a turn still being written, with its
  thought re-opened in the family's own markers. It is on by default, as
  in llama.cpp's server; `false` in the request or
  `--no-prefill-assistant` on the server turns it off
- Tool calls in eleven wire formats: the format the served
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

The table compares that description with what frink does; the gaps
are the roadmap.

Its structural claims are pinned by
`crates/frink-server/tests/documented_capabilities.rs`, so a row that
stops matching the code turns a test red.

| Capability | In frink today |
|---|---|
| Bandwidth-adaptive CPU/GPU co-execution (`q*`) | **Partial.** `qstar::BandwidthProfile` is in `frink-core` and used by `frink bench-bw`, a measurement tool. The serving path does not consult it. |
| Full-layer double-buffered prefill streaming | **Built, not wired.** Kept for the out-of-core work, which names it. |
| Global LRU expert caching | **Yes, singular, and ranked by routing counts.** `expert_store` is wired into both decode paths; for a GGUF model its entries live in the slots of a `SlotDevice` (host RAM, or on Metal one Shared buffer the GPU reads in place), eviction ranks by the decoder's activation counts, and decode is proven identical to resident at half, a quarter and one slot of the expert bytes. |
| Graph-compatible execution | **Off by default.** Execution is eager. `FRINK_CUDA_GRAPH=1` replays a CUDA decode token of the resident stack from a captured graph; it has not run on a GPU. `ExecutionPlan` is built and read by nothing. |
| FTW fast weight format | **No.** GGUF only. |
| Semantic anchor checkpoints for KV | **Yes.** `anchor::decode_slide` and `WindowPolicy` are wired into `generate.rs` and the batch scheduler. |
| Agentic context edits without recompute | **Partial, and no longer leaking.** The radix prefix cache shares pages and reports `cached_tokens` on both the private and the batched path. `evict` is called from `acquire_paged_caches`, which both paths go through, when a request cannot find pages: the tree gives back what no live lease has locked, and the caller retries immediately rather than charging reclamation against its deadline. What is still partial is the edit itself: a changed middle forces a recompute from that point, because the cache is a prefix tree. |
| Elastic VRAM re-allocation without restart | **Partial.** `POST /v1/cache/rebuild` re-splits KV pool geometry at runtime. Moving bytes between an expert cache and KV is not implemented. |
| MXFP4 / BF16 | **Yes**, executable. GGUF-block MXFP4 has Metal and CUDA matvec and GEMM kernels; the safetensors two-buffer form is CPU-only. |
| NVFP4 / FP8 | **Refused by name, not executed.** NVFP4 is recognized and sized by `frink-gguf` (tag 40) and has no execution path, so a file carrying it stops instead of computing something else. FP8 appears only as a KV cache wire (`--ctk fp8`), not as a weight format. |
| DeepSeek-V4-Flash, GLM-5.2, Kimi K3 | **Loaders and primitives only.** Nothing has run end to end on a real checkpoint. |
| OpenAI + Anthropic compatible APIs | **Yes**, both, plus Responses. Tool calls parsed in eleven wire formats. |
| NVIDIA RTX 30/40/50 | **Runs, correct, behind, and the published receipts are stale.** Receipts on an RTX 3060 and an RTX 3090 (0.17.1 and 0.21.0) and an L40S (0.59.0); correct by `frink verify` and `frink parity`. The RTX receipts read 22x to 43x on prefill and predate the resident prefill and the tensor-core GEMM; the ledger marks every CUDA row stale. The last RTX 3090 prefill number measured after those landed is 4.2x (1932 tok/s against ~8,200), not a receipt. No GPU in CI, and no receipt on this build. |

Two notes. Frink runs on Apple Metal, which that description does not
cover, and Metal is where it is fastest
([`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)). The largest gap
is missing from this table: running a model that does not fit in
memory executes its residency policy (set
`FRINK_EXPERT_CACHE_BYTES` below the expert bytes), and has been checked
on models that fit, under a budget. It has not yet run a model that
really exceeds RAM, and streamed layers decode several times slower
than resident ones even when every expert is cached
(`docs/plans/out-of-core-moe.md`).

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

`qstar` (the `q*` bandwidth split), `expert_cache`, `expert_budget`,
`placement` and `residency`, all in `frink-core`. Each is covered by
unit tests and none of them is on a serving path, so do not read a
benchmark as evidence for any of them. They are waiting on
[`plans/out-of-core-moe.md`](plans/out-of-core-moe.md): running a model
larger than memory, which is the largest thing they would enable.

`expert_slots` is on the decode path. With `FRINK_EXPERT_CACHE_BYTES`
set, the expert store keeps its entries in the fixed slots of a
`SlotDevice` (`frink-core/src/expert_slab.rs`): an anonymous host
mapping on CPU, and on Metal one Shared buffer that the CPU writes and
the kernels read in place, built at load by
`frink-models/src/loader/expert_store_build.rs`. `frink-core`'s
`CudaExpertPool` implements the same trait under `--features cuda`,
compile-verified with its hardware test left `#[ignore]`d, so on a real
card the property is written down and not yet measured.

Inside `frink-server::policy`, the modules carrying an unwired half
name the roadmap item that would close it, at their declaration in
`policy/mod.rs`. `grep -n "allow(dead_code)" crates/frink-server/src/policy/mod.rs`
is the list of what still owes a caller.

Frink Studio, the web UI in [`ui/`](../ui), is a separate app that
talks to this API over HTTP. `frink-server` does not serve it, and
`GET /` on it is a 404.

See [`API.md`](API.md) and [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).
