# Roadmap

The goal is to match or beat [llama.cpp](https://github.com/ggerganov/llama.cpp)
tok/s on the same host, the same backend and the same GGUF. Current
numbers: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

What ships today: [`FEATURES.md`](FEATURES.md) ·
[`MODELS.md`](MODELS.md).

## Closed on 2026-09-01

Against [`plans/llama-cpp-gap-inventory.md`](plans/llama-cpp-gap-inventory.md),
whose §7 listed seven things that were silently wrong. All seven are
fixed:

- The repetition penalty compounded as `penalty^n`, and temperature ran
  before the truncation filters. Both were live on every `frink run` at
  the defaults
- `phi3` windowed every layer where llama.cpp windows none
- `logit_bias` was dropped on `/v1/chat/completions`, and JSON-object
  mode was dropped under continuous batching
- `Q5_0` prefilled on the GPU and decoded on the CPU
- The attention-softcap refusal gated on a GGUF key no converter writes,
  so it could never fire while reading as coverage

Also landed: `--min-p` and `--repeat-last-n` at llama.cpp's defaults, a
GBNF grammar engine wired to both HTTP routes and all three decode
paths, a CUDA batched GEMM (**never run on a GPU**, see item 4 below),
and a triage of every unaudited architecture refusal into
fixture-away / one-match-arm / new-code / unknown.

## Behaviour change, unreleased: penalties now see the prompt

**This changes generated text for every user at the default
`--repeat-penalty 1.1`, and belongs at the top of the next release's
notes.**

llama.cpp seeds its sampler with every prompt token before the first
draw (`tools/server/server-context.cpp:386-390`, and `llama-cli` does
the same), so `penalty_last_n` slides over the tail of
`prompt ++ generated`. frink slid it over `generated` alone. Same
checkpoint, same flags, same prompt, different text.

A token that occurs in the prompt is now penalised on its FIRST
generated occurrence rather than its second. `--repeat-penalty 1.0` or
`--repeat-last-n 0` reproduces the old output exactly.

This was taken as a parity fix rather than left as a documented
divergence because it is a DEFINITION, not a default. This project
already carries one deliberate deviation, `--repeat-penalty` defaulting
to 1.1 against llama.cpp's 1.0, and that one is visible on a typed flag.
This was the same flag meaning a different thing: invisible from the
command line, and invisible to `frink parity`, which compares logits
and tokenizers rather than sampled text.

It also closed a five-way disagreement inside frink. The window was a
`&[usize]` and five call sites chose four different answers, with
`kimi_generate` quietly the only one that matched llama.cpp.
`PenaltyWindow` has no constructor taking a single slice, so a caller
cannot build one without saying what its prompt is, and the two places
that genuinely have none say `&[]` visibly in the diff.

## Closed on 2026-09-02

- **Vulkan is a third backend**, behind `--features vulkan`. A `Q8_0`
  matvec and nothing else, so a prefill still runs on the host. The
  backend registry list is generated from the seam rather than
  hand-kept, which is what caught a `Q5_0` Metal matvec that had been
  half-wired since the previous release
- **Encoder embeddings.** A BGE / E5 / GTE checkpoint can be the loaded
  model, `/v1/embeddings` pools it the way the file says to, and the
  generating routes answer 501 naming the model. Checked against
  llama.cpp
- **WordPiece tokenization**, verified against llama.cpp
- **Lazy grammars**, so `tool_choice: "required"` and a named
  `tool_choice` are enforced rather than asked for in the prompt
- **`pattern` in the JSON-schema-to-GBNF converter**
- **`response_format: json_schema`** is served rather than refused, and
  the second decision site that answered it with "only json_object is
  supported" is deleted. A forced `tool_choice` beside a schema is now
  refused against the *resolved* grammar: the check asked `self.grammar`
  and a schema would have walked past it
- **The GGUF parser is bounded against a hostile file** (#24, #25, #26):
  every count, string length, array length and nesting depth is checked
  against what the file can actually contain, rather than against a
  chosen constant
- The repack cache served a dead mapping's bytes, and the CLI banner
  reported a KV dtype the run does not keep

### Second batch, same day

- **`/v1/rerank`** runs the checkpoint's own classification head rather
  than a cosine similarity. Such a checkpoint could not load at all
  before: `assert_every_tensor_consumed` rejected the `cls.*` tensors
  nobody read
- **Speculative decoding with a real draft model.** `-d` /
  `--model-draft` loads a second GGUF; the rejection rule was already
  lossless at every temperature. Measured on a 3B target with a 1B
  drafter: 40 tokens in 12 verification steps, acceptance length 3.33
- **llama.cpp's `-hf user/repo:QUANT`** on `run`, `serve` and
  `download`, with a cache under `FRINK_CACHE`. Plus the server flags a
  copied `llama-server` command carries: `-c`, `--api-key`,
  `--api-key-file`, `--alias`, `--ctk`, `--hf-file`, and `--jinja` /
  `--no-warmup` / `--flash-attn` accepted rather than fatal
- **`--presence-penalty` and `--frequency-penalty` on the CLI.** The
  engine and the HTTP API had always honoured them while `run`
  hardcoded both to zero
- **Five one-match-arm architectures admitted**, so unaudited went 46 to
  41 and audited 11 to 16
- **Forced `tool_choice` on ten of the eleven wire formats**, up from
  three, with the grammar's literals built from the same marker
  description the parser reads with. `gemma4` got a shape of its own
  (`call:NAME{k:v}` in gemma's quoting) and `minimax_m3` joined the
  element shape; only `muse_glimmer` refuses, because its call boundary
  is a channel header whose recipient name is a fact about a template
  rather than about the format
- **Gemma-2-27B and Gemma-3-4B/12B/27B numerics**, and Gemma-3 4B+ is
  back on the fused Metal stacks with a per-layer `LayerRope`
- **The response cache cannot store a cancelled answer**, enforced by a
  private-field token rather than a check beside the data
- `max_tokens` from an HTTP body reaching `Vec::with_capacity(usize::MAX)`,
  and the batched prefill re-running the whole prompt on top of the
  radix prefix it had just adopted

Verified but NOT measured, and both need a quiet host: the speculative
decoding speedup, and the speed recovery from Gemma-3 4B+ returning to
the fused Metal path.

## Tracked as issues

Open work now has a GitHub issue each, so nothing depends on a person
remembering it. How work lands is written down in
[`plans/contribution-workflow.md`](plans/contribution-workflow.md): a
completed feature is a branch and a pull request, a defect is an issue,
and the two are never the same artifact.

Everything open, as of 2026-09-02:

| # | What | Blocked on |
|---|---|---|
| [#27](https://github.com/antonellof/frink/issues/27) | CPU decode is scheduling-bound. **Measured 2026-09-04 on quiet rented hosts.** On 20-core aarch64 the persistent pool is **+123% at 3B and +87% at 8B**, which takes decode from losing to llama.cpp to BEATING it (23.14 vs 17.86, 12.41 vs 9.06). On 10-core x86 it is +49/+23/+15%. Still opt-in behind `FRINK_CPU_POOL=spin` | A decision, not a measurement. The pool regresses 37% at 135M on aarch64, reproducibly on a quiet host, so the default cannot simply flip. Needs a size or work rule, which is what `MIN_TASK_MACS` was |
| [#128](https://github.com/antonellof/frink/issues/128) | Decode carries a fixed per-token cost of roughly 60 ms that no thread count or pool removes: 135M runs at 13 to 15 tok/s on 4, 8 and 19 aarch64 threads while llama.cpp does 190 to 204 | Finding what the constant IS. It is flat in thread count, so it is not fork-join, and flat in model size, so it is not arithmetic |
| [#126](https://github.com/antonellof/frink/issues/126) | `frink bench --n-gpu-layers 0` does not force CPU, and `bench_suite.rs` uses that flag for every published `cpu` row, so the ledger's CPU numbers may include Metal | Nothing. The backend decision is cached in a `OnceLock` before the flag can apply; the fix is to stop having two answers |
| [#127](https://github.com/antonellof/frink/issues/127) | x86_64 CPU throughput looks ~10x off llama.cpp (3B Q4_K_M at 1.03 tok/s on 10 Broadwell cores), and `benchmarks/RESULTS.md` has no x86 row to show it | A `llama-bench` comparison on the same x86 host, then finding whether the AVX2 arms are reached at all |
| [#29](https://github.com/antonellof/frink/issues/29) | A forced `tool_choice` reaches ten of the eleven wire formats. `muse_glimmer` still answers 501 | A muse-glimmer checkpoint or its chat template. The block is already an element grammar this can write; what is missing is which recipient name the template addresses a tool with, and how much of the channel header the rendered prompt already wrote |
| [#61](https://github.com/antonellof/frink/issues/61) | No KV store evicts behind a sliding window, so Gemma-3-4B holds 9.1 GB where 1.6 GB would do | Splitting `KvCache::seq_len` into positions and rows first. The design and the ordered steps are in the issue |
| [#70](https://github.com/antonellof/frink/issues/70) | `frink quantize` writes Q8_0, Q4_K_S/M, Q5_K_S/M and Q6_K byte-identically to llama.cpp, with or without `--imatrix`, and refuses the rest by name; `frink imatrix` produces the matrix | Q2_K/Q3_K and the IQ-tier encoders. Steps 2 to 4 are in the issue |
| [#82](https://github.com/antonellof/frink/issues/82) | Rerank scores are the head minus its pooler, so a thresholding client gets a range that never fires | A converter that keeps `bert.pooler.dense`, which frink can now write. Ordering is unaffected |

Closed on 2026-09-02 and worth knowing about: the GGUF parser hardening
(#24, #25, #26, #30, #31, #32), `max_tokens` reaching
`Vec::with_capacity(usize::MAX)` (#36), every continuous-batching cache
hit returning wrong logits (#37), the response cache serving
unconstrained answers to grammar requests (#35) and cancelled answers as
finished ones (#57), the Gemma numerics (#39, #40), the unigram
tokenizer panicking per request on a short scores array (#34), the KV
budget pricing a window no store implements (#33), `/tokenize` refusing
for an encoder (#28), penalties never seeing the prompt (#55, #73), and
`/v1/rerank` ranking the answering document LAST (#43, #44).

## Speed gaps against llama.cpp

Closed since the last pass over this list: bench last-token `lm_head`
only, CPU Q4_K batch GEMM (`gemm_q4_kx8_group` in `weight_matrix`),
SmolLM2 Metal greedy lm_head, OLMoE Metal gather plus `mul_mm_sg` and
Q4_0 `mul_mv_id`, the Qwen shared-expert loader fallback, CPU MoE
token-to-expert bucketing (`moe_ffn_batch`), and dense Metal prefill
stack QKV bias with QK-norm (Qwen2.5 / Qwen3 / Gemma-3).

Still open (see the Open section of
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)):

- **Metal prefill.** The Qwen2.5 / Qwen3 / Gemma-3 Q8_0 dense stack now
  includes QKV bias and QK-norm, which took it from roughly 18–21×
  hybrid CPU projection down to 1.2–2.1×. What remains: OLMoE
  gather → `mul_mm_sg` → scatter against a fused `kernel_mul_mm_id`,
  dense 1–3B (~1.5–3×), and a compiled graph or pre-encoded command
  buffer replay to get sub-1.5B models to 1× or better.
- **CPU prefill and decode.** Phi-4 and Mistral Q4_K `pp512` need
  measuring again after the GEMM change. i8mm SMMLA if the gap is still
  above 1×. A persistent decode threadpool.
- **Correctness.** Gemma-4 end-to-end chat smoke test. The older
  Gemma-2 Metal greedy check stays in the tree for regressions only,
  and is not part of the published suite.

## Where the project goes next

This section is the reader-facing half of
[`plans/roadmap.md`](plans/roadmap.md), which holds every open item
ranked, and [`plans/north-star.md`](plans/north-star.md), which is the
rule that ranks them. The plan's order is:

| # | Theme |
|---|---|
| 1 | Fix what is wrong, and close the oracle's hole |
| 2 | Model layer: audit, vocabulary, then split |
| 3 | Close the 41 unaudited architectures, auditing outward |
| 4 | Out-of-core MoE, and one large real checkpoint |
| 5 | CPU decode scaling |
| 6 | Hardware reach: the backend seam, then Vulkan |
| 7 | The rest: embeddings, quants, serving, measurement |

What follows is what those steps mean for someone running the engine
rather than working on it. Where the two disagree, the plan is right and
this is stale.

1. **Run bigger models on the same hardware.** Make Qwen3 35B-A3B Q5
   usable on a box that today handles Q4, or an 8B. Most of what
   follows serves this.
2. **RAM and VRAM optimization.** Residency planning already exists
   (`frink inspect-plan`). What is missing is acting on it hard enough
   to change which models fit: tighter KV (3-bit, quantized CTK),
   streaming expert residency, and not materializing activations
   nothing reads.
3. **Hybrid CPU/GPU, especially for MoE.** Routed experts are the
   natural split. Hot experts stay resident on the GPU, cold ones get
   streamed or run on CPU. `PlacementPlan` and `ExpertStore` are the
   groundwork.
4. **CUDA performance.** The matvec kernels build and run on real
   hardware. Nobody has tuned them. The batched `Q8_0`/`Q4_0` GEMM
   added on 2026-09-01 has **never executed on a GPU at all**: its
   evidence is a scalar twin plus a host harness that runs the emitted
   CUDA against a barrier shim, and its hardware test is `#[ignore]`d
   with "NEVER RUN" as the reason. Putting it on a card, and then the
   dp4a/MMQ integer path and the other five quant kinds, is the work.
5. **Tool calling and full OpenAI API compatibility.** See
   [`API.md`](API.md). GBNF grammars and the lazy grammars behind a
   forced `tool_choice` both ship, and so does
   `response_format: json_schema`, through the same converter a forced
   `tool_choice` compiles its arguments with. What is left is a forced
   `tool_choice` on the eight wire formats that are not JSON-object
   shaped ([#29](https://github.com/antonellof/frink/issues/29)), and
   MCP invocation.
6. **Docker images**, so evaluating any of this stops requiring a Rust
   toolchain.

**Models**

- Gemma-4 tokenizer (`gemma4`) plus an end-to-end chat smoke test (the
  engine loads today)
- Qwen3.5 (`qwen35`, `qwen35moe`, `qwen3next`) and Llama 4 run on
  the generic path since 2026-09-14 (`tests/qwen35_graphs.rs`,
  `tests/llama4_graphs.rs`); no engine for either
- MiniMax-M3's sparse attention (`minimax-m2` runs on the generic path)
- Vision (projector plus generate)
- Real GLM-5.2, DeepSeek V4 and full Kimi, run end to end
- MTP draft heads
- Qwen2-MoE and Mixtral pins, once the GGUF fits Host B

**Serving**

- Tool calling: the OpenAI `tools` / `tool_choice` request and response
  shape. *Eleven wire formats now parse* (`frink-server::policy::parser::tool_call`),
  every call in a response rather than the first, and a reasoning
  model's chain of thought comes back as `reasoning_content`. Five of
  the eleven stream `tool_calls[].index` argument deltas.
  `tool_choice: "required"` and a named `tool_choice` compile a *lazy*
  grammar (llama.cpp's `trigger_patterns`, a grammar switched on partway
  through a response) from the request's own `tools`, on **eight of the
  eleven** formats. The grammar's literals come from the same marker
  description the parser reads with, which is what makes widening it
  safe: two hand-kept tables, one for writing a call and one for reading
  it, would drift into output the engine forced and could not parse
  back. `gemma4`, `minimax_m3` and `muse_glimmer` answer 501 for stated
  reasons rather than for effort (#29). What is left here is argument
  deltas for the six JSON-payload formats, and streamed tool calls on
  the continuous-batching path
- JSON-schema constrained decoding, **done**. The GBNF engine and the
  `grammar` request field shipped on 2026-09-01, on chat, completions
  and all three decode paths; `response_format: json_schema` and
  llama.cpp's bare `json_schema` field followed on 2026-09-02, compiled
  by the same converter and decided at the one site that resolves every
  spelling of "constrain the output"
- MCP tool invocation. Anthropic streaming and tools now ship
- The rest of the OpenAI API surface (see [`API.md`](API.md))
- Docker images (CPU, Metal and CUDA variants)
- Throughput measurement for concurrent continuous-batching requests
  (Metal parallel fix shipped 0.15.2; CB garbled-output fix and Host B
  serving receipts in 0.15.3, see
  [`plans/metal-parallel-concurrency.md`](plans/metal-parallel-concurrency.md)
  and [`benchmarks/receipts/serving/`](../benchmarks/receipts/serving/))
- Full KV layer offload, multi-GPU, tensor parallel, PD disaggregation

**Beyond the llama.cpp surface**

The goal is to replace llama.cpp: the same models, the same command
shapes, the same or better performance. That is the floor, not the
ceiling, and these are the serving-engine features worth having on top
of it, ranked by what they buy on hardware people own. Each is a
feature, not a port: the shapes are public, the implementations are
frink's.

- **Multimodal.** A vision tower and a projector beside the text
  decoder, and the image preprocessing to feed it. The largest single
  item on this list, and the one llama.cpp also has, so it is first.
- **Speculative decoding over HTTP.** The engine half is built,
  lossless and tested and the server cannot reach it; see
  [`plans/server-speculative-decoding.md`](plans/server-speculative-decoding.md).
  A multi-token-prediction drafter is a second `Drafter` impl for the
  seventeen architectures whose MTP blocks frink already skips by name.
- **Quantized safetensors, and quantizing on load.** Reading FP8
  blockwise, GPTQ, AWQ, MXFP4 and NVFP4 checkpoints directly, and
  quantizing a BF16 checkpoint while loading it rather than converting
  first. Both sit behind a generic `config.json` to `ModelConfig`
  loader that does not exist yet: `frink-models` reads safetensors for
  one dedicated stack and one pooler, nothing more.
- **Tensor parallel and multi-node.** Sharding one model across
  devices, then across hosts.
- **Prefill/decode disaggregation and KV connectors**, which the
  out-of-core plan is the groundwork for.
- **CUDA graph capture.** On Metal the equivalent, one encoded graph
  per token, is what 0.25.0's submission collapsing approximates by
  hand.

**Consistency gaps found by audit, not yet closed**

- `--list-devices` does not list Vulkan on either binary, although
  `--features vulkan` builds a Vulkan backend. The listing is one
  function now (`frink_models::devices`), so adding it is one edit
  rather than two, but `frink-vulkan` is reachable only through
  `frink-core`'s optional feature and that plumbing is a row of its
  own.

**KV cache and memory**

- A 3-bit KV dtype. The Metal Hadamard rotation on the CTK path shipped: `--ctk q4_0` rotates K
- Act on the residency plan `inspect-plan` produces: stream cold
  experts, bound the KV budget, report what a host really fits
- Hybrid CPU/GPU expert placement for MoE, the main lever for running a
  larger model or a higher quant on unchanged hardware

**Wiring the ported serving policy**

The ported FreeToken serving policy (see [`FEATURES.md`](FEATURES.md))
is complete and tested. It no longer lives in a crate of its own: the
serving half is `frink-server::policy` and the MoE expert-residency
half is in `frink-core` beside `expert_store`. Roughly half of it
now drives something: the two output parsers, the withhold rule, the
effort/thinking probe, the request ring, the batcher's status and pool
accounting, the two maintenance endpoints, the DeepSeek-V4 KV tier
sizing, and `radix`, which shares KV pages between prompts on the
paged-KV path. `FEATURES.md` has the per-module split.

Closed since the last pass: `frink bench-bw` measures this host's
CPU-MoE bandwidth and writes the profile `qstar` reads, so a deployment
no longer has to take the unbenchmarked one-fetch-per-step default
(the PCIe half still needs a CUDA benchmark host). `POST
/v1/cache/rebuild` moves VRAM between the expert cache and KV without a
restart, validated by `policy::pool` and rolled back by
`policy::rebuild`.

Still waiting on a consumer:

- Drive expert residency from `frink_core::expert_cache` and the `q*`
  split, which needs the other half FreeToken has and frink does not:
  a *persistent* GPU expert cache (`frink-moe::run_expert_placed`
  re-uploads every weight matrix per call) and a CPU MoE path that can
  run concurrently with a device copy
- Make paged KV, and the radix cache riding on it, correct on **CUDA**.
  Metal is done: the prefill left K/V on the device and the paged
  prefill copied host placeholders into the page store, the prefill now
  downloads the real rows, the startup refusal is lifted, and
  `paged_metal_parity` pins identical greedy ids against the contiguous
  cache on a dense, an MoE and a sliding-window model. CUDA has no
  equivalent hardware run
- Give the radix cache an aggregate hit rate on `/v1/stats` and
  `/metrics`, and an eviction budget. `RadixCache::evict` is written and
  tested with nothing calling it, so back pressure today is the page
  store running out
- Make prefix reuse work on the continuous-batching path. Paged KV and
  continuous batching are separate switches and the sharing only
  happens under the first
- Size the pools with `frink_core::expert_budget::plan_cache_budget` at
  load, not only when a rebuild asks for a new geometry
- Find consumers for `frink_core::placement` and `residency`, and for
  `policy::anchor`'s `prefill_slide`. The multi-currency prefix cache
  (`cache_manager`, `radix::swa`, `radix::hybrid`, `window_pool`,
  `state_pool`), the cache report renderer and the process supervisor
  were DELETED rather than left waiting: see
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for what each was
  and why it went
- Publish into the radix tree from the continuous batcher too.
  `batch_scheduler` adopts from the tree (`acquire_paged_caches`) but
  never calls `publish_to_radix`, so under `FRINK_CONTINUOUS_BATCHING=1`
  prefix sharing is adopt-only and the tree is filled by nothing

A full recursive re-read of the reference (six readers over its 435
files, checked against every frink crate rather than against the port's
own scope) found 34 further items, now tracked individually in the plan
below. One of them is a correctness bug in shipped code rather than an
omission: `route_top_k_grouped` implements "k from every group" where
the DeepSeek-V3/GLM rule scores each group by the sum of its top-2
biased scores and then runs one global top-k. That one is first.

Staged plan, including what the port left behind entirely (semantic
anchor checkpoints, the cache manager, the window slide) and what a
CUDA-side parity would actually cost:
[`plans/archive/freetoken-parity.md`](plans/archive/freetoken-parity.md).

**Engineering practice worth taking from llama.cpp**

- Something equivalent to `test-backend-ops`: every kernel checked
  against a CPU reference across shapes and quant kinds, so no backend
  gets merged on the strength of running fast alone.
