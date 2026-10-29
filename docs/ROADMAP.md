# Roadmap

The goal is to match or beat [llama.cpp](https://github.com/ggerganov/llama.cpp)
tok/s on the same host, the same backend and the same GGUF. Current
numbers: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

What ships today: [`FEATURES.md`](FEATURES.md) ·
[`MODELS.md`](MODELS.md).

## What already closed

A roadmap lists what comes next. What has shipped is in
[`CHANGELOG.md`](../CHANGELOG.md), which carries every row with its
date and its evidence. This page used to restate a month of it and went
stale doing so.

Two closures stay here because they changed behaviour a reader may have
depended on, where most closures only add something:

- **The repetition penalty compounded as `penalty^n`, and temperature
  ran before the truncation filters.** Both were live on every
  `frink run` at the defaults. The chain is llama.cpp's order now, and
  the penalty is applied once per candidate rather than once per
  occurrence, so a generation at the same seed and flags can differ
  from one taken before 2026-09-01.
- **Penalties now see the prompt.** llama-server seeds its sampler with
  every prompt token before the first draw, and frink's HTTP path did
  not, so `--repeat-last-n` meant one thing in `frink run` and another
  over HTTP. It has one meaning now: the window is the tail of
  `prompt ++ generated` on both.


## Tracked as issues

Each piece of open work has a GitHub issue, so nothing depends on a
person remembering it. How work lands is in
[`plans/contribution-workflow.md`](plans/contribution-workflow.md): a
completed feature is a branch and a pull request, a defect is an issue,
and the two are never the same artifact.

[The open list](https://github.com/antonellof/frink/issues) is the live
answer. A table here goes stale, and this one did: six of its eight
rows were closed while it still called them open. Four issues are open
today, all of them performance:

| # | What |
|---|---|
| [#259](https://github.com/antonellof/frink/issues/259) | CUDA prefill is host-bound: 196 synchronous round trips and 3.1 GB over PCIe per `pp512` step |
| [#133](https://github.com/antonellof/frink/issues/133) | CUDA decode is memory-bound: the matvec kernels reach 5.3% of card bandwidth where llama.cpp reaches 60.4% |
| [#61](https://github.com/antonellof/frink/issues/61) | No KV store evicts behind a sliding window, so a windowed model costs full-attention memory |
| [#27](https://github.com/antonellof/frink/issues/27) | CPU decode is scheduling-bound: rayon fork-join per operation where llama.cpp uses a persistent spin-barrier pool |


## Speed gaps against llama.cpp

Closed since the last pass over this list: bench computes the
`lm_head` for the last token only, CPU Q4_K batch GEMM
(`gemm_q4_kx8_group` in `weight_matrix`), SmolLM2 Metal greedy lm_head, OLMoE Metal gather plus `mul_mm_sg` and
Q4_0 `mul_mv_id`, the Qwen shared-expert loader fallback, CPU MoE
token-to-expert bucketing (`moe_ffn_batch`), and dense Metal prefill
stack QKV bias with QK-norm (Qwen2.5 / Qwen3 / Gemma-3).

Still open (see the Open section of
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)):

- **Metal prefill.** The Qwen2.5 / Qwen3 / Gemma-3 Q8_0 dense stack now
  includes QKV bias and QK-norm, which took it from roughly 18 to 21×
  (hybrid CPU projection) down to 1.2 to 2.1×. What remains: OLMoE's
  gather, `mul_mm_sg` and scatter against a fused `kernel_mul_mm_id`,
  dense 1 to 3B models (about 1.5 to 3×), and a compiled graph or
  pre-encoded command buffer replay to get sub-1.5B models to 1× or
  better.
- **CPU prefill and decode.** Phi-4 and Mistral Q4_K `pp512` need
  measuring again after the GEMM change, then i8mm SMMLA if the gap is
  still above 1×, and a persistent decode threadpool.
- **Correctness.** Gemma-4 end-to-end chat smoke test. The older
  Gemma-2 Metal greedy check stays in the tree for regressions only and
  is not part of the published suite.

## Where the project goes next

This section is the reader-facing summary of the plans.
[`plans/north-star.md`](plans/north-star.md) holds the rule that ranks
the work, and the open items are tracked as
[GitHub issues](https://github.com/antonellof/frink/issues). The order
is:

| # | Theme |
|---|---|
| 1 | Fix what is wrong, and close the oracle's hole |
| 2 | Model layer: audit, vocabulary, then split |
| 3 | Close the 9 unaudited architectures, auditing outward |
| 4 | Out-of-core MoE, and one large real checkpoint |
| 5 | CPU decode scaling |
| 6 | Hardware reach: the backend seam, then Vulkan |
| 7 | The rest: embeddings, quants, serving, measurement |

The list below describes what those steps mean for someone running the
engine rather than working on it.

1. **Run bigger models on the same hardware.** Make Qwen3 35B-A3B Q5
   usable on a box that today handles Q4, or an 8B. Most of the items
   below serve this goal.
2. **RAM and VRAM optimization.** Residency planning already exists
   (`frink inspect-plan`). What is missing is acting on it firmly
   enough to change which models fit: tighter KV (3-bit, quantized CTK),
   streaming expert residency, and not materializing activations
   nothing reads.
3. **Hybrid CPU/GPU, especially for MoE.** Routed experts are the
   natural split. Hot experts stay resident on the GPU, cold ones get
   streamed or run on CPU. `PlacementPlan` and `ExpertStore` are the
   groundwork.
4. **CUDA performance.** The batched GEMM (`mul_mm`) first ran on a
   GPU on 2026-09-15: all 13 hardware tests passed on an RTX 3090 and
   `frink verify --backend cuda` was token-identical on five quant
   kinds. Since 0.22.0 it runs on the tensor cores on `sm_80` and up,
   and CUDA prefill keeps dense layers on the device. The RTX 3090 row
   is still about 4.3× behind llama.cpp on prefill and 2.75× on decode
   ([#259](https://github.com/antonellof/frink/issues/259),
   [#133](https://github.com/antonellof/frink/issues/133)). Decode is
   limited by memory-request concurrency rather than arithmetic, so the
   work there is the matvec's load pattern, not a faster inner loop.
5. **Tool calling and full OpenAI API compatibility.** See
   [`API.md`](API.md). GBNF grammars and the lazy grammars behind a
   forced `tool_choice` both ship. So does
   `response_format: json_schema`, through the same converter a forced
   `tool_choice` compiles its arguments with. A forced `tool_choice`
   now reaches ten of the eleven wire formats. What is left is the
   eleventh (`muse_glimmer`,
   [#29](https://github.com/antonellof/frink/issues/29)) and MCP
   invocation.
6. **Docker images**, so evaluating any of this stops requiring a Rust
   toolchain.

**Models**

- Vision: a projector plus generate. Nothing exists for it today
- MiniMax-M3's sparse attention
- The remaining unaudited graphs: a per-token adapter selection
  (`graniteswitch`), a delta-net over a hybrid memory index
  (`qwen4exp`), a second expert bank (`grovemoe`), and the seven the
  2026-10-28 pin move brought in, less `lfm3`, which closed the same
  day, and `kolibri1`, which closed the next on a second selection
  rule: `flint` is a tokenizer away, and `beam`,
  `maion-coder`, `k2-horizon` and `muse-spark` need new code
  ([`plans/parity-audit-2026-10-28.md`](plans/parity-audit-2026-10-28.md))
- Real GLM-5.2, DeepSeek V4 and full Kimi, run end to end on a
  checkpoint rather than a fixture
- MTP draft heads
- Qwen2-MoE and Mixtral pins, once the GGUF fits Host B

**Serving**

- Tool calling. Eleven wire formats parse, and a forced `tool_choice`
  compiles a lazy grammar from the request's own `tools` on ten of
  them. `muse_glimmer` is the eleventh and answers 501 by design, and
  #29 records why. What is left: argument deltas for the
  six JSON-payload formats, and streamed tool calls on the
  continuous-batching path
- MCP tool invocation
- The rest of the OpenAI API surface. On the generation wires only
  `use_beam_search`, `prompt_embeds` and `suffix` are still refused,
  each on its merits. What is missing is whole endpoints
  (`/pooling`, `/classify`), not fields (see [`API.md`](API.md))
- Docker images (CPU, Metal and CUDA variants)
- Throughput measurement for concurrent continuous-batching requests
  (the Metal parallel fix shipped in 0.15.2, and the CB garbled-output
  fix and Host B serving receipts in 0.15.3; see
  [`plans/archive/metal-parallel-concurrency.md`](plans/archive/metal-parallel-concurrency.md)
  and [`benchmarks/receipts/serving/`](../benchmarks/receipts/serving/))
- Full KV layer offload, multi-GPU, tensor parallel, PD disaggregation

**Beyond the llama.cpp surface**

The goal is to replace llama.cpp: the same models, the same command
shapes, the same or better performance. That is the baseline, and the
serving-engine features below go beyond it. They are ranked by
what they buy on hardware people own. Each is a feature written here
rather than ported: the shapes are public, and the implementations are
frink's.

- **Multimodal.** A vision tower and a projector beside the text
  decoder, and the image preprocessing to feed it. It is the largest
  single item on this list and the one llama.cpp also has, so it comes
  first.
- **Speculative decoding over HTTP.** The engine half is built,
  lossless and tested, but the server cannot reach it; see
  [`plans/server-speculative-decoding.md`](plans/server-speculative-decoding.md).
  A multi-token-prediction drafter is a second `Drafter` impl for the
  architectures whose MTP blocks frink already skips by name
  (`mtp_blocks::NEXTN_READERS`).
- **Quantized safetensors, and quantizing on load.** Reading FP8
  blockwise, GPTQ, AWQ, MXFP4 and NVFP4 checkpoints directly, and
  quantizing a BF16 checkpoint while loading it rather than converting
  first. Both sit behind a generic `config.json` to `ModelConfig`
  loader that does not exist yet. `frink-models` reads safetensors only
  for one dedicated stack and one pooler.
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
  function now (`frink_models::devices`), so adding it takes one edit
  instead of two. However, `frink-vulkan` is reachable only through
  `frink-core`'s optional feature, and that plumbing is a separate
  item.

**KV cache and memory**

- A 3-bit KV dtype. The Metal Hadamard rotation on the CTK path has shipped: `--ctk q4_0` rotates K
- Act on the residency plan `inspect-plan` produces: stream cold
  experts, bound the KV budget, report what a host really fits
- Hybrid CPU/GPU expert placement for MoE, the main lever for running a
  larger model or a higher quant on unchanged hardware

**Wiring the ported serving policy**

The ported FreeToken serving policy (see [`FEATURES.md`](FEATURES.md))
is complete and tested. It no longer lives in a crate of its own: the
serving half is `frink-server::policy`, and the MoE expert-residency
half is in `frink-core` beside `expert_store`. Roughly half of it is
now in use: the two output parsers, the withhold rule, the
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
- Make paged KV, and the radix cache built on it, correct on CUDA.
  Metal is done. The prefill used to leave K/V on the device while the
  paged prefill copied host placeholders into the page store; the
  prefill now downloads the real rows, the startup refusal is lifted, and
  `paged_metal_parity` pins identical greedy ids against the contiguous
  cache on a dense, an MoE and a sliding-window model. CUDA has no
  equivalent hardware run
- Give the radix cache an aggregate hit rate on `/v1/stats` and
  `/metrics`, and an eviction budget. `RadixCache::evict` is written and
  tested but nothing calls it, so back pressure today comes from the
  page store running out
- Make prefix reuse work on the continuous-batching path. Paged KV and
  continuous batching are separate switches, and sharing only happens
  under the first
- Size the pools with `frink_core::expert_budget::plan_cache_budget` at
  load, not only when a rebuild asks for a new geometry
- Find consumers for `frink_core::placement` and `residency`, and for
  `policy::anchor`'s `prefill_slide`. The multi-currency prefix cache
  (`cache_manager`, `radix::swa`, `radix::hybrid`, `window_pool`,
  `state_pool`), the cache report renderer and the process supervisor
  were deleted instead of being left to wait: see
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for what each was
  and why it went
- Publish into the radix tree from the continuous batcher too.
  `batch_scheduler` adopts from the tree (`acquire_paged_caches`) but
  never calls `publish_to_radix`, so under `FRINK_CONTINUOUS_BATCHING=1`
  prefix sharing is adopt-only and the tree is filled by nothing

A full recursive re-read of the reference (six readers over its 435
files, checked against every frink crate rather than against the port's
own scope) found 34 further items, now tracked individually in the plan
below. One of them is a correctness bug in shipped code, not an
omission: `route_top_k_grouped` implements "k from every group" where
the DeepSeek-V3/GLM rule scores each group by the sum of its top-2
biased scores and then runs one global top-k. That bug is first in the
plan.

Staged plan, including what the port left behind entirely (semantic
anchor checkpoints, the cache manager, the window slide) and what a
CUDA-side parity would actually cost:
[`plans/archive/freetoken-parity.md`](plans/archive/freetoken-parity.md).

**Engineering practice worth taking from llama.cpp**

- Something equivalent to `test-backend-ops`: every kernel checked
  against a CPU reference across shapes and quant kinds, so no backend
  is merged only because it runs fast.
