# Configuration

Prefer CLI flags ([CLI.md](CLI.md)); a flag overrides its variable.
Environment variables are for server deployments and advanced tuning.

Everything Frink reads from the environment is listed here, in three
namespaces:

- `FRINK_*`: operator configuration. Sections
  [Server](#server) through [Kernel-lookup registry](#kernel-lookup-registry).
- `FRINK_METAL_*_TIMING` / `_BARRIER_LOG`: Metal instrumentation that
  prints numbers and changes nothing else. See
  [Metal instrumentation](#metal-instrumentation).
- `FRINK_TEST_*`: fixtures for `#[ignore]`d tests. See
  [Test and development fixtures](#test-and-development-fixtures).

llama-server's own `LLAMA_ARG_*` variables are read as the flags they
are declared on; see [CLI.md](CLI.md#server-flags).

## Server

| Variable | Purpose |
|---|---|
| `FRINK_MODEL_PATH` | GGUF path (or Kimi dir); same as `-m`. An encoder-only checkpoint (WordPiece / `bert`) serves `/v1/embeddings`, and the generating routes answer 501 naming the model |
| `FRINK_EMBEDDING_MODEL_PATH` | An encoder served alongside a generative model in one process. Wins over an active encoder |
| `FRINK_MODEL_DIR` | Extra directory `GET /admin/models` scans and `POST /admin/download` writes into. Default: the directory holding `FRINK_MODEL_PATH`; with neither, downloads are refused (`412`) |
| `FRINK_MODEL_NAME` | The model name in `/v1/models` and every response's `model` field. Same as `--alias` |
| `FRINK_ADDR` | Bind address, e.g. `127.0.0.1:8383`. `--host` / `--port` win over it |
| `FRINK_LORA` | Comma-separated `path:scale` LoRA specs, what `--lora` / `--lora-scaled` set. Applied at every model load, hot-swaps included, or the checkpoint is refused by name |
| `FRINK_LORA_INIT_WITHOUT_APPLY` | `1`, load the adapters at scale 0 until `POST /lora-adapters` sets them (`--lora-init-without-apply`) |
| `FRINK_ENDPOINT_SLOTS` | `0` refuses `GET /slots`. Default on. `--slots` / `--no-slots` set it |
| `FRINK_SPM_INFILL` | `1` assembles `/infill` prompts suffix first (`--spm-infill`) |
| `FRINK_LOGICAL_BATCH` | The `n_batch` `/infill` budgets its prompt with. Set by `-b`; default 2048 |
| `FRINK_EXIT_ON_STDIN_CLOSE` | `1`, exit on stdin EOF (`--exit-on-stdin-close`). Off by default so a `/dev/null` stdin does not stop the server |
| `FRINK_TOKIO_WORKERS` | `frink-server` async worker threads (default `2`). `--threads-http` sets it |
| `FRINK_CACHE` | Where `-hf` puts downloads. Default `$XDG_CACHE_HOME/frink`, else `~/.cache/frink`; models go under `hub/<owner>__<repo>/`. llama.cpp spells this `LLAMA_CACHE` |

## Hugging Face

Read by `-hf`, `frink download`, `frink pull` and `POST /admin/download`
(see [API.md](API.md)).

| Variable | Purpose |
|---|---|
| `HF_TOKEN` · `HUGGING_FACE_HUB_TOKEN` | Bearer token for gated or private repos. First one set wins. `-hft` / `--hf-token` sets `HF_TOKEN` |
| `HF_ENDPOINT` | Hub base URL for a mirror or an air-gapped cache. Default `https://huggingface.co` |

## Backends

Usually set by `-dev` / `-ngl`. Set these when embedding Frink as a
library or overriding the CLI. The `FRINK_CUDA_*` A/B levers are read
once per process, and any value other than the ones listed is the
default.

| Variable | Purpose |
|---|---|
| `FRINK_METAL` | `1` / `0` / `auto`, Metal offload |
| `FRINK_METAL_ATTN` | `1` / `0`, fused Metal attention with the KV cache on the device. `-dev` / `-ngl` set it to `1` on Metal unless already set (so `FRINK_METAL_ATTN=0 -ngl 99` is the unfused Metal path); `-dev cpu`, `-dev cuda` and `-dev vulkan` set it to `0` |
| `FRINK_CTK` | KV cache type, what `-ctk` / `--ctk` sets (`--ctk` wins); values per backend under [`--ctk` on each backend](CLI.md#completion-flags). `frink parity`, `frink bench` and `frink-server` read it too |
| `FRINK_CUDA` | `1` / `0` / `auto` (build with `--features cuda`) |
| `FRINK_CUDA_MUL_MM` | Batched prefill GEMM body. Default on `sm_80`+: Q8_0 and Q4_K take the int8 GEMM (llama.cpp's MMQ), other kinds the f16 tensor-core body. `tc` keeps every kind on the tensor-core body, `simt` on the f32 SIMT one |
| `FRINK_CUDA_PREFILL_ATTN` | `simt` keeps prefill attention on the f32 SIMT kernel where an `sm_80`+ device would take the tensor-core kernel (head widths 64, 128, 256) |
| `FRINK_CUDA_MOE_MMQ` | `1` runs routed prefill expert projections on the int8 grouped GEMM (Q8_0, Q4_K, MXFP4; `sm_80`+) instead of the f16 tensor-core grouped GEMM. Off by default (slower on an A100) |
| `FRINK_CUDA_MMQ_COLS` | `64` or `128` holds every dense int8 prefill GEMM at that token tile. Default: Q8_0 takes 128 where those tiles fill every multiprocessor, else 64; other kinds 64 |
| `FRINK_CUDA_MATVEC` | CUDA decode matvecs. Unset (or `dp4a`): Q4_K, Q6_K, Q8_0 and MXFP4 decode as llama.cpp's `mmvq` does (activation quantized once to q8_1, `dp4a` dots, llama.cpp's launch geometry), putting decode on llama.cpp CUDA's rounding (ADR 0009); other kinds run f32 coalesced matvecs. `wide` keeps every kind on the f32 kernels, `narrow` too at one block per warp per loop trip |
| `FRINK_CUDA_MATVEC_WPR` | `1`, `2`, `4` or `8` warps per output row for every coalesced decode matvec. Unset: one warp per trip of a row's blocks up to 4, and 8 for a long row of a matrix of at most 8192 rows (`matvec_kinds::coalesced::wpr_for`) |
| `FRINK_CUDA_DECODE` | `host` keeps every CUDA decode token on the per-op path (each projection a synchronous round trip, the rest on the host). Default: dense layers run through the resident stack at one row, with one download per run of layers. `off` / `0` mean the default |
| `FRINK_CUDA_DEVICE_ROUTE` | `0` routes every CUDA decode token's MoE layers on the host. Default: on the device when gating is plain softmax top-k (optionally renormalised) or gpt-oss's, the experts share one matvec kind, and there are at most 256 experts and 16 active. Prefill routes on the host either way |
| `FRINK_CUDA_GQA` | `1` serves the per-token GQA reduction from the CUDA `gqa_decode` kernel, falling back to the host on a launch error. Off by default until its hardware tests run |
| `FRINK_CUDA_GRAPH` | `1` replays a CUDA decode token of the resident stack from a captured CUDA graph (position and input written to fixed buffers before each replay). A run of layers captures after one eager token with every routed layer on the device, and again when its KV mirror grows (from 256 rows, doubling). Prefill, a run with a host-routed layer, and a device below `sm_70` or without stream-ordered allocation run eagerly; a failed capture says why once on stderr. Unset or `0` is eager; any other value is refused by name. Off by default until its hardware test (`prefill::decode_graph`) runs on a GPU |
| `FRINK_CUDA_COUNT` | `1` prints to stderr the host waits and the H2D, D2H and D2D copies each CUDA decode token and prefill step made, when first seen and whenever the count changes. Counting is always on (`frink_cuda::xfer`), so a counted run measures the same binary. Process-wide |
| `FRINK_CUDA_TIME` | `1` prints where each CUDA prefill step's host wall time went (describing, enqueueing, waiting for the download, pushing K/V to the host, logits), and the same means per decode token every 64 tokens |
| `FRINK_VULKAN` | `1` / `0` / `auto` (build with `--features vulkan`). `--device vulkan` sets `1`; `--device cpu`, `none` and `-ngl 0` set `0`. **Unset is off**: the Vulkan path (kinds in `docs/KERNELS.md`) is correct but not yet measured against the CPU |
| `GGML_VK_VISIBLE_DEVICES` | llama.cpp's variable, read the same way: comma- or space-separated indices into the loader's device list. Unset is every discrete and integrated GPU; frink runs on the first. The `-dev` row of [CLI.md](CLI.md#deliberate-differences-from-llamacpp) lists the checks it does not make |
| `FRINK_VULKAN_LOADER` | Path to a `libvulkan`, when the platform default is not found (the error names this variable) |
| `FRINK_HIP_LIB` · `FRINK_HIPRTC_LIB` | Paths to `libamdhip64` and `libhiprtc` for `frink-rocm`, which is not a backend yet |
| `FRINK_KV_WINDOW` | On by default: a sliding-window layer's host KV keeps its last `window` rows plus slack, as llama.cpp's `swa_full = false`. `0` keeps every row. See [`FRINK_KV_WINDOW`](#frink_kv_window) |
| `FRINK_CPU_THREADS` | Worker threads; same as `-t`. Default: performance cores (`hw.perflevel0.physicalcpu` on macOS), as llama.cpp |
| `FRINK_CPU_POOL` | A/B override of the CPU scheduler. Unset: a single-token decode step uses a persistent spin-then-park pool (as llama.cpp's `ggml_threadpool`) when its widest layer matvec (LM head excluded) is at least `par::policy::SPIN_MIN_OP_MACS` multiply-accumulates, else rayon fork-join; other operations choose per operation by the same constant. `spin` / `persistent` / `1` / `on` pins the pool, `rayon` / `0` / `off` pins fork-join. Output is token-identical on all settings |
| `FRINK_CPU_POOL_SPIN_US` | Microseconds a `spin` worker spins before parking (default 100) |
| `FRINK_CPU_INT_DOT` | int8×int8 matvec + repacked GEMV/GEMM. On by default in `frink` / `frink-server` (`0` opts out), off in the library so golden cross-validation stays reference-exact. aarch64 takes both halves; x86_64 with AVX2+FMA takes only the batched GEMM (Q4_K, Q5_K, Q6_K, Q8_0, Q4_0) and keeps decode on the AVX2 f32 dot; x86 without AVX2 takes neither. Repacked copies are bounded by `FRINK_REPACK_CACHE_BYTES` |
| `FRINK_METAL_FA_VEC` | `0` disables FA-vec attention for decode and prefill, falling back to the online-softmax GQA kernel. Default on for `head_dim` in {64, 96, 128, 256}. Prefill at 64 / 128 / 256 with at least 8 new tokens takes the simdgroup-MMA `flash_attn_ext` kernel, or on an M5 the tensor-API `flash_attn_ext_tensor` where llama.cpp picks it (at least 20 new tokens, 32 for a quantized cache, and `ceil(n/32) * n_heads * head_dim >= 8192`); `FRINK_METAL_TENSOR=0` turns that off with the tensor GEMM |
| `FRINK_METAL_TENSOR` | The Metal 4 tensor API (`mpp::tensor_ops::matmul2d`) for the prefill GEMMs (dense `mul_mm`, MoE `mul_mm_id`) and prefill flash attention, llama.cpp's `GGML_METAL_HAS_TENSOR` kernels. Unset: on where llama.cpp turns it on (a Metal 4 GPU named M5, M6, A19 or A20) and a probe kernel compiles. `0` turns it off (llama.cpp's `GGML_METAL_TENSOR_DISABLE`), `1` turns it on for an older Metal 4 chip (`GGML_METAL_TENSOR_ENABLE`) |
| `FRINK_METAL_SCRATCH_BUDGET_BYTES` | Ceiling on pooled Metal scratch buffers (default 768 MiB); past it a returned buffer is dropped |
| `FRINK_METAL_WEIGHT_CACHE_BYTES` | Ceiling on the resident Metal weight-buffer cache. Default (and any unparseable value) is unlimited; cap it on a small machine |
| `FRINK_CPU_KV_OFFLOAD` | `1`, sync Metal KV to host after each decode step |

## Tuning

| Variable | Purpose |
|---|---|
| `FRINK_CONTINUOUS_BATCHING` | `1` turns on the batched decode worker, `0` keeps the private decode loop; unset is off. `-cb` / `--cont-batching` and `--parallel N` set `1`, `--no-cont-batching` `0`. Beside a KV pool or `FRINK_PREFIX_CACHE_ENTRIES` (without paged KV), or on a dedicated engine (Kimi, MLA, GLM-5.2), the server refuses to start, naming the conflict; paged KV composes with it |
| `FRINK_CB_MAX_SEQS` | Cap on in-flight sequences (`-np N` / `--parallel N`). Default unlimited. Reported as `frink_scheduler_max_seqs` on `/metrics` |
| `FRINK_CB_PREFILL_CHUNK`, `FRINK_CHUNKED_PREFILL` | Prompt tokens per forward pass, on the batch scheduler and the private loop respectively. On the scheduler it is the granularity at which a long prompt yields to decodes. `-b` / `-ub` set both (the smaller of the two). Defaults: 128 on the scheduler, unchunked on the private loop |
| `FRINK_CB_MAX_QUEUE` | Requests allowed to wait for admission (default `512`); past it, `503` + `Retry-After` |
| `FRINK_CB_MAX_BATCH_TOKENS` | Tokens one tick may run, prompt and generated together (default `256`). The prefill chunk shrinks by the rows decoding in the same tick, never below 16 prompt tokens |
| `FRINK_CB_KV_BLOCKS` | Total KV blocks the scheduler may hand out. Unset: derived at load alongside `FRINK_CB_MAX_CONTEXT`, or absent when the model cannot be priced. A request reserves `ceil((prompt + max_tokens) / block_size)` blocks for its lifetime |
| `FRINK_CB_KV_BLOCK_SIZE` | Token positions per KV block (default `256`). A request bigger than the whole budget gets `400` (`code: device_memory_budget_exceeded`), not `503` |
| `FRINK_CB_PREEMPT_AFTER_TICKS` | Admission passes the next job may wait before the most recently admitted decoding rows are preempted for it (unset: never; `0`: as soon as it does not fit). An evicted row keeps its output and sampler state and is re-prefilled from prompt plus generated tokens, so its stream is unbroken. Each row is preempted at most 4 times. Rows on a sliding-window paged lease are never evicted. Counted as `frink_scheduler_preempted_total` on `/metrics` |
| `FRINK_CB_MAX_CONTEXT` | Token positions (prompt + `max_tokens`) one request may ask for, on both decode paths; `-c` sets it. Over it, `400` with `code: context_length_exceeded`, `estimated_bytes` and `limit_bytes`. Unset: derived at load from weights + per-token KV against the device budget, capped at the trained context; no ceiling when the model cannot be priced |
| `FRINK_MODEL_DRAFT` | Draft GGUF for speculative decoding, what `-md` / `--model-draft` sets; see [Server flags](CLI.md#server-flags) |
| `FRINK_DRAFT_MAX` | Tokens drafted per verification step (`--draft-max`). Default `3` |
| `FRINK_DRAFT_P_MIN` | Stop a draft block at a token below this probability (`--draft-p-min`). Default `0` |
| `FRINK_SLOT_SAVE_PATH` | Directory for slot files (`POST /slots/{id}?action=save\|restore`), what `--slot-save-path DIR` sets. Unset, the route answers 501. Requires `FRINK_PREFIX_CACHE_ENTRIES` |
| `FRINK_CHAT_TEMPLATE` | A chat template in place of the checkpoint's: Jinja source, or `chatml`. Set by `--chat-template` / `--chat-template-file` |
| `FRINK_CHAT_TEMPLATE_KWARGS` | JSON object of template variables for every render, under a request's own. Set by `--chat-template-kwargs`, `-rea`, `--reasoning-effort`, `--reasoning-preserve`. Default `{"preserve_reasoning": true}` |
| `FRINK_REASONING_BUDGET` | Default `reasoning_budget_tokens`: `-1` unrestricted (default), `0` end the thought at once, `N` tokens. `--reasoning-budget N` |
| `FRINK_PREFILL_ASSISTANT` | Continue a trailing assistant message by default. `0`/`false` off. `--prefill-assistant` / `--no-prefill-assistant`. Default on |
| `FRINK_SSE_ORPHAN_TIMEOUT_MS` | How long a streaming send waits for a stalled client before the generation is cancelled (default `30000`; `0` disables) |
| `FRINK_KV_POOL_BLOCKS` | KV block-pool size (blocks). Bounds the KV all requests may hold; each request still owns a private buffer |
| `FRINK_KV_POOL_BLOCK_SIZE` | Tokens per KV pool block |
| `FRINK_KV_POOL_QUEUE_TIMEOUT_MS` | How long a request waits for free KV before rejection (default `0`). Applies to the pool and paged KV |
| `FRINK_KV_BYTE_BUDGET` | Byte ceiling for the KV block pool. Exclusive with `FRINK_KV_POOL_BLOCKS` |
| `FRINK_PAGED_KV_BLOCKS` | Blocks per layer of paged KV: shared pages read through a block table. `frink-server` only. Exclusive with `FRINK_KV_POOL_BLOCKS`/`FRINK_KV_BYTE_BUDGET` and `FRINK_PREFIX_CACHE_ENTRIES` (the server stops naming both). Turns on [prefix sharing](#sharing-pages-between-prompts) |
| `FRINK_PAGED_KV_BLOCK_SIZE` | Positions per paged-KV block. Required with `FRINK_PAGED_KV_BLOCKS` |
| `FRINK_PAGED_KV_SLIDE_INTERVAL` | Decode steps between window slides on a paged store (default 128). Applies only when *every* layer slides by the same window |
| `FRINK_PREFIX_CACHE_ENTRIES` | Prefix-cache capacity for the private decode loop: whole KV snapshots in an LRU, reported under `GET /cache/stats`. Each entry is a full copy (Qwen3-8B: about 1 GiB per 7k tokens), so a single agent client wants 1 or 2. A follow-up turn reuses the last one, greedy Metal requests included. Keeps the private loop and is refused beside paged KV |
| `FRINK_EXPERT_CACHE_BYTES` | MoE expert-streaming cache budget; see [Expert streaming](#expert-streaming) |
| `FRINK_SSD_STREAMING` | `1`, stream MoE experts from disk; `0`, never. Unset decides at load; see [Expert streaming](#expert-streaming) |
| `FRINK_REPACK_CACHE_BYTES` | Bytes the interleaved-weight (`repack`) cache may retain, evicting least-recently-used. `0` disables it (every CPU matvec repacks per call). Unset: a quarter of available memory less the fit headroom and `FRINK_EXPERT_CACHE_BYTES` |
| `FRINK_GPU_VRAM_BUDGET_BYTES` | Cap GPU-resident MoE experts (`0` = CPU experts on Metal) |
| `FRINK_DEVICE_BUDGET_BYTES` | Override the probed memory budget the pre-load KV check plans against (Metal `recommendedMaxWorkingSetSize` / free VRAM / host RAM minus a reserve). For container limits and shared GPUs |
| `FRINK_PIN_BUDGET_GB` | Override the page-locking budget for expert-bank placement. Unset: no cap on plain Linux; 40% of physical RAM on WSL |
| `FRINK_BENCHBW_PATH` | Path to this host's bandwidth profile (`frink bench-bw`). Otherwise one file per GPU uuid under `$XDG_CACHE_HOME/frink/benchbw/`, then the legacy `benchbw.json`. A file that does not parse yields no profile |
| `FRINK_DECODE_LOG_INTERVAL` | Decode forwards between batch status lines (default 40; `0` logs every one; unparseable takes the default) |
| `FRINK_ACCOUNTING_OUTBOX` | Directory accounting receipts are written to, atomically and idempotently by receipt id, before `POST /v1/admin/prepare-stop` answers. Unset: no outbox |
| `FRINK_INSTANCE_ID` | Names this engine generation for the receipt id. Unset: the pid plus the process's start time |
| `FRINK_QOS_LOG` | `1`, log each rayon worker's macOS QoS class at pool start |
| `NO_COLOR` | Any value turns off ANSI colour in `frink run` / `frink cli`. Colour is also off whenever stderr is not a terminal |
| `FRINK_COLOR` | `1`, force ANSI colour on even when stderr is not a terminal |
| `FRINK_QUIET` | Any value drops the `loaded in ...` line `frink run` prints to stderr |

### Expert streaming

Off by default; it turns itself on only when the weights will not fit
(checkpoint size against available memory, less 4 GiB for KV and
activations), and logs both figures and the cache size it chose. It is
slower than running resident. `FRINK_SSD_STREAMING=0` forces resident
loading anyway; `FRINK_EXPERT_CACHE_BYTES` sets the budget explicitly
and wins over the automatic choice. A host whose available memory is
unknown resolves to resident.

The budget buys whole slots of the largest expert's size; the load
line says how many (`frink: expert store: N of M routed experts
resident`). The most-picked experts stay resident, and a miss colder
than every resident expert is read uncached. On Metal the slots are one
Shared buffer, not counted again by `FRINK_METAL_WEIGHT_CACHE_BYTES`,
and a streamed layer takes the per-layer top-k launch instead of the
packed whole-stack decode.

### Sharing pages between prompts

Paged KV also turns on a radix prefix cache over the same pages; there
is no separate switch. Each prompt is matched against a tree of
computed prefixes keyed by page; the matched pages are locked and
reference-counted, so many conversations off one system prompt hold one
copy of its KV. Reuse shows as `usage.cached_tokens`. There is no
aggregate hit rate on `/v1/stats` or `/metrics` and no eviction knob:
back pressure is the page store running out plus
`FRINK_KV_POOL_QUEUE_TIMEOUT_MS`.

A request that sends `cache_salt` gets its own tree (same page pool);
eviction takes from each namespace in turn. This is a different
mechanism from `FRINK_PREFIX_CACHE_ENTRIES` (whole copied snapshots);
the two cannot be on at once.

## Security and transport

Everything except `/health` sits behind `FRINK_API_KEY` when it is
set, including `/admin/*`, `/metrics` and `/cache/stats`.

| Var | Effect |
|---|---|
| `FRINK_API_KEY` | Bearer token (`Authorization: Bearer <key>`) required by every route except `/health`, the `/admin` control surface included. `--api-key` / `--api-key-file` set it |
| `FRINK_ALLOW_UNAUTHENTICATED_REMOTE` | `1`, permit binding a non-loopback address with no API key. Without it the server refuses to start that way |
| `FRINK_TLS_CERT` / `FRINK_TLS_KEY` | PEM cert and key; both or neither (one alone is a startup error). Unset is plain HTTP. `--ssl-cert-file` / `--ssl-key-file` set them |
| `FRINK_CORS_ORIGINS` | Comma-separated exact origins; `*` is rejected. Needed to serve Frink Studio (`ui/`) from another origin; `ui/`'s dev server (`npm run dev`) proxies the API instead |
| `FRINK_RATE_LIMIT_PER_MINUTE` | Global request cap. A non-integer value is a startup error |
| `FRINK_JOURNAL_PATH` | Where the process-lifecycle journal is written |
| `FRINK_CONVERSATIONS_DIR` | Where server-side conversations are stored, one JSON file each (default `./frink-conversations`). Nothing is evicted; the caps refuse with a reason |

## Kernel-lookup registry

Every kernel lookup is resolved once at load and recorded; the registry
is then sealed, and a later miss that takes a fallback warns once with
its call site (`frink_core::kernel_registry`).

| Variable | Purpose |
|---|---|
| `FRINK_KERNEL_REGISTRY` | `1`, print the whole load-time kernel table (backend × op × quant kind, with tensor role and fallback). `0`, record nothing. Default: print only misses that run off the selected accelerator |
| `FRINK_ALLOW_UNKNOWN_TENSORS` | `1`, load a checkpoint carrying tensors this build never reads, with a warning. Default is refusal, since an unread tensor is a missing term of the graph |
| `FRINK_ALLOW_MULTIPLE_INSTANCES` | `1`, start even though another frink process holds a model. See [One model at a time](CLI.md#one-model-at-a-time); `--allow-multiple-instances` does the same for one run |
| `FRINK_INSTANCE_DIR` | Where the running-instance registry lives (default `$XDG_CACHE_HOME/frink/instances`, else `~/.cache/frink/instances`) |
| `FRINK_STRICT_KERNELS` | `1`, refuse a model whose weights have no kernel on the selected accelerator instead of running a slower path. Set it in CI and benchmark harnesses |
| `FRINK_ALLOW_UNAUDITED_ARCH` | Run an architecture never verified against llama.cpp on the shared generic-GQA path. Off by default (refused); set it to compare against llama.cpp yourself |

## Metal instrumentation

These print numbers and change nothing else.

| Variable | Purpose |
|---|---|
| `FRINK_METAL_MM_TIMING` | `1`, total the host's wall-clock setup / GPU-wait / readback microseconds across the prefill GEMM paths |
| `FRINK_METAL_GPU_TIMING` | `1`, running mean of GPU-clock milliseconds per tagged submission (`moe-decode/tok`, `dense-decode/tok`, `prefill-dense-stack`), excluding host stalls |
| `FRINK_METAL_KERNEL_TIMING` | `1`, attribute the GPU time of the dense decode stack and the dense prefill stack to dispatch kinds (matvec or GEMM, attention, norm, RoPE, ...; prefill rows are prefixed `pf_`). Each op group gets its own sampled encoder while on, so compare ratios between runs measured the same way |
| `FRINK_METAL_BARRIER_LOG` | `1`, log the running barriers-per-op ratio from `MemRanges` (`1.00` is fully serialised) |

## Test and development fixtures

`FRINK_TEST_*` points an `#[ignore]`d test at a local checkpoint too
large to commit. `cargo test --workspace` passes with none set; the
tests that read them skip.

| Variable | Purpose |
|---|---|
| `FRINK_TEST_MODELS_DIR` | Root the real-GGUF sweeps scan (default `models`), for `bos_policy`, `chat_template_real_gguf`, `paged_metal_parity` and `model_swap_isolation`; useful in a worktree. Unrelated to `FRINK_MODEL_DIR` |
| `FRINK_TEST_GEMMA2_GGUF` | Gemma-2 GGUF for the Metal quality gate |
| `FRINK_TEST_QWEN2MOE_GGUF` | Qwen2-MoE GGUF for the "capital of France" check |
| `FRINK_TEST_SMOLLM2_GGUF` | SmolLM2 GGUF for the same check on Metal |
| `FRINK_TEST_PAGED_PARITY_GGUF` | GGUF for the paged-vs-contiguous KV parity test |
| `FRINK_TEST_RECEIPT_CHECKPOINT` | The pinned Llama-3.1 Q4_K_M GGUF the checkpoint-receipt test hashes |
| `FRINK_TEST_KIMI_SHARD_DIR` · `FRINK_TEST_KIMI_MOE_SHARD_DIR` · `FRINK_TEST_KIMI_TOKENIZER_PATH` | Kimi shards and tokenizer for the real-data tests |
| `FRINK_TEST_INSPECT_PATH` | A real `.gguf` for the tensor-table dump in `frink-gguf` |
| `FRINK_TEST_CHAT_TEMPLATE_NOW` | Pins the clock chat templates read (`strftime_now`, `date_string`, `datetime`) to these Unix seconds, as UTC. Unset, local time. The only one read from library code |
| `FRINK_REQUIRE_VULKAN` | `1` turns a Vulkan test's "no device, skipped" into a failure (CI's lavapipe job) |

Development tools:

| Variable | Purpose |
|---|---|
| `FRINK_LLAMA_LOGITS` | Path to a `llama_logits` reference dumper for `frink parity` (`--dumper`). Otherwise `target/llama_logits`, then `.local-scripts/llama_logits` |
| `FRINK_PRESET` | With `FRINK_MODEL_PATH` unset only: the architecture sketch `frink-server` builds random weights for (default `glm-5.2`), with a warning |

## Removed

Setting these does nothing; the default they chose between is now the
only path.

`FRINK_METAL_MATMUL` · `FRINK_METAL_MUL_MM` · `FRINK_METAL_LOGITS` ·
`FRINK_METAL_GREEDY_GPU` · `FRINK_METAL_WEIGHT_COPY` ·
`FRINK_METAL_PREFILL_FUSE_O` · `FRINK_METAL_FA_EXT` ·
`FRINK_METAL_FA_MMA` · `FRINK_METAL_FA_NQ` ·
`FRINK_METAL_MOE_RESIDENT` · `FRINK_METAL_MOE_STACK` ·
`FRINK_METAL_MOE_ABLATE` · `FRINK_METAL_MOE_FUSED_GATE_UP` ·
`FRINK_METAL_MOE_GATE_THEN_SILU` · `FRINK_METAL_MOE_BARRIER_LOG` ·
`FRINK_GEMV_DEDICATED` · `FRINK_GEMV_THREADS` ·
`FRINK_MIN_TASK_MACS`

## `FRINK_KV_WINDOW`

A sliding-window layer only attends over its last `window` positions,
so the contiguous host store drops older rows per layer: an alternating
model (Gemma-2/3, gpt-oss) evicts in its windowed layers (those
`attention.sliding_window_pattern` or the per-layer array names) and
keeps everything in its full-attention ones. On Gemma-3-4B at 32768
tokens of host f32 KV:

| | bytes |
|---|---|
| `FRINK_KV_WINDOW=0` (every layer holds everything) | 9,126,805,504 |
| Default, at rest | 1,678,098,432 |
| Default, admission ceiling (prefill peak) | 1,962,934,272 |

A windowed layer's rows cycle between `window + reserve` and
`window + slack` (half a window of slack, half of it a rollback
reserve). Admission and `--ctx-size auto` price each windowed layer at
the top of that cycle (`KvWindow::max_rows`, via `KvBudget`), so a run
with the switch on is offered the context it can really carry. Output
is token-identical with it on or off.

How state that moves backwards is handled:

- **Speculative decoding.** Every speculative caller (the prompt-lookup
  drafter, `--model-draft`) bounds its draft by
  `KvCache::rollback_limit`. A window too small for the server's
  5-token draft decodes without speculating.
- **The prefix cache** stores windowed caches and reuses one only for a
  match it can still truncate to; a request diverging further back than
  the reserve is a miss, counted as `refused_windowed`.
- **`n > 1`** forks clone the caches, window included.
- **Slot files** hold a whole history, so a slot save keeps every row.
- **Context shift** is refused before any of this matters.

It turns itself off under `FRINK_METAL_ATTN` (on by default on Metal;
`FRINK_METAL_ATTN=0` attends on the host and evicts as the CPU does),
and on a draft model's decoder. The paged store
(`FRINK_PAGED_KV_BLOCKS`) and the KV pool (`FRINK_KV_POOL_BLOCKS`) do
not evict, and are still priced at the full number
([#61](https://github.com/antonellof/frink/issues/61)), as is the
server's own pre-load ceiling. The byte figures in a context-length
refusal (`ContextCeiling::bytes_for`) are the unwindowed ones; the
decision itself uses the windowed ceiling.
