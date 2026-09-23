# Configuration

Prefer CLI flags ([CLI.md](CLI.md)). Environment variables are for server
deployments and advanced tuning. Flags override env when both are set.

Everything Frink reads from the environment is listed here. Three
namespaces, and the prefix tells you which:

- `FRINK_*`: operator configuration. Sections
  [Server](#server) through [Kernel-lookup registry](#kernel-lookup-registry).
- `FRINK_METAL_*_TIMING` / `_BARRIER_LOG`: Metal instrumentation. They
  print numbers and change nothing else. See
  [Metal instrumentation](#metal-instrumentation).
- `FRINK_TEST_*`: fixtures that point an `#[ignore]`d test at a local
  checkpoint. Not configuration; see
  [Test and development fixtures](#test-and-development-fixtures).

## Server

| Variable | Purpose |
|---|---|
| `FRINK_MODEL_PATH` | GGUF path (or Kimi dir); same as `-m`. An encoder-only checkpoint (WordPiece / `bert`) is accepted here and serves `/v1/embeddings`; the generating routes then answer 501 naming the model |
| `FRINK_EMBEDDING_MODEL_PATH` | An encoder served ALONGSIDE a generative model in one process, which the single active-model slot cannot express. Wins over an active encoder, since explicit config beats a hot-swap |
| `FRINK_MODEL_DIR` | Extra directory `GET /admin/models` scans, and the one `POST /admin/download` writes into. Without it, the directory holding `FRINK_MODEL_PATH` is used; with neither, downloads are refused (`412`) rather than guessing a location |
| `FRINK_ADDR` | Bind address, e.g. `127.0.0.1:8383` |
| `FRINK_API_KEY` | Require `Authorization: Bearer <key>`. Also gates the whole `/admin` control surface, which can swap models and write files |
| `FRINK_LORA` | Comma-separated `path:scale` LoRA adapter specs, what `--lora` / `--lora-scaled` lower to. Read at every model load, so a hot-swapped checkpoint gets the same adapters or is refused by name |
| `FRINK_LORA_INIT_WITHOUT_APPLY` | `1`, load the adapters at scale 0 until `POST /lora-adapters` sets them (`--lora-init-without-apply`) |

## Hugging Face

Read only by `POST /admin/download` (see [API.md](API.md)).

| Variable | Purpose |
|---|---|
| `HF_TOKEN` · `HUGGING_FACE_HUB_TOKEN` | Bearer token for gated or private repos. First one set wins |
| `HF_ENDPOINT` | Hub base URL for a mirror or an air-gapped cache. Default `https://huggingface.co` |

## Backends

Usually set by `-dev` / `-ngl`. Set these only when embedding Frink as a
library or overriding the CLI.

| Variable | Purpose |
|---|---|
| `FRINK_METAL` | `1` / `0` / `auto`, Metal offload |
| `FRINK_METAL_ATTN` | `1` / `0`, fused Metal attention + resident KV |
| `FRINK_CTK` | KV dtype, llama.cpp's `-ctk` set plus frink's `fp8`; served are `f16` (default), `q8_0`, `fp8` and `q4_0`, and the rest fall back with a printed note. Same as `--ctk`, and like it **Metal only**: the CPU and CUDA KV cache is the host `Vec<f32>`. `--ctk` wins when given, this variable when it is not |
| `FRINK_CUDA` | `1` / `0` / `auto` (build with `--features cuda`) |
| `FRINK_CUDA_MUL_MM` | `simt` keeps the batched GEMM on the f32 SIMT body where the device (`sm_80`+) would take the tensor-core body. An A/B lever, not a tuning knob: the tensor-core body is the faster one wherever it runs |
| `FRINK_VULKAN` | `1` / `0` / `auto` (build with `--features vulkan`). `Q8_0` matvec only and no GEMM, so a prefill stays on the host |
| `FRINK_VULKAN_LOADER` | Path to a `libvulkan` the loader should use. Only needed when the platform default is not found; the error names this variable |
| `FRINK_MODEL_NAME` | What the served model is called in `/v1/models` and every response's `model` field. Same as `--alias`. Read in one place, so it cannot apply to some routes and not others |
| `FRINK_CACHE` | Where `-hf` puts downloaded checkpoints. Default `$XDG_CACHE_HOME/frink`, else `~/.cache/frink`; models go under `hub/<owner>__<repo>/`. llama.cpp spells this `LLAMA_CACHE` |
| `FRINK_KV_WINDOW` | `1` evicts KV rows behind a sliding window on the CPU contiguous store. **Off by default.** Gemma-3-4B at 32768 tokens holds 1.69 GiB at rest instead of 9.13 GiB, with a 1.96 GiB admission ceiling, and `--ctx-size auto` is priced against that ceiling rather than the unwindowed one. Output is token-identical either way, tested on a real quantized SWA checkpoint. Turns itself off under `FRINK_METAL_ATTN`, and does not apply to a draft model; disables the prefix cache, which cannot represent a windowed cache. See the note below |
| `FRINK_CPU_POOL` | **An A/B override, not the decision.** Unset, the scheduler is chosen **per operation from its size**: an operation carrying at least `par::policy::SPIN_MIN_OP_MACS` multiply-accumulates (2.1M, i.e. `rows x cols`) runs on a persistent pool parked on a spin-then-park barrier, the shape llama.cpp's `ggml_threadpool` uses; anything smaller forks and joins with rayon as before. `spin` / `persistent` / `1` / `on` pins the pool at every size; `rayon` / `0` / `off` pins fork-join at every size and is the exact revert. The rule exists because the pool is not uniformly better, **measured on 2026-09-04**: on a quiet 20-core Cortex-A725 (aarch64, `tg128`, 19 threads) it is **+123% at 3B and +87% at 8B**, taking decode from LOSING to llama.cpp to beating it (23.14 vs 17.86 tok/s at 3B, 12.41 vs 9.06 at 8B), and **-37% at 135M**, reproducibly and on quiet hosts, so it is not contention. On a quiet 10-core Xeon it is +49% / +23% / +15% at 135M / 3B / 8B. The crossover constant is *bracketed* by those model-level numbers rather than swept; see its doc comment. Output is token-identical on all settings, which is tested |
| `FRINK_CPU_POOL_SPIN_US` | Microseconds a `spin` worker spins before parking. Default 100. A pool that never parks burns a core per thread on an idle server |
| `FRINK_CPU_THREADS` | Worker threads; same as `-t`. Default: **performance cores** (`hw.perflevel0.physicalcpu` on macOS), matching llama.cpp, not logical cores |
| `FRINK_CPU_INT_DOT` | int8×int8 matvec + repacked GEMV/GEMM. **On by default** in `frink` / `frink-server`; `0` opts out. Off in the library so golden cross-validation stays reference-exact. It is the master switch, not the whole rule: which half of the tier a workload takes is per architecture. **aarch64** takes both halves. **x86_64 with AVX2+FMA** takes the batched GEMM (prefill) and leaves decode on the AVX2 f32 dot, because the int8 matvec measured 4x to 8.8x slower there (#127) while the batch half has AVX2 kernels for Q4_K, Q5_K, Q6_K, Q8_0 and Q4_0 (#152). An x86 host without AVX2 takes neither. Either half repacks weights into the interleaved layout, so the retained copies are bounded by `FRINK_REPACK_CACHE_BYTES` below — on x86 this is what starts spending that budget, since nothing repacked there before |
| `FRINK_METAL_FA_VEC` | `0`, disable llama-style FA-vec for decode **and** prefill and fall back to the legacy online-softmax GQA. Default **on** for `head_dim` in {64, 96, 128, 256}; other widths take the legacy kernel either way. Prefill at 64 / 128 / 256 with at least 8 new tokens goes further and takes the simdgroup-MMA `flash_attn_ext` kernel, which is not separately switchable |
| `FRINK_METAL_SCRATCH_BUDGET_BYTES` | Ceiling on the pooled Metal scratch buffers (default 768 MiB). Past it a returned buffer is dropped rather than kept for reuse. Lower it on a small machine where the pool competes with the weights |
| `FRINK_METAL_WEIGHT_CACHE_BYTES` | Ceiling on the resident Metal weight-buffer cache. Default is effectively unlimited, which is right on unified memory; cap it on a small machine. An unparseable value also means unlimited |
| `FRINK_CUDA_GQA` | `1`, serve the per-token GQA reduction from the CUDA `gqa_decode` kernel instead of the host path, falling back to the host on any launch error. Off by default: the kernel's numerical parity is gated on hardware tests that need a real GPU |
| `FRINK_CUDA_GRAPH` | `1`, request CUDA-graph capture and replay for decode. Nothing enqueues into a captured stream yet, so today this changes nothing; it is groundwork with a pending hardware receipt (`docs/ROADMAP.md`) |

## Tuning

| Variable | Purpose |
|---|---|
| `FRINK_CONTINUOUS_BATCHING` | `1` enables, `0` disables. When unset on Metal builds with fused attention, continuous batching is **on by default** for safe parallel serving. Also: `frink serve --cont-batching` / `-cb`, `--no-cont-batching`. |
| `FRINK_CB_MAX_SEQS` | Continuous batching: cap on in-flight sequences (llama.cpp `-np`). CLI: `-np N` / `--parallel N`. Default: unlimited. Reported as `frink_scheduler_max_seqs` on `/metrics` |
| `FRINK_CB_PREFILL_CHUNK`, `FRINK_CHUNKED_PREFILL` | Prompt tokens per forward pass, on the batch scheduler and the private decode loop respectively. One number with two spellings; `-b` / `-ub` set both from one value (the smaller of the two flags, llama.cpp's own rule). Defaults: 128 on the scheduler, unchunked on the private loop |
| `FRINK_SLOT_SAVE_PATH` | Directory for slot files (`POST /slots/{id}?action=save\|restore`). CLI: `--slot-save-path DIR`. Unset means the route refuses with a 501 naming the flag. Requires `FRINK_PREFIX_CACHE_ENTRIES`, which is where a restored slot lives |
| `FRINK_REASONING_BUDGET` | Server default for `reasoning_budget_tokens`: `-1` unrestricted, `0` end the thought as it opens, `N` tokens of thought before the closer is forced. CLI: `--reasoning-budget N`. Default: `-1` |
| `FRINK_PREFILL_ASSISTANT` | Whether a trailing assistant message is continued by default (`continue_final_message` auto). `0`/`false` off. CLI: `--prefill-assistant` / `--no-prefill-assistant`. Default: on, as llama.cpp |
| `FRINK_CB_PREFILL_CHUNK` | Continuous batching: prompt tokens per prefill chunk (default `128`). The scheduler runs one chunk plus one batched decode step per tick, so this is the granularity at which a long prompt yields to in-flight decodes |
| `FRINK_CB_MAX_QUEUE` | Continuous batching: requests allowed to wait for admission (default `512`). Past it, new requests get `503` + `Retry-After` instead of queueing without bound |
| `FRINK_CB_MAX_BATCH_TOKENS` | Continuous batching: tokens one tick may run, prompt and generated together (default `256`). The prefill chunk is sized against this minus the number of rows decoding in the same tick, so admitting a prompt costs the rows already answering less as the batch widens. It can only ever make a chunk smaller than `FRINK_CB_PREFILL_CHUNK`, and a floor of 16 prompt tokens per tick keeps a saturated batch from deadlocking |
| `FRINK_CB_KV_BLOCKS` | Continuous batching: total KV blocks the scheduler may hand out. Unset means it is *derived* at load alongside `FRINK_CB_MAX_CONTEXT`, or absent when the model cannot be priced. Admission is `blocks_needed <= blocks_free`, where a request needs `ceil((prompt + max_tokens) / block_size)` blocks reserved for its whole lifetime |
| `FRINK_CB_KV_BLOCK_SIZE` | Continuous batching: token positions per KV block (default `256`). The admission quantum. A request bigger than the whole budget gets `400` (`code: device_memory_budget_exceeded`), not `503` -- retrying cannot fix it |
| `FRINK_CB_MAX_CONTEXT` | Token positions (prompt + `max_tokens`) any one request may ask for, on **both** decode paths. Over it, `400` with `code: context_length_exceeded`, `estimated_bytes` and `limit_bytes`. Unset means the ceiling is *derived* at load from weights + per-token KV against the device budget, capped at the model's trained context; unset **and** unpriceable (no device-budget probe, or a header the planner cannot read) means no ceiling |
| `FRINK_SSE_ORPHAN_TIMEOUT_MS` | How long a streaming send waits for a client to make room before the stream is declared abandoned and the generation cancelled (default `30000`). `0` disables the deadline. Guards against a client that is neither reading nor disconnected pinning a blocking thread and the model handle it holds |
| `FRINK_CHUNKED_PREFILL` | Private (non-batched) generate path only: split a long prefill into N-token `forward_batch` chunks. Unrelated to `FRINK_CB_PREFILL_CHUNK`, which is a *scheduling* quantum, not a batch-shape one |
| `FRINK_CPU_KV_OFFLOAD` | `1`, sync Metal KV to host after each decode step |
| `FRINK_TOKIO_WORKERS` | `frink-server` async worker threads (default `2`); keeps the HTTP runtime from oversubscribing the decode pool |
| `FRINK_QOS_LOG` | `1`, log each rayon worker's macOS QoS class at pool start |
| `FRINK_EXIT_ON_STDIN_CLOSE` | `1`, exit on stdin EOF (same as `--exit-on-stdin-close`); off by default so a `/dev/null` stdin does not stop the server |
| `FRINK_KV_POOL_BLOCKS` | KV block-pool size (blocks). Bounds how much KV all requests may hold; each request still owns a private contiguous buffer |
| `FRINK_KV_POOL_BLOCK_SIZE` | Tokens per KV pool block |
| `FRINK_KV_POOL_QUEUE_TIMEOUT_MS` | How long a request waits for free KV before it is rejected. Applies to both the pool and paged KV |
| `FRINK_KV_BYTE_BUDGET` | Byte ceiling for the KV block pool, independent of block count |
| `FRINK_PAGED_KV_BLOCKS` | Blocks per layer of real paged KV: shared page storage many requests read through a block table, rather than a private buffer each. `frink-server` only. Mutually exclusive with `FRINK_KV_POOL_BLOCKS`/`FRINK_KV_BYTE_BUDGET` and with `FRINK_PREFIX_CACHE_ENTRIES`; setting an excluded pair stops the server with an error naming both. Read the paragraph under this table before using it |
| `FRINK_PAGED_KV_BLOCK_SIZE` | Positions per paged-KV block. Must be set together with `FRINK_PAGED_KV_BLOCKS`, or the server stops |
| `FRINK_PAGED_KV_SLIDE_INTERVAL` | Decode steps between window slides on a paged store (default 128). Only applies when *every* layer of the served model slides by the same window, because a page group holds one block in each layer, so a single full-attention layer disables sliding entirely. A smaller number returns pages sooner and costs a page operation more often; the admission bound pays for whatever accumulates in between |
| `FRINK_PREFIX_CACHE_ENTRIES` | Prefix-cache capacity for the private generate path: whole KV snapshots in an LRU list, reported under `GET /cache/stats`. Mutually exclusive with continuous batching and with paged KV. Paged KV carries no such exclusion: it composes with continuous batching and shares prefixes through the radix tree instead |
| `FRINK_EXPERT_CACHE_BYTES` | MoE expert-streaming cache budget |
| `FRINK_REPACK_CACHE_BYTES` | Bytes the interleaved-weight (`repack`) cache may retain. **`0` disables it**, which makes every CPU matvec rebuild its interleaved copy per call: the behaviour before that cache existed, correct and slower, and the way to run a memory-constrained host. Unset, the budget is derived from what the host says is available, less the standard fit headroom, less whatever `FRINK_EXPERT_CACHE_BYTES` has committed, divided by four — the expert budget is subtracted from the same pool rather than competing with it, because on unified memory a repacked byte and an expert byte are the same RAM. The cache evicts least-recently-used entries to stay under whatever it gets, and a matrix that does not fit is packed uncached. Retaining these copies is worth ~90% of a CPU decode token on Q8_0 (#128) and cost +527 MB of peak footprint at TinyLlama-1.1B Q8_0 unbounded |
| `FRINK_SSD_STREAMING` | `1`, stream MoE experts from disk |

Streaming is **off by default and turns itself on only when the weights
will not fit.** It is strictly slower than running resident, so it is a
way to run a model that otherwise could not run at all, not a default.
The automatic decision compares the checkpoint's size against the host's
available memory, reserves 4 GiB for the KV cache and activations, and
if the weights still do not fit it enables streaming and logs the two
figures and the cache size it chose.

`FRINK_SSD_STREAMING=0` refuses that: it forces resident loading even
when the weights do not fit, because an operator who says so may know
something the probe does not. `FRINK_EXPERT_CACHE_BYTES` sets the
budget explicitly and also wins over the automatic choice. A host whose
available memory cannot be determined resolves to resident rather than
to streaming: guessing a machine is short would silently put every user
on the slow path.

On Metal there is a further cost today. Several MoE fast paths still
accept only resident experts, so streaming currently gives up the fused
Metal MoE kernels as well. The warning says so when it fires.
| `FRINK_GPU_VRAM_BUDGET_BYTES` | Cap GPU-resident MoE experts (`0` = CPU experts on Metal) |
| `FRINK_DEVICE_BUDGET_BYTES` | Override the probed memory budget the pre-load KV check plans against (Metal `recommendedMaxWorkingSetSize` / free VRAM / host RAM minus a reserve). For container limits and shared GPUs |
| `FRINK_PIN_BUDGET_GB` | Override the page-locking budget expert-bank placement plans against. Unset means no cap on plain Linux; on WSL, where WDDM-backed CUDA caps pinning near half of RAM *shared across processes*, it defaults to 40% of physical RAM |
| `FRINK_BENCHBW_PATH` | Explicit path to this host's measured bandwidth profile. Otherwise one file per GPU uuid under `$XDG_CACHE_HOME/frink/benchbw/`, then the legacy `benchbw.json`. A file that exists but does not parse yields **no** profile rather than falling through, so a corrupt per-card profile never silently borrows another card's numbers |
| `FRINK_DECODE_LOG_INTERVAL` | Decode forwards between two batch status lines (default 40). `0` logs every forward. An unparseable value takes the default rather than failing a server to start over a log setting |
| `FRINK_ACCOUNTING_OUTBOX` | Directory accounting receipts are written to, atomically and idempotently by receipt id, before `POST /v1/admin/prepare-stop` answers. Unset means no outbox and no persistence step |
| `FRINK_INSTANCE_ID` | Names this engine generation for the receipt id. Unset falls back to the pid plus the process's wall-clock start, which is stable for the life of the process and different in the next one |

### Reading KV back after a Metal prefill

Paged KV used to be refused on any GPU backend, because it returned
fluent wrong tokens there. Measured on an M2 Pro with Llama-3.2-3B
Q4_K_M, same prompt and same seed: CPU paged and CPU contiguous both
answered "Blue and Red.", and Metal paged answered "Blue ( question
mark;>a> is a> is".

The cause was not the page indirection. A Metal prefill leaves K/V on
the device and fills the host cache with placeholder rows, which the
contiguous decode path knows and reads around; the paged prefill copied
those placeholders into the page store, and decode then attended over a
prompt the model never saw. The prefill now downloads the real rows for
the caller that reads them.

Held by `cargo test -p frink-models --features metal --test
paged_metal_parity -- --ignored`, which greedy-decodes the same prompt
twice in one process, once through each cache, on a dense model, an MoE
model and a sliding-window model. It runs one model per process on
purpose: two checkpoints in one process used not to answer the same as
either alone on Metal, because the resident-buffer caches keyed on a
host address a dropped model's allocator had already handed on. That
was GitHub issue #180 and is fixed; `model_swap_isolation` is the check
that holds it, and the per-process isolation here stays because this
suite should not be at the mercy of it either way.

`FRINK_PREFIX_CACHE_ENTRIES` had the same bug and no refusal in front
of it. A stored snapshot is the host rows, so on Metal it was all zeros,
and the next request restoring it answered nonsense at full speed --
"Blue and red." became " question mark of the day. The question of the
day is a question of the day." `sync_metal_attn_kv_to_host` could not
repair it: that function appends past `seq_len`, and the placeholder
fill has already advanced `seq_len` past the region needing filled. The
prefill now downloads the real rows when a prefix cache is configured,
and only then, since nothing else on that path reads them back.

### Sharing pages between prompts

Turning paged KV on also turns on a radix prefix cache over the same
pages. There is no separate switch, because sharing means two sequences
pointing at one page rather than one of them holding a copy, and only
the paged store can do that.

Each new prompt is matched against a tree of already-computed prefixes,
keyed by page. The matched prefix is locked for the request's life and
its page groups have their reference counts raised, so a thousand
conversations off one system prompt hold one copy of its KV rather than
a thousand. What the request reused shows up as `usage.cached_tokens`.
There is no aggregate hit rate on `/v1/stats` or `/metrics`, and no
eviction knob: back pressure comes from the page store running out and
`FRINK_KV_POOL_QUEUE_TIMEOUT_MS` turning waiting requests away.

The tree is keyed by token ids and shared across the deployment, so a
request that wants its prefixes to itself sends `cache_salt`: a
namespace is a whole tree there, the pages stay one pool, and eviction
takes from each namespace in turn rather than by a global LRU, so one
busy caller cannot evict every quiet one.

This is a different mechanism from `FRINK_PREFIX_CACHE_ENTRIES`, which
stores whole contiguous KV snapshots and copies them. The two cannot be
on at once.

## Security and transport

Everything except `/health` sits behind `FRINK_API_KEY` when it is
set, including `/admin/*`, `/metrics` and `/cache/stats`. A Prometheus
scraper pointed at a keyed server gets a `401`.

| Var | Effect |
|---|---|
| `FRINK_API_KEY` | Bearer token required by every route except `/health` |
| `FRINK_ALLOW_UNAUTHENTICATED_REMOTE` | `1`, permit binding a non-loopback address with no API key. Without it the server **refuses to start** in that configuration, which is deliberate: an unauthenticated model endpoint on a LAN address is an open proxy |
| `FRINK_TLS_CERT` / `FRINK_TLS_KEY` | PEM cert and key. **Both or neither**, setting one alone is a startup error. Unset means plain HTTP |
| `FRINK_CORS_ORIGINS` | Comma-separated **exact** origins. `*` is rejected on purpose, because a wildcard plus a bearer token is a credential-leak shape. **Required to serve Frink Studio (`ui/`) from another origin**, set it to that origin exactly. `ui/`'s dev server proxies the API instead, so `npm run dev` needs none of this |
| `FRINK_RATE_LIMIT_PER_MINUTE` | Global request cap. A non-integer value is a startup error, not a silent default |
| `FRINK_JOURNAL_PATH` | Where the process-lifecycle journal is written |
| `FRINK_CONVERSATIONS_DIR` | Where server-side conversations are stored, one JSON file each (default `./frink-conversations`). Created on first write. Nothing is evicted: the caps refuse with a reason rather than dropping an older conversation |

## Kernel-lookup registry

Every kernel lookup the model will make is resolved once at load and
recorded; the registry is then sealed, and a later lookup that misses and
takes a fallback warns once with its call site. See
`frink_core::kernel_registry`.

| Variable | Purpose |
|---|---|
| `FRINK_KERNEL_REGISTRY` | `1`, print the whole load-time kernel table (one line per backend × op × quant kind, with the tensor role and the fallback). `0`, record nothing. Default: record, print only the misses that will run off the selected accelerator |
| `FRINK_ALLOW_UNKNOWN_TENSORS` | `1`, load a checkpoint that carries tensors this build never reads, with a warning, instead of refusing. An unread tensor is a missing term of the graph (gpt-oss attention sinks, `exp_probs_b`), so the default is refusal: a wrong answer is worse than no answer |
| `FRINK_ALLOW_MULTIPLE_INSTANCES` | `1`, start even though another frink process is already holding a model. Default is refusal: two models on one box do not share it, they thrash it, and every timing either reports becomes noise. `--allow-multiple-instances` on `frink` / `frink-server` does the same for one run |
| `FRINK_INSTANCE_DIR` | Where the running-instance registry lives (default `$XDG_CACHE_HOME/frink/instances`, else `~/.cache/frink/instances`). One small file per live process, pruned when its pid is gone |
| `FRINK_STRICT_KERNELS` | `1`, refuse to load a model whose weights have no kernel on the selected accelerator, instead of running it on a slower path. Set this in CI and in benchmark harnesses so a number cannot be published for a backend it was not taken on |
| `FRINK_ALLOW_UNAUDITED_ARCH` | Run an architecture that has never been verified against llama.cpp. Off by default: such a model is refused rather than run on the shared generic-GQA path, which ASSUMES plain GQA and was already wrong for gpt2, mpt, refact, bloom and jais. Set it to compare the output against llama.cpp yourself |

## Metal instrumentation

These print numbers and change nothing else: no kernel is selected
differently, no output moves. They exist so a Metal change can be
attributed rather than guessed at, and they are the only in-tree way to
do that without a GPU capture.

| Variable | Purpose |
|---|---|
| `FRINK_METAL_MM_TIMING` | `1`, accumulate **wall-clock** setup / GPU-wait / readback microseconds across the prefill GEMM paths and print the totals. This is how long the host waited, which is what a `pp512` number is made of |
| `FRINK_METAL_GPU_TIMING` | `1`, accumulate **GPU-clock** milliseconds per tagged submission (`moe-decode/tok`, `dense-decode/tok`, `prefill-dense-stack`) from the command buffer's own timestamps, and print a running mean. Different question from the above: this one excludes host stalls |
| `FRINK_METAL_KERNEL_TIMING` | `1`, attribute the dense decode stack's GPU time to dispatch KINDS (matvec, attention, norm, RoPE, ...) and print a table. Apple GPUs sample the timestamp counter only at encoder boundaries, so every op group gets its own sampled encoder while this is on: the small-kernel rows carry an encoder boundary each and the between-encoder gaps are their own row. The ratio between two builds or two models measured the same way is the number it exists for |
| `FRINK_METAL_BARRIER_LOG` | `1`, log the running barriers-per-op ratio from `MemRanges`. `1.00` means the pass is fully serialised; lower means dispatches are overlapping. This is the direct measure of what a graph change bought |

## Test and development fixtures

Not configuration. `FRINK_TEST_*` exists so an `#[ignore]`d test can find
a checkpoint that is too large to commit, and points at a local file or
directory. `cargo test --workspace` passes with none of them set; the
tests that read them skip instead.

| Variable | Purpose |
|---|---|
| `FRINK_TEST_MODELS_DIR` | Root the real-GGUF sweeps scan (default `models`). Read by `bos_policy`, `chat_template_real_gguf`, `paged_metal_parity` and `model_swap_isolation` -- a git worktree has no `models/` of its own, which is what this is for. Unrelated to `FRINK_MODEL_DIR`, which is server config |
| `FRINK_TEST_GEMMA2_GGUF` | Gemma-2 GGUF for the Metal quality gate |
| `FRINK_TEST_QWEN2MOE_GGUF` | Qwen2-MoE GGUF for the "capital of France" check |
| `FRINK_TEST_SMOLLM2_GGUF` | SmolLM2 GGUF for the same check on Metal |
| `FRINK_TEST_PAGED_PARITY_GGUF` | GGUF for the paged-vs-contiguous KV parity test |
| `FRINK_TEST_RECEIPT_CHECKPOINT` | The pinned Llama-3.1 Q4_K_M GGUF the checkpoint-receipt test hashes |
| `FRINK_TEST_KIMI_SHARD_DIR` · `FRINK_TEST_KIMI_MOE_SHARD_DIR` · `FRINK_TEST_KIMI_TOKENIZER_PATH` | Kimi shards and tokenizer for the real-data tests |
| `FRINK_TEST_INSPECT_PATH` | A real `.gguf` for the tensor-table dump in `frink-gguf` (a print, not an assertion) |
| `FRINK_TEST_CHAT_TEMPLATE_NOW` | Pins `strftime_now`'s clock to Unix seconds so a template that stamps the date renders the same string every run. The only one of these read from library code rather than a test |

Two more that are development tools rather than deployment settings:

| Variable | Purpose |
|---|---|
| `FRINK_LLAMA_LOGITS` | Path to a locally built `llama_logits` reference dumper for `frink parity`. Otherwise `target/llama_logits`, then `.local-scripts/llama_logits`. `--dumper` is the flag form |
| `FRINK_PRESET` | Only consulted when `FRINK_MODEL_PATH` is unset, and then only to name which architecture sketch `frink-server` should build **random weights** for (default `glm-5.2`). It logs a warning saying so. A served model never reaches this path |

## Removed

These were switches, not configuration: each one only chose between a
default and a path that was slower, unproven, or produced deliberate
garbage for profiling. They are gone and the default is now the only
path. Setting them does nothing.

`FRINK_METAL_MATMUL` · `FRINK_METAL_MUL_MM` · `FRINK_METAL_LOGITS` ·
`FRINK_METAL_GREEDY_GPU` · `FRINK_METAL_WEIGHT_COPY` ·
`FRINK_METAL_PREFILL_FUSE_O` · `FRINK_METAL_FA_EXT` ·
`FRINK_METAL_FA_MMA` · `FRINK_METAL_FA_NQ` ·
`FRINK_METAL_MOE_RESIDENT` · `FRINK_METAL_MOE_STACK` ·
`FRINK_METAL_MOE_ABLATE` · `FRINK_METAL_MOE_FUSED_GATE_UP` ·
`FRINK_METAL_MOE_GATE_THEN_SILU` · `FRINK_METAL_MOE_BARRIER_LOG` ·
`FRINK_GEMV_DEDICATED` · `FRINK_GEMV_THREADS` ·
`FRINK_MIN_TASK_MACS`

## `FRINK_KV_WINDOW`, and why it is off

A sliding-window layer only ever attends over its last `window`
positions, so the rows before that are dead weight. Nothing in frink
evicted them until now, which is why a windowed model cost exactly what
a full-attention model of the same shape cost.

What it saves, on Gemma-3-4B at 32768 tokens of host f32 KV:

| | bytes |
|---|---|
| Off (every layer holds everything) | 9,126,805,504 |
| On, at rest | 1,692,590,080 |
| On, admission ceiling (prefill peak) | 1,962,934,272 |

The peak is not the full 9.13 GiB because prefill evicts per layer: one
layer holds the whole prompt at a time rather than all 34 at once. Five
of the 34 layers are full attention and still hold everything, which is
most of what remains.

The ceiling prices each windowed layer at `window + slack` rows -- the
top of the cycle a draining cache runs through, `KvWindow::max_rows` --
rather than at the exact instantaneous count, which oscillates. That
costs 0.7% against a number that already carries a whole layer's prompt,
and it buys a cost that never falls as the context grows, which is what
`--ctx-size auto`'s search for the largest fitting context needs.

**It is off by default because it is new, not because it is doubted.**
Output is token-identical with it on or off, asserted on identical logit
vectors as well as identical token ids, on gemma-2-2b-it-Q4_K_M with its
real SWA layout and real quantized tensors.

It disables itself in three cases rather than guessing:

- Under `FRINK_METAL_ATTN`, because five sites treat Metal's own
  sequence length and the host cache's resident rows as interchangeable.
  They are equal only while nothing evicts. Metal is step 3 of
  [#61](https://github.com/antonellof/frink/issues/61).
- On a draft model's decoder, which rolls its cache back arbitrarily far
  during speculative verification.
- Beside the prefix cache, which refuses to store a windowed cache
  rather than hand back a truncation it cannot represent.

**Admission and `--ctx-size auto` price what the switch really keeps.**
`KvBudget` carries the same per-layer residency the stores evict with,
so a run with the switch on is offered the context it can really carry
instead of one divided by every layer's full per-token cost. On
gemma-2-2b-it-Q4_K_M (window 4096 on 13 of 26 layers) against a
3.72 GiB budget, `ctx auto` reads 6912 tokens with the switch off and
7680 with it on; the same header at a 32768 context prices its KV line
at 6.50 GiB off and 4.06 GiB on. With the switch off the residency says
every layer keeps everything, which is the arithmetic this engine always
had, asserted rather than assumed.

What is still priced at the full number, because no store evicts there
yet: the paged store ([#61](https://github.com/antonellof/frink/issues/61)
step 4) and the prompt region while a prefill batch is being written
(step 5).

The byte figures inside a *context-length* refusal
(`ContextCeiling::bytes_for`) are still the unwindowed ones. They are a
message, not a decision -- the decision is the position ceiling above,
which does know.
