# Roadmap

> **One goal.** Frink replaces llama.cpp on the hardware people own (a
> Mac, one consumer NVIDIA card, a CPU box): the same models, the same
> commands, the same tokens, at least the same speed, and a server that
> holds up under load.

This page is the goal and the only ranking of the work. Each item is a
GitHub issue; the decisions are [ADRs](../adr/README.md); a plan file
in this directory is the design of one item in flight, never a second
ranking. What has shipped is in [`CHANGELOG.md`](../../CHANGELOG.md),
with its evidence.

## Done means

For a GGUF that llama.cpp runs at the pinned commit
([ADR 0001](../adr/0001-llama-cpp-is-the-reference.md)):

1. **It loads**, or refuses in one sentence naming what is missing
   ([ADR 0002](../adr/0002-refuse-rather-than-approximate.md)).
2. **Same tokens** at temperature 0, checked by `frink parity` against
   libllama, including a multi-step greedy decode, and the same prompt
   for the same chat request
   ([ADR 0011](../adr/0011-the-oracle-runs-in-ci.md)).
3. **It runs on the backend asked for**: every quant kind has a kernel
   there, or the receipt says it fell back to the host
   ([ADR 0010](../adr/0010-coverage-before-speed.md)).
4. **Not slower** on the same host, file and backend
   ([ADR 0006](../adr/0006-measurement.md)).
5. **Same command**: a llama.cpp command line works unchanged, or the
   difference is listed ([ADR 0005](../adr/0005-same-commands-as-llama-cpp.md)).

## Where it stands

Each row names what decides it, so this page holds no figure of its
own. Checked 2026-11-13, at 0.63.0.

| Point | Decided by | State |
|---|---|---|
| 1. Loads | `frink archs`, `tests/arch_coverage.rs`, the triage in [`MODELS.md`](../MODELS.md) | Every text-generation architecture llama.cpp names runs or refuses by name. A few refuse as unaudited and need new code (#538). |
| 2. Same tokens | `frink parity` and CI's oracle job (`scripts/oracle.sh`); the render census for prompts | Every served architecture matches on its fixtures on the CPU. A pure Q2_K checkpoint is WRONG on the CPU (#584). The Metal decode agrees with the CPU within rounding on every graph fixture; `hrm_text` amplifies that rounding past the line and is listed (`KNOWN_METAL_DIVERGENCES`). Some pre-tokenizer arms still fall back to GPT-2's (#629). The chat-prompt differences left are listed deviations |
| 3. Backend | [`KERNELS.md`](../KERNELS.md), generated and drift-tested | Every kind has a kernel on CPU, Metal and CUDA. Some CUDA kinds have never run on a GPU (#405). Vulkan serves a decode and prefill stack for its kinds, opt-in (#409) |
| 4. Not slower | `frink bench --render`, [`RESULTS.md`](../../benchmarks/RESULTS.md) | On an M5 Max, Metal is level with or ahead of llama.cpp on prefill and decode for every family measured (dense 135M to 12B, Gemma-2/3/4, gpt-oss, OLMoE, Qwen3-30B-A3B), in best-of interleaved runs on a shared host (#631); a quiet-host run by a person is still owed (ADR 0006). CPU is close on decode and behind on some prefill. CUDA is behind on both (#133, #259). Many published rows predate the current code (#542) |
| 5. Same command | `crates/frink-cli/src/llama_flags.rs`, which pins the gap | The common flags work. A tail of `llama-cli` / `llama-server` flags and `LLAMA_ARG_*` variables is still unaccepted (#404) |

## Priorities

One list, in order. The order is: wrong answers against llama.cpp
first, then the reference itself, then Metal (the backend at parity,
verifiable on our own hardware), then architectures, then the backends
that need a rental, then the command and serving surface. Code that
needs hardware is finished with its ignored tests first; the checks are
batched, and a rental is asked for before it is booked.

### A. Correctness against llama.cpp (urgent)

A wrong token is the one failure the goal cannot tolerate, and the
oracle makes each of these cheap to verify.

1. **Pure Q2_K is WRONG on the CPU** (#584). Since #625 `frink parity`
   compares the int8-activation path `frink run` takes, and pure Q3_K
   reads DRIFT like Q4_K_M. Pure Q2_K stays WRONG (decode KL 1.8e-2)
   from the listed attention deviation: llama.cpp's ARM CPU flash
   attention runs in f16 and frink's in f32, and this file amplifies
   it. Closing it means following ggml's per-ISA f16 attention (item 3).
2. **Pre-tokenizer arms still missing** (#629): `falcon`, `jais-2`,
   `chameleon`, `kimi-k2`, `spark2_5`, `youtu` fall back to GPT-2's
   split; the GPT4O lookahead and `ignore_merges` differ.
3. **The listed numeric deviations** ([ADR 0009](../adr/0009-numerics-follow-ggml.md)),
   largest first: CPU attention under 64 query rows and in decode.

### B. The reference

4. **Keep the oracle on llama.cpp's head.** Move `tools/llama_cpp_pin`
   when upstream adds architectures or changes served behaviour, run
   `scripts/oracle.sh`, and replace
   [`parity-audit-2026-10-28.md`](parity-audit-2026-10-28.md) with the
   new pin's audit (its title is held to the pin by
   `tests/oracle_manifest.rs`). The scheduled `--report-only` run
   against master says when.
5. **Every quant kind on every backend, and a test that says which do
   not** (#405). The matrix is generated; the open half is receipts and
   `frink inspect-plan` naming a host fallback. Draft PR #619 adds CUDA
   routed-expert kernels for every kind: decide first whether it should
   reuse the resident stack's kernels instead (ADR 0004).

### C. Metal

The backend at parity. Everything here can be checked on our own Mac,
without a rental.

6. **Verify what landed untested on Metal.** `frink-metal`'s ignored
   tests and `frink-models`' Metal graph suites pass on an M5 Max, and
   CI runs both. Then `frink-server` with a draft model and with a DFlash draft, once on
   the private loop and once with `--cont-batching`: several
   temperature-0 requests return the same text as without the draft,
   and `usage` reports accepted draft tokens (#600, #533).
7. **Re-measure the stale Metal rows** (#542), on a quiet host, by a
   person (ADR 0006).
8. **MoE larger than memory** (#406,
   [`out-of-core-moe.md`](out-of-core-moe.md)): the decode top-k path
   serves a stored expert; the packed `mul_mv_id` path and the Metal
   prefill MoE paths are still resident-only, then asynchronous reads
   and a run that really exceeds memory.
9. **Bonsai PTQ1_0 prefill** (#540): decode is at parity, the prefill
   GEMM is not.

### D. Architectures

10. **One file per architecture** (#403,
   [`arch-registry-migration.md`](arch-registry-migration.md)): step
   9c (`step35`, `mimo2`, `lfm3`), then the audited list, the catalog
   and the layer plan derived from the registry.
11. **The architectures that refuse as unaudited** (#538), each as one
    file with a libllama golden and a greedy decode, or a refusal with
    its reason in [`MODELS.md`](../MODELS.md).
12. **Multimodal** (#407): the Qwen-family projector first (`--mmproj`,
    image input on chat completions, Responses and Messages), every
    other projector type refused by name. The largest group of models
    llama.cpp serves and frink does not.

### E. CUDA and other GPUs (one batched rental)

13. **CUDA to parity**: decode (#133), prefill (#259), and the kinds
    that have never run on a GPU (#405). The largest speed gap, on the
    hardware most people own.
14. **Run the queued CUDA hardware tests**: the commands in the bodies
    of #606, #613, #615, #578 and #619. CUDA graph decode replay
    (`FRINK_CUDA_GRAPH=1`, #537) stays off until its test passes and
    the replay is measured.
15. **Vulkan beyond its current stack** (#409), and **AMD**: ROCm on
    RDNA and CDNA, the Ryzen AI NPU spike (#494, [`amd.md`](amd.md)).
    Vulkan has run only on MoltenVK and lavapipe. Queued for a real
    driver (RADV, AMDVLK, or an Intel iGPU if no AMD rental exists):
    `cargo test -p frink-vulkan --features vulkan` and
    `FRINK_VULKAN=1 cargo test -p frink-core --features vulkan vulkan -- --test-threads 1`,
    then any tok/s figure and the default flip, then zero-copy weights
    (`VK_EXT_external_memory_host`) and subgroup and cooperative-matrix
    tuning.

### F. Same commands, same API

16. **The flag and `LLAMA_ARG_*` tail** (#404), including automatic
    memory planning (`--fit` and family, pricing target, draft and every
    slot) and `--cache-reuse`: implement, or refuse by name.

### G. A server that holds up under load

Speculation inside the continuous batch has landed (#614, #615) and
DFlash drafts with it (#606, #608).

17. **Drafters still missing**: multi-token-prediction heads as a
    drafter (#533), recurrent targets drafting in the batch (needs a
    batched checkpoint), and per-token recurrent snapshots so a partial
    accept does not replay.
18. **Scheduling**: preemption on by default after a measurement
    (#408), and a request priority (#541).
19. **Memory**: sliding-window eviction in the paged store and the pool
    (#61); KV quantization on the host and CUDA stores (#535); the disk
    KV tier wired into serving, or deleted (#536); continuous batching
    on by default off Metal once a receipt exists.

### H. Reach

20. **Quantized safetensors** (FP8 blockwise, GPTQ, AWQ) and
    quantizing a BF16 checkpoint while loading it (#539).
21. **CPU decode scheduling** (#27): a persistent worker pool in place
    of a fork-join per operation, and the quiet-host receipt.
22. **Container images** for CPU and CUDA (#544). Small; lands whenever
    someone wants it.

**Not now:** tensor, pipeline and multi-node parallelism,
prefill/decode disaggregation, MCP tool invocation, an embedded chat
page or desktop shell, and the 2023 architecture tail beyond what the
generic path covers. Each comes back when the items above are done, or
when someone argues it ahead of one.

## Where frink should win

Parity makes frink a substitute. These make it the better choice, and
each is a claim that can be measured:

- **Serving under load.** Continuous batching, paged KV, a prefix
  cache, preemption and speculation inside the batch give higher
  aggregate tok/s at 16 concurrent requests than `llama-server` on the
  same host (items 17 to 19).
- **Models larger than memory.** A MoE whose weights exceed RAM decodes
  at a stated tok/s on a 32 GB machine (item 8).
- **Never silently wrong.** An unaudited model refuses where llama.cpp
  often runs it approximately, and a request field frink does not
  implement is refused by name.
- **A Rust library, not only a binary.** `frink-inference` embeds in a
  Rust program with no C toolchain.

## Picking the next task

Take the highest item whose issue nobody is working on and whose files
no open branch is editing. If the change needs hardware, finish the
code and its ignored tests, and list the verification commands in the
PR. When an item lands, delete it here in the same PR; when a plan file
finishes, its lasting decisions become ADRs and the file is deleted
(git keeps it).
