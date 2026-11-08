# North star

> **One goal.** Frink replaces llama.cpp on the hardware people own (a
> Mac, one consumer NVIDIA card, a CPU box): the same models, the same
> commands, the same tokens, at least the same speed, and a server that
> holds up under load.

This page is the goal and the only ranking. The work is GitHub issues,
the decisions are [ADRs](../adr/README.md), and a plan in this
directory is the design of one item, never a second ranking. What has
shipped is in [`CHANGELOG.md`](../../CHANGELOG.md), with its evidence.

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

Each row names the command or file that decides it, so this page holds
no figure of its own. Checked 2026-11-08.

| Point | Decided by | State |
|---|---|---|
| 1. Loads | `frink archs`, `tests/arch_coverage.rs`, the triage in [`MODELS.md`](../MODELS.md) | Every text-generation architecture llama.cpp names runs or refuses by name. A handful refuse as unaudited and need new code |
| 2. Same tokens | `frink parity` and the decode-step goldens in CI's oracle job; the render census (`crates/frink-server/tests/llama_renders/known_differences.txt`) for prompts | Every served architecture matches. Every chat-prompt difference left is a deviation the docs list |
| 3. Backend | [`KERNELS.md`](../KERNELS.md), generated and drift-tested | Every kind has a kernel on CPU, Metal and CUDA. Some CUDA kinds have never run on a GPU (#405). Vulkan runs decode matvecs for a few kinds, opt-in (#409) |
| 4. Not slower | `frink bench --render`, [`RESULTS.md`](../../benchmarks/RESULTS.md) | Metal is level or ahead. CPU is close on decode and behind on some prefill. CUDA is behind on both (#133, #259). Most published rows predate the current code and are marked stale |
| 5. Same command | `crates/frink-cli/src/llama_flags.rs`, which pins the gap | The common flags work. A long tail of `llama-cli` and `llama-server` flags and `LLAMA_ARG_*` variables is still unaccepted (#404) |

## Priorities

In order. Ranked by what each buys the people the goal names, then by
how cheaply it can be verified. Code that needs hardware is finished
with its ignored tests first, and the checks are batched into one
rental, asked for before it is booked. "Untracked" means the item still
needs an issue before it is started.

### The parity floor

1. **CUDA to parity.** Decode (#133: the resident dense stack and the
   coalesced matvec have landed and wait for a hardware run), prefill
   (#259), and the kinds that have never run on a GPU (#405). The
   largest gap, on the hardware most people own. Next step: the batched
   rental.
2. **Same commands, same API.** The flag and `LLAMA_ARG_*` tail (#404).
   OpenAI fields the server accepts and ignores
   (`max_completion_tokens`, `stream_options`, `user`,
   `parallel_tool_calls`, `stream` on `/v1/completions`), which breaks
   the rule that an unimplemented field is refused by name (untracked).
   The `llama-server` endpoints frink does not serve: `/infill`,
   `/props`, `GET /slots`, `/apply-template`, slot erase (untracked).
3. **Numerics.** The activation quantizers' last bit (#476), and the
   statistic `frink bench` prints (#481).

### A server that holds up under load

What a serving engine does that frink does not yet, ranked by what it
buys on one machine.

4. **Speculative decoding in the batched server.** The engine half is
   lossless and shipped (`frink run -md`). The server speculates only
   with prompt lookup or a draft model (`-md`, as `llama-server`) on its
   private loop, so on Metal, where continuous batching is the default,
   it does not speculate at all. Next: speculation inside the batch,
   then multi-token-prediction heads as a second drafter (untracked).
5. **Scheduling.** Preemption on by default after a measurement, and
   a request priority (#408). `/metrics` reports prefix reuse and the
   latency histograms a scheduler is tuned against.
6. **Memory.** Sliding-window eviction in the paged store and the pool
   (#61). KV quantization on the host and CUDA stores, not only Metal's
   device store (untracked). The disk KV tier wired into serving
   (untracked). MoE models larger than memory (#406).
7. **CUDA graph capture for decode.** `FRINK_CUDA_GRAPH` exists and
   changes nothing yet (untracked). Rides the same rental as item 1.

### Reach

8. **Multimodal** (#407): a vision projector and image input on
   `/v1/chat/completions`. The largest group of models llama.cpp serves
   and frink does not.
9. **One file per architecture** (#403,
   [ADR 0003](../adr/0003-one-file-per-architecture.md)), then the
   architectures that refuse as unaudited, from the 2026-10-28 pin move
   and the triage in [`MODELS.md`](../MODELS.md) (untracked).
10. **Other GPUs.** Vulkan beyond decode matvecs (#409), and AMD:
    ROCm on RDNA and CDNA, and the Ryzen AI NPU spike (#494,
    [`amd.md`](amd.md)).
11. **Quantized safetensors** (FP8 blockwise, GPTQ, AWQ) and
    quantizing a BF16 checkpoint while loading it (untracked).
12. **The CPU and Metal leftovers.** CPU decode's quiet-host receipt
    and the last fixed threshold (#27). Bonsai's PTQ1_0 prefill GEMM on
    Metal, decode being at parity (untracked).

**Not now:** tensor and pipeline parallel and multi-node serving,
prefill/decode disaggregation, MCP tool invocation, a desktop shell,
and the 2023 architecture tail beyond what the generic path covers.
Docker images are small and can land whenever someone wants them. Each
comes back when the items above are done, or when someone argues it
ahead of one.

## Where frink should win

Parity makes frink a substitute. These make it the better choice, and
each is a claim that can be measured:

- **Serving under load.** Continuous batching, paged KV, a prefix
  cache, preemption and speculation inside the batch give higher
  aggregate tok/s at 16 concurrent requests than `llama-server` on the
  same host (items 4 to 6).
- **Models larger than memory.** A MoE whose weights exceed RAM decodes
  at a stated tok/s on a 32 GB machine (#406).
- **Never silently wrong.** An unaudited model refuses where llama.cpp
  often runs it approximately, and a request field frink does not
  implement is refused by name.
- **A Rust library, not only a binary.** `frink-inference` embeds in a
  Rust program with no C toolchain.

## Picking the next task

Take the highest priority whose issue nobody is working on and whose
files no open branch is editing. If the change needs hardware, finish
the code and its ignored tests, and list the verification commands in
the PR.
