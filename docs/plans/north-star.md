# North star

> Frink is what you reach for instead of llama.cpp: the same models, the
> same commands, the same tokens, and at least the same speed, on the
> hardware people own. Then it wins where llama.cpp is weakest.

This page is the goal and the ranking. Work is tracked as GitHub
issues; decisions are in [`../adr/`](../adr/README.md).

## Done means

For a GGUF that llama.cpp runs at the pinned commit
([ADR 0001](../adr/0001-llama-cpp-is-the-reference.md)):

1. **It loads**, or refuses in one sentence naming what is missing
   ([ADR 0002](../adr/0002-refuse-rather-than-approximate.md)).
2. **Same tokens** at temperature 0, checked by `frink parity` against
   libllama, including a multi-step greedy decode
   ([ADR 0011](../adr/0011-the-oracle-runs-in-ci.md)).
3. **It runs on the backend asked for**: every quant kind has a kernel
   there, or the receipt says it fell back to the host
   ([ADR 0010](../adr/0010-coverage-before-speed.md)).
4. **Not slower** on the same host, file and backend
   ([ADR 0006](../adr/0006-measurement.md)).
5. **Same command**: a llama.cpp command line works unchanged, or the
   difference is listed ([ADR 0005](../adr/0005-same-commands-as-llama-cpp.md)).

## Scoreboard

One row per point above. Each number comes from a command, so this page
never holds a figure of its own.

| Point | Measured by | Target |
|---|---|---|
| 1. Loads | `frink archs` | every text-generation architecture llama.cpp names runs or refuses by name |
| 2. Same tokens | `frink parity` over the fixtures, in CI (#402) | every served architecture matches |
| 3. Runs on the backend | the kernel coverage matrix (#405) | no host fallback for any kind on CPU, Metal or CUDA |
| 4. Not slower | `frink bench --render` ([`RESULTS.md`](../../benchmarks/RESULTS.md)) | gap at or below 1.00 on every suite row, per backend |
| 5. Same command | the llama.cpp flag table test (#404) | zero unaccepted flags for `llama-cli`, `llama-server`, `llama-bench` |

As of 2026-10-28: points 1 and 2 are close for text generation; Metal
meets point 4; CPU is close on decode and behind on prefill; CUDA is far
behind on both and has host-only kinds; Vulkan has one kernel.

## Where frink should win

Parity makes frink a substitute. These make it the better choice. Each
is a claim that can be measured:

- **Models larger than memory.** A MoE whose weights exceed RAM decodes
  at a stated tok/s on a 32 GB machine (#406).
- **Serving under load.** Continuous batching, paged KV, a prefix
  cache and preemption (#408) give higher aggregate tok/s at 16
  concurrent requests than `llama-server` on the same host.
- **Never silently wrong.** An unaudited model refuses; llama.cpp often
  runs it approximately.
- **A Rust library, not only a binary.** `frink-inference` embeds in a
  Rust program with no C toolchain.

## Priorities

In order. Each is an issue with a "done when". An item that cannot be
cut into steps that each land is re-cut before it is started. Hardware
verification is batched into one rental once the code is ready.

1. **The oracle in CI** (#402). Everything else moves faster when
   "same tokens" is checked on every PR and a pin bump is a list.
2. **CUDA to parity**: decode (#133), prefill (#259), host-only kinds
   (#405). The hardware most people own, and the largest gap.
3. **One file per architecture** (#403,
   [ADR 0003](../adr/0003-one-file-per-architecture.md)). Makes the
   remaining models cheap and removes the table-agreement bugs.
4. **Same commands** (#404). Cheap, and most of what "alternative"
   means to a user.
5. **Models larger than memory** (#406).
6. **Serving**: preemption (#408), sliding-window KV eviction (#61).
7. **CPU decode scheduling** (#27) and the remaining CPU prefill gaps.
8. **Multimodal** (#407). The largest group of models llama.cpp serves
   and frink does not.
9. **Vulkan** (#409), for AMD and Intel GPUs.

Not now: speculative decoding with trained drafters, a desktop shell,
the 2023 architecture tail beyond what the generic path covers. Each
comes back when the items above are done, or when someone argues it
ahead of one.

## Picking the next task

Take the highest priority whose issue nobody is working on and whose
files no open branch is editing. If the change needs hardware, finish
the code and its ignored tests, and list the verification commands in
the PR.
