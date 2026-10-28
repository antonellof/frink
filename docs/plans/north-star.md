# North star

> Frink is what you reach for instead of llama.cpp: the same models, the
> same commands, the same tokens, and at least the same speed, on the
> hardware people own.

This page is the goal and the ranking. Work is tracked as GitHub
issues; decisions are in [`../adr/`](../adr/README.md).

## Done means

For a GGUF that llama.cpp runs at the pinned commit
([ADR 0001](../adr/0001-llama-cpp-is-the-reference.md)):

1. **It loads**, or refuses in one sentence naming what is missing
   ([ADR 0002](../adr/0002-refuse-rather-than-approximate.md)).
2. **Same tokens** at temperature 0, checked by `frink parity` against
   libllama, including a multi-step greedy decode.
3. **Not slower** on the same host, file and backend
   ([ADR 0006](../adr/0006-measurement.md)).
4. **Same command**: a llama.cpp command line works unchanged, or the
   difference is listed ([ADR 0005](../adr/0005-same-commands-as-llama-cpp.md)).

## Where we are

Counts come from the code, not from this page: `frink archs` for
architectures, [`../MODELS.md`](../MODELS.md) for the triage,
[`../../benchmarks/RESULTS.md`](../../benchmarks/RESULTS.md) for speed.
In short, as of 2026-10-28:

| | State |
|---|---|
| Models | Most text-generation architectures llama.cpp names run with a libllama golden; about ten refuse by name; multimodal, embedding, diffusion and audio are deferred |
| Metal | At or past llama.cpp |
| CPU | Close on decode, behind on prefill |
| CUDA | Far behind (#133, #259) |
| Vulkan | One Q8_0 matvec |

## Priorities

In order. An item that cannot be cut into steps that each land is
re-cut before it is started.

1. **CUDA to parity.** It is the hardware most people own and the
   largest gap. Decode is memory-concurrency bound (#133); prefill is
   host-bound (#259).
2. **One file per architecture**
   ([ADR 0003](../adr/0003-one-file-per-architecture.md)). Gates the
   long tail of models and removes the table-agreement bugs.
3. **Same commands** ([ADR 0005](../adr/0005-same-commands-as-llama-cpp.md)).
   Cheap, and most of what "alternative" means to a user.
4. **Models larger than memory.** MoE expert streaming with a residency
   policy that actually executes
   ([`out-of-core-moe.md`](out-of-core-moe.md)). The one place frink
   can be chosen on merit, not just matched.
5. **CPU decode scheduling** (#27) and the remaining CPU prefill gaps.
6. **Serving throughput**: preemption, sliding-window KV eviction
   (#61), batched decode on every backend.
7. **Vulkan**, for AMD and Intel GPUs.

Not now: speculative decoding with trained drafters, a desktop shell,
the 2023 architecture tail beyond what the generic path already covers.
Each comes back when the items above are done or when someone argues
it ahead of one.
