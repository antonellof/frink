---
name: "north star: the Rust alternative to llama.cpp"
overview: "THE GOAL, stated once so every other plan can be ranked against it: Frink should be what somebody reaches for INSTEAD OF llama.cpp. Same models, same command shapes, same or better performance, on the hardware people actually own. That is a bigger claim than 'a fast Rust inference engine' and it sets a bar that is checkable rather than aspirational: for any GGUF a user can run under llama.cpp, frink should run it, produce the same tokens, and not be slower. WHERE THE PROJECT ACTUALLY IS, re-counted 2026-10-28 against llama.cpp `604cc430`, which names 166 architectures: 100 run on ONE shared generic path, each with a libllama golden (`capability::AUDITED_GENERIC_GQA`); 4 more have dedicated engines, two of them with cross-engine evidence; 11 text-generation architectures refuse as unaudited, each triaged with the line that decides it; the rest are deferred scopes (encoder/embedding, multimodal, diffusion, audio). The loads-and-is-WRONG class is CLOSED because the generic path is opt-in. On speed (receipts from builds 0.17 to 0.21, re-measurement pending): Metal is at or past parity (pp512 0.99x-1.09x, tg128 0.60x-0.96x), CPU is close on decode and behind on prefill (1.04x-1.34x and 1.04x-4.45x on a Ryzen 9 3900X), and CUDA is far behind (22x-43x prefill, 2x-9x decode). THE SEQUENCING ARGUMENT: 'same models as llama.cpp' is 140 architectures and years of work. The wedge that gets there while being useful the whole way is RECENT BIG MODELS ON CONSUMER HARDWARE, including models too big for the machine's memory, because that is where llama.cpp is weakest and where a new engine can be chosen on merit rather than on being a rewrite. This plan ranks the other plans. It contains no engineering."
todos:
  - id: the-bar
    content: "NOT A TASK, the definition of done for the whole project, written so it can be argued with. Frink reaches parity when, for a GGUF a user can run under llama.cpp: (1) it LOADS, or refuses with a sentence naming exactly what is missing, and never loads-and-computes-something-else; (2) it produces the SAME TOKENS at temperature 0, checked against llama.cpp's own logits rather than against a golden file this project wrote; (3) it is NOT SLOWER on the same host, file and backend; (4) the COMMAND a user already knows works, or the difference is documented. Item (2) is the one that cannot be faked and the one this project already has the tool for: `frink parity` with `tools/llama_logits.c`. Item (1) is where frink currently differs most from llama.cpp, and deliberately so: llama.cpp will often run something approximately, this project's standing rule is to refuse. That rule stays. A refusal is a gap, not a defect."
    status: pending
  - id: t1-scale-the-model-layer
    content: "STATUS 2026-10-28: decoder.rs reached 8050 lines and loader.rs 6024, and both were split into modules; the registry-of-small-files end state below is not reached yet. TIER 1, and it gates every other model item, which is why it is first. `crates/frink-models/src/decoder.rs` is 6438 LINES and holds the generic path plus per-architecture branching. Adding a model means editing it, and reviewing a model change means reading it. llama.cpp adds a model as ONE SMALL FILE in `src/models/`, which is why it has 140 of them and frink has 47. This is not a style preference, it is the reason the counts differ. Evidence it already costs correctness: wave 0 found FIVE model features (`attention_scale`, `post_attn_norm`, `post_ffn_norm`, gpt-oss `o_bias`, `gpt_oss_ffn`) that the paged decode loop had LOST by being a copy of the contiguous one, three of them wrong on CPU too. A copy-per-path structure loses features one at a time and nothing notices. See `model-registry-reorg.md` for the design. DONE MEANS: adding an architecture is a new file, a trait impl, a registry line, a fixture and a parity test, with no edit to a shared multi-thousand-line file."
    status: pending
  - id: t1-close-the-68
    content: "STATUS 2026-10-28: 100 audited, 11 triaged refusals left after the pin moved to 604cc430 (0 fixture-away, 2 one match arm, 8 new code, 1 unknown). TIER 1. 57 architectures share one generic GQA path against llama.cpp's bespoke graphs, and five were found silently wrong. TWO STEPS ARE DONE. (1) THE CHEAP FIX: the generic path is now OPT-IN. An architecture not on `capability::AUDITED_GENERIC_GQA` returns `LoadError::UnauditedArchitecture` rather than falling onto generic GQA and running, which converted the unaudited set from 'silently wrong' into 'honestly refused'. (2) THE TRIAGE, landed 2026-09-01: all 46 refusals are classified fixture-away (9) / one-match-arm (7) / new-code (26) / unknown (4), each naming the `llama.cpp/src/models/*.cpp` line that decides it, pinned by `tests/unaudited_triage.rs`. That triage IS the work queue. WHAT REMAINS: execute it, auditing outward from what people run: `qwen3moe` (done, fixture-pinned), `glm4moe`, `deepseek2` (MLA), `minimax-m2` (fixture-away), `gemma4`, before the 2023 tail. The tail IS in scope under this goal, unlike under a purely edge-focused one: 'same models as llama.cpp' includes `bloom` and `gpt2`. It is last, not excluded."
    status: pending
  - id: t1-out-of-core-execution
    content: "STATUS 2026-10-28: streamed experts now run the same kernels as resident ones and the OLMoE garbage is pinned by a test; still nothing executes the residency policy, and no Metal or host-RAM SlotDevice exists. TIER 1, and the strongest single reason to choose frink over llama.cpp rather than merely tie with it. Make a model larger than memory run. A 200B MoE at Q4 is roughly 110 GiB; consumer machines have 16 to 128. An MoE touches only top-k of N experts per token, so the working set is a fraction of the weights, which is what makes this reachable. frink-core ALREADY HOLDS THE POLICY: `expert_cache` (LRU with copy plans), `expert_slots` (bounded pool behind a `SlotDevice` trait), `placement`, `residency`, `footprint`, `pool`. NOTHING EXECUTES IT except `expert_pool::CudaExpertPool`, which is compile-verified only with its hardware test `#[ignore]`d. So the capability exists on paper and not in the engine. NEEDS: a `SlotDevice` for Metal and one for host RAM. START WITH MoE, not dense weight streaming, which is a far worse trade and is not in scope."
    status: pending
  - id: t1-moe-execution-quality
    content: "TIER 1, because the recent models worth running are overwhelmingly MoE and MoE is where this engine is weakest. Three defects, all measured. (1) The Metal MoE path NEVER records `activation_counts`: `crates/frink-metal/src/attn.rs:5620` returns `vec![Vec::new(); layers.len()]`, commented as avoiding a sync tax, and `MoeDecodeScratch.ids` is a single top_k buffer reused per layer so it must be widened to `top_k * n_layers` or replaced by an atomic histogram. `placement_plan` has been reading an all-zero hotness signal on Metal, which matters DIRECTLY for out-of-core since eviction is only as good as its hotness input. (2) OLMoE CPU is 2.19x prefill / 2.46x decode while its Metal rows are at parity. (3) `concurrent-cpu-moe-executor` and `double-buffered-prefill` are unstarted. MoE bench coverage rested on ONE model until Qwen1.5-MoE (shared experts, which OLMoE lacks) was added."
    status: pending
  - id: t2-same-commands
    content: "TIER 2, cheap, and it is most of what 'alternative to llama.cpp' means in practice to a user. Somebody switching should not have to relearn flags. `frink bench` already mirrors `llama-bench`'s `-p`/`-n`/`-r` shape and `frink` mirrors `-m`/`-p`/`-n`/`-ngl`. AUDIT the rest against `llama-cli`, `llama-server` and `llama-bench` and close the gaps that are one-line aliases: `-c`/`--ctx-size`, `-t`/`--threads`, `-b`/`--batch-size`, `--temp`, `--top-k`, `--top-p`, `--repeat-penalty`, `-s`/`--seed`, `-ngl`, `--split-mode`, `--main-gpu`, `-fa`/`--flash-attn`. Where frink deliberately differs, DOCUMENT the difference rather than silently accepting a flag that means something else, which is worse than rejecting it. `frink download` landed with `hf download`'s exact syntax for the same reason: a command copied off a model card should just work."
    status: pending
  - id: t2-vulkan-is-most-consumer-gpus
    content: "TIER 2, and it is the largest single hardware gap. Frink has CPU, Metal and CUDA. That is Apple and NVIDIA and NOTHING ELSE: every AMD and Intel GPU has no path. llama.cpp ships Vulkan and reaches all of them with one backend, so 'same hardware as llama.cpp' is currently false by a wide margin. The four Vulkan items in `amd-strix-halo` are filed as blocked on owning a Strix Halo box, which is the wrong framing: Vulkan is testable on ANY machine with a Vulkan driver including an Intel iGPU, and Strix Halo is a tuning target rather than a prerequisite. Re-scope them off that hardware."
    status: pending
  - id: t2-cpu-performance
    content: "TIER 2, and the demotion needs saying plainly because the work is good and recent. Every red row in the ledger is CPU: all 16 comparable rows, prefill 1.41x to 5.06x, decode 1.68x to 3.55x. Under the bar in `the-bar` this is a genuine parity failure, so it cannot be dropped. It sits at TIER 2 only because it makes no new model RUNNABLE. Measured evidence that redirects the approach: the gap scales INVERSELY with model size, 5.06x at 135M down to 1.41x at 7B, the signature of fixed per-matmul overhead rather than kernel throughput, and every i8mm kernel already exists and is on the hot path. `cpu-decode-scaling` (fork-join, llama's persistent spin-barrier pool) is the remaining lever; the kernel items are largely spent. First evidence the recent work paid: TinyLlama CPU decode 48.11 to 54.41 tok/s, gap 2.15x to 2.00x."
    status: pending
  - id: t3-ranked-below
    content: "NOT ABANDONED, ranked below, recorded so nobody re-raises them without an argument. (1) `dflash-speculative-decoding`, all 9 items: it makes an already-running model faster, never makes a new one runnable, and sits behind `dflash-checkpoint-reality`, a GO/NO-GO needing a published drafter checkpoint that may not exist in loadable form. Frink does not train, and a drafter without weights proposes noise, which is slower than none. (2) `ui-tauri-shell`: the web UI works and is standalone. (3) The 2023 architecture tail: in scope under this goal, but after the recent families, and handled safely in the meantime by the opt-in inversion in `t1-close-the-68`."
    status: pending
  - id: verification-without-downloading-everything
    content: "CROSS-CUTTING, the answer to 'do we have to test every model'. No. 140 real checkpoints is tens of terabytes and would still only prove the ones downloaded. THREE MECHANISMS ALREADY EXIST HERE, each applied to far less than it could be. (1) DIFFERENTIAL, the decisive one: `frink parity` plus `tools/llama_logits.c` compares first-token logit distributions against llama.cpp on the same file, needs no knowledge of the architecture, and is not run across the suite. (2) SYNTHETIC FIXTURES: `scripts/make_dots1_fixture.py` and `make_gptoss_fixture.py`, two architectures out of 139. A fixture is kilobytes, needs no download, and catches wrong tensor wiring and wrong refusals. One per supported architecture is the best coverage-per-byte available and should be a requirement of the registry in `t1-scale-the-model-layer`. (3) PINNED PROPERTY TABLES transcribed from llama.cpp's loader: `LLAMA_NO_ROPE` does this for one property and is exactly what caught the gpt2/bloom class. Extend to ALiBi slopes, norm placement, QKV bias, MoE routing bias."
    status: pending
  - id: measurement-needs-more-than-one-laptop
    content: "CROSS-CUTTING, and it now blocks the claim rather than merely annoying us. Every number comes from ONE 32 GiB M2 Pro that is also the development machine. Already hit: two Metal rows skipped mid-suite when macOS CoreSuggestions spiked the load, a whole suite lost to an orphaned process holding the instance lock, Mixtral permanently unmeasurable, CUDA never measured, x86 never executed. 'Same or better performance than llama.cpp' cannot be claimed on hardware nobody has run. LANDED 2026-08-27: a receipt `host_spec` field recording CPU model, core split, RAM and OS but NOT hostname or user, and a render that REFUSES to merge receipts from different hosts rather than silently mixing them. STILL NEEDED: a bare-metal rental for CPU and x86, NOT a shared-vCPU instance, because a noisy neighbour's steal time does not appear in the guest's load average and would defeat the quiet-host guard invisibly; and a decision on whether CUDA earns a dedicated box or whether 'must compile' stays the honest claim."
    status: pending
isProject: false
---

# North star: the Rust alternative to llama.cpp

> Same models. Same command shapes. Same or better performance. On the
> hardware people actually own.
>
> This plan contains no engineering. It is the ranking the other plans
> are read through.

## The bar, so it can be argued with

For any GGUF a user can run under llama.cpp:

1. It **loads**, or refuses naming exactly what is missing. Never
   loads-and-computes-something-else.
2. It produces the **same tokens** at temperature 0, checked against
   llama.cpp's own logits.
3. It is **not slower** on the same host, file and backend.
4. The **command** a user already knows works, or the difference is
   documented.

Point 1 is where frink deliberately differs: llama.cpp will often run
something approximately, and this project refuses instead. That rule
stays. A refusal is a gap, not a defect.

## Counted, not estimated

Re-counted 2026-10-28 against `capability::architecture_catalog`,
`tests/llama_cpp_arch_names.txt` (llama.cpp `604cc430`, 166 names) and
[`../manifests/architecture_manifest.md`](../manifests/architecture_manifest.md),
by what happens when a real checkpoint loads rather than by whether the
name is known:

| Outcome | Count |
|---|---|
| **A. Runs on the generic path, with a libllama golden** | **100** |
| **B. Dedicated engines** | 4 (`Mla` and `Gemma4` with cross-engine evidence, `Glm52` and `Kimi` without) |
| **C. Refuses as unaudited**, triaged and named | 11 (0 fixture-away, 2 one match arm, 8 new code, 1 unknown) |
| **D. Deferred scope** | 31 (13 encoder/embedding, 11 multimodal, 4 diffusion, 3 audio) |
| **E. Loads and is WRONG** | **closed** |

Category A was 11 on 2026-09-01. Category C fell from 46 to 2 over
September and rose to 11 when the pin moved six weeks, which is what
parity with a moving target looks like: nine of the thirteen new
upstream names are text generation
([`parity-audit-2026-10-28.md`](parity-audit-2026-10-28.md)).

| | llama.cpp | frink |
|---|---|---|
| Architectures | 166 named, 169 graph files | **100 generic with goldens**, 4 engines, 11 triaged refusals |
| Metal `pp512` | baseline | 0.99x-1.09x, at parity |
| Metal `tg128` | baseline | **0.60x-0.96x, faster on every row** |
| CPU `pp512` / `tg128` | baseline | 1.04x-4.45x / 1.04x-1.34x slower (Ryzen 9 3900X) |
| CUDA `pp512` / `tg128` | baseline | 22x-43x / 2x-9x slower |
| GPU backends | CUDA, Metal, Vulkan, SYCL, HIP | CUDA, Metal; Vulkan at a Q8_0-matvec beachhead |

The speed rows are receipts from builds 0.17 to 0.21 and are marked
stale in [`../../benchmarks/RESULTS.md`](../../benchmarks/RESULTS.md).
A gap measured against a moving binary means nothing until it is
measured again.

## Why the model layer was first

llama.cpp has its architecture count because adding one is a small new
file. Frink got from 11 proven to 100 by building SEAMS (a table row
per architecture on a shared body, one fixture and one golden each)
rather than by editing a 6800-line file per model. The file grew anyway,
to 8050 lines, and was split into modules on 2026-10-28
([`model-layer-reorg.md`](model-layer-reorg.md)). The lesson that put
this first still holds: a copy-per-path structure loses features one at
a time and nothing notices. The paged decode loop once lost five, three
of them wrong on CPU too.

## The cheap fix that beat the expensive audit: DONE

Auditing every unproven architecture is expensive. Inverting the default
was not, and it **landed**. An unrecognised architecture no longer falls
onto the generic path and runs; it has to be on
`capability::AUDITED_GENERIC_GQA` or it returns
`LoadError::UnauditedArchitecture` (`FRINK_ALLOW_UNAUDITED_ARCH=1`
overrides). That turned the silently-wrong risks into honest refusals.

The follow-on also landed: all 46 refusals are **triaged** into
fixture-away (9), one-match-arm (7), new-code (26) and unknown (4), each
naming the `llama.cpp/src/models/*.cpp` line that decides it, so the
refusal message distinguishes a week of work from a quarter. That triage
is the work queue the audit-outward item needs. What remains is
executing it.

## Ranking

**Shipped and working beats big and undone.** That rule outranks the
tiers below. An item that cannot be cut into steps which each land is
filed wrong, and the fix is to re-cut it, not to start it.

The executable order lives in [`roadmap.md`](roadmap.md). It differs
from a pure tier sort in one deliberate way: correctness and the
verification oracle come FIRST, ahead of the model layer refactor,
because some checkpoints are wrong right now and their fixes are small,
and because refactoring 15464 lines across two crates without a working
cross-engine oracle means refactoring known-wrong code with no way to
tell if you broke it.

```
TIER 1  scale the model layer     seams DONE (100 proven); file split 2026-10-28
        close the refusals        11 left, 2 of them one match arm away
        out-of-core execution     the reason to choose frink, not tie
        MoE execution quality     hotness signal, CPU executor, residency

TIER 2  same commands             cheap, and most of what "alternative" means
        vulkan                    every AMD and Intel GPU, currently none
        cpu performance           every red row, but unlocks no new model

TIER 3  dflash speculative        faster, not newly-possible, and blocked
        tauri shell               the web UI already works
        the 2023 tail             in scope, but last
```
