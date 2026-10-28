# Plans

Two files hold the plan:

- **[`north-star.md`](north-star.md)** is the goal and the ranking rule.
  Be the Rust alternative to llama.cpp: same models, same command
  shapes, same or better performance, on the hardware people actually
  own.
- **[`roadmap.md`](roadmap.md)** is every open item, merged by theme.

Four items are large enough to carry their own design document:

- **[`speculative-decoding.md`](speculative-decoding.md)**, the one
  decode item that raises throughput without buying hardware. Decode
  reads every weight per token, so bandwidth divided by model bytes is a
  hard ceiling; a draft model changes what is read per token rather than
  how fast. The lossless half already ships and is tested at 200k
  samples. What is missing is a drafter worth having.
- **[`model-layer-reorg.md`](model-layer-reorg.md)**, splitting the
  decoder so architectures scale. `decoder.rs` was 6438 lines when that
  document was written and 8050 on 2026-10-28, when it and `loader.rs`
  (6024) were split into modules.
- **[`out-of-core-moe.md`](out-of-core-moe.md)**, running a 155 GB model
  on a 32 GB machine.
- **[`gdn-resident-state.md`](gdn-resident-state.md)**, a recurrent
  layer that does not come back to the host. It was the measured next
  step for Bonsai-2-27B decode, which has since gone from 7.2 to 11.17
  tok/s against the PrismML fork's 11.46, with the per-token ledger that says why, the delta-rule and
  gated-norm kernels that already exist and are pinned against the CPU
  definition, the reason wiring them TODAY is a loss (3.1 MB of state
  per layer, copied both ways), and four measured non-results so they
  are not tried again.

One audit sits beside them:

- **[`parity-audit-2026-10-28.md`](parity-audit-2026-10-28.md)**,
  the llama.cpp pin moved again (`5b59b83` to `604cc430`, 913 commits)
  and every census re-run against it: 13 new architecture names, the
  served-architecture behaviour that moved upstream, and what was fixed.
  Its predecessor, [`parity-audit-2026-09-19.md`](parity-audit-2026-09-19.md),
  still carries the serving-feature gaps a GGUF architecture count
  cannot see.

How work lands is written down too:

- **[`contribution-workflow.md`](contribution-workflow.md)**, the rule
  that a completed feature is a branch and a pull request, a defect is a
  GitHub issue, and the two are never the same artifact. It also carries
  the parallel-agent rules, whose first failure is two branches editing
  one file.

One item has a written **verdict** rather than a design:

- **[`vulkan-beachhead-verdict.md`](vulkan-beachhead-verdict.md)**, the
  `d-hardware-reach` GO/NO-GO. GO: a Q8_0 matvec ran as a hand-emitted
  SPIR-V shader on a real device and matched its scalar twin. It also
  carries the survey of the backend seam a third backend would need,
  which is `backend-seam-refactor`'s to-do list.

Parity inventory and deltas against llama.cpp:

- **[`parity-audit-2026-10-28.md`](parity-audit-2026-10-28.md)** — the current re-measurement of the llama.cpp surface, at the `604cc430` pin
- **[`parity-audit-2026-09-19.md`](parity-audit-2026-09-19.md)** — the previous one, at `5b59b83`, and the serving-feature gaps
- **[`serving-parity-audit.md`](serving-parity-audit.md)** — continuous batching, paged KV, prefix caching and cache-aware admission read against the code, every claim carrying the line that decides it (`documented_serving.rs` holds the citations)
- **[`server-speculative-decoding.md`](server-speculative-decoding.md)** — the engine has it, the server cannot reach it
- **[`done/several-completions-per-request.md`](done/several-completions-per-request.md)** — `n` > 1 as a KV fork, not as a loop that re-prefills
- **[`llama-cpp-gap-inventory.md`](llama-cpp-gap-inventory.md)** — evidence-backed differential (not a plan)
- **[`archive/llama-cpp-full-parity-audit-2026-09-02.md`](archive/llama-cpp-full-parity-audit-2026-09-02.md)** — file map + sweep + priority plan
- **[`archive/llama-cpp-parity-update-2026-09-03.md`](archive/llama-cpp-parity-update-2026-09-03.md)** — post-merge delta (Qwen MoE Metal, Phi-4 LongRoPE, sweep)
- **[`cpu-cuda-parity.md`](cpu-cuda-parity.md)** — the two backends that
  are not at parity, ordered by what was measured on rented hosts on
  2026-09-04 rather than by tok/s. Carries the kernel-coverage matrix,
  because a gap column cannot show a format the backend never runs

Everything else is history: [`archive/`](archive/) holds superseded
plans, whose open items were merged into the roadmap by theme;
[`on-hold/`](on-hold/) holds work ranked below the goal, each with the
condition that brings it back; and [`done/`](done/) holds plans that
finished. The counts in each of those READMEs are derived from the
files by `crates/frink-models/tests/documented_counts.rs`, because two
of them had already drifted.

## Where the project stands

Re-measured **2026-10-28**, against llama.cpp `604cc430`. Every number
here comes from a command or a receipt, named beside it; the closure
history behind them (how each architecture row closed, September 2026)
is [`archive/project-status-2026-09.md`](archive/project-status-2026-09.md),
and the per-release record is [`CHANGELOG.md`](../../CHANGELOG.md).

**Models.** llama.cpp names 166 architectures (`tests/llama_cpp_arch_names.txt`,
held 166 of 166 by `tests/arch_coverage.rs`).

| Outcome | Count | Source |
|---|---|---|
| Runs on the generic path **with a libllama golden** | **100** | `capability::AUDITED_GENERIC_GQA` |
| Dedicated engines | 4 | `Mla` and `Gemma4` have cross-engine evidence; `Glm52` and `Kimi` have none |
| Text generation, refused as unaudited and triaged | 11 | `tests/unaudited_triage.rs`: 0 fixture-away, 2 one match arm (`kolibri1`, `lfm3`), 8 new code, 1 unknown |
| Deferred scopes | 31 | `frink archs`: 13 encoder/embedding, 11 multimodal, 4 diffusion, 3 audio |
| **Loads and is WRONG** | **closed** | the generic path is opt-in |

The refusing count rose from 4 to 11 when the pin moved, because nine of
the thirteen new upstream names are text generation
([`parity-audit-2026-10-28.md`](parity-audit-2026-10-28.md)). The two
one-match-arm rows are the next cheapest work in the tree. The audit
also found three served architectures whose upstream behaviour had
changed (`mellum`, `nemotron`, `lfm2`), and all three are fixed or
refused by name.

**Speed** ([`benchmarks/RESULTS.md`](../../benchmarks/RESULTS.md)).
Every published row was measured on builds 0.17 to 0.21, and the table
marks them stale. Re-measuring 0.49 on a rented CPU+CUDA box is
scripted and waiting on account credit.

| Backend | Prefill gap | Decode gap |
|---|---|---|
| Metal (M2 Pro) | 0.99x to 1.09x | 0.60x to 0.96x (faster) |
| CPU (Ryzen 9 3900X) | 1.04x to 4.45x | 1.04x to 1.34x |
| CUDA (RTX 3090 / 3060) | 22x to 43x | 2x to 9x |

Ternary-Bonsai-2-27B on the M2 Pro: decode 11.17 against the reference
fork's 11.46 tok/s (97.5%), flat with context; prefill 43.1 against
66.8.

**What moved in the 2026-10-28 pass:**

- The host KV cache keeps **f16**, as llama.cpp's does: half the bytes
  per token on CPU, and `frink parity` now compares like with like.
  Parity audit §2.4 is closed.
- The **llama.cpp pin** moved 913 commits, and every census was re-run.
- **Expert streaming** runs the same kernels as resident experts. The
  OLMoE garbage from August had been fixed but was never pinned; a
  test with int-dot on pins it now.
- **Decode-step goldens.** Greedy decode is checked against llama.cpp
  at every step on four decode bodies, closing the hole that let Metal
  produce fluent nonsense while `frink parity` read MATCH.
- **`decoder.rs` and `loader.rs` split**, so an architecture is added
  in a small file ([`model-layer-reorg.md`](model-layer-reorg.md)).

**Still open, in north-star order:** the CUDA gap
([#259](https://github.com/antonellof/frink/issues/259),
[#133](https://github.com/antonellof/frink/issues/133)); out-of-core
MoE, where nothing executes the residency policy and the Metal
`SlotDevice` is not started ([`out-of-core-moe.md`](out-of-core-moe.md));
CPU decode scheduling ([#27](https://github.com/antonellof/frink/issues/27));
Vulkan beyond its beachhead; and the deferred scopes, multimodal first.

Do not read the architecture catalog as a support matrix: `frink
parity` is the oracle.

## The rules that keep being re-learned

**A plan's own status field is a claim, not evidence.** Verify against
the code. A merged PR once marked `paged-decode-path` complete while it
returned wrong tokens on Metal.

**One agent owns a file.** Two branches editing the same file produce a
merge nobody can review, and this project has already had one branch
silently revert three others.

**No agent runs benchmarks.** Measurement needs a quiet host, and a
loaded run reads 25-45% low.

**Refusing is not a defect.** llama.cpp will often run something
approximately; this project stops and names what is missing. A refusal
is a gap in coverage, not a bug.
