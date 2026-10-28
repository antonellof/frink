# CLAUDE.md

Guidance for agents working in this repo.

## What this is

Pure-Rust GGUF / MoE inference engine: mmap loaders, quantized CPU +
Metal + CUDA kernels, OpenAI-compatible `frink-server`.

**The goal is to be the Rust alternative to llama.cpp**: same models,
same command shapes, same or better performance, on the hardware people
actually own. `docs/plans/north-star.md` is the ranking every other plan
is read through, and `docs/plans/README.md` is the index.


Status: **100** architectures run with evidence
(`capability::AUDITED_GENERIC_GQA`), 4 more have dedicated engines,
and everything else refuses by name. The generic path is opt-in, so an
unaudited architecture stops instead of guessing. Do not restate the
counts elsewhere in this file; they go stale. `docs/MODELS.md` holds
the triage (pinned by `crates/frink-models/tests/documented_counts.rs`),
and `docs/plans/parity-audit-2026-10-28.md` holds the current
llama.cpp reference pin (`604cc430`, 169 graphs) with every census and
its command. How each row closed, and what each closure found, is in
`docs/plans/archive/closures-2026-09.md`.

## Adding an architecture: lessons that cost something

Each of these was learned on a real row; the closure log has the
details.

- Measure the reach before building a seam. Grep all of llama.cpp's
  `src/models/*.cpp` for the op or tensor and write down every graph
  that shares it, and why each does or does not. Several rows closed
  together because one cause sat behind them; others closed alone, and
  knowing that in advance was worth as much.
- Read the generic builder as well as the graph file. `maple`'s clamp
  form is decided in `llama-graph.cpp` by architecture, not in
  `maple.cpp`.
- A fixture must be able to tell the candidate forms apart. A clamp
  that never binds, a permutation that is the identity at one head, or
  a scale an RMSNorm cancels all pass for the wrong implementation.
- Read the converter's base class and the loader's key reads before
  classifying a row. `pangu-embedded` was deferred for weeks as an
  embedding model because of its name; several keys a file carries are
  dead metadata upstream and must stay dead here.
- A verdict that names its own closing evidence should go and get it.
  `minimax-m2` was refused for a week while its fixture sat in the
  tree.
- A tolerance widened until it passes everywhere measures the machine,
  not the model. Pick the fixture shape whose drift is explainable.
- `frink parity` reads the first token, which is prefill. A decode bug
  can produce fluent nonsense while parity stays MATCH; compare a
  greedy multi-step decode against a reference as well.

Do not read the architecture catalog as a support matrix. `frink
parity` is the oracle: its tokenizer half matches llama.cpp on every
local checkpoint libllama can load, and its logit half MATCHES on
Q8_0/IQ4_NL while DRIFTING on K-quants, for a known reason that is not
a frink bug (`docs/plans/llama-cpp-gap-inventory.md` §10).

Capabilities: `docs/FEATURES.md`. Models & speed ledger: `docs/MODELS.md`,
`benchmarks/RESULTS.md`. Planned: `docs/ROADMAP.md`.

| Doc | Role |
|---|---|
| `docs/FEATURES.md` | capabilities overview |
| `docs/CLI.md` | `frink` flags + `frink chat` |
| `docs/MODELS.md` | what runs / what doesn’t |
| `docs/API.md` | OpenAI compatibility matrix |
| `docs/AGENTS_COOKBOOK.md` | point IDEs at `frink-server` |
| `docs/CONFIG.md` | env vars |
| `benchmarks/RESULTS.md` | tok/s vs llama.cpp (Gap = llama/frink); `frink bench` ledger |
| `benchmarks/README.md` | how `frink bench` / `llama-bench` is measured |
| `docs/ROADMAP.md` | planned work |
| `docs/plans/README.md` | **the plan index and priority order** |
| `docs/plans/north-star.md` | the goal, and how plans are ranked against it |

## Commands

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

cargo build --workspace --features cuda
cargo test -p frink-cuda --features cuda -- --ignored   # needs GPU

cargo build -p frink-cli -p frink-server --features metal
cargo test -p frink-metal --features metal -- --ignored   # needs Metal

# Completion (also: frink run -m …)
./target/debug/frink -m model.gguf -p "Hi" -n 64 --temp 0 --no-cnv
./target/debug/frink -m model.gguf -p "Hi" -n 64 --ngl 99   # Metal

./target/debug/frink presets | archs | caps | inspect <gguf> | inspect-plan <gguf>
./target/debug/frink smoke <preset> | run-kimi <dir>
./target/debug/frink chat --url http://127.0.0.1:8383   # needs frink-server

FRINK_MODEL_PATH=model.gguf FRINK_ADDR=127.0.0.1:8383 ./target/debug/frink-server

# Bench vs llama-bench (no HTTP). Models: benchmarks/suite.json
./target/release/frink bench -m model.gguf -p 512 -n 128 --compare
./target/release/frink bench --suite --fit-host --skip-missing
./target/release/frink bench --suite --id llama32_3b_q4km --backend metal
./target/release/frink bench --render
```

Fixtures and golden values were generated and cross-validated with
independent NumPy references.

Tests are mostly `#[cfg(test)]` next to the code. Integration:
`crates/frink-models/tests/gguf_roundtrip.rs`. Never un-ignore CUDA /
Metal hardware tests without a real GPU.

## How to write code here

**Keep files small, modules narrow, and the binary light.** This is not
style preference, it is the repo's most expensive lesson.

Largest files today (2026-10-28), each a split waiting for its next
change: `frink-server/src/lib.rs` 10587 lines, `frink-metal/src/gpu.rs`
9360, `frink-quant/src/lib.rs` 8250, `frink-metal/src/attn.rs` 7056.
Nothing shrinks on its own.

What did work is splitting a file before adding to it:
`frink-quant/src/repack.rs` was 6446 lines and is now a directory of
ten; `frink-models/src/sampling.rs` went 1569 to 894 with four
submodules beside it; `frink-cuda/src/mul_mm.rs` is 865 with its kinds
in two directories. Each of those splits happened because somebody was
about to add to the file and split it first. That is the whole
mechanism, and it is the only one that has ever worked here.

On 2026-10-28 the two worst were split the same way, before the
next architecture landed: `frink-models/src/decoder.rs` went 8050 to
254 lines and `loader.rs` 6024 to 108, as module roots over one file
per concept in `decoder/` and `loader/` (the host forward bodies, the
Metal eligibility and launch glue, the FFN and MoE helpers, the
config resolution, the tensor readers, the tests by path). Pure
moves. Two functions are each a file of their own because splitting
them would be a rewrite: `Decoder::forward_token_on_worker`
(`decoder/forward_token.rs`, 1242 lines) and `ModelConfig::from_gguf`
(`loader/model_config.rs`, 1319). They are the next thing to shrink.

Big files are why adding a model used to mean editing a 7000-line
file, so nobody added one. The same decode layer used to be written out
about ELEVEN times across `decoder.rs` and `attn.rs`, which lost EIGHT
model features one at a time, each silently:
`attention_scale`, `post_attn_norm`, `post_ffn_norm`, gpt-oss `o_bias`,
`gpt_oss_ffn`, the four the Metal MoE decode stack ignored, and the
four-way drift of the GPU-router eligibility check, where the prefill
sites tested three conditions, the fused decode two and the whole-stack
decode NONE. A copy diverges from its original and nothing notices.

**THIS IS THE DOMINANT BUG SHAPE IN THIS REPO, and 2026-09-01 found a
dozen more instances of it in one day**, not all of them copied code.
The general form is TWO STRUCTURES THAT MUST AGREE ABOUT ONE THING,
WITH NOTHING ENFORCING IT: two spellings of a GGUF key (`unsupported_
feature_keys` gated on one no converter writes, so it never fired); two
copies of a default (the response cache restated every `unwrap_or`
independently of the sampler); four hand-written `SamplingParams`
literals in one file; three tables that had to agree about a Metal
kernel's threadgroup geometry (a correct kernel returned zeros for half
its rows); a wire struct and a sampler that silently disagreed about
which fields exist (SIX parameters accepted and ignored).

2026-09-04/05 added three more, all found by measuring rather than
reading: a benchmark receipt carrying `backend: "cpu"` beside
`backend_active: "Metal"`, in the same file, with nothing comparing the
two, for **13 of 13** published CPU rows; `KINDS` and the host-check
tool's fixed shape list, where adding the K-quants made the tool panic
on the first one and check NOTHING for a day while still exiting green;
and `FRINK_CPU_INT_DOT`, a default that was correct on the
architecture its kernels were written for and cost the other one 4x to
8.8x of decode.

The durable fixes are never the individual patches. They are the places
where disagreement now fails to COMPILE or turns a test red: an
exhaustive destructure with no `..`, one predicate the four call sites
share, a derived table instead of a restated one, and a test asserting
every refused key is one a converter actually writes.

Rules that follow from that:

- **A new file beats a new section.** If a change would push a file past
  roughly 1000 lines, split it first, then make the change.
- **One concept per module.** A module named after a noun that holds
  three unrelated things is three modules.
- **Never copy a code path to vary it.** Parameterise the original. The
  precedent that works: `forward_multi_seq` takes a `MultiSeqKv`
  parameter rather than having a paged twin. The precedent that failed:
  `forward_token_paged` was a copy, and lost five features -- it is now
  collapsed onto one `attn_block` taking a `KvStep`.
- **Dead code is a liability, not an asset.** Delete on sight unless it
  serves a named roadmap theme; if it does, wire it or say where it is
  going. `frink-edge` was 5,400 uncalled lines and is now dissolved.
- **A gate that cannot fire is worse than no gate**, because it reads as
  coverage. Check that a refusal's condition is reachable at all: this
  repo shipped one keyed on a GGUF spelling nothing writes.
- **No new crate for something one crate uses.** `frink-edge` became a
  crate instead of an integration and half of it was never called.

Rust specifics this repo holds to:

- `cargo clippy --workspace --all-targets -- -D warnings` is a gate, not
  advice. Also run it `--release`: `debug_assert!` type-checks its
  argument in release, and a `#[cfg(debug_assertions)]` method called
  from one broke every release build while all of CI stayed green.
- Prefer borrowing to cloning on any path that runs per token. Hoist
  feature probes out of loops: `is_aarch64_feature_detected!` ran 131k
  times in one Mistral-7B projection before it was hoisted.
- `unsafe` needs a `// SAFETY:` comment stating the invariant, and a
  scalar twin it is checked against. Every SIMD arm here has one.
- Return `Result` and name what is missing. A model this engine only
  partly implements must STOP, never compute something else. A refusal
  is coverage, not a defect.
- Tests live in `#[cfg(test)]` beside the code. A test that cannot fail
  is not a test: sabotage it once and confirm it goes red.
- **Confirm the sabotage LANDED.** A mutation that did not apply is
  indistinguishable from a test that holds. One sabotage here passed
  and proved nothing because `cargo fmt` had wrapped the target line
  across three lines, so the patch never matched. Grep the file for the
  mutated text before believing a green run, and if a test survives the
  first sabotage attempt, suspect the sabotage before believing the
  test.

## Measuring, and not fooling yourself

Every performance claim in this repo has to survive these. They are
here because each one was broken in a single week, and each break cost
either a wrong number in `benchmarks/RESULTS.md` or a merged-nothing
PR.

- **Rent a box; do not measure on this laptop.** `suggestd` holds ~97%
  of a core on it indefinitely and respawns hot when killed. CPU and
  CUDA rows come from vast.ai; the M2 Pro is for Metal, where it is the
  only hardware that can run the backend. Pick offers where
  `cpu_cores_effective == cpu_cores`, so no co-tenant shares the CPU.
  Four instances cost $0.49. Destroy them the moment the receipts are
  copied back, and confirm with `vastai show instances`.
- **A load average cannot see one busy core.** `frink bench`'s
  `--max-load 2.0` guard passed for an entire day while one of six
  cores was pegged, because one core does not move a six-core average
  enough to trip it. The guard is necessary and not sufficient: check
  `ps -eo pcpu,comm | sort -rn | head` too, and treat any process above
  ~90% as disqualifying.
- **One instantaneous sample is not a measurement.** "CUDA decode runs
  at 36% GPU utilization" came from a single `nvidia-smi` taken AFTER a
  bench had finished, so it caught an idle moment. Sampled five times
  DURING the run, the real figure was 86-93%. That one number sent a
  day of work at a host-side cost that did not exist and produced a PR
  measured 22% SLOWER. Sample repeatedly, and sample while the thing
  runs.
- **Verify the flag did what it says, from the artifact.** `frink
  bench --n-gpu-layers 0` is documented as forcing CPU and did not:
  the backend is decided once per process and cached, so the flag
  arrived too late. Read `backend_active` in the receipt, not the flag
  you passed. A receipt whose label disagrees with the backend that ran
  is now refused at write time and asserted over the committed set,
  because this was found only after publishing 13 wrong rows.
- **Check that the lever can reach the target before pulling it.** At
  the (wrong) 36% figure, removing EVERY host round-trip was worth at
  most 1/0.36 = 2.8x against a 9x-17x gap. That arithmetic was written
  down before the code was, and reading past it cost the PR. If the
  best case does not close the gap, the diagnosis is incomplete
  whatever else is true.
- **Interleave A/B when comparing two builds**, and report the raw
  sequence. `main, branch, main, branch` catches drift that two
  sequential runs hide.
- **A gap column cannot show a missing kernel.** A CUDA K-quant prefill
  read 4.88 tok/s against llama.cpp's 1586.80 and looked like a
  performance problem; there was no GEMM at all, and the fallback still
  answered correctly. Check coverage before profiling.
- **Check what the kernel is actually limited BY, before optimising
  anything in it.** Convert the measurement into the resource: for a
  decode matvec, `weights_bytes * tok/s` is achieved memory bandwidth,
  and that number is one line of arithmetic. frink reaches **5.3%** of
  an RTX 3060's 360 GB/s where llama.cpp reaches **60.4%**, so decode
  is limited by memory-request concurrency. A port of llama.cpp's
  `dp4a` inner loop was written, verified correct on hardware, and
  measured **under 1% faster**, because four MACs per instruction buys
  nothing in a kernel that is waiting on loads. The source diff between
  two kernels tells you what is different; it does not tell you which
  difference is the limit.

## Working with agents

- **One worktree per agent.** Two agents in the same checkout collided
  here: one detected another's half-finished refactor, backed its own
  work out, and redid it in isolation. Use `isolation: "worktree"`.
- **Do not redo an agent's task while it runs.** A narrower, better
  version of a deletion was in flight while the same deletion was
  attempted by hand; the hand version removed a live hardware test with
  it.
- **Agents do not tag, publish, force-push, rent hardware, or
  benchmark.** They implement and open a PR; verification on real
  hardware is the parent session's, and a kernel merges only after
  `cargo test -p frink-cuda --features cuda -- --ignored` has run on a
  GPU.

## Architecture

```
frink-gguf + frink-quant
        → frink-core (WeightMatrix, RoPE, GQA, KV; optional cuda/metal)
        → frink-moe
        → frink-models (loader, Decoder, Kimi/GLM/DS4 stacks)
        → frink-cli / frink-server

frink-api  (routes + wire DTOs, serde-only) → frink-server + clients
frink-cuda / frink-metal / frink-vulkan  (kernels, below frink-core)
frink-safetensors  (safetensors reader, beside frink-gguf)
frink-inference  (the published facade re-exporting the workspace)
```

**The FreeToken port** (Apache-2.0; see `docs/THIRD_PARTY_NOTICES.md`,
which is a licence obligation and must stay accurate) is a Rust port of
FreeToken's host-side decision logic. It used to be a crate of its own,
`frink-edge`; it now lives in the crates that use it, and the parts
nothing would ever use are deleted.

- **`frink-core`** holds the MoE expert-residency half, beside
  `expert_store`: `expert_cache`, `expert_slots` (the `SlotDevice`
  seam), `expert_pool` (its CUDA implementation), `expert_budget`,
  `qstar`, `bench_profile`, `residency`, `placement`. `expert_store` is
  the SINGLE holder of the expert byte budget -- on unified memory two
  budgets are the same RAM counted twice.
- **`frink-server::policy`** holds the serving half: the two parsers,
  the radix prefix cache, anchor/window slide, scheduler, serving stats,
  maintenance, pool, rebuild, outbox, footprint, effort probing.

Wired today: the parsers, the stop-string withhold rule, the radix
prefix cache over paged KV, effort probing, stats, maintenance, outbox,
footprint, and the scheduler's status reporting. STILL GROUNDWORK, and
this is the gap that matters most: the whole `frink-core` expert
residency stack holds the policy for running a model larger than memory
(`docs/plans/out-of-core-moe.md`) and nothing executes it except a
compile-only CUDA pool whose hardware test is `#[ignore]`d.

Anything in `policy` with an unwired half names the roadmap item that
would close it, at its declaration in `policy/mod.rs`. That
`allow(dead_code)` list is meant to be read as a to-do, not as cover.

Load path: GGUF mmap → keep quantized → fused dequant+dot →
RMSNorm → GQA(+RoPE) → MoE/dense FFN. Serving: `FRINK_MODEL_PATH`
GGUF or Kimi dir; generation on `spawn_blocking`.

Presets `glm_5_2` / `deepseek_v4_pro` / `kimi_k3` are sketches,
not proof of real-checkpoint support. `test_*_fixture` presets match
Python test GGUFs only.

Never add Co-Authored-By, Claude-Session, or Generated with Claude Code lines to commit messages or PR bodies.