# Handover: where development stands and how to pick it up

Dated 2026-11-07, at `main` = `c09c21dd`, release **v0.55.0**. Read
this after [`CLAUDE.md`](../../CLAUDE.md) and
[`north-star.md`](north-star.md); it says what changed recently, what
is half-done, what is blocked on hardware, and the working recipes
that are not written down anywhere else. Update the date line and the
"Recent work" table when you hand over again; delete the file when it
stops being true.

## State in one paragraph

`main` is clean and green. Releases v0.51.0 to v0.55.0 shipped in the
last stretch (GitHub tarballs for darwin-arm64 and linux-x86_64,
every crate on crates.io). Four PRs are open, green and waiting for a
merge: see "Open PRs" below. Work since v0.50 went into four areas:
- CPU integer dots for every weight kind (#405)
- llama-bench flag coverage (#404)
- three correctness fixes (#470, #421, #408-adjacent)
- AMD (#494): a ROCm beachhead that compiles every CUDA kernel for
  RDNA, NPU detection, and a Vulkan backend that now decodes K-quant
  models correctly but is still slower than the CPU

## Open PRs

Each is cut from `main` and none depends on another, so any order
works. `frink-core --features metal` stays red on #510 and #511 until
#506 lands, because #506 holds the fix.

| PR | What | Issue |
|---|---|---|
| #506 | `frink-core`'s unit tests pass on a device build, and CI now runs them | #501 |
| #510 | Chat templates: 47 of llama.cpp's 75 bundled templates failed, most silently losing their tool format. Census in the oracle job, 36 jinja2 goldens | #507 |
| #511 | `--chat-template`, `--chat-template-file`, `--chat-template-kwargs`, `-rea`, `--reasoning-effort` | #404 |
| #505 | This file's corrections | none |

## Recent work (newest first)

| PR | What | Issue |
|---|---|---|
| #503 | journal panic-hook test counted other threads' panics (flake) | none |
| #502 | release 0.55.0 | none |
| #500 | Vulkan: Q/K/V and gate/up as one submission each (`Runtime::matvec_many`); seam tests compared Vulkan with itself, now `apply_cpu` | #409 |
| #499 | Vulkan: `frink_vulkan::runtime::Runtime` (resident weights, cached pipelines), a workgroup per row, Q8_0 in GLSL, hand-emitted SPIR-V emitter deleted; `frink_quant::fingerprint` shared with `frink-cuda` | #409 |
| #498 | release 0.54.0 | none |
| #497 | Vulkan Q4_0/Q4_K/Q5_K/Q6_K matvecs (GLSL → committed SPIR-V); `WeightMatrix::apply` had no Vulkan gate at all | #409 |
| #496 | `--list-devices` names a Ryzen AI NPU (sysfs, Linux) | #494 |
| #495 | `crates/frink-rocm`: hiprtc via dlopen, every `frink-cuda` kernel compiles for 8 RDNA targets in CI; CDNA refused as wave64 | #494 |
| #480–#492 | llama-bench flags: `--no-warmup --delay --prio --progress -o -oe -hf -hff -hft -ctk -ctv -d -pg -b -ub --offline` | #404 |
| #485–#487 | preempted-row requeue; `llama-embed` refused by name; SWA window keys | #408, #421, #470 |
| #473–#479 | CPU integer dots for every remaining weight kind | #405 |

The CHANGELOG has the full text of each.

## Vulkan (#409): exactly where it is

Layout of `crates/frink-vulkan`:
- `shaders/*.comp` + `matvec_common.glsl`: GLSL source.
- `shaders/spv/`: committed SPIR-V, rebuilt with `shaders/build.sh`.
  `spv/GENERATOR` pins glslang 16.5.0; a test recompiles the shaders and
  fails on drift when that version is installed (`brew install glslang
  spirv-tools`).
- `src/shaders.rs`: the one table (`MATVEC`). Every `.spv` must have a
  row, and a test enforces it.
- `src/runtime.rs`: `Runtime` (device, pipelines, resident weights,
  `matvec`, `matvec_many`, `force_staging`).
- `src/dispatch.rs`: one typed entry point per kind, plus the
  device tests against `frink_quant::dot_*_f32_scalar`.

Wiring in `frink-core` (`weight_matrix/gpu_backend.rs`):
- `Vulkan` impls `BackendCaps` and `BackendDispatch`.
- `vulkan_runtime()` holds the process-wide `Mutex<Runtime>`.
- `vulkan_matvec_many` is the fused arm of `apply_gpu_multi`.
- `WeightMatrix::apply` gates on `Vulkan::dense_enabled()`.

`FRINK_VULKAN` unset means **off** (opt-in) on purpose.

Measured once on the M2 Pro (MoltenVK), Llama-3.2-1B Q4_K_M, with the
greedy tokens identical to the CPU's at every stage: 0.9 → 14.3 → 25.5
→ ~30 tok/s against the CPU's ~113. These are laptop numbers to steer
the work, not benchmarks (ADR 0006).

Why it is still slow: the host runs norms, RoPE and attention between
matvecs, so each layer is about four queue submissions plus waits, at
roughly 0.2–0.5 ms each on MoltenVK.

Next steps, in order:
1. **A layer on the device**: RMSNorm, RoPE, attention over a
   device-resident KV, SwiGLU, residual. That is one submission per
   layer, or per token, which is the shape `frink-metal` already has.
   Start from the Metal resident decode path as the reference for what
   the stack must do, and from llama.cpp's `ggml-vulkan` shaders for
   kernel shape.
2. **A fused dense FFN** (gate/up → SwiGLU → down in one submission,
   like `apply_gpu_dense_ffn_swiglu`). This is a smaller win than step
   1 and a step toward it.
3. **More kinds**: Q5_0, Q4_1, Q5_1, Q2_K, Q3_K, IQ4_NL/XS, MXFP4, from
   the `docs/KERNELS.md` Vulkan column.
4. **GEMM for prefill**: today `gemm_supported` is false and prefill
   runs on the host.
5. **Then flip the default**, but only with a measured win on rented
   hardware.

Run on a Radeon when one is available (RADV or AMDVLK, no flags
needed): `cargo test -p frink-vulkan --features vulkan` and
`FRINK_VULKAN=1 cargo test -p frink-core --features vulkan vulkan --
--test-threads 1`. Note that `FRINK_VULKAN=1` over the whole
`frink-core` suite is very slow; filter it.

## AMD (#494): the rest

Plan: [`amd.md`](amd.md).

- **ROCm (`frink-rocm`, `publish = false`)**:
  - Phase R1 is done. The CI job `rocm-hiprtc-census` compiles every
    kernel for every RDNA target, with no GPU.
  - Phase R2 (the backend wired into `frink-core`) is **blocked on an
    RDNA GPU**. The first command on hardware is `cargo test -p
    frink-rocm --features rocm -- --ignored gpu --nocapture` (the Q8_0
    launch test, which has never run).
  - CDNA (MI-series) needs the wave64 audit described in `amd.md`
    first.
- **NPU (XDNA2)**: detection only. A backend means IRON/mlir-aie
  kernels through XRT, which is a research spike that needs hardware.
  FastFlowLM was assessed and rejected (closed kernels, own model
  format); see `amd.md`.
- **Renting**: vast.ai listed **no AMD GPUs** across 2,000 offers on
  2026-11-07, and the account balance was $0. AMD GPUs are on AMD
  Developer Cloud, RunPod, TensorWave and Hot Aisle, which need an
  account the owner creates. Per `CLAUDE.md`, agents do not rent
  hardware. The user authorised vast.ai for Vulkan/AMD testing in the
  session that wrote this file, and that approval ended with it: ask
  again before any rental. When one is approved, see
  `.local-scripts/vast_gpu_receipts.sh` for the safety pattern (a cost
  cap, `trap` teardown, no credentials on the box), and destroy the
  instance afterwards.
- Local Docker images for the ROCm census: `frink-hiprtc`,
  `frink-hiprtc-rust` (x86_64, ROCm 6.4). Run with:
  ```
  docker run --rm --platform linux/amd64 -v $PWD:/src \
    -v frink-rocm-target:/target -v frink-cargo-reg:/root/.cargo/registry \
    -w /src -e CARGO_TARGET_DIR=/target frink-hiprtc-rust \
    cargo test -p frink-rocm --features rocm -- --ignored census
  ```

## Open issues, and what blocks each

| Issue | State | Blocker / next step |
|---|---|---|
| #409 Vulkan | in progress | see above; no hardware needed for steps 1–4 (MoltenVK) |
| #494 AMD | R1 done | RDNA GPU for R2; XDNA hardware for the NPU |
| #501 `frink-core --features metal` has 3 failing tests | fixed in #506 | waiting for a merge |
| #507 templates llama.cpp renders fail or lose their tool format | fixed in #510 | waiting for a merge |
| #508 `preserve_reasoning` default missing: multi-turn reasoning renders a different prompt on five template families | open | changes tokens, so it needs a check against libllama's own render, which `frink parity` does not cover. Add `--reasoning-preserve` with it |
| #509 tool-call `arguments` always a map | open | needs a `supports_object_arguments` probe. Check first for templates that PRINT arguments, which would be silent |
| #476 Q8/Q8_K quantizer last bit vs ggml | open | a bit-exact port made parity *worse*; needs a decision |
| #481 bench median±pop-stddev vs llama-bench mean±sample-stddev | open | user decision (receipts comparability) |
| #408 preempt instead of refusing admission | partly done | "preempt on by default" needs a measurement/decision |
| #404 llama flags | ongoing | remaining bench gaps (`-v`, `-embd`, …) are in `crates/frink-cli/src/llama_flags.rs` (`PINNED_NOT_ACCEPTED`) |
| #405 kernels per kind | CPU done | Metal/CUDA/Vulkan columns in `docs/KERNELS.md` |
| #133, #259 CUDA decode/prefill | open | CUDA kernels are unverified on hardware; one batched rental |
| #403 one file per architecture | open | stale worktree `registry-step9b` (see below) |
| #406, #407, #61, #27 | open | per `north-star.md` order |

Items that need no hardware, in `north-star.md` order: #508 (same
tokens, ahead of everything here), more #404 flags (priority 4), then
Vulkan step 1 or 2 (priority 9, ahead of its rank only if the user
argues it there). For #404, go by what real command lines use:
`-l`/`--logit-bias`, `--pooling`, `--embeddings`, `--reranking`,
`-to`/`--timeout`, `--api-prefix` and `--keep` are the likely next
batch. MTP draft heads (`--mtp` errors by name today) sit under "Not
now" in `north-star.md`; see
[`speculative-decoding.md`](speculative-decoding.md) if that changes.

## Things not to touch

- Worktrees `.claude/worktrees/cuda-kinds` (`feat/cuda-moe-id`) and
  `.claude/worktrees/registry-step9b`
  (`refactor/arch-registry-swa-b-403`) hold another agent's
  uncommitted work (19 and 22 dirty files). The agent is gone, but the
  work is not ours to delete. The same goes for branch `box/agent` (2
  unmerged commits).
- A Python process from an unrelated project (`~/Desktop/Dev/LASA`)
  sometimes loads the CPU. CPU timings taken under it are meaningless
  (3 tok/s instead of 113). Check `uptime` before reading any number.

## Recipes

**Per feature:**
1. Create a worktree: `git worktree add .claude/worktrees/<name> -b
   <branch> origin/main`.
2. Make the change, then sabotage each new test once (break the code,
   see it go red, restore).
3. Run the gates: `cargo fmt --all`, clippy `-D warnings` in debug and
   `--release` (plus the feature builds you touched, for example
   `-p frink-cli --features vulkan`), and `cargo test --workspace`.
4. Open the PR and wait for CI to be green. Merging is the user's call:
   in auto mode `gh pr merge` is denied as a merge without review, so
   say the PR is ready rather than looking for another way to land it.
5. After the merge, delete the worktree's `target/`, the worktree, and
   the local and remote branch. The disk is shared; check `df -h .`.

**Release** (only when the user asks):
1. Open a `chore(release): X - title` PR that bumps:
   - `version = "…"` in the root `Cargo.toml` (workspace and internal
     pins)
   - `Cargo.lock`, with `cargo update --workspace --offline`
   - `FRINK_VERSION=vX` and `frink-inference = "0.X"` in `README.md`
     and `crates/frink-inference/README.md`

   It also dates the CHANGELOG `[Unreleased]` section; merge duplicate
   `###` headings.
2. After the merge, run `git tag -a vX.Y.Z <merge-sha>` with the
   message `vX.Y.Z - title` plus the body, then push the tag.
   `release.yml` and `publish-crates.yml` run from it.
3. Check `gh release view` and crates.io.

**Parity:**
- The reference is llama.cpp at `tools/llama_cpp_pin` (`604cc430`),
  built with `tools/oracle/llama_build.sh $PIN target/oracle/$PIN`
  (BLAS off).
- `.scratch/llama.cpp-604cc430` is the source at the pin: read
  `common/arg.cpp` and `common/jinja/` here, and `models/templates/` is
  the corpus for `FRINK_LLAMA_TEMPLATES`.
- `.scratch/llama.cpp` (at `5b59b83`, BLAS on) is for requantizing
  only: `build-nofma/bin/llama-quantize --allow-requantize [--pure]
  [--imatrix]`. IQ kinds need an imatrix from `frink imatrix`.
- Goldens: CI's x86 answer is authoritative. If CI rewrites a golden
  beyond 3e-6, apply its `oracle-rewritten-goldens` artifact.

**x86 AVX2 tests on this Mac:** `ROSETTA_ADVERTISE_AVX=1 cargo test
--target x86_64-apple-darwin …`. AVX-512 cannot be tested here, even
under emulation.

**Feature flags tests need:**
- `frink-models --features hub` for the Hub tests
- `frink-core --features argv` for the argv tests
- `--features vulkan` on `frink-vulkan` / `frink-core` / `frink-cli`
  for the Vulkan tests

**Test models on this machine:**
- `~/.peerclawd/models/llama-3.2-1b-instruct-q4_k_m.gguf`
- `~/.cache/frink/hub/bartowski__Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- `~/.cache/ferrox/hub/bartowski__SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf`

**Quick Vulkan end-to-end:**
```
cargo build --release -p frink-cli --features vulkan
FRINK_VULKAN=1 ./target/release/frink -m <gguf> -p "The capital of France is" \
  -n 24 --temp 0 --no-cnv -dev cpu
```
The output must be token-identical to the same command with
`FRINK_VULKAN=0`.

## Lessons from this stretch

- **A probe that treats an error as "no" hides the bug behind it.**
  `probe_tools_consumed` read a render error as "this template has no
  tool support", so a broken filter became a silent fallback and every
  request still returned 200. Failures that change behaviour need to
  be loud.
- **A sweep is only as wide as its inputs.** The GGUF sweep rendered one
  plain conversation and missed 47 of 75 templates. Upstream's own test
  corpus, with tools, found them in minutes.
- **Before believing a sabotage result, check that the baseline was
  green and that the named test is the one that failed.** One pass read
  all red because the suite was already failing. And when a mutation
  survives, check that it changed anything: `matrix.clone()` on a
  non-`Clone` type copies the reference, and clap's overrides are mutual.
- **Compare with a recent baseline.** A plain-chat difference against an
  installed 0.52 looked like a regression, and disappeared against 0.54.
- **`\u` in a command can be decoded before it reaches the file.**
  A test's `\u00e9` became `é`, so the test compared correct output with
  the wrong expectation. Write the backslash escaped.
- **`cd` persists between Bash calls here.** A `cd` into `.scratch` left
  the next command running in the wrong tree. Use absolute paths.
- **Run the real model, not only the unit tests.** The Vulkan seam
  passed its tests while no model matvec could reach it (#497).
  Implausible numbers gave it away: a "Vulkan" run at the CPU's speed.
- **When a fast path is gated on the same switch, check what a test's
  "reference" side actually calls.** `apply` became the Vulkan path, so
  the seam tests compared Vulkan with itself (#500).
- **Python slice-and-replace edits on Rust sources:** anchor to the
  start of a line. A `pub struct Vulkan;` in a doc comment, and an
  8-space pattern inside a 12-space line, each matched the wrong place.
  An `assert count == 1` before writing caught both.
- **macOS has no `timeout`.** `timeout N cargo …` fails silently inside
  a `| grep` pipeline and looks like a pass. Use the tool's own time
  limit.
