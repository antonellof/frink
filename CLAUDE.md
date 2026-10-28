# CLAUDE.md

Guidance for agents working in this repo.

## What this is

Frink: a pure-Rust GGUF inference engine (CPU, Metal, CUDA) with an
OpenAI-compatible server. The goal is to be the Rust alternative to
llama.cpp: same models, same commands, same tokens, same or better
speed. Read [`docs/plans/north-star.md`](docs/plans/north-star.md) for
the ranking and [`docs/adr/`](docs/adr/README.md) for the decisions
behind every rule below.

Status: **101** architectures run with evidence
(`capability::AUDITED_GENERIC_GQA`), 4 more have dedicated engines,
and everything else refuses by name. Do not restate the counts
elsewhere; `crates/frink-models/tests/documented_counts.rs` pins this
line.

## Commands

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings      # also with --release
cargo fmt --all

cargo build --workspace --features cuda
cargo test -p frink-cuda --features cuda -- --ignored      # needs a GPU
cargo build -p frink-cli -p frink-server --features metal
cargo test -p frink-metal --features metal -- --ignored    # needs Metal

./target/debug/frink -m model.gguf -p "Hi" -n 64 --temp 0 --no-cnv
./target/debug/frink -m model.gguf -p "Hi" -n 64 -ngl 99   # Metal
./target/debug/frink parity -m model.gguf \
    --dumper target/llama_logits --dumper target/llama_logits_scratch
./target/debug/frink archs | caps | inspect <gguf>
FRINK_MODEL_PATH=model.gguf ./target/debug/frink-server

./target/release/frink bench -m model.gguf -p 512 -n 128 --compare
./target/release/frink bench --suite --fit-host --skip-missing
```

The llama.cpp reference is a source build at the pinned commit in
`.scratch/llama.cpp`; `tools/build_llama_logits.sh` builds the parity
dumper against it. Build it twice (see the script): with one reference,
`frink parity` cannot call a K-quant result wrong.

## Rules

Each rule has an ADR with the reason. Read it before arguing with the
rule.

- **llama.cpp is the oracle** ([0001](docs/adr/0001-llama-cpp-is-the-reference.md)).
  Read the upstream graph, `llama-graph.cpp` and the loader's key reads
  before writing. When frink disagrees, assume frink is wrong.
- **Refuse, never approximate** ([0002](docs/adr/0002-refuse-rather-than-approximate.md)).
  Return `Result` and name what is missing. A refusal is coverage.
- **One file per architecture** ([0003](docs/adr/0003-one-file-per-architecture.md)).
  New architectures go in their own file mirroring
  `src/models/<arch>.cpp`.
- **Parameterise, never copy a code path** ([0004](docs/adr/0004-one-path-per-layer.md)).
  Where two structures must agree, make disagreement fail to compile
  or fail a test.
- **llama.cpp's flags mean the same here** ([0005](docs/adr/0005-same-commands-as-llama-cpp.md)).
  A flag accepted but not honoured is an error.
- **Measure on quiet, rented hardware** ([0006](docs/adr/0006-measurement.md)).
  Agents never benchmark.
- **Numerics follow ggml** ([0009](docs/adr/0009-numerics-follow-ggml.md)).
  Quant layouts, f16 KV, the sampler order and tokenisation match
  llama.cpp. An unlisted deviation is a bug.
- **Coverage before speed** ([0010](docs/adr/0010-coverage-before-speed.md)).
  Check the kernel matrix before profiling; close a missing kernel
  before tuning an existing one.
- **Goldens come from the oracle** ([0011](docs/adr/0011-the-oracle-runs-in-ci.md)).
  A model-level golden is a cache of libllama's answer, never a value
  frink produced.
- **Small files, few crates** ([0008](docs/adr/0008-small-files-few-crates.md)).
  Split a file before it passes ~1000 lines. No new crate for one user.
  Delete dead code.

Rust specifics:

- Clippy with `-D warnings` is a gate, in debug and release.
- `unsafe` needs a `// SAFETY:` comment and a scalar twin it is tested
  against.
- Borrow on per-token paths; hoist feature probes out of loops.
- Tests live in `#[cfg(test)]` beside the code. Sabotage a new test
  once, grep that the mutation landed, and see it go red.
- Never un-ignore a CUDA or Metal hardware test without the hardware.

## Adding an architecture

1. Grep llama.cpp's `src/models/*.cpp` for every op the new graph uses
   and note which other graphs share it.
2. Read the converter's base class and the loader's key reads; keys
   that are dead upstream stay dead here.
3. Write a synthetic fixture that can tell the candidate forms apart (a
   clamp that binds, a permutation that is not the identity).
4. Build a libllama golden; check first-token logits and a greedy
   multi-step decode.
5. Add the row, update the triage, and keep `cargo test --workspace`
   green.

## Before opening a PR

- It closes or advances a named issue, and says which.
- `cargo fmt --all`, clippy with `-D warnings` in debug and `--release`,
  and `cargo test --workspace` are green.
- Anything that changes tokens has `frink parity` evidence; anything
  that needs hardware lists the exact commands still to run.
- New tests were sabotaged once and went red.
- Docs that state what changed are updated in the same PR, and no
  count is restated outside the one place a test pins it.

## Picking work

Take the highest priority in
[`docs/plans/north-star.md`](docs/plans/north-star.md) whose issue
nobody is working on and whose files no open branch is editing.

## Working with agents

- One worktree per agent (`isolation: "worktree"`).
- Do not redo an agent's task while it runs.
- Agents implement and open a PR. They do not tag, publish,
  force-push, rent hardware or benchmark. A kernel merges only after
  its hardware tests ran on real hardware.

## Layout

```
frink-gguf + frink-quant → frink-core → frink-moe → frink-models
                                                  → frink-cli / frink-server
frink-api        wire DTOs, serde only
frink-cuda / frink-metal / frink-vulkan   kernels, below frink-core
frink-safetensors   beside frink-gguf
frink-inference     published facade
```

The FreeToken port (Apache-2.0) lives in `frink-core` (expert
residency) and `frink-server::policy` (serving).
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md) is a
licence obligation and must stay accurate.

Docs: [`docs/FEATURES.md`](docs/FEATURES.md),
[`docs/CLI.md`](docs/CLI.md), [`docs/MODELS.md`](docs/MODELS.md),
[`docs/API.md`](docs/API.md), [`docs/CONFIG.md`](docs/CONFIG.md),
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

Never add Co-Authored-By, Claude-Session, or Generated with Claude Code
lines to commit messages or PR bodies.
