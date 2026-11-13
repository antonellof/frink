# Contributing

Frink aims to be the Rust alternative to llama.cpp: the same models,
the same commands, the same tokens. [`docs/plans/north-star.md`](docs/plans/north-star.md)
is the goal and the one ranking of open work, and
[`docs/adr/`](docs/adr/README.md) holds the decisions behind every rule.
[`CLAUDE.md`](CLAUDE.md) states the rules in short form; this file adds
the process around them.

## What to make: a PR or an issue

| What you have | What you make |
|---|---|
| A feature that is complete, builds, and has tests | A branch and a pull request |
| A defect, a gap, an open question | A GitHub issue |
| A defect you fixed inside the feature you were already writing | Stays in the PR, described in its body |
| A defect in code nobody asked you to touch | An issue, not a drive-by commit |

To pick work, take the highest priority in the north star whose issue
nobody is working on and whose files no open branch is editing.

An issue, opened as soon as a defect is found, states what is wrong in
one sentence, the input and wrong output or crash, the
`path/to/file.rs:line`, and whether untrusted input can reach it
(security problems go through [`SECURITY.md`](SECURITY.md)). If it is
two structures that must agree with nothing enforcing it, say so in the
title.

### Branches and pull requests

- One feature per branch, `feat/<short-name>` or `fix/<short-name>`,
  cut from `main`.
- The PR closes or advances a named issue and says which, and says what
  changed, why, and what would have to be true for it to be wrong. A
  test that goes red when the change is reverted is evidence; "it
  compiles" is not.
- Token-changing work carries `frink parity` evidence. Hardware work
  lists the exact commands still to run; a kernel merges only after its
  hardware tests ran on real hardware.
- Docs that describe what changed are updated in the same PR.

## Checks before a PR

```
cargo fmt --all
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --release -- -D warnings
```

The release clippy matters: `debug_assert!` type-checks its argument in
release, so code gated on `debug_assertions` can fail only there.

On Apple silicon, several kernels have a NEON path and a scalar
fallback the local build never compiles, so a lint in the fallback is
red only on CI. Check a second architecture before pushing:

```
scripts/check-cross-target.sh           # clippy for x86_64-apple-darwin
rustup target add x86_64-apple-darwin   # once, if it refuses
```

To run it as a pre-push hook (`.githooks/` is gitignored; append rather
than overwrite an existing hook):

```
mkdir -p .githooks
echo 'exec "$(git rev-parse --show-toplevel)/scripts/check-cross-target.sh"' \
  >> .githooks/pre-push
chmod +x .githooks/pre-push
git config core.hooksPath .githooks
```

### What else CI runs

- `cargo clippy -p frink-cli -p frink-server --features cuda` and
  `frink-cuda`, compiled without a GPU or toolkit; no CUDA kernel runs.
  CUDA hardware tests stay `#[ignore]`d.
- On `macos-latest`, clippy with `--features metal` for `frink-metal`,
  `frink-cli` and `frink-server`, and `frink-metal`'s `--ignored` tests
  on the runner's GPU.
- `frink-vulkan` on lavapipe, and `frink-rocm` compiling every kernel
  for every RDNA target through hiprtc.
- `frink-cli` in release with the tarball's features (`serve` on Linux,
  `serve metal` on macOS).
- The llama.cpp oracle (`scripts/oracle.sh` locally), plus a weekly run
  against llama.cpp `master` that reports drift.
- `ui/`: `npm run licenses`, `typecheck`, `lint` and `build`, but not
  `npm test`; run `npm run check` in `ui/` before touching `ui/src/lib/`.

### Goldens and tests

- Tests live in `#[cfg(test)]` or a crate's `tests/`. Sabotage a new
  test once and see it go red. Kernel goldens are checked against an
  independent reference.
- The goldens in `crates/frink-models/tests/` are libllama's CPU
  answers, so `tests/common/` pins every test process to the host
  backend, even on a `--features metal` build, and fails if a Metal
  dispatch is encoded after the pin. `FRINK_GOLDEN_BACKEND=metal` runs
  the same comparisons on the GPU at `METAL_GOLDEN_TOL`
  (`tests/common/backend.rs`) as a measurement; the Metal gate is
  `decode_step_graphs::metal_decode_matches_cpu_and_llama_cpp`
  (`--ignored`, needs a GPU).

## Adding a model architecture

An architecture not on an audited list is refused, and each refusal
says why (`capability::unaudited_triage`). To move a row from refused to
served, follow the steps in [`CLAUDE.md`](CLAUDE.md#adding-an-architecture),
with these specifics:

- Read the generic builders the graph calls as well; they branch on the
  architecture too.
- The file is `crates/frink-models/src/models/<arch>.rs`. Split a file
  that would pass about 1000 lines first, as a pure move.
- The fixture script lives under `scripts/` (for example
  `scripts/make_phi2_fixture.py`). The test under
  `crates/frink-models/tests/` covers prefill, decode, continuous
  batching and a greedy multi-step decode, and records the KL divergence
  and max logit delta in its header, as `phi2_graphs.rs` does.
- On a real checkpoint, `frink parity -m <file>` compares tokenizer and
  logits against libllama.

## Benchmarks and presets

Read [ADR 0006](docs/adr/0006-measurement.md) and
[`benchmarks/README.md`](benchmarks/README.md) first.

- Preset fields are confirmed against a primary source or listed in
  `best_effort_fields`. A hardware claim names its machine, or says
  compile-tested only.
- Speed numbers come from `frink bench --suite` against `llama-bench`.
  `benchmarks/RESULTS.md` is generated from runs that
  `benchmarks/suite.json` drives; never edit it by hand. To add a model,
  add a `suite.json` entry, put the GGUF under `models/`, and run
  `frink bench --suite --id <id> --fit-host --skip-missing`.
- A run that stops before timing anything means the host is busy, hot
  or short on memory. Fix the host; `--max-load 0` produces a number
  nobody may publish.
- Never force a thread count on either engine; llama.cpp loses 2-4x
  above the performance cores.
- Apple Silicon spread is about 20%. A smaller claim needs an
  interleaved A/B, counting rounds won.
- Record negative results in the issue or PR (`.scratch/` is
  gitignored).

## Running several agents at once

Agents follow [`CLAUDE.md`](CLAUDE.md#working-with-agents). Partition
the work by file before launching: two items that need the same file
are one item, or run one after the other. An agent reports what it did
not do; 80% with the missing 20% named beats a quietly narrowed scope.
crates.io versions cannot be unpublished, which is why agents never
publish.

## Releasing

Only when a maintainer asks for a release.

1. Open a `chore(release): X.Y.Z - title` PR that bumps
   `version = "…"` in the root `Cargo.toml` (the workspace and every
   internal pin), `Cargo.lock` with `cargo update --workspace
   --offline`, and `FRINK_VERSION=vX.Y.Z` and `frink-inference = "X.Y"`
   in `README.md` and `crates/frink-inference/README.md`. It also dates
   the CHANGELOG's `[Unreleased]` section, keeps an empty
   `[Unreleased]` above it, and merges duplicate `###` headings.
2. After the merge, tag the merge commit:
   `git tag -a vX.Y.Z <merge-sha>`, with the message `vX.Y.Z - title`
   and the release PR's body, then push the tag. `release.yml` builds
   the tarballs, and `publish-crates.yml` publishes every crate. A tag
   push publishes for real. Only a manual dispatch defaults to a dry
   run.
3. Check `gh release view vX.Y.Z` for both tarballs and their
   checksums, and that the publish run finished green.

## Documentation

Capabilities are in `docs/FEATURES.md`, flags in `docs/CLI.md`, models
in `docs/MODELS.md`, HTTP routes in `docs/API.md`, environment in
`docs/CONFIG.md`, agent setup in `docs/AGENTS_COOKBOOK.md`, decisions in
`docs/adr/`, speed in `benchmarks/RESULTS.md` (generated) and its method
in `benchmarks/README.md`, and open work in `docs/plans/north-star.md`
and `docs/plans/README.md`.

Do not duplicate those. A plan in `docs/plans/` designs one north-star
item; when it lands, its decisions become ADRs and the plan is deleted
([ADR 0007](docs/adr/0007-issues-plans-adrs.md)). A count is stated in
at most one place, pinned by a test, or not at all.
