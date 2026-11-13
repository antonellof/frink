# Contributing

Frink aims to be the Rust alternative to llama.cpp: the same models,
the same commands, the same tokens. [`docs/plans/north-star.md`](docs/plans/north-star.md)
is the goal and the one ranking of open work, and
[`docs/adr/`](docs/adr/README.md) holds the decisions behind every rule
below. [`CLAUDE.md`](CLAUDE.md) is the same rules in short form, for
coding agents.

## What to make: a PR or an issue

| What you have | What you make |
|---|---|
| A feature that is complete, builds, and has tests | A branch and a pull request |
| A defect, a gap, an open question | A GitHub issue |
| A defect you fixed inside the feature you were already writing | Stays in the PR, described in its body |
| A defect in code nobody asked you to touch | An issue, not a drive-by commit |

A fix inside an unrelated PR is a fix nobody reviewed, in a diff nobody
expected.

To pick work, take the highest priority in the north star whose issue
nobody is working on and whose files no open branch is editing.

### Issues

Open one as soon as a defect is identified, before deciding whether to
fix it. It states:

1. What is wrong, in one sentence.
2. The failure, concretely: the input, and the wrong output or the
   crash.
3. Where: `path/to/file.rs:line`.
4. Whether untrusted input can reach it. A security problem goes
   through [`SECURITY.md`](SECURITY.md), not a public issue.

If it is an instance of the repo's most common bug shape, two
structures that must agree about one thing with nothing enforcing it,
say so in the title.

### Branches and pull requests

- One feature per branch, `feat/<short-name>` or `fix/<short-name>`,
  cut from `main`, not from another feature branch.
- The PR closes or advances a named issue and says which.
- The body says what changed, why, and what would have to be true for
  it to be wrong. "It compiles" is not evidence; a test that goes red
  when the change is reverted is.
- Anything that changes tokens carries `frink parity` evidence.
  Anything that needs hardware lists the exact commands still to run;
  a kernel merges only after its hardware tests ran on real hardware.
- Docs that describe what changed are updated in the same PR.

## Checks before a PR

Run all five. CI runs the first four on the workspace and the release
clippy on `frink-cli`:

```
cargo fmt --all
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --release -- -D warnings
```

The release clippy is not redundant: `debug_assert!` type-checks its
argument in release, so code gated on `debug_assertions` can compile in
debug and fail in release.

On an Apple-silicon checkout that is not enough, because several
kernels exist twice: a NEON path under `#[cfg(target_arch = "aarch64")]`
and a portable scalar fallback the local build never compiles. A lint
that fires only in the fallback is invisible locally and red on CI.
Compile for a second architecture before pushing:

```
scripts/check-cross-target.sh    # clippy for x86_64-apple-darwin, ~2s warm
rustup target add x86_64-apple-darwin   # once, if it refuses
```

To run it automatically, call it from a local pre-push hook under
`.githooks/`. That directory is gitignored and may already hold a hook
of your own, so add the line rather than replacing the file, make the
hook executable, and point git at the directory once per clone:

```
mkdir -p .githooks
echo 'exec "$(git rev-parse --show-toplevel)/scripts/check-cross-target.sh"' \
  >> .githooks/pre-push
chmod +x .githooks/pre-push
git config core.hooksPath .githooks
```

CI's Linux `clippy` job compiles the same fallbacks, so the hook only
finds them a round trip earlier.

### What else CI runs

- The `cuda` feature must compile without a GPU or CUDA toolkit:
  `cargo clippy -p frink-cli -p frink-server --features cuda`. CI
  builds that chain and `frink-cuda`, but runs no CUDA kernel.
- On `macos-latest`, CI compiles
  `cargo clippy -p frink-metal -p frink-cli -p frink-server --features metal`,
  and a second job runs `frink-metal`'s `--ignored` tests on the
  runner's GPU. CUDA hardware tests stay `#[ignore]`d; never un-ignore
  one without the hardware.
- `frink-vulkan` runs on lavapipe (Vulkan on the CPU), and
  `frink-rocm` compiles every kernel for every RDNA target through
  hiprtc, with no GPU.
- `frink-cli` in release, with the features the tarball ships (`serve`
  on Linux, `serve metal` on macOS).
- The llama.cpp oracle: libllama at the pin in `tools/llama_cpp_pin`,
  goldens regenerated, `frink parity` on every fixture
  (`scripts/oracle.sh` runs the same thing locally). A weekly job runs
  it against llama.cpp `master` and reports what drifts.
- The `ui/` job runs `npm run licenses`, `typecheck`, `lint` and
  `build`. It does not run `npm test`, so run `npm run check` in `ui/`
  before a PR that touches `ui/src/lib/`.

### Goldens and tests

- Tests live in `#[cfg(test)]` beside the code, or in a crate's
  `tests/`. Sabotage a new test once, check the mutation landed, and
  see it go red.
- A model-level golden is a cache of libllama's answer, never a value
  frink produced. Kernel goldens are checked against an independent
  reference, not only against frink itself.
- The goldens in `crates/frink-models/tests/` are libllama's CPU
  answers, so the shared helpers in `tests/common/` pin every test
  process to the host backend, even on a `--features metal` build, and
  fail if any Metal dispatch is encoded after the pin.
  `FRINK_GOLDEN_BACKEND=metal` is the opt-in Metal golden mode: the
  same comparisons on the GPU at `METAL_GOLDEN_TOL` (see
  `tests/common/backend.rs` for the measured spread behind it). That
  mode is a measurement, since many suites also assert host-only
  properties; the Metal gate is
  `decode_step_graphs::metal_decode_matches_cpu_and_llama_cpp`
  (`--ignored`, needs a GPU).
- `unsafe` needs a `// SAFETY:` comment and a scalar twin it is tested
  against.

## Adding a model architecture

A model frink only partly implements must stop with an error that names
what is missing, never compute something else. An architecture that is
not on an audited list is refused, and each refusal says why
(`capability::unaudited_triage`). To move a row from refused to served:

1. Read llama.cpp's graph for it (`src/models/<arch>.cpp`), the tensor
   loader and hyperparameter reads, and any generic builder the graph
   calls, since those branch on the architecture too. Grep the other
   graphs for each op it uses, so the change lands as one table or
   seam rather than a special case. Keys that are dead upstream stay
   dead here.
2. Put it in its own file, `crates/frink-models/src/models/<arch>.rs`,
   mirroring the upstream file. Parameterise the existing code path;
   never copy one to vary it. If a file would grow past about 1000
   lines, split it first, as a pure move.
3. Build a small synthetic fixture with a script under `scripts/` (for
   example `scripts/make_phi2_fixture.py`) that can tell the candidate
   forms apart: a clamp that binds, a permutation that is not the
   identity. Run it through libllama to get golden logits, and add a
   test under `crates/frink-models/tests/` that checks frink against
   them on the prefill, decode and continuous-batching paths, including
   a greedy multi-step decode. Record the KL divergence and the max
   logit delta in the test's header, as
   `crates/frink-models/tests/phi2_graphs.rs` does.
4. Sabotage the new code once and confirm the test goes red.
5. Add the catalog row and update the triage. On a real checkpoint,
   `frink parity -m <file>` compares frink's tokenizer and logits
   against libllama's.

## Benchmarks and presets

Read [ADR 0006](docs/adr/0006-measurement.md) first.

- Preset fields: confirm each one against a primary source, or list it
  in `best_effort_fields`.
- A hardware claim names the machine it was measured on, or says
  compile-tested only.
- Speed numbers come from `frink bench --suite` against `llama-bench`,
  with no HTTP in the loop. `benchmarks/suite.json` drives the runs and
  `benchmarks/RESULTS.md` is generated from them, so never edit that
  table by hand. To measure a new model, add an entry to `suite.json`,
  put the GGUF under `models/`, then run
  `frink bench --suite --id <id> --fit-host --skip-missing`.
- A run that stops before timing anything is telling you the host is
  busy, hot, or short on free memory. Fix the host rather than reaching
  for `--max-load 0`, which turns off all three checks and produces a
  number nobody may publish. `benchmarks/README.md` has each one.
- Never force a thread count on either engine. llama.cpp defaults to
  performance cores and loses 2-4x above them, so pinning both to the
  same count flatters frink instead of making the comparison fair.
- Run-to-run spread on Apple Silicon is around 20%. A claim under that
  needs an interleaved A/B: alternate the two binaries round by round in
  one session and count rounds won. Two batches of runs will not do it.
- Record negative results too, in the issue or the PR description.
  (`.scratch/` is gitignored, so notes left there never reach the
  repository.)

## Running several agents at once

Agents follow [`CLAUDE.md`](CLAUDE.md). The rules that break first as
the number of agents grows:

- One worktree per agent, and one agent owns a file. Partition the work
  by file before launching. Two items that need the same file are one
  item, or they run one after the other.
- Agents implement and open a PR. They do not benchmark (a loaded
  machine reads 25-45% low; an agent that wants a number opens an issue
  asking for it), and they do not tag, publish, force-push or rent
  hardware. crates.io versions cannot be unpublished.
- An agent reports what it did not do. A task delivered at 80% with the
  missing 20% named is worth more than one that quietly narrowed its
  scope.
- Each `target/` is 10 to 40 GB. Check `df -h .` before a full build,
  and delete the `target/` of a finished worktree.

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

| Doc | Role |
|---|---|
| `docs/FEATURES.md` | Capabilities overview |
| `docs/CLI.md` | CLI flags and examples |
| `docs/MODELS.md` | Supported models |
| `docs/API.md` | HTTP API matrix |
| `docs/CONFIG.md` | Environment variables |
| `docs/AGENTS_COOKBOOK.md` | Pointing an IDE or coding agent at the server |
| `docs/adr/` | The decisions, one file each |
| `benchmarks/RESULTS.md` | Speed vs llama.cpp (generated) |
| `benchmarks/README.md` | How those numbers are measured |
| `docs/plans/north-star.md` | The goal and the ranking of the open work |
| `docs/plans/README.md` | Index of the plans in flight |

Do not duplicate those. A plan in `docs/plans/` is the design of one
item on the north star; when the work lands its decisions become ADRs
and the plan is deleted, with git as the archive
([ADR 0007](docs/adr/0007-issues-plans-adrs.md)). A count is stated in
at most one place, pinned by a test, or not at all.
