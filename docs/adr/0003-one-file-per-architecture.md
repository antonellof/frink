# 0003. One file per architecture, built from shared blocks

Status: Accepted
Date: 2026-10-28

## Context

llama.cpp adds a model as one file in `src/models/<arch>.cpp` that
calls shared builders (`build_norm`, `build_attn`, `build_ffn`,
`build_moe_ffn`) from `llama-graph.cpp`. Its architecture list and
tensor names live in `llama-arch.cpp`, hyperparameter reads in
`llama-model.cpp`.

Frink instead spreads an architecture across about a hundred feature
modules in `crates/frink-models/src/` (one per optional behaviour) and
a set of tables in `capability.rs` (5000+ lines, mostly triage prose)
and `loader/arch_tables.rs`. To answer "what does `kolibri1` do" you
grep a dozen tables. To add one you edit all of them, and a table you
forget fails silently. This is the repo's dominant bug shape: several
structures that must agree, with nothing enforcing it.

## Decision

Mirror llama.cpp's layout, in Rust:

| llama.cpp | frink |
|---|---|
| `llama-arch.cpp` (enum, names, tensor names) | `frink-models/src/arch.rs` |
| `llama-model.cpp` (hparams) | `frink-models/src/loader/` |
| `llama-graph.cpp` (`build_attn`, `build_ffn`, ...) | `frink-models/src/decoder/` blocks |
| `src/models/<arch>.cpp` | `frink-models/src/models/<arch>.rs` |

Each `models/<arch>.rs` returns a `LayerPlan` (the per-layer choices:
norm placement, RoPE layers, window, gates, FFN kind, MoE routing)
built from the hparams. The shared decoder executes a `LayerPlan` on
every backend. Feature tables are derived from the plans, not restated.
A file names the llama.cpp source it was read from, with line numbers.

Migration is incremental: new architectures go in `models/` from the
first one; existing ones move a family at a time, each move a pure
refactor checked by its existing golden.

## Consequences

Adding an architecture becomes: one file, one registry line, one
fixture, one golden. Reviewers compare `models/<arch>.rs` against
`src/models/<arch>.cpp` side by side. The triage prose in
`capability.rs` moves to the triage test and shrinks to one line per
row. This supersedes the open half of
`docs/plans/model-layer-reorg.md`.
