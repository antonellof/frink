# 0005. Same commands and flags as llama.cpp's tools

Status: Proposed
Date: 2026-10-28

## Context

Most of what "alternative to llama.cpp" means to a user is that the
command they already have works. Frink already mirrors part of it
(`-m`, `-p`, `-n`, `-ngl`, `frink bench`'s `-p/-n/-r`), but the rest
grew its own names, and `docs/CLI.md` is 1400+ lines describing them.

## Decision

- Each frink entry point maps to one llama.cpp tool and accepts its
  flags with the same names and meanings:

  | llama.cpp | frink |
  |---|---|
  | `llama-cli` | `frink` / `frink run` |
  | `llama-server` | `frink-server` (also `frink serve`) |
  | `llama-bench` | `frink bench` |
  | `llama-quantize` | `frink quantize` |
  | `llama-perplexity` | `frink perplexity` |

- `llama-server` flags and its `LLAMA_ARG_*` environment variables are
  accepted. Frink's own `FRINK_*` names stay as aliases.
- `frink bench` prints the same table and JSON fields as `llama-bench`,
  so one tool renders both.
- A flag frink accepts but does not honour is an error, never silently
  ignored. A deliberate difference is listed in one table in
  `docs/CLI.md`.
- Frink-only features use long flags with no short form, so they never
  collide with a future llama.cpp flag.

## Consequences

A test parses llama.cpp's `common/arg.cpp` at the pin and lists every
flag frink does not accept, so the gap is a number, not a feeling.
`docs/CLI.md` shrinks to the differences plus frink-only features.
