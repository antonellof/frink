# 0001. llama.cpp is the reference and the oracle

Status: Accepted
Date: 2026-10-28

## Context

Frink's goal is to be a Rust alternative to llama.cpp. A golden file
this project writes can only prove frink agrees with itself; several
bugs here passed every in-repo golden and disagreed with llama.cpp.
llama.cpp also moves weekly, so "matches llama.cpp" means nothing
without a commit.

## Decision

- Correctness means agreeing with llama.cpp at a pinned commit, recorded
  in the current parity audit under `docs/plans/`. Moving the pin is a
  deliberate change with its own audit.
- `frink parity` (with `tools/llama_logits.c` linked against libllama)
  is the oracle: tokenizer output, first-token logits, and a greedy
  multi-step decode. Prefill parity alone is not enough.
- When frink and llama.cpp disagree, the default assumption is that
  frink is wrong. The exceptions are written down with evidence (the
  K-quant drift in `docs/plans/llama-cpp-gap-inventory.md` §10).
- New behaviour is read from llama.cpp's source before it is written:
  the graph in `src/models/*.cpp`, the generic builder in
  `llama-graph.cpp`, and the key reads in `llama-model.cpp`.

## Consequences

Every architecture needs a libllama-built golden, which means a source
build of llama.cpp at the pin. Fixtures are synthetic and small, so no
real checkpoint download is needed to add a row.
