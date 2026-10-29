# 0011. The oracle runs in CI, and the pin moves on a schedule

Status: Accepted
Date: 2026-10-28

## Context

ADR 0001 makes llama.cpp the oracle, but CI never builds libllama.
Goldens are generated on a laptop and checked in, so a regression
against llama.cpp shows up only when someone regenerates them. Moving
the pin from `5b59b83` to `604cc430` (913 commits) took a manual audit
and found three served architectures whose upstream behaviour had
changed under us.

## Decision

- CI builds llama.cpp at the pin (cached by commit) and the
  `tools/llama_logits.c` dumper against it, and runs `frink parity` on
  every synthetic fixture: tokenizer, first-token logits and a greedy
  multi-step decode.
- A scheduled job runs the same suite against llama.cpp `master` and
  reports which architectures drift. Moving the pin starts from that
  list.
- A checked-in model-level golden is a cache of the oracle's answer,
  not a second source of truth: CI regenerates it and fails when it
  differs. Kernel goldens checked against independent NumPy references
  are unaffected.

## Consequences

CI gets slower by one cached C++ build. In exchange, done-means point 2
is enforced on every PR rather than on each audit. Tracked in #402.
