# Architecture decision records

One file per decision that shapes the code and would be expensive to
reverse. Each is short: the context that forced it, the decision, and
what it costs. A decision is changed by a new ADR that supersedes the
old one, never by editing the old one's Decision section.

Status is one of `Accepted`, `Proposed` or `Superseded by NNNN`.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-llama-cpp-is-the-reference.md) | llama.cpp is the reference and the oracle | Accepted |
| [0002](0002-refuse-rather-than-approximate.md) | Refuse rather than approximate | Accepted |
| [0003](0003-one-file-per-architecture.md) | One file per architecture, built from shared blocks | Accepted |
| [0004](0004-one-path-per-layer.md) | One code path per layer, parameterised, never copied | Accepted |
| [0005](0005-same-commands-as-llama-cpp.md) | Same commands and flags as llama.cpp's tools | Accepted |
| [0006](0006-measurement.md) | Performance claims come from quiet, rented hardware | Accepted |
| [0007](0007-issues-plans-adrs.md) | Issues track work, ADRs record decisions, plans are short-lived | Accepted |
| [0008](0008-small-files-few-crates.md) | Small files, narrow modules, few crates | Accepted |

## Template

```markdown
# NNNN. Title

Status: Proposed | Accepted | Superseded by NNNN
Date: YYYY-MM-DD

## Context
What forced a decision. Facts and measurements, one paragraph.

## Decision
What we do, stated so a reviewer can check a PR against it.

## Consequences
What it costs, what it rules out, what to watch.
```
