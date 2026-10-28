# 0002. Refuse rather than approximate

Status: Accepted
Date: 2026-10-28

## Context

Before 2026-09, unrecognised architectures fell onto a shared generic
path and ran. Five of them were silently wrong: fluent output, wrong
model. A user cannot tell a wrong answer from a right one.

## Decision

- An architecture runs only if it is on `capability::AUDITED_GENERIC_GQA`
  with a libllama golden, or has a dedicated engine with its own
  evidence. Everything else returns `LoadError::UnauditedArchitecture`.
- A refusal is one sentence that names what is missing. The detailed
  reasoning lives in the triage test or the issue, not in the error
  string.
- A GGUF key or tensor frink does not implement stops the load. It is
  never ignored.

## Consequences

Frink runs fewer files than llama.cpp would, and says so. A refusal is
a coverage gap, not a defect. `FRINK_ALLOW_UNAUDITED_ARCH=1` exists for
investigation only.
