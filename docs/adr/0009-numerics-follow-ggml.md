# 0009. Numerics follow ggml unless a deviation is written down

Status: Accepted
Date: 2026-10-28

## Context

"Same tokens" depends on more than the graph. Frink has diverged from
llama.cpp through choices that each looked harmless: an f32 host KV
cache where llama.cpp keeps f16, a repetition penalty applied per
occurrence instead of once per candidate, temperature applied before
the truncation filters, and penalties that did not see the prompt over
HTTP. Each produced different tokens from the same file and seed.

## Decision

- Quant block layouts, dequantisation and the activation quantisation
  used by integer dots match ggml bit for bit.
- KV cache precision defaults to llama.cpp's (f16), and the sampler
  chain runs in llama.cpp's order with its semantics.
- Tokenisation follows `llama-vocab.cpp` at the pin, including its
  pre-tokenizer regexes.
- Accumulation order and fused kernels may differ when the result stays
  inside the reference spread `frink parity` measures between two
  llama.cpp builds.
- Any other deviation is listed, with its reason and measured effect,
  in `docs/plans/llama-cpp-gap-inventory.md`. An unlisted deviation is
  a bug.

## Consequences

Some optimisations that change rounding are off the table unless they
stay inside the reference spread. A user switching from llama.cpp gets
the same output at the same seed.
