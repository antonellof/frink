# 0008. Small files, narrow modules, few crates

Status: Accepted
Date: 2026-10-28

## Context

Large files are why adding a model used to mean editing a 7000-line
file, so nobody did. Every split that worked happened because someone
was about to add to a file and split it first (`repack.rs`,
`sampling.rs`, `decoder.rs`, `loader.rs`). `frink-edge` became a crate
instead of an integration and 5,400 of its lines were never called.

## Decision

- If a change would push a file past ~1000 lines, split it first as a
  pure move, then make the change.
- One concept per module.
- No new crate for code one crate uses.
- Dead code is deleted unless it is wired to a named issue.
- A gate that cannot fire is removed or fixed; check a refusal's
  condition is reachable.

## Consequences

The largest files today (`frink-server/src/lib.rs`,
`frink-metal/src/gpu.rs`, `frink-quant/src/lib.rs`,
`frink-metal/src/attn.rs`) are each split before their next feature.
