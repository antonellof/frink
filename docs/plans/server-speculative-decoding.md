# Speculative decoding in the server

Status: **not started**. The engine half is built, tested and lossless;
the server cannot reach it. This says what wiring it actually costs,
because the obvious answer ("call the function") is wrong for a reason
worth writing down.

## What exists

- `frink_models::speculative`: the `Drafter` trait,
  `PromptLookupSpeculator` (n-gram over the history, no second model),
  `speculative_decode_with` / `_observed`, and the Leviathan / Chen
  rejection rule with `accept_or_resample` pinned by unit tests.
- `frink_models::draft_model::DraftModelSpeculator`: a second GGUF as
  the drafter, refused at construction when the vocabularies differ.
- `frink run -d draft.gguf`: a real generation path, with the
  refusals it needs (no grammar, no recurrent target, no recurrent
  draft, a drafter whose KV is on the device).

## What does not

```
$ grep -rn 'speculat' crates/frink-server/src --include=*.rs -l
crates/frink-server/src/stats/requests.rs
$ grep -rn 'with_speculation' crates/frink-server/src | grep -v test
(nothing)
```

The server has an acceptance-rate metric, a test that the metric
reaches the admin ring, and **no producer for it**. A metrics column no
code path can fill reads as coverage, which is the same defect class as
a gate that cannot fire.

The serving engines ship several drafting methods between them
(n-gram, suffix, EAGLE-style, MLP and MTP heads), and llama.cpp's
server ships `--model-draft` with `--draft-max` / `--draft-min`. frink
ships none of them over HTTP.

## Why it is not "call the function"

`speculative_decode_with` does its OWN sampling from a
`SamplingParams`, because the rejection rule needs `p(x)` from the
target and `q(x)` from the draft under the SAME sampler. The server's
sampler is `crate::sample_step::sample_next` inside
`generate::sample_until_stop`, and it is not that: it carries the
grammar machine, the penalty window over `prompt ++ generated`, logit
bias, `n_probs`, the stop-string withhold rule and the UTF-8 stream.

So a naive wiring gives a request two samplers that must agree about
one thing -- **this repo's dominant bug shape** -- and the failure is
silent: fluent text at a plausible accept rate, with the grammar and
the penalties quietly not applied to the drafted positions.

## The shape that would be correct

One of these, and the choice is the design decision this plan exists to
force:

1. **Verify through the server's sampler.** `speculative_decode_with`
   takes a callback that, given the target's logits at a position and
   the draft's proposal, answers accept/reject and returns the token --
   implemented by `sample_step`. The rejection rule stays in
   `frink-models`; the DISTRIBUTION comes from the one sampler the
   non-speculative path uses. Cost: a new seam in the speculative
   module, and every sampler feature has to be expressible for a
   position that may be rolled back (the grammar machine in particular
   needs a checkpoint/restore).
2. **Refuse what cannot be verified.** Speculate only for requests
   whose sampling is reproducible by `SamplingParams` -- no grammar, no
   logit bias, no penalties -- and take the ordinary path otherwise.
   Cheap, honest, and narrow enough that a coding agent's requests
   (which use tools, hence grammars) would never hit it.

(1) is the real answer and (2) is a defensible first step ONLY if the
refusal is per-request and visible in the response's usage block, not a
silent fallback.

## What else has to hold

- **The continuous batcher.** Speculation and batching are two
  schedulers for one KV cache; llama.cpp's server disables the draft
  when a slot is shared. frink should refuse the pair by name before
  it is measured.
- **Paged KV and the prefix cache.** The drafter rolls back the
  positions the target rejects; a paged store and a radix cache both
  have to be able to undo them. `KvCache::truncate` already refuses a
  middle position for a recurrent layer (`frink_core::
  recurrent_state`), which is the same question.
- **Device-resident KV.** `frink run` already refuses a drafter whose
  KV lives on the device, because it cannot roll back rows it cannot
  see. The server's Metal path is exactly that case, so on Apple
  silicon this is a CPU-only feature until the device KV mirror
  (`KvCache::metal_attn`, 0.25.0) learns to truncate.
- **The metric.** `stats::requests::with_speculation` takes
  (accepted, drafted, steps, per-position rates). Filling it is the
  point: a speedup without an accept rate cannot be reproduced or
  debugged.

## How it would be measured

Against `benchmarks/suite.json` on a rented box, interleaved A/B, with
the acceptance length reported beside the tok/s. A draft model costs
memory and prefill; the ledger has to show both sides or the number
means nothing.
