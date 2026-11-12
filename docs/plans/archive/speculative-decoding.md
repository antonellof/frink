---
name: "speculative decoding with a real draft model, on CPU, CUDA and Metal"
overview: "THE ONE ITEM THAT RAISES DECODE THROUGHPUT WITHOUT BUYING HARDWARE. Decode is memory-bandwidth bound: to emit one token the engine reads every weight in the model, so a 17 GB checkpoint on a 960 GB/s card cannot exceed ~56 tok/s no matter how good the kernels are. That ceiling is arithmetic, not engineering. Speculative decoding breaks it by changing WHAT IS READ PER TOKEN rather than how fast it is read: a small draft model proposes k tokens, the target verifies all k in ONE pass over its weights, and the rejection rule guarantees the output distribution is exactly the target's. FRINK ALREADY HAS THE HARD HALF. `speculative.rs` implements the Leviathan/Chen rejection rule, is lossless at every temperature rather than only at `--temp 0`, and proves it against 200k sampled tokens. STATUS 2026-10-28: steps 1 to 3 have landed. `draft_model.rs` holds `DraftModelSpeculator` (a second GGUF as the drafter, its KV rolled back on rejection, a vocabulary-size mismatch refused by name), and `frink run` takes `-d` / `--model-draft`, `--draft-max` (alias `--draft`) and `--draft-p-min`. The server speculates with prompt-lookup since 0.28.0; it has no draft-model path yet, which is step 4 and `server-speculative-decoding.md`. This plan added a second GGUF as the draft model, wired it through the CLI, and refuses when the two checkpoints do not share a vocabulary. SCOPE: CPU, CUDA and Metal, which is every backend frink has. No AMD-specific work."
---

# Speculative decoding with a real draft model

## Why this is the highest-leverage decode item

Decode reads the whole model per token. That gives a hard ceiling:

```
tokens/sec <= memory bandwidth / model bytes
```

A 17 GB checkpoint at 960 GB/s cannot pass ~56 tok/s. Better kernels
move the engine toward that number and cannot move it past. Every other
decode optimisation on the roadmap is a fight for the gap between where
frink is and where that ceiling sits. This item moves the ceiling.

The mechanism is one sentence: a 2 GB drafter proposes k tokens, the
17 GB target checks all k in a single pass over its weights, good
guesses are kept and bad ones discarded, and the text is exactly what
the target would have written alone. Published chases of this on one
desktop card report 38 to over 100 tok/s on a 27B model, and every gain
after the first came from making the guesser better and the check
cheaper rather than from new hardware.

## What frink already has

`crates/frink-models/src/speculative.rs`, 1343 lines:

- `speculative_decode_with`, which verifies a proposed block in one
  `forward_batch` call and does not care who proposed it.
- `accept_or_resample`, the speculative-sampling rejection rule: accept
  a draft token with probability `min(1, p(x)/q(x))`, and on rejection
  resample from the normalised residual `max(0, p - q)`. **Lossless at
  every temperature**, not only at `--temp 0`, which is the difference
  between the claim and the property. The old argmax test was the
  `temperature = 0` special case and silently biased generation toward
  the target's argmax above it.
- Three tests that would catch a regression in that: 200k tokens through
  the rule against deliberately bad draft distributions, a temperature
  1.0 decode against exactly enumerated per-position marginals, and
  token-for-token identity with a plain loop at temperature 0.
- `frink-api`'s `Usage` already declares `acceptance_length`,
  `draft_tokens`, `accepted_draft_tokens` and
  `draft_accept_rate_per_position`, the last per position rather than
  folded into the mean, because a drafter that is right at position 0
  and useless by position 7 has the same mean as a uniformly mediocre
  one and the two want opposite block sizes.

So the verification half is done, tested, and backend-agnostic: it goes
through `forward_batch`, which already runs on CPU, CUDA and Metal.

## Status

Checked against the code on 2026-10-28.

- **Step 1 landed.** `crates/frink-models/src/draft_model.rs` holds
  `DraftModelSpeculator`, a `Drafter` that owns its own `Decoder` and
  cache and truncates the cache to the synced history at the top of
  every `propose`.
- **Step 2 landed in part.** A draft model whose vocabulary size differs
  from the target's is refused by name (`VocabMismatch`). The hash of
  the token strings described below is not implemented; a `Decoder`
  alone sees only the size.
- **Step 3 landed.** `frink run` takes `-d` / `--model-draft`,
  `--draft-max` (alias `--draft`) and `--draft-p-min`. `--draft-min` is
  not implemented.
- **Step 4 is open for a draft model.** The server speculates with
  `PromptLookupSpeculator` since 0.28.0 and fills the `Usage` fields
  when it does; it cannot load a second GGUF.
  [`server-speculative-decoding.md`](server-speculative-decoding.md)
  says what that costs.
- **Step 5 is open.** `--mtp` still errors by design.

The text below is the plan as written, before steps 1 to 3 landed.

## What was missing

**A drafter worth having.** The only `Drafter` in the tree was
`PromptLookupSpeculator`, an n-gram match over the history with no model
at all. It is free and it helps on repetitive text, which is why it was
first. It cannot carry a coding workload.

**Any wiring at all.** `frink speculative` is a demo command on
synthetic random weights, so the hit rate it prints says nothing about a
real checkpoint. `frink run` had no draft flags, and the server had no
speculative path. `--mtp` errors by design.

## The steps, each of which ships on its own

Ranked by `ship-small-or-do-not-start`. Step 1 alone is a usable
speedup; nothing here is a prerequisite for a later step being useful.

### 1. `DraftModelSpeculator`: a second GGUF as the drafter

A `Drafter` that owns its own `Decoder` and its own `KvCache`, runs k
cheap `forward_token` steps, and returns the block with each position's
sampled distribution so the rejection rule has the `q(x)` it needs.

Two things this step must get right:

- **`Drafter::propose` has to take `&mut self`.** A draft model carries
  KV state across calls. Interior mutability would hide that in a
  `RefCell` and cost a runtime borrow check on the hot path for nothing.
- **The draft KV must roll back on rejection.** When the target rejects
  position i, the drafter has already advanced its cache past it. If
  that is not truncated the drafter's context silently diverges from the
  target's, and the symptom is a collapsing accept rate rather than an
  error. This is the same shape as everything else in this repo: two
  structures that must agree about one thing.

### 2. Refuse a mismatched vocabulary, loudly

The gotcha that costs everyone a day. Draft and target must agree on
token ids. If they do not, `q(x)` and `p(x)` are indexed by different
vocabularies, the rejection rule is comparing unrelated numbers, and the
result is not the target's distribution while looking exactly like it
is: fluent text, no error, a plausible accept rate.

So this is a refusal, checked at load, naming both checkpoints and what
differs: vocabulary size first, then a hash of the token strings, since
equal sizes do not imply equal vocabularies. Per this repo's rule a
refusal is coverage, and this one is the difference between lossless
being true and being a claim.

### 3. CLI flags, spelled as llama.cpp spells them

`-md` / `--model-draft`, `--draft-max` (alias `--draft`), `--draft-min`,
`--draft-p-min`. A client that already drives `llama-server` should not
have to learn new names. Report acceptance length and the per-position
accept rate on stderr beside the existing throughput line.

### 4. The server path, and the `Usage` fields it already declares

`FRINK_DRAFT_MODEL_PATH`, the draft decoder held beside the target, and
the four speculation fields populated instead of absent. `None` and
`Some(1.0)` are different answers there: "speculation did not run"
versus "speculation ran and never helped".

Continuous batching is explicitly NOT in this step. A shared drafter
across concurrent sequences is a different design and belongs after a
single-stream path is measured.

### 5. Make the guesser better and the check cheaper

Only after 1 to 4 are measured, because this is where effort goes once
the shape works, and because none of it is verifiable without a baseline:

- Adaptive block size from the observed per-position accept rate, rather
  than a fixed k. The per-position vector in `Usage` exists for this.
- Stop drafting when the drafter's own confidence falls below
  `--draft-p-min`, rather than always proposing k.
- Tree drafting and batched verification of several branches.
- A learned drafter behind the same trait: an MTP head (`--mtp` is
  reserved for exactly this and errors today), or EAGLE-style
  hidden-state conditioning, for which `Drafter::propose` already
  receives the target's final-layer hidden state.

## How this gets measured, and by whom

**Not by an agent.** A loaded host reads 25 to 45 percent low, and a bad
number published is worse than no number.

The claim to test is not "tok/s went up". It is two claims:

1. **The output distribution is unchanged.** At `--temp 0` the text must
   be token-for-token identical to a plain decode of the same prompt and
   seed. That is a test, not a benchmark, and it runs in CI.
2. **Tok/s went up on a real checkpoint**, measured with `frink bench`
   on a quiet host, reported as a pair with the acceptance length, since
   a speedup without an accept rate cannot be reproduced or debugged.

Acceptance is genre-conditional: a drafter that carries code will do
worse on prose. Report which prompt produced the number.

## Not in scope

AMD and HIP. frink's backends are CPU, CUDA and Metal, and Vulkan is
one `Q8_0` matvec. Nothing in this plan is backend-specific: the draft
model is another `Decoder` and verification is another `forward_batch`,
so all three backends get it from the same code.
