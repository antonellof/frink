# Several completions per request (`n` > 1)

Status: **DONE for both OpenAI routes, 2026-09-22.** `/v1/completions`
and `/v1/chat/completions` serve `n` from one prefill, verified on a
real model. llama.cpp's native `/completion` refuses it by name (one
`content`, nowhere to put a second answer), and so does the streaming
chat path -- see (5), which is the one decision this plan changed.

What is left, each its own row: copy-on-write for the paged store so a
paged request can fork; round-robin streaming; `best_of`, which needs
a scoring rule.

`n` is the OpenAI field for "give me `k` samples of this prompt". Until
2026-09-22 frink answered it with a 200 and one choice on two of its
three generation routes, and a 501 on the third; it is now a 501 on all
three (`crate::unimplemented_fields`). This says what implementing it
actually costs, because the obvious answer -- "run the loop `k` times"
-- throws away the only reason the feature exists.

## Why it is not "loop k times"

The whole value of `n` is that the **prompt is prefilled once**. A
caller who wanted `k` independent generations could already send `k`
requests; what they cannot do from outside is share the prefill. On a
6000-token prompt with a 32-token answer, `k = 4` done naively is four
prefills and 128 decode steps; done properly it is one prefill and 128
decode steps, which on the measured prefill/decode ratio of this engine
is most of the work.

So the feature is a KV-cache fork, and the fork is where the difficulty
is.

## What forks and what does not

| store | fork | note |
|---|---|---|
| `Kv::Contiguous(Vec<KvCache>)` | **clone** | `KvCache` is `Clone`; the prefix cache already forks one per request |
| `Kv::Paged(PagedLease)` | **not yet** | a lease owns block ids; two readers of one block list need copy-on-write at the first write, which the block store does not have |
| a recurrent layer's `RecurrentState` | **clones with the cache** | it is a reduction over the prefix, and a fork of the prefix is a fork of the state, so clone is correct where `truncate` is not |

The paged arm is the one that needs new code, and it is the arm the
prefix cache runs on. Copy-on-write at block granularity is the right
answer and is the same mechanism that would let the radix cache share
blocks between requests instead of cloning rows.

## (3) The decision

**Fork the contiguous store, refuse the paged one by name, per
request.**

Not a silent fallback to re-prefill: a caller who asked for `n = 4` and
got four prefills paid four times for the thing the field exists to
avoid, and nothing in the response would say so. A 501 naming the store
and the flag that selects it is honest, and it keeps the paged arm's
copy-on-write as a row of its own rather than smuggling a half version
of it in under this one.

Why this beats the alternatives:

* **Re-prefill per choice** is the naive loop. It is correct and it is
  the feature in name only. If it shipped, the acceptance measurement
  below would read the same as `k` separate requests, and there would
  be no signal left to tell anyone the real version had not landed.
* **Copy-on-write first** is the complete answer and is a bigger row
  than this one. Doing it first means no `n` at all until block
  sharing lands, and block sharing wants its own measurement (it is
  also a memory win for the radix cache, independently of `n`).

## The decisions that follow, each forced

1. **Seeds.** `k` samples of one prompt must differ, and a seeded
   request must still be reproducible. Choice `i` samples from
   `seed + i` (derived, not drawn), so `n: 4, seed: 7` is stable across
   runs and across `n` -- choice 0 of `n = 4` is byte-identical to the
   single answer of `n = 1`. A caller who sends no seed gets the
   existing behaviour per choice.
2. **Greedy.** At `temperature = 0` every choice is the same token
   sequence, because the sampler is deterministic and the prefix is
   shared. That is not a bug and is not worth a refusal: it is what the
   parameters say. The docs state it.
3. **Stopping.** Each choice carries its own `finish_reason`, its own
   stop-string matcher and its own grammar machine. They are `k`
   independent walks that happen to start from one cache.
4. **Usage.** `prompt_tokens` is counted **once** -- it was prefilled
   once, and reporting it `k` times would overstate the bill by exactly
   the saving the feature makes. `completion_tokens` is the sum over
   choices. This is the one place a reader can see that the prefill was
   shared, so it is also the acceptance test.
5. **Streaming.** SSE chunks carry `choices[].index`, and a client is
   entitled to interleaved indices. Round-robin -- one token per live
   choice per pass -- is the right answer and is NOT what shipped:
   `sample_until_stop` runs a choice to completion, so interleaving
   needs a sampler that can be stepped one token at a time per choice,
   which is a row of its own.

   **So `n` > 1 with `stream` is refused BY NAME.** Emitting choice 0
   to its end and then choice 1 would be sequential delivery wearing
   an `index` field, and a client reading those indices would be
   misled. Refusing is the same argument as
   `crate::unimplemented_fields`: a 501 a caller can read beats a 200
   they cannot check.
6. **The prefix cache write-back.** It stores one continuation per
   prompt and cannot represent `k`. Choice 0 is written back and the
   rest are not, because choice 0 is the one a subsequent `n = 1`
   request with the same seed would reproduce.
7. **Speculation.** The drafter is per choice, and its n-gram history
   is that choice's own. Sharing one drafter across choices would let
   choice 0's text steer choice 3's drafts, which is not wrong but is
   not measurable either; per choice is the honest default.
8. **`best_of`.** Deliberately NOT part of this row. It needs a
   scoring rule to pick "best", and the only defensible one is summed
   logprob, which means `prompt_logprobs`-shaped machinery frink does
   not have. It stays refused by name and says so.

## What has to hold

* **The admission budget must price `k`.** `FRINK_CB_MAX_CONTEXT`
  admits `prompt + max_tokens`; with `n` it is
  `prompt + k * max_tokens` of KV, because the forks are live at the
  same time. A request that fits at `n = 1` and not at `n = 4` must be
  refused with the same `context_length_exceeded` shape and the real
  arithmetic, not discovered as an allocation failure.
* **Continuous batching.** `k` forks of one request are `k` rows to the
  scheduler. Either they enter the batch as `k` rows (and the block
  budget already knows how to count that) or the pair is refused by
  name. Refuse first, measure second.
* **Cancellation** has to cancel all `k`.

## The prerequisite, found by trying it

`generate::generate` returns `(FinishReason, Usage)` and emits through
an `impl FnMut(&str)`. `n` makes both plural: the return grows a choice
dimension and `emit` grows a choice index. There is exactly ONE
production caller (`lib.rs:2235`), which is the good news; there are
about thirty call sites inside `generate.rs` itself, all tests, which
is the bad.

`generate.rs` is **4784 lines**. This repo's most expensive lesson says
a change that would grow a file past roughly a thousand lines splits
the file first, and the reason is written down: the same decode layer
was once spelled out eleven times across two big files and lost eight
model features one at a time, each silently. Threading a choice index
through thirty call sites of a 4.8k-line file is how that happens
again.

So the order is:

1. ~~**Split `generate.rs`.**~~ Done: `crate::request_tail`. The request tail alone -- usage assembly,
   reasoning counting, the radix publish, the prefix-cache write-back
   -- is a module, and it is exactly the part that must run for choice
   0 and not for the rest. Splitting it is what makes that rule
   expressible instead of an `if` in the middle of a long function.
2. ~~**Parameterise by `n`**~~ Done. `generate` returns
   `Vec<FinishReason>` and emits with a choice index; the contiguous
   store forks from the post-prefill state, the paged store and the
   recurrent engines are refused by name, and `prompt_tokens` is
   counted once while `completion_tokens` sums.

   One thing the tests found that the plan had not: the seed
   derivation cannot be pinned by comparing choice 0 to `n = 1`.
   Choice 0 runs on the request's own `params`, so no change to
   `seed + i` can move it -- a sabotage of the derivation leaves that
   comparison green. The derivation needs its own test, and it is the
   one that asserts the other choices DIFFER from choice 0 and that
   two runs of the same seeded request agree.
3. ~~**Wire `n`**~~ Done for the one OpenAI route whose response has a
   `choices` array to put the answers in.
   `unimplemented_fields::SERVES_SEVERAL_CHOICES` is that list, so the
   field is served where it has a home and refused BY NAME where it
   does not -- llama.cpp's native `/completion` returns one `content`,
   and neither the Anthropic nor the Responses wire has the array at
   all.

4. ~~**`/v1/chat/completions`**~~ Done. `CachedCompletion` holds every
   choice rather than one, which is what the `n` already in
   `GenerationKey` was promising -- keying `n` and then storing the
   first of three would have been a key stricter than the cache. Each
   choice is parsed for tool calls and reasoning in its own right,
   because a tool call in choice 2 is a tool call and reading only
   choice 0 would return the others as raw marker text. `cacheable()`
   now requires EVERY choice to be complete, not just the first.

Step 1 is worth doing whether or not `n` ever lands, which is the test
of whether a prerequisite is real or an excuse.

## How it would be measured

The acceptance number is not tok/s, it is **prefills per request**:
`n = 4` must show one prefill in the usage block, not four. A/B against
four separate requests with the same prompt, interleaved, reporting
both wall clock and `prompt_tokens`. If `prompt_tokens` for `n = 4`
equals four times the `n = 1` figure, the fork did not happen and the
row did not land, whatever the wall clock says.
