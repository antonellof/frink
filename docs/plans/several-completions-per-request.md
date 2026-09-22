# Several completions per request (`n` > 1)

Status: **design decided 2026-09-22, see (3); implementation next.**

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
   entitled to interleaved indices. **Decoded round-robin**, one token
   per live choice per pass, so a slow choice cannot starve the others
   and the first token of every choice arrives at nearly the same time.
   Sequential choice-at-a-time would make choice 3's first token arrive
   after choices 0-2 finished, which no client expects.
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

## How it would be measured

The acceptance number is not tok/s, it is **prefills per request**:
`n = 4` must show one prefill in the usage block, not four. A/B against
four separate requests with the same prompt, interleaved, reporting
both wall clock and `prompt_tokens`. If `prompt_tokens` for `n = 4`
equals four times the `n = 1` figure, the fork did not happen and the
row did not land, whatever the wall clock says.
