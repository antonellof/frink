---
name: dFlash / dFlash2, block-diffusion speculative decoding
overview: "GOAL: give frink a real speculative drafter. Today `frink-models::speculative` is prompt-lookup only (an n-gram match over the history, no model), and `--mtp` errors. dFlash is a block-diffusion drafter: it predicts a whole K-token block in ONE forward pass instead of autoregressively, conditioned on the target model's own hidden states, sharing the target's embedding and LM head, and the target verifies the block in one batched pass. Published numbers: mean acceptance length 4.80 tok and 2.7-3.4x throughput on the drafter's own benchmark table, 3.3x vs EAGLE-3's 2.1x on GSM8K at equal acceptance length. THE HONEST CONSTRAINT, stated up front: dFlash is a TRAINED drafter and frink does not train. Everything here is worthless without a published drafter checkpoint in a format frink can load, so `dflash-checkpoint-reality` is the first todo and it is a GO/NO-GO, if no loadable checkpoint exists, the useful subset of this plan is the engine-side work (lossless verification, the drafter trait, cache resume, acceptance metrics), which is worth doing on its own and is written to stand alone."
todos:
  - id: dflash-checkpoint-reality
    content: "GO/NO-GO, do this before writing any kernel. Establish whether a dFlash drafter can be loaded by frink AT ALL. Answer three questions with evidence, not inference: (1) does a published checkpoint exist and under what licence (search names the HF repo `incoai/Qwen3.8-27B-DFlash2`, and claims 3.5M downloads and integrations across the major serving stacks including llama.cpp); (2) since llama.cpp is claimed as an integration, is there a GGUF convert path and what tensor names + metadata keys does it emit, read llama.cpp's converter and its dFlash graph, that is this repo's default method and it settles the format question by reading rather than guessing; (3) does the checkpoint ship the drafter ALONE (sharing the target's embedding + LM head) or a standalone model. If the answer to (1) is no, STOP and record that; the engine-side todos below still stand"
    status: pending
  - id: spec-lossless-verify
    content: "LANDED (branch wf/spec-engine-lossless). `accept_or_resample` in frink-models/src/speculative.rs implements the rejection rule: accept with min(1, p_target(x)/p_draft(x)), else resample from the normalised residual max(0, p_target - p_draft). p_target comes from `sampling::sampling_distribution`, a new function publishing the EXACT distribution `Sampler::sample` draws from (penalties, temperature, top-k, top-p) -- verifying against a raw softmax while the caller sampled with top_p=0.9 would be lossless with respect to a model nobody is running, so `sample_with_mask` and `sampling_distribution` now share one filter/normalise tail. Proof, not assertion: `resampling_reproduces_the_target_distribution` pushes 200k tokens through the rule with four draft distributions (including one whose support misses the target's mode) and asserts total variation from the target < 0.01; `speculative_decode_at_temperature_matches_plain_sampling` runs a real decode at temperature 1.0 against the target's EXACTLY enumerated per-position marginals (vocab 6, horizon 3, every prefix walked) rather than against a second noisy Monte Carlo. Mutation-checked: restoring the argmax accept rule leaves all 21 other tests in the module green and fails exactly those two -- which is also the cleanest statement of what the old rule was, the temperature-0 special case. Dropping the residual subtraction fails the first. TWO BUGS FOUND ON THE WAY, both fixed with their own tests: (1) `Sampler::next_u64` implemented plain xorshift64, not the xorshift64* its doc claimed; its state IS its output, so for every seed below ~4000 the first draw landed in the bottom eighth of [0,1) -- a request asking for seed:42 always got its first token off the bottom of the CDF, and batch_scheduler.rs seeds one Sampler per request. Restored the output multiply. (2) the drafter was asked to continue a history that stopped one token short of the anchor, shifting every proposal by one position."
    status: completed
  - id: spec-drafter-trait
    content: "LANDED (branch wf/spec-engine-lossless). `Drafter::propose(history, target_hidden: &[f32], max_len) -> DraftBlock` in frink-models/src/speculative.rs, DraftBlock carrying tokens plus one `DraftDist` per position. DraftDist is SPARSE ((token, prob) support, not a dense vocab vector) because the distributions drafters really produce are: prompt lookup's support is one token, a real drafter's is a top-k, not 150k floats per drafted position. Its documented contract is that the support must be the distribution the token was ACTUALLY sampled from -- the rejection rule corrects any q, however bad, but only if q is honest, so a drafter that truncates its own softmax must report the truncated renormalised one. `Decoder::forward_batch_with_hidden` now returns the last-layer hidden states `forward_batch` was computing and discarding; both go through one `logits_from_flat_hidden` so a second copy of the softcap cannot drift, and `forward_batch` keeps its move-only path so prefill pays nothing extra. PromptLookupSpeculator stays as the impl (its `propose` became `propose_tokens`, wrapped by the trait method with one-hot dists). Tests: the history the drafter sees is asserted to include the anchor token, and the hidden state it receives is asserted to be a full hidden_dim vector every round; both mutation-checked. NOT DONE, deliberately: no model-based drafter exists to be a second impl, so the hidden-state argument has one consumer (a test) rather than a real one. The shape is argued from what dFlash/EAGLE need, not validated against a working drafter."
    status: completed
  - id: spec-cache-resume
    content: "LANDED (branch wf/spec-engine-lossless). `SpeculativeOptions::start_pos` states where the prompt begins in an already-warm cache; every position and every rollback length in the loop is now absolute, so a non-zero base stops being a special case rather than being special-cased. A start_pos disagreeing with the caches' seq_len is REFUSED (assert) instead of silently truncating the caller's context -- fail-closed applied to an off-by-start_pos that would otherwise corrupt a prefix-cache hit rather than error. Uniform exit contract: on return the caches hold context + prompt + every generated token EXCEPT the last, whose KV is not computed until it is fed, and the next call passes it as its anchor. That needed one real design fix -- the draft is capped one short of the remaining budget so the run's last token is always committed as an anchor; without it the cache ended in two different states depending on whether the budget ran out mid-block or on an anchor. Three tests: a warm-cache run matches a cold one token for token, two back-to-back calls equal one long call, and the exact absolute cache length is asserted after forced rejections. Mutation-checked: making the rollback length relative to the call fails all three."
    status: completed
  - id: spec-acceptance-metrics
    content: "HALF LANDED (branch wf/spec-engine-lossless): the metrics and the wire contract are in, the PRODUCER is not. Landed: SpeculativeDecodeResult gains verification_steps, drafted_tokens, accepted_tokens, evaluated_at_position / accepted_at_position, plus acceptance_length() (completion tokens / verification steps, prefill excluded -- deliberately not tokens_per_call, which charges the one-off prefill against the average) and accept_rate_per_position(). Per-position rates are CONDITIONAL on reaching the position: positions after a rejection are never evaluated and are not counted, because counting them makes the accept rate decay with the block size instead of with the drafter. `frink speculative` prints the whole curve. On the wire, `Usage` gains acceptance_length / draft_tokens / accepted_draft_tokens / draft_accept_rate_per_position (all Option and omitted when absent, since None and 1.0 are different claims), and stats::entry carries acceptance_length plus the per-position curve into the /admin/stats ring. NOT LANDED, and it is the half serving needs: nothing populates any of it, because frink-server has no speculative decode path at all -- that is spec-server-integration, still pending. Today every response omits these fields; docs/API.md says so in as many words. The mutation check that mattered was on the metric itself: the first per-position test only asserted the counts were non-increasing, which crediting every position of every rejected block ALSO satisfies (it makes them equal). It now asserts position i+1 is evaluated no more often than position i was accepted, and that the scenario rejected something so the check is not vacuous."
    status: completed
  - id: dflash-drafter-forward
    content: "The drafter itself: a small transformer (published as ~5 layers, hidden dim below the target's, embedding and LM head SHARED with the target so the parameter count is 1-5% of the target). Load it as a second GGUF whose embedding/output tensors are absent and resolved against the already-loaded target, that sharing is the whole reason the drafter is cheap, so a loader that duplicates them defeats it. Runs on the CPU path first; Metal is dflash-metal below"
    status: pending
  - id: dflash-block-diffusion
    content: "The drafting procedure. Start the K-token block as mask tokens, run a small number of denoising steps (sources say typically 4-8, configurable) that refine all positions JOINTLY, and emit the block plus per-position distributions. This is the part with no precedent in frink, there is no diffusion or masked-denoising machinery anywhere in the tree. Block size K and step count are the two knobs (the serving stacks expose it as a block-size flag, e.g. 8 or 16). NOTE A CONTRADICTION IN THE SOURCES, do not resolve it by picking the convenient one: the arXiv summary describes TREE verification of multiple continuations, while the integration write-up states explicitly that dFlash is LINEAR block diffusion and not tree-based. frink's verify loop is linear today. Read the reference implementation before building either"
    status: pending
  - id: dflash-kv-injection
    content: "The trick that makes the drafter cheap on long context, and the one most likely to be got wrong. Rather than have the drafter re-encode the prompt, extract the target model's hidden representations for the context tokens and INJECT them into the drafter's KV cache, so the drafter skips modelling the full context from scratch. The reference integration materialises the injected KV immediately (layer-batched linear projection, fused norm+RoPE post-processing) rather than lazily, specifically so the drafter's KV does not balloon and so prefix sharing still works. frink's prefix cache now has block identity and payload-derived signatures (frink-core/src/kv_block.rs, kv_signature.rs), injected drafter KV is DERIVED state and must not be confused with target KV under the same block hash. Salt it, or the persistent cache will hand a drafter block to a target model. `kv-cache-signature` in docs/plans/serving-and-tiered-kv.md exists for exactly this class of mistake"
    status: pending
  - id: dflash2-path-selector
    content: "dFlash2 innovation 1, and the cheapest real win in the whole plan. Independent per-position argmax gives a block where each token is plausible alone and the sequence is incoherent. Keep the top 16 candidates per position and score adjacent pairs with a bilinear form: S_t(a,b) = U_t(b) + <A(a) (*) H(h_t), B(b)>, where U_t(b) is the drafter's own logit for candidate b, A and B are 256-dim predecessor/candidate embeddings, H(h_t) is a context gate over which components matter, and (*) is elementwise product. Then Viterbi one coherent path through the block. Published cost/benefit: +2.0M parameters and +0.6% latency for +0.34-0.47 tokens of acceptance length, against DSpark's +77.8M and +9.6% for less. Pure inference-side arithmetic given the weights"
    status: pending
  - id: dflash2-two-tap-conv
    content: "dFlash2 innovation 2, the fix for suffix decay (99.5% accuracy at the first drafted position falling to 87.8% at the last). Two-tap dynamic depthwise convolutions before and after each attention and feed-forward sublayer: Conv_k(x)_t = k_t,0 (*) x_t + k_t,1 (*) x_{t-1}, where each coefficient is a learned base kernel plus a content-dependent correction and every 16 channels share one correction vector. Position 0 reads the last VERIFIED token's representation; later positions read their immediate predecessor; all positions still compute in parallel, which is the property the whole method rests on. Published cost: +16.5M params (3%) and +0.7% latency, letting a 5-layer drafter approach a 15-layer one. Diagnosis behind it, worth keeping because it is testable: within-block attention mass collapses from 30% at layer 1 to 8% at layer 5"
    status: pending
  - id: dflash-metal
    content: "Metal path for the drafter block forward + denoising steps. Do NOT start until the CPU drafter agrees with a reference: this repo's rule is that a GPU path which cannot match CPU is refused rather than admitted (see coverage-phi-metal-rope, where Phi-4-mini is CPU-only on exactly those grounds). The drafter is small and runs K positions per step, so it is a batched-GEMM shape the existing `mul_mm_sg` covers; the denoising loop is the new dispatch pattern. Guard with `frink verify --backend metal` before any timing"
    status: pending
  - id: spec-server-integration
    content: "Wire speculation into frink-server: `batch_scheduler.rs` decode loop, per-request enable/disable, and interaction with the chunked prefill state machine that just landed. Two hazards specific to serving, neither present in the CLI demo: a rejected block must not leave the request's KV cache longer than its accepted prefix under CONCURRENCY (the rollback is per-request, the pool is shared), and a drafter is a second set of weights competing for the same device budget, `mem-preload-kv-budget` in the serving plan must account for it or admission will overcommit"
    status: pending
  - id: spec-bench-harness
    content: "`frink bench` measures pp512/tg128 with no speculation, so none of this is visible in benchmarks/RESULTS.md. Add acceptance length and speculative tok/s as first-class bench outputs, on the SAME quiet-host contract as every other row (the 2.0 load bar is now enforced in frink-cli/src/host_state.rs). Speculation speedup is workload-dependent by construction, code and math accept long, open chat accepts short, so a single number is a lie; report per-workload, mirroring how the sources report GSM8K / HumanEval / MT-Bench separately"
    status: pending
isProject: false
---

# dFlash / dFlash2, block-diffusion speculative decoding

> Plan for implementing [dFlash 2](https://inco.ai/blog/dflash2/) in frink.
> Sources read for this plan: the dFlash2 post above, the
> [arXiv paper](https://arxiv.org/pdf/2602.06036) (*DFlash: Block Diffusion
> for Flash Speculative Decoding*), and the
> [LMSYS integration write-up](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/).
> Where they disagree, the disagreement is recorded rather than resolved.

## What dFlash actually is

It is **not** a flash-attention variant. The name is misleading if you come
to it from FlashAttention: dFlash is a **speculative decoding drafter**, and
the "d" is *diffusion*.

Ordinary speculative decoding runs a small draft model autoregressively for
K steps, then has the target verify all K at once. The draft phase is
therefore K sequential forward passes of a small model, cheap per step, but
still serial, and that serial chain is what caps the speedup.

dFlash replaces it with **block diffusion**: the drafter emits the whole
K-token block in **one** forward pass, starting from mask tokens and running
a handful of joint denoising steps over all positions at once. It is
conditioned on the target model's own hidden states and shares the target's
embedding and LM head, so it is 1–5% of the target's parameters.

The verification half is unchanged and is the part frink already has.

## What frink already has, and what it does not

Already here, this is why the plan is tractable at all:

| Piece | Where |
|---|---|
| Batched verify of a draft block in one call | `Decoder::forward_batch` |
| Target hidden states, handed to the drafter | `Decoder::forward_batch_with_hidden` |
| Lossless accept/reject + KV rollback | `accept_or_resample`, `KvCache::truncate` |
| A drafter interface to plug a model into | `Drafter` / `DraftBlock` / `DraftDist` |
| Resumable decode over a warm cache | `SpeculativeOptions::start_pos` |
| Acceptance length + per-position accept rate | `SpeculativeDecodeResult` |
| A CLI entry point | `frink speculative` |

The first six of those rows describe work that landed on
`wf/spec-engine-lossless` (the four `spec-*` todos); before it, verification
was argmax-only, the drafter was welded to n-gram lookup, and the cache had
to be fresh.

Not here:

- **Any trained drafter.** `frink speculative` is prompt-lookup: an n-gram
  match over the history with no model at all. `--mtp` errors by design and
  says so in `docs/CLI.md`. The `Drafter` trait now exists for one to plug
  into, but nothing implements it except prompt lookup.
- **Any diffusion / masked-denoising machinery.** Nothing in the tree
  resembles it.
- **Serving integration.** `speculative_decode_with` accepts a warm cache
  now, but `frink-server` still never calls it, so the acceptance metrics
  on the wire are always absent. That is `spec-server-integration`.

## The constraint that decides whether this plan is real

**frink does not train models.** dFlash's whole value is in learned weights:
the 5-layer drafter, the 2.0M-parameter path selector, the 16.5M-parameter
two-tap convolutions. Implementing the forward pass without a checkpoint
produces a drafter that proposes noise, and a drafter that proposes noise is
strictly slower than no drafter, every rejected block costs a target forward
pass that would otherwise have produced a token.

So `dflash-checkpoint-reality` is first and is a **go/no-go**. Search
indicates a Hugging Face repo (`incoai/Qwen3.8-27B-DFlash2`) and claims
integrations across the major serving stacks including llama.cpp. The
llama.cpp claim is the load-bearing one for frink: if llama.cpp converts and
runs these drafters, then a GGUF representation exists, and reading its
converter and graph settles the tensor names, the metadata keys and the
embedding-sharing question by reading rather than by guessing. That is this
repo's default method and it applies here.

**If no loadable checkpoint exists, stop at that todo.** The four engine-side
items, `spec-lossless-verify`, `spec-drafter-trait`, `spec-cache-resume`,
`spec-acceptance-metrics`, are worth landing regardless. They are what makes
*any* drafter usable in serving, they close a real correctness hole at
temp > 0, and they are the prerequisites for MTP heads, EAGLE, or dFlash
whenever a checkpoint does appear.

## Published numbers, and how much to trust them

All of these are the sources' own, measured on the sources' own hardware and
harness. None has been reproduced here. They are recorded so the plan has a
target to fail against, not as claims frink can make.

dFlash2 post, mean acceptance length and throughput vs autoregressive:

| Model (as named by the source) | Acceptance length | Throughput |
|---|---|---|
| Qwen3.8-27B | 4.80 (vs MTP 4.28, DSpark 3.62) | 2.7–3.4× |
| Muse Glimmer | 5.70 | 3.1–4.6× |

LMSYS, dFlash vs EAGLE-3 (acceptance length / speedup), note the middle
column: on GSM8K the acceptance lengths are **identical** at 4.2 and the
speedup still goes 2.1× → 3.3×, because the win is in the *draft* phase
being one pass instead of K:

| Task | EAGLE-3 | dFlash |
|---|---|---|
| GSM8K | 4.2 / 2.1× | 4.2 / 3.3× |
| HumanEval | 4.3 / 2.2× | 4.0 / 3.2× |
| MT-Bench | 3.1 / 1.4× | 3.0 / 2.2× |

That table is the single most useful thing in the sources for frink, because
it says the mechanism being ported is *drafting latency*, not draft *quality*.
Drafting latency is something a CPU/Metal engine can actually move.

## dFlash2's two additions

Both are inference-side arithmetic given weights, and both come with a stated
diagnosis that frink can independently test once `spec-acceptance-metrics`
reports per-position accept rates.

**1. Path selector.** Predicting every position independently gives a block
where each token is individually plausible and the sequence does not cohere.
Keep the top 16 candidates per position, score adjacent pairs

```
S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>
```

(`U_t(b)` the drafter's own logit, `A`/`B` 256-dim predecessor/candidate
embeddings, `H(h_t)` a context gate, `⊙` elementwise), then take the best
path. Cost +2.0M params and +0.6% latency for +0.34–0.47 acceptance tokens,
against DSpark's +77.8M and +9.6%.

**2. Two-tap dynamic convolution.** Fixes *suffix decay*, per-position
accuracy falling from 99.5% at the first drafted position to 87.8% at the
last, because within-block attention mass collapses from 30% at layer 1 to 8%
at layer 5.

```
Conv_k(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}
```

inserted before and after each attention and FFN sublayer. Each coefficient is
a learned base kernel plus a content-dependent correction, with one correction
vector per 16 channels. Position 0 reads the last *verified* token; later
positions read their immediate predecessor; everything still computes in
parallel. +16.5M params (3%), +0.7% latency, and a 5-layer drafter reaches
roughly 15-layer quality.

## Order of work

1. `dflash-checkpoint-reality`, go/no-go, and it gates everything below it.
2. `spec-lossless-verify`, correctness, independent of dFlash, do it anyway.
3. `spec-drafter-trait` + `spec-cache-resume` + `spec-acceptance-metrics`,    the engine-side foundation, all three independent of dFlash.
4. `dflash-drafter-forward` → `dflash-block-diffusion` → `dflash-kv-injection`.
5. `dflash2-path-selector` → `dflash2-two-tap-conv`.
6. `spec-server-integration`, then `dflash-metal`, then `spec-bench-harness`.

## Measurement contract

Same as `docs/plans/llama-cpp-parity-push.md`: quiet host (1-minute load
average below 2.0, now enforced by `frink-cli/src/host_state.rs`), both
engines measured in the same session, interleaved A/B, receipts checked in.

One addition specific to speculation: **report per workload.** Acceptance
length is a property of the *text*, not just the model, code and math accept
long blocks, open-ended chat accepts short ones. A single averaged speedup
number for speculative decoding is not a measurement, and the sources
themselves report GSM8K, HumanEval and MT-Bench separately for that reason.
