---
name: "out-of-core MoE: run a model larger than the machine"
overview: "From `north-star.md`'s `t1-out-of-core-execution`: run a MoE checkpoint bigger than RAM, since a token touches only top-k of N experts. Target: DeepSeek-V4-Flash-0731 (`deepseek4`, 304B total / ~13B active, 155 GB at UD-Q4_K_XL) against a 32 GiB M2 Pro, which is 4.8x oversubscription; the plan promises the mechanism, measured, with the ratio it supports stated in the output, not that file on that laptop."
todos:
  - id: measure-what-exists-before-designing-anything
    content: "Run OLMoE-1B-7B Q4_0 and Qwen1.5-MoE resident and under `FRINK_EXPERT_CACHE_BYTES` at several budgets, on CPU and Metal, recording tok/s, `ExpertStoreStats` and RSS. Done means a published table of what streaming costs today."
    status: pending
  - id: stored-experts-disable-metal
    content: "Decode half landed in `42468f5`: `ExpertBacking::Stored` passes `layer_supports_metal_moe_resident` and `try_metal_moe_topk` pins each top-k expert. Still resident-only: the packed `mul_mv_id` path, `try_metal_moe_experts_resident`, `metal_prefill_moe` and `try_metal_moe_prefill_batch`."
    status: pending
  - id: one-budget-not-two
    content: "One device-side slab holds streamed experts, read straight from the file, with no host byte tier beneath it, so unified memory holds one copy of each hot expert."
    status: completed
  - id: metal-slot-device
    content: "A Metal `SlotDevice` (one Shared `MTLBuffer` per bank) implementing the existing `expert_slots.rs` trait, as `CudaExpertPool` does. Remaining refinements: one size class with off-size layers rejected, `mlock` on the slab, and halving the allocation on failure."
    status: completed
  - id: byte-budget-to-expert-count
    content: "Wire `expert_budget::plan_cache_budget` and `net_cache_budget_bytes`, which already turn bytes into a slot count with two layers of prefill headroom and have no caller. Add an auto mode beside `FRINK_EXPERT_CACHE_BYTES`, where a bare number is an exact slot count and a `GB` suffix a byte budget, and record why the auto fraction stops where it does."
    status: pending
  - id: shipped-profile-and-runtime-learning
    content: "Seed hotness from a shipped profile for the cold start and learn at runtime after it (selections counted even on a miss, decayed every 16 tokens). Ship the profile as data keyed to the checkpoint hash, produced by frink from `routing_skew`; a mismatched profile is ignored with a line saying so."
    status: pending
  - id: metal-routing-readback-off-the-encode-thread
    content: "Metal's hotness signal is all zeros: `attn.rs` skips the expert-id download, and `scratch.ids` holds only the last layer's selection. Widen it to `top_k * n_layers` or replace it with an atomic histogram the kernel increments, read off the encode thread."
    status: pending
  - id: async-expert-reads
    content: "`ExpertStore::prefetch` reads one expert at a time on the calling thread, three preads each. Add a pread worker pool that starts after the router and joins before the expert GEMMs (discarding a stale batch, falling back to synchronous reads), plus `F_RDADVISE` readahead and `POSIX_MADV_DONTNEED` on eviction."
    status: pending
  - id: prefill-is-a-different-problem-with-an-easier-answer
    content: "Decode prefetch is intra-layer: start reads right after the router and overlap the rest of the layer, with no next-layer guessing. Prefill streams whole layers sequentially, double-buffered, through the existing unwired `ExpertCache::prefetch_prefill_layer`."
    status: pending
  - id: simulate-a-smaller-machine
    content: "Add a `--simulate-used-memory N` that mmaps, touches and `mlock`s N GiB in 256 MiB chunks before load, so the oversubscription ratio is testable on the dev machine. Extend `gguf_roundtrip.rs`'s bit-identical store test to Metal and to a 2x, 3x and 5x curve on a model that fits."
    status: pending
  - id: tell-the-user-the-cost-first
    content: "Before the run, extend `run.rs`'s budget report with the oversubscription ratio, the slot count and layers it covers, and an expected tok/s range from this host's own curve, behind the existing `--strict-budget`. The reference numbers: streaming costs ~3.5x on decode and 20-30x on prefill at 1.47x oversubscription."
    status: pending
  - id: deepseek4-is-not-the-prerequisite-and-must-not-become-one
    content: "Frink refuses `deepseek4` (dedicated architecture, synthetic-only decoder), so build and verify everything here on MoEs that load (OLMoE, Qwen1.5-MoE, `test_moe_fixture`). Track the DeepSeek-V4 loader separately."
    status: pending
  - id: no-repack-and-what-would-change-that
    content: "Stock GGUF is enough: one expert is one contiguous range per tensor, three seeks for gate, up and down. Interleave them per expert (24 reads to 8 at top-k 8) only if measurement after async reads shows seek count binding."
    status: pending
  - id: the-first-three
    content: "Order: measure what exists, unblock Metal for streamed layers, then build the shrink-the-machine harness and produce a 2x/3x/5x curve. The rest follows in the order under Order below."
    status: pending
isProject: false
---

# Out-of-core MoE: run a model larger than the machine

An MoE touches top-k of N experts per token, so the working set is a
fraction of the weights. This expands `north-star.md`'s
`t1-out-of-core-execution`. The primary reference is a C engine that
already ships this, kept under `.scratch/ds4`.

## Status, 2026-10-28 (issue #406)

- **The residency policy executes.** With `FRINK_EXPERT_CACHE_BYTES`,
  the expert store keeps entries in a `SlotDevice`
  (`crates/frink-core/src/expert_slab.rs`): an anonymous host mapping on
  CPU, one Shared `MTLBuffer` on Metal
  (`crates/frink-metal/src/expert_slab.rs`) that the CPU writes, the CPU
  kernels read in place and the weight cache aliases. One copy per hot
  expert, counted once. This closes `one-budget-not-two` and the device
  half of `metal-slot-device`.
- **Eviction ranks by real routing counts** from
  `MoeWeights::activation_counts`: the victim is the coldest unpinned
  entry, and a miss colder than every evictable entry is served without
  admission. A top-k is pinned in one `acquire_many`.
- **Evidence.** `crates/frink-models/tests/out_of_core_moe.rs` decodes a
  4-layer, 8-expert Q4_0 MoE at 1/2, 1/4 and one slot of the expert
  bytes: logits bit-identical to resident on the host, tokens identical
  on Metal, evictions and peak resident bytes asserted, and the slab
  asserted bound. Removing the pin check turns it red. On OLMoE-1B-7B
  Q4_0, 64 greedy tokens match at full, half and quarter budgets.
- **Cost today** (M2 Pro, one unpublishable run, file in the page
  cache): Metal decode 133.8 tok/s resident, 40.2 with the store at
  100%, 27.5 at 50%, 15.2 at 25%; CPU 40.0, 17.7, 6.9, 4.6. The 100%
  rows are the path's cost, not misses': a streamed layer takes the
  per-layer top-k launch with host routing, and on the CPU its views are
  never repack-cached.
- **CPU streaming is correct and pinned.** The OLMoE garbage of August
  came from the int-dot repack cache keying by buffer address while the
  store reused buffers (fixed 2026-08-28). `tests/streaming_int_dot.rs`
  now runs with int-dot on, and one `MoeWeights::with_experts` hands
  either backing to the same body. Slot views are `WeightBytes::Shared`,
  which `WeightBytes::map_id` refuses to key.
- **Metal guard.** `ExpertBacking::Stored` passes
  `layer_supports_metal_moe_resident` (`decoder/metal_eligibility.rs`)
  since `42468f5`; the packed `mul_mv_id` path, the resident phase-2
  helper and both Metal prefill MoE paths still decline it.
- **Not done:** async reads, a decay window on hotness, the packed and
  prefill paths for streamed layers, a shipped hotlist,
  `--simulate-used-memory`, a run on Qwen1.5-MoE (shared experts), and a
  run on a model that really exceeds RAM.

## The questions

| | Answer |
|---|---|
| **Layout** | Stock GGUF, read positionally at raw offsets (`loader/expert_source.rs`), as the reference engine does. A repack would turn 3 seeks per expert into 1; only worth it if seeks prove binding. |
| **Budget** | Bytes in, expert slot count out. `expert_budget::plan_cache_budget` (`pool.rs:170`) already computes it with two layers of prefill headroom. Nothing calls it. |
| **Policy** | Both: a shipped profile seeds the cold start, runtime counts take over, through one counter. |
| **Prefetch** | Within a layer: start the load after the router, overlap the rest of the layer. Prefill streams whole layers, double-buffered. |
| **Cost** | ~4.8 tok/s decode and ~3-5 tok/s prefill at 1.47x oversubscription on a 128 GB M5 Max (reference engine's README). |
| **Verification** | Small MoE, tiny budget (`gguf_roundtrip.rs:446`, `out_of_core_moe.rs`), plus a harness that locks RAM away so the ratio is testable on hardware we own. |
| **Order** | Measure, unblock Metal, shrink the machine. |

## Caches

The streamed path now has one slab. What remains is the split
vocabulary: `ExpertCache` + `expert_slots` (plans, hotness histogram,
prefill double buffer, many tests, no external caller) beside the wired
`ExpertStore`, and `ResidencyPlan` defined in both
`frink-core/src/residency.rs` and `frink-moe/src/lib.rs`. A host
`SlotDevice` (`HostSlotMemory`) and the SSD source
(`GgufExpertSource`) both exist.

## Hotness

On Metal, `activation_counts` is all zeros: `attn.rs` skips the
expert-id download ("sync tax"), and `scratch.ids` is one `top_k`
buffer rebound by every layer, so even a read after the wait would see
only the last layer. `placement_plan` has been feeding zeros into
`PlacementPlan::from_budget`.

A shipped profile owns the cold start, which on a 4x-oversubscribed
model is every first token. Runtime learning owns the rest, since a
profile is per checkpoint and per workload. `routing_histogram`
(`expert_cache.rs`) is the profiler and `ExpertStore::prefetch` the seed
applicator; ship the profile as data keyed to the checkpoint hash, and
ignore a mismatched one with a line saying so.

## mmap

Free: demand paging, and on Metal the whole file aliased as one
`MTLBuffer` (`newBufferWithBytesNoCopy`). Not free: control over
eviction order and readahead, and the page cache competes for the same
RAM. Fix with `POSIX_MADV_DONTNEED` on eviction and `F_RDADVISE` before
each pread.

## Cost, said before the run

Reference numbers, GLM 5.2 IQ2_XXS (188 GiB) on a 128 GB M5 Max, 1.47x
oversubscribed:

| | fully resident (2 Macs, TP) | one Mac, SSD streaming |
|---|---|---|
| decode | ~16.8 tok/s | **~4.8 tok/s** |
| prefill (4096 tok) | ~94 tok/s | **~3-5 tok/s** |

Prefill is the disaster, not decode. The target here is 4.8x
oversubscribed; the reference engine offers no 32 GB recipe for any
DeepSeek V4 model, and its smallest Flash quant (~81 GB, ~2.5x here) is
the reachable target. `run.rs` already prints a budget report and
suggests `FRINK_EXPERT_CACHE_BYTES` when nothing fits; add the ratio,
slot count and an expected tok/s range from this host's own curve.

## deepseek4 is a consumer, not a gate

`capability.rs` registers `deepseek4` as dedicated ("needs CSA/HCA +
mHC assembly"), and `deepseek_v4_decoder.rs` is synthetic weights only.
Build and verify everything on MoEs that load (OLMoE Q4_0,
Qwen1.5-MoE, `test_moe_fixture`) and track the loader separately.

## Order

```
1  measure what exists          no code; the plan rests on an unmeasured claim
2  unblock Metal                packed and prefill paths for streamed layers
3  shrink the machine           --simulate-used-memory; makes the ratio testable

then  metal routing readback    cheap, wrong independent of this plan
      async expert reads        a pread pool instead of one thread
      byte budget to slot count
      prefill double buffer     already built in frink-core, unwired
      shipped profile + runtime
```

`no-repack` stays closed unless step 1 reopens it.
