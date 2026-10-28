---
name: "out-of-core MoE: run a model larger than the machine"
overview: "GOAL, from `north-star.md`'s `t1-out-of-core-execution`: make a MoE checkpoint bigger than RAM run, because an MoE touches only top-k of N experts per token and the working set is therefore a fraction of the weights. Concrete target: DeepSeek-V4-Flash-0731, `general.architecture` = `deepseek4`, 304B total / ~13B active, 256 routed + 1 shared, 155 GB at UD-Q4_K_XL, against a 32 GiB M2 Pro. THE INVESTIGATION CHANGED THE PLAN THREE TIMES, and each correction is worth more than the original guess. (1) FRINK ALREADY STREAMS EXPERTS. `crates/frink-core/src/expert_store.rs` is a 608-line bounded, lease-protected byte cache, wired into both decode paths behind `FRINK_EXPERT_CACHE_BYTES` / `FRINK_SSD_STREAMING`, reading experts with positional `read_exact_at` preads (`GgufExpertSource` in `crates/frink-models/src/loader/expert_source.rs`), and proven bit-identical to the resident path at both a generous budget and a 1-byte budget (`crates/frink-models/tests/gguf_roundtrip.rs:446`). The starting position is not zero; it is 'built, tested on a fixture, never run against anything large'. (2) TURNING IT ON DISABLES METAL. `ExpertBacking::Stored` failed the guard in `layer_supports_metal_moe_resident` and was explicitly rejected in the fused top-k path with the comment `// Streaming experts: fall back (can't hold all refs easily).` So on the exact backend this user runs, the feature that lets you exceed RAM switches off the kernels that make frink competitive. That is the blocker, and it is one enum arm, not a research problem. (3) THE ON-DISK LAYOUT QUESTION HAS A CHEAP ANSWER. `ds4_layer_pack.c` is NOT a disk format; it is a monotonic-contiguous multi-GPU layer PLACEMENT packer (`ds4_layer_pack.h:23`, callers at `ds4.c:56649`, `ds4.c:56920`) and has nothing to do with streaming. ds4 streams from the stock GGUF: `ds4_gpu_stream_expert_table` carries `model_map`, `gate_offset`, `up_offset`, `down_offset` (`ds4_gpu.h:196-205`) and the miss path is a plain `pread(g_model_fd, ...)` (`ds4_metal.m:11968`). No repack, ever. Frink already does the same thing. WHAT IS ACTUALLY MISSING, in order of how much it costs: a Metal `SlotDevice` so streamed experts stay on the GPU path; ONE budget instead of the two independent caches frink currently has; asynchronous reads (ds4 runs a 9-thread pread pool, `ds4_metal.m:11914`, frink's `ExpertStore::prefetch` at `expert_store.rs:173` is a synchronous `for` loop over `acquire`); a hotness signal, which on Metal is currently all zeros; and honesty about the result, which measured on ds4's own hardware is 4.8 tok/s decode at 1.47x oversubscription. HONESTY UP FRONT: 155 GB against 32 GiB is 4.8x oversubscription. ds4, which ships this capability, does not offer a 32 GB recipe for any DeepSeek V4 model; its smallest Flash quant is ~81 GB and it recommends that for 96 to 128 GB machines (`.scratch/ds4/download_model.sh:52`). This plan does not promise the 155 GB file on this laptop. It promises the mechanism, measured, with the ratio it actually supports stated in the output."
todos:
  - id: measure-what-exists-before-designing-anything
    content: "FIRST, AND IT MAY INVALIDATE ITEMS BELOW. Expert streaming is already wired and already tested for correctness; nobody has ever measured it. Run a MoE that fits in RAM (OLMoE-1B-7B Q4_0 and Qwen1.5-MoE are both in the bench suite, `benchmarks/RESULTS.md:60`) twice: resident, then with `FRINK_EXPERT_CACHE_BYTES` set below the routed-expert total, on CPU and on Metal. Record tok/s, `ExpertStoreStats` (`expert_store.rs:88`: hits, misses, evictions, pass_throughs, bytes_read, resident_bytes), and RSS. THREE THINGS THIS SETTLES that no amount of design can. (a) The Metal number will be catastrophic and the reason is known in advance (`stored-experts-disable-metal`), so this is the receipt that justifies fixing it rather than an assertion. (b) The CPU number tells us the real cost of the pread path at a known miss rate, which is the only input to any tok/s promise. (c) `pass_throughs` tells us whether the degrade-to-uncached path (`expert_store.rs:36`) is being hit in practice, which changes whether the budget arithmetic in `byte-budget-to-expert-count` needs a floor. DO THIS FIRST because two of the items below are written against a predicted failure and one measurement can retire them."
    status: pending
  - id: stored-experts-disable-metal
    content: "STATUS 2026-10-28: the decode half landed in `42468f5`. `layer_supports_metal_moe_resident` (`decoder/metal_eligibility.rs`) accepts `ExpertBacking::Stored`, and `try_metal_moe_topk` (`decoder/metal_moe.rs`) materialises each top-k expert into an owned `ExpertWeights` whose `WeightBytes::Shared` carries its own pin, which is the lease fix described below. Still resident-only: the packed `mul_mv_id` path, `try_metal_moe_experts_resident`, and the Metal prefill MoE paths (`metal_prefill_moe`, `try_metal_moe_prefill_batch`). The text below is the item as written. THE BLOCKER, and the highest value-per-line item in this plan. `Decoder::layer_supports_metal_moe_resident` required `matches!(layer.moe.experts, ExpertBacking::Resident(_))`, the fused top-k path (`try_metal_moe_topk`) returned `None` on `ExpertBacking::Stored { .. }`, and the Metal prefill FFN path takes the same `let ExpertBacking::Resident(experts) = ... else { return None }` shape (`metal_prefill_moe`, `try_metal_moe_prefill_batch`). So `FRINK_SSD_STREAMING=1` on Metal silently drops every routed layer onto the fallback path. The comment says why, and it is honest: `can't hold all refs easily`. A `Stored` expert's bytes live behind an `ExpertLease` whose lifetime is scoped to one `with_expert` call (`MoeWeights::with_expert`, `decoder/weights.rs`), and the fused kernel wants N experts' `WeightMatrix` launches alive at once. THE FIX IS THE LEASE MODEL, NOT THE KERNEL: `ExpertLease::shared_buf` already returns an `Arc<Vec<u8>>` that extends the pin (`expert_store.rs:80`), so holding top-k leases across one `launch_moe_topk_swiglu` call is exactly what the store was designed to permit. This is the concrete content of the north star's 'a `SlotDevice` for Metal' once you look at what actually stands in the way. UNTIL THIS LANDS, no out-of-core claim on Metal means anything, and `measure-what-exists-before-designing-anything` will produce a number that measures the fallback path rather than the feature."
    status: pending
  - id: one-budget-not-two
    content: "THE STRUCTURAL DECISION, and it has to be made before a Metal slab is written or it gets made by accident. Frink has TWO independent expert caching systems that do not know each other exists. (A) `frink-core::expert_store::ExpertStore`: wired, working, bit-identical-tested, pread source, pure LRU by a `last_used` stamp (`expert_store.rs:104-110`), produces `Arc<Vec<u8>>` HOST buffers. (B) `frink_core::expert_cache::ExpertCache` + `expert_slots::ExpertSlots`: 113 tests across six files, a slot/plan model (`CopyPlan` at `expert_cache.rs:83`, `EnsurePlan` at `:100`, `PrefillPlan` at `:332`), a routing histogram (`routing_histogram` at `:809`), a prefill double buffer (`prefetch_prefill_layer` at `:1001`), and a `SlotDevice` trait (`expert_slots.rs:105`) whose only production implementor is `frink-core`'s compile-verified `CudaExpertPool` (`crates/frink-core/src/expert_pool.rs:141`). A HOST implementor already exists too: `HostSlotMemory` (`expert_slots.rs:671`), so the north star's 'a `SlotDevice` for host RAM' is DONE and unwired, not missing. There is even a third residency vocabulary in `frink-moe::PlacementPlan` (`crates/frink-moe/src/lib.rs:705`). The split is visible in the type names: `ResidencyPlan` exists in BOTH `frink-core/src/residency.rs:191` and `frink-moe/src/lib.rs`, and `RebuildRejected` in BOTH `expert_cache.rs:1150` and `pool.rs:231`. WHY THIS IS URGENT ON THIS TARGET SPECIFICALLY: on Apple unified memory a host byte cache and a device buffer cache are the SAME physical RAM. `Stored` produces `WeightBytes::Shared` (`StoredExpertLayout::materialize`, `loader/expert_source.rs`), which is not a registered mmap, so it cannot take the `BytesNoCopy` alias path (`crates/frink-metal/src/gpu.rs:5097`) and instead gets COPIED into an `MTLBuffer` held by the weight cache bounded by `FRINK_METAL_WEIGHT_CACHE_BYTES` (inference, from `find_registered_mmap` matching only registered mmaps at `gpu.rs:5145`; confirm by instrumenting before relying on it). Two copies of every hot expert on a machine whose whole problem is that it has 32 GiB. RECOMMENDED SHAPE, matching what ds4 actually does: ONE device-side slab, pread directly into it, no host byte tier underneath. ds4's slab is `newBufferWithLength:options:MTLResourceStorageModeShared` (`ds4_metal.m:12479`), NOT `BytesNoCopy`, and experts are `pread` straight into `[buf contents]` (`ds4_metal.m:14846-14869`) followed by `didModifyRange:` (`ds4_metal.m:14876`). That is one copy total, file to unified memory, and it is what a Metal `SlotDevice` should be."
    status: pending
  - id: metal-slot-device
    content: "The device half, written against `expert_slots.rs:105`'s existing trait rather than a new one. `SlotDevice` is object-safe and taken as `&mut dyn` throughout (`expert_slots.rs:439`, `:454`, `:465`, `:536`, `:559`), with exactly two required methods, `write_slot(bank, dst_slot, src)` and `copy_slot(bank, dst_slot, src_slot)`, plus defaulted `begin_plan(CopyRoute)` and `flush()`. A `MetalExpertPool` mirrors `CudaExpertPool` (`crates/frink-core/src/expert_pool.rs:103`): one `MTLBuffer` per bank sized `slots * row_bytes[bank]` from `SlotGeometry` (`expert_slots.rs:67`), `write_slot` copying into `contents()` at the slot offset, `copy_slot` as a device-to-device blit, `flush` closing the blit encoder. FOUR THINGS TO COPY FROM ds4 BECAUSE THEY ARE NOT OBVIOUS. (1) SINGLE SIZE CLASS. ds4 freezes the slot size at `gate_expert_bytes * 2 + down_expert_bytes` page-rounded (`ds4_metal.m:12675`) and REJECTS off-size layers rather than adopting them (`ds4_metal.m:11601`), because a mixed-precision GGUF that boosts one layer to a bigger quant would otherwise make every ordinary layer bypass the cache; ds4 picks the MOST COMMON size class rather than the first layer's, for exactly that reason (`ds4.c:4536-4544`). Frink's `SlotGeometry.row_bytes` is already a per-bank `Vec<usize>`, so this is a load-time validation, not a redesign. (2) `mlock` THE SLOTS. `ds4_metal.m:12563` locks slab pages so macOS's memory compressor cannot swap out the cache you just paid an SSD read for. This is a SECOND, unrelated use of memory locking in ds4; see `simulate-a-smaller-machine` for the first. (3) IN-FLIGHT SAFETY IS A SEQUENCE COUNTER, not a lock: ds4 stamps entries with `inflight_seq` against `g_stream_expert_cache_done_seq` (`ds4_metal.m:1161-1187`) so a slot a queued kernel is still reading cannot be evicted. Frink has the equivalent guard already and it is stronger, because it is structural: `ExpertSlots` refuses to evict a leased entry and an `Arc` with `strong_count > 1` is simply not freeable (`expert_store.rs:26-33`). Do not replace it with a counter. (4) The slab allocator halves its request on failure rather than giving up (`ds4_metal.m:12727`), which is what makes an over-optimistic auto budget degrade instead of failing to start."
    status: pending
  - id: byte-budget-to-expert-count
    content: "THE BUDGET QUESTION, and frink has already written the arithmetic without anyone noticing it is ds4's. A user says a number of bytes; the engine converts it to a COUNT OF EXPERT SLOTS, because a slab of fixed-size slots is the only thing you can bound. ds4: `ds4_ssd_cache_experts_for_byte_budget` is literally `bytes / per_expert_bytes` (`ds4_ssd.c:74`), and `ds4_ssd_auto_cache_plan` (`ds4_ssd.c:113`) takes a percentage of the backend's recommended working set, subtracts non-routed weights, and divides. FRINK ALREADY HAS BOTH HALVES. `frink_core::expert_budget::plan_cache_budget` (`crates/frink-core/src/expert_budget.rs:140`) computes `(spare as u64) / bytes_per_expert` at `pool.rs:170`, fills the expert cache before KV, and floors at `2 * num_experts` when `prefill_overlap` is on (`pool.rs:158-162`), which is ds4's `DS4_STREAMING_PREFILL_HEADROOM_LAYERS = 2` (`ds4.c:4567`) arrived at independently. `net_cache_budget_bytes(memory_ratio, baseline_free, weights, fixed)` at `pool.rs:56` is `ds4_ssd_auto_cache_plan`'s percentage split. NOTHING CALLS EITHER FUNCTION (no external call sites for `plan_cache_budget` anywhere in the repo). THE WORK IS WIRING, NOT MATH. Surface: keep `FRINK_EXPERT_CACHE_BYTES` (it exists, `docs/CONFIG.md:79`) and add a `--ssd-streaming`-equivalent that means 'auto', defaulting via `plan_cache_budget` against the probed budget frink already computes for KV (`crates/frink-cli/src/run.rs:295-350`). COPY ds4's TWO-MEANING ARGUMENT, because it is genuinely useful and cheap: `ds4_parse_streaming_cache_experts_arg` (`ds4_ssd.c:47`) treats a bare number as an EXACT SLOT COUNT with no accounting and an `NGB` suffix as a BYTE BUDGET that also reserves the prefill headroom (`ds4_help.c:172`). The bare count is what you use when benchmarking and want the accounting out of the way. ds4's auto default is 80% of the recommended working set with a documented reason for not going higher (transient spikes ride on top of the steady state and the OOM killer does not negotiate, `ds4_ssd.c:88-105`); frink should record the same reasoning next to whatever number it picks."
    status: pending
  - id: shipped-profile-and-runtime-learning
    content: "THE POLICY QUESTION, and the answer is BOTH, for a reason that is specific and not a compromise. ds4 ships a PRECOMPUTED, PROFILED hotlist as a C header: `ds4_streaming_hotlist.inc` is 13,334 lines of `{layer, expert}` pairs, header comment `sorted by hits/weight`, with a second file per model family (`ds4_streaming_hotlist_glm52.inc`, `sorted by preload priority`, noting `First 4096 entries preserve the original Q2 routed-expert hot seed`). It is applied at LOAD AND PREFILL BOUNDARIES ONLY (`ds4.c:31057` from `ds4.c:35054`, `ds4.c:48533`, `ds4.c:58856`), calling `note_route_hotness(layer, expert, priority)` (`ds4_metal.m:16626`) to prime the SAME counter the runtime then maintains. So it is a SEED, not a substitute. At runtime ds4 measures for real: `note_selected_hotness` adds 1 per selected expert per decode token (`ds4_metal.m:12913`), `note_frequency_hotness` adds the full occurrence count over a prefill batch (`ds4_metal.m:12936`), and the counters decay by a right shift every 16 tokens (`ds4_metal.m:12867`, `:615`) so it is a moving window rather than a lifetime tally. Eviction is hotness-LFU with an LRU tiebreak (`ds4_metal.m:13936-13942`), and the comment there is the load-bearing insight: hotness counts SELECTIONS EVEN ON A MISS, because hit-count LFU penalises an expert that is selected constantly but evicted before its second hit (`ds4_metal.m:13918-13922`). WHEN EACH IS RIGHT. A shipped profile is right for the COLD START, which on a 4x-oversubscribed model is not a warm-up detail: the first tokens are the ones a user judges, and with no seed every one of them is a full miss. It is also the only signal available where runtime measurement is unavailable or too expensive. Runtime learning is right for EVERYTHING AFTER, because a profile is per checkpoint AND per workload, and a hotlist profiled on general chat is wrong for a user who only writes Rust. FRINK SHOULD DO BOTH, and it is already shaped for it: `ExpertCache` accumulates `routing_freq` and exposes `routing_histogram` (`expert_cache.rs:809`) and `routing_skew` (`:830`), which is the profiling instrument, and `ExpertStore::prefetch(&[ExpertKey])` (`expert_store.rs:173`) is the seed applicator whose doc comment already says `caller supplies the hotlist` and which no caller supplies. Ship the profile as DATA, not as a generated `.rs` header: a small file next to the GGUF or keyed by the checkpoint's own hash, produced by `frink` itself from `routing_skew` on a profiling run. A 13k-line generated source file is ds4 solving a C build problem frink does not have. THE REFUSAL RULE APPLIES: a profile whose checkpoint hash does not match must be IGNORED with a line saying so, never applied to a different file, because a wrong hotlist is worse than none (it evicts the right experts to make room for the wrong ones)."
    status: pending
  - id: metal-routing-readback-off-the-encode-thread
    content: "THE EVIDENCE THAT THE POLICY QUESTION IS NOT ACADEMIC, and the reason frink cannot currently do the runtime half. `crates/frink-metal/src/attn.rs:5622` is `let all_ids = vec![Vec::new(); layers.len()];`, preceded by the comment `Skip expert-id host download on the hot path (sync tax). Hotness tracking can be re-enabled later via a side channel if needed.` The buffer it would read is `scratch.ids`, allocated once at `top_k` capacity (`attn.rs:4557`, resized at `:4637`) and rebound by every layer inside one command buffer (`attn.rs:5235`, `:5246`, `:5274`, `:5291`), so even a post-wait read would yield only the LAST layer's selection. Both halves of the defect are real and independent: the download is skipped, AND the buffer could not answer it. Downstream, `MoeWeights::placement_plan` (`crates/frink-models/src/decoder/weights.rs`) feeds `activation_counts` into `PlacementPlan::from_budget` (`crates/frink-moe/src/lib.rs:705`), so frink's eviction and placement policy has been reading an all-zero hotness signal on Metal. The repo already knows: `crates/frink-cli/src/layer_divergence.rs:676` has a test named `a_side_that_recorded_nothing_is_not_agreement` written precisely because scoring zeros as a match `would announce that routing agreed when one side never said where it routed`. HOW ds4 PAYS THE SYNC TAX WITHOUT PAYING IT: it does not skip the readback, it MOVES IT OFF THE ENCODING THREAD. `ds4_gpu_signal_batch_and_wait_event(\"selected-id readback\")` then `ds4_gpu_tensor_read(selected, ...)` (`ds4_metal.m:38855-38869`) run on a dedicated worker, `metal_graph_selected_async_load_worker_main` (`ds4.c:21386`), which waits on a GPU event and then starts the load. `ds4_gpu_stream_expert_cache_note_service_thread` (`ds4_metal.m:514`) records that thread's `pthread_self()` so any cache path that would wait on a command buffer from it FAILS THE LOAD INSTEAD and lets the caller retry synchronously (`ds4_metal.m:509-512`, checked at `:14056`, `:14430`, `:14700`). ds4 also caches the ids it read so the later expert-compute stage skips a SECOND sync (`ds4_metal.m:15859`, taken at `:15885`). THE MINIMUM FIX FOR FRINK, independent of everything else in this plan and worth doing anyway: widen `scratch.ids` to `top_k * n_layers`, or replace it with an atomic histogram buffer the kernel increments, which costs no readback ordering at all and is the better answer if the only consumer is hotness rather than exact per-token routing."
    status: pending
  - id: async-expert-reads
    content: "THE MEASURABLE PERFORMANCE GAP, and it is not subtle. `ExpertStore::prefetch` is `for &key in keys { let _ = self.acquire(key); }` (`expert_store.rs:173-177`): synchronous, on the calling thread, one expert at a time. `GgufExpertSource::read_expert` issues three sequential `read_exact_at` calls (gate, up, down) and returns (`crates/frink-models/src/loader/expert_source.rs`). At queue depth 1 an NVMe SSD delivers a fraction of its rated bandwidth; the device wants tens of requests in flight. ds4 runs a persistent condition-variable pread pool, DEFAULT 9 THREADS, max 18, tunable by `DS4_METAL_STREAMING_EXPERT_PREAD_THREADS` (`ds4_metal.m:11914-11928`, workers at `:12003-12073`), specifically to parallelise 3 tensors times N missing experts across the SSD's queue. `begin_selected_load` fires the pool and RETURNS (`ds4_metal.m:15421-15434`); the join is a separate `pending_load_finish` (`ds4_metal.m:15340`), and if the in-flight batch turns out not to be the one now needed it is DISCARDED rather than waited on (`ds4_metal.m:15451`). It also degrades to a synchronous read on the calling thread if the pool cannot start (`ds4_metal.m:15437`), which is the right failure mode. TWO CHEAP MACOS-SPECIFIC WINS ds4 TAKES AND FRINK DOES NOT. (1) `fcntl(fd, F_RDADVISE, &ra)` issued just before each pread (`ds4_metal.m:11821`, called at `:14812`, `:15276`, `:15568`). (2) `posix_madvise(..., POSIX_MADV_DONTNEED)` on eviction to stop the page cache growing behind your own cache (`ds4_metal.m:12811`, opt-in). That second one is the same instinct as FreeToken's explicit `drop_page_cache` after reading a shard (`.scratch/FreeToken/python/freetoken/models/deepseek_v4/weight.py:53`, `:198`, `:229`), and it is the sharpest argument for why mmap plus the page cache does not get this for free: the page cache is a second, uncontrolled cache competing for the same RAM as yours, with no eviction order you can influence and no readahead you can direct. That is exactly what a hotlist and a bounded slab fix. NOTE THE ORDER OF WORK: frink's model already has the right shape for this because `acquire` reads OUTSIDE the store lock so concurrent misses on different experts already overlap their I/O (`expert_store.rs:39-43`). What is missing is a caller that issues them concurrently."
    status: pending
  - id: prefill-is-a-different-problem-with-an-easier-answer
    content: "THE PREFETCH QUESTION, whose premise is half wrong in a useful way. The question is: what do you fetch before you need it, when routing for layer L+1 is unknown until L finishes? THE ANSWER IS THAT YOU DO NOT NEED L+1. ds4's prefetch is INTRA-LAYER: `begin_selected_load` is called right after the ROUTER kernel of layer il, then `ds4_gpu_flush_commands()` lets the GPU keep working while the pool preads (`ds4.c:42103`, `:42119`), so the SSD read overlaps the REST OF LAYER il's GPU WORK. There is no next-layer speculation and no speculative router anywhere in ds4. The window you are exploiting is the gap between 'the router said which experts' and 'the expert GEMMs need the bytes', which within one layer is real, known-exact, and requires predicting nothing. AND PREFILL IS A COMPLETELY DIFFERENT REGIME. During prefill, hundreds of tokens per layer route to most experts, so you stream the WHOLE LAYER: perfectly sequential, perfectly predictable (layer order is known in advance), bandwidth-bound instead of latency-bound, and double-bufferable. That is why ds4 reserves exactly two full routed layers of headroom (`DS4_STREAMING_PREFILL_HEADROOM_LAYERS = 2`, `ds4.c:4567`) and why FreeToken's DSV4 MoE has a `base whole-layer streaming prefill` separate from its `slot-cache / cpu / hybrid decode paths` (`.scratch/FreeToken/python/freetoken/models/deepseek_v4/moe.py:74-76`). It also matters for LAYOUT, which is the one place layout does matter: within a single packed 3D GGUF expert tensor, expert e's slice is contiguous and the stride is `expert_bytes`, so a whole-layer prefill read of `ffn_gate_exps` is ONE LARGE SEQUENTIAL READ, not 256 random ones. FRINK ALREADY HAS THIS BUILT AND UNWIRED: `ExpertCache::prefetch_prefill_layer(layer, bank_feat_bytes) -> PrefillPlan` (`crates/frink-core/src/expert_cache.rs:1001`), with `begin_prefill` (`:963`), `release_prefill_layer` (`:1109`), `prefill_overlap_fits` (`:930`), `MissRun` coalescing (`:275`), a `whole_layer` flag on `BankEntry` (`:311`), and TEN dedicated tests covering parity rotation, miss-run coalescing, snapshot-versus-live map, and refusing to reuse a buffer before release. Zero external call sites. THE HONEST CAVEAT: ds4's measured prefill under streaming is 3 to 5 tok/s at 4096 tokens versus 94 tok/s for the same model fully resident across two machines (README.md:718-722), so 'sequential and predictable' means the double buffer works, not that prefill is cheap. README.md:319 says `Long prefills can still be fast` and that measured table disagrees with it; trust the table."
    status: pending
  - id: simulate-a-smaller-machine
    content: "THE VERIFICATION QUESTION, and ds4 hands over the answer as a 60-line function. You cannot test a 155 GB checkpoint on a 32 GiB machine, and you do not need to, because THE THING UNDER TEST IS THE RATIO, NOT THE FILE. Two mechanisms, and frink already has the better one. (1) ALREADY IN THE REPO: `store_backed_experts_produce_bit_identical_logits_to_resident` (`crates/frink-models/tests/gguf_roundtrip.rs:446`) loads `frink_real_moe_test.gguf` at a 64 MiB budget AND at a ONE BYTE budget, and asserts `assert_eq!` on f32 logits with no tolerance at both, because same bytes through same kernels is bit-identical by construction. The 1-byte case forces every single acquire down the uncached pass-through path. That is precisely the 'small MoE with an artificially tiny budget' answer, it is committed, and it is the pattern every item in this plan should extend rather than replace. What it does NOT cover: it is CPU-only and single-threaded, so it proves correctness and says nothing about the Metal path, concurrency, or eviction ORDER. (2) COPY FROM ds4: `--simulate-used-memory NGB` (`ds4_cli.c:1903`, help at `ds4_help.c:175`) is `ds4_ssd_memory_lock_acquire` (`ds4_ssd.c:172`), which `mmap`s N GiB anonymous, TOUCHES EVERY PAGE, and `mlock`s it in 256 MiB chunks before the model loads, so the rest of the run genuinely sees a smaller machine. The touch-then-lock and the chunking are both deliberate (a single huge `mlock` is hard to diagnose and can create long uninterruptible VM work on macOS, `ds4_ssd.c:159-163`). NOTE THIS IS A DIFFERENT LOCK FROM `metal-slot-device`'s: that one pins the cache so it stays; this one steals RAM so it cannot be used. Same syscall, opposite purpose. WITH BOTH, THE TEST MATRIX IS REACHABLE ON THE DEV MACHINE: take a MoE that fits (OLMoE Q4_0, Qwen1.5-MoE), lock away enough RAM to create a 2x, 3x and 5x oversubscription, and measure tok/s and `ExpertStoreStats` at each. That is a real oversubscription curve on hardware we have, and it is what lets `tell-the-user-the-cost-first` say a number instead of a shrug. It also makes the 155 GB question answerable by extrapolation with the extrapolation stated as such."
    status: pending
  - id: tell-the-user-the-cost-first
    content: "THE HONEST-COST QUESTION. THE NUMBERS WE HAVE, all from ds4's own README on hardware far better than the target, GLM 5.2 IQ2_XXS at 188 GiB, which is 1.47x oversubscription on a 128 GB M5 Max (README.md:718-722): SSD streaming decode ~4.8 tok/s and prefill ~3 to 5 tok/s at 4096 tokens, against ~16.8 tok/s decode and ~94 tok/s prefill for the same model fully resident across two machines. So streaming costs roughly 3.5x on decode and 20 to 30x ON PREFILL, and prefill is the disaster, not decode. ds4's own framing is careful and worth transcribing: streaming `is not as fast as fitting the full model in RAM`, it `still needs memory for non-routed weights, KV cache, graph scratch, activations, and the routed expert cache`, and it is useful only because `routed experts dominate model size and modern Mac SSDs are fast enough to make cache misses tolerable` (README.md:311-320). WHAT FRINK SHOULD PRINT, BEFORE THE RUN STARTS, not after the user waits four minutes for a first token. Frink already has most of the machinery: `crates/frink-cli/src/run.rs:295-350` computes a device budget, prices weights against it, prints `budget`, `fit` and `budget.caveat()`, and already knows to suggest `stream experts (FRINK_EXPERT_CACHE_BYTES)` when nothing fits (`run.rs:318`). ADD THREE LINES TO IT: (a) the oversubscription ratio, routed-expert bytes against the chosen cache budget, stated as a number; (b) the resulting expert slot count and how many layers' worth that is, which is `ds4_ssd_cache_plan`'s `cache_experts` and `effective_cache_bytes` (`ds4_ssd.h:12-17`) and is what makes an abstract budget legible; (c) an EXPECTED tok/s range interpolated from the oversubscription curve `simulate-a-smaller-machine` produces, explicitly labelled as an estimate from this host's own measurements. RESPECT THE EXISTING REFUSAL RULE: `--strict-budget` already turns a budget warning into a refusal (`run.rs:344`), and out-of-core should hook the same switch rather than inventing a second one. AND SAY THE UNCOMFORTABLE THING WHERE THE USER WILL SEE IT: at 4.8x, which is 155 GB against 32 GiB, we have no evidence anyone has made this usable. ds4 ships this capability and does not offer a 32 GB recipe for any DeepSeek V4 model; its smallest Flash quant is ~81 GB, recommended for 96 to 128 GB machines (`download_model.sh:52`). The reachable target on THIS laptop is the smaller quant at ~2.5x, which is within a factor of two of the ratio ds4 has actually demonstrated."
    status: pending
  - id: deepseek4-is-not-the-prerequisite-and-must-not-become-one
    content: "SEQUENCING, and getting this wrong would stall the whole plan behind an unrelated piece of work. FRINK CANNOT LOAD DeepSeek-V4-Flash TODAY, at any size, resident or streamed. `crates/frink-models/src/capability.rs:553` registers `deepseek4` as `dedicated(\"DeepSeek V4 needs CSA/HCA + mHC assembly; generic GQA Decoder is not valid\")`, and `crates/frink-models/src/loader/tests/config.rs` has a test asserting a `deepseek4` GGUF returns `LoadError::DedicatedArchitectureRequired`. That refusal is correct under the north star's rule (`the-bar` item 1: refuse rather than load-and-compute-something-else). The dedicated path that would replace it, `crates/frink-models/src/deepseek_v4_decoder.rs`, says of itself in its module doc: `Synthetic weights only, not a real checkpoint path. No GGUF loader, no Engine wiring, no claim of oracle-correct output against a production DeepSeek V4 file.` The real primitives are genuinely there and independently tested (`frink-core/src/csa_hca_compress.rs`, `deepseek_v4_attention.rs`, `frink-models/src/hyper_connections.rs`, `output_projection.rs`, all transcribed from llama.cpp PR #24162), and `deepseek_v4_pro()` (`crates/frink-models/src/config.rs:519`) is a preset sketch, which CLAUDE.md already says plainly. Its own doc lists what is deliberately unbuilt: incremental DSV4 KV state, CSA's `coff=2` dual-role projection, hash-based first-layer MoE selection, multi-layer stacking. THEREFORE: build and verify every item in this plan against a MoE THAT ALREADY LOADS. OLMoE-1B-7B Q4_0 and Qwen1.5-MoE are in the bench suite; `test_moe_fixture` is in the repo. deepseek4 is a CONSUMER of out-of-core, not a gate on it, and the two should be tracked separately or the streaming work will be blocked for months on a decoder. A useful side effect of doing it this way round: out-of-core landing first means that when the deepseek4 loader does arrive, the thing that makes it runnable on consumer hardware is already measured. `.scratch/FreeToken/python/freetoken/models/deepseek_v4/` (2039 lines across `moe.py`, `compress.py`, `weight.py`, `attention.py`) is the working reference for that later work, and its `weight.py:88-91` refuses to yield routed experts at all outside offload mode, which is a strong hint about how tightly the two are coupled in a real DSV4 implementation."
    status: pending
  - id: no-repack-and-what-would-change-that
    content: "THE LAYOUT QUESTION, answered NO with the conditions under which the answer flips. STOCK GGUF IS SUFFICIENT AND BOTH REFERENCE ENGINES AGREE. ds4 streams from the unmodified GGUF: `ds4_gpu_stream_expert_table` is `{ model_map, model_size, layer, n_total_expert, gate_offset, up_offset, down_offset, gate_expert_bytes, down_expert_bytes }` (`ds4_gpu.h:196-205`), raw byte offsets into the mapped file, and the miss path preads that fd (`ds4_metal.m:11968`). Frink does the same: `GgufExpertSource` holds `(file index, offset, len)` per expert and reads them positionally, with the doc comment stating the choice explicitly, `no mmap of the expert region, no shared seek cursor` (`crates/frink-models/src/loader/expert_source.rs`). No repacking step exists in either. `ds4_layer_pack.c` is not a counter-example: it computes a monotonic-contiguous LAYER-TO-DEVICE assignment for multi-GPU (`ds4_layer_pack.h:23-46`, `ds4_compute_layer_placement` called at `ds4.c:56649`, printed at `ds4.c:56920`), it is pure C99 with no I/O, and it never touches a file. WHY STOCK GGUF IS GOOD ENOUGH: within one packed 3D expert tensor, expert e's rows are contiguous with stride `expert_bytes`, so a single expert is one contiguous range per tensor and a whole layer is one large sequential range. THE COST YOU DO PAY, stated precisely so it can be measured rather than argued: gate, up and down are three SEPARATE tensors at distant file offsets, so one expert costs THREE seeks, not one. At top-k 8 that is 24 random reads per layer per token. This is where a repack would help, and it is a small one: interleave gate/up/down per expert so `(layer, expert)` is a single contiguous range, turning 24 reads into 8. WHEN TO DO IT: only after `measure-what-exists-before-designing-anything` and `async-expert-reads` show seek count is the binding constraint rather than raw bandwidth or queue depth. A 9-thread pread pool may hide the three-seek cost entirely, in which case a repack buys nothing and costs a 155 GB rewrite. IF IT EVER HAPPENS: it must be a separate artifact produced by a `frink repack` command, keyed to the source checkpoint's hash, with the ORIGINAL GGUF STILL USABLE AND STILL THE DEFAULT. A format only frink can read is a reason not to choose frink, which is the opposite of the north star."
    status: pending
  - id: the-first-three
    content: "ORDER, chosen so each lands and is verifiable alone. STEP 1: `measure-what-exists-before-designing-anything`. No code changes. Run OLMoE Q4_0 and Qwen1.5-MoE resident, then under `FRINK_EXPERT_CACHE_BYTES` at a few budgets, on CPU and Metal, and record tok/s plus `ExpertStoreStats`. DONE MEANS a table in `benchmarks/RESULTS.md` and a number for what streaming costs today. This is first because the whole plan rests on a claim (`stored-experts-disable-metal`) that is currently read from source rather than measured, and because if the CPU cost turns out to be small the priority order below changes. STEP 2: `stored-experts-disable-metal`. Make `ExpertBacking::Stored` survive the three Metal guards (`layer_supports_metal_moe_resident`, `try_metal_moe_topk`, `metal_prefill_moe`) by holding top-k `ExpertLease`s across the fused launch, which `ExpertLease::shared_buf` (`expert_store.rs:80`) already exists to permit. DONE MEANS the step 1 Metal row re-measured and no longer catastrophic, plus a bit-identical assertion against the resident Metal path in the shape of `gguf_roundtrip.rs:446`. Smallest change with the largest effect, and it is the concrete first half of the north star's 'a `SlotDevice` for Metal'. STEP 3: `simulate-a-smaller-machine`, the `--simulate-used-memory` equivalent, ported from `ds4_ssd.c:172` including the touch-then-lock and the 256 MiB chunking. DONE MEANS an oversubscription curve at 2x, 3x and 5x on a model we already have, which is the input every later item needs: it tells `tell-the-user-the-cost-first` what to print, it tells `shipped-profile-and-runtime-learning` how much a cold start actually costs, and it is the only way any claim about the 155 GB target gets made honestly on this hardware. AFTER THOSE THREE, ranked but not scheduled: `metal-routing-readback-off-the-encode-thread` (cheap, fixes a defect that is wrong independent of this plan), `async-expert-reads` (largest expected speedup), `one-budget-not-two` (must be decided before a Metal slab is written), then `metal-slot-device`, `byte-budget-to-expert-count`, `prefill-is-a-different-problem-with-an-easier-answer`, `shipped-profile-and-runtime-learning`. `no-repack-and-what-would-change-that` stays closed unless step 1 reopens it."
    status: pending
isProject: false
---

# Out-of-core MoE: run a model larger than the machine

> An MoE touches top-k of N experts per token. The working set is a
> fraction of the weights. That is the whole reason this is reachable,
> and it is the only reason.

Expands `north-star.md`'s `t1-out-of-core-execution`. Primary reference
is `.scratch/ds4` (DwarfStar), a C engine that already ships this.

## Status, 2026-10-28

Read this before the corrections below, two of which no longer hold.

- **Streaming is correct on CPU and pinned.** The OLMoE-1B-7B Q4_0
  garbage (" amongst amongst" streamed, " Paris." resident) came from the
  int-dot repack cache keying interleaved copies by buffer ADDRESS while
  the store freed and reused buffers. It was fixed on 2026-08-28 and
  never pinned, because the fixture test ran with `FRINK_CPU_INT_DOT`
  off, the library default, and only the int-dot kernels use the cache.
  `tests/streaming_int_dot.rs` runs the shared store-vs-resident check
  with int-dot on, and putting the address key back turns it red. The
  same work found streamed experts running DIFFERENT kernels from
  resident ones (a serial routed body, and no batched prefill), which
  showed in the last bits of the logits; one `MoeWeights::with_experts`
  now hands either backing to the same body.
- **The second correction below is stale.** `ExpertBacking::Stored`
  passes `layer_supports_metal_moe_resident` now
  (`crates/frink-models/src/decoder/metal_eligibility.rs`): the fused
  launch materialises each expert into an owned view that pins its store
  entry. A streamed layer still copies the expert into an `MTLBuffer`
  per use rather than holding it in a device slot, so the two-copies
  problem in "Two caches, one RAM" stands.
- **The Metal `SlotDevice` is not started**, deliberately: it was
  deferred on 2026-10-28 because the session doing this work could not
  compile Metal kernels, and a slot pool whose kernels nobody ran is
  the `CudaExpertPool` situation again. What it needs is the
  `metal-slot-device` todo above, unchanged. The one thing to add to it:
  slots are REUSED, so a slot's address says nothing about what it
  holds, and a slot view must stay out of the repack cache (hand it out
  as `WeightBytes::Shared`, which `WeightBytes::map_id` already refuses
  to key). That is the August bug, one layer down.
- **Not yet done:** a run on a real MoE with shared experts (Qwen1.5-MoE)
  and any Metal run of the streamed path.

## Three corrections the investigation produced

**Frink already streams experts.** `crates/frink-core/src/expert_store.rs`
is a bounded, lease-protected byte cache, wired into both decode paths
behind `FRINK_EXPERT_CACHE_BYTES` / `FRINK_SSD_STREAMING`, and proven
bit-identical to the resident path at a generous budget **and** at a
one-byte budget (`crates/frink-models/tests/gguf_roundtrip.rs:446`).
The starting position is "built, tested on a fixture, never run against
anything large", not zero.

**Turning it on used to disable Metal.** `ExpertBacking::Stored` failed
the guard in `layer_supports_metal_moe_resident` and was explicitly
rejected in the fused top-k path:

```rust
ExpertBacking::Stored { .. } => {
    // Streaming experts: fall back (can't hold all refs easily).
    return None;
}
```

On the exact backend this user runs, the feature that lets you exceed
RAM switched off the kernels that make frink competitive. Since
`42468f5` the guard (`decoder/metal_eligibility.rs`) accepts
`ExpertBacking::Stored`, and `try_metal_moe_topk` (`decoder/metal_moe.rs`)
materialises top-k into owned views that each hold a pin on the store.
The packed `mul_mv_id` path, the resident phase-2 helper and the two
Metal prefill MoE paths still decline `Stored`.

**`ds4_layer_pack` is not a disk format.** It is a monotonic-contiguous
multi-GPU *layer placement* packer (`ds4_layer_pack.h:23`, callers at
`ds4.c:56649` and `ds4.c:56920`). It is pure C99, does no I/O, and has
nothing to do with streaming. The layout question has a cheap answer:
both reference engines stream from the **stock GGUF**.

## The seven questions

| | Answer |
|---|---|
| **Layout** | Stock GGUF. ds4 preads the model fd at raw offsets (`ds4_gpu.h:196`, `ds4_metal.m:11968`); frink already does the same (`loader/expert_source.rs`). A repack would turn 3 seeks per expert into 1, and is worth it only if seek count proves binding. |
| **Budget** | Bytes in, **expert slot count** out. `ds4_ssd.c:74` is `bytes / per_expert_bytes`. `frink_core::expert_budget::plan_cache_budget` (`pool.rs:170`) already computes it, with ds4's two-layer prefill headroom (`pool.rs:158`) arrived at independently. Nothing calls it. |
| **Policy** | **Both.** A shipped profile owns the cold start; runtime learning owns everything after. ds4 seeds its runtime counter *from* the hotlist (`ds4_metal.m:16626`), so they are one mechanism with two sources. |
| **Prefetch** | You never need layer L+1. ds4 prefetches **within** a layer: start the load right after the router kernel, overlap the rest of that layer's GPU work (`ds4.c:42103`). Prefill is a different regime: stream whole layers, double-buffered. |
| **Cost** | ~4.8 tok/s decode, ~3-5 tok/s prefill, measured by ds4 at **1.47x** oversubscription on a 128 GB M5 Max (`README.md:718`). Prefill is the disaster, not decode. |
| **Verification** | Small MoE, artificially tiny budget. Already committed (`gguf_roundtrip.rs:446`). Add ds4's `--simulate-used-memory` (`ds4_ssd.c:172`) to make the *ratio* testable on hardware we own. |
| **Order** | Measure. Unblock Metal. Build the shrink-the-machine harness. |

## Two caches, one RAM

Frink has two independent expert caching systems that do not know each
other exists, plus a third residency vocabulary in `frink-moe`. The
split shows in the type names: `ResidencyPlan` exists in both
`frink-core/src/residency.rs:191` and `frink-moe/src/lib.rs`;
`RebuildRejected` in both `expert_cache.rs:1150` and `pool.rs:231`.

| | `frink-core::ExpertStore` | `frink-core::ExpertCache` + `expert_slots` |
|---|---|---|
| Status | **wired**, bit-identical tested | **113 tests, zero external callers** |
| Source | pread, `read_exact_at` | plans only, no I/O |
| Policy | pure LRU stamp | hotness histogram, LFU-shaped |
| Prefill | none | double buffer, 10 tests |
| Device | `Arc<Vec<u8>>` host bytes | `SlotDevice` trait |

On Apple unified memory a host byte cache and a device buffer cache are
**the same physical RAM**. `Stored` produces `WeightBytes::Shared`,
which is not a registered mmap and so cannot take the `BytesNoCopy`
alias path (`crates/frink-metal/src/gpu.rs:5097`); inference is that it
gets copied into a second `MTLBuffer`. Two copies of every hot expert on
a machine whose entire problem is 32 GiB.

ds4's shape avoids it: one `MTLResourceStorageModeShared` slab
(`ds4_metal.m:12479`), preads straight into `[buf contents]`
(`ds4_metal.m:14846`), `didModifyRange:` (`ds4_metal.m:14876`). One copy
total, file to unified memory.

Note what is **not** missing: a host `SlotDevice` already exists
(`HostSlotMemory`, `expert_slots.rs:671`), and an SSD-backed source
already exists (`GgufExpertSource`). The gap is Metal, plus the wiring.

## Why the hotlist matters here specifically

Frink tries to learn expert hotness at runtime through
`activation_counts`. On Metal that signal is **all zeros**:

```rust
// crates/frink-metal/src/attn.rs:5619
// Skip expert-id host download on the hot path (sync tax). Hotness
// tracking can be re-enabled later via a side channel if needed.
let all_ids = vec![Vec::new(); layers.len()];
```

Both halves of the defect are real and independent. The download is
skipped; and `scratch.ids` is a single `top_k`-capacity buffer
(`attn.rs:4557`) rebound by every layer inside one command buffer
(`attn.rs:5235`, `:5246`, `:5274`, `:5291`), so even a post-wait read
would yield only the last layer's selection. Downstream,
`placement_plan` (`decoder/weights.rs`) has been feeding zeros into
`PlacementPlan::from_budget` (`frink-moe/src/lib.rs:705`).

The repo already knows. `crates/frink-cli/src/layer_divergence.rs:676`
holds a test named `a_side_that_recorded_nothing_is_not_agreement`,
written because scoring zeros as a match "would announce that routing
agreed when one side never said where it routed."

ds4 sidesteps the problem by profiling **offline** and shipping the
answer: `ds4_streaming_hotlist.inc` is 13,334 lines of `{layer, expert}`
pairs, "sorted by hits/weight", one file per model family.

**These are two different designs, not one broken and one working.**

- A **shipped profile** owns the cold start. On a 4x-oversubscribed
  model that is not a warm-up detail: the first tokens are the ones a
  user judges, and with no seed every one of them is a full miss.
- **Runtime learning** owns everything after, because a profile is per
  checkpoint *and* per workload. A hotlist profiled on general chat is
  wrong for someone who only writes Rust.

ds4 does both, and the join is one line: the hotlist calls
`note_route_hotness(layer, expert, priority)` (`ds4_metal.m:16626`),
priming the same counter that `note_selected_hotness`
(`ds4_metal.m:12913`) then maintains, decaying by a right shift every 16
tokens (`ds4_metal.m:12867`).

Frink should do both, and is already shaped for it: `routing_histogram`
(`expert_cache.rs:809`) is the profiler, `ExpertStore::prefetch`
(`expert_store.rs:173`) is the seed applicator whose doc comment already
says "caller supplies the hotlist" and which no caller supplies. Ship
the profile as **data keyed to the checkpoint hash**, not a generated
`.rs` file: a 13k-line header is ds4 solving a C build problem frink
does not have. A profile whose hash does not match must be **ignored
with a line saying so**. A wrong hotlist is worse than none, because it
evicts the right experts to make room for the wrong ones.

## What mmap gets you free, and what it does not

Free: demand paging. Frink already mmaps (`frink-gguf/src/lib.rs:272`)
and on Metal already aliases the whole file as one `MTLBuffer` via
`newBufferWithBytesNoCopy` (`frink-metal/src/gpu.rs:5097`), so a
GPU-side read of a cold weight can fault in from SSD with no copy.

Not free, and this is the part a hotlist fixes: **no control over
eviction order and no control over readahead**. Worse, the page cache is
a *second* uncontrolled cache competing for the same RAM as yours. Both
reference engines fight it explicitly. ds4 calls
`posix_madvise(..., POSIX_MADV_DONTNEED)` on eviction
(`ds4_metal.m:12811`); FreeToken calls `drop_page_cache` right after
reading a shard (`.scratch/FreeToken/.../deepseek_v4/weight.py:53`).

Both also issue readahead frink does not: ds4 fires
`fcntl(fd, F_RDADVISE, &ra)` before every pread (`ds4_metal.m:11821`).

## What this costs, said before the run

All numbers from ds4's README on hardware better than the target. GLM
5.2 IQ2_XXS, 188 GiB, on a 128 GB M5 Max, which is **1.47x**
oversubscription (`README.md:718-722`):

| | fully resident (2 Macs, TP) | one Mac, SSD streaming |
|---|---|---|
| decode | ~16.8 tok/s | **~4.8 tok/s** |
| prefill (4096 tok) | ~94 tok/s | **~3-5 tok/s** |

Roughly 3.5x on decode and 20-30x on prefill. `README.md:319` claims
"Long prefills can still be fast"; that table disagrees with it. Trust
the table.

**The target ratio is 4.8x, not 1.47x.** 155 GB against 32 GiB. ds4
ships this capability and offers no 32 GB recipe for any DeepSeek V4
model; its smallest Flash quant is ~81 GB, recommended for 96-128 GB
machines (`download_model.sh:52`). The reachable target on this laptop
is the smaller quant at ~2.5x, within a factor of two of what ds4 has
actually demonstrated.

Frink already prints a budget report before a run
(`crates/frink-cli/src/run.rs:295-350`) and already suggests
`FRINK_EXPERT_CACHE_BYTES` when nothing fits (`run.rs:318`). Add the
oversubscription ratio, the slot count and how many layers that is, and
an expected tok/s range from this host's own curve. Hook the existing
`--strict-budget` switch (`run.rs:344`) rather than inventing a second.

## deepseek4 is a consumer, not a gate

Frink cannot load DeepSeek-V4-Flash today at any size.
`capability.rs:553` registers `deepseek4` as
`dedicated("DeepSeek V4 needs CSA/HCA + mHC assembly; generic GQA
Decoder is not valid")`, and `loader/tests/config.rs` tests that it returns
`DedicatedArchitectureRequired`. That refusal is correct under the north
star's rule. The dedicated path, `deepseek_v4_decoder.rs`, says of
itself: "**Synthetic weights only, not a real checkpoint path.** No GGUF
loader, no `Engine` wiring."

So build and verify all of this against a MoE that already loads: OLMoE
Q4_0, Qwen1.5-MoE, `test_moe_fixture`. Track the deepseek4 loader
separately, or streaming stalls for months behind a decoder.

## Order

```
1  measure what exists          no code; the plan rests on an unmeasured claim
2  unblock Metal                one enum arm; largest effect per line
3  shrink the machine           --simulate-used-memory; makes the ratio testable

then  metal routing readback    cheap, wrong independent of this plan
      async expert reads        ds4 runs 9 pread threads; frink runs 1
      one budget not two        decide before a Metal slab is written
      metal slot device
      byte budget to slot count
      prefill double buffer     already built in frink-core, unwired
      shipped profile + runtime
```

`no-repack` stays closed unless step 1 reopens it.
