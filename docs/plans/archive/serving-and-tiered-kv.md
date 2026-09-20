---
name: serving — tiered KV, prefill/decode fairness, admission
overview: "GOAL: make frink-server behave well under real concurrent load and across restarts — a disk-backed prefix cache that survives a process restart, chunked prefill that does not stall in-flight decodes, and an admission gate that answers 'will this fit' before accepting rather than OOMing later. Sourced from a read-only study of oMLX (.scratch/omlx) whose paged/SSD KV cache and time-debt scheduler are its two genuinely mature subsystems. KEY CORRECTION: oMLX's 'paged KV cache' is NOT paged attention — CacheBlock holds no tensor data — so all of it sits on top of a contiguous per-sequence KV, which is exactly what frink already has."
todos:
  - id: kv-block-hashing
    content: "LANDED as `frink-core::kv_block` (`BlockHasher`, `BlockHash`). Parent-chained SHA-256, root-seeded at H(domain,\"root\",model,extra_keys); each block is H(domain,\"block\",model,extra_keys,parent,token_ids). Every field is length-prefixed, so `[\"ab\",\"c\"]` and `[\"a\",\"bc\"]` cannot collide; the domain tag is versioned so an encoding change orphans old blocks instead of misreading them. `chain()` hashes WHOLE blocks only -- a growing tail has no stable identity. Sampling params are excluded on purpose (KV is sampling-independent). Golden digests cross-validated against an independent Python hashlib reference, per this repo's fixture convention. NOT DONE: nothing consumes it yet -- `PrefixCache` still does a linear longest-common-prefix scan over whole sequences, and no block STORE exists (that is `kv-ssd-tier`); `extra_keys` is a slot with no LoRA/multimodal producer wired to it."
    status: completed
  - id: kv-cache-signature
    content: "LANDED as `frink-core::kv_signature`. `CacheSignature::from_payload` MEASURES the tensors (layers, kv heads, head dim, dtype, token depth) and takes no expected-shape argument to fill a gap from; `seq_len` is treated as a claim and checked against `k.len()`. `UnverifiedBlock::verify` runs three ordered checks: unmarked -> refuse (absence is never agreement); recorded stamp vs measured payload -> `PayloadMismatch` (a stamp may not vouch for a width the payload lacks); only then measured vs reader expectation -> `Incompatible`, naming the field that changed. Format version has an explicit readable-set. Confirmed the central test FAILS when `verify` is patched to trust the stamp instead of the payload. NOT DONE: nothing stores or reads blocks yet, so no signature is written to disk (that is `kv-ssd-tier`); `KvDtype` has one variant (F32) because `KvCache` is `Vec<f32>` -- the enum exists so an f16/quantized KV tier invalidates old blocks instead of reinterpreting their bytes; the signature is not yet bound to a `BlockHash`, since nothing yet holds the two together."
    status: completed
  - id: kv-ssd-tier
    content: "LANDED as `frink-core::kv_disk` (`DiskKvStore`, `DiskConfig`, `DiskStats`, `encode_block`/`decode_block`), ~3.5k lines with its tests. One file per block named by its `BlockHash` and sharded into subdirectories by a hex prefix (`shard_chars`), body = per layer all K elements then all V, dtype passed through unchanged (`KvDtype`, F32 only today because `KvCache` is `Vec<f32>`), no compression. Format version 2 with an explicit `READABLE_FORMAT_VERSIONS` set -- an unknown version is refused, never guessed; v1 was dropped rather than read with the sliding window assumed absent. Publish is temp-file (`<root>/.tmp/<hash>.<pid>.<n>.tmp`) + `fsync` + `rename`, and the post-rename re-check is there: a block evicted while its bytes were in flight has its file removed rather than left behind (`a_block_evicted_mid_write_does_not_leave_its_file_behind`, `no_temp_files_survive_a_successful_write`). Because a temp file can still be torn by a crash, the format records its own lengths and a SHA-256 over header+body, and the lengths are checked against the real file size BEFORE anything is hashed or parsed, so a 4 GB `body_len` on a 200-byte file is refused by arithmetic rather than by allocating (`a_truncated_file_is_refused_at_every_cut_point`, `a_torn_file_on_disk_is_a_miss_and_is_quarantined`). Compatibility is not duplicated here: a decoded file is an `UnverifiedBlock` and only `kv_signature`'s `verify` against the reader's own expectation yields a `KvBlock`. LRU eviction to the budget, `reindex()` reattaches a new process to what the previous one published (`a_new_store_reattaches_to_what_the_previous_one_published`), and `DiskStats` carries `read_nanos`/`write_nanos` -- the time-valued fields whose absence the plan says makes oMLX's tier impossible to tune. NOT DONE: nothing in production consumes it. `PrefixCache` is still a whole-sequence linear scan, no call site cuts a sequence into blocks, `frink-server` never opens a `DiskKvStore` and no env var names a cache root, so the durable prefix cache does not yet survive a restart in a running deployment -- the format and the store are built and tested, the producer is not. There is also no RAM tier above it (the plan's 'ship it on or do not claim it' point is moot until one exists)."
    status: completed
  - id: kv-ssd-async-read
    content: "LANDED, and asynchronous by construction rather than as a retrofit: `DiskKvStore::get` is literally `read_async(..).wait()`, so there is no synchronous read path a prefetch would have to work around. `read_async` returns a `ReadHandle` (`is_ready` / `try_claim` / `wait`); `prefetch(&[BlockHash], &CacheSignature)` hands a whole chain to `reader_threads` background workers, and a demand read that arrives for a block already being read JOINS that read instead of repeating it (one `ReadSlot` per block in a staging map). Staging is capacity-bounded (`prefetch_capacity`) and a refused prefetch is counted (`prefetch_dropped`) rather than allowed to grow -- a prefetch is a hint, and the demand path must never be starved by one. A staged read is keyed by the signature it was issued under, so a prefetch made under one config cannot hand its answer to a reader expecting another shape. `clear_prefetch()` releases what a cancelled request read ahead. Counters: `prefetch_issued/dropped/hits/waits`, `async_reads`, `staged_blocks`, plus `read_nanos`. Tests: `a_prefetched_block_is_already_read_when_the_request_arrives`, `a_whole_chain_can_be_read_ahead_in_one_call`, `a_request_joins_a_read_already_running_rather_than_repeating_it`, `a_staged_read_is_not_reused_by_a_reader_that_wants_another_shape`, `prefetching_is_bounded`, `a_read_handle_can_be_polled_to_completion`, `a_miss_is_answered_without_dispatching_a_read`, and `without_reader_threads_reads_run_on_the_caller` (zero reader threads = the read happens inline, so a test or an embedder can run the tier single-threaded). NOT DONE: no production caller prefetches, for the same reason as `kv-ssd-tier` -- nothing yet cuts a request's prompt into a block chain to read ahead of."
    status: completed
  - id: kv-write-ordering
    content: "LANDED, and stated as an invariant in the module doc: **buffer -> index -> queue**. The payload is reachable before anything claims the block exists, and on the way out the file is published before the buffered copy is released; a reader holds the index lock while it consults the buffer, because 'not on disk yet' and 'here is the payload' have to be ONE decision -- as two, a writer can publish and release in between and the reader finds nothing. A broken invariant surfaces as `StoreError::MissingPayload`, not as a quiet miss, because a correctness bug that degrades into a cache miss is one nobody ever finds. The queue is bounded and NEVER drops: a full queue makes the calling thread write the block itself, counted as `DiskStats::inline_writes` (`a_full_queue_writes_inline_rather_than_dropping_the_block`). Entries carry a `generation`, so a write that finishes after its entry was evicted and re-created cannot mark the NEW entry published, and a superseded queued write cannot overwrite the newer block (`a_superseded_queued_write_does_not_overwrite_the_newer_block`, `a_queued_write_evicted_before_it_runs_is_skipped`). The ordering itself is tested by an injectable `WriteOrder` probe: `a_reader_never_sees_an_index_hit_with_no_payload`, `a_queued_block_is_readable_before_it_reaches_disk`, and a concurrent-reader version -- plus the two deliberate mutations `indexing_before_buffering_is_caught` and `releasing_the_buffer_before_publishing_is_caught`, which assert the probe FAILS under a reordered write path, so the invariant's test is itself mutation-checked in the file."
    status: completed
  - id: kv-disk-budget
    content: "LANDED. `effective_capacity = min(max_bytes, used + free - reserve)` with the `free - reserve` term deliberately SIGNED: clamping at zero would make the ceiling equal `used` exactly when the disk is fullest, leaving the filesystem to do the refusing, which is the failure the budget exists to prevent. Free space comes from a `FreeSpaceProbe` (real `statvfs` `f_bavail * f_frsize` on unix, injectable so the near-full behaviour is testable), cached for `free_space_ttl` (default 2s) so a block write is not a syscall, and thrown away IMMEDIATELY on `ENOSPC` -- whatever the cached reading says, it is wrong now -- with an eviction pass fired on the spot so the next write has somewhere to go. An unmeasurable filesystem falls back to the configured `max_bytes` rather than refusing to cache. Counters: `enospc`, `space_clamped`, `effective_capacity`. Tests: `the_ceiling_falls_when_the_filesystem_fills_up`, `the_free_space_reading_is_cached_for_its_ttl`, `enospc_throws_away_the_cached_free_space`, `an_unmeasurable_filesystem_falls_back_to_the_configured_budget`, `eviction_keeps_the_store_inside_its_budget`, `reindex_evicts_down_to_the_budget`, and `the_platform_probe_measures_a_real_filesystem` against a real temp dir. NOT DONE: the budget is bytes on one root directory only -- no per-model or per-tenant sub-budget, and no way for an operator to set it, since nothing in `frink-server` constructs a `DiskConfig` yet (see `kv-ssd-tier`)."
    status: completed
  - id: kv-swa-block-alignment
    content: "LANDED as `frink-core::kv_swa` (`BlockLayout`, `aligned_block_size`) plus the layout being carried in `CacheSignature`. CORRECTION TO THE PLAN'S WORDING, stated in the module doc: the implementable relation is `window % block_size == 0` (the block size divides the window), not the operands the other way round -- forcing block size up to a multiple of a 128-token gpt-oss window would make every block a whole window. The same constraint a paged KV cache asserts anywhere. `BlockLayout` is only constructible through `new`, so holding one is the proof the rule holds; `aligned_block_size` rounds a requested size DOWN to a divisor so a config layer never hands the cache a size the cache refuses. `ModelConfig::kv_block_window` / `kv_block_layout` bridge it to real models: an alternating-SWA model is constrained as soon as ANY layer slides (5/6 full-attention is not 5/6 exempt). The durable half: `CacheSignature` gained `layout`, the block file format went to v2 with block-size and window fields, and v1 was DROPPED from the readable set rather than read with the window assumed absent -- a v1 file cannot say what window it was cut under, and absence is never agreement, so a restart onto this build starts cold once. `block_size` is payload-checked (a stored block is exactly one whole block, so a stamp claiming 64 over a 48-token payload is `BlockSizeMismatch`); the window is not tensor-provable, so it is carried like `model` and settled against the reader's expectation. Tests confirmed to FAIL when the checks are removed: `a_block_written_under_a_different_window_is_refused_not_reused`, `a_full_causal_reader_will_not_take_a_sliding_window_block`, `a_block_cut_at_a_different_block_size_is_incompatible`, `a_stamp_may_not_claim_a_block_size_the_payload_lacks`, plus the disk-tier `a_block_written_under_one_window_is_not_served_to_a_reader_expecting_another` (write, drop the store, reopen, reindex, ask with a changed window -> miss + `incompatible` counter, and an unchanged window still hits). NOT DONE, deliberately: nothing yet CUTS a sequence into blocks -- `PrefixCache` is still a whole-sequence scan -- so no production call site passes a `BlockLayout` today; the guard is in place before the producer, which is the order that keeps a mis-aligned block from ever becoming durable. And the KV is still stored whole with masking rather than truncated to the window, so the alignment rule is currently prophylactic for frink's own kernels and load-bearing only for the durable format."
    status: completed
  - id: sched-chunked-prefill
    content: "LANDED. `PrefillState` in batch_scheduler.rs is the resumable state machine (caches, tokens_processed, tokens_remaining) with `step_chunk(&mut self) -> bool /* done */`; the worker runs ONE chunk (round-robin over waiting prompts) plus ONE batched decode step per tick, so a long prompt costs an in-flight decode one chunk, not its whole length, and N long prompts still cost one chunk per tick (the oMLX flaw the plan says not to inherit). Chunk size from `FRINK_CB_PREFILL_CHUNK` (default 128) or `BatcherConfig` for tests; `FRINK_CB_MAX_SEQS` now counts prefilling prompts too, since each holds a full KV set. Counters (prefill_chunks / prefill_tokens / decode_steps) on `/metrics`. NOT DONE, deliberately: (a) a chunk is still a per-token `forward_token` loop, not `forward_batch` -- chunking here is a scheduling boundary and is asserted bit-identical to the sequential prefill it replaced, so no throughput claim is made and no benchmark row moves; (b) no time-debt gate, that is `sched-time-debt`; (c) the batcher still cannot use the prefix cache or KV pool, so a chunk boundary is not yet a cache-block boundary."
    status: completed
  - id: sched-time-debt
    content: "Time-debt prefill/decode interleaving: GPUs cannot preempt a running kernel, so chunk DURATION is the scheduling quantum. Cap contended chunks in ms converted to tokens via measured prefill tok/s; each chunk accrues duration*share debt; decode wall-time repays it; the gate blocks the next chunk until debt clears"
    status: pending
  - id: sched-keyed-row-state
    content: "LANDED. `batch_scheduler.rs` keeps in-flight rows in `Rows { state: HashMap<Uid, Slot>, order: Vec<Uid> }` -- keyed state plus an explicit admission order (HashMap iteration order is not deterministic, batch composition must be). `swap_remove` on a `Vec<Slot>` is gone: a stale `Uid` resolves to `None`, never to whichever row moved into the vacated position. The per-step batch is still a slice for the kernel, but `active[j]` holds a `Uid`, so the scatter after `forward_multi_seq` cannot land on the wrong request. Per-request sampler state already lived in the row (not a global RNG) and is documented as deliberate. Tests: `removing_a_row_never_reassigns_another_rows_state` (with the positional `swap_remove` failure spelled out beside it), `flush_replies_on_each_rows_own_channel`, and `a_row_leaving_mid_batch_does_not_shift_its_neighbours_output` -- three concurrent rows where one trips a stop mid-batch, so the batch is narrower than the row table on that tick; confirmed to FAIL when the scatter is patched to address rows by batch position. NOT DONE: no per-row logits processor / constrained-decoding state exists yet (json_object is a whole-request flag), so the specific oMLX json_schema collapse has no frink analogue to regress -- the table is simply built so it cannot happen."
    status: completed
  - id: sched-deferred-abort
    content: "LANDED, and it closes a gap the `cancel` module previously documented as unfixable ('a continuous-batching request ... is not covered'). `CancelToken::on_cancel` (new) hands the scheduler a callback that does exactly one thing -- push an `AbortId` into `AbortInbox` -- and `apply_aborts` drains that set at the TOP of a worker tick, before any forward pass, and does the batch mutation there. The hook fires immediately if the token is already cancelled, so a cancel racing its own submission is not lost; ids that name a job not yet through the channel are CARRIED across ticks rather than dropped, for the same reason. A request is stopped wherever it is: still queued (queue slot released, nothing prefilled), mid-prefill (blocks released, remaining chunks never run), or decoding (marked `FinishReason::Cancelled` and left to leave through `flush_finished`, so there is exactly ONE path that releases blocks and replies -- a second removal path is a second place to forget one of those). Partial output survives, since a cancelled generation has tokens and discarding them serves nobody. ON THE GPU SYNC: frink-metal already calls `waitUntilCompleted` per dispatch, so no command buffer outlives a `forward_multi_seq` call and the step boundary really is a point where nothing is in flight -- there is no separate sync to insert, and none was faked. What the deferral buys is that the mutation happens on the thread that owns the step at all, rather than from whichever HTTP handler received the cancel while `std::mem::take` has a row's caches lifted out into the batch. Tests confirmed to FAIL when broken: `a_decoding_request_stops_when_it_is_cancelled` (finishes with Length after all 4000 tokens without the drain), `a_cancel_that_arrives_before_its_job_is_not_lost` (fails when unmatched ids are dropped), `a_cancelled_row_leaves_through_the_one_exit_every_row_uses` (fails when `mark_cancelled` removes instead of marking -- catches the double-reply), `a_cancelled_prefill_is_abandoned_and_gives_its_blocks_back` (fails when the release is dropped), plus `cancelling_one_request_leaves_its_neighbours_running`. NOT DONE: no cancellation inside a prefill CHUNK -- the check is between chunks, so a request is charged at most `FRINK_CB_PREFILL_CHUNK` tokens after the cancel, which is a bounded cost and the reason chunking landed first; and `AbortInbox` carries an unmatched id indefinitely (bounded by the number of cancels for jobs whose send failed, which only happens when the worker is already dead)."
    status: completed
  - id: sched-block-admission
    content: "LANDED. `BlockBudget` in batch_scheduler.rs is an integer ledger of KV blocks (`FRINK_CB_KV_BLOCKS` total, `FRINK_CB_KV_BLOCK_SIZE` positions per block, default 256); a request needs `ceil((prompt + max_tokens) / block_size)` blocks, reserved at admission for its whole lifetime and released in `Rows::flush_finished`. No byte watermark, no hysteresis band, no pressure enforcer -- frink reads its KV layout from the GGUF header, so 'will this fit' has an exact integer answer. The worker now keeps its own `waiting: VecDeque<Job>` and `QueueGate` releases a slot at ADMISSION rather than when the job leaves the channel, so the queue cap still counts a job parked for capacity. Two rejections, deliberately different: `blocks_needed > blocks_total` is `DecodeError::KvBudgetExceeded` -> 400 with `retry_after_secs() == None`, refused before a queue slot is reserved (an empty server would refuse it too, so 503 would be a lie); momentary pressure just waits. Counters split accordingly -- `kv_rejected_too_large` vs `queue_rejected`, plus `kv_blocks_total/free/peak/size` on `/metrics`. Admission is strict FIFO: a head job that does not fit stops the line rather than being skipped, because skip-ahead starves large requests indefinitely behind a stream of small ones. Tests confirmed to FAIL against the broken version: `concurrent_requests_never_hold_more_blocks_than_the_budget` (6x2 blocks against a 4-block budget; peak is measured from the ROWS, not from the ledger, because a ledger-derived peak cannot exceed the budget however broken admission is and would report the invariant instead of checking it), `every_admitted_request_gives_its_blocks_back`, `a_job_rejected_at_validation_gives_its_blocks_back`, `a_head_job_that_does_not_fit_holds_the_line` (fails under both a no-op reserve and a skip-ahead policy). DELIBERATELY NOT DONE: the budget is in blocks of POSITIONS, not bytes -- it is not yet derived from `frink-models::kv_budget`'s per-token KV bytes and a device budget, so an operator sets `FRINK_CB_KV_BLOCKS` by hand rather than the server deriving it from the model (that join is `mem-preload-kv-budget`); no preemption or eviction, so an admitted request keeps its blocks until it finishes; and the reservation is worst-case (`prompt + max_tokens`) rather than growing with the sequence, which over-charges a request that stops early -- correct, but conservative."
    status: completed
  - id: sched-output-mailbox
    content: "LANDED, but NOT as written -- the plan's premise is wrong and the fix is a different one, so read this rather than the plan's 1e bullet. The SSE channel is NOT unbounded: it is a tokio::sync::mpsc::channel(64), bounded since it was written, and a send that fails because the receiver was dropped already flips the request's cancel token. So a disconnected consumer was handled and a slow one already had real backpressure -- there was no unbounded memory growth to stop, and a single-slot COALESCING mailbox would have been actively wrong here: every chunk of a token stream is content, not a redundant state update that a newer one supersedes. What WAS broken is the second hazard the previous run named. `Sender::blocking_send` parks the calling thread until there is room, and a receiver that is neither draining nor dropped -- a zero TCP window, a paused client, a device that left coverage without the socket noticing -- provides neither, ever. Generation runs on spawn_blocking, so the parked send permanently pins a blocking-pool thread, the Arc<Model> that keeps /admin/models/unload from freeing the weights, and the request's CancelGuard (so the id never leaves the registry). Once per stalled client, with nothing able to unstick it. `frink-server::sse::send_or_orphan` gives the send a deadline -- 30s default, FRINK_SSE_ORPHAN_TIMEOUT_MS, 0 restores the old block-forever behaviour -- and reports a timeout as the SAME Err a disconnect already produces, so the existing cancel path handles it and there is one stop path rather than two. Every blocking_send in the SSE handler goes through it, terminal frames included, so the thread always terminates. Tests confirmed to FAIL when broken: `a_receiver_that_never_reads_does_not_park_the_sender_forever` (a receiver held and never polled; against a plain blocking_send the send never returns and the test's own five-second guard reports Elapsed), `a_dropped_receiver_is_a_disconnect_and_is_reported_at_once` (fails when Closed is mapped to Orphaned -- the two mean the same thing to the caller but not to whoever reads the log, and a closed channel must not wait out the deadline), and `a_draining_receiver_gets_every_event_in_order`, which is the guard against fixing the hang by dropping events. NOT DONE, deliberately: no mailbox and no coalescing, for the reason above; the deadline is a fixed duration rather than derived from a measured token rate; and under continuous batching the emit closure is not used at all (overlap = !tools_active && batcher.is_none()), so the batched path streams nothing through this and is unaffected either way."
    status: completed
  - id: sched-stop-buffering
    content: "LANDED as `frink-server::stop` (`StopMatcher`, `StopStep`, `resolve_stop_tokens`), used by BOTH `generate::sample_until_stop` and the batch scheduler, so a row in a batch and a row on its own cannot disagree about where an answer ends. CORRECTION TO THE PLAN'S PREMISE: frink did NOT have only a text-level scan -- it already withheld `longest_stop - 1` bytes on both paths, which is safe. What it lacked was (a) the token-level layer entirely and (b) precision in the text layer. Layer 1: a stop string that encodes to exactly one token is matched on the ID, before detokenization, and is treated exactly like EOS -- not emitted, not counted in usage. That answers a question the text layer provably cannot: `a_stop_token_that_renders_as_nothing_is_still_a_stop` shows a control token rendering to \"\" that the text scan can never see. Multi-token stop strings are deliberately NOT given a token-level form: a multi-token encoding is not a statement about how the model will emit that text. Layer 2: `partial_suffix_len` withholds the longest suffix that is a PROPER prefix of some stop, instead of a fixed byte count, so with `stop: [\"<|im_end|>\"]` ordinary text is no longer permanently 9 bytes behind the model; byte-wise comparison with a `floor_char_boundary` guard, so a split can never land mid-character. Stop-token ids are resolved in exactly one place (`run_generation_emit`, the only layer holding both the request's stop strings and the model's tokenizer) and ride on `GenerationParams`, so the batched and private paths cannot drift. Tests confirmed to FAIL when broken: `a_token_level_stop_ends_generation_and_never_reaches_the_output` and `a_token_level_stop_ends_a_batched_row` (both run to the token limit without the id check), `a_disproved_partial_is_released_by_the_token_that_disproves_it` plus six `stop::` unit tests (all fail under the fixed hold-back). NOT DONE: no incremental/streaming matcher state across the disk-cached prefix, no regex or token-sequence stops, and empty stop strings are dropped rather than rejected at the API edge -- an empty stop would otherwise match at position 0 and end every answer at its first token."
    status: completed
  - id: sched-queue-cap
    content: "LANDED. `QueueGate` in batch_scheduler.rs bounds jobs WAITING for admission (in-flight sequences remain `FRINK_CB_MAX_SEQS`'s business); default 512 via `FRINK_CB_MAX_QUEUE`. Over the cap, `generate` returns `DecodeError::QueueFull { queued, cap }` -> 503, with `retry_after_seconds` in the body and a `Retry-After` header stamped by `limits::retry_after`, a layer that marks any 503 lacking one (a 503 is by definition temporary, so this also fixes the pre-existing KV-pool and no-model-loaded 503s). Reservation is a CAS loop, not check-then-act; `queue_gate_never_exceeds_its_cap_under_concurrent_submitters` (32 threads x 64 rounds) was confirmed to FAIL with a load-then-fetch_add gate. `queue_depth` / `queue_rejected` on `/metrics`. DELIBERATELY NOT DONE: the Retry-After value is a fixed 1s, not queue depth divided by measured throughput -- the honest computation needs a drain-rate estimate this scheduler does not keep yet; the cap only guards the continuous-batching path, since the private `generate` path has no queue to cap; and the cap counts jobs, not tokens or bytes, so one huge prompt still counts as one."
    status: completed
  - id: mem-preload-kv-budget
    content: "LANDED. The CLI half was already there and was verified rather than reimplemented: `frink_models::kv_budget` plus frink-cli/src/run.rs do the pre-load arithmetic -- `--ctx-size auto`, a pre-load `KvBudget::check`, a `Ceiling::DeviceMemory` refusal. The OPEN HALF this todo named -- frink-server does not price its model at load, so it admits on configured ceilings instead of measured ones -- is closed by `frink-server::budget` (arithmetic in `mem-ctx-auto`, refusal in `mem-typed-rejection`). `price_batcher_config` runs once per load, at startup and on /admin/models/load alike, and the swapped-in model is priced against ITS OWN path rather than FRINK_MODEL_PATH, so a model swap cannot leave the new model admitting on the old one's arithmetic. NOT DONE: the server does not REFUSE TO LOAD an overcommitting checkpoint the way `--strict-budget` does on the CLI -- it loads and admits against the derived ceiling, or against none when the fit is zero, for the reason spelled out in `budget`'s module doc (a fit of zero is an estimate saying the model should not have loaded, and it did)."
    status: completed
  - id: mem-ctx-auto
    content: "LANDED for frink-server; the CLI half was already there and is left alone. `frink_models::kv_budget::KvBudget::max_context` (closed form: (budget - weights - headroom) / (marginal_per_token * concurrency), sliding layers subtracted out of the divisor and added back as a constant) plus `--ctx-size auto` in frink-cli/src/run.rs already existed and were VERIFIED, not reimplemented. What was missing was the server pricing its model at load, which is what `frink-server::budget` now does: `price_gguf` probes the device budget and builds a `ResidencyReport` from the checkpoint's own header, and `derive_limits` turns the priced fit into the two ceilings the server admits on -- `max_context` (positions any one request may hold) and `kv_blocks` (the whole-server ledger, FLOORED to whole blocks, because a partial block is not a block to promise). `apply_derived` is the precedence rule as a pure function: a derived number may only ever occupy an EMPTY slot, never override an operator's FRINK_CB_MAX_CONTEXT / FRINK_CB_KV_BLOCKS. Three refusals to derive, all deliberate: an unknown device budget derives nothing (the same rule the CLI's `resolve_ctx_size` follows -- refusing on the strength of a number we do not have is worse than not refusing); a header the planner cannot read derives nothing (MLA/Gemma4/GLM carry their own hparams and never build a `ModelConfig`); and a fit of ZERO tokens derives nothing rather than installing a ceiling of zero, because that is not a ceiling, it is a refusal to serve a model that already loaded, decided by an estimate over mmap'd pages which `kv_budget`'s own module doc calls an upper bound rather than a measurement. That case logs loudly and names FRINK_DEVICE_BUDGET_BYTES. Same reasoning one step down: a fit under one block leaves `kv_blocks` absent rather than Some(0), which would refuse every request. Tests confirmed to FAIL when broken: `a_roomy_machine_is_still_capped_at_the_models_trained_context` (fails when the cap is usize::MAX instead of gguf_ctx), `a_partial_block_is_floored_away_rather_than_promised` (fails under div_ceil), `a_model_that_leaves_no_room_derives_no_ceiling_at_all` (fails when the zero-fit guard is dropped), `a_configured_ceiling_is_never_overridden_by_a_derived_one` + `an_absent_ceiling_is_filled_and_the_two_slots_are_independent` + `a_fit_smaller_than_one_block_leaves_the_ledger_absent` (all three fail when `apply_derived` assigns unconditionally), plus `the_derived_context_is_the_room_left_after_weights_divided_by_the_per_token_cost` against a hand-computed 64 bytes/token. NOT DONE, deliberately: no bisect-and-verify-with-a-real-prefill variant -- the closed form is what landed, and the plan itself calls the bisect the more honest of the two, so this is the cheaper half and is labelled as such; the KV width is budgeted as f32 (what `frink_core::cache::KvCache` really keeps on both paths) even under Metal, which over-charges KV and so UNDER-states the fitting context rather than over-stating it; and concurrency is priced at 1, so `max_context` says what ONE request may hold and the block ledger is what makes N requests share it."
    status: completed
  - id: mem-typed-rejection
    content: "OPEN HALF NOW LANDED. The half that existed (3412df9) covered the continuous-batching path only, and continuous batching is opt-in AND switched off outright whenever a KV pool or prefix cache is configured -- so the common deployment had no context ceiling at all and answered an oversized request with a 503. `frink-server::budget::ContextCeiling` is now the whole ceiling in ONE object: the position limit, the `KvShape` that prices a refusal in real bytes, and the rejection counter. `activate_loaded_model` builds exactly one per loaded model and hands the same `Arc` to the batch scheduler's `BlockBudget` and to the private `generate` path, so the two cannot drift the way two copies of the same arithmetic would -- the discipline `crate::stop` already applies to stop sequences. `BlockBudget` lost its own max_context/shape/rejected_context_length fields and delegates, so /metrics' `kv_rejected_context_length` now counts refusals from BOTH paths. The check runs in `generate` before any KV is acquired and before a single forward pass. Second half, a bug in its own right: a request whose worst case exceeds the WHOLE KV pool used to sleep through FRINK_KV_POOL_QUEUE_TIMEOUT_MS and leave with a 503 + Retry-After for blocks that do not exist. `pool_immovable_refusal` answers it up front with a typed 400 naming device_memory_budget_exceeded, priced in bytes; momentary exhaustion (a big-enough pool currently held) keeps its honest 503. And the ceilings are no longer merely CONFIGURED -- see `mem-ctx-auto`: unset means derived from weights + per-token KV against the device budget. Tests confirmed to FAIL when broken: `a_request_past_the_context_ceiling_is_refused_before_any_kv_is_acquired` (fails when the check is removed; also asserts the pool is untouched and no chunk is emitted), `a_request_inside_the_ceiling_is_admitted_unchanged` (the guard against a ceiling that refuses everything passing the first test), `a_request_too_big_for_the_whole_pool_is_a_400_not_a_retryable_503` and `generate_fails_at_admission_not_mid_decode_when_the_pool_cannot_cover_the_worst_case` and `generate_rejects_the_request_without_leaking_blocks_when_the_pool_is_too_small` (all fail without `pool_immovable_refusal`), plus the PRE-EXISTING `a_request_longer_than_the_context_ceiling_names_that_ceiling` and `the_context_ceiling_is_reported_before_the_device_ceiling`, which now fail when `ContextCeiling::refusal` is broken -- proving the batched path really does run on the shared object. THREE EXISTING TESTS WERE CHANGED, stated plainly: they asserted KvPoolExhausted (503) for pools that were PERMANENTLY too small, which was the misleading answer, so they now assert the typed 400; the two that were really about admission timing were rewritten to exhaust a pool that is genuinely big enough, which is the only situation in which a 503 is a true statement. NOT DONE: the engine paths (Kimi/MLA/Gemma4/GLM-5.2) get no ceiling -- they never build a `ModelConfig`, so there is no `KvShape` to price them with, and inventing one would be worse than admitting there is none; `run_generation_emit` therefore applies the ceiling only around `generate`, not around `generate_engine`."
    status: completed
isProject: false
---

# serving — tiered KV, prefill/decode fairness, admission

> Written **2026-08-13** from a read-only study of **oMLX** under
> `.scratch/omlx` (Apache-2.0, Python + MLX). Its paged/SSD prefix cache
> (158 + 122 tests) and its time-debt scheduler are the two subsystems
> worth learning from; almost everything else in it is compensating for
> MLX's allocator.
>
> Scope note: this is *serving* behaviour. It does not close a single
> `benchmarks/RESULTS.md` row — those belong to
> [`llama-cpp-parity-push.md`](llama-cpp-parity-push.md). Keep the two
> tracks separate so a serving change is never credited with an engine
> speedup.

## The correction that makes this cheap

**oMLX's "paged KV cache" is not paged attention.** `CacheBlock` holds no
tensor data at all — it is `block_id`, `ref_count`, `block_hash`,
free-list pointers, `token_count`. Attention still runs on a stock
*contiguous* KV cache, rebuilt by concatenating per-block slices on
restore.

So blocks are a **storage, dedup and eviction granularity for a prefix
cache**, not a memory layout for the attention kernel. That is exactly why
it ports: it sits on top of a contiguous per-sequence KV, which is what
frink has today. **No paged-attention kernel work is required for any of
this.**

## What frink already has

- `batch_scheduler.rs` — continuous batching of *decode*, opt-in via
  `FRINK_CONTINUOUS_BATCHING=1`, with a `FRINK_CB_MAX_SEQS` cap.
  Prefill **is** chunked now (`sched-chunked-prefill`, landed): one
  bounded `PrefillState::step_chunk` plus one batched decode step per
  tick. Each chunk is still a per-token `forward_token` loop, so this
  bought fairness, not throughput.
- An in-memory `PrefixCache`, a response cache, `session.rs`, a KV pool.
- Exact knowledge of its own KV layout from the GGUF header — the thing
  oMLX lacks and works around everywhere.

Worth stating plainly: on the batching axis frink is **not behind**
oMLX. oMLX's continuous batching is decode-only too (`prefill_batch_size`
is hardcoded to 1), it has **no preemption** (`RequestStatus.PREEMPTED` is
defined and never assigned), **no priority policy** (the enum exists and
is never read), and no per-step token budget (`max_num_batched_tokens` is
unreferenced dead config). What it has that frink does not is chunked
prefill with a fairness mechanism, and a durable cache tier.

## Phase 1 — chunked prefill and fairness

### 1a. Resumable prefill (prerequisite)

A per-request state machine holding `cache`, `tokens_remaining`,
`tokens_processed`, plus boundary bookkeeping, with
`fn step_chunk(&mut self) -> Result<bool /* done */>`. This converts an
unbounded prefill into a bounded unit of work, which is what makes a
single-threaded step loop possible at all.

### 1b. Time-debt interleaving

The load-bearing insight, and it is backend-agnostic: **a GPU cannot
preempt a running kernel, so chunk duration *is* the scheduling
quantum.** Not token count — duration.

1. Contended chunk size is derived in **milliseconds**, converted to
   tokens via measured prefill tok/s, floored and capped.
2. Each chunk accrues `chunk_seconds × share` of debt.
3. Decode wall-time repays it.
4. The gate blocks the next chunk until the debt clears.

oMLX's constants (0.5 fair share, 500 ms stall target) were measured on an
M3 Ultra and are a reasonable starting point, not a derivation. One flaw
not to inherit: it advances *all* pending chunked prefills before decode
runs, so N concurrent long prompts stall decode for N chunks per step.
Cap the per-step prefill work.

### 1c. Keyed row state

Per-row state must be a `HashMap<Uid, RowState>`, never a `Vec` parallel
to the batch. This is a correctness bug class, not a style preference:
oMLX ships a monkey-patched batch step that rebuilds positional arrays
from a registry on *every* step, because a plain chat request joining a
batch that served a `json_schema` request collapsed every processor slot
to `None` and silently applied the wrong constraints to the wrong row.
frink owns its whole stack and can simply build it right.

Related: per-request RNG state lives in the row struct. oMLX seeds a
*global* RNG per request and its own comments admit this breaks under
concurrency.

### 1d. Deferred abort

Cancellation enqueues an id into a shared set; the inference thread drains
it at a step boundary and performs the batch mutation, **syncing the GPU
first** because slicing KV against in-flight command buffers corrupts
them. frink-metal has the identical hazard. In Rust this is an mpsc
drain at the top of the step — more natural than the Python original.

### 1e. Output mailbox and stop sequences

- **Single-slot coalescing mailbox** per request with an explicit merge,
  not an unbounded channel: a slow or disconnected SSE consumer must not
  grow memory. Pair it with an orphan reap for streams abandoned rather
  than closed.
- **Stop sequences need two layers**, and frink likely has only the
  second: a token-level matcher, *plus* output buffering that withholds
  any output suffix which is a prefix of a stop string so a partial match
  never reaches the wire. Without the buffering layer, streaming stop
  sequences leak fragments.
- **Queue cap → 503 + Retry-After.** Trivial; prevents retry storms from
  growing memory without bound.

## Phase 2 — admission that answers before it accepts

oMLX's structural weakness, stated plainly so frink does not copy it: it
admits a model on **weights-only** cost (`sum(*.safetensors) × 1.05`, no
KV, no activations) and then discovers the KV cost per request forever
after. Its prefill-eviction callback, background pressure enforcer,
adaptive chunk throttle, abort ladder and stall-timeout killer are all
compensating for that one under-charge.

frink does not need any of it:

```
weights + n_ctx × per_token_kv + activation_headroom  ≤  device_budget
per_token_kv = n_layers × n_kv_heads × head_dim × bytes × 2
```

All four terms are exact from the GGUF header at `inspect-plan` time.
Variants worth carrying: the sliding-window cap
(`min(tokens, window + chunk − 1)`) and the MLA form
(`kv_lora_rank + rope_dim` per layer).

From that follow, cheaply:

- **`--ctx auto`** — closed-form `(budget − weights) // per_token`, or
  bisect the real admission predicate and verify with an actual prefill.
  The bisect is the more honest of the two and oMLX implements it, but
  only as an opt-in admin job that unloads every model first.
- **Admission on an integer block count** once the block cache exists:
  `blocks_needed ≤ blocks_free`. Strictly better than a byte watermark,
  and available to frink precisely because its allocator is not opaque.
- **Typed rejection**: a 400 naming `estimated_bytes`, `limit_bytes` and
  *which* ceiling binds — with split counters for "prompt too big" versus
  "system under pressure", so an operator is not sent to the wrong knob.

## Phase 3 — the disk tier

Design, with the pieces that matter:

- **Block hashing**: parent-chained SHA-256 over
  `(model, parent_hash, token_ids, extra_keys)`, root-seeded.
  `extra_keys` is the salt slot for LoRA/multimodal identity. Sampling
  params are correctly *not* part of the key — KV is sampling-independent.
- **`cache_signature`**, and this is the one to get right: stamp
  compatibility **from the block's own payload, never from the manager's
  expectation**, and reject blocks with no recorded depth rather than
  trusting them. oMLX states the rule as "a signature must never vouch for
  a width the payload does not have". It is the difference between a
  persistent cache and silent corruption after a config change.
- **On disk**: one file per block, sharded into subdirectories by hash
  prefix, per-layer flattened keys, dtype passed through unchanged, no
  compression, an explicit format version with a readable-set, and
  metadata carrying block hash, token count, layer count, block size and
  the signature.
- **Publish atomically** with temp-file + rename, then re-check the block
  was not evicted mid-write.
- **Write ordering invariant**: buffer → index → queue. A concurrent
  reader must never see an index hit for a block with no file and no
  buffer.
- **Backpressure**: bounded queue; on full, write **inline on the calling
  thread** rather than drop. Count the fallbacks.
- **Disk budget** clamped against real free space with a TTL'd stat,
  invalidated on ENOSPC, so eviction fires before the filesystem does.
- **Block size must be a multiple of the sliding-window size** for SWA
  models. Non-obvious and silent when wrong.

Two things oMLX got wrong that frink should not inherit:

1. **The read is synchronous on the request path with no prefetch** —
   they were blocked by a Metal deadlock that frink does not have. Build
   it async and prefetched from the start; retrofitting it there was
   impossible.
2. **Its RAM tier is off by default**, so the advertised "hot tier fills,
   then spills to SSD" never happens out of the box. If frink ships a RAM
   tier, ship it on, or do not claim it.

Also: measure the block read and block write. oMLX's cache stats contain
**zero time-valued fields** — hit rate is observable, "SSD hit versus
recompute" latency is not, which makes the tier impossible to tune.

## Explicitly not ported

- **Rotated low-bit KV quantization.** Metal kernels generated as Python
  f-strings and JIT-compiled per `(key_bits, val_bits, dim)`; NumPy QR and
  Lloyd–Max at codec-build time. Only the *format* idea is portable: store
  an fp16 norm plus rotated packed indices and rebuild the codec
  deterministically from `(dim, bits, seed)` rather than serializing it.
  Note its asymmetric K/V precision is narrower than advertised — K and V
  differ only at fractional bit depths; at integer bits they are identical.
- **Byte-watermark memory accounting**, `phys_footprint` sampling, wired
  limits and jetsam avoidance, the macOS `free + inactive + active × ratio`
  dynamic ceiling, allocator-cache hygiene, lazy-array eval ceremony. All
  MLX/Apple-UMA artifacts.
- **The unload settle barrier**, at least unchanged: it polls the
  allocator until freed bytes come back. Against frink's **mmap'd**
  quantized weights that would report a false timeout — freeing address
  space is not freeing RSS. This is the single biggest conceptual mismatch
  between the two engines.
- Monkey-patching as an integration mechanism. frink needs trait dispatch.

## Sequencing

1. **1a + 1b** (chunked prefill + time-debt). Largest behavioural win,
   and the prerequisite for the rest.
2. **1c–1e** (keyed rows, deferred abort, mailbox, stop buffering, queue
   cap). Correctness under concurrency; cheap.
3. **Phase 2** (pre-load KV budget, `--ctx auto`, typed rejection).
   Independent of 1 and 3, and the highest user-visible value per line.
4. **Phase 3** (disk tier). Largest, and worth doing only after the block
   hashing and signature discipline from Phase 2 exist.
