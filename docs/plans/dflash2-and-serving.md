# Plan: DFlash2 speculative decoding and the serving features around it, on every backend

## Context

The feature set: a trained DFlash2 block-diffusion drafter paired with each
target model, memory and context sized automatically, cached prefixes reused,
concurrent requests batched, plus vision, tool calling and a chat page. Engines
that offer it today are Metal only. The goal is to bring it to Frink on CPU,
Metal and CUDA (Vulkan where its kernels reach).

What research found:

- **The drafter has an oracle.** llama.cpp merged DFlash2 (PR 27342, 2026-08-27).
  The pinned commit `604cc430` has `src/models/dflash.cpp`, the `dflash` GGUF
  arch, and `--spec-type draft-dflash` in `common/speculative.cpp`. GGUF drafts
  are published on Hugging Face (`z-lab/Qwen3.8-27B-DFlash2-GGUF` and others,
  Apache-2.0). This clears the GO/NO-GO in
  `docs/plans/on-hold/dflash-speculative-decoding.md`.
- **Frink already has the verify half.** `crates/frink-models/src/speculative.rs`
  (`Drafter`, `DraftBlock`, `DraftDist`, `accept_or_resample`,
  `speculative_decode_observed`), `draft_model.rs`, CLI flags `-md/--draft-max/
  --draft-p-min`, and server verify-by-agreement in
  `crates/frink-server/src/sampling_loop.rs` (`DecodeEngine`, `verify_block`).
- **Gaps for DFlash2:** no intermediate-layer hidden capture (only final layer
  via `forward_batch_with_hidden`), no `dflash` engine (enum-only in
  `capability.rs`), the speculative loop asserts on recurrent targets (the Qwen
  targets with drafts are GDN hybrids), no draft model on the server, no
  drafting in the continuous batcher, paged KV cannot roll back, device KV
  rollback unverified.
- **Around it:** tool calling and the OpenAI/Anthropic endpoints exist. Prefix
  caching, batching and memory budgeting exist but are opt-in, mutually
  exclusive in places, or warn instead of deciding. Vision is a refusal stub.
  `/` is a 404. `DiskKvStore` exists with no consumer.

Decisions taken with the user: DFlash2 end to end first; chat page is Frink
Studio (`ui/`) embedded in the server; vision is the Qwen-family mmproj first,
every other projector refuses by name.

Rules that shape the work: llama.cpp is the oracle (ADR 0001), flags mean what
they mean there (0005), refuse rather than approximate (0002), one file per
architecture (0003), parameterise never copy (0004), goldens come from libllama
(0011), agents never benchmark (0006). No competitor names in any repo file,
commit or PR; describe features only.

Deliberately not ported: the adaptive "verify budget" (it makes greedy output
vary run to run, which breaks "same tokens"), Neural Engine prefill, MLX
targets, the desktop app, agent launcher subcommands, per-shape precompiled
Metal kernels. Kernel work stays under the existing coverage matrix (#405).

## Phase 0: paperwork (done)

- Tracking issue #600. Speculation in the batch stays #533, the disk
  tier #536, vision #407.
- `north-star.md` lists trained drafters as priority 10 instead of
  "Not now"; `dflash-speculative-decoding.md` is back from `on-hold/`
  with its GO/NO-GO resolved (GO).
- The `speculative.rs` module doc names every drafter, not only prompt
  lookup.

## Phase 1: DFlash2 end to end (first deliverable)

Each step is its own PR, CPU first, token parity before any backend work.

1. **Read upstream.** `src/models/dflash.cpp` (graph, key reads, the
   `set_swa_pattern(0)` and hyper-connection branches),
   `common/speculative.cpp:983-1400` (`common_speculative_impl_draft_dflash`:
   target-layer extraction, KV injection, `[id_last, mask x (block-1)]` input,
   lattice walk, probabilistic mode and its seed), the converter in
   `conversion/qwen.py`. Note which keys are dead. DSpark and LiliCorr branches
   in the same file refuse by name.
2. **Target hidden taps.** Add an optional capture list to the batch forward in
   `crates/frink-models/src/decoder/entry.rs` / `decoder/batch.rs`
   (generalising `forward_batch_with_hidden`, not a copy): return the residual
   output of named layers per position. Mean over streams for hyper-connected
   targets only if upstream does so; otherwise refuse. Test with a synthetic
   fixture whose layers are distinguishable.
3. **`dflash` engine.** New file `crates/frink-models/src/models/dflash.rs`
   (or the crate's existing per-arch location), loading the draft GGUF:
   context projection (`fc`), layers with the two-tap grouped dynamic
   convolution (`blk.N.attn_conv_base/proj`, `ffn_conv_base/proj`), non-causal
   sliding attention over injected context KV, the candidate selector
   (`selector_predecessor`, `selector_successor`, `selector_hidden`, rank and
   top-k from metadata), mask token from `tokenizer.ggml.mask_token_id`, shared
   embedding/LM head resolved from the target when the draft omits them, `d2t`
   map when present. All matmuls through `WeightMatrix` so backends come for
   free. Move `dflash` from `EnumOnly` to a draft-head capability row (it still
   refuses as a standalone model).
4. **`DFlashDrafter: Drafter`.** Implements `propose` from the taps: greedy
   lattice walk at temp 0; probabilistic walk returning `q` as `DraftDist` for
   `accept_or_resample` when sampling. Draft-side KV keyed off target position,
   truncated on rejection. Compatibility check target/draft (hidden size, vocab
   by token-string hash, layer ids in range) refuses with the mismatch named;
   this also closes the open step 2 of `archive/speculative-decoding.md`.
5. **Recurrent rollback.** The Qwen hybrid targets have GDN layers with
   `rollback_limit() == 0`. Follow what llama.cpp does at the pin for
   speculative decode on recurrent memory (checkpoint and restore of the
   recurrent state around the verify batch); implement as snapshot/restore in
   `frink-core/src/cache.rs` next to `truncate`, used by
   `speculative_decode_observed` in place of the assert. If upstream refuses
   that combination, Frink refuses too and Phase 1 targets a pure-attention
   family first.
6. **CLI.** `--spec-type draft-dflash` (also auto-detected from the draft's
   arch, as upstream), `--spec-draft-n-max/-n-min`, `-hfd` only if `-hf`
   already exists. Flags accepted but not honoured are errors. Update
   `crates/frink-cli/tests/llama_cpp_flags.txt` and `docs/CLI.md`.
7. **Server.** `-md` / `FRINK_DRAFT_MODEL_PATH` on `frink-server`;
   `ServerEngine` in `generate.rs` holds `Option<Box<dyn Drafter>>` instead of
   the concrete prompt-lookup type; verification stays verify-by-agreement
   (`verify_block`), so grammars and penalties keep working; drop the stale
   `UNWIRED` attributes. Fill per-position acceptance in usage and report
   acceptance length per verification step.
8. **Device rollback.** Tests (ignored, hardware) that a target on Metal and on
   CUDA truncates its device KV mirror correctly after a rejected block
   (`KvCache::metal_attn`, `cuda_attn`, `decoder/metal_kv.rs`); fix or refuse
   by name. Drafter on device follows once its KV keeps a host copy or the
   mirror truncates.
9. **Batcher.** Drafting inside `serving/batch/worker.rs`: one drafter pass per
   lane, one batched verify through `forward_multi_seq_kv`, per-lane accept and
   truncate. Paged KV needs block-tail truncation in
   `frink-core/src/cache/paged.rs` and the radix tree must not publish
   unverified rows; drafter KV is salted in the prefix signature. Until this
   lands, speculation with continuous batching refuses by name.

Evidence for Phase 1: a synthetic draft fixture plus a libllama golden of the
drafter's lattice for a fixed target-hidden input (extend the dumper in
`tools/build_llama_logits.sh`); `--temp 0` output with a DFlash2 draft is
token-identical to plain decode (CI test, existing pattern at
`speculative.rs:895`); sabotage each new test once.

## Phase 2: serving defaults on every backend

- **Automatic memory planning.** llama.cpp's `--fit`, `--fit-target`,
  `--fit-ctx`, `--fit-print` on `frink run` and `frink-server`, built on
  `device_budget.rs`, `kv_budget.rs`, `budget::price_gguf`/`derive_limits`.
  Price target + draft + mmproj + N slots (today it prices one request), add a
  Vulkan arm to `BudgetBackend`, and refuse a model that cannot fit instead of
  serving with a warning. Idle release maps to the existing `/sleep` path.
- **Prefix reuse by default.** Make paged KV + radix the default store when
  batching is on, derive its size from the fit, honour `cache_prompt` on the
  chat endpoints, add `--cache-reuse` and `-cram` or list them as refused.
  Recurrent models: cache the recurrent state at block boundaries (reuses the
  Phase 1 snapshot).
- **Disk tier.** Wire `frink_core::kv_disk::DiskKvStore` as the cold tier
  behind the radix tree (`--slot-save-path` family), or delete it (ADR 0008).
- **Batching by default off Metal.** Turn continuous batching on by default for
  CPU and CUDA once a hardware receipt exists; CUDA resident KV is single
  sequence today, so this needs multi-sequence resident KV or stays opt-in with
  the reason stated.
- **Endpoint gaps.** `stream` on `/v1/completions` (today silently ignored,
  which is an ADR 0005 bug: implement or refuse now), `/props`, `GET /slots`,
  `/apply-template`.

## Phase 3: vision, Qwen-family mmproj (#407)

- Read `tools/mtmd` at the pin for the Qwen projector types; load `--mmproj`
  (`-mm`, `--mmproj-auto`) in place of `mmproj.rs`'s warning and
  `vl_engine.rs`'s stub; other projector types refuse by name.
- Encoder in its own file, matmuls through `WeightMatrix`; image preprocessing
  matched to mtmd; embeddings spliced into the prompt at the template's image
  marker.
- Server: accept `image_url` (data URLs and base64) in chat completions and
  Responses, `image` blocks in Anthropic Messages; remove the refusals at
  `lib.rs:1405` and `anthropic.rs:570`. PDFs stay refused.
- Golden: libllama image embeddings for a fixed image, then first-token logits.

## Phase 4: chat page (Frink Studio embedded)

- Cargo feature `webui` on `frink-server`; `build.rs` embeds `ui/dist` when
  present; routes for `/` and assets, outside auth like `/health` only for
  static files. Flags `--webui` / `--no-webui` as upstream. Without the feature
  `/` stays 404 and `--webui` is an error.
- Release workflow builds `ui/` before the server; Studio's API base defaults
  to same origin. Update `docs/API.md` and `FEATURES.md` ("/ stays a 404").

## Hardware and measurement

No rentals during code. Each PR lists its ignored hardware tests; one batched
vast.ai rental (CPU, CUDA) after Phase 1 and again after Phase 2, asked for
first; Metal checks on the M2 Pro. Speed claims come only from `frink bench`
with acceptance length reported, run by the user, never by an agent.

## Verification

- Every PR: `cargo fmt --all`; clippy `-D warnings` debug and `--release`;
  `cargo test --workspace`; `scripts/oracle.sh` for anything that changes
  tokens.
- Phase 1 end to end: `frink -m qwen-target.gguf -md qwen-dflash2.gguf -p ...
  --temp 0 --no-cnv -n 256` equals the same command without `-md`, on
  `--device cpu`, `-ngl 99` (Metal) and CUDA; `frink parity` unchanged for the
  target; the same llama.cpp command line (`llama-cli -m ... -md ...
  --spec-type draft-dflash`) is accepted unchanged.
- Server: `FRINK_MODEL_PATH=... FRINK_DRAFT_MODEL_PATH=... frink-server`, then
  a streamed chat completion with tools and one with a JSON schema; usage shows
  drafted/accepted counts; 4 concurrent requests return the same text as
  sequential at temp 0.
- Phase 2: `--fit-print` output matches `frink inspect-plan`; an oversized
  model refuses; a replayed long prompt reports cached tokens.
- Phase 3: image request against the server vs `llama-mtmd-cli` at temp 0.
- Phase 4: `curl -i localhost:8383/` returns the page with the feature, 404
  without.
