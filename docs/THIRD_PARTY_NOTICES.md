# Third-party notices

Thanks to the projects whose public designs, file formats, and CLI
conventions shaped frink's architecture: llama.cpp, candle, and
others credited below. This file records that influence, plus the
license and copyright notices required wherever real upstream source
was read or adapted.

## GGUF file format

The GGUF binary format (magic, version, key-value metadata section,
tensor descriptor table, alignment padding, block-quantized tensor
data) originates from the ggml / llama.cpp project and is publicly
documented at:

  https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

`frink-gguf` is an independent implementation written against that
public specification. ggml and llama.cpp are MIT licensed:

  Copyright (c) 2023-2026 The ggml authors / Georgi Gerganov and
  contributors (llama.cpp)

## safetensors file format

The safetensors binary format (8-byte little-endian header length, UTF-8
JSON header describing each tensor's dtype/shape/byte range, followed by
raw tensor data) is a public format originated and maintained by Hugging
Face, documented and reference-implemented at:

  https://github.com/huggingface/safetensors

`frink-safetensors` is an independent implementation written against
that public format. The header-parsing and byte-layout logic is original,
but the specific validation rules it enforces (tensor offsets must be
re-sorted by `data_offsets` before checking they're contiguous starting
at 0, non-overlapping, size-consistent with the declared dtype+shape, and
must exactly cover the rest of the file) were checked directly against
that reference implementation's `Metadata::validate` to make sure a
real, non-obvious edge case (JSON key order not matching offset order)
was handled correctly, not guessed. `huggingface/safetensors` is Apache-2.0
licensed:

  Copyright (c) 2023-2026 Hugging Face and contributors

## Q4_0 / Q8_0 / Q6_K block quantization layouts

The block-quantization conventions dequantized by `frink-quant`
(fixed-size blocks of packed low-bit values sharing one f16 scale) are
the public layout used throughout the ggml/llama.cpp ecosystem. The
dequantization routines here are written independently against that
public layout description.

## IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S codebook tables

The low-bit IQ quantization formats are defined not only by a block
layout but by fixed numeric codebook grids (`iq1s_grid`, `iq2xxs_grid`,
`iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `ksigns_iq2xs`,
`kmask_iq2xs`). These tables are part of the format itself -- a file
quantized with them cannot be decoded without these exact values, and
they cannot be independently re-derived.
`crates/frink-quant/src/iq_tables.rs` reproduces them from the
MIT-licensed ggml project's `ggml-common.h`:

  Copyright (c) 2023-2026 The ggml authors / Georgi Gerganov and
  contributors (llama.cpp)

The dequantization *code* using those tables is written independently
against ggml's published dequant semantics, and is cross-validated
against the real compiled ggml implementation -- for IQ1_S/IQ2_XXS/
IQ3_XXS through an independent Python reference, and for IQ2_XS/IQ2_S/
IQ3_S/IQ1_M by linking `ggml-quants.c` and asserting bit-exact equality
with its own output (`crates/frink-quant/src/iq_tier_goldens.rs`).

## Speculative decoding design

`frink-models::speculative`'s prompt-lookup speculative decoding
(propose candidate continuation tokens by finding a repeat of the
current context elsewhere in token history, verify them in a batched
forward pass) follows the prompt-lookup-decoding idea that is common
across serving engines -- no code is copied; the n-gram matching,
batched verification, and accept/reject accounting in frink's
implementation are original.

## MoE expert-offload design

The CPU/GPU expert-placement model in `frink-moe` (`PlacementPlan`,
default-CPU-with-GPU-overrides) is a Rust-native design inspired by the
tensor-name-regex offload controls popularized by ik_llama.cpp
(`--cpu-moe`, `-ncmoe`, `--override-tensor`):

  https://github.com/ikawrakow/ik_llama.cpp

ik_llama.cpp is a fork of llama.cpp and inherits its MIT license.

## Metal Concurrent MoE hazard tracking

`frink-metal`'s `MemRanges` (and the one-Concurrent-command-buffer MoE
encode path that uses it) follows the same hazard-tracking idea as
llama.cpp / ggml `ggml_mem_ranges` under `MTLDispatchTypeConcurrent`:
overlapping dispatches may share sources when destinations do not alias.
The Rust implementation in `crates/frink-metal/src/mem_ranges.rs` is
written independently against that public design; no ggml source lines
are copied.

## CUDA execution path and hardware capability detection

`frink-cuda`'s `HardwareProfile`/`SimdCaps` detection (probe once,
report a plain struct, let performance decisions derive from the
detected machine) is an original Rust-native design; no code is copied.

`frink-cuda`'s optional `cuda` feature uses `cudarc`
(https://github.com/coreylowman/cudarc), an MIT-licensed Rust CUDA
binding crate, configured for dynamic loading (no CUDA toolkit
required to build). `cudarc` is pinned to
version 0.11.9 in `frink-cuda/Cargo.toml`; see that file's comment for
why (newer releases' transitive dependencies require a newer rustc
than this project's development environment has had available).

The CUDA C kernel sources in `frink-cuda/src/gpu.rs` are original code
written to mirror `frink-quant`'s CPU fused-dequant math; see that
module's doc comments for their verification status.

## GBNF grammar engine

`frink-models::grammar` is a direct Rust port of llama.cpp's
grammar-constrained decoding: `src/llama-grammar.h` and
`src/llama-grammar.cpp`. This is a port of the ALGORITHM, not an
independent reimplementation from a specification: the GBNF parser, the
stack machine (`advance_stack`, `match_char`, `match_partial_char`), the
shared-prefix candidate walk (`reject_candidates`) and both UTF-8
decoders follow upstream's structure and arithmetic closely enough that
the correspondence is the point, since divergence would mean accepting a
different language.

The eight `.gbnf` grammars in `grammar::grammars` are reproduced verbatim
from llama.cpp's `grammars/` directory, and the parser goldens are
transcribed from its `tests/test-grammar-parser.cpp`.

llama.cpp is MIT licensed:

  Copyright (c) 2023-2024 The ggml authors and llama.cpp
  contributors (llama.cpp)

Two deliberate, documented divergences, both in the module's own doc
comments:

- `"a"{4,2}` (a repetition whose maximum is below its minimum) hangs
  upstream: `max_times - min_times` wraps as unsigned and loops roughly
  2^64 times, and upstream's `> 2000` element guard does not catch it.
  frink saturates to zero optional copies, so the rule reads as
  `"a"{4}`.
- Token pieces are handled as bytes rather than `&str`, because a piece
  holding one byte of a multi-byte character is not valid UTF-8.

One upstream asymmetry is REPRODUCED rather than corrected, and pinned
by its own test: `reject_candidates` keeps an empty piece on a satisfied
grammar while `accept_token` drops every empty stack and then fails.
That is harmless only because the sampler masks empty pieces before the
grammar sees them, and matching upstream is worth more than quietly
being different.

## Tokenizer pre-tokenization pattern

The GPT2 byte-to-unicode remap table and the pre-tokenization regex
pattern used by `frink-models::tokenizer::GgufBpeTokenizer` reproduce
the publicly documented algorithm from OpenAI's GPT-2 `encoder.py`
(the `bytes_to_unicode()` function and the pre-tokenization regex),
reimplemented independently in Rust using the MIT/Apache-2.0-licensed
`regex` crate (https://github.com/rust-lang/regex). One deliberate,
documented deviation: the real pattern's negative-lookahead whitespace
clause is dropped, since the `regex` crate does not support lookaround
assertions by design (it stays linear-time). See the tokenizer
module's doc comments for the exact practical effect of that
difference.

## SentencePiece-BPE tokenizer algorithm and test data

`frink-models::tokenizer::GgufSpmTokenizer` implements the
SentencePiece-BPE encoding algorithm (score-prioritized pairwise
symbol merging via a priority queue, plus UTF-8 byte-fallback), which
is publicly documented behavior of the SentencePiece library
(https://github.com/google/sentencepiece) and of llama.cpp's own
`llm_tokenizer_spm` for GGUF files reporting `tokenizer.ggml.model =
"llama"`. Frink's implementation is written independently against
that public description; no source code is copied from either project.

The test fixtures `tests/fixtures/llama-spm-vocab.gguf` and its
accompanying `.gguf.inp`/`.gguf.out` reference test cases are
downloaded directly, unmodified, from `ggml-org/llama.cpp`'s own
repository (`models/ggml-vocab-llama-spm.gguf` and its `.inp`/`.out`
companions), used here under the same terms as the rest of that MIT-
licensed project, purely as a correctness test oracle -- verifying
frink's independent implementation produces the same output as
llama.cpp's own tokenizer for the same real vocabulary and real test
inputs.

## OpenAI-compatible HTTP surface

The `/v1/chat/completions`, `/v1/models`, and `/health` endpoint shapes
in `frink-server` follow the now-industry-standard OpenAI Chat
Completions API convention that llama.cpp's server and the other
OpenAI-compatible engines implement, including this project's own
sibling, antonellof/cognitora-inference. No server code is copied from
any of them; the wire format is a public API contract, not source
code.

## Response caching design

`frink-server::cache::ResponseCache` (a whole-response LRU+TTL cache
for exact-repeat requests) is a from-scratch Rust implementation
inspired by the general design of Shimmy's
`src/cache/response_cache.rs` (keying a cache by prompt + model +
generation parameters, with TTL expiry). No code is copied; the
eviction bookkeeping, key digest, and hit/miss accounting in frink's
version are original.

## Architecture vocabulary

RMSNorm, RoPE, grouped-query attention (GQA), and SwiGLU-gated MoE
feed-forward blocks are standard, widely published transformer
building blocks (see the LLaMA, Mixtral, and DeepSeek-V3 technical
reports). Frink's implementations in `frink-core` and `frink-moe`
are original Rust code written against those public descriptions.

## Sliding-window attention and Qwen2-MoE shared-expert gating

`frink_core::attention::causal_gqa_attention_windowed` (sliding-window
GQA attention) and `MoeWeights::shared_expert_gate`'s sigmoid-gated
shared-expert combine (`frink-models::decoder`) were identified as real
capability gaps by reading `huggingface/candle`'s real, Apache-2.0-licensed
source (`candle-transformers/src/models/mixtral.rs` and `qwen2_moe.rs`),
which mask attention scores where `key_pos + sliding_window < query_pos`
and scale the shared-expert output by `sigmoid(shared_expert_gate(x))`
respectively. The exact formulas were independently re-confirmed against
the real `transformers` library source
(`transformers/models/qwen2_moe/modeling_qwen2_moe.py`) and llama.cpp's
real `src/models/qwen2moe.cpp` (confirming the exact real GGUF tensor
name, `blk.N.ffn_gate_inp_shexp.weight`) before being written into
frink. Frink's implementation is written independently (candle uses a
different tensor/backend abstraction throughout), not copied
line-for-line from candle's source, but the two algorithms and their
real GGUF tensor mapping were confirmed via candle as the reference,
under the same Apache-2.0 license as frink itself.

## Frink Studio frontend (`ui/`)

The web UI is a separate React application under [`ui/`](../ui). It is
distributed as built JavaScript, so every dependency's licence travels
with the bundle rather than staying a lockfile detail. **All of them are
permissive: MIT, Apache-2.0 or ISC.** No AGPL or GPL dependency is
permitted, and `npm run licenses` (`ui/scripts/check-licenses.mjs`)
walks the whole installed tree on every CI run and fails the build on
anything that is not.

Runtime dependencies, which is to say the code that ends up in the
shipped bundle:

| Package | Licence |
|---|---|
| `react`, `react-dom` | MIT |
| `react-router` | MIT |
| `@assistant-ui/react`, `@assistant-ui/react-markdown`, `assistant-stream` | MIT |
| `remark-gfm` (and the `react-markdown` / unified tree under it) | MIT |
| `@radix-ui/react-*` (dialog, label, popover, separator, slider, slot, tabs, tooltip) | MIT |
| `@tanstack/react-table` | MIT |
| `tailwindcss`, `@tailwindcss/vite`, `tw-animate-css` | MIT |
| `clsx`, `tailwind-merge` | MIT |
| `class-variance-authority` | Apache-2.0 |
| `lucide-react` | ISC |

Build-time only (Vite, TypeScript, ESLint and their trees) is MIT and
Apache-2.0, with two exceptions that contribute no code to the bundle
and are allow-listed by name in the licence checker: `lightningcss`
(MPL-2.0), Tailwind's CSS transformer, and `caniuse-lite` (CC-BY-4.0),
a browser-support data table read during the build and discarded.

No component source, stylesheet, class-name string or design-token value
was copied from any of the products studied while designing this UI. In
particular **Unsloth Studio is AGPL-3.0** and frink is Apache-2.0: it
was read for information architecture and for which libraries a shipped
product of this kind chooses, and nothing else. Camelid (MIT) was read
the same way, for feature and layout ideas only.

## FreeToken: edge-native MoE serving policy

Frink contains a Rust port of the host-side decision logic in
[FreeToken](https://github.com/FlashML-org/FreeToken), the edge-native
MoE serving engine described in *FreeToken: Efficient Edge-Native MoE
Serving with Bandwidth-Adaptive Execution*
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157), Yang, Fan, Pan,
Xi, Wang, Sun, Keutzer, Han, Zaharia, Xu and Stoica, 2026). This is a
real port, not independent design work: the algorithms, the constants,
the tie-breaking rules and the module boundaries follow FreeToken's
Python source directly, and each Rust module names the file it came
from. FreeToken is Apache-2.0 licensed, the same license as frink:

  Copyright (c) 2026 FlashML and the FreeToken contributors

The port originally lived in a crate of its own, `frink-edge`. It no
longer does: the modules were moved into the crates that use them, and
the ones nothing would ever use were deleted. This notice tracks where
each surviving piece now lives, because the attribution follows the
code and not the crate name.

**The MoE expert-residency half, in `frink-core`**, beside
`expert_store`, which holds the byte budget it sizes:

| frink module | FreeToken source |
|---|---|
| `frink_core::expert_cache` | `python/freetoken/moe/offload_cache.py`, `moe/offload_kernels.py` |
| `frink_core::expert_slots` | the slot/bank model behind `moe/offload_cache.py` |
| `frink_core::expert_budget` | `python/freetoken/engine/cache_budget.py` |
| `frink_core::qstar` | `python/freetoken/moe/bench_profile.py`, the split in `moe/offload_kernels.py` |
| `frink_core::bench_profile` | `python/freetoken/moe/bench_profile.py` (path resolution) |
| `frink_core::residency` | `python/freetoken/moe/offload_cache.py` (residency plans) |
| `frink_core::placement` | `python/freetoken/engine/engine.py` (CPU-layer selection) |
| `frink_core::summary_stats` | `python/freetoken/server/stats.py` (percentile, mean-of-present) |
| `frink_core::expert_pool` | frink's own CUDA `SlotDevice`, against the ported trait |

**The serving-policy half, in `frink-server::policy`:**

| frink module | FreeToken source |
|---|---|
| `policy::radix::{tree,plain}` | `python/freetoken/kvcache/radix_cache.py` |
| `serving::admission` | `python/freetoken/scheduler/{prefill,decode,table}.py` |
| `serving::batch::status` | `python/freetoken/scheduler/status.py`, the pool-occupancy helpers of `scheduler/scheduler.py` |
| `policy::parser::reasoning` | `python/freetoken/server/reasoning_parser.py` |
| `policy::parser::tool_call` | `python/freetoken/server/function_call_parser.py` |
| `policy::effort` | `python/freetoken/tokenizer/effort.py`, part of `tokenizer/tokenize.py` |
| `policy::detokenize` | `python/freetoken/tokenizer/detokenize.py` |
| `policy::anchor` | `python/freetoken/scheduler/cache.py` (tool-call anchor, window slide) |
| `policy::pool_budget` | `python/freetoken/engine/cache_budget.py` |
| `policy::rebuild` | the live re-split in `python/freetoken/engine/engine.py` |
| `stats::ring` | `python/freetoken/server/request_ring.py` |
| `stats::{rate,serving}` | `python/freetoken/server/stats.py` |
| `policy::maintenance` | `python/freetoken/server/accounting.py`, the gate in `server/api_server.py` |
| `policy::outbox` | `python/freetoken/server/accounting.py` (stop receipts) |
| `policy::footprint` | `python/freetoken/server/footprint.py` |

**In `frink-models`:**

| frink module | FreeToken source |
|---|---|
| `deepseek_v4_budget` | `python/freetoken/models/deepseek_v4/` (KV tier sizing, the compressor schedule) |

**In `frink-cli`:**

| frink module | FreeToken source |
|---|---|
| `bench_client` | `python/freetoken/bench/serving.py` |

**What was ported and has since been DELETED**, recorded here because
the port happened and the attribution should say so honestly rather
than quietly dropping the rows:

| deleted module | FreeToken source | why |
|---|---|---|
| `cache_manager` | `python/freetoken/scheduler/cache.py` | a second page ledger; frink's own `frink_core::cache::KvBlockPool` is wired and running |
| `radix::swa` | `python/freetoken/kvcache/swa_radix_cache.py` | reachable only from `cache_manager` |
| `radix::hybrid` | `python/freetoken/kvcache/hybrid_radix_cache.py` | reachable only from `cache_manager`; frink refuses recurrent checkpoints |
| `window_pool` | `python/freetoken/kvcache/hybrid_swa_pool.py` | reachable only from `cache_manager` |
| `state_pool` | `python/freetoken/kvcache/linear_state_pool.py`, `attention/linear.py` | reachable only from `cache_manager` |
| `cache_report` | `python/freetoken/cache_report.py` | rendered a `CacheGeometry` nothing built |
| `supervisor` | `python/freetoken/daemon/serve_manager.py` | manages an engine child process; frink ships one binary |


FreeToken credits prior art of its own, llama.cpp among it: its radix
prefix cache follows the standard `RadixCache` / `SWARadixCache` design
and its incremental detokenization borrows the printable-text heuristic
that `transformers`' `TextStreamer` uses. Those lineages carry through
this port. (What Apache-2.0 obliges is FreeToken's copyright and
licence identification, above; this paragraph is frink's own account of
where the design came from.)

What was **not** ported, and why: everything in FreeToken that computes
rather than decides. The Triton and CUDA kernels, the C++ CPU MoE
executor, the FTW weight loader's O_DIRECT/mmap machinery, the pinned
host-bank allocator and the CUDA-graph capture ordering are all
torch/CUDA-bound, and frink has its own equivalents or its own plans
for them. The ported policy is deliberately tensor-free: every module
takes measured numbers and returns a decision, which is why it is all
testable on a host with no GPU and no model.

A full recursive re-read of the reference in August 2026 -- six readers
over its 435 files, checked against every frink crate rather than
against the port's own scope -- found 34 further pieces the first pass
had missed. They are tracked individually in
[`docs/plans/archive/freetoken-parity.md`](plans/archive/freetoken-parity.md) rather
than summarised here, because each closes on its own. That review also
found one place where the port had got a *shipped* rule wrong rather
than merely omitted it: `route_top_k_grouped` implemented "k from every
group" instead of the DeepSeek-V3/GLM `noaux_tc` rule. It is recorded in
that plan as `noaux-tc-group-limited-routing`.

Two *decision*-side pieces are also absent, and are absent because they
are single-family grammars for checkpoints frink does not run, not
because they were hard:

- **MiniMax-M3's tool-call grammar.** Its reasoning markers and its
  adaptive leading-closer are ported; its namespaced, recursively
  nested `]<]minimax[>[<k>…` element grammar for *arguments* is not, so
  `ToolCallFormat` has no M3 arm. Its reasoning parser is unaffected.
- **muse-glimmer's ATEM channel format**, on both the reasoning and the
  tool-call side. It is one vendor's channel protocol with its own
  header-span and synthetic-terminator rules, and nothing in
  `docs/MODELS.md` runs it.

Adding either is mechanical: both slot into the same `ReasoningFormat` /
`ToolCallFormat` tables as the nine families that are here.

Where frink already had a mechanism the port would have duplicated, the
port plugs into it rather than shadowing it. `frink-core::kv_block`
(content-addressed KV blocks), `frink-core::expert_store` (the SSD
expert tier), `frink-server::serving::batch` (continuous batching),
`frink-server::stop` (which delegates its withhold rule to
`policy::detokenize` so there is one implementation of it in the
workspace).

## Design inspiration, not code reuse

The following projects informed frink's architecture and are credited
here for that influence. Aside from the section immediately above, none
of their source code appears in this repository:

- **llama.cpp** (ggml-org) -- GGUF, mmap-based weight loading, CPU/GPU
  hybrid execution model.
- **ik_llama.cpp** (ikawrakow) -- MoE-aware CPU/GPU tensor placement,
  fused MoE operators, SOTA quantization types.
- **Candle** (Hugging Face) -- pure-Rust tensor/model execution
  patterns, GGUF-in-Rust precedent (see also the section above for the
  one case where candle's real source was read directly to close a
  confirmed capability gap).
- **antonellof/cognitora-inference** -- the author's own orchestration
  layer above the serving engines; frink is designed to be pluggable
  into cognitora's `cgn-agent` as an additional engine backend.

If frink ever vendors or adapts actual source lines from an MIT or
Apache-2.0 licensed project, the applicable upstream copyright and
license notice will be added to this file alongside that code, per the
terms of those licenses.
