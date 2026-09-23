# llama.cpp full parity audit (2026-09-02)

**Companion to** [`llama-cpp-gap-inventory.md`](../llama-cpp-gap-inventory.md) (evidence-backed differential) and [`llama-cpp-parity-review-2026-09-02.md`](llama-cpp-parity-review-2026-09-02.md) (prioritized actions). This document is the **complete audit run**: file-by-file C++→Rust mapping, live `frink parity` sweep on all local checkpoints, and a ranked priority plan.

**North star:** same GGUF, same command shapes, same or better performance on hardware people own ([`north-star.md`](../north-star.md)).

**Evidence sources:**
- llama.cpp checkout: `.scratch/llama.cpp` (947 source files under `src/`, `ggml/src/`, `common/`, `tools/`)
- frink: `main` branch, 2026-09-02 evening
- Parity run: `.scratch/parity-run-2026-09-02/` (19 local GGUFs, CPU-only, `FRINK_METAL=0 FRINK_CUDA=0`)
- File map: `scripts/llama_cpp_file_map.py` → `.scratch/parity-run-2026-09-02/file_map.json`

---

## Executive summary

| Dimension | llama.cpp | frink | Gap severity |
|-----------|-----------|--------|--------------|
| Source files (C++/CUDA/Metal) | **1,103** mapped | **308** Rust files, **~230k** lines | Structural: 140 per-arch graphs vs 1 decoder |
| Architecture graphs | **140** hand-written | **16** audited + 4 dedicated engines | **P0** — 124 graphs unported |
| Backends | CPU, CUDA, Metal, Vulkan, SYCL, HIP, OpenCL, … | CPU, Metal, CUDA (partial), Vulkan (beachhead) | **P0** Vulkan; **P1** CUDA GEMM |
| CLI tools | 15+ binaries | 23+ subcommands; every core tool present | gguf-split ported 2026-09-09, imatrix and batched-bench 2026-09-11 |
| Logit parity (local sweep) | reference | 19 models tested | **2 WRONG**, 8 DRIFT (expected K-quant), 6 MATCH, 1 TIE-FLIP, 2 encoder skip |

### Live parity sweep (2026-09-02)

| Model | Tokenizer | Logits | Notes |
|-------|-----------|--------|-------|
| TinyLlama Q8_0 | MATCH | **MATCH** | Baseline |
| Llama-3.2-1B Q8_0 | MATCH | **MATCH** | |
| Llama-3.2-1B IQ4_XS | MATCH | **MATCH** | |
| Llama-3.2-3B Q4_K_M | MATCH | **MATCH** | |
| Mistral-7B Q4_K_M | MATCH | **MATCH** | SWA dense |
| OLMoE Q4_0 | MATCH | **MATCH** | MoE |
| Llama-3.2-1B Q4/Q5/Q6_K | MATCH | DRIFT | Expected: llama.cpp Q8_K activation quant |
| Gemma-2-2B Q4_K_M | MATCH | DRIFT | Same K-quant activation path |
| Qwen1.5-MoE, Qwen2.5, Yi-1.5 Q4_K_M | MATCH | DRIFT | Same |
| Meta-Llama-3.1-8B Q4_K_M | MATCH | TIE-FLIP | Near-tie, not wrong graph |
| **DeepSeek-R1-Distill Q4_K_M** | MATCH | **WRONG** | KL 2.0e-2 — investigate graph |
| **Phi-4-mini Q4_K_M** | MATCH | **WRONG** | KL 3.6e-2 — investigate graph |
| Gemma-4 E2B Q4_K_M | SKIP | SKIP | Homebrew libllama too old; needs scratch build |
| BGE-small, ms-marco MiniLM | DIVERGES (emoji) | N/A | Encoder-only; BERT emoji tokenization |
| BGE/ms-marco load | — | REFUSE | Expected: embedding scope |

**Actionable from sweep:** DeepSeek-R1-Distill and Phi-4-mini need layer-divergence investigation (`frink layer-divergence`). K-quant DRIFT is documented and expected (§10 of gap inventory). Rebuild reference from `.scratch/llama.cpp` for gemma-4 parity.

---

## Part 1: File-by-file C++ → Rust conversion map

### 1.1 Scale comparison

| Tree | Files | Lines (approx) | Organization |
|------|-------|----------------|--------------|
| llama.cpp `src/` + `ggml/src/` + `common/` + `tools/` | 1,103 | ~350k+ | Per-arch graphs, ggml tensor IR, 18 backends |
| frink `crates/` | 308 `.rs` | 229,785 | Hand-written decode graph, 2.5 backends |

### 1.2 Core library mapping

| llama.cpp (C++) | frink (Rust) | Status | Notes |
|-----------------|---------------|--------|-------|
| `src/llama.cpp` | `frink-models/src/lib.rs` | partial | No libllama C API |
| `src/llama-model.cpp` | `frink-models/src/loader.rs` (2k+ lines) | partial | One loader vs per-arch wiring |
| `src/llama-model-loader.cpp` | `frink-gguf/src/lib.rs` | **ported** | GGUF mmap |
| `src/llama-model-saver.cpp` | `frink-gguf/src/writer.rs` | partial | Write header/tensors; no full export |
| `src/llama-arch.cpp` | `frink-models/src/capability.rs` (2.3k lines) | partial | 150 catalog rows; 16 audited |
| `src/llama-hparams.cpp` | `frink-models/src/config.rs` | partial | Hyperparameter parsing |
| `src/llama-vocab.cpp` | `frink-models/src/tokenizer.rs` | partial | 19-case parity; BERT emoji edge |
| `src/llama-context.cpp` | `frink-models/src/decoder.rs` (6,702 lines) | partial | Contiguous + paged paths |
| `src/llama-batch.cpp` | `frink-server/src/serving/batch/` | partial | CB landed; incremental stream gap |
| `src/llama-graph.cpp` | `decoder.rs` + `attn_block.rs` | partial | No ggml graph abstraction |
| `src/llama-sampler.cpp` (4,106 lines) | `frink-models/src/sampling.rs` (834 lines) | partial | Missing: dry, xtc, typ_p, top_n_sigma, mirostat |
| `src/llama-grammar.cpp` (1,522 lines) | `frink-models/src/grammar/` | partial | GBNF + JSON schema landed |
| `src/llama-chat.cpp` | `frink-server/src/completion/` | partial | Template rendering |
| `src/llama-kv-cache*.cpp` (6 variants) | `frink-core/src/cache.rs`, `kv_budget.rs` | partial | Standard GQA only; no DSA/ISWA/MSA |
| `src/llama-memory*.cpp` (5 variants) | `frink-core/expert_cache.rs`, `residency` | partial | Policy exists; not executed on Metal/RAM |
| `src/llama-mmap.cpp` | `frink-gguf/src/lib.rs` | **ported** | |
| `src/llama-quant.cpp` | `frink-quant/src/` | partial | Read all; write Q8_0; K-encoders in progress |
| `src/llama-adapter.cpp` | — | **missing** | LoRA adapters |
| `src/unicode*.cpp` | `tokenizer/unicode.rs` | partial | Normalization |

### 1.3 Architecture graphs (140 files → 1 decoder)

llama.cpp: `src/models/*.cpp` — one file per architecture, ~50–200 lines each.

frink: `decoder.rs` + `engine_factory.rs` + 4 dedicated engines:

| Engine | Architectures | llama.cpp counterpart |
|--------|---------------|----------------------|
| Generic `Decoder` | 16 audited of 57 generic-gqa rows | 140 graphs collapsed |
| `MlaEngine` | deepseek2, deepseek32, mistral4 | `deepseek2.cpp`, `deepseek.cpp`, … |
| `Glm52Engine` | glm-dsa, glm4 | `glm.cpp`, `glm4.cpp` |
| `KimiEngine` | kimi-linear, kimi_k3 | `kimi.cpp` |
| `Gemma4Engine` | gemma4, gemma4-assistant | `gemma4.cpp` |

**Audited (evidence):** llama, qwen2, qwen2moe, qwen3, qwen3moe, olmoe, gemma2, gemma3, phi3, gpt-oss, dots1, bailingmoe, deepseek, maincoder, hunyuan-moe, seed_oss

**41 unaudited refusals** (triaged): 9 fixture-away, 2 one-match-arm, 26 new-code, 4 unknown

### 1.4 ggml backends

| llama.cpp backend | Files | frink | Status |
|-------------------|-------|--------|--------|
| `ggml-cpu/` | 64 | `frink-quant` + `frink-core` | partial — AVX2/NEON/i8mm; no AVX512/SVE/AMX |
| `ggml-cuda/` | 274 | `frink-cuda` (5 files, ~2.6k lines) | partial — 8 kernels; **no GEMM, no MoE FA** |
| `ggml-metal/` | 10 | `frink-metal` (8 files, ~20k lines) | partial — competitive MoE; missing sinks/ALiBi |
| `ggml-vulkan/` | 145 | `frink-vulkan` | partial — Q8_0 beachhead only (verdict GO) |
| `ggml-sycl/` | 157 | — | **missing** |
| `ggml-hip/` | shim | — | **missing** (falls from CUDA) |
| `ggml-opencl/` | 172 | — | **missing** |
| `ggml-blas/` | 1 | — | **missing** |
| `ggml-rpc/` | 1 | — | **missing** |
| Other (CANN, WebGPU, …) | — | — | **missing** |

### 1.5 Tools mapping

| llama.cpp tool | frink command | Status |
|----------------|----------------|--------|
| `llama-cli` | `frink run` | partial — flag parity mostly done |
| `llama-server` | `frink serve` | partial — slot save/restore present, `GET /slots` / `/props` / `/infill` absent |
| `llama-bench` | `frink bench` | **ported** |
| `llama-quantize` | `frink quantize` | partial — Q8_0, Q4_K_S/M, Q5_K_S/M, Q6_K byte-identical, with and without `--imatrix`; Q2_K/Q3_K, IQ tiers, MXFP4 refused by name |
| `llama-perplexity` | `frink perplexity` | partial — corpus ppl; no HellaSwag |
| `llama-tokenize` | via `frink parity` tokenizer sweep | partial |
| `llama-gguf-split` | `frink gguf-split` | **ported**: split by tensors or size, merge, `--no-tensor-first-split`, `--dry-run` |
| `llama-imatrix` | `frink imatrix` | **ported** (2026-09-11): same file format both ways, GGUF and legacy `.dat`; sums agree to the forward pass's precision, not bit for bit (see `docs/CLI.md`) |
| `llama-batched-bench` | `frink batched-bench` | **ported** (2026-09-11): same sweep, same ten columns and JSONL keys, drives `forward_multi_seq` directly; `-b`, `-kvu`, `-fa`, `-tb` refused by name |
| `llama-mtmd` (multimodal) | — | **missing** |
| `llama-tts` | — | **missing** |
| `llama-rpc` | — | **missing** |
| `llama-export-lora` | — | **missing** |
| `llama-cvector-generator` | — | **missing** |
| `llama-fit-params` | — | **missing** |

### 1.6 frink crate → llama.cpp responsibility

| Crate | Lines | llama.cpp equivalent |
|-------|-------|---------------------|
| `frink-models` | ~45k | `src/llama-*.cpp` + `src/models/` |
| `frink-quant` | ~8k | `ggml-quants.c`, `llama-quant.cpp`, CPU SIMD |
| `frink-core` | ~15k | `ggml-cpu`, weight ops, KV, kernel registry |
| `frink-metal` | ~20k | `ggml-metal/` |
| `frink-cuda` | ~2.6k | `ggml-cuda/` (5% coverage) |
| `frink-server` | ~10k | `tools/server/` |
| `frink-cli` | ~8k | `tools/cli/`, `common/arg.cpp` |
| `frink-gguf` | ~2k | `llama-model-loader.cpp`, mmap |
| `frink-moe` | ~3k | MoE routing scattered in llama graphs |
| `frink-vulkan` | ~1k | `ggml-vulkan/` beachhead |
| `frink-api` | ~1k | Server wire DTOs |
| `frink-safetensors` | ~2k | Kimi/DS4 safetensors loaders |
| `frink-inference` | ~1k | Shared inference types |

---

## Part 2: What's missing (by category)

### 2.1 Correctness gaps (P0)

1. **DeepSeek-R1-Distill Q4_K_M** — WRONG logits (KL 2e-2). Same family as qwen2 (audited); likely a R1-specific template or graph edge.
2. **Phi-4-mini Q4_K_M** — WRONG logits (KL 3.6e-2). phi3 is audited; Phi-4 may need dedicated handling.
3. **Reference vintage** — Homebrew libllama cannot load gemma-4; rebuild from `.scratch/llama.cpp` per `tools/build_llama_logits.sh`.
4. **BERT emoji tokenization** — WordPiece models diverge on ZWJ emoji sequences (encoder scope; affects embedding checkpoints).

### 2.2 Architecture coverage (P0)

- **124 of 140** llama.cpp graphs have no audited frink path
- **41** refuse as unaudited (triaged queue in `capability.rs`)
- **58 dedicated** + **32 deferred** refuse by name
- Prerequisite: [`model-layer-reorg.md`](../model-layer-reorg.md) — split `decoder.rs` so adding an arch is a new file, not a 6700-line edit

### 2.3 Backend gaps (P0–P1)

| Gap | Impact | Size |
|-----|--------|------|
| CUDA batched GEMM + mmvq | Prefill 28 vs 57466 tok/s on 4090 (documented) | L → XL |
| Vulkan backend | AMD/Intel/Android GPUs unreachable | XL |
| Metal attention sinks | gpt-oss runs on CPU silently | M |
| F16/BF16 GPU paths | Dequant-to-F32 overhead | M |
| 15/21 quant types missing Metal matvec | Silent CPU fallback | L |
| AVX512/VNNI | x86 CPU gap | L |

### 2.4 Serving & CLI (P1)

| Gap | llama.cpp | frink |
|-----|-----------|--------|
| Slot save/load | yes | **`POST /slots/{id}?action=save\|restore`** (2026-09-11), gated on `--slot-save-path`, restoring into the prefix cache; the file carries a checkpoint fingerprint and a mismatch is refused by name. `erase` refused by name, `GET /slots` absent |
| `-np` / `--parallel` | yes | wired (was already; the row was stale). Read back as `frink_scheduler_max_seqs` on `/metrics` since 2026-09-11 |
| `-b` / `-ub` batch flags | yes | **wired** (2026-09-11), one number on both decode paths, read back as `frink_scheduler_prefill_chunk` |
| Partial `-ngl` | yes | all-or-nothing |
| Streamed CB output | token stream | buffers full completion |
| gguf-split merge/split | yes | **`frink gguf-split`**, both directions |
| imatrix | yes | **`frink imatrix`** + `frink quantize --imatrix`, byte-identical output |

### 2.5 Sampling (mostly closed)

Closed since gap inventory: GBNF, JSON schema, logit_bias, presence/frequency penalty, `--repeat-last-n`, `--samplers` ordering.

Still missing: `dry`, `xtc`, `typ_p`, `top_n_sigma`, mirostat, infill, adaptive_p.

### 2.6 Embeddings & multimodal (P1)

- 12 encoder architectures refuse (`bert`, `nomic-bert`, `jina-bert-v2/v3`, …)
- `/v1/embeddings` pools decoder hidden states only
- No `/v1/rerank` route
- No multimodal (`mtmd`) path

### 2.7 Performance (P2)

- CPU: all 16 bench rows 1.41x–5.06x slower than llama.cpp
- Metal: at or past parity (8/12 decode rows faster)
- MoE Metal `activation_counts` fixed; prefill routing still drops counts

---

## Part 3: Priority plan

Ranked per [`north-star.md`](../north-star.md) and [`roadmap.md`](../roadmap.md).

### P0 — Correctness and trust (this week)

| # | Item | Effort | Evidence |
|---|------|--------|----------|
| 1 | **Investigate DeepSeek-R1-Distill + Phi-4-mini WRONG** | S | `frink layer-divergence -m …` on both |
| 2 | **Rebuild llama_logits from `.scratch/llama.cpp`** | S | Unblocks gemma-4 parity |
| 3 | **Execute fixture-away triage queue** (9 archs) | M | bailingmoe2, minimax-m2, olmo2, … — one fixture each |
| 4 | **Close remaining one-match-arm rows** (2) | S | Named in `unaudited_triage.rs` |

### P1 — Coverage expansion (this month)

| # | Item | Effort | Blocks |
|---|------|--------|--------|
| 5 | **Model layer reorg phase 1** — extract `attn_block`, `rope`, per-arch trait | L | Everything below |
| 6 | **K-quant encoders** (#70) — Q4_K_M write parity | L | `frink quantize` usefulness |
| 7 | **CUDA mul_mm + mmvq** — port from Metal `mul_mm_sg_impl` | L | CUDA prefill |
| 8 | **Server: ~~`-np`~~, ~~slot save/load~~, streamed CB** | M | Serving parity. `-np` was already wired; slot save/restore and `-b`/`-ub` landed 2026-09-11. Streamed CB output remains |
| 9 | **Embedding model path** — WordPiece + BERT loader | L | BGE/E5/nomic-embed |
| ~~10~~ | ~~**gguf-split utility**~~ | S | **done 2026-09-09**: `frink gguf-split` |

### P2 — Hardware reach (this quarter)

| # | Item | Effort | Unlocks |
|---|------|--------|---------|
| 11 | **Vulkan backend** (post beachhead) | XL | AMD/Intel/Android |
| 12 | **CPU fork-join pool** | M | 1.4x–5x CPU gap |
| 13 | **Out-of-core MoE execution** | XL | Models > RAM |
| 14 | **Audit outward: glm4moe, deepseek2 MLA evidence** | M | Large model claims |
| 15 | **x86 AVX512 measurement** | M | Commercial edge |

### P3 — Long tail

- 26 new-code architecture graphs (apertus xIELU, dbrx LayerNorm, bitnet, …)
- Multimodal (`mtmd`), TTS, RPC
- LoRA adapters (imatrix and batched-bench ported 2026-09-11)
- SYCL/HIP/OpenCL backends

---

## Part 4: Recommended execution slices

Each slice ships something a user can verify:

### Slice A: "Fix the two WRONG models" (1–2 days)
```bash
FRINK_METAL=0 FRINK_CUDA=0 frink layer-divergence \
  -m models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
FRINK_METAL=0 FRINK_CUDA=0 frink layer-divergence \
  -m models/Phi-4-mini-instruct-Q4_K_M.gguf
```
Close when both return MATCH on `frink parity`.

### Slice B: "Reference from scratch" (half day)
```bash
cmake -B /tmp/llamabuild -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  .scratch/llama.cpp
cmake --build /tmp/llamabuild --target llama -j8
LLAMA_CPP_PREFIX=/tmp/llamabuild bash tools/build_llama_logits.sh
frink parity -m models/gemma-4-E2B-it-Q4_K_M.gguf
```

### Slice C: "Fixture-away batch" (1 week)
For each of the 9 fixture-away architectures in `unaudited_triage.rs`:
1. `scripts/make_*_fixture.py` (synthetic GGUF, kilobytes)
2. libllama golden logits in `tests/`
3. Add to `AUDITED_GENERIC_GQA`
4. `frink parity` on fixture

### Slice D: "Model layer reorg phase 1" (2–3 weeks)
Per [`model-layer-reorg.md`](../model-layer-reorg.md):
1. Extract shared block vocabulary (`AttnBlock`, `FfnBlock`, norm slots)
2. One architecture (`olmo2` — post-norm wiring) as proof
3. Gate: no edit to `decoder.rs` for new archs after phase 2

---

## Appendix A: Parity run raw results

Full logs: `.scratch/parity-run-2026-09-02/*.log`

Regenerate:
```bash
git checkout main
cargo build -p frink-cli --release
bash tools/build_llama_logits.sh
export FRINK_METAL=0 FRINK_CUDA=0
for f in models/*.gguf; do frink parity -m "$f"; done
```

## Appendix B: File map regeneration

```bash
python3 scripts/llama_cpp_file_map.py > .scratch/parity-run-2026-09-02/file_map.json
```

## Appendix C: Architecture manifest

```bash
frink archs --write docs/manifests/architecture_manifest.md
```

---

*Generated 2026-09-02 on main branch. Re-run parity and file map after significant merges.*
