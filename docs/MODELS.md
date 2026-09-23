# Models

What Frink runs, and how it compares to llama.cpp on the same host.
Speed table: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)
(`frink bench` vs `llama-bench`). Suite list:
[`benchmarks/suite.json`](../benchmarks/suite.json). Architecture list:
`frink archs` →
[`manifests/architecture_manifest.md`](manifests/architecture_manifest.md).

**Gap** = `llama / frink`. Values below 1.0 mean Frink is faster.

Suite policy: keep the **current** generation per family. Llama-3.2, not
3.1. Gemma-3/4, not Gemma-2. Phi-4, not Phi-3. Older GGUFs still load
when the architecture is supported, they are simply not measured in the
published table. To measure a new model, add a suite entry and put the
GGUF under `models/`.

## Recommended starters

| Model | Notes |
|---|---|
| SmolLM2-135M-Instruct Q8_0 | Tiny. Metal ahead of llama, CPU well behind |
| TinyLlama-1.1B-Chat Q8_0 | Smallest verified smoke |
| Phi-4-mini-Instruct Q4_K_M | Metal works again. The RoPE kernels now carry `n_rot` (96 of head_dim 128) and LongRoPE's `attn_factor`, and `verify --backend metal` returns identical CPU and Metal token ids with prefill covered. The Metal rows in `benchmarks/RESULTS.md` predate that fix and were taken on the wrong graph. **Do not quote them until Phi-4 is measured again.** |
| Llama-3.2-3B-Instruct Q4_K_M | Metal flagship in the suite |
| Gemma-4-E2B-IT Q4_K_M | Dedicated engine + `gemma4` BPE |

```bash
./target/release/frink -m /path/to/model.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

./target/release/frink-server -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8383

./target/release/frink chat --url http://127.0.0.1:8383
```

## Verified (Host B)

Gap = `llama / frink` from `frink bench` vs `llama-bench` (tg128 unless
noted). **Bold** = frink faster. Neither engine's thread count is forced.

| Model | Metal decode | CPU decode |
|---|---|---|
| SmolLM2-135M Q8_0 | **0.67×** | 2.44× |
| Qwen2.5-0.5B Q8_0 | **0.70×** | 1.66× |
| Qwen3-0.6B Q8_0 | **0.71×** | 1.63× |
| Gemma-3-1B-IT Q8_0 | **0.88×** | 1.31× |
| Llama-3.2-1B IQ4_XS | **0.94×** |, |
| Llama-3.2-1B Q4_K_M | 1.00× |, |
| TinyLlama-1.1B Q8_0 | **0.85×** | 1.49× |
| Llama-3.2-3B Q4_K_M | **0.96×** |, |
| Phi-4-mini Q4_K_M |, (owed, see above) | 1.22× |
| Mistral-7B-v0.2 Q4_K_M | 1.00× | 1.17× |
| OLMoE-1B-7B Q4_0 | 1.41× | 1.50× |

These numbers drift as runs are refreshed.
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) is generated
straight from the raw timing files, so trust it over this hand-written
summary.

Prefill is **closed on Metal for dense models** (every dense `pp512` row
is 1.02–1.08×). What is left on `pp512` is CPU across the board, plus
OLMoE (1.11×) and Gemma-3-1B (1.18×) on Metal.

## Other support

| Model / family | Status |
|---|---|
| Yi, MiroThinker, Qwen2-MoE / Qwen1.5-MoE | Run. Not in the measured suite |
| Mixtral | In the suite, skipped on 32 GiB Host B (`--fit-host`) |
| DeepSeek-V2 / V3, Mistral-Large-3, PLM (`deepseek2`, `mistral4`, `plm`) | Run on the MLA engine, both tensor forms, with YaRN as every real export declares it. **Real checkpoint: PLM-1.8B-Instruct Q8_0.** A scaling type other than YaRN is refused by name |
| GLM-4-0414 / Z1 / OCR, GLM-4.5 / 4.5-Air / 4.6 | Run. A vision tower's `rope.dimension_sections` is refused by name on `glm4` (llama.cpp rotates M-RoPE over permuted weights) and served on `glm4moe`. No real checkpoint run for 4.5: the smallest export is 106B |
| Gemma-4-E2B | Runs on a dedicated engine with the `gemma4` BPE tokenizer. Checked against a libllama that has `gemma4.cpp` |
| gpt-oss | Runs, **CPU only**: no Metal kernel implements attention sinks |
| Llama 4 (Scout, Maverick) | Runs, **CPU only**: the fused Metal launches take one window per layer, and Llama 4's window is chunked |
| Cohere2 MoE (30B-A3B) | Runs |
| MiniMax | `minimax-01` (456B-A45B) and `minimax-m2` run. `minimax-m3` **will not load**: it needs MiniMax Sparse Attention |
| LFM2, LFM2-MoE | Run. A file declaring `attention.sliding_window` is refused by name (no export writes it) |
| Granite 4.0 hybrid, Nemotron-H, Jamba, Mamba, Mamba-2, Falcon-H1, PLaMo-2 | Run. **No prefix-cache reuse and no `--model-draft`**: a recurrent state cannot be rolled back to a middle position. Nemotron-3 Super's `moe_latent_size` is refused by name |
| Qwen3.5 dense / MoE, Qwen3-Next | Run. Same recurrent limits as above |
| openPangu-Embedded (1B / 7B) | Runs. A decoder, not an embedding model |
| Ternary-Bonsai-2-27B (PrismML) | **Runs, verified on the real checkpoint** against PrismML's llama.cpp fork: first-token KL 2.1e-5. `PTQ1_0` has a CPU dot, a Metal matvec and a Metal GEMM; `PQ2_0` is recognised, not executed. Speed on an M2 Pro: 43.1 prefill / 11.17 decode against the fork's 66.8 / 11.46, decode flat with context |
| Kimi K3 / GLM-5.2 / DeepSeek V4 | Loaders and primitives only. Nothing run end to end |
| Vision | Finds an mmproj file and warns. An `image_url` in a request is an error |
| MTP / speculative | `--mtp` errors by design. Speculation is prompt-lookup only: an n-gram match over the history, no draft model |
| Embeddings and rerank | `/v1/embeddings` for a GGUF decoder (mean/last pool) and for the `bert`, `nomic-bert` and `jina-bert-v3` encoders, each checked against llama.cpp's pooled embedding. `/v1/rerank` needs a cross-encoder with a rank head; `/v1/score` takes either and says which answered |

## When a model will not load

Some checkpoints stop with an error instead of running. That is
deliberate: a model whose graph Frink only partly implements would
load, run fast, and return fluent text computed by the wrong maths,
with nothing in the output to tell you. An error you can read beats
output you cannot trust.

The error always names the reason. Six things cause it.

1. **The architecture is unknown.** Not in the capability registry.

2. **Known, not implemented.** The refusal names the missing feature.
   `minimax-m3` is the current example: it needs MiniMax Sparse
   Attention.

3. **The file carries weights Frink never reads.** The loader records
   every tensor name it looks up and stops if any are left over:
   *"checkpoint carries N tensor(s) this build never reads, so its
   graph is not the one this build computes."* This catches a missing
   graph feature automatically rather than one at a time, which is how
   `attn_sinks` and `exp_probs_b` were both found. Tensors for parts
   Frink does not claim to run (`mm.`, `v.`, `mmproj.`, `resampler.`,
   `audio.`) are ignored.

4. **The file declares a scale factor Frink does not apply.**
   `{arch}.logit_scale`, `{arch}.residual_scale`,
   `{arch}.embedding_scale` and `{arch}.attention.scale` are
   hyperparameters rather than weights, so check 3 cannot see them,
   and a file declaring one would otherwise load while computing a
   differently-scaled graph than it was trained as. The architectures
   that DO apply them (Granite, MiniCPM, Grok, Command-R and the rest)
   are one table, and the refusal list is derived from it rather than
   restated beside it.

   MiniCPM is the case a key-presence gate could not catch: llama.cpp
   assigns three multipliers *before* reading the file, so a MiniCPM
   export declaring nothing is still scaled by all three. It is served
   by name, not by detection.

5. **Position is encoded some other way than RoPE.** ALiBi, a learned
   absolute position table, or no rotation at all. All of these run
   now (`gpt2`, `starcoder`, `bloom`, `mpt`, `refact`, `jais`,
   Baichuan-13B); the check remains because the generic path's guess
   is "plain GQA with RoPE" and it was wrong for exactly this group
   five times.

6. **Nobody has verified this architecture against llama.cpp.** The
   shared generic-GQA decoder is a *guess*, so it is opt-in: an
   architecture reaches it only with a benchmark row, a pinned logit
   comparison against real `libllama`, or a fixture. **99
   architectures run with** that evidence today; four more stop with
   `UnauditedArchitecture`. `frink archs` prints the current list, and
   `docs/manifests/architecture_manifest.md` is the generated copy.

   `FRINK_ALLOW_UNAUDITED_ARCH=1` runs one anyway. Compare the output
   against llama.cpp yourself before trusting it.

**Gemma-2-27B and Gemma-3-4B/12B/27B were corrected on 2026-09-02**,
in two ways that both produced fluent text: the 27B sizes took
`1/sqrt(head_dim)` where llama.cpp takes `1/sqrt(n_embd/n_head)` for
that size alone, and Gemma-3 4B and up applied linear RoPE scaling to
the sliding layers llama.cpp ropes unscaled. Gemma-3-1B is the one
size with no `rope_scaling` and was the audited fixture, which is why
neither was visible.

The evidence is llama.cpp's source and loader tests, **not** a logit
comparison: no checkpoint of those sizes exists on the development
host, and the per-layer RoPE fix is proven on synthetic two-layer
stacks. `frink parity` against a real gemma-3-4b is what would settle
it. No published benchmark number is affected:
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) carries Gemma-3-1B
only.


### What "unaudited" costs you, per architecture

"Unaudited" is not one thing. None of the 4 is a fixture or a single
match arm away any more: they need an attention implementation or a
reading nobody has done, and the refusal says which, with the
`llama.cpp/src/models/*.cpp` line that decides it:

| Class | Means |
|---|---|
| `FIXTURE-AWAY` | Frink already computes this graph. What is missing is evidence. |
| `ONE MATCH ARM` | One small, named piece: an activation, a norm slot, a routing flag, an ordering. |
| `NEW CODE` | A different attention or residual structure. Not close. |
| `UNKNOWN` | Reading both trees did not settle it. The message says what would. |

All 4 have now been read on both sides (`frink_models::capability`,
pinned by `crates/frink-models/tests/unaudited_triage.rs`). The
distribution is the headline answer to "how far is Frink from llama.cpp
on models":

| Class | Count |
|---|---|
| fixture-away | 0 |
| one match arm | 0 |
| new code | 3 |
| unknown | 1 |

**Both cheap classes are empty.** `gemma` was the last fixture-away row
and `chatglm` the last one-match-arm row; nothing still refusing is one
fixture or one arm away. That is a better answer than the count alone:
the cheap wins are spent, and what is left is 3 rows needing a
different graph plus one name nobody can get a file for.

The counts in this table are pinned against the catalog by
`crates/frink-models/tests/documented_counts.rs`, because they had
already gone stale once in the direction that matters: the table read
`new code 1` while the catalog held three, so the document that exists
to say how far frink is from llama.cpp understated the gap by half.

**The two cheap classes are empty**, which is a better answer than the
count alone: nothing still refusing is one fixture or one match arm
away. `gemma` was the last fixture-away row and `chatglm` the last
one-match-arm row, both closed on 2026-09-10.


**New code (3).** A different attention or residual structure.

| Architecture | What is missing |
|---|---|
| `graniteswitch` | A second routing stage over expert groups |
| `qwen4exp` | A per-token adapter selection the decoder has no site for |
| `grovemoe` | A second expert bank, and llama.cpp's graph and the reference model disagree about it: the graph feeds the chunk experts the routed experts' OUTPUT and gathers their weights at the CHUNK index, `modeling_grove_moe.py` does neither. There is no single graph to match |

Each refusal prints its own blocker with the
`llama.cpp/src/models/*.cpp` line that decides it, so `frink -m <file>`
on an unsupported checkpoint is the authoritative answer rather than
this table. The verdicts live on the catalog row
(`frink_models::capability`).

**How the column emptied** is a long story and it is not this
document's: forty-odd rows closed between 2026-09-02 and 2026-09-20,
each with a libllama-golden fixture, and each is a CHANGELOG entry.
What is worth carrying here is the rule those closures kept finding:
several refusals usually share ONE cause, and measuring a seam's reach
across all 155 of llama.cpp's graphs before building it is what turned
single rows into groups.

The recurring shapes, and which are still open:


| Shape | Status |
|---|---|
| A per-head LayerNorm on Q and K, one weight row per head | **Open**: refused by name (`frink_models::qk_layer_norm`) from a `stablelm` fixture libllama runs. Built by StableLM-2-12B, Command-R+ (64 layers) and `chameleon` |
| A second expert bank | **Open**: `grovemoe`, above |
| A second routing stage, a per-token adapter | **Open**: `graniteswitch`, `qwen4exp`, above |
| Everything else in this list | **Closed.** Per-layer head counts and FFN widths, parallel residuals, LayerNorm in both forms, learned position tables, ALiBi, NoPE layers, sliding-window arrays, MTP blocks inside `block_count`, split K/V head widths, per-layer activation parameters, per-position attention temperature, gated attention, attention sinks, recurrent blocks at the attention site, ungated and non-SwiGLU FFNs. Each closed with a libllama-golden fixture; the CHANGELOG has the dates and the KL numbers |

**Unknown (1).** `phi4` is the only row left here. It is not in
llama.cpp's `LLM_ARCH_NAMES` -- `src/llama-arch.cpp` carries `phi3` and
no phi4 entry -- so there is no reference graph to diff against, and
Frink admits it as phi3's fused-QKV / fused gate+up graph on the
assumption that a file spelling it means the same thing. It refuses
until a real file settles that, and its message says which tensor in
`blk.0` would decide it.

**`mistral`, `mixtral` and `yi` are not architectures.** None is in
llama.cpp's `LLM_ARCH_NAMES` or gguf-py's `MODEL_ARCH_NAMES`, and
libllama refuses a file declaring one (`unknown model architecture:
'mistral'`, measured). Every real checkpoint of all three declares
`general.architecture = llama`, which is audited and runs. Frink
refuses the three strings by name and says the actionable thing:
re-convert with `convert_hf_to_gguf.py` and the file loads as `llama`.

That also closed a live hazard: the three sat on the generic path with
NEOX RoPE while `llama` is in llama.cpp's NORM group, so a file
spelling `mistral` would have been rotated on the wrong pairs of every
Q/K head.


## Quantization support

Parsed and executable on CPU: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`,
`Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`,
`IQ4_NL`, `IQ4_XS`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S`, `MXFP4`, `TQ1_0`, `PTQ1_0`.

"Executable" is not one speed. What a format actually gets, read off
the kernel tables (`frink_quant`'s dispatch functions, and
`metal_matvec_kind_name` / `metal_mul_mm_kind_supported` /
`cuda_matvec_kind_supported` / `cuda_mul_mm_kind_supported` in
`frink-core`'s `weight_matrix.rs`):

| Tier | Formats | CPU SIMD | GPU |
|---|---|---|---|
| Full | `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` | AVX2 + NEON, plus the int-dot path (`FRINK_CPU_INT_DOT=1`) | Metal matvec + simdgroup GEMM, CUDA matvec |
| Metal only | `IQ4_XS`, `Q5_0` | AVX2 + NEON | Metal matvec + simdgroup GEMM; no CUDA kernel of either kind |
| Metal only, scalar CPU | `PTQ1_0` (PrismML ternary) | scalar | Metal matvec + simdgroup GEMM (`frink-metal/src/ternary.rs`); no CUDA kernel |
| CPU-vectorized | `Q4_1`, `Q5_1`, `Q8_1`, `Q2_K`, `Q3_K`, `IQ4_NL`, safetensors two-buffer `MXFP4` | AVX2 + NEON | none |
| AVX2 only | `IQ1_S`, `IQ2_XXS`, `IQ3_XXS` | AVX2; **scalar on ARM** | none |
| Scalar only | `IQ2_XS`, `IQ2_S`, `IQ3_S`, `IQ1_M`, GGUF-block `MXFP4` | none | none |

`Q5_0` moved up on 2026-09-01. It had a Metal simdgroup GEMM and no
matvec, so its prefill ran on the GPU and every decode step fell back to
the CPU, silently. The matvec now exists
(`Q5_0_MATVEC_KERNEL_SRC`, `frink-metal/src/gpu.rs:439`) and
`metal_matvec_kind_name` / `metal_mul_mm_kind_supported` name the same
seven kinds. It is **correct by construction and unmeasured**: there is
no `Q5_0` checkpoint in `benchmarks/suite.json`, so no row in
`RESULTS.md` covers it.

Metal's MoE indexed GEMM (`mul_mm_id`) is narrower still: `Q4_0`,
`Q8_0` and `Q4_K` only.

The GPU column deliberately says **CUDA matvec** and not CUDA GEMM.
`cuda_mul_mm_kind_supported` does hold `Q8_0` and `Q4_0`, and that
kernel has never executed on a GPU, so it is not a tier this table can
promise anything about. See Backends below.

Three caveats that matter in practice:

- **The IQ tiers split, and the split matters if you are choosing a
  quant.** The bottom two rows load and produce correct output, and they
  are slow. That was deliberate. They were added for coverage, because
  before them those tags could not be decoded at all, which ruled out 5
  of the 16 published Unsloth `UD-*` variants. A vectorized path was
  left out rather than written without a golden vector that could tell
  it apart from the scalar one.
- **On an Apple machine the "AVX2 only" row is the scalar row.**
  `IQ1_S`, `IQ2_XXS` and `IQ3_XXS` have x86 kernels and no NEON ones, so
  on ARM they run at the same speed as the scalar tier below them.
- **`I32`, `TQ2_0`, `NVFP4`, `Q1_0`, `Q2_0` and `PQ2_0` are recognized
  and sized, but nothing executes them.** They parse, `frink inspect`
  reports their real footprint, and a checkpoint that needs one stops
  with an error naming the format rather than being quietly skipped or
  silently mis-measured. `TQ1_0` has the CPU trit dot (`frink_quant::
  ternary` is one codec for the two layouts) and no GPU kernel, and no
  real `TQ1_0` checkpoint has been run through it; `PTQ1_0` is the one
  ternary format verified end to end (Bonsai-2-27B, above).

`IQ2_XS`, `IQ2_S`, `IQ3_S` and `IQ1_M` were validated bit-exact against
llama.cpp's own `dequantize_row_*` by linking `ggml-quants.c`, not by
re-reading the spec. They have not been validated end to end on a
published `UD-*` checkpoint.

## Backends

| Backend | What it covers |
|---|---|
| CPU | Dense and MoE. `FRINK_CPU_INT_DOT=1` (Q4_Kx8 / Q8_0x4 / Q5·Q6 int-dot) on suite runs |
| Metal | Dense, MoE, FA-vec, fused MoE encode groups, `mul_mm_id` prefill, quantized KV |
| CUDA | Matvec + resident weights + FFN fuse. A batched `Q8_0`/`Q4_0` GEMM exists and has never run on a GPU (see below) |

Every number on this page was taken on CPU or Apple Metal. CUDA compiles
and runs, has no pinned benchmark host, and has no published timings, so
treat a Windows or Linux install as CPU-only in practice.

A batched quantized GEMM for CUDA (`Q8_0` and `Q4_0` only) is in the
tree and reachable from a wide prefill, and it has **never executed on a
GPU**. Its evidence is a thread-by-thread scalar twin plus a host
harness that compiles and runs the emitted CUDA against a barrier shim
(`crates/frink-cuda/tools/mul_mm_host_check/run.sh`); the hardware test
is `#[ignore]`d with "NEVER RUN" as its reason. That is not a
performance claim, and no row in `RESULTS.md` rests on it.

Paged KV used to be refused on Metal and CUDA, because the paged
attention path there returned fluent wrong tokens. That refusal is
**lifted**: a Metal prefill left K/V on the device and filled the host
cache with placeholders that the paged prefill then copied into the page
store, and the prefill now downloads the real rows for the caller that
reads them. Pinned on hardware by `cargo test -p frink-models --features
metal --test paged_metal_parity -- --ignored`, which gets identical
greedy ids from the paged and contiguous caches on a dense, an MoE and a
sliding-window model. CUDA carries no equivalent hardware run. See
[`CONFIG.md`](CONFIG.md).

Capabilities overview: [`FEATURES.md`](FEATURES.md).
Planned work: [`ROADMAP.md`](ROADMAP.md).
