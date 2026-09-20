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
| Yi (text) | Works (GenericGqa, Neox RoPE), not in suite yet |
| MiroThinker | Works via `qwen3moe` |
| Qwen2-MoE / Qwen1.5-MoE | Loads. Not in the current suite (OLMoE is the MoE entry) |
| Mixtral | In the suite, skipped on 32 GiB Host B (`--fit-host`) |
| MLA (`deepseek2` / `mistral4` / `plm`) | Dense-lead + MoE-after-dense via `MlaEngine`, checked against libllama. **Real checkpoint: PLM-1.8B-Instruct Q8_0** (`frink parity`, through the new MLA arm): tokenizer MATCH, graph 4.5e-5 from llama.cpp on the dequantized f32 file, and the Q8_0 file reads `WRONG` at 3.5e-2 because llama.cpp's own 8-bit activation quantization costs it 3.7e-2 on this graph where frink loses 2.8e-9 (`docs/plans/llama-cpp-gap-inventory.md` §10.1). Fixtures: `deepseek2` in BOTH tensor forms -- the split `attn_k_b` / `attn_v_b` every real export carries (the absorbed attention over a latent cache, `frink_core::mla_absorbed`; refused until 2026-09-12) and the legacy combined `attn_kv_b` -- KL 2.35e-15 and 3.57e-15 (`tests/deepseek2_graphs.rs`), and `plm` (PLM-1.8B) KL 1.87e-13 (`tests/plm_graphs.rs`). The lite DeepSeek-V2 layer counts take the direct-Q form. YaRN as every real DeepSeek-V2 / V3 export declares it is `frink_models::mla_yarn` (the `pe` frequency rewrite, the magnitude, `mscale^2` in `kq_scale`, with `deepseek2.cpp:34-37`'s `/ 0.1` on `yarn_log_multiplier` and `llama-context.cpp:210-213`'s DeepSeek-V2 rule), checked in both generations' shapes and both tensor forms, KL 2.98e-15 / 4.42e-15 / 2.94e-15. Any other scaling type, or `yarn` without `original_context_length`, is refused by name |
| GLM-4-0414 / GLM-Z1 / GLM-OCR (`glm4`) | **Generic path, audited 2026-09-12** (`tests/glm4_graphs.rs`, KL 9.67e-15). It had been sent to the GLM-5.2 MLA loader for four keys `glm4.cpp` never reads, so a real GLM-4-9B-0414 failed on `q_lora_rank`; the graph is plain GQA with Gemma-2's two post norms in Gemma-2's slots and a fused SwiGLU, all of which the generic decoder already served. A GLM-4.1V text tower's `rope.dimension_sections` is refused by name (`frink_models::mrope`): llama.cpp rotates it M-RoPE over converter-permuted weights and its logits differ from the plain file's by 0.72 (measured) |
| GLM-4.5 / 4.5-Air / 4.6 (`glm4moe`) | **Generic path, audited 2026-09-12** (`tests/glm4moe_graphs.rs`, KL 1.51e-15 and 2.41e-15 on the 355B and Air shapes). Its refusal had two lives -- sent to the MLA loader for a `q_lora_rank` it never carries, then named for its pre-FFN norm stored as `post_attention_norm` -- and the second was one row in `norm_sites::PRE_FFN_NORM_IS_POST_ATTENTION_NORM`. A GLM-4.5V text tower's `rope.dimension_sections` rotates NEOX here, and libllama's M-RoPE logits on text positions are byte-identical (measured). No real checkpoint run yet: the smallest export is 106B |
| Gemma-4-E2B | Dedicated `Gemma4Engine` + SPM-style `gemma4` BPE tokenizer + `<|turn>` chat wrap. GGUF: `models/gemma-4-E2B-it-Q4_K_M.gguf` (`unsloth/gemma-4-E2B-it-GGUF`). Suite id `gemma4_e2b_q4km`, Homebrew llama may still lack `gemma4` arch. **Cross-engine evidence since 2026-09-12**: against a libllama built from `.scratch/llama.cpp` (1269cb1, which has `gemma4.cpp`), `frink parity` reads tokenizer MATCH (21 cases x 2) and logits MATCH, KL 5.1e-4 on Q4_K_M with top-10 overlap 10/10 -- the K-quant band, so the graph agrees and the file is not one a WRONG line can be drawn on. |
| gpt-oss | **CPU only.** Attention sinks, alternating sliding-window attention, biased router and the `swiglu_oai` clamp, checked against llama.cpp's own reference logits. Metal stops with an error, because no Metal kernel implements attention sinks. The paged-KV decode path runs it: all three attention arms are bit-identical to their contiguous twins |
| Cohere2 MoE (`cohere2moe`: the 49-layer 30B-A3B) | **Generic path, audited 2026-09-14** (`tests/cohere2moe_graphs.rs`, KL 1.7e-14 LayerNorm, 1.3e-14 RMSNorm, the MTP-block file byte-identical to the trunk's golden in libllama and here, 6.4e-15 softmax with `norm_w`). The `cohere2` graph with routed experts: a layer rotates when it slides OR sits in the dense prefix (`rope_layers::SlidingOrLeadingDense`), `(moe_out + shexp) * 0.5` on a layer with a shared expert (`parallel_dense_ffn::SHARED_EXPERT_SUM_SCALE`), the norm function from which epsilon key the file carries (`norm::NORM_BY_RMS_EPS_KEY`), the window array at TRUNK length (`swa_layers::ARRAY_AT_TRUNK_LENGTH`) |
| Llama 4 (`llama4`: Scout 17B-16E, Maverick 17B-128E) | **Generic path, audited 2026-09-14** (`tests/llama4_graphs.rs`, KL 1.1e-12 on the 16-expert and 128-expert shapes, and the last of 8200 positions across the 8192-position chunk boundary on both the prefill and the row body). Five seams: the CHUNKED window (`frink_models::chunked_swa`, a query sees its own chunk), the literal attention temperature on the unrotated layers (`attn_temperature::LITERAL_ATTN_TEMPERATURE`), a weightless per-head QK norm after RoPE on the rotating layers for every expert count but 128 (`weightless_qk_norm`), the routing weight on the expert's INPUT (`routed_weight_site`, `llama-graph.cpp:1947`) and the interleave step the tensor loader honours (`moe_interleave::INTERLEAVE_STEP_HONOURED_BY_LOADER`). A converter's `sliding_window 0` (all-full-attention MobileLLM) and a zero expert count are refused by name: libllama aborts on the first and refuses the second. CPU only: the fused Metal launches take one window per layer |
| MiniMax | `minimax-01` (MiniMax-Text-01, 456B-A45B) **runs** since 2026-09-20: lightning attention as a recurrent block on the layers `attention.recurrent_layers` / `full_attention_interval` name (`frink_models::lightning`, `AttnShape::Lightning`), plain GQA with partial NEOX RoPE elsewhere, a four-expert softmax MoE on every layer, and the pre-norm residual topology its REQUIRED `residual_scale` multiplies (`frink_models::normed_residual`; the layer input is discarded, which is ONE graph of the 155). KL 1.3e-11 to 1.3e-9 on five fixtures (`tests/minimax_01_graphs.rs`). `minimax-m2` (MiniMax-M2) **runs** on the generic path since 2026-09-14: plain GQA + whole-vector QK norm + partial NEOX RoPE + a sigmoid MoE with `exp_probs_b`, KL 3.4e-15 against libllama on the fixture that had said it was a fixture away (`tests/minimax_m2_graphs.rs`). `minimax-m3` **will not load**: it needs MiniMax Sparse Attention (a per-layer indexer driving its own MSA KV cache), of which frink has only the block-selection rule |
| LFM2 (`lfm2`: LFM2-350M / 700M / 1.2B / 2.6B, LFM2-VL's text tower; `lfm2moe`: LFM2-8B-A1B, 24B-A2B) | **Generic path, audited 2026-09-14** (`tests/lfm2_graphs.rs`, KL 3.2e-12 on the split, fused-QKV and separate-`output` fixtures, 6.0e-13 on the MoE, contiguous, paged and multi-seq). The first hybrid row: `lfm2.cpp:9-11` marks a layer recurrent when `head_count_kv` is 0, and its short convolution (`frink_models::shortconv`) runs at the attention site with its state kept as the layer's KV history. `lfm2moe` is the same graph with leading dense layers and a sigmoid MoE with `exp_probs_b`; a file declaring `attention.sliding_window` is refused by name (no export writes it) |
| Granite 4.0 (`granitehybrid`: H-Micro 3B, H-Tiny 7B-A1B, H-Small 32B-A9B; `granite-hybrid` alias) | **Generic path, audited 2026-09-14** (`tests/granite_hybrid_graphs.rs`, KL 1.9e-13 NoPE dense, 7.9e-13 rotated, 1.0e-13 MoE + shared expert). The first Mamba-2 row: the block runs where attention would be on the zero-KV layers (`frink_models::mamba2`), its state as a `RecurrentState` beside the layer's cache. `rope.scaling.finetuned = false` (every real export) rotates nothing. No prefix-cache reuse and no `--model-draft` on such a model: a Mamba state cannot be rolled back to a middle position |
| Nemotron-H (`nemotron_h`: 8B / 47B / 56B, Nemotron-3 Nano dense) | **Generic path, audited 2026-09-14** (`tests/nemotron_h_graphs.rs`, KL 2.0e-13 plain, 1.4e-12 with the three optional biases, 7.5e-14 separate `output`). One block per layer (Mamba-2, NoPE attention, or the ungated ReLU-squared FFN) under one norm; no prefix-cache reuse and no `--model-draft`, as for every Mamba model. `nemotron_h_moe` (Nemotron-3 Nano 30B-A3B) **runs** too (KL 3.6e-13): a sigmoid MoE of ungated ReLU-squared experts with the router bias, plus an ungated shared expert; a file declaring `moe_latent_size` (Nemotron-3 Super) is refused by name |
| Jamba (`jamba`: AI21 Jamba-v0.1 / 1.5), Mamba (`mamba`: 130M to 2.8B, FalconMamba-7B), Mamba-2 (`mamba2`: Mamba-Codestral-7B) | **Generic path, audited 2026-09-14** (`tests/mamba_graphs.rs`, KL 7.3e-12 / 2.3e-12 / 1.8e-12 with `ssm.dt_b_c_rms` / 3.6e-13). The Mamba-1 block (`frink_models::mamba1`) and the pure recurrent models with no attention anywhere; no prefix-cache reuse and no `--model-draft`, as for every Mamba model |
| Falcon-H1 (`falcon-h1`: 0.5B to 34B) | **Generic path, audited 2026-09-14** (`tests/falcon_h1_graphs.rs`, KL 1.3e-13 plain, 6.2e-13 without `ssm_norm`, 3.2e-13 separate `output`). Attention and Mamba-2 in parallel on every layer (`frink_models::mamba2::PARALLEL_WITH_ATTENTION`); no prefix-cache reuse and no `--model-draft`, as for every Mamba model |
| openPangu-Embedded (`pangu-embedded`: 1B / 7B) | **Generic path, audited 2026-09-14** (`tests/pangu_embedded_graphs.rs`, KL 1.5e-13 on split, fused-QKV and separate-`output` fixtures). A decoder LLM, not an embedding model; `llama.cpp`'s graph plus a required `attn_output.bias` |
| Qwen3.5 dense (`qwen35`: 0.8B / 2B / 4B / 9B / 27B) | **Generic path, audited 2026-09-14** (`tests/qwen35_graphs.rs`, KL 4.1e-13 / 4.1e-13 (the `attention.recurrent_layers` spelling) / 5.7e-13). The gated delta net where attention would be (`frink_models::gdn`, `frink_core::gdn`), gated full attention on every fourth layer, partial IMROPE. The old GDN scaffold (`gdn.rs`, `hybrid_gguf_loader.rs`, 1.8k lines that had never met libllama) is deleted. `qwen35moe` (Qwen3.5-35B-A3B, 122B-A10B, 397B-A17B) **runs** too (KL 2.9e-11, `qwen2moe`'s FFN with the sigmoid-gated shared expert); `qwen3next` (Qwen3-Next-80B-A3B) **runs** too (KL 8.7e-12): the same layers with the V heads grouped over the K heads (`HeadMap::Grouped`) and beta / alpha in one `ssm_ba` projection. No prefix-cache reuse and no `--model-draft`, as for every recurrent model |
| Ternary-Bonsai-2-27B (PrismML; `qwen35` + `PTQ1_0` + folded Hadamard) | **Runs, verified on the real checkpoint 2026-09-18** against PrismML's llama.cpp fork (`frink parity`, first-token KL 2.1e-5 on CPU, 2.3e-5 on Metal decode, 2.2e-6 through the Metal GEMM path; tokenizer MATCH on 1198 tokens, `pre=qwen35`). `PTQ1_0` (ggml type 143: 128 trits in 24 + 2 bytes and an f16 scale, `frink_quant::ternary`) has a scalar CPU dot, a Metal matvec and a Metal simdgroup GEMM (`frink-metal/src/ternary.rs`); `PQ2_0` (142) is recognised and sized, not executed. The `prism.hadamard.*` metadata (block-1024 normalised Walsh-Hadamard with explicit signs, folded into every listed weight; the inverse on `token_embd` rows; the tiled-to-grouped head permutation on `ssm_out`) is `frink_models::hadamard_fold` and `WeightMatrix::Folded`, applied to the activation before every launch and to the embedding row after the lookup. Speed on the M2 Pro, back to back on a quiet box: **43.1 prefill / 11.17 decode** against the fork's 66.8 / 11.46 (0.23.1 was 32.7 / 7.7), and decode is FLAT with context -- 11.17 / 11.19 / 11.16 at 32 / 300 / 600 tokens against the fork's 11.46 / 11.50 -- where before 0.25.0 it sloped, because the attention ran on the host and its work grows with `seq_len`. A four-layer group on this hybrid (one attention layer, three recurrent) is ONE Metal submission with no host step inside it: the recurrent layers run end to end on the device and wait once, the attention layer's tail rides in the next group's command buffer and its projections in the previous one's, and the attention itself reads the sequence's own device KV mirror (`KvCache::metal_attn`). That took a decode token from 192 command buffers to 69 to 20. What remains is measured in `docs/plans/gdn-resident-state.md` and is not plumbing: the token's GPU time is 88.8 ms against the fork's whole token of 86.7-87.3, so the last ~3% is inside the PTQ1_0 matvec |
| Kimi K3 / GLM-5.2 / DeepSeek V4 | Loaders and primitives only. Nothing has been run end to end on a real checkpoint |
| Vision | Finds an mmproj file and warns about it. An `image_url` in a request returns an error |
| MTP / speculative | `--mtp` errors by design. `frink speculative` is prompt-lookup only (an n-gram match over the history, no draft model) and runs on **synthetic random weights**, so the hit rate it prints is not representative of a real drafter. Plan for a real one: [`docs/plans/on-hold/dflash-speculative-decoding.md`](plans/on-hold/dflash-speculative-decoding.md) |
| Embeddings | `/v1/embeddings` for a GGUF decoder (mean/last pool), and for the BERT-family ENCODERS `bert`, `nomic-bert` (nomic-embed-text v1 / v1.5) and `jina-bert-v3` (jina-embeddings-v3), each checked against llama.cpp's own pooled embedding (`tests/bert_family_graphs.rs`; the real-checkpoint comparison is `tests/bert_llama_cpp_parity.rs`, `--ignored`). `bert_gguf_loader::ENCODER_ARCHS` is the table: the three differ from each other in a rotation and an FFN and in nothing else. `/v1/rerank` uses the same encoder with a rank head |

## When a model will not load

Some checkpoints stop with an error instead of running. That is
deliberate. The alternative is worse: a model whose graph Frink only
partly implements will load, run fast, and return fluent text computed
by the wrong maths, and nothing in the output tells you. An error you
can read beats output you cannot trust.

The error always names the reason. Six things cause it:

1. **Frink does not know the architecture.** It is not in the
   capability registry.

2. **Frink knows it and has not implemented it.** `minimax-m3` stops
   with the missing feature named (`llama4` left this list on
   2026-09-14).
   The parallel residual, `x + attn(norm(x)) + ffn(norm(x))`, used to
   be this list's biggest group and is served now
   (`frink_models::parallel_residual`; `gptneox` and `plamo` run on
   it, and `command-r`, `falcon`, `phi2`, `cohere2` and `cohere2moe`
   since), so nothing stops for it any more. `minicpm`
   was on this list and no longer is:
   it never had a different residual, only three multipliers llama.cpp
   applies whether or not the GGUF declares them, and those are
   implemented now (see 4 below).

3. **The file contains weights Frink never reads.** The GGUF reader
   records every tensor name a loader looks up and stops if any are left
   over: *"checkpoint carries N tensor(s) this build never reads, so its
   graph is not the one this build computes."* Unread weights mean the
   file describes a model Frink is not computing. This catches missing
   graph features automatically rather than one at a time, which is how
   `attn_sinks` and `exp_probs_b` were both found. Tensors for parts
   Frink does not claim to run (`mm.`, `v.`, `mmproj.`, `resampler.`,
   `audio.`) are ignored.

4. **The file declares a scale factor Frink does not apply.** These are
   hyperparameters rather than weights, so the check above cannot see
   them, and a checkpoint declaring one would otherwise load cleanly
   while computing a differently-scaled graph than it was trained as.
   `{arch}.logit_scale`, `{arch}.residual_scale`,
   `{arch}.embedding_scale` and `{arch}.attention.scale` stop the load
   unless they hold a value that changes nothing.

   **Granite is the exception now, and the refusal list is derived from
   the same table that says so.** `granite`, `granitemoe` and the
   `granite-moe` alias apply all four, checked against llama.cpp's own
   logits (`crates/frink-models/tests/granite_family_graphs.rs`), and
   `frink_models::scalar_multipliers` implements them once for all
   three rather than once per architecture. `residual_scale` was the
   expensive half: it multiplies both branch outputs of every layer,
   which meant collapsing eighteen hand-written residual adds in
   `decoder.rs` onto one function that takes the scalar as a parameter,
   and fencing the fused Metal launches -- which fold the residual in on
   device with no uniform for a multiplier -- off any model that
   declares one. Getting half of that right gives you a model that
   loads, runs, and returns wrong answers with nothing in the output to
   say so, which is what the refusal existed to prevent and what the
   fixtures now check.

   One Granite case still stops, a metadata-only fact:
   `{arch}.logit_scale` is REQUIRED for Granite (`granite.cpp:7` reads
   it with no default), so a file omitting it is refused rather than
   defaulted to 1.0 -- llama.cpp cannot load such a file either.
   `{arch}.rope.scaling.finetuned = false` WAS refused outright, because
   `granite.cpp:33-35` reads that key as a switch for **RoPE itself**
   and the generic decoder had no way to express "unrotated"; it has
   one since `gpt2` closed (`RopeLayers::Never`), and since 2026-09-14
   the file runs unrotated as llama.cpp runs it
   (`frink_models::rope_finetuned`). Every Granite-4.0 hybrid export
   writes the key false (`conversion/granite.py:253-256`), which is what
   turned the refusal into a row.

   **MiniCPM runs now**, and it is the case a key-presence gate could
   never have caught. llama.cpp assigns its three multipliers
   (`minicpm.cpp:5-7`: 12.0, `1.4/sqrt(n_layer)` and `256/n_embd`) and
   only THEN lets the file override them (`:12-14`), so an export
   declaring nothing is still scaled three ways and the metadata looks
   ordinary -- which is why the row was refused on its architecture
   string. Its graph is `llama_model_granite::graph` verbatim
   (`models.h:1594-1601`), so the fix was a DEFAULTS field on the same
   table, and the fixture that evidences it declares no scaling key at
   all: a fixture that declared them would pass with or without the
   hook. A second fixture declares all three and pins that the FILE
   still wins, which one fixture cannot see
   (`crates/frink-models/tests/minicpm_graphs.rs`). MiniCPM reads no
   `attention.scale` (`minicpm.cpp:3-24` has no such key), so that one
   is still refused for this row, by the derived list -- measured, too:
   libllama's logits for a file declaring it are byte-identical to the
   file without it.

   `command-r` runs since 2026-09-12: its `logit_scale` is a MULTIPLY
   the graph skips at zero or absent (`LogitScaleUse::AsIsOptional`),
   and the parallel residual over the weighted LayerNorm that was its
   real blocker is served (`tests/command_r_graphs.rs`, KL 1.0e-15).
   `cohere2` (Command-R7B) followed on 2026-09-14: its sliding-only
   rotation is `rope_layers::SlidingOnly`, the `exaone-moe` rule, which
   the first census had missed (`tests/cohere2_graphs.rs`, KL 1.0e-14).

5. **The architecture encodes position some other way than RoPE.** The
   generic decoder rotates every Q and K head of every layer unless
   `rope_layers` says otherwise. `gpt2` and `starcoder` use a learned
   absolute position table instead, and run since 2026-09-14
   (`frink_models::position_embd`, `RopeLayers::Never`); `mpt`,
   `refact`, `bloom`, `jais` and Baichuan-13B use ALiBi, and run since
   the same day (`frink_models::alibi`, `frink_core::alibi`). This
   was the least visible failure of the five: `bloom` and `refact`
   hardcode their ALiBi slope in llama.cpp's own loader and carry no
   GGUF key at all, and `baichuan` tells its 7B (rotating) from its
   13B (ALiBi) by layer count alone, so neither check 3 nor check 4
   could ever have seen them; the registry decides, and the two tables
   that must agree about the layer count (the bias, the absence of
   rotation) are pinned against each other.

6. **Nobody has ever verified this architecture against llama.cpp.**
   The shared generic-GQA decoder is a *guess*: it assumes plain GQA
   because nothing said otherwise, and that guess was already wrong for
   the five architectures in cause 5. So the generic path is opt-in.
   An architecture reaches it only if there is a benchmark row, a pinned
   logit comparison against real `libllama`, or a fixture; **76** do
   today (`llama`, `qwen`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`,
   `olmoe`, `olmo2`, `chatglm`, `deepseek`, `bailingmoe`, `bailingmoe2`,
   `seed_oss`, `maincoder`, `hunyuan-moe`, `hunyuan-dense`, `ernie4_5`,
   `ernie4_5-moe`, `internlm2`, `xverse`, `baichuan`, `exaone`,
   `exaone4`, `exaone-moe`, `smollm3`, `plamo3`, `granite`, `granitemoe`,
   `granite-moe`, `minicpm`, `olmo`, `dbrx`, `grok`, `arcee`, `deci`,
   `openelm`, `afmoe`, `laguna`, `mellum`, `apertus`, `step35`,
   `mistral3`, `smallthinker`, `bitnet`, `mimo2`, `nanbeige`, `talkie`,
   `arctic`, `glm4moe`, `glm4`, `orion`, `nemotron`, `starcoder2`,
   `codeshell`, `jais2`, `stablelm`, `gptneox`, `plamo`, `command-r`,
   `falcon`, `phi2`, `cohere2`, `phimoe`, `gpt2`, `starcoder`, `refact`,
   `bloom`, `mpt`, `jais`, `minimax-m2`, `gemma`, `gemma2`, `gemma3`, `phi3`,
   `gpt-oss`, `dots1`).
   The other **2** stop with `UnauditedArchitecture`. (`plm` is not in
   the 54 and not in the 2: it runs on the MLA engine, `DedicatedOnly`,
   with its own golden.)
   `FRINK_ALLOW_UNAUDITED_ARCH=1` runs one anyway; compare the output
   against llama.cpp yourself before you trust it.

**Gemma-2-27B, Gemma-3-4B/12B/27B: corrected 2026-09-02.** Those four
sizes were quietly wrong until then, in two ways that both produce
fluent text. The 27B checkpoints took `1/sqrt(head_dim)` as their
attention scale where llama.cpp takes `1/sqrt(n_embd/n_head)` for that
size alone, selected by layer count; and Gemma-3 4B and up applied
linear RoPE scaling to the sliding-window layers llama.cpp ropes
unscaled, because `rope_theta` is per layer while `rope_freqs` was
global. Gemma-3-1B is the one size with no `rope_scaling`, and it was
the audited fixture, which is why neither was visible.

The evidence is llama.cpp's source and header-driven loader tests, NOT a
logit comparison: no checkpoint of those sizes exists on the development
host. A `frink parity` run against one is what would settle it.

A cost came with it and has since been paid back. Gemma-3 4B and up
briefly left the fused Metal dense stacks, because those took one
`freq_factors` slice for a whole run of layers while these models need
one per layer: correct, and slower. The stacks now carry a per-layer
`LayerRope` holding the base and its divisors together, so supplying one
without the other does not compile, and Gemma-3 4B+ is back on the fused
path (#63).

**That last part is proven on synthetic layers, not on a checkpoint.**
No Gemma-3 4B/12B/27B GGUF exists on the development host, and
Gemma-3-1B would prove nothing because it declares no rope scaling. The
Metal tests build a two-layer stack carrying Gemma-3's two answers
(base 1e6 divided by 8, and base 1e4 divided by 1) and assert the fused
and per-layer launches agree bit for bit, then re-run the old bug on
purpose so neither can pass vacuously. `frink layer-divergence` on a
real gemma-3-4b, plus `frink parity` at Q8_0, is what would settle it.

No published number changes:
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) carries Gemma-3-1B
only, and 1B is neither 27B nor rope-scaled, so it is untouched by all
of this. The speed recovery from returning to the fused path is
unmeasured, because measuring it needs a quiet host.


### What "unaudited" costs you, per architecture

"Unaudited" is not one thing. None of the 2 is a fixture or a single
match arm away any more: they need an attention implementation or a
reading nobody has done, and the refusal says which, with the
`llama.cpp/src/models/*.cpp` line that decides it:

| Class | Means |
|---|---|
| `FIXTURE-AWAY` | Frink already computes this graph. What is missing is evidence. |
| `ONE MATCH ARM` | One small, named piece: an activation, a norm slot, a routing flag, an ordering. |
| `NEW CODE` | A different attention or residual structure. Not close. |
| `UNKNOWN` | Reading both trees did not settle it. The message says what would. |

All 2 have now been read on both sides (`frink_models::capability`,
pinned by `crates/frink-models/tests/unaudited_triage.rs`). The
distribution is the headline answer to "how far is Frink from llama.cpp
on models":

| Class | Count |
|---|---|
| fixture-away | 0 |
| one match arm | 0 |
| new code | 1 |
| unknown | 1 |

**Both cheap classes are empty.** `gemma` was the last fixture-away row
and `chatglm` the last one-match-arm row; nothing still refusing is one
fixture or one arm away. That is a better answer than the count alone:
the cheap wins are spent, and what is left is 1 row needing a
different graph plus one name nobody can get a file for.

It was 47 until the triage itself removed one. Reading
`src/models/minicpm3.cpp:5-6,41-46` showed `minicpm3` requires
`q_lora_rank`/`kv_lora_rank` and the DeepSeek-2
`attn_q_a`/`attn_q_b`/`attn_kv_a_mqa`/`attn_kv_b` tensor set: it is an
MLA model that was never on the generic path, so it now refuses by name
(naming both the MLA tensor set and MiniCPM's three hardcoded
multipliers) rather than as unaudited. The count going down for the
right reason.

**Fixture-away (0).** The class started at 9 and is empty.
`internlm2`, `exaone`, `ernie4_5`, `bailingmoe2`, `xverse`, `baichuan`
(the 7B then; the 13B followed on 2026-09-14 with ALiBi) and `plamo3`
were admitted with libllama-golden fixtures on 2026-09-03, `gemma`
followed, and `chatglm` left the class the other way: an attempt to
build its fixture found the fused `attn_qkv.bias` that every real
ChatGLM2/3 export carries and frink dropped, so it became ONE MATCH
ARM before closing as that.

**One match arm (0).** `chatglm` was the last row here and closed on
2026-09-10. Its arm was the fused `attn_qkv.bias`: llama.cpp's
`create_tensor_qkv` puts the bias where the weight is, and frink split
the fused weight while reading the bias only under the split
`attn_q.bias` names, so ChatGLM2/3's `add_qkv_bias: true` was dropped
and all three projections ran unbiased. Both halves now come out of one
decision in `qkv_fused`, sliced by the same spans.

**`qwen` came with it, and needed a second arm nobody had named.** The
`chatglm` verdict predicted the bias would close both rows. The bias
really is shared -- `qwen.cpp:28` marks it REQUIRED, stronger than
chatglm's optional one -- but building the fixture found that
`qwen.cpp:33-35` also sizes every FFN matrix at `n_ff / 2`, because
Qwen-1's `intermediate_size` counts gate and up together. That costs no
logits (frink loads the dense FFN by tensor name and uses each
matrix's own shape) and still made `expert_ffn_dim` twice the real
width, which is what every memory estimate prices the FFN from. Both
rows are audited against libllama now, and a test compares the declared
FFN width against the matrices that load, over every dense fixture.

The other seven one-match-arm rows had closed earlier. `seed_oss` and
the gpt-oss norm slot;
`deepseek` and top-k renormalisation; `bailingmoe` and a
`leading_dense_block_count` llama.cpp reads but never uses;
`hunyuan-moe`, `maincoder` and `hunyuan-dense`, which all wanted the
same flag (QK norm applied *after* RoPE rather than before, plus, for
`hunyuan-dense` alone, the NTK-alpha RoPE base rescale); and
`ernie4_5-moe`, whose interleaved MoE layers landed as a REFUSAL rather
than an implementation, because llama.cpp's own tensor loader
(`ernie4-5.cpp:49`) has no interleave step in it while its graph
(`ernie4-5-moe.cpp:64`) does, so an interleaved checkpoint cannot be
loaded by llama.cpp either. The step every published ERNIE-4.5 MoE
checkpoint carries is 1, and that is what Frink runs and pins against
libllama.

**New code (1).** A different attention or residual structure. The
recurring shapes, which is how the column emptied:

The column moved for the first time on 2026-09-10, three times: 26 to
24, 24 to 21, then 21 to 20, and on 2026-09-11 seven times more, 20 to
19, 19 to 17, 17 to 14, 14 to 12, 12 to 11, 11 to 9 and 9 to 8, and on
2026-09-12 seven times, 8 to 7, 7 to 6, 6 to 5, 5 to 4, 4 to 3, 3 to 2 and 2 to 1. The first two took several rows at
once, and for the same reason -- each found ONE cause behind several
refusals. The fourth did too, and the count hides it: the per-layer
RoPE gate closed THREE refusals and only one of them (`exaone-moe`) was
ever in this column. The fifth is a different lesson, below: `grok` and
`dbrx` each closed by extending a seam that had landed the day before,
and the clamp one of them needed closed a refusal-by-name on a third
row. The sixth is two lessons at once: `deci` and `openelm` closed
together on a seam whose reach was MEASURED across all 155 graphs
before it was built, and `arcee` closed alone because the constant that
said it shared a cause with `plm` had been read from one file and not
the other. The seventh is the sixth's lesson applied to the sixth's
leftovers: the per-layer shape seam had narrowed `afmoe`, `laguna` and
`step35` to the same last word, `wqkv_gate`, and reading the three
graphs side by side found one op with two free parameters rather than
one graph -- two rows closed on it and the third said so. The ninth is
the seventh's leftover and a new pair: `apertus` and `step35` closed
TOGETHER on the per-layer ACTIVATION PARAMETER seam, and the question
"are xIELU's four arrays and Step-3.5's clamp arrays one seam" was
answered by reading both graphs first -- one plumbing question, `layer
il runs its FFN activation with these scalars`, and two activation
bodies, with the clamp's routed-versus-dense SITE the one thing the
second needed of the plumbing that the first did not. The tenth is the
reach measurement coming back with "one": `mistral3` closed ALONE on
the per-position attention temperature because the other two graphs
that build the input are on other engines, and its verdict's second
half, one GGUF key, found a defect in every YaRN checkpoint on the
generic path. The eleventh is the same measurement with the same
answer for a different reason: `smallthinker` closed ALONE on the
router operand because the reach was counted over all fifty-nine
`build_moe_ffn` call sites first, and the three other graphs that pass
a precomputed `probs_in` share the MECHANISM and not the cause. The
twelfth is the smallest reach there is: `bitnet` closed ALONE on two
norm slots that one graph of 155 creates, and the seam is a `bool`
because there is no second shape to name. The thirteenth is the
biggest row of the year and the one the KV cache was built without:
`mimo2` closed ALONE on a V head width that differs from K's, and the
reach came back as one generic-path converter plus the MLA engine,
which had carried the pair since it existed. The fourteenth is the
verdict's own last sentence taken literally: `nanbeige`'s said "it is
the copy that has no home", and the closure gave the copy no home
either -- it is a mapping. The fifteenth is four seams landing for one
row: `talkie` needed four things and each was one graph of 155, so
none could be built for anything else, and all four landed together.
The sixteenth closed on an engine that already had its attention:
`plm`'s three differences from DeepSeek-2 became one table, and the
MLA engine got its FIRST libllama golden, which it had run every
DeepSeek-V2 without. The seventeenth is a reach measurement coming
back with TWO: `arctic`'s dense FFN summed with its experts is Grok-2's
shape too, so one table closed the row and lifted a refusal by name.

`arctic` closed on `frink_models::parallel_dense_ffn` and a third
`RouterInput` variant. `arctic.cpp:118-154` runs a dense SiLU FFN
sized `{n_embd, n_embd}` (`:38-42`) on `ffn_norm(ffn_inp)`, the router
AND the experts on `ffn_norm_exps(inpSA)` -- the layer INPUT under a
SECOND per-layer weight (`:45,135-152`) -- and sums the two (`:154`).
Every `build_moe_ffn` graph that also reads a dense `ffn_up` was
listed (2026-09-12): all but two use the triple on their leading dense
layers or as `_shexp`, and the two that SUM it with the routed output
are `grok.cpp:171-184` (Grok-2: the same `cur`, GELU, `sqrt(2)/2` on
the sum) and `arctic.cpp`. So `PARALLEL_DENSE_FFN_ARCHITECTURES` is
two rows and two columns -- `DensePresence::{Required, Optional}` and
`sum_scale` -- served through the shared-expert slot, which was
already a dense FFN on every token added to the routed sum with the
architecture's dense activation; what it lacked was the tensor names
and the scale, which is why the module had been a refusal.
`MoeWeights::parallel_sum_scale` is applied to the whole branch beside
`apply_down_scale` at every site. `grep -l FFN_NORM_EXPS src/models/
*.cpp` is `arctic.cpp` alone, so the operand is
`RouterInput::NormedLayerInput`, carrying the one fact that
distinguishes it from `smallthinker`'s (`experts_read_router_operand`);
`Decoder::router_operand`, the one constructor, norms the layer input
where `attn_norm` is applied, and the combine bodies take the routed
operand and the dense operand as TWO arguments. Every fused Metal MoE
launch refuses the model (shared experts on every layer; the router
predicate). KL 6.23e-14, and the same golden for a file declaring
`expert_weights_scale = 2.5`, which `arctic.cpp:3-14` never read
(libllama byte-identical, measured); Grok-2 KL 3.28e-10 at the
GELU-table tolerance. One measurement rather than a sabotage:
`grok.cpp:180` scales the sum and `:186` RMS-norms it, and an RMSNorm
is invariant under a positive scalar up to eps, so the `sqrt(2)/2`
moves libllama's own logits by 2.5e-4 and no more; the test pins that
rather than pretending to see it. Sabotaging the routed operand at the
row site and at the batch site each turns the golden red.

`plm` closed on `frink_models::mla_arch` and `frink_models::
mla_q_proj`, on the MLA engine. `plm.cpp:84-166` is `deepseek2.cpp`'s
naive MLA branch line for line: a per-head nope / pe split of Q, the
compressed KV RMS-normed and re-expanded through `attn_kv_b`, ONE
shared roped key repeated onto every head, `kq_scale =
1/sqrt(n_embd_head_k)`. `grep -l ATTN_KV_A_MQA` over all 155 graphs is
six files and `plm` is the only one on no engine, so the question was
what it needs of the engine it belongs on, and the answer is three
columns of one table (`MLA_ENGINE_ARCHS`), not a second engine: a
DIRECT `attn_q` (`plm.cpp:32`; `deepseek2.cpp:104-115` creates the same
when `q_lora_rank == 0`, and `:8,11-13` decide that from the LAYER
COUNT -- 27, 26, or 48 with a 128256 vocabulary -- BEFORE reading the
key, so `MlaQProj` is an enum the forward pass cannot reach without the
answer and a lite file's key is dead metadata, as upstream), an ungated
`LLM_FFN_RELU_SQR` dense FFN (`:181-187`; `GluAct::ReluSqr` with the
gate aliased as the generic loader does for `arcee`, carried ON
`MlaDenseFfn` so the body cannot run it through SwiGLU), and a tied
lm_head the graph never reads an `output.weight` for (`:23-24`;
libllama REFUSES a file carrying one -- `done_getting_tensors: wrong
number of tensors; expected 30, got 29`, measured on
`plm_decoy_output_tiny.gguf` -- so frink refuses it too rather than
prefer the decoy). The head widths come from `attention.key_length` /
`value_length`, because `llama-hparams.cpp:259-265` fall back to them
when the `_mla` keys are absent and that is what `conversion/plm.py:
16-17` writes. KL 1.87e-13, max logit delta 1.67e-6. Building it found
two things in the engine: it REQUIRED `attention.q_lora_rank`, so every
DeepSeek-V2-Lite, GigaChat3-10B-A1.8B and Kanana-2-30B-A3B export
failed on a key llama.cpp never reads for those layer counts (a
27-layer synthetic file that carries the key loads direct now, and the
same file with the low-rank pair is refused naming the stray tensors);
and it read no `rope.scaling.*` at all, so every real DeepSeek-V2 / V3
export -- all YaRN -- would have run at factor 1 with `kq_scale`
missing `deepseek2.cpp:312-319`'s mscale term. That is a refusal by
name now, with the lines; `none` is served, because `plm` and the
dense DeepSeek exports write it.

`talkie` closed on four seams. Every `build_norm` in `talkie.cpp` is
`(x, nullptr, nullptr, LLM_NORM_RMS)` -- the embeddings before layer 0
(`:50`), `attn_norm` (`:68`), the K norm (`:90`), `ffn_norm` (`:110`),
the final norm (`:137`) -- so the file carries no norm tensor at all;
`NormOp::RmsNoParams` is the RMS twin of OLMo-1's `LayerNormNoParams`,
through the same `NormFunction` table, so no site asks the file for a
weight it does not have. `attn_q_norm` is `{1, n_head}` (`:26`): RMS
over each head, then ONE scalar per head, applied after RoPE with a
weightless per-head K norm beside it (`:82-91`) --
`QkNormStyle::PerHeadScalar`, decided by architecture because the
weight's length (`n_head`) is ambiguous with `head_dim` on a file
where the two agree. (Building it measured something llama.cpp's own
comment does not say: a per-head RMSNorm is invariant under RoPE and a
per-head scalar commutes with it, so "reference applies qknorm after
rope" is honoured and unobservable on this shape, unlike hunyuan's
per-element weights.) The embedding skip stream: the normed embedding
is kept as `embd_skip` and every layer adds `embd_skip * out_scale`
after its FFN residual (`:52,123-126`; `blk.N.layer_output_scale`) --
`frink_models::skip_stream`, one `bool` for both halves, the norm at
the ONE embedding site and the add at the end of BOTH FFN bodies,
through an `Option<SkipStream>` argument so no body can be reached
with the question unasked. And the two `{1}` companions its converter
writes on every export (`conversion/talkie.py:26-31`),
`attn_output.scale` and `ffn_down.scale`: `build_lora_mm` multiplies
them onto `wo` and `down`, and `frink_models::weight_scales` now
SERVES exactly those two for any architecture
(`AttnWeights::o_scale`, `MoeWeights::down_scale`) while still refusing
the rest by name. `logit_scale` is REQUIRED and multiplied
(`MultiplierSupport::TALKIE`, the `grok` use). Every fused Metal launch
refuses the model. Two libllama-golden fixtures
(`tests/skip_stream_graphs.rs`): KL 6.43e-14 for the converter's
shape, 1.47e-14 without the gains, whose golden differs -- and dropping
the gains from the first file lands exactly on the second's golden,
which is the measurement that they are applied and not tolerated.
Zeroing `out_scale` and setting the Q gains to one each diverge;
sabotaging the skip add turns four tests red.

`nanbeige` closed on `frink_models::layer_loops`. `nanbeige.cpp:6-12`
read `num_loops` and `skip_loop_final_norm`; with `n_loops > 1`,
`:19-31` set `n_layer_all = n_phys * n_loops` and copy every per-layer
array from physical layer `i` to each logical slot `i + j * n_phys`,
`:47-66` create tensors for the physical layers only, `:69-73` alias
`layers[i + j * n_phys] = layers[i]`, and `:167-175` norm the residual
with `output_norm` after the last layer of every pass but the final
one, unless the flag skips it. One graph of 155 reads either key
(measured). The weights are shared and the KV is not, and the seam
says exactly that: `Decoder::layers` stays the physical vector the
loader filled, `ModelConfig::n_layers` is the LOGICAL count every KV
cache, per-layer table and budget is sized by (`n_layer_all`
upstream), `Decoder::layer_for(l)` / `physical_index(l)` are the ONE
mapping from a logical index to its weights, its gpt-oss side-table
row and its residency plan, the three host bodies iterate logical
indices and ask it, `LayerShapes::replicated` is the copy `:24-26`
makes of the arrays, and the loop norm sits at the end of BOTH FFN
bodies so every caller of either gets it. The fused Metal launches
refuse a looped model, because each indexes weights and KV buffers with
one `l`. Three libllama-golden fixtures (`tests/layer_loop_graphs.rs`;
libllama prints `n_layer = 4` for the two-block file): KL 3.06e-13
(two passes over two layers, loop norm between them and NOT after the
last), 4.05e-13 (`skip_loop_final_norm`), 8.70e-13 (`num_loops = 1`,
the plain Llama every non-looping export is). Skipping the loop norm,
adding it where the file skips it, and running the physical layers
once each diverge; the batched body agrees with the row body over
twelve positions; sabotaging the mapping turns four tests red.

`mimo2` closed on `frink_models::kv_head_dims`. `conversion/mimo.py:
154` writes `attention.value_length` from `v_head_dim` apart from the
`attention.key_length` the base converter writes from `head_dim` --
`192` and `128` on MiMo-V2-Flash, the same on V2.5 -- and `mimo2.cpp:
47-48,132-140,152-154` size and view K and V separately, with `wo` at
`n_embd_head_v * n_head` (`:52`). Fourteen converters write
`value_length`; three write it apart from `key_length`, and two of
those are MLA (`deepseek.py`, `plm.py`), on the engine that has taken
two widths since it existed. Eighty-nine graphs assert the two equal.
So the seam admits the pair for ONE architecture by table and keeps
refusing it, naming the assert, for everyone else. `ModelConfig::
v_head_dim` is an `Option<usize>` that is `Some` only when the widths
differ -- a second `usize` beside `head_dim` would have been two fields
that must agree with nothing enforcing it, and the first test that
mutated `head_dim` proved it -- and `v_head_dim()` is the one accessor.
Every consumer took the V width: `KvCache` and `PagedKvStore` size and
index V by it (`new_split`; the K width is `head_dim` as before), the
three contiguous single-query kernels -- plain, windowed, with sinks --
collapsed onto ONE `causal_gqa_attention_row` that accumulates over it
(they were one loop varied by a window bound and a sink term, and the
V width would have been a fourth decoration to add to each), the paged
kernel reads it off the store, the batched prefill kernel's PV tile
takes its own offset and stride, `check_gqa_projection_widths` sizes
`v_proj` and `o_proj` by it, `qkv_fused::FusedQkvRows` cuts the fused
`attn_qkv` at it, the two batched host bodies split V by it, and the
KV budget prices K and V separately. What refuses: every fused Metal
launch through `metal_can_serve_model` (one head width for the KV
buffers, the attention tile and the `wo` fold), the CUDA resident hook
(now reached only for a plain full-attention layer at one width), the
slot file and the KV block file (one `head_dim` in each header). The
row's second half is `attention.value_scale` (`:14-17,180-183`;
`0.707` on every real export), one reader of 155 measured over all of
`src/`, applied after `wo` in the one attention tail
(`frink_models::attn_value_scale`). Three libllama-golden fixtures
(`tests/split_kv_head_dim_graphs.rs`), each carrying everything a real
export carries -- the per-layer `head_count_kv` array, the per-layer
window array with `rope.freq_base_swa`, sinks on every layer, sigmoid
routing with `exp_probs_b`, partial NEOX RoPE over the 12-wide K head,
MoE on every layer: KL 5.42e-15 for the converter's fused `attn_qkv`
(K rows at 12, V rows at 8), 5.42e-15 for the split spelling (libllama
byte-identical for the two files), 3.49e-15 without the value scale.
A V width read as K's is refused naming the tensor; the value scale
dropped or added diverges; the batched body agrees with the row body
at twelve positions. Building it found two things. `mimo2.cpp:227`
passes the SIGMOID literal into `build_moe_ffn`, so the file's
`expert_gating_func` is never read: parsing every `build_moe_ffn` call
in all 155 graphs, three pass the SIGMOID literal (`llama4`, `mimo2`,
`nemotron-h`), twenty-six SOFTMAX, nineteen the hparam, and the
loader's `GATING_LITERAL_ARCHITECTURES` carries the one on this
engine. And the bisection that found the last 2e-3 of KL found a
generic defect: frink honoured `expert_weights_scale` for EVERY
architecture, while llama.cpp reads the key in twenty per-architecture
loaders and nowhere else -- the fixture declares `2.5`, `mimo2.cpp`
never reads it, and libllama's golden is unscaled. `EXPERT_WEIGHTS_
SCALE_READERS` and `EXPERT_WEIGHTS_NORM_READERS` in the loader are the
readers, measured (eight and seven on this path), and a file carrying
either key on any other architecture now gets the graph's literal, as
upstream. No real export of a non-reader writes either key, so no
published checkpoint changed; a hand-written one would have.

`bitnet` closed on `frink_models::sub_norms`. `bitnet.cpp:24,36`
require `attn_sub_norm` `{n_embd}` and `ffn_sub_norm` `{n_ff}`, two
RMSNorms INSIDE the sublayers where the generic decoder's four sites
are all outside them: `:101-106` norm the attention output -- the
concatenated heads after the V sum -- BEFORE `wo`, the other side of
that matmul from Gemma's `post_attention_norm`, and `:127-141` call
`build_ffn` with a NULL down projection, norm the `silu(gate) * up`
product, and apply `ffn_down` by hand. `grep -l` over all 155 graphs
for either tensor is `bitnet.cpp`; that is why `ModelConfig::
block_sub_norms` is a `bool` and not an enum. It has two readers: the
loader REQUIRES the pair on it (and refuses the FFN one on a routed
layer, where `build_moe_ffn` has no such site), and `metal_can_serve_
model` refuses every fused launch on it, since none has a norm at
either site; the per-layer fused attention loses its view of the layer
through the exhaustive destructure in `metal_attn_view` as well. The
arithmetic landed where the tails already were: `attn_out_to_residual_
rows`, the one attention tail, and `frink_moe::run_expert_sub_normed`,
which shares its gate/up half with `run_expert` through a new
`expert_activated` and CANNOT reach the fused on-device SwiGLU, because
that kernel runs `down` itself with nothing between; `dense_ffn_batch`
applies it per row and skips its fused batch kernel for the same
reason. One libllama-golden fixture, KL 1.88e-14, with the norm weights
drawn AWAY from one so that skipping either norm, applying either with
unit weights, or reading `attn_sub_norm` as the post-attention norm
each diverges by orders of magnitude; a test measures all five. Two
more things the row pinned: the LM head is `tok_embd` unconditionally
(`:164`; no `output` tensor is created), which the tied-embedding
fallback already served, and `conversion/bitnet.py:19-20` writes
`rope.scaling.type = linear` at factor 1.0 on every export, which
resolves to no correction. The verdict's third sentence became a
refusal by name instead of an unread-tensor error: `bitnet.cpp:27-43`
create OPTIONAL per-projection `.scale` tensors, `build_lora_mm`
multiplies each projection's output by them (`llama-graph.cpp:
1492-1494`), and since `llama-model.cpp:1355-1440` a generic pass
creates `.scale` and `.input_scale` companions beside EVERY
architecture's projections (the NVFP4 converter writes them). The
current BitNet converter folds the scale into the ternary weights and
writes none, older exports carry them, and libllama's logits for a
fixture with seven `2.0` scales differ from the unscaled file's from
the first value (measured). `frink_models::weight_scales` refuses any
file carrying either suffix before the unread-tensor gate, which
`FRINK_ALLOW_UNKNOWN_TENSORS=1` could have talked past into a model
running every projection at the wrong magnitude. What the row does NOT
give you yet is a real BitNet-b1.58-2B-4T: the published GGUFs are
`i2_s` (the bitnet.cpp fork's type) or `TQ1_0` / `TQ2_0`, which
`frink-gguf` inspects and refuses at execution; a Q8_0 or F16
re-export of the same weights runs.

`smallthinker` closed on `frink_models::router_input`, the last of
the three things its verdict named and the one no seam had touched.
llama.cpp's `build_moe_ffn` (`llama-graph.cpp:1914-1948`) computes the
router logits from the SAME normed input the experts read unless the
caller hands it a precomputed `probs_in`; every `build_moe_ffn(` call
in all 155 graphs was parsed for that argument before a line was
written. Four of fifty-nine pass one: `smallthinker.cpp:111` computes
the logits from `inpL`, the residual stream as it ENTERS the layer,
before `attn_norm` and before attention; `grovemoe.cpp:133` routes on
the normed FFN input (the default) and precomputes only to share the
logits between its two expert banks; `gemma4.cpp:289-294` and
`nemotron-h.cpp:210-232` route on something else again and are on
their own engines. So `RouterInput` has two variants and one table
row, the operand is captured in ONE function (`Decoder::router_operand`)
at the point each host body applies `attn_norm`, and it is carried as
LOGITS rather than as a vector, so the post-attention residual -- the
same `Vec`, mutated in place -- cannot be handed to the router by
mistake. Every Metal path that runs the router on the GPU reads
`normed2` and nothing else, so `gpu_router_matches_host_routing`, the
predicate they already share, answers false for the row. The second
thing was the experts: `LLM_FFN_RELU` with a REAL gate (`:62,158`,
`ggml_reglu_split`, `relu(gate) * up`), which the verdict had called
one match arm because `GluAct::Reglu` already existed -- it had served
`arcee` by ALIASING the gate to the up matrix, so `ffn_is_ungated` said
"no gate on disk" for the variant, and a SmallThinker loaded through
it would have dropped its gate tensors and computed `relu(up) * up`.
`FfnActivation::ReluSqr` (ungated, `arcee`, `plm`) and
`FfnActivation::Reglu` (gated, `smallthinker` alone; `t5.cpp`'s two
`LLM_FFN_RELU` hits are `build_ffn` with a NULL gate on the
encoder-decoder engine) are two variants now, and a test pins that
`ffn_is_ungated` and `layer_ffn_acts` agree about the gate for every
variant. The third was `n_swa`: `smallthinker.cpp:4-8` reads
`attention.sliding_window`, tests it for `> 0`, and on that branch
assigns 4096 -- the value it just read is a flag and then overwritten
-- so `capability::swa_window_override` has a third answer, `Pin`,
beside `Honour` and `Drop`, and `swa_disabled_by_arch` is derived from
it. The fixture declares 3, narrower than its six-token prompt, and
libllama's logits for that file and the same file declaring 4096 are
BYTE-IDENTICAL, which is the measurement that the pin is upstream's
behaviour; honouring the declared value moves frink's logits by far
more than the tolerance and a test says so. Three libllama-golden
fixtures (`tests/router_input_graphs.rs`): KL 1.13e-14 (window
declared, sigmoid gating, NoPE on layers 0 and 4 -- the FIRST layer of
each period, the other phase from smollm3's), 1.27e-14 (no window, every
layer rotated, softmax gating: the converter's other arm), 6.56e-15 (a
hand-written `sliding_window_pattern = 2` with `rope.freq_base_swa`,
pinning that the SWA period comes from the key while the NoPE step
stays the literal 4). Routing on the normed input picks different
experts on this fixture and a test pins that divergence, so the seam
cannot be silently reverted to the default. Building it collapsed the
THREE batched FFN tails (the Metal-prefill arm and the host arm of the
prefill body, and the multi-sequence body) onto one
`Decoder::ffn_block_batch`, because the seam needed an eighth fact in
all of them and they differed by a gpt-oss branch and an FFN-free check
that were present in one and unreachable in another.

`mistral3` closed on `frink_models::attn_temperature`, and the count
that mattered was taken before the seam was written: `grep -ln
'attn_temp\|temperature_scale\|build_inp_attn_scale' src/models/*.cpp`
over all 155 graphs is `mistral3.cpp`, `llama4.cpp` and
`deepseek2.cpp` (plus three false hits: `grok.cpp:23` reads
`temperature_length` and applies it nowhere, `dflash.cpp:133` /
`deepseek4.cpp:124` name a hyper-connection TENSOR, `plamo3.cpp:140`
is a local). All three multiply Q by the same `[n_tokens]` input
`llama-graph.cpp:163-167` fills with `log(floor((pos + offset) /
floor_scale) + 1) * scale + 1`, AFTER RoPE and BEFORE `build_attn`
with `kq_scale` untouched; what differs is where the three constants
come from -- `mistral3.cpp:5,14-17` reads `attention.temperature_scale`
and floors on `hparams.n_ctx_orig_yarn`, `deepseek2.cpp:46-47` reads
the same scale with `attention.temperature_length` as the floor,
`llama4.cpp:15-17` seeds 0.1 / 8192 / 1.0 from literals and applies
them only to its no-RoPE layers (`:175` is an `else if` on the RoPE
branch). So `AttnTemperature` is the three constants, `scale_at(pos)`
is the formula in llama.cpp's own precision (single up to the floor,
double from the log), `ModelConfig::attn_temperature` is the one
accessor, and ONE helper applies it in the CPU row body and both
batched host bodies, taking the row's position as a function so the
three bodies' three spellings of "which position is row `b`" are three
callers of one loop. No fused Metal launch has a per-token Q scale, so
`metal_can_serve_model` -- the predicate `residual_scale`, `clamp_kqv`
and the per-layer shapes already share -- keeps such a model on the
host bodies. The floor is the part a reader gets wrong:
`llama-model.cpp:1164-1165` seeds `n_ctx_orig_yarn` from
`context_length` BEFORE the YaRN key overrides it, so a Ministral with
no `original_context_length` floors on its context length, and a
fixture with exactly that shape measures it -- libllama's logits are
byte-identical to the file that declares the key. KL 9.16e-15 on both,
with a floor of 2 that steps TWICE inside the six-token prompt, and
5.14e-15 on the plain file. The MLA engine (`deepseek2` / `mistral4`,
which is Mistral-Large-3) REFUSES a nonzero scale by name now, where
it used to load and drop both keys: it had no golden to check an
implementation against when that landed (it has `plm`'s and
`deepseek2`'s since 2026-09-12), so an implementation there would have
been a guess.
`llama4` runs on this seam since 2026-09-14: its three literals are
`attn_temperature::LITERAL_ATTN_TEMPERATURE`, and the per-layer gate
its verdict had named is `AttnTemperature::unrotated_layers_only`,
read against `ModelConfig::layer_rotates` by the one helper.

Two corrections came with it. The verdict had described `mistral3` as
"leading-dense + MoE + shared expert": `mistral3.cpp:64-84` is EITHER
dense OR MoE on every layer (no `leading_dense_block_count` is read),
and its `_shexp` tensors are created only under an `n_ff_shexp` its
hparams never set and are read by no line of its graph -- a `mistral3`
file is a `llama` file with three keys, which is what every real
Ministral-3 is. And `mistral3.cpp:9` reads
`rope.scaling.yarn_log_multiplier`, whose only job is to adjust YaRN's
MAGNITUDE term -- and frink did not apply that term for ANY
architecture. `llama-context.cpp:196-231` multiplies
`rope.scaling.attn_factor` by `get_mscale(factor, 1) /
get_mscale(factor, log_mul)` (`1 + 0.1 ln factor` with no multiplier)
on top of ggml's own `rope_yarn` term, which it cancels; frink's
`rope_attn_factor` carried the key alone. Every YaRN checkpoint on the
generic path -- the `*-128K` Qwen3 exports among them -- was roped at
the right frequencies and the wrong magnitude, both q and k, so
attention logits low by `(1 + 0.1 ln factor)^2`, 1.30x at factor 4.
`frink_models::yarn_magnitude` folds the term into the same field the
CPU helper and the Metal `mscale` uniform already read, and two
fixtures with the factor at 4 evidence both arms against libllama:
KL 9.14e-15 without the multiplier and 3.45e-15 with it at 0.5, where
libllama's own log line reads `yarn_attn_factor = 1.0648`. Only
`mistral3` reads the multiplier on the generic path (measured;
`deepseek2`, `deepseek32` and `glm-dsa` apply it inside their own
`kq_scale` on other engines), so the key is dead metadata for every
other architecture here as upstream.

`apertus` and `step35` were the PER-LAYER-ACTIVATION pair.
`apertus.cpp:6-9` reads `xielu.alpha_n`, `xielu.alpha_p`, `xielu.beta`
and `xielu.eps` -- no architecture prefix, `llama-arch.cpp:370-373` --
as REQUIRED `n_layer`-long arrays, or one scalar broadcast to every
layer (`get_key_or_arr`), and `:132-138` hands layer `il`'s four to
`ggml_xielu` over that layer's `ffn_up` output, ungated; `ggml_xielu`
folds `beta + softplus(alpha_n)` and `softplus(alpha_p)` at graph build
and the CPU op is `alpha_p' x^2 + beta x` above zero and `(expm1(min(x,
eps)) - x) alpha_n' + beta x` below. `step35.cpp:28-29` reads
`swiglu_clamp_exp` and `swiglu_clamp_shexp` as OPTIONAL arrays, and
llama.cpp's GENERIC `build_moe_ffn` (`llama-graph.cpp:2146-2164`) and
`build_ffn` (`:1751-1768`) apply layer `il`'s entry when it is above
`1e-6` as `min(silu(gate), l) * clamp(up, -l, l)` -- the routed experts
read one array and the shared experts AND the leading dense layers read
the other, because `build_ffn` is both. `grep -l ggml_xielu` over the
155 graphs is `apertus.cpp` alone; `grep -l LLM_KV_SWIGLU_CLAMP` is
`step35.cpp`, `deepseek4.cpp` and `dflash.cpp`, the last two on their
own engine. So: `frink_models::act_layers` reads both key families
the way `get_key_or_arr` does (an array at exactly `n_layer` or a
scalar broadcast; a wrong length refused naming the key, as upstream);
`FfnActivation::Xielu` and `::SwigluClamped` CARRY their tables, so the
kind and the parameters cannot disagree; `frink_moe::GluAct` gained
the two bodies (`Xielu(params)`, `SwigluClamped { limit }`), which cost
it `Eq` and `Copy`-free `fn` pointers -- `gate_fn() -> fn(f32) -> f32`
became `combine(gate, up)`, because with the gate aliased to up
`xielu(gate) * up` is the wrong function by a factor of the input; and
`ModelConfig::layer_ffn_acts(il)` is the ONE accessor, answering a
`routed` / `dense` pair, which replaced the model-wide
`GluAct::from(ffn_activation)` at every FFN body -- that conversion no
longer exists, because it cannot be written for a variant that needs
the layer. The whole-model question the fused Metal stacks ask,
`model_ffn_act()`, is `None` for both, by TYPE rather than by comparing
layers, so every fused launch refuses them through the predicate it
already shared. `step35`'s other thing, the half-width rotary on its
full layers (`:9` halves `n_rot_full` AFTER `llama-model.cpp:1222`
seeded `n_rot_swa`, so `n_rot(il)` is 128 on the sliding layers and 64
on the full ones with no key saying so), landed on the seam that had
refused it from the other direction: `ModelConfig::rope_dim_swa` is the
two-valued width `n_rot(il)` already was upstream, `layer_rope(il)`
hands out a `LayerRopeParams { theta, freq_factors, rot_dim }` so no
rotation site takes the pair without the width, and the fused Metal
launches -- one `rot_dim` uniform for every layer -- are fenced off a
model whose widths differ. That lifted Laguna-XS.2's
`rope.dimension_count_swa` refusal by name with it: the fixture that
had evidenced the refusal matches libllama now, KL 7.29e-14. Real
Step-3.5-Flash also carries a llama3 `rope_freqs.weight` that
`step35.cpp:247` passes to the full layers ONLY (`is_swa ? nullptr :
...`, the one generic-path graph that does, measured); the loader takes
the tensor's first `n_rot_full/2` bands for the full layers and divides
by nothing on the sliding ones for that architecture, and refuses two
widths with divisors for any other, since nothing upstream says which
layers would take which. Five libllama-golden fixtures
(`tests/per_layer_activation_graphs.rs`, `tests/clamped_swiglu_graphs.rs`):
`apertus` with two layers' parameters differing in all four (KL
4.91e-14) and with the SCALAR spelling llama.cpp broadcasts (KL
5.06e-14, and the two goldens differ); `step35` with both arrays and a
zero beside each nonzero entry (KL 1.59e-13), with neither key (plain
SwiGLU, KL 2.59e-13, and the two goldens differ), and with a NextN
block inside `block_count` (byte-identical to the trunk's golden
upstream, KL 1.59e-13) -- each carrying the per-layer head counts, the
window array with its own base, the per-head sigmoid gate, sigmoid
routing by default, `exp_probs_b`, a shared expert on every MoE layer
and a dense layer decided by tensor presence, rather than asserting
them. Building them corrected one sentence of `apertus`'s verdict:
`apertus.cpp:50,52` CREATE `attn_q_norm.bias` / `attn_k_norm.bias`
and `:93,96` pass `NULL` as the bias, so they are loaded and never
read; a fixture that carries them measures libllama's logits
byte-identical with and without, and `frink_models::unread_tensors`
records the slot so `assert_every_tensor_consumed` can tell "ignored
as llama.cpp ignores it" from "missing".

`afmoe` and `laguna` were the GATED-ATTENTION pair, and `step35` is
the third graph with the op. llama.cpp's `LLM_TENSOR_ATTN_GATE`
(`blk.N.attn_gate.weight`) is created by six of the 155 graphs
(measured); three of those -- `qwen3next`, `qwen35`, `qwen35moe` --
keep the gated delta-net's `z` projection under that name, a different
op on a different engine, which is why the seam is keyed by
architecture and not by tensor presence. In the three that gate their
softmax attention the gate is projected from the SAME normed input
Q/K/V read and multiplied into the attention output after the
softmax-weighted V sum and before `wo` -- identical -- and what differs
is the activation (`afmoe.cpp:183` and `step35.cpp:272` sigmoid,
`laguna.cpp:246` SOFTPLUS), the width (`afmoe.cpp:73` one value per
channel, `step35.cpp:96` one per head broadcast over `head_dim`,
`laguna.cpp:110-124` either, read off the stored tensor with an abort
for anything else) and whether the tensor may be absent
(`step35.cpp:96` `TENSOR_NOT_REQUIRED`). So `frink_models::attn_gate`
has two axes, `GateAct` and `GateWidth`, a per-architecture table that
pins the activation and the ADMISSIBLE widths, and a loader that reads
the width off the tensor and refuses one the table does not admit. It
is applied in ONE function for the row body and the two batched host
bodies -- which collapsed the three hand-written `o_proj` / `o_bias` /
`post_attn_norm` tails onto it, so the gate was added to one place
rather than three -- and refused on every fused Metal launch by an
EXHAUSTIVE destructure of `AttnWeights` (`Decoder::metal_attn_view`),
so a field added to that struct does not compile until the Metal side
says whether the kernels serve it. Three libllama-golden fixtures
(`tests/gated_attention_graphs.rs`): `afmoe` (sigmoid / per element,
KL 7.03e-13), `laguna` in its M.1 shape (softplus / per element, KL
1.51e-13) and in its XS.2 shape (softplus / per head, a `head_count`
ARRAY so the per-head gate is sized by each layer's own count, a window
with its own base, KL 9.57e-14). Building them found two more things.
`afmoe.cpp:120` scales its embeddings by `sqrt(n_embd)` from arithmetic,
the only non-Gemma graph that does (measured) -- one table now,
`capability::embeddings_scaled_by_sqrt_n_embd`, where the Gemma side
had been a family match in the loader. And the loader's shared-expert
inference probed `blk.0` for a `_shexp` tensor when the count key was
absent, which answered 0 for every leading-dense model: `laguna.cpp:20`
assigns `n_expert_shared = 1` before reading a key its converter never
writes, and its layer 0 is dense, so a real Laguna export would have
loaded with its three REQUIRED shared-expert tensors unread on every
MoE layer. The probe is the first MoE layer now. One Laguna thing stays
refused by name, from a fixture libllama runs: a window together with
a RoPE scaling (`laguna.cpp:48,181-193` run the sliding layers with
YaRN off -- the Olmo-3 rule, one table now for `olmo2`, `mellum` and
`laguna` where it had been `arch == "olmo2"`). The other, a
`rope.dimension_count_swa` differing from `rope.dimension_count`
(`llama-hparams.cpp:85-91` gives the sliding layers their own rotary
width), was refused by `frink_models::swa_geometry` for one day and is
SERVED since `step35` closed on the same two-valued width; the two
`_swa` head-width keys stay refused there for every architecture. Real
Laguna-M.1 has neither; real Laguna-XS.2 has both and stops at the
scaling. `step35` said the gate was done and closed the same day on
the two things it had led with.

`mimo2`'s sinks moved the same way without closing the row. Four
graphs pass `attn_sinks` into the one `build_attn_mha`
(`openai-moe.cpp:115`, `mimo2.cpp:177`, `dflash`, `deepseek4`), so the
rule is "the tensor is present", not "the architecture is gpt-oss":
`AttnWeights::sinks` is loaded by presence on the generic path, the
gpt-oss loader checks it was (the tensor is REQUIRED there), and the
fused Metal launches refuse a layer with sinks by the same exhaustive
destructure -- by the tensor, on a model that is not gpt-oss, which a
test pins. What stopped `mimo2` next was what every real export
carries: `conversion/mimo.py` ALWAYS appends three NEXTN blocks inside
`block_count` and writes `nextn_predict_layers = 3`, and ALWAYS writes
`attention.sliding_window_pattern` as the per-layer `hybrid_layer_pattern`
ARRAY, which for this architecture is a per-layer bool even as a scalar
(`get_key_or_arr(..., is_swa_impl, n_layer)` broadcasts it) and not a
period. Both closed on 2026-09-11 (below), and the last thing --
MiMo-V2-Flash's `head_dim` of 192 beside a `v_head_dim` of 128, a V
width that differs from K's on every layer, which no frink KV cache
or attention kernel took -- closed on 2026-09-12 (above).

**The per-layer window ARRAY and the NextN blocks** are two seams that
landed together on 2026-09-11, because two verdicts named both and a
third row (`exaone-moe`) was over-refused on both. llama.cpp reads
`attention.sliding_window_pattern` with `get_key_or_arr`, and that name
hides three behaviours decided by which overload each graph calls:
the scalar overload, non-required, IGNORES an array and keeps the
seeded period (`llama-model-loader.cpp:502-507`; `exaone4.cpp:8`,
`exaone-moe.cpp:7`, `olmo2.cpp:10`, fifteen graphs); the array overload
honours it at `block_count` length and BROADCASTS a scalar as a bool
(`:474-478`; `mimo2.cpp:12`, `step35.cpp:26`, `gemma4`, `dflash`); and
`mellum.cpp:12-17` / `cohere2moe.cpp:32-36` try the first then the
second. `frink_models::swa_layers` is one enum (`All`, `Period`,
`PerLayer`) behind the one accessor every backend already asked,
`ModelConfig::layer_sliding_window(il)`, so the fused Metal stacks did
not need touching: they ask per layer. Every real EXAONE-4 32B,
EXAONE-MoE and Olmo-3 export carries the array (`conversion/exaone.py
:84`, `olmo.py:59-66`) and was refused over a value upstream never
reads; a fixture with the array INVERTED measures that libllama's
logits do not move, and frink matches both (KL 1.43e-14). `mellum` is
the one generic-path graph that honours the array, so it is the row
that evidences that branch -- its fixture's [T, T, F, T] disagrees with
the seeded period-4 [T, T, T, F] on two layers, KL 1.02e-14 -- and it
closed, with its window-plus-YaRN half (every real Mellum2) refused by
name as before.

The NextN blocks: `llama-model.cpp:1092` reads `block_count` into
`n_layer_all`, `llama-hparams.cpp:280-282` defines `n_layer()` as
`n_layer_all - n_layer_nextn`, `llama-graph.cpp:1433` builds every
graph over `n_layer()`, and each tensor loader creates the trailing
blocks `TENSOR_SKIP`. Only the SEVENTEEN graphs that read the key
subtract (measured; `frink_models::mtp_blocks::NEXTN_READERS`); for
any other a nonzero key stays refused, as upstream would fail on the
unread `nextn.*` tensors. `ModelConfig::n_layers` is the trunk now and
`n_mtp_blocks` the rest; the loader marks the skipped blocks' tensors
as deliberately unread so `assert_every_tensor_consumed` can tell
"skipped as llama.cpp does" from "missing from the graph". Two orderings
in llama.cpp were copied rather than tidied: `exaone4.cpp:4` tests
`n_layer() == 64` BEFORE `:18` reads the key, so a 64-trunk EXAONE-4
with a block appended gets NO window there and gets none here; and the
per-layer shape arrays are read at `block_count` length
(`llama-model.cpp:1148-1156` run before `load_arch_hparams`). Building
it found the GLM (`glm4moe`, `glm-dsa`, `glm4`), MLA (`deepseek2`) and
hybrid (`qwen3next`, `qwen35`, `qwen35moe`) dedicated loaders taking
`block_count` verbatim while every one of those graphs subtracts
upstream and their converters append the block inside `block_count`
(`glm.py:99`, `deepseek.py:457`): a real GLM-4.5 or DeepSeek-V3 file
would have run its MTP block as one more decoder layer with the
`nextn.*` tensors silently unread. All four dedicated loaders take the
trunk from the same function now. K-EXAONE's shape -- one block after
the trunk, the array at trunk length -- has a fixture, KL 1.09e-14.

`olmo2` and `exaone4` were the POST-NORM-ONLY pair -- no `attn_norm` and
no `ffn_norm` at all, both sublayers reading the raw residual, each
branch's output normed before its residual add -- and they closed
together because reading `olmo2.cpp:45-52,92,160-182` against
`exaone4.cpp:60-67,118,152-169` showed one graph, not two. One
implementation (`frink_models::norm`), one fixture each
(`tests/post_norm_only_graphs.rs`). One sub-case stays refused by name:
an `olmo2` with BOTH a sliding window and a RoPE scaling (Olmo-3) ropes
its sliding and full layers differently, decided by llama.cpp with no
GGUF key, the `baichuan` shape. EXAONE-4 32B was the other, and it is
CLOSED -- see the next paragraph.

`exaone-moe`, `smollm3` and EXAONE-4 32B were the PER-LAYER-RoPE trio,
and the claim that they are one cause was checked before it was
assumed. `exaone4.cpp:116` is `use_rope = is_swa(il) || swa_type ==
NONE`; `exaone-moe.cpp:136,155-161` is `is_swa(il)` around the same two
`ggml_rope_ext` calls, and `exaone-moe.cpp:4` pins `swa_type` to
`STANDARD`, which nails the second disjunct false -- identical, not
similar. `smollm3.cpp:5,69` is a different variant of the same enum,
`(il + 1) % 4 != 0`, with no window involved. All six architectures
llama.cpp gates this way (`smallthinker`, `afmoe` and `llama4` are the
other three; all three closed later, on other seams) sit in ONE table,
`frink_models::rope_layers`, and `ModelConfig::layer_rope` answers
`None` for an unrotated layer -- an `Option` around the base and the
divisors rather than a `bool` beside them, so no rotation site can take
the pair without answering the third question. Every site was checked:
the CPU head loop, the YaRN `attn_factor` (an argument to
`ggml_rope_ext`, so it goes with it), the four per-layer Metal launches
(now one `LayerRope` argument instead of a loose base/divisor pair),
and both fused Metal stacks, whose RoPE dispatch had been written in
unconditionally the way OLMo-1's final norm had. One libllama-golden
fixture per row (`tests/no_rope_layer_graphs.rs`): KL 2.05e-12 on the
64-layer EXAONE-4 32B file, 1.43e-14 on `exaone-moe`, 5.29e-15 on
`smollm3`. Building them found that EXAONE-4 1.2B must IGNORE a window
its file declares (`exaone4.cpp:4-14` reaches `set_swa_pattern` only at
64 layers), which `capability::swa_disabled_by_arch` now carries beside
the `phi3` case; that `nextn_predict_layers` was unrefused everywhere
(MTP blocks are inside `block_count` and llama.cpp skips them), which
`unsupported_feature_keys` now gates on the value the converters
actually write; and that `default_swa_layout`'s comment calling
`smallthinker` LIVE was wrong -- it had refused on its router since it
was triaged, and closed on that router on 2026-09-12 (above).

`granite`, `granitemoe` and the `granite-moe` alias were the SCALAR
MULTIPLIER trio, and the same story again: `granite-moe.cpp` has no
graph of its own (`models.h:1583-1591` is
`using graph = llama_model_granite::graph`), so the two upstream rows
differ in the FFN and in nothing else, and the third is a frink-only
alias for the second. One implementation
(`frink_models::scalar_multipliers`), one libllama-golden fixture each
(`tests/granite_family_graphs.rs`). Half the verdict stayed a refusal --
see cause 4 above for `rope.scaling.finetuned`, served on 2026-09-14
when Granite-4.0 needed it.

`olmo` (OLMo-1) closed ALONE, and that is the interesting part. It is a
THIRD norm shape: pre-norm like llama, but `olmo.cpp:65-67,104-106,128-130`
normalise with `build_norm(x, NULL, NULL, LLM_NORM, il)` -- a
non-parametric LayerNorm -- and `olmo.cpp:15-36` creates no norm tensor
of any kind, not even an `output_norm`. Before writing it, the question
"what else shares this cause" was answered by measurement rather than
hope: every `build_norm` call in all 155 of llama.cpp's
`src/models/*.cpp` graphs was scanned for a null weight argument, and
all three hits are `olmo.cpp`. `openelm`, `bitnet`, `arcee`, `mellum`,
`nanbeige` and `deci` were the candidates and none of them qualifies.
The LayerNorm *function* is shared -- `dbrx` and the bias group below --
but at the time none of those was one variant away, so a
weighted-LayerNorm variant would have had no caller and was deliberately
not written. Half of OLMo-1's verdict stayed a refusal, and it is the
half that read like an aside: `olmo.cpp:5` reads
`{arch}.attention.clamp_kqv`, `llama-graph.cpp:1611-1652` clamps Q, K
and V by it inside `build_qkv`, and `conversion/olmo.py:23-25` writes it
for every checkpoint whose HF config has a `clip_qkv` -- OLMo-7B-Twin-2T
and OLMo-1.7-7B do, at 8.0; the original OLMo-7B does not. A second
fixture measures that llama.cpp answers differently with it, so it is
not a no-op that could be ignored. Both halves of that paragraph turned
out to be one day old.

`dbrx` and `grok` closed on 2026-09-11, each by extending a seam that
had landed the day before, and that is the whole reason they were cheap
enough to take. `dbrx`'s three blockers were the weighted LayerNorm --
the variant the `olmo` work had refused to write without a caller, and
`dbrx` is the caller (`NormOp::LayerNorm`, `dbrx.cpp:69-71,110-112,
140-142`) -- a REQUIRED `attention.clamp_kqv` (`dbrx.cpp:5`), and its
pre-FFN norm stored as `blk.N.attn_output_norm` (`:34,110-113`). The
clamp was the expensive one and the one worth the most: the three host
bodies each applied the QKV bias in their own hand-written loop, which
is exactly why the OLMo clamp had been refused rather than implemented
(a clamp added to some copies and not the others is this repo's
dominant bug shape), so the three loops collapsed onto one helper
(`decoder/qkv_bias.rs`) and the clamp is a line in it, with the fused
Metal launches fenced off through the same predicate as
`residual_scale`. The clamped OLMo fixture, which used to evidence a
refusal, now matches libllama on all three paths (KL 1e-11 class), so
OLMo-7B-Twin-2T and OLMo-1.7-7B run. KL on the DBRX fixture: 3.4e-12,
max |delta| 6.1e-6. A DBRX file without the clamp key is refused, as
libllama refuses it (`key not found in model: dbrx.attention.clamp_kqv`,
measured).

`grok` was the MiniCPM shape and the verdict said so: `grok.cpp:5-12`
seeds SEVEN hyper-parameters before `:14-27` let the file override them,
so a Grok-1 export declaring none is still scaled by all of them.
`MultiplierDefaults::Grok` is the hook, on the same table as MiniCPM's,
and two of the seven needed a column the table did not have:
`logit_scale` is a MULTIPLY (`grok.cpp:211`, `LogitScaleUse::AsIs`, the
variant the module had named as absent), and the attention scale comes
from `{arch}.attention.output_scale`, applied INSIDE the tanh softcap
with `kq_scale = 1.0f` (`:137`, `llama-graph.cpp:2572-2582`) -- which is
arithmetically "pre-scale Q, then softcap", i.e. the `attention_scale`
slot plus the softcap Gemma-2 already uses, so no new attention code.
The other two keys Grok reads, `router_logit_softcapping` and
`attention.temperature_length`, are applied NOWHERE in llama.cpp's
graph (no other reference under `src/`, measured), so frink neither
applies nor refuses them. `attn_output_norm` is Grok's POST-attention
norm -- the same tensor name `dbrx` stores its pre-FFN norm under --
which is why `frink_models::norm_sites` exists: one table for which
tensor feeds which site, replacing the `if` chain the loader restated
at every site. Two fixtures, as MiniCPM needed: one declaring NO key
(the only shape that can tell the hook from its absence) and one
declaring every key at a value far from its default (pinning that the
file wins; a hook merged the wrong way round agrees with llama.cpp on
exactly the files that prove it exists). KL 4.7e-10 and 1.6e-10, max
|delta| 6.3e-5 and 4.0e-5 -- at the GeGLU tolerance, and measured to be
entirely llama.cpp's f16 GELU table: with frink's GELU made to emulate
the table both files agree to 1.0e-7 / 1.5e-7. Grok-2's parallel dense
FFN (`grok.cpp:171-184`, summed with the experts at `sqrt(2)/2`) was
refused by name from a fixture that has it until `arctic` closed on
the same seam (`frink_models::parallel_dense_ffn`); it is served and
checked against its own golden now.

`deci` and `openelm` were the PER-LAYER-SHAPE pair. llama.cpp reads
`{arch}.attention.head_count`, `.head_count_kv` and
`{arch}.feed_forward_length` as a scalar OR an `n_layer`-long array for
every architecture (`get_key_or_arr`, `llama-model.cpp:1149-1158`),
keeps three per-layer arrays, and hands most graphs layer 0 through
`LLAMA_LOAD_LOCALS`. Before a line was written, all 155
`src/models/*.cpp` were scanned for `n_head(i)`, `n_head_kv(i)`,
`n_ff(i)`, `n_embd_k_gqa(i)`, `n_embd_v_gqa(i)`, `n_rot(i)` and the
`_arr` fields, in both the tensor loader and the graph. Twenty-two
files read a per-layer shape somewhere; seventeen honour one in BOTH
places, and those are `frink_models::layer_shapes::PER_LAYER_SHAPE_ARCHS`
with what each still needs: `deci`, `openelm` and `plamo3` on the
generic path, and `laguna` and `step35` with them since they closed;
`mimo2` on the generic path, closed on the split K/V head width;
`nanbeige`, which copies the arrays to loop
its layers; `gemma4` and `gemma4-assistant` on a dedicated engine; and
the seven hybrid recurrent rows (`jamba`, `lfm2`, `lfm2moe`,
`nemotron-h`, `plamo2`, `granite-hybrid`, `kimi-linear`), where
`n_head_kv(i) == 0` means "this layer is recurrent" -- and since
2026-09-14 `layer_shapes::ZeroKvLayer` says WHICH recurrent block, with
`lfm2`'s short convolution served (`frink_models::shortconv`). Three things the
scan corrected: `n_rot(il)` is NOT an array upstream
(`llama-hparams.cpp:85-91` is `is_swa(il) ? n_rot_swa : n_rot_full`),
so `step35`'s and `laguna`'s "per-layer rotary width" is a two-valued
SWA/full field and a smaller seam than this one; `granite.cpp:204` reads
`n_head(il)` in its graph while sizing tensors from layer 0, so a
heterogeneous Granite file fails in llama.cpp's own loader and is not in
the table; and `hunyuan-moe`, `qwen3next` and the RWKV rows read
`n_ff(i)` only as a fallback for a shared-expert width.

The seam is `ModelConfig::layer_shape(il)`, the ONE accessor for a
layer's `AttnShape::{Gqa, Linear, Absent}` and `ffn_dim`; the scalars
`n_heads` / `n_kv_heads` are documented as the WIDEST layer's, for
budgets, and no layer body reads them. `ModelConfig::new_kv_caches` /
`new_paged_kv` size each layer's cache from its own shape, replacing
some ninety hand-written `KvCache::new(config.n_kv_heads, ...)` sites,
and `KvCache::push` asserts the row width so a cache built from the
scalar panics on the first token of a narrower layer rather than storing
a misaligned history. The fused Metal launches take ONE `n_heads` and
one KV geometry, so `metal_can_serve_model` keeps a non-uniform model
off all of them; the CUDA resident KV is gated on the same predicate;
the slot-file writer refuses a set of layers whose geometries differ.
`deci`'s three layer kinds -- `deci.cpp:107-109` passes the residual
through with no norm when `n_head == 0`, `:115-118` runs `attn_norm`
then `wo` alone when `n_head_kv == 0`, `:147-149` skips the FFN when
`n_ff == 0` -- are the enum, and the one combination llama.cpp handles
by discarding a computed branch (an FFN-free layer WITH attention: the
`continue` at `:147-149` runs before the residual add at `:150-153`) is
refused by name from a fixture that has it, with the drop MEASURED:
scaling that layer's attention weights by 3 leaves libllama's logits
byte-identical. Building the fixtures also found that libllama ABORTS
when such a layer is the LAST one (`GGML_ASSERT(buffer)`,
ggml-backend.cpp:194: the `inp_out_ids` `get_rows` result never rejoins
the graph), so a real export whose final block is a no-op cannot be run
by llama.cpp at all. Three libllama-golden fixtures
(`tests/per_layer_shape_graphs.rs`): the Nemotron shape with one layer
of each kind (KL 1.44e-13), the DeciLM-7B shape with only
`head_count_kv` varying (7.29e-13), and openelm with three layers
sharing no KV width and no FFN width, one fused `attn_qkv` per layer
split by that layer's own counts (1.28e-13). `openelm`'s old refusal --
a missing-hparam error for keys its file carries, because
`GgufValue::as_u64` returned `None` for the arrays its converter writes
-- is gone with it.

`arcee` closed ALONE on the UNGATED ReLU-squared FFN, and the reason it
was alone is the opposite of `olmo`'s. The cause is genuinely shared:
five graphs pass `LLM_FFN_RELU_SQR` (`arcee`, `plm`, `nemotron`,
`jais2`, `nemotron-h`, measured by grep). But the verdict constant the
row shared with `plm` had been written from `arcee.cpp` alone, and
`diff arcee.cpp plm.cpp` is 150 lines: `plm.cpp:16-19,32-36,84-166` is
DeepSeek-2's MLA attention -- `attn_kv_a_mqa`, `attn_kv_a_norm`,
`attn_kv_b`, Q split into nope/pe views, one shared roped key repeated
across the heads -- which the generic decoder does not have at all.
`plm` stays refused with a verdict that names the half that is done and
the half that is not. The FFN itself is `down(relu(up(x))^2)`
(`arcee.cpp:39-40,123-128`: `ffn_up` and `ffn_down`, no gate,
`LLM_FFN_SEQ`). It is spelled without a fourth expert shape:
`FfnActivation::ReluSqr` maps to `GluAct::Reglu` (`relu(gate) * up`) and
the loader ALIASES the expert's gate to its up matrix, so every gated
path computes `relu(up)^2` with no branch and the two dense hot paths
skip the aliased matmul. The durable part is what it removed: six
launch sites derived the fused Metal kernels' `gelu: bool` as
`!is_swiglu()`, which reads "not SwiGLU, therefore GELU" and would have
run this activation as GELU; `GluAct::fused_kernel_gelu_flag` returns
`None` for it and every site refuses on `None`. KL 2.27e-14
(`tests/ungated_ffn_graphs.rs`). A file WITH an `ffn_gate` is refused by
name, as libllama refuses it (`wrong number of tensors; expected 21, got
19`, measured).

| Shape | Architectures |
|---|---|
| Per-layer head counts or FFN width | CLOSED (`frink_models::layer_shapes`): `deci`, `openelm`, `laguna`, `step35` and `mimo2` run on it |
| A norm the generic decoder always applies and the model does not have (or a norm it does not have a slot for) | CLOSED: `olmo`, `olmo2`, `exaone4`, `dbrx`, `bitnet` and `talkie` were all here; `bitnet`'s two INNER norms are `frink_models::sub_norms`, `talkie`'s weightless RMS is `NormOp::RmsNoParams` and its skip stream `frink_models::skip_stream`; the LayerNorm WITH a bias is `NormOp::LayerNormBias`, on which `orion` and `nemotron` closed (`tests/biased_layer_norm_graphs.rs`) |
| Required `attn_output.bias` / `ffn_up.bias` / `ffn_down.bias` with no slot on the dense path | CLOSED (`frink_models::proj_bias`): `starcoder2`, `codeshell` and `jais2` run on it, a `llama` file with the optional biases runs where it was refused as unread, and gpt-oss's `o_bias` moved onto the same slot; `output.bias` on the LM head is a slot too (`proj_bias::OUTPUT_BIAS_CREATORS`; `phi2` runs on it); `phimoe` runs on both slots and `starcoder` on the same tables with its learned positions; the group is empty |
| LayerNorm rather than RMSNorm | CLOSED for the weightless (`olmo`), weighted (`dbrx`, `command-r`, `cohere2`, `cohere2moe`, whose file picks between LayerNorm and RMS by which epsilon key it carries) and biased (`orion`, `nemotron`, `starcoder2`, `codeshell`, `jais2`, `stablelm`, `gptneox`, `falcon`, `phi2`, `gpt2`, `starcoder`) forms, and for the biased RMSNorm (`phimoe`, `NormOp::RmsBias`) |
| A learned absolute position table added to the embeddings, no rotation | CLOSED (`frink_models::position_embd`, `rope_layers::RopeLayers::Never`): `gpt2` and `starcoder` run on it, and `mpt`'s optional table with its ALiBi |
| ALiBi: a per-head linear position bias on every score, no rotation | CLOSED (`frink_models::alibi`, `frink_core::alibi`; the three host kernels take the slopes, every fused GPU path refuses): `refact`, `bloom`, `mpt`, `jais` and Baichuan-13B run on it; `bloom`'s embedding norm is `norm_sites::EMBEDDING_NORM_ARCHITECTURES`, `jais`'s `1/d` attention scale `capability::attention_scale_override`, `mpt`'s `clamp_kqv` served |
| A parallel residual, `x + attn(norm(x)) + ffn(norm(x))` | CLOSED (`frink_models::parallel_residual`): `gptneox` (Pythia; two norms under `use_parallel_residual`, both values matched) and `plamo` (one shared norm) run on it, and the `stablelm` layer without `ffn_norm` matches where it was refused; eight of 155 graphs build the shape in two spellings and the table names each with its deciding rule; `command-r` (Command-R 35B, Aya-23) followed on it with the weighted LayerNorm and its `logit_scale` multiply; `falcon` runs on both arms, Falcon-40B's `attn_norm_2` crossing the two pre-norm slots per layer (`norm_sites::ATTN_NORM_2_FEEDS_ATTENTION`); `phi2` runs on it with the `output.bias` slot (`Decoder::output_bias`); `cohere2` (Command-R7B) runs on it with its sliding-only rotation (`rope_layers::SlidingOnly`); `cohere2moe` runs on it with routed experts, its dense-prefix rotation and the `0.5` on the shared-expert sum (`tests/cohere2moe_graphs.rs`) |
| A per-head LayerNorm on Q and K with a distinct weight per head (`{n_embd_head_k, n_head}`, `LLM_NORM`) | REFUSED by name (`frink_models::qk_layer_norm`), from a `stablelm` fixture libllama runs (8.73); `stablelm` (12B), `command-r` (64 layers), `chameleon` build it |
| Unkeyed NoPE layers, RoPE skipped on some layers with no GGUF key | CLOSED for all eight (`frink_models::rope_layers`; the first census counted six, `cohere2` and `cohere2moe` spell the gate `if (is_swa)`): `exaone-moe`, `smollm3`, EXAONE-4 32B, `afmoe`, `smallthinker`, `cohere2`, `llama4` and `cohere2moe` (`RopeLayers::SlidingOrLeadingDense`) run on it |
| A branch fed from the raw layer input rather than the post-attention residual | CLOSED (`frink_models::router_input`): `smallthinker`'s router reads it raw (`RawLayerInput`); `arctic`'s router AND experts read it under a second norm (`NormedLayerInput`), and its dense FFN summed with the experts is `frink_models::parallel_dense_ffn`, Grok-2's shape too |
| Hardcoded scales applied even when the GGUF carries no key | none left (`grok` was here and is CLOSED on the MiniCPM defaults hook; `mistral3` was here by mistake -- its scale comes from a key -- and is CLOSED) |
| A gated attention tensor (`wqkv_gate`) | CLOSED (`frink_models::attn_gate`): `afmoe`, `laguna` and `step35` run on it |
| Attention sinks outside gpt-oss | CLOSED as a tensor-presence fact (`AttnWeights::sinks`, CPU; the fused Metal launches refuse the layer): `mimo2` runs on it |
| A per-layer sliding-window ARRAY (`is_swa_impl`) | CLOSED (`frink_models::swa_layers`): `mellum`, `step35` and `mimo2` run on it and the EXAONE / Olmo-3 over-refusal is lifted |
| NEXTN/MTP blocks inside `block_count` | CLOSED (`frink_models::mtp_blocks`) for the seventeen graphs that read the key, on the generic path and all four dedicated loaders; refused by name elsewhere |
| A V head width that differs from the K head width | CLOSED (`frink_models::kv_head_dims`): `mimo2` runs on it on the host paths; every fused Metal launch, the CUDA resident hook, the slot file and the KV block file refuse a split model |
| An FFN activation whose PARAMETERS vary by layer (xIELU's four arrays; the SwiGLU clamp arrays by site) | CLOSED (`frink_models::act_layers`): `apertus` and `step35` run on it |
| A second rotary width on the sliding layers (`n_rot(il)`: `rope.dimension_count_swa`, or `step35`'s halved full width) | CLOSED (`ModelConfig::rope_dim_swa`, `frink_models::swa_geometry`): `step35` and the Laguna-XS.2 shape run on it; the two `_swa` HEAD-width keys stay refused by name, and two widths with per-band divisors are refused for any architecture but `step35` |
| An ungated or non-SwiGLU FFN | CLOSED for the ungated ReLU-squared form (`FfnActivation::ReluSqr`), the gated ReLU form (`FfnActivation::Reglu`) and xIELU (`FfnActivation::Xielu`): `arcee`, `smallthinker` and `apertus` run on them, and `plm` on the MLA engine (`MlaDenseFfn::act`) |
| A per-position attention temperature | CLOSED (`frink_models::attn_temperature`): `mistral3` runs on it from a key and `llama4` from literals on its unrotated layers; `deepseek2` / `mistral4` (Mistral-Large-3) refuse it by name on the MLA engine |
| A recurrent block at the attention site (`head_count_kv 0`, or the layers `attention.recurrent_layers` / `full_attention_interval` name) | CLOSED for the gated delta net (`frink_models::gdn`, `frink_core::gdn`, `AttnShape::Gdn`): `qwen35` runs on it, its attention layers' interleaved `wq` gate in `attn_gate::Q_INTERLEAVED_GATE_ARCHS`. CLOSED for the short convolution (`frink_models::shortconv`, `layer_shapes::AttnShape::ShortConv`, the state as the layer's KV history on all three backings): `lfm2` and `lfm2moe` run on it. CLOSED for the Mamba-2 block (`frink_models::mamba2`, `frink_core::mamba2`, `AttnShape::Mamba2`, the state as `frink_core::recurrent_state::RecurrentState` beside the cache): `granitehybrid` and `nemotron_h` run on it (the latter's one-block-per-layer topology is `layer_shapes::BLOCK_WITHOUT_FFN_KEEPS_ITS_OUTPUT` plus `norm_sites::ONE_NORM_PER_LAYER`). `falcon-h1` runs attention and Mamba-2 in PARALLEL on every layer (`ModelConfig::parallel_ssm`). CLOSED for the Mamba-1 block (`frink_models::mamba1`, `AttnShape::Mamba1`): `jamba` runs on it, `mamba` and `mamba2` run every layer as the block (`layer_shapes::PURE_RECURRENT`). CLOSED for PLaMo-2's own spelling (`frink_models::plamo2_ssm`, `AttnShape::Plamo2Ssm`): `plamo2` runs on it, with the per-head-distinct QK norm (`QkNormStyle::PerHeadDistinct`) and the `plamo2` tokenizer |
| Something structurally new | `grovemoe` (a second expert bank -- and, read against `modeling_grove_moe.py` on 2026-09-12, llama.cpp's graph feeds the chunk experts the routed experts' OUTPUT and gathers their weights at the CHUNK index where the reference does neither, so there is no one graph to match; its verdict says so); `plm` (MLA attention on a dense model) was here and is CLOSED on the MLA engine (`frink_models::mla_arch`); `arctic` (a parallel dense + MoE layer) was here and is CLOSED (`frink_models::parallel_dense_ffn`); `mellum` was here on "two per-layer RoPE variants", which is the Olmo-3 rule refused by name, and is CLOSED; `mistral3` was here on the temperature and is CLOSED; `nanbeige` was here on running the same layers more than once and is CLOSED (`frink_models::layer_loops`) |

**Unknown (1).** `phi4` is the only row left here. It is not in
llama.cpp's `LLM_ARCH_NAMES` -- `src/llama-arch.cpp` carries `phi3` and
no phi4 entry -- so there is no reference graph to diff against, and
Frink admits it as phi3's fused-QKV / fused gate+up graph on the
assumption that a file spelling it means the same thing. It refuses
until a real file settles that, and its message says which tensor in
`blk.0` would decide it.

**`mistral`, `mixtral` and `yi` were the other three, and were resolved
on 2026-09-10 by finding they are not architectures.** The old verdict
asked for "a real GGUF whose `general.architecture` is literally one of
these three". No such file can be produced: none of the three is in
llama.cpp's `LLM_ARCH_NAMES` or in gguf-py's `MODEL_ARCH_NAMES`
(`mistral3` and `mistral4` are the only strings under that prefix), and
libllama REFUSES a file declaring one -- `unknown model architecture:
'mistral'`, measured on a synthetic llama-shaped file written under
each string. The two real checkpoints on the development host,
`Mistral-7B-Instruct-v0.2-Q4_K_M.gguf` and `Yi-1.5-6B-Chat-Q4_K_M.gguf`,
both declare `general.architecture = llama`, which is audited and runs.

So all three are refused as *strings*, not triaged as architectures,
and the refusal says the actionable thing: re-convert with
`convert_hf_to_gguf.py` and your file will load as `llama`. Moving them
also closed a live hazard. They sat on the generic path with NEOX RoPE
while `llama` -- the graph they claim to be -- is in llama.cpp's NORM
group, so a file spelling `mistral` would have been rotated on the
wrong pairs of every Q/K head, and the only test that compares RoPE
layouts could not see it, because a name absent from llama.cpp's table
is a `continue` there. That skip now has to be declared by name.

`FRINK_ALLOW_UNKNOWN_TENSORS=1` loads the checkpoint anyway and accepts
whatever comes out. Use it while you debug, not to get past the error
and carry on.

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
