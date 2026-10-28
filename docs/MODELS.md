# Models

Which checkpoints Frink runs, and which stop with an error instead.

`frink archs` prints the live list;
[`manifests/architecture_manifest.md`](manifests/architecture_manifest.md)
is the generated copy. Speed is
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md); what the engine
can do is [`FEATURES.md`](FEATURES.md).

## What runs

| Family | Notes |
|---|---|
| Llama 3.x, TinyLlama, Mistral, Yi, MiroThinker | |
| `llama-embed` | Llama's graph with the output head skipped. `/v1/embeddings` pools its hidden states |
| Qwen2.5, Qwen3, Qwen2-MoE / Qwen1.5-MoE, Qwen3-MoE | |
| Qwen3.5 dense and MoE, Qwen3-Next | Recurrent: no prefix-cache reuse, no `--model-draft` |
| Gemma-2, Gemma-3 | |
| Gemma-4-E2B | Dedicated engine, `gemma4` BPE tokenizer |
| Phi-2, Phi-3, Phi-4-mini, Phi-3.5-MoE | |
| SmolLM2, SmolLM3, OLMo, OLMo-2, OLMoE, Olmo-3 | |
| Mixtral, DBRX, Arctic, Grok, Jais, MPT, BLOOM, Refact, GPT-2, GPT-NeoX, StarCoder, StarCoder2, Falcon | |
| DeepSeek-V2 / V3, Mistral-Large-3, PLM | MLA engine, both tensor forms, YaRN as real exports declare it. Any other scaling type is refused by name |
| GLM-4-0414 / Z1 / OCR, GLM-4.5 / 4.5-Air / 4.6 | A vision tower's `rope.dimension_sections` is refused by name on `glm4`, served on `glm4moe` |
| Command-R, Command-R7B, Cohere2 MoE | Command-R+ (64 layers) is refused: it needs a per-head QK LayerNorm |
| EXAONE-4, EXAONE-MoE, Granite 3, Granite 4.0 hybrid | Granite 4.0 is recurrent: no prefix-cache reuse, no `--model-draft` |
| Nemotron, Nemotron-H, Nemotron-3 Nano | Recurrent. Nemotron-3 Super's `moe_latent_size` is refused by name, and so is the `domyn-small` pre-tokenizer |
| Jamba, Mamba, Mamba-2, FalconMamba, Falcon-H1, PLaMo-2 | Recurrent |
| LFM2, LFM2-MoE | A file declaring `attention.sliding_window` is refused by name, and so is an LFM2-d1 decision head (`decision.block_count`) |
| MiniMax-Text-01, MiniMax-M2 | |
| gpt-oss | **CPU only**: no Metal kernel implements attention sinks |
| Llama 4 Scout, Llama 4 Maverick | **CPU only**: its window is chunked, and the fused Metal launches take one window per layer |
| openPangu-Embedded | A decoder, not an embedding model |
| BitNet, Apertus, Step-3.5, Laguna, Mellum, SmallThinker, MiMo-V2, Nanbeige, Talkie, Arcee, Deci, OpenELM, StableLM | StableLM-2-12B is refused: it needs a per-head QK LayerNorm |
| Ternary-Bonsai-2-27B (PrismML) | Verified on the real checkpoint. `PTQ1_0` runs on CPU and Metal; `PQ2_0` is recognised, not executed |
| BERT, nomic-bert, jina-bert-v3 | Encoders, for `/v1/embeddings`. A cross-encoder with a rank head answers `/v1/rerank`; `/v1/score` takes either |

100 architectures run, each backed by a benchmark row, a pinned logit
comparison against real `libllama`, or a fixture. The list above is by
family. `frink archs` lists them by GGUF architecture string and is the
authoritative list.

## What does not run

| | |
|---|---|
| MiniMax-M3 | Needs MiniMax Sparse Attention |
| Kimi K3, GLM-5.2, DeepSeek V4 | Loaders and primitives only; nothing run end to end |
| `graniteswitch`, `qwen4exp`, `grovemoe`, `phi4` | Unaudited; see below |
| Vision | Frink finds an mmproj file and warns. An `image_url` in a request is an error |
| MTP draft heads | `--mtp` errors by design. Speculation is prompt-lookup only |

## Why a model stops

A model whose graph Frink only partly implements would load, run fast,
and return fluent text computed by the wrong maths, with nothing in the
output to tell you. The loader refuses such a model with an error
instead. The message names the reason, which is one of these six.

1. **The architecture is unknown.** It is not in the capability registry.

2. **Known, not implemented.** The refusal names the missing feature.

3. **The file carries weights Frink never reads.** The loader records
   every tensor name it looks up and stops if any are left over. This
   catches a missing graph feature automatically instead of one at a
   time by hand, which is how
   `attn_sinks` and `exp_probs_b` were both found.
   Tensors for parts Frink does not claim to run (`mm.`, `v.`,
   `mmproj.`, `resampler.`, `audio.`) are ignored.

4. **The file declares a scale factor Frink does not apply.**
   `{arch}.logit_scale`, `{arch}.residual_scale`,
   `{arch}.embedding_scale` and `{arch}.attention.scale` are
   hyperparameters, not weights, so check 3 cannot see them. Without
   this check, a file declaring one would load and compute a graph
   scaled differently from the one it was trained as.

5. **Position is encoded some other way than RoPE.** ALiBi, a learned
   absolute position table, or no rotation at all. All of these run.
   The check stays because the generic path assumes "plain GQA with
   RoPE", and that assumption was wrong for this group five times.

6. **Nobody has verified this architecture against llama.cpp.** The
   shared generic-GQA decoder is a guess, so it is opt-in.
   `FRINK_ALLOW_UNAUDITED_ARCH=1` runs one anyway; compare the output
   against llama.cpp yourself before trusting it.

## The eleven unaudited architectures

None of the 11 is a fixture away. Two are a single match arm away
(table rows on seams frink already serves), and the rest need code or a
reading nobody has done. All 11 have now been read on both sides. Each
refusal prints its own blocker with the `llama.cpp/src/models/*.cpp`
line that decides it, so trust `frink -m <file>` over this table.

| Class | Count |
|---|---|
| fixture-away | 0 |
| one match arm | 2 |
| new code | 8 |
| unknown | 1 |

The count was 4 until 2026-10-28, when the llama.cpp pin moved from
`5b59b83` to `604cc430` and seven of its thirteen new architecture names
turned out to be generic-path candidates
([`plans/parity-audit-2026-10-28.md`](plans/parity-audit-2026-10-28.md)).
Because upstream keeps adding architectures, this count rises whenever
the pin moves.

| Architecture | What is missing |
|---|---|
| `kolibri1` | Table rows only: RoPE on the sliding layers, the window array, sandwich norms, sigmoid routing with a shared expert. Needs a libllama golden |
| `lfm3` | Table rows only: LFM2's layers plus a per-head sigmoid attention gate, the window array, and `rope_freqs` on the full layers alone |
| `flint` | Its `flint` pre-tokenizer is a sequence of two regexes, which the tokenizer cannot express. The graph itself is served seams |
| `beam` | A scaled residual (`h / sqrt(i) + r * sqrt((i - 1) / i)`) and a power-form attention temperature |
| `maion-coder` | A latent MoE (the experts run between `ffn_latent_down` and `ffn_latent_up`) |
| `k2-horizon` | A grouped RMS norm, a base-2 softplus attention gate, and value experts (MoVA) |
| `muse-spark` | A norm-preserving residual with per-channel alpha/beta tensors, and a latent MoE with a post-expert norm |
| `graniteswitch` | A per-token adapter selection the decoder has no site for |
| `qwen4exp` | A delta-net over a hybrid memory index, a memory class frink does not have |
| `grovemoe` | A second expert bank, and llama.cpp's graph and the reference model disagree about it. There is no single graph to match |
| `phi4` | Not in llama.cpp's `LLM_ARCH_NAMES`, so there is no reference graph to diff against. Frink would admit it as `phi3`'s graph on the assumption that a file spelling it means the same thing, and refuses until a real file settles that |

The counts here are pinned against the catalog by
`crates/frink-models/tests/documented_counts.rs`. They once went
stale in the direction that matters, reading `new code 1` while the
catalog held three.

## `mistral`, `mixtral` and `yi` are not architectures

None is in llama.cpp's `LLM_ARCH_NAMES` or gguf-py's
`MODEL_ARCH_NAMES`, and libllama refuses a file declaring one. Every
real checkpoint of all three declares `general.architecture = llama`,
which is audited and runs. Frink refuses the three strings by name and
tells you to re-convert with `convert_hf_to_gguf.py`, after which the
file loads as `llama`.

This also fixed a real defect. The three sat on the generic path with
NEOX RoPE while `llama` is in llama.cpp's NORM group, so a file
spelling `mistral` would have been rotated on the wrong pairs of every
Q/K head.
