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
| Nemotron, Nemotron-H, Nemotron-3 Nano | Recurrent. Nemotron-3 Super's `moe_latent_size` is refused by name |
| Jamba, Mamba, Mamba-2, FalconMamba, Falcon-H1, PLaMo-2 | Recurrent |
| LFM2, LFM2-MoE | A file declaring `attention.sliding_window` is refused by name |
| MiniMax-Text-01, MiniMax-M2 | |
| gpt-oss | **CPU only**: no Metal kernel implements attention sinks |
| Llama 4 Scout, Llama 4 Maverick | **CPU only**: its window is chunked, and the fused Metal launches take one window per layer |
| openPangu-Embedded | A decoder, not an embedding model |
| BitNet, Apertus, Step-3.5, Laguna, Mellum, SmallThinker, MiMo-V2, Nanbeige, Talkie, Arcee, Deci, OpenELM, StableLM | StableLM-2-12B is refused: it needs a per-head QK LayerNorm |
| Ternary-Bonsai-2-27B (PrismML) | Verified on the real checkpoint. `PTQ1_0` runs on CPU and Metal; `PQ2_0` is recognised, not executed |
| BERT, nomic-bert, jina-bert-v3 | Encoders, for `/v1/embeddings`. A cross-encoder with a rank head answers `/v1/rerank`; `/v1/score` takes either |

**99 architectures run with** a benchmark row, a pinned logit
comparison against real `libllama`, or a fixture behind each. The list
above is by family; `frink archs` is by GGUF architecture string and is
the authoritative one.

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
output to tell you. An error you can read beats output you cannot
trust, so the loader refuses. The message always names the reason, and
one of six things caused it.

1. **The architecture is unknown.** Not in the capability registry.

2. **Known, not implemented.** The refusal names the missing feature.

3. **The file carries weights Frink never reads.** The loader records
   every tensor name it looks up and stops if any are left over. This
   catches a missing graph feature automatically rather than one at a
   time, which is how `attn_sinks` and `exp_probs_b` were both found.
   Tensors for parts Frink does not claim to run (`mm.`, `v.`,
   `mmproj.`, `resampler.`, `audio.`) are ignored.

4. **The file declares a scale factor Frink does not apply.**
   `{arch}.logit_scale`, `{arch}.residual_scale`,
   `{arch}.embedding_scale` and `{arch}.attention.scale` are
   hyperparameters rather than weights, so check 3 cannot see them, and
   a file declaring one would otherwise load while computing a
   differently-scaled graph than it was trained as.

5. **Position is encoded some other way than RoPE.** ALiBi, a learned
   absolute position table, or no rotation at all. All of these run;
   the check remains because the generic path's guess is "plain GQA
   with RoPE" and it was wrong for exactly this group five times.

6. **Nobody has verified this architecture against llama.cpp.** The
   shared generic-GQA decoder is a guess, so it is opt-in.
   `FRINK_ALLOW_UNAUDITED_ARCH=1` runs one anyway; compare the output
   against llama.cpp yourself before trusting it.

## The four unaudited architectures

None of the 4 is a fixture or a single match arm away: they need an
attention implementation or a reading nobody has done. All 4 have now
been read on both sides, and each refusal prints its own blocker with
the `llama.cpp/src/models/*.cpp` line that decides it, so
`frink -m <file>` is the authoritative answer rather than this table.

| Class | Count |
|---|---|
| fixture-away | 0 |
| one match arm | 0 |
| new code | 3 |
| unknown | 1 |

Both cheap classes are empty, which is a better answer than the count:
nothing still refusing is one fixture or one arm away.

| Architecture | What is missing |
|---|---|
| `graniteswitch` | A second routing stage over expert groups |
| `qwen4exp` | A per-token adapter selection the decoder has no site for |
| `grovemoe` | A second expert bank, and llama.cpp's graph and the reference model disagree about it. There is no single graph to match |
| `phi4` | Not in llama.cpp's `LLM_ARCH_NAMES`, so there is no reference graph to diff against. Frink would admit it as `phi3`'s graph on the assumption that a file spelling it means the same thing, and refuses until a real file settles that |

The counts here are pinned against the catalog by
`crates/frink-models/tests/documented_counts.rs`: they had gone stale
once in the direction that matters, reading `new code 1` while the
catalog held three.

## `mistral`, `mixtral` and `yi` are not architectures

None is in llama.cpp's `LLM_ARCH_NAMES` or gguf-py's
`MODEL_ARCH_NAMES`, and libllama refuses a file declaring one. Every
real checkpoint of all three declares `general.architecture = llama`,
which is audited and runs. Frink refuses the three strings by name and
says the actionable thing: re-convert with `convert_hf_to_gguf.py` and
the file loads as `llama`.

That also closed a live hazard: the three sat on the generic path with
NEOX RoPE while `llama` is in llama.cpp's NORM group, so a file
spelling `mistral` would have been rotated on the wrong pairs of every
Q/K head.
