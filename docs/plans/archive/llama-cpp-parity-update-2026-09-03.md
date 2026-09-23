# llama.cpp parity update (2026-09-03)

Companion delta to [`llama-cpp-full-parity-audit-2026-09-02.md`](llama-cpp-full-parity-audit-2026-09-02.md).

**North star:** [`north-star.md`](../north-star.md)

---

## Closed since 2026-09-02

| Item | Issue/PR | Notes |
|------|----------|-------|
| Qwen1.5-MoE Metal MoE prefill `CommandFailed` | [#96](https://github.com/antonellof/frink/issues/96) / [#97](https://github.com/antonellof/frink/pull/97) | Q5_0 expert down weights lacked `mul_mv_id` kernels; 12/24 layers fell back to CPU |
| Phi-4 LongRoPE at parity `n_ctx` | [#98](https://github.com/antonellof/frink/issues/98) / [#100](https://github.com/antonellof/frink/pull/100) | `apply_runtime_context(n_tokens + 8)`; KL 3.6e-2 → 1.2e-2; K-quant lm_head reclassified DRIFT |
| DeepSeek-R1-Distill WRONG vs Qwen2.5 DRIFT | [#99](https://github.com/antonellof/frink/issues/99) / [#100](https://github.com/antonellof/frink/pull/100) | Untied Q6K `output.weight`; same graph as Qwen2.5; higher KL threshold for K-quant lm_head |
| Gemma-4 parity via Gemma4Engine | [#101](https://github.com/antonellof/frink/issues/101) / [#100](https://github.com/antonellof/frink/pull/100) | `prefill_logits` routes gemma4 through `Gemma4Engine`; scratch `llama_logits` for tokenizer |

---

## Live parity sweep (2026-09-03, CPU, 19 GGUFs)

**Reference:** Homebrew `llama.cpp` via default `bash tools/build_llama_logits.sh`
(`brew --prefix llama.cpp`, libllama 7650). Release CPU:
`FRINK_METAL=0 FRINK_CUDA=0`. Gemma-4 requires scratch `llama_logits` (Homebrew
cannot load the checkpoint); see reference vintage below.

| Verdict | Count | Models |
|---------|-------|--------|
| **MATCH** | 6 | TinyLlama Q8_0, Llama-3.2-1B Q8_0/IQ4_XS, Llama-3.2-3B, Mistral-7B, OLMoE |
| **DRIFT** | 10 | All K-quants incl. Qwen2.5 + DeepSeek-R1 + Phi-4 (expected Q8_K activation quant on K-quants — §10 gap inventory) |
| **TIE-FLIP** | 1 | Llama-3.1-8B |
| **WRONG** | 0 | none, on either reference, since #102 corrected the K-quant lm_head line |
| Skip | 2 | Gemma-4 (Homebrew libllama cannot load checkpoint), BERT encoders |

### Reference vintage

Parity verdicts depend on which `llama_logits` binary links against. Homebrew is
the default for CI and issue closure; scratch `.scratch/llama.cpp` is required when
Homebrew cannot load a checkpoint or when updating the reference baseline.

| Model | Homebrew (b7650) | Scratch (ggml 0.18.0) |
|-------|------------------|-----------------------|
| Qwen2.5-1.5B Q4_K_M | DRIFT (KL 7.7e-3) | DRIFT (KL 2.7e-2, top-1 match) |
| Phi-4-mini Q4_K_M | DRIFT (KL 1.179e-2) | DRIFT (KL 3.800e-3) |
| DeepSeek-R1-Distill Q4_K_M | DRIFT (KL 9.2e-3) | DRIFT |
| gemma-2-2b Q4_K_M | DRIFT (KL 6.5e-3) | TIE-FLIP (KL 1.53e-2) |
| Gemma-4 E2B Q4_K_M | load fail | **MATCH** (KL 1.7e-4) |

Two corrections to the first version of this table. **The Phi-4-mini
numbers were in the wrong columns**: measured 2026-09-04, Homebrew is
1.179e-2 and scratch 3.800e-3, not the other way round. And Qwen2.5 on
scratch is **DRIFT**, not WRONG, since #102.

### Why the verdicts moved with the reference ([#102](https://github.com/antonellof/frink/issues/102))

Settled by taking frink out of the experiment: dump both references'
logits for the same token ids and compare them TO EACH OTHER.

| Pair | KL |
|---|---|
| llama.cpp b7650 against llama.cpp ggml 0.18.0 | **2.735e-2** |
| b7650 against frink | 7.679e-3 |
| ggml 0.18.0 against frink | 2.669e-2 |

**The two llama.cpp builds disagree with each other by more than frink
disagrees with either.** frink's top-1/top-2 gap (1.5488) reproduces
b7650's (1.5505) to 0.1%; the newer build reads 1.0144. frink was never
the outlier.

Where it comes from: across nine checkpoints the two builds are
**bit-identical on Q8_0** and differ on every K-quant and IQ tier. The
newer `libggml-cpu` carries interleaved-repack kernels for `block_q5_K`
and `block_q6_K` that the bottle lacks (261 repack symbols against 83).
Not operator fusion: `GGML_CPU_DISABLE_FUSION=1` changes nothing.

So the `vec_dot_type` difference in §10 of the gap inventory is not only
a frink-versus-llama.cpp story. **llama.cpp reproduces it against
itself**, between two of its own builds, which is the sharper statement.

The consequence for the oracle: `KL_WRONG_LM_HEAD_KQUANT` was a bare
2.5e-2, BELOW that 2.735e-2 spread, so the "these graphs disagree" line
fired on a difference llama.cpp produces against itself. It is now
derived from the measured spread with the ordering asserted at compile
time. `KL_WRONG` stays at 1e-2, because the same experiment puts the
build-to-build spread on a Q8_0 lm_head at exactly zero.

Every parity row now prints the reference it used, and
`frink parity --dump-logits` writes both vectors before the verdict, so
a run keeps its evidence.

### Phi-4-mini ([#98](https://github.com/antonellof/frink/issues/98))

`verify_engine::load_and_tokenize` calls `apply_runtime_context(n_tokens + 8)` before decode load, matching `tools/llama_logits.c`. KL improved **3.6e-2 → 1.2e-2**; reclassified **DRIFT** (tied Q6K lm_head uses K-quant activation path).

### DeepSeek-R1-Distill ([#99](https://github.com/antonellof/frink/issues/99))

Same qwen2 family as Qwen2.5-1.5B (DRIFT 7.7e-3, same top-1 12095). Untied **Q6K** `output.weight` explains KL **2.0e-2** vs sibling — not a graph bug. Reclassified **DRIFT**. That threshold was `2.5e-2` when this was written and is now derived from the measured reference-build spread (#102), which is why the same verdict no longer depends on which libllama produced the reference.

### Gemma-4 ([#101](https://github.com/antonellof/frink/issues/101))

Scratch-built `llama_logits` from `.scratch/llama.cpp` loads gemma-4; tokenizer **MATCH**. Logit parity runs through `Gemma4Engine::forward_token` in `prefill_logits`.

---

## Priority plan (updated)

### P0 — This week

| # | Item | Status |
|---|------|--------|
| 1 | Phi-4 LongRoPE at parity `n_ctx` | **Done** — PR #100 |
| 2 | DeepSeek-R1-Distill reclassification | **Done** — PR #100 |
| 3 | Gemma-4 parity via Gemma4Engine | **Done** — PR #100 |
| 4 | Rebuild `llama_logits` from `.scratch/llama.cpp` | **Done** — `build_llama_logits.sh` cmake layout |
| 5 | Fixture-away triage | **Done** (2026-09-03) -- seven admitted with libllama-golden fixtures, `chatglm` reclassified ONE MATCH ARM. Audited 16 to 23, unaudited 41 to 34 |

### P1 — Unchanged from audit

Model layer reorg phase 1, K-quant encoders, CUDA mul_mm, server slot save/load, embedding path, gguf-split.

---

## Regenerate

```bash
cargo build -p frink-cli --release
# Homebrew or cmake scratch build:
bash tools/build_llama_logits.sh
# cmake example:
# cmake -B /tmp/llamabuild ... .scratch/llama.cpp && cmake --build /tmp/llamabuild --target llama -j8
# LLAMA_CPP_PREFIX=/tmp/llamabuild bash tools/build_llama_logits.sh

export FRINK_METAL=0 FRINK_CUDA=0 FRINK_ALLOW_MULTIPLE_INSTANCES=1
mkdir -p .scratch/parity-run-2026-09-03
for f in models/*.gguf; do frink parity -m "$f"; done
```

*Updated 2026-09-03 on PR #100 branch.*
