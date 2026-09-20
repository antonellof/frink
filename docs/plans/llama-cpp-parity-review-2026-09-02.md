# llama.cpp parity review (2026-09-02)

Companion to [`llama-cpp-gap-inventory.md`](llama-cpp-gap-inventory.md) (evidence-backed differential). This document ranks **what to do next** for parity with llama.cpp as a serving + CLI stack.

**North star:** same GGUF, same command shapes, same or better performance on hardware people own ([`north-star.md`](north-star.md)).

---

## Executive summary

**Re-checked against the code on 2026-09-02, evening.** Several rows
below were closed the same day this document was written, so the table
now carries what is true rather than what was true at breakfast.

| Area | Parity level | Priority |
|------|--------------|----------|
| Architecture coverage | **16** audited + 4 dedicated engines vs 140 llama.cpp graphs | P0, expand audited set |
| CLI flag semantics | `-ngl` still refuses a partial count deliberately; `-e` and `--repeat-last-n` already match | P2, documented rather than divergent |
| Server flags | **DONE**: `-c`, `--api-key`, `--api-key-file`, `--alias`, `--ctk`, `-hf`, `--hf-file`, `-cb`, `-np` | closed |
| Sampling / grammar | **Done**: GBNF, JSON Schema, forced `tool_choice`, the two penalties, and `--samplers` ordering with unimplemented samplers refused by name | closed, bar the samplers frink does not implement (`dry`, `xtc`, `typ_p`, `top_n_sigma`) |
| Tools (quantize, perplexity) | `frink quantize` writes Q8_0 byte-identically; `frink perplexity` agrees with `llama-perplexity` to within a fifth of a standard error on five checkpoints | P1, K-quant encoders (#70); HellaSwag and the other corpus sub-tools remain |
| Serving / batching | CB auto-on Metal; incremental CB streaming | P1, slot save/load |
| Metal concurrency | Phase 1 shipped (#46 closed); per-request Metal KV is follow-up | P2 |

### Closed since this document was written

- Architecture coverage 11 to **16**: five one-match-arm rows admitted
  with libllama-golden fixtures, so unaudited went 46 to 41.
- Every server flag listed under "Server flags to env today": `-c`,
  `--api-key`, `--api-key-file`, `-fa`, plus `--alias`, `--ctk`,
  `--hf-file`.
- `-hf` on both the run and serve paths, with `repo:QUANT` resolution.
- `logit_bias` on chat, `response_format: json_schema`, and JSON schema
  under continuous batching.
- `--presence-penalty` and `--frequency-penalty`, which the engine had
  always honoured while the CLI hardcoded both to zero.
- The refusal split by triage verdict.

### What the re-check found that this document did not

`quantize` is filed here as a missing TOOL. It is a missing
CAPABILITY: `frink-quant` has `quantize_q8_0` and two activation
quantizers, and nothing that writes a K-quant. A `frink quantize`
subcommand is not the work; K-quant encoders with importance-weighted
rounding are, and pretending otherwise would produce a command that
can only emit Q8_0 while its name implies llama.cpp's range. Tracked
with the step-wise shape it would need as
[#70](https://github.com/antonellof/frink/issues/70).

---

## P0 — Correctness and trust

### 1. Flag semantics that silently diverge (fix or refuse)

From gap inventory §4.1:

- **`-ngl`**: llama.cpp partial layer offload vs frink all-or-nothing — **refuse with clear error** until partial placement exists, or implement layer counting.
- **`-e` / escape**: default off in frink vs on in llama.cpp — default-on or `--no-escape` alias parity.
- **`--repeat-penalty`**: windowed vs full-history — add `--repeat-last-n` and match default 64.

### 2. Metal parallel decode (#46)

**Done on branch `fix/metal-parallel-concurrency` (partial):**

- Auto-enable continuous batching on Metal (safe multi-request path)
- Single-flight gate for private-loop Metal when CB disabled
- Poison-tolerant `metal_attn_kv` locks + empty-`hidden` CPU fallback
- CLI `--cont-batching` / `-cb`, `--no-cont-batching`

**Still open:**

- Per-request Metal KV residency (true parallel private path)
- Incremental streaming under continuous batching
- Metal concurrency integration test on real GGUF

### 3. Architecture refusals vs silent wrong

The unaudited-arch refusal gate is **good** (better than silent wrong). Next:

- Split generic `UnauditedArchitecture` messages by triage verdict (fixture-away vs new-code) — gap inventory §1.3
- Close **fixture-away** rows first: `bailingmoe2`, `minimax-m2`, `olmo2`/`seed_oss` post-norms wiring

---

## P1 — Serving parity

### Continuous batching & parallelism

| llama.cpp | frink (after this branch) | Gap |
|-----------|----------------------------|-----|
| `-cb` / `--cont-batching` | `--cont-batching`, `-cb`, `FRINK_CONTINUOUS_BATCHING` | **CLI added**; document auto-default on Metal |
| `-np` / `--parallel` | `FRINK_CB_MAX_SEQS` only | Add `-np` → env |
| Slot save/load | None | Missing |
| Streamed CB output | Token stream | frink buffers full completion under CB |

### Server flags → env today

High-value CLI additions (map to existing `FRINK_*`):

- ~~`-c` / `--ctx-size`~~ **done**, sets `FRINK_CB_MAX_CONTEXT`
- ~~`--api-key`~~ **done**, plus `--api-key-file`, which is the form to
  prefer on a shared host since an argument is visible in `ps`
- ~~`-fa`~~ **done**, accepted; `-fa off` refuses by name and points at
  `FRINK_METAL_ATTN=0`, because fused attention is a backend property
  here rather than a per-run switch
- `-b` / `-ub` → prefill/decode batch envs, still open

### API surface

Strong: OpenAI chat/completions, SSE, embeddings, health handshake, admin models.

Gaps (inventory §3): `logit_bias` dropped on chat, JSON schema under CB, Anthropic/OpenAI responses parity edges, multimodal.

---

## P1 — CLI & tools

### Missing llama.cpp tools

| Tool | frink | Action |
|------|--------|--------|
| `quantize` | None | **Critical** — users need llama.cpp side-by-side |
| `perplexity` / benchmarks | `bench`, `parity`, no corpus ppl | Add perplexity or document `frink bench` scope |
| `gguf-split` | read shards only | Merge/split utility |
| `imatrix` | None | Lower priority |

### `frink run` vs `llama-cli`

Align: `-n` default (-1 = EOS), `-cnv`/`-no-cnv`, `-sys` spelling, `-hf` on run path, grammar/json flags.

---

## P2 — Performance parity

- **Engine bench:** `frink bench` vs `llama-bench` — keep receipt discipline ([`benchmarks/README.md`](../../benchmarks/README.md))
- **HTTP bench:** `frink serve-bench` vs `llama-server` load — extend parallel/concurrency scenarios
- **Kernel parity:** Metal/CUDA gap inventory in [`llama-cpp-gap-inventory.md`](llama-cpp-gap-inventory.md) §2

---

## P2 — Architecture scale

- **Model layer reorg** ([`model-layer-reorg.md`](model-layer-reorg.md)) — prerequisite for 140-arch maintenance
- **Out-of-core MoE** — separate track
- **Vulkan** — verdict GO; backend seam refactor pending

---

## Recommended roadmap slices

1. **Merge `fix/metal-parallel-concurrency`** — Metal serving safe by default *(done in 0.15.2)*
2. **Flag semantics sprint** — `-ngl`, escape, repeat-last-n (refuse > wrong)
3. **Server CLI parity pack** — `-c`, `--api-key`, document env mapping table
4. **Fixture-away architectures** — 3–5 new audited rows with tiny GGUF fixtures
5. **Slot save/load** — llama.cpp KV serialize/restore (Frink has no equivalent yet)
6. **Quantize tool** — or official doc pointing to llama.cpp quantize

---

## Verification discipline

Every parity claim needs:

1. Side-by-side command on same GGUF
2. Test that fails when reverted
3. Receipt or HTTP bench JSON checked in

Use [`llama-cpp-gap-inventory.md`](llama-cpp-gap-inventory.md) as the evidence ledger; update counts when code changes, not docs alone.

---

## References

- Issue [#46](https://github.com/antonellof/frink/issues/46) — Metal parallel decode (closed in 0.15.2)
- [`docs/plans/metal-parallel-concurrency.md`](metal-parallel-concurrency.md) — design
- [`docs/CONFIG.md`](../CONFIG.md) — env vars
- [`docs/API.md`](../API.md) — HTTP surface
