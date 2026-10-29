# Architecture registry migration (#403)

Implements [ADR 0003](../adr/0003-one-file-per-architecture.md). Line
numbers are from `e1f8771`.

## Where architecture identity flows today

- `ModelConfig::from_gguf` (`loader/model_config.rs:46-52`) reads
  `general.architecture`, resolves the catalog row
  (`capability::resolve_profile`, `capability.rs:2115`), then makes
  about 60 name-keyed calls in an order the file header says matters.
  The unaudited gate runs last (`:1199`).
- `Decoder::from_gguf` (`loader/decoder_from_gguf.rs:82`) keys its
  tensor tables on the canonical name (`capability.rs:3376`) and
  `family` / `memory` on the raw name (`:775-784`).
- The decoder never reads the name; it reads `ModelConfig`, the layer
  weights, and the hand-written Metal predicates in
  `decoder/metal_eligibility.rs:79-180`.

About 110 static tables and 90 functions are keyed by architecture,
across `capability.rs`, `loader/arch_tables.rs` and roughly 40 feature
modules (norm, QK, FFN, SWA, RoPE, attention, residual, MoE, shapes,
tensors, refusals, engine choice). A few decisions are code, not rows
(`capability.rs:3534`, `:3695`, `:3875`, `rope_layers.rs:252`).

## Target types

- `models/facts.rs`: `ArchFacts`, one field group per table family,
  reusing today's row types; `ArchFacts::DEFAULT` means "absent from
  every table"; `HparamRules` holds function pointers for the
  code-shaped decisions.
- `ModelPlan { config: ModelConfig, layers, decoder: DecoderFacts }`.
  `ModelConfig` is unchanged. `DecoderFacts` absorbs `ExecutionPlan`.
- `LayerPlan` is a pure function, `ModelConfig::layer_plan(l)`, built
  from today's per-layer accessors. Not a cache: 86 tests mutate
  `d.config.*` after load.
- `models/mod.rs`: `ModelDef { names, source, profile, facts, evidence,
  plan }` and `REGISTRY`. `AUDITED_GENERIC_GQA` becomes derived from
  entries with evidence, keeping its name and API.

Guards (`models/tests.rs`): every registry name resolves exactly once;
audited if and only if it has evidence; no registry name remains in a
legacy table; an exhaustive `ModelConfig` destructure in the Metal
eligibility test so a new field forces a Metal decision.

## Steps, one PR each, no behaviour change

| # | Content | Verified by |
|---|---|---|
| 0 | Snapshot every fixture's `ModelConfig` (or refusal text) and per-layer weight presence | itself, sabotaged once |
| 1 | `ArchFacts` + `from_tables`; loaders read facts, call order unchanged | snapshots |
| 2 | First family: `llama` + Granite (`granite`, `granitemoe`, `granite-moe`) | `granite_family_graphs.rs`, snapshots |
| 3 | `olmo`, `olmo2`, `exaone4` | `olmo_graphs.rs`, `post_norm_only_graphs.rs` |
| 4 | Qwen family | `qwen3moe_graph.rs`, `qwen2moe_paris.rs` |
| 5 | Triage rows one match arm / fixture away | their tests |
| 6 | MoE routing families (olmoe, dots1, gpt-oss, phimoe, dbrx, grok, llama4) | per-family tests |
| 7 | Parallel residual families | `parallel_residual_graphs.rs` |
| 8 | ALiBi and learned positions | `alibi_graphs.rs`, `position_embd_graphs.rs` |
| 9 | SWA families, after sliding-window KV eviction (#61) lands | `swa_pattern.rs`, `window_array_graphs.rs` |
| 10 | `AUDITED_GENERIC_GQA` derived from the registry | `documented_counts.rs`, `arch_coverage.rs` |
| 11 | Catalog and `ArchProfile` move to `arch.rs` | `arch_coverage.rs` |
| 12 | Triage prose shrinks to one line per row; arguments move into `tests/unaudited_triage.rs`, evidence prose into each `models/<arch>.rs` header | `unaudited_triage.rs` |
| 13 | `layer_plan` on the host paths | decoder tests |
| 14 | Metal eligibility derived from the plan | ignored Metal tests on hardware |
| 15 | Fold `ExecutionPlan` into `ModelPlan` | `documented_capabilities.rs:160` |

## Progress

- Step 0: landed in #415 (`tests/model_config_snapshot.rs`).
- Step 1a: `models/facts.rs` and `ArchFacts::from_tables`, built from
  the raw name. `ModelConfig::from_gguf` reads 39 lookups through the
  facts: 34 rows, the two refused-key lists derived from them, and 3
  `HparamRules` (`swa_window_override`, `attention_scale_override`,
  `rope_layers`). The two lists are derived in `models/refused_keys.rs`; the
  `capability.rs` functions are wrappers. `models/facts_tests.rs`
  holds every tabled name to its legacy answer, `DEFAULT` to an absent
  name, and the loader source to no direct call of a routed lookup.
- Step 1b, open: the 29 calls in `from_gguf` that still take the name
  beside a hyperparameter or the file (`mtp_blocks::trunk_layers`,
  `kv_head_dims`, `norm_function_for_file`, `moe_interleave`,
  `clamp_kqv`, `attn_temperature`, `swa_geometry::rotary_widths`,
  `yarn_log_mul_for`, `post_norm_eps`, `weightless_qk_norm`, `alibi`,
  ...), and `Decoder::from_gguf`'s tensor tables (canonical name) and
  `family` / `memory` (raw name). Where a resolver is a row lookup
  followed by arithmetic, 1b moves the row into the facts and keeps the
  arithmetic shared; the file readers stay loader code.

## Risks

1. Raw versus canonical names must be reproduced exactly by
   `ModelDef.names`.
2. Named refusals must still precede the unaudited refusal.
3. Refused rows with table entries written ahead stay in the legacy
   tables until audited.
4. Metal eligibility parity tests need hardware (step 14).
5. SWA window is shared with KV eviction; steps 9, 13, 14 wait for #61.
6. The audited count is pinned in several docs; step 10 must keep it.

`capability.rs` should fall from about 5000 lines to about 1500 after
step 12, and under 1000 after step 11.
