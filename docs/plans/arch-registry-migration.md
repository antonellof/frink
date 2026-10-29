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
- Step 1a: landed in #428. `models/facts.rs` and `ArchFacts::from_tables`,
  built from the raw name. `ModelConfig::from_gguf` reads 39 lookups
  through the facts: 34 rows, the two refused-key lists derived from
  them (`models/refused_keys.rs`), and 3 `HparamRules`
  (`swa_window_override`, `attention_scale_override`, `rope_layers`).
- Step 1b: `TensorFacts`. `Decoder::from_gguf` reads 12 lookups through
  the facts: 10 tensor rows from the canonical name (the gpt-oss extra
  tensors, the norm-site row, the per-head QK norm rows, the
  interleaved Q gate, the post-norm pairing row, the shared-expert sum
  scale, the weightless embedding norm, QK norm after RoPE) and
  `family` / `memory` from the raw name. Three resolvers were split
  into a row the facts carry and a shared `_for` / `sum_scale_on_layer`
  step; `NormSites::GENERIC` is the one statement of the default row.
- Guards (`models/facts_tests.rs`): every tabled name is held to its
  legacy answer, `DEFAULT` to an absent name, and both loaders' source
  to no direct call of a routed lookup, with the raw versus canonical
  split pinned.
- Still direct: in `ModelConfig::from_gguf`, 29 calls that take the
  name beside a hyperparameter or the file (`mtp_blocks::trunk_layers`,
  `kv_head_dims`, `norm_function_for_file`, `moe_interleave`,
  `clamp_kqv`, `attn_temperature`, `swa_geometry::rotary_widths`,
  `yarn_log_mul_for`, `post_norm_eps`, `weightless_qk_norm`, `alibi`
  and others); in `Decoder::from_gguf`, 17 `load_*` / probe helpers
  that read the file (`weight_scales`, `position_embd`,
  `parallel_residual`, `NormSites::for_layer`, `sub_norms`, `attn_gate`,
  `proj_bias`, `mamba2`, `layer_shapes::load_non_gqa_attention`,
  `moe_interleave::dense_by_router_absence`, `parallel_dense_ffn`,
  `skip_stream`, `unread_tensors`). Those move with their family, from
  step 2 on: a resolver that is a row plus arithmetic splits as the
  three above did, and a file reader stays loader code taking its row.

- Step 2: `models/llama.rs`, `models/granite.rs`, `models/granite_moe.rs`
  (`granitemoe` and the `granite-moe` alias) and `models::REGISTRY`.
  Their rows left `scalar_multipliers::MULTIPLIER_ARCHITECTURES`,
  `rope_finetuned::ROPE_GATED_ON_FINETUNED` and both `proj_bias`
  creator tables. `models/tests.rs` holds the three registry guards
  (each name resolves once, audited if and only if it has evidence, no
  registry name in a behaviour table) and keeps the catalog fields to
  the catalog.
- Step 3: `models/olmo.rs`, `models/olmo2.rs`, `models/exaone4.rs`.
  Their rows left `capability.rs` (`POST_NORM_ONLY_ARCHITECTURES` and
  `NON_PARAMETRIC_LAYER_NORM`, now empty and deleted; the two
  predicates derive from the norm-site row), `default_swa_layout`, the
  two SWA RoPE follow lists, `swa_window_override`, `rope_layers`,
  `swa_geometry`, `clamp_kqv` and `mtp_blocks::NEXTN_READERS`.
  `exaone4`'s two code-shaped rules are functions in its file
  (`HparamRules`). `rope_layers::PER_LAYER_ROPE_GATES` keeps its
  `exaone4` row: it is a census of llama.cpp's graphs that the rule is
  checked against, not a row the loader reads. Divergences found and
  filed, not fixed in the move: #440, #441 (fixed since, with #439 and
  #446).
- Step 4: `models/qwen.rs`, `qwen2.rs`, `qwen2moe.rs`, `qwen3.rs`,
  `qwen3moe.rs`. Their rows left `NO_TOPK_RENORMALIZE_ARCHITECTURES`,
  `proj_bias::OUTPUT_BIAS_CREATORS` and `FFN_LENGTH_COUNTS_GATE_AND_UP`
  (emptied and deleted). `qwen3` and `qwen3moe` had no row in any table:
  their facts are `ArchFacts::DEFAULT`. No snapshot fixture reaches the
  `qwen2` and `qwen2moe` rows, so `models/tests.rs` pins them. Divergence
  found and filed, not fixed: #447.

- `kolibri1` (#414), outside the step order: the first architecture
  ADDED in `crate::models` rather than moved there
  (`models/kolibri1.rs`). It grew the facts by three fields:
  `MoeFacts::selection` (`frink_moe::SelectionRule`, where
  `exp_probs_b` enters the top-k), `SwaFacts::rope_base_key_unread` (a
  graph that seeds the sliding base and reads no `rope.freq_base_swa`)
  and `HparamRules::load_checks` (the throws in an architecture's own
  `load_arch_hparams` / `load_arch_tensors` that no table states). The
  lookups its rows touch answer through `ArchFacts::of` with
  `_from_table` twins (`window_required`, both
  `swa_rope_*_follows_model`, `NormSites::for_arch`,
  `rope_layers::rope_layers`). Two of its rows stay in legacy tables
  the facts do not carry yet: the sliding-window array read
  (`swa_layers::PER_LAYER_ARRAY_READERS`) and the catalog row.
- Step 5: skipped. The triage reads (0, 0, 8, 1), so no row is one
  match arm or one fixture away.
- Step 6a: `models/olmoe.rs`, `dots1.rs`, `phimoe.rs`, `dbrx.rs`,
  `grok.rs`. Their rows left the MoE routing tables, `proj_bias`,
  `clamp_kqv` (`CLAMP_KQV_REQUIRED` emptied and deleted),
  `scalar_multipliers`, `LOGIT_SOFTCAP_ARCHITECTURES`, `uses_geglu`,
  `WEIGHTED_LAYER_NORM`, `BIASED_RMS_NORM`,
  `PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM` and `POST_NORMS_UNDER_GROK_NAMES`
  (all three emptied and deleted; the norm predicates derive from the
  norm-site row), `YARN_BETA_READERS` and the parallel dense FFN table.
  `phimoe`'s dropped window is a rule in its file. Step 6b is `gpt-oss`
  and `llama4`, whose rows also sit behind `attn_temperature`,
  `chunked_swa`, `moe_interleave` and `weightless_qk_norm`.
- Step 6b: `models/gpt_oss.rs` (`src/models/openai-moe.cpp`) and
  `models/llama4.rs`. The four single-row resolvers `chunked_swa`,
  `moe_interleave`, `attn_temperature` (its literal) and
  `routed_weight_site` lost their only row and their tables are
  deleted; each now takes its row from `ArchFacts` (`*_for`), and
  `weightless_qk_norm` and `llama4`'s per-layer RoPE are
  `HparamRules`. Divergence found and filed, not fixed: #458.
- Step 7a: `models/stablelm.rs`, `gptneox.rs`, `phi2.rs`, `falcon.rs`,
  `plamo.rs`. Their parallel-residual rows are named constants in
  `parallel_residual` that the files reference
  (`parallel_residual_rows()` is the census over table and registry),
  and the resolvers take the row (`*_for`). Rows also left
  `BIASED_LAYER_NORM`, `uses_gelu_ungated`, `proj_bias`,
  `QKV_BIAS_CREATORS` and `PER_HEAD_LAYER_NORM_QK`;
  `ATTN_NORM_2_FEEDS_ATTENTION` (falcon only) is deleted and
  `NormSites::for_layer` takes the row. Step 7b is the Cohere graphs
  (`command-r`, `cohere2`, `cohere2moe`), whose rows also span the SWA,
  multiplier, MoE and MTP tables.

What step 2 changed about the target design:

- `ModelDef` is `{ names, source, facts, evidence }` for now. `profile`
  needs the catalog to be `const` (step 11), and `plan` waits for
  `layer_plan` (step 13); `ArchFacts::of` fills `profile` and `audited`
  from `capability.rs` until then.
- Reading the facts in the loaders is not enough. A public name-keyed
  lookup whose row moves (`multiplier_support`, `attn_out_presence`,
  `ffn_presence`, `rope_finetuned::unrotated`) has other readers
  (`rope_scaling`'s YaRN default, tests, `capability.rs`), so it now
  answers through `ArchFacts::of`, and `from_tables` reads a
  `_from_table` twin so the two cannot recurse. Every later step does
  this for each table it empties.
- A family's rows are not all in the routed tables. Two of granite's
  three sat behind step 1's "still direct" resolvers and were split
  here (`unrotated_for`, the `proj_bias` loaders taking their row);
  later families will split the resolvers their rows touch.
- `llama-embed` stays outside the registry: it keeps its catalog row on
  the config side and reaches `llama`'s definition on the decoder side
  through `canonical_architecture`, so `ModelDef.names` holds raw names
  only.
- Found while reading `granite-moe.cpp`: its hparam loader does not read
  `rope.scaling.finetuned`, so llama.cpp rotates a `granitemoe` file
  declaring `false` and frink does not. No converter writes the key for
  it. Kept unchanged here (this step is a refactor) and recorded in
  `models/granite_moe.rs`.

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
