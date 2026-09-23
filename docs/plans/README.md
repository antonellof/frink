# Plans

Two files hold the plan:

- **[`north-star.md`](north-star.md)** is the goal and the ranking rule.
  Be the Rust alternative to llama.cpp: same models, same command
  shapes, same or better performance, on the hardware people actually
  own.
- **[`roadmap.md`](roadmap.md)** is every open item, merged by theme.

Four items are large enough to carry their own design document:

- **[`speculative-decoding.md`](speculative-decoding.md)**, the one
  decode item that raises throughput without buying hardware. Decode
  reads every weight per token, so bandwidth divided by model bytes is a
  hard ceiling; a draft model changes what is read per token rather than
  how fast. The lossless half already ships and is tested at 200k
  samples. What is missing is a drafter worth having.
- **[`model-layer-reorg.md`](model-layer-reorg.md)**, splitting the
  decoder so architectures scale. It was 6438 lines when that document
  was written and is 6702 today, the first time it has shrunk.
- **[`out-of-core-moe.md`](out-of-core-moe.md)**, running a 155 GB model
  on a 32 GB machine.
- **[`gdn-resident-state.md`](gdn-resident-state.md)**, a recurrent
  layer that does not come back to the host. It is the measured next
  step for Bonsai-2-27B decode (7.2 tok/s against the PrismML fork's
  11.5), with the per-token ledger that says why, the delta-rule and
  gated-norm kernels that already exist and are pinned against the CPU
  definition, the reason wiring them TODAY is a loss (3.1 MB of state
  per layer, copied both ways), and four measured non-results so they
  are not tried again.

One audit sits beside them:

- **[`parity-audit-2026-09-19.md`](parity-audit-2026-09-19.md)**,
  the llama.cpp surface re-measured against a moved pin, plus the
  serving-feature gaps that a GGUF architecture count cannot see.

How work lands is written down too:

- **[`contribution-workflow.md`](contribution-workflow.md)**, the rule
  that a completed feature is a branch and a pull request, a defect is a
  GitHub issue, and the two are never the same artifact. It also carries
  the parallel-agent rules, whose first failure is two branches editing
  one file.

One item has a written **verdict** rather than a design:

- **[`vulkan-beachhead-verdict.md`](vulkan-beachhead-verdict.md)**, the
  `d-hardware-reach` GO/NO-GO. GO: a Q8_0 matvec ran as a hand-emitted
  SPIR-V shader on a real device and matched its scalar twin. It also
  carries the survey of the backend seam a third backend would need,
  which is `backend-seam-refactor`'s to-do list.

Parity inventory and deltas against llama.cpp:

- **[`parity-audit-2026-09-19.md`](parity-audit-2026-09-19.md)** — the current re-measurement of the llama.cpp surface and the serving-feature gaps
- **[`serving-parity-audit.md`](serving-parity-audit.md)** — continuous batching, paged KV, prefix caching and cache-aware admission read against the code, every claim carrying the line that decides it (`documented_serving.rs` holds the citations)
- **[`server-speculative-decoding.md`](server-speculative-decoding.md)** — the engine has it, the server cannot reach it
- **[`done/several-completions-per-request.md`](done/several-completions-per-request.md)** — `n` > 1 as a KV fork, not as a loop that re-prefills
- **[`llama-cpp-gap-inventory.md`](llama-cpp-gap-inventory.md)** — evidence-backed differential (not a plan)
- **[`archive/llama-cpp-full-parity-audit-2026-09-02.md`](archive/llama-cpp-full-parity-audit-2026-09-02.md)** — file map + sweep + priority plan
- **[`archive/llama-cpp-parity-update-2026-09-03.md`](archive/llama-cpp-parity-update-2026-09-03.md)** — post-merge delta (Qwen MoE Metal, Phi-4 LongRoPE, sweep)
- **[`cpu-cuda-parity.md`](cpu-cuda-parity.md)** — the two backends that
  are not at parity, ordered by what was measured on rented hosts on
  2026-09-04 rather than by tok/s. Carries the kernel-coverage matrix,
  because a gap column cannot show a format the backend never runs

Everything else is history: [`archive/`](archive/) holds superseded
plans, whose open items were merged into the roadmap by theme;
[`on-hold/`](on-hold/) holds work ranked below the goal, each with the
condition that brings it back; and [`done/`](done/) holds plans that
finished. The counts in each of those READMEs are derived from the
files by `crates/frink-models/tests/documented_counts.rs`, because two
of them had already drifted.

## Where the project stands

Re-audited 2026-09-12, by what happens when a real checkpoint loads
rather than by whether the architecture name is known:

| Outcome | Count |
|---|---|
| Runs, **with evidence** | **92** (`capability::AUDITED_GENERIC_GQA`) |
| Loads on a dedicated engine | 4 engines (`Mla`, `Glm52`, `Kimi`, `Gemma4`); `Mla` has cross-engine evidence since 2026-09-12 (`plm`, `tests/plm_graphs.rs`; `deepseek2` in both tensor forms, `tests/deepseek2_graphs.rs`; the real PLM-1.8B through `frink parity`), `Gemma4` has it on the real Gemma-4-E2B (parity MATCH, KL 5.1e-4 on Q4_K_M, against a libllama that has `gemma4.cpp`), `Glm52` and `Kimi` none |
| Refuses as **unaudited**, now triaged | 2 |
| Off the generic path: refuses by name, or reaches one of those 4 engines | 53 (22 `dedicated` + 31 `deferred` in the manifest; `glm4moe`, `glm4`, `orion`, `nemotron`, `starcoder2`, `codeshell`, `jais2`, `stablelm`, `gptneox`, `plamo`, `command-r`, `falcon`, `phi2`, `cohere2`, `phimoe`, `gpt2`, `starcoder`, `refact`, `bloom`, `mpt`, `jais`, `minimax-m2`, `lfm2`, `lfm2moe`, `granitehybrid`, `granite-hybrid`, `nemotron_h`, `nemotron_h_moe`, `falcon-h1`, `jamba`, `mamba`, `mamba2`, `qwen35`, `qwen35moe`, `qwen3next`, `llama4` and `cohere2moe` left the dedicated column for the generic path on 2026-09-12 / 14 and `plm` went the other way) |
| **Loads and is WRONG** | **closed** |

Counts reproduce from
[`../manifests/architecture_manifest.md`](../manifests/architecture_manifest.md),
regenerated with `frink archs --write`: 150 rows, 94 generic-gqa (92 of
them audited), 22 dedicated, 31 deferred, 3 test fixtures.

The "loads and is WRONG" class is closed because the generic path is
opt-in: an architecture not on the audited list stops rather than
guessing. The five strings that used to compute ALiBi or learned
position embeddings as though they were NEOX RoPE (`gpt2`, `mpt`,
`refact`, `bloom`, `jais`) became `DedicatedOnly` refusals, pinned by a
test; all five left that test on 2026-09-14 the right way round,
audited on a rule that rotates nothing with the position they DO
encode served (`frink_models::position_embd` for `gpt2`,
`frink_models::alibi` for the four ALiBi rows and Baichuan-13B), and
the test pins that a row of that group is generic ONLY under that
rule.

The 2 unaudited refusals split 0 fixture-away / 0 one-match-arm /
1 new-code / 1 unknown, each naming the `llama.cpp/src/models/*.cpp`
line that decides it. **Both cheap classes are empty**: nothing still
refusing is one fixture or one arm away, so every row left needs a
different graph. Five one-match-arm rows closed on 2026-09-02
(`seed_oss`, `maincoder`, `bailingmoe`, `deepseek`, `hunyuan-moe`),
seven fixture-away rows on 2026-09-03, `gemma`, `hunyuan-dense` and
`ernie4_5-moe` after them, and on 2026-09-10 `olmo2` and `exaone4`
(below) plus `chatglm` -- the last one-match-arm row -- and `qwen`,
which the same arm turned out to close only halfway, the three
Granite rows and `olmo` (both below), and on 2026-09-11 `exaone-moe`
(below), then `grok` and `dbrx` on seams landed the day before, then
`arcee` on the ungated ReLU-squared FFN and `deci` and `openelm`
together on the per-layer shape seam (`frink_models::layer_shapes`,
sized by a scan of all 140 graphs before it was built), then `afmoe`
and `laguna` together on the gated attention
(`frink_models::attn_gate`, one op with two free parameters behind
three verdicts, read side by side first), then `mellum` on the
per-layer window array, `apertus` and `step35` together on the
per-layer activation parameters, and `mistral3` on the per-position
attention temperature (`frink_models::attn_temperature`, whose
reach -- three graphs of 140 -- was measured first and came back with
one generic-path row), and on 2026-09-12 `smallthinker` on the MoE
router operand (`frink_models::router_input`: fifty-nine
`build_moe_ffn` call sites parsed first, four pass a precomputed
`probs_in`, one on this engine routes on something other than the
normed FFN input; its "one match arm" ReLU experts turned out to need
a variant, because the one that existed served `arcee` by aliasing a
gate SmallThinker really has), and `bitnet` on the two norms INSIDE
the blocks (`frink_models::sub_norms`: one graph of 140 creates
either tensor, so the seam is a `bool`; its optional per-projection
`.scale` tensors, which llama.cpp applies for every architecture, are
refused by name in `frink_models::weight_scales`), and `mimo2` on the
split K/V head width (`frink_models::kv_head_dims`: one generic-path
converter writes the two widths apart, the KV cache, the one row
kernel the three contiguous arms collapsed onto, the batched prefill
kernel and every check took the V width, and the bisection to its
last 2e-3 of KL found `expert_weights_scale` honoured for every
architecture where llama.cpp reads it in twenty loaders), and
`nanbeige` on the layer loop (`frink_models::layer_loops`: the weights
are shared and the KV is not, so the seam is a logical-to-physical
mapping and a loop norm, not a copy of the weights), and `talkie` on
four seams at once (`NormOp::RmsNoParams`, `QkNormStyle::PerHeadScalar`,
`frink_models::skip_stream`, and the two `.scale` companions
`frink_models::weight_scales` now serves), each one graph of 140,
and `plm` on the MLA engine (`frink_models::mla_arch`,
`frink_models::mla_q_proj`: its attention was already there, and the
three ways it differs from DeepSeek-2 -- a direct `attn_q`, an ungated
ReLU-squared dense FFN, a tied lm_head -- are one table; the direct-Q
column also lifts the refusal of every lite DeepSeek-V2 export, and
the fixture is that engine's FIRST libllama golden), and `arctic` on
the parallel dense + MoE layer (`frink_models::parallel_dense_ffn`:
the dense FFN summed with the experts is the shared-expert slot under
the dense names plus a scale on the sum, two graphs of 140, and Grok-2's
refusal by name lifted with it; the routed branch reading the layer
input under a second norm is `RouterInput::NormedLayerInput`, one graph
of 140). Each with a libllama-golden fixture. `minicpm` closed on
2026-09-10 and is not in that arithmetic: it was refused BY NAME rather
than as unaudited, so it raises the audited count without lowering the
refusing one; `smollm3` and EXAONE-4 32B closed with `exaone-moe` on
2026-09-11 and are the same case, one a DedicatedOnly refusal and the
other a refusal by name.

The three UNKNOWN rows `mistral`, `mixtral` and `yi` closed the same
day by turning out not to be architectures: libllama refuses all three
strings outright and every real checkpoint of all three declares
`llama`, so they are refused as spellings now rather than triaged as
graphs. `phi4` is the one UNKNOWN left.

**The NEW CODE column moved for the first time on 2026-09-10**, three
times: 26 to 24, 24 to 21, then 21 to 20, and on 2026-09-11 seven times
more, 20 to 19, 19 to 17, 17 to 14, 14 to 12, 12 to 11, 11 to 9 and 9
to 8. The first two took several rows
at once for the same reason -- each found ONE cause behind several
refusals. The fourth did too and the count hides it: the per-layer RoPE
gate closed three refusals, and only `exaone-moe` was in this column.
The sixth is the per-layer shape seam, whose reach was measured across
all 140 graphs before it was built (`layer_shapes::PER_LAYER_SHAPE_ARCHS`
is the record): it closed `deci` and `openelm` and narrowed `laguna`,
`mimo2` and `step35` to what else each needs. The seventh took the
seam's leftovers: `afmoe`, `laguna` and `step35` had been narrowed to
the same last word, `wqkv_gate`, and reading the three graphs side by
side found one op with two free parameters (`frink_models::attn_gate`),
so two closed and the third says the gate is done. `mimo2`'s sinks
became a tensor-presence fact on the same day without closing it:
every real export carries MTP blocks and a per-layer window array;
it closed on 2026-09-12 on its split K/V head width.
The tenth, `mistral3`, is what a reach measurement looks like when it
comes back with one: the other two graphs that build the temperature
input are on other engines (`llama4` from literals, `deepseek2` /
`mistral4` from the same key, which the MLA loader refuses by name
now where it dropped it), and the verdict's second half -- one GGUF
key, `yarn_log_multiplier` -- found YaRN's magnitude term missing for
every architecture on the generic path (`frink_models::yarn_magnitude`).

`olmo` is the one that did not, and it is worth reading for the way the
question was settled rather than for the row. "What else shares this
cause" was answered by MEASUREMENT before any code was written: every
`build_norm` call in all 140 of llama.cpp's `src/models/*.cpp` graphs
was scanned for a null weight argument, and all three hits are
`olmo.cpp`. `openelm`, `bitnet`, `arcee`, `mellum`, `nanbeige` and
`deci` were the rows the search was aimed at and not one of them norms
without parameters -- they are checked-and-recorded now instead of
still-plausible, which is most of the value. The LayerNorm *function*
IS shared, by `dbrx` and the `nemotron` / `orion` / `stablelm` /
`codeshell` / `jais2` / `starcoder` / `starcoder2` / `phimoe` bias
group; `capability::NON_PARAMETRIC_LAYER_NORM` carries the finding.
Read row by row on 2026-09-12, two of the eight turned out to need
nothing but the biased variant, and `orion` and `nemotron` closed on
`NormOp::LayerNormBias` (`capability::BIASED_LAYER_NORM`,
`tests/biased_layer_norm_graphs.rs`), and `starcoder2`, `codeshell` and
`jais2` followed on `frink_models::proj_bias`, the projection biases
whose reach over the 140 graphs is 33 and 27 files, most of them
OPTIONAL -- a `llama` file with biases used to be refused as unread;
`stablelm` followed on the norm alone once `stablelm.cpp`'s two other
shapes -- the parallel residual a layer without `ffn_norm` builds,
and the per-head LayerNorm QK norm a layer with `attn_q_norm` builds
-- had a refusal by name each, with the reach of both measured
(`frink_models::parallel_residual`: eight graphs in two spellings;
`frink_models::qk_layer_norm`: three); the two left say what else.
The parallel residual landed the same day as the seam the table had
sized: `gptneox` (Pythia) and `plamo` closed on it, one per arm, and
the `stablelm` parallel fixture matches where it was refused
(`tests/parallel_residual_graphs.rs`). What it cost the bodies is one
value captured before attention beside the router's operand
(`decoder::ffn_block::BranchInputs`) and one match on it where each
body computed `ffn_norm(h)`; what it cost the fused Metal launches is
one predicate clause. `command-r` (Command-R 35B, Aya-23) followed
the same day, because what it needed on top was two things that
already existed: `dbrx`'s weighted LayerNorm without a bias, and the
`logit_scale` multiply `grok` and `talkie` had, in an OPTIONAL form
the graph skips at zero (`tests/command_r_graphs.rs`, KL 1.0e-15;
Command-R+'s QK LayerNorm refused from a 64-layer fixture). `falcon`
followed on BOTH arms: Falcon-7B is the shared norm, and Falcon-40B's
optional `attn_norm_2` is the two-norm arm with the tensor names
crossed relative to `gptneox` (attention under `attn_norm_2`, the FFN
under `attn_norm`; `norm_sites::ATTN_NORM_2_FEEDS_ATTENTION`, decided
per layer, `tests/falcon_graphs.rs`). `phi2` followed on 2026-09-14
on ONE slot, `output.bias` on the LM head (`Decoder::output_bias`,
`proj_bias::OUTPUT_BIAS_CREATORS`: three graphs of 140), added in the
one place the head's post-projection transforms run and fenced off the
fused argmax stacks (`tests/phi2_graphs.rs`). `cohere2` (Command-R7B)
followed the same day, and it is a census correction: its "rotation on
the sliding layers only" is `rope_layers::SlidingOnly`, the rule
`exaone-moe` had closed on, which the module's census of six had
missed because it grepped for `use_rope` and `cohere2.cpp:91` spells
the gate `if (is_swa)` (`tests/cohere2_graphs.rs`, KL 1.0e-14; the
window key REQUIRED upstream is refused when absent, measured against
libllama's own refusal). `cohere2moe` closed on 2026-09-14 on three
rows (`tests/cohere2moe_graphs.rs`, KL 1.7e-14): the
`|| il < n_layer_dense_lead` rotation variant its refusal had
recorded (`RopeLayers::SlidingOrLeadingDense`), the `0.5` on
`moe_out + shexp` (`parallel_dense_ffn::SHARED_EXPERT_SUM_SCALE`, the
field Grok-2's sum scale already fills), and a norm FUNCTION the FILE
decides (`norm::NORM_BY_RMS_EPS_KEY`, one graph of 140); its window
array is read at trunk length because it reads the MTP count first
(`swa_layers::ARRAY_AT_TRUNK_LENGTH`), and the MTP block itself was
already `mtp_blocks`' (libllama's golden for the file with the block
is byte-identical to the trunk's). No parallel-residual row is
refused any more. `phimoe` (Phi-3.5-MoE), the last of the bias group but
`starcoder`, closed the same day and corrected its own refusal on the
way: the norm biases it "required LayerNorm" for are RMSNorm biases
(`phi3.cpp:99-102` under `LLM_NORM_RMS`, `NormOp::RmsBias`, one graph of
140), and its two projection biases were slots already
(`tests/phimoe_graphs.rs`, KL 1.9e-11); the window key it writes is
dead metadata as `phi3`'s, and a test had asserted the opposite.
`gpt2` and `starcoder` closed together the same day on the seam the
bias group's last row had named: a learned position table added to the
embeddings and NO rotation (`frink_models::position_embd`,
`rope_layers::RopeLayers::Never`; three graphs of 140 create the
tensor, `mpt`'s optional). The bias group of `tests/attn_bias.rs` is
empty (`tests/position_embd_graphs.rs`, KL 1.9e-7 each). ALiBi
followed the same day as ONE seam for FIVE rows: `refact`, `bloom`,
`mpt`, `jais` and Baichuan-13B, whose reach was measured over the 140
before a line was written (seven graphs set `f_max_alibi_bias`, five
on the generic path, in three spellings). The slopes are one function
in `frink-core` and one additive term in the three host attention
kernels; `rope_layers::Never` is derived from the same table, so the
bias and the absence of rotation cannot disagree about Baichuan's
layer count; every fused GPU path refuses. Each row had one more
thing that was a table entry: `bloom`'s embedding norm, `jais`'s
`1/d` attention scale, `mpt`'s clamp (`tests/alibi_graphs.rs`).
`minimax-m2` (MiniMax-M2) closed the same day with NO code: its
refusal had said "unaudited, not unimplemented, a fixture away" for a
week while the fixture sat in `tests/fixtures/`; running it through
libllama gave the golden, KL 3.4e-15 on the first try
(`tests/minimax_m2_graphs.rs`).
`lfm2` and `lfm2moe` (LFM2, LFM2-8B-A1B) closed the same day as the
FIRST hybrid rows, and not on the hybrid engine: `lfm2.cpp:192-208` is the generic layer with a
short convolution where attention would be, so it is a third
`AttnShape` (`frink_models::shortconv`) with its state kept as the
layer's KV history, KL 3.2e-12 on three fixtures and 6.0e-13 on the
MoE's (`tests/lfm2_graphs.rs`). Reach measured first: two graphs of 140
build the block; the Mamba-2 hybrids share the "zero KV heads means
recurrent" rule and nothing else, and `layer_shapes::ZeroKvLayer`
names each one's block.
`pangu-embedded` (openPangu-Embedded-1B / 7B) closed the same day out
of the DEFERRED column, where no row had ever closed from: it had been
filed as "embedding variant" from its name, and it is a decoder LLM
whose graph is `llama.cpp`'s with a required `attn_output.bias`, one
row in `proj_bias` (`tests/pangu_embedded_graphs.rs`, KL 1.5e-13).
`granitehybrid` (Granite 4.0) closed the same day as the first MAMBA-2
row, on the second recurrent seam: where LFM2's conv state was a
window and rode as KV history, a Mamba state is a reduction and rides
as `frink_core::recurrent_state::RecurrentState` beside the layer's
cache, cloned and cleared with it and REFUSED a truncate to a middle
position, which is what fences the prefix cache and speculative
decoding off such models (llama.cpp's server re-prefills them for the
same reason). `frink_core::mamba2` is ggml's conv and scan steps;
`frink_models::mamba2` is `build_mamba2_layer` once, for the four
graphs that call it. KL 1.9e-13 / 7.9e-13 / 1.0e-13 on the NoPE,
rotated and MoE fixtures (`tests/granite_hybrid_graphs.rs`); the
Granite `rope.scaling.finetuned` refusal became `RopeLayers::Never`
on the way, with its own fixture's golden.
`nemotron_h` (Nemotron-H) closed the same day on the same Mamba-2 seam
plus two table rows for its one-block-per-layer topology: a block with
no FFN whose output is ADDED (`BLOCK_WITHOUT_FFN_KEEPS_ITS_OUTPUT`;
deci's is discarded, which is why that combination was refused) and an
FFN-only layer whose pre-norm is `attn_norm` (`norm_sites::
ONE_NORM_PER_LAYER`), KL 2.0e-13 (`tests/nemotron_h_graphs.rs`).
`nemotron_h_moe` (Nemotron-3 Nano 30B-A3B) followed in the next PR:
ungated ReLU-squared experts and shared expert (the gate aliased to
`up` as the dense loader already did), the sigmoid literal and the two
`expert_weights_*` keys in their reader tables, `moe_latent_size`
refused by name, KL 3.6e-13.
`falcon-h1` (Falcon-H1) closed next on the same block in a third
position: IN PARALLEL with attention on every layer, both reading the
same normed input, summed before the residual (`ModelConfig::
parallel_ssm`; one pair of helpers for the three host bodies), KL
1.3e-13 (`tests/falcon_h1_graphs.rs`). Every Mamba-2 caller of the
140 graphs is served now. `jamba`, `mamba` and `mamba2` closed next
on Mamba-1's `build_mamba_layer` (`frink_models::mamba1`; the scan
kernel gained its per-state decay arm, `Decay::PerState`) and on
`layer_shapes::PURE_RECURRENT`, the rule that a model with no heads
anywhere is every layer the block, KL 7.3e-12 / 2.3e-12 / 3.6e-13
(`tests/mamba_graphs.rs`). Of the Mamba family only `plamo2`'s own
spelling is left.
`qwen35` (Qwen3.5 dense) closed next, and it is the row the hybrid
engine scaffold had been waiting for since it was written: the gated
delta net went on the SAME seam the Mamba blocks did (`frink_core::
gdn` is the delta rule as `delta-net-base.cpp` computes it, `frink_
models::gdn` the block, `AttnShape::Gdn` the site), the attention
layers' interleaved `wq` gate is split after the projection, and the
1.8k lines of `gdn.rs` / `hybrid_gguf_loader.rs` that had never met
libllama are deleted. KL 4.1e-13 on the first run
(`tests/qwen35_graphs.rs`); grouping the V heads instead of tiling
them, or dropping the gate's sigmoid, each turns five tests red.
`qwen35moe` followed with no code: `qwen2moe`'s FFN under Qwen3.5's
layers, KL 2.9e-11. `qwen3next` followed on two tables: the grouped
head map the port had assumed for everyone, and the fused `ssm_ba`
projection, KL 8.7e-12. Every gated-delta-net graph in llama.cpp is
served; of the hybrid family only `plamo2`'s Mamba-1 spelling is
left.

`llama4` (Scout, Maverick) closed on 2026-09-14 on four seams, three
of them a per-layer gate on a seam that existed. The new one is
`frink-models/src/chunked_swa.rs`: `llama4.cpp:13-14` set
`LLAMA_SWA_TYPE_CHUNKED` at a literal 8192 and `llama-hparams.h:
419-425` mask every key before the query's own chunk, so a query at
`p` sees `p % 8192 + 1` positions where a sliding layer sees a
constant; `ModelConfig::layer_window_for_query` is that number, the
batched prefill takes a per-query arm when a batch straddles a
boundary (`BatchWindow`), eviction keeps the last 8192 rows (a
superset of any chunk), and the fused Metal launches refuse the
model. The temperature is the literal row of `attn_temperature` with
`unrotated_layers_only`; the weightless per-head QK norm after RoPE
on the rotating layers is a `bool` (`weightless_qk_norm`, one
reachable graph of 140); the interleave step is honoured because
`llama4.cpp:64` is the ONE tensor loader that branches on it. The
fourth was not in `llama4.cpp` at all: `llama-graph.cpp:1947`
multiplies the sigmoid weight into the expert's INPUT for
`LLM_ARCH_LLAMA4` alone (`routed_weight_site`), and the fixture's
logits move by 0.86 between the two sites. Building it found
`route_top_k_sigmoid` renormalising whatever `norm_topk_prob` said
(every sigmoid row so far had declared the key true). KL 1.1e-12 on
both shapes and at the last of 8200 positions across the chunk
boundary, on the prefill and row bodies; `--noswa` (libllama aborts,
`llama-graph.cpp:159`) and `--dense` (libllama refuses) are refusals
by name.

`olmo2` and `exaone4` closed TOGETHER, because they are one residual
topology and not two. Neither has an `attn_norm` or an `ffn_norm`
tensor; both read the raw residual at each sublayer and norm each
branch's output before its residual add (`olmo2.cpp:45-52,92,160-182`,
`exaone4.cpp:60-67,118,152-169`, line for line the same graph).
`frink_models::norm` is the one implementation and
`tests/post_norm_only_graphs.rs` the evidence, a libllama-golden fixture
each. One sub-case stays refused BY NAME rather than being swept in: an
`olmo2` carrying both a sliding window and a RoPE scaling (Olmo-3) ropes
its two kinds of layer differently. EXAONE-4 32B (`block_count == 64`),
whose full-attention layers get no RoPE at all, was the other and is
closed (next paragraph but one).
`olmo` (OLMo-1) is a THIRD shape -- pre-norm with a non-parametric
LayerNorm at all three sites, `olmo.cpp:65-67,104-106,128-130` -- and
closed as a third variant of the same enum
(`tests/olmo_graphs.rs`). `Decoder::final_norm` became a `NormOp` with
it: OLMo-1's final norm has no weights either, and the fused Metal
stacks that fold `final_norm + lm_head + argmax` had
`Some(&self.final_norm)` written into them unconditionally. Half its
verdict stayed a refusal, and the half that looked like an aside:
`olmo.cpp:5` reads `{arch}.attention.clamp_kqv`,
`llama-graph.cpp:1611-1652` clamps Q, K and V by it, and
`conversion/olmo.py:23-25` writes it for every checkpoint whose HF
config carries a `clip_qkv` -- OLMo-7B-Twin-2T and OLMo-1.7-7B do, the
original OLMo-7B does not. A second fixture measures that llama.cpp's
own logits move when the key is present, so it is not a no-op that
could be ignored.

`exaone-moe`, `smollm3` and EXAONE-4 32B closed TOGETHER on 2026-09-11,
on the per-layer RoPE gate, and the pairing was CHECKED before it was
assumed: `exaone4.cpp:116` is `use_rope = is_swa(il) || swa_type ==
NONE`, `exaone-moe.cpp:136,155-161` is `is_swa(il)` around the same two
`ggml_rope_ext` calls, and `exaone-moe.cpp:4` pins `swa_type` to
`STANDARD`, which nails the second disjunct false. Identical, not
similar. `smollm3.cpp:5,69` is a different variant of the same enum
(`(il + 1) % 4 != 0`, no window). `frink_models::rope_layers` is one
table for all six architectures llama.cpp gates this way, with
`smallthinker`, `afmoe` and `llama4` in it; all three closed later
on other seams. The durable part is the type: `ModelConfig::layer_rope` returns
`Option<(base, divisors)>`, so a rotation site cannot take the pair
without answering whether to rotate, and the Metal stacks take an
`Option<LayerRope>` per layer -- their RoPE dispatch had been written in
unconditionally, the OLMo-1 final-norm shape again -- while the four
per-layer Metal launches lost their loose base/divisor parameter pair
for one `LayerRope`. `tests/no_rope_layer_graphs.rs`: KL 2.05e-12 on a
64-layer EXAONE-4 fixture (64 because `exaone4.cpp:4` tests equality),
1.43e-14 on `exaone-moe`, 5.29e-15 on `smollm3`. Found on the way:
EXAONE-4 1.2B must IGNORE a window its file declares
(`exaone4.cpp:4-14` reaches `set_swa_pattern` only at 64 layers), now
in `capability::swa_disabled_by_arch` beside `phi3`; and
`nextn_predict_layers` -- MTP blocks INSIDE `block_count`, which
llama.cpp skips -- was refused nowhere, so a real EXAONE-MoE export
with an MTP head would have run it as two extra decoder layers. (Since
2026-09-11 the blocks are skipped as llama.cpp skips them,
`frink_models::mtp_blocks`, and the per-layer window array that the
same exports carry is `frink_models::swa_layers`.)

`minicpm` was never an unaudited row: it was refused BY NAME, because
`minicpm.cpp:5-7` assigns an embedding multiplier of 12.0, a residual
multiplier of `1.4/sqrt(n_layer)` and a logit multiplier of `256/n_embd`
BEFORE `:12-14` lets the file override them, so a key-PRESENCE gate sees
nothing in a file that is still scaled three ways. It runs Granite's
graph verbatim (`models.h:1594-1601`), so the fix was a DEFAULTS field
on the table `scalar_multipliers` already had, and the fixture that
evidences it declares no scaling key at all -- the only fixture shape
that can tell the hook from its absence. A second one declares all three
and pins the merge ORDER, which one fixture cannot see.

`granite`, `granitemoe` and the `granite-moe` alias closed together too,
on ONE implementation of the four scalar multipliers they share
(`frink_models::scalar_multipliers`, `tests/granite_family_graphs.rs`).
`granite-moe.cpp` has no graph of its own -- `models.h:1583-1591` is
`using graph = llama_model_granite::graph` -- so the two upstream rows
differ in the FFN and in nothing else, and the third is a frink-only
alias for the second. Deriving
`capability::unsupported_scaling_keys` from that same table instead of
restating it beside it found a live gap on the way past: the Gemma
family was exempted from all four keys while reading none of them, so a
hand-written `gemma3.residual_scale` would have loaded and been ignored.
Half the Granite verdict stayed a refusal for four days: llama.cpp
reads `{arch}.rope.scaling.finetuned` as a switch for RoPE itself, and
a file declaring it false runs unrotated, which frink could not express
until `RopeLayers::Never` existed; it is served since 2026-09-14, when
Granite-4.0 (whose every export writes the key false) closed on it.

| | llama.cpp | frink |
|---|---|---|
| Per-architecture graphs | 140 hand-written | 150 catalog rows, **37 proven** |
| Metal `pp512` | baseline | 0.98x-1.10x, at parity |
| Metal `tg128` | baseline | **8 of 12 rows faster** |
| CPU, all rows | baseline | **1.41x-5.06x slower** |
| GPU backends | CUDA, Metal, Vulkan, SYCL, HIP | CUDA, Metal |

Do not read the architecture catalog as a support matrix.

## The rules that keep being re-learned

**A plan's own status field is a claim, not evidence.** Verify against
the code. A merged PR once marked `paged-decode-path` complete while it
returned wrong tokens on Metal.

**One agent owns a file.** Two branches editing the same file produce a
merge nobody can review, and this project has already had one branch
silently revert three others.

**No agent runs benchmarks.** Measurement needs a quiet host, and a
loaded run reads 25-45% low.

**Refusing is not a defect.** llama.cpp will often run something
approximately; this project stops and names what is missing. A refusal
is a gap in coverage, not a bug.
