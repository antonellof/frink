# CLAUDE.md

Guidance for agents working in this repo.

## What this is

Pure-Rust GGUF / MoE inference engine: mmap loaders, quantized CPU +
Metal + CUDA kernels, OpenAI-compatible `frink-server`.

**The goal is to be the Rust alternative to llama.cpp**: same models,
same command shapes, same or better performance, on the hardware people
actually own. `docs/plans/north-star.md` is the ranking every other plan
is read through, and `docs/plans/README.md` is the index.

Honest position, re-audited 2026-09-19 against a MOVED PIN. **100**
architectures run with evidence (`capability::AUDITED_GENERIC_GQA`), 4
more have dedicated engines, and everything else REFUSES. The "loads
and is WRONG" class is closed: the generic path is opt-in, so an
unaudited architecture stops instead of guessing.

**The refusing count went UP on 2026-09-19, from 2 to 10, and that is
what parity with a moving target looks like.** The pinned llama.cpp was
six weeks and 792 commits old; moving it to `5b59b83` brought fifteen
new graphs, of which twelve are text generation. Eight are generic-path
candidates and are triaged (`granite_swa`, `graniteswitch`,
`muse-glimmer`, `maple`, `spark2_5`, `hrm_text`, `minimax-01`,
`qwen4exp`), four need an attention this engine does not have and are
refused by name (`bailingmoe3`, `dots3note`, `hy_v4`, `kimi-k3`), and
two are text-to-speech. A count that only ever fell would mean nobody
was reading upstream. `docs/plans/parity-audit-2026-09-19.md` is the
re-measurement of the llama.cpp surface and the serving features.

`spark2_5` (Spark-2.5 1.7B) closed the same day, which is what a ONE
MATCH ARM verdict is supposed to cost: `spark2-5.cpp:41,97-105` is a
per-head sigmoid attention gate, one row of `crate::attn_gate` beside
`step35`'s, and the rest of its graph -- the window ARRAY with
`rope.freq_base_swa`, per-layer head counts that SIZE the gate, a gated
GELU FFN -- was already served, each by an existing table that had to
gain one name. KL 7.70e-7 at the GELU-table line, and with frink's
GELU made to emulate ggml's f16 table the same file agrees to 6.08e-6
(KL 3.34e-12), so the residual is llama.cpp's own approximation and the
row is pinned at 5e-3 with that measurement beside it. Three sabotages
(the activation, the window-array reader, the GeGLU row) each move the
logits by more than 0.8.

`maple` (Maple-20B) closed the same day and is the better lesson,
because its verdict was RIGHT about the row and INCOMPLETE about the
graph. The row was one `crate::rope_layers` entry --
`maple.cpp:88` rotates only the sliding layers, `cohere2`'s rule -- and
with the clamp arrays zeroed the fixture matched libllama exactly.
With them nonzero it was 0.12 off, and the cause is not in
`maple.cpp` at all: `llama-graph.cpp:2228` sends FOUR architectures
(`maple`, `deepseek4`, `hy_v4`, `dflash` with hyper-connections) to
`ggml_swiglu_clamp`, which clamps the gate BEFORE the SiLU
(`ggml/src/ggml-cpu/ops.cpp:3450-3454`), where every other graph takes
the `else` branch and clamps the SiLU's OUTPUT. `frink_moe::ClampForm`
is the two forms and `act_layers::CLAMP_BEFORE_SILU` the list; the two
agree wherever `silu(x) <= limit`, so a fixture whose clamp never binds
cannot tell them apart. Reading the graph file is not enough when the
graph calls a GENERIC builder: the builder branches on the
architecture too.

`granite_swa` (Granite 4.1) closed the same day as the third row, on
`RopeLayers::FileMask`: `granite-swa.cpp:43` reads
`attention.rope_pattern`, one entry per layer, and it is the FIRST
upstream graph that lets the FILE say which layers rotate where every
other per-layer RoPE gate here is a rule read out of a literal
(`grep -rn LLM_KV_ATTENTION_ROPE_PATTERN src/models/*.cpp` is that one
line; `llama-model.cpp:1314` seeds the array with 1 for everyone, so
honouring it elsewhere would diverge on a file carrying it as dead
metadata). Its other blocker, the `expert_used_count` ARRAY, had
already been fixed as a silent default. The fixture's window array and
rope pattern DISAGREE about which layer is special -- layer 1 is the
full-attention one, layer 2 the unrotated one -- so a loader that read
either into the other's field is caught by the golden rather than by
luck.

`muse-glimmer` closed the same day as the fourth, on two norm facts
nothing else upstream has. `muse-glimmer.cpp:69` norms the EMBEDDINGS
with a weightless RMS -- `bloom`'s embedding norm, the only other one
of the 155, has a weight, and every other weightless RMS is a layer
slot -- and `:63` runs the post-attention and post-FFN norms at a
LITERAL eps of 1e-8 while the pre-norms use the model's key.
`norm_sites::WEIGHTLESS_EMBEDDING_NORM` and
`norm::POST_NORM_EPS_LITERAL` are the two tables,
`ModelConfig::post_norm_eps()` the one accessor the three host
post-norm sites read, and every fused Metal launch and the CUDA
prefill refuse a model whose two epsilons differ, because each bakes
ONE epsilon into its kernel. Its fixture declares the model epsilon
four orders larger than the literal, so the two cannot be swapped
without moving the logits.

`hrm_text` (DFM Mimir 1B) closed the same day as the fifth, and it is
the first decoder here that is not a walk down ONE residual stream.
`hrm-text.cpp:183-196` runs `h_cycles` cycles of `l_cycles` LOW stacks
and one HIGH stack; every stack reads `zH + zL` and replaces one of
the two, `zH` starting as the embeddings and `zL` as a learned
`hrm.z_l_init` row, and the lm_head reads `zH` with no final norm
because every stack ends with its own weightless RMS. Its stacks are
ALIASES: the file holds `2 * layers_per_stack` blocks while
`block_count` is the expanded slot count, and each slot keeps its own
KV. `crate::layer_loops::LayerLoops` is an enum now -- `Repeat`
(nanbeige) and `Hrm` -- so the mapping, the pass norm and the stream
schedule are one value the four host bodies ask, and `crate::hrm` is
the two-stream state, one type with two methods rather than four
copies of "hold two vectors". Two fixtures: the alternating schedule
matches EXACTLY and the deep one (two LOW passes in a row, the case
the alternating order never reaches) at a measured line. Every stack
ends with a weightless RMS, so a renormalised residual amplifies the
1e-7 two f32 reduction orders differ by: two stacks exact, three
3.5e-5, six 1.6e-4 on arm64 and 1.1e-3 on x86_64 CI. The deep fixture
is the THREE-stack schedule for that reason -- a tolerance widened
until it passes everywhere is evidence of the machine, not of the
model.

`minimax-01` (MiniMax-Text-01, 456B-A45B) closed on 2026-09-20 as the
sixth, and it is the row where the BLOCK was the cheap half. Its
recurrent mask is Qwen3.5's -- the same two keys, read by the same
`crate::gdn::recurrent_layers`, with the interval seeded 8 instead of 4
-- so `crate::gdn::INTERVAL_RECURRENT_ARCHITECTURES` gained a third
column, the AttnShape those layers run, rather than a second reader;
`RecurrentMask` is the mask and the block as ONE value from ONE
constructor, because a file whose mask came from one architecture and
whose block came from another is this repo's dominant bug shape.
`crate::lightning` is the block on the `AttnShape` seam the Mamba rows
built, one `SsmBlock` variant, with its state -- one `head_dim x
head_dim` KV per head, `llama-hparams.cpp:249-253` -- riding the cache
as every other recurrent block's. TWO facts in it are invisible in the
tensor shapes and were both wrong here for a day: `minimax-01.cpp:303`
runs the WHOLE fused projection through SiLU before the split, and
`:305-309` read it HEAD-major (`[q|k|v]` per head) rather than as three
blocks -- a permutation that is the identity at one head, so the
fixture has four. What actually needed new code is the RESIDUAL:
`:249,428-431,440,455-458` make each sublayer's PRE-NORM OUTPUT, times
a REQUIRED `residual_scale`, the stream its branch joins, and the layer
input (`inpSA`, `:244`) is bound, sliced by `inp_out_ids` at `:424`,
and never added to anything. `crate::normed_residual` is that fact, one
graph of the 155, measured by reading each `ggml_scale` argument in the
five files that read the key; `ResidualScaleUse` is the ONE column of
`MultiplierSupport` that answers both meanings, so the key cannot be
given twice; and `Decoder::pre_norm_residual` is the one function every
host body norms the stream through, because a site that normed by hand
would read the right vector and keep the wrong topology --
`no_host_body_norms_the_stream_by_hand` greps for exactly that, with
the whitespace stripped so `cargo fmt` cannot silence it. The identity
scale is NOT passed through `scale_or_none`, because the layer input is
discarded at 1.0 too, and the fixture that declares 1.0 is what pins
that: libllama's logits for it are not the ordinary graph's. KL 5.2e-10
/ 9.5e-11 / 1.3e-9 / 1.3e-11 on four fixtures, at the `phimoe` class
because every layer is a four-expert MoE; four sabotages each move the
logits by more than 3.

So **four** triaged refusals are left, and they say which of three
things is missing: **0 are a fixture away, 0 are one match arm away**,
3 need new code, 1 is unknown with the question stated. Both cheap
classes are EMPTY again, which is where they were before the pin
moved; SIX of the eight rows the pin brought in have closed, and the
two that are left each need a block this engine does not have (a
per-token adapter selection, a delta-net over a hybrid memory index).

Three things the pin move found that are not new architectures at all:
`kimi_k3` had been spelled with an UNDERSCORE in the catalog since it
was added while `llama-arch.cpp:155` writes `kimi-k3`, so that refusal
could never fire on a real file; `nextn_predict_layers` is now read by
llama.cpp's COMMON loader for every architecture (upstream 9d81721),
which makes frink's seventeen-reader gate an over-refusal and is
recorded in `crate::mtp_blocks` as the next row; and
`expert_used_count` is read there as scalar-OR-ARRAY, where frink read
a scalar and silently defaulted to 2 for the array spelling that
`conversion/nemotron.py:574` writes for Nemotron-H Puzzle -- an
architecture frink SERVES. A uniform array is honoured now and a
varying one stops by name. Five
one-match-arm rows closed on 2026-09-02, seven fixture-away rows on
2026-09-03, `gemma`, `hunyuan-dense` and `ernie4_5-moe` on 2026-09-09,
and `olmo2`, `exaone4`, `chatglm`, `qwen`, the three Granite rows and
`olmo` on 2026-09-10, and `exaone-moe`, `grok`, `dbrx`, `arcee`, `deci`,
`openelm`, `afmoe`, `laguna`, `mellum`, `apertus`, `step35` and
`mistral3` on 2026-09-11, and `smallthinker`, `bitnet`, `mimo2`,
`nanbeige`, `talkie`, `plm` and `arctic` on 2026-09-12, each with a
libllama-golden fixture, which is what moved 46 to 41 to 34 to 31 to 29
to 28 to 25 to 22 to 21 to 20 to 18 to 15 to 13 to 12 to 10 to 9 to 8
to 7 to 6 to 5 to 4 to 3 to 2; the step from 28 to 25
was moving the three alias rows off
the generic path rather than a closure, and the step from 4 to 3 moved
`plm` onto the MLA ENGINE rather than the generic path, so it lowers
the refusing number without raising `AUDITED_GENERIC_GQA` -- the
engine it joined now has a libllama golden, which it never had. `minicpm` moved too and is not in that count: it
was refused BY NAME, never as unaudited, so it raises the audited number
without lowering the refusing one; `glm4moe` (GLM-4.5 / 4.5-Air / 4.6)
is the same case on 2026-09-12, a `DedicatedOnly` refusal on its norm
slot that turned out to be ONE row in `norm_sites::
PRE_FFN_NORM_IS_POST_ATTENTION_NORM` -- the refusal had named the slot
for a year and nobody had tried the row on the table that already
served gpt-oss's identical slot; KL 1.5e-15 on the first run
(`tests/glm4moe_graphs.rs`). `glm4` (GLM-4-0414) followed the same
day with NO code change at all -- its profile moved from `dedicated`
to `gqa_norm` and the fixture matched at 9.7e-15 -- because the row
had been sent to the GLM-5.2 MLA loader for keys `glm4.cpp` never
reads, exactly the `glm4moe` defect on the family's dense members.
The one thing either needs beyond the generic path is
`frink-models/src/mrope.rs`: a vision export's text tower declares
`rope.dimension_sections`, and llama.cpp's M-RoPE on text positions
is NEOX band for band, which is `glm4moe`'s layout already (served,
libllama byte-identical) and is NOT `glm4`'s NORM -- the converter
permutes those weights to NEOX order, libllama's logits move by 0.72,
and frink refuses the file by name. `smollm3` and EXAONE-4 32B closed
with `exaone-moe` and are the same case, one a DedicatedOnly refusal and
the other a refusal by name; the clamped OLMo-1 checkpoints closed with
`dbrx` the same way.
BOTH cheap classes being EMPTY is the honest headline: nothing still
refusing is one fixture or one arm away, so every row that is left
needs a different graph.

**On 2026-09-10 the NEW CODE column moved for the first time**, three
times: 26 to 24, 24 to 21, then 21 to 20, and on 2026-09-11 seven times
more, 20 to 19, 19 to 17, 17 to 14, 14 to 12, 12 to 11, 11 to 9 and 9
to 8, and on 2026-09-12 seven times more, 8 to 7, 7 to 6, 6 to 5, 5 to
4, 4 to 3, 3 to 2 and 2 to 1. The first two took several rows
at once for the same reason, and it is the lesson: each found ONE cause
behind several refusals. The fourth did too and the column hides it:
the per-layer RoPE gate closed THREE refusals and only `exaone-moe` was
in the column. The fifth is the lesson's other half: `grok` and `dbrx`
each closed by extending a seam that had landed the day BEFORE -- the
MiniCPM defaults hook, the `NormOp` enum, the norm-slot decision -- by
one column, and the clamp `dbrx` needed closed a third row's
refusal-by-name with it. The sixth is both lessons and a correction:
`deci` and `openelm` closed TOGETHER on the per-layer shape seam, whose
reach was MEASURED across all 155 graphs before it was built, and
`arcee` closed ALONE because the verdict that said it shared a cause
with `plm` had been read from one file and not the other. The seventh
is the sixth's lesson applied to the sixth's leftovers: the shape seam
had narrowed `afmoe`, `laguna` and `step35` to the same last word,
`wqkv_gate`, and reading the three graphs SIDE BY SIDE before calling
them one cause found one op with two free parameters rather than one
graph -- two rows closed on it, the third says so. The eighth is the
lesson with its count hidden the other way round: the per-layer window
ARRAY was named by THREE places (`mimo2`, `step35`, and #193's report
on EXAONE) and moved the column by ONE, because the row it closed
(`mellum`) was none of the three -- it is the only generic-path graph
that HONOURS the array -- while the three it was aimed at were either
over-refused on a value llama.cpp ignores (lifted, with a fixture that
measures the ignoring) or still need something the seam does not touch.
Reading `get_key_or_arr`'s two overloads before assuming "an array is
per-layer truth" is what found that: for fifteen graphs it is dead
metadata, and honouring it would have been wrong on every real EXAONE
and Olmo-3 file. The ninth is the seventh's leftover and the question
the task asked first: are xIELU's four per-layer arrays and Step-3.5's
two per-layer clamp arrays ONE seam? Read side by side, they are one
plumbing question -- "layer `il` runs its FFN activation with these
scalars", `get_key_or_arr` at `n_layer` length for both -- and TWO
activation bodies, with the clamp's routed-versus-dense SITE the one
thing the second needed of the plumbing that the first did not. Both
rows closed on it, 11 to 9. The tenth is what a reach measurement looks
like when it comes back with ONE: `mistral3` closed alone, 9 to 8,
because the other two graphs that build the temperature input are on
other engines, and its verdict's second half -- one GGUF key -- found
a defect in every YaRN checkpoint on the generic path. The eleventh
is the same measurement answering ONE for a different reason:
`smallthinker`'s router MECHANISM (a precomputed `probs_in`) is shared
with three graphs and its CAUSE (routing on the raw layer input) with
none on this engine. The twelfth is the smallest reach there is:
`bitnet` closed ALONE, 7 to 6, on two norm slots one graph of 155
creates, and the seam is a `bool` because there is no second shape to
name. The same day's OTHER reading did not close a row and is worth
as much: `grovemoe`'s graph, read against `modeling_grove_moe.py`,
feeds its chunk experts the routed experts' OUTPUT (`grovemoe.cpp:
148-152`) and gathers their weights at the CHUNK index
(`llama-graph.cpp:2035-2039`) where the reference does neither, so
there is no single graph to match and its verdict says so. The
thirteenth is the row the KV cache was built without: `mimo2` closed
ALONE, 6 to 5, on a V head width that differs from K's, whose reach is
one generic-path converter plus the MLA engine, which had carried the
pair since it existed. The fourteenth, `nanbeige`, 5 to 4, is the
verdict's last sentence taken literally: "it is the copy that has no
home" closed as a mapping, not a copy. The fifteenth, `talkie`, 4 to
3, is four seams for one row, each one graph of 155, so none could be
built for anything else and all four landed together. The sixteenth,
`plm`, 3 to 2, closed on an engine that already had its attention: the
work was a table of the three ways it differs from DeepSeek-2 and the
engine's FIRST golden, and the table's direct-Q column had a second
caller waiting -- every lite DeepSeek-V2, which the loader had refused
for a key llama.cpp does not read on those files. The seventeenth,
`arctic`, 2 to 1, is the reach measurement coming back with TWO where
the verdict had implied one: the dense FFN summed with the experts is
Grok-2's shape as well, refused by name for a day and a half from a
fixture that now has a golden, and the two rows are one table with
two columns (presence, scale on the sum); the operand its verdict said
the router seam "does not reach" is a third variant of that seam.

`arctic` closed on `frink-models/src/parallel_dense_ffn.rs` and a
third `RouterInput` variant. `arctic.cpp:118-154` runs a dense SiLU
FFN sized `{n_embd, n_embd}` (`:38-42`) on `ffn_norm(ffn_inp)`, the
router AND the experts on `ffn_norm_exps(inpSA)` -- the layer INPUT
under a SECOND per-layer weight (`:45,135-152`) -- and sums the two
(`:154`). Reach, measured over every `build_moe_ffn` graph that also
reads a dense `ffn_up`: all but two use the triple on their leading
dense layers or as `_shexp`; `grok.cpp:171-184` (Grok-2: same `cur`,
GELU, the sum scaled by `sqrt(2)/2`) and `arctic.cpp` SUM it with the
routed output. So `PARALLEL_DENSE_FFN_ARCHITECTURES` is two rows and
two columns -- `DensePresence::{Required, Optional}` and `sum_scale`
-- served through the shared-expert slot, which was already "a dense
FFN on every token added to the routed sum, with the architecture's
dense activation", under the dense names, plus
`MoeWeights::parallel_sum_scale` applied to the whole branch beside
`apply_down_scale` at every site. `grep -l FFN_NORM_EXPS` is
`arctic.cpp` alone, so the operand is `RouterInput::NormedLayerInput`
carrying the one fact that distinguishes it from `smallthinker`'s
(`experts_read_router_operand`), captured by the ONE constructor
`Decoder::router_operand` at the point `attn_norm` is applied, and
`combine_ffn_outputs_for_position` / `moe_ffn_batch` take the routed
operand and the dense operand as TWO arguments so a caller cannot hand
the experts the wrong vector without saying so. Every fused Metal MoE
launch refuses the model through the predicate that refused
`RawLayerInput`, and through the shared-expert check. KL 6.23e-14
(arctic), the same golden for a file declaring
`expert_weights_scale = 2.5` (libllama byte-identical: `arctic.cpp:
3-14` never read it), 3.28e-10 for Grok-2 at the GELU-table tolerance.
One measurement rather than a sabotage: `grok.cpp:180` scales the sum
and `:186` RMS-norms it, and an RMSNorm is invariant under a positive
scalar up to eps, so the `sqrt(2)/2` moves libllama's own logits by
2.5e-4 and no more; the test pins that. Sabotaging the routed operand
at the row site and at the batch site each turns the golden red
(confirmed).

The MLA engine's second and third goldens are `deepseek2` itself, in
both tensor forms (`tests/deepseek2_graphs.rs`, KL 2.35e-15 and
3.57e-15). `frink-models/src/mla.rs`'s `MlaKvB` is the two forms as
one enum: `Combined` (`attn_kv_b`, the legacy converter's and `plm`'s)
expands the latent per head and attends with per-head caches
(`deepseek2.cpp:600-635`); `Split` (`attn_k_b` / `attn_v_b`, EVERY
DeepSeek export since the `_mla` keys existed, which the loader had
refused as "not wired") absorbs the query through `wk_b`, attends as
MQA over the latent `concat(c, k_pe)` and pulls the result through
`wv_b` (`:563-598`; `frink-core/src/mla_absorbed.rs`, whose unit test
pins the two forms equal), with the cache `kv_lora_rank + qk_rope`
wide instead of `n_heads * (qk_nope + qk_rope + v)`. `kq_scale` stays
`1/sqrt(qk_nope + qk_rope)` for both -- the latent width is the
plausible wrong number, and the sabotage that uses it moves the logits
by 2.2e-3. The fixture had never produced a golden because
`scripts/make_deepseek2_fixture.py` wrote `head_count_kv = n_head`
where `conversion/deepseek.py:307-308` writes 1 for every MLA export
("converts into MQA"); the script had blamed llama.cpp for the
`ggml.c:3942` abort. Its `--legacy-kv-b` file derives the combined
matrix and the split pair from ONE draw exactly as the converter
splits them, and libllama's two branches agree on the pair to 1.79e-7,
which is the number that says the derivation is the converter's.

YaRN on that engine is `frink-models/src/mla_yarn.rs`, and it is
three places in llama.cpp read in order, because a reading of any one
of them is wrong in a way only libllama's logits show:
`deepseek2.cpp:34-37` DIVIDE `rope.scaling.yarn_log_multiplier` by 0.1
(the converter writes `0.1 * mscale_all_dim`, "for legacy reasons");
`llama-context.cpp:194-231` compute the `attn_factor` handed to
`ggml_rope_ext`, with `LLM_ARCH_DEEPSEEK2` alone taking `mscale ==
mscale_all_dim` when the latter is not 1 (DeepSeek-V2's config has
both at 0.707) -- `mistral4` shares the loader and not the arch test,
which is why `mla_arch` has a column for it -- then cancel the
`(1 + 0.1 ln F)` ggml multiplies back in; and `deepseek2.cpp:438-448`
undo the cancel, take `mscale = attn_factor_org * (1 + 0.1 L ln F)`,
and fold `mscale^2` into `kq_scale`. `MlaYarn` is the three observable
pieces (per-band divisors on the `pe` slice, ONE magnitude on `q_pe` /
`k_pe`, ONE softmax scale), an ARGUMENT to `mla_forward_token` so no
rotation site or attention body is reached without it. For both real
shapes the magnitude comes out at exactly 1 -- libllama's own log line
says `setting new yarn_attn_factor = 1.0000` -- and the whole effect
is the rewrite plus `kq_scale`, which every real DeepSeek was missing
here. Three fixtures (V2's `0.707`, V3's `1.0`, V2 on the legacy
combined form), KL 2.98e-15, 4.42e-15, 2.94e-15; YaRN moves the plain
golden by 3.6e-3 and the two generations differ by 1.5e-3, and
sabotaging the `/ 0.1` turns both red. A `yarn` without
`original_context_length` and any other scaling type stay refused by
name. With the split tensors and YaRN both served, a real DeepSeek-V2
/ V3 export has nothing left that this engine refuses on its
attention; what it has not had is a real file run through it.

`plm` closed on `frink-models/src/mla_arch.rs` and `mla_q_proj.rs`, on
the MLA engine. `plm.cpp:84-166` is `deepseek2.cpp`'s naive MLA branch
line for line (`grep -l ATTN_KV_A_MQA` over all 155 graphs is six
files, `plm` the only one on no engine), and what differs is three
things, one table: a DIRECT `attn_q` (`plm.cpp:32`; `deepseek2.cpp:
104-115` creates the same when `q_lora_rank == 0`, and `:8,11-13`
decide that from the LAYER COUNT -- 27, 26, or 48 with a 128256
vocabulary -- BEFORE reading the key, so a lite file's `q_lora_rank`
is dead metadata and `MlaQProj` is an enum the forward pass cannot
reach without the answer), an ungated `LLM_FFN_RELU_SQR` dense FFN
(`:181-187`, `GluAct::ReluSqr` with the gate aliased as for `arcee`,
carried ON `MlaDenseFfn` so the body cannot run it through SwiGLU), and
a tied lm_head the graph never reads an `output.weight` for (`:23-24`;
libllama REFUSES a file carrying one, `done_getting_tensors: wrong
number of tensors; expected 30, got 29`, measured, so frink refuses it
too rather than prefer the decoy). The head widths come from
`attention.key_length` / `value_length` because `llama-hparams.cpp:
259-265` fall back to them when the `_mla` keys are absent, which is
what `conversion/plm.py:16-17` writes. KL 1.87e-13 on the MLA engine's
first libllama golden; the loader's lite rule has a 27-layer synthetic
test that carries the key and is direct anyway. Two things it found:
the engine had REQUIRED `attention.q_lora_rank`, so every
DeepSeek-V2-Lite / GigaChat3 / Kanana-2 export failed on a key upstream
never reads for them; and it read no `rope.scaling.*` at all, so every
real DeepSeek-V2 / V3 export (all YaRN) would have run at factor 1 with
`kq_scale` missing `deepseek2.cpp:312-319`'s mscale -- REFUSED by name
now, with the lines.

`talkie` closed on four seams. `NormOp::RmsNoParams` (`talkie.cpp:50,
68,90,110,137` are all `build_norm(x, nullptr, nullptr, LLM_NORM_RMS)`;
the RMS twin of OLMo-1's `LayerNormNoParams`, through the same
`NormFunction` table). `QkNormStyle::PerHeadScalar` (`attn_q_norm` is
`{1, n_head}`, `:26`, one scalar per head after a per-head RMS, applied
after RoPE with a weightless per-head K norm, `:82-91`; decided by
architecture because the length is ambiguous with `head_dim`; and a
per-head RMS is invariant under RoPE, so the order is honoured and
unobservable on this shape). `frink-models/src/skip_stream.rs` (the
normed embedding added into every layer's output times
`layer_output_scale`, `:52,123-126`; one `bool`, the norm at the ONE
embedding site, the add at the end of BOTH FFN bodies through an
`Option<SkipStream>` argument). And `weight_scales.rs` now SERVES the
two `.scale` companions talkie's converter writes (`attn_output.scale`,
`ffn_down.scale`; `AttnWeights::o_scale`, `MoeWeights::down_scale`) for
any architecture, refusing the rest as before. `logit_scale` REQUIRED
and multiplied (`MultiplierSupport::TALKIE`). KL 6.43e-14 and 1.47e-14
(no gains; dropping them from the first file lands exactly on the
second's golden); sabotaging the skip add turns four tests red.

`nanbeige` closed on `frink-models/src/layer_loops.rs`. `nanbeige.cpp:
6-12` read `num_loops` / `skip_loop_final_norm`; `:19-31` set
`n_layer_all = n_phys * n_loops` and replicate the per-layer arrays;
`:69-73` alias `layers[i + j * n_phys] = layers[i]`; `:167-175` norm the
residual with `output_norm` after every pass but the last unless the
flag skips it. One graph of 155 reads either key. The weights are
shared and the KV is not: `Decoder::layers` stays physical,
`ModelConfig::n_layers` is the logical count every KV cache and
per-layer table is sized by, `Decoder::layer_for(l)` /
`physical_index(l)` are the ONE mapping the three host bodies, the
gpt-oss side table and the residency plan go through,
`LayerShapes::replicated` is the copy `:24-26` makes of the arrays, and
the loop norm sits at the end of BOTH FFN bodies. The fused Metal
launches refuse a looped model (one `l` for weights and KV). KL
3.06e-13, 4.05e-13 (`skip_loop_final_norm`), 8.70e-13 (`num_loops = 1`,
plain Llama); sabotaging the mapping turns four tests red.

`mimo2` closed on `frink-models/src/kv_head_dims.rs`. `conversion/
mimo.py:154` writes `attention.value_length` from `v_head_dim` apart
from the `attention.key_length` the base converter writes from
`head_dim` (`192` / `128` on MiMo-V2-Flash), and `mimo2.cpp:47-48,
132-140,152-154` size and view K and V separately with `wo` at
`n_embd_head_v * n_head` (`:52`). Fourteen converters write
`value_length`, three apart from `key_length`, two of those MLA;
eighty-nine graphs assert the two equal. So the seam admits the pair
for one architecture by table and keeps refusing it, naming the
assert, for everyone else. `ModelConfig::v_head_dim` is an
`Option<usize>`, `Some` only when the widths differ -- a second `usize`
beside `head_dim` was tried first and the first test that mutated
`head_dim` left V behind -- and `v_head_dim()` the one accessor. Every
consumer took it: `KvCache` / `PagedKvStore` (`new_split`), the three
contiguous single-query kernels collapsed onto ONE
`causal_gqa_attention_row` (they were one loop varied by a window
bound and a sink term), the paged kernel reads it off the store, the
batched prefill kernel's PV tile takes its own offset and stride,
`check_gqa_projection_widths`, `qkv_fused::FusedQkvRows`, both batched
host bodies, the KV budget. Refusing: every fused Metal launch
(`metal_can_serve_model`), the CUDA resident hook, the slot file, the
KV block file (one `head_dim` in each header). The row's second half,
`attention.value_scale` (`:14-17,180-183`, `0.707` on every export),
is `frink-models/src/attn_value_scale.rs`: one reader of 155, applied
after `wo` in the one attention tail. KL 5.42e-15 (the converter's
fused `attn_qkv`, K rows at 12, V rows at 8), 5.42e-15 (split
spelling; libllama byte-identical for the two), 3.49e-15 (no value
scale); each fixture carries the per-layer `head_count_kv` array, the
window array with `rope.freq_base_swa`, sinks, sigmoid routing with
`exp_probs_b`, partial NEOX RoPE and MoE on every layer, as a real
export does. Two findings. `mimo2.cpp:227` passes the SIGMOID literal
into `build_moe_ffn`, so the key is never read: parsing every
`build_moe_ffn` call in all 155 graphs, three pass SIGMOID, twenty-six
SOFTMAX, nineteen the hparam (`GATING_LITERAL_ARCHITECTURES`). And the
bisection that found the last 2e-3 of KL found frink honouring
`expert_weights_scale` for EVERY architecture where llama.cpp reads
the key in twenty per-architecture loaders and nowhere else -- the
fixture declares `2.5`, `mimo2.cpp` never reads it, libllama's golden
is unscaled. `EXPERT_WEIGHTS_SCALE_READERS` / `EXPERT_WEIGHTS_NORM_
READERS` are the readers, measured; no real export of a non-reader
writes either key.

`bitnet` closed on `frink-models/src/sub_norms.rs`. `bitnet.cpp:24,36`
require `attn_sub_norm` `{n_embd}` and `ffn_sub_norm` `{n_ff}`, two
RMSNorms INSIDE the sublayers where the decoder's four sites are all
outside them: `:101-106` norm the attention output BEFORE `wo` (the
other side of that matmul from Gemma's `post_attention_norm`) and
`:127-141` call `build_ffn` with a NULL down, norm `silu(gate) * up`,
and apply `ffn_down` by hand. `grep -l` for either tensor over all 155
graphs is `bitnet.cpp`, so `ModelConfig::block_sub_norms` is a `bool`
with two readers: the loader REQUIRES the pair on it (and refuses the
FFN one on a routed layer, where `build_moe_ffn` has no such site), and
`metal_can_serve_model` refuses every fused launch on it; the
exhaustive destructure in `metal_attn_view` refuses the layer as well.
The arithmetic landed where the tails already were -- `attn_out_to_
residual_rows`, the one attention tail, and `frink_moe::
run_expert_sub_normed`, which shares its gate/up half with `run_expert`
(a new `expert_activated`) and cannot reach the fused on-device SwiGLU
because that kernel runs `down` itself; `dense_ffn_batch` applies it
per row and skips its fused batch kernel for the same reason. KL
1.88e-14, norm weights drawn AWAY from one so skipping either, applying
either with unit weights, or reading the attention one as the
post-norm each diverges by orders of magnitude, measured. Building it
found the per-tensor `.scale` companions: `bitnet.cpp:27-43` create
them optionally, `build_lora_mm` multiplies each projection's output
by them, and since `llama-model.cpp:1355-1440` a GENERIC pass creates
`.scale` / `.input_scale` beside every architecture's projections (the
NVFP4 converter writes them; the current BitNet converter folds them
in). libllama's logits for a fixture with seven `2.0` scales differ
from the unscaled file's, measured, so `frink-models/src/
weight_scales.rs` refuses either suffix by name BEFORE the
unread-tensor gate, which `FRINK_ALLOW_UNKNOWN_TENSORS=1` could have
talked past into every projection at the wrong magnitude. A real
BitNet-b1.58-2B-4T still needs `TQ1_0` / `TQ2_0` (or `i2_s`) kernels,
which `frink-gguf` inspects and refuses at execution; the row is
evidenced on F32 and runs a Q8_0 / F16 re-export.

`smallthinker` closed on `frink-models/src/router_input.rs`. Every
`build_moe_ffn(` call in all 155 graphs was parsed for its `probs_in`
argument before a line was written: fifty-nine sites, four pass one.
`smallthinker.cpp:111` computes the router logits from `inpL`, the
residual as it ENTERS the layer, before `attn_norm` and before
attention; `grovemoe.cpp:133` routes on the normed FFN input (the
default) and precomputes only to share the logits between two expert
banks; `gemma4.cpp:289-294` and `nemotron-h.cpp:210-232` route on
something else again on their own engines. So `RouterInput` is two
variants and one table row; `Decoder::router_operand` is the ONE
constructor, called where each host body applies `attn_norm`, and it
carries LOGITS rather than the operand so the post-attention residual
-- the same `Vec`, mutated in place -- cannot reach the router by
mistake; `gpu_router_matches_host_routing`, the predicate every GPU
router path already shares, answers false for the row. Its verdict's
"one match arm", the `LLM_FFN_RELU` experts, was NOT one: `GluAct::
Reglu` existed and had served `arcee` by ALIASING gate to up, so
`ffn_is_ungated` answered "no gate on disk" for it, and a SmallThinker
loaded through that variant would have dropped its REAL gate tensors
and computed `relu(up) * up`. `FfnActivation::ReluSqr` (ungated) and
`::Reglu` (gated, `smallthinker` alone -- `t5.cpp`'s two hits are a
NULL-gate `build_ffn` on the encoder-decoder engine) are two variants,
and a test pins that `ffn_is_ungated` and `layer_ffn_acts` agree for
every one. The third thing, `smallthinker.cpp:4-8` reading
`attention.sliding_window`, testing it for `> 0` and then assigning
4096 over it, is a third answer (`Pin`) on the one table
`swa_disabled_by_arch` is derived from; the fixture declares 3 and
libllama's logits for it and for the same file declaring 4096 are
byte-identical, measured. KL 1.13e-14, 1.27e-14 (no window, the
converter's softmax arm), 6.56e-15 (a keyed `sliding_window_pattern =
2` beside the literal NoPE step of 4). Building it collapsed the THREE
batched FFN tails onto one `Decoder::ffn_block_batch`, because the
seam needed an eighth fact in all of them.

`mistral3` closed on `frink-models/src/attn_temperature.rs`. `grep -ln
'attn_temp\|temperature_scale\|build_inp_attn_scale' src/models/*.cpp`
over all 155 graphs is `mistral3.cpp`, `llama4.cpp` and `deepseek2.cpp`
(three false hits recorded in the module: `grok.cpp:23` reads
`temperature_length` and applies it nowhere, `dflash` / `deepseek4`
name a hyper-connection TENSOR, `plamo3.cpp:140` is a local). All three
multiply Q by the same `[n_tokens]` input `llama-graph.cpp:163-167`
fills with `log(floor((pos + offset) / floor_scale) + 1) * scale + 1`,
AFTER RoPE and BEFORE `build_attn` with `kq_scale` untouched; what
differs is where the constants come from: `mistral3.cpp:5,14-17` reads
`attention.temperature_scale` and floors on `hparams.n_ctx_orig_yarn`,
`deepseek2.cpp:46-47` reads the same scale with
`attention.temperature_length` as the floor, `llama4.cpp:15-17` seeds
0.1 / 8192 / 1.0 from literals and gates the multiply on its no-RoPE
layers (`:175` is an `else if`). `AttnTemperature` is the three
constants, `scale_at(pos)` is the formula in llama.cpp's precision
(single up to the floor, double from the log), `ModelConfig::
attn_temperature` the one accessor, ONE helper on the CPU row body and
both batched host bodies taking the row's position as a function -- so
`pos`, `start_pos + b` and `positions[b]` are three callers of one loop
-- and `metal_can_serve_model` refuses the fused launches, none of
which has a per-token Q scale. The floor is what a reader gets wrong:
`llama-model.cpp:1164-1165` seeds `n_ctx_orig_yarn` from
`context_length` BEFORE the YaRN key overrides it, so a Ministral with
no `original_context_length` floors on its context length; a fixture
of exactly that shape measures it (libllama byte-identical to the file
with the key). KL 9.16e-15 with a floor of 2 that steps TWICE inside
the six-token prompt, 5.14e-15 plain. The MLA engine (`deepseek2` /
`mistral4`, i.e. Mistral-Large-3) REFUSES a nonzero scale by name now
where it dropped both keys, because it had no golden to check an
implementation against (it has `plm`'s and `deepseek2`'s since
2026-09-12). Two corrections: the verdict's "leading-dense
+ MoE + shared expert" was wrong twice over (`mistral3.cpp:64-84` is
dense OR MoE on every layer, and the `_shexp` tensors are created under
an `n_ff_shexp` its hparams never set and read by no graph line; a
`mistral3` file is a `llama` file with three keys); and `mistral3.cpp:9`
reads `rope.scaling.yarn_log_multiplier`, whose job is to adjust YaRN's
MAGNITUDE term -- which frink applied for NO architecture.
`llama-context.cpp:196-231` multiplies `rope.scaling.attn_factor` by
`get_mscale(factor, 1) / get_mscale(factor, log_mul)` on top of ggml's
`rope_yarn` term, which it cancels; frink's `rope_attn_factor` carried
the key alone, so every YaRN checkpoint on the generic path had
attention logits low by `(1 + 0.1 ln factor)^2`, 1.30x at factor 4.
`frink-models/src/yarn_magnitude.rs` folds it into the field the CPU
helper and the Metal `mscale` uniform already read; two fixtures at
factor 4 evidence both arms, KL 9.14e-15 and 3.45e-15, the second
matching libllama's own `yarn_attn_factor = 1.0648` log line. Only
`mistral3` reads the multiplier on the generic path (measured), so it
is dead metadata everywhere else, as upstream.

`apertus` and `step35` closed on `frink-models/src/act_layers.rs`.
`apertus.cpp:6-9` reads `xielu.alpha_n` / `.alpha_p` / `.beta` / `.eps`
(no architecture prefix, `llama-arch.cpp:370-373`) as REQUIRED
`n_layer`-long arrays or one scalar broadcast, and `:132-138` hands
layer `il`'s four to `ggml_xielu` over its `ffn_up` output, ungated,
with the softplus folded at graph build; `grep -l ggml_xielu` over the
155 graphs is that one file. `step35.cpp:28-29` reads
`swiglu_clamp_exp` / `_shexp` as OPTIONAL arrays and llama.cpp's
GENERIC `build_moe_ffn` (`llama-graph.cpp:2146-2164`) and `build_ffn`
(`:1751-1768`) apply layer `il`'s entry above `1e-6` as `min(silu(gate),
l) * clamp(up, -l, l)`, the routed experts from one array and the
shared experts AND the leading dense layers from the other, because
`build_ffn` is both; three graphs read the keys and two are on their
own engine. `FfnActivation::Xielu` and `::SwigluClamped` CARRY their
tables so kind and parameters cannot disagree, `frink_moe::GluAct`
gained the two bodies -- which cost it `Eq`, and turned `gate_fn() ->
fn(f32) -> f32` into `combine(gate, up)`, because with the gate aliased
to up `xielu(gate) * up` is the wrong function by a factor of the input
-- and `ModelConfig::layer_ffn_acts(il)` answers a `routed` / `dense`
pair at every FFN body; the model-wide `GluAct::from(ffn_activation)`
no longer exists, because it cannot be written for a variant that
needs the layer, and `model_ffn_act()` is `None` for both BY TYPE, so
every fused Metal launch refuses them through the predicate it already
shared. `step35`'s other thing, the half-width rotary on its full
layers (`:9` halves `n_rot_full` AFTER `llama-model.cpp:1222` seeded
`n_rot_swa`, no key), landed on the seam that had refused it from the
other direction: `ModelConfig::rope_dim_swa` is the two-valued width
`n_rot(il)` already was upstream, `layer_rope(il)` hands out
`LayerRopeParams { theta, freq_factors, rot_dim }` so no rotation site
takes the pair without the width, and the fused Metal launches (one
`rot_dim` uniform) are fenced off a model whose widths differ. That
lifted Laguna-XS.2's `rope.dimension_count_swa` refusal by name with
it: the fixture that had evidenced the refusal matches libllama, KL
7.29e-14. Real Step-3.5-Flash also carries a llama3 `rope_freqs.weight`
that `step35.cpp:247` passes to the full layers ONLY (measured, the one
generic-path graph that does); the loader takes the tensor's first
`n_rot_full/2` bands for the full layers and divides by nothing on the
sliding ones for that architecture, and refuses two widths with
divisors for any other. KL 4.91e-14 (apertus, arrays) and 5.06e-14
(the scalar spelling llama.cpp broadcasts; the goldens differ);
1.59e-13 (step35, both arrays with a zero beside each nonzero entry),
2.59e-13 (neither key, plain SwiGLU; the goldens differ), 1.59e-13
(a NextN block inside `block_count`, byte-identical to the trunk's
golden upstream). Building them corrected one sentence of `apertus`'s
verdict: `apertus.cpp:50,52` CREATE `attn_q_norm.bias` /
`attn_k_norm.bias` and `:93,96` pass `NULL` as the bias, so they are
never read; a fixture that carries them measures libllama's logits
byte-identical with and without, and `frink-models/src/
unread_tensors.rs` records the slot so `assert_every_tensor_consumed`
can tell "ignored as llama.cpp ignores it" from "missing".

`mellum` closed on `frink-models/src/swa_layers.rs`, and the
EXAONE-4 32B / EXAONE-MoE / Olmo-3 over-refusal lifted with it. llama.cpp
reads `attention.sliding_window_pattern` with `get_key_or_arr`, which
is THREE behaviours: the scalar overload with `required = false`
returns false on an array and the seeded period stands
(`llama-model-loader.cpp:502-507`; `exaone4.cpp:8`, `exaone-moe.cpp:7`,
`olmo2.cpp:10`, fifteen graphs -- measured), the array overload takes
the array at `n_layer()` length and BROADCASTS a scalar as a bool
(`:474-478`; `mimo2.cpp:12`, `step35.cpp:26`, `gemma4.cpp:5`,
`dflash.cpp:69`), and `mellum.cpp:12-17` / `cohere2moe.cpp:32-36` try
the first then the second. `SwaLayers` is one enum (`All`, `Period`,
`PerLayer`) replacing the two fields (`swa_pattern`, `swa_dense_first`)
behind the ONE accessor every backend already asked,
`ModelConfig::layer_sliding_window(il)`; the fused Metal stacks ask per
layer, so this was the sixth thing looked for in them and the first not
found. Two fixtures carry the EXAONE array (`conversion/exaone.py:84`
writes it for every 32B and MoE export), one agreeing with the seeded
period and one INVERTED, and libllama's logits are byte-identical for
both and for the base -- that is the measurement that ignoring is
upstream's answer -- KL 1.43e-14. `mellum`'s fixture array disagrees
with the seed on two layers and libllama honours it (`is_swa =
1, 1, 0, 1`), KL 1.02e-14; its window-with-YaRN half, which every real
Mellum2 declares, stays refused by name in `swa_geometry`.

`frink-models/src/mtp_blocks.rs` landed beside it, because `mimo2`,
`step35` and the same EXAONE report named the NextN blocks too.
`llama-model.cpp:1092` reads `block_count` into `n_layer_all`,
`llama-hparams.cpp:280-282` defines `n_layer()` as `n_layer_all -
n_layer_nextn`, `llama-graph.cpp:1433` builds every graph over
`n_layer()`, and each tensor loader creates the trailing blocks
`TENSOR_SKIP` (`exaone-moe.cpp:52-57`, `mimo2.cpp:51-52`). Only the
SEVENTEEN graphs that read `nextn_predict_layers` subtract (`grep -l`
over all 155, `NEXTN_READERS`); a nonzero key on any other stays
refused, where upstream would run every block and then fail on the
unread `nextn.*` tensors. `ModelConfig::n_layers` is the trunk and
`n_mtp_blocks` the rest, the skipped tensors are marked deliberately
unread so `assert_every_tensor_consumed` can tell "skipped as llama.cpp
does" from "missing", and two orderings were copied rather than tidied:
`exaone4.cpp:4` tests `n_layer() == 64` BEFORE `:18` reads the key, so
a 64-trunk EXAONE-4 with a block appended gets NO window there and gets
none here (pinned), and the per-layer shape arrays are read at
`block_count` length because `llama-model.cpp:1148-1156` run before
`load_arch_hparams`. The defect it found is in the dedicated engines,
which the task said to check: the GLM (`glm4moe`, `glm-dsa`, `glm4`),
MLA (`deepseek2`) and hybrid (`qwen3next`, `qwen35`, `qwen35moe`)
loaders took `block_count` verbatim while every one of those graphs
subtracts upstream and their converters append the block inside
`block_count` (`glm.py:99`, `deepseek.py:457`), so a real GLM-4.5 or
DeepSeek-V3 export would have run its MTP block as one more decoder
layer with the `nextn.*` tensors silently unread. All four dedicated
loaders take the trunk from `trunk_layers` now, and the MLA loader has
a trunk-only fixture that fails without it. K-EXAONE's real shape --
one block, the array at trunk length -- has a fixture, KL 1.09e-14.
`mimo2` still refused for one more day, on what no seam touched:
MiMo-V2-Flash is `head_dim: 192, v_head_dim: 128`, a V width that
differs from K's, which every KV cache and attention kernel here took
as one number until `kv_head_dims` (above). `step35` led with its clamp arrays and its half-width
rotary on the full layers, and closed on both the same day (above).

`afmoe` and `laguna` are ONE seam, `frink-models/src/attn_gate.rs`.
llama.cpp's `LLM_TENSOR_ATTN_GATE` is created by six of the 155 graphs
(measured); three of them (`qwen3next`, `qwen35`, `qwen35moe`) keep the
gated delta-net's `z` projection under that name, a different op on a
different engine, which is why the seam is keyed by architecture and
not by tensor presence. In the three that gate their softmax attention
the gate is projected from the SAME normed input Q/K/V read and
multiplied into the attention output after the softmax-weighted V sum
and before `wo` -- identical -- and what differs is the activation
(`afmoe.cpp:183`, `step35.cpp:272` sigmoid; `laguna.cpp:246` SOFTPLUS),
the width (`afmoe.cpp:73` per channel, `step35.cpp:96` per head,
`laguna.cpp:110-124` either, read off the stored tensor with an abort
for anything else) and whether the tensor may be absent (`step35.cpp:96`
`TENSOR_NOT_REQUIRED`). So the type has two axes, `GateAct` and
`GateWidth`, a per-architecture table pins the activation and the
ADMISSIBLE widths, and the loader reads the width off the tensor and
refuses one the table does not admit. The durable parts: the gate is
applied in ONE function for the row body and the two batched host
bodies, which collapsed the three hand-written `o_proj` / `o_bias` /
`post_attn_norm` tails onto it, so it was added to one place rather
than three; and every fused Metal launch takes its view of a layer's
attention weights from ONE exhaustive destructure of `AttnWeights`
(`Decoder::metal_attn_view`, no `..`), which answers `None` for a gate
or for sinks, so a field added to that struct does not compile until
the Metal side says whether the kernels serve it -- the fifth and sixth
things found written into those stacks unconditionally. KL 7.03e-13
(afmoe), 1.51e-13 (laguna M.1 shape, per element), 9.57e-14 (laguna
XS.2 shape, per head, `head_count` as an array, a window). Building
them found two defects: `afmoe.cpp:120` scales its embeddings by
`sqrt(n_embd)` from arithmetic, the only non-Gemma graph that does
(measured), so the Gemma family match in the loader is a table now;
and the shared-expert inference probed `blk.0` for a `_shexp` tensor
when `expert_shared_count` was absent, which is 0 for every
leading-dense model -- `laguna.cpp:20` assigns the count before reading
a key its converter never writes, so a real Laguna export would have
loaded with its REQUIRED shared-expert tensors unread on every MoE
layer. One Laguna thing stays refused by name from a fixture libllama
runs: a window together with a RoPE scaling -- the Olmo-3 rule, one
table now for `olmo2`, `mellum` and `laguna` where it had been `arch ==
"olmo2"`. The other, a `rope.dimension_count_swa` differing from
`rope.dimension_count` (`frink-models/src/swa_geometry.rs`), was
refused for one day and is served since `step35` closed on the same
two-valued width; the two `_swa` head-width keys stay refused there.
Real Laguna-M.1 has neither; real Laguna-XS.2 has both. `mimo2`'s sinks moved off the gpt-oss NAME onto
the TENSOR (`AttnWeights::sinks`; four graphs pass it into the one
`build_attn_mha`) without closing the row, because every real MiMo-V2
export carries three MTP blocks inside `block_count` and a per-layer
window ARRAY; both became seams (below), and the row closed on its
split K/V head width the day after (above).

`deci` and `openelm` are ONE seam, `frink-models/src/layer_shapes.rs`.
llama.cpp reads `head_count`, `head_count_kv` and `feed_forward_length`
as scalar-or-array for EVERY architecture (`llama-model.cpp:1149-1158`)
and hands most graphs layer 0 through `LLAMA_LOAD_LOCALS`; frink
carried all three as scalars that every host body read once above its
layer loop. Before a line was written, all 155 `src/models/*.cpp` were
scanned for `n_head(i)`, `n_head_kv(i)`, `n_ff(i)`, `n_embd_k_gqa(i)`,
`n_embd_v_gqa(i)`, `n_rot(i)` and the `_arr` fields, in both the tensor
loader and the graph: 22 files read one somewhere, 17 honour one in
both places, and `layer_shapes::PER_LAYER_SHAPE_ARCHS` records each
with what it still needs -- `deci`, `openelm`, `plamo3` served;
`laguna`, `mimo2`, `step35` narrowed to their other blocker (a
`wqkv_gate`, attention sinks, per-layer clamp arrays); `nanbeige`
copying the arrays to loop its layers; `gemma4` on its own engine; and
seven hybrid recurrent rows where `n_head_kv(i) == 0` means "recurrent".
The scan corrected two readings on the way: `n_rot(il)` is NOT an array
upstream (`llama-hparams.cpp:85-91` is `is_swa(il) ? n_rot_swa :
n_rot_full`), so `step35`'s "per-layer rotary width" is a two-valued
field and a smaller seam; and `granite.cpp:204` reads `n_head(il)` in
its graph while sizing tensors from layer 0, so a heterogeneous Granite
file fails in llama.cpp's own loader. The durable parts: `ModelConfig::
layer_shape(il)` is the ONE accessor and `AttnShape::{Gqa, Linear,
Absent}` is an enum, so deci's two attention-less layer kinds cannot be
spelled as a zero count that a `0..n_heads` loop silently accepts;
`ModelConfig::new_kv_caches` replaced some ninety hand-written
`KvCache::new(config.n_kv_heads, ..)` sites and `KvCache::push` asserts
the width, so a cache built from the scalar panics on a narrower layer's
first token; `metal_can_serve_model` (the predicate `residual_scale` and
`clamp_kqv` already shared) refuses every fused Metal launch for a
non-uniform model, because each takes one `n_heads` and one KV geometry;
the CUDA resident KV and the slot-file writer refuse on the same fact.
Two things the fixtures found in llama.cpp itself: an FFN-free layer
WITH attention has its attention output DISCARDED (`deci.cpp:147-149`
`continue`s before the residual add at `:150-153`; scaling that layer's
attention weights by 3 leaves libllama's logits byte-identical,
measured), which frink refuses by name rather than pins; and if such a
layer is LAST, libllama aborts (`GGML_ASSERT(buffer)`,
ggml-backend.cpp:194), so a real export whose final block is a no-op
cannot run there at all. KL 1.44e-13 (Nemotron shape, one layer of
each kind), 7.29e-13 (DeciLM-7B shape, `head_count_kv` alone varying),
1.28e-13 (openelm, three layers sharing no width).

`arcee` is the ungated ReLU-squared FFN, `down(relu(up(x))^2)`
(`arcee.cpp:39-40,123-128`), and it is spelled without a fourth
`ExpertWeights` shape: `FfnActivation::ReluSqr` maps to
`GluAct::Reglu` (`relu(gate) * up`) and the loader ALIASES the gate to
the up matrix, so every gated path computes `relu(up)^2` with no branch
and the two dense hot paths skip the aliased matmul. What it removed
matters more than what it added: SIX launch sites derived the fused
Metal kernels' `gelu: bool` as `!is_swiglu()`, which reads "not SwiGLU,
therefore GELU", and a third activation would have run as GELU on all
of them; `GluAct::fused_kernel_gelu_flag` is `None` for it and every
site refuses on `None`. KL 2.27e-14. `plm` did NOT close with it, and
that is the correction: the shared verdict constant had been written
from `arcee.cpp`, and `diff arcee.cpp plm.cpp` is 150 lines of
DeepSeek-2 MLA attention (`plm.cpp:16-19,32-36,84-166`) that the
generic decoder does not have. Five graphs pass `LLM_FFN_RELU_SQR`
(measured); `capability::uses_relu_sqr` lists them so the next one to
close finds its FFN already named.

`olmo` is the exception that says what the rule is really made of. It
closed ALONE, and before writing a line of code the question "what else
shares this cause" was answered by MEASUREMENT rather than by hope:
every `build_norm` call in all 155 of llama.cpp's `src/models/*.cpp`
graphs was scanned for a null weight argument, and all three hits are
`olmo.cpp`. So there was no second row to take, and knowing that in
advance is worth as much as a shared cause would have been -- the six
rows the search was aimed at (`openelm`, `bitnet`, `arcee`, `mellum`,
`nanbeige`, `deci`) are now checked-and-recorded rather than
still-plausible. What IS shared is the LayerNorm *function* with a
learned weight: `dbrx` plus the `nemotron` / `orion` / `stablelm` /
`codeshell` / `jais2` / `starcoder` / `starcoder2` / `phimoe` bias
group. At the time none of them was one variant away, because each
refused for more than the norm, so that variant was deliberately NOT
written. The next day `dbrx` became the caller: its other two blockers
were one implementation each, so `NormOp::LayerNorm` (weight, no bias)
landed WITH a row that uses it. The bias form waited until 2026-09-12,
when the group was read row by row instead of as a group: `orion` and
`nemotron` need NOTHING else -- Orion-14B is a Llama with the biased
norm, Nemotron-4 the same norm on the ReLU-squared FFN `arcee` had
already served -- so `NormOp::LayerNormBias` landed with two callers
(`capability::BIASED_LAYER_NORM`, `tests/biased_layer_norm_graphs.rs`,
KL 2.3e-11 and 5.1e-13), `NormFunction::resolve` asks the file for each
PART the function has so the biased form cannot be built with the bias
forgotten, and the six the group still holds each say what else they
need (`attn_output.bias` / `ffn_up.bias` / `ffn_down.bias` for
`starcoder2`, `codeshell`, `jais2`; those plus learned positions for
`starcoder`; a parallel residual for `stablelm`; an `output.bias` and
LongRoPE for `phimoe`). Nemotron's OPTIONAL projection biases were
refused as unread from a fixture whose libllama logits differ by 8.07
from the plain file's -- for one PR.

`stablelm` closed the same day on that norm and nothing new, and what
it took was reading `stablelm.cpp` for what it decides by TENSOR
PRESENCE: three shapes behind one architecture string and no key
among them. A layer WITH `ffn_norm` is the sequential layer
(StableLM-2-1.6B, StableLM-3B-4E1T) and runs, KL 3.4e-13
(`tests/stablelm_graphs.rs`). A layer WITHOUT it is the PARALLEL
residual (`:135-137`, `cur = inpSA`: the FFN reads the normed input
attention read and the layer sums three terms), and
`frink-models/src/parallel_residual.rs` refuses it by name from a
fixture libllama runs (its logits move by 8.85). A layer with
`attn_q_norm` applies a per-head LAYERNORM with a DISTINCT weight per
head (`:34-35,84-97`, `{n_embd_head_k, n_head}`, `LLM_NORM`), which
the loader's length rule would have read as one RMS over the whole
projection and the fused Metal attention infers the same way from the
same length -- two wrongs that agree -- so
`frink-models/src/qk_layer_norm.rs` refuses it by name too (8.73).
StableLM-2-12B has both. `use_parallel_residual`, which every export
writes, is read by NOTHING in the graph: libllama's logits with the
key `true` are byte-identical to the file with it `false`, and a
fixture pins that frink ignores it the same way. Both refusals carry
their reach: the parallel residual is EIGHT of 155 graphs in two
spellings (one shared norm: `stablelm`, `phi2`, `falcon`-7B,
`command-r`, `cohere2`, `cohere2moe`, `plamo`; two norms: `gptneox` under the
key, `falcon`-40B under `attn_norm_2`), and the per-head QK LayerNorm
is three (`stablelm`, `command-r` at 64 layers, `chameleon`), each
recorded in its table with the line that decides it, so the seam that
serves either is sized from the table and not from one graph.

The parallel residual was served the same day, and the table did the
sizing: `gptneox` (Pythia, GPT-NeoX-20B) and `plamo` (PLaMo-13B)
closed on it, ONE PER ARM, and the `stablelm` parallel fixture matches
where it had been refused (`tests/parallel_residual_graphs.rs`; the
GELU rows at the f16-table line, `plamo` KL 1.6e-13). The seam is
smaller than the refusal made it sound, and reading the bodies is
what showed that: the sequential bodies compute `h = x + attn` and
then `h + ffn(normed2)`, which IS the three-term sum, so the only
difference is what `normed2` is -- a norm of the LAYER INPUT, which
attention has already been added on top of by the time the FFN body
runs. So it is captured before attention at the point
`crate::router_input` already captured the router's operand, and the
two travel as ONE value from ONE constructor
(`decoder::ffn_block::BranchInputs`, `Decoder::branch_inputs`), so a
body cannot take one and forget the other; `MoeWeights::parallel` is
the per-layer fact (the `stablelm` row is per layer), and a shared-norm
layer's pre-FFN slot is `NormOp::None` because there is no tensor;
`ModelConfig::parallel_residual` is the model-level fact
`metal_can_serve_model` refuses on, because every fused launch bakes
`ffn_norm` over the post-attention residual into its kernel. The
gptneox fixture pair is the sabotage that matters: the same weights
under the key `true` and `false` are 3.73 apart in libllama, and both
match, so a seam that ignored the field would be caught by the golden
and not only by a test that flips it. `gptneox` also drew on three
seams from the two days before -- the biased LayerNorm, the fused
`attn_qkv` bias, the required projection biases with the ungated GELU
-- which is what "one row per PR" buys: the next row finds its other
blockers already named. `command-r` (Command-R 35B, Aya-23) is the
proof of that one PR later: what it needed on top of the residual was
`dbrx`'s weighted LayerNorm without a bias and the `logit_scale`
multiply `grok` / `talkie` already had, in an OPTIONAL form
(`command-r.cpp:4` reads it `required = false`, `:137` skips it at
zero; `LogitScaleUse::AsIsOptional`, with a fixture that omits the key
and matches libllama's unscaled logits, ratio 0.0625 to the scaled
file's). KL 1.0e-15 (`tests/command_r_graphs.rs`). Command-R+ (64
layers) carries the per-head LayerNorm QK norm llama.cpp REQUIRES at
that depth (`:28-31`) and is refused by name from a 64-layer fixture
libllama runs. `falcon` (Falcon-7B / 40B / 180B) followed on BOTH
arms of the seam, and reading `falcon.cpp:79-85,124` beside
`gptneox.cpp:149` is what the table's `second_norm` column records:
every Falcon layer is parallel, and the OPTIONAL `attn_norm_2` that
40B carries does not switch the residual on, it picks the ARM -- and
it norms the layer input for ATTENTION while `attn_norm` keeps feeding
the FFN, so the two tensor names are crossed relative to gptneox's.
`norm_sites::ATTN_NORM_2_FEEDS_ATTENTION` crosses the two pre-norm
slots on exactly the layers that carry it (`NormSites::for_layer`; one
graph of 155 on the generic path, measured), and the 7B fixture proved
the first table row wrong before any code ran: it had said "parallel
WHEN `attn_norm_2` is present", and a 7B file refused to load its
missing `ffn_norm`. KL 3.8e-8 and 1.9e-7 at the f16 GELU-table line
(`tests/falcon_graphs.rs`); swapping the two slots back diverges by
more than 1. `phi2` (Phi-2, Phi-1.5) followed on 2026-09-14 on the ONE
slot its reason had named: `output.bias` on the LM head
(`phi2.cpp:22,136`, REQUIRED; `proj_bias::OUTPUT_BIAS_CREATORS` is
three graphs of 155, `phimoe` required and `qwen2` optional). It is
`Decoder::output_bias`, added in `decoder::lm_head::Logits` -- the one
place the head's post-projection transforms already ran, before the
multiplier and the cap because it belongs to the matmul -- and
`FoldedLmHead::permit` refuses a head that has one, because unlike the
cap and the multiplier a bias is not monotone across the vocabulary
and a folded argmax would be a different token. KL 2.9e-7 at the f16
GELU-table line, the fused and split QKV spellings on one golden
(`tests/phi2_graphs.rs`). `cohere2` (Command-R7B, Command-A) followed
the same day on a CENSUS CORRECTION rather than new code: its refusal
had named "a rotation on the sliding layers only, the inverse of the
per-layer RoPE gate", and it is not the inverse, it is the gate --
`cohere2.cpp:91` is `if (is_swa)` around `ggml_rope_ext`, the
`exaone-moe` rule `rope_layers` had served since 2026-09-11. The
module's census of SIX had been made by grepping for the word
`use_rope`; grepping for `is_swa` beside `ggml_rope_ext` finds eight,
`cohere2` and `cohere2moe` (whose `|| il < n_layer_dense_lead` is a
variant the enum does not have yet, recorded). One arm, one
`WEIGHTED_LAYER_NORM` row, one `MultiplierSupport` row for its
REQUIRED `logit_scale`, and one refusal for the window key llama.cpp
REQUIRES (`swa_geometry::window_required`; a fixture without it is
refused by libllama with `key not found`, measured). KL 1.0e-14
(`tests/cohere2_graphs.rs`), 8.9e-14 with the scalar pattern key at 2.
`cohere2moe` (the 49-layer 30B-A3B) closed on 2026-09-14 on exactly
what that reason had named, plus one thing it had not:
`RopeLayers::SlidingOrLeadingDense` (`cohere2moe.cpp:192`, the dense
prefix rotates although it does not slide), the `0.5` on `moe_out +
shexp` (`:248-260`, `parallel_dense_ffn::SHARED_EXPERT_SUM_SCALE`,
recorded on the SAME `parallel_sum_scale` field Grok-2's sum scale
fills, so the FFN bodies apply one scale at one site whichever names
the dense branch came from), and the norm FUNCTION decided by the
FILE (`:4-11,166`: `LLM_NORM` unless `layer_norm_rms_epsilon` is
present and nonzero; `norm::NORM_BY_RMS_EPS_KEY`, one graph of 155,
`ModelConfig::norm_function` the one field every site reads). The
thing the reason had not named: `:23` reads `nextn_predict_layers`
BEFORE `:35` reads the window array, so `n_layer()` there is the
TRUNK and the array is one entry per trunk layer, where `mimo2.cpp:12`
/ `step35.cpp:26` read it before the MTP count and take
`block_count` entries -- `swa_layers::ARRAY_AT_TRUNK_LENGTH`, found
by the MTP fixture refusing to load. KL 1.7e-14 (LayerNorm), 1.3e-14
(RMS), the MTP file byte-identical to the trunk's golden in libllama
and here, 6.4e-15 (softmax, `norm_w = true`); dropping the `0.5`,
rotating the prefix as `cohere2`'s rule would not, rotating the full
layer, or renormalising each turns the test red. No
parallel-residual row is refused any more.

`phimoe` (Phi-3.5-MoE) closed the same day, and it is the bias group's
last row but `starcoder`, with a correction to its own refusal: the
"LayerNorm biases" the reason named are RMSNorm biases. `phimoe.cpp`
has no graph of its own (`models.h:632`, `using graph =
llama_model_phi3::graph`), and `phi3.cpp:99-102,137-139,174-177` pass
each norm's bias to `LLM_NORM_RMS`; `phi3` never creates one, so its
files take the plain RMS and `phimoe`'s take `NormOp::RmsBias`,
`capability::BIASED_RMS_NORM`, one graph of 155 on the generic path
(measured: `chameleon` passes NULL there, `deepseek32` / `glm-dsa` are
other engines). Its two projection biases were slots already
(`attn_output.bias`, `output.bias`), so the row is one `NormFunction`
variant. A second correction came free: `phimoe.cpp:3-10` read no
window key, so the `attention.sliding_window` every export writes is
dead metadata as `phi3`'s (libllama `n_swa = 0` on a file declaring
8, measured) -- and a test had asserted that `phimoe` HONOURS its
window, which nothing had ever checked. KL 1.9e-11 (LongRoPE's long
pair in use, attn factor 1.0955) and 1.9e-12 (plain), at the `orion`
line for the same reason (`tests/phimoe_graphs.rs`). The loader's
norm-function census listed two of five function lists; it lists all
five now.

`gpt2` and `starcoder` closed together on `frink-models/src/
position_embd.rs`, and they are ONE graph read side by side (`diff
gpt2.cpp starcoder.cpp` is a size table and `head_count_kv 1`): the
`gptneox` sequential layer with a learned `position_embd.weight`
`{n_embd, n_ctx_train}` gathered at `inp_pos` and ADDED to the token
embedding before layer 0 (`gpt2.cpp:19,74-77`), and no `ggml_rope`
anywhere. Reach measured over the 155: three generic-path graphs
create the tensor (`mpt`'s OPTIONAL beside its ALiBi, recorded), four
encoders. The seam is two facts: the table added at the ONE embedding
site (`Decoder::embed_token` takes `pos` now, so no caller can embed
without saying where), and `rope_layers::RopeLayers::Never`, a rule
that rotates nothing -- deliberately NOT `SlidingOnly` with no window,
which would start rotating the day such a file declared one. The
`LLAMA_NO_ROPE` test that had pinned `gpt2` off the generic path pins
it on it now under exactly that rule, and the four ALiBi rows stay
where they were. Every fused Metal launch is fenced off a model with a
table, because the GPU embedding gather has no add. KL 1.9e-7 each at
the f16 GELU-table line (`tests/position_embd_graphs.rs`); dropping
the table or rotating every layer -- what the generic path used to do
to GPT-2 -- diverges by more than 1. The "LayerNorm-with-bias group"
of `tests/attn_bias.rs` is EMPTY: nine rows, nine closures, each by the
dropped bias being implemented rather than the row being edited.

ALiBi closed the rest of the `LLAMA_ROPE_TYPE_NONE` group the same
day, FIVE rows on one seam, and the count was measured before the
seam was built: `grep -n f_max_alibi_bias src/models/*.cpp` over the
155 is seven files, five on the generic path, in THREE spellings --
the literal `8.0f` (`bloom.cpp:18`, `refact.cpp:12`), the literal at
40 layers only (`baichuan.cpp:11-14`, the 13B; the 7B rotates and was
audited already), and `attention.max_alibi_bias` read optional
(`mpt.cpp:6`, `jais.cpp:5`). `frink-core/src/alibi.rs` is llama.cpp's
slope formula (`ggml-cpu/ops.cpp:5489-5508`: `n_head_log2`, `m0`,
`m1`, the odd powers for the tail of a non-power-of-two head count),
and the three host attention kernels -- row, paged, batched prefill --
take the slopes as ONE additive term, `slope_h * (p_key - p_query)`
after the scale and the softcap, where `ggml_soft_max_ext` adds
`slope * mask`; a frink-core test pins the three against a naive
reference and each other. `frink-models/src/alibi.rs` is the table,
and `rope_layers::RopeLayers::Never` is DERIVED from it for these rows,
so the bias and the absence of rotation cannot disagree about
Baichuan's layer count -- the thing the old refusal had said "no key
could see". Every fused Metal launch and the CUDA resident attention
refuse a model with a bias. Each row then had one more thing, and each
was a table entry: `bloom` norms its EMBEDDINGS (`token_embd_norm`,
`norm_sites::EMBEDDING_NORM_ARCHITECTURES`, the one decoder of 155
that does), `jais` passes `kq_scale = 1/d` (`jais.cpp:83`, the one
graph with a literal other than `1/sqrt(d)`;
`capability::attention_scale_override`), `mpt` clamps
(`CLAMPED_QKV_ARCHITECTURES`, where it had been "deliberately absent"
as a refused row) and carries an optional `position_embd` the table
from the PR before serves. KL 6.4e-13 (refact), 8.3e-8 / 3.6e-7 /
1.6e-7 (bloom, mpt, mpt with a position table: GELU rows at the table
line), 7.4e-13 (jais), 1.1e-12 (Baichuan-13B at 40 layers)
(`tests/alibi_graphs.rs`). The first run of the six goldens found the
two things the table had not: `mpt`'s clamp was not applied (4.8 off)
and `jais` was scaled by `1/sqrt(d)` (1.36 off), which is what a
golden is for.

`minimax-m2` (MiniMax-M2, the 230B MoE) closed the same day with NO
code, and it is the cheapest lesson in this file: its refusal had said
"UNAUDITED, not unimplemented -- plain GQA, whole-vector QK norm,
partial NEOX RoPE, a sigmoid MoE with `exp_probs_b`, all of which the
generic path has; what is missing is a fixture" for a week, while
`scripts/make_minimax_fixture.py` and `tests/fixtures/minimax_m2_tiny.
gguf` sat in the tree evidencing that very claim. Running the fixture
through libllama took a minute and matched at KL 3.4e-15 on the first
try (`tests/minimax_m2_graphs.rs`). A verdict that names its own
closing evidence and does not go and get it is a refusal that could
have been a row.

`lfm2` (LFM2-350M / 700M / 1.2B / 2.6B) and `lfm2moe` (LFM2-8B-A1B)
closed the same day as the FIRST hybrid rows, and the lesson is where
they closed: not on the hybrid
engine that had refused it for a year but on the generic path.
`lfm2.cpp:192-208` is the generic layer -- `attn_norm`, a block, the
residual add, `ffn_norm`, the FFN -- with a short convolution where
attention would be on the layers whose `head_count_kv` is 0 (`:9-11`),
so it is a THIRD `AttnShape` beside deci's two (`layer_shapes::
AttnShape::ShortConv`, `frink-models/src/shortconv.rs`) rather than
a second engine. Reach measured first: `grep -l shortconv` over the
155 graphs is `lfm2.cpp` and `lfm2moe.cpp`, ONE graph (`models.h:
1899`); the Mamba-2 hybrids share only the rule "zero KV heads means
recurrent", and `layer_shapes::ZeroKvLayer` is that rule as a table --
the same two counts had been deci's `Linear` for every architecture,
so a Jamba file would have loaded its Mamba layers as `wo`-only
attention and failed on a missing tensor. The state is the layer's KV
HISTORY: one `n_embd` row of `b * x` per token with no V
(`AttnShape::cache_geometry`), read back as the last `l_cache` rows,
where llama.cpp keeps the `l_cache - 1` it needs (`n_embd_r()`).
Keeping the history is what lets the layer truncate, page, snapshot
and share a prefix through the same code every attention layer uses;
the paged arm reads the window through the block table and a test
straddles a block boundary with it. `token_embd_norm.weight` is its
OUTPUT norm (`llama-arch.cpp:384`, "fix for wrong tensor name") and
`bloom`'s EMBEDDING norm -- the `attn_output_norm` case again, one row
in `norm_sites`. KL 3.2e-12 on the split, fused-QKV and
separate-`output` fixtures (`tests/lfm2_graphs.rs`); reversing the
taps or dropping the state turns every golden red; `lfm2moe` is the
same graph with `leading_dense_block_count` dense layers and a sigmoid
MoE with `exp_probs_b` REQUIRED on the rest, KL 6.0e-13, its
`expert_weights_scale` dead metadata as `mimo2`'s. A window stays
refused by name: `lfm2.cpp:24-29` windows the attention layers ALONE,
which `swa_layers` cannot spell, and no export writes the key.

`pangu-embedded` (openPangu-Embedded-1B / 7B, Huawei) closed the same
day from the DEFERRED column, which no row had left before, and the
lesson is the cheapest kind: the row had been classified from its
NAME. "Embedded" means edge devices; `PanguEmbeddedForCausalLM` is a
decoder with an `lm_head`, `conversion/pangu.py` is a `TextModel`, and
frink had it in the catalog as "embedding variant; deferred" AND in
`embedding_model::NOT_YET` as "a decoder embedding path", two copies
of one wrong reading. `pangu-embed.cpp` is `llama.cpp`'s graph with a
REQUIRED `attn_output.bias` (`:37`) and NEOX RoPE: one `proj_bias`
row, one profile line, KL 1.5e-13 on three fixtures
(`tests/pangu_embedded_graphs.rs`). Reading the converter's base
class takes a minute; the deferred column has thirty-one rows left and
every one of them deserves that minute.

`granitehybrid` (Granite 4.0: H-Micro, H-Tiny, H-Small) closed the
same day as the FIRST MAMBA-2 row, and it is the second recurrent seam
in one day, which is what shows where the two differ. `granite-hybrid.
cpp:128-142` is Granite's layer with `build_mamba2_layer`
(`mamba-base.cpp:149-288`) where attention would be on the zero-KV
layers, so the block is a fourth `AttnShape` (`Mamba2`,
`frink-models/src/mamba2.rs`, `frink-core/src/mamba2.rs` for ggml's
conv and scan steps) on the same table row `lfm2` used. What is NOT
the same is the state: LFM2's is a window of its inputs and rode as
the layer's KV history, which every cache consumer already knew how
to truncate, page and fork; a Mamba state is a REDUCTION over the
whole prefix, and `frink_core::recurrent_state::RecurrentState`
beside the cache says so in the one place it matters --
`KvCache::truncate` REFUSES a middle position (`can_truncate_to`), the
prefix cache does not store such a cache, `--model-draft` refuses
such a model, and the block pushes EMPTY rows so `positions()` still
counts, because `serving/batch/row.rs` reads layer 0's cache as "how
far this row has got" and Granite-4.0's layer 0 is a Mamba layer.
llama.cpp's `llama_memory_recurrent::seq_rm` refuses `p0 > 0` for the
same reason. Two measurements: `ssm_conv1d.bias` is
`TENSOR_NOT_REQUIRED` at `granite-hybrid.cpp:63` and `ggml_add`ed
unconditionally at `mamba-base.cpp:222`, so libllama SEGFAULTS on a
file without it and frink requires it; and the Granite
`rope.scaling.finetuned` refusal, which had said "frink has no way to
express unrotated" for four days while `RopeLayers::Never` had
existed for one, is served (`rope_finetuned::unrotated`), because
`conversion/granite.py:253-256` writes the key FALSE for every
Granite-4.0 hybrid export and no row could match without it. KL
1.9e-13 (NoPE dense), 7.9e-13 (rotated, Bamba's shape), 1.0e-13 (MoE
with the shared expert), `tests/granite_hybrid_graphs.rs`; resetting
the state every token or dropping the `exp` from the decay turns five
tests red. `nemotron-h` (one block per layer), `falcon-h1` (attention
and Mamba-2 in PARALLEL), `jamba` and `plamo2` (Mamba-1) each named
what they still needed in `layer_shapes::ZeroKvLayer`; all three have closed since.

`nemotron_h` (Nemotron-H 8B / 47B / 56B, Nemotron-3 Nano dense) closed
the same day on that seam, and what it added is two table rows and no
arithmetic: `nemotron-h.cpp:143-158` runs every layer as ONE block --
Mamba-2 where both arrays are zero (`:9-11`), attention where only the
FFN width is, the ungated ReLU-squared FFN otherwise -- under ONE
`attn_norm` with ONE residual add. On the generic layer that is "a
block with `ffn_dim 0`", which `LayerShapes::resolve` had REFUSED for
every architecture because deci DISCARDS such a block's output
(`deci.cpp:147-149`) and Nemotron-H adds it (`:157`): one reading per
graph, `BLOCK_WITHOUT_FFN_KEEPS_ITS_OUTPUT`. And "an FFN with no
block", whose pre-norm is `attn_norm` because the graph has one norm
per layer (`norm_sites::ONE_NORM_PER_LAYER`; exact, because a layer
reads one slot or the other and never both). `ZeroKvLayer::
Mamba2UnlessFfn` is the rule that reads the second array. Its
attention never calls `ggml_rope_ext` (`:181-193`), so the NEOX group
entry is a filler and `rope_layers` answers `Never`. KL 2.0e-13 /
1.4e-12 / 7.5e-14 (`tests/nemotron_h_graphs.rs`); pointing the FFN
slot at `ffn_norm` turns every test red. `nemotron_h_moe`
(Nemotron-3 Nano 30B-A3B) closed in the next PR on what the loader
already had: `load_dense_expert` aliased an ungated FFN's gate to
`up`, and the routed and shared experts (`:82-89,209-227`, a null
gate into `build_moe_ffn`) take the same alias, which
`GluAct::ReluSqr` never reads; the sigmoid literal (`:218`) and the
two `expert_weights_*` keys (`:18-19`, which Nano's converter writes
as `norm_topk_prob true` / `routed_scaling_factor 2.5`) are rows in
the loader's reader tables; a layer with `ffn_dim 0` takes the dense
arm whatever the model's MoE says, so the block-only layers load no
experts. `moe_latent_size` (Nemotron-3 Super, `:206-208`) is refused
by name in `unsupported_feature_keys`. KL 3.6e-13.

`falcon-h1` (Falcon-H1 0.5B to 34B) closed next, and it is the
Mamba-2 block in its THIRD position: `falcon-h1.cpp:137-161` runs
attention AND the block on every layer, in parallel on the same
`attn_norm` output, summed before the one residual add. So the layer
is `AttnShape::Gqa` with `AttnWeights::mamba2` set
(`mamba2::PARALLEL_WITH_ATTENTION`, `ModelConfig::parallel_ssm`), its
cache holds the attention rows AND the block's `RecurrentState`,
attention counts the positions, and `Decoder::parallel_ssm_rows` /
`add_parallel_ssm` are ONE pair the row body and both batched bodies
call -- the block runs before the KV push so the two can never
disagree about which token the state saw. Two measurements:
`attn_output.bias` is created at `:76` and `:154` passes NULL for it,
so it is `unread_tensors`' second row; and `ffn_norm` is the
two-argument `LLM_TN` spelling (`:80`, `blk.N.ffn_norm` with no
`.weight`), which libllama REQUIRES -- the `.weight` spelling fails
`check_tensor_dims` -- and which `norm_sites` had accepted since
`plamo3`. KL 1.3e-13 / 6.2e-13 (no `ssm_norm`) / 3.2e-13
(`tests/falcon_h1_graphs.rs`); running the block on a fresh state
every token turns six tests red. Every `build_mamba2_layer` caller
of the 155 graphs is served; `jamba`, `plamo2` and `mamba` are
`build_mamba_layer` (Mamba-1) and said so.

`plamo2` (PLaMo-2 1B / 2B / 8B) closed on 2026-09-18 on its own SSM
spelling, `frink-models/src/plamo2_ssm.rs`: Mamba-1's dt / B / C path
(one projection of the conv output, REQUIRED weighted norms) in the
order B, C, dt, feeding Mamba-2's per-head scan (dt, A, D per head,
`head_dim = d_inner / n_heads`, `Decay::PerHead`), z and x interleaved
PER HEAD in `ssm_in`'s output, no conv bias, `dt_dim = max(64,
n_embd / 16)` a literal of the graph (`plamo2.cpp:38,285`), and the
three norm tensors spelled with NO `.weight` (the two-argument `tn`,
`:77-79`, and the converter's exact-match entries). Its attention is
the generic path plus one QK-norm style: `QkNormStyle::
PerHeadDistinct`, RMS per head with a DISTINCT weight row per head
(`{qk_dim, n_head}`, `:92-93,163,166`), the same length as a
whole-vector weight and so decided by architecture (`capability::
PER_HEAD_DISTINCT_QK_NORM`, one RMS row of the five graphs that
create that shape); the fused Metal / CUDA launches refuse the style.
And `plamo2.cpp:19` reads only `n_head_kv(il)`, while the converter
writes BOTH head arrays as 0 on an SSM layer, so `(0, 0)` means the
block on this architecture (`AttnShape::Plamo2Ssm`) and deci's
attention-free layer everywhere else. KL 3.0e-12 on both head-count
spellings (`tests/plamo2_graphs.rs`); swapping B and C turns it red.
The finding is llama.cpp's: `llama-model.cpp:1189-1201` seed `n_rot`
from layer 0's head count and set it to 0 when that is 0, so libllama
runs every current PLaMo-2 export UNROTATED (`print_info: n_rot = 0`,
measured; 1.3 in the logits), and `rope.dimension_count` cannot
restore it because that read sits inside the same branch. frink
rotates, as `modeling_plamo.py` does and as `:239,245` were written
to; the golden is the same weights with `head_count` scalar, an
equally valid file libllama rotates, and the unrotated logits are
kept beside it as a number. The real checkpoint also needed the
`plamo2` TOKENIZER (`tokenizer/plamo2.rs`, a port of
`llm_tokenizer_plamo2`, `llama-vocab.cpp:1351-1616`: a suffix-table
segmenter searched backwards, `<0xXX>` byte fallback), which
`frink parity` holds byte-identical to libllama on 1,264 tokens of
the corpus, CJK and emoji included. On the one public GGUF
(`mmnga/plamo-2-1b-gguf`, an older conversion with a scalar
`head_count`) libllama itself returns all-NaN logits; frink answers
"Paris. Paris is the capital of France."

`jamba`, `mamba` and `mamba2` closed next on that block and on one
rule. `frink-models/src/mamba1.rs` is `build_mamba_layer`
(`mamba-base.cpp:4-148`) once: the same conv step and the same scan
kernel as Mamba-2's with `n_head = d_inner`, `head_dim = 1` and a
per-STATE decay (`frink_core::mamba2::Decay::PerState`, the kernel's
`src3->ne[0] != 1` arm, one `exp` per state element), dt / B / C from
one projection of the conv output, the RMS norms on them when the
three weights exist (Jamba, REQUIRED) or `ssm.dt_b_c_rms` sets them
weightless (FalconMamba), dt projected up with its bias.
`ssm_block::SsmBlock` is the one value the decoder holds for either
generation, so a third is one variant. The rule is `layer_shapes::
PURE_RECURRENT`: a pure Mamba converter writes `head_count 0`,
`head_count_kv` absent and `feed_forward_length 0`, uniform zeros
that `LayerShapes::resolve` had read as a zero-head GQA model; on the
two named architectures every layer is the block and nothing else,
`head_dim` is 0 because nothing reads one, and the FFN-free block
keeps its output (`mamba.cpp:88`). Jamba is the hybrid: Mamba-1 where
the KV array is 0, attention with NO RoPE elsewhere (`jamba.cpp:98`),
and the FFN dense or MoE PER LAYER by whether the file has a router
(`:89-101,152`; `moe_interleave::DENSE_LAYER_BY_ROUTER_ABSENCE`, one
loader of 155 that reads the file for it), softmax with `norm_w =
false`. KL 7.3e-12 / 2.3e-12 / 1.8e-12 / 3.6e-13
(`tests/mamba_graphs.rs`), the Mamba-1 rows at the `orion` tolerance
class because `d_inner` one-element heads each take their own `exp`;
swapping B and C turns five tests red. Every Mamba graph in llama.cpp
is served but `plamo2`'s own spelling, which closed on 2026-09-18 (below).

`qwen35` (Qwen3.5 dense, 0.8B to 27B) closed next, and it is the row
the "hybrid engine" had been a scaffold FOR since 2026-09-01: 852
lines of `gdn.rs` ported from a Python reference, 915 of
`hybrid_gguf_loader.rs`, a stub that refused everything, and not one
libllama golden between them. The block went on the seam the Mamba
rows had just built: `frink-core/src/gdn.rs` is the autoregressive
delta rule as `delta-net-base.cpp:289-365` computes it (decay, the
state's prediction, the beta-scaled error, the rank-one update, the
read-out with the scaled query), `frink-models/src/gdn.rs` is
`qwen35.cpp:236-317` around it, `AttnShape::Gdn` is the site, and the
state rides on the cache as every other recurrent block's. Two things
the port had wrong that only a golden shows: Qwen3.5's V heads read
K heads TILED (`h % n_k`, `llama-model.cpp:524-526`; the converter
reorders V heads for `ggml_repeat`), where the reference and the port
grouped them (`h / ratio`, Qwen3-Next's order), and the l2 norms take
the model's RMS epsilon, not a literal. The attention layers gate
through a double-width `wq` with the gate interleaved per head
(`:191-199`); it is split AFTER the projection
(`attn_gate::split_interleaved_q_gate`) so a quantized `wq` stays one
matrix on its quantized path, and the sigmoid multiplies in the one
attention tail. Which layers are recurrent comes from two keys, not
the head counts (`gdn::recurrent_layers`: the array over `n_layer_all`
when present, else the interval); `LayerShapes::resolve` takes that
mask. Partial IMROPE over `rope.dimension_sections` is NEOX band for
band on text positions (`mrope::MROPE_READERS`), pinned by the golden
with the sections in the file. KL 4.1e-13 on the first run
(`tests/qwen35_graphs.rs`); grouping the heads or dropping the
sigmoid each turns five tests red. The scaffold is deleted.
`qwen35moe` (Qwen3.5-35B-A3B, 122B-A10B, 397B-A17B) followed in the
next PR with NO code: `qwen35moe.cpp:496-538` is `qwen2moe`'s FFN --
softmax, `norm_w = true`, the shared expert scaled by its own sigmoid
gate -- under Qwen3.5's layers, KL 2.9e-11 at the `orion` tolerance.
`qwen3next` (Qwen3-Next-80B-A3B) followed on the two tables its
refusal had named: `gdn::GROUPED_HEAD_ARCHITECTURES` (its V heads
read K heads `h / ratio`, `qwen3next.cpp:521-539`, the map the
deleted port had assumed for everyone) and `gdn::BetaAlpha::Fused`
(beta and alpha in one `ssm_ba` projection laid out per K group,
`:422-436`), plain NEOX RoPE, KL 8.7e-12; swapping beta and alpha
or tiling the heads each turns its test red. Every gated-delta-net
graph in llama.cpp is served.

`llama4` (Llama 4 Scout / Maverick) closed on 2026-09-14, and it is
the row that says "read the graph AND the helper it calls":
`llama4.cpp` has four things on top of Llama, and the one that cost
the most was not in it. `frink-models/src/chunked_swa.rs` is the
new seam: `llama4.cpp:13-14` set `LLAMA_SWA_TYPE_CHUNKED` at a
literal 8192 (the file's value is never read, and a declared ZERO is
the branch libllama ABORTS on, `llama-graph.cpp:159`, refused by
name), `llama-hparams.h:419-425` mask every key before the query's
own chunk, so a query at `p` sees `p % 8192 + 1` positions where a
sliding layer sees a constant. `ModelConfig::layer_window_for_query`
is that number for the row body, `BatchWindow` sends the batched
prefill to a per-query arm when a batch straddles a boundary and to
the blocked kernel when it sits inside the first chunk, eviction
keeps the last 8192 rows, and `metal_can_serve_model` refuses the
model. The other three each landed as a per-layer gate on a seam
that existed: the temperature 0.1 / 8192 / 1.0 from literals
(`attn_temperature::LITERAL_ATTN_TEMPERATURE`) on the layers that do
NOT rotate (`AttnTemperature::unrotated_layers_only`); a weightless
per-head RMS on Q and K AFTER RoPE on the layers that do, for every
expert count but 128 (`llama4.cpp:43,182-188`; `weightless_qk_norm`,
a `bool`, one reachable graph of 155); and the interleave step,
honoured because `llama4.cpp:64` is the ONE tensor loader of 155
that branches on it (`moe_interleave::
INTERLEAVE_STEP_HONOURED_BY_LOADER`; ERNIE's does not, which is why
that arm is a refusal). The fourth thing is `llama-graph.cpp:1947`:
`weight_before_ffn = arch == LLM_ARCH_LLAMA4`, the sigmoid weight
multiplied into the expert's INPUT before `mul_mat_id` where every
other graph multiplies the output -- SwiGLU is not homogeneous, the
fixture's logits move by 0.86 between the two, and
`frink-models/src/routed_weight_site.rs` scales the row per slot at
the two host sites that gather a routed input. Building it found
`route_top_k_sigmoid` renormalising whatever `norm_topk_prob` said:
every sigmoid row before this one declared the key true, so the
ignored argument had never changed an answer; `llama4.cpp:228`
passes `false`. KL 1.1e-12 on the 16- and 128-expert shapes and at
the last of 8200 positions across the chunk boundary on both bodies
(`REF_N_CTX` on the reference tool); each seam switched off turns
the test red; a zero expert count is refused as `llama4.cpp:49-51`
refuse it. The engine stub that had refused the row is deleted.

`frink-models/src/proj_bias.rs` closed `starcoder2`, `codeshell` and
`jais2` the same day, and it is the reach measurement that says what
the seam is: `grep -l 'ATTN_OUT, "bias"'` over the 155 graphs is 33
files and `FFN_UP, "bias"` 27, MOST of them OPTIONAL -- `llama.cpp`'s
own graph, `granite`, `deci`, `mistral3`, `minicpm`, `nemotron` create
the biases `TENSOR_NOT_REQUIRED` and `build_ffn` / `build_attn` add
them when present -- so a `llama` file WITH biases was one frink
refused as carrying unread tensors while llama.cpp ran it. The
arithmetic is the same in every graph (`up_b` / `gate_b` before the
activation, `down_b` after `down`, `wo_b` after `wo` and after `wo_s`),
so it lives in two slots -- `AttnWeights::o_bias` in the ONE attention
tail, where gpt-oss's bias moved from its side table, and
`MoeWeights::dense_bias` (`frink_moe::DenseBias`,
`run_expert_biased`, whose gate/up projections come from the same
`gate_up_projections` the unbiased body uses) in the row and batched
dense bodies -- and the per-architecture fact is which graphs CREATE
the tensors, two tables with a `Required` / `Optional` column, so a
bias on an architecture whose graph never creates it stays unread and
refused as llama.cpp refuses the file. The two GELU rows needed
`FfnActivation::GeluUngated` (`LLM_FFN_GELU` under `LLM_FFN_SEQ`,
eleven graphs, measured), aliased as `ReluSqr` is. Every fused Metal
dense launch and the attention view fence on the two fields. Four
goldens (`tests/proj_bias_graphs.rs`): `jais2` 6.6e-13, a `llama` with
all four biases 2.7e-13 (the one file that exercises `ffn_gate.bias`),
`starcoder2` and `codeshell` at 4.7e-7 / 4.7e-6 KL with 3e-3 / 8e-3
max deltas that are ENTIRELY llama.cpp's f16 GELU table -- with
frink's GELU made to emulate the table (input and output rounded to
f16) both agree to 2e-13 / 1e-12 -- and `nemotron_biases`, refused the
PR before, at 2.1e-13.

`olmo2` and `exaone4` closed TOGETHER, because they are ONE residual
topology: no `attn_norm` and no `ffn_norm` tensor, both sublayers
reading the raw residual, each branch's output normed before its
residual add. Reading `olmo2.cpp:45-52,92,160-182` beside
`exaone4.cpp:60-67,118,152-169` gives the same graph line for line, so
they got one implementation (`frink-models/src/norm.rs`) and a
fixture each. One sub-case stays refused by name rather than swept in --
an `olmo2` with both a window and a RoPE scaling; EXAONE-4 32B, whose
full-attention layers get no RoPE at all, was the other and closed on
2026-09-11 (below). `olmo` (OLMo-1) is a
THIRD shape, pre-norm with a non-parametric LayerNorm, and closed on
2026-09-10 as a third variant of the same enum; the type is
`frink-models/src/norm.rs`'s `NormOp` now rather than `PreNorm`,
because `Decoder::final_norm` is one too -- OLMo-1's final norm has no
weights either, and the fused Metal stacks had `Some(&self.final_norm)`
written into them unconditionally. Its `attention.clamp_kqv` stayed a
REFUSAL for one day: `llama-graph.cpp:1611-1652` clamps Q, K and V by
it, `conversion/olmo.py:23-25` really writes it for OLMo-7B-Twin-2T and
OLMo-1.7-7B, and a second fixture measures that llama.cpp's own logits
move when it is present. It was refused rather than implemented because
the three host bodies each applied the QKV bias in their own loop and a
clamp added to some of them would have been the dominant bug shape
again; when `dbrx` needed the same clamp as a REQUIRED key, the three
loops collapsed onto one helper (`decoder/qkv_bias.rs`) and the clamp
became a line in it, the fused Metal launches fenced off through the
same predicate as `residual_scale`. That second fixture now matches
libllama on all three paths instead of evidencing a refusal.

`granite`, `granitemoe` and the `granite-moe` alias closed together for
the same kind of reason: they differ in the FFN, not in the four scalar
multipliers llama.cpp applies to them, and `granite-moe.cpp` has no
graph of its own at all (`models.h:1583-1591` is
`using graph = llama_model_granite::graph`).
`frink-models/src/scalar_multipliers.rs` implements the multipliers
once, parameterised, and `capability::unsupported_scaling_keys` -- which
still refuses those keys everywhere else -- is now DERIVED from that
same table instead of restated beside it. That derivation found a live
gap on the way past: the whole Gemma family was exempted from all four
keys while reading NONE of them, so a hand-written
`gemma3.residual_scale` would have loaded and been ignored. The
expensive half was `residual_scale`: it multiplies both branch outputs
of every layer, so `decoder.rs`'s EIGHTEEN hand-written residual adds
collapsed onto one function taking the scalar as a parameter, and the
four Metal eligibility checks gained one shared predicate rather than
four spellings of it. MiniCPM ran the same llama.cpp graph
object -- `models.h:1594-1601` is
`using graph = llama_model_granite::graph` -- and closed on 2026-09-10
on the defaults hook that was predicted: `minicpm.cpp:5-7` assigns 12.0,
`1.4/sqrt(n_layer)` and `256/n_embd` BEFORE `:12-14` lets the file
override them, so a MiniCPM export declaring nothing is still scaled by
all three and a key-PRESENCE gate has nothing to see. That is why it was
refused by name rather than detected. `MultiplierDefaults` is a FIELD of
the same table, so a default for a key the graph does not apply is not
expressible, and the fixture that evidences it declares NO key at all --
the only fixture shape that can tell the hook from its absence. A second
one declares all three and pins that the file still wins, because a hook
merged the wrong way round would agree with llama.cpp on exactly the
files that prove it exists. Command-R closed on 2026-09-12 once its
parallel residual was served (below).

`grok` closed on 2026-09-11 on that same hook, one day after it landed,
and the verdict had predicted it: `grok.cpp:5-12` seeds SEVEN
hyper-parameters before `:14-27` let the file override them. Two of
them needed a column the table did not have -- `logit_scale` is a
MULTIPLY there (`:211`, the `AsIs` variant the module had named as
deliberately absent) and the attention scale comes from a fifth key,
`attention.output_scale`, applied INSIDE the tanh softcap with
`kq_scale = 1.0f` (`:137`, `llama-graph.cpp:2572-2582`), which is
"pre-scale Q, then softcap" and so the existing `attention_scale` slot
plus the existing softcap, no new attention code. Two more of the seven
(`router_logit_softcapping`, `attention.temperature_length`) are read
by llama.cpp and applied NOWHERE in its graph -- measured, no other
reference under `src/` -- so frink neither applies nor refuses them.
`dbrx` closed the same day on the three blockers its verdict named, and
one of them is why `frink-models/src/norm_sites.rs` exists:
`blk.N.attn_output_norm` is `dbrx`'s PRE-FFN norm (`dbrx.cpp:34,110-113`)
and `grok`'s POST-attention norm (`grok.cpp:62,143-148`), one tensor
name feeding two different sites, decided by architecture. The loader
had restated the norm-slot decision as an `if` chain at every site;
it is one table now, read at all five. Both rows are GeGLU-or-clamp
shaped in their evidence: DBRX KL 3.4e-12; Grok KL 4.7e-10 and 1.6e-10
on the no-key and all-keys fixtures, at the GeGLU tolerance because
llama.cpp's f16 GELU table is the approximate side -- measured by
making frink's GELU emulate the table, at which point both files agree
to 1e-7. Grok-2's parallel dense FFN (`grok.cpp:171-184`) was refused
by name from a fixture that has it until `arctic` closed on the same
seam (`parallel_dense_ffn.rs`, above); the fixture has a golden now.

`exaone-moe`, `smollm3` and EXAONE-4 32B closed together on 2026-09-11
on the PER-LAYER RoPE gate, and the claim that they are one cause was
read before it was assumed: `exaone4.cpp:116` is
`use_rope = is_swa(il) || swa_type == NONE`, `exaone-moe.cpp:136,155-161`
is `is_swa(il)` around the same two `ggml_rope_ext` calls, and
`exaone-moe.cpp:4` pins `swa_type` to STANDARD, which makes the second
disjunct false -- identical, not similar. `smollm3.cpp:5,69` is another
variant of the same enum, `(il + 1) % 4 != 0`. llama.cpp gates rotation
this way in SIX architectures, always from a literal and never from a
GGUF key, and `frink-models/src/rope_layers.rs` is one table for all
six; `smallthinker`, `afmoe` and `llama4` are in it, the first two
closed later on other seams, and `llama4` still refuses for other
things and its verdict says so. The durable part is
the type, not the arms: `ModelConfig::layer_rope` returns
`Option<(base, divisors)>`, so no rotation site can take the pair
without answering whether to rotate -- the CPU head loop, the YaRN
`attn_factor` (an argument to `ggml_rope_ext`, so it goes with it), the
four per-layer Metal launches (which lost a loose base/divisor
parameter pair for one `LayerRope`) and both fused Metal stacks, whose
RoPE dispatch had been written in unconditionally the way OLMo-1's
final norm had. A 64-layer fixture evidences the 32B because
`exaone4.cpp:4` tests equality, and building the three found two more
things: EXAONE-4 1.2B must IGNORE a window its file declares
(`exaone4.cpp:4-14` reaches `set_swa_pattern` only at 64 layers, so a
window key below that is dead metadata -- `capability::
swa_disabled_by_arch` carries it beside `phi3`), and
`nextn_predict_layers` was refused NOWHERE: MTP blocks are inside
`block_count` and llama.cpp skips them, so a real EXAONE-MoE export
with an MTP head would have run its speculative head as two more
decoder layers. It is gated on the value now, because the converter
writes the key as `0` for the sizes that have no head.

Building those fixtures keeps finding defects worth more than the
admissions. The last three UNKNOWN rows -- `mistral`, `mixtral`, `yi` --
closed on 2026-09-10 by turning out NOT TO BE ARCHITECTURES: libllama
refuses all three strings (`unknown model architecture: 'mistral'`,
measured), every real checkpoint of all three declares `llama`, and the
rows had been sitting on the generic path with NEOX RoPE while `llama`
is NORM -- a wrong-pairs rotation that `rope_layout_matches_llama_cpp`
could not see, because a name absent from llama.cpp's table is a
`continue` there. `chatglm`'s arm was predicted to close `qwen` too and
closed it only halfway: the fused `attn_qkv.bias` really is shared, but
`qwen.cpp:33-35` also halves every FFN matrix, which cost no logits and
made `expert_ffn_dim` twice the real width. `plamo3` could never have
loaded a real checkpoint, because it is the only architecture upstream
whose post-norms use the two-argument `LLM_TN` overload and frink asked for the wrong spelling;
a gate refused every file carrying `attention.sliding_window_pattern` as
unimplemented while the feature was already implemented, which made the
loader's own read of that key unreachable; llama.cpp's own
`ernie4_5-moe` tensor loader (`ernie4-5.cpp:49`) has no interleave step
in it while its graph (`ernie4-5-moe.cpp:64`) does, so no interleaved
ERNIE-4.5 MoE checkpoint can be loaded by llama.cpp at all and that arm
had to land as a refusal; and `hunyuan-dense`'s verdict cited a
converter line (`conversion/hunyuan.py:356`) that belongs to
`hunyuan-vl`, not to it; and no Granite converter writes
`{arch}.rope.scaling.finetuned`, so the RoPE-off switch llama.cpp reads
out of that key (`granite.cpp:33-35`) can only be reached by a
hand-written file -- which is why it landed as a refusal with a fixture
that actually carries the key rather than as a gate nobody could fire.
`unaudited_triage` carries the verdict and the
llama.cpp line that decides it. llama.cpp hand-writes 155
per-architecture graphs (140 at the 2026-08-04 pin, 155 at `5b59b83`); `decoder.rs` is 6752 lines and that is why the
counts differ.

**Ternary-Bonsai-2-27B ran on the real checkpoint on 2026-09-18**, the
first non-llama.cpp quantization format here (PrismML's `PTQ1_0`, with
a Hadamard rotation folded into the weights), verified against the
fork's libllama at first-token KL 2.1e-5 on all three paths (`docs/
FEATURES.md`). The day's lesson is the old one three more times: the
PTQ1_0 matvec returned zeros on its first run because
`rows_per_threadgroup` walked a COPY of the matvec kind list, and
`apply_gpu_multi` and `apply_gpu_batch` each matched kinds by hand and
had been quietly running Q5_0 through their fallbacks for two weeks
while the capability table and the kernel registry said otherwise.
Each is derived from the one table now, with a test that holds it.
The speed gap that remains is **11.17 vs 11.46 tok/s decode** and
**43.1 vs 66.8 prefill** on the M2 Pro, measured back to back on a
quiet box at the reference's own `tg32` shape, and decode is at
**97.5%** of it. Prefill's HOST side is finished: the gated delta-net
recurrence runs chunked (the state read once per CHUNK of rows, not
once per row) and then on the device, and the blocked attention the
fused block refuses now runs on the GPU too; a `sample` of a prefill
has nothing host-side above the noise.

Decode went 7.1 to 11.17 by collapsing command buffers, and the
count is the whole story: 192 waits a token, then 69, then 20. A
recurrent layer is ONE submission end to end -- `attn_norm`, the four
projections, the gates, the convolution, the l2 norms, the delta
rule, the gated norm, the rotation, `ssm_out`, the residual, the FFN
norm, the FFN, the residual. Consecutive recurrent layers commit back
to back against one device buffer and wait ONCE. Then the attention
layer stopped having submissions of its own: its TAIL rides in the
next run's command buffer, its PROJECTIONS ride in the previous
one's, and the attention itself reads a device KV mirror
(`KvCache::metal_attn`), so a four-layer group on a hybrid is one
submission and one wait with no host step inside it.

**Decode is FLAT with context now, which it was not.** 11.17 / 11.19
/ 11.16 at 32 / 300 / 600 tokens, against the reference's 11.46 /
11.50. The slope it used to have -- 11.1 at 32 and 10.95 at 300 --
was a host attention whose work grows with `seq_len`, and the shape
of the curve is worth more than the tok/s: a number that is flat
where the reference is flat says the remaining difference is a
constant, not an algorithm.

What is left is that constant, ~3% in the PTQ1_0 matvec's inner loop,
and `docs/plans/gdn-resident-state.md` records what has already been
eliminated against it -- three measured-neutral memory hypotheses, two
concurrency schemes (one slower, one correct and flat, because a
single matvec already dispatches 4352 threadgroups and fills the
part), and the per-stage breakdown (head 0.286 ms, branch 0.251, FFN
0.762 per recurrent layer). It also records that the bandwidth probe
those hypotheses were aimed at reports a third of the truth and has
never once predicted production; it subtracts its own launch cost
now, and is still only good for comparing kernel variants. And it
records the bug that nearly shipped with the device attention: the
first version encoded it on a CONCURRENT encoder whose tail expects
a serial one, and the model generated FLUENT NONSENSE while `frink
parity` stayed MATCH, because parity reads the first token and the
first token is prefill. Nothing in the suite compares a DECODE step
against a reference; a greedy 100-token generation against the same
build with the device path off is what caught it, and that comparison
should be a test.

Two earlier levers are worth as much as the ones that worked: the
device-side Hadamard bought 9% of decode and COST 6% of prefill,
where the host transform is already parallel; and a spin-then-block
wait on the command buffer, aimed at 0.166 ms of wake-up, measured
3.7 tok/s against 7.1 -- polling the status through objc takes the
core the host work needs. Both are recorded where the code is.

`nomic-bert` (nomic-embed-text v1 / v1.5) embeds since 2026-09-19, and
it is the first row closed on the ENCODER rather than the decoder: the
audit that ranked the work put the embedding family above the exotic
new architectures, because `/v1/embeddings` has users and MiniMax-01
has 456B parameters. It shares `bert.cpp`'s graph and differs in two
lines -- NEOX RoPE on Q/K where `bert` adds a position table, and a
gated SiLU FFN where `bert`'s is an ungated GELU --
`bert_gguf_loader::ENCODER_ARCHS` is the table, and the row found
something the existing parity test had attributed to quantization: on
an F32 fixture the two engines agree EXACTLY with the attention
switched off and differ by ~3e-4 with it on, which is not the Q8_0
activation story `tests/bert_llama_cpp_parity.rs` tells. The
bisection is recorded in `tests/nomic_bert_graphs.rs` with what it
eliminated (a uniform softmax still differs; f16 K/V does not explain
it).

**`--ctk q4` grew a Hadamard rotation on 2026-09-19**, and its lesson
is about measurement rather than kernels. The 4-bit KV store rotates K
at append (a deterministic per-head, per-channel sign flip and a
normalized Walsh-Hadamard transform) and puts Q through the same matrix
at read, so `q . k` is unchanged; V is left alone, worth 39% of the
error on K against 12% on V. Two metrics were discarded first: a
free-running greedy generation is chaotic, so one flipped token makes
the rest unrelated and six prompts scored anywhere from 0 to 134
characters for the SAME build; and `frink perplexity` is byte-identical
for `f16`, `q8_0` and `q4` because that path never touches the Metal
store, which is a metric that does not move when the thing under test
changes. The one that works is next-token agreement with an f16 store
over sixty long-context windows: 45/60 for the unrotated wire and
52/60 for the shipped rotated one. The number that matters more is a
NULL result found by accident: an earlier build of the identical
scheme, differing only in the arbitrary per-channel sign pattern,
scored 48/60. Four windows in sixty is therefore the spread of the
METRIC, not a difference between schemes, so a KV comparison here is
only believable when it is wider than that. Rotation against no
rotation (52 against 45) clears it; the scale-granularity question
does not, and stays open rather than acted on.
Two host paths had to learn the rotation with the kernel,
`upload_from_host` and `tokens_host`, because a rotated row handed to a
host kernel whose query is not rotated is a different model and neither
would have failed loudly.

The `--ctk` values are named after the WIRE now (`f16`, `q8_0`, `fp8`,
`q4`), because the previous set carried two names for one 34-byte store
and one name for a store that had no implementation.

Do not read the architecture catalog as a support matrix. `frink
parity` is the oracle: its tokenizer half matches llama.cpp on every
local checkpoint libllama can load, and its logit half MATCHES on
Q8_0/IQ4_NL while DRIFTING on K-quants — for a known reason that is not
a frink bug (`docs/plans/llama-cpp-gap-inventory.md` §10).

Capabilities: `docs/FEATURES.md`. Models & speed ledger: `docs/MODELS.md`,
`benchmarks/RESULTS.md`. Planned: `docs/ROADMAP.md`.

| Doc | Role |
|---|---|
| `docs/FEATURES.md` | capabilities overview |
| `docs/CLI.md` | `frink` flags + `frink chat` |
| `docs/MODELS.md` | what runs / what doesn’t |
| `docs/API.md` | OpenAI compatibility matrix |
| `docs/AGENTS_COOKBOOK.md` | point IDEs at `frink-server` |
| `docs/CONFIG.md` | env vars |
| `benchmarks/RESULTS.md` | tok/s vs llama.cpp (Gap = llama/frink); `frink bench` ledger |
| `benchmarks/README.md` | how `frink bench` / `llama-bench` is measured |
| `docs/ROADMAP.md` | planned work |
| `docs/plans/README.md` | **the plan index and priority order** |
| `docs/plans/north-star.md` | the goal, and how plans are ranked against it |

## Commands

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

cargo build --workspace --features cuda
cargo test -p frink-cuda --features cuda -- --ignored   # needs GPU

cargo build -p frink-cli -p frink-server --features metal
cargo test -p frink-metal --features metal -- --ignored   # needs Metal

# Completion (also: frink run -m …)
./target/debug/frink -m model.gguf -p "Hi" -n 64 --temp 0 --no-cnv
./target/debug/frink -m model.gguf -p "Hi" -n 64 --ngl 99   # Metal

./target/debug/frink presets | archs | caps | inspect <gguf> | inspect-plan <gguf>
./target/debug/frink smoke <preset> | run-kimi <dir>
./target/debug/frink chat --url http://127.0.0.1:8383   # needs frink-server

FRINK_MODEL_PATH=model.gguf FRINK_ADDR=127.0.0.1:8383 ./target/debug/frink-server

# Bench vs llama-bench (no HTTP). Models: benchmarks/suite.json
./target/release/frink bench -m model.gguf -p 512 -n 128 --compare
./target/release/frink bench --suite --fit-host --skip-missing
./target/release/frink bench --suite --id llama32_3b_q4km --backend metal
./target/release/frink bench --render
```

Fixtures and golden values were generated and cross-validated with
independent NumPy references.

Tests are mostly `#[cfg(test)]` next to the code. Integration:
`crates/frink-models/tests/gguf_roundtrip.rs`. Never un-ignore CUDA /
Metal hardware tests without a real GPU.

## How to write code here

**Keep files small, modules narrow, and the binary light.** This is not
style preference, it is the repo's most expensive lesson.

Re-measured 2026-09-10, against the 2026-09-03 numbers:

| File | Was | Now | |
|---|---|---|---|
| `frink-server/src/lib.rs` | 9628 | **9775** | grew |
| `frink-metal/src/gpu.rs` | 8935 | **9018** | grew |
| `frink-metal/src/attn.rs` | 9331 | **8715** | shrank |
| `frink-quant/src/lib.rs` | 8239 | **8242** | flat |
| `frink-models/src/decoder.rs` | 6752 | **6780** | grew |

The per-layer shape wave (2026-09-11) left `decoder.rs` at 6842 and
`loader.rs` at 5005: the seam itself is a 700-line new file
(`layer_shapes.rs`) and the one row-level FFN body it needed
(`decoder/ffn_block.rs`) replaced three copies, but the two batched
attention bodies still had to be edited in place, and each of those
edits is a labelled block around 400 lines nobody has yet split out.
The per-layer activation wave (2026-09-11, later) left `decoder.rs` at
7068 and `loader.rs` at 5250, and moved `frink-moe/src/lib.rs` from
2150 to 2043: `GluAct` left it for `glu_act.rs` BEFORE the fourth and
fifth variants were added, which is the rule working; the two seams
are new files (`act_layers.rs`, `unread_tensors.rs`); and what
`decoder.rs` gained is a `layer_idx` threaded into six FFN bodies plus
a Metal fence test, which is the cost of an accessor the bodies did
not have a layer to ask with. The attention-temperature wave
(2026-09-11, last) left `decoder.rs` at 7118 and `loader.rs` at 5290:
two new files (`attn_temperature.rs`, `yarn_magnitude.rs`), one helper
in `decoder/rope.rs`, and what the big two gained is three one-line
call sites, one predicate clause, one fence test and one resolution
block -- the shape a seam SHOULD leave behind, and still fifty lines
nobody split out first.

**One of five shrank, and the rule still lost on balance.** `attn.rs`
gave up 616 lines only because a change was made to it and the split
came first, which is the rule working as written. `lib.rs` gained 147
adding four sampler steps to three routes, and `decoder.rs` crept up
again. Nothing shrinks on its own.

What did work is not in that table, because the files left it:
`frink-quant/src/repack.rs` was 6446 lines and is now a directory of
ten; `frink-models/src/sampling.rs` went 1569 to 894 with four
submodules beside it; `frink-cuda/src/mul_mm.rs` is 865 with its kinds
in two directories. Each of those splits happened because somebody was
about to add to the file and split it first. That is the whole
mechanism, and it is the only one that has ever worked here.

Those files are why llama.cpp has 155 architectures and frink has 93
proven. Adding a model means editing a 6750-line file, so nobody adds
one. The same decode layer used to be written out about ELEVEN times
across `decoder.rs` and `attn.rs`, which has already lost EIGHT model
features one at a time, each silently:
`attention_scale`, `post_attn_norm`, `post_ffn_norm`, gpt-oss `o_bias`,
`gpt_oss_ffn`, the four the Metal MoE decode stack ignored, and the
four-way drift of the GPU-router eligibility check, where the prefill
sites tested three conditions, the fused decode two and the whole-stack
decode NONE. A copy diverges from its original and nothing notices.

**THIS IS THE DOMINANT BUG SHAPE IN THIS REPO, and 2026-09-01 found a
dozen more instances of it in one day** — not all of them copied code.
The general form is TWO STRUCTURES THAT MUST AGREE ABOUT ONE THING,
WITH NOTHING ENFORCING IT: two spellings of a GGUF key (`unsupported_
feature_keys` gated on one no converter writes, so it never fired); two
copies of a default (the response cache restated every `unwrap_or`
independently of the sampler); four hand-written `SamplingParams`
literals in one file; three tables that had to agree about a Metal
kernel's threadgroup geometry (a correct kernel returned zeros for half
its rows); a wire struct and a sampler that silently disagreed about
which fields exist (SIX parameters accepted and ignored).

2026-09-04/05 added three more, all found by measuring rather than
reading: a benchmark receipt carrying `backend: "cpu"` beside
`backend_active: "Metal"`, in the same file, with nothing comparing the
two, for **13 of 13** published CPU rows; `KINDS` and the host-check
tool's fixed shape list, where adding the K-quants made the tool panic
on the first one and check NOTHING for a day while still exiting green;
and `FRINK_CPU_INT_DOT`, a default that was correct on the
architecture its kernels were written for and cost the other one 4x to
8.8x of decode.

The durable fixes are never the individual patches. They are the places
where disagreement now fails to COMPILE or turns a test red: an
exhaustive destructure with no `..`, one predicate the four call sites
share, a derived table instead of a restated one, and a test asserting
every refused key is one a converter actually writes.

Rules that follow from that:

- **A new file beats a new section.** If a change would push a file past
  roughly 1000 lines, split it first, then make the change.
- **One concept per module.** A module named after a noun that holds
  three unrelated things is three modules.
- **Never copy a code path to vary it.** Parameterise the original. The
  precedent that works: `forward_multi_seq` takes a `MultiSeqKv`
  parameter rather than having a paged twin. The precedent that failed:
  `forward_token_paged` was a copy, and lost five features -- it is now
  collapsed onto one `attn_block` taking a `KvStep`.
- **Dead code is a liability, not an asset.** Delete on sight unless it
  serves a named roadmap theme; if it does, wire it or say where it is
  going. `frink-edge` was 5,400 uncalled lines and is now dissolved.
- **A gate that cannot fire is worse than no gate**, because it reads as
  coverage. Check that a refusal's condition is reachable at all: this
  repo shipped one keyed on a GGUF spelling nothing writes.
- **No new crate for something one crate uses.** `frink-edge` became a
  crate instead of an integration and half of it was never called.

Rust specifics this repo holds to:

- `cargo clippy --workspace --all-targets -- -D warnings` is a gate, not
  advice. Also run it `--release`: `debug_assert!` type-checks its
  argument in release, and a `#[cfg(debug_assertions)]` method called
  from one broke every release build while all of CI stayed green.
- Prefer borrowing to cloning on any path that runs per token. Hoist
  feature probes out of loops: `is_aarch64_feature_detected!` ran 131k
  times in one Mistral-7B projection before it was hoisted.
- `unsafe` needs a `// SAFETY:` comment stating the invariant, and a
  scalar twin it is checked against. Every SIMD arm here has one.
- Return `Result` and name what is missing. A model this engine only
  partly implements must STOP, never compute something else. A refusal
  is coverage, not a defect.
- Tests live in `#[cfg(test)]` beside the code. A test that cannot fail
  is not a test: sabotage it once and confirm it goes red.
- **Confirm the sabotage LANDED.** A mutation that did not apply is
  indistinguishable from a test that holds. One sabotage here passed
  and proved nothing because `cargo fmt` had wrapped the target line
  across three lines, so the patch never matched. Grep the file for the
  mutated text before believing a green run, and if a test survives the
  first sabotage attempt, suspect the sabotage before believing the
  test.

## Measuring, and not fooling yourself

Every performance claim in this repo has to survive these. They are
here because each one was broken in a single week, and each break cost
either a wrong number in `benchmarks/RESULTS.md` or a merged-nothing
PR.

- **Rent a box; do not measure on this laptop.** `suggestd` holds ~97%
  of a core on it indefinitely and respawns hot when killed. CPU and
  CUDA rows come from vast.ai; the M2 Pro is for Metal, where it is the
  only hardware that can run the backend. Pick offers where
  `cpu_cores_effective == cpu_cores`, so no co-tenant shares the CPU.
  Four instances cost $0.49. Destroy them the moment the receipts are
  copied back, and confirm with `vastai show instances`.
- **A load average cannot see one busy core.** `frink bench`'s
  `--max-load 2.0` guard passed for an entire day while one of six
  cores was pegged, because one core does not move a six-core average
  enough to trip it. The guard is necessary and not sufficient: check
  `ps -eo pcpu,comm | sort -rn | head` too, and treat any process above
  ~90% as disqualifying.
- **One instantaneous sample is not a measurement.** "CUDA decode runs
  at 36% GPU utilization" came from a single `nvidia-smi` taken AFTER a
  bench had finished, so it caught an idle moment. Sampled five times
  DURING the run, the real figure was 86-93%. That one number sent a
  day of work at a host-side cost that did not exist and produced a PR
  measured 22% SLOWER. Sample repeatedly, and sample while the thing
  runs.
- **Verify the flag did what it says, from the artifact.** `frink
  bench --n-gpu-layers 0` is documented as forcing CPU and did not:
  the backend is decided once per process and cached, so the flag
  arrived too late. Read `backend_active` in the receipt, not the flag
  you passed. A receipt whose label disagrees with the backend that ran
  is now refused at write time and asserted over the committed set,
  because this was found only after publishing 13 wrong rows.
- **Check that the lever can reach the target before pulling it.** At
  the (wrong) 36% figure, removing EVERY host round-trip was worth at
  most 1/0.36 = 2.8x against a 9x-17x gap. That arithmetic was written
  down before the code was, and reading past it cost the PR. If the
  best case does not close the gap, the diagnosis is incomplete
  whatever else is true.
- **Interleave A/B when comparing two builds**, and report the raw
  sequence. `main, branch, main, branch` catches drift that two
  sequential runs hide.
- **A gap column cannot show a missing kernel.** A CUDA K-quant prefill
  read 4.88 tok/s against llama.cpp's 1586.80 and looked like a
  performance problem; there was no GEMM at all, and the fallback still
  answered correctly. Check coverage before profiling.
- **Check what the kernel is actually limited BY, before optimising
  anything in it.** Convert the measurement into the resource: for a
  decode matvec, `weights_bytes * tok/s` is achieved memory bandwidth,
  and that number is one line of arithmetic. frink reaches **5.3%** of
  an RTX 3060's 360 GB/s where llama.cpp reaches **60.4%**, so decode
  is limited by memory-request concurrency. A port of llama.cpp's
  `dp4a` inner loop was written, verified correct on hardware, and
  measured **under 1% faster**, because four MACs per instruction buys
  nothing in a kernel that is waiting on loads. The source diff between
  two kernels tells you what is different; it does not tell you which
  difference is the limit.

## Working with agents

- **One worktree per agent.** Two agents in the same checkout collided
  here: one detected another's half-finished refactor, backed its own
  work out, and redid it in isolation. Use `isolation: "worktree"`.
- **Do not redo an agent's task while it runs.** A narrower, better
  version of a deletion was in flight while the same deletion was
  attempted by hand; the hand version removed a live hardware test with
  it.
- **Agents do not tag, publish, force-push, rent hardware, or
  benchmark.** They implement and open a PR; verification on real
  hardware is the parent session's, and a kernel merges only after
  `cargo test -p frink-cuda --features cuda -- --ignored` has run on a
  GPU.

## Architecture

```
frink-gguf + frink-quant
        → frink-core (WeightMatrix, RoPE, GQA, KV; optional cuda/metal)
        → frink-moe
        → frink-models (loader, Decoder, Kimi/GLM/DS4 stacks)
        → frink-cli / frink-server

frink-api  (routes + wire DTOs, serde-only) → frink-server + clients
```

**The FreeToken port** (Apache-2.0; see `docs/THIRD_PARTY_NOTICES.md`,
which is a licence obligation and must stay accurate) is a Rust port of
FreeToken's host-side decision logic. It used to be a crate of its own,
`frink-edge`; it now lives in the crates that use it, and the parts
nothing would ever use are deleted.

- **`frink-core`** holds the MoE expert-residency half, beside
  `expert_store`: `expert_cache`, `expert_slots` (the `SlotDevice`
  seam), `expert_pool` (its CUDA implementation), `expert_budget`,
  `qstar`, `bench_profile`, `residency`, `placement`. `expert_store` is
  the SINGLE holder of the expert byte budget -- on unified memory two
  budgets are the same RAM counted twice.
- **`frink-server::policy`** holds the serving half: the two parsers,
  the radix prefix cache, anchor/window slide, scheduler, serving stats,
  maintenance, pool, rebuild, outbox, footprint, effort probing.

Wired today: the parsers, the stop-string withhold rule, the radix
prefix cache over paged KV, effort probing, stats, maintenance, outbox,
footprint, and the scheduler's status reporting. STILL GROUNDWORK, and
this is the gap that matters most: the whole `frink-core` expert
residency stack holds the policy for running a model larger than memory
(`docs/plans/out-of-core-moe.md`) and nothing executes it except a
compile-only CUDA pool whose hardware test is `#[ignore]`d.

Anything in `policy` with an unwired half names the roadmap item that
would close it, at its declaration in `policy/mod.rs`. That
`allow(dead_code)` list is meant to be read as a to-do, not as cover.

Load path: GGUF mmap → keep quantized → fused dequant+dot →
RMSNorm → GQA(+RoPE) → MoE/dense FFN. Serving: `FRINK_MODEL_PATH`
GGUF or Kimi dir; generation on `spawn_blocking`.

Presets `glm_5_2` / `deepseek_v4_pro` / `kimi_k3` are sketches,
not proof of real-checkpoint support. `test_*_fixture` presets match
Python test GGUFs only.

Never add Co-Authored-By, Claude-Session, or Generated with Claude Code lines to commit messages or PR bodies.
