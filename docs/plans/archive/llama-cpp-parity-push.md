---
name: llama.cpp parity push
overview: "MAIN GOAL: fix every performance gap against llama.cpp — gap ≤ 1.0× on every red engine suite row (29 at the start of the push, 25 as published on 2026-08-13), with answer parity, plus honest model/weight coverage (no arch that silently computes the wrong graph). VALIDATION RULE: every improvement is validated by a full `--suite --fit-host --skip-missing` + `--render` run, not just the model it targeted — a change is not landed until the whole ledger is re-measured and no row regressed. Default method: read `.scratch/llama.cpp` and port kernels/glue to Frink Rust/MSL. Re-ranked 2026-08-10 from a four-way code audit: the CPU prefill gap is a scalar activation re-interleave inside the i8mm GEMM (not the arithmetic tier), and the Metal prefill gap is a 12.5%-lane-occupancy attention kernel (not command-buffer batching). Both prior diagnoses were wrong and are corrected below."
todos:
  - id: c1-smollm3-rotates-layers-it-must-not
    content: "SILENTLY WRONG TODAY, on a widely downloaded checkpoint, and the cheapest severe bug in this plan. llama.cpp's `src/models/smollm3.cpp:5` sets `hparams.n_no_rope_layer_step = 4` UNCONDITIONALLY and `:69` computes `use_rope = (il + 1) % 4 != 0`, so NINE of SmolLM3-3B's 36 layers get NO RoPE AT ALL. There is no GGUF key for it: no `LLM_KV` entry exists in llama-arch.cpp and nothing in frink reads one. Frink admits `smollm3` as plain `gqa_norm` (capability.rs:213); the tensor set matches the generic llama set exactly, no scaling keys, no softcap keys, and the RoPE layout `Norm` agrees with the pin (tests/rope_layout.rs:144). So the file loads clean, runs at full speed, and rotates every layer. This is the SAME CLASS as the gpt2/mpt/refact/bloom/jais bug fixed on 2026-08-27, still live, on a more popular model than any of those five. FIX: a per-layer rope skip plus an `n_no_rope_layer_step` entry in the registry. Or refuse `smollm3` today, which is one line and matches this project's own rule."
    status: pending
  - id: c5-one-bpe-regex-for-every-checkpoint
    content: "SILENTLY WRONG on EVERY BPE model frink already claims to run: Llama-3.x, Qwen2/2.5/3, DeepSeek, SmolLM. `tokenizer.ggml.pre` is read in exactly one place (tokenizer.rs:31) and used only to decide BOS prepending (:37). ZERO of llama.cpp's ~30 pre-tokenizer regex variants are implemented; every BPE model gets `gpt2_pretokenize_regex()` (:308-311). Verified divergences: `\"1234567\"` is one run in frink and `123|456|7` in llama.cpp, so EVERY number of 4+ digits gets different ids; `\"DON'T\"` splits `DON|'|T` versus `DON|'T`; `\"hello  world\"` splits `hello|  |world` versus `hello| | world`, so any run of 2+ whitespace differs, which means INDENTED CODE AND BLANK LINES tokenize differently. That last one contradicts the code's own comment at :299-307 claiming only trailing whitespace is affected. The stated blocker (Rust `regex` has no lookaround) is undercut by `fancy-regex` already being a dependency of the same crate and already used for this in kimi_tokenizer.rs:34. INVISIBLE because BPE tests are round-trip only (:1363-1505) with no golden pair, and because `frink parity` hands llama.cpp explicit token ids, excluding the tokenizer from the only cross-engine oracle in the repo. CHEAPEST FIRST MOVE IS NOT THE REGEXES: add `.inp`/`.out` goldens from llama.cpp's CI corpus the way SPM already has, which turns an invisible defect into a failing test."
    status: pending
  - id: c6-unknown-tokenizer-falls-back-to-bytes
    content: "SILENTLY WRONG, and trivial to fix. An unhandled `tokenizer.ggml.model` warns and returns `ServerTokenizer::Byte` (frink-server/src/model.rs:146-151), with the same fallback at four CLI sites (run.rs:685-691, :851, :986, :1122). `ByteTokenizer` maps byte to id (tokenizer.rs:224-227). So any `bert`/wpm, `rwkv` or `none` vocab GENERATES FLUENT GARBAGE rather than refusing. Only verify_engine.rs:107 hard-errors. Three call sites plus four CLI clones, and it converts guaranteed-garbage into a readable refusal. This is the project's stated rule applied to the one place that ignores it."
    status: pending
  - id: c4-linear-rope-scaling-is-ignored
    content: "SILENTLY WRONG, and deliberate, which is why it needs a decision rather than a patch. `yarn_scaling_from_gguf` returns `None` for any scaling type other than `yarn` (loader.rs:733-736) and `rope.scaling.factor` is read nowhere else. A checkpoint declaring `linear` scaling with factor 4.0 loads and ropes at UNSCALED positions where llama.cpp divides them by 4. There is a TEST asserting this behaviour (loader.rs:2374-2389) reasoning that leaving positions alone beats roping them wrong in a new way. Both are wrong output, and under this project's own rule (docs/MODELS.md:104-110) it should REFUSE. Affects the older long-context community rescales (`*-16k`, `*-32k` Llama-2 derivatives). YaRN declared without `rope.scaling.original_context_length` is dropped the same way (:2406-2418) where llama.cpp falls back to `n_ctx_train`. Refusal is one line; implementing linear scaling is also small."
    status: pending
  - id: c2-c3-sliding-window-phase-and-period
    content: "SILENTLY WRONG on three archs, and one transcription error. `default_swa_pattern` (capability.rs:766-778) covers 6 architectures; llama.cpp hardcodes a period for 18. Live misses on the generic path: `mellum` (src/models/mellum.cpp:11, period 4) and `exaone-moe` (:4-8, unconditional SWA with n_swa=128 and period 4) both get EVERY layer windowed, so layers that should be full-attention see a 128-token history. `smallthinker` diverges FOUR ways at once (capability.rs:261): llama.cpp runs its MoE experts with RELU not SiLU (smallthinker.cpp:158) while frink hardcodes Swiglu for every non-Gemma non-Phi arch (loader.rs:481-486); it hardcodes n_swa=4096 overriding the file (:8); it uses `set_swa_pattern(4, dense_first=true)` and frink has NO `dense_first` concept at all (config.rs:346-362 implements only the dense_first=false phase); and it feeds the router the pre-norm residual, not the post-ffn_norm hidden (:111). SEPARATELY a transcription error: `gemma3n` is `Some(6)` at capability.rs:773 but gemma3n.cpp:4 says 5. Inert while gemma3n refuses, still wrong. NOTE the sibling table `swa_rope_base_follows_model` (:793-810) was checked exhaustively and IS complete and correct, so this is a gap in one table, not a systemic failure."
    status: pending
  - id: wpm-and-the-embedding-model-gap
    content: "THE LARGEST RAW CHECKPOINT-COUNT GAP IN THE WHOLE AUDIT, larger than any decoder family, and it is not an architecture problem. Frink handles 3 of llama.cpp's 6 `tokenizer.ggml.model` values: `gpt2`/`gemma4` to BPE, `llama` to SPM, `t5` to Unigram. MISSING: `bert`/wpm (no WordPiece code anywhere), `rwkv` (no trie tokenizer), `none`. Combined with the byte fallback in `c6-unknown-tokenizer-falls-back-to-bytes`, those three generate garbage rather than refusing. On top of that, 12 encoder/embedding architectures (`bert`, `nomic-bert`, `jina-bert-v2/v3`, `modern-bert`, `gemma-embedding`, `llama-embed`) all refuse, and `/v1/embeddings` only pools a DECODER GGUF's hidden states. So every BGE, E5, nomic-embed, jina-embed and GTE, which is what people actually use for embeddings, cannot be served. Also no `/v1/rerank` route at all, and `classifier.output_labels`/`pooling_type` are unread. Medium to large effort, and it unlocks more published files than anything else in this plan."
    status: pending
  - id: two-cheap-quant-correctness-fixes
    content: "SMALL, and both are real defects rather than gaps. (1) UNKNOWN GGUF TYPES SIZE AS ZERO BYTES: `block_layout` returns `(0, 1)` for `Other(_)` (frink-gguf/src/lib.rs:230), so `byte_len` short-circuits to 0 (:262-270) and `tensor_bytes` returns an EMPTY SLICE WITH NO ERROR (:334-350). A TQ2_0 file does not fail to parse, it silently under-counts in every size estimate and `inspect` plan, then fails later somewhere less informative. Should fail loudly at the layout. (2) MXFP4 IS ACCEPTED AS A WEIGHT MATRIX (loader.rs:1172) AND AS AN MoE EXPERT (:1220) BUT REJECTED BY `load_f32_vec` (:1124), despite `dequant_mxfp4_gguf` existing and `WeightMatrix::dequant` already calling it. An MXFP4 1-D norm or bias tensor is a hard load error. One missing match arm, and it violates the invariant that function's own comment states at :1104-1109. ALSO: genuinely missing live ggml types are TQ1_0/TQ2_0 ternary plus NVFP4/Q1_0/Q2_0; the `Q4_0_4_4` family is missing but has been REMOVED from ggml, so it is not a gap."
    status: pending
  - id: docs-models-md-is-wrong-in-both-directions
    content: "SMALL, and unusual because it is the only place found where the docs UNDERSTATE frink while overstating it in the same sentence. docs/MODELS.md:157-159 is wrong on three counts: IQ4_XS HAS a Metal matvec AND a Metal simdgroup GEMM (frink-metal/src/gpu.rs:4819, :3320, tested :7141, :8327); IQ4_NL and IQ4_XS BOTH have NEON kernels (frink-quant/src/lib.rs:4219, :4259); and 'scalar + AVX2' OVERSTATES the second tier, since IQ2_XS, IQ2_S, IQ3_S, IQ1_M and GGUF-block MXFP4 are scalar only, as the code says at :5551-5556. SEPARATELY, a real inconsistency worth a decision: a Q5_0 Metal GEMM EXISTS (gpu.rs:3039, exposed :3318) but Q5_0 is excluded from both `metal_mul_mm_kind_supported` and `metal_matvec_kind_name` (frink-core/src/weight_matrix.rs:313-343), so Q5_0 MoE experts run on Metal while Q5_0 dense weights fall to CPU, and there is no Q5_0 Metal test."
    status: pending
  - id: cpu-act-interleave-hoist
    content: "Hoist Q8_K activation interleave out of gemm_q4_kx8_q8_k_neon_i8mm into a Q8KActivationsX4 built once per apply_batch (llama block_q8_Kx4 / ggml_quantize_mat_q8_K_4x8). ALREADY LANDED BEFORE wf/cpu-kernels, verified against the code and not against a status: repack.rs has Q8KActsX4 + prepare_q8_k_acts_x4 (and Q8ActsX4 + prepare_q8_acts_x4 for the Q8_0 family), and all five apply_batch arms in weight_matrix.rs build the quads with acts.par_chunks(4) before the row-group loop opens. Nothing was left to hoist. wf/cpu-kernels extends the same idea one level out — see cpu-actquant-flat — and separately hoists the OTHER per-call cost the quad hoist left behind: each gemm_*_group_x4 re-ran is_aarch64_feature_detected!(\"i8mm\") on every (row-group x quad) call, which is 131k calls on one Mistral-7B q_proj at batch 512, and a relaxed atomic load is not something LLVM may hoist out of the caller's loop. New frink_quant::AccelX4 is probed once per matmul and passed to the new _on entry points; the probing wrappers stay for other callers. Bit-exactness of the dispatch parameter is pinned by accel_x4_only_picks_a_kernel_it_never_changes_the_answer, which also checks that AccelX4::Portable still reaches the scalar reference on an i8mm host. NOT MEASURED — no benchmarks were run on this branch, wave 4 owns every number"
    status: completed
  - id: cpu-kill-transpose
    content: "Delete the 7 serial [rows,batch] -> [batch,rows] scatter loops; have group kernels write [batch,rows] directly with a dst row stride (llama forward_mul_mat_one_chunk). FIRST HALF ALREADY LANDED, verified on wf/cpu-kernels against the code: weight_matrix.rs allocates the result directly in [batch, rows] and each rayon task writes it through the BatchOut raw-pointer wrapper; the [rows][batch] staging vec and the serial post-pass are both gone, and the doc comment on BatchOut records the removal. SECOND HALF DELIBERATELY NOT DONE and this is the reason, not an omission: what remains is not a transpose but a 32-float stack tile (Q4_KX8_NROWS x 4) that the kernel fills and the caller copies out in nrows-contiguous runs — out[(t*nc+j)*rows + g*nrows + r] is contiguous in r, so the copy is already 4 sequential 8-float stores out of L1. Giving the kernels a dst row stride instead would mean passing an aliasing raw pointer into frink-quant so parallel row-groups can write one shared plane, which trades a real unsafe invariant for a copy that is not on any profile. Reopen only with a `sample` that shows the copy"
    status: pending
  - id: cpu-i8mm-q5k-q6k
    content: "Port ggml_gemm_q5_K_8x8_q8_K + ggml_gemm_q6_K_8x8_q8_K; flip q5_kx8_interleave/q6_kx8_interleave to 8 under i8mm. Upstream DOES have these — the in-tree 'until it lands' comments are wrong. ALREADY LANDED BEFORE wf/cpu-kernels, verified against the code: repack.rs neon has gemm_q5_kx8_q8_k_neon_i8mm and gemm_q6_kx8_q8_k_neon_i8mm, both #[target_feature(enable = \"neon,i8mm\")]; q5_kx8_interleave() and q6_kx8_interleave() both return 8 on an i8mm host; q5_kx8_gemm_uses_acts_x4 / q6_kx8_gemm_uses_acts_x4 gate the call sites; and q5_kx8_interleave8_neon_matches_references_when_available / q6_kx8_interleave8_neon_matches_references_when_available pin them to the scalar references. The 'until it lands' comments are gone with them"
    status: completed
  - id: cpu-i8mm-q8_0-q4_0
    content: "Un-pin Q8_0X4_INTERLEAVE/Q4_0X4_INTERLEAVE from 4; port ggml_gemm_q8_0_4x8_q8_0 + ggml_gemm_q4_0_4x8_q8_0 (SMMLA) + ggml_quantize_mat_q8_0_4x8. STALE FOR Q8_0 (2026-08-18, wf/cpu-prefill-gemma3): a `sample` of Gemma-3-1B Q8_0 cpu pp512 shows frink_quant::repack::neon::gemm_q8_0x4_q8_0_neon_i8mm as the top non-idle symbol (57.0%), i.e. the SMMLA Q8_0 GEMM already exists and is on the hot path. Whatever is left of the Q8_0 rows is NOT the arithmetic tier. Re-scope to Q4_0 only, or close. CLOSED on wf/cpu-kernels: the note was right about Q8_0 and equally true of Q4_0, which the re-scope guessed was still owed. gemm_q4_0x4_q8_0_neon_i8mm exists next to the Q8_0 one, both under neon,i8mm; the two INTERLEAVE constants are still 4 but they are now only the DotProd default — q8_0x4_interleave() and q4_0x4_interleave() both return 8 when i8mm is detected, which is the un-pinning the item asked for; prepare_q8_acts_x4 is ggml_quantize_mat_q8_0_4x8 with the quantization already done; and q8_0_q4_0_interleave8_neon_matches_references_when_available covers both. Nothing in this item was left to build"
    status: completed
  - id: cpu-actquant-flat
    content: "De-nest activation quantization (serial internals, parallelize once at apply_batch over row-quads into one wdata buffer); share one quant pass across q/k/v and gate/up. FIRST CLAUSE ALREADY LANDED: quantize_activations_q8_k / _q8 are serial inside and carry the comment explaining why (every batch caller is already in a Rayon region), and apply_batch opens exactly one region over positions. SECOND CLAUSE FINISHED ON wf/cpu-kernels, and it was only half done: quantize_batch_acts shared the QUANTIZATION across q/k/v and gate/up, but each consumer then rebuilt the interleaved quads for itself, so a Q4_K attention block ran ggml_quantize_mat_q8_K_4x8 three times over the same plane and a dense FFN twice. BatchActs now carries the quads (and the width they were built at) alongside the acts, so q/k/v prepare one set between them and gate/up one. Sharing crosses kinds on purpose — Q4_K/Q5_K/Q6_K read one Q8_K set, Q8_0/Q4_0 one Q8_0 set — because the three predicates and the three quad widths are identical; a const assert in weight_matrix.rs fails the build if Q4_KX8_GEMM_NC / Q5_KX8_GEMM_NC / Q8K_ACTS_X4_NC ever stop agreeing, since that divergence would be a wrong answer rather than a panic. A set from another width or another position count is refused and re-quantized locally (shared_acts_are_reused_only_at_the_matching_length_and_width, which runs regardless of FRINK_CPU_INT_DOT because the guard is not gated on it). THE 'ONE WDATA BUFFER' CLAUSE IS NOT DONE: Q8KActsX4 owns three Vecs per quad, so a batch-512 matmul still makes ~1900 small allocations. Flattening them needs the kernels to take slices instead of an owning struct, a frink-quant API change with no measurement behind it — back-of-envelope it is under 1% of a GEMM that runs for milliseconds. NOT MEASURED — no benchmarks were run on this branch"
    status: completed
  - id: cpu-decode-scaling
    content: "CPU tg128 is the widest axis left (8 red rows, SmolLM2 2.44x, and the only axis with nothing at parity). Cause is measured: fork-join scaling, not per-thread throughput. IMPLEMENTED ON wf/cpu-pool-retry (e2bf801, NOT merged): pool retried with try_lock, poison-as-ownership, and one-runtime-per-thread; the hang ships as a regression test. NOT MERGED ON PURPOSE — the perf thesis is still unmeasured, which is half of why the first attempt was rejected. Every A/B window this session was eaten by concurrent builds (host load 3-30 against a 2.0 bar); the one partial reading taken under load showed dense +7% and OLMoE -21%, and the -21% is what motivated the one-runtime-per-thread rule, so it must be re-taken after that rule, not before. Merge only behind a quiet-host A/B on TinyLlama + OLMoE + Qwen2.5. DEADLOCK UNDERSTOOD (2026-08-14, reproduced + sampled on the branch): pool worker blocked in a rayon latch (a pool task called rayon) -> rayon workers blocked on the pool submit mutex -> submitter holding it, spinning for done. Coexistence of two runtimes with a blocking lock between them, NOT the memory ordering the module argued about, and NOT specific to FRINK_CPU_THREADS=1. Retry rules in the body: try_lock never block, leaf-only tasks, flatten apply_three / run_expert / the MoE par_iter into one region, ship the hang as a test"
    status: pending
  - id: cpu-prefill-attn-block
    content: "Block prefill attention: QK^T tile GEMM + vectorized softmax + V GEMM, replacing the per-KV-position online_attn_accumulate with 2 scalar expf per position. PARTIALLY DONE: causal_gqa_attention_prefill_shared_kv (attention.rs:632) is already the three-pass blocked form, but only the non-windowed decoder arm reaches it — see cpu-gemma3-prefill. Still owed after that: the blocked form is itself the dominant CPU cost on small models. `sample` of SmolLM2-135M Q8_0 cpu pp512 (2026-08-18, host load 41 so timing is junk, symbol shares are not) puts the inlined per-(query-block, head) closure of causal_gqa_attention_prefill_shared_kv at 8265/15700 = 53% of non-idle vs 30% for the Q8_0 GEMM, against ~8% of the model's FLOPs. Cause: pass 1 is a dot_f32 per (query, KV position) with head_dim=64 and no K reuse across the query block. Real fix is a K-tile x Q-tile register GEMM as llama does (KQ via ggml_mul_mat), not the current row-at-a-time loop"
    status: pending
  - id: cpu-gemma3-prefill
    content: "Gemma-3-1B cpu pp512 1.65x (not the 1.94x previously recorded — llama's own number was 548 under load, 468 quiet) is the worst CPU prefill row. DIAGNOSED 2026-08-18 on wf/cpu-prefill-gemma3 by `sample` over `frink bench -p 512 -n 0 -ngl 0` (15s, 1ms). Two Gemma-3-specific costs, neither shared with the TinyLlama 1.01x control: (1) SWA layers never reach the blocked attention. decoder.rs:3741 branches on layer_sliding_window(l); Gemma-3 is swa_pattern=6 so 22 of 26 layers take the Some(window) arm, which is a per-query causal_gqa_attention_windowed_softcap (attention.rs:546) built on online_attn_accumulate (attention.rs:146) — 2 scalar expf plus a head_dim-wide rescale per KV position, Rayon over 512 query rows only, not query x head. The None arm is the already-blocked causal_gqa_attention_prefill_shared_kv (attention.rs:632). TinyLlama and SmolLM2 have no sliding_window so they never enter it. Measured 7167/36476 = 19.6% of non-idle samples. At pp512 the window (512) covers the whole prompt, so this arm does the identical KV work as the fast arm — it is pure code cost. (2) geglu (matmul.rs:233) calls libm tanhf per element: tanhf 3688 + stub 218 = 10.7% of non-idle, entered from dense_ffn_batch. Gemma uses GeGLU; TinyLlama/SmolLM2 use SiLU. GEMM itself is fine — gemm_q8_0x4_q8_0_neon_i8mm is already the i8mm SMMLA kernel and is 57.0%, so cpu-i8mm-q8_0-q4_0 as written in this plan is stale for Q8_0 (correction: it is 4x4, not the 4x8 the other item claims — Q8K_ACTS_X4_NC is 4 — but the conclusion holds, the kernel exists and is on the hot path). BOTH HALVES NOW LANDED. (1) fixed on main in 62fed61 by wf/cpu-attention. (2) fixed on wf/cpu-kernels, and it reached further than this item scoped it: the identity 0.5*x*(1 + tanh(u)) == x / (1 + exp(-2u)) replaces the libm tanhf with one vectorized exponential, and the SAME rewrite applies to swiglu, whose libm expf per element was the identical cost on every SwiGLU model in the ledger — TinyLlama, SmolLM2, Qwen, Mistral, Llama — not just the Gemma family that named this item. matmul.rs now carries the ggml_v_expf kernel with NEON and AVX2 arms and a scalar one-lane twin so the vector body and the scalar tail agree bit for bit, which is what keeps decode and prefill on the same answer when the slice ends in a different place. THE CLAMP IS NOT THE SOFTMAX CLAMP: a softmax argument is <= 0, a gate argument has no sign, and clamping the high side alone divides an unbounded numerator by a saturated exponential — gelu(-1e30) came out -1.6e-8 where it should be -0 — so the kernels select zero at t >= 87 instead, which is exact to within 1.5e-37. ACCURACY, measured against an f64 reference over 240001 samples rather than argued: the naive f64 tanh reference is itself useless here (it cancels for x < -7 exactly as the f32 one does, and scored the new form at 1.97e22 relative error against a value that had already collapsed), so the reference is the algebraically identical non-cancelling form. Against it, the tanh form loses more than 0.1% of the VALUE at 5047 of the 240001 samples, up to every significant bit, and the new form at none; scaled by |x| instead the new form is 1.16x behind at its worst point (1.13e-7 vs 9.74e-8, both one to two ulp, the extra rounding being the divide). SiLU is a wash on both metrics since its formula did not change, only which exponential runs. NOT MEASURED FOR SPEED — no benchmarks were run on that branch. OWED, and it is real: crates/frink-core/src/attention.rs holds a private copy of the same ggml_v_expf kernel and matmul.rs now holds a second, because wf/cpu-kernels does not own attention.rs and the softmax copy is not public. They should be one module. The two differ only in the clamp, and that difference is documented on both sides"
    status: completed
  - id: metal-fa-mma
    content: "Port kernel_flash_attn_ext MMA (Q.K^T AND P.V via simdgroup_half8x8) for d=64 — attn.rs had ZERO simdgroup MMA, 16 of 128 lanes active"
    status: completed
  - id: metal-fa-mma-d128
    content: "Parameterise the MMA macro over head dim so it emits _d64 and _d128. A/B: Qwen3-0.6B 1.71x, Phi-4-mini 1.20x, Llama-3.2-3B 1.20x, Mistral-7B 1.13x (commit 0ee4d0b)"
    status: completed
  - id: metal-fa-mma-d256
    content: "Extend the MMA macro to d=256 (Gemma-3 metal pp512 1.18x). HEAD DIM CONFIRMED 2026-08-18 on models/hf_test/gemma-3-1b-it-Q8_0.gguf: attn_q [1152,1024] = 4 heads x 256, attn_k/attn_v [1152,256] = 1 kv head x 256, `inspect-plan` says `1 kv-heads x 256 head-dim` — d=256 is the real width. LANDED (branch wf/metal-fa-mma-d256): replaced the three `own = tiisg < D4` guards on the `so` accumulator (zero-init, softmax rescale, epilogue) with llama's lane loop `for (i = tiisg; i < D4; i += NW)` (ggml-metal.metal:6529-6535, :6826-6838, :7024-7034), which degenerates to the same single iteration on the same lanes when D4 <= NW, then instantiated gqa_prefill_fa_ext_mma_d256 and routed head_dim 256 to it. 28 KiB threadgroup at QN=8/C=64 — the last width under Apple's 32 KiB. PROVISIONAL interleaved A/B (host load 29-35, other agents building): Gemma-3-1B pp512 mma 2544.91/2564.22/2514.59 vs fa_vec 2245.59/2303.43/2201.85 = 1.13x. Guard A/B base-vs-new binaries in the same window: Qwen3-0.6B (d=128) 3226 -> 3320 mean, Llama-3.2-1B (d=64) 1907 -> 1936 mean — both inside noise, no regression. verify --backend metal identical cpu-vs-metal ids with prefill covered on Gemma-3-1B (64 and 300 tokens, MMA on and off), Qwen3-0.6B and Llama-3.2-1B. OWED: a suite run on a quiet host — RESULTS.md still advertises Gemma-3-1B metal pp512 2363.87 / 1.18x and was NOT touched"
    status: completed
  - id: suite-owed-d128
    content: "PAID (cb27b24, 2026-08-13, started at load 1.95): d=128 MMA published. Qwen3-0.6B 1.81->1.03x, Phi-4-mini 1.24->1.04x, Llama-3.2-3B 1.08->1.04x, Mistral-7B 1.10->1.05x. 25 red rows -> 21"
    status: completed
  - id: metal-moe-stack
    content: "Worst row on any backend: OLMoE metal pp512 2.62x (was 2.48x — frink 626->587, a real -6% inside host spread), and it owns the last Metal tg128 red row too (1.41x). Move MoE layers onto the fused prefill stack: MoE PrefillDenseLayerMetal variant, GPU router+top-k, wire the already-written-but-uncalled encode_moe_mm_id_map0. Kills ~112 command buffers per pp512. LANDED (ee35372): PrefillFfnMetal enum (Dense | Moe), new moe_router_mm_f32 + moe_topk_softmax_batch kernels, map0 + mul_mm_id_f16 now on the hot path. Interleaved A/B pp512 1412/1398/1417 vs 711/724/715 = 1.98x; tg128 unchanged (decode never took this path); verify cpu-vs-metal identical with prefill covered, stack on and off"
    status: completed
  - id: suite-owed-moe-stack
    content: "PAID (2026-08-14, started at load 1.82): re-ran --suite --id olmoe_q4 --backend metal, the only ledger row metal-moe-stack can move. OLMoE metal pp512 587.49 -> 1402.38, gap 2.62x -> 1.11x against llama 1552.23 measured in the same session. tg128 116.17 (gap 1.41x) unchanged, as predicted — decode never took this path. RESULTS.md re-rendered from the new receipt; no other row was re-measured, so no other row moved"
    status: completed
  - id: metal-moe-decode
    content: "LANDED and merged (e4bd770). Measured by stage ablation (FRINK_METAL_MOE_ABLATE against FRINK_METAL_GPU_TIMING) rather than guessed, and the first three hypotheses were all wrong: decode already routed on the GPU, already used one command buffer per token, and already used llama's mul_mv_id. The actual cost was the ROUTER at ~3.6 ms of ~8.5 ms GPU per token (42%), more than the experts it selects (2.9 ms), split across two single-lane kernels: f32_matvec (one thread per row, so a 64x2048 router ran as ONE threadgroup) and moe_topk_softmax (opens with `if (tid != 0u) return;`, dispatched 1x1x1). f32_matvec rewritten as ggml's kernel_mul_mv_t_t_impl shape (NR0=2, simd_sum plus threadgroup fold, coalesced lanes); decode switched to the existing simdgroup moe_topk_softmax_batch with n_tokens=1 and the single-lane kernel deleted. Router ~3.6 -> ~0.5 ms, total GPU per token -25%. CORRECTION FROM metal-barrier-ranges: this item's claim that the remaining ~2.4 ms/tok attention at ~63 GB/s is explained by '~8 barrier points per layer' DOES NOT HOLD. Barrier counts are ~0.99 per tracked op both before and after hazard tracking. The next attempt on this row should sample or GPU-capture the attention kernel, not touch synchronisation again. OWED: quiet-host suite run; RESULTS.md still says 1.41x"
    status: completed
  - id: metal-barrier-ranges
    content: "LANDED and merged (03d395c). MemRanges tracker (ported from llama's ggml_mem_ranges / ggml_metal_op_encode) replaces 15 blanket per-layer barriers in the dense PREFILL layer and the last 19 in the dense DECODE stack; zero blanket barriers remain in attn.rs. moe_ranges.rs renamed mem_ranges.rs. Both named fusions landed: add_rms_norm_f32_to_f16_batch and silu_mul/gelu_mul_f32_to_f16, bit-identical to the pairs they replace, which also retires the f32 act plane (saves batch*max_gate*4 bytes). MemRanges::serial() added for the Gemma sandwich path, where llama skips barriers too. TWO HIDDEN BUFFERS had to be pulled into the tracker to make this SAFE rather than merely fast: the shared f16 KV-dequant scratch in encode_gqa_*_with_kv, and encode_moe_prefill_ffn's private router/expert scratch, both of which were relying on the caller's blanket barrier for cross-layer ordering. THE ITEM REFUTES ITS OWN PREMISE, and this is the load-bearing finding: FRINK_METAL_BARRIER_LOG reports ~0.99 barriers per tracked op BEFORE AND AFTER, on dense and MoE decode alike. A B=1 decode layer is a strict dependency chain, so a range tracker has nothing to let overlap and can only narrow each barrier's scope, worth ~1%. The arithmetic agrees: Llama-3.2-1B Q4_K_M is ~0.8 GB and 6.1 ms/tok is ~130 GB/s against the M2 Pro's 200 GB/s, so decode is already weight-bandwidth-bound. PROVISIONAL, host load 6-34: prefill 398.6/400.9/397.1 -> 394.7/394.1/392.3 ms (-1.3%, 3 of 3); decode inside spread. 54 frink verify --backend metal runs, all green with prefill covered, across dense/MoE/SWA/sandwich/QK-norm/small-hidden models and prompt lengths 64/100/300/512, because a missing barrier is a race and one green run proves nothing. OWED: quiet-host suite run; no tok/s published"
    status: completed
  - id: metal-mm-occupancy
    content: "LANDED and merged (fcf119c). BC_OUT-templated mul_mm_sg with _a exact-tile entry points needing 6144B of threadgroup memory instead of always 8192B, selected by llama's own bc_out condition. New test mul_mm_sg_variant_picks_the_small_threadgroup_only_on_exact_tiles is pure logic and catches the failure that matters: picking the _a variant for a ragged shape would write past the matrix. A (128, 512, 64) shape was added to the seven *_matches_the_matvec_it_replaces hardware tests so the new kernels are actually exercised. PROVISIONAL, 704-token prompt: 397.2 -> 391.9 ms mean (-1.35%, 6 of 8). LIMIT WORTH KNOWING: the variant fires only when batch % 32 == 0, and batch is the whole prompt length because prefill is not chunked, so on real traffic it fires roughly 1 prompt in 32. Padding the batch would make it always fire but changes what the GEMM computes; not attempted. OWED: quiet-host suite run"
    status: completed
  - id: tooling-verify-prompt-len
    content: "`frink verify` passed vacuously on prefill kernels (fixed 6-token prompt vs the n_q >= 8 gate). Added --prompt-tokens/--prompt, and every verdict now states whether prefill was covered. Landed bfd1c1a"
    status: completed
  - id: tooling-cpu-metal-divergence
    content: "DID NOT REPRODUCE (2026-08-13, bfd1c1a): with the new length-aware verify, greedy ids match cpu vs metal at 41/49/128/300-token prompts on TinyLlama and 49/300 on Phi-4-mini, and across 8 models at 40. Either the MMA work fixed it or the original was logit drift that never flipped an argmax. Reopen only with a reproducer"
    status: completed
  - id: tooling-kernel-registry
    content: "Sealed kernel-lookup registry: record every dispatch lookup at model build, seal, warn/fail on a later miss that takes a fallback. Landed 99a69ab (frink-core/src/kernel_registry.rs, docs/CONFIG.md)"
    status: completed
  - id: tooling-quant-sensitivity
    content: "LANDED as `frink quant-sensitivity` (branch wf/tooling). Round-trips ONE tensor at a time through a candidate format (q4_0 or q8_0), scores relative_mse per block against the values it started from, swaps the result into the loaded model, runs a real prefill and reports KL(clean||perturbed) over the next-token distribution. The `propagate the float output forward` clause is satisfied structurally rather than sequentially: every other weight is the checkpoint's own on every run, so no tensor inherits damage from the layers above it, which is stronger than propagating a clean activation through a partially-quantized stack. Both a weight-space and an output-space column are printed because they disagree, and the disagreement is the finding -- a tensor can round-trip badly and barely move the logits, or the reverse; only the second is a reason to spend bits. FIRST RESULT (Llama-3.2-1B-Instruct-Q4_K_M, 112 tensors, 16 prompt tokens, candidate q4_0, 2m17s): total 3.17e-2 nats, no single tensor flips the greedy token on its own, and the family rollup is ffn_down 32.2% / attn_output 13.7% / attn_v 13.0% / ffn_up 12.6% / ffn_gate 10.9% / attn_q 9.0% / attn_k 8.5%. So Q4_K_M's static rule (keep ffn_down and attn_v a tier higher) is RIGHT on this checkpoint and now has a number behind it; the surprise is attn_output, which no mix protects and which carries more than attn_v here. Worst single tensor is blk.1.ffn_down (4.6e-3 nats, 7x the median). Two guards: the sweep refuses to start with FRINK_CPU_INT_DOT=1, whose repack cache is keyed by BUFFER ADDRESS and would serve a swapped-in tensor another tensor's repacked bytes, and it forces the CPU path because swapping a WeightMatrix invalidates the Metal packed-expert planes. Only q4_0/q8_0 are candidates: both have an exactly-specified rounding rule and the fastest CPU kernels, so an insensitive tensor is one that can move to a FASTER kernel and not merely a smaller one. The K-quants' block search belongs in a quantizer, not a diagnostic"
    status: completed
  - id: tooling-quality-eval
    content: "Real gap: frink validates NUMERICS (NumPy goldens) but cannot answer 'did this quantization damage the model?', so no honest quality claim can go in docs/MODELS.md. Shape: fix the input, reference at full precision, sweep candidates, report a distortion metric (KL over logits), pick the smallest clearing a budget. Neither reference project has an LLM implementation to lift"
    status: pending
  - id: tooling-bench-discipline
    content: "ADOPT into frink bench: warmup before any timing (shader/JIT compilation), temp=0 on timed runs, assert prompt length before AND after generation, assert zero cache hits, record thermal pressure with each result. frink is already AHEAD on repeat-and-median and on having a checked-in ledger at all. LANDED on branch wf/bench-discipline-rest (e3159aa, c4911a5), NOT merged to main. Earlier work (afdf2de, 2010be8) enforced the quiet-host load bar and the one-process-per-model gate; everything else now lives in frink-cli/src/bench_guard.rs as free functions over plain data, each with a test that fails when the invariant is broken (12 mutation checks run, all killed). Warmup: WARMUP_REPS discarded reps, asserted AFTER the fact by check_timed_samples, so a later edit to a loop bound cannot put a shader compile back into a published median; `-r 0` is refused instead of printing a median over an empty sample set. Prompt length: check_prompt_before (length + in-vocabulary) and check_prefill_after / check_decode_after on every rep, on the Gemma-4 paths too, which previously had NO assertions at all. Zero cache hits: check_caches_cold inspects k/v element counts as well as seq_len, because a cache whose length was reset while keeping its contents is exactly what prefix reuse looks like from outside; an empty cache list is a failure, not a vacuous pass. temp=0 has TWO halves, because it is not one claim: check_same_workload digests the token stream fed inside the timed region (identical input), and check_same_result compares the greedy token id of each rep's final logits (identical output) — the digest alone cannot see the engine computing something different from the same tokens, e.g. routing that depends on a warmed expert cache or a reduction whose order varies. check_sample_rates refuses a non-finite rate, which no length or cache check can see. Thermal pressure IS measured, not implied: macOS answers NSProcessInfo.thermalState without sudo and with no new dependency (three libobjc symbols); pmset -g therm's CPU_Speed_Limit was the old signal and is absent on Apple Silicon, so the field was null on every run here while implying a measurement — it is kept as a secondary cap. ThermalReading carries an explicit `measured` flag and a `source`, keeping `we did not look` structurally distinct from `we looked and it was nominal`; a run refuses at `serious` or above behind the same `--max-load 0` escape as the load bar. Receipt schema 2: host_thermal_start/_end objects, warmup_reps, per-test workload_digest, and the non-default FRINK_* knobs in effect. CAVEAT: check_same_result was verified end-to-end on CPU only (forced divergence on a 135M GGUF made it fire with the right message); no Metal or CUDA host was available, so if a GPU kernel is nondeterministic at the argmax this gate will refuse there and that refusal is the correct signal, not a bug to silence"
    status: completed
  - id: tooling-layer-divergence
    content: "LANDED as `frink layer-divergence` (branch wf/tooling). One prefill per backend, in a child process each because the backend is a process-lifetime OnceLock, then every layer's KvCache is read back and scored per head: the SPREAD of the per-head magnitude ratios, with the mean printed beside it so the reader can watch the mean fail to notice. No decoder instrumentation and no env gate were needed -- K and V for every position are already in the cache, laid out [seq, n_kv_heads, head_dim], so the per-head magnitudes are there for free. Reading rule: layer L's K/V come from layer L's INPUT, so a first divergence at L means L's norm/QKV projection or whatever produced its input; layers below are exonerated. What it cannot see is layer L's own attention output and FFN, which reach the cache only through L+1. MEASURED NOISE FLOOR, which is what makes the threshold honest: cpu-vs-cpu is exactly 1.0 on every head (the self-test), cpu-vs-metal on Llama-3.2-1B Q4_K_M and OLMoE Q4_0 spreads 1.6e-5 to 1.3e-4, and the default --tol 1e-3 sits ~8x above the worst of that. TWO THINGS THE FIRST REAL RUN FOUND. (1) Metal's fused prefill stack calls `advance_len` on the host KvCache and keeps the values on the device, so the host copy comes back the right LENGTH and full of zeros, and `sync_metal_attn_kv_to_host` cannot repair it in place because it appends only past `cache.seq_len`. The fix here is to pull into a SECOND, empty set of caches, which the device fills from position 0; the tool reports how many layers it had to read that way (16 of 16 on both models). Anyone reading host KV after a Metal prefill -- prefix cache, continuous batching -- is reading zeros. (2) The Metal MoE path never records `MoeWeights::activation_counts`: OLMoE's CPU side reports 128 selections per layer and the Metal side 0. That counter is what `placement_plan` calls observed hotness, so expert placement on Metal is planning against a histogram nobody filled in. The routing column prints `no counts` for exactly this rather than scoring it 0.0000 TV, and an all-zero-magnitude side is a hard error, not a row of 0.00e0 -- a backend whose K and V are all zero has a per-head ratio spread of zero, which would otherwise read as perfect agreement"
    status: completed
  - id: coverage-fail-closed
    content: "BLOCKING CORRECTNESS: ~50 archs are admitted to the generic GQA path and emit wrong logits instead of refusing. Gate on required graph features; refuse what is not implemented. LANDED (c3db570) as a tensor-consumption gate rather than a feature list: ShardedGguf records every tensor name a loader looks up and the load refuses on leftovers, which catches gpt-oss attn_sinks and ffn_exp_probs_b by construction instead of by enumeration. Refused exactly 1 of the 13 local GGUFs — Phi-4-mini, correctly: partial rotary, LongRoPE factor sets, and rope.scaling.attn_factor were all unimplemented and are now implemented on CPU"
    status: completed
  - id: coverage-phi-metal-rope
    content: "LANDED and merged (1a30a91). rot_dim and mscale uniforms on rope_norm/rope_neox, so partial rotary (n_rot < head_dim) and LongRoPE mscale run on Metal and Phi-4-mini is no longer refused the Metal path. A second bug was found and fixed on the way: rope_freqs was sized by the head width rather than by n_rot. frink verify was also extended to cover the FUSED Metal attention rather than only the matvecs, which is what made the check meaningful for this model. The CPU-side trap recorded in 6dbd860/621acad was avoided: attn_factor scales only the rotated channels, never the pass-through channels beyond n_rot. OWED, and it is not optional: Phi-4-mini's published Metal pp512/tg128 rows were taken on the WRONG GRAPH before the refusal landed and are known-invalid. docs/MODELS.md currently shows its Metal decode cell as refused. Both that cell and the benchmarks/suite.json entry need revisiting from a quiet-host re-measure, not from a guess"
    status: completed
  - id: tooling-answer-parity-instrument
    content: "LANDED as `frink parity` (0165504 + 8d7e664, branch wf/answer-parity). `.local-scripts/llama_logits.c` links the installed libllama and dumps the last-position logits for an EXPLICIT token-id sequence — same spirit as the IQ-tier goldens linking ggml's own ggml-quants.c, since a reference you re-implemented is not a reference. frink tokenizes and hands llama.cpp the ids, so the tokenizer is removed from the experiment rather than trusted. Reports KL both ways, total variation, max |delta p|, top-k overlap, and where llama's top-1 ranks for frink, plus WHICH frink backend ran (the reference is pinned to llama CPU, so a Metal-side run is a compound comparison and says so). Four verdicts because three are not failures: MATCH (within f32 accumulation-order noise), DRIFT (same top-1, distributions moved further than reordering explains), TIE-FLIP (top-1 differs but llama's own top-2 margin is under the observed noise), WRONG. FIRST RESULT, and it settles this item's own premise: TinyLlama Q8_0 cpu-vs-cpu is MATCH — KL(llama||frink) 8.4e-5 nats, top-10 overlap 10/10, same top-1 — at 8 and at 32 tokens. So the greedy-text divergence recorded above was drift amplification, NOT a wrong graph, exactly as the tie-flip hypothesis predicted. SWEEP DONE (17 local checkpoints, cpu-vs-cpu, 32-token prompts): 16 MATCH, 1 DRIFT. The DRIFT was Phi-4-mini — the one model whose RoPE code was newest — and it was a REAL BUG, not a tolerance question: ggml folds attn_factor into cos/sin inside rope_yarn so it reaches only the rotated channels, and ggml-cpu/ops.cpp then copies [n_rot, head_dim) through untouched, while frink scaled the whole head. Phi rotates 96 of 128 dims with attn_factor 1.1902, so 32 dims per head were scaled that llama.cpp leaves alone. Fixed; the same instrument confirms it (KL 1.808e-3 -> 2.358e-4 nats, top-10 7/10 -> 9/10, DRIFT -> MATCH). So the instrument paid for itself on its first sweep, and the 'answers match llama' half of quality-gates now has an oracle behind it for every local checkpoint"
    status: completed
  - id: coverage-f16
    content: "F16 tensors did not load at all (GgmlType::F16 parsed and sized, no dequant arm anywhere). dequant_f16 + shared widen_plain_float across all 7 loaders. Landed 7ef74f1"
    status: completed
  - id: coverage-iq-tiers-published
    content: "SUPPORT, highest coverage priority: ggml tags 17/21/22/29 (IQ2_XS, IQ3_S, IQ2_S, IQ1_M) fall to GgmlType::Other in frink-gguf, so 5 of the 16 published Unsloth UD-* variants cannot be decoded. IQ3_S is worst — it appears inside IQ3_M mixes and the low-bit UD recipes docs/MODELS.md already targets. LANDED: all four decode on CPU (scalar, matching the sibling IQ formats' state — no NEON/GPU), wired through QuantKind + all six loader tables + dequant_any. Validated by linking llama.cpp's ggml-quants.c and asserting BIT-EXACT equality with dequantize_row_iq2_xs/_iq2_s/_iq3_s/_iq1_m on 4 blocks each, including the all-0xff maximum-grid-index/all-signs corner; mutation-checked non-vacuous. NOT validated on a real published UD-* checkpoint — none downloaded"
    status: completed
  - id: coverage-jinja-templates
    content: "DONE (branch wf/coverage). CORRECTION TO THIS ITEM'S OWN TEXT, verified against the code rather than the plan: `NEITHER THE CLI NOR THE SERVER CALLS IT` stopped being true at 85a0905. Both call sites are switched today -- `frink-server`'s PromptTemplate wraps frink_models::chat_template (chat_template.rs:55-104, five loaders in model.rs plus Kimi via from_source) and `frink-cli`'s ChatKind compiles the same Jinja through the same constructor (run.rs:423-462, all four backend paths). Both read the identical four metadata keys, so they cannot disagree on WHICH template. WHAT WAS STILL WRONG, and is the actual work of this branch: they disagreed with HuggingFace and llama.cpp on the rendered BYTES. minijinja defaults trim_blocks/lstrip_blocks to false; transformers compiles every chat template with ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True) and llama.cpp hardcodes the same pair for chat templates (common/jinja/lexer.cpp:112-118). A template written with explicit `{%- -%}` control is unaffected, which is why 14 of 15 local checkpoints rendered identically and this went unnoticed; TinyLlama-1.1B-Chat is not written that way, and every prompt it was served carried 13 bytes of stray blank line (`\\n\\n<|user|>...</s>\\n\\n\\n\\n\\n<|assistant|>`) that the checkpoint was never trained on. Fixed in chat_template.rs::new_environment, mutation-checked. EVIDENCE: tests/templates grew from 5 real templates to 15 (added Mistral-7B-Instruct-v0.2 -- the `[INST]` case this whole item is named after and which was NOT among the five -- Phi-3-mini, Phi-4-mini, DeepSeek-R1-Distill, Yi-1.5, gemma-2, Qwen1.5-MoE, Qwen3, SmolLM2, Llama-3.2), each extracted verbatim from a GGUF in models/. Each now has .expected goldens for two conversation shapes, generated by REAL jinja2 via scripts/chat_template_goldens.py, i.e. by a different implementation than the code under test; 28 exact-byte comparisons plus 2 pinned refusals (gemma-2 and Mistral v0.2 raise on a system turn, and the template's own raise_exception message must reach the caller). DISCLOSED DEVIATION, now documented instead of mis-stated: frink's tojson sorts keys, which is stock jinja2's policy but NOT what transformers (sort_keys=False) or llama.cpp (refuses sort_keys=true, value.cpp:251) do. It cannot be fixed in this module -- serde_json::Map is a BTreeMap without a workspace-wide preserve_order feature, so a tool schema is already sorted before the filter runs. Affects key order inside a <tools> block only. NOT DONE, and out of this branch's file ownership: frink-cli's `run-kimi` reads tokenizer_config.json for special tokens only and hands the prompt to kimi_generate unframed, while frink-server's Kimi loader reads chat_template out of that same file (model.rs:576-588) -- the two Kimi paths really do disagree, and closing it needs crates/frink-cli/src/main.rs"
    status: completed
  - id: coverage-stop-token-truth
    content: "SUPPORT, cheap + high impact: stop on the whole EOG set, not `eos_token_id` alone, and settle who adds BOS. DONE (branch wf/coverage-stop-bos). STOP TOKENS: the CLI half was 60f7435; the server half is landed now. `eos_id: Option<usize>` is replaced by `tokenizer::StopTokens` across the five *Loaded structs, their five *Model mirrors, generate/generate_engine and the batch_scheduler worker, so `I only have the metadata EOS` is an explicit `StopTokens::from_eos` at the two call sites that really are that (the synthetic demo model, the batcher test harness) rather than the default everywhere. Kimi K3 gets its set from special-token NAMES (`StopTokens::from_special_tokens`), so `[EOT]` ends a turn there too. Two regression tests, one per decode path, each built so the metadata EOS is a token the model never samples and the turn ender is the one it does; both mutation-checked (disabling either stop check fails exactly its own test). BOS: measured, not reasoned about. New `crates/frink-models/tests/bos_policy.rs` renders every local GGUF template through the minijinja evaluator, encodes with that checkpoint own tokenizer, and counts leading BOS ids. Result over 26 checkpoints: 18 templates emit `{{ bos_token }}` themselves (gemma-2/3/4, Mistral, Phi-3, Llama-3, DeepSeek-R1-Distill), 8 do not, TinyLlama is the local Unsloth-style stripped case, and 0 double. So frink does NOT double-BOS today — but only because every prepend site had the idempotent guard; mutation-checked, making the insert unconditional doubles BOS on 6 of the 26. The rule is now stated once in `tokenizer::prepend_bos` (template owns BOS when it prints one, loader otherwise, added idempotently), the four hand-inlined copies of the guard are gone, and it is written down in docs/CLI.md#who-adds-bos + docs/API.md. CORRECTION to this item premise: chat templates are still SNIFFED on the serving path — `frink_models::chat_template` (minijinja) exists and is tested but neither frink-cli nor frink-server calls it, they still use their own marker-sniffing enums. So today the production render emits no BOS at all and the double-BOS risk is latent; the sweep measures the path coverage-jinja-templates will switch to. CORRECTION kept from before: llama.cpp DOES treat gpt-oss `<|end|>` as EOG (llama-vocab.cpp:2806)"
    status: completed
  - id: coverage-mxfp4-gptoss
    content: "CPU GRAPH LANDED AND VALIDATED (2026-08-18, wf/coverage-gptoss). CORRECTION to this item's premise: gpt-oss did NOT load-and-answer-wrongly — it died on a missing `blk.0.ffn_norm.weight`, because gpt-oss norms the FFN input with `post_attention_norm`. The tensor-consumption gate from coverage-fail-closed was never reached. Everything behind that first missing tensor was still wrong, and three of those were silent. LANDED on CPU as one all-or-nothing `Decoder::gpt_oss` side table: attention sinks (new `causal_gqa_attention_sinks` in frink-core, windowed+full in one kernel, sink fold copied from ggml's flash-attn `sinks` block), swiglu_oai (clamped gate / alpha sigmoid / +1 up, from `ggml_compute_forward_swiglu_oai_f32`), SOFTMAX_WEIGHT routing (top-k on raw biased logits THEN softmax over winners only — frink's existing Softmax gating would have weighted the same experts wrong), attn-output + router + all three expert biases, and `post_attention_norm` read into the pre-FFN slot. THREE REGISTRY BUGS FIXED that were silently wrong for archs that already load: gpt-oss was on the interleaved-RoPE list (llama says NEOX, live load prints `rope type = 2`); `attention.sliding_window` with no `sliding_window_pattern` key windowed EVERY layer (llama hardcodes the period per arch — now `capability::default_swa_pattern`: gpt-oss 2, cohere2/exaone4/olmo2 4, gemma2 2, gemma3 6); SWA rope base defaulted to 10000 for everything (only the Gemma-3 lineage relies on that llama_hparams default — now `swa_rope_base_follows_model`, so gpt-oss SWA layers stop rotating at 10000 instead of 150000). VALIDATION: `crates/frink-models/tests/gptoss_graph.rs` vs golden logits from llama.cpp's OWN gpt-oss implementation reading the same synthetic 2-layer/4-expert fixture (`scripts/make_gptoss_fixture.py` + `scripts/gptoss_reference_logits.cpp` linked against a real libllama). Worst case 1.8e-7 across prefill/decode/multi-seq. Reaching that needed the reference pinned to an F32 KV cache with flash-attn DISABLED — both accumulate in F16 and put a ~1.6e-4 floor under the comparison, wide enough to hide a real graph error. Mutation-checked: killing sinks / router bias / expert bias / o_bias / SWA pattern / NEOX layout each moves logits >1000x tolerance. GPU REFUSES rather than disagrees: `layer_supports_metal_attn` + `metal_prefill_dense_stack_run_len` return early, paged-KV decode asserts. STILL OPEN: MXFP4 Metal+CUDA kernels (no GPU in that environment, untouched); the gpt-oss FFN runs routed experts serially with no batched/fused fast path (correctness-first, deliberate, and it is a real prefill cost); UNVALIDATED on a real published gpt-oss checkpoint — none was downloaded, so the claim is 'matches llama.cpp on a synthetic checkpoint carrying every structural feature', not 'runs gpt-oss-20b'"
    status: completed
  - id: coverage-cheap-archs
    content: "HALF LANDED (merged), and the fixture built for it found a bigger bug than the item. LANDED: exp_probs_b is now loaded and applied to the selection score as llama's build_moe_ffn does, plus expert_weights_scale (was ignored) and expert_weights_norm (was guessed). CORRECTION TO THIS PLAN'S OWN TEXT: the on-disk tensor is `blk.N.exp_probs_b.bias`, NOT `ffn_exp_probs_b` as this item and two loaders spelled it. mla_gguf_loader and glm52_gguf_loader had the same wrong name AND synthetic fixtures that wrote the wrong name, so their tests agreed with the bug; GLM failed loudly, MLA was SILENT, meaning a real DeepSeek-V3-shaped checkpoint routed with its bias dropped. THE BIGGER FIND: 24 archs on the generic GQA path rotated the wrong RoPE pairs (afmoe, apertus, bailingmoe2, codeshell, dots1, exaone, exaone-moe, grovemoe, hunyuan-dense/moe, laguna, mellum, mimo2, minicpm3, nemotron, openelm, orion, plamo, plamo3, seed_oss, smallthinker, starcoder2, step35, talkie). tests/rope_layout.rs pins the table against a transcription of llama_model_rope_type. Proof for both: scripts/make_dots1_fixture.py plus tests/moe_routing_bias.rs compare prefill/decode/multi-seq against golden logits from llama.cpp's own dots1 implementation on the same file, worst case 1e-6, and it was the RoPE fix that took it there from 1.1e-2. REFUSED RATHER THAN FAKED: granite/minicpm multipliers (residual_scale touches every CPU residual add and the fused Metal kernels, and minicpm applies its three multipliers even when the GGUF omits every key) and cohere2 parallel residual, extended after reading each source to command-r, cohere2, cohere2moe, falcon, gptneox, phi2, plamo. GPU MoE fast paths now refuse a biased or scaled layer rather than route without the bias. SECOND PASS (branch wf/coverage) found the same class of bug one level worse, by auditing llama_model_rope_type's FIRST group rather than its NEOX group. `LLAMA_ROPE_TYPE_NONE` archs must not be rotated AT ALL, and five of them -- gpt2, mpt, refact, bloom, jais -- were admitted as GenericGqa { rope: Neox }. gpt2 uses learned absolute position embeddings; the other four use ALiBi. Nothing downstream could have caught it: bloom and refact hardcode f_max_alibi_bias = 8.0f in load_arch_hparams with no GGUF key, so the metadata gates see nothing, and mpt carries no tensor the generic loader fails to consume, so assert_every_tensor_consumed sees nothing either. WORSE, tests/rope_layout.rs could not catch it BY CONSTRUCTION: its lookup miss is a `continue`, and the NONE group was deliberately absent from the transcription, so the exact archs with the highest stakes were the ones the pin skipped. Now closed: capability.rs refuses all five by name with the reason each needs, LLAMA_NO_ROPE transcribes the group (21 names), and no_rope_architectures_never_reach_a_rotating_path asserts none of them is on a rotating path. SIXTH CASE, conditional and therefore in loader.rs not the registry: baichuan is one arch string covering two positional schemes and llama.cpp picks between them on block_count alone (baichuan.cpp:11-14, with its own 'TODO: become GGUF KV parameter'), so Baichuan-13B (40 layers) uses ALiBi and no RoPE while the 7B rotates -- refused at block_count == 40, with the 7B proved to still pass the gate. NOTHING LEFT TO ADD: the registry now covers all 139 names in the pinned LLM_ARCH_NAMES, and the_reference_table_is_complete_enough_to_be_worth_pinning asserts every transcribed name resolves, so it cannot silently fall behind. STILL OPEN: the multiplier and parallel-residual math itself, bias plus expert-groups (the refusal is at loader.rs:1835 and implementing it means frink-moe + decoder.rs, owned by other waves), and no validation on any published dots1-family checkpoint"
    status: pending
  - id: hygiene-clippy
    content: "Restore the documented `clippy --workspace --all-targets -- -D warnings` gate. Was red at HEAD: 10 errors default-features + 25 more under --features metal. Landed c8a4cc6"
    status: completed
  - id: legacy-cleanup
    content: "MOSTLY DONE (merged), audited rather than assumed. DELETED: encode_moe_gather_rows / encode_moe_scatter_rows, their two MSL kernels and the contig_in/contig_out scratch buffers, all #[allow(dead_code)] with zero callers and superseded by the GPU map0 kernel. NOT DEAD, DELIBERATELY KEPT, each checked for callers first: pack_q8_k_qs_x4_i8 is already gone from production and its only survivor is the golden reference the replacement is checked against; the by_row transposes were already gone (zero hits); the CUDA per-position batch arm is LIVE, and without it a batched prefill falls to CPU and never touches the GPU, so deleting it would be a regression while CUDA mul_mm stays out of scope; the host moe_mm_id_map0 is still called by launch_moe_prefill_q4_0"
    status: completed
  - id: suite-validate-every-change
    content: "MANDATORY per improvement: full `bench --suite --fit-host --skip-missing` + `--render` on a quiet host after every landed change; compare against the previous ledger; revert or explain any regressed row"
    status: pending
  - id: quality-gates
    content: "Golden/kernel tests + answer-parity smoke; row closed only at gap <=1.0x AND answers match llama"
    status: pending
  - id: close-all-red-rows
    content: "Definition of done: all 21 currently-red rows (6 Metal pp512, 1 Metal tg128, 6 CPU pp512, 8 CPU tg128) at gap <=1.0x, and Gemma-4 given a real llama baseline so it stops being unmeasurable"
    status: pending
isProject: false
---

# llama.cpp parity push

> Working plan for closing the measured gaps against
> [`benchmarks/RESULTS.md`](../../../benchmarks/RESULTS.md). Re-ranked
> **2026-08-10** from a four-way read-only audit of `frink` against
> `.scratch/llama.cpp` (CPU prefill, CPU kernel coverage, Metal, and
> model/weight coverage). Every claim below carries a `file:line`; the two
> load-bearing ones were re-verified by hand before landing this document.
>
> Ordering is by measured gap × known cause, not by phase number. Phase 1
> (CPU prefill) and Phase 4 (coverage) are independent and can proceed in
> parallel. The frontmatter `todos` list is the checklist.

## Where this stands (2026-08-13, ledger regenerated)

The owed suite run is **paid** (`cb27b24`, started at load 1.95). Done and
published: **Phase 1 CPU prefill**, **Metal dense prefill**, **d=64 MMA**,
**d=128 MMA**. Done as correctness/tooling (no row moves): **sealed kernel
registry** (`99a69ab`), **F16 loading** (`7ef74f1`), **prefill-capable
`frink verify`** (`bfd1c1a`), **the clippy gate** (`c8a4cc6`).

**21 red rows** (29 at the start of the push, 25 before the d=128 run).
Still 21 after `metal-moe-stack`: OLMoE metal `pp512` fell 2.62x -> 1.11x
but a row only closes at <= 1.0x, so it is a much smaller red, not a
green:

| Axis | Red | Worst | Owner |
|---|---|---|---|
| CPU `tg128` | 8 | SmolLM2 2.44× | `cpu-decode-scaling` |
| CPU `pp512` | 6 | Gemma-3-1B 1.65× | `cpu-gemma3-prefill`, then 1a–1d |
| Metal `pp512` | 6 | Gemma-3-1B 1.18× | `metal-fa-mma-d256` (kernel landed 2026-08-18; row awaits a quiet-host suite run) |
| Metal `tg128` | 1 | OLMoE 1.41× | `cpu-decode-scaling`-shaped, GPU side |

**Dense Metal prefill is finished as a workstream.** Every dense row is
1.02–1.08×, and the d=128 kernel moved Qwen3-0.6B by 76% (1936 → 3400
tok/s). What is left on Metal is MoE and one d=256 row.

### What this run corrected

Both corrections are the same failure: reading the old table instead of
measuring both engines together. The plan already forbids this; it still
happened twice in one session.

- **The pre-08-13 llama CPU column was measured under load and reads
  low.** TinyLlama CPU `tg128`: llama 60.64 → 91.74 while frink rose
  55.98 → 61.58 — the row went 1.08× → 1.49× with no frink regression.
  Same on OLMoE CPU `tg128` (llama 65.71 → 107.57). Any "regression"
  spanning that boundary needs re-deriving from same-session numbers.
- **Gemma-3-1B CPU `pp512` is 1.65×, not 1.94×** — llama 548 → 468. It is
  still the worst CPU prefill row, but the gap was never as wide as the
  plan's framing assumed.

One real regression: **OLMoE Metal `pp512` 626 → 587 (-6%)**, gap 2.48× →
2.62×. Inside the ~20% host spread, so not conclusive on its own — but it
is the wrong direction on the worst row in the ledger.

### Coverage findings from the 2026-08-13 external study

Two shipped products were read read-only under `.scratch/` (oMLX,
Unsloth). Neither yields a kernel to port — oMLX's forward pass is
mlx-lm's, and Unsloth does not write GGUF at all (it shells out to
llama.cpp). What they yield is a **compatibility checklist against what
is actually published**, and three items on it are correctness bugs:

- ~~**5 of 16 published `UD-*` variants are undecodable.**~~ **FIXED**
  (`coverage-iq-tiers-published`). ggml tags 17, 21, 22, 29 (`IQ2_XS`,
  `IQ3_S`, `IQ2_S`, `IQ1_M`) hit `GgmlType::Other` in
  `frink-gguf/src/lib.rs`; `IQ3_S` mattered most, since it appears
  inside `IQ3_M` mixes and inside the low-bit recipes `docs/MODELS.md`
  already claims as targets. All four now decode on CPU (scalar only,
  matching the sibling IQ formats). Validated by linking llama.cpp's
  `ggml-quants.c` and asserting bit-exact equality with its own
  `dequantize_row_*` output, not by re-reading the spec. Still
  unvalidated end-to-end on a real published `UD-*` checkpoint.
- **gpt-oss loads and silently computes the wrong graph.** frink
  decodes MXFP4 (tag 39) and routes `gpt-oss` to `generic-gqa`, and
  there is no attention-sink code anywhere in `frink-models` or
  `frink-core`. Unsloth publishes gpt-oss GGUFs as MXFP4-only. So the
  file loads, runs fast, and is wrong — the `coverage-fail-closed` bug
  class with a widely-distributed model behind it.
- **Stop-token and BOS handling is model-specific in ways frink does
  not encode.** gpt-oss `<|end|>` ends every non-final turn and is *not*
  EOG (treating it as one truncates every reply); gemma-4's EOS is
  `<turn|>`; `tokenizer.ggml.token_type == CONTROL(3)` is the authority
  for parseable specials; and Unsloth deliberately strips
  `{{ bos_token }}` from the template it bakes into the GGUF, so a
  loader that renders the template *and* auto-adds BOS double-BOSes.

One structural gap behind all of it: `chat_template.rs` is a six-variant
sniffed enum with hand-written renderers. Every new family falls back to
`Plain`, and the tool-call formats are unreachable without a real Jinja
renderer. `ChatTemplate::Gemma4` was checked and does not implement the
real gemma-4 template.

Nothing here moves a benchmark row. All of it decides whether a model
that loads produces the right tokens.

## MoE prefill stack (2026-08-14, commit ee35372)

OLMoE's MoE layers could not join `launch_prefill_dense_stack`, so each
layer paid host QKV/O projections, one command buffer for attention, a
host round-trip to route on the CPU, and a second command buffer for
`launch_moe_prefill_q4_0`. `PrefillDenseLayerMetal` now carries a
`PrefillFfnMetal` enum (`Dense` | `Moe`); the MoE arm encodes router
GEMM → top-k softmax → `mul_mm_id_map0` → indexed gate/up → SiLU mul →
indexed down → weighted sum into the *same* encoder, so a MoE layer adds
no command buffer at all.

Two kernels were missing and are new: `moe_router_mm_f32` (every MoE
GGUF ships `ffn_gate_inp` as F32 — no `mul_mm_sg` variant covers F32)
and `moe_topk_softmax_batch` (one simdgroup per token; the existing
kernel was single-token, dispatched 1×1). `encode_moe_mm_id_map0` and
`encode_mul_mm_id_f16` had been written ahead of this work and sat
`#[allow(dead_code)]`; they are now on the hot path.

Interleaved A/B, `-p 512 -n 0 -r 2 --ngl 99`, three pairs, host load ~6:

| `FRINK_METAL_MOE_STACK` | OLMoE metal pp512 |
|---|---|
| `1` | 1412 / 1398 / 1417 tok/s |
| `0` | 711 / 724 / 715 tok/s |

**1.98×.** `tg128` is unchanged (112–115 both arms): decode never took
this path, so the Metal `tg128` red row is *not* closed by this change.
`frink verify --backend metal --prompt-tokens 64` is identical cpu vs
metal with the stack on and off.

Owed: a quiet-host suite run (`suite-owed-moe-stack`). Published
`RESULTS.md` still says 2.62×.

One thing this cost: prefill routes on the GPU now, so it no longer
feeds `record_activations`. Expert hotness for `inspect-plan` comes from
decode only.

## MoE decode: where the 1.41× actually goes (2026-08-18)

The prefill stack's A/B already proved decode does not take that path.
What decode *does* take is `Decoder::forward_token` →
`launch_moe_decode_stack` (`crates/frink-models/src/decoder.rs:1644`),
and none of the guessed shapes were true:

- Routing is **not** on the host. `encode_moe_layer_fused`
  (`crates/frink-metal/src/attn.rs:5415`) encodes router GEMM → top-k →
  experts on the GPU.
- There is **one command buffer for the whole token**, not one per layer:
  `launch_moe_decode_stack` opens a single Concurrent encoder for all 16
  layers (`crates/frink-metal/src/attn.rs:5975`).
- The expert mat-vec is **not** a per-expert loop and does have a
  simdgroup variant: `q4_0_moe_matvec_id`
  (`crates/frink-metal/src/gpu.rs:714`) is a faithful port of llama's
  `mul_vec_q_n_f32_impl` (`NR0=4`, `NSG=2`, grid `(rows/8, 1, n_slots)`),
  identical to `kernel_mul_mv_id_q4_0_f32`.

### Method

`FRINK_METAL_GPU_TIMING=1` gives GPU ms/tok for the one command buffer.
A new `FRINK_METAL_MOE_ABLATE` (`crates/frink-metal/src/attn.rs`,
`MoeAblate`) drops whole stages from the encoded graph — output is
garbage, timing is not — so each stage's share is read off by
subtraction. This is the only way to attribute time inside a single
Concurrent encoder short of a Metal capture.

`frink bench -m models/olmoe-1b-7b-0924-q4_0.gguf -p 0 -n 128 -r 1
--n-gpu-layers 99`, Host B, **host load 8–200 (other agents building)**,
so treat every number as PROVISIONAL and the *shares* as the finding:

| dropped stage | GPU ms/tok | Δ vs baseline |
|---|---|---|
| — (baseline) | 7.7–8.6 | — |
| `topk` (`moe_topk_softmax`) | 6.3–6.6 | **−1.3 to −2.2** |
| `rmv` (F32 router mat-vec) | 6.7–6.8 | **−1.1 to −1.8** |
| `router` (both) | 4.9 | **−3.6** |
| `attn` (QKV+RoPE+KV+GQA+O) | 6.3–6.5 | −1.6 to −2.0 |
| `ffn` (gate/up/silu/down/sum) | 5.1–5.6 | −2.8 to −2.9 |
| `gateup` only | 6.8 | −1.7 |
| `down` only | 7.4 | −1.1 |
| everything | 0.64 | — |

**Measured cause: the router, not the experts.** Selecting 8 of 64
experts costs ~3.6 ms of ~8.5 ms GPU (≈ 42 %) — *more* than reading the
50 M active expert weights it selects (2.9 ms). Two single-lane kernels:

1. **`f32_matvec` (`crates/frink-metal/src/gpu.rs:324`)** is one thread
   per output row, walking `cols` serially, dispatched
   `rows.div_ceil(64)` threadgroups of 64
   (`crates/frink-metal/src/gpu.rs:7748`). `ffn_gate_inp` is
   `64 × 2048` F32, so the whole router GEMM is **one threadgroup on one
   GPU core**, with each lane striding a different 8 KB row — no
   coalescing, no simdgroup reduction. llama runs the same tensor through
   `kernel_mul_mv_f32_f32_4` (`nsg=min(4,(ne00+127)/128)=4`, `nr0=2`,
   float4 loads, `helper_mv_reduce_and_write`) — 32 threadgroups × 128
   threads. ~1.1–1.8 ms/tok.
2. **`moe_topk_softmax` (`crates/frink-metal/src/gpu.rs:665`)** opens
   with `if (tid != 0u) return;`, is dispatched `1×1×1`
   (`crates/frink-metal/src/gpu.rs:5948`), holds `float probs[256]` in
   *private* address space (spills), and selection-sorts `k·n = 512`
   comparisons on that one lane — 16 times per token, each behind a
   barrier with the rest of the GPU idle. ~1.3–2.2 ms/tok. The batched
   simdgroup version `moe_topk_softmax_batch`
   (`crates/frink-metal/src/gpu.rs:1175`) already exists — the prefill
   stack added it — and decode simply never calls it.

Secondary, not MoE-specific: wall 9.4 ms/tok vs GPU 7.75 ms/tok. The
~1.6 ms host remainder is `lm_head` (50304 × 2048 Q4_0, ~58 MB) run on
the **CPU** every step, because `out_launch` is `None` unless
`metal_greedy_argmax_active()` (server-only thread-local) or
`FRINK_METAL_LOGITS` is set (`crates/frink-models/src/decoder.rs:1601`).
`FRINK_METAL_LOGITS=1` is worse (91 vs 106 tok/s) because it downloads
the full vocab; the fix is a GPU argmax path that `bench` can take, which
is a separate change from this one.

### The fix

Both are ports, not inventions:

1. `f32_matvec` is now `kernel_mul_mv_t_t_impl<float,float>`: `NR0 = 2`
   rows per threadgroup, `NSG = min(4, ceil(cols/128))` simdgroups
   splitting the reduction axis, `simd_sum` then a threadgroup fold
   (`helper_mv_reduce_and_write`). Consecutive lanes read consecutive
   columns. Dispatch goes from `ceil(rows/64)` threadgroups of 64 (one
   thread per row) to `ceil(rows/2)` of `32*nsg` — for the router,
   1 threadgroup → 32.
2. Decode's top-k now calls the existing simdgroup
   `moe_topk_softmax_batch` with `n_tokens = 1`. The single-lane
   `moe_topk_softmax` kernel and its encoder are deleted — that was its
   only caller.

Interleaved A/B, `-p 0 -n 128 -r 2 --ngl 99`, three pairs, **host load
121–162** — so wall `tok/s` is unusable (± 20 on some reps) and the GPU
clock from `FRINK_METAL_GPU_TIMING` is the signal:

| | GPU ms/tok |
|---|---|
| before | 8.36 / 8.12 / 8.59 |
| after | 6.23 / 6.06 / 6.23 |

**−2.1 ms/tok, −25 % of GPU time**, PROVISIONAL. A second three-pair
A/B once the host dropped to **load 14–16** — still not quiet, still
PROVISIONAL, but with usable wall clock:

| | GPU ms/tok | wall tok/s |
|---|---|---|
| before | 7.76 / 7.99 / 7.61 | 97.9 / 90.7 / 113.4 |
| after | 5.55 / 5.32 / 5.28 | 118.9 / 152.3 / 145.7 |

Median 113.4 → 145.7 tok/s. Against the llama 164.23 in the published
row that would be **1.41× → ~1.13×**, but that is a cross-session
comparison against a number measured on a quiet host, so it is not a
result — it is the reason to go take the suite run.

Re-ablating the fixed
build: router 3.6 ms → ~0.5 ms (`topk` ~0.06, `rmv` ~0.4). What is left
is attention ~2.4 ms and experts ~2.4 ms; at ~483 MB of active expert
weights the expert mat-vec is already running at roughly M2 Pro's peak
bandwidth, so **attention is now the slack** — 151 MB in 2.4 ms is
~63 GB/s, and the likely cause is the ~8 barrier points per layer
(`metal-barrier-ranges`), not the `q4_0_matvec` kernel.

`frink verify --backend metal --prompt-tokens 64` on OLMoE: 24 tokens
identical cpu vs metal.

Owed: a quiet-host suite run. `RESULTS.md` still says 1.41× and must not
be edited until then. Not fixed here, and each worth ~1 ms/tok:
`metal-barrier-ranges` on the decode attention chain, and the host-side
`lm_head`.

### Next levers, in order

1. `cpu-decode-scaling` — 8 red rows, the only axis with nothing at
   parity, and the cause is already measured (fork-join scaling; frink
   beats llama at one thread on Mistral-7B). Retry the persistent pool
   with the `wf/cpu-threadpool` deadlock understood first.
2. `cpu-gemma3-prefill` — the one CPU prefill row that is an outlier
   rather than a trend. Diagnosed 2026-08-18, see §1h: 19.6% in the
   windowed (SWA) prefill attention arm that never reaches the blocked
   kernel, 10.7% in libm `tanhf` under GeGLU.
3. `metal-fa-mma-d256` and the OLMoE Metal `tg128` row (1.41×), which
   the prefill stack did not touch.

## Historical ledger snapshots

Removed. Four point-in-time copies of the results table lived here (v0.4.0,
the MMA port, the d=128 run, and a ranking the file itself labelled
superseded). `benchmarks/RESULTS.md` is generated from the receipts and is
the only table worth reading; keeping frozen copies beside it meant five
answers to one question, four of them wrong. Receipts under
`benchmarks/receipts/engine/` carry the version each row was measured on.

## d=256 MMA (2026-08-18, branch `wf/metal-fa-mma-d256`)

**Head dim confirmed first.** `frink inspect` on
`models/hf_test/gemma-3-1b-it-Q8_0.gguf`: `attn_q [1152, 1024]` = 4 heads
x 256, `attn_k`/`attn_v` `[1152, 256]` = 1 kv head x 256,
`attn_q_norm`/`attn_k_norm` `[256]`; `inspect-plan` prints
`1 kv-heads x 256 head-dim`. The 1.18x row really is a width the macro
could not instantiate, so the kernel work was warranted.

**What the cap actually was.** Three places touch the `so` accumulator a
row at a time — zero-init, the online-softmax rescale, and the epilogue —
and all three were written as `if (own) so4[tiisg] ...` with
`own = tiisg < D4`. That assigns one `float4` per lane, hence `D/4 <= 32`.
llama has no such cap (`kernel_flash_attn_ext_impl` reaches DV 576): it
walks the same three points with `for (i = tiisg; i < DV4; i += NW)`
(`ggml-metal.metal:6529-6535`, `:6826-6838`, `:7024-7034`). Ported that
loop verbatim in shape. Below `D4 <= NW` it runs exactly one iteration on
exactly the lanes `own` selected, at the same index, so d=64 and d=128
compute the same values; at d=256 (`D4 == 64`) one lane carries two
`float4` columns.

Then `gqa_prefill_fa_ext_mma_d256` is instantiated from the same macro and
head_dim 256 routes to it. At QN=8 / C=64 it needs 28 KiB of threadgroup
memory — under Apple's 32 KiB, but the last width that fits at this
tiling. As at d=128 there is no scalar `fa_ext` here, so
`FRINK_METAL_FA_MMA=0` sends d=256 back to `gqa_prefill_fa_vec_d256`,
which is therefore the A/B reference for both correctness and timing.
Sliding-window layers are untouched: prefill has no `kv_start` at all, and
`metal_prefill_dense_swa_fits` (`decoder.rs:833-843`) already keeps a
windowed layer off the GPU prefill path unless the whole batch fits inside
the window.

**PROVISIONAL A/B — host load 29-35 (1-min), other agents building
concurrently. Not publishable.** Interleaved, `-p 512 -n 0 -r 2 --ngl 99`,
3 pairs, Gemma-3-1B-IT Q8_0 metal `pp512`:

| arm | rep 1 | rep 2 | rep 3 | mean |
|---|---|---|---|---|
| `FRINK_METAL_FA_MMA=1` (new) | 2544.91 | 2564.22 | 2514.59 | **2541** |
| `FRINK_METAL_FA_MMA=0` (fa_vec) | 2245.59 | 2303.43 | 2201.85 | 2250 |

= **1.13x** over the kernel this replaces.

**Guard A/B for the landed widths.** The lane loop touches the d=64 and
d=128 kernels' source, so the pre-change binary was rebuilt from
`b09aedb` and interleaved against the new one in the same window
(load 29-30), 3 pairs each:

| Model | head dim | base mean | new mean |
|---|---|---|---|
| Qwen3-0.6B Q8_0 | 128 | 3226 | 3320 |
| Llama-3.2-1B-Instruct Q8_0 | 64 | 1907 | 1936 |

Both inside run-to-run spread; no regression in either direction.

**Correctness.** `cargo test -p frink-metal --features metal -- --ignored
--test-threads=1`: 45 passed, 0 failed, including the new
`gqa_prefill_fa_ext_mma_d256_matches_cpu_and_fa_vec` (Gemma-3-1B's own
4-head / 1-kv-head / softcap shape, padded cache tails, exact 8-row fits,
long-prefix / short-batch) and the unchanged d=64 / d=128 tests.
`frink verify --backend metal` reports identical cpu-vs-metal ids **with
prefill covered** on Gemma-3-1B at 64 and 300 tokens with MMA on *and*
off, and on Qwen3-0.6B and Llama-3.2-1B at 300.

**Owed: a suite run on a quiet host.** `RESULTS.md` was deliberately not
touched and still advertises Gemma-3-1B metal `pp512` 2363.87 against
llama 2786.02 = 1.18x. The A/B above is a relative measurement only, and
the llama column must be re-measured in the same window as frink before
any new gap is written down.

### Rejected: the persistent-threadpool branch

`wf/cpu-threadpool` was implemented and **not merged**. Adversarial review
found a reproducible deadlock in its new public seam, contradicting the
module's own safety argument; its A/B knob did not isolate the change it
existed to measure; and its perf thesis was unverified by the author's own
admission. The diagnosis it rests on is still correct (scaling, not
throughput). Retry with the deadlock understood first — reason explicitly
about `FRINK_CPU_THREADS=1` and rayon nesting before writing code.

**Deadlock reproduced and diagnosed (2026-08-14).** Built the branch and
ran a probe that submits pool regions from Rayon workers whose *tasks*
call back into Rayon. It hangs, and `sample` gives the whole cycle:

| threads | where they are |
|---|---|
| `frink-pool-0..4` | `run_tasks → trampoline → rayon bridge → in_worker → LockLatch::wait_and_reset` |
| Rayon workers | `par_chunks_indexed → CpuPool::dispatch → Mutex::lock` (the submit lock) |
| submitter | holds the submit lock, spinning for `done == n_workers` |

pool worker → Rayon worker → submit lock → submitter → pool worker.

So it is **not** a memory-ordering bug, and the module's ordering
argument was never the problem. It is coexistence: two runtimes with a
*blocking* lock between them, and a task body that can reach the other
runtime. `FRINK_CPU_THREADS=1` is not needed to trigger it (a
single-thread OLMoE decode on the branch completes fine); Rayon nesting
alone is sufficient, and MoE decode nests by construction
(`outs.par_iter_mut()` over experts, `rayon::join` in `run_expert`,
`apply_three` for q/k/v).

Rules the retry must satisfy, each one aimed at a specific edge of that
cycle:

1. **Never block on the pool.** `dispatch` takes the submit lock with
   `try_lock`; a losing caller runs the Rayon path instead. That deletes
   the `Rayon worker → submit lock` edge, which is the only edge that
   needs another runtime to make progress.
2. **Pool tasks are leaf kernels.** No task body may enter Rayon. That
   deletes the `pool worker → Rayon worker` edge.
3. **Flatten instead of nest.** The three sites that fan out with Rayon
   during decode (`apply_three`, `run_expert`, the MoE `par_iter_mut`)
   become *one* pool region over a fused task list, issued by the decode
   thread — which is also the actual win, since it cuts regions per
   layer rather than making each one cheaper.
4. **The probe ships as a test.** The hang above is a two-runtime
   regression test; it belongs in-tree so this cannot be re-landed
   silently.

### Gaps in our own tooling, found by using it

- ~~**`frink verify` passes vacuously on prefill kernels.**~~ **FIXED
  (`bfd1c1a`).** The prompt was fixed at 6 tokens while every batched
  prefill kernel gates on `n_q >= 8`. `--prompt-tokens N` stretches the
  prompt by repeating its body (one BOS kept), `--prompt` overrides the
  text, and the child reports the tokenized length back so every verdict
  ends with `prefill covered` or `decode only`. A vacuous pass is now
  visibly labelled as one.
- ~~**CPU and Metal diverge on longer prompts.**~~ **Did not reproduce
  (2026-08-13).** With the length-aware `verify`, greedy ids are identical
  cpu vs metal at 41 (natural text), 49, 128 and 300 tokens on TinyLlama,
  at 49 and 300 on Phi-4-mini, and at 40 tokens across TinyLlama,
  Phi-4-mini, Llama-3.2-1B (Q4_K_M and IQ4_XS), Llama-3.2-3B, Mistral-7B,
  OLMoE and Gemma-2-2B — the first real-weight coverage of both MMA
  kernels. Either the MMA work fixed it or the original observation was
  logit drift that never flipped an argmax. Not claimed as a fix; reopen
  with a reproducer.

## What a survey of two peer Rust engines actually yielded

Both were read at source level, not from their READMEs. The negatives
matter more than the positives, because they close off shortcuts:

- **Neither has a simdgroup-MMA Metal attention or GEMM to port.** Both
  ship scalar "one thread per output cell" Metal kernels whose own header
  comments describe the tiled MMA version as future work — exactly where
  frink is. 2a must be written from llama.cpp's `kernel_flash_attn_ext`.
- **Neither helps with CPU decode scaling.** One is GPU-only end to end
  (no CPU backend implementor at all, only a validation oracle); the
  other's kernels are CUDA FFI. There is no peer persistent-threadpool
  design to study; llama.cpp's `ggml_barrier` remains the reference.
- **One's "paged attention" is orchestration only** — the gather kernel
  lives in an unvendored external crate. Portable part is the *contract*,
  not an implementation.
- Its scheduler is FIFO with alternating prefill-only / decode-only steps,
  not true single-step interleaving. Simpler than advertised, and a
  reasonable first target, but it carries no fairness policy to inherit.

Portable and worth taking, in value order:

1. **Sealed kernel-lookup registry** (see below) — smallest, highest hit
   rate against bugs frink actually had.
2. **Per-head magnitude-ratio std** as the divergence fingerprint (below).
3. **Paged-KV metadata contract**: a flat block pool with a free-list, a
   per-sequence `block_table: Vec<u32>`, and a `slot_mapping` array of
   physical slots (`block_id * block_size + offset`) built per step, over
   **one pre-sized allocation per layer** rather than OS paging. Page size
   32 on Metal / 64 on CUDA. This is the shape that decouples the KV
   budget from context reserved up front, and it needs no virtual-memory
   tricks. frink would still write the gather itself.
4. **Asymmetric K/V precision.** K dominates attention-score accuracy and
   V tolerates far less; pairing a higher-precision K with a lower-
   precision V (per-layer overridable) buys memory that uniform KV quant
   cannot. Pairs with the low-bit KV and rotation work on the roadmap.
5. **Decode sparse-V gate**: skip V dequant+accumulate entirely where the
   softmax weight is below ~1e-3. Bandwidth win at long context,
   independent of any MMA work.

frink already has a prefix cache; the peer implementation (parent-hash
chained block hashing, ref-counted LRU eviction restricted to leaf blocks)
is worth diffing against ours rather than adopting wholesale.

## Debuggability: per-layer divergence, not just per-output

`frink verify` (landed) compares greedy token ids between the CPU
reference and a GPU backend. It answers *whether* a backend is wrong. It
does not answer *where*, and that is most of the work: the d=64 softcap
race took a full investigation to localise even though the symptom was
obvious, because the only observable was the final logits.

**Adopt a layer-by-layer divergence comparator, environment-gated so it
costs nothing when off.** The shape that works elsewhere:

- A reference forward in f32 with per-layer hooks, run against the
  backend under test.
- **Per-head magnitude ratio, and watch its standard deviation, not its
  mean.** A mean near 1 with high across-head variance is the fingerprint
  of a pointer/layout bug; precision noise keeps variance low. That single
  distinction separates "wrong indexing" from "expected f16 drift"
  immediately — the call that cost the most time on the d=64 kernel.
  Companions: flat cosine, flat relative-L2, and a best-match-permutation
  cosine, which catches heads computed correctly but written to the wrong
  slot.
- Per layer, per tensor of interest (prenorm hidden state, attention
  output, FFN output, residual sum): magnitude ratio and top-K overlap
  against the reference, not just max-abs-diff — a ratio catches a
  scale bug that an absolute threshold sized for the output tensor
  misses at layer 2.
- Print the first layer whose ratio leaves a band, then stop. That index
  is the diagnosis.
- Gate every dump behind an env var checked once, so the instrumentation
  is not in the hot path when unset (the same `OnceLock` discipline the
  Metal env reads now use).

Also worth taking: **MoE routing dumps** — gate logits, chosen expert ids
and per-expert output magnitudes per layer. MoE is frink's worst family
on every axis, and today there is no way to see whether a bad row is a
routing bug or a kernel bug. The Qwen shared-expert bug (40% of the
active FFN silently skipped) is exactly the class of defect this makes
visible immediately, and it went unnoticed until a code audit found it.

Neither is a performance change. Both are prerequisites for doing the
performance changes quickly and for not shipping another green row that
computes the wrong thing.

## Silent-fallback detection (cheap; we keep hitting this)

frink's worst bug class this cycle was not a wrong kernel but a *missing*
one, silently replaced by a slow path:

- IQ4_XS batched prefill ran on the **CPU** because `metal_kind_supported`
  and `apply_gpu`'s kind table disagreed — 13.7x on that row, and the only
  symptom was a slow benchmark.
- Gemma-4-E2B is slower on Metal than on CPU: same smell.
- `frink-cli --features cuda` did not compile at all, so every CUDA claim
  was untested.

**LANDED (`99a69ab`, `frink-core/src/kernel_registry.rs`).** The design,
kept here because the remaining work is to widen its coverage as new
dispatch seams are added — an unrecorded seam is invisible to it. Record
every kernel/dispatch lookup made while constructing the model, with
`#[track_caller]` so each record carries its call site. Seal the registry
once loaded; any later lookup that misses and takes a fallback is a silent
slow path and should fail loudly, or at least warn once. That converts
"why is this row slow" into a startup error. It would have caught all
three of the above at load time instead of via a benchmark and an audit.

## Memory: what actually decides which models fit

Roadmap item 1 ("run bigger models on the same hardware") has no concrete
mechanism attached beyond the existing residency plan. Two that do:

- **Per-layer precision mixing.** Keep boundary layers (embedding-adjacent
  and final) at higher precision and quantize the middle harder, rather
  than applying one format uniformly. This changes which models fit
  without the quality cliff of dropping the whole model a tier — the
  usual argument for it is that first and last layers are the most
  sensitive to quantization error.
- **Paged / block-sparse KV with page-index prefetch.** frink's KV is
  contiguous per sequence. Paging it decouples the KV budget from the
  context length reserved up front, which is the thing that currently
  forces a smaller context or a smaller model.

Both are large. They are recorded here because item 1 currently names a
goal without naming a mechanism, and these are the two mechanisms.

## Status of the previous plan

Phases 0–3 of the prior plan are **done** and their rows moved: Metal dense
prefill went from ~18–21× to ~1.2–2.1×, Metal decode is at or below parity on
most dense models, MoE loader/shexp bugs are fixed, SmolLM2 Metal greedy
lm_head is fixed, CPU MoE bucketing landed.

What did **not** work as predicted, and why the plan is re-ranked:

| Prior belief | Reality (2026-08-10 audit) |
|---|---|
| CPU Q4_K GEMM would give 5.82× → 3.2×; it gave +6%, so "loop structure is not the problem, the arithmetic tier is" | Half right. The i8mm GEMM **is** live and reachable. It is fronted by a scalar activation re-interleave that costs ~4× the GEMM it feeds. Loop structure *was* the problem, one level down. |
| Sub-1.5B Metal pp512 needs a compiled graph / pre-encoded command buffers | False. Frink already issues **1 command buffer per prefill graph**; llama issues 2. The gap is a scalar attention kernel with 16 of 128 lanes active. |
| Upstream has no i8mm `gemm_q5_K_8x8` / `gemm_q6_K_8x8` ("until it lands") | Both exist upstream and are selected on any NEON+i8mm host. The in-tree comments are wrong and should be deleted with the fix. |
| Arch coverage is a list of missing loaders | Worse: the name registry is complete, so ~50 archs **load and produce wrong logits** instead of refusing. |

## Defaults

- **Backend order:** CPU prefill first (5–11× gaps, largest in the ledger and
  the cause is now known), Metal attention second, Metal MoE third, coverage
  and correctness in parallel (they are cheap and independent).
- **Iteration style:** CLI-bench the largest-gap model, change one lever,
  re-bench the same model. That is the *inner* loop for deciding whether to
  keep a change — it is not the validation.
- **Validation is the full suite, every time.** Once a change is kept, run
  `bench --suite --fit-host --skip-missing` + `--render` and diff the whole
  ledger against the previous one. A kernel that speeds up its target model
  and quietly costs 15% somewhere else is a regression, and the only way to
  see it is to measure every row. No change is considered landed on a
  partial measurement.
- **Bench load:** Never run frink and llama benches in parallel, and never
  while subagents or builds are running. Check `uptime` before quoting
  numbers; abort above ~2.0. Prefer `frink bench … --compare` (sequential in
  one process).
- **Success bar:** gap ≤ 1.0× on every engine suite row computing the right
  model. Gap = `llama / frink`. Within ~5% counts as closed; run-to-run
  spread on Host B is ~20%, so claims tighter than that need interleaved A/B.
- **Quality:** no speed win without golden/kernel tests and answer parity.
- **Coverage:** an arch that computes the wrong graph is a bug, not a gap. It
  must refuse to load until implemented.
- **Port from llama.cpp:** read the reference under `.scratch/llama.cpp`
  first, then port to Rust (and MSL strings in `frink-metal`). Cite the
  llama file/symbol in the commit body.

## Measurement contract (non-negotiable)

```bash
cargo build --release -p frink-cli --features metal

# INNER LOOP (deciding whether a change is worth keeping):
# Never -t; ngl 0 (CPU) or 99 (Metal)
./target/release/frink bench -m <gguf> -p 512 -n 128 -r 3 --n-gpu-layers 0 --compare
# Prefill-only for faster iteration when decode is already healthy:
./target/release/frink bench -m <gguf> -p 512 -n 0 -r 3 --n-gpu-layers 0 --compare

# VALIDATION (mandatory after every kept change, before it counts as landed):
uptime                                   # abort above ~2.0 load
./target/release/frink bench --suite --fit-host --skip-missing
./target/release/frink bench --render
git diff benchmarks/RESULTS.md            # read EVERY row, not just the target
```

The suite is the unit of truth. A change that improves its target row and
regresses another has not made the engine faster — it has moved the gap. The
`--render` diff is the artifact that proves otherwise, and it belongs in the
commit that made the change.

**Regenerating the ledger is not optional and is never deferred.** It is part
of landing a change, not a follow-up to schedule or a question to ask. Two ways
this has actually gone wrong:

- Phase 1 (PRs #2–#8) landed eight kernel changes on x86, where every
  aarch64-gated kernel is compiled out, so none of them could be measured and
  none of them were. `RESULTS.md` then advertised CPU `pp512` at 3.2–5.8×
  behind for days after the real figure had become 0.83–1.87×. Work that
  cannot be measured on the host that wrote it is not landed; it is staged.
- Spot-checking new frink numbers against the *stale* llama numbers already
  in `RESULTS.md` produced a confident and wrong "frink is ahead" claim. Both
  engines must be measured in the same session. `--compare` does this; reading
  the old table does not.

If the box is too loaded to measure (`uptime` above ~2.0), wait for it. Do not
substitute a spot-check, and do not publish a number taken under load — known-
good rows read 25–45% low, which is larger than most of the gaps being chased.

---

## Phase 1 — CPU prefill (largest gaps in the ledger: 3.5×–11×)

Current CPU pp512 vs tg128 is the tell — batching buys almost nothing:

| Model | frink pp512 | frink tg128 | pp/tg | llama pp/tg | gap |
|---|---|---|---|---|---|
| Phi-4-mini Q4_K_M | 10.50 | 11.03 | **0.95×** | 3.88× | 10.97× |
| Mistral-7B Q4_K_M | 5.39 | 6.50 | **0.83×** | 2.76× | 9.80× |
| SmolLM2-135M Q8_0 | 155.46 | 93.39 | 1.66× | 5.36× | 10.68× |
| Qwen3-0.6B Q8_0 | 80.31 | 52.24 | 1.54× | 4.87× | 6.57× |

Batch 1 and batch 512 run at the same speed, so the batch dimension is being
consumed inside the kernels rather than exploited.

### 1a. Hoist the activation interleave (single largest CPU win)

`gemm_q4_kx8_q8_k_neon_i8mm` (`crates/frink-quant/src/repack.rs:2446`) calls
`pack_q8_k_qs_x4_i8` (`repack.rs:2428`) **inside** its `for b in 0..nb` loop,
at `repack.rs:2463`. That helper is a scalar loop over 1024 elements with a
div and a mod per element. The kernel is invoked once per (row-group,
4-activation tile) from `weight_matrix.rs:1407-1423`, so the *same* activation
bytes are re-interleaved once per weight row-group.

Cost per matmul: `rows·batch·cols/8` scalar gather-stores against
`rows·batch·cols/32` `vmmlaq_s32` for the real math — roughly 4× the
instruction count and far more µops, all scalar. Decode (batch 1) never enters
this path (it uses `dot_q4_k_q8_neon_i8mm`, `lib.rs:852`), which is exactly why
`pp ≈ tg`.

llama does this once: `ggml_quantize_mat_q8_K_4x8` writes the interleaved
`block_q8_Kx4` into `params->wdata` (`ggml-cpu/repack.cpp:4298-4307`), and
`ggml_gemm_q4_K_8x8_q8_K` (`arch/arm/repack.cpp:3752`) reads it directly with
the activation-quad loop **outer** and the weight-group loop **inner**.

**Fix:** make the interleaved layout the storage form. Add
`Q8KActivationsX4 { qs: Vec<i8> /* pre-interleaved */, d, bsums }`, build it
once per `apply_batch`, change the kernel to take `&Q8KActivationsX4`, delete
`pack_q8_k_qs_x4_i8`. Keep a bit-exactness test against the current kernel.

### 1b. Delete the output transpose

Seven copies of the same serial scatter loop convert `[rows,batch]` to
`[batch,rows]`: `weight_matrix.rs:1324-1330`, `1381-1387`, `1451-1457`,
`1538-1544`, `1624-1630`, `1649-1654`, `1681-1687`. Single-threaded,
`O(rows×batch)` with stride-`rows` writes into a 3–16 MB buffer, sitting
downstream of an `O(rows×batch×cols)` parallel GEMM — its Amdahl share grows
as 1a lands.

llama never transposes: `forward_mul_mat_one_chunk`
(`ggml-cpu/repack.cpp:4204-4248`) passes a `dst_ptr` plus row stride and the
kernel stores `s[m*bs + n]` straight into final layout.

**Fix:** give the group kernels `dst: *mut f32` + `dst_row_stride`; drop
`by_row` entirely; make the Rayon unit a `(row-chunk, batch-chunk)` tile
writing disjoint sub-rectangles.

### 1c. i8mm for Q5_K and Q6_K

`q5_kx8_interleave()` (`repack.rs:622`) and `q6_kx8_interleave()`
(`repack.rs:1138`) hard-return 4 on aarch64, so the `interleave == 8` guard
never holds and only `_sdot` kernels can be selected. Q4_K_M puts `attn_v` and
about half of `ffn_down` in Q6_K, so a real slice of the FFN runs at SDOT rate
(16 MAC/instr) instead of SMMLA rate (32 MAC/instr, 2 activation rows free).

Port `ggml_gemm_q5_K_8x8_q8_K` (`arch/arm/repack.cpp:4272`, guard 4293) and
`ggml_gemm_q6_K_8x8_q8_K` (`arch/arm/repack.cpp:4721`, guard 4742). Mirror the
detection `q4_kx8_interleave()` already does correctly at `repack.rs:33-44`.
Delete the two "until i8mm … lands" comments — they are false.

> **Uncommitted work in the tree:** `repack.rs` currently has an unstaged
> Q6_K NEON **sdot** GEMV/GEMM (`gemv_q6_kx8_q8_k_neon_sdot`,
> `gemm_q6_kx8_q8_k_neon_sdot`) plus `weight_matrix.rs:1555` flipping
> `use_kx8` to true on aarch64. `cargo test -p frink-quant` passes (96/96,
> including `q6_kx8_gemm_matches_the_gemv_run_once_per_activation`). Land it
> as the correctness-preserving step, then upgrade it to the 8×8 SMMLA shape
> rather than treating sdot as the destination.

### 1d. i8mm for Q8_0 and Q4_0

`Q8_0X4_INTERLEAVE` (`repack.rs:378`) and `Q4_0X4_INTERLEAVE`
(`repack.rs:1431`) are hardcoded to 4, so no SMMLA path can exist for either.
llama's `ggml_repack_get_optimal_repack_type` (`ggml-cpu/repack.cpp:4699`)
picks `q8_0_4x8_q8_0` whenever `neon && matmul_int8` — i.e. always on M2 —
landing on `ggml_gemm_q8_0_4x8_q8_0` (`arch/arm/repack.cpp:5006`), fed by
`ggml_quantize_mat_q8_0_4x8` (`arch/arm/repack.cpp:119`). Same story for
`ggml_gemm_q4_0_4x8_q8_0` (`arch/arm/repack.cpp:2307`).

This is the structural source of the Q8_0 CPU gaps (SmolLM2 10.68×,
Qwen3-0.6B 6.57×, Qwen2.5-0.5B 5.84×, Gemma-3-1B 5.14×, TinyLlama 3.52×).
Needs the interleaved activation packer from 1a generalized to Q8_0.

### 1e. De-nest activation quantization

`weight_matrix.rs:1269-1276` (and 1333, 1390, 1460, 1547) run
`(0..batch_size).into_par_iter().map(quantize_activations_q8*)`, and those
functions are themselves Rayon regions (`lib.rs:683-690`, `lib.rs:724-728`).
So each of 512 outer tasks opens an inner parallel region. The Q8_0 one splits
on `par_chunks_mut(32)` over `i8` — 32-byte chunks, two per cache line, with
`d.par_iter_mut()` writing adjacent `f32`: guaranteed false sharing on every
store. Per prefill: ~100k nested regions and ~200k heap allocations.

llama quantizes once, thread-split by *column block*, then one barrier
(`ggml-cpu.c:1322-1359`). **Fix:** serial internals, parallelize only at the
`apply_batch` level over row-quads into one contiguous `wdata`-style buffer.
Also cache the quantized activations per `normed_batch` so q/k/v share one
pass and gate/up share one — currently 5 quantizations where 2 suffice.

### 1f. Chunked work-stealing

Frink does a fresh Rayon fork-join per matmul with a static `with_min_len`
from `min_rows_per_task` (`weight_matrix.rs:442-445`) — about 6 tasks for a
192-row `k_proj` on a 10-thread pool. llama chunks over **both** row and batch
dims with an atomic `current_chunk` and `nchunk0·nchunk1 ≥ nth*4`, plus a
re-chunk fallback (`ggml-cpu.c:1391-1430`, `repack.cpp:4355-4382`).
Persistent spin-barrier pool (`ggml-cpu.c:584-606`) only if decode is still
>1.0× after the kernel work.

### 1g. Block the prefill attention

`causal_gqa_attention_prefill_shared_kv` (`attention.rs:570-613`) parallelizes
over `n_q × n_heads`, but each task walks KV one position at a time through
`online_attn_accumulate` (`attention.rs:146-175`): two scalar `f32::exp` calls
and a full-width rescale **per KV position**, with no K-tile reuse across
queries. Port llama's shape — `QKᵀ` as a real GEMM, vectorized softmax over
the whole row, `V·P` as a second GEMM.

### 1h. Gemma-3-1B CPU `pp512` — measured diagnosis (2026-08-18)

`sample` over `frink bench -m models/hf_test/gemma-3-1b-it-Q8_0.gguf -p 512
-n 0 -r 6 --n-gpu-layers 0`, 15 s at 1 ms, on `wf/cpu-prefill-gemma3`. Host
load 16.5 → 13.9 across the run, so the **wall-clock number from that run is
worthless** (212.86 t/s against a published 284.54) — only the symbol shares
below are used. Non-idle top-of-stack total ≈ 36 476 samples
(`__psynch_cvwait` 33 636 and `swtch_pri` 1 648 excluded as idle):

| symbol | samples | share |
|---|---|---|
| `frink_quant::repack::neon::gemm_q8_0x4_q8_0_neon_i8mm` | 20 782 | 57.0% |
| `frink_core::attention::causal_gqa_attention_windowed_softcap` | 7 167 | 19.6% |
| `tanhf` (libsystem_m, + its DYLD stub) | 3 906 | 10.7% |
| everything else | ≈ 4 600 | 12.7% |

**Finding 1 — SWA layers never reach the blocked attention.** The CPU prefill
attention dispatch at `decoder.rs:3741` branches on
`self.config.layer_sliding_window(l)`:

- `None` → `causal_gqa_attention_prefill_shared_kv` (`attention.rs:632`), the
  three-pass blocked form: `Q·Kᵀ` row, one vectorised softmax, then `V·P`,
  Rayon over `(query-block of 8) × head`.
- `Some(window)` → a per-query-row loop calling
  `causal_gqa_attention_windowed_softcap` (`attention.rs:546`), which is
  `online_attn_accumulate` (`attention.rs:146`): **two scalar `expf` and a
  full `head_dim`-wide rescale of the accumulator per KV position**, with
  Rayon over the 512 query rows only — the head axis stays serial inside each
  task.

Gemma-3 is `swa_pattern = 6`, so `layer_sliding_window` (`config.rs:328`)
returns `Some(512)` for 22 of its 26 layers. TinyLlama (the 1.01× control)
and SmolLM2 both have no `sliding_window` at all, so neither ever enters this
arm. That is the whole reason Gemma-3 is an outlier rather than part of the
trend.

The work is not even different: at `pp512` with `window = 512`,
`window_start = seq_len.saturating_sub(512)` is 0 for every position, so the
windowed arm visits exactly the same KV range as the full-causal arm. It is
19.6% of the process spent on a slower way of computing the same thing.

**Finding 2 — GeGLU pays a libm `tanhf` per element.** The `tanhf` samples
are entered from `dense_ffn_batch` → `frink_core::matmul::geglu`
(`matmul.rs:233`), which maps `gelu()` (`matmul.rs:225`) elementwise; `gelu`
is the tanh approximation and calls `f32::tanh`, i.e. `libsystem_m`'s scalar
`tanhf`, once per FFN element. Gemma-3 runs `26 × 512 × 6912 ≈ 92 M` of them
per `pp512`. TinyLlama and SmolLM2 are SwiGLU and never call it. llama.cpp
does not pay this either: `ggml_vec_geglu_f32`
(`ggml/src/ggml-cpu/vec.h:1414`) under `GGML_GELU_FP16` reads a 65 536-entry
`ggml_table_gelu_f16` lookup instead of evaluating `tanhf`.

**Not a finding — the Q8_0 GEMM tier.** `gemm_q8_0x4_q8_0_neon_i8mm` is the
top symbol, i.e. the SMMLA Q8_0 kernel `cpu-i8mm-q8_0-q4_0` proposes to write
already exists and is already on the hot path. That todo is stale for Q8_0.

**SmolLM2-135M is a different problem.** A `sample` of its `pp512` (same
session, host load 41 — timing junk, shares still usable) puts the inlined
per-task closure of `causal_gqa_attention_prefill_shared_kv` at 8 265 / 15 700
= 53% of non-idle against 30% for the Q8_0 GEMM, while attention is only ~8%
of the model's FLOPs. So the *blocked* form is itself ~12× off the GEMM's
efficiency: pass 1 is a `dot_f32` per `(query, KV position)` at `head_dim`
64, with no K reuse across the 8-query block. Fixing Gemma-3 does not fix
SmolLM2; SmolLM2 needs the real K-tile × Q-tile GEMM under
`cpu-prefill-attn-block`.

### CPU order and expected movement

1. **1a** — the dominant term on every Q4_K row. Re-bench Phi-4 immediately.
2. **1b** — compounds with 1a; grows in relative weight as 1a lands.
3. **1d** — unlocks the five Q8_0 rows, which are the widest gaps after Phi-4.
4. **1c** — finishes Q4_K_M's FFN tail.
5. **1e**, **1f** — scheduling overhead, worth re-measuring after 1a–1d.
6. **1g** — last; ~5% and shared with decode.

---

## Phase 2 — Metal prefill attention (the real sub-1.5B lever)

**The prior "needs a compiled graph" hypothesis is dead.** Frink's
`forward_hidden_batch` already finds the maximal run of consecutive dense
layers (`decoder.rs:746-785`) and hands it to `launch_prefill_dense_stack`
(`decoder.rs:842`), which uses **one** command buffer, one concurrent encoder,
one commit, one `waitUntilCompleted` for all layers
(`frink-metal/src/attn.rs:6925-6953`). llama uses 2 CBs per graph
(`ggml-metal-context.m:463-466`). Frink is ahead here; graph pre-encoding is
now the *last* item, not a prerequisite.

### 2a-0. First: `gqa_prefill_fa_ext_d64` is red (blocking 2a)

Two `--ignored` GPU tests fail at the v0.4.0 tag, and have been failing
unnoticed because `cargo test --workspace` does not run ignored tests:

```
gqa_prefill_fa_ext_d64_matches_cpu
  hd=64 n_q=40 pre=9 sc=Some(50.0) max_diff=0.286
  worst=(542, 0.37141415, 0.085109815) tol=0.005
gqa_prefill_fa_ext_matches_fa_vec_d64
  fa_ext vs fa_vec max_diff=0.286  (same element, same values)
```

0.371 vs 0.085 is a factor of 4.4 on one output element — wrong attention,
not rounding, and the two tests agree on which element. The failing shape is
d=64 **with softcap and a nonzero KV prefix**; no model in the suite hits it
(Gemma uses softcap but d=256), and all three d=64 suite models — SmolLM2,
TinyLlama, Llama-3.2-1B — generate correct text on Metal. So no published row
is invalid. But the kernel is default-on for d=64 / n_q>=8, so the next d=64
model with softcap would be silently wrong.

Fix this before rewriting the kernel for MMA. Rewriting on top of a red test
means never knowing which failure the rewrite introduced.

**Also add `--ignored` GPU tests to a checked runbook.** They are the only
tests that exercise Metal at all, and nothing in the normal test command runs
them:

```bash
cargo test -p frink-metal --features metal -- --ignored --test-threads=1
```

(`--test-threads=1` matters: the resident weight cache is keyed by
`(pointer, length)` plus a content fingerprint, and concurrent fixtures of
equal size do collide on address reuse.)

### 2a. Port `kernel_flash_attn_ext` MMA (blocking for sub-1.5B ≤1×)

`grep -c "simdgroup_multiply_accumulate\|simdgroup_half8x8" attn.rs` → **0**.

**Measured note on the score phase:** naively widening it from
`if (sgitg == 0u)` to all four simdgroups (partitioning keys by `cc`, whose
`ss` columns are disjoint) does **not** work — it was tried and the d=64
tests moved. The single-simdgroup guard is load-bearing for a reason its
comment does not state; find that reason before restructuring.
Every prefill attention kernel is a vector/`simd_sum` kernel. In the default
d=64 path `gqa_prefill_fa_ext_d64` (default-on for d=64, n_q≥8 —
`attn.rs:3616-3650`), despite the name:

- Q·Kᵀ runs in **one simdgroup only** (`if (sgitg == 0u)`, `attn.rs:1718`) —
  3 of 4 simdgroups idle for the whole score phase;
- inside it only 16 of 32 lanes participate (`own = tiisg < 16`,
  `attn.rs:1675, 1724`) → **16/128 = 12.5% lane utilization**;
- it is a scalar `dot` + `simd_sum` per (query, key) pair — 512 shuffle
  reductions per 64-key chunk per (head, query-block), `attn.rs:1719-1727`;
- P·V is a scalar FMA gather, per the code's own comment at
  `attn.rs:1774-1786`: *"P·V: scalar gather (V staging + MMA layout still
  WIP)"*.

llama uses real 8×8 MMA for both Q·Kᵀ and P·V (`ggml-metal.metal:6701,
6720-6721, 6878-6879, 6901-6904`; template 7069, d=64 instantiation 7126).

This explains the size scaling exactly: attention is ~8% of SmolLM2-135M's
prefill FLOPs (h=576) but ~1% of Mistral-7B's (h=4096), and the measured gap
runs 2.98× → 1.84× → 1.62× → 1.29× → 1.21× monotonically with hidden size.
An in-tree profile already agrees: *"62% of prefill inside that legacy
kernel's waitUntilCompleted"* (`attn.rs:1266-1268`).

**Expected:** SmolLM2/TinyLlama/Qwen2.5/Llama-3.2-1B pp512 2.98× → ~1.4×;
~1.15× on 3B+.

### 2b. Barrier ranges + fusion

`encode_prefill_dense_layer` (`attn.rs:6615-6785`) emits ~19 dispatches and
**15 blanket `memoryBarrierWithScope(Buffers)`** per layer — 450 full drains
for a 30-layer SmolLM2 pass. llama emits 0-or-1 barrier per node, only on a
real RAW/WAR/WAW overlap found by `ggml_mem_ranges_check`
(`ggml-metal-ops.cpp:221-224`, `ggml-metal-common.cpp:124-153`), after a
graph-optimize pass that reorders up to 64 nodes to widen concurrent sets.
Also fuse `rmsnorm + f32→f16` and `silu_mul + f32→f16` to remove 3 tensor
passes per layer (`attn.rs:6698, 6746, 6758`). ~5–10%.

### 2c. GEMM occupancy

The simdgroup GEMM tile is **byte-for-byte llama's** — NR0/NR1/NK = 64/32/32,
4 simdgroups, `mc[8]`, `ma[4]`, `mb[2]`, 2×2 SG tiling (`gpu.rs:2488-2568` vs
`ggml-metal.metal:10186-10314`). One difference: frink always requests 8192 B
threadgroup memory (`gpu.rs:3746`) because the partial-tile staging path
shares the allocation, where llama compiles a `bc_out=false` variant needing
6144 B (`ggml-metal-device.cpp:793`). Costs one threadgroup of occupancy per
core. ~3–8% on small-hidden models.

### 2d. Host-side leftovers

Every `apply_batch` does three `std::env::var` lookups per GEMM
(`weight_matrix.rs:2099, 2109`) and a 64-sample page-touching
`weight_fingerprint` per weight per call (`gpu.rs:4989-5006`, called at
`gpu.rs:5039` *before* the cache lookup). ~1–2%, mostly on the MoE path.

---

## Phase 3 — Metal MoE (OLMoE pp512 2.73×, tg128 1.54×)

Frink **already has a true `mul_mm_id`** — no gather, no scatter.
`mul_mm_id_impl` reads src1 indirectly (`gpu.rs:2824-2831`), writes dst
indirectly (`gpu.rs:2906-2919`), one z-slice per expert with a `tpe[im]`
early-return (`gpu.rs:2810-2813`). That matches `kernel_mul_mm_id`
(`ggml-metal.metal:10485-10520`). The prior plan item "port a real
`mul_mm_id`" is **already done**.

What is actually missing is that MoE layers are excluded from the fused stack
(`metal_prefill_dense_layer_eligible` is dense-only, `decoder.rs:726-728`), so
they fall to the legacy per-op path. Per layer that is q/k/v/o/router
`apply_batch` (each its own CB + commit + wait + host readback), plus the attn
block CB, plus the MoE CB: **~7 CBs / 7 syncs / 7 readbacks per layer × 16
layers ≈ 112 command buffers per pp512**, against llama's 2.

Fourteen distinct CPU round-trips per MoE layer are enumerated in the audit —
CPU rms_norm, CPU QKV bias, CPU QK-norm, CPU residual adds, CPU `route_top_k`
softmax+top-k, host `ids`/`route` buffer builds, and a **host** `map0`.

Note: `encode_moe_mm_id_map0` and its kernels
`moe_mm_id_map0_ne20_{2,4,6,8}` are already written (`gpu.rs:1006-1085`,
`gpu.rs:3483`) and have **zero callers** — the host version at `gpu.rs:5956`
is used instead. llama runs map0 as a kernel
(`ggml-metal.metal:10385-10437`).

**Fix:** add MoE variants to `PrefillDenseLayerMetal` (`attn.rs:6497-6514`);
encode router GEMM → GPU `soft_max_topk` → `encode_moe_mm_id_map0` →
`encode_mul_mm_id`×2 → `silu_mul` → `encode_mul_mm_id` → weighted sum →
residual into the existing stack encoder. **No new MSL required** beyond
wiring the written map0 kernel and a GPU top-k.

**Expected:** MoE pp512 2.73× → ~1.3×, tg128 1.54× → ~1.1×.

---

## Phase 4 — Coverage and honest refusal (parallel track, independent of perf)

### 4a. Fail closed on unimplemented graphs (correctness, blocking)

Every `LLM_ARCH_NAMES` string in `llama-arch.cpp:9-147` has an entry in
`capability.rs:196-590`, so nothing fails as "unknown arch". About 50 archs
are admitted to `ArchPath::GenericGqa`, whose graph features the generic
decoder does not implement — **they load and produce wrong logits rather than
refusing**, which is worse than missing.

The generic decoder implements (`tensor_role.rs:40-59`): attn_norm, q/k/v/qkv,
output, q_norm/k_norm, post_attention_norm, ffn_norm/gate/up/down,
post_ffw_norm, ffn_gate_inp + `_exps`, shared experts, grouped top-k, Q/K/V
bias only, Gemma SWA + embedding scale.

Absent everywhere on that path: ALiBi, learned `position_embd`, attn/ffn/output
bias beyond QKV, attention sinks, partial rotary (`n_rot < head_dim`),
residual/embedding/attn multipliers, parallel attn+FFN residual, non-Gemma
logit softcap, and MoE routing bias `ffn_exp_probs_b`.

Related: `unsupported_feature_keys` (`capability.rs:616-640`) refuses
softcap/SWA only via `{arch}.attention.sliding_window_pattern`. Archs that set
plain `{arch}.attention.sliding_window` without a pattern key — gpt-oss,
cohere2, exaone4 — pass the check and silently run full attention.

**Fix:** derive required graph features per arch, gate `GenericGqa` admission
on them, and refuse with a named-feature error otherwise. Add a test that
every arch on the generic path declares only implemented features.

### 4b. F16 does not load — DONE (`7ef74f1`)

`GgmlType::F16` was parsed (`frink-gguf/src/lib.rs:143`) and sized
(`lib.rs:171`) but had **no dequant arm anywhere** — the only two references
in the whole workspace are those two lines. Every F16 tensor hits
`UnsupportedDtype` (`loader.rs:735`). The BF16 arm was the template.

Fixed by `frink_quant::dequant_f16` (via `half::f16::to_f32`, not a bit
shift — f16 subnormals do not widen by shifting the way bf16 does) plus
a shared `loader::widen_plain_float` covering F32/F16/BF16 for all seven
GGUF loaders, which previously each inlined their own two-way match.

### 4c. Ranked coverage additions

| # | Addition | Why | Cost | Port from |
|---|---|---|---|---|
| 1 | ~~F16 tensor loading~~ **DONE** (`7ef74f1`) | was a hard load error | XS | mirrored the BF16 arm |
| 2 | MXFP4 Metal + CUDA | gpt-oss-20b/120b; CPU-scalar only now | M | `ggml-metal.metal` `kernel_mul_mv_mxfp4_f32`, `ggml-cuda/mmvq.cu` |
| 3 | gpt-oss graph: attn sinks, swiglu_oai clamp, SWA | pairs with #2; silently wrong today | M | `src/models/openai-moe.cpp` |
| 4 | `ffn_exp_probs_b` in generic MoE loader | unlocks dots1, ernie4_5-moe, bailingmoe2, exaone-moe, hunyuan-moe, afmoe in one change | S | `llama-graph.cpp` `build_moe_ffn` |
| 5 | Metal/CUDA Q2_K + Q3_K + IQ4_NL matvec | Q3_K_M/Q2_K standard for 30B+; CPU-only now | M | `ggml-metal.metal`, `ggml-cuda/vecdotq.cuh` |
| 6 | ~~IQ2_XS / IQ2_S / IQ3_S / IQ1_M~~ **DONE** | tiers Unsloth Dynamic ships; CPU scalar, goldens bit-exact vs linked `ggml-quants.c` | M | `ggml-quants.c` |
| 7 | Granite / MiniCPM multipliers | ~3 scalars, widely quantized archs | XS | `src/models/granite.cpp`, `minicpm.cpp` |
| 8 | Cohere2 / Command-R parallel residual + logit scale | common GGUFs, wrong today | S | `src/models/cohere2.cpp` |
| 9 | Partial rotary (`n_rot`) + full bias | fixes stablelm, phi2, gptneox, starcoder2, gpt2 at once | S | `src/models/stablelm.cpp`, `starcoder2.cpp` |
| 10 | olmo2 post-norm + ALiBi (bloom/mpt/jais) | last structural families in the "admitted but wrong" bucket | S/M | `src/models/olmo2.cpp`, `bloom.cpp` |

**Below the line:** recurrent (`mamba2`, `rwkv7`) and hybrid (`qwen3next`,
`lfm2`) need a whole state-carrying engine, and frink already fails closed on
them — they cost more and hurt less than anything above.

---

## Phase 5 — Hygiene and legacy cleanup

- ~~**Restore the clippy gate.**~~ **DONE (`c8a4cc6`).** It was red on both
  feature sets — 10 errors with default features (6 `frink-quant`,
  4 `frink-models`) and 25 more under `--features metal`, which nothing in
  CI or the documented command was covering. All mechanical. Note the metal
  set is a *separate* gate: run
  `cargo clippy -p frink-cli -p frink-server -p frink-metal --features metal
  --all-targets -- -D warnings` too, or half the workspace stays unlinted.
- `cpu_int_dot_enabled()` (`weight_matrix.rs:297-306`) is off unless a binary
  opts in. The bench, CLI and server all call `default_cpu_int_dot_on()`, so
  suite numbers are fine — but any embedder of `frink-core` silently falls
  through to the f32 dequant-dot at `weight_matrix.rs:1637-1647`, a much
  slower engine. Move the default into the getter.
- Asymmetric contract: `Q8_0X4_GEMM_NC = 8` (kernel tiles internally) vs
  `Q4_KX8_GEMM_NC = 4` (caller chunks). Unify while touching both in 1a/1b.
- CUDA batch arm (`weight_matrix.rs:1224-1243`) is still a per-position matvec
  loop — already flagged in its own comment, not on the M2 path, but the same
  class of bug as everything in Phase 1.
- Delete as replacements land: `pack_q8_k_qs_x4_i8`, the seven `by_row`
  transposes, the host `map0` at `gpu.rs:5956` once the kernel is wired, the
  false "until i8mm … lands" comments, and stale ROADMAP/RESULTS bullets.
- Keep: one scalar/non-SIMD CPU reference for golden tests; per-op Metal
  fallback until the fused stack covers MoE too.

---

## Out of scope this push

- CUDA parity (no Host B pin); keep CUDA builds compiling only.
- Serving-suite prefill claims (the engine table is the parity ledger).
- Recurrent/hybrid/vision/embedding engines (Phase 4 "below the line").
- Broad IQ coverage beyond item 4c.6 unless it unblocks a suite row.

---

## Definition of done

The push is finished when **all red rows** in
[`benchmarks/RESULTS.md`](../../../benchmarks/RESULTS.md) read ≤ 1.0×, with
answer parity. Tracked explicitly so "mostly done" is not a resting place.
**25 red as published** (29 at the start of the push); the Metal `pp512`
count drops to 5 when the owed d=128 suite run publishes:

| Backend / test | Rows still red | Worst | Owning item |
|---|---|---|---|
| CPU `tg128` | 8 — SmolLM2 2.45, Qwen3-0.6B 1.74, Qwen2.5-0.5B 1.67, Gemma-3-1B 1.66, Phi-4 1.23, Mistral 1.20, OLMoE 1.11, TinyLlama 1.08 | 2.45× | `cpu-decode-scaling` |
| CPU `pp512` | 7 — Gemma-3-1B 1.94, SmolLM2 1.40, Mistral 1.31, Qwen2.5-0.5B 1.28, OLMoE 1.23, Phi-4 1.21, Qwen3-0.6B 1.16 | 1.94× | `cpu-gemma3-prefill`, 1a–1d |
| Metal `pp512` | 9 — OLMoE 2.48, Qwen3-0.6B 1.81†, Phi-4 1.24†, Gemma-3-1B 1.17, Qwen2.5-0.5B 1.12, Mistral 1.10†, Llama-3.2-1B 1.08, IQ4_XS 1.08, Llama-3.2-3B 1.08† | 2.48× | `metal-moe-stack`, `metal-fa-mma-d256` |
| Metal `tg128` | 1 — OLMoE 1.38 | 1.38× | `metal-moe-stack` |

† already fixed by the d=128 MMA in an interleaved A/B (1.07×, 1.04×,
0.97×, 0.89× respectively) but **not published** — the suite run is owed.
TinyLlama (1.01×) and SmolLM2 (1.02×) Metal `pp512` count as closed, as
do all 9 remaining Metal `tg128` rows.

Plus one row that is **not measurable today**: Gemma-4-E2B has no llama column
because Homebrew `llama-bench` lacks the `gemma4` arch, and frink's own
number uses sequential `forward_token` for `pp*` (`bench_model.rs:148-152`).
Both sides need fixing before that row can be called anything — a blank gap
is not a closed gap.

Track the ledger after each validation run; a phase is done when its rows are
green **and stay green** in a later suite run.

## Verification loop (every change)

1. `cargo test` for touched crates + Metal shape sweeps
   (`assert_mul_mm_sg_matches_matvec`); bit-exactness tests against the
   superseded kernel for every 1a–1d port.
2. Targeted `frink bench … --compare` on the affected model, quiet host —
   keep the change only if the gap shrank.
3. Answer check: same prompt, greedy, vs llama.cpp, sequential.
4. **Full `--suite --fit-host --skip-missing` + `--render`.** Diff every row
   against the previous ledger. Any regression beyond the ~20% host spread is
   either explained in the commit body or the change is reverted.
5. Row closed only at gap ≤ 1.0× **and** matching answers. Within ~5% counts;
   >1.05× means keep going.
6. Remove the superseded legacy path in the same or next commit.
7. Commit the regenerated `RESULTS.md` + receipts with the change that earned
   them, so every speed claim in git history has a measurement behind it.

```mermaid
flowchart LR
  CPU[Phase 1 CPU prefill: hoist interleave, kill transpose, i8mm tiers] --> Bench[Bench same model]
  Metal[Phase 2 Metal FA MMA] --> Bench
  MoE[Phase 3 MoE on fused stack] --> Bench
  Bench --> Answers[Answer parity vs llama]
  Answers --> Suite[Full suite plus render - every change]
  Suite --> Reg{Any row regressed?}
  Reg -->|yes| Revert[Revert or explain in commit]
  Reg -->|no| Legacy[Delete superseded path, commit RESULTS]
  Revert --> Bench
  Legacy --> Done{All 29 rows under 1.0x?}
  Done -->|no| Next[Next lever]
  Next --> Bench
  Cover[Phase 4 fail-closed plus F16 plus MXFP4] --> Suite
```
