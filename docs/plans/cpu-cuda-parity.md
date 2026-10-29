# CPU and CUDA parity with llama.cpp

Metal is done. This plan is about the two backends that are not, and it
is written from measurements taken on 2026-09-04 on rented, dedicated
hosts rather than from the ledger, because the ledger could not see any
of it: it had no x86 row, no CUDA row, and its CPU rows were Metal runs
mislabelled (#126).

Read `north-star.md` first. This plan is subordinate to it: same
models, same command shapes, same or better performance, on the
hardware people actually own. Most of them own x86 with an NVIDIA card.

## What parity means here, precisely

Three claims, in order. A step that improves one while breaking a
higher one is a regression, not progress.

1. **It answers the same.** Token-identical greedy output, or a KL
   within the reference spread. `frink parity` is the oracle.
2. **It runs at all.** The kind has a kernel on that backend. Falling
   back to the host is not running: it is a different program with the
   same output.
3. **It is not slower.** `gap = llama / frink` at or below 1.0 on the
   same host, same GGUF, same backend, quiet host.

Claim 2 is where CUDA fails today, and it is invisible in a gap column
because a fallback still produces numbers.

## Measured state, 2026-09-15

Re-measured on a rented, dedicated AMD Ryzen 9 3900X (12 cores, Zen 2:
AVX2 and no AVX-512) with an RTX 3090 (Ampere, CUDA 12.4, llama.cpp
`1269cb1` built twice, CPU-only for the CPU rows and with CUDA for the
CUDA rows). Every row below has a receipt in `benchmarks/receipts/
engine/` and is rendered in `RESULTS.md`; the 7945HX section there is
the same code BEFORE #159 and is kept as the before.

### CPU, x86 (the first x86 rows since #159)

| model | pp512 gap | tg128 gap |
|---|---|---|
| Llama-3.2-1B Q5_K_M | **1.04x** | 1.16x |
| Llama-3.2-1B Q4_K_M | 1.29x | 1.11x |
| Llama-3.2-1B Q6_K | 1.36x | 1.16x |
| Llama-3.2-3B Q4_K_M | 1.25x | 1.13x |
| Llama-3.1-8B Q4_K_M | 1.20x | 1.10x |
| Mistral-7B Q4_K_M | 1.20x | 1.09x |
| Phi-4-mini Q4_K_M | 1.18x | 1.12x |
| Gemma-2-2B Q4_K_M | 1.36x | 1.09x |
| Gemma-3-1B Q8_0 | 1.63x | 1.04x |
| Qwen3-0.6B Q8_0 | 1.50x | 1.15x |
| Qwen2.5-0.5B Q8_0 | 1.64x | 1.16x |
| TinyLlama-1.1B Q8_0 | 1.52x | 1.08x |
| SmolLM2-135M Q8_0 | 2.08x | 1.34x |
| Llama-3.2-1B IQ4_XS | **4.45x** | 1.17x |

**x86 prefill went from 6x-10x to 1.0x-1.6x** on the K-quants, which is
#159's AVX2 GEMMs finally measured. The first run of this suite had ONE
row still at **8.56x**: Llama-3.2-1B Q5_K_M prefilled at 44 tok/s
against llama.cpp's 379 while Q4_K and Q6_K sat at 1.2x on the same
host. The Q5_K batch arm in `weight_matrix.rs` gated its Kx8 path on
`cfg!(target_arch = "aarch64")` where the Q4_K arm takes the path
unconditionally and the Q6_K arm asks `q6_kx8_gemm_uses_acts_x4`, so
every x86 Q5_K prefill fell through to the per-row GEMM -- class 2, a
written-down claim about an architecture beside kernels that had an
AVX2 body since #159, the exact shape #239 fixed in `int_dot_tier_here`.
One predicate (`q5k_batch_takes_kx8`) later it reads 0.91x on a 5950X
and 1.04x here, and Phi-4-mini, whose `attn_qkv` is Q5_K, went 3.24x
to 1.18x with it. (The 5950X box went offline mid-run; its numbers are
in the session log, not the ledger.)

What is left on x86, by class:

1. **No kernel: IQ4_XS prefill, 4.45x.** The batched arm for every
   kind without a Kx8 tier is the generic fallback: per row, per
   activation, `dot_iq4_xs_f32`, which re-decodes the row's nibbles
   `batch` times per prefill and multiplies in f32. llama.cpp's
   `ggml_vec_dot_iq4_xs_q8_K` is an int8 dot over Q8_K-quantized
   activations (the codebook lookup into `maddubs` / `sdot`).
   **Closed on the same day**, `frink_quant::iq4_xs_q8` (scalar twin,
   SDOT, AVX2; the activations quantized once per matmul on both the
   single-vector and the batched path): on the M2 Pro, interleaved
   twice against the previous binary, Llama-3.2-1B IQ4_XS prefill went
   53 to 170 tok/s and decode 40 to 85, and `frink parity` against
   libllama is MATCH at KL 3.8e-5. The x86 number needs a rented box;
   the local ratio is the evidence that the missing kernel was the
   gap. The same fallback still serves IQ4_NL, Q2_K, Q3_K, Q5_0, Q4_1
   and MXFP4, none of which is in the suite.
3. **Fixed per-op cost: the Q8_0 rows at 1.5x-2.1x that shrink with
   size** (SmolLM2 2.08x, Qwen 1.5x-1.6x, TinyLlama 1.5x, 8B 1.2x), and
   SmolLM2's decode at 1.34x where every other decode row is 1.04x to
   1.17x. #128's constant, seen from the prefill side.

### CUDA (RTX 3090, Ampere)

Step 1's exit criterion is half met. **The K-quant GEMM is correct on
hardware**: `cargo test -p frink-cuda --features cuda -- --ignored`
passes all 13 tests, and `frink verify --backend cuda` is
token-identical to the CPU on Q4_K_M, Q5_K_M, Q6_K, Q8_0 and IQ4_XS
Llama / TinyLlama checkpoints over a 64-token prompt and 24 generated
tokens. One tolerance was wrong, not one kernel:
`launch_mul_mm_matches_the_scalar_twin` bounded the GPU-vs-twin error
relative to the RESULT, and with random fixture weights a 256-column
sum cancels down to 0.6 while its terms' L1 is in the thousands, so a
4.8e-4 absolute drift from FMA contraction read as a 1e-4 relative
failure on one Q5_K element. The bound is result-relative plus an
absolute floor per column now, and the measured worst case is written
into it.

**It is not within an order of magnitude**, and the gap is WIDER on
Ampere than on the Xeon+3060 row: prefill 25.5x to 43.3x, decode 2.75x
to 9.25x (IQ4_XS decode is the 9.25x; the K-quants are 2.75x to
3.36x). Sampled with `nvidia-smi -lms 500` DURING the runs, twenty-six
and thirty-one busy samples on Llama-3.2-3B Q4_K_M:

| workload | GPU util | mem util | power |
|---|---|---|---|
| pp512 (323 tok/s) | **30% to 39%** | 1% | 175 W |
| tg256 (66 tok/s) | 45% to 75% | 19% to 30% | 250 W to 308 W |

So prefill leaves the 3090 idle two thirds of the time at half the
power decode draws, which is the signature of a launch-bound or
host-synchronised graph, NOT of slow arithmetic: a slow kernel would
pin utilization. This is the lead the 2026-09-04 section called "50%,
needing repeated sampling"; it is repeated now and it holds. Decode's
19% to 30% memory utilization against llama.cpp's ~60% is #133's
memory-bound half, unchanged. The order of work for CUDA is therefore:
count the launches and host round trips in one prefill step (the
per-op `cudaStreamSynchronize` shape) before touching any kernel
body, because at 35% utilization the arithmetic cannot be more than a
third of the problem.

**Counted, 2026-09-17, from the code.** The batched host body
(`decoder.rs`, `forward_batch_*`) runs a dense layer as seven
`WeightMatrix::apply_batch` calls -- `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate`, `up`, `down` -- and on CUDA each one is
`frink_cuda::mul_mm_launch::launch_mul_mm`, which is ONE round trip:
`x_batch[..].to_vec()` (a host copy), `htod_copy` of it (pageable),
`alloc_zeros` for the output, one launch, `dtoh_sync_copy`. Nothing
else in the layer touches the device: the norms, RoPE, the causal
attention over the whole prompt, the SwiGLU and the residual adds run
on the host between the matmuls. The shared activation quantization
that lets the CPU quantize `normed_batch` once for q/k/v
(`quantize_batch_acts`) returns `None` on CUDA, so q, k and v each
upload the same 512-row activation. Per Llama-3.2-3B layer at pp512
(`n_embd 3072`, `n_ff 8192`, `n_kv 8 x 128`), in f32:

| call | up (MB) | down (MB) |
|---|---|---|
| q, k, v | 3 x 6.3 | 6.3 + 2.1 + 2.1 |
| o | 6.3 | 6.3 |
| gate, up | 2 x 6.3 | 2 x 16.8 |
| down | 16.8 | 6.3 |
| **layer** | **54.5** | **56.6** |

**111 MB per layer, 3.1 GB per 512-token prefill over 28 layers, in
196 synchronous round trips**, against the 6 MB of token embeddings
llama.cpp uploads once and the logits it downloads once
(`ggml_backend_cuda_graph_compute` has no sync inside the node loop;
see step 2). At the 3 to 6 GB/s a pageable `cudaMemcpy` gets on PCIe
3.0 that is 0.5 to 1.0 s of the 1.58 s a pp512 step takes at 323
tok/s, and the GPU is idle for all of it, which is the 30% to 39% the
sampler saw. This is the arithmetic the #136 retraction asked for,
run BEFORE the code: the round trips alone can account for most of
the step, so removing them can reach the target for prefill, where
for decode they could not (a decode round trip carries 12 KB, not 6
MB, and the decode GPU sits at 86% to 93%).

What reaches it is the layer staying on the device -- embeddings in,
logits out -- not a cheaper round trip: chaining only the matmuls
that are adjacent (q/k/v, then gate/up/down) still leaves three round
trips and 42 MB per layer, a 2.6x cut against a 25x to 43x gap. So
the CUDA prefill work is the Metal work again, in order: a batched
RMSNorm, RoPE, a causal prefill attention kernel and the residual add
on the device, `launch_mul_mm` taking a device pointer for its
activation, and one download at the end; and it is verified the way
step 1 was, by the hardware test suite and `frink verify --backend
cuda`, before any receipt. The direct measurement that confirms the
split (memcpy time against kernel time inside one pp512 step, `nsys`
or event-timed) is the first thing to take on the next rented box; it
is cheaper than the first kernel and it is what says whether the
seven-call shape or the GEMM body is the bigger half once the copies
are gone.

**Measured the same day, and the count held.** `nsys` on `main`'s
pp512 (RTX 3090, Llama-3.2-3B Q4_K_M, two steps): 394 `cuMemcpyDtoH`
and 591 `cuMemcpyHtoD`, 0.88 s of memcpy per step against 0.39 s of
GEMM kernels (`q4_k_mul_mm` 1.9 ms per call, `q6_k_mul_mm` 2.2 ms).
The copies were two thirds of the step, as the arithmetic said.

**The resident stack landed (`frink_cuda::prefill`, #260).** A run
of dense layers per launch: the hidden batch up once, five small
kernels (row RMSNorm, bias add, RoPE, causal GQA, residual add) plus
the existing SwiGLU between GEMMs that now take device pointers
(`enqueue_mul_mm`), the hidden batch down once, and each layer's K/V
rows down for the host cache, which stays authoritative. `frink
verify --backend cuda` token-identical on Llama-3.2-3B / 1B Q4_K_M
and Qwen3-0.6B Q8_0 (QK norm), all 17 hardware tests green, and
**pp512 305 to 912 tok/s, 2.9x**, interleaved against `main` three
times (main 305 / 319 / 305, stack 883 / 900 / 912). Per-step memcpy
fell from 0.88 s to under 0.1 s, and most of what is left is the
one-time weight upload the profile's first step carries. The gap on
that row is 25.5x to about 9x.

What is left on CUDA prefill, from the same profile of the new
binary, per two steps: `q4_k_mul_mm` 0.63 s, `causal_gqa_prefill_f32`
0.14 s, `q6_k_mul_mm` 0.12 s, everything else under 0.01 s. So the
GEMM is now three quarters of the GPU time at about 7 TFLOPS on a
card whose f32 peak is 35 and whose int8 tensor cores are what
llama.cpp's `mmq` uses; the attention kernel (one warp per query and
head, no tiling) is the other sixth. Those are the next two items,
in that order, and both are kernel bodies now that the copies are
gone -- which is the order the 2026-09-15 paragraph asked for.

**Both landed the same day (#261).** `mul_mm_tc` puts the product on
`mma.sync.m16n8k16` (f16 operands, f32 accumulation) under the same
per-kind dequantization; the attention kernel reads `float4` slices
with four query rows per block. RTX 3090, three interleaves with the
SIMT body in the same binary (`FRINK_CUDA_MUL_MM=simt`): pp512
Llama-3.2-3B Q4_K_M **975 to 1932 tok/s**, Llama-3.2-1B 2368 to 4577.
Per two steps the kernels are now `q4_k_mul_mm_tc` 0.24 s (0.70 ms a
call, from 1.88), `causal_gqa_prefill_f32` 0.054 s (0.96 ms a layer,
from 2.5), `q6_k_mul_mm_tc` 0.054 s: about 0.17 s of GPU time per
step against a measured 0.26 s per step, so a third of the step is
now HOST time between launches, which no kernel can remove. Against
llama.cpp's ~8,200 tok/s on this card the row is at **4.3x** (from
25.5x on 2026-09-15). The next three items, in order: profile the
host third (`perf`, not a guess); the GEMM's remaining 3x against
`mmq` (int8 operands and scales applied after the MMA, no per-tile
dequant); a K/V-tiled attention kernel. Building the tensor-core body
found `resident_cuda_weights` trusting a host address that a freed
buffer can hand to another tensor (fixed with a byte fingerprint).

## Measured state, 2026-09-04

### CUDA (GTX 1080, CUDA 12.4, llama.cpp built with CUDA on the same box)

First execution of this code path on a GPU. `Cuda::gemm_supported`
carried the comment "UNRUN ON HARDWARE" until this run.

| model | pp512 gap | tg128 gap |
|---|---|---|
| gemma-2-2b Q4_K_M | **369x** | 19.3x |
| Llama-3.2-3B Q4_K_M | **325x** | 17.5x |
| Llama-3.2-1B Q4_K_M | 284x | 15.1x |
| Llama-3.2-1B Q6_K | 280x | 13.0x |
| SmolLM2-135M Q8_0 | 17.2x | 9.2x |
| Qwen3-0.6B Q8_0 | 15.4x | 11.3x |
| TinyLlama-1.1B Q8_0 | 15.2x | 10.7x |
| Qwen2.5-0.5B Q8_0 | 14.9x | 12.8x |
| gemma-3-1b Q8_0 | 12.4x | 11.0x |

The K-quant prefill rows are a **missing kernel**, not slowness: there
was no `mul_mm` for them, so a 512-token prefill issued 512 matvec
launches. Fixed in this branch; **unverified on hardware**.

### CPU, aarch64 (20-core Cortex-A725, i8mm, idle)

| model | test | frink default | frink `spin` | llama.cpp |
|---|---|---|---|---|
| 3B Q4_K_M | pp512 | **132.53** | | 46.40 |
| 3B Q4_K_M | tg128 | 10.38 | **23.14** | 17.86 |
| 8B Q4_K_M | pp512 | **61.17** | | 19.15 |
| 8B Q4_K_M | tg128 | 6.64 | **12.41** | 9.06 |

**Prefill is already a 3x lead.** Decode is a loss with the default
thread pool and a win with the persistent one.

### CPU, x86 (10-core Xeon E5-2630 v4, idle)

Llama-3.2-3B Q4_K_M: **1.03 tok/s** tg128. Roughly an order of
magnitude off. Not diagnosed (#127).

### Every backend, small models

SmolLM2-135M decode is 13 to 15 tok/s at 4, 8 and 19 aarch64 threads
while llama.cpp does 190 to 204. Flat in thread count, unmoved by
either pool: a fixed cost of about 60 ms per token (#128).

## Coverage: which kinds have a kernel

llama.cpp implements every `QuantKind` on CPU, CUDA and Metal. frink
does not, and that is the parity gap a tok/s column cannot show.

The table that used to sit here was prose and drifted. It is now
[`docs/KERNELS.md`](../KERNELS.md), generated from the dispatch tables
by `frink caps --kernels` and checked by a test (#405,
[ADR 0010](../adr/0010-coverage-before-speed.md)). A `HOST` cell there
means the tensor is decoded on the host and the GPU is idle for that
matmul. It still answers correctly, which is why this never surfaced
as a bug.

How a new CUDA kernel is verified before a GPU runs it: its arithmetic
is checked against `frink_quant` by a Rust twin, and for the GEMM the
emitted CUDA C is executed on a host CPU and compared to that twin bit
for bit (`crates/frink-cuda/tools/mul_mm_host_check/run.sh`). `cargo
test -p frink-cuda --features cuda -- --ignored` on a real device is
the exit criterion, plus a bench row.

## The four kinds of gap, and why the distinction matters

Ordering work by tok/s alone puts the 369x row first and gets the
priorities wrong. These are different problems with different fixes.

1. **No kernel.** The backend cannot run the format and silently uses
   another one. Fix: write the kernel. Measurable before and after,
   and the before number is meaningless as a performance signal.
2. **Wrong dispatch.** A kernel exists and is not selected, or is
   selected for the wrong shape. Fix: the predicate. Costs nothing to
   run and is usually a large win. `FRINK_CPU_POOL` is this.
3. **Fixed per-token overhead.** Independent of weights, threads and
   backend. Fix: find the constant. Worth more than any kernel at small
   sizes and worth nothing at large ones.
4. **Genuinely slower arithmetic.** The kernel is right and loses.
   Fix: profile. This is the only class that needs new performance
   work, and it is the class frink has the least of.

Today's evidence says frink's remaining gap is mostly 1, 2 and 3.
That is good news and it should change the order of work.

## Ordered plan

Each step names its exit criterion. "Measured" always means: quiet
host, `frink bench` guard passing, `ps` checked for a single busy
core, receipt committed.

### 1. Verify the K-quant CUDA GEMM on hardware  [blocks everything else on CUDA]

Written and held against `frink_quant::dequant_q*_k` sub-block by
sub-block, but the CUDA C is a second transcription of the same
arithmetic and no GPU has run it. Rent one box, run
`frink parity` for correctness and `frink bench --suite --backend
cuda` for the gap.

**Exit:** Q4_K/Q5_K/Q6_K prefill within the same order of magnitude as
llama.cpp, and `parity` agreeing on tokens. If the kernel is wrong,
this is where it is caught, before any of it is published.

### 2. Close the CUDA decode gap (9x to 17x)

Present on every kind including Q8_0, so it is not the missing GEMM.
Three candidates were tested on hardware on 2026-09-04 and **two are
now ruled out**:

- **The GQA reduction is not it.** `FRINK_CUDA_GQA=1` is correct
  (`verify` is token-identical) and **42% SLOWER**: 6.85 tok/s against
  11.88 on Llama-3.2-1B Q4_K_M. It also never compiled before that day
  (NVRTC has no `INFINITY`), so the flag had never run at all.
- **CUDA graphs are not it, yet.** `FRINK_CUDA_GRAPH=1` measures
  11.80 against 11.84 off, exactly as its own doc predicts: nothing
  enqueues into a captured stream, so it is groundwork.
- **The GPU is NOT idle, and the claim that it was is retracted.** An
  earlier version of this plan said `nvidia-smi` reported 36%
  utilization during decode, and concluded the cost was host-side. That
  number came from a single instantaneous sample taken AFTER a bench
  run had finished, so it caught an idle moment. Sampled five times
  DURING a `tg256` decode on a GTX 1080, `main` sits at **86% to 93%**.
  Decode is **kernel-bound**, and the host-side theory this plan was
  built on for a day was wrong.

  The correction cost a PR (#136, measured 22% SLOWER on hardware:
  8.24 against 10.55 tok/s). The arithmetic that should have caught it
  was available before the code was written: at 36% utilization,
  removing every host round-trip buys at most 1/0.36 = 2.8x against a
  9x to 17x gap. A lever that cannot reach the target is the wrong
  lever even if the diagnosis is right, and here the diagnosis was also
  wrong.

**And prefill is a second, larger problem that widens with hardware.**
Re-measured on an RTX 3060 (Ampere): prefill is **55x to 57x** off
llama.cpp, against ~11x on the GTX 1080. llama.cpp is 2.4x faster on
Ampere than on Pascal; frink is not faster at all. Decode is roughly
unchanged at 11.5x to 12.2x. The GPU sits about half idle during
prefill (0%, 57%, 50%) where decode runs it at ~90%, so the two have
different signatures and are probably different bugs. Thread count is
not it: `-t 4` against frink's chosen `-t 1` is worth 25% and leaves
44x.

Treat the 50% as a lead needing repeated sampling, not a conclusion.
One instantaneous utilization sample already cost this plan a day and a
PR.

**What llama.cpp's CUDA backend does**, read out of
`.scratch/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`. Kept because it
is true and useful, with the caveat that it is NOT the explanation for
this gap:

- `ggml_backend_cuda_graph_compute` runs a whole token's graph, and its
  node loop contains **zero** `cudaStreamSynchronize` or
  `cudaDeviceSynchronize` calls. Every sync in the file is at the graph
  boundary, in `tensor_set` / `tensor_get` / `cpy_tensor`.
- Every tensor, including every intermediate activation, is allocated
  in a device buffer up front. Residency is a tensor's default state,
  not an optimisation applied to a pair of calls, so the host never
  sees an intermediate.

That is a real architectural difference and it settles the identity
question a residency scheme would face: a tensor's device buffer IS its
identity, for its lifetime, which is stronger than any length or epoch
comparison. But frink's decode already runs the GPU at ~90%, so it is
not waiting on the host, and closing this difference would not close
the gap. Recorded so the next reader does not re-derive it and reach
the conclusion this plan already retracted.

**The cause is the kernels.** frink's CUDA matvec uses one 256-thread
block per row with a shared-memory tree reduction; ggml-cuda uses
warp-level `dp4a` with no shared-memory round trip. At ~90%
utilization and 9x to 17x off, that is where essentially all of the
difference is. Compare one kernel against its ggml-cuda equivalent for
the same kind and shape, and close the arithmetic.

**A dead API that should still go.** 
`frink-cuda/src/gpu.rs` defines `DeviceAct` / `upload_act` /
`matvec_into` / `download_act`, whose doc says they exist "so a
matvec's output can be fed straight into the next matvec without a
DtoH/HtoD round-trip (the exact per-call upload/download overhead that
made CUDA decode bandwidth-starved)". **All four have zero uses outside
that file.** The decode path takes `launch_matvec`, which returns
`Vec<f32>` and therefore ends in `dtoh_sync_copy`: every matmul
uploads, allocates, launches, synchronises and downloads, on the order
of a hundred times per token.

Five DtoH points exist per dense decode layer and chaining can reach
only three of them: norms, RoPE and the attention reduction sit between
the matmuls on the host, so there is no consecutive-matmul pair left
for `matvec_into` to serve. Measured, and the reason the narrow fix
could never have worked even had the diagnosis been right.

**Exit:** tok/s against llama.cpp on the same GPU and model, with
utilization already high on both sides. Not a utilization target.

**The hazard to design around first, and it was real.** Metal's
equivalent used to match on LENGTH alone, which this note called safe
"only because exactly one site sets it and it is cleared aggressively".
It was not safe. There is one publisher but THREE consumers, each
routinely handed a `hidden_dim`-long activation that is not the
published one, and the publication was a thread-local raw pointer into
`DECODE_SCRATCH` -- a process-wide `Mutex` the pointer escaped, so two
concurrent `frink-server` requests could have one answer the other's
`lm_head` with its own activation, lengths agreeing by construction.
Fixed in `frink-metal/src/resident_act.rs` (issue #166): the
publication lives inside the thing the mutex protects, records the host
address and length of the exact vector the stack returned, is dropped by
any borrow of the buffer it describes, and holds the guard while the
buffer is bound. Read that module before writing the CUDA twin.
Whatever carries residency needs an identity the caller cannot get
wrong, not a length comparison.

### 3. Decide the CPU pool by work size, not by environment variable

`FRINK_CPU_POOL=spin` is +123% at 3B and +87% at 8B on aarch64 and
takes decode PAST llama.cpp. It is -37% at 135M. So the default cannot
flip and cannot stay: it needs a rule.

`MIN_TASK_MACS` already computes the quantity the rule needs. #27
proposes deleting it; the measurements say keep it and use it to select
the pool per operation.

**Exit:** one predicate, shared by every call site, with the crossover
measured on both aarch64 and x86 rather than guessed. `spin` stops
being a user-visible knob.

### 4. Find the 60 ms (#128)  [NAMED 2026-09-10: rayon's cold submit]

**The premise of this step was wrong twice, and both corrections are
recorded because each one cost work.**

The first framing, "a fixed ~60 ms per token, flat in model size", was
corrected by #155: the cost was proportional to gate and up projection
BYTES and fired only on Q8_0 and Q4_0, because the int-dot matvec
repacked the whole weight matrix on every call. That was 89% to 90% of
decode and it is fixed.

The second framing, in a comment on #128, measured 5.48 us per
fork-join region, multiplied by ~210 regions per token, got 6.7% of the
token, and concluded scheduling could not be what remained. **The
arithmetic was right and the denominator was stale.** It was taken
against a 17.23 ms token, i.e. WITH the repack bug still inflating the
work. Once #155 removed that work the token fell to about 5 ms and the
same fixed dispatch became a much larger share of it.

#### What it actually is

`rayon::join` and the `par_iter` bridges both funnel into
`Registry::in_worker`, which has two arms with very different costs.
From a rayon worker: run one half inline, post the other for stealing,
wait on a `SpinLatch`, no syscall. From any other thread: inject the
job and block on a `LockLatch`, which is a pthread mutex and condvar.
Every forward pass was driven from a thread rayon did not own, so it
paid the second arm once per region, roughly five per layer.

Sampled with `sample` on an M2 Pro over SmolLM2-135M Q8_0 `tg128`,
CPU-only, `-t 6`:

| | share of the driving thread's wall time |
|---|---|
| `__psynch_cvwait` under rayon's `LockLatch` | **74%** |
| attention (`causal_gqa_attention_softcap`) | 10% |
| the one matvec that ran inline (`o_proj`) | 5% |

Across all twelve threads in that process, the NEON `q8_0x4` matvec
kernel held 6.6% of the samples and `__psynch_cvwait` 63.7%. During the
74% the driving thread spent asleep, the six workers it was waiting for
held about an eighth as many kernel samples between them: most of the
wait was the round trip, not the work.

#### The fix, and what it measured

`frink_core::par::on_workers` wraps a whole forward pass in one
`rayon::scope`, so the step runs on a worker and every nested region
takes the hot arm. `~150` cold entries per token become **one**.
`decoder/entry.rs` is the one place every public `Decoder::forward_*`
does this, and `par::cold_regions` is a per-thread operation counter so
the property is a test rather than a stopwatch.

Interleaved `main, branch, main, branch` on the M2 Pro, CPU only, which
is NOT a benchmark host, so these are ratios and not ledger rows:

| model | decode | prefill |
|---|---|---|
| SmolLM2-135M Q8_0 | **+29%** (4 of 4 rounds) | flat |
| Llama-3.2-3B Q4_K_M | **+9%** (4 of 4 rounds) | flat |
| Llama-3.1-8B Q4_K_M | +3%, swap-bound on this host | not run |

Against `llama-bench` on the same host and file, best-of-3 at `tg64`,
the gap moved from about 1.9x to about 1.5x. The host was too noisy for
that number to be worth more than its order of magnitude; the
main-versus-branch ratio is the reliable half.

Metal and CUDA are deliberately NOT promoted. Moving the step off the
main thread changes Metal's output (`Llama-3.2-3B Q4_K_M --ngl 99`,
greedy, diverges around the tenth token, deterministically on both
sides), because the Metal stack carries thread-local state across a
step. `on_workers` asks `weight_matrix::active_backend` first, and with
that gate the Metal answer is byte-identical to `main` over three runs.

**Exit:** 135M decode within 2x of llama.cpp on a QUIET host. Still
owed: the numbers above are from a laptop, so the ledger row in
`benchmarks/RESULTS.md` has not moved and must be re-measured on the
rented aarch64 box it was taken on.

### 5. x86 CPU  [DONE for decode, 2026-09-04]

The answer was a **default**, not a missing kernel.
`FRINK_CPU_INT_DOT` defaults on, its interleaved integer kernels are
aarch64-only, and on x86 it selected a scalar loop while bypassing the
AVX2 f32 dot that does exist. Cost: 4x to 8.8x of decode. Fixed
architecture-aware; Llama-3.2-1B Q4_K_M went from 6.8x off llama.cpp to
**1.4x**.

Note the shape of the error, because it recurs: the AVX2 arms were
present and correct, and were being SKIPPED. Two of the three
hypotheses in this issue (no x86 SIMD, slow x86 SIMD) were wrong, and
the code read as if they were right.

**What is left on x86:**

- **Prefill is still 6x to 10x.** That is now the biggest CPU gap in
  the ledger and has had no investigation at all.
- **No x86 int8 path exists.** Zen 4 advertises `avx512_vnni` and
  nothing here uses it. That is the natural successor to this fix, and
  `FRINK_CPU_INT_DOT=1` stays available precisely so such a port can
  measure itself against the f32 path.

### 6. Kernel coverage, by what people actually run

Not alphabetically, and not all 21. In order of how often a checkpoint
in the wild uses it.

The CUDA kinds below were written without a GPU and each was first
marked unverified on hardware. All of them have since run on one: on
2026-09-15 the `--ignored` CUDA suite passed on a rented RTX 3090 (all
13 hardware tests, with `launch_mul_mm_matches_the_scalar_twin` walking
every kind in `mul_mm::KINDS`), as the Status table at the end records.
The GEMMs are what CUDA prefill runs, and its remaining gap is tracked
in [#259](https://github.com/antonellof/frink/issues/259). The bench
rows each bullet asks for are a separate exit and are not all taken.

- **CUDA Q5_0, landed 2026-09-05, verified on hardware 2026-09-15.** Matvec and
  GEMM together, because `a_cuda_kind_with_a_matvec_also_has_a_gemm`
  forbids half of it. The GEMM's emitted C is executed on the host by
  `crates/frink-cuda/tools/mul_mm_host_check/run.sh` and matches the
  Rust twin bit for bit; the matvec's C has no such harness and no GPU
  had run either at the time. `cargo test -p frink-cuda --features cuda
  -- --ignored` was the exit criterion, plus a Q5_0 bench row.
- **CUDA IQ4_NL and IQ4_XS, landed 2026-09-09, verified on hardware
  2026-09-15.** Matvec and GEMM together. They are codebook formats, so
  `MulMmKind` grew a `codebook: Option<Codebook>` field and
  `kernel_src` emits it as a `__constant__ float[16]` ahead of
  `dequant_src`; the `dequant_twin` seam is unchanged, because its
  contract was always "given the block and `il`, write 16 floats in
  ascending element order" and a table lookup satisfies it exactly as
  an affine transform does. The GEMM's `Codebook` row is ONE slice: the
  emitter formats it into the CUDA and the Rust twin indexes it, so
  there is nothing to drift. The matvec kernels are `&'static str` and
  carry the sixteen values as a literal, which is a second structure --
  `every_embedded_codebook_is_the_mul_mm_codebook` parses them back out
  of the kernel text and holds them to the `Codebook`, bit for bit.
- **MXFP4 on GPU, landed 2026-09-09, verified on CUDA hardware 2026-09-15.** gpt-oss
  ships it and no GPU backend had it at all, so every expert decoded on
  the host with the device idle. Matvec and GEMM, the same codebook
  seam, with the E2M1 table and an E8M0 scale helper. 17-byte blocks:
  the only odd stride in the table, so nothing in either kernel may
  assume a block pointer is aligned to anything.
- **CUDA Q2_K and Q3_K, landed 2026-09-09, verified on hardware 2026-09-15.**
  Common in small-memory builds, and they were absent on all three GPU
  backends. Matvec and GEMM, both affine. Q3_K is the fiddly one: its
  third quant bit is a bit plane in `hmask` and it is INVERTED (a set
  bit means bias 0, a clear one bias 4), and its six-bit scales use a
  four-arm packing that is not Q4_K's. Sabotaging that inversion in the
  emitted CUDA alone makes the host check report 3,968 mismatches out
  of 4,096, so the check sees it. Metal still has neither.
- The IQ1/IQ2/IQ3 family last: rare, and each is a separate codebook.

**Exit per kind:** a kernel, a scalar twin, a `parity` run, and a bench
row. A kind without a bench row is unmeasured, not done.

### 7. Make the ledger able to hold the answer

Partly landed: the renderer now groups by host, receipts carry a host
slug, the suite lists `cuda` on 13 entries instead of 1, and a receipt
whose label disagrees with the backend that ran is refused at write
time.

Still owed: an x86 CPU row, an aarch64 CPU row from a quiet host, and a
CUDA row taken after step 1. `RESULTS.md` currently has no CPU rows at
all, which is honest and temporary.

## Traps, each one paid for already

- **A gap column cannot see a missing kernel.** The 369x row looked
  like a performance problem and was an absent GEMM. Check coverage
  before profiling.
- **A load average cannot see one busy core.** Every CPU number in this
  session was first taken while a daemon held 97% of a core, and the
  `--max-load 2.0` guard passed throughout.
- **A backend label is not the backend.** All 13 published CPU receipts
  recorded `backend_active: "Metal"`. Two fields in one file, with
  nothing comparing them.
- **A fallback is not a failure, which is what makes it dangerous.** A
  kind with no kernel still answers correctly, so nothing goes red.
- **One host is not a platform.** The pool wins on x86 at every size
  and loses at 135M on aarch64. A single-machine ledger would have
  published either as universal.
- **Do not benchmark on a laptop with a UI.** Rent a box. The whole
  investigation behind this plan cost under a dollar.
- **One instantaneous sample is not a measurement.** The claim that
  CUDA decode ran at 36% GPU utilization came from a single
  `nvidia-smi` taken after a bench had finished. It sent a day of work
  at a host-side cost that does not exist: the real figure, sampled
  during the run, is 86% to 93%. Sample repeatedly, and sample WHILE
  the thing is running.
- **Check whether the lever can reach the target before pulling it.**
  At the (wrong) 36% figure, removing every host round-trip was worth
  at most 2.8x against a 9x to 17x gap. That arithmetic was in the
  agent's PR body before the code was written, and reading past it cost
  a merged-nothing PR.

## Status

| Step | Issue | State |
|---|---|---|
| 1 CUDA K-quant GEMM verified | #131 | **done**: verify token-identical on RTX 3090 (2026-09-15), all 13 hardware tests pass |
| 2 CUDA decode | #133 | GQA and graphs ruled out; GPU at 86% to 93% util, kernel-bound; 2.75x to 9.25x on Ampere |
| 2b CUDA prefill | #259 | resident stack (#260) 305 to 912 tok/s, tensor-core GEMM and float4 attention (#261) to 1932; 4.3x off llama.cpp; host third of the step next |
| 3 CPU pool rule | #27 | measured, needs the predicate |
| 4 fixed per-token cost | #128 | named: rayon's cold submit |
| 5 x86 decode | #127 | **done**: default was wrong, 6.8x to 1.4x; 1.04x to 1.17x on Zen 2 (2026-09-15) |
| 5b x86 prefill | | 1.0x to 1.4x on the K-quants after #159; Q5_K gate (#257) and IQ4_XS (#258) closed 2026-09-15; small Q8_0 models 1.5x to 2.1x remain |
| 6 kernel coverage | | Q4_K/Q5_K/Q6_K, Q5_0, Q2_K/Q3_K/IQ4_NL/IQ4_XS/MXFP4 on CUDA, verified on hardware 2026-09-15; the kinds still host-only are in [`docs/KERNELS.md`](../KERNELS.md) |
| 7 ledger | #126 | **done**: three hosts, and a committed-receipt check |
