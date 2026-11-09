# 0009. Numerics follow ggml unless a deviation is written down

Status: Accepted
Date: 2026-10-28

## Context

"Same tokens" depends on more than the graph. Frink has diverged from
llama.cpp through choices that each looked harmless: an f32 host KV
cache where llama.cpp keeps f16, a repetition penalty applied per
occurrence instead of once per candidate, temperature applied before
the truncation filters, and penalties that did not see the prompt over
HTTP. Each produced different tokens from the same file and seed.

An activation quantizer is a staircase, so "within an ulp" does not
carry through it: an input one ulp from a rounding boundary moves a
whole quant step. #476 found frink's `Q8_K` and `Q8_0` quantizers an
ulp (and, for `Q8_0`'s f16 scale, more) away from ggml's, and the RMS
norm in front of them summing in f32 where ggml sums in f64. Fed
llama.cpp's own dumped inputs, frink's Q8_0 matmuls then differed from
libllama's by up to 2e-3 per op; with all three matched they are
bit-identical, and layer 0 up to the attention is bit-identical on
Llama-3.2-1B Q8_0 and gpt-oss-20b. First-token KL could not show
this: past the first quant that flips, the per-layer distance grows to
what two llama.cpp builds measure against each other, whichever form
frink uses.

## Decision

- Quant block layouts, dequantisation and the activation quantisation
  used by integer dots match ggml bit for bit, per ISA where ggml's is
  (`quantize_row_q8_0` has a NEON, an AVX and a scalar body).
- The RMS norm, the input of every layer's first quantizer, follows
  `ggml_compute_forward_rms_norm_f32`'s arithmetic: squares summed in
  order in f64.
- KV cache precision defaults to llama.cpp's (f16), and the sampler
  chain runs in llama.cpp's order with its semantics.
- Tokenisation follows `llama-vocab.cpp` at the pin, including its
  pre-tokenizer regexes.
- Accumulation order and fused kernels may differ when the result stays
  inside the reference spread `frink parity` measures between two
  llama.cpp builds.
- Any other deviation is listed below, with its reason and measured
  effect. An unlisted deviation is a bug.

## Consequences

Some optimisations that change rounding are off the table unless they
stay inside the reference spread. A user switching from llama.cpp gets
the same output at the same seed.

`frink parity --dump-tensors` compares the two engines tensor by
tensor. A numerics change is judged by where the first difference
enters and against the spread between two llama.cpp builds, not by one
first-token KL.

## Known deviations

Measured with `frink parity --dump-tensors` against libllama at the
pin, CPU, BLAS off, on an M2 Pro (#476).

| Where | llama.cpp | frink | Measured |
|---|---|---|---|
| CPU attention, under 64 query rows and in decode | flash attention rounds Q to f16; on ARM the K·Q dot and the V accumulator run in f16 (`vfmaq_f16`) | Q and both accumulators in f32 over the f16 cache | `kqv_out-0` 8e-4 relative on Llama-3.2-1B Q4_K_M, 6-token prompt, with Q/K/V matching to 2e-7. From 64 rows llama.cpp takes a tiled f32 path and the step is gone |
| RoPE | `ggml_rope_cache_init`'s angles | frink's own | 2-5e-7 relative on bit-identical Q/K |
| K-quant and MXFP4 GEMMs | the build's repack kernel (`q4_K_8x4` portable, `8x8` with i8mm) | its own reduction order | from bit-identical inputs, Q4_K rows 1.1e-7 relative (median), what the two llama.cpp builds measure against each other; Q6_K rows 6e-8, where the two builds agree bit for bit. Q8_0 rows are bit-identical |
