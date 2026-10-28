# The Vulkan beachhead: GO

[`archive/roadmap-2026-09.md`](archive/roadmap-2026-09.md)'s `d-hardware-reach` asks for "`vulkan-beachhead` as a
GO/NO-GO on the smallest honest slice (`ash` plus one SPIR-V kernel),
not a full backend", and says a written verdict is the deliverable.
This is that verdict. Dated 2026-09-01.

**The answer is GO**, on the narrow question actually asked, with the
costs below written down so nobody has to rediscover them. The question
was *can frink reach a Vulkan device from Rust, upload a quantized
weight verbatim, run one compute shader, and read back a correct
answer?* It can, and it did.

**GO on reachability is not GO on the backend.** The full Vulkan
backend still has to clear `vulkan-prefill-gemm`, which is where the
archived `amd-strix-halo` plan already put the real decision. Nothing
here is a performance claim; no number was measured, deliberately.

## What ran

```
$ cargo run -p frink-vulkan --features vulkan --example probe
vulkan device: Apple M2 Pro (Vulkan 1.0.357)

$ cargo test -p frink-vulkan --features vulkan
test dispatch::tests::gpu_matvec_matches_the_scalar_twin ... ok
test result: ok. 17 passed; 0 failed
```

A Q8_0 matvec, `y[row] = dot(dequant(W[row]), x)`, as a hand-emitted
SPIR-V compute shader, executed on the GPU through MoltenVK, at four
shapes: `(rows, blocks) = (1,1), (9,3), (65,4), (200,11)`. Those cover a
row stride that is not 4-byte aligned (`3 * 34 = 102`), a row count that
is not a multiple of the 64-wide workgroup, and a dispatch spanning
several workgroups. Every output matched the scalar twin within 1e-4
relative, the same tolerance `frink-cuda`'s hardware test uses and for
the same reason: a GPU is free to contract `acc + a * b` into an FMA.

Held to this repo's standard for a GPU kernel, which is a scalar twin
checked against an independent reference:

| Claim | Checked by | Status |
|---|---|---|
| The f16 scale decode is right | `half` crate, **all 65,536** bit patterns | exact |
| The twin is right | `frink_quant::dequant_q8_0` + a plain dot | **bit-exact**, 4 shapes |
| The emitted module is legal SPIR-V | `spirv-val --target-env vulkan1.0` | accepted |
| The shader agrees with the twin | a real GPU | 1e-4 relative, 4 shapes |
| The tests can fail | sabotage | see below |

Two sabotages were run and both went red where they should:

- Deleting the `(k & 3) * 8` byte-shift in the twin reddened the two
  matvec tests and left the alignment-independent ones green.
- Reading the quantized bytes one byte early **in the shader only**
  reddened `gpu_matvec_matches_the_scalar_twin` and nothing else --
  `spirv-val` still passed the module. A validator proves legality, not
  correctness; only the twin catches a valid shader that computes the
  wrong thing.

### The host this ran on, stated plainly

An Apple M2 Pro, which had **no Vulkan anything** at the start of this
work: no loader, no MoltenVK, no ICD manifest, no `glslangValidator`,
`VULKAN_SDK` unset. `brew install molten-vk vulkan-loader glslang`
(all bottled) was enough, and after it the Homebrew loader finds the
ICD with no `VK_ICD_FILENAMES` needed.

That is itself a finding: **the roadmap's re-scoping is correct and
then some.** The four Vulkan items were filed as blocked on owning a
Strix Halo box. They are not blocked on an Intel iGPU either. They are
runnable on the M2 Pro that is already the only measurement host this
project has, after a three-formula install.

## The costs, concretely

### 1. `ash` is a sane dependency. It is the only one added.

`ash` is a thin FFI binding: no build script, no bindgen, no vendored
C++. With `default-features = false, features = ["loaded", "std"]` it
opens `libvulkan` with `dlopen` at runtime, so **no Vulkan SDK, headers
or linker flags are needed to build**, exactly the shape `frink-cuda`
chose for cudarc. `cargo build --features vulkan` succeeds on a machine
with no driver.

`Cargo.lock` grew by exactly one third-party crate, `ash` (`libloading`
was already in the tree). Compare `vulkano`, which layers a large safe
abstraction and would decide frink's resource model for it, and
`wgpu`, which is a second graphics abstraction entire. The roadmap
named `ash` and the roadmap was right.

One portability trap, worth writing down because falling into it looks
exactly like a NO-GO: **MoltenVK is invisible without opting in.**
Without `VK_KHR_portability_enumeration` and the
`ENUMERATE_PORTABILITY_KHR` instance flag, `vkEnumeratePhysicalDevices`
succeeds and returns zero devices on macOS. "No Vulkan device on this
Mac" is the wrong conclusion and is one line of code away.

### 2. Shaders can be built with no C++ toolchain. It is not free.

There are three ways to get SPIR-V into a Rust binary:

1. **`glslangValidator` / `glslc` in `build.rs`.** What llama.cpp does
   (it builds a `vulkan-shaders-gen` C++ program at configure time).
   Makes a C++ toolchain a build prerequisite for every frink user who
   enables the feature, forever. `frink-cuda` deliberately refused the
   equivalent.
2. **Commit pre-built `.spv` blobs.** No build dependency, but the repo
   carries opaque binaries no reviewer can read, which drift from
   whatever source produced them.
3. **Emit the words from Rust.** No build step at all, ordinary
   reviewable Rust, testable with no GPU.

The beachhead took option 3 and it works: `crates/frink-vulkan/src/spirv.rs`
is a ~250-line word emitter and the shader is ~560 lines of Rust that
produce an 801-word (3,204-byte) module. `spirv-val` accepts it. The
whole SPIR-V half compiles and is tested **unconditionally**, on a
machine with no driver — which is strictly more than `frink-cuda` can
say, since NVRTC compiles at runtime.

**And it does not scale.** ~560 lines of Rust per kernel, for the
*simplest possible* kernel: one invocation per row, no subgroup
reduction, no shared-memory tiling, no integer dot. `frink-metal` is
19,881 lines with roughly 180 kernel definitions. Hand-emission at this
rate is a six-figure line count and is not a serious proposal.

So the recommendation for a real backend is **option 3 for the
beachhead, a shader source language for the backend** — and the choice
of language is a real decision, not a formality:

- **GLSL via `glslangValidator` in `build.rs`** matches llama.cpp and
  makes shader authoring cheap, at the price of a C++ build dependency
  this repo has refused twice.
- **WGSL via `naga`**, which is pure Rust, keeps "no C++ at build time"
  and is the only option that does. It costs a large dependency and a
  shading language nobody here writes.
- **`rust-gpu`** compiles Rust to SPIR-V and would let the kernel and
  its scalar twin be *the same source*, which is the strongest form of
  the twin discipline this repo has. It is also the least mature.

`ci-x86-and-vulkan` should be read in light of this: a `--features
vulkan` compile-only job is genuinely more valuable than the CUDA
equivalent, because a broken shader fails at build or test time with no
GPU, and `spirv-val` is a real external oracle that has no CUDA analogue.

### 3. ggml block sizes are not word multiples, and that has teeth

A Q8_0 block is **34 bytes** — an f16 scale plus 32 int8s — and 34 is
not a multiple of 4. A row of `n` blocks is `34n` bytes, 4-byte aligned
only when `n` is even.

SPIR-V's logical addressing has no byte pointer. A storage buffer is an
array of some type, and the smallest type available without the `Int8` /
`StorageBuffer8BitAccess` capabilities — which are optional, and
optional is precisely the wrong bet for the old Intel and Android
devices this whole theme exists to reach — is `uint`. So **every byte of
every weight** is read as `(w[k >> 2] >> ((k & 3) * 8)) & 0xff`, and the
f16 scale is decoded with an integer `OpSelect` chain rather than a
hardware `float16_t`.

This is not a Q8_0 quirk. Q4_0 is 18 bytes, Q4_K is 144, Q6_K is 210.
*Any* Vulkan backend for frink either does this byte extraction
everywhere, or repacks weights on upload — and repacking gives up the
zero-copy-from-mmap property that `amd-strix-halo` built its entire UMA
argument on. llama.cpp's Vulkan backend takes the extraction road. So
did this. The cost is real and is now measured rather than assumed:
seven integer ops per weight byte, in a decode kernel that is already
memory-bound. On a device that *does* have `Int8` and
`shaderFloat16`, a second specialization would drop most of it, which
is a real optimization axis a backend must plan for and the beachhead
deliberately did not take.

### 4. Zero-copy residency is UNPROVEN, and it is the biggest open risk

The beachhead **copies** weights into host-visible device memory. The
zero-copy import the archived plan wants —
`VK_EXT_external_memory_host`, importing the GGUF mmap directly, as
`frink-metal`'s `register_weight_mmap` / `BytesNoCopy` does — was not
attempted. It is an optional extension, it needs page-aligned host
pointers, and it is a property of a backend rather than of a GO/NO-GO.

That plan already flagged this as the thing to test first inside
`vulkan-beachhead`, and it was not tested. It should be the first item
of `vulkan-decode-path`, ahead of any kernel: on a unified-memory box a
staging copy doubles the footprint, which is one of the two things
Strix Halo punishes.

### 5. How much of `frink-metal` transfers

More of the *structure* than of the *code*, and none of the kernels.

Transfers: the seam shape (`WeightMatrix` hands a `&[u8]` of mmapped
GGUF bytes and the backend decides residency), the per-kind capability
table, the fused-launch entry points (`apply_gpu_multi`,
`apply_gpu_dense_ffn_swiglu`), the `mul_mm_sg_impl` *tiling* argument,
and every unpack function — which are themselves llama's
`dequantize_q*` and were already ported once, Metal to CUDA.

Does not transfer: simdgroup intrinsics (`simdgroup_multiply_accumulate`
has no portable Vulkan equivalent; `VK_KHR_cooperative_matrix` exists
but is optional and newer than the devices this targets), the Metal
Shading Language sources themselves, `MTLBuffer`-based residency, and
runtime shader compilation — Metal compiles MSL at runtime, Vulkan
consumes pre-built SPIR-V, which is *better* for testing and worse for
specialization.

Realistic size of a full Vulkan backend, extrapolating from
`frink-metal`'s 19,881 lines: **the same order, 15,000 to 25,000
lines**, plus the shader-language decision above. That is an XL item and
the roadmap already calls it one.

## What this changes in the plan

1. **`vulkan-beachhead` is GO and can be closed**, on the terms it set:
   a written verdict on the smallest honest slice. The acceptance
   criterion it also asked for — "a NUMBER" against a tuned-AVX-512 CPU
   baseline — **cannot be met yet and should move to
   `vulkan-decode-path`**, because `x86-first-measurement` has not run
   and there is no tuned CPU baseline to compare against. A number from
   a kernel that rebuilds its pipeline every call would be worse than no
   number.
2. **The four Vulkan items lose their hardware block entirely.** Not
   "testable on an Intel iGPU" — testable on the M2 Pro that is already
   Host B, via MoltenVK, after `brew install molten-vk vulkan-loader`.
3. **`backend-seam-refactor` is still first, and this crate is
   deliberately not wired in.** `frink-vulkan` has no caller.
   `WeightMatrix::apply_gpu` knows two backends, and adding a third to
   the seam as it stands means a third copy of four hand-kept tables
   with mismatched signatures, plus a share of 214 backend `#[cfg]`
   sites across 26 files. The survey below is what that item has to fix.
4. **Zero-copy residency moves to the front of `vulkan-decode-path`**,
   per §4.

## Appendix: the seam a third backend needs

Surveyed 2026-09-01 against `crates/frink-core/src/weight_matrix.rs`,
which this pass did not modify.

There is no backend trait. `kernel_registry::Backend`
(`kernel_registry.rs:73`) is observability-only by explicit design
("Nothing in this module may change a dispatch decision"). Dispatch is
literal `#[cfg(feature = "...")]` blocks in sequence: CUDA arm, Metal
arm, CPU fallthrough.

Four things exist once per backend, and a third backend copies all four:

| Axis | Metal | CUDA |
|---|---|---|
| Kind → matvec kernel | `metal_matvec_kind_name` (`:398`), returns `Option<&'static str>`, 7 kinds | `cuda_matvec_kind_supported` (`:522`), returns `bool`, 5 kinds |
| Kind → batched GEMM | `metal_mul_mm_kind_supported` (`:419`), 7 kinds | `cuda_mul_mm_kind_supported` (`:509`), 2 kinds |
| Launch fn alias | `MetalMatvecLaunchFn` (`:570`), **4 args** | `CudaMatvecLaunchFn` (`:564`), **5 args** |
| Enable probe | `metal_dense_enabled` (`:596`) | `cuda_dense_enabled` (`:612`), byte-identical body |

Those four are then re-selected by hand at six call sites: `apply_gpu`
(`:2861`, two near-identical `match kind` tables), `apply_gpu_multi`
(`:2975`), `apply_gpu_dense_ffn_swiglu` (`:3109`),
`apply_batch_with_acts` (`:2002` / `:2062`), `quantize_batch_acts`
(`:1888` / `:1899`), and `apply_gpu_batch` (`:3251`) — which is
`#[cfg(feature = "metal")]`-only, so CUDA's batch path lives inline in
`apply_batch_with_acts` instead of sharing that entry point. The
asymmetry is already load-bearing.

Outside the two backend crates there are **155** `feature = "metal"`
cfg sites and **59** `feature = "cuda"` sites across **26 files**;
`decoder.rs` alone holds 77. `backend-seam-refactor`'s acceptance
criterion is "adding a stub backend touches ≤ 5 files". It is currently
missed by 5x.

What the seam has to become:

1. **One launch signature.** Unify on the wider CUDA arity —
   `fn(&[u8], &[f32], rows, row_bytes, n_blocks_per_row) -> Result<Vec<f32>, E>`
   — with one error type the backend errors convert into. The arity
   difference alone is what forces two dispatch tables that cannot be
   written as one. `frink-vulkan`'s `q8_0_matvec` already takes exactly
   this argument list, on purpose.
2. **One capability table per backend behind one shape.** A trait whose
   four members are `matvec_kernel`, `gemm_supported`, `dense_enabled`,
   `backend_id` — all associated functions with no receiver, which is
   what the current free functions already are, so this is a mechanical
   lift rather than a redesign. The `debug_assert_eq!` at `:2933` that
   guards `apply_gpu`'s Metal launch table against
   `metal_matvec_kind_name` becomes structurally impossible to violate
   instead of merely asserted.
3. **One ordered backend list**, so "CUDA first, then Metal, then CPU"
   and `active_backend()` (`:539`) stop being two hand-kept copies of
   the same precedence.
4. **`Backend`'s variants come from that same list**, so a backend
   cannot be dispatched to without being reportable.

The one thing that does **not** need to change for Vulkan: both existing
backends already take a `&[u8]` of mmapped GGUF bytes and decide
residency inside the backend crate — Metal wraps zero-copy, CUDA copies,
Vulkan would do one or the other per §4. The `&[u8]` seam accommodates
all three.
