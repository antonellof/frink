# AMD: ROCm GPUs, Ryzen AI NPUs, and every Radeon

What this page is: the plan for running frink on AMD hardware, the
research it rests on, and what each phase can and cannot be verified
with. Dated 2026-11-07. Tracked as #494.

**Status, 2026-11-13.** R1 landed: `frink-rocm` compiles every
`frink-cuda` kernel for every RDNA target through `hiprtc`, held by
CI's `rocm-hiprtc-census` job (no GPU); its launch test has not run on
hardware. Phase V moved on under #409: Vulkan serves a decode and
prefill stack, verified on MoltenVK and lavapipe only. R2, R3 and N
have not started.

## Three kinds of AMD silicon, three different answers

| Silicon | Examples | Path | Why |
|---|---|---|---|
| RDNA GPUs (wave32) | RX 6000/7000/9000, Radeon 890M, Strix Halo's 8060S (gfx1030, gfx1100-1102, gfx1150/1151, gfx1200/1201) | **HIP/ROCm**, reusing `frink-cuda`'s kernel text | llama.cpp's own answer: `ggml-hip` compiles `ggml-cuda`'s sources through a 332-line `vendors/hip.h` shim, adding no AMD kernels of its own. frink's CUDA kernels are runtime-compiled text, so `hiprtc` takes the same text behind a prelude |
| CDNA GPUs (wave64) | MI210, MI300X, MI355X (gfx90a, gfx942, gfx950) | HIP/ROCm, **after a wave64 audit** | 37 sites in `frink-cuda` assume a 32-lane warp (`% 32`, `& 31`, `lane = tid % 32`). RDNA runs them as written; CDNA is wave64 only, so those kernels are refused there by name until each is made width-aware (llama.cpp's `WARP_SIZE` / `ggml_cuda_get_physical_warp_size`) |
| Any AMD GPU without ROCm | APUs on Windows, older Radeons, any Linux box with RADV | **Vulkan** (#409) | ROCm's supported-GPU list is short; Vulkan reaches every Radeon. `frink-vulkan` runs on any driver, including MoltenVK on a Mac |
| Ryzen AI NPU (XDNA2) | Strix, Strix Halo, Kraken, Gorgon Point | **Research spike first** (below) | No GGUF-capable NPU kernel exists in the open |

## The NPU, read against FastFlowLM

[FastFlowLM](https://github.com/ROCm/FastFlowLM) (now under AMD's ROCm
organization) runs LLMs on XDNA2 NPUs through `amdxdna` (Linux 7.0+ or
`amdxdna-dkms`) and AMD's XRT stack. Three facts decide what frink can
take from it:

1. **Its NPU kernels are closed binaries.** The orchestration and CLI
   are MIT (`LICENSE_RUNTIME.txt`); the kernels are free to
   redistribute unmodified but not open (`LICENSE_BINARY.txt`), and
   their interfaces are not documented.
2. **It runs its own model packages**, pulled from Hugging Face as
   pre-built kernel and weight sets per model tag (`flm run
   llama3.2:1b`), not GGUF.
3. The open route is **`mlir-aie` / IRON** (Xilinx, MLIR-based,
   Python front end) compiling AIE kernels to an `xclbin` plus
   instruction stream, loaded through XRT.

So a native frink NPU backend means writing AIE kernels (a quantized
GEMV first) in IRON and loading them through XRT from Rust: a research
project, and one that needs XDNA hardware to verify. Phase N below
scopes it. Driving FastFlowLM's server from frink would not be a
backend at all: it would serve FastFlowLM's models, not the user's
GGUF, and frink's parity guarantee would not cover a single token.

## Phases

| Phase | Deliverable | Verified by | Hardware needed |
|---|---|---|---|
| R1 | `frink-rocm`: HIP prelude, served-target table, `hiprtc` compile of **every** `frink-cuda` kernel for every RDNA target, `dlopen`ed HIP runtime, a Q8_0 matvec launch against its scalar twin | A CI job compiles the whole census in a ROCm container (no GPU); the launch test is `#[ignore]`d | none for the census; an RDNA GPU for the launch |
| R2 | ROCm as a `frink-core` backend: the kernel matrix column, dense decode through the shared dispatch | Kernel-matrix tests; `frink parity` on hardware | RDNA GPU (rental: AMD Developer Cloud, or a Radeon host) |
| R3 | CDNA: each 32-lane assumption made width-aware, the matrix column for gfx942 | The census for gfx942; hardware tests on an MI300 | MI300 |
| V | Vulkan backend (#409), the path for every AMD GPU without ROCm | MoltenVK here; RADV in CI | any Vulkan driver |
| N | NPU spike: detect `amdxdna`, an IRON quantized GEMV, loaded via XRT from Rust, against its scalar twin | A written GO/NO-GO, as the Vulkan beachhead had | an XDNA2 laptop |

The tensor-core `mul_mm` body (`mma.sync`, `ldmatrix`) is NVIDIA-only
and is never compiled for AMD; the generic `mul_mm` serves prefill there
until a WMMA (RDNA3+) or MFMA (CDNA) body is written against
llama.cpp's.

Agents do not rent hardware (CLAUDE.md), so every phase lands with its
hardware tests ignored and the exact commands to run them listed, as
the CUDA kernels did.
