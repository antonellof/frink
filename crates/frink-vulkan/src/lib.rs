//! `frink-vulkan`: the Vulkan **beachhead**, not a backend.
//!
//! # What this crate is
//!
//! One question, answered end to end: *can frink reach a Vulkan
//! device from Rust, upload a quantized weight verbatim, run one
//! compute shader, and read back a correct answer?* That is the
//! `vulkan-beachhead` GO/NO-GO in `docs/plans/archive/roadmap-2026-09.md`'s
//! `d-hardware-reach`, and it is deliberately scoped below a backend:
//! **one kernel, one quant kind, no integration**.
//!
//! It IS now wired into `frink-core`, as of 2026-09-01, once
//! `backend-seam-refactor` landed the seam the verdict asked for:
//! `frink_core::weight_matrix::gpu_backend::Vulkan` is a third
//! `BackendCaps` / `BackendDispatch`, behind `frink-core`'s `vulkan`
//! feature. What it wires up is only what is in here -- ONE Q8_0
//! matvec -- and the seam says so: no GEMM for any kind, no matvec for
//! any other kind, and guard tests that go red if either changes
//! without a shader behind it.
//!
//! That does not promote this crate to a backend. `q8_0_matvec` still
//! rebuilds its entire pipeline per call, weights are still staged
//! rather than imported zero-copy, and nothing here has a measured
//! number. `vulkan-decode-path` and `vulkan-prefill-gemm` are still
//! where a real backend gets decided.
//!
//! # What has actually run
//!
//! See `docs/plans/vulkan-beachhead-verdict.md`. The rule this repo
//! holds CUDA to applies here unchanged: nothing in this crate may be
//! described as a measured capability in `docs/FEATURES.md` or
//! `docs/MODELS.md` on the strength of compiling.
//!
//! # Layout
//!
//! - [`spirv`] -- a minimal SPIR-V word emitter, ~250 lines, no
//!   external shader compiler and no build script.
//! - [`q8_0_shader`] -- the Q8_0 matvec compute shader, emitted from
//!   Rust.
//! - [`q8_0_reference`] -- its scalar twin, checked against
//!   `frink_quant` and the `half` crate.
//! - [`device`] -- the `ash` host layer (`--features vulkan`):
//!   instance, device, buffers, descriptors, dispatch, readback.
//!
//! The first three compile and are tested **unconditionally**, on a
//! machine with no Vulkan driver at all. A broken shader fails
//! `cargo test -p frink-vulkan` with no GPU, which is strictly more
//! than `frink-cuda` can say for its kernels (NVRTC compiles at
//! runtime).

pub mod q8_0_reference;
pub mod q8_0_shader;
pub mod spirv;

#[cfg(feature = "vulkan")]
pub mod device;

#[cfg(feature = "vulkan")]
pub mod dispatch;
