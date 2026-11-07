//! `frink-vulkan`: decode matvecs on any Vulkan GPU.
//!
//! # What this crate is
//!
//! The Vulkan half of `frink_core::weight_matrix::gpu_backend::Vulkan`
//! (#409): a matvec per quant kind for decode, behind `frink-core`'s
//! `vulkan` feature, so a Q4_K_M, Q5_K_M, Q6_K, Q4_0 or Q8_0 model's
//! matvecs can run on any Vulkan device, including the AMD and Intel
//! GPUs that ROCm and CUDA do not reach. Weights are uploaded once and
//! stay resident; pipelines are built once.
//!
//! What it is not, yet, is a backend in the sense Metal and CUDA are:
//! there is no GEMM (a prefill stays on the host), no attention, and
//! each matvec still waits for its own result, so a token pays one
//! queue round trip per projection. It began as the `vulkan-beachhead`
//! GO/NO-GO (`docs/plans/vulkan-beachhead-verdict.md`), and its seam
//! guard tests still go red if a kind is claimed without a shader.
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
//! - [`shaders`] -- the GLSL matvec shaders' table and their committed
//!   SPIR-V (`shaders/`), validated and drift-checked with no GPU.
//! - [`device`] -- the `ash` host layer (`--features vulkan`): loader,
//!   instance, device and compute queue.
//! - [`runtime`] -- the matvec state kept across calls: pipelines,
//!   resident weights, one command buffer.
//! - [`dispatch`] -- one typed entry point per kind, each held against
//!   its `frink_quant` scalar twin on a real device.

pub mod shaders;

#[cfg(feature = "vulkan")]
pub mod device;

#[cfg(feature = "vulkan")]
pub mod dispatch;

#[cfg(feature = "vulkan")]
pub mod runtime;
