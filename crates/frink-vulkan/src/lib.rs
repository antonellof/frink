//! `frink-vulkan`: decode matvecs on any Vulkan GPU.
//!
//! # What this crate is
//!
//! The Vulkan half of `frink_core::weight_matrix::gpu_backend::Vulkan`
//! (#409): a matvec per quant kind for decode, behind `frink-core`'s
//! `vulkan` feature, so a K-quant, legacy, IQ4 or MXFP4 model's
//! matvecs can run on any Vulkan device, including the AMD and Intel
//! GPUs that ROCm and CUDA do not reach. Weights are uploaded once and
//! stay resident; pipelines are built once.
//!
//! What it is not, yet, is a backend in the sense Metal and CUDA are:
//! there is no GEMM (a prefill stays on the host), no attention, and
//! each projection (or each fused Q/K/V and gate/up set) still waits
//! for its own result, so a layer pays several queue round trips. It began as the `vulkan-beachhead`
//! GO/NO-GO (`docs/plans/archive/vulkan-beachhead-verdict.md`), and its seam
//! guard tests still go red if a kind is claimed without a shader.
//!
//! # What has actually run
//!
//! See `docs/plans/archive/vulkan-beachhead-verdict.md`. The rule this repo
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

/// The environment variable that turns a test's "no device, skipped"
/// into a failure.
pub const REQUIRE_ENV: &str = "FRINK_REQUIRE_VULKAN";

/// Called by a test that could not run because something it needs (a
/// device, `spirv-val`) is missing. Without `FRINK_REQUIRE_VULKAN=1`
/// it says so on stderr and the test passes, which keeps `cargo test`
/// free on a driverless machine. With it set, as CI's lavapipe job
/// does, it panics: a job that exists to run these tests must not go
/// green by skipping them.
pub fn skipped(why: &str) {
    if std::env::var(REQUIRE_ENV).is_ok_and(|v| v == "1") {
        panic!("{REQUIRE_ENV}=1 but {why}");
    }
    eprintln!("SKIPPED: {why}");
}

#[cfg(feature = "vulkan")]
pub mod device;

#[cfg(feature = "vulkan")]
pub mod dispatch;

#[cfg(feature = "vulkan")]
pub mod runtime;
