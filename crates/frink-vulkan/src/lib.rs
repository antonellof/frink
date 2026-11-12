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
//! A decode token of a plain llama-family model runs 26 layers per
//! submission (`frink_models::decoder::vulkan_stack`, over
//! [`runtime::ops`]): norms, RoPE, an f16 KV and attention included.
//! What it is not, yet, is a backend in the sense Metal and CUDA are:
//! there is no GEMM (a prefill stays on the host), and a layer the
//! stack declines runs on the host with its matvecs offloaded, each
//! submission (Q/K/V together, the output projection, and a dense
//! SwiGLU FFN as one, [`runtime::Runtime::dense_ffn_swiglu`]) waiting
//! for its own result, about three queue round
//! trips. It began as the `vulkan-beachhead`
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
//!   resident weights, one command buffer; the fused FFN in
//!   `runtime/ffn.rs`; the layer ops a device-resident decode chains
//!   (RMSNorm, add, RoPE, f16 KV append, attention) in [`runtime::ops`].
//! - [`dispatch`] -- one typed entry point per kind, each held against
//!   its `frink_quant` scalar twin on a real device.

pub mod select;
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
