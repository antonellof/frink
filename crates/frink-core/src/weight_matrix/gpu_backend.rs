//! The seam every GPU backend goes through.
//!
//! Four things used to exist once per backend as free functions in
//! `weight_matrix.rs`, with no shape holding them together:
//!
//! | Axis | Metal | CUDA |
//! |---|---|---|
//! | kind → matvec | `metal_matvec_kind_name` → `Option<&str>` | `cuda_matvec_kind_supported` → `bool` |
//! | kind → GEMM | `metal_mul_mm_kind_supported` | `cuda_mul_mm_kind_supported` |
//! | launch alias | `MetalMatvecLaunchFn`, 4 args | `CudaMatvecLaunchFn`, 5 args |
//! | enable probe | `metal_dense_enabled` | `cuda_dense_enabled`, byte-identical body |
//!
//! and they were re-selected by hand at every dispatch site, so
//! `apply_gpu` carried two near-identical `match kind` tables that could
//! only be kept honest by a `debug_assert!`. A third backend would have
//! copied all four. This module is the shape they now share.
//!
//! # What a third backend has to provide
//!
//! Exactly this, and nothing else:
//!
//! 1. A unit type (`pub struct Vulkan;`).
//! 2. [`BackendCaps`] — the two capability tables plus an id and a
//!    display name. **Compiled unconditionally**, with no dependency on
//!    the backend crate, because the tables are a property of the kernel
//!    set rather than of the build, and gating them would make them
//!    untestable on the CPU builds that run `cargo test --workspace`.
//!    This is the rule that kept `metal_matvec_kind_name` un-`cfg`'d and
//!    it is load-bearing: `probe_kernels_for` asks what Metal *would*
//!    resolve from a build with no Metal.
//! 3. [`BackendDispatch`] under `#[cfg(feature = "…")]` — the enable
//!    probe, a device-free launch table, and one launch entry point.
//! 4. One line in [`gpu_backend_table`], which is **the** ordered list:
//!    enum variant, cargo feature / registry name, and seam type, once.
//!
//! `Vulkan` is what that recipe looks like when it is followed: a
//! one-kernel backend (Q8_0 matvec, no GEMM, no batch path) added
//! without touching `apply_gpu` or `active_backend`.
//!
//! # The one list, and its three consumers
//!
//! [`gpu_backend_table`] is expanded by exactly three things, so a
//! backend cannot exist in one of them and not the others:
//!
//! - [`crate::kernel_registry::Backend`]'s variants and their names.
//!   This is verdict point 4 — "a backend cannot be dispatched to
//!   without being reportable" — and it is now structural rather than
//!   hand-kept. It works because the registry and this module are the
//!   same crate; a `macro_rules!` cannot generate an enum in a
//!   *different* crate, so a backend crate could never own its own
//!   variant.
//! - [`with_gpu_backends`], the `#[cfg]`-gated dispatch order, which
//!   [`crate::weight_matrix::WeightMatrix::apply_gpu`] and
//!   [`crate::weight_matrix::active_backend`] both expand.
//! - [`with_gpu_backend_caps`], the **ungated** one, which
//!   `probe_kernels_for` expands so a CPU-only build can still ask what
//!   Metal or Vulkan *would* resolve.
//!
//! # The launch signature, and the fifth argument
//!
//! `frink-cuda`'s `launch_*_matvec` takes
//! `(weights, x, rows, row_bytes, n_blocks_per_row)`; `frink-metal`'s
//! takes the first four. [`BackendDispatch::launch_matvec`] takes the
//! four plus the [`QuantKind`], and every backend derives the rest.
//!
//! That is deliberate, and it is not the wider arity the beachhead
//! verdict proposed. `n_blocks_per_row` is **redundant information**,
//! not missing information: it is `row_bytes / block_bytes(kind)`, and
//! Metal already recomputes exactly that inside
//! `frink_metal::gpu::matvec_launch_meta`, which hands back the block
//! size for the kind. Hoisting it into the shared signature would buy a
//! backend nothing it cannot derive, and would cost a
//! `block_bytes(kind)` that is total over all 21 `QuantKind`s — the
//! existing one, `WeightMatrix::block_bytes_for_kind`, is deliberately
//! partial and `unreachable!()`s outside the five CUDA kinds, and
//! Metal's `IQ4_XS` is not one of them. So the seam passes the kind and
//! lets each backend ask its own table.

use crate::kernel_registry::Backend;
use crate::weight_matrix::QuantKind;

/// A backend launch failure, flattened to its rendered message.
///
/// `frink_metal::gpu::MetalError` and `frink_cuda::gpu::CudaError` are
/// different types living behind different features, and the only thing
/// any caller does with either is print it before falling back — so the
/// seam carries the message rather than an enum that would have to grow
/// a variant per backend crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError(String);

impl BackendError {
    /// Renders any backend error into the one shape the seam carries.
    pub fn new(e: impl std::fmt::Display) -> Self {
        BackendError(e.to_string())
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a backend can run, asked without the backend crate present.
///
/// Every member is an associated function with no receiver, which is
/// what the free functions this replaced already were, so implementing
/// it is a lift rather than a redesign.
pub trait BackendCaps {
    /// How [`crate::kernel_registry`] reports this backend. Dispatch and
    /// observability read the same constant, so a backend cannot be
    /// dispatched to under one name and reported under another.
    const ID: Backend;

    /// Human-readable name, for the one message a dispatch failure
    /// prints.
    const NAME: &'static str;

    /// What a batched prefill actually runs on for a kind this backend
    /// has a matvec but no GEMM for — the string
    /// `probe_kernels_for` records as the fallback.
    ///
    /// It lives here because it is a property of the backend and it was
    /// previously a `match backend` arm inside `probe_kernels_for`: a
    /// third `Backend` variant would have silently inherited Metal's
    /// wording (`"Metal N x matvec batch"`) by falling through `_`, and
    /// the registry's whole job is to name the path that will really
    /// run. Ungated, like the rest of [`BackendCaps`], because the
    /// probe is ungated.
    const GEMM_FALLBACK: &'static str;

    /// Which quant kinds have a **matvec** kernel (the decode path), as
    /// the kernel name the backend's own launch-meta table is keyed by,
    /// or `None` for a kind with no kernel.
    ///
    /// Returning the name rather than a `bool` is what let the two
    /// backends share one member: CUDA only ever needed the `bool`
    /// (`.is_some()`), Metal needs the name to look up
    /// `frink_metal::gpu::matvec_launch_meta`, and a `bool` cannot be
    /// widened after the fact without another table.
    fn matvec_kernel(kind: QuantKind) -> Option<&'static str>;

    /// Which quant kinds have a **batched GEMM** (the prefill path). A
    /// kind with a matvec but no GEMM still runs on the accelerator — as
    /// `batch` separate matvecs over the same weights, which is the
    /// 13.7x shape, and which is why these are two predicates and not
    /// one.
    fn gemm_supported(kind: QuantKind) -> bool;
}

/// What a backend can actually do, which needs its crate compiled in.
pub trait BackendDispatch: BackendCaps {
    /// What [`crate::weight_matrix::WeightMatrix::apply_gpu`] will do
    /// next if this backend's launch fails, named for the log line.
    /// A property of this backend's position in
    /// [`with_gpu_backends`], not of the backend itself.
    const MATVEC_FALLBACK: &'static str;

    /// Whether dense matmuls should try this backend in this process.
    /// Decided once from the environment, then cached for the process
    /// lifetime. See `env_or_probe` for the grammar, which is shared.
    fn dense_enabled() -> bool;

    /// Whether a launch **function** exists for `kind`, asked without a
    /// device and without launching anything.
    ///
    /// This is the question [`BackendCaps::matvec_kernel`] claims to
    /// answer, asked of the code that actually runs. They are two
    /// structures that must agree about one thing, and they cannot be
    /// merged: the kernel NAMES are needed on builds where the backend
    /// crate is not a dependency and these function pointers do not
    /// exist. So the agreement is a test —
    /// `every_kind_a_compiled_backend_claims_can_actually_be_launched`,
    /// over every backend and all 23 kinds — rather than the
    /// `debug_assert!` that used to guard it, which fired only for
    /// kinds a run actually reached and only in debug.
    ///
    /// `Q5_0` is why. It was in Metal's capability table with no launch
    /// function behind it from the day it was added, so batched prefill
    /// ran on the GPU while single-token decode silently fell to the
    /// CPU, and a release build just ran slower.
    fn has_launch(kind: QuantKind) -> bool;

    /// One matvec. `None` means "this backend has no kernel for `kind`",
    /// which is a different answer from `Some(Err(_))`, "the kernel
    /// exists and the launch failed" — the caller logs only the second.
    fn launch_matvec(
        kind: QuantKind,
        weights: &[u8],
        x: &[f32],
        rows: usize,
        row_bytes: usize,
    ) -> Option<Result<Vec<f32>, BackendError>>;
}

/// The per-kind Metal launch table, split out of
/// [`BackendDispatch::launch_matvec`] so it can be checked for EVERY
/// kind without a device.
///
/// It has to agree with [`Metal::matvec_kernel`], and it cannot be the
/// same table: the kernel NAMES are needed on builds where
/// `frink-metal` is not a dependency and these function pointers do not
/// exist. So the agreement is asserted, and asserting it only inside
/// `launch_matvec` was not enough -- that fires just for kinds a run
/// actually reaches, in debug. `Q5_0` was in the capability table and
/// missing here from the day it was added, and the symptom was
/// single-token decode silently falling to the CPU while batched
/// prefill ran on the GPU.
#[cfg(feature = "metal")]
fn metal_matvec_launch(kind: QuantKind) -> Option<MetalMatvecLaunchFn> {
    match kind {
        QuantKind::Q8_0 => Some(frink_metal::gpu::launch_q8_0_matvec),
        QuantKind::Q4_0 => Some(frink_metal::gpu::launch_q4_0_matvec),
        QuantKind::Q4K => Some(frink_metal::gpu::launch_q4_k_matvec),
        QuantKind::Q5_0 => Some(frink_metal::gpu::launch_q5_0_matvec),
        QuantKind::Q5K => Some(frink_metal::gpu::launch_q5_k_matvec),
        QuantKind::Q6K => Some(frink_metal::gpu::launch_q6_k_matvec),
        QuantKind::IQ4XS => Some(frink_metal::gpu::launch_iq4_xs_matvec),
        QuantKind::Ptq1_0 => Some(frink_metal::gpu::launch_ptq1_0_matvec),
        QuantKind::Q2K => Some(frink_metal::q2_k::launch_q2_k_matvec),
        QuantKind::Q3K => Some(frink_metal::q3_k::launch_q3_k_matvec),
        QuantKind::IQ4NL => Some(frink_metal::codebook4::launch_iq4_nl_matvec),
        QuantKind::Mxfp4Gguf => Some(frink_metal::codebook4::launch_mxfp4_matvec),
        QuantKind::IQ2XXS => Some(frink_metal::iq::launch_iq2_xxs_matvec),
        QuantKind::IQ2XS => Some(frink_metal::iq::launch_iq2_xs_matvec),
        QuantKind::IQ2S => Some(frink_metal::iq::launch_iq2_s_matvec),
        QuantKind::IQ3XXS => Some(frink_metal::iq::launch_iq3_xxs_matvec),
        QuantKind::IQ3S => Some(frink_metal::iq::launch_iq3_s_matvec),
        QuantKind::IQ1S => Some(frink_metal::iq::launch_iq1_s_matvec),
        QuantKind::IQ1M => Some(frink_metal::iq::launch_iq1_m_matvec),
        QuantKind::Q4_1 => Some(frink_metal::dq_mv::launch_q4_1_matvec),
        QuantKind::Q5_1 => Some(frink_metal::dq_mv::launch_q5_1_matvec),
        QuantKind::Tq1_0 => Some(frink_metal::dq_mv::launch_tq1_0_matvec),
        _ => None,
    }
}

/// The per-kind CUDA launch table, split out of
/// [`BackendDispatch::launch_matvec`] for the same reason
/// [`metal_matvec_launch`] was: so
/// [`BackendDispatch::has_launch`] can check it against
/// [`Cuda::matvec_kernel`] for EVERY kind, without a GPU. It was inline
/// in `launch_matvec` and therefore had no guard at all — the hole that
/// cost Metal a Q5_0 decode path.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_matvec_launch(kind: QuantKind) -> Option<CudaMatvecLaunchFn> {
    match kind {
        QuantKind::Q8_0 => Some(frink_cuda::gpu::launch_q8_0_matvec),
        QuantKind::Q4_0 => Some(frink_cuda::gpu::launch_q4_0_matvec),
        QuantKind::Q5_0 => Some(frink_cuda::gpu::launch_q5_0_matvec),
        QuantKind::Q2K => Some(frink_cuda::gpu::launch_q2_k_matvec),
        QuantKind::Q3K => Some(frink_cuda::gpu::launch_q3_k_matvec),
        QuantKind::Q4K => Some(frink_cuda::gpu::launch_q4_k_matvec),
        QuantKind::Q5K => Some(frink_cuda::gpu::launch_q5_k_matvec),
        QuantKind::Q6K => Some(frink_cuda::gpu::launch_q6_k_matvec),
        QuantKind::IQ4NL => Some(frink_cuda::gpu::launch_iq4_nl_matvec),
        QuantKind::IQ4XS => Some(frink_cuda::gpu::launch_iq4_xs_matvec),
        QuantKind::Mxfp4Gguf => Some(frink_cuda::gpu::launch_mxfp4_matvec),
        QuantKind::Q4_1 => Some(frink_cuda::gpu::launch_q4_1_matvec),
        QuantKind::Q5_1 => Some(frink_cuda::gpu::launch_q5_1_matvec),
        QuantKind::Tq1_0 => Some(frink_cuda::gpu::launch_tq1_0_matvec),
        QuantKind::Ptq1_0 => Some(frink_cuda::gpu::launch_ptq1_0_matvec),
        QuantKind::IQ2XXS => Some(frink_cuda::gpu::launch_iq2_xxs_matvec),
        QuantKind::IQ2XS => Some(frink_cuda::gpu::launch_iq2_xs_matvec),
        QuantKind::IQ2S => Some(frink_cuda::gpu::launch_iq2_s_matvec),
        QuantKind::IQ3XXS => Some(frink_cuda::gpu::launch_iq3_xxs_matvec),
        QuantKind::IQ3S => Some(frink_cuda::gpu::launch_iq3_s_matvec),
        QuantKind::IQ1S => Some(frink_cuda::gpu::launch_iq1_s_matvec),
        QuantKind::IQ1M => Some(frink_cuda::gpu::launch_iq1_m_matvec),
        _ => None,
    }
}

/// The per-kind Vulkan launch table. One row, and the guard test is
/// what keeps it one row: adding a kind to [`Vulkan::matvec_kernel`]
/// without a shader here fails
/// `every_kind_a_compiled_backend_claims_can_actually_be_launched`.
#[cfg(feature = "vulkan")]
fn vulkan_matvec_launch(kind: QuantKind) -> Option<VulkanMatvecLaunchFn> {
    match kind {
        QuantKind::Q8_0 => Some(frink_vulkan::dispatch::q8_0_matvec),
        _ => None,
    }
}

/// The Metal backend (`frink-metal`).
pub struct Metal;

/// The CUDA backend (`frink-cuda`).
pub struct Cuda;

/// The Vulkan backend (`frink-vulkan`) — **one kernel wide**.
///
/// `frink-vulkan` is the `vulkan-beachhead` GO/NO-GO slice, not a
/// backend: a single hand-emitted SPIR-V Q8_0 matvec, checked against a
/// scalar twin and run on a real device through MoltenVK. See
/// `docs/plans/vulkan-beachhead-verdict.md`.
///
/// This impl is what wiring that slice into the seam costs, and it is
/// deliberately not more than the slice supports:
///
/// - **Q8_0 and nothing else.** [`Vulkan::matvec_kernel`] names one
///   kind; every other kind reports no kernel, which is the honest
///   answer and is what makes the registry say "NO KERNEL … falls back
///   to CPU apply_cpu" instead of quietly running slow.
/// - **No GEMM at all.** [`Vulkan::gemm_supported`] is false for every
///   kind. There is no `mul_mm` shader, and `apply_batch_with_acts` has
///   no Vulkan arm, so a batched prefill runs on the host —
///   [`Vulkan::GEMM_FALLBACK`] says exactly that.
/// - **No performance claim.** `q8_0_matvec` rebuilds its entire
///   pipeline per call. Nothing here may be reported as a measured
///   capability; the verdict says so and this comment repeats it
///   because the code is now reachable.
pub struct Vulkan;

impl BackendCaps for Metal {
    const ID: Backend = Backend::Metal;
    const NAME: &'static str = "Metal";
    /// `apply_gpu_batch` re-reads the whole weight matrix once per
    /// position, on the GPU. Still Metal, still the 13.7x shape.
    const GEMM_FALLBACK: &'static str = "Metal N x matvec batch";

    /// As the kernel name [`frink_metal::gpu::matvec_launch_meta`]
    /// resolves.
    ///
    /// This is the single source of truth for that question. It is *not*
    /// `#[cfg(feature = "metal")]`-gated deliberately: the table is a
    /// property of the kernel set, and gating it would make it
    /// untestable on the builds that run `cargo test --workspace`.
    ///
    /// Duplicating this list is how IQ4_XS batched prefill silently ran
    /// on the CPU — `metal_kind_supported` and `apply_gpu_batch`'s kind
    /// table disagreed by exactly one entry, and the only symptom was a
    /// benchmark row 13.7x behind. Every Metal-kind question now routes
    /// through here.
    fn matvec_kernel(kind: QuantKind) -> Option<&'static str> {
        match kind {
            QuantKind::Q8_0
            | QuantKind::Q4_0
            | QuantKind::Q5_0
            | QuantKind::Q4K
            | QuantKind::Q5K
            | QuantKind::Q6K
            | QuantKind::IQ4XS
            | QuantKind::Ptq1_0
            | QuantKind::Q2K
            | QuantKind::Q3K
            | QuantKind::IQ4NL
            | QuantKind::Mxfp4Gguf
            | QuantKind::IQ2XXS
            | QuantKind::IQ2XS
            | QuantKind::IQ2S
            | QuantKind::IQ3XXS
            | QuantKind::IQ3S
            | QuantKind::IQ1S
            | QuantKind::IQ1M
            | QuantKind::Q4_1
            | QuantKind::Q5_1
            | QuantKind::Tq1_0 => Some(kind.name()),
            _ => None,
        }
    }

    /// The `*_mul_mm_sg` simdgroup GEMMs.
    ///
    /// The invariant that this set equals [`Metal::matvec_kernel`]'s is
    /// asserted by a test, so adding a matvec kernel without a GEMM
    /// fails the suite instead of a benchmark.
    fn gemm_supported(kind: QuantKind) -> bool {
        // Q5_0 JOINED 2026-09-01, and the two-year-old comment this
        // replaced named the exact condition: "the honest close is a
        // `q5_0_matvec` plus a Q5_0 row in the bench suite, not a sixth
        // entry in this list."
        //
        // The matvec now exists (`Q5_0_MATVEC_KERNEL_SRC`), so the split
        // this list was protecting against is gone: Q5_0 was already
        // getting GPU prefill through `mul_mm_sg_launch` and `mapped_sg`,
        // which never consulted this table, while every decode step fell
        // back to the CPU for want of the matvec. That is the mixed
        // CPU/GPU path the old comment feared, and it was live rather
        // than hypothetical.
        //
        // The bench row is still owed: there is no Q5_0 checkpoint in
        // `benchmarks/suite.json`, so this path is
        // CORRECT-BY-CONSTRUCTION and UNMEASURED.
        // `Llama-3.2-1B-Instruct-Q5_K_M` is Q5_K, not Q5_0.
        matches!(
            kind,
            QuantKind::Q8_0
                | QuantKind::Q4_0
                | QuantKind::Q5_0
                | QuantKind::Q4K
                | QuantKind::Q5K
                | QuantKind::Q6K
                | QuantKind::IQ4XS
                | QuantKind::Ptq1_0
                | QuantKind::Q2K
                | QuantKind::Q3K
                | QuantKind::IQ4NL
                | QuantKind::Mxfp4Gguf
                | QuantKind::IQ2XXS
                | QuantKind::IQ2XS
                | QuantKind::IQ2S
                | QuantKind::IQ3XXS
                | QuantKind::IQ3S
                | QuantKind::IQ1S
                | QuantKind::IQ1M
                | QuantKind::Q4_1
                | QuantKind::Q5_1
                | QuantKind::Tq1_0
        )
    }
}

impl BackendCaps for Cuda {
    const ID: Backend = Backend::Cuda;
    const NAME: &'static str = "CUDA";
    /// `apply_batch_with_acts` decomposes a CUDA prefill into one
    /// matvec per position for every kind off [`Cuda::gemm_supported`].
    const GEMM_FALLBACK: &'static str = "CUDA per-position matvec";

    /// The decode path, and the arm that has actually run on a GPU --
    /// for six of its nine kinds.
    ///
    /// **DERIVED from `frink_cuda::matvec_kinds::KINDS`, not restated.**
    /// That table is compiled on every build (it is CUDA C text and
    /// three strings per row; nothing in it needs `cudarc`), and
    /// `frink-cuda` is an unconditional dependency for exactly this
    /// reason. The set used to be written out here as a `matches!` and
    /// checked against the kernel table by a test that only ran under
    /// `--features cuda`; over-claiming there sends a decode to an
    /// NVRTC module that does not exist, and under-claiming leaves a
    /// kernel nothing ever calls. Both have happened.
    ///
    /// The name is returned only to share
    /// [`BackendCaps::matvec_kernel`]'s shape with Metal; nothing on the
    /// CUDA path reads it, because `frink-cuda`'s launchers are named
    /// functions rather than entries in a string-keyed table.
    fn matvec_kernel(kind: QuantKind) -> Option<&'static str> {
        frink_cuda::matvec_kinds::kind_by_name(kind.name()).map(|_| kind.name())
    }

    /// The `mul_mm` prefill path.
    ///
    /// **DERIVED from `frink_cuda::mul_mm::KINDS`**, for the same
    /// reason and by the same mechanism as [`Cuda::matvec_kernel`]
    /// above. It equals that set, and
    /// `the_matvec_table_and_the_mul_mm_table_name_the_same_kinds` in
    /// `frink-cuda` is what keeps it equal -- a kind with one kernel
    /// and not the other splits a forward pass across two devices.
    ///
    /// It did not until 2026-09-04: only Q8_0 and Q4_0 had a
    /// matrix-matrix product, so a K-quant prefill decomposed into one
    /// matvec launch per position. Measured on a GTX 1080, that cost
    /// Llama-3.2-3B Q4_K_M 4.88 tok/s of pp512 against llama.cpp's
    /// 1586.80, a 325x gap on the most common quantization in
    /// circulation (#131). IQ4_XS is still absent: it is a codebook
    /// lookup rather than an affine dequant.
    ///
    /// Q5_0 joined on 2026-09-05, matvec and GEMM together, because
    /// `a_cuda_kind_with_a_matvec_also_has_a_gemm` makes half of it
    /// fail the suite -- and because half of it is the shape that cost
    /// Metal a Q5_0 decode path: GPU prefill with every decode step on
    /// the host.
    ///
    /// IQ4_NL, IQ4_XS and MXFP4 joined on 2026-09-09, matvec and GEMM
    /// together. They are CODEBOOK formats: the stored 4-bit code is an
    /// index into a sixteen-entry table, not a magnitude, so the kernel
    /// carries that table in `__constant__` memory. gpt-oss ships
    /// MXFP4 and no GPU backend had it at all, so every expert decoded
    /// on the host with the device idle.
    ///
    /// **UNRUN ON HARDWARE.** The kernel is checked against a scalar
    /// twin and by executing the emitted CUDA C on the host, and has
    /// never executed on a GPU. See `crates/frink-cuda/src/mul_mm.rs`.
    fn gemm_supported(kind: QuantKind) -> bool {
        frink_cuda::mul_mm::kind_by_name(kind.name()).is_some()
    }
}

impl BackendCaps for Vulkan {
    const ID: Backend = Backend::Vulkan;
    const NAME: &'static str = "Vulkan";
    /// There is no Vulkan batch entry point of any kind:
    /// `apply_gpu_batch` is `#[cfg(feature = "metal")]` and
    /// `apply_batch_with_acts` has a CUDA arm and a Metal arm. So a
    /// prefill against a Vulkan-resident kind runs on the host, and
    /// this names the host path rather than inventing a GPU one.
    const GEMM_FALLBACK: &'static str = crate::kernel_registry::host_fallback::BATCH;

    /// Exactly one kind, because there is exactly one shader:
    /// `frink_vulkan::q8_0_shader`.
    ///
    /// Everything else must report `None` rather than something
    /// plausible. A capability table that over-claims is how a kind ends
    /// up "supported" with no kernel behind it, which this repo has now
    /// paid for twice (IQ4_XS prefill, Q5_0 decode). The guard test
    /// checks this against [`vulkan_matvec_launch`] for all 23 kinds.
    fn matvec_kernel(kind: QuantKind) -> Option<&'static str> {
        match kind {
            QuantKind::Q8_0 => Some(kind.name()),
            _ => None,
        }
    }

    /// No kind, for any kind. The beachhead emitted one matvec shader
    /// and deliberately no `mul_mm`; the verdict puts a real GEMM in
    /// `vulkan-prefill-gemm`, which is where the backend decision
    /// actually lives.
    ///
    /// This is the one place the Metal invariant
    /// (`every_metal_matvec_kind_also_has_a_metal_gemm`: matvec set ==
    /// GEMM set) is knowingly not held, and it is held open rather than
    /// papered over: Q8_0 decodes on Vulkan and prefills on the CPU,
    /// the registry records the split by name, and `frink bench` would
    /// show it.
    fn gemm_supported(_kind: QuantKind) -> bool {
        false
    }
}

/// The `FRINK_METAL` / `FRINK_CUDA` grammar, which was written out
/// twice in bodies that were byte-identical apart from the alias:
///
/// - `0|false|off|cpu` — force CPU
/// - `1|true|on|<alias>` — force this backend
/// - unset / anything else — whatever `probe` says
///
/// `probe` is only called when the environment did not decide, which is
/// what keeps a forced-off build from opening a device.
///
/// Compiled when a backend needs it, and under `test` so the grammar
/// stays checked on the CPU-only builds that run `cargo test`.
#[cfg(any(feature = "metal", feature = "cuda", feature = "vulkan", test))]
fn env_or_probe(value: Option<&str>, on_alias: &str, probe: impl FnOnce() -> bool) -> bool {
    match value {
        Some("0") | Some("false") | Some("off") | Some("cpu") => false,
        Some("1") | Some("true") | Some("on") => true,
        Some(v) if v == on_alias => true,
        _ => probe(),
    }
}

/// A `frink_metal::gpu::launch_*_matvec` function pointer's signature
/// (`weights`/`x` borrowed; row block count is derived inside
/// `frink_metal::gpu`).
#[cfg(feature = "metal")]
type MetalMatvecLaunchFn =
    fn(&[u8], &[f32], usize, usize) -> Result<Vec<f32>, frink_metal::gpu::MetalError>;

/// A `frink_cuda::gpu::launch_*_matvec` function pointer's signature
/// (all five real kernels share it exactly).
#[cfg(feature = "cuda")]
type CudaMatvecLaunchFn =
    fn(&[u8], &[f32], usize, usize, usize) -> Result<Vec<f32>, frink_cuda::gpu::CudaError>;

/// A `frink_vulkan::dispatch` matvec's signature. Same five arguments
/// as [`CudaMatvecLaunchFn`] -- `frink-vulkan` was written to this
/// list on purpose -- plus the borrowed [`frink_vulkan::device::Context`]
/// in front, because Vulkan keeps no process-global device inside its
/// own crate the way `frink_metal::gpu` and `frink_cuda::gpu` do.
/// [`vulkan_context`] is that global, and it lives here so the beachhead
/// crate stays a beachhead.
#[cfg(feature = "vulkan")]
type VulkanMatvecLaunchFn = fn(
    &frink_vulkan::device::Context,
    &[u8],
    &[f32],
    usize,
    usize,
    usize,
) -> Result<Vec<f32>, frink_vulkan::device::VulkanError>;

#[cfg(feature = "metal")]
impl BackendDispatch for Metal {
    const MATVEC_FALLBACK: &'static str = "falling back to CPU";

    fn has_launch(kind: QuantKind) -> bool {
        metal_matvec_launch(kind).is_some()
    }

    fn dense_enabled() -> bool {
        // Before the cell: a host pin holds whatever was decided first.
        if super::host_pin::host_backend_pinned() {
            return false;
        }
        use std::sync::OnceLock;
        // A `static` inside a generic function is shared across every
        // monomorphization, so this cache cannot be hoisted into a
        // default trait method: each backend needs its own cell.
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            let v = std::env::var("FRINK_METAL").ok();
            env_or_probe(v.as_deref(), "metal", || {
                frink_metal::gpu::probe().is_some()
            })
        })
    }

    fn launch_matvec(
        kind: QuantKind,
        weights: &[u8],
        x: &[f32],
        rows: usize,
        row_bytes: usize,
    ) -> Option<Result<Vec<f32>, BackendError>> {
        let launch = metal_matvec_launch(kind);
        // This table and `Metal::matvec_kernel` answer the same question
        // and must never diverge; when they did, IQ4_XS prefill silently
        // moved to the CPU. They CANNOT be one table -- the names are
        // needed on builds where `frink-metal` is not a dependency and
        // these function pointers do not exist -- so the agreement stays
        // asserted rather than structural.
        debug_assert_eq!(
            launch.is_some(),
            Self::matvec_kernel(kind).is_some(),
            "apply_gpu's Metal launch table disagrees with metal_matvec_kind_name for {:?}",
            kind
        );
        let launch = launch?;
        Some(launch(weights, x, rows, row_bytes).map_err(BackendError::new))
    }
}

#[cfg(feature = "cuda")]
impl BackendDispatch for Cuda {
    const MATVEC_FALLBACK: &'static str = "trying next backend / CPU";

    fn has_launch(kind: QuantKind) -> bool {
        cuda_matvec_launch(kind).is_some()
    }

    fn dense_enabled() -> bool {
        // Before the cell: a host pin holds whatever was decided first.
        if super::host_pin::host_backend_pinned() {
            return false;
        }
        use std::sync::OnceLock;
        // See the note on `Metal::dense_enabled` for why this cell is
        // not shared through a default method.
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            let v = std::env::var("FRINK_CUDA").ok();
            env_or_probe(v.as_deref(), "cuda", || frink_cuda::gpu::probe().is_some())
        })
    }

    fn launch_matvec(
        kind: QuantKind,
        weights: &[u8],
        x: &[f32],
        rows: usize,
        row_bytes: usize,
    ) -> Option<Result<Vec<f32>, BackendError>> {
        let launch = cuda_matvec_launch(kind)?;

        // `FRINK_CUDA=0` must mean the CPU, and a build with the
        // `cuda` feature on a host with no driver must fall back, not
        // die. Neither was true here until 2026-09-09.
        //
        // `Vulkan::launch_matvec` has carried this guard since it
        // landed, and its comment said Metal and CUDA did not need one
        // because "their launchers no-op into an error when their
        // device is absent". That is true of Metal. It is NOT true of
        // CUDA: `cudarc` resolves `libcuda` through a lazily loaded
        // symbol table and PANICS (`cudarc-0.11.9/src/lib.rs:98`,
        // "Unable to dynamically load the cuda shared library") rather
        // than returning an error, so the `Result` this arm is written
        // around never gets a chance to be `Err`. A panic in a rayon
        // worker is not a fallback.
        //
        // It was latent rather than harmless: any quantized matvec on
        // such a build aborted the process. Nothing in the suite
        // reached it because the one test that dispatches a real kind
        // through here is `#[ignore]`d, and the test that dispatches an
        // unsupported one picked a kind that returned `None` above --
        // until Q2_K gained a kernel and stopped being unsupported.
        //
        // `dense_enabled()` is a `OnceLock` over a probe that catches
        // its own panics, so this costs one atomic load after the first
        // call.
        if !Self::dense_enabled() {
            return None;
        }

        // Derived here rather than at the seam: `block_bytes_for_kind`
        // is `unreachable!()` outside the CUDA-dispatchable kinds, and
        // reaching it is gated on the match above having named one.
        let n_blocks_per_row =
            row_bytes / crate::weight_matrix::WeightMatrix::block_bytes_for_kind(kind);
        Some(launch(weights, x, rows, row_bytes, n_blocks_per_row).map_err(BackendError::new))
    }
}

/// The process-wide Vulkan device, opened at most once.
///
/// `frink_metal::gpu` and `frink_cuda::gpu` each keep their device
/// inside their own crate, so their launch functions take no context.
/// `frink-vulkan` deliberately does not: it is a beachhead whose
/// `Context` is created and dropped by its own tests, and giving it a
/// hidden global would have made the GO/NO-GO slice into infrastructure.
/// So the global lives here, on the seam's side of the boundary.
///
/// `Mutex`, not a bare `Context`: a `vk::Queue` must be externally
/// synchronized, and `apply_gpu` is called from rayon workers.
/// Serializing them is correct and is not a regression, because
/// `q8_0_matvec` rebuilds its entire pipeline per call and is not a
/// performance path in the first place — see the verdict.
///
/// `None` means the device could not be opened. That is reported once,
/// here, rather than once per matvec.
#[cfg(feature = "vulkan")]
fn vulkan_context() -> Option<&'static std::sync::Mutex<frink_vulkan::device::Context>> {
    use std::sync::{Mutex, OnceLock};
    static CTX: OnceLock<Option<Mutex<frink_vulkan::device::Context>>> = OnceLock::new();
    CTX.get_or_init(|| match frink_vulkan::device::Context::new() {
        Ok(ctx) => Some(Mutex::new(ctx)),
        Err(e) => {
            eprintln!(
                "frink: Vulkan device unavailable, {}: {e}",
                Vulkan::MATVEC_FALLBACK
            );
            None
        }
    })
    .as_ref()
}

#[cfg(feature = "vulkan")]
impl BackendDispatch for Vulkan {
    /// Last in [`gpu_backend_table`], so there is nothing after it.
    const MATVEC_FALLBACK: &'static str = "falling back to CPU";

    fn has_launch(kind: QuantKind) -> bool {
        vulkan_matvec_launch(kind).is_some()
    }

    fn dense_enabled() -> bool {
        // Before the cell: a host pin holds whatever was decided first.
        if super::host_pin::host_backend_pinned() {
            return false;
        }
        use std::sync::OnceLock;
        // See the note on `Metal::dense_enabled` for why this cell is
        // not shared through a default method.
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            let v = std::env::var("FRINK_VULKAN").ok();
            env_or_probe(v.as_deref(), "vulkan", || {
                frink_vulkan::device::probe().is_ok()
            })
        })
    }

    fn launch_matvec(
        kind: QuantKind,
        weights: &[u8],
        x: &[f32],
        rows: usize,
        row_bytes: usize,
    ) -> Option<Result<Vec<f32>, BackendError>> {
        let launch = vulkan_matvec_launch(kind)?;

        // Unlike Metal, whose launcher no-ops into an error when its
        // device is absent, `frink-vulkan` has no global to consult --
        // so the env grammar is honoured here or not at all. (CUDA was
        // in this sentence too, and should not have been: see
        // `Cuda::launch_matvec`, which now carries the same guard.)
        // `FRINK_VULKAN=0` must mean the CPU, not "open a device
        // anyway". Returning `None` (rather than an error) is right:
        // "this backend is not running here" is the same answer as "no
        // kernel", and both mean try the next backend.
        if !Self::dense_enabled() {
            return None;
        }
        let ctx = vulkan_context()?;

        // `frink_vulkan::dispatch::q8_0_matvec` asserts its shape
        // invariants, which is right for a test-driven beachhead and
        // wrong for a dispatch path: a panic in a rayon worker is not a
        // fallback. Checked here so a mismatch is an error the caller
        // logs and recovers from.
        let block_bytes = crate::weight_matrix::WeightMatrix::block_bytes_for_kind(kind);
        let n_blocks_per_row = row_bytes / block_bytes;
        if weights.len() != rows * row_bytes
            || row_bytes != n_blocks_per_row * block_bytes
            || x.len() != n_blocks_per_row * frink_vulkan::q8_0_shader::BLOCK_ELEMS
            || rows == 0
            || n_blocks_per_row == 0
        {
            return Some(Err(BackendError::new(format!(
                "Vulkan {} matvec shape rejected: {} weight bytes, {} activations, \
                 rows={rows} row_bytes={row_bytes}",
                kind.name(),
                weights.len(),
                x.len(),
            ))));
        }

        let guard = match ctx.lock() {
            Ok(g) => g,
            // A poisoned mutex means another thread panicked mid-
            // dispatch; the device may be mid-submission, so refuse
            // rather than reuse it.
            Err(_) => {
                return Some(Err(BackendError::new(
                    "Vulkan context poisoned by an earlier panic",
                )))
            }
        };
        Some(
            launch(&guard, weights, x, rows, row_bytes, n_blocks_per_row)
                .map_err(BackendError::new),
        )
    }
}

/// **The** backend table: one row per GPU backend, in dispatch
/// precedence order — CUDA first, then Metal, then Vulkan, then the CPU
/// fallthrough the caller supplies.
///
/// Vulkan is last on purpose. On the only machine that can run all
/// three it reaches the GPU through MoltenVK, i.e. through Metal, and
/// it has one kernel; a native backend must win over a translation
/// layer wrapping it.
///
/// Each row is `(enum variant, feature / registry name, seam type)`.
/// The three are the same string in three grammars, and that is exactly
/// why they are written once: the cargo feature, the
/// [`crate::kernel_registry::Backend`] variant and the type were three
/// hand-kept lists, and "two structures that must agree about one
/// thing" is the dominant bug shape in this repo.
///
/// It expands `$mac!` ONCE with every row, so a consumer can build a
/// single item (an `enum`) from it and not just a sequence of
/// statements. `$extra` is passed through in brackets ahead of the rows
/// so a consumer that needs its own callback — [`with_gpu_backends`] —
/// can forward one without re-listing the table.
macro_rules! gpu_backend_table {
    ($mac:path $(, $extra:tt)*) => {
        $mac! {
            [$($extra),*]
            (Cuda, "cuda", Cuda),
            (Metal, "metal", Metal),
            (Vulkan, "vulkan", Vulkan),
        }
    };
}
pub(crate) use gpu_backend_table;

/// Expands `$mac!(Backend)` once per **compiled-in** backend, in
/// [`gpu_backend_table`] order.
///
/// This exists because the order was hand-copied at every dispatch site
/// and in `active_backend`, and a macro is the only way to keep static
/// dispatch, per-backend `#[cfg]`, and one written-down order at the
/// same time. A third backend is one line in the table, not here.
///
/// `$mac` must tolerate being expanded zero times: on a CPU-only build
/// this produces nothing, so define it `#[allow(unused_macros)]`.
macro_rules! with_gpu_backends {
    ($mac:ident) => {
        $crate::weight_matrix::gpu_backend::gpu_backend_table!(
            $crate::weight_matrix::gpu_backend::gpu_backend_dispatch_rows,
            $mac
        );
    };
}
pub(crate) use with_gpu_backends;

/// [`with_gpu_backends`]'s row expander. The `#[cfg(feature = …)]` is
/// built from the table's own name column, so a backend cannot be in
/// the list under one feature and gated on another.
macro_rules! gpu_backend_dispatch_rows {
    ([$mac:ident] $(($variant:ident, $feature:literal, $ty:ident)),* $(,)?) => {
        $(
            #[cfg(feature = $feature)]
            $mac!($crate::weight_matrix::gpu_backend::$ty);
        )*
    };
}
pub(crate) use gpu_backend_dispatch_rows;

/// Expands `$mac!(Backend)` once per backend **whether or not it is
/// compiled in**, in [`gpu_backend_table`] order.
///
/// The ungated twin of [`with_gpu_backends`], and the reason
/// [`BackendCaps`] is ungated: `probe_kernels_for` has to answer "what
/// would Metal resolve for this kind" on a build with no Metal, which
/// is the only way the kernel-coverage tests run under a plain
/// `cargo test`. A consumer of this must therefore stay inside
/// [`BackendCaps`] — [`BackendDispatch`] does not exist for a backend
/// whose feature is off.
///
/// Never expands zero times, so `$mac` needs no `unused_macros` cover.
macro_rules! with_gpu_backend_caps {
    ($mac:ident) => {
        $crate::weight_matrix::gpu_backend::gpu_backend_table!(
            $crate::weight_matrix::gpu_backend::gpu_backend_caps_rows,
            $mac
        );
    };
}
pub(crate) use with_gpu_backend_caps;

/// [`with_gpu_backend_caps`]'s row expander.
macro_rules! gpu_backend_caps_rows {
    ([$mac:ident] $(($variant:ident, $feature:literal, $ty:ident)),* $(,)?) => {
        $(
            $mac!($crate::weight_matrix::gpu_backend::$ty);
        )*
    };
}
pub(crate) use gpu_backend_caps_rows;

#[cfg(test)]
mod tests {
    use super::*;

    /// The env grammar both enable probes share. `probe` must not be
    /// consulted when the environment already decided — a forced-off
    /// build must never open a device.
    #[test]
    fn env_decides_before_the_probe_is_consulted() {
        for forced_off in ["0", "false", "off", "cpu"] {
            assert!(!env_or_probe(Some(forced_off), "metal", || panic!(
                "probed after {forced_off}"
            )));
        }
        for forced_on in ["1", "true", "on"] {
            assert!(env_or_probe(Some(forced_on), "metal", || panic!(
                "probed after {forced_on}"
            )));
        }
    }

    /// Each backend's own alias forces it on; the *other* backend's
    /// alias is not a value it understands, so it falls through to the
    /// probe rather than silently forcing.
    #[test]
    fn the_alias_is_per_backend() {
        assert!(env_or_probe(Some("metal"), "metal", || false));
        assert!(env_or_probe(Some("cuda"), "cuda", || false));
        assert!(!env_or_probe(Some("cuda"), "metal", || false));
        assert!(!env_or_probe(Some("metal"), "cuda", || false));
    }

    /// Unset, or a value the grammar does not name, defers to the probe.
    #[test]
    fn an_unrecognised_value_defers_to_the_probe() {
        assert!(env_or_probe(None, "metal", || true));
        assert!(!env_or_probe(None, "metal", || false));
        assert!(env_or_probe(Some("auto"), "metal", || true));
        assert!(!env_or_probe(Some("auto"), "metal", || false));
    }

    /// A backend cannot be dispatched to under one name and reported
    /// under another: dispatch and the registry read the same constant.
    ///
    /// [`Backend`]'s variants are generated from the same table these
    /// impls are listed in, so a *missing* variant is now impossible —
    /// but `const ID` is still written by hand in each impl, so naming
    /// another backend's variant is not. That is what the distinctness
    /// check catches, and the count check catches a variant with no
    /// backend behind it.
    #[test]
    fn every_backend_id_is_distinct_and_an_accelerator() {
        let mut ids = Vec::new();
        macro_rules! collect_id {
            ($b:ty) => {
                assert!(
                    <$b as BackendCaps>::ID.is_accelerator(),
                    "{} is reported as the CPU",
                    <$b as BackendCaps>::NAME
                );
                ids.push((<$b as BackendCaps>::ID, <$b as BackendCaps>::NAME));
            };
        }
        with_gpu_backend_caps!(collect_id);

        for (i, (id, name)) in ids.iter().enumerate() {
            for (other_id, other_name) in &ids[i + 1..] {
                assert_ne!(
                    id, other_id,
                    "{name} and {other_name} both report as {id} -- one of them is \
                     dispatched to under a registry identity that is not its own"
                );
            }
        }
        assert_eq!(
            ids.len() + 1,
            Backend::ALL.len(),
            "the registry has a backend variant no BackendCaps impl claims: {:?} vs {ids:?}",
            Backend::ALL
        );
    }

    /// Vulkan is **one kernel wide** and the seam must keep saying so.
    ///
    /// `frink-vulkan` has exactly one shader, `q8_0_shader`. If this
    /// table ever grows a kind, either a shader landed with it (and this
    /// test is the place to say so) or the table now over-claims — which
    /// is how IQ4_XS prefill and Q5_0 decode each silently moved to the
    /// CPU while the capability report said "GPU".
    ///
    /// Ungated on purpose, like [`BackendCaps`] itself: this is a
    /// property of the kernel set, so it is checked on every build,
    /// including the CPU-only one that runs `cargo test --workspace`.
    #[test]
    fn vulkan_claims_exactly_one_matvec_kind_and_no_gemm() {
        let claimed: Vec<QuantKind> = QuantKind::ALL
            .iter()
            .copied()
            .filter(|&k| Vulkan::matvec_kernel(k).is_some())
            .collect();
        assert_eq!(
            claimed,
            vec![QuantKind::Q8_0],
            "frink-vulkan has one shader (q8_0_shader); the capability table claims {claimed:?}"
        );
        for &k in QuantKind::ALL {
            assert!(
                !Vulkan::gemm_supported(k),
                "{k:?}: there is no Vulkan mul_mm shader, so a claimed GEMM would send \
                 prefill to a kernel that does not exist"
            );
        }
    }

    /// The GEMM twin of the test below: `Metal::gemm_supported` must
    /// answer exactly what `frink_metal::gpu::mul_mm_sg_meta` has a row
    /// for, because `apply_gpu_batch` launches through that table now.
    /// It used to launch through a per-kind match, and the match lacked
    /// Q5_0 and PTQ1_0 while this table claimed both: prefill for those
    /// kinds ran N matvecs with the kernel registry recording a GEMM hit.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_gemm_claims_are_exactly_the_gemm_table() {
        for &kind in QuantKind::ALL {
            let claimed = Metal::gemm_supported(kind);
            let has_row = frink_metal::gpu::mul_mm_sg_meta(kind.name()).is_some();
            assert_eq!(
                claimed, has_row,
                "{kind:?}: gemm_supported {claimed}, mul_mm_sg_meta row {has_row}"
            );
        }
    }

    /// Every kind a **compiled-in** backend's capability table CLAIMS
    /// must have a launch function behind it, for all 23 kinds and
    /// without a device.
    ///
    /// `Q5_0` did not, on Metal, from the day it was added. The kernel
    /// source and the `matvec_launch_meta` row landed together and both
    /// capability tables were widened on the strength of them — but
    /// `apply_gpu`'s single-matvec decode path dispatches through a
    /// per-kind `launch_*_matvec` FUNCTION, and there was no Q5_0 one.
    /// So batched prefill ran on the GPU while single-token decode
    /// silently fell to the CPU: exactly the mixed CPU/GPU split that
    /// widening was supposed to close.
    ///
    /// The `debug_assert_eq!` in `Metal::launch_matvec` did guard this,
    /// but only for kinds a run actually reaches, and only in debug. A
    /// release build just ran slower. This checks the whole table up
    /// front — and, since it expands over
    /// [`with_gpu_backends`] rather than naming Metal, CUDA and Vulkan
    /// each get it for free instead of Metal getting it three times.
    /// CUDA had no such guard at all before its launch table was split
    /// out of `launch_matvec`; Vulkan gets one on its first day.
    #[test]
    fn every_kind_a_compiled_backend_claims_can_actually_be_launched() {
        #[allow(unused_macros)]
        macro_rules! check_launch_table {
            ($b:ty) => {
                let mut claimed_without_launch = Vec::new();
                let mut launchable_unclaimed = Vec::new();
                for &kind in QuantKind::ALL {
                    let claimed = <$b as BackendCaps>::matvec_kernel(kind).is_some();
                    let launchable = <$b as BackendDispatch>::has_launch(kind);
                    if claimed && !launchable {
                        claimed_without_launch.push(kind);
                    }
                    if launchable && !claimed {
                        launchable_unclaimed.push(kind);
                    }
                }
                assert!(
                    claimed_without_launch.is_empty(),
                    "{} claims a matvec nothing can launch: {claimed_without_launch:?} -- \
                     decode falls to the CPU for these while batched prefill runs on the GPU",
                    <$b as BackendCaps>::NAME
                );
                assert!(
                    launchable_unclaimed.is_empty(),
                    "{} has a launch for {launchable_unclaimed:?} that its capability \
                     table does not claim, so nothing will ever call it",
                    <$b as BackendCaps>::NAME
                );
            };
        }
        with_gpu_backends!(check_launch_table);
    }

    /// A kind that claims a matvec must name itself the way the
    /// backend's launch-meta table is keyed, for every backend and not
    /// just Metal. Expanded over the ungated table, so it holds on a
    /// CPU-only build and a third backend gets it for free — which it
    /// did not when the body hand-listed `[Metal, Cuda]`.
    #[test]
    fn a_claimed_matvec_kernel_is_named_after_its_kind() {
        macro_rules! check_names {
            ($b:ty) => {
                for &k in QuantKind::ALL {
                    if let Some(name) = <$b as BackendCaps>::matvec_kernel(k) {
                        assert_eq!(
                            name,
                            k.name(),
                            "{} names {k:?} {name:?}",
                            <$b as BackendCaps>::NAME
                        );
                    }
                }
            };
        }
        with_gpu_backend_caps!(check_names);
    }

    /// The GPU matvec against the CPU one, on a real device.
    ///
    /// This is the only test here that opens a device, and it is the
    /// only one that can catch the seam wiring the right kernel to the
    /// wrong arguments — a `row_bytes` that is not `n_blocks * 34`, or
    /// an activation length derived from the wrong block size, are both
    /// legal calls that produce a wrong number. `frink-vulkan`'s own
    /// twin test proves the shader; this proves the *call*.
    ///
    /// Skips, loudly, when no device is reachable, so it is not a test
    /// that cannot fail: on a host with Vulkan it asserts, and it was
    /// checked by sabotage (feeding `rows + 1`) before being committed.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_vulkan_seam_matvec_matches_the_cpu_matvec() {
        use crate::weight_matrix::{WeightBytes, WeightMatrix};

        if !Vulkan::dense_enabled() {
            eprintln!("no Vulkan device reachable; the seam matvec was NOT checked");
            return;
        }

        // Three blocks per row: 3 * 34 = 102 bytes, deliberately not a
        // multiple of 4, which is the alignment case the shader's byte
        // extraction exists for.
        let (rows, blocks) = (9usize, 3usize);
        let cols = blocks * 32;
        let f32_weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 37) as f32 - 18.0) / 11.0)
            .collect();
        let mut data = Vec::new();
        for r in 0..rows {
            data.extend_from_slice(&frink_quant::quantize_q8_0(
                &f32_weights[r * cols..(r + 1) * cols],
            ));
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) / 5.0).collect();

        let m = WeightMatrix::Quantized {
            data: WeightBytes::Owned(data),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        let WeightMatrix::Quantized { data, .. } = &m else {
            unreachable!()
        };
        let got = Vulkan::launch_matvec(QuantKind::Q8_0, data.as_slice(), &x, rows, blocks * 34)
            .expect("Q8_0 has a Vulkan kernel")
            .expect("the launch must succeed once a device is open");
        let want = m.apply(&x);

        assert_eq!(got.len(), want.len());
        for (r, (g, w)) in got.iter().zip(&want).enumerate() {
            // The same 1e-4 relative tolerance frink-cuda's hardware
            // test uses, and for the same reason: a GPU may contract
            // `acc + a * b` into an FMA.
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "row {r}: vulkan {g} vs cpu {w}"
            );
        }
    }

    /// A shape the kernel cannot honour must come back as an ERROR the
    /// caller logs, never as a panic in a rayon worker.
    /// `frink_vulkan::dispatch::q8_0_matvec` asserts its invariants,
    /// which is right for a beachhead and fatal on a dispatch path, so
    /// the seam checks them first. Needs no device: the check runs
    /// before the context is used.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_mismatched_vulkan_shape_is_an_error_not_a_panic() {
        if !Vulkan::dense_enabled() {
            eprintln!("no Vulkan device reachable; the shape guard was NOT checked");
            return;
        }
        // 2 rows of 1 block each, but an activation sized for 2 blocks.
        let weights = vec![0u8; 2 * 34];
        let x = vec![0.0f32; 64];
        let out = Vulkan::launch_matvec(QuantKind::Q8_0, &weights, &x, 2, 34);
        assert!(
            matches!(out, Some(Err(_))),
            "a mismatched shape must be a reported error, got {out:?}"
        );
    }
}
