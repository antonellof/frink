//! CUDA execution path via `cudarc` 0.11.9 (pinned to this version
//! because newer `cudarc`/`libloading` releases require a rustc newer
//! than the 1.75 available in an earlier development environment for
//! this project -- see Cargo.toml's comment). Uses dynamic loading so
//! this compiles without a CUDA toolkit installed.
//!
//! # Verified on real GPU hardware
//!
//! Five of the six matvec kernels (`launch_q8_0_matvec`,
//! `launch_q4_0_matvec`, `launch_q4_k_matvec`, `launch_q5_k_matvec`,
//! `launch_q6_k_matvec`) have
//! been compiled by NVRTC and executed on real NVIDIA GPUs (RTX 3060s,
//! rented on vast.ai), with their output cross-checked against
//! `frink_quant`'s scalar reference for real, non-trivial quantized test
//! data -- not just "the launch didn't error." Run this verification
//! yourself with:
//!   cargo test -p frink-cuda --features cuda -- --ignored
//!
//! Real bugs caught and fixed by actually running this on hardware
//! (none were visible from reading the code):
//! - The Q4_0 test's own fixture-generation code built one 18-byte Q4_0
//!   block per row regardless of the test's `cols` value, so for a
//!   64-column test matrix (2 blocks/row) the weight buffer ended up
//!   half the size the test's own indexing expected -- an out-of-bounds
//!   panic in the test harness itself, not the kernel.
//! - Q4_K/Q5_K/Q6_K failed a fixed absolute tolerance (`1e-2`) against
//!   reference values in the 1e7-1e8 range (small, single-digit-to-tens
//!   diffs) -- root-caused against llama.cpp's real CUDA K-quant kernel
//!   design (integer `dp4a` dot product, converting to float only once
//!   per block, specifically to avoid GPU/CPU float non-associativity)
//!   and its own test convention (`max_nmse_err()`, a relative metric,
//!   never exact equality) -- not a kernel bug. Fixed by switching to a
//!   relative-error assertion (`assert_close_relative`).
//! - That same relative-tolerance assertion then hit a second, narrower
//!   bug on real hardware: `GPU=NaN CPU reference=NaN` for a Q6_K row
//!   whose pseudo-random test bytes happened to decode as a NaN/Inf f16
//!   scale (garbage in, garbage out, identically on both backends) --
//!   `(got - want).abs() <= tol` is always false for NaN under IEEE754,
//!   so two backends that correctly *agreed* (both NaN) still failed the
//!   assertion. Fixed by treating NaN==NaN as agreement explicitly.
//!
//! Q8_0/Q4_0/Q4_K/Q5_K all pass on real hardware as of 2026-07-31; Q6_K's
//! NaN-comparison fix has not yet been re-run on hardware (see that
//! test's own `#[ignore]` message).
//!
//! **`launch_q5_0_matvec` is the sixth, and NO GPU HAS RUN IT.** It
//! landed 2026-09-05 (`docs/plans/cpu-cuda-parity.md` §6) and is a
//! transcription of `frink-metal`'s Q5_0 matvec, which is itself
//! `ggml`'s `dequantize_row_q5_0`. Nothing above applies to it: the
//! only checks it has are that the CUDA C names the entry point its
//! table row claims and strides by `Q5_0_BLOCK_BYTES`. Until
//! `cargo test -p frink-cuda --features cuda -- --ignored` has run
//! `launch_q5_0_matvec_matches_cpu_reference` on a device and the
//! result is written down, CUDA Q5_0 is a claim, not a capability.
//!
//! The device-probe path (`probe()`) was already verified earlier
//! (tested to degrade correctly with no GPU present -- including a real
//! bug found and fixed in `cudarc`'s panic-instead-of-`Result::Err`
//! behavior when no CUDA library exists at all). A second real bug in
//! that same test surfaced the first time it actually ran *with* a real
//! GPU present (full workspace suite on rented hardware, 2026-07-31):
//! the test hard-coded "no CUDA device here," true on the
//! no-GPU sandbox it was written on but false on real rented hardware --
//! fixed to check both real outcomes instead of assuming one.
//!
//! `WeightMatrix::apply_gpu`-dispatched real MoE generation (a real
//! third-party checkpoint, OLMoE-1B-7B-0924, `FRINK_GPU_VRAM_BUDGET_BYTES`
//! enabled) has been measured end to end: 16.345s for 64 tokens on a
//! rented RTX 3060, correct output, after fixing the per-call CUDA
//! context/recompilation overhead (`shared_device`/`ensure_module_loaded`
//! below) that made the same request take 15+ minutes (never completing)
//! before that fix -- see `docs/MODELS.md` for the full comparison.
//!
//! Not yet verified: a full FFN (multiple fused matvecs) rather than one
//! projection at a time, and behavior on GPUs other than the RTX 3060s
//! this was tested against.

/// What `probe()` found on the host, if anything.
pub struct CudaInfo {
    pub device_count: usize,
    pub first_device_name: Option<String>,
    pub total_vram_bytes: u64,
    /// Currently *free* device memory, the second half of
    /// `cuMemGetInfo`. This is what an admission check should divide
    /// up, not `total_vram_bytes`: another process (or another
    /// framework in this one) may already hold most of the card. It is
    /// a snapshot -- true when probed, not a reservation.
    pub free_vram_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("cudarc driver initialization failed: {0:?}")]
    DriverInit(String),
    #[error("NVRTC kernel compilation failed: {0:?}")]
    KernelCompile(String),
    #[error("kernel launch failed: {0:?}")]
    Launch(String),
    /// A shape or format this CUDA path does not implement. Distinct
    /// from `Launch` on purpose: the caller must fall back to a path
    /// that can compute it, never compute something else.
    #[error("unsupported on the CUDA path: {0}")]
    Unsupported(String),
}

/// Probes for CUDA devices. Returns `None` on any failure to load the
/// driver or find a device -- which, on this development sandbox (no
/// GPU, likely no NVIDIA driver at all), is the correct and expected
/// result every time, and is exactly what's tested.
///
/// Real finding from actually running this in the sandbox: `cudarc`
/// 0.11.9's dynamic-loading path does not return a clean `Result::Err`
/// when the `libcuda`/`libnvcuda` shared library itself cannot be
/// found at all -- it panics
/// (`cudarc::panic_no_lib_found`/`driver::sys::lib`). That is a
/// meaningfully different failure mode than "library loaded, but no
/// device present," so this function opens the library itself first
/// ([`driver_library_opens`]) and returns `None` when it will not, and
/// keeps a `catch_unwind` behind that for a driver cudarc cannot use.
/// The process's panic hook is left alone.
pub fn probe() -> Option<CudaInfo> {
    // cudarc panics on first use when the driver library will not open.
    // This used to swap the process's panic hook for a silent one around
    // the call, which silenced every panic any other thread raised in
    // that window, a server's journal record included. Opening the
    // library first, as cudarc does, keeps a GPU-less host quiet without
    // touching the hook.
    if !driver_library_opens() {
        return None;
    }
    // A backstop for a driver that opens but lacks a symbol cudarc needs:
    // that panics with cudarc's message, which is worth seeing.
    std::panic::catch_unwind(|| {
        let dev = cudarc::driver::CudaDevice::new(0).ok()?;
        let name = dev.name().ok();
        let (free, total) = cudarc::driver::result::mem_get_info().ok()?;
        Some(CudaInfo {
            device_count: 1,
            first_device_name: name,
            total_vram_bytes: total as u64,
            free_vram_bytes: free as u64,
        })
    })
    .unwrap_or(None)
}

/// Whether the CUDA driver library opens under the names cudarc 0.11
/// tries (`driver::sys::lib`: `cuda`, then `nvcuda`), so that cudarc's
/// own open, which panics on failure, will succeed.
fn driver_library_opens() -> bool {
    ["cuda", "nvcuda"].iter().any(|name| {
        // SAFETY: opening the driver library runs its initialisers, the
        // same open cudarc makes next; the handle is dropped at once and
        // cudarc's own handle keeps the library loaded.
        unsafe { libloading::Library::new(libloading::library_filename(name)) }.is_ok()
    })
}

/// Process-wide CUDA device handle, created once and reused for every
/// kernel launch. Before this existed, `launch_matvec` called
/// `CudaDevice::new(0)` on *every single call* -- a fresh CUDA context
/// per matvec, real-hardware-measured (via the OLMoE end-to-end stall
/// documented in `docs/MODELS.md`) to make GPU-dispatched MoE
/// generation impractically slow: up to ~384 individual expert-tensor
/// dispatches per token (16 layers x 8 active experts x 3 tensors) each
/// paying full context-creation + NVRTC-recompilation overhead. A
/// `Mutex<Option<Arc<CudaDevice>>>` rather than `OnceLock` because
/// `OnceLock::get_or_try_init` (needed to propagate a `DriverError` on
/// first-call failure) is not available on this project's pinned
/// minimum rustc (1.75, see Cargo.toml); this only locks briefly to
/// clone the `Arc` or initialize once, never for the actual kernel work.
static CUDA_DEVICE: std::sync::Mutex<Option<std::sync::Arc<cudarc::driver::CudaDevice>>> =
    std::sync::Mutex::new(None);

/// The shared device's compute capability major, asked once. `0` when
/// the query fails, which no capability gate treats as "new enough".
pub(crate) fn compute_capability_major(dev: &std::sync::Arc<cudarc::driver::CudaDevice>) -> i32 {
    static CC_MAJOR: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *CC_MAJOR.get_or_init(|| {
        dev.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .unwrap_or(0)
    })
}

/// The shared device's compute capability `(major, minor)`, asked once;
/// `(0, 0)` when the query fails.
pub(crate) fn compute_capability(dev: &std::sync::Arc<cudarc::driver::CudaDevice>) -> (i32, i32) {
    static CC: std::sync::OnceLock<(i32, i32)> = std::sync::OnceLock::new();
    *CC.get_or_init(|| {
        let minor = dev
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .unwrap_or(0);
        (compute_capability_major(dev), minor)
    })
}

/// The shared device's multiprocessor count, asked once; `1` when the
/// query fails.
pub(crate) fn multiprocessor_count(dev: &std::sync::Arc<cudarc::driver::CudaDevice>) -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        dev.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .map_or(1, |n| n.max(1) as usize)
    })
}

pub(crate) fn shared_device() -> Result<std::sync::Arc<cudarc::driver::CudaDevice>, CudaError> {
    let mut guard = CUDA_DEVICE.lock().unwrap();
    if let Some(dev) = guard.as_ref() {
        return Ok(dev.clone());
    }
    let dev =
        cudarc::driver::CudaDevice::new(0).map_err(|e| CudaError::DriverInit(format!("{e:?}")))?;
    *guard = Some(dev.clone());
    Ok(dev)
}

// The module cache moved to `crate::modules`; re-exported so every
// existing path still resolves.
pub(crate) use crate::modules::{
    ensure_module_loaded, ensure_module_loaded_lazy, ensure_module_loaded_lazy_for_arch,
};

// The resident weight cache moved to `crate::resident`; re-exported
// here so `crate::gpu::resident_cuda_weights` still resolves.
pub(crate) use crate::resident::{resident_cuda_f32, resident_cuda_weights, ResidentCudaWeights};

// The matvec launch path (the plan that pairs a kernel with its
// geometry, the enqueue primitives, the multi-matvec and fused-FFN
// launches) moved to `crate::matvec_launch`; re-exported so every
// existing path still resolves.
#[cfg(test)]
use crate::matvec_kinds::coalesced::coalesced_kind;
pub(crate) use crate::matvec_launch::{
    alloc_out, enqueue_matvec_in, enqueue_matvec_multi, enqueue_matvec_multi_table, launch_matvec,
    MatvecInput, MultiShape,
};
#[cfg(test)]
use crate::matvec_launch::{
    enqueue_matvec, enqueue_matvec_at, matvec_launch_plan, COALESCED_WARPS_PER_BLOCK,
};
pub use crate::matvec_launch::{
    launch_dense_ffn_swiglu, launch_matvec_multi, time_matvec, MatvecLaunch,
};
// The kernel TEXT moved to `matvec_kinds`, one module per format
// family; `gpu.rs` keeps the device plumbing. Re-exported so every
// existing path still resolves -- a refactor that moves a definition
// should not move a public name.
// The launchers of the matvecs generated from a GEMM dequant, one line
// each in `gen_launch`; re-exported so every kind's launcher has the
// same path.
pub use crate::gen_launch::*;
pub use crate::matvec_kinds::coalesced::{
    Q4_K_MATVEC_COALESCED_KERNEL_SRC, Q5_K_MATVEC_COALESCED_KERNEL_SRC,
    Q6_K_MATVEC_COALESCED_KERNEL_SRC, Q8_0_MATVEC_COALESCED_KERNEL_SRC,
};
pub use crate::matvec_kinds::codebook::{
    IQ4_NL_MATVEC_KERNEL_SRC, IQ4_XS_MATVEC_KERNEL_SRC, MXFP4_MATVEC_KERNEL_SRC,
};
pub use crate::matvec_kinds::kquant::{
    Q2_K_MATVEC_KERNEL_SRC, Q3_K_MATVEC_KERNEL_SRC, Q4_K_MATVEC_KERNEL_SRC, Q5_K_MATVEC_KERNEL_SRC,
    Q6_K_MATVEC_KERNEL_SRC,
};
pub use crate::matvec_kinds::legacy::{
    Q4_0_MATVEC_KERNEL_SRC, Q5_0_MATVEC_KERNEL_SRC, Q8_0_MATVEC_KERNEL_SRC,
};

/// The per-kind matvec table, keyed by GGUF quant name: source, NVRTC
/// module cache key, entry point. `None` means CUDA has no matvec for
/// that format and the caller must fall back and say so.
///
/// This is the SINGLE holder of those three strings. It was not: the
/// same five-row table was written out twice more, inline in
/// `frink-core`'s `apply_gpu_multi` and `apply_gpu_dense_ffn_swiglu`,
/// and a kind added to one and not the others silently lost the fused
/// launch while the capability report kept saying GPU. That is the
/// shape `frink-metal`'s `matvec_launch_meta` already exists to
/// prevent, and this is its CUDA counterpart -- keyed the same way, by
/// `QuantKind::name()`, so `frink-core` can ask for a kind without
/// depending on a CUDA type.
pub fn matvec_launch_meta(kind_name: &str) -> Option<(&'static str, &'static str, &'static str)> {
    crate::matvec_kinds::kind_by_name(kind_name).map(|k| (k.source(), k.module_name, k.fn_name))
}

/// Looks a kind up in [`matvec_launch_meta`] and launches it. Every
/// `launch_q*_matvec` below is this call with its name filled in, so
/// the launchers cannot name a module or entry point the table does
/// not.
pub(crate) fn launch_matvec_by_kind(
    kind_name: &str,
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    let (kernel_src, module_name, fn_name) = matvec_launch_meta(kind_name)
        .ok_or_else(|| CudaError::Unsupported(format!("no CUDA matvec kernel for {kind_name}")))?;
    launch_matvec(
        kernel_src,
        module_name,
        fn_name,
        weights,
        x,
        rows,
        row_bytes,
        n_blocks_per_row,
    )
}

/// Launches the Q8_0 matvec kernel. Verified on real GPU hardware -- see module docs.
pub fn launch_q8_0_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("Q8_0", weights, x, rows, row_bytes, n_blocks_per_row)
}

/// Launches the Q4_0 matvec kernel. Verified on real GPU hardware -- see module docs.
pub fn launch_q4_0_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("Q4_0", weights, x, rows, row_bytes, n_blocks_per_row)
}

/// Launches the Q5_0 matvec kernel. **NEVER RUN ON A GPU** -- see
/// `Q5_0_MATVEC_KERNEL_SRC`'s doc comment for what is and is not
/// established about it.
pub fn launch_q5_0_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("Q5_0", weights, x, rows, row_bytes, n_blocks_per_row)
}

/// Launches the Q2_K matvec kernel. **NEVER RUN ON A GPU** -- see
/// [`crate::matvec_kinds::kquant`] for what is and is not established.
pub fn launch_q2_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("Q2_K", weights, x, rows, row_bytes, n_blocks_per_row)
}

/// Launches the Q3_K matvec kernel. **NEVER RUN ON A GPU** -- see
/// [`crate::matvec_kinds::kquant`] for what is and is not established.
pub fn launch_q3_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("Q3_K", weights, x, rows, row_bytes, n_blocks_per_row)
}

/// Launches the IQ4_NL matvec kernel. **NEVER RUN ON A GPU** -- see
/// [`crate::matvec_kinds::codebook`] for what is and is not established.
pub fn launch_iq4_nl_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("IQ4_NL", weights, x, rows, row_bytes, n_blocks_per_row)
}

/// Launches the IQ4_XS matvec kernel. **NEVER RUN ON A GPU** -- see
/// [`crate::matvec_kinds::codebook`] for what is and is not established.
pub fn launch_iq4_xs_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("IQ4_XS", weights, x, rows, row_bytes, n_blocks_per_row)
}

/// Launches the MXFP4 matvec kernel. **NEVER RUN ON A GPU** -- see
/// [`crate::matvec_kinds::codebook`] for what is and is not established.
pub fn launch_mxfp4_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("MXFP4", weights, x, rows, row_bytes, n_blocks_per_row)
}

/// Launches the Q4_K matvec kernel. Verified on real GPU hardware -- see module docs.
pub fn launch_q4_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("Q4_K", weights, x, rows, row_bytes, n_blocks_per_row)
}

/// Launches the Q5_K matvec kernel. Verified on real GPU hardware -- see module docs.
pub fn launch_q5_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("Q5_K", weights, x, rows, row_bytes, n_blocks_per_row)
}

/// Launches the Q6_K matvec kernel. NOT yet re-verified on real GPU
/// hardware since the signed-scale fix -- see `Q6_K_MATVEC_KERNEL_SRC`'s
/// doc comment.
pub fn launch_q6_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    launch_matvec_by_kind("Q6_K", weights, x, rows, row_bytes, n_blocks_per_row)
}

pub(crate) use crate::glu::glu_mul_device;
pub use crate::glu::{
    GluCuda, GELU_MUL_KERNEL_SRC, SILU_MUL_KERNEL_SRC, SWIGLU_OAI_ALPHA, SWIGLU_OAI_LIMIT,
    SWIGLU_OAI_SRC,
};

/// CUDA C source for a fused residual add + RMSNorm:
/// `out[i] = (x[i] + residual[i]) / sqrt(mean((x+residual)^2) + eps) * weight[i]`.
/// Mirrors `frink_core::matmul::rms_norm` applied to `x + residual` and
/// `frink-metal/src/elem.rs`'s `rms_norm_f32` kernel (add is fused in
/// the first pass over elements instead of a separate vec-add launch).
pub const FUSED_ADD_RMSNORM_KERNEL_SRC: &str = r#"
extern "C" __global__ void fused_add_rmsnorm_f32(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    int n,
    float eps
) {
    __shared__ float partial[256];
    int tid = threadIdx.x;
    int tg = blockDim.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += tg) {
        float v = x[i] + residual[i];
        acc += v * v;
    }
    partial[tid] = acc;
    __syncthreads();
    for (int stride = tg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf(partial[0] / (float)n + eps);
    for (int i = tid; i < n; i += tg) {
        float v = x[i] + residual[i];
        out[i] = v * inv_rms * weight[i];
    }
}
"#;

/// Computes `rms_norm(x + residual, weight, eps)` on the device: one
/// HtoD upload each for `x`, `residual`, and `weight`, one DtoH for the
/// result. Fuses the elementwise add with the RMSNorm reduction so the
/// sum never materializes on the host.
pub fn launch_fused_add_rmsnorm(
    x: &[f32],
    residual: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, CudaError> {
    use cudarc::driver::LaunchAsync;

    assert_eq!(x.len(), residual.len());
    assert_eq!(x.len(), weight.len());
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let dev = shared_device()?;
    ensure_module_loaded(
        &dev,
        FUSED_ADD_RMSNORM_KERNEL_SRC,
        "frink_fused_add_rmsnorm",
        "fused_add_rmsnorm_f32",
    )?;
    let func = dev
        .get_func("frink_fused_add_rmsnorm", "fused_add_rmsnorm_f32")
        .ok_or_else(|| {
            CudaError::KernelCompile(
                "function 'fused_add_rmsnorm_f32' not found after load_ptx".to_string(),
            )
        })?;

    let d_x = crate::xfer::htod(&dev, "fused_add_rmsnorm x upload", x)?;
    let d_residual = crate::xfer::htod(&dev, "fused_add_rmsnorm residual upload", residual)?;
    let d_weight = crate::xfer::htod(&dev, "fused_add_rmsnorm weight upload", weight)?;
    let mut d_out = dev
        .alloc_zeros::<f32>(n)
        .map_err(|e| CudaError::Launch(format!("fused_add_rmsnorm out alloc: {e:?}")))?;

    let block = 256u32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
    };
    unsafe {
        func.launch(
            cfg,
            (&d_x, &d_residual, &d_weight, &mut d_out, n as i32, eps),
        )
        .map_err(|e| CudaError::Launch(format!("fused_add_rmsnorm launch: {e:?}")))?;
    }

    crate::xfer::dtoh(&dev, "fused_add_rmsnorm download", &d_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The process's panic hook is the binary's: a server journals every
    /// panic through it. Nothing in this crate may swap it, even briefly,
    /// because a swap silences whatever another thread panics with in the
    /// window (`probe` did, and lost the server journal test's record).
    #[test]
    fn no_source_here_swaps_the_process_panic_hook() {
        // Spelled in pieces so this test's own text does not match.
        let calls = [concat!("set_", "hook("), concat!("take_", "hook(")];
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut stack = vec![src];
        let mut scanned = 0;
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    for call in calls {
                        assert!(!text.contains(call), "{} calls {call}", path.display());
                    }
                    scanned += 1;
                }
            }
        }
        assert!(scanned > 20, "scanned {scanned} files");
    }

    #[test]
    fn probe_degrades_cleanly_regardless_of_real_hardware_presence() {
        // This assertion used to hard-code "no CUDA device here," true
        // on the sandbox this was written on but false the first time
        // this actually ran on rented GPU hardware
        // (real RTX 3060, 2026-07-31) --
        // caught a real test bug, not a probe() bug: a CI-style test
        // that only passes in one specific environment isn't testing
        // the real invariant (probe() never panics and reports
        // something sane either way), it's testing "this machine has no
        // GPU," which isn't frink's to assert. Check both real outcomes
        // instead of assuming one.
        match probe() {
            None => {} // no device present -- also correct, nothing further to check
            Some(info) => {
                assert!(
                    info.device_count >= 1,
                    "reported a device but device_count=0"
                );
                assert!(
                    info.total_vram_bytes > 0,
                    "reported a device but total_vram_bytes=0"
                );
                assert!(
                    info.free_vram_bytes <= info.total_vram_bytes,
                    "free VRAM ({}) cannot exceed total ({})",
                    info.free_vram_bytes,
                    info.total_vram_bytes
                );
            }
        }
    }

    /// Builds `rows` real, non-trivial Q8_0-quantized rows (not all-
    /// zero weights, which would pass trivially even with a broken
    /// kernel) using `frink_quant::quantize_q8_0`, so the ignored GPU
    /// tests below check actual numerical agreement with the CPU
    /// reference, not just "the launch didn't error." The reference dots
    /// the activation the matvec reads ([`reads`]): its q8_1 rounding on
    /// the dp4a kernel, the f32 values on the coalesced one.
    fn real_q8_0_test_matrix(rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let mut weights = Vec::new();
        let mut all_rows_f32 = Vec::new();
        for r in 0..rows {
            let row: Vec<f32> = (0..cols)
                .map(|i| (((r * cols + i) as f32) * 0.037).sin())
                .collect();
            weights.extend(frink_quant::quantize_q8_0(&row));
            all_rows_f32.push(row);
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.019).cos()).collect();
        let xr = reads("q8_0_matvec", cols / frink_quant::Q8_0_BLOCK_ELEMS, &x);
        let expected: Vec<f32> = all_rows_f32
            .iter()
            .map(|row| frink_quant::dot_q8_0_f32_scalar(&frink_quant::quantize_q8_0(row), &xr))
            .collect();
        (weights, x, expected)
    }

    /// What the decode matvec `fn_name` reads of `x` in this process:
    /// the q8_1 rounding under the dp4a kernels (the default for Q4_K,
    /// Q6_K, Q8_0 and MXFP4), `x` itself under the f32 kernels, by the
    /// same once-per-process switch the launch reads.
    fn reads(fn_name: &str, n_blocks: usize, x: &[f32]) -> Vec<f32> {
        crate::matvec_kinds::dp4a::matvec_reads(fn_name, n_blocks, x)
    }

    #[test]
    #[ignore = "requires real CUDA hardware -- verified passing on an RTX 3060 (vast.ai); run with --ignored on a CUDA-capable machine to re-verify; reference now dots the matvec's own input (q8_1 under the dp4a default), not yet run on hardware with it"]
    fn launch_q8_0_matvec_matches_cpu_reference() {
        // Run manually on a machine with an actual CUDA device:
        //   cargo test -p frink-cuda --features cuda -- --ignored
        let rows = 4;
        let cols = 64;
        let row_bytes = (cols / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES;
        let (weights, x, expected) = real_q8_0_test_matrix(rows, cols);

        let result = launch_q8_0_matvec(
            &weights,
            &x,
            rows,
            row_bytes,
            cols / frink_quant::Q8_0_BLOCK_ELEMS,
        )
        .expect("kernel launch must succeed on real CUDA hardware");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "row {i}: GPU={got} CPU reference={want}"
            );
        }
    }

    #[test]
    #[ignore = "requires real CUDA hardware -- verified passing on an RTX 3060 (vast.ai); run with --ignored on a CUDA-capable machine to re-verify"]
    fn launch_q4_0_matvec_matches_cpu_reference() {
        // Run manually on a machine with an actual CUDA device:
        //   cargo test -p frink-cuda --features cuda -- --ignored
        let rows = 4;
        let cols = 64;
        let blocks_per_row = cols / frink_quant::Q4_0_BLOCK_ELEMS;
        let row_bytes = blocks_per_row * frink_quant::Q4_0_BLOCK_BYTES;

        // Regression note: an earlier version of this test built exactly
        // one 18-byte Q4_0 block per row regardless of `cols`, so for
        // cols=64 (2 blocks/row, row_bytes=36) the `weights` buffer ended
        // up 72 bytes instead of the 144 bytes `row_bytes_slice` below
        // actually indexes into -- caught by a real out-of-bounds panic
        // the first time this test ran on real GPU hardware. Fixed by
        // building `blocks_per_row` blocks per row, matching `cols`.
        let mut weights = Vec::new();
        for r in 0..rows {
            for b in 0..blocks_per_row {
                weights.extend_from_slice(
                    &half::f16::from_f32(0.05 + (r * blocks_per_row + b) as f32 * 0.01)
                        .to_le_bytes(),
                );
                for i in 0..16u8 {
                    let lo = (i + r as u8 + b as u8) % 16;
                    let hi = (15 - i + r as u8 + b as u8) % 16;
                    weights.push(lo | (hi << 4));
                }
            }
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.09).sin()).collect();
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                let row_bytes_slice = &weights[r * row_bytes..(r + 1) * row_bytes];
                frink_quant::dot_q4_0_f32_scalar(row_bytes_slice, &x)
            })
            .collect();

        let result = launch_q4_0_matvec(
            &weights,
            &x,
            rows,
            row_bytes,
            cols / frink_quant::Q4_0_BLOCK_ELEMS,
        )
        .expect("kernel launch must succeed on real CUDA hardware");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "row {i}: GPU={got} CPU reference={want}"
            );
        }
    }

    /// Every row of the matvec table has to be launchable through
    /// [`matvec_launch_meta`], and the meta lookup has to be the table
    /// rather than a second copy of it.
    ///
    /// This used to spell six kind names and six negatives. It is
    /// driven by [`crate::matvec_kinds::KINDS`] now, so a format added
    /// to the table is covered here the moment it exists, and the
    /// negatives moved to `matvec_kinds`'s own
    /// `a_kind_with_no_matvec_does_not_resolve` -- where the table is.
    /// Naming three kinds that "must not resolve" beside a table that
    /// decides it is exactly the pair of structures this repo keeps
    /// paying for: IQ4_XS was on that negative list until it got a
    /// kernel, and moving it meant remembering two files.
    #[test]
    fn matvec_launch_meta_is_the_matvec_table() {
        for k in crate::matvec_kinds::KINDS {
            let (src, module, func) = matvec_launch_meta(k.name)
                .unwrap_or_else(|| panic!("{}: in the table but not launchable", k.name));
            assert_eq!(module, k.module_name, "{}", k.name);
            assert_eq!(func, k.fn_name, "{}", k.name);
            assert!(
                src.contains(&format!("void {func}(")),
                "{}: the source in {module} does not define {func}",
                k.name
            );
        }
        assert!(
            matvec_launch_meta("Q8_1").is_none(),
            "a kind with no kernel resolved to a CUDA matvec"
        );
    }

    /// The Q5_0 kernel's block stride and element stride are literals in
    /// CUDA C, so nothing but this holds them to the format's real
    /// geometry. A wrong stride walks the row past the first block and
    /// every value after it is garbage -- and `nl` is 2 here, not the
    /// K-quants' 16, which is the assumption easiest to carry over by
    /// mistake.
    #[test]
    fn the_q5_0_matvec_strides_by_the_real_block_geometry() {
        assert_eq!(frink_quant::Q5_0_BLOCK_BYTES, 22);
        assert_eq!(frink_quant::Q5_0_BLOCK_ELEMS, 32);
        assert!(
            Q5_0_MATVEC_KERNEL_SRC.contains("(size_t)b * 22"),
            "the Q5_0 matvec does not stride by Q5_0_BLOCK_BYTES"
        );
        assert!(
            Q5_0_MATVEC_KERNEL_SRC.contains("int base = b * 32;"),
            "the Q5_0 matvec does not step the activation by Q5_0_BLOCK_ELEMS"
        );
    }

    #[test]
    #[ignore = "requires real CUDA hardware -- NEVER RUN: the Q5_0 matvec has never executed on a GPU. Run with --ignored on a CUDA-capable machine and record the result before any doc claims CUDA Q5_0 works"]
    fn launch_q5_0_matvec_matches_cpu_reference() {
        // Run manually on a machine with an actual CUDA device:
        //   cargo test -p frink-cuda --features cuda -- --ignored
        let rows = 4;
        let cols = 64;
        let blocks_per_row = cols / frink_quant::Q5_0_BLOCK_ELEMS;
        let row_bytes = blocks_per_row * frink_quant::Q5_0_BLOCK_BYTES;

        // `frink_quant` has no Q5_0 quantizer (the format is load-only
        // here), so the fixture is built byte by byte. The scale is
        // pinned finite -- a random f16 is NaN or Inf often enough to
        // be the usual outcome -- while `qh` is deliberately varied, so
        // both the low half's fifth bit (`qh` bit j) and the high
        // half's (`qh` bit j + 16) are set on some elements and clear
        // on others. A kernel that dropped `qh` entirely would still
        // produce plausible numbers against a fixture where it is zero.
        let mut weights = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let idx = (r * blocks_per_row + b) as u32;
                weights.extend_from_slice(
                    &half::f16::from_f32(0.05 + idx as f32 * 0.01).to_le_bytes(),
                );
                let qh = 0x9E3D_7A51u32.wrapping_mul(idx + 1);
                weights.extend_from_slice(&qh.to_le_bytes());
                for i in 0..16u8 {
                    let lo = (i + r as u8 + b as u8) % 16;
                    let hi = (15 - i + r as u8 + b as u8) % 16;
                    weights.push(lo | (hi << 4));
                }
            }
        }
        assert_eq!(weights.len(), rows * row_bytes);

        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.09).sin()).collect();
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                frink_quant::dot_q5_0_f32_scalar(&weights[r * row_bytes..(r + 1) * row_bytes], &x)
            })
            .collect();

        let result = launch_q5_0_matvec(&weights, &x, rows, row_bytes, blocks_per_row)
            .expect("kernel launch must succeed on real CUDA hardware");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            // Absolute, like the Q8_0/Q4_0 tests and unlike the K-quant
            // ones: Q5_0 accumulates an exact integer dot per block and
            // applies the scale once, so there is no per-element
            // scale multiply for an FMA to reassociate.
            assert!(
                (got - want).abs() < 1e-2,
                "row {i}: GPU={got} CPU reference={want}"
            );
        }
    }

    /// Deterministic pseudo-random byte generator for building real,
    /// non-trivial K-quant block bytes (no `quantize_qX_k` producer
    /// exists in `frink_quant` -- these formats are load-only, never
    /// produced by frink -- so tests build arbitrary-but-valid-shaped
    /// bytes directly, the same convention `frink-core::weight_matrix`'s
    /// and `frink-models::kimi_loader`'s own MXFP4 tests already use).
    fn pseudo_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    /// GPU-vs-CPU agreement check for the K-quant kernels below, using a
    /// *relative* error bound instead of the fixed absolute `1e-2` the
    /// Q8_0/Q4_0 tests above use. The K-quant kernels apply each block's
    /// scale/min *inside* a per-element `acc += (d1 * q - min1) * x[i]`
    /// accumulation (hundreds of float multiply-adds per row), unlike
    /// Q8_0/Q4_0's structure (accumulate an exact integer dot product
    /// first, multiply by scale once at the end) -- so K-quant results are
    /// exposed to per-element GPU-vs-CPU float rounding differences
    /// (nvcc/NVRTC contracts `a*b+c`-shaped expressions into a single-
    /// rounding `fma` instruction by default; plain Rust `f32` arithmetic
    /// does not auto-contract). Real hardware run (RTX 3060, 2026-07-31)
    /// measured relative errors around 1e-7 (e.g. diff=40 against a
    /// ~97,454,420-magnitude reference) -- consistent with float32
    /// machine epsilon times a handful of differently-ordered rounding
    /// steps, not a logic bug. This is exactly why llama.cpp's own real
    /// CUDA K-quant kernels (`vec_dot_q6_K_q8_1_impl_mmvq`,
    /// `ggml/src/ggml-cuda/vecdotq.cuh`) quantize activations to int8 too
    /// and reduce via an *integer* `dp4a` dot product, converting to
    /// float only once per block for the final scale multiply --
    /// deliberately avoiding this exact class of divergence -- and why
    /// llama.cpp's own backend-comparison tests
    /// (`tests/test-backend-ops.cpp`) check `max_nmse_err()` (a relative
    /// error metric), never exact/absolute equality, for quantized ops.
    fn assert_close_relative(got: f32, want: f32, row: usize) {
        // Real hardware run (RTX 3060, 2026-07-31) hit `GPU=NaN CPU
        // reference=NaN` for Q6_K row 1: `real_k_quant_test_matrix`'s
        // pseudo-random block bytes aren't pinned to safe values (unlike
        // some other tests here), so by chance a row's raw `d`/scale
        // bytes decoded as an f16 NaN/Inf bit pattern -- garbage in,
        // garbage out, identically on both backends. GPU and CPU *agree*
        // in that case (both NaN from the same degenerate input), which
        // is a real pass, not a failure to paper over; plain `(got -
        // want).abs() <= tol` fails to see that because IEEE754 NaN
        // comparisons are always false. Only a mismatched
        // NaN-vs-non-NaN pairing is a genuine disagreement.
        if want.is_nan() {
            assert!(
                got.is_nan(),
                "row {row}: CPU reference is NaN but GPU={got} is not"
            );
            return;
        }
        let tol = 1e-4 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "row {row}: GPU={got} CPU reference={want} (relative tolerance {tol})"
        );
    }

    /// Builds `rows` real (non-zero, non-trivial) blocks of `block_bytes`
    /// each for a K-quant format, and the matching `expected` output via
    /// `scalar_dot` (`frink_quant::dot_q{4,5,6}_k_f32_scalar` -- already
    /// independently verified elsewhere in this workspace), so the
    /// ignored GPU tests below check real numerical agreement with that
    /// trusted CPU reference, not just "the launch didn't error."
    ///
    /// `fn_name` is the kind's matvec, so the reference dots what that
    /// matvec reads ([`reads`]).
    fn real_k_quant_test_matrix(
        rows: usize,
        cols: usize,
        block_bytes: usize,
        fn_name: &str,
        scalar_dot: impl Fn(&[u8], &[f32]) -> f32,
    ) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let n_blocks_per_row = cols / 256;
        let row_bytes = n_blocks_per_row * block_bytes;
        let mut weights = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            weights.extend(pseudo_bytes(r as u32 + 1, row_bytes));
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.021).sin()).collect();
        let xr = reads(fn_name, n_blocks_per_row, &x);
        let expected: Vec<f32> = (0..rows)
            .map(|r| scalar_dot(&weights[r * row_bytes..(r + 1) * row_bytes], &xr))
            .collect();
        (weights, x, expected)
    }

    /// Every coalesced kernel, at BOTH widths, against its scalar twin
    /// on a real device. The twins prove the lane mapping without a GPU
    /// and `tools/matvec_host_check` runs the CUDA C text on a host
    /// CPU; this proves NVRTC accepts it and that the vector loads, the
    /// alignment they assume and the warp-shuffle reduction hold on
    /// hardware, none of which the other two can see.
    ///
    /// Walks [`crate::matvec_kinds::coalesced::COALESCED`] rather than a
    /// list of its own, so a new kind cannot be added with its hardware
    /// check quietly left out. Block counts 3 and 12: 3 is fewer than a
    /// wide warp's blocks per trip for every kind (so some lanes idle),
    /// 12 is a 3072-wide K-quant row.
    ///
    ///   cargo test -p frink-cuda --features cuda -- --ignored
    #[test]
    #[ignore = "requires real CUDA hardware. Passed 2026-11-09 on an A10G (sm_86, Hugging Face Jobs) at both widths for every coalesced kind, MXFP4 included; the narrow width earlier on an RTX 3060/3070"]
    fn every_coalesced_matvec_matches_its_twin_at_both_widths() {
        use crate::matvec_kinds::coalesced::{MatvecWidth, COALESCED};
        let rows = 37usize; // not a multiple of the warps per block
        let dev = super::shared_device().expect("device");
        for kind in COALESCED {
            let twin = crate::coalesced_twin::twin_for(kind.entry)
                .unwrap_or_else(|| panic!("{} has no twin", kind.entry));
            let mv = crate::matvec_kinds::KINDS
                .iter()
                .find(|m| m.fn_name == kind.base_fn)
                .expect("routed from a MatvecKind");
            let mm = crate::mul_mm::kind_by_name(mv.name).expect("geometry");
            for (width, wpr) in [
                (MatvecWidth::Narrow, 1),
                (MatvecWidth::Wide, 1),
                (MatvecWidth::Wide, 2),
                (MatvecWidth::Wide, 4),
                (MatvecWidth::Narrow, 8),
            ] {
                for (n_blocks_per_row, seed) in [(3usize, 7u32), (12, 23)] {
                    let row_bytes = n_blocks_per_row * mm.block_bytes;
                    let cols = n_blocks_per_row * mm.block_elems;
                    let mut weights = pseudo_bytes(seed, rows * row_bytes);
                    // MXFP4's E8M0 scale near 2^0: a random byte reaches
                    // 2^128, an infinity no relative tolerance can hold.
                    if mv.name == "MXFP4" {
                        for b in weights.chunks_mut(mm.block_bytes) {
                            b[0] = 120 + b[0] % 16;
                        }
                    }
                    let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.021).sin()).collect();
                    // Enter through the ordinary kernel identity, not the
                    // coalesced one, so this exercises the ROUTING too.
                    let launch = super::MatvecLaunch {
                        kernel_src: mv.source(),
                        module_name: mv.module_name,
                        fn_name: mv.fn_name,
                        weights: &weights,
                        rows,
                        row_bytes,
                        n_blocks_per_row,
                    };
                    let d_x = dev.htod_copy(x.clone()).expect("upload");
                    let (d_out, _w) = super::enqueue_matvec_at(&dev, &launch, &d_x, width, wpr)
                        .unwrap_or_else(|e| panic!("{} {width:?} x{wpr}: {e:?}", mv.name));
                    let got = dev.dtoh_sync_copy(&d_out).expect("download");
                    assert_eq!(got.len(), rows, "{}", mv.name);
                    let lanes = kind.lanes(width);
                    for r in 0..rows {
                        let row = &weights[r * row_bytes..(r + 1) * row_bytes];
                        let want = twin(row, &x, n_blocks_per_row, lanes);
                        assert!(
                            !(want.is_nan() ^ got[r].is_nan()),
                            "{} {width:?} x{wpr} row {r}: GPU={} twin={want}",
                            mv.name,
                            got[r]
                        );
                        assert_close_relative(got[r], want, r);
                    }
                }
            }
        }
    }

    /// Several experts' projections in one launch equal the same
    /// projections launched one at a time, bit for bit: a slot of the
    /// routed-expert form is the single kernel's arithmetic on its own
    /// weights, input and output. Once with one input for every expert
    /// (gate, up) and once with an input of its own each (down).
    #[test]
    #[ignore = "requires real CUDA hardware. Passed 2026-11-09 on an A10G (sm_86, Hugging Face Jobs)"]
    fn several_experts_in_one_launch_equal_one_launch_each() {
        use crate::matvec_kinds::coalesced::COALESCED;
        let dev = super::shared_device().expect("device");
        let (rows, k) = (37usize, 3usize);
        for kind in COALESCED {
            let mv = crate::matvec_kinds::KINDS
                .iter()
                .find(|m| m.fn_name == kind.base_fn)
                .expect("routed from a MatvecKind");
            let mm = crate::mul_mm::kind_by_name(mv.name).expect("geometry");
            let n_blocks_per_row = 12;
            let row_bytes = n_blocks_per_row * mm.block_bytes;
            let cols = n_blocks_per_row * mm.block_elems;
            let weights: Vec<Vec<u8>> = (0..k as u32)
                .map(|e| crate::mul_mm_ref::fixtures::weights(mm, rows, cols, 50 + e))
                .collect();
            let launches: Vec<super::MatvecLaunch<'_>> = weights
                .iter()
                .map(|w| super::MatvecLaunch {
                    kernel_src: mv.source(),
                    module_name: mv.module_name,
                    fn_name: mv.fn_name,
                    weights: w,
                    rows,
                    row_bytes,
                    n_blocks_per_row,
                })
                .collect();
            for x_stride in [0, cols] {
                let x: Vec<f32> = (0..k * cols).map(|i| ((i as f32) * 0.017).sin()).collect();
                let d_x = dev.htod_copy(x.clone()).unwrap();
                let (d_all, _held) =
                    super::enqueue_matvec_multi(&dev, &launches, &MatvecInput::new(&d_x), x_stride)
                        .expect("multi");
                let all = dev.dtoh_sync_copy(&d_all).unwrap();
                for (e, launch) in launches.iter().enumerate() {
                    let xe = dev
                        .htod_copy(x[e * x_stride..e * x_stride + cols].to_vec())
                        .unwrap();
                    let (d_one, _w) = super::enqueue_matvec(&dev, launch, &xe).expect("one");
                    let one = dev.dtoh_sync_copy(&d_one).unwrap();
                    let got = &all[e * rows..(e + 1) * rows];
                    for (r, (a, b)) in got.iter().zip(&one).enumerate() {
                        assert!(
                            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                            "{} expert {e} row {r} stride {x_stride}: multi {a} one {b}",
                            mv.name
                        );
                    }
                }
            }
        }
    }

    /// A coalesced kernel launched with the block-per-row geometry
    /// indexes as if it had one row per warp while getting one row per
    /// block, and returns wrong numbers rather than failing. So the
    /// kernel and its geometry are chosen together, and this checks
    /// they stay that way at both widths -- on the host, with no GPU.
    #[test]
    fn the_plan_pairs_each_kernel_with_its_own_geometry() {
        use crate::matvec_kinds::coalesced::MatvecWidth;
        let rows = 37usize; // not a multiple of the warps per block

        for kind in ["Q4_K", "Q5_K", "Q6_K", "Q8_0"] {
            let (src, module, fn_name) = super::matvec_launch_meta(kind).expect("kernel");
            for (width, wpr) in [MatvecWidth::Narrow, MatvecWidth::Wide]
                .into_iter()
                .flat_map(|w| [1usize, 2, 4, 8].map(|r| (w, r)))
            {
                let plan = super::matvec_launch_plan(fn_name, src, module, rows, width, wpr);
                let ck = super::coalesced_kind(fn_name).expect("coalesced");
                assert!(
                    plan.entry.ends_with("_coalesced"),
                    "{kind}: plan chose {}, not the coalesced kernel",
                    plan.entry
                );
                assert_eq!(
                    plan.module,
                    ck.module_for(width, wpr),
                    "{kind} {width:?} x{wpr}: module key"
                );
                assert_eq!(
                    wpr == 1,
                    plan.module == ck.module(width),
                    "{kind}: one key per wpr"
                );
                let text = plan.text();
                assert!(
                    text.contains(&format!("void {}(", plan.entry)),
                    "{kind}: plan's source does not define {}",
                    plan.entry
                );
                assert!(
                    text.starts_with(&format!(
                        "#define FX_LANES {}\n#define FX_WPR {wpr}\n",
                        ck.lanes(width)
                    )),
                    "{kind} {width:?} x{wpr}: source compiled for the wrong geometry"
                );
                let cfg = plan.cfg;
                // Rows per block: the block's warps over the row's warps.
                let warps = (super::COALESCED_WARPS_PER_BLOCK / wpr) as u32;
                assert_eq!(
                    cfg.block_dim,
                    (super::COALESCED_WARPS_PER_BLOCK as u32 * 32, 1, 1),
                    "{kind}: block_dim"
                );
                assert_eq!(cfg.shared_mem_bytes, 0, "{kind}: needs no shared memory");
                // Every row must land in some warp, including the tail.
                assert!(
                    cfg.grid_dim.0 * warps >= rows as u32,
                    "{kind}: {} blocks x {warps} warps does not cover {rows} rows",
                    cfg.grid_dim.0
                );
                assert!(
                    (cfg.grid_dim.0 - 1) * warps < rows as u32,
                    "{kind}: {} blocks is more than the tail needs",
                    cfg.grid_dim.0
                );
            }
        }

        // A kind with no coalesced kernel keeps the old geometry, and
        // keeps its own kernel rather than silently getting another's,
        // whatever the width.
        let (src, module, fn_name) = super::matvec_launch_meta("Q4_0").expect("kernel");
        for width in [MatvecWidth::Narrow, MatvecWidth::Wide] {
            let plan = super::matvec_launch_plan(fn_name, src, module, rows, width, 4);
            assert_eq!(plan.entry, fn_name);
            assert!(std::ptr::eq(plan.module, module));
            assert_eq!(plan.text(), src);
            assert_eq!(plan.cfg.grid_dim, (rows as u32, 1, 1), "one block per row");
            assert_eq!(plan.cfg.block_dim, (256, 1, 1));
            assert!(
                plan.cfg.shared_mem_bytes > 0,
                "the block-per-row kernels reduce through shared memory"
            );
        }
    }

    #[test]
    #[ignore = "requires real CUDA hardware -- verified passing on an RTX 3060 (vast.ai, 2026-07-31) with the relative-tolerance fix; run with --ignored on a CUDA-capable machine to re-verify; reference now dots the matvec's own input (q8_1 under the dp4a default), not yet run on hardware with it"]
    fn launch_q4_k_matvec_matches_cpu_reference() {
        let rows = 3;
        let cols = 256; // 1 Q4_K super-block per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            frink_quant::Q4_K_BLOCK_BYTES,
            "q4_k_matvec",
            frink_quant::dot_q4_k_f32_scalar,
        );

        let result = launch_q4_k_matvec(&weights, &x, rows, frink_quant::Q4_K_BLOCK_BYTES, 1)
            .expect("kernel launch must succeed on real CUDA hardware");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    #[test]
    #[ignore = "requires real CUDA hardware -- verified passing on an RTX 3060 (vast.ai, 2026-07-31) with the relative-tolerance fix; run with --ignored on a CUDA-capable machine to re-verify"]
    fn launch_q5_k_matvec_matches_cpu_reference() {
        let rows = 3;
        let cols = 256; // 1 Q5_K super-block per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            frink_quant::Q5_K_BLOCK_BYTES,
            "q5_k_matvec",
            frink_quant::dot_q5_k_f32_scalar,
        );

        let result = launch_q5_k_matvec(&weights, &x, rows, frink_quant::Q5_K_BLOCK_BYTES, 1)
            .expect("kernel launch must succeed on real CUDA hardware");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    #[test]
    #[ignore = "requires real CUDA hardware -- 2026-07-31 real RTX 3060 run failed on row 1 with GPU=NaN CPU reference=NaN (assert_close_relative didn't treat NaN==NaN as agreement; fixed, but not yet re-verified on hardware); run with --ignored on a CUDA-capable machine to re-verify; reference now dots the matvec's own input (q8_1 under the dp4a default), not yet run on hardware with it"]
    fn launch_q6_k_matvec_matches_cpu_reference() {
        let rows = 3;
        let cols = 512; // 2 Q6_K super-blocks per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            frink_quant::Q6_K_BLOCK_BYTES,
            "q6_k_matvec",
            frink_quant::dot_q6_k_f32_scalar,
        );

        let result = launch_q6_k_matvec(&weights, &x, rows, frink_quant::Q6_K_BLOCK_BYTES * 2, 2)
            .expect("kernel launch must succeed on real CUDA hardware");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    #[test]
    #[ignore = "requires real CUDA hardware -- verifies that launch_matvec_multi (shared x upload) matches N sequential single-matvec launches; run with --ignored on a CUDA-capable machine; reference now dots the matvec's own input (q8_1 under the dp4a default), not yet run on hardware with it"]
    fn launch_matvec_multi_matches_sequential() {
        // Build three Q4_K matrices (different shapes) that share the same
        // activation dimension.
        let cols = 256;
        let rows_a = 2;
        let rows_b = 3;
        let rows_c = 4;
        let (weights_a, x, expected_a) = real_k_quant_test_matrix(
            rows_a,
            cols,
            frink_quant::Q4_K_BLOCK_BYTES,
            "q4_k_matvec",
            frink_quant::dot_q4_k_f32_scalar,
        );
        let (weights_b, _, expected_b) = real_k_quant_test_matrix(
            rows_b,
            cols,
            frink_quant::Q4_K_BLOCK_BYTES,
            "q4_k_matvec",
            frink_quant::dot_q4_k_f32_scalar,
        );
        let (weights_c, _, expected_c) = real_k_quant_test_matrix(
            rows_c,
            cols,
            frink_quant::Q4_K_BLOCK_BYTES,
            "q4_k_matvec",
            frink_quant::dot_q4_k_f32_scalar,
        );

        let launches = [
            MatvecLaunch {
                kernel_src: Q4_K_MATVEC_KERNEL_SRC,
                module_name: "frink_q4_k",
                fn_name: "q4_k_matvec",
                weights: weights_a.as_slice(),
                rows: rows_a,
                row_bytes: frink_quant::Q4_K_BLOCK_BYTES,
                n_blocks_per_row: 1,
            },
            MatvecLaunch {
                kernel_src: Q4_K_MATVEC_KERNEL_SRC,
                module_name: "frink_q4_k",
                fn_name: "q4_k_matvec",
                weights: weights_b.as_slice(),
                rows: rows_b,
                row_bytes: frink_quant::Q4_K_BLOCK_BYTES,
                n_blocks_per_row: 1,
            },
            MatvecLaunch {
                kernel_src: Q4_K_MATVEC_KERNEL_SRC,
                module_name: "frink_q4_k",
                fn_name: "q4_k_matvec",
                weights: weights_c.as_slice(),
                rows: rows_c,
                row_bytes: frink_quant::Q4_K_BLOCK_BYTES,
                n_blocks_per_row: 1,
            },
        ];

        let results = launch_matvec_multi(&x, &launches)
            .expect("multi-matvec must succeed on real CUDA hardware");
        assert_eq!(results.len(), 3);

        // Check first matrix output.
        assert_eq!(results[0].len(), expected_a.len());
        for (i, (got, want)) in results[0].iter().zip(expected_a.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }

        // Check second matrix output.
        assert_eq!(results[1].len(), expected_b.len());
        for (i, (got, want)) in results[1].iter().zip(expected_b.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }

        // Check third matrix output.
        assert_eq!(results[2].len(), expected_c.len());
        for (i, (got, want)) in results[2].iter().zip(expected_c.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    /// The fused on-device dense SwiGLU FFN
    /// (`launch_dense_ffn_swiglu`: one `x` upload → gate matvec → up
    /// matvec → `silu(gate)*up` kernel → down matvec → one download)
    /// must produce the same result as running the three matvecs
    /// separately on the CPU reference and doing SwiGLU on the host.
    /// This proves the device-resident activation chaining (no per-
    /// matvec DtoH/HtoD) and the `silu_mul_f32` kernel are correct, not
    /// just that the launches didn't error. Q8_0 is used because
    /// `frink_quant::quantize_q8_0` gives an exact producer/consumer
    /// pair, keeping this a real numerical-agreement check.
    #[test]
    #[ignore = "requires real CUDA hardware -- run with --ignored on a CUDA-capable machine to verify the fused dense-FFN activation-residency path; reference now dots the matvec's own input (q8_1 under the dp4a default), not yet run on hardware with it"]
    fn launch_dense_ffn_swiglu_matches_sequential_cpu() {
        let hidden_dim = 64; // 2 Q8_0 blocks per row
        let ffn_dim = 96; // 3 Q8_0 blocks per row; gate/up output length

        let make_row = |cols: usize, seed: f32| -> Vec<f32> {
            (0..cols)
                .map(|i| (((i as f32) - (cols as f32) / 2.0) * 0.013 * seed).sin())
                .collect()
        };
        // Build a quantized [rows, cols] matrix plus keep each row's f32
        // so the CPU reference dots the *same* quantized bytes back.
        let build = |rows: usize, cols: usize, seed: f32| -> (Vec<u8>, usize) {
            let mut packed = Vec::new();
            for r in 0..rows {
                packed.extend(frink_quant::quantize_q8_0(&make_row(cols, seed + r as f32)));
            }
            let row_bytes = (cols / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES;
            (packed, row_bytes)
        };

        let (gate_w, gate_rb) = build(ffn_dim, hidden_dim, 1.0);
        let (up_w, up_rb) = build(ffn_dim, hidden_dim, 2.0);
        let (down_w, down_rb) = build(hidden_dim, ffn_dim, 3.0);
        let x = make_row(hidden_dim, 0.7);

        let blk = |cols: usize| cols / frink_quant::Q8_0_BLOCK_ELEMS;
        let gate = MatvecLaunch {
            kernel_src: Q8_0_MATVEC_KERNEL_SRC,
            module_name: "frink_q8_0",
            fn_name: "q8_0_matvec",
            weights: gate_w.as_slice(),
            rows: ffn_dim,
            row_bytes: gate_rb,
            n_blocks_per_row: blk(hidden_dim),
        };
        let up = MatvecLaunch {
            kernel_src: Q8_0_MATVEC_KERNEL_SRC,
            module_name: "frink_q8_0",
            fn_name: "q8_0_matvec",
            weights: up_w.as_slice(),
            rows: ffn_dim,
            row_bytes: up_rb,
            n_blocks_per_row: blk(hidden_dim),
        };
        let down = MatvecLaunch {
            kernel_src: Q8_0_MATVEC_KERNEL_SRC,
            module_name: "frink_q8_0",
            fn_name: "q8_0_matvec",
            weights: down_w.as_slice(),
            rows: hidden_dim,
            row_bytes: down_rb,
            n_blocks_per_row: blk(ffn_dim),
        };

        let gpu = launch_dense_ffn_swiglu(&gate, &up, &down, &x)
            .expect("fused FFN must launch on real CUDA hardware");

        // CPU reference: gate/up matvecs, host SwiGLU, down matvec.
        let cpu_matvec = |w: &[u8], row_bytes: usize, rows: usize, act: &[f32]| -> Vec<f32> {
            (0..rows)
                .map(|r| {
                    frink_quant::dot_q8_0_f32_scalar(&w[r * row_bytes..(r + 1) * row_bytes], act)
                })
                .collect::<Vec<f32>>()
        };
        // Each matvec dots what it reads: `x` for gate and up, the
        // device's SwiGLU output for down.
        let xr = reads("q8_0_matvec", blk(hidden_dim), &x);
        let g = cpu_matvec(&gate_w, gate_rb, ffn_dim, &xr);
        let u = cpu_matvec(&up_w, up_rb, ffn_dim, &xr);
        let act: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
            .collect();
        let expected = cpu_matvec(
            &down_w,
            down_rb,
            hidden_dim,
            &reads("q8_0_matvec", blk(ffn_dim), &act),
        );

        assert_eq!(gpu.len(), expected.len());
        for (i, (got, want)) in gpu.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    /// Fused `rms_norm(x + residual, weight, eps)` on device must agree
    /// with the CPU reference (`frink_core::matmul::rms_norm` on the
    /// elementwise sum).
    #[test]
    #[ignore = "requires real CUDA hardware -- run with --ignored to verify fused_add_rmsnorm vs CPU rms_norm"]
    fn launch_fused_add_rmsnorm_matches_cpu_reference() {
        let n = 128;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.031).sin()).collect();
        let residual: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).cos()).collect();
        let weight: Vec<f32> = (0..n).map(|i| 1.0 + ((i as f32) * 0.003).sin()).collect();
        let eps = 1e-5f32;

        let sum: Vec<f32> = x.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
        let mean_sq = sum.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        let expected: Vec<f32> = sum
            .iter()
            .zip(weight.iter())
            .map(|(v, w)| v * scale * w)
            .collect();

        let gpu = launch_fused_add_rmsnorm(&x, &residual, &weight, eps)
            .expect("fused_add_rmsnorm must launch on CUDA hardware");

        assert_eq!(gpu.len(), expected.len());
        for (i, (got, want)) in gpu.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }
}
