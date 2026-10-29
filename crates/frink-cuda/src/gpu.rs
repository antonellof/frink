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
/// device present," and this function has to catch it explicitly with
/// `catch_unwind` or a CPU-only host would crash the whole process
/// just from calling this probe. This is exactly the kind of bug that
/// only surfaces by actually running code against a real (lack of)
/// environment, not by reading API docs.
pub fn probe() -> Option<CudaInfo> {
    // Temporarily silence the default panic hook: on a host with no
    // CUDA shared library at all, the catch_unwind below is expected
    // to trigger on every single call (e.g. every process start that
    // checks capabilities), and printing a full Rust panic backtrace
    // for an anticipated, handled condition would be noise, not a bug
    // report. The previous hook is always restored, even if the
    // probed closure panics.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| {
        let dev = cudarc::driver::CudaDevice::new(0).ok()?;
        let name = dev.name().ok();
        let (free, total) = cudarc::driver::result::mem_get_info().ok()?;
        Some(CudaInfo {
            device_count: 1,
            first_device_name: name,
            total_vram_bytes: total as u64,
            free_vram_bytes: free as u64,
        })
    });
    std::panic::set_hook(previous_hook);
    result.unwrap_or(None)
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

/// Which `module_name`s have already been NVRTC-compiled and
/// `load_ptx`'d onto the shared device -- `CudaDevice::load_ptx` always
/// recompiles and overwrites on every call (verified directly against
/// `cudarc` 0.11.9's own source: `modules.insert(module_name.into(),
/// module)` with no existing-key check), so skipping a repeat call for
/// an already-loaded kernel has to be tracked here, not assumed.
static LOADED_MODULES: std::sync::Mutex<Option<std::collections::HashSet<&'static str>>> =
    std::sync::Mutex::new(None);

pub(crate) fn ensure_module_loaded(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    kernel_src: &str,
    module_name: &'static str,
    fn_name: &'static str,
) -> Result<(), CudaError> {
    ensure_module_loaded_lazy(dev, module_name, fn_name, || kernel_src.to_string())
}

/// The same load-once cache, for a kernel whose source is *generated*
/// rather than a `&'static str` literal (`mul_mm`'s per-quant-kind
/// bodies). `src` is called only on a cache miss, so a per-token launch
/// does not re-format a translation unit it will not compile.
pub(crate) fn ensure_module_loaded_lazy(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    module_name: &'static str,
    fn_name: &'static str,
    src: impl FnOnce() -> String,
) -> Result<(), CudaError> {
    ensure_module_loaded_lazy_for_arch(dev, module_name, fn_name, None, src)
}

/// [`ensure_module_loaded_lazy`] with an explicit NVRTC target, for a
/// kernel whose instructions need one (`mma.sync.m16n8k16` is
/// `compute_80`); `None` is NVRTC's default target, which every other
/// kernel here compiles for.
pub(crate) fn ensure_module_loaded_lazy_for_arch(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    module_name: &'static str,
    fn_name: &'static str,
    arch: Option<&'static str>,
    src: impl FnOnce() -> String,
) -> Result<(), CudaError> {
    let mut guard = LOADED_MODULES.lock().unwrap();
    let set = guard.get_or_insert_with(std::collections::HashSet::new);
    if set.contains(module_name) {
        return Ok(());
    }
    let opts = cudarc::nvrtc::CompileOptions {
        arch,
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(src(), opts)
        .map_err(|e| CudaError::KernelCompile(format!("{e:?}")))?;
    dev.load_ptx(ptx, module_name, &[fn_name])
        .map_err(|e| CudaError::KernelCompile(format!("{e:?}")))?;
    set.insert(module_name);
    Ok(())
}

/// Compiles (once) and launches a matvec kernel from `kernel_src`/`fn_name`
/// (either `Q8_0_MATVEC_KERNEL_SRC`/`"q8_0_matvec"` or
/// `Q4_0_MATVEC_KERNEL_SRC`/`"q4_0_matvec"` -- both share the same
/// launch config and buffer shapes, only the per-block unpack differs),
/// using cudarc 0.11.9's actual API (`CudaDevice::load_ptx` /
/// `get_func` / `htod_copy` / `alloc_zeros` / `dtoh_sync_copy` / the
/// `LaunchAsync` trait). Verified on real GPU hardware -- see the
/// module doc comment. The CUDA context and compiled kernel are now
/// process-wide and persistent (see `shared_device`/`ensure_module_loaded`) --
/// quantized weight buffers are also cached by host pointer+length
/// (`resident_cuda_weights`) so decode does not re-upload multi-GB
/// matrices every token. Activations still upload per call, and that is
/// deliberate: keeping them device-resident between matmuls was tried
/// (PR #136) and measured 22% SLOWER on a GTX 1080, because decode is
/// kernel-bound at ~90% utilization rather than host-bound. See
/// `docs/plans/cpu-cuda-parity.md` step 2.
#[allow(clippy::too_many_arguments)] // shared internal launch plumbing; each parameter is a distinct, clearly-named buffer/shape value, not something worth bundling into a struct for one private callee.
fn launch_matvec(
    kernel_src: &'static str,
    module_name: &'static str,
    fn_name: &'static str,
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, CudaError> {
    // Which kernel actually runs is `enqueue_matvec`'s decision, so
    // every caller gets the same one -- see `matvec_launch_plan`.
    let dev = shared_device()?;
    let d_x = dev
        .htod_copy(x.to_vec())
        .map_err(|e| CudaError::Launch(format!("{e:?}")))?;
    let launch = MatvecLaunch {
        kernel_src,
        module_name,
        fn_name,
        weights,
        rows,
        row_bytes,
        n_blocks_per_row,
    };
    // `_weights` holds the resident-weight Arc alive until the DtoH sync.
    let (d_out, _weights) = enqueue_matvec(&dev, &launch, &d_x)?;
    dev.dtoh_sync_copy(&d_out)
        .map_err(|e| CudaError::Launch(format!("{e:?}")))
}

/// How many warps a coalesced matvec block carries; one holder, in
/// `matvec_kinds::coalesced`, so the host check launches the geometry
/// this file does.
const COALESCED_WARPS_PER_BLOCK: usize = crate::matvec_kinds::coalesced::WARPS_PER_BLOCK;

/// Which kernel a matvec actually runs, and the geometry that kernel
/// needs.
///
/// The two must agree: a coalesced kernel launched with the
/// block-per-row geometry computes one row per BLOCK while indexing as
/// if it had one row per WARP, and returns wrong numbers rather than
/// failing. That is the repo's dominant bug shape, so the choice is
/// made once, here, by the only function that decides it, and
/// `the_plan_pairs_each_kernel_with_its_own_geometry` checks the
/// pairing on the host.
///
/// This lives in `enqueue_matvec` rather than in `launch_matvec`
/// because the fused dense FFN and `launch_matvec_multi` enqueue
/// directly. Routing at the `launch_matvec` level left the FFN --
/// gate, up and down, about 83% of the weight bytes a Llama-3.2-3B
/// decode step reads -- on the uncoalesced kernels while the
/// benchmarks said "coalesced".
fn matvec_launch_plan(
    fn_name: &'static str,
    kernel_src: &'static str,
    module_name: &'static str,
    rows: usize,
    width: MatvecWidth,
) -> MatvecPlan {
    match coalesced_kind(fn_name) {
        Some(kind) => MatvecPlan {
            source: PlanSource::Coalesced(kind, width),
            module: kind.module(width),
            entry: kind.entry,
            // The same grid at both widths: one row per warp. Width
            // changes what a warp does inside its row, not which row
            // it owns, so the A/B varies exactly one thing.
            cfg: cudarc::driver::LaunchConfig {
                grid_dim: (rows.div_ceil(COALESCED_WARPS_PER_BLOCK) as u32, 1, 1),
                block_dim: ((COALESCED_WARPS_PER_BLOCK * 32) as u32, 1, 1),
                shared_mem_bytes: 0,
            },
        },
        None => MatvecPlan {
            source: PlanSource::Static(kernel_src),
            module: module_name,
            entry: fn_name,
            cfg: cudarc::driver::LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
            },
        },
    }
}

/// What [`matvec_launch_plan`] decided: the NVRTC module and entry
/// point to launch, where its text comes from, and its geometry.
struct MatvecPlan {
    source: PlanSource,
    module: &'static str,
    entry: &'static str,
    cfg: cudarc::driver::LaunchConfig,
}

/// A block-per-row kernel is a literal; a coalesced one is generated
/// per width ([`CoalescedKind::source`]), and only on a module-cache
/// miss.
enum PlanSource {
    Static(&'static str),
    Coalesced(&'static CoalescedKind, MatvecWidth),
}

impl MatvecPlan {
    fn text(&self) -> String {
        match self.source {
            PlanSource::Static(src) => src.to_string(),
            PlanSource::Coalesced(kind, width) => kind.source(width),
        }
    }
}

/// Enqueues one matvec kernel on the shared device's default stream:
/// ensures the module is compiled/loaded, fetches the function, pins
/// the (process-resident) weight buffer, allocates the output, and
/// launches — **without** any host synchronization. `d_x` is an
/// already-on-device activation slice (uploaded once by the caller and
/// reused across chained matvecs, e.g. the fused dense FFN), so no
/// per-call HtoD of the activation happens here. Returns the device
/// output slice plus the resident-weight `Arc`, which the caller must
/// keep alive until it syncs (the kernel reads that buffer
/// asynchronously). This is the single per-launch primitive shared by
/// [`launch_matvec`], [`launch_matvec_multi`] and
/// [`launch_dense_ffn_swiglu`].
fn enqueue_matvec(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    launch: &MatvecLaunch<'_>,
    d_x: &cudarc::driver::CudaSlice<f32>,
) -> Result<
    (
        cudarc::driver::CudaSlice<f32>,
        std::sync::Arc<ResidentCudaWeights>,
    ),
    CudaError,
> {
    // `FRINK_CUDA_MATVEC=narrow` is the A/B lever (docs/CONFIG.md).
    enqueue_matvec_at(dev, launch, d_x, MatvecWidth::from_env())
}

/// [`enqueue_matvec`] at an explicit width, so the hardware test can
/// hold both widths against their twins whatever the environment says.
fn enqueue_matvec_at(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    launch: &MatvecLaunch<'_>,
    d_x: &cudarc::driver::CudaSlice<f32>,
    width: MatvecWidth,
) -> Result<
    (
        cudarc::driver::CudaSlice<f32>,
        std::sync::Arc<ResidentCudaWeights>,
    ),
    CudaError,
> {
    use cudarc::driver::LaunchAsync;

    let plan = matvec_launch_plan(
        launch.fn_name,
        launch.kernel_src,
        launch.module_name,
        launch.rows,
        width,
    );
    let (module, entry, cfg) = (plan.module, plan.entry, plan.cfg);

    ensure_module_loaded_lazy(dev, module, entry, || plan.text())?;
    let func = dev.get_func(module, entry).ok_or_else(|| {
        CudaError::KernelCompile(format!("function '{entry}' not found after load_ptx"))
    })?;

    let d_weights = resident_cuda_weights(dev, launch.weights)?;
    let mut d_out = dev
        .alloc_zeros::<f32>(launch.rows)
        .map_err(|e| CudaError::Launch(format!("output alloc: {e:?}")))?;

    unsafe {
        func.launch(
            cfg,
            (
                &d_weights.slice,
                d_x,
                &mut d_out,
                launch.rows as i32,
                launch.row_bytes as i32,
                launch.n_blocks_per_row as i32,
            ),
        )
        .map_err(|e| CudaError::Launch(format!("kernel {entry}: {e:?}")))?;
    }

    Ok((d_out, d_weights))
}

pub(crate) struct ResidentCudaWeights {
    pub(crate) slice: cudarc::driver::CudaSlice<u8>,
    #[allow(dead_code)] // Kept for diagnostics / future eviction logic.
    nbytes: usize,
    /// What the bytes at the key looked like when they were uploaded.
    /// A host buffer can be freed and its address handed to a different
    /// buffer of the same length, and `(pointer, len)` cannot tell the
    /// two apart: the tensor-core `mul_mm` hardware test multiplied one
    /// fixture's weights by another fixture's activations for exactly
    /// that reason (2026-09-17), and a merged LoRA or any other
    /// short-lived matrix could do the same in a server. Production
    /// weights are one mmap for the life of the process, which is why
    /// nothing had noticed. This is the same identity hazard
    /// `frink-metal/src/resident_act.rs` (#166) closed for
    /// activations, closed the cheap way for weights: a sample of the
    /// bytes, compared on every hit.
    fingerprint: WeightFingerprint,
}

/// A cheap sample of a weight buffer: its first and last 64 bytes and
/// 64 from the middle. Not a hash of the whole tensor (that is
/// gigabytes per token); enough that a different buffer at the same
/// address is caught unless it agrees on 192 sampled bytes.
#[derive(PartialEq, Eq, Clone, Debug)]
struct WeightFingerprint {
    head: [u8; 64],
    mid: [u8; 64],
    tail: [u8; 64],
}

impl WeightFingerprint {
    fn of(bytes: &[u8]) -> Self {
        fn take(bytes: &[u8], at: usize) -> [u8; 64] {
            let mut out = [0u8; 64];
            let at = at.min(bytes.len());
            let n = (bytes.len() - at).min(64);
            out[..n].copy_from_slice(&bytes[at..at + n]);
            out
        }
        Self {
            head: take(bytes, 0),
            mid: take(bytes, bytes.len() / 2),
            tail: take(bytes, bytes.len().saturating_sub(64)),
        }
    }
}

// SAFETY: slices live on the process-wide shared CudaDevice and are
// only read by kernels; same sharing model as Metal's ResidentWeightBuffer.
unsafe impl Send for ResidentCudaWeights {}
unsafe impl Sync for ResidentCudaWeights {}

type CudaWeightCacheKey = (usize, usize);
type CudaWeightCacheMap =
    std::collections::HashMap<CudaWeightCacheKey, std::sync::Arc<ResidentCudaWeights>>;

static CUDA_WEIGHT_CACHE: std::sync::Mutex<Option<CudaWeightCacheMap>> =
    std::sync::Mutex::new(None);

pub(crate) fn resident_cuda_weights(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    weights: &[u8],
) -> Result<std::sync::Arc<ResidentCudaWeights>, CudaError> {
    let key = (weights.as_ptr() as usize, weights.len());
    let fingerprint = WeightFingerprint::of(weights);
    let mut guard = CUDA_WEIGHT_CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(std::collections::HashMap::new);
    if let Some(cached) = cache.get(&key) {
        if cached.fingerprint == fingerprint {
            return Ok(cached.clone());
        }
        // Same address and length, different bytes: the buffer this
        // entry was uploaded from is gone. Re-upload rather than serve
        // another tensor's weights.
        cache.remove(&key);
    }
    let slice = dev
        .htod_copy(weights.to_vec())
        .map_err(|e| CudaError::Launch(format!("{e:?}")))?;
    let cached = std::sync::Arc::new(ResidentCudaWeights {
        slice,
        nbytes: weights.len(),
        fingerprint,
    });
    cache.insert(key, cached.clone());
    Ok(cached)
}

/// A model-owned f32 vector resident on the device: norm weights, QKV
/// biases, RoPE divisors, which the resident prefill stack used to
/// upload again for every layer of every step (three to seven pageable
/// `htod_copy`s per layer, each a `cudaMalloc` and a synchronous copy
/// between the kernels that should have been back to back). Keyed and
/// checked exactly as [`resident_cuda_weights`]: the same
/// `(pointer, len)` key, the same byte fingerprint on every hit, for
/// the same reason.
pub(crate) struct ResidentF32 {
    pub(crate) slice: cudarc::driver::CudaSlice<f32>,
    fingerprint: WeightFingerprint,
}

type F32CacheMap = std::collections::HashMap<CudaWeightCacheKey, std::sync::Arc<ResidentF32>>;

static CUDA_F32_CACHE: std::sync::Mutex<Option<F32CacheMap>> = std::sync::Mutex::new(None);

pub(crate) fn resident_f32(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    values: &[f32],
) -> Result<std::sync::Arc<ResidentF32>, CudaError> {
    // SAFETY: an f32 slice viewed as its bytes, for the fingerprint
    // only; same length in bytes, same lifetime, never written.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    let key = (values.as_ptr() as usize, values.len());
    let fingerprint = WeightFingerprint::of(bytes);
    let mut guard = CUDA_F32_CACHE.lock().unwrap_or_else(|p| p.into_inner());
    let cache = guard.get_or_insert_with(std::collections::HashMap::new);
    if let Some(cached) = cache.get(&key) {
        if cached.fingerprint == fingerprint {
            return Ok(cached.clone());
        }
        cache.remove(&key);
    }
    let slice = dev
        .htod_copy(values.to_vec())
        .map_err(|e| CudaError::Launch(format!("{e:?}")))?;
    let cached = std::sync::Arc::new(ResidentF32 { slice, fingerprint });
    cache.insert(key, cached.clone());
    Ok(cached)
}

// The kernel TEXT moved to `matvec_kinds`, one module per format
// family; `gpu.rs` keeps the device plumbing. Re-exported so every
// existing path still resolves -- a refactor that moves a definition
// should not move a public name.
use crate::matvec_kinds::coalesced::{coalesced_kind, CoalescedKind, MatvecWidth};
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
    crate::matvec_kinds::kind_by_name(kind_name).map(|k| (k.src, k.module_name, k.fn_name))
}

/// Looks a kind up in [`matvec_launch_meta`] and launches it. Every
/// `launch_q*_matvec` below is this call with its name filled in, so
/// the launchers cannot name a module or entry point the table does
/// not.
fn launch_matvec_by_kind(
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

/// Metadata for a single matvec launch in a multi-matvec batch (shared
/// activation upload). Each describes one weight matrix to dispatch
/// against the same `x`.
///
/// `weights` must be a stable host slice (mmap / owned matrix storage) —
/// never a temporary `to_vec()` copy. [`resident_cuda_weights`] keys by
/// pointer+length; a per-call clone would miss the cache and re-upload
/// multi-GB matrices every QKV fuse (the Vast ~2 tok/s failure mode).
pub struct MatvecLaunch<'a> {
    pub kernel_src: &'static str,
    pub module_name: &'static str,
    pub fn_name: &'static str,
    pub weights: &'a [u8],
    pub rows: usize,
    pub row_bytes: usize,
    pub n_blocks_per_row: usize,
}

/// Uploads `x` once, enqueues N matvec kernels (resident weights), then
/// downloads N outputs. Mirrors ggml-cuda's "enqueue on stream, sync at
/// the edge" discipline as far as cudarc 0.11.9 allows without a full
/// device-resident graph: one HtoD for `x`, N kernel launches, then N
/// DtoH. Used by `WeightMatrix::apply_gpu_multi` for Q/K/V and gate+up.
pub fn launch_matvec_multi(
    x: &[f32],
    launches: &[MatvecLaunch<'_>],
) -> Result<Vec<Vec<f32>>, CudaError> {
    if launches.is_empty() {
        return Ok(Vec::new());
    }

    let dev = shared_device()?;

    let d_x = dev
        .htod_copy(x.to_vec())
        .map_err(|e| CudaError::Launch(format!("x upload: {e:?}")))?;

    // Hold weight Arcs + device outs until after all launches so kernels
    // can overlap before any host sync (llama: sync only at graph edge).
    let mut weight_arcs = Vec::with_capacity(launches.len());
    let mut d_outs = Vec::with_capacity(launches.len());

    for launch in launches {
        let (d_out, d_weights) = enqueue_matvec(&dev, launch, &d_x)?;
        weight_arcs.push(d_weights);
        d_outs.push(d_out);
    }
    drop(weight_arcs);

    let mut results = Vec::with_capacity(d_outs.len());
    for (i, d_out) in d_outs.into_iter().enumerate() {
        let out = dev.dtoh_sync_copy(&d_out).map_err(|e| {
            CudaError::Launch(format!("output download {}: {e:?}", launches[i].fn_name))
        })?;
        results.push(out);
    }

    Ok(results)
}

/// CUDA C source for an elementwise SwiGLU pair fuse:
/// `out[i] = silu(gate[i]) * up[i]`, `silu(x) = x / (1 + exp(-x))`.
/// Mirrors `frink_core::matmul::swiglu` (which is `silu(gate)*up`) and
/// `frink-metal/src/elem.rs`'s `silu_mul_f32` kernel exactly, so the
/// fused dense-FFN path keeps activations on-device between the
/// gate/up matvecs and the down matvec instead of downloading two
/// vectors, computing SwiGLU on the host, and re-uploading.
pub const SILU_MUL_KERNEL_SRC: &str = r#"
extern "C" __global__ void silu_mul_f32(
    const float* gate,
    const float* up,
    float* out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}
"#;

/// Enqueues the elementwise `silu(gate)*up` kernel on the shared
/// device's default stream (no host sync), returning the device output
/// slice. `gate`/`up` are device slices of length `n` produced by the
/// two FFN input matvecs; the result feeds straight into the down
/// matvec without ever touching host memory.
pub(crate) fn silu_mul_device(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    gate: &cudarc::driver::CudaSlice<f32>,
    up: &cudarc::driver::CudaSlice<f32>,
    n: usize,
) -> Result<cudarc::driver::CudaSlice<f32>, CudaError> {
    use cudarc::driver::LaunchAsync;

    ensure_module_loaded(dev, SILU_MUL_KERNEL_SRC, "frink_silu_mul", "silu_mul_f32")?;
    let func = dev
        .get_func("frink_silu_mul", "silu_mul_f32")
        .ok_or_else(|| {
            CudaError::KernelCompile("function 'silu_mul_f32' not found after load_ptx".to_string())
        })?;
    let mut d_out = dev
        .alloc_zeros::<f32>(n)
        .map_err(|e| CudaError::Launch(format!("silu_mul out alloc: {e:?}")))?;
    let block = 256u32;
    let grid = (n as u32).div_ceil(block);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(cfg, (gate, up, &mut d_out, n as i32))
            .map_err(|e| CudaError::Launch(format!("silu_mul launch: {e:?}")))?;
    }
    Ok(d_out)
}

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

    let d_x = dev
        .htod_copy(x.to_vec())
        .map_err(|e| CudaError::Launch(format!("fused_add_rmsnorm x upload: {e:?}")))?;
    let d_residual = dev
        .htod_copy(residual.to_vec())
        .map_err(|e| CudaError::Launch(format!("fused_add_rmsnorm residual upload: {e:?}")))?;
    let d_weight = dev
        .htod_copy(weight.to_vec())
        .map_err(|e| CudaError::Launch(format!("fused_add_rmsnorm weight upload: {e:?}")))?;
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

    dev.dtoh_sync_copy(&d_out)
        .map_err(|e| CudaError::Launch(format!("fused_add_rmsnorm download: {e:?}")))
}

/// Fused dense SwiGLU FFN entirely on-device: uploads `x` once, runs
/// the gate and up matvecs (device-resident weights), fuses
/// `silu(gate)*up` in a kernel, runs the down matvec, and downloads the
/// result once. This replaces the previous "3× `WeightMatrix::apply`
/// with an HtoD+DtoH per matvec plus a host-side SwiGLU" pattern for
/// dense/shared experts, cutting the activation traffic from six host↔
/// device transfers to one up + one down. Weight buffers are already
/// process-resident (see `resident_cuda_weights`); only the single
/// activation upload and single result download cross the bus.
///
/// `gate`/`up` must have the same output row count (the FFN hidden dim)
/// and the same input `cols` as `x`; `down`'s input `cols` must equal
/// that FFN hidden dim (its `n_blocks_per_row` covers the SwiGLU output
/// length). The caller (`frink-core`) guarantees this by construction
/// from the expert's gate/up/down shapes.
pub fn launch_dense_ffn_swiglu(
    gate: &MatvecLaunch<'_>,
    up: &MatvecLaunch<'_>,
    down: &MatvecLaunch<'_>,
    x: &[f32],
) -> Result<Vec<f32>, CudaError> {
    let dev = shared_device()?;

    let d_x = dev
        .htod_copy(x.to_vec())
        .map_err(|e| CudaError::Launch(format!("ffn x upload: {e:?}")))?;

    // Bind the resident-weight Arcs (`_wg`/`_wu`/`_wd`) for the whole
    // function so every kernel's weight buffer outlives the final sync.
    let (d_gate, _wg) = enqueue_matvec(&dev, gate, &d_x)?;
    let (d_up, _wu) = enqueue_matvec(&dev, up, &d_x)?;
    let d_act = silu_mul_device(&dev, &d_gate, &d_up, gate.rows)?;
    let (d_out, _wd) = enqueue_matvec(&dev, down, &d_act)?;

    dev.dtoh_sync_copy(&d_out)
        .map_err(|e| CudaError::Launch(format!("ffn out download: {e:?}")))
}

#[cfg(test)]
mod fingerprint_tests {
    use super::WeightFingerprint;

    #[test]
    fn a_buffer_that_differs_anywhere_sampled_has_a_different_fingerprint() {
        let base: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let f = WeightFingerprint::of(&base);
        assert_eq!(f, WeightFingerprint::of(&base.clone()));
        for at in [0usize, 63, 500, 531, 936, 999] {
            let mut other = base.clone();
            other[at] ^= 0xff;
            assert_ne!(f, WeightFingerprint::of(&other), "byte {at} is sampled");
        }
        // Short buffers are sampled whole.
        let short = vec![1u8, 2, 3];
        assert_ne!(
            WeightFingerprint::of(&short),
            WeightFingerprint::of(&[1u8, 2, 4])
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    /// reference, not just "the launch didn't error."
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
        let expected: Vec<f32> = all_rows_f32
            .iter()
            .map(|row| frink_quant::dot_q8_0_f32_scalar(&frink_quant::quantize_q8_0(row), &x))
            .collect();
        (weights, x, expected)
    }

    #[test]
    #[ignore = "requires real CUDA hardware -- verified passing on an RTX 3060 (vast.ai); run with --ignored on a CUDA-capable machine to re-verify"]
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
            matvec_launch_meta("Q4_1").is_none(),
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
    fn real_k_quant_test_matrix(
        rows: usize,
        cols: usize,
        block_bytes: usize,
        scalar_dot: impl Fn(&[u8], &[f32]) -> f32,
    ) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let n_blocks_per_row = cols / 256;
        let row_bytes = n_blocks_per_row * block_bytes;
        let mut weights = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            weights.extend(pseudo_bytes(r as u32 + 1, row_bytes));
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.021).sin()).collect();
        let expected: Vec<f32> = (0..rows)
            .map(|r| scalar_dot(&weights[r * row_bytes..(r + 1) * row_bytes], &x))
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
    #[ignore = "requires real CUDA hardware. The WIDE width (the default since #133) has NEVER executed on a GPU; the narrow width passed on an RTX 3060/3070. Run with --ignored and record the result"]
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
            for width in [MatvecWidth::Narrow, MatvecWidth::Wide] {
                for (n_blocks_per_row, seed) in [(3usize, 7u32), (12, 23)] {
                    let row_bytes = n_blocks_per_row * mm.block_bytes;
                    let cols = n_blocks_per_row * mm.block_elems;
                    let weights = pseudo_bytes(seed, rows * row_bytes);
                    let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.021).sin()).collect();
                    // Enter through the ordinary kernel identity, not the
                    // coalesced one, so this exercises the ROUTING too.
                    let launch = super::MatvecLaunch {
                        kernel_src: mv.src,
                        module_name: mv.module_name,
                        fn_name: mv.fn_name,
                        weights: &weights,
                        rows,
                        row_bytes,
                        n_blocks_per_row,
                    };
                    let d_x = dev.htod_copy(x.clone()).expect("upload");
                    let (d_out, _w) = super::enqueue_matvec_at(&dev, &launch, &d_x, width)
                        .unwrap_or_else(|e| panic!("{} {width:?}: {e:?}", mv.name));
                    let got = dev.dtoh_sync_copy(&d_out).expect("download");
                    assert_eq!(got.len(), rows, "{}", mv.name);
                    let lanes = kind.lanes(width);
                    for r in 0..rows {
                        let row = &weights[r * row_bytes..(r + 1) * row_bytes];
                        let want = twin(row, &x, n_blocks_per_row, lanes);
                        assert!(
                            !(want.is_nan() ^ got[r].is_nan()),
                            "{} {width:?} row {r}: GPU={} twin={want}",
                            mv.name,
                            got[r]
                        );
                        assert_close_relative(got[r], want, r);
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
            for width in [MatvecWidth::Narrow, MatvecWidth::Wide] {
                let plan = super::matvec_launch_plan(fn_name, src, module, rows, width);
                let ck = super::coalesced_kind(fn_name).expect("coalesced");
                assert!(
                    plan.entry.ends_with("_coalesced"),
                    "{kind}: plan chose {}, not the coalesced kernel",
                    plan.entry
                );
                assert_eq!(
                    plan.module,
                    ck.module(width),
                    "{kind} {width:?}: module key"
                );
                let text = plan.text();
                assert!(
                    text.contains(&format!("void {}(", plan.entry)),
                    "{kind}: plan's source does not define {}",
                    plan.entry
                );
                assert!(
                    text.starts_with(&format!("#define FX_LANES {}\n", ck.lanes(width))),
                    "{kind} {width:?}: source compiled for the wrong width"
                );
                let cfg = plan.cfg;
                let warps = super::COALESCED_WARPS_PER_BLOCK as u32;
                assert_eq!(cfg.block_dim, (warps * 32, 1, 1), "{kind}: block_dim");
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
            let plan = super::matvec_launch_plan(fn_name, src, module, rows, width);
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
    #[ignore = "requires real CUDA hardware -- verified passing on an RTX 3060 (vast.ai, 2026-07-31) with the relative-tolerance fix; run with --ignored on a CUDA-capable machine to re-verify"]
    fn launch_q4_k_matvec_matches_cpu_reference() {
        let rows = 3;
        let cols = 256; // 1 Q4_K super-block per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            frink_quant::Q4_K_BLOCK_BYTES,
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
    #[ignore = "requires real CUDA hardware -- 2026-07-31 real RTX 3060 run failed on row 1 with GPU=NaN CPU reference=NaN (assert_close_relative didn't treat NaN==NaN as agreement; fixed, but not yet re-verified on hardware); run with --ignored on a CUDA-capable machine to re-verify"]
    fn launch_q6_k_matvec_matches_cpu_reference() {
        let rows = 3;
        let cols = 512; // 2 Q6_K super-blocks per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            frink_quant::Q6_K_BLOCK_BYTES,
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
    #[ignore = "requires real CUDA hardware -- verifies that launch_matvec_multi (shared x upload) matches N sequential single-matvec launches; run with --ignored on a CUDA-capable machine"]
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
            frink_quant::dot_q4_k_f32_scalar,
        );
        let (weights_b, _, expected_b) = real_k_quant_test_matrix(
            rows_b,
            cols,
            frink_quant::Q4_K_BLOCK_BYTES,
            frink_quant::dot_q4_k_f32_scalar,
        );
        let (weights_c, _, expected_c) = real_k_quant_test_matrix(
            rows_c,
            cols,
            frink_quant::Q4_K_BLOCK_BYTES,
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
    #[ignore = "requires real CUDA hardware -- run with --ignored on a CUDA-capable machine to verify the fused dense-FFN activation-residency path"]
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
        let g = cpu_matvec(&gate_w, gate_rb, ffn_dim, &x);
        let u = cpu_matvec(&up_w, up_rb, ffn_dim, &x);
        let act: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
            .collect();
        let expected = cpu_matvec(&down_w, down_rb, hidden_dim, &act);

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
