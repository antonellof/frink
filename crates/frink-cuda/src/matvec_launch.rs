//! The decode matvec launch path, moved out of `gpu.rs` (ADR 0008):
//! which kernel a matvec runs and with what geometry
//! ([`matvec_launch_plan`]), the per-launch enqueue primitives every
//! caller shares, the routed-expert form, and the multi-matvec and
//! fused-FFN launches built on them. `gpu.rs` keeps the device, the
//! module cache and the per-kind launchers, and re-exports these names
//! so every existing path still resolves.

use crate::dp4a_launch::dp4a_route;
pub(crate) use crate::dp4a_launch::MatvecInput;
use crate::gpu::{
    ensure_module_loaded_lazy, glu_mul_device, resident_cuda_weights, shared_device, CudaError,
    GluCuda, ResidentCudaWeights,
};
use crate::matvec_kinds::coalesced::{coalesced_kind, CoalescedKind, MatvecWidth};

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
/// matrices every token. Activations upload per call HERE: this is the
/// one-matrix launch a lone projection takes (the lm_head, a layer the
/// resident stack declines). Chaining only adjacent matmuls was tried
/// (PR #136) and measured 22% SLOWER on a GTX 1080, because norms,
/// RoPE and attention sat on the host between them; a dense decode
/// layer now runs whole on the device instead (`crate::prefill`, at one
/// row). See `docs/plans/cpu-cuda-parity.md` step 2.
#[allow(clippy::too_many_arguments)] // shared internal launch plumbing; each parameter is a distinct, clearly-named buffer/shape value, not something worth bundling into a struct for one private callee.
pub(crate) fn launch_matvec(
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
    let d_x = crate::xfer::htod(&dev, "matvec x upload", x)?;
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
    crate::xfer::dtoh(&dev, "matvec out download", &d_out)
}

/// How many warps a coalesced matvec block carries; one holder, in
/// `matvec_kinds::coalesced`, so the host check launches the geometry
/// this file does.
pub(crate) const COALESCED_WARPS_PER_BLOCK: usize = crate::matvec_kinds::coalesced::WARPS_PER_BLOCK;

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
pub(crate) fn matvec_launch_plan(
    fn_name: &'static str,
    kernel_src: &'static str,
    module_name: &'static str,
    rows: usize,
    width: MatvecWidth,
    wpr: usize,
) -> MatvecPlan {
    match coalesced_kind(fn_name) {
        Some(kind) => MatvecPlan {
            source: PlanSource::Coalesced(kind, width, wpr),
            module: kind.module_for(width, wpr),
            entry: kind.entry,
            // The same grid at both widths: `wpr` warps per row. Width
            // changes what a warp does inside its row, not which row
            // it owns, so the A/B varies exactly one thing.
            cfg: cudarc::driver::LaunchConfig {
                grid_dim: (rows.div_ceil(COALESCED_WARPS_PER_BLOCK / wpr) as u32, 1, 1),
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
pub(crate) struct MatvecPlan {
    source: PlanSource,
    pub(crate) module: &'static str,
    pub(crate) entry: &'static str,
    pub(crate) cfg: cudarc::driver::LaunchConfig,
}

/// A block-per-row kernel is a literal; a coalesced one is generated
/// per width ([`CoalescedKind::source`]), and only on a module-cache
/// miss.
pub(crate) enum PlanSource {
    Static(&'static str),
    Coalesced(&'static CoalescedKind, MatvecWidth, usize),
}

impl MatvecPlan {
    pub(crate) fn text(&self) -> String {
        match self.source {
            PlanSource::Static(src) => src.to_string(),
            PlanSource::Coalesced(kind, width, wpr) => kind.source(width, wpr),
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
///
/// A kind with a dp4a kernel (the default) quantizes
/// `d_x` here, for this launch alone; a caller with several matvecs on
/// one vector calls [`enqueue_matvec_in`] with one [`MatvecInput`].
pub(crate) fn enqueue_matvec(
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
    enqueue_matvec_in(dev, launch, &MatvecInput::new(d_x))
}

/// [`enqueue_matvec`] on an input other matvecs may share: the dp4a
/// kernel when [`dp4a_route`] says so (the input's q8_1 copy made by
/// the first matvec that needs it), the f32 kernels otherwise.
pub(crate) fn enqueue_matvec_in(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    launch: &MatvecLaunch<'_>,
    input: &MatvecInput<'_>,
) -> Result<
    (
        cudarc::driver::CudaSlice<f32>,
        std::sync::Arc<ResidentCudaWeights>,
    ),
    CudaError,
> {
    if let Some(kind) = dp4a_route(launch.fn_name, launch.n_blocks_per_row, input.len(), 0, 1) {
        return crate::dp4a_launch::enqueue_dp4a(dev, kind, launch, input);
    }
    let d_x = input.x();
    // `FRINK_CUDA_MATVEC=wide|narrow` is the A/B lever (docs/CONFIG.md).
    let width = MatvecWidth::from_env();
    let wpr = crate::matvec_kinds::coalesced::matvec_wpr_override().unwrap_or_else(|| {
        coalesced_kind(launch.fn_name).map_or(1, |k| {
            crate::matvec_kinds::coalesced::wpr_for(
                launch.rows,
                launch.n_blocks_per_row,
                32 / k.lanes(width),
            )
        })
    });
    enqueue_matvec_at(dev, launch, d_x, width, wpr)
}

/// Seconds per launch of the matvec the plan picks for `launch`, over
/// `iters` launches on an activation already on the device, after one
/// warm-up launch that also compiles the kernel. For the
/// `matvec_bench` example; nothing on a serving path calls it.
pub fn time_matvec(launch: &MatvecLaunch<'_>, x: &[f32], iters: usize) -> Result<f64, CudaError> {
    let dev = shared_device()?;
    let d_x = crate::xfer::htod(&dev, "matvec bench activation", x)?;
    drop(enqueue_matvec(&dev, launch, &d_x)?);
    crate::xfer::sync(&dev, "matvec bench warm-up")?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters.max(1) {
        drop(enqueue_matvec(&dev, launch, &d_x)?);
    }
    crate::xfer::sync(&dev, "matvec bench")?;
    Ok(t0.elapsed().as_secs_f64() / iters.max(1) as f64)
}

/// The same projection of several routed experts in ONE launch: one
/// expert per `blockIdx.y`, each reading its weights through a table of
/// device pointers, its input `x_stride` floats along `d_x` (`0` when
/// every expert reads the same input), and writing its own `rows` of the
/// output, `launches.len() * rows` in all. `Unsupported` when the
/// experts do not share one coalesced kind and shape, and the caller
/// launches them one at a time instead. Every dp4a kind is also a
/// coalesced one (`every_dp4a_kind_has_a_coalesced_kernel`), so the
/// gate is the same whichever kernels the switch picks.
pub(crate) fn enqueue_matvec_multi(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    launches: &[MatvecLaunch<'_>],
    input: &MatvecInput<'_>,
    x_stride: usize,
) -> Result<
    (
        cudarc::driver::CudaSlice<f32>,
        Vec<std::sync::Arc<ResidentCudaWeights>>,
    ),
    CudaError,
> {
    use cudarc::driver::DevicePtr;
    let Some(first) = launches.first() else {
        return Err(CudaError::Unsupported(
            "matvec multi: no experts".to_string(),
        ));
    };
    if coalesced_kind(first.fn_name).is_none() {
        return Err(CudaError::Unsupported(format!(
            "matvec multi: {} is not coalesced",
            first.fn_name
        )));
    }
    if launches.iter().any(|l| {
        l.fn_name != first.fn_name
            || l.rows != first.rows
            || l.row_bytes != first.row_bytes
            || l.n_blocks_per_row != first.n_blocks_per_row
            || l.weights.len() != first.rows * first.row_bytes
    }) {
        return Err(CudaError::Unsupported(
            "matvec multi: the experts differ in kind or shape".to_string(),
        ));
    }
    let k = launches.len();
    let held: Vec<_> = launches
        .iter()
        .map(|l| resident_cuda_weights(dev, l.weights))
        .collect::<Result<_, _>>()?;
    let table: Vec<u64> = held.iter().map(|w| *w.slice.device_ptr()).collect();
    let d_table = crate::xfer::htod(dev, "matvec multi table", &table)?;
    let shape = MultiShape::of(first);
    // SAFETY: the table holds `k` live device pointers to `rows *
    // row_bytes` bytes of one coalesced kind each (checked above), held
    // by the caller until its wait.
    let d_out =
        unsafe { enqueue_matvec_multi_table(dev, &shape, &d_table.slice(..), k, input, x_stride) }?;
    Ok((d_out, held))
}

/// The shape every expert of a multi launch shares.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MultiShape {
    pub fn_name: &'static str,
    pub rows: usize,
    pub row_bytes: usize,
    pub n_blocks_per_row: usize,
}

impl MultiShape {
    pub(crate) fn of(l: &MatvecLaunch<'_>) -> Self {
        Self {
            fn_name: l.fn_name,
            rows: l.rows,
            row_bytes: l.row_bytes,
            n_blocks_per_row: l.n_blocks_per_row,
        }
    }
}

/// The launch half of [`enqueue_matvec_multi`], with the weights'
/// device pointers already in a device table: `table[j]` is slot `j`'s
/// weights. The routed decode gathers that table on the device from the
/// chosen experts' ids, so no host round trip picks them.
///
/// # Safety
/// `table` holds `k` device pointers, each to `shape.rows *
/// shape.row_bytes` live bytes of `shape.fn_name`'s kind, alive until
/// the stream has run the launch; `input` covers `(k - 1) * x_stride +
/// shape.n_blocks_per_row` blocks of activations.
pub(crate) unsafe fn enqueue_matvec_multi_table(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    shape: &MultiShape,
    table: &cudarc::driver::CudaView<'_, u64>,
    k: usize,
    input: &MatvecInput<'_>,
    x_stride: usize,
) -> Result<cudarc::driver::CudaSlice<f32>, CudaError> {
    use cudarc::driver::{DeviceSlice, LaunchAsync};
    if let Some(dk) = dp4a_route(
        shape.fn_name,
        shape.n_blocks_per_row,
        input.len(),
        x_stride,
        k,
    ) {
        // SAFETY: forwarded: the caller's contract is the dp4a launch's.
        return unsafe {
            crate::dp4a_launch::enqueue_dp4a_multi_table(dev, dk, shape, table, k, input, x_stride)
        };
    }
    let d_x = input.x();
    let kind = coalesced_kind(shape.fn_name).ok_or_else(|| {
        CudaError::Unsupported(format!("matvec multi: {} is not coalesced", shape.fn_name))
    })?;
    if k == 0 || table.len() < k {
        return Err(CudaError::Unsupported(
            "matvec multi: the table is shorter than its slots".to_string(),
        ));
    }
    let width = MatvecWidth::from_env();
    let wpr = crate::matvec_kinds::coalesced::matvec_wpr_override().unwrap_or_else(|| {
        crate::matvec_kinds::coalesced::wpr_for(
            shape.rows,
            shape.n_blocks_per_row,
            32 / kind.lanes(width),
        )
    });
    let module = kind.multi_module_for(width, wpr);
    ensure_module_loaded_lazy(dev, module, kind.entry, || kind.multi_source(width, wpr))?;
    let func = dev.get_func(module, kind.entry).ok_or_else(|| {
        CudaError::KernelCompile(format!(
            "function '{}' not found after load_ptx",
            kind.entry
        ))
    })?;
    // SAFETY: as for `enqueue_matvec_at`, per expert: each `blockIdx.y`
    // slot writes every one of its `rows` outputs.
    let mut d_out = unsafe { alloc_out(dev, k * shape.rows, "matvec multi output") }?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            shape.rows.div_ceil(COALESCED_WARPS_PER_BLOCK / wpr) as u32,
            k as u32,
            1,
        ),
        block_dim: ((COALESCED_WARPS_PER_BLOCK * 32) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: (const void*, const float*, float*, int x4) matched
    // positionally by the kernel's FX_MULTI parameter list; the table's
    // first `k` pointers are live weights of this kind and shape and `d_x`
    // covers every slot's activations (the caller's contract), and
    // `d_out` is `k * rows`.
    unsafe {
        func.launch(
            cfg,
            (
                table,
                d_x,
                &mut d_out,
                shape.rows as i32,
                shape.row_bytes as i32,
                shape.n_blocks_per_row as i32,
                x_stride as i32,
            ),
        )
        .map_err(|e| CudaError::Launch(format!("kernel {} multi: {e:?}", kind.entry)))?;
    }
    Ok(d_out)
}

/// [`enqueue_matvec`] at an explicit width and warps per row, so the
/// hardware test can hold every geometry against the twins whatever
/// the environment says.
pub(crate) fn enqueue_matvec_at(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    launch: &MatvecLaunch<'_>,
    d_x: &cudarc::driver::CudaSlice<f32>,
    width: MatvecWidth,
    wpr: usize,
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
        wpr,
    );
    let (module, entry, cfg) = (plan.module, plan.entry, plan.cfg);

    ensure_module_loaded_lazy(dev, module, entry, || plan.text())?;
    let func = dev.get_func(module, entry).ok_or_else(|| {
        CudaError::KernelCompile(format!("function '{entry}' not found after load_ptx"))
    })?;

    let d_weights = resident_cuda_weights(dev, launch.weights)?;
    // SAFETY: every matvec kernel gives each output row one warp (or
    // one block), returns early only for a row past `rows`, and writes
    // `out[row]` for every row below it; the launch plan's grid covers
    // `rows`.
    let mut d_out = unsafe { alloc_out(dev, launch.rows, "matvec output") }?;

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

/// A device buffer for one kernel's output, NOT zeroed.
///
/// `alloc_zeros` enqueues a memset on the stream before the kernel
/// that then overwrites every element, and the resident stack made
/// about fourteen of them per layer per decode token: serialized GPU
/// work and a driver call each, for nothing a kernel reads.
///
/// # Safety
///
/// The memory is uninitialised. The caller passes it to a launch that
/// writes every one of its `n` elements before anything reads one, and
/// says why in its own `SAFETY` comment.
pub(crate) unsafe fn alloc_out(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    n: usize,
    what: &str,
) -> Result<cudarc::driver::CudaSlice<f32>, CudaError> {
    // SAFETY: forwarded to the caller, per the contract above.
    unsafe { dev.alloc::<f32>(n) }.map_err(|e| CudaError::Launch(format!("{what} alloc: {e:?}")))
}

/// Metadata for a single matvec launch in a multi-matvec batch (shared
/// activation upload). Each describes one weight matrix to dispatch
/// against the same `x`.
///
/// `weights` must be a stable host slice (mmap / owned matrix storage) —
/// never a temporary `to_vec()` copy. [`resident_cuda_weights`] keys by
/// pointer+length; a per-call clone would miss the cache and re-upload
/// multi-GB matrices every QKV fuse (the Vast ~2 tok/s failure mode).
#[derive(Clone, Copy)]
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

    let d_x = crate::xfer::htod(&dev, "x upload", x)?;

    // Hold weight Arcs + device outs until after all launches so kernels
    // can overlap before any host sync (llama: sync only at graph edge).
    let mut weight_arcs = Vec::with_capacity(launches.len());
    let mut d_outs = Vec::with_capacity(launches.len());

    // One input for every launch: Q, K and V (or gate and up) read the
    // same vector, so under dp4a it is quantized once for all of them.
    let input = MatvecInput::new(&d_x);
    for launch in launches {
        let (d_out, d_weights) = enqueue_matvec_in(&dev, launch, &input)?;
        weight_arcs.push(d_weights);
        d_outs.push(d_out);
    }
    drop(weight_arcs);

    let mut results = Vec::with_capacity(d_outs.len());
    for (i, d_out) in d_outs.into_iter().enumerate() {
        let out = crate::xfer::dtoh(&dev, launches[i].fn_name, &d_out)?;
        results.push(out);
    }

    Ok(results)
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

    let d_x = crate::xfer::htod(&dev, "ffn x upload", x)?;

    // Bind the resident-weight Arcs (`_wg`/`_wu`/`_wd`) for the whole
    // function so every kernel's weight buffer outlives the final sync.
    let input = MatvecInput::new(&d_x);
    let (d_gate, _wg) = enqueue_matvec_in(&dev, gate, &input)?;
    let (d_up, _wu) = enqueue_matvec_in(&dev, up, &input)?;
    let d_act = glu_mul_device(&dev, GluCuda::Silu, &d_gate, &d_up, gate.rows)?;
    let (d_out, _wd) = enqueue_matvec(&dev, down, &d_act)?;

    crate::xfer::dtoh(&dev, "ffn out download", &d_out)
}
