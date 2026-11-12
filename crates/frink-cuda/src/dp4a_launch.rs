//! The dp4a decode matvecs on the device (the default; #133): the q8_1 quantizer, the one-matrix launch, the routed-expert
//! launch, and [`MatvecInput`], the activation every matvec of a step
//! is handed so that one quantization serves every matvec that reads
//! the same vector.
//!
//! [`crate::matvec_launch`] decides per launch whether a matvec runs
//! here ([`dp4a_route`]) or on the f32 kernels; nothing else does.

use crate::gpu::{
    alloc_out, ensure_module_loaded_lazy_for_arch, resident_cuda_weights, CudaError, MatvecLaunch,
    MultiShape, ResidentCudaWeights,
};
pub(crate) use crate::matvec_kinds::dp4a::route as dp4a_route;
use crate::matvec_kinds::dp4a::{
    nvrtc_arch, quantize_source, Dp4aGeometry, Dp4aKind, Q8_1_BLOCK_BYTES, Q8_1_BLOCK_ELEMS,
    QUANTIZE_ENTRY, QUANTIZE_MODULE, QUANTIZE_THREADS,
};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, CudaView, LaunchAsync, LaunchConfig};
use std::cell::OnceCell;
use std::sync::Arc;

/// One activation vector on the device, as the matvecs of a step read
/// it: the f32 values, and the q8_1 copy the dp4a kernels read, made by
/// the first matvec that needs it and reused by every later one.
///
/// The copy cannot go stale: the input borrows `x` immutably for its
/// whole life, and every write to a device buffer through `cudarc`
/// takes it mutably. A caller that runs several matvecs on one vector
/// (Q, K and V; gate and up; a layer's router and its experts) builds
/// ONE of these and hands it to each; one built per launch quantizes
/// per launch, which is what a lone projection costs anyway.
pub(crate) struct MatvecInput<'x> {
    x: &'x CudaSlice<f32>,
    q8: OnceCell<CudaSlice<u8>>,
}

impl<'x> MatvecInput<'x> {
    pub(crate) fn new(x: &'x CudaSlice<f32>) -> Self {
        Self {
            x,
            q8: OnceCell::new(),
        }
    }

    /// The f32 activation.
    pub(crate) fn x(&self) -> &'x CudaSlice<f32> {
        self.x
    }

    /// Floats in the activation.
    pub(crate) fn len(&self) -> usize {
        cudarc::driver::DeviceSlice::len(self.x)
    }

    /// The q8_1 copy of the whole vector, quantized on the first call.
    pub(crate) fn q8(&self, dev: &Arc<CudaDevice>) -> Result<&CudaSlice<u8>, CudaError> {
        if let Some(y) = self.q8.get() {
            return Ok(y);
        }
        let y = enqueue_quantize_q8_1(dev, self.x)?;
        Ok(self.q8.get_or_init(|| y))
    }

    /// Whether the quantizer has run for this input: for the test that
    /// several matvecs on one input quantize it once.
    #[cfg(test)]
    pub(crate) fn quantized(&self) -> bool {
        self.q8.get().is_some()
    }
}

/// The device's compute capability, asked once.
fn compute_capability(dev: &Arc<CudaDevice>) -> (i32, i32) {
    static CC: std::sync::OnceLock<(i32, i32)> = std::sync::OnceLock::new();
    *CC.get_or_init(|| {
        let minor = dev
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .unwrap_or(0);
        (crate::gpu::compute_capability_major(dev), minor)
    })
}

/// Compiles (once) and fetches a dp4a module's entry point for the
/// device's own target ([`nvrtc_arch`]). An NVRTC older than the device
/// cannot name its target; the module is then compiled for NVRTC's
/// default, which the driver JIT-compiles, and runs the kernels' scalar
/// integer dot: the same integers, slower, said once on stderr.
fn load(
    dev: &Arc<CudaDevice>,
    module: &'static str,
    entry: &'static str,
    src: impl Fn() -> String,
) -> Result<CudaFunction, CudaError> {
    let arch = nvrtc_arch(compute_capability(dev));
    if let Err(e) = ensure_module_loaded_lazy_for_arch(dev, module, entry, arch, &src) {
        let Some(target) = arch else {
            return Err(e);
        };
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            eprintln!(
                "frink-cuda: NVRTC could not compile the dp4a matvecs for {target} ({e}); \
                 they run on NVRTC's default target with a scalar integer dot"
            )
        });
        ensure_module_loaded_lazy_for_arch(dev, module, entry, None, &src)?;
    }
    dev.get_func(module, entry).ok_or_else(|| {
        CudaError::KernelCompile(format!("function '{entry}' not found after load_ptx"))
    })
}

/// Quantizes `x` to q8_1 on the device: `len / 32` blocks of 36 bytes.
/// `Unsupported` unless `x` is a non-empty whole number of blocks.
pub(crate) fn enqueue_quantize_q8_1(
    dev: &Arc<CudaDevice>,
    x: &CudaSlice<f32>,
) -> Result<CudaSlice<u8>, CudaError> {
    let n = cudarc::driver::DeviceSlice::len(x);
    if n == 0 || !n.is_multiple_of(Q8_1_BLOCK_ELEMS) || i32::try_from(n).is_err() {
        return Err(CudaError::Unsupported(format!(
            "q8_1 quantizer: {n} values is not a whole number of 32-value blocks"
        )));
    }
    let func = load(dev, QUANTIZE_MODULE, QUANTIZE_ENTRY, quantize_source)?;
    let bytes = n / Q8_1_BLOCK_ELEMS * Q8_1_BLOCK_BYTES;
    // SAFETY: uninitialised, and fully written by the launch below before
    // anything reads it: thread `i < n` stores quant `i`, and lane 0 of
    // each warp (one warp per block, `n` a multiple of 32) stores the
    // block's four `ds` bytes. Twin: `dp4a_twin::quantize_q8_1`, held to
    // it byte for byte on the host and on a device.
    let mut y = unsafe { dev.alloc::<u8>(bytes) }
        .map_err(|e| CudaError::Launch(format!("q8_1 output alloc: {e:?}")))?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(QUANTIZE_THREADS) as u32, 1, 1),
        block_dim: (QUANTIZE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: (const float*, unsigned char*, int), the kernel's parameter
    // list; `x` holds `n` floats and `y` `n / 32` blocks, and the grid's
    // threads past `n` return before touching either.
    unsafe { func.launch(cfg, (x, &mut y, n as i32)) }
        .map_err(|e| CudaError::Launch(format!("kernel {QUANTIZE_ENTRY}: {e:?}")))?;
    Ok(y)
}

/// Shape checks every dp4a launch makes before it touches the device:
/// the row is whole blocks of `kind`, the rows are the weights, and
/// every slot's q8_1 input is inside `y`.
fn check_shape(
    kind: &Dp4aKind,
    shape: &MultiShape,
    y_bytes: usize,
    y_stride: usize,
    slots: usize,
) -> Result<(), CudaError> {
    let q8_row = shape.n_blocks_per_row * kind.q8_per_block() * Q8_1_BLOCK_BYTES;
    let y_needed = y_stride
        .checked_mul(Q8_1_BLOCK_BYTES)
        .zip(slots.checked_sub(1))
        .and_then(|(stride, s)| s.checked_mul(stride))
        .and_then(|v| v.checked_add(q8_row));
    let fits = i32::try_from(shape.rows).is_ok()
        && i32::try_from(shape.row_bytes).is_ok()
        && i32::try_from(y_stride).is_ok()
        && shape.rows > 0
        && shape.n_blocks_per_row > 0
        && shape.n_blocks_per_row.checked_mul(kind.block_bytes) == Some(shape.row_bytes)
        && y_needed.is_some_and(|n| n <= y_bytes);
    if fits {
        Ok(())
    } else {
        Err(CudaError::Unsupported(format!(
            "{}: {} rows of {} blocks ({} bytes) against {y_bytes} q8_1 bytes",
            kind.entry, shape.rows, shape.n_blocks_per_row, shape.row_bytes
        )))
    }
}

/// One matvec on the dp4a kernel `kind`, at llama.cpp's geometry for
/// the device and the row. The weights are made resident; the caller
/// holds the returned `Arc` until its wait.
pub(crate) fn enqueue_dp4a(
    dev: &Arc<CudaDevice>,
    kind: &'static Dp4aKind,
    launch: &MatvecLaunch<'_>,
    input: &MatvecInput<'_>,
) -> Result<(CudaSlice<f32>, Arc<ResidentCudaWeights>), CudaError> {
    let g = kind.geometry(compute_capability(dev), launch.n_blocks_per_row);
    enqueue_dp4a_at(dev, kind, launch, input.q8(dev)?, g)
}

/// [`enqueue_dp4a`] on a q8_1 input already on the device, at an
/// explicit geometry, so the hardware test can run every geometry.
pub(crate) fn enqueue_dp4a_at(
    dev: &Arc<CudaDevice>,
    kind: &'static Dp4aKind,
    launch: &MatvecLaunch<'_>,
    y: &CudaSlice<u8>,
    g: Dp4aGeometry,
) -> Result<(CudaSlice<f32>, Arc<ResidentCudaWeights>), CudaError> {
    let shape = MultiShape::of(launch);
    check_shape(kind, &shape, cudarc::driver::DeviceSlice::len(y), 0, 1)?;
    if launch.weights.len() != shape.rows * shape.row_bytes {
        return Err(CudaError::Unsupported(format!(
            "{}: {} weight bytes for {} rows of {}",
            kind.entry,
            launch.weights.len(),
            shape.rows,
            shape.row_bytes
        )));
    }
    let module = kind.module_for(g, false);
    let func = load(dev, module, kind.entry, || kind.source(g, false))?;
    let d_w = resident_cuda_weights(dev, launch.weights)?;
    // SAFETY: the grid gives every row a threadblock's lane `i` (rows
    // `FQ_ROWS * blockIdx.x + i`), which stores it; rows past the
    // matrix store nothing.
    let mut d_out = unsafe { alloc_out(dev, shape.rows, "dp4a matvec output") }?;
    let cfg = LaunchConfig {
        grid_dim: (g.grid_x(shape.rows) as u32, 1, 1),
        block_dim: (g.threads() as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: (const void*, const unsigned char*, float*, int x3), the
    // kernel's parameter list; the weights are `rows * row_bytes` bytes
    // of `kind` and `y` covers a row's q8_1 blocks (both checked above),
    // and `d_out` is `rows` long. Twin: `dp4a_twin::dp4a_matvec_row`.
    unsafe {
        func.launch(
            cfg,
            (
                &d_w.slice,
                y,
                &mut d_out,
                shape.rows as i32,
                shape.row_bytes as i32,
                shape.n_blocks_per_row as i32,
            ),
        )
    }
    .map_err(|e| CudaError::Launch(format!("kernel {}: {e:?}", kind.entry)))?;
    Ok((d_out, d_w))
}

/// The routed-expert form: `k` slots, slot `j`'s weights at `table[j]`,
/// its input `x_stride` floats along `input` (`0` when every slot reads
/// the same vector), its `rows` outputs at `j * rows`.
///
/// # Safety
/// `table` holds `k` device pointers, each to `shape.rows *
/// shape.row_bytes` live bytes of `kind`, alive until the stream has run
/// the launch.
pub(crate) unsafe fn enqueue_dp4a_multi_table(
    dev: &Arc<CudaDevice>,
    kind: &'static Dp4aKind,
    shape: &MultiShape,
    table: &CudaView<'_, u64>,
    k: usize,
    input: &MatvecInput<'_>,
    x_stride: usize,
) -> Result<CudaSlice<f32>, CudaError> {
    if k == 0 || cudarc::driver::DeviceSlice::len(table) < k || u32::try_from(k).is_err() {
        return Err(CudaError::Unsupported(
            "dp4a multi: the table is shorter than its slots".to_string(),
        ));
    }
    let y = input.q8(dev)?;
    let y_stride = x_stride / Q8_1_BLOCK_ELEMS;
    check_shape(
        kind,
        shape,
        cudarc::driver::DeviceSlice::len(y),
        y_stride,
        k,
    )?;
    let g = kind.geometry(compute_capability(dev), shape.n_blocks_per_row);
    let module = kind.module_for(g, true);
    let func = load(dev, module, kind.entry, || kind.source(g, true))?;
    // SAFETY: as for `enqueue_dp4a_at`, per slot: each `blockIdx.y`
    // stores every one of its `rows` outputs.
    let mut d_out = unsafe { alloc_out(dev, k * shape.rows, "dp4a multi output") }?;
    let cfg = LaunchConfig {
        grid_dim: (g.grid_x(shape.rows) as u32, k as u32, 1),
        block_dim: (g.threads() as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: (const void*, const unsigned char*, float*, int x4), the
    // kernel's FX_MULTI parameter list; the table's first `k` pointers
    // are live weights of this kind and shape (the caller's contract),
    // every slot's input is inside `y` (checked above), `d_out` is
    // `k * rows`.
    unsafe {
        func.launch(
            cfg,
            (
                table,
                y,
                &mut d_out,
                shape.rows as i32,
                shape.row_bytes as i32,
                shape.n_blocks_per_row as i32,
                y_stride as i32,
            ),
        )
    }
    .map_err(|e| CudaError::Launch(format!("kernel {} multi: {e:?}", kind.entry)))?;
    Ok(d_out)
}

#[cfg(test)]
mod tests;
