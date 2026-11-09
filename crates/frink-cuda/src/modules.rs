//! The NVRTC module cache every kernel launch goes through, moved out
//! of `gpu.rs` (ADR 0008); `gpu.rs` re-exports it so every existing
//! path still resolves.

use crate::gpu::CudaError;

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
