//! The NVRTC module cache every kernel launch goes through, moved out
//! of `gpu.rs` (ADR 0008); `gpu.rs` re-exports it so every existing
//! path still resolves.

use crate::gpu::CudaError;

/// Every module NVRTC has compiled for the process, and which device
/// objects have loaded which. `CudaDevice::load_ptx` always recompiles
/// and overwrites on every call (verified directly against `cudarc`
/// 0.11.9's own source: `modules.insert(module_name.into(), module)`
/// with no existing-key check), so skipping a repeat call for an
/// already-loaded kernel has to be tracked here, not assumed.
///
/// Loaded is per device object because a module is: the shared device
/// and the device a decode graph is recorded on (`crate::graph`) are
/// two `CudaDevice`s over one context, each with its own module table,
/// and a kernel launched through one is recorded on that one's stream.
/// The compiled text is kept so the second loads it without compiling
/// again ([`preload_modules`]).
#[derive(Default)]
struct ModuleCache {
    /// Each module's PTX and its one entry point.
    ptx: std::collections::HashMap<&'static str, (cudarc::nvrtc::Ptx, &'static str)>,
    /// `(device object, module)` pairs already loaded.
    loaded: std::collections::HashSet<(usize, &'static str)>,
}

static MODULES: std::sync::Mutex<Option<ModuleCache>> = std::sync::Mutex::new(None);

fn device_key(dev: &std::sync::Arc<cudarc::driver::CudaDevice>) -> usize {
    std::sync::Arc::as_ptr(dev) as usize
}

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
    let mut guard = MODULES.lock().unwrap();
    let cache = guard.get_or_insert_with(ModuleCache::default);
    let key = (device_key(dev), module_name);
    if cache.loaded.contains(&key) {
        return Ok(());
    }
    let ptx = match cache.ptx.get(module_name) {
        Some((ptx, _)) => ptx.clone(),
        None => {
            let opts = cudarc::nvrtc::CompileOptions {
                arch,
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(src(), opts)
                .map_err(|e| CudaError::KernelCompile(format!("{e:?}")))?;
            cache.ptx.insert(module_name, (ptx.clone(), fn_name));
            ptx
        }
    };
    dev.load_ptx(ptx, module_name, &[fn_name])
        .map_err(|e| CudaError::KernelCompile(format!("{e:?}")))?;
    cache.loaded.insert(key);
    Ok(())
}

/// Loads into `dev` every module compiled so far that it has not
/// loaded, from the kept PTX: what a capture device does before a
/// capture, so the warm token's kernels are all there and none is
/// loaded inside the capture.
pub(crate) fn preload_modules(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<(), CudaError> {
    let mut guard = MODULES.lock().unwrap();
    let cache = guard.get_or_insert_with(ModuleCache::default);
    let d = device_key(dev);
    let missing: Vec<(&'static str, cudarc::nvrtc::Ptx, &'static str)> = cache
        .ptx
        .iter()
        .filter(|(m, _)| !cache.loaded.contains(&(d, **m)))
        .map(|(m, (ptx, f))| (*m, ptx.clone(), *f))
        .collect();
    for (module, ptx, entry) in missing {
        dev.load_ptx(ptx, module, &[entry])
            .map_err(|e| CudaError::KernelCompile(format!("{e:?}")))?;
        cache.loaded.insert((d, module));
    }
    Ok(())
}
