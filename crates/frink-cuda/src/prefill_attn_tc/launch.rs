//! The launch of the tensor-core prefill attention, and the choice
//! between it and the SIMT kernel that `enqueue_causal_gqa_prefill`
//! makes for every batched layer.

use super::{
    device_takes, grid, heads_per_block, simt_requested, variant_for, AttnArgs, Variant,
    STATIC_SMEM_LIMIT, THREADS,
};
use crate::gpu::{compute_capability_major, CudaError};
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig,
};
use std::sync::{Arc, Mutex, OnceLock};

/// Which body a batched prefill attention runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillAttnBody {
    /// `prefill::kernels::CAUSAL_GQA_PREFILL_KERNEL_SRC`: f32 FMA. Every
    /// device, every head width up to 256.
    Simt,
    /// `crate::prefill_attn_tc`: f16 `mma.sync` with f32 accumulation.
    /// `sm_80` and up, head widths 64, 128 and 256.
    TensorCore,
}

impl PrefillAttnBody {
    /// The tensor-core body when the shape is one it is built for, the
    /// device can run it and `FRINK_CUDA_PREFILL_ATTN` does not say
    /// `simt`. Otherwise the SIMT body. Every input is read once per
    /// process, so this costs a few loads a layer.
    pub fn select(dev: &Arc<CudaDevice>, args: &AttnArgs) -> Self {
        match variant_for(args) {
            Some(v)
                if !forced_simt()
                    && device_takes(v, compute_capability_major(dev), smem_optin(dev)) =>
            {
                Self::TensorCore
            }
            _ => Self::Simt,
        }
    }
}

/// `FRINK_CUDA_PREFILL_ATTN=simt`, read once.
pub fn forced_simt() -> bool {
    static FORCED: OnceLock<bool> = OnceLock::new();
    *FORCED.get_or_init(|| simt_requested(std::env::var("FRINK_CUDA_PREFILL_ATTN").ok().as_deref()))
}

/// The dynamic shared memory one block may opt in to on the shared
/// device, asked once. 48 KiB when the driver does not say.
fn smem_optin(dev: &Arc<CudaDevice>) -> usize {
    use cudarc::driver::sys::CUdevice_attribute_enum as Dev;
    static OPTIN: OnceLock<usize> = OnceLock::new();
    *OPTIN.get_or_init(|| {
        dev.attribute(Dev::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .map_or(STATIC_SMEM_LIMIT, |v| v.max(0) as usize)
    })
}

/// The variant's function, compiled for `sm_80` and loaded once, and
/// opted in to its dynamic shared memory once when that is above what a
/// launch gets without asking.
fn function(dev: &Arc<CudaDevice>, v: &'static Variant) -> Result<CudaFunction, CudaError> {
    use cudarc::driver::sys::CUfunction_attribute_enum as Fun;
    static OPTED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let (module, entry) = v.names;
    crate::gpu::ensure_module_loaded_lazy_for_arch(
        dev,
        module,
        entry,
        Some(crate::mul_mm_tc::ARCH),
        || v.kernel_src(),
    )?;
    let func = dev.get_func(module, entry).ok_or_else(|| {
        CudaError::KernelCompile(format!("function '{entry}' not found after load_ptx"))
    })?;
    if v.smem_bytes() > STATIC_SMEM_LIMIT {
        let mut opted = OPTED.lock().unwrap_or_else(|p| p.into_inner());
        if !opted.contains(&module) {
            func.set_attribute(
                Fun::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                v.smem_bytes() as i32,
            )
            .map_err(|e| {
                CudaError::Launch(format!(
                    "{entry}: opting in to {} shared bytes: {e:?}",
                    v.smem_bytes()
                ))
            })?;
            opted.push(module);
        }
    }
    Ok(func)
}

/// Causal GQA for `n_q` query rows over `k_all` / `v_all` holding
/// `start_pos + n_q` positions, on the tensor cores, at the group size
/// [`heads_per_block`] picks. Returns `[n_q, n_heads, head_dim]`.
pub(crate) fn enqueue_prefill_attn_tc(
    dev: &Arc<CudaDevice>,
    q: &CudaSlice<f32>,
    k_all: &CudaSlice<f32>,
    v_all: &CudaSlice<f32>,
    n_q: usize,
    args: &AttnArgs,
    sinks: Option<&CudaSlice<f32>>,
) -> Result<CudaSlice<f32>, CudaError> {
    let group = heads_per_block(args.gqa(), n_q);
    enqueue_prefill_attn_tc_with_group(dev, q, k_all, v_all, n_q, args, sinks, group)
}

/// [`enqueue_prefill_attn_tc`] at `group` query heads per block, which
/// the hardware tests set to every value to reach each row mapping.
#[allow(clippy::too_many_arguments)] // The buffers and shape of the launch above plus the one choice it makes. A struct would only rename them.
pub(crate) fn enqueue_prefill_attn_tc_with_group(
    dev: &Arc<CudaDevice>,
    q: &CudaSlice<f32>,
    k_all: &CudaSlice<f32>,
    v_all: &CudaSlice<f32>,
    n_q: usize,
    args: &AttnArgs,
    sinks: Option<&CudaSlice<f32>>,
    group: usize,
) -> Result<CudaSlice<f32>, CudaError> {
    let AttnArgs {
        n_heads,
        n_kv_heads,
        head_dim,
        start_pos,
        window: _,
        scale: _,
        softcap: _,
    } = *args;
    let v = variant_for(args).ok_or_else(|| {
        CudaError::Unsupported(format!(
            "prefill_attn_tc: head_dim {head_dim}, {n_heads} heads over {n_kv_heads} (the \
             kernel takes 64, 128 or 256 and a whole GQA ratio)"
        ))
    })?;
    if !super::GROUPS.contains(&group) || n_q == 0 || start_pos + n_q > i32::MAX as usize {
        return Err(CudaError::Unsupported(format!(
            "prefill_attn_tc: {n_q} rows after {start_pos} at {group} heads per block"
        )));
    }
    let (cc, optin) = (compute_capability_major(dev), smem_optin(dev));
    if !device_takes(v, cc, optin) {
        return Err(CudaError::Unsupported(format!(
            "prefill_attn_tc: compute capability {cc} with {optin} bytes of shared memory a \
             block (the kernel needs sm_80 and {} bytes)",
            v.smem_bytes()
        )));
    }
    let kv_len = (start_pos + n_q) * n_kv_heads * head_dim;
    debug_assert!(q.len() >= n_q * n_heads * head_dim);
    debug_assert!(k_all.len() >= kv_len && v_all.len() >= kv_len);
    debug_assert!(sinks.is_none_or(|s| s.len() >= n_heads));
    let func = function(dev, v)?;
    // SAFETY: every (token, head) of the batch is one row of one block
    // (`super::block_row`, `the_grid_rows_cover_every_token_and_head_once`)
    // and the quad of lanes holding a row stores all `head_dim` of its
    // channels, eight per n8 tile.
    let mut out =
        unsafe { crate::gpu::alloc_out(dev, n_q * n_heads * head_dim, "prefill_attn_tc") }?;
    let cfg = LaunchConfig {
        grid_dim: grid(n_q, n_kv_heads, args.gqa(), group),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: v.smem_bytes() as u32,
    };
    let sinks = sinks.map_or(0u64, |s| *s.device_ptr());
    // SAFETY: the arguments are (const float* x3, float*, const float*
    // or null, int x4, float x2, int), matched positionally, with the
    // head counts packed as the kernel unpacks them (`variant_for` holds
    // them below 2^16 and 2^15). The sinks pointer is null or `n_heads`
    // logits the caller holds until the stream's wait. The grid is
    // `grid`'s for this group, and the kernel guards every row past
    // `n_q` or past its KV head's query heads. Every key it reads is
    // below `start_pos + n_q`, which the K/V buffers cover (the debug
    // asserts, and the stack's shapes). The `float4` loads need
    // `head_dim % 4 == 0` (64, 128 or 256) on buffers cudarc allocates
    // 256-byte aligned. The dynamic shared memory is `smem_bytes`, within
    // the device's limit (`device_takes` above), and `function` opted the
    // function in to it when it is above 48 KiB.
    unsafe {
        func.launch(
            cfg,
            (
                q,
                k_all,
                v_all,
                &mut out,
                sinks,
                n_q as i32,
                (n_heads | (n_kv_heads << 16)) as i32,
                start_pos as i32,
                args.window_or_zero().min(i32::MAX as usize) as i32,
                args.q_factor(),
                args.softcap_or_zero(),
                group as i32,
            ),
        )
        .map_err(|e| CudaError::Launch(format!("prefill_attn_tc: {e:?}")))?;
    }
    Ok(out)
}
