//! Device-to-device enqueues for the prefill kernels: each takes device
//! slices, launches on the shared device's default stream and returns
//! without a host sync. `launch_prefill_dense_layer` chains them; the
//! hardware tests call them one at a time against a host twin.

use super::kernels::{
    ADD_BIAS_ROWS, ADD_BIAS_ROWS_KERNEL_SRC, ADD_ROWS, ADD_ROWS_KERNEL_SRC, CAUSAL_GQA_PREFILL,
    CAUSAL_GQA_PREFILL_KERNEL_SRC, RMSNORM_ROWS, RMSNORM_ROWS_KERNEL_SRC, ROPE_ROWS,
    ROPE_ROWS_KERNEL_SRC,
};
use crate::decode_attn;
use crate::gpu::{ensure_module_loaded, CudaError};
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig,
};
use std::sync::Arc;

/// Threads per block for the elementwise kernels and the row norm.
const BLOCK: u32 = 256;

fn func_for(
    dev: &Arc<CudaDevice>,
    src: &'static str,
    (module, name): (&'static str, &'static str),
) -> Result<CudaFunction, CudaError> {
    ensure_module_loaded(dev, src, module, name)?;
    dev.get_func(module, name).ok_or_else(|| {
        CudaError::KernelCompile(format!("function '{name}' not found after load_ptx"))
    })
}

fn elementwise_cfg(total: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((total as u32).div_ceil(BLOCK).max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_err(what: &str, e: impl std::fmt::Debug) -> CudaError {
    CudaError::Launch(format!("{what}: {e:?}"))
}

/// `out[r] = rms_norm(x[r], w, eps)` for `n_rows` rows of `n`.
pub(crate) fn enqueue_rmsnorm_rows(
    dev: &Arc<CudaDevice>,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    n_rows: usize,
    n: usize,
    eps: f32,
) -> Result<CudaSlice<f32>, CudaError> {
    debug_assert!(x.len() >= n_rows * n);
    debug_assert!(w.len() >= n);
    let func = func_for(dev, RMSNORM_ROWS_KERNEL_SRC, RMSNORM_ROWS)?;
    // SAFETY: one block per row (`grid = n_rows`), and each block's
    // threads stride over all `n` elements of its row's output.
    let mut out = unsafe { crate::gpu::alloc_out(dev, n_rows * n, "rmsnorm_rows") }?;
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the kernel's parameter list is (const float*, const
    // float*, float*, int, int, float), matched positionally; `x` holds
    // `n_rows * n` values and `w` holds `n` (the debug asserts above,
    // and the caller's shape checks), `out` is allocated at exactly
    // `n_rows * n`, and the kernel returns for a block past `n_rows`.
    unsafe {
        func.launch(cfg, (x, w, &mut out, n_rows as i32, n as i32, eps))
            .map_err(|e| launch_err("rmsnorm_rows", e))?;
    }
    Ok(out)
}

/// `x[r][i] += bias[i]`, in place.
pub(crate) fn enqueue_add_bias_rows(
    dev: &Arc<CudaDevice>,
    x: &mut CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    n_rows: usize,
    n: usize,
) -> Result<(), CudaError> {
    debug_assert!(x.len() >= n_rows * n);
    debug_assert!(bias.len() >= n);
    let func = func_for(dev, ADD_BIAS_ROWS_KERNEL_SRC, ADD_BIAS_ROWS)?;
    let cfg = elementwise_cfg(n_rows * n);
    // SAFETY: (float*, const float*, int, int) matched positionally;
    // the kernel bounds-checks its index against `n_rows * n`, which
    // both buffers cover.
    unsafe {
        func.launch(cfg, (x, bias, n_rows as i32, n as i32))
            .map_err(|e| launch_err("add_bias_rows", e))?;
    }
    Ok(())
}

/// `a[i] += b[i]` over `n`, in place.
pub(crate) fn enqueue_add_rows(
    dev: &Arc<CudaDevice>,
    a: &mut CudaSlice<f32>,
    b: &CudaSlice<f32>,
    n: usize,
) -> Result<(), CudaError> {
    debug_assert!(a.len() >= n && b.len() >= n);
    let func = func_for(dev, ADD_ROWS_KERNEL_SRC, ADD_ROWS)?;
    let cfg = elementwise_cfg(n);
    // SAFETY: (float*, const float*, int) matched positionally; the
    // kernel bounds-checks against `n`, which both buffers cover.
    unsafe {
        func.launch(cfg, (a, b, n as i32))
            .map_err(|e| launch_err("add_rows", e))?;
    }
    Ok(())
}

/// The rotation's per-layer inputs, as one value so a rotation site
/// cannot take the base without the divisors
/// (`frink_models::config::ModelConfig::layer_rope`).
pub struct RopeArgs<'a> {
    pub theta: f32,
    pub freq_factors: Option<&'a CudaSlice<f32>>,
    pub rot_dim: usize,
    pub neox: bool,
    pub mscale: f32,
}

/// RoPE in place over `[n_rows, n_heads, head_dim]`, row `r` at position
/// `start_pos + r`.
pub(crate) fn enqueue_rope_rows(
    dev: &Arc<CudaDevice>,
    x: &mut CudaSlice<f32>,
    n_rows: usize,
    n_heads: usize,
    head_dim: usize,
    start_pos: usize,
    rope: &RopeArgs<'_>,
) -> Result<(), CudaError> {
    debug_assert!(x.len() >= n_rows * n_heads * head_dim);
    debug_assert!(rope.rot_dim >= 2 && rope.rot_dim.is_multiple_of(2) && rope.rot_dim <= head_dim);
    if let Some(ff) = rope.freq_factors {
        debug_assert!(ff.len() >= rope.rot_dim / 2);
    }
    let func = func_for(dev, ROPE_ROWS_KERNEL_SRC, ROPE_ROWS)?;
    let total = n_rows * n_heads * (rope.rot_dim / 2);
    let cfg = elementwise_cfg(total);
    // A null pointer for "no divisors": the kernel tests it. cudarc has
    // no `Option<&CudaSlice>` argument, so an absent table is a zero
    // device pointer, which `sys::CUdeviceptr` spells as 0.
    let ff_ptr: cudarc::driver::sys::CUdeviceptr = match rope.freq_factors {
        Some(ff) => *ff.device_ptr(),
        None => 0,
    };
    // SAFETY: (float*, const float*, int x5, float, int, float)
    // matched positionally; every thread past `n_rows * n_heads *
    // rot_dim/2` returns, `x` covers `n_rows * n_heads * head_dim` and
    // the two indices a thread touches are below `rot_dim <= head_dim`
    // inside its own head; `ff_ptr` is either 0 (never dereferenced)
    // or a live buffer of at least `rot_dim / 2` values held by the
    // caller for the duration of the stream.
    unsafe {
        func.launch(
            cfg,
            (
                x,
                ff_ptr,
                n_rows as i32,
                n_heads as i32,
                head_dim as i32,
                rope.rot_dim as i32,
                start_pos as i32,
                rope.theta,
                i32::from(rope.neox),
                rope.mscale,
            ),
        )
        .map_err(|e| launch_err("rope_rows", e))?;
    }
    Ok(())
}

/// The attention geometry the kernel takes, as one value.
pub struct AttnArgs {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub start_pos: usize,
    /// `None` for full causal attention.
    pub window: Option<usize>,
    pub scale: f32,
    /// `None` for no logit softcap.
    pub softcap: Option<f32>,
}

/// Dynamic shared bytes the tiled prefill attention needs at `rpw`
/// rows per warp and `bc` keys per tile, for a head of `head_dim`: the
/// block's Q rows, the K tile (rows padded by 4 floats), the V tile,
/// and each warp's 32 probabilities.
pub(crate) fn prefill_attn_smem(head_dim: usize, rpw: usize, bc: usize) -> usize {
    4 * (4 * rpw * head_dim + bc * (head_dim + 4) + bc * head_dim + 4 * 32)
}

/// The largest tile that fits `limit` bytes, in the order a wider tile
/// shares more: `(rows per warp, keys per tile, bytes)`.
pub(crate) fn prefill_attn_tile(head_dim: usize, limit: usize) -> Option<(usize, usize, usize)> {
    [(8, 32), (4, 32), (4, 16), (2, 16), (2, 8), (1, 8)]
        .into_iter()
        .map(|(rpw, bc)| (rpw, bc, prefill_attn_smem(head_dim, rpw, bc)))
        .find(|&(_, _, bytes)| bytes <= limit)
}

/// The shared memory a launch of the prefill attention may ask for:
/// the device's opt-in limit, which the function is opted into once
/// (above 48 KiB a launch must be). 48 KiB on a driver that refuses.
fn prefill_attn_smem_limit(dev: &Arc<CudaDevice>, func: &CudaFunction) -> Result<usize, CudaError> {
    use cudarc::driver::sys::{CUdevice_attribute_enum as Dev, CUfunction_attribute_enum as Fun};
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    if let Some(&limit) = LIMIT.get() {
        return Ok(limit);
    }
    let optin = dev
        .attribute(Dev::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        .map(|v| v.max(0) as usize)
        .unwrap_or(48 * 1024);
    let limit = match func.set_attribute(
        Fun::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        optin as i32,
    ) {
        Ok(()) => optin,
        Err(_) => 48 * 1024,
    };
    Ok(*LIMIT.get_or_init(|| limit))
}

/// Causal GQA for `n_q` query rows over `k_all` / `v_all` holding
/// `start_pos + n_q` positions. Returns `[n_q, n_heads, head_dim]`.
pub(crate) fn enqueue_causal_gqa_prefill(
    dev: &Arc<CudaDevice>,
    q: &CudaSlice<f32>,
    k_all: &CudaSlice<f32>,
    v_all: &CudaSlice<f32>,
    n_q: usize,
    args: &AttnArgs,
) -> Result<CudaSlice<f32>, CudaError> {
    let AttnArgs {
        n_heads,
        n_kv_heads,
        head_dim,
        start_pos,
        window,
        scale,
        softcap,
    } = *args;
    if head_dim > 256 || !head_dim.is_multiple_of(4) {
        return Err(CudaError::Unsupported(format!(
            "causal_gqa_prefill: head_dim {head_dim} (the kernel takes a multiple of 4 up to 256)"
        )));
    }
    if n_heads >= 1 << 16 || n_kv_heads >= 1 << 16 {
        return Err(CudaError::Unsupported(format!(
            "causal_gqa_prefill: {n_heads} heads over {n_kv_heads} (the launch packs each in 16 bits)"
        )));
    }
    debug_assert!(q.len() >= n_q * n_heads * head_dim);
    debug_assert!(k_all.len() >= (start_pos + n_q) * n_kv_heads * head_dim);
    debug_assert!(v_all.len() >= (start_pos + n_q) * n_kv_heads * head_dim);
    let func = func_for(dev, CAUSAL_GQA_PREFILL_KERNEL_SRC, CAUSAL_GQA_PREFILL)?;
    let optin = prefill_attn_smem_limit(dev, &func)?;
    let (rpw, bc, smem) = prefill_attn_tile(head_dim, optin).ok_or_else(|| {
        CudaError::Unsupported(format!(
            "causal_gqa_prefill: no key tile of a head of {head_dim} fits {optin} bytes of shared \
             memory"
        ))
    })?;
    let mut out = dev
        .alloc_zeros::<f32>(n_q * n_heads * head_dim)
        .map_err(|e| launch_err("causal_gqa_prefill alloc", e))?;
    let cfg = LaunchConfig {
        grid_dim: ((n_q as u32).div_ceil(4 * rpw as u32), n_heads as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    // SAFETY: (const float* x3, float*, int x5, float, float, int)
    // matched positionally, the head counts and the tile packed as the
    // kernel unpacks them (each below 2^16 and 2^8); the grid covers `n_q x n_heads`, every key
    // index read is below `start_pos + n_q`, which the K/V buffers
    // cover, the `float4` views need `head_dim % 4 == 0` (checked above)
    // on buffers cudarc allocates 256-byte aligned, and the dynamic
    // shared memory is `prefill_attn_tile`'s size for this `rpw` and
    // `bc`, within the limit the function was opted into.
    unsafe {
        func.launch(
            cfg,
            (
                q,
                k_all,
                v_all,
                &mut out,
                n_q as i32,
                (n_heads | (n_kv_heads << 16)) as i32,
                head_dim as i32,
                start_pos as i32,
                window.map_or(0, |w| w as i32),
                scale,
                softcap.filter(|c| *c > 0.0).unwrap_or(0.0),
                (rpw | (bc << 8)) as i32,
            ),
        )
        .map_err(|e| launch_err("causal_gqa_prefill", e))?;
    }
    Ok(out)
}

/// Attention for ONE query row (`crate::decode_attn`): the keys split
/// into chunks of [`decode_attn::CHUNK`], one block per (head, chunk),
/// and a second launch merging the chunks. `k_all` / `v_all` hold at
/// least `start_pos + 1` rows; the query is at position `start_pos`.
/// Returns `[n_heads, head_dim]`. Two launches and two scratch
/// allocations, no host sync.
pub(crate) fn enqueue_decode_attn(
    dev: &Arc<CudaDevice>,
    q: &CudaSlice<f32>,
    k_all: &CudaSlice<f32>,
    v_all: &CudaSlice<f32>,
    args: &AttnArgs,
) -> Result<CudaSlice<f32>, CudaError> {
    let AttnArgs {
        n_heads,
        n_kv_heads,
        head_dim,
        start_pos,
        window,
        scale,
        softcap,
    } = *args;
    if !decode_attn::head_dim_ok(head_dim) {
        return Err(CudaError::Unsupported(format!(
            "decode_attn: head_dim {head_dim} (the kernel takes a multiple of 4 up to 256)"
        )));
    }
    let seq_len = start_pos + 1;
    let (lo, hi) = decode_attn::key_range(seq_len, window);
    let n_chunks = decode_attn::n_chunks(hi - lo);
    debug_assert!(q.len() >= n_heads * head_dim);
    debug_assert!(k_all.len() >= seq_len * n_kv_heads * head_dim);
    debug_assert!(v_all.len() >= seq_len * n_kv_heads * head_dim);
    let src = decode_attn::DECODE_ATTN_KERNEL_SRC;
    // One translation unit, loaded as two modules: `load_ptx`
    // registers only the entry points it is told, one per module here.
    let partial = func_for(dev, src, decode_attn::PARTIAL)?;
    let combine = func_for(dev, src, decode_attn::COMBINE)?;
    // SAFETY: the partial grid is exactly n_heads x n_chunks, and every
    // block writes its slot's `head_dim` accumulators and its (m, l)
    // pair, an empty chunk included; the combine gives each head a
    // block and each of its `head_dim` channels a thread.
    let mut part = unsafe {
        crate::gpu::alloc_out(
            dev,
            decode_attn::scratch_len(n_heads, n_chunks, head_dim),
            "decode_attn scratch",
        )
    }?;
    // SAFETY: as above, the combine writes every `out[h * head_dim + d]`.
    let mut out = unsafe { crate::gpu::alloc_out(dev, n_heads * head_dim, "decode_attn") }?;
    let (grid, block) = decode_attn::geometry::partial(n_heads, n_chunks);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };
    // SAFETY: (const float* x3, float*, int x6, float x2) matched
    // positionally. The grid is exactly n_heads x n_chunks, so every
    // block's (head, chunk) is in range; every key index is in
    // `lo..seq_len`, rows the K/V buffers cover (debug asserts above,
    // the stack's reserve before it); the `float4` views need
    // `head_dim % 4 == 0` (checked above) on buffers cudarc allocates
    // 256-byte aligned; the scratch is allocated at exactly
    // `scratch_len`, every slot the kernel writes.
    unsafe {
        partial
            .launch(
                cfg,
                (
                    q,
                    k_all,
                    v_all,
                    &mut part,
                    n_heads as i32,
                    n_kv_heads as i32,
                    head_dim as i32,
                    lo as i32,
                    hi as i32,
                    n_chunks as i32,
                    scale,
                    softcap.filter(|c| *c > 0.0).unwrap_or(0.0),
                ),
            )
            .map_err(|e| launch_err("decode_attn_partial", e))?;
    }
    let (grid, block) = decode_attn::geometry::combine(n_heads, head_dim);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };
    // SAFETY: (const float*, float*, int x3) matched positionally;
    // one block per head and one thread per channel, both in range by
    // construction; the scratch holds the `n_chunks` slots per head the
    // partial launch wrote, ordered before this on the same stream.
    unsafe {
        combine
            .launch(
                cfg,
                (
                    &part,
                    &mut out,
                    n_heads as i32,
                    head_dim as i32,
                    n_chunks as i32,
                ),
            )
            .map_err(|e| launch_err("decode_attn_combine", e))?;
    }
    Ok(out)
}
