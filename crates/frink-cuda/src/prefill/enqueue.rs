//! Device-to-device enqueues for the prefill kernels: each takes device
//! slices, launches on the shared device's default stream and returns
//! without a host sync. `launch_prefill_dense_layer` chains them; the
//! hardware tests call them one at a time against a host twin.

use super::kernels::{
    ADD_BIAS_ROWS, ADD_BIAS_ROWS_KERNEL_SRC, ADD_BIAS_TAB_ROWS, ADD_BIAS_TAB_ROWS_KERNEL_SRC,
    ADD_ROWS, ADD_ROWS_KERNEL_SRC, AXPY, AXPY_KERNEL_SRC, CAUSAL_GQA_PREFILL,
    CAUSAL_GQA_PREFILL_KERNEL_SRC, GATHER_PTRS, GATHER_PTRS_KERNEL_SRC, MOE_COMBINE,
    MOE_COMBINE_KERNEL_SRC, MOE_COMBINE_ROWS, MOE_COMBINE_ROWS_KERNEL_SRC, MOE_COUNT,
    MOE_GROUP_KERNEL_SRC, MOE_OFFSETS, MOE_SCATTER, MOE_TOPK, MOE_TOPK_KERNEL_SRC, RMSNORM_ROWS,
    RMSNORM_ROWS_KERNEL_SRC, ROPE_ROWS, ROPE_ROWS_KERNEL_SRC, ROUTER_F32, ROUTER_F32_KERNEL_SRC,
};
use crate::decode_attn;
use crate::gpu::{ensure_module_loaded, CudaError};
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtr, DeviceSlice,
    LaunchAsync, LaunchConfig,
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

/// Threads per row for the row RMSNorm: a few rows (a decode token's
/// one) get up to a quarter as many threads as the row has elements, so
/// a row of 4096 is not walked sixteen deep by 256 threads; many rows
/// (a prefill batch) keep 256 and fill the GPU with blocks instead.
/// Always a multiple of 32 between 32 and 1024.
pub(crate) fn rmsnorm_threads(n_rows: usize, n: usize) -> u32 {
    if n_rows >= 64 {
        return BLOCK;
    }
    (n / 4).next_power_of_two().clamp(32, 1024) as u32
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
        block_dim: (rmsnorm_threads(n_rows, n), 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the kernel's parameter list is (const float*, const
    // float*, float*, int, int, float), matched positionally; the block
    // is a multiple of 32 threads up to 1024 (`rmsnorm_threads`), the
    // 32 warp slots the kernel reduces through; `x` holds
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

/// `y[..n] += a * x[..n]`, in place.
pub(crate) fn enqueue_axpy(
    dev: &Arc<CudaDevice>,
    y: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    a: f32,
    n: usize,
) -> Result<(), CudaError> {
    debug_assert!(y.len() >= n && x.len() >= n);
    let func = func_for(dev, AXPY_KERNEL_SRC, AXPY)?;
    let cfg = elementwise_cfg(n);
    // SAFETY: (float*, const float*, float, int) matched positionally;
    // the kernel bounds-checks against `n`, which both buffers cover.
    unsafe {
        func.launch(cfg, (y, x, a, n as i32))
            .map_err(|e| launch_err("axpy", e))?;
    }
    Ok(())
}

/// `sum_j w[j] * y[j * n ..]`, the layer's routed experts combined.
pub(crate) fn enqueue_moe_combine(
    dev: &Arc<CudaDevice>,
    y: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    k: usize,
    n: usize,
) -> Result<CudaSlice<f32>, CudaError> {
    debug_assert!(y.len() >= k * n && w.len() >= k);
    let func = func_for(dev, MOE_COMBINE_KERNEL_SRC, MOE_COMBINE)?;
    // SAFETY: thread `i < n` writes `out[i]`, and the grid covers `n`.
    let mut out = unsafe { crate::gpu::alloc_out(dev, n, "moe combine") }?;
    let cfg = elementwise_cfg(n);
    // SAFETY: (const float*, const float*, float*, int, int) matched
    // positionally; `y` holds `k * n`, `w` `k`, `out` `n`.
    unsafe {
        func.launch(cfg, (y, w, &mut out, k as i32, n as i32))
            .map_err(|e| launch_err("moe_combine", e))?;
    }
    Ok(out)
}

/// `x[r] += tab[row_group[r]]` for each of `n_rows` rows of `n`, or
/// `tab[r]` without a `row_group`.
pub(crate) fn enqueue_add_bias_tab_rows(
    dev: &Arc<CudaDevice>,
    x: &mut CudaSlice<f32>,
    tab: &CudaView<'_, u64>,
    row_group: Option<&CudaSlice<i32>>,
    n_rows: usize,
    n: usize,
) -> Result<(), CudaError> {
    debug_assert!(x.len() >= n_rows * n);
    debug_assert!(row_group.is_none_or(|g| g.len() >= n_rows));
    debug_assert!(row_group.is_some() || tab.len() >= n_rows);
    let func = func_for(dev, ADD_BIAS_TAB_ROWS_KERNEL_SRC, ADD_BIAS_TAB_ROWS)?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK).max(1), n_rows as u32, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let row_group = row_group.map_or(0u64, |g| *cudarc::driver::DevicePtr::device_ptr(g));
    // SAFETY: (float*, const u64*, const int* or null, int) matched
    // positionally; a grid row per row of `x` and thread `i < n` of it;
    // every `row_group` entry (or, without one, every row index) indexes
    // `tab` (the caller builds them from one grouping), and every table
    // entry is a live device vector of at least `n` floats the caller
    // holds until its wait.
    unsafe {
        func.launch(cfg, (x, tab, row_group, n as i32))
            .map_err(|e| launch_err("add_bias_tab_rows", e))?;
    }
    Ok(())
}

/// The routing of `batch` rows of router logits on the device
/// (`kernels::MOE_TOPK_KERNEL_SRC`): `k` expert ids into `ids` and
/// their weights, `[batch][k]` each.
#[allow(clippy::too_many_arguments)]
pub(crate) fn enqueue_moe_topk(
    dev: &Arc<CudaDevice>,
    logits: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    n_experts: usize,
    k: usize,
    mode: i32,
    batch: usize,
    ids: &mut CudaViewMut<'_, i32>,
) -> Result<CudaSlice<f32>, CudaError> {
    if n_experts > 256 || k == 0 || k > 16 || k > n_experts {
        return Err(CudaError::Unsupported(format!(
            "moe_topk: {k} of {n_experts} experts (the kernel takes up to 16 of 256)"
        )));
    }
    debug_assert!(logits.len() >= batch * n_experts && ids.len() >= batch * k);
    debug_assert!(bias.is_none_or(|b| b.len() >= n_experts));
    let func = func_for(dev, MOE_TOPK_KERNEL_SRC, MOE_TOPK)?;
    // SAFETY: lane 0 of each token's warp writes all `k` of its weights.
    let mut w = unsafe { crate::gpu::alloc_out(dev, batch * k, "moe weights") }?;
    let bias = bias.map_or(0u64, |b| *cudarc::driver::DevicePtr::device_ptr(b));
    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: (const float*, const float* or null, int x3, int*, float*)
    // matched positionally; one warp per token reads its `n_experts`
    // logits (and `n_experts` biases) and lane 0 writes its `k` ids and
    // weights, every buffer that long (the debug asserts, the check
    // above and the caller's shapes).
    unsafe {
        func.launch(
            cfg,
            (logits, bias, n_experts as i32, k as i32, mode, ids, &mut w),
        )
        .map_err(|e| launch_err("moe_topk", e))?;
    }
    Ok(w)
}

/// A routed batch's pairs grouped by expert ([`MOE_GROUP_KERNEL_SRC`]).
pub(crate) struct MoeGroups {
    /// `n + 1` pair indices: expert `e`'s pairs are `[off[e], off[e+1])`.
    pub group_off: CudaSlice<i32>,
    /// The token each grouped pair reads.
    pub gather: CudaSlice<i32>,
    /// Where pair `t * k + s`'s output row lands.
    pub pair_of: CudaSlice<i32>,
    /// The expert of each grouped pair, for its bias rows.
    pub pair_group: CudaSlice<i32>,
}

/// Groups `pairs` routed pairs (`ids`, `k` a token) by their `n`
/// experts on the device: three launches and no host wait.
pub(crate) fn enqueue_moe_group(
    dev: &Arc<CudaDevice>,
    ids: &CudaView<'_, i32>,
    pairs: usize,
    k: usize,
    n: usize,
) -> Result<MoeGroups, CudaError> {
    debug_assert!(ids.len() >= pairs && k > 0);
    let zeros =
        |len: usize, what: &str| dev.alloc_zeros::<i32>(len).map_err(|e| launch_err(what, e));
    let mut counts = zeros(n, "moe counts")?;
    let mut cursor = zeros(n, "moe cursor")?;
    let mut group_off = zeros(n + 1, "moe offsets")?;
    let mut gather = zeros(pairs, "moe gather")?;
    let mut pair_of = zeros(pairs, "moe pair_of")?;
    let mut pair_group = zeros(pairs, "moe pair_group")?;
    let count = func_for(dev, MOE_GROUP_KERNEL_SRC, MOE_COUNT)?;
    let offsets = func_for(dev, MOE_GROUP_KERNEL_SRC, MOE_OFFSETS)?;
    let scatter = func_for(dev, MOE_GROUP_KERNEL_SRC, MOE_SCATTER)?;
    let one = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: (const int*, int, int*) / (const int*, int, int*) /
    // (const int*, int x2, const int*, int* x4) matched positionally;
    // every id is an expert below `n` (the routing kernel picks among
    // `n`), so every counts, cursor and offsets index is in range; the
    // offsets' last entry is `pairs`, so every scattered place is below
    // `pairs`, the length of gather and pair_group; thread `p < pairs`
    // writes `pair_of[p]`. The three launches run in order on the stream.
    unsafe {
        count
            .launch(elementwise_cfg(pairs), (ids, pairs as i32, &mut counts))
            .map_err(|e| launch_err("moe_count", e))?;
        offsets
            .launch(one, (&counts, n as i32, &mut group_off))
            .map_err(|e| launch_err("moe_offsets", e))?;
        scatter
            .launch(
                elementwise_cfg(pairs),
                (
                    ids,
                    pairs as i32,
                    k as i32,
                    &group_off,
                    &mut cursor,
                    &mut gather,
                    &mut pair_of,
                    &mut pair_group,
                ),
            )
            .map_err(|e| launch_err("moe_scatter", e))?;
    }
    Ok(MoeGroups {
        group_off,
        gather,
        pair_of,
        pair_group,
    })
}

/// `out[t][s] = all[t][ids[s]]` for `n_tabs` tables of `n` pointers.
pub(crate) fn enqueue_gather_ptrs(
    dev: &Arc<CudaDevice>,
    all: &CudaSlice<u64>,
    ids: &CudaView<'_, i32>,
    n: usize,
    n_tabs: usize,
    k: usize,
) -> Result<CudaSlice<u64>, CudaError> {
    debug_assert!(all.len() >= n_tabs * n && ids.len() >= k);
    let func = func_for(dev, GATHER_PTRS_KERNEL_SRC, GATHER_PTRS)?;
    // SAFETY: uninitialised, and thread `i < n_tabs * k` writes `out[i]`
    // before anything reads it.
    let mut out = unsafe { dev.alloc::<u64>(n_tabs * k) }
        .map_err(|e| launch_err("gathered pointers alloc", e))?;
    let cfg = elementwise_cfg(n_tabs * k);
    // SAFETY: (const u64*, const int*, u64*, int x3) matched positionally;
    // every id is below `n` (the routing kernel picks among `n`), so
    // every read is inside `all`'s `n_tabs * n` entries.
    unsafe {
        func.launch(cfg, (all, ids, &mut out, n as i32, n_tabs as i32, k as i32))
            .map_err(|e| launch_err("gather_ptrs", e))?;
    }
    Ok(out)
}

/// The routed prefill's combine, `[batch][n]` from `y[pairs][n]`.
pub(crate) fn enqueue_moe_combine_rows(
    dev: &Arc<CudaDevice>,
    y: &CudaSlice<f32>,
    pair_of: &CudaSlice<i32>,
    w: &CudaSlice<f32>,
    k: usize,
    n: usize,
    batch: usize,
) -> Result<CudaSlice<f32>, CudaError> {
    debug_assert!(pair_of.len() >= batch * k && w.len() >= batch * k);
    let func = func_for(dev, MOE_COMBINE_ROWS_KERNEL_SRC, MOE_COMBINE_ROWS)?;
    // SAFETY: a grid row per token, and thread `i < n` of it writes
    // `out[t][i]`.
    let mut out = unsafe { crate::gpu::alloc_out(dev, batch * n, "moe combine rows") }?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK).max(1), batch as u32, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: (const float*, const int*, const float*, float*, int, int)
    // matched positionally; every `pair_of` entry is a row of `y` (the
    // caller's routing), `w` and `pair_of` hold `batch * k`, `out`
    // `batch * n`.
    unsafe {
        func.launch(cfg, (y, pair_of, w, &mut out, k as i32, n as i32))
            .map_err(|e| launch_err("moe_combine_rows", e))?;
    }
    Ok(out)
}

/// `w[rows][cols] . x[t]` for each of `batch` rows of `x`, f32, the
/// router's logits, `[batch][rows]`.
pub(crate) fn enqueue_router_f32(
    dev: &Arc<CudaDevice>,
    w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
    batch: usize,
) -> Result<CudaSlice<f32>, CudaError> {
    debug_assert!(w.len() >= rows * cols && x.len() >= batch * cols);
    let func = func_for(dev, ROUTER_F32_KERNEL_SRC, ROUTER_F32)?;
    // SAFETY: one warp per row over a grid of `ceil(rows / 8)` blocks of
    // eight warps, a grid row per token, and every warp below `rows`
    // writes its token's `out[row]`.
    let mut out = unsafe { crate::gpu::alloc_out(dev, batch * rows, "router logits") }?;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(8).max(1), batch as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: (const float*, const float*, float*, int, int) matched
    // positionally; `w` holds `rows * cols` and `x` `batch * cols` (the
    // debug assert, and the stack's shape check), `out` `batch * rows`.
    unsafe {
        func.launch(cfg, (w, x, &mut out, rows as i32, cols as i32))
            .map_err(|e| launch_err("router_f32_matvec", e))?;
    }
    Ok(out)
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
/// block's Q rows, the K tile (rows padded by 4 floats) and the V tile.
pub(crate) fn prefill_attn_smem(head_dim: usize, rpw: usize, bc: usize) -> usize {
    4 * (4 * rpw * head_dim + bc * (head_dim + 4) + bc * head_dim)
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
    sinks: Option<&CudaSlice<f32>>,
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
    if n_q >= 1 << 15 {
        return Err(CudaError::Unsupported(format!(
            "causal_gqa_prefill: a batch of {n_q} rows (the launch packs it in 15 bits)"
        )));
    }
    debug_assert!(sinks.is_none_or(|s| s.len() >= n_heads));
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
    // SAFETY: (const float* x3, float*, int x2, const float*, int x2,
    // float, float, int) matched positionally, the shape, the head
    // counts and the tile packed as the kernel unpacks them (`n_q` below
    // 2^15, the rest below 2^16 and 2^8); the sinks pointer is null or
    // `n_heads` logits (the debug assert, and the stack's shape check);
    // the grid covers `n_q x n_heads`, every key
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
                (head_dim | (n_q << 16)) as i32,
                (n_heads | (n_kv_heads << 16)) as i32,
                sink_ptr(sinks),
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

/// A kernel's nullable `const float*` argument: the slice's device
/// address, or null. The caller keeps the slice alive until the stream
/// has run the launch (the stack holds it until its one wait).
fn sink_ptr(s: Option<&CudaSlice<f32>>) -> u64 {
    use cudarc::driver::DevicePtr;
    s.map_or(0, |s| *s.device_ptr())
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
    sinks: Option<&CudaSlice<f32>>,
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
    debug_assert!(sinks.is_none_or(|s| s.len() >= n_heads));
    // SAFETY: (const float*, float*, int x3, const float*) matched
    // positionally; the sinks pointer is null or `n_heads` logits (the
    // debug assert, and the stack's shape check);
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
                    sink_ptr(sinks),
                ),
            )
            .map_err(|e| launch_err("decode_attn_combine", e))?;
    }
    Ok(out)
}
