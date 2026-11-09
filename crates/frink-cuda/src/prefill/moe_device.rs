//! A routed layer's decode token routed on the device.
//!
//! The host-routed decode (`moe::run_routed`) downloads each routed
//! layer's logits and waits for the host to pick the experts: on an
//! L40S that left a Qwen3-30B-A3B token at 9.0 ms, 7.5 ms of it the host
//! enqueueing between 48 such waits while the GPU sat idle. Here the
//! top-k and the weights are computed on the device
//! (`kernels::MOE_TOPK_KERNEL_SRC`, the host's rules for the gating it
//! covers), each chosen expert's weight pointers are gathered from a
//! per-layer table of all of them, and the existing routed matvecs read
//! those, so a token makes no wait until the stack's one download. The
//! chosen ids come back with it, for the host's activation counts.
//!
//! A layer this cannot serve (an expert kind with no coalesced matvec,
//! too many experts, a gating rule the kernel does not have) is
//! recorded as such and keeps the host routing.

use super::enqueue::{
    enqueue_add_bias_tab_rows, enqueue_gather_ptrs, enqueue_moe_combine, enqueue_moe_combine_rows,
    enqueue_moe_group, enqueue_moe_topk, enqueue_router_f32,
};
use super::{ExpertProjCuda, Held, RouterCuda};
use crate::gpu::{glu_mul_device, CudaError, GluCuda, MultiShape};
use crate::mul_mm_launch::GroupedShape;
use crate::resident::ResidentCuda;
use cudarc::driver::{CudaDevice, CudaSlice, CudaViewMut, DevicePtr};
use frink_quant::fingerprint::WeightFingerprint;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The gating rules the device routes by, each the host's function of
/// the same name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceGating {
    /// `route_top_k_softmax`, the chosen renormalised when `norm`.
    Softmax { norm: bool },
    /// `route_top_k_softmax_weight` (gpt-oss).
    SoftmaxWeight,
}

impl DeviceGating {
    fn mode(self) -> i32 {
        match self {
            DeviceGating::Softmax { norm: false } => 0,
            DeviceGating::Softmax { norm: true } => 1,
            DeviceGating::SoftmaxWeight => 2,
        }
    }
}

/// Where a layer's experts come from when its tables are not yet on the
/// device: the decoder, asked once per layer for the process.
pub trait ExpertSource: Sync {
    /// Every expert of layer `layer` in id order, with its biases. Fewer
    /// than the layer has (one without a matvec) declines the layer.
    fn experts(&self, layer: usize) -> Vec<ExpertProjCuda<'_>>;
}

/// `FRINK_CUDA_DEVICE_ROUTE=0|off|false` keeps every decode token's
/// routing on the host, for an A/B. Read once.
pub fn device_routing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FRINK_CUDA_DEVICE_ROUTE").as_deref(),
            Ok("0" | "off" | "false")
        )
    })
}

/// What a routed layer needs to be routed on the device at a decode
/// token.
#[derive(Clone, Copy)]
pub struct DeviceRouting<'a> {
    /// Identifies the layer's experts across calls: any stable address
    /// of the layer's (the decoder passes its experts' slice).
    pub key: usize,
    /// The first expert's gate and the last expert's down bytes, held
    /// against the cached tables on every hit (`resident.rs`'s check).
    pub probe: [&'a [u8]; 2],
    pub n_experts: usize,
    pub k: usize,
    pub gating: DeviceGating,
    /// gpt-oss's router bias, added to the logits before the top-k.
    pub router_bias: Option<&'a [f32]>,
    /// The layer's experts, asked only when its tables are not cached.
    pub source: &'a dyn ExpertSource,
    pub layer: usize,
}

/// A layer's experts on the device: one table of every expert's weight
/// pointers per projection (and per bias, for biased experts), and the
/// shapes the routed matvecs launch at.
struct ExpertTables {
    /// `[gate | up | down (| gate bias | up bias | down bias)]`, each
    /// `n` pointers.
    tabs: CudaSlice<u64>,
    n: usize,
    biased: bool,
    shapes: [MultiShape; 3],
    /// The grouped GEMM's shapes, when every expert's GEMM bytes are the
    /// bytes its matvec table points at (a prefill batch's path).
    gemm: Option<[GroupedShape<'static>; 3]>,
    probe: [WeightFingerprint; 2],
    // The uploads the pointers point into, alive with the tables.
    _weights: Vec<Arc<crate::gpu::ResidentCudaWeights>>,
    _vectors: Vec<Arc<ResidentCuda<f32>>>,
}

// SAFETY: the slices live on the process-wide device and are only read
// by kernels, as `ResidentCuda`'s are.
unsafe impl Send for ExpertTables {}
unsafe impl Sync for ExpertTables {}

/// `None` records a layer the device cannot route, so it is not tried
/// again every token.
type TableMap = HashMap<usize, Option<Arc<ExpertTables>>>;
static TABLES: Mutex<Option<TableMap>> = Mutex::new(None);

/// The layer's tables, built on first use, or `None` when the device
/// cannot route it.
fn tables_for(
    dev: &Arc<CudaDevice>,
    spec: &DeviceRouting<'_>,
) -> Result<Option<Arc<ExpertTables>>, CudaError> {
    let probe = spec.probe.map(WeightFingerprint::of);
    let mut guard = TABLES.lock().unwrap_or_else(|p| p.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    match map.get(&spec.key) {
        Some(Some(t)) if t.probe == probe && t.n == spec.n_experts => return Ok(Some(t.clone())),
        Some(None) => return Ok(None),
        _ => {}
    }
    let built = build_tables(dev, spec, probe)?;
    map.insert(spec.key, built.clone());
    Ok(built)
}

/// Projection `which` (gate, up, down) of an expert, as its matvec.
fn pick_matvec<'a>(e: &ExpertProjCuda<'a>, which: usize) -> Option<crate::gpu::MatvecLaunch<'a>> {
    [e.gate, e.up, e.down][which].matvec
}

fn build_tables(
    dev: &Arc<CudaDevice>,
    spec: &DeviceRouting<'_>,
    probe: [WeightFingerprint; 2],
) -> Result<Option<Arc<ExpertTables>>, CudaError> {
    let experts = spec.source.experts(spec.layer);
    let n = experts.len();
    if n != spec.n_experts || n == 0 || n > 256 || spec.k > 16 || spec.k > n {
        return Ok(None);
    }
    let biased = experts[0].bias.is_some();
    if experts.iter().any(|e| e.bias.is_some() != biased) {
        return Ok(None);
    }
    // One coalesced kind and shape per projection, across every expert.
    let mut shapes = Vec::with_capacity(3);
    for which in 0..3 {
        let Some(first) = pick_matvec(&experts[0], which) else {
            return Ok(None);
        };
        let shape = MultiShape::of(&first);
        if crate::matvec_kinds::coalesced::coalesced_kind(shape.fn_name).is_none() {
            return Ok(None);
        }
        let all_same = experts.iter().all(|e| {
            pick_matvec(e, which).is_some_and(|l| {
                l.fn_name == shape.fn_name
                    && l.rows == shape.rows
                    && l.row_bytes == shape.row_bytes
                    && l.n_blocks_per_row == shape.n_blocks_per_row
                    && l.weights.len() == shape.rows * shape.row_bytes
            })
        });
        if !all_same {
            return Ok(None);
        }
        shapes.push(shape);
    }
    let mut weights = Vec::with_capacity(3 * n);
    let mut vectors = Vec::new();
    let mut ptrs = vec![0u64; if biased { 6 } else { 3 } * n];
    for (e, ex) in experts.iter().enumerate() {
        for (which, p) in [ex.gate, ex.up, ex.down].iter().enumerate() {
            let w =
                crate::gpu::resident_cuda_weights(dev, p.matvec.expect("checked above").weights)?;
            ptrs[which * n + e] = *w.slice.device_ptr();
            weights.push(w);
        }
        if let Some(b) = ex.bias {
            for (which, v) in [b.gate, b.up, b.down].iter().enumerate() {
                let r = crate::resident::resident_cuda_f32(dev, v)?;
                ptrs[(3 + which) * n + e] = *r.slice.device_ptr();
                vectors.push(r);
            }
        }
    }
    let tabs = crate::xfer::htod(dev, "expert tables", &ptrs)?;
    let gemm_of = |which: usize| -> Option<GroupedShape<'static>> {
        let g0 = [experts[0].gate, experts[0].up, experts[0].down][which].gemm;
        let same = experts.iter().all(|e| {
            let p = [e.gate, e.up, e.down][which];
            let g = p.gemm;
            std::ptr::eq(g.kind, g0.kind)
                && g.rows == g0.rows
                && g.cols == g0.cols
                && g.row_bytes == g0.row_bytes
                && p.matvec
                    .is_some_and(|m| std::ptr::eq(m.weights.as_ptr(), g.data.as_ptr()))
        });
        same.then_some(GroupedShape {
            kind: g0.kind,
            n_rows: g0.rows,
            n_cols: g0.cols,
            row_bytes: g0.row_bytes,
        })
    };
    let gemm = match (gemm_of(0), gemm_of(1), gemm_of(2)) {
        (Some(g), Some(u), Some(d)) => Some([g, u, d]),
        _ => None,
    };
    Ok(Some(Arc::new(ExpertTables {
        tabs,
        n,
        biased,
        shapes: [shapes[0], shapes[1], shapes[2]],
        gemm,
        probe,
        _weights: weights,
        _vectors: vectors,
    })))
}

/// A decode token's routed FFN over `normed2` with the routing on the
/// device: the chosen ids written to `ids` (`k` of them), the summed,
/// weighted experts returned. `None`, before any launch, when the layer
/// is one the device cannot route, and the caller routes on the host.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_device_routed(
    dev: &Arc<CudaDevice>,
    held: &mut Held,
    router: RouterCuda<'_>,
    glu: GluCuda,
    normed2: &CudaSlice<f32>,
    hidden_dim: usize,
    ffn_dim: usize,
    spec: &DeviceRouting<'_>,
    ids: &mut CudaViewMut<'_, i32>,
) -> Result<Option<CudaSlice<f32>>, CudaError> {
    let Some(tables) = tables_for(dev, spec)? else {
        return Ok(None);
    };
    let [gs, us, ds] = tables.shapes;
    if gs.rows != ffn_dim || us.rows != ffn_dim || ds.rows != hidden_dim {
        return Ok(None);
    }
    let n = tables.n;
    let k = spec.k;
    let logits = match router {
        RouterCuda::F32 { data, rows } => {
            let w = held.vector(dev, data)?;
            enqueue_router_f32(dev, &w.slice, normed2, rows, hidden_dim, 1)?
        }
        RouterCuda::Quant(p) => held.proj(dev, &p, normed2, 1)?,
    };
    let bias = match spec.router_bias {
        Some(b) => Some(held.vector(dev, b)?),
        None => None,
    };
    let w = enqueue_moe_topk(
        dev,
        &logits,
        bias.as_ref().map(|b| &b.slice),
        n,
        k,
        spec.gating.mode(),
        1,
        ids,
    )?;
    let n_tabs = if tables.biased { 6 } else { 3 };
    let slot = enqueue_gather_ptrs(dev, &tables.tabs, &ids.slice(..k), n, n_tabs, k)?;
    let tab = |t: usize| slot.slice(t * k..(t + 1) * k);
    // SAFETY (all three): each slot table holds `k` pointers gathered from
    // the layer's tables, whose uploads the tables keep alive and the
    // stack holds below until its wait; the shapes were checked equal for
    // every expert when the tables were built, and the activations are
    // one row (gate, up) or `k` rows of `ffn_dim` (down).
    let mut g =
        unsafe { crate::gpu::enqueue_matvec_multi_table(dev, &gs, &tab(0), k, normed2, 0) }?;
    let mut u =
        unsafe { crate::gpu::enqueue_matvec_multi_table(dev, &us, &tab(1), k, normed2, 0) }?;
    if tables.biased {
        enqueue_add_bias_tab_rows(dev, &mut g, &tab(3), None, k, ffn_dim)?;
        enqueue_add_bias_tab_rows(dev, &mut u, &tab(4), None, k, ffn_dim)?;
    }
    let act = glu_mul_device(dev, glu, &g, &u, k * ffn_dim)?;
    drop((g, u));
    let mut d =
        unsafe { crate::gpu::enqueue_matvec_multi_table(dev, &ds, &tab(2), k, &act, ffn_dim) }?;
    drop(act);
    if tables.biased {
        enqueue_add_bias_tab_rows(dev, &mut d, &tab(5), None, k, hidden_dim)?;
    }
    held.keep.push(tables);
    Ok(Some(enqueue_moe_combine(dev, &d, &w, k, hidden_dim)?))
}

/// A routed prefill batch with the routing and the grouping on the
/// device: the router over every row, the top-k of every token, its
/// pairs grouped by expert, each projection one grouped GEMM over every
/// expert of the layer (an empty one exits at once), and each token's
/// experts summed in its routing's order. No host wait; the chosen ids
/// (`batch * k` of them, token-major) are written to `ids`. `None`,
/// before any launch, for a layer the device cannot route this way.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_device_routed_batch(
    dev: &Arc<CudaDevice>,
    held: &mut Held,
    router: RouterCuda<'_>,
    glu: GluCuda,
    normed2: &CudaSlice<f32>,
    batch: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    spec: &DeviceRouting<'_>,
    ids: &mut CudaViewMut<'_, i32>,
) -> Result<Option<CudaSlice<f32>>, CudaError> {
    if !super::moe::routed_batch_supported() {
        return Ok(None);
    }
    let Some(tables) = tables_for(dev, spec)? else {
        return Ok(None);
    };
    let Some([gs, us, ds]) = tables.gemm else {
        return Ok(None);
    };
    if gs.n_rows != ffn_dim || us.n_rows != ffn_dim || ds.n_rows != hidden_dim {
        return Ok(None);
    }
    let (n, k) = (tables.n, spec.k);
    let pairs = batch * k;
    let logits = match router {
        RouterCuda::F32 { data, rows } => {
            let w = held.vector(dev, data)?;
            enqueue_router_f32(dev, &w.slice, normed2, rows, hidden_dim, batch)?
        }
        RouterCuda::Quant(p) => held.proj(dev, &p, normed2, batch)?,
    };
    let bias = match spec.router_bias {
        Some(b) => Some(held.vector(dev, b)?),
        None => None,
    };
    let w = enqueue_moe_topk(
        dev,
        &logits,
        bias.as_ref().map(|b| &b.slice),
        n,
        k,
        spec.gating.mode(),
        batch,
        ids,
    )?;
    drop(logits);
    let groups = enqueue_moe_group(dev, &ids.slice(..pairs), pairs, k, n)?;
    let tab = |t: usize| tables.tabs.slice(t * n..(t + 1) * n);
    // An expert takes a token at most once, so no group exceeds `batch`.
    let x16 = crate::mul_mm_launch::f32_to_f16(dev, normed2, batch * hidden_dim)?;
    // SAFETY (all three): the tables hold every expert's live weights of
    // the checked shape, kept alive by the tables the stack holds below;
    // the offsets end at `pairs` and no group passes `batch`; the gather
    // names token rows of `x16`, and the down projection reads the pairs'
    // own activation rows.
    let mut g = unsafe {
        crate::mul_mm_launch::enqueue_mul_mm_grouped_dev(
            dev,
            &gs,
            &tab(0),
            n,
            &groups.group_off,
            Some(&groups.gather),
            &x16,
            pairs,
            batch,
        )
    }?;
    let mut u = unsafe {
        crate::mul_mm_launch::enqueue_mul_mm_grouped_dev(
            dev,
            &us,
            &tab(1),
            n,
            &groups.group_off,
            Some(&groups.gather),
            &x16,
            pairs,
            batch,
        )
    }?;
    drop(x16);
    if tables.biased {
        enqueue_add_bias_tab_rows(
            dev,
            &mut g,
            &tab(3),
            Some(&groups.pair_group),
            pairs,
            ffn_dim,
        )?;
        enqueue_add_bias_tab_rows(
            dev,
            &mut u,
            &tab(4),
            Some(&groups.pair_group),
            pairs,
            ffn_dim,
        )?;
    }
    let act = glu_mul_device(dev, glu, &g, &u, pairs * ffn_dim)?;
    drop((g, u));
    let act16 = crate::mul_mm_launch::f32_to_f16(dev, &act, pairs * ffn_dim)?;
    drop(act);
    let mut d = unsafe {
        crate::mul_mm_launch::enqueue_mul_mm_grouped_dev(
            dev,
            &ds,
            &tab(2),
            n,
            &groups.group_off,
            None,
            &act16,
            pairs,
            batch,
        )
    }?;
    drop(act16);
    if tables.biased {
        enqueue_add_bias_tab_rows(
            dev,
            &mut d,
            &tab(5),
            Some(&groups.pair_group),
            pairs,
            hidden_dim,
        )?;
    }
    let out = enqueue_moe_combine_rows(dev, &d, &groups.pair_of, &w, k, hidden_dim, batch)?;
    held.keep.push(tables);
    Ok(Some(out))
}
