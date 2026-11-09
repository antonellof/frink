//! The routed (MoE) FFN of a resident stack layer.
//!
//! The router runs on the device and its logits come back to the
//! caller's routing, the host's own (`RouteFn`), so every gating rule
//! routes as it does on the CPU. A decode token's experts then run as
//! five launches, the matvecs indexed by expert; a prefill batch's
//! tokens are grouped by expert and each projection of every chosen
//! expert is one grouped tensor-core GEMM. Either way each token's
//! experts are summed from zero in its routing's order, as the host's
//! `combine_expert_outputs` sums them, before the residual add.

use super::enqueue::{
    enqueue_add_bias_tab_rows, enqueue_axpy, enqueue_moe_combine, enqueue_moe_combine_rows,
    enqueue_router_f32,
};
use super::{
    run_expert, DeviceRouting, ExpertBiasCuda, ExpertProjCuda, Held, ProjWeights, RouteFn,
    RouterCuda,
};
use crate::gpu::{glu_mul_device, shared_device, CudaError, GluCuda, MatvecInput};
use crate::xfer;
use cudarc::driver::{CudaDevice, CudaSlice, CudaViewMut};
use std::sync::Arc;

/// A routed layer's FFN over `normed2` (`batch` rows): the summed,
/// weighted experts of each token, `[batch][hidden_dim]`, ready for the
/// residual add. A decode token's router and every expert's gate and up
/// read `normed2` through the one input, so its q8_1 copy (dp4a) is
/// made once for all of them.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_routed(
    dev: &Arc<CudaDevice>,
    held: &mut Held,
    router: RouterCuda<'_>,
    glu: GluCuda,
    normed2: &MatvecInput<'_>,
    batch: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    index: usize,
    route: &mut RouteFn<'_, '_>,
    device: Option<(DeviceRouting<'_>, CudaViewMut<'_, i32>)>,
) -> Result<(CudaSlice<f32>, bool), CudaError> {
    if batch > 1 {
        // A batch runs the GEMMs, which read the f32 rows.
        let x = normed2.x();
        if let Some((spec, mut ids)) = device {
            if let Some(out) = super::moe_device::run_device_routed_batch(
                dev, held, router, glu, x, batch, hidden_dim, ffn_dim, &spec, &mut ids,
            )? {
                return Ok((out, true));
            }
        }
        let out = run_routed_batch(
            dev, held, router, glu, x, batch, hidden_dim, ffn_dim, index, route,
        )?;
        return Ok((out, false));
    }
    if let Some((spec, mut ids)) = device {
        if let Some(out) = super::moe_device::run_device_routed(
            dev, held, router, glu, normed2, hidden_dim, ffn_dim, &spec, &mut ids,
        )? {
            return Ok((out, true));
        }
    }
    // The layer's one host wait: the router's logits, for the caller's
    // routing.
    let logits = match router {
        RouterCuda::F32 { data, rows } => {
            let w = held.vector(dev, data)?;
            enqueue_router_f32(dev, &w.slice, normed2.x(), rows, hidden_dim, 1)?
        }
        RouterCuda::Quant(p) => held.proj(dev, &p, normed2, 1)?,
    };
    let logits = xfer::dtoh(dev, "router logits", &logits)?;
    let picks = route(index, &logits)?;
    for (ex, _) in &picks {
        check_expert_shapes(ex, hidden_dim, ffn_dim)?;
    }
    // All the experts' gate, up and down in one launch each when their
    // kinds have a coalesced matvec; one expert at a time otherwise.
    if let Some(sum) = run_experts_together(dev, held, &picks, glu, normed2, hidden_dim, ffn_dim)? {
        return Ok((sum, false));
    }
    let mut sum = dev
        .alloc_zeros::<f32>(hidden_dim)
        .map_err(|e| CudaError::Launch(format!("MoE sum alloc: {e:?}")))?;
    for (ex, w) in &picks {
        let y = run_expert(dev, held, ex, glu, normed2, 1, ffn_dim)?;
        enqueue_axpy(dev, &mut sum, &y, *w, hidden_dim)?;
    }
    Ok((sum, false))
}

/// Whether a routed layer can ride the stack in a prefill batch on this
/// device: its experts run through the grouped tensor-core GEMM, which
/// needs `sm_80`. A decode token's routed layer needs only the matvecs.
/// Probed once: the decoder asks for every layer of every step.
pub fn routed_batch_supported() -> bool {
    static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        shared_device().is_ok_and(|dev| {
            crate::mul_mm_launch::MulMmBody::select(&dev)
                == crate::mul_mm_launch::MulMmBody::TensorCore
        })
    })
}

/// Which projection of an expert.
#[derive(Clone, Copy)]
enum Proj {
    Gate,
    Up,
    Down,
}

impl Proj {
    fn of<'p, 'a>(self, e: &'p ExpertProjCuda<'a>) -> &'p ProjWeights<'a> {
        match self {
            Proj::Gate => &e.gate,
            Proj::Up => &e.up,
            Proj::Down => &e.down,
        }
    }
}

/// A routed layer over a prefill batch. The router runs over every row,
/// its logits come back once, and the caller routes each token. The
/// tokens' expert assignments ("pairs") are grouped by expert, and each
/// projection of every chosen expert is ONE grouped tensor-core GEMM
/// (`mul_mm_launch::enqueue_mul_mm_grouped`), gate and up reading each
/// pair's token row and down reading its pair's activation. Each token's
/// output is then the sum of its pairs' outputs from zero in its
/// routing's order, as `combine_expert_outputs` sums them.
#[allow(clippy::too_many_arguments)]
fn run_routed_batch(
    dev: &Arc<CudaDevice>,
    held: &mut Held,
    router: RouterCuda<'_>,
    glu: GluCuda,
    normed2: &CudaSlice<f32>,
    batch: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    index: usize,
    route: &mut RouteFn<'_, '_>,
) -> Result<CudaSlice<f32>, CudaError> {
    let n_experts = router.rows();
    let logits = match router {
        RouterCuda::F32 { data, rows } => {
            let w = held.vector(dev, data)?;
            enqueue_router_f32(dev, &w.slice, normed2, rows, hidden_dim, batch)?
        }
        RouterCuda::Quant(p) => held.proj(dev, &p, &MatvecInput::new(normed2), batch)?,
    };
    // The layer's one host wait.
    let logits = xfer::dtoh(dev, "router logits", &logits)?;
    // Route every token, and group its picks by expert: an expert is
    // identified by its gate's bytes.
    let mut experts: Vec<ExpertProjCuda<'_>> = Vec::new();
    let mut members: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut weights = Vec::new();
    let mut k = None;
    for t in 0..batch {
        let picks = route(index, &logits[t * n_experts..(t + 1) * n_experts])?;
        if *k.get_or_insert(picks.len()) != picks.len() || picks.is_empty() {
            return Err(CudaError::Unsupported(
                "resident stack: tokens of a batch routed to different expert counts".to_string(),
            ));
        }
        for (slot, (ex, w)) in picks.into_iter().enumerate() {
            let key = ex.gate.gemm.data.as_ptr();
            let g = match experts
                .iter()
                .position(|e| e.gate.gemm.data.as_ptr() == key)
            {
                Some(g) => g,
                None => {
                    check_expert_shapes(&ex, hidden_dim, ffn_dim)?;
                    experts.push(ex);
                    members.push(Vec::new());
                    experts.len() - 1
                }
            };
            members[g].push((t, slot));
            weights.push(w);
        }
    }
    let k = k.unwrap_or(0);
    let mut group_off = Vec::with_capacity(experts.len() + 1);
    let mut gather = Vec::with_capacity(batch * k);
    let mut pair_of = vec![0i32; batch * k];
    group_off.push(0i32);
    for m in &members {
        for &(t, slot) in m {
            pair_of[t * k + slot] = gather.len() as i32;
            gather.push(t as i32);
        }
        group_off.push(gather.len() as i32);
    }
    let one_kind = |which: Proj| {
        let first = which.of(&experts[0]);
        experts.iter().all(|e| {
            let p = which.of(e);
            std::ptr::eq(p.kind(), first.kind()) && p.gemm.row_bytes == first.gemm.row_bytes
        })
    };
    if !one_kind(Proj::Gate) || !one_kind(Proj::Up) || !one_kind(Proj::Down) {
        return Err(CudaError::Unsupported(
            "resident stack: a layer's experts in more than one kind".to_string(),
        ));
    }
    // Each projection of the chosen experts: their weights through a
    // table, on the int8 grouped GEMM where the kind has one
    // (`moe_device::grouped_proj`, the device-routed path's choice too).
    let table_of = |held: &mut Held,
                    which: Proj|
     -> Result<
        (CudaSlice<u64>, crate::mul_mm_launch::GroupedShape<'static>),
        CudaError,
    > {
        let p0 = which.of(&experts[0]);
        let mut ptrs = Vec::with_capacity(experts.len());
        for e in &experts {
            let w = crate::gpu::resident_cuda_weights(dev, which.of(e).gemm.data)?;
            ptrs.push(*cudarc::driver::DevicePtr::device_ptr(&w.slice));
            held.weights.push(w);
        }
        let shape = crate::mul_mm_launch::GroupedShape {
            kind: p0.kind(),
            n_rows: p0.rows(),
            n_cols: p0.cols(),
            row_bytes: p0.gemm.row_bytes,
        };
        Ok((xfer::htod(dev, "grouped table", &ptrs)?, shape))
    };
    let pairs = gather.len();
    // Each pair's bias rows are its expert's, through the tables.
    let ex_bias: Vec<_> = experts.iter().map(|e| e.bias).collect();
    let biases = match bias_tables(dev, held, &ex_bias)? {
        Some(tabs) => {
            let pair_group: Vec<i32> = group_off
                .windows(2)
                .enumerate()
                .flat_map(|(g, w)| std::iter::repeat_n(g as i32, (w[1] - w[0]) as usize))
                .collect();
            let [gb, ub, db] = tabs;
            Some((
                (gb, ub, db),
                xfer::htod(dev, "routed pair experts", &pair_group)?,
            ))
        }
        None => None,
    };
    let n_groups = experts.len();
    let d_off = xfer::htod(dev, "routed offsets", &group_off)?;
    let d_gather = xfer::htod(dev, "routed gather", &gather)?;
    let (gt, gs) = table_of(held, Proj::Gate)?;
    let (ut, us) = table_of(held, Proj::Up)?;
    let (dt, ds) = table_of(held, Proj::Down)?;
    let gp = super::moe_device::grouped_proj;
    let mut g = gp(
        dev,
        &gs,
        &gt.slice(..),
        n_groups,
        &d_off,
        Some(&d_gather),
        normed2,
        batch,
        pairs,
    )?;
    let mut u = gp(
        dev,
        &us,
        &ut.slice(..),
        n_groups,
        &d_off,
        Some(&d_gather),
        normed2,
        batch,
        pairs,
    )?;
    if let Some(((gb, ub, _), pg)) = &biases {
        enqueue_add_bias_tab_rows(dev, &mut g, &gb.slice(..), Some(pg), pairs, ffn_dim)?;
        enqueue_add_bias_tab_rows(dev, &mut u, &ub.slice(..), Some(pg), pairs, ffn_dim)?;
    }
    let act = glu_mul_device(dev, glu, &g, &u, pairs * ffn_dim)?;
    drop((g, u));
    let mut d = gp(
        dev,
        &ds,
        &dt.slice(..),
        n_groups,
        &d_off,
        None,
        &act,
        batch,
        pairs,
    )?;
    drop(act);
    if let Some(((_, _, db), pg)) = &biases {
        enqueue_add_bias_tab_rows(dev, &mut d, &db.slice(..), Some(pg), pairs, hidden_dim)?;
    }
    let d_pair_of = xfer::htod(dev, "routed pairs", &pair_of)?;
    // `weights` is token-major in slot order, the order of `pair_of`.
    let d_w = xfer::htod(dev, "routed weights", &weights)?;
    enqueue_moe_combine_rows(dev, &d, &d_pair_of, &d_w, k, hidden_dim, batch)
}

/// The device bias tables of a set of routed experts, one entry per
/// expert in order: the gate's, up's and down's biases. `None` when no
/// expert carries biases. Refused when some do and some do not, which no
/// checkpoint does (gpt-oss biases every expert).
fn bias_tables(
    dev: &Arc<CudaDevice>,
    held: &mut Held,
    biases: &[Option<ExpertBiasCuda<'_>>],
) -> Result<Option<[CudaSlice<u64>; 3]>, CudaError> {
    match biases.iter().filter(|b| b.is_some()).count() {
        0 => return Ok(None),
        n if n == biases.len() => {}
        _ => {
            return Err(CudaError::Unsupported(
                "resident stack: a layer's experts with and without biases".to_string(),
            ))
        }
    }
    let mut table = |which: usize| -> Result<CudaSlice<u64>, CudaError> {
        let ptrs = biases
            .iter()
            .flatten()
            .map(|b| {
                let v = held.vector(dev, [b.gate, b.up, b.down][which])?;
                Ok(*cudarc::driver::DevicePtr::device_ptr(&v.slice))
            })
            .collect::<Result<Vec<u64>, CudaError>>()?;
        xfer::htod(dev, "routed bias table", &ptrs)
    };
    Ok(Some([table(0)?, table(1)?, table(2)?]))
}

/// A token's routed experts as five launches: every expert's gate, every
/// expert's up, the gated activation over all of them, every expert's
/// down, and the weighted combine, with three bias launches between
/// them for experts that carry biases. `None`, before any launch, when the
/// experts' projections do not share a coalesced kind
/// (`gpu::enqueue_matvec_multi`), and the caller runs them one by one.
fn run_experts_together(
    dev: &Arc<CudaDevice>,
    held: &mut Held,
    picks: &[(ExpertProjCuda<'_>, f32)],
    glu: GluCuda,
    x: &MatvecInput<'_>,
    hidden_dim: usize,
    ffn_dim: usize,
) -> Result<Option<CudaSlice<f32>>, CudaError> {
    let k = picks.len();
    let gate: Option<Vec<_>> = picks.iter().map(|(ex, _)| ex.gate.matvec).collect();
    let up: Option<Vec<_>> = picks.iter().map(|(ex, _)| ex.up.matvec).collect();
    let down: Option<Vec<_>> = picks.iter().map(|(ex, _)| ex.down.matvec).collect();
    let (Some(gate), Some(up), Some(down)) = (gate, up, down) else {
        return Ok(None);
    };
    let same = |ls: &[crate::gpu::MatvecLaunch<'_>]| {
        ls.windows(2).all(|w| w[0].fn_name == w[1].fn_name)
            && crate::matvec_kinds::coalesced::coalesced_kind(ls[0].fn_name).is_some()
    };
    if k == 0 || !same(&gate) || !same(&up) || !same(&down) {
        return Ok(None);
    }
    // Biased experts (gpt-oss): slot `j`'s rows gain expert `j`'s biases.
    let ex_bias: Vec<_> = picks.iter().map(|(ex, _)| ex.bias).collect();
    let biases = bias_tables(dev, held, &ex_bias)?;
    let (mut g, wg) = crate::gpu::enqueue_matvec_multi(dev, &gate, x, 0)?;
    let (mut u, wu) = crate::gpu::enqueue_matvec_multi(dev, &up, x, 0)?;
    if let Some([gb, ub, _]) = &biases {
        enqueue_add_bias_tab_rows(dev, &mut g, &gb.slice(..), None, k, ffn_dim)?;
        enqueue_add_bias_tab_rows(dev, &mut u, &ub.slice(..), None, k, ffn_dim)?;
    }
    let act = glu_mul_device(dev, glu, &g, &u, k * ffn_dim)?;
    drop((g, u));
    let (mut d, wd) =
        crate::gpu::enqueue_matvec_multi(dev, &down, &MatvecInput::new(&act), ffn_dim)?;
    drop(act);
    if let Some([_, _, db]) = &biases {
        enqueue_add_bias_tab_rows(dev, &mut d, &db.slice(..), None, k, hidden_dim)?;
    }
    held.weights.extend(wg.into_iter().chain(wu).chain(wd));
    let weights: Vec<f32> = picks.iter().map(|(_, w)| *w).collect();
    let d_w = xfer::htod(dev, "routed weights", &weights)?;
    Ok(Some(enqueue_moe_combine(dev, &d, &d_w, k, hidden_dim)?))
}

/// A routed expert's shapes against its layer, and the one-row view
/// every projection needs, before it touches the device.
pub(super) fn check_expert_shapes(
    ex: &ExpertProjCuda<'_>,
    hidden_dim: usize,
    ffn_dim: usize,
) -> Result<(), CudaError> {
    if ex.gate.rows() != ffn_dim
        || ex.up.rows() != ffn_dim
        || ex.gate.cols() != hidden_dim
        || ex.up.cols() != hidden_dim
        || ex.down.rows() != hidden_dim
        || ex.down.cols() != ffn_dim
    {
        return Err(CudaError::Unsupported(
            "resident stack: a routed expert's shapes do not agree with its layer".to_string(),
        ));
    }
    for m in [&ex.gate, &ex.up, &ex.down] {
        m.validate(1)?;
    }
    if let Some(b) = ex.bias {
        if b.gate.len() != ffn_dim || b.up.len() != ffn_dim || b.down.len() != hidden_dim {
            return Err(CudaError::Unsupported(
                "resident stack: a routed expert's biases do not agree with its layer".to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
