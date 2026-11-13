//! The device side of [`crate::moe_id`]: one routed SwiGLU step for
//! every token of a layer, as ggml's `build_moe_ffn` runs it on CUDA:
//! `mul_mat_id(gate)`, `mul_mat_id(up)`, `silu * up`, `mul_mat_id(down)`.
//!
//! Before this a CUDA MoE layer ran each routed expert as its own fused
//! FFN launch (`launch_dense_ffn_swiglu`): an upload, three matvecs and
//! a synchronous download per expert, so `top_k` round trips per token
//! per layer. Here the whole step is one activation upload, one ids
//! upload (plus the two row maps for `mul_mm_id`), four kernels and ONE
//! download, whatever `top_k` is.
//!
//! The planes go resident through the same pointer-keyed cache every
//! weight uses (`resident_cuda_weights`), and every copy and wait goes
//! through [`crate::xfer`], so `FRINK_CUDA_COUNT` sees them.
//!
//! # UNVERIFIED ON HARDWARE
//!
//! Never run on a GPU; the hardware test below is owed.

use crate::gpu::{
    ensure_module_loaded_lazy, resident_cuda_weights, shared_device, silu_mul_device, CudaError,
    ResidentCudaWeights,
};
use crate::moe_id::{
    mm_id_names, mm_id_src, mv_id_names, mv_id_src, IdMap, IdPlane, MM_ID_MIN_ROWS,
};
use crate::mul_mm::{BM, BN, THREADS};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};
use std::sync::Arc;

/// One MoE layer's routed experts as three contiguous planes.
#[derive(Debug, Clone, Copy)]
pub struct MoePlanesCuda<'a> {
    /// `[n_experts][ffn][hidden]`.
    pub gate: IdPlane<'a>,
    /// `[n_experts][ffn][hidden]`.
    pub up: IdPlane<'a>,
    /// `[n_experts][hidden][ffn]`.
    pub down: IdPlane<'a>,
}

impl MoePlanesCuda<'_> {
    /// The three planes agree with each other and with `ids`.
    pub fn validate(&self, ids: &[u32], n_tokens: usize, top_k: usize) -> Result<(), CudaError> {
        let unsupported = |m: String| CudaError::Unsupported(m);
        for p in [&self.gate, &self.up, &self.down] {
            p.validate(ids).map_err(unsupported)?;
        }
        let (g, u, d) = (&self.gate, &self.up, &self.down);
        if u.rows != g.rows
            || u.cols != g.cols
            || d.cols != g.rows
            || d.rows != g.cols
            || u.n_experts != g.n_experts
            || d.n_experts != g.n_experts
        {
            return Err(unsupported(
                "moe id: the gate, up and down planes disagree".into(),
            ));
        }
        if top_k == 0 || ids.len() != n_tokens * top_k || ids.is_empty() {
            return Err(unsupported(format!(
                "moe id: {} ids for {n_tokens} tokens of top {top_k}",
                ids.len()
            )));
        }
        Ok(())
    }
}

/// Which kernel a step runs: `mul_mm_id` once some expert gathers
/// [`MM_ID_MIN_ROWS`] rows, else `mul_mv_id`. One decision per step,
/// shared by the gate, up and down launches.
pub fn takes_mm_id(ids: &[u32], top_k: usize, n_experts: usize) -> bool {
    IdMap::build(ids, top_k, n_experts, true).max_count() >= MM_ID_MIN_ROWS
}

/// The device copy of one map.
struct DeviceMap {
    map_b: CudaSlice<i32>,
    map_d: CudaSlice<i32>,
    experts: CudaSlice<i32>,
    n_active: usize,
    max_count: usize,
}

impl DeviceMap {
    fn upload(dev: &Arc<CudaDevice>, m: &IdMap) -> Result<Self, CudaError> {
        let up = |what, v: &[i32]| crate::xfer::htod(dev, what, v);
        Ok(DeviceMap {
            map_b: up("moe id map_b upload", &m.map_b)?,
            map_d: up("moe id map_d upload", &m.map_d)?,
            experts: up("moe id experts upload", &m.experts())?,
            n_active: m.eids.len(),
            max_count: m.max_count(),
        })
    }
}

/// How one projection is launched.
enum Route<'a> {
    /// `mul_mv_id`: the slot ids, and `slot / x_div` is the activation row.
    Mv {
        ids: &'a CudaSlice<i32>,
        x_div: usize,
    },
    /// `mul_mm_id`: a row map.
    Mm(&'a DeviceMap),
}

/// Enqueues one routed projection over `n_slots` output rows. Returns
/// the `[n_slots][plane.rows]` output and the plane's resident `Arc`.
fn enqueue_id(
    dev: &Arc<CudaDevice>,
    p: &IdPlane<'_>,
    x: &CudaSlice<f32>,
    n_slots: usize,
    route: Route<'_>,
) -> Result<(CudaSlice<f32>, Arc<ResidentCudaWeights>), CudaError> {
    let w = resident_cuda_weights(dev, &p.data[..p.n_experts * p.expert_stride])?;
    let mut out = dev
        .alloc_zeros::<f32>(n_slots * p.rows)
        .map_err(|e| CudaError::Launch(format!("moe id output alloc: {e:?}")))?;
    let (rows, cols, rb) = (p.rows as i32, p.cols as i32, p.row_bytes() as i32);
    let stride = p.expert_stride as u64;
    match route {
        Route::Mv { ids, x_div } => {
            let (module, entry) = mv_id_names(p.kind);
            ensure_module_loaded_lazy(dev, module, entry, || mv_id_src(p.kind).to_string())?;
            let func = dev
                .get_func(module, entry)
                .ok_or_else(|| CudaError::KernelCompile(format!("'{entry}' not found")))?;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (p.rows as u32, n_slots as u32, 1),
                block_dim: (crate::matvec_kinds::generic::THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_blocks = (p.cols / p.kind.block_elems) as i32;
            // SAFETY: the arguments are the kernel's signature in order.
            // `validate` checked every id is below `n_experts` and the
            // plane holds `n_experts * expert_stride` bytes, so each
            // block reads inside one expert's rows; `x` holds
            // `(n_slots / x_div) * cols` floats for the gate and up
            // (`n_tokens * hidden`) and `n_slots * cols` for the down;
            // `out` holds `n_slots * rows`. Twin: `moe_id::mv_id_reference`.
            unsafe {
                func.launch(
                    cfg,
                    (
                        &w.slice,
                        x,
                        &mut out,
                        ids,
                        rows,
                        rb,
                        n_blocks,
                        n_slots as i32,
                        x_div as i32,
                        cols,
                        stride,
                    ),
                )
            }
            .map_err(|e| CudaError::Launch(format!("kernel {entry}: {e:?}")))?;
        }
        Route::Mm(m) => {
            let (module, entry) = mm_id_names(p.kind);
            ensure_module_loaded_lazy(dev, module, entry, || mm_id_src(p.kind))?;
            let func = dev
                .get_func(module, entry)
                .ok_or_else(|| CudaError::KernelCompile(format!("'{entry}' not found")))?;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (
                    m.max_count.div_ceil(BN) as u32,
                    p.rows.div_ceil(BM) as u32,
                    m.n_active as u32,
                ),
                block_dim: (THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the arguments are the kernel's signature in order
            // (the dense GEMM's seven, then the map and the stride).
            // Every map entry was built from validated ids: `map_b`
            // indexes rows of `x`, `map_d` rows of `out` (both
            // `< n_slots`), `eids` experts of the plane. A block past its
            // expert's count returns before any read. Twin:
            // `moe_id::mm_id_reference`.
            unsafe {
                func.launch(
                    cfg,
                    (
                        &w.slice,
                        x,
                        &mut out,
                        rows,
                        cols,
                        m.max_count as i32,
                        rb,
                        &m.map_b,
                        &m.map_d,
                        &m.experts,
                        stride,
                    ),
                )
            }
            .map_err(|e| CudaError::Launch(format!("kernel {entry}: {e:?}")))?;
        }
    }
    Ok((out, w))
}

/// One routed SwiGLU step: `x` is `n_tokens` rows of `hidden`, `ids`
/// the `n_tokens * top_k` routed experts (slot `t * top_k + j` is token
/// `t`'s `j`-th). Returns `[n_slots][hidden]`, slot `s` the expert's
/// `down(silu(gate(x)) * up(x))` for its token, unweighted: the caller
/// combines with the routing weights in its own order.
pub fn launch_moe_swiglu_id(
    planes: &MoePlanesCuda<'_>,
    x: &[f32],
    n_tokens: usize,
    ids: &[u32],
    top_k: usize,
) -> Result<Vec<f32>, CudaError> {
    planes.validate(ids, n_tokens, top_k)?;
    let hidden = planes.gate.cols;
    if x.len() != n_tokens * hidden {
        return Err(CudaError::Unsupported(format!(
            "moe id: {} activations for {n_tokens} rows of {hidden}",
            x.len()
        )));
    }
    let ffn = planes.gate.rows;
    let n_slots = ids.len();
    let n_experts = planes.gate.n_experts;
    let dev = shared_device()?;
    let d_x = crate::xfer::htod(&dev, "moe id activation upload", x)?;
    let mm = takes_mm_id(ids, top_k, n_experts);
    let mut held = Vec::with_capacity(3);
    let down = if mm {
        let tok = DeviceMap::upload(&dev, &IdMap::build(ids, top_k, n_experts, true))?;
        let slot = DeviceMap::upload(&dev, &IdMap::build(ids, top_k, n_experts, false))?;
        let (g, wg) = enqueue_id(&dev, &planes.gate, &d_x, n_slots, Route::Mm(&tok))?;
        let (u, wu) = enqueue_id(&dev, &planes.up, &d_x, n_slots, Route::Mm(&tok))?;
        let act = silu_mul_device(&dev, &g, &u, n_slots * ffn)?;
        let (d, wd) = enqueue_id(&dev, &planes.down, &act, n_slots, Route::Mm(&slot))?;
        held.extend([wg, wu, wd]);
        d
    } else {
        let ids_i: Vec<i32> = ids.iter().map(|&e| e as i32).collect();
        let d_ids = crate::xfer::htod(&dev, "moe id ids upload", &ids_i)?;
        let tok = Route::Mv {
            ids: &d_ids,
            x_div: top_k,
        };
        let (g, wg) = enqueue_id(&dev, &planes.gate, &d_x, n_slots, tok)?;
        let tok = Route::Mv {
            ids: &d_ids,
            x_div: top_k,
        };
        let (u, wu) = enqueue_id(&dev, &planes.up, &d_x, n_slots, tok)?;
        let act = silu_mul_device(&dev, &g, &u, n_slots * ffn)?;
        let slot = Route::Mv {
            ids: &d_ids,
            x_div: 1,
        };
        let (d, wd) = enqueue_id(&dev, &planes.down, &act, n_slots, slot)?;
        held.extend([wg, wu, wd]);
        d
    };
    // The step's one host wait.
    let out = crate::xfer::dtoh(&dev, "moe id download", &down)?;
    drop(held);
    debug_assert_eq!(out.len(), n_slots * hidden);
    Ok(out)
}

#[cfg(test)]
mod hardware_tests {
    use super::*;
    use crate::moe_id::{mm_id_reference, mv_id_reference, silu_mul_reference};
    use crate::mul_mm::KINDS;
    use crate::mul_mm_ref::fixtures;

    /// The routed step on the device against the twins' step, for every
    /// kind, through both kernels: one token (decode, `mul_mv_id`) and
    /// enough tokens that experts gather past [`MM_ID_MIN_ROWS`]
    /// (`mul_mm_id`). Relative tolerance, because a real GPU contracts
    /// `acc += a * b` into an FMA and the twins do not.
    ///
    /// Run on a machine with an actual CUDA device:
    ///   cargo test -p frink-cuda --features cuda -- --ignored moe_swiglu_id
    #[test]
    #[ignore = "requires real CUDA hardware. No mul_mv_id or mul_mm_id kernel has EVER executed on a GPU; run with --ignored on a CUDA device and record the result"]
    fn every_kind_moe_swiglu_id_matches_the_twins() {
        let (n_experts, top_k, ffn) = (6usize, 2usize, 256usize);
        for k in KINDS {
            let hidden = 256usize.next_multiple_of(k.block_elems);
            let plane = |rows, cols, seed| fixtures::weights(k, n_experts * rows, cols, seed);
            let (gw, uw, dw) = (
                plane(ffn, hidden, 1),
                plane(ffn, hidden, 2),
                plane(hidden, ffn, 3),
            );
            let mk = |data: &'static [u8], rows, cols| IdPlane {
                kind: k,
                data,
                n_experts,
                expert_stride: rows * (cols / k.block_elems) * k.block_bytes,
                rows,
                cols,
            };
            let leak = |v: Vec<u8>| -> &'static [u8] { Box::leak(v.into_boxed_slice()) };
            let planes = MoePlanesCuda {
                gate: mk(leak(gw), ffn, hidden),
                up: mk(leak(uw), ffn, hidden),
                down: mk(leak(dw), hidden, ffn),
            };
            for n_tokens in [1usize, 24] {
                let ids: Vec<u32> = (0..n_tokens * top_k)
                    .map(|s| ((s * 7 + s / 3) % n_experts) as u32)
                    .collect();
                let x: Vec<f32> = (0..n_tokens * hidden)
                    .map(|i| ((i as f32) * 0.011).sin())
                    .collect();
                let mm = takes_mm_id(&ids, top_k, n_experts);
                assert_eq!(mm, n_tokens > 1, "{}: the shape picks the kernel", k.name);
                let (g, u) = if mm {
                    let m = IdMap::build(&ids, top_k, n_experts, true);
                    (
                        mm_id_reference(&planes.gate, &m, &x, ids.len()),
                        mm_id_reference(&planes.up, &m, &x, ids.len()),
                    )
                } else {
                    (
                        mv_id_reference(&planes.gate, &ids, &x, top_k),
                        mv_id_reference(&planes.up, &ids, &x, top_k),
                    )
                };
                let act = silu_mul_reference(&g, &u);
                let want = if mm {
                    let m = IdMap::build(&ids, top_k, n_experts, false);
                    mm_id_reference(&planes.down, &m, &act, ids.len())
                } else {
                    mv_id_reference(&planes.down, &ids, &act, 1)
                };
                let got = launch_moe_swiglu_id(&planes, &x, n_tokens, &ids, top_k)
                    .expect("the routed step must launch on real CUDA hardware");
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let tol = 1e-3 * w.abs().max(1.0);
                    assert!(
                        (g - w).abs() <= tol || (g.is_nan() && w.is_nan()),
                        "{} {n_tokens} tokens [{i}]: GPU {g}, twin {w}",
                        k.name
                    );
                }
            }
        }
    }
}
