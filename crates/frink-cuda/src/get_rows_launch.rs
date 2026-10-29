//! The device side of [`crate::get_rows`]: NVRTC compile once, gather
//! from the resident table, no host wait. The resident stack
//! ([`crate::prefill`]) enqueues it ahead of its first layer; the
//! host-slice entry point is for the hardware test.

use crate::get_rows::{grid_dims, kernel_src, names, validate, THREADS};
use crate::gpu::{
    ensure_module_loaded_lazy, resident_cuda_weights, shared_device, CudaError, ResidentCudaWeights,
};
use crate::mul_mm::MulMmKind;
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// Enqueues the gather of `n_ids` rows (`d_ids` already on the device)
/// out of `table`, made resident on first use. Returns the
/// `n_ids * n_cols` output and the table's `Arc`, which the caller
/// holds until it synchronises. The shape is the caller's to validate.
pub(crate) fn enqueue_get_rows(
    dev: &Arc<CudaDevice>,
    kind: &MulMmKind,
    table: &[u8],
    d_ids: &CudaSlice<u32>,
    n_cols: usize,
    row_bytes: usize,
) -> Result<(CudaSlice<f32>, Arc<ResidentCudaWeights>), CudaError> {
    use cudarc::driver::{DeviceSlice, LaunchAsync};

    let n_ids = d_ids.len();
    let (module, entry) = names(kind);
    ensure_module_loaded_lazy(dev, module, entry, || kernel_src(kind))?;
    let func = dev.get_func(module, entry).ok_or_else(|| {
        CudaError::KernelCompile(format!("function '{entry}' not found after load_ptx"))
    })?;
    let d_table = resident_cuda_weights(dev, table)?;
    let mut d_out = dev
        .alloc_zeros::<f32>(n_ids * n_cols)
        .map_err(|e| CudaError::Launch(format!("get_rows output alloc: {e:?}")))?;
    let (gx, gy) = grid_dims(n_ids, n_cols);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (gx as u32, gy as u32, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the argument list is the kernel's signature in order
    // (table, ids, dst, n_ids, n_cols, row_bytes). `d_table` holds
    // `n_rows * row_bytes` bytes and every id is below `n_rows`
    // (`validate`, which every caller runs first), so each thread reads
    // inside one row; `d_out` holds `n_ids * n_cols` floats and the
    // kernel writes `[r * n_cols, (r + 1) * n_cols)` for `r < n_ids`
    // only. The scalar twin is `get_rows::get_rows_reference`.
    unsafe {
        func.launch(
            cfg,
            (
                &d_table.slice,
                d_ids,
                &mut d_out,
                n_ids as i32,
                n_cols as i32,
                row_bytes as i32,
            ),
        )
        .map_err(|e| CudaError::Launch(format!("kernel {entry}: {e:?}")))?;
    }
    Ok((d_out, d_table))
}

/// Gathers `ids` out of `table` (`n_rows` rows of `row_bytes`) and
/// downloads the rows: one upload of the ids, one download.
pub fn launch_get_rows(
    kind: &MulMmKind,
    table: &[u8],
    n_rows: usize,
    ids: &[u32],
    n_cols: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, CudaError> {
    validate(kind, table.len(), n_rows, ids, n_cols, row_bytes)
        .map_err(|e| CudaError::Unsupported(e.to_string()))?;
    let dev = shared_device()?;
    let d_ids = crate::xfer::htod(&dev, "get_rows ids upload", ids)?;
    let (d_out, held) = enqueue_get_rows(&dev, kind, table, &d_ids, n_cols, row_bytes)?;
    let out = crate::xfer::dtoh(&dev, "get_rows download", &d_out)?;
    drop(held);
    Ok(out)
}

#[cfg(test)]
mod hardware_tests {
    use crate::get_rows::get_rows_reference;
    use crate::mul_mm::KINDS;
    use crate::mul_mm_ref::fixtures;

    /// Every kind's gather on the device against its scalar twin, BIT
    /// FOR BIT: a gather is a dequant and a copy, with no accumulation
    /// a contracted FMA could reorder, so there is no tolerance to
    /// allow. The twin is held to `frink_quant`'s dequant on the host.
    ///
    /// Run on a machine with an actual CUDA device:
    ///   cargo test -p frink-cuda --features cuda -- --ignored get_rows
    #[test]
    #[ignore = "requires real CUDA hardware. No get_rows kernel has EVER executed on a GPU; run with --ignored on a CUDA device and record the result before any doc claims the gather runs"]
    fn every_cuda_get_rows_matches_the_twin_bit_for_bit() {
        let n_rows = 11;
        let ids = [10u32, 0, 4, 4, 9, 1];
        for k in KINDS {
            let n_cols = 3 * k.block_elems.max(64);
            let row_bytes = (n_cols / k.block_elems) * k.block_bytes;
            let table = fixtures::weights(k, n_rows, n_cols, 4049);
            let want = get_rows_reference(k, &table, n_rows, &ids, n_cols, row_bytes).unwrap();
            let got = super::launch_get_rows(k, &table, n_rows, &ids, n_cols, row_bytes)
                .expect("the gather must launch on real CUDA hardware");
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                    "{} element {i}: GPU {g}, twin {w}",
                    k.name
                );
            }
        }
    }
}
