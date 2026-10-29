//! The device side of `mul_mm`: NVRTC compile, upload, launch, download.
//!
//! Run on hardware since 2026-09-15 (RTX 3090, CUDA 12.4): every kind
//! and shape in [`tests::launch_mul_mm_matches_the_scalar_twin`]
//! passes, and `frink verify --backend cuda` is token-identical to
//! the CPU on Q4_K_M, Q5_K_M, Q6_K, Q8_0 and IQ4_XS checkpoints. It is
//! written against `cudarc` 0.11.9's API the same way `gpu.rs`'s
//! matvec launchers are, and reuses their plumbing -- the process-wide
//! device (`shared_device`), the load-once NVRTC cache
//! (`ensure_module_loaded_lazy`) and the pointer-keyed resident weight
//! cache (`resident_cuda_weights`) -- rather than growing a second copy
//! of any of it.
//!
//! Two entry points: [`launch_mul_mm`] takes and returns host slices
//! (one upload, one synchronous download), and [`enqueue_mul_mm`] is
//! the launch alone over device slices, which the resident prefill
//! stack ([`crate::prefill`]) chains seven of per layer. The
//! arithmetic both launch is checked on the host by
//! [`crate::mul_mm_ref`]. The hardware test is `#[ignore]`d because it
//! needs a device; run it on real hardware and write down what
//! happened.

use crate::gpu::{resident_cuda_weights, shared_device, CudaError};
use crate::mul_mm::{grid_dims, kernel_src, validate_shape, MulMmKind, THREADS};

/// Batched quantized GEMM: `dst[token][row] = sum_k W[row][k] * x[token][k]`.
///
/// * `weights` -- `n_rows` quantized rows of `row_bytes` each, in
///   `kind`'s format, exactly as the GGUF mmap holds them. Cached on the
///   device by host pointer, so a repeated call for the same tensor does
///   not re-upload it.
/// * `x_batch` -- `batch` activation rows of `n_cols` f32, row-major,
///   the layout `WeightMatrix::apply_batch` already has.
/// * returns `batch * n_rows` f32 as `out[token * n_rows + row]`, again
///   the layout `apply_batch` already returns.
///
/// Returns [`CudaError::Unsupported`] for a shape this kernel does not
/// implement, so the caller can fall back and disclose it rather than
/// computing something else.
pub fn launch_mul_mm(
    kind: &MulMmKind,
    weights: &[u8],
    x_batch: &[f32],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, CudaError> {
    validate_shape(
        kind,
        weights.len(),
        x_batch.len(),
        n_rows,
        n_cols,
        batch,
        row_bytes,
    )
    .map_err(|e| CudaError::Unsupported(e.to_string()))?;

    let dev = shared_device()?;
    let d_x = crate::xfer::htod(&dev, "mul_mm activation upload", &x_batch[..batch * n_cols])?;
    let (d_out, d_weights) =
        enqueue_mul_mm(&dev, kind, weights, &d_x, n_rows, n_cols, batch, row_bytes)?;
    let out = crate::xfer::dtoh(&dev, "mul_mm output download", &d_out)?;
    drop(d_weights);
    Ok(out)
}

/// The launch alone: `d_x` is already on the device and the output
/// stays there. This is what [`launch_mul_mm`] wraps in an upload and
/// a download, and what the resident prefill layer
/// ([`crate::prefill`]) chains seven of without either. The caller
/// holds the returned weight `Arc` until it synchronises: the kernel
/// reads that buffer asynchronously.
///
/// `validate_shape` is the caller's: this function trusts the shape
/// it is handed, and both callers check it first.
#[allow(clippy::too_many_arguments)]
pub(crate) fn enqueue_mul_mm(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    kind: &MulMmKind,
    weights: &[u8],
    d_x: &cudarc::driver::CudaSlice<f32>,
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
) -> Result<
    (
        cudarc::driver::CudaSlice<f32>,
        std::sync::Arc<crate::gpu::ResidentCudaWeights>,
    ),
    CudaError,
> {
    // The tensor-core body on Ampere and up, the SIMT body below it;
    // `FRINK_CUDA_MUL_MM=simt` keeps the SIMT body for an A/B.
    let body = MulMmBody::select(dev);
    enqueue_mul_mm_with_body(
        dev, body, kind, weights, d_x, n_rows, n_cols, batch, row_bytes,
    )
}

/// [`enqueue_mul_mm`] with the body chosen by the caller: the hardware
/// tests hold both bodies against the twin whatever the device would
/// pick.
#[allow(clippy::too_many_arguments)]
pub(crate) fn enqueue_mul_mm_with_body(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    body: MulMmBody,
    kind: &MulMmKind,
    weights: &[u8],
    d_x: &cudarc::driver::CudaSlice<f32>,
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
) -> Result<
    (
        cudarc::driver::CudaSlice<f32>,
        std::sync::Arc<crate::gpu::ResidentCudaWeights>,
    ),
    CudaError,
> {
    use cudarc::driver::LaunchAsync;

    let (module_name, fn_name, src): (&'static str, &'static str, Box<dyn FnOnce() -> String>) =
        match body {
            MulMmBody::TensorCore => {
                let (m, f) = crate::mul_mm_tc::tc_names(kind);
                (
                    m,
                    f,
                    Box::new(move || crate::mul_mm_tc::tc_kernel_src(kind)),
                )
            }
            MulMmBody::Simt => (
                kind.module_name,
                kind.fn_name,
                Box::new(move || kernel_src(kind)),
            ),
        };
    crate::gpu::ensure_module_loaded_lazy_for_arch(dev, module_name, fn_name, body.arch(), src)?;
    let func = dev.get_func(module_name, fn_name).ok_or_else(|| {
        CudaError::KernelCompile(format!("function '{fn_name}' not found after load_ptx"))
    })?;

    // `d_weights` must outlive the launch: the kernel reads that buffer
    // asynchronously and only the caller's DtoH synchronizes.
    let d_weights = resident_cuda_weights(dev, weights)?;
    let mut d_out = dev
        .alloc_zeros::<f32>(batch * n_rows)
        .map_err(|e| CudaError::Launch(format!("mul_mm output alloc: {e:?}")))?;

    let (grid_x, grid_y, threads) = match body {
        MulMmBody::TensorCore => {
            let (x, y) = crate::mul_mm_tc::tc_grid_dims(n_rows, batch);
            (x, y, crate::mul_mm_tc::TTHREADS)
        }
        MulMmBody::Simt => {
            let (x, y) = grid_dims(n_rows, batch);
            (x, y, THREADS)
        }
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (threads as u32, 1, 1),
        // The tiles are declared `__shared__` inside the kernel, so the
        // dynamic shared-memory request is zero. Asking for more here
        // would be added to the static allocation, not replace it.
        shared_mem_bytes: 0,
    };

    // SAFETY: `func` was compiled from `kernel_src(kind)`, whose
    // parameter list is (const uchar*, const float*, float*, int, int,
    // int, int) and is matched positionally by the tuple below. Each
    // buffer is at least the size the kernel indexes: `validate_shape`
    // has established `weights.len() >= n_rows * row_bytes` and
    // `d_x.len() >= batch * n_cols`, `d_out` is allocated at exactly
    // `batch * n_rows`, and the kernel bounds-checks every store against
    // `n_rows`/`batch`. The grid covers `ceil(batch/BN) x
    // ceil(n_rows/BM)` tiles, so no thread addresses a row beyond
    // `n_rows - 1` (out-of-range rows are clamped inside the kernel).
    // `d_weights` is held alive across the launch by the caller.
    unsafe {
        func.launch(
            cfg,
            (
                &d_weights.slice,
                d_x,
                &mut d_out,
                n_rows as i32,
                n_cols as i32,
                batch as i32,
                row_bytes as i32,
            ),
        )
        .map_err(|e| CudaError::Launch(format!("kernel {fn_name}: {e:?}")))?;
    }
    Ok((d_out, d_weights))
}

/// Which GEMM body a launch takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MulMmBody {
    /// `mul_mm.rs`: f32 FMA micro-tiles. Every device.
    Simt,
    /// `mul_mm_tc.rs`: f16 `mma.sync` with f32 accumulation. `sm_80`+.
    TensorCore,
}

impl MulMmBody {
    /// The tensor-core body when the device can run it and
    /// `FRINK_CUDA_MUL_MM` does not say `simt`; the SIMT body otherwise.
    /// `FRINK_CUDA_MUL_MM=tc` on a device below `sm_80` still answers
    /// SIMT, because a PTX the driver cannot load is not an A/B.
    pub fn select(dev: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Self {
        let forced_simt = std::env::var("FRINK_CUDA_MUL_MM")
            .map(|v| v.eq_ignore_ascii_case("simt"))
            .unwrap_or(false);
        if forced_simt || crate::gpu::compute_capability_major(dev) < crate::mul_mm_tc::MIN_CC_MAJOR
        {
            Self::Simt
        } else {
            Self::TensorCore
        }
    }

    /// The NVRTC target the body needs, `None` for the default.
    pub fn arch(self) -> Option<&'static str> {
        match self {
            Self::Simt => None,
            Self::TensorCore => Some(crate::mul_mm_tc::ARCH),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul_mm::{BM, Q8_0};
    use crate::mul_mm_ref::mul_mm_reference;

    #[test]
    fn a_shape_the_kernel_cannot_do_is_refused_without_touching_the_device() {
        // n_cols is not a whole K-tile. This must come back as a named
        // refusal, not a launch attempt -- on a host with no CUDA
        // library at all, reaching `shared_device()` would be a
        // different error entirely.
        let err = launch_mul_mm(&Q8_0, &[0; 68], &[0.0; 48], 2, 48, 1, 34).unwrap_err();
        match err {
            CudaError::Unsupported(msg) => {
                assert!(msg.contains("K-tile"), "unhelpful refusal: {msg}");
            }
            other => panic!("expected a named refusal, got {other:?}"),
        }
    }

    /// The one test that would close the gap this module leaves open.
    ///
    /// It compares the kernel against [`mul_mm_reference`] -- the same
    /// scalar twin the host-side tests already hold to `frink_quant`
    /// -- on an exact-tile shape, a partial tile on both axes, and a
    /// narrow batch, for EVERY kind in [`KINDS`].
    ///
    /// It used to name three kinds and pick their fixtures out of an
    /// index-keyed `match`, so the kinds it did not name (the K-quants
    /// then, the codebook formats now) were covered only by
    /// `tools/mul_mm_host_check/run.sh` -- which executes the emitted C
    /// on a host CPU and therefore cannot see anything NVRTC or a warp
    /// scheduler would. The loop is over the table now, and the
    /// fixtures come from [`crate::mul_mm_ref::fixtures`], so a kind
    /// added to `KINDS` is on this list the moment it exists.
    ///
    /// Run it on a machine with a real device:
    ///   cargo test -p frink-cuda --features cuda -- --ignored
    #[test]
    #[ignore = "requires real CUDA hardware. Last run 2026-09-15 on an RTX 3090 (CUDA 12.4): every kind and shape passes at the tolerance below, and the same day `frink verify --backend cuda` was token-identical to the CPU on Q4_K_M, Q5_K_M, Q6_K, Q8_0 and IQ4_XS checkpoints"]
    fn launch_mul_mm_matches_the_scalar_twin() {
        use crate::mul_mm::KINDS;

        for kind in KINDS {
            for (n_rows, cols, batch) in [(BM * 2, 128usize, 32), (BM + 7, 96, 37), (33, 64, 3)] {
                // `validate_shape` refuses a column count that is not a
                // whole super-block, so round up per kind rather than
                // reusing one list of literals across formats whose
                // block sizes differ by 8x.
                let n_cols = cols.next_multiple_of(kind.block_elems);
                let row_bytes = (n_cols / kind.block_elems) * kind.block_bytes;
                let weights = crate::mul_mm_ref::fixtures::weights(kind, n_rows, n_cols, 4242);
                let x: Vec<f32> = (0..batch * n_cols)
                    .map(|i| ((i as f32) * 0.019).cos())
                    .collect();

                let want = mul_mm_reference(kind, &weights, &x, n_rows, n_cols, batch, row_bytes)
                    .expect("the twin must accept every shape the kernel accepts");
                let got = launch_with_body(
                    MulMmBody::Simt,
                    kind,
                    &weights,
                    &x,
                    n_rows,
                    n_cols,
                    batch,
                    row_bytes,
                );

                assert_eq!(got.len(), want.len());
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    let where_ = format!("{} {n_rows}x{n_cols}x{batch} element {i}", kind.name);
                    // NaN==NaN is agreement, not failure: a fixture byte
                    // pattern that decodes as a NaN f16 scale produces a
                    // NaN identically on both sides, and IEEE754 makes
                    // every comparison against it false. This is the
                    // rule a real RTX 3060 run forced on `gpu.rs`'s
                    // `assert_close_relative`; inheriting it here rather
                    // than rediscovering it on rented hardware.
                    if w.is_nan() {
                        assert!(g.is_nan(), "{where_}: twin is NaN but GPU={g} is not");
                        continue;
                    }
                    // Relative to the RESULT plus an absolute floor that
                    // grows with the column count, and deliberately not
                    // exact. The host execution check
                    // (`tools/mul_mm_host_check`) IS bit-exact against
                    // this same twin, but only because it disables FP
                    // contraction; a real GPU contracts `acc += a * b`
                    // into an FMA, so the accumulator legitimately drifts
                    // from the twin's over `n_cols` steps -- and the drift
                    // is proportional to the L1 norm of the products, not
                    // to the sum they cancel down to. Measured on an RTX
                    // 3090 (2026-09-15): with random fixture weights the
                    // worst |GPU - twin| is 4.8e-4 over 256 columns
                    // (Q5_K, a sum near 0.62), which is eps_f32 times an
                    // L1 of a few thousand, and every other kind and
                    // shape is under it; a result-relative 1e-4 alone
                    // failed that element while `frink verify` was
                    // token-identical to the CPU on the same kinds. A
                    // real bug (a wrong scale, a wrong sub-block offset)
                    // is off by the magnitude of a term, orders above
                    // this floor.
                    let tol = 1e-4 * w.abs().max(1.0) + 4e-6 * n_cols as f32;
                    assert!((g - w).abs() <= tol, "{where_}: GPU={g} twin={w}");
                }
            }
        }
    }

    /// One launch of the chosen body, host slices in and out, for the
    /// tests: `launch_mul_mm` itself takes the device's pick.
    #[allow(clippy::too_many_arguments)]
    fn launch_with_body(
        body: MulMmBody,
        kind: &MulMmKind,
        weights: &[u8],
        x: &[f32],
        n_rows: usize,
        n_cols: usize,
        batch: usize,
        row_bytes: usize,
    ) -> Vec<f32> {
        let dev = shared_device().expect("a CUDA device");
        let d_x = dev.htod_copy(x.to_vec()).unwrap();
        let (d_out, w) = enqueue_mul_mm_with_body(
            &dev, body, kind, weights, &d_x, n_rows, n_cols, batch, row_bytes,
        )
        .expect("kernel launch must succeed on real CUDA hardware");
        let out = dev.dtoh_sync_copy(&d_out).unwrap();
        drop(w);
        out
    }

    /// The tensor-core body against the twin, in two halves.
    ///
    /// First, EXACT: a Q8_0 / Q4_0 / Q5_0 fixture whose f16 scale is a
    /// power of two, so every dequantized weight (an integer of at most
    /// eight bits times `2^-4`) and every activation (`k / 64`) is
    /// exactly representable in f16, and the only difference left
    /// between the two bodies is the f32 accumulation order. That half
    /// holds the tight tolerance of the SIMT test, and it is what
    /// catches a wrong fragment layout: a swapped k pair or a
    /// transposed C fragment is off by the magnitude of a term.
    ///
    /// Second, EVERY kind on the shared fixtures, where the f16 rounding
    /// of each operand (2^-11 relative) is real and the bound has to
    /// admit it: `2^-10` of the L1 norm of the products, which the test
    /// computes from the twin's own dequantization, plus the SIMT floor.
    #[test]
    #[ignore = "requires real CUDA hardware (sm_80 and up); not yet run"]
    fn the_tensor_core_body_matches_the_scalar_twin() {
        use crate::mul_mm::{KINDS, Q4_0, Q5_0, Q8_0};

        let dev = shared_device().expect("a CUDA device");
        assert_eq!(
            MulMmBody::select(&dev),
            MulMmBody::TensorCore,
            "this test needs an sm_80+ device"
        );

        // Exact half.
        for kind in [&Q8_0, &Q4_0, &Q5_0] {
            for (n_rows, n_cols, batch) in [
                (256usize, 128usize, 128usize),
                (135, 96, 37),
                (33, 64, 3),
                (128, 256, 200),
            ] {
                let n_cols = n_cols.next_multiple_of(kind.block_elems);
                let row_bytes = (n_cols / kind.block_elems) * kind.block_bytes;
                let mut weights = crate::mul_mm_ref::fixtures::weights(kind, n_rows, n_cols, 77);
                for block in weights.chunks_exact_mut(kind.block_bytes) {
                    // Scale bits: exponent 11 (2^-4), mantissa zero, sign kept.
                    let bits = u16::from(block[0]) | (u16::from(block[1]) << 8);
                    let bits = (bits & 0x8000) | (11 << 10);
                    block[0] = bits as u8;
                    block[1] = (bits >> 8) as u8;
                }
                let x: Vec<f32> = (0..batch * n_cols)
                    .map(|i| ((i * 37 + 11) % 129) as f32 / 64.0 - 1.0)
                    .collect();
                let want =
                    mul_mm_reference(kind, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
                let got = launch_with_body(
                    MulMmBody::TensorCore,
                    kind,
                    &weights,
                    &x,
                    n_rows,
                    n_cols,
                    batch,
                    row_bytes,
                );
                assert_eq!(got.len(), want.len());
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let tol = 1e-4 * w.abs().max(1.0) + 4e-6 * n_cols as f32;
                    assert!(
                        (g - w).abs() <= tol,
                        "{} exact {n_rows}x{n_cols}x{batch} element {i}: TC={g} twin={w}",
                        kind.name
                    );
                }
            }
        }

        // Rounded half: every kind, the bound from the products' L1.
        for kind in KINDS {
            for (n_rows, cols, batch) in [
                (BM * 2, 128usize, 32),
                (BM + 7, 96, 37),
                (33, 64, 3),
                (300, 256, 150),
            ] {
                let n_cols = cols.next_multiple_of(kind.block_elems);
                let row_bytes = (n_cols / kind.block_elems) * kind.block_bytes;
                let weights = crate::mul_mm_ref::fixtures::weights(kind, n_rows, n_cols, 4242);
                let x: Vec<f32> = (0..batch * n_cols)
                    .map(|i| ((i as f32) * 0.019).cos())
                    .collect();
                let want =
                    mul_mm_reference(kind, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
                let l1 = crate::mul_mm_ref::product_l1(
                    kind, &weights, &x, n_rows, n_cols, batch, row_bytes,
                )
                .unwrap();
                let got = launch_with_body(
                    MulMmBody::TensorCore,
                    kind,
                    &weights,
                    &x,
                    n_rows,
                    n_cols,
                    batch,
                    row_bytes,
                );
                assert_eq!(got.len(), want.len());
                for (i, ((g, w), l1)) in got.iter().zip(&want).zip(&l1).enumerate() {
                    if w.is_nan() {
                        assert!(g.is_nan(), "{}: twin is NaN but TC={g} is not", kind.name);
                        continue;
                    }
                    // Each operand carries at most 2^-11 relative error
                    // from the f16 rounding, so each product at most
                    // ~2^-10; the sum's error is bounded by that times
                    // the L1 of the products. Twice that, plus the SIMT
                    // floor, and it still catches a wrong scale or
                    // offset, which moves a result by a whole term.
                    let tol = 2.0 * l1 / 1024.0 + 1e-4 * w.abs().max(1.0) + 4e-6 * n_cols as f32;
                    assert!(
                        (g - w).abs() <= tol,
                        "{} {n_rows}x{n_cols}x{batch} element {i}: TC={g} twin={w} l1={l1}",
                        kind.name
                    );
                }
            }
        }
    }
}
