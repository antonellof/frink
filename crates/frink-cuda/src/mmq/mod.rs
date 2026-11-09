//! The int8 prefill GEMM, llama.cpp's MMQ: activations quantized to
//! q8_1 once per GEMM input ([`quantize`]) and multiplied against the
//! quantized weights on the int8 tensor cores ([`kernel`]).
//!
//! It exists for two reasons. The f16 tensor-core GEMM (`mul_mm_tc`)
//! reached about 107 TFLOPS on an L40S, where Ada's int8 tensor rate is
//! twice its f16 one. And llama.cpp's CUDA prefill quantizes every
//! quantized weight type's activations to q8_1, so an f16-activation
//! GEMM was a numerics deviation from the oracle (ADR 0009) as well as
//! the slower path. Selected by `FRINK_CUDA_MUL_MM=mmq` for the kinds it
//! has a loader for, while it is measured.

pub mod kernel;
pub mod quantize;

#[cfg(feature = "cuda")]
mod launch {
    use super::kernel::{self, MmqKind};
    use super::quantize;
    use crate::gpu::CudaError;
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use std::sync::Arc;

    /// `out[t][r] = w[r] . x[t]` for `batch` rows `x` of `n_cols`: the
    /// activations quantized, then the int8 GEMM. The caller holds the
    /// returned weights until it synchronises.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_mmq(
        dev: &Arc<CudaDevice>,
        kind: MmqKind,
        weights: &[u8],
        x: &CudaSlice<f32>,
        n_rows: usize,
        n_cols: usize,
        batch: usize,
        row_bytes: usize,
    ) -> Result<(CudaSlice<f32>, Arc<crate::gpu::ResidentCudaWeights>), CudaError> {
        let block = kind.block();
        if n_rows == 0
            || batch == 0
            || !n_cols.is_multiple_of(block.0)
            || row_bytes != n_cols / block.0 * block.1
            || weights.len() != n_rows * row_bytes
        {
            return Err(CudaError::Unsupported(format!(
                "mmq {kind:?}: {n_rows} x {n_cols} (row {row_bytes} B) over {batch}"
            )));
        }
        let ncols_pad = kernel::ncols_pad(batch);
        let y = quantize::enqueue_quantize(dev, kind.layout(), x, None, n_cols, batch, ncols_pad)?;
        let held = crate::gpu::resident_cuda_weights(dev, weights)?;
        let (src, module, name) = kind.source();
        crate::gpu::ensure_module_loaded_lazy_for_arch(
            dev,
            module,
            name,
            Some(kernel::ARCH),
            || src,
        )?;
        let func = dev.get_func(module, name).ok_or_else(|| {
            CudaError::KernelCompile(format!("function '{name}' not found after load_ptx"))
        })?;
        // SAFETY: uninitialised; every `(token < batch, row < n_rows)`
        // output is written by the one thread whose fragment holds it.
        let mut out = unsafe { crate::gpu::alloc_out(dev, batch * n_rows, "mmq output") }?;
        let (grid, blk) = kernel::geometry(n_rows, ncols_pad);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: blk,
            shared_mem_bytes: 0,
        };
        // SAFETY: (const u8*, const u8*, float*, int x5) matched
        // positionally; the weights are `n_rows * row_bytes` (checked),
        // every row the tile reads is clamped below `n_rows` and every
        // block past `n_blocks` is skipped; the activations hold
        // `kpad / 128 * ncols_pad` blocks, every one a tile of `ncols_pad /
        // 64` reads; the output stores are guarded by `batch` and `n_rows`.
        unsafe {
            func.launch(
                cfg,
                (
                    &held.slice,
                    &y,
                    &mut out,
                    n_rows as i32,
                    (n_cols / block.0) as i32,
                    batch as i32,
                    ncols_pad as i32,
                    row_bytes as i32,
                ),
            )
            .map_err(|e| CudaError::Launch(format!("{name}: {e:?}")))?;
        }
        Ok((out, held))
    }
}

#[cfg(feature = "cuda")]
pub(crate) use launch::enqueue_mmq;

/// The int8 GEMM's arithmetic on the host, in its order: per output,
/// over the blocks of K ascending, the exact int32 dot of the block's
/// quants scaled into the accumulator as `acc = (C * dw) * dy + acc`, the
/// last multiply fused with the add as the device compiles it.
#[cfg(test)]
pub(crate) fn mmq_q8_0_twin(
    w: &[u8],
    y: &[u8],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    ncols_pad: usize,
) -> Vec<f32> {
    let n_blocks = n_cols / 32;
    let row_bytes = n_blocks * 34;
    let mut out = vec![0f32; batch * n_rows];
    for col in 0..batch {
        for row in 0..n_rows {
            let mut acc = 0f32;
            for blk in 0..n_blocks.div_ceil(4) * 4 {
                let (dw, wq): (f32, Vec<i8>) = if blk < n_blocks {
                    let b = &w[row * row_bytes + blk * 34..row * row_bytes + blk * 34 + 34];
                    (
                        half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
                        b[2..].iter().map(|&v| v as i8).collect(),
                    )
                } else {
                    (0.0, vec![0; 32])
                };
                let yb = &y[((blk / 4) * ncols_pad + col) * quantize::BLOCK_BYTES..];
                let sub = blk % 4;
                let dy = f32::from_le_bytes(yb[4 * sub..4 * sub + 4].try_into().unwrap());
                let c: i32 = (0..32)
                    .map(|e| i32::from(wq[e]) * i32::from(yb[16 + sub * 32 + e] as i8))
                    .sum();
                acc = (c as f32 * dw).mul_add(dy, acc);
            }
            out[col * n_rows + row] = acc;
        }
    }
    out
}

/// [`mmq_q8_0_twin`] for Q4_K. Per 32-value run it takes
/// `acc = (dm.x * dy) * C + acc`, then `acc = dm.y * s + acc`, where `dm`
/// is the run's `half(d * sc)` and `half(-dmin * m)` and `(dy, s)` are
/// the activations' half scale and sum.
#[cfg(test)]
pub(crate) fn mmq_q4_k_twin(
    w: &[u8],
    y: &[u8],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    ncols_pad: usize,
) -> Vec<f32> {
    let n_sb = n_cols / 256;
    let row_bytes = n_sb * 144;
    let h = |b: &[u8]| half::f16::from_le_bytes([b[0], b[1]]).to_f32();
    let round_h = |x: f32| half::f16::from_f32(x).to_f32();
    let mut out = vec![0f32; batch * n_rows];
    for col in 0..batch {
        for row in 0..n_rows {
            let mut acc = 0f32;
            for sb in 0..n_sb {
                let blk = &w[row * row_bytes + sb * 144..row * row_bytes + sb * 144 + 144];
                let (d, dmin, q) = (h(&blk[0..2]), h(&blk[2..4]), &blk[4..16]);
                for j in 0..8 {
                    let (sc, m) = if j < 4 {
                        (q[j] & 63, q[j + 4] & 63)
                    } else {
                        (
                            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
                        )
                    };
                    let dmx = round_h(d * f32::from(sc));
                    let dmy = round_h(-dmin * f32::from(m));
                    // Run j's 32 nibbles: the low or high halves of qs
                    // bytes (j / 2) * 32 .. + 32.
                    let qs = &blk[16 + (j / 2) * 32..16 + (j / 2) * 32 + 32];
                    let step = sb * 2 + j / 4;
                    let sub = j % 4;
                    let yb = &y[(step * ncols_pad + col) * quantize::BLOCK_BYTES..];
                    let dy = h(&yb[4 * sub..4 * sub + 2]);
                    let sy = h(&yb[4 * sub + 2..4 * sub + 4]);
                    let c: i32 = (0..32)
                        .map(|e| {
                            let nib = if j % 2 == 0 { qs[e] & 0xF } else { qs[e] >> 4 };
                            i32::from(nib) * i32::from(yb[16 + sub * 32 + e] as i8)
                        })
                        .sum();
                    acc = (dmx * dy).mul_add(c as f32, acc);
                    acc = dmy.mul_add(sy, acc);
                }
            }
            out[col * n_rows + row] = acc;
        }
    }
    out
}

#[cfg(all(test, feature = "cuda"))]
mod hw_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// The twin has to be RIGHT, not only consistent with the kernel: the
    /// int8 GEMM over q8_1 activations lands within the activations'
    /// rounding of the f32 reference, over a K with a partial step and a
    /// batch with padding columns.
    /// The Q4_K twin, held to the f32 reference the same way, over two
    /// superblocks a row (both halves of each).
    #[test]
    fn the_q4_k_twin_is_the_gemm_within_the_q8_1_rounding() {
        let (n_rows, n_cols, batch) = (7usize, 512usize, 3usize);
        let w = crate::mul_mm_ref::fixtures::weights(&crate::mul_mm::Q4_K, n_rows, n_cols, 13);
        let x: Vec<f32> = (0..batch * n_cols)
            .map(|i| ((i as f32) * 0.23).sin() * (1u32 << ((i / 32) % 5)) as f32)
            .collect();
        let ncols_pad = kernel::ncols_pad(batch);
        let y = quantize::quantize_twin(&x, None, n_cols, batch, ncols_pad, quantize::Layout::Ds4);
        let got = mmq_q4_k_twin(&w, &y, n_rows, n_cols, batch, ncols_pad);
        let want = crate::mul_mm_ref::mul_mm_reference(
            &crate::mul_mm::Q4_K,
            &w,
            &x,
            n_rows,
            n_cols,
            batch,
            n_cols / 256 * 144,
        )
        .unwrap();
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        for (i, (g, r)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - r).abs() <= 2e-2 * rms,
                "[{i}]: twin {g} reference {r} (rms {rms})"
            );
        }
    }

    #[test]
    fn the_q8_0_twin_is_the_gemm_within_the_q8_1_rounding() {
        let (n_rows, n_cols, batch) = (9usize, 160usize, 5usize);
        let w = crate::mul_mm_ref::fixtures::weights(&crate::mul_mm::Q8_0, n_rows, n_cols, 7);
        // Each 32-value run at its own amplitude (1x to 16x), so a scale
        // taken from the wrong run is off by a whole factor.
        let x: Vec<f32> = (0..batch * n_cols)
            .map(|i| ((i as f32) * 0.21).cos() * (1u32 << ((i / 32) % 5)) as f32)
            .collect();
        let ncols_pad = kernel::ncols_pad(batch);
        let y = quantize::quantize_twin(&x, None, n_cols, batch, ncols_pad, quantize::Layout::D4);
        let got = mmq_q8_0_twin(&w, &y, n_rows, n_cols, batch, ncols_pad);
        let want = crate::mul_mm_ref::mul_mm_reference(
            &crate::mul_mm::Q8_0,
            &w,
            &x,
            n_rows,
            n_cols,
            batch,
            n_cols / 32 * 34,
        )
        .unwrap();
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        for (i, (g, r)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - r).abs() <= 2e-2 * rms,
                "[{i}]: twin {g} reference {r} (rms {rms})"
            );
        }
    }
}
