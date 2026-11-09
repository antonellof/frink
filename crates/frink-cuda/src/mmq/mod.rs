//! The int8 prefill GEMM, llama.cpp's MMQ: activations quantized to
//! q8_1 once per GEMM input ([`quantize`]) and multiplied against the
//! quantized weights on the int8 tensor cores ([`kernel`]).
//!
//! It exists for two reasons. The f16 tensor-core GEMM (`mul_mm_tc`)
//! reached about 107 TFLOPS on an L40S, where Ada's int8 tensor rate is
//! twice its f16 one. And llama.cpp's CUDA prefill quantizes every
//! quantized weight type's activations to q8_1, so an f16-activation
//! GEMM was a numerics deviation from the oracle (ADR 0009) as well as
//! the slower path. The default for Q8_0 and Q4_K on `sm_80` and up
//! (`FRINK_CUDA_MUL_MM=tc` keeps the f16 body); Q6_K has a loader but
//! measured slower than the f16 body, so it keeps that.

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

    /// A routed layer's experts' projection in ONE int8 launch: group `g`
    /// multiplies the weights `table[g]` by the activation columns
    /// `group_off[g] .. group_off[g + 1]` of `y` (q8_1 blocks the
    /// quantizer wrote in grouped order, `ncols_pad` columns) into output
    /// rows of the same indices, `pairs` rows of `n_rows` in all.
    ///
    /// # Safety
    /// `table` holds `n_groups` device pointers to `n_rows * row_bytes`
    /// live bytes of `kind` each, alive until the stream has run the
    /// launch; `group_off` holds `n_groups + 1` non-decreasing pair
    /// indices from 0 to `pairs`, no group larger than `max_group`; `y`
    /// holds `ncols_pad >= pairs + 63` columns of `n_cols` values.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_mmq_grouped(
        dev: &Arc<CudaDevice>,
        kind: MmqKind,
        table: &cudarc::driver::CudaView<'_, u64>,
        n_groups: usize,
        group_off: &CudaSlice<i32>,
        y: &CudaSlice<u8>,
        ncols_pad: usize,
        pairs: usize,
        max_group: usize,
        n_rows: usize,
        n_cols: usize,
        row_bytes: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let block = kind.block();
        if n_groups == 0
            || pairs == 0
            || !n_cols.is_multiple_of(block.0)
            || row_bytes != n_cols / block.0 * block.1
            || ncols_pad < pairs + kernel::COLS - 1
        {
            return Err(CudaError::Unsupported(format!(
                "mmq grouped {kind:?}: {n_groups} groups of {n_rows} x {n_cols} over {pairs}"
            )));
        }
        let (src, module, name) = kind.grouped_source();
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
        // SAFETY: every group's blocks write every one of its pairs' rows
        // below `n_rows`, and the groups' pairs are `0 .. pairs`.
        let mut out = unsafe { crate::gpu::alloc_out(dev, pairs * n_rows, "mmq grouped output") }?;
        let cfg = LaunchConfig {
            grid_dim: (
                max_group.max(1).div_ceil(kernel::COLS) as u32,
                n_rows.div_ceil(kernel::ROWS) as u32,
                n_groups as u32,
            ),
            block_dim: (kernel::THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: (const u64*, const int*, const u8*, float*, int x5)
        // matched positionally by the FX_GROUPED parameter list, under the
        // caller's contract above: a block's tile of columns starts at its
        // group's first pair and reads at most 63 past its last, inside
        // `ncols_pad`; the stores are guarded by the group's count.
        unsafe {
            func.launch(
                cfg,
                (
                    table,
                    group_off,
                    y,
                    &mut out,
                    n_rows as i32,
                    (n_cols / block.0) as i32,
                    0i32,
                    ncols_pad as i32,
                    row_bytes as i32,
                ),
            )
            .map_err(|e| CudaError::Launch(format!("{name}: {e:?}")))?;
        }
        Ok(out)
    }
}

#[cfg(feature = "cuda")]
pub(crate) use launch::{enqueue_mmq, enqueue_mmq_grouped};

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

/// [`mmq_q8_0_twin`] for Q6_K. Per half superblock (128 values) a
/// temporary takes `tmp = (C0 * sc0 + C1 * sc1) * dy + tmp` for each
/// 32-value run (`C0`, `C1` its two 16-value dots, `sc` their int8
/// sub-scales), then `acc = tmp * d + acc`.
#[cfg(test)]
pub(crate) fn mmq_q6_k_twin(
    w: &[u8],
    y: &[u8],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    ncols_pad: usize,
) -> Vec<f32> {
    let n_sb = n_cols / 256;
    let row_bytes = n_sb * 210;
    let mut out = vec![0f32; batch * n_rows];
    for col in 0..batch {
        for row in 0..n_rows {
            let mut acc = 0f32;
            for sb in 0..n_sb {
                let blk = &w[row * row_bytes + sb * 210..row * row_bytes + sb * 210 + 210];
                let (ql, qh, sc) = (&blk[..128], &blk[128..192], &blk[192..208]);
                let d = half::f16::from_le_bytes([blk[208], blk[209]]).to_f32();
                for h in 0..2 {
                    let q = |v: usize| -> i32 {
                        let (r, l) = (v / 32, v % 32);
                        let lb = ql[h * 64 + (r & 1) * 32 + l];
                        let nib = if r < 2 { lb & 0xF } else { lb >> 4 };
                        let hb = (qh[h * 32 + l] >> (2 * r)) & 3;
                        i32::from(nib | (hb << 4)) - 32
                    };
                    let step = sb * 2 + h;
                    let yb = &y[(step * ncols_pad + col) * quantize::BLOCK_BYTES..];
                    let mut tmp = 0f32;
                    for kk in 0..4 {
                        let dy = f32::from_le_bytes(yb[4 * kk..4 * kk + 4].try_into().unwrap());
                        let dot = |lo: usize| -> i32 {
                            (lo..lo + 16)
                                .map(|e| q(kk * 32 + e) * i32::from(yb[16 + kk * 32 + e] as i8))
                                .sum()
                        };
                        let (c0, c1) = (dot(0), dot(16));
                        let sc0 = f32::from(sc[h * 8 + kk * 2] as i8);
                        let sc1 = f32::from(sc[h * 8 + kk * 2 + 1] as i8);
                        tmp = (c0 as f32 * sc0 + c1 as f32 * sc1).mul_add(dy, tmp);
                    }
                    acc = tmp.mul_add(d, acc);
                }
            }
            out[col * n_rows + row] = acc;
        }
    }
    out
}

/// [`mmq_q8_0_twin`] for MXFP4: each block's codes as ggml's doubled
/// E2M1 integers under `e8m0_to_fp32(e) * 0.5`, Q8_0's order otherwise.
#[cfg(test)]
pub(crate) fn mmq_mxfp4_twin(
    w: &[u8],
    y: &[u8],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    ncols_pad: usize,
) -> Vec<f32> {
    const KV_X2: [i32; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let n_blocks = n_cols / 32;
    let row_bytes = n_blocks * 17;
    let mut out = vec![0f32; batch * n_rows];
    for col in 0..batch {
        for row in 0..n_rows {
            let mut acc = 0f32;
            for blk in 0..n_blocks.div_ceil(4) * 4 {
                let (dw, wq): (f32, Vec<i32>) = if blk < n_blocks {
                    let b = &w[row * row_bytes + blk * 17..row * row_bytes + blk * 17 + 17];
                    let e = u32::from(b[0]);
                    let scale = if e == 0 {
                        f32::from_bits(0x0040_0000)
                    } else {
                        f32::from_bits(e << 23)
                    };
                    (
                        scale * 0.5,
                        (0..32)
                            .map(|v| KV_X2[((b[1 + v % 16] >> (4 * (v / 16))) & 15) as usize])
                            .collect(),
                    )
                } else {
                    (0.0, vec![0; 32])
                };
                let yb = &y[((blk / 4) * ncols_pad + col) * quantize::BLOCK_BYTES..];
                let sub = blk % 4;
                let dy = f32::from_le_bytes(yb[4 * sub..4 * sub + 4].try_into().unwrap());
                let c: i32 = (0..32)
                    .map(|e| wq[e] * i32::from(yb[16 + sub * 32 + e] as i8))
                    .sum();
                acc = (c as f32 * dw).mul_add(dy, acc);
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
    /// The MXFP4 twin, held to the f32 reference the same way, over a K
    /// that is not a whole step.
    #[test]
    fn the_mxfp4_twin_is_the_gemm_within_the_q8_1_rounding() {
        let (n_rows, n_cols, batch) = (7usize, 2880usize, 3usize);
        let w = crate::mul_mm_ref::fixtures::weights(&crate::mul_mm::MXFP4, n_rows, n_cols, 23);
        let x: Vec<f32> = (0..batch * n_cols)
            .map(|i| ((i as f32) * 0.31).sin() * (1u32 << ((i / 32) % 5)) as f32)
            .collect();
        let ncols_pad = kernel::ncols_pad(batch);
        let y = quantize::quantize_twin(&x, None, n_cols, batch, ncols_pad, quantize::Layout::D4);
        let got = mmq_mxfp4_twin(&w, &y, n_rows, n_cols, batch, ncols_pad);
        let want = crate::mul_mm_ref::mul_mm_reference(
            &crate::mul_mm::MXFP4,
            &w,
            &x,
            n_rows,
            n_cols,
            batch,
            n_cols / 32 * 17,
        )
        .unwrap();
        let l1 = crate::mul_mm_ref::product_l1(
            &crate::mul_mm::MXFP4,
            &w,
            &x,
            n_rows,
            n_cols,
            batch,
            n_cols / 32 * 17,
        )
        .unwrap();
        for (i, ((g, r), l)) in got.iter().zip(&want).zip(&l1).enumerate() {
            assert!(
                (g - r).abs() <= 5e-3 * l,
                "[{i}]: twin {g} reference {r} (l1 {l})"
            );
        }
    }

    /// The Q6_K twin, held to the f32 reference the same way.
    #[test]
    fn the_q6_k_twin_is_the_gemm_within_the_q8_1_rounding() {
        let (n_rows, n_cols, batch) = (7usize, 512usize, 3usize);
        let w = crate::mul_mm_ref::fixtures::weights(&crate::mul_mm::Q6_K, n_rows, n_cols, 17);
        let x: Vec<f32> = (0..batch * n_cols)
            .map(|i| ((i as f32) * 0.29).sin() * (1u32 << ((i / 32) % 5)) as f32)
            .collect();
        let ncols_pad = kernel::ncols_pad(batch);
        let y = quantize::quantize_twin(&x, None, n_cols, batch, ncols_pad, quantize::Layout::D4);
        let got = mmq_q6_k_twin(&w, &y, n_rows, n_cols, batch, ncols_pad);
        let want = crate::mul_mm_ref::mul_mm_reference(
            &crate::mul_mm::Q6_K,
            &w,
            &x,
            n_rows,
            n_cols,
            batch,
            n_cols / 256 * 210,
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
