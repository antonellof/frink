//! The scalar twin of the CUDA `mul_mm` kernel: the same GEMM, executed
//! on the host, one emulated threadblock at a time.
//!
//! # Why this exists
//!
//! There is no NVIDIA GPU in the environment this kernel was written in,
//! so the kernel itself cannot be run. This repo's answer to an
//! unrunnable arm is the same one every `unsafe` SIMD path here already
//! gets: a scalar twin implementing identical arithmetic, checked
//! against an independent reference on data the host can compute.
//!
//! [`mul_mm_reference`] is not "a matmul that should give the same
//! answer". It is a **transcription of the kernel**: same tile
//! enumeration, same thread-to-micro-tile mapping, same out-of-range row
//! clamp, same zero-fill of absent tokens, same k-ascending accumulation
//! order, same shared-memory layout. Barriers are modelled by running
//! every thread's load phase before any thread's compute phase, which is
//! exactly what `__syncthreads()` guarantees. Read it beside `BODY_SRC`
//! in [`crate::mul_mm`]; a line that does not correspond to a line there
//! is a bug in one of them.
//!
//! That buys three things a device would otherwise have to: that the
//! index arithmetic addresses the elements it means to, that the
//! per-kind unpack decodes the format, and that partial tiles neither
//! write out of bounds nor drop rows.
//!
//! It does **not**, by itself, establish that the *emitted CUDA C* says
//! the same thing this file says -- the two are hand-transcribed from
//! each other. `tools/mul_mm_host_check/run.sh` closes that gap by
//! compiling the real kernel text and running it against this twin on
//! the host (bit-exact as of 2026-09-01). What remains for hardware:
//! that NVRTC accepts the source, that the barriers survive a real warp
//! scheduler, that the launch config is valid, and what it costs. Those
//! stay unproven until someone runs
//! `cargo test -p frink-cuda --features cuda -- --ignored`.

use crate::mul_mm::{
    grid_dims, validate_shape, MulMmKind, MulMmUnsupported, BK, BM, BN, SUB, THREADS, TM, TN,
};

/// Host emulation of the CUDA `mul_mm` kernel.
///
/// Returns `batch * n_rows` floats laid out as `out[token * n_rows +
/// row]`, the layout `WeightMatrix::apply_batch` already produces.
///
/// This is a correctness reference, not a fast path: it is the kernel's
/// arithmetic, thread by thread, and is roughly as slow as that sounds.
pub fn mul_mm_reference(
    kind: &MulMmKind,
    weights: &[u8],
    x_batch: &[f32],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MulMmUnsupported> {
    validate_shape(
        kind,
        weights.len(),
        x_batch.len(),
        n_rows,
        n_cols,
        batch,
        row_bytes,
    )?;

    let mut dst = vec![0f32; batch * n_rows];
    let (grid_x, grid_y) = grid_dims(n_rows, batch);
    for by in 0..grid_y {
        for bx in 0..grid_x {
            emulate_block(
                kind, weights, x_batch, &mut dst, n_rows, n_cols, batch, row_bytes, bx, by,
            );
        }
    }
    Ok(dst)
}

/// `sum_k |W[row][k] * x[token][k]|` for every output element, in the
/// same `dst[token * n_rows + row]` layout as [`mul_mm_reference`]: the
/// L1 norm of the products each result sums, which is what bounds the
/// rounding error of any GEMM that rounds its operands (the tensor-core
/// body rounds both to f16). Dequantizes with the kind's own twin, so
/// it is the reference's arithmetic and not a second reading of the
/// format.
#[allow(clippy::too_many_arguments)]
pub fn product_l1(
    kind: &MulMmKind,
    weights: &[u8],
    x_batch: &[f32],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MulMmUnsupported> {
    validate_shape(
        kind,
        weights.len(),
        x_batch.len(),
        n_rows,
        n_cols,
        batch,
        row_bytes,
    )?;
    let subs_per_row = n_cols / SUB;
    let nl = kind.nl();
    let mut out = vec![0f32; batch * n_rows];
    let mut reg = [0f32; SUB];
    for row in 0..n_rows {
        let rp = &weights[row * row_bytes..(row + 1) * row_bytes];
        for sub in 0..subs_per_row {
            let block = &rp[(sub / nl) * kind.block_bytes..(sub / nl + 1) * kind.block_bytes];
            (kind.dequant_twin)(block, sub % nl, &mut reg);
            for token in 0..batch {
                let xs = &x_batch[token * n_cols + sub * SUB..token * n_cols + (sub + 1) * SUB];
                let acc: f32 = reg.iter().zip(xs).map(|(w, x)| (w * x).abs()).sum();
                out[token * n_rows + row] += acc;
            }
        }
    }
    Ok(out)
}

/// One threadblock of the kernel: `blockIdx = (bx, by)`, `THREADS`
/// threads, `sa`/`sb` standing in for the two `__shared__` tiles.
#[allow(clippy::too_many_arguments)] // The kernel's own parameter list plus its block index; bundling it would only move the same values behind a name that says less.
fn emulate_block(
    kind: &MulMmKind,
    src0: &[u8],
    src1: &[f32],
    dst: &mut [f32],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
    bx: usize,
    by: usize,
) {
    let nl = kind.nl();
    let r0 = by * BM;
    let r1 = bx * BN;

    // `__shared__ float sa[BK][BM]` / `sb[BK][BN]`.
    let mut sa = vec![0f32; BK * BM];
    let mut sb = vec![0f32; BK * BN];
    // Per-thread `float acc[TN][TM]`, all registers of the block at once.
    let mut acc = vec![0f32; THREADS * TN * TM];

    let mut k0 = 0usize;
    while k0 < n_cols {
        // --- load phase (every thread, then the barrier) ---
        for tid in 0..THREADS {
            if tid < BM * (BK / SUB) {
                let lr = tid / (BK / SUB);
                let ils = tid % (BK / SUB);
                let mut row = r0 + lr;
                if row >= n_rows {
                    row = n_rows - 1;
                }
                let rp = &src0[row * row_bytes..(row + 1) * row_bytes];
                let sub = (k0 / SUB) + ils;
                let xb = &rp[(sub / nl) * kind.block_bytes..];
                let mut reg = [0f32; SUB];
                (kind.dequant_twin)(xb, sub % nl, &mut reg);
                for (i, v) in reg.iter().enumerate() {
                    sa[(SUB * ils + i) * BM + lr] = *v;
                }
            }

            let mut idx = tid;
            while idx < BK * BN {
                let j = idx / BK;
                let kk = idx % BK;
                let col = r1 + j;
                sb[kk * BN + j] = if col < batch {
                    src1[col * n_cols + k0 + kk]
                } else {
                    0.0
                };
                idx += THREADS;
            }
        }

        // --- compute phase (after `__syncthreads()`) ---
        for tid in 0..THREADS {
            let tx = tid % (BM / TM);
            let ty = tid / (BM / TM);
            let acc = &mut acc[tid * TN * TM..(tid + 1) * TN * TM];
            for kk in 0..BK {
                let mut a = [0f32; TM];
                let mut b = [0f32; TN];
                for (m, slot) in a.iter_mut().enumerate() {
                    *slot = sa[kk * BM + tx * TM + m];
                }
                for (n, slot) in b.iter_mut().enumerate() {
                    *slot = sb[kk * BN + ty * TN + n];
                }
                for n in 0..TN {
                    for m in 0..TM {
                        acc[n * TM + m] += a[m] * b[n];
                    }
                }
            }
        }

        k0 += BK;
    }

    // --- store ---
    for tid in 0..THREADS {
        let tx = tid % (BM / TM);
        let ty = tid / (BM / TM);
        let acc = &acc[tid * TN * TM..(tid + 1) * TN * TM];
        for n in 0..TN {
            let col = r1 + ty * TN + n;
            if col >= batch {
                continue;
            }
            for m in 0..TM {
                let row = r0 + tx * TM + m;
                if row < n_rows {
                    dst[col * n_rows + row] = acc[n * TM + m];
                }
            }
        }
    }
}

/// Weight fixtures every `mul_mm` test builds on, in ONE place.
///
/// Three tests need "a plausible quantized weight matrix for this
/// kind": the dequant-twin comparison in [`crate::mul_mm`], the GEMM
/// comparison below, and the `#[ignore]`d hardware test in
/// [`crate::mul_mm_launch`]. Each used to carry its own builder, keyed
/// by kind, and adding a format meant remembering all three -- with a
/// `_ =>` arm in one of them silently pinning the wrong bytes for the
/// next row. They share this module instead.
///
/// The bytes are deliberately arbitrary rather than the output of a
/// real quantizer: `frink_quant` has no encoder for most of these
/// formats, and a random nibble is a legal code in every one of them.
/// Only the SCALE fields are constrained, and only to keep them finite
/// -- a random `half` is Inf or NaN often enough to be the usual
/// outcome, and a matrix of NaN proves nothing about an unpack.
#[cfg(test)]
pub(crate) mod fixtures {
    use crate::mul_mm::MulMmKind;

    /// Deterministic pseudo-random bytes, the generator shape
    /// `gpu.rs`'s K-quant fixtures already use.
    pub(crate) fn pseudo_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    /// Forces the f16 at `at` to a finite value while keeping its sign
    /// and mantissa random.
    ///
    /// The exponent field goes to 11, so the scale lands near 2^-4 with
    /// a random sign and a random fraction. Pinning the whole `half` to
    /// one constant would make every block of the matrix identical in
    /// its scale, which hides a kernel that reads the scale of the
    /// wrong block.
    fn finite_f16(block: &mut [u8], at: usize) {
        let bits = u16::from(block[at]) | (u16::from(block[at + 1]) << 8);
        let bits = (bits & 0x83FF) | (11 << 10);
        block[at] = bits as u8;
        block[at + 1] = (bits >> 8) as u8;
    }

    /// Constrains one block's scale fields for `kind`, leaving every
    /// byte that carries quant data alone.
    ///
    /// **No catch-all arm.** A kind added to `KINDS` without a row here
    /// panics with its own name the first time any of the three tests
    /// touches it, rather than inheriting a guess. `IQ4_XS`'s per-group
    /// 6-bit scales at bytes 2..8 are deliberately left random: they
    /// are assembled from two places and that assembly is the part of
    /// the format most likely to be transcribed wrong.
    pub(crate) fn pin_finite_scales(kind: &MulMmKind, block: &mut [u8]) {
        match kind.name {
            // `half d` at the head, nothing else.
            "Q8_0" | "Q4_0" | "Q5_0" | "IQ4_NL" | "IQ4_XS" => finite_f16(block, 0),
            // `half d`, then `half m`.
            "Q4_1" | "Q5_1" => {
                finite_f16(block, 0);
                finite_f16(block, 2);
            }
            // The trit formats keep their one `half d` LAST.
            "TQ1_0" => finite_f16(block, 52),
            "PTQ1_0" => finite_f16(block, 26),
            // The IQ grid kinds keep `half d` at the head.
            "IQ2_XXS" | "IQ2_XS" | "IQ2_S" | "IQ3_XXS" | "IQ3_S" | "IQ1_S" => finite_f16(block, 0),
            // IQ1_M has no `half d`: its scale is the top nibbles of the
            // four scale words at 48..56, sc[3]'s at bits 12..16. Byte 55
            // carries the f16's sign and exponent bits 12..14; byte 53's
            // top two bits are exponent bits 10..11. Exponent 11, as
            // `finite_f16` pins, with the sign and every sub-scale left
            // random.
            "IQ1_M" => {
                block[53] |= 0xC0;
                block[55] = (block[55] & 0x8F) | 0x20;
            }
            // `half d`, then `half dmin`.
            "Q4_K" | "Q5_K" => {
                finite_f16(block, 0);
                finite_f16(block, 2);
            }
            // Q6_K keeps its `half d` at the END of the block.
            "Q6_K" => finite_f16(block, 208),
            // Q2_K puts its scales FIRST and both halves last.
            "Q2_K" => {
                finite_f16(block, 80);
                finite_f16(block, 82);
            }
            // Q3_K has one `half d`, also at the end. Its twelve scale
            // bytes at 96..108 stay random: they are a six-bit
            // four-arm assembly and that assembly is the part of the
            // format most likely to be transcribed wrong.
            "Q3_K" => finite_f16(block, 108),
            // One E8M0 byte, not an f16. 123..=130 is 2^-4 .. 2^3.
            "MXFP4" => block[0] = 123 + (block[0] & 7),
            other => panic!("{other}: no scale-pinning rule; add one beside the KINDS row"),
        }
    }

    /// One block: random bytes with `kind`'s scale fields made finite.
    pub(crate) fn block(kind: &MulMmKind, seed: u32) -> Vec<u8> {
        let mut b = pseudo_bytes(seed, kind.block_bytes);
        pin_finite_scales(kind, &mut b);
        b
    }

    /// A whole `n_rows x n_cols` weight matrix in `kind`'s format.
    pub(crate) fn weights(kind: &MulMmKind, n_rows: usize, n_cols: usize, seed: u32) -> Vec<u8> {
        assert!(
            n_cols.is_multiple_of(kind.block_elems),
            "{}: {n_cols} columns is not a whole number of blocks",
            kind.name
        );
        let blocks = (n_cols / kind.block_elems) * n_rows;
        let mut out = Vec::with_capacity(blocks * kind.block_bytes);
        for b in 0..blocks {
            out.extend_from_slice(&block(kind, seed.wrapping_add(b as u32 * 7919)));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul_mm::{f16_to_f32, kernel_src, KINDS, Q4_0, Q8_0};

    /// Deterministic pseudo-random bytes, the same generator shape
    /// `gpu.rs`'s K-quant fixtures already use.
    fn pseudo_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    /// Q8_0 weights produced by `frink_quant`'s real quantizer, so the
    /// bytes are a format a loader would actually hand the kernel.
    fn q8_0_matrix(n_rows: usize, n_cols: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for r in 0..n_rows {
            let row: Vec<f32> = (0..n_cols)
                .map(|i| (((r * n_cols + i) as f32) * 0.037).sin())
                .collect();
            out.extend(frink_quant::quantize_q8_0(&row));
        }
        out
    }

    /// Q4_0 has no quantizer in `frink_quant` (the format is load-only
    /// here), so build valid block bytes directly: an f16 scale that is
    /// exactly representable, then 16 nibble pairs.
    fn q4_0_matrix(n_rows: usize, n_cols: usize) -> Vec<u8> {
        let blocks = n_cols / 32;
        let mut out = Vec::new();
        for r in 0..n_rows {
            for b in 0..blocks {
                let scale = half::f16::from_f32(0.05 + ((r * blocks + b) % 13) as f32 * 0.01);
                out.extend_from_slice(&scale.to_le_bytes());
                let nibbles = pseudo_bytes((r * blocks + b) as u32 + 7, 16);
                out.extend_from_slice(&nibbles);
            }
        }
        out
    }

    fn activations(batch: usize, n_cols: usize) -> Vec<f32> {
        (0..batch * n_cols)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect()
    }

    /// An independent GEMM: dequantize with `frink_quant` (a different
    /// implementation, written for a different purpose, cross-validated
    /// against NumPy) and do a plain dot product. This is the thing the
    /// twin is checked against; if the twin's tiling or its unpack is
    /// wrong, these disagree.
    fn independent_gemm(
        dequant_row: impl Fn(&[u8]) -> Vec<f32>,
        weights: &[u8],
        x: &[f32],
        n_rows: usize,
        n_cols: usize,
        batch: usize,
        row_bytes: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; batch * n_rows];
        for r in 0..n_rows {
            let w = dequant_row(&weights[r * row_bytes..(r + 1) * row_bytes]);
            assert_eq!(w.len(), n_cols);
            for t in 0..batch {
                let xr = &x[t * n_cols..(t + 1) * n_cols];
                let mut acc = 0f32;
                for k in 0..n_cols {
                    acc += w[k] * xr[k];
                }
                out[t * n_rows + r] = acc;
            }
        }
        out
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let scale = w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol * scale,
                "{what}: element {i}: twin={g} reference={w}"
            );
        }
    }

    /// The f16 helper the kernels use is hand-written bit surgery, not a
    /// library call. Hold it against `half`, denormals and specials
    /// included.
    #[test]
    fn f16_twin_matches_half_crate_over_every_bit_pattern() {
        for bits in 0u32..=0xFFFF {
            let bits = bits as u16;
            let want = half::f16::from_bits(bits).to_f32();
            let got = f16_to_f32(bits);
            if want.is_nan() {
                assert!(got.is_nan(), "bits {bits:#06x}: want NaN, got {got}");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "bits {bits:#06x}: got {got}, want {want}"
                );
            }
        }
    }

    /// The single most likely porting error: `il` selecting the wrong 16
    /// elements, or the right 16 in the wrong order. Decode a super-block
    /// sub-block by sub-block and require it to equal `frink_quant`'s
    /// whole-block dequantization, element for element.
    #[test]
    fn q8_0_sub_block_twin_reconstructs_the_block_in_order() {
        let row: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.31).sin()).collect();
        let bytes = frink_quant::quantize_q8_0(&row);
        let want = frink_quant::dequant_q8_0(&bytes).unwrap();
        for (blk, chunk) in bytes.chunks(Q8_0.block_bytes).enumerate() {
            for il in 0..Q8_0.nl() {
                let mut reg = [0f32; SUB];
                (Q8_0.dequant_twin)(chunk, il, &mut reg);
                for (i, got) in reg.iter().enumerate() {
                    let want = want[blk * Q8_0.block_elems + il * SUB + i];
                    assert_eq!(*got, want, "block {blk} il {il} elem {i}");
                }
            }
        }
    }

    #[test]
    fn q4_0_sub_block_twin_reconstructs_the_block_in_order() {
        let bytes = q4_0_matrix(1, 64);
        let want = frink_quant::dequant_q4_0(&bytes).unwrap();
        for (blk, chunk) in bytes.chunks(Q4_0.block_bytes).enumerate() {
            for il in 0..Q4_0.nl() {
                let mut reg = [0f32; SUB];
                (Q4_0.dequant_twin)(chunk, il, &mut reg);
                for (i, got) in reg.iter().enumerate() {
                    let want = want[blk * Q4_0.block_elems + il * SUB + i];
                    // llama computes `d*q + (-8*d)`, frink_quant
                    // computes `(q-8)*d`; the two differ by at most one
                    // fp32 rounding of the same value.
                    assert!(
                        (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                        "block {blk} il {il} elem {i}: twin={got} reference={want}"
                    );
                }
            }
        }
    }

    /// Exact tiling: rows and batch both land on tile boundaries, so no
    /// clamp or zero-fill is exercised. If this fails, the K-loop or the
    /// micro-tile mapping is wrong.
    #[test]
    fn q8_0_gemm_twin_matches_independent_reference_on_exact_tiles() {
        let (n_rows, n_cols, batch) = (BM * 2, 128, BN);
        let row_bytes = (n_cols / 32) * Q8_0.block_bytes;
        let weights = q8_0_matrix(n_rows, n_cols);
        let x = activations(batch, n_cols);
        let got = mul_mm_reference(&Q8_0, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        let want = independent_gemm(
            |r| frink_quant::dequant_q8_0(r).unwrap(),
            &weights,
            &x,
            n_rows,
            n_cols,
            batch,
            row_bytes,
        );
        assert_close(&got, &want, 1e-5, "q8_0 exact tiles");
    }

    /// Partial tiles on both axes: `n_rows` is not a multiple of `BM`
    /// (exercises the out-of-range row clamp) and `batch` is not a
    /// multiple of `BN` (exercises the zero-filled B-tile columns and the
    /// bounds-checked store). This is where a kernel written without a
    /// device usually writes past the end.
    #[test]
    fn q8_0_gemm_twin_matches_independent_reference_on_partial_tiles() {
        let (n_rows, n_cols, batch) = (BM + 7, 96, BN + 5);
        let row_bytes = (n_cols / 32) * Q8_0.block_bytes;
        let weights = q8_0_matrix(n_rows, n_cols);
        let x = activations(batch, n_cols);
        let got = mul_mm_reference(&Q8_0, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        let want = independent_gemm(
            |r| frink_quant::dequant_q8_0(r).unwrap(),
            &weights,
            &x,
            n_rows,
            n_cols,
            batch,
            row_bytes,
        );
        assert_close(&got, &want, 1e-5, "q8_0 partial tiles");
    }

    /// batch = 1 is the decode shape. A GEMM that is only right for wide
    /// batches would still be wrong for every token after the prompt.
    #[test]
    fn q8_0_gemm_twin_matches_independent_reference_at_batch_one() {
        let (n_rows, n_cols, batch) = (37, 64, 1);
        let row_bytes = (n_cols / 32) * Q8_0.block_bytes;
        let weights = q8_0_matrix(n_rows, n_cols);
        let x = activations(batch, n_cols);
        let got = mul_mm_reference(&Q8_0, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        let want = independent_gemm(
            |r| frink_quant::dequant_q8_0(r).unwrap(),
            &weights,
            &x,
            n_rows,
            n_cols,
            batch,
            row_bytes,
        );
        assert_close(&got, &want, 1e-5, "q8_0 batch 1");
    }

    /// The second kind through the same GEMM body. This is the test that
    /// says the seam is a seam: nothing but the table row changed.
    #[test]
    fn q4_0_gemm_twin_matches_independent_reference_on_partial_tiles() {
        let (n_rows, n_cols, batch) = (BM + 3, 96, BN + 9);
        let row_bytes = (n_cols / 32) * Q4_0.block_bytes;
        let weights = q4_0_matrix(n_rows, n_cols);
        let x = activations(batch, n_cols);
        let got = mul_mm_reference(&Q4_0, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        let want = independent_gemm(
            |r| frink_quant::dequant_q4_0(r).unwrap(),
            &weights,
            &x,
            n_rows,
            n_cols,
            batch,
            row_bytes,
        );
        assert_close(&got, &want, 1e-5, "q4_0 partial tiles");
    }

    /// Every kind in the table through the same GEMM body, against a
    /// dequantize-then-dot built on `frink_quant`.
    ///
    /// This is the test that says the seam is a seam: nothing but the
    /// table row changes between Q8_0 and IQ4_XS, including for the
    /// codebook kinds, whose `dequant_src` indexes a `__constant__`
    /// array where the affine kinds multiply and add.
    ///
    /// It iterates [`KINDS`] rather than naming rows, so a format added
    /// to the table with no `frink_quant` dequant beside it fails here
    /// instead of shipping unchecked. Shapes: exact tiles, then partial
    /// on both axes (the out-of-range row clamp and the zero-filled
    /// B-tile columns), then a narrow batch.
    #[test]
    fn every_kind_gemm_twin_matches_the_independent_reference() {
        type Dequant = fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError>;
        let dequants: &[(&str, Dequant)] = &[
            ("Q8_0", frink_quant::dequant_q8_0),
            ("Q4_0", frink_quant::dequant_q4_0),
            ("Q5_0", frink_quant::dequant_q5_0),
            ("Q4_K", frink_quant::dequant_q4_k),
            ("Q5_K", frink_quant::dequant_q5_k),
            ("Q2_K", frink_quant::dequant_q2_k),
            ("Q3_K", frink_quant::dequant_q3_k),
            ("Q6_K", frink_quant::dequant_q6_k),
            ("IQ4_NL", frink_quant::dequant_iq4_nl),
            ("IQ4_XS", frink_quant::dequant_iq4_xs),
            ("MXFP4", frink_quant::dequant_mxfp4_gguf),
            ("Q4_1", frink_quant::dequant_q4_1),
            ("Q5_1", frink_quant::dequant_q5_1),
            ("TQ1_0", |b| {
                frink_quant::ternary::dequant_trits(b, frink_quant::ternary::TQ1_0)
            }),
            ("PTQ1_0", |b| {
                frink_quant::ternary::dequant_trits(b, frink_quant::ternary::PTQ1_0)
            }),
            ("IQ2_XXS", frink_quant::dequant_iq2_xxs),
            ("IQ2_XS", frink_quant::dequant_iq2_xs),
            ("IQ2_S", frink_quant::dequant_iq2_s),
            ("IQ3_XXS", frink_quant::dequant_iq3_xxs),
            ("IQ3_S", frink_quant::dequant_iq3_s),
            ("IQ1_S", frink_quant::dequant_iq1_s),
            ("IQ1_M", frink_quant::dequant_iq1_m),
        ];

        for k in KINDS {
            let (_, dequant) = dequants
                .iter()
                .find(|(name, _)| *name == k.name)
                .unwrap_or_else(|| panic!("{}: in KINDS with no frink_quant dequant", k.name));

            for (n_rows, cols, batch) in [(BM * 2, 128usize, BN), (BM + 7, 96, BN + 9), (37, 64, 3)]
            {
                // Rounded UP to a whole super-block: `validate_shape`
                // refuses anything else, and it is right to -- a
                // 128-column Q4_K row is not a Q4_K row.
                let n_cols = cols.next_multiple_of(k.block_elems);
                let row_bytes = (n_cols / k.block_elems) * k.block_bytes;
                let weights = fixtures::weights(k, n_rows, n_cols, 4242);
                let x = activations(batch, n_cols);

                let got =
                    mul_mm_reference(k, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
                let want = independent_gemm(
                    |r| dequant(r).unwrap(),
                    &weights,
                    &x,
                    n_rows,
                    n_cols,
                    batch,
                    row_bytes,
                );
                assert_close(
                    &got,
                    &want,
                    1e-5,
                    &format!("{} {n_rows}x{n_cols}x{batch}", k.name),
                );
            }
        }
    }

    /// Sabotage check for the tests above: perturbing one weight byte
    /// must move the twin's output. A GEMM test that passes on data it
    /// never reads is not a test.
    #[test]
    fn twin_output_depends_on_every_part_of_the_weight_matrix() {
        let (n_rows, n_cols, batch) = (BM + 7, 96, 3);
        let row_bytes = (n_cols / 32) * Q8_0.block_bytes;
        let base = q8_0_matrix(n_rows, n_cols);
        let x = activations(batch, n_cols);
        let want = mul_mm_reference(&Q8_0, &base, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        // Last row, last block, last quant: the corner a tiling bug is
        // most likely to skip.
        let mut poked = base.clone();
        let last = poked.len() - 1;
        poked[last] = poked[last].wrapping_add(64);
        let got = mul_mm_reference(&Q8_0, &poked, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        assert!(
            got.iter().zip(want.iter()).any(|(a, b)| a != b),
            "poking the last weight byte changed nothing -- the twin is not reading it"
        );
    }

    #[test]
    fn shape_validation_names_what_it_refuses() {
        // n_cols not a whole number of blocks.
        let err = mul_mm_reference(&Q8_0, &[], &[], 4, 48, 1, 51).unwrap_err();
        assert!(
            matches!(err, MulMmUnsupported::ColsNotTileAligned { .. }),
            "got {err:?}"
        );
        // row_bytes inconsistent with n_cols.
        let err = mul_mm_reference(&Q8_0, &[0; 999], &[0.0; 64], 4, 64, 1, 33).unwrap_err();
        assert!(
            matches!(err, MulMmUnsupported::RowBytesMismatch { .. }),
            "got {err:?}"
        );
        // Weight buffer short of n_rows * row_bytes.
        let err = mul_mm_reference(&Q8_0, &[0; 68], &[0.0; 64], 4, 64, 1, 68).unwrap_err();
        assert!(
            matches!(err, MulMmUnsupported::WeightsTooSmall { .. }),
            "got {err:?}"
        );
        // Activation buffer short of batch * n_cols.
        let err = mul_mm_reference(&Q8_0, &[0; 272], &[0.0; 64], 4, 64, 2, 68).unwrap_err();
        assert!(
            matches!(err, MulMmUnsupported::ActivationsTooSmall { .. }),
            "got {err:?}"
        );
        assert!(mul_mm_reference(&Q8_0, &[], &[], 0, 64, 1, 68).is_err());
    }

    /// The emitted translation unit has to define the entry point the
    /// launch path asks NVRTC for, and has to carry the geometry the
    /// twin assumed. This cannot prove the C compiles; it does catch a
    /// table row whose `fn_name` no longer matches its source.
    #[test]
    fn emitted_source_defines_the_entry_point_and_the_geometry() {
        for k in KINDS {
            let src = kernel_src(k);
            assert!(
                src.contains(&format!("__global__ void {}(", k.fn_name)),
                "{}: emitted source does not define {}",
                k.name,
                k.fn_name
            );
            assert!(
                src.contains("void frink_dequant_sub("),
                "{}: no unpack function",
                k.name
            );
            assert!(
                src.contains("float frink_f16_to_f32("),
                "{}: no f16 helper",
                k.name
            );
            assert!(
                src.contains(&format!("#define FX_BLOCK_BYTES {}\n", k.block_bytes)),
                "{}: block geometry not defined from the Rust constant",
                k.name
            );
            assert!(
                src.contains(&format!("#define FX_NL {}\n", k.nl())),
                "{}: sub-block count not defined from the Rust constant",
                k.name
            );
            // The twin's whole claim is that it walks the same tiles as
            // the kernel, and it reads BM/BN/BK/TM/TN from Rust while
            // the kernel reads them from these `#define`s. Two
            // structures that have to agree about one thing: pin them.
            // Without this, retuning a constant in Rust and leaving a
            // literal in the emitter gives a kernel that launches, and
            // a twin that agrees with itself about the wrong shape.
            for (name, value) in [
                ("FX_BM", BM),
                ("FX_BN", BN),
                ("FX_BK", BK),
                ("FX_TM", TM),
                ("FX_TN", TN),
                ("FX_THREADS", THREADS),
                ("FX_SUB", SUB),
            ] {
                assert!(
                    src.contains(&format!("#define {name} {value}\n")),
                    "{}: {name} is not emitted as the Rust constant {value}",
                    k.name
                );
            }
            // The `float4` loads are what keep a warp off eight banks.
            // A rewrite that quietly went back to scalar reads would be
            // correct and several times slower, which is the kind of
            // regression a numeric test cannot see.
            assert!(
                src.contains("const float4 v = *(const float4*)&sa[kk]")
                    && src.contains("const float4 v = *(const float4*)&sb[kk]"),
                "{}: the inner loop no longer loads its operands as float4",
                k.name
            );
            assert!(
                src.contains(&format!("#define FX_THREADS {THREADS}\n")),
                "{}: thread count not defined from the Rust constant",
                k.name
            );
            assert!(
                !src.contains("FX_FN_NAME"),
                "{}: unsubstituted name",
                k.name
            );
        }
        // Module-cache and entry-point collisions are checked pairwise
        // over the whole table by
        // `mul_mm::dequant_twin_tests::declared_block_geometry_is_the_gguf_geometry`,
        // which is where the table lives. Two named kinds here would
        // have been a third structure agreeing with the other two.
    }

    /// A batch of one is a matvec, and the matvec kernels are the arm
    /// that has actually run on hardware. The GEMM must not claim it.
    ///
    /// The threshold is DERIVED from the tile width, so this asserts the
    /// property rather than the number. It used to spell `8`, which was
    /// `BN / 4` at the time; retuning `BN` to 64 moved the threshold and
    /// turned a passing test red for no defect. A test that restates a
    /// derived constant is one more structure that has to agree with
    /// another one.
    #[test]
    fn single_token_dispatches_stay_on_the_matvec_path() {
        use crate::mul_mm::worth_a_gemm;
        assert!(!worth_a_gemm(0));
        assert!(!worth_a_gemm(1), "one token is a matvec at any tile width");

        // Monotone, so there is one threshold rather than a range of
        // shapes that flip back and forth.
        let threshold = (1..=4 * BN)
            .find(|b| worth_a_gemm(*b))
            .expect("some batch is worth a GEMM");
        assert!(threshold >= 2, "never a single token");
        assert!(
            threshold <= BN,
            "a full tile of tokens must be worth a GEMM, threshold {threshold} > BN {BN}"
        );
        for b in 1..4 * BN {
            assert_eq!(
                worth_a_gemm(b),
                b >= threshold,
                "batch {b} disagrees with threshold {threshold}"
            );
        }
    }

    /// A partial tile on both axes must still be *covered* by the grid:
    /// one row or one token past a tile boundary needs a second tile, not
    /// a silently dropped output. (The tile geometry itself is pinned by
    /// the `const _: () = assert!(..)` gates in `mul_mm`, which fail the
    /// build rather than a test.)
    #[test]
    fn the_grid_covers_a_partial_tile_on_both_axes() {
        assert_eq!(grid_dims(BM, BN), (1, 1));
        assert_eq!(grid_dims(BM + 1, BN + 1), (2, 2));
        assert_eq!(grid_dims(BM * 3, BN * 2 + 1), (3, 3));
    }
}
