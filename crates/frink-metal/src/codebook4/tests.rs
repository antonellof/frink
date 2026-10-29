//! The IQ4_NL and MXFP4 kernels against `frink_quant`'s CPU dequant,
//! on real hardware (`--ignored`), at shapes that are not multiples of
//! anything the kernels tile by.

use super::*;

/// One of the two formats, as the tests build and check it.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Iq4Nl,
    Mxfp4,
}

impl Kind {
    const ALL: [Kind; 2] = [Kind::Iq4Nl, Kind::Mxfp4];

    fn block_bytes(self) -> usize {
        match self {
            Kind::Iq4Nl => frink_quant::IQ4_NL_BLOCK_BYTES,
            Kind::Mxfp4 => frink_quant::MXFP4_GGUF_BLOCK_BYTES,
        }
    }

    fn dequant(self, bytes: &[u8]) -> Vec<f32> {
        match self {
            Kind::Iq4Nl => frink_quant::dequant_iq4_nl(bytes),
            Kind::Mxfp4 => frink_quant::dequant_mxfp4_gguf(bytes),
        }
        .expect("whole blocks")
    }

    fn matvec(self, w: &[u8], x: &[f32], rows: usize, rb: usize) -> Vec<f32> {
        match self {
            Kind::Iq4Nl => launch_iq4_nl_matvec(w, x, rows, rb),
            Kind::Mxfp4 => launch_mxfp4_matvec(w, x, rows, rb),
        }
        .expect("matvec launch")
    }

    fn gemm(self, w: &[u8], x: &[f32], rows: usize, rb: usize, batch: usize) -> Vec<f32> {
        match self {
            Kind::Iq4Nl => launch_iq4_nl_mul_mm_sg(w, x, rows, rb, batch),
            Kind::Mxfp4 => launch_mxfp4_mul_mm_sg(w, x, rows, rb, batch),
        }
        .expect("mul_mm_sg launch")
    }
}

/// Every code byte random, every scale realistic: an IQ4_NL `d` of
/// 0.004..0.012, an MXFP4 exponent of 2^-9..2^-4. Random scale bytes
/// decode to magnitudes no quantizer writes and overflow the GEMM's
/// `half` tile, which tests a regime that cannot occur. The second block
/// of every MXFP4 matrix carries `e = 0` and the third `e = 1`, so the
/// subnormal arm of the E8M0 decode is read. (The GPU flushes the
/// `2^-127` products to zero; the error that leaves is under 1e-37.)
fn blocks(kind: Kind, rows: usize, blocks_per_row: usize, seed: u32) -> Vec<u8> {
    let bb = kind.block_bytes();
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = move || {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (state >> 16) as u8
    };
    let mut w: Vec<u8> = (0..rows * blocks_per_row * bb).map(|_| next()).collect();
    for b in 0..rows * blocks_per_row {
        let at = b * bb;
        match kind {
            Kind::Iq4Nl => {
                let d = half::f16::from_f32(0.004 + 0.001 * (b % 9) as f32);
                w[at..at + 2].copy_from_slice(&d.to_le_bytes());
            }
            Kind::Mxfp4 => {
                w[at] = match b {
                    1 => 0,
                    2 => 1,
                    _ => 118 + (b % 6) as u8,
                };
            }
        }
    }
    w
}

fn activations(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + salt * 13) as f32 * 0.031).sin())
        .collect()
}

/// `W x` per activation, in f64, from the CPU dequant.
fn reference(kind: Kind, w: &[u8], x_batch: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let dense = kind.dequant(w);
    assert_eq!(dense.len(), rows * cols);
    let mut out = Vec::new();
    for x in x_batch.chunks_exact(cols) {
        for row in dense.chunks_exact(cols) {
            let acc: f64 = row
                .iter()
                .zip(x)
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            out.push(acc as f32);
        }
    }
    out
}

/// Worst absolute error, and the tolerance it is held to: `rel` of the
/// largest reference magnitude, floored at `rel` itself.
fn check(label: &str, got: &[f32], want: &[f32], rel: f32) -> f32 {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
    let tol = rel * scale;
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let e = (g - w).abs();
        assert!(
            e.is_finite() && e <= tol,
            "{label} idx {i}: gpu {g}, cpu {w} (tol {tol})"
        );
        worst = worst.max(e);
    }
    println!("{label}: max |gpu - cpu| = {worst:.3e} (tol {tol:.1e})");
    worst
}

/// Rows not a multiple of the four a threadgroup covers; blocks per row
/// that are fewer than, equal to, and not a multiple of the sixteen
/// lane pairs a simdgroup strides by.
const MATVEC_SHAPES: &[(usize, usize)] = &[
    (1, 32),
    (3, 96),
    (7, 544),
    (16, 512),
    (67, 1056),
    (130, 2880),
];

#[test]
#[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
fn codebook4_matvecs_match_the_cpu_dequant() {
    for kind in Kind::ALL {
        for (s, &(rows, cols)) in MATVEC_SHAPES.iter().enumerate() {
            let bpr = cols / 32;
            let w = blocks(kind, rows, bpr, 17 + s as u32);
            let x = activations(cols, s);
            let got = kind.matvec(&w, &x, rows, bpr * kind.block_bytes());
            let want = reference(kind, &w, &x, rows, cols);
            check(&format!("{kind:?} matvec {rows}x{cols}"), &got, &want, 1e-5);
        }
    }
}

/// Exact tile grids (64x32 and 2x2 of them) take the `_a` pipeline, the
/// rest the ragged-edge store; a batch under a tile, and a K of one
/// block.
const GEMM_SHAPES: &[(usize, usize, usize)] = &[
    (64, 512, 32),
    (128, 512, 64),
    (9, 96, 7),
    (5, 32, 4),
    (70, 544, 33),
    (131, 2880, 45),
];

/// The GEMM stages weights and activations as `half` (ggml's
/// `kernel_mul_mm_*_f32` does the same), so it is held to 1e-3 of the
/// output's magnitude rather than the matvec's 1e-5. Measured on an M2
/// Pro the worst is 3e-4 of it.
#[test]
#[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
fn codebook4_gemms_match_the_cpu_dequant() {
    for kind in Kind::ALL {
        for (s, &(rows, cols, batch)) in GEMM_SHAPES.iter().enumerate() {
            let bpr = cols / 32;
            let w = blocks(kind, rows, bpr, 41 + s as u32);
            let x: Vec<f32> = (0..batch)
                .flat_map(|b| activations(cols, s * 64 + b))
                .collect();
            let got = kind.gemm(&w, &x, rows, bpr * kind.block_bytes(), batch);
            let want = reference(kind, &w, &x, rows, cols);
            check(
                &format!("{kind:?} gemm {rows}x{cols}x{batch}"),
                &got,
                &want,
                1e-3,
            );
        }
    }
}

/// The fused prefill stacks bind the `_f16` entries (half activations)
/// and, on an exact grid, their `_a` siblings. Every one of them has to
/// exist in the GEMM translation unit, or the first prefill that reaches
/// it fails to build its pipeline.
#[test]
#[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
fn every_codebook4_gemm_entry_compiles() {
    let shared = crate::gpu::shared_metal().expect("Metal device");
    for kind in ["IQ4_NL", "MXFP4"] {
        let (base, _, _) = crate::gpu::mul_mm_sg_meta(kind).expect("GEMM row");
        let f16 = crate::gpu::mul_mm_sg_f16_fn(base).expect("f16 sibling");
        for name in [base, f16] {
            let a = crate::gpu::mul_mm_sg_aligned_fn(name).expect("aligned sibling");
            for f in [name, a] {
                if let Err(e) = crate::gpu::warm_mul_mm_sg_pipeline(&shared.device, f) {
                    panic!("{kind}: {f} does not build: {e}");
                }
            }
        }
    }
}
