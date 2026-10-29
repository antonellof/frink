//! The kernel-vs-CPU check the `Q2_K`, `Q3_K` and generated kernels run
//! (#405): one harness, so every kind is held to the same shapes and
//! the same bounds.
//!
//! The reference is `frink_quant`'s dequant (the CPU path, itself held
//! to ggml's `dequantize_row_*`), dotted in f64. Each output is held to
//! a bound derived from what the kernel is allowed to lose, as a
//! fraction of `mag = sum |w x|` over its row:
//!
//! - the matvec is f32 throughout, so only accumulation order differs:
//!   [`MATVEC_REL`];
//! - the GEMM stages BOTH operands as `half` (ggml's `mul_mm` does the
//!   same), so each product carries at most `2u + u^2` relative error
//!   with `u = 2^-11`, which is `9.77e-4`: [`GEMM_REL`].
//!
//! A wrong mask, shift, scale index or block stride moves an output by
//! a large fraction of `mag`, far outside either bound.

use crate::gpu::MetalError;

/// The f32 matvec's bound, as a fraction of `sum |w x|`.
const MATVEC_REL: f64 = 1e-5;
/// The half-staged GEMM's bound: `2u + u^2`, `u = 2^-11`, rounded up.
const GEMM_REL: f64 = 1.0e-3;

type MatvecFn = fn(&[u8], &[f32], usize, usize) -> Result<Vec<f32>, MetalError>;
type GemmFn = fn(&[u8], &[f32], usize, usize, usize) -> Result<Vec<f32>, MetalError>;

/// One kind under test.
pub(crate) struct CheckKind {
    pub name: &'static str,
    pub block_bytes: usize,
    /// Elements per block (256 for the K-quants and the IQ and trit
    /// formats, 32 for the legacy ones).
    pub block_elems: usize,
    /// The CPU dequant of whole blocks.
    pub dequant: fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError>,
    /// `n_blocks` blocks of realistic bytes from a seed.
    pub blocks: fn(u32, usize) -> Vec<u8>,
    pub matvec: MatvecFn,
    pub gemm: GemmFn,
}

/// Deterministic pseudo-random bytes.
pub(crate) fn pseudo_bytes(seed: u32, len: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        })
        .collect()
}

/// Matvec shapes `(rows, cols)`. The row counts are not multiples of
/// either kernel's rows per threadgroup (8 for Q2_K, 4 for Q3_K), and one
/// is a single row, so the ragged-group guard and the row clamp are
/// reached. The block counts per row are 1, 3 and 5: the kernels stride
/// blocks four lane groups apart, so 1 leaves three idle and 3 and 5
/// end on a partial stride. A K-quant row is whole 256-weight blocks
/// by construction, so no K off that grid is legal.
const MATVEC_SHAPES: &[(usize, usize)] = &[(1, 256), (5, 768), (13, 1280), (67, 256), (130, 768)];

/// GEMM shapes `(rows, cols, batch)`: the exact-tile grid (the `_a`
/// pipeline) at one and 2x2 tiles, smaller than a tile in both
/// dimensions, and ragged edges past full tiles.
const GEMM_SHAPES: &[(usize, usize, usize)] = &[
    (64, 512, 32),
    (128, 512, 64),
    (9, 256, 7),
    (70, 768, 33),
    (13, 1280, 5),
];

/// The CPU reference for `rows x cols` weights times each activation:
/// `(value, sum |w x|)` per output, `[batch, rows]`.
fn reference(
    kind: &CheckKind,
    weights: &[u8],
    rows: usize,
    cols: usize,
    xs: &[f32],
) -> Vec<(f64, f64)> {
    let w = (kind.dequant)(weights).expect("whole blocks");
    assert_eq!(w.len(), rows * cols);
    xs.chunks_exact(cols)
        .flat_map(|x| {
            w.chunks_exact(cols).map(move |row| {
                row.iter().zip(x).fold((0f64, 0f64), |(s, m), (&a, &b)| {
                    let p = f64::from(a) * f64::from(b);
                    (s + p, m + p.abs())
                })
            })
        })
        .collect()
}

fn activations(cols: usize, batch: usize) -> Vec<f32> {
    (0..batch * cols)
        .map(|i| ((i % cols + 23 * (i / cols)) as f32 * 0.027).sin())
        .collect()
}

/// Worst `|got - want|` and worst `|got - want| / mag`, asserting the
/// bound on every output.
fn compare(label: &str, got: &[f32], want: &[(f64, f64)], rel: f64) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "{label}: output length");
    let (mut worst_abs, mut worst_rel) = (0f64, 0f64);
    for (i, (&g, &(w, mag))) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs();
        let bound = rel * mag.max(1e-3);
        assert!(
            err <= bound,
            "{label} output {i}: kernel {g} CPU {w} |err| {err:e} over the bound {bound:e} (mag {mag:e})"
        );
        worst_abs = worst_abs.max(err);
        worst_rel = worst_rel.max(err / mag.max(1e-3));
    }
    (worst_abs, worst_rel)
}

/// The matvec against the CPU dequant, over [`MATVEC_SHAPES`].
pub(crate) fn check_matvec(kind: &CheckKind) {
    for (s, &(rows, cols)) in MATVEC_SHAPES.iter().enumerate() {
        let n_blocks = cols / kind.block_elems;
        let row_bytes = n_blocks * kind.block_bytes;
        let weights = (kind.blocks)(s as u32 + 1, rows * n_blocks);
        let x = activations(cols, 1);
        let got = (kind.matvec)(&weights, &x, rows, row_bytes).expect("matvec launch");
        let want = reference(kind, &weights, rows, cols, &x);
        let label = format!("{} matvec {rows}x{cols}", kind.name);
        let (abs, rel) = compare(&label, &got, &want, MATVEC_REL);
        eprintln!("{label}: max |err| {abs:.3e}, max |err|/sum|wx| {rel:.3e}");
    }
}

/// The simdgroup GEMM against the CPU dequant, over [`GEMM_SHAPES`].
pub(crate) fn check_gemm(kind: &CheckKind) {
    for (s, &(rows, cols, batch)) in GEMM_SHAPES.iter().enumerate() {
        let n_blocks = cols / kind.block_elems;
        let row_bytes = n_blocks * kind.block_bytes;
        let weights = (kind.blocks)(s as u32 + 101, rows * n_blocks);
        let xs = activations(cols, batch);
        let got = (kind.gemm)(&weights, &xs, rows, row_bytes, batch).expect("gemm launch");
        let want = reference(kind, &weights, rows, cols, &xs);
        let label = format!("{} gemm {rows}x{cols}x{batch}", kind.name);
        let (abs, rel) = compare(&label, &got, &want, GEMM_REL);
        eprintln!("{label}: max |err| {abs:.3e}, max |err|/sum|wx| {rel:.3e}");
    }
}
