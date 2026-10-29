//! The decode matvec built from a kind's GEMM dequant (#405): for the
//! kinds that have no hand-written matvec, the matvec is the same
//! `frink_dequant_sub` the GEMM's A-tile loader and the gather call,
//! so each kind's dequant is written once. Metal's `dq_matvec_impl`
//! is the same idea.
//!
//! Geometry is the block-per-row one every kernel in [`super::legacy`]
//! uses, so `gpu.rs`'s launch plan needs nothing new: one threadblock
//! per output row, [`THREADS`] threads striding the row's 16-element
//! sub-blocks, each thread summing its sub-blocks in order, then a
//! tree reduction in shared memory. [`matvec_twin`] is that schedule in
//! Rust, operation for operation, and `tools/matvec_host_check` runs
//! the emitted CUDA C against it bit for bit.
//!
//! The activation is f32, as in every CUDA matvec here. ggml's `mmvq`
//! quantizes it to Q8_1 for an integer dot; the f32 dot differs only
//! in accumulation, which ADR 0009 allows inside the reference spread.
//!
//! # UNVERIFIED ON HARDWARE
//!
//! No GPU has run these. `every_cuda_matvec_matches_the_cpu_reference`
//! (`--ignored`) is the check owed.

use crate::mul_mm::{kernel_src_with_body, MulMmKind, SUB};

/// Threads per row, the block-per-row launch `gpu.rs` gives every
/// kernel that has no coalesced replacement.
pub const THREADS: usize = 256;

/// The body. `FX_ROW_STEP` and `FX_X_STEP` are replaced by the kind's
/// block bytes and elements as literals, the spelling
/// `every_matvec_strides_by_the_real_block_geometry` reads back.
const BODY_SRC: &str = r#"
extern "C" __global__ void FX_FN_NAME(
    const unsigned char* weights,
    const float* x,
    float* out,
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;

    __shared__ float partial[256];
    float acc = 0.0f;

    const int n_sub = n_blocks_per_row * FX_NL;
    for (int s = (int)threadIdx.x; s < n_sub; s += (int)blockDim.x) {
        const unsigned char* block = row_ptr + (size_t)(s / FX_NL) * FX_ROW_STEP;
        const int base = (s / FX_NL) * FX_X_STEP;
        float reg[FX_SUB];
        frink_dequant_sub(block, s % FX_NL, reg);
        const float* xs = x + base + (s % FX_NL) * FX_SUB;
        float sub_acc = 0.0f;
#pragma unroll
        for (int i = 0; i < FX_SUB; i++) {
            sub_acc += reg[i] * xs[i];
        }
        acc += sub_acc;
    }

    partial[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}
"#;

/// The entry point for `kind`'s generated matvec: `<kind>_matvec`, the
/// name its [`super::MatvecKind`] row declares.
pub fn entry(kind: &MulMmKind) -> String {
    format!("{}_matvec", kind.name.to_ascii_lowercase())
}

/// The complete translation unit for `kind`, built once per process
/// and leaked, because `MatvecLaunch` carries the text as
/// `&'static str`.
pub fn matvec_src(kind: &MulMmKind) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static SRC: OnceLock<Mutex<HashMap<&'static str, &'static str>>> = OnceLock::new();
    let mut map = SRC
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    map.entry(kind.name).or_insert_with(|| {
        let body = BODY_SRC
            .replace("FX_FN_NAME", &entry(kind))
            .replace("FX_ROW_STEP;", &format!("{};", kind.block_bytes))
            .replace("FX_X_STEP;", &format!("{};", kind.block_elems));
        Box::leak(kernel_src_with_body(kind, &body).into_boxed_str())
    })
}

/// Scalar twin of the generated matvec for one row: thread `t` of
/// [`THREADS`] sums sub-blocks `t, t + THREADS, ...`, each as its own
/// 16-term sum, then the shared-memory tree adds the partials.
pub fn matvec_twin(kind: &MulMmKind, row: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let nl = kind.nl();
    let n_sub = n_blocks * nl;
    let mut partial = [0f32; THREADS];
    for (t, p) in partial.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut s = t;
        while s < n_sub {
            let at = (s / nl) * kind.block_bytes;
            let mut reg = [0f32; SUB];
            (kind.dequant_twin)(&row[at..at + kind.block_bytes], s % nl, &mut reg);
            let xs = &x[(s / nl) * kind.block_elems + (s % nl) * SUB..];
            let mut sub_acc = 0f32;
            for i in 0..SUB {
                sub_acc += reg[i] * xs[i];
            }
            acc += sub_acc;
            s += THREADS;
        }
        *p = acc;
    }
    let mut stride = THREADS / 2;
    while stride > 0 {
        for t in 0..stride {
            partial[t] += partial[t + stride];
        }
        stride /= 2;
    }
    partial[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul_mm_ref::fixtures;

    /// The twin is a matvec: it agrees with dequantize-then-dot over the
    /// whole row, for every generated kind, across a row long enough
    /// that threads wrap.
    ///
    /// Sabotage: step `xs` by `(s % nl) * SUB + 1` in the twin and this
    /// names the kind.
    #[test]
    fn the_twin_is_dequantize_then_dot() {
        for mv in super::super::KINDS {
            let Some(k) = super::super::generated_kind(mv) else {
                continue;
            };
            let n_blocks = (THREADS * SUB * 2).div_ceil(k.block_elems) + 3;
            let cols = n_blocks * k.block_elems;
            let row = fixtures::weights(k, 1, cols, 5151);
            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin()).collect();
            let mut want = 0f64;
            for s in 0..n_blocks * k.nl() {
                let at = (s / k.nl()) * k.block_bytes;
                let mut reg = [0f32; SUB];
                (k.dequant_twin)(&row[at..at + k.block_bytes], s % k.nl(), &mut reg);
                for (i, r) in reg.iter().enumerate() {
                    want += f64::from(*r) * f64::from(x[s * SUB + i]);
                }
            }
            let got = matvec_twin(k, &row, &x, n_blocks);
            let tol = 1e-4 * want.abs().max(1.0);
            assert!(
                (f64::from(got) - want).abs() <= tol,
                "{}: twin {got}, dequantize-then-dot {want}",
                k.name
            );
        }
    }

    /// Each generated unit defines its entry once, with the dequant it
    /// was built from, and is cached.
    #[test]
    fn every_generated_unit_defines_its_entry() {
        for mv in super::super::KINDS {
            let Some(k) = super::super::generated_kind(mv) else {
                continue;
            };
            let src = matvec_src(k);
            assert_eq!(src.matches(&format!("void {}(", entry(k))).count(), 1);
            assert!(src.contains(k.dequant_src), "{}", k.name);
            assert!(!src.contains("FX_ROW_STEP") && !src.contains("FX_X_STEP"));
            assert!(std::ptr::eq(src, matvec_src(k)), "{}: not cached", k.name);
        }
    }
}
