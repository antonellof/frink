//! The base-3 trit formats: ggml's TQ1_0 and PrismML's PTQ1_0
//! (`frink_quant::ternary`). Each is a [`MulMmKind`] row plus the Rust
//! twin of its `dequant_src`, statement for statement, as in
//! [`super::legacy`].
//!
//! ggml's CUDA backend has no TQ1_0 kernel at the pin and upstream has
//! no PTQ1_0 at all, so the reference for both is the CPU dequantizer
//! (ggml's `dequantize_row_tq1_0`, the PrismML fork's
//! `dequantize_row_ptq1_0`), which `frink_quant::ternary::unpack_trits`
//! transcribes. The element order of each 16-value sub-block is the one
//! `frink-metal`'s `TQ1_0Dequant` and `ptq1_0_dequant_16` derive from
//! that walk:
//!
//! - TQ1_0 (`qs[48]; qh[4]; half d`, `nl` 16): sub-block `il < 10` is
//!   trit `il / 2` of `qs[16 (il % 2)..][..16]`; `il < 15` is trit
//!   `il - 10` of `qs[32..48]`; `il = 15` is the `qh` tail, element `e`
//!   being trit `e / 4` of `qh[e % 4]`.
//! - PTQ1_0 (`qs[24]; qh[2]; half d`, `nl` 8): `il < 5` is trit `il` of
//!   `qs[0..16]`; `il = 5, 6` are trits `2 (il - 5)` and `2 (il - 5) + 1`
//!   of `qs[16..24]`; `il = 7` is trit 4 of `qs[16..24]`, then element
//!   `8 + 2 n + h` is trit `n` of `qh[h]`.
//!
//! A trit is `((q * 3^n) mod 256) * 3 >> 8`, minus one: the CPU's u8
//! trick (`frink_trit`, emitted for every kind by
//! [`crate::mul_mm::kernel_src_with_body`] beside the f16 decode). The
//! value `d * t` with `t` in `{-1, 0, 1}` is exact, so the device and the
//! host agree bit for bit.
//!
//! The Hadamard rotation Bonsai folds into PTQ1_0 weights is undone on
//! the activation before any launch (`frink_core::weight_matrix::
//! hadamard`); these kernels see a plain ternary matrix.

use crate::mul_mm::{f16_to_f32, MulMmKind, SUB};

/// `frink_trit`, the helper both dequants call. Emitted once per unit.
pub const TRIT_SRC: &str = r#"
__device__ __forceinline__ float frink_trit(unsigned char q, unsigned int p) {
    const unsigned int qp = ((unsigned int)q * p) & 0xFFu;
    return (float)((qp * 3u) >> 8) - 1.0f;
}
"#;

/// `3^n` for trit `n`, most significant first.
const POW3: [u32; 5] = [1, 3, 9, 27, 81];

/// Scalar twin of `frink_trit`.
fn trit(q: u8, p: u32) -> f32 {
    let qp = (u32::from(q) * p) & 0xFF;
    ((qp * 3) >> 8) as f32 - 1.0
}

/// TQ1_0: 54-byte blocks of 256 elements, the f16 scale at byte 52.
pub const TQ1_0: MulMmKind = MulMmKind {
    name: "TQ1_0",
    module_name: "frink_mul_mm_tq1_0",
    fn_name: "tq1_0_mul_mm",
    block_bytes: 54,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const unsigned int pow3[5] = {1u, 3u, 9u, 27u, 81u};
    const float d = frink_f16_to_f32(
        (unsigned short)xb[52] | ((unsigned short)xb[53] << 8));
    if (il < 10) {
        const unsigned int p = pow3[il / 2];
        const unsigned char* q = xb + 16 * (il % 2);
        for (int m = 0; m < 16; m++) {
            reg[m] = d * frink_trit(q[m], p);
        }
    } else if (il < 15) {
        const unsigned int p = pow3[il - 10];
        for (int m = 0; m < 16; m++) {
            reg[m] = d * frink_trit(xb[32 + m], p);
        }
    } else {
        for (int e = 0; e < 16; e++) {
            reg[e] = d * frink_trit(xb[48 + e % 4], pow3[e / 4]);
        }
    }
}
"#,
    dequant_twin: dequant_sub_tq1_0,
    codebook: None,
};

/// Scalar twin of [`TQ1_0`]'s `dequant_src`.
fn dequant_sub_tq1_0(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_to_f32(u16::from(xb[52]) | (u16::from(xb[53]) << 8));
    if il < 10 {
        let p = POW3[il / 2];
        let q = &xb[16 * (il % 2)..];
        for (m, r) in reg.iter_mut().enumerate() {
            *r = d * trit(q[m], p);
        }
    } else if il < 15 {
        let p = POW3[il - 10];
        for (m, r) in reg.iter_mut().enumerate() {
            *r = d * trit(xb[32 + m], p);
        }
    } else {
        for (e, r) in reg.iter_mut().enumerate() {
            *r = d * trit(xb[48 + e % 4], POW3[e / 4]);
        }
    }
}

/// PTQ1_0: 28-byte blocks of 128 elements, the f16 scale at byte 26.
pub const PTQ1_0: MulMmKind = MulMmKind {
    name: "PTQ1_0",
    module_name: "frink_mul_mm_ptq1_0",
    fn_name: "ptq1_0_mul_mm",
    block_bytes: 28,
    block_elems: 128,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const unsigned int pow3[5] = {1u, 3u, 9u, 27u, 81u};
    const float d = frink_f16_to_f32(
        (unsigned short)xb[26] | ((unsigned short)xb[27] << 8));
    if (il < 5) {
        const unsigned int p = pow3[il];
        for (int m = 0; m < 16; m++) {
            reg[m] = d * frink_trit(xb[m], p);
        }
    } else if (il < 7) {
        const int n0 = (il - 5) * 2;
        for (int j = 0; j < 2; j++) {
            for (int m = 0; m < 8; m++) {
                reg[j * 8 + m] = d * frink_trit(xb[16 + m], pow3[n0 + j]);
            }
        }
    } else {
        for (int m = 0; m < 8; m++) {
            reg[m] = d * frink_trit(xb[16 + m], pow3[4]);
        }
        for (int n = 0; n < 4; n++) {
            for (int h = 0; h < 2; h++) {
                reg[8 + n * 2 + h] = d * frink_trit(xb[24 + h], pow3[n]);
            }
        }
    }
}
"#,
    dequant_twin: dequant_sub_ptq1_0,
    codebook: None,
};

/// Scalar twin of [`PTQ1_0`]'s `dequant_src`.
fn dequant_sub_ptq1_0(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_to_f32(u16::from(xb[26]) | (u16::from(xb[27]) << 8));
    if il < 5 {
        let p = POW3[il];
        for (m, r) in reg.iter_mut().enumerate() {
            *r = d * trit(xb[m], p);
        }
    } else if il < 7 {
        let n0 = (il - 5) * 2;
        for j in 0..2 {
            for m in 0..8 {
                reg[j * 8 + m] = d * trit(xb[16 + m], POW3[n0 + j]);
            }
        }
    } else {
        for m in 0..8 {
            reg[m] = d * trit(xb[16 + m], POW3[4]);
        }
        for n in 0..4 {
            for h in 0..2 {
                reg[8 + n * 2 + h] = d * trit(xb[24 + h], POW3[n]);
            }
        }
    }
}
