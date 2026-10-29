//! `Q2_K` on Metal: the decode matvec and the GEMM functor (#405).
//!
//! The block is ggml's `block_q2_K` (`frink_quant::Q2_K_BLOCK_BYTES`,
//! 84 bytes for 256 weights): `scales[16]`, `qs[64]`, then `half d`,
//! `half dmin`, the scales FIRST and the super-block halves last, the
//! opposite order from Q4_K/Q5_K. Each scale byte holds one 16-weight
//! sub-block's 4-bit scale (low nibble) and 4-bit min (high nibble), and
//! each `qs` byte holds four 2-bit values at shifts 0, 2, 4, 6. Element
//! `128 n + 32 j + 16 h + l` (n, h in 0..2, j in 0..4, l in 0..16) is
//! `d * (scales[s] & 15) * ((qs[32 n + 16 h + l] >> 2 j) & 3) - dmin *
//! (scales[s] >> 4)` with `s = 8 n + 2 j + h`, which is
//! `frink_quant::dequant_q2_k` and ggml's `dequantize_row_q2_K`.
//!
//! The decode is written ONCE here as [`Q2_K_DEQUANT_MSL`]: the GEMM
//! translation unit (`gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC`) splices in the
//! functor, and the matvec walks the same field offsets.

use crate::gpu::{launch_matvec_kind_with, launch_mul_mm_sg, MatvecEpilogue, MetalError};

/// The kind name both launch tables are keyed by.
pub const KIND: &str = "Q2_K";

/// Sixteen consecutive dequantized values of sub-block `il` (0..16), as
/// the shared simdgroup GEMM body asks for them (ggml's
/// `dequantize_q2_K` contract), and the functor that carries the block
/// geometry. Spliced into the GEMM translation unit.
pub const Q2_K_DEQUANT_MSL: &str = r#"
static inline void q2_k_dequant_16(device const uchar* xb, short il, thread float4x4& reg) {
    const float d    = float(*(device const half*)(xb + 80));
    const float dmin = float(*(device const half*)(xb + 82));
    const uchar sc = xb[il];
    device const uchar* q = xb + 16 + 32 * (il / 8) + 16 * (il & 1);
    const uchar shift = uchar(2 * ((il / 2) & 3));
    const float dl = d * float(sc & 0xF);
    const float ml = dmin * float(sc >> 4);
    for (short i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * float((q[i] >> shift) & 3) - ml;
    }
}

struct Q2KDequant {
    static constexpr constant short NL = 16;
    static constexpr constant short BLOCK_BYTES = 84;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        q2_k_dequant_16(xb, il, reg);
    }
};
"#;

/// The GEMM entry lines, spliced after the shared body's other entries.
pub const Q2_K_GEMM_ENTRIES_MSL: &str = r#"
MUL_MM_SG_ENTRY(q2_k_mul_mm_sg, Q2KDequant)
MUL_MM_SG_F16_ENTRY(q2_k_mul_mm_sg_f16, Q2KDequant)
MUL_MM_ID_ENTRY(q2_k_mul_mm_id, Q2KDequant)
MUL_MM_ID_F16_ENTRY(q2_k_mul_mm_id_f16, Q2KDequant)
"#;

/// Rows one threadgroup owns: four per simdgroup, two simdgroups.
pub const ROWS_PER_TG: usize = 8;

/// The decode matvec, ggml's `kernel_mul_mv_q2_K_f32` geometry
/// (`N_R0_Q2_K 4`, `N_SG_Q2_K 2`): 64 threads, four rows per simdgroup.
/// Lane `t` takes the blocks `t / 8, t / 8 + 4, ...`; within a block,
/// lane `t % 8` owns eight `qs` bytes (`8 (t % 4)` into half `t / 8 % 2`)
/// and so the 32 weights those bytes hold at the four shifts. The 32
/// activations and their four per-shift sums are loaded once per block
/// and dotted against each of the four rows; the min term is the sum of
/// the activations times `dmin * min`, so it costs one multiply per
/// sub-block rather than one per weight. A row past `n_rows` reads the
/// last real row (its result is dropped at the store), so a ragged last
/// threadgroup never reads past the matrix.
const Q2_K_MATVEC_BODY_MSL: &str = r#"
// `table` is the id kernels' shared argument (`crate::moe_mv_id`); a
// K-quant stages nothing in it.
static inline void q2_k_matvec_impl(
    device const uchar* weights,
    device const float* x,
    device float* out,
    uint row_bytes,
    uint n_blocks_per_row,
    uint n_rows,
    uint tgpig,
    uint tiisg,
    uint sgitg,
    threadgroup float* table
) {
    constexpr uint NR = 4u;
    constexpr uint NSG = 2u;
    const uint first_row = (tgpig * NSG + sgitg) * NR;
    if (first_row >= n_rows) return;

    const short ix = short(tiisg / 8u); // block phase, 0..4
    const short it = short(tiisg % 8u);
    const short iq = it / 4;            // which 128-weight half
    const short ir = it % 4;            // which eight bytes of its 32
    const short h  = ir / 2;            // sub-block parity of those bytes

    device const uchar* row_ptr[NR];
    for (uint r = 0u; r < NR; ++r) {
        row_ptr[r] = weights + (size_t)min(first_row + r, n_rows - 1u) * row_bytes;
    }

    float yl[32];
    float sumf[NR] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float* y = x + ix * 256 + 128 * iq + 8 * ir;

    for (uint ib = uint(ix); ib < n_blocks_per_row; ib += 4u) {
        float4 sumy = float4(0.0f);
        for (short j = 0; j < 4; ++j) {
            for (short i = 0; i < 8; ++i) {
                yl[8 * j + i] = y[32 * j + i];
                sumy[j] += yl[8 * j + i];
            }
        }
        for (uint r = 0u; r < NR; ++r) {
            device const uchar* blk = row_ptr[r] + (size_t)ib * 84u;
            device const uchar* q = blk + 16 + 32 * iq + 8 * ir;
            device const uchar* sc = blk + 8 * iq + h;
            float4 acc = float4(0.0f);
            for (short i = 0; i < 8; ++i) {
                const uchar b = q[i];
                acc[0] += yl[i +  0] * float(b & 3);
                acc[1] += yl[i +  8] * float((b >> 2) & 3);
                acc[2] += yl[i + 16] * float((b >> 4) & 3);
                acc[3] += yl[i + 24] * float(b >> 6);
            }
            const float d    = float(*(device const half*)(blk + 80));
            const float dmin = float(*(device const half*)(blk + 82));
            sumf[r] += d * (acc[0] * float(sc[0] & 0xF) + acc[1] * float(sc[2] & 0xF)
                          + acc[2] * float(sc[4] & 0xF) + acc[3] * float(sc[6] & 0xF))
                     - dmin * (sumy[0] * float(sc[0] >> 4) + sumy[1] * float(sc[2] >> 4)
                             + sumy[2] * float(sc[4] >> 4) + sumy[3] * float(sc[6] >> 4));
        }
        y += 4 * 256;
    }

    for (uint r = 0u; r < NR; ++r) {
        const float tot = simd_sum(sumf[r]);
        if (tiisg == 0u && first_row + r < n_rows) {
            out[first_row + r] = tot;
        }
    }
}

kernel void q2_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    q2_k_matvec_impl(weights, x, out, row_bytes, n_blocks_per_row, n_rows,
                       tgpig, tiisg, sgitg, nullptr);
}

MOE_MV_ID_ENTRIES(q2_k, q2_k_matvec_impl)
"#;

/// The matvec translation unit: the shared decode text, then the body.
/// Built once and leaked, because the pipeline cache keys on a
/// `&'static str`.
pub fn q2_k_matvec_src() -> &'static str {
    use std::sync::LazyLock;
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        Box::leak(
            format!(
                "#include <metal_stdlib>\nusing namespace metal;\n{Q2_K_DEQUANT_MSL}\n{}\n{Q2_K_MATVEC_BODY_MSL}",
                crate::moe_mv_id::MOE_MV_ID_MSL
            )
            .into_boxed_str(),
        )
    });
    &SRC
}

/// One Q2_K matvec, through the launch table (`matvec_launch_meta`).
pub fn launch_q2_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec_kind_with(KIND, weights, x, rows, row_bytes, MatvecEpilogue::default())
        .ok_or(MetalError::CommandFailed)?
}

/// The Q2_K simdgroup GEMM, through the GEMM table (`mul_mm_sg_meta`).
pub fn launch_q2_k_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_mul_mm_sg(KIND, weights, x_batch, rows, row_bytes, batch_size)?
        .ok_or(MetalError::CommandFailed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k_quant_check::{check_gemm, check_matvec, CheckKind};

    /// Pseudo-random scale and quant bytes with `d` and `dmin` forced to
    /// realistic magnitudes, so the GEMM's half tiles stay in range.
    fn blocks(seed: u32, n_blocks: usize) -> Vec<u8> {
        let mut bytes = crate::k_quant_check::pseudo_bytes(seed, n_blocks * 84);
        for (b, blk) in bytes.as_chunks_mut::<84>().0.iter_mut().enumerate() {
            let d = half::f16::from_f32(0.010 + 0.002 * (b % 3) as f32);
            let dmin = half::f16::from_f32(0.004 + 0.001 * (b % 5) as f32);
            blk[80..82].copy_from_slice(&d.to_le_bytes());
            blk[82..84].copy_from_slice(&dmin.to_le_bytes());
        }
        bytes
    }

    const Q2K: CheckKind = CheckKind {
        name: KIND,
        block_bytes: frink_quant::Q2_K_BLOCK_BYTES,
        dequant: frink_quant::dequant_q2_k,
        blocks,
        matvec: launch_q2_k_matvec,
        gemm: launch_q2_k_mul_mm_sg,
    };

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn q2_k_matvec_matches_the_cpu_dequant() {
        check_matvec(&Q2K);
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn q2_k_mul_mm_sg_matches_the_cpu_dequant() {
        check_gemm(&Q2K);
    }
}
