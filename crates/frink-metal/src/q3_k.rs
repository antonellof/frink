//! `Q3_K` on Metal: the decode matvec and the GEMM functor (#405).
//!
//! The block is ggml's `block_q3_K` (`frink_quant::Q3_K_BLOCK_BYTES`,
//! 110 bytes for 256 weights): `hmask[32]`, `qs[64]`, `scales[12]`,
//! `half d`. A weight is three bits, two from `qs` (four a byte, shifts
//! 0, 2, 4, 6) and a high bit from `hmask`; the high bit CLEAR means
//! subtract 4. Each 16-weight sub-block has a 6-bit scale, biased by 32,
//! packed across the 12 bytes: scale `s` takes its low nibble from
//! `scales[s % 8]` (the high nibble for `s >= 8`) and its top two bits
//! from `scales[8 + s % 4]` at shift `2 (s / 4)`. That is the per-byte
//! form of `dequantize_row_q3_K`'s `aux[]` unpack, the one
//! `frink_quant::q3_k_unpack_scales` transcribes. Element `128 n + 32 j
//! + 16 h + l` (n, h in 0..2, j in 0..4, l in 0..16) is `d * (scale(8 n
//! + 2 j + h) - 32) * (((qs[32 n + 16 h + l] >> 2 j) & 3) - (hmask[16 h
//! + l] & (1 << (4 n + j)) ? 0 : 4))`.
//!
//! The decode is written ONCE here as [`Q3_K_DEQUANT_MSL`]: the GEMM
//! translation unit (`gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC`) splices in the
//! functor, and the matvec's translation unit starts with the same text,
//! so both read the scales through one `q3_k_scale`.

use crate::gpu::{launch_matvec_kind_with, launch_mul_mm_sg, MatvecEpilogue, MetalError};

/// The kind name both launch tables are keyed by.
pub const KIND: &str = "Q3_K";

/// The scale unpack, the 16-value decode the shared simdgroup GEMM body
/// asks for (ggml's `dequantize_q3_K` contract), and the functor. Spliced
/// into the GEMM translation unit and prepended to the matvec's.
pub const Q3_K_DEQUANT_MSL: &str = r#"
// Sub-block `s`'s scale, unbiased (-32..31). `scales` is the block's
// 12 packed bytes (offset 96).
static inline float q3_k_scale(device const uchar* scales, short s) {
    const uint lo = (uint(scales[s & 7]) >> (4 * (s >> 3))) & 0xFu;
    const uint hi = (uint(scales[8 + (s & 3)]) >> (2 * (s >> 2))) & 3u;
    return float(int(lo | (hi << 4)) - 32);
}

static inline void q3_k_dequant_16(device const uchar* xb, short il, thread float4x4& reg) {
    const float d = float(*(device const half*)(xb + 108));
    const float dl = d * q3_k_scale(xb + 96, il);
    device const uchar* q = xb + 32 + 32 * (il / 8) + 16 * (il & 1);
    device const uchar* hm = xb + 16 * (il & 1);
    const uchar m = uchar(1u << (il / 2));
    const uchar shift = uchar(2 * ((il / 2) & 3));
    for (short i = 0; i < 16; ++i) {
        const int v = int((q[i] >> shift) & 3) - ((hm[i] & m) ? 0 : 4);
        reg[i / 4][i % 4] = dl * float(v);
    }
}

struct Q3KDequant {
    static constexpr constant short NL = 16;
    static constexpr constant short BLOCK_BYTES = 110;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        q3_k_dequant_16(xb, il, reg);
    }
};
"#;

/// The GEMM entry lines, spliced after the shared body's other entries.
pub const Q3_K_GEMM_ENTRIES_MSL: &str = r#"
MUL_MM_SG_ENTRY(q3_k_mul_mm_sg, Q3KDequant)
MUL_MM_SG_F16_ENTRY(q3_k_mul_mm_sg_f16, Q3KDequant)
MUL_MM_ID_ENTRY(q3_k_mul_mm_id, Q3KDequant)
MUL_MM_ID_F16_ENTRY(q3_k_mul_mm_id_f16, Q3KDequant)
"#;

/// Rows one threadgroup owns: two per simdgroup, two simdgroups.
pub const ROWS_PER_TG: usize = 4;

/// The decode matvec, at ggml's `kernel_mul_mv_q3_K_f32` geometry
/// (`N_R0_Q3_K 2`, `N_SG_Q3_K 2`): 64 threads, two rows per simdgroup.
/// The lane split is the Q2_K matvec's (`crate::q2_k`): lane `t` takes
/// the blocks `t / 8, t / 8 + 4, ...`, and within a block owns eight
/// `qs` bytes and the same eight `hmask` bytes, so 32 weights across the
/// four shifts. Per shift the lane forms `sum y * low2` and the sum of
/// the activations whose high bit is clear, and the sub-block's weight
/// is `d * scale * (first - 4 * second)`. Rows past `n_rows` read the
/// last real row and are dropped at the store.
const Q3_K_MATVEC_BODY_MSL: &str = r#"
// `table` is the id kernels' shared argument (`crate::moe_mv_id`); a
// K-quant stages nothing in it.
static inline void q3_k_matvec_impl(
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
    constexpr uint NR = 2u;
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
    float sumf[NR] = {0.0f, 0.0f};
    device const float* y = x + ix * 256 + 128 * iq + 8 * ir;

    for (uint ib = uint(ix); ib < n_blocks_per_row; ib += 4u) {
        for (short j = 0; j < 4; ++j) {
            for (short i = 0; i < 8; ++i) {
                yl[8 * j + i] = y[32 * j + i];
            }
        }
        for (uint r = 0u; r < NR; ++r) {
            device const uchar* blk = row_ptr[r] + (size_t)ib * 110u;
            uchar qb[8];
            uchar hb[8];
            for (short i = 0; i < 8; ++i) {
                qb[i] = blk[32 + 32 * iq + 8 * ir + i];
                hb[i] = blk[8 * ir + i];
            }
            float acc = 0.0f;
            for (short j = 0; j < 4; ++j) {
                const uchar m = uchar(1u << (4 * iq + j));
                float low = 0.0f;
                float clear = 0.0f;
                for (short i = 0; i < 8; ++i) {
                    const float yv = yl[8 * j + i];
                    low += yv * float((qb[i] >> (2 * j)) & 3);
                    clear += (hb[i] & m) ? 0.0f : yv;
                }
                acc += q3_k_scale(blk + 96, 8 * iq + 2 * j + h) * (low - 4.0f * clear);
            }
            sumf[r] += float(*(device const half*)(blk + 108)) * acc;
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

kernel void q3_k_matvec(
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
    q3_k_matvec_impl(weights, x, out, row_bytes, n_blocks_per_row, n_rows,
                       tgpig, tiisg, sgitg, nullptr);
}

MOE_MV_ID_ENTRIES(q3_k, q3_k_matvec_impl)
"#;

/// The matvec translation unit: the shared decode text, then the body.
pub fn q3_k_matvec_src() -> &'static str {
    use std::sync::LazyLock;
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        Box::leak(
            format!(
                "#include <metal_stdlib>\nusing namespace metal;\n{Q3_K_DEQUANT_MSL}\n{}\n{Q3_K_MATVEC_BODY_MSL}",
                crate::moe_mv_id::MOE_MV_ID_MSL
            )
            .into_boxed_str(),
        )
    });
    &SRC
}

/// One Q3_K matvec, through the launch table (`matvec_launch_meta`).
pub fn launch_q3_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec_kind_with(KIND, weights, x, rows, row_bytes, MatvecEpilogue::default())
        .ok_or(MetalError::CommandFailed)?
}

/// The Q3_K simdgroup GEMM, through the GEMM table (`mul_mm_sg_meta`).
pub fn launch_q3_k_mul_mm_sg(
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

    /// Pseudo-random `hmask`, `qs` and scale bytes (so every high-bit
    /// and scale-packing case is reached) with `d` forced realistic.
    fn blocks(seed: u32, n_blocks: usize) -> Vec<u8> {
        let mut bytes = crate::k_quant_check::pseudo_bytes(seed, n_blocks * 110);
        for (b, blk) in bytes.as_chunks_mut::<110>().0.iter_mut().enumerate() {
            let d = half::f16::from_f32(0.003 + 0.0005 * (b % 4) as f32);
            blk[108..110].copy_from_slice(&d.to_le_bytes());
        }
        bytes
    }

    const Q3K: CheckKind = CheckKind {
        name: KIND,
        block_bytes: frink_quant::Q3_K_BLOCK_BYTES,
        block_elems: 256,
        dequant: frink_quant::dequant_q3_k,
        blocks,
        matvec: launch_q3_k_matvec,
        gemm: launch_q3_k_mul_mm_sg,
    };

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn q3_k_matvec_matches_the_cpu_dequant() {
        check_matvec(&Q3K);
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn q3_k_mul_mm_sg_matches_the_cpu_dequant() {
        check_gemm(&Q3K);
    }
}
