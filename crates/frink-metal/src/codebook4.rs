//! IQ4_NL and MXFP4 on Metal: the two 32-element formats whose 4-bit
//! code is an index into a sixteen-entry table rather than a magnitude.
//!
//! They share everything but the scale and the table, so they share one
//! body. A block is a scale followed by sixteen bytes `qs`; the low
//! nibble of `qs[j]` is element `j` and the high nibble element `j + 16`
//! (ggml `dequantize_iq4_nl` / `dequantize_mxfp4`, and
//! `frink_quant::dequant_iq4_nl` / `dequant_mxfp4_gguf` on the host).
//!
//! | kind | block | scale | table |
//! |---|---|---|---|
//! | IQ4_NL | 18 bytes | `half d` at byte 0 | `kvalues_iq4nl` |
//! | MXFP4 | 17 bytes | E8M0 byte at byte 0, `2^(e-127)` | E2M1, `kvalues_mxfp4_f` |
//!
//! The MXFP4 table and scale are the ones ggml's Metal shaders use at
//! the pin (`kernels/common.h`: real E2M1 values times the unhalved
//! `e8m0_to_fp32`), which multiply out to the same products as the
//! doubled-integer table times `ggml_e8m0_to_fp32_half` on the CPU.
//!
//! An MXFP4 block is 17 bytes, so a block (and a row) can start at any
//! byte: every read of it here is a byte read. IQ4_NL rows are 2-byte
//! aligned, the `half` scale's own alignment, like Q4_0's.
//!
//! The prefill GEMM stages the dequantized weights as `half`, as
//! ggml's `kernel_mul_mm_mxfp4_f32` does. Its `mul_mm_id` (the MoE
//! indexed GEMM) stages MXFP4 as `bfloat` where the device has it; this
//! module has no `mul_mm_id` or `mul_mv_id` entry, see
//! `crate::gpu::mul_mm_id_meta`.

use crate::gpu::{MatvecEpilogue, MetalError};

/// The two formats, the codebook-dequant functor the GEMM instantiates
/// for each, and the per-format pieces the matvec reads. Spliced into
/// the GEMM translation unit (`gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC`) and
/// prepended to the matvec's.
pub const CODEBOOK4_DEQUANT_MSL: &str = r#"
// ggml `kvalues_iq4nl` and `kvalues_mxfp4_f` (`kernels/common.h`).
constant float CB4_IQ4NL_VALUES[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
       1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
};
constant float CB4_MXFP4_VALUES[16] = {
    0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
    -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

struct CB4IQ4NL {
    static constexpr constant short BLOCK_BYTES = 18;
    static constexpr constant short QS = 2;
    static inline float scale(device const uchar* b) {
        return float(*(device const half*)b);
    }
    static inline float value(uint code) { return CB4_IQ4NL_VALUES[code]; }
};

// ggml `e8m0_to_fp32`: the byte is the f32 exponent field, and 0 is
// the one subnormal, 2^-127.
struct CB4MXFP4 {
    static constexpr constant short BLOCK_BYTES = 17;
    static constexpr constant short QS = 1;
    static inline float scale(device const uchar* b) {
        const uint e = uint(b[0]);
        return as_type<float>(e == 0u ? 0x00400000u : (e << 23));
    }
    static inline float value(uint code) { return CB4_MXFP4_VALUES[code]; }
};

// Sixteen consecutive values of a 32-element block: `il = 0` is the
// low nibbles (elements 0..16), `il = 1` the high (16..32). NL = 2, as
// for the other 32-element kinds.
template<typename F>
struct Codebook4Dequant {
    static constexpr constant short NL = 2;
    static constexpr constant short BLOCK_BYTES = F::BLOCK_BYTES;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        const float d = F::scale(xb);
        device const uchar* qs = xb + F::QS;
        const uchar shift = il ? 4 : 0;
        for (short i = 0; i < 16; ++i) {
            reg[i / 4][i % 4] = d * F::value((qs[i] >> shift) & 0xF);
        }
    }
};

typedef Codebook4Dequant<CB4IQ4NL> IQ4NLDequant;
typedef Codebook4Dequant<CB4MXFP4> MXFP4Dequant;
"#;

/// The GEMM entry lines, spliced after the shared body's other entries.
pub const CODEBOOK4_GEMM_ENTRIES_MSL: &str = r#"
MUL_MM_SG_ENTRY(iq4_nl_mul_mm_sg, IQ4NLDequant)
MUL_MM_SG_F16_ENTRY(iq4_nl_mul_mm_sg_f16, IQ4NLDequant)
MUL_MM_SG_ENTRY(mxfp4_mul_mm_sg, MXFP4Dequant)
MUL_MM_SG_F16_ENTRY(mxfp4_mul_mm_sg_f16, MXFP4Dequant)
"#;

/// The matvec translation unit: both entry points, one body.
pub fn codebook4_matvec_src() -> &'static str {
    use std::sync::LazyLock;
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        Box::leak(
            format!(
                "#include <metal_stdlib>\nusing namespace metal;\n{CODEBOOK4_DEQUANT_MSL}\n{CODEBOOK4_MATVEC_BODY_MSL}"
            )
            .into_boxed_str(),
        )
    });
    &SRC
}

/// The geometry of ggml's `kernel_mul_mv_iq4_nl_f32` and
/// `kernel_mul_mv_mxfp4_f32` (`N_R0 = 2`, `N_SG = 2`): two simdgroups of
/// two rows, four rows and 64 threads a threadgroup. A lane pair covers
/// one block, each lane eight bytes of `qs` (elements `8 it ..` from the
/// low nibbles and `16 + 8 it ..` from the high), and the sixteen pairs
/// of a simdgroup walk blocks `ix, ix + 16, ...`. The table sits in 32
/// floats of threadgroup memory.
///
/// Unlike ggml, a row past `n_rows` is clamped to the last row before
/// it is read: ggml's buffers are padded, these are not.
const CODEBOOK4_MATVEC_BODY_MSL: &str = r#"
template<typename F>
static inline void codebook4_matvec_impl(
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
    constexpr uint NSG = 2u;
    constexpr uint NR0 = 2u;
    const uint first_row = (tgpig * NSG + sgitg) * NR0;

    const uint ix = tiisg / 2u;
    const uint it = tiisg % 2u;

    // Both simdgroups write the same 32 values; no lane returns before
    // the barrier.
    table[tiisg] = F::value(tiisg % 16u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf[NR0] = { 0.f, 0.f };
    device const float* yb = x + ix * 32u + it * 8u;

    for (uint ib = ix; ib < n_blocks_per_row; ib += 16u) {
        device const float4* y4 = (device const float4*)yb;
        const float4 y_lo0 = y4[0];
        const float4 y_lo1 = y4[1];
        const float4 y_hi0 = y4[4];
        const float4 y_hi1 = y4[5];

        for (uint row = 0u; row < NR0; ++row) {
            const uint r = min(first_row + row, n_rows - 1u);
            device const uchar* blk =
                weights + (size_t)r * row_bytes + (size_t)ib * uint(F::BLOCK_BYTES);
            device const uchar* q = blk + F::QS + 8u * it;

            const float4 lo0 = float4(table[q[0] & 0xF], table[q[1] & 0xF],
                                      table[q[2] & 0xF], table[q[3] & 0xF]);
            const float4 hi0 = float4(table[q[0] >> 4], table[q[1] >> 4],
                                      table[q[2] >> 4], table[q[3] >> 4]);
            const float4 lo1 = float4(table[q[4] & 0xF], table[q[5] & 0xF],
                                      table[q[6] & 0xF], table[q[7] & 0xF]);
            const float4 hi1 = float4(table[q[4] >> 4], table[q[5] >> 4],
                                      table[q[6] >> 4], table[q[7] >> 4]);

            float4 acc = (y_lo0 * lo0 + y_lo1 * lo1) + (y_hi0 * hi0 + y_hi1 * hi1);
            sumf[row] += F::scale(blk) * ((acc[0] + acc[1]) + (acc[2] + acc[3]));
        }
        yb += 16u * 32u;
    }

    for (uint row = 0u; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0u && first_row + row < n_rows) {
            out[first_row + row] = tot;
        }
    }
}

#define CODEBOOK4_MATVEC_ENTRY(NAME, F)                                     \
kernel void NAME(                                                           \
    device const uchar* weights [[buffer(0)]],                              \
    device const float* x [[buffer(1)]],                                    \
    device float* out [[buffer(2)]],                                        \
    constant uint& row_bytes [[buffer(3)]],                                 \
    constant uint& n_blocks_per_row [[buffer(4)]],                          \
    constant uint& n_rows [[buffer(5)]],                                    \
    uint tgpig [[threadgroup_position_in_grid]],                            \
    uint tiisg [[thread_index_in_simdgroup]],                               \
    uint sgitg [[simdgroup_index_in_threadgroup]],                          \
    threadgroup float* table [[threadgroup(0)]]                             \
) {                                                                         \
    codebook4_matvec_impl<F>(weights, x, out, row_bytes, n_blocks_per_row,  \
                             n_rows, tgpig, tiisg, sgitg, table);           \
}

CODEBOOK4_MATVEC_ENTRY(iq4_nl_matvec, CB4IQ4NL)
CODEBOOK4_MATVEC_ENTRY(mxfp4_matvec, CB4MXFP4)
"#;

/// Threadgroup threads and table bytes for the two matvecs, read by
/// `gpu::encode_matvec_with_offsets`.
pub(crate) const CODEBOOK4_MATVEC_TG: (usize, usize) = (64, 32 * 4);

fn matvec(
    kind: &str,
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    crate::gpu::launch_matvec_kind_with(
        kind,
        weights,
        x,
        rows,
        row_bytes,
        MatvecEpilogue::default(),
    )
    .ok_or(MetalError::CommandFailed)?
}

fn gemm(
    kind: &str,
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    crate::gpu::launch_mul_mm_sg(kind, weights, x_batch, rows, row_bytes, batch_size)?
        .ok_or(MetalError::CommandFailed)
}

/// The IQ4_NL decode matvec.
pub fn launch_iq4_nl_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    matvec("IQ4_NL", weights, x, rows, row_bytes)
}

/// The MXFP4 decode matvec (gpt-oss's expert weights).
pub fn launch_mxfp4_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    matvec("MXFP4", weights, x, rows, row_bytes)
}

/// The IQ4_NL simdgroup GEMM; `x_batch` is `[batch, cols]`, the result
/// `[batch, rows]`.
pub fn launch_iq4_nl_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    gemm("IQ4_NL", weights, x_batch, rows, row_bytes, batch_size)
}

/// The MXFP4 simdgroup GEMM.
pub fn launch_mxfp4_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    gemm("MXFP4", weights, x_batch, rows, row_bytes, batch_size)
}

#[cfg(test)]
mod tests;
