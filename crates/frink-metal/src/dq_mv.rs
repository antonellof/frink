//! Q4_1, Q5_1 and TQ1_0 on Metal: the decode matvec built on each
//! kind's GEMM dequant functor, so the functor is the kind's only
//! dequant (#405).
//!
//! - `Q4_1Dequant` and `Q5_1Dequant` (below) are `Q4_0Dequant` and
//!   `Q5_0Dequant` (`gpu.rs`) with ggml's min: block `half d, half m`,
//!   then Q5_1's `uint32 qh`, then sixteen nibble bytes, element
//!   `d * q + m` (ggml `dequantize_q4_1` / `dequantize_q5_1`, and
//!   `frink_quant::dequant_q4_1` / `dequant_q5_1`). Like their siblings
//!   they compose ggml's `uint16` reads from bytes, because a row is only
//!   2-byte aligned here.
//! - `TQ1_0Dequant` is `crate::ternary`'s, beside PTQ1_0's, sharing its
//!   trit decode.
//!
//! The functor is what the simdgroup GEMM (f32 and f16 activations),
//! `mul_mm_id` and the embedding gather instantiate (through
//! `crate::gen_kinds`), and what [`dq_matvec_impl`] reads here; the MoE
//! `mul_mv_id` kernels are generated from that `_impl`
//! (`crate::moe_mv_id`). One dequant per kind, everywhere.
//!
//! The matvec: two simdgroups of four rows (eight rows and 64 threads a
//! threadgroup, the geometry of the other generated kinds). Lane `t`
//! takes the 16-value chunks `t, t + 32, ...` of the row, loads its
//! sixteen activations once, and dots them against each row's chunk as
//! the functor dequantizes it. ggml's `mul_vec_q_n` for Q4_1 / Q5_1
//! instead folds the min into a per-block activation sum; the products
//! are the same and the rounding order differs, which ADR 0009 allows.
//! ggml's Metal backend has no TQ1_0 kernel at the pin (it refuses the
//! type for `mul_mat` and `get_rows`), so TQ1_0's reference is ggml's
//! CPU `dequantize_row_tq1_0`, which `frink_quant::ternary` transcribes.
//!
//! A row past `n_rows` is clamped to the last row before it is read;
//! its result is dropped.

use crate::gen_kinds::gen_launchers;

/// The Q4_1 and Q5_1 functors, spliced into the GEMM translation unit
/// and prepended to the matvec unit.
pub(crate) const DQ_FUNCTORS_MSL: &str = r#"
// llama `dequantize_q4_1`. GGUF: half d, half m, 16 bytes of 32 nibbles.
struct Q4_1Dequant {
    static constexpr constant short NL = 2;
    static constexpr constant short BLOCK_BYTES = 20;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        const float d = float(*(device const half*)(xb));
        const float m = float(*(device const half*)(xb + 2));
        device const uchar* qs = xb + 4;
        const float d1 = il ? d / 16.0f : d;
        const float d2 = d1 / 256.0f;
        const ushort mask0 = il ? 0x00F0 : 0x000F;
        const ushort mask1 = mask0 << 8;
        for (short i = 0; i < 8; ++i) {
            const ushort w = ushort(qs[2 * i]) | (ushort(qs[2 * i + 1]) << 8);
            reg[i / 2][2 * (i % 2) + 0] = (float(w & mask0) * d1) + m;
            reg[i / 2][2 * (i % 2) + 1] = (float(w & mask1) * d2) + m;
        }
    }
};

// llama `dequantize_q5_1`. GGUF: half d, half m, uint32 qh, 16 qs bytes.
struct Q5_1Dequant {
    static constexpr constant short NL = 2;
    static constexpr constant short BLOCK_BYTES = 24;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        const float d = float(*(device const half*)(xb));
        const float m = float(*(device const half*)(xb + 2));
        const uint qh = uint(xb[4]) | (uint(xb[5]) << 8) | (uint(xb[6]) << 16)
            | (uint(xb[7]) << 24);
        device const uchar* qs = xb + 8;
        const ushort mask = il ? 0x00F0 : 0x000F;
        const int x_mv = il ? 4 : 0;
        const int gh_mv = il ? 12 : 0;
        const int gh_bk = il ? 0 : 4;
        for (short i = 0; i < 8; ++i) {
            const ushort w = ushort(qs[2 * i]) | (ushort(qs[2 * i + 1]) << 8);
            const uchar xh_0 = uchar(((qh >> (gh_mv + 2 * i)) << gh_bk) & 0x10);
            const uchar xh_1 = uchar(((qh >> (gh_mv + 2 * i + 1)) << gh_bk) & 0x10);
            const int x0 = int((((w) & mask) >> x_mv) | xh_0);
            const int x1 = int((((w >> 8) & mask) >> x_mv) | xh_1);
            reg[i / 2][2 * (i % 2) + 0] = d * float(x0) + m;
            reg[i / 2][2 * (i % 2) + 1] = d * float(x1) + m;
        }
    }
};
"#;

/// The matvec body over any 16-value functor. `table` is the id
/// kernels' shared argument (`crate::moe_mv_id`); nothing is staged.
const DQ_MATVEC_BODY_MSL: &str = r#"
template<typename DQ>
static inline void dq_matvec_impl(
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

    device const uchar* row_ptr[NR];
    for (uint r = 0u; r < NR; ++r) {
        row_ptr[r] = weights + (size_t)min(first_row + r, n_rows - 1u) * row_bytes;
    }

    const uint nl = uint(DQ::NL);
    const uint n_chunks = n_blocks_per_row * nl;
    float sumf[NR] = {0.f, 0.f, 0.f, 0.f};

    for (uint c = tiisg; c < n_chunks; c += 32u) {
        device const float4* y4 = (device const float4*)(x + 16u * c);
        const float4 y0 = y4[0], y1 = y4[1], y2 = y4[2], y3 = y4[3];
        const size_t at = (size_t)(c / nl) * uint(DQ::BLOCK_BYTES);
        const short il = short(c % nl);
        for (uint r = 0u; r < NR; ++r) {
            float4x4 w;
            DQ::get(row_ptr[r] + at, il, w);
            sumf[r] += (dot(y0, w[0]) + dot(y1, w[1])) + (dot(y2, w[2]) + dot(y3, w[3]));
        }
    }

    for (uint r = 0u; r < NR; ++r) {
        const float tot = simd_sum(sumf[r]);
        if (tiisg == 0u && first_row + r < n_rows) {
            out[first_row + r] = tot;
        }
    }
}

#define DQ_MATVEC_ENTRY(NAME, DQ)                                           \
kernel void NAME(                                                           \
    device const uchar* weights [[buffer(0)]],                              \
    device const float* x [[buffer(1)]],                                    \
    device float* out [[buffer(2)]],                                        \
    constant uint& row_bytes [[buffer(3)]],                                 \
    constant uint& n_blocks_per_row [[buffer(4)]],                          \
    constant uint& n_rows [[buffer(5)]],                                    \
    uint tgpig [[threadgroup_position_in_grid]],                            \
    uint tiisg [[thread_index_in_simdgroup]],                               \
    uint sgitg [[simdgroup_index_in_threadgroup]]                           \
) {                                                                         \
    dq_matvec_impl<DQ>(weights, x, out, row_bytes, n_blocks_per_row,        \
                       n_rows, tgpig, tiisg, sgitg, nullptr);               \
}
"#;

/// The matvec translation unit: the functors (TQ1_0's from
/// `crate::ternary`), the body, and a matvec and both MoE id entries
/// per functor-family row of `gen_kinds::KINDS`.
pub(crate) fn dq_matvec_src() -> &'static str {
    use crate::gen_kinds::{Family, KINDS};
    use std::sync::LazyLock;
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        let mut s = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n{}\n{DQ_FUNCTORS_MSL}\n{}\n{DQ_MATVEC_BODY_MSL}",
            crate::ternary::PTQ1_0_DEQUANT_MSL,
            crate::moe_mv_id::MOE_MV_ID_MSL
        );
        for k in KINDS.iter().filter(|k| k.family == Family::Functor) {
            s.push_str(&format!(
                "DQ_MATVEC_ENTRY({}, {dq})\nMOE_MV_ID_ENTRIES({}, dq_matvec_impl<{dq}>)\n",
                k.matvec,
                k.prefix,
                dq = k.dequant
            ));
        }
        Box::leak(s.into_boxed_str())
    });
    &SRC
}

gen_launchers! {
    "Q4_1", launch_q4_1_matvec, launch_q4_1_mul_mm_sg;
    "Q5_1", launch_q5_1_matvec, launch_q5_1_mul_mm_sg;
    "TQ1_0", launch_tq1_0_matvec, launch_tq1_0_mul_mm_sg;
}

/// Test blocks of `name`: random quant bytes with each `half` field
/// realistic (`d`, and the min a negative offset of a few steps), so the
/// GEMM's `half` tiles stay in range.
#[cfg(test)]
pub(crate) fn test_blocks(name: &str, seed: u32, n_blocks: usize) -> Vec<u8> {
    let k = crate::gen_kinds::by_name(name).expect("a generated kind");
    let bb = k.block_bytes;
    let mut w = crate::k_quant_check::pseudo_bytes(seed, n_blocks * bb);
    let h = |v: f32| half::f16::from_f32(v).to_le_bytes();
    for b in 0..n_blocks {
        let at = b * bb;
        let d = 0.004 + 0.001 * (b % 7) as f32;
        match name {
            "Q4_1" | "Q5_1" => {
                w[at..at + 2].copy_from_slice(&h(d));
                w[at + 2..at + 4].copy_from_slice(&h(-d * (4.0 + (b % 5) as f32)));
            }
            "TQ1_0" => w[at + 52..at + 54].copy_from_slice(&h(d)),
            other => panic!("{other}"),
        }
    }
    w
}

#[cfg(test)]
mod tests;
