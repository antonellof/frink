//! The grid-codebook formats on Metal: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
//! IQ3_S, IQ1_S and IQ1_M (#405).
//!
//! All seven are 256-element super-blocks of eight 32-element
//! sub-blocks, and every one decodes a group of eight consecutive
//! elements the same way: an index into a shared grid (`crate::iq_grids`)
//! names eight magnitudes, signs (IQ2/IQ3) or a delta (IQ1) are applied,
//! and the group is scaled. So each kind is written here as ONE pair of
//! MSL functions over the block bytes:
//!
//! - `group(b, g, v)`: the eight raw values of group `g` (0..32), the
//!   signed grid magnitudes, or for IQ1 the grid's `0..2` nibbles;
//! - `affine(b, g)`: `(mul, add)` with element `= mul * v + add`. Only
//!   IQ1's `add` is nonzero (`dl * (-1 +- delta)`).
//!
//! Everything else is generated from that pair: the 16-value dequant
//! functor ([`IqDequant`]) the simdgroup GEMM, `mul_mm_id` and the
//! embedding gather (`crate::embd`) instantiate, and the decode matvec
//! `iq_matvec_impl`, from which `crate::moe_mv_id` generates the MoE
//! `mul_mv_id` kernels.
//!
//! The functors follow ggml's Metal `dequantize_iq*` at the pin field
//! for field, and the dequant is bit-identical to it: ggml's
//! `dl * grid[i] * (sign ? -1 : 1)` is `mul * v` with `v = +-grid[i]`,
//! and its `dl * nibble + ml` is `mul * v + add`. The block layouts are
//! `frink_quant`'s `for_each_iq*` walks (the CPU path).
//!
//! The matvec has the geometry of ggml's `kernel_mul_mv_iq*_f32`
//! without the split variant (`N_R0 = 4`, `N_SG = 2`): two simdgroups of
//! four rows, eight rows and 64 threads a threadgroup, lane `t` taking
//! sub-blocks `t, t + 32, ...` of the row with its 32 activations in
//! registers. It differs from ggml's in two ways, neither of which
//! changes a dequantized value:
//!
//! - ggml folds each kind's scale differently around the dot (and
//!   applies IQ2's `0.25` and IQ3_XXS's `0.5` once, at the store); this
//!   one applies each group's `mul` to that group's dot. The products
//!   are the same, the rounding order differs, as ADR 0009 allows.
//! - ggml stages some grids in threadgroup memory; this one reads them
//!   from `constant` memory, as ggml's own IQ2_S and IQ1 matvecs do. A
//!   speed choice, left for after coverage (ADR 0010).
//!
//! A row past `n_rows` is clamped to the last row before it is read
//! (ggml's buffers are padded, these are not); its result is dropped.

use crate::gen_kinds::{gen_launchers, Family, GenKind, KINDS};

/// The seven formats' `group` / `affine` pairs and the dequant functor
/// built from them. Spliced into the GEMM translation unit
/// (`gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC`) after the grid tables, and
/// prepended to the matvec unit.
const IQ_FUNCTORS_MSL: &str = r#"
static inline uint iq_u16(device const uchar* p) {
    return uint(p[0]) | (uint(p[1]) << 8);
}
static inline uint iq_u32(device const uchar* p) {
    return iq_u16(p) | (iq_u16(p + 2) << 16);
}
static inline float iq_d(device const uchar* b) {
    return float(*(device const half*)b);
}
// Eight signed magnitudes of an eight-byte grid row.
static inline void iq_signed8(constant uchar* grid, uint signs, thread float* v) {
    for (short j = 0; j < 8; ++j) {
        v[j] = float(grid[j]) * (signs & IQ_KMASK[j] ? -1.f : 1.f);
    }
}
// Two four-byte grid rows, eight signs.
static inline void iq_signed4x2(constant uchar* g1, constant uchar* g2, uint signs,
                                thread float* v) {
    for (short j = 0; j < 4; ++j) {
        v[j]     = float(g1[j]) * (signs & IQ_KMASK[j]     ? -1.f : 1.f);
        v[j + 4] = float(g2[j]) * (signs & IQ_KMASK[j + 4] ? -1.f : 1.f);
    }
}
// An IQ1 grid row: element i in the low nibble of byte i, i + 4 high.
static inline void iq_nibbles8(constant uchar* grid, thread float* v) {
    for (short j = 0; j < 4; ++j) {
        v[j]     = float(grid[j] & 0xF);
        v[j + 4] = float(grid[j] >> 4);
    }
}

// block_iq2_xxs (66 bytes): half d, then per sub-block four grid bytes
// and a u32 of four 7-bit sign indices and a 4-bit scale.
struct IQ2XXS {
    static constexpr constant short BLOCK_BYTES = 66;
    static inline void group(device const uchar* b, short g, thread float* v) {
        const short ib = g / 4, l = g % 4;
        const uint aux = iq_u32(b + 2 + 8 * ib + 4);
        constant uchar* grid = (constant uchar*)(IQ2XXS_GRID + b[2 + 8 * ib + l]);
        iq_signed8(grid, IQ_KSIGNS[(aux >> (7 * l)) & 127], v);
    }
    static inline float2 affine(device const uchar* b, short g) {
        const uint aux = iq_u32(b + 2 + 8 * (g / 4) + 4);
        return float2(iq_d(b) * (0.5f + float(aux >> 28)) * 0.25f, 0.f);
    }
};

// block_iq2_xs (74 bytes): half d, 32 u16 codes (9-bit grid index,
// 7-bit sign index), 8 scale bytes (a nibble per 16 elements).
struct IQ2XS {
    static constexpr constant short BLOCK_BYTES = 74;
    static inline void group(device const uchar* b, short g, thread float* v) {
        const uint code = iq_u16(b + 2 + 2 * g);
        constant uchar* grid = (constant uchar*)(IQ2XS_GRID + (code & 511));
        iq_signed8(grid, IQ_KSIGNS[code >> 9], v);
    }
    static inline float2 affine(device const uchar* b, short g) {
        const uint s = (b[66 + g / 4] >> (4 * ((g % 4) / 2))) & 0xF;
        return float2(iq_d(b) * (0.5f + float(s)) * 0.25f, 0.f);
    }
};

// block_iq2_s (82 bytes): half d, 32 low index bytes, 32 sign bytes,
// 8 qh bytes (two high index bits per group), 8 scale bytes.
struct IQ2S {
    static constexpr constant short BLOCK_BYTES = 82;
    static inline void group(device const uchar* b, short g, thread float* v) {
        const short ib = g / 4, l = g % 4;
        const uint idx = uint(b[2 + g]) | ((uint(b[66 + ib]) << (8 - 2 * l)) & 0x300);
        iq_signed8((constant uchar*)(IQ2S_GRID + idx), b[34 + g], v);
    }
    static inline float2 affine(device const uchar* b, short g) {
        const uint s = (b[74 + g / 4] >> (4 * ((g % 4) / 2))) & 0xF;
        return float2(iq_d(b) * (0.5f + float(s)) * 0.25f, 0.f);
    }
};

// block_iq3_xxs (98 bytes): half d, 64 grid bytes (one per four
// elements), then per sub-block a u32 of four 7-bit sign indices and a
// 4-bit scale.
struct IQ3XXS {
    static constexpr constant short BLOCK_BYTES = 98;
    static inline void group(device const uchar* b, short g, thread float* v) {
        const short ib = g / 4, l = g % 4;
        const uint aux = iq_u32(b + 66 + 4 * ib);
        constant uchar* g1 = (constant uchar*)(IQ3XXS_GRID + b[2 + 2 * g]);
        constant uchar* g2 = (constant uchar*)(IQ3XXS_GRID + b[2 + 2 * g + 1]);
        iq_signed4x2(g1, g2, IQ_KSIGNS[(aux >> (7 * l)) & 127], v);
    }
    static inline float2 affine(device const uchar* b, short g) {
        const uint aux = iq_u32(b + 66 + 4 * (g / 4));
        return float2(iq_d(b) * (0.5f + float(aux >> 28)) * 0.5f, 0.f);
    }
};

// block_iq3_s (110 bytes): half d, 64 low index bytes, 8 qh bytes (the
// ninth index bit per grid row), 32 sign bytes, 4 scale bytes (a nibble
// per sub-block, scale 1 + 2 s).
struct IQ3S {
    static constexpr constant short BLOCK_BYTES = 110;
    static inline void group(device const uchar* b, short g, thread float* v) {
        const short ib = g / 4, l = g % 4;
        const uint qh = b[66 + ib];
        const uint i1 = uint(b[2 + 2 * g])     | ((qh << (8 - 2 * l)) & 256);
        const uint i2 = uint(b[2 + 2 * g + 1]) | ((qh << (7 - 2 * l)) & 256);
        iq_signed4x2((constant uchar*)(IQ3S_GRID + i1), (constant uchar*)(IQ3S_GRID + i2),
                     b[74 + g], v);
    }
    static inline float2 affine(device const uchar* b, short g) {
        const short ib = g / 4;
        const int s = (b[106 + ib / 2] >> (4 * (ib % 2))) & 0xF;
        return float2(iq_d(b) * float(1 + 2 * s), 0.f);
    }
};

// block_iq1_s (50 bytes): half d, 32 low index bytes, then per
// sub-block a u16: four 3-bit high index fields, a 3-bit scale and the
// delta's sign.
struct IQ1S {
    static constexpr constant short BLOCK_BYTES = 50;
    static inline void group(device const uchar* b, short g, thread float* v) {
        const short l = g % 4;
        const uint qh = iq_u16(b + 34 + 2 * (g / 4));
        const uint idx = uint(b[2 + g]) | (((qh >> (3 * l)) & 7) << 8);
        iq_nibbles8((constant uchar*)(IQ1S_GRID_GPU + idx), v);
    }
    static inline float2 affine(device const uchar* b, short g) {
        const uint qh = iq_u16(b + 34 + 2 * (g / 4));
        const float dl = iq_d(b) * float(2 * int((qh >> 12) & 7) + 1);
        return float2(dl, dl * (qh & 0x8000 ? -1.f - IQ1_DELTA : -1.f + IQ1_DELTA));
    }
};

// block_iq1_m (56 bytes): 32 low index bytes, 16 qh bytes (per group a
// 3-bit high index field and the delta's sign), and four u16 scale
// words whose top nibbles are the f16 super-scale.
struct IQ1M {
    static constexpr constant short BLOCK_BYTES = 56;
    static inline void group(device const uchar* b, short g, thread float* v) {
        const uint qh = b[32 + g / 2];
        const uint idx = uint(b[g]) | ((qh << (g % 2 ? 4 : 8)) & 0x700);
        iq_nibbles8((constant uchar*)(IQ1S_GRID_GPU + idx), v);
    }
    static inline float2 affine(device const uchar* b, short g) {
        const short ib = g / 4, l = g % 4;
        const uint s0 = iq_u16(b + 48), s1 = iq_u16(b + 50);
        const uint s2 = iq_u16(b + 52), s3 = iq_u16(b + 54);
        const ushort bits = ushort((s0 >> 12) | ((s1 >> 8) & 0x00F0)
                                   | ((s2 >> 4) & 0x0F00) | (s3 & 0xF000));
        const float d = float(as_type<half>(bits));
        const uint sc = iq_u16(b + 48 + 2 * (ib / 2));
        const float dl = d * float(2 * int((sc >> (6 * (ib % 2) + 3 * (l / 2))) & 7) + 1);
        const uint qh = b[32 + g / 2];
        const bool neg = qh & (g % 2 ? 0x80 : 0x08);
        return float2(dl, dl * (neg ? -1.f - IQ1_DELTA : -1.f + IQ1_DELTA));
    }
};

// ggml `dequantize_iq*`: sixteen values, groups 2 il and 2 il + 1.
template<typename K>
struct IqDequant {
    static constexpr constant short NL = 16;
    static constexpr constant short BLOCK_BYTES = K::BLOCK_BYTES;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        for (short h = 0; h < 2; ++h) {
            const short g = 2 * il + h;
            float v[8];
            K::group(xb, g, v);
            const float2 a = K::affine(xb, g);
            for (short i = 0; i < 8; ++i) {
                reg[(8 * h + i) / 4][i % 4] = a.x * v[i] + a.y;
            }
        }
    }
};
"#;

/// The decode matvec, one body for the seven (see the module docs).
/// `table` is the id kernels' shared argument (`crate::moe_mv_id`);
/// nothing is staged in it.
const IQ_MATVEC_BODY_MSL: &str = r#"
template<typename K>
static inline void iq_matvec_impl(
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

    const uint nb32 = n_blocks_per_row * 8u;
    float yl[32];
    float sumf[NR] = {0.f, 0.f, 0.f, 0.f};
    device const float* y = x + 32u * tiisg;

    for (uint ib32 = tiisg; ib32 < nb32; ib32 += 32u) {
        float4 sumy = float4(0.f);
        for (short i = 0; i < 32; ++i) {
            yl[i] = y[i];
            sumy[i / 8] += yl[i];
        }
        const uint ibl = ib32 / 8u;
        const short ib = short(ib32 % 8u);
        for (uint r = 0u; r < NR; ++r) {
            device const uchar* blk = row_ptr[r] + (size_t)ibl * uint(K::BLOCK_BYTES);
            float acc = 0.f;
            for (short l = 0; l < 4; ++l) {
                const short g = 4 * ib + l;
                float v[8];
                K::group(blk, g, v);
                float s = 0.f;
                for (short j = 0; j < 8; ++j) {
                    s += yl[8 * l + j] * v[j];
                }
                const float2 a = K::affine(blk, g);
                acc += a.x * s + a.y * sumy[l];
            }
            sumf[r] += acc;
        }
        y += 32u * 32u;
    }

    for (uint r = 0u; r < NR; ++r) {
        const float tot = simd_sum(sumf[r]);
        if (tiisg == 0u && first_row + r < n_rows) {
            out[first_row + r] = tot;
        }
    }
}

#define IQ_MATVEC_ENTRY(NAME, K)                                            \
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
    iq_matvec_impl<K>(weights, x, out, row_bytes, n_blocks_per_row,         \
                      n_rows, tgpig, tiisg, sgitg, nullptr);                \
}
"#;

/// The functors, after the grid tables, for the GEMM translation unit.
pub(crate) fn gemm_functors_msl() -> &'static str {
    use std::sync::LazyLock;
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        let mut s = format!(
            "{}\nconstant float IQ1_DELTA = 0.125f;\n{IQ_FUNCTORS_MSL}",
            crate::iq_grids::iq_grids_msl()
        );
        for (k, group) in kinds() {
            s.push_str(&format!("typedef IqDequant<{group}> {};\n", k.dequant));
        }
        Box::leak(s.into_boxed_str())
    });
    &SRC
}

/// The matvec translation unit: tables, functors, the body, a matvec
/// and both MoE id entries per kind.
pub(crate) fn iq_matvec_src() -> &'static str {
    use std::sync::LazyLock;
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        let mut s = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n{}\n{}\n{IQ_MATVEC_BODY_MSL}",
            gemm_functors_msl(),
            crate::moe_mv_id::MOE_MV_ID_MSL
        );
        for (k, f) in kinds() {
            s.push_str(&format!(
                "IQ_MATVEC_ENTRY({}, {f})\nMOE_MV_ID_ENTRIES({}, iq_matvec_impl<{f}>)\n",
                k.matvec, k.prefix
            ));
        }
        Box::leak(s.into_boxed_str())
    });
    &SRC
}

gen_launchers! {
    "IQ2_XXS", launch_iq2_xxs_matvec, launch_iq2_xxs_mul_mm_sg;
    "IQ2_XS", launch_iq2_xs_matvec, launch_iq2_xs_mul_mm_sg;
    "IQ2_S", launch_iq2_s_matvec, launch_iq2_s_mul_mm_sg;
    "IQ3_XXS", launch_iq3_xxs_matvec, launch_iq3_xxs_mul_mm_sg;
    "IQ3_S", launch_iq3_s_matvec, launch_iq3_s_mul_mm_sg;
    "IQ1_S", launch_iq1_s_matvec, launch_iq1_s_mul_mm_sg;
    "IQ1_M", launch_iq1_m_matvec, launch_iq1_m_mul_mm_sg;
}

/// The IQ rows of the shared table, with their `group` / `affine` struct.
pub(crate) fn kinds() -> impl Iterator<Item = (&'static GenKind, &'static str)> {
    KINDS.iter().filter_map(|k| match k.family {
        Family::Iq { group } => Some((k, group)),
        Family::Functor => None,
    })
}

#[cfg(test)]
/// Test blocks of `name`: every grid and sign byte random, every scale field realistic, so the
/// GEMM's `half` tiles stay in range. Random IQ1_M scale words would
/// put a random f16 (often a NaN) in the super-scale, so their top
/// nibbles are overwritten with `d`'s bits.
pub(crate) fn test_blocks(name: &str, seed: u32, n_blocks: usize) -> Vec<u8> {
    let k = crate::gen_kinds::by_name(name).expect("an IQ kind");
    let bb = k.block_bytes;
    let mut w = crate::k_quant_check::pseudo_bytes(seed, n_blocks * bb);
    for b in 0..n_blocks {
        let at = b * bb;
        let jitter = (b % 7) as f32;
        let d = match name {
            "IQ3_S" => 0.0015 + 0.0002 * jitter,
            "IQ3_XXS" => 0.004 + 0.0005 * jitter,
            _ => 0.008 + 0.001 * jitter,
        };
        let h = half::f16::from_f32(d);
        if name == "IQ1_M" {
            let bits = h.to_bits();
            for i in 0..4 {
                let nib = ((bits >> (4 * i)) & 0xF) as u8;
                let byte = &mut w[at + 49 + 2 * i];
                *byte = (*byte & 0x0F) | (nib << 4);
            }
        } else {
            w[at..at + 2].copy_from_slice(&h.to_le_bytes());
        }
    }
    w
}

#[cfg(test)]
mod tests;
