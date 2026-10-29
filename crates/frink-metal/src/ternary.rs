//! PrismML's `PTQ1_0` on Metal: the trit decode ONCE, as the functor the
//! simdgroup GEMM instantiates and the body the matvec calls.
//!
//! The format is `frink_quant::ternary`'s (`block_ptq1_0 { qs[24];
//! qh[2]; half d }`, 128 weights, five trits a byte in `qs` and four in
//! `qh`, the scale LAST); the element order is the one that module's
//! `unpack_trits` walks and its doc comment derives from the fork's
//! `dequantize_row_ptq1_0`. Sixteen consecutive elements `[16 il, 16 il
//! + 16)` are: for `il < 5`, trit `il` of the first sixteen `qs` bytes;
//! for `il = 5, 6`, trits `2(il-5)` and `2(il-5)+1` of `qs[16..24]`;
//! for `il = 7`, trit 4 of `qs[16..24]` and then the eight `qh` values
//! (`120 + 2n + h`). A trit is read as `((q * 3^n) mod 256) * 3 >> 8`,
//! exactly the u8 trick the fork uses, so the two decoders cannot
//! disagree on a byte the quantizer produced.
//!
//! The Hadamard rotation Bonsai folds into these weights is not here:
//! it is undone on the activation before any launch
//! (`frink_core::weight_matrix::hadamard`).

/// The trit decode and the PTQ1_0 and TQ1_0 GEMM functors, spliced into
/// the GEMM translation unit (`gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC`) and
/// prepended to the PTQ1_0 matvec's and to `crate::dq_mv`'s, which runs
/// TQ1_0's matvec on its functor.
pub const PTQ1_0_DEQUANT_MSL: &str = r#"
// One trit of a five-trit (qs) or four-trit (qh) byte, as
// -1 / 0 / +1. `p` is 3^n for trit n, most significant first.
constant uchar PTQ1_0_POW3[5] = {1, 3, 9, 27, 81};
static inline float ptq1_0_trit(uchar q, uchar p) {
    const uint qp = (uint(q) * uint(p)) & 0xFFu;
    return float((qp * 3u) >> 8) - 1.0f;
}

// Sixteen consecutive dequantized values of sub-block `il` (0..8) of a
// 128-element PTQ1_0 block; see the module doc for the element order.
static inline void ptq1_0_dequant_16(device const uchar* xb, short il, thread float4x4& reg) {
    const float d = float(*(device const half*)(xb + 26));
    if (il < 5) {
        const uchar p = PTQ1_0_POW3[il];
        for (short m = 0; m < 16; ++m) {
            reg[m / 4][m % 4] = d * ptq1_0_trit(xb[m], p);
        }
    } else if (il < 7) {
        const short n0 = (il - 5) * 2;
        for (short j = 0; j < 2; ++j) {
            const uchar p = PTQ1_0_POW3[n0 + j];
            for (short m = 0; m < 8; ++m) {
                const short e = j * 8 + m;
                reg[e / 4][e % 4] = d * ptq1_0_trit(xb[16 + m], p);
            }
        }
    } else {
        for (short m = 0; m < 8; ++m) {
            reg[m / 4][m % 4] = d * ptq1_0_trit(xb[16 + m], PTQ1_0_POW3[4]);
        }
        for (short n = 0; n < 4; ++n) {
            for (short h = 0; h < 2; ++h) {
                const short e = 8 + n * 2 + h;
                reg[e / 4][e % 4] = d * ptq1_0_trit(xb[24 + h], PTQ1_0_POW3[n]);
            }
        }
    }
}

struct PTQ1_0Dequant {
    static constexpr constant short NL = 8;
    static constexpr constant short BLOCK_BYTES = 28;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        ptq1_0_dequant_16(xb, il, reg);
    }
};

// ggml TQ1_0 (`block_tq1_0 { qs[48]; qh[4]; half d }`, 256 elements):
// the same trit codec over a wider block. ggml `dequantize_row_tq1_0`
// walks one stage of 32 qs bytes (element 32 n + m is trit n of qs[m]),
// one of 16 (160 + 16 n + m, trit n of qs[32 + m]), then qh (240 + 4 n
// + h, trit n of qh[h]). Sixteen elements [16 il, 16 il + 16): il < 10
// is trit il / 2 of qs[16 (il % 2) ..], il < 15 trit il - 10 of
// qs[32 ..], il = 15 the qh tail.
struct TQ1_0Dequant {
    static constexpr constant short NL = 16;
    static constexpr constant short BLOCK_BYTES = 54;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        const float d = float(*(device const half*)(xb + 52));
        if (il < 10) {
            const uchar p = PTQ1_0_POW3[il / 2];
            device const uchar* q = xb + 16 * (il % 2);
            for (short m = 0; m < 16; ++m) {
                reg[m / 4][m % 4] = d * ptq1_0_trit(q[m], p);
            }
        } else if (il < 15) {
            const uchar p = PTQ1_0_POW3[il - 10];
            for (short m = 0; m < 16; ++m) {
                reg[m / 4][m % 4] = d * ptq1_0_trit(xb[32 + m], p);
            }
        } else {
            for (short e = 0; e < 16; ++e) {
                reg[e / 4][e % 4] = d * ptq1_0_trit(xb[48 + e % 4], PTQ1_0_POW3[e / 4]);
            }
        }
    }
};
"#;

/// The GEMM entry lines, spliced after the shared body's other entries.
pub const PTQ1_0_GEMM_ENTRIES_MSL: &str = r#"
MUL_MM_SG_ENTRY(ptq1_0_mul_mm_sg, PTQ1_0Dequant)
MUL_MM_SG_F16_ENTRY(ptq1_0_mul_mm_sg_f16, PTQ1_0Dequant)
"#;

/// The decode matvec: two simdgroups of four rows each (`rows_per_tg
/// 8`, 64 threads, the `q4_0_matvec` geometry). Each lane owns the
/// blocks `lane, lane + 32, ...` of its rows; for every sub-block of
/// sixteen the activation slice is loaded once and dotted against each
/// of the four rows' decoded values, then the lanes reduce with
/// `simd_sum`.
pub fn ptq1_0_matvec_src() -> &'static str {
    use std::sync::LazyLock;
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        Box::leak(
            format!(
                "#include <metal_stdlib>\nusing namespace metal;\n{PTQ1_0_DEQUANT_MSL}\n{PTQ1_0_MATVEC_BODY_MSL}"
            )
            .into_boxed_str(),
        )
    });
    &SRC
}

/// The matvec, the fork's `kernel_mul_mv_ptq1_0_f32` shape
/// (`kernels/mul_mv.metal`): eight lanes cover one block and each owns
/// WHOLE BYTES (`qs[2it]`, `qs[2it+1]`, `qs[16+it]`, one trit of `qh`),
/// so a block's 26 bytes are read once across the eight rather than
/// five times by one lane; four blocks per simdgroup step; four rows per
/// threadgroup sharing the staged activations. The trit is peeled on
/// the float pipe: with `u = q/256` and `g_k = floor(3^k u)` (exact,
/// `3^5 q < 2^16`), trit `n` is `g_{n+1} - 3 g_n`, and the byte's dot
/// collapses to `sum_{k=1..4} g_k (y_{k-1} - 3 y_k) + g_5 y_4`, whose
/// coefficients depend on the activations alone and are staged once
/// per block for all four rows. The `-1` offset is `sumy` subtracted
/// once. The first version of this kernel gave a lane a whole block and
/// decoded with integer ops; it reached 2.4 tok/s on Bonsai-2-27B
/// where the fork reaches 11 on the same GPU.
const PTQ1_0_MATVEC_BODY_MSL: &str = r#"
// One row's five bytes, read BEFORE any of them is used.
//
// The bytes a lane needs from a block are five separate non-contiguous
// loads, and the kernel's four rows used to be loaded and consumed one
// at a time: one memory request in flight per lane, with the whole
// float-pipe decode waiting on it. The probe said what that costs --
// PTQ1_0 at Bonsai's `17408x5120` took 1.004 ms against Q4_0's 0.897 on
// the same shape while reading 2.6x FEWER bytes -- so the kernel is
// latency-bound and not bandwidth-bound, and the fix is to have four
// rows' requests outstanding at once.
struct ptq1_0_row {
    ushort b01; // qs[2 it] and qs[2 it + 1], one aligned load
    uchar b2;   // qs[16 + it]
    uchar bh;   // qh[it & 1]
    half  d;    // the block scale
};

static inline ptq1_0_row ptq1_0_load_row(device const uchar* qb, short it) {
    ptq1_0_row r;
    // `2 * it` is even, so the pair of bytes this lane owns is one
    // aligned 16-bit load rather than two 8-bit ones. The memory side
    // of this kernel measures 34 GB/s against Q4_0's 55 on the same
    // rows, with the decode removed, so the load COUNT is what it is
    // bound by and not the byte count.
    r.b01 = *(device const ushort*)(qb + 2 * it);
    r.b2 = qb[16 + it];
    r.bh = qb[24 + (it & 1)];
    r.d  = *(device const half*)(qb + 26);
    return r;
}

static inline float ptq1_0_dot_loaded(ptq1_0_row r, thread const float* yl, float sumy) {
    float acc = 0.0f;
    const uchar bs[2] = { uchar(r.b01 & 0xFFu), uchar(r.b01 >> 8) };
    for (short k = 0; k < 2; ++k) {
        const float u = float(bs[k]) * (1.0f / 256.0f);
        thread const float* c = yl + 5 * k;
        acc += floor(  3.0f * u) * c[0];
        acc += floor(  9.0f * u) * c[1];
        acc += floor( 27.0f * u) * c[2];
        acc += floor( 81.0f * u) * c[3];
        acc += floor(243.0f * u) * c[4];
    }
    {
        const float u = float(r.b2) * (1.0f / 256.0f);
        thread const float* c = yl + 10;
        acc += floor(  3.0f * u) * c[0];
        acc += floor(  9.0f * u) * c[1];
        acc += floor( 27.0f * u) * c[2];
        acc += floor( 81.0f * u) * c[3];
        acc += floor(243.0f * u) * c[4];
    }
    {
        const float u  = float(r.bh) * (1.0f / 256.0f);
        const float p0 = yl[16];
        acc += (floor(3.0f * p0 * u) - 3.0f * floor(p0 * u)) * yl[15];
    }
    return (acc - sumy) * float(r.d);
}

static inline float ptq1_0_dot_coeffs(device const uchar* qb, thread const float* yl, float sumy, short it) {
    float acc = 0.0f;
    for (short k = 0; k < 2; ++k) {
        const float u = float(qb[2 * it + k]) * (1.0f / 256.0f);
        thread const float* c = yl + 5 * k;
        acc += floor(  3.0f * u) * c[0];
        acc += floor(  9.0f * u) * c[1];
        acc += floor( 27.0f * u) * c[2];
        acc += floor( 81.0f * u) * c[3];
        acc += floor(243.0f * u) * c[4];
    }
    {
        const float u = float(qb[16 + it]) * (1.0f / 256.0f);
        thread const float* c = yl + 10;
        acc += floor(  3.0f * u) * c[0];
        acc += floor(  9.0f * u) * c[1];
        acc += floor( 27.0f * u) * c[2];
        acc += floor( 81.0f * u) * c[3];
        acc += floor(243.0f * u) * c[4];
    }
    {
        // qh: eight elements, one per lane, trit it>>1 of byte qh[it&1].
        const float u  = float(qb[24 + (it & 1)]) * (1.0f / 256.0f);
        const float p0 = yl[16];
        acc += (floor(3.0f * p0 * u) - 3.0f * floor(p0 * u)) * yl[15];
    }
    const float d = float(*(device const half*)(qb + 26));
    return (acc - sumy) * d;
}

kernel void ptq1_0_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint NR = 4u;
    constexpr uint NSG = 1u;
    const uint first_row = (tgpig * NSG + sg) * NR;
    if (first_row >= n_rows) return;

    const short ix = short(lane / 8u);
    const short it = short(lane % 8u);

    // 15 collapse coefficients, the qh activation, and this lane's 3^n.
    float yl[17];
    float sumf[NR] = { 0.0f };
    {
        const float pow3f[4] = { 1.0f, 3.0f, 9.0f, 27.0f };
        yl[16] = pow3f[it >> 1];
    }

    device const float* yb = x + ix * 128;
    for (uint ib = uint(ix); ib < n_blocks_per_row; ib += 4u) {
        float sumy = 0.0f;
        for (short k = 0; k < 2; ++k) {
            const short m = 2 * it + k;
            float y[5];
            for (short n = 0; n < 5; ++n) {
                y[n] = yb[n * 16 + m];
                sumy += y[n];
            }
            for (short n = 0; n < 4; ++n) {
                yl[5 * k + n] = y[n] - 3.0f * y[n + 1];
            }
            yl[5 * k + 4] = y[4];
        }
        {
            float y[5];
            for (short n = 0; n < 5; ++n) {
                y[n] = yb[80 + n * 8 + it];
                sumy += y[n];
            }
            for (short n = 0; n < 4; ++n) {
                yl[10 + n] = y[n] - 3.0f * y[n + 1];
            }
            yl[14] = y[4];
        }
        {
            const float v = yb[120 + it];
            yl[15] = v;
            sumy += v;
        }
        // All four rows' bytes requested first, then decoded: four
        // outstanding requests per lane instead of one.
        ptq1_0_row rows[NR];
        #pragma unroll
        for (uint rr = 0u; rr < NR; ++rr) {
            const uint row = min(first_row + rr, n_rows - 1u);
            device const uchar* block = weights + (size_t)row * row_bytes + (size_t)ib * 28u;
            rows[rr] = ptq1_0_load_row(block, it);
        }
        #pragma unroll
        for (uint rr = 0u; rr < NR; ++rr) {
            if (first_row + rr >= n_rows) continue;
            sumf[rr] += ptq1_0_dot_loaded(rows[rr], yl, sumy);
        }
        yb += 128 * 4;
    }
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = first_row + rr;
        const float tot = simd_sum(sumf[rr]);
        if (lane == 0u && row < n_rows) {
            out[row] = tot;
        }
    }
}
"#;
