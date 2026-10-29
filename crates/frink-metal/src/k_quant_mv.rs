//! The Q5_K, Q6_K and IQ4_XS decode matvecs, each written once as an
//! `_impl` function: the plain matvec is a thin kernel over it, and the
//! MoE id kernels (`crate::moe_mv_id`) are generated from it (#448).
//! Moved out of `crate::gpu` unchanged apart from that split.

use std::sync::LazyLock;

/// ggml-metal `kernel_mul_mv_q5_K_f32` port: `N_R0=1` row per simdgroup,
/// `NSG=2` simdgroups per TG (2 rows / 64 threads). Register-local
/// `yl`/`yh` activation packs with Q5_K's 5th-bit `qh` plane — same
/// dequant identity as `frink_quant::dot_q5_k_f32_scalar`. Host
/// dispatches `ceil(n_rows/2)` threadgroups of 64 threads.
///
/// `N_R0` stays at 1 (not 2 like Q4_K/Q6_K): ggml-metal reduced it after
/// a real register-spill regression (`llama.cpp` #20399).
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// `launch_q5_k_matvec_matches_cpu_reference`).
const Q5_K_MATVEC_BODY_MSL: &str = r#"


static inline void q5_k_matvec_impl(
    device const uchar* weights,
    device const float* x,
    device float* out,
    uint row_bytes,
    uint n_blocks_per_row,
    uint n_rows,
    uint tgpig,
    ushort tiisg,
    ushort sgitg,
    threadgroup float* table
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 1;
    constexpr short NW = 32;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;


    const short tid = tiisg / 4;
    const short ix = tiisg % 4;
    const short iq = tid / 4;
    const short ir = tid % 4;

    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uchar hm1 = uchar(1u << (2 * iq));
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    const int first_row = int(tgpig * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);

    if (first_row >= int(n_rows)) {
        return;
    }

    float sumf = 0.0f;
    float yl[16];
    float yh[16];

    uint16_t sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    device const float* y1 = x + ix * 256 + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uchar* block0 =
            weights + (size_t)first_row * row_bytes + (size_t)i * 176u;
        device const uchar* q1 = block0 + 48 + q_offset;
        device const uchar* qh = block0 + 16 + l0;
        device const half* dh = (device const half*)(block0);
        device const uint16_t* a =
            (device const uint16_t*)(block0 + 4) + iq;

        device const float* y2 = y1 + 128;
        float4 sumy = float4(0.0f);
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];
            sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32];
            sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l + 0];
            sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32];
            sumy[3] += yh[l + 8];
        }

        device const uchar* q2 = q1 + 64;

        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        float4 acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f);
        for (short l = 0; l < 8; ++l) {
            uchar h = qh[l];
            acc1[0] += yl[l + 0] * float(q1[l] & 0x0F);
            acc1[1] += yl[l + 8] * float(q1[l] & 0xF0);
            acc1[2] += yh[l + 0] * float(q2[l] & 0x0F);
            acc1[3] += yh[l + 8] * float(q2[l] & 0xF0);
            acc2[0] += (h & hm1) ? yl[l + 0] : 0.0f;
            acc2[1] += (h & hm2) ? yl[l + 8] : 0.0f;
            acc2[2] += (h & hm3) ? yh[l + 0] : 0.0f;
            acc2[3] += (h & hm4) ? yh[l + 8] : 0.0f;
        }

        sumf += float(dh[0])
                * (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0])
                    + float(sc8[1]) * (acc1[1] / 16.0f + 16.0f * acc2[1])
                    + float(sc8[4]) * (acc1[2] + 16.0f * acc2[2])
                    + float(sc8[5]) * (acc1[3] / 16.0f + 16.0f * acc2[3]))
            - float(dh[1])
                * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3])
                    + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

        y1 += 4 * 256;
    }

    float sum_all = simd_sum(sumf);
    if (tiisg == 0 && first_row < int(n_rows)) {
        out[first_row] = sum_all;
    }
}

kernel void q5_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid_tg [[thread_position_in_threadgroup]]
) {
    q5_k_matvec_impl(weights, x, out, row_bytes, n_blocks_per_row, n_rows, tgpig,
                       ushort(tid_tg % 32u), ushort(tid_tg / 32u), nullptr);
}

MOE_MV_ID_ENTRIES(q5_k, q5_k_matvec_impl)
"#;

/// The q5_k matvec translation unit, with its MoE id entries.
/// Built once and leaked: the pipeline cache keys on a `&'static str`.
pub fn q5_k_matvec_src() -> &'static str {
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        Box::leak(
            format!(
                "#include <metal_stdlib>\nusing namespace metal;\n{}\n{}",
                crate::moe_mv_id::MOE_MV_ID_MSL,
                Q5_K_MATVEC_BODY_MSL
            )
            .into_boxed_str(),
        )
    });
    &SRC
}

/// ggml-metal `kernel_mul_mv_q6_K_f32` port: `N_R0=2` rows per simdgroup,
/// `NSG=2` simdgroups per TG (4 rows / 64 threads). Register-local `yl`
/// packs; signed int8 sub-block scales match
/// `frink_quant::dot_q6_k_f32_scalar`. Host dispatches `ceil(n_rows/4)`
/// threadgroups of 64 threads.
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// `launch_q6_k_matvec_matches_cpu_reference`, which specifically
/// includes a negative-scale byte in its test data).
const Q6_K_MATVEC_BODY_MSL: &str = r#"


static inline void q6_k_matvec_impl(
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
    constexpr short NSG = 2;
    constexpr short nr0 = 2;
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int first_row = int(tgpig * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);

    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8; // 0 or 1
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;

    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    float sumf[2] = {0.0f, 0.0f};
    float yl[16];

    for (int i = ix; i < nb; i += 2) {
        device const uchar* block0 =
            weights + (size_t)first_row * row_bytes + (size_t)i * 210u;
        device const uchar* q1 = block0 + q_offset_l;
        device const uchar* q2 = q1 + 32;
        device const uchar* qh = block0 + 128 + q_offset_h;
        device const char* sc = (device const char*)(block0 + 192 + is);
        device const half* dh = (device const half*)(block0 + 208);

        device const float* y = x + i * 256 + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l + 0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = float4(0.0f);

            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0]
                    * float(int((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1]
                    * float(int((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2]
                    * float(int((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3]
                    * float(int((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += float(dh[0])
                * (sums[0] * float(sc[0]) + sums[1] * float(sc[2])
                    + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));

            q1 += row_bytes;
            q2 += row_bytes;
            qh += row_bytes;
            sc += row_bytes;
            dh += row_bytes / 2;
        }
    }

    for (int row = 0; row < nr0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < int(n_rows)) {
            out[first_row + row] = sum_all;
        }
    }
}

kernel void q6_k_matvec(
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
    q6_k_matvec_impl(weights, x, out, row_bytes, n_blocks_per_row, n_rows, tgpig,
                       tiisg, sgitg, nullptr);
}

MOE_MV_ID_ENTRIES(q6_k, q6_k_matvec_impl)
"#;

/// The q6_k matvec translation unit, with its MoE id entries.
/// Built once and leaked: the pipeline cache keys on a `&'static str`.
pub fn q6_k_matvec_src() -> &'static str {
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        Box::leak(
            format!(
                "#include <metal_stdlib>\nusing namespace metal;\n{}\n{}",
                crate::moe_mv_id::MOE_MV_ID_MSL,
                Q6_K_MATVEC_BODY_MSL
            )
            .into_boxed_str(),
        )
    });
    &SRC
}

/// ggml-metal `kernel_mul_mv_iq4_xs_f32` port: `N_R0=2` rows per
/// simdgroup, `NSG=2` simdgroups per TG (4 rows / 64 threads). IQ4_XS
/// blocks are 136 bytes / 256 elements: f16 super-scale `d`, 16 bits of
/// high scale bits, 4 bytes of low scale nibbles, then 128 bytes of
/// 4-bit indices into the shared IQ4_NL non-linear codebook (loaded
/// into 32 floats of threadgroup memory, one copy per 16 lanes). Each
/// lane owns 8 consecutive bytes of `qs` (16 elements: low nibbles →
/// elems j, high nibbles → elems j+16 of the 32-elem sub-block), with
/// odd/even lanes-of-16 walking odd/even blocks. Same dequant identity
/// as `frink_quant::dot_iq4_xs_f32`.
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference (see
/// `launch_iq4_xs_matvec_matches_cpu_reference`).
const IQ4_XS_MATVEC_BODY_MSL: &str = r#"


constant float kvalues_iq4nl_f[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

static inline void iq4_xs_matvec_impl(
    device const uchar* weights,
    device const float* x,
    device float* out,
    uint row_bytes,
    uint n_blocks_per_row,
    uint n_rows,
    uint tgpig,
    uint tiisg,
    uint sgitg,
    threadgroup float* shmem_f32
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;

    const int nb = int(n_blocks_per_row);
    const int first_row = int(tgpig * NSG + sgitg) * nr0;

    const short ix = short(tiisg) / 16; // 0/1: block parity
    const short it = short(tiisg) % 16;
    const short ib = it / 2;            // 0..7: 32-elem sub-block
    const short il = it % 2;            // 0/1: 8-byte half of qs sub-block

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[nr0] = {0.0f, 0.0f};

    device const float* yb = x + ix * 256 + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uchar* q8 = (thread const uchar*)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {
        device const float4* y4 = (device const float4*)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < nr0; ++row) {
            device const uchar* blk = weights
                + (size_t)(first_row + row) * row_bytes + (size_t)ibl * 136u;
            device const uint32_t* q4 =
                (device const uint32_t*)(blk + 8u + 16u * ib + 8u * il);

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = float4(shmem_f32[q8[0]], shmem_f32[q8[1]],
                         shmem_f32[q8[2]], shmem_f32[q8[3]]);
            qf2 = float4(shmem_f32[q8[4]], shmem_f32[q8[5]],
                         shmem_f32[q8[6]], shmem_f32[q8[7]]);
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = float4(shmem_f32[q8[0]], shmem_f32[q8[1]],
                         shmem_f32[q8[2]], shmem_f32[q8[3]]);
            qf2 = float4(shmem_f32[q8[4]], shmem_f32[q8[5]],
                         shmem_f32[q8[6]], shmem_f32[q8[7]]);
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const ushort scales_h = *(device const ushort*)(blk + 2);
            const int ls = int(((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf)
                | (((scales_h >> (2 * ib)) & 3) << 4)) - 32;
            sumf[row] += float(*(device const half*)blk) * float(ls)
                * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * 256;
    }

    for (short row = 0; row < nr0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < int(n_rows)) {
            out[first_row + row] = sum_all;
        }
    }
}

kernel void iq4_xs_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* shmem_f32 [[threadgroup(0)]]
) {
    iq4_xs_matvec_impl(weights, x, out, row_bytes, n_blocks_per_row, n_rows, tgpig,
                       tiisg, sgitg, shmem_f32);
}

MOE_MV_ID_ENTRIES(iq4_xs, iq4_xs_matvec_impl)
"#;

/// The iq4_xs matvec translation unit, with its MoE id entries.
/// Built once and leaked: the pipeline cache keys on a `&'static str`.
pub fn iq4_xs_matvec_src() -> &'static str {
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        Box::leak(
            format!(
                "#include <metal_stdlib>\nusing namespace metal;\n{}\n{}",
                crate::moe_mv_id::MOE_MV_ID_MSL,
                IQ4_XS_MATVEC_BODY_MSL
            )
            .into_boxed_str(),
        )
    });
    &SRC
}
