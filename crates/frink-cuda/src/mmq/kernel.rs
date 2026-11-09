//! The int8 tensor-core GEMM body (llama.cpp's MMQ, `mmq.cuh` at the
//! pin, simplified): weight rows staged in shared memory as int8 with
//! their scales, the activations as the q8_1 blocks
//! [`super::quantize`] wrote, and `mma.sync.m16n8k32.s8` giving each
//! 32-value run's exact int32 dot, which is then scaled into an f32
//! accumulator. One block is 128 weight rows by 64 tokens, 8 warps of
//! 32 x 32; a step of K is 128 values.

/// Weight rows per block.
pub const ROWS: usize = 128;
/// Tokens per block.
pub const COLS: usize = 64;

/// The wide tile's tokens: llama.cpp's MMQ takes 128 at `pp512` on
/// Ampere and Ada, reading each weight tile once per 128 tokens.
pub const WIDE_COLS: usize = 128;

/// `FRINK_CUDA_MMQ_COLS=64` or `128` holds every dense int8 GEMM at that
/// token tile, an A/B lever over [`dense_cols_for`]'s choice. Read once.
fn cols_override() -> Option<usize> {
    static COLS_ENV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *COLS_ENV.get_or_init(|| match std::env::var("FRINK_CUDA_MMQ_COLS").as_deref() {
        Ok("64") => Some(COLS),
        Ok("128") => Some(WIDE_COLS),
        _ => None,
    })
}

/// The token tile a dense int8 GEMM of `n_rows` over `batch` tokens takes
/// on a device of `n_sm` multiprocessors. Q8_0 takes [`WIDE_COLS`] when
/// its wide tiles still fill every multiprocessor once, halving the
/// weight reads (on an A10G its 4096 x 14336 x 512 GEMM went from 1.73
/// to 1.24 ms, on an A100 from 1.34 to 1.25), and [`COLS`] when they do
/// not (its 1024-row projections, slower wide on both). Every other
/// kind keeps [`COLS`]: Q4_K measured slower wide on both cards.
pub fn dense_cols_for(kind: MmqKind, n_rows: usize, batch: usize, n_sm: usize) -> usize {
    if let Some(cols) = cols_override() {
        return cols;
    }
    let wide_tiles = n_rows.div_ceil(ROWS) * batch.div_ceil(WIDE_COLS);
    if kind == MmqKind::Q8_0 && wide_tiles >= n_sm.max(1) {
        WIDE_COLS
    } else {
        COLS
    }
}

/// Whether a device of compute capability `cc` converts the GEMM's mma
/// results with `frink_mmq_i2f`'s integer add (`FX_FAST_I2F`) rather
/// than `cvt`. Only sm_80: on an A100 it took the Q4_K 14336 x 4096 x
/// 512 GEMM from 1.16 to 0.84 ms, and on an A10G (sm_86, whose integer
/// lanes are half its f32 lanes) it made the same GEMM 15% slower.
pub fn fast_i2f_for(cc: (i32, i32)) -> bool {
    cc == (8, 0)
}

/// `src` with the conversion `fast_i2f_for` chose.
pub fn with_i2f(src: String, fast: bool) -> String {
    if fast {
        format!("#define FX_FAST_I2F 1\n{src}")
    } else {
        src
    }
}
/// Threads per block: 8 warps.
pub const THREADS: usize = 256;

/// The NVRTC target: `mma.m16n8k32` with s8 operands needs `sm_80`.
pub const ARCH: &str = "compute_80";

/// Helpers every kind's snippet may call, ahead of the snippet.
const PRELUDE_SRC: &str = r#"
__device__ __forceinline__ float frink_mmq_f16(unsigned short bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}
// (float)c. Under FX_FAST_I2F, for |c| <= 2^22 and exactly: c added to
// the bits of 1.5 * 2^23, whose ulp is 1, then 1.5 * 2^23 taken away.
// An integer add and an f32 add in place of cvt.rn.f32.s32, which runs
// at a quarter of the f32 rate and converts every mma result. Every
// kind's run dot is far inside the range (`I2F_LIMIT`). Faster on sm_80
// only (`fast_i2f_for`).
#ifndef FX_FAST_I2F
#define FX_FAST_I2F 0
#endif
__device__ __forceinline__ float frink_mmq_i2f(int c) {
#if FX_FAST_I2F
    return __int_as_float(c + 0x4B400000) - 12582912.0f;
#else
    return (float)c;
#endif
}
"#;

/// The largest `|c|` [`PRELUDE_SRC`]'s `frink_mmq_i2f` converts exactly.
pub const I2F_LIMIT: i64 = 1 << 22;

/// `frink_mmq_i2f`, on the host.
#[cfg(test)]
pub(crate) fn i2f_twin(c: i32) -> f32 {
    f32::from_bits(c.wrapping_add(0x4B40_0000) as u32) - 12_582_912.0
}

/// The skeleton every kind shares. The kind supplies `FX_N_STEPS` (steps
/// of 128 values a row takes), `FX_XD` (scale floats per 32-value run),
/// `FX_LOAD_X` (the weight-tile loader, writing `x_qs`, `x_d` and the
/// per-row `x_e`), `FX_Y_SCALE` (an activation run's scale and sum),
/// `FX_DOT` (the run's int32 dot or dots), `FX_ACC` (folding them into
/// the accumulator `a`, or the step's `tmp`, in ggml's order) and
/// `FX_STEP_END` (folding a step's `tmp` into the accumulator).
const BODY_SRC: &str = r#"
#define FX_I 128
#ifndef FX_J
#define FX_J 64
#endif
// Each warp: 32 rows by FX_J / 2 tokens, FX_NF fragments of 8 tokens.
#define FX_NF (FX_J / 16)
#define FX_NT 256
// The weight tile's row stride in ints: 32 of data and 4 of padding,
// so a row stride that is 4 mod 8 makes `ldmatrix` conflict-free.
#define FX_XS 36
// A q8_1 block in ints: 4 of header and 32 of values.
#define FX_YS 36

__device__ __forceinline__ void frink_mma_s8_k16(int* c, const unsigned int* a, unsigned int b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

__device__ __forceinline__ void frink_mma_s8(int* c, const unsigned int* a, const unsigned int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Four 8x8 b16 matrices out of shared memory: the A fragment of
// `mma.m16n8k32.s8` for a 16-row by 32-byte slice of the weight tile.
__device__ __forceinline__ void frink_mmq_ldsm_x4(unsigned int* r, const int* p) {
    unsigned int a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

#ifndef FX_GROUPED
#define FX_GROUPED 0
#endif

extern "C" __global__ void __launch_bounds__(FX_NT) FX_FN_NAME(
#if FX_GROUPED
    // One routed expert per blockIdx.z: its weights through `wtab`, its
    // pairs the activation columns `group_off[z] .. group_off[z + 1]`
    // (the quantizer wrote them in grouped order) and the output rows of
    // the same indices.
    const unsigned long long* __restrict__ wtab,
    const int* __restrict__ group_off,
#else
    const unsigned char* __restrict__ w,
#endif
    const unsigned char* __restrict__ y,
    float* __restrict__ dst,
    int n_rows,
    int n_blocks,
    int batch,
    int ncols_pad,
    int row_bytes
) {
    __shared__ __align__(16) int x_qs[FX_I * FX_XS];
    __shared__ float x_d[FX_I * 4 * FX_XD];
    __shared__ float x_e[FX_I];
    __shared__ __align__(16) int y_s[FX_J * FX_YS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp & 3;
    const int wn = warp >> 2;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int r0 = blockIdx.y * FX_I;
    const int c0 = blockIdx.x * FX_J;
#if FX_GROUPED
    const int y_col0 = group_off[blockIdx.z];
    // A whole block past its group's pairs has nothing to do; the test is
    // the block's, so no barrier below is reached by part of it.
    if (c0 >= group_off[blockIdx.z + 1] - y_col0) return;
    const unsigned char* w = (const unsigned char*)wtab[blockIdx.z];
    batch = group_off[blockIdx.z + 1] - y_col0;
#else
    const int y_col0 = 0;
#endif

    float acc[2][FX_NF][4];
#pragma unroll
    for (int mf = 0; mf < 2; mf++)
#pragma unroll
        for (int nf = 0; nf < FX_NF; nf++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mf][nf][e] = 0.0f;

    const int n_steps = FX_N_STEPS;
    for (int ks = 0; ks < n_steps; ks++) {
        float tmp[2][FX_NF][4];
#pragma unroll
        for (int mf = 0; mf < 2; mf++)
#pragma unroll
            for (int nf = 0; nf < FX_NF; nf++)
#pragma unroll
                for (int e = 0; e < 4; e++) tmp[mf][nf][e] = 0.0f;
        FX_LOAD_X
        // The step's activation blocks for this tile's tokens are
        // contiguous (K-block-major): one copy of 64 x 144 bytes.
        const uint4* ysrc = (const uint4*)(y + ((size_t)ks * ncols_pad + y_col0 + c0) * 144);
        for (int e = tid; e < FX_J * FX_YS / 4; e += FX_NT) ((uint4*)y_s)[e] = ysrc[e];
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            unsigned int a[2][4];
#pragma unroll
            for (int mf = 0; mf < 2; mf++) {
                const int row = wm * 32 + mf * 16 + (lane & 15);
                frink_mmq_ldsm_x4(a[mf], &x_qs[row * FX_XS + kk * 8 + (lane >> 4) * 4]);
            }
#pragma unroll
            for (int nf = 0; nf < FX_NF; nf++) {
                const int col = wn * (FX_J / 2) + nf * 8 + g;
                unsigned int b[2];
                b[0] = (unsigned int)y_s[col * FX_YS + 4 + kk * 8 + t];
                b[1] = (unsigned int)y_s[col * FX_YS + 4 + kk * 8 + 4 + t];
                const int ca = wn * (FX_J / 2) + nf * 8 + 2 * t;
                float dy0, sy0, dy1, sy1;
                FX_Y_SCALE(ca, dy0, sy0);
                FX_Y_SCALE(ca + 1, dy1, sy1);
#pragma unroll
                for (int mf = 0; mf < 2; mf++) {
                    int c[4] = {0, 0, 0, 0};
                    int c2[4] = {0, 0, 0, 0};
                    FX_DOT(c, c2, a[mf], b);
                    const int ra = wm * 32 + mf * 16 + g;
                    FX_ACC(acc[mf][nf][0], tmp[mf][nf][0], c[0], c2[0], ra, dy0, sy0);
                    FX_ACC(acc[mf][nf][1], tmp[mf][nf][1], c[1], c2[1], ra, dy1, sy1);
                    FX_ACC(acc[mf][nf][2], tmp[mf][nf][2], c[2], c2[2], ra + 8, dy0, sy0);
                    FX_ACC(acc[mf][nf][3], tmp[mf][nf][3], c[3], c2[3], ra + 8, dy1, sy1);
                }
            }
        }
#pragma unroll
        for (int mf = 0; mf < 2; mf++) {
            const int ra = wm * 32 + mf * 16 + g;
#pragma unroll
            for (int nf = 0; nf < FX_NF; nf++) {
                FX_STEP_END(acc[mf][nf][0], tmp[mf][nf][0], ra);
                FX_STEP_END(acc[mf][nf][1], tmp[mf][nf][1], ra);
                FX_STEP_END(acc[mf][nf][2], tmp[mf][nf][2], ra + 8);
                FX_STEP_END(acc[mf][nf][3], tmp[mf][nf][3], ra + 8);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mf = 0; mf < 2; mf++) {
        const int row0 = r0 + wm * 32 + mf * 16 + g;
        const int row1 = row0 + 8;
#pragma unroll
        for (int nf = 0; nf < FX_NF; nf++) {
            const int col0 = c0 + wn * (FX_J / 2) + nf * 8 + 2 * t;
            const int col1 = col0 + 1;
            const size_t o0 = (size_t)(y_col0 + col0) * n_rows;
            const size_t o1 = (size_t)(y_col0 + col1) * n_rows;
            if (col0 < batch) {
                if (row0 < n_rows) dst[o0 + row0] = acc[mf][nf][0];
                if (row1 < n_rows) dst[o0 + row1] = acc[mf][nf][2];
            }
            if (col1 < batch) {
                if (row0 < n_rows) dst[o1 + row0] = acc[mf][nf][1];
                if (row1 < n_rows) dst[o1 + row1] = acc[mf][nf][3];
            }
        }
    }
}
"#;

/// Q8_0's weight tile for one step (four 34-byte blocks a row): each
/// thread one int of a row's 128 quants, read as two 2-byte loads (the
/// block is only 2-byte aligned), a warp a row; then the four scales.
/// A row past `n_rows` repeats the last row (its outputs are never
/// stored); a block past the row's end is zeros.
const Q8_0_LOAD_X: &str = r#"
#define FX_N_STEPS ((n_blocks + 3) / 4)
#define FX_XD 1
// D4 activations: one f32 scale per run; `acc += C * dw * dy`
// (`vec_dot_q8_0_q8_1_mma`).
#define FX_Y_SCALE(col, d, s) { d = ((const float*)&y_s[(col) * FX_YS])[kk]; s = 0.0f; }
#define FX_DOT(c, c2, af, b) frink_mma_s8(c, af, b);
#define FX_ACC(a, t, c, c2, row, d, s) a += frink_mmq_i2f(c) * x_d[(row) * 4 + kk] * (d);
#define FX_STEP_END(a, t, row)
#define FX_LOAD_X \
    { \
        const int kb0 = ks * 4; \
        const int ii = tid & 31; \
        const int blk = kb0 + (ii >> 3); \
        for (int pass = 0; pass < FX_I / 8; pass++) { \
            const int i = pass * 8 + (tid >> 5); \
            const int row = min(r0 + i, n_rows - 1); \
            int v = 0; \
            if (blk < n_blocks) { \
                const unsigned short* p = (const unsigned short*)(w + (size_t)row * row_bytes \
                    + (size_t)blk * 34 + 2 + 4 * (ii & 7)); \
                v = (int)((unsigned int)p[0] | ((unsigned int)p[1] << 16)); \
            } \
            x_qs[i * FX_XS + ii] = v; \
        } \
        for (int e = tid; e < FX_I * 4; e += FX_NT) { \
            const int i = e >> 2; \
            const int b = kb0 + (e & 3); \
            const int row = min(r0 + i, n_rows - 1); \
            x_d[e] = b < n_blocks \
                ? frink_mmq_f16(*(const unsigned short*)(w + (size_t)row * row_bytes + (size_t)b * 34)) \
                : 0.0f; \
        } \
    }
"#;

/// Q4_K's weight tile for one step, half a 144-byte superblock a row:
/// its 64 bytes of nibbles unpacked into the four 32-value runs they hold
/// (a byte's low nibble in run `2p`, its high nibble in run `2p + 1`),
/// and each run's `half(d * sc)` and `half(-dmin * m)`, rounded to half as
/// `load_tiles_q4_K`'s half2 product rounds them. DS4 activations; per run
/// `acc += (dm.x * dy) * C`, then `acc += dm.y * s`
/// (`vec_dot_q4_K_q8_1_mma`), `s` being the run's sum before quantizing.
const Q4_K_LOAD_X: &str = r#"
#define FX_N_STEPS (n_blocks * 2)
#define FX_XD 2
__device__ __forceinline__ float frink_mmq_round_h(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return frink_mmq_f16(h);
}
#define FX_Y_SCALE(col, d, s) \
    { \
        const unsigned int hs = (unsigned int)y_s[(col) * FX_YS + kk]; \
        d = frink_mmq_f16((unsigned short)(hs & 0xffffu)); \
        s = frink_mmq_f16((unsigned short)(hs >> 16)); \
    }
#define FX_DOT(c, c2, af, b) frink_mma_s8(c, af, b);
#define FX_ACC(a, t, c, c2, row, d, s) \
    { \
        a += x_d[((row) * 4 + kk) * 2] * (d) * frink_mmq_i2f(c); \
        a += x_d[((row) * 4 + kk) * 2 + 1] * (s); \
    }
#define FX_STEP_END(a, t, row)
#define FX_LOAD_X \
    { \
        const int sbk = ks >> 1; \
        const int h = ks & 1; \
        const int ii = tid & 15; \
        const int p = ii >> 3; \
        for (int pass = 0; pass < FX_I / 16; pass++) { \
            const int i = pass * 16 + (tid >> 4); \
            const int row = min(r0 + i, n_rows - 1); \
            const unsigned char* blk = w + (size_t)row * row_bytes + (size_t)sbk * 144; \
            const int q = *(const int*)(blk + 16 + h * 64 + 4 * ii); \
            x_qs[i * FX_XS + (2 * p) * 8 + (ii & 7)] = q & 0x0F0F0F0F; \
            x_qs[i * FX_XS + (2 * p + 1) * 8 + (ii & 7)] = (q >> 4) & 0x0F0F0F0F; \
        } \
        for (int e = tid; e < FX_I * 4; e += FX_NT) { \
            const int i = e >> 2; \
            const int j = h * 4 + (e & 3); \
            const int row = min(r0 + i, n_rows - 1); \
            const unsigned char* blk = w + (size_t)row * row_bytes + (size_t)sbk * 144; \
            const unsigned char* q = blk + 4; \
            int sc, m; \
            if (j < 4) { \
                sc = q[j] & 63; \
                m = q[j + 4] & 63; \
            } else { \
                sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); \
                m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); \
            } \
            const float d = frink_mmq_f16(*(const unsigned short*)blk); \
            const float dmin = frink_mmq_f16(*(const unsigned short*)(blk + 2)); \
            x_d[e * 2] = frink_mmq_round_h(d * (float)sc); \
            x_d[e * 2 + 1] = frink_mmq_round_h(-dmin * (float)m); \
        } \
    }
"#;

/// Q6_K's weight tile for one step, half a 210-byte superblock a row: each
/// of the half's 128 values rebuilt from its 4 low bits (`ql`) and 2 high
/// bits (`qh`) with the -32 offset applied at load (`load_tiles_q6_K`'s
/// `__vsub4`), its 16-value sub-scales as floats (exact small ints) and
/// the superblock's `d` per row. A 32-value run takes two k16 dots, one
/// per sub-scale; the step accumulates `tmp += (C0 * sc0 + C1 * sc1) * dy`
/// and folds `acc += tmp * d` at its end (`vec_dot_q6_K_q8_1_mma`; the
/// int products are exact in f32). The block is only 2-byte aligned.
const Q6_K_LOAD_X: &str = r#"
#define FX_N_STEPS (n_blocks * 2)
#define FX_XD 2
#define FX_Y_SCALE(col, d, s) { d = ((const float*)&y_s[(col) * FX_YS])[kk]; s = 0.0f; }
#define FX_DOT(c, c2, af, b) \
    { \
        frink_mma_s8_k16(c, af, b[0]); \
        frink_mma_s8_k16(c2, af + 2, b[1]); \
    }
#define FX_ACC(a, t, c, c2, row, d, s) \
    t += (frink_mmq_i2f(c) * x_d[((row) * 4 + kk) * 2] \
        + frink_mmq_i2f(c2) * x_d[((row) * 4 + kk) * 2 + 1]) * (d);
#define FX_STEP_END(a, t, row) a += (t) * x_e[row];
__device__ __forceinline__ unsigned int frink_mmq_u32_a2(const unsigned char* p) {
    const unsigned short* h = (const unsigned short*)p;
    return (unsigned int)h[0] | ((unsigned int)h[1] << 16);
}
#define FX_LOAD_X \
    { \
        const int sbk = ks >> 1; \
        const int h = ks & 1; \
        const int ii = tid & 31; \
        const int r = ii >> 3; \
        const int l0 = 4 * (ii & 7); \
        for (int pass = 0; pass < FX_I / 8; pass++) { \
            const int i = pass * 8 + (tid >> 5); \
            const int row = min(r0 + i, n_rows - 1); \
            const unsigned char* blk = w + (size_t)row * row_bytes + (size_t)sbk * 210; \
            const unsigned int ql = frink_mmq_u32_a2(blk + h * 64 + (r & 1) * 32 + l0); \
            const unsigned int qh = frink_mmq_u32_a2(blk + 128 + h * 32 + l0); \
            const unsigned int lo = ((r < 2) ? ql : (ql >> 4)) & 0x0F0F0F0Fu; \
            const unsigned int hi = ((qh >> (2 * r)) & 0x03030303u) << 4; \
            x_qs[i * FX_XS + ii] = (int)__vsub4(lo | hi, 0x20202020u); \
        } \
        for (int e = tid; e < FX_I * 8; e += FX_NT) { \
            const int i = e >> 3; \
            const int row = min(r0 + i, n_rows - 1); \
            const unsigned char* blk = w + (size_t)row * row_bytes + (size_t)sbk * 210; \
            x_d[e] = (float)(signed char)blk[192 + h * 8 + (e & 7)]; \
        } \
        for (int i = tid; i < FX_I; i += FX_NT) { \
            const int row = min(r0 + i, n_rows - 1); \
            x_e[i] = frink_mmq_f16(*(const unsigned short*)(w + (size_t)row * row_bytes \
                + (size_t)sbk * 210 + 208)); \
        } \
    }
"#;

/// MXFP4's weight tile for one step (four 17-byte blocks a row): each
/// int of a row holds four values of one block, the low nibbles of its
/// code bytes for the block's first 16 values and the high nibbles for
/// the next 16, each E2M1 code as ggml's doubled `kvalues_mxfp4` integer;
/// the block's scale is `e8m0_to_fp32(e) * 0.5`, halving the doubling
/// back (`load_tiles_mxfp4`). D4 activations and Q8_0's dot
/// (`vec_dot_q8_0_q8_1_mma`). The block is only byte-aligned.
const MXFP4_LOAD_X: &str = r#"
#define FX_N_STEPS ((n_blocks + 3) / 4)
#define FX_XD 1
#define FX_Y_SCALE(col, d, s) { d = ((const float*)&y_s[(col) * FX_YS])[kk]; s = 0.0f; }
#define FX_DOT(c, c2, af, b) frink_mma_s8(c, af, b);
#define FX_ACC(a, t, c, c2, row, d, s) a += frink_mmq_i2f(c) * x_d[(row) * 4 + kk] * (d);
#define FX_STEP_END(a, t, row)
// E2M1 doubled: 0, 1, 2, 3, 4, 6, 8, 12 and their negatives.
__device__ __forceinline__ int frink_mmq_e2m1x2(unsigned int v) {
    const int m = (int)(v & 7u);
    const int a = m < 4 ? m : ((2 | (m & 1)) << ((m >> 1) - 1));
    return (v & 8u) ? -a : a;
}
#define FX_LOAD_X \
    { \
        const int kb0 = ks * 4; \
        const int ii = tid & 31; \
        const int blk = kb0 + (ii >> 3); \
        const int j = ii & 7; \
        const int shift = j < 4 ? 0 : 4; \
        for (int pass = 0; pass < FX_I / 8; pass++) { \
            const int i = pass * 8 + (tid >> 5); \
            const int row = min(r0 + i, n_rows - 1); \
            int v = 0; \
            if (blk < n_blocks) { \
                const unsigned char* q = w + (size_t)row * row_bytes + (size_t)blk * 17 + 1 \
                    + 4 * (j & 3); \
                v = (frink_mmq_e2m1x2(q[0] >> shift) & 0xff) \
                    | ((frink_mmq_e2m1x2(q[1] >> shift) & 0xff) << 8) \
                    | ((frink_mmq_e2m1x2(q[2] >> shift) & 0xff) << 16) \
                    | ((frink_mmq_e2m1x2(q[3] >> shift) & 0xff) << 24); \
            } \
            x_qs[i * FX_XS + ii] = v; \
        } \
        for (int e = tid; e < FX_I * 4; e += FX_NT) { \
            const int i = e >> 2; \
            const int b = kb0 + (e & 3); \
            const int row = min(r0 + i, n_rows - 1); \
            float d = 0.0f; \
            if (b < n_blocks) { \
                const unsigned int eb = w[(size_t)row * row_bytes + (size_t)b * 17]; \
                d = (eb == 0u ? __int_as_float(0x00400000) : __int_as_float((int)(eb << 23))) \
                    * 0.5f; \
            } \
            x_d[e] = d; \
        } \
    }
"#;

/// The kinds the int8 GEMM has a weight loader for, named as GGUF names
/// them.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmqKind {
    Q8_0,
    Q4_K,
    Q6_K,
    MXFP4,
}

impl MmqKind {
    /// The largest `|c|` one `mma` result can hold for this kind: its
    /// run's length times its largest weight code times q8_1's 127.
    /// Q8_0's codes reach -128, Q4_K's nibbles 15, Q6_K's centred codes
    /// -32 over 16-value runs (`m16n8k16`), MXFP4's doubled E2M1 12.
    pub fn max_run_dot(self) -> i64 {
        let (run, code) = match self {
            Self::Q8_0 => (32, 128),
            Self::Q4_K => (32, 15),
            Self::Q6_K => (16, 32),
            Self::MXFP4 => (32, 12),
        };
        run * code * 127
    }

    /// The kind for a GGUF quant name, `None` for one without a loader.
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "Q8_0" => Some(Self::Q8_0),
            "Q4_K" => Some(Self::Q4_K),
            "Q6_K" => Some(Self::Q6_K),
            "MXFP4" => Some(Self::MXFP4),
            _ => None,
        }
    }

    /// `(values, bytes)` of one weight block.
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::Q8_0 => (32, 34),
            Self::Q4_K => (256, 144),
            Self::Q6_K => (256, 210),
            Self::MXFP4 => (32, 17),
        }
    }

    /// The activations' header layout this kind's dot reads.
    pub fn layout(self) -> super::quantize::Layout {
        match self {
            Self::Q8_0 => super::quantize::Layout::D4,
            Self::Q4_K => super::quantize::Layout::Ds4,
            Self::Q6_K => super::quantize::Layout::D4,
            Self::MXFP4 => super::quantize::Layout::D4,
        }
    }

    /// [`Self::source`] for the grouped form: a routed layer's experts in
    /// one launch, one per `blockIdx.z` (`FX_GROUPED`).
    pub fn grouped_source(self) -> (String, &'static str, &'static str) {
        let (src, _, _) = self.source();
        let (module, name) = match self {
            Self::Q8_0 => ("frink_mmq_q8_0_grouped", "frink_mmq_q8_0_grouped"),
            Self::Q4_K => ("frink_mmq_q4_k_grouped", "frink_mmq_q4_k_grouped"),
            Self::Q6_K => ("frink_mmq_q6_k_grouped", "frink_mmq_q6_k_grouped"),
            Self::MXFP4 => ("frink_mmq_mxfp4_grouped", "frink_mmq_mxfp4_grouped"),
        };
        let (_, plain) = self.names();
        (
            format!("#define FX_GROUPED 1\n{}", src.replace(plain, name)),
            module,
            name,
        )
    }

    /// The plain form's NVRTC module and function names.
    fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::Q8_0 => ("frink_mmq_q8_0", "frink_mmq_q8_0"),
            Self::Q4_K => ("frink_mmq_q4_k", "frink_mmq_q4_k"),
            Self::Q6_K => ("frink_mmq_q6_k", "frink_mmq_q6_k"),
            Self::MXFP4 => ("frink_mmq_mxfp4", "frink_mmq_mxfp4"),
        }
    }

    /// [`Self::source`] with a [`WIDE_COLS`]-token tile.
    pub fn wide_source(self) -> (String, &'static str, &'static str) {
        let (src, _, _) = self.source();
        let (module, name) = match self {
            Self::Q8_0 => ("frink_mmq_q8_0_wide", "frink_mmq_q8_0_wide"),
            Self::Q4_K => ("frink_mmq_q4_k_wide", "frink_mmq_q4_k_wide"),
            Self::Q6_K => ("frink_mmq_q6_k_wide", "frink_mmq_q6_k_wide"),
            Self::MXFP4 => ("frink_mmq_mxfp4_wide", "frink_mmq_mxfp4_wide"),
        };
        let (_, plain) = self.names();
        (
            format!("#define FX_J {WIDE_COLS}\n{}", src.replace(plain, name)),
            module,
            name,
        )
    }

    /// The translation unit and its NVRTC module and function names.
    pub fn source(self) -> (String, &'static str, &'static str) {
        let (module, name) = self.names();
        let load = match self {
            Self::Q8_0 => Q8_0_LOAD_X,
            Self::Q4_K => Q4_K_LOAD_X,
            Self::Q6_K => Q6_K_LOAD_X,
            Self::MXFP4 => MXFP4_LOAD_X,
        };
        (
            format!("#define FX_FN_NAME {name}\n{PRELUDE_SRC}{load}{BODY_SRC}"),
            module,
            name,
        )
    }
}

/// `(grid, block)`: token tiles of `cols` along x, row tiles along y.
pub fn geometry(
    n_rows: usize,
    ncols_pad: usize,
    cols: usize,
) -> ((u32, u32, u32), (u32, u32, u32)) {
    (
        ((ncols_pad / cols) as u32, n_rows.div_ceil(ROWS) as u32, 1),
        (THREADS as u32, 1, 1),
    )
}

/// The activation columns padded to whole [`COLS`]-token tiles.
pub fn ncols_pad(batch: usize) -> usize {
    ncols_pad_for(batch, COLS)
}

/// The activation columns padded to whole `cols`-token tiles.
pub fn ncols_pad_for(batch: usize, cols: usize) -> usize {
    batch.div_ceil(cols) * cols
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fast conversion is `(float)c` bit for bit over its whole range
    /// and wrong one past each end, so the range is the real one; every
    /// kind's dots fit in it; and the kernel spells the twin's constants.
    #[test]
    fn the_fast_int_to_float_is_exact_for_every_run_dot() {
        for c in -I2F_LIMIT..=I2F_LIMIT {
            let c = c as i32;
            assert_eq!(i2f_twin(c).to_bits(), (c as f32).to_bits(), "{c}");
        }
        let past = I2F_LIMIT as i32 + 1;
        assert_ne!(i2f_twin(past), past as f32);
        assert_ne!(i2f_twin(-past), -past as f32);
        for k in [MmqKind::Q8_0, MmqKind::Q4_K, MmqKind::Q6_K, MmqKind::MXFP4] {
            assert!(k.max_run_dot() <= I2F_LIMIT, "{k:?}: {}", k.max_run_dot());
            let (src, _, _) = k.source();
            assert!(!src.contains("(float)(c"), "{k:?} converts with cvt");
        }
        assert!(PRELUDE_SRC.contains("__int_as_float(c + 0x4B400000) - 12582912.0f"));
        assert!(with_i2f("x".into(), true).starts_with("#define FX_FAST_I2F 1\n"));
        assert_eq!(with_i2f("x".into(), false), "x");
        assert!(fast_i2f_for((8, 0)));
        assert!(!fast_i2f_for((8, 6)) && !fast_i2f_for((8, 9)) && !fast_i2f_for((9, 0)));
    }

    /// Q8_0 goes wide only where the wide grid fills the device once;
    /// the other kinds stay narrow. (Not under `FRINK_CUDA_MMQ_COLS`.)
    #[test]
    fn the_dense_tile_is_wide_only_for_q8_0_grids_that_fill_the_device() {
        if std::env::var_os("FRINK_CUDA_MMQ_COLS").is_some() {
            return;
        }
        // An A10G (80 SMs) and an A100 (108) at pp512, Llama-3.1-8B shapes.
        for n_sm in [80, 108] {
            assert_eq!(dense_cols_for(MmqKind::Q8_0, 4096, 512, n_sm), WIDE_COLS);
            assert_eq!(dense_cols_for(MmqKind::Q8_0, 14336, 512, n_sm), WIDE_COLS);
            assert_eq!(dense_cols_for(MmqKind::Q8_0, 1024, 512, n_sm), COLS);
            assert_eq!(dense_cols_for(MmqKind::Q4_K, 14336, 512, n_sm), COLS);
            assert_eq!(dense_cols_for(MmqKind::MXFP4, 14336, 512, n_sm), COLS);
        }
        // 96 wide tiles fill an A10G and not an A100; a short batch is narrow.
        assert_eq!(dense_cols_for(MmqKind::Q8_0, 3072, 512, 80), WIDE_COLS);
        assert_eq!(dense_cols_for(MmqKind::Q8_0, 3072, 512, 108), COLS);
        assert_eq!(dense_cols_for(MmqKind::Q8_0, 4096, 64, 80), COLS);
    }

    /// The tile constants the Rust side launches with are the ones the
    /// kernel text spells: a mismatch indexes shared memory wrongly and
    /// returns numbers rather than failing.
    #[test]
    fn the_kernel_text_spells_the_launch_s_tile() {
        let (src, module, name) = MmqKind::Q8_0.source();
        assert!(src.contains(&format!("#define FX_I {ROWS}\n")));
        assert!(src.contains(&format!("#ifndef FX_J\n#define FX_J {COLS}\n")));
        assert!(src.contains(&format!("#define FX_NT {THREADS}\n")));
        assert!(src.contains(&format!("#define FX_FN_NAME {name}\n")));
        assert!(src.contains("void __launch_bounds__(FX_NT) FX_FN_NAME("));
        assert_eq!(module, "frink_mmq_q8_0");
        // The wide form defines its width ahead of the default, under its
        // own names, so the two never share a cached module.
        let (wide, wmod, wname) = MmqKind::Q8_0.wide_source();
        assert!(wide.starts_with(&format!("#define FX_J {WIDE_COLS}\n")));
        assert!(wide.contains(&format!("#define FX_FN_NAME {wname}\n")));
        assert_ne!(wmod, module);
        // 8 warps of 32 rows by `cols / 2` tokens cover the block, in
        // fragments of 8 tokens.
        for cols in [COLS, WIDE_COLS] {
            assert_eq!((THREADS / 32) * 32 * (cols / 2), ROWS * cols);
            assert!(cols.is_multiple_of(16));
        }
        assert_eq!(ncols_pad(1), 64);
        assert_eq!(ncols_pad(130), 192);
        assert_eq!(ncols_pad_for(130, WIDE_COLS), 256);
        assert_eq!(geometry(4096, 512, COLS), ((8, 32, 1), (256, 1, 1)));
        assert_eq!(geometry(4096, 512, WIDE_COLS), ((4, 32, 1), (256, 1, 1)));
        assert_eq!(MmqKind::by_name("Q8_0"), Some(MmqKind::Q8_0));
        assert_eq!(MmqKind::by_name("Q4_K"), Some(MmqKind::Q4_K));
        assert_eq!(MmqKind::by_name("Q6_K"), Some(MmqKind::Q6_K));
        assert_eq!(MmqKind::by_name("MXFP4"), Some(MmqKind::MXFP4));
        assert_eq!(MmqKind::by_name("Q5_K"), None);
        for k in [MmqKind::Q8_0, MmqKind::Q4_K, MmqKind::Q6_K, MmqKind::MXFP4] {
            let (src, _, _) = k.source();
            for m in [
                "FX_N_STEPS",
                "FX_XD",
                "FX_LOAD_X",
                "FX_Y_SCALE",
                "FX_DOT",
                "FX_ACC",
                "FX_STEP_END",
            ] {
                assert!(src.contains(&format!("#define {m}")), "{k:?} lacks {m}");
            }
            // NVRTC compiles one pass: every helper is defined before
            // anything calls it (the Q4_K snippet once called the f16
            // helper ahead of its definition, which only a device saw).
            for helper in [
                "frink_mmq_f16",
                "frink_mmq_round_h",
                "frink_mmq_u32_a2",
                "frink_mmq_e2m1x2",
            ] {
                let Some(first) = src.find(&format!("{helper}(")) else {
                    continue;
                };
                let def = src
                    .find(&format!(" {helper}("))
                    .filter(|&i| src[..i].ends_with("float") || src[..i].ends_with("int"));
                assert_eq!(
                    def.map(|d| d + 1),
                    Some(first),
                    "{k:?}: {helper} used before defined"
                );
            }
        }
    }
}
