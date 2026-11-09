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
/// Threads per block: 8 warps.
pub const THREADS: usize = 256;

/// The NVRTC target: `mma.m16n8k32` with s8 operands needs `sm_80`.
pub const ARCH: &str = "compute_80";

/// The skeleton every kind shares. The kind supplies `FX_N_STEPS` (steps
/// of 128 values a row takes), `FX_XD` (scale floats per 32-value run),
/// `FX_LOAD_X` (the weight-tile loader, writing `x_qs` and `x_d`),
/// `FX_Y_SCALE` (reading an activation run's scale and sum) and `FX_ACC`
/// (scaling one int32 dot into the accumulator, in ggml's order).
const BODY_SRC: &str = r#"
#define FX_I 128
#define FX_J 64
#define FX_NT 256
// The weight tile's row stride in ints: 32 of data and 4 of padding,
// so a row stride that is 4 mod 8 makes `ldmatrix` conflict-free.
#define FX_XS 36
// A q8_1 block in ints: 4 of header and 32 of values.
#define FX_YS 36

__device__ __forceinline__ float frink_mmq_f16(unsigned short bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
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

extern "C" __global__ void FX_FN_NAME(
    const unsigned char* __restrict__ w,
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

    float acc[2][4][4];
#pragma unroll
    for (int mf = 0; mf < 2; mf++)
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mf][nf][e] = 0.0f;

    const int n_steps = FX_N_STEPS;
    for (int ks = 0; ks < n_steps; ks++) {
        FX_LOAD_X
        // The step's activation blocks for this tile's tokens are
        // contiguous (K-block-major): one copy of 64 x 144 bytes.
        const uint4* ysrc = (const uint4*)(y + ((size_t)ks * ncols_pad + c0) * 144);
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
            for (int nf = 0; nf < 4; nf++) {
                const int col = wn * 32 + nf * 8 + g;
                unsigned int b[2];
                b[0] = (unsigned int)y_s[col * FX_YS + 4 + kk * 8 + t];
                b[1] = (unsigned int)y_s[col * FX_YS + 4 + kk * 8 + 4 + t];
                const int ca = wn * 32 + nf * 8 + 2 * t;
                float dy0, sy0, dy1, sy1;
                FX_Y_SCALE(ca, dy0, sy0);
                FX_Y_SCALE(ca + 1, dy1, sy1);
#pragma unroll
                for (int mf = 0; mf < 2; mf++) {
                    int c[4] = {0, 0, 0, 0};
                    frink_mma_s8(c, a[mf], b);
                    const int ra = wm * 32 + mf * 16 + g;
                    FX_ACC(acc[mf][nf][0], c[0], ra, dy0, sy0);
                    FX_ACC(acc[mf][nf][1], c[1], ra, dy1, sy1);
                    FX_ACC(acc[mf][nf][2], c[2], ra + 8, dy0, sy0);
                    FX_ACC(acc[mf][nf][3], c[3], ra + 8, dy1, sy1);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mf = 0; mf < 2; mf++) {
        const int row0 = r0 + wm * 32 + mf * 16 + g;
        const int row1 = row0 + 8;
#pragma unroll
        for (int nf = 0; nf < 4; nf++) {
            const int col0 = c0 + wn * 32 + nf * 8 + 2 * t;
            const int col1 = col0 + 1;
            if (col0 < batch) {
                if (row0 < n_rows) dst[(size_t)col0 * n_rows + row0] = acc[mf][nf][0];
                if (row1 < n_rows) dst[(size_t)col0 * n_rows + row1] = acc[mf][nf][2];
            }
            if (col1 < batch) {
                if (row0 < n_rows) dst[(size_t)col1 * n_rows + row0] = acc[mf][nf][1];
                if (row1 < n_rows) dst[(size_t)col1 * n_rows + row1] = acc[mf][nf][3];
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
#define FX_ACC(a, c, row, d, s) a += (float)(c) * x_d[(row) * 4 + kk] * (d);
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
#define FX_ACC(a, c, row, d, s) \
    { \
        a += x_d[((row) * 4 + kk) * 2] * (d) * (float)(c); \
        a += x_d[((row) * 4 + kk) * 2 + 1] * (s); \
    }
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

/// The kinds the int8 GEMM has a weight loader for, named as GGUF names
/// them.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmqKind {
    Q8_0,
    Q4_K,
}

impl MmqKind {
    /// The kind for a GGUF quant name, `None` for one without a loader.
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "Q8_0" => Some(Self::Q8_0),
            "Q4_K" => Some(Self::Q4_K),
            _ => None,
        }
    }

    /// `(values, bytes)` of one weight block.
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::Q8_0 => (32, 34),
            Self::Q4_K => (256, 144),
        }
    }

    /// The activations' header layout this kind's dot reads.
    pub fn layout(self) -> super::quantize::Layout {
        match self {
            Self::Q8_0 => super::quantize::Layout::D4,
            Self::Q4_K => super::quantize::Layout::Ds4,
        }
    }

    /// The translation unit and its NVRTC module and function names.
    pub fn source(self) -> (String, &'static str, &'static str) {
        let (module, name, load) = match self {
            Self::Q8_0 => ("frink_mmq_q8_0", "frink_mmq_q8_0", Q8_0_LOAD_X),
            Self::Q4_K => ("frink_mmq_q4_k", "frink_mmq_q4_k", Q4_K_LOAD_X),
        };
        (
            format!("#define FX_FN_NAME {name}\n{load}{BODY_SRC}"),
            module,
            name,
        )
    }
}

/// `(grid, block)`: token tiles along x, row tiles along y.
pub fn geometry(n_rows: usize, ncols_pad: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    (
        ((ncols_pad / COLS) as u32, n_rows.div_ceil(ROWS) as u32, 1),
        (THREADS as u32, 1, 1),
    )
}

/// The activation columns padded to whole token tiles.
pub fn ncols_pad(batch: usize) -> usize {
    batch.div_ceil(COLS) * COLS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tile constants the Rust side launches with are the ones the
    /// kernel text spells: a mismatch indexes shared memory wrongly and
    /// returns numbers rather than failing.
    #[test]
    fn the_kernel_text_spells_the_launch_s_tile() {
        let (src, module, name) = MmqKind::Q8_0.source();
        assert!(src.contains(&format!("#define FX_I {ROWS}\n")));
        assert!(src.contains(&format!("#define FX_J {COLS}\n")));
        assert!(src.contains(&format!("#define FX_NT {THREADS}\n")));
        assert!(src.contains(&format!("#define FX_FN_NAME {name}\n")));
        assert!(src.contains("void FX_FN_NAME("));
        assert_eq!(module, "frink_mmq_q8_0");
        // 8 warps of 32 x 32 cover the block's 128 x 64.
        assert_eq!((THREADS / 32) * 32 * 32, ROWS * COLS);
        assert_eq!(ncols_pad(1), 64);
        assert_eq!(ncols_pad(130), 192);
        assert_eq!(geometry(4096, 512), ((8, 32, 1), (256, 1, 1)));
        assert_eq!(MmqKind::by_name("Q8_0"), Some(MmqKind::Q8_0));
        assert_eq!(MmqKind::by_name("Q4_K"), Some(MmqKind::Q4_K));
        assert_eq!(MmqKind::by_name("Q6_K"), None);
        for k in [MmqKind::Q8_0, MmqKind::Q4_K] {
            let (src, _, _) = k.source();
            for m in ["FX_N_STEPS", "FX_XD", "FX_LOAD_X", "FX_Y_SCALE", "FX_ACC"] {
                assert!(src.contains(&format!("#define {m}")), "{k:?} lacks {m}");
            }
        }
    }
}
