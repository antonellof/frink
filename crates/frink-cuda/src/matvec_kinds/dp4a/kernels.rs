//! The dp4a matvecs' CUDA C text: what the quantizer and the kernels
//! share, the quantizer, the row loop, and one trait struct and entry
//! point per kind. Moved out of `dp4a.rs` (ADR 0008), which keeps the
//! table that names each of these, the geometry and the routing rule.

/// What the quantizer and every dp4a kernel share: the `block_q8_1`
/// layout, the integer dot, the f16 conversions and the word loads.
///
/// Each device-only instruction has a portable twin under `#else`,
/// which is what `tools/matvec_host_check` executes and what `hiprtc`
/// compiles: `dp4a.s32.s32` and four signed byte products; `cvt` to
/// and from f16 and the bit-level conversions (exact one way,
/// round-to-nearest-even the other, which is `__float2half`'s rule);
/// `prmt` with its sign-replicate bits cleared and an eight-byte
/// select. The hardware test is what holds each pair to agreeing.
pub const Q8_1_COMMON: &str = r#"
#define FQ_Q8_1_BYTES 36

__device__ __forceinline__ int fq_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    int r;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
#else
    return c + (int)(signed char)a * (int)(signed char)b
             + (int)(signed char)(a >> 8) * (int)(signed char)(b >> 8)
             + (int)(signed char)(a >> 16) * (int)(signed char)(b >> 16)
             + (int)(signed char)(a >> 24) * (int)(signed char)(b >> 24);
#endif
}

// The f16 at a 2-aligned address, widened (exactly).
__device__ __forceinline__ float fq_f16(const unsigned char* p) {
    const unsigned short h = *(const unsigned short*)p;
#if defined(__CUDA_ARCH__)
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
#else
    return fx_f16_to_f32(h);
#endif
}

// f32 to f16, round to nearest even: `__float2half`, so `make_half2`.
__device__ __forceinline__ unsigned short fq_f32_to_f16(float f) {
#if defined(__CUDA_ARCH__)
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
#else
    union { float f; unsigned int u; } c;
    c.f = f;
    const unsigned int sign = (c.u >> 16) & 0x8000u;
    const unsigned int ax = c.u & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) return (unsigned short)(sign | 0x7C00u | (ax > 0x7F800000u ? 0x0200u : 0u));
    if (ax >= 0x477FF000u) return (unsigned short)(sign | 0x7C00u);
    if (ax < 0x38800000u) {
        if (ax < 0x33000000u) return (unsigned short)sign;
        const unsigned int m = (ax & 0x7FFFFFu) | 0x800000u;
        const unsigned int shift = 126u - (ax >> 23);
        unsigned int k = m >> shift;
        const unsigned int rem = m & ((1u << shift) - 1u);
        const unsigned int half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (k & 1u))) k += 1u;
        return (unsigned short)(sign | k);
    }
    const unsigned int r = ax - 0x38000000u;
    return (unsigned short)(sign | ((r + 0x0FFFu + ((r >> 13) & 1u)) >> 13));
#endif
}

// 32 bits from a 1-, 2- or 4-aligned address (llama.cpp's get_int_b1/b2/b4).
__device__ __forceinline__ int fq_int_b1(const unsigned char* p) {
    return (int)((unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
                 | ((unsigned int)p[3] << 24));
}

__device__ __forceinline__ int fq_int_b2(const unsigned char* p) {
    const unsigned short* h = (const unsigned short*)p;
    return (int)((unsigned int)h[0] | ((unsigned int)h[1] << 16));
}

__device__ __forceinline__ int fq_int_b4(const unsigned char* p) {
    return *(const int*)p;
}

// Byte `s & 7` of the eight bytes {y:x} for each of the selector's four
// low nibbles: `__byte_perm`, which ignores each nibble's top bit.
__device__ __forceinline__ unsigned int fq_perm(unsigned int x, unsigned int y, unsigned int s) {
    s &= 0x7777u;
#if defined(__CUDA_ARCH__)
    unsigned int r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(s));
    return r;
#else
    const unsigned long long v = ((unsigned long long)y << 32) | (unsigned long long)x;
    unsigned int r = 0;
    for (int n = 0; n < 4; ++n) {
        r |= (unsigned int)((v >> (8u * ((s >> (4 * n)) & 7u))) & 0xFFu) << (8 * n);
    }
    return r;
#endif
}
"#;

/// The activation quantizer, `quantize_q8_1` (`quantize.cu:54-99`) for
/// one contiguous vector: one thread per value, a warp per block. The
/// warp's `amax` and f32 `sum` come from the xor butterfly
/// (`warp_reduce_max`, `warp_reduce_sum`), `d = amax / 127`, `q =
/// roundf(x / d)`, and `amax == 0` gives `d = 0` and every `q = 0` (the
/// division is skipped, as upstream skips it). Lane 0 stores `ds` as
/// two f16s: `make_half2(d, sum)`.
pub const QUANTIZE_Q8_1_KERNEL_SRC: &str = r#"
extern "C" __global__ void quantize_q8_1(
    const float* __restrict__ x,
    unsigned char* __restrict__ y,
    int n
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    // `n` is a multiple of 32 (the launcher's contract), so a warp is one
    // block_q8_1 and leaves whole: no lane waits on a partner that left.
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi);
    float sum = xi;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : (int)roundf(xi / d);
    unsigned char* b = y + (size_t)(i / 32) * FQ_Q8_1_BYTES;
    b[4 + i % 32] = (unsigned char)(signed char)q;
    if (i % 32 == 0) {
        *(unsigned int*)b = (unsigned int)fq_f32_to_f16(d)
                            | ((unsigned int)fq_f32_to_f16(sum) << 16);
    }
}
"#;

/// The row loop and the reduction every dp4a kernel shares, after
/// [`Q8_1_COMMON`]. A kind is a trait struct (`block_bytes`,
/// `q8_per_block`, `qi`, `vdr`, `dot`) handed to `fq_matvec`.
pub const DP4A_PRELUDE: &str = r#"
#ifndef FQ_NWARPS
#error "FQ_NWARPS (warps per threadblock) must be defined by the launcher"
#endif
#if FQ_NWARPS != 1 && FQ_NWARPS != 2 && FQ_NWARPS != 4
#error "FQ_NWARPS must be 1, 2 or 4"
#endif
#ifndef FQ_ROWS
#error "FQ_ROWS (rows per threadblock) must be defined by the launcher"
#endif
#if FQ_ROWS != 1 && FQ_ROWS != FQ_NWARPS
#error "FQ_ROWS must be 1 or FQ_NWARPS"
#endif
#ifndef FX_MULTI
#define FX_MULTI 0
#endif
#define FQ_THREADS (FQ_NWARPS * 32)

#if FX_MULTI
// One routed expert per blockIdx.y: its weights from a table of device
// pointers, its q8_1 input `y_stride` blocks along, its own `rows` out.
#define FQ_SLOT_SETUP \
    const unsigned char* weights = \
        (const unsigned char*)((const unsigned long long*)wsrc)[blockIdx.y]; \
    const unsigned char* y = y_in + (size_t)blockIdx.y * (size_t)y_stride * FQ_Q8_1_BYTES; \
    float* out = out_in + (size_t)blockIdx.y * (size_t)rows;
#else
#define FQ_SLOT_SETUP \
    const unsigned char* weights = (const unsigned char*)wsrc; \
    const unsigned char* y = y_in; \
    float* out = out_in;
#endif

// mul_mat_vec_q's reduction: warps 1.. hand their partial sums to warp 0
// through shared memory, warp 0 adds them in warp order, reduces across
// its lanes with the xor butterfly (every lane ends with the same sum),
// and lane `i` stores row `row0 + i`. Every thread of the block calls it.
__device__ __forceinline__ void fq_store_rows(const float* tmp, float* out, int row0, int rows) {
    __shared__ float fq_red[FQ_NWARPS > 1 ? FQ_NWARPS - 1 : 1][FQ_ROWS][32];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp > 0) {
        #pragma unroll
        for (int i = 0; i < FQ_ROWS; ++i) fq_red[warp - 1][i][lane] = tmp[i];
    }
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int i = 0; i < FQ_ROWS; ++i) {
        float v = tmp[i];
        #pragma unroll
        for (int l = 0; l < FQ_NWARPS - 1; ++l) v += fq_red[l][i][lane];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == i && row0 + i < rows) out[row0 + i] = v;
    }
}

// mul_mat_vec_q for one column: thread `tid` takes weight blocks
// `tid / (qi/vdr)`, stepping FQ_THREADS / (qi/vdr), at quant word
// `vdr * (tid % (qi/vdr))`, for each of the threadblock's FQ_ROWS rows.
// A row past the matrix reads the last row and stores nothing.
template <class K>
__device__ __forceinline__ void fq_matvec(
    const unsigned char* weights, const unsigned char* y, float* out,
    int rows, int row_bytes, int n_blocks_per_row
) {
    const int tpb = K::qi / K::vdr;
    const int tid = threadIdx.x;
    const int row0 = (int)blockIdx.x * FQ_ROWS;
    const int kqs = K::vdr * (tid % tpb);
    float tmp[FQ_ROWS];
    #pragma unroll
    for (int i = 0; i < FQ_ROWS; ++i) tmp[i] = 0.0f;
    for (int kbx = tid / tpb; kbx < n_blocks_per_row; kbx += FQ_THREADS / tpb) {
        const unsigned char* yb = y + (size_t)kbx * K::q8_per_block * FQ_Q8_1_BYTES;
        #pragma unroll
        for (int i = 0; i < FQ_ROWS; ++i) {
            const int r = row0 + i < rows ? row0 + i : rows - 1;
            tmp[i] += K::dot(weights + (size_t)r * row_bytes + (size_t)kbx * K::block_bytes, yb, kqs);
        }
    }
    fq_store_rows(tmp, out, row0, rows);
}
"#;

/// Q4_K against q8_1: `vec_dot_q4_K_q8_1` and its `_impl_vmmq`. Thread
/// word `iqs` (even, 0..30) owns `qs` bytes `32*g + 4*r ..+4` and
/// `+16 ..+4`, `g = iqs/8`, `r = (iqs/2) % 4`: low nibbles are sub-block
/// `2g`'s elements, high nibbles `2g + 1`'s, against q8_1 blocks `2g`
/// and `2g + 1`. The min term multiplies `m` by the sum of the INTEGER
/// q8 values (a dp4a against ones), as upstream's mmvq does, not by the
/// stored f32 sum. The scales are unpacked branch-free from three u16
/// words, as upstream does it; the twin unpacks them with
/// `q4_k_scale_min`, so the host check holds the two together.
pub const Q4_K_DP4A_SRC: &str = r#"
struct fq_q4_k {
    static const int block_bytes = 144;
    static const int q8_per_block = 8;
    static const int qi = 32;
    static const int vdr = 2;

    static __device__ __forceinline__ float dot(const unsigned char* blk, const unsigned char* y, int iqs) {
        const int bq8_offset = 2 * ((iqs / 2) / 4);
        const int r = (iqs / 2) % 4;
        const unsigned char* q4 = blk + 16 + 16 * bq8_offset + 4 * r;
        const int v0 = fq_int_b4(q4);
        const int v1 = fq_int_b4(q4 + 16);

        // Sub-blocks 2j and 2j + 1: their 6-bit scales and mins.
        const unsigned short* s16 = (const unsigned short*)(blk + 4);
        const int j = bq8_offset / 2;
        const int jm = j & 1;
        const unsigned int s0 = s16[jm + 0];
        const unsigned int s2 = s16[jm + 2];
        const unsigned int s4 = s16[jm + 4];
        const unsigned int hi = (unsigned int)-(int)(j >= 2);
        const unsigned int a0 = ((s0 & 0x3f3fu) & ~hi) | ((((s4 >> 0) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2)) & hi);
        const unsigned int a1 = ((s2 & 0x3f3fu) & ~hi) | ((((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2)) & hi);
        const int sc[2] = { (int)(a0 & 0xFFu), (int)((a0 >> 8) & 0xFFu) };
        const int m[2] = { (int)(a1 & 0xFFu), (int)((a1 >> 8) & 0xFFu) };

        float sumf_d = 0.0f;
        float sumf_m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned char* b8 = y + (bq8_offset + i) * FQ_Q8_1_BYTES;
            const float d8 = fq_f16(b8);
            const int u0 = fq_int_b4(b8 + 4 + 4 * r);
            const int u1 = fq_int_b4(b8 + 4 + 16 + 4 * r);
            const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
            const int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
            const int dot1 = fq_dp4a(v1i, u1, fq_dp4a(v0i, u0, 0));
            const int dot2 = fq_dp4a(0x01010101, u1, fq_dp4a(0x01010101, u0, 0));
            sumf_d += d8 * (float)(dot1 * sc[i]);
            sumf_m += d8 * (float)(dot2 * m[i]);
        }
        return fq_f16(blk) * sumf_d - fq_f16(blk + 2) * sumf_m;
    }
};

extern "C" __global__ void q4_k_matvec_dp4a(
    const void* __restrict__ wsrc,
    const unsigned char* __restrict__ y_in,
    float* out_in,
    int rows,
    int row_bytes,
    int n_blocks_per_row
#if FX_MULTI
    , int y_stride
#endif
) {
    FQ_SLOT_SETUP
    fq_matvec<fq_q4_k>(weights, y, out, rows, row_bytes, n_blocks_per_row);
}
"#;

/// Q6_K against q8_1: `vec_dot_q6_K_q8_1` and its `_impl_mmvq`, vdr 2.
/// Thread word `iqs` (even, 0..30) owns `ql` bytes `4*iqs ..+8`, which
/// sit in half `iqs / 16` of the block and in its `ql[l]` or `ql[l + 32]`
/// run as `(iqs % 16) / 8` says; the matching `qh` word is shifted by
/// `2 * ((iqs % 16) / 8)` so its two bits per weight land at bits 4..5
/// of each byte. The `- 32` is a second dp4a against `0xE0` bytes.
/// Q6_K blocks are 210 bytes, so only 2-byte loads are aligned.
pub const Q6_K_DP4A_SRC: &str = r#"
struct fq_q6_k {
    static const int block_bytes = 210;
    static const int q8_per_block = 8;
    static const int qi = 32;
    static const int vdr = 2;

    static __device__ __forceinline__ float dot(const unsigned char* blk, const unsigned char* y, int iqs) {
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        unsigned int vl[2], vh[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            vl[j] = (unsigned int)fq_int_b2(blk + 4 * (iqs + j));
            vh[j] = (unsigned int)fq_int_b2(blk + 128 + 4 * (8 * (iqs / 16) + iqs % 8 + j)) >> vh_shift;
        }
        const signed char* scales = (const signed char*)(blk + 192 + scale_offset);
        float sumf = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned char* b8 = y + (bq8_offset + 2 * i) * FQ_Q8_1_BYTES;
            const int sc = scales[4 * i];
            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int u = fq_int_b4(b8 + 4 + 4 * (iqs % 8 + j));
                const int vil = (int)((vl[j] >> (4 * i)) & 0x0F0F0F0Fu);
                const int vih = (int)(((vh[j] >> (4 * i)) << 4) & 0x30303030u);
                sumi = fq_dp4a(vil | vih, u, sumi);
                sumi = fq_dp4a((int)0xE0E0E0E0u, u, sumi);
            }
            sumf += fq_f16(b8) * (float)(sumi * sc);
        }
        return fq_f16(blk + 208) * sumf;
    }
};

extern "C" __global__ void q6_k_matvec_dp4a(
    const void* __restrict__ wsrc,
    const unsigned char* __restrict__ y_in,
    float* out_in,
    int rows,
    int row_bytes,
    int n_blocks_per_row
#if FX_MULTI
    , int y_stride
#endif
) {
    FQ_SLOT_SETUP
    fq_matvec<fq_q6_k>(weights, y, out, rows, row_bytes, n_blocks_per_row);
}
"#;

/// Q8_0 against q8_1: `vec_dot_q8_0_q8_1`, vdr 2. Thread word `iqs`
/// (0, 2, 4, 6) owns quants `4*iqs ..+8`; the block is 34 bytes, so
/// they are read as `u16` pairs. Result `(d_w * d_x) * sumi`, in that
/// order.
pub const Q8_0_DP4A_SRC: &str = r#"
struct fq_q8_0 {
    static const int block_bytes = 34;
    static const int q8_per_block = 1;
    static const int qi = 8;
    static const int vdr = 2;

    static __device__ __forceinline__ float dot(const unsigned char* blk, const unsigned char* y, int iqs) {
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = fq_dp4a(fq_int_b2(blk + 2 + 4 * (iqs + i)), fq_int_b4(y + 4 + 4 * (iqs + i)), sumi);
        }
        return fq_f16(blk) * fq_f16(y) * (float)sumi;
    }
};

extern "C" __global__ void q8_0_matvec_dp4a(
    const void* __restrict__ wsrc,
    const unsigned char* __restrict__ y_in,
    float* out_in,
    int rows,
    int row_bytes,
    int n_blocks_per_row
#if FX_MULTI
    , int y_stride
#endif
) {
    FQ_SLOT_SETUP
    fq_matvec<fq_q8_0>(weights, y, out, rows, row_bytes, n_blocks_per_row);
}
"#;

/// MXFP4 against q8_1: `vec_dot_mxfp4_q8_1`, vdr 2, two threads per
/// 17-byte block. Thread word `iqs` (0 or 2) owns code bytes `4*iqs
/// ..+8`; each word's eight E2M1 codes become ggml's `kvalues_mxfp4`
/// (E2M1 doubled, so int8) through `get_int_from_table_16`'s byte
/// permutes, the table held in four immediates. Low nibbles pair with
/// q8 values `4*(iqs+l)..`, high ones with the same plus 16. Scale
/// `(e8m0 * 0.5) * d_x`, upstream's order.
pub const MXFP4_DP4A_SRC: &str = r#"
// kvalues_mxfp4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}.
__device__ __forceinline__ void fq_mxfp4_lookup(int q4, int* lo, int* hi) {
    const unsigned int t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const unsigned int q = (unsigned int)q4;
    const unsigned int sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    unsigned int tmp[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const unsigned int shift = 16u * (unsigned int)i;
        const unsigned int low = fq_perm(t0, t1, q >> shift);
        const unsigned int high = fq_perm(t2, t3, q >> shift);
        tmp[i] = fq_perm(low, high, sel >> shift);
    }
    *lo = (int)fq_perm(tmp[0], tmp[1], 0x6420u);
    *hi = (int)fq_perm(tmp[0], tmp[1], 0x7531u);
}

struct fq_mxfp4 {
    static const int block_bytes = 17;
    static const int q8_per_block = 1;
    static const int qi = 4;
    static const int vdr = 2;

    static __device__ __forceinline__ float dot(const unsigned char* blk, const unsigned char* y, int iqs) {
        int sumi = 0;
        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            int lo, hi;
            fq_mxfp4_lookup(fq_int_b1(blk + 1 + 4 * (iqs + l)), &lo, &hi);
            sumi = fq_dp4a(lo, fq_int_b4(y + 4 + 4 * (iqs + l)), sumi);
            sumi = fq_dp4a(hi, fq_int_b4(y + 4 + 4 * (iqs + l + 4)), sumi);
        }
        // ggml_cuda_e8m0_to_fp32: the byte IS an f32 exponent field, 2^-127 for 0.
        const unsigned int e = blk[0];
        const float e8m0 = e == 0u ? __int_as_float(0x00400000) : __int_as_float((int)(e << 23));
        const float d = e8m0 * 0.5f * fq_f16(y);
        return d * (float)sumi;
    }
};

extern "C" __global__ void mxfp4_matvec_dp4a(
    const void* __restrict__ wsrc,
    const unsigned char* __restrict__ y_in,
    float* out_in,
    int rows,
    int row_bytes,
    int n_blocks_per_row
#if FX_MULTI
    , int y_stride
#endif
) {
    FQ_SLOT_SETUP
    fq_matvec<fq_mxfp4>(weights, y, out, rows, row_bytes, n_blocks_per_row);
}
"#;
