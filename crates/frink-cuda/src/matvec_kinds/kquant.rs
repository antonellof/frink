//! The 256-element K-quant matvec kernels: Q2_K, Q3_K, Q4_K, Q5_K and
//! Q6_K.
//!
//! Moved here verbatim from `gpu.rs`; see [`super::legacy`] for why.
//! Q2_K and Q3_K joined on 2026-09-09 and have no coalesced twin.
//! These are the *uncoalesced* kernels. Q4_K, Q5_K, Q6_K and Q8_0 also
//! have coalesced rewrites, in [`super::coalesced`] beside
//! `coalesced_matvec_kernel`, because choosing between them is a launch
//! decision rather than a format one.

pub const Q4_K_MATVEC_KERNEL_SRC: &str = r#"
extern "C" __device__ float frink_f16_to_f32(unsigned short bits) {
    unsigned int sign = (bits >> 15) & 0x1u;
    unsigned int exp = (bits >> 10) & 0x1Fu;
    unsigned int mant = bits & 0x3FFu;
    float scale;
    if (exp == 0) {
        scale = ldexpf((float)mant, -24);
    } else if (exp == 31) {
        scale = mant ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
    } else {
        scale = ldexpf((float)(mant | 0x400), (int)exp - 25);
    }
    return sign ? -scale : scale;
}

extern "C" __device__ void frink_q4_k_scale_min(
    int j, const unsigned char* scales, unsigned char* sc, unsigned char* m
) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

extern "C" __global__ void q4_k_matvec(
    const unsigned char* weights, // [rows * row_bytes], row_bytes = n_blocks_per_row * 144
    const float* x,               // [cols]
    float* out,                   // [rows]
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;

    __shared__ float partial[256];
    float acc = 0.0f;

    for (int blk = threadIdx.x; blk < n_blocks_per_row; blk += blockDim.x) {
        const unsigned char* block = row_ptr + blk * 144;
        unsigned short d_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d = frink_f16_to_f32(d_bits);
        float dmin = frink_f16_to_f32(dmin_bits);
        const unsigned char* scales = block + 4;
        const unsigned char* qs = block + 16;
        int x_base = blk * 256;

        int is = 0, q_off = 0, base = 0;
        #pragma unroll
        for (int oi = 0; oi < 4; oi++) {
            unsigned char sc1, m1, sc2, m2;
            frink_q4_k_scale_min(is, scales, &sc1, &m1);
            frink_q4_k_scale_min(is + 1, scales, &sc2, &m2);
            float d1 = d * (float)sc1, min1 = dmin * (float)m1;
            float d2 = d * (float)sc2, min2 = dmin * (float)m2;
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                acc += (d1 * (float)(qs[q_off + l] & 0x0F) - min1) * x[x_base + base + l];
            }
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                acc += (d2 * (float)(qs[q_off + l] >> 4) - min2) * x[x_base + base + 32 + l];
            }
            q_off += 32;
            base += 64;
            is += 2;
        }
    }

    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}
"#;

/// CUDA C source for a fused Q5_K dequant+dot kernel: same super-block/
/// scale-min structure as Q4_K, but each nibble gets a 5th bit from a
/// 32-byte `qh` buffer (mirrors `frink_quant::dot_q5_k_f32_scalar`
/// exactly: 176-byte blocks = 2-byte `d` + 2-byte `dmin` + 12 bytes
/// scales + 32 bytes `qh` + 128 bytes `qs`).
///
/// Verified: compiled by NVRTC and executed on a real GPU, matching
/// the CPU reference exactly -- see the module doc comment.
pub const Q5_K_MATVEC_KERNEL_SRC: &str = r#"
extern "C" __device__ float frink_f16_to_f32(unsigned short bits) {
    unsigned int sign = (bits >> 15) & 0x1u;
    unsigned int exp = (bits >> 10) & 0x1Fu;
    unsigned int mant = bits & 0x3FFu;
    float scale;
    if (exp == 0) {
        scale = ldexpf((float)mant, -24);
    } else if (exp == 31) {
        scale = mant ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
    } else {
        scale = ldexpf((float)(mant | 0x400), (int)exp - 25);
    }
    return sign ? -scale : scale;
}

extern "C" __device__ void frink_q4_k_scale_min(
    int j, const unsigned char* scales, unsigned char* sc, unsigned char* m
) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

extern "C" __global__ void q5_k_matvec(
    const unsigned char* weights, // [rows * row_bytes], row_bytes = n_blocks_per_row * 176
    const float* x,               // [cols]
    float* out,                   // [rows]
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;

    __shared__ float partial[256];
    float acc = 0.0f;

    for (int blk = threadIdx.x; blk < n_blocks_per_row; blk += blockDim.x) {
        const unsigned char* block = row_ptr + blk * 176;
        unsigned short d_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d = frink_f16_to_f32(d_bits);
        float dmin = frink_f16_to_f32(dmin_bits);
        const unsigned char* scales = block + 4;
        const unsigned char* qh = block + 16;
        const unsigned char* qs = block + 48;
        int x_base = blk * 256;

        int is = 0;
        unsigned char u1 = 1, u2 = 2;
        #pragma unroll
        for (int oi = 0; oi < 4; oi++) {
            unsigned char sc1, m1, sc2, m2;
            frink_q4_k_scale_min(is, scales, &sc1, &m1);
            frink_q4_k_scale_min(is + 1, scales, &sc2, &m2);
            float d1 = d * (float)sc1, min1 = dmin * (float)m1;
            float d2 = d * (float)sc2, min2 = dmin * (float)m2;
            const unsigned char* ql = qs + oi * 32;
            int xb = x_base + oi * 64;
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                int hi = (qh[l] & u1) ? 16 : 0;
                acc += (d1 * (float)((ql[l] & 0x0F) + hi) - min1) * x[xb + l];
            }
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                int hi = (qh[l] & u2) ? 16 : 0;
                acc += (d2 * (float)((ql[l] >> 4) + hi) - min2) * x[xb + 32 + l];
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}
"#;

/// CUDA C source for a fused Q6_K dequant+dot kernel: mirrors
/// `frink_quant::dot_q6_k_f32_scalar`'s exact math (210-byte blocks of
/// 256 elements: 128 bytes `ql` + 64 bytes `qh` + 16 *signed* int8
/// scale bytes + 2-byte `d`, split into two 128-element halves). The
/// 16 per-sub-block scales are signed in the GGUF Q6_K format --
/// an earlier version of this kernel (and of the scalar CPU path it
/// mirrors) read them as unsigned, which agreed with itself but not
/// with the format; both were fixed together and are covered by the
/// negative-scale golden in `frink-quant`
/// (`q6_k_signed_scale_dequant_matches_independent_python_reference`).
pub const Q6_K_MATVEC_KERNEL_SRC: &str = r#"
extern "C" __device__ float frink_f16_to_f32(unsigned short bits) {
    unsigned int sign = (bits >> 15) & 0x1u;
    unsigned int exp = (bits >> 10) & 0x1Fu;
    unsigned int mant = bits & 0x3FFu;
    float scale;
    if (exp == 0) {
        scale = ldexpf((float)mant, -24);
    } else if (exp == 31) {
        scale = mant ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
    } else {
        scale = ldexpf((float)(mant | 0x400), (int)exp - 25);
    }
    return sign ? -scale : scale;
}

extern "C" __global__ void q6_k_matvec(
    const unsigned char* weights, // [rows * row_bytes], row_bytes = n_blocks_per_row * 210
    const float* x,               // [cols]
    float* out,                   // [rows]
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;

    __shared__ float partial[256];
    float acc = 0.0f;

    for (int blk = threadIdx.x; blk < n_blocks_per_row; blk += blockDim.x) {
        const unsigned char* block = row_ptr + blk * 210;
        const unsigned char* ql_full = block;
        const unsigned char* qh_full = block + 128;
        const unsigned char* sc_full = block + 192;
        unsigned short d_bits = (unsigned short)block[208] | ((unsigned short)block[209] << 8);
        float d = frink_f16_to_f32(d_bits);
        int x_base = blk * 256;

        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const unsigned char* ql = ql_full + half * 64;
            const unsigned char* qh = qh_full + half * 32;
            const unsigned char* sc = sc_full + half * 8;
            int xh_base = x_base + half * 128;

            #pragma unroll
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32;
                int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) - 32;
                acc += d * (float)(signed char)sc[is] * (float)q1 * x[xh_base + l];
                acc += d * (float)(signed char)sc[is + 2] * (float)q2 * x[xh_base + l + 32];
                acc += d * (float)(signed char)sc[is + 4] * (float)q3 * x[xh_base + l + 64];
                acc += d * (float)(signed char)sc[is + 6] * (float)q4 * x[xh_base + l + 96];
            }
        }
    }

    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}
"#;

/// Fused Q2_K dequant+dot. 84-byte super-blocks of 256 elements:
/// 16 scale bytes FIRST, then 64 bytes of packed 2-bit quants, then
/// `half d` and `half dmin` at the END.
///
/// Each scale byte holds two 4-bit fields: the low nibble scales the
/// quant, the high nibble scales the min that is subtracted. Mirrors
/// `frink_quant::dequant_q2_k`'s loop order exactly -- `n` over the
/// two 128-element halves, `j` over the four 2-bit fields
/// (`shift = 2j`), then two 16-element halves with consecutive scale
/// indices -- because that order IS the element order, and a kernel
/// that walks it differently pairs every quant with the wrong
/// activation.
///
/// **UNVERIFIED ON HARDWARE.** No GPU has run this, and there is no
/// host harness for a matvec kernel.
pub const Q2_K_MATVEC_KERNEL_SRC: &str = r#"
extern "C" __global__ void q2_k_matvec(
    const unsigned char* weights, // [rows * row_bytes]
    const float* x,               // [cols]
    float* out,                   // [rows]
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;

    __shared__ float partial[256];
    float acc = 0.0f;

    for (int b = threadIdx.x; b < n_blocks_per_row; b += blockDim.x) {
        const unsigned char* block = row_ptr + (size_t)b * 84;
        const unsigned char* scales = block;
        const unsigned char* qs = block + 16;

        unsigned short dbits = (unsigned short)block[80] | ((unsigned short)block[81] << 8);
        unsigned short mbits = (unsigned short)block[82] | ((unsigned short)block[83] << 8);
        float dv[2];
        unsigned short src[2];
        src[0] = dbits;
        src[1] = mbits;
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            unsigned int sign = (src[k] >> 15) & 0x1u;
            unsigned int exp = (src[k] >> 10) & 0x1Fu;
            unsigned int mant = src[k] & 0x3FFu;
            float v;
            if (exp == 0) {
                v = ldexpf((float)mant, -24);
            } else if (exp == 31) {
                v = mant ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
            } else {
                v = ldexpf((float)(mant | 0x400), (int)exp - 25);
            }
            dv[k] = sign ? -v : v;
        }
        const float d = dv[0];
        const float dmin = dv[1];

        int base = b * 256;
        int idx = 0;
        int is = 0;
        for (int n = 0; n < 2; n++) {
            const unsigned char* q = qs + 32 * n;
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    unsigned char sc = scales[is++];
                    float dl = d * (float)(sc & 0x0F);
                    float ml = dmin * (float)(sc >> 4);
                    const unsigned char* qh = q + 16 * half;
                    #pragma unroll
                    for (int l = 0; l < 16; l++) {
                        acc += (dl * (float)((qh[l] >> shift) & 3) - ml) * x[base + idx];
                        idx++;
                    }
                }
                shift += 2;
            }
        }
    }

    partial[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}
"#;

/// Fused Q3_K dequant+dot. 110-byte super-blocks of 256 elements: 32
/// `hmask` bytes, 64 bytes of packed 2-bit quants, 12 scale bytes,
/// `half d`.
///
/// The third bit of each quant is a BIT PLANE in `hmask`, and it is
/// INVERTED: a set bit means bias 0, a clear bit means bias 4. `m`
/// sweeps all eight bit positions across the whole super-block rather
/// than restarting per half. Mirrors `frink_quant::dequant_q3_k`'s
/// loop order, which is the element order.
///
/// **UNVERIFIED ON HARDWARE.** No GPU has run this, and there is no
/// host harness for a matvec kernel.
pub const Q3_K_MATVEC_KERNEL_SRC: &str = r#"
__device__ __forceinline__ unsigned char frink_q3_k_scale_mv(
    const unsigned char* raw, int is
) {
    if (is < 4) {
        return (unsigned char)((raw[is] & 0xF) | (((raw[is + 8] >> 0) & 3) << 4));
    } else if (is < 8) {
        return (unsigned char)((raw[is] & 0xF) | (((raw[is + 4] >> 2) & 3) << 4));
    } else if (is < 12) {
        return (unsigned char)((raw[is - 8] >> 4) | (((raw[is] >> 4) & 3) << 4));
    } else {
        return (unsigned char)((raw[is - 8] >> 4) | (((raw[is - 4] >> 6) & 3) << 4));
    }
}

extern "C" __global__ void q3_k_matvec(
    const unsigned char* weights, // [rows * row_bytes]
    const float* x,               // [cols]
    float* out,                   // [rows]
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;

    __shared__ float partial[256];
    float acc = 0.0f;

    for (int b = threadIdx.x; b < n_blocks_per_row; b += blockDim.x) {
        const unsigned char* block = row_ptr + (size_t)b * 110;
        const unsigned char* hmask = block;
        const unsigned char* qs = block + 32;
        const unsigned char* scales = block + 96;

        unsigned short bits = (unsigned short)block[108] | ((unsigned short)block[109] << 8);
        unsigned int sign = (bits >> 15) & 0x1u;
        unsigned int exp = (bits >> 10) & 0x1Fu;
        unsigned int mant = bits & 0x3FFu;
        float d_all;
        if (exp == 0) {
            d_all = ldexpf((float)mant, -24);
        } else if (exp == 31) {
            d_all = mant ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
        } else {
            d_all = ldexpf((float)(mant | 0x400), (int)exp - 25);
        }
        if (sign) d_all = -d_all;

        int base = b * 256;
        int idx = 0;
        int is = 0;
        unsigned char m = 1;
        for (int n = 0; n < 2; n++) {
            const unsigned char* q = qs + 32 * n;
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    float dl = d_all * ((float)frink_q3_k_scale_mv(scales, is) - 32.0f);
                    is++;
                    const unsigned char* qh = q + 16 * half;
                    const unsigned char* hh = hmask + 16 * half;
                    #pragma unroll
                    for (int l = 0; l < 16; l++) {
                        int raw = (int)((qh[l] >> shift) & 3);
                        int bias = (hh[l] & m) ? 0 : 4;
                        acc += dl * (float)(raw - bias) * x[base + idx];
                        idx++;
                    }
                }
                shift += 2;
                m <<= 1;
            }
        }
    }

    partial[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}
"#;
