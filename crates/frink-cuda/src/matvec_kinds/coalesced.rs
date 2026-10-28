//! The coalesced matvec kernels: one warp per output row, the warp's
//! lanes reading one super-block (or Q8_0 block) as a contiguous run.
//!
//! Moved here from `gpu.rs` (#133). They are CUDA C text plus the table
//! that routes a kind to its coalesced kernel, so nothing here needs
//! `cudarc`; the launch plan that pairs a kernel with its geometry stays
//! in `gpu.rs`. Their scalar twins are in [`crate::coalesced_twin`].

/// CUDA C for the coalesced Q5_K matvec: one warp per row, lane `l`
/// owning element `l` of each 32-element group, so the warp's loads of
/// `qs`, `qh` and the activations are each contiguous.
///
/// Q5_K adds a fifth bit per weight over Q4_K, held in `qh` as one
/// BIT-PLANE per group: bit `2*oi` for the group's low nibbles and bit
/// `2*oi + 1` for its high ones. `q5_k_matvec_coalesced_row` in
/// `coalesced_twin.rs` is the scalar twin that pins that down.
pub const Q5_K_MATVEC_COALESCED_KERNEL_SRC: &str = r#"
__device__ __forceinline__ float frink_f16_to_f32_q5co(unsigned short bits) {
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

__device__ __forceinline__ void frink_q4_k_scale_min_q5co(
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

extern "C" __global__ void q5_k_matvec_coalesced(
    const unsigned char* weights,
    const float* x,
    float* out,
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps + warp;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;
    float acc = 0.0f;

    for (int blk = 0; blk < n_blocks_per_row; ++blk) {
        const unsigned char* block = row_ptr + (size_t)blk * 176;
        const float d = frink_f16_to_f32_q5co(
            (unsigned short)block[0] | ((unsigned short)block[1] << 8));
        const float dmin = frink_f16_to_f32_q5co(
            (unsigned short)block[2] | ((unsigned short)block[3] << 8));
        const unsigned char* scales = block + 4;
        const unsigned char* qh = block + 16;
        const unsigned char* qs = block + 48;
        const int x_base = blk * 256;
        const unsigned char h = qh[lane];

        #pragma unroll
        for (int oi = 0; oi < 4; ++oi) {
            unsigned char sc1, m1, sc2, m2;
            frink_q4_k_scale_min_q5co(2 * oi, scales, &sc1, &m1);
            frink_q4_k_scale_min_q5co(2 * oi + 1, scales, &sc2, &m2);
            const float d1 = d * (float)sc1, min1 = dmin * (float)m1;
            const float d2 = d * (float)sc2, min2 = dmin * (float)m2;
            const unsigned char ql = qs[oi * 32 + lane];
            const unsigned char u1 = (unsigned char)(1u << (2 * oi));
            const unsigned char u2 = (unsigned char)(2u << (2 * oi));
            const int xb = x_base + oi * 64;
            const int hi1 = (h & u1) ? 16 : 0;
            const int hi2 = (h & u2) ? 16 : 0;
            acc += (d1 * (float)((ql & 0x0F) + hi1) - min1) * x[xb + lane];
            acc += (d2 * (float)((ql >> 4) + hi2) - min2) * x[xb + 32 + lane];
        }
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, s);
    }
    if (lane == 0) out[row] = acc;
}
"#;

/// CUDA C for the coalesced Q8_0 matvec. A Q8_0 block holds exactly 32
/// quants, so a warp maps onto one block with no leftovers: lane `l`
/// takes byte `l` and the warp's load is 32 contiguous bytes, against
/// the 1088-byte stride the one-thread-per-block kernel used.
///
/// The quants are SIGNED; the twin has a test that fails on an
/// unsigned read.
pub const Q8_0_MATVEC_COALESCED_KERNEL_SRC: &str = r#"
__device__ __forceinline__ float frink_f16_to_f32_q8co(unsigned short bits) {
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

extern "C" __global__ void q8_0_matvec_coalesced(
    const unsigned char* weights,
    const float* x,
    float* out,
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps + warp;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;
    float acc = 0.0f;

    for (int blk = 0; blk < n_blocks_per_row; ++blk) {
        const unsigned char* block = row_ptr + (size_t)blk * 34;
        const float d = frink_f16_to_f32_q8co(
            (unsigned short)block[0] | ((unsigned short)block[1] << 8));
        const signed char q = (signed char)block[2 + lane];
        acc += d * (float)q * x[blk * 32 + lane];
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, s);
    }
    if (lane == 0) out[row] = acc;
}
"#;

/// Q6_K matvec with a coalesced access pattern.
///
/// The kernel this replaces gives each thread a whole 210-byte
/// super-block, so a warp's 32 loads land 210 bytes apart. Here the
/// warp takes one super-block and lane `l` takes the index the old
/// inner loop iterated, which makes `ql[l]`, `ql[l+32]` and `qh[l]`
/// each contiguous across the warp.
///
/// Q6_K is worth doing right after Q4_K because a `Q4_K_M` checkpoint
/// is not all Q4_K: its output tensor is usually Q6_K, so this kernel
/// runs every token too.
///
/// Same arithmetic and same f32 activations as before, so it stays
/// token-identical.
pub const Q6_K_MATVEC_COALESCED_KERNEL_SRC: &str = r#"
__device__ __forceinline__ float frink_f16_to_f32_q6co(unsigned short bits) {
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

extern "C" __global__ void q6_k_matvec_coalesced(
    const unsigned char* weights,
    const float* x,
    float* out,
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps + warp;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;
    const int is = lane / 16;
    float acc = 0.0f;

    for (int blk = 0; blk < n_blocks_per_row; ++blk) {
        const unsigned char* block = row_ptr + (size_t)blk * 210;
        const float d = frink_f16_to_f32_q6co(
            (unsigned short)block[208] | ((unsigned short)block[209] << 8));
        const int x_base = blk * 256;

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned char* ql = block + half * 64;
            const unsigned char* qh = block + 128 + half * 32;
            const signed char* sc = (const signed char*)(block + 192 + half * 8);
            const int xh = x_base + half * 128;

            const int q1 = (int)((ql[lane] & 0x0F) | ((qh[lane] & 0x03) << 4)) - 32;
            const int q2 = (int)((ql[lane + 32] & 0x0F) | (((qh[lane] >> 2) & 0x03) << 4)) - 32;
            const int q3 = (int)((ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4)) - 32;
            const int q4 = (int)((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4)) - 32;

            acc += d * (float)sc[is] * (float)q1 * x[xh + lane];
            acc += d * (float)sc[is + 2] * (float)q2 * x[xh + lane + 32];
            acc += d * (float)sc[is + 4] * (float)q3 * x[xh + lane + 64];
            acc += d * (float)sc[is + 6] * (float)q4 * x[xh + lane + 96];
        }
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, s);
    }
    if (lane == 0) out[row] = acc;
}
"#;

/// Q4_K matvec with a COALESCED access pattern.
///
/// The kernel this replaces walks whole 144-byte super-blocks per
/// thread (`blk += blockDim.x`), so adjacent lanes read addresses 144
/// bytes apart and every lane in a warp touches a different cache
/// line. Measured consequence on an RTX 3060: 19.0 GB/s of weight
/// traffic, **5.3%** of the card's 360 GB/s, where llama.cpp reaches
/// 60.4%. A decode matvec is a streaming read of the weights, so that
/// percentage IS the gap, and the inner arithmetic is irrelevant while
/// it holds (a `dp4a` port measured under 1%, #142).
///
/// Here one warp takes one super-block at a time and lane `l` reads
/// `qs[4l .. 4l+4)`. Thirty-two lanes then cover the 128 quantized
/// bytes as one contiguous run instead of 32 scattered ones.
///
/// The activation stays f32 on purpose. Quantizing it to int8 is what
/// llama.cpp does, and on a real checkpoint it diverges from the CPU
/// reference at token 4, which `frink verify` refuses. This change is
/// meant to be token-identical.
///
/// Lane to data mapping, which is the part to get right:
///   off = 4*l           byte offset into the 128 qs bytes
///   oi  = l/8           which 32-byte group, so which sub-block PAIR
///   low nibbles  -> activations at blk*256 + oi*64 + (off%32)
///   high nibbles -> the same, plus 32
/// The 32 bytes of a group carry the first 32 activations in their low
/// nibbles and the next 32 in their high nibbles, so a lane that pairs
/// byte `i` with activation `i` for both halves reads the wrong place.
pub const Q4_K_MATVEC_COALESCED_KERNEL_SRC: &str = r#"
__device__ __forceinline__ float frink_f16_to_f32_co(unsigned short bits) {
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

__device__ __forceinline__ void frink_q4_k_scale_min_co(
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

extern "C" __global__ void q4_k_matvec_coalesced(
    const unsigned char* weights,
    const float* x,
    float* out,
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps + warp;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;
    const int off = 4 * lane;
    const int oi = lane / 8;
    const int within = off % 32;

    float acc = 0.0f;
    for (int blk = 0; blk < n_blocks_per_row; ++blk) {
        const unsigned char* block = row_ptr + (size_t)blk * 144;
        const float d = frink_f16_to_f32_co(
            (unsigned short)block[0] | ((unsigned short)block[1] << 8));
        const float dmin = frink_f16_to_f32_co(
            (unsigned short)block[2] | ((unsigned short)block[3] << 8));
        const unsigned char* scales = block + 4;
        const unsigned char* qs = block + 16;

        unsigned char sc1, m1, sc2, m2;
        frink_q4_k_scale_min_co(2 * oi, scales, &sc1, &m1);
        frink_q4_k_scale_min_co(2 * oi + 1, scales, &sc2, &m2);
        const float d1 = d * (float)sc1, min1 = dmin * (float)m1;
        const float d2 = d * (float)sc2, min2 = dmin * (float)m2;

        // The whole warp's 32 loads cover qs[0..128) contiguously.
        const uchar4 w = *(const uchar4*)(qs + off);
        const int xb = blk * 256 + oi * 64 + within;
        const unsigned char wb[4] = { w.x, w.y, w.z, w.w };
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc += (d1 * (float)(wb[i] & 0x0F) - min1) * x[xb + i];
            acc += (d2 * (float)(wb[i] >> 4) - min2) * x[xb + 32 + i];
        }
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, s);
    }
    if (lane == 0) out[row] = acc;
}
"#;

/// The coalesced matvec kernel for `fn_name`, or `None` if that quant
/// kind still runs the old one-thread-per-super-block kernel.
///
/// This is the ONLY place a coalesced kernel is named. Four launchers
/// that each hard-coded one kind is exactly the shape this repo keeps
/// getting bitten by: structures that must agree with nothing enforcing
/// it. `every_coalesced_kernel_is_reachable_from_the_table` fails if a
/// kernel const is added and this table is not updated, so a kernel
/// cannot sit in the file unreachable.
pub fn coalesced_matvec_kernel(
    fn_name: &str,
) -> Option<(&'static str, &'static str, &'static str)> {
    match fn_name {
        "q4_k_matvec" => Some((
            Q4_K_MATVEC_COALESCED_KERNEL_SRC,
            "frink_q4_k_coalesced",
            "q4_k_matvec_coalesced",
        )),
        "q5_k_matvec" => Some((
            Q5_K_MATVEC_COALESCED_KERNEL_SRC,
            "frink_q5_k_coalesced",
            "q5_k_matvec_coalesced",
        )),
        "q6_k_matvec" => Some((
            Q6_K_MATVEC_COALESCED_KERNEL_SRC,
            "frink_q6_k_coalesced",
            "q6_k_matvec_coalesced",
        )),
        "q8_0_matvec" => Some((
            Q8_0_MATVEC_COALESCED_KERNEL_SRC,
            "frink_q8_0_coalesced",
            "q8_0_matvec_coalesced",
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A coalesced kernel the table does not name is dead code that
    /// reads as coverage. The entry points are derived from this file,
    /// not restated, so adding a kernel without routing it fails here.
    #[test]
    fn every_coalesced_kernel_is_reachable_from_the_table() {
        let src = include_str!("coalesced.rs");
        let mut found = 0usize;
        for line in src.lines() {
            let Some(rest) = line.strip_prefix(r#"extern "C" __global__ void "#) else {
                continue;
            };
            let Some(entry) = rest.split('(').next() else {
                continue;
            };
            let Some(base) = entry.strip_suffix("_coalesced") else {
                continue;
            };
            found += 1;
            let routed = coalesced_matvec_kernel(base).unwrap_or_else(|| {
                panic!("kernel {entry} is not reachable: no table row for {base}")
            });
            assert_eq!(
                routed.2, entry,
                "table row for {base} names the wrong entry point"
            );
            assert!(
                routed.0.contains(entry),
                "table row for {base} points at a source that does not define {entry}"
            );
        }
        assert!(
            found >= 4,
            "expected at least four coalesced kernels, found {found}"
        );
    }
}
