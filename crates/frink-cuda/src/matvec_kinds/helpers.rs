//! CUDA C helpers every matvec family reads a block header with: the
//! f16 decode and the Q4_K six-bit scale unpack. One copy, prepended to
//! the coalesced kernels ([`super::coalesced::CoalescedKind::source`])
//! and to the integer-dot ones ([`super::dp4a`]), so the two families
//! cannot decode a scale differently.

/// The shared helpers, as CUDA C text. They need nothing defined first.
pub const BLOCK_HELPERS: &str = r#"
__device__ __forceinline__ float fx_f16_to_f32(unsigned short bits) {
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

__device__ __forceinline__ float fx_f16_at(const unsigned char* p) {
    return fx_f16_to_f32((unsigned short)p[0] | ((unsigned short)p[1] << 8));
}

__device__ __forceinline__ void fx_q4_k_scale_min(
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
"#;
