//! frink-quant: dequantization kernels for the block-quantized tensor
//! formats used by GGUF files (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K).
//!
//! These block layouts are a public, widely documented convention
//! (originated in ggml). The functions here are independent
//! implementations written against that public layout description, not
//! copied from any other project's source. Q4_K and Q6_K in particular
//! are the dominant real-world GGUF quantization formats (most
//! published checkpoints ship as Q4_K_M or similar K-quant mixes, not
//! the legacy Q4_0/Q8_0 formats). These are checked against independent
//! Python cross-validation, following the same discipline as
//! `frink-models`'
//! GGUF-roundtrip tests.

pub mod encode;
pub use encode::q4_k::{encode_block_q4_k, encode_row_q4_k};
pub use encode::q5_k::{encode_block_q5_k, encode_row_q5_k};
pub use encode::q6_k::{encode_block_q6_k, encode_row_q6_k, probe_q6_k_group};
pub use encode::{encode_block_q8_0, encode_row_q8_0};

pub mod iq4_xs_q8;
/// The base-3 trit formats, ggml's `TQ1_0` and PrismML's `PTQ1_0`.
pub mod ternary;
pub use iq4_xs_q8::{dot_iq4_xs_q8_k, dot_iq4_xs_q8_k_scalar};
pub mod iq_tables;
/// ggml-produced golden vectors for the IQ2_XS/IQ2_S/IQ3_S/IQ1_M
/// kernels. Test-only: a ~60 KB data blob has no business in a release
/// build, and nothing outside the tests reads it.
#[cfg(test)]
mod iq_tier_goldens;
pub mod repack;

pub use repack::{
    batch_gemm_is_accelerated, gemm_q4_0x4_group, gemm_q4_0x4_group_x4, gemm_q4_0x4_group_x4_on,
    gemm_q4_kx8_group, gemm_q4_kx8_group_x4, gemm_q4_kx8_group_x4_on, gemm_q5_kx8_group,
    gemm_q5_kx8_group_x4, gemm_q5_kx8_group_x4_on, gemm_q6_kx8_group, gemm_q6_kx8_group_x4,
    gemm_q6_kx8_group_x4_on, gemm_q8_0x4_group, gemm_q8_0x4_group_x4, gemm_q8_0x4_group_x4_on,
    gemv_q4_0x4_group, gemv_q4_kx8_group, gemv_q4_kx8_q8_k, gemv_q5_kx8_group, gemv_q5_kx8_q8_k,
    gemv_q6_kx8_group, gemv_q6_kx8_q8_k, gemv_q8_0x4_group, gemv_q8_0x4_q8_0,
    interleaved_gemm_is_accelerated, make_block_q4_0x4, make_block_q4_kx8, make_block_q5_kx8,
    make_block_q6_kx8, make_block_q8_0x4, pack_q4_0_matrix_x4, pack_q4_k_matrix_x8,
    pack_q5_k_matrix_x8, pack_q6_k_matrix_x8, pack_q8_0_matrix_x4, preferred_interleave,
    prepare_q8_acts_x4, prepare_q8_k_acts_x4, q4_0x4_gemm_uses_acts_x4, q4_0x4_interleave,
    q4_kx8_gemm_uses_acts_x4, q4_kx8_interleave, q5_kx8_gemm_uses_acts_x4, q5_kx8_interleave,
    q6_kx8_gemm_uses_acts_x4, q6_kx8_interleave, q8_0x4_gemm_uses_acts_x4, q8_0x4_interleave,
    AccelX4, Q8ActsX4, Q8KActsX4, Q4_0X4_BLOCK_BYTES, Q4_0X4_GEMM_NC, Q4_0X4_INTERLEAVE,
    Q4_0X4_NROWS, Q4_KX8_BLOCK_BYTES, Q4_KX8_GEMM_NC, Q4_KX8_NROWS, Q5_KX8_BLOCK_BYTES,
    Q5_KX8_GEMM_NC, Q5_KX8_NROWS, Q6_KX8_BLOCK_BYTES, Q6_KX8_GEMM_NC, Q6_KX8_NROWS, Q8K_ACTS_X4_NC,
    Q8_0X4_BLOCK_BYTES, Q8_0X4_GEMM_NC, Q8_0X4_INTERLEAVE, Q8_0X4_NROWS,
};

use half::f16;

/// Q8_0: 32 int8 values sharing one f16 scale. 34 bytes per block.
pub const Q8_0_BLOCK_BYTES: usize = 34;
pub const Q8_0_BLOCK_ELEMS: usize = 32;

/// Q4_0: 32 packed 4-bit values (16 bytes) sharing one f16 scale. 18 bytes per block.
pub const Q4_0_BLOCK_BYTES: usize = 18;
pub const Q4_0_BLOCK_ELEMS: usize = 32;

/// Q4_1: like Q4_0 but asymmetric -- an f16 scale `d` *and* an f16 min
/// `m` (value = `q*d + m`, no `-8` bias), 32 packed 4-bit values.
/// Layout: d(2) + m(2) + qs(16) = 20 bytes. Verified against real
/// `ggml-common.h`/`ggml-quants.c` source, not guessed.
pub const Q4_1_BLOCK_BYTES: usize = 20;
pub const Q4_1_BLOCK_ELEMS: usize = 32;

/// Q5_0: like Q4_0 (single f16 scale `d`, symmetric `-16` bias) but
/// each element gets a 5th bit from a 4-byte `qh` bitplane. Layout:
/// d(2) + qh(4) + qs(16) = 22 bytes.
pub const Q5_0_BLOCK_BYTES: usize = 22;
pub const Q5_0_BLOCK_ELEMS: usize = 32;

/// Q5_1: Q5_0's 5th-bit scheme combined with Q4_1's asymmetric `d`+`m`
/// (no bias subtraction). Layout: d(2) + m(2) + qh(4) + qs(16) = 24
/// bytes.
pub const Q5_1_BLOCK_BYTES: usize = 24;
pub const Q5_1_BLOCK_ELEMS: usize = 32;

/// Q8_1: like Q8_0 (32 signed 8-bit values, one f16 scale `d`) plus an
/// extra f16 field `s` that upstream ggml uses only as a precomputed
/// per-block sum for its own fused SIMD dot-product kernels -- not
/// needed for correct dequantization, since `y = qs*d` is unaffected
/// by it. Layout: d(2) + s(2) + qs(32) = 36 bytes.
pub const Q8_1_BLOCK_BYTES: usize = 36;
pub const Q8_1_BLOCK_ELEMS: usize = 32;

/// The randomized Hadamard rotation the 4-bit KV wire applies to K
/// before quantizing it.
pub mod kv_rotation;

/// Metal `FRINK_CTK=q4` KV block: 32 elems → f16 scale + 16 nibble bytes.
pub const Q4_KV_GROUP: usize = 32;
pub const Q4_KV_BLOCK_BYTES: usize = 18;

/// Metal `FRINK_CTK=fp8` KV block: 32 elems → f16 scale + 32 E4M3-ish bytes.
/// Codes are absmax-scaled int8 in [-127,127] (portable stand-in for E4M3).
pub const FP8_KV_GROUP: usize = 32;
pub const FP8_KV_BLOCK_BYTES: usize = 34;

/// Pack f32 into Metal 4-bit KV blocks (no rotation).
pub fn pack_q4_kv_blocks(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % Q4_KV_GROUP, 0);
    let n_blocks = x.len() / Q4_KV_GROUP;
    let mut out = vec![0u8; n_blocks * Q4_KV_BLOCK_BYTES];
    for b in 0..n_blocks {
        let chunk = &x[b * Q4_KV_GROUP..(b + 1) * Q4_KV_GROUP];
        let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / 7.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let bits = f16::from_f32(scale).to_le_bytes();
        let dst = &mut out[b * Q4_KV_BLOCK_BYTES..(b + 1) * Q4_KV_BLOCK_BYTES];
        dst[0] = bits[0];
        dst[1] = bits[1];
        for i in 0..16 {
            let q0 = (chunk[i * 2] * inv).round().clamp(-8.0, 7.0) as i8;
            let q1 = (chunk[i * 2 + 1] * inv).round().clamp(-8.0, 7.0) as i8;
            dst[2 + i] = ((q0 as u8) & 0x0f) | (((q1 as u8) & 0x0f) << 4);
        }
    }
    out
}

/// Unpack [`pack_q4_kv_blocks`].
pub fn unpack_q4_kv_blocks(bytes: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !bytes.len().is_multiple_of(Q4_KV_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(bytes.len(), Q4_KV_BLOCK_BYTES));
    }
    let n_blocks = bytes.len() / Q4_KV_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * Q4_KV_GROUP);
    for b in 0..n_blocks {
        let block = &bytes[b * Q4_KV_BLOCK_BYTES..(b + 1) * Q4_KV_BLOCK_BYTES];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for i in 0..16 {
            let byte = block[2 + i];
            let q0 = ((byte & 0x0f) as i8) << 4 >> 4;
            let q1 = ((byte >> 4) as i8) << 4 >> 4;
            out.push(q0 as f32 * scale);
            out.push(q1 as f32 * scale);
        }
    }
    Ok(out)
}

/// Pack f32 into Metal fp8-style KV blocks (scaled int8, Q8_0-compatible layout).
pub fn pack_fp8_kv_blocks(x: &[f32]) -> Vec<u8> {
    // Same wire layout as Q8_0 — reuse for host upload/download.
    quantize_q8_0(x)
}

/// Unpack [`pack_fp8_kv_blocks`].
pub fn unpack_fp8_kv_blocks(bytes: &[u8]) -> Result<Vec<f32>, QuantError> {
    dequant_q8_0(bytes)
}

/// Q4_K: a 256-element super-block, split into 8 32-element sub-blocks,
/// each with its own 6-bit scale and 6-bit min (packed into 12 bytes),
/// plus one shared f16 scale-of-scales `d` and scale-of-mins `dmin`.
/// Layout: d(2) + dmin(2) + scales(12) + qs(128) = 144 bytes.
pub const Q4_K_BLOCK_BYTES: usize = 144;
pub const Q4_K_BLOCK_ELEMS: usize = 256;
const Q4_K_SCALE_BYTES: usize = 12;

/// Q5_K: the same 8-sub-blocks-of-32 / 6-bit-scale-and-min layout as
/// Q4_K (same 12-byte packed scales, same unpacking), but each element
/// gets a 5th bit from a separate 32-byte `qh` bitplane (one bit per
/// element, 256 bits total) instead of Q4_K's plain 4-bit nibble.
/// Layout: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176 bytes.
pub const Q5_K_BLOCK_BYTES: usize = 176;
pub const Q5_K_BLOCK_ELEMS: usize = 256;

/// Q6_K: a 256-element super-block, split into 16 16-element sub-blocks
/// each with its own signed 8-bit scale, plus one shared f16
/// super-block scale `d`. Layout: ql(128) + qh(64) + scales(16) + d(2)
/// = 210 bytes.
pub const Q6_K_BLOCK_BYTES: usize = 210;
pub const Q6_K_BLOCK_ELEMS: usize = 256;

/// Q2_K: a 256-element super-block, 16 sub-blocks of 16, each with its
/// own 4-bit scale and 4-bit min packed one byte per sub-block (not
/// Q4_K's cross-byte 6-bit packing -- a real, verified difference, not
/// assumed), plus one shared f16 super-block scale `d` and f16
/// super-block min-scale `dmin`. Layout: scales(16) + qs(64) + d(2) +
/// dmin(2) = 84 bytes -- note `d`/`dmin` come *after* `scales`/`qs`,
/// the opposite field order from every other K-quant format here,
/// verified directly against real `ggml-common.h`/`ggml-quants.c`
/// source (`block_q2_K`, `dequantize_row_q2_K`).
pub const Q2_K_BLOCK_BYTES: usize = 84;
pub const Q2_K_BLOCK_ELEMS: usize = 256;
const Q2_K_SCALE_BYTES: usize = 16;

/// Q3_K: a 256-element super-block, 16 sub-blocks of 16, each with its
/// own signed 6-bit scale (packed via a byte-wise interleaving scheme
/// across 12 bytes, verified against `dequantize_row_q3_K`'s real
/// `aux[]` unpacking -- see `q3_k_unpack_scales`'s doc comment), a
/// 3-bit value per element (2 low bits from `qs`, 1 high bit from
/// `hmask`, centered by `-4` when the high bit is *clear*), scaled by
/// one shared f16 `d`. Layout: hmask(32) + qs(64) + scales(12) + d(2)
/// = 110 bytes.
pub const Q3_K_BLOCK_BYTES: usize = 110;
pub const Q3_K_BLOCK_ELEMS: usize = 256;
const Q3_K_SCALE_BYTES: usize = 12;

#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    #[error("buffer length {0} is not a multiple of the block size {1}")]
    Misaligned(usize, usize),
    #[error("MXFP4 packed buffer is {0} bytes but scales buffer implies {1} bytes ({1} = scales.len() * MXFP4_GROUP_SIZE / 2)")]
    Mxfp4RowMismatch(usize, usize),
}

/// BF16 isn't a block-quantized format at all -- it's IEEE-754 binary32
/// truncated to its sign bit + 8 exponent bits + 7 mantissa bits (the
/// upper 16 bits of an f32), so widening it back to f32 is an exact,
/// lossless bit shift: `f32::from_bits((bits as u32) << 16)`, zero-
/// padding the low 16 mantissa bits rather than any real
/// dequantization math. Included here anyway (rather than as a one-off
/// in `frink-models::loader`) so every real element type frink
/// recognizes has one obvious home.
pub fn dequant_bf16(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(2) {
        return Err(QuantError::Misaligned(src.len(), 2));
    }
    Ok(src
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect())
}

/// F16 (IEEE-754 binary16) widened to f32. Like [`dequant_bf16`] this is
/// a plain element type, not a block format: every f16 value is exactly
/// representable in f32, so the widening is lossless. `GgmlType::F16` is
/// what `llama-quantize --pure`-free conversions and every `*-f16.gguf`
/// carry, and it is also the dtype ggml uses for `token_embd` in some
/// mixed checkpoints.
pub fn dequant_f16(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(2) {
        return Err(QuantError::Misaligned(src.len(), 2));
    }
    Ok(src
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect())
}

/// Dequantize a Q8_0 buffer into f32.
pub fn dequant_q8_0(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q8_0_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q8_0_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * Q8_0_BLOCK_ELEMS);
    for b in 0..n_blocks {
        let block = &src[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for i in 0..Q8_0_BLOCK_ELEMS {
            let q = block[2 + i] as i8;
            out.push(q as f32 * scale);
        }
    }
    Ok(out)
}

/// Dequantize a Q4_0 buffer into f32. Each byte packs two 4-bit nibbles
/// (low nibble = element i, high nibble = element i+16), each nibble
/// biased by -8 before scaling, matching the public Q4_0 convention.
pub fn dequant_q4_0(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q4_0_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q4_0_BLOCK_BYTES;
    let mut out = vec![0f32; n_blocks * Q4_0_BLOCK_ELEMS];
    for b in 0..n_blocks {
        let block = &src[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let nibbles = &block[2..18];
        let base = b * Q4_0_BLOCK_ELEMS;
        for i in 0..16 {
            let byte = nibbles[i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out[base + i] = lo as f32 * scale;
            out[base + i + 16] = hi as f32 * scale;
        }
    }
    Ok(out)
}

/// Unpacks one Q4_K super-block's 8 (scale, min) pairs from its 12-byte
/// packed `scales` field. ggml packs these as 6-bit values using a
/// scheme where the first 4 sub-blocks store their scale/min directly
/// in the low 6 bits of `scales[0..4]`/`scales[4..8]`, and the last 4
/// borrow their low 4 bits from `scales[4..8]`'s high nibble and their
/// high 2 bits from `scales[0..4]`'s top bits -- packing 8 six-bit
/// scales and 8 six-bit mins (96 bits total) into 12 bytes without
/// wasting any padding bits.
fn q4_k_scale_min(j: usize, scales: &[u8; Q4_K_SCALE_BYTES]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Dequantize a Q4_K buffer into f32. See the module doc comment and
/// `Q4_K_BLOCK_BYTES` for the block layout.
pub fn dequant_q4_k(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q4_K_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q4_K_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q4_K_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * Q4_K_BLOCK_ELEMS);
    for block in src.as_chunks::<Q4_K_BLOCK_BYTES>().0 {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
        let qs = &block[16..144];

        let mut is = 0usize;
        let mut q_off = 0usize;
        for _ in 0..4 {
            let (sc1, m1) = q4_k_scale_min(is, &scales);
            let (sc2, m2) = q4_k_scale_min(is + 1, &scales);
            let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                out.push(d1 * (qs[q_off + l] & 0x0F) as f32 - min1);
            }
            for l in 0..32 {
                out.push(d2 * (qs[q_off + l] >> 4) as f32 - min2);
            }
            q_off += 32;
            is += 2;
        }
    }
    Ok(out)
}

/// Fused Q4_K dequant+dot: identical math to `dequant_q4_k`, but
/// accumulated directly against `x` instead of materializing a
/// dequantized row. Dispatches to SIMD when the host CPU supports it,
/// same mechanism as `dot_q8_0_f32`.
pub fn dot_q4_k_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q4_k_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q4_k_f32_neon(row_bytes, x) };
        }
    }
    dot_q4_k_f32_scalar(row_bytes, x)
}

pub fn dot_q4_k_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q4_K_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    let mut base = 0usize;
    for block in row_bytes.as_chunks::<Q4_K_BLOCK_BYTES>().0 {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
        let qs = &block[16..144];

        let mut is = 0usize;
        let mut q_off = 0usize;
        for _ in 0..4 {
            let (sc1, m1) = q4_k_scale_min(is, &scales);
            let (sc2, m2) = q4_k_scale_min(is + 1, &scales);
            let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                acc += (d1 * (qs[q_off + l] & 0x0F) as f32 - min1) * x[base + l];
            }
            for l in 0..32 {
                acc += (d2 * (qs[q_off + l] >> 4) as f32 - min2) * x[base + 32 + l];
            }
            q_off += 32;
            base += 64;
            is += 2;
        }
    }
    acc
}

/// Dequantize a Q5_K buffer into f32. See the module doc comment and
/// `Q5_K_BLOCK_BYTES` for the block layout. Shares Q4_K's scale/min
/// packing (`q4_k_scale_min`) and 4-outer-iteration structure; the only
/// difference is each nibble gets a 5th bit from `qh`, whose 32 bytes
/// are reused across all 4 outer iterations at different bit positions
/// (`u1`/`u2`, doubling by 4 each iteration) rather than being consumed
/// sequentially the way `qs` is.
pub fn dequant_q5_k(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q5_K_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q5_K_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q5_K_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * Q5_K_BLOCK_ELEMS);
    for block in src.as_chunks::<Q5_K_BLOCK_BYTES>().0 {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
        let qh = &block[16..48];
        let qs = &block[48..176];

        let mut is = 0usize;
        let (mut u1, mut u2) = (1u8, 2u8);
        for oi in 0..4 {
            let (sc1, m1) = q4_k_scale_min(is, &scales);
            let (sc2, m2) = q4_k_scale_min(is + 1, &scales);
            let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
            let ql = &qs[oi * 32..oi * 32 + 32];
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * ((ql[l] & 0x0F) + hi) as f32 - min1);
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * ((ql[l] >> 4) + hi) as f32 - min2);
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

/// Fused Q5_K dequant+dot: identical math to `dequant_q5_k`, but
/// accumulated directly against `x` instead of materializing a
/// dequantized row. Dispatches to SIMD when available, same mechanism
/// as `dot_q8_0_f32`.
pub fn dot_q5_k_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q5_k_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q5_k_f32_neon(row_bytes, x) };
        }
    }
    dot_q5_k_f32_scalar(row_bytes, x)
}

pub fn dot_q5_k_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q5_K_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    let mut base = 0usize;
    for block in row_bytes.as_chunks::<Q5_K_BLOCK_BYTES>().0 {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
        let qh = &block[16..48];
        let qs = &block[48..176];

        let mut is = 0usize;
        let (mut u1, mut u2) = (1u8, 2u8);
        for oi in 0..4 {
            let (sc1, m1) = q4_k_scale_min(is, &scales);
            let (sc2, m2) = q4_k_scale_min(is + 1, &scales);
            let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
            let ql = &qs[oi * 32..oi * 32 + 32];
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                acc += (d1 * ((ql[l] & 0x0F) + hi) as f32 - min1) * x[base + l];
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                acc += (d2 * ((ql[l] >> 4) + hi) as f32 - min2) * x[base + 32 + l];
            }
            base += 64;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    acc
}

/// Dequantize a Q6_K buffer into f32. See the module doc comment and
/// `Q6_K_BLOCK_BYTES` for the block layout.
pub fn dequant_q6_k(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q6_K_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q6_K_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q6_K_BLOCK_BYTES;
    let mut out = vec![0f32; n_blocks * Q6_K_BLOCK_ELEMS];
    for (b, block) in src.as_chunks::<Q6_K_BLOCK_BYTES>().0.iter().enumerate() {
        let ql_full = &block[0..128];
        let qh_full = &block[128..192];
        let sc_full = &block[192..208];
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let out_base = b * Q6_K_BLOCK_ELEMS;

        for half in 0..2 {
            let ql = &ql_full[half * 64..half * 64 + 64];
            let qh = &qh_full[half * 32..half * 32 + 32];
            let sc = &sc_full[half * 8..half * 8 + 8];
            let y = &mut out[out_base + half * 128..out_base + half * 128 + 128];

            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                y[l] = d * (sc[is] as i8 as f32) * (q1 as f32);
                y[l + 32] = d * (sc[is + 2] as i8 as f32) * (q2 as f32);
                y[l + 64] = d * (sc[is + 4] as i8 as f32) * (q3 as f32);
                y[l + 96] = d * (sc[is + 6] as i8 as f32) * (q4 as f32);
            }
        }
    }
    Ok(out)
}

/// Fused Q6_K dequant+dot: identical math to `dequant_q6_k`, but
/// accumulated directly against `x` instead of materializing a
/// dequantized row. Dispatches to SIMD when available, same mechanism
/// as `dot_q8_0_f32`.
pub fn dot_q6_k_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q6_k_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q6_k_f32_neon(row_bytes, x) };
        }
    }
    dot_q6_k_f32_scalar(row_bytes, x)
}

pub fn dot_q6_k_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q6_K_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    let mut x_base = 0usize;
    for block in row_bytes.as_chunks::<Q6_K_BLOCK_BYTES>().0 {
        let ql_full = &block[0..128];
        let qh_full = &block[128..192];
        let sc_full = &block[192..208];
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

        for half in 0..2 {
            let ql = &ql_full[half * 64..half * 64 + 64];
            let qh = &qh_full[half * 32..half * 32 + 32];
            let sc = &sc_full[half * 8..half * 8 + 8];
            let xh = &x[x_base..x_base + 128];

            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                acc += d * (sc[is] as i8 as f32) * (q1 as f32) * xh[l];
                acc += d * (sc[is + 2] as i8 as f32) * (q2 as f32) * xh[l + 32];
                acc += d * (sc[is + 4] as i8 as f32) * (q3 as f32) * xh[l + 64];
                acc += d * (sc[is + 6] as i8 as f32) * (q4 as f32) * xh[l + 96];
            }
            x_base += 128;
        }
    }
    acc
}

/// Quantize an f32 slice into Q8_0 blocks, zero-padding a partial
/// trailing block. Used by test fixtures and by the CPU reference
/// "quantize activations for a symmetric int8 matmul" path, where the
/// vector length is not guaranteed to be a whole number of blocks.
///
/// The per-block arithmetic is [`encode::encode_block_q8_0`], not a
/// second spelling of it: this function used to have its own, which
/// divided by the scale where llama.cpp multiplies by its reciprocal
/// and stored a scale of 1.0 for an all-zero block where llama.cpp
/// stores 0.0. Both differences are invisible to a value comparison
/// and both produce different bytes, which is exactly the kind of
/// silent divergence a second copy of a code path creates. The tail
/// padding is the ONLY thing this adds.
///
/// A *weight* encoder wants [`encode::encode_row_q8_0`] instead, which
/// refuses a ragged length rather than padding it: padding a weight row
/// writes more elements than its shape declares.
pub fn quantize_q8_0(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len().div_ceil(Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES);
    for chunk in src.chunks(Q8_0_BLOCK_ELEMS) {
        let mut block = [0f32; Q8_0_BLOCK_ELEMS];
        block[..chunk.len()].copy_from_slice(chunk);
        encode::encode_block_q8_0(&block, &mut out);
    }
    out
}

/// Fused dot product between one Q8_0-quantized row (stored as raw
/// block bytes) and an f32 activation vector, without ever
/// materializing a dequantized f32 copy of the row. This is the
/// memory-bandwidth-saving trick llama.cpp's quantized matmul kernels
/// rely on: for large weight matrices, bandwidth (not FLOPs) dominates
/// inference cost, and Q8_0 moves 4x fewer bytes than a dequant-then-
/// matmul approach that expands every weight to f32 up front.
///
/// Dispatches to an AVX2+FMA SIMD kernel at runtime when the host CPU
/// supports it (checked via `is_x86_feature_detected!`), falling back
/// to the portable scalar loop
/// otherwise. Both paths are tested against each other for exact
/// numerical agreement.
pub fn dot_q8_0_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q8_0_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q8_0_f32_neon(row_bytes, x) };
        }
    }
    dot_q8_0_f32_scalar(row_bytes, x)
}

pub fn dot_q8_0_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q8_0_BLOCK_BYTES, 0);
    debug_assert_eq!(
        row_bytes.len() / Q8_0_BLOCK_BYTES * Q8_0_BLOCK_ELEMS,
        x.len()
    );
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q8_0_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let base = b * Q8_0_BLOCK_ELEMS;
        let mut block_acc = 0f32;
        for i in 0..Q8_0_BLOCK_ELEMS {
            let q = block[2 + i] as i8;
            block_acc += (q as f32) * x[base + i];
        }
        acc += block_acc * scale;
    }
    acc
}

/// An activation vector quantized to signed 8-bit in 32-element blocks,
/// each with its own f32 scale (`d`), so it can feed the integer
/// `vec_dot` paths against Q8_0 weights. This mirrors llama.cpp's
/// `quantize_row_q8_1` (minus the block sum, which is only needed for
/// asymmetric weight formats): quantizing the shared activation once per
/// matvec turns every weight-row dot into an int8×int8 → int32 reduction
/// (`vdotq_s32` / `_mm256_maddubs`-class ops) plus a single scale, which
/// is what lets llama.cpp's CPU matmul stay in integer SIMD.
#[derive(Clone, Debug)]
pub struct Q8Activations {
    /// Signed 8-bit quantized values, `n_blocks * 32` long.
    pub q: Vec<i8>,
    /// Per-block scale, `n_blocks` long. `x ≈ q * d`.
    pub d: Vec<f32>,
}

impl Q8Activations {
    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }
}

/// ggml `block_q8_K` activations for K-quant int-dot (`Q4_K`/`Q5_K`/`Q6_K`).
/// Super-blocks of 256 elements with 16-wide `bsums` for the min term.
#[derive(Clone, Debug)]
pub struct Q8KActivations {
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    /// Per 16-wide group sums of `q`, `n_blocks * 16` long.
    pub bsums: Vec<i16>,
}

impl Q8KActivations {
    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }
}

/// Quantize activations to ggml `Q8_K` (256-elem super-blocks). Positive
/// scale convention (`d = amax/127`) matching our `Q8_0` path; `bsums`
/// enable the Q4_K min correction without re-scanning `q`.
pub fn quantize_activations_q8_k(x: &[f32]) -> Q8KActivations {
    debug_assert_eq!(x.len() % Q4_K_BLOCK_ELEMS, 0);
    let n_blocks = x.len() / Q4_K_BLOCK_ELEMS;
    let mut q = vec![0i8; n_blocks * Q4_K_BLOCK_ELEMS];
    let mut d = vec![0f32; n_blocks];
    let mut bsums = vec![0i16; n_blocks * 16];
    let quant_one =
        |(q_slot, d_slot, bsum_slot, chunk): (&mut [i8], &mut f32, &mut [i16], &[f32])| {
            let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            *d_slot = scale;
            for (i, &v) in chunk.iter().enumerate() {
                let qi = (v * inv).round();
                q_slot[i] = qi.clamp(-127.0, 127.0) as i8;
            }
            for (slot, group) in bsum_slot.iter_mut().zip(q_slot.as_chunks::<16>().0) {
                *slot = group.iter().map(|&q| q as i32).sum::<i32>() as i16;
            }
        };
    // Serial on purpose: every batch caller is already inside a Rayon
    // region (one task per activation), so an inner region here nested
    // ~batch_size fork-joins per matmul; and one row's blocks are far too
    // little work to amortize one. llama quantizes serially per thread
    // chunk too (`ggml_compute_forward_mul_mat`, `ggml-cpu.c`).
    for (b, chunk) in x.as_chunks::<Q4_K_BLOCK_ELEMS>().0.iter().enumerate() {
        quant_one((
            &mut q[b * Q4_K_BLOCK_ELEMS..(b + 1) * Q4_K_BLOCK_ELEMS],
            &mut d[b],
            &mut bsums[b * 16..(b + 1) * 16],
            chunk,
        ));
    }
    Q8KActivations { q, d, bsums }
}

/// Quantize an activation row to [`Q8Activations`] (32-element blocks,
/// ggml `quantize_row_q8_0` rounding: `d = amax/127`, `q = round(x/d)`).
/// `x.len()` must be a multiple of 32.
pub fn quantize_activations_q8(x: &[f32]) -> Q8Activations {
    debug_assert_eq!(x.len() % Q8_0_BLOCK_ELEMS, 0);
    let n_blocks = x.len() / Q8_0_BLOCK_ELEMS;
    let mut q = vec![0i8; n_blocks * Q8_0_BLOCK_ELEMS];
    let mut d = vec![0f32; n_blocks];
    let quant_one = |(q_slot, d_slot, chunk): (&mut [i8], &mut f32, &[f32])| {
        let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        *d_slot = scale;
        for (i, &v) in chunk.iter().enumerate() {
            // round-half-away-from-zero, clamped to i8 range.
            let qi = (v * inv).round();
            q_slot[i] = qi.clamp(-127.0, 127.0) as i8;
        }
    };
    // Serial on purpose — see `quantize_activations_q8_k`. The parallel
    // split this replaces was also 32-byte `q` chunks (two per cache
    // line) with adjacent `d` writes: false sharing on every store.
    for (b, chunk) in x.as_chunks::<Q8_0_BLOCK_ELEMS>().0.iter().enumerate() {
        quant_one((
            &mut q[b * Q8_0_BLOCK_ELEMS..(b + 1) * Q8_0_BLOCK_ELEMS],
            &mut d[b],
            chunk,
        ));
    }
    Q8Activations { q, d }
}

/// Integer `vec_dot` of a Q8_0 weight row against pre-quantized Q8
/// activations: `Σ_blocks d_w * d_a * Σ_i (q_w · q_a)`. Dispatches to a
/// NEON `dotprod` / AVX2 kernel when available, else the scalar loop.
/// Numerically ≈ [`dot_q8_0_f32`] up to activation-quant error.
pub fn dot_q8_0_q8(row_bytes: &[u8], act: &Q8Activations) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_x86::dot_q8_0_q8_avx2(row_bytes, act) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { simd_aarch64::dot_q8_0_q8_neon_sdot(row_bytes, act) };
        }
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q8_0_q8_neon(row_bytes, act) };
        }
    }
    dot_q8_0_q8_scalar(row_bytes, act)
}

pub fn dot_q8_0_q8_scalar(row_bytes: &[u8], act: &Q8Activations) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q8_0_BLOCK_BYTES, 0);
    let n_blocks = row_bytes.len() / Q8_0_BLOCK_BYTES;
    debug_assert_eq!(n_blocks, act.n_blocks());
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q8_0_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let dw = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let base = b * Q8_0_BLOCK_ELEMS;
        let mut isum = 0i32;
        for i in 0..Q8_0_BLOCK_ELEMS {
            let qw = block[2 + i] as i8 as i32;
            let qa = act.q[base + i] as i32;
            isum += qw * qa;
        }
        acc += dw * act.d[b] * isum as f32;
    }
    acc
}

/// Integer `vec_dot` of a Q4_0 weight row against pre-quantized Q8
/// activations (llama.cpp `ggml_vec_dot_q4_0_q8_0`). Opt-in via
/// `FRINK_CPU_INT_DOT` for Q4_0 matvecs.
pub fn dot_q4_0_q8(row_bytes: &[u8], act: &Q8Activations) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_x86::dot_q4_0_q8_avx2(row_bytes, act) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { simd_aarch64::dot_q4_0_q8_neon_sdot(row_bytes, act) };
        }
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q4_0_q8_neon(row_bytes, act) };
        }
    }
    dot_q4_0_q8_scalar(row_bytes, act)
}

/// Two contiguous Q4_0 rows × one Q8 act (shared act loads). Faster than
/// two [`dot_q4_0_q8`] calls on Apple DotProd.
pub fn dot_q4_0_q8_2row(row0: &[u8], row1: &[u8], act: &Q8Activations) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod")
            && row0.len() == row1.len()
            && row0.len().is_multiple_of(Q4_0_BLOCK_BYTES)
        {
            return unsafe { simd_aarch64::dot_q4_0_q8_neon_sdot_2row(row0, row1, act) };
        }
    }
    (dot_q4_0_q8(row0, act), dot_q4_0_q8(row1, act))
}

pub fn dot_q4_0_q8_scalar(row_bytes: &[u8], act: &Q8Activations) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q4_0_BLOCK_BYTES, 0);
    let n_blocks = row_bytes.len() / Q4_0_BLOCK_BYTES;
    debug_assert_eq!(n_blocks, act.n_blocks());
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q4_0_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let dw = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let base = b * Q4_0_BLOCK_ELEMS;
        let mut isum = 0i32;
        for i in 0..16 {
            let qs = block[2 + i];
            let q0 = (qs & 0x0F) as i32 - 8;
            let q1 = (qs >> 4) as i32 - 8;
            isum += q0 * act.q[base + i] as i32;
            isum += q1 * act.q[base + 16 + i] as i32;
        }
        acc += dw * act.d[b] * isum as f32;
    }
    acc
}

/// Integer `vec_dot` of a Q4_K weight row against [`Q8KActivations`]
/// (llama.cpp `ggml_vec_dot_q4_K_q8_K`). Opt-in via `FRINK_CPU_INT_DOT`.
pub fn dot_q4_k_q8(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_x86::dot_q4_k_q8_avx2(row_bytes, act) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            return unsafe { simd_aarch64::dot_q4_k_q8_neon_i8mm(row_bytes, act) };
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { simd_aarch64::dot_q4_k_q8_neon_sdot(row_bytes, act) };
        }
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q4_k_q8_neon(row_bytes, act) };
        }
    }
    dot_q4_k_q8_scalar(row_bytes, act)
}

pub fn dot_q4_k_q8_scalar(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q4_K_BLOCK_BYTES, 0);
    let n_blocks = row_bytes.len() / Q4_K_BLOCK_BYTES;
    debug_assert_eq!(n_blocks, act.n_blocks());
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q4_K_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
        let qs = &block[16..144];
        let da = act.d[b];
        let q8 = &act.q[b * Q4_K_BLOCK_ELEMS..(b + 1) * Q4_K_BLOCK_ELEMS];
        let bsums = &act.bsums[b * 16..(b + 1) * 16];

        let mut sum_min = 0i32;
        for i in 0..8 {
            let (_, m) = q4_k_scale_min(i, &scales);
            sum_min += m as i32 * (bsums[2 * i] as i32 + bsums[2 * i + 1] as i32);
        }
        acc -= dmin * da * sum_min as f32;

        let mut q_off = 0usize;
        let mut base = 0usize;
        let mut is = 0usize;
        for _ in 0..4 {
            let (sc1, _) = q4_k_scale_min(is, &scales);
            let (sc2, _) = q4_k_scale_min(is + 1, &scales);
            let mut isum1 = 0i32;
            let mut isum2 = 0i32;
            for l in 0..32 {
                isum1 += (qs[q_off + l] & 0x0F) as i32 * q8[base + l] as i32;
            }
            for l in 0..32 {
                isum2 += (qs[q_off + l] >> 4) as i32 * q8[base + 32 + l] as i32;
            }
            acc += d * da * (sc1 as f32 * isum1 as f32 + sc2 as f32 * isum2 as f32);
            q_off += 32;
            base += 64;
            is += 2;
        }
    }
    acc
}

/// Integer `vec_dot` of a Q5_K weight row against [`Q8KActivations`]
/// (llama.cpp `ggml_vec_dot_q5_K_q8_K`). Opt-in via `FRINK_CPU_INT_DOT`.
pub fn dot_q5_k_q8(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { simd_aarch64::dot_q5_k_q8_neon_sdot(row_bytes, act) };
        }
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q5_k_q8_neon(row_bytes, act) };
        }
    }
    dot_q5_k_q8_scalar(row_bytes, act)
}

pub fn dot_q5_k_q8_scalar(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q5_K_BLOCK_BYTES, 0);
    let n_blocks = row_bytes.len() / Q5_K_BLOCK_BYTES;
    debug_assert_eq!(n_blocks, act.n_blocks());
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q5_K_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
        let qh = &block[16..48];
        let qs = &block[48..176];
        let da = act.d[b];
        let q8 = &act.q[b * Q5_K_BLOCK_ELEMS..(b + 1) * Q5_K_BLOCK_ELEMS];
        let bsums = &act.bsums[b * 16..(b + 1) * 16];

        let mut sum_min = 0i32;
        for i in 0..8 {
            let (_, m) = q4_k_scale_min(i, &scales);
            sum_min += m as i32 * (bsums[2 * i] as i32 + bsums[2 * i + 1] as i32);
        }
        acc -= dmin * da * sum_min as f32;

        let mut q_off = 0usize;
        let mut base = 0usize;
        let mut is = 0usize;
        let (mut u1, mut u2) = (1u8, 2u8);
        for _ in 0..4 {
            let (sc1, _) = q4_k_scale_min(is, &scales);
            let (sc2, _) = q4_k_scale_min(is + 1, &scales);
            let mut isum1 = 0i32;
            let mut isum2 = 0i32;
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                isum1 += ((qs[q_off + l] & 0x0F) + hi) as i32 * q8[base + l] as i32;
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                isum2 += ((qs[q_off + l] >> 4) + hi) as i32 * q8[base + 32 + l] as i32;
            }
            acc += d * da * (sc1 as f32 * isum1 as f32 + sc2 as f32 * isum2 as f32);
            q_off += 32;
            base += 64;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    acc
}

/// How many activations one [`gemm_q5_k_q8_row`] / [`gemm_q6_k_q8_row`]
/// keeps in flight. Amortizes weight-block scale/qh/qs loads over the
/// batch (Phi-4 Q5_K qkv / Q6_K ffn_down) without full Kx8 repack.
pub const Q5_K_GEMM_NC: usize = 4;
pub const Q6_K_GEMM_NC: usize = 4;

/// One Q5_K weight row × `acts.len()` Q8_K activations → `out[j]`.
///
/// Block-outer loop so each Q5_K block's scales / qh / qs are decoded once
/// and reused across activations (llama.cpp GEMM motivation without the
/// `block_q5_Kx8` interleave).
pub fn gemm_q5_k_q8_row(row_bytes: &[u8], acts: &[Q8KActivations], out: &mut [f32]) {
    assert_eq!(out.len(), acts.len());
    if acts.is_empty() {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if acts.len() <= Q5_K_GEMM_NC && std::arch::is_aarch64_feature_detected!("dotprod") {
            unsafe {
                simd_aarch64::gemm_q5_k_q8_neon_sdot(row_bytes, acts, out);
            }
            return;
        }
    }
    gemm_q5_k_q8_row_scalar(row_bytes, acts, out);
}

pub fn gemm_q5_k_q8_row_scalar(row_bytes: &[u8], acts: &[Q8KActivations], out: &mut [f32]) {
    debug_assert_eq!(row_bytes.len() % Q5_K_BLOCK_BYTES, 0);
    out.fill(0.0);
    let n_blocks = row_bytes.len() / Q5_K_BLOCK_BYTES;
    for act in acts {
        debug_assert_eq!(n_blocks, act.n_blocks());
    }
    for (b, block) in row_bytes
        .as_chunks::<Q5_K_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
        let qh = &block[16..48];
        let qs = &block[48..176];
        let mut mins = [0u8; 8];
        let mut sc_only = [0u8; 8];
        for i in 0..8 {
            let (s, m) = q4_k_scale_min(i, &scales);
            sc_only[i] = s;
            mins[i] = m;
        }
        for (j, act) in acts.iter().enumerate() {
            let da = act.d[b];
            let q8 = &act.q[b * Q5_K_BLOCK_ELEMS..(b + 1) * Q5_K_BLOCK_ELEMS];
            let bsums = &act.bsums[b * 16..(b + 1) * 16];
            let mut sum_min = 0i32;
            for i in 0..8 {
                sum_min += mins[i] as i32 * (bsums[2 * i] as i32 + bsums[2 * i + 1] as i32);
            }
            out[j] -= dmin * da * sum_min as f32;

            let mut q_off = 0usize;
            let mut base = 0usize;
            let mut is = 0usize;
            let (mut u1, mut u2) = (1u8, 2u8);
            for _ in 0..4 {
                let sc1 = sc_only[is];
                let sc2 = sc_only[is + 1];
                let mut isum1 = 0i32;
                let mut isum2 = 0i32;
                for l in 0..32 {
                    let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                    isum1 += ((qs[q_off + l] & 0x0F) + hi) as i32 * q8[base + l] as i32;
                }
                for l in 0..32 {
                    let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                    isum2 += ((qs[q_off + l] >> 4) + hi) as i32 * q8[base + 32 + l] as i32;
                }
                out[j] += d * da * (sc1 as f32 * isum1 as f32 + sc2 as f32 * isum2 as f32);
                q_off += 32;
                base += 64;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }
}

/// One Q6_K weight row × `acts.len()` Q8_K activations → `out[j]`.
pub fn gemm_q6_k_q8_row(row_bytes: &[u8], acts: &[Q8KActivations], out: &mut [f32]) {
    assert_eq!(out.len(), acts.len());
    if acts.is_empty() {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if acts.len() <= Q6_K_GEMM_NC && std::arch::is_aarch64_feature_detected!("dotprod") {
            unsafe {
                simd_aarch64::gemm_q6_k_q8_neon_sdot(row_bytes, acts, out);
            }
            return;
        }
    }
    gemm_q6_k_q8_row_scalar(row_bytes, acts, out);
}

pub fn gemm_q6_k_q8_row_scalar(row_bytes: &[u8], acts: &[Q8KActivations], out: &mut [f32]) {
    out.fill(0.0);
    for (j, act) in acts.iter().enumerate() {
        out[j] = dot_q6_k_q8_scalar(row_bytes, act);
    }
}

/// Integer `vec_dot` of a Q6_K weight row against [`Q8KActivations`]
/// (llama.cpp `ggml_vec_dot_q6_K_q8_K`). Opt-in via `FRINK_CPU_INT_DOT`.
pub fn dot_q6_k_q8(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { simd_aarch64::dot_q6_k_q8_neon_sdot(row_bytes, act) };
        }
    }
    dot_q6_k_q8_scalar(row_bytes, act)
}

pub fn dot_q6_k_q8_scalar(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q6_K_BLOCK_BYTES, 0);
    let n_blocks = row_bytes.len() / Q6_K_BLOCK_BYTES;
    debug_assert_eq!(n_blocks, act.n_blocks());
    // Q6_K uses 256-elem super-blocks; Q8_K acts share that width.
    debug_assert_eq!(Q6_K_BLOCK_ELEMS, Q4_K_BLOCK_ELEMS);
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q6_K_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let ql_full = &block[0..128];
        let qh_full = &block[128..192];
        let sc_full = &block[192..208];
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
        let da = act.d[b];
        let q8 = &act.q[b * Q6_K_BLOCK_ELEMS..(b + 1) * Q6_K_BLOCK_ELEMS];
        let mut isum = 0i32;

        for half in 0..2 {
            let ql = &ql_full[half * 64..half * 64 + 64];
            let qh = &qh_full[half * 32..half * 32 + 32];
            let sc = &sc_full[half * 8..half * 8 + 8];
            let q8h = &q8[half * 128..half * 128 + 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i8 as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32;
                isum += (sc[is] as i8 as i32) * q1 * (q8h[l] as i32);
                isum += (sc[is + 2] as i8 as i32) * q2 * (q8h[l + 32] as i32);
                isum += (sc[is + 4] as i8 as i32) * q3 * (q8h[l + 64] as i32);
                isum += (sc[is + 6] as i8 as i32) * q4 * (q8h[l + 96] as i32);
            }
        }
        acc += d * da * isum as f32;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    use super::{
        e8m0_scale, q3_k_unpack_scales, q4_k_scale_min, q5_fifth_bits, Q8Activations,
        Q8KActivations, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES, KVALUES_IQ4NL,
        MXFP4_GROUP_SIZE, Q2_K_BLOCK_BYTES, Q2_K_SCALE_BYTES, Q3_K_BLOCK_BYTES, Q3_K_SCALE_BYTES,
        Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_ELEMS, Q4_K_BLOCK_BYTES,
        Q4_K_BLOCK_ELEMS, Q4_K_SCALE_BYTES, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS, Q5_1_BLOCK_BYTES,
        Q5_1_BLOCK_ELEMS, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, Q8_0_BLOCK_BYTES,
        Q8_0_BLOCK_ELEMS, Q8_1_BLOCK_BYTES, Q8_1_BLOCK_ELEMS,
    };
    use half::f16;
    use std::arch::x86_64::*;

    /// AVX2+FMA fused Q8_0 dot product. Each 32-element block is
    /// processed as four 8-wide lanes: sign-extend 8 int8 quantized
    /// values to i32 (`_mm256_cvtepi8_epi32`), convert to f32, and
    /// fused-multiply-accumulate against the matching 8 activation
    /// values, then horizontally sum and apply the block's shared f16
    /// scale. Safety: caller must have already checked
    /// `is_x86_feature_detected!("avx2")` and `"fma"`; the function
    /// itself additionally asserts the buffer lengths line up, same as
    /// the scalar path.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8_0_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q8_0_BLOCK_BYTES, 0);
        debug_assert_eq!(
            row_bytes.len() / Q8_0_BLOCK_BYTES * Q8_0_BLOCK_ELEMS,
            x.len()
        );
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q8_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q8_0_BLOCK_ELEMS;
            let qs = &block[2..34];

            let mut block_acc = _mm256_setzero_ps();
            for g in 0..4 {
                let raw8 = _mm_loadl_epi64(qs.as_ptr().add(g * 8) as *const __m128i);
                let i32x8 = _mm256_cvtepi8_epi32(raw8);
                let f32x8 = _mm256_cvtepi32_ps(i32x8);
                let xv = _mm256_loadu_ps(x.as_ptr().add(base + g * 8));
                block_acc = _mm256_fmadd_ps(f32x8, xv, block_acc);
            }
            acc += hsum256_ps(block_acc) * scale;
        }
        acc
    }

    /// AVX2 integer Q8_0 × Q8 dot: sign-extend both operands' int8 halves
    /// to i16, `_mm256_madd_epi16` into i32 pairs (no AVX-512 VNNI needed),
    /// horizontally sum, and scale by `d_w * d_a` per block. Matches
    /// [`super::dot_q8_0_q8_scalar`] exactly (pure integer products).
    /// Safety: caller checked `is_x86_feature_detected!("avx2")`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_q8_0_q8_avx2(row_bytes: &[u8], act: &Q8Activations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q8_0_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q8_0_BLOCK_BYTES, act.n_blocks());
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q8_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let dw = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q8_0_BLOCK_ELEMS;
            let w = _mm256_loadu_si256(block.as_ptr().add(2) as *const __m256i);
            let a = _mm256_loadu_si256(act.q.as_ptr().add(base) as *const __m256i);
            let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
            let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
            let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
            let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
            let prod =
                _mm256_add_epi32(_mm256_madd_epi16(w_lo, a_lo), _mm256_madd_epi16(w_hi, a_hi));
            // horizontal sum of 8 i32 lanes
            let hi128 = _mm256_extracti128_si256(prod, 1);
            let lo128 = _mm256_castsi256_si128(prod);
            let mut sum128 = _mm_add_epi32(lo128, hi128);
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b00_00_00_01));
            let isum = _mm_cvtsi128_si32(sum128);
            acc += dw * act.d[b] * isum as f32;
        }
        acc
    }

    /// AVX2 Q4_0 × Q8 int-dot. Nibble unpack + signed bias, then
    /// `_mm256_madd_epi16` against activation i16. Safety: caller
    /// checked `avx2`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_q4_0_q8_avx2(row_bytes: &[u8], act: &Q8Activations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_0_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q4_0_BLOCK_BYTES, act.n_blocks());
        let low_mask = _mm_set1_epi8(0x0F);
        let bias = _mm_set1_epi8(8);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q4_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let dw = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q4_0_BLOCK_ELEMS;
            let qs = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i);
            let lo = _mm_sub_epi8(_mm_and_si128(qs, low_mask), bias);
            let hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qs, 4), low_mask), bias);
            // Interleave lo (0..15) then hi (16..31) into 32 i8 → widen to i16.
            let w = _mm256_set_m128i(hi, lo);
            let a = _mm256_loadu_si256(act.q.as_ptr().add(base) as *const __m256i);
            let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
            let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
            let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
            let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
            let prod =
                _mm256_add_epi32(_mm256_madd_epi16(w_lo, a_lo), _mm256_madd_epi16(w_hi, a_hi));
            let hi128 = _mm256_extracti128_si256(prod, 1);
            let lo128 = _mm256_castsi256_si128(prod);
            let mut sum128 = _mm_add_epi32(lo128, hi128);
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b00_00_00_01));
            let isum = _mm_cvtsi128_si32(sum128);
            acc += dw * act.d[b] * isum as f32;
        }
        acc
    }

    /// AVX2 Q4_K × Q8_K int-dot. Matches [`super::dot_q4_k_q8_scalar`].
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_q4_k_q8_avx2(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_K_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q4_K_BLOCK_BYTES, act.n_blocks());
        let low_mask = _mm256_set1_epi8(0x0F_u8 as i8);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q4_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            let qs = &block[16..144];
            let da = act.d[b];
            let q8 = act.q.as_ptr().add(b * Q4_K_BLOCK_ELEMS);
            let bsums = &act.bsums[b * 16..(b + 1) * 16];

            let mut sum_min = 0i32;
            for i in 0..8 {
                let (_, m) = q4_k_scale_min(i, &scales);
                sum_min += m as i32 * (bsums[2 * i] as i32 + bsums[2 * i + 1] as i32);
            }
            acc -= dmin * da * sum_min as f32;

            let mut q_off = 0usize;
            let mut base = 0usize;
            let mut is = 0usize;
            for _ in 0..4 {
                let (sc1, _) = q4_k_scale_min(is, &scales);
                let (sc2, _) = q4_k_scale_min(is + 1, &scales);
                let packed = _mm256_loadu_si256(qs.as_ptr().add(q_off) as *const __m256i);
                let lo = _mm256_and_si256(packed, low_mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                let a0 = _mm256_loadu_si256(q8.add(base) as *const __m256i);
                let a1 = _mm256_loadu_si256(q8.add(base + 32) as *const __m256i);
                let isum1 = madd_i8_avx2(lo, a0);
                let isum2 = madd_i8_avx2(hi, a1);
                acc += d * da * (sc1 as f32 * isum1 as f32 + sc2 as f32 * isum2 as f32);
                q_off += 32;
                base += 64;
                is += 2;
            }
        }
        acc
    }

    #[target_feature(enable = "avx2")]
    unsafe fn madd_i8_avx2(w: __m256i, a: __m256i) -> i32 {
        let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
        let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
        let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
        let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
        let prod = _mm256_add_epi32(_mm256_madd_epi16(w_lo, a_lo), _mm256_madd_epi16(w_hi, a_hi));
        let hi128 = _mm256_extracti128_si256(prod, 1);
        let lo128 = _mm256_castsi256_si128(prod);
        let mut sum128 = _mm_add_epi32(lo128, hi128);
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b00_00_00_01));
        _mm_cvtsi128_si32(sum128)
    }

    /// AVX2+FMA fused Q4_0 dot product. Each block packs 32 4-bit
    /// values into 16 bytes: byte `i`'s low nibble is element `i`,
    /// high nibble is element `i+16`, both biased by -8. High-nibble
    /// extraction uses the standard `_mm_srli_epi16(bytes, 4) & 0x0F`
    /// trick (shifting as 16-bit lanes, then masking per-byte, avoids
    /// needing a per-byte shift instruction which x86 SIMD doesn't
    /// have below AVX-512). Safety: same contract as
    /// `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4_0_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_0_BLOCK_BYTES, 0);
        let bias = _mm_set1_epi8(8);
        let low_mask = _mm_set1_epi8(0x0F);

        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q4_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q4_0_BLOCK_ELEMS;
            let nibbles = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i);

            let lo_nibbles = _mm_sub_epi8(_mm_and_si128(nibbles, low_mask), bias);
            let hi_nibbles =
                _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(nibbles, 4), low_mask), bias);

            let mut block_acc = _mm256_setzero_ps();
            // elements 0..16 (lo_nibbles), two 8-wide groups
            for (group_idx, half) in [
                (0usize, lo_nibbles),
                (1usize, _mm_srli_si128(lo_nibbles, 8)),
                (2usize, hi_nibbles),
                (3usize, _mm_srli_si128(hi_nibbles, 8)),
            ] {
                let i32x8 = _mm256_cvtepi8_epi32(half);
                let f32x8 = _mm256_cvtepi32_ps(i32x8);
                let elem_base = base + group_idx * 8;
                let xv = _mm256_loadu_ps(x.as_ptr().add(elem_base));
                block_acc = _mm256_fmadd_ps(f32x8, xv, block_acc);
            }
            acc += hsum256_ps(block_acc) * scale;
        }
        acc
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(hi, lo);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(sums2)
    }

    /// Widens 16 unsigned nibble-derived byte values (0..=15, or 0..=31
    /// once Q5_K has OR'd in a 5th bit) held in the low and high halves
    /// of `part` into 8 lanes of f32 via `_mm256_cvtepu8_epi32` (zero-
    /// extending unsigned widen, unlike Q8_0/Q4_0's signed
    /// `_mm256_cvtepi8_epi32` -- K-quant nibbles are never negative
    /// before the affine `d*q - min` transform is applied), then
    /// dequantizes as `d*q - min` and fused-multiply-accumulates
    /// against the matching 8 activations. Called twice per 16-byte
    /// group (`part` = the low 8 bytes, then the high 8 bytes via
    /// `_mm_srli_si128(part, 8)`) to cover all 16 lanes, mirroring the
    /// existing Q4_0 AVX2 kernel's `_mm_srli_si128(lo_nibbles, 8)`
    /// idiom for the same reason (AVX2 has no direct 16-lane u8->i32
    /// widen).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn fma_affine8(
        part: __m128i,
        d: f32,
        min: f32,
        x: &[f32],
        x_base: usize,
        acc: __m256,
    ) -> __m256 {
        let i32x8 = _mm256_cvtepu8_epi32(part);
        let f32x8 = _mm256_cvtepi32_ps(i32x8);
        let weight = _mm256_fmsub_ps(f32x8, _mm256_set1_ps(d), _mm256_set1_ps(min));
        let xv = _mm256_loadu_ps(x.as_ptr().add(x_base));
        _mm256_fmadd_ps(weight, xv, acc)
    }

    /// AVX2+FMA fused Q4_K dot product. Mirrors `dot_q4_0_f32_avx2`'s
    /// nibble-splitting structure (low/high nibble of each byte are two
    /// independent output elements, each 16-byte load's nibbles split
    /// into two 8-wide `_mm256_cvtepu8_epi32` groups via
    /// `_mm_srli_si128(_, 8)`), scaled up from Q4_0's 16 bytes/block to
    /// Q4_K's 32 bytes/sub-block (two 16-byte loads instead of one),
    /// with the affine `d*q - min` transform (independent (scale, min)
    /// pairs for the low-nibble half and the high-nibble half) instead
    /// of Q4_0's single symmetric `d*(q-8)`. Safety: same contract as
    /// `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4_k_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_K_BLOCK_BYTES, 0);
        let low_mask = _mm_set1_epi8(0x0F);
        let mut acc = 0f32;
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<Q4_K_BLOCK_BYTES>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            let qs = &block[16..144];

            let mut is = 0usize;
            let mut q_off = 0usize;
            for _ in 0..4 {
                let (sc1, m1) = q4_k_scale_min(is, &scales);
                let (sc2, m2) = q4_k_scale_min(is + 1, &scales);
                let d1 = d * sc1 as f32;
                let min1 = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let min2 = dmin * m2 as f32;

                let mut lo_acc = _mm256_setzero_ps();
                let mut hi_acc = _mm256_setzero_ps();
                for g in 0..2 {
                    let raw16 = _mm_loadu_si128(qs.as_ptr().add(q_off + g * 16) as *const __m128i);
                    let lo_nib = _mm_and_si128(raw16, low_mask);
                    let hi_nib = _mm_and_si128(_mm_srli_epi16(raw16, 4), low_mask);

                    for (part_idx, part) in
                        [lo_nib, _mm_srli_si128(lo_nib, 8)].into_iter().enumerate()
                    {
                        lo_acc =
                            fma_affine8(part, d1, min1, x, x_base + g * 16 + part_idx * 8, lo_acc);
                    }
                    for (part_idx, part) in
                        [hi_nib, _mm_srli_si128(hi_nib, 8)].into_iter().enumerate()
                    {
                        hi_acc = fma_affine8(
                            part,
                            d2,
                            min2,
                            x,
                            x_base + 32 + g * 16 + part_idx * 8,
                            hi_acc,
                        );
                    }
                }
                acc += hsum256_ps(lo_acc) + hsum256_ps(hi_acc);
                q_off += 32;
                x_base += 64;
                is += 2;
            }
        }
        acc
    }

    /// AVX2+FMA fused Q5_K dot product: identical structure to
    /// `dot_q4_k_f32_avx2`, but before widening, each nibble gets a 5th
    /// bit OR'd in from the block's `qh` bitplane. The per-lane "is bit
    /// `u1`/`u2` set in this byte of `qh`" test uses an equality-based
    /// mask (`_mm_cmpeq_epi8(masked, zero)`, inverted via
    /// `_mm_andnot_si128`) rather than `_mm_cmpgt_epi8`: `u1`/`u2` sweep
    /// up to 128 (`u2` reaches `0x80`), which as a *signed* i8 is
    /// negative, so a signed greater-than comparison would silently
    /// misclassify a set high bit as "not greater than zero" -- the
    /// equality test is agnostic to that sign issue since it only asks
    /// "is the masked byte zero or not." Safety: same contract as
    /// `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q5_k_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q5_K_BLOCK_BYTES, 0);
        let low_mask = _mm_set1_epi8(0x0F);
        let zero = _mm_setzero_si128();
        let sixteen = _mm_set1_epi8(16);
        let mut acc = 0f32;
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<Q5_K_BLOCK_BYTES>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            let qh = &block[16..48];
            let qs = &block[48..176];

            let mut is = 0usize;
            let (mut u1, mut u2) = (1u8, 2u8);
            for _oi in 0..4 {
                let (sc1, m1) = q4_k_scale_min(is, &scales);
                let (sc2, m2) = q4_k_scale_min(is + 1, &scales);
                let d1 = d * sc1 as f32;
                let min1 = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let min2 = dmin * m2 as f32;
                let ql = &qs[is / 2 * 32..is / 2 * 32 + 32];
                let u1_vec = _mm_set1_epi8(u1 as i8);
                let u2_vec = _mm_set1_epi8(u2 as i8);

                let mut lo_acc = _mm256_setzero_ps();
                let mut hi_acc = _mm256_setzero_ps();
                for g in 0..2 {
                    let raw16 = _mm_loadu_si128(ql.as_ptr().add(g * 16) as *const __m128i);
                    let qh16 = _mm_loadu_si128(qh.as_ptr().add(g * 16) as *const __m128i);

                    let lo_nib = _mm_and_si128(raw16, low_mask);
                    let hi_nib = _mm_and_si128(_mm_srli_epi16(raw16, 4), low_mask);

                    let is_zero1 = _mm_cmpeq_epi8(_mm_and_si128(qh16, u1_vec), zero);
                    let hi_bit1 = _mm_andnot_si128(is_zero1, sixteen);
                    let is_zero2 = _mm_cmpeq_epi8(_mm_and_si128(qh16, u2_vec), zero);
                    let hi_bit2 = _mm_andnot_si128(is_zero2, sixteen);

                    let lo_full = _mm_or_si128(lo_nib, hi_bit1);
                    let hi_full = _mm_or_si128(hi_nib, hi_bit2);

                    for (part_idx, part) in [lo_full, _mm_srli_si128(lo_full, 8)]
                        .into_iter()
                        .enumerate()
                    {
                        lo_acc =
                            fma_affine8(part, d1, min1, x, x_base + g * 16 + part_idx * 8, lo_acc);
                    }
                    for (part_idx, part) in [hi_full, _mm_srli_si128(hi_full, 8)]
                        .into_iter()
                        .enumerate()
                    {
                        hi_acc = fma_affine8(
                            part,
                            d2,
                            min2,
                            x,
                            x_base + 32 + g * 16 + part_idx * 8,
                            hi_acc,
                        );
                    }
                }
                acc += hsum256_ps(lo_acc) + hsum256_ps(hi_acc);
                x_base += 64;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        acc
    }

    /// AVX2+FMA fused Q6_K dot product. Each 32-element group (`q1..q4`
    /// in the scalar reference) is processed 16 lanes at a time: the
    /// 6-bit value is `(ql nibble) | (qh 2-bit field << 4)`. Unlike the
    /// NEON kernel (which centers by `-32` in the signed-int domain
    /// before converting to f32), this widens the raw *unsigned* 0..=63
    /// value straight to f32 via `_mm256_cvtepu8_epi32` and subtracts
    /// `32.0` as a float afterward (`_mm256_sub_ps`) -- simpler here
    /// since x86 has no cheap signed-widen-with-bias trick to match
    /// NEON's, and float subtraction of a small exact integer bias from
    /// a small exact integer value is itself exact, so the two
    /// approaches agree bit-for-bit on every representable input. The
    /// `qh` 2-bit-field shift amount (0/2/4/6) must be a compile-time
    /// constant at `_mm_srli_epi16`'s call site (`rustc` rejects a
    /// plain runtime `i32` there with "attempt to use a non-constant
    /// value in a constant" -- confirmed directly, not assumed), hence
    /// `q6_k_group_avx2`'s `const QH_SHIFT` generic, monomorphized once
    /// per group at its four call sites below (unlike NEON's equivalent
    /// split, x86's shift-by-immediate accepts N=0 fine, so no separate
    /// zero-shift function is needed here). Safety: same contract as
    /// `dot_q8_0_f32_avx2`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn q6_k_group_avx2<const QH_SHIFT: i32, const HI_NIBBLE: bool>(
        ql: &[u8],
        ql_off: usize,
        qh: &[u8],
        sc: &[u8],
        sc_base: usize,
        d: f32,
        x: &[f32],
        x_base: usize,
        out_off: usize,
        low_mask: __m128i,
        two_bit_mask: __m128i,
        bias: __m256,
    ) -> f32 {
        let mut acc = 0f32;
        for sub in 0..2usize {
            let byte_off = sub * 16;
            let ql_raw = _mm_loadu_si128(ql.as_ptr().add(ql_off + byte_off) as *const __m128i);
            let qh_raw = _mm_loadu_si128(qh.as_ptr().add(byte_off) as *const __m128i);

            let nib = if HI_NIBBLE {
                _mm_and_si128(_mm_srli_epi16(ql_raw, 4), low_mask)
            } else {
                _mm_and_si128(ql_raw, low_mask)
            };
            let qh_field = _mm_and_si128(_mm_srli_epi16(qh_raw, QH_SHIFT), two_bit_mask);
            let raw6 = _mm_or_si128(nib, _mm_slli_epi16(qh_field, 4));

            let scale = d * (sc[sc_base + sub] as i8) as f32;
            let elem_base = x_base + out_off + sub * 16;
            for (part_idx, part) in [raw6, _mm_srli_si128(raw6, 8)].into_iter().enumerate() {
                let i32x8 = _mm256_cvtepu8_epi32(part);
                let f32x8 = _mm256_sub_ps(_mm256_cvtepi32_ps(i32x8), bias);
                let xv = _mm256_loadu_ps(x.as_ptr().add(elem_base + part_idx * 8));
                let weighted = _mm256_mul_ps(f32x8, _mm256_set1_ps(scale));
                acc += hsum256_ps(_mm256_mul_ps(weighted, xv));
            }
        }
        acc
    }

    /// AVX2+FMA fused Q6_K dot product: dispatches each of the four
    /// 32-element groups per half-block (`q1..q4` in the scalar
    /// reference) to `q6_k_group_avx2`, monomorphized once per group's
    /// (compile-time-constant) `qh` shift amount and nibble half.
    /// Safety: same contract as `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q6_k_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q6_K_BLOCK_BYTES, 0);
        debug_assert_eq!(
            row_bytes.len() / Q6_K_BLOCK_BYTES * Q6_K_BLOCK_ELEMS,
            x.len()
        );
        let low_mask = _mm_set1_epi8(0x0F);
        let two_bit_mask = _mm_set1_epi8(0x03);
        let bias = _mm256_set1_ps(32.0);

        let mut acc = 0f32;
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<Q6_K_BLOCK_BYTES>().0 {
            let ql_full = &block[0..128];
            let qh_full = &block[128..192];
            let sc_full = &block[192..208];
            let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

            for half in 0..2 {
                let ql = &ql_full[half * 64..half * 64 + 64];
                let qh = &qh_full[half * 32..half * 32 + 32];
                let sc = &sc_full[half * 8..half * 8 + 8];
                let half_base = x_base + half * 128;

                acc += q6_k_group_avx2::<0, false>(
                    ql,
                    0,
                    qh,
                    sc,
                    0,
                    d,
                    x,
                    half_base,
                    0,
                    low_mask,
                    two_bit_mask,
                    bias,
                );
                acc += q6_k_group_avx2::<2, false>(
                    ql,
                    32,
                    qh,
                    sc,
                    2,
                    d,
                    x,
                    half_base,
                    32,
                    low_mask,
                    two_bit_mask,
                    bias,
                );
                acc += q6_k_group_avx2::<4, true>(
                    ql,
                    0,
                    qh,
                    sc,
                    4,
                    d,
                    x,
                    half_base,
                    64,
                    low_mask,
                    two_bit_mask,
                    bias,
                );
                acc += q6_k_group_avx2::<6, true>(
                    ql,
                    32,
                    qh,
                    sc,
                    6,
                    d,
                    x,
                    half_base,
                    96,
                    low_mask,
                    two_bit_mask,
                    bias,
                );
            }
            x_base += Q6_K_BLOCK_ELEMS;
        }
        acc
    }

    /// Decodes 8 real E2M1 codebook values (one nibble byte per lane,
    /// each 0..=15, held in the low 8 bytes of `nib`) into `__m256`,
    /// arithmetically rather than via a 16-entry float lookup table --
    /// see `simd_aarch64::mxfp4_nibbles_to_f32_quads`'s doc comment for
    /// the derivation (identical formula, just AVX2 intrinsics:
    /// `_mm_shuffle_epi8` for the 2-bit-exponent -> `{pow2,bias}` lookup
    /// instead of NEON's `vqtbl1q_u8`, `_mm256_cvtepu8_epi32` to widen
    /// instead of NEON's `widen_u8x16_to_f32_quads`).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mxfp4_nibbles_to_f32x8(nib: __m128i) -> __m256 {
        let sign_bit = _mm_and_si128(nib, _mm_set1_epi8(0x8));
        let e = _mm_and_si128(_mm_srli_epi16(nib, 1), _mm_set1_epi8(0x3));
        let m = _mm_and_si128(nib, _mm_set1_epi8(0x1));

        let pow2_table = _mm_setr_epi8(1, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let bias_table = _mm_setr_epi8(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let pow2_u8 = _mm_shuffle_epi8(pow2_table, e);
        let bias_u8 = _mm_shuffle_epi8(bias_table, e);

        let pow2_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pow2_u8));
        let bias_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bias_u8));
        let m_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(m));
        let sign_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(sign_bit));

        // magnitude = pow2 * (bias + 0.5*m); value = magnitude * (1 - 0.25*sign)
        let magnitude = _mm256_mul_ps(pow2_f, _mm256_fmadd_ps(m_f, _mm256_set1_ps(0.5), bias_f));
        let sign_mul = _mm256_fnmadd_ps(sign_f, _mm256_set1_ps(0.25), _mm256_set1_ps(1.0));
        _mm256_mul_ps(magnitude, sign_mul)
    }

    /// AVX2+FMA fused MXFP4 dequant+dot -- same real math as
    /// `dot_mxfp4_row_f32_scalar` (real E2M1 codebook + E8M0 scale),
    /// decoded via `mxfp4_nibbles_to_f32x8` instead of the scalar
    /// path's 16-entry `KVALUES_MXFP4` table lookup. Cross-validated
    /// against the scalar reference across many packed-byte patterns
    /// (see this module's tests) -- CI runs this on real x86_64
    /// hardware, matching the project's established
    /// verify-on-real-hardware-not-just-compile discipline for every
    /// other AVX2 kernel here.
    pub unsafe fn dot_mxfp4_row_f32_avx2(packed: &[u8], scales: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(packed.len(), scales.len() * (MXFP4_GROUP_SIZE / 2));
        let low_mask = _mm_set1_epi8(0x0F);
        let mut acc = 0f32;
        let mut x_base = 0usize;
        for (g, &e_byte) in scales.iter().enumerate() {
            let d = e8m0_scale(e_byte);
            let group = &packed[g * 16..(g + 1) * 16];
            let bytes = _mm_loadu_si128(group.as_ptr() as *const __m128i);
            let lo_nib = _mm_and_si128(bytes, low_mask);
            let hi_nib = _mm_and_si128(_mm_srli_epi16(bytes, 4), low_mask);

            let mut block_acc = _mm256_setzero_ps();
            for (half_idx, nib) in [
                (0usize, lo_nib),
                (1usize, _mm_srli_si128(lo_nib, 8)),
                (2usize, hi_nib),
                (3usize, _mm_srli_si128(hi_nib, 8)),
            ] {
                let vals = mxfp4_nibbles_to_f32x8(nib);
                let elem_base = x_base + half_idx * 8;
                let xv = _mm256_loadu_ps(x.as_ptr().add(elem_base));
                block_acc = _mm256_fmadd_ps(vals, xv, block_acc);
            }
            acc += hsum256_ps(block_acc) * d;
            x_base += MXFP4_GROUP_SIZE;
        }
        acc
    }

    /// AVX2+FMA fused Q8_1 dot product. Mathematically identical to
    /// `dot_q8_0_f32_avx2` (`y = q*d`, no `min` term) -- Q8_1's block
    /// just has an extra 2-byte field between `d` and the int8 values,
    /// so the quantized bytes start at offset 4 instead of offset 2.
    /// Safety: same contract as `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8_1_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q8_1_BLOCK_BYTES, 0);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q8_1_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q8_1_BLOCK_ELEMS;
            let qs = &block[4..36];

            let mut block_acc = _mm256_setzero_ps();
            for g in 0..4 {
                let raw8 = _mm_loadl_epi64(qs.as_ptr().add(g * 8) as *const __m128i);
                let i32x8 = _mm256_cvtepi8_epi32(raw8);
                let f32x8 = _mm256_cvtepi32_ps(i32x8);
                let xv = _mm256_loadu_ps(x.as_ptr().add(base + g * 8));
                block_acc = _mm256_fmadd_ps(f32x8, xv, block_acc);
            }
            acc += hsum256_ps(block_acc) * d;
        }
        acc
    }

    /// AVX2+FMA fused Q4_1 dot product. Same nibble-splitting structure
    /// as `dot_q4_0_f32_avx2`, but asymmetric (`y = nibble*d + m`, no
    /// bias subtraction) -- reuses `fma_affine8` (which computes `q*d -
    /// min`) by passing `-m` as `min`, since `q*d - (-m) == q*d + m`.
    /// Safety: same contract as `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4_1_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_1_BLOCK_BYTES, 0);
        let low_mask = _mm_set1_epi8(0x0F);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q4_1_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let base = b * Q4_1_BLOCK_ELEMS;
            let nibbles = _mm_loadu_si128(block.as_ptr().add(4) as *const __m128i);

            let lo_nibbles = _mm_and_si128(nibbles, low_mask);
            let hi_nibbles = _mm_and_si128(_mm_srli_epi16(nibbles, 4), low_mask);

            let mut lo_acc = _mm256_setzero_ps();
            let mut hi_acc = _mm256_setzero_ps();
            for (part_idx, part) in [lo_nibbles, _mm_srli_si128(lo_nibbles, 8)]
                .into_iter()
                .enumerate()
            {
                lo_acc = fma_affine8(part, d, -m, x, base + part_idx * 8, lo_acc);
            }
            for (part_idx, part) in [hi_nibbles, _mm_srli_si128(hi_nibbles, 8)]
                .into_iter()
                .enumerate()
            {
                hi_acc = fma_affine8(part, d, -m, x, base + 16 + part_idx * 8, hi_acc);
            }
            acc += hsum256_ps(lo_acc) + hsum256_ps(hi_acc);
        }
        acc
    }

    /// AVX2+FMA fused Q5_0 dot product. The 5th-bit-per-element
    /// extraction (`q5_fifth_bits`) is done in scalar prep, once per
    /// block, into a stack-local `[i8; 32]` array (each value already
    /// includes the `-16` symmetric bias) -- deliberately not
    /// vectorized, since the real per-lane-varying bit-position test
    /// this needs is a correctness-sensitive detail not worth risking a
    /// hand-rolled SIMD mistake on for a single already-small (16-bit)
    /// bitplane; the actual per-element multiply-accumulate over all 32
    /// elements, where the real throughput cost lives, is fully
    /// vectorized exactly like `dot_q8_0_f32_avx2`. Safety: same
    /// contract as `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q5_0_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q5_0_BLOCK_BYTES, 0);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q5_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
            let qs = &block[6..22];
            let base = b * Q5_0_BLOCK_ELEMS;

            let mut vals = [0i8; 32];
            for j in 0..16 {
                let (xh_0, xh_1) = q5_fifth_bits(qh, j);
                vals[j] = (((qs[j] & 0x0F) | xh_0) as i32 - 16) as i8;
                vals[j + 16] = (((qs[j] >> 4) | xh_1) as i32 - 16) as i8;
            }

            let mut block_acc = _mm256_setzero_ps();
            for g in 0..4 {
                let raw8 = _mm_loadl_epi64(vals.as_ptr().add(g * 8) as *const __m128i);
                let i32x8 = _mm256_cvtepi8_epi32(raw8);
                let f32x8 = _mm256_cvtepi32_ps(i32x8);
                let xv = _mm256_loadu_ps(x.as_ptr().add(base + g * 8));
                block_acc = _mm256_fmadd_ps(f32x8, xv, block_acc);
            }
            acc += hsum256_ps(block_acc) * d;
        }
        acc
    }

    /// AVX2+FMA fused Q5_1 dot product. Same 5th-bit scalar-prep
    /// approach as `dot_q5_0_f32_avx2`, but asymmetric (`y = q*d + m`,
    /// no `-16` bias) -- see that function's doc comment for why the
    /// bit extraction stays scalar. Safety: same contract as
    /// `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q5_1_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q5_1_BLOCK_BYTES, 0);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q5_1_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let qh = u32::from_le_bytes(block[4..8].try_into().unwrap());
            let qs = &block[8..24];
            let base = b * Q5_1_BLOCK_ELEMS;

            let mut vals = [0u8; 32];
            for j in 0..16 {
                let (xh_0, xh_1) = q5_fifth_bits(qh, j);
                vals[j] = (qs[j] & 0x0F) | xh_0;
                vals[j + 16] = (qs[j] >> 4) | xh_1;
            }

            let mut block_acc = _mm256_setzero_ps();
            for g in 0..4 {
                let raw8 = _mm_loadl_epi64(vals.as_ptr().add(g * 8) as *const __m128i);
                let i32x8 = _mm256_cvtepu8_epi32(raw8);
                let f32x8 = _mm256_cvtepi32_ps(i32x8);
                let weight = _mm256_fmadd_ps(f32x8, _mm256_set1_ps(d), _mm256_set1_ps(m));
                let xv = _mm256_loadu_ps(x.as_ptr().add(base + g * 8));
                block_acc = _mm256_fmadd_ps(weight, xv, block_acc);
            }
            acc += hsum256_ps(block_acc);
        }
        acc
    }

    /// AVX2+FMA fused Q2_K dot product. Mirrors `dot_q4_k_f32_avx2`'s
    /// sub-block loop, but each element is a 2-bit value (`(byte >>
    /// shift) & 3`) instead of a nibble, and each sub-block's
    /// (scale, min) is one plain byte (`sc & 0x0F` / `sc >> 4`), not
    /// Q4_K's cross-byte 6-bit packing. `shift` only ever takes the
    /// values 0/2/4/6, and `_mm_srli_epi16` requires a compile-time-
    /// constant shift amount, so the 4 shift values are unrolled as 4
    /// literal call sites via this macro rather than a runtime loop --
    /// same reason this file's `q6_k_group_avx2` takes `QH_SHIFT` as a
    /// const generic. The same "shift 16-bit lanes, mask per byte"
    /// trick `dot_q4_0_f32_avx2` uses for nibbles generalizes exactly
    /// to 2-bit fields: masking with `0x03` after `_mm_srli_epi16`
    /// discards the neighboring byte's bits that leak into the shift,
    /// for any of the 4 shift amounts. Safety: same contract as
    /// `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q2_k_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q2_K_BLOCK_BYTES, 0);
        let two_bit_mask = _mm_set1_epi8(3);
        let mut acc = 0f32;
        let mut x_base = 0usize;

        macro_rules! q2_k_sub_block {
            ($shift:literal, $q:expr, $scales:expr, $is:expr, $d:expr, $dmin:expr, $x:expr, $x_base:expr, $acc:expr) => {{
                let sc1 = $scales[$is];
                $is += 1;
                let dl1 = $d * (sc1 & 0x0F) as f32;
                let ml1 = $dmin * (sc1 >> 4) as f32;
                let sc2 = $scales[$is];
                $is += 1;
                let dl2 = $d * (sc2 & 0x0F) as f32;
                let ml2 = $dmin * (sc2 >> 4) as f32;

                let lo16 = _mm_loadu_si128($q.as_ptr() as *const __m128i);
                let hi16 = _mm_loadu_si128($q.as_ptr().add(16) as *const __m128i);
                let lo2 = _mm_and_si128(_mm_srli_epi16(lo16, $shift), two_bit_mask);
                let hi2 = _mm_and_si128(_mm_srli_epi16(hi16, $shift), two_bit_mask);

                let mut lo_acc = _mm256_setzero_ps();
                let mut hi_acc = _mm256_setzero_ps();
                for (part_idx, part) in [lo2, _mm_srli_si128(lo2, 8)].into_iter().enumerate() {
                    lo_acc = fma_affine8(part, dl1, ml1, $x, $x_base + part_idx * 8, lo_acc);
                }
                for (part_idx, part) in [hi2, _mm_srli_si128(hi2, 8)].into_iter().enumerate() {
                    hi_acc = fma_affine8(part, dl2, ml2, $x, $x_base + 16 + part_idx * 8, hi_acc);
                }
                $acc += hsum256_ps(lo_acc) + hsum256_ps(hi_acc);
                $x_base += 32;
            }};
        }

        for block in row_bytes.as_chunks::<Q2_K_BLOCK_BYTES>().0 {
            let scales: &[u8; Q2_K_SCALE_BYTES] = block[0..16].try_into().unwrap();
            let qs = &block[16..80];
            let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
            let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();

            let mut is = 0usize;
            for n in 0..2 {
                let q = &qs[n * 32..n * 32 + 32];
                q2_k_sub_block!(0, q, scales, is, d, dmin, x, x_base, acc);
                q2_k_sub_block!(2, q, scales, is, d, dmin, x, x_base, acc);
                q2_k_sub_block!(4, q, scales, is, d, dmin, x, x_base, acc);
                q2_k_sub_block!(6, q, scales, is, d, dmin, x, x_base, acc);
            }
        }
        acc
    }

    /// AVX2+FMA fused Q3_K dot product. Same 2-bit-field extraction
    /// trick as `dot_q2_k_f32_avx2` (shift-then-mask, 4 literal shift
    /// values), plus a 3rd bit tested from `hmask` the same way
    /// `dot_q5_k_f32_avx2` tests Q5_K's 5th bit (`_mm_cmpeq_epi8`
    /// against zero, inverted, since the tested bit position `m` sweeps
    /// up to `0x80`, which as signed i8 would misclassify under a
    /// signed greater-than test). `bias` (4 or 0) is applied as a
    /// per-lane select between two constant vectors rather than a
    /// branch. The 6-bit per-sub-block scale unpacking
    /// (`q3_k_unpack_scales`) runs once per block on the scalar side
    /// (cheap, real bit-shuffling not worth vectorizing for a
    /// once-per-block cost), reusing the existing scalar helper exactly
    /// rather than re-deriving it. Safety: same contract as
    /// `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q3_k_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q3_K_BLOCK_BYTES, 0);
        let two_bit_mask = _mm_set1_epi8(3);
        let zero = _mm_setzero_si128();
        let four = _mm_set1_epi8(4);
        let mut acc = 0f32;
        let mut x_base = 0usize;

        macro_rules! q3_k_sub_block {
            ($shift:literal, $q:expr, $hmask:expr, $m_vec:expr, $dl1:expr, $dl2:expr, $x:expr, $x_base:expr, $acc:expr) => {{
                let lo16 = _mm_loadu_si128($q.as_ptr() as *const __m128i);
                let hi16 = _mm_loadu_si128($q.as_ptr().add(16) as *const __m128i);
                let lo2 = _mm_and_si128(_mm_srli_epi16(lo16, $shift), two_bit_mask);
                let hi2 = _mm_and_si128(_mm_srli_epi16(hi16, $shift), two_bit_mask);

                let hmask_lo = _mm_loadu_si128($hmask.as_ptr() as *const __m128i);
                let hmask_hi = _mm_loadu_si128($hmask.as_ptr().add(16) as *const __m128i);
                // bit_clear_* is all-ones (0xFF) per lane where the hmask bit is
                // CLEAR (bias=4), all-zero where it's set (bias=0) -- matching
                // the scalar reference's `if hmask[l] & m != 0 { 0 } else { 4 }`.
                let bit_clear_lo = _mm_cmpeq_epi8(_mm_and_si128(hmask_lo, $m_vec), zero);
                let bit_clear_hi = _mm_cmpeq_epi8(_mm_and_si128(hmask_hi, $m_vec), zero);
                let bias_lo = _mm_and_si128(bit_clear_lo, four);
                let bias_hi = _mm_and_si128(bit_clear_hi, four);
                let raw_lo = _mm_sub_epi8(lo2, bias_lo);
                let raw_hi = _mm_sub_epi8(hi2, bias_hi);

                let mut lo_acc = _mm256_setzero_ps();
                let mut hi_acc = _mm256_setzero_ps();
                for (part_idx, part) in [raw_lo, _mm_srli_si128(raw_lo, 8)].into_iter().enumerate()
                {
                    let i32x8 = _mm256_cvtepi8_epi32(part);
                    let f32x8 = _mm256_cvtepi32_ps(i32x8);
                    let xv = _mm256_loadu_ps($x.as_ptr().add($x_base + part_idx * 8));
                    lo_acc = _mm256_fmadd_ps(f32x8, xv, lo_acc);
                }
                for (part_idx, part) in [raw_hi, _mm_srli_si128(raw_hi, 8)].into_iter().enumerate()
                {
                    let i32x8 = _mm256_cvtepi8_epi32(part);
                    let f32x8 = _mm256_cvtepi32_ps(i32x8);
                    let xv = _mm256_loadu_ps($x.as_ptr().add($x_base + 16 + part_idx * 8));
                    hi_acc = _mm256_fmadd_ps(f32x8, xv, hi_acc);
                }
                $acc += hsum256_ps(lo_acc) * $dl1 + hsum256_ps(hi_acc) * $dl2;
                $x_base += 32;
            }};
        }

        for block in row_bytes.as_chunks::<Q3_K_BLOCK_BYTES>().0 {
            let hmask = &block[0..32];
            let qs = &block[32..96];
            let scales_raw: &[u8; Q3_K_SCALE_BYTES] = block[96..108].try_into().unwrap();
            let d_all = f16::from_le_bytes([block[108], block[109]]).to_f32();
            let scales = q3_k_unpack_scales(scales_raw);

            let mut is = 0usize;
            let mut m = 1u8;
            for n in 0..2 {
                let q = &qs[n * 32..n * 32 + 32];
                for shift in [0u32, 2, 4, 6] {
                    let dl1 = d_all * (scales[is] as f32 - 32.0);
                    let dl2 = d_all * (scales[is + 1] as f32 - 32.0);
                    is += 2;
                    let m_vec = _mm_set1_epi8(m as i8);
                    match shift {
                        0 => q3_k_sub_block!(0, q, hmask, m_vec, dl1, dl2, x, x_base, acc),
                        2 => q3_k_sub_block!(2, q, hmask, m_vec, dl1, dl2, x, x_base, acc),
                        4 => q3_k_sub_block!(4, q, hmask, m_vec, dl1, dl2, x, x_base, acc),
                        6 => q3_k_sub_block!(6, q, hmask, m_vec, dl1, dl2, x, x_base, acc),
                        _ => unreachable!(),
                    }
                    m <<= 1;
                }
            }
        }
        acc
    }

    /// AVX2 fused IQ4_NL dot product. `KVALUES_IQ4NL`'s 16 entries are
    /// arbitrary (non-arithmetic) signed values, so unlike MXFP4's
    /// bit-twiddled reconstruction, the natural AVX2 idiom is a direct
    /// 16-entry table lookup via `_mm_shuffle_epi8` (`pshufb`), which is
    /// exactly a 4-bit-index-into-16-byte-table lookup within each
    /// 128-bit lane -- precisely this shape. Safety: same contract as
    /// `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_iq4_nl_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % IQ4_NL_BLOCK_BYTES, 0);
        let low_mask = _mm_set1_epi8(0x0F);
        let codebook = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let mut acc = 0f32;
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<IQ4_NL_BLOCK_BYTES>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..18];
            let bytes = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
            let lo_idx = _mm_and_si128(bytes, low_mask);
            let hi_idx = _mm_and_si128(_mm_srli_epi16(bytes, 4), low_mask);
            let lo_vals = _mm_shuffle_epi8(codebook, lo_idx);
            let hi_vals = _mm_shuffle_epi8(codebook, hi_idx);

            let mut block_acc = _mm256_setzero_ps();
            for (half_idx, vals) in [
                (0usize, lo_vals),
                (1usize, _mm_srli_si128(lo_vals, 8)),
                (2usize, hi_vals),
                (3usize, _mm_srli_si128(hi_vals, 8)),
            ] {
                let i32x8 = _mm256_cvtepi8_epi32(vals);
                let f32x8 = _mm256_cvtepi32_ps(i32x8);
                let xv = _mm256_loadu_ps(x.as_ptr().add(x_base + half_idx * 8));
                block_acc = _mm256_fmadd_ps(f32x8, xv, block_acc);
            }
            acc += hsum256_ps(block_acc) * d;
            x_base += IQ4_NL_BLOCK_ELEMS;
        }
        acc
    }

    /// AVX2 fused IQ4_XS dot product. Same codebook lookup as
    /// `dot_iq4_nl_f32_avx2`, repeated per 32-element sub-block (8 per
    /// 256-element block), each with its own 6-bit scale unpacked
    /// exactly as the scalar reference does (once per sub-block, cheap,
    /// not vectorized). Safety: same contract as `dot_q8_0_f32_avx2`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_iq4_xs_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % IQ4_XS_BLOCK_BYTES, 0);
        let low_mask = _mm_set1_epi8(0x0F);
        let codebook = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let mut acc = 0f32;
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<IQ4_XS_BLOCK_BYTES>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let scales_h = u16::from_le_bytes([block[2], block[3]]);
            let scales_l = &block[4..8];
            let qs = &block[8..136];

            for ib in 0..8 {
                let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                    | (((scales_h >> (2 * ib)) & 3) as u8) << 4;
                let dl = d * (ls as f32 - 32.0);
                let sub = &qs[ib * 16..ib * 16 + 16];
                let bytes = _mm_loadu_si128(sub.as_ptr() as *const __m128i);
                let lo_idx = _mm_and_si128(bytes, low_mask);
                let hi_idx = _mm_and_si128(_mm_srli_epi16(bytes, 4), low_mask);
                let lo_vals = _mm_shuffle_epi8(codebook, lo_idx);
                let hi_vals = _mm_shuffle_epi8(codebook, hi_idx);

                let mut sub_acc = _mm256_setzero_ps();
                for (half_idx, vals) in [
                    (0usize, lo_vals),
                    (1usize, _mm_srli_si128(lo_vals, 8)),
                    (2usize, hi_vals),
                    (3usize, _mm_srli_si128(hi_vals, 8)),
                ] {
                    let i32x8 = _mm256_cvtepi8_epi32(vals);
                    let f32x8 = _mm256_cvtepi32_ps(i32x8);
                    let xv = _mm256_loadu_ps(x.as_ptr().add(x_base + half_idx * 8));
                    sub_acc = _mm256_fmadd_ps(f32x8, xv, sub_acc);
                }
                acc += hsum256_ps(sub_acc) * dl;
                x_base += 32;
            }
        }
        acc
    }

    /// Expands one 8-value grid row of *unsigned* byte magnitudes into
    /// 8 f32 lanes with the format's per-element signs applied --
    /// shared by the IQ2_XXS/IQ3_XXS kernels below. `signs` is the
    /// 8-bit `ksigns_iq2xs` pattern for this row; a set bit `j` (the
    /// same `kmask_iq2xs` convention the scalar path uses) negates
    /// lane `j`, done here by XORing the f32 sign bit from a bit-test
    /// mask rather than multiplying by ±1.0.
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn iq_grid_row_signed_f32(row_le: u64, signs: u8) -> __m256 {
        let mags = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_set_epi64x(0, row_le as i64)));
        let bit_mask = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let bits = _mm256_and_si256(_mm256_set1_epi32(signs as i32), bit_mask);
        let neg = _mm256_cmpeq_epi32(bits, bit_mask);
        let sign_bit = _mm256_and_si256(neg, _mm256_set1_epi32(0x8000_0000_u32 as i32));
        _mm256_xor_ps(mags, _mm256_castsi256_ps(sign_bit))
    }

    /// AVX2+FMA fused IQ1_S dot: same walk as the scalar reference
    /// (grid rows of signed int8, per-group scale `dl` and additive
    /// `delta`), vectorized 8 elements at a time. Verified directly
    /// against the scalar path on real x86_64 hardware (this module's
    /// tests), whose goldens are themselves cross-validated against
    /// the compiled ggml implementation.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_iq1_s_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % crate::IQ1_S_BLOCK_BYTES, 0);
        let mut acc = _mm256_setzero_ps();
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<{ crate::IQ1_S_BLOCK_BYTES }>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..34];
            let qh = &block[34..50];
            for ib in 0..8 {
                let h = u16::from_le_bytes([qh[2 * ib], qh[2 * ib + 1]]);
                let dl = d * (2.0 * ((h >> 12) & 7) as f32 + 1.0);
                let delta = if h & 0x8000 != 0 {
                    -crate::IQ1S_DELTA
                } else {
                    crate::IQ1S_DELTA
                };
                let dl_v = _mm256_set1_ps(dl);
                let delta_v = _mm256_set1_ps(delta);
                for l in 0..4 {
                    let idx = qs[4 * ib + l] as usize | ((((h >> (3 * l)) & 7) as usize) << 8);
                    let row = crate::iq_tables::IQ1S_GRID[idx];
                    let g = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_set_epi64x(0, row as i64)));
                    let vals = _mm256_mul_ps(dl_v, _mm256_add_ps(g, delta_v));
                    let xv = _mm256_loadu_ps(x.as_ptr().add(x_base));
                    acc = _mm256_fmadd_ps(vals, xv, acc);
                    x_base += 8;
                }
            }
        }
        hsum256_ps(acc)
    }

    /// AVX2+FMA fused IQ2_XXS dot -- same decode as the scalar
    /// reference (u16 codes -> grid rows + ksigns patterns + packed
    /// 4-bit group scale), 8 elements per FMA. Verification: see
    /// `dot_iq1_s_f32_avx2`'s doc comment.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_iq2_xxs_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % crate::IQ2_XXS_BLOCK_BYTES, 0);
        let mut acc = _mm256_setzero_ps();
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<{ crate::IQ2_XXS_BLOCK_BYTES }>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            for ib32 in 0..8 {
                let g0 = u16::from_le_bytes([block[2 + 8 * ib32], block[3 + 8 * ib32]]);
                let g1 = u16::from_le_bytes([block[4 + 8 * ib32], block[5 + 8 * ib32]]);
                let g2 = u16::from_le_bytes([block[6 + 8 * ib32], block[7 + 8 * ib32]]);
                let g3 = u16::from_le_bytes([block[8 + 8 * ib32], block[9 + 8 * ib32]]);
                let aux32_1 = g2 as u32 | ((g3 as u32) << 16);
                let db = _mm256_set1_ps(d * (0.5 + (aux32_1 >> 28) as f32) * 0.25);
                let aux8 = [
                    (g0 & 0xFF) as usize,
                    (g0 >> 8) as usize,
                    (g1 & 0xFF) as usize,
                    (g1 >> 8) as usize,
                ];
                for (l, &code) in aux8.iter().enumerate() {
                    let signs =
                        crate::iq_tables::KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 127) as usize];
                    let vals = iq_grid_row_signed_f32(crate::iq_tables::IQ2XXS_GRID[code], signs);
                    let xv = _mm256_loadu_ps(x.as_ptr().add(x_base));
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(db, vals), xv, acc);
                    x_base += 8;
                }
            }
        }
        hsum256_ps(acc)
    }

    /// AVX2+FMA fused IQ3_XXS dot -- two u32 grid rows per 8 elements,
    /// combined into one 8-byte magnitude row, then the shared
    /// sign/scale path. Verification: see `dot_iq1_s_f32_avx2`'s doc
    /// comment.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_iq3_xxs_f32_avx2(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % crate::IQ3_XXS_BLOCK_BYTES, 0);
        let mut acc = _mm256_setzero_ps();
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<{ crate::IQ3_XXS_BLOCK_BYTES }>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..66];
            let sas = &block[66..98];
            for ib32 in 0..8 {
                let aux32 = u32::from_le_bytes([
                    sas[4 * ib32],
                    sas[4 * ib32 + 1],
                    sas[4 * ib32 + 2],
                    sas[4 * ib32 + 3],
                ]);
                let db = _mm256_set1_ps(d * (0.5 + (aux32 >> 28) as f32) * 0.5);
                for l in 0..4 {
                    let signs = crate::iq_tables::KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                    let r1 = crate::iq_tables::IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize];
                    let r2 = crate::iq_tables::IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize];
                    let row = (r1 as u64) | ((r2 as u64) << 32);
                    let vals = iq_grid_row_signed_f32(row, signs);
                    let xv = _mm256_loadu_ps(x.as_ptr().add(x_base));
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(db, vals), xv, acc);
                    x_base += 8;
                }
            }
        }
        hsum256_ps(acc)
    }
}

/// ARM NEON kernels, mirroring `simd_x86`'s structure and math exactly
/// (same block layouts, same bias/scale handling) but using NEON's
/// 128-bit vectors: 16 int8 lanes per load instead of AVX2's 32-lane
/// (4x8) processing, widened in two steps (int8 -> int16 -> int32) via
/// `vmovl_*` rather than AVX2's single-step `_mm256_cvtepi8_epi32`,
/// since NEON has no direct int8-to-int32 widen instruction. NEON is
/// part of the aarch64 baseline ISA (unlike AVX2 on x86_64, which is
/// optional), so `is_aarch64_feature_detected!` is expected to always
/// return true on real aarch64 hardware -- kept for the same "detect,
/// don't assume" discipline the AVX2 dispatch uses, and so this
/// degrades gracefully if ever compiled for a hypothetical NEON-less
/// aarch64 target.
#[cfg(target_arch = "aarch64")]
mod simd_aarch64 {
    use super::{
        e8m0_scale, q3_k_unpack_scales, q4_k_scale_min, q5_fifth_bits, Q8Activations,
        Q8KActivations, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES, KVALUES_IQ4NL,
        MXFP4_GROUP_SIZE, Q2_K_BLOCK_BYTES, Q2_K_SCALE_BYTES, Q3_K_BLOCK_BYTES, Q3_K_SCALE_BYTES,
        Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_ELEMS, Q4_K_BLOCK_BYTES,
        Q4_K_BLOCK_ELEMS, Q4_K_SCALE_BYTES, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS, Q5_1_BLOCK_BYTES,
        Q5_1_BLOCK_ELEMS, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS,
        Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS, Q8_1_BLOCK_BYTES, Q8_1_BLOCK_ELEMS,
    };
    use half::f16;
    use std::arch::aarch64::*;

    /// NEON fused Q8_0 dot product. Each 32-element block is processed
    /// as two 16-wide loads, each widened int8 -> int16 -> int32 (via
    /// `vmovl_s8` then `vmovl_s16`, splitting low/high halves with
    /// `vget_low`/`vget_high` at each step since NEON widening
    /// instructions only operate on 64-bit half-registers), converted
    /// to f32, and fused-multiply-accumulated against the matching
    /// activation values with `vfmaq_f32`, then horizontally summed
    /// with `vaddvq_f32` (an aarch64-only reduction intrinsic) and
    /// scaled by the block's shared f16 scale. Safety: caller must have
    /// already checked `is_aarch64_feature_detected!("neon")`; the
    /// function itself additionally asserts the buffer lengths line up,
    /// same as the scalar path.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q8_0_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q8_0_BLOCK_BYTES, 0);
        debug_assert_eq!(
            row_bytes.len() / Q8_0_BLOCK_BYTES * Q8_0_BLOCK_ELEMS,
            x.len()
        );
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q8_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q8_0_BLOCK_ELEMS;
            let qs = &block[2..34];

            let mut block_acc = vdupq_n_f32(0.0);
            for g in 0..2 {
                let raw16 = vld1q_s8(qs.as_ptr().add(g * 16) as *const i8);
                let lo16 = vmovl_s8(vget_low_s8(raw16));
                let hi16 = vmovl_s8(vget_high_s8(raw16));
                for (half_idx, half16) in [lo16, hi16].into_iter().enumerate() {
                    let lo32 = vmovl_s16(vget_low_s16(half16));
                    let hi32 = vmovl_s16(vget_high_s16(half16));
                    let f_lo = vcvtq_f32_s32(lo32);
                    let f_hi = vcvtq_f32_s32(hi32);
                    let elem_base = base + g * 16 + half_idx * 8;
                    let x_lo = vld1q_f32(x.as_ptr().add(elem_base));
                    let x_hi = vld1q_f32(x.as_ptr().add(elem_base + 4));
                    block_acc = vfmaq_f32(block_acc, f_lo, x_lo);
                    block_acc = vfmaq_f32(block_acc, f_hi, x_hi);
                }
            }
            acc += vaddvq_f32(block_acc) * scale;
        }
        acc
    }

    /// NEON integer Q8_0 × Q8 dot via widening multiply (no SDOT).
    /// Prefer [`dot_q8_0_q8_neon_sdot`] when `dotprod` is available.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q8_0_q8_neon(row_bytes: &[u8], act: &Q8Activations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q8_0_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q8_0_BLOCK_BYTES, act.n_blocks());
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q8_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let dw = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q8_0_BLOCK_ELEMS;
            let mut isum = vdupq_n_s32(0);
            for g in 0..2 {
                let w = vld1q_s8(block.as_ptr().add(2 + g * 16) as *const i8);
                let a = vld1q_s8(act.q.as_ptr().add(base + g * 16));
                let prod_lo = vmull_s8(vget_low_s8(w), vget_low_s8(a));
                let prod_hi = vmull_s8(vget_high_s8(w), vget_high_s8(a));
                isum = vpadalq_s16(isum, prod_lo);
                isum = vpadalq_s16(isum, prod_hi);
            }
            acc += dw * act.d[b] * vaddvq_s32(isum) as f32;
        }
        acc
    }

    /// Stable SDOT via inline asm (`vdotq_s32` is nightly-only).
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn neon_sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
        acc
    }

    /// NEON Q8_0 × Q8 int-dot with SDOT (Apple Silicon / ARMv8.2+).
    /// Two-block unroll + float4 scale-accumulate (llama.cpp ARM style).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q8_0_q8_neon_sdot(row_bytes: &[u8], act: &Q8Activations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q8_0_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q8_0_BLOCK_BYTES, act.n_blocks());
        let nb = row_bytes.len() / Q8_0_BLOCK_BYTES;
        let mut sumv0 = vdupq_n_f32(0.0);
        let mut sumv1 = vdupq_n_f32(0.0);
        let mut b = 0usize;
        while b + 1 < nb {
            let block0 = row_bytes.as_ptr().add(b * Q8_0_BLOCK_BYTES);
            let block1 = row_bytes.as_ptr().add((b + 1) * Q8_0_BLOCK_BYTES);
            let dw0 = f16::from_le_bytes([*block0, *block0.add(1)]).to_f32();
            let dw1 = f16::from_le_bytes([*block1, *block1.add(1)]).to_f32();
            let base0 = b * Q8_0_BLOCK_ELEMS;
            let base1 = (b + 1) * Q8_0_BLOCK_ELEMS;
            let mut isum0 = vdupq_n_s32(0);
            let mut isum1 = vdupq_n_s32(0);
            for g in 0..2 {
                let w0 = vld1q_s8(block0.add(2 + g * 16) as *const i8);
                let w1 = vld1q_s8(block1.add(2 + g * 16) as *const i8);
                let a0 = vld1q_s8(act.q.as_ptr().add(base0 + g * 16));
                let a1 = vld1q_s8(act.q.as_ptr().add(base1 + g * 16));
                isum0 = neon_sdot(isum0, w0, a0);
                isum1 = neon_sdot(isum1, w1, a1);
            }
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(isum0), dw0 * act.d[b]);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(isum1), dw1 * act.d[b + 1]);
            b += 2;
        }
        let mut acc = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
        if b < nb {
            let block = row_bytes.as_ptr().add(b * Q8_0_BLOCK_BYTES);
            let dw = f16::from_le_bytes([*block, *block.add(1)]).to_f32();
            let base = b * Q8_0_BLOCK_ELEMS;
            let mut isum = vdupq_n_s32(0);
            for g in 0..2 {
                let w = vld1q_s8(block.add(2 + g * 16) as *const i8);
                let a = vld1q_s8(act.q.as_ptr().add(base + g * 16));
                isum = neon_sdot(isum, w, a);
            }
            acc += dw * act.d[b] * vaddvq_s32(isum) as f32;
        }
        acc
    }

    /// NEON Q4_0 × Q8 int-dot. Unpack nibbles → signed i8, then same
    /// `vmull_s8`/`vpadalq_s16` reduction as Q8×Q8. Safety: caller
    /// checked neon.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q4_0_q8_neon(row_bytes: &[u8], act: &Q8Activations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_0_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q4_0_BLOCK_BYTES, act.n_blocks());
        let bias = vdupq_n_s8(8);
        let low_mask = vdupq_n_u8(0x0F);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q4_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let dw = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q4_0_BLOCK_ELEMS;
            let nibbles = vld1q_u8(block.as_ptr().add(2));
            let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(nibbles, low_mask)), bias);
            let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(nibbles, 4)), bias);
            let mut isum = vdupq_n_s32(0);
            // lo = elems 0..15, hi = elems 16..31 — matches act layout.
            let a0 = vld1q_s8(act.q.as_ptr().add(base));
            let a1 = vld1q_s8(act.q.as_ptr().add(base + 16));
            let p0_lo = vmull_s8(vget_low_s8(lo), vget_low_s8(a0));
            let p0_hi = vmull_s8(vget_high_s8(lo), vget_high_s8(a0));
            let p1_lo = vmull_s8(vget_low_s8(hi), vget_low_s8(a1));
            let p1_hi = vmull_s8(vget_high_s8(hi), vget_high_s8(a1));
            isum = vpadalq_s16(isum, p0_lo);
            isum = vpadalq_s16(isum, p0_hi);
            isum = vpadalq_s16(isum, p1_lo);
            isum = vpadalq_s16(isum, p1_hi);
            acc += dw * act.d[b] * vaddvq_s32(isum) as f32;
        }
        acc
    }

    /// Two weight rows × one act: share Q8 loads, dual SDOT accumulate.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q4_0_q8_neon_sdot_2row(
        row0: &[u8],
        row1: &[u8],
        act: &Q8Activations,
    ) -> (f32, f32) {
        debug_assert_eq!(row0.len(), row1.len());
        debug_assert_eq!(row0.len() % Q4_0_BLOCK_BYTES, 0);
        let bias = vdupq_n_s8(8);
        let low_mask = vdupq_n_u8(0x0F);
        let nb = row0.len() / Q4_0_BLOCK_BYTES;
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        for b in 0..nb {
            let p0 = row0.as_ptr().add(b * Q4_0_BLOCK_BYTES);
            let p1 = row1.as_ptr().add(b * Q4_0_BLOCK_BYTES);
            let dw0 = f16::from_le_bytes([*p0, *p0.add(1)]).to_f32();
            let dw1 = f16::from_le_bytes([*p1, *p1.add(1)]).to_f32();
            let base = b * Q4_0_BLOCK_ELEMS;
            let a_lo = vld1q_s8(act.q.as_ptr().add(base));
            let a_hi = vld1q_s8(act.q.as_ptr().add(base + 16));
            let nib0 = vld1q_u8(p0.add(2));
            let nib1 = vld1q_u8(p1.add(2));
            let lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(nib0, low_mask)), bias);
            let hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(nib0, 4)), bias);
            let lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(nib1, low_mask)), bias);
            let hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(nib1, 4)), bias);
            let mut is0 = neon_sdot(vdupq_n_s32(0), lo0, a_lo);
            is0 = neon_sdot(is0, hi0, a_hi);
            let mut is1 = neon_sdot(vdupq_n_s32(0), lo1, a_lo);
            is1 = neon_sdot(is1, hi1, a_hi);
            let scale = act.d[b];
            sum0 = vmlaq_n_f32(sum0, vcvtq_f32_s32(is0), dw0 * scale);
            sum1 = vmlaq_n_f32(sum1, vcvtq_f32_s32(is1), dw1 * scale);
        }
        (vaddvq_f32(sum0), vaddvq_f32(sum1))
    }

    /// NEON Q4_0 × Q8 with SDOT. Two-block unroll + float4 scale-accumulate.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q4_0_q8_neon_sdot(row_bytes: &[u8], act: &Q8Activations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_0_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q4_0_BLOCK_BYTES, act.n_blocks());
        let bias = vdupq_n_s8(8);
        let low_mask = vdupq_n_u8(0x0F);
        let nb = row_bytes.len() / Q4_0_BLOCK_BYTES;
        let mut sumv0 = vdupq_n_f32(0.0);
        let mut sumv1 = vdupq_n_f32(0.0);
        let mut b = 0usize;
        while b + 1 < nb {
            let block0 = row_bytes.as_ptr().add(b * Q4_0_BLOCK_BYTES);
            let block1 = row_bytes.as_ptr().add((b + 1) * Q4_0_BLOCK_BYTES);
            let dw0 = f16::from_le_bytes([*block0, *block0.add(1)]).to_f32();
            let dw1 = f16::from_le_bytes([*block1, *block1.add(1)]).to_f32();
            let base0 = b * Q4_0_BLOCK_ELEMS;
            let base1 = (b + 1) * Q4_0_BLOCK_ELEMS;
            let nib0 = vld1q_u8(block0.add(2));
            let nib1 = vld1q_u8(block1.add(2));
            let lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(nib0, low_mask)), bias);
            let hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(nib0, 4)), bias);
            let lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(nib1, low_mask)), bias);
            let hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(nib1, 4)), bias);
            let mut isum0 = neon_sdot(vdupq_n_s32(0), lo0, vld1q_s8(act.q.as_ptr().add(base0)));
            isum0 = neon_sdot(isum0, hi0, vld1q_s8(act.q.as_ptr().add(base0 + 16)));
            let mut isum1 = neon_sdot(vdupq_n_s32(0), lo1, vld1q_s8(act.q.as_ptr().add(base1)));
            isum1 = neon_sdot(isum1, hi1, vld1q_s8(act.q.as_ptr().add(base1 + 16)));
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(isum0), dw0 * act.d[b]);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(isum1), dw1 * act.d[b + 1]);
            b += 2;
        }
        let mut acc = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
        if b < nb {
            let block = &row_bytes[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES];
            let dw = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q4_0_BLOCK_ELEMS;
            let nibbles = vld1q_u8(block.as_ptr().add(2));
            let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(nibbles, low_mask)), bias);
            let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(nibbles, 4)), bias);
            let mut isum = neon_sdot(vdupq_n_s32(0), lo, vld1q_s8(act.q.as_ptr().add(base)));
            isum = neon_sdot(isum, hi, vld1q_s8(act.q.as_ptr().add(base + 16)));
            acc += dw * act.d[b] * vaddvq_s32(isum) as f32;
        }
        acc
    }

    #[target_feature(enable = "neon")]
    unsafe fn neon_i8_dot_widen(mut isum: int32x4_t, w: int8x16_t, a: int8x16_t) -> int32x4_t {
        let prod_lo = vmull_s8(vget_low_s8(w), vget_low_s8(a));
        let prod_hi = vmull_s8(vget_high_s8(w), vget_high_s8(a));
        isum = vpadalq_s16(isum, prod_lo);
        vpadalq_s16(isum, prod_hi)
    }

    /// NEON Q4_K × Q8_K int-dot (widening path).
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q4_k_q8_neon(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_K_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q4_K_BLOCK_BYTES, act.n_blocks());
        let low_mask = vdupq_n_u8(0x0F);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q4_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            let qs = &block[16..144];
            let da = act.d[b];
            let q8 = act.q.as_ptr().add(b * Q4_K_BLOCK_ELEMS);
            let bsums = &act.bsums[b * 16..(b + 1) * 16];

            let mut sum_min = 0i32;
            for i in 0..8 {
                let (_, m) = q4_k_scale_min(i, &scales);
                sum_min += m as i32 * (bsums[2 * i] as i32 + bsums[2 * i + 1] as i32);
            }
            acc -= dmin * da * sum_min as f32;

            let mut q_off = 0usize;
            let mut base = 0usize;
            let mut is = 0usize;
            for _ in 0..4 {
                let (sc1, _) = q4_k_scale_min(is, &scales);
                let (sc2, _) = q4_k_scale_min(is + 1, &scales);
                let mut isum1 = vdupq_n_s32(0);
                let mut isum2 = vdupq_n_s32(0);
                for g in 0..2 {
                    let packed = vld1q_u8(qs.as_ptr().add(q_off + g * 16));
                    let lo = vreinterpretq_s8_u8(vandq_u8(packed, low_mask));
                    let hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
                    let a0 = vld1q_s8(q8.add(base + g * 16));
                    let a1 = vld1q_s8(q8.add(base + 32 + g * 16));
                    isum1 = neon_i8_dot_widen(isum1, lo, a0);
                    isum2 = neon_i8_dot_widen(isum2, hi, a1);
                }
                acc += d
                    * da
                    * (sc1 as f32 * vaddvq_s32(isum1) as f32
                        + sc2 as f32 * vaddvq_s32(isum2) as f32);
                q_off += 32;
                base += 64;
                is += 2;
            }
        }
        acc
    }

    /// NEON Q4_K × Q8_K on i8mm hosts. llama.cpp `ggml_vec_dot_q4_K_q8_K`
    /// uses SMMLA only for nrc==2 / repacked GEMM tiles (see repack.cpp);
    /// single-row vec-dot stays on dotprod until frink Q4_K repack lands.
    /// Dispatched when `is_aarch64_feature_detected!("i8mm")` so callers
    /// can prefer the feature without changing numerics.
    #[target_feature(enable = "neon,i8mm")]
    pub unsafe fn dot_q4_k_q8_neon_i8mm(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        dot_q4_k_q8_neon_sdot(row_bytes, act)
    }

    /// NEON Q4_K × Q8_K with SDOT.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q4_k_q8_neon_sdot(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_K_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q4_K_BLOCK_BYTES, act.n_blocks());
        let low_mask = vdupq_n_u8(0x0F);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q4_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            let qs = &block[16..144];
            let da = act.d[b];
            let q8 = act.q.as_ptr().add(b * Q4_K_BLOCK_ELEMS);
            let bsums = &act.bsums[b * 16..(b + 1) * 16];

            let mut sum_min = 0i32;
            for i in 0..8 {
                let (_, m) = q4_k_scale_min(i, &scales);
                sum_min += m as i32 * (bsums[2 * i] as i32 + bsums[2 * i + 1] as i32);
            }
            acc -= dmin * da * sum_min as f32;

            let mut q_off = 0usize;
            let mut base = 0usize;
            let mut is = 0usize;
            for _ in 0..4 {
                let (sc1, _) = q4_k_scale_min(is, &scales);
                let (sc2, _) = q4_k_scale_min(is + 1, &scales);
                let mut isum1 = vdupq_n_s32(0);
                let mut isum2 = vdupq_n_s32(0);
                for g in 0..2 {
                    let packed = vld1q_u8(qs.as_ptr().add(q_off + g * 16));
                    let lo = vreinterpretq_s8_u8(vandq_u8(packed, low_mask));
                    let hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
                    let a0 = vld1q_s8(q8.add(base + g * 16));
                    let a1 = vld1q_s8(q8.add(base + 32 + g * 16));
                    isum1 = neon_sdot(isum1, lo, a0);
                    isum2 = neon_sdot(isum2, hi, a1);
                }
                acc += d
                    * da
                    * (sc1 as f32 * vaddvq_s32(isum1) as f32
                        + sc2 as f32 * vaddvq_s32(isum2) as f32);
                q_off += 32;
                base += 64;
                is += 2;
            }
        }
        acc
    }

    /// NEON Q5_K × Q8_K int-dot (widening path).
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q5_k_q8_neon(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q5_K_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q5_K_BLOCK_BYTES, act.n_blocks());
        let low_mask = vdupq_n_u8(0x0F);
        let sixteen = vdupq_n_u8(16);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q5_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            let qh = block.as_ptr().add(16);
            let qs = &block[48..176];
            let da = act.d[b];
            let q8 = act.q.as_ptr().add(b * Q5_K_BLOCK_ELEMS);
            let bsums = &act.bsums[b * 16..(b + 1) * 16];

            let mut sum_min = 0i32;
            for i in 0..8 {
                let (_, m) = q4_k_scale_min(i, &scales);
                sum_min += m as i32 * (bsums[2 * i] as i32 + bsums[2 * i + 1] as i32);
            }
            acc -= dmin * da * sum_min as f32;

            let mut q_off = 0usize;
            let mut base = 0usize;
            let mut is = 0usize;
            let (mut u1, mut u2) = (1u8, 2u8);
            for _ in 0..4 {
                let (sc1, _) = q4_k_scale_min(is, &scales);
                let (sc2, _) = q4_k_scale_min(is + 1, &scales);
                let mut isum1 = vdupq_n_s32(0);
                let mut isum2 = vdupq_n_s32(0);
                let u1_vec = vdupq_n_u8(u1);
                let u2_vec = vdupq_n_u8(u2);
                for g in 0..2 {
                    let packed = vld1q_u8(qs.as_ptr().add(q_off + g * 16));
                    let qh16 = vld1q_u8(qh.add(g * 16));
                    let lo_nib = vandq_u8(packed, low_mask);
                    let hi_nib = vshrq_n_u8(packed, 4);
                    let hi_bit1 = vandq_u8(vtstq_u8(qh16, u1_vec), sixteen);
                    let hi_bit2 = vandq_u8(vtstq_u8(qh16, u2_vec), sixteen);
                    let lo = vreinterpretq_s8_u8(vorrq_u8(lo_nib, hi_bit1));
                    let hi = vreinterpretq_s8_u8(vorrq_u8(hi_nib, hi_bit2));
                    let a0 = vld1q_s8(q8.add(base + g * 16));
                    let a1 = vld1q_s8(q8.add(base + 32 + g * 16));
                    isum1 = neon_i8_dot_widen(isum1, lo, a0);
                    isum2 = neon_i8_dot_widen(isum2, hi, a1);
                }
                acc += d
                    * da
                    * (sc1 as f32 * vaddvq_s32(isum1) as f32
                        + sc2 as f32 * vaddvq_s32(isum2) as f32);
                q_off += 32;
                base += 64;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        acc
    }

    /// NEON Q5_K × Q8_K with SDOT (llama.cpp `ggml_vec_dot_q5_K_q8_K` ARM).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q5_k_q8_neon_sdot(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q5_K_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q5_K_BLOCK_BYTES, act.n_blocks());
        let low_mask = vdupq_n_u8(0x0F);
        let sixteen = vdupq_n_u8(16);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q5_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            let qh = block.as_ptr().add(16);
            let qs = &block[48..176];
            let da = act.d[b];
            let q8 = act.q.as_ptr().add(b * Q5_K_BLOCK_ELEMS);
            let bsums = &act.bsums[b * 16..(b + 1) * 16];

            let mut sum_min = 0i32;
            for i in 0..8 {
                let (_, m) = q4_k_scale_min(i, &scales);
                sum_min += m as i32 * (bsums[2 * i] as i32 + bsums[2 * i + 1] as i32);
            }
            acc -= dmin * da * sum_min as f32;

            let mut q_off = 0usize;
            let mut base = 0usize;
            let mut is = 0usize;
            let (mut u1, mut u2) = (1u8, 2u8);
            for _ in 0..4 {
                let (sc1, _) = q4_k_scale_min(is, &scales);
                let (sc2, _) = q4_k_scale_min(is + 1, &scales);
                let mut isum1 = vdupq_n_s32(0);
                let mut isum2 = vdupq_n_s32(0);
                let u1_vec = vdupq_n_u8(u1);
                let u2_vec = vdupq_n_u8(u2);
                for g in 0..2 {
                    let packed = vld1q_u8(qs.as_ptr().add(q_off + g * 16));
                    let qh16 = vld1q_u8(qh.add(g * 16));
                    let lo_nib = vandq_u8(packed, low_mask);
                    let hi_nib = vshrq_n_u8(packed, 4);
                    let hi_bit1 = vandq_u8(vtstq_u8(qh16, u1_vec), sixteen);
                    let hi_bit2 = vandq_u8(vtstq_u8(qh16, u2_vec), sixteen);
                    let lo = vreinterpretq_s8_u8(vorrq_u8(lo_nib, hi_bit1));
                    let hi = vreinterpretq_s8_u8(vorrq_u8(hi_nib, hi_bit2));
                    let a0 = vld1q_s8(q8.add(base + g * 16));
                    let a1 = vld1q_s8(q8.add(base + 32 + g * 16));
                    isum1 = neon_sdot(isum1, lo, a0);
                    isum2 = neon_sdot(isum2, hi, a1);
                }
                acc += d
                    * da
                    * (sc1 as f32 * vaddvq_s32(isum1) as f32
                        + sc2 as f32 * vaddvq_s32(isum2) as f32);
                q_off += 32;
                base += 64;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        acc
    }

    /// Q5_K row × up to [`Q5_K_GEMM_NC`] activations (weight blocks loaded once).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemm_q5_k_q8_neon_sdot(
        row_bytes: &[u8],
        acts: &[Q8KActivations],
        out: &mut [f32],
    ) {
        debug_assert_eq!(out.len(), acts.len());
        debug_assert!(acts.len() <= super::Q5_K_GEMM_NC);
        out.fill(0.0);
        if acts.is_empty() {
            return;
        }
        let low_mask = vdupq_n_u8(0x0F);
        let sixteen = vdupq_n_u8(16);
        let n = acts.len();
        for (b, block) in row_bytes
            .as_chunks::<Q5_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            let qh = block.as_ptr().add(16);
            let qs = &block[48..176];
            let mut mins = [0u8; 8];
            let mut sc_only = [0u8; 8];
            for i in 0..8 {
                let (s, m) = q4_k_scale_min(i, &scales);
                sc_only[i] = s;
                mins[i] = m;
            }
            for j in 0..n {
                let act = &acts[j];
                let da = act.d[b];
                let bsums = &act.bsums[b * 16..(b + 1) * 16];
                let mut sum_min = 0i32;
                for i in 0..8 {
                    sum_min += mins[i] as i32 * (bsums[2 * i] as i32 + bsums[2 * i + 1] as i32);
                }
                out[j] -= dmin * da * sum_min as f32;
            }
            let mut q_off = 0usize;
            let mut base = 0usize;
            let mut is = 0usize;
            let (mut u1, mut u2) = (1u8, 2u8);
            for _ in 0..4 {
                let sc1 = sc_only[is];
                let sc2 = sc_only[is + 1];
                let u1_vec = vdupq_n_u8(u1);
                let u2_vec = vdupq_n_u8(u2);
                // Decode weight quants once per 32-byte group.
                let mut lo_cols = [vreinterpretq_s8_u8(vdupq_n_u8(0)); 2];
                let mut hi_cols = [vreinterpretq_s8_u8(vdupq_n_u8(0)); 2];
                for g in 0..2 {
                    let packed = vld1q_u8(qs.as_ptr().add(q_off + g * 16));
                    let qh16 = vld1q_u8(qh.add(g * 16));
                    let lo_nib = vandq_u8(packed, low_mask);
                    let hi_nib = vshrq_n_u8(packed, 4);
                    let hi_bit1 = vandq_u8(vtstq_u8(qh16, u1_vec), sixteen);
                    let hi_bit2 = vandq_u8(vtstq_u8(qh16, u2_vec), sixteen);
                    lo_cols[g] = vreinterpretq_s8_u8(vorrq_u8(lo_nib, hi_bit1));
                    hi_cols[g] = vreinterpretq_s8_u8(vorrq_u8(hi_nib, hi_bit2));
                }
                for j in 0..n {
                    let q8 = acts[j].q.as_ptr().add(b * Q5_K_BLOCK_ELEMS);
                    let da = acts[j].d[b];
                    let mut isum1 = vdupq_n_s32(0);
                    let mut isum2 = vdupq_n_s32(0);
                    for g in 0..2 {
                        let a0 = vld1q_s8(q8.add(base + g * 16));
                        let a1 = vld1q_s8(q8.add(base + 32 + g * 16));
                        isum1 = neon_sdot(isum1, lo_cols[g], a0);
                        isum2 = neon_sdot(isum2, hi_cols[g], a1);
                    }
                    out[j] += d
                        * da
                        * (sc1 as f32 * vaddvq_s32(isum1) as f32
                            + sc2 as f32 * vaddvq_s32(isum2) as f32);
                }
                q_off += 32;
                base += 64;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    /// Q6_K row × up to [`Q6_K_GEMM_NC`] activations — decode ql/qh once
    /// per sub-block, reuse across acts (Phi-4 `ffn_down` Q6_K).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn gemm_q6_k_q8_neon_sdot(
        row_bytes: &[u8],
        acts: &[Q8KActivations],
        out: &mut [f32],
    ) {
        debug_assert_eq!(out.len(), acts.len());
        debug_assert!(acts.len() <= super::Q6_K_GEMM_NC);
        out.fill(0.0);
        let n = acts.len();
        if n == 0 {
            return;
        }
        let m4b = vdupq_n_u8(0x0F);
        let mone = vdupq_n_u8(3);
        for (b, block) in row_bytes
            .as_chunks::<Q6_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d_all = f16::from_le_bytes([block[208], block[209]]).to_f32();
            let ql = block.as_ptr();
            let qh = block.as_ptr().add(128);
            let scale = block.as_ptr().add(192) as *const i8;
            let scales = vld1q_s8(scale);
            let q6scales0 = vmovl_s8(vget_low_s8(scales));
            let q6scales1 = vmovl_s8(vget_high_s8(scales));

            let mut isum_mins = [0i32; 4];
            let mut isums = [0i32; 4];
            for j in 0..n {
                let bsums = acts[j].bsums.as_ptr().add(b * 16);
                let q8sums0 = vld1q_s16(bsums);
                let q8sums1 = vld1q_s16(bsums.add(8));
                let prod = vaddq_s32(
                    vaddq_s32(
                        vmull_s16(vget_low_s16(q8sums0), vget_low_s16(q6scales0)),
                        vmull_s16(vget_high_s16(q8sums0), vget_high_s16(q6scales0)),
                    ),
                    vaddq_s32(
                        vmull_s16(vget_low_s16(q8sums1), vget_low_s16(q6scales1)),
                        vmull_s16(vget_high_s16(q8sums1), vget_high_s16(q6scales1)),
                    ),
                );
                isum_mins[j] = vaddvq_s32(prod);
            }

            for half in 0..2usize {
                let q6 = ql.add(half * 64);
                let qhp = qh.add(half * 32);
                let sc = scale.add(half * 8);
                let act_off = half * 128;

                let qh0 = vld1q_u8(qhp);
                let qh1 = vld1q_u8(qhp.add(16));
                let q6_0 = vld1q_u8(q6);
                let q6_1 = vld1q_u8(q6.add(16));
                let q6_2 = vld1q_u8(q6.add(32));
                let q6_3 = vld1q_u8(q6.add(48));

                let h0 = vshlq_n_u8(vandq_u8(mone, qh0), 4);
                let h1 = vshlq_n_u8(vandq_u8(mone, qh1), 4);
                let mut shifted = vshrq_n_u8(qh0, 2);
                let h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                shifted = vshrq_n_u8(qh1, 2);
                let h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let wb0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6_0, m4b), h0));
                let wb1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6_1, m4b), h1));
                let wb2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6_2, m4b), h2));
                let wb3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6_3, m4b), h3));
                let sc0 = *sc.add(0) as i32;
                let sc1 = *sc.add(1) as i32;
                let sc2 = *sc.add(2) as i32;
                let sc3 = *sc.add(3) as i32;
                let z = vdupq_n_s32(0);
                for j in 0..n {
                    let q8p = acts[j].q.as_ptr().add(b * Q6_K_BLOCK_ELEMS + act_off);
                    isums[j] += vaddvq_s32(neon_sdot(z, wb0, vld1q_s8(q8p))) * sc0
                        + vaddvq_s32(neon_sdot(z, wb1, vld1q_s8(q8p.add(16)))) * sc1
                        + vaddvq_s32(neon_sdot(z, wb2, vld1q_s8(q8p.add(32)))) * sc2
                        + vaddvq_s32(neon_sdot(z, wb3, vld1q_s8(q8p.add(48)))) * sc3;
                }

                shifted = vshrq_n_u8(qh0, 4);
                let h0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                shifted = vshrq_n_u8(qh1, 4);
                let h1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                shifted = vshrq_n_u8(qh0, 6);
                let h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                shifted = vshrq_n_u8(qh1, 6);
                let h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let wb0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_0, 4), h0));
                let wb1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_1, 4), h1));
                let wb2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_2, 4), h2));
                let wb3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_3, 4), h3));
                let sc0 = *sc.add(4) as i32;
                let sc1 = *sc.add(5) as i32;
                let sc2 = *sc.add(6) as i32;
                let sc3 = *sc.add(7) as i32;
                for j in 0..n {
                    let q8p = acts[j].q.as_ptr().add(b * Q6_K_BLOCK_ELEMS + act_off + 64);
                    isums[j] += vaddvq_s32(neon_sdot(z, wb0, vld1q_s8(q8p))) * sc0
                        + vaddvq_s32(neon_sdot(z, wb1, vld1q_s8(q8p.add(16)))) * sc1
                        + vaddvq_s32(neon_sdot(z, wb2, vld1q_s8(q8p.add(32)))) * sc2
                        + vaddvq_s32(neon_sdot(z, wb3, vld1q_s8(q8p.add(48)))) * sc3;
                }
            }
            for j in 0..n {
                out[j] += d_all * acts[j].d[b] * (isums[j] - 32 * isum_mins[j]) as f32;
            }
        }
    }

    /// NEON Q6_K × Q8_K with SDOT (llama.cpp `ggml_vec_dot_q6_K_q8_K` ARM).
    /// Quants are assembled as unsigned 0..63 then corrected with
    /// `isum - 32 * sum(scale * bsums)` — same as ggml's NEON path.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q6_k_q8_neon_sdot(row_bytes: &[u8], act: &Q8KActivations) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q6_K_BLOCK_BYTES, 0);
        debug_assert_eq!(row_bytes.len() / Q6_K_BLOCK_BYTES, act.n_blocks());
        let m4b = vdupq_n_u8(0x0F);
        let mone = vdupq_n_u8(3);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q6_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d_all = f16::from_le_bytes([block[208], block[209]]).to_f32();
            let da = act.d[b];
            let ql = block.as_ptr();
            let qh = block.as_ptr().add(128);
            let scale = block.as_ptr().add(192) as *const i8;
            let q8 = act.q.as_ptr().add(b * Q6_K_BLOCK_ELEMS);
            let bsums = act.bsums.as_ptr().add(b * 16);

            let scales = vld1q_s8(scale);
            let q6scales0 = vmovl_s8(vget_low_s8(scales));
            let q6scales1 = vmovl_s8(vget_high_s8(scales));
            let q8sums0 = vld1q_s16(bsums);
            let q8sums1 = vld1q_s16(bsums.add(8));
            let prod = vaddq_s32(
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums0), vget_low_s16(q6scales0)),
                    vmull_s16(vget_high_s16(q8sums0), vget_high_s16(q6scales0)),
                ),
                vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums1), vget_low_s16(q6scales1)),
                    vmull_s16(vget_high_s16(q8sums1), vget_high_s16(q6scales1)),
                ),
            );
            let isum_mins = vaddvq_s32(prod);
            let mut isum = 0i32;
            let mut q6 = ql;
            let mut qhp = qh;
            let mut q8p = q8;
            let mut sc = scale;
            for _ in 0..2 {
                let qh0 = vld1q_u8(qhp);
                let qh1 = vld1q_u8(qhp.add(16));
                qhp = qhp.add(32);
                let q6_0 = vld1q_u8(q6);
                let q6_1 = vld1q_u8(q6.add(16));
                let q6_2 = vld1q_u8(q6.add(32));
                let q6_3 = vld1q_u8(q6.add(48));
                q6 = q6.add(64);
                let q8_0 = vld1q_s8(q8p);
                let q8_1 = vld1q_s8(q8p.add(16));
                let q8_2 = vld1q_s8(q8p.add(32));
                let q8_3 = vld1q_s8(q8p.add(48));
                q8p = q8p.add(64);

                let h0 = vshlq_n_u8(vandq_u8(mone, qh0), 4);
                let h1 = vshlq_n_u8(vandq_u8(mone, qh1), 4);
                let mut shifted = vshrq_n_u8(qh0, 2);
                let h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                shifted = vshrq_n_u8(qh1, 2);
                let h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);

                let b0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6_0, m4b), h0));
                let b1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6_1, m4b), h1));
                let b2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6_2, m4b), h2));
                let b3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6_3, m4b), h3));
                let z = vdupq_n_s32(0);
                isum += vaddvq_s32(neon_sdot(z, b0, q8_0)) * (*sc.add(0) as i32)
                    + vaddvq_s32(neon_sdot(z, b1, q8_1)) * (*sc.add(1) as i32)
                    + vaddvq_s32(neon_sdot(z, b2, q8_2)) * (*sc.add(2) as i32)
                    + vaddvq_s32(neon_sdot(z, b3, q8_3)) * (*sc.add(3) as i32);
                sc = sc.add(4);

                let q8_0 = vld1q_s8(q8p);
                let q8_1 = vld1q_s8(q8p.add(16));
                let q8_2 = vld1q_s8(q8p.add(32));
                let q8_3 = vld1q_s8(q8p.add(48));
                q8p = q8p.add(64);
                shifted = vshrq_n_u8(qh0, 4);
                let h0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                shifted = vshrq_n_u8(qh1, 4);
                let h1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                shifted = vshrq_n_u8(qh0, 6);
                let h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                shifted = vshrq_n_u8(qh1, 6);
                let h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
                let b0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_0, 4), h0));
                let b1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_1, 4), h1));
                let b2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_2, 4), h2));
                let b3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_3, 4), h3));
                isum += vaddvq_s32(neon_sdot(z, b0, q8_0)) * (*sc.add(0) as i32)
                    + vaddvq_s32(neon_sdot(z, b1, q8_1)) * (*sc.add(1) as i32)
                    + vaddvq_s32(neon_sdot(z, b2, q8_2)) * (*sc.add(2) as i32)
                    + vaddvq_s32(neon_sdot(z, b3, q8_3)) * (*sc.add(3) as i32);
                sc = sc.add(4);
            }
            acc += d_all * da * (isum - 32 * isum_mins) as f32;
        }
        acc
    }

    /// NEON fused Q4_0 dot product. Each block's 16 nibble-packed bytes
    /// are loaded once, split into low/high nibbles with
    /// `vandq_u8`/`vshrq_n_u8` (a per-byte shift, simpler than AVX2's
    /// 16-bit-lane-shift-then-mask trick since NEON shifts natively at
    /// byte granularity), then each 16-lane nibble group goes through
    /// the same unsigned-widen -> signed-bias-subtract -> widen-to-i32
    /// -> f32 -> FMA sequence as Q8_0 above. Safety: same contract as
    /// `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q4_0_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_0_BLOCK_BYTES, 0);
        let bias = vdupq_n_s16(8);
        let low_mask = vdupq_n_u8(0x0F);

        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q4_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q4_0_BLOCK_ELEMS;
            let nibbles = vld1q_u8(block.as_ptr().add(2));

            let lo_nibbles = vandq_u8(nibbles, low_mask); // elements 0..16
            let hi_nibbles = vshrq_n_u8(nibbles, 4); // elements 16..32

            let mut block_acc = vdupq_n_f32(0.0);
            for (group_idx, nib_u8) in [lo_nibbles, hi_nibbles].into_iter().enumerate() {
                let lo16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(nib_u8))), bias);
                let hi16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(nib_u8))), bias);
                for (half_idx, half16) in [lo16, hi16].into_iter().enumerate() {
                    let lo32 = vmovl_s16(vget_low_s16(half16));
                    let hi32 = vmovl_s16(vget_high_s16(half16));
                    let f_lo = vcvtq_f32_s32(lo32);
                    let f_hi = vcvtq_f32_s32(hi32);
                    let elem_base = base + group_idx * 16 + half_idx * 8;
                    let x_lo = vld1q_f32(x.as_ptr().add(elem_base));
                    let x_hi = vld1q_f32(x.as_ptr().add(elem_base + 4));
                    block_acc = vfmaq_f32(block_acc, f_lo, x_lo);
                    block_acc = vfmaq_f32(block_acc, f_hi, x_hi);
                }
            }
            acc += vaddvq_f32(block_acc) * scale;
        }
        acc
    }

    /// Widens 16 unsigned nibble values (0..=15 or 0..=31 once a 5th
    /// bit has been OR'd in for Q5_K) into four `float32x4_t` quads, in
    /// lane order -- the shared u8 -> u16 -> u32 -> f32 widening step
    /// every K-quant NEON kernel below needs, factored out once rather
    /// than repeated per format.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn widen_u8x16_to_f32_quads(
        v: uint8x16_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
        let u16_lo = vmovl_u8(vget_low_u8(v)); // lanes 0..8
        let u16_hi = vmovl_u8(vget_high_u8(v)); // lanes 8..16
        (
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_lo))), // lanes 0..4
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_lo))), // lanes 4..8
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_hi))), // lanes 8..12
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_hi))), // lanes 12..16
        )
    }

    /// Dequantizes 16 nibble-derived f32 values (`quads`, in element
    /// order) as `d * q - min` and fused-multiply-accumulates each
    /// against the matching 16 activations starting at `x[x_base..]`,
    /// into `acc`. Shared by Q4_K's and Q5_K's NEON kernels, which both
    /// use this exact affine (scale, min) dequant form per 32-element
    /// sub-block.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn fma_affine16(
        quads: (float32x4_t, float32x4_t, float32x4_t, float32x4_t),
        d: f32,
        min_vec: float32x4_t,
        x: &[f32],
        x_base: usize,
        mut acc: float32x4_t,
    ) -> float32x4_t {
        let (q0, q1, q2, q3) = quads;
        let mut i = 0usize;
        for q in [q0, q1, q2, q3] {
            let w = vsubq_f32(vmulq_n_f32(q, d), min_vec);
            let xv = vld1q_f32(x.as_ptr().add(x_base + i));
            acc = vfmaq_f32(acc, w, xv);
            i += 4;
        }
        acc
    }

    /// NEON fused Q4_K dot product. Mirrors `dot_q4_0_f32_neon`'s
    /// nibble-splitting structure (low/high nibble of each byte are two
    /// independent output elements), scaled up from Q4_0's 16
    /// bytes/block to Q4_K's 32 bytes/sub-block, with the affine `d*q -
    /// min` transform (two independent (scale, min) pairs, one for the
    /// low-nibble half and one for the high-nibble half) instead of
    /// Q4_0's single symmetric `d*(q-8)`. Safety: same contract as
    /// `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q4_k_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_K_BLOCK_BYTES, 0);
        let low_mask = vdupq_n_u8(0x0F);
        let mut acc = 0f32;
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<Q4_K_BLOCK_BYTES>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            let qs = &block[16..144];

            // One vector accumulator per block — avoid a horizontal
            // reduce on every 32-element group (4× per super-block).
            let mut vec_acc = vdupq_n_f32(0.0);
            let mut is = 0usize;
            let mut q_off = 0usize;
            for _ in 0..4 {
                let (sc1, m1) = q4_k_scale_min(is, &scales);
                let (sc2, m2) = q4_k_scale_min(is + 1, &scales);
                let d1 = d * sc1 as f32;
                let min1_vec = vdupq_n_f32(dmin * m1 as f32);
                let d2 = d * sc2 as f32;
                let min2_vec = vdupq_n_f32(dmin * m2 as f32);

                for g in 0..2 {
                    let raw16 = vld1q_u8(qs.as_ptr().add(q_off + g * 16));
                    let lo_nib = vandq_u8(raw16, low_mask);
                    let hi_nib = vshrq_n_u8(raw16, 4);
                    vec_acc = fma_affine16(
                        widen_u8x16_to_f32_quads(lo_nib),
                        d1,
                        min1_vec,
                        x,
                        x_base + g * 16,
                        vec_acc,
                    );
                    vec_acc = fma_affine16(
                        widen_u8x16_to_f32_quads(hi_nib),
                        d2,
                        min2_vec,
                        x,
                        x_base + 32 + g * 16,
                        vec_acc,
                    );
                }
                q_off += 32;
                x_base += 64;
                is += 2;
            }
            acc += vaddvq_f32(vec_acc);
        }
        acc
    }

    /// NEON fused Q5_K dot product: identical structure to
    /// `dot_q4_k_f32_neon`, but before widening, each nibble gets a 5th
    /// bit OR'd in from the block's `qh` bitplane. The per-lane "is bit
    /// `u1`/`u2` set in this byte of `qh`" test uses
    /// `vtstq_u8`(bitwise-AND-then-nonzero-test, giving an all-ones or
    /// all-zeros mask per lane) `AND`ed with a lane of `16` -- the
    /// standard NEON idiom for a per-lane conditional add when the
    /// condition is itself a bitwise test. Safety: same contract as
    /// `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q5_k_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q5_K_BLOCK_BYTES, 0);
        let low_mask = vdupq_n_u8(0x0F);
        let sixteen = vdupq_n_u8(16);
        let mut acc = 0f32;
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<Q5_K_BLOCK_BYTES>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            let qh = &block[16..48];
            let qs = &block[48..176];

            let mut is = 0usize;
            let (mut u1, mut u2) = (1u8, 2u8);
            for oi in 0..4 {
                let (sc1, m1) = q4_k_scale_min(is, &scales);
                let (sc2, m2) = q4_k_scale_min(is + 1, &scales);
                let d1 = d * sc1 as f32;
                let min1_vec = vdupq_n_f32(dmin * m1 as f32);
                let d2 = d * sc2 as f32;
                let min2_vec = vdupq_n_f32(dmin * m2 as f32);
                let ql = &qs[oi * 32..oi * 32 + 32];
                let u1_vec = vdupq_n_u8(u1);
                let u2_vec = vdupq_n_u8(u2);

                let mut lo_acc = vdupq_n_f32(0.0);
                let mut hi_acc = vdupq_n_f32(0.0);
                for g in 0..2 {
                    let raw16 = vld1q_u8(ql.as_ptr().add(g * 16));
                    let qh16 = vld1q_u8(qh.as_ptr().add(g * 16));

                    let lo_nib = vandq_u8(raw16, low_mask);
                    let hi_nib = vshrq_n_u8(raw16, 4);
                    let hi_bit1 = vandq_u8(vtstq_u8(qh16, u1_vec), sixteen);
                    let hi_bit2 = vandq_u8(vtstq_u8(qh16, u2_vec), sixteen);

                    lo_acc = fma_affine16(
                        widen_u8x16_to_f32_quads(vorrq_u8(lo_nib, hi_bit1)),
                        d1,
                        min1_vec,
                        x,
                        x_base + g * 16,
                        lo_acc,
                    );
                    hi_acc = fma_affine16(
                        widen_u8x16_to_f32_quads(vorrq_u8(hi_nib, hi_bit2)),
                        d2,
                        min2_vec,
                        x,
                        x_base + 32 + g * 16,
                        hi_acc,
                    );
                }
                acc += vaddvq_f32(lo_acc) + vaddvq_f32(hi_acc);
                x_base += 64;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        acc
    }

    /// Widens 16 raw 6-bit values (0..=63, already `nibble | (2bit <<
    /// 4)`-assembled) into four `float32x4_t` quads, centered by `-32`
    /// (Q6_K's fixed bias -- unlike Q4_K/Q5_K's per-sub-block `min`,
    /// this is the same constant for every element). The 0..=63 range
    /// fits safely in an `i16` after a bit-cast from `u16`, so
    /// subtracting the bias in the signed 16-bit domain before the
    /// final widen-to-i32-then-f32 step is exact.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn widen_u8x16_centered_to_f32_quads(
        v: uint8x16_t,
        bias16: int16x8_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
        let s16_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))), bias16);
        let s16_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))), bias16);
        (
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_lo))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_lo))),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_hi))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_hi))),
        )
    }

    /// Multiplies 16 f32 values (`quads`) by the single shared scalar
    /// `scale` and fused-multiply-accumulates each against the matching
    /// 16 activations starting at `x[x_base..]`. Q6_K's dequant is pure
    /// `scale * centered_value` (no per-element `min` subtraction, only
    /// a fixed bias already folded in by the caller), unlike Q4_K/Q5_K's
    /// `fma_affine16`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn fma_scaled16(
        quads: (float32x4_t, float32x4_t, float32x4_t, float32x4_t),
        scale: f32,
        x: &[f32],
        x_base: usize,
        mut acc: float32x4_t,
    ) -> float32x4_t {
        let (q0, q1, q2, q3) = quads;
        let mut i = 0usize;
        for q in [q0, q1, q2, q3] {
            let xv = vld1q_f32(x.as_ptr().add(x_base + i));
            acc = vfmaq_f32(acc, vmulq_n_f32(q, scale), xv);
            i += 4;
        }
        acc
    }

    /// One (q1/q2/q3/q4 in the scalar reference) 32-element group
    /// within a Q6_K half-block: 16 lanes at a time (`sub` selects
    /// which 16), the 6-bit value is `(ql nibble) | (qh 2-bit field <<
    /// 4)`, scaled by `sc[sc_base + sub]` (elements 0..16 of the group
    /// use one sub-block scale, 16..32 use the next) and `d`. The `qh`
    /// 2-bit field's shift amount is a NEON shift-by-immediate, which
    /// Rust's intrinsics require as a compile-time constant -- hence
    /// this being a `const QH_SHIFT` generic, monomorphized once per
    /// group (0/2/4/6) at its four call sites below, rather than a
    /// runtime loop variable. Safety: same contract as
    /// `dot_q8_0_f32_neon`.
    #[inline]
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn q6_k_group<const QH_SHIFT: i32, const HI_NIBBLE: bool>(
        ql: &[u8],
        ql_off: usize,
        qh: &[u8],
        sc: &[u8],
        sc_base: usize,
        d: f32,
        x: &[f32],
        x_base: usize,
        out_off: usize,
        low_mask: uint8x16_t,
        two_bit_mask: uint8x16_t,
        bias16: int16x8_t,
    ) -> f32 {
        let mut acc = 0f32;
        for sub in 0..2usize {
            let byte_off = sub * 16;
            let ql_raw = vld1q_u8(ql.as_ptr().add(ql_off + byte_off));
            let qh_raw = vld1q_u8(qh.as_ptr().add(byte_off));

            let nib = if HI_NIBBLE {
                vshrq_n_u8::<4>(ql_raw)
            } else {
                vandq_u8(ql_raw, low_mask)
            };
            // QH_SHIFT is only ever 2, 4, or 6 here (q1's shift-0 case
            // is handled separately by `q6_k_group_q1` below): NEON's
            // shift-by-immediate intrinsics require their N in 1..=8 as
            // a genuine compile-time constant, and that assertion is
            // checked at monomorphization time even inside a dead
            // branch, so a runtime `if QH_SHIFT == 0` guard here would
            // still fail to compile for the QH_SHIFT=0 instantiation.
            let qh_field = vandq_u8(vshrq_n_u8::<QH_SHIFT>(qh_raw), two_bit_mask);
            let raw6 = vorrq_u8(nib, vshlq_n_u8::<4>(qh_field));

            let scale = d * (sc[sc_base + sub] as i8) as f32;
            let quads = widen_u8x16_centered_to_f32_quads(raw6, bias16);
            let acc_vec = fma_scaled16(
                quads,
                scale,
                x,
                x_base + out_off + sub * 16,
                vdupq_n_f32(0.0),
            );
            acc += vaddvq_f32(acc_vec);
        }
        acc
    }

    /// Same as `q6_k_group`, specialized for q1 (`QH_SHIFT` would be 0,
    /// which is out of NEON's valid shift-immediate range) -- the `qh`
    /// 2-bit field is already at bit position 0, so no shift is needed
    /// before masking. Always low-nibble (`HI_NIBBLE = false` in
    /// `q6_k_group`'s terms), matching the scalar reference's `q1`.
    #[inline]
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn q6_k_group_q1(
        ql: &[u8],
        qh: &[u8],
        sc: &[u8],
        d: f32,
        x: &[f32],
        x_base: usize,
        low_mask: uint8x16_t,
        two_bit_mask: uint8x16_t,
        bias16: int16x8_t,
    ) -> f32 {
        let mut acc = 0f32;
        // `sub` drives both the byte offset into `ql`/`qh` and the
        // index into `sc` -- not just the latter, so clippy's
        // iterator-based rewrite doesn't fit.
        #[allow(clippy::needless_range_loop)]
        for sub in 0..2usize {
            let byte_off = sub * 16;
            let ql_raw = vld1q_u8(ql.as_ptr().add(byte_off));
            let qh_raw = vld1q_u8(qh.as_ptr().add(byte_off));

            let nib = vandq_u8(ql_raw, low_mask);
            let qh_field = vandq_u8(qh_raw, two_bit_mask);
            let raw6 = vorrq_u8(nib, vshlq_n_u8::<4>(qh_field));

            let scale = d * (sc[sub] as i8) as f32;
            let quads = widen_u8x16_centered_to_f32_quads(raw6, bias16);
            let acc_vec = fma_scaled16(quads, scale, x, x_base + sub * 16, vdupq_n_f32(0.0));
            acc += vaddvq_f32(acc_vec);
        }
        acc
    }

    /// NEON fused Q6_K dot product: dispatches each of the four
    /// 32-element groups per half-block (`q1..q4` in the scalar
    /// reference) to `q6_k_group`, monomorphized once per group's
    /// (compile-time-constant) `qh` shift amount and nibble half.
    /// Safety: same contract as `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q6_k_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q6_K_BLOCK_BYTES, 0);
        debug_assert_eq!(
            row_bytes.len() / Q6_K_BLOCK_BYTES * Q6_K_BLOCK_ELEMS,
            x.len()
        );
        let low_mask = vdupq_n_u8(0x0F);
        let two_bit_mask = vdupq_n_u8(0x03);
        let bias16 = vdupq_n_s16(32);

        let mut acc = 0f32;
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<Q6_K_BLOCK_BYTES>().0 {
            let ql_full = &block[0..128];
            let qh_full = &block[128..192];
            let sc_full = &block[192..208];
            let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

            for half in 0..2 {
                let ql = &ql_full[half * 64..half * 64 + 64];
                let qh = &qh_full[half * 32..half * 32 + 32];
                let sc = &sc_full[half * 8..half * 8 + 8];
                let half_base = x_base + half * 128;

                // q1: ql[0..32] low nibble, no qh shift needed, out 0, sc[0..2]
                acc += q6_k_group_q1(ql, qh, sc, d, x, half_base, low_mask, two_bit_mask, bias16);
                // q2: ql[32..64] low nibble, qh shift 2, out 32, sc[2..4]
                acc += q6_k_group::<2, false>(
                    ql,
                    32,
                    qh,
                    sc,
                    2,
                    d,
                    x,
                    half_base,
                    32,
                    low_mask,
                    two_bit_mask,
                    bias16,
                );
                // q3: ql[0..32] high nibble, qh shift 4, out 64, sc[4..6]
                acc += q6_k_group::<4, true>(
                    ql,
                    0,
                    qh,
                    sc,
                    4,
                    d,
                    x,
                    half_base,
                    64,
                    low_mask,
                    two_bit_mask,
                    bias16,
                );
                // q4: ql[32..64] high nibble, qh shift 6, out 96, sc[6..8]
                acc += q6_k_group::<6, true>(
                    ql,
                    32,
                    qh,
                    sc,
                    6,
                    d,
                    x,
                    half_base,
                    96,
                    low_mask,
                    two_bit_mask,
                    bias16,
                );
            }
            x_base += Q6_K_BLOCK_ELEMS;
        }
        acc
    }

    /// Decodes 16 real E2M1 codebook values (one nibble byte per lane,
    /// each 0..=15, in `nib`) into four `float32x4_t` quads --
    /// arithmetically, not via a 16-entry float lookup table. Real
    /// E2M1 bit layout: bit3=sign, bits2:1=exponent `e` (0..3),
    /// bit0=mantissa `m` (0 or 1). Derivation (verified by hand against
    /// every real `KVALUES_MXFP4` entry): for `e=0`, `magnitude = 0.5*m`;
    /// for `e>=1`, `magnitude = 2^(e-1) * (1 + 0.5*m)`. Both cases are one
    /// formula, `magnitude = pow2(e) * (bias(e) + 0.5*m)`, where
    /// `pow2(e) = [1,1,2,4][e]` and `bias(e) = [0,1,1,1][e]` -- looked up
    /// via `vqtbl1q_u8` (a real 16-entry byte-table-lookup instruction;
    /// `e` is always in 0..3, so this is always an exact, in-range
    /// lookup, never the "index >=16 -> zero" out-of-range case). Sign
    /// is folded in as a multiplier (`1.0 - 0.25*sign_bit`, where
    /// `sign_bit` is 0 or 8) to avoid a branch/select. Cross-validated
    /// against the scalar `KVALUES_MXFP4` table across every real
    /// nibble value (see this module's tests).
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn mxfp4_nibbles_to_f32_quads(
        nib: uint8x16_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
        let sign_bit = vandq_u8(nib, vdupq_n_u8(0x8));
        let e = vandq_u8(vshrq_n_u8(nib, 1), vdupq_n_u8(0x3));
        let m = vandq_u8(nib, vdupq_n_u8(0x1));

        let pow2_table: [u8; 16] = [1, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let bias_table: [u8; 16] = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let pow2_u8 = vqtbl1q_u8(vld1q_u8(pow2_table.as_ptr()), e);
        let bias_u8 = vqtbl1q_u8(vld1q_u8(bias_table.as_ptr()), e);

        let (p0, p1, p2, p3) = widen_u8x16_to_f32_quads(pow2_u8);
        let (b0, b1, b2, b3) = widen_u8x16_to_f32_quads(bias_u8);
        let (m0, m1, m2, m3) = widen_u8x16_to_f32_quads(m);
        let (s0, s1, s2, s3) = widen_u8x16_to_f32_quads(sign_bit);

        let half = vdupq_n_f32(0.5);
        let quarter = vdupq_n_f32(0.25);
        let one = vdupq_n_f32(1.0);

        let decode = |p: float32x4_t, b: float32x4_t, m: float32x4_t, s: float32x4_t| {
            let magnitude = vmulq_f32(p, vfmaq_f32(b, m, half)); // p * (b + 0.5*m)
            let sign_mul = vfmsq_f32(one, s, quarter); // 1.0 - 0.25*s
            vmulq_f32(magnitude, sign_mul)
        };

        (
            decode(p0, b0, m0, s0),
            decode(p1, b1, m1, s1),
            decode(p2, b2, m2, s2),
            decode(p3, b3, m3, s3),
        )
    }

    /// NEON fused MXFP4 dequant+dot -- same real math as
    /// `dot_mxfp4_row_f32_scalar` (real E2M1 codebook + E8M0 scale),
    /// decoded via `mxfp4_nibbles_to_f32_quads` instead of the scalar
    /// path's 16-entry `KVALUES_MXFP4` table lookup. Cross-validated
    /// against the scalar reference across many packed-byte patterns
    /// (see this module's tests) -- verified directly on real aarch64
    /// hardware (Apple M2 Pro), matching the project's established
    /// verify-on-real-hardware discipline for every other NEON kernel
    /// here.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_mxfp4_row_f32_neon(packed: &[u8], scales: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(packed.len(), scales.len() * (MXFP4_GROUP_SIZE / 2));
        let low_mask = vdupq_n_u8(0x0F);
        let mut acc = 0f32;
        let mut x_base = 0usize;
        for (g, &e_byte) in scales.iter().enumerate() {
            let d = e8m0_scale(e_byte);
            let group = &packed[g * 16..(g + 1) * 16];
            let bytes = vld1q_u8(group.as_ptr());
            let lo_nib = vandq_u8(bytes, low_mask);
            let hi_nib = vshrq_n_u8(bytes, 4);

            let mut block_acc = vdupq_n_f32(0.0);
            for (half_idx, nib) in [lo_nib, hi_nib].into_iter().enumerate() {
                let (v0, v1, v2, v3) = mxfp4_nibbles_to_f32_quads(nib);
                let elem_base = x_base + half_idx * 16;
                for (i, v) in [v0, v1, v2, v3].into_iter().enumerate() {
                    let xv = vld1q_f32(x.as_ptr().add(elem_base + i * 4));
                    block_acc = vfmaq_f32(block_acc, v, xv);
                }
            }
            acc += vaddvq_f32(block_acc) * d;
            x_base += MXFP4_GROUP_SIZE;
        }
        acc
    }

    /// NEON fused Q8_1 dot product. Mathematically identical to
    /// `dot_q8_0_f32_neon` (`y = q*d`) -- see the AVX2 sibling's doc
    /// comment for why. Safety: same contract as `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q8_1_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q8_1_BLOCK_BYTES, 0);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q8_1_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let base = b * Q8_1_BLOCK_ELEMS;
            let qs = &block[4..36];

            let mut block_acc = vdupq_n_f32(0.0);
            for g in 0..2 {
                let raw16 = vld1q_s8(qs.as_ptr().add(g * 16) as *const i8);
                let lo16 = vmovl_s8(vget_low_s8(raw16));
                let hi16 = vmovl_s8(vget_high_s8(raw16));
                for (half_idx, half16) in [lo16, hi16].into_iter().enumerate() {
                    let lo32 = vmovl_s16(vget_low_s16(half16));
                    let hi32 = vmovl_s16(vget_high_s16(half16));
                    let f_lo = vcvtq_f32_s32(lo32);
                    let f_hi = vcvtq_f32_s32(hi32);
                    let elem_base = base + g * 16 + half_idx * 8;
                    let x_lo = vld1q_f32(x.as_ptr().add(elem_base));
                    let x_hi = vld1q_f32(x.as_ptr().add(elem_base + 4));
                    block_acc = vfmaq_f32(block_acc, f_lo, x_lo);
                    block_acc = vfmaq_f32(block_acc, f_hi, x_hi);
                }
            }
            acc += vaddvq_f32(block_acc) * scale;
        }
        acc
    }

    /// NEON fused Q4_1 dot product. Same nibble-splitting structure as
    /// `dot_q4_0_f32_neon`, but asymmetric (`y = nibble*d + m`, no bias
    /// subtraction): widens each nibble as unsigned (0..=15) then
    /// applies `q*d + m` directly instead of `(q-8)*d`. Safety: same
    /// contract as `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q4_1_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q4_1_BLOCK_BYTES, 0);
        let low_mask = vdupq_n_u8(0x0F);

        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q4_1_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let base = b * Q4_1_BLOCK_ELEMS;
            let nibbles = vld1q_u8(block.as_ptr().add(4));

            let lo_nibbles = vandq_u8(nibbles, low_mask); // elements 0..16
            let hi_nibbles = vshrq_n_u8(nibbles, 4); // elements 16..32

            let mut block_acc = vdupq_n_f32(0.0);
            for (group_idx, nib_u8) in [lo_nibbles, hi_nibbles].into_iter().enumerate() {
                let lo16 = vmovl_u8(vget_low_u8(nib_u8));
                let hi16 = vmovl_u8(vget_high_u8(nib_u8));
                for (half_idx, half16) in [lo16, hi16].into_iter().enumerate() {
                    let lo32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(half16)));
                    let hi32 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(half16)));
                    let elem_base = base + group_idx * 16 + half_idx * 8;
                    let x_lo = vld1q_f32(x.as_ptr().add(elem_base));
                    let x_hi = vld1q_f32(x.as_ptr().add(elem_base + 4));
                    let w_lo = vfmaq_n_f32(vdupq_n_f32(m), lo32, d);
                    let w_hi = vfmaq_n_f32(vdupq_n_f32(m), hi32, d);
                    block_acc = vfmaq_f32(block_acc, w_lo, x_lo);
                    block_acc = vfmaq_f32(block_acc, w_hi, x_hi);
                }
            }
            acc += vaddvq_f32(block_acc);
        }
        acc
    }

    /// NEON fused Q5_0 dot product. Same scalar-prep-then-vectorize
    /// approach as `simd_x86::dot_q5_0_f32_avx2` -- see that function's
    /// doc comment for why the 5th-bit extraction stays scalar while
    /// the 32-element multiply-accumulate is fully vectorized. Safety:
    /// same contract as `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q5_0_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q5_0_BLOCK_BYTES, 0);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q5_0_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
            let qs = &block[6..22];
            let base = b * Q5_0_BLOCK_ELEMS;

            let mut vals = [0i8; 32];
            for j in 0..16 {
                let (xh_0, xh_1) = q5_fifth_bits(qh, j);
                vals[j] = (((qs[j] & 0x0F) | xh_0) as i32 - 16) as i8;
                vals[j + 16] = (((qs[j] >> 4) | xh_1) as i32 - 16) as i8;
            }

            let mut block_acc = vdupq_n_f32(0.0);
            for g in 0..2 {
                let raw16 = vld1q_s8(vals.as_ptr().add(g * 16));
                let lo16 = vmovl_s8(vget_low_s8(raw16));
                let hi16 = vmovl_s8(vget_high_s8(raw16));
                for (half_idx, half16) in [lo16, hi16].into_iter().enumerate() {
                    let lo32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(half16)));
                    let hi32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(half16)));
                    let elem_base = base + g * 16 + half_idx * 8;
                    let x_lo = vld1q_f32(x.as_ptr().add(elem_base));
                    let x_hi = vld1q_f32(x.as_ptr().add(elem_base + 4));
                    block_acc = vfmaq_f32(block_acc, lo32, x_lo);
                    block_acc = vfmaq_f32(block_acc, hi32, x_hi);
                }
            }
            acc += vaddvq_f32(block_acc) * d;
        }
        acc
    }

    /// NEON fused Q5_1 dot product. Same 5th-bit scalar-prep approach
    /// as `dot_q5_0_f32_neon`, but asymmetric (`y = q*d + m`, no `-16`
    /// bias). Safety: same contract as `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q5_1_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q5_1_BLOCK_BYTES, 0);
        let mut acc = 0f32;
        for (b, block) in row_bytes
            .as_chunks::<Q5_1_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let qh = u32::from_le_bytes(block[4..8].try_into().unwrap());
            let qs = &block[8..24];
            let base = b * Q5_1_BLOCK_ELEMS;

            let mut vals = [0u8; 32];
            for j in 0..16 {
                let (xh_0, xh_1) = q5_fifth_bits(qh, j);
                vals[j] = (qs[j] & 0x0F) | xh_0;
                vals[j + 16] = (qs[j] >> 4) | xh_1;
            }

            let mut block_acc = vdupq_n_f32(0.0);
            for g in 0..2 {
                let raw16 = vld1q_u8(vals.as_ptr().add(g * 16));
                let lo16 = vmovl_u8(vget_low_u8(raw16));
                let hi16 = vmovl_u8(vget_high_u8(raw16));
                for (half_idx, half16) in [lo16, hi16].into_iter().enumerate() {
                    let lo32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(half16)));
                    let hi32 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(half16)));
                    let elem_base = base + g * 16 + half_idx * 8;
                    let x_lo = vld1q_f32(x.as_ptr().add(elem_base));
                    let x_hi = vld1q_f32(x.as_ptr().add(elem_base + 4));
                    let w_lo = vfmaq_n_f32(vdupq_n_f32(m), lo32, d);
                    let w_hi = vfmaq_n_f32(vdupq_n_f32(m), hi32, d);
                    block_acc = vfmaq_f32(block_acc, w_lo, x_lo);
                    block_acc = vfmaq_f32(block_acc, w_hi, x_hi);
                }
            }
            acc += vaddvq_f32(block_acc);
        }
        acc
    }

    /// NEON fused Q2_K dot product. Mirrors `dot_q4_k_f32_neon`'s
    /// sub-block loop with a 2-bit field (`(byte >> shift) & 3`) instead
    /// of a nibble, and a trivial one-byte-per-sub-block (scale, min)
    /// pairing. `shift` only ever takes 0/2/4/6, and NEON's
    /// `vshrq_n_u8` accepts a literal immediate the same way this file's
    /// `vshrq_n_u8::<4>`/`vshrq_n_u8(_, 4)` calls elsewhere do -- unrolled
    /// via a macro over the 4 literal shift values, same reasoning as
    /// the AVX2 sibling. Safety: same contract as `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q2_k_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q2_K_BLOCK_BYTES, 0);
        let two_bit_mask = vdupq_n_u8(3);
        let mut acc = 0f32;
        let mut x_base = 0usize;

        // NEON's `vshrq_n_u8` requires its immediate shift in 1..=8 (a
        // shift of 0 fails a compile-time static assertion) -- unlike
        // AVX2's `_mm_srli_epi16`, which allows 0. The `0` literal
        // pattern below is matched before the general `$shift:literal`
        // arm, so the shift=0 case never generates a call to
        // `vshrq_n_u8` at all, just the plain mask.
        macro_rules! shr2 {
            (0, $v:expr) => {
                vandq_u8($v, two_bit_mask)
            };
            ($shift:literal, $v:expr) => {
                vandq_u8(vshrq_n_u8($v, $shift), two_bit_mask)
            };
        }

        macro_rules! q2_k_sub_block {
            ($shift:tt, $q:expr, $scales:expr, $is:expr, $d:expr, $dmin:expr, $x:expr, $x_base:expr, $acc:expr) => {{
                let sc1 = $scales[$is];
                $is += 1;
                let dl1 = $d * (sc1 & 0x0F) as f32;
                let min1_vec = vdupq_n_f32($dmin * (sc1 >> 4) as f32);
                let sc2 = $scales[$is];
                $is += 1;
                let dl2 = $d * (sc2 & 0x0F) as f32;
                let min2_vec = vdupq_n_f32($dmin * (sc2 >> 4) as f32);

                let lo16 = vld1q_u8($q.as_ptr());
                let hi16 = vld1q_u8($q.as_ptr().add(16));
                let lo2 = shr2!($shift, lo16);
                let hi2 = shr2!($shift, hi16);

                let lo_acc = fma_affine16(
                    widen_u8x16_to_f32_quads(lo2),
                    dl1,
                    min1_vec,
                    $x,
                    $x_base,
                    vdupq_n_f32(0.0),
                );
                let hi_acc = fma_affine16(
                    widen_u8x16_to_f32_quads(hi2),
                    dl2,
                    min2_vec,
                    $x,
                    $x_base + 16,
                    vdupq_n_f32(0.0),
                );
                $acc += vaddvq_f32(lo_acc) + vaddvq_f32(hi_acc);
                $x_base += 32;
            }};
        }

        for block in row_bytes.as_chunks::<Q2_K_BLOCK_BYTES>().0 {
            let scales: &[u8; Q2_K_SCALE_BYTES] = block[0..16].try_into().unwrap();
            let qs = &block[16..80];
            let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
            let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();

            let mut is = 0usize;
            for n in 0..2 {
                let q = &qs[n * 32..n * 32 + 32];
                q2_k_sub_block!(0, q, scales, is, d, dmin, x, x_base, acc);
                q2_k_sub_block!(2, q, scales, is, d, dmin, x, x_base, acc);
                q2_k_sub_block!(4, q, scales, is, d, dmin, x, x_base, acc);
                q2_k_sub_block!(6, q, scales, is, d, dmin, x, x_base, acc);
            }
        }
        acc
    }

    /// NEON fused Q3_K dot product. Same 2-bit-field extraction as
    /// `dot_q2_k_f32_neon` (4 literal shift values), plus a 3rd bit
    /// tested from `hmask` via `vtstq_u8` (real bit-test intrinsic,
    /// all-ones per lane where the AND is nonzero) -- inverted with
    /// `vmvnq_u8` since Q3_K's bias is 4 when the bit is CLEAR, the
    /// opposite of Q5_K's "add 16 when set" convention. The 6-bit
    /// per-sub-block scale unpacking (`q3_k_unpack_scales`) runs once
    /// per block on the scalar side, same as the AVX2 sibling. Safety:
    /// same contract as `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q3_k_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % Q3_K_BLOCK_BYTES, 0);
        let two_bit_mask = vdupq_n_u8(3);
        let four = vdupq_n_u8(4);
        let mut acc = 0f32;
        let mut x_base = 0usize;

        // See `dot_q2_k_f32_neon`'s `shr2!` for why shift=0 needs its
        // own arm: NEON's `vshrq_n_u8` requires its immediate in 1..=8.
        macro_rules! shr2 {
            (0, $v:expr) => {
                vandq_u8($v, two_bit_mask)
            };
            ($shift:literal, $v:expr) => {
                vandq_u8(vshrq_n_u8($v, $shift), two_bit_mask)
            };
        }

        macro_rules! q3_k_sub_block {
            ($shift:tt, $q:expr, $hmask:expr, $m_vec:expr, $dl1:expr, $dl2:expr, $x:expr, $x_base:expr, $acc:expr) => {{
                let lo16 = vld1q_u8($q.as_ptr());
                let hi16 = vld1q_u8($q.as_ptr().add(16));
                let lo2 = shr2!($shift, lo16);
                let hi2 = shr2!($shift, hi16);

                let hmask_lo = vld1q_u8($hmask.as_ptr());
                let hmask_hi = vld1q_u8($hmask.as_ptr().add(16));
                // bit_clear_* is all-ones per lane where the hmask bit is
                // CLEAR (bias=4), all-zero where it's set (bias=0) --
                // matching the scalar reference's `if hmask[l] & m != 0
                // { 0 } else { 4 }`.
                let bit_clear_lo = vmvnq_u8(vtstq_u8(hmask_lo, $m_vec));
                let bit_clear_hi = vmvnq_u8(vtstq_u8(hmask_hi, $m_vec));
                let bias_lo = vandq_u8(bit_clear_lo, four);
                let bias_hi = vandq_u8(bit_clear_hi, four);

                let raw_lo_i16_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo2))), {
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bias_lo)))
                });
                let raw_lo_i16_hi =
                    vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo2))), {
                        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bias_lo)))
                    });
                let raw_hi_i16_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi2))), {
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bias_hi)))
                });
                let raw_hi_i16_hi =
                    vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi2))), {
                        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bias_hi)))
                    });

                let mut lo_acc = vdupq_n_f32(0.0);
                let mut hi_acc = vdupq_n_f32(0.0);
                for (i, half16) in [raw_lo_i16_lo, raw_lo_i16_hi].into_iter().enumerate() {
                    let lo32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(half16)));
                    let hi32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(half16)));
                    let elem_base = $x_base + i * 8;
                    let x_lo = vld1q_f32($x.as_ptr().add(elem_base));
                    let x_hi = vld1q_f32($x.as_ptr().add(elem_base + 4));
                    lo_acc = vfmaq_f32(lo_acc, lo32, x_lo);
                    lo_acc = vfmaq_f32(lo_acc, hi32, x_hi);
                }
                for (i, half16) in [raw_hi_i16_lo, raw_hi_i16_hi].into_iter().enumerate() {
                    let lo32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(half16)));
                    let hi32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(half16)));
                    let elem_base = $x_base + 16 + i * 8;
                    let x_lo = vld1q_f32($x.as_ptr().add(elem_base));
                    let x_hi = vld1q_f32($x.as_ptr().add(elem_base + 4));
                    hi_acc = vfmaq_f32(hi_acc, lo32, x_lo);
                    hi_acc = vfmaq_f32(hi_acc, hi32, x_hi);
                }
                $acc += vaddvq_f32(lo_acc) * $dl1 + vaddvq_f32(hi_acc) * $dl2;
                $x_base += 32;
            }};
        }

        for block in row_bytes.as_chunks::<Q3_K_BLOCK_BYTES>().0 {
            let hmask = &block[0..32];
            let qs = &block[32..96];
            let scales_raw: &[u8; Q3_K_SCALE_BYTES] = block[96..108].try_into().unwrap();
            let d_all = f16::from_le_bytes([block[108], block[109]]).to_f32();
            let scales = q3_k_unpack_scales(scales_raw);

            let mut is = 0usize;
            let mut m = 1u8;
            for n in 0..2 {
                let q = &qs[n * 32..n * 32 + 32];
                for shift in [0u32, 2, 4, 6] {
                    let dl1 = d_all * (scales[is] as f32 - 32.0);
                    let dl2 = d_all * (scales[is + 1] as f32 - 32.0);
                    is += 2;
                    let m_vec = vdupq_n_u8(m);
                    match shift {
                        0 => q3_k_sub_block!(0, q, hmask, m_vec, dl1, dl2, x, x_base, acc),
                        2 => q3_k_sub_block!(2, q, hmask, m_vec, dl1, dl2, x, x_base, acc),
                        4 => q3_k_sub_block!(4, q, hmask, m_vec, dl1, dl2, x, x_base, acc),
                        6 => q3_k_sub_block!(6, q, hmask, m_vec, dl1, dl2, x, x_base, acc),
                        _ => unreachable!(),
                    }
                    m <<= 1;
                }
            }
        }
        acc
    }

    /// NEON fused IQ4_NL dot product. `KVALUES_IQ4NL`'s 16 arbitrary
    /// entries are looked up via `vqtbl1q_s8` (a real 16-entry
    /// byte-table-lookup instruction; every index is 0..=15 via the
    /// `& 0x0F` mask, so this is always an in-range lookup) -- same
    /// idea as `mxfp4_nibbles_to_f32_quads`'s use of `vqtbl1q_u8` for
    /// its sub-tables, but a direct value lookup instead of an
    /// arithmetic reconstruction, since `KVALUES_IQ4NL` isn't a clean
    /// power-of-2 pattern. Safety: same contract as `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_iq4_nl_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % IQ4_NL_BLOCK_BYTES, 0);
        let low_mask = vdupq_n_u8(0x0F);
        let codebook = vld1q_s8(KVALUES_IQ4NL.as_ptr());
        let mut acc = 0f32;
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<IQ4_NL_BLOCK_BYTES>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..18];
            let bytes = vld1q_u8(qs.as_ptr());
            let lo_idx = vandq_u8(bytes, low_mask);
            let hi_idx = vshrq_n_u8(bytes, 4);
            let lo_vals = vqtbl1q_s8(codebook, lo_idx);
            let hi_vals = vqtbl1q_s8(codebook, hi_idx);

            let mut block_acc = vdupq_n_f32(0.0);
            for (half_idx, vals) in [lo_vals, hi_vals].into_iter().enumerate() {
                let lo16 = vmovl_s8(vget_low_s8(vals));
                let hi16 = vmovl_s8(vget_high_s8(vals));
                for (i, half16) in [lo16, hi16].into_iter().enumerate() {
                    let lo32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(half16)));
                    let hi32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(half16)));
                    let elem_base = x_base + half_idx * 16 + i * 8;
                    let x_lo = vld1q_f32(x.as_ptr().add(elem_base));
                    let x_hi = vld1q_f32(x.as_ptr().add(elem_base + 4));
                    block_acc = vfmaq_f32(block_acc, lo32, x_lo);
                    block_acc = vfmaq_f32(block_acc, hi32, x_hi);
                }
            }
            acc += vaddvq_f32(block_acc) * d;
            x_base += IQ4_NL_BLOCK_ELEMS;
        }
        acc
    }

    /// NEON fused IQ4_XS dot product. Same codebook lookup as
    /// `dot_iq4_nl_f32_neon`, repeated per 32-element sub-block, each
    /// with its own 6-bit scale unpacked exactly as the scalar
    /// reference does. Safety: same contract as `dot_q8_0_f32_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_iq4_xs_f32_neon(row_bytes: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row_bytes.len() % IQ4_XS_BLOCK_BYTES, 0);
        let low_mask = vdupq_n_u8(0x0F);
        let codebook = vld1q_s8(KVALUES_IQ4NL.as_ptr());
        let mut acc = 0f32;
        let mut x_base = 0usize;
        for block in row_bytes.as_chunks::<IQ4_XS_BLOCK_BYTES>().0 {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let scales_h = u16::from_le_bytes([block[2], block[3]]);
            let scales_l = &block[4..8];
            let qs = &block[8..136];

            for ib in 0..8 {
                let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                    | (((scales_h >> (2 * ib)) & 3) as u8) << 4;
                let dl = d * (ls as f32 - 32.0);
                let sub = &qs[ib * 16..ib * 16 + 16];
                let bytes = vld1q_u8(sub.as_ptr());
                let lo_idx = vandq_u8(bytes, low_mask);
                let hi_idx = vshrq_n_u8(bytes, 4);
                let lo_vals = vqtbl1q_s8(codebook, lo_idx);
                let hi_vals = vqtbl1q_s8(codebook, hi_idx);

                let mut sub_acc = vdupq_n_f32(0.0);
                for (half_idx, vals) in [lo_vals, hi_vals].into_iter().enumerate() {
                    let lo16 = vmovl_s8(vget_low_s8(vals));
                    let hi16 = vmovl_s8(vget_high_s8(vals));
                    for (i, half16) in [lo16, hi16].into_iter().enumerate() {
                        let lo32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(half16)));
                        let hi32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(half16)));
                        let elem_base = x_base + half_idx * 16 + i * 8;
                        let x_lo = vld1q_f32(x.as_ptr().add(elem_base));
                        let x_hi = vld1q_f32(x.as_ptr().add(elem_base + 4));
                        sub_acc = vfmaq_f32(sub_acc, lo32, x_lo);
                        sub_acc = vfmaq_f32(sub_acc, hi32, x_hi);
                    }
                }
                acc += vaddvq_f32(sub_acc) * dl;
                x_base += 32;
            }
        }
        acc
    }
}

/// Same idea for Q4_0: fused dequant + dot, no intermediate f32 buffer.
/// Dispatches to AVX2+FMA when available, same mechanism as
/// `dot_q8_0_f32`.
pub fn dot_q4_0_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q4_0_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q4_0_f32_neon(row_bytes, x) };
        }
    }
    dot_q4_0_f32_scalar(row_bytes, x)
}

pub fn dot_q4_0_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q4_0_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q4_0_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let nibbles = &block[2..18];
        let base = b * Q4_0_BLOCK_ELEMS;
        let mut block_acc = 0f32;
        for i in 0..16 {
            let byte = nibbles[i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            block_acc += (lo as f32) * x[base + i];
            block_acc += (hi as f32) * x[base + i + 16];
        }
        acc += block_acc * scale;
    }
    acc
}

/// Dequantize a Q4_1 buffer into f32. Formula verified against real
/// `ggml-quants.c::dequantize_row_q4_1`: `y = q*d + m`, no bias
/// subtraction (unlike Q4_0's symmetric `q-8`).
pub fn dequant_q4_1(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q4_1_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q4_1_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q4_1_BLOCK_BYTES;
    let mut out = vec![0f32; n_blocks * Q4_1_BLOCK_ELEMS];
    for (b, block) in src.as_chunks::<Q4_1_BLOCK_BYTES>().0.iter().enumerate() {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let nibbles = &block[4..20];
        let base = b * Q4_1_BLOCK_ELEMS;
        for i in 0..16 {
            let byte = nibbles[i];
            out[base + i] = (byte & 0x0F) as f32 * d + m;
            out[base + i + 16] = (byte >> 4) as f32 * d + m;
        }
    }
    Ok(out)
}

/// Fused Q4_1 dequant+dot, same math as `dequant_q4_1`. Dispatches to
/// AVX2+FMA or NEON when available, same mechanism as `dot_q4_0_f32`.
pub fn dot_q4_1_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q4_1_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q4_1_f32_neon(row_bytes, x) };
        }
    }
    dot_q4_1_f32_scalar(row_bytes, x)
}

pub fn dot_q4_1_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q4_1_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q4_1_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let nibbles = &block[4..20];
        let base = b * Q4_1_BLOCK_ELEMS;
        for i in 0..16 {
            let byte = nibbles[i];
            acc += ((byte & 0x0F) as f32 * d + m) * x[base + i];
            acc += ((byte >> 4) as f32 * d + m) * x[base + i + 16];
        }
    }
    acc
}

/// Unpacks the 5th bit for element `j` (of 16, low-nibble group) and
/// `j+16` (high-nibble group) from Q5_0/Q5_1's shared 4-byte `qh`
/// bitplane, exactly matching `ggml-quants.c`'s real bit indexing:
/// `xh_0` reads bit `j`, `xh_1` reads bit `j+16`, both placed at bit 4
/// (value 0 or 16) ready to OR into the corresponding nibble.
#[inline]
fn q5_fifth_bits(qh: u32, j: usize) -> (u8, u8) {
    let xh_0 = ((qh >> j) << 4) as u8 & 0x10;
    let xh_1 = (qh >> (j + 12)) as u8 & 0x10;
    (xh_0, xh_1)
}

/// Dequantize a Q5_0 buffer into f32. Formula verified against real
/// `ggml-quants.c::dequantize_row_q5_0`: symmetric, `y = (q-16)*d`
/// where `q` is the 4-bit nibble with the 5th bit from `qh` ORed in.
pub fn dequant_q5_0(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q5_0_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q5_0_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q5_0_BLOCK_BYTES;
    let mut out = vec![0f32; n_blocks * Q5_0_BLOCK_ELEMS];
    for (b, block) in src.as_chunks::<Q5_0_BLOCK_BYTES>().0.iter().enumerate() {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
        let qs = &block[6..22];
        let base = b * Q5_0_BLOCK_ELEMS;
        for j in 0..16 {
            let (xh_0, xh_1) = q5_fifth_bits(qh, j);
            let x0 = ((qs[j] & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((qs[j] >> 4) | xh_1) as i32 - 16;
            out[base + j] = x0 as f32 * d;
            out[base + j + 16] = x1 as f32 * d;
        }
    }
    Ok(out)
}

/// Fused Q5_0 dequant+dot, same math as `dequant_q5_0`. Dispatches to
/// AVX2+FMA or NEON when available, same mechanism as `dot_q4_0_f32`.
pub fn dot_q5_0_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q5_0_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q5_0_f32_neon(row_bytes, x) };
        }
    }
    dot_q5_0_f32_scalar(row_bytes, x)
}

pub fn dot_q5_0_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q5_0_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q5_0_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
        let qs = &block[6..22];
        let base = b * Q5_0_BLOCK_ELEMS;
        for j in 0..16 {
            let (xh_0, xh_1) = q5_fifth_bits(qh, j);
            let x0 = ((qs[j] & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((qs[j] >> 4) | xh_1) as i32 - 16;
            acc += (x0 as f32 * d) * x[base + j];
            acc += (x1 as f32 * d) * x[base + j + 16];
        }
    }
    acc
}

/// Dequantize a Q5_1 buffer into f32. Formula verified against real
/// `ggml-quants.c::dequantize_row_q5_1`: Q5_0's 5th-bit scheme, but
/// asymmetric like Q4_1 (`y = q*d + m`, no `-16` bias).
pub fn dequant_q5_1(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q5_1_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q5_1_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q5_1_BLOCK_BYTES;
    let mut out = vec![0f32; n_blocks * Q5_1_BLOCK_ELEMS];
    for (b, block) in src.as_chunks::<Q5_1_BLOCK_BYTES>().0.iter().enumerate() {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh = u32::from_le_bytes(block[4..8].try_into().unwrap());
        let qs = &block[8..24];
        let base = b * Q5_1_BLOCK_ELEMS;
        for j in 0..16 {
            let (xh_0, xh_1) = q5_fifth_bits(qh, j);
            let x0 = (qs[j] & 0x0F) | xh_0;
            let x1 = (qs[j] >> 4) | xh_1;
            out[base + j] = x0 as f32 * d + m;
            out[base + j + 16] = x1 as f32 * d + m;
        }
    }
    Ok(out)
}

/// Fused Q5_1 dequant+dot, same math as `dequant_q5_1`. Dispatches to
/// AVX2+FMA or NEON when available, same mechanism as `dot_q4_0_f32`.
pub fn dot_q5_1_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q5_1_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q5_1_f32_neon(row_bytes, x) };
        }
    }
    dot_q5_1_f32_scalar(row_bytes, x)
}

pub fn dot_q5_1_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q5_1_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q5_1_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh = u32::from_le_bytes(block[4..8].try_into().unwrap());
        let qs = &block[8..24];
        let base = b * Q5_1_BLOCK_ELEMS;
        for j in 0..16 {
            let (xh_0, xh_1) = q5_fifth_bits(qh, j);
            let x0 = (qs[j] & 0x0F) | xh_0;
            let x1 = (qs[j] >> 4) | xh_1;
            acc += (x0 as f32 * d + m) * x[base + j];
            acc += (x1 as f32 * d + m) * x[base + j + 16];
        }
    }
    acc
}

/// Dequantize a Q8_1 buffer into f32. Formula verified against real
/// `ggml-quants.c::dequantize_row_q8_1`: identical to Q8_0 (`y = q*d`)
/// -- the extra `s` field (upstream: a precomputed per-block sum used
/// only by ggml's own fused SIMD dot kernels) doesn't change the
/// dequantized value and is intentionally unread here.
pub fn dequant_q8_1(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q8_1_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q8_1_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q8_1_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * Q8_1_BLOCK_ELEMS);
    for block in src.as_chunks::<Q8_1_BLOCK_BYTES>().0 {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for i in 0..Q8_1_BLOCK_ELEMS {
            let q = block[4 + i] as i8;
            out.push(q as f32 * d);
        }
    }
    Ok(out)
}

/// Fused Q8_1 dequant+dot, same math as `dequant_q8_1`. Dispatches to
/// AVX2+FMA or NEON when available -- mathematically identical to
/// Q8_0 (`y = q*d`), so the SIMD kernels are Q8_0's kernels with the
/// quantized bytes read from offset 4 instead of offset 2 (Q8_1's
/// block has an extra 2-byte field between `d` and the int8 values).
pub fn dot_q8_1_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q8_1_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q8_1_f32_neon(row_bytes, x) };
        }
    }
    dot_q8_1_f32_scalar(row_bytes, x)
}

pub fn dot_q8_1_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q8_1_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    for (b, block) in row_bytes
        .as_chunks::<Q8_1_BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let base = b * Q8_1_BLOCK_ELEMS;
        let mut block_acc = 0f32;
        for i in 0..Q8_1_BLOCK_ELEMS {
            let q = block[4 + i] as i8;
            block_acc += (q as f32) * x[base + i];
        }
        acc += block_acc * d;
    }
    acc
}

/// Dequantize a Q2_K buffer into f32. Formula verified against real
/// `ggml-quants.c::dequantize_row_q2_K`: 16 sub-blocks of 16 elements,
/// each sub-block's `(scale, min)` packed one byte per sub-block
/// (`sc & 0xF` = 4-bit scale, `sc >> 4` = 4-bit min -- much simpler
/// than Q4_K's cross-byte 6-bit packing), value = `d*scale*raw2bit -
/// dmin*min`, `raw2bit` in 0..=3 (2 bits per element from `qs`, 4
/// elements packed per byte).
pub fn dequant_q2_k(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q2_K_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q2_K_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q2_K_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * Q2_K_BLOCK_ELEMS);
    for block in src.as_chunks::<Q2_K_BLOCK_BYTES>().0 {
        let scales: &[u8; Q2_K_SCALE_BYTES] = block[0..16].try_into().unwrap();
        let qs = &block[16..80];
        let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
        let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();

        let mut is = 0usize;
        for n in 0..2 {
            let q = &qs[n * 32..n * 32 + 32];
            let mut shift = 0u32;
            for _j in 0..4 {
                let sc1 = scales[is];
                is += 1;
                let (dl1, ml1) = (d * (sc1 & 0x0F) as f32, dmin * (sc1 >> 4) as f32);
                for &byte in &q[0..16] {
                    let raw = (byte >> shift) & 3;
                    out.push(dl1 * raw as f32 - ml1);
                }

                let sc2 = scales[is];
                is += 1;
                let (dl2, ml2) = (d * (sc2 & 0x0F) as f32, dmin * (sc2 >> 4) as f32);
                for &byte in &q[16..32] {
                    let raw = (byte >> shift) & 3;
                    out.push(dl2 * raw as f32 - ml2);
                }
                shift += 2;
            }
        }
    }
    Ok(out)
}

/// Fused Q2_K dequant+dot, same math as `dequant_q2_k`. Dispatches to
/// AVX2+FMA or NEON when available, same mechanism as `dot_q4_k_f32`.
pub fn dot_q2_k_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q2_k_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q2_k_f32_neon(row_bytes, x) };
        }
    }
    dot_q2_k_f32_scalar(row_bytes, x)
}

pub fn dot_q2_k_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q2_K_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    let mut x_base = 0usize;
    for block in row_bytes.as_chunks::<Q2_K_BLOCK_BYTES>().0 {
        let scales: &[u8; Q2_K_SCALE_BYTES] = block[0..16].try_into().unwrap();
        let qs = &block[16..80];
        let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
        let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();

        let mut is = 0usize;
        for n in 0..2 {
            let q = &qs[n * 32..n * 32 + 32];
            let mut shift = 0u32;
            for _j in 0..4 {
                let sc1 = scales[is];
                is += 1;
                let (dl1, ml1) = (d * (sc1 & 0x0F) as f32, dmin * (sc1 >> 4) as f32);
                for l in 0..16 {
                    let raw = (q[l] >> shift) & 3;
                    acc += (dl1 * raw as f32 - ml1) * x[x_base + l];
                }

                let sc2 = scales[is];
                is += 1;
                let (dl2, ml2) = (d * (sc2 & 0x0F) as f32, dmin * (sc2 >> 4) as f32);
                for l in 0..16 {
                    let raw = (q[l + 16] >> shift) & 3;
                    acc += (dl2 * raw as f32 - ml2) * x[x_base + l + 16];
                }
                shift += 2;
                x_base += 32;
            }
        }
    }
    acc
}

/// Unpacks Q3_K's 12-byte packed `scales` field into 16 signed 6-bit
/// values (range -32..=31 after the caller subtracts 32), transcribed
/// exactly from `dequantize_row_q3_K`'s real `aux[]` byte-wise
/// interleaving (four `u32`-at-a-time operations, here done per-byte
/// since Rust has no ambient SIMD-in-a-register trick to mirror C's
/// `uint32_t` shortcut) -- not reverse-engineered from the bit layout
/// alone, since a plausible-looking guess at this specific packing
/// would be easy to get wrong in a way indistinguishable from correct
/// without the real source.
fn q3_k_unpack_scales(raw: &[u8; Q3_K_SCALE_BYTES]) -> [i8; 16] {
    const KMASK1: u8 = 0x03;
    const KMASK2: u8 = 0x0F;
    let mut out = [0u8; 16];
    for j in 0..4 {
        let (a0, a1, tmp) = (raw[j], raw[4 + j], raw[8 + j]);
        // `tmp >> 0` (a no-op, dropped) kept as an explicit `>> 0` in
        // the real C source purely for symmetry with the `>>2`/`>>4`/
        // `>>6` siblings below; clippy correctly flags it as dead code
        // once written idiomatically in Rust.
        out[j] = (a0 & KMASK2) | ((tmp & KMASK1) << 4);
        out[4 + j] = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        out[8 + j] = (a0 >> 4) | (((tmp >> 4) & KMASK1) << 4);
        out[12 + j] = (a1 >> 4) | (((tmp >> 6) & KMASK1) << 4);
    }
    // Values are always in 0..64 (6 significant bits, top 2 bits of
    // each byte never set), so this bit-cast to i8 is exactly the
    // `int8_t` reinterpretation the real C code performs.
    out.map(|b| b as i8)
}

/// Dequantize a Q3_K buffer into f32. Formula verified against real
/// `ggml-quants.c::dequantize_row_q3_K`: 16 sub-blocks of 16 elements,
/// value = `d_all*(scale-32)*(raw3bit-bias)`, `raw3bit` = 2 bits from
/// `qs` plus 1 high bit from `hmask` (bit `m`, `m` sweeping all 8 bit
/// positions across the whole block -- `hmask` is indexed the same way
/// regardless of which half of `qs` is active, only the bit tested
/// changes), `bias` = 4 when the high bit is clear, 0 when set.
pub fn dequant_q3_k(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(Q3_K_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), Q3_K_BLOCK_BYTES));
    }
    let n_blocks = src.len() / Q3_K_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * Q3_K_BLOCK_ELEMS);
    for block in src.as_chunks::<Q3_K_BLOCK_BYTES>().0 {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw: &[u8; Q3_K_SCALE_BYTES] = block[96..108].try_into().unwrap();
        let d_all = f16::from_le_bytes([block[108], block[109]]).to_f32();
        let scales = q3_k_unpack_scales(scales_raw);

        let mut is = 0usize;
        let mut m = 1u8;
        for n in 0..2 {
            let q = &qs[n * 32..n * 32 + 32];
            let mut shift = 0u32;
            for _j in 0..4 {
                let dl1 = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let raw = ((q[l] >> shift) & 3) as i32;
                    let bias = if hmask[l] & m != 0 { 0 } else { 4 };
                    out.push(dl1 * (raw - bias) as f32);
                }

                let dl2 = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let raw = ((q[l + 16] >> shift) & 3) as i32;
                    let bias = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                    out.push(dl2 * (raw - bias) as f32);
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
    Ok(out)
}

/// Fused Q3_K dequant+dot, same math as `dequant_q3_k`. Dispatches to
/// AVX2+FMA or NEON when available, same mechanism as `dot_q4_k_f32`.
pub fn dot_q3_k_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_q3_k_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_q3_k_f32_neon(row_bytes, x) };
        }
    }
    dot_q3_k_f32_scalar(row_bytes, x)
}

pub fn dot_q3_k_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % Q3_K_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    let mut x_base = 0usize;
    for block in row_bytes.as_chunks::<Q3_K_BLOCK_BYTES>().0 {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw: &[u8; Q3_K_SCALE_BYTES] = block[96..108].try_into().unwrap();
        let d_all = f16::from_le_bytes([block[108], block[109]]).to_f32();
        let scales = q3_k_unpack_scales(scales_raw);

        let mut is = 0usize;
        let mut m = 1u8;
        for n in 0..2 {
            let q = &qs[n * 32..n * 32 + 32];
            let mut shift = 0u32;
            for _j in 0..4 {
                let dl1 = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let raw = ((q[l] >> shift) & 3) as i32;
                    let bias = if hmask[l] & m != 0 { 0 } else { 4 };
                    acc += (dl1 * (raw - bias) as f32) * x[x_base + l];
                }

                let dl2 = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let raw = ((q[l + 16] >> shift) & 3) as i32;
                    let bias = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                    acc += (dl2 * (raw - bias) as f32) * x[x_base + l + 16];
                }
                shift += 2;
                m <<= 1;
                x_base += 32;
            }
        }
    }
    acc
}

pub const IQ4_NL_BLOCK_BYTES: usize = 18;
pub const IQ4_NL_BLOCK_ELEMS: usize = 32;
pub const IQ4_XS_BLOCK_BYTES: usize = 136;
pub const IQ4_XS_BLOCK_ELEMS: usize = 256;

/// The 16-entry non-linear codebook shared by IQ4_NL and IQ4_XS: a 4-bit
/// index maps to one of these signed `i8` values instead of a linear
/// `nibble*scale` transform. Verified against real ggml-quants.c
/// (`kvalues_iq4nl`) rather than derived.
pub(crate) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub fn dequant_iq4_nl(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(IQ4_NL_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), IQ4_NL_BLOCK_BYTES));
    }
    let n_blocks = src.len() / IQ4_NL_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * IQ4_NL_BLOCK_ELEMS);
    for block in src.as_chunks::<IQ4_NL_BLOCK_BYTES>().0 {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..18];
        let mut lo = [0f32; 16];
        let mut hi = [0f32; 16];
        for (j, &byte) in qs.iter().enumerate() {
            lo[j] = d * KVALUES_IQ4NL[(byte & 0xf) as usize] as f32;
            hi[j] = d * KVALUES_IQ4NL[(byte >> 4) as usize] as f32;
        }
        out.extend_from_slice(&lo);
        out.extend_from_slice(&hi);
    }
    Ok(out)
}

/// Fused IQ4_NL dequant+dot, same math as `dequant_iq4_nl`. Dispatches
/// to AVX2+FMA or NEON when available, same mechanism as `dot_q4_0_f32`.
pub fn dot_iq4_nl_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_iq4_nl_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_iq4_nl_f32_neon(row_bytes, x) };
        }
    }
    dot_iq4_nl_f32_scalar(row_bytes, x)
}

pub fn dot_iq4_nl_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % IQ4_NL_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    let mut x_base = 0usize;
    for block in row_bytes.as_chunks::<IQ4_NL_BLOCK_BYTES>().0 {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..18];
        for (j, &byte) in qs.iter().enumerate() {
            acc += (d * KVALUES_IQ4NL[(byte & 0xf) as usize] as f32) * x[x_base + j];
            acc += (d * KVALUES_IQ4NL[(byte >> 4) as usize] as f32) * x[x_base + 16 + j];
        }
        x_base += IQ4_NL_BLOCK_ELEMS;
    }
    acc
}

pub fn dequant_iq4_xs(src: &[u8]) -> Result<Vec<f32>, QuantError> {
    if !src.len().is_multiple_of(IQ4_XS_BLOCK_BYTES) {
        return Err(QuantError::Misaligned(src.len(), IQ4_XS_BLOCK_BYTES));
    }
    let n_blocks = src.len() / IQ4_XS_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * IQ4_XS_BLOCK_ELEMS);
    for block in src.as_chunks::<IQ4_XS_BLOCK_BYTES>().0 {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];

        for ib in 0..8 {
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                | (((scales_h >> (2 * ib)) & 3) as u8) << 4;
            let dl = d * (ls as f32 - 32.0);
            let sub = &qs[ib * 16..ib * 16 + 16];
            let mut lo = [0f32; 16];
            let mut hi = [0f32; 16];
            for (j, &byte) in sub.iter().enumerate() {
                lo[j] = dl * KVALUES_IQ4NL[(byte & 0xf) as usize] as f32;
                hi[j] = dl * KVALUES_IQ4NL[(byte >> 4) as usize] as f32;
            }
            out.extend_from_slice(&lo);
            out.extend_from_slice(&hi);
        }
    }
    Ok(out)
}

/// Fused IQ4_XS dequant+dot, same math as `dequant_iq4_xs`. Dispatches
/// to AVX2+FMA or NEON when available, same mechanism as `dot_q4_0_f32`.
pub fn dot_iq4_xs_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_iq4_xs_f32_avx2(row_bytes, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_iq4_xs_f32_neon(row_bytes, x) };
        }
    }
    dot_iq4_xs_f32_scalar(row_bytes, x)
}

pub fn dot_iq4_xs_f32_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len() % IQ4_XS_BLOCK_BYTES, 0);
    let mut acc = 0f32;
    let mut x_base = 0usize;
    for block in row_bytes.as_chunks::<IQ4_XS_BLOCK_BYTES>().0 {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];

        for ib in 0..8 {
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                | (((scales_h >> (2 * ib)) & 3) as u8) << 4;
            let dl = d * (ls as f32 - 32.0);
            let sub = &qs[ib * 16..ib * 16 + 16];
            for (j, &byte) in sub.iter().enumerate() {
                acc += (dl * KVALUES_IQ4NL[(byte & 0xf) as usize] as f32) * x[x_base + j];
                acc += (dl * KVALUES_IQ4NL[(byte >> 4) as usize] as f32) * x[x_base + 16 + j];
            }
            x_base += 32;
        }
    }
    acc
}

/// Elements per MXFP4 scale group (real, confirmed both from ggml's
/// `QK_MXFP4` and directly from a real Kimi K3 shard's own tensor shapes:
/// `*.weight_scale` is `in_dim/32` bytes, `*.weight_packed` is `in_dim/2`
/// bytes).
pub const MXFP4_GROUP_SIZE: usize = 32;

/// Real (non-doubled) E2M1 4-bit float codebook: sign + 2 exponent bits +
/// 1 mantissa bit, per the OCP Microscaling Formats v1.0 spec. Verified
/// against real `ggml-common.h`'s `kvalues_mxfp4` table, which stores
/// these same 16 values pre-doubled (paired with a scale halved by
/// `ggml_e8m0_to_fp32_half`) purely so ggml's table can stay `int8_t`;
/// the two conventions multiply out identically. Frink uses the real,
/// undoubled values directly against the real (unhalved) E8M0 scale below
/// instead, since there's no int8-table constraint here.
const KVALUES_MXFP4: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// OCP MX E8M0 scale byte -> `2^(e-127)` (bias 127, same bias convention
/// as an IEEE754 f32 exponent field). Implemented by placing `e` directly
/// into an f32's exponent bits (mantissa zero) -- exact, not an
/// approximation -- exactly mirroring real `ggml_e8m0_to_fp32`. `e = 0`
/// is special-cased (the direct bit-shift would just produce `0.0`, not
/// the intended `2^-127`) using the same subnormal bit pattern the real
/// implementation uses. `e = 255` is reserved for NaN by the OCP spec and
/// is not specially handled, matching that same real implementation's own
/// documented limitation ("does not handle NaN").
fn e8m0_scale(e: u8) -> f32 {
    if e == 0 {
        f32::from_bits(0x0040_0000)
    } else {
        f32::from_bits((e as u32) << 23)
    }
}

/// Dequantizes one row of Kimi K3's MXFP4-packed expert weights. Unlike
/// every other kernel in this module, MXFP4 here is NOT a single
/// interleaved byte stream -- Kimi K3's real safetensors checkpoint
/// stores the packed 4-bit codes and the per-group E8M0 scales as two
/// separate tensors (`*.weight_packed`, `*.weight_scale`; confirmed
/// directly against a real shard header's tensor shapes, not ggml's own
/// combined-block GGUF convention), so this takes both buffers directly
/// rather than one combined block stream. `packed` is `in_dim/2` bytes
/// (2 nibble-packed E2M1 codes per byte, low-nibble-first-half /
/// high-nibble-second-half within each 32-element group -- same
/// convention as this module's other nibble-packed formats); `scales` is
/// `in_dim/MXFP4_GROUP_SIZE` bytes (one E8M0 scale byte per group).
pub fn dequant_mxfp4_row(packed: &[u8], scales: &[u8]) -> Result<Vec<f32>, QuantError> {
    let expected_packed_len = scales.len() * (MXFP4_GROUP_SIZE / 2);
    if packed.len() != expected_packed_len {
        return Err(QuantError::Mxfp4RowMismatch(
            packed.len(),
            expected_packed_len,
        ));
    }
    let mut out = Vec::with_capacity(scales.len() * MXFP4_GROUP_SIZE);
    for (g, &e) in scales.iter().enumerate() {
        let d = e8m0_scale(e);
        let group = &packed[g * (MXFP4_GROUP_SIZE / 2)..(g + 1) * (MXFP4_GROUP_SIZE / 2)];
        let mut lo = [0f32; MXFP4_GROUP_SIZE / 2];
        let mut hi = [0f32; MXFP4_GROUP_SIZE / 2];
        for (j, &byte) in group.iter().enumerate() {
            lo[j] = d * KVALUES_MXFP4[(byte & 0xf) as usize];
            hi[j] = d * KVALUES_MXFP4[(byte >> 4) as usize];
        }
        out.extend_from_slice(&lo);
        out.extend_from_slice(&hi);
    }
    Ok(out)
}

/// Fused MXFP4 dequant+dot, same math as `dequant_mxfp4_row`. Dispatches
/// to AVX2+FMA or NEON when available (see `simd_x86::dot_mxfp4_row_f32_avx2`/
/// `simd_aarch64::dot_mxfp4_row_f32_neon`), same mechanism as
/// `dot_q4_0_f32` -- this is the hot path for every routed expert's FFN
/// in a real Kimi K3 forward pass, so unlike Q4_0/Q8_0's optional
/// legacy-format status, keeping this scalar-only directly costs real
/// inference speed.
pub fn dot_mxfp4_row_f32(packed: &[u8], scales: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_mxfp4_row_f32_avx2(packed, scales, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_aarch64::dot_mxfp4_row_f32_neon(packed, scales, x) };
        }
    }
    dot_mxfp4_row_f32_scalar(packed, scales, x)
}

pub fn dot_mxfp4_row_f32_scalar(packed: &[u8], scales: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(packed.len(), scales.len() * (MXFP4_GROUP_SIZE / 2));
    let mut acc = 0f32;
    let mut x_base = 0usize;
    for (g, &e) in scales.iter().enumerate() {
        let d = e8m0_scale(e);
        let group = &packed[g * (MXFP4_GROUP_SIZE / 2)..(g + 1) * (MXFP4_GROUP_SIZE / 2)];
        for (j, &byte) in group.iter().enumerate() {
            acc += (d * KVALUES_MXFP4[(byte & 0xf) as usize]) * x[x_base + j];
            acc += (d * KVALUES_MXFP4[(byte >> 4) as usize]) * x[x_base + MXFP4_GROUP_SIZE / 2 + j];
        }
        x_base += MXFP4_GROUP_SIZE;
    }
    acc
}

// ---------------------------------------------------------------------
// IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S: the
// codebook-grid low-bit formats used throughout published "Dynamic"
// low-bit GGUFs of large MoE models.
// Unlike every format above, an element's magnitude comes from a shared
// grid table (`iq_tables`) indexed by packed code bits, with signs
// applied from a shared 7-bit sign-pattern table (the `_XXS`/`IQ2_XS`
// tier) or from literal sign bytes (the `_S` tier) -- not from an
// arithmetic transform of the stored bits. Layouts and semantics
// written against ggml's published dequant reference
// (`dequantize_row_iq1_s`/`_iq1_m`/`_iq2_xxs`/`_iq2_xs`/`_iq2_s`/
// `_iq3_xxs`/`_iq3_s` in `ggml/src/ggml-quants.c`); cross-validated
// against the real compiled ggml implementation -- for the `_XXS` tier
// via an independent Python reference checked against
// `ggml_get_type_traits(...)->to_float`, and for IQ2_XS/IQ2_S/IQ3_S/
// IQ1_M by linking ggml-quants.c directly and asserting bit-exact
// equality with its output (see this module's tests).
//
// A wrong grid index or a wrong sign/scale unpack in these formats does
// not produce obviously broken numbers -- it produces plausible ones
// from the same codebook. So every one of them is pinned to ggml's own
// bytes rather than to a self-consistent re-derivation, and the pinned
// blocks deliberately include the all-ones pattern (maximum grid index,
// every sign bit, maximum scale nibbles) and the all-zeros pattern.
// ---------------------------------------------------------------------

/// IQ1_S: d(f16) + 32 low-index bytes + 8 u16 (3 high index bits + 3
/// scale bits + sign-of-delta per 32-element group). 1.5625 bpw.
pub const IQ1_S_BLOCK_BYTES: usize = 50;
pub const IQ1_S_BLOCK_ELEMS: usize = 256;
/// IQ1_M: 32 low-index bytes + 16 qh bytes (3 high index bits + a
/// sign-of-delta bit per 8-element group) + 8 scale bytes. 1.75 bpw.
/// The only IQ format with no f16 scale field -- see `for_each_iq1_m`.
pub const IQ1_M_BLOCK_BYTES: usize = 56;
pub const IQ1_M_BLOCK_ELEMS: usize = 256;
/// IQ2_XXS: d(f16) + 32 u16 codes (grid indices + packed scale/signs).
/// 2.0625 bpw.
pub const IQ2_XXS_BLOCK_BYTES: usize = 66;
pub const IQ2_XXS_BLOCK_ELEMS: usize = 256;
/// IQ2_XS: d(f16) + 32 u16 codes (9-bit grid index + 7-bit sign index)
/// + 8 scale bytes (two 4-bit scales per 32-element group). 2.3125 bpw.
pub const IQ2_XS_BLOCK_BYTES: usize = 74;
pub const IQ2_XS_BLOCK_ELEMS: usize = 256;
/// IQ2_S: d(f16) + 32 low-index bytes + 32 literal sign bytes + 8 qh
/// bytes (2 high index bits per group of 8) + 8 scale bytes. 2.5625 bpw.
pub const IQ2_S_BLOCK_BYTES: usize = 82;
pub const IQ2_S_BLOCK_ELEMS: usize = 256;
/// IQ3_XXS: d(f16) + 64 grid-index bytes + 8 u32 scale/sign words.
/// 3.0625 bpw.
pub const IQ3_XXS_BLOCK_BYTES: usize = 98;
pub const IQ3_XXS_BLOCK_ELEMS: usize = 256;
/// IQ3_S: d(f16) + 64 low-index bytes + 8 qh bytes (one 9th index bit
/// per grid code) + 32 literal sign bytes + 4 scale bytes (two 4-bit
/// scales per pair of 32-element groups). 3.4375 bpw.
pub const IQ3_S_BLOCK_BYTES: usize = 110;
pub const IQ3_S_BLOCK_ELEMS: usize = 256;

/// ggml's IQ1S_DELTA: the constant additive shift applied to every
/// IQ1_S grid value, signed per 32-element group. IQ1_M's IQ1M_DELTA is
/// the same 0.125 in ggml-common.h, applied per 8-element group; kept as
/// one constant here because the two are defined equal upstream and a
/// second name would only invite them to drift apart in this file.
const IQ1S_DELTA: f32 = 0.125;

/// `+1.0` when the matching bit in an IQ sign byte is clear, `-1.0` when
/// it is set. Every IQ2/IQ3 format signs its grid magnitudes this way;
/// only the provenance of `signs` differs (a `KSIGNS_IQ2XS` lookup for
/// the `_XXS`/`IQ2_XS` tier, a literal stored byte for the `_S` tier).
#[inline]
fn iq_sign(signs: u8, j: usize) -> f32 {
    if signs & iq_tables::KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

#[inline]
fn read_f16(bytes: &[u8]) -> f32 {
    f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

/// Shared IQ1_S per-block walk: calls `emit(elem_index, value)` for all
/// 256 elements, so dequant and fused-dot stay one algorithm.
#[inline]
fn for_each_iq1_s(block: &[u8], mut emit: impl FnMut(usize, f32)) {
    let d = read_f16(block);
    let qs = &block[2..34];
    let qh = &block[34..50];
    let mut idx = 0usize;
    for ib in 0..8 {
        let h = u16::from_le_bytes([qh[2 * ib], qh[2 * ib + 1]]);
        let dl = d * (2.0 * ((h >> 12) & 7) as f32 + 1.0);
        let delta = if h & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            let grid_index = qs[4 * ib + l] as usize | ((((h >> (3 * l)) & 7) as usize) << 8);
            let row = iq_tables::IQ1S_GRID[grid_index];
            for j in 0..8 {
                let v = ((row >> (8 * j)) & 0xFF) as u8 as i8;
                emit(idx, dl * (v as f32 + delta));
                idx += 1;
            }
        }
    }
}

/// Shared IQ2_XXS per-block walk (same emit contract as IQ1_S above).
#[inline]
fn for_each_iq2_xxs(block: &[u8], mut emit: impl FnMut(usize, f32)) {
    let d = read_f16(block);
    let qs: Vec<u16> = block[2..66]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut idx = 0usize;
    for ib32 in 0..8 {
        let g = &qs[4 * ib32..4 * ib32 + 4];
        let aux32_1 = g[2] as u32 | ((g[3] as u32) << 16);
        let db = d * (0.5 + (aux32_1 >> 28) as f32) * 0.25;
        let aux8 = [
            (g[0] & 0xFF) as usize,
            (g[0] >> 8) as usize,
            (g[1] & 0xFF) as usize,
            (g[1] >> 8) as usize,
        ];
        for (l, &code) in aux8.iter().enumerate() {
            let row = iq_tables::IQ2XXS_GRID[code];
            let signs = iq_tables::KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                let mag = ((row >> (8 * j)) & 0xFF) as f32;
                emit(idx, db * mag * iq_sign(signs, j));
                idx += 1;
            }
        }
    }
}

/// Shared IQ3_XXS per-block walk (same emit contract as IQ1_S above).
#[inline]
fn for_each_iq3_xxs(block: &[u8], mut emit: impl FnMut(usize, f32)) {
    let d = read_f16(block);
    let qs = &block[2..66];
    let sas = &block[66..98];
    let mut idx = 0usize;
    for ib32 in 0..8 {
        let aux32 = u32::from_le_bytes([
            sas[4 * ib32],
            sas[4 * ib32 + 1],
            sas[4 * ib32 + 2],
            sas[4 * ib32 + 3],
        ]);
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = iq_tables::KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            let g1 = iq_tables::IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize];
            let g2 = iq_tables::IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize];
            for j in 0..4 {
                emit(
                    idx + j,
                    db * ((g1 >> (8 * j)) & 0xFF) as f32 * iq_sign(signs, j),
                );
            }
            for j in 0..4 {
                emit(
                    idx + 4 + j,
                    db * ((g2 >> (8 * j)) & 0xFF) as f32 * iq_sign(signs, j + 4),
                );
            }
            idx += 8;
        }
    }
}

/// Shared IQ2_XS per-block walk (same emit contract as IQ1_S above).
///
/// IQ2_XS is IQ2_XXS with the scales pulled out of the code words: each
/// u16 code now spends all 16 bits on payload (9-bit grid index + 7-bit
/// `KSIGNS_IQ2XS` index), and the per-group scales move into their own
/// 8 trailing bytes, two 4-bit scales per 32-element group. The `l/2`
/// split below is ggml's: within a group of 32, codes 0-1 take the low
/// nibble's scale and codes 2-3 the high nibble's.
#[inline]
fn for_each_iq2_xs(block: &[u8], mut emit: impl FnMut(usize, f32)) {
    let d = read_f16(block);
    let qs = &block[2..66];
    let scales = &block[66..74];
    let mut idx = 0usize;
    for ib32 in 0..8 {
        let db = [
            d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let code = u16::from_le_bytes([qs[8 * ib32 + 2 * l], qs[8 * ib32 + 2 * l + 1]]);
            let row = iq_tables::IQ2XS_GRID[(code & 511) as usize];
            let signs = iq_tables::KSIGNS_IQ2XS[(code >> 9) as usize];
            for j in 0..8 {
                let mag = ((row >> (8 * j)) & 0xFF) as f32;
                emit(idx, db[l / 2] * mag * iq_sign(signs, j));
                idx += 1;
            }
        }
    }
}

/// Shared IQ2_S per-block walk (same emit contract as IQ1_S above).
///
/// IQ2_S spends its extra quarter-bit on *literal* signs: instead of a
/// 7-bit index into `KSIGNS_IQ2XS` (which can only express the 128 sign
/// patterns of even parity), each group of 8 elements gets a full sign
/// byte. That frees the code word of sign bits entirely, so the grid
/// index widens to 10 bits -- 8 from `qs` plus 2 pulled out of the
/// group's `qh` byte, a different 2-bit field per code (`l` selects
/// which). Note ggml declares `qs` as one 64-byte array and then aliases
/// its second half as the sign bytes; the two halves are named
/// separately here because they are unrelated payloads.
#[inline]
fn for_each_iq2_s(block: &[u8], mut emit: impl FnMut(usize, f32)) {
    let d = read_f16(block);
    let qs = &block[2..34];
    let sign_bytes = &block[34..66];
    let qh = &block[66..74];
    let scales = &block[74..82];
    let mut idx = 0usize;
    for ib32 in 0..8 {
        let db = [
            d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let hi = ((qh[ib32] as usize) << (8 - 2 * l)) & 0x300;
            let row = iq_tables::IQ2S_GRID[qs[4 * ib32 + l] as usize | hi];
            let signs = sign_bytes[4 * ib32 + l];
            for j in 0..8 {
                let mag = ((row >> (8 * j)) & 0xFF) as f32;
                emit(idx, db[l / 2] * mag * iq_sign(signs, j));
                idx += 1;
            }
        }
    }
}

/// Shared IQ3_S per-block walk (same emit contract as IQ1_S above).
///
/// IQ3_S is to IQ3_XXS what IQ2_S is to IQ2_XXS: literal sign bytes
/// instead of `KSIGNS_IQ2XS` indices, and the freed bits spent widening
/// the grid index to 9 bits (8 from `qs`, the 9th from the group's `qh`
/// byte, one bit per code). Scales are the odd part: there are only 4
/// scale bytes for 8 groups of 32, so one byte's two nibbles cover
/// *two consecutive groups* -- low nibble for the even group, high
/// nibble for the odd one -- and the scale is `1 + 2*nibble` (an odd
/// integer multiplier), not the `(0.5 + nibble) * 0.25` of the IQ2 tier.
///
/// ggml writes this as a loop stepping `ib32` by 2 with pointer bumps
/// inside; unrolled here to a plain per-group loop with explicit
/// offsets, which is the same traversal with the aliasing spelled out.
#[inline]
fn for_each_iq3_s(block: &[u8], mut emit: impl FnMut(usize, f32)) {
    let d = read_f16(block);
    let qs = &block[2..66];
    let qh = &block[66..74];
    let sign_bytes = &block[74..106];
    let scales = &block[106..110];
    let mut idx = 0usize;
    for ib32 in 0..8 {
        let nibble = if ib32 % 2 == 0 {
            scales[ib32 / 2] & 0xF
        } else {
            scales[ib32 / 2] >> 4
        };
        let db = d * (1.0 + 2.0 * nibble as f32);
        for l in 0..4 {
            // The 9th index bit for code `2l` is qh bit `2l`, and for
            // code `2l+1` it is qh bit `2l+1` -- ggml expresses both as
            // a left shift landing that bit on 256.
            let h = qh[ib32] as usize;
            let i1 = qs[8 * ib32 + 2 * l] as usize | ((h << (8 - 2 * l)) & 256);
            let i2 = qs[8 * ib32 + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256);
            let g1 = iq_tables::IQ3S_GRID[i1];
            let g2 = iq_tables::IQ3S_GRID[i2];
            let signs = sign_bytes[4 * ib32 + l];
            for j in 0..4 {
                emit(
                    idx + j,
                    db * ((g1 >> (8 * j)) & 0xFF) as f32 * iq_sign(signs, j),
                );
            }
            for j in 0..4 {
                emit(
                    idx + 4 + j,
                    db * ((g2 >> (8 * j)) & 0xFF) as f32 * iq_sign(signs, j + 4),
                );
            }
            idx += 8;
        }
    }
}

/// Shared IQ1_M per-block walk (same emit contract as IQ1_S above).
///
/// IQ1_M reuses IQ1_S's 2048-entry signed grid and its `+/-delta` shift,
/// but restructures everything around it, and it is the one IQ format
/// with **no f16 scale field**: the block's 16 scale bits are scattered
/// as the top nibble of each of the four 16-bit scale words, and are
/// reassembled here into an f16 bit pattern. The remaining 12 bits of
/// each word carry four 3-bit sub-scales (two 32-element groups per
/// word, two sub-scales per group covering 16 elements each), so the
/// scale resolution is twice IQ1_S's.
///
/// The delta sign is also finer-grained than IQ1_S's: one bit per 8
/// elements (`qh` bits 3 and 7) rather than one per 32.
#[inline]
fn for_each_iq1_m(block: &[u8], mut emit: impl FnMut(usize, f32)) {
    let qs = &block[0..32];
    let qh = &block[32..48];
    let scales = &block[48..56];
    let sc: [u16; 4] =
        std::array::from_fn(|k| u16::from_le_bytes([scales[2 * k], scales[2 * k + 1]]));
    // Top nibble of sc[0]..sc[3] -> f16 bits 0-3, 4-7, 8-11, 12-15.
    let d = f16::from_bits(
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000),
    )
    .to_f32();
    let mut idx = 0usize;
    for ib in 0..8 {
        let shift = 6 * (ib % 2);
        let dl = [
            d * (2.0 * ((sc[ib / 2] >> shift) & 7) as f32 + 1.0),
            d * (2.0 * ((sc[ib / 2] >> (shift + 3)) & 7) as f32 + 1.0),
        ];
        let (h0, h1) = (qh[2 * ib] as usize, qh[2 * ib + 1] as usize);
        // Grid index high bits: qh nibble bits 0-2 of each half-byte.
        // Bits 3 and 7 of each qh byte are the delta signs instead.
        let grid_idx = [
            qs[4 * ib] as usize | ((h0 << 8) & 0x700),
            qs[4 * ib + 1] as usize | ((h0 << 4) & 0x700),
            qs[4 * ib + 2] as usize | ((h1 << 8) & 0x700),
            qs[4 * ib + 3] as usize | ((h1 << 4) & 0x700),
        ];
        let delta = [
            if h0 & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if h0 & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if h1 & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if h1 & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
        ];
        for l in 0..4 {
            let row = iq_tables::IQ1S_GRID[grid_idx[l]];
            for j in 0..8 {
                let v = ((row >> (8 * j)) & 0xFF) as u8 as i8;
                emit(idx, dl[l / 2] * (v as f32 + delta[l]));
                idx += 1;
            }
        }
    }
}

macro_rules! iq_dequant_and_dot {
    ($dequant:ident, $dot_scalar:ident, $walk:ident, $bytes:ident, $elems:ident) => {
        pub fn $dequant(src: &[u8]) -> Result<Vec<f32>, QuantError> {
            if !src.len().is_multiple_of($bytes) {
                return Err(QuantError::Misaligned(src.len(), $bytes));
            }
            let n_blocks = src.len() / $bytes;
            let mut out = vec![0f32; n_blocks * $elems];
            for (b, block) in src.chunks_exact($bytes).enumerate() {
                let base = b * $elems;
                $walk(block, |i, v| out[base + i] = v);
            }
            Ok(out)
        }

        pub fn $dot_scalar(row_bytes: &[u8], x: &[f32]) -> f32 {
            debug_assert_eq!(row_bytes.len() % $bytes, 0);
            let mut acc = 0f32;
            let mut x_base = 0usize;
            for block in row_bytes.chunks_exact($bytes) {
                $walk(block, |i, v| acc += v * x[x_base + i]);
                x_base += $elems;
            }
            acc
        }
    };
}

/// Hand-written dispatch for the IQ codebook formats: AVX2+FMA when the
/// host supports it (verified directly against the scalar reference on
/// real x86_64 hardware -- see this module's tests), scalar otherwise.
/// No NEON kernels yet for these formats (no aarch64 host was available
/// to verify one on; the scalar path serves ARM).
macro_rules! iq_dispatch {
    ($dot:ident, $dot_scalar:ident, $avx2:ident) => {
        pub fn $dot(row_bytes: &[u8], x: &[f32]) -> f32 {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return unsafe { simd_x86::$avx2(row_bytes, x) };
                }
            }
            $dot_scalar(row_bytes, x)
        }
    };
}

iq_dispatch!(dot_iq1_s_f32, dot_iq1_s_f32_scalar, dot_iq1_s_f32_avx2);
iq_dispatch!(
    dot_iq2_xxs_f32,
    dot_iq2_xxs_f32_scalar,
    dot_iq2_xxs_f32_avx2
);
iq_dispatch!(
    dot_iq3_xxs_f32,
    dot_iq3_xxs_f32_scalar,
    dot_iq3_xxs_f32_avx2
);

/// IQ2_XS / IQ2_S / IQ3_S / IQ1_M dispatch: scalar only. These landed
/// for *coverage* -- before them, tags 17/21/22/29 fell to
/// `GgmlType::Other` and the tensor could not be decoded at all, which
/// silently ruled out 5 of the 16 published Unsloth `UD-*` variants.
/// They deliberately match the state of their older siblings' NEON/GPU
/// story (none), rather than growing a vectorized path that no golden
/// vector would then be able to distinguish from the scalar one.
pub fn dot_iq2_xs_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    dot_iq2_xs_f32_scalar(row_bytes, x)
}

pub fn dot_iq2_s_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    dot_iq2_s_f32_scalar(row_bytes, x)
}

pub fn dot_iq3_s_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    dot_iq3_s_f32_scalar(row_bytes, x)
}

pub fn dot_iq1_m_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    dot_iq1_m_f32_scalar(row_bytes, x)
}

/// GGUF block-MXFP4 dispatch: scalar only so far (the two-buffer
/// safetensors MXFP4 form has AVX2/NEON kernels above; this block form
/// hasn't needed one yet).
pub fn dot_mxfp4_gguf_f32(row_bytes: &[u8], x: &[f32]) -> f32 {
    dot_mxfp4_gguf_f32_scalar(row_bytes, x)
}

iq_dequant_and_dot!(
    dequant_iq1_s,
    dot_iq1_s_f32_scalar,
    for_each_iq1_s,
    IQ1_S_BLOCK_BYTES,
    IQ1_S_BLOCK_ELEMS
);
iq_dequant_and_dot!(
    dequant_iq2_xxs,
    dot_iq2_xxs_f32_scalar,
    for_each_iq2_xxs,
    IQ2_XXS_BLOCK_BYTES,
    IQ2_XXS_BLOCK_ELEMS
);
iq_dequant_and_dot!(
    dequant_iq3_xxs,
    dot_iq3_xxs_f32_scalar,
    for_each_iq3_xxs,
    IQ3_XXS_BLOCK_BYTES,
    IQ3_XXS_BLOCK_ELEMS
);
iq_dequant_and_dot!(
    dequant_iq2_xs,
    dot_iq2_xs_f32_scalar,
    for_each_iq2_xs,
    IQ2_XS_BLOCK_BYTES,
    IQ2_XS_BLOCK_ELEMS
);
iq_dequant_and_dot!(
    dequant_iq2_s,
    dot_iq2_s_f32_scalar,
    for_each_iq2_s,
    IQ2_S_BLOCK_BYTES,
    IQ2_S_BLOCK_ELEMS
);
iq_dequant_and_dot!(
    dequant_iq3_s,
    dot_iq3_s_f32_scalar,
    for_each_iq3_s,
    IQ3_S_BLOCK_BYTES,
    IQ3_S_BLOCK_ELEMS
);
iq_dequant_and_dot!(
    dequant_iq1_m,
    dot_iq1_m_f32_scalar,
    for_each_iq1_m,
    IQ1_M_BLOCK_BYTES,
    IQ1_M_BLOCK_ELEMS
);

/// GGUF block-MXFP4 (ggml type tag 39): one 17-byte block = 1 E8M0
/// scale byte + 16 nibble bytes covering 32 elements, low nibble ->
/// element `j`, high nibble -> element `j+16`. Same E2M1 codebook and
/// E8M0 scale math as the Kimi safetensors two-buffer MXFP4 path above
/// (`dot_mxfp4_row_f32`) -- ggml expresses it as doubled-integer
/// kvalues times a half scale (`2^(e-128)`), this module as true E2M1
/// values times the full `2^(e-127)` scale; the products are identical
/// across the whole E8M0 range including the `e < 2` denormal
/// patterns. Only the byte layout differs: interleaved 17-byte blocks
/// in one stream here, two separate packed/scale tensors there.
pub const MXFP4_GGUF_BLOCK_BYTES: usize = 17;
pub const MXFP4_GGUF_BLOCK_ELEMS: usize = 32;

/// Shared GGUF-block-MXFP4 per-block walk (same emit contract as the
/// IQ walks above).
#[inline]
fn for_each_mxfp4_gguf(block: &[u8], mut emit: impl FnMut(usize, f32)) {
    let d = e8m0_scale(block[0]);
    for (j, &byte) in block[1..17].iter().enumerate() {
        emit(j, d * KVALUES_MXFP4[(byte & 0x0F) as usize]);
        emit(j + 16, d * KVALUES_MXFP4[(byte >> 4) as usize]);
    }
}

iq_dequant_and_dot!(
    dequant_mxfp4_gguf,
    dot_mxfp4_gguf_f32_scalar,
    for_each_mxfp4_gguf,
    MXFP4_GGUF_BLOCK_BYTES,
    MXFP4_GGUF_BLOCK_ELEMS
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_kv_blocks_roundtrip_reasonable() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.17).sin() * 2.0).collect();
        let packed = pack_q4_kv_blocks(&x);
        assert_eq!(packed.len(), 2 * Q4_KV_BLOCK_BYTES);
        let y = unpack_q4_kv_blocks(&packed).unwrap();
        assert_eq!(y.len(), 64);
        let mut err = 0.0f32;
        for (a, b) in x.iter().zip(y.iter()) {
            err += (a - b).abs();
        }
        err /= x.len() as f32;
        assert!(err < 0.2, "mean abs err {err}");
    }

    #[test]
    fn q8_0_roundtrip_is_within_quantization_error() {
        let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.37).collect();
        let packed = quantize_q8_0(&original);
        assert_eq!(packed.len(), Q8_0_BLOCK_BYTES);
        let restored = dequant_q8_0(&packed).unwrap();
        assert_eq!(restored.len(), 32);
        for (a, b) in original.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.1, "a={a} b={b}");
        }
    }

    #[test]
    fn quantize_activations_q8_reconstructs_within_quant_error() {
        let x: Vec<f32> = (0..64)
            .map(|i| ((i as f32) * 0.13 - 4.0).sin() * 3.0)
            .collect();
        let act = quantize_activations_q8(&x);
        assert_eq!(act.n_blocks(), 2);
        assert_eq!(act.q.len(), 64);
        for (b, chunk) in x.as_chunks::<32>().0.iter().enumerate() {
            let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let tol = amax / 127.0 + 1e-6;
            for (i, &v) in chunk.iter().enumerate() {
                let recon = act.q[b * 32 + i] as f32 * act.d[b];
                assert!((recon - v).abs() <= tol, "b={b} i={i} v={v} recon={recon}");
            }
        }
    }

    #[test]
    fn quantize_activations_q8_handles_all_zero_block() {
        let act = quantize_activations_q8(&[0f32; 32]);
        assert_eq!(act.d[0], 0.0);
        assert!(act.q.iter().all(|&q| q == 0));
    }

    #[test]
    fn quantize_activations_q8_parallel_matches_serial() {
        let x: Vec<f32> = (0..512)
            .map(|i| ((i as f32) * 0.07 - 8.0).sin() * 2.5)
            .collect();
        let got = quantize_activations_q8(&x);
        let n_blocks = x.len() / Q8_0_BLOCK_ELEMS;
        let mut q = vec![0i8; n_blocks * Q8_0_BLOCK_ELEMS];
        let mut d = vec![0f32; n_blocks];
        for (b, chunk) in x.as_chunks::<Q8_0_BLOCK_ELEMS>().0.iter().enumerate() {
            let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            d[b] = scale;
            let base = b * Q8_0_BLOCK_ELEMS;
            for (i, &v) in chunk.iter().enumerate() {
                let qi = (v * inv).round();
                q[base + i] = qi.clamp(-127.0, 127.0) as i8;
            }
        }
        assert_eq!(got.q, q);
        assert_eq!(got.d, d);
    }

    #[test]
    fn quantize_activations_q8_k_parallel_matches_serial() {
        let x: Vec<f32> = (0..1024)
            .map(|i| ((i as f32) * 0.05 - 12.0).cos() * 1.7)
            .collect();
        let got = quantize_activations_q8_k(&x);
        let n_blocks = x.len() / Q4_K_BLOCK_ELEMS;
        let mut q = vec![0i8; n_blocks * Q4_K_BLOCK_ELEMS];
        let mut d = vec![0f32; n_blocks];
        let mut bsums = vec![0i16; n_blocks * 16];
        for (b, chunk) in x.as_chunks::<Q4_K_BLOCK_ELEMS>().0.iter().enumerate() {
            let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            d[b] = scale;
            let base = b * Q4_K_BLOCK_ELEMS;
            for (i, &v) in chunk.iter().enumerate() {
                let qi = (v * inv).round();
                q[base + i] = qi.clamp(-127.0, 127.0) as i8;
            }
            let bsum_base = b * 16;
            for g in 0..16 {
                let mut s = 0i32;
                let off = base + g * 16;
                for i in 0..16 {
                    s += q[off + i] as i32;
                }
                bsums[bsum_base + g] = s as i16;
            }
        }
        assert_eq!(got.q, q);
        assert_eq!(got.d, d);
        assert_eq!(got.bsums, bsums);
    }

    #[test]
    fn dot_q4_k_q8_matches_scalar_and_tracks_float_dot() {
        let n_blocks = 3;
        let cols = n_blocks * Q4_K_BLOCK_ELEMS;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.017 - 2.1).sin() * 1.8)
            .collect();
        // Build a synthetic Q4_K row via quantize then re-pack? Use dequant
        // round-trip: quantize floats with a simple pattern into Q4_K by
        // packing known nibbles (same as other K-quant tests).
        let mut weights = Vec::with_capacity(n_blocks * Q4_K_BLOCK_BYTES);
        for b in 0..n_blocks {
            weights.extend_from_slice(&f16::from_f32(0.05 + b as f32 * 0.01).to_le_bytes());
            weights.extend_from_slice(&f16::from_f32(0.01 + b as f32 * 0.002).to_le_bytes());
            // 12 scale bytes: simple low-6-bit pattern
            for i in 0..12u8 {
                weights.push(20 + i.wrapping_mul(3));
            }
            for i in 0..128u8 {
                weights.push(i.wrapping_mul(17).wrapping_add(b as u8));
            }
        }
        let act = quantize_activations_q8_k(&x);
        let dispatched = dot_q4_k_q8(&weights, &act);
        let scalar = dot_q4_k_q8_scalar(&weights, &act);
        assert_eq!(dispatched, scalar, "dispatch must match scalar");
        let float_dot = dot_q4_k_f32(&weights, &x);
        let err = (dispatched - float_dot).abs();
        let scale = float_dot.abs().max(1.0);
        assert!(
            err / scale < 0.05,
            "int-dot vs f32 relative err {err}/{scale} too large (int={dispatched} f32={float_dot})"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn dot_q4_k_q8_i8mm_matches_scalar_when_available() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }
        let n_blocks = 3;
        let cols = n_blocks * Q4_K_BLOCK_ELEMS;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.017 - 2.1).sin() * 1.8)
            .collect();
        let mut weights = Vec::with_capacity(n_blocks * Q4_K_BLOCK_BYTES);
        for b in 0..n_blocks {
            weights.extend_from_slice(&f16::from_f32(0.05 + b as f32 * 0.01).to_le_bytes());
            weights.extend_from_slice(&f16::from_f32(0.01 + b as f32 * 0.002).to_le_bytes());
            for i in 0..12u8 {
                weights.push(20 + i.wrapping_mul(3));
            }
            for i in 0..128u8 {
                weights.push(i.wrapping_mul(17).wrapping_add(b as u8));
            }
        }
        let act = quantize_activations_q8_k(&x);
        let scalar = dot_q4_k_q8_scalar(&weights, &act);
        let i8mm = unsafe { simd_aarch64::dot_q4_k_q8_neon_i8mm(&weights, &act) };
        assert_eq!(i8mm, scalar, "i8mm must match scalar");
        let dispatched = dot_q4_k_q8(&weights, &act);
        assert_eq!(
            dispatched, scalar,
            "dispatch must match scalar on i8mm host"
        );
    }

    #[test]
    fn dot_q5_k_q8_matches_scalar_and_tracks_float_dot() {
        let x: Vec<f32> = (0..Q5_K_BLOCK_ELEMS)
            .map(|i| ((i as f32) * 0.013 - 1.7).sin() * 1.5)
            .collect();
        let act = quantize_activations_q8_k(&x);
        let dispatched = dot_q5_k_q8(&Q5_K_TEST_BLOCK, &act);
        let scalar = dot_q5_k_q8_scalar(&Q5_K_TEST_BLOCK, &act);
        assert_eq!(dispatched, scalar, "dispatch must match scalar");
        let float_dot = dot_q5_k_f32(&Q5_K_TEST_BLOCK, &x);
        let err = (dispatched - float_dot).abs();
        let scale = float_dot.abs().max(1.0);
        assert!(
            err / scale < 0.05,
            "Q5_K int-dot vs f32 relative err {err}/{scale} (int={dispatched} f32={float_dot})"
        );
    }

    #[test]
    fn gemm_q5_k_q8_row_matches_per_act_dots() {
        let acts: Vec<_> = (0..Q5_K_GEMM_NC)
            .map(|j| {
                let x: Vec<f32> = (0..Q5_K_BLOCK_ELEMS)
                    .map(|i| ((i as f32) * 0.013 - 1.7 + j as f32).sin() * 1.5)
                    .collect();
                quantize_activations_q8_k(&x)
            })
            .collect();
        let mut out = vec![0f32; acts.len()];
        gemm_q5_k_q8_row(&Q5_K_TEST_BLOCK, &acts, &mut out);
        for (j, act) in acts.iter().enumerate() {
            let want = dot_q5_k_q8(&Q5_K_TEST_BLOCK, act);
            let err = (out[j] - want).abs();
            assert!(
                err < 1e-4,
                "act {j}: gemm {got} vs dot {want}",
                got = out[j]
            );
        }
    }

    #[test]
    fn gemm_q6_k_q8_row_matches_per_act_dots() {
        let acts: Vec<_> = (0..Q6_K_GEMM_NC)
            .map(|j| {
                let x: Vec<f32> = (0..Q6_K_BLOCK_ELEMS)
                    .map(|i| ((i as f32) * 0.011 - 0.9 + j as f32).cos() * 1.9)
                    .collect();
                quantize_activations_q8_k(&x)
            })
            .collect();
        let mut out = vec![0f32; acts.len()];
        gemm_q6_k_q8_row(&Q6_K_TEST_BLOCK, &acts, &mut out);
        for (j, act) in acts.iter().enumerate() {
            let want = dot_q6_k_q8(&Q6_K_TEST_BLOCK, act);
            let err = (out[j] - want).abs();
            assert!(
                err < 1e-3,
                "act {j}: gemm {got} vs dot {want}",
                got = out[j]
            );
        }
    }

    #[test]
    fn dot_q6_k_q8_matches_scalar_and_tracks_float_dot() {
        let x: Vec<f32> = (0..Q6_K_BLOCK_ELEMS)
            .map(|i| ((i as f32) * 0.011 - 0.9).cos() * 1.9)
            .collect();
        let act = quantize_activations_q8_k(&x);
        let dispatched = dot_q6_k_q8(&Q6_K_TEST_BLOCK, &act);
        let scalar = dot_q6_k_q8_scalar(&Q6_K_TEST_BLOCK, &act);
        assert_eq!(dispatched, scalar, "dispatch must match scalar");
        let float_dot = dot_q6_k_f32(&Q6_K_TEST_BLOCK, &x);
        let err = (dispatched - float_dot).abs();
        let scale = float_dot.abs().max(1.0);
        assert!(
            err / scale < 0.05,
            "Q6_K int-dot vs f32 relative err {err}/{scale} (int={dispatched} f32={float_dot})"
        );
    }

    #[test]
    fn dot_q8_0_q8_dispatch_matches_scalar_and_float_dot() {
        // Random-ish Q8_0 weight row + activations; the integer dot must
        // equal its own scalar path exactly and the float dot closely.
        let n_blocks = 5;
        let cols = n_blocks * Q8_0_BLOCK_ELEMS;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.019 - 1.3).cos() * 2.7)
            .collect();

        let mut weights = Vec::with_capacity(n_blocks * Q8_0_BLOCK_BYTES);
        for b in 0..n_blocks {
            weights.extend_from_slice(&f16::from_f32(0.021 + b as f32 * 0.004).to_le_bytes());
            for i in 0..Q8_0_BLOCK_ELEMS {
                weights.push(((i as i32 * 7 + b as i32 * 3) % 255 - 127) as i8 as u8);
            }
        }

        let act = quantize_activations_q8(&x);
        let dispatched = dot_q8_0_q8(&weights, &act);
        let scalar = dot_q8_0_q8_scalar(&weights, &act);
        assert_eq!(
            dispatched.to_bits(),
            scalar.to_bits(),
            "SIMD int dot must match scalar int dot bit-for-bit"
        );

        let float_dot = dot_q8_0_f32(&weights, &x);
        // Activation quant error is ~amax/127 per element; the aggregate
        // relative error stays small for this many terms.
        let rel = (dispatched - float_dot).abs() / float_dot.abs().max(1e-6);
        assert!(
            rel < 0.02,
            "int dot {dispatched} vs float {float_dot} rel={rel}"
        );
    }

    #[test]
    fn dot_q4_0_q8_dispatch_matches_scalar_and_float_dot() {
        let n_blocks = 5;
        let cols = n_blocks * Q4_0_BLOCK_ELEMS;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.019 - 1.3).cos() * 2.7)
            .collect();

        let mut weights = Vec::with_capacity(n_blocks * Q4_0_BLOCK_BYTES);
        for b in 0..n_blocks {
            weights.extend_from_slice(&f16::from_f32(0.021 + b as f32 * 0.004).to_le_bytes());
            for i in 0..16 {
                weights.push(((i as u32 * 13 + b as u32 * 7) % 256) as u8);
            }
        }

        let act = quantize_activations_q8(&x);
        let dispatched = dot_q4_0_q8(&weights, &act);
        let scalar = dot_q4_0_q8_scalar(&weights, &act);
        assert_eq!(
            dispatched.to_bits(),
            scalar.to_bits(),
            "SIMD Q4_0 int dot must match scalar bit-for-bit"
        );

        let float_dot = dot_q4_0_f32(&weights, &x);
        let rel = (dispatched - float_dot).abs() / float_dot.abs().max(1e-6);
        assert!(
            rel < 0.03,
            "Q4_0 int dot {dispatched} vs float {float_dot} rel={rel}"
        );
    }

    #[test]
    fn q4_0_zero_nibble_maps_to_negative_bias() {
        // scale = 1.0, nibble 0 -> (0 - 8) * scale = -8.0
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&[0u8; 16]); // all nibbles zero
        let out = dequant_q4_0(&block).unwrap();
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| v == -8.0));
    }

    #[test]
    fn rejects_misaligned_buffers() {
        let bad = vec![0u8; 5];
        assert!(dequant_q8_0(&bad).is_err());
        assert!(dequant_q4_0(&bad).is_err());
    }

    #[test]
    fn q4_1_affine_nibble_maps_to_scale_plus_min() {
        // d=2.0, m=5.0, nibble=1 (both halves of every byte) ->
        // 1*2+5 = 7.0 for every element.
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(2.0).to_le_bytes());
        block.extend_from_slice(&f16::from_f32(5.0).to_le_bytes());
        block.extend_from_slice(&[0x11u8; 16]); // lo=1, hi=1
        let out = dequant_q4_1(&block).unwrap();
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| (v - 7.0).abs() < 1e-6));
    }

    #[test]
    fn q5_0_fifth_bit_extends_range_past_a_plain_nibble() {
        // d=1.0, qs nibble=0, but qh sets bit 0 (affects element 0's
        // low nibble): x0 = (0 | 16) - 16 = 0 still (5th bit set
        // brings it back to the *middle* of the 5-bit range, unlike a
        // 4-bit nibble's max of 15 -8=7). Pick a qh bit that's
        // unambiguous: set bit 1 (element j=1's low nibble) instead,
        // -> x = (0|16)-16 = 0... use a clearer case: nibble=15,
        // qh bit set -> x = (15|16)-16 = 31-16 = 15 (16|15=31 since
        // bits don't overlap: nibble uses bits 0-3, 5th bit is bit 4).
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let mut qh = [0u8; 4];
        qh[0] |= 1 << 0; // sets bit 0 of qh -> element j=0's 5th bit
        block.extend_from_slice(&qh);
        let mut qs = [0u8; 16];
        qs[0] = 0x0F; // low nibble = 15 for element 0
        block.extend_from_slice(&qs);
        let out = dequant_q5_0(&block).unwrap();
        assert_eq!(out.len(), 32);
        // element 0: nibble=15, 5th bit set -> q=15|16=31, x=31-16=15
        assert_eq!(out[0], 15.0);
        // every other element: nibble=0, no 5th bit -> q=0, x=0-16=-16
        assert_eq!(out[1], -16.0);
    }

    #[test]
    fn q5_1_fifth_bit_without_bias_subtraction() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
        let mut qh = [0u8; 4];
        qh[0] |= 1 << 0;
        block.extend_from_slice(&qh);
        let mut qs = [0u8; 16];
        qs[0] = 0x0F;
        block.extend_from_slice(&qs);
        let out = dequant_q5_1(&block).unwrap();
        assert_eq!(out.len(), 32);
        // element 0: q = 15|16 = 31, x = 31*1+0 = 31 (no -16 bias)
        assert_eq!(out[0], 31.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn q8_1_matches_q8_0_math_ignoring_the_extra_sum_field() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block.extend_from_slice(&f16::from_f32(999.0).to_le_bytes()); // s: must be ignored
        let qs: Vec<i8> = (0..32).map(|i| i - 16).collect();
        block.extend_from_slice(&i8_to_u8_bytes(&qs));
        let out = dequant_q8_1(&block).unwrap();
        assert_eq!(out.len(), 32);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i as f32 - 16.0) * 0.5);
        }
    }

    /// Test-only `i8` -> `u8` byte reinterpretation; `i8`/`u8` share
    /// layout, so this is just a bit-pattern-preserving cast per
    /// element.
    fn i8_to_u8_bytes(src: &[i8]) -> Vec<u8> {
        src.iter().map(|&b| b as u8).collect()
    }

    #[test]
    fn legacy_formats_fused_dot_matches_dequant_then_dot() {
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.07).sin()).collect();

        let mut q4_1 = Vec::new();
        q4_1.extend_from_slice(&f16::from_f32(0.3).to_le_bytes());
        q4_1.extend_from_slice(&f16::from_f32(-1.2).to_le_bytes());
        q4_1.extend_from_slice(
            &(0..16)
                .map(|i| (i as u8) | ((15 - i as u8) << 4))
                .collect::<Vec<u8>>(),
        );
        let expected: f32 = dequant_q4_1(&q4_1)
            .unwrap()
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
        let fused = dot_q4_1_f32(&q4_1, &x);
        assert!(
            (fused - expected).abs() < 1e-3,
            "Q4_1: fused={fused} expected={expected}"
        );

        let mut q5_0 = Vec::new();
        q5_0.extend_from_slice(&f16::from_f32(0.4).to_le_bytes());
        q5_0.extend_from_slice(&[0xA5, 0x3C, 0x00, 0xFF]);
        q5_0.extend_from_slice(
            &(0..16)
                .map(|i| (i as u8) | ((15 - i as u8) << 4))
                .collect::<Vec<u8>>(),
        );
        let expected: f32 = dequant_q5_0(&q5_0)
            .unwrap()
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
        let fused = dot_q5_0_f32(&q5_0, &x);
        assert!(
            (fused - expected).abs() < 1e-3,
            "Q5_0: fused={fused} expected={expected}"
        );

        let mut q5_1 = Vec::new();
        q5_1.extend_from_slice(&f16::from_f32(0.2).to_le_bytes());
        q5_1.extend_from_slice(&f16::from_f32(0.9).to_le_bytes());
        q5_1.extend_from_slice(&[0x12, 0x34, 0x56, 0x78]);
        q5_1.extend_from_slice(
            &(0..16)
                .map(|i| (i as u8) | ((15 - i as u8) << 4))
                .collect::<Vec<u8>>(),
        );
        let expected: f32 = dequant_q5_1(&q5_1)
            .unwrap()
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
        let fused = dot_q5_1_f32(&q5_1, &x);
        assert!(
            (fused - expected).abs() < 1e-3,
            "Q5_1: fused={fused} expected={expected}"
        );

        let mut q8_1 = Vec::new();
        q8_1.extend_from_slice(&f16::from_f32(0.6).to_le_bytes());
        q8_1.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
        let qs: Vec<i8> = (0..32).map(|i| ((i * 7) % 61) as i8 - 30).collect();
        q8_1.extend_from_slice(&i8_to_u8_bytes(&qs));
        let expected: f32 = dequant_q8_1(&q8_1)
            .unwrap()
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
        let fused = dot_q8_1_f32(&q8_1, &x);
        assert!(
            (fused - expected).abs() < 1e-3,
            "Q8_1: fused={fused} expected={expected}"
        );
    }

    #[test]
    fn legacy_formats_reject_misaligned_buffers() {
        let bad = vec![0u8; 5];
        assert!(dequant_q4_1(&bad).is_err());
        assert!(dequant_q5_0(&bad).is_err());
        assert!(dequant_q5_1(&bad).is_err());
        assert!(dequant_q8_1(&bad).is_err());
    }

    #[test]
    fn bf16_widening_is_exact_for_round_values() {
        // Values with zero low-mantissa bits round-trip through
        // f32->bf16 truncation exactly, so this is a real equality
        // check, not an approximate one.
        for v in [0.0f32, 1.0, -1.0, 2.5, -0.5, 100.0, -100.0] {
            let bf16_bits = (v.to_bits() >> 16) as u16;
            let bytes = bf16_bits.to_le_bytes();
            let restored = dequant_bf16(&bytes).unwrap();
            assert_eq!(restored, vec![v], "bf16 round-trip mismatch for {v}");
        }
    }

    #[test]
    fn bf16_widening_matches_hand_computed_bits() {
        // 1.0f32 = 0x3F800000; its bf16 truncation is the top 16 bits,
        // 0x3F80. Widening back must reproduce exactly 0x3F800000.
        let bytes = 0x3F80u16.to_le_bytes();
        let out = dequant_bf16(&bytes).unwrap();
        assert_eq!(out, vec![1.0f32]);
        assert_eq!(out[0].to_bits(), 0x3F800000);
    }

    #[test]
    fn bf16_rejects_odd_length_buffers() {
        let bad = vec![0u8; 3];
        assert!(dequant_bf16(&bad).is_err());
    }

    #[test]
    fn f16_widening_is_exact_and_covers_the_special_values() {
        // Every f16 is exactly representable in f32, so equality holds
        // for all finite inputs -- including subnormals, which a naive
        // shift-based widening gets wrong.
        let subnormal = f16::from_bits(0x0001); // 2^-24, smallest f16 subnormal
        let cases: Vec<f16> = [0.0f32, -0.0, 1.0, -1.0, 2.5, -0.5, 65504.0, -65504.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .chain(std::iter::once(subnormal))
            .collect();
        let bytes: Vec<u8> = cases.iter().flat_map(|h| h.to_le_bytes()).collect();
        let out = dequant_f16(&bytes).unwrap();
        assert_eq!(out.len(), cases.len());
        for (got, want) in out.iter().zip(cases.iter()) {
            assert_eq!(got.to_bits(), want.to_f32().to_bits());
        }
        assert_eq!(out[8], 2f32.powi(-24));

        // Infinity survives; f16 max (65504) is not clamped.
        let inf = f16::INFINITY.to_le_bytes();
        assert!(dequant_f16(&inf).unwrap()[0].is_infinite());
    }

    #[test]
    fn f16_rejects_odd_length_buffers() {
        let bad = vec![0u8; 5];
        assert!(dequant_f16(&bad).is_err());
    }

    #[test]
    fn fused_q8_0_dot_matches_dequant_then_dot() {
        let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.37).collect();
        let packed = quantize_q8_0(&original);
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01 - 0.16).collect();

        let dequanted = dequant_q8_0(&packed).unwrap();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let fused = dot_q8_0_f32(&packed, &x);
        assert!(
            (fused - expected).abs() < 1e-3,
            "fused={fused} expected={expected}"
        );
    }

    #[test]
    fn dispatched_dot_matches_scalar_reference_across_many_blocks() {
        // 5 blocks (160 elements) so the test exercises multiple
        // AVX2 iterations, not just one, and uses varied values
        // (including negatives and zero) to catch sign-extension bugs
        // in the SIMD path specifically.
        let n_blocks = 5;
        let original: Vec<f32> = (0..n_blocks * 32)
            .map(|i| ((i as f32) - (n_blocks * 16) as f32) * 0.29)
            .collect();
        let packed = quantize_q8_0(&original);
        let x: Vec<f32> = (0..n_blocks * 32)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();

        let dispatched = dot_q8_0_f32(&packed, &x);
        let scalar = dot_q8_0_f32_scalar(&packed, &x);
        assert!(
            (dispatched - scalar).abs() < 1e-2,
            "dispatched={dispatched} scalar={scalar} (should match regardless of which SIMD path the host CPU takes)"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernel_matches_scalar_directly_when_available() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipping: host CPU lacks AVX2/FMA");
            return;
        }
        let n_blocks = 8;
        let original: Vec<f32> = (0..n_blocks * 32)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.11)
            .collect();
        let packed = quantize_q8_0(&original);
        let x: Vec<f32> = (0..n_blocks * 32)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();

        let simd = unsafe { simd_x86::dot_q8_0_f32_avx2(&packed, &x) };
        let scalar = dot_q8_0_f32_scalar(&packed, &x);
        assert!(
            (simd - scalar).abs() < 1e-2,
            "AVX2 kernel diverged from scalar: simd={simd} scalar={scalar}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_q4_0_kernel_matches_scalar_directly_when_available() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipping: host CPU lacks AVX2/FMA");
            return;
        }
        // Build several Q4_0 blocks with varied nibble patterns
        // (including 0x0, 0xF, and mixed) to exercise both the low-
        // and high-nibble extraction paths and the -8 bias at both
        // extremes.
        let n_blocks = 6;
        let mut packed = Vec::new();
        for b in 0..n_blocks {
            packed.extend_from_slice(&half::f16::from_f32(0.05 + b as f32 * 0.01).to_le_bytes());
            for i in 0..16u8 {
                let lo = (i + b as u8) % 16;
                let hi = (15 - i + b as u8) % 16;
                packed.push(lo | (hi << 4));
            }
        }
        let x: Vec<f32> = (0..n_blocks * 32)
            .map(|i| ((i as f32) * 0.09).sin())
            .collect();

        let simd = unsafe { simd_x86::dot_q4_0_f32_avx2(&packed, &x) };
        let scalar = dot_q4_0_f32_scalar(&packed, &x);
        assert!(
            (simd - scalar).abs() < 1e-2,
            "AVX2 Q4_0 kernel diverged from scalar: simd={simd} scalar={scalar}"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_kernel_matches_scalar_directly_when_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipping: host CPU lacks NEON (unexpected on real aarch64 hardware)");
            return;
        }
        let n_blocks = 8;
        let original: Vec<f32> = (0..n_blocks * 32)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.11)
            .collect();
        let packed = quantize_q8_0(&original);
        let x: Vec<f32> = (0..n_blocks * 32)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();

        let simd = unsafe { simd_aarch64::dot_q8_0_f32_neon(&packed, &x) };
        let scalar = dot_q8_0_f32_scalar(&packed, &x);
        assert!(
            (simd - scalar).abs() < 1e-2,
            "NEON kernel diverged from scalar: simd={simd} scalar={scalar}"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_q4_0_kernel_matches_scalar_directly_when_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipping: host CPU lacks NEON (unexpected on real aarch64 hardware)");
            return;
        }
        // Build several Q4_0 blocks with varied nibble patterns
        // (including 0x0, 0xF, and mixed) to exercise both the low-
        // and high-nibble extraction paths and the -8 bias at both
        // extremes.
        let n_blocks = 6;
        let mut packed = Vec::new();
        for b in 0..n_blocks {
            packed.extend_from_slice(&half::f16::from_f32(0.05 + b as f32 * 0.01).to_le_bytes());
            for i in 0..16u8 {
                let lo = (i + b as u8) % 16;
                let hi = (15 - i + b as u8) % 16;
                packed.push(lo | (hi << 4));
            }
        }
        let x: Vec<f32> = (0..n_blocks * 32)
            .map(|i| ((i as f32) * 0.09).sin())
            .collect();

        let simd = unsafe { simd_aarch64::dot_q4_0_f32_neon(&packed, &x) };
        let scalar = dot_q4_0_f32_scalar(&packed, &x);
        assert!(
            (simd - scalar).abs() < 1e-2,
            "NEON Q4_0 kernel diverged from scalar: simd={simd} scalar={scalar}"
        );
    }

    #[test]
    fn dispatched_q4_0_matches_scalar_reference() {
        let n_blocks = 4;
        let mut packed = Vec::new();
        for b in 0..n_blocks {
            packed.extend_from_slice(&half::f16::from_f32(0.2).to_le_bytes());
            for i in 0..16u8 {
                packed.push((i % 16) | (((15 - i + b as u8) % 16) << 4));
            }
        }
        let x: Vec<f32> = (0..n_blocks * 32)
            .map(|i| (i as f32) * 0.02 - 1.0)
            .collect();

        let dispatched = dot_q4_0_f32(&packed, &x);
        let scalar = dot_q4_0_f32_scalar(&packed, &x);
        assert!(
            (dispatched - scalar).abs() < 1e-2,
            "dispatched={dispatched} scalar={scalar}"
        );
    }

    #[test]
    fn fused_q4_0_dot_matches_dequant_then_dot() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&[0x12u8; 16]); // arbitrary nibble pattern
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

        let dequanted = dequant_q4_0(&block).unwrap();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_q4_0_f32(&block, &x);
        assert!(
            (fused - expected).abs() < 1e-3,
            "fused={fused} expected={expected}"
        );
    }

    // Cross-validation data generated by an independent Python
    // implementation of the Q4_K/Q6_K public
    // block-quantization formats, written from the same public layout
    // description as the Rust code above but not derived from it.
    // Generated by an independent Python reference -- do not hand-edit.
    const Q4_K_TEST_BLOCK: [u8; 144] = [
        0x66, 0x2a, 0x66, 0x2a, 0x02, 0x02, 0x02, 0x02, 0x4f, 0x4b, 0x10, 0x12, 0x42, 0xe4, 0xc1,
        0xb2, 0x64, 0xa8, 0x70, 0x2d, 0x6a, 0xa6, 0x76, 0x79, 0xa6, 0xf7, 0x5a, 0xda, 0x37, 0x87,
        0x38, 0xd5, 0xf9, 0xfa, 0xc2, 0x98, 0x33, 0x94, 0x48, 0x59, 0x46, 0x73, 0xb2, 0x3b, 0x28,
        0x18, 0x2e, 0x02, 0xe4, 0x5d, 0x86, 0xa9, 0x93, 0x39, 0x51, 0x75, 0x5f, 0xb6, 0xac, 0x0a,
        0x17, 0x35, 0x8d, 0xf7, 0x97, 0x7a, 0x95, 0xf5, 0x51, 0xc9, 0xdd, 0xb8, 0xdf, 0x7a, 0x69,
        0xdb, 0xcb, 0xfe, 0xa6, 0xf0, 0x69, 0xf6, 0xf2, 0xc6, 0xad, 0xb4, 0x68, 0x9f, 0xad, 0x7f,
        0xd6, 0x40, 0x8f, 0x14, 0xca, 0xdb, 0xa9, 0x7d, 0x89, 0xb6, 0xad, 0x96, 0xa9, 0x69, 0x96,
        0xaa, 0x98, 0x79, 0x06, 0x9a, 0x86, 0x74, 0xff, 0xde, 0x8e, 0xf0, 0xf0, 0x3f, 0xcd, 0xdd,
        0x7d, 0x7f, 0x0c, 0x3d, 0x0e, 0x7f, 0x88, 0x8f, 0xf7, 0x95, 0x83, 0x13, 0x11, 0x85, 0x55,
        0x0c, 0x5c, 0x7b, 0x9e, 0x51, 0x48, 0x69, 0x67, 0x1e,
    ];
    const Q4_K_GOLDEN: [f32; 256] = [
        -0.349915, 0.0499878, -0.749817, 0.549866, 0.249939, -0.149963, -0.149963, 0.149963,
        -0.149963, -0.0499878, 0.249939, 0.249939, -0.0499878, -0.0499878, 0.0499878, -0.249939,
        0.149963, 0.249939, -0.549866, 0.0499878, -0.44989, -0.349915, 0.0499878, 0.149963,
        -0.149963, -0.44989, -0.549866, 0.349915, 0.0499878, 0.0499878, 0.649841, -0.549866,
        0.0499878, 0.44989, 0.149963, -0.349915, 0.0499878, 0.44989, 0.149963, 0.149963, 0.44989,
        0.949768, -0.0499878, 0.749817, -0.249939, 0.249939, -0.249939, 0.749817, 0.949768,
        0.949768, 0.649841, 0.349915, -0.249939, 0.349915, -0.149963, -0.0499878, -0.149963,
        0.149963, 0.549866, -0.249939, -0.349915, -0.44989, -0.349915, -0.549866, -0.399902,
        0.499878, -0.199951, 0.0999756, -0.499878, 0.0999756, -0.699829, -0.299927, 0.699829,
        -0.199951, 0.399902, 0.199951, -0.0999756, -0.299927, 0.499878, -0.0999756, -0.0999756,
        0.199951, -0.299927, -0.299927, -0.699829, 0.0999756, 0.499878, 0.0, 0.699829, 0.199951,
        0.0999756, 0.299927, 0.299927, 0.599854, -0.199951, -0.799805, 0.499878, -0.399902,
        -0.0999756, 0.0999756, 0.0, -0.599854, -0.399902, -0.199951, -0.399902, 0.199951,
        0.0999756, -0.89978, -0.799805, -0.599854, -0.0999756, 0.599854, 0.0, -0.199951, 0.0,
        0.599854, -0.399902, 0.299927, 0.399902, 0.199951, 0.399902, -0.199951, -0.299927,
        0.399902, 0.299927, 0.599854, 0.0999756, 0.599854, -0.0999756, -0.399902, -0.799805,
        -0.399902, 0.299927, -0.599854, -0.199951, 0.499878, 0.299927, 0.499878, -0.399902,
        -0.999756, 0.499878, -0.599854, 0.0, 0.0999756, -0.0999756, 0.299927, -0.0999756,
        -0.399902, 0.299927, -0.399902, -0.0999756, -0.0999756, -0.399902, 0.0, -0.199951,
        -0.0999756, -0.399902, 0.0, -0.399902, -0.599854, -0.299927, 1.49963, 1.49963, 0.89978,
        0.499878, 0.699829, -0.299927, 0.299927, 0.499878, -0.0999756, 1.09973, -0.699829,
        0.0999756, -1.29968, 0.89978, 1.09973, 0.499878, -0.0999756, 0.0999756, 0.699829, 0.499878,
        0.299927, 0.499878, -0.299927, 0.299927, 0.499878, 0.299927, -0.0999756, -1.49963,
        0.299927, 0.0999756, -0.0999756, 0.149963, 0.0999756, 0.0999756, -0.599854, -0.599854,
        0.149963, 0.0499878, 0.0499878, 0.0499878, 0.149963, 0.0, 0.0499878, 0.0999756, 0.149963,
        -0.199951, 0.149963, -0.249939, -0.349915, -0.44989, -0.44989, -0.549866, -0.349915,
        -0.349915, 0.0, 0.0, -0.0499878, 0.0999756, -0.549866, -0.199951, -0.149963, -0.249939,
        0.0999756, 0.949768, 0.749817, 0.249939, 0.949768, 0.949768, -0.249939, 0.649841, 0.749817,
        0.149963, 0.149963, -0.549866, -0.249939, -0.549866, 0.149963, 0.249939, 0.249939,
        0.949768, 0.349915, 0.249939, -0.44989, -0.44989, 0.249939, -0.0499878, -0.549866,
        -0.0499878, 0.149963, 0.349915, -0.0499878, -0.149963, 0.0499878, 0.0499878, -0.44989,
    ];

    // Generated by an independent Python reference -- do not hand-edit.
    #[rustfmt::skip]
    const Q5_K_TEST_BLOCK: [u8; 176] = [
        0x66, 0x2a, 0x66, 0x2a, 0x01, 0x01, 0x01, 0x01, 0x4f, 0x4b, 0x10, 0x12, 0x41, 0xe2, 0xc1,
        0xb1, 0x72, 0x2f, 0x20, 0x07, 0x31, 0x0c, 0x38, 0xb3, 0x9c, 0xb8, 0xad, 0x2f, 0x9a, 0xea,
        0x17, 0xd0, 0xee, 0x93, 0x9e, 0x3e, 0x74, 0xbb, 0x28, 0x18, 0x39, 0x25, 0xb6, 0x09, 0x18,
        0x29, 0x1c, 0x1d, 0x29, 0x41, 0x40, 0x0a, 0x74, 0x7d, 0xfd, 0x21, 0xdd, 0x6d, 0x45, 0x73,
        0x0e, 0x1e, 0xc0, 0x4a, 0xfc, 0xf3, 0x8e, 0x24, 0x6b, 0x34, 0x7d, 0xbe, 0x94, 0xde, 0x59,
        0x7a, 0x35, 0x30, 0x36, 0x0a, 0xf9, 0x4a, 0x9b, 0xa2, 0x26, 0x21, 0xa2, 0xfa, 0xdf, 0x4b,
        0x29, 0x64, 0x6f, 0xbb, 0xca, 0x0f, 0x3c, 0xda, 0x20, 0xf4, 0x93, 0x86, 0xab, 0x6e, 0xb9,
        0xe5, 0xd5, 0xa0, 0x82, 0xd6, 0x41, 0xff, 0x12, 0xbc, 0x34, 0xbb, 0xab, 0xb8, 0x20, 0x2f,
        0xbb, 0x5f, 0x0c, 0x10, 0xcf, 0x49, 0xc5, 0x86, 0x5c, 0xdf, 0xff, 0x78, 0x44, 0x26, 0x3b,
        0xc2, 0x23, 0x3d, 0x2b, 0xe9, 0x00, 0x12, 0xf8, 0xea, 0xe2, 0x9e, 0x5e, 0x50, 0x20, 0x9f,
        0x9d, 0x8d, 0x7d, 0x7f, 0xcc, 0x1d, 0x0e, 0x13, 0xf8, 0xc2, 0xf1, 0x3d, 0x08, 0x2f, 0x23,
        0x13, 0xac, 0x0d, 0xa7, 0xe7, 0x20, 0xa3, 0x90, 0xb7, 0xc8, 0x28,
    ];
    const Q5_K_GOLDEN: [f32; 256] = [
        -0.299927, 0.0999756, -0.749817, 0.549866, 0.249939, -0.0999756, -0.0999756, 0.0999756,
        -0.0999756, -0.0999756, 0.299927, 0.199951, -0.0499878, -0.0499878, 0.0499878, -0.249939,
        -0.149963, 0.199951, -0.0499878, -0.549866, -0.199951, 0.249939, -0.0999756, -0.0499878,
        0.249939, 0.749817, -0.299927, 0.549866, -0.499878, 0.0499878, -0.44989, 0.549866,
        0.349915, 0.44989, -0.349915, 0.249939, -0.199951, -0.199951, 0.199951, 0.349915,
        0.0999756, -0.249939, -0.349915, 0.599854, 0.249939, 0.299927, 0.849792, -0.349915,
        0.999756, 0.999756, 0.649841, 0.349915, -0.249939, 0.399902, -0.199951, 0.0, -0.0999756,
        0.0999756, 0.499878, -0.199951, -0.399902, -0.399902, -0.399902, -0.549866, -0.349915,
        0.499878, -0.249939, 0.0999756, -0.499878, 0.0499878, -0.699829, -0.299927, 0.749817,
        -0.249939, 0.44989, 0.199951, -0.0499878, -0.249939, 0.499878, -0.0499878, 0.599854,
        -0.299927, 0.0, 0.199951, 0.149963, -0.499878, -0.249939, -0.0999756, -0.349915, 0.249939,
        0.249939, -0.799805, -0.699829, -0.499878, 0.0499878, 0.749817, -0.149963, 0.0999756,
        -0.44989, -0.399902, -0.799805, 0.0, 0.399902, -0.149963, 0.549866, 0.0999756, 0.0,
        0.199951, 0.199951, 0.44989, -0.299927, -0.89978, 0.0499878, -0.249939, 0.0, 0.649841,
        -0.44989, 0.299927, 0.399902, 0.199951, 0.44989, -0.199951, -0.249939, 0.399902, 0.299927,
        0.549866, 0.0999756, 0.649841, -0.0999756, -0.399902, -0.799805, -0.44989, 0.349915,
        -0.599854, -0.199951, 0.549866, 0.349915, 0.549866, -0.399902, -0.999756, 0.549866,
        -0.549866, 0.0499878, 0.0999756, -0.399902, 0.549866, 0.549866, 0.199951, 0.0, 0.0999756,
        -0.44989, -0.0999756, -0.0499878, -0.349915, 0.349915, -0.549866, -0.199951, -0.89978,
        0.199951, 0.299927, 0.199951, 1.19971, 0.399902, -0.399902, 1.09973, -0.399902, 0.299927,
        0.299927, -0.399902, 0.599854, 0.0999756, 0.199951, -0.299927, 0.499878, -0.299927,
        -0.699829, 0.599854, -0.199951, 0.0, 0.799805, 0.499878, 0.299927, 0.399902, -0.299927,
        0.299927, 0.399902, 0.299927, -0.0999756, -1.49963, 0.199951, 0.0, -0.0999756, 0.299927,
        0.0999756, 0.0999756, -0.599854, -0.599854, 0.149963, 0.0499878, 0.0499878, 0.0499878,
        0.149963, 0.0, 0.0499878, 0.0999756, 0.349915, -0.199951, 0.299927, 0.249939, 0.0499878,
        -0.199951, 0.149963, 0.349915, -0.44989, 0.0, 0.0499878, -0.249939, -0.249939, -0.599854,
        -0.44989, -0.599854, -0.249939, -0.199951, -0.199951, 0.149963, -0.0999756, -0.299927,
        -0.299927, -0.44989, -0.0999756, -0.0999756, 0.649841, 0.599854, 0.599854, 0.849792,
        -0.499878, 0.249939, 0.299927, 0.199951, 0.849792, 0.999756, 0.399902, 0.249939, -0.44989,
        -0.44989, 0.299927, -0.0499878, -0.549866, -0.0499878, 0.149963, 0.349915, -0.0499878,
        -0.0999756, 0.0, 0.0499878, -0.44989,
    ];

    #[test]
    fn q5_k_dequant_matches_independent_python_reference() {
        let got = dequant_q5_k(&Q5_K_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), Q5_K_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(Q5_K_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "Q5_K element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn q5_k_fused_dot_matches_dequant_then_dot() {
        let dequanted = dequant_q5_k(&Q5_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.023).sin()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_q5_k_f32(&Q5_K_TEST_BLOCK, &x);
        assert!(
            (fused - expected).abs() < 1e-2,
            "fused={fused} expected={expected}"
        );
    }

    #[test]
    fn q5_k_rejects_misaligned_buffers() {
        let bad = vec![0u8; 5];
        assert!(dequant_q5_k(&bad).is_err());
    }

    const Q6_K_TEST_BLOCK: [u8; 210] = [
        0xe0, 0xa5, 0x40, 0x5c, 0x8d, 0x3a, 0x0a, 0x26, 0xfb, 0x4b, 0x6e, 0x9a, 0xdf, 0x3e, 0xa3,
        0xc4, 0xf8, 0x2b, 0x1d, 0x95, 0x76, 0x7d, 0x3b, 0xcd, 0xfd, 0xef, 0xc2, 0x0b, 0x07, 0x63,
        0x29, 0xfb, 0x81, 0x57, 0xbe, 0xbe, 0x06, 0xf7, 0x3a, 0x92, 0xc4, 0x43, 0xff, 0xad, 0xac,
        0x7e, 0x0f, 0x00, 0x2a, 0x4f, 0xf0, 0xf8, 0xa9, 0xfa, 0x3c, 0x90, 0x6d, 0x73, 0x2d, 0x5a,
        0xe6, 0xc6, 0x46, 0xf2, 0x0d, 0x55, 0x4c, 0x25, 0x38, 0x71, 0x2b, 0x35, 0x38, 0x82, 0x16,
        0x37, 0x5f, 0x32, 0x61, 0x02, 0xdd, 0x2f, 0x6f, 0x7b, 0x1f, 0xb4, 0x1a, 0x1b, 0x3e, 0x4f,
        0x11, 0xa3, 0x17, 0x40, 0x5a, 0x5f, 0x76, 0xcd, 0x19, 0x27, 0x9b, 0xc7, 0xc8, 0xf7, 0xf7,
        0xee, 0xf4, 0x86, 0xd9, 0xfd, 0xa7, 0xfe, 0x9e, 0xac, 0x70, 0x53, 0x5b, 0x76, 0xfb, 0x39,
        0xf8, 0x4b, 0x98, 0xfe, 0xd0, 0x06, 0x21, 0x4c, 0x4d, 0xbe, 0x10, 0x2b, 0x06, 0x65, 0xc9,
        0x5e, 0xf9, 0x95, 0x72, 0xae, 0x99, 0xd9, 0x7e, 0x15, 0xbd, 0x5e, 0x6d, 0xe8, 0x25, 0x8a,
        0xd5, 0x99, 0xc6, 0x6b, 0x69, 0xc7, 0x84, 0xc6, 0xa4, 0xf7, 0xb9, 0x6d, 0x68, 0x45, 0x0e,
        0x65, 0x69, 0xeb, 0xe6, 0xeb, 0xe9, 0x28, 0xa6, 0xb9, 0x96, 0xf2, 0xe8, 0xa7, 0x9b, 0x6e,
        0x79, 0x8a, 0x68, 0x65, 0x59, 0x98, 0x8b, 0x44, 0x41, 0x98, 0x9a, 0x56, 0x01, 0x01, 0x01,
        0x02, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x02, 0x02, 0x01, 0x01, 0x01, 0x02, 0x1f, 0x25,
    ];
    const Q6_K_GOLDEN: [f32; 256] = [
        -0.320068, 0.100021, -0.640137, 0.56012, 0.260056, -0.120026, -0.120026, 0.120026,
        -0.100021, -0.100021, 0.28006, 0.200043, -0.0200043, -0.0400085, 0.0600128, -0.240051,
        -0.160034, 0.220047, -0.0600128, -0.540115, -0.200043, 0.260056, -0.100021, -0.0600128,
        0.260056, 0.620132, -0.28006, 0.540115, -0.500107, 0.0600128, -0.460098, 0.540115,
        0.340073, 0.460098, -0.360077, 0.28006, -0.200043, -0.180038, 0.200043, 0.360077,
        0.0800171, -0.260056, -0.340073, 0.580124, 0.240051, 0.28006, 0.620132, -0.320068, 1.04022,
        1.24026, 0.640137, 0.320068, -0.28006, 0.400085, -0.160034, 0.0, -0.120026, 0.120026,
        0.520111, -0.240051, -0.400085, -0.400085, -0.400085, -0.56012, -0.360077, 0.520111,
        -0.240051, 0.100021, -0.480103, 0.0600128, -0.640137, -0.28006, 0.620132, -0.240051,
        0.440094, 0.180038, -0.0600128, -0.260056, 0.520111, -0.0800171, 0.620132, -0.28006,
        0.0200043, 0.180038, 0.14003, -0.500107, -0.260056, -0.0800171, -0.340073, 0.28006,
        0.240051, -0.640137, -0.640137, -0.520111, 0.0400085, 0.620132, -0.160034, 0.100021,
        -0.42009, -0.42009, -0.640137, -0.0200043, 0.380081, -0.14003, 0.56012, 0.0800171,
        -0.0200043, 0.200043, 0.200043, 0.460098, -0.320068, -0.640137, 0.0400085, -0.240051,
        -0.0200043, 0.620132, -0.440094, 0.300064, 0.380081, 0.180038, 0.440094, -0.180038,
        -0.28006, 0.42009, 0.28006, 0.56012, 0.0800171, 0.620132, -0.120026, -0.440094, -0.800171,
        -0.440094, 0.320068, -0.600128, -0.200043, 0.840179, 0.320068, 0.720154, -0.400085,
        -1.00021, 0.600128, -0.56012, 0.0400085, 0.0800171, -0.380081, 0.620132, 0.620132,
        0.220047, -0.0200043, 0.0800171, -0.440094, -0.100021, -0.0400085, -0.340073, 0.340073,
        -0.580124, -0.180038, -0.640137, 0.200043, 0.300064, 0.240051, 1.16025, 0.360077,
        -0.360077, 1.08023, -0.360077, 0.320068, 0.28006, -0.360077, 0.56012, 0.160034, 0.240051,
        -0.28006, 0.520111, -0.360077, -0.720154, 0.56012, -0.160034, 0.0, 0.760162, 0.440094,
        0.240051, 0.440094, -0.28006, 0.320068, 0.440094, 0.320068, -0.0800171, -1.28027, 0.240051,
        0.0400085, -0.160034, 0.320068, 0.100021, 0.0800171, -0.600128, -0.580124, 0.14003,
        0.0400085, 0.0600128, 0.0600128, 0.160034, 0.0200043, 0.0600128, 0.100021, 0.380081,
        -0.200043, 0.320068, 0.260056, 0.0400085, -0.200043, 0.14003, 0.340073, -0.42009,
        0.0200043, 0.0200043, -0.260056, -0.240051, -0.620132, -0.440094, -0.620132, -0.240051,
        -0.220047, -0.220047, 0.14003, -0.0800171, -0.300064, -0.28006, -0.460098, -0.0800171,
        -0.0800171, 0.620132, 0.620132, 0.600128, 0.620132, -0.480103, 0.260056, 0.300064,
        0.200043, 0.620132, 1.00021, 0.400085, 0.28006, -0.440094, -0.440094, 0.28006, -0.0400085,
        -0.520111, -0.0400085, 0.160034, 0.360077, -0.0400085, -0.120026, 0.0, 0.0800171,
        -0.480103,
    ];

    // Generated by an independent Python reference -- do not hand-edit.
    // Same input values as Q6_K_TEST_BLOCK, but every odd sub-block
    // stores a *negative* int8 scale. Q6_K scales are signed in the
    // public format; this fixture is what distinguishes a correctly
    // signed decoder from one that reads scale bytes as unsigned
    // (-1 read as 255) -- the all-positive fixture above cannot.
    const Q6_K_SIGNED_SCALES_TEST_BLOCK: [u8; 210] = [
        0xe0, 0xa5, 0x40, 0x5c, 0x8d, 0x3a, 0x0a, 0x26, 0xfb, 0x4b, 0x6e, 0x9a, 0xdf, 0x3e, 0xa3,
        0xc4, 0x18, 0xe5, 0xf3, 0x7b, 0x9a, 0x93, 0xd5, 0x43, 0x13, 0x20, 0x4e, 0xf5, 0xf9, 0xad,
        0xe7, 0x05, 0x81, 0x57, 0xbe, 0xbe, 0x06, 0xf7, 0x3a, 0x92, 0xc4, 0x43, 0xff, 0xad, 0xac,
        0x7e, 0x0f, 0x00, 0xe6, 0xc0, 0x10, 0x08, 0x67, 0x16, 0xd4, 0x70, 0xa3, 0x9d, 0xe3, 0xb6,
        0x2a, 0x4a, 0xca, 0x0e, 0x0d, 0x55, 0x4c, 0x25, 0x38, 0x71, 0x2b, 0x35, 0x38, 0x82, 0x16,
        0x37, 0x5f, 0x32, 0x61, 0x02, 0x33, 0xe1, 0xa1, 0x95, 0xf1, 0x5c, 0xf6, 0xf5, 0xd2, 0xc1,
        0xff, 0x6d, 0xf9, 0xcf, 0xb6, 0xb1, 0x76, 0xcd, 0x19, 0x27, 0x9b, 0xc7, 0xc8, 0xf7, 0xf7,
        0xee, 0xf4, 0x86, 0xd9, 0xfd, 0xa7, 0xfe, 0x72, 0x64, 0x90, 0xbd, 0xb5, 0x9a, 0x15, 0xd7,
        0x18, 0xc5, 0x78, 0x12, 0x3f, 0x0a, 0xef, 0xc4, 0x4d, 0xbe, 0x10, 0x2b, 0x06, 0x65, 0xc9,
        0x5e, 0xf9, 0x95, 0x72, 0xae, 0x99, 0xd9, 0x7e, 0x15, 0x42, 0xa1, 0x96, 0x17, 0xda, 0x75,
        0x2a, 0x6a, 0x39, 0x94, 0x96, 0x38, 0x7b, 0x39, 0x5b, 0x08, 0xb9, 0x6d, 0x68, 0x45, 0x0e,
        0x65, 0x69, 0xeb, 0xe6, 0xeb, 0xe9, 0x28, 0xa6, 0xb9, 0x96, 0xf2, 0x17, 0x58, 0x68, 0x91,
        0x86, 0x75, 0x97, 0x9a, 0xa6, 0x67, 0x74, 0xbb, 0xbe, 0xa7, 0x65, 0xa9, 0x01, 0xff, 0x01,
        0xfe, 0x01, 0xff, 0x01, 0xff, 0x02, 0xff, 0x02, 0xfe, 0x01, 0xff, 0x01, 0xfe, 0x1f, 0x25,
    ];
    const Q6_K_SIGNED_SCALES_GOLDEN: [f32; 256] = [
        -0.320068, 0.100021, -0.640137, 0.56012, 0.260056, -0.120026, -0.120026, 0.120026,
        -0.100021, -0.100021, 0.28006, 0.200043, -0.0200043, -0.0400085, 0.0600128, -0.240051,
        -0.160034, 0.220047, -0.0600128, -0.540115, -0.200043, 0.260056, -0.100021, -0.0600128,
        0.260056, 0.640137, -0.28006, 0.540115, -0.500107, 0.0600128, -0.460098, 0.540115,
        0.340073, 0.460098, -0.360077, 0.28006, -0.200043, -0.180038, 0.200043, 0.360077,
        0.0800171, -0.260056, -0.340073, 0.580124, 0.240051, 0.28006, 0.620132, -0.320068, 1.04022,
        1.28027, 0.640137, 0.320068, -0.28006, 0.400085, -0.160034, -0.0, -0.120026, 0.120026,
        0.520111, -0.240051, -0.400085, -0.400085, -0.400085, -0.56012, -0.360077, 0.520111,
        -0.240051, 0.100021, -0.480103, 0.0600128, -0.640137, -0.28006, 0.620132, -0.240051,
        0.440094, 0.180038, -0.0600128, -0.260056, 0.520111, -0.0800171, 0.620132, -0.28006,
        0.0200043, 0.180038, 0.14003, -0.500107, -0.260056, -0.0800171, -0.340073, 0.28006,
        0.240051, -0.620132, -0.620132, -0.520111, 0.0400085, 0.640137, -0.160034, 0.100021,
        -0.42009, -0.42009, -0.640137, -0.0200043, 0.380081, -0.14003, 0.56012, 0.0800171,
        -0.0200043, 0.200043, 0.200043, 0.460098, -0.320068, -0.640137, 0.0400085, -0.240051,
        -0.0200043, 0.640137, -0.440094, 0.300064, 0.380081, 0.180038, 0.440094, -0.180038,
        -0.28006, 0.42009, 0.28006, 0.56012, 0.0800171, 0.640137, -0.120026, -0.440094, -0.800171,
        -0.440094, 0.320068, -0.600128, -0.200043, 0.840179, 0.320068, 0.720154, -0.400085,
        -1.00021, 0.600128, -0.56012, 0.0400085, 0.0800171, -0.380081, 0.620132, 0.620132,
        0.220047, -0.0200043, 0.0800171, -0.440094, -0.100021, -0.0400085, -0.340073, 0.340073,
        -0.580124, -0.180038, -0.620132, 0.200043, 0.300064, 0.240051, 1.16025, 0.360077,
        -0.360077, 1.08023, -0.360077, 0.320068, 0.28006, -0.360077, 0.56012, 0.160034, 0.240051,
        -0.28006, 0.520111, -0.360077, -0.720154, 0.56012, -0.160034, -0.0, 0.760162, 0.440094,
        0.240051, 0.440094, -0.28006, 0.320068, 0.440094, 0.320068, -0.0800171, -1.24026, 0.240051,
        0.0400085, -0.160034, 0.320068, 0.100021, 0.0800171, -0.600128, -0.580124, 0.14003,
        0.0400085, 0.0600128, 0.0600128, 0.160034, 0.0200043, 0.0600128, 0.100021, 0.380081,
        -0.200043, 0.320068, 0.260056, 0.0400085, -0.200043, 0.14003, 0.340073, -0.42009,
        0.0200043, 0.0200043, -0.260056, -0.240051, -0.620132, -0.440094, -0.620132, -0.240051,
        -0.220047, -0.220047, 0.14003, -0.0800171, -0.300064, -0.28006, -0.460098, -0.0800171,
        -0.0800171, 0.620132, 0.620132, 0.600128, 0.620132, -0.480103, 0.260056, 0.300064,
        0.200043, 0.620132, 1.00021, 0.400085, 0.28006, -0.440094, -0.440094, 0.28006, -0.0400085,
        -0.520111, -0.0400085, 0.160034, 0.360077, -0.0400085, -0.120026, -0.0, 0.0800171,
        -0.480103,
    ];

    #[test]
    fn q4_k_dequant_matches_independent_python_reference() {
        let got = dequant_q4_k(&Q4_K_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), Q4_K_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(Q4_K_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "Q4_K element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn q4_k_fused_dot_matches_dequant_then_dot() {
        let dequanted = dequant_q4_k(&Q4_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.017).sin()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_q4_k_f32(&Q4_K_TEST_BLOCK, &x);
        assert!(
            (fused - expected).abs() < 1e-2,
            "fused={fused} expected={expected}"
        );
    }

    #[test]
    fn q6_k_dequant_matches_independent_python_reference() {
        let got = dequant_q6_k(&Q6_K_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), Q6_K_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(Q6_K_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "Q6_K element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn q6_k_fused_dot_matches_dequant_then_dot() {
        let dequanted = dequant_q6_k(&Q6_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.021).cos()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_q6_k_f32(&Q6_K_TEST_BLOCK, &x);
        assert!(
            (fused - expected).abs() < 1e-2,
            "fused={fused} expected={expected}"
        );
    }

    // Generated by an independent Python reference -- do not hand-edit.
    // Random-but-well-formed blocks (any byte pattern is structurally
    // valid for these formats; `d` pinned to a small non-NaN f16).
    // The Python reference itself is cross-validated against the real
    // compiled ggml implementation.
    // Generated by an independent Python reference -- do not hand-edit.
    const IQ1_S_TEST_BLOCK: [u8; 50] = [
        0x0a, 0x2f, 0xfa, 0x06, 0x1e, 0x37, 0x6f, 0xe3, 0x62, 0xd0, 0xb6, 0xa4, 0x25, 0xae, 0x76,
        0x14, 0x72, 0x5b, 0xfa, 0x05, 0xd1, 0xf1, 0x2a, 0x4c, 0xad, 0x29, 0xae, 0xf4, 0xcf, 0x0c,
        0x96, 0x51, 0x58, 0x03, 0x6d, 0xd3, 0x10, 0x92, 0x70, 0xff, 0x61, 0x58, 0xc8, 0x30, 0x25,
        0x64, 0x49, 0x85, 0xc0, 0x24,
    ];
    const IQ1_S_GOLDEN: [f32; 256] = [
        1.05861, 1.05861, 1.05861, -0.15123, -0.15123, -0.15123, -1.36107, 1.05861, -1.36107,
        -1.36107, 1.05861, -0.15123, -1.36107, -0.15123, 1.05861, -0.15123, -0.15123, -0.15123,
        -1.36107, -0.15123, -0.15123, -0.15123, 1.05861, -0.15123, -1.36107, -0.15123, -1.36107,
        -0.15123, -0.15123, -0.15123, -0.15123, -1.36107, -0.371201, 0.288712, -0.0412445,
        0.288712, -0.0412445, -0.0412445, -0.371201, -0.371201, 0.288712, -0.0412445, -0.0412445,
        -0.0412445, -0.0412445, -0.0412445, -0.371201, -0.0412445, -0.0412445, -0.371201,
        -0.0412445, -0.0412445, -0.0412445, -0.0412445, -0.371201, -0.371201, 0.288712, 0.288712,
        0.288712, 0.288712, -0.371201, -0.371201, 0.288712, -0.371201, 1.44356, 1.44356, 1.44356,
        -1.856, 1.44356, 1.44356, -1.856, -1.856, -1.856, 1.44356, -0.206223, -1.856, -1.856,
        -0.206223, -0.206223, 1.44356, 1.44356, -0.206223, -0.206223, -0.206223, -0.206223,
        -0.206223, 1.44356, -0.206223, -0.206223, 1.44356, -1.856, 1.44356, -0.206223, -0.206223,
        1.44356, 1.44356, 0.15123, 1.36107, 1.36107, 1.36107, 1.36107, 0.15123, 0.15123, -1.05861,
        0.15123, 0.15123, -1.05861, 1.36107, 0.15123, 0.15123, 0.15123, 0.15123, 1.36107, 1.36107,
        -1.05861, 1.36107, 1.36107, 0.15123, 0.15123, -1.05861, 0.15123, 1.36107, -1.05861,
        -1.05861, -1.05861, 1.36107, 0.15123, 0.15123, 0.866135, 0.0962372, -0.67366, 0.0962372,
        0.866135, -0.67366, 0.0962372, -0.67366, 0.866135, 0.0962372, 0.0962372, 0.866135,
        0.866135, -0.67366, 0.0962372, -0.67366, -0.67366, 0.866135, 0.0962372, 0.0962372,
        -0.67366, 0.0962372, 0.0962372, 0.0962372, 0.866135, -0.67366, 0.0962372, 0.866135,
        0.0962372, -0.67366, 0.0962372, -0.67366, 1.60854, 0.178726, 1.60854, 0.178726, 0.178726,
        0.178726, 1.60854, 0.178726, 1.60854, 0.178726, -1.25108, 1.60854, 1.60854, 0.178726,
        0.178726, 0.178726, 0.178726, 0.178726, 1.60854, 1.60854, 0.178726, 1.60854, -1.25108,
        -1.25108, -1.25108, -1.25108, 0.178726, -1.25108, 1.60854, 0.178726, 1.60854, -1.25108,
        0.0962372, -0.123734, -0.0137482, -0.0137482, 0.0962372, 0.0962372, -0.0137482, -0.123734,
        -0.123734, 0.0962372, -0.123734, 0.0962372, 0.0962372, -0.123734, 0.0962372, -0.123734,
        -0.0137482, 0.0962372, -0.0137482, -0.0137482, 0.0962372, -0.123734, -0.123734, 0.0962372,
        -0.123734, -0.0137482, -0.0137482, 0.0962372, -0.123734, -0.0137482, 0.0962372, -0.123734,
        0.618668, 0.618668, -0.481186, 0.618668, -0.481186, 0.618668, -0.481186, -0.481186,
        -0.481186, 0.0687408, 0.618668, 0.0687408, -0.481186, 0.0687408, -0.481186, -0.481186,
        0.0687408, 0.0687408, 0.618668, 0.618668, 0.618668, 0.618668, -0.481186, 0.0687408,
        0.618668, 0.0687408, -0.481186, 0.0687408, -0.481186, 0.0687408, 0.618668, -0.481186,
    ];

    const IQ2_XXS_TEST_BLOCK: [u8; 66] = [
        0x29, 0x30, 0xd9, 0x33, 0x95, 0x4c, 0x08, 0x1e, 0xad, 0x79, 0x49, 0xf2, 0x8d, 0x5f, 0x93,
        0xea, 0x78, 0x18, 0x98, 0xb9, 0x94, 0x14, 0xad, 0xce, 0xca, 0x1d, 0xab, 0x81, 0x53, 0x4a,
        0x68, 0xd0, 0x59, 0x96, 0x36, 0x5d, 0xbe, 0x20, 0xc4, 0xff, 0xe4, 0x2c, 0xcd, 0x2f, 0x4f,
        0x4f, 0x67, 0x53, 0xc6, 0xd5, 0xa2, 0xfb, 0xc7, 0xf3, 0xe2, 0x6b, 0xf1, 0x99, 0x23, 0x1e,
        0x2d, 0x5e, 0x8c, 0x78, 0xc2, 0x31,
    ];
    const IQ2_XXS_GOLDEN: [f32; 256] = [
        1.95007, 1.95007, 1.95007, -6.09398, 6.09398, 1.95007, 1.95007, -10.4816, 1.95007, 1.95007,
        -1.95007, -10.4816, -6.09398, -6.09398, 1.95007, 1.95007, 6.09398, 6.09398, -1.95007,
        10.4816, -6.09398, -1.95007, 1.95007, -6.09398, -1.95007, 1.95007, -1.95007, -6.09398,
        1.95007, 1.95007, -6.09398, 1.95007, -0.390015, -1.2188, 0.390015, 0.390015, -0.390015,
        0.390015, 1.2188, -0.390015, -0.390015, 0.390015, -0.390015, 0.390015, -0.390015, 1.2188,
        -1.2188, 2.09633, -0.390015, -0.390015, 2.09633, 1.2188, 0.390015, -0.390015, -0.390015,
        1.2188, -0.390015, -2.09633, 1.2188, 0.390015, 1.2188, 1.2188, -1.2188, -0.390015,
        -0.390015, 2.09633, -0.390015, -1.2188, 2.09633, -0.390015, 0.390015, 1.2188, -0.390015,
        0.390015, -1.2188, -2.09633, -0.390015, 1.2188, 1.2188, 1.2188, -0.390015, -0.390015,
        0.390015, -2.09633, 1.2188, -0.390015, 0.390015, 1.2188, 2.09633, -0.390015, -2.09633,
        -2.09633, 0.390015, -0.390015, -0.390015, -0.390015, 13.2767, 2.47009, 2.47009, -13.2767,
        7.71904, -13.2767, -2.47009, -7.71904, 2.47009, 2.47009, 13.2767, 2.47009, 2.47009,
        -13.2767, 13.2767, -2.47009, -2.47009, -2.47009, -13.2767, 2.47009, 7.71904, -2.47009,
        -7.71904, -2.47009, 2.47009, -2.47009, 7.71904, 2.47009, -2.47009, -2.47009, 7.71904,
        -2.47009, 0.650024, 0.650024, -2.03133, 0.650024, 3.49388, 2.03133, -0.650024, 0.650024,
        -2.03133, -3.49388, -0.650024, -2.03133, -0.650024, -2.03133, -2.03133, -0.650024,
        -0.650024, -0.650024, 0.650024, 0.650024, -0.650024, 3.49388, 2.03133, -2.03133, -2.03133,
        -0.650024, -0.650024, 0.650024, 0.650024, -2.03133, -0.650024, -0.650024, -10.9692,
        -10.9692, -3.51013, 3.51013, 3.51013, -10.9692, -18.867, -10.9692, 3.51013, -18.867,
        -3.51013, 3.51013, 10.9692, -10.9692, 3.51013, -3.51013, -3.51013, 3.51013, 10.9692,
        -18.867, -3.51013, 3.51013, 10.9692, -3.51013, 3.51013, -3.51013, -10.9692, -18.867,
        3.51013, -3.51013, 10.9692, 3.51013, 2.47009, -2.47009, 2.47009, 2.47009, 13.2767,
        -7.71904, -2.47009, -7.71904, -7.71904, -13.2767, -2.47009, 2.47009, -7.71904, 2.47009,
        -13.2767, -13.2767, -2.47009, 13.2767, -13.2767, 7.71904, 2.47009, 2.47009, -13.2767,
        -7.71904, -13.2767, -2.47009, -13.2767, -2.47009, 2.47009, 7.71904, -7.71904, -13.2767,
        2.84386, 0.910034, -4.89143, -0.910034, 0.910034, 2.84386, 0.910034, 0.910034, -4.89143,
        0.910034, 4.89143, 0.910034, -4.89143, -0.910034, -0.910034, 0.910034, -0.910034, 0.910034,
        0.910034, -0.910034, 2.84386, 2.84386, 0.910034, 0.910034, 0.910034, -0.910034, -0.910034,
        -4.89143, 0.910034, 2.84386, 2.84386, -0.910034,
    ];

    const IQ3_XXS_TEST_BLOCK: [u8; 98] = [
        0x71, 0x31, 0x16, 0x0a, 0x79, 0x04, 0x5d, 0x87, 0xae, 0x2a, 0x4a, 0x43, 0xfd, 0x02, 0xba,
        0x6c, 0x10, 0x42, 0x80, 0xe5, 0x1d, 0x08, 0x22, 0xcb, 0x21, 0x54, 0xf9, 0xaa, 0x8e, 0xc2,
        0xf2, 0x34, 0x66, 0x1e, 0x2a, 0xef, 0x19, 0xae, 0x48, 0x47, 0x29, 0xa0, 0x72, 0xd1, 0x31,
        0xc0, 0x65, 0x49, 0xde, 0x79, 0x32, 0xe6, 0x4d, 0xb6, 0x55, 0x3f, 0x4d, 0xf1, 0x18, 0xbb,
        0x18, 0x59, 0x4c, 0x31, 0xa3, 0xb2, 0x34, 0xdd, 0xf6, 0x4a, 0x91, 0x51, 0x3f, 0x3e, 0x40,
        0x69, 0xad, 0xbf, 0x1a, 0xd0, 0x05, 0xfb, 0xbe, 0x8b, 0x0b, 0xdd, 0xdf, 0x7d, 0x94, 0x74,
        0x92, 0x3e, 0xff, 0x04, 0x2a, 0xc4, 0xea, 0xc9,
    ];
    const IQ3_XXS_GOLDEN: [f32; 256] = [
        1.5304, 23.7211, -4.59119, 1.5304, -10.7128, -23.7211, 1.5304, -1.5304, 7.65198, -7.65198,
        7.65198, -7.65198, -10.7128, -4.59119, 1.5304, 1.5304, -4.59119, -23.7211, 16.8344,
        -4.59119, -13.7736, 23.7211, -16.8344, -7.65198, -13.7736, -1.5304, -1.5304, 13.7736,
        -23.7211, 10.7128, -13.7736, -1.5304, -3.57092, 1.19031, 5.95154, 3.57092, -18.4498,
        10.7128, 1.19031, 3.57092, -5.95154, -1.19031, 13.0934, 18.4498, 10.7128, -1.19031,
        1.19031, -1.19031, -18.4498, 1.19031, -8.33215, -10.7128, -13.0934, -1.19031, -3.57092,
        5.95154, -3.57092, 5.95154, 3.57092, 1.19031, -10.7128, -8.33215, -1.19031, 3.57092,
        3.91101, 60.6207, 27.3771, 19.5551, 35.1991, 35.1991, -35.1991, -50.8431, 11.733, -27.3771,
        19.5551, 3.91101, -11.733, 27.3771, -3.91101, -3.91101, -43.0211, 60.6207, -19.5551,
        3.91101, -50.8431, -19.5551, 11.733, 43.0211, -60.6207, 43.0211, -19.5551, -3.91101,
        -11.733, -27.3771, -27.3771, 11.733, 5.27136, -68.5277, 36.8995, -81.7061, -68.5277,
        36.8995, 68.5277, -36.8995, 26.3568, 15.8141, 5.27136, 36.8995, 57.985, -5.27136, 81.7061,
        -47.4423, -5.27136, -47.4423, -15.8141, 81.7061, -47.4423, 68.5277, 68.5277, 5.27136,
        26.3568, 26.3568, 5.27136, -26.3568, -36.8995, 36.8995, -26.3568, -5.27136, 71.1634,
        -32.1383, -41.3207, -4.59119, -22.9559, -32.1383, 4.59119, -71.1634, -41.3207, -4.59119,
        -22.9559, 4.59119, -41.3207, 4.59119, 4.59119, 41.3207, 4.59119, -22.9559, -13.7736,
        -13.7736, 13.7736, -13.7736, 13.7736, 13.7736, 32.1383, 13.7736, 41.3207, -4.59119,
        13.7736, -13.7736, -32.1383, -32.1383, -39.5352, -33.1586, -7.65198, -12.7533, -17.8546,
        28.0573, -17.8546, 28.0573, -12.7533, -17.8546, 28.0573, -2.55066, -17.8546, -22.9559,
        -28.0573, 22.9559, -2.55066, 12.7533, 2.55066, 12.7533, -12.7533, 12.7533, -7.65198,
        -7.65198, 22.9559, 33.1586, -2.55066, 33.1586, 12.7533, -12.7533, 12.7533, 12.7533,
        0.85022, -1.87048, 1.87048, 0.170044, -1.87048, 0.170044, 1.87048, 2.21057, -0.85022,
        0.510132, -0.85022, -0.510132, -2.63568, -1.19031, -0.85022, 1.5304, 2.21057, 1.5304,
        -1.19031, -0.510132, -1.19031, -0.85022, -0.170044, -0.510132, -0.85022, -0.510132,
        -0.85022, 0.510132, 2.21057, -0.85022, 0.510132, 2.63568, 21.2555, -4.2511, 21.2555,
        -4.2511, 46.7621, -38.2599, 29.7577, -38.2599, 21.2555, 4.2511, 21.2555, -4.2511, 4.2511,
        46.7621, 38.2599, -12.7533, -4.2511, -12.7533, 21.2555, -12.7533, 21.2555, -29.7577,
        46.7621, 4.2511, -65.892, -38.2599, -38.2599, -29.7577, 29.7577, 46.7621, -4.2511,
        -38.2599,
    ];

    #[test]
    fn iq1_s_dequant_matches_independent_python_reference() {
        let got = dequant_iq1_s(&IQ1_S_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), IQ1_S_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(IQ1_S_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "IQ1_S element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn iq2_xxs_dequant_matches_independent_python_reference() {
        let got = dequant_iq2_xxs(&IQ2_XXS_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), IQ2_XXS_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(IQ2_XXS_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "IQ2_XXS element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn iq3_xxs_dequant_matches_independent_python_reference() {
        let got = dequant_iq3_xxs(&IQ3_XXS_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), IQ3_XXS_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(IQ3_XXS_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "IQ3_XXS element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn iq_lowbit_fused_dots_match_dequant_then_dot() {
        type DequantFn = fn(&[u8]) -> Result<Vec<f32>, QuantError>;
        type DotFn = fn(&[u8], &[f32]) -> f32;
        let x: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.027).sin()).collect();
        let cases: [(&[u8], usize, DequantFn, DotFn); 3] = [
            (&IQ1_S_TEST_BLOCK, 4, dequant_iq1_s, dot_iq1_s_f32),
            (&IQ2_XXS_TEST_BLOCK, 4, dequant_iq2_xxs, dot_iq2_xxs_f32),
            (&IQ3_XXS_TEST_BLOCK, 4, dequant_iq3_xxs, dot_iq3_xxs_f32),
        ];
        for (block, n, dequant, dot) in cases {
            let packed = repeat_block(block, n);
            let dequanted = dequant(&packed).unwrap();
            let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let fused = dot(&packed, &x[..dequanted.len()]);
            assert!(
                (fused - expected).abs() < 1e-2,
                "fused={fused} expected={expected}"
            );
        }
    }

    /// Direct AVX2-vs-scalar comparison for the three IQ kernels on
    /// many random blocks (fully random codes/signs/scales, `d`
    /// pinned non-NaN) -- run on real x86_64 hardware, not just the
    /// committed golden block.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_iq_kernels_match_scalar_directly_on_random_blocks() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            eprintln!("skipping: host CPU lacks AVX2+FMA");
            return;
        }
        type ScalarFn = fn(&[u8], &[f32]) -> f32;
        type Avx2Fn = unsafe fn(&[u8], &[f32]) -> f32;
        let cases: [(&str, usize, ScalarFn, Avx2Fn); 3] = [
            (
                "iq1_s",
                IQ1_S_BLOCK_BYTES,
                dot_iq1_s_f32_scalar,
                simd_x86::dot_iq1_s_f32_avx2,
            ),
            (
                "iq2_xxs",
                IQ2_XXS_BLOCK_BYTES,
                dot_iq2_xxs_f32_scalar,
                simd_x86::dot_iq2_xxs_f32_avx2,
            ),
            (
                "iq3_xxs",
                IQ3_XXS_BLOCK_BYTES,
                dot_iq3_xxs_f32_scalar,
                simd_x86::dot_iq3_xxs_f32_avx2,
            ),
        ];
        for (name, block_bytes, scalar, avx2) in cases {
            for trial in 0..16u32 {
                let n_blocks = 3;
                let mut bytes =
                    pseudo_random_bytes(trial.wrapping_mul(97) + 5, n_blocks * block_bytes);
                for b in 0..n_blocks {
                    // pin each block's f16 `d` to a safe small value
                    let d = half::f16::from_f32(0.05 + 0.01 * trial as f32).to_le_bytes();
                    bytes[b * block_bytes] = d[0];
                    bytes[b * block_bytes + 1] = d[1];
                }
                let x: Vec<f32> = (0..n_blocks * 256)
                    .map(|i| ((i as f32) * 0.017 + trial as f32).sin())
                    .collect();
                let s = scalar(&bytes, &x);
                let v = unsafe { avx2(&bytes, &x) };
                // Tolerance covers accumulation-order drift only (the
                // 8-lane FMA sums in a different order than scalar,
                // over per-term magnitudes up to ~100 here); any real
                // decode bug -- wrong grid row, sign, or scale --
                // shifts the result by orders of magnitude more than
                // this on random codes.
                let tol = 2e-3_f32.max(s.abs() * 1e-3);
                assert!(
                    (s - v).abs() < tol,
                    "{name} trial {trial}: scalar={s} avx2={v}"
                );
            }
        }
    }

    // Only called from `avx2_iq_kernels_match_scalar_directly_on_random_blocks`,
    // which is itself `#[cfg(target_arch = "x86_64")]` -- this must carry
    // the same gate or it's dead code (and fails `-D warnings`) on
    // non-x86_64 hosts (e.g. aarch64 Apple Silicon).
    #[cfg(target_arch = "x86_64")]
    fn pseudo_random_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 16) as u8
            })
            .collect()
    }

    #[test]
    fn iq_lowbit_dequant_rejects_misaligned_buffers() {
        let bad = vec![0u8; 7];
        assert!(dequant_iq1_s(&bad).is_err());
        assert!(dequant_iq2_xxs(&bad).is_err());
        assert!(dequant_iq3_xxs(&bad).is_err());
        assert!(dequant_iq2_xs(&bad).is_err());
        assert!(dequant_iq2_s(&bad).is_err());
        assert!(dequant_iq3_s(&bad).is_err());
        assert!(dequant_iq1_m(&bad).is_err());
    }

    /// IQ2_XS / IQ2_S / IQ3_S / IQ1_M against the **real compiled ggml
    /// dequantizers**, not a second reading of the spec.
    ///
    /// This is the whole job for these four formats. They are codebook
    /// formats: a wrong grid index, a swapped scale nibble or an
    /// off-by-one in the sign unpack does not produce obviously broken
    /// numbers, it produces other plausible numbers out of the same
    /// codebook. So the goldens in `iq_tier_goldens` are ggml's own
    /// output (see that module's header for how they were produced and
    /// why those particular blocks), and the comparison is **exact** --
    /// every arithmetic step here is expressible in f32 without
    /// reassociation, so any difference at all is a decode bug, not
    /// rounding.
    #[test]
    fn iq_tier_dequant_matches_real_ggml_exactly() {
        type DequantFn = fn(&[u8]) -> Result<Vec<f32>, QuantError>;
        let cases: [(&str, &[u8], &[f32], DequantFn); 4] = [
            (
                "IQ2_XS",
                &iq_tier_goldens::IQ2_XS_TEST_BLOCKS,
                &iq_tier_goldens::IQ2_XS_GOLDEN,
                dequant_iq2_xs,
            ),
            (
                "IQ2_S",
                &iq_tier_goldens::IQ2_S_TEST_BLOCKS,
                &iq_tier_goldens::IQ2_S_GOLDEN,
                dequant_iq2_s,
            ),
            (
                "IQ3_S",
                &iq_tier_goldens::IQ3_S_TEST_BLOCKS,
                &iq_tier_goldens::IQ3_S_GOLDEN,
                dequant_iq3_s,
            ),
            (
                "IQ1_M",
                &iq_tier_goldens::IQ1_M_TEST_BLOCKS,
                &iq_tier_goldens::IQ1_M_GOLDEN,
                dequant_iq1_m,
            ),
        ];
        for (name, blocks, golden, dequant) in cases {
            let got = dequant(blocks).unwrap();
            assert_eq!(got.len(), golden.len(), "{name}: element count");
            for (i, (a, b)) in got.iter().zip(golden.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} element {i} (block {}, offset {}): rust={a} ggml={b}",
                    i / 256,
                    i % 256
                );
            }
        }
    }

    /// The saturated first block of each fixture is the one that pins
    /// the *high* end of every packed field, so spell out what it is
    /// asserting: with every byte 0xff, each format must reach its
    /// maximum grid index -- the single most likely thing to get wrong
    /// when a format widens its index by stealing bits from `qh`.
    ///
    /// Derived here from the grid tables directly, so this test fails
    /// even if the golden fixture were regenerated from a broken
    /// harness.
    #[test]
    fn iq_tier_all_ones_block_reaches_the_maximum_grid_index() {
        // IQ2_XS: code = 0xffff -> grid index 511 (the top of a 512-row
        // grid), sign index 127 -> ksigns 255 -> every element negative.
        // Scale nibble 15 -> db = d * (0.5 + 15) * 0.25.
        let d = f16::from_le_bytes([
            iq_tier_goldens::IQ2_XS_TEST_BLOCKS[0],
            iq_tier_goldens::IQ2_XS_TEST_BLOCKS[1],
        ])
        .to_f32();
        let mag = (iq_tables::IQ2XS_GRID[511] & 0xFF) as f32;
        assert_eq!(
            iq_tier_goldens::IQ2_XS_GOLDEN[0],
            -(d * (0.5 + 15.0) * 0.25) * mag
        );

        // IQ2_S: qs byte 0xff plus 2 high bits from qh -> grid index
        // 1023, the top of a 1024-row grid; sign byte 0xff.
        let d = f16::from_le_bytes([
            iq_tier_goldens::IQ2_S_TEST_BLOCKS[0],
            iq_tier_goldens::IQ2_S_TEST_BLOCKS[1],
        ])
        .to_f32();
        let mag = (iq_tables::IQ2S_GRID[1023] & 0xFF) as f32;
        assert_eq!(
            iq_tier_goldens::IQ2_S_GOLDEN[0],
            -(d * (0.5 + 15.0) * 0.25) * mag
        );

        // IQ3_S: qs byte 0xff plus the 9th bit from qh -> grid index
        // 511; scale nibble 15 -> db = d * (1 + 2*15) = 31*d.
        let d = f16::from_le_bytes([
            iq_tier_goldens::IQ3_S_TEST_BLOCKS[0],
            iq_tier_goldens::IQ3_S_TEST_BLOCKS[1],
        ])
        .to_f32();
        let mag = (iq_tables::IQ3S_GRID[511] & 0xFF) as f32;
        assert_eq!(iq_tier_goldens::IQ3_S_GOLDEN[0], -(d * 31.0) * mag);

        // IQ1_M: qs byte 0xff plus 3 high bits from qh -> grid index
        // 2047, the top of the shared 2048-row IQ1 grid. Its scale is
        // the f16 reassembled from the scale words' top nibbles, and
        // its sub-scale nibble is 7 -> 2*7+1 = 15. The grid values are
        // *signed*, and qh bit 3 is set so delta is negative.
        let sc: [u16; 4] = std::array::from_fn(|k| {
            u16::from_le_bytes([
                iq_tier_goldens::IQ1_M_TEST_BLOCKS[48 + 2 * k],
                iq_tier_goldens::IQ1_M_TEST_BLOCKS[48 + 2 * k + 1],
            ])
        });
        let d = f16::from_bits(
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000),
        )
        .to_f32();
        let v = (iq_tables::IQ1S_GRID[2047] & 0xFF) as u8 as i8;
        assert_eq!(
            iq_tier_goldens::IQ1_M_GOLDEN[0],
            d * 15.0 * (v as f32 - IQ1S_DELTA)
        );
    }

    /// The fused dots for the new tier must agree with dequant-then-dot
    /// on the same bytes -- the same invariant
    /// `iq_lowbit_fused_dots_match_dequant_then_dot` pins for the older
    /// formats, restated here because these four share only the macro,
    /// not the walk.
    #[test]
    fn iq_tier_fused_dots_match_dequant_then_dot() {
        type DequantFn = fn(&[u8]) -> Result<Vec<f32>, QuantError>;
        type DotFn = fn(&[u8], &[f32]) -> f32;
        let x: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.031).cos()).collect();
        let cases: [(&str, &[u8], DequantFn, DotFn); 4] = [
            (
                "IQ2_XS",
                &iq_tier_goldens::IQ2_XS_TEST_BLOCKS,
                dequant_iq2_xs,
                dot_iq2_xs_f32,
            ),
            (
                "IQ2_S",
                &iq_tier_goldens::IQ2_S_TEST_BLOCKS,
                dequant_iq2_s,
                dot_iq2_s_f32,
            ),
            (
                "IQ3_S",
                &iq_tier_goldens::IQ3_S_TEST_BLOCKS,
                dequant_iq3_s,
                dot_iq3_s_f32,
            ),
            (
                "IQ1_M",
                &iq_tier_goldens::IQ1_M_TEST_BLOCKS,
                dequant_iq1_m,
                dot_iq1_m_f32,
            ),
        ];
        for (name, blocks, dequant, dot) in cases {
            let dequanted = dequant(blocks).unwrap();
            let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let fused = dot(blocks, &x[..dequanted.len()]);
            assert!(
                (fused - expected).abs() <= expected.abs() * 1e-5 + 1e-3,
                "{name}: fused={fused} expected={expected}"
            );
        }
    }

    // Generated by an independent Python reference -- do not hand-edit.
    // 4 GGUF-block-MXFP4 blocks with distinct pinned E8M0 scale bytes;
    // the Python reference is cross-validated against the real compiled
    // ggml implementation across the FULL random E8M0 range (including
    // the e<2 denormal patterns).
    const MXFP4_GGUF_TEST_BLOCKS: [u8; 68] = [
        0x79, 0xb4, 0x8d, 0xe2, 0x62, 0x5d, 0xbb, 0x9d, 0x54, 0xe6, 0xdb, 0x94, 0x59, 0x7d, 0x28,
        0xf9, 0x79, 0x7a, 0xfc, 0xc1, 0xfa, 0x1e, 0x53, 0x5b, 0x0e, 0xc2, 0x5a, 0x2f, 0x0c, 0x82,
        0x4d, 0xcb, 0x11, 0x28, 0x7b, 0x7c, 0xb6, 0x45, 0xe0, 0xb0, 0x52, 0x40, 0x51, 0xec, 0x30,
        0x1a, 0xd2, 0x17, 0xf3, 0xbb, 0xfc, 0x7c, 0x8f, 0xf0, 0x67, 0x83, 0x88, 0x9d, 0x79, 0xdb,
        0xf4, 0x45, 0x29, 0x78, 0xe6, 0xf4, 0x99, 0xea,
    ];
    const MXFP4_GGUF_GOLDEN: [f32; 128] = [
        0.03125, -0.046875, 0.015625, 0.015625, -0.046875, -0.0234375, -0.046875, 0.03125, 0.0625,
        -0.0234375, 0.03125, -0.0078125, -0.046875, 0.0, -0.0078125, -0.0078125, -0.0234375, 0.0,
        -0.0625, 0.0625, 0.046875, -0.0234375, -0.0078125, 0.046875, -0.0625, -0.046875,
        -0.0078125, 0.046875, 0.09375, 0.015625, -0.09375, 0.09375, -0.0625, 0.015625, -0.03125,
        -0.125, 0.046875, -0.046875, -0.125, 0.03125, -0.03125, -0.1875, -0.0625, 0.03125,
        -0.09375, -0.046875, 0.015625, 0.0, -0.1875, -0.0625, -0.1875, 0.015625, 0.09375, 0.09375,
        0.0, -0.0625, 0.09375, 0.03125, 0.0, 0.0, 0.0625, -0.0625, 0.015625, 0.03125, -0.125, 0.25,
        0.1875, 0.0, 0.0, 0.0625, 0.0, 0.03125, -0.125, 0.0, -0.0625, 0.0625, 0.375, 0.09375,
        -0.09375, -0.125, 0.375, -0.09375, 0.125, -0.25, -0.09375, 0.1875, 0.125, 0.1875, -0.25,
        0.09375, 0.03125, -0.1875, 0.03125, -0.375, -0.09375, -0.375, -0.75, 0.0, 0.75, 0.1875,
        0.0, -0.375, -0.0625, -0.1875, 0.25, 0.375, -0.0625, 0.0, 0.5, 0.25, -0.0625, -0.125, 0.0,
        -0.75, 0.5, 0.0, 0.0, -0.0625, 0.75, -0.375, -0.75, 0.25, 0.125, 0.75, -0.5, -0.75,
        -0.0625, -0.5,
    ];

    #[test]
    fn mxfp4_gguf_dequant_matches_independent_python_reference() {
        let got = dequant_mxfp4_gguf(&MXFP4_GGUF_TEST_BLOCKS).unwrap();
        assert_eq!(got.len(), MXFP4_GGUF_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(MXFP4_GGUF_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "MXFP4-GGUF element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn mxfp4_gguf_fused_dot_matches_dequant_then_dot() {
        let dequanted = dequant_mxfp4_gguf(&MXFP4_GGUF_TEST_BLOCKS).unwrap();
        let x: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.031).cos()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_mxfp4_gguf_f32(&MXFP4_GGUF_TEST_BLOCKS, &x);
        assert!(
            (fused - expected).abs() < 1e-2,
            "fused={fused} expected={expected}"
        );
    }

    /// The GGUF block form and the Kimi two-buffer form are the same
    /// math in different byte layouts -- deinterleaving a block row
    /// into (packed, scales) buffers and running the two-buffer kernel
    /// must produce the same result.
    #[test]
    fn mxfp4_gguf_block_form_agrees_with_two_buffer_form() {
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        for block in MXFP4_GGUF_TEST_BLOCKS
            .as_chunks::<MXFP4_GGUF_BLOCK_BYTES>()
            .0
        {
            scales.push(block[0]);
            packed.extend_from_slice(&block[1..17]);
        }
        let x: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.019).sin()).collect();
        let a = dot_mxfp4_gguf_f32(&MXFP4_GGUF_TEST_BLOCKS, &x);
        let b = dot_mxfp4_row_f32(&packed, &scales, &x);
        assert!((a - b).abs() < 1e-4, "block={a} two-buffer={b}");
    }

    // Generated by an independent Python reference -- do not hand-edit.
    // Q6_K block whose int8 sub-block scales include *negative* values
    // (9 of 16 in this draw). Q6_K is the only K-quant whose sub-block
    // scales are signed; every other Q6_K golden in this file happens
    // to have all-positive scales, which is exactly why a scalar path
    // that read them as unsigned passed all of those tests while
    // disagreeing with the format (and with the AVX2/NEON kernels) on
    // real checkpoints.
    const Q6_K_SIGNED_TEST_BLOCK: [u8; 210] = [
        0x10, 0x5b, 0x5f, 0x45, 0x4a, 0xa0, 0x3f, 0x10, 0xf2, 0x7f, 0xdd, 0xf5, 0x25, 0x03, 0xc3,
        0x12, 0x74, 0xe1, 0x4e, 0x42, 0xf1, 0x04, 0xe1, 0xad, 0xc6, 0x55, 0x59, 0x4b, 0x5a, 0xfc,
        0xf5, 0x3f, 0xc5, 0x0b, 0xac, 0x7b, 0x4c, 0xd4, 0x19, 0xa6, 0x27, 0xdd, 0xf4, 0x7d, 0x9c,
        0xfc, 0x03, 0xd2, 0x5f, 0xe3, 0xff, 0x9c, 0xa6, 0x74, 0xa0, 0xe1, 0xbe, 0xf0, 0x26, 0xdb,
        0x4b, 0x23, 0xa0, 0xbc, 0xb1, 0x94, 0xd7, 0x7e, 0xcf, 0xf7, 0x97, 0xb4, 0xac, 0x1f, 0xb1,
        0x9f, 0xb7, 0xbe, 0xa3, 0xb5, 0xd2, 0xd4, 0x6d, 0x9c, 0x3d, 0xf3, 0x5f, 0x0e, 0x64, 0xbf,
        0x54, 0x40, 0xc8, 0xef, 0x9d, 0xc3, 0xf3, 0x4c, 0xb0, 0xf8, 0x54, 0xcf, 0xf3, 0x12, 0xcc,
        0x2f, 0x0c, 0xee, 0xab, 0x5d, 0x8d, 0x0b, 0x19, 0xb2, 0x99, 0xbd, 0x4a, 0xec, 0x04, 0xb3,
        0xf6, 0xc1, 0xb9, 0xf8, 0x1d, 0xfe, 0x51, 0xea, 0x99, 0xe5, 0x75, 0x5b, 0x98, 0x28, 0x05,
        0x18, 0x8a, 0x9f, 0xda, 0xb7, 0xb6, 0xe5, 0x5b, 0x3a, 0x52, 0x49, 0xcc, 0x72, 0xff, 0x61,
        0x91, 0x95, 0xa2, 0xa1, 0x5d, 0xd5, 0xc4, 0x7d, 0xb1, 0x0b, 0xda, 0xa9, 0xa2, 0x97, 0x1e,
        0x7e, 0xe9, 0xa2, 0xd6, 0xdd, 0x0e, 0x94, 0x21, 0xa4, 0x67, 0x92, 0xad, 0x46, 0xab, 0xe1,
        0xe2, 0x3b, 0x21, 0x69, 0x2a, 0x1e, 0xd3, 0xea, 0xa4, 0xdf, 0xa6, 0xd2, 0xff, 0x01, 0xfe,
        0xff, 0x01, 0xff, 0x01, 0x01, 0x02, 0xff, 0xff, 0x01, 0xfe, 0x02, 0x01, 0xff, 0x1f, 0x25,
    ];
    const Q6_K_SIGNED_GOLDEN: [f32; 256] = [
        0.320068, 0.100021, 0.0200043, -0.42009, 0.440094, 0.640137, 0.0200043, 0.640137,
        -0.0400085, -0.620132, -0.260056, -0.42009, -0.100021, 0.260056, -0.380081, -0.0400085,
        0.0800171, -0.300064, -0.360077, 0.0400085, 0.340073, -0.240051, -0.300064, -0.0600128,
        0.120026, -0.220047, -0.14003, -0.100021, -0.440094, -0.0800171, -0.220047, 0.620132,
        -0.200043, 0.200043, 0.160034, -0.440094, -0.480103, -0.160034, 0.28006, -0.240051,
        -0.28006, -1.16025, -0.160034, 0.120026, 0.160034, 0.160034, -0.120026, -0.0800171,
        0.340073, -0.0600128, -0.620132, 0.400085, -0.440094, 0.56012, 0.640137, 0.300064,
        0.360077, 0.640137, -0.440094, 0.100021, 0.100021, -0.380081, 0.640137, -0.240051,
        -0.300064, 0.100021, 0.42009, -0.240051, -0.240051, 0.200043, -0.580124, -0.300064,
        -0.340073, -0.180038, -0.0600128, 0.620132, 0.360077, 0.0, -0.0800171, 0.340073, 0.180038,
        0.360077, 0.56012, -0.400085, -0.620132, -0.0, 0.0400085, 0.120026, -0.240051, -0.100021,
        0.220047, 0.240051, 0.540115, -0.620132, -0.620132, 0.580124, 0.240051, 0.320068,
        -0.120026, -0.180038, 0.0800171, -0.380081, -0.620132, -0.440094, 0.0400085, 0.260056,
        0.620132, 0.14003, 0.180038, 0.620132, -0.320068, -0.380081, -0.220047, -0.0400085,
        0.620132, -0.14003, 0.520111, -0.180038, 0.200043, 0.28006, 0.220047, 0.300064, -0.28006,
        0.580124, 0.400085, -0.28006, 0.200043, -0.42009, 0.0400085, -0.480103, 0.28006, 1.20026,
        0.600128, 0.28006, -0.360077, 0.160034, 0.480103, -0.0400085, 0.0400085, -0.680145,
        -0.360077, -0.720154, 0.760162, 0.200043, 0.28006, -0.0800171, -0.580124, 0.0800171,
        -0.260056, -0.380081, 0.0200043, 0.0400085, -0.0800171, -0.300064, -0.400085, -0.0,
        0.480103, -0.620132, -0.260056, -0.0600128, -0.0600128, -0.240051, 0.640137, 0.160034,
        -0.400085, -0.620132, -0.0600128, 0.600128, 0.0800171, -0.620132, -0.56012, 0.0400085,
        0.42009, 0.0600128, 0.0600128, 0.42009, 0.500107, -0.28006, 0.180038, -0.380081, -0.440094,
        0.240051, -0.56012, 0.0600128, 0.120026, 0.340073, -0.460098, 0.160034, -0.0600128,
        0.600128, -0.300064, -0.440094, 0.200043, -0.360077, -0.520111, 0.360077, 0.160034,
        -1.24026, -0.360077, -0.440094, 0.240051, 0.600128, 0.840179, 0.28006, -0.440094,
        -0.440094, -0.400085, 0.200043, 0.520111, -0.760162, 0.240051, 0.360077, 0.120026, 1.24026,
        0.200043, 0.0, 0.240051, -0.200043, -0.440094, 0.160034, 0.480103, -0.0800171, 0.360077,
        -0.160034, 0.620132, 0.0800171, 0.220047, 0.300064, -0.540115, -0.0800171, 0.620132,
        0.0200043, 0.56012, 0.360077, -0.640137, 0.28006, -0.440094, 0.100021, -0.160034, 0.0,
        -0.0200043, 0.100021, -0.180038, -0.540115, -0.400085, 0.360077, 0.640137, 0.100021,
        0.340073, 0.400085, -0.540115, -0.620132, -0.0200043, -0.620132, -0.100021, -0.600128,
    ];

    #[test]
    fn q6_k_signed_scale_dequant_matches_independent_python_reference() {
        let got = dequant_q6_k(&Q6_K_SIGNED_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), Q6_K_SIGNED_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(Q6_K_SIGNED_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "Q6_K signed-scale element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn q6_k_signed_scale_fused_dot_matches_dequant_then_dot() {
        let dequanted = dequant_q6_k(&Q6_K_SIGNED_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.023).sin()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_q6_k_f32(&Q6_K_SIGNED_TEST_BLOCK, &x);
        assert!(
            (fused - expected).abs() < 1e-2,
            "fused={fused} expected={expected}"
        );
    }

    #[test]
    fn dispatched_q6_k_matches_scalar_on_signed_scales() {
        // On AVX2/NEON hosts this compares the SIMD kernel (which always
        // read the scales as signed) against the scalar path directly on
        // a negative-scale block -- the comparison that would have caught
        // the scalar path's unsigned-scale bug.
        let n_blocks = 4;
        let packed = repeat_block(&Q6_K_SIGNED_TEST_BLOCK, n_blocks);
        let x: Vec<f32> = (0..256 * n_blocks)
            .map(|i| ((i as f32) * 0.019).sin())
            .collect();
        let dispatched = dot_q6_k_f32(&packed, &x);
        let scalar = dot_q6_k_f32_scalar(&packed, &x);
        assert!(
            (dispatched - scalar).abs() < 1e-1,
            "dispatched={dispatched} scalar={scalar}"
        );
    }

    #[test]
    fn q6_k_dequant_matches_python_reference_with_negative_scales() {
        // Regression test for a real bug: the scalar dequant read the
        // signed int8 sub-block scales as unsigned, so any negative
        // scale (e.g. -1 -> 255) corrupted its whole sub-block. The
        // all-positive-scale fixture above could never catch that.
        let got = dequant_q6_k(&Q6_K_SIGNED_SCALES_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), Q6_K_SIGNED_SCALES_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(Q6_K_SIGNED_SCALES_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "Q6_K signed-scale element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn q6_k_fused_dot_matches_dequant_then_dot_with_negative_scales() {
        let dequanted = dequant_q6_k(&Q6_K_SIGNED_SCALES_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.023).sin()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_q6_k_f32(&Q6_K_SIGNED_SCALES_TEST_BLOCK, &x);
        assert!(
            (fused - expected).abs() < 1e-2,
            "fused={fused} expected={expected}"
        );
    }

    #[test]
    fn q6_k_scalar_dot_matches_python_reference_with_negative_scales() {
        // Pins the *scalar* path specifically (not whatever SIMD path
        // `dot_q6_k_f32` dispatches to on this host) against the
        // independent Python golden, so scalar/SIMD can never again
        // disagree on scale signedness without a test failing.
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.023).sin()).collect();
        let expected: f32 = Q6_K_SIGNED_SCALES_GOLDEN
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
        let scalar = dot_q6_k_f32_scalar(&Q6_K_SIGNED_SCALES_TEST_BLOCK, &x);
        assert!(
            (scalar - expected).abs() < 1e-2,
            "scalar={scalar} expected={expected}"
        );
    }

    #[test]
    fn q4_k_and_q6_k_reject_misaligned_buffers() {
        let bad = vec![0u8; 5];
        assert!(dequant_q4_k(&bad).is_err());
        assert!(dequant_q6_k(&bad).is_err());
    }

    // Generated by an independent Python reference -- do not hand-edit.
    // Random-but-well-formed block bytes (d/dmin/d_all pinned to
    // realistic small scales to keep golden values readable and avoid
    // any risk of an f16 NaN/Inf bit pattern; scales/qs/hmask/qh fully
    // random) cross-validated against an independent Python
    // dequantizer written from the same public layout description.
    const Q2_K_TEST_BLOCK: [u8; 84] = [
        0x92, 0x32, 0xc9, 0x0e, 0x0f, 0xf8, 0x10, 0xf0, 0xd1, 0x82, 0xca, 0x81, 0x7f, 0x11, 0xdb,
        0xff, 0x78, 0xf8, 0xab, 0xc5, 0x60, 0x0c, 0xc0, 0xbc, 0xa6, 0x52, 0x56, 0x1b, 0xc0, 0x36,
        0x6b, 0x6e, 0xbb, 0x53, 0x32, 0x90, 0x0a, 0x41, 0x67, 0x97, 0x48, 0x76, 0x86, 0x23, 0xd5,
        0x8e, 0x9e, 0x02, 0xc1, 0x1b, 0xea, 0x9c, 0xb7, 0x55, 0xc3, 0x1b, 0xf4, 0x59, 0xc6, 0xef,
        0x11, 0x61, 0xbc, 0x54, 0xd7, 0x8a, 0x6d, 0xed, 0x9e, 0xe7, 0x48, 0x69, 0x8e, 0x3a, 0x30,
        0x6c, 0xd8, 0xdc, 0x85, 0xc1, 0xec, 0x35, 0x14, 0x32,
    ];
    const Q2_K_GOLDEN: [f32; 256] = [
        -1.70947, -1.70947, 0.51123, -0.969238, -1.70947, -1.70947, -1.70947, -1.70947, -0.229004,
        -0.229004, -0.229004, 0.51123, -1.70947, -0.229004, 0.51123, -0.229004, 1.65088, 1.65088,
        0.910645, -0.569824, 0.910645, 0.17041, 1.65088, 1.65088, -0.569824, 0.910645, 0.910645,
        1.65088, 0.17041, 0.910645, 0.910645, 0.910645, 4.38281, 4.38281, 4.38281, 1.05176,
        -2.2793, 7.71387, -2.2793, 7.71387, 1.05176, -2.2793, 1.05176, 4.38281, -2.2793, 1.05176,
        4.38281, 7.71387, 10.3633, 0.0, 0.0, 0.0, 10.3633, 0.0, 5.18164, 5.18164, 10.3633, 5.18164,
        5.18164, 0.0, 5.18164, 15.5449, 15.5449, 0.0, 16.6553, 16.6553, 11.1035, 0.0, 11.1035, 0.0,
        0.0, 16.6553, 11.1035, 5.55176, 5.55176, 5.55176, 0.0, 16.6553, 11.1035, 11.1035, 6.03369,
        0.111816, 6.03369, 0.111816, -2.84912, -2.84912, 3.07275, 0.111816, -2.84912, 6.03369,
        -2.84912, 3.07275, 0.111816, -2.84912, 0.111816, -2.84912, -0.189941, -0.189941, -0.189941,
        -0.189941, -0.189941, -0.189941, -0.189941, -0.189941, -0.189941, -0.189941, -0.189941,
        -0.189941, -0.189941, -0.189941, -0.189941, -0.189941, -2.84912, -2.84912, -2.84912,
        -2.84912, -2.84912, -2.84912, -2.84912, -2.84912, -2.84912, -2.84912, -2.84912, -2.84912,
        -2.84912, -2.84912, -2.84912, -2.84912, -2.09912, -1.35889, -1.729, -2.46924, -1.35889,
        -2.09912, -1.35889, -1.35889, -2.46924, -2.09912, -1.729, -1.35889, -2.09912, -2.09912,
        -2.46924, -2.46924, 0.701172, -0.0390625, -0.779297, -0.779297, -0.0390625, 0.701172,
        -1.51953, -0.779297, -0.0390625, -0.0390625, -1.51953, -1.51953, -1.51953, -1.51953,
        -0.779297, -0.779297, -2.2793, 5.12305, 5.12305, 8.82422, 1.42188, 1.42188, -2.2793,
        5.12305, 1.42188, 5.12305, 1.42188, 8.82422, -2.2793, -2.2793, 8.82422, 1.42188, -1.14941,
        -0.779297, -0.40918, -0.40918, -0.40918, -1.14941, -0.779297, -0.779297, -0.40918,
        -0.779297, -1.51953, -0.40918, -0.779297, -0.40918, -1.14941, -1.51953, -1.32959, 4.22217,
        9.77393, 4.22217, 15.3257, 4.22217, -1.32959, 4.22217, 15.3257, 4.22217, -1.32959, 9.77393,
        4.22217, 9.77393, 15.3257, 4.22217, 0.180176, -0.189941, 0.550293, 0.550293, 0.180176,
        0.550293, -0.189941, 0.550293, -0.189941, 0.92041, 0.92041, 0.550293, 0.180176, 0.180176,
        -0.189941, -0.189941, 9.74463, -2.46924, 9.74463, 5.67334, 5.67334, 1.60205, 9.74463,
        -2.46924, 9.74463, 1.60205, 9.74463, 9.74463, -2.46924, 1.60205, 5.67334, 1.60205, 13.8062,
        8.25439, 2.70264, 13.8062, 8.25439, 13.8062, 2.70264, 2.70264, 8.25439, -2.84912, -2.84912,
        2.70264, 13.8062, 13.8062, 8.25439, 13.8062,
    ];

    const Q3_K_TEST_BLOCK: [u8; 110] = [
        0x56, 0xf2, 0xb4, 0x2b, 0xd5, 0x6f, 0x51, 0x71, 0x3c, 0x0a, 0xb9, 0x1d, 0xd0, 0xb9, 0x3b,
        0xb3, 0x0f, 0xff, 0x8c, 0xb2, 0x83, 0x3a, 0x3d, 0x24, 0xb1, 0x12, 0x56, 0xe3, 0x23, 0x54,
        0xf2, 0xfa, 0x7f, 0xdf, 0x31, 0xe1, 0x18, 0x26, 0x6e, 0xcd, 0x5b, 0x38, 0xee, 0xbd, 0x9f,
        0x8c, 0x57, 0x47, 0x0b, 0x11, 0xcb, 0xfb, 0xb4, 0x83, 0xa0, 0x4e, 0x0b, 0xd4, 0xa7, 0x85,
        0xe0, 0x60, 0xf3, 0xb3, 0xe3, 0x95, 0x43, 0xc6, 0x05, 0x05, 0x77, 0x53, 0xed, 0x23, 0xcc,
        0x6a, 0x0e, 0x89, 0xa1, 0x79, 0x85, 0xf6, 0x6e, 0x5a, 0x23, 0x63, 0xbe, 0x53, 0xfa, 0xa2,
        0x2b, 0xe9, 0xcd, 0xce, 0xf8, 0x3d, 0x6f, 0xd0, 0x42, 0x6e, 0x3b, 0x7f, 0x23, 0x26, 0xd3,
        0xb9, 0x18, 0xbf, 0xa4, 0x34,
    ];
    const Q3_K_GOLDEN: [f32; 256] = [
        -8.99121, -8.99121, -26.9736, 8.99121, 0.0, 17.9824, 17.9824, 8.99121, -8.99121, -35.9648,
        17.9824, 8.99121, -8.99121, 0.0, 26.9736, 26.9736, -13.9219, -4.64062, 4.64062, 4.64062,
        -0.0, 4.64062, -0.0, 9.28125, -13.9219, 18.5625, 4.64062, -4.64062, -0.0, 18.5625, 4.64062,
        4.64062, -26.1035, -26.1035, 34.8047, -0.0, 17.4023, -8.70117, 8.70117, 8.70117, 17.4023,
        -17.4023, 8.70117, 8.70117, 8.70117, 8.70117, -8.70117, -8.70117, 17.4023, 0.0, -17.4023,
        17.4023, 8.70117, 0.0, -34.8047, -8.70117, -17.4023, 8.70117, 8.70117, 8.70117, 0.0,
        -34.8047, 0.0, 0.0, -18.2725, 18.2725, -18.2725, 12.1816, -6.09082, -12.1816, 12.1816,
        24.3633, -6.09082, 6.09082, 12.1816, -18.2725, 18.2725, 24.3633, 18.2725, 24.3633, 0.0,
        4.35059, 0.0, -4.35059, -4.35059, -17.4023, 8.70117, 0.0, -17.4023, -13.0518, 8.70117,
        -17.4023, -8.70117, 8.70117, -4.35059, -4.35059, -2.61035, -0.870117, -3.48047, 2.61035,
        -3.48047, 0.0, -2.61035, -0.870117, 0.870117, 0.0, 2.61035, 1.74023, -1.74023, 1.74023,
        0.870117, -2.61035, 0.0, 0.0, 19.1426, -6.38086, -12.7617, 12.7617, 12.7617, -19.1426,
        -25.5234, -6.38086, -12.7617, -12.7617, -6.38086, -19.1426, -6.38086, 12.7617, -8.70117,
        -2.90039, -8.70117, 5.80078, -2.90039, 8.70117, -8.70117, -8.70117, -2.90039, 2.90039,
        -0.0, -5.80078, -5.80078, -2.90039, -2.90039, -2.90039, -25.2334, 16.8223, -16.8223,
        16.8223, -8.41113, 25.2334, 16.8223, -8.41113, 16.8223, 16.8223, 25.2334, -25.2334,
        -25.2334, 16.8223, 0.0, 8.41113, 13.9219, -3.48047, -0.0, -3.48047, 10.4414, -3.48047,
        10.4414, -0.0, -10.4414, 13.9219, -10.4414, 6.96094, 3.48047, -6.96094, -0.0, -6.96094,
        -19.1426, 6.38086, -6.38086, 12.7617, -25.5234, 0.0, 19.1426, 0.0, 12.7617, -25.5234,
        -12.7617, 12.7617, 19.1426, -6.38086, 12.7617, 19.1426, 11.0215, 5.51074, -22.043, -22.043,
        0.0, 0.0, 16.5322, 5.51074, -11.0215, -11.0215, -22.043, -11.0215, 0.0, -22.043, -11.0215,
        -5.51074, -8.12109, 6.09082, -4.06055, -6.09082, -4.06055, -4.06055, -2.03027, -6.09082,
        -2.03027, -4.06055, 4.06055, 4.06055, -8.12109, 0.0, 6.09082, 6.09082, 8.70117, -17.4023,
        -8.70117, 8.70117, -0.0, 34.8047, 26.1035, 26.1035, 8.70117, 34.8047, -26.1035, 26.1035,
        -0.0, -17.4023, 17.4023, -8.70117, -1.16016, 1.74023, 0.580078, 0.580078, 0.0, -1.74023,
        -1.16016, -1.74023, 1.74023, -1.16016, -2.32031, 1.74023, -0.580078, -0.580078, 1.74023,
        0.0,
    ];

    #[test]
    fn q2_k_dequant_matches_independent_python_reference() {
        let got = dequant_q2_k(&Q2_K_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), Q2_K_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(Q2_K_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "Q2_K element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn q2_k_fused_dot_matches_dequant_then_dot() {
        let dequanted = dequant_q2_k(&Q2_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.019).sin()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_q2_k_f32(&Q2_K_TEST_BLOCK, &x);
        assert!(
            (fused - expected).abs() < 1e-1,
            "fused={fused} expected={expected}"
        );
    }

    #[test]
    fn q3_k_dequant_matches_independent_python_reference() {
        let got = dequant_q3_k(&Q3_K_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), Q3_K_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(Q3_K_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "Q3_K element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn q3_k_fused_dot_matches_dequant_then_dot() {
        let dequanted = dequant_q3_k(&Q3_K_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.023).cos()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_q3_k_f32(&Q3_K_TEST_BLOCK, &x);
        assert!(
            (fused - expected).abs() < 1e-1,
            "fused={fused} expected={expected}"
        );
    }

    #[test]
    fn q2_k_and_q3_k_reject_misaligned_buffers() {
        let bad = vec![0u8; 5];
        assert!(dequant_q2_k(&bad).is_err());
        assert!(dequant_q3_k(&bad).is_err());
    }

    // Generated by an independent Python reference -- do not hand-edit.
    // Random-but-well-formed block bytes (d pinned to a realistic small
    // scale; qs/scales_l/scales_h fully random) cross-validated against
    // an independent Python dequantizer written from the same public
    // layout description (real ggml-quants.c / ggml-common.h source).
    const IQ4_NL_TEST_BLOCK: [u8; 18] = [
        0xf6, 0x34, 0x3c, 0x7f, 0x90, 0x6a, 0xdc, 0x0f, 0x77, 0xfc, 0xb9, 0x1c, 0xdf, 0x74, 0xe0,
        0x40, 0x5d, 0xf3,
    ];
    const IQ4_NL_GOLDEN: [f32; 32] = [
        16.4331, 35.0366, -39.3774, 7.75146, 16.4331, 35.0366, -3.10059, 16.4331, 4.03076, 16.4331,
        35.0366, -15.1929, -39.3774, -39.3774, 21.394, -20.1538, -20.1538, -3.10059, 4.03076,
        -6.82129, 21.394, -39.3774, -3.10059, 35.0366, 11.7822, -32.2461, 21.394, -3.10059,
        27.5952, -15.1929, -10.8521, 35.0366,
    ];

    const IQ4_XS_TEST_BLOCK: [u8; 136] = [
        0x5c, 0x33, 0xb4, 0x39, 0xd1, 0x64, 0x97, 0x82, 0xcb, 0xbd, 0x88, 0x95, 0xf3, 0x60, 0x2a,
        0xb5, 0xe7, 0x24, 0xd3, 0xee, 0xfe, 0x71, 0x13, 0xbe, 0x70, 0x84, 0x48, 0x79, 0x7b, 0x3e,
        0xf0, 0x55, 0xdc, 0xb2, 0xb2, 0xde, 0x32, 0xa1, 0x5b, 0x02, 0x01, 0xdc, 0x2a, 0xbb, 0xf7,
        0x0b, 0x8a, 0x88, 0xdd, 0x0b, 0x02, 0x7e, 0x5e, 0x76, 0x87, 0x30, 0x1e, 0x1c, 0xcf, 0x48,
        0xd7, 0x61, 0xf3, 0x51, 0x52, 0x17, 0x98, 0x0a, 0x87, 0xcf, 0x02, 0x91, 0xc8, 0xee, 0xc0,
        0x91, 0x69, 0x2a, 0x4f, 0x64, 0x68, 0xa7, 0xb2, 0xe6, 0x98, 0x21, 0x81, 0x75, 0x53, 0x2a,
        0x8d, 0x12, 0xae, 0xe0, 0xea, 0x0c, 0x75, 0xff, 0x22, 0x5e, 0x25, 0x19, 0xda, 0x2e, 0x51,
        0x4e, 0x81, 0xdc, 0x0e, 0x78, 0x86, 0xd7, 0x58, 0xb5, 0xb7, 0xf6, 0x45, 0xa9, 0x0a, 0x83,
        0xfd, 0x2a, 0x12, 0x7d, 0xf0, 0x12, 0x97, 0xe2, 0xfe, 0xf4, 0xd0, 0xa2, 0x11, 0x14, 0x78,
        0xdb,
    ];
    const IQ4_XS_GOLDEN: [f32; 256] = [
        -270.917, -491.928, -7.12939, 249.529, 463.411, 905.433, -178.235, 249.529, 71.2939,
        349.34, 463.411, -634.516, -634.516, 741.457, 463.411, -634.516, -377.858, -270.917,
        -7.12939, -92.6821, -805.622, 156.847, 591.74, -270.917, -634.516, 591.74, -491.928,
        -634.516, -805.622, 71.2939, 741.457, -270.917, 87.6226, 33.8071, -0.689941, -8.96924,
        -26.2178, -61.4048, 87.6226, 24.1479, -36.5669, 57.2651, 57.2651, -61.4048, 57.2651,
        71.7539, -26.2178, 57.2651, 6.89941, -0.689941, 33.8071, 6.89941, 6.89941, 44.8462,
        -77.9634, 24.1479, -47.606, -26.2178, -26.2178, -47.606, 44.8462, -17.2485, 24.1479,
        87.6226, -478.359, 243.779, 114.99, 174.785, -45.9961, 174.785, 114.99, 4.59961, 317.373,
        174.785, -381.768, 409.365, 409.365, -101.191, -45.9961, -584.15, -584.15, 317.373,
        -381.768, 174.785, 519.756, -584.15, 4.59961, 4.59961, 317.373, -584.15, -584.15, -45.9961,
        -160.986, -45.9961, 4.59961, -298.975, 122.81, 73.1338, 155.927, 1.37988, -13.7988,
        -143.508, -89.6924, -143.508, -114.53, -13.7988, 1.37988, 34.4971, -13.7988, 155.927,
        -114.53, -143.508, -143.508, -143.508, 73.1338, -67.6143, 95.2119, -30.3574, 155.927,
        -48.2959, -48.2959, -143.508, 17.9385, -175.245, 1.37988, 73.1338, -175.245, 17.9385,
        -2.06982, -184.214, 262.868, 215.262, -26.9077, -51.7456, -233.89, 101.421, -2.06982,
        20.6982, 171.795, 45.5361, -2.06982, 215.262, 215.262, 72.4438, -109.701, -184.214,
        -109.701, -26.9077, 45.5361, 171.795, 101.421, 45.5361, 45.5361, -51.7456, -78.6533,
        -184.214, -26.9077, 171.795, -2.06982, 20.6982, -134.539, 51.7456, 142.818, -171.795,
        184.214, -262.868, 51.7456, 109.701, -72.4438, 233.89, -171.795, 184.214, -72.4438,
        26.9077, 51.7456, 184.214, -72.4438, -171.795, 2.06982, -215.262, 51.7456, 184.214,
        184.214, -262.868, -20.6982, 233.89, -171.795, -72.4438, -171.795, -215.262, 142.818,
        -171.795, -430.523, 368.429, -430.523, 219.401, 368.429, 4.13965, -91.0723, -41.3965,
        4.13965, -144.888, -41.3965, -91.0723, -144.888, 53.8154, 103.491, -269.077, -144.888,
        -202.843, 4.13965, 285.636, -525.735, -41.3965, 4.13965, 285.636, -144.888, 157.307,
        157.307, 467.78, -202.843, 103.491, -525.735, 4.13965, -380.848, -137.988, 458.121,
        -380.848, 700.98, 458.121, 55.1953, 458.121, -491.238, 270.457, 700.98, 458.121, 574.031,
        270.457, -5.51953, -209.742, -623.707, 458.121, 574.031, 55.1953, -623.707, 574.031,
        -71.7539, -491.238, -623.707, -623.707, -380.848, -137.988, 574.031, 574.031, 55.1953,
        -380.848,
    ];

    #[test]
    fn iq4_nl_dequant_matches_independent_python_reference() {
        let got = dequant_iq4_nl(&IQ4_NL_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), IQ4_NL_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(IQ4_NL_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "IQ4_NL element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn iq4_nl_fused_dot_matches_dequant_then_dot() {
        let dequanted = dequant_iq4_nl(&IQ4_NL_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.019).sin()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_iq4_nl_f32(&IQ4_NL_TEST_BLOCK, &x);
        assert!(
            (fused - expected).abs() < 1e-1,
            "fused={fused} expected={expected}"
        );
    }

    #[test]
    fn iq4_xs_dequant_matches_independent_python_reference() {
        let got = dequant_iq4_xs(&IQ4_XS_TEST_BLOCK).unwrap();
        assert_eq!(got.len(), IQ4_XS_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(IQ4_XS_GOLDEN.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-1,
                "IQ4_XS element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn iq4_xs_fused_dot_matches_dequant_then_dot() {
        let dequanted = dequant_iq4_xs(&IQ4_XS_TEST_BLOCK).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.023).cos()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_iq4_xs_f32(&IQ4_XS_TEST_BLOCK, &x);
        assert!(
            (fused - expected).abs() < 1e-1,
            "fused={fused} expected={expected}"
        );
    }

    #[test]
    fn iq4_nl_and_iq4_xs_reject_misaligned_buffers() {
        let bad = vec![0u8; 5];
        assert!(dequant_iq4_nl(&bad).is_err());
        assert!(dequant_iq4_xs(&bad).is_err());
    }

    // Generated by an independent Python reference -- do not hand-edit. Scale
    // bytes deliberately span e=0 (2^-127, the special subnormal-adjacent
    // case) and a mid-range exponent (e=130 -> 2^3 = 8.0), packed nibbles
    // fully random.
    const MXFP4_TEST_PACKED: [u8; 32] = [
        0xaa, 0xf9, 0x12, 0xda, 0x04, 0xac, 0xce, 0x2d, 0xbf, 0x4c, 0xc3, 0x06, 0x67, 0x59, 0xd1,
        0xa3, 0xea, 0xf1, 0x8f, 0x5d, 0xe5, 0xe6, 0x9e, 0x77, 0x73, 0x9c, 0x6f, 0x14, 0x5f, 0x1f,
        0xd9, 0x5e,
    ];
    const MXFP4_TEST_SCALES: [u8; 2] = [0x00, 0x82];
    const MXFP4_GOLDEN: [f32; 64] = [
        -5.87747e-39,
        -2.93874e-39,
        5.87747e-39,
        -5.87747e-39,
        1.17549e-38,
        -1.17549e-38,
        -2.35099e-38,
        -1.76324e-38,
        -3.52648e-38,
        -1.17549e-38,
        8.81621e-39,
        2.35099e-38,
        3.52648e-38,
        -2.93874e-39,
        2.93874e-39,
        8.81621e-39,
        -5.87747e-39,
        -3.52648e-38,
        2.93874e-39,
        -1.76324e-38,
        0.0,
        -5.87747e-39,
        -1.17549e-38,
        5.87747e-39,
        -8.81621e-39,
        1.17549e-38,
        -1.17549e-38,
        0.0,
        2.35099e-38,
        1.76324e-38,
        -1.76324e-38,
        -5.87747e-39,
        -8.0,
        4.0,
        -48.0,
        -24.0,
        24.0,
        32.0,
        -32.0,
        48.0,
        12.0,
        -16.0,
        -48.0,
        16.0,
        -48.0,
        -48.0,
        -4.0,
        -32.0,
        -32.0,
        -48.0,
        -0.0,
        24.0,
        -32.0,
        -32.0,
        -4.0,
        48.0,
        48.0,
        -4.0,
        32.0,
        4.0,
        24.0,
        4.0,
        -24.0,
        24.0,
    ];

    #[test]
    fn mxfp4_dequant_matches_independent_python_reference() {
        let got = dequant_mxfp4_row(&MXFP4_TEST_PACKED, &MXFP4_TEST_SCALES).unwrap();
        assert_eq!(got.len(), MXFP4_GOLDEN.len());
        for (i, (a, b)) in got.iter().zip(MXFP4_GOLDEN.iter()).enumerate() {
            let tol = 1e-38f32.max(b.abs() * 1e-3);
            assert!(
                (a - b).abs() < tol,
                "MXFP4 element {i}: rust={a} python={b}"
            );
        }
    }

    #[test]
    fn mxfp4_fused_dot_matches_dequant_then_dot() {
        let dequanted = dequant_mxfp4_row(&MXFP4_TEST_PACKED, &MXFP4_TEST_SCALES).unwrap();
        let x: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.037).sin()).collect();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let fused = dot_mxfp4_row_f32(&MXFP4_TEST_PACKED, &MXFP4_TEST_SCALES, &x);
        assert!(
            (fused - expected).abs() < 1e-3,
            "fused={fused} expected={expected}"
        );
    }

    #[test]
    fn mxfp4_scale_byte_zero_and_max_match_the_e8m0_formula() {
        // e=0 is the special subnormal-adjacent case (2^-127); e=127 is
        // the OCP MX bias point (scale 1.0, i.e. the E2M1 values verbatim).
        assert!((e8m0_scale(0) - 2f32.powi(-127)).abs() < 1e-45);
        assert_eq!(e8m0_scale(127), 1.0);
        assert_eq!(e8m0_scale(128), 2.0);
    }

    #[test]
    fn mxfp4_simd_dispatch_matches_scalar_across_every_possible_packed_byte_value() {
        // 16 groups of 16 bytes each = 256 total packed bytes, covering
        // every possible u8 value exactly once (each byte encodes 2
        // nibbles, so this exercises every (lo_nibble, hi_nibble) pair
        // the real E2M1 codebook can ever see) -- exhaustive coverage
        // for the SIMD decode logic (mxfp4_nibbles_to_f32_quads /
        // mxfp4_nibbles_to_f32x8), which is new, hand-derived
        // arithmetic (not a direct port of already-tested code) and so
        // needs its own thorough cross-validation against the scalar
        // KVALUES_MXFP4 table lookup, not just the one golden fixture
        // above.
        let packed: Vec<u8> = (0..=255u8).collect();
        let n_groups = packed.len() / (MXFP4_GROUP_SIZE / 2);
        // Varied scale bytes (not all identical), staying within the
        // realistic/non-overflowing range this module's own doc
        // comments already establish (0xFF reserved for NaN; very high
        // bytes combined with E2M1's max magnitude of 6 can legitimately
        // overflow f32::MAX).
        let scales: Vec<u8> = (0..n_groups).map(|i| ((i * 17 + 3) % 180) as u8).collect();
        let x: Vec<f32> = (0..n_groups * MXFP4_GROUP_SIZE)
            .map(|i| ((i as f32) * 0.013).cos())
            .collect();

        let scalar = dot_mxfp4_row_f32_scalar(&packed, &scales, &x);
        let dispatched = dot_mxfp4_row_f32(&packed, &scales, &x);
        assert!(
            (scalar - dispatched).abs() < scalar.abs() * 1e-3 + 1e-3,
            "scalar={scalar} dispatched (SIMD)={dispatched}"
        );

        #[cfg(target_arch = "aarch64")]
        {
            let neon = unsafe { simd_aarch64::dot_mxfp4_row_f32_neon(&packed, &scales, &x) };
            assert!(
                (scalar - neon).abs() < scalar.abs() * 1e-3 + 1e-3,
                "scalar={scalar} neon={neon}"
            );
        }
    }

    #[test]
    fn mxfp4_rejects_a_packed_scales_length_mismatch() {
        let bad_packed = vec![0u8; 15]; // one byte short of 16 for a single 32-elem group
        let scales = [0u8; 1];
        assert!(matches!(
            dequant_mxfp4_row(&bad_packed, &scales),
            Err(QuantError::Mxfp4RowMismatch(15, 16))
        ));
    }

    /// Repeats a single-block golden fixture `n` times, so multi-block
    /// SIMD dispatch (not just a single loop iteration) gets exercised.
    fn repeat_block(block: &[u8], n: usize) -> Vec<u8> {
        block
            .iter()
            .copied()
            .cycle()
            .take(block.len() * n)
            .collect()
    }

    #[test]
    fn dispatched_q4_k_matches_scalar_reference_across_many_blocks() {
        let n_blocks = 4;
        let packed = repeat_block(&Q4_K_TEST_BLOCK, n_blocks);
        let x: Vec<f32> = (0..256 * n_blocks)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let dispatched = dot_q4_k_f32(&packed, &x);
        let scalar = dot_q4_k_f32_scalar(&packed, &x);
        assert!(
            (dispatched - scalar).abs() < 1e-1,
            "dispatched={dispatched} scalar={scalar}"
        );
    }

    #[test]
    fn dispatched_q5_k_matches_scalar_reference_across_many_blocks() {
        let n_blocks = 4;
        let packed = repeat_block(&Q5_K_TEST_BLOCK, n_blocks);
        let x: Vec<f32> = (0..256 * n_blocks)
            .map(|i| ((i as f32) * 0.011).cos())
            .collect();
        let dispatched = dot_q5_k_f32(&packed, &x);
        let scalar = dot_q5_k_f32_scalar(&packed, &x);
        assert!(
            (dispatched - scalar).abs() < 1e-1,
            "dispatched={dispatched} scalar={scalar}"
        );
    }

    #[test]
    fn dispatched_q6_k_matches_scalar_reference_across_many_blocks() {
        let n_blocks = 4;
        let packed = repeat_block(&Q6_K_TEST_BLOCK, n_blocks);
        let x: Vec<f32> = (0..256 * n_blocks)
            .map(|i| ((i as f32) * 0.019).sin())
            .collect();
        let dispatched = dot_q6_k_f32(&packed, &x);
        let scalar = dot_q6_k_f32_scalar(&packed, &x);
        assert!(
            (dispatched - scalar).abs() < 1e-1,
            "dispatched={dispatched} scalar={scalar}"
        );
    }

    #[test]
    fn dispatched_q6_k_matches_scalar_reference_with_negative_scales() {
        // Same shape as the test above, but on the negative-scale
        // fixture: this is the case where the scalar reference and the
        // SIMD kernels historically *disagreed* (scalar read the signed
        // scales as unsigned), so all-positive parity was vacuous.
        let n_blocks = 4;
        let packed = repeat_block(&Q6_K_SIGNED_SCALES_TEST_BLOCK, n_blocks);
        let x: Vec<f32> = (0..256 * n_blocks)
            .map(|i| ((i as f32) * 0.019).sin())
            .collect();
        let dispatched = dot_q6_k_f32(&packed, &x);
        let scalar = dot_q6_k_f32_scalar(&packed, &x);
        assert!(
            (dispatched - scalar).abs() < 1e-1,
            "dispatched={dispatched} scalar={scalar}"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_q4_k_kernel_matches_scalar_directly_when_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipping: host CPU lacks NEON");
            return;
        }
        let n_blocks = 4;
        let packed = repeat_block(&Q4_K_TEST_BLOCK, n_blocks);
        let x: Vec<f32> = (0..256 * n_blocks)
            .map(|i| ((i as f32) * 0.037).cos())
            .collect();
        let simd = unsafe { simd_aarch64::dot_q4_k_f32_neon(&packed, &x) };
        let scalar = dot_q4_k_f32_scalar(&packed, &x);
        assert!(
            (simd - scalar).abs() < 1e-1,
            "NEON Q4_K kernel diverged from scalar: simd={simd} scalar={scalar}"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_q5_k_q8_kernel_matches_scalar_directly_when_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipping: host CPU lacks NEON");
            return;
        }
        let n_blocks = 4;
        let packed = repeat_block(&Q5_K_TEST_BLOCK, n_blocks);
        let x: Vec<f32> = (0..256 * n_blocks)
            .map(|i| ((i as f32) * 0.029).sin())
            .collect();
        let act = quantize_activations_q8_k(&x);
        let dispatched = dot_q5_k_q8(&packed, &act);
        let scalar = dot_q5_k_q8_scalar(&packed, &act);
        assert_eq!(
            dispatched,
            scalar,
            "Q5_K×Q8_K dispatch must match scalar (dotprod={})",
            std::arch::is_aarch64_feature_detected!("dotprod")
        );
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let sdot = unsafe { simd_aarch64::dot_q5_k_q8_neon_sdot(&packed, &act) };
            assert_eq!(sdot, scalar, "NEON SDOT Q5_K×Q8_K diverged from scalar");
        }
        if std::arch::is_aarch64_feature_detected!("neon") {
            let neon = unsafe { simd_aarch64::dot_q5_k_q8_neon(&packed, &act) };
            assert_eq!(neon, scalar, "NEON widen Q5_K×Q8_K diverged from scalar");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_q5_k_kernel_matches_scalar_directly_when_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipping: host CPU lacks NEON");
            return;
        }
        let n_blocks = 4;
        let packed = repeat_block(&Q5_K_TEST_BLOCK, n_blocks);
        let x: Vec<f32> = (0..256 * n_blocks)
            .map(|i| ((i as f32) * 0.029).sin())
            .collect();
        let simd = unsafe { simd_aarch64::dot_q5_k_f32_neon(&packed, &x) };
        let scalar = dot_q5_k_f32_scalar(&packed, &x);
        assert!(
            (simd - scalar).abs() < 1e-1,
            "NEON Q5_K kernel diverged from scalar: simd={simd} scalar={scalar}"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_q6_k_kernel_matches_scalar_directly_when_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipping: host CPU lacks NEON");
            return;
        }
        let n_blocks = 4;
        let packed = repeat_block(&Q6_K_TEST_BLOCK, n_blocks);
        let x: Vec<f32> = (0..256 * n_blocks)
            .map(|i| ((i as f32) * 0.041).cos())
            .collect();
        let simd = unsafe { simd_aarch64::dot_q6_k_f32_neon(&packed, &x) };
        let scalar = dot_q6_k_f32_scalar(&packed, &x);
        assert!(
            (simd - scalar).abs() < 1e-1,
            "NEON Q6_K kernel diverged from scalar: simd={simd} scalar={scalar}"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_q6_k_kernel_matches_scalar_directly_on_negative_scales() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipping: host CPU lacks NEON");
            return;
        }
        let n_blocks = 4;
        let packed = repeat_block(&Q6_K_SIGNED_SCALES_TEST_BLOCK, n_blocks);
        let x: Vec<f32> = (0..256 * n_blocks)
            .map(|i| ((i as f32) * 0.041).cos())
            .collect();
        let simd = unsafe { simd_aarch64::dot_q6_k_f32_neon(&packed, &x) };
        let scalar = dot_q6_k_f32_scalar(&packed, &x);
        assert!(
            (simd - scalar).abs() < 1e-1,
            "NEON Q6_K kernel diverged from scalar on negative scales: simd={simd} scalar={scalar}"
        );
    }

    #[test]
    fn q4_k_scalar_matches_independent_python_reference_via_dispatch_entrypoint() {
        // The public `dot_q4_k_f32`/`dot_q5_k_f32`/`dot_q6_k_f32`
        // dispatch functions must still agree with the
        // already-Python-cross-validated dequant golden values, not
        // just with themselves -- guards against a SIMD kernel and the
        // scalar kernel agreeing with each other while both being
        // wrong in the same way.
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.017).sin()).collect();
        let dequanted = dequant_q4_k(&Q4_K_TEST_BLOCK).unwrap();
        let expected: f32 = dequanted.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let dispatched = dot_q4_k_f32(&Q4_K_TEST_BLOCK, &x);
        assert!((dispatched - expected).abs() < 1e-2);
    }

    // --- SIMD coverage for the 8 previously-scalar-only formats ---

    fn q4_1_test_block() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&f16::from_f32(0.3).to_le_bytes());
        b.extend_from_slice(&f16::from_f32(-1.2).to_le_bytes());
        b.extend_from_slice(
            &(0..16)
                .map(|i| (i as u8) | ((15 - i as u8) << 4))
                .collect::<Vec<u8>>(),
        );
        b
    }

    fn q5_0_test_block() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&f16::from_f32(0.4).to_le_bytes());
        b.extend_from_slice(&[0xA5, 0x3C, 0x00, 0xFF]);
        b.extend_from_slice(
            &(0..16)
                .map(|i| (i as u8) | ((15 - i as u8) << 4))
                .collect::<Vec<u8>>(),
        );
        b
    }

    fn q5_1_test_block() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&f16::from_f32(0.2).to_le_bytes());
        b.extend_from_slice(&f16::from_f32(0.9).to_le_bytes());
        b.extend_from_slice(&[0x12, 0x34, 0x56, 0x78]);
        b.extend_from_slice(
            &(0..16)
                .map(|i| (i as u8) | ((15 - i as u8) << 4))
                .collect::<Vec<u8>>(),
        );
        b
    }

    fn q8_1_test_block() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&f16::from_f32(0.6).to_le_bytes());
        b.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
        let qs: Vec<i8> = (0..32).map(|i| ((i * 7) % 61) as i8 - 30).collect();
        b.extend_from_slice(&i8_to_u8_bytes(&qs));
        b
    }

    #[test]
    fn dispatched_matches_scalar_for_the_8_newly_simd_formats_across_many_blocks() {
        let n_blocks = 4;

        let q4_1 = repeat_block(&q4_1_test_block(), n_blocks);
        let x32 = |seed: f32| -> Vec<f32> {
            (0..32 * n_blocks)
                .map(|i| ((i as f32) * seed).sin())
                .collect()
        };
        let x = x32(0.031);
        assert!((dot_q4_1_f32(&q4_1, &x) - dot_q4_1_f32_scalar(&q4_1, &x)).abs() < 1e-1);

        let q5_0 = repeat_block(&q5_0_test_block(), n_blocks);
        let x = x32(0.037);
        assert!((dot_q5_0_f32(&q5_0, &x) - dot_q5_0_f32_scalar(&q5_0, &x)).abs() < 1e-1);

        let q5_1 = repeat_block(&q5_1_test_block(), n_blocks);
        let x = x32(0.041);
        assert!((dot_q5_1_f32(&q5_1, &x) - dot_q5_1_f32_scalar(&q5_1, &x)).abs() < 1e-1);

        let q8_1 = repeat_block(&q8_1_test_block(), n_blocks);
        let x = x32(0.043);
        assert!((dot_q8_1_f32(&q8_1, &x) - dot_q8_1_f32_scalar(&q8_1, &x)).abs() < 1e-1);

        let q2_k = repeat_block(&Q2_K_TEST_BLOCK, n_blocks);
        let x256 = |seed: f32| -> Vec<f32> {
            (0..256 * n_blocks)
                .map(|i| ((i as f32) * seed).cos())
                .collect()
        };
        let x = x256(0.013);
        assert!((dot_q2_k_f32(&q2_k, &x) - dot_q2_k_f32_scalar(&q2_k, &x)).abs() < 1e-1);

        let q3_k = repeat_block(&Q3_K_TEST_BLOCK, n_blocks);
        let x = x256(0.017);
        assert!((dot_q3_k_f32(&q3_k, &x) - dot_q3_k_f32_scalar(&q3_k, &x)).abs() < 1e-1);

        let iq4_nl = repeat_block(&IQ4_NL_TEST_BLOCK, n_blocks);
        let x = x32(0.019);
        assert!((dot_iq4_nl_f32(&iq4_nl, &x) - dot_iq4_nl_f32_scalar(&iq4_nl, &x)).abs() < 1e-1);

        let iq4_xs = repeat_block(&IQ4_XS_TEST_BLOCK, n_blocks);
        let x = x256(0.023);
        assert!((dot_iq4_xs_f32(&iq4_xs, &x) - dot_iq4_xs_f32_scalar(&iq4_xs, &x)).abs() < 1e-1);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_kernels_match_scalar_directly_for_the_8_newly_simd_formats() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipping: host CPU lacks NEON");
            return;
        }
        let n_blocks = 4;
        let x32 = |seed: f32| -> Vec<f32> {
            (0..32 * n_blocks)
                .map(|i| ((i as f32) * seed).sin())
                .collect()
        };
        let x256 = |seed: f32| -> Vec<f32> {
            (0..256 * n_blocks)
                .map(|i| ((i as f32) * seed).cos())
                .collect()
        };

        let q4_1 = repeat_block(&q4_1_test_block(), n_blocks);
        let x = x32(0.031);
        let simd = unsafe { simd_aarch64::dot_q4_1_f32_neon(&q4_1, &x) };
        assert!((simd - dot_q4_1_f32_scalar(&q4_1, &x)).abs() < 1e-1);

        let q5_0 = repeat_block(&q5_0_test_block(), n_blocks);
        let x = x32(0.037);
        let simd = unsafe { simd_aarch64::dot_q5_0_f32_neon(&q5_0, &x) };
        assert!((simd - dot_q5_0_f32_scalar(&q5_0, &x)).abs() < 1e-1);

        let q5_1 = repeat_block(&q5_1_test_block(), n_blocks);
        let x = x32(0.041);
        let simd = unsafe { simd_aarch64::dot_q5_1_f32_neon(&q5_1, &x) };
        assert!((simd - dot_q5_1_f32_scalar(&q5_1, &x)).abs() < 1e-1);

        let q8_1 = repeat_block(&q8_1_test_block(), n_blocks);
        let x = x32(0.043);
        let simd = unsafe { simd_aarch64::dot_q8_1_f32_neon(&q8_1, &x) };
        assert!((simd - dot_q8_1_f32_scalar(&q8_1, &x)).abs() < 1e-1);

        let q2_k = repeat_block(&Q2_K_TEST_BLOCK, n_blocks);
        let x = x256(0.013);
        let simd = unsafe { simd_aarch64::dot_q2_k_f32_neon(&q2_k, &x) };
        assert!((simd - dot_q2_k_f32_scalar(&q2_k, &x)).abs() < 1e-1);

        let q3_k = repeat_block(&Q3_K_TEST_BLOCK, n_blocks);
        let x = x256(0.017);
        let simd = unsafe { simd_aarch64::dot_q3_k_f32_neon(&q3_k, &x) };
        assert!((simd - dot_q3_k_f32_scalar(&q3_k, &x)).abs() < 1e-1);

        let iq4_nl = repeat_block(&IQ4_NL_TEST_BLOCK, n_blocks);
        let x = x32(0.019);
        let simd = unsafe { simd_aarch64::dot_iq4_nl_f32_neon(&iq4_nl, &x) };
        assert!((simd - dot_iq4_nl_f32_scalar(&iq4_nl, &x)).abs() < 1e-1);

        let iq4_xs = repeat_block(&IQ4_XS_TEST_BLOCK, n_blocks);
        let x = x256(0.023);
        let simd = unsafe { simd_aarch64::dot_iq4_xs_f32_neon(&iq4_xs, &x) };
        assert!((simd - dot_iq4_xs_f32_scalar(&iq4_xs, &x)).abs() < 1e-1);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernels_match_scalar_directly_for_the_8_newly_simd_formats() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            eprintln!("skipping: host CPU lacks AVX2+FMA");
            return;
        }
        let n_blocks = 4;
        let x32 = |seed: f32| -> Vec<f32> {
            (0..32 * n_blocks)
                .map(|i| ((i as f32) * seed).sin())
                .collect()
        };
        let x256 = |seed: f32| -> Vec<f32> {
            (0..256 * n_blocks)
                .map(|i| ((i as f32) * seed).cos())
                .collect()
        };

        let q4_1 = repeat_block(&q4_1_test_block(), n_blocks);
        let x = x32(0.031);
        let simd = unsafe { simd_x86::dot_q4_1_f32_avx2(&q4_1, &x) };
        assert!((simd - dot_q4_1_f32_scalar(&q4_1, &x)).abs() < 1e-1);

        let q5_0 = repeat_block(&q5_0_test_block(), n_blocks);
        let x = x32(0.037);
        let simd = unsafe { simd_x86::dot_q5_0_f32_avx2(&q5_0, &x) };
        assert!((simd - dot_q5_0_f32_scalar(&q5_0, &x)).abs() < 1e-1);

        let q5_1 = repeat_block(&q5_1_test_block(), n_blocks);
        let x = x32(0.041);
        let simd = unsafe { simd_x86::dot_q5_1_f32_avx2(&q5_1, &x) };
        assert!((simd - dot_q5_1_f32_scalar(&q5_1, &x)).abs() < 1e-1);

        let q8_1 = repeat_block(&q8_1_test_block(), n_blocks);
        let x = x32(0.043);
        let simd = unsafe { simd_x86::dot_q8_1_f32_avx2(&q8_1, &x) };
        assert!((simd - dot_q8_1_f32_scalar(&q8_1, &x)).abs() < 1e-1);

        let q2_k = repeat_block(&Q2_K_TEST_BLOCK, n_blocks);
        let x = x256(0.013);
        let simd = unsafe { simd_x86::dot_q2_k_f32_avx2(&q2_k, &x) };
        assert!((simd - dot_q2_k_f32_scalar(&q2_k, &x)).abs() < 1e-1);

        let q3_k = repeat_block(&Q3_K_TEST_BLOCK, n_blocks);
        let x = x256(0.017);
        let simd = unsafe { simd_x86::dot_q3_k_f32_avx2(&q3_k, &x) };
        assert!((simd - dot_q3_k_f32_scalar(&q3_k, &x)).abs() < 1e-1);

        let iq4_nl = repeat_block(&IQ4_NL_TEST_BLOCK, n_blocks);
        let x = x32(0.019);
        let simd = unsafe { simd_x86::dot_iq4_nl_f32_avx2(&iq4_nl, &x) };
        assert!((simd - dot_iq4_nl_f32_scalar(&iq4_nl, &x)).abs() < 1e-1);

        let iq4_xs = repeat_block(&IQ4_XS_TEST_BLOCK, n_blocks);
        let x = x256(0.023);
        let simd = unsafe { simd_x86::dot_iq4_xs_f32_avx2(&iq4_xs, &x) };
        assert!((simd - dot_iq4_xs_f32_scalar(&iq4_xs, &x)).abs() < 1e-1);
    }
}
