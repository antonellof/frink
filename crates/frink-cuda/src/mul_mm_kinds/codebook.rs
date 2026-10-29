//! The codebook formats the CUDA `mul_mm` GEMM consumes: IQ4_NL, IQ4_XS
//! and MXFP4. Each is a [`MulMmKind`] row plus the Rust twin of its
//! `dequant_src`, written statement for statement beside it.
//!
//! # What makes these a family
//!
//! Every other kind in the table is *affine*: the stored bits are a
//! magnitude, and dequantizing is `scale * code + bias`. These three
//! store an INDEX. A 4-bit code selects one of sixteen fixed values,
//! and only then is a scale applied. The sixteen values are part of the
//! format -- arbitrary constants no arithmetic re-derives -- so the
//! kernel needs the table resident, which is what
//! [`Codebook`](crate::mul_mm::Codebook) and the `__constant__` array
//! [`kernel_src`](crate::mul_mm::kernel_src) emits from it are for.
//!
//! The `dequant_src` / `dequant_twin` seam itself is unchanged: the
//! contract was always "given the block and `il`, write `SUB` floats in
//! ascending element order", and a table lookup satisfies it exactly as
//! an affine transform does.
//!
//! # Provenance
//!
//! Layout and arithmetic from llama.cpp
//! `ggml/src/ggml-cuda/dequantize.cuh:408` (`dequantize_iq4_nl`),
//! `:424` (`dequantize_iq4_xs`) and `:439` (`dequantize_mxfp4`), with
//! the codebooks at `ggml/src/ggml-common.h:1120` (`kvalues_iq4nl`) and
//! `:1126` (`kvalues_fp4`, aliased to `kvalues_mxfp4` at `:1129`).
//!
//! llama's `il`/`ib` decomposition is NOT reused: it splits a
//! super-block across 32 CUDA threads, four elements each, which is a
//! different partition from this GEMM's sixteen-element sub-block. The
//! *element mapping* is the thing taken, and the twins are held against
//! `frink_quant`'s whole-block dequants, which are independent
//! implementations of the same formats.
//!
//! # UNVERIFIED ON HARDWARE
//!
//! No kernel here has executed on a GPU. See [`crate::mul_mm`]'s module
//! docs for exactly what the twin and the host check do and do not
//! establish.

use crate::mul_mm::{f16_to_f32, Codebook, MulMmKind, SUB};

/// The 16-entry non-linear codebook IQ4_NL and IQ4_XS share.
///
/// ggml stores it as `int8_t`; it is held as `f32` here because that is
/// what both the emitted `__constant__` array and the twin's multiply
/// want, and every entry is exactly representable either way. Pinned to
/// `frink_quant`'s copy by `the_iq4_codebook_is_frink_quants` below,
/// which decodes a real block rather than comparing two literals.
const KVALUES_IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

/// The IQ4 codebook as a [`Codebook`] row. One constant, shared by both
/// kinds that use it and by both of their twins.
const IQ4NL_CODEBOOK: Codebook = Codebook {
    c_name: "frink_kvalues_iq4nl",
    values: &KVALUES_IQ4NL,
};

/// The E2M1 4-bit float codebook MXFP4 indexes: sign, two exponent
/// bits, one mantissa bit, per the OCP Microscaling Formats v1.0 spec.
///
/// These are the REAL values. ggml stores them doubled as `int8_t` and
/// pairs them with a halved scale (`ggml_e8m0_to_fp32_half`) purely to
/// keep its table integral; `frink_quant` uses the true values against
/// the true `2^(e-127)` scale, the products are identical, and this
/// follows `frink_quant` so that the twin can be held against
/// `dequant_mxfp4_gguf` bit for bit rather than approximately.
///
/// Code 8 is NEGATIVE zero, and it has to stay that way: it is the sign
/// bit set with exponent and mantissa clear, and `0.0` in its place
/// would be a different float.
const KVALUES_MXFP4: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

const MXFP4_CODEBOOK: Codebook = Codebook {
    c_name: "frink_kvalues_mxfp4",
    values: &KVALUES_MXFP4,
};

/// IQ4_NL: `half d`, then 16 bytes holding 32 four-bit codes. The low
/// nibble of byte `j` is element `j`, the high nibble is element
/// `j + 16` -- Q4_0's packing, with a codebook lookup where Q4_0 has a
/// `- 8` bias.
///
/// **`nl` is 2**, so `il` picks the LOW half (elements 0..16) or the
/// HIGH half (16..32) of one 32-element block. That is the same bit
/// that picks the nibble, which is why one `shift` serves both.
pub const IQ4_NL: MulMmKind = MulMmKind {
    name: "IQ4_NL",
    module_name: "frink_mul_mm_iq4_nl",
    fn_name: "iq4_nl_mul_mm",
    block_bytes: 18,
    block_elems: 32,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = frink_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const unsigned char* qs = xb + 2;
    const int shift = il ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        reg[i] = d * frink_kvalues_iq4nl[(qs[i] >> shift) & 0xF];
    }
}
"#,
    dequant_twin: dequant_sub_iq4_nl,
    codebook: Some(IQ4NL_CODEBOOK),
    grids: &[],
};

/// Scalar twin of [`IQ4_NL`]'s `dequant_src`.
fn dequant_sub_iq4_nl(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_to_f32(u16::from(xb[0]) | (u16::from(xb[1]) << 8));
    let qs = &xb[2..2 + 16];
    let shift = if il != 0 { 4 } else { 0 };
    for (r, q) in reg.iter_mut().zip(qs.iter()) {
        *r = d * KVALUES_IQ4NL[usize::from((q >> shift) & 0xF)];
    }
}

/// IQ4_XS: `half d`, `uint16 scales_h`, `uint8 scales_l[4]`, then 128
/// bytes holding 256 four-bit codes -- 136 bytes for a 256-element
/// super-block.
///
/// The super-block is eight 32-element groups, each with its own 6-bit
/// scale assembled from two places: the low four bits from a nibble of
/// `scales_l` and the high two from a 2-bit field of `scales_h`, then
/// biased by -32. A group's own 32 elements are packed the way IQ4_NL
/// packs a block, so with `nl` 16 the sub-block index splits as
/// `ib = il / 2` (which group) and `il & 1` (which nibble).
///
/// Note that `ib` is then used twice more, in two different ways:
/// `scales_l[ib / 2]` picks the byte and `4 * (ib & 1)` the nibble
/// inside it, while `scales_h` is shifted by `2 * ib`. Three
/// derivations of one index, which is the transcription error this
/// format invites.
pub const IQ4_XS: MulMmKind = MulMmKind {
    name: "IQ4_XS",
    module_name: "frink_mul_mm_iq4_xs",
    fn_name: "iq4_xs_mul_mm",
    block_bytes: 136,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = frink_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const unsigned int scales_h =
        (unsigned int)xb[2] | ((unsigned int)xb[3] << 8);
    const unsigned char* scales_l = xb + 4;

    const int ib = il / 2;
    const unsigned char* qs = xb + 8 + 16 * ib;
    const int shift = (il & 1) ? 4 : 0;

    const unsigned int ls =
        ((unsigned int)(scales_l[ib / 2] >> (4 * (ib & 1))) & 0xFu)
        | (((scales_h >> (2 * ib)) & 3u) << 4);
    const float dl = d * ((float)ls - 32.0f);
#pragma unroll
    for (int i = 0; i < 16; i++) {
        reg[i] = dl * frink_kvalues_iq4nl[(qs[i] >> shift) & 0xF];
    }
}
"#,
    dequant_twin: dequant_sub_iq4_xs,
    codebook: Some(IQ4NL_CODEBOOK),
    grids: &[],
};

/// Scalar twin of [`IQ4_XS`]'s `dequant_src`.
fn dequant_sub_iq4_xs(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_to_f32(u16::from(xb[0]) | (u16::from(xb[1]) << 8));
    let scales_h = u32::from(xb[2]) | (u32::from(xb[3]) << 8);
    let scales_l = &xb[4..8];

    let ib = il / 2;
    let qs = &xb[8 + 16 * ib..8 + 16 * ib + 16];
    let shift = if il & 1 != 0 { 4 } else { 0 };

    let ls =
        (u32::from(scales_l[ib / 2] >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
    let dl = d * (ls as f32 - 32.0);
    for (r, q) in reg.iter_mut().zip(qs.iter()) {
        *r = dl * KVALUES_IQ4NL[usize::from((q >> shift) & 0xF)];
    }
}

/// MXFP4 in its GGUF block form (ggml type tag 39): one E8M0 scale
/// byte, then 16 bytes holding 32 four-bit E2M1 codes, packed the way
/// IQ4_NL packs its block. 17 bytes for 32 elements.
///
/// **17 is odd**, which is the one thing about this row worth staring
/// at. Every other block in the table has an even stride, so a row of
/// blocks was always at least 2-byte aligned and the `half` at its head
/// could be read as one. Here the scale is a single byte and there is
/// no wider read to misalign, but the *next* block starts on an odd
/// offset -- so nothing in this kernel may assume a block pointer is
/// aligned to anything. It reads bytes only, and the GEMM body strides
/// by `FX_BLOCK_BYTES` as `size_t`, so both are already correct; this
/// note exists so a future widening does not quietly break it.
///
/// gpt-oss ships MXFP4. The CPU has had it since the Kimi K3 work and
/// no GPU backend has, so a gpt-oss prefill decodes every expert on the
/// host with the device idle.
pub const MXFP4: MulMmKind = MulMmKind {
    name: "MXFP4",
    module_name: "frink_mul_mm_mxfp4",
    fn_name: "mxfp4_mul_mm",
    block_bytes: 17,
    block_elems: 32,
    dequant_src: r#"
__device__ __forceinline__ float frink_e8m0_to_f32(unsigned char e) {
    // An E8M0 scale byte IS an f32 exponent field (bias 127), so
    // placing it there is exact rather than an approximation. `e == 0`
    // has to be special-cased: shifting it in would give 0.0, and the
    // format means 2^-127, which is the subnormal bit pattern below.
    // `e == 255` is reserved for NaN by the OCP spec and is not handled
    // here, matching ggml's own documented limitation.
    return e == 0 ? __int_as_float(0x00400000)
                  : __int_as_float((int)((unsigned int)e << 23));
}

__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = frink_e8m0_to_f32(xb[0]);
    const unsigned char* qs = xb + 1;
    const int shift = il ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        reg[i] = d * frink_kvalues_mxfp4[(qs[i] >> shift) & 0xF];
    }
}
"#,
    dequant_twin: dequant_sub_mxfp4,
    codebook: Some(MXFP4_CODEBOOK),
    grids: &[],
};

/// Scalar twin of the `frink_e8m0_to_f32` in [`MXFP4`]'s
/// `dequant_src`. Bit surgery on both sides, so this is bit-identical
/// rather than close; `frink_quant`'s `e8m0_scale` is the same three
/// lines and is private to that crate, so this is a transcription of
/// the arithmetic and not a call.
fn e8m0_to_f32(e: u8) -> f32 {
    if e == 0 {
        f32::from_bits(0x0040_0000)
    } else {
        f32::from_bits(u32::from(e) << 23)
    }
}

/// Scalar twin of [`MXFP4`]'s `dequant_src`.
fn dequant_sub_mxfp4(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = e8m0_to_f32(xb[0]);
    let qs = &xb[1..1 + 16];
    let shift = if il != 0 { 4 } else { 0 };
    for (r, q) in reg.iter_mut().zip(qs.iter()) {
        *r = d * KVALUES_MXFP4[usize::from((q >> shift) & 0xF)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The IQ4 codebook is sixteen arbitrary constants that no
    /// arithmetic re-derives, so a transposed pair is invisible in
    /// every way except the numbers coming out slightly wrong.
    ///
    /// It is checked by DECODING rather than by comparing two literal
    /// arrays: a block whose scale is exactly 1.0 and whose 16 bytes
    /// carry each code once in each nibble position dequantizes, under
    /// `frink_quant`, to the codebook itself. That reaches
    /// `frink_quant`'s private table through its public API, which
    /// comparing literals could not.
    ///
    /// Sabotage: swap two entries of `KVALUES_IQ4NL` and this names the
    /// index.
    #[test]
    fn the_iq4_codebook_is_frink_quants() {
        // 0x3C00 is f16 1.0, so the scale drops out of the product.
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C;
        // Byte j holds code j in its low nibble and code j in its high
        // nibble, so the block decodes to the codebook twice over.
        for j in 0..16usize {
            block[2 + j] = (j as u8) | ((j as u8) << 4);
        }
        let decoded = frink_quant::dequant_iq4_nl(&block).expect("iq4_nl dequant");
        assert_eq!(decoded.len(), 32);
        for (code, want) in KVALUES_IQ4NL.iter().enumerate() {
            assert_eq!(decoded[code], *want, "low nibble, code {code}");
            assert_eq!(decoded[16 + code], *want, "high nibble, code {code}");
        }
    }

    /// Same argument for the E2M1 table, through `dequant_mxfp4_gguf`.
    ///
    /// The scale byte is 127, which is `2^0`; that also pins
    /// [`e8m0_to_f32`]'s bias, because a bias-128 reading would make
    /// every value here exactly half.
    #[test]
    fn the_mxfp4_codebook_and_its_scale_bias_are_frink_quants() {
        let mut block = vec![0u8; 17];
        block[0] = 127;
        for j in 0..16usize {
            block[1 + j] = (j as u8) | ((j as u8) << 4);
        }
        let decoded = frink_quant::dequant_mxfp4_gguf(&block).expect("mxfp4 dequant");
        assert_eq!(decoded.len(), 32);
        assert_eq!(e8m0_to_f32(127), 1.0, "E8M0 bias is 127, not 128");
        for (code, want) in KVALUES_MXFP4.iter().enumerate() {
            assert_eq!(
                decoded[code].to_bits(),
                want.to_bits(),
                "low nibble, code {code}: negative zero is a distinct value here"
            );
            assert_eq!(decoded[16 + code].to_bits(), want.to_bits(), "high, {code}");
        }
    }

    /// `e8m0_to_f32` is bit surgery with a special case, and the
    /// special case is the whole reason it is not a one-liner. Walk
    /// every one of the 256 bytes and require `2^(e-127)`, computed a
    /// different way.
    #[test]
    fn every_e8m0_byte_is_two_to_the_e_minus_127() {
        for e in 0u32..=255 {
            let got = e8m0_to_f32(e as u8);
            let want = if e == 0 {
                // 2^-127 is subnormal in f32 and `powi(-127)` on the
                // literal 2.0 still produces it exactly, since it is a
                // power of two within the subnormal range.
                2f32.powi(-127)
            } else {
                2f32.powi(e as i32 - 127)
            };
            assert_eq!(got.to_bits(), want.to_bits(), "e = {e}");
        }
    }
}
