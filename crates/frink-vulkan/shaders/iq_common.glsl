// The codebook kinds, IQ1_S to IQ3_S. Each including shader defines
// IQ_BLOCK_BYTES and `float sub_dot(uint blk, uint il, uint xb)`, the
// dot of 16-value sub-block `il` of the 256-value block at byte offset
// `blk` with the activations from `xb`, then includes iq_main.glsl. One
// unit per sub-block, as ggml's and frink-cuda's dequants are cut
// (frink_cuda::mul_mm_kinds::iq, `frink_dequant_sub`).
//
// THE GRIDS ARE A BUFFER, binding 3, uploaded once from
// frink_quant::iq_tables (frink_vulkan::shaders::iq_table_words), so no
// table is typed a second time here. ggml declares them as device
// arrays for the same reason a const array would be wrong: every
// invocation indexes a different row. A 64-bit grid row is two words,
// low first. The offsets below are held to the Rust ones by
// `shaders::tests::the_glsl_table_offsets_are_the_rust_ones`.
#include "matvec_common.glsl"

layout(std430, binding = 3) readonly buffer IqTables { uint iq_tab[]; };

#define IQ_KSIGNS 0u
#define IQ_GRID_IQ2XXS 128u
#define IQ_GRID_IQ2XS 640u
#define IQ_GRID_IQ2S 1664u
#define IQ_GRID_IQ3XXS 3712u
#define IQ_GRID_IQ3S 3968u
#define IQ_GRID_IQ1S 4480u

// ksigns_iq2xs[i]: the sign byte for a 7-bit code.
uint ksigns(uint i) {
    return iq_tab[IQ_KSIGNS + i];
}

uint u16_at(uint k) {
    return byte_at(k) | (byte_at(k + 1u) << 8);
}

uint u32_at(uint k) {
    return u16_at(k) | (u16_at(k + 2u) << 16);
}

// Eight values `db * magnitude * sign` against X(xo..xo+8): magnitudes
// are the bytes of `lo` then `hi`, and bit j of `signs` negates value j
// (kmask_iq2xs is 1 << j). IQ2's 64-bit row and IQ3's two 32-bit rows
// are the same eight bytes.
float grid8_dot(float db, uint lo, uint hi, uint signs, uint xo) {
    float s = 0.0;
    for (uint j = 0u; j < 8u; ++j) {
        uint g = j < 4u ? lo : hi;
        float mag = float((g >> ((j & 3u) * 8u)) & 0xffu);
        s += ((signs >> j) & 1u) != 0u ? -mag * X(xo + j) : mag * X(xo + j);
    }
    return db * s;
}

// Row `r` of a 64-bit grid at `base`, as grid8_dot.
float grid64_dot(uint base, uint r, float db, uint signs, uint xo) {
    return grid8_dot(db, iq_tab[base + 2u * r], iq_tab[base + 2u * r + 1u], signs, xo);
}

// IQ1's row: eight signed bytes, each `dl * (v + delta)`.
float iq1_row_dot(uint r, float dl, float delta, uint xo) {
    uint lo = iq_tab[IQ_GRID_IQ1S + 2u * r], hi = iq_tab[IQ_GRID_IQ1S + 2u * r + 1u];
    float s = 0.0;
    for (uint j = 0u; j < 8u; ++j) {
        uint g = j < 4u ? lo : hi;
        int v = int((g >> ((j & 3u) * 8u)) & 0xffu);
        v = v >= 128 ? v - 256 : v;
        s += (float(v) + delta) * X(xo + j);
    }
    return dl * s;
}
