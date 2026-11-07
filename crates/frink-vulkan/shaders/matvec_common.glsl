// The matvec interface every frink-vulkan shader keeps
// (crates/frink-vulkan/src/dispatch.rs, `matvec`): weights as uint words
// at binding 0, activations at 1, outputs at 2; push constants rows,
// n_blocks_per_row, row_bytes; one invocation per row.
//
// Weights are read as BYTES out of uint words, because ggml block
// strides (18, 144, 176, 210) are not word multiples and 8-bit storage
// is an optional capability (docs/plans/vulkan-beachhead-verdict.md, §3).

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer Weights { uint w[]; };
layout(std430, binding = 1) readonly buffer Acts { float x[]; };
layout(std430, binding = 2) writeonly buffer Out { float y[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint n_blocks_per_row;
    uint row_bytes;
} p;

uint byte_at(uint k) {
    return (w[k >> 2] >> ((k & 3u) * 8u)) & 0xffu;
}

// An IEEE half at byte offset k, little-endian.
float f16_at(uint k) {
    return unpackHalf2x16(byte_at(k) | (byte_at(k + 1u) << 8)).x;
}

// A byte reinterpreted as a signed int8.
int i8_at(uint k) {
    int v = int(byte_at(k));
    return v >= 128 ? v - 256 : v;
}

// ggml's get_scale_min_k4 (frink_quant::q4_k_scale_min) over the 12
// packed scale bytes starting at byte offset s.
void scale_min_k4(uint j, uint s, out float sc, out float m) {
    if (j < 4u) {
        sc = float(byte_at(s + j) & 63u);
        m = float(byte_at(s + j + 4u) & 63u);
    } else {
        sc = float((byte_at(s + j + 4u) & 0x0fu) | ((byte_at(s + j - 4u) >> 6) << 4));
        m = float((byte_at(s + j + 4u) >> 4) | ((byte_at(s + j) >> 6) << 4));
    }
}
