// The matvec interface every frink-vulkan shader keeps
// (crates/frink-vulkan/src/runtime.rs, `Runtime::matvec`): weights as
// uint words at binding 0, activations at 1, outputs at 2; push
// constants rows, n_blocks_per_row, row_bytes.
//
// ONE WORKGROUP PER ROW, the shape of llama.cpp's mul_mat_vec: the row
// is cut into units (a block, or a sub-block with its own scale), the 64
// invocations take units tid, tid+64, ..., and `finish` sums their
// partials in shared memory. A row walked by one invocation, as the
// first shaders did, reads its weights at about a tenth of the memory
// bandwidth. Rows beyond 65535 (an lm_head) arrive as a 2-D grid, so
// the row index folds y in.
//
// Weights are read as BYTES out of uint words, because ggml block
// strides (18, 34, 144, 176, 210) are not word multiples and 8-bit
// storage is an optional capability
// (docs/plans/archive/vulkan-beachhead-verdict.md, §3).

#define WG 64u
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer Weights { uint w[]; };
layout(std430, binding = 1) readonly buffer Acts { float x[]; };
layout(std430, binding = 2) writeonly buffer Out { float y[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint n_blocks_per_row;
    uint row_bytes;
} p;

shared float partial[WG];

uint matvec_row() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

// Reduce every invocation's `acc` and write the row. Called by all 64
// invocations, in uniform control flow (a whole workgroup shares its
// row, so the `row >= rows` early return is uniform too).
void finish(uint row, float acc) {
    uint tid = gl_LocalInvocationID.x;
    partial[tid] = acc;
    barrier();
    for (uint s = WG / 2u; s > 0u; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        barrier();
    }
    if (tid == 0u) y[row] = partial[0];
}

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
