// The matvec interface every frink-vulkan shader keeps
// (crates/frink-vulkan/src/runtime.rs, `Runtime::matvec`): weights as
// uint words at binding 0, activations at 1, outputs at 2; push
// constants rows, n_blocks_per_row, row_bytes, x_stride.
//
// A GEMM IS THE SAME SHADER OVER A DEEPER GRID. The grid's z is the
// batch position: position z reads its activations at `z * x_stride`
// (through `X`) and writes its row at `z * rows + row`, the host's
// `[batch][rows]` layout. A matvec dispatches z = 1 and leaves x_stride
// zero, so both offsets are zero and nothing about it changes.
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
// (bc5074aa:docs/plans/archive/vulkan-beachhead-verdict.md, §3).

#define WG 64u
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer Weights { uint w[]; };
layout(std430, binding = 1) readonly buffer Acts { float x[]; };
layout(std430, binding = 2) writeonly buffer Out { float y[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint n_blocks_per_row;
    uint row_bytes;
    uint x_stride;
    uint n_cols;
} p;

shared float partial[WG];

#ifdef GATHER
// THE EMBEDDING GATHER IS THE SAME SHADER WITH A ONE-HOT ACTIVATION
// (build.sh compiles every matvec a second time with -DGATHER). One
// workgroup per output column c: row `x_stride` of the table against
// e_c is that row's value at c, exactly, because every other term is a
// product with 0.0. So each kind's decode is written once, and a gather
// cannot disagree with its matvec. `rows` is the table's row count and
// `n_cols` the row's width.
uint gather_col;
#define X(i) (((i) == gather_col) ? 1.0 : 0.0)

uint matvec_row() {
    gather_col = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    return p.x_stride;
}

void matvec_write(uint row, float v) {
    if (gather_col < p.n_cols) y[gather_col] = v;
}
#else
// Activation `i` of this workgroup's batch position.
uint x_base;
#define X(i) x[x_base + (i)]

uint matvec_row() {
    x_base = gl_WorkGroupID.z * p.x_stride;
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

void matvec_write(uint row, float v) {
    y[gl_WorkGroupID.z * p.rows + row] = v;
}
#endif

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
    if (tid == 0u) matvec_write(row, partial[0]);
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
