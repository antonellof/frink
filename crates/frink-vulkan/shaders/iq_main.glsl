// The matvec over 16-value sub-blocks; see iq_common.glsl.
void main() {
    uint row = matvec_row();
    if (row >= p.rows) return;
    uint base = row * p.row_bytes;
    float acc = 0.0;
    for (uint u = gl_LocalInvocationID.x; u < p.n_blocks_per_row * 16u; u += WG) {
        uint b = u >> 4, il = u & 15u;
        acc += sub_dot(base + b * IQ_BLOCK_BYTES, il, b * 256u + il * 16u);
    }
    finish(row, acc);
}
