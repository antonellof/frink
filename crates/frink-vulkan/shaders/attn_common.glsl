// The decode attention interface shared by attn_scores and attn_values
// (frink_core::attention::causal_gqa_attention_row without window,
// sinks, softcap or ALiBi). The f16 KV cache is one buffer: K at word
// `k_off`, V at word `v_off`, each [ctx, n_kv_heads, head_dim] in
// halves, two per word, so head_dim is even. Query head h reads KV head
// h / (n_heads / n_kv_heads). One workgroup per query head.
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Kv { uint kv[]; };
layout(push_constant) uniform Push {
    // n_heads in the low half, n_kv_heads in the high half: the block
    // is eight words and the window needed one.
    uint heads;
    uint window;
    uint head_dim;
    uint seq_len;
    uint k_off;
    uint v_off;
    uint scale_bits;
    uint seq_stride;
} p;

#define N_HEADS (p.heads & 0xffffu)
#define N_KV_HEADS (p.heads >> 16)

// THE GRID'S z IS THE BATCH POSITION (a prefill; a decode token is
// z = 0). Position z's query is the z-th [n_heads, head_dim] of q, it
// attends over seq_len + z cached positions (its own K and V are
// already stored), and its scores are rows of `seq_stride`.
uint attn_seq() {
    return p.seq_len + gl_WorkGroupID.z;
}
// The first position a query attends over: a sliding window of
// `window` positions that includes its own, or everything when the
// window is 0 (frink_core::attention::causal_gqa_attention_row).
uint attn_start() {
    uint seq = attn_seq();
    return (p.window != 0u && seq > p.window) ? seq - p.window : 0u;
}
uint attn_row(uint h) {
    return gl_WorkGroupID.z * N_HEADS + h;
}
shared float red[64];

uint attn_head() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

// The word index of channel d of KV head `kvh` at position t, from the
// region starting at `off`.
uint kv_word(uint off, uint t, uint kvh, uint d) {
    return off + ((t * N_KV_HEADS + kvh) * p.head_dim + d) / 2u;
}

float kv_at(uint off, uint t, uint kvh, uint d) {
    vec2 two = unpackHalf2x16(kv[kv_word(off, t, kvh, d)]);
    return (d & 1u) == 0u ? two.x : two.y;
}
