// The decode attention interface shared by attn_scores and attn_values
// (frink_core::attention::causal_gqa_attention_row without window,
// sinks, softcap or ALiBi). The f16 KV cache is one buffer: K at word
// `k_off`, V at word `v_off`, each [ctx, n_kv_heads, head_dim] in
// halves, two per word, so head_dim is even. Query head h reads KV head
// h / (n_heads / n_kv_heads). One workgroup per query head.
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Kv { uint kv[]; };
layout(push_constant) uniform Push {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint seq_len;
    uint k_off;
    uint v_off;
    uint scale_bits;
} p;
shared float red[64];

uint attn_head() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

// The word index of channel d of KV head `kvh` at position t, from the
// region starting at `off`.
uint kv_word(uint off, uint t, uint kvh, uint d) {
    return off + ((t * p.n_kv_heads + kvh) * p.head_dim + d) / 2u;
}

float kv_at(uint off, uint t, uint kvh, uint d) {
    vec2 two = unpackHalf2x16(kv[kv_word(off, t, kvh, d)]);
    return (d & 1u) == 0u ? two.x : two.y;
}
