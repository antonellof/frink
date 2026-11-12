// ggml's kvalues_iq4nl (frink_quant::KVALUES_IQ4NL), shared by IQ4_NL
// and IQ4_XS: -127 -104 -83 -65 -49 -35 -22 -10 1 13 25 38 53 69 89
// 113, as int8 packed four to a word, low byte first.
//
// Packed, not a `const float[16]`: with the array, lavapipe (Mesa
// 25.2.8, x86_64 and aarch64) segfaults inside the IQ4_NL shader once
// the activation read carries the batch offset (matvec_common.glsl's
// `X`), even at offset zero. The same array with `x[i]` ran, and so
// does MXFP4's array with the offset; the cause inside Mesa was not
// found. MoltenVK runs both forms.
float kvalue_iq4nl(uint i) {
    uint w = i < 8u ? (i < 4u ? 0xbfad9881u : 0xf6eaddcfu)
                    : (i < 12u ? 0x26190d01u : 0x71594535u);
    int v = int((w >> ((i & 3u) * 8u)) & 0xffu);
    return float(v >= 128 ? v - 256 : v);
}
