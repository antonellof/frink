// The base-3 trit formats, TQ1_0 and PTQ1_0 (frink_quant::ternary): a
// block is QS_LEN bytes of five trits, QH_LEN bytes of four, then an
// f16 scale. The including shader defines QS_LEN and QH_LEN, and that
// is the whole difference between the two.
//
// `qs` is consumed in stages of 32, 16 and 8 bytes, as many whole
// stages of each width as fit, largest first. Within a stage of c bytes
// starting at element e, trit n of byte m is element e + n * c + m; a
// `qh` byte's four trits are elements e + n * QH_LEN + m. Trit n of a
// byte q is ((q * 3^n mod 256) * 3) >> 8, and its value is that minus 1
// (frink_quant::ternary::unpack_trits).
//
// ONE UNIT PER BYTE: a byte's trits are spread across its stage, so a
// unit gathers five (or four) activations at stride c.
#include "matvec_common.glsl"

#define TRIT_UNITS (QS_LEN + QH_LEN)
#define TRIT_BLOCK_BYTES (QS_LEN + QH_LEN + 2u)
#define TRIT_BLOCK_ELEMS (QS_LEN * 5u + QH_LEN * 4u)

// Byte k of the block at byte offset blk, against the block's
// activations starting at xb: the sum of trit * x over its elements.
float trit_unit(uint blk, uint xb, uint k) {
    uint c = QH_LEN, m = k - QS_LEN, e = QS_LEN * 5u, n_trits = 4u;
    if (k < QS_LEN) {
        n_trits = 5u;
        uint j = 0u;
        e = 0u;
        bool found = false;
        for (uint w = 32u; w >= 8u && !found; w >>= 1) {
            while (!found && j + w <= QS_LEN) {
                if (k < j + w) {
                    c = w;
                    m = k - j;
                    found = true;
                } else {
                    j += w;
                    e += 5u * w;
                }
            }
        }
    }
    uint q = byte_at(blk + k);
    float s = 0.0;
    uint pw = 1u;
    for (uint n = 0u; n < n_trits; ++n) {
        uint t = (((q * pw) & 0xffu) * 3u) >> 8;
        s += float(int(t) - 1) * X(xb + e + n * c + m);
        pw *= 3u;
    }
    return s;
}

void main() {
    uint row = matvec_row();
    if (row >= p.rows) return;
    uint base = row * p.row_bytes;
    float acc = 0.0;
    for (uint u = gl_LocalInvocationID.x; u < p.n_blocks_per_row * TRIT_UNITS; u += WG) {
        uint b = u / TRIT_UNITS, k = u % TRIT_UNITS;
        uint blk = base + b * TRIT_BLOCK_BYTES;
        acc += f16_at(blk + TRIT_UNITS) * trit_unit(blk, b * TRIT_BLOCK_ELEMS, k);
    }
    finish(row, acc);
}
