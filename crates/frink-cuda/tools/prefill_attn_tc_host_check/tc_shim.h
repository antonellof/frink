#pragma once
// The tensor-core pieces of the CUDA execution model, on top of
// ../mul_mm_host_check/shim.h, enough to run the prefill attention body
// (`frink_cuda::prefill_attn_tc`) on a host CPU. The body is the real
// kernel text. What this file replaces is exactly
// `frink_cuda::mul_mm_tc::TC_PTX_SRC`, the four inline-PTX helpers,
// with their definitions from the PTX ISA.
//
// * `frink_pack2h` is two `cvt.rn.f16.f32`, the low half first.
// * `frink_ldsm_x4`: lanes 8i .. 8i + 7 address the eight 16-byte rows
//   of matrix i, and lane l receives, of every matrix, row l / 4 at
//   columns 2 (l % 4) and 2 (l % 4) + 1.
// * `frink_ldsm_x4_trans` reads the same rows, and lane l receives
//   column l / 4 at rows 2 (l % 4) and 2 (l % 4) + 1.
// * `frink_mma16816` is D = C + A B for a 16x16 A (row) and a 16x8 B
//   (col) of f16, with f32 C and D, every lane's registers placed where
//   the ISA's `m16n8k16` fragment figures place them.
//
// Each warp-wide helper publishes its lane's operand, meets the warp
// at its barrier, reads its result from the published values and
// meets the warp again before a slot can be reused, as
// `__shfl_xor_sync` does in shim.h. A warp whose lanes do not all reach
// the same helper deadlocks here. On a GPU that control flow around
// `mma.sync` or `ldmatrix` is undefined.
//
// The products of two f16 are exact in f32 and the sixteen adds of one
// MMA run in k order, each rounded, so with FP contraction off the body
// performs the same f32 operations as `prefill_attn_tc_twin`. A tensor
// core adds in its own order, which is the hardware test's business.
//
// The stand-ins are checked before they are trusted. run.sh first runs
// the tensor-core GEMM body (`mul_mm_tc`), which passed against its
// reference on real sm_86 and sm_89 hardware, through them, and it must
// equal the exact f16 product summed in k order bit for bit. A wrong
// fragment placement in `frink_ldsm_x4` or `frink_mma16816` cannot pass
// that. `frink_ldsm_x4_trans` has no hardware-proven user yet. It is
// the ISA's figure for `.trans`, the transpose of the plain form above.
#include <stdint.h>
#include <string.h>

#include "shim.h"

#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))

// The GEMM body's two other builtins.
static inline unsigned int __ldg(const unsigned int* p) { return *p; }
static inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
    uint4 r = {x, y, z, w};
    return r;
}

struct float2 {
    float x, y;
};
struct uint2 {
    unsigned int x, y;
};
static inline float2 make_float2(float x, float y) {
    float2 r = {x, y};
    return r;
}
static inline float4 make_float4(float x, float y, float z, float w) {
    float4 r = {x, y, z, w};
    return r;
}
static inline int min(int a, int b) { return a < b ? a : b; }

// The dynamic shared memory of the one block running at a time.
alignas(16) static unsigned short g_tc_dyn_smem[64 * 1024];

static inline unsigned short fx_f2h(float x) {
    _Float16 h = (_Float16)x;
    unsigned short b;
    memcpy(&b, &h, 2);
    return b;
}
static inline float fx_h2f(unsigned int b) {
    const unsigned short s = (unsigned short)(b & 0xffffu);
    _Float16 h;
    memcpy(&h, &s, 2);
    return (float)h;
}

static inline unsigned int frink_pack2h(float lo, float hi) {
    return (unsigned int)fx_f2h(lo) | ((unsigned int)fx_f2h(hi) << 16);
}

static const unsigned short* g_ldsm_row[1024];
static unsigned int g_mma_a[1024][4];
static unsigned int g_mma_b[1024][2];

static inline void frink_ldsm_x4(unsigned int* r, const unsigned short* p) {
    const unsigned t = threadIdx.x, lane = t % 32, base = t - lane;
    g_ldsm_row[t] = p;
    g_warp_bar[t / 32]->wait();
    for (unsigned i = 0; i < 4; i++) {
        const unsigned short* row = g_ldsm_row[base + 8 * i + lane / 4];
        r[i] = (unsigned int)row[2 * (lane % 4)] | ((unsigned int)row[2 * (lane % 4) + 1] << 16);
    }
    g_warp_bar[t / 32]->wait();
}

static inline void frink_ldsm_x4_trans(unsigned int* r, const unsigned short* p) {
    const unsigned t = threadIdx.x, lane = t % 32, base = t - lane;
    g_ldsm_row[t] = p;
    g_warp_bar[t / 32]->wait();
    for (unsigned i = 0; i < 4; i++) {
        const unsigned short* r0 = g_ldsm_row[base + 8 * i + 2 * (lane % 4)];
        const unsigned short* r1 = g_ldsm_row[base + 8 * i + 2 * (lane % 4) + 1];
        r[i] = (unsigned int)r0[lane / 4] | ((unsigned int)r1[lane / 4] << 16);
    }
    g_warp_bar[t / 32]->wait();
}

// A[row][k] of the warp's published A registers: lane (row % 8) * 4 +
// (k % 8) / 2, register row / 8 + 2 * (k / 8), half k % 2.
static inline float fx_mma_a(unsigned base, unsigned row, unsigned k) {
    const unsigned src = base + (row % 8) * 4 + (k % 8) / 2;
    return fx_h2f(g_mma_a[src][row / 8 + 2 * (k / 8)] >> (16 * (k % 2)));
}
// B[k][col]: lane col * 4 + (k % 8) / 2, register k / 8, half k % 2.
static inline float fx_mma_b(unsigned base, unsigned k, unsigned col) {
    const unsigned src = base + col * 4 + (k % 8) / 2;
    return fx_h2f(g_mma_b[src][k / 8] >> (16 * (k % 2)));
}

static inline void frink_mma16816(float* c, const unsigned int* a, const unsigned int* b) {
    const unsigned t = threadIdx.x, lane = t % 32, base = t - lane;
    for (int i = 0; i < 4; i++) g_mma_a[t][i] = a[i];
    g_mma_b[t][0] = b[0];
    g_mma_b[t][1] = b[1];
    g_warp_bar[t / 32]->wait();
    // C: c0 and c1 at row g, columns 2 tg and 2 tg + 1, then c2 and c3
    // at row g + 8.
    const unsigned g = lane / 4, tg = lane % 4;
    for (unsigned e = 0; e < 4; e++) {
        const unsigned row = g + (e >= 2 ? 8 : 0);
        const unsigned col = 2 * tg + (e & 1);
        float acc = c[e];
        for (unsigned k = 0; k < 16; k++) acc += fx_mma_a(base, row, k) * fx_mma_b(base, k, col);
        c[e] = acc;
    }
    g_warp_bar[t / 32]->wait();
}
