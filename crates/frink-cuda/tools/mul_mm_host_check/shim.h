#pragma once
// Enough of the CUDA execution model to run an UNMODIFIED `__global__`
// body on a host CPU: one std::thread per CUDA thread, a counting
// barrier standing in for `__syncthreads()`, and one block at a time so
// that `__shared__` -> function-`static` is a faithful stand-in.
//
// This exists because frink's CUDA kernels are written in an
// environment with no NVIDIA GPU. It does NOT emulate a GPU: no warp
// scheduling, no memory model, no coalescing, no races. It checks
// exactly one thing -- that the arithmetic and the index math in the
// emitted CUDA C are the arithmetic and index math the Rust scalar twin
// (`frink_cuda::mul_mm_ref`) was tested against.
#include <math.h>
#include <stddef.h>
#include <string.h>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

struct Dim3 {
    unsigned int x, y, z;
};

// CUDA's four-float vector type, which the GEMM's inner loop reads its
// micro-tile operands through. Deliberately NOT `alignas(16)`: the
// kernel reinterpret-casts a `__shared__` row into one of these, and
// this shim maps `__shared__` onto a function-`static` array whose
// alignment nothing guarantees. Over-declaring the alignment would let
// the compiler assume something the host stand-in does not provide,
// which is a different bug from the one this tool is looking for.
//
// Its absence is why this tool checked NOTHING between the `float4`
// inner loop landing and 2026-09-09: every kind failed to COMPILE, the
// `set -e` at the top of run.sh aborted the script, and the run was
// simply never green rather than quietly wrong. A tool that cannot
// compile the thing it checks is worse than no tool.
struct float4 {
    float x, y, z, w;
};
thread_local Dim3 blockIdx;
thread_local Dim3 threadIdx;
Dim3 blockDim, gridDim;

struct Barrier {
    unsigned n, count = 0, gen = 0;
    std::mutex m;
    std::condition_variable cv;
    explicit Barrier(unsigned n) : n(n) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        unsigned g = gen;
        if (++count == n) {
            count = 0;
            gen++;
            cv.notify_all();
        } else {
            cv.wait(lk, [&] { return gen != g; });
        }
    }
};
static Barrier* g_bar = nullptr;

static inline void __syncthreads(void) { g_bar->wait(); }
static inline float __int_as_float(int i) {
    float f;
    memcpy(&f, &i, 4);
    return f;
}

// A `__constant__` array is device-global read-only storage; on the
// host a plain namespace-scope array is the faithful stand-in. It is
// deliberately NOT `static const`: the codebook kinds are the only
// users, every one of them reads theirs, and `static const` would
// invite `-Wunused-const-variable` on any kind that stopped.
#define __constant__
#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__ static
#define __restrict__

// ---- Warp-level pieces, used by the matvec host check only. ----
//
// CUDA's 128-bit unsigned vector, which the wide coalesced matvec reads
// a 16-byte run of quants through. Not `alignas(16)`, for the same
// reason `float4` above is not.
struct uint4 {
    unsigned int x, y, z, w;
};

// `__shfl_down_sync` over a full warp: every lane publishes its value,
// the warp meets at its own barrier, each lane reads `lane + delta`
// (its own value when that is past the warp, which is CUDA's rule),
// and the warp meets again before the slot can be reused. A barrier
// PER WARP, not per block, because a warp whose row is past the end
// returns before reducing, and a block-wide barrier would wait for it
// forever. `matvec_host_check/main.cpp` sets `g_warp_bar` up.
static Barrier** g_warp_bar = nullptr;
static float g_shfl[1024];
static inline float __shfl_down_sync(unsigned /*mask*/, float v, unsigned delta) {
    const unsigned t = threadIdx.x;
    const unsigned lane = t % 32;
    g_shfl[t] = v;
    g_warp_bar[t / 32]->wait();
    const float r = lane + delta < 32 ? g_shfl[t + delta] : v;
    g_warp_bar[t / 32]->wait();
    return r;
}

// `__shfl_xor_sync` over a full warp, the butterfly the one-row
// attention reduces with (`decode_attn`): each lane reads lane
// `lane ^ mask` of its own warp. Same two-barrier protocol, and the
// same per-warp barriers, as `__shfl_down_sync` above.
static inline float __shfl_xor_sync(unsigned /*mask*/, float v, int lane_mask) {
    const unsigned t = threadIdx.x;
    const unsigned lane = t % 32;
    g_shfl[t] = v;
    g_warp_bar[t / 32]->wait();
    const float r = g_shfl[(t - lane) + (lane ^ (unsigned)lane_mask)];
    g_warp_bar[t / 32]->wait();
    return r;
}

// CUDA's integer `max`, which device code has without a header.
static inline int max(int a, int b) { return a > b ? a : b; }
