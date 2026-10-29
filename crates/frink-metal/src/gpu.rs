//! Real Metal compute dispatch for GGML matvec kernels, using
//! `objc2-metal`'s bindings to the system Metal framework (no separate
//! CUDA-toolkit-style SDK needed -- the framework ships with macOS).
//!
//! Five matvec kernels are implemented: `Q8_0`/`Q4_0`/`Q4_K`/`Q5_K`/
//! `Q6_K`, plus simdgroup-matrix `mul_mm` kernels for prefill
//! (`batch >= 4`) that reuse each weight-block load across the batch.
//!
//! **Verified directly on real hardware** (unlike `frink-cuda`, which
//! needed a rented GPU): this crate is developed on an Apple M2 Pro, so
//! all five `launch_q*_matvec_matches_cpu_reference` tests below have
//! actually been run against the real GPU, not just compiled -- see
//! each test's `#[ignore]` note for why they're still ignored by
//! default (CI/other contributors' machines may not have a
//! Metal-capable GPU at all, same reasoning as the CUDA hardware
//! tests). All five passed cleanly on the first real-hardware run, no
//! bug-fixing needed (unlike the CUDA-side K-quant history recorded in
//! `docs/MODELS.md`) -- `launch_q6_k_matvec_matches_cpu_reference`
//! specifically hit the same real degenerate case the CUDA test did
//! (pseudo-random block bytes decoding to a NaN `half` scale on one
//! row) and the GPU/CPU outputs agreed (both NaN), confirming
//! `assert_close_relative`'s NaN-vs-NaN handling is doing real work
//! here too, not dead code copied over unused.
//!
//! **Persistent device/pipeline/weight cache**: `shared_metal`/
//! `ensure_pipeline` below reuse one process-wide `MTLDevice` +
//! `MTLCommandQueue`, and cache one compiled `MTLComputePipelineState`
//! per kernel function name, instead of recreating them on every call
//! -- the same per-call overhead problem
//! `frink-cuda::gpu::shared_device`/`ensure_module_loaded` fixed for
//! CUDA. Quantized weight buffers are also cached by host pointer+length
//! (`resident_weight_buffer`) so decode does not re-upload multi-GB
//! matrices every token. f32 norm buffers are also cached by host
//! pointer+length via `resident_f32_buffer`. Q4_K/Q6_K kernels are
//! multi-row (4 rows per threadgroup) and Q5_K is NSG=2 / N_R0=1
//! (2 rows per TG — ggml keeps N_R0_Q5_K at 1 to avoid register spill);
//! Q8_0 is NSG=4 / N_R0=2 (2 rows / 128 threads); Q4_0 is NSG=2 /
//! N_R0=4 (8 rows / 64 threads) matching ggml `mul_mv_q4_0_f32`. This
//! needs an explicit `unsafe impl Send + Sync` for
//! `SharedMetal`/`CachedPipeline`/`ResidentWeightBuffer`/`ResidentF32Buffer`
//! because
//! `objc2-metal`'s `Retained<ProtocolObject<dyn T>>` wrapper is
//! unconditionally `!Send`/`!Sync` (it holds a `NonNull` pointer, and
//! `NonNull` is `!Send`/`!Sync` regardless of what it points to, forcing
//! any wrapper to opt back in explicitly) -- see the safety comment on
//! those impls for why sharing these specific object kinds across
//! threads is sound. `MTLCommandBuffer`/`MTLComputeCommandEncoder` are
//! *not* included in the cache and are still created fresh per call,
//! because (unlike the device/queue/library/pipeline/weight buffers)
//! Apple documents those two types as requiring single-threaded,
//! single-use access -- exactly matching what `launch_matvec` already
//! does (a fresh command buffer/encoder every call, never stored).

use crate::dispatch::dispatch_counted;
use crate::moe_ids::IdsBinding;
use crate::resident_cache::{get_or_build, HostKey, Resident};
use crate::timing::mm_timing_add;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLDispatchType, MTLLibrary, MTLResource, MTLResourceOptions, MTLSize,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("no Metal device available on this machine")]
    NoDevice,
    #[error("Metal kernel failed to compile: {0}")]
    CompileFailed(String),
    #[error("Metal function `{0}` not found in compiled library")]
    FunctionNotFound(&'static str),
    #[error("Metal compute pipeline creation failed: {0}")]
    PipelineFailed(String),
    #[error("Metal buffer allocation failed")]
    BufferAllocFailed,
    #[error("Metal command buffer/encoder creation failed")]
    CommandFailed,
}

/// Concurrent compute encoder (llama.cpp `MTLDispatchTypeConcurrent`).
/// Independent dispatches (e.g. MoE gate∥up, Q∥K∥V) can overlap; callers
/// must insert [`memory_barrier_buffers`] between RAW/WAR/WAW hazards.
pub(crate) fn compute_encoder_concurrent(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, MetalError> {
    cmd_buf
        .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
        .ok_or(MetalError::CommandFailed)
}

/// Buffer-scope barrier — same as llama `ggml_metal_encoder_memory_barrier`.
/// Prefer [`memory_barrier_resources`] when the conflicting set is known:
/// scope-Buffers waits for *all* buffer traffic (including huge weight
/// reads), which bubbles the GPU on OLMoE Concurrent encode.
#[inline]
pub(crate) fn memory_barrier_buffers(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
    crate::dispatch::note_barrier();
    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
}

/// Resource-scoped Concurrent barrier over an ALREADY-BUILT resource list.
///
/// Only the listed buffers are ordered, so subsequent dispatches that
/// don't touch them can overlap with in-flight work on other buffers
/// (e.g. weight reads from a prior matvec).
///
/// This takes the list rather than building one because the hazard
/// tracker already holds the pending set in exactly this form and reuses
/// its allocation across the ~160 barriers a decode token emits;
/// [`memory_barrier_resources`] is the convenience wrapper for the
/// handful of call sites that name their buffers inline.
#[inline]
pub(crate) fn memory_barrier_resource_list(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    resources: &mut [NonNull<ProtocolObject<dyn MTLResource>>],
) {
    if resources.is_empty() {
        // Nothing named: fall back to scope-Buffers, which is counted by
        // the delegate.
        memory_barrier_buffers(encoder);
        return;
    }
    let head = NonNull::new(resources.as_mut_ptr()).expect("non-empty slice has a non-null base");
    crate::dispatch::note_barrier();
    // SAFETY: `head` points at `resources.len()` live resource pointers,
    // each taken from a buffer bound into this encoder's command buffer
    // and therefore alive for the whole encode pass.
    unsafe {
        encoder.memoryBarrierWithResources_count(head, resources.len());
    }
}

/// Resource-scoped Concurrent barrier over buffers named inline.
#[inline]
pub(crate) fn memory_barrier_resources(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    bufs: &[&ProtocolObject<dyn MTLBuffer>],
) {
    // MTLBuffer: MTLResource — build a contiguous pointer list for the API.
    let mut resources: Vec<NonNull<ProtocolObject<dyn MTLResource>>> =
        Vec::with_capacity(bufs.len());
    for b in bufs {
        let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(*b);
        resources.push(NonNull::from(r));
    }
    memory_barrier_resource_list(encoder, &mut resources);
}

/// The resident-activation hand-off from the dense stack to the next
/// matvec used to live here, as a thread-local raw pointer into the
/// process-wide decode scratch, matched on LENGTH alone. It is now
/// [`crate::resident_act`], which states the identity it matches on and
/// why a length was not one.
pub use crate::resident_act::{clear_resident_activation, resident_activation_reuses};

/// Returns the default Metal device's name, or `None` if this machine
/// has no Metal-capable GPU (real check, not a compile-time guess).
pub fn probe() -> Option<String> {
    let device = MTLCreateSystemDefaultDevice()?;
    Some(device.name().to_string())
}

/// Apple's own answer to "how many bytes should this process keep
/// resident on the GPU": `MTLDevice.recommendedMaxWorkingSetSize`.
///
/// This is the real device query, not a guess derived from installed
/// RAM -- on unified-memory Apple Silicon it is the share of physical
/// memory the driver is willing to let one process hold in GPU-resident
/// allocations, and Metal starts paging (or failing allocations) past
/// it. `None` when there is no Metal device.
///
/// It is a *recommendation* and a snapshot: nothing reserves it, and
/// other processes draw from the same pool. Treat it as a ceiling to
/// plan against, never as a guarantee.
pub fn probe_recommended_working_set_bytes() -> Option<u64> {
    let device = MTLCreateSystemDefaultDevice()?;
    Some(device.recommendedMaxWorkingSetSize())
}

/// ggml-metal `kernel_mul_mv_q8_0_f32` port: `N_R0=2` rows per
/// threadgroup, `NSG=4` simdgroups (128 threads) cooperating on the
/// same two rows, each thread owning `NQ=8` contiguous int8 quants of a
/// block per pass. Same dequant identity as
/// `frink_quant::dot_q8_0_f32_scalar` (34-byte block: 2-byte f16 scale
/// plus 32 int8 values). Replaces the legacy one-TG-per-row scalar kernel,
/// which left most of the memory system idle on Q8_0-heavy models
/// (TinyLlama Q8_0 decode was ~1.5x behind llama.cpp).
///
/// Cross-simdgroup reduction goes through 8 floats of threadgroup
/// memory (2 rows x 4 simdgroups). Host dispatches `ceil(n_rows/2)`
/// threadgroups of 128 threads with 32 bytes of TG memory.
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// module docs).
pub const Q8_0_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q8_0_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* partial [[threadgroup(0)]]
) {
    constexpr short NSG = 4;
    constexpr short nr0 = 2;
    constexpr short NQ = 8;

    const int nb = int(n_blocks_per_row);
    const int first_row = int(tgpig) * nr0;

    device const uchar* row_ptr[nr0];
    for (short row = 0; row < nr0; ++row) {
        row_ptr[row] = weights + (size_t)(first_row + row) * row_bytes;
    }

    // 4 threads per block (NQ=8 quants each), 8 blocks per simdgroup
    // pass, stride NSG*NQ = 32 blocks per threadgroup pass.
    const short ix = short(tiisg) / (32 / NQ); // 0..7: block within pass
    const short il = short(tiisg) % (32 / NQ); // 0..3: quant slice

    const int ib0 = int(sgitg) * NQ + ix;

    float sumf[nr0] = {0.0f, 0.0f};
    float yl[NQ];

    device const float* yb = x + ib0 * 32 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uchar* block = row_ptr[row] + (size_t)ib * 34u;
            device const char* qs = (device const char*)(block + 2) + il * NQ;

            float sumq = 0.0f;
            #pragma clang loop unroll(full)
            for (short i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }

            sumf[row] += sumq * float(*(device const half*)(block));
        }

        yb += NSG * NQ * 32;
    }

    for (short row = 0; row < nr0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0) {
            partial[row * NSG + sgitg] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (short row = 0; row < nr0; ++row) {
            if (first_row + row < int(n_rows)) {
                float total = 0.0f;
                for (short sg = 0; sg < NSG; ++sg) {
                    total += partial[row * NSG + sg];
                }
                out[first_row + row] = total;
            }
        }
    }
}
"#;

/// Runs the Q8_0 matvec kernel: `weights` is `rows` rows of
/// `row_bytes`-byte Q8_0-quantized data (GGML layout), `x` is the
/// dense `[cols]` input vector, returns the `[rows]` dot-product
/// output. `row_bytes` must equal `(cols / 32) * 34`.
pub fn launch_q8_0_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        Q8_0_MATVEC_KERNEL_SRC,
        "q8_0_matvec",
        34,
        32,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// CUDA-kernel-equivalent MSL source for a fused Q4_0 dequant+dot
/// kernel: same one-threadgroup-per-row / threadgroup-reduction
/// structure as `Q8_0_MATVEC_KERNEL_SRC`, but unpacking Q4_0's 18-byte
/// blocks (2-byte `half` scale + 16 bytes of packed 4-bit nibbles, low
/// nibble = element `i`, high nibble = element `i+16`, both biased by
/// -8) to mirror `frink_quant::dot_q4_0_f32_scalar`'s exact math and
/// Dense f32 matvec (every MoE GGUF ships the router `ffn_gate_inp` as
/// F32, so this is on the decode hot path 16-plus times per token).
///
/// ggml-metal `kernel_mul_mv_t_t_impl<float, float>` port: `NR0 = 2`
/// output rows per threadgroup, `NSG` simdgroups splitting the reduction
/// axis between them, per-lane partials folded by `simd_sum` and then
/// across simdgroups through threadgroup memory
/// (`helper_mv_reduce_and_write`). Consecutive lanes read consecutive
/// columns, so every load coalesces.
///
/// It replaces a one-thread-per-row kernel, which for a `64 x 2048`
/// router meant the entire GEMM ran as a *single* 64-thread threadgroup
/// on one GPU core with each lane striding its own 8 KB row: ~1.1-1.8
/// ms/tok on OLMoE (see the MoE decode diagnosis in
/// `docs/plans/llama-cpp-parity-push.md`).
///
/// Host dispatches `ceil(rows/2)` threadgroups of `32 * nsg` threads.
pub const F32_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void f32_matvec(
    device const float* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]
) {
    // ggml N_R0 for f32 x f32. NSG comes from the dispatch (<= 8).
    constexpr uint NR0 = 2u;
    constexpr uint NW = 32u;
    constexpr uint MAX_NSG = 8u;
    threadgroup float part[NR0 * MAX_NSG];

    if (rows == 0u) return;
    const uint r0 = tgpig.x * NR0;
    if (r0 >= rows) return;

    const uint tid = sg * NW + lane;
    const uint nth = nsg * NW;

    // Clamp so the hot loop is branch-free; drop OOB rows at the write.
    uint rr[NR0];
    for (uint r = 0u; r < NR0; ++r) {
        rr[r] = min(r0 + r, rows - 1u);
    }

    float sumf[NR0] = { 0.0f };
    for (uint i = tid; i < cols; i += nth) {
        const float xv = x[i];
        for (uint r = 0u; r < NR0; ++r) {
            sumf[r] += weights[(size_t)rr[r] * cols + i] * xv;
        }
    }

    for (uint r = 0u; r < NR0; ++r) {
        const float s = simd_sum(sumf[r]);
        if (lane == 0u) {
            part[r * MAX_NSG + sg] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane < NR0) {
        float tot = 0.0f;
        for (uint g = 0u; g < nsg; ++g) {
            tot += part[lane * MAX_NSG + g];
        }
        if (r0 + lane < rows) {
            out[r0 + lane] = tot;
        }
    }
}
"#;

/// ggml-metal `kernel_mul_mv_q4_0_f32` / `mul_vec_q_n_f32_impl` port:
/// `N_R0=4` rows per simdgroup, `NSG=2` simdgroups (64 threads → 8 rows
/// per TG). Same half-block `yl` packing and nibble dequant as the MoE
/// `*_id` kernels / `frink_quant::dot_q4_0_f32_scalar`. Replaces the
/// legacy one-TG-per-row scalar kernel — OLMoE Q/K/V/O + embd/lm_head
/// all go through this path.
///
/// Host dispatches `ceil(n_rows/8)` threadgroups of 64 threads.
///
/// Verified: see `launch_q4_0_matvec_matches_cpu_reference`.
/// Q5_0 matrix-vector product.
///
/// Q5_0 has had a Metal simdgroup GEMM (`q5_0_mul_mm_sg`) for some time,
/// so a Q5_0 checkpoint ran its PREFILL on the GPU while every DECODE
/// step fell back to the CPU for want of this kernel -- the half of the
/// run that dominates wall-clock for an interactive session.
///
/// Deliberately written straight rather than in the nibble-packed style
/// of `Q4_0_MATVEC_KERNEL_SRC`. That kernel folds the `/256`, `/16` and
/// `/4096` scaling into the activation so it can mask four nibbles out
/// of one `ushort` at once; Q5_0's fifth bit lives in a separate 32-bit
/// `qh` field indexed differently for the low and high halves, so the
/// same trick needs two more shift chains and is much easier to get
/// subtly wrong. The dequantisation here is `ggml`'s reference form.
///
/// One block per lane, 32 lanes striding the row: simpler than Q4_0's
/// half-block split and correct for any block count, including rows
/// whose block count is not a multiple of the simdgroup width.
pub const Q5_0_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float q5_0_mv_dot(device const uchar* block, thread const float* yl) {
    const float d = float(*(device const half*)block);
    const uint qh = uint(block[2]) | (uint(block[3]) << 8)
        | (uint(block[4]) << 16) | (uint(block[5]) << 24);
    device const uchar* qs = block + 6;
    float acc = 0.0f;
    for (uint j = 0u; j < 16u; ++j) {
        // ggml `dequantize_row_q5_0`: the low half takes bit `j` of qh
        // shifted up into position 4, the high half takes bit `j + 16`
        // shifted down into it.
        const uint xh_0 = ((qh >> j) << 4) & 0x10u;
        const uint xh_1 = (qh >> (j + 12u)) & 0x10u;
        const int x0 = int((uint(qs[j]) & 0x0Fu) | xh_0) - 16;
        const int x1 = int((uint(qs[j]) >> 4) | xh_1) - 16;
        acc += yl[j] * float(x0) + yl[j + 16u] * float(x1);
    }
    return d * acc;
}

kernel void q5_0_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint NR = 4u;
    constexpr uint NSG = 2u;
    const uint first_row = (tgpig * NSG + sg) * NR;
    if (first_row >= n_rows) return;

    float acc[NR] = { 0.0f };
    for (uint b = lane; b < n_blocks_per_row; b += 32u) {
        float yl[32];
        device const float* yb = x + b * 32u;
        for (uint i = 0u; i < 32u; ++i) {
            yl[i] = yb[i];
        }
        for (uint rr = 0u; rr < NR; ++rr) {
            const uint row = first_row + rr;
            if (row >= n_rows) continue;
            device const uchar* block =
                weights + (size_t)row * row_bytes + (size_t)b * 22u;
            acc[rr] += q5_0_mv_dot(block, yl);
        }
    }
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = first_row + rr;
        const float sum = simd_sum(acc[rr]);
        if (lane == 0u && row < n_rows) {
            out[row] = sum;
        }
    }
}
"#;

/// llama.cpp `mul_mv_id` slot-parallel MoE matvec for Q5_0 packed planes.
pub const Q5_0_MOE_MATVEC_ID_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float q5_0_mv_dot(device const uchar* block, thread const float* yl) {
    const float d = float(*(device const half*)block);
    const uint qh = uint(block[2]) | (uint(block[3]) << 8)
        | (uint(block[4]) << 16) | (uint(block[5]) << 24);
    device const uchar* qs = block + 6;
    float acc = 0.0f;
    for (uint j = 0u; j < 16u; ++j) {
        const uint xh_0 = ((qh >> j) << 4) & 0x10u;
        const uint xh_1 = (qh >> (j + 12u)) & 0x10u;
        const int x0 = int((uint(qs[j]) & 0x0Fu) | xh_0) - 16;
        const int x1 = int((uint(qs[j]) >> 4) | xh_1) - 16;
        acc += yl[j] * float(x0) + yl[j + 16u] * float(x1);
    }
    return d * acc;
}

kernel void q5_0_moe_matvec_id(
    device const uchar* w_all [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    constant uint& row_bytes [[buffer(4)]],
    constant uint& n_blocks [[buffer(5)]],
    constant uint& n_rows [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],
    constant uint& expert_stride [[buffer(8)]],
    constant uint& n_tokens [[buffer(9)]],
    constant uint& x_stride [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint NR = 4u;
    constexpr uint NSG = 2u;
    const uint group = tgpig.x;
    const uint slot = tgpig.z;
    const uint first_row = (group * NSG + sg) * NR;
    const uint n_slots = n_tokens * top_k;
    if (slot >= n_slots) return;
    const uint token = slot / top_k;
    const uint eid = uint(ids[slot]);
    device const uchar* w = w_all + (size_t)eid * expert_stride;
    float acc[NR] = { 0.0f };
    device const uchar* ax[NR];
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = min(first_row + rr, n_rows > 0u ? n_rows - 1u : 0u);
        ax[rr] = w + (size_t)row * row_bytes;
    }
    for (uint b = lane; b < n_blocks; b += 32u) {
        float yl[32];
        device const float* yb = x + (size_t)token * x_stride + b * 32u;
        for (uint i = 0u; i < 32u; ++i) {
            yl[i] = yb[i];
        }
        for (uint rr = 0u; rr < NR; ++rr) {
            const uint row = first_row + rr;
            if (row >= n_rows) continue;
            acc[rr] += q5_0_mv_dot(ax[rr] + (size_t)b * 22u, yl);
        }
    }
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = first_row + rr;
        const float sum = simd_sum(acc[rr]);
        if (lane == 0u && row < n_rows) {
            out[(size_t)slot * n_rows + row] = sum;
        }
    }
}

kernel void q5_0_moe_down_id(
    device const uchar* down_all [[buffer(0)]],
    device const float* act [[buffer(1)]],
    device float* expert_out [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    constant uint& row_bytes [[buffer(4)]],
    constant uint& n_blocks [[buffer(5)]],
    constant uint& hidden_rows [[buffer(6)]],
    constant uint& ffn_rows [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    constant uint& expert_stride [[buffer(9)]],
    constant uint& n_tokens [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint NR = 4u;
    constexpr uint NSG = 2u;
    const uint group = tgpig.x;
    const uint slot = tgpig.z;
    const uint first_row = (group * NSG + sg) * NR;
    const uint n_slots = n_tokens * top_k;
    if (slot >= n_slots) return;
    const uint eid = uint(ids[slot]);
    device const uchar* wd = down_all + (size_t)eid * expert_stride;
    device const float* xa = act + (size_t)slot * ffn_rows;
    float acc[NR] = { 0.0f };
    device const uchar* ax[NR];
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = min(first_row + rr, hidden_rows > 0u ? hidden_rows - 1u : 0u);
        ax[rr] = wd + (size_t)row * row_bytes;
    }
    for (uint b = lane; b < n_blocks; b += 32u) {
        float yl[32];
        device const float* yb = xa + b * 32u;
        for (uint i = 0u; i < 32u; ++i) {
            yl[i] = yb[i];
        }
        for (uint rr = 0u; rr < NR; ++rr) {
            const uint row = first_row + rr;
            if (row >= hidden_rows) continue;
            acc[rr] += q5_0_mv_dot(ax[rr] + (size_t)b * 22u, yl);
        }
    }
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = first_row + rr;
        const float sum = simd_sum(acc[rr]);
        if (lane == 0u && row < hidden_rows) {
            expert_out[(size_t)slot * hidden_rows + row] = sum;
        }
    }
}
"#;

pub const Q4_0_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float q4_0_mv_half_dot(
    device const uchar* block,
    float sumy,
    thread const float* yl,
    uint il
) {
    const float d = float(*(device const half*)block);
    device const ushort* qs = (device const ushort*)block + 1u + il / 2u;
    float4 acc = float4(0.0f);
    for (uint i = 0u; i < 8u; i += 2u) {
        const ushort q = qs[i / 2u];
        acc[0] += yl[i] * float(q & 0x000Fu);
        acc[1] += yl[i + 1u] * float(q & 0x0F00u);
        acc[2] += yl[i + 8u] * float(q & 0x00F0u);
        acc[3] += yl[i + 9u] * float(q & 0xF000u);
    }
    return d * (sumy * -8.0f + acc[0] + acc[1] + acc[2] + acc[3]);
}

kernel void q4_0_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint NR = 4u;
    constexpr uint NSG = 2u;
    const uint first_row = (tgpig * NSG + sg) * NR;
    if (first_row >= n_rows) return;

    float acc[NR] = { 0.0f };
    const uint ix = lane / 2u;
    const uint il = (lane % 2u) * 8u;
    device const float* yb = x + ix * 32u + il;
    for (uint b = ix; b < n_blocks_per_row; b += 16u) {
        float yl[16];
        float sumy = 0.0f;
        for (uint i = 0u; i < 8u; i += 2u) {
            sumy += yb[i] + yb[i + 1u] + yb[i + 16u] + yb[i + 17u];
            yl[i] = yb[i];
            yl[i + 1u] = yb[i + 1u] / 256.0f;
            yl[i + 8u] = yb[i + 16u] / 16.0f;
            yl[i + 9u] = yb[i + 17u] / 4096.0f;
        }
        for (uint rr = 0u; rr < NR; ++rr) {
            const uint row = first_row + rr;
            if (row >= n_rows) continue;
            device const uchar* block =
                weights + (size_t)row * row_bytes + (size_t)b * 18u;
            acc[rr] += q4_0_mv_half_dot(block, sumy, yl, il);
        }
        yb += 16u * 32u;
    }
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = first_row + rr;
        const float sum = simd_sum(acc[rr]);
        if (lane == 0u && row < n_rows) {
            out[row] = sum;
        }
    }
}
"#;

/// OLMoE decode specialization: all selected Q4_0 experts share one
/// activation. Gate+up and SiLU are computed for every `(expert,row)` in
/// one dispatch; weighted down projections are reduced in a second dispatch.
/// Metal exposes 31 buffer slots, enough for 8 gate + 8 up tensors plus
/// activation/output/shape arguments (OLMoE top-k is 8).
const Q4_0_MOE_TOPK_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float q4_0_half_dot(
    device const uchar* block,
    float sumy,
    thread const float* yl,
    uint il
) {
    const float d = float(*(device const half*)block);
    device const ushort* qs = (device const ushort*)block + 1u + il / 2u;
    float4 acc = float4(0.0f);
    for (uint i = 0u; i < 8u; i += 2u) {
        const ushort q = qs[i / 2u];
        acc[0] += yl[i] * float(q & 0x000Fu);
        acc[1] += yl[i + 1u] * float(q & 0x0F00u);
        acc[2] += yl[i + 8u] * float(q & 0x00F0u);
        acc[3] += yl[i + 9u] * float(q & 0xF000u);
    }
    return d * (sumy * -8.0f + acc[0] + acc[1] + acc[2] + acc[3]);
}

inline float q4_0_load_y(
    device const float* yb,
    thread float* yl
) {
    float sumy = 0.0f;
    for (uint i = 0u; i < 8u; i += 2u) {
        sumy += yb[i] + yb[i + 1u] + yb[i + 16u] + yb[i + 17u];
        yl[i] = yb[i];
        yl[i + 1u] = yb[i + 1u] / 256.0f;
        yl[i + 8u] = yb[i + 16u] / 16.0f;
        yl[i + 9u] = yb[i + 17u] / 4096.0f;
    }
    return sumy;
}

kernel void q4_0_moe_gate_up(
    device const uchar* wg0 [[buffer(0)]],
    device const uchar* wg1 [[buffer(1)]],
    device const uchar* wg2 [[buffer(2)]],
    device const uchar* wg3 [[buffer(3)]],
    device const uchar* wg4 [[buffer(4)]],
    device const uchar* wg5 [[buffer(5)]],
    device const uchar* wg6 [[buffer(6)]],
    device const uchar* wg7 [[buffer(7)]],
    device const uchar* wu0 [[buffer(8)]],
    device const uchar* wu1 [[buffer(9)]],
    device const uchar* wu2 [[buffer(10)]],
    device const uchar* wu3 [[buffer(11)]],
    device const uchar* wu4 [[buffer(12)]],
    device const uchar* wu5 [[buffer(13)]],
    device const uchar* wu6 [[buffer(14)]],
    device const uchar* wu7 [[buffer(15)]],
    device const float* x [[buffer(16)]],
    device float* act [[buffer(17)]],
    constant uint& row_bytes [[buffer(18)]],
    constant uint& n_blocks [[buffer(19)]],
    constant uint& ffn_rows [[buffer(20)]],
    constant uint& n_experts [[buffer(21)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint NR = 4u;
    constexpr uint NSG = 2u;
    constexpr uint ROWS_PER_TG = NR * NSG;
    const uint row_groups = (ffn_rows + ROWS_PER_TG - 1u) / ROWS_PER_TG;
    const uint expert = tgpig / row_groups;
    const uint group = tgpig - expert * row_groups;
    const uint first_row = (group * NSG + sg) * NR;
    if (expert >= n_experts) return;

    device const uchar* wg = wg0;
    device const uchar* wu = wu0;
    switch (expert) {
        case 1: wg = wg1; wu = wu1; break;
        case 2: wg = wg2; wu = wu2; break;
        case 3: wg = wg3; wu = wu3; break;
        case 4: wg = wg4; wu = wu4; break;
        case 5: wg = wg5; wu = wu5; break;
        case 6: wg = wg6; wu = wu6; break;
        case 7: wg = wg7; wu = wu7; break;
        default: break;
    }
    float ga[NR] = { 0.0f };
    float ua[NR] = { 0.0f };
    const uint ix = lane / 2u;
    const uint il = (lane % 2u) * 8u;
    device const float* yb = x + ix * 32u + il;
    for (uint b = ix; b < n_blocks; b += 16u) {
        float yl[16];
        const float sumy = q4_0_load_y(yb, yl);
        for (uint rr = 0u; rr < NR; ++rr) {
            const uint row = first_row + rr;
            if (row >= ffn_rows) continue;
            device const uchar* gb = wg + (size_t)row * row_bytes + (size_t)b * 18u;
            device const uchar* ub = wu + (size_t)row * row_bytes + (size_t)b * 18u;
            ga[rr] += q4_0_half_dot(gb, sumy, yl, il);
            ua[rr] += q4_0_half_dot(ub, sumy, yl, il);
        }
        yb += 16u * 32u;
    }

    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = first_row + rr;
        const float g = simd_sum(ga[rr]);
        const float u = simd_sum(ua[rr]);
        if (lane == 0u && row < ffn_rows) {
            act[(size_t)expert * ffn_rows + row] =
                (g / (1.0f + exp(-g))) * u;
        }
    }
}

kernel void q4_0_moe_down(
    device const uchar* wd0 [[buffer(0)]],
    device const uchar* wd1 [[buffer(1)]],
    device const uchar* wd2 [[buffer(2)]],
    device const uchar* wd3 [[buffer(3)]],
    device const uchar* wd4 [[buffer(4)]],
    device const uchar* wd5 [[buffer(5)]],
    device const uchar* wd6 [[buffer(6)]],
    device const uchar* wd7 [[buffer(7)]],
    device const float* act [[buffer(8)]],
    device float* expert_out [[buffer(9)]],
    constant uint& row_bytes [[buffer(10)]],
    constant uint& n_blocks [[buffer(11)]],
    constant uint& hidden_rows [[buffer(12)]],
    constant uint& ffn_rows [[buffer(13)]],
    constant uint& n_experts [[buffer(14)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint NR = 4u;
    constexpr uint NSG = 2u;
    constexpr uint ROWS_PER_TG = NR * NSG;
    const uint row_groups = (hidden_rows + ROWS_PER_TG - 1u) / ROWS_PER_TG;
    const uint expert = tgpig / row_groups;
    const uint group = tgpig - expert * row_groups;
    const uint first_row = (group * NSG + sg) * NR;
    if (expert >= n_experts) return;

    device const uchar* wd = wd0;
    switch (expert) {
        case 1: wd = wd1; break;
        case 2: wd = wd2; break;
        case 3: wd = wd3; break;
        case 4: wd = wd4; break;
        case 5: wd = wd5; break;
        case 6: wd = wd6; break;
        case 7: wd = wd7; break;
        default: break;
    }
    device const float* xa = act + (size_t)expert * ffn_rows;
    float acc[NR] = { 0.0f };
    const uint ix = lane / 2u;
    const uint il = (lane % 2u) * 8u;
    device const float* yb = xa + ix * 32u + il;
    for (uint b = ix; b < n_blocks; b += 16u) {
        float yl[16];
        const float sumy = q4_0_load_y(yb, yl);
        for (uint rr = 0u; rr < NR; ++rr) {
            const uint row = first_row + rr;
            if (row >= hidden_rows) continue;
            device const uchar* block =
                wd + (size_t)row * row_bytes + (size_t)b * 18u;
            acc[rr] += q4_0_half_dot(block, sumy, yl, il);
        }
        yb += 16u * 32u;
    }

    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = first_row + rr;
        const float sum = simd_sum(acc[rr]);
        if (lane == 0u && row < hidden_rows) {
            expert_out[(size_t)expert * hidden_rows + row] = sum;
        }
    }
}

/// Weighted sum over top-k expert outs. `n_tokens==1` is decode; prefill
/// uses `n_tokens=T` with `[T,K]` route / `[T,K,H]` expert_out / `[T,H]` out.
kernel void moe_weighted_sum(
    device const float* expert_out [[buffer(0)]],
    device const float* route [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& hidden_rows [[buffer(3)]],
    constant uint& n_experts [[buffer(4)]],
    constant uint& n_tokens [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    const uint n = n_tokens * hidden_rows;
    if (i >= n) return;
    const uint token = i / hidden_rows;
    const uint dim = i - token * hidden_rows;
    float sum = 0.0f;
    const uint base_e = token * n_experts;
    for (uint e = 0u; e < n_experts; ++e) {
        sum += route[base_e + e]
            * expert_out[((size_t)base_e + e) * hidden_rows + dim];
    }
    out[(size_t)token * hidden_rows + dim] = sum;
}

/// Decode: `h += weighted_sum(expert_out, route)` — fuses MoE combine +
/// residual (llama graph `ggml_add` after down). Prefill keeps plain sum.
kernel void moe_weighted_sum_residual(
    device const float* expert_out [[buffer(0)]],
    device const float* route [[buffer(1)]],
    device float* h [[buffer(2)]],
    constant uint& hidden_rows [[buffer(3)]],
    constant uint& n_experts [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= hidden_rows) return;
    float sum = 0.0f;
    for (uint e = 0u; e < n_experts; ++e) {
        sum += route[e] * expert_out[(size_t)e * hidden_rows + i];
    }
    h[i] += sum;
}

/// llama.cpp `mul_mv_id` style: packed Q4_0 plane, `ids[slot]` selects
/// expert. Prefill: `n_tokens>1`, slots = `n_tokens * top_k`, `x` strided.
/// One weight stream per dispatch (gate / up / down) — fused gate+up hurt
/// occupancy vs sequential matvecs (OLMoE Metal gap vs llama).
kernel void q4_0_moe_matvec_id(
    device const uchar* w_all [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    constant uint& row_bytes [[buffer(4)]],
    constant uint& n_blocks [[buffer(5)]],
    constant uint& n_rows [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],
    constant uint& expert_stride [[buffer(8)]],
    constant uint& n_tokens [[buffer(9)]],
    constant uint& x_stride [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    // llama.cpp Q4_0 mul_mv: NR0=4, NSG=2. (NSG=4 tried for dense — regress.)
    constexpr uint NR = 4u;
    constexpr uint NSG = 2u;
    // llama.cpp grid: (row_groups, 1, n_slots) × (32, NSG, 1)
    const uint group = tgpig.x;
    const uint slot = tgpig.z;
    const uint first_row = (group * NSG + sg) * NR;
    const uint n_slots = n_tokens * top_k;
    if (slot >= n_slots) return;
    const uint token = slot / top_k;
    const uint eid = uint(ids[slot]);
    device const uchar* w = w_all + (size_t)eid * expert_stride;
    float acc[NR] = { 0.0f };
    device const uchar* ax[NR];
    for (uint rr = 0u; rr < NR; ++rr) {
        // Match llama mul_vec: always bind a row ptr (clamp) so the hot
        // loop stays branch-free; discard OOB rows only at the write.
        const uint row = min(first_row + rr, n_rows > 0u ? n_rows - 1u : 0u);
        ax[rr] = w + (size_t)row * row_bytes;
    }
    const uint ix = lane / 2u;
    const uint il = (lane % 2u) * 8u;
    device const float* yb = x + (size_t)token * x_stride + ix * 32u + il;
    for (uint b = ix; b < n_blocks; b += 16u) {
        float yl[16];
        const float sumy = q4_0_load_y(yb, yl);
        #pragma clang loop unroll(full)
        for (uint rr = 0u; rr < NR; ++rr) {
            acc[rr] += q4_0_half_dot(ax[rr] + (size_t)b * 18u, sumy, yl, il);
        }
        yb += 16u * 32u;
    }
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = first_row + rr;
        const float sum = simd_sum(acc[rr]);
        if (lane == 0u && row < n_rows) {
            out[(size_t)slot * n_rows + row] = sum;
        }
    }
}

kernel void q4_0_moe_down_id(
    device const uchar* down_all [[buffer(0)]],
    device const float* act [[buffer(1)]],
    device float* expert_out [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    constant uint& row_bytes [[buffer(4)]],
    constant uint& n_blocks [[buffer(5)]],
    constant uint& hidden_rows [[buffer(6)]],
    constant uint& ffn_rows [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    constant uint& expert_stride [[buffer(9)]],
    constant uint& n_tokens [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint NR = 4u;
    constexpr uint NSG = 2u;
    const uint group = tgpig.x;
    const uint slot = tgpig.z;
    const uint first_row = (group * NSG + sg) * NR;
    const uint n_slots = n_tokens * top_k;
    if (slot >= n_slots) return;
    const uint eid = uint(ids[slot]);
    device const uchar* wd = down_all + (size_t)eid * expert_stride;
    device const float* xa = act + (size_t)slot * ffn_rows;
    float acc[NR] = { 0.0f };
    device const uchar* ax[NR];
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = min(first_row + rr, hidden_rows > 0u ? hidden_rows - 1u : 0u);
        ax[rr] = wd + (size_t)row * row_bytes;
    }
    const uint ix = lane / 2u;
    const uint il = (lane % 2u) * 8u;
    device const float* yb = xa + ix * 32u + il;
    for (uint b = ix; b < n_blocks; b += 16u) {
        float yl[16];
        const float sumy = q4_0_load_y(yb, yl);
        #pragma clang loop unroll(full)
        for (uint rr = 0u; rr < NR; ++rr) {
            acc[rr] += q4_0_half_dot(ax[rr] + (size_t)b * 18u, sumy, yl, il);
        }
        yb += 16u * 32u;
    }
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = first_row + rr;
        const float sum = simd_sum(acc[rr]);
        if (lane == 0u && row < hidden_rows) {
            expert_out[(size_t)slot * hidden_rows + row] = sum;
        }
    }
}

/// Fused down × top-k + weighted sum (llama decode: one op writes MoE out).
/// Grid depth = `n_tokens` (not n_slots) — loops experts in-kernel.
kernel void q4_0_moe_down_id_sum(
    device const uchar* down_all [[buffer(0)]],
    device const float* act [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    device const float* route [[buffer(4)]],
    constant uint& row_bytes [[buffer(5)]],
    constant uint& n_blocks [[buffer(6)]],
    constant uint& hidden_rows [[buffer(7)]],
    constant uint& ffn_rows [[buffer(8)]],
    constant uint& top_k [[buffer(9)]],
    constant uint& expert_stride [[buffer(10)]],
    constant uint& n_tokens [[buffer(11)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint NR = 4u;
    constexpr uint NSG = 2u;
    const uint group = tgpig.x;
    const uint token = tgpig.z;
    const uint first_row = (group * NSG + sg) * NR;
    if (token >= n_tokens) return;
    float acc[NR] = { 0.0f };
    for (uint k = 0u; k < top_k; ++k) {
        const uint slot = token * top_k + k;
        const uint eid = uint(ids[slot]);
        const float rw = route[slot];
        device const uchar* wd = down_all + (size_t)eid * expert_stride;
        device const float* xa = act + (size_t)slot * ffn_rows;
        float partial[NR] = { 0.0f };
        device const uchar* ax[NR];
        for (uint rr = 0u; rr < NR; ++rr) {
            const uint row = min(first_row + rr, hidden_rows > 0u ? hidden_rows - 1u : 0u);
            ax[rr] = wd + (size_t)row * row_bytes;
        }
        const uint ix = lane / 2u;
        const uint il = (lane % 2u) * 8u;
        device const float* yb = xa + ix * 32u + il;
        for (uint b = ix; b < n_blocks; b += 16u) {
            float yl[16];
            const float sumy = q4_0_load_y(yb, yl);
            #pragma clang loop unroll(full)
            for (uint rr = 0u; rr < NR; ++rr) {
                partial[rr] += q4_0_half_dot(ax[rr] + (size_t)b * 18u, sumy, yl, il);
            }
            yb += 16u * 32u;
        }
        for (uint rr = 0u; rr < NR; ++rr) {
            acc[rr] += rw * simd_sum(partial[rr]);
        }
    }
    for (uint rr = 0u; rr < NR; ++rr) {
        const uint row = first_row + rr;
        if (lane == 0u && row < hidden_rows) {
            out[(size_t)token * hidden_rows + row] = acc[rr];
        }
    }
}

/// llama `kernel_mul_mm_id_map0`: per-expert token/slot lists from routing
/// ids `[n_tokens, top_k]`. One thread per expert.
template<short ne20>
static inline void moe_mm_id_map0_impl(
    device const int* src2,
    device uint* htpe,
    device int* hids,
    uint n_tokens,
    ushort tpitg,
    ushort ntg,
    threadgroup ushort* shmem
) {
    const short ide = short(tpitg);
    uint32_t n_all = 0;
    device int* ids_i32 = hids + ide * n_tokens;

    for (uint i21 = 0; i21 < n_tokens; i21 += ntg) {
        if (i21 + tpitg < n_tokens) {
            device const int* row = src2 + (i21 + tpitg) * ne20;
            threadgroup ushort* sids = shmem + tpitg * ne20;
#pragma clang loop unroll(full)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = ushort(row[i20]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short t = 0; t < short(ntg); t++) {
            if (i21 + uint(t) >= n_tokens) {
                break;
            }
            threadgroup const ushort* sids = shmem + uint(t) * ne20;
            short sel = 0;
#pragma clang loop unroll(full)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ushort(ide)) * (i20 + 1);
            }
            ids_i32[n_all] = int((i21 + uint(t)) * uint(ne20) + uint(sel - 1));
            n_all += sel > 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    htpe[ide] = n_all;
}

kernel void moe_mm_id_map0_ne20_2(
    device const int* src2 [[buffer(0)]],
    device uint* htpe [[buffer(1)]],
    device int* hids [[buffer(2)]],
    constant uint& n_tokens [[buffer(3)]],
    ushort tpitg [[thread_position_in_threadgroup]],
    ushort ntg [[threads_per_threadgroup]],
    threadgroup ushort* shmem [[threadgroup(0)]]
) { moe_mm_id_map0_impl<2>(src2, htpe, hids, n_tokens, tpitg, ntg, shmem); }

kernel void moe_mm_id_map0_ne20_4(
    device const int* src2 [[buffer(0)]],
    device uint* htpe [[buffer(1)]],
    device int* hids [[buffer(2)]],
    constant uint& n_tokens [[buffer(3)]],
    ushort tpitg [[thread_position_in_threadgroup]],
    ushort ntg [[threads_per_threadgroup]],
    threadgroup ushort* shmem [[threadgroup(0)]]
) { moe_mm_id_map0_impl<4>(src2, htpe, hids, n_tokens, tpitg, ntg, shmem); }

kernel void moe_mm_id_map0_ne20_6(
    device const int* src2 [[buffer(0)]],
    device uint* htpe [[buffer(1)]],
    device int* hids [[buffer(2)]],
    constant uint& n_tokens [[buffer(3)]],
    ushort tpitg [[thread_position_in_threadgroup]],
    ushort ntg [[threads_per_threadgroup]],
    threadgroup ushort* shmem [[threadgroup(0)]]
) { moe_mm_id_map0_impl<6>(src2, htpe, hids, n_tokens, tpitg, ntg, shmem); }

kernel void moe_mm_id_map0_ne20_8(
    device const int* src2 [[buffer(0)]],
    device uint* htpe [[buffer(1)]],
    device int* hids [[buffer(2)]],
    constant uint& n_tokens [[buffer(3)]],
    ushort tpitg [[thread_position_in_threadgroup]],
    ushort ntg [[threads_per_threadgroup]],
    threadgroup ushort* shmem [[threadgroup(0)]]
) { moe_mm_id_map0_impl<8>(src2, htpe, hids, n_tokens, tpitg, ntg, shmem); }

/// Gather rows for `mul_mm_id` host map0: `dst[i*dst_stride+j] =
/// src[ids[i]*src_stride+j]`. One threadgroup per batch row.
kernel void moe_gather_rows(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const int* ids [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& src_stride [[buffer(5)]],
    constant uint& dst_stride [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row >= batch_size) {
        return;
    }
    const uint id = uint(ids[row]);
    device const float* srow = src + (size_t)id * src_stride;
    device float* drow = dst + (size_t)row * dst_stride;
    for (uint j = tid; j < cols; j += tg_size) {
        drow[j] = srow[j];
    }
}

/// Scatter rows after batched GEMM: `dst[ids[i]*dst_stride+j] =
/// src[i*src_stride+j]`.
kernel void moe_scatter_rows(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const int* ids [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& src_stride [[buffer(5)]],
    constant uint& dst_stride [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row >= batch_size) {
        return;
    }
    const uint id = uint(ids[row]);
    device const float* srow = src + (size_t)row * src_stride;
    device float* drow = dst + (size_t)id * dst_stride;
    for (uint j = tid; j < cols; j += tg_size) {
        drow[j] = srow[j];
    }
}

/// Prefill router GEMM with **F32** weights (`ffn_gate_inp` ships F32 in
/// every MoE GGUF we load): `out[t*n_experts + e] = dot(w[e], x[t])`.
/// One simdgroup per (expert, token); `hidden` is the reduction length.
/// Kept in f32 end to end — the top-k selection below is sensitive to
/// near-ties, so this is the one prefill GEMM that does not go through
/// the f16 activation plane.
kernel void moe_router_mm_f32(
    device const float* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& n_experts [[buffer(4)]],
    constant uint& n_tokens [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint e = tgpig.x;
    const uint t = tgpig.y;
    if (e >= n_experts || t >= n_tokens) {
        return;
    }
    device const float* wr = w + (size_t)e * hidden;
    device const float* xr = x + (size_t)t * hidden;
    float acc = 0.0f;
    for (uint i = lane; i < hidden; i += 32u) {
        acc += wr[i] * xr[i];
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        out[(size_t)t * n_experts + e] = acc;
    }
}

/// Softmax over `n` logits (`n<=256`) per token, writing top-`k` ids +
/// probs. `renormalize!=0` divides selected probs by their sum (Mixtral);
/// OLMoE leaves them as global softmax mass (`renormalize==0`).
///
/// One simdgroup per token, probs in
/// threadgroup memory (`n` floats) instead of a 256-float per-thread
/// array. Writes `ids[t*k + j]` / `weights[t*k + j]`.
///
/// Tie policy: lowest expert index wins. Decode calls this with
/// `n_tokens == 1`; the single-lane kernel it replaced cost ~1.3-2.2
/// ms/tok across OLMoE's 16 layers.
kernel void moe_topk_softmax_batch(
    device const float* logits [[buffer(0)]],
    device int* ids [[buffer(1)]],
    device float* weights [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& renormalize [[buffer(5)]],
    constant uint& n_tokens [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* probs [[threadgroup(0)]]
) {
    const uint t = tg;
    if (t >= n_tokens) {
        return;
    }
    device const float* row = logits + (size_t)t * n;

    float lmax = -INFINITY;
    for (uint i = lane; i < n; i += 32u) {
        lmax = max(lmax, row[i]);
    }
    const float mx = simd_max(lmax);

    float lsum = 0.0f;
    for (uint i = lane; i < n; i += 32u) {
        const float p = exp(row[i] - mx);
        probs[i] = p;
        lsum += p;
    }
    const float sum = simd_sum(lsum);
    const float inv = 1.0f / sum;
    for (uint i = lane; i < n; i += 32u) {
        probs[i] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0u; j < k; ++j) {
        float best_p = -1.0f;
        uint best_i = 0u;
        for (uint i = lane; i < n; i += 32u) {
            if (probs[i] > best_p) {
                best_p = probs[i];
                best_i = i;
            }
        }
        const float m = simd_max(best_p);
        const uint cand = (best_p == m) ? best_i : 0xffffffffu;
        const uint sel = simd_min(cand);
        if (lane == 0u) {
            ids[(size_t)t * k + j] = int(sel);
            weights[(size_t)t * k + j] = m;
            probs[sel] = -1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (renormalize != 0u && lane == 0u) {
        float s = 0.0f;
        for (uint j = 0u; j < k; ++j) {
            s += weights[(size_t)t * k + j];
        }
        if (s > 0.0f) {
            const float invs = 1.0f / s;
            for (uint j = 0u; j < k; ++j) {
                weights[(size_t)t * k + j] *= invs;
            }
        }
    }
}
"#;

/// Launches the Q4_0 matvec kernel. Verified on a real Apple M2 Pro GPU
/// -- see `Q4_0_MATVEC_KERNEL_SRC`'s doc comment.
pub fn launch_q4_0_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        Q4_0_MATVEC_KERNEL_SRC,
        "q4_0_matvec",
        18,
        32,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// Q4_0 multi-activation matmul (prefill weight-reuse path).
///
/// Correctness-first: same dequant as [`Q4_0_MATVEC_KERNEL_SRC`]
/// (`scale * (nibble - 8)`), one threadgroup per weight row, threads
/// stride over Q4_0 blocks and accumulate into a small batch tile
/// (`NB=8`) before a simd_sum reduce. Not ggml-metal's simdgroup
/// `mul_mm` tile (that scaffold returned zeros on M2 — kept out until
/// indexing matches). Host flattens `x_batch` as `[batch, cols]`,
/// returns `[batch, rows]`.
pub const Q4_0_MUL_MM_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4_0_mul_mm(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* partial [[threadgroup(0)]]
) {
    if (row >= n_rows) {
        return;
    }
    constexpr short NB = 8;
    const int cols = int(n_blocks_per_row) * 32;
    device const uchar* row_ptr = weights + (size_t)row * row_bytes;

    for (int bt = 0; bt < int(batch_size); bt += NB) {
        float acc[8];
        for (short b = 0; b < NB; ++b) {
            acc[b] = 0.0f;
        }
        for (uint blk = tid; blk < n_blocks_per_row; blk += tg_size) {
            device const uchar* block = row_ptr + blk * 18u;
            const float scale = float(*(device const half*)block);
            const uint base = blk * 32u;
            for (short b = 0; b < NB; ++b) {
                const int batch_idx = bt + b;
                if (batch_idx >= int(batch_size)) {
                    break;
                }
                device const float* xb = x + (size_t)batch_idx * cols + base;
                float block_acc = 0.0f;
                for (uint i = 0; i < 16u; i++) {
                    const uchar byte = block[2 + i];
                    const int lo = (int)(byte & 0x0Fu) - 8;
                    const int hi = (int)((byte >> 4) & 0x0Fu) - 8;
                    block_acc += float(lo) * xb[i];
                    block_acc += float(hi) * xb[i + 16];
                }
                acc[b] += block_acc * scale;
            }
        }
        for (short b = 0; b < NB; ++b) {
            const int batch_idx = bt + b;
            if (batch_idx >= int(batch_size)) {
                break;
            }
            partial[tid] = acc[b];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float s = simd_sum(acc[b]);
            if ((tid & 31u) == 0u) {
                partial[tid / 32u] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float total = 0.0f;
                const uint nsg = (tg_size + 31u) / 32u;
                for (uint i = 0u; i < nsg; i++) {
                    total += partial[i];
                }
                out[(size_t)batch_idx * n_rows + row] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#;

/// Launches Q4_0 multi-activation matmul (see [`Q4_0_MUL_MM_KERNEL_SRC`]).
///
/// Dequant matches [`Q4_0_MATVEC_KERNEL_SRC`]. `x_batch` is `[batch, cols]`;
/// returns `[batch, rows]`.
pub fn launch_q4_0_mul_mm(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    let n_blocks_per_row = row_bytes / 18;
    let cols = n_blocks_per_row * 32;
    assert_eq!(weights.len(), rows * row_bytes);
    assert_eq!(x_batch.len(), batch_size * cols);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let mut x_owned = x_batch.to_vec();
    let x_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(x_owned.as_mut_ptr() as *mut _).unwrap(),
            x_owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;

    let weights_buf = resident_weight_buffer(device, weights)?;
    let out_elems = batch_size * rows;
    let out_scratch = borrow_scratch(device, out_elems * 4, MTLResourceOptions::StorageModeShared)?;
    let out_buf = out_scratch.get();

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let enc = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    encode_q4_0_mul_mm(
        &enc,
        device,
        &weights_buf,
        &x_buf,
        out_buf,
        row_bytes,
        n_blocks_per_row,
        rows,
        batch_size,
    )?;
    enc.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let out_ptr = out_buf.contents();
    let out_slice =
        unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, out_elems) };
    Ok(out_slice.to_vec())
}

/// Encode one `q4_0_mul_mm` dispatch (caller owns encoder barriers).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q4_0_mul_mm(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    weights_buf: &ResidentWeightBuffer,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    row_bytes: usize,
    n_blocks_per_row: usize,
    rows: usize,
    batch_size: usize,
) -> Result<(), MetalError> {
    let pipeline = ensure_pipeline(device, Q4_0_MUL_MM_KERNEL_SRC, "q4_0_mul_mm")?;
    let tg = 64u32;
    unsafe {
        enc.setComputePipelineState(&pipeline.0);
        enc.setBuffer_offset_atIndex(Some(&weights_buf.buffer), weights_buf.weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(x_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        let mut row_bytes_u32 = row_bytes as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut row_bytes_u32 as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut n_blocks_u32 = n_blocks_per_row as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut n_blocks_u32 as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        let mut n_rows_u32 = rows as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut n_rows_u32 as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut batch_u32 = batch_size as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut batch_u32 as *mut u32 as *mut _).unwrap(),
            4,
            6,
        );
        enc.setThreadgroupMemoryLength_atIndex((tg as usize) * 4, 0);
    }

    dispatch_counted(
        enc,
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// ggml-metal `kernel_mul_mv_q4_K_f32` port: `N_R0=2` rows per simdgroup,
/// `NSG=2` simdgroups per TG (4 rows / 64 threads). Register-local `yl`/`yh`
/// activation packs (no shared-`x` tile) with masked nibble dots — same
/// dequant identity as `frink_quant::dot_q4_k_f32_scalar`. Host dispatches
/// `ceil(n_rows/4)` threadgroups of 64 threads.
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// `launch_q4_k_matvec_matches_cpu_reference`).
pub const Q4_K_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;  // 0...3
    const short it = tiisg % 8;  // 0...7
    const short iq = it / 4;     // 0 or 1
    const short ir = it % 4;     // 0...3

    const int first_row = int(tgpig * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);

    float yl[16];
    float yh[16];
    float sumf[2] = {0.0f, 0.0f};

    device const float* y4 = x + ix * 256 + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = float4(0.0f);

        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];
            sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];
            sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128];
            sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160];
            sumy[3] += yh[i + 8];
        }

        device const uchar* block0 =
            weights + (size_t)first_row * row_bytes + (size_t)ib * 144u;
        device const uint16_t* sc =
            (device const uint16_t*)(block0 + 4) + iq;
        device const uint16_t* q1 =
            (device const uint16_t*)(block0 + 16) + 16 * iq + 4 * ir;
        device const half* dh = (device const half*)(block0);

        uint16_t sc16[4];
        thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t* q2 = q1 + 32;

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);

            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * float(q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * float(q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0])
                    * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0])
                        + (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1])
                            * (1.0f / 16.0f)
                        + (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4])
                        + (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5])
                            * (1.0f / 16.0f))
                - float(dh[1])
                    * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3])
                        + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

            q1 += row_bytes / 2;
            sc += row_bytes / 2;
            dh += row_bytes / 2;
        }

        y4 += 4 * 256;
    }

    for (int row = 0; row < nr0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < int(n_rows)) {
            out[first_row + row] = sum_all;
        }
    }
}
"#;

/// llama.cpp `mul_mv_id` slot-parallel MoE matvec for Q4_K packed planes.
pub const Q4_K_MOE_MATVEC_ID_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4_k_moe_matvec_id(
    device const uchar* w_all [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    constant uint& row_bytes [[buffer(4)]],
    constant uint& n_blocks_per_row [[buffer(5)]],
    constant uint& n_rows [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],
    constant uint& expert_stride [[buffer(8)]],
    constant uint& n_tokens [[buffer(9)]],
    constant uint& x_stride [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint slot = tgpig.z;
    const uint n_slots = n_tokens * top_k;
    if (slot >= n_slots) return;
    const uint token = slot / top_k;
    const uint eid = uint(ids[slot]);
    device const uchar* weights = w_all + (size_t)eid * expert_stride;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int first_row = int(tgpig.x * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);

    float yl[16];
    float yh[16];
    float sumf[2] = {0.0f, 0.0f};

    device const float* y4 = x + (size_t)token * x_stride + ix * 256 + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = float4(0.0f);

        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];
            sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];
            sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128];
            sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160];
            sumy[3] += yh[i + 8];
        }

        device const uchar* block0 =
            weights + (size_t)first_row * row_bytes + (size_t)ib * 144u;
        device const uint16_t* sc =
            (device const uint16_t*)(block0 + 4) + iq;
        device const uint16_t* q1 =
            (device const uint16_t*)(block0 + 16) + 16 * iq + 4 * ir;
        device const half* dh = (device const half*)(block0);

        uint16_t sc16[4];
        thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t* q2 = q1 + 32;

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);

            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * float(q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * float(q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0])
                    * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0])
                        + (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1])
                            * (1.0f / 16.0f)
                        + (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4])
                        + (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5])
                            * (1.0f / 16.0f))
                - float(dh[1])
                    * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3])
                        + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

            q1 += row_bytes / 2;
            sc += row_bytes / 2;
            dh += row_bytes / 2;
        }

        y4 += 4 * 256;
    }

    for (int row = 0; row < nr0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < int(n_rows)) {
            out[(size_t)slot * n_rows + first_row + row] = sum_all;
        }
    }
}

kernel void q4_k_moe_down_id(
    device const uchar* down_all [[buffer(0)]],
    device const float* act [[buffer(1)]],
    device float* expert_out [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    constant uint& row_bytes [[buffer(4)]],
    constant uint& n_blocks_per_row [[buffer(5)]],
    constant uint& hidden_rows [[buffer(6)]],
    constant uint& ffn_rows [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    constant uint& expert_stride [[buffer(9)]],
    constant uint& n_tokens [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint slot = tgpig.z;
    const uint n_slots = n_tokens * top_k;
    if (slot >= n_slots) return;
    const uint eid = uint(ids[slot]);
    device const uchar* weights = down_all + (size_t)eid * expert_stride;
    device const float* xa = act + (size_t)slot * ffn_rows;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int first_row = int(tgpig.x * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);

    float yl[16];
    float yh[16];
    float sumf[2] = {0.0f, 0.0f};

    device const float* y4 = xa + ix * 256 + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = float4(0.0f);

        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];
            sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];
            sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128];
            sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160];
            sumy[3] += yh[i + 8];
        }

        device const uchar* block0 =
            weights + (size_t)first_row * row_bytes + (size_t)ib * 144u;
        device const uint16_t* sc =
            (device const uint16_t*)(block0 + 4) + iq;
        device const uint16_t* q1 =
            (device const uint16_t*)(block0 + 16) + 16 * iq + 4 * ir;
        device const half* dh = (device const half*)(block0);

        uint16_t sc16[4];
        thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t* q2 = q1 + 32;

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);

            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * float(q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * float(q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0])
                    * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0])
                        + (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1])
                            * (1.0f / 16.0f)
                        + (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4])
                        + (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5])
                            * (1.0f / 16.0f))
                - float(dh[1])
                    * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3])
                        + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

            q1 += row_bytes / 2;
            sc += row_bytes / 2;
            dh += row_bytes / 2;
        }

        y4 += 4 * 256;
    }

    for (int row = 0; row < nr0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < int(hidden_rows)) {
            expert_out[(size_t)slot * hidden_rows + first_row + row] = sum_all;
        }
    }
}
"#;

/// llama.cpp `mul_mv_id` slot-parallel MoE matvec for Q8_0 packed planes.
pub const Q8_0_MOE_MATVEC_ID_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q8_0_moe_matvec_id(
    device const uchar* w_all [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    constant uint& row_bytes [[buffer(4)]],
    constant uint& n_blocks_per_row [[buffer(5)]],
    constant uint& n_rows [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],
    constant uint& expert_stride [[buffer(8)]],
    constant uint& n_tokens [[buffer(9)]],
    constant uint& x_stride [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* partial [[threadgroup(0)]]
) {
    constexpr short NSG = 4;
    constexpr short nr0 = 2;
    constexpr short NQ = 8;

    const uint slot = tgpig.z;
    const uint n_slots = n_tokens * top_k;
    if (slot >= n_slots) return;
    const uint token = slot / top_k;
    const uint eid = uint(ids[slot]);
    device const uchar* weights = w_all + (size_t)eid * expert_stride;

    const int nb = int(n_blocks_per_row);
    const int first_row = int(tgpig.x) * nr0;

    device const uchar* row_ptr[nr0];
    for (short row = 0; row < nr0; ++row) {
        row_ptr[row] = weights + (size_t)(first_row + row) * row_bytes;
    }

    const short ix = short(tiisg) / (32 / NQ);
    const short il = short(tiisg) % (32 / NQ);

    const int ib0 = int(sgitg) * NQ + ix;

    float sumf[nr0] = {0.0f, 0.0f};
    float yl[NQ];

    device const float* yb = x + (size_t)token * x_stride + ib0 * 32 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uchar* block = row_ptr[row] + (size_t)ib * 34u;
            device const char* qs = (device const char*)(block + 2) + il * NQ;

            float sumq = 0.0f;
            #pragma clang loop unroll(full)
            for (short i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }

            sumf[row] += sumq * float(*(device const half*)(block));
        }

        yb += NSG * NQ * 32;
    }

    for (short row = 0; row < nr0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0) {
            partial[row * NSG + sgitg] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (short row = 0; row < nr0; ++row) {
            if (first_row + row < int(n_rows)) {
                float total = 0.0f;
                for (short sg = 0; sg < NSG; ++sg) {
                    total += partial[row * NSG + sg];
                }
                out[(size_t)slot * n_rows + first_row + row] = total;
            }
        }
    }
}

kernel void q8_0_moe_down_id(
    device const uchar* down_all [[buffer(0)]],
    device const float* act [[buffer(1)]],
    device float* expert_out [[buffer(2)]],
    device const int* ids [[buffer(3)]],
    constant uint& row_bytes [[buffer(4)]],
    constant uint& n_blocks_per_row [[buffer(5)]],
    constant uint& hidden_rows [[buffer(6)]],
    constant uint& ffn_rows [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    constant uint& expert_stride [[buffer(9)]],
    constant uint& n_tokens [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* partial [[threadgroup(0)]]
) {
    constexpr short NSG = 4;
    constexpr short nr0 = 2;
    constexpr short NQ = 8;

    const uint slot = tgpig.z;
    const uint n_slots = n_tokens * top_k;
    if (slot >= n_slots) return;
    const uint eid = uint(ids[slot]);
    device const uchar* weights = down_all + (size_t)eid * expert_stride;
    device const float* xa = act + (size_t)slot * ffn_rows;

    const int nb = int(n_blocks_per_row);
    const int first_row = int(tgpig.x) * nr0;

    device const uchar* row_ptr[nr0];
    for (short row = 0; row < nr0; ++row) {
        row_ptr[row] = weights + (size_t)(first_row + row) * row_bytes;
    }

    const short ix = short(tiisg) / (32 / NQ);
    const short il = short(tiisg) % (32 / NQ);

    const int ib0 = int(sgitg) * NQ + ix;

    float sumf[nr0] = {0.0f, 0.0f};
    float yl[NQ];

    device const float* yb = xa + ib0 * 32 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma clang loop unroll(full)
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uchar* block = row_ptr[row] + (size_t)ib * 34u;
            device const char* qs = (device const char*)(block + 2) + il * NQ;

            float sumq = 0.0f;
            #pragma clang loop unroll(full)
            for (short i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }

            sumf[row] += sumq * float(*(device const half*)(block));
        }

        yb += NSG * NQ * 32;
    }

    for (short row = 0; row < nr0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0) {
            partial[row * NSG + sgitg] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (short row = 0; row < nr0; ++row) {
            if (first_row + row < int(hidden_rows)) {
                float total = 0.0f;
                for (short sg = 0; sg < NSG; ++sg) {
                    total += partial[row * NSG + sg];
                }
                expert_out[(size_t)slot * hidden_rows + first_row + row] = total;
            }
        }
    }
}
"#;

/// Launches the Q4_K matvec kernel. Verified on a real Apple M2 Pro GPU
/// -- see `Q4_K_MATVEC_KERNEL_SRC`'s doc comment.
pub fn launch_q4_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        Q4_K_MATVEC_KERNEL_SRC,
        "q4_k_matvec",
        144,
        256,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// Q4_K multi-activation matmul (prefill weight-reuse path).
///
/// Correctness-first, same shape as [`Q4_0_MUL_MM_KERNEL_SRC`]: one
/// threadgroup per weight row, threads stride over the row's Q4_K blocks
/// and accumulate an `NB`-wide batch tile before a `simd_sum` reduce, so
/// each row's quantized bytes are read once and reused across the batch.
/// The per-block dequant is the exact `frink_quant::dot_q4_k_f32_scalar`
/// identity (6-bit packed scales/mins via `q4_k_scale_min`), so results
/// match [`Q4_K_MATVEC_KERNEL_SRC`]. This is deliberately *not*
/// ggml-metal's `simdgroup_matrix` tile -- that scaffold produced wrong
/// dequant on M2, so it is kept out until its indexing is proven.
///
/// Host flattens `x_batch` as `[batch, cols]`; returns `[batch, rows]`.
pub const Q4_K_MUL_MM_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline uchar2 q4_k_scale_min(uint j, device const uchar* scales) {
    if (j < 4u) {
        return uchar2(scales[j] & 63u, scales[j + 4u] & 63u);
    }
    return uchar2(
        (scales[j + 4u] & 0x0Fu) | ((scales[j - 4u] >> 6u) << 4u),
        (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u));
}

kernel void q4_k_mul_mm(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* partial [[threadgroup(0)]]
) {
    if (row >= n_rows) {
        return;
    }
    constexpr short NB = 8;
    const int cols = int(n_blocks_per_row) * 256;
    device const uchar* row_ptr = weights + (size_t)row * row_bytes;

    for (int bt = 0; bt < int(batch_size); bt += NB) {
        float acc[8];
        for (short b = 0; b < NB; ++b) {
            acc[b] = 0.0f;
        }
        for (uint blk = tid; blk < n_blocks_per_row; blk += tg_size) {
            device const uchar* block = row_ptr + (size_t)blk * 144u;
            const float d = float(*(device const half*)block);
            const float dmin = float(*(device const half*)(block + 2));
            device const uchar* scales = block + 4;
            device const uchar* qs = block + 16;
            const uint base = blk * 256u;
            for (short b = 0; b < NB; ++b) {
                const int batch_idx = bt + b;
                if (batch_idx >= int(batch_size)) {
                    break;
                }
                device const float* xb = x + (size_t)batch_idx * cols + base;
                float block_acc = 0.0f;
                uint q_off = 0u;
                uint xoff = 0u;
                for (short is = 0; is < 8; is += 2) {
                    const uchar2 sm1 = q4_k_scale_min(uint(is), scales);
                    const uchar2 sm2 = q4_k_scale_min(uint(is) + 1u, scales);
                    const float d1 = d * float(sm1.x);
                    const float min1 = dmin * float(sm1.y);
                    const float d2 = d * float(sm2.x);
                    const float min2 = dmin * float(sm2.y);
                    for (short l = 0; l < 32; ++l) {
                        block_acc += (d1 * float(qs[q_off + l] & 0x0Fu) - min1) * xb[xoff + l];
                    }
                    for (short l = 0; l < 32; ++l) {
                        block_acc += (d2 * float(qs[q_off + l] >> 4) - min2) * xb[xoff + 32 + l];
                    }
                    q_off += 32u;
                    xoff += 64u;
                }
                acc[b] += block_acc;
            }
        }
        for (short b = 0; b < NB; ++b) {
            const int batch_idx = bt + b;
            if (batch_idx >= int(batch_size)) {
                break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float s = simd_sum(acc[b]);
            if ((tid & 31u) == 0u) {
                partial[tid / 32u] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float total = 0.0f;
                const uint nsg = (tg_size + 31u) / 32u;
                for (uint i = 0u; i < nsg; i++) {
                    total += partial[i];
                }
                out[(size_t)batch_idx * n_rows + row] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#;

/// True simdgroup GEMM for Q4_K weights — a port of llama.cpp's
/// `kernel_mul_mm` (`ggml-metal.metal`, the non-tensor-ops branch).
///
/// This is the kernel prefill was missing. Everything else in the Metal
/// prefill path decomposed a batched matmul into N independent matvecs,
/// so a 512-token prompt re-read the entire weight matrix 512 times.
/// Here each 64x32 output tile reads its slice of the weights **once**
/// into threadgroup memory, dequantized to half, and every one of the 32
/// tokens in the tile consumes it from there via `simdgroup_multiply_accumulate`.
///
/// Shape, matching llama exactly (these constants are load-bearing — the
/// index arithmetic below is derived from them):
///   NR0 = 64  weight rows per threadgroup  (M direction)
///   NR1 = 32  tokens per threadgroup       (N direction)
///   NK  = 32  K elements per iteration
///   128 threads = 4 simdgroups, each owning a 32x16 quadrant of the tile
///
/// A is `[n_rows, K]` Q4_K, B is `[batch, K]` f32, C is `[batch, n_rows]`
/// f32 — the same layouts the matvec path already uses, so this drops in
/// without touching callers.
/// The GEMM translation unit: the K-quant, legacy and IQ4_XS functors
/// written here, plus PrismML's ternary functor spliced in from
/// `crate::ternary` so the trit decode exists once (the matvec is
/// built from the same text), and the embedding gathers
/// (`crate::embd::get_rows_msl`), one per functor. Built on first use and leaked, because
/// `ensure_pipeline` keys its cache on a `&'static str`.
pub static K_QUANT_MUL_MM_SG_KERNEL_SRC: std::sync::LazyLock<&'static str> =
    std::sync::LazyLock::new(|| {
        let src = K_QUANT_MUL_MM_SG_KERNEL_BODY
            .replacen(
                "// @FRINK_TERNARY_FUNCTOR@",
                crate::ternary::PTQ1_0_DEQUANT_MSL,
                1,
            )
            .replacen(
                "// @FRINK_TERNARY_ENTRIES@",
                crate::ternary::PTQ1_0_GEMM_ENTRIES_MSL,
                1,
            )
            .replacen(
                "// @FRINK_LOW_K_FUNCTORS@",
                &[crate::q2_k::Q2_K_DEQUANT_MSL, crate::q3_k::Q3_K_DEQUANT_MSL].concat(),
                1,
            )
            .replacen(
                "// @FRINK_LOW_K_ENTRIES@",
                &[
                    crate::q2_k::Q2_K_GEMM_ENTRIES_MSL,
                    crate::q3_k::Q3_K_GEMM_ENTRIES_MSL,
                ]
                .concat(),
                1,
            )
            .replacen(
                "// @FRINK_CODEBOOK4_FUNCTORS@",
                crate::codebook4::CODEBOOK4_DEQUANT_MSL,
                1,
            )
            .replacen(
                "// @FRINK_CODEBOOK4_ENTRIES@",
                crate::codebook4::CODEBOOK4_GEMM_ENTRIES_MSL,
                1,
            )
            .replacen("// @FRINK_GET_ROWS@", &crate::embd::get_rows_msl(), 1);
        Box::leak(src.into_boxed_str())
    });

const K_QUANT_MUL_MM_SG_KERNEL_BODY: &str = r#"
#include <metal_stdlib>
using namespace metal;

// @FRINK_TERNARY_FUNCTOR@
// @FRINK_CODEBOOK4_FUNCTORS@

// @FRINK_LOW_K_FUNCTORS@

// llama `get_scale_min_k4_just2`: Q4_K packs eight 6-bit scale/min pairs
// into 12 bytes, the low four pairs plainly and the high four with their
// top 2 bits borrowed from the low bytes.
static inline uchar2 q4k_scale_min_just2(int j, int k, device const uchar* q) {
    return j < 4
        ? uchar2(uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63))
        : uchar2(uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2)),
                 uchar((q[j + 4 + k] >> 4)  | ((q[j - 0 + k] & 0xc0) >> 2)));
}

// llama `dequantize_q4_K`: 16 consecutive values of sub-block `il` of a
// 256-element Q4_K super-block. Byte layout is the GGUF one: d, dmin,
// 12 scale bytes, 128 nibble-packed quants.
static inline void q4k_dequant_16(device const uchar* xb, short il, thread float4x4& reg) {
    const float d_all = float(*(device const half*)(xb));
    const float min   = float(*(device const half*)(xb + 2));
    device const uchar* scales = xb + 4;
    device const uchar* q = xb + 16;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = q4k_scale_min_just2(is, il / 2, scales);
    const float d  = il < 2 ? d_all : d_all / 16.0f;
    const float dl = d * float(sc[0]);
    const float ml = min * float(sc[1]);
    const ushort mask = il < 2 ? 0x0F : 0xF0;

    for (short i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * float(q[i] & mask) - ml;
    }
}

// Dequant functors. The tile/simdgroup machinery below is identical for
// every K-quant -- only the 16-value unpack differs -- so it lives in one
// templated body rather than being copy-pasted per format. The previous
// generation of these kernels *was* copy-pasted, which is exactly how
// `gqa_prefill_fa_vec_d256` ended up handling half a head.
//
// Each functor also carries its block geometry, because the shared body
// needs it to walk the row: NL is how many 16-value sub-blocks a
// super-block holds (llama's `nl` template argument -- 16 for the
// 256-element K-quants, 2 for the 32-element legacy ones), and
// BLOCK_BYTES is the on-disk stride of one super-block.
struct Q4KDequant {
    static constexpr constant short NL = 16;
    static constexpr constant short BLOCK_BYTES = 144;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        q4k_dequant_16(xb, il, reg);
    }
};

// llama `dequantize_q8_0`. GGUF layout: half d, then 32 int8 quants.
// 32-element block, so NL = 2: `il` selects the low or high 16.
struct Q8_0Dequant {
    static constexpr constant short NL = 2;
    static constexpr constant short BLOCK_BYTES = 34;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        const float d = float(*(device const half*)(xb));
        device const char* qs = (device const char*)(xb + 2) + 16 * il;
        for (short i = 0; i < 16; ++i) {
            reg[i / 4][i % 4] = float(qs[i]) * d;
        }
    }
};

// llama `dequantize_q4_0`. GGUF layout: half d, then 16 bytes holding 32
// nibbles. llama reads them as uint16 pairs; the tensor is only 2-byte
// aligned per row here, so this composes the same words from bytes
// instead (same reason `Q6KDequant` does).
struct Q4_0Dequant {
    static constexpr constant short NL = 2;
    static constexpr constant short BLOCK_BYTES = 18;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        const float d = float(*(device const half*)(xb));
        device const uchar* qs = xb + 2;
        const float d1 = il ? d / 16.0f : d;
        const float d2 = d1 / 256.0f;
        const float md = -8.0f * d;
        const ushort mask0 = il ? 0x00F0 : 0x000F;
        const ushort mask1 = mask0 << 8;
        for (short i = 0; i < 8; ++i) {
            const ushort w = ushort(qs[2 * i]) | (ushort(qs[2 * i + 1]) << 8);
            reg[i / 2][2 * (i % 2) + 0] = d1 * float(w & mask0) + md;
            reg[i / 2][2 * (i % 2) + 1] = d2 * float(w & mask1) + md;
        }
    }
};

// llama `dequantize_q5_0`. GGUF: half d, uint32 qh, 16 qs bytes (32 nibbles).
// Byte-composed qs words — same alignment caveat as `Q4_0Dequant`.
struct Q5_0Dequant {
    static constexpr constant short NL = 2;
    static constexpr constant short BLOCK_BYTES = 22;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        const float d = float(*(device const half*)(xb));
        const float md = -16.0f * d;
        const uint qh = uint(xb[2]) | (uint(xb[3]) << 8) | (uint(xb[4]) << 16)
            | (uint(xb[5]) << 24);
        device const uchar* qs = xb + 6;
        const ushort mask = il ? 0x00F0 : 0x000F;
        const int x_mv = il ? 4 : 0;
        const int gh_mv = il ? 12 : 0;
        const int gh_bk = il ? 0 : 4;
        for (short i = 0; i < 8; ++i) {
            const ushort w = ushort(qs[2 * i]) | (ushort(qs[2 * i + 1]) << 8);
            const uchar xh_0 = uchar(((qh >> (gh_mv + 2 * i)) << gh_bk) & 0x10);
            const uchar xh_1 = uchar(((qh >> (gh_mv + 2 * i + 1)) << gh_bk) & 0x10);
            const int x0 = int((((w) & mask) >> x_mv) | xh_0);
            const int x1 = int((((w >> 8) & mask) >> x_mv) | xh_1);
            reg[i / 2][2 * (i % 2) + 0] = d * float(x0) + md;
            reg[i / 2][2 * (i % 2) + 1] = d * float(x1) + md;
        }
    }
};

// llama `dequantize_q5_K`. GGUF layout: half d, half dmin, scales[12],
// qh[32], qs[128] -- the 5th bit of each quant lives in `qh`.
struct Q5KDequant {
    static constexpr constant short NL = 16;
    static constexpr constant short BLOCK_BYTES = 176;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        const float d_all = float(*(device const half*)(xb));
        const float min = float(*(device const half*)(xb + 2));
        device const uchar* scales = xb + 4;
        device const uchar* q = xb + 48 + 32 * (il / 4) + 16 * (il & 1);
        device const uchar* qh = xb + 16 + 16 * (il & 1);

        const short is = (il / 4) * 2;
        const uchar ul = 1 << (il / 2);
        il = il & 3;
        const uchar2 sc = q4k_scale_min_just2(is, il / 2, scales);
        const float d = il < 2 ? d_all : d_all / 16.0f;
        const float dl = d * float(sc[0]);
        const float ml = min * float(sc[1]);

        const ushort mask = il < 2 ? 0x0F : 0xF0;
        const float qh_val = il < 2 ? 16.0f : 256.0f;
        for (short i = 0; i < 16; ++i) {
            reg[i / 4][i % 4] =
                dl * (float(q[i] & mask) + ((qh[i] & ul) ? qh_val : 0.0f)) - ml;
        }
    }
};

// llama `dequantize_iq4_xs`. GGUF layout: half d, uint16 scales_h,
// scales_l[4], qs[128]. Values come from the shared IQ4 codebook rather
// than an affine dequant, which is why this format had no GEMM before.
constant float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

struct IQ4XSDequant {
    static constexpr constant short NL = 16;
    static constexpr constant short BLOCK_BYTES = 136;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        const float dv = float(*(device const half*)(xb));
        const ushort scales_h = ushort(xb[2]) | (ushort(xb[3]) << 8);
        device const uchar* scales_l = xb + 4;
        device const uchar* qs = xb + 8;

        const short ib32 = il / 2;
        il = il % 2;
        device const uchar* q4 = qs + 16 * ib32;
        const int ls = int((scales_l[ib32 / 2] >> (4 * (ib32 % 2))) & 0xF)
                     | (int((scales_h >> (2 * ib32)) & 3) << 4);
        const float d = dv * float(ls - 32);
        const uchar shift = 4 * il;
        for (short i = 0; i < 4; ++i) {
            for (short j = 0; j < 4; ++j) {
                reg[i][j] = d * kvalues_iq4nl_f[(q4[4 * i + j] >> shift) & 0xF];
            }
        }
    }
};

// llama `dequantize_q6_K`. GGUF layout: ql[128], qh[64], int8 scales[16],
// half d.
struct Q6KDequant {
    static constexpr constant short NL = 16;
    static constexpr constant short BLOCK_BYTES = 210;
    static inline void get(device const uchar* xb, short il, thread float4x4& reg) {
        const float d_all = float(*(device const half*)(xb + 208));
        device const uchar* ql8 = xb;
        device const uchar* qh8 = xb + 128;
        device const char* scales = (device const char*)(xb + 192);

        const short ql_off = 64 * (il / 8) + 32 * ((il / 2) & 1) + 16 * (il & 1);
        const short qh_off = 32 * (il / 8) + 16 * (il & 1);
        const float sc = float(scales[(il % 2) + 2 * (il / 2)]);
        il = (il / 2) & 3;

        const uint kmask1 = il > 1 ? (il > 2 ? 0xC0C0C0C0u : 0x30303030u)
                                   : (il > 0 ? 0x0C0C0C0Cu : 0x03030303u);
        const uint kmask2 = il > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
        const float ml = d_all * sc * 32.0f;
        const float dl0 = d_all * sc;
        const float dl1 = dl0 / 256.0f;
        const float dl2 = dl0 / (256.0f * 256.0f);
        const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
        const uchar shr_h = il > 2 ? 2 : 0;
        const uchar shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);
        const uchar shr_l = il > 1 ? 4 : 0;

        for (short i = 0; i < 4; ++i) {
            // Rebuild llama's uint16 pair reads from bytes: the tensor is
            // only 2-byte aligned per row, so a ushort* cast is not safe.
            device const uchar* lp = ql8 + ql_off + 4 * i;
            device const uchar* hp = qh8 + qh_off + 4 * i;
            const uint low = ((uint(lp[0]) | (uint(lp[1]) << 8))
                           | ((uint(lp[2]) | (uint(lp[3]) << 8)) << 16)) & kmask2;
            const uint high = ((uint(hp[0]) | (uint(hp[1]) << 8))
                            | ((uint(hp[2]) | (uint(hp[3]) << 8)) << 16)) & kmask1;
            const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
            reg[i][0] = dl0 * float(q & 0xFFu) - ml;
            reg[i][1] = dl1 * float(q & 0xFF00u) - ml;
            reg[i][2] = dl2 * float(q & 0xFF0000u) - ml;
            reg[i][3] = dl3 * float(q & 0xFF000000u) - ml;
        }
    }
};

template<typename DQ, bool BC_OUT>
static inline void mul_mm_sg_impl(
    device const uchar* src0,
    device const float* src1,
    device float* dst,
    uint n_rows,
    uint n_cols,
    uint batch,
    uint row_bytes,
    threadgroup char* shmem,
    uint3 tgpig,
    ushort tiitg,
    ushort sgitg
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;   // 2 threads cover one row's 32 k-values
    constexpr short NL1 = NK / 8;    // 4 threads cover one token's 32 k-values
    constexpr short NL  = DQ::NL;    // 16-value sub-blocks per super-block

    const int r0 = int(tgpig.y) * NR0;
    const int r1 = int(tgpig.x) * NR1;

    const short nr0 = (int(n_rows) - r0 < NR0) ? short(int(n_rows) - r0) : NR0;
    const short nr1 = (int(batch)  - r1 < NR1) ? short(int(batch)  - r1) : NR1;

    // Clamp instead of branching: an out-of-range lane re-reads a valid
    // row and its result is discarded at the store. Keeps the hot loop
    // uniform, exactly as llama does.
    const short lr0 = (short(tiitg) / NL0) < nr0 ? (short(tiitg) / NL0) : nr0 - 1;
    const short lr1 = (short(tiitg) / NL1) < nr1 ? (short(tiitg) / NL1) : nr1 - 1;
    const short il0 = short(tiitg) % NL0;

    short il = il0;
    device const uchar* x = src0 + (size_t)row_bytes * (r0 + lr0);

    const short iy = 8 * (short(tiitg) % NL1);
    device const float* y = src1 + (size_t)(r1 + lr1) * n_cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < n_cols; loop_k += NK) {
        float4x4 temp_a;
        DQ::get(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scatter the 16 dequantized values into the 8x8-tile-major
        // layout `simdgroup_load` expects.
#pragma clang loop unroll(full)
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (short(tiitg) / NL0) / 8;
            const short lx = (short(tiitg) / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = half(temp_a[i / 4][i % 4]);
        }

        {
            const short sx = short(tiitg) % NL1;
            const short sy = (short(tiitg) / NL1) / 8;
            const short ly = (short(tiitg) / NL1) % 8;
            const short ib = 4 * sx + sy;
            // One 8-wide vector load + convert + store, not 8 scalar
            // round trips. This is llama's non-bounds-checked B-tile
            // path (`*(threadgroup S1_2x4 *)(sb + ...) = (S1_2x4)(*(device
            // T1_2x4 *) y)`); the scalar loop it replaces was costing
            // 8 loads and 8 stores per thread per K step. Both sides are
            // 32-byte aligned: `iy` is a multiple of 8 floats and every
            // real tensor's column count is a multiple of 8.
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
                half2x4(*(device const float2x4*)y);
        }

        // Advance two 16-value sub-blocks; step to the next super-block
        // when they wrap. For a 32-element format (NL == 2) `il` never
        // moves and every iteration steps a block, which is correct:
        // one block is exactly NK there.
        il = (il + 2 < NL) ? il + 2 : il % 2;
        if (il < 2) {
            x += DQ::BLOCK_BYTES;
        }
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

#pragma clang loop unroll(full)
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // BC_OUT=false is compiled only for dispatches whose dst tile grid is
    // exact (n_rows % 64 == 0 && batch % 32 == 0). Dropping the staging
    // branch drops the threadgroup allocation from 8192 to 4096+2048 =
    // 6144 bytes, which is what buys the extra resident threadgroup per
    // core. llama picks the same two pipelines: ggml-metal-device.cpp
    // `res.smem = bc_out ? 8192 : (4096 + 2048)`.
    if (!BC_OUT || (r0 + NR0 <= int(n_rows) && r1 + NR1 <= int(batch))) {
        device float* C = dst
            + (r0 + 32 * (sgitg & 1))
            + (size_t)(r1 + 16 * (sgitg >> 1)) * n_rows;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (size_t)n_rows * (i / 4), n_rows, 0, false);
        }
    } else if (BC_OUT) {
        // Partial tile: stage through threadgroup memory and copy only
        // the rows/columns that exist, so we never write past the matrix.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp = ((threadgroup float*)shmem)
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (short j = short(tiitg); j < nr1; j += NR1) {
                device float* D = dst + r0 + (size_t)(r1 + j) * n_rows;
                threadgroup const float* C = ((threadgroup float*)shmem) + j * NR0;
                // float4 for the aligned bulk, scalars for the tail --
                // llama does the same. The edge tile is copied by a
                // single simdgroup, so a scalar loop here serializes the
                // whole dispatch behind it.
                device float4* D4 = (device float4*)D;
                threadgroup const float4* C4 = (threadgroup const float4*)C;
                short i = 0;
                for (; i < nr0 / 4; i++) {
                    D4[i] = C4[i];
                }
                for (i *= 4; i < nr0; i++) {
                    D[i] = C[i];
                }
            }
        }
    }
}

// Same tile/simdgroup body as mul_mm_sg_impl but src1 is half (llama
// kernel_mul_mm_*_f16). Cuts activation bandwidth ~2× on dense prefill.
template<typename DQ, bool BC_OUT>
static inline void mul_mm_sg_impl_f16(
    device const uchar* src0,
    device const half* src1,
    device float* dst,
    uint n_rows,
    uint n_cols,
    uint batch,
    uint row_bytes,
    threadgroup char* shmem,
    uint3 tgpig,
    ushort tiitg,
    ushort sgitg
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short NL  = DQ::NL;

    const int r0 = int(tgpig.y) * NR0;
    const int r1 = int(tgpig.x) * NR1;

    const short nr0 = (int(n_rows) - r0 < NR0) ? short(int(n_rows) - r0) : NR0;
    const short nr1 = (int(batch)  - r1 < NR1) ? short(int(batch)  - r1) : NR1;

    const short lr0 = (short(tiitg) / NL0) < nr0 ? (short(tiitg) / NL0) : nr0 - 1;
    const short lr1 = (short(tiitg) / NL1) < nr1 ? (short(tiitg) / NL1) : nr1 - 1;
    const short il0 = short(tiitg) % NL0;

    short il = il0;
    device const uchar* x = src0 + (size_t)row_bytes * (r0 + lr0);

    const short iy = 8 * (short(tiitg) % NL1);
    device const half* y = src1 + (size_t)(r1 + lr1) * n_cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < n_cols; loop_k += NK) {
        float4x4 temp_a;
        DQ::get(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (short(tiitg) / NL0) / 8;
            const short lx = (short(tiitg) / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = half(temp_a[i / 4][i % 4]);
        }

        {
            const short sx = short(tiitg) % NL1;
            const short sy = (short(tiitg) / NL1) / 8;
            const short ly = (short(tiitg) / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
                *(device const half2x4*)y;
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        if (il < 2) {
            x += DQ::BLOCK_BYTES;
        }
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

#pragma clang loop unroll(full)
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // BC_OUT=false is compiled only for dispatches whose dst tile grid is
    // exact (n_rows % 64 == 0 && batch % 32 == 0). Dropping the staging
    // branch drops the threadgroup allocation from 8192 to 4096+2048 =
    // 6144 bytes, which is what buys the extra resident threadgroup per
    // core. llama picks the same two pipelines: ggml-metal-device.cpp
    // `res.smem = bc_out ? 8192 : (4096 + 2048)`.
    if (!BC_OUT || (r0 + NR0 <= int(n_rows) && r1 + NR1 <= int(batch))) {
        device float* C = dst
            + (r0 + 32 * (sgitg & 1))
            + (size_t)(r1 + 16 * (sgitg >> 1)) * n_rows;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (size_t)n_rows * (i / 4), n_rows, 0, false);
        }
    } else if (BC_OUT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp = ((threadgroup float*)shmem)
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (short j = short(tiitg); j < nr1; j += NR1) {
                device float* D = dst + r0 + (size_t)(r1 + j) * n_rows;
                threadgroup const float* C = ((threadgroup float*)shmem) + j * NR0;
                device float4* D4 = (device float4*)D;
                threadgroup const float4* C4 = (threadgroup const float4*)C;
                short i = 0;
                for (; i < nr0 / 4; i++) {
                    D4[i] = C4[i];
                }
                for (i *= 4; i < nr0; i++) {
                    D[i] = C[i];
                }
            }
        }
    }
}


// llama `kernel_mul_mm_id`: simdgroup GEMM with indexed src1 rows and
// scattered dst rows (MoE prefill). One z-slice per expert; `tpe[im]` is
// the batch count for that expert; `ids[im*ids_stride + j]` encodes
// `token*top_k + slot` for output layout `dst[id*rows + r]`.
template<typename DQ>
static inline void mul_mm_id_impl(
    device const uchar* src0,
    device const float* src1,
    device float* dst,
    device const int* ids,
    device const uint* tpe,
    uint expert_stride,
    uint n_rows,
    uint n_cols,
    uint top_k,
    uint ids_stride,
    uint row_bytes,
    uint src1_per_slot,
    threadgroup char* shmem,
    uint3 tgpig,
    ushort tiitg,
    ushort tiisg,
    ushort sgitg
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short NL  = DQ::NL;

    const uint im = tgpig.z;
    const int r0 = int(tgpig.y) * NR0;
    const int r1 = int(tgpig.x) * NR1;

    const int neh1 = int(tpe[im]);
    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (int(n_rows) - r0 < NR0) ? short(int(n_rows) - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;

    const short lr0 = (short(tiitg) / NL0) < nr0 ? (short(tiitg) / NL0) : nr0 - 1;
    const short lr1 = (short(tiitg) / NL1) < nr1 ? (short(tiitg) / NL1) : nr1 - 1;

    const short il0 = short(tiitg) % NL0;
    short il = il0;

    const int id = ids[im * ids_stride + r1 + lr1];
    const int src1_row = src1_per_slot != 0u ? id : (id / int(top_k));

    device const uchar* row_base = src0 + (size_t)im * expert_stride;
    device const uchar* x = row_base + (size_t)row_bytes * (r0 + lr0);

    const short iy = 8 * (short(tiitg) % NL1);
    device const float* y = src1 + (size_t)src1_row * n_cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < n_cols; loop_k += NK) {
        float4x4 temp_a;
        DQ::get(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (short(tiitg) / NL0) / 8;
            const short lx = (short(tiitg) / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = half(temp_a[i / 4][i % 4]);
        }

        {
            const short sx = short(tiitg) % NL1;
            const short sy = (short(tiitg) / NL1) / 8;
            const short ly = (short(tiitg) / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
                half2x4(*(device const float2x4*)y);
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        if (il < 2) {
            x += DQ::BLOCK_BYTES;
        }
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

#pragma clang loop unroll(full)
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* temp_str = ((threadgroup float*)shmem)
        + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int sid = ids[im * ids_stride + r1 + j];
        device float* D = dst + (size_t)sid * n_rows + r0;
        device float4* D4 = (device float4*)D;
        threadgroup float* C = ((threadgroup float*)shmem) + j * NR0;
        threadgroup float4* C4 = (threadgroup float4*)C;
        int i = int(tiisg);
        for (; i < nr0 / 4; i += 32) {
            D4[i] = C4[i];
        }
        i = (4 * (nr0 / 4)) + int(tiisg);
        for (; i < nr0; i += 32) {
            D[i] = C[i];
        }
    }
}

// f16 src1 twin of `mul_mm_id_impl` (dense prefill `mul_mm_sg_f16` path).
template<typename DQ>
static inline void mul_mm_id_impl_f16(
    device const uchar* src0,
    device const half* src1,
    device float* dst,
    device const int* ids,
    device const uint* tpe,
    uint expert_stride,
    uint n_rows,
    uint n_cols,
    uint top_k,
    uint ids_stride,
    uint row_bytes,
    uint src1_per_slot,
    threadgroup char* shmem,
    uint3 tgpig,
    ushort tiitg,
    ushort tiisg,
    ushort sgitg
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short NL  = DQ::NL;

    const uint im = tgpig.z;
    const int r0 = int(tgpig.y) * NR0;
    const int r1 = int(tgpig.x) * NR1;

    const int neh1 = int(tpe[im]);
    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (int(n_rows) - r0 < NR0) ? short(int(n_rows) - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;

    const short lr0 = (short(tiitg) / NL0) < nr0 ? (short(tiitg) / NL0) : nr0 - 1;
    const short lr1 = (short(tiitg) / NL1) < nr1 ? (short(tiitg) / NL1) : nr1 - 1;

    const short il0 = short(tiitg) % NL0;
    short il = il0;

    const int id = ids[im * ids_stride + r1 + lr1];
    const int src1_row = src1_per_slot != 0u ? id : (id / int(top_k));

    device const uchar* row_base = src0 + (size_t)im * expert_stride;
    device const uchar* x = row_base + (size_t)row_bytes * (r0 + lr0);

    const short iy = 8 * (short(tiitg) % NL1);
    device const half* y = src1 + (size_t)src1_row * n_cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < n_cols; loop_k += NK) {
        float4x4 temp_a;
        DQ::get(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (short(tiitg) / NL0) / 8;
            const short lx = (short(tiitg) / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = half(temp_a[i / 4][i % 4]);
        }

        {
            const short sx = short(tiitg) % NL1;
            const short sy = (short(tiitg) / NL1) / 8;
            const short ly = (short(tiitg) / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
                *(device const half2x4*)y;
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        if (il < 2) {
            x += DQ::BLOCK_BYTES;
        }
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);

#pragma clang loop unroll(full)
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* temp_str = ((threadgroup float*)shmem)
        + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int sid = ids[im * ids_stride + r1 + j];
        device float* D = dst + (size_t)sid * n_rows + r0;
        device float4* D4 = (device float4*)D;
        threadgroup float* C = ((threadgroup float*)shmem) + j * NR0;
        threadgroup float4* C4 = (threadgroup float4*)C;
        int i = int(tiisg);
        for (; i < nr0 / 4; i += 32) {
            D4[i] = C4[i];
        }
        i = (4 * (nr0 / 4)) + int(tiisg);
        for (; i < nr0; i += 32) {
            D[i] = C[i];
        }
    }
}

#define MUL_MM_ID_ENTRY(NAME, DQ)                                           \
kernel void NAME(                                                           \
    device const uchar* src0 [[buffer(0)]],                                 \
    device const float* src1 [[buffer(1)]],                                 \
    device float* dst [[buffer(2)]],                                        \
    device const int* ids [[buffer(3)]],                                    \
    device const uint* tpe [[buffer(4)]],                                   \
    constant uint& n_rows [[buffer(5)]],                                    \
    constant uint& n_cols [[buffer(6)]],                                    \
    constant uint& top_k [[buffer(7)]],                                     \
    constant uint& ids_stride [[buffer(8)]],                                \
    constant uint& row_bytes [[buffer(9)]],                                 \
    constant uint& expert_stride [[buffer(10)]],                            \
    constant uint& src1_per_slot [[buffer(11)]],                            \
    threadgroup char* shmem [[threadgroup(0)]],                             \
    uint3 tgpig [[threadgroup_position_in_grid]],                           \
    ushort tiitg [[thread_index_in_threadgroup]],                           \
    ushort tiisg [[thread_index_in_simdgroup]],                             \
    ushort sgitg [[simdgroup_index_in_threadgroup]]                         \
) {                                                                         \
    mul_mm_id_impl<DQ>(src0, src1, dst, ids, tpe, expert_stride, n_rows,   \
                       n_cols, top_k, ids_stride, row_bytes, src1_per_slot,  \
                       shmem, tgpig, tiitg, tiisg, sgitg);                  \
}

#define MUL_MM_ID_F16_ENTRY(NAME, DQ)                                       \
kernel void NAME(                                                           \
    device const uchar* src0 [[buffer(0)]],                                 \
    device const half* src1 [[buffer(1)]],                                  \
    device float* dst [[buffer(2)]],                                        \
    device const int* ids [[buffer(3)]],                                    \
    device const uint* tpe [[buffer(4)]],                                   \
    constant uint& n_rows [[buffer(5)]],                                    \
    constant uint& n_cols [[buffer(6)]],                                    \
    constant uint& top_k [[buffer(7)]],                                     \
    constant uint& ids_stride [[buffer(8)]],                                \
    constant uint& row_bytes [[buffer(9)]],                                 \
    constant uint& expert_stride [[buffer(10)]],                            \
    constant uint& src1_per_slot [[buffer(11)]],                            \
    threadgroup char* shmem [[threadgroup(0)]],                             \
    uint3 tgpig [[threadgroup_position_in_grid]],                           \
    ushort tiitg [[thread_index_in_threadgroup]],                           \
    ushort tiisg [[thread_index_in_simdgroup]],                             \
    ushort sgitg [[simdgroup_index_in_threadgroup]]                         \
) {                                                                         \
    mul_mm_id_impl_f16<DQ>(src0, src1, dst, ids, tpe, expert_stride, n_rows, \
                           n_cols, top_k, ids_stride, row_bytes,             \
                           src1_per_slot, shmem, tgpig, tiitg, tiisg, sgitg); \
}

#define MUL_MM_SG_ENTRY(NAME, DQ)                                           \
kernel void NAME(                                                           \
    device const uchar* src0 [[buffer(0)]],                                 \
    device const float* src1 [[buffer(1)]],                                 \
    device float* dst [[buffer(2)]],                                        \
    constant uint& n_rows [[buffer(3)]],                                    \
    constant uint& n_cols [[buffer(4)]],                                    \
    constant uint& batch [[buffer(5)]],                                     \
    constant uint& row_bytes [[buffer(6)]],                                 \
    threadgroup char* shmem [[threadgroup(0)]],                             \
    uint3 tgpig [[threadgroup_position_in_grid]],                           \
    ushort tiitg [[thread_index_in_threadgroup]],                           \
    ushort sgitg [[simdgroup_index_in_threadgroup]]                         \
) {                                                                         \
    mul_mm_sg_impl<DQ, true>(src0, src1, dst, n_rows, n_cols, batch,        \
                             row_bytes, shmem, tgpig, tiitg, sgitg);        \
}                                                                           \
kernel void NAME##_a(                                                       \
    device const uchar* src0 [[buffer(0)]],                                 \
    device const float* src1 [[buffer(1)]],                                 \
    device float* dst [[buffer(2)]],                                        \
    constant uint& n_rows [[buffer(3)]],                                    \
    constant uint& n_cols [[buffer(4)]],                                    \
    constant uint& batch [[buffer(5)]],                                     \
    constant uint& row_bytes [[buffer(6)]],                                 \
    threadgroup char* shmem [[threadgroup(0)]],                             \
    uint3 tgpig [[threadgroup_position_in_grid]],                           \
    ushort tiitg [[thread_index_in_threadgroup]],                           \
    ushort sgitg [[simdgroup_index_in_threadgroup]]                         \
) {                                                                         \
    mul_mm_sg_impl<DQ, false>(src0, src1, dst, n_rows, n_cols, batch,       \
                              row_bytes, shmem, tgpig, tiitg, sgitg);       \
}

#define MUL_MM_SG_F16_ENTRY(NAME, DQ)                                       \
kernel void NAME(                                                           \
    device const uchar* src0 [[buffer(0)]],                                 \
    device const half* src1 [[buffer(1)]],                                  \
    device float* dst [[buffer(2)]],                                        \
    constant uint& n_rows [[buffer(3)]],                                    \
    constant uint& n_cols [[buffer(4)]],                                    \
    constant uint& batch [[buffer(5)]],                                     \
    constant uint& row_bytes [[buffer(6)]],                                 \
    threadgroup char* shmem [[threadgroup(0)]],                             \
    uint3 tgpig [[threadgroup_position_in_grid]],                           \
    ushort tiitg [[thread_index_in_threadgroup]],                           \
    ushort sgitg [[simdgroup_index_in_threadgroup]]                         \
) {                                                                         \
    mul_mm_sg_impl_f16<DQ, true>(src0, src1, dst, n_rows, n_cols, batch,    \
                                 row_bytes, shmem, tgpig, tiitg, sgitg);    \
}                                                                           \
kernel void NAME##_a(                                                       \
    device const uchar* src0 [[buffer(0)]],                                 \
    device const half* src1 [[buffer(1)]],                                  \
    device float* dst [[buffer(2)]],                                        \
    constant uint& n_rows [[buffer(3)]],                                    \
    constant uint& n_cols [[buffer(4)]],                                    \
    constant uint& batch [[buffer(5)]],                                     \
    constant uint& row_bytes [[buffer(6)]],                                 \
    threadgroup char* shmem [[threadgroup(0)]],                             \
    uint3 tgpig [[threadgroup_position_in_grid]],                           \
    ushort tiitg [[thread_index_in_threadgroup]],                           \
    ushort sgitg [[simdgroup_index_in_threadgroup]]                         \
) {                                                                         \
    mul_mm_sg_impl_f16<DQ, false>(src0, src1, dst, n_rows, n_cols, batch,   \
                                  row_bytes, shmem, tgpig, tiitg, sgitg);   \
}

MUL_MM_SG_ENTRY(q4_k_mul_mm_sg, Q4KDequant)
MUL_MM_SG_ENTRY(q6_k_mul_mm_sg, Q6KDequant)
MUL_MM_SG_ENTRY(q5_k_mul_mm_sg, Q5KDequant)
MUL_MM_SG_ENTRY(q8_0_mul_mm_sg, Q8_0Dequant)
MUL_MM_SG_ENTRY(q4_0_mul_mm_sg, Q4_0Dequant)
MUL_MM_SG_ENTRY(q5_0_mul_mm_sg, Q5_0Dequant)
MUL_MM_SG_ENTRY(iq4_xs_mul_mm_sg, IQ4XSDequant)
// @FRINK_TERNARY_ENTRIES@
// @FRINK_LOW_K_ENTRIES@
// @FRINK_CODEBOOK4_ENTRIES@
MUL_MM_ID_ENTRY(q4_0_mul_mm_id, Q4_0Dequant)
MUL_MM_ID_ENTRY(q4_k_mul_mm_id, Q4KDequant)
MUL_MM_ID_ENTRY(q8_0_mul_mm_id, Q8_0Dequant)
MUL_MM_ID_F16_ENTRY(q4_0_mul_mm_id_f16, Q4_0Dequant)
MUL_MM_ID_F16_ENTRY(q4_k_mul_mm_id_f16, Q4KDequant)
MUL_MM_ID_F16_ENTRY(q8_0_mul_mm_id_f16, Q8_0Dequant)
MUL_MM_SG_F16_ENTRY(q4_k_mul_mm_sg_f16, Q4KDequant)
MUL_MM_SG_F16_ENTRY(q6_k_mul_mm_sg_f16, Q6KDequant)
MUL_MM_SG_F16_ENTRY(q5_k_mul_mm_sg_f16, Q5KDequant)
MUL_MM_SG_F16_ENTRY(q8_0_mul_mm_sg_f16, Q8_0Dequant)
MUL_MM_SG_F16_ENTRY(q4_0_mul_mm_sg_f16, Q4_0Dequant)
MUL_MM_SG_F16_ENTRY(q5_0_mul_mm_sg_f16, Q5_0Dequant)
MUL_MM_SG_F16_ENTRY(iq4_xs_mul_mm_sg_f16, IQ4XSDequant)
// @FRINK_GET_ROWS@
"#;

/// Launches the simdgroup Q4_K GEMM ([`K_QUANT_MUL_MM_SG_KERNEL_SRC`]).
///
/// `x_batch` is `[batch, cols]` f32, the result is `[batch, rows]` f32 —
/// identical to [`launch_q4_k_mul_mm`], so the two are directly
/// comparable and the tests assert exactly that.
///
/// Requires `cols % 256 == 0` (Q4_K super-block size), which every real
/// Q4_K tensor satisfies by construction.
pub fn launch_q4_k_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_k_quant_mul_mm_sg(
        weights,
        x_batch,
        rows,
        row_bytes,
        batch_size,
        "q4_k_mul_mm_sg",
        144,
        256,
        None,
    )
}

/// Q5_K twin of [`launch_q4_k_mul_mm_sg`] (`*_Q5_K_M` checkpoints).
pub fn launch_q5_k_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_k_quant_mul_mm_sg(
        weights,
        x_batch,
        rows,
        row_bytes,
        batch_size,
        "q5_k_mul_mm_sg",
        176,
        256,
        None,
    )
}

/// Q8_0 twin of [`launch_q4_k_mul_mm_sg`]. Q8_0 carried the *worst*
/// prefill rows in the suite (SmolLM2 metal `pp512` 29.6x behind,
/// Qwen2.5-0.5B 20.6x, Gemma-3-1B 20.8x) purely because no GEMM existed
/// for it and every token re-read the whole matrix.
pub fn launch_q8_0_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_k_quant_mul_mm_sg(
        weights,
        x_batch,
        rows,
        row_bytes,
        batch_size,
        "q8_0_mul_mm_sg",
        34,
        32,
        None,
    )
}

/// Q4_0 twin of [`launch_q4_k_mul_mm_sg`] (OLMoE and the other Q4_0
/// checkpoints). Replaces `launch_q4_0_mul_mm`, which was a batched
/// matvec rather than a tiled GEMM.
pub fn launch_q4_0_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_k_quant_mul_mm_sg(
        weights,
        x_batch,
        rows,
        row_bytes,
        batch_size,
        "q4_0_mul_mm_sg",
        18,
        32,
        None,
    )
}

/// IQ4_XS twin of [`launch_q4_k_mul_mm_sg`]. The IQ codebook kinds were
/// explicitly kept *off* the batched path (`prefers_gpu_batch`) because
/// the batched-matvec fallback lost to N x matvec for them; with a real
/// GEMM that reason no longer applies.
pub fn launch_iq4_xs_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_k_quant_mul_mm_sg(
        weights,
        x_batch,
        rows,
        row_bytes,
        batch_size,
        "iq4_xs_mul_mm_sg",
        136,
        256,
        None,
    )
}

/// PTQ1_0 twin, on the same body through `crate::ternary`'s dequant
/// functor (`MUL_MM_SG_ENTRY(ptq1_0_mul_mm_sg, PTQ1_0Dequant)`).
pub fn launch_ptq1_0_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_k_quant_mul_mm_sg(
        weights,
        x_batch,
        rows,
        row_bytes,
        batch_size,
        "ptq1_0_mul_mm_sg",
        28,
        128,
        None,
    )
}

/// Q6_K twin of [`launch_q4_k_mul_mm_sg`]. `ffn_down` and `attn_v` are
/// Q6_K in every `*_Q4_K_M` checkpoint -- `ffn_down` alone is a third of
/// the FFN -- so leaving Q6_K on the batched-matvec path capped what the
/// Q4_K GEMM could deliver.
pub fn launch_q6_k_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_k_quant_mul_mm_sg(
        weights,
        x_batch,
        rows,
        row_bytes,
        batch_size,
        "q6_k_mul_mm_sg",
        210,
        256,
        None,
    )
}

/// Recycled scratch `MTLBuffer`s, keyed by exact byte length.
///
/// Every batched launch used to allocate its output and intermediates
/// fresh and drop them on return: at batch 512 on an 8B model the FFN
/// down-projection alone allocates and frees 28 MiB, and allocation
/// accounts for roughly 318 ms of an 8B prefill. llama.cpp allocates
/// nothing while encoding a graph — its buffers are kept alive across
/// submissions (`ggml_metal_device_rsets_keep_alive`).
///
/// Reuse is safe because every launch here is synchronous: it commits
/// and calls `waitUntilCompleted` before the guard drops, so the GPU is
/// provably done with the buffer before it returns to the pool.
struct ScratchPool {
    free: HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    bytes: usize,
}

// SAFETY: same justification as `SharedMetal` -- `MTLBuffer`s are safe
// to share across threads; only `MTLCommandBuffer`/encoders are not.
unsafe impl Send for ScratchPool {}

static SCRATCH_POOL: Mutex<Option<ScratchPool>> = Mutex::new(None);

/// Upper bound on pooled bytes. Past this, buffers are dropped on return
/// rather than retained, so a one-off huge batch cannot pin memory for
/// the life of the process.
fn scratch_pool_budget_bytes() -> usize {
    std::env::var("FRINK_METAL_SCRATCH_BUDGET_BYTES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(768 * 1024 * 1024)
}

/// A scratch buffer borrowed from [`SCRATCH_POOL`], returned on drop.
struct ScratchBuffer {
    buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    bytes: usize,
}

impl ScratchBuffer {
    fn get(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.buf
            .as_ref()
            .expect("borrowed for the guard's lifetime")
    }
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        let Some(buf) = self.buf.take() else { return };
        let mut guard = match SCRATCH_POOL.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let pool = guard.get_or_insert_with(|| ScratchPool {
            free: HashMap::new(),
            bytes: 0,
        });
        if pool.bytes + self.bytes > scratch_pool_budget_bytes() {
            return; // over budget: let it go
        }
        pool.bytes += self.bytes;
        pool.free.entry(self.bytes).or_default().push(buf);
    }
}

/// A zero-initialisation-free scratch buffer of exactly `bytes` bytes.
/// Contents are undefined — every caller either fully overwrites it with
/// a kernel or copies into it first.
fn borrow_scratch(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    bytes: usize,
    storage: MTLResourceOptions,
) -> Result<ScratchBuffer, MetalError> {
    // Private and Shared buffers are not interchangeable, so they are
    // pooled under distinct keys: the low bit of the length is unused
    // (every allocation here is a multiple of 4) and carries the mode.
    let key = bytes | usize::from(storage.contains(MTLResourceOptions::StorageModePrivate));
    if let Ok(mut guard) = SCRATCH_POOL.lock() {
        if let Some(pool) = guard.as_mut() {
            if let Some(buf) = pool.free.get_mut(&key).and_then(|v| v.pop()) {
                pool.bytes = pool.bytes.saturating_sub(key);
                return Ok(ScratchBuffer {
                    buf: Some(buf),
                    bytes: key,
                });
            }
        }
    }
    let buf = device
        .newBufferWithLength_options(bytes, storage)
        .ok_or(MetalError::BufferAllocFailed)?;
    Ok(ScratchBuffer {
        buf: Some(buf),
        bytes: key,
    })
}

/// One quantized matrix bound for the simdgroup GEMM, as a descriptor
/// so several can be encoded into a single command buffer.
pub struct MulMmSgLaunch<'a> {
    pub weights: &'a [u8],
    pub rows: usize,
    pub row_bytes: usize,
    pub fn_name: &'static str,
    pub block_bytes: usize,
    pub block_elems: usize,
}

impl MulMmSgLaunch<'_> {
    fn cols(&self) -> usize {
        (self.row_bytes / self.block_bytes) * self.block_elems
    }
}

/// The simdgroup GEMM for any kind [`mul_mm_sg_meta`] names, read off
/// that table. `Ok(None)` is "no GEMM for this kind"; the per-kind
/// `launch_*_mul_mm_sg` wrappers above are this call with the row
/// written out, and `apply_gpu_batch` used to name them one by one in
/// a match that lacked Q5_0 and PTQ1_0 while `gemm_supported` said both
/// had a GEMM, so a Bonsai prefill ran N matvecs per matrix with the
/// kernel registry reporting a GEMM hit.
pub fn launch_mul_mm_sg(
    kind: &str,
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Option<Vec<f32>>, MetalError> {
    let Some((fn_name, block_bytes, block_elems)) = mul_mm_sg_meta(kind) else {
        return Ok(None);
    };
    launch_k_quant_mul_mm_sg(
        weights,
        x_batch,
        rows,
        row_bytes,
        batch_size,
        fn_name,
        block_bytes,
        block_elems,
        None,
    )
    .map(Some)
}

/// Kernel name and block geometry for the kinds that have a simdgroup
/// GEMM. `None` for anything still limited to a matvec.
pub fn mul_mm_sg_meta(kind: &str) -> Option<(&'static str, usize, usize)> {
    match kind {
        "Q4_K" => Some(("q4_k_mul_mm_sg", 144, 256)),
        "Q5_K" => Some(("q5_k_mul_mm_sg", 176, 256)),
        "Q6_K" => Some(("q6_k_mul_mm_sg", 210, 256)),
        "Q8_0" => Some(("q8_0_mul_mm_sg", 34, 32)),
        "Q4_0" => Some(("q4_0_mul_mm_sg", 18, 32)),
        "Q5_0" => Some(("q5_0_mul_mm_sg", 22, 32)),
        "IQ4_XS" => Some(("iq4_xs_mul_mm_sg", 136, 256)),
        "PTQ1_0" => Some(("ptq1_0_mul_mm_sg", 28, 128)),
        "Q2_K" => Some(("q2_k_mul_mm_sg", 84, 256)),
        "Q3_K" => Some(("q3_k_mul_mm_sg", 110, 256)),
        "IQ4_NL" => Some(("iq4_nl_mul_mm_sg", 18, 32)),
        "MXFP4" => Some(("mxfp4_mul_mm_sg", 17, 32)),
        _ => None,
    }
}

/// MoE indexed simdgroup GEMM (`kernel_mul_mm_id_*`).
pub fn mul_mm_id_meta(kind: &str) -> Option<(&'static str, usize, usize)> {
    match kind {
        "Q4_K" => Some(("q4_k_mul_mm_id", 144, 256)),
        "Q8_0" => Some(("q8_0_mul_mm_id", 34, 32)),
        "Q4_0" => Some(("q4_0_mul_mm_id", 18, 32)),
        _ => None,
    }
}

pub fn mul_mm_id_f16_meta(kind: &str) -> Option<(&'static str, usize, usize)> {
    mul_mm_id_meta(kind).map(|(base, bb, be)| {
        (
            match base {
                "q4_k_mul_mm_id" => "q4_k_mul_mm_id_f16",
                "q8_0_mul_mm_id" => "q8_0_mul_mm_id_f16",
                "q4_0_mul_mm_id" => "q4_0_mul_mm_id_f16",
                _ => base,
            },
            bb,
            be,
        )
    })
}

fn moe_mm_id_map0_fn(top_k: usize) -> Option<&'static str> {
    match top_k {
        2 => Some("moe_mm_id_map0_ne20_2"),
        4 => Some("moe_mm_id_map0_ne20_4"),
        6 => Some("moe_mm_id_map0_ne20_6"),
        8 => Some("moe_mm_id_map0_ne20_8"),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_mm_id_map0(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    route_ids: &ProtocolObject<dyn MTLBuffer>,
    tpe_buf: &ProtocolObject<dyn MTLBuffer>,
    mm_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    n_experts: u32,
    n_tokens: u32,
    top_k: usize,
) -> Result<(), MetalError> {
    let fn_name = moe_mm_id_map0_fn(top_k).ok_or(MetalError::CommandFailed)?;
    let pipeline = ensure_pipeline(device, Q4_0_MOE_TOPK_KERNEL_SRC, fn_name)?;
    let smem = (n_experts as usize) * top_k * 2;
    enc.setComputePipelineState(&pipeline.0);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(route_ids), 0, 0);
        enc.setBuffer_offset_atIndex(Some(tpe_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(mm_ids_buf), 0, 2);
        let mut nt = n_tokens;
        enc.setBytes_length_atIndex(NonNull::new(&mut nt as *mut u32 as *mut _).unwrap(), 4, 3);
        enc.setThreadgroupMemoryLength_atIndex(smem, 0);
    }
    dispatch_counted(
        enc,
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: n_experts as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mul_mm_id_f16(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    fn_name: &'static str,
    w: &ResidentWeightBuffer,
    expert_stride: u32,
    x_h: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    ids_buf: &ProtocolObject<dyn MTLBuffer>,
    tpe_buf: &ProtocolObject<dyn MTLBuffer>,
    n_experts: u32,
    n_tokens: u32,
    top_k: u32,
    rows: u32,
    cols: u32,
    row_bytes: u32,
    src1_per_slot: u32,
) -> Result<(), MetalError> {
    let pipeline = ensure_pipeline(device, *K_QUANT_MUL_MM_SG_KERNEL_SRC, fn_name)?;
    unsafe {
        enc.setComputePipelineState(&pipeline.0);
        enc.setBuffer_offset_atIndex(Some(&w.buffer), w.weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(x_h), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(ids_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(tpe_buf), 0, 4);
        for (idx, mut v) in [
            (5usize, rows),
            (6, cols),
            (7, top_k),
            (8, n_tokens),
            (9, row_bytes),
            (10, expert_stride),
            (11, src1_per_slot),
        ] {
            enc.setBytes_length_atIndex(
                NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                4,
                idx,
            );
        }
        enc.setThreadgroupMemoryLength_atIndex(8192, 0);
        dispatch_counted(
            enc,
            MTLSize {
                width: (n_tokens as usize).div_ceil(32),
                height: (rows as usize).div_ceil(64),
                depth: n_experts as usize,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_router_mm_f32(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    router_w: &ProtocolObject<dyn MTLBuffer>,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    logits_buf: &ProtocolObject<dyn MTLBuffer>,
    hidden: u32,
    n_experts: u32,
    n_tokens: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, Q4_0_MOE_TOPK_KERNEL_SRC, "moe_router_mm_f32")?;
    enc.setComputePipelineState(&pipe.0);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(router_w), 0, 0);
        enc.setBuffer_offset_atIndex(Some(x_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(logits_buf), 0, 2);
        for (idx, mut v) in [(3usize, hidden), (4, n_experts), (5, n_tokens)] {
            enc.setBytes_length_atIndex(
                NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                4,
                idx,
            );
        }
    }
    dispatch_counted(
        enc,
        MTLSize {
            width: n_experts as usize,
            height: n_tokens as usize,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Batched softmax top-k routing (`n ≤ 256`, `k ≤ 8`), one simdgroup per
/// token. Writes `ids [n_tokens, k]` and `weights [n_tokens, k]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_topk_softmax_batch(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    logits: &ProtocolObject<dyn MTLBuffer>,
    ids: IdsBinding<'_>,
    weights: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    k: u32,
    renormalize: bool,
    n_tokens: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, Q4_0_MOE_TOPK_KERNEL_SRC, "moe_topk_softmax_batch")?;
    enc.setComputePipelineState(&pipe.0);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
        enc.setBuffer_offset_atIndex(Some(ids.buf), ids.offset, 1);
        enc.setBuffer_offset_atIndex(Some(weights), 0, 2);
        for (idx, mut v) in [
            (3usize, n),
            (4, k),
            (5, u32::from(renormalize)),
            (6, n_tokens),
        ] {
            enc.setBytes_length_atIndex(
                NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                4,
                idx,
            );
        }
        enc.setThreadgroupMemoryLength_atIndex((n as usize) * 4, 0);
    }
    dispatch_counted(
        enc,
        MTLSize {
            width: n_tokens as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Routed-expert FFN for one layer of the fused prefill stack: GPU router
/// GEMM → top-k softmax → `mul_mm_id_map0` → indexed gate/up GEMM → SiLU
/// mul → indexed down GEMM → weighted sum. Everything stays on device, so
/// a MoE layer costs the same *zero* extra command buffers a dense layer
/// does (llama.cpp `build_moe_ffn` shape).
///
/// `x_f32` is the FFN-normed activation `[T, H]` (router input, kept f32
/// because top-k is tie-sensitive); `x_f16` is the same rows in f16 (the
/// expert GEMM input). `out` receives `[T, H]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_prefill_ffn(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    mrs: &mut crate::mem_ranges::MemRanges,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    moe: &PrefillMoeMetal<'_>,
    bound: &MoePackedResident,
    router_w: &ResidentF32Buffer,
    x_f32: &ProtocolObject<dyn MTLBuffer>,
    x_f16: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_tokens: usize,
) -> Result<(), MetalError> {
    let packed = &moe.packed;
    let top_k = moe.top_k;
    let n_experts = packed.n_experts;
    let hidden = packed.hidden_rows;
    let ffn = packed.ffn_rows;
    let n_slots = n_tokens * top_k;

    let (gate_fn, gate_bb, gate_be) =
        mul_mm_id_f16_meta(packed.gate_kind).ok_or(MetalError::CommandFailed)?;
    let (up_fn, up_bb, up_be) =
        mul_mm_id_f16_meta(packed.up_kind).ok_or(MetalError::CommandFailed)?;
    let (down_fn, down_bb, down_be) =
        mul_mm_id_f16_meta(packed.down_kind).ok_or(MetalError::CommandFailed)?;
    let up_row_bytes = packed.up_stride / ffn;
    let gate_cols = ((packed.gate_row_bytes / gate_bb) * gate_be) as u32;
    let up_cols = ((up_row_bytes / up_bb) * up_be) as u32;
    let down_cols = ((packed.down_row_bytes / down_bb) * down_be) as u32;

    moe_prefill_scratch_ex(device, n_tokens, n_slots, ffn, hidden, n_experts, top_k)?;

    TL_MOE_PREFILL.with(|cell| {
        let scratch = cell.borrow();
        let scratch = scratch.as_ref().ok_or(MetalError::CommandFailed)?;
        let logits = scratch.router_logits.as_ref();
        let route_ids = scratch.route_ids.as_ref();
        let route_w = scratch.route_w.as_ref();
        let tpe = scratch.mm_id_tpe.as_ref();
        let mm_ids = scratch.mm_id_ids.as_ref();
        let gate_buf = scratch.gate.as_ref();
        let up_buf = scratch.up.as_ref();
        let half_slots = scratch.half_in.as_ref();
        let expert_out = scratch.expert_out.as_ref();

        mrs.begin_op(enc, &[x_f32], &[logits]);
        encode_moe_router_mm_f32(
            enc,
            device,
            &router_w.buffer,
            x_f32,
            logits,
            hidden as u32,
            n_experts as u32,
            n_tokens as u32,
        )?;
        mrs.end_op(&[x_f32], &[logits]);
        mrs.begin_op(enc, &[logits], &[route_ids, route_w]);
        encode_moe_topk_softmax_batch(
            enc,
            device,
            logits,
            IdsBinding::whole(route_ids),
            route_w,
            n_experts as u32,
            top_k as u32,
            moe.renormalize,
            n_tokens as u32,
        )?;
        mrs.end_op(&[logits], &[route_ids, route_w]);
        mrs.begin_op(enc, &[route_ids], &[tpe, mm_ids]);
        encode_moe_mm_id_map0(
            enc,
            device,
            route_ids,
            tpe,
            mm_ids,
            n_experts as u32,
            n_tokens as u32,
            top_k,
        )?;
        mrs.end_op(&[route_ids], &[tpe, mm_ids]);

        mrs.begin_op(enc, &[x_f16, mm_ids, tpe], &[gate_buf, up_buf]);
        encode_mul_mm_id_f16(
            enc,
            device,
            gate_fn,
            &bound.gate,
            packed.gate_stride as u32,
            x_f16,
            gate_buf,
            mm_ids,
            tpe,
            n_experts as u32,
            n_tokens as u32,
            top_k as u32,
            ffn as u32,
            gate_cols,
            packed.gate_row_bytes as u32,
            0,
        )?;
        encode_mul_mm_id_f16(
            enc,
            device,
            up_fn,
            &bound.up,
            packed.up_stride as u32,
            x_f16,
            up_buf,
            mm_ids,
            tpe,
            n_experts as u32,
            n_tokens as u32,
            top_k as u32,
            ffn as u32,
            up_cols,
            up_row_bytes as u32,
            0,
        )?;
        mrs.end_op(&[x_f16, mm_ids, tpe], &[gate_buf, up_buf]);
        // SwiGLU straight to f16 — the staging convert is folded in.
        mrs.begin_op(enc, &[gate_buf, up_buf], &[half_slots]);
        crate::elem::encode_act_mul_f32_to_f16(
            enc,
            device,
            gate_buf,
            up_buf,
            half_slots,
            (n_slots * ffn) as u32,
            false,
        )?;
        mrs.end_op(&[gate_buf, up_buf], &[half_slots]);
        mrs.begin_op(enc, &[half_slots, mm_ids, tpe], &[expert_out]);
        encode_mul_mm_id_f16(
            enc,
            device,
            down_fn,
            &bound.down,
            packed.down_stride as u32,
            half_slots,
            expert_out,
            mm_ids,
            tpe,
            n_experts as u32,
            n_tokens as u32,
            top_k as u32,
            hidden as u32,
            down_cols,
            packed.down_row_bytes as u32,
            1,
        )?;
        mrs.end_op(&[half_slots, mm_ids, tpe], &[expert_out]);
        mrs.begin_op(enc, &[expert_out, route_w], &[out]);
        encode_moe_prefill_weighted_sum(
            enc,
            device,
            expert_out,
            route_w,
            out,
            hidden as u32,
            top_k as u32,
            n_tokens as u32,
        )?;
        mrs.end_op(&[expert_out, route_w], &[out]);
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_prefill_weighted_sum(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    expert_out_buf: &ProtocolObject<dyn MTLBuffer>,
    route_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    hidden_rows: u32,
    top_k: u32,
    n_tokens: u32,
) -> Result<(), MetalError> {
    const SUM_TG: usize = 256;
    let weighted_sum = ensure_pipeline(device, Q4_0_MOE_TOPK_KERNEL_SRC, "moe_weighted_sum")?;
    encoder.setComputePipelineState(&weighted_sum.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(expert_out_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(route_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        let mut hr = hidden_rows;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hr as *mut u32 as *mut _).unwrap(), 4, 3);
        let mut tk = top_k;
        encoder.setBytes_length_atIndex(NonNull::new(&mut tk as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut nt = n_tokens;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nt as *mut u32 as *mut _).unwrap(), 4, 5);
    }
    let sum_elems = (n_tokens as usize) * (hidden_rows as usize);
    dispatch_counted(
        encoder,
        MTLSize {
            width: sum_elems.div_ceil(SUM_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: SUM_TG,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mul_mm_id(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    fn_name: &'static str,
    w: &ResidentWeightBuffer,
    expert_stride: u32,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    ids_buf: &ProtocolObject<dyn MTLBuffer>,
    tpe_buf: &ProtocolObject<dyn MTLBuffer>,
    n_experts: u32,
    n_tokens: u32,
    top_k: u32,
    rows: u32,
    cols: u32,
    row_bytes: u32,
    src1_per_slot: u32,
) -> Result<(), MetalError> {
    let pipeline = ensure_pipeline(device, *K_QUANT_MUL_MM_SG_KERNEL_SRC, fn_name)?;
    unsafe {
        enc.setComputePipelineState(&pipeline.0);
        enc.setBuffer_offset_atIndex(Some(&w.buffer), w.weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(x_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(ids_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(tpe_buf), 0, 4);
        for (idx, mut v) in [
            (5usize, rows),
            (6, cols),
            (7, top_k),
            (8, n_tokens),
            (9, row_bytes),
            (10, expert_stride),
            (11, src1_per_slot),
        ] {
            enc.setBytes_length_atIndex(
                NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                4,
                idx,
            );
        }
        enc.setThreadgroupMemoryLength_atIndex(8192, 0);
        dispatch_counted(
            enc,
            MTLSize {
                width: (n_tokens as usize).div_ceil(32),
                height: (rows as usize).div_ceil(64),
                depth: n_experts as usize,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

/// llama's `bc_out`, as a pipeline choice.
///
/// The `mul_mm_sg` epilogue has two arms: a fast one that `simdgroup_store`s
/// straight to `dst`, and a staging arm for the ragged edge tile that goes
/// through a 64x32 f32 threadgroup buffer. That staging buffer is what forces
/// the 8192-byte threadgroup allocation — the GEMM's own `sa`/`sb` tiles only
/// need 4096 + 2048 = 6144.
///
/// When the dst tile grid is exact (`rows % 64 == 0 && batch % 32 == 0`) the
/// staging arm is unreachable, so the `_a` pipeline compiles it out and asks
/// for 6144 bytes. On a 32 KiB per-core threadgroup budget that is 5 resident
/// threadgroups instead of 4. Same split llama makes in
/// `ggml-metal-device.cpp`: `res.smem = bc_out ? 8192 : (4096 + 2048)`.
pub(crate) const MUL_MM_SG_SMEM_BC: usize = 8192;
pub(crate) const MUL_MM_SG_SMEM_ALIGNED: usize = 4096 + 2048;

pub(crate) fn mul_mm_sg_aligned_fn(fn_name: &str) -> Option<&'static str> {
    Some(match fn_name {
        "q4_k_mul_mm_sg" => "q4_k_mul_mm_sg_a",
        "q5_k_mul_mm_sg" => "q5_k_mul_mm_sg_a",
        "q6_k_mul_mm_sg" => "q6_k_mul_mm_sg_a",
        "q8_0_mul_mm_sg" => "q8_0_mul_mm_sg_a",
        "q4_0_mul_mm_sg" => "q4_0_mul_mm_sg_a",
        "q5_0_mul_mm_sg" => "q5_0_mul_mm_sg_a",
        "iq4_xs_mul_mm_sg" => "iq4_xs_mul_mm_sg_a",
        "q4_k_mul_mm_sg_f16" => "q4_k_mul_mm_sg_f16_a",
        "q5_k_mul_mm_sg_f16" => "q5_k_mul_mm_sg_f16_a",
        "q6_k_mul_mm_sg_f16" => "q6_k_mul_mm_sg_f16_a",
        "q8_0_mul_mm_sg_f16" => "q8_0_mul_mm_sg_f16_a",
        "q4_0_mul_mm_sg_f16" => "q4_0_mul_mm_sg_f16_a",
        "q5_0_mul_mm_sg_f16" => "q5_0_mul_mm_sg_f16_a",
        "iq4_xs_mul_mm_sg_f16" => "iq4_xs_mul_mm_sg_f16_a",
        "q2_k_mul_mm_sg" => "q2_k_mul_mm_sg_a",
        "q3_k_mul_mm_sg" => "q3_k_mul_mm_sg_a",
        "q2_k_mul_mm_sg_f16" => "q2_k_mul_mm_sg_f16_a",
        "q3_k_mul_mm_sg_f16" => "q3_k_mul_mm_sg_f16_a",
        "iq4_nl_mul_mm_sg" => "iq4_nl_mul_mm_sg_a",
        "iq4_nl_mul_mm_sg_f16" => "iq4_nl_mul_mm_sg_f16_a",
        "mxfp4_mul_mm_sg" => "mxfp4_mul_mm_sg_a",
        "mxfp4_mul_mm_sg_f16" => "mxfp4_mul_mm_sg_f16_a",
        _ => return None,
    })
}

/// The half-activation sibling of a `*_mul_mm_sg` entry point
/// (`MUL_MM_SG_F16_ENTRY`). One table for the fused prefill encode and
/// its pipeline warm-up, which each kept a copy.
pub(crate) fn mul_mm_sg_f16_fn(fn_name: &str) -> Option<&'static str> {
    Some(match fn_name {
        "q4_k_mul_mm_sg" => "q4_k_mul_mm_sg_f16",
        "q5_k_mul_mm_sg" => "q5_k_mul_mm_sg_f16",
        "q6_k_mul_mm_sg" => "q6_k_mul_mm_sg_f16",
        "q8_0_mul_mm_sg" => "q8_0_mul_mm_sg_f16",
        "q4_0_mul_mm_sg" => "q4_0_mul_mm_sg_f16",
        "q5_0_mul_mm_sg" => "q5_0_mul_mm_sg_f16",
        "iq4_xs_mul_mm_sg" => "iq4_xs_mul_mm_sg_f16",
        "iq4_nl_mul_mm_sg" => "iq4_nl_mul_mm_sg_f16",
        "mxfp4_mul_mm_sg" => "mxfp4_mul_mm_sg_f16",
        "q2_k_mul_mm_sg" => "q2_k_mul_mm_sg_f16",
        "q3_k_mul_mm_sg" => "q3_k_mul_mm_sg_f16",
        _ => return None,
    })
}

/// Pipeline name + threadgroup allocation for one `mul_mm_sg` dispatch.
pub(crate) fn mul_mm_sg_variant(
    fn_name: &'static str,
    rows: usize,
    batch: usize,
) -> (&'static str, usize) {
    if rows.is_multiple_of(64) && batch.is_multiple_of(32) {
        if let Some(a) = mul_mm_sg_aligned_fn(fn_name) {
            return (a, MUL_MM_SG_SMEM_ALIGNED);
        }
    }
    (fn_name, MUL_MM_SG_SMEM_BC)
}

pub(crate) fn encode_mul_mm_sg(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    l: &MulMmSgLaunch<'_>,
    w: &ResidentWeightBuffer,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    batch_size: usize,
) -> Result<(), MetalError> {
    encode_mul_mm_sg_offset(enc, device, l, w, 0, x_buf, out_buf, batch_size)
}

/// Like [`encode_mul_mm_sg`] but `x_buf` holds half activations (llama
/// `kernel_mul_mm_*_f16`). Prefill converts f32→f16 once per RMSNorm.
pub(crate) fn encode_mul_mm_sg_f16(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    l: &MulMmSgLaunch<'_>,
    w: &ResidentWeightBuffer,
    x_h: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    batch_size: usize,
) -> Result<(), MetalError> {
    let fn_f16 = mul_mm_sg_f16_fn(l.fn_name).ok_or(MetalError::CommandFailed)?;
    let (fn_pick, smem) = mul_mm_sg_variant(fn_f16, l.rows, batch_size);
    let pipeline = ensure_pipeline(device, *K_QUANT_MUL_MM_SG_KERNEL_SRC, fn_pick)?;
    unsafe {
        enc.setComputePipelineState(&pipeline.0);
        enc.setBuffer_offset_atIndex(Some(&w.buffer), w.weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(x_h), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        for (idx, mut v) in [
            (3usize, l.rows as u32),
            (4, l.cols() as u32),
            (5, batch_size as u32),
            (6, l.row_bytes as u32),
        ] {
            enc.setBytes_length_atIndex(
                NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                4,
                idx,
            );
        }
        enc.setThreadgroupMemoryLength_atIndex(smem, 0);
        dispatch_counted(
            enc,
            MTLSize {
                width: batch_size.div_ceil(32),
                height: l.rows.div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mul_mm_sg_offset(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    l: &MulMmSgLaunch<'_>,
    w: &ResidentWeightBuffer,
    weight_byte_offset: usize,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    batch_size: usize,
) -> Result<(), MetalError> {
    encode_mul_mm_sg_offset_ex(
        enc,
        device,
        l,
        w,
        weight_byte_offset,
        x_buf,
        0,
        out_buf,
        0,
        batch_size,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mul_mm_sg_offset_ex(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    l: &MulMmSgLaunch<'_>,
    w: &ResidentWeightBuffer,
    weight_byte_offset: usize,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    x_byte_offset: usize,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    out_byte_offset: usize,
    batch_size: usize,
) -> Result<(), MetalError> {
    let (fn_pick, smem) = mul_mm_sg_variant(l.fn_name, l.rows, batch_size);
    let pipeline = ensure_pipeline(device, *K_QUANT_MUL_MM_SG_KERNEL_SRC, fn_pick)?;
    unsafe {
        enc.setComputePipelineState(&pipeline.0);
        enc.setBuffer_offset_atIndex(Some(&w.buffer), w.weight_offset + weight_byte_offset, 0);
        enc.setBuffer_offset_atIndex(Some(x_buf), x_byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), out_byte_offset, 2);
        for (idx, mut v) in [
            (3usize, l.rows as u32),
            (4, l.cols() as u32),
            (5, batch_size as u32),
            (6, l.row_bytes as u32),
        ] {
            enc.setBytes_length_atIndex(
                NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                4,
                idx,
            );
        }
        enc.setThreadgroupMemoryLength_atIndex(smem, 0);
        dispatch_counted(
            enc,
            MTLSize {
                width: batch_size.div_ceil(32),
                height: l.rows.div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

/// Whole dense FFN for a batch of positions in **one** command buffer:
/// gate GEMM, up GEMM, the activation, then the down GEMM, with the two
/// intermediates staying in device memory.
///
/// Run as three separate `launch_*_mul_mm_sg` calls instead, the same
/// work costs three command-buffer round trips per layer plus four host
/// copies of tensors that are `batch x ffn_dim` -- 29 MB apiece at batch
/// 512 on an 8B model -- and the activation runs on the CPU. Profiling
/// put ~20-25% of every GEMM call in that copying. llama.cpp never pays
/// it because it submits the whole graph, not one node at a time.
///
/// Prefill evidence: set `FRINK_METAL_MM_TIMING=1` — this path and
/// [`launch_k_quant_mul_mm_sg`] accumulate setup/gpu/readback totals
/// (printed every 224 calls). For a fuller one-CB-per-layer prefill see
/// [`crate::attn::launch_prefill_dense_layer`].
///
/// `x_batch` is `[batch, hidden]`, the result `[batch, down.rows]`.
pub fn launch_dense_ffn_swiglu_batch(
    gate: &MulMmSgLaunch<'_>,
    up: &MulMmSgLaunch<'_>,
    down: &MulMmSgLaunch<'_>,
    x_batch: &[f32],
    batch_size: usize,
    gelu: bool,
) -> Result<Vec<f32>, MetalError> {
    if batch_size == 0 || gate.rows == 0 || down.rows == 0 {
        return Ok(vec![0.0; batch_size * down.rows]);
    }
    let hidden = gate.cols();
    if hidden == 0 || up.cols() != hidden || down.cols() != gate.rows || up.rows != gate.rows {
        return Err(MetalError::CommandFailed);
    }
    if x_batch.len() != batch_size * hidden {
        return Err(MetalError::CommandFailed);
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;
    let timing = std::env::var_os("FRINK_METAL_MM_TIMING").is_some();
    let t_setup = std::time::Instant::now();

    // Shared scratch + memcpy (no per-call `newBufferWithBytes` alloc).
    let x_elems = batch_size * hidden;
    let x_scratch = borrow_scratch(device, x_elems * 4, MTLResourceOptions::StorageModeShared)?;
    let x_buf = x_scratch.get();
    unsafe {
        std::ptr::copy_nonoverlapping(
            x_batch.as_ptr() as *const u8,
            x_buf.contents().as_ptr() as *mut u8,
            x_elems * 4,
        );
    }

    let gate_w = resident_weight_buffer(device, gate.weights)?;
    let up_w = resident_weight_buffer(device, up.weights)?;
    let down_w = resident_weight_buffer(device, down.weights)?;

    let ffn_elems = batch_size * gate.rows;
    // Gate / up / activation never leave the GPU, so they can be Private.
    // All four come from the scratch pool: at batch 512 on an 8B these
    // are 28 MiB apiece and allocating them per layer dominated `setup`.
    let mid = |n: usize| borrow_scratch(device, n * 4, MTLResourceOptions::StorageModePrivate);
    let gate_buf = mid(ffn_elems)?;
    let up_buf = mid(ffn_elems)?;
    let act_buf = mid(ffn_elems)?;
    let out_elems = batch_size * down.rows;
    let out_scratch = borrow_scratch(device, out_elems * 4, MTLResourceOptions::StorageModeShared)?;
    let (gate_buf, up_buf, act_buf, out_buf) = (
        gate_buf.get(),
        up_buf.get(),
        act_buf.get(),
        out_scratch.get(),
    );
    let setup_us = t_setup.elapsed().as_micros();

    // Decode-stack pattern: Concurrent encode + barriers around RAW edges
    // so gate∥up can overlap like llama `ggml_metal_op`.
    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let enc = compute_encoder_concurrent(&cmd_buf)?;
    encode_mul_mm_sg(&enc, device, gate, &gate_w, x_buf, gate_buf, batch_size)?;
    encode_mul_mm_sg(&enc, device, up, &up_w, x_buf, up_buf, batch_size)?;
    memory_barrier_buffers(&enc);
    if gelu {
        crate::elem::encode_gelu_mul(&enc, device, gate_buf, up_buf, act_buf, ffn_elems as u32)?;
    } else {
        crate::elem::encode_silu_mul(&enc, device, gate_buf, up_buf, act_buf, ffn_elems as u32)?;
    }
    memory_barrier_buffers(&enc);
    encode_mul_mm_sg(&enc, device, down, &down_w, act_buf, out_buf, batch_size)?;
    enc.endEncoding();
    let t_gpu = std::time::Instant::now();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    let gpu_us = t_gpu.elapsed().as_micros();

    let t_read = std::time::Instant::now();
    let ptr = out_buf.contents().as_ptr() as *const f32;
    let out = unsafe { std::slice::from_raw_parts(ptr, out_elems) }.to_vec();
    if timing {
        mm_timing_add(setup_us, gpu_us, t_read.elapsed().as_micros());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_k_quant_mul_mm_sg(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
    fn_name: &'static str,
    block_bytes: usize,
    block_elems: usize,
    fold: Option<&crate::hadamard::FoldPlan<'_>>,
) -> Result<Vec<f32>, MetalError> {
    if batch_size == 0 || rows == 0 {
        return Ok(vec![0.0; batch_size * rows]);
    }
    let n_blocks_per_row = row_bytes / block_bytes;
    let cols = n_blocks_per_row * block_elems;
    if cols == 0 {
        return Err(MetalError::CommandFailed);
    }
    assert_eq!(weights.len(), rows * row_bytes);
    assert_eq!(x_batch.len(), batch_size * cols);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;
    let timing = std::env::var_os("FRINK_METAL_MM_TIMING").is_some();
    let t_setup = std::time::Instant::now();

    // `newBufferWithBytes` copies, so the extra `to_vec` this used to do
    // was a second full copy of the activation batch on every call --
    // 8 MB per FFN matrix at batch 512, ~224 calls per 8B prefill.
    let x_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(x_batch.as_ptr() as *mut std::ffi::c_void).unwrap(),
            std::mem::size_of_val(x_batch),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;

    let weights_buf = resident_weight_buffer(device, weights)?;
    let out_elems = batch_size * rows;
    let out_scratch = borrow_scratch(device, out_elems * 4, MTLResourceOptions::StorageModeShared)?;
    let out_buf = out_scratch.get();

    let (fn_pick, smem) = mul_mm_sg_variant(fn_name, rows, batch_size);
    let pipeline = ensure_pipeline(device, *K_QUANT_MUL_MM_SG_KERNEL_SRC, fn_pick)?;
    let setup_us = t_setup.elapsed().as_micros();

    let clock = crate::timing::SubmitClock::start();
    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let enc = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    // The rotation first, per row, on the batch buffer this GEMM is
    // about to read. `x_buf` is this call's own upload, so rewriting it
    // in place reaches nobody else.
    if let Some(plan) = fold {
        let signs_buf = match plan.signs {
            None => None,
            Some(signs) => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        signs.as_ptr() as *const u8,
                        std::mem::size_of_val(signs),
                    )
                };
                Some(resident_weight_buffer(device, bytes)?)
            }
        };
        crate::hadamard::encode_fold_rows(
            &enc,
            device,
            &x_buf,
            cols,
            batch_size,
            plan,
            signs_buf.as_ref().map(|b| &*b.buffer),
        )?;
    }

    unsafe {
        enc.setComputePipelineState(&pipeline.0);
        enc.setBuffer_offset_atIndex(Some(&weights_buf.buffer), weights_buf.weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        for (idx, mut v) in [
            (3usize, rows as u32),
            (4, cols as u32),
            (5, batch_size as u32),
            (6, row_bytes as u32),
        ] {
            enc.setBytes_length_atIndex(
                NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                4,
                idx,
            );
        }
        // 4096 B for the dequantized A tile + 2048 B for B; the
        // partial-tile store path reuses the same allocation as a
        // 64x32 f32 staging buffer, which needs 8192. `mul_mm_sg_variant`
        // picks the exact-tile pipeline (no staging arm) when it can.
        enc.setThreadgroupMemoryLength_atIndex(smem, 0);

        let grid = MTLSize {
            width: batch_size.div_ceil(32),
            height: rows.div_ceil(64),
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        dispatch_counted(&enc, grid, tg);
    }
    enc.endEncoding();
    let t_gpu = std::time::Instant::now();
    // Timed like every other submission: the prefill GEMM is where a
    // batch spends its GPU, and an untimed command buffer reads as
    // host time in the ledger (the confusion issue #149 was about).
    crate::timing::commit_wait_note(&cmd_buf, "gemm-prefill", 32, clock);
    let gpu_us = t_gpu.elapsed().as_micros();

    let t_read = std::time::Instant::now();
    let ptr = out_buf.contents().as_ptr() as *const f32;
    let out = unsafe { std::slice::from_raw_parts(ptr, out_elems) }.to_vec();
    if timing {
        // Where prefill time actually goes, per matrix. Set
        // FRINK_METAL_MM_TIMING=1 and read the totals at the end.
        mm_timing_add(setup_us, gpu_us, t_read.elapsed().as_micros());
    }
    Ok(out)
}

/// Launches Q4_K multi-activation matmul (see [`Q4_K_MUL_MM_KERNEL_SRC`]).
///
/// Dequant matches [`Q4_K_MATVEC_KERNEL_SRC`] /
/// `frink_quant::dot_q4_k_f32`. `x_batch` is `[batch, cols]`; returns
/// `[batch, rows]`. One threadgroup per weight row, 64 threads striding
/// over the row's Q4_K blocks.
pub fn launch_q4_k_mul_mm(
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    let n_blocks_per_row = row_bytes / 144;
    let cols = n_blocks_per_row * 256;
    assert_eq!(weights.len(), rows * row_bytes);
    assert_eq!(x_batch.len(), batch_size * cols);

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let mut x_owned = x_batch.to_vec();
    let x_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(x_owned.as_mut_ptr() as *mut _).unwrap(),
            x_owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;

    let weights_buf = resident_weight_buffer(device, weights)?;
    let out_elems = batch_size * rows;
    let out_buf = device
        .newBufferWithLength_options(out_elems * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;

    let pipeline = ensure_pipeline(device, Q4_K_MUL_MM_KERNEL_SRC, "q4_k_mul_mm")?;

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let enc = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;

    let tg = 64u32;
    unsafe {
        enc.setComputePipelineState(&pipeline.0);
        enc.setBuffer_offset_atIndex(Some(&weights_buf.buffer), weights_buf.weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
        let mut row_bytes_u32 = row_bytes as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut row_bytes_u32 as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut n_blocks_u32 = n_blocks_per_row as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut n_blocks_u32 as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        let mut n_rows_u32 = rows as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut n_rows_u32 as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut batch_u32 = batch_size as u32;
        enc.setBytes_length_atIndex(
            NonNull::new(&mut batch_u32 as *mut u32 as *mut _).unwrap(),
            4,
            6,
        );
        enc.setThreadgroupMemoryLength_atIndex((tg as usize) * 4, 0);
    }

    dispatch_counted(
        &enc,
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let out_ptr = out_buf.contents();
    let out_slice =
        unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, out_elems) };
    Ok(out_slice.to_vec())
}

/// ggml-metal `kernel_mul_mv_q5_K_f32` port: `N_R0=1` row per simdgroup,
/// `NSG=2` simdgroups per TG (2 rows / 64 threads). Register-local
/// `yl`/`yh` activation packs with Q5_K's 5th-bit `qh` plane — same
/// dequant identity as `frink_quant::dot_q5_k_f32_scalar`. Host
/// dispatches `ceil(n_rows/2)` threadgroups of 64 threads.
///
/// `N_R0` stays at 1 (not 2 like Q4_K/Q6_K): ggml-metal reduced it after
/// a real register-spill regression (`llama.cpp` #20399).
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// `launch_q5_k_matvec_matches_cpu_reference`).
pub const Q5_K_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q5_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid_tg [[thread_position_in_threadgroup]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 1;
    constexpr short NW = 32;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort tiisg = tid_tg % NW;
    const ushort sgitg = tid_tg / NW;

    const short tid = tiisg / 4;
    const short ix = tiisg % 4;
    const short iq = tid / 4;
    const short ir = tid % 4;

    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uchar hm1 = uchar(1u << (2 * iq));
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    const int first_row = int(tgpig * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);

    if (first_row >= int(n_rows)) {
        return;
    }

    float sumf = 0.0f;
    float yl[16];
    float yh[16];

    uint16_t sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    device const float* y1 = x + ix * 256 + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uchar* block0 =
            weights + (size_t)first_row * row_bytes + (size_t)i * 176u;
        device const uchar* q1 = block0 + 48 + q_offset;
        device const uchar* qh = block0 + 16 + l0;
        device const half* dh = (device const half*)(block0);
        device const uint16_t* a =
            (device const uint16_t*)(block0 + 4) + iq;

        device const float* y2 = y1 + 128;
        float4 sumy = float4(0.0f);
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];
            sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32];
            sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l + 0];
            sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32];
            sumy[3] += yh[l + 8];
        }

        device const uchar* q2 = q1 + 64;

        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        float4 acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f);
        for (short l = 0; l < 8; ++l) {
            uchar h = qh[l];
            acc1[0] += yl[l + 0] * float(q1[l] & 0x0F);
            acc1[1] += yl[l + 8] * float(q1[l] & 0xF0);
            acc1[2] += yh[l + 0] * float(q2[l] & 0x0F);
            acc1[3] += yh[l + 8] * float(q2[l] & 0xF0);
            acc2[0] += (h & hm1) ? yl[l + 0] : 0.0f;
            acc2[1] += (h & hm2) ? yl[l + 8] : 0.0f;
            acc2[2] += (h & hm3) ? yh[l + 0] : 0.0f;
            acc2[3] += (h & hm4) ? yh[l + 8] : 0.0f;
        }

        sumf += float(dh[0])
                * (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0])
                    + float(sc8[1]) * (acc1[1] / 16.0f + 16.0f * acc2[1])
                    + float(sc8[4]) * (acc1[2] + 16.0f * acc2[2])
                    + float(sc8[5]) * (acc1[3] / 16.0f + 16.0f * acc2[3]))
            - float(dh[1])
                * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3])
                    + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

        y1 += 4 * 256;
    }

    float sum_all = simd_sum(sumf);
    if (tiisg == 0 && first_row < int(n_rows)) {
        out[first_row] = sum_all;
    }
}
"#;

/// Launches the Q5_K matvec kernel. Verified on a real Apple M2 Pro GPU
/// -- see `Q5_K_MATVEC_KERNEL_SRC`'s doc comment.
pub fn launch_q5_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        Q5_K_MATVEC_KERNEL_SRC,
        "q5_k_matvec",
        176,
        256,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// ggml-metal `kernel_mul_mv_q6_K_f32` port: `N_R0=2` rows per simdgroup,
/// `NSG=2` simdgroups per TG (4 rows / 64 threads). Register-local `yl`
/// packs; signed int8 sub-block scales match
/// `frink_quant::dot_q6_k_f32_scalar`. Host dispatches `ceil(n_rows/4)`
/// threadgroups of 64 threads.
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference exactly (see
/// `launch_q6_k_matvec_matches_cpu_reference`, which specifically
/// includes a negative-scale byte in its test data).
pub const Q6_K_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q6_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int first_row = int(tgpig * NSG + sgitg) * nr0;
    const int nb = int(n_blocks_per_row);

    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8; // 0 or 1
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;

    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    float sumf[2] = {0.0f, 0.0f};
    float yl[16];

    for (int i = ix; i < nb; i += 2) {
        device const uchar* block0 =
            weights + (size_t)first_row * row_bytes + (size_t)i * 210u;
        device const uchar* q1 = block0 + q_offset_l;
        device const uchar* q2 = q1 + 32;
        device const uchar* qh = block0 + 128 + q_offset_h;
        device const char* sc = (device const char*)(block0 + 192 + is);
        device const half* dh = (device const half*)(block0 + 208);

        device const float* y = x + i * 256 + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l + 0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = float4(0.0f);

            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0]
                    * float(int((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1]
                    * float(int((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2]
                    * float(int((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3]
                    * float(int((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += float(dh[0])
                * (sums[0] * float(sc[0]) + sums[1] * float(sc[2])
                    + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));

            q1 += row_bytes;
            q2 += row_bytes;
            qh += row_bytes;
            sc += row_bytes;
            dh += row_bytes / 2;
        }
    }

    for (int row = 0; row < nr0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < int(n_rows)) {
            out[first_row + row] = sum_all;
        }
    }
}
"#;

/// Launches the Q6_K matvec kernel. Verified on a real Apple M2 Pro GPU
/// -- see `Q6_K_MATVEC_KERNEL_SRC`'s doc comment.
/// The Q5_0 matvec, and the wrapper whose absence made `Q5_0` a
/// half-supported kind.
///
/// `Q5_0_MATVEC_KERNEL_SRC` and the `matvec_launch_meta` row landed
/// together, and both the capability tables were widened on the strength
/// of them. But `apply_gpu`'s single-matvec decode path dispatches
/// through a per-kind `launch_*_matvec` FUNCTION, and there was no Q5_0
/// one -- so `apply_gpu_batch`, `apply_gpu_multi` and the fused FFN all
/// ran Q5_0 on the GPU while single-token decode silently fell to the
/// CPU. That is exactly the mixed CPU/GPU split the widening was
/// supposed to close.
///
/// Derived from `matvec_launch_meta` rather than restating the
/// constants. The six wrappers above each hard-code the block size and
/// element count that the meta table already holds, which is a seventh
/// copy of the same data and is why one kind could be present in the
/// table and absent here without anything noticing.
pub fn launch_q5_0_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    let (src, name, block_bytes, block_elems, _) =
        matvec_launch_meta("Q5_0").ok_or(MetalError::CommandFailed)?;
    launch_matvec(
        src,
        name,
        block_bytes,
        block_elems,
        weights,
        x,
        rows,
        row_bytes,
    )
}

pub fn launch_q6_k_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        Q6_K_MATVEC_KERNEL_SRC,
        "q6_k_matvec",
        210,
        256,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// ggml-metal `kernel_mul_mv_iq4_xs_f32` port: `N_R0=2` rows per
/// simdgroup, `NSG=2` simdgroups per TG (4 rows / 64 threads). IQ4_XS
/// blocks are 136 bytes / 256 elements: f16 super-scale `d`, 16 bits of
/// high scale bits, 4 bytes of low scale nibbles, then 128 bytes of
/// 4-bit indices into the shared IQ4_NL non-linear codebook (loaded
/// into 32 floats of threadgroup memory, one copy per 16 lanes). Each
/// lane owns 8 consecutive bytes of `qs` (16 elements: low nibbles →
/// elems j, high nibbles → elems j+16 of the 32-elem sub-block), with
/// odd/even lanes-of-16 walking odd/even blocks. Same dequant identity
/// as `frink_quant::dot_iq4_xs_f32`.
///
/// Verified: compiled by the system Metal compiler and executed on a
/// real Apple M2 Pro GPU, matching the CPU reference (see
/// `launch_iq4_xs_matvec_matches_cpu_reference`).
pub const IQ4_XS_MATVEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant float kvalues_iq4nl_f[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

kernel void iq4_xs_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& row_bytes [[buffer(3)]],
    constant uint& n_blocks_per_row [[buffer(4)]],
    constant uint& n_rows [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* shmem_f32 [[threadgroup(0)]]
) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;

    const int nb = int(n_blocks_per_row);
    const int first_row = int(tgpig * NSG + sgitg) * nr0;

    const short ix = short(tiisg) / 16; // 0/1: block parity
    const short it = short(tiisg) % 16;
    const short ib = it / 2;            // 0..7: 32-elem sub-block
    const short il = it % 2;            // 0/1: 8-byte half of qs sub-block

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[nr0] = {0.0f, 0.0f};

    device const float* yb = x + ix * 256 + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uchar* q8 = (thread const uchar*)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {
        device const float4* y4 = (device const float4*)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < nr0; ++row) {
            device const uchar* blk = weights
                + (size_t)(first_row + row) * row_bytes + (size_t)ibl * 136u;
            device const uint32_t* q4 =
                (device const uint32_t*)(blk + 8u + 16u * ib + 8u * il);

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = float4(shmem_f32[q8[0]], shmem_f32[q8[1]],
                         shmem_f32[q8[2]], shmem_f32[q8[3]]);
            qf2 = float4(shmem_f32[q8[4]], shmem_f32[q8[5]],
                         shmem_f32[q8[6]], shmem_f32[q8[7]]);
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = float4(shmem_f32[q8[0]], shmem_f32[q8[1]],
                         shmem_f32[q8[2]], shmem_f32[q8[3]]);
            qf2 = float4(shmem_f32[q8[4]], shmem_f32[q8[5]],
                         shmem_f32[q8[6]], shmem_f32[q8[7]]);
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const ushort scales_h = *(device const ushort*)(blk + 2);
            const int ls = int(((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf)
                | (((scales_h >> (2 * ib)) & 3) << 4)) - 32;
            sumf[row] += float(*(device const half*)blk) * float(ls)
                * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * 256;
    }

    for (short row = 0; row < nr0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < int(n_rows)) {
            out[first_row + row] = sum_all;
        }
    }
}
"#;

/// Launches the IQ4_XS matvec kernel. Verified on a real Apple M2 Pro
/// GPU -- see `IQ4_XS_MATVEC_KERNEL_SRC`'s doc comment.
pub fn launch_iq4_xs_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        IQ4_XS_MATVEC_KERNEL_SRC,
        "iq4_xs_matvec",
        136,
        256,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// Launches the PTQ1_0 matvec kernel (`crate::ternary`).
pub fn launch_ptq1_0_matvec(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    launch_matvec(
        crate::ternary::ptq1_0_matvec_src(),
        "ptq1_0_matvec",
        28,
        128,
        weights,
        x,
        rows,
        row_bytes,
    )
}

/// Kernel metadata for building a [`MatvecLaunch`] from a GGML quant
/// tag name used by `WeightMatrix` (Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K).
/// The fifth field is rows-per-threadgroup (`1` for legacy one-row
/// kernels; `2` for Q5_K/Q8_0; `4` for Q4_K/Q6_K/IQ4_XS; `8` for Q4_0).
/// Every kind [`matvec_launch_meta`] answers for. `rows_per_threadgroup`
/// walks this list to find a kernel by entry-point name; it used to
/// carry its own copy of the names, and the copy lacked the newest kind
/// (PTQ1_0), so that kernel was dispatched at one row per threadgroup
/// and answered zeros for every row but the first group's. The test
/// below holds the two together.
pub const MATVEC_KINDS: &[&str] = &[
    "F32", "Q8_0", "Q4_0", "Q5_0", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "PTQ1_0", "Q2_K", "Q3_K",
    "IQ4_NL", "MXFP4",
];

pub fn matvec_launch_meta(kind: &str) -> Option<(&'static str, &'static str, usize, usize, usize)> {
    match kind {
        "F32" => Some((F32_MATVEC_KERNEL_SRC, "f32_matvec", 4, 1, 1)),
        "Q8_0" => Some((Q8_0_MATVEC_KERNEL_SRC, "q8_0_matvec", 34, 32, 2)),
        "Q4_0" => Some((Q4_0_MATVEC_KERNEL_SRC, "q4_0_matvec", 18, 32, 8)),
        "Q5_0" => Some((Q5_0_MATVEC_KERNEL_SRC, "q5_0_matvec", 22, 32, 8)),
        "Q4_K" => Some((Q4_K_MATVEC_KERNEL_SRC, "q4_k_matvec", 144, 256, 4)),
        "Q5_K" => Some((Q5_K_MATVEC_KERNEL_SRC, "q5_k_matvec", 176, 256, 2)),
        "Q6_K" => Some((Q6_K_MATVEC_KERNEL_SRC, "q6_k_matvec", 210, 256, 4)),
        "IQ4_XS" => Some((IQ4_XS_MATVEC_KERNEL_SRC, "iq4_xs_matvec", 136, 256, 4)),
        "PTQ1_0" => Some((
            crate::ternary::ptq1_0_matvec_src(),
            "ptq1_0_matvec",
            28,
            128,
            4,
        )),
        "Q2_K" => Some((
            crate::q2_k::q2_k_matvec_src(),
            "q2_k_matvec",
            84,
            256,
            crate::q2_k::ROWS_PER_TG,
        )),
        "Q3_K" => Some((
            crate::q3_k::q3_k_matvec_src(),
            "q3_k_matvec",
            110,
            256,
            crate::q3_k::ROWS_PER_TG,
        )),
        "IQ4_NL" => Some((
            crate::codebook4::codebook4_matvec_src(),
            "iq4_nl_matvec",
            18,
            32,
            4,
        )),
        "MXFP4" => Some((
            crate::codebook4::codebook4_matvec_src(),
            "mxfp4_matvec",
            17,
            32,
            4,
        )),
        _ => None,
    }
}

/// One process-wide `MTLDevice` + `MTLCommandQueue`, created once and
/// reused for every kernel launch (mirrors
/// `frink_cuda::gpu::shared_device`'s `Mutex<Option<Arc<...>>>`
/// pattern exactly, including the reason for `Mutex` over `OnceLock`:
/// this project's pinned minimum rustc predates
/// `OnceLock::get_or_try_init`).
pub(crate) struct SharedMetal {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

// SAFETY: Apple's Metal documentation states that `MTLDevice` and
// `MTLCommandQueue` objects are safe to use from multiple threads
// simultaneously (unlike `MTLCommandBuffer`/`MTLComputeCommandEncoder`,
// which are explicitly documented as requiring single-threaded,
// single-use access -- and which this module never caches, only ever
// creates fresh per call, exactly because of that distinction).
// `objc2-metal`'s `Retained<ProtocolObject<dyn T>>` is unconditionally
// `!Send`/`!Sync` regardless of `T` (it wraps a `NonNull`, which is
// `!Send`/`!Sync` no matter what it points to, forcing every crate that
// wraps an Objective-C object to explicitly assert thread-safety rather
// than getting it for free) -- so this is a deliberate, narrow opt-in
// for exactly the two object kinds Apple documents as safe to share,
// not a blanket assertion about arbitrary Objective-C objects.
unsafe impl Send for SharedMetal {}
unsafe impl Sync for SharedMetal {}

static SHARED_METAL: Mutex<Option<Arc<SharedMetal>>> = Mutex::new(None);

pub(crate) fn shared_metal() -> Result<Arc<SharedMetal>, MetalError> {
    let mut guard = SHARED_METAL.lock().unwrap();
    if let Some(shared) = guard.as_ref() {
        return Ok(shared.clone());
    }
    let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
    let queue = device.newCommandQueue().ok_or(MetalError::CommandFailed)?;
    let shared = Arc::new(SharedMetal { device, queue });
    *guard = Some(shared.clone());
    Ok(shared)
}

/// One compiled `MTLComputePipelineState`, cached by kernel function
/// name so a given kernel is only ever compiled once per process
/// (mirrors `frink_cuda::gpu::ensure_module_loaded`'s
/// compile-once/reuse behavior for NVRTC modules).
pub(crate) struct CachedPipeline(pub(crate) Retained<ProtocolObject<dyn MTLComputePipelineState>>);

// SAFETY: same justification as `SharedMetal` above -- Apple documents
// `MTLComputePipelineState` (like `MTLDevice`/`MTLCommandQueue`) as
// safe to use from multiple threads simultaneously once created.
unsafe impl Send for CachedPipeline {}
unsafe impl Sync for CachedPipeline {}

static PIPELINE_CACHE: Mutex<Option<HashMap<&'static str, Arc<CachedPipeline>>>> = Mutex::new(None);

thread_local! {
    /// Hot-path mirror of [`PIPELINE_CACHE`]: MoE/dense encode hits this
    /// without taking the process Mutex (~100+ lookups/token on OLMoE).
    static TL_PIPELINE_CACHE: RefCell<HashMap<&'static str, Arc<CachedPipeline>>> =
        RefCell::new(HashMap::new());
}

pub(crate) fn ensure_pipeline(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    kernel_src: &'static str,
    fn_name: &'static str,
) -> Result<Arc<CachedPipeline>, MetalError> {
    if let Some(cached) = TL_PIPELINE_CACHE.with(|c| c.borrow().get(fn_name).cloned()) {
        return Ok(cached);
    }
    let cached = {
        let mut guard = PIPELINE_CACHE.lock().unwrap();
        let cache = guard.get_or_insert_with(HashMap::new);
        if let Some(cached) = cache.get(fn_name) {
            cached.clone()
        } else {
            let src = NSString::from_str(kernel_src);
            let library = device
                .newLibraryWithSource_options_error(&src, None)
                .map_err(|e| MetalError::CompileFailed(e.to_string()))?;
            let func_name = NSString::from_str(fn_name);
            let function = library
                .newFunctionWithName(&func_name)
                .ok_or(MetalError::FunctionNotFound(fn_name))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| MetalError::PipelineFailed(e.to_string()))?;
            let cached = Arc::new(CachedPipeline(pipeline));
            cache.insert(fn_name, cached.clone());
            cached
        }
    };
    TL_PIPELINE_CACHE.with(|c| {
        c.borrow_mut().insert(fn_name, cached.clone());
    });
    Ok(cached)
}

/// Compile-once helper for prefill [`encode_mul_mm_sg`] kernels. Used by
/// [`crate::attn::MetalGraph::warm_prefill_pipelines`] to front-load Metal
/// pipeline creation before the first dense-prefill command buffer.
pub(crate) fn warm_mul_mm_sg_pipeline(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    fn_name: &'static str,
) -> Result<Arc<CachedPipeline>, MetalError> {
    ensure_pipeline(device, *K_QUANT_MUL_MM_SG_KERNEL_SRC, fn_name)
}

/// Process-wide cache of quantized weight `MTLBuffer`s, looked up by
/// the host slice's base pointer and length and served only when the
/// entry can still prove it holds those bytes -- see
/// [`crate::resident_cache`] for why a lookup key is not an identity.
pub(crate) struct ResidentWeightBuffer {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Byte offset of the weight bytes within `buffer`. Non-zero only on
    /// the zero-copy (`BytesNoCopy`) path, where the buffer wraps a whole
    /// registered GGUF mmap and the tensor starts at this file offset.
    /// Copy-path buffers always have offset 0. Every kernel that binds
    /// this buffer at argument slot 0 must bind it at this offset.
    pub(crate) weight_offset: usize,
    nbytes: usize,
    /// Sampled fingerprint of the host bytes this buffer was built from.
    /// `(pointer, length)` is *not* an identity: free a tensor and the
    /// allocator can hand the same address and length to a different one,
    /// after which the cache would serve the old contents. Long-lived
    /// mmap-backed weights never hit that, but owned `WeightBytes` and
    /// short-lived test fixtures do. Checked on every cache hit.
    fingerprint: u64,
    /// True when this entry aliases a registered mmap (see the lazy
    /// fingerprint note in `resident_weight_buffer`).
    mmap_backed: bool,
    /// Keeps the registered mmap MTLBuffer entry alive for NoCopy aliases.
    _keepalive: Option<Arc<ResidentMmapFile>>,
}

/// Cheap content fingerprint: FNV-1a over at most 64 sampled 8-byte
/// words spread across the slice, plus the length. Hashing a 200 MB
/// tensor on every matmul would cost more than the upload it is
/// protecting; sampling makes an accidental collision between two
/// distinct tensors of equal length vanishingly unlikely at constant
/// cost.
fn weight_fingerprint(weights: &[u8]) -> u64 {
    const SAMPLES: usize = 64;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x100_0000_01b3);
    };
    mix(weights.len() as u64);
    let stride = (weights.len() / SAMPLES).max(1);
    let mut off = 0usize;
    while off < weights.len() {
        let end = (off + 8).min(weights.len());
        let mut word = [0u8; 8];
        word[..end - off].copy_from_slice(&weights[off..end]);
        mix(u64::from_le_bytes(word));
        off += stride;
    }
    h
}

// SAFETY: same justification as `SharedMetal` -- `MTLBuffer` created
// once and only read by compute kernels is safe to share across threads
// that each build their own command buffer/encoder.
unsafe impl Send for ResidentWeightBuffer {}
unsafe impl Sync for ResidentWeightBuffer {}

impl Resident for ResidentWeightBuffer {
    /// One proof per way this entry can have been built.
    ///
    /// A `BytesNoCopy` alias holds an `Arc<ResidentMmapFile>`, and that
    /// keepalive is what makes its address an identity: the mapping
    /// cannot be unmapped while the entry lives, so the kernel cannot
    /// reissue the range to a second file. There is also nothing to
    /// compare, because the device bytes ARE the host bytes.
    ///
    /// A copied entry owns its bytes, and the host allocation behind
    /// the address can be freed and reissued, so it carries a
    /// fingerprint of what it was built from. Sampled rather than
    /// exhaustive: the copy path is what an expert-streaming lease
    /// takes, where comparing a whole matrix per matvec would cost what
    /// the upload it skips costs. That is a weaker proof than
    /// [`ResidentF32Buffer`]'s, and saying so is the honest version.
    fn still_holds(&self, host: &[u8]) -> bool {
        self.mmap_backed || self.fingerprint == weight_fingerprint(host)
    }

    fn resident_bytes(&self) -> usize {
        self.nbytes
    }
}

type WeightCacheMap = HashMap<HostKey, Arc<ResidentWeightBuffer>>;

static WEIGHT_CACHE: Mutex<Option<WeightCacheMap>> = Mutex::new(None);

thread_local! {
    static TL_WEIGHT_CACHE: RefCell<WeightCacheMap> = RefCell::new(HashMap::new());
}

pub(crate) fn resident_weight_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    weights: &[u8],
) -> Result<Arc<ResidentWeightBuffer>, MetalError> {
    get_or_build(&WEIGHT_CACHE, &TL_WEIGHT_CACHE, weights, || {
        build_resident_weight_buffer(device, weights)
    })
}

/// VM page size used for `BytesNoCopy` alignment. Apple Silicon uses
/// 16 KiB pages; Intel Macs use 4 KiB. Over-aligning is never wrong (a
/// 16 KiB boundary is also a 4 KiB boundary); under-aligning makes
/// `newBufferWithBytesNoCopy` return nil, which we handle by falling
/// back to a copy, so a mismatch degrades gracefully rather than crashing.
#[cfg(target_arch = "aarch64")]
const METAL_VM_PAGE: usize = 16384;
#[cfg(not(target_arch = "aarch64"))]
const METAL_VM_PAGE: usize = 4096;

/// One MTLBuffer wrapping an entire GGUF mmap (page-aligned file image).
/// Tensor slices reuse this buffer at `range.start` offsets — same
/// residency model as llama.cpp's mmap-backed Metal buffers.
struct ResidentMmapFile {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Keeps the underlying mapping alive for as long as any weight
    /// buffer aliases it via `BytesNoCopy`. `None` for a region whose
    /// memory `buffer` itself owns (an expert slot slab,
    /// [`crate::expert_slab`]), where retaining the buffer is the
    /// keepalive.
    _mmap: Option<Arc<memmap2::Mmap>>,
    base_ptr: usize,
    len: usize,
}

// SAFETY: read-only MTLBuffer + immutable mmap, shared across threads
// that each build their own command buffer (same as ResidentWeightBuffer).
unsafe impl Send for ResidentMmapFile {}
unsafe impl Sync for ResidentMmapFile {}

type MmapFileCache = HashMap<usize, Arc<ResidentMmapFile>>;
static MMAP_FILE_CACHE: Mutex<Option<MmapFileCache>> = Mutex::new(None);

/// Register a GGUF mmap so later [`resident_weight_buffer`] calls whose
/// slices sit inside it can alias the file with `BytesNoCopy` instead of
/// copying. Safe to call multiple times for the same `Arc` (idempotent).
/// Call from the loader when taking [`WeightBytes::Mapped`] views.
pub fn register_weight_mmap(mmap: Arc<memmap2::Mmap>) {
    if mmap.is_empty() {
        return;
    }
    let key = mmap.as_ptr() as usize;
    {
        let guard = MMAP_FILE_CACHE.lock().unwrap();
        if let Some(cache) = guard.as_ref() {
            if cache.contains_key(&key) {
                return;
            }
        }
    }
    let Ok(shared) = shared_metal() else {
        return;
    };
    let device = &shared.device;
    let base = mmap.as_ptr() as usize;
    // mmap returns a page-aligned pointer; length must be a page multiple
    // for BytesNoCopy — round the file length up within the mapping's
    // VM region (the OS maps whole pages for the file).
    let buf_len = mmap.len().div_ceil(METAL_VM_PAGE) * METAL_VM_PAGE;
    // SAFETY: `mmap.as_ptr()` is page-aligned; `buf_len` is a page
    // multiple covering only pages the kernel already mapped for this
    // file. `_mmap` keepalive in ResidentMmapFile outlives the MTLBuffer.
    // `None` deallocator => Metal does not free host memory.
    let nocopy = unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            NonNull::new(base as *mut _).unwrap(),
            buf_len,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    };
    let Some(buffer) = nocopy else {
        return;
    };
    let entry = Arc::new(ResidentMmapFile {
        buffer,
        _mmap: Some(mmap),
        base_ptr: base,
        len: buf_len,
    });
    let mut guard = MMAP_FILE_CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(HashMap::new);
    cache.entry(key).or_insert(entry);
}

/// Registers a Shared buffer the process allocated (an expert slot
/// slab) so a weight slice inside its contents aliases it exactly as a
/// slice inside a registered GGUF mmap does: no copy, and no bytes
/// charged to the weight cache. On unified memory that is what keeps
/// the slab's bytes counted once, by the expert store that owns them.
pub(crate) fn register_owned_region(
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    base_ptr: usize,
    len: usize,
) {
    let entry = Arc::new(ResidentMmapFile {
        buffer,
        _mmap: None,
        base_ptr,
        len,
    });
    let mut guard = MMAP_FILE_CACHE.lock().unwrap();
    guard
        .get_or_insert_with(HashMap::new)
        .insert(base_ptr, entry);
}

/// Undoes [`register_owned_region`], and drops the process-wide weight
/// cache's aliases of it so they stop retaining the region's buffer.
/// A thread-local mirror may still hold an alias until its thread
/// exits; that keeps the memory allocated (so its address cannot be
/// handed to anything else) and is never served for other bytes.
pub(crate) fn unregister_owned_region(base_ptr: usize) {
    let removed = {
        let mut guard = MMAP_FILE_CACHE.lock().unwrap();
        guard.as_mut().and_then(|c| c.remove(&base_ptr))
    };
    let Some(region) = removed else {
        return;
    };
    if let Some(cache) = WEIGHT_CACHE.lock().unwrap().as_mut() {
        cache.retain(|_, w| {
            !w._keepalive
                .as_ref()
                .is_some_and(|k| Arc::ptr_eq(k, &region))
        });
    }
}

fn find_registered_mmap(weights: &[u8]) -> Option<(Arc<ResidentMmapFile>, usize)> {
    let start = weights.as_ptr() as usize;
    let end = start + weights.len();
    let guard = MMAP_FILE_CACHE.lock().unwrap();
    let cache = guard.as_ref()?;
    for file in cache.values() {
        if start >= file.base_ptr && end <= file.base_ptr + file.len {
            return Some((file.clone(), start - file.base_ptr));
        }
    }
    None
}

/// Build a resident weight buffer for `weights`.
///
/// Prefer zero-copy: if the slice lives inside a mmap registered via
/// [`register_weight_mmap`], alias that file's MTLBuffer at the tensor
/// byte offset (no `to_vec` double). Owned / unregistered slices fall
/// back to a Shared copy.
fn build_resident_weight_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    weights: &[u8],
) -> Result<ResidentWeightBuffer, MetalError> {
    if !weights.is_empty() {
        if let Some((file, offset)) = find_registered_mmap(weights) {
            if file._mmap.is_none() {
                crate::expert_slab::note_slab_bind();
            }
            return Ok(ResidentWeightBuffer {
                buffer: file.buffer.clone(),
                weight_offset: offset,
                nbytes: 0,      // aliased: no cache-budget cost
                fingerprint: 0, // unused: mmap ranges cannot alias
                mmap_backed: true,
                _keepalive: Some(file),
            });
        }
    }

    // Fallback: copy the bytes into a fresh Shared buffer (offset 0).
    let mut weights_owned = weights.to_vec();
    let buffer = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(weights_owned.as_mut_ptr() as *mut _).unwrap(),
            weights_owned.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;
    Ok(ResidentWeightBuffer {
        buffer,
        weight_offset: 0,
        nbytes: weights.len(),
        fingerprint: weight_fingerprint(weights),
        mmap_backed: false,
        _keepalive: None,
    })
}

pub(crate) struct ResidentF32Buffer {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    nbytes: usize,
}

impl Resident for ResidentF32Buffer {
    /// An exhaustive compare, because this entry can prove nothing
    /// else: it always copies, so it neither aliases the host bytes nor
    /// keeps their allocation alive, and what it caches are the RMSNorm
    /// gammas and RoPE frequency factors owned by a `Decoder` that
    /// `/admin/models/load` drops.
    ///
    /// Exact rather than sampled, and affordable for the same reason it
    /// is needed: these are `hidden_dim` floats, single-digit KB, so
    /// the compare costs a fraction of the upload it avoids. That
    /// leaves no residual probability of serving one model's norms to
    /// another (GitHub issue #180).
    fn still_holds(&self, host: &[u8]) -> bool {
        if self.nbytes != host.len() {
            return false;
        }
        // SAFETY: `buffer` is a `StorageModeShared` buffer this process
        // allocated with exactly `nbytes` bytes and only ever reads on
        // the GPU, so its contents pointer is valid and readable for
        // that length for as long as this entry lives.
        let device = unsafe {
            std::slice::from_raw_parts(self.buffer.contents().as_ptr() as *const u8, self.nbytes)
        };
        device == host
    }

    fn resident_bytes(&self) -> usize {
        self.nbytes
    }
}

// SAFETY: same justification as `SharedMetal` -- `MTLBuffer` created
// once and only read by compute kernels is safe to share across threads
// that each build their own command buffer/encoder.
unsafe impl Send for ResidentF32Buffer {}
unsafe impl Sync for ResidentF32Buffer {}

type F32CacheMap = HashMap<HostKey, Arc<ResidentF32Buffer>>;

static F32_CACHE: Mutex<Option<F32CacheMap>> = Mutex::new(None);

thread_local! {
    static TL_F32_CACHE: RefCell<F32CacheMap> = RefCell::new(HashMap::new());
}

pub(crate) fn resident_f32_buffer(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &[f32],
) -> Result<Arc<ResidentF32Buffer>, MetalError> {
    let nbytes = std::mem::size_of_val(data);
    // SAFETY: an initialised `[f32]` is `nbytes` initialised bytes, and
    // `u8` has no alignment requirement. Read-only, and the borrow of
    // `data` outlives the view.
    let host = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, nbytes) };
    get_or_build(&F32_CACHE, &TL_F32_CACHE, host, || {
        let mut data_owned = data.to_vec();
        // SAFETY: `data_owned` holds `nbytes` initialised bytes and is
        // alive across the call, which copies them into the buffer.
        let buffer = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(data_owned.as_mut_ptr() as *mut _).unwrap(),
                nbytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocFailed)?;
        Ok(ResidentF32Buffer { buffer, nbytes })
    })
}

/// One quantized matvec to encode into a fused Metal command buffer
/// (shared activation `x`, one `waitUntilCompleted` for the batch).
#[derive(Clone, Copy)]
pub struct MatvecLaunch<'a> {
    pub kernel_src: &'static str,
    pub fn_name: &'static str,
    pub block_bytes: usize,
    pub block_elems: usize,
    pub weights: &'a [u8],
    pub rows: usize,
    pub row_bytes: usize,
    /// Output rows owned by one threadgroup (`1` = legacy; `2` = Q5_K; `4` = Q4_K/Q6_K).
    pub rows_per_tg: usize,
}

/// What runs over every output vector after the matvecs, in the same
/// command buffer, before the wait.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MatvecEpilogue {
    /// Gemma-2's `final_logit_softcapping`: `y = cap * tanh(y / cap)`.
    pub softcap: Option<f32>,
}

/// Encodes every launch into a single compute command buffer sharing
/// one uploaded `x`, then waits once. Independent projections that
/// share an activation (e.g. Q/K/V) should use this instead of N
/// separate `launch_*_matvec` calls.
pub fn launch_matvec_fused(
    x: &[f32],
    launches: &[MatvecLaunch<'_>],
) -> Result<Vec<Vec<f32>>, MetalError> {
    launch_matvec_fused_with(x, launches, MatvecEpilogue::default())
}

/// One matvec of the kind named as `matvec_launch_meta` names it, with
/// an epilogue: the lm_head of a sampled Gemma-2 decode token, softcap
/// included, so the host never touches the 256k logits before sampling
/// them. `None` when the kind has no Metal matvec.
pub fn launch_matvec_kind_with(
    kind: &str,
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    epilogue: MatvecEpilogue,
) -> Option<Result<Vec<f32>, MetalError>> {
    let (kernel_src, fn_name, block_bytes, block_elems, rows_per_tg) = matvec_launch_meta(kind)?;
    let launch = MatvecLaunch {
        kernel_src,
        fn_name,
        block_bytes,
        block_elems,
        weights,
        rows,
        row_bytes,
        rows_per_tg,
    };
    Some(launch_matvec_fused_with(x, &[launch], epilogue).map(|mut outs| outs.pop().unwrap()))
}

/// [`launch_matvec_fused`] with an [`MatvecEpilogue`] applied to every
/// output in the same command buffer.
pub fn launch_matvec_fused_with(
    x: &[f32],
    launches: &[MatvecLaunch<'_>],
    epilogue: MatvecEpilogue,
) -> Result<Vec<Vec<f32>>, MetalError> {
    launch_matvec_fused_folded(x, launches, epilogue, None)
}

/// [`launch_matvec_fused_with`], with PrismML's folded Hadamard applied
/// to `x` on the device first (`crate::hadamard`).
///
/// One command buffer for the rotation AND the matvecs it feeds: the
/// host used to run the butterfly itself and hand down a fresh vector,
/// which is a CPU pass and an upload per projection. The fold is shared
/// by every launch here for the same reason `x` is -- they are the same
/// activation, and a caller with two different folds has two different
/// activations and must make two calls.
pub fn launch_matvec_fused_folded(
    x: &[f32],
    launches: &[MatvecLaunch<'_>],
    epilogue: MatvecEpilogue,
    fold: Option<&crate::hadamard::FoldPlan<'_>>,
) -> Result<Vec<Vec<f32>>, MetalError> {
    if launches.is_empty() {
        return Ok(Vec::new());
    }
    for launch in launches {
        let n_blocks_per_row = launch.row_bytes / launch.block_bytes;
        assert_eq!(
            launch.weights.len(),
            launch.rows * launch.row_bytes,
            "weights must be exactly rows * row_bytes"
        );
        assert_eq!(
            x.len(),
            n_blocks_per_row * launch.block_elems,
            "x must have exactly n_blocks_per_row * block_elems elements"
        );
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    // Reuses the dense stack's own `x` buffer when `x` IS the vector
    // that stack just returned, and uploads otherwise. One helper, not
    // one copy per consumer: see `crate::resident_act`.
    //
    // A fold rewrites the activation IN PLACE, and the reused buffer
    // belongs to the shared decode scratch, so a folded launch takes a
    // private copy instead. That is the same single upload the host
    // path already paid -- what it saves is the butterfly, not the copy.
    let clock = crate::timing::SubmitClock::start();
    let x_buf = match fold {
        None => crate::resident_act::upload_or_reuse(device, x)?,
        Some(plan) => {
            plan.check(x.len())?;
            crate::resident_act::upload_private(device, x)?
        }
    };
    // Signs live as long as the fold does and never change, so the
    // weight cache holds them for free rather than re-uploading 70 KB
    // per projection.
    let signs_buf = match fold.and_then(|p| p.signs) {
        None => None,
        Some(signs) => {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    signs.as_ptr() as *const u8,
                    std::mem::size_of_val(signs),
                )
            };
            Some(resident_weight_buffer(device, bytes)?)
        }
    };

    let mut weight_bufs = Vec::with_capacity(launches.len());
    let mut out_bufs = Vec::with_capacity(launches.len());
    for launch in launches {
        weight_bufs.push(resident_weight_buffer(device, launch.weights)?);
        out_bufs.push(
            device
                .newBufferWithLength_options(launch.rows * 4, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
        );
    }

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    // Unretained references were measured here and are NOT kept: 7.00
    // tok/s against 7.1 on Bonsai, i.e. nothing, for an `unsafe` whose
    // invariant is that every bound buffer outlives the wait.
    // One compute encoder for the whole fused batch — creating an
    // encoder per matvec (previous behavior) paid Metal encoder setup
    // cost N times and is a large share of the ~14× gap vs ggml-metal.
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    // The rotation runs first on a serial encoder, so every matvec
    // below reads the rotated vector without a barrier of its own.
    if let Some(plan) = fold {
        crate::hadamard::encode_fold(
            &encoder,
            device,
            &x_buf,
            x.len(),
            plan,
            signs_buf.as_ref().map(|b| &*b.buffer),
        )?;
    }
    for (i, launch) in launches.iter().enumerate() {
        encode_matvec(
            &encoder,
            device,
            launch,
            &weight_bufs[i],
            &x_buf,
            &out_bufs[i],
        )?;
    }
    // A serial encoder: the epilogue reads what the matvecs wrote
    // without a barrier of its own.
    if let Some(cap) = epilogue.softcap {
        for (i, launch) in launches.iter().enumerate() {
            crate::elem::encode_softcap(&encoder, device, &out_bufs[i], launch.rows as u32, cap)?;
        }
    }
    encoder.endEncoding();
    // The lm_head of every sampled (non-greedy) decode token runs here,
    // in a SECOND command buffer after the dense stack. Untimed, its GPU
    // time read as host time (GitHub issue #149).
    crate::timing::commit_wait_note(&cmd_buf, "matvec-fused", 32, clock);

    let mut outs = Vec::with_capacity(launches.len());
    for (i, launch) in launches.iter().enumerate() {
        let out_ptr = out_bufs[i].contents();
        let out_slice =
            unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, launch.rows) };
        outs.push(out_slice.to_vec());
    }
    Ok(outs)
}

/// Dense SwiGLU FFN on Metal with device-resident activations:
/// one upload of `x`, gate+up matvecs → SiLU×up → down, one download.
/// Matches CUDA [`frink_cuda::gpu::launch_dense_ffn_swiglu`] for MoE
/// experts; weights stay in the resident cache across calls.
pub fn launch_dense_ffn_swiglu(
    gate: &MatvecLaunch<'_>,
    up: &MatvecLaunch<'_>,
    down: &MatvecLaunch<'_>,
    x: &[f32],
) -> Result<Vec<f32>, MetalError> {
    launch_dense_ffn_swiglu_folded(gate, up, down, x, None, None)
}

/// [`launch_dense_ffn_swiglu`] with PrismML's folded rotation applied on
/// the device to each of the two activations the FFN reads: `x` before
/// gate/up, and the SwiGLU output before `down`
/// (`crate::hadamard`).
///
/// The whole FFN is one command buffer either way; the folds are what
/// let a Bonsai layer take it at all, and the layer drops from two
/// submissions to one.
pub fn launch_dense_ffn_swiglu_folded(
    gate: &MatvecLaunch<'_>,
    up: &MatvecLaunch<'_>,
    down: &MatvecLaunch<'_>,
    x: &[f32],
    fold_x: Option<&crate::hadamard::FoldPlan<'_>>,
    fold_act: Option<&crate::hadamard::FoldPlan<'_>>,
) -> Result<Vec<f32>, MetalError> {
    assert_eq!(gate.rows, up.rows, "gate/up row counts must match");
    assert!(down.rows > 0);
    let n_blocks_gate = gate.row_bytes / gate.block_bytes;
    assert_eq!(
        x.len(),
        n_blocks_gate * gate.block_elems,
        "x length must match gate cols"
    );
    assert_eq!(
        up.row_bytes / up.block_bytes * up.block_elems,
        x.len(),
        "up cols must match x"
    );
    let n_blocks_down = down.row_bytes / down.block_bytes;
    assert_eq!(
        n_blocks_down * down.block_elems,
        gate.rows,
        "down cols must equal gate rows (SwiGLU width)"
    );

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;
    // Timed like every other submission: an untimed command buffer's
    // GPU time reads as host time in the ledger, which is the exact
    // confusion issue #149 was. This path carries a whole FFN, so it
    // was the largest thing the ledger could not see.
    let clock = crate::timing::SubmitClock::start();

    // A rotation rewrites the activation in place, so a folded launch
    // takes a private copy rather than the shared decode scratch (see
    // `launch_matvec_fused_folded`).
    let x_buf = match fold_x {
        None => crate::resident_act::upload_or_reuse(device, x)?,
        Some(plan) => {
            plan.check(x.len())?;
            crate::resident_act::upload_private(device, x)?
        }
    };
    if let Some(plan) = fold_act {
        plan.check(gate.rows)?;
    }
    let signs_buf = |plan: Option<&crate::hadamard::FoldPlan<'_>>| match plan.and_then(|p| p.signs)
    {
        None => Ok(None),
        Some(signs) => {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    signs.as_ptr() as *const u8,
                    std::mem::size_of_val(signs),
                )
            };
            resident_weight_buffer(device, bytes).map(Some)
        }
    };
    let x_signs = signs_buf(fold_x)?;
    let act_signs = signs_buf(fold_act)?;

    let gate_w = resident_weight_buffer(device, gate.weights)?;
    let up_w = resident_weight_buffer(device, up.weights)?;
    let down_w = resident_weight_buffer(device, down.weights)?;
    let gate_buf = device
        .newBufferWithLength_options(gate.rows * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;
    let up_buf = device
        .newBufferWithLength_options(up.rows * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;
    let act_buf = device
        .newBufferWithLength_options(gate.rows * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;
    let out_buf = device
        .newBufferWithLength_options(down.rows * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    if let Some(plan) = fold_x {
        crate::hadamard::encode_fold(
            &encoder,
            device,
            &x_buf,
            x.len(),
            plan,
            x_signs.as_ref().map(|b| &*b.buffer),
        )?;
    }
    encode_matvec(&encoder, device, gate, &gate_w, &x_buf, &gate_buf)?;
    encode_matvec(&encoder, device, up, &up_w, &x_buf, &up_buf)?;
    crate::elem::encode_silu_mul(
        &encoder,
        device,
        &gate_buf,
        &up_buf,
        &act_buf,
        gate.rows as u32,
    )?;
    if let Some(plan) = fold_act {
        crate::hadamard::encode_fold(
            &encoder,
            device,
            &act_buf,
            gate.rows,
            plan,
            act_signs.as_ref().map(|b| &*b.buffer),
        )?;
    }
    encode_matvec(&encoder, device, down, &down_w, &act_buf, &out_buf)?;
    encoder.endEncoding();
    crate::timing::commit_wait_note(&cmd_buf, "dense-ffn", 32, clock);

    let out_ptr = out_buf.contents();
    Ok(unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, down.rows).to_vec() })
}

/// One routed expert's launches + combine weight for [`launch_moe_topk_swiglu`].
pub struct MoeExpertLaunch<'a> {
    pub gate: MatvecLaunch<'a>,
    pub up: MatvecLaunch<'a>,
    pub down: MatvecLaunch<'a>,
    pub weight: f32,
}

/// Contiguous packed expert tensors for llama-style `mul_mv_id` MoE.
pub struct MoePackedQ4<'a> {
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub down: &'a [u8],
    pub gate_stride: usize,
    pub up_stride: usize,
    pub down_stride: usize,
    pub n_experts: usize,
    pub ffn_rows: usize,
    pub hidden_rows: usize,
    pub gate_row_bytes: usize,
    pub down_row_bytes: usize,
    pub gate_kind: &'static str,
    pub up_kind: &'static str,
    pub down_kind: &'static str,
}

/// Routed-expert FFN description for one fused-prefill-stack layer
/// (`PrefillFfnMetal::Moe`). `router_w` is the F32 `ffn_gate_inp` plane
/// `[n_experts, hidden]`; the experts come from the same packed planes
/// the host-routed `launch_moe_prefill_q4_0` path uses.
pub struct PrefillMoeMetal<'a> {
    pub router_w: &'a [f32],
    pub top_k: usize,
    /// `norm_topk_prob`: renormalize the selected probabilities.
    pub renormalize: bool,
    pub packed: MoePackedQ4<'a>,
}

impl PrefillMoeMetal<'_> {
    /// `true` when every piece this layer needs has a Metal kernel and
    /// the shapes are inside the routing kernels' limits.
    pub fn is_supported(&self) -> bool {
        self.top_k > 0
            && moe_mm_id_map0_fn(self.top_k).is_some()
            && self.packed.n_experts <= 256
            && self.router_w.len() == self.packed.n_experts * self.packed.hidden_rows
            && mul_mm_id_f16_meta(self.packed.gate_kind).is_some()
            && mul_mm_id_f16_meta(self.packed.up_kind).is_some()
            && mul_mm_id_f16_meta(self.packed.down_kind).is_some()
    }
}

struct MoeMatvecIdDispatch {
    kernel_src: &'static str,
    fn_name: &'static str,
    rows_per_tg: usize,
    tg_threads: (usize, usize, usize),
    tg_mem: usize,
}

fn moe_matvec_id_dispatch(kind: &str) -> Option<MoeMatvecIdDispatch> {
    match kind {
        "Q4_0" => Some(MoeMatvecIdDispatch {
            kernel_src: Q4_0_MOE_TOPK_KERNEL_SRC,
            fn_name: "q4_0_moe_matvec_id",
            rows_per_tg: 8,
            tg_threads: (32, 2, 1),
            tg_mem: 0,
        }),
        "Q5_0" => Some(MoeMatvecIdDispatch {
            kernel_src: Q5_0_MOE_MATVEC_ID_KERNEL_SRC,
            fn_name: "q5_0_moe_matvec_id",
            rows_per_tg: 8,
            tg_threads: (32, 2, 1),
            tg_mem: 0,
        }),
        "Q4_K" => Some(MoeMatvecIdDispatch {
            kernel_src: Q4_K_MOE_MATVEC_ID_KERNEL_SRC,
            fn_name: "q4_k_moe_matvec_id",
            rows_per_tg: 4,
            tg_threads: (64, 1, 1),
            tg_mem: 0,
        }),
        "Q8_0" => Some(MoeMatvecIdDispatch {
            kernel_src: Q8_0_MOE_MATVEC_ID_KERNEL_SRC,
            fn_name: "q8_0_moe_matvec_id",
            rows_per_tg: 2,
            tg_threads: (128, 1, 1),
            tg_mem: 32,
        }),
        _ => None,
    }
}

fn moe_down_id_dispatch(kind: &str) -> Option<MoeMatvecIdDispatch> {
    match kind {
        "Q4_0" => Some(MoeMatvecIdDispatch {
            kernel_src: Q4_0_MOE_TOPK_KERNEL_SRC,
            fn_name: "q4_0_moe_down_id",
            rows_per_tg: 8,
            tg_threads: (32, 2, 1),
            tg_mem: 0,
        }),
        "Q5_0" => Some(MoeMatvecIdDispatch {
            kernel_src: Q5_0_MOE_MATVEC_ID_KERNEL_SRC,
            fn_name: "q5_0_moe_down_id",
            rows_per_tg: 8,
            tg_threads: (32, 2, 1),
            tg_mem: 0,
        }),
        "Q4_K" => Some(MoeMatvecIdDispatch {
            kernel_src: Q4_K_MOE_MATVEC_ID_KERNEL_SRC,
            fn_name: "q4_k_moe_down_id",
            rows_per_tg: 4,
            tg_threads: (64, 1, 1),
            tg_mem: 0,
        }),
        "Q8_0" => Some(MoeMatvecIdDispatch {
            kernel_src: Q8_0_MOE_MATVEC_ID_KERNEL_SRC,
            fn_name: "q8_0_moe_down_id",
            rows_per_tg: 2,
            tg_threads: (128, 1, 1),
            tg_mem: 32,
        }),
        _ => None,
    }
}

fn moe_row_blocks(row_bytes: usize, kind: &str) -> Result<u32, MetalError> {
    let (_, _, block_bytes, _, _) = matvec_launch_meta(kind).ok_or(MetalError::CommandFailed)?;
    Ok((row_bytes / block_bytes) as u32)
}

fn moe_x_stride(row_bytes: usize, kind: &str) -> Result<u32, MetalError> {
    let (_, _, block_bytes, block_elems, _) =
        matvec_launch_meta(kind).ok_or(MetalError::CommandFailed)?;
    Ok(((row_bytes / block_bytes) * block_elems) as u32)
}

#[allow(clippy::too_many_arguments)]
fn encode_moe_matvec_id(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    kind: &str,
    w: &ResidentWeightBuffer,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    ids: IdsBinding<'_>,
    row_bytes: u32,
    n_blocks: u32,
    n_rows: u32,
    top_k: u32,
    expert_stride: u32,
    n_tokens: u32,
    x_stride: u32,
    n_slots: usize,
) -> Result<(), MetalError> {
    let disp = moe_matvec_id_dispatch(kind).ok_or(MetalError::CommandFailed)?;
    let pipe = ensure_pipeline(device, disp.kernel_src, disp.fn_name)?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&w.buffer), w.weight_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(ids.buf), ids.offset, 3);
        let mut rb = row_bytes;
        encoder.setBytes_length_atIndex(NonNull::new(&mut rb as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut blocks = n_blocks;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut blocks as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut rows = n_rows;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut rows as *mut u32 as *mut _).unwrap(),
            4,
            6,
        );
        let mut tk = top_k;
        encoder.setBytes_length_atIndex(NonNull::new(&mut tk as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut stride = expert_stride;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut stride as *mut u32 as *mut _).unwrap(),
            4,
            8,
        );
        let mut nt = n_tokens;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nt as *mut u32 as *mut _).unwrap(), 4, 9);
        let mut xs = x_stride;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut xs as *mut u32 as *mut _).unwrap(),
            4,
            10,
        );
        if disp.tg_mem > 0 {
            encoder.setThreadgroupMemoryLength_atIndex(disp.tg_mem, 0);
        }
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: (n_rows as usize).div_ceil(disp.rows_per_tg),
            height: 1,
            depth: n_slots,
        },
        MTLSize {
            width: disp.tg_threads.0,
            height: disp.tg_threads.1,
            depth: disp.tg_threads.2,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_moe_down_id(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    kind: &str,
    down_w: &ResidentWeightBuffer,
    act_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_out_buf: &ProtocolObject<dyn MTLBuffer>,
    ids: IdsBinding<'_>,
    row_bytes: u32,
    n_blocks: u32,
    hidden_rows: u32,
    ffn_rows: u32,
    top_k: u32,
    expert_stride: u32,
    n_tokens: u32,
    n_slots: usize,
) -> Result<(), MetalError> {
    let disp = moe_down_id_dispatch(kind).ok_or(MetalError::CommandFailed)?;
    let pipe = ensure_pipeline(device, disp.kernel_src, disp.fn_name)?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&down_w.buffer), down_w.weight_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(act_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(expert_out_buf), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(ids.buf), ids.offset, 3);
        let mut rb = row_bytes;
        encoder.setBytes_length_atIndex(NonNull::new(&mut rb as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut blocks = n_blocks;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut blocks as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut hidden = hidden_rows;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut hidden as *mut u32 as *mut _).unwrap(),
            4,
            6,
        );
        let mut ffn = ffn_rows;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut ffn as *mut u32 as *mut _).unwrap(),
            4,
            7,
        );
        let mut tk = top_k;
        encoder.setBytes_length_atIndex(NonNull::new(&mut tk as *mut u32 as *mut _).unwrap(), 4, 8);
        let mut stride = expert_stride;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut stride as *mut u32 as *mut _).unwrap(),
            4,
            9,
        );
        let mut nt = n_tokens;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut nt as *mut u32 as *mut _).unwrap(),
            4,
            10,
        );
        if disp.tg_mem > 0 {
            encoder.setThreadgroupMemoryLength_atIndex(disp.tg_mem, 0);
        }
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: (hidden_rows as usize).div_ceil(disp.rows_per_tg),
            height: 1,
            depth: n_slots,
        },
        MTLSize {
            width: disp.tg_threads.0,
            height: disp.tg_threads.1,
            depth: disp.tg_threads.2,
        },
    );
    Ok(())
}

/// Prefill map0: per-expert token/slot lists (llama `kernel_mul_mm_id_map0`).
fn moe_mm_id_map0(
    ids: &[i32],
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
) -> (Vec<Vec<(i32, i32)>>, usize) {
    let mut per: Vec<Vec<(i32, i32)>> = vec![Vec::new(); n_experts];
    let n_slots = n_tokens * top_k;
    for (slot, &id) in ids.iter().take(n_slots).enumerate() {
        let eid = id as usize;
        if eid < n_experts {
            let token = (slot / top_k) as i32;
            per[eid].push((token, slot as i32));
        }
    }
    let max_batch = per.iter().map(|v| v.len()).max().unwrap_or(0);
    (per, max_batch)
}

/// llama.cpp `ne21_mm_id_min` — prefer `mul_mm_id` when tokens-per-expert
/// (after map0) can be large; gate on total prompt tokens as a proxy.
const MOE_MM_ID_TOKEN_MIN: usize = 8;

/// Reused MoE prefill scratch (avoids alloc+wait tax every layer).
struct MoePrefillScratch {
    n_slots: usize,
    ffn: usize,
    hidden: usize,
    n_tokens: usize,
    n_experts: usize,
    act: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    up: Retained<ProtocolObject<dyn MTLBuffer>>,
    expert_out: Retained<ProtocolObject<dyn MTLBuffer>>,
    out: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// GPU map0: tokens-per-expert `[n_experts]`.
    mm_id_tpe: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// GPU map0: slot ids `[n_experts, n_tokens]`.
    mm_id_ids: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// f16 activations for `mul_mm_id_f16` (size ≥ `n_slots * ffn`).
    half_in: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// GPU router: logits `[n_tokens, n_experts]`.
    router_logits: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// GPU routing: expert ids `[n_tokens, top_k]`.
    route_ids: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// GPU routing: combine weights `[n_tokens, top_k]`.
    route_w: Retained<ProtocolObject<dyn MTLBuffer>>,
    top_k: usize,
}

thread_local! {
    static TL_MOE_PREFILL: RefCell<Option<MoePrefillScratch>> = const { RefCell::new(None) };
}

fn moe_prefill_scratch(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n_tokens: usize,
    n_slots: usize,
    ffn: usize,
    hidden: usize,
    n_experts: usize,
) -> Result<(), MetalError> {
    moe_prefill_scratch_ex(device, n_tokens, n_slots, ffn, hidden, n_experts, 0)
}

/// [`moe_prefill_scratch`] plus the routing planes the fused stack needs
/// (`top_k > 0`: router logits, ids, combine weights).
#[allow(clippy::too_many_arguments)]
fn moe_prefill_scratch_ex(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n_tokens: usize,
    n_slots: usize,
    ffn: usize,
    hidden: usize,
    n_experts: usize,
    top_k: usize,
) -> Result<(), MetalError> {
    TL_MOE_PREFILL.with(|cell| {
        let mut slot = cell.borrow_mut();
        let need = match slot.as_ref() {
            None => true,
            Some(s) => {
                s.n_slots < n_slots
                    || s.ffn < ffn
                    || s.hidden < hidden
                    || s.n_tokens < n_tokens
                    || s.n_experts < n_experts
                    || s.top_k < top_k
            }
        };
        if need {
            // Never shrink: a top_k=0 (host-routed) grow must not drop the
            // routing planes a previous fused-stack call allocated.
            let (n_slots, ffn, hidden, n_tokens, n_experts, top_k) = match slot.as_ref() {
                Some(s) => (
                    n_slots.max(s.n_slots),
                    ffn.max(s.ffn),
                    hidden.max(s.hidden),
                    n_tokens.max(s.n_tokens),
                    n_experts.max(s.n_experts),
                    top_k.max(s.top_k),
                ),
                None => (n_slots, ffn, hidden, n_tokens, n_experts, top_k),
            };
            let half_elems = (n_tokens * hidden).max(n_slots * ffn);
            let route_elems = (n_tokens * top_k).max(1);
            *slot = Some(MoePrefillScratch {
                n_slots,
                ffn,
                hidden,
                n_tokens,
                n_experts,
                top_k,
                act: device
                    .newBufferWithLength_options(
                        n_slots * ffn * 4,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
                gate: device
                    .newBufferWithLength_options(
                        n_slots * ffn * 4,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
                up: device
                    .newBufferWithLength_options(
                        n_slots * ffn * 4,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
                expert_out: device
                    .newBufferWithLength_options(
                        n_slots * hidden * 4,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
                out: device
                    .newBufferWithLength_options(
                        n_tokens * hidden * 4,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
                mm_id_tpe: device
                    .newBufferWithLength_options(
                        n_experts * 4,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
                mm_id_ids: device
                    .newBufferWithLength_options(
                        n_experts * n_tokens * 4,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
                half_in: device
                    .newBufferWithLength_options(
                        half_elems * 2,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
                router_logits: device
                    .newBufferWithLength_options(
                        n_tokens * n_experts * 4,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
                route_ids: device
                    .newBufferWithLength_options(
                        route_elems * 4,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
                route_w: device
                    .newBufferWithLength_options(
                        route_elems * 4,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocFailed)?,
            });
        }
        Ok(())
    })
}

/// Pre-bound packed expert planes (llama: experts stay in one MTLBuffer
/// after load; encode only rebinds ids/scratch). Keyed by gate base ptr.
pub(crate) struct MoePackedResident {
    key: usize,
    pub(crate) gate: Arc<ResidentWeightBuffer>,
    pub(crate) up: Arc<ResidentWeightBuffer>,
    pub(crate) down: Arc<ResidentWeightBuffer>,
}

thread_local! {
    /// Hoisted packed gate/up/down MTLBuffers — one resolve per layer,
    /// not per token (ROADMAP “expert residency hoist”).
    static TL_MOE_PACKED: RefCell<HashMap<usize, MoePackedResident>> =
        RefCell::new(HashMap::new());
}

pub(crate) fn moe_packed_resident(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    packed: &MoePackedQ4<'_>,
) -> Result<MoePackedResident, MetalError> {
    let key = packed.gate.as_ptr() as usize;
    if let Some(hit) = TL_MOE_PACKED.with(|c| {
        c.borrow().get(&key).map(|r| MoePackedResident {
            key: r.key,
            gate: r.gate.clone(),
            up: r.up.clone(),
            down: r.down.clone(),
        })
    }) {
        return Ok(hit);
    }
    let gate = resident_weight_buffer(device, packed.gate)?;
    let up = resident_weight_buffer(device, packed.up)?;
    let down = resident_weight_buffer(device, packed.down)?;
    let bound = MoePackedResident {
        key,
        gate,
        up,
        down,
    };
    TL_MOE_PACKED.with(|c| {
        c.borrow_mut().insert(
            key,
            MoePackedResident {
                key: bound.key,
                gate: bound.gate.clone(),
                up: bound.up.clone(),
                down: bound.down.clone(),
            },
        );
    });
    Ok(bound)
}

/// Concurrent-safe burst: packed `matvec_id(gate) ∥ matvec_id(up)`.
/// Caller must end the compute encoder (or barrier) before SiLU/down.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q4_0_moe_gate_up_id(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    packed: &MoePackedQ4<'_>,
    ids: IdsBinding<'_>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    up_buf: &ProtocolObject<dyn MTLBuffer>,
    top_k: u32,
    n_tokens: u32,
) -> Result<(), MetalError> {
    assert_eq!(packed.gate_stride, packed.up_stride);
    assert!(n_tokens >= 1 && top_k >= 1);
    let bound = moe_packed_resident(device, packed)?;
    let x_stride = moe_x_stride(packed.gate_row_bytes, packed.gate_kind)?;
    let gate_blocks = moe_row_blocks(packed.gate_row_bytes, packed.gate_kind)?;
    let up_row_bytes = packed.up_stride / packed.ffn_rows;
    let up_blocks = moe_row_blocks(up_row_bytes, packed.up_kind)?;
    let n_slots = (n_tokens as usize) * (top_k as usize);
    encode_moe_matvec_id(
        encoder,
        device,
        packed.gate_kind,
        &bound.gate,
        x_buf,
        gate_buf,
        ids,
        packed.gate_row_bytes as u32,
        gate_blocks,
        packed.ffn_rows as u32,
        top_k,
        packed.gate_stride as u32,
        n_tokens,
        x_stride,
        n_slots,
    )?;
    encode_moe_matvec_id(
        encoder,
        device,
        packed.up_kind,
        &bound.up,
        x_buf,
        up_buf,
        ids,
        up_row_bytes as u32,
        up_blocks,
        packed.ffn_rows as u32,
        top_k,
        packed.up_stride as u32,
        n_tokens,
        x_stride,
        n_slots,
    )?;
    Ok(())
}

/// Encode packed-id Q4_0 MoE: optional gate∥up → silu_mul → down_id → sum.
///
/// Set `gate_up_done=true` when [`encode_q4_0_moe_gate_up_id`] already ran
/// in a prior Concurrent encoder window (MoE stack on Host B).
/// `n_tokens=1` is decode; prefill passes `T`.
/// When `residual_into_out` and `n_tokens==1`, the final sum does
/// `out += weighted` (decode residual fuse); otherwise `out = weighted`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q4_0_moe_id(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    packed: &MoePackedQ4<'_>,
    ids: IdsBinding<'_>,
    route: &ProtocolObject<dyn MTLBuffer>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    up_buf: &ProtocolObject<dyn MTLBuffer>,
    act_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_out_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    top_k: u32,
    n_tokens: u32,
    residual_into_out: bool,
    gate_up_done: bool,
) -> Result<(), MetalError> {
    assert_eq!(packed.gate_stride, packed.up_stride);
    assert!(n_tokens >= 1);
    assert!(top_k >= 1);
    let bound = moe_packed_resident(device, packed)?;
    let down_w = &bound.down;
    let n_slots = (n_tokens as usize) * (top_k as usize);

    if !gate_up_done {
        encode_q4_0_moe_gate_up_id(
            encoder, device, x_buf, packed, ids, gate_buf, up_buf, top_k, n_tokens,
        )?;
        memory_barrier_resources(encoder, &[gate_buf, up_buf]);
    }
    crate::elem::encode_silu_mul(
        encoder,
        device,
        gate_buf,
        up_buf,
        act_buf,
        (n_slots * packed.ffn_rows) as u32,
    )?;
    memory_barrier_resources(encoder, &[act_buf]);

    let weighted_sum = ensure_pipeline(device, Q4_0_MOE_TOPK_KERNEL_SRC, "moe_weighted_sum")?;

    let down_blocks = moe_row_blocks(packed.down_row_bytes, packed.down_kind)?;
    encode_moe_down_id(
        encoder,
        device,
        packed.down_kind,
        down_w,
        act_buf,
        expert_out_buf,
        ids,
        packed.down_row_bytes as u32,
        down_blocks,
        packed.hidden_rows as u32,
        packed.ffn_rows as u32,
        top_k,
        packed.down_stride as u32,
        n_tokens,
        n_slots,
    )?;
    memory_barrier_resources(encoder, &[expert_out_buf]);

    const SUM_TG: usize = 256;
    if residual_into_out && n_tokens == 1 {
        let fuse = ensure_pipeline(
            device,
            Q4_0_MOE_TOPK_KERNEL_SRC,
            "moe_weighted_sum_residual",
        )?;
        encoder.setComputePipelineState(&fuse.0);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(expert_out_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(route), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
            let mut hidden_rows = packed.hidden_rows as u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut hidden_rows as *mut u32 as *mut _).unwrap(),
                4,
                3,
            );
            let mut tk = top_k;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut tk as *mut u32 as *mut _).unwrap(),
                4,
                4,
            );
        }
        dispatch_counted(
            encoder,
            MTLSize {
                width: packed.hidden_rows.div_ceil(SUM_TG),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: SUM_TG,
                height: 1,
                depth: 1,
            },
        );
    } else {
        encoder.setComputePipelineState(&weighted_sum.0);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(expert_out_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(route), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
            let mut hidden_rows = packed.hidden_rows as u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut hidden_rows as *mut u32 as *mut _).unwrap(),
                4,
                3,
            );
            let mut tk = top_k;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut tk as *mut u32 as *mut _).unwrap(),
                4,
                4,
            );
            let mut nt = n_tokens;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut nt as *mut u32 as *mut _).unwrap(),
                4,
                5,
            );
        }
        let sum_elems = (n_tokens as usize) * packed.hidden_rows;
        dispatch_counted(
            encoder,
            MTLSize {
                width: sum_elems.div_ceil(SUM_TG),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: SUM_TG,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

/// Whether the fused `mul_mm_id` path can run this pack.
///
/// TWO conditions, and only the first used to be checked.
///
/// 1. Some expert has gathered at least [`MOE_MM_ID_TOKEN_MIN`] tokens
///    (llama `ne21_mm_id_min`). Below that, a host loop over 64 sparse
///    experts is slower than slot-parallel `mul_mv_id` -- OLMoE
///    fair-chat regressed ~143 to 84 prompt/s when this was ignored.
/// 2. `mul_mm_id_meta` has a kernel for all three planes. It knows
///    Q4_0, Q8_0 and Q4_K, and NOTHING ELSE.
///
/// The second was missing, and the loader does not agree with it: the
/// packed planes admit Q5_0, Q5_K, Q6_K and IQ4_XS as well, because
/// `mapped_sg` gates on `mul_mm_sg_meta`, which covers all seven. So a
/// Q6_K MoE checkpoint packed fine, ran fine while every expert stayed
/// under eight tokens, and then failed with a bare
/// `MetalError::CommandFailed` the moment one expert gathered eight --
/// a prompt-length-dependent failure with nothing in it naming the
/// quantization.
///
/// Falling back to `mul_mv_id` is the right answer rather than a
/// refusal: that path handles all seven kinds and is merely slower, so
/// the checkpoint runs.
fn moe_use_mm_id(max_batch: usize, gate_kind: &str, up_kind: &str, down_kind: &str) -> bool {
    max_batch >= MOE_MM_ID_TOKEN_MIN
        && mul_mm_id_meta(gate_kind).is_some()
        && mul_mm_id_meta(up_kind).is_some()
        && mul_mm_id_meta(down_kind).is_some()
}

/// Whether the packed-id MoE prefill matvec path supports all three planes.
pub fn moe_packed_mul_mv_id_supported(gate_kind: &str, up_kind: &str, down_kind: &str) -> bool {
    moe_matvec_id_dispatch(gate_kind).is_some()
        && moe_matvec_id_dispatch(up_kind).is_some()
        && moe_down_id_dispatch(down_kind).is_some()
}

/// Prefill MoE FFN: packed-id experts over `n_tokens` positions in one CB.
/// `x_batch` is `[T, H]`, `ids`/`route` are `[T, top_k]` (host-routed).
///
/// When some expert has ≥[`MOE_MM_ID_TOKEN_MIN`] gathered tokens (llama
/// `ne21_mm_id_min`), uses fused `mul_mm_id` (indexed GEMM, no gather/scatter);
/// otherwise slot-parallel `mul_mv_id`.
pub fn launch_moe_prefill_q4_0(
    x_batch: &[f32],
    n_tokens: usize,
    packed: &MoePackedQ4<'_>,
    ids: &[i32],
    route: &[f32],
    top_k: usize,
) -> Result<Vec<f32>, MetalError> {
    assert!(n_tokens > 0);
    assert!(top_k > 0 && top_k <= 8);
    assert_eq!(x_batch.len(), n_tokens * packed.hidden_rows);
    assert_eq!(ids.len(), n_tokens * top_k);
    assert_eq!(route.len(), n_tokens * top_k);
    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;
    let n_slots = n_tokens * top_k;
    let hidden = packed.hidden_rows;
    let ffn = packed.ffn_rows;
    let (per_expert, max_batch) = moe_mm_id_map0(ids, n_tokens, top_k, packed.n_experts);
    let use_mm_id = moe_use_mm_id(
        max_batch,
        packed.gate_kind,
        packed.up_kind,
        packed.down_kind,
    );

    let x_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(x_batch.as_ptr() as *mut _).unwrap(),
            x_batch.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;
    let mut ids_mut = ids.to_vec();
    let ids_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(ids_mut.as_mut_ptr() as *mut _).unwrap(),
            ids_mut.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;
    let mut route_mut = route.to_vec();
    let route_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(route_mut.as_mut_ptr() as *mut _).unwrap(),
            route_mut.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;
    moe_prefill_scratch(device, n_tokens, n_slots, ffn, hidden, packed.n_experts)?;

    let result = TL_MOE_PREFILL.with(|cell| {
        let scratch = cell.borrow();
        let scratch = scratch.as_ref().ok_or(MetalError::CommandFailed)?;
        let gate_buf = scratch.gate.as_ref();
        let up_buf = scratch.up.as_ref();
        let act_buf = scratch.act.as_ref();
        let expert_out_buf = scratch.expert_out.as_ref();
        let out_buf = scratch.out.as_ref();
        let mm_id_tpe_buf: &ProtocolObject<dyn MTLBuffer> = scratch.mm_id_tpe.as_ref();
        let mm_id_ids_buf: &ProtocolObject<dyn MTLBuffer> = scratch.mm_id_ids.as_ref();

        let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;

        if use_mm_id {
            // Resolve weights and pipeline names before opening an encoder.
            // `moe_packed_resident` can fail; creating the encoder first and
            // then `?`-ing left it without `endEncoding`, which poisons the
            // queue (`Command encoder released without endEncoding`) and
            // makes every later layer report bare `CommandFailed`.
            let bound = moe_packed_resident(device, packed)?;
            let (gate_fn, gate_bb, gate_be) =
                mul_mm_id_meta(packed.gate_kind).ok_or(MetalError::CommandFailed)?;
            let (up_fn, up_bb, up_be) =
                mul_mm_id_meta(packed.up_kind).ok_or(MetalError::CommandFailed)?;
            let (down_fn, down_bb, down_be) =
                mul_mm_id_meta(packed.down_kind).ok_or(MetalError::CommandFailed)?;
            let hidden_u = hidden as u32;
            let ffn_u = ffn as u32;
            let top_k_u = top_k as u32;
            let n_tokens_u = n_tokens as u32;
            let n_experts_u = packed.n_experts as u32;
            let gate_cols = ((packed.gate_row_bytes / gate_bb) * gate_be) as u32;
            let up_row_bytes = packed.up_stride / ffn;
            let up_cols = ((up_row_bytes / up_bb) * up_be) as u32;
            let down_cols = ((packed.down_row_bytes / down_bb) * down_be) as u32;

            let mut tpe_host = vec![0u32; packed.n_experts];
            let mut ids_host = vec![0i32; packed.n_experts * n_tokens];
            for (eid, list) in per_expert.iter().enumerate() {
                tpe_host[eid] = list.len() as u32;
                for (i, &(_token, slot)) in list.iter().enumerate() {
                    ids_host[eid * n_tokens + i] = slot;
                }
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tpe_host.as_ptr(),
                    mm_id_tpe_buf.contents().as_ptr() as *mut u32,
                    tpe_host.len(),
                );
                std::ptr::copy_nonoverlapping(
                    ids_host.as_ptr(),
                    mm_id_ids_buf.contents().as_ptr() as *mut i32,
                    ids_host.len(),
                );
            }

            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(MetalError::CommandFailed)?;
            let enc_result = (|| {
                encode_mul_mm_id(
                    &encoder,
                    device,
                    gate_fn,
                    &bound.gate,
                    packed.gate_stride as u32,
                    &x_buf,
                    gate_buf,
                    mm_id_ids_buf,
                    mm_id_tpe_buf,
                    n_experts_u,
                    n_tokens_u,
                    top_k_u,
                    ffn_u,
                    gate_cols,
                    packed.gate_row_bytes as u32,
                    0,
                )?;
                encode_mul_mm_id(
                    &encoder,
                    device,
                    up_fn,
                    &bound.up,
                    packed.up_stride as u32,
                    &x_buf,
                    up_buf,
                    mm_id_ids_buf,
                    mm_id_tpe_buf,
                    n_experts_u,
                    n_tokens_u,
                    top_k_u,
                    ffn_u,
                    up_cols,
                    up_row_bytes as u32,
                    0,
                )?;
                memory_barrier_resources(&encoder, &[gate_buf, up_buf]);
                crate::elem::encode_silu_mul(
                    &encoder,
                    device,
                    gate_buf,
                    up_buf,
                    act_buf,
                    (n_slots * ffn) as u32,
                )?;
                memory_barrier_resources(&encoder, &[act_buf]);
                encode_mul_mm_id(
                    &encoder,
                    device,
                    down_fn,
                    &bound.down,
                    packed.down_stride as u32,
                    act_buf,
                    expert_out_buf,
                    mm_id_ids_buf,
                    mm_id_tpe_buf,
                    n_experts_u,
                    n_tokens_u,
                    top_k_u,
                    hidden_u,
                    down_cols,
                    packed.down_row_bytes as u32,
                    1,
                )?;
                memory_barrier_resources(&encoder, &[expert_out_buf]);
                encode_moe_prefill_weighted_sum(
                    &encoder,
                    device,
                    expert_out_buf,
                    &route_buf,
                    out_buf,
                    hidden_u,
                    top_k_u,
                    n_tokens_u,
                )
            })();
            encoder.endEncoding();
            enc_result?;
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        } else {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(MetalError::CommandFailed)?;
            let enc_result = encode_q4_0_moe_id(
                &encoder,
                device,
                &x_buf,
                packed,
                IdsBinding::whole(&ids_buf),
                &route_buf,
                gate_buf,
                up_buf,
                act_buf,
                expert_out_buf,
                out_buf,
                top_k as u32,
                n_tokens as u32,
                false,
                false,
            );
            encoder.endEncoding();
            enc_result?;
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        let ptr = out_buf.contents();
        Ok(unsafe {
            std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n_tokens * hidden).to_vec()
        })
    })?;
    Ok(result)
}

/// Encode llama-style batched Q4_0 MoE (gate+up+SiLU, down, weighted sum)
/// into an existing compute encoder. `experts.len()` must be in `1..=8`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q4_0_moe_topk(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    experts: &[MoeExpertLaunch<'_>],
    act_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_out_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), MetalError> {
    debug_assert!(!experts.is_empty() && experts.len() <= 8);
    let hidden = experts[0].down.rows;
    let ffn = experts[0].gate.rows;
    let input_blocks = experts[0].gate.row_bytes / 18;
    let down_blocks = experts[0].down.row_bytes / 18;
    // llama.cpp Q4_0 tuning: N_SG=2, N_R0=4 → 8 rows/TG.
    let tg = 64usize;
    const ROWS_PER_TG: usize = 8;

    let mut gate_w = Vec::with_capacity(experts.len());
    let mut up_w = Vec::with_capacity(experts.len());
    let mut down_w = Vec::with_capacity(experts.len());
    for ex in experts {
        gate_w.push(resident_weight_buffer(device, ex.gate.weights)?);
        up_w.push(resident_weight_buffer(device, ex.up.weights)?);
        down_w.push(resident_weight_buffer(device, ex.down.weights)?);
    }

    let mut route: Vec<f32> = experts.iter().map(|e| e.weight).collect();
    let route_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(route.as_mut_ptr() as *mut _).unwrap(),
            route.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;

    let gate_up = ensure_pipeline(device, Q4_0_MOE_TOPK_KERNEL_SRC, "q4_0_moe_gate_up")?;
    let down = ensure_pipeline(device, Q4_0_MOE_TOPK_KERNEL_SRC, "q4_0_moe_down")?;
    let weighted_sum = ensure_pipeline(device, Q4_0_MOE_TOPK_KERNEL_SRC, "moe_weighted_sum")?;

    encoder.setComputePipelineState(&gate_up.0);
    unsafe {
        // Unused slots bind expert 0; `n_experts` prevents reads.
        for slot in 0..8usize {
            let i = slot.min(experts.len() - 1);
            encoder.setBuffer_offset_atIndex(
                Some(&gate_w[i].buffer),
                gate_w[i].weight_offset,
                slot,
            );
            encoder.setBuffer_offset_atIndex(
                Some(&up_w[i].buffer),
                up_w[i].weight_offset,
                slot + 8,
            );
        }
        encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 16);
        encoder.setBuffer_offset_atIndex(Some(act_buf), 0, 17);
        let mut row_bytes = experts[0].gate.row_bytes as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut row_bytes as *mut u32 as *mut _).unwrap(),
            4,
            18,
        );
        let mut blocks = input_blocks as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut blocks as *mut u32 as *mut _).unwrap(),
            4,
            19,
        );
        let mut ffn_rows = ffn as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut ffn_rows as *mut u32 as *mut _).unwrap(),
            4,
            20,
        );
        let mut n_experts = experts.len() as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_experts as *mut u32 as *mut _).unwrap(),
            4,
            21,
        );
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: experts.len() * ffn.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );

    encoder.setComputePipelineState(&down.0);
    unsafe {
        for slot in 0..8usize {
            let i = slot.min(experts.len() - 1);
            encoder.setBuffer_offset_atIndex(
                Some(&down_w[i].buffer),
                down_w[i].weight_offset,
                slot,
            );
        }
        encoder.setBuffer_offset_atIndex(Some(act_buf), 0, 8);
        encoder.setBuffer_offset_atIndex(Some(expert_out_buf), 0, 9);
        let mut row_bytes = experts[0].down.row_bytes as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut row_bytes as *mut u32 as *mut _).unwrap(),
            4,
            10,
        );
        let mut blocks = down_blocks as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut blocks as *mut u32 as *mut _).unwrap(),
            4,
            11,
        );
        let mut hidden_rows = hidden as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut hidden_rows as *mut u32 as *mut _).unwrap(),
            4,
            12,
        );
        let mut ffn_rows = ffn as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut ffn_rows as *mut u32 as *mut _).unwrap(),
            4,
            13,
        );
        let mut n_experts = experts.len() as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_experts as *mut u32 as *mut _).unwrap(),
            4,
            14,
        );
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: experts.len() * hidden.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );

    encoder.setComputePipelineState(&weighted_sum.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(expert_out_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&route_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        let mut hidden_rows = hidden as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut hidden_rows as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut n_experts = experts.len() as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_experts as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        let mut n_tokens = 1u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_tokens as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
    }
    const SUM_TG: usize = 256;
    dispatch_counted(
        encoder,
        MTLSize {
            width: hidden.div_ceil(SUM_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: SUM_TG,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn launch_q4_0_moe_topk_batched(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    experts: &[MoeExpertLaunch<'_>],
) -> Result<Vec<f32>, MetalError> {
    debug_assert!(!experts.is_empty() && experts.len() <= 8);
    let hidden = experts[0].down.rows;
    let ffn = experts[0].gate.rows;

    let act_buf = device
        .newBufferWithLength_options(
            experts.len() * ffn * 4,
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or(MetalError::BufferAllocFailed)?;
    let expert_out_buf = device
        .newBufferWithLength_options(
            experts.len() * hidden * 4,
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or(MetalError::BufferAllocFailed)?;
    let out_buf = device
        .newBufferWithLength_options(hidden * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_q4_0_moe_topk(
        &encoder,
        device,
        x_buf,
        experts,
        &act_buf,
        &expert_out_buf,
        &out_buf,
    )?;
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let out_ptr = out_buf.contents();
    Ok(unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, hidden).to_vec() })
}

/// Top-k MoE SwiGLU on Metal in **one** command buffer: upload `x` once,
/// run gate+up+SiLU×+down for every routed expert, weighted-accumulate
/// into a single output, one download. Cuts the ~8 serial CB waits/layer
/// (OLMoE) down to one — the main Metal MoE orchestration tax vs llama.
pub fn launch_moe_topk_swiglu(
    x: &[f32],
    experts: &[MoeExpertLaunch<'_>],
) -> Result<Vec<f32>, MetalError> {
    if experts.is_empty() {
        return Ok(Vec::new());
    }
    let hidden = experts[0].down.rows;
    let ffn = experts[0].gate.rows;
    for ex in experts {
        assert_eq!(ex.gate.rows, ffn);
        assert_eq!(ex.up.rows, ffn);
        assert_eq!(ex.down.rows, hidden);
        let n_blocks = ex.gate.row_bytes / ex.gate.block_bytes;
        assert_eq!(x.len(), n_blocks * ex.gate.block_elems);
        assert_eq!(
            ex.down.row_bytes / ex.down.block_bytes * ex.down.block_elems,
            ffn
        );
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let x_buf = crate::resident_act::upload_or_reuse(device, x)?;

    let q4_0_batched = experts.len() <= 8
        && experts.iter().all(|ex| {
            ex.gate.fn_name == "q4_0_matvec"
                && ex.up.fn_name == "q4_0_matvec"
                && ex.down.fn_name == "q4_0_matvec"
                && ex.gate.block_bytes == 18
                && ex.up.block_bytes == 18
                && ex.down.block_bytes == 18
                && ex.gate.block_elems == 32
                && ex.up.block_elems == 32
                && ex.down.block_elems == 32
                && ex.gate.row_bytes == experts[0].gate.row_bytes
                && ex.up.row_bytes == experts[0].up.row_bytes
                && ex.down.row_bytes == experts[0].down.row_bytes
        });
    if q4_0_batched {
        return launch_q4_0_moe_topk_batched(device, queue, &x_buf, experts);
    }

    // Per-expert scratch so dispatches do not false-share one gate/up/act
    // buffer (experts are independent — serial reuse forced full barriers).
    let mut gate_bufs = Vec::with_capacity(experts.len());
    let mut up_bufs = Vec::with_capacity(experts.len());
    let mut act_bufs = Vec::with_capacity(experts.len());
    let mut down_bufs = Vec::with_capacity(experts.len());
    for _ in experts {
        gate_bufs.push(
            device
                .newBufferWithLength_options(ffn * 4, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
        );
        up_bufs.push(
            device
                .newBufferWithLength_options(ffn * 4, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
        );
        act_bufs.push(
            device
                .newBufferWithLength_options(ffn * 4, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
        );
        down_bufs.push(
            device
                .newBufferWithLength_options(hidden * 4, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferAllocFailed)?,
        );
    }
    // Accumulator must start at zero (newBuffer contents are undefined).
    let mut zeros = vec![0f32; hidden];
    let out_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(zeros.as_mut_ptr() as *mut _).unwrap(),
            hidden * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;

    let mut weight_bufs = Vec::with_capacity(experts.len() * 3);
    for ex in experts {
        weight_bufs.push(resident_weight_buffer(device, ex.gate.weights)?);
        weight_bufs.push(resident_weight_buffer(device, ex.up.weights)?);
        weight_bufs.push(resident_weight_buffer(device, ex.down.weights)?);
    }

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    // Phase all gates, then ups, then silu, then downs, then axpy — same
    // dependency order as sequential, but no buffer reuse hazards between
    // experts so the GPU can overlap independent dispatches.
    for (i, ex) in experts.iter().enumerate() {
        encode_matvec(
            &encoder,
            device,
            &ex.gate,
            &weight_bufs[i * 3],
            &x_buf,
            &gate_bufs[i],
        )?;
    }
    for (i, ex) in experts.iter().enumerate() {
        encode_matvec(
            &encoder,
            device,
            &ex.up,
            &weight_bufs[i * 3 + 1],
            &x_buf,
            &up_bufs[i],
        )?;
    }
    for (i, _) in experts.iter().enumerate() {
        crate::elem::encode_silu_mul(
            &encoder,
            device,
            &gate_bufs[i],
            &up_bufs[i],
            &act_bufs[i],
            ffn as u32,
        )?;
    }
    for (i, ex) in experts.iter().enumerate() {
        encode_matvec(
            &encoder,
            device,
            &ex.down,
            &weight_bufs[i * 3 + 2],
            &act_bufs[i],
            &down_bufs[i],
        )?;
    }
    for (i, ex) in experts.iter().enumerate() {
        crate::elem::encode_axpy(
            &encoder,
            device,
            &out_buf,
            &down_bufs[i],
            ex.weight,
            hidden as u32,
        )?;
    }
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let out_ptr = out_buf.contents();
    Ok(unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, hidden).to_vec() })
}

/// Encodes one quantized matvec into an existing compute encoder (no
/// commit/wait). Used by [`crate::attn`] to fuse QKV→RoPE→GQA→O.
pub(crate) fn encode_matvec(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    launch: &MatvecLaunch<'_>,
    weight: &ResidentWeightBuffer,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), MetalError> {
    encode_matvec_with_offsets(encoder, device, launch, weight, x_buf, 0, out_buf, 0)
}
/// The threadgroup row count `matvec_launch_meta` declares for the
/// kernel with this entry-point name.
///
/// One source of truth for a number that used to live in two places.
/// Falls back to 1 only when no meta row names this kernel, which is a
/// kernel outside the table entirely rather than a forgotten row.
fn rows_per_threadgroup(fn_name: &str) -> usize {
    for kind in MATVEC_KINDS {
        if let Some((_, name, _, _, rows_per_tg)) = matvec_launch_meta(kind) {
            if name == fn_name {
                return rows_per_tg;
            }
        }
    }
    1
}

/// Shared internal launch plumbing for every matvec kernel in this
/// module: the exact device/library/pipeline/buffer/dispatch sequence
/// `launch_q8_0_matvec` used on its own before this helper existed
/// (extracted once a second, third, fourth, and fifth near-identical
/// copy would otherwise have been needed) -- only the compiled kernel
/// source/function name and each format's per-row block byte/element
/// counts differ between formats. The device/command queue, each
/// kernel's compiled pipeline, and quantized weight buffers are
/// process-wide and persistent (see `shared_metal`/`ensure_pipeline`/
/// `resident_weight_buffer` above) -- only the per-call activation
/// upload, command buffer/encoder, and result download remain.
/// Single-matvec callers go through [`launch_matvec_fused`].
#[allow(clippy::too_many_arguments)]
fn launch_matvec(
    kernel_src: &'static str,
    fn_name: &'static str,
    block_bytes: usize,
    block_elems: usize,
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MetalError> {
    // Looked up in `matvec_launch_meta`, NOT restated here.
    //
    // This was a second hardcoded table keyed on `fn_name`, with a
    // `_ => 1` default — so a kernel added to `matvec_launch_meta` and
    // not to this match silently dispatched one row per threadgroup
    // instead of eight, and every row past the first was never written.
    // Adding `q5_0_matvec` hit exactly that: the kernel was correct and
    // the output was zeros.
    //
    // Two tables that must agree about one number, with nothing
    // enforcing it, is the bug shape this codebase has paid for
    // repeatedly. `matvec_launch_meta` already carries the value as its
    // fifth field; the default is now the honest 1 only for a kernel
    // that has no meta row at all.
    let rows_per_tg = rows_per_threadgroup(fn_name);
    let mut outs = launch_matvec_fused(
        x,
        &[MatvecLaunch {
            kernel_src,
            fn_name,
            block_bytes,
            block_elems,
            weights,
            rows,
            row_bytes,
            rows_per_tg,
        }],
    )?;
    Ok(outs.pop().unwrap())
}

/// Like [`encode_matvec`], but binds `x` / `out` at byte offsets into
/// shared buffers. Used by [`launch_matvec_batch`] so N activations
/// share one uploaded `x_batch` buffer and one output buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_matvec_with_offsets(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    launch: &MatvecLaunch<'_>,
    weight: &ResidentWeightBuffer,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    x_byte_offset: usize,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    out_byte_offset: usize,
) -> Result<(), MetalError> {
    let cached_pipeline = ensure_pipeline(device, launch.kernel_src, launch.fn_name)?;
    let pipeline = &cached_pipeline.0;

    if launch.fn_name == "f32_matvec" {
        let cols = launch.row_bytes / 4;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&weight.buffer), weight.weight_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(x_buf), x_byte_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(out_buf), out_byte_offset, 2);
            let mut cols_u = cols as u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut cols_u as *mut u32 as *mut _).unwrap(),
                4,
                3,
            );
            let mut rows_u = launch.rows as u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut rows_u as *mut u32 as *mut _).unwrap(),
                4,
                4,
            );
        }
        // ggml `get_pipeline_mul_mv` for f32 x f32: NR0=2 rows per
        // threadgroup, NSG = min(4, ceil(ne00/128)) simdgroups sharing the
        // reduction axis. `MAX_NSG` in the kernel is 8.
        let nsg = cols.div_ceil(128).clamp(1, 4);
        dispatch_counted(
            encoder,
            MTLSize {
                width: launch.rows.div_ceil(2),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: nsg,
                depth: 1,
            },
        );
        return Ok(());
    }

    let n_blocks_per_row = launch.row_bytes / launch.block_bytes;
    let rows_per_tg = launch.rows_per_tg.max(1);
    let (tg_threads, tg_mem_bytes) = match launch.fn_name {
        // ggml Q4_0: NSG=2 × NR0=4 → 8 rows / 64 threads; simd_sum only.
        "q4_0_matvec" => (64usize, 0usize),
        // Q5_0: NSG=2 × NR=4 → the same 8 rows / 64 threads, simd_sum
        // only. This entry is REQUIRED, not an optimisation: the
        // `rows_per_tg > 1` default below dispatches 32 threads, i.e.
        // ONE simdgroup, so `sg` is always 0 and the second group of
        // four rows is never written. A correct kernel then returns
        // zeros for half its rows, which is what it did.
        "q5_0_matvec" => (64usize, 0usize),
        // ggml Q4_K / Q5_K / Q6_K: 2 simdgroups × 32 lanes, register packs (no TG mem).
        "q4_k_matvec" | "q5_k_matvec" | "q6_k_matvec" => (64usize, 0usize),
        // ggml Q8_0: NSG=4 simdgroups on nr0=2 rows; 2x4 floats of TG
        // reduce scratch (min 16-byte TG allocation granularity).
        "q8_0_matvec" => (128usize, 32usize),
        // ggml IQ4_XS: 2 simdgroups x 32 lanes; 32 floats of TG memory
        // hold the non-linear codebook (one copy per 16 lanes).
        "iq4_xs_matvec" => (64usize, 128usize),
        // PTQ1_0: the fork's geometry, ONE simdgroup on NR=4 rows
        // (`N_SG_PTQ1_0 1`, `N_R0_PTQ1_0 4`), simd_sum only.
        "ptq1_0_matvec" => (32usize, 0usize), // NR rows per simdgroup
        // Q2_K (NR=4) and Q3_K (NR=2): 2 simdgroups x 32 lanes, simd_sum only.
        "q2_k_matvec" | "q3_k_matvec" => (64usize, 0usize),
        // ggml IQ4_NL / MXFP4: 2 simdgroups x 2 rows, codebook in TG memory.
        "iq4_nl_matvec" | "mxfp4_matvec" => crate::codebook4::CODEBOOK4_MATVEC_TG,
        _ if rows_per_tg > 1 => (32usize, 256 * 4),
        _ => {
            let tg = n_blocks_per_row.next_power_of_two().clamp(32, 256);
            (tg, tg * 4)
        }
    };
    let n_tg = launch.rows.div_ceil(rows_per_tg);
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&weight.buffer), weight.weight_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(x_buf), x_byte_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf), out_byte_offset, 2);
        let mut row_bytes_u32 = launch.row_bytes as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut row_bytes_u32 as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut n_blocks_u32 = n_blocks_per_row as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_blocks_u32 as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        if rows_per_tg > 1 {
            let mut n_rows_u32 = launch.rows as u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut n_rows_u32 as *mut u32 as *mut _).unwrap(),
                4,
                5,
            );
        }
        if tg_mem_bytes > 0 {
            encoder.setThreadgroupMemoryLength_atIndex(tg_mem_bytes, 0);
        }
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// One weight matrix × N activations in a single Metal command buffer
/// (shared resident weights, one `x_batch` upload, one
/// `waitUntilCompleted`).
///
/// Distinct from [`launch_matvec_fused`] (shared **one** `x`, different
/// weight matrices) and from `MatvecLaunch::rows_per_tg` (multiple
/// **weight rows** per threadgroup for a single activation). This is
/// multi-`x`: `x_batch` is layout `[batch, cols]`, returned `y` is
/// `[batch, rows]`.
///
/// For a kind with a simdgroup `mul_mm` kernel, prefer that at
/// `batch_size >= 4` (real multi-x kernel with weight reuse). This path
/// encodes N matvecs.
pub fn launch_matvec_batch(
    launch: &MatvecLaunch<'_>,
    x_batch: &[f32],
    batch_size: usize,
) -> Result<Vec<f32>, MetalError> {
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    let n_blocks_per_row = launch.row_bytes / launch.block_bytes;
    let cols = n_blocks_per_row * launch.block_elems;
    assert_eq!(
        launch.weights.len(),
        launch.rows * launch.row_bytes,
        "weights must be exactly rows * row_bytes"
    );
    assert_eq!(
        x_batch.len(),
        batch_size * cols,
        "x_batch must be batch_size * cols"
    );

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    let mut x_owned = x_batch.to_vec();
    let x_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(x_owned.as_mut_ptr() as *mut _).unwrap(),
            x_owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;

    let weight_buf = resident_weight_buffer(device, launch.weights)?;
    let out_elems = batch_size * launch.rows;
    let out_buf = device
        .newBufferWithLength_options(out_elems * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    for b in 0..batch_size {
        encode_matvec_with_offsets(
            &encoder,
            device,
            launch,
            &weight_buf,
            &x_buf,
            b * cols * 4,
            &out_buf,
            b * launch.rows * 4,
        )?;
    }
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let out_ptr = out_buf.contents();
    let out_slice =
        unsafe { std::slice::from_raw_parts(out_ptr.as_ptr() as *const f32, out_elems) };
    Ok(out_slice.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loader packs seven quant kinds into MoE planes; `mul_mm_id`
    /// has kernels for three. Batch size alone used to decide, so the
    /// other four failed with `CommandFailed` as soon as one expert
    /// gathered eight tokens -- and stayed fine below that, which is
    /// what made it a prompt-length-dependent failure rather than a
    /// load error.
    #[test]
    fn a_pack_without_a_mul_mm_id_kernel_falls_back_instead_of_failing() {
        let big = MOE_MM_ID_TOKEN_MIN;
        for kind in ["Q4_0", "Q8_0", "Q4_K"] {
            assert!(
                moe_use_mm_id(big, kind, kind, kind),
                "{kind} has a mul_mm_id kernel and should use it"
            );
        }
        // Every kind `mapped_sg` admits that `mul_mm_id_meta` does not
        // know. These are the four that used to crash.
        for kind in ["Q5_0", "Q5_K", "Q6_K", "IQ4_XS"] {
            assert!(
                !moe_use_mm_id(big, kind, kind, kind),
                "{kind} has no mul_mm_id kernel and must fall back to mul_mv_id"
            );
        }
        // A pack is only as fused as its weakest plane: one unsupported
        // kind is enough, whichever of the three it is.
        assert!(!moe_use_mm_id(big, "Q6_K", "Q4_0", "Q4_0"));
        assert!(!moe_use_mm_id(big, "Q4_0", "Q6_K", "Q4_0"));
        assert!(!moe_use_mm_id(big, "Q4_0", "Q4_0", "Q6_K"));
        // The batch-size rule still applies to a supported pack.
        assert!(!moe_use_mm_id(
            MOE_MM_ID_TOKEN_MIN - 1,
            "Q4_0",
            "Q4_0",
            "Q4_0"
        ));
    }

    fn real_q8_0_test_matrix(rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.037).sin()).collect();
        let mut weights = Vec::new();
        let mut expected = Vec::new();
        for r in 0..rows {
            let row_vals: Vec<f32> = (0..cols)
                .map(|i| ((r * 7 + i) as f32 * 0.013).cos())
                .collect();
            let q = frink_quant::quantize_q8_0(&row_vals);
            expected.push(frink_quant::dot_q8_0_f32_scalar(&q, &x));
            weights.extend_from_slice(&q);
        }
        (weights, x, expected)
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q8_0_matvec_matches_cpu_reference() {
        let rows = 8;
        let cols = 256;
        let row_bytes = (cols / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES;
        let (weights, x, expected) = real_q8_0_test_matrix(rows, cols);

        let result = launch_q8_0_matvec(&weights, &x, rows, row_bytes).expect("kernel launch");

        assert_eq!(result.len(), expected.len());
        for (i, (a, b)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "row {i}: gpu={a} cpu={b} (diff too large)"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_iq4_xs_matvec_matches_cpu_reference() {
        // No IQ4_XS quantizer exists in `frink_quant` (encode is
        // llama.cpp-side); any bit pattern is a valid block, so build
        // deterministic pseudo-random blocks (finite small `d`) and
        // compare against the fused CPU dot. rows=6 exercises the
        // n_rows guard (not a multiple of rows_per_tg=4); 2 blocks/row
        // exercises the odd/even block split across lane groups.
        let rows = 6;
        let cols = 512;
        let blocks_per_row = cols / frink_quant::IQ4_XS_BLOCK_ELEMS;
        let row_bytes = blocks_per_row * frink_quant::IQ4_XS_BLOCK_BYTES;

        let mut state = 0x1234_5678u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let mut weights = vec![0u8; rows * row_bytes];
        for b in weights.iter_mut() {
            *b = next();
        }
        for r in 0..rows {
            for ib in 0..blocks_per_row {
                let off = r * row_bytes + ib * frink_quant::IQ4_XS_BLOCK_BYTES;
                let d = half::f16::from_f32(0.01 + 0.002 * (r * blocks_per_row + ib) as f32);
                weights[off..off + 2].copy_from_slice(&d.to_le_bytes());
            }
        }

        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.05).sin()).collect();
        let expected: Vec<f32> = (0..rows)
            .map(|r| frink_quant::dot_iq4_xs_f32(&weights[r * row_bytes..(r + 1) * row_bytes], &x))
            .collect();

        let result = launch_iq4_xs_matvec(&weights, &x, rows, row_bytes).expect("kernel launch");

        assert_eq!(result.len(), expected.len());
        for (i, (a, b)) in result.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-3 * b.abs().max(1.0);
            assert!((a - b).abs() < tol, "row {i}: gpu={a} cpu={b} tol={tol}");
        }
    }

    /// `MATVEC_KINDS` and `matvec_launch_meta` agree: every listed kind
    /// has a row, and every kind the meta answers for is listed (the
    /// meta is a `match`, so the second half is the first half plus the
    /// kinds the CPU side knows).
    #[test]
    fn every_matvec_kind_is_listed_and_every_listed_kind_has_a_row() {
        for kind in MATVEC_KINDS {
            let (_, fn_name, _, _, rows_per_tg) =
                matvec_launch_meta(kind).expect("listed kind has a meta row");
            assert_eq!(
                rows_per_threadgroup(fn_name),
                rows_per_tg,
                "{kind}: {fn_name} resolves to the fallback geometry"
            );
        }
        // Every kind the CPU side names for Metal is in the list
        // (`frink-core`'s gpu_backend test holds the other direction
        // through `metal_matvec_launch`); the cheap check here is that
        // the GEMM table's kinds all have a matvec.
        for kind in [
            "Q8_0", "Q4_0", "Q5_0", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "PTQ1_0", "Q2_K", "Q3_K",
            "IQ4_NL", "MXFP4",
        ] {
            assert!(mul_mm_sg_meta(kind).is_some(), "{kind} has no GEMM row");
            assert!(
                MATVEC_KINDS.contains(&kind),
                "{kind} has a GEMM and no matvec"
            );
        }
    }

    /// The PTQ1_0 matvec against `frink_quant::ternary`'s scalar dot on
    /// a ternary row the quantizer packed: rows=6 exercises the row
    /// guard (NR=4 x NSG=2), 5 blocks/row leaves lanes idle.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_ptq1_0_matvec_matches_cpu_reference() {
        use frink_quant::ternary::{dot_trits_f32, quantize_trits, PTQ1_0};
        let rows = 6;
        let cols = 640;
        let row_bytes = cols / 128 * 28;
        let mut seed = 5u32;
        let mut weights = Vec::with_capacity(rows * row_bytes);
        let mut rows_f32 = Vec::new();
        for _ in 0..rows {
            let w: Vec<f32> = (0..cols)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    match (seed >> 24) % 3 {
                        0 => -0.5,
                        1 => 0.0,
                        _ => 0.5,
                    }
                })
                .collect();
            weights.extend(quantize_trits(&w, PTQ1_0));
            rows_f32.push(w);
        }
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.05).sin()).collect();
        let expected: Vec<f32> = (0..rows)
            .map(|r| dot_trits_f32(&weights[r * row_bytes..(r + 1) * row_bytes], PTQ1_0, &x))
            .collect();
        let result = launch_ptq1_0_matvec(&weights, &x, rows, row_bytes).expect("kernel launch");
        assert_eq!(result.len(), expected.len());
        for (i, (a, b)) in result.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-3 * b.abs().max(1.0);
            assert!((a - b).abs() < tol, "row {i}: gpu={a} cpu={b} tol={tol}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn probe_finds_a_real_device_name() {
        let name = probe().expect("this dev machine has a real Metal GPU");
        assert!(!name.is_empty());
    }

    /// Build a Q5_0 block from 32 explicit 0..31 codes, in ggml's
    /// on-disk layout.
    ///
    /// Hand-built rather than quantized, because `frink-quant` has
    /// `dequant_q5_0` and no `quantize_q5_0` — and because a test that
    /// round-trips through frink's own quantizer could not catch a
    /// kernel that mirrors that quantizer's mistake. These codes are
    /// chosen to put the fifth bit on both sides of the split: the low
    /// half reads bit `j` of `qh`, the high half reads bit `j + 16`, and
    /// a kernel that confuses the two passes any test whose codes are
    /// all under 16.
    fn q5_0_block(d: f32, codes: [u8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(22);
        block.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        let mut qh: u32 = 0;
        for (j, &c) in codes.iter().enumerate() {
            assert!(c < 32, "Q5_0 codes are 5-bit");
            if c & 0x10 != 0 {
                qh |= 1 << j;
            }
        }
        block.extend_from_slice(&qh.to_le_bytes());
        for j in 0..16 {
            block.push((codes[j] & 0x0F) | ((codes[j + 16] & 0x0F) << 4));
        }
        block
    }

    /// The Q5_0 matvec, against `frink_quant::dequant_q5_0` as the
    /// reference.
    ///
    /// Q5_0 had a Metal GEMM but no matvec, so a Q5_0 checkpoint ran
    /// prefill on the GPU and every decode step on the CPU — the half of
    /// the run that dominates an interactive session. This is the test
    /// that lets the kernel be trusted.
    ///
    /// Shapes deliberately include a row count that is not a multiple of
    /// the 8 rows a threadgroup covers (so the tail is clamped at the
    /// write) and a block count that is not a multiple of the 32-lane
    /// stride (so some lanes contribute nothing).
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn q5_0_matvec_matches_the_reference_dequantizer() {
        for (rows, blocks_per_row) in [(8usize, 32usize), (5, 3), (1, 1), (17, 40)] {
            let cols = blocks_per_row * 32;
            let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.041).sin()).collect();

            let mut weights: Vec<u8> = Vec::new();
            let mut expected = Vec::with_capacity(rows);
            for r in 0..rows {
                let mut row_bytes: Vec<u8> = Vec::new();
                for b in 0..blocks_per_row {
                    let d = 0.05 + 0.01 * ((r + b) % 7) as f32;
                    let codes: [u8; 32] =
                        std::array::from_fn(|i| ((r * 13 + b * 5 + i * 3) % 32) as u8);
                    row_bytes.extend_from_slice(&q5_0_block(d, codes));
                }
                let dequantized = frink_quant::dequant_q5_0(&row_bytes).expect("reference dequant");
                assert_eq!(dequantized.len(), cols);
                let acc: f64 = dequantized
                    .iter()
                    .zip(x.iter())
                    .map(|(w, xv)| (w * xv) as f64)
                    .sum();
                expected.push(acc as f32);
                weights.extend_from_slice(&row_bytes);
            }

            let got = launch_matvec(
                Q5_0_MATVEC_KERNEL_SRC,
                "q5_0_matvec",
                22,
                32,
                &weights,
                &x,
                rows,
                blocks_per_row * 22,
            )
            .expect("kernel launch");

            assert_eq!(got.len(), rows, "rows={rows} blocks={blocks_per_row}");
            for (i, (g, w)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-3 * w.abs().max(1.0),
                    "row {i} of {rows} (blocks={blocks_per_row}): got {g}, reference {w}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn f32_matvec_matches_cpu_reference() {
        // Shapes that exercise the simdgroup port's edges: an odd row
        // count (the last threadgroup's second row is clamped and must be
        // dropped at the write), a column count that is neither a multiple
        // of the 32-lane simdgroup nor of 128 (so `nsg` clamps and the
        // strided loop leaves a ragged tail), a router-shaped case, and a
        // single row.
        for (rows, cols) in [(64usize, 2048usize), (7, 200), (1, 33), (2, 4096)] {
            let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.037).sin()).collect();
            let mut weights_f32 = Vec::with_capacity(rows * cols);
            let mut expected = Vec::with_capacity(rows);
            for r in 0..rows {
                let mut acc = 0.0f64;
                for (i, xv) in x.iter().enumerate() {
                    let w = ((r * 7 + i) as f32 * 0.013).cos();
                    weights_f32.push(w);
                    acc += (w * xv) as f64;
                }
                expected.push(acc as f32);
            }
            let weights: Vec<u8> = weights_f32.iter().flat_map(|w| w.to_le_bytes()).collect();

            let got = launch_matvec(
                F32_MATVEC_KERNEL_SRC,
                "f32_matvec",
                4,
                1,
                &weights,
                &x,
                rows,
                cols * 4,
            )
            .expect("kernel launch");

            assert_eq!(got.len(), rows, "rows={rows} cols={cols}");
            for (i, (g, w)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-3 * w.abs().max(1.0),
                    "rows={rows} cols={cols} row {i}: got {g}, want {w}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_0_matvec_matches_cpu_reference() {
        // Real, non-trivial Q4_0 rows built directly (no `quantize_q4_0`
        // producer exists in `frink_quant` -- Q4_0 is load-only in this
        // codebase, same convention `frink-cuda`'s Q4_0 test uses).
        // Built `blocks_per_row` blocks per row (not a single hard-coded
        // block regardless of `cols`) -- an earlier version of the
        // equivalent CUDA test got this wrong and only caught it via a
        // real out-of-bounds panic on real GPU hardware; built correctly
        // here from the start given that documented lesson.
        let rows = 4;
        let cols = 64;
        let blocks_per_row = cols / frink_quant::Q4_0_BLOCK_ELEMS;
        let row_bytes = blocks_per_row * frink_quant::Q4_0_BLOCK_BYTES;

        let mut weights = Vec::new();
        for r in 0..rows {
            for b in 0..blocks_per_row {
                weights.extend_from_slice(
                    &half::f16::from_f32(0.05 + (r * blocks_per_row + b) as f32 * 0.01)
                        .to_le_bytes(),
                );
                for i in 0..16u8 {
                    let lo = (i + r as u8 + b as u8) % 16;
                    let hi = (15 - i + r as u8 + b as u8) % 16;
                    weights.push(lo | (hi << 4));
                }
            }
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.09).sin()).collect();
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                let row_slice = &weights[r * row_bytes..(r + 1) * row_bytes];
                frink_quant::dot_q4_0_f32_scalar(row_slice, &x)
            })
            .collect();

        let result = launch_q4_0_matvec(&weights, &x, rows, row_bytes).expect("kernel launch");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "row {i}: GPU={got} CPU reference={want}"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_0_moe_topk_batched_matches_cpu_reference() {
        fn q4_matrix(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
            let blocks = cols / frink_quant::Q4_0_BLOCK_ELEMS;
            let mut out = Vec::with_capacity(rows * blocks * frink_quant::Q4_0_BLOCK_BYTES);
            for r in 0..rows {
                for b in 0..blocks {
                    out.extend_from_slice(
                        &half::f16::from_f32(
                            0.01 + ((r * blocks + b + seed as usize) % 17) as f32 * 0.003,
                        )
                        .to_le_bytes(),
                    );
                    for i in 0..16u8 {
                        let lo = i.wrapping_add(r as u8).wrapping_add(seed) & 15;
                        let hi = (15u8.wrapping_sub(i))
                            .wrapping_add(b as u8)
                            .wrapping_add(seed)
                            & 15;
                        out.push(lo | (hi << 4));
                    }
                }
            }
            out
        }

        let hidden = 64;
        let ffn = 96;
        let top_k = 3;
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.071).sin()).collect();
        let route = [0.5f32, 0.3, 0.2];
        let mut gates = Vec::new();
        let mut ups = Vec::new();
        let mut downs = Vec::new();
        for e in 0..top_k {
            gates.push(q4_matrix(ffn, hidden, (e * 3 + 1) as u8));
            ups.push(q4_matrix(ffn, hidden, (e * 3 + 2) as u8));
            downs.push(q4_matrix(hidden, ffn, (e * 3 + 3) as u8));
        }

        let gu_row_bytes = (hidden / frink_quant::Q4_0_BLOCK_ELEMS) * frink_quant::Q4_0_BLOCK_BYTES;
        let down_row_bytes = (ffn / frink_quant::Q4_0_BLOCK_ELEMS) * frink_quant::Q4_0_BLOCK_BYTES;
        let mut expected = vec![0f32; hidden];
        for e in 0..top_k {
            let mut act = vec![0f32; ffn];
            for (r, slot) in act.iter_mut().enumerate() {
                let range = r * gu_row_bytes..(r + 1) * gu_row_bytes;
                let g = frink_quant::dot_q4_0_f32_scalar(&gates[e][range.clone()], &x);
                let u = frink_quant::dot_q4_0_f32_scalar(&ups[e][range], &x);
                *slot = frink_core_silu(g) * u;
            }
            for (r, slot) in expected.iter_mut().enumerate() {
                let range = r * down_row_bytes..(r + 1) * down_row_bytes;
                *slot += route[e] * frink_quant::dot_q4_0_f32_scalar(&downs[e][range], &act);
            }
        }

        let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
            matvec_launch_meta("Q4_0").unwrap();
        let launches: Vec<MoeExpertLaunch<'_>> = (0..top_k)
            .map(|e| MoeExpertLaunch {
                gate: MatvecLaunch {
                    kernel_src: src,
                    fn_name,
                    block_bytes,
                    block_elems,
                    weights: &gates[e],
                    rows: ffn,
                    row_bytes: gu_row_bytes,
                    rows_per_tg,
                },
                up: MatvecLaunch {
                    kernel_src: src,
                    fn_name,
                    block_bytes,
                    block_elems,
                    weights: &ups[e],
                    rows: ffn,
                    row_bytes: gu_row_bytes,
                    rows_per_tg,
                },
                down: MatvecLaunch {
                    kernel_src: src,
                    fn_name,
                    block_bytes,
                    block_elems,
                    weights: &downs[e],
                    rows: hidden,
                    row_bytes: down_row_bytes,
                    rows_per_tg,
                },
                weight: route[e],
            })
            .collect();

        let got = launch_moe_topk_swiglu(&x, &launches).expect("batched Q4_0 MoE");
        for (i, (&g, &w)) in got.iter().zip(&expected).enumerate() {
            let tol = 5e-3 * w.abs().max(1.0);
            assert!((g - w).abs() <= tol, "elem {i}: gpu={g} cpu={w} tol={tol}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_moe_prefill_q4_0_matches_per_token_packed() {
        fn q4_matrix(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
            let blocks = cols / frink_quant::Q4_0_BLOCK_ELEMS;
            let mut out = Vec::with_capacity(rows * blocks * frink_quant::Q4_0_BLOCK_BYTES);
            for r in 0..rows {
                for b in 0..blocks {
                    out.extend_from_slice(
                        &half::f16::from_f32(
                            0.01 + ((r * blocks + b + seed as usize) % 17) as f32 * 0.003,
                        )
                        .to_le_bytes(),
                    );
                    for i in 0..16u8 {
                        let lo = i.wrapping_add(r as u8).wrapping_add(seed) & 15;
                        let hi = (15u8.wrapping_sub(i))
                            .wrapping_add(b as u8)
                            .wrapping_add(seed)
                            & 15;
                        out.push(lo | (hi << 4));
                    }
                }
            }
            out
        }

        let hidden = 64;
        let ffn = 96;
        let n_experts = 4;
        let top_k = 2;
        // ≥ MOE_MM_ID_TOKEN_MIN so this exercises fused `mul_mm_id`, not `mul_mv_id`.
        let n_tokens = 16;
        let gu_row_bytes = (hidden / frink_quant::Q4_0_BLOCK_ELEMS) * frink_quant::Q4_0_BLOCK_BYTES;
        let down_row_bytes = (ffn / frink_quant::Q4_0_BLOCK_ELEMS) * frink_quant::Q4_0_BLOCK_BYTES;
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        for e in 0..n_experts {
            gate.extend(q4_matrix(ffn, hidden, (e * 3 + 1) as u8));
            up.extend(q4_matrix(ffn, hidden, (e * 3 + 2) as u8));
            down.extend(q4_matrix(hidden, ffn, (e * 3 + 3) as u8));
        }
        let gate_stride = ffn * gu_row_bytes;
        let down_stride = hidden * down_row_bytes;
        let packed = MoePackedQ4 {
            gate: &gate,
            up: &up,
            down: &down,
            gate_stride,
            up_stride: gate_stride,
            down_stride,
            n_experts,
            ffn_rows: ffn,
            hidden_rows: hidden,
            gate_row_bytes: gu_row_bytes,
            down_row_bytes,
            gate_kind: "Q4_0",
            up_kind: "Q4_0",
            down_kind: "Q4_0",
        };
        let mut x_batch = Vec::with_capacity(n_tokens * hidden);
        let mut ids = Vec::with_capacity(n_tokens * top_k);
        let mut route = Vec::with_capacity(n_tokens * top_k);
        let mut expected = Vec::with_capacity(n_tokens * hidden);
        for t in 0..n_tokens {
            let x: Vec<f32> = (0..hidden)
                .map(|i| ((i + t * 7) as f32 * 0.071).sin())
                .collect();
            let e0 = t % n_experts;
            let e1 = (t + 1) % n_experts;
            let w0 = 0.6f32;
            let w1 = 0.4f32;
            ids.push(e0 as i32);
            ids.push(e1 as i32);
            route.push(w0);
            route.push(w1);
            x_batch.extend_from_slice(&x);
            let mut out_t = vec![0f32; hidden];
            for (eid, w) in [(e0, w0), (e1, w1)] {
                let mut act = vec![0f32; ffn];
                let g_base = eid * gate_stride;
                let u_base = eid * gate_stride;
                let d_base = eid * down_stride;
                for r in 0..ffn {
                    let range = g_base + r * gu_row_bytes..g_base + (r + 1) * gu_row_bytes;
                    let g = frink_quant::dot_q4_0_f32_scalar(&gate[range.clone()], &x);
                    let u = frink_quant::dot_q4_0_f32_scalar(
                        &up[u_base + r * gu_row_bytes..u_base + (r + 1) * gu_row_bytes],
                        &x,
                    );
                    act[r] = frink_core_silu(g) * u;
                }
                for (r, slot) in out_t.iter_mut().enumerate() {
                    let range = d_base + r * down_row_bytes..d_base + (r + 1) * down_row_bytes;
                    *slot += w * frink_quant::dot_q4_0_f32_scalar(&down[range], &act);
                }
            }
            expected.extend_from_slice(&out_t);
        }
        let got = launch_moe_prefill_q4_0(&x_batch, n_tokens, &packed, &ids, &route, top_k)
            .expect("prefill MoE");
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &w)) in got.iter().zip(&expected).enumerate() {
            let tol = 5e-3 * w.abs().max(1.0);
            assert!((g - w).abs() <= tol, "elem {i}: gpu={g} cpu={w} tol={tol}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_moe_prefill_mixed_q4_k_q8_0_matches_per_token_packed() {
        let hidden = 512;
        let ffn = 512;
        let n_experts = 4;
        let top_k = 2;
        let n_tokens = 3;
        let gu_row_bytes = (hidden / frink_quant::Q4_K_BLOCK_ELEMS) * frink_quant::Q4_K_BLOCK_BYTES;
        let down_row_bytes = (ffn / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES;

        fn q8_matrix(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
            let blocks = cols / frink_quant::Q8_0_BLOCK_ELEMS;
            let mut out = Vec::with_capacity(rows * blocks * frink_quant::Q8_0_BLOCK_BYTES);
            for r in 0..rows {
                for b in 0..blocks {
                    out.extend_from_slice(
                        &half::f16::from_f32(
                            0.02 + ((r * blocks + b + seed as usize) % 13) as f32 * 0.004,
                        )
                        .to_le_bytes(),
                    );
                    for i in 0..32u8 {
                        out.push(i.wrapping_add(r as u8).wrapping_add(seed).wrapping_mul(3));
                    }
                }
            }
            out
        }

        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        for e in 0..n_experts {
            gate.extend(pseudo_bytes((e * 3 + 1) as u32, ffn * gu_row_bytes));
            up.extend(pseudo_bytes((e * 3 + 2) as u32, ffn * gu_row_bytes));
            down.extend(q8_matrix(hidden, ffn, (e * 3 + 3) as u8));
        }
        let gate_stride = ffn * gu_row_bytes;
        let down_stride = hidden * down_row_bytes;
        let packed = MoePackedQ4 {
            gate: &gate,
            up: &up,
            down: &down,
            gate_stride,
            up_stride: gate_stride,
            down_stride,
            n_experts,
            ffn_rows: ffn,
            hidden_rows: hidden,
            gate_row_bytes: gu_row_bytes,
            down_row_bytes,
            gate_kind: "Q4_K",
            up_kind: "Q4_K",
            down_kind: "Q8_0",
        };
        let mut x_batch = Vec::with_capacity(n_tokens * hidden);
        let mut ids = Vec::with_capacity(n_tokens * top_k);
        let mut route = Vec::with_capacity(n_tokens * top_k);
        let mut expected = Vec::with_capacity(n_tokens * hidden);
        for t in 0..n_tokens {
            let x: Vec<f32> = (0..hidden)
                .map(|i| ((i + t * 7) as f32 * 0.071).sin())
                .collect();
            let e0 = t % n_experts;
            let e1 = (t + 1) % n_experts;
            let w0 = 0.6f32;
            let w1 = 0.4f32;
            ids.push(e0 as i32);
            ids.push(e1 as i32);
            route.push(w0);
            route.push(w1);
            x_batch.extend_from_slice(&x);
            let mut out_t = vec![0f32; hidden];
            for (eid, w) in [(e0, w0), (e1, w1)] {
                let mut act = vec![0f32; ffn];
                let g_base = eid * gate_stride;
                let u_base = eid * gate_stride;
                let d_base = eid * down_stride;
                for r in 0..ffn {
                    let range = g_base + r * gu_row_bytes..g_base + (r + 1) * gu_row_bytes;
                    let g = frink_quant::dot_q4_k_f32_scalar(&gate[range.clone()], &x);
                    let u = frink_quant::dot_q4_k_f32_scalar(
                        &up[u_base + r * gu_row_bytes..u_base + (r + 1) * gu_row_bytes],
                        &x,
                    );
                    act[r] = frink_core_silu(g) * u;
                }
                for (r, slot) in out_t.iter_mut().enumerate() {
                    let range = d_base + r * down_row_bytes..d_base + (r + 1) * down_row_bytes;
                    *slot += w * frink_quant::dot_q8_0_f32_scalar(&down[range], &act);
                }
            }
            expected.extend_from_slice(&out_t);
        }
        let got = launch_moe_prefill_q4_0(&x_batch, n_tokens, &packed, &ids, &route, top_k)
            .expect("prefill mixed MoE");
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &w)) in got.iter().zip(&expected).enumerate() {
            assert_close_relative(g, w, i);
        }
    }

    /// Qwen1.5-MoE-A2.7B scale: 60 experts, H=2048, FFN=1408, top_k=4,
    /// 27-token prefill. Host-routed path (`launch_moe_prefill_q4_0`) because
    /// shared experts disable the fused GPU-router stack.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_moe_prefill_qwen15_moe_scale_host_routed() {
        let hidden = 2048;
        let ffn = 1408;
        let n_experts = 60;
        let top_k = 4;
        let n_tokens = 27;
        let gu_row_bytes = (hidden / frink_quant::Q4_K_BLOCK_ELEMS) * frink_quant::Q4_K_BLOCK_BYTES;
        let down_row_bytes = (ffn / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES;

        fn q8_matrix(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
            let blocks = cols / frink_quant::Q8_0_BLOCK_ELEMS;
            let mut out = Vec::with_capacity(rows * blocks * frink_quant::Q8_0_BLOCK_BYTES);
            for r in 0..rows {
                for b in 0..blocks {
                    out.extend_from_slice(
                        &half::f16::from_f32(
                            0.02 + ((r * blocks + b + seed as usize) % 13) as f32 * 0.004,
                        )
                        .to_le_bytes(),
                    );
                    for i in 0..32u8 {
                        out.push(i.wrapping_add(r as u8).wrapping_add(seed).wrapping_mul(3));
                    }
                }
            }
            out
        }

        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        for e in 0..n_experts {
            gate.extend(pseudo_bytes((e * 3 + 1) as u32, ffn * gu_row_bytes));
            up.extend(pseudo_bytes((e * 3 + 2) as u32, ffn * gu_row_bytes));
            down.extend(q8_matrix(hidden, ffn, (e * 3 + 3) as u8));
        }
        let gate_stride = ffn * gu_row_bytes;
        let down_stride = hidden * down_row_bytes;
        let packed = MoePackedQ4 {
            gate: &gate,
            up: &up,
            down: &down,
            gate_stride,
            up_stride: gate_stride,
            down_stride,
            n_experts,
            ffn_rows: ffn,
            hidden_rows: hidden,
            gate_row_bytes: gu_row_bytes,
            down_row_bytes,
            gate_kind: "Q4_K",
            up_kind: "Q4_K",
            down_kind: "Q8_0",
        };

        let mut x_batch = Vec::with_capacity(n_tokens * hidden);
        let mut ids = Vec::with_capacity(n_tokens * top_k);
        let mut route = Vec::with_capacity(n_tokens * top_k);
        for t in 0..n_tokens {
            x_batch.extend((0..hidden).map(|i| ((i + t * 11) as f32 * 0.0031).sin()));
            for k in 0..top_k {
                ids.push(((t * 3 + k) % n_experts) as i32);
                route.push(1.0 / top_k as f32);
            }
        }

        launch_moe_prefill_q4_0(&x_batch, n_tokens, &packed, &ids, &route, top_k)
            .expect("Qwen-scale host-routed MoE prefill must not fail with CommandFailed");
    }

    fn frink_core_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// The fused-prefill-stack MoE FFN (GPU router + top-k + `mul_mm_id`)
    /// against a host reference that routes with the same softmax top-k.
    /// Covers what the host-routed `launch_moe_prefill_q4_0` test cannot:
    /// `moe_router_mm_f32`, `moe_topk_softmax_batch`, the GPU `map0`, and
    /// the f16-src1 `mul_mm_id` twins.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn encode_moe_prefill_ffn_matches_cpu_routed_reference() {
        fn q4_matrix(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
            let blocks = cols / frink_quant::Q4_0_BLOCK_ELEMS;
            let mut out = Vec::with_capacity(rows * blocks * frink_quant::Q4_0_BLOCK_BYTES);
            for r in 0..rows {
                for b in 0..blocks {
                    out.extend_from_slice(
                        &half::f16::from_f32(
                            0.01 + ((r * blocks + b + seed as usize) % 17) as f32 * 0.003,
                        )
                        .to_le_bytes(),
                    );
                    for i in 0..16u8 {
                        let lo = i.wrapping_add(r as u8).wrapping_add(seed) & 15;
                        let hi = (15u8.wrapping_sub(i))
                            .wrapping_add(b as u8)
                            .wrapping_add(seed)
                            & 15;
                        out.push(lo | (hi << 4));
                    }
                }
            }
            out
        }

        let hidden = 64;
        let ffn = 96;
        let n_experts = 8;
        let top_k = 2;
        let n_tokens = 24;
        let gu_row_bytes = (hidden / frink_quant::Q4_0_BLOCK_ELEMS) * frink_quant::Q4_0_BLOCK_BYTES;
        let down_row_bytes = (ffn / frink_quant::Q4_0_BLOCK_ELEMS) * frink_quant::Q4_0_BLOCK_BYTES;
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        for e in 0..n_experts {
            gate.extend(q4_matrix(ffn, hidden, (e * 3 + 1) as u8));
            up.extend(q4_matrix(ffn, hidden, (e * 3 + 2) as u8));
            down.extend(q4_matrix(hidden, ffn, (e * 3 + 3) as u8));
        }
        let gate_stride = ffn * gu_row_bytes;
        let down_stride = hidden * down_row_bytes;
        let packed = MoePackedQ4 {
            gate: &gate,
            up: &up,
            down: &down,
            gate_stride,
            up_stride: gate_stride,
            down_stride,
            n_experts,
            ffn_rows: ffn,
            hidden_rows: hidden,
            gate_row_bytes: gu_row_bytes,
            down_row_bytes,
            gate_kind: "Q4_0",
            up_kind: "Q4_0",
            down_kind: "Q4_0",
        };
        // Router rows are well separated so top-k never sits on a tie.
        let router_w: Vec<f32> = (0..n_experts * hidden)
            .map(|i| {
                let e = i / hidden;
                (((i % hidden) as f32 * 0.031).cos() + e as f32 * 0.17) * 0.5
            })
            .collect();
        let x_batch: Vec<f32> = (0..n_tokens * hidden)
            .map(|i| ((i as f32) * 0.0137).sin())
            .collect();

        let moe = PrefillMoeMetal {
            router_w: &router_w,
            top_k,
            renormalize: false,
            packed,
        };
        assert!(moe.is_supported());

        // CPU reference: f32 router GEMM, softmax top-k, expert SwiGLU.
        let mut expected = Vec::with_capacity(n_tokens * hidden);
        for t in 0..n_tokens {
            let x = &x_batch[t * hidden..(t + 1) * hidden];
            let logits: Vec<f32> = (0..n_experts)
                .map(|e| {
                    router_w[e * hidden..(e + 1) * hidden]
                        .iter()
                        .zip(x)
                        .map(|(w, v)| w * v)
                        .sum::<f32>()
                })
                .collect();
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
            let mut order: Vec<usize> = (0..n_experts).collect();
            order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

            let mut out_t = vec![0f32; hidden];
            for &eid in &order[..top_k] {
                let w = probs[eid];
                let mut act = vec![0f32; ffn];
                let g_base = eid * gate_stride;
                let d_base = eid * down_stride;
                for (r, slot) in act.iter_mut().enumerate() {
                    let range = g_base + r * gu_row_bytes..g_base + (r + 1) * gu_row_bytes;
                    let g = frink_quant::dot_q4_0_f32_scalar(&gate[range.clone()], x);
                    let u = frink_quant::dot_q4_0_f32_scalar(&up[range], x);
                    *slot = frink_core_silu(g) * u;
                }
                for (r, slot) in out_t.iter_mut().enumerate() {
                    let range = d_base + r * down_row_bytes..d_base + (r + 1) * down_row_bytes;
                    *slot += w * frink_quant::dot_q4_0_f32_scalar(&down[range], &act);
                }
            }
            expected.extend_from_slice(&out_t);
        }

        let shared = shared_metal().expect("metal device");
        let device = &shared.device;
        let queue = &shared.queue;
        let mut x_owned = x_batch.clone();
        let x_buf = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(x_owned.as_mut_ptr() as *mut _).unwrap(),
                x_owned.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("x buffer");
        let xh_buf = device
            .newBufferWithLength_options(
                n_tokens * hidden * 2,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("x f16 buffer");
        let out_buf = device
            .newBufferWithLength_options(
                n_tokens * hidden * 4,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("out buffer");
        let bound = moe_packed_resident(device, &moe.packed).expect("packed resident");
        let router = resident_f32_buffer(device, &router_w).expect("router resident");

        let cmd_buf = queue.commandBuffer().expect("command buffer");
        let encoder = compute_encoder_concurrent(&cmd_buf).expect("encoder");
        crate::elem::encode_f32_to_f16(
            &encoder,
            device,
            &x_buf,
            &xh_buf,
            (n_tokens * hidden) as u32,
        )
        .expect("f32->f16");
        memory_barrier_buffers(&encoder);
        encode_moe_prefill_ffn(
            &encoder,
            &mut crate::mem_ranges::MemRanges::new(),
            device,
            &moe,
            &bound,
            &router,
            &x_buf,
            &xh_buf,
            &out_buf,
            n_tokens,
        )
        .expect("moe prefill ffn");
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let got = unsafe {
            std::slice::from_raw_parts(out_buf.contents().as_ptr() as *const f32, n_tokens * hidden)
        };
        for (i, (&g, &w)) in got.iter().zip(&expected).enumerate() {
            // f16 expert activations: looser than the f32-src1 path.
            let tol = 2e-2 * w.abs().max(1.0);
            assert!((g - w).abs() <= tol, "elem {i}: gpu={g} cpu={w} tol={tol}");
        }
    }

    /// Deterministic pseudo-random byte generator for building real,
    /// non-trivial K-quant block bytes -- no `quantize_qX_k` producer
    /// exists in `frink_quant` (these formats are load-only), the same
    /// convention `frink-cuda`'s equivalent tests use.
    fn pseudo_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    /// GPU-vs-CPU agreement check for the K-quant kernels, using a
    /// *relative* error bound rather than a fixed absolute one -- same
    /// reasoning and same tolerance as `frink-cuda::gpu::tests::assert_close_relative`:
    /// these kernels apply each block's scale/min *inside* a per-element
    /// `acc += (d1 * q - min1) * x[i]` accumulation (hundreds of float
    /// multiply-adds per row), so GPU-vs-CPU results can differ by
    /// float-rounding-order alone (Metal's default fast-math mode may
    /// contract `a*b+c` into a single-rounding fused multiply-add the
    /// same way NVRTC does; plain Rust `f32` arithmetic does not
    /// auto-contract). Also treats NaN==NaN as agreement, for the same
    /// reason the CUDA-side helper does: pseudo-random block bytes can
    /// happen to decode as a NaN/Inf `half` scale, and two backends that
    /// both produce NaN from the same degenerate input have actually
    /// agreed, even though IEEE754 NaN comparisons are always false.
    fn assert_close_relative(got: f32, want: f32, row: usize) {
        if want.is_nan() {
            assert!(
                got.is_nan(),
                "row {row}: CPU reference is NaN but GPU={got} is not"
            );
            return;
        }
        let tol = 1e-4 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "row {row}: GPU={got} CPU reference={want} (relative tolerance {tol})"
        );
    }

    /// Builds `rows` real (non-zero, non-trivial) blocks of `block_bytes`
    /// each for a K-quant format, and the matching `expected` output via
    /// `scalar_dot` (`frink_quant::dot_q{4,5,6}_k_f32_scalar` --
    /// independently verified elsewhere in this workspace), so the
    /// ignored GPU tests below check real numerical agreement with that
    /// trusted CPU reference, not just "the launch didn't error."
    fn real_k_quant_test_matrix(
        rows: usize,
        cols: usize,
        block_bytes: usize,
        scalar_dot: impl Fn(&[u8], &[f32]) -> f32,
    ) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let n_blocks_per_row = cols / 256;
        let row_bytes = n_blocks_per_row * block_bytes;
        let mut weights = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            weights.extend(pseudo_bytes(r as u32 + 1, row_bytes));
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.021).sin()).collect();
        let expected: Vec<f32> = (0..rows)
            .map(|r| scalar_dot(&weights[r * row_bytes..(r + 1) * row_bytes], &x))
            .collect();
        (weights, x, expected)
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_k_matvec_matches_cpu_reference() {
        // 5 rows exercises the multi-row TG (NR=4) plus a partial last group.
        let rows = 5;
        let cols = 512; // 2 Q4_K super-blocks per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            frink_quant::Q4_K_BLOCK_BYTES,
            frink_quant::dot_q4_k_f32_scalar,
        );

        let result = launch_q4_k_matvec(&weights, &x, rows, frink_quant::Q4_K_BLOCK_BYTES * 2)
            .expect("kernel launch");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q5_k_matvec_matches_cpu_reference() {
        // 5 rows exercises NSG=2 (2 rows/TG) plus a partial last group.
        let rows = 5;
        let cols = 512; // 2 Q5_K super-blocks per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            frink_quant::Q5_K_BLOCK_BYTES,
            frink_quant::dot_q5_k_f32_scalar,
        );

        let result = launch_q5_k_matvec(&weights, &x, rows, frink_quant::Q5_K_BLOCK_BYTES * 2)
            .expect("kernel launch");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q6_k_matvec_matches_cpu_reference() {
        let rows = 5; // multi-row TG (NR=4) + partial last group
        let cols = 512; // 2 Q6_K super-blocks per row
        let (weights, x, expected) = real_k_quant_test_matrix(
            rows,
            cols,
            frink_quant::Q6_K_BLOCK_BYTES,
            frink_quant::dot_q6_k_f32_scalar,
        );

        let result = launch_q6_k_matvec(&weights, &x, rows, frink_quant::Q6_K_BLOCK_BYTES * 2)
            .expect("kernel launch");
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*got, *want, i);
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_matvec_batch_matches_sequential_launches() {
        // Multi-x batch (distinct from rows_per_tg multi-row): N
        // activations share one weight matrix / one CB wait.
        let rows = 8;
        let cols = 256;
        let batch_size = 4;
        let row_bytes = (cols / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES;
        let (weights, _x0, _) = real_q8_0_test_matrix(rows, cols);

        let mut x_batch = Vec::with_capacity(batch_size * cols);
        let mut expected = Vec::with_capacity(batch_size * rows);
        for b in 0..batch_size {
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i + b * 17) as f32 * 0.041).sin())
                .collect();
            let y = launch_q8_0_matvec(&weights, &x, rows, row_bytes).expect("single launch");
            expected.extend_from_slice(&y);
            x_batch.extend_from_slice(&x);
        }

        let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
            matvec_launch_meta("Q8_0").expect("Q8_0 meta");
        let launch = MatvecLaunch {
            kernel_src: src,
            fn_name,
            block_bytes,
            block_elems,
            weights: &weights,
            rows,
            row_bytes,
            rows_per_tg,
        };
        let got = launch_matvec_batch(&launch, &x_batch, batch_size).expect("batch launch");
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "elem {i}: batch={a} sequential={b} (diff too large)"
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_k_mul_mm_matches_cpu_matvec() {
        // Q4_K mul_mm vs N× matvec: same dequant identity, so the batched
        // path must reproduce each per-activation matvec.
        let rows = 9;
        let cols = 512; // 2 Q4_K blocks/row
        let batch_size = 7;
        let row_bytes = (cols / 256) * frink_quant::Q4_K_BLOCK_BYTES;
        let (weights, _x0, _) = real_k_quant_test_matrix(
            rows,
            cols,
            frink_quant::Q4_K_BLOCK_BYTES,
            frink_quant::dot_q4_k_f32_scalar,
        );

        let mut x_batch = Vec::with_capacity(batch_size * cols);
        let mut expected = Vec::with_capacity(batch_size * rows);
        for b in 0..batch_size {
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i + b * 23) as f32 * 0.027).sin())
                .collect();
            let y = launch_q4_k_matvec(&weights, &x, rows, row_bytes).expect("matvec");
            expected.extend_from_slice(&y);
            x_batch.extend_from_slice(&x);
        }

        let got =
            launch_q4_k_mul_mm(&weights, &x_batch, rows, row_bytes, batch_size).expect("mul_mm");
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*a, *b, i);
        }
    }

    /// Q6_K counterpart of [`realistic_q4_k_matrix`]: the block's `d` half
    /// (bytes 208..210) is forced to a realistic magnitude so the value
    /// range matches what the format actually produces. Left random, it
    /// dotted to -2.4e8 and overflowed the GEMM's half tiles to NaN.
    fn realistic_q6_k_matrix(rows: usize, cols: usize) -> Vec<u8> {
        let n_blocks_per_row = cols / 256;
        let row_bytes = n_blocks_per_row * frink_quant::Q6_K_BLOCK_BYTES;
        let mut weights = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            let mut row = pseudo_bytes(r as u32 + 7, row_bytes);
            for b in 0..n_blocks_per_row {
                let d_off = b * frink_quant::Q6_K_BLOCK_BYTES + 208;
                let d = half::f16::from_f32(0.0015 + 0.0004 * ((r + b) % 4) as f32);
                row[d_off..d_off + 2].copy_from_slice(&d.to_le_bytes());
            }
            weights.extend_from_slice(&row);
        }
        weights
    }

    /// Q6_K twin of the Q4_K GEMM check. Same reasoning: shapes chosen to
    /// include ones that are not multiples of the 64x32 tile.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q6_k_mul_mm_sg_matches_the_matvec_it_replaces() {
        for &(rows, cols, batch_size) in &[(64usize, 512usize, 32usize), (9, 512, 7), (70, 768, 33)]
        {
            let row_bytes = (cols / 256) * frink_quant::Q6_K_BLOCK_BYTES;
            let weights = realistic_q6_k_matrix(rows, cols);

            let mut x_batch = Vec::with_capacity(batch_size * cols);
            let mut expected = Vec::with_capacity(batch_size * rows);
            for b in 0..batch_size {
                let x: Vec<f32> = (0..cols)
                    .map(|i| ((i + b * 23) as f32 * 0.027).sin())
                    .collect();
                let y = launch_q6_k_matvec(&weights, &x, rows, row_bytes).expect("matvec");
                expected.extend_from_slice(&y);
                x_batch.extend_from_slice(&x);
            }

            let got = launch_q6_k_mul_mm_sg(&weights, &x_batch, rows, row_bytes, batch_size)
                .expect("simdgroup q6_k mul_mm");
            assert_eq!(got.len(), expected.len(), "{rows}x{cols}x{batch_size}");
            let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
            let tol = 1e-3 * scale.max(1.0);
            for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol,
                    "{rows}x{cols}x{batch_size} idx {i}: gemm={a} matvec={b} (tol {tol})"
                );
            }
        }
    }

    /// Q4_K blocks whose scale halves are realistic in magnitude.
    ///
    /// `real_k_quant_test_matrix` fills every byte pseudo-randomly, which
    /// lets `d`/`dmin` decode to enormous f16 values (one row dotted to
    /// 1.5e8). An f32 matvec absorbs that; the simdgroup GEMM stages its
    /// weight tile in `half` and overflows to NaN. Real Q4_K weights are
    /// O(0.1), so the fixture was testing a regime the format never
    /// produces. Quant nibbles and the packed 6-bit scale bytes stay
    /// pseudo-random -- that is the part worth exercising.
    fn realistic_q4_k_matrix(rows: usize, cols: usize) -> Vec<u8> {
        let n_blocks_per_row = cols / 256;
        let row_bytes = n_blocks_per_row * frink_quant::Q4_K_BLOCK_BYTES;
        let mut weights = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            let mut row = pseudo_bytes(r as u32 + 1, row_bytes);
            for b in 0..n_blocks_per_row {
                let base = b * frink_quant::Q4_K_BLOCK_BYTES;
                let d = half::f16::from_f32(0.008 + 0.001 * ((r + b) % 5) as f32);
                let dmin = half::f16::from_f32(0.003 + 0.0005 * ((r + b) % 3) as f32);
                row[base..base + 2].copy_from_slice(&d.to_le_bytes());
                row[base + 2..base + 4].copy_from_slice(&dmin.to_le_bytes());
            }
            weights.extend_from_slice(&row);
        }
        weights
    }

    /// The simdgroup GEMM must agree with the matvec it replaces. This is
    /// the assertion the previous simdgroup attempt could not make -- its
    /// dequant did not match, so it was left returning `Err`.
    ///
    /// Dimensions are deliberately *not* multiples of the 64x32 tile, so
    /// the partial-tile store path is exercised too; a kernel that only
    /// handles whole tiles silently corrupts the edges of a real prompt.
    /// The exact-tile pipeline may only be picked when *both* dst dims tile
    /// exactly: it has no ragged-edge store path, so choosing it for a
    /// ragged shape writes past the matrix. Pure selection logic, no GPU.
    #[test]
    fn mul_mm_sg_variant_picks_the_small_threadgroup_only_on_exact_tiles() {
        assert_eq!(
            mul_mm_sg_variant("q4_k_mul_mm_sg", 2048, 512),
            ("q4_k_mul_mm_sg_a", MUL_MM_SG_SMEM_ALIGNED)
        );
        assert_eq!(
            mul_mm_sg_variant("q4_k_mul_mm_sg_f16", 8192, 64),
            ("q4_k_mul_mm_sg_f16_a", MUL_MM_SG_SMEM_ALIGNED)
        );
        // Ragged in exactly one dim is still ragged.
        for (rows, batch) in [(2048usize, 513usize), (2050, 512), (70, 33), (0, 0)] {
            let (name, smem) = mul_mm_sg_variant("q4_k_mul_mm_sg", rows, batch);
            if rows.is_multiple_of(64) && batch.is_multiple_of(32) {
                continue;
            }
            assert_eq!(name, "q4_k_mul_mm_sg", "{rows}x{batch}");
            assert_eq!(smem, MUL_MM_SG_SMEM_BC, "{rows}x{batch}");
        }
        // A kernel with no `_a` sibling must fall back, not be renamed.
        assert_eq!(
            mul_mm_sg_variant("q4_k_mul_mm_id", 2048, 512),
            ("q4_k_mul_mm_id", MUL_MM_SG_SMEM_BC)
        );
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_k_mul_mm_sg_matches_the_matvec_it_replaces() {
        for &(rows, cols, batch_size) in &[
            // Exact tile grids take the `_a` (bc_out=false) pipeline, which
            // compiles the ragged-edge staging arm out and drops the
            // threadgroup allocation to 6144 -- so these two shapes are the
            // only coverage that kernel gets. 128x64 gives it more than one
            // tile in both dims, where a wrong epilogue guard would show.
            (64usize, 512usize, 32usize), // exactly one tile
            (128, 512, 64),               // exact grid, 2x2 tiles
            (9, 512, 7),                  // smaller than a tile in both dims
            (70, 768, 33),                // one full tile plus a ragged edge
        ] {
            let row_bytes = (cols / 256) * frink_quant::Q4_K_BLOCK_BYTES;
            let weights = realistic_q4_k_matrix(rows, cols);

            let mut x_batch = Vec::with_capacity(batch_size * cols);
            let mut expected = Vec::with_capacity(batch_size * rows);
            for b in 0..batch_size {
                let x: Vec<f32> = (0..cols)
                    .map(|i| ((i + b * 23) as f32 * 0.027).sin())
                    .collect();
                let y = launch_q4_k_matvec(&weights, &x, rows, row_bytes).expect("matvec");
                expected.extend_from_slice(&y);
                x_batch.extend_from_slice(&x);
            }

            let got = launch_q4_k_mul_mm_sg(&weights, &x_batch, rows, row_bytes, batch_size)
                .expect("simdgroup mul_mm");
            assert_eq!(got.len(), expected.len(), "{rows}x{cols}x{batch_size}");

            // Tolerance is absolute and scaled to the magnitude of the
            // data, not relative per element. The GEMM stages its tiles in
            // `half`, so its error tracks the size of the terms being
            // summed -- not the size of the result, which can cancel to
            // near zero. Measured on this fixture: max |expected| 447,
            // worst absolute deviation 0.081, i.e. 1.8e-4 of scale. A
            // per-element relative check flags those cancelled entries at
            // 950% while a genuine indexing bug -- which moves results by
            // order-of-magnitude the data itself -- would sail past it.
            let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
            let tol = 1e-3 * scale.max(1.0);
            for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol,
                    "{rows}x{cols}x{batch_size} idx {i}: gemm={a} matvec={b} (tol {tol})"
                );
            }
        }
    }

    /// Pseudo-random blocks whose leading `half` scales are forced to a
    /// realistic magnitude, for the formats that keep `d` (and optionally
    /// `dmin`) at the front of the block: Q8_0, Q4_0, Q5_K, IQ4_XS. Same
    /// reasoning as [`realistic_q4_k_matrix`] — random scale halves decode
    /// to values the format never produces and overflow the GEMM's `half`
    /// tiles to NaN, so the test would be exercising a regime that cannot
    /// occur rather than the indexing that can actually be wrong.
    fn realistic_blocks(
        rows: usize,
        blocks_per_row: usize,
        block_bytes: usize,
        seed: u32,
        d: f32,
        dmin: Option<f32>,
    ) -> Vec<u8> {
        let row_bytes = blocks_per_row * block_bytes;
        let mut weights = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            let mut row = pseudo_bytes(r as u32 + seed, row_bytes);
            for b in 0..blocks_per_row {
                let base = b * block_bytes;
                let dv = half::f16::from_f32(d * (1.0 + 0.1 * ((r + b) % 5) as f32));
                row[base..base + 2].copy_from_slice(&dv.to_le_bytes());
                if let Some(m) = dmin {
                    let mv = half::f16::from_f32(m * (1.0 + 0.1 * ((r + b) % 3) as f32));
                    row[base + 2..base + 4].copy_from_slice(&mv.to_le_bytes());
                }
            }
            weights.extend_from_slice(&row);
        }
        weights
    }

    /// Shared body for "the new simdgroup GEMM agrees with the matvec it
    /// replaces", parameterized by format. Shapes deliberately include
    /// ones that are not multiples of the 64x32 tile so the partial-tile
    /// store path is exercised: a kernel that only handles whole tiles
    /// corrupts the edges of a real prompt silently.
    #[allow(clippy::too_many_arguments)]
    fn assert_mul_mm_sg_matches_matvec(
        label: &str,
        block_elems: usize,
        block_bytes: usize,
        seed: u32,
        d: f32,
        dmin: Option<f32>,
        matvec: impl Fn(&[u8], &[f32], usize, usize) -> Result<Vec<f32>, MetalError>,
        gemm: impl Fn(&[u8], &[f32], usize, usize, usize) -> Result<Vec<f32>, MetalError>,
    ) {
        for &(rows, cols, batch_size) in &[
            // Exact tile grids take the `_a` (bc_out=false) pipeline, which
            // compiles the ragged-edge staging arm out and drops the
            // threadgroup allocation to 6144 -- so these two shapes are the
            // only coverage that kernel gets. 128x64 gives it more than one
            // tile in both dims, where a wrong epilogue guard would show.
            (64usize, 512usize, 32usize), // exactly one tile
            (128, 512, 64),               // exact grid, 2x2 tiles
            (9, 512, 7),                  // smaller than a tile in both dims
            (70, 768, 33),                // one full tile plus a ragged edge
        ] {
            let blocks_per_row = cols / block_elems;
            let row_bytes = blocks_per_row * block_bytes;
            let weights = realistic_blocks(rows, blocks_per_row, block_bytes, seed, d, dmin);

            let mut x_batch = Vec::with_capacity(batch_size * cols);
            let mut expected = Vec::with_capacity(batch_size * rows);
            for b in 0..batch_size {
                let x: Vec<f32> = (0..cols)
                    .map(|i| ((i + b * 23) as f32 * 0.027).sin())
                    .collect();
                let y = matvec(&weights, &x, rows, row_bytes).expect("matvec");
                expected.extend_from_slice(&y);
                x_batch.extend_from_slice(&x);
            }

            let got = gemm(&weights, &x_batch, rows, row_bytes, batch_size).expect("mul_mm_sg");
            assert_eq!(
                got.len(),
                expected.len(),
                "{label} {rows}x{cols}x{batch_size}"
            );
            let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
            let tol = 1e-3 * scale.max(1.0);
            for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol,
                    "{label} {rows}x{cols}x{batch_size} idx {i}: gemm={a} matvec={b} (tol {tol})"
                );
            }
        }
    }

    /// Q8_0 is a 32-element format, so it runs the shared GEMM body with
    /// `NL = 2` — the path where `il` never advances and every iteration
    /// steps a whole block. That is exactly the arithmetic this test
    /// exists to pin down.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q8_0_mul_mm_sg_matches_the_matvec_it_replaces() {
        assert_mul_mm_sg_matches_matvec(
            "q8_0",
            32,
            34,
            11,
            0.01,
            None,
            launch_q8_0_matvec,
            launch_q8_0_mul_mm_sg,
        );
    }

    /// Q4_0: the other `NL = 2` format, and the one whose dequant packs
    /// two values per byte, so a wrong nibble mask shows up here.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_0_mul_mm_sg_matches_the_matvec_it_replaces() {
        assert_mul_mm_sg_matches_matvec(
            "q4_0",
            32,
            18,
            13,
            0.02,
            None,
            launch_q4_0_matvec,
            launch_q4_0_mul_mm_sg,
        );
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q5_k_mul_mm_sg_matches_the_matvec_it_replaces() {
        assert_mul_mm_sg_matches_matvec(
            "q5_k",
            256,
            176,
            17,
            0.008,
            Some(0.003),
            launch_q5_k_matvec,
            launch_q5_k_mul_mm_sg,
        );
    }

    /// IQ4_XS reads its values from the IQ4 codebook rather than an affine
    /// dequant, so this is the first codebook format on the batched path.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_iq4_xs_mul_mm_sg_matches_the_matvec_it_replaces() {
        assert_mul_mm_sg_matches_matvec(
            "iq4_xs",
            256,
            136,
            19,
            0.0002,
            None,
            launch_iq4_xs_matvec,
            launch_iq4_xs_mul_mm_sg,
        );
    }

    /// PTQ1_0's GEMM against its matvec (which the test above pins to
    /// the scalar reference). `realistic_blocks` writes the scale at
    /// byte 0 and PTQ1_0 keeps it at the END of the block, so the
    /// fixture is a real `quantize_trits` row rather than pseudo-bytes;
    /// the 128-element block makes NL = 8, the widest the shared body
    /// runs, and the il = 5, 6 (16-weight stage) and il = 7 (8-weight
    /// stage) arms of `ptq1_0_dequant_16` are reached only here.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_ptq1_0_mul_mm_sg_matches_the_matvec_it_replaces() {
        use frink_quant::ternary::{quantize_trits, PTQ1_0};
        for &(rows, cols, batch_size) in &[
            (64usize, 512usize, 32usize),
            (128, 512, 64),
            (9, 512, 7),
            (70, 768, 33),
        ] {
            let row_bytes = cols / 128 * 28;
            let mut seed = 5u32 + rows as u32;
            let mut weights = Vec::with_capacity(rows * row_bytes);
            for _ in 0..rows {
                let w: Vec<f32> = (0..cols)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        match (seed >> 24) % 3 {
                            0 => -0.5,
                            1 => 0.0,
                            _ => 0.5,
                        }
                    })
                    .collect();
                weights.extend(quantize_trits(&w, PTQ1_0));
            }
            let mut x_batch = Vec::with_capacity(batch_size * cols);
            let mut expected = Vec::with_capacity(batch_size * rows);
            for b in 0..batch_size {
                let x: Vec<f32> = (0..cols)
                    .map(|i| ((i + b * 23) as f32 * 0.027).sin())
                    .collect();
                expected
                    .extend(launch_ptq1_0_matvec(&weights, &x, rows, row_bytes).expect("matvec"));
                x_batch.extend_from_slice(&x);
            }
            let got = launch_ptq1_0_mul_mm_sg(&weights, &x_batch, rows, row_bytes, batch_size)
                .expect("mul_mm_sg");
            assert_eq!(got.len(), expected.len());
            let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
            let tol = 1e-3 * scale.max(1.0);
            for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol,
                    "ptq1_0 {rows}x{cols}x{batch_size} idx {i}: gemm={a} matvec={b} (tol {tol})"
                );
            }
        }
    }

    /// GEMM throughput at Bonsai's FFN shapes and a prefill batch, in
    /// TFLOP/s, printed rather than asserted: run with `--nocapture`.
    ///
    /// The question it answers is whether the PTQ1_0 GEMM is slow or
    /// the shared simdgroup body is: Q4_K runs the SAME body with a
    /// different dequant functor, so the two numbers side by side say
    /// which. An M2 Pro peaks near 6.8 TFLOP/s in f32.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn gemm_throughput_probe() {
        let batch = 128usize;
        for &(rows, cols) in &[(17408usize, 5120usize), (5120usize, 17408usize)] {
            let x: Vec<f32> = (0..batch * cols)
                .map(|i| (i as f32 * 0.001).sin())
                .collect();
            let flops = 2.0 * rows as f64 * cols as f64 * batch as f64;

            let ptq_row_bytes = cols / 128 * 28;
            let ptq = pseudo_bytes(3, rows * ptq_row_bytes);
            launch_ptq1_0_mul_mm_sg(&ptq, &x, rows, ptq_row_bytes, batch).expect("warm");
            let n = 10;
            let t = std::time::Instant::now();
            for _ in 0..n {
                launch_ptq1_0_mul_mm_sg(&ptq, &x, rows, ptq_row_bytes, batch).expect("ptq");
            }
            let ptq_s = t.elapsed().as_secs_f64() / n as f64;

            let q4k_row_bytes = cols / 256 * 144;
            let q4k = realistic_blocks(rows, cols / 256, 144, 5, 0.01, Some(0.003));
            launch_q4_k_mul_mm_sg(&q4k, &x, rows, q4k_row_bytes, batch).expect("warm");
            let t = std::time::Instant::now();
            for _ in 0..n {
                launch_q4_k_mul_mm_sg(&q4k, &x, rows, q4k_row_bytes, batch).expect("q4k");
            }
            let q4k_s = t.elapsed().as_secs_f64() / n as f64;

            eprintln!(
                "{rows}x{cols} b{batch}: PTQ1_0 {:.2} ms ({:.2} TFLOP/s) | Q4_K {:.2} ms ({:.2} TFLOP/s)",
                ptq_s * 1e3,
                flops / ptq_s / 1e12,
                q4k_s * 1e3,
                flops / q4k_s / 1e12,
            );
        }
    }

    /// Achieved weight bandwidth of the PTQ1_0 matvec on Bonsai's two
    /// FFN shapes, printed, not asserted: run with `--nocapture`.
    ///
    /// # What this measures, and what it does not
    ///
    /// Wall time around `launch_*`, which is a command buffer's worth
    /// of HOST cost -- an output allocation, an activation upload, a
    /// commit and a wait -- plus the kernel. On the small shapes that
    /// host cost is most of the number: the `64x512` row is almost
    /// entirely it, which is why it is measured FIRST and subtracted
    /// from the rest as `net`.
    ///
    /// Even net, this is one matvec in isolation and the production
    /// path reads 65 GB/s on the same matrices where this reads 21. So
    /// the ONLY sound use of these numbers is comparing one variant of
    /// the kernel against another in the same run. Three kernel
    /// hypotheses were aimed at the raw figure before that was
    /// understood (`docs/plans/gdn-resident-state.md`), which is the
    /// reason for this comment.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn ptq1_0_matvec_bandwidth_probe() {
        // The fixed per-launch cost, measured in THIS run on a shape
        // whose kernel is negligible, so the rows below can be reported
        // net of it.
        let mut fixed = 0.0f64;
        for &(rows, cols) in &[
            (64usize, 512usize),
            (17408usize, 5120usize),
            (5120usize, 17408usize),
            (248320usize, 5120usize),
        ] {
            let row_bytes = cols / 128 * 28;
            let weights = pseudo_bytes(3, rows * row_bytes);
            let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).sin()).collect();
            launch_ptq1_0_matvec(&weights, &x, rows, row_bytes).expect("warm");
            let n = 20;
            let t = std::time::Instant::now();
            for _ in 0..n {
                launch_ptq1_0_matvec(&weights, &x, rows, row_bytes).expect("kernel");
            }
            let per = t.elapsed().as_secs_f64() / n as f64;
            let q4_row_bytes = cols / 32 * 18;
            let q4 = pseudo_bytes(4, rows * q4_row_bytes);
            launch_q4_0_matvec(&q4, &x, rows, q4_row_bytes).expect("warm");
            let t = std::time::Instant::now();
            for _ in 0..n {
                launch_q4_0_matvec(&q4, &x, rows, q4_row_bytes).expect("kernel");
            }
            let per_q4 = t.elapsed().as_secs_f64() / n as f64;
            if rows == 64 {
                // The smallest shape IS the per-launch cost.
                fixed = per.min(per_q4);
                eprintln!(
                    "ptq1_0 per-launch host cost: {:.3} ms (subtracted below)",
                    fixed * 1e3
                );
                continue;
            }
            let net = (per - fixed).max(1e-9);
            let net_q4 = (per_q4 - fixed).max(1e-9);
            eprintln!(
                "ptq1_0 {rows}x{cols}: {:.3} ms raw, {:.3} net, {:.1} GB/s | q4_0: {:.3} raw, \
                 {:.3} net, {:.1} GB/s",
                per * 1e3,
                net * 1e3,
                (rows * row_bytes) as f64 / net / 1e9,
                per_q4 * 1e3,
                net_q4 * 1e3,
                (rows * q4_row_bytes) as f64 / net_q4 / 1e9
            );
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn launch_q4_0_mul_mm_matches_cpu_matvec() {
        // Q4_0 mul_mm vs N× matvec. Finite f16 scales (same construction as
        // `launch_q4_0_matvec_matches_cpu_reference`) — raw pseudo_bytes can
        // decode to NaN halves and only exercise NaN==NaN agreement.
        let rows = 67;
        let cols = 320; // 10 Q4_0 blocks/row
        let batch_size = 9;
        let blocks_per_row = cols / frink_quant::Q4_0_BLOCK_ELEMS;
        let row_bytes = blocks_per_row * frink_quant::Q4_0_BLOCK_BYTES;

        let mut weights = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            for b in 0..blocks_per_row {
                weights.extend_from_slice(
                    &half::f16::from_f32(0.05 + (r * blocks_per_row + b) as f32 * 0.01)
                        .to_le_bytes(),
                );
                for i in 0..16u8 {
                    let lo = (i + r as u8 + b as u8) % 16;
                    let hi = (15 - i + r as u8 + b as u8) % 16;
                    weights.push(lo | (hi << 4));
                }
            }
        }

        let mut x_batch = Vec::with_capacity(batch_size * cols);
        let mut expected = Vec::with_capacity(batch_size * rows);
        for b in 0..batch_size {
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i + b * 23) as f32 * 0.027).sin())
                .collect();
            let y = launch_q4_0_matvec(&weights, &x, rows, row_bytes).expect("matvec");
            expected.extend_from_slice(&y);
            x_batch.extend_from_slice(&x);
        }

        let got =
            launch_q4_0_mul_mm(&weights, &x_batch, rows, row_bytes, batch_size).expect("mul_mm");
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_close_relative(*a, *b, i);
        }
    }
}
