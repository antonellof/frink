//! The chunked gated delta rule on the device.
//!
//! `frink_core::gdn_chunk` is the definition and the oracle: same
//! unrolling of the rank-one updates across a chunk, same products of
//! decays (never quotients), same forward substitution. What changes is
//! where it runs, and why THAT is now worth doing:
//!
//! - The row-at-a-time recurrence is bandwidth-bound (36 GB/s of state),
//!   and moving it to the GPU lost three ways
//!   (`docs/plans/gdn-resident-state.md`).
//! - Chunking trades 1.5x the multiply-adds for a 32nd of that traffic,
//!   so the CPU version is now COMPUTE-bound, measured at ~20 GFLOP/s
//!   across six cores and 47% of a Bonsai prefill step.
//!
//! Compute-dense work with its traffic already amortised is what a GPU
//! is for, so the same algebra moves over and one layer's whole prefill
//! recurrence becomes a single command buffer.
//!
//! One threadgroup per value head, one thread per state ROW. The
//! chunk's keys, queries and updates live in threadgroup memory, which
//! is what bounds the chunk width here (`CHUNK`, 16, against the CPU's
//! 32): three `C x head_dim` tiles have to fit.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};

use crate::gdn::{DeltaShape, HeadMapKind};
use crate::gpu::{ensure_pipeline, MetalError};

/// Rows per chunk on the device.
///
/// Three `CHUNK x head_dim` tiles of threadgroup memory (keys, queries,
/// updates) plus the `CHUNK x CHUNK` triangles have to fit in a
/// threadgroup's 32 KiB: at `head_dim` 128 that is 24 KiB of tiles for
/// 16 rows, and 48 KiB for 32. The CPU path chunks by 32 because its
/// limit is cache rather than threadgroup memory.
pub const CHUNK: usize = 16;

/// The widest head this kernel serves, one thread per state row.
///
/// Three `CHUNK x MAX_HEAD_DIM` tiles is 24 KiB at this width, and the
/// three `CHUNK x CHUNK` triangles 3 KiB, against a threadgroup's 32
/// KiB. Raising it means shrinking `CHUNK`, so the two are one arithmetic
/// and a test holds it.
pub const MAX_HEAD_DIM: usize = 128;

pub const GDN_CHUNK_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define CHUNK 16

// One threadgroup per value head, one thread per state row `j`.
//
//   M[t] = S_0[j] . k_t          (registers, one pass over the row)
//   N[t] = S_0[j] . q_t
//   pred = A_t M[t] + sum_{u<t} R[t][u] (k_u . k_t) d[u][j]
//   d[t][j] = beta_t (v_t[j] - pred)
//   out_t[j] = (A_t N[t] + sum_{u<=t} R[t][u] (k_u . q_t) d[u][j]) / sqrt(S)
//   S_C[j] = A_C S_0[j] + sum_t R[C][t] d[t][j] k_t
//
// which is `frink_core::gdn_chunk::delta_chunk` line for line, with
// the `j` loop spread over threads and the `t` loop kept sequential
// behind threadgroup barriers, because that is the recurrence.
kernel void gdn_chunk_step(
    device float* state [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device const float* g [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],
    constant uint4& span [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]
) {
    // dims: n_k_heads, n_v_heads, head_dim, map flag.
    // span: first row of this chunk, rows in it, total rows, unused.
    const uint n_k = dims.x;
    const uint n_v = dims.y;
    const uint S   = dims.z;
    const uint start = span.x;
    const uint c     = span.y;
    const uint rows  = span.z;

    const uint h = tgid;
    if (h >= n_v || tid >= S) {
        return;
    }
    const uint kh = (dims.w == 0u) ? (h % n_k) : (h / (n_v / n_k));
    const uint key_row = n_k * S;
    const uint val_row = n_v * S;

    threadgroup float ks[CHUNK * 128];
    threadgroup float qs[CHUNK * 128];
    threadgroup float ds[CHUNK * 128];
    threadgroup float gram[CHUNK * CHUNK];
    threadgroup float qk[CHUNK * CHUNK];
    threadgroup float ratio[CHUNK * CHUNK];
    threadgroup float acc[CHUNK];

    // The chunk's key and query tiles, loaded once for the whole head,
    // and ZERO-PADDED to the full `CHUNK` when the tail is short.
    //
    // The padding is what lets every hot loop below run to the
    // compile-time `CHUNK` rather than to the runtime `c`: a loop whose
    // trip count the compiler cannot see does not unroll, and an array
    // indexed by a variable it cannot unroll is spilled out of
    // registers into device memory. That spill measured 50 GFLOP/s on a
    // 6.8 TFLOP/s part.
    for (uint idx = tid; idx < CHUNK * S; idx += tcount) {
        const uint t = idx / S;
        const uint i = idx % S;
        const bool live = t < c;
        ks[t * S + i] = live ? k[(start + t) * key_row + kh * S + i] : 0.0f;
        qs[t * S + i] = live ? q[(start + t) * key_row + kh * S + i] : 0.0f;
        ds[t * S + i] = 0.0f;
    }
    for (uint idx = tid; idx < CHUNK * CHUNK; idx += tcount) {
        ratio[idx] = 0.0f;
        gram[idx] = 0.0f;
        qk[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Decay products: `R[t][u]` for `u <= t`, and `A_t = R[t][-1]`.
    // Every entry is a product and never a quotient, so a chunk of tiny
    // decays underflows to zero rather than dividing by it.
    if (tid == 0) {
        for (uint t = 0; t < c; ++t) {
            ratio[t * CHUNK + t] = 1.0f;
            for (int u = int(t) - 1; u >= 0; --u) {
                ratio[t * CHUNK + u] =
                    ratio[t * CHUNK + u + 1] * exp(g[(start + u + 1) * n_v + h]);
            }
            const float decay = exp(g[(start + t) * n_v + h]);
            acc[t] = (t == 0) ? decay : acc[t - 1] * decay;
        }
    }
    // The two triangles over the chunk's own vectors.
    for (uint idx = tid; idx < c * c; idx += tcount) {
        const uint t = idx / c;
        const uint u = idx % c;
        if (u > t) {
            continue;
        }
        float sk = 0.0f;
        float sq = 0.0f;
        for (uint i = 0; i < S; ++i) {
            sk += ks[t * S + i] * ks[u * S + i];
            sq += qs[t * S + i] * ks[u * S + i];
        }
        gram[t * CHUNK + u] = sk;
        qk[t * CHUNK + u] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // This thread's state row, and its two reductions against the whole
    // chunk from one pass over it.
    device float* row = state + (size_t)h * S * S + (size_t)tid * S;
    float m[CHUNK];
    float n[CHUNK];
    #pragma unroll
    for (uint t = 0; t < CHUNK; ++t) {
        m[t] = 0.0f;
        n[t] = 0.0f;
    }
    for (uint i = 0; i < S; ++i) {
        const float r = row[i];
        #pragma unroll
        for (uint t = 0; t < CHUNK; ++t) {
            m[t] += r * ks[t * S + i];
            n[t] += r * qs[t * S + i];
        }
    }

    const float scale = 1.0f / sqrt(float(S));
    // The recurrence: sequential in `t`, parallel in `j`.
    for (uint t = 0; t < c; ++t) {
        float pred = acc[t] * m[t];
        float read = acc[t] * n[t];
        for (uint u = 0; u < t; ++u) {
            const float w = ratio[t * CHUNK + u];
            const float du = ds[u * S + tid];
            pred += w * gram[t * CHUNK + u] * du;
            read += w * qk[t * CHUNK + u] * du;
        }
        const float b = beta[(start + t) * n_v + h];
        const float dj = b * (v[(start + t) * val_row + h * S + tid] - pred);
        ds[t * S + tid] = dj;
        out[(start + t) * val_row + h * S + tid] =
            (read + qk[t * CHUNK + t] * dj) * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The state, updated once for the whole chunk.
    const float a_last = acc[c - 1];
    float w[CHUNK];
    #pragma unroll
    for (uint t = 0; t < CHUNK; ++t) {
        // Zero past the tail: `ratio` and `ds` were padded, so a short
        // chunk contributes nothing through the padded rows.
        w[t] = ratio[(c - 1) * CHUNK + t] * ds[t * S + tid];
    }
    for (uint i = 0; i < S; ++i) {
        float r = row[i] * a_last;
        #pragma unroll
        for (uint t = 0; t < CHUNK; ++t) {
            r += w[t] * ks[t * S + i];
        }
        row[i] = r;
    }
    // `rows` is carried for the caller's bounds and unused here.
    (void) rows;
}
"#;

/// Every chunk of `rows` tokens through the delta rule, in ONE command
/// buffer: the state stays on the device between chunks, and the
/// dispatches are ordered because the recurrence is.
///
/// `state` is read and written in place; the caller's page-aligned
/// allocation is wrapped without a copy
/// (`frink_core::recurrent_state::AlignedF32`, `crate::gdn::buffer_no_copy`).
///
/// # Safety
///
/// `state_ptr` must point at `state_bytes` readable-writable bytes that
/// outlive the call, and both must be page-aligned. The caller must
/// hold the allocation exclusively across the call, which waits for the
/// GPU before returning, because the GPU writes those very bytes rather
/// than a copy of them.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_delta_chunk(
    shape: DeltaShape,
    rows: usize,
    state_ptr: *mut f32,
    state_bytes: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>, MetalError> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    if shape.head_dim == 0
        || shape.head_dim > MAX_HEAD_DIM
        || shape.n_k_heads == 0
        || shape.n_v_heads == 0
        || !shape.n_v_heads.is_multiple_of(shape.n_k_heads)
    {
        return Err(MetalError::CommandFailed);
    }
    let key_row = shape.n_k_heads * shape.head_dim;
    let val_row = shape.n_v_heads * shape.head_dim;
    if q.len() != rows * key_row
        || k.len() != rows * key_row
        || v.len() != rows * val_row
        || g.len() != rows * shape.n_v_heads
        || beta.len() != rows * shape.n_v_heads
    {
        return Err(MetalError::CommandFailed);
    }

    let shared = crate::gpu::shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;
    let upload = |xs: &[f32]| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let mut owned = xs.to_vec();
        // SAFETY: `owned` outlives the call and the buffer copies it.
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(owned.as_mut_ptr() as *mut _).ok_or(MetalError::BufferAllocFailed)?,
                std::mem::size_of_val(owned.as_slice()),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocFailed)
    };
    // SAFETY: the caller's contract (page-aligned, exclusively borrowed,
    // outlives the call), forwarded.
    let state_buf = unsafe { crate::gdn::buffer_no_copy(device, state_ptr, state_bytes) }
        .ok_or(MetalError::BufferAllocFailed)?;
    let (q_buf, k_buf, v_buf) = (upload(q)?, upload(k)?, upload(v)?);
    let (g_buf, beta_buf) = (upload(g)?, upload(beta)?);
    let out_buf = device
        .newBufferWithLength_options(rows * val_row * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;

    let pipe = ensure_pipeline(device, GDN_CHUNK_KERNEL_SRC, "gdn_chunk_step")?;
    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encoder.setComputePipelineState(&pipe.0);
    let threads = shape.head_dim.min(pipe.0.maxTotalThreadsPerThreadgroup());
    let mut start = 0;
    while start < rows {
        let c = CHUNK.min(rows - start);
        unsafe {
            for (idx, buf) in [
                &*state_buf,
                &*q_buf,
                &*k_buf,
                &*v_buf,
                &*g_buf,
                &*beta_buf,
                &*out_buf,
            ]
            .into_iter()
            .enumerate()
            {
                encoder.setBuffer_offset_atIndex(Some(buf), 0, idx);
            }
            let mut dims: [u32; 4] = [
                shape.n_k_heads as u32,
                shape.n_v_heads as u32,
                shape.head_dim as u32,
                match shape.map {
                    HeadMapKind::Tiled => 0,
                    HeadMapKind::Grouped => 1,
                },
            ];
            encoder.setBytes_length_atIndex(
                NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(),
                16,
                7,
            );
            let mut span: [u32; 4] = [start as u32, c as u32, rows as u32, 0];
            encoder.setBytes_length_atIndex(
                NonNull::new(span.as_mut_ptr() as *mut _).unwrap(),
                16,
                8,
            );
        }
        // One dispatch per chunk, in order: the chunks ARE the
        // recurrence, and a serial encoder runs them back to back
        // against the same state without a submission between.
        crate::dispatch::dispatch_counted(
            &encoder,
            MTLSize {
                width: shape.n_v_heads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        start += c;
    }
    encoder.endEncoding();
    let clock = crate::timing::SubmitClock::start();
    crate::timing::commit_wait_note(&cmd_buf, "gdn-chunk", 16, clock);

    // SAFETY: shared storage of exactly `rows * val_row` floats, written
    // by kernels this call has waited for.
    unsafe {
        let o =
            std::slice::from_raw_parts(out_buf.contents().as_ptr() as *const f32, rows * val_row);
        Ok(o.to_vec())
    }
}
