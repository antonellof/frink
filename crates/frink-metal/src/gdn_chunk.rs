//! The gated delta rule over a batch of rows, on the device.
//!
//! The host chunks it (`frink_core::gdn_chunk`): the row-at-a-time
//! recurrence is bandwidth-bound on a CPU (36 GB/s of state,
//! `1ebda4fc:docs/plans/gdn-resident-state.md`), and chunking trades
//! 1.5x the multiply-adds for a 32nd of that traffic. On the GPU the
//! state never leaves registers, so the trade buys nothing: this steps
//! the rule token by token, `frink_core::gdn::delta_step` row for row,
//! the shape llama.cpp's own `kernel_gated_delta_net` has, and one
//! layer's whole prefill recurrence is one dispatch.
//!
//! A chunked version of this kernel (16 rows a chunk, one thread a
//! state row, one threadgroup a head) ran first; on Bonsai-2-27B's 48
//! heads it left the GPU nearly idle, 2.05 ms a layer at 512 rows on an
//! M5 Max against this one's 0.50.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};

use crate::gdn::{DeltaShape, HeadMapKind};
use crate::gpu::{ensure_pipeline, MetalError};

/// Simdgroups per threadgroup.
const SG_PER_TG: usize = 4;

/// The widest head this kernel serves: four `float4`s of a state row a
/// lane, eight lanes a row.
pub const MAX_HEAD_DIM: usize = 128;

/// `(entry, lanes a state row takes)` for a head width, or `None` for
/// a width with no instantiation: a power of two from 4 to
/// [`MAX_HEAD_DIM`].
fn seq_entry(head_dim: usize) -> Option<(&'static str, usize)> {
    Some(match head_dim {
        4 => ("gdn_seq_rows_1_1", 1),
        8 => ("gdn_seq_rows_2_1", 2),
        16 => ("gdn_seq_rows_4_1", 4),
        32 => ("gdn_seq_rows_8_1", 8),
        64 => ("gdn_seq_rows_8_2", 8),
        128 => ("gdn_seq_rows_8_4", 8),
        _ => return None,
    })
}

pub const GDN_CHUNK_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// `LPR` lanes per state row `j` of one value head, `32 / LPR` rows a
// simdgroup, the row's `S = 4 LPR NV` columns held in registers as
// `NV` float4s a lane (lane `c` of a row owns columns `4 (v LPR + c)
// ..+4`), and every token of the batch stepped in order inside the
// kernel:
//
//   S[j] *= exp(g_t)
//   d     = beta_t (v_t[j] - S[j] . k_t)
//   S[j] += d k_t
//   out_t[j] = (S[j] . q_t) / sqrt(S)
//
// which is `frink_core::gdn::delta_step` and llama.cpp's own
// `kernel_gated_delta_net` (`ggml-metal/kernels/gated_delta_net.metal`)
// row for row. Nothing a row computes reads another row, so the only
// reductions are the two dot products, across the row's `LPR` lanes.
//
// It replaced a chunked kernel (16 tokens at a time, one thread per
// row, one threadgroup per head) that kept 48 threadgroups of 128
// threads busy on Bonsai-2-27B's 48 heads: 2.05 ms a layer at 512 rows
// on an M5 Max. A simdgroup a row, 32 lanes of four columns each, took
// 1.08 ms, most of it the per-token reductions and loads rather than
// the arithmetic; eight lanes of sixteen columns spend four times
// fewer of those on the same row.
template<short LPR, short NV>
kernel void gdn_seq_rows(
    device float* state [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device const float* g [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],
    constant uint4& span [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort n_sg [[simdgroups_per_threadgroup]]
) {
    // dims: n_k_heads, n_v_heads, head_dim, map flag.
    // span: total rows, the floats between consecutive rows of q, k and
    // v (0: each packed at its own width; a batch of `[q | k | v]` rows
    // passes its convolution width and binds the three at their
    // offsets), unused, unused.
    const uint n_k = dims.x;
    const uint n_v = dims.y;
    const uint S   = dims.z;
    const uint rows = span.x;
    constexpr ushort RPS = 32 / LPR;

    const uint h = tgid.y;
    const ushort c = lane % LPR;
    const uint j = (tgid.x * n_sg + sg) * RPS + lane / LPR;
    // A row's lanes leave together, and the shuffles below never leave
    // a row, so none of them reads a lane that has gone.
    if (h >= n_v || j >= S) {
        return;
    }
    const uint kh = (dims.w == 0u) ? (h % n_k) : (h / (n_v / n_k));
    const uint key_row = n_k * S;
    const uint val_row = n_v * S;
    const uint qk_stride = (span.y == 0u) ? key_row : span.y;
    const uint v_stride = (span.y == 0u) ? val_row : span.y;
    const float scale = 1.0f / sqrt(float(S));

    device float4* row = (device float4*)(state + (size_t)h * S * S + (size_t)j * S);
    float4 ls[NV];
    for (short e = 0; e < NV; ++e) {
        ls[e] = row[e * LPR + c];
    }
    device const float* kp = k + kh * S;
    device const float* qp = q + kh * S;
    device const float* vp = v + h * S + j;
    device float* op = out + h * S + j;

    for (uint t = 0; t < rows; ++t) {
        const float decay = exp(g[t * n_v + h]);
        const float b = beta[t * n_v + h];
        float4 kt[NV];
        float4 qt[NV];
        float4 sk4 = 0.0f;
        for (short e = 0; e < NV; ++e) {
            kt[e] = ((device const float4*)kp)[e * LPR + c];
            qt[e] = ((device const float4*)qp)[e * LPR + c];
            ls[e] *= decay;
            sk4 += ls[e] * kt[e];
        }
        float sk = (sk4.x + sk4.y) + (sk4.z + sk4.w);
        for (ushort o = LPR / 2; o > 0; o >>= 1) {
            sk += simd_shuffle_xor(sk, o);
        }
        const float d = b * (vp[0] - sk);
        float4 y4 = 0.0f;
        for (short e = 0; e < NV; ++e) {
            ls[e] += kt[e] * d;
            y4 += ls[e] * qt[e];
        }
        float y = (y4.x + y4.y) + (y4.z + y4.w);
        for (ushort o = LPR / 2; o > 0; o >>= 1) {
            y += simd_shuffle_xor(y, o);
        }
        if (c == 0) {
            op[0] = y * scale;
        }
        kp += qk_stride;
        qp += qk_stride;
        vp += v_stride;
        op += val_row;
    }
    for (short e = 0; e < NV; ++e) {
        row[e * LPR + c] = ls[e];
    }
}

#define GDN_SEQ_ROWS(LPR, NV) \
template [[host_name("gdn_seq_rows_" #LPR "_" #NV)]] kernel void gdn_seq_rows<LPR, NV>( \
    device float*, device const float*, device const float*, device const float*, \
    device const float*, device const float*, device float*, constant uint4&, \
    constant uint4&, uint2, ushort, ushort, ushort);
GDN_SEQ_ROWS(1, 1)
GDN_SEQ_ROWS(2, 1)
GDN_SEQ_ROWS(4, 1)
GDN_SEQ_ROWS(8, 1)
GDN_SEQ_ROWS(8, 2)
GDN_SEQ_ROWS(8, 4)
"#;

/// Where a chunked recurrence reads Q, K and V: each a buffer and a
/// float offset into it, and the floats between consecutive rows, `0`
/// for each packed at its own width (`n_k_heads * head_dim` for Q and
/// K, `n_v_heads * head_dim` for V).
pub struct ChunkInputs<'a> {
    pub q: (&'a ProtocolObject<dyn MTLBuffer>, usize),
    pub k: (&'a ProtocolObject<dyn MTLBuffer>, usize),
    pub v: (&'a ProtocolObject<dyn MTLBuffer>, usize),
    pub row_stride: usize,
}

/// Every chunk of `rows` tokens through the delta rule, encoded into
/// the caller's `encoder` as ONE dispatch that steps the chunks in
/// order against the same `state`. The ONE encoding of the recurrence;
/// [`launch_delta_chunk`] wraps it in a command buffer of its own.
#[allow(clippy::too_many_arguments)]
pub fn encode_delta_chunk(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    shape: DeltaShape,
    rows: usize,
    state: &ProtocolObject<dyn MTLBuffer>,
    inputs: ChunkInputs<'_>,
    g: &ProtocolObject<dyn MTLBuffer>,
    beta: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), MetalError> {
    if shape.head_dim == 0
        || shape.head_dim > MAX_HEAD_DIM
        || shape.n_k_heads == 0
        || shape.n_v_heads == 0
        || !shape.n_v_heads.is_multiple_of(shape.n_k_heads)
        || (inputs.row_stride != 0
            && inputs.row_stride < shape.n_v_heads.max(shape.n_k_heads) * shape.head_dim)
        // The kernel reads Q, K and the state as float4s.
        || !inputs.row_stride.is_multiple_of(4)
        || [inputs.q.1, inputs.k.1].iter().any(|o| !o.is_multiple_of(4))
    {
        return Err(MetalError::CommandFailed);
    }
    let s = shape.head_dim;
    let (entry, lanes_per_row) = seq_entry(s).ok_or(MetalError::CommandFailed)?;
    let pipe = ensure_pipeline(device, GDN_CHUNK_KERNEL_SRC, entry)?;
    encoder.setComputePipelineState(&pipe.0);
    if 32 * SG_PER_TG > pipe.0.maxTotalThreadsPerThreadgroup() {
        return Err(MetalError::CommandFailed);
    }
    let rows_per_tg = SG_PER_TG * (32 / lanes_per_row);
    unsafe {
        for (idx, (buf, off)) in [
            (state, 0usize),
            inputs.q,
            inputs.k,
            inputs.v,
            (g, 0),
            (beta, 0),
            (out, 0),
        ]
        .into_iter()
        .enumerate()
        {
            encoder.setBuffer_offset_atIndex(Some(buf), off * 4, idx);
        }
        let mut dims: [u32; 4] = [
            shape.n_k_heads as u32,
            shape.n_v_heads as u32,
            s as u32,
            match shape.map {
                HeadMapKind::Tiled => 0,
                HeadMapKind::Grouped => 1,
            },
        ];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 16, 7);
        let mut span: [u32; 4] = [rows as u32, inputs.row_stride as u32, 0, 0];
        encoder.setBytes_length_atIndex(NonNull::new(span.as_mut_ptr() as *mut _).unwrap(), 16, 8);
    }
    crate::dispatch::dispatch_counted(
        encoder,
        MTLSize {
            width: s.div_ceil(rows_per_tg),
            height: shape.n_v_heads,
            depth: 1,
        },
        MTLSize {
            width: 32 * SG_PER_TG,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

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

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_delta_chunk(
        &encoder,
        device,
        shape,
        rows,
        &state_buf,
        ChunkInputs {
            q: (&q_buf, 0),
            k: (&k_buf, 0),
            v: (&v_buf, 0),
            row_stride: 0,
        },
        &g_buf,
        &beta_buf,
        &out_buf,
    )?;
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
