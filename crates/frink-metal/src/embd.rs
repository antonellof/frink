//! Metal embedding gather (`get_rows`): dequant one vocab row into f32.
//!
//! llama.cpp keeps the embedding on the GPU via `kernel_get_rows_*`.
//! Without a gather the decode path does `WeightMatrix::dequant_row` on
//! the host and copies the row into the dense-stack scratch, a host edge
//! before an otherwise fused command buffer. These kernels write straight
//! into `scratch.h`, for every kind in [`EmbdKind::ALL`] (#405).

use crate::dispatch::dispatch_counted;
use crate::gpu::{
    ensure_pipeline, resident_weight_buffer, shared_metal, MetalError, ResidentWeightBuffer,
    K_QUANT_MUL_MM_SG_KERNEL_SRC,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};
use std::ptr::NonNull;

/// ggml `kernel_get_rows_q`: one thread per 16 values of the row,
/// through the kind's GEMM dequant functor (`DQ::get(block, il, reg)`,
/// the same text the simdgroup GEMM reads its tiles with). Spliced into
/// the GEMM translation unit (`gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC`), so
/// there is one dequant per kind. Every read is through the functor,
/// which reads bytes (or a `half` at the format's own alignment), so a
/// row may start at any byte, as an MXFP4 row does.
const GET_ROWS_BODY_MSL: &str = r#"
template<typename DQ>
static inline void get_rows_impl(
    device const uchar* weights,
    device float* out,
    uint row_bytes,
    uint n_cols,
    uint row_idx,
    uint gid
) {
    if (gid >= n_cols / 16u) return;
    device const uchar* xb = weights + (size_t)row_idx * row_bytes
        + (size_t)(gid / uint(DQ::NL)) * uint(DQ::BLOCK_BYTES);
    float4x4 reg;
    DQ::get(xb, short(gid % uint(DQ::NL)), reg);
    device float* o = out + (size_t)gid * 16u;
    for (short i = 0; i < 16; ++i) {
        o[i] = reg[i / 4][i % 4];
    }
}

#define GET_ROWS_ENTRY(NAME, DQ)                                            \
kernel void NAME(                                                           \
    device const uchar* weights [[buffer(0)]],                              \
    device float* out [[buffer(1)]],                                        \
    constant uint& row_bytes [[buffer(2)]],                                 \
    constant uint& n_cols [[buffer(3)]],                                    \
    constant uint& row_idx [[buffer(4)]],                                   \
    uint gid [[thread_position_in_grid]]                                    \
) {                                                                         \
    get_rows_impl<DQ>(weights, out, row_bytes, n_cols, row_idx, gid);      \
}
"#;

/// The gather text for the GEMM translation unit: the body, then one
/// entry per [`EmbdKind::ALL`] row, generated from the table so the
/// entry points and the lookups cannot name different sets.
pub(crate) fn get_rows_msl() -> String {
    let mut s = String::from(GET_ROWS_BODY_MSL);
    for k in EmbdKind::ALL {
        s.push_str(&format!(
            "GET_ROWS_ENTRY({}, {})\n",
            k.get_rows_fn_name(),
            k.dequant_functor()
        ));
    }
    s
}

/// Threads per threadgroup of a gather; each thread writes 16 values.
const GET_ROWS_TG: usize = 64;

/// Which embedding quant we can gather on Metal. Defined in the ungated
/// [`crate::embd_kinds`] so the kernel coverage matrix can read it.
pub use crate::embd_kinds::EmbdKind;

/// Encode a single-row embedding gather into `out` (length `n_cols`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_get_rows(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    kind: EmbdKind,
    weight: &ResidentWeightBuffer,
    out: &ProtocolObject<dyn MTLBuffer>,
    row_bytes: u32,
    n_cols: u32,
    row_idx: u32,
) -> Result<(), MetalError> {
    debug_assert!(
        n_cols.is_multiple_of(16),
        "a gather writes whole 16-value chunks"
    );
    let pipe = ensure_pipeline(
        device,
        *K_QUANT_MUL_MM_SG_KERNEL_SRC,
        kind.get_rows_fn_name(),
    )?;
    encoder.setComputePipelineState(&pipe.0);
    // SAFETY: the buffers outlive the encoder (the caller holds them
    // until the command buffer completes), and each `setBytes` reads 4
    // bytes from a live local `u32` that Metal copies at the call.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&weight.buffer), weight.weight_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 1);
        let mut rb = row_bytes;
        let mut nc = n_cols;
        let mut ri = row_idx;
        encoder.setBytes_length_atIndex(NonNull::new(&mut rb as *mut _ as *mut _).unwrap(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::new(&mut nc as *mut _ as *mut _).unwrap(), 4, 3);
        encoder.setBytes_length_atIndex(NonNull::new(&mut ri as *mut _ as *mut _).unwrap(), 4, 4);
    }
    let tg = GET_ROWS_TG;
    let n_tg = (n_cols as usize / 16).div_ceil(tg);
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Host-facing gather for tests / one-shot use.
pub fn launch_get_rows(
    kind: EmbdKind,
    weights: &[u8],
    rows: usize,
    row_bytes: usize,
    n_cols: usize,
    row_idx: usize,
) -> Result<Vec<f32>, MetalError> {
    assert!(row_idx < rows);
    assert_eq!(weights.len(), rows * row_bytes);
    let (_, _, block_bytes, block_elems, _) =
        crate::gpu::matvec_launch_meta(kind.kind_name()).ok_or(MetalError::CommandFailed)?;
    assert_eq!(row_bytes % block_bytes, 0, "a row is whole blocks");
    assert_eq!(n_cols, (row_bytes / block_bytes) * block_elems);

    let shared = shared_metal()?;
    let device = &shared.device;
    let w = resident_weight_buffer(device, weights)?;
    let out = device
        .newBufferWithLength_options(n_cols * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)?;
    let cmd = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_get_rows(
        &enc,
        device,
        kind,
        &w,
        &out,
        row_bytes as u32,
        n_cols as u32,
        row_idx as u32,
    )?;
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ptr = out.contents();
    // SAFETY: `out` is a shared-storage buffer of `n_cols` f32s and the
    // command buffer that wrote it has completed.
    Ok(unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n_cols).to_vec() })
}

#[cfg(all(test, feature = "metal"))]
mod tests;
