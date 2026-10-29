//! Metal embedding gather (`get_rows`): dequant one vocab row into f32.
//!
//! llama.cpp keeps embd on-GPU via `kernel_get_rows_*`. Frink's decode
//! path previously did `WeightMatrix::dequant_row` on the host then
//! `copy_f32_into` into the dense-stack scratch — a pure host edge before
//! an otherwise fused CB. These kernels write straight into `scratch.h`.

use crate::dispatch::dispatch_counted;
use crate::gpu::{
    ensure_pipeline, resident_weight_buffer, shared_metal, MetalError, ResidentWeightBuffer,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};
use std::ptr::NonNull;

const GET_ROWS_Q4_K_SRC: &str = r#"
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

kernel void get_rows_q4_k(
    device const uchar* weights [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& row_bytes [[buffer(2)]],
    constant uint& n_cols [[buffer(3)]],
    constant uint& row_idx [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_cols) return;
    uint block = gid / 256u;
    uint j = gid % 256u;
    device const uchar* b = weights + row_idx * row_bytes + block * 144u;
    half d_h = *((device const half*)(b + 0));
    half dmin_h = *((device const half*)(b + 2));
    float d = float(d_h);
    float dmin = float(dmin_h);
    device const uchar* scales = b + 4;
    device const uchar* qs = b + 16;

    uint group = j / 64u;       // 0..3
    uint within = j % 64u;
    uint is = group * 2u;
    uchar2 sm1 = q4_k_scale_min(is, scales);
    uchar2 sm2 = q4_k_scale_min(is + 1u, scales);
    uint q_off = group * 32u;
    if (within < 32u) {
        float d1 = d * float(sm1.x);
        float min1 = dmin * float(sm1.y);
        out[gid] = d1 * float(qs[q_off + within] & 0x0Fu) - min1;
    } else {
        float d2 = d * float(sm2.x);
        float min2 = dmin * float(sm2.y);
        out[gid] = d2 * float(qs[q_off + (within - 32u)] >> 4u) - min2;
    }
}
"#;

const GET_ROWS_Q6_K_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void get_rows_q6_k(
    device const uchar* weights [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& row_bytes [[buffer(2)]],
    constant uint& n_cols [[buffer(3)]],
    constant uint& row_idx [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_cols) return;
    uint block = gid / 256u;
    uint j = gid % 256u;
    device const uchar* b = weights + row_idx * row_bytes + block * 210u;
    device const uchar* ql_full = b + 0;
    device const uchar* qh_full = b + 128;
    device const uchar* sc_full = b + 192;
    float d = float(*((device const half*)(b + 208)));

    uint half_i = j / 128u;
    uint jj = j % 128u;
    device const uchar* ql = ql_full + half_i * 64u;
    device const uchar* qh = qh_full + half_i * 32u;
    device const char* sc = (device const char*)(sc_full + half_i * 8u);

    // Inverse of dequant_q6_k's four lanes at each l in 0..32.
    uint lane = jj / 32u; // 0..3 → y[l], y[l+32], y[l+64], y[l+96]
    uint l = jj % 32u;
    uint is = l / 16u;
    char q;
    char s;
    if (lane == 0u) {
        q = char((ql[l] & 0x0Fu) | ((qh[l] & 3u) << 4u)) - 32;
        s = sc[is];
    } else if (lane == 1u) {
        q = char((ql[l + 32u] & 0x0Fu) | (((qh[l] >> 2u) & 3u) << 4u)) - 32;
        s = sc[is + 2];
    } else if (lane == 2u) {
        q = char((ql[l] >> 4u) | (((qh[l] >> 4u) & 3u) << 4u)) - 32;
        s = sc[is + 4];
    } else {
        q = char((ql[l + 32u] >> 4u) | (((qh[l] >> 6u) & 3u) << 4u)) - 32;
        s = sc[is + 6];
    }
    out[gid] = d * float(s) * float(q);
}
"#;

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
    let (src, name) = match kind {
        EmbdKind::Q4K => (GET_ROWS_Q4_K_SRC, "get_rows_q4_k"),
        EmbdKind::Q6K => (GET_ROWS_Q6_K_SRC, "get_rows_q6_k"),
    };
    let pipe = ensure_pipeline(device, src, name)?;
    encoder.setComputePipelineState(&pipe.0);
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
    let tg = 256usize;
    let n_tg = (n_cols as usize).div_ceil(tg);
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
    let block_bytes = match kind {
        EmbdKind::Q4K => 144,
        EmbdKind::Q6K => 210,
    };
    assert_eq!(n_cols, (row_bytes / block_bytes) * 256);

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
    Ok(unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n_cols).to_vec() })
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use frink_quant::{dequant_q4_k, dequant_q6_k, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};

    /// Each gather is keyed by the matvec kernel the Metal table really
    /// resolves for its kind, so the decode path's `from_fn_name` and
    /// the coverage matrix's `from_kind_name` cannot disagree. Runs
    /// without a device.
    #[test]
    fn every_gather_is_keyed_by_its_kinds_matvec_kernel() {
        for k in EmbdKind::ALL {
            let meta = crate::gpu::matvec_launch_meta(k.kind_name())
                .unwrap_or_else(|| panic!("{k:?}: a gather for a kind with no matvec kernel"));
            assert_eq!(meta.1, k.matvec_fn_name(), "{k:?}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn get_rows_q4_k_matches_cpu() {
        let cols = 512usize; // 2 blocks
        let row_bytes = (cols / 256) * Q4_K_BLOCK_BYTES;
        let rows = 4usize;
        let mut weights = vec![0u8; rows * row_bytes];
        for (i, b) in weights.iter_mut().enumerate() {
            *b = (i.wrapping_mul(37) % 251) as u8;
        }
        // Valid-ish f16 scales in each block header.
        for r in 0..rows {
            for blk in 0..(cols / 256) {
                let off = r * row_bytes + blk * Q4_K_BLOCK_BYTES;
                weights[off..off + 2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
                weights[off + 2..off + 4]
                    .copy_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
            }
        }
        let row = 2usize;
        let cpu = dequant_q4_k(&weights[row * row_bytes..(row + 1) * row_bytes]).unwrap();
        let gpu = launch_get_rows(EmbdKind::Q4K, &weights, rows, row_bytes, cols, row).unwrap();
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "q4k {i}: {a} vs {b}");
        }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn get_rows_q6_k_matches_cpu() {
        let cols = 512usize;
        let row_bytes = (cols / 256) * Q6_K_BLOCK_BYTES;
        let rows = 3usize;
        let mut weights = vec![0u8; rows * row_bytes];
        for (i, b) in weights.iter_mut().enumerate() {
            *b = (i.wrapping_mul(41) % 251) as u8;
        }
        for r in 0..rows {
            for blk in 0..(cols / 256) {
                let off = r * row_bytes + blk * Q6_K_BLOCK_BYTES;
                weights[off + 208..off + 210]
                    .copy_from_slice(&half::f16::from_f32(0.02).to_le_bytes());
            }
        }
        let row = 1usize;
        let cpu = dequant_q6_k(&weights[row * row_bytes..(row + 1) * row_bytes]).unwrap();
        let gpu = launch_get_rows(EmbdKind::Q6K, &weights, rows, row_bytes, cols, row).unwrap();
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "q6k {i}: {a} vs {b}");
        }
    }
}
