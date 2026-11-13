//! Metal elementwise ops for decode-layer residency: residual add,
//! SiLU×up / GELU×up (the GLU pairs), axpy, argmax. The norms are
//! `crate::norm`. Used by the fused dense-layer path so activations
//! stay on-GPU between attention and FFN.

use crate::dispatch::dispatch_counted;
use crate::gpu::{ensure_pipeline, MetalError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};
use std::ptr::NonNull;

const VEC_ADD_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// `a[i] += b[i % period]`: a plain add when `period == n`, and a bias
// broadcast over `n / period` rows otherwise (one dispatch for a whole
// prefill batch instead of one per token).
kernel void vec_add_f32(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& period [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        a[i] += b[i % period];
    }
}

kernel void f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        dst[i] = half(src[i]);
    }
}
"#;

/// Fused residual add + RMSNorm×γ (ggml F=3-style, Pre-LN shape):
/// `h[i] += add[i]`, then `out[i] = rms_norm(h) * weight[i]`.
/// Replaces a separate `vec_add` + `rms_norm` pair (~2× dispatches/layer).
/// Uses simd_sum reduction (matches ggml `kernel_rms_norm_mul_add_f32`).
/// In-place per-head RMSNorm×γ (Qwen3 / Gemma-3 QK-norm): each head of
/// `head_dim` elements is normalized independently with the shared
/// `weight[head_dim]`. One simdgroup per head (head_dim ≤ 256 → ≤ 8
/// elements/lane), matching ggml's RMS_NORM on a [head_dim, n_head] view.
/// `x *= sigmoid(g)`, in place: the gate a Qwen3.5 attention layer
/// applies to its attention output before `wo`
/// (`frink_models::attn_gate::apply_interleaved_gate`, the same
/// arithmetic). Its own kernel rather than `silu_mul`'s, because that
/// one multiplies by `g * sigmoid(g)` and the factor of `g` is the
/// whole difference.
const SIGMOID_MUL_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_mul_f32(
    device float* x [[buffer(0)]],
    device const float* g [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        x[i] *= 1.0f / (1.0f + exp(-g[i]));
    }
}
"#;

const SILU_MUL_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + exp(-g))) * up[i];
    }
}

// Same activation, half store — folds the f32→f16 staging convert that
// used to sit between SwiGLU and the down `mul_mm_sg_f16`. The product is
// still formed in f32 and rounded once, exactly as the two-dispatch pair
// did, so this is bit-identical to silu_mul_f32 + f32_to_f16.
kernel void silu_mul_f32_to_f16(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float g = gate[i];
        out[i] = half((g / (1.0f + exp(-g))) * up[i]);
    }
}
"#;

/// Gemma-2's `final_logit_softcapping`, in place over a vocabulary of
/// logits: `y = cap * tanh(y * inv)`, with `inv = 1 / cap` computed on
/// the host exactly as `frink_core::matmul::softcap_inplace` computes
/// it, so the argument to `tanh` is the same float on both sides and
/// `precise::tanh` is what separates them (about an ulp). This ran on
/// the CPU after the lm_head's command buffer had completed, 0.65 ms
/// per token for 256k logits (PR #202), more than the whole encode
/// phase; as an epilogue in that buffer it is one dispatch.
const SOFTCAP_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softcap_f32(
    device float* y [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant float& cap [[buffer(2)]],
    constant float& inv [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        y[i] = cap * precise::tanh(y[i] * inv);
    }
}
"#;

/// `y[i] += a * x[i]` — MoE weighted expert accumulate.
const AXPY_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void axpy_f32(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant float& a [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        y[i] += a * x[i];
    }
}
"#;

/// Gemma GeGLU pair: `gelu(gate) * up`, tanh approximation matching
/// `frink_core::matmul::gelu` (HF / llama.cpp convention).
const GELU_MUL_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gelu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float g = gate[i];
        const float K = 0.7978845608028654f; // sqrt(2/pi)
        const float C = 0.044715f;
        float gelu = 0.5f * g * (1.0f + precise::tanh(K * (g + C * g * g * g)));
        out[i] = gelu * up[i];
    }
}

// Half store — see `silu_mul_f32_to_f16`.
kernel void gelu_mul_f32_to_f16(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float g = gate[i];
        const float K = 0.7978845608028654f; // sqrt(2/pi)
        const float C = 0.044715f;
        float gelu = 0.5f * g * (1.0f + precise::tanh(K * (g + C * g * g * g)));
        out[i] = half(gelu * up[i]);
    }
}
"#;

/// Parallel argmax over `n` floats (one threadgroup). Each thread scans a
/// strided slice, then a tree-reduce keeps the first index on ties (`>`).
/// Sequential single-thread scan of vocab (~128k) was measured to erase the
/// gain from keeping lm_head on-GPU (llama.cpp leaves logits on device and
/// samples without a host round-trip).
const ARGMAX_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void argmax_f32(
    device const float* x [[buffer(0)]],
    device uint* out_idx [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* sh_v [[threadgroup(0)]],
    threadgroup uint* sh_i [[threadgroup(1)]]
) {
    float bv = -INFINITY;
    uint bi = 0u;
    for (uint i = tid; i < n; i += tg) {
        float v = x[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    sh_v[tid] = bv;
    sh_i[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float v2 = sh_v[tid + stride];
            uint i2 = sh_i[tid + stride];
            float v1 = sh_v[tid];
            uint i1 = sh_i[tid];
            if (v2 > v1 || (v2 == v1 && i2 < i1)) {
                sh_v[tid] = v2;
                sh_i[tid] = i2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        out_idx[0] = sh_i[0];
    }
}
"#;

pub(crate) fn encode_vec_add(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    encode_vec_add_at(encoder, device, a, 0, b, n)
}

/// Contiguous `float` → `half` (dense-prefill GEMM src1 / llama `*_f16`).
pub(crate) fn encode_f32_to_f16(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, VEC_ADD_KERNEL_SRC, "f32_to_f16")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
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

/// `a[a_offset_bytes/4 ..] += b[0..n]` (in-place on `a`).
pub(crate) fn encode_vec_add_at(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    a: &ProtocolObject<dyn MTLBuffer>,
    a_offset_bytes: usize,
    b: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    encode_vec_add_period(encoder, device, a, a_offset_bytes, b, n, n)
}

/// `a[r][i] += b[i]` for every one of `rows` contiguous rows of `n_row`
/// floats: a bias added to a whole batch in one dispatch.
pub(crate) fn encode_vec_add_rows(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    n_row: u32,
    rows: u32,
) -> Result<(), MetalError> {
    encode_vec_add_period(encoder, device, a, 0, b, n_row * rows, n_row)
}

fn encode_vec_add_period(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    a: &ProtocolObject<dyn MTLBuffer>,
    a_offset_bytes: usize,
    b: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    period: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, VEC_ADD_KERNEL_SRC, "vec_add_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), a_offset_bytes, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        for (idx, mut v) in [(2usize, n), (3, period)] {
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                4,
                idx,
            );
        }
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
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

pub(crate) fn encode_silu_mul(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, SILU_MUL_KERNEL_SRC, "silu_mul_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(gate), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(up), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 2);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
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

/// `x *= sigmoid(g)` in place over `n` elements.
pub(crate) fn encode_sigmoid_mul(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    g: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, SIGMOID_MUL_KERNEL_SRC, "sigmoid_mul_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(g), 0, 1);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
    }
    let tg = 256usize;
    dispatch_counted(
        encoder,
        MTLSize {
            width: (n as usize).div_ceil(tg),
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

/// `y = cap * tanh(y / cap)` in place, Gemma-2's final logit softcap.
/// A `cap <= 0.0` encodes nothing, exactly as the CPU version applies
/// nothing.
pub(crate) fn encode_softcap(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    y: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    cap: f32,
) -> Result<(), MetalError> {
    if cap <= 0.0 {
        return Ok(());
    }
    let pipe = ensure_pipeline(device, SOFTCAP_KERNEL_SRC, "softcap_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(y), 0, 0);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            1,
        );
        let mut cap_f = cap;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut cap_f as *mut f32 as *mut _).unwrap(),
            4,
            2,
        );
        // The same `1.0 / softcap` the CPU path multiplies by.
        let mut inv_f = 1.0f32 / cap;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut inv_f as *mut f32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    let tg = 256usize;
    dispatch_counted(
        encoder,
        MTLSize {
            width: (n as usize).div_ceil(tg),
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

/// `y[i] += a * x[i]`.
pub(crate) fn encode_axpy(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    y: &ProtocolObject<dyn MTLBuffer>,
    x: &ProtocolObject<dyn MTLBuffer>,
    a: f32,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, AXPY_KERNEL_SRC, "axpy_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(y), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(x), 0, 1);
        let mut a_f = a;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut a_f as *mut f32 as *mut _).unwrap(),
            4,
            2,
        );
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
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

/// `out = gelu(gate) * up` (Gemma GeGLU; tanh-approx gelu).
/// SwiGLU / GeGLU writing `half` — [`encode_silu_mul`] / [`encode_gelu_mul`]
/// with the following f32→f16 staging convert folded in. Bit-identical to
/// the two-dispatch pair (the product is still formed in f32 and rounded
/// once); saves one dispatch and one barrier per layer.
pub(crate) fn encode_act_mul_f32_to_f16(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    gelu: bool,
) -> Result<(), MetalError> {
    let (src, name) = if gelu {
        (GELU_MUL_KERNEL_SRC, "gelu_mul_f32_to_f16")
    } else {
        (SILU_MUL_KERNEL_SRC, "silu_mul_f32_to_f16")
    };
    let pipe = ensure_pipeline(device, src, name)?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(gate), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(up), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 2);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
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

pub(crate) fn encode_gelu_mul(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, GELU_MUL_KERNEL_SRC, "gelu_mul_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(gate), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(up), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 2);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
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

/// Encode argmax of `n` f32 values in `x` into a single `u32` at `out_idx[0]`.
pub(crate) fn encode_argmax(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    out_idx: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, ARGMAX_KERNEL_SRC, "argmax_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    // Power-of-two TG; 1024 covers vocab with ~128 iters/thread on Llama-3.
    let tg = 1024u32;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_idx), 0, 1);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
        encoder.setThreadgroupMemoryLength_atIndex((tg as usize) * 4, 0);
        encoder.setThreadgroupMemoryLength_atIndex((tg as usize) * 4, 1);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: 1,
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

/// Front-load elementwise pipelines used by [`crate::attn::launch_prefill_dense_layer`].
pub(crate) fn warm_prefill_elem_pipelines(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    gelu_ffn: bool,
) -> Result<(), MetalError> {
    crate::norm::warm_prefill_norm_pipelines(device)?;
    ensure_pipeline(device, VEC_ADD_KERNEL_SRC, "vec_add_f32")?;
    ensure_pipeline(device, VEC_ADD_KERNEL_SRC, "f32_to_f16")?;
    if gelu_ffn {
        ensure_pipeline(device, GELU_MUL_KERNEL_SRC, "gelu_mul_f32")?;
        ensure_pipeline(device, GELU_MUL_KERNEL_SRC, "gelu_mul_f32_to_f16")?;
    } else {
        ensure_pipeline(device, SILU_MUL_KERNEL_SRC, "silu_mul_f32")?;
        ensure_pipeline(device, SILU_MUL_KERNEL_SRC, "silu_mul_f32_to_f16")?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::gpu::{shared_metal, MetalError};
    use crate::norm::{encode_add_rms_norm, encode_rms_norm};
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLResourceOptions};
    use std::ptr::NonNull;

    pub(crate) fn upload(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        data: &[f32],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let mut owned = data.to_vec();
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(owned.as_mut_ptr() as *mut _).unwrap(),
                owned.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocFailed)
    }

    pub(crate) fn alloc(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        n: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        device
            .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocFailed)
    }

    pub(crate) fn read_f32(buf: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<f32> {
        let ptr = buf.contents();
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n).to_vec() }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rms_norm_silu_add_match_cpu() {
        let shared = shared_metal().expect("metal");
        let device = &shared.device;
        let n = 128usize;
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin()).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        let cpu_rms: Vec<f32> = x
            .iter()
            .zip(w.iter())
            .map(|(v, ww)| v * scale * ww)
            .collect();

        let x_buf = upload(device, &x).unwrap();
        let w_buf = upload(device, &w).unwrap();
        let out_buf = alloc(device, n).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_rms_norm(&enc, device, &x_buf, &w_buf, &out_buf, n as u32, eps).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu_rms = read_f32(&out_buf, n);
        for (i, (a, b)) in cpu_rms.iter().zip(gpu_rms.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "rms {i}: {a} vs {b}");
        }

        let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
        let cpu_silu: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let g_buf = upload(device, &gate).unwrap();
        let u_buf = upload(device, &up).unwrap();
        let s_buf = alloc(device, n).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_silu_mul(&enc, device, &g_buf, &u_buf, &s_buf, n as u32).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu_silu = read_f32(&s_buf, n);
        for (i, (a, b)) in cpu_silu.iter().zip(gpu_silu.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "silu {i}: {a} vs {b}");
        }

        let mut a = cpu_rms.clone();
        let b = cpu_silu.clone();
        for (aa, bb) in a.iter_mut().zip(b.iter()) {
            *aa += bb;
        }
        let a_buf = upload(device, &cpu_rms).unwrap();
        let b_buf = upload(device, &b).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_vec_add(&enc, device, &a_buf, &b_buf, n as u32).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu_add = read_f32(&a_buf, n);
        for (i, (aa, bb)) in a.iter().zip(gpu_add.iter()).enumerate() {
            let tol = 1e-4 * aa.abs().max(1.0);
            assert!((aa - bb).abs() <= tol, "add {i}: {aa} vs {bb}");
        }

        // Fused h+=add; out=rms_norm(h)*w
        let h0: Vec<f32> = (0..n).map(|i| (i as f32 * 0.09).sin()).collect();
        let addend: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).cos()).collect();
        let mut h_cpu = h0.clone();
        for (h, a) in h_cpu.iter_mut().zip(addend.iter()) {
            *h += *a;
        }
        let mean_sq2 = h_cpu.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let scale2 = 1.0 / (mean_sq2 + eps).sqrt();
        let cpu_fused: Vec<f32> = h_cpu
            .iter()
            .zip(w.iter())
            .map(|(v, ww)| v * scale2 * ww)
            .collect();
        let h_buf = upload(device, &h0).unwrap();
        let add_buf = upload(device, &addend).unwrap();
        let fused_out = alloc(device, n).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_add_rms_norm(
            &enc, device, &h_buf, &add_buf, &w_buf, &fused_out, n as u32, eps,
        )
        .unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu_h = read_f32(&h_buf, n);
        let gpu_fused = read_f32(&fused_out, n);
        for (i, (a, b)) in h_cpu.iter().zip(gpu_h.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "fused h {i}: {a} vs {b}");
        }
        for (i, (a, b)) in cpu_fused.iter().zip(gpu_fused.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "fused out {i}: {a} vs {b}");
        }
    }

    /// The GPU softcap must be the host `softcap_inplace` to within an
    /// ulp of `tanh`, because the sampled decode path applies one and
    /// the CPU reference path the other, and `frink verify` compares
    /// their tokens. A `cap <= 0` must apply nothing, as the host does.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn softcap_matches_the_host_kernel() {
        let shared = shared_metal().expect("metal");
        let device = &shared.device;
        let n = 1000usize;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 90.0).collect();
        for cap in [30.0f32, 50.0] {
            let inv = 1.0 / cap;
            let want: Vec<f32> = x.iter().map(|v| cap * (v * inv).tanh()).collect();
            let buf = upload(device, &x).unwrap();
            let cmd = shared.queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            encode_softcap(&enc, device, &buf, n as u32, cap).unwrap();
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            let got = read_f32(&buf, n);
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= 2e-6 * w.abs().max(1.0),
                    "cap {cap} elem {i}: gpu {g} host {w}"
                );
                assert!(g.abs() < cap, "cap {cap} elem {i}: {g} escaped the cap");
            }
        }
        let buf = upload(device, &x).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_softcap(&enc, device, &buf, n as u32, 0.0).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        assert_eq!(read_f32(&buf, n), x, "a cap of 0 applies nothing");
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn argmax_matches_host_including_ties() {
        let shared = shared_metal().expect("metal");
        let device = &shared.device;
        // Unique max at index 7; then a tie at 2 and 9 (first wins).
        let mut x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
        x[7] = 100.0;
        let host = x
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(host, 7);

        let x_buf = upload(device, &x).unwrap();
        let idx_buf = alloc(device, 1).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_argmax(&enc, device, &x_buf, &idx_buf, x.len() as u32).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ptr = idx_buf.contents();
        let gpu = unsafe { *(ptr.as_ptr() as *const u32) as usize };
        assert_eq!(gpu, host);

        x[7] = 0.0;
        x[2] = 50.0;
        x[9] = 50.0;
        // Match Metal/`>` scan (first index wins). Rust `Iterator::max_by`
        // on Equal keeps the later element — do not use it for the oracle.
        let mut host_tie = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in x.iter().enumerate() {
            if v > best {
                best = v;
                host_tie = i;
            }
        }
        assert_eq!(host_tie, 2);
        let x_buf = upload(device, &x).unwrap();
        // Shared-mode upload must be visible to the host immediately.
        let uploaded = read_f32(&x_buf, x.len());
        assert_eq!(uploaded[2], 50.0);
        assert_eq!(uploaded[9], 50.0);
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_argmax(&enc, device, &x_buf, &idx_buf, x.len() as u32).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ptr = idx_buf.contents();
        let gpu = unsafe { *(ptr.as_ptr() as *const u32) as usize };
        assert_eq!(
            gpu, host_tie,
            "gpu={gpu} host={host_tie}; x[2]={} x[9]={}",
            x[2], x[9]
        );
    }
}
