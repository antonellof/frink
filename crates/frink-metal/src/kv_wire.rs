//! The KV wire: what a token's K and V look like in device memory, and
//! the kernels that put them there and read them back.
//!
//! Split out of `attn.rs` when the K rotation landed, because
//! that change touches the append kernel, the dtype table and the
//! dequant path and none of the attention kernels, which is the line
//! this module draws. `attn.rs` asks this module what a dtype costs,
//! whether a geometry is servable and how to append; it owns the
//! softmax.
//!
//! Selected by `FRINK_CTK` / `--ctk`: `f16`, `q8_0`, `fp8` (the Q8_0
//! wire under another name) and `q4`.

use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::attn::{dispatch_counted, MetalKvBuffers};
use crate::gpu::{ensure_pipeline, MetalError};

const KV_APPEND_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Append f32 K/V token into an f16-resident cache (llama.cpp default).
//
// K and V are one dispatch: the grid's HEIGHT is the plane count, so
// `gid.y` picks the pair of buffers and no uniform has to carry it.
// Every call site appends K and V at the same offset and length, and
// GitHub issue #149 makes the second encode worth removing.
kernel void kv_append(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& offset_elems [[buffer(2)]],
    constant uint& n_elems [[buffer(3)]],
    device const float* src2 [[buffer(4)]],
    device half* dst2 [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    if (i >= n_elems) return;
    device const float* s = (gid.y == 0u) ? src : src2;
    device half* d = (gid.y == 0u) ? dst : dst2;
    d[offset_elems + i] = half(s[i]);
}
"#;

/// ggml Q8_0: 32 int8 values + one f16 scale (34 bytes). One thread / block.
const KV_APPEND_Q8_0_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// K and V in one dispatch; grid height is the plane count (see
// `kv_append`).
kernel void kv_append_q8_0(
    device const float* src_in [[buffer(0)]],
    device uchar* dst_in [[buffer(1)]],
    constant uint& offset_elems [[buffer(2)]],
    constant uint& n_elems [[buffer(3)]],
    device const float* src2 [[buffer(4)]],
    device uchar* dst2 [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.x;
    device const float* src = (gid.y == 0u) ? src_in : src2;
    device uchar* dst = (gid.y == 0u) ? dst_in : dst2;
    const uint BLOCK = 32u;
    const uint BLOCK_BYTES = 34u;
    uint n_blocks = n_elems / BLOCK;
    if (b >= n_blocks) return;
    uint src_base = b * BLOCK;
    float amax = 0.0f;
    for (uint i = 0u; i < BLOCK; i++) {
        amax = fmax(amax, fabs(src[src_base + i]));
    }
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    uint dst_block = (offset_elems / BLOCK) + b;
    uint dst_base = dst_block * BLOCK_BYTES;
    half d_h = half(d);
    dst[dst_base + 0] = uchar(as_type<ushort>(d_h) & 0xFFu);
    dst[dst_base + 1] = uchar(as_type<ushort>(d_h) >> 8u);
    for (uint i = 0u; i < BLOCK; i++) {
        int q = int(round(src[src_base + i] * id));
        q = clamp(q, -127, 127);
        dst[dst_base + 2u + i] = uchar(char(q));
    }
}
"#;

const DEQUANT_Q8_0_TO_F16_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dequant_q8_0_to_f16(
    device const uchar* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n_elems [[buffer(2)]],
    uint b [[thread_position_in_grid]]
) {
    const uint BLOCK = 32u;
    const uint BLOCK_BYTES = 34u;
    uint n_blocks = n_elems / BLOCK;
    if (b >= n_blocks) return;
    uint src_base = b * BLOCK_BYTES;
    ushort d_bits = ushort(src[src_base]) | (ushort(src[src_base + 1u]) << 8u);
    float d = float(as_type<half>(d_bits));
    uint dst_base = b * BLOCK;
    for (uint i = 0u; i < BLOCK; i++) {
        char q = char(src[src_base + 2u + i]);
        dst[dst_base + i] = half(float(q) * d);
    }
}
"#;

/// 4-bit KV: f16 scale + 16 nibble bytes / 32 elems (18 B), with K
/// rotated first.
///
/// One threadgroup per (head, plane), `head_dim` threads. Plane 0 is K
/// and takes the randomized Hadamard rotation of
/// `frink_quant::kv_rotation` (sign flip by the same hash, then the
/// orthonormal butterfly) before the per-32-group absmax; plane 1 is V
/// and does not, because the rotation is worth 39% of the error on K
/// and 12% on V, measured, and a rotated V would additionally need the
/// attention output rotated back per head.
///
/// `rotate` is a uniform rather than a second kernel so the K and V
/// appends stay ONE dispatch (GitHub issue #149), and it is 0 for a
/// head width the rotation cannot serve, which keeps the unrotated wire
/// this file shipped before.
const KV_APPEND_Q4_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Must match `frink_quant::kv_rotation::rotation_sign` exactly: the
// append writes under this pattern and the query rotation reads under
// it, so a difference of one bit is a different model, not a lossier
// one.
inline float rotation_sign(uint head_idx, uint channel) {
    uint h = head_idx * 0x9E3779B1u + channel * 0x85EBCA6Bu;
    h ^= h >> 15;
    h *= 0xC2B2AE35u;
    h ^= h >> 13;
    return (h & 1u) ? -1.0f : 1.0f;
}

kernel void kv_append_q4(
    device const float* src_in [[buffer(0)]],
    device uchar* dst_in [[buffer(1)]],
    constant uint& offset_elems [[buffer(2)]],
    constant uint& n_elems [[buffer(3)]],
    device const float* src2 [[buffer(4)]],
    device uchar* dst2 [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& rotate [[buffer(7)]],
    constant uint& n_kv_heads [[buffer(8)]],
    threadgroup float* sh [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    uint head = tgid.x;
    uint t = tid.x;
    device const float* src = (tgid.y == 0u) ? src_in : src2;
    device uchar* dst = (tgid.y == 0u) ? dst_in : dst2;
    const uint BLOCK = 32u;
    const uint BLOCK_BYTES = 18u;
    uint n_heads = n_elems / head_dim;
    if (head >= n_heads) return;

    // The sign pattern follows the KV head, not the token, so a query
    // rotated by `rotate_q_q4` sees the pattern its own K was
    // stored under.
    uint head_in_row = (n_kv_heads > 0u) ? (head % n_kv_heads) : 0u;
    uint src_base = head * head_dim;

    float x = src[src_base + t];
    if (rotate != 0u && tgid.y == 0u) {
        x *= rotation_sign(head_in_row, t);
        sh[t] = x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = 1u; step < head_dim; step <<= 1u) {
            float a = sh[t];
            float b = sh[t ^ step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sh[t] = (t & step) ? (b - a) : (a + b);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        x = sh[t] * rsqrt(float(head_dim));
    }
    sh[t] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One thread per 32-element group packs that group's block.
    if (t % BLOCK != 0u) return;
    uint g = t / BLOCK;
    uint dst_block = ((offset_elems + src_base) / BLOCK) + g;
    uint dst_base = dst_block * BLOCK_BYTES;
    float amax = 0.0f;
    for (uint i = 0u; i < BLOCK; i++) {
        amax = fmax(amax, fabs(sh[t + i]));
    }
    float d = amax / 7.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    half d_h = half(d);
    dst[dst_base + 0] = uchar(as_type<ushort>(d_h) & 0xFFu);
    dst[dst_base + 1] = uchar(as_type<ushort>(d_h) >> 8u);
    for (uint i = 0u; i < 16u; i++) {
        int q0 = int(round(sh[t + 2u * i] * id));
        int q1 = int(round(sh[t + 2u * i + 1u] * id));
        q0 = clamp(q0, -8, 7);
        q1 = clamp(q1, -8, 7);
        dst[dst_base + 2u + i] = uchar((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
}
"#;

/// Rotate Q by the same matrix the stored K was rotated by.
///
/// `(H S q) . (H S k) = q . k`, so this is what makes a rotated store
/// readable: every attention kernel keeps computing an ordinary dot
/// product and neither of them knows the wire changed.
const ROTATE_Q_Q4_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Same pattern as the append above and as
// `frink_quant::kv_rotation::rotation_sign`.
inline float rotation_sign(uint head_idx, uint channel) {
    uint h = head_idx * 0x9E3779B1u + channel * 0x85EBCA6Bu;
    h ^= h >> 15;
    h *= 0xC2B2AE35u;
    h ^= h >> 13;
    return (h & 1u) ? -1.0f : 1.0f;
}

kernel void rotate_q_q4(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& n_kv_heads [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    threadgroup float* sh [[threadgroup(0)]],
    uint head [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]]
) {
    // `head` walks `n_q * n_heads` in the token-major order the prefill
    // kernels read Q in, so the query head inside its token is
    // `head % n_heads`, and the KV head it is grouped onto is that
    // divided by the group size. Taking the query's own index would
    // rotate by a pattern its K was never stored under.
    uint h_in_token = head % n_heads;
    uint kv_head = h_in_token / (n_heads / n_kv_heads);
    uint base = head * head_dim;
    sh[t] = src[base + t] * rotation_sign(kv_head, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1u; step < head_dim; step <<= 1u) {
        float a = sh[t];
        float b = sh[t ^ step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sh[t] = (t & step) ? (b - a) : (a + b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    dst[base + t] = sh[t] * rsqrt(float(head_dim));
}
"#;

const DEQUANT_Q4_TO_F16_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dequant_q4_to_f16(
    device const uchar* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n_elems [[buffer(2)]],
    uint b [[thread_position_in_grid]]
) {
    const uint BLOCK = 32u;
    const uint BLOCK_BYTES = 18u;
    uint n_blocks = n_elems / BLOCK;
    if (b >= n_blocks) return;
    uint src_base = b * BLOCK_BYTES;
    ushort d_bits = ushort(src[src_base]) | (ushort(src[src_base + 1u]) << 8u);
    float d = float(as_type<half>(d_bits));
    uint dst_base = b * BLOCK;
    for (uint i = 0u; i < 16u; i++) {
        uchar byte = src[src_base + 2u + i];
        int q0 = int(char((byte & 0xFu) << 4) >> 4);
        int q1 = int(char((byte >> 4) << 4) >> 4);
        dst[dst_base + 2u * i] = half(float(q0) * d);
        dst[dst_base + 2u * i + 1u] = half(float(q1) * d);
    }
}
"#;
/// Device KV cache element type (llama.cpp `-ctk` analogue).
///
/// Selected via `FRINK_CTK` ([`metal_kv_dtype`]). Every variant names
/// the WIRE it writes, because that is the only thing a reader of a
/// stored block needs to know: three 8-bit spellings that share one
/// 34-byte layout, and one 4-bit layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalKvDtype {
    F16,
    /// ggml Q8_0: 32 int8 codes and one f16 scale, 34 bytes.
    Q8_0,
    /// 8-bit KV under the name a client may ask for it by. The store is
    /// [`Self::Q8_0`]'s: the codes are absmax-scaled int8 in
    /// `[-127, 127]`, a portable stand-in rather than real E4M3, and
    /// the module says so rather than the name implying otherwise.
    Fp8,
    /// 4-bit KV: 32 codes in 16 nibble bytes and one f16 scale, 18
    /// bytes, with the Hadamard rotation on K where the head width
    /// allows it ([`q4_rotation_viable`]). Named `q4_0` because that is
    /// llama.cpp's spelling for a 4-bit `-ctk`.
    Q4_0,
}

impl MetalKvDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Fp8 => "fp8",
            Self::Q4_0 => "q4_0",
        }
    }

    /// Exhaustive on purpose: a wire added here has to say whether it
    /// runs before it can be selected.
    pub fn is_implemented(self) -> bool {
        match self {
            Self::F16 | Self::Q8_0 | Self::Fp8 | Self::Q4_0 => true,
        }
    }

    /// True when attention must dequant store → f16 scratch before FA/GQA.
    pub fn needs_f16_scratch(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Fp8 | Self::Q4_0)
    }

    /// Uses ggml Q8_0 / fp8 34-byte blocks.
    pub(crate) fn is_q8_wire(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Fp8)
    }
}

/// Whether a 4-bit store of this geometry rotates its K.
///
/// The rotation needs a power-of-two head width for the butterfly and
/// a multiple of 32 so a head's boundary is also a quantization group's
/// boundary. A head the rotation cannot serve keeps the unrotated wire
/// rather than being refused: the unrotated 4-bit store is what this
/// file shipped before and it is still correct, just lossier.
///
/// This is the ONE place the question is answered.
/// [`MetalKvBuffers::k_rotated`](crate::attn::MetalKvBuffers) is set
/// from it at construction and every later reader, the append and the
/// query rotation, asks the buffer rather than recomputing, because a
/// store written rotated and read unrotated is a wrong answer no
/// assertion would catch.
pub fn q4_rotation_viable(head_dim: usize) -> bool {
    frink_quant::kv_rotation::rotation_viable(head_dim)
        && head_dim.is_multiple_of(frink_quant::Q4_KV_GROUP)
}

/// True when `n_kv_heads * head_dim` is a multiple of ggml Q8_0 block size (32).
pub fn metal_kv_q8_0_viable(n_kv_heads: usize, head_dim: usize) -> bool {
    (n_kv_heads * head_dim).is_multiple_of(frink_quant::Q8_0_BLOCK_ELEMS)
}

/// The 4-bit wire's 32-element group alignment.
pub fn metal_kv_q4_viable(n_kv_heads: usize, head_dim: usize) -> bool {
    (n_kv_heads * head_dim).is_multiple_of(frink_quant::Q4_KV_GROUP)
}

/// Dtype actually used for new [`MetalKvBuffers`] (unimplemented / non-viable → F16).
pub fn effective_metal_kv_dtype(n_kv_heads: usize, head_dim: usize) -> MetalKvDtype {
    let requested = metal_kv_dtype();
    if !requested.is_implemented() {
        return MetalKvDtype::F16;
    }
    if requested.is_q8_wire() && !metal_kv_q8_0_viable(n_kv_heads, head_dim) {
        static WARNED: OnceLock<()> = OnceLock::new();
        let _ = WARNED.get_or_init(|| {
            eprintln!(
                "FRINK_CTK={}: n_kv_heads*head_dim={} not divisible by {}; using f16",
                requested.as_str(),
                n_kv_heads * head_dim,
                frink_quant::Q8_0_BLOCK_ELEMS
            );
        });
        return MetalKvDtype::F16;
    }
    if requested == MetalKvDtype::Q4_0 && !metal_kv_q4_viable(n_kv_heads, head_dim) {
        static WARNED: OnceLock<()> = OnceLock::new();
        let _ = WARNED.get_or_init(|| {
            eprintln!(
                "FRINK_CTK=q4_0: n_kv_heads*head_dim={} not divisible by {}; using f16",
                n_kv_heads * head_dim,
                frink_quant::Q4_KV_GROUP
            );
        });
        return MetalKvDtype::F16;
    }
    requested
}

/// Parse `FRINK_CTK` / `-ctk`-style strings. Unknown → [`MetalKvDtype::F16`].
pub fn parse_metal_kv_dtype(raw: Option<&str>) -> MetalKvDtype {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("q8_0") | Some("q8") => MetalKvDtype::Q8_0,
        Some("fp8") | Some("e4m3") => MetalKvDtype::Fp8,
        Some("q4_0") => MetalKvDtype::Q4_0,
        Some("f16") | Some("fp16") | Some("half") | Some("bf16") => MetalKvDtype::F16,
        _ => MetalKvDtype::F16,
    }
}

/// KV dtype requested by `FRINK_CTK` (default F16).
///
/// Unimplemented dtypes emit a one-time stderr warning; callers keep F16 buffers.
pub fn metal_kv_dtype() -> MetalKvDtype {
    static DTYPE: OnceLock<MetalKvDtype> = OnceLock::new();
    *DTYPE.get_or_init(|| {
        let dt = parse_metal_kv_dtype(std::env::var("FRINK_CTK").ok().as_deref());
        if !dt.is_implemented() {
            static WARNED: OnceLock<()> = OnceLock::new();
            let _ = WARNED.get_or_init(|| {
                eprintln!(
                    "FRINK_CTK={}: Metal {} KV cache not implemented yet; using f16 buffers",
                    dt.as_str(),
                    dt.as_str()
                );
            });
        }
        dt
    })
}
/// Kernel + block geometry for one KV wire format.
///
/// One table rather than three near-identical encoders: the f16, Q8_0
/// and 4-bit appends previously restated the same threadgroup sizing,
/// the same alignment check and the same buffer bindings, which is the
/// shape that loses a fix in two of three copies.
struct KvAppendKernel {
    src: &'static str,
    name: &'static str,
    /// f32 elements one dispatched unit handles: 1 for the f16 copy, the
    /// block size for a quantized wire, which is also the alignment
    /// `offset_elems` and `n_elems` must satisfy.
    elems_per_unit: u32,
}

fn kv_append_kernel(dtype: MetalKvDtype) -> KvAppendKernel {
    if dtype.is_q8_wire() {
        return KvAppendKernel {
            src: KV_APPEND_Q8_0_KERNEL_SRC,
            name: "kv_append_q8_0",
            elems_per_unit: frink_quant::Q8_0_BLOCK_ELEMS as u32,
        };
    }
    match dtype {
        MetalKvDtype::Q4_0 => KvAppendKernel {
            src: KV_APPEND_Q4_KERNEL_SRC,
            name: "kv_append_q4",
            elems_per_unit: frink_quant::Q4_KV_GROUP as u32,
        },
        _ => KvAppendKernel {
            src: KV_APPEND_KERNEL_SRC,
            name: "kv_append",
            elems_per_unit: 1,
        },
    }
}

fn encode_dequant_q8_0_to_f16(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    n_elems: u32,
) -> Result<(), MetalError> {
    if !n_elems.is_multiple_of(frink_quant::Q8_0_BLOCK_ELEMS as u32) {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(
        device,
        DEQUANT_Q8_0_TO_F16_KERNEL_SRC,
        "dequant_q8_0_to_f16",
    )?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let mut n = n_elems;
        encoder.setBytes_length_atIndex(NonNull::new(&mut n as *mut u32 as *mut _).unwrap(), 4, 2);
    }
    let n_blocks = (n_elems as usize) / frink_quant::Q8_0_BLOCK_ELEMS;
    let tg = 256usize.min(n_blocks).max(1);
    let n_tg = n_blocks.div_ceil(tg);
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

fn encode_dequant_q4_to_f16(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    n_elems: u32,
) -> Result<(), MetalError> {
    if !n_elems.is_multiple_of(frink_quant::Q4_KV_GROUP as u32) {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, DEQUANT_Q4_TO_F16_KERNEL_SRC, "dequant_q4_to_f16")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let mut n = n_elems;
        encoder.setBytes_length_atIndex(NonNull::new(&mut n as *mut u32 as *mut _).unwrap(), 4, 2);
    }
    let n_blocks = (n_elems as usize) / frink_quant::Q4_KV_GROUP;
    let tg = 256usize.min(n_blocks).max(1);
    let n_tg = n_blocks.div_ceil(tg);
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

pub fn encode_kv_dequant_to_f16(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    dtype: MetalKvDtype,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    n_elems: u32,
) -> Result<(), MetalError> {
    match dtype {
        d if d.is_q8_wire() => encode_dequant_q8_0_to_f16(encoder, device, src, dst, n_elems),
        MetalKvDtype::Q4_0 => encode_dequant_q4_to_f16(encoder, device, src, dst, n_elems),
        _ => Err(MetalError::CommandFailed),
    }
}

/// Append this token's K and V into the layer's cache in ONE dispatch.
///
/// Both planes always land at the same `offset_elems` with the same
/// `n_elems` -- there is no caller that appends one without the other --
/// so the kernel takes the second pair of buffers and the grid's height
/// selects between them. That halves this step's encode cost, which is
/// the whole point of GitHub issue #149: the K and V appends were 32 of
/// the 242 dispatches a Llama-3.2-1B decode token encoded.
///
/// Taking both planes as parameters is also why there is no `KvPlane`
/// enum any more: a single-plane entry point would be a second code path
/// to keep in step with this one.
pub fn encode_kv_store_append(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    k_src: &ProtocolObject<dyn MTLBuffer>,
    v_src: &ProtocolObject<dyn MTLBuffer>,
    kv: &MetalKvBuffers,
    offset_elems: u32,
    n_elems: u32,
) -> Result<(), MetalError> {
    let kernel = kv_append_kernel(kv.dtype);
    let unit = kernel.elems_per_unit;
    // A quantized wire writes whole blocks, so a token that does not sit
    // on a block boundary would corrupt its neighbour. Refuse instead.
    if !offset_elems.is_multiple_of(unit) || !n_elems.is_multiple_of(unit) {
        return Err(MetalError::CommandFailed);
    }
    let units = (n_elems / unit) as usize;
    let pipe = ensure_pipeline(device, kernel.src, kernel.name)?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(k_src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&kv.k), 0, 1);
        let mut off = offset_elems;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut off as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
        let mut n = n_elems;
        encoder.setBytes_length_atIndex(NonNull::new(&mut n as *mut u32 as *mut _).unwrap(), 4, 3);
        encoder.setBuffer_offset_atIndex(Some(v_src), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&kv.v), 0, 5);
    }
    if kv.dtype == MetalKvDtype::Q4_0 {
        // Per (head, plane), because the rotation is over a whole head
        // and the quantization groups inside it share a threadgroup.
        let head_dim = kv.head_dim;
        if head_dim == 0 || !n_elems.is_multiple_of(head_dim as u32) {
            return Err(MetalError::CommandFailed);
        }
        if !offset_elems.is_multiple_of(head_dim as u32) {
            return Err(MetalError::CommandFailed);
        }
        let heads = (n_elems / head_dim as u32) as usize;
        unsafe {
            let mut hd = head_dim as u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(),
                4,
                6,
            );
            let mut rot = u32::from(kv.k_rotated);
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut rot as *mut u32 as *mut _).unwrap(),
                4,
                7,
            );
            let mut nkv = kv.n_kv_heads as u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut nkv as *mut u32 as *mut _).unwrap(),
                4,
                8,
            );
            encoder.setThreadgroupMemoryLength_atIndex(head_dim * 4, 0);
        }
        dispatch_counted(
            encoder,
            MTLSize {
                width: heads,
                height: 2,
                depth: 1,
            },
            MTLSize {
                width: head_dim,
                height: 1,
                depth: 1,
            },
        );
        return Ok(());
    }

    let tg = 256usize.min(units).max(1);
    let n_tg = units.div_ceil(tg);
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_tg,
            // Plane 0 is K, plane 1 is V.
            height: 2,
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

/// Rotate Q into `dst` by the matrix the rotated 4-bit store used.
///
/// Caller contract, and the reason this is not optional: a store whose
/// `k_rotated` is set must have its query put through this before any
/// attention kernel reads the dequantized K. The two sites that read
/// that store (`encode_gqa_with_kv` and `encode_gqa_prefill_with_kv`)
/// do it in the same block that fills the f16 scratch, so the rotated
/// K and the rotated Q are produced together or not at all.
#[allow(clippy::too_many_arguments)]
pub fn encode_rotate_q_q4(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    n_q: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
) -> Result<(), MetalError> {
    if head_dim == 0 || n_kv_heads == 0 || n_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, ROTATE_Q_Q4_KERNEL_SRC, "rotate_q_q4")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let mut hd = head_dim;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 2);
        let mut nkv = n_kv_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut nkv as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut nh = n_heads;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nh as *mut u32 as *mut _).unwrap(), 4, 4);
        encoder.setThreadgroupMemoryLength_atIndex(head_dim as usize * 4, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: (n_q as usize) * (n_heads as usize),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: head_dim as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Every (source, entry point) this dtype's store needs warmed, in one
/// place.
///
/// The prefill warmup used to restate the dtype-to-kernel mapping that
/// [`kv_append_kernel`] and [`encode_kv_dequant_to_f16`] already hold,
/// which is the two-structures-that-must-agree shape: a dtype whose
/// append kernel changed here would have kept warming the old one
/// there. Derived from the same table instead.
pub fn kv_wire_pipelines(dtype: MetalKvDtype) -> Vec<(&'static str, &'static str)> {
    let append = kv_append_kernel(dtype);
    let mut out = vec![(append.src, append.name)];
    if dtype.needs_f16_scratch() {
        out.push(dequant_kernel(dtype));
    }
    out
}

/// The dequant-to-f16 kernel for a dtype that needs the scratch.
///
/// Exhaustive on purpose: a new wire format has to say which kernel
/// reads it back rather than inheriting one.
fn dequant_kernel(dtype: MetalKvDtype) -> (&'static str, &'static str) {
    match dtype {
        MetalKvDtype::Q8_0 | MetalKvDtype::Fp8 => {
            (DEQUANT_Q8_0_TO_F16_KERNEL_SRC, "dequant_q8_0_to_f16")
        }
        MetalKvDtype::Q4_0 => (DEQUANT_Q4_TO_F16_KERNEL_SRC, "dequant_q4_to_f16"),
        MetalKvDtype::F16 => {
            unreachable!("{} does not use the f16 scratch", dtype.as_str())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both 4-bit kernels have to COMPILE, and a compile failure used
    /// to surface as `Command encoder released without endEncoding` in
    /// whatever test ran next, because the error travels up through a
    /// `?` that skips `endEncoding`.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn q4_kernels_compile() {
        let shared = crate::gpu::shared_metal().expect("metal");
        for (src, name) in [
            (KV_APPEND_Q4_KERNEL_SRC, "kv_append_q4"),
            (ROTATE_Q_Q4_KERNEL_SRC, "rotate_q_q4"),
        ] {
            ensure_pipeline(&shared.device, src, name).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
