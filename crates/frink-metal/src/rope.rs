//! RoPE on Metal: the pairing/width/scale type the kernels take, the
//! per-layer base-and-divisors type, the decode (one position) and
//! prefill (many positions) kernels, their encoders, and the host-upload
//! launchers the parity tests use.
//!
//! Moved out of `attn.rs` along the seam GitHub issue #149's kernel
//! follow-up touches first: the decode kernel dispatched one THREAD per
//! head and looped over every rotary pair inside it, which is why RoPE
//! cost a Gemma-2-2B token more GPU time than its attention did.

use crate::attn::upload_f32;
use crate::dispatch::dispatch_counted;
use crate::gpu::{ensure_pipeline, shared_metal, MetalError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLSize,
};
use std::ptr::NonNull;
use std::sync::Arc;

/// RoPE pairing convention for Metal kernels. Mirrors
/// `frink_models::config::RopeLayout` / llama.cpp `llama_rope_type`
/// without pulling models into this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalRopeLayout {
    /// Adjacent pairs `(2*i, 2*i+1)` — `LLAMA_ROPE_TYPE_NORM`.
    Norm,
    /// Split-half pairs `(i, i+half)` — `LLAMA_ROPE_TYPE_NEOX`.
    Neox,
}

/// Everything the Metal RoPE kernels need beyond the base frequency:
/// the pairing convention, the rotary width, and ggml `rope_yarn`'s
/// magnitude scale.
///
/// Threaded as one value because all three are properties of the same
/// `ggml_rope_ext` call. `rot_dim` is ggml `n_dims` (`hparams.n_rot`,
/// GGUF `<arch>.rope.dimension_count`): channels `[n_rot, head_dim)` are
/// copied through untouched, exactly as `kernel_rope_norm`'s else-branch
/// does. `attn_factor` is ggml's `mscale`, folded into `cos`/`sin`
/// inside `rope_yarn`, so it *cannot* reach the pass-through channels —
/// scaling the whole head instead is a different graph, and was one on
/// the CPU side until `frink parity` caught it (Phi-4-mini, 96 of 128
/// dims rotated at `attn_factor` 1.1902).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalRope {
    pub layout: MetalRopeLayout,
    /// Rotary width when narrower than `head_dim`; `None` = whole head.
    pub rot_dim: Option<usize>,
    /// ggml `rope_yarn` `mscale` / llama.cpp `yarn_attn_factor`.
    /// `1.0` for every architecture that does not set the key.
    pub attn_factor: f32,
}

impl MetalRope {
    /// Whole-head rotation with no magnitude scale — what every
    /// architecture except the Phi-3/Phi-4 family wants.
    pub fn new(layout: MetalRopeLayout) -> Self {
        Self {
            layout,
            rot_dim: None,
            attn_factor: 1.0,
        }
    }

    /// Same rotation, but `attn_factor` already multiplied into q/k by
    /// the caller. Rotation is linear, so pre-scaling the rotated
    /// channels host-side is identical to folding `mscale` into
    /// `cos`/`sin` here — but doing both would square it.
    pub fn attn_factor_applied_by_caller(self) -> Self {
        Self {
            attn_factor: 1.0,
            ..self
        }
    }

    /// `n_dims` as the kernels want it: `0` means "whole head".
    fn rot_dim_uniform(&self) -> u32 {
        self.rot_dim.unwrap_or(0) as u32
    }
}

/// The half of a RoPE call that varies from LAYER to layer: the
/// frequency base and the per-band divisors. [`MetalRope`] carries the
/// half that does not (pairing, rotary width, magnitude scale), so a
/// fused stack takes one `MetalRope` and one of these per layer.
///
/// Both halves live in one struct because they vary TOGETHER and
/// llama.cpp varies them together (`llama-model.cpp:2029-2035`,
/// mirrored by `frink_models::config::ModelConfig::layer_rope`).
/// Splitting them is what this type exists to prevent: the fused stacks
/// used to take a per-layer `rope_theta` beside ONE `freq_factors`
/// slice for the whole run, so a model whose sliding layers scale
/// differently from its full-attention ones (Gemma-3 4B/12B/27B:
/// `rope_scaling {linear, factor 8}` on the full layers, unscaled on
/// the sliding ones) could not ride them at all. Answering the base
/// question without answering the divisor question no longer compiles.
///
/// **An `Option<LayerRope>` of `None` means this layer does not rotate
/// at all** -- llama.cpp's per-layer `use_rope`, which six
/// architectures upstream gate (`frink_models::rope_layers`). It is
/// spelled as the absence of the whole struct rather than as a `bool`
/// beside it, so a fused stack cannot take a base and divisors for a
/// layer it must not rotate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerRope<'a> {
    /// This layer's RoPE frequency base (`rope_theta`, or
    /// `rope_theta_swa` on a sliding layer).
    pub theta: f32,
    /// This layer's per-band divisors (`rope_freqs.weight`, folded with
    /// any linear `freq_scale`), `n_rot/2` long. `None` = divide by
    /// nothing.
    pub freq_factors: Option<&'a Arc<[f32]>>,
}

/// [`LayerRope`] as the private encoders take it, once the divisors are
/// already a resident device buffer: the frequency base and that buffer,
/// or `None` for a layer that does not rotate.
///
/// One `Option` around the pair rather than an `Option<f32>` beside an
/// `Option<&Buffer>`, because "no base" and "no divisors" are different
/// questions and only the outer one decides whether a dispatch happens.
pub(crate) type EncodedRope<'a> = Option<(f32, Option<&'a ProtocolObject<dyn MTLBuffer>>)>;

// Norm (interleaved) and NeoX (split-half) pairing are one templated
// kernel, instantiated twice, so `encode_rope` only swaps the entry
// point. Math mirrors `frink_core::attention::{apply_rope_interleaved,
// apply_rope}` -- no Candle / third-party RoPE dependency.
//
// ONE THREAD PER ROTARY PAIR. The previous kernels dispatched one thread
// per head -- threadgroups of width 1 -- and looped over every pair
// inside it, `pow` + `sin` + `cos` per iteration: 128 serial
// transcendentals per thread on a 256-wide head, on 1/32 of each SIMD
// unit. Measured serialized (`crate::kernel_bench`) that was 61.6 us
// for Gemma-2-2B's 8+4 heads and 21.4 us for Llama-3.2-3B's 24+8,
// against llama.cpp's 2 x 2.8 us for the same rotation. The per-pair
// arithmetic is unchanged, expression for expression, so the result
// is the same; only the thread it runs on differs.
//
// Two destinations (buffers 0 and 9) because every decode call site
// ropes Q and then K with the same theta, position and freq factors,
// and RoPE touches each head independently, so the two are one
// dispatch (GitHub issue #149). A caller with one destination passes
// `n_heads2 = 0`. `n_tokens > 1` is the prefill shape, heads packed
// `[token][head][head_dim]` from `base_pos`; decode is `n_tokens = 1`.
const ROPE_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// NEOX = false: adjacent pairs `(2i, 2i+1)`, ggml `kernel_rope_norm`.
// NEOX = true: split-half pairs `(i, i + n_dims/2)`, `kernel_rope_neox`.
// ggml `n_dims` (`rot_dim`) scopes both the pairing and the frequency
// exponent: channels `[n_dims, head_dim)` pass through untouched, and
// `mscale` (ggml `rope_yarn`'s attn_factor) is folded into cos/sin so
// it reaches the rotated channels only.
template <bool NEOX>
kernel void rope_pairs(
    device float* vecs [[buffer(0)]],
    constant uint& n_heads [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    constant uint& base_pos [[buffer(4)]],
    device const float* freq_factors [[buffer(5)]],
    constant uint& use_freq_factors [[buffer(6)]],
    constant uint& rot_dim [[buffer(7)]],
    constant float& mscale [[buffer(8)]],
    device float* vecs2 [[buffer(9)]],
    constant uint& n_heads2 [[buffer(10)]],
    constant uint& n_tokens [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    uint rot = (rot_dim == 0u || rot_dim > head_dim) ? head_dim : rot_dim;
    uint half_dim = rot / 2u;
    if (i >= half_dim || h >= n_heads + n_heads2 || t >= n_tokens) return;
    // Heads [0, n_heads) rotate `vecs`; [n_heads, n_heads + n_heads2)
    // rotate `vecs2`. Distinct buffers, so the two ranges never alias.
    const bool second = h >= n_heads;
    device float* base = second ? vecs2 : vecs;
    const uint head = second ? (h - n_heads) : h;
    const uint heads_here = second ? n_heads2 : n_heads;
    device float* vec = base + (t * heads_here + head) * head_dim;
    const uint pos = base_pos + t;
    float freq = 1.0f / pow(theta, (2.0f * float(i)) / float(rot));
    float angle = float(pos) * freq;
    if (use_freq_factors != 0u) {
        angle /= freq_factors[i];
    }
    float s = sin(angle) * mscale;
    float c = cos(angle) * mscale;
    const uint ia = NEOX ? i : 2u * i;
    const uint ib = NEOX ? (i + half_dim) : (2u * i + 1u);
    float a = vec[ia];
    float b = vec[ib];
    vec[ia] = a * c - b * s;
    vec[ib] = a * s + b * c;
}

template [[host_name("rope_norm_pairs")]] kernel void rope_pairs<false>(
    device float*, constant uint&, constant uint&, constant float&, constant uint&,
    device const float*, constant uint&, constant uint&, constant float&, device float*,
    constant uint&, constant uint&, uint3);
template [[host_name("rope_neox_pairs")]] kernel void rope_pairs<true>(
    device float*, constant uint&, constant uint&, constant float&, constant uint&,
    device const float*, constant uint&, constant uint&, constant float&, device float*,
    constant uint&, constant uint&, uint3);
"#;

/// The kernel (source, entry point) for a pairing convention. One
/// table, read by both encoders and by the prefill pipeline warm-up;
/// one kernel serves decode (`n_tokens = 1`) and prefill.
pub(crate) fn rope_kernel(layout: MetalRopeLayout) -> (&'static str, &'static str) {
    match layout {
        MetalRopeLayout::Norm => (ROPE_KERNEL_SRC, "rope_norm_pairs"),
        MetalRopeLayout::Neox => (ROPE_KERNEL_SRC, "rope_neox_pairs"),
    }
}

/// Threads per threadgroup along the rotary-pair axis. Pairs per head
/// are 32 (d=64) to 128 (d=256), so 32 keeps every threadgroup full
/// for every head width the kernels serve and never idles a lane on a
/// partial-rotary width such as Phi-3's 48.
const ROPE_PAIRS_PER_THREADGROUP: usize = 32;

/// ggml sizes `rope_freqs` (`src2` in `kernel_rope_norm`) by the ROTARY
/// width: it is indexed `[i0/2]` for `i0 < n_dims`, so it carries
/// `n_rot/2` entries, which is narrower than `head_dim/2` under partial
/// rotary. Checking it against the head width instead rejects every
/// Phi-3/Phi-4 checkpoint at the door.
pub(crate) fn assert_freq_factors_len(
    freq_factors: Option<&[f32]>,
    rope: MetalRope,
    head_dim: usize,
) {
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), rope.rot_dim.unwrap_or(head_dim) / 2);
    }
}

#[allow(clippy::too_many_arguments)]
/// One RoPE destination: `n_heads` contiguous `head_dim` f32 heads.
pub(crate) struct RopeTarget<'a> {
    pub(crate) vecs: &'a ProtocolObject<dyn MTLBuffer>,
    pub(crate) n_heads: u32,
}

/// RoPE one or two destinations in a SINGLE dispatch.
///
/// Every call site rotates Q and then K with the same `theta`, `pos`,
/// `head_dim` and `freq_factors`, into different buffers, and RoPE is
/// independent per head -- so the pair is one dispatch, not two. Taking
/// `second` as a parameter rather than adding a fused twin is what keeps
/// the single- and two-destination paths from drifting: there is one
/// kernel and one encoder, and `None` simply makes the second range
/// empty.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rope(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    rope: MetalRope,
    first: RopeTarget<'_>,
    second: Option<RopeTarget<'_>>,
    head_dim: u32,
    theta: f32,
    pos: u32,
    freq_factors: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<(), MetalError> {
    encode_rope_grid(
        encoder,
        device,
        rope,
        first,
        second,
        head_dim,
        theta,
        pos,
        1,
        freq_factors,
    )
}

/// RoPE `n_tokens` consecutive positions from `base_pos` in one
/// dispatch: `vecs` is `[n_tokens, n_heads, head_dim]`. The prefill
/// shape of [`encode_rope`], on the same kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rope_batch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    rope: MetalRope,
    vecs: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    head_dim: u32,
    theta: f32,
    base_pos: u32,
    n_tokens: u32,
    freq_factors: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<(), MetalError> {
    if n_tokens == 0 {
        return Ok(());
    }
    encode_rope_grid(
        encoder,
        device,
        rope,
        RopeTarget { vecs, n_heads },
        None,
        head_dim,
        theta,
        base_pos,
        n_tokens,
        freq_factors,
    )
}

/// The one encoder behind [`encode_rope`] and [`encode_rope_batch`]:
/// a grid of (rotary pair, head across both destinations, token).
#[allow(clippy::too_many_arguments)]
fn encode_rope_grid(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    rope: MetalRope,
    first: RopeTarget<'_>,
    second: Option<RopeTarget<'_>>,
    head_dim: u32,
    theta: f32,
    base_pos: u32,
    n_tokens: u32,
    freq_factors: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<(), MetalError> {
    let vecs = first.vecs;
    let n_heads = first.n_heads;
    // With no second destination the kernel's `h < n_heads` branch is the
    // only reachable one, so binding `vecs` again at index 9 is a valid
    // buffer the kernel never reads.
    let (vecs2, n_heads2) = match &second {
        Some(t) => (t.vecs, t.n_heads),
        None => (vecs, 0),
    };
    let (src, name) = rope_kernel(rope.layout);
    let pipe = ensure_pipeline(device, src, name)?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(vecs), 0, 0);
        let mut n_heads_u = n_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_heads_u as *mut u32 as *mut _).unwrap(),
            4,
            1,
        );
        let mut head_dim_u = head_dim;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut head_dim_u as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
        let mut theta_f = theta;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut theta_f as *mut f32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut pos_u = base_pos;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut pos_u as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        if let Some(ff) = freq_factors {
            encoder.setBuffer_offset_atIndex(Some(ff), 0, 5);
            let mut use_ff = 1u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut use_ff as *mut u32 as *mut _).unwrap(),
                4,
                6,
            );
        } else {
            // Unused device buffer slot — bind a 4-byte scratch so index 5 is valid.
            let mut scratch = [0u8; 4];
            encoder.setBytes_length_atIndex(
                NonNull::new(scratch.as_mut_ptr() as *mut _).unwrap(),
                4,
                5,
            );
            let mut use_ff = 0u32;
            encoder.setBytes_length_atIndex(
                NonNull::new(&mut use_ff as *mut u32 as *mut _).unwrap(),
                4,
                6,
            );
        }
        let mut rot_dim_u = rope.rot_dim_uniform();
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut rot_dim_u as *mut u32 as *mut _).unwrap(),
            4,
            7,
        );
        let mut mscale_f = rope.attn_factor;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut mscale_f as *mut f32 as *mut _).unwrap(),
            4,
            8,
        );
        encoder.setBuffer_offset_atIndex(Some(vecs2), 0, 9);
        let mut n_heads2_u = n_heads2;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_heads2_u as *mut u32 as *mut _).unwrap(),
            4,
            10,
        );
        let mut n_tokens_u = n_tokens;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_tokens_u as *mut u32 as *mut _).unwrap(),
            4,
            11,
        );
    }
    // The pair axis is sized from the HEAD width, an upper bound on the
    // rotary width; the kernel derives the rotary width from `rot_dim`
    // and exits past it, so the rule lives in one place.
    let pairs_upper = (head_dim as usize).div_ceil(2);
    dispatch_counted(
        encoder,
        MTLSize {
            width: pairs_upper.div_ceil(ROPE_PAIRS_PER_THREADGROUP),
            height: (n_heads + n_heads2) as usize,
            depth: n_tokens as usize,
        },
        MTLSize {
            width: ROPE_PAIRS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Host-upload RoPE only (parity testing). Applies `layout` RoPE in-place
/// across `n_heads` packed heads in `vecs` (`n_heads * head_dim`).
pub fn launch_rope_heads_host(
    vecs: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    layout: MetalRope,
    theta: f32,
    pos: usize,
    freq_factors: Option<&[f32]>,
) -> Result<(), MetalError> {
    assert_eq!(vecs.len(), n_heads * head_dim);
    assert_freq_factors_len(freq_factors, layout, head_dim);
    let shared = shared_metal()?;
    let device = &shared.device;
    let buf = upload_f32(device, vecs)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };
    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_rope(
        &encoder,
        device,
        layout,
        RopeTarget {
            vecs: &buf,
            n_heads: n_heads as u32,
        },
        None,
        head_dim as u32,
        theta,
        pos as u32,
        ff_buf.as_deref(),
    )?;
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    let ptr = buf.contents();
    let out = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, vecs.len()) };
    vecs.copy_from_slice(out);
    Ok(())
}

/// Host-upload multi-pos RoPE (parity testing). Layout `[n_tokens, n_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn launch_rope_heads_batch_host(
    vecs: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    n_tokens: usize,
    layout: MetalRope,
    theta: f32,
    base_pos: usize,
    freq_factors: Option<&[f32]>,
) -> Result<(), MetalError> {
    assert_eq!(vecs.len(), n_tokens * n_heads * head_dim);
    assert_freq_factors_len(freq_factors, layout, head_dim);
    if n_tokens == 0 {
        return Ok(());
    }
    let shared = shared_metal()?;
    let device = &shared.device;
    let buf = upload_f32(device, vecs)?;
    let ff_buf = match freq_factors {
        Some(ff) => Some(upload_f32(device, ff)?),
        None => None,
    };
    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_rope_batch(
        &encoder,
        device,
        layout,
        &buf,
        n_heads as u32,
        head_dim as u32,
        theta,
        base_pos as u32,
        n_tokens as u32,
        ff_buf.as_deref(),
    )?;
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    let ptr = buf.contents();
    let out = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, vecs.len()) };
    vecs.copy_from_slice(out);
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn cpu_rope_norm(
        vec: &mut [f32],
        pos: usize,
        theta: f32,
        freq_factors: Option<&[f32]>,
    ) {
        let dim = vec.len();
        let half = dim / 2;
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
            let angle = match freq_factors {
                Some(ff) => pos as f32 * freq / ff[i],
                None => pos as f32 * freq,
            };
            let (sin, cos) = angle.sin_cos();
            let a = vec[2 * i];
            let b = vec[2 * i + 1];
            vec[2 * i] = a * cos - b * sin;
            vec[2 * i + 1] = a * sin + b * cos;
        }
    }

    fn cpu_rope_neox(vec: &mut [f32], pos: usize, theta: f32, freq_factors: Option<&[f32]>) {
        let dim = vec.len();
        let half = dim / 2;
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
            let angle = match freq_factors {
                Some(ff) => pos as f32 * freq / ff[i],
                None => pos as f32 * freq,
            };
            let (sin, cos) = angle.sin_cos();
            let a = vec[i];
            let b = vec[i + half];
            vec[i] = a * cos - b * sin;
            vec[i + half] = a * sin + b * cos;
        }
    }

    /// One head, exactly as `frink_models::Decoder` composes it on the
    /// CPU: `apply_rope_attn_factor` scales the rotated channels, then
    /// `apply_rope_head_theta` rotates `[0, n_rot)` and leaves the tail
    /// alone. Pre-scaling and folding `mscale` into cos/sin are the same
    /// thing (rotation is linear) — which is the property the Metal
    /// kernel has to reproduce.
    fn cpu_rope_head(vec: &mut [f32], rope: MetalRope, pos: usize, theta: f32, ff: Option<&[f32]>) {
        let head_dim = vec.len();
        let rot = rope.rot_dim.unwrap_or(head_dim).min(head_dim);
        for v in vec[..rot].iter_mut() {
            *v *= rope.attn_factor;
        }
        let slice = &mut vec[..rot];
        match rope.layout {
            MetalRopeLayout::Norm => cpu_rope_norm(slice, pos, theta, ff),
            MetalRopeLayout::Neox => cpu_rope_neox(slice, pos, theta, ff),
        }
    }

    fn assert_rope_parity_with(rope: MetalRope, head_dim: usize, with_ff: bool) {
        let n_heads = 3;
        let pos = 5usize;
        let theta = 10000.0f32;
        let rot = rope.rot_dim.unwrap_or(head_dim);
        let ff: Option<Vec<f32>> = if with_ff {
            Some((0..rot / 2).map(|i| 0.8 + i as f32 * 0.15).collect())
        } else {
            None
        };
        let src: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let mut cpu = src.clone();
        let mut gpu = src.clone();
        for h in 0..n_heads {
            let slice = &mut cpu[h * head_dim..(h + 1) * head_dim];
            cpu_rope_head(slice, rope, pos, theta, ff.as_deref());
        }
        launch_rope_heads_host(&mut gpu, n_heads, head_dim, rope, theta, pos, ff.as_deref())
            .expect("metal rope");
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "{rope:?} hd={head_dim} ff={with_ff} elem {i}: cpu={a} gpu={b} tol={tol}"
            );
        }
        // The pass-through tail is not "close to" the input, it IS the
        // input: ggml copies `[n_rot, ne0)` and `mscale` cannot reach it.
        if rot < head_dim {
            for h in 0..n_heads {
                for d in rot..head_dim {
                    let i = h * head_dim + d;
                    assert_eq!(
                        gpu[i], src[i],
                        "{rope:?} pass-through channel {d} of head {h} must be untouched"
                    );
                }
            }
        }
    }

    fn assert_rope_parity(layout: MetalRopeLayout, with_ff: bool) {
        assert_rope_parity_with(MetalRope::new(layout), 8, with_ff);
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_norm_matches_cpu() {
        assert_rope_parity(MetalRopeLayout::Norm, false);
        assert_rope_parity(MetalRopeLayout::Norm, true);
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_neox_matches_cpu() {
        assert_rope_parity(MetalRopeLayout::Neox, false);
        assert_rope_parity(MetalRopeLayout::Neox, true);
    }

    /// Partial rotary (`n_rot < head_dim`, Phi-3/Phi-4's 96 of 128): the
    /// rotated prefix must match the CPU decoder and the tail must come
    /// back bit-identical. Both layouts, because NeoX's split-half
    /// pairing is over `n_rot`, so getting the width wrong there
    /// re-pairs channels rather than merely rotating too many.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_partial_rotary_matches_cpu() {
        for layout in [MetalRopeLayout::Norm, MetalRopeLayout::Neox] {
            for with_ff in [false, true] {
                let rope = MetalRope {
                    rot_dim: Some(6),
                    ..MetalRope::new(layout)
                };
                assert_rope_parity_with(rope, 8, with_ff);
            }
        }
    }

    /// `rot_dim == head_dim` must be the same graph as `None`, so a
    /// checkpoint whose `rope.dimension_count` equals the head width
    /// cannot take a different path from one that omits the key.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_rot_dim_equal_to_head_dim_is_whole_head() {
        for layout in [MetalRopeLayout::Norm, MetalRopeLayout::Neox] {
            assert_rope_parity_with(
                MetalRope {
                    rot_dim: Some(8),
                    ..MetalRope::new(layout)
                },
                8,
                false,
            );
        }
    }

    /// The bug the CPU side shipped once and `frink parity` caught:
    /// `attn_factor` is ggml's `mscale`, folded into cos/sin inside
    /// `rope_yarn`, so it can only reach the ROTATED channels. A kernel
    /// that scales the whole head is a different model.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_mscale_scales_only_the_rotated_channels() {
        let head_dim = 8;
        let rot = 4;
        let n_heads = 2;
        let src: Vec<f32> = (0..n_heads * head_dim).map(|i| 1.0 + i as f32).collect();

        for layout in [MetalRopeLayout::Norm, MetalRopeLayout::Neox] {
            let rope = MetalRope {
                layout,
                rot_dim: Some(rot),
                attn_factor: 1.1902381,
            };
            // Position 0: cos = 1, sin = 0, so the rotated channels come
            // out as exactly `x * mscale` and the magnitude scale is
            // readable straight off the output.
            let mut gpu = src.clone();
            launch_rope_heads_host(&mut gpu, n_heads, head_dim, rope, 10000.0, 0, None)
                .expect("metal rope");
            for h in 0..n_heads {
                for d in 0..rot {
                    let i = h * head_dim + d;
                    let want = src[i] * rope.attn_factor;
                    assert!(
                        (gpu[i] - want).abs() <= 1e-4 * want.abs(),
                        "{layout:?} rotated channel {d} of head {h}: got {} want {want}",
                        gpu[i]
                    );
                }
                for d in rot..head_dim {
                    let i = h * head_dim + d;
                    assert_eq!(
                        gpu[i], src[i],
                        "{layout:?} pass-through channel {d} of head {h} must NOT take mscale"
                    );
                }
            }
            // And with no partial rotary the whole head takes it, so the
            // narrow case cannot quietly become the rule.
            let whole = MetalRope {
                rot_dim: None,
                ..rope
            };
            let mut gpu = src.clone();
            launch_rope_heads_host(&mut gpu, n_heads, head_dim, whole, 10000.0, 0, None)
                .expect("metal rope");
            for (i, (g, s)) in gpu.iter().zip(src.iter()).enumerate() {
                let want = s * whole.attn_factor;
                assert!(
                    (g - want).abs() <= 1e-4 * want.abs(),
                    "{layout:?} elem {i}: got {g} want {want}"
                );
            }
        }
    }

    fn assert_rope_batch_parity_with(rope: MetalRope, head_dim: usize, with_ff: bool) {
        let n_heads = 3;
        let n_tokens = 4;
        let base_pos = 2usize;
        let theta = 10000.0f32;
        let rot = rope.rot_dim.unwrap_or(head_dim);
        let ff: Option<Vec<f32>> = if with_ff {
            Some((0..rot / 2).map(|i| 0.8 + i as f32 * 0.15).collect())
        } else {
            None
        };
        let src: Vec<f32> = (0..n_tokens * n_heads * head_dim)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let mut cpu = src.clone();
        let mut gpu = src.clone();
        for t in 0..n_tokens {
            let pos = base_pos + t;
            for h in 0..n_heads {
                let off = (t * n_heads + h) * head_dim;
                let slice = &mut cpu[off..off + head_dim];
                cpu_rope_head(slice, rope, pos, theta, ff.as_deref());
            }
        }
        launch_rope_heads_batch_host(
            &mut gpu,
            n_heads,
            head_dim,
            n_tokens,
            rope,
            theta,
            base_pos,
            ff.as_deref(),
        )
        .expect("metal rope batch");
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "{rope:?} batch hd={head_dim} ff={with_ff} elem {i}: cpu={a} gpu={b} tol={tol}"
            );
        }
        if rot < head_dim {
            for t in 0..n_tokens {
                for h in 0..n_heads {
                    for d in rot..head_dim {
                        let i = (t * n_heads + h) * head_dim + d;
                        assert_eq!(
                            gpu[i], src[i],
                            "{rope:?} batch pass-through channel {d} (tok {t}, head {h})"
                        );
                    }
                }
            }
        }
    }

    fn assert_rope_batch_parity(layout: MetalRopeLayout, with_ff: bool) {
        assert_rope_batch_parity_with(MetalRope::new(layout), 8, with_ff);
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_batch_norm_matches_cpu() {
        assert_rope_batch_parity(MetalRopeLayout::Norm, false);
        assert_rope_batch_parity(MetalRopeLayout::Norm, true);
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_batch_neox_matches_cpu() {
        assert_rope_batch_parity(MetalRopeLayout::Neox, false);
        assert_rope_batch_parity(MetalRopeLayout::Neox, true);
    }

    /// Prefill's batched kernel carries the same two uniforms as the
    /// decode one; a `rot_dim`/`mscale` that only reached decode would
    /// give Metal prefill and Metal decode different RoPE.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_batch_partial_rotary_and_mscale_match_cpu() {
        for layout in [MetalRopeLayout::Norm, MetalRopeLayout::Neox] {
            for with_ff in [false, true] {
                let rope = MetalRope {
                    layout,
                    rot_dim: Some(6),
                    attn_factor: 1.1902381,
                };
                assert_rope_batch_parity_with(rope, 8, with_ff);
            }
        }
    }

    /// Phi-4-mini's real RoPE shape, at positions that cross its
    /// `rope.scaling.original_context_length` (4096) — the boundary
    /// where `ModelConfig::apply_runtime_context` switches LongRoPE from
    /// the short factor set to the long one. The long set is the harder
    /// case *and* the one every default-context run picks, so it is what
    /// this pins: 128-wide heads with 96 rotated, `attn_factor`
    /// 1.1902381, and a 48-entry factor vector rising to ~47.8 exactly
    /// as `rope_factors_long.weight` does.
    ///
    /// Tolerance is absolute, not relative to 1e-4: at position ~4100
    /// the lowest band's angle is ~4100 radians, where an f32 mantissa
    /// already costs ~2.4e-4 rad of argument-reduction error, and Metal
    /// and Rust do not reduce identically. Anything the `rot_dim` /
    /// `mscale` wiring could get wrong is orders of magnitude larger —
    /// the mutation check moves these elements by whole units.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rope_phi_long_context_factors_match_cpu_past_orig_ctx() {
        let head_dim = 128usize;
        let rot = 96usize;
        let n_heads = 2usize;
        let n_tokens = 16usize;
        // Straddles 4096: the run this stands in for is one whose
        // context exceeds `original_context_length`.
        let base_pos = 4090usize;
        let theta = 10000.0f32;
        let rope = MetalRope {
            layout: MetalRopeLayout::Neox,
            rot_dim: Some(rot),
            attn_factor: 1.1902381,
        };
        let ff: Vec<f32> = (0..rot / 2)
            .map(|i| 1.0 + (i as f32 / ((rot / 2 - 1) as f32)).powf(3.0) * 46.77)
            .collect();

        let src: Vec<f32> = (0..n_tokens * n_heads * head_dim)
            .map(|i| (i as f32 * 0.037).sin())
            .collect();
        let mut cpu = src.clone();
        let mut gpu = src.clone();
        for t in 0..n_tokens {
            for h in 0..n_heads {
                let off = (t * n_heads + h) * head_dim;
                cpu_rope_head(
                    &mut cpu[off..off + head_dim],
                    rope,
                    base_pos + t,
                    theta,
                    Some(&ff),
                );
            }
        }
        launch_rope_heads_batch_host(
            &mut gpu,
            n_heads,
            head_dim,
            n_tokens,
            rope,
            theta,
            base_pos,
            Some(&ff),
        )
        .expect("metal rope batch");

        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 3e-3,
                "long-context elem {i}: cpu={a} gpu={b}"
            );
        }
        for t in 0..n_tokens {
            for h in 0..n_heads {
                for d in rot..head_dim {
                    let i = (t * n_heads + h) * head_dim + d;
                    assert_eq!(
                        gpu[i], src[i],
                        "long-context pass-through channel {d} (tok {t}, head {h})"
                    );
                }
            }
        }
    }
}
