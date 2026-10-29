//! PrismML's folded Hadamard rotation, on the device.
//!
//! A Bonsai weight carries `W S H`, so every matmul against it has to
//! see `perm -> signs -> FWHT` applied to its activation first
//! (`frink_core::weight_matrix::hadamard`, which is the same three
//! steps and the definition this kernel is checked against). Doing that
//! on the host cost two things per matmul: the butterfly itself (12% of
//! a Bonsai-2-27B decode step, measured) and a fresh upload of the
//! transformed vector, which is what kept each projection in a command
//! buffer of its own.
//!
//! Encoded into the SAME command buffer as the matvec it feeds, the
//! rotation is a prologue rather than a round trip, and a caller that
//! already has `x` on the device never brings it back.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::gpu::{ensure_pipeline, MetalError};

/// The three facts a rotation needs, in the order they apply.
///
/// Mirrors `frink_core::weight_matrix::hadamard::HadamardFold` field
/// for field. It is passed rather than read from a global because the
/// same vector can feed matrices with different folds, and a kernel
/// that guessed would be wrong in a way only the logits show.
#[derive(Debug, Clone, Copy)]
pub struct FoldPlan<'a> {
    /// Walsh-Hadamard block width, a power of two. `1024` for Bonsai.
    pub block: usize,
    /// One `+1` / `-1` per input element, or `None` for the identity
    /// sign pattern.
    pub signs: Option<&'a [f32]>,
    /// `ssm_out`'s tiled-to-grouped head permutation `(hd, nk, rep)`,
    /// applied BEFORE the signs, or `None` when the input order is
    /// already the one the weights were folded in.
    pub perm: Option<(usize, usize, usize)>,
}

impl FoldPlan<'_> {
    /// Rejects a plan this kernel cannot serve, rather than rotating by
    /// the wrong matrix. `width` is the vector the plan will be applied
    /// to.
    pub fn check(&self, width: usize) -> Result<(), MetalError> {
        if self.block == 0 || !self.block.is_power_of_two() || self.block > MAX_BLOCK {
            return Err(MetalError::CommandFailed);
        }
        if width == 0 || !width.is_multiple_of(self.block) {
            return Err(MetalError::CommandFailed);
        }
        if let Some(signs) = self.signs {
            if signs.len() != width {
                return Err(MetalError::CommandFailed);
            }
        }
        if let Some((hd, nk, rep)) = self.perm {
            if hd == 0 || nk == 0 || rep == 0 || hd * nk * rep != width {
                return Err(MetalError::CommandFailed);
            }
        }
        Ok(())
    }
}

/// The widest block one threadgroup can hold in the scratch below.
/// Bonsai's is 1024; a file declaring more is refused rather than
/// silently split, because a split butterfly is a different matrix.
pub const MAX_BLOCK: usize = 1024;

/// Threads per group. Each handles `block / (2 * THREADS)` butterflies
/// per stage, so one group covers a whole block and the stages can
/// synchronise with a threadgroup barrier instead of a second dispatch.
const THREADS: usize = 256;

/// `perm -> signs -> FWHT`, in place, one threadgroup per block.
///
/// The permutation is a GATHER at load: element `g` of the output takes
/// `d + hd * (k + nk * r)` where `d = g % hd`, `r = (g / hd) % rep` and
/// `k = (g / hd) / rep`, which is `tiled_to_grouped` read backwards.
/// Doing it on load is what keeps the whole rotation one pass; doing it
/// as its own dispatch would need a second buffer for the same answer.
pub const HADAMARD_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void hadamard_fold(
    device float* x [[buffer(0)]],
    device const float* signs [[buffer(1)]],
    constant uint& block [[buffer(2)]],
    constant uint& has_signs [[buffer(3)]],
    constant uint4& perm [[buffer(4)]],
    constant uint& row_width [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]
) {
    threadgroup float tile[1024];
    const uint base = tgid * block;
    // A batch is R rows of `row_width`, laid out contiguously; the
    // rotation is per ROW, so the signs and the permutation index
    // within a row and the row's own offset is added back. One row is
    // the same case with `row_width` equal to the whole vector.
    const uint row_base = (base / row_width) * row_width;
    // perm.w is 1 when a permutation applies; hd, nk, rep are x, y, z.
    for (uint i = tid; i < block; i += tcount) {
        const uint g = base + i;
        const uint gl = g - row_base;
        uint src = g;
        if (perm.w != 0u) {
            const uint hd = perm.x, nk = perm.y, rep = perm.z;
            const uint d = gl % hd;
            const uint t = gl / hd;
            const uint r = t % rep;
            const uint k = t / rep;
            src = row_base + d + hd * (k + nk * r);
        }
        float v = x[src];
        if (has_signs != 0u) {
            v *= signs[gl];
        }
        tile[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 1u; h < block; h <<= 1u) {
        // Each thread owns `block / 2 / tcount` butterflies of this
        // stage, indexed so the pair (j, j + h) is never split across
        // two threads.
        for (uint b = tid; b < (block >> 1u); b += tcount) {
            const uint lo = ((b / h) * (h << 1u)) + (b % h);
            const uint hi = lo + h;
            const float a = tile[lo];
            const float c = tile[hi];
            tile[lo] = a + c;
            tile[hi] = a - c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = 1.0f / sqrt(float(block));
    for (uint i = tid; i < block; i += tcount) {
        x[base + i] = tile[i] * scale;
    }
}
"#;

/// Encodes the rotation of `x_buf` in place. `signs_buf` must be the
/// plan's signs when it has any; the caller owns both buffers for the
/// life of the command buffer.
pub fn encode_fold(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    width: usize,
    plan: &FoldPlan<'_>,
    signs_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<(), MetalError> {
    encode_fold_rows(encoder, device, x_buf, width, 1, plan, signs_buf)
}

/// [`encode_fold`] over `rows` consecutive activations of `row_width`,
/// which is what a prefill batch is. The per-row rotation is the same
/// one a single token takes, so the two share this body rather than a
/// batched copy of it.
pub fn encode_fold_rows(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    row_width: usize,
    rows: usize,
    plan: &FoldPlan<'_>,
    signs_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<(), MetalError> {
    plan.check(row_width)?;
    if rows == 0 {
        return Ok(());
    }
    let width = row_width * rows;
    if plan.signs.is_some() && signs_buf.is_none() {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, HADAMARD_KERNEL_SRC, "hadamard_fold")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
        // Metal requires a bound buffer even for the unused argument;
        // `x_buf` is read-only there and never indexed when
        // `has_signs` is zero.
        encoder.setBuffer_offset_atIndex(Some(signs_buf.unwrap_or(x_buf)), 0, 1);
        let mut block = plan.block as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut block as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
        let mut has_signs: u32 = u32::from(plan.signs.is_some());
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut has_signs as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut perm: [u32; 4] = match plan.perm {
            Some((hd, nk, rep)) => [hd as u32, nk as u32, rep as u32, 1],
            None => [0, 0, 0, 0],
        };
        encoder.setBytes_length_atIndex(NonNull::new(perm.as_mut_ptr() as *mut _).unwrap(), 16, 4);
        let mut row_w = row_width as u32;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut row_w as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
    }
    let groups = width / plan.block;
    crate::dispatch::dispatch_counted(
        encoder,
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS.min(plan.block),
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
