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

/// The widest ROW a permuted rotation serves: one threadgroup holds the
/// whole row, `MAX_PERM_ROW` floats of threadgroup memory, 32 KiB.
/// Bonsai's `ssm_out` input is 6144.
pub const MAX_PERM_ROW: usize = 8192;

/// `perm -> signs -> FWHT`, in place.
///
/// The permutation is a GATHER at load: element `g` of the output takes
/// `d + hd * (k + nk * r)` where `d = g % hd`, `r = (g / hd) % rep` and
/// `k = (g / hd) / rep`, which is `tiled_to_grouped` read backwards.
///
/// Without a permutation a block only ever reads itself, so one
/// threadgroup per BLOCK rotates in place (`hadamard_fold`). With one
/// it reads across the whole row, and a threadgroup per block would
/// gather from blocks other threadgroups had already overwritten: a
/// race that changed Bonsai's logits from run to run (#540). So a
/// permuted plan takes `hadamard_fold_row`, one threadgroup per ROW,
/// which loads the whole row before it writes any of it.
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

// The same three steps over one whole ROW per threadgroup, for a plan
// with a permutation: every gather completes before any write, so the
// rotation is in place without a race. The butterflies of every block
// in the row run in each stage together.
kernel void hadamard_fold_row(
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
    threadgroup float tile[8192];
    const uint row_base = tgid * row_width;
    const uint hd = perm.x, nk = perm.y, rep = perm.z;
    for (uint gl = tid; gl < row_width; gl += tcount) {
        const uint d = gl % hd;
        const uint t = gl / hd;
        const uint r = t % rep;
        const uint k = t / rep;
        float v = x[row_base + d + hd * (k + nk * r)];
        if (has_signs != 0u) {
            v *= signs[gl];
        }
        tile[gl] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_block = block >> 1u;
    for (uint h = 1u; h < block; h <<= 1u) {
        for (uint b = tid; b < (row_width >> 1u); b += tcount) {
            const uint blk = b / half_block;
            const uint bb = b % half_block;
            const uint lo = blk * block + ((bb / h) * (h << 1u)) + (bb % h);
            const uint hi = lo + h;
            const float a = tile[lo];
            const float c = tile[hi];
            tile[lo] = a + c;
            tile[hi] = a - c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = 1.0f / sqrt(float(block));
    for (uint i = tid; i < row_width; i += tcount) {
        x[row_base + i] = tile[i] * scale;
    }
}

// `hadamard_fold` for a block of B = 128 to 1024, the butterflies in
// registers: a thread holds four consecutive values for the stages
// below 128 (two in the thread, five across the simdgroup by
// `simd_shuffle_xor`), then a thread holds each of 128 columns'
// `B / 128` values 128 apart for the stages above, through ONE
// threadgroup barrier. Every butterfly is the tile kernel's -- the same
// pair, the same stage order, `lo + hi` and `lo - hi` -- so the result
// is the same to the bit; what goes is nine of its ten barriers and
// most of its threadgroup traffic. At Bonsai's block of 1024 that
// kernel ran a 512 x 17408 fold at 200 GB/s.
template<ushort B>
kernel void hadamard_fold_reg(
    device float* x [[buffer(0)]],
    device const float* signs [[buffer(1)]],
    constant uint& block [[buffer(2)]],
    constant uint& has_signs [[buffer(3)]],
    constant uint4& perm [[buffer(4)]],
    constant uint& row_width [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tile[B];
    const uint base = tgid * B;
    const uint row_base = (base / row_width) * row_width;
    float4 v = ((device const float4*)(x + base))[tid];
    if (has_signs != 0u) {
        v *= ((device const float4*)(signs + (base - row_base)))[tid];
    }
    // h = 1, then h = 2, inside the thread.
    const float4 a = float4(v.x + v.y, v.x - v.y, v.z + v.w, v.z - v.w);
    v = float4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    // h = 4 .. 64: the partner four values are lane `lane ^ m`'s.
    for (ushort m = 1; m < 32; m <<= 1) {
        const float4 p = simd_shuffle_xor(v, m);
        v = (lane & m) ? (p - v) : (v + p);
    }
    const float scale = 1.0f / sqrt(float(B));
    if (B == 128) {
        ((device float4*)(x + base))[tid] = v * scale;
        return;
    }
    ((threadgroup float4*)tile)[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // h = 128 .. B / 2: values 128 apart, in one thread. A block of
    // 256 has 64 threads, so each takes two of the 128 columns.
    constexpr ushort N = B / 128;
    constexpr ushort T = B / 4;
    for (ushort l = tid; l < 128; l += T) {
        float r[N];
        for (ushort i = 0; i < N; ++i) {
            r[i] = tile[l + 128 * i];
        }
        for (ushort s = 1; s < N; s <<= 1) {
            for (ushort i = 0; i < N; ++i) {
                if ((i & s) == 0) {
                    const float lo = r[i];
                    const float hi = r[i + s];
                    r[i] = lo + hi;
                    r[i + s] = lo - hi;
                }
            }
        }
        for (ushort i = 0; i < N; ++i) {
            x[base + l + 128 * i] = r[i] * scale;
        }
    }
}

#define HADAMARD_FOLD_REG(B) \
template [[host_name("hadamard_fold_reg_" #B)]] kernel void hadamard_fold_reg<B>( \
    device float*, device const float*, constant uint&, constant uint&, \
    constant uint4&, constant uint&, uint, ushort, ushort);
HADAMARD_FOLD_REG(128)
HADAMARD_FOLD_REG(256)
HADAMARD_FOLD_REG(512)
HADAMARD_FOLD_REG(1024)
"#;

/// The register kernel for a block it is instantiated for.
fn reg_entry(block: usize) -> Option<&'static str> {
    Some(match block {
        128 => "hadamard_fold_reg_128",
        256 => "hadamard_fold_reg_256",
        512 => "hadamard_fold_reg_512",
        1024 => "hadamard_fold_reg_1024",
        _ => return None,
    })
}

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
    encode_fold_rows_on(
        encoder, device, x_buf, row_width, rows, plan, signs_buf, true,
    )
}

/// [`encode_fold_rows`], with the register kernel allowed (`reg`) or
/// not: the tile kernel stays the reference the register one is held
/// to bit for bit, and the kernel for blocks it is not instantiated for.
#[allow(clippy::too_many_arguments)]
fn encode_fold_rows_on(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    row_width: usize,
    rows: usize,
    plan: &FoldPlan<'_>,
    signs_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    reg: bool,
) -> Result<(), MetalError> {
    plan.check(row_width)?;
    if rows == 0 {
        return Ok(());
    }
    let width = row_width * rows;
    if plan.signs.is_some() && signs_buf.is_none() {
        return Err(MetalError::CommandFailed);
    }
    // A permuted row is held whole by one threadgroup; wider than its
    // scratch is refused rather than rotated in racing pieces.
    let per_row = plan.perm.is_some();
    if per_row && row_width > MAX_PERM_ROW {
        return Err(MetalError::CommandFailed);
    }
    let reg = if per_row || !reg {
        None
    } else {
        reg_entry(plan.block)
    };
    let pipe = ensure_pipeline(
        device,
        HADAMARD_KERNEL_SRC,
        match (per_row, reg) {
            (true, _) => "hadamard_fold_row",
            (false, Some(entry)) => entry,
            (false, None) => "hadamard_fold",
        },
    )?;
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
    let (groups, threads) = if per_row {
        use objc2_metal::MTLComputePipelineState;
        (rows, pipe.0.maxTotalThreadsPerThreadgroup().clamp(1, 1024))
    } else if reg.is_some() {
        (width / plan.block, plan.block / 4)
    } else {
        (width / plan.block, THREADS.min(plan.block))
    };
    crate::dispatch::dispatch_counted(
        encoder,
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLResourceOptions};

    /// `perm -> signs -> FWHT` per row on the host: the definition
    /// `frink_core::weight_matrix::hadamard` states, written out again
    /// here because frink-metal does not depend on frink-core.
    fn host_rows(x: &[f32], width: usize, plan: &FoldPlan<'_>) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        for (row, dst) in x.chunks(width).zip(out.chunks_mut(width)) {
            for (g, d) in dst.iter_mut().enumerate() {
                let src = match plan.perm {
                    Some((hd, nk, rep)) => {
                        let t = g / hd;
                        g % hd + hd * (t / rep + nk * (t % rep))
                    }
                    None => g,
                };
                *d = row[src] * plan.signs.map_or(1.0, |s| s[g]);
            }
            for blk in dst.chunks_mut(plan.block) {
                let mut h = 1;
                while h < plan.block {
                    for lo in 0..plan.block {
                        if lo & h == 0 {
                            let (a, c) = (blk[lo], blk[lo + h]);
                            blk[lo] = a + c;
                            blk[lo + h] = a - c;
                        }
                    }
                    h <<= 1;
                }
                let scale = 1.0 / (plan.block as f32).sqrt();
                blk.iter_mut().for_each(|v| *v *= scale);
            }
        }
        out
    }

    /// A batch of PERMUTED rows rotates in place to the host's answer,
    /// at Bonsai's `ssm_out` shape (six blocks a row), repeated.
    ///
    /// What this does NOT catch is the race `hadamard_fold_row` exists
    /// for: the per-block kernel passes it too, because a gather from a
    /// block another core has just rewritten mostly still reads the old
    /// bytes. The race showed on the real checkpoint instead, as a
    /// Bonsai parity KL that moved from run to run (#540).
    /// The register kernel IS the tile kernel: every block it is
    /// instantiated for, signed and not, several rows of several
    /// blocks, the same bits out, and both within rounding of the host
    /// definition.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn the_register_fold_is_the_tile_fold_to_the_bit() {
        let shared = crate::gpu::shared_metal().expect("metal");
        let device = &shared.device;
        let new_buf = |data: &[f32]| unsafe {
            device
                .newBufferWithBytes_length_options(
                    NonNull::new(data.as_ptr() as *mut _).unwrap(),
                    std::mem::size_of_val(data),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("buffer")
        };
        for block in [128usize, 256, 512, 1024] {
            for signed in [false, true] {
                let (rows, width) = (5usize, 3 * block);
                let signs: Vec<f32> = (0..width)
                    .map(|i| if (i * 7) % 5 < 2 { -1.0 } else { 1.0 })
                    .collect();
                let plan = FoldPlan {
                    block,
                    signs: signed.then_some(&signs[..]),
                    perm: None,
                };
                let x: Vec<f32> = (0..rows * width)
                    .map(|i| ((i as f32) * 0.0137).sin() + (i % 97) as f32 * 0.01)
                    .collect();
                let signs_buf = new_buf(&signs);
                let run = |reg: bool| {
                    let x_buf = new_buf(&x);
                    let cmd = shared.queue.commandBuffer().expect("cmd");
                    let enc = cmd.computeCommandEncoder().expect("enc");
                    encode_fold_rows_on(
                        &enc,
                        device,
                        &x_buf,
                        width,
                        rows,
                        &plan,
                        signed.then_some(&*signs_buf),
                        reg,
                    )
                    .expect("this block is servable");
                    enc.endEncoding();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    // SAFETY: shared storage of `rows * width` floats
                    // the waited-for kernel wrote.
                    unsafe {
                        std::slice::from_raw_parts(
                            x_buf.contents().as_ptr() as *const f32,
                            rows * width,
                        )
                        .to_vec()
                    }
                };
                let (tile, reg) = (run(false), run(true));
                let want = host_rows(&x, width, &plan);
                for (i, ((t, r), w)) in tile.iter().zip(&reg).zip(&want).enumerate() {
                    assert_eq!(
                        t.to_bits(),
                        r.to_bits(),
                        "block {block} signed {signed} element {i}: tile {t} register {r}"
                    );
                    assert!((r - w).abs() <= 1e-4 * (1.0 + w.abs()), "element {i}");
                }
            }
        }
    }

    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn a_permuted_batch_rotates_in_place_without_a_race() {
        let (rows, width, block) = (48usize, 6144usize, 1024usize);
        let signs: Vec<f32> = (0..width)
            .map(|i| if (i * 7) % 5 < 2 { -1.0 } else { 1.0 })
            .collect();
        let plan = FoldPlan {
            block,
            signs: Some(&signs),
            perm: Some((128, 16, 3)),
        };
        let x: Vec<f32> = (0..rows * width)
            .map(|i| ((i as f32) * 0.0137).sin() + (i % 97) as f32 * 0.01)
            .collect();
        let want = host_rows(&x, width, &plan);
        let shared = crate::gpu::shared_metal().expect("metal");
        let device = &shared.device;
        let new_buf = |data: &[f32]| unsafe {
            device
                .newBufferWithBytes_length_options(
                    NonNull::new(data.as_ptr() as *mut _).unwrap(),
                    std::mem::size_of_val(data),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("buffer")
        };
        let signs_buf = new_buf(&signs);
        for rep in 0..8 {
            let x_buf = new_buf(&x);
            let cmd = shared.queue.commandBuffer().expect("cmd");
            let enc = cmd.computeCommandEncoder().expect("enc");
            encode_fold_rows(&enc, device, &x_buf, width, rows, &plan, Some(&signs_buf))
                .expect("a 6144-wide permuted row is servable");
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            // SAFETY: shared storage of `rows * width` floats the
            // waited-for kernel wrote.
            let got = unsafe {
                std::slice::from_raw_parts(x_buf.contents().as_ptr() as *const f32, rows * width)
            };
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                    "rep {rep} element {i}: device {g} host {w}"
                );
            }
        }
    }
}
