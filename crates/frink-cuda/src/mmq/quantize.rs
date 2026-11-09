//! The activations of an int8 GEMM, quantized once per GEMM input as
//! llama.cpp's `quantize_mmq_q8_1` quantizes them (`quantize.cu` at the
//! pin): each run of 32 values gets a scale `d = amax / 127` and the
//! values `round(x * 127 / amax)` as int8, and runs of 128 values are
//! stored as 144-byte blocks, K-block-major (`block (kb, col)` at
//! `kb * ncols_pad + col`), so a GEMM tile of columns is one contiguous
//! copy. The 16-byte header holds four f32 scales (`D4`, for Q8_0,
//! MXFP4, Q6_K) or four `half2(d, sum)` (`DS4`, for Q4_K and Q5_K, the
//! sum being of the values before quantization, which the min term
//! takes).

/// Values in one block: four runs of 32.
pub const QK: usize = 128;
/// Bytes in one block: the 16-byte header and the 128 int8.
pub const BLOCK_BYTES: usize = 144;
/// Values per thread, one `float4`.
const PER_THREAD: usize = 4;
/// Threads per launch block: 512 values of one column.
pub const THREADS: usize = 128;

/// The header layout a weight kind's GEMM reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// One f32 scale per 32 values.
    D4,
    /// One `half2(d, sum)` per 32 values.
    Ds4,
}

/// The kernel. `FX_DS4` picks the header layout.
pub const QUANTIZE_SRC: &str = r#"
#ifndef FX_DS4
#define FX_DS4 0
#endif

__device__ __forceinline__ unsigned int frink_q8_pack2h(float lo, float hi) {
    unsigned int r;
    asm("{ .reg .f16 l, h; cvt.rn.f16.f32 l, %1; cvt.rn.f16.f32 h, %2; mov.b32 %0, {l, h}; }"
        : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

extern "C" __global__ void FX_FN_NAME(
    const float* __restrict__ x,
    const int* __restrict__ gather,
    unsigned char* __restrict__ y,
    int k,
    int kpad,
    int ncols,
    int ncols_pad
) {
    const int col = blockIdx.x;
    const int i0 = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (i0 >= kpad) return;
    // A padding column, or the tail past `k`, quantizes as zeros, so the
    // buffer needs no clearing.
    const bool live = col < ncols;
    const int row = !live ? 0 : (gather != nullptr ? gather[col] : col);
    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (live && i0 < k) xi = *(const float4*)(x + (size_t)row * k + i0);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
    for (int off = 4; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
#if FX_DS4
    float sum = xi.x + xi.y + xi.z + xi.w;
    for (int off = 4; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
#endif
    // A run of zeros: zero values and a zero scale (llama.cpp reaches the
    // same through 0 * inf converting to 0).
    const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    char4 q;
    q.x = (signed char)roundf(xi.x * d_inv);
    q.y = (signed char)roundf(xi.y * d_inv);
    q.z = (signed char)roundf(xi.z * d_inv);
    q.w = (signed char)roundf(xi.w * d_inv);
    const float d = amax > 0.0f ? 1.0f / d_inv : 0.0f;
    const int kb = i0 / 128;
    const int iqs = i0 % 128;
    unsigned char* blk = y + ((size_t)kb * ncols_pad + col) * 144;
    *(char4*)(blk + 16 + iqs) = q;
    if (iqs % 32 == 0) {
#if FX_DS4
        *(unsigned int*)(blk + 4 * (iqs / 32)) = frink_q8_pack2h(d, sum);
#else
        *(float*)(blk + 4 * (iqs / 32)) = d;
#endif
    }
}
"#;

/// The translation unit and its NVRTC names for `layout`.
pub fn source(layout: Layout) -> (String, &'static str, &'static str) {
    let (module, name) = names(layout);
    let ds4 = u8::from(layout == Layout::Ds4);
    (
        format!("#define FX_DS4 {ds4}\n#define FX_FN_NAME {name}\n{QUANTIZE_SRC}"),
        module,
        name,
    )
}

/// The NVRTC module and function names for `layout`.
pub fn names(layout: Layout) -> (&'static str, &'static str) {
    match layout {
        Layout::D4 => ("frink_q8_1_mmq_d4", "frink_quantize_q8_1_mmq_d4"),
        Layout::Ds4 => ("frink_q8_1_mmq_ds4", "frink_quantize_q8_1_mmq_ds4"),
    }
}

/// `k` rounded up to whole blocks.
pub fn kpad(k: usize) -> usize {
    k.div_ceil(QK) * QK
}

/// The buffer's length in bytes for `k` values of `ncols_pad` columns.
pub fn buffer_bytes(k: usize, ncols_pad: usize) -> usize {
    kpad(k) / QK * ncols_pad * BLOCK_BYTES
}

/// `(grid, block)`: a grid column per output column, a row of blocks per
/// 512 values.
pub fn geometry(k: usize, ncols_pad: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let per_block = THREADS * PER_THREAD;
    (
        (ncols_pad as u32, kpad(k).div_ceil(per_block) as u32, 1),
        (THREADS as u32, 1, 1),
    )
}

#[cfg(feature = "cuda")]
mod launch {
    use super::*;
    use crate::gpu::CudaError;
    use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
    use std::sync::Arc;

    /// Quantizes `ncols` rows of `k` f32 (row `gather[c]` for column `c`
    /// when a gather is given) into a buffer of `ncols_pad` columns.
    pub(crate) fn enqueue_quantize(
        dev: &Arc<CudaDevice>,
        layout: Layout,
        x: &CudaSlice<f32>,
        gather: Option<&CudaSlice<i32>>,
        k: usize,
        ncols: usize,
        ncols_pad: usize,
    ) -> Result<CudaSlice<u8>, CudaError> {
        if !k.is_multiple_of(PER_THREAD) || ncols_pad < ncols {
            return Err(CudaError::Unsupported(format!(
                "q8_1 quantize: {k} values (a multiple of 4) of {ncols} columns in {ncols_pad}"
            )));
        }
        let (src, module, name) = source(layout);
        crate::gpu::ensure_module_loaded_lazy(dev, module, name, || src)?;
        let func = dev.get_func(module, name).ok_or_else(|| {
            CudaError::KernelCompile(format!("function '{name}' not found after load_ptx"))
        })?;
        // SAFETY: uninitialised; every block of every column, padding
        // included, is written by the grid below before anything reads it
        // (`geometry` covers `ncols_pad` columns and `kpad` values).
        let mut y = unsafe { dev.alloc::<u8>(buffer_bytes(k, ncols_pad)) }
            .map_err(|e| CudaError::Launch(format!("q8_1 buffer alloc: {e:?}")))?;
        let (grid, block) = geometry(k, ncols_pad);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };
        let gather = gather.map_or(0u64, |g| *g.device_ptr());
        // SAFETY: (const float*, const int* or null, u8*, int x4) matched
        // positionally; `x` holds the rows the columns name (`ncols` rows,
        // or every gathered row, of `k` each; the caller's shape), its
        // float4 loads are aligned because `k` is a multiple of 4 on a
        // 256-byte-aligned allocation, and every block index is below
        // `kpad / 128 * ncols_pad`, the buffer's length in blocks.
        unsafe {
            func.launch(
                cfg,
                (
                    x,
                    gather,
                    &mut y,
                    k as i32,
                    kpad(k) as i32,
                    ncols as i32,
                    ncols_pad as i32,
                ),
            )
            .map_err(|e| CudaError::Launch(format!("{name}: {e:?}")))?;
        }
        Ok(y)
    }
}

#[cfg(feature = "cuda")]
pub(crate) use launch::enqueue_quantize;

/// The kernel's arithmetic on the host, in its order: the twin the
/// hardware test holds it to bit for bit.
#[cfg(test)]
pub(crate) fn quantize_twin(
    x: &[f32],
    gather: Option<&[i32]>,
    k: usize,
    ncols: usize,
    ncols_pad: usize,
    layout: Layout,
) -> Vec<u8> {
    let mut y = vec![0u8; buffer_bytes(k, ncols_pad)];
    for col in 0..ncols_pad {
        let row = if col < ncols {
            gather.map_or(col, |g| g[col] as usize)
        } else {
            usize::MAX
        };
        for run in 0..kpad(k) / 32 {
            // Eight threads of four values.
            let vals: Vec<[f32; 4]> = (0..8)
                .map(|t| {
                    let i0 = run * 32 + t * 4;
                    if row == usize::MAX || i0 >= k {
                        [0.0; 4]
                    } else {
                        let r = &x[row * k + i0..row * k + i0 + 4];
                        [r[0], r[1], r[2], r[3]]
                    }
                })
                .collect();
            let amax = vals
                .iter()
                .flat_map(|v| v.iter().map(|x| x.abs()))
                .fold(0f32, f32::max);
            // The xor tree over offsets 4, 2, 1, as lane 0 of the eight
            // ends it (every lane ends with the same sum).
            let p: Vec<f32> = vals.iter().map(|v| v[0] + v[1] + v[2] + v[3]).collect();
            let q1: Vec<f32> = (0..8).map(|j| p[j] + p[j ^ 4]).collect();
            let q2: Vec<f32> = (0..8).map(|j| q1[j] + q1[j ^ 2]).collect();
            let sum = q2[0] + q2[1];
            let d_inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            let d = if amax > 0.0 { 1.0 / d_inv } else { 0.0 };
            let kb = run * 32 / QK;
            let blk = (kb * ncols_pad + col) * BLOCK_BYTES;
            let sub = run % 4;
            for (t, v) in vals.iter().enumerate() {
                for (e, &xv) in v.iter().enumerate() {
                    y[blk + 16 + sub * 32 + t * 4 + e] = ((xv * d_inv).round() as i8) as u8;
                }
            }
            let hdr = &mut y[blk + 4 * sub..blk + 4 * sub + 4];
            match layout {
                Layout::D4 => hdr.copy_from_slice(&d.to_le_bytes()),
                Layout::Ds4 => {
                    hdr[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                    hdr[2..].copy_from_slice(&half::f16::from_f32(sum).to_le_bytes());
                }
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The twin's blocks dequantize back to the values within a scale
    /// step, the padding is zeros, and the gather reads its rows: the
    /// twin has to be right, not only consistent with the kernel.
    #[test]
    fn the_twin_quantizes_within_a_step_and_pads_with_zeros() {
        let (k, ncols, ncols_pad) = (200usize, 3usize, 8usize);
        let x: Vec<f32> = (0..4 * k)
            .map(|i| ((i as f32) * 0.37).sin() * 2.0)
            .collect();
        let gather = [3i32, 0, 2];
        for layout in [Layout::D4, Layout::Ds4] {
            let y = quantize_twin(&x, Some(&gather), k, ncols, ncols_pad, layout);
            assert_eq!(y.len(), buffer_bytes(k, ncols_pad));
            for col in 0..ncols_pad {
                for i in 0..kpad(k) {
                    let blk = ((i / QK) * ncols_pad + col) * BLOCK_BYTES;
                    let sub = (i % QK) / 32;
                    let d = match layout {
                        Layout::D4 => f32::from_le_bytes(
                            y[blk + 4 * sub..blk + 4 * sub + 4].try_into().unwrap(),
                        ),
                        Layout::Ds4 => half::f16::from_le_bytes(
                            y[blk + 4 * sub..blk + 4 * sub + 2].try_into().unwrap(),
                        )
                        .to_f32(),
                    };
                    let q = y[blk + 16 + i % QK] as i8;
                    let want = if col < ncols && i < k {
                        x[gather[col] as usize * k + i]
                    } else {
                        0.0
                    };
                    let got = f32::from(q) * d;
                    assert!(
                        (got - want).abs() <= d * 0.51 + want.abs() * 1e-3 + 1e-6,
                        "{layout:?} col {col} i {i}: {got} for {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_geometry_covers_every_block() {
        assert_eq!(kpad(2880), 2944);
        assert_eq!(kpad(4096), 4096);
        assert_eq!(buffer_bytes(4096, 64), 32 * 64 * 144);
        let ((gx, gy, _), (bx, _, _)) = geometry(2880, 64);
        assert_eq!((gx, bx), (64, 128));
        assert!(gy as usize * 512 >= kpad(2880));
    }
}
