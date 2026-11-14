//! The Metal 4 tensor-API arm of the prefill GEMM (llama
//! `kernel_mul_mm` under `GGML_METAL_HAS_TENSOR`, `kernels/mul_mm.metal`).
//!
//! On an M5 (and A19) GPU the `mpp::tensor_ops::matmul2d` op runs on the
//! per-core matrix units; the simdgroup-matrix GEMM in
//! `gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC` does not reach them, and reads
//! about a third of llama.cpp's prefill rate there. llama turns the
//! tensor API on for M5, M6, A19 and A20 only (`ggml-metal-device.m`:
//! older chips measured no gain or a loss), after compiling a probe
//! kernel; this module makes the same decision once per process.
//!
//! The arm keeps the GEMM's entry points and buffer layout. Only the
//! tile changes: 64 weight rows by 128 tokens, the weights dequantized
//! to `half` in threadgroup memory 32 K-values at a time, the
//! activations read from device memory as they are. Host dispatch asks
//! [`mm_tile`] for the grid and [`mm_smem`] for the allocation.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion,
    MTLLibrary,
};
use std::sync::OnceLock;

/// The `#define` that selects the tensor arm. A library whose source
/// carries it is compiled at Metal language 4.0 ([`compile_options`]).
pub(crate) const TENSOR_DEFINE: &str = "#define FRINK_METAL_HAS_TENSOR 1\n";

/// Weight rows per tensor tile (llama `NRA = 16 * 2 * 2`).
pub(crate) const NRA: usize = 64;
/// Tokens per tensor tile (llama `NRB = 16 * 4 * 2`).
pub(crate) const NRB: usize = 128;
/// Threadgroup bytes: the `half` A tile, `NRA x 32` (llama
/// `res.smem = NRA * N_MM_NK_TOTAL * sizeof(half)`).
pub(crate) const SMEM: usize = NRA * 32 * 2;

/// Whether this process's GPU runs the tensor arm. `FRINK_METAL_TENSOR=0`
/// turns it off, `=1` turns it on for a chip llama leaves it off on
/// (llama's `GGML_METAL_TENSOR_DISABLE` / `_ENABLE`); either way it is
/// on only if the device is Metal 4 and the probe kernel compiles.
pub(crate) fn has_tensor() -> bool {
    static HAS: OnceLock<bool> = OnceLock::new();
    *HAS.get_or_init(|| {
        let force = std::env::var("FRINK_METAL_TENSOR").ok();
        if force.as_deref() == Some("0") {
            return false;
        }
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return false;
        };
        if !device.supportsFamily(MTLGPUFamily::Metal4) {
            return false;
        }
        let name = device.name().to_string();
        let new_chip = ["M5", "M6", "A19", "A20"].iter().any(|c| name.contains(c));
        if !new_chip && force.as_deref() != Some("1") {
            return false;
        }
        probe_compiles(&device)
    })
}

/// Tokens and rows per GEMM tile on this device: `(batch, rows)`.
pub(crate) fn mm_tile() -> (usize, usize) {
    if has_tensor() {
        (NRB, NRA)
    } else {
        (32, 64)
    }
}

/// Compile options for a library: Metal 4.0 when the source selects the
/// tensor arm (the tensor headers exist only at that language version),
/// the default otherwise.
pub(crate) fn compile_options(src: &str) -> Option<Retained<MTLCompileOptions>> {
    if !src.contains(TENSOR_DEFINE) {
        return None;
    }
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    Some(options)
}

/// The prefix spliced at the top of the GEMM translation unit: the
/// define and the tensor headers when [`has_tensor`], nothing otherwise.
pub(crate) fn gemm_prelude() -> String {
    if has_tensor() {
        format!("{TENSOR_DEFINE}{HEADERS}")
    } else {
        String::new()
    }
}

pub(crate) const HEADERS: &str = "#include <metal_tensor>\n\
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";

/// llama's probe (`ggml-metal-device.m`), so a toolchain without the
/// tensor headers falls back instead of failing the first prefill.
fn probe_compiles(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> bool {
    let src = format!("{TENSOR_DEFINE}#include <metal_stdlib>\n{HEADERS}{PROBE}");
    let Some(options) = compile_options(&src) else {
        return false;
    };
    let Ok(lib) =
        device.newLibraryWithSource_options_error(&NSString::from_str(&src), Some(&options))
    else {
        return false;
    };
    let Some(f) = lib.newFunctionWithName(&NSString::from_str("frink_tensor_probe")) else {
        return false;
    };
    device.newComputePipelineStateWithFunction_error(&f).is_ok()
}

const PROBE: &str = r#"
using namespace metal;
using namespace mpp::tensor_ops;
kernel void frink_tensor_probe(
    tensor<device half, dextents<int32_t, 2>> A [[buffer(0)]],
    tensor<device half, dextents<int32_t, 2>> B [[buffer(1)]],
    device float * C [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
    auto tA = A.slice(0, (int)tgid.y);
    auto tB = B.slice((int)tgid.x, 0);
    matmul2d<matmul2d_descriptor(16, 16, dynamic_extent), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    auto sA = tA.slice(0, 0);
    auto sB = tB.slice(0, 0);
    mm.run(sB, sA, cT);
    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16));
    cT.store(tC);
}
"#;

/// The tensor GEMM body, spliced into the GEMM translation unit in
/// place of the simdgroup `mul_mm_sg_impl` / `_f16` when the define is
/// set. `T1` is the activation type (`float`, or `half` for the `_f16`
/// entries). `DQ::get(block, il, reg)` yields 16 consecutive weights,
/// sub-block `il` of a `DQ::NL * 16`-value block, as in the simdgroup
/// arm. Layout as there: A is `[n_rows, K]` blocks of `row_bytes`, B is
/// `[batch, K]`, C is `[batch, n_rows]`.
pub(crate) const TENSOR_IMPL_MSL: &str = r#"
#ifdef FRINK_METAL_HAS_TENSOR
template<typename DQ, typename T1>
static inline void mul_mm_tensor_impl(
    device const uchar* src0,
    device const T1* src1,
    device float* dst,
    uint n_rows,
    uint n_cols,
    uint batch,
    uint row_bytes,
    threadgroup char* shmem,
    uint3 tgpig,
    ushort tiitg
) {
    constexpr int NRA = 64;
    constexpr int NRB = 128;
    constexpr int NK = 32;
    constexpr short NL = DQ::NL;
    constexpr int A_WORK = NRA * (NK / 16);
    constexpr int NUM_THREADS = 128;

    const int K = int(n_cols);
    const int M = int(n_rows);
    const int N = int(batch);
    const int ra = int(tgpig.y) * NRA;
    const int rb = int(tgpig.x) * NRB;

    threadgroup half* sa = (threadgroup half*)shmem;
    device T1* ptrB = (device T1*)src1;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NRA));
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, K}));
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < A_WORK; work += NUM_THREADS) {
            const int row = work / (NK / 16);
            const short k_base = short(work % (NK / 16)) * 16;
            const int k_pos = loop_k + k_base;
            threadgroup half* out = sa + row * NK + k_base;
            if (ra + row < M && k_pos < K) {
                float4x4 t;
                DQ::get(src0 + (size_t)row_bytes * (ra + row)
                            + (size_t)(k_pos / (16 * NL)) * DQ::BLOCK_BYTES,
                        short((k_pos / 16) % NL), t);
#pragma clang loop unroll(full)
                for (short i = 0; i < 16; i++) {
                    out[i] = (k_pos + i < K) ? half(t[i / 4][i % 4]) : half(0);
                }
            } else {
#pragma clang loop unroll(full)
                for (short i = 0; i < 16; i++) {
                    out[i] = half(0);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int k_ext = min(NK, K - loop_k);
        auto tAv = tensor(sa, dextents<int32_t, 2>(k_ext, NRA), array<int, 2>({1, NK}));
        auto tBv = tensor(ptrB + loop_k + (size_t)rb * K,
                          dextents<int32_t, 2>(k_ext, N - rb), array<int, 2>({1, K}));
        mm.run(tBv, tAv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // No slice: its byte offset is int32 and wraps at 2 GiB (llama #28748).
    const int tile_m = min(NRA, M - ra);
    const int tile_n = min(NRB, N - rb);
    device float* dst_tile = dst + (size_t)rb * M + ra;
    auto tD = tensor(dst_tile, dextents<int32_t, 2>(tile_m, tile_n), array<int, 2>({1, M}));
    cT.store(tD);
}

template<typename DQ, bool BC_OUT>
static inline void mul_mm_sg_impl(
    device const uchar* src0, device const float* src1, device float* dst,
    uint n_rows, uint n_cols, uint batch, uint row_bytes,
    threadgroup char* shmem, uint3 tgpig, ushort tiitg, ushort sgitg
) {
    mul_mm_tensor_impl<DQ, float>(src0, src1, dst, n_rows, n_cols, batch,
                                  row_bytes, shmem, tgpig, tiitg);
}

template<typename DQ, bool BC_OUT>
static inline void mul_mm_sg_impl_f16(
    device const uchar* src0, device const half* src1, device float* dst,
    uint n_rows, uint n_cols, uint batch, uint row_bytes,
    threadgroup char* shmem, uint3 tgpig, ushort tiitg, ushort sgitg
) {
    mul_mm_tensor_impl<DQ, half>(src0, src1, dst, n_rows, n_cols, batch,
                                 row_bytes, shmem, tgpig, tiitg);
}
// Eight activations as half, from either activation type.
static inline half2x4 mm_load_b8(device const float* y) {
    return half2x4(*(device const float2x4*)y);
}
static inline half2x4 mm_load_b8(device const half* y) {
    return *(device const half2x4*)y;
}

// llama `kernel_mul_mm_id`, tensor arm: the simdgroup tile (64 rows x 32
// gathered tokens) with both operands staged row-major in threadgroup
// memory and the product taken by two 16-token `matmul2d` halves, the
// upper one skipped when the expert did not fill it. The result goes
// through the same `[token][row]` f32 staging the simdgroup arm uses,
// then is scattered by `ids`.
template<typename DQ, typename T1>
static inline void mul_mm_id_tensor_impl(
    device const uchar* src0,
    device const T1* src1,
    device float* dst,
    device const int* ids,
    device const uint* tpe,
    uint expert_stride,
    uint n_rows,
    uint n_cols,
    uint top_k,
    uint ids_stride,
    uint row_bytes,
    uint src1_per_slot,
    threadgroup char* shmem,
    uint3 tgpig,
    ushort tiitg,
    ushort tiisg,
    ushort sgitg
) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    threadgroup float* sc = (threadgroup float*)(shmem);

    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NR1H = NR1 / 2;
    constexpr short NK  = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short NL  = DQ::NL;

    uint im;
    int r1;
    int neh1;
    if (!mm_id_tile(tpe, tgpig.x, im, r1, neh1)) {
        return;
    }
    const int r0 = int(tgpig.y) * NR0;

    const short nr0 = (int(n_rows) - r0 < NR0) ? short(int(n_rows) - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? short(neh1 - r1) : NR1;

    const short lr0 = (short(tiitg) / NL0) < nr0 ? (short(tiitg) / NL0) : nr0 - 1;
    const short lr1 = (short(tiitg) / NL1) < nr1 ? (short(tiitg) / NL1) : nr1 - 1;

    const short il0 = short(tiitg) % NL0;
    short il = il0;

    const int id = ids[im * ids_stride + r1 + lr1];
    const int src1_row = src1_per_slot != 0u ? id : (id / int(top_k));

    device const uchar* x = src0 + (size_t)im * expert_stride + (size_t)row_bytes * (r0 + lr0);

    const short iy = 8 * (short(tiitg) % NL1);
    device const T1* y = src1 + (size_t)src1_row * n_cols + iy;

    auto tA  = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NK, NR1H));
    auto tB1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb + NR1H * NK, dextents<int32_t, 2>(NK, NR1H));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1H, NR0, NK, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT0 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto cT1 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();

    const bool has_hi = nr1 > NR1H;

    for (uint loop_k = 0; loop_k < n_cols; loop_k += NK) {
        float4x4 temp_a;
        DQ::get(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (short(tiitg) / NL0) / 8;
            const short lx = i % 8;
            const short ly = (short(tiitg) / NL0) % 8;
            *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = half(temp_a[i / 4][i % 4]);
        }

        {
            const short sx = short(tiitg) % NL1;
            const short sy = (short(tiitg) / NL1) / 8;
            const short ly = (short(tiitg) / NL1) % 8;
            *(threadgroup half2x4*)(sb + NK * (8 * sy + ly) + 8 * sx) = mm_load_b8(y);
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        if (il < 2) {
            x += DQ::BLOCK_BYTES;
        }
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA  = tA.slice(0, 0);
        auto sB0 = tB0.slice(0, 0);
        mm.run(sB0, sA, cT0);
        if (has_hi) {
            auto sB1 = tB1.slice(0, 0);
            mm.run(sB1, sA, cT1);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1H));
    cT0.store(tC0);
    if (has_hi) {
        auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + NR1H * NR0, dextents<int32_t, 2>(NR0, NR1H));
        cT1.store(tC1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int sid = ids[im * ids_stride + r1 + j];
        device float* D = dst + (size_t)sid * n_rows + r0;
        device float4* D4 = (device float4*)D;
        threadgroup float* C = sc + j * NR0;
        threadgroup float4* C4 = (threadgroup float4*)C;
        int i = int(tiisg);
        for (; i < nr0 / 4; i += 32) {
            D4[i] = C4[i];
        }
        i = (4 * (nr0 / 4)) + int(tiisg);
        for (; i < nr0; i += 32) {
            D[i] = C[i];
        }
    }
}

#define FRINK_MM_ID_ARGS src0, src1, dst, ids, tpe, expert_stride, n_rows, n_cols, top_k, \
    ids_stride, row_bytes, src1_per_slot, shmem, tgpig, tiitg, tiisg, sgitg
#define FRINK_MM_ID_PARAMS(T1) device const uchar* src0, device const T1* src1, \
    device float* dst, device const int* ids, device const uint* tpe, uint expert_stride, \
    uint n_rows, uint n_cols, uint top_k, uint ids_stride, uint row_bytes, \
    uint src1_per_slot, threadgroup char* shmem, uint3 tgpig, ushort tiitg, \
    ushort tiisg, ushort sgitg

template<typename DQ>
static inline void mul_mm_id_impl(FRINK_MM_ID_PARAMS(float)) {
    mul_mm_id_tensor_impl<DQ, float>(FRINK_MM_ID_ARGS);
}

template<typename DQ>
static inline void mul_mm_id_impl_f16(FRINK_MM_ID_PARAMS(half)) {
    mul_mm_id_tensor_impl<DQ, half>(FRINK_MM_ID_ARGS);
}
#endif
"#;

#[cfg(test)]
mod tests {
    use objc2_metal::MTLDevice;

    /// The probe failing to compile turns the tensor arm off silently,
    /// and prefill drops to a third of llama's rate with no error. On
    /// a chip llama enables it for, it must come out on.
    #[test]
    #[ignore = "needs Metal"]
    fn the_tensor_arm_is_on_where_llama_turns_it_on() {
        let Some(d) = objc2_metal::MTLCreateSystemDefaultDevice() else {
            return;
        };
        let name = d.name().to_string();
        let new_chip = ["M5", "M6", "A19", "A20"].iter().any(|c| name.contains(c));
        if new_chip && std::env::var_os("FRINK_METAL_TENSOR").is_none() {
            assert!(
                super::has_tensor(),
                "{name}: the tensor probe did not compile"
            );
            assert_eq!(super::mm_tile(), (super::NRB, super::NRA));
        }
    }
}
