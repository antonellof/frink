//! `mul_mm` on the tensor cores: the same per-kind dequantization as
//! [`crate::mul_mm`], a different inner product.
//!
//! The SIMT body in `mul_mm.rs` reached about 7 TFLOPS on an RTX 3090
//! (`q4_k_mul_mm`, 1.9 ms per Llama-3.2-3B projection at pp512), which
//! after the resident prefill stack (#259) was three quarters of the
//! GPU time of a prefill step. llama.cpp's `mmq` runs the same product
//! on the tensor cores as int8 `mma`. This body takes the middle road
//! that reuses every unpack function in the table unchanged: each tile
//! of weights is dequantized into shared memory as f16, the activation
//! tile is converted to f16 on the way in, and the product is
//! `mma.sync.m16n8k16` with f32 accumulation. f16 keeps eleven bits of
//! each operand where llama.cpp's `q8_1` activations keep eight, so
//! the result is closer to the f32 twin than upstream's is to its own.
//!
//! `mma.sync.m16n8k16.f16` needs `sm_80`, so the launch asks the
//! device for its compute capability and a pre-Ampere card keeps the
//! SIMT body; nothing here replaces it.
//!
//! Fragment layouts are the PTX ISA's for `m16n8k16` with `.f16`
//! operands and `.f32` accumulators, written out in `TC_BODY_SRC`
//! beside the loads that honour them: lane `l` has `g = l / 4` and
//! `t = l % 4`; A holds rows `g` and `g + 8` at k `2t, 2t+1` and
//! `2t+8, 2t+9`; B holds token `g` at the same k; C holds rows `g`,
//! `g + 8` at tokens `2t, 2t+1`. Both shared tiles are stored
//! k-contiguous with a row stride of 40 halves, which is what makes
//! the eight rows a fragment load touches land on distinct banks.

use crate::mul_mm::{kernel_src_with_body, MulMmKind};

/// Weight rows per block tile.
pub const TBM: usize = 128;
/// Tokens per block tile.
pub const TBN: usize = 128;
/// K elements per tile step: two `k16` MMA steps.
pub const TBK: usize = 32;
/// Warps along M and N; eight warps, 256 threads.
pub const WM: usize = 4;
pub const WN: usize = 2;
pub const TTHREADS: usize = 32 * WM * WN;
/// Padded shared row stride in halves. `TBK + 8`: 80 bytes, so the
/// eight rows a fragment touches start 20 words apart and cover eight
/// distinct bank groups.
pub const LDS: usize = TBK + 8;

const _: () = assert!(
    TBM / WM == 32 && TBN / WN == 64,
    "the warp tile is 32 rows x 64 tokens"
);
const _: () = assert!(
    TBM * (TBK / crate::mul_mm::SUB) == TTHREADS,
    "one sub-block per thread per step"
);
const _: () = assert!(
    TBN * 2 == TTHREADS,
    "two threads per token row of the B tile"
);
const _: () = assert!(TBK == 32, "the B loader converts sixteen floats per thread");
const _: () = assert!(
    MIN_STAGE_ELEMS.is_multiple_of(TBK),
    "a stage is whole tiles"
);
const _: () = assert!(
    (TBN / WN / 8).is_multiple_of(2),
    "B fragments load two n8 tiles per ldmatrix.x4"
);
const _: () = assert!(
    (LDS * 2).is_multiple_of(16),
    "shared rows stay 16-byte aligned"
);

/// The compute capability the body needs: `mma.sync.m16n8k16` with
/// f16 operands is `sm_80` and up.
pub const MIN_CC_MAJOR: i32 = 8;
/// The NVRTC target for the body.
pub const ARCH: &str = "compute_80";

/// The kernel body. `FX_*` geometry names are `#define`d by
/// [`tc_kernel_src`]; the dequant helpers come from the kind's row
/// exactly as for the SIMT body.
pub const TC_BODY_SRC: &str = r#"
__device__ __forceinline__ unsigned int frink_pack2h(float lo, float hi) {
    unsigned int r;
    asm("{ .reg .f16 l, h; cvt.rn.f16.f32 l, %1; cvt.rn.f16.f32 h, %2; mov.b32 %0, {l, h}; }"
        : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

__device__ __forceinline__ void frink_mma16816(float* c, const unsigned int* a, const unsigned int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One 8x8 b16 matrix load per group of eight lanes, four matrices, out of
// shared memory: the fragment layout `mma.m16n8k16` reads for A, and for
// two n8 fragments of B.
__device__ __forceinline__ void frink_ldsm_x4(unsigned int* r, const unsigned short* p) {
    unsigned int a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

#ifndef FX_GROUPED
#define FX_GROUPED 0
#endif

extern "C" __global__ void FX_FN_NAME(
#if FX_GROUPED
    // One routed expert per blockIdx.z: its weights through `wtab`, its
    // tokens the pairs `group_off[z] .. group_off[z + 1]`, pair p reading
    // activation row `gather[p]` and writing output row p.
    const unsigned long long* __restrict__ wtab,
    const int* __restrict__ group_off,
    const int* __restrict__ gather,
#else
    const unsigned char* __restrict__ src0,
#endif
    const unsigned short* __restrict__ src1,
    float* __restrict__ dst,
    int n_rows,
    int n_cols,
    int batch,
    int row_bytes
) {
#if FX_GROUPED
    const int g_off = group_off[blockIdx.z];
    const int g_cnt = group_off[blockIdx.z + 1] - g_off;
    // A whole block past its group's tokens has nothing to do; the test
    // is the block's, so no barrier below is reached by part of it.
    if ((int)(blockIdx.x * FX_TBN) >= g_cnt) return;
    const unsigned char* src0 = (const unsigned char*)wtab[blockIdx.z];
    batch = g_cnt;
// A null gather is the identity: pair `p` reads activation row `p`.
#define FX_SRC_ROW(col) ((size_t)(gather != nullptr ? gather[g_off + (col)] : g_off + (col)))
#define FX_DST_ROW(col) ((size_t)(g_off + (col)))
#else
#define FX_SRC_ROW(col) ((size_t)(col))
#define FX_DST_ROW(col) ((size_t)(col))
#endif
    // The weight rows' raw bytes for one stage of FX_SE columns, staged
    // by coalesced word loads, then dequantized out of shared memory: a
    // lane of the dequant reads its own row, and reading that row from
    // global memory a byte at a time is what bounded the body before.
    __shared__ __align__(16) unsigned int sraw[FX_TBM * FX_RAW_W];
    __shared__ __align__(16) unsigned short sa[FX_TBM * FX_LDS];
    __shared__ __align__(16) unsigned short sb[FX_TBN * FX_LDS];

    const int r0 = blockIdx.y * FX_TBM;
    const int c0 = blockIdx.x * FX_TBN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp % FX_WM;
    const int wn = warp / FX_WM;
    const size_t total_bytes = (size_t)n_rows * (size_t)row_bytes;

    float acc[FX_MF][FX_NF][4];
#pragma unroll
    for (int mf = 0; mf < FX_MF; mf++) {
#pragma unroll
        for (int nf = 0; nf < FX_NF; nf++) {
            acc[mf][nf][0] = 0.f; acc[mf][nf][1] = 0.f;
            acc[mf][nf][2] = 0.f; acc[mf][nf][3] = 0.f;
        }
    }

    for (int s0 = 0; s0 < n_cols; s0 += FX_SE) {
        const int se = min(FX_SE, n_cols - s0);
        const int sbytes = (se / FX_BLOCK_ELEMS) * FX_BLOCK_BYTES;
        const size_t stage_off = (size_t)(s0 / FX_BLOCK_ELEMS) * (size_t)FX_BLOCK_BYTES;

        // Every lane is past the previous stage's last read of sraw.
        __syncthreads();
        // Out-of-range rows re-read the last valid row (llama's clamp)
        // and are dropped at the store. A word is loaded whole when it
        // lies inside the matrix and byte by byte at its very end.
        for (int idx = tid; idx < FX_TBM * FX_RAW_W; idx += FX_TTHREADS) {
            const int lr = idx / FX_RAW_W;
            const int w = idx - lr * FX_RAW_W;
            int row = r0 + lr;
            if (row >= n_rows) row = n_rows - 1;
            const size_t a = (size_t)row * (size_t)row_bytes + stage_off;
            const size_t wa = (a & ~(size_t)3) + (size_t)w * 4;
            unsigned int v = 0u;
            if (wa < a + (size_t)sbytes) {
                if (wa + 4 <= total_bytes) {
                    v = __ldg((const unsigned int*)(src0 + wa));
                } else {
                    for (int b = 0; b < 4; b++) {
                        if (wa + b < total_bytes) v |= (unsigned int)src0[wa + b] << (8 * b);
                    }
                }
            }
            sraw[lr * FX_RAW_W + w] = v;
        }

        for (int t0 = 0; t0 < se; t0 += FX_TBK) {
            const int k0 = s0 + t0;
            // The raw stage is complete (first tile), and every lane is
            // past the previous tile's fragment loads.
            __syncthreads();

            // A tile: one FX_SUB-element sub-block per thread, dequantized
            // to f32 by the kind's own function out of the staged bytes
            // and packed to f16 pairs.
            {
                const int lr = tid >> 1;
                const int ils = tid & 1;
                int row = r0 + lr;
                if (row >= n_rows) row = n_rows - 1;
                const int o = (int)(((size_t)row * (size_t)row_bytes + stage_off) & 3);
                const unsigned char* rp = (const unsigned char*)&sraw[lr * FX_RAW_W] + o;
                const int sub = (t0 / FX_SUB) + ils;
                float reg[FX_SUB];
                frink_dequant_sub(
                    rp + (size_t)(sub / FX_NL) * (size_t)FX_BLOCK_BYTES,
                    sub % FX_NL,
                    reg);
                unsigned int* dp = (unsigned int*)&sa[lr * FX_LDS + ils * FX_SUB];
#pragma unroll
                for (int i = 0; i < FX_SUB; i += 2) {
                    dp[i / 2] = frink_pack2h(reg[i], reg[i + 1]);
                }
            }

            // B tile: two threads per token, sixteen k each, as two
            // 16-byte loads of the f16 activations the launch converted
            // once (n_cols is a multiple of 32 and the buffer a whole
            // allocation, so each run is 16-byte aligned). Tokens past
            // the batch are zero.
            {
                const int j = tid >> 1;
                const int kh = (tid & 1) * 16;
                const int col = c0 + j;
                uint4* dp = (uint4*)&sb[j * FX_LDS + kh];
                if (col < batch) {
                    const uint4* sp =
                        (const uint4*)(src1 + FX_SRC_ROW(col) * (size_t)n_cols + (size_t)(k0 + kh));
                    dp[0] = sp[0];
                    dp[1] = sp[1];
                } else {
                    dp[0] = make_uint4(0u, 0u, 0u, 0u);
                    dp[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }

            __syncthreads();

            // Fragments by ldmatrix: lane l addresses row (l & 7) of
            // matrix (l >> 3). For A the four matrices are rows 0-7 and
            // 8-15 at k 0-7, then the same rows at k 8-15 (a0..a3); for
            // B they are tokens 0-7 at k 0-7 and 8-15 (b0, b1 of one n8
            // fragment), then tokens 8-15 (the next fragment).
            const int lm = lane >> 3;
            const int lr8 = lane & 7;
#pragma unroll
            for (int ks = 0; ks < FX_TBK; ks += 16) {
                unsigned int af[FX_MF][4];
#pragma unroll
                for (int mf = 0; mf < FX_MF; mf++) {
                    const int row = wm * FX_WARP_M + mf * 16 + (lm & 1) * 8 + lr8;
                    frink_ldsm_x4(af[mf], &sa[row * FX_LDS + ks + (lm >> 1) * 8]);
                }
#pragma unroll
                for (int nf = 0; nf < FX_NF; nf += 2) {
                    const int tok = wn * FX_WARP_N + nf * 8 + (lm >> 1) * 8 + lr8;
                    unsigned int bf[4];
                    frink_ldsm_x4(bf, &sb[tok * FX_LDS + ks + (lm & 1) * 8]);
#pragma unroll
                    for (int mf = 0; mf < FX_MF; mf++) {
                        frink_mma16816(acc[mf][nf], af[mf], &bf[0]);
                        frink_mma16816(acc[mf][nf + 1], af[mf], &bf[2]);
                    }
                }
            }
        }
    }

    // C fragments: rows g and g + 8, tokens 2t and 2t + 1.
    const int g = lane >> 2;
    const int tg = lane & 3;
#pragma unroll
    for (int mf = 0; mf < FX_MF; mf++) {
        const int row0 = r0 + wm * FX_WARP_M + mf * 16 + g;
        const int row1 = row0 + 8;
#pragma unroll
        for (int nf = 0; nf < FX_NF; nf++) {
            const int col0 = c0 + wn * FX_WARP_N + nf * 8 + tg * 2;
            const int col1 = col0 + 1;
            if (col0 < batch) {
                if (row0 < n_rows) dst[FX_DST_ROW(col0) * (size_t)n_rows + (size_t)row0] = acc[mf][nf][0];
                if (row1 < n_rows) dst[FX_DST_ROW(col0) * (size_t)n_rows + (size_t)row1] = acc[mf][nf][2];
            }
            if (col1 < batch) {
                if (row0 < n_rows) dst[FX_DST_ROW(col1) * (size_t)n_rows + (size_t)row0] = acc[mf][nf][1];
                if (row1 < n_rows) dst[FX_DST_ROW(col1) * (size_t)n_rows + (size_t)row1] = acc[mf][nf][3];
            }
        }
    }
}
"#;

/// Columns whose raw bytes one stage holds: one block for the kinds
/// whose block is at least [`MIN_STAGE_ELEMS`] wide, enough whole
/// blocks to reach it for the narrow ones. A whole number of `TBK`
/// tiles either way, and a whole number of blocks, so a sub-block's
/// header is always in the stage beside its quants.
pub const fn stage_elems(kind: &MulMmKind) -> usize {
    if kind.block_elems >= MIN_STAGE_ELEMS {
        kind.block_elems
    } else {
        MIN_STAGE_ELEMS
    }
}

/// The narrowest raw stage: four 32-column blocks.
pub const MIN_STAGE_ELEMS: usize = 128;

/// Shared words per staged row: the stage's bytes, plus three for a
/// row that does not start on a word, rounded up to an odd count so
/// the dequant's lanes, one row each, land on distinct banks.
pub const fn raw_words(kind: &MulMmKind) -> usize {
    let bytes = stage_elems(kind) / kind.block_elems * kind.block_bytes;
    ((bytes + 3).div_ceil(4)) | 1
}

/// Static shared bytes the body declares for `kind`: the raw stage and
/// the two f16 tiles. Kept under the 48 KiB a launch gets without
/// opting in, for every kind (`every_kind_fits_static_shared_memory`).
pub const fn shared_bytes(kind: &MulMmKind) -> usize {
    TBM * raw_words(kind) * 4 + (TBM + TBN) * LDS * 2
}

/// f32 activations to f16 (round to nearest even, `cvt.rn.f16.f32`,
/// what the body's per-tile conversion did), once per launch: the
/// body reads its B tiles as f16.
pub const F32_TO_F16_SRC: &str = r#"
extern "C" __global__ void frink_f32_to_f16(const float* __restrict__ x, unsigned short* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x[i]));
        y[i] = h;
    }
}
"#;

/// NVRTC module and function of [`F32_TO_F16_SRC`].
pub const F32_TO_F16: (&str, &str) = ("frink_f32_to_f16", "frink_f32_to_f16");

/// The complete translation unit for one kind's tensor-core GEMM: the
/// SIMT unit's preamble (f16 decode, K-scale unpack, codebook, the
/// kind's `frink_dequant_sub`) with this body in place of the SIMT
/// one, under the names [`tc_names`] gives the kind.
pub fn tc_kernel_src(kind: &MulMmKind) -> String {
    let defines = format!(
        "#define FX_TBM {TBM}\n\
         #define FX_TBN {TBN}\n\
         #define FX_TBK {TBK}\n\
         #define FX_WM {WM}\n\
         #define FX_WN {WN}\n\
         #define FX_WARP_M {}\n\
         #define FX_WARP_N {}\n\
         #define FX_MF {}\n\
         #define FX_NF {}\n\
         #define FX_LDS {LDS}\n\
         #define FX_TTHREADS {TTHREADS}\n\
         #define FX_BLOCK_ELEMS {}\n\
         #define FX_SE {se}\n\
         #define FX_RAW_W {}\n",
        TBM / WM,
        TBN / WN,
        TBM / WM / 16,
        TBN / WN / 8,
        kind.block_elems,
        raw_words(kind),
        se = stage_elems(kind),
    );
    let (_, fn_name) = tc_names(kind);
    let body = TC_BODY_SRC.replace("FX_FN_NAME", fn_name);
    format!("{defines}{}", kernel_src_with_body(kind, &body))
}

/// [`tc_kernel_src`] for the grouped form: the experts of a routed layer
/// in one launch, one per `blockIdx.z` (`FX_GROUPED`).
pub fn tc_grouped_kernel_src(kind: &MulMmKind) -> String {
    let (_, fn_name) = tc_grouped_names(kind);
    let src = tc_kernel_src(kind);
    let (_, plain) = tc_names(kind);
    format!("#define FX_GROUPED 1\n{}", src.replace(plain, fn_name))
}

/// The NVRTC names of [`tc_grouped_kernel_src`]: the tensor-core names
/// with `_grouped` after them.
pub fn tc_grouped_names(kind: &MulMmKind) -> (&'static str, &'static str) {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<HashMap<&'static str, (&'static str, &'static str)>>> =
        OnceLock::new();
    let (m, f) = tc_names(kind);
    *NAMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .entry(kind.name)
        .or_insert_with(|| {
            (
                Box::leak(format!("{m}_grouped").into_boxed_str()),
                Box::leak(format!("{f}_grouped").into_boxed_str()),
            )
        })
}

/// The NVRTC module and function names for a kind's tensor-core
/// kernel: the SIMT names with a `_tc` suffix, leaked once so the
/// load-once cache can key on `&'static str` as it does for every
/// other kernel.
pub fn tc_names(kind: &MulMmKind) -> (&'static str, &'static str) {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<HashMap<&'static str, (&'static str, &'static str)>>> =
        OnceLock::new();
    let mut map = NAMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    *map.entry(kind.name).or_insert_with(|| {
        (
            Box::leak(format!("{}_tc", kind.module_name).into_boxed_str()),
            Box::leak(format!("{}_tc", kind.fn_name).into_boxed_str()),
        )
    })
}

/// Grid for a launch: tokens along x, rows along y, as the SIMT body.
pub fn tc_grid_dims(n_rows: usize, batch: usize) -> (usize, usize) {
    (batch.div_ceil(TBN), n_rows.div_ceil(TBM))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul_mm::KINDS;

    #[test]
    fn every_kind_emits_a_tensor_core_unit_that_names_its_dequant() {
        for kind in KINDS {
            let src = tc_kernel_src(kind);
            let (module, fn_name) = tc_names(kind);
            assert!(module.ends_with("_tc") && fn_name.ends_with("_tc"));
            assert!(
                src.contains(&format!("__global__ void {fn_name}(")),
                "{}",
                kind.name
            );
            assert!(src.contains("frink_dequant_sub("), "{}", kind.name);
            assert!(src.contains("mma.sync.aligned.m16n8k16"), "{}", kind.name);
            assert!(
                src.contains("#define FX_SUB 16\n"),
                "{}: the sub-block width the loader assumes",
                kind.name
            );
            // No stale SIMT geometry name survives into this unit's body.
            assert!(!src.contains("FX_TM"), "{}", kind.name);
        }
    }

    /// The body declares its tiles statically, and a static
    /// declaration past 48 KiB fails to load. Q6_K's 210-byte block is
    /// the widest stage.
    #[test]
    fn every_kind_fits_static_shared_memory() {
        for kind in KINDS {
            assert!(
                shared_bytes(kind) <= 48 * 1024,
                "{}: {} bytes",
                kind.name,
                shared_bytes(kind)
            );
            let stage = stage_elems(kind);
            assert!(stage.is_multiple_of(kind.block_elems) && stage.is_multiple_of(TBK));
            let bytes = stage / kind.block_elems * kind.block_bytes;
            assert!(raw_words(kind) * 4 >= bytes + 3, "{}", kind.name);
            assert_eq!(raw_words(kind) % 2, 1, "{}", kind.name);
        }
    }

    #[test]
    fn the_names_are_stable_across_calls() {
        let a = tc_names(&crate::mul_mm::Q8_0);
        let b = tc_names(&crate::mul_mm::Q8_0);
        assert!(std::ptr::eq(a.0, b.0) && std::ptr::eq(a.1, b.1));
    }
}
