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
__device__ __forceinline__ unsigned int frink_smem_addr(const void* p) {
    unsigned int a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}

__device__ __forceinline__ void frink_ldsm_x4(unsigned int* r, const unsigned short* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(frink_smem_addr(p)));
}

// 16 bytes global -> shared, asynchronously; `src_bytes` of them read, the
// rest zero-filled (0 for a token past the batch).
__device__ __forceinline__ void frink_cp16(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
        :: "r"(frink_smem_addr(dst)), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void frink_cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void frink_cp_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// One word of stage `s`'s raw bytes for staged row `lr`, word `w`:
// whole when it lies inside the matrix, byte by byte at its very end,
// zero past the stage. Out-of-range rows re-read the last valid row
// (llama's clamp) and are dropped at the store.
__device__ __forceinline__ unsigned int frink_raw_word(
    const unsigned char* __restrict__ src0, size_t total_bytes, int n_rows, int n_cols,
    int row_bytes, int r0, int s, int idx
) {
    const int lr = idx / FX_RAW_W;
    const int w = idx - lr * FX_RAW_W;
    int row = r0 + lr;
    if (row >= n_rows) row = n_rows - 1;
    const int s0 = s * FX_SE;
    const int se = min(FX_SE, n_cols - s0);
    const size_t sbytes = (size_t)(se / FX_BLOCK_ELEMS) * FX_BLOCK_BYTES;
    const size_t a = (size_t)row * (size_t)row_bytes + (size_t)(s0 / FX_BLOCK_ELEMS) * FX_BLOCK_BYTES;
    const size_t wa = (a & ~(size_t)3) + (size_t)w * 4;
    if (wa >= a + sbytes) return 0u;
    if (wa + 4 <= total_bytes) return __ldg((const unsigned int*)(src0 + wa));
    unsigned int v = 0u;
    for (int b = 0; b < 4; b++) {
        if (wa + b < total_bytes) v |= (unsigned int)src0[wa + b] << (8 * b);
    }
    return v;
}

// A tile: tile `t0 / FX_TBK` of stage `s`, one FX_SUB-element sub-block
// per thread, dequantized out of the staged bytes by the kind's own
// function and packed to f16 pairs.
__device__ __forceinline__ void frink_dequant_tile(
    unsigned short* sa_buf, const unsigned int* raw, int s, int t0, int r0, int n_rows,
    int row_bytes, int tid
) {
    const int lr = tid >> 1;
    const int ils = tid & 1;
    int row = r0 + lr;
    if (row >= n_rows) row = n_rows - 1;
    const size_t stage_off = (size_t)(s * FX_SE / FX_BLOCK_ELEMS) * FX_BLOCK_BYTES;
    const int o = (int)(((size_t)row * (size_t)row_bytes + stage_off) & 3);
    const unsigned char* rp = (const unsigned char*)&raw[lr * FX_RAW_W] + o;
    const int sub = (t0 / FX_SUB) + ils;
    float reg[FX_SUB];
    frink_dequant_sub(rp + (size_t)(sub / FX_NL) * (size_t)FX_BLOCK_BYTES, sub % FX_NL, reg);
    unsigned int* dp = (unsigned int*)&sa_buf[lr * FX_LDS + ils * FX_SUB];
#pragma unroll
    for (int i = 0; i < FX_SUB; i += 2) dp[i / 2] = frink_pack2h(reg[i], reg[i + 1]);
}

// B tile: the f16 activations of the block's FX_TBN tokens at columns
// k0 .. k0 + 32, by cp.async, a token past the batch zero-filled. The
// activations were converted once per launch (`frink_f32_to_f16`), so
// each block no longer converts them per tile.
__device__ __forceinline__ void frink_load_b(
    unsigned short* sb_buf, const unsigned short* __restrict__ src1, int c0, int k0,
    int n_cols, int batch, int tid
) {
    for (int c = tid; c < FX_TBN * 4; c += FX_TTHREADS) {
        const int j = c >> 2;
        const int p = c & 3;
        const int col = c0 + j;
        const unsigned short* src =
            src1 + (size_t)(col < batch ? col : 0) * (size_t)n_cols + (size_t)(k0 + p * 8);
        frink_cp16(&sb_buf[j * FX_LDS + p * 8], src, col < batch ? 16 : 0);
    }
}

extern "C" __global__ void __launch_bounds__(FX_TTHREADS, 1) FX_FN_NAME(
    const unsigned char* __restrict__ src0,
    const unsigned short* __restrict__ src1,
    float* __restrict__ dst,
    int n_rows,
    int n_cols,
    int batch,
    int row_bytes
) {
    // Dynamic shared memory: the A and B f16 tiles twice each (the tile
    // the tensor cores read and the one being built), then the raw bytes
    // of two stages (the one being dequantized and the next one, which
    // arrives a slice per tile). See `mul_mm_tc::shared_bytes`.
    extern __shared__ __align__(16) unsigned char fx_smem[];
    unsigned short* sa = (unsigned short*)fx_smem;
    unsigned short* sb = sa + 2 * FX_TBM * FX_LDS;
    unsigned int* sraw = (unsigned int*)(sb + 2 * FX_TBN * FX_LDS);

    const int r0 = blockIdx.y * FX_TBM;
    const int c0 = blockIdx.x * FX_TBN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp % FX_WM;
    const int wn = warp / FX_WM;
    const size_t total_bytes = (size_t)n_rows * (size_t)row_bytes;
    const int n_tiles = n_cols / FX_TBK;
    const int tiles_per_stage = FX_SE / FX_TBK;
    const int n_stages = (n_cols + FX_SE - 1) / FX_SE;
    const int raw_words = FX_TBM * FX_RAW_W;
    // Words of the next stage each tile of a full stage prefetches.
    const int slice = (raw_words + tiles_per_stage - 1) / tiles_per_stage;

    float acc[FX_MF][FX_NF][4];
#pragma unroll
    for (int mf = 0; mf < FX_MF; mf++) {
#pragma unroll
        for (int nf = 0; nf < FX_NF; nf++) {
            acc[mf][nf][0] = 0.f; acc[mf][nf][1] = 0.f;
            acc[mf][nf][2] = 0.f; acc[mf][nf][3] = 0.f;
        }
    }

    // Prologue: stage 0's raw bytes, tile 0 of A and B.
    for (int idx = tid; idx < raw_words; idx += FX_TTHREADS) {
        sraw[idx] = frink_raw_word(src0, total_bytes, n_rows, n_cols, row_bytes, r0, 0, idx);
    }
    frink_load_b(sb, src1, c0, 0, n_cols, batch, tid);
    frink_cp_commit();
    __syncthreads();
    frink_dequant_tile(sa, sraw, 0, 0, r0, n_rows, row_bytes, tid);
    frink_cp_wait_all();
    __syncthreads();

    const int lm = lane >> 3;
    const int lr8 = lane & 7;
    for (int t = 0; t < n_tiles; t++) {
        const int cur = t & 1;
        const int nxt = cur ^ 1;
        const int s = t / tiles_per_stage;
        const int tt = t - s * tiles_per_stage;
        const bool next_stage = s + 1 < n_stages;

        // 1. This tile's slice of the next stage's raw words, into
        //    registers: the loads are in flight while the MMAs run.
        unsigned int pf[FX_PF];
        const int base = tt * slice;
        const int end = min(base + slice, raw_words);
#pragma unroll
        for (int p = 0; p < FX_PF; p++) {
            const int idx = base + tid + p * FX_TTHREADS;
            pf[p] = (next_stage && idx < end)
                ? frink_raw_word(src0, total_bytes, n_rows, n_cols, row_bytes, r0, s + 1, idx)
                : 0u;
        }
        // 2. The next tile of B, asynchronously.
        if (t + 1 < n_tiles) {
            frink_load_b(sb + nxt * FX_TBN * FX_LDS, src1, c0, (t + 1) * FX_TBK, n_cols, batch, tid);
        }
        frink_cp_commit();

        // 3. This tile's MMAs.
        const unsigned short* sa_cur = sa + cur * FX_TBM * FX_LDS;
        const unsigned short* sb_cur = sb + cur * FX_TBN * FX_LDS;
#pragma unroll
        for (int ks = 0; ks < FX_TBK; ks += 16) {
            unsigned int af[FX_MF][4];
#pragma unroll
            for (int mf = 0; mf < FX_MF; mf++) {
                const int row = wm * FX_WARP_M + mf * 16 + (lm & 1) * 8 + lr8;
                frink_ldsm_x4(af[mf], &sa_cur[row * FX_LDS + ks + (lm >> 1) * 8]);
            }
#pragma unroll
            for (int nf = 0; nf < FX_NF; nf += 2) {
                const int tok = wn * FX_WARP_N + nf * 8 + (lm >> 1) * 8 + lr8;
                unsigned int bf[4];
                frink_ldsm_x4(bf, &sb_cur[tok * FX_LDS + ks + (lm & 1) * 8]);
#pragma unroll
                for (int mf = 0; mf < FX_MF; mf++) {
                    frink_mma16816(acc[mf][nf], af[mf], &bf[0]);
                    frink_mma16816(acc[mf][nf + 1], af[mf], &bf[2]);
                }
            }
        }

        // 4. The prefetched slice into the next stage's raw buffer.
        if (next_stage) {
            unsigned int* raw_nxt = sraw + ((s + 1) & 1) * raw_words;
#pragma unroll
            for (int p = 0; p < FX_PF; p++) {
                const int idx = base + tid + p * FX_TTHREADS;
                if (idx < end) raw_nxt[idx] = pf[p];
            }
        }

        // 5. The next tile of A. When it opens the next stage, every
        //    slice of that stage has to be in shared memory first.
        if (t + 1 < n_tiles) {
            const int s1 = (t + 1) / tiles_per_stage;
            const int tt1 = (t + 1) - s1 * tiles_per_stage;
            if (s1 != s) __syncthreads();
            frink_dequant_tile(sa + nxt * FX_TBM * FX_LDS, sraw + (s1 & 1) * raw_words, s1,
                tt1 * FX_TBK, r0, n_rows, row_bytes, tid);
        }
        frink_cp_wait_all();
        __syncthreads();
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
                if (row0 < n_rows) dst[(size_t)col0 * (size_t)n_rows + (size_t)row0] = acc[mf][nf][0];
                if (row1 < n_rows) dst[(size_t)col0 * (size_t)n_rows + (size_t)row1] = acc[mf][nf][2];
            }
            if (col1 < batch) {
                if (row0 < n_rows) dst[(size_t)col1 * (size_t)n_rows + (size_t)row0] = acc[mf][nf][1];
                if (row1 < n_rows) dst[(size_t)col1 * (size_t)n_rows + (size_t)row1] = acc[mf][nf][3];
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

/// Dynamic shared bytes the body asks for: the A and B f16 tiles twice
/// each and two stages of raw bytes. Under [`MAX_SHARED_BYTES`] for
/// every kind (`every_kind_fits_the_opt_in_shared_memory`).
pub const fn shared_bytes(kind: &MulMmKind) -> usize {
    2 * TBM * raw_words(kind) * 4 + 2 * (TBM + TBN) * LDS * 2
}

/// The smallest opt-in shared memory per block of the devices the body
/// runs on: `sm_86`, `sm_89` and `sm_120` give 99 KiB; `sm_80` and
/// `sm_90` more.
pub const MAX_SHARED_BYTES: usize = 99 * 1024;

/// Raw words each thread prefetches per tile: a full stage's words
/// spread over the stage's tiles and the block's threads.
pub const fn prefetch_words(kind: &MulMmKind) -> usize {
    let words = TBM * raw_words(kind);
    let tiles = stage_elems(kind) / TBK;
    words.div_ceil(tiles).div_ceil(TTHREADS)
}

/// f32 activations to f16 (round to nearest even, `cvt.rn.f16.f32`,
/// what the body's per-tile conversion did), once per launch: the body
/// reads its B tiles as f16 by `cp.async`.
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
         #define FX_RAW_W {}\n\
         #define FX_PF {}\n",
        TBM / WM,
        TBN / WN,
        TBM / WM / 16,
        TBN / WN / 8,
        kind.block_elems,
        raw_words(kind),
        prefetch_words(kind),
        se = stage_elems(kind),
    );
    let (_, fn_name) = tc_names(kind);
    let body = TC_BODY_SRC.replace("FX_FN_NAME", fn_name);
    format!("{defines}{}", kernel_src_with_body(kind, &body))
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
                src.contains(&format!(
                    "__global__ void __launch_bounds__(FX_TTHREADS, 1) {fn_name}("
                )),
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

    /// The body's dynamic shared memory fits the opt-in limit of every
    /// device it runs on, for every kind; Q6_K's 210-byte block is the
    /// widest stage. The prefetch covers a whole stage.
    #[test]
    fn every_kind_fits_the_opt_in_shared_memory() {
        for kind in KINDS {
            assert!(
                shared_bytes(kind) <= MAX_SHARED_BYTES,
                "{}: {} bytes",
                kind.name,
                shared_bytes(kind)
            );
            let stage = stage_elems(kind);
            assert!(stage.is_multiple_of(kind.block_elems) && stage.is_multiple_of(TBK));
            let bytes = stage / kind.block_elems * kind.block_bytes;
            assert!(raw_words(kind) * 4 >= bytes + 3, "{}", kind.name);
            assert_eq!(raw_words(kind) % 2, 1, "{}", kind.name);
            let tiles = stage / TBK;
            assert!(
                prefetch_words(kind) * TTHREADS * tiles >= TBM * raw_words(kind),
                "{}: the prefetch leaves words of the stage unloaded",
                kind.name
            );
        }
    }

    #[test]
    fn the_names_are_stable_across_calls() {
        let a = tc_names(&crate::mul_mm::Q8_0);
        let b = tc_names(&crate::mul_mm::Q8_0);
        assert!(std::ptr::eq(a.0, b.0) && std::ptr::eq(a.1, b.1));
    }
}
