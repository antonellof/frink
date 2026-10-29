//! `mul_mm`: a batched quantized GEMM for CUDA -- the CUDA C source and
//! the per-quant-kind dispatch table.
//!
//! # Run on hardware
//!
//! The kernels have executed on a GPU: the `--ignored` hardware tests
//! (`cargo test -p frink-cuda --features cuda -- --ignored`) passed on an
//! RTX 3090 on 2026-09-15, covering every kind [`KINDS`] held that day, and the
//! tensor-core body has served batched prefill since 0.22.0 (see
//! `CHANGELOG.md` and `docs/plans/cpu-cuda-parity.md`). The standing rule
//! still applies to any new kind or body added here: a CUDA claim is
//! worth nothing until that command has run on a real device and the
//! result is written down.
//!
//! What *is* established, and how:
//!
//! 1. **It is a port, not a design.** The arithmetic comes from
//!    `frink-metal`'s `mul_mm_sg_impl` (`crates/frink-metal/src/gpu.rs`),
//!    which is at parity with llama.cpp on Metal and has goldens. The
//!    per-kind unpack functions in [`crate::mul_mm_kinds`] are
//!    line-for-line transcriptions of that file's `Q8_0Dequant` /
//!    `Q4_0Dequant` / `Q5_0Dequant` / `Q5KDequant` / `Q6KDequant`
//!    functors, which are themselves llama's `dequantize_q8_0` /
//!    `dequantize_q4_0` / `dequantize_q5_0` / `dequantize_q5_K` /
//!    `dequantize_q6_K`. The three codebook rows come from llama's
//!    `ggml/src/ggml-cuda/dequantize.cuh` directly (`:408`, `:424`,
//!    `:439`), because Metal has no MXFP4 and its IQ4_XS functor uses a
//!    different sub-block partition.
//! 2. **It has a scalar twin.** [`crate::mul_mm_ref`] emulates this
//!    kernel on the CPU -- same tiling, same clamping, same index
//!    arithmetic, same accumulation order -- and its tests, which run in
//!    the default (no-GPU) build, hold it against an independent
//!    dequantize-then-GEMM built on `frink_quant`. A transcription
//!    error in the index math or the unpack shows up there.
//! 3. **Geometry cannot drift.** Every tile constant in the emitted CUDA
//!    is `#define`d from the Rust constants in this module, which are
//!    the same constants the twin uses. The kernel and its twin cannot
//!    disagree about a tile size; they can only disagree about the ~10
//!    lines of unpack expression, which is what (1) and (2) cover.
//!
//! 4. **The emitted C itself was executed.**
//!    `tools/mul_mm_host_check/run.sh` compiles each generated `.cu`
//!    against a keyword-and-barrier shim and runs it on the host CPU --
//!    one real thread per CUDA thread, a counting barrier for
//!    `__syncthreads()`, one block at a time so `__shared__` behaves --
//!    then compares it to the twin. Result on 2026-09-01, macOS/clang,
//!    both kinds then in the table, three shapes (exact tiles, partial
//!    on both axes, narrow batch): **zero mismatches, bit for bit**,
//!    including 1,458 positions where a degenerate f16 scale made both
//!    sides NaN together. Deleting one term of the CUDA-only index
//!    arithmetic makes it fail, so it is a check and not a formality.
//!
//!    Re-run on 2026-09-05 over all six kinds: **zero mismatches**
//!    again, 40,932 compared positions. That run was the first for the
//!    K-quants and for Q5_0 -- the tool took `n_cols` from a fixed list
//!    that is not a whole Q4_K super-block, so from the day the
//!    K-quants landed it panicked on the first one and checked
//!    NOTHING. A tool that cannot fire is worse than no tool; `n_cols`
//!    is now rounded up per kind. Sabotaging Q5_0's `qh` shift
//!    (`12` for `16`) makes it report 6,096 mismatches.
//!
//!    Re-run on 2026-09-09 over all ELEVEN kinds -- the three codebook
//!    formats, Q2_K and Q3_K included: **zero mismatches**, 75,042
//!    compared positions.
//!    That run was also the tool's first since the inner loop started
//!    reading its operands as `float4` -- the host shim had no such
//!    type, so every kind failed to compile and `set -e` aborted the
//!    script. It was never green rather than green and blind, but the
//!    effect on coverage was the same. Sabotaging IQ4_XS's `scales_h`
//!    shift (`2 * il` for `2 * ib`) in the CUDA only makes it report
//!    4,000 mismatches out of 4,096, and inverting Q3_K's `hmask` bias
//!    test the same way reports 3,968.
//!
//! What none of that covers, and what only hardware can settle: that
//! NVRTC accepts the source (clang and NVRTC are different front ends),
//! that the barrier placement survives a real warp scheduler, that the
//! launch configuration is valid on the target device, and what any of
//! it costs. A real GPU also contracts `acc += a * b` into an FMA, so
//! on-device results will be *close to* rather than equal to the twin's;
//! the hardware test compares with a relative tolerance for that reason.
//!
//! # Why this shape
//!
//! `frink-cuda` had no matrix-matrix product of any kind, so a batched
//! prefill decomposed into one matvec per position
//! (`crates/frink-core/src/weight_matrix.rs`, the CUDA batch arm). This
//! is the naive tiled `mul_mm` half of
//! `docs/plans/llama-cpp-gap-inventory.md` §2.7; the `dp4a` integer path
//! (llama's `mmq.cu`) is explicitly *not* attempted here.
//!
//! Unlike Metal's version this uses no matrix-fragment intrinsics
//! (`wmma`/`mma`): the K-loop is a plain fp32 FMA over shared-memory
//! tiles. That costs the constant factor and buys a kernel whose
//! arithmetic a CPU twin can reproduce exactly, which is the only kind
//! of correctness available without a device.

/// Rows of the weight matrix per threadblock tile.
pub const BM: usize = 64;
/// Tokens (batch entries) per threadblock tile.
pub const BN: usize = 128;
/// K-elements consumed per tile step. Must be a multiple of [`SUB`], and
/// every real quantized row length is a multiple of 32, so a K-loop that
/// steps 32 never straddles a partial block.
pub const BK: usize = 32;
/// Rows of the output micro-tile each thread owns.
pub const TM: usize = 4;
/// Columns of the output micro-tile each thread owns.
pub const TN: usize = 8;
/// Threads per block. `BM/TM * BN/TN` -- one thread per micro-tile.
pub const THREADS: usize = (BM / TM) * (BN / TN);
/// Elements produced by one call to the per-kind unpack function. This
/// is llama's (and `frink-metal`'s) sub-block granularity: `il` selects
/// which 16 consecutive elements of a super-block to decode.
pub const SUB: usize = 16;

// Compile-time geometry gates. These are the invariants the kernel's
// `tx`/`ty` decomposition and its A-tile loader assume; a retune that
// broke one would otherwise produce a kernel that launches and is
// wrong, so they fail the build rather than a test.
const _: () = assert!(THREADS == (BM / TM) * (BN / TN));
const _: () = assert!(BM.is_multiple_of(TM) && BN.is_multiple_of(TN));
const _: () = assert!(
    BK.is_multiple_of(SUB),
    "the K-tile must be whole sub-blocks"
);
const _: () = assert!(
    BM * (BK / SUB) <= THREADS,
    "the A-tile loader uses a prefix of the block"
);
const _: () = assert!(THREADS <= 1024, "CUDA caps a block at 1024 threads");
// The inner loop reads its micro-tile operands as `float4`, which is
// what keeps a warp's 32 lanes off eight shared-memory banks. That
// needs each thread's slice to start 16-byte aligned: the micro-tile
// widths must be multiples of four, and so must the shared rows they
// index into, or the fourth lane of a row starts mid-vector.
const _: () = assert!(
    TM.is_multiple_of(4) && TN.is_multiple_of(4),
    "the inner loop loads float4 from shared memory"
);
const _: () = assert!(
    BM.is_multiple_of(4) && BN.is_multiple_of(4),
    "a shared row must keep the next row 16-byte aligned"
);

/// One quantized weight format the GEMM can consume.
///
/// Adding a format is a row in [`KINDS`] plus a `dequant_src` snippet --
/// never a second copy of the GEMM body. That is the seam
/// `frink-metal` proved: its `mul_mm_sg_impl` is one templated body
/// with seven `Dequant` functors, written that way *because* the
/// previous copy-per-format generation is how
/// `gqa_prefill_fa_vec_d256` ended up handling half a head.
#[derive(Debug, Clone, Copy)]
pub struct MulMmKind {
    /// GGUF quant name, for error messages.
    pub name: &'static str,
    /// NVRTC module cache key. Must be unique per kind.
    pub module_name: &'static str,
    /// `__global__` entry point name inside that module.
    pub fn_name: &'static str,
    /// On-disk stride of one super-block.
    pub block_bytes: usize,
    /// Elements one super-block decodes to.
    pub block_elems: usize,
    /// CUDA C defining
    /// `void frink_dequant_sub(const unsigned char* xb, int il, float* reg)`,
    /// writing `SUB` floats: the elements at `[SUB*il, SUB*il + SUB)`
    /// of the super-block at `xb`, in ascending element order.
    ///
    /// **The contract is that signature and nothing narrower.** It was
    /// once informally "multiply the stored code by a scale and add a
    /// bias", which is true of every affine format and true of no
    /// codebook one: IQ4_NL, IQ4_XS and MXFP4 read a 4-bit code and
    /// *index a table* with it. Those kinds fill [`Self::codebook`] and
    /// look the value up; nothing else about the seam changes, which is
    /// the point of the seam.
    pub dequant_src: &'static str,
    /// The 16-entry table `dequant_src` indexes, for the formats whose
    /// stored code is an index rather than a magnitude. `None` for the
    /// affine kinds, which need no table.
    ///
    /// [`kernel_src`] emits this as a `__constant__` array ahead of
    /// `dequant_src`, so it is visible to every thread of the block
    /// without being reloaded per element.
    pub codebook: Option<Codebook>,
    /// The scalar twin of `dequant_src`: the same arithmetic in Rust,
    /// on the host, in the same order. It sits in this struct rather
    /// than in a parallel table so a kind cannot be added without one --
    /// the untestable half and the testable half are the same row.
    pub dequant_twin: fn(xb: &[u8], il: usize, reg: &mut [f32; SUB]),
    /// The IQ grid and sign tables `dequant_src` indexes, emitted ahead
    /// of it from `frink_quant::iq_tables` ([`crate::iq_grids`]). Empty
    /// for every kind that reads no grid.
    pub grids: &'static [crate::iq_grids::IqGrid],
}

/// The 16-entry value table a codebook format's 4-bit code indexes.
///
/// One slice serves both halves of the kernel: [`kernel_src`] formats
/// [`Self::values`] into the emitted `__constant__` array, and the Rust
/// `dequant_twin` beside it indexes the same `values`. There is no
/// second copy of the numbers to drift, which matters more here than
/// for an affine format -- a codebook is 16 arbitrary constants that no
/// arithmetic can re-derive, so a single transposed pair would decode
/// every tensor slightly wrong and nothing would look obviously broken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Codebook {
    /// The `__constant__` array's identifier in the emitted CUDA C.
    /// Must be what `dequant_src` spells.
    pub c_name: &'static str,
    /// The values, code 0 first.
    pub values: &'static [f32; 16],
}

/// Emits one codebook as a `__constant__` array.
///
/// `{:?}` on an `f32` round-trips in Rust, and every value in every
/// table here is exactly representable, so the emitted literal names the
/// same float the twin uses. `-0.0` survives it, which MXFP4 needs: its
/// code 8 is negative zero and printing it as `0` would be a different
/// number.
fn codebook_src(cb: &Codebook) -> String {
    let values: Vec<String> = cb.values.iter().map(|v| format!("{v:?}f")).collect();
    format!(
        "\n__constant__ float {}[16] = {{{}}};\n",
        cb.c_name,
        values.join(", ")
    )
}

impl MulMmKind {
    /// 16-value sub-blocks per super-block -- llama's `nl` template
    /// argument (2 for the 32-element legacy formats, 16 for the
    /// 256-element K-quants).
    pub const fn nl(&self) -> usize {
        self.block_elems / SUB
    }
}

// The per-kind rows live in `mul_mm_kinds`, one module per format
// family, because this file is the GEMM and they are the formats. They
// are re-exported here unchanged: `mul_mm::Q4_K` is the path every
// caller and every test already uses, and a refactor that moves a
// definition should not move a public name.
pub use crate::mul_mm_kinds::kquant::K_SCALE_MIN_SRC;
pub use crate::mul_mm_kinds::{
    IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4, PTQ1_0, Q2_K,
    Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, TQ1_0,
};

/// Scalar twin of the CUDA `frink_f16_to_f32` in `F16_SRC`: the same
/// bit surgery, including the `exp == 31` NaN/Inf arm. `ldexpf(m, e)`
/// with an exact power of two is a multiply, so this is bit-identical
/// rather than merely close.
///
/// Held against `half::f16` by a test, which is what makes it a twin of
/// something and not a second guess.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);
    let scale = if exp == 0 {
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 31 {
        if mant != 0 {
            f32::from_bits(0x7fc0_0000)
        } else {
            f32::from_bits(0x7f80_0000)
        }
    } else {
        ((mant | 0x400) as f32) * 2f32.powi(exp as i32 - 25)
    };
    if sign != 0 {
        -scale
    } else {
        scale
    }
}

/// The dispatch table. A caller looks up by GGUF quant name; a new
/// format is one row here.
pub const KINDS: &[MulMmKind] = &[
    Q8_0, Q4_0, Q5_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4, Q4_1, Q5_1, TQ1_0,
    PTQ1_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M,
];

/// Looks up a kind by its GGUF quant name (`"Q4_0"`, `"Q8_0"`).
/// `None` means this GEMM does not implement that format -- the caller
/// must fall back and say so, never compute something else.
pub fn kind_by_name(name: &str) -> Option<&'static MulMmKind> {
    KINDS.iter().find(|k| k.name == name)
}

/// f16 -> f32 by explicit bit surgery, shared with the matvec kernels in
/// `gpu.rs`. NVRTC has `__half` available but only with the CUDA headers
/// on the include path, which this crate deliberately does not require.
const F16_SRC: &str = r#"
__device__ __forceinline__ float frink_f16_to_f32(unsigned short bits) {
    unsigned int sign = (bits >> 15) & 0x1u;
    unsigned int exp = (bits >> 10) & 0x1Fu;
    unsigned int mant = bits & 0x3FFu;
    float scale;
    if (exp == 0) {
        scale = ldexpf((float)mant, -24);
    } else if (exp == 31) {
        scale = mant ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
    } else {
        scale = ldexpf((float)(mant | 0x400), (int)exp - 25);
    }
    return sign ? -scale : scale;
}
"#;

/// The GEMM body, identical for every quant kind.
///
/// `src0` is the quantized weight matrix, `n_rows` rows of `row_bytes`
/// each. `src1` is `batch` activation rows of `n_cols` f32. `dst` is
/// written as `dst[token * n_rows + row]`, which is the layout
/// `WeightMatrix::apply_batch` already returns.
///
/// Every `FX_*` name is `#define`d by [`kernel_src`] from this module's
/// Rust constants, so the emitted kernel and [`crate::mul_mm_ref`]
/// cannot disagree about geometry.
///
/// The out-of-range row clamp is llama's trick, kept from
/// `mul_mm_sg_impl`: a lane whose row does not exist re-reads the last
/// valid row rather than branching, so the load loop stays uniform, and
/// its result is thrown away by the bounds check at the store. Reading
/// a real row also means the dequant never touches unmapped bytes.
const BODY_SRC: &str = r#"
extern "C" __global__ void FX_FN_NAME(
    const unsigned char* __restrict__ src0,
    const float* __restrict__ src1,
    float* __restrict__ dst,
    int n_rows,
    int n_cols,
    int batch,
    int row_bytes
) {
    __shared__ float sa[FX_BK][FX_BM];
    __shared__ float sb[FX_BK][FX_BN];

    const int r0 = blockIdx.y * FX_BM;
    const int r1 = blockIdx.x * FX_BN;
    const int tid = threadIdx.x;

    // Micro-tile owner: `tx` walks rows, `ty` walks tokens.
    const int tx = tid % (FX_BM / FX_TM);
    const int ty = tid / (FX_BM / FX_TM);

    float acc[FX_TN][FX_TM];
#pragma unroll
    for (int n = 0; n < FX_TN; n++) {
#pragma unroll
        for (int m = 0; m < FX_TM; m++) {
            acc[n][m] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < n_cols; k0 += FX_BK) {
        // Guards the previous iteration's reads of sa/sb.
        __syncthreads();

        // A-tile: one thread decodes one FX_SUB-element sub-block, so
        // FX_BM * (FX_BK / FX_SUB) threads cover the tile. Stored
        // k-major so the K-loop below reads one row of sa per step.
        if (tid < FX_BM * (FX_BK / FX_SUB)) {
            const int lr = tid / (FX_BK / FX_SUB);
            const int ils = tid % (FX_BK / FX_SUB);
            int row = r0 + lr;
            if (row >= n_rows) {
                row = n_rows - 1;
            }
            const unsigned char* rp =
                src0 + (size_t)row * (size_t)row_bytes;
            const int sub = (k0 / FX_SUB) + ils;
            float reg[FX_SUB];
            frink_dequant_sub(
                rp + (size_t)(sub / FX_NL) * (size_t)FX_BLOCK_BYTES,
                sub % FX_NL,
                reg);
#pragma unroll
            for (int i = 0; i < FX_SUB; i++) {
                sa[FX_SUB * ils + i][lr] = reg[i];
            }
        }

        // B-tile: consecutive threads read consecutive k of one token.
        // Tokens past the end are zero-filled rather than skipped, so
        // the K-loop needs no per-token predicate.
        for (int idx = tid; idx < FX_BK * FX_BN; idx += FX_THREADS) {
            const int j = idx / FX_BK;
            const int kk = idx % FX_BK;
            const int col = r1 + j;
            sb[kk][j] = (col < batch)
                ? src1[(size_t)col * (size_t)n_cols + (size_t)(k0 + kk)]
                : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < FX_BK; kk++) {
            float a[FX_TM];
            float b[FX_TN];
            // One 16-byte load per four operands, not four 4-byte ones.
            // A warp's 32 lanes take 16 distinct `tx`, so the scalar
            // form had them striding four floats apart across eight
            // banks -- a four-way conflict on the hottest load in the
            // kernel. As `float4` the same 16 lanes read 256 contiguous
            // bytes, which the hardware serves without conflict.
#pragma unroll
            for (int m = 0; m < FX_TM; m += 4) {
                const float4 v = *(const float4*)&sa[kk][tx * FX_TM + m];
                a[m + 0] = v.x;
                a[m + 1] = v.y;
                a[m + 2] = v.z;
                a[m + 3] = v.w;
            }
#pragma unroll
            for (int n = 0; n < FX_TN; n += 4) {
                const float4 v = *(const float4*)&sb[kk][ty * FX_TN + n];
                b[n + 0] = v.x;
                b[n + 1] = v.y;
                b[n + 2] = v.z;
                b[n + 3] = v.w;
            }
#pragma unroll
            for (int n = 0; n < FX_TN; n++) {
#pragma unroll
                for (int m = 0; m < FX_TM; m++) {
                    acc[n][m] += a[m] * b[n];
                }
            }
        }
    }

    for (int n = 0; n < FX_TN; n++) {
        const int col = r1 + ty * FX_TN + n;
        if (col >= batch) {
            continue;
        }
        for (int m = 0; m < FX_TM; m++) {
            const int row = r0 + tx * FX_TM + m;
            if (row < n_rows) {
                dst[(size_t)col * (size_t)n_rows + (size_t)row] = acc[n][m];
            }
        }
    }
}
"#;

/// Emits the complete CUDA C translation unit for one quant kind.
///
/// Deterministic and side-effect free, which is what lets the tests in
/// [`crate::mul_mm_ref`] assert against it without a device.
pub fn kernel_src(kind: &MulMmKind) -> String {
    let defines = format!(
        "#define FX_BM {}\n\
         #define FX_BN {}\n\
         #define FX_BK {}\n\
         #define FX_TM {}\n\
         #define FX_TN {}\n\
         #define FX_THREADS {}\n",
        BM, BN, BK, TM, TN, THREADS,
    );
    let body = BODY_SRC.replace("FX_FN_NAME", kind.fn_name);
    format!("{defines}{}", kernel_src_with_body(kind, &body))
}

/// The kind's preamble -- the f16 decode, llama's 6-bit K-scale
/// unpack, the codebook when the kind has one, and the kind's own
/// `frink_dequant_sub` -- followed by `body`. Shared by the SIMT unit
/// above and the tensor-core unit in [`crate::mul_mm_tc`], so the two
/// bodies dequantize with the same functions by construction. `FX_SUB`,
/// `FX_NL` and `FX_BLOCK_BYTES`, which every dequant function reads,
/// are defined here for the same reason.
pub fn kernel_src_with_body(kind: &MulMmKind, body: &str) -> String {
    let defines = format!(
        "#define FX_SUB {}\n\
         #define FX_NL {}\n\
         #define FX_BLOCK_BYTES {}\n",
        SUB,
        kind.nl(),
        kind.block_bytes,
    );
    // The codebook comes from the kind's own row, so this is not a
    // second table that has to agree with `KINDS` -- it is `KINDS`.
    let codebook = kind.codebook.as_ref().map(codebook_src).unwrap_or_default();
    // Q4_K and Q5_K share llama's 6-bit scale/min unpack. It is emitted
    // for every kind rather than conditionally: an unused `__device__`
    // helper costs nothing after NVRTC's dead-code pass, and a
    // per-kind include list is one more table that has to agree with
    // another one.
    // The trit helper TQ1_0 and PTQ1_0 read is emitted the same way.
    let trit = crate::mul_mm_kinds::ternary::TRIT_SRC;
    // The grids too come from the kind's own row.
    let grids = crate::iq_grids::grids_src(kind.grids);
    format!(
        "{defines}{F16_SRC}{K_SCALE_MIN_SRC}{trit}{codebook}{grids}{}{body}",
        kind.dequant_src
    )
}

/// Why a `mul_mm` dispatch was refused. Named rather than silent: a
/// shape this kernel cannot do must fall back to a path that can, and
/// the caller has to be able to say which.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MulMmUnsupported {
    /// `n_cols` is not a multiple of the K-tile. Every real GGUF row
    /// length is a multiple of 32, so this means a synthetic shape.
    ColsNotTileAligned { n_cols: usize, tile: usize },
    /// `n_cols` is not a whole number of super-blocks for this kind.
    ColsNotBlockAligned {
        n_cols: usize,
        block_elems: usize,
        kind: &'static str,
    },
    /// `row_bytes` does not match `n_cols` worth of super-blocks.
    RowBytesMismatch {
        row_bytes: usize,
        expected: usize,
        kind: &'static str,
    },
    /// The weight buffer is not `n_rows * row_bytes`.
    WeightsTooSmall { got: usize, want: usize },
    /// The activation buffer is not `batch * n_cols`.
    ActivationsTooSmall { got: usize, want: usize },
    /// A zero-sized dispatch. Not an error the caller must handle
    /// specially, but not something to launch a grid for either.
    EmptyShape,
}

impl std::fmt::Display for MulMmUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColsNotTileAligned { n_cols, tile } => {
                write!(f, "mul_mm: n_cols {n_cols} is not a multiple of the K-tile {tile}")
            }
            Self::ColsNotBlockAligned {
                n_cols,
                block_elems,
                kind,
            } => write!(
                f,
                "mul_mm: n_cols {n_cols} is not a whole number of {kind} blocks ({block_elems} elems)"
            ),
            Self::RowBytesMismatch {
                row_bytes,
                expected,
                kind,
            } => write!(
                f,
                "mul_mm: row_bytes {row_bytes} does not match {expected} for {kind} at this n_cols"
            ),
            Self::WeightsTooSmall { got, want } => {
                write!(f, "mul_mm: weight buffer is {got} bytes, needs {want}")
            }
            Self::ActivationsTooSmall { got, want } => {
                write!(f, "mul_mm: activation buffer is {got} floats, needs {want}")
            }
            Self::EmptyShape => write!(f, "mul_mm: empty shape"),
        }
    }
}

impl std::error::Error for MulMmUnsupported {}

/// The shape checks the kernel's index arithmetic assumes, in one place
/// so the launch path and the scalar twin cannot check different things.
pub fn validate_shape(
    kind: &MulMmKind,
    weights_len: usize,
    x_len: usize,
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
) -> Result<(), MulMmUnsupported> {
    if n_rows == 0 || n_cols == 0 || batch == 0 {
        return Err(MulMmUnsupported::EmptyShape);
    }
    if !n_cols.is_multiple_of(BK) {
        return Err(MulMmUnsupported::ColsNotTileAligned { n_cols, tile: BK });
    }
    if !n_cols.is_multiple_of(kind.block_elems) {
        return Err(MulMmUnsupported::ColsNotBlockAligned {
            n_cols,
            block_elems: kind.block_elems,
            kind: kind.name,
        });
    }
    let expected_row_bytes = (n_cols / kind.block_elems) * kind.block_bytes;
    if row_bytes != expected_row_bytes {
        return Err(MulMmUnsupported::RowBytesMismatch {
            row_bytes,
            expected: expected_row_bytes,
            kind: kind.name,
        });
    }
    let want_weights = n_rows * row_bytes;
    if weights_len < want_weights {
        return Err(MulMmUnsupported::WeightsTooSmall {
            got: weights_len,
            want: want_weights,
        });
    }
    let want_x = batch * n_cols;
    if x_len < want_x {
        return Err(MulMmUnsupported::ActivationsTooSmall {
            got: x_len,
            want: want_x,
        });
    }
    Ok(())
}

/// Whether a batched dispatch of this shape is worth a GEMM at all.
///
/// One token is a matvec, and `gpu.rs`'s matvec kernels are the arm that
/// has actually run on hardware; sending a single row through the tile
/// would waste every output column but one. The caller should
/// keep using `apply_gpu` below this threshold.
pub fn worth_a_gemm(batch: usize) -> bool {
    // `.max(2)` so this stays "never a single token" even if the tile
    // width is retuned downward.
    batch >= (BN / 4).max(2)
}

/// Grid dimensions for a dispatch, shared by the launch path and the
/// twin's block loop so they enumerate exactly the same tiles.
pub fn grid_dims(n_rows: usize, batch: usize) -> (usize, usize) {
    (batch.div_ceil(BN), n_rows.div_ceil(BM))
}

#[cfg(test)]
mod dequant_twin_tests {
    use super::*;

    /// Every twin, against the CPU dequant this project already holds
    /// against llama.cpp.
    ///
    /// This is the only oracle available without a GPU, and it is a real
    /// one: `frink_quant::dequant_*` decodes a whole super-block, so
    /// sub-block `il` of the kernel must equal elements `[16*il,
    /// 16*il+16)` of it. A transcription that mixes up llama's three
    /// different uses of `il` produces plausible numbers from the wrong
    /// offsets, and that is exactly what this catches.
    ///
    /// The case list is checked to COVER [`KINDS`]: a kind added to the
    /// table with no `frink_quant` dequant beside it gets a kernel, a
    /// twin, and no evidence that either decodes the format. The scale
    /// pinning comes from [`crate::mul_mm_ref::fixtures`], the one
    /// place that knows where each format keeps its scales.
    #[test]
    fn every_dequant_twin_matches_the_cpu_dequant() {
        type Dequant = fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError>;
        let cases: &[(&MulMmKind, Dequant)] = &[
            (&Q8_0, frink_quant::dequant_q8_0),
            (&Q4_0, frink_quant::dequant_q4_0),
            (&Q5_0, frink_quant::dequant_q5_0),
            (&Q4_K, frink_quant::dequant_q4_k),
            (&Q5_K, frink_quant::dequant_q5_k),
            (&Q2_K, frink_quant::dequant_q2_k),
            (&Q3_K, frink_quant::dequant_q3_k),
            (&Q6_K, frink_quant::dequant_q6_k),
            (&IQ4_NL, frink_quant::dequant_iq4_nl),
            (&IQ4_XS, frink_quant::dequant_iq4_xs),
            (&MXFP4, frink_quant::dequant_mxfp4_gguf),
            (&Q4_1, frink_quant::dequant_q4_1),
            (&Q5_1, frink_quant::dequant_q5_1),
            (&TQ1_0, |b| {
                frink_quant::ternary::dequant_trits(b, frink_quant::ternary::TQ1_0)
            }),
            (&PTQ1_0, |b| {
                frink_quant::ternary::dequant_trits(b, frink_quant::ternary::PTQ1_0)
            }),
            (&IQ2_XXS, frink_quant::dequant_iq2_xxs),
            (&IQ2_XS, frink_quant::dequant_iq2_xs),
            (&IQ2_S, frink_quant::dequant_iq2_s),
            (&IQ3_XXS, frink_quant::dequant_iq3_xxs),
            (&IQ3_S, frink_quant::dequant_iq3_s),
            (&IQ1_S, frink_quant::dequant_iq1_s),
            (&IQ1_M, frink_quant::dequant_iq1_m),
        ];

        for k in KINDS {
            assert!(
                cases.iter().any(|(c, _)| c.name == k.name),
                "{}: in KINDS with no dequant-twin case",
                k.name
            );
        }

        for (k, dequant) in cases {
            for seed in [1u32, 7, 12345] {
                let block = crate::mul_mm_ref::fixtures::block(k, seed);
                let want = dequant(&block).expect("cpu dequant");
                assert_eq!(want.len(), k.block_elems, "{} block size", k.name);

                for il in 0..k.nl() {
                    let mut reg = [0f32; SUB];
                    (k.dequant_twin)(&block, il, &mut reg);
                    for (j, got) in reg.iter().enumerate() {
                        let expect = want[SUB * il + j];
                        let tol = expect.abs().max(1.0) * 1e-5;
                        assert!(
                            (got - expect).abs() <= tol,
                            "{} seed {seed} sub-block {il} element {j}: \
                             kernel twin {got} vs cpu dequant {expect}",
                            k.name
                        );
                    }
                }
            }
        }
    }

    /// The block geometry each kind declares has to be the geometry the
    /// format actually has, or the GEMM walks the row with the wrong
    /// stride and every number after the first block is garbage.
    ///
    /// Driven by [`KINDS`] and answered from `frink_quant`'s
    /// constants, so neither half is a hand-written list: a row added
    /// to the table with no `frink_quant` geometry beside it fails
    /// here, and a row whose geometry is a literal that drifted from
    /// the format fails here too.
    ///
    /// `nl` is part of that geometry and is NOT the same for every row.
    /// It falls out of `block_elems / SUB`, so the assertion is that
    /// `block_elems` is a whole number of sub-blocks rather than a
    /// restated 2 or 16 -- the K-quants are 256-element super-blocks
    /// (`nl` 16) and the legacy and codebook 32-element kinds are `nl`
    /// 2, and inheriting the wrong one asks the kernel for sub-blocks
    /// 2..16 of a block that has two.
    #[test]
    fn declared_block_geometry_is_the_gguf_geometry() {
        let geometry: &[(&str, usize, usize)] = &[
            (
                "Q8_0",
                frink_quant::Q8_0_BLOCK_BYTES,
                frink_quant::Q8_0_BLOCK_ELEMS,
            ),
            (
                "Q4_0",
                frink_quant::Q4_0_BLOCK_BYTES,
                frink_quant::Q4_0_BLOCK_ELEMS,
            ),
            (
                "Q5_0",
                frink_quant::Q5_0_BLOCK_BYTES,
                frink_quant::Q5_0_BLOCK_ELEMS,
            ),
            (
                "Q4_K",
                frink_quant::Q4_K_BLOCK_BYTES,
                frink_quant::Q4_K_BLOCK_ELEMS,
            ),
            (
                "Q5_K",
                frink_quant::Q5_K_BLOCK_BYTES,
                frink_quant::Q5_K_BLOCK_ELEMS,
            ),
            (
                "Q2_K",
                frink_quant::Q2_K_BLOCK_BYTES,
                frink_quant::Q2_K_BLOCK_ELEMS,
            ),
            (
                "Q3_K",
                frink_quant::Q3_K_BLOCK_BYTES,
                frink_quant::Q3_K_BLOCK_ELEMS,
            ),
            (
                "Q6_K",
                frink_quant::Q6_K_BLOCK_BYTES,
                frink_quant::Q6_K_BLOCK_ELEMS,
            ),
            (
                "IQ4_NL",
                frink_quant::IQ4_NL_BLOCK_BYTES,
                frink_quant::IQ4_NL_BLOCK_ELEMS,
            ),
            (
                "IQ4_XS",
                frink_quant::IQ4_XS_BLOCK_BYTES,
                frink_quant::IQ4_XS_BLOCK_ELEMS,
            ),
            (
                "MXFP4",
                frink_quant::MXFP4_GGUF_BLOCK_BYTES,
                frink_quant::MXFP4_GGUF_BLOCK_ELEMS,
            ),
            ("Q4_1", frink_quant::Q4_1_BLOCK_BYTES, 32),
            ("Q5_1", frink_quant::Q5_1_BLOCK_BYTES, 32),
            (
                "TQ1_0",
                frink_quant::ternary::TQ1_0_BLOCK_BYTES,
                frink_quant::ternary::TQ1_0_BLOCK_ELEMS,
            ),
            (
                "PTQ1_0",
                frink_quant::ternary::PTQ1_0_BLOCK_BYTES,
                frink_quant::ternary::PTQ1_0_BLOCK_ELEMS,
            ),
            (
                "IQ2_XXS",
                frink_quant::IQ2_XXS_BLOCK_BYTES,
                frink_quant::IQ2_XXS_BLOCK_ELEMS,
            ),
            (
                "IQ2_XS",
                frink_quant::IQ2_XS_BLOCK_BYTES,
                frink_quant::IQ2_XS_BLOCK_ELEMS,
            ),
            (
                "IQ2_S",
                frink_quant::IQ2_S_BLOCK_BYTES,
                frink_quant::IQ2_S_BLOCK_ELEMS,
            ),
            (
                "IQ3_XXS",
                frink_quant::IQ3_XXS_BLOCK_BYTES,
                frink_quant::IQ3_XXS_BLOCK_ELEMS,
            ),
            (
                "IQ3_S",
                frink_quant::IQ3_S_BLOCK_BYTES,
                frink_quant::IQ3_S_BLOCK_ELEMS,
            ),
            (
                "IQ1_S",
                frink_quant::IQ1_S_BLOCK_BYTES,
                frink_quant::IQ1_S_BLOCK_ELEMS,
            ),
            (
                "IQ1_M",
                frink_quant::IQ1_M_BLOCK_BYTES,
                frink_quant::IQ1_M_BLOCK_ELEMS,
            ),
        ];

        for k in KINDS {
            let (_, bytes_, elems) = geometry
                .iter()
                .find(|(name, _, _)| *name == k.name)
                .unwrap_or_else(|| panic!("{}: in KINDS with no frink_quant geometry", k.name));
            assert_eq!(k.block_bytes, *bytes_, "{} block_bytes", k.name);
            assert_eq!(k.block_elems, *elems, "{} block_elems", k.name);
            assert_eq!(
                k.block_elems,
                k.nl() * SUB,
                "{}: nl() must partition the super-block into {SUB}-element sub-blocks",
                k.name
            );
            assert_eq!(
                kind_by_name(k.name).map(|f| f.name),
                Some(k.name),
                "{}: does not resolve by its own name",
                k.name
            );
        }

        // Two kinds must not collide in the process-wide NVRTC module
        // cache, and two must not share an entry point.
        for (i, a) in KINDS.iter().enumerate() {
            for b in &KINDS[i + 1..] {
                assert_ne!(a.module_name, b.module_name, "{} vs {}", a.name, b.name);
                assert_ne!(a.fn_name, b.fn_name, "{} vs {}", a.name, b.name);
            }
        }

        // A kind with no kernel must not resolve. Resolving would send
        // a GEMM to a module that cannot compile, and the caller would
        // have no way to fall back honestly.
        for absent in ["Q8_1", "F16", "BF16"] {
            assert!(
                kind_by_name(absent).is_none(),
                "{absent} resolved to a mul_mm kernel that does not exist"
            );
        }
    }

    /// Every codebook row's `dequant_src` has to actually spell the
    /// `__constant__` array its [`Codebook`] declares, and the emitted
    /// array has to carry that row's values -- parsed back out of the
    /// text rather than compared to a second copy of the formatting.
    ///
    /// This is what makes "one slice serves both halves" true rather
    /// than intended: the numbers are read out of the generated CUDA C
    /// and held against the slice the Rust twin indexes.
    ///
    /// Sabotage: change one entry of `KVALUES_IQ4NL` and this names the
    /// index; rename `c_name` without renaming it in `dequant_src` and
    /// the first assertion fires.
    #[test]
    fn an_emitted_codebook_is_the_slice_the_twin_indexes() {
        let mut seen = 0usize;
        for k in KINDS {
            let Some(cb) = k.codebook else {
                assert!(
                    !kernel_src(k).contains("__constant__"),
                    "{}: no codebook declared but one is emitted",
                    k.name
                );
                continue;
            };
            seen += 1;
            let src = kernel_src(k);
            assert!(
                k.dequant_src.contains(cb.c_name),
                "{}: dequant_src never indexes {}",
                k.name,
                cb.c_name
            );
            let decl = format!("__constant__ float {}[16] = {{", cb.c_name);
            let at = src
                .find(&decl)
                .unwrap_or_else(|| panic!("{}: {} is not emitted", k.name, cb.c_name));
            let body = &src[at + decl.len()..];
            let body = &body[..body.find('}').expect("unterminated codebook")];
            let got: Vec<f32> = body
                .split(',')
                .map(|t| {
                    t.trim()
                        .trim_end_matches('f')
                        .parse::<f32>()
                        .unwrap_or_else(|e| panic!("{}: {t:?}: {e}", k.name))
                })
                .collect();
            assert_eq!(got.len(), 16, "{}: codebook is not 16 entries", k.name);
            for (i, (g, w)) in got.iter().zip(cb.values.iter()).enumerate() {
                // Bit comparison, not `==`: MXFP4's code 8 is negative
                // zero, and `-0.0 == 0.0` would let it through.
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{}: codebook entry {i}: emitted {g}, twin indexes {w}",
                    k.name
                );
            }
        }
        assert!(seen >= 3, "the codebook kinds stopped declaring codebooks");
    }
}
