//! The integer-dot decode matvecs: llama.cpp's `mmvq` (#133).
//!
//! The activation is quantized ONCE to `block_q8_1` (32 int8 values, an
//! f16 scale `d = amax / 127` and the f16 of the f32 sum of the values;
//! `quantize.cu:54-99`), and each weight row is dotted against it four
//! int8 pairs at a time with `dp4a`, converting to float once per
//! block (`vecdotq.cuh`'s `vec_dot_*_q8_1`). Against the coalesced f32
//! kernels in [`super::coalesced`] that cuts the activation bytes a row
//! reads by 32 * 4 / 36 = 3.6x and the instructions per weight by three
//! to four. It is also what llama.cpp's CUDA decode computes, so it
//! moves frink toward the CUDA oracle (ADR 0009); the f32 kernels stay
//! the default until a hardware receipt says otherwise
//! (`FRINK_CUDA_MATVEC=dp4a`, read once, selects these).
//!
//! # Geometry, llama.cpp's for one column on NVIDIA
//!
//! `FQ_NWARPS` warps per threadblock (`calc_nwarps`): 4, or 2 for the
//! K-quants on the Volta/Turing table. Thread `tid` takes weight blocks
//! `tid / (qi / vdr)`, stepping `FQ_NWARPS * 32 / (qi / vdr)`, and quant
//! word `vdr * (tid % (qi / vdr))` inside each: 16 threads per Q4_K or
//! Q6_K super-block (8 bytes of quants each), 4 per Q8_0 block, 2 per
//! MXFP4 block. The warps' partial sums meet in shared memory, warp 0
//! adds them in warp order and reduces with the xor butterfly
//! (`warp_reduce_sum`). `FQ_ROWS` rows per threadblock is 1, or
//! `FQ_NWARPS` when the row is too short for one row to keep the block
//! busy (`should_use_small_k`): every thread then dots the same
//! activation block against `FQ_ROWS` rows.
//!
//! Text only, like the coalesced kernels: the launch lives in
//! `dp4a_launch.rs`, the scalar twins in [`crate::dp4a_twin`], and
//! `tools/matvec_host_check/run.sh` runs this text on a host CPU
//! against the twins bit for bit.

use super::coalesced::MATVEC_WIDTH_ENV;
use super::helpers::BLOCK_HELPERS;

/// Values per `block_q8_1` (`QK8_1`).
pub const Q8_1_BLOCK_ELEMS: usize = 32;
/// Bytes per `block_q8_1`: `half2 ds` then 32 int8 quants.
pub const Q8_1_BLOCK_BYTES: usize = 36;
/// Threads per quantizer threadblock (`CUDA_QUANTIZE_BLOCK_SIZE`).
pub const QUANTIZE_THREADS: usize = 256;
/// The quantizer's entry point and NVRTC module key.
pub const QUANTIZE_ENTRY: &str = "quantize_q8_1";
pub const QUANTIZE_MODULE: &str = "frink_quantize_q8_1";

/// The value of [`MATVEC_WIDTH_ENV`] that selects these kernels.
pub const DP4A_VALUE: &str = "dp4a";

/// Whether `FRINK_CUDA_MATVEC`'s value selects the dp4a matvecs:
/// `dp4a`, any case. Everything else keeps the f32 coalesced kernels,
/// whose own reading of the variable (`narrow` or not) is unchanged.
pub fn parse_dp4a(value: Option<&str>) -> bool {
    matches!(value, Some(v) if v.trim().eq_ignore_ascii_case(DP4A_VALUE))
}

/// [`parse_dp4a`] of this process's environment, read once: a matvec
/// runs a few hundred times per decoded token.
pub fn dp4a_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_dp4a(std::env::var(MATVEC_WIDTH_ENV).ok().as_deref()))
}

/// The NVRTC target a dp4a module compiles for on a device of compute
/// capability `cc`: the device's own, so `__CUDA_ARCH__` admits the
/// `dp4a` instruction (`sm_61` and up; NVRTC's default target is
/// older). `None` below `sm_61`, where the kernels' scalar integer dot
/// computes the same integers, slower.
pub fn nvrtc_arch(cc: (i32, i32)) -> Option<&'static str> {
    if cc < (6, 1) {
        return None;
    }
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<HashMap<(i32, i32), &'static str>>> = OnceLock::new();
    Some(
        NAMES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(cc)
            .or_insert_with(|| Box::leak(format!("compute_{}{}", cc.0, cc.1).into_boxed_str())),
    )
}

/// Warps per threadblock and rows per threadblock of one dp4a launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dp4aGeometry {
    /// `FQ_NWARPS`: 1, 2 or 4.
    pub nwarps: usize,
    /// `FQ_ROWS`: 1 or `nwarps`.
    pub rows: usize,
}

impl Dp4aGeometry {
    pub fn threads(&self) -> usize {
        32 * self.nwarps
    }

    /// Threadblocks along x for a matrix of `rows` rows.
    pub fn grid_x(&self, rows: usize) -> usize {
        rows.div_ceil(self.rows)
    }
}

/// One dp4a kernel: what routes to it, its text, and the block shape
/// constants llama.cpp's `ggml_cuda_type_traits` and `get_vdr_mmvq`
/// give its kind. The kernel text restates the four numbers in its
/// trait struct; `the_kernel_traits_agree_with_the_table` holds the two
/// together.
#[derive(Debug, Clone, Copy)]
pub struct Dp4aKind {
    /// The `MatvecKind::fn_name` this kernel replaces under the switch.
    pub base_fn: &'static str,
    /// The `__global__` entry point.
    pub entry: &'static str,
    /// The kernel text, without the defines and the preludes.
    pub src: &'static str,
    /// NVRTC module key stem, suffixed per geometry.
    pub module: &'static str,
    /// Weights per block (`qk`).
    pub qk: usize,
    /// 32-bit words of quants per block (`qi`).
    pub qi: usize,
    /// Words one thread takes from a block (`vdr`).
    pub vdr: usize,
    /// Bytes per block.
    pub block_bytes: usize,
}

impl Dp4aKind {
    /// Threads sharing one weight block (`qi / vdr`).
    pub fn threads_per_block(&self) -> usize {
        self.qi / self.vdr
    }

    /// Weight blocks one warp covers per loop trip
    /// (`blocks_per_iter_1warp`).
    pub fn blocks_per_warp_trip(&self) -> usize {
        self.vdr * 32 / self.qi
    }

    /// `q8_1` blocks per weight block.
    pub fn q8_per_block(&self) -> usize {
        self.qk / Q8_1_BLOCK_ELEMS
    }

    /// The geometry llama.cpp launches for one column (`ncols_dst ==
    /// 1`) of a row of `n_blocks` blocks on an NVIDIA device of compute
    /// capability `cc`: `calc_nwarps` (the Volta/Turing table gives the
    /// K-quants two warps), then `should_use_small_k` and
    /// `calc_rows_per_block`. None of these four kinds is on its
    /// small-k exclusion lists or takes two rows otherwise.
    pub fn geometry(&self, cc: (i32, i32), n_blocks: usize) -> Dp4aGeometry {
        let turing_table = cc.0 == 7;
        let nwarps = if turing_table && self.qk == 256 { 2 } else { 4 };
        let small_k = nwarps > 1 && n_blocks < nwarps * self.blocks_per_warp_trip();
        Dp4aGeometry {
            nwarps,
            rows: if small_k { nwarps } else { 1 },
        }
    }

    /// The complete translation unit at `g`, in the routed-expert form
    /// (`FX_MULTI`: one slot per `blockIdx.y`, weights from a device
    /// pointer table, activations `y_stride` q8_1 blocks apart) when
    /// `multi`.
    pub fn source(&self, g: Dp4aGeometry, multi: bool) -> String {
        format!(
            "#define FQ_NWARPS {}\n#define FQ_ROWS {}\n{}{BLOCK_HELPERS}{Q8_1_COMMON}{DP4A_PRELUDE}{}",
            g.nwarps,
            g.rows,
            if multi { "#define FX_MULTI 1\n" } else { "" },
            self.src
        )
    }

    /// The NVRTC module key of [`Self::source`], leaked once per
    /// geometry so the load-once cache keys on `&'static str`.
    pub fn module_for(&self, g: Dp4aGeometry, multi: bool) -> &'static str {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};
        type Key = (&'static str, Dp4aGeometry, bool);
        static NAMES: OnceLock<Mutex<HashMap<Key, &'static str>>> = OnceLock::new();
        NAMES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry((self.module, g, multi))
            .or_insert_with(|| {
                let m = if multi { "_multi" } else { "" };
                let s = format!("{}_w{}_r{}{m}", self.module, g.nwarps, g.rows);
                Box::leak(s.into_boxed_str())
            })
    }
}

/// The dp4a kernels. The ONLY place one is named;
/// `every_dp4a_kernel_is_reachable_from_the_table` fails if a kernel is
/// added to this file and not here.
pub const DP4A: &[Dp4aKind] = &[
    Dp4aKind {
        base_fn: "q4_k_matvec",
        entry: "q4_k_matvec_dp4a",
        src: Q4_K_DP4A_SRC,
        module: "frink_q4_k_dp4a",
        qk: 256,
        qi: 32,
        vdr: 2,
        block_bytes: 144,
    },
    Dp4aKind {
        base_fn: "q6_k_matvec",
        entry: "q6_k_matvec_dp4a",
        src: Q6_K_DP4A_SRC,
        module: "frink_q6_k_dp4a",
        qk: 256,
        qi: 32,
        vdr: 2,
        block_bytes: 210,
    },
    Dp4aKind {
        base_fn: "q8_0_matvec",
        entry: "q8_0_matvec_dp4a",
        src: Q8_0_DP4A_SRC,
        module: "frink_q8_0_dp4a",
        qk: 32,
        qi: 8,
        vdr: 2,
        block_bytes: 34,
    },
    Dp4aKind {
        base_fn: "mxfp4_matvec",
        entry: "mxfp4_matvec_dp4a",
        src: MXFP4_DP4A_SRC,
        module: "frink_mxfp4_dp4a",
        qk: 32,
        qi: 4,
        vdr: 2,
        block_bytes: 17,
    },
];

/// The dp4a kernel replacing the block-per-row kernel `fn_name`, or
/// `None` if that kind has none and keeps the f32 kernels.
pub fn dp4a_kind(fn_name: &str) -> Option<&'static Dp4aKind> {
    DP4A.iter().find(|k| k.base_fn == fn_name)
}

/// The quantizer's complete translation unit.
pub fn quantize_source() -> String {
    format!("{BLOCK_HELPERS}{Q8_1_COMMON}{QUANTIZE_Q8_1_KERNEL_SRC}")
}

/// What the quantizer and every dp4a kernel share: the `block_q8_1`
/// layout, the integer dot, the f16 conversions and the word loads.
///
/// Each device-only instruction has a portable twin under `#else`,
/// which is what `tools/matvec_host_check` executes and what `hiprtc`
/// compiles: `dp4a.s32.s32` and four signed byte products; `cvt` to
/// and from f16 and the bit-level conversions (exact one way,
/// round-to-nearest-even the other, which is `__float2half`'s rule);
/// `prmt` with its sign-replicate bits cleared and an eight-byte
/// select. The hardware test is what holds each pair to agreeing.
pub const Q8_1_COMMON: &str = r#"
#define FQ_Q8_1_BYTES 36

__device__ __forceinline__ int fq_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    int r;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
#else
    return c + (int)(signed char)a * (int)(signed char)b
             + (int)(signed char)(a >> 8) * (int)(signed char)(b >> 8)
             + (int)(signed char)(a >> 16) * (int)(signed char)(b >> 16)
             + (int)(signed char)(a >> 24) * (int)(signed char)(b >> 24);
#endif
}

// The f16 at a 2-aligned address, widened (exactly).
__device__ __forceinline__ float fq_f16(const unsigned char* p) {
    const unsigned short h = *(const unsigned short*)p;
#if defined(__CUDA_ARCH__)
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
#else
    return fx_f16_to_f32(h);
#endif
}

// f32 to f16, round to nearest even: `__float2half`, so `make_half2`.
__device__ __forceinline__ unsigned short fq_f32_to_f16(float f) {
#if defined(__CUDA_ARCH__)
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
#else
    union { float f; unsigned int u; } c;
    c.f = f;
    const unsigned int sign = (c.u >> 16) & 0x8000u;
    const unsigned int ax = c.u & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) return (unsigned short)(sign | 0x7C00u | (ax > 0x7F800000u ? 0x0200u : 0u));
    if (ax >= 0x477FF000u) return (unsigned short)(sign | 0x7C00u);
    if (ax < 0x38800000u) {
        if (ax < 0x33000000u) return (unsigned short)sign;
        const unsigned int m = (ax & 0x7FFFFFu) | 0x800000u;
        const unsigned int shift = 126u - (ax >> 23);
        unsigned int k = m >> shift;
        const unsigned int rem = m & ((1u << shift) - 1u);
        const unsigned int half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (k & 1u))) k += 1u;
        return (unsigned short)(sign | k);
    }
    const unsigned int r = ax - 0x38000000u;
    return (unsigned short)(sign | ((r + 0x0FFFu + ((r >> 13) & 1u)) >> 13));
#endif
}

// 32 bits from a 1-, 2- or 4-aligned address (llama.cpp's get_int_b1/b2/b4).
__device__ __forceinline__ int fq_int_b1(const unsigned char* p) {
    return (int)((unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
                 | ((unsigned int)p[3] << 24));
}

__device__ __forceinline__ int fq_int_b2(const unsigned char* p) {
    const unsigned short* h = (const unsigned short*)p;
    return (int)((unsigned int)h[0] | ((unsigned int)h[1] << 16));
}

__device__ __forceinline__ int fq_int_b4(const unsigned char* p) {
    return *(const int*)p;
}

// Byte `s & 7` of the eight bytes {y:x} for each of the selector's four
// low nibbles: `__byte_perm`, which ignores each nibble's top bit.
__device__ __forceinline__ unsigned int fq_perm(unsigned int x, unsigned int y, unsigned int s) {
    s &= 0x7777u;
#if defined(__CUDA_ARCH__)
    unsigned int r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(y), "r"(s));
    return r;
#else
    const unsigned long long v = ((unsigned long long)y << 32) | (unsigned long long)x;
    unsigned int r = 0;
    for (int n = 0; n < 4; ++n) {
        r |= (unsigned int)((v >> (8u * ((s >> (4 * n)) & 7u))) & 0xFFu) << (8 * n);
    }
    return r;
#endif
}
"#;

/// The activation quantizer, `quantize_q8_1` (`quantize.cu:54-99`) for
/// one contiguous vector: one thread per value, a warp per block. The
/// warp's `amax` and f32 `sum` come from the xor butterfly
/// (`warp_reduce_max`, `warp_reduce_sum`), `d = amax / 127`, `q =
/// roundf(x / d)`, and `amax == 0` gives `d = 0` and every `q = 0` (the
/// division is skipped, as upstream skips it). Lane 0 stores `ds` as
/// two f16s: `make_half2(d, sum)`.
pub const QUANTIZE_Q8_1_KERNEL_SRC: &str = r#"
extern "C" __global__ void quantize_q8_1(
    const float* __restrict__ x,
    unsigned char* __restrict__ y,
    int n
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    // `n` is a multiple of 32 (the launcher's contract), so a warp is one
    // block_q8_1 and leaves whole: no lane waits on a partner that left.
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi);
    float sum = xi;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : (int)roundf(xi / d);
    unsigned char* b = y + (size_t)(i / 32) * FQ_Q8_1_BYTES;
    b[4 + i % 32] = (unsigned char)(signed char)q;
    if (i % 32 == 0) {
        *(unsigned int*)b = (unsigned int)fq_f32_to_f16(d)
                            | ((unsigned int)fq_f32_to_f16(sum) << 16);
    }
}
"#;

/// The row loop and the reduction every dp4a kernel shares, after
/// [`Q8_1_COMMON`]. A kind is a trait struct (`block_bytes`,
/// `q8_per_block`, `qi`, `vdr`, `dot`) handed to `fq_matvec`.
pub const DP4A_PRELUDE: &str = r#"
#ifndef FQ_NWARPS
#error "FQ_NWARPS (warps per threadblock) must be defined by the launcher"
#endif
#if FQ_NWARPS != 1 && FQ_NWARPS != 2 && FQ_NWARPS != 4
#error "FQ_NWARPS must be 1, 2 or 4"
#endif
#ifndef FQ_ROWS
#error "FQ_ROWS (rows per threadblock) must be defined by the launcher"
#endif
#if FQ_ROWS != 1 && FQ_ROWS != FQ_NWARPS
#error "FQ_ROWS must be 1 or FQ_NWARPS"
#endif
#ifndef FX_MULTI
#define FX_MULTI 0
#endif
#define FQ_THREADS (FQ_NWARPS * 32)

#if FX_MULTI
// One routed expert per blockIdx.y: its weights from a table of device
// pointers, its q8_1 input `y_stride` blocks along, its own `rows` out.
#define FQ_SLOT_SETUP \
    const unsigned char* weights = \
        (const unsigned char*)((const unsigned long long*)wsrc)[blockIdx.y]; \
    const unsigned char* y = y_in + (size_t)blockIdx.y * (size_t)y_stride * FQ_Q8_1_BYTES; \
    float* out = out_in + (size_t)blockIdx.y * (size_t)rows;
#else
#define FQ_SLOT_SETUP \
    const unsigned char* weights = (const unsigned char*)wsrc; \
    const unsigned char* y = y_in; \
    float* out = out_in;
#endif

// mul_mat_vec_q's reduction: warps 1.. hand their partial sums to warp 0
// through shared memory, warp 0 adds them in warp order, reduces across
// its lanes with the xor butterfly (every lane ends with the same sum),
// and lane `i` stores row `row0 + i`. Every thread of the block calls it.
__device__ __forceinline__ void fq_store_rows(const float* tmp, float* out, int row0, int rows) {
    __shared__ float fq_red[FQ_NWARPS > 1 ? FQ_NWARPS - 1 : 1][FQ_ROWS][32];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp > 0) {
        #pragma unroll
        for (int i = 0; i < FQ_ROWS; ++i) fq_red[warp - 1][i][lane] = tmp[i];
    }
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int i = 0; i < FQ_ROWS; ++i) {
        float v = tmp[i];
        #pragma unroll
        for (int l = 0; l < FQ_NWARPS - 1; ++l) v += fq_red[l][i][lane];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == i && row0 + i < rows) out[row0 + i] = v;
    }
}

// mul_mat_vec_q for one column: thread `tid` takes weight blocks
// `tid / (qi/vdr)`, stepping FQ_THREADS / (qi/vdr), at quant word
// `vdr * (tid % (qi/vdr))`, for each of the threadblock's FQ_ROWS rows.
// A row past the matrix reads the last row and stores nothing.
template <class K>
__device__ __forceinline__ void fq_matvec(
    const unsigned char* weights, const unsigned char* y, float* out,
    int rows, int row_bytes, int n_blocks_per_row
) {
    const int tpb = K::qi / K::vdr;
    const int tid = threadIdx.x;
    const int row0 = (int)blockIdx.x * FQ_ROWS;
    const int kqs = K::vdr * (tid % tpb);
    float tmp[FQ_ROWS];
    #pragma unroll
    for (int i = 0; i < FQ_ROWS; ++i) tmp[i] = 0.0f;
    for (int kbx = tid / tpb; kbx < n_blocks_per_row; kbx += FQ_THREADS / tpb) {
        const unsigned char* yb = y + (size_t)kbx * K::q8_per_block * FQ_Q8_1_BYTES;
        #pragma unroll
        for (int i = 0; i < FQ_ROWS; ++i) {
            const int r = row0 + i < rows ? row0 + i : rows - 1;
            tmp[i] += K::dot(weights + (size_t)r * row_bytes + (size_t)kbx * K::block_bytes, yb, kqs);
        }
    }
    fq_store_rows(tmp, out, row0, rows);
}
"#;

/// Q4_K against q8_1: `vec_dot_q4_K_q8_1` and its `_impl_vmmq`. Thread
/// word `iqs` (even, 0..30) owns `qs` bytes `32*g + 4*r ..+4` and
/// `+16 ..+4`, `g = iqs/8`, `r = (iqs/2) % 4`: low nibbles are sub-block
/// `2g`'s elements, high nibbles `2g + 1`'s, against q8_1 blocks `2g`
/// and `2g + 1`. The min term multiplies `m` by the sum of the INTEGER
/// q8 values (a dp4a against ones), as upstream's mmvq does, not by the
/// stored f32 sum. The scales are unpacked branch-free from three u16
/// words, as upstream does it; the twin unpacks them with
/// `q4_k_scale_min`, so the host check holds the two together.
pub const Q4_K_DP4A_SRC: &str = r#"
struct fq_q4_k {
    static const int block_bytes = 144;
    static const int q8_per_block = 8;
    static const int qi = 32;
    static const int vdr = 2;

    static __device__ __forceinline__ float dot(const unsigned char* blk, const unsigned char* y, int iqs) {
        const int bq8_offset = 2 * ((iqs / 2) / 4);
        const int r = (iqs / 2) % 4;
        const unsigned char* q4 = blk + 16 + 16 * bq8_offset + 4 * r;
        const int v0 = fq_int_b4(q4);
        const int v1 = fq_int_b4(q4 + 16);

        // Sub-blocks 2j and 2j + 1: their 6-bit scales and mins.
        const unsigned short* s16 = (const unsigned short*)(blk + 4);
        const int j = bq8_offset / 2;
        const int jm = j & 1;
        const unsigned int s0 = s16[jm + 0];
        const unsigned int s2 = s16[jm + 2];
        const unsigned int s4 = s16[jm + 4];
        const unsigned int hi = (unsigned int)-(int)(j >= 2);
        const unsigned int a0 = ((s0 & 0x3f3fu) & ~hi) | ((((s4 >> 0) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2)) & hi);
        const unsigned int a1 = ((s2 & 0x3f3fu) & ~hi) | ((((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2)) & hi);
        const int sc[2] = { (int)(a0 & 0xFFu), (int)((a0 >> 8) & 0xFFu) };
        const int m[2] = { (int)(a1 & 0xFFu), (int)((a1 >> 8) & 0xFFu) };

        float sumf_d = 0.0f;
        float sumf_m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned char* b8 = y + (bq8_offset + i) * FQ_Q8_1_BYTES;
            const float d8 = fq_f16(b8);
            const int u0 = fq_int_b4(b8 + 4 + 4 * r);
            const int u1 = fq_int_b4(b8 + 4 + 16 + 4 * r);
            const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
            const int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
            const int dot1 = fq_dp4a(v1i, u1, fq_dp4a(v0i, u0, 0));
            const int dot2 = fq_dp4a(0x01010101, u1, fq_dp4a(0x01010101, u0, 0));
            sumf_d += d8 * (float)(dot1 * sc[i]);
            sumf_m += d8 * (float)(dot2 * m[i]);
        }
        return fq_f16(blk) * sumf_d - fq_f16(blk + 2) * sumf_m;
    }
};

extern "C" __global__ void q4_k_matvec_dp4a(
    const void* __restrict__ wsrc,
    const unsigned char* __restrict__ y_in,
    float* out_in,
    int rows,
    int row_bytes,
    int n_blocks_per_row
#if FX_MULTI
    , int y_stride
#endif
) {
    FQ_SLOT_SETUP
    fq_matvec<fq_q4_k>(weights, y, out, rows, row_bytes, n_blocks_per_row);
}
"#;

/// Q6_K against q8_1: `vec_dot_q6_K_q8_1` and its `_impl_mmvq`, vdr 2.
/// Thread word `iqs` (even, 0..30) owns `ql` bytes `4*iqs ..+8`, which
/// sit in half `iqs / 16` of the block and in its `ql[l]` or `ql[l + 32]`
/// run as `(iqs % 16) / 8` says; the matching `qh` word is shifted by
/// `2 * ((iqs % 16) / 8)` so its two bits per weight land at bits 4..5
/// of each byte. The `- 32` is a second dp4a against `0xE0` bytes.
/// Q6_K blocks are 210 bytes, so only 2-byte loads are aligned.
pub const Q6_K_DP4A_SRC: &str = r#"
struct fq_q6_k {
    static const int block_bytes = 210;
    static const int q8_per_block = 8;
    static const int qi = 32;
    static const int vdr = 2;

    static __device__ __forceinline__ float dot(const unsigned char* blk, const unsigned char* y, int iqs) {
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        unsigned int vl[2], vh[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            vl[j] = (unsigned int)fq_int_b2(blk + 4 * (iqs + j));
            vh[j] = (unsigned int)fq_int_b2(blk + 128 + 4 * (8 * (iqs / 16) + iqs % 8 + j)) >> vh_shift;
        }
        const signed char* scales = (const signed char*)(blk + 192 + scale_offset);
        float sumf = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned char* b8 = y + (bq8_offset + 2 * i) * FQ_Q8_1_BYTES;
            const int sc = scales[4 * i];
            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int u = fq_int_b4(b8 + 4 + 4 * (iqs % 8 + j));
                const int vil = (int)((vl[j] >> (4 * i)) & 0x0F0F0F0Fu);
                const int vih = (int)(((vh[j] >> (4 * i)) << 4) & 0x30303030u);
                sumi = fq_dp4a(vil | vih, u, sumi);
                sumi = fq_dp4a((int)0xE0E0E0E0u, u, sumi);
            }
            sumf += fq_f16(b8) * (float)(sumi * sc);
        }
        return fq_f16(blk + 208) * sumf;
    }
};

extern "C" __global__ void q6_k_matvec_dp4a(
    const void* __restrict__ wsrc,
    const unsigned char* __restrict__ y_in,
    float* out_in,
    int rows,
    int row_bytes,
    int n_blocks_per_row
#if FX_MULTI
    , int y_stride
#endif
) {
    FQ_SLOT_SETUP
    fq_matvec<fq_q6_k>(weights, y, out, rows, row_bytes, n_blocks_per_row);
}
"#;

/// Q8_0 against q8_1: `vec_dot_q8_0_q8_1`, vdr 2. Thread word `iqs`
/// (0, 2, 4, 6) owns quants `4*iqs ..+8`; the block is 34 bytes, so
/// they are read as `u16` pairs. Result `(d_w * d_x) * sumi`, in that
/// order.
pub const Q8_0_DP4A_SRC: &str = r#"
struct fq_q8_0 {
    static const int block_bytes = 34;
    static const int q8_per_block = 1;
    static const int qi = 8;
    static const int vdr = 2;

    static __device__ __forceinline__ float dot(const unsigned char* blk, const unsigned char* y, int iqs) {
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = fq_dp4a(fq_int_b2(blk + 2 + 4 * (iqs + i)), fq_int_b4(y + 4 + 4 * (iqs + i)), sumi);
        }
        return fq_f16(blk) * fq_f16(y) * (float)sumi;
    }
};

extern "C" __global__ void q8_0_matvec_dp4a(
    const void* __restrict__ wsrc,
    const unsigned char* __restrict__ y_in,
    float* out_in,
    int rows,
    int row_bytes,
    int n_blocks_per_row
#if FX_MULTI
    , int y_stride
#endif
) {
    FQ_SLOT_SETUP
    fq_matvec<fq_q8_0>(weights, y, out, rows, row_bytes, n_blocks_per_row);
}
"#;

/// MXFP4 against q8_1: `vec_dot_mxfp4_q8_1`, vdr 2, two threads per
/// 17-byte block. Thread word `iqs` (0 or 2) owns code bytes `4*iqs
/// ..+8`; each word's eight E2M1 codes become ggml's `kvalues_mxfp4`
/// (E2M1 doubled, so int8) through `get_int_from_table_16`'s byte
/// permutes, the table held in four immediates. Low nibbles pair with
/// q8 values `4*(iqs+l)..`, high ones with the same plus 16. Scale
/// `(e8m0 * 0.5) * d_x`, upstream's order.
pub const MXFP4_DP4A_SRC: &str = r#"
// kvalues_mxfp4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}.
__device__ __forceinline__ void fq_mxfp4_lookup(int q4, int* lo, int* hi) {
    const unsigned int t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const unsigned int q = (unsigned int)q4;
    const unsigned int sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    unsigned int tmp[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const unsigned int shift = 16u * (unsigned int)i;
        const unsigned int low = fq_perm(t0, t1, q >> shift);
        const unsigned int high = fq_perm(t2, t3, q >> shift);
        tmp[i] = fq_perm(low, high, sel >> shift);
    }
    *lo = (int)fq_perm(tmp[0], tmp[1], 0x6420u);
    *hi = (int)fq_perm(tmp[0], tmp[1], 0x7531u);
}

struct fq_mxfp4 {
    static const int block_bytes = 17;
    static const int q8_per_block = 1;
    static const int qi = 4;
    static const int vdr = 2;

    static __device__ __forceinline__ float dot(const unsigned char* blk, const unsigned char* y, int iqs) {
        int sumi = 0;
        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            int lo, hi;
            fq_mxfp4_lookup(fq_int_b1(blk + 1 + 4 * (iqs + l)), &lo, &hi);
            sumi = fq_dp4a(lo, fq_int_b4(y + 4 + 4 * (iqs + l)), sumi);
            sumi = fq_dp4a(hi, fq_int_b4(y + 4 + 4 * (iqs + l + 4)), sumi);
        }
        // ggml_cuda_e8m0_to_fp32: the byte IS an f32 exponent field, 2^-127 for 0.
        const unsigned int e = blk[0];
        const float e8m0 = e == 0u ? __int_as_float(0x00400000) : __int_as_float((int)(e << 23));
        const float d = e8m0 * 0.5f * fq_f16(y);
        return d * (float)sumi;
    }
};

extern "C" __global__ void mxfp4_matvec_dp4a(
    const void* __restrict__ wsrc,
    const unsigned char* __restrict__ y_in,
    float* out_in,
    int rows,
    int row_bytes,
    int n_blocks_per_row
#if FX_MULTI
    , int y_stride
#endif
) {
    FQ_SLOT_SETUP
    fq_matvec<fq_mxfp4>(weights, y, out, rows, row_bytes, n_blocks_per_row);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// A dp4a kernel the table does not name is dead code that reads as
    /// coverage; the entry points are read off this file.
    #[test]
    fn every_dp4a_kernel_is_reachable_from_the_table() {
        let src = include_str!("dp4a.rs");
        let mut found = 0usize;
        for line in src.lines() {
            let Some(rest) = line.strip_prefix(r#"extern "C" __global__ void "#) else {
                continue;
            };
            let entry = rest.split('(').next().unwrap_or("");
            let Some(base) = entry.strip_suffix("_dp4a") else {
                assert_eq!(entry, QUANTIZE_ENTRY, "an unrouted kernel: {entry}");
                continue;
            };
            found += 1;
            let k = dp4a_kind(base).unwrap_or_else(|| panic!("{entry}: no table row for {base}"));
            assert_eq!(k.entry, entry);
            assert!(
                k.src.contains(&format!("void {entry}(")),
                "{entry}: wrong source"
            );
        }
        assert_eq!(found, DP4A.len());
        assert!(QUANTIZE_Q8_1_KERNEL_SRC.contains(&format!("void {QUANTIZE_ENTRY}(")));
    }

    /// The four shape numbers live twice, in the table (the launch and
    /// the twins read them) and in each kernel's trait struct (the
    /// kernel reads them). Read off the text, so the two cannot differ.
    /// Each row's block shape is also the GEMM table's for the kind.
    ///
    /// Sabotage: `block_bytes = 210` in the Q6_K struct to 211 fails here.
    #[test]
    fn the_kernel_traits_agree_with_the_table() {
        for k in DP4A {
            let field = |name: &str| -> usize {
                let pat = format!("static const int {name} = ");
                let at = k
                    .src
                    .find(&pat)
                    .unwrap_or_else(|| panic!("{}: no {name}", k.entry));
                k.src[at + pat.len()..]
                    .split(';')
                    .next()
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or_else(|| panic!("{}: {name} is not a number", k.entry))
            };
            assert_eq!(field("block_bytes"), k.block_bytes, "{}", k.entry);
            assert_eq!(field("q8_per_block"), k.q8_per_block(), "{}", k.entry);
            assert_eq!(field("qi"), k.qi, "{}", k.entry);
            assert_eq!(field("vdr"), k.vdr, "{}", k.entry);
            let mv = crate::matvec_kinds::KINDS
                .iter()
                .find(|m| m.fn_name == k.base_fn)
                .unwrap_or_else(|| panic!("{}: routes from {}, no MatvecKind", k.entry, k.base_fn));
            let mm = crate::mul_mm::kind_by_name(mv.name).expect("block geometry");
            assert_eq!(mm.block_bytes, k.block_bytes, "{}", mv.name);
            assert_eq!(mm.block_elems, k.qk, "{}", mv.name);
            // qi words of 4 bytes cover the block's quants at QR values a byte.
            assert!(k.qi.is_multiple_of(k.vdr) && 32 % k.threads_per_block() == 0);
        }
    }

    /// llama.cpp's launch for one column, on the decode shapes that
    /// matter: Llama-3.1-8B (K 4096 and 14336), a 1536-wide model where
    /// small-k engages for the K-quants, gpt-oss-20b's 2880, and the
    /// Turing table's two warps for the K-quants only.
    #[test]
    fn the_geometry_is_llama_cpp_s_for_one_column() {
        let k = |name: &str| dp4a_kind(name).expect(name);
        let ada = (8, 9);
        let g = |w, r| Dp4aGeometry { nwarps: w, rows: r };
        assert_eq!(k("q4_k_matvec").geometry(ada, 16), g(4, 1), "Q4_K K=4096");
        assert_eq!(k("q4_k_matvec").geometry(ada, 8), g(4, 1), "Q4_K K=2048");
        assert_eq!(
            k("q4_k_matvec").geometry(ada, 6),
            g(4, 4),
            "Q4_K K=1536: small k"
        );
        assert_eq!(k("q6_k_matvec").geometry(ada, 56), g(4, 1), "Q6_K K=14336");
        assert_eq!(k("q8_0_matvec").geometry(ada, 128), g(4, 1), "Q8_0 K=4096");
        assert_eq!(k("q8_0_matvec").geometry(ada, 31), g(4, 4), "Q8_0 K=992");
        assert_eq!(k("q8_0_matvec").geometry(ada, 32), g(4, 1), "Q8_0 K=1024");
        assert_eq!(k("mxfp4_matvec").geometry(ada, 90), g(4, 1), "MXFP4 K=2880");
        assert_eq!(k("mxfp4_matvec").geometry(ada, 63), g(4, 4), "MXFP4 K=2016");
        let turing = (7, 5);
        assert_eq!(k("q4_k_matvec").geometry(turing, 16), g(2, 1));
        assert_eq!(k("q6_k_matvec").geometry(turing, 3), g(2, 2));
        assert_eq!(k("q8_0_matvec").geometry(turing, 128), g(4, 1));
        for kind in DP4A {
            assert_eq!(kind.threads_per_block() * kind.blocks_per_warp_trip(), 32);
        }
    }

    /// Every geometry the launch can pick compiles to a source that
    /// carries it, under a module key no other geometry or kind shares.
    #[test]
    fn every_geometry_has_its_own_source_and_module() {
        let mut keys = Vec::new();
        for k in DP4A {
            for (nwarps, rows) in [(4, 1), (4, 4), (2, 1), (2, 2)] {
                for multi in [false, true] {
                    let g = Dp4aGeometry { nwarps, rows };
                    let src = k.source(g, multi);
                    assert!(src.starts_with(&format!(
                        "#define FQ_NWARPS {nwarps}\n#define FQ_ROWS {rows}\n"
                    )));
                    assert_eq!(src.contains("#define FX_MULTI 1"), multi);
                    assert!(src.contains(BLOCK_HELPERS) && src.contains(k.src));
                    keys.push(k.module_for(g, multi));
                }
            }
        }
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), keys.len(), "duplicate module key in {keys:?}");
        assert!(!keys.contains(&QUANTIZE_MODULE));
        assert!(quantize_source().contains(QUANTIZE_Q8_1_KERNEL_SRC));
    }

    /// Every dp4a kind has a coalesced kernel too. The routed-expert
    /// paths admit a layer by `coalesced_kind(..).is_some()` and the
    /// f32 kernel is a dp4a kind's fallback for an input the quantizer
    /// cannot take, so a dp4a kind without one would lose both.
    #[test]
    fn every_dp4a_kind_has_a_coalesced_kernel() {
        for k in DP4A {
            let ck = crate::matvec_kinds::coalesced::coalesced_kind(k.base_fn)
                .unwrap_or_else(|| panic!("{} has no coalesced kernel", k.entry));
            assert_eq!(ck.entry.replace("_coalesced", "_dp4a"), k.entry);
        }
    }

    /// Opt-in: only `dp4a` selects these kernels, and it does not read
    /// as `narrow`, so the f32 fallback for the other kinds stays wide.
    #[test]
    fn only_dp4a_selects_the_integer_matvecs() {
        use crate::matvec_kinds::coalesced::MatvecWidth;
        assert!(!parse_dp4a(None));
        assert!(parse_dp4a(Some("dp4a")) && parse_dp4a(Some(" DP4A ")));
        for other in ["", "narrow", "wide", "dp4", "1"] {
            assert!(!parse_dp4a(Some(other)), "{other}");
        }
        assert_eq!(MatvecWidth::parse(Some("dp4a")), MatvecWidth::Wide);
        assert_eq!(nvrtc_arch((8, 9)), Some("compute_89"));
        assert_eq!(nvrtc_arch((12, 0)), Some("compute_120"));
        assert_eq!(nvrtc_arch((6, 0)), None, "sm_60 has no dp4a");
        assert_eq!(nvrtc_arch((6, 1)), Some("compute_61"));
    }
}
