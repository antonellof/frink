//! The coalesced matvec kernels: one warp per output row, the warp's
//! lanes reading the row's blocks as contiguous runs.
//!
//! They are CUDA C text plus the table that routes a kind to its
//! coalesced kernel, so nothing here needs `cudarc`; the launch plan
//! that pairs a kernel with its geometry stays in `gpu.rs`. Their
//! scalar twins are in [`crate::coalesced_twin`], and
//! `tools/matvec_host_check/run.sh` executes this exact text on a host
//! CPU against those twins.
//!
//! # One kernel per kind, two widths (#133)
//!
//! Every kernel here is compiled with `FX_LANES`, the number of lanes
//! that share ONE block (a 256-element super-block, or a 32-element
//! Q8_0 block). A warp therefore works on `32 / FX_LANES` blocks of
//! its row at once, and each lane reads a run of `FX_RUN` consecutive
//! quants from one of them.
//!
//! - [`MatvecWidth::Narrow`] is `FX_LANES = 32`: the whole warp on one
//!   block per loop trip. This is the geometry #144 to #146 shipped,
//!   with the same lane-to-data mapping and summation order.
//! - [`MatvecWidth::Wide`] gives each lane an 8 to 24 byte run, so a
//!   warp has four blocks of its row in flight per trip (eight for
//!   Q8_0).
//!
//! The reason is the arithmetic in the PR that introduced it: a
//! narrow warp has 144 bytes of Q4_K (34 of Q8_0) in flight per round
//! trip to DRAM and walks the row serially, one block per trip, so a
//! 3072-wide row is 12 dependent round trips for Q4_K and 96 for Q8_0.
//! Wide cuts both by the blocks-per-trip factor and amortises each
//! block's header decode over four times the bytes.
//!
//! `FRINK_CUDA_MATVEC=narrow` selects the narrow width for an A/B;
//! anything else (or unset) is wide. Both widths are the same source,
//! so they cannot drift apart: the A/B varies exactly one number.

/// How many warps (so rows) one coalesced matvec threadblock carries.
/// Eight keeps enough rows in flight per SM to hold loads outstanding.
/// `gpu.rs` launches with it and `tools/matvec_host_check` runs with
/// it, so the two cannot disagree about the grid.
pub const WARPS_PER_BLOCK: usize = 8;

/// Which geometry the coalesced matvec kernels are compiled for. See
/// the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatvecWidth {
    /// One block per warp per loop trip (`FX_LANES = 32`).
    Narrow,
    /// Several blocks per warp per loop trip, per [`CoalescedKind::wide_lanes`].
    Wide,
}

/// The environment switch that selects [`MatvecWidth::Narrow`].
pub const MATVEC_WIDTH_ENV: &str = "FRINK_CUDA_MATVEC";

impl MatvecWidth {
    /// Parses the switch's value: `narrow` (any case) is narrow, and
    /// everything else, including unset, is the default.
    pub fn parse(value: Option<&str>) -> Self {
        match value {
            Some(v) if v.trim().eq_ignore_ascii_case("narrow") => Self::Narrow,
            _ => Self::Wide,
        }
    }

    /// The width this process runs, read from [`MATVEC_WIDTH_ENV`] once.
    /// Hoisted out of the per-launch path: a matvec runs a few hundred
    /// times per decoded token.
    pub fn from_env() -> Self {
        static WIDTH: std::sync::OnceLock<MatvecWidth> = std::sync::OnceLock::new();
        *WIDTH.get_or_init(|| Self::parse(std::env::var(MATVEC_WIDTH_ENV).ok().as_deref()))
    }
}

/// One coalesced kernel and what it takes to compile it at each width.
#[derive(Debug, Clone, Copy)]
pub struct CoalescedKind {
    /// The `MatvecKind::fn_name` of the block-per-row kernel this one
    /// replaces; the launch plan routes on it.
    pub base_fn: &'static str,
    /// The `__global__` entry point, the same name at both widths.
    pub entry: &'static str,
    /// The kernel text, without the prelude and `FX_LANES`.
    pub src: &'static str,
    /// NVRTC module cache key at [`MatvecWidth::Narrow`].
    pub narrow_module: &'static str,
    /// NVRTC module cache key at [`MatvecWidth::Wide`]. Distinct from
    /// the narrow one, or the first width loaded would serve both.
    pub wide_module: &'static str,
    /// `FX_LANES` at [`MatvecWidth::Wide`].
    pub wide_lanes: usize,
    /// The smallest `FX_LANES` the kernel's per-lane run allows. The
    /// kernel text refuses anything smaller with `#error`; this is the
    /// same bound, so a host test can hold the table to it.
    pub min_lanes: usize,
}

impl CoalescedKind {
    /// `FX_LANES` for `width`.
    pub fn lanes(&self, width: MatvecWidth) -> usize {
        match width {
            MatvecWidth::Narrow => 32,
            MatvecWidth::Wide => self.wide_lanes,
        }
    }

    /// The NVRTC module key for `width`.
    pub fn module(&self, width: MatvecWidth) -> &'static str {
        match width {
            MatvecWidth::Narrow => self.narrow_module,
            MatvecWidth::Wide => self.wide_module,
        }
    }

    /// The complete translation unit for `width`: `FX_LANES`, the
    /// shared prelude, then the kernel.
    pub fn source(&self, width: MatvecWidth) -> String {
        format!(
            "#define FX_LANES {}\n{}{}",
            self.lanes(width),
            COALESCED_PRELUDE,
            self.src
        )
    }
}

/// The coalesced kernels. This is the ONLY place one is named. Four
/// launchers that each hard-coded one kind is the shape this repo keeps
/// getting bitten by, so `every_coalesced_kernel_is_reachable_from_the_
/// table` fails if a kernel is added to this file and not to this list.
pub const COALESCED: &[CoalescedKind] = &[
    CoalescedKind {
        base_fn: "q4_k_matvec",
        entry: "q4_k_matvec_coalesced",
        src: Q4_K_MATVEC_COALESCED_KERNEL_SRC,
        narrow_module: "frink_q4_k_coalesced",
        wide_module: "frink_q4_k_coalesced_wide",
        // 16 bytes of `qs` per lane: one `uint4` load.
        wide_lanes: 8,
        min_lanes: 4,
    },
    CoalescedKind {
        base_fn: "q5_k_matvec",
        entry: "q5_k_matvec_coalesced",
        src: Q5_K_MATVEC_COALESCED_KERNEL_SRC,
        narrow_module: "frink_q5_k_coalesced",
        wide_module: "frink_q5_k_coalesced_wide",
        // Four elements of each 32-group per lane: 4 B of `qh` and
        // 4 x 4 B of `qs`, each one 32-bit load.
        wide_lanes: 8,
        min_lanes: 1,
    },
    CoalescedKind {
        base_fn: "q6_k_matvec",
        entry: "q6_k_matvec_coalesced",
        src: Q6_K_MATVEC_COALESCED_KERNEL_SRC,
        narrow_module: "frink_q6_k_coalesced",
        wide_module: "frink_q6_k_coalesced_wide",
        // Four of each 32 indices per lane, 24 bytes per block. The
        // block is 210 bytes, so only 2-byte loads are aligned.
        wide_lanes: 8,
        min_lanes: 2,
    },
    CoalescedKind {
        base_fn: "q8_0_matvec",
        entry: "q8_0_matvec_coalesced",
        src: Q8_0_MATVEC_COALESCED_KERNEL_SRC,
        narrow_module: "frink_q8_0_coalesced",
        wide_module: "frink_q8_0_coalesced_wide",
        // Eight quants per lane, eight blocks per warp per trip. The
        // block is 34 bytes, so the quants are read as four `u16`s.
        wide_lanes: 4,
        min_lanes: 1,
    },
];

/// The coalesced kernel replacing the block-per-row kernel `fn_name`,
/// or `None` if that kind still runs the block-per-row kernel.
pub fn coalesced_kind(fn_name: &str) -> Option<&'static CoalescedKind> {
    COALESCED.iter().find(|k| k.base_fn == fn_name)
}

/// Helpers every coalesced kernel shares, prepended to each one by
/// [`CoalescedKind::source`]. One copy, where the four kernels used to
/// carry four suffixed copies of the f16 decode and two of the Q4_K
/// scale unpack.
///
/// The loaders take the run length as a template argument so that the
/// branch on it folds at compile time: a 16-byte run becomes one
/// 128-bit load, a 1-byte run one byte load. Their alignment contract
/// is the caller's: `fx_load_bytes_a4` needs a 4-aligned pointer when
/// `N % 4 == 0`, `fx_load_bytes_a2` a 2-aligned one when `N` is even,
/// and `fx_load_floats` a 16-aligned one when `N % 4 == 0`. Every call
/// site below states why its pointer qualifies.
pub const COALESCED_PRELUDE: &str = r#"
#ifndef FX_LANES
#error "FX_LANES (lanes per block) must be defined by the launcher"
#endif
#if FX_LANES < 1 || FX_LANES > 32 || (32 % FX_LANES) != 0
#error "FX_LANES must divide the warp"
#endif
#define FX_BLOCKS_PER_TRIP (32 / FX_LANES)

__device__ __forceinline__ float fx_f16_to_f32(unsigned short bits) {
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

__device__ __forceinline__ float fx_f16_at(const unsigned char* p) {
    return fx_f16_to_f32((unsigned short)p[0] | ((unsigned short)p[1] << 8));
}

__device__ __forceinline__ void fx_q4_k_scale_min(
    int j, const unsigned char* scales, unsigned char* sc, unsigned char* m
) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

template <int N>
__device__ __forceinline__ void fx_load_bytes_a4(const unsigned char* p, unsigned char* b) {
    if (N == 16) {
        const uint4 v = *(const uint4*)p;
        const unsigned int w[4] = { v.x, v.y, v.z, v.w };
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) b[4 * k + i] = (unsigned char)(w[k] >> (8 * i));
        }
    } else if (N % 4 == 0) {
        #pragma unroll
        for (int k = 0; k < N / 4; ++k) {
            const unsigned int w = ((const unsigned int*)p)[k];
            #pragma unroll
            for (int i = 0; i < 4; ++i) b[4 * k + i] = (unsigned char)(w >> (8 * i));
        }
    } else {
        #pragma unroll
        for (int e = 0; e < N; ++e) b[e] = p[e];
    }
}

template <int N>
__device__ __forceinline__ void fx_load_bytes_a2(const unsigned char* p, unsigned char* b) {
    if (N % 2 == 0) {
        #pragma unroll
        for (int k = 0; k < N / 2; ++k) {
            const unsigned short w = ((const unsigned short*)p)[k];
            b[2 * k] = (unsigned char)w;
            b[2 * k + 1] = (unsigned char)(w >> 8);
        }
    } else {
        #pragma unroll
        for (int e = 0; e < N; ++e) b[e] = p[e];
    }
}

template <int N>
__device__ __forceinline__ void fx_load_floats(const float* p, float* f) {
    if (N % 4 == 0) {
        #pragma unroll
        for (int k = 0; k < N / 4; ++k) {
            const float4 v = ((const float4*)p)[k];
            f[4 * k] = v.x;
            f[4 * k + 1] = v.y;
            f[4 * k + 2] = v.z;
            f[4 * k + 3] = v.w;
        }
    } else {
        #pragma unroll
        for (int e = 0; e < N; ++e) f[e] = p[e];
    }
}

__device__ __forceinline__ float fx_warp_sum(float acc) {
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, s);
    }
    return acc;
}
"#;

/// CUDA C for the coalesced Q5_K matvec.
///
/// Q5_K adds a fifth bit per weight over Q4_K, held in `qh` as one
/// BIT-PLANE per group: bit `2*oi` for the group's low nibbles and bit
/// `2*oi + 1` for its high ones. Lane `sub` of a block owns elements
/// `[sub*R, sub*R + R)` of each 32-element group, `R = 32 / FX_LANES`,
/// so the warp's loads of `qs`, `qh` and the activations are each
/// contiguous. `q5_k_matvec_coalesced_row` in `coalesced_twin.rs` is
/// the scalar twin that pins that down.
///
/// Alignment: a Q5_K block is 176 = 11 * 16 bytes, so every block of
/// every row starts 16-aligned; `qh` is at +16, `qs` at +48, and the
/// run start `sub*R` is a multiple of `R`.
pub const Q5_K_MATVEC_COALESCED_KERNEL_SRC: &str = r#"
#define FX_RUN (32 / FX_LANES)

extern "C" __global__ void q5_k_matvec_coalesced(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* out,
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps + warp;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;
    const int sub = lane % FX_LANES;
    const int l0 = sub * FX_RUN;
    float acc = 0.0f;

    for (int blk = lane / FX_LANES; blk < n_blocks_per_row; blk += FX_BLOCKS_PER_TRIP) {
        const unsigned char* block = row_ptr + (size_t)blk * 176;
        const float d = fx_f16_at(block);
        const float dmin = fx_f16_at(block + 2);
        const unsigned char* scales = block + 4;
        unsigned char h[FX_RUN];
        fx_load_bytes_a4<FX_RUN>(block + 16 + l0, h);

        #pragma unroll
        for (int oi = 0; oi < 4; ++oi) {
            unsigned char sc1, m1, sc2, m2;
            fx_q4_k_scale_min(2 * oi, scales, &sc1, &m1);
            fx_q4_k_scale_min(2 * oi + 1, scales, &sc2, &m2);
            const float d1 = d * (float)sc1, min1 = dmin * (float)m1;
            const float d2 = d * (float)sc2, min2 = dmin * (float)m2;
            unsigned char q[FX_RUN];
            fx_load_bytes_a4<FX_RUN>(block + 48 + oi * 32 + l0, q);
            float xl[FX_RUN], xh[FX_RUN];
            const float* xb = x + blk * 256 + oi * 64 + l0;
            fx_load_floats<FX_RUN>(xb, xl);
            fx_load_floats<FX_RUN>(xb + 32, xh);
            const unsigned char u1 = (unsigned char)(1u << (2 * oi));
            const unsigned char u2 = (unsigned char)(2u << (2 * oi));
            #pragma unroll
            for (int e = 0; e < FX_RUN; ++e) {
                const int hi1 = (h[e] & u1) ? 16 : 0;
                const int hi2 = (h[e] & u2) ? 16 : 0;
                acc += (d1 * (float)((q[e] & 0x0F) + hi1) - min1) * xl[e];
                acc += (d2 * (float)((q[e] >> 4) + hi2) - min2) * xh[e];
            }
        }
    }

    acc = fx_warp_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

/// CUDA C for the coalesced Q8_0 matvec. Lane `sub` of a block takes
/// quants `[sub*R, sub*R + R)`, `R = 32 / FX_LANES`. Narrow is one
/// byte per lane and one 34-byte block per warp per trip; wide is
/// eight bytes per lane and eight blocks per trip.
///
/// The quants are SIGNED; the twin has a test that fails on an
/// unsigned read.
///
/// Alignment: a block is 34 bytes, so a block starts 2-aligned only,
/// and its quants at +2 are 2-aligned too. Hence `u16` loads, not
/// wider ones. The activations start at `blk*32 + sub*R`, a multiple
/// of 4 floats whenever `R` is.
pub const Q8_0_MATVEC_COALESCED_KERNEL_SRC: &str = r#"
#define FX_RUN (32 / FX_LANES)

extern "C" __global__ void q8_0_matvec_coalesced(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* out,
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps + warp;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;
    const int l0 = (lane % FX_LANES) * FX_RUN;
    float acc = 0.0f;

    for (int blk = lane / FX_LANES; blk < n_blocks_per_row; blk += FX_BLOCKS_PER_TRIP) {
        const unsigned char* block = row_ptr + (size_t)blk * 34;
        const float d = fx_f16_at(block);
        unsigned char q[FX_RUN];
        fx_load_bytes_a2<FX_RUN>(block + 2 + l0, q);
        float xv[FX_RUN];
        fx_load_floats<FX_RUN>(x + blk * 32 + l0, xv);
        #pragma unroll
        for (int e = 0; e < FX_RUN; ++e) {
            acc += d * (float)(signed char)q[e] * xv[e];
        }
    }

    acc = fx_warp_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

/// Q6_K matvec with a coalesced access pattern.
///
/// The block-per-row kernel gives each thread a whole 210-byte
/// super-block, so a warp's 32 loads land 210 bytes apart. Here lane
/// `sub` of a block takes indices `[sub*R, sub*R + R)` of the 32 the
/// old inner loop iterated, `R = 32 / FX_LANES`, which makes `ql[l]`,
/// `ql[l+32]` and `qh[l]` each contiguous across the warp.
///
/// Q6_K matters more than its share of a checkpoint suggests: a
/// `Q4_K_M` file stores its output tensor as Q6_K, so this kernel runs
/// every token.
///
/// Alignment: 210 bytes per block, so 2-byte loads only. `R` is at
/// most 16, so a lane's run never straddles the two scale groups
/// (`is = l0 / 16`).
pub const Q6_K_MATVEC_COALESCED_KERNEL_SRC: &str = r#"
#define FX_RUN (32 / FX_LANES)
#if FX_RUN > 16
#error "Q6_K: a lane's run must stay inside one 16-index scale group"
#endif

extern "C" __global__ void q6_k_matvec_coalesced(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* out,
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps + warp;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;
    const int l0 = (lane % FX_LANES) * FX_RUN;
    const int is = l0 / 16;
    float acc = 0.0f;

    for (int blk = lane / FX_LANES; blk < n_blocks_per_row; blk += FX_BLOCKS_PER_TRIP) {
        const unsigned char* block = row_ptr + (size_t)blk * 210;
        const float d = fx_f16_at(block + 208);
        const int x_base = blk * 256;

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned char* ql = block + half * 64;
            const unsigned char* qh = block + 128 + half * 32;
            const signed char* sc = (const signed char*)(block + 192 + half * 8);
            unsigned char a[FX_RUN], b[FX_RUN], h[FX_RUN];
            fx_load_bytes_a2<FX_RUN>(ql + l0, a);
            fx_load_bytes_a2<FX_RUN>(ql + 32 + l0, b);
            fx_load_bytes_a2<FX_RUN>(qh + l0, h);
            const float* xh = x + x_base + half * 128 + l0;
            float x1[FX_RUN], x2[FX_RUN], x3[FX_RUN], x4[FX_RUN];
            fx_load_floats<FX_RUN>(xh, x1);
            fx_load_floats<FX_RUN>(xh + 32, x2);
            fx_load_floats<FX_RUN>(xh + 64, x3);
            fx_load_floats<FX_RUN>(xh + 96, x4);

            #pragma unroll
            for (int e = 0; e < FX_RUN; ++e) {
                const int q1 = (int)((a[e] & 0x0F) | ((h[e] & 0x03) << 4)) - 32;
                const int q2 = (int)((b[e] & 0x0F) | (((h[e] >> 2) & 0x03) << 4)) - 32;
                const int q3 = (int)((a[e] >> 4) | (((h[e] >> 4) & 0x03) << 4)) - 32;
                const int q4 = (int)((b[e] >> 4) | (((h[e] >> 6) & 0x03) << 4)) - 32;
                acc += d * (float)sc[is] * (float)q1 * x1[e];
                acc += d * (float)sc[is + 2] * (float)q2 * x2[e];
                acc += d * (float)sc[is + 4] * (float)q3 * x3[e];
                acc += d * (float)sc[is + 6] * (float)q4 * x4[e];
            }
        }
    }

    acc = fx_warp_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

/// Q4_K matvec with a COALESCED access pattern.
///
/// The block-per-row kernel walks whole 144-byte super-blocks per
/// thread (`blk += blockDim.x`), so adjacent lanes read addresses 144
/// bytes apart and every lane in a warp touches a different cache
/// line. Measured consequence on an RTX 3060: 19.0 GB/s of weight
/// traffic, **5.3%** of the card's 360 GB/s, where llama.cpp reaches
/// 60.4% (#133).
///
/// The activation stays f32 on purpose. Quantizing it to int8 is what
/// llama.cpp does, and on a real checkpoint it diverges from the CPU
/// reference at token 4, which `frink verify` refuses.
///
/// Lane to data mapping, which is the part to get right. With
/// `B = 128 / FX_LANES` bytes per lane and `sub = lane % FX_LANES`:
///   off = sub*B         byte offset into the 128 qs bytes
///   oi  = off/32        which 32-byte group, so which sub-block PAIR
///   low nibbles  -> activations at blk*256 + oi*64 + (off%32)
///   high nibbles -> the same, plus 32
/// The 32 bytes of a group carry the first 32 activations in their low
/// nibbles and the next 32 in their high nibbles, so a lane that pairs
/// byte `i` with activation `i` for both halves reads the wrong place.
/// `B` must not exceed 32, or a run would straddle two groups.
///
/// Alignment: 144 = 9 * 16 bytes per block and `qs` at +16, so the
/// `B = 16` run is one aligned `uint4`; the activation run starts at a
/// multiple of 4 floats.
pub const Q4_K_MATVEC_COALESCED_KERNEL_SRC: &str = r#"
#define FX_RUN (128 / FX_LANES)
#if FX_RUN > 32 || (FX_RUN % 4) != 0
#error "Q4_K: FX_LANES must be 4, 8, 16 or 32"
#endif

extern "C" __global__ void q4_k_matvec_coalesced(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* out,
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps + warp;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;
    const int off = (lane % FX_LANES) * FX_RUN;
    const int oi = off / 32;
    const int within = off % 32;

    float acc = 0.0f;
    for (int blk = lane / FX_LANES; blk < n_blocks_per_row; blk += FX_BLOCKS_PER_TRIP) {
        const unsigned char* block = row_ptr + (size_t)blk * 144;
        const float d = fx_f16_at(block);
        const float dmin = fx_f16_at(block + 2);
        const unsigned char* scales = block + 4;

        unsigned char sc1, m1, sc2, m2;
        fx_q4_k_scale_min(2 * oi, scales, &sc1, &m1);
        fx_q4_k_scale_min(2 * oi + 1, scales, &sc2, &m2);
        const float d1 = d * (float)sc1, min1 = dmin * (float)m1;
        const float d2 = d * (float)sc2, min2 = dmin * (float)m2;

        unsigned char q[FX_RUN];
        fx_load_bytes_a4<FX_RUN>(block + 16 + off, q);
        const float* xb = x + blk * 256 + oi * 64 + within;
        float xl[FX_RUN], xh[FX_RUN];
        fx_load_floats<FX_RUN>(xb, xl);
        fx_load_floats<FX_RUN>(xb + 32, xh);
        #pragma unroll
        for (int i = 0; i < FX_RUN; ++i) {
            acc += (d1 * (float)(q[i] & 0x0F) - min1) * xl[i];
            acc += (d2 * (float)(q[i] >> 4) - min2) * xh[i];
        }
    }

    acc = fx_warp_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// A coalesced kernel the table does not name is dead code that
    /// reads as coverage. The entry points are derived from this file,
    /// not restated, so adding a kernel without routing it fails here.
    #[test]
    fn every_coalesced_kernel_is_reachable_from_the_table() {
        let src = include_str!("coalesced.rs");
        let mut found = 0usize;
        for line in src.lines() {
            let Some(rest) = line.strip_prefix(r#"extern "C" __global__ void "#) else {
                continue;
            };
            let Some(entry) = rest.split('(').next() else {
                continue;
            };
            let Some(base) = entry.strip_suffix("_coalesced") else {
                continue;
            };
            found += 1;
            let routed = coalesced_kind(base).unwrap_or_else(|| {
                panic!("kernel {entry} is not reachable: no table row for {base}")
            });
            assert_eq!(
                routed.entry, entry,
                "table row for {base} names the wrong entry point"
            );
            assert!(
                routed.src.contains(&format!("void {entry}(")),
                "table row for {base} points at a source that does not define {entry}"
            );
        }
        assert!(
            found >= 4,
            "expected at least four coalesced kernels, found {found}"
        );
    }

    /// Each row must route from a block-per-row kernel that exists,
    /// with module keys unique across both widths. Two rows sharing a
    /// key would have the second served the first one's PTX by the
    /// process-wide module cache, and the narrow and wide keys of one
    /// row sharing a key would make the A/B compare a kernel to itself.
    #[test]
    fn every_row_routes_from_a_real_kernel_with_unique_module_keys() {
        let mut keys: Vec<&str> = Vec::new();
        for k in COALESCED {
            assert!(
                crate::matvec_kinds::KINDS
                    .iter()
                    .any(|m| m.fn_name == k.base_fn),
                "{}: routes from {}, which no MatvecKind defines",
                k.entry,
                k.base_fn
            );
            keys.push(k.narrow_module);
            keys.push(k.wide_module);
        }
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), keys.len(), "duplicate module key in {keys:?}");
    }

    /// The table's lane counts must be ones the kernel text accepts and
    /// that actually widen. A `wide_lanes` of 32 would make the switch
    /// a no-op and the A/B a measurement of noise; one below
    /// `min_lanes` is an NVRTC `#error` at the first decoded token.
    #[test]
    fn every_wide_width_is_legal_and_actually_wider() {
        for k in COALESCED {
            let lanes = k.lanes(MatvecWidth::Wide);
            assert!(
                32 % lanes == 0,
                "{}: {lanes} lanes does not divide a warp",
                k.entry
            );
            assert!(
                lanes >= k.min_lanes,
                "{}: {lanes} lanes is below the kernel's minimum {}",
                k.entry,
                k.min_lanes
            );
            assert!(
                lanes < k.lanes(MatvecWidth::Narrow),
                "{}: wide is not wider than narrow",
                k.entry
            );
            assert_ne!(k.module(MatvecWidth::Wide), k.module(MatvecWidth::Narrow));
        }
    }

    /// The source handed to NVRTC carries its width. A source that
    /// dropped the define fails the prelude's `#error`, and one that
    /// carried the wrong width would compile and compute a different
    /// lane mapping from the one its twin checks.
    #[test]
    fn the_source_defines_the_width_it_was_asked_for() {
        for k in COALESCED {
            for w in [MatvecWidth::Narrow, MatvecWidth::Wide] {
                let src = k.source(w);
                let first = src.lines().next().unwrap_or("");
                assert_eq!(first, format!("#define FX_LANES {}", k.lanes(w)));
                assert!(src.contains(COALESCED_PRELUDE) && src.contains(k.src));
            }
        }
    }

    /// The switch is opt-OUT: only `narrow` selects the old geometry,
    /// so a typo cannot silently turn the default off.
    #[test]
    fn only_narrow_selects_the_narrow_width() {
        assert_eq!(MatvecWidth::parse(None), MatvecWidth::Wide);
        assert_eq!(MatvecWidth::parse(Some("narrow")), MatvecWidth::Narrow);
        assert_eq!(MatvecWidth::parse(Some(" NARROW ")), MatvecWidth::Narrow);
        for other in ["", "wide", "1", "0", "narow"] {
            assert_eq!(
                MatvecWidth::parse(Some(other)),
                MatvecWidth::Wide,
                "{other}"
            );
        }
    }
}
