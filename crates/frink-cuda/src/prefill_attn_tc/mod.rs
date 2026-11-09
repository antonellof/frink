//! Prefill attention on the tensor cores (#259): the kernel text, one
//! translation unit per head width, its launch geometry, and a host
//! twin of its arithmetic.
//!
//! The SIMT prefill kernel (`prefill::kernels::CAUSAL_GQA_PREFILL_KERNEL_SRC`)
//! does every product as an f32 FMA and measured about 4.6 TFLOPS on
//! an L40S, 15 ms of a 165 ms Llama-3.1-8B pp512 step. This one takes
//! the shape of llama.cpp's `fattn-mma-f16.cuh` at the pin: Q, K and V
//! as f16 operands of `mma.sync.m16n8k16` with f32 accumulation, an
//! online softmax in f32 between the two products, and the query heads
//! that share a KV head in one block, so they share its K/V tiles.
//! Where it differs from upstream:
//!
//! * The causal and window bounds are computed rather than read from a
//!   mask tensor. A block walks only the keys its rows can see, a warp
//!   skips a tile none of its rows sees, and a tile all of its rows see
//!   in full takes no predicate. At pp512 llama.cpp multiplies the keys
//!   its mask removes as well.
//! * O accumulates in f32, where llama.cpp's kernel can keep f16
//!   accumulators for `V * P`.
//! * K and V are f32 on the device here (the resident stack's own rows
//!   and `prefill::kv_mirror`), rounded to f16 as each tile is staged.
//!   That is the rounding llama.cpp's default f16 KV cache applies when
//!   it stores them.
//!
//! Numerics (ADR 0009): Q is multiplied by the scale (over the softcap
//! when there is one, llama.cpp's `scale /= logit_softcap`) in f32 and
//! rounded to f16. K, V and the probabilities are rounded to f16, and
//! everything else is f32. [`twin::f16_error_bound`] is the error that
//! rounding can put on an output, derived from the operands, and the
//! hardware tests hold the kernel to it.
//!
//! `mma.sync.m16n8k16` with f16 operands needs `sm_80`. Below that, for
//! another head width, or with `FRINK_CUDA_PREFILL_ATTN=simt`, the SIMT
//! kernel runs (`launch::PrefillAttnBody`). The PTX is NVIDIA's, so
//! `frink-rocm`'s census leaves this unit out, as it does the
//! tensor-core GEMM.
//!
//! Always compiled: it is text, pure geometry and a twin, and the host
//! check (`tools/prefill_attn_tc_host_check/run.sh`) executes the text
//! on a CPU against the twin without the `cuda` feature.

pub mod kernel;
/// The launch, and the choice between this kernel and the SIMT one.
#[cfg(feature = "cuda")]
pub mod launch;
pub mod twin;

pub use twin::{f16_error_bound, prefill_attn_tc_twin, round_f16};

/// The attention geometry every attention launch takes, as one value.
#[derive(Clone, Copy, Debug)]
pub struct AttnArgs {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub start_pos: usize,
    /// `None` for full causal attention. Key `t` is visible to the
    /// query at position `p` iff `t <= p` and `p - t < window`.
    pub window: Option<usize>,
    pub scale: f32,
    /// `None` for no logit softcap.
    pub softcap: Option<f32>,
}

impl AttnArgs {
    /// The softcap a kernel takes: the cap, or 0 for none.
    pub fn softcap_or_zero(&self) -> f32 {
        self.softcap.filter(|c| *c > 0.0).unwrap_or(0.0)
    }

    /// What Q is multiplied by before it is rounded to f16: the scale,
    /// over the softcap when there is one, so that `cap * tanh(q . k)`
    /// is the capped logit.
    pub fn q_factor(&self) -> f32 {
        let cap = self.softcap_or_zero();
        if cap > 0.0 {
            self.scale / cap
        } else {
            self.scale
        }
    }

    /// The window a kernel takes: its width, or 0 for none.
    pub fn window_or_zero(&self) -> usize {
        self.window.unwrap_or(0)
    }

    /// Query heads per KV head.
    pub fn gqa(&self) -> usize {
        self.n_heads / self.n_kv_heads.max(1)
    }
}

/// Rows per block: query (token, head) pairs that share one KV head.
pub const ROWS: usize = 64;
/// Warps per block, each one `m16n8k16` row tile of sixteen rows.
pub const WARPS: usize = 4;
/// Threads per block.
pub const THREADS: usize = 32 * WARPS;
/// Halves of padding on every shared row: a row stride of `head_dim +
/// 8` halves puts the eight rows one `ldmatrix` reads on distinct bank
/// groups (`the_shared_rows_an_ldmatrix_reads_hit_distinct_banks`).
pub const PAD: usize = 8;
/// What a launch gets without opting in to more.
pub const STATIC_SMEM_LIMIT: usize = 48 * 1024;
/// The query heads per block a launch may ask for.
pub const GROUPS: [usize; 4] = [8, 4, 2, 1];

const _: () = assert!(ROWS == 16 * WARPS, "a warp owns one m16 row tile");

/// One head width's translation unit.
#[derive(Debug)]
pub struct Variant {
    pub head_dim: usize,
    /// Keys per tile.
    pub keys_per_tile: usize,
    /// Whether a warp holds its Q fragments in registers. At 256 they
    /// would cost 64 registers beside the 128 of O, so Q stays in
    /// shared memory and is read again per tile.
    pub q_in_registers: bool,
    /// NVRTC module and entry point.
    pub names: (&'static str, &'static str),
}

/// Every head width the kernel is built for: 64 and 128 with 64-key
/// tiles and Q in registers, 256 with 32-key tiles.
pub const VARIANTS: [Variant; 3] = [
    Variant {
        head_dim: 64,
        keys_per_tile: 64,
        q_in_registers: true,
        names: ("frink_prefill_attn_tc_d64", "prefill_attn_tc_d64"),
    },
    Variant {
        head_dim: 128,
        keys_per_tile: 64,
        q_in_registers: true,
        names: ("frink_prefill_attn_tc_d128", "prefill_attn_tc_d128"),
    },
    Variant {
        head_dim: 256,
        keys_per_tile: 32,
        q_in_registers: false,
        names: ("frink_prefill_attn_tc_d256", "prefill_attn_tc_d256"),
    },
];

// The shapes the kernel text assumes, for every row of the table: whole
// k16 steps and pairs of n8 tiles over the head and over a key tile,
// staging loops with no remainder, and (Q in registers) a staged Q that
// fits where the K and V tiles go next.
const _: () = {
    let mut i = 0;
    while i < VARIANTS.len() {
        let v = &VARIANTS[i];
        assert!(v.head_dim.is_multiple_of(16) && v.keys_per_tile.is_multiple_of(16));
        assert!((ROWS * v.head_dim / 4).is_multiple_of(THREADS));
        assert!((v.keys_per_tile * v.head_dim / 4).is_multiple_of(THREADS));
        assert!(!v.q_in_registers || ROWS <= 2 * v.keys_per_tile);
        assert!(
            (v.ld() * 2).is_multiple_of(16),
            "ldmatrix rows are 16-byte aligned"
        );
        i += 1;
    }
};

impl Variant {
    /// Shared row stride, in halves.
    pub const fn ld(&self) -> usize {
        self.head_dim + PAD
    }

    /// Dynamic shared bytes a launch asks for: the K and V tiles, and
    /// Q's 64 rows when they stay in shared memory.
    pub const fn smem_bytes(&self) -> usize {
        let q_rows = if self.q_in_registers { 0 } else { ROWS };
        2 * self.ld() * (q_rows + 2 * self.keys_per_tile)
    }

    /// The `#define`s the body reads.
    pub fn defines(&self) -> String {
        format!(
            "#define FX_D {}\n\
             #define FX_BC {}\n\
             #define FX_LD {}\n\
             #define FX_Q_REGS {}\n\
             #define FX_ROWS {ROWS}\n\
             #define FX_THREADS {THREADS}\n",
            self.head_dim,
            self.keys_per_tile,
            self.ld(),
            u8::from(self.q_in_registers),
        )
    }

    /// The body under this variant's entry point name.
    pub fn body(&self) -> String {
        kernel::BODY_SRC.replace("FX_FN_NAME", self.names.1)
    }

    /// The complete translation unit NVRTC compiles.
    pub fn kernel_src(&self) -> String {
        format!(
            "{}{}{}",
            self.defines(),
            crate::mul_mm_tc::TC_PTX_SRC,
            self.body()
        )
    }
}

/// The variant for a head width, if the kernel is built for it.
pub fn variant(head_dim: usize) -> Option<&'static Variant> {
    VARIANTS.iter().find(|v| v.head_dim == head_dim)
}

/// The variant a launch with `args` takes, when the shape is one the
/// kernel takes: a head width it is built for, query heads a whole
/// multiple of the KV heads, and head counts that pack into one `int`.
pub fn variant_for(args: &AttnArgs) -> Option<&'static Variant> {
    let heads_ok = args.n_kv_heads > 0
        && args.n_heads.is_multiple_of(args.n_kv_heads)
        && args.n_heads < 1 << 16
        && args.n_kv_heads < 1 << 15;
    variant(args.head_dim).filter(|_| heads_ok)
}

/// Whether a device of compute capability `cc_major`, which lets one
/// block opt in to `smem_optin` bytes of shared memory, runs `variant`.
pub fn device_takes(variant: &Variant, cc_major: i32, smem_optin: usize) -> bool {
    cc_major >= crate::mul_mm_tc::MIN_CC_MAJOR
        && variant.smem_bytes() <= smem_optin.max(STATIC_SMEM_LIMIT)
}

/// `FRINK_CUDA_PREFILL_ATTN`'s value asks for the SIMT kernel.
pub fn simt_requested(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim().eq_ignore_ascii_case("simt"))
}

/// Blocks a launch takes at `g` query heads per block.
pub fn blocks(gqa: usize, n_q: usize, g: usize) -> usize {
    n_q.div_ceil(ROWS / g) * gqa.div_ceil(g)
}

/// Query heads per block for `gqa` query heads per KV head and a batch
/// of `n_q` tokens: the `G` of [`GROUPS`] that needs the fewest blocks
/// of [`ROWS`] rows, the larger on a tie. A wide group fills the rows of
/// a short batch with heads, a narrow one wastes no rows on a head
/// count that is not a power of two (seven heads take eight rows of a
/// G = 8 block, while at pp512 G = 1 pads nothing).
pub fn heads_per_block(gqa: usize, n_q: usize) -> usize {
    GROUPS
        .into_iter()
        .min_by_key(|&g| blocks(gqa, n_q, g))
        .unwrap_or(1)
}

/// The grid: token tiles along x, (KV head, head group) along y.
pub fn grid(n_q: usize, n_kv_heads: usize, gqa: usize, g: usize) -> (u32, u32, u32) {
    (
        n_q.div_ceil(ROWS / g) as u32,
        (n_kv_heads * gqa.div_ceil(g)) as u32,
        1,
    )
}

/// The (token, query head) that row `r` of token tile `bx` and head
/// block `by` is, or `None` for a row past the batch or past its KV
/// head's query heads: the kernel's mapping, which the twin walks. (The
/// kernel launches tile `bx` as `blockIdx.x = gridDim.x - 1 - bx`.)
pub fn block_row(
    (bx, by): (usize, usize),
    r: usize,
    g: usize,
    gqa: usize,
    n_q: usize,
) -> Option<(usize, usize)> {
    let n_hg = gqa.div_ceil(g);
    let kvh = by / n_hg;
    let hi = (by - kvh * n_hg) * g + r % g;
    let j = bx * (ROWS / g) + r / g;
    (j < n_q && hi < gqa).then_some((j, kvh * gqa + hi))
}

/// The first key a block whose first token is at `pos0` reads: the
/// first row's window start, or 0.
pub fn first_key(pos0: usize, window: usize) -> usize {
    if window > 0 {
        (pos0 + 1).saturating_sub(window)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_emits_a_unit_that_names_its_entry_and_its_geometry() {
        for v in &VARIANTS {
            let src = v.kernel_src();
            assert!(src.contains(&format!("__global__ void {}(", v.names.1)));
            assert!(v.names.0.ends_with(&format!("_d{}", v.head_dim)));
            assert!(src.contains(&format!("#define FX_D {}\n", v.head_dim)));
            assert!(src.contains(&format!("#define FX_BC {}\n", v.keys_per_tile)));
            assert!(src.contains(&format!("#define FX_LD {}\n", v.ld())));
            assert!(src.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
            assert!(src.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"));
            assert!(src.contains(kernel::SMEM_DECL));
            assert!(!src.contains("FX_FN_NAME"));
        }
        let names: std::collections::HashSet<_> = VARIANTS.iter().map(|v| v.names).collect();
        assert_eq!(names.len(), VARIANTS.len(), "a module name repeats");
    }

    /// The budgets the module doc and the launch rely on: 64 and 128
    /// under the 48 KiB every launch gets, 256 under the 99 KiB the
    /// smallest sm_80+ opt-in gives (sm_86, sm_89).
    #[test]
    fn the_shared_tiles_fit_the_budgets_the_launch_assumes() {
        let bytes: Vec<usize> = VARIANTS.iter().map(Variant::smem_bytes).collect();
        assert_eq!(bytes, [18_432, 34_816, 67_584]);
        for v in &VARIANTS {
            assert!(v.smem_bytes() <= 99 * 1024, "{}", v.head_dim);
            assert_eq!(
                v.smem_bytes() <= STATIC_SMEM_LIMIT,
                v.head_dim <= 128,
                "{}",
                v.head_dim
            );
            // A staged Q fits where the K and V tiles go.
            if v.q_in_registers {
                assert!(ROWS * v.ld() * 2 <= v.smem_bytes());
            }
        }
    }

    /// The eight 16-byte rows one `ldmatrix` matrix reads, at the padded
    /// stride, cover all 32 banks once: no bank conflict on Q, K or V.
    #[test]
    fn the_shared_rows_an_ldmatrix_reads_hit_distinct_banks() {
        for v in &VARIANTS {
            let mut banks: Vec<usize> = (0..8)
                .flat_map(|row| {
                    let byte = row * v.ld() * 2;
                    (0..4).map(move |w| (byte / 4 + w) % 32)
                })
                .collect();
            banks.sort_unstable();
            banks.dedup();
            assert_eq!(banks.len(), 32, "head_dim {}", v.head_dim);
        }
    }

    #[test]
    fn the_tensor_cores_take_the_widths_and_devices_they_were_built_for() {
        let args = |n_heads, n_kv_heads, head_dim| AttnArgs {
            n_heads,
            n_kv_heads,
            head_dim,
            start_pos: 0,
            window: None,
            scale: 1.0,
            softcap: None,
        };
        for d in [64, 128, 256] {
            assert_eq!(variant_for(&args(32, 8, d)).map(|v| v.head_dim), Some(d));
        }
        for d in [32, 80, 96, 112, 192, 512] {
            assert!(variant_for(&args(32, 8, d)).is_none(), "{d}");
        }
        assert!(variant_for(&args(30, 8, 128)).is_none(), "uneven GQA");
        assert!(variant_for(&args(8, 0, 128)).is_none(), "no KV heads");
        let (d128, d256) = (variant(128).unwrap(), variant(256).unwrap());
        assert!(
            !device_takes(d128, 7, 99 * 1024),
            "sm_75 has no m16n8k16.f16"
        );
        assert!(device_takes(d128, 8, 0), "128 needs no opt-in");
        assert!(!device_takes(d256, 8, 48 * 1024), "256 needs an opt-in");
        assert!(device_takes(d256, 8, 99 * 1024));
        assert!(device_takes(d256, 9, 227 * 1024));
    }

    #[test]
    fn only_simt_selects_the_simt_kernel() {
        assert!(simt_requested(Some("simt")));
        assert!(simt_requested(Some("SIMT")));
        assert!(simt_requested(Some(" simt\n")));
        for other in [None, Some(""), Some("tc"), Some("0"), Some("simd")] {
            assert!(!simt_requested(other), "{other:?}");
        }
    }

    /// Powers of two get one group per KV head, as llama.cpp groups
    /// them. Seven heads take eight-row groups for a short batch and
    /// one-row ones at pp512, where G = 8 would waste an eighth.
    #[test]
    fn the_head_group_is_the_one_with_fewest_blocks() {
        for (gqa, n_q, g) in [
            (1, 512, 1),
            (2, 512, 2),
            (4, 512, 4),
            (8, 512, 8),
            (8, 3, 8),
            (4, 40, 4),
            (7, 13, 8),
            (7, 512, 1),
            (3, 512, 1),
            (3, 10, 4),
            (12, 512, 4),
            (16, 512, 8),
        ] {
            assert_eq!(heads_per_block(gqa, n_q), g, "gqa {gqa} n_q {n_q}");
            for other in GROUPS {
                assert!(blocks(gqa, n_q, g) <= blocks(gqa, n_q, other));
            }
        }
        assert_eq!(grid(77, 2, 4, 4), (5, 2, 1));
        assert_eq!(grid(13, 1, 7, 8), (2, 1, 1));
        assert_eq!(grid(512, 4, 7, 1), (8, 28, 1));
    }

    /// The grid's rows are every (token, head) of the batch exactly
    /// once, at every group size, padded rows aside.
    #[test]
    fn the_grid_rows_cover_every_token_and_head_once() {
        for (n_q, n_heads, n_kv_heads) in [(13, 7, 1), (77, 8, 2), (64, 32, 8), (5, 12, 1)] {
            let gqa = n_heads / n_kv_heads;
            for g in GROUPS {
                let (gx, gy, _) = grid(n_q, n_kv_heads, gqa, g);
                let mut seen = vec![0u8; n_q * n_heads];
                for bx in 0..gx as usize {
                    for by in 0..gy as usize {
                        for r in 0..ROWS {
                            if let Some((j, h)) = block_row((bx, by), r, g, gqa, n_q) {
                                assert_eq!(h / gqa, by / gqa.div_ceil(g), "a row left its KV head");
                                seen[j * n_heads + h] += 1;
                            }
                        }
                    }
                }
                assert!(
                    seen.iter().all(|&c| c == 1),
                    "n_q {n_q} heads {n_heads} g {g}"
                );
            }
        }
    }

    #[test]
    fn the_first_key_is_the_first_rows_window_start() {
        assert_eq!(first_key(10, 0), 0);
        assert_eq!(first_key(10, 4), 7);
        assert_eq!(first_key(10, 11), 0);
        assert_eq!(first_key(10, 12), 0);
        assert_eq!(first_key(0, 1), 0);
    }
}
