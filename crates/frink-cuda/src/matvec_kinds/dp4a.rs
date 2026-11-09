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

// The kernel text, moved to its own file; re-exported so every path
// still resolves.
mod kernels;
pub use kernels::{
    DP4A_PRELUDE, MXFP4_DP4A_SRC, Q4_K_DP4A_SRC, Q6_K_DP4A_SRC, Q8_0_DP4A_SRC, Q8_1_COMMON,
    QUANTIZE_Q8_1_KERNEL_SRC,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// A dp4a kernel the table does not name is dead code that reads as
    /// coverage; the entry points are read off this file.
    #[test]
    fn every_dp4a_kernel_is_reachable_from_the_table() {
        let src = include_str!("dp4a/kernels.rs");
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
