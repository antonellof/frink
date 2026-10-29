//! The kinds whose every Metal entry point is generated from one
//! dequant: the seven IQ1/IQ2/IQ3 grid formats (`crate::iq`) and Q4_1,
//! Q5_1 and TQ1_0 (`crate::dq_mv`) (#405).
//!
//! [`KINDS`] is the one table. Each row names the GGUF kind, its block
//! geometry, the MSL dequant functor its GEMM, `mul_mm_id` and gather
//! instantiate, and every entry point, derived from one prefix. The GEMM
//! entry lines are generated from it ([`gemm_entries_msl`]), each
//! family's matvec unit generates its own matvec and `mul_mv_id` lines
//! from it, and `gpu.rs`'s lookups read it, so the source and the
//! lookups cannot name different sets.

/// Which body a kind's matvec is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    /// `crate::iq`: the `group` / `affine` pair named by `group`.
    Iq { group: &'static str },
    /// `crate::dq_mv`: the matvec reads the GEMM functor itself.
    Functor,
}

/// One generated kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GenKind {
    pub name: &'static str,
    pub family: Family,
    /// The 16-value functor (`NL`, `BLOCK_BYTES`, `get`).
    pub dequant: &'static str,
    pub block_bytes: usize,
    pub block_elems: usize,
    pub prefix: &'static str,
    pub matvec: &'static str,
    pub mm_sg: &'static str,
    pub mm_sg_a: &'static str,
    pub mm_sg_f16: &'static str,
    pub mm_sg_f16_a: &'static str,
    pub mm_id: &'static str,
    pub mm_id_f16: &'static str,
    pub moe_matvec_id: &'static str,
    pub moe_down_id: &'static str,
}

macro_rules! gen_kind {
    ($name:literal, $family:expr, $dequant:literal, $bytes:expr, $elems:expr, $p:literal) => {
        GenKind {
            name: $name,
            family: $family,
            dequant: $dequant,
            block_bytes: $bytes,
            block_elems: $elems,
            prefix: $p,
            matvec: concat!($p, "_matvec"),
            mm_sg: concat!($p, "_mul_mm_sg"),
            mm_sg_a: concat!($p, "_mul_mm_sg_a"),
            mm_sg_f16: concat!($p, "_mul_mm_sg_f16"),
            mm_sg_f16_a: concat!($p, "_mul_mm_sg_f16_a"),
            mm_id: concat!($p, "_mul_mm_id"),
            mm_id_f16: concat!($p, "_mul_mm_id_f16"),
            moe_matvec_id: concat!($p, "_moe_matvec_id"),
            moe_down_id: concat!($p, "_moe_down_id"),
        }
    };
}

const fn iq(group: &'static str) -> Family {
    Family::Iq { group }
}

use frink_quant as q;

/// Every generated kind, with `frink_quant`'s block sizes.
pub(crate) const KINDS: [GenKind; 10] = [
    gen_kind!(
        "IQ2_XXS",
        iq("IQ2XXS"),
        "IQ2XXSDequant",
        q::IQ2_XXS_BLOCK_BYTES,
        256,
        "iq2_xxs"
    ),
    gen_kind!(
        "IQ2_XS",
        iq("IQ2XS"),
        "IQ2XSDequant",
        q::IQ2_XS_BLOCK_BYTES,
        256,
        "iq2_xs"
    ),
    gen_kind!(
        "IQ2_S",
        iq("IQ2S"),
        "IQ2SDequant",
        q::IQ2_S_BLOCK_BYTES,
        256,
        "iq2_s"
    ),
    gen_kind!(
        "IQ3_XXS",
        iq("IQ3XXS"),
        "IQ3XXSDequant",
        q::IQ3_XXS_BLOCK_BYTES,
        256,
        "iq3_xxs"
    ),
    gen_kind!(
        "IQ3_S",
        iq("IQ3S"),
        "IQ3SDequant",
        q::IQ3_S_BLOCK_BYTES,
        256,
        "iq3_s"
    ),
    gen_kind!(
        "IQ1_S",
        iq("IQ1S"),
        "IQ1SDequant",
        q::IQ1_S_BLOCK_BYTES,
        256,
        "iq1_s"
    ),
    gen_kind!(
        "IQ1_M",
        iq("IQ1M"),
        "IQ1MDequant",
        q::IQ1_M_BLOCK_BYTES,
        256,
        "iq1_m"
    ),
    gen_kind!(
        "Q4_1",
        Family::Functor,
        "Q4_1Dequant",
        q::Q4_1_BLOCK_BYTES,
        32,
        "q4_1"
    ),
    gen_kind!(
        "Q5_1",
        Family::Functor,
        "Q5_1Dequant",
        q::Q5_1_BLOCK_BYTES,
        32,
        "q5_1"
    ),
    gen_kind!(
        "TQ1_0",
        Family::Functor,
        "TQ1_0Dequant",
        q::ternary::TQ1_0_BLOCK_BYTES,
        256,
        "tq1_0"
    ),
];

/// Rows one matvec threadgroup owns in both bodies: four per
/// simdgroup, two simdgroups.
pub(crate) const ROWS_PER_TG: usize = 8;

/// Threads and threadgroup bytes of every generated matvec.
pub(crate) const MATVEC_TG: (usize, usize) = (64, 0);

/// The kind named `name` (`"IQ2_XXS"`, `"Q4_1"`, ...).
pub(crate) fn by_name(name: &str) -> Option<&'static GenKind> {
    KINDS.iter().find(|k| k.name == name)
}

/// The kind whose decode matvec is `fn_name`.
pub(crate) fn by_matvec_fn(fn_name: &str) -> Option<&'static GenKind> {
    KINDS.iter().find(|k| k.matvec == fn_name)
}

/// The translation unit holding `k`'s matvec and `mul_mv_id` entries.
pub(crate) fn matvec_src(k: &GenKind) -> &'static str {
    match k.family {
        Family::Iq { .. } => crate::iq::iq_matvec_src(),
        Family::Functor => crate::dq_mv::dq_matvec_src(),
    }
}

/// `matvec_launch_meta`'s row for a generated kind.
pub(crate) fn matvec_launch_meta(
    kind: &str,
) -> Option<(&'static str, &'static str, usize, usize, usize)> {
    let k = by_name(kind)?;
    Some((
        matvec_src(k),
        k.matvec,
        k.block_bytes,
        k.block_elems,
        ROWS_PER_TG,
    ))
}

/// `mul_mm_sg_meta` / `mul_mm_id_meta`'s row for a generated kind.
pub(crate) fn gemm_meta(kind: &str, id: bool) -> Option<(&'static str, usize, usize)> {
    let k = by_name(kind)?;
    Some((
        if id { k.mm_id } else { k.mm_sg },
        k.block_bytes,
        k.block_elems,
    ))
}

/// The exact-tile (`_a`) sibling of a generated `mul_mm_sg` entry.
pub(crate) fn aligned_fn(fn_name: &str) -> Option<&'static str> {
    KINDS.iter().find_map(|k| {
        if k.mm_sg == fn_name {
            Some(k.mm_sg_a)
        } else if k.mm_sg_f16 == fn_name {
            Some(k.mm_sg_f16_a)
        } else {
            None
        }
    })
}

/// The half-activation sibling of a generated `mul_mm_sg` or
/// `mul_mm_id` entry.
pub(crate) fn f16_fn(fn_name: &str) -> Option<&'static str> {
    KINDS.iter().find_map(|k| {
        if k.mm_sg == fn_name {
            Some(k.mm_sg_f16)
        } else if k.mm_id == fn_name {
            Some(k.mm_id_f16)
        } else {
            None
        }
    })
}

/// The GEMM and `mul_mm_id` entry lines for every generated kind.
pub(crate) fn gemm_entries_msl() -> String {
    let mut s = String::new();
    for k in &KINDS {
        let dq = k.dequant;
        s.push_str(&format!(
            "MUL_MM_SG_ENTRY({}, {dq})\nMUL_MM_SG_F16_ENTRY({}, {dq})\n\
             MUL_MM_ID_ENTRY({}, {dq})\nMUL_MM_ID_F16_ENTRY({}, {dq})\n",
            k.mm_sg, k.mm_sg_f16, k.mm_id, k.mm_id_f16
        ));
    }
    s
}

/// The `matvec_launch_meta`-shaped launch the two families share.
pub(crate) fn launch_matvec(
    kind: &str,
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, crate::gpu::MetalError> {
    crate::gpu::launch_matvec_kind_with(
        kind,
        weights,
        x,
        rows,
        row_bytes,
        crate::gpu::MatvecEpilogue::default(),
    )
    .ok_or(crate::gpu::MetalError::CommandFailed)?
}

/// The simdgroup GEMM launch the two families share.
pub(crate) fn launch_gemm(
    kind: &str,
    weights: &[u8],
    x_batch: &[f32],
    rows: usize,
    row_bytes: usize,
    batch_size: usize,
) -> Result<Vec<f32>, crate::gpu::MetalError> {
    crate::gpu::launch_mul_mm_sg(kind, weights, x_batch, rows, row_bytes, batch_size)?
        .ok_or(crate::gpu::MetalError::CommandFailed)
}

/// `pub fn launch_<kind>_matvec` and `launch_<kind>_mul_mm_sg` for each
/// `(kind, matvec, gemm)` triple.
macro_rules! gen_launchers {
    ($($kind:literal, $mv:ident, $mm:ident;)*) => {$(
        #[doc = concat!("The ", $kind, " decode matvec.")]
        pub fn $mv(
            weights: &[u8],
            x: &[f32],
            rows: usize,
            row_bytes: usize,
        ) -> Result<Vec<f32>, crate::gpu::MetalError> {
            crate::gen_kinds::launch_matvec($kind, weights, x, rows, row_bytes)
        }

        #[doc = concat!("The ", $kind, " simdgroup GEMM; `x_batch` is `[batch, cols]`, ")]
        #[doc = "the result `[batch, rows]`."]
        pub fn $mm(
            weights: &[u8],
            x_batch: &[f32],
            rows: usize,
            row_bytes: usize,
            batch_size: usize,
        ) -> Result<Vec<f32>, crate::gpu::MetalError> {
            crate::gen_kinds::launch_gemm($kind, weights, x_batch, rows, row_bytes, batch_size)
        }
    )*};
}
pub(crate) use gen_launchers;

/// Test blocks of a generated kind, from its family's generator.
#[cfg(test)]
pub(crate) fn test_blocks(name: &str, seed: u32, n_blocks: usize) -> Vec<u8> {
    match by_name(name).expect("a generated kind").family {
        Family::Iq { .. } => crate::iq::test_blocks(name, seed, n_blocks),
        Family::Functor => crate::dq_mv::test_blocks(name, seed, n_blocks),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prefixes, kind names and every entry point are distinct, or two
    /// kinds would share a cached pipeline (the cache keys on the name).
    #[test]
    fn every_name_is_distinct() {
        let mut names: Vec<&str> = KINDS
            .iter()
            .flat_map(|k| {
                [
                    k.name,
                    k.matvec,
                    k.mm_sg,
                    k.mm_sg_a,
                    k.mm_sg_f16,
                    k.mm_sg_f16_a,
                    k.mm_id,
                    k.mm_id_f16,
                    k.moe_matvec_id,
                    k.moe_down_id,
                ]
            })
            .collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n);
    }

    /// Every entry point the table names is generated once, in the unit
    /// its launch reads it from, and the sibling lookups agree. Runs
    /// without a device.
    #[test]
    fn every_entry_point_is_generated_once_in_its_unit() {
        let mm = *crate::gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC;
        let once = |src: &str, line: String| {
            assert_eq!(src.matches(&line).count(), 1, "{line}");
        };
        for k in &KINDS {
            let mv = matvec_src(k);
            let impl_ = match k.family {
                Family::Iq { group } => {
                    once(mv, format!("IQ_MATVEC_ENTRY({}, {group})", k.matvec));
                    format!("iq_matvec_impl<{group}>")
                }
                Family::Functor => {
                    once(mv, format!("DQ_MATVEC_ENTRY({}, {})", k.matvec, k.dequant));
                    format!("dq_matvec_impl<{}>", k.dequant)
                }
            };
            once(mv, format!("MOE_MV_ID_ENTRIES({}, {impl_})", k.prefix));
            let dq = k.dequant;
            once(mm, format!("MUL_MM_SG_ENTRY({}, {dq})", k.mm_sg));
            once(mm, format!("MUL_MM_SG_F16_ENTRY({}, {dq})", k.mm_sg_f16));
            once(mm, format!("MUL_MM_ID_ENTRY({}, {dq})", k.mm_id));
            once(mm, format!("MUL_MM_ID_F16_ENTRY({}, {dq})", k.mm_id_f16));
            assert_eq!(aligned_fn(k.mm_sg), Some(k.mm_sg_a));
            assert_eq!(aligned_fn(k.mm_sg_f16), Some(k.mm_sg_f16_a));
            assert_eq!(f16_fn(k.mm_sg), Some(k.mm_sg_f16));
            assert_eq!(f16_fn(k.mm_id), Some(k.mm_id_f16));
        }
    }

    /// Each kind's geometry is the one `frink_quant` dequantizes, and a
    /// row is whole 16-value chunks, which both bodies require.
    #[test]
    fn every_kind_has_whole_sixteen_value_chunks() {
        for k in &KINDS {
            assert!(k.block_elems.is_multiple_of(16), "{}", k.name);
            assert!(
                k.block_bytes.is_multiple_of(2),
                "{}: halves stay aligned",
                k.name
            );
        }
    }
}
