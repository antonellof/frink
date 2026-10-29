//! The IQ1/IQ2/IQ3 grid and sign tables as Metal `constant` arrays,
//! generated from `frink_quant::iq_tables` (#405).
//!
//! One copy of each table exists in the source tree: the CPU's, which
//! `frink_quant` reproduces from ggml's `ggml-common.h`. The MSL text is
//! built from it once, and the same text is spliced into both
//! translation units that read the grids (the GEMM unit, which also
//! holds the gathers and `mul_mm_id`, and the IQ matvec unit, which
//! also holds `mul_mv_id`). So the GPU cannot read a table the CPU does
//! not, and no table is written out twice.
//!
//! The one table ggml's Metal shaders spell differently is IQ1_S's grid:
//! the CPU's `iq1s_grid` holds eight signed bytes in `{-1, 0, 1}` per
//! row, and the shaders read `iq1s_grid_gpu`, the same row as eight
//! nibbles `v + 1`, element `i` in the low nibble of byte `i` and
//! element `i + 4` in its high nibble (`dequantize_iq1_s`). The shader
//! form is derived here from the CPU form ([`iq1s_grid_gpu`]), and its
//! test pins rows of ggml's `iq1s_grid_gpu` at the pin.

use frink_quant::iq_tables as t;
use std::fmt::Write;

/// The IQ1_S grid in ggml's shader form: row `r` of `iq1s_grid` as
/// eight nibbles `v + 1`, byte `i` holding element `i` low and `i + 4`
/// high.
pub(crate) fn iq1s_grid_gpu() -> Vec<u32> {
    t::IQ1S_GRID
        .iter()
        .map(|&row| {
            let v = |j: usize| (((row >> (8 * j)) as u8 as i8) + 1) as u32;
            (0..4).fold(0u32, |acc, i| acc | ((v(i) | (v(i + 4) << 4)) << (8 * i)))
        })
        .collect()
}

fn table<T: Copy>(out: &mut String, ty: &str, name: &str, rows: &[T], fmt: impl Fn(T) -> String) {
    let _ = write!(out, "constant {ty} {name}[{}] = {{", rows.len());
    for (i, &r) in rows.iter().enumerate() {
        if i % 8 == 0 {
            out.push_str("\n    ");
        }
        out.push_str(&fmt(r));
        out.push_str(", ");
    }
    out.push_str("\n};\n");
}

/// Every grid and sign table the IQ functors read, as MSL. Built once
/// and leaked: both translation units splice the same `&'static str`.
pub(crate) fn iq_grids_msl() -> &'static str {
    use std::sync::LazyLock;
    static SRC: LazyLock<&'static str> = LazyLock::new(|| {
        let mut s = String::from(
            "// From frink_quant::iq_tables (ggml `ggml-common.h`); see `crate::iq_grids`.\n",
        );
        let u64x = |v: u64| format!("0x{v:016x}ul");
        let u32x = |v: u32| format!("0x{v:08x}u");
        let u8d = |v: u8| v.to_string();
        table(&mut s, "uchar", "IQ_KMASK", &t::KMASK_IQ2XS, u8d);
        table(&mut s, "uchar", "IQ_KSIGNS", &t::KSIGNS_IQ2XS, u8d);
        table(&mut s, "ulong", "IQ2XXS_GRID", &t::IQ2XXS_GRID, u64x);
        table(&mut s, "ulong", "IQ2XS_GRID", &t::IQ2XS_GRID, u64x);
        table(&mut s, "ulong", "IQ2S_GRID", &t::IQ2S_GRID, u64x);
        table(&mut s, "uint", "IQ3XXS_GRID", &t::IQ3XXS_GRID, u32x);
        table(&mut s, "uint", "IQ3S_GRID", &t::IQ3S_GRID, u32x);
        table(&mut s, "uint", "IQ1S_GRID_GPU", &iq1s_grid_gpu(), u32x);
        Box::leak(s.into_boxed_str())
    });
    &SRC
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows of ggml's `iq1s_grid_gpu` at the pin (`ggml-common.h`: the
    /// first eight, and the last), which the derivation must reproduce.
    #[test]
    fn the_derived_iq1s_shader_grid_is_ggmls() {
        let g = iq1s_grid_gpu();
        assert_eq!(g.len(), 2048);
        assert_eq!(
            g[..8],
            [
                0x00000000, 0x00000002, 0x00000101, 0x00000200, 0x00000202, 0x00010001, 0x00010101,
                0x00020000
            ]
        );
        assert_eq!(g[2047], 0x22222222);
    }

    /// Each table is declared once, at its CPU length.
    #[test]
    fn every_table_is_declared_once_at_its_length() {
        let src = iq_grids_msl();
        for decl in [
            "uchar IQ_KMASK[8]",
            "uchar IQ_KSIGNS[128]",
            "ulong IQ2XXS_GRID[256]",
            "ulong IQ2XS_GRID[512]",
            "ulong IQ2S_GRID[1024]",
            "uint IQ3XXS_GRID[256]",
            "uint IQ3S_GRID[512]",
            "uint IQ1S_GRID_GPU[2048]",
        ] {
            assert_eq!(src.matches(decl).count(), 1, "{decl}");
        }
    }
}
