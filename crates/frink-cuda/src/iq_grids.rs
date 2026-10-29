//! The IQ1/IQ2/IQ3 grid and sign tables as CUDA `__device__` arrays,
//! generated from `frink_quant::iq_tables` (#405).
//!
//! One copy of each table exists in the source tree: the CPU's, which
//! `frink_quant` reproduces from ggml's `ggml-common.h`. A kind's
//! [`MulMmKind::grids`](crate::mul_mm::MulMmKind::grids) names the
//! tables its dequant reads, and [`crate::mul_mm::kernel_src_with_body`]
//! emits exactly those from this module ahead of the dequant, so the
//! GEMM, the matvec and the gather of a kind read the same text, and
//! the Rust twins index the same `frink_quant` arrays. Nothing is typed
//! twice.
//!
//! `__device__` rather than `__constant__`, as ggml declares its grids
//! (`GGML_TABLE_BEGIN` under CUDA): the lanes of a warp index different
//! rows, and constant memory serializes divergent reads.
//!
//! IQ1_S/IQ1_M read the CPU form of `iq1s_grid` (eight signed bytes
//! per row), not ggml CUDA's `iq1s_grid_gpu` nibble form: the value
//! `d * (v + delta)` is the same exact product either way, and the CPU
//! form is the one table `frink_quant` holds.

use frink_quant::iq_tables as t;
use std::fmt::Write;

/// One table a dequant reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IqGrid {
    /// `kmask_iq2xs`: bit `j` of a sign byte.
    Kmask,
    /// `ksigns_iq2xs`: the 128 even-parity sign bytes.
    Ksigns,
    Iq2xxs,
    Iq2xs,
    Iq2s,
    Iq3xxs,
    Iq3s,
    Iq1s,
}

impl IqGrid {
    /// The identifier the dequant source spells.
    pub fn c_name(self) -> &'static str {
        match self {
            IqGrid::Kmask => "FRINK_KMASK_IQ2XS",
            IqGrid::Ksigns => "FRINK_KSIGNS_IQ2XS",
            IqGrid::Iq2xxs => "FRINK_IQ2XXS_GRID",
            IqGrid::Iq2xs => "FRINK_IQ2XS_GRID",
            IqGrid::Iq2s => "FRINK_IQ2S_GRID",
            IqGrid::Iq3xxs => "FRINK_IQ3XXS_GRID",
            IqGrid::Iq3s => "FRINK_IQ3S_GRID",
            IqGrid::Iq1s => "FRINK_IQ1S_GRID",
        }
    }

    /// The table as one CUDA C declaration.
    pub fn src(self) -> String {
        let u8d = |v: &u8| v.to_string();
        let u32x = |v: &u32| format!("0x{v:08x}u");
        let u64x = |v: &u64| format!("0x{v:016x}ull");
        match self {
            IqGrid::Kmask => decl("unsigned char", self.c_name(), &t::KMASK_IQ2XS, u8d),
            IqGrid::Ksigns => decl("unsigned char", self.c_name(), &t::KSIGNS_IQ2XS, u8d),
            IqGrid::Iq2xxs => decl("unsigned long long", self.c_name(), &t::IQ2XXS_GRID, u64x),
            IqGrid::Iq2xs => decl("unsigned long long", self.c_name(), &t::IQ2XS_GRID, u64x),
            IqGrid::Iq2s => decl("unsigned long long", self.c_name(), &t::IQ2S_GRID, u64x),
            IqGrid::Iq3xxs => decl("unsigned int", self.c_name(), &t::IQ3XXS_GRID, u32x),
            IqGrid::Iq3s => decl("unsigned int", self.c_name(), &t::IQ3S_GRID, u32x),
            IqGrid::Iq1s => decl("unsigned long long", self.c_name(), &t::IQ1S_GRID, u64x),
        }
    }
}

fn decl<T>(ty: &str, name: &str, rows: &[T], fmt: impl Fn(&T) -> String) -> String {
    let mut out = format!("\n__device__ const {ty} {name}[{}] = {{", rows.len());
    for (i, r) in rows.iter().enumerate() {
        if i % 8 == 0 {
            out.push_str("\n    ");
        }
        let _ = write!(out, "{}, ", fmt(r));
    }
    out.push_str("\n};\n");
    out
}

/// Every table in `grids`, in order, as CUDA C.
pub fn grids_src(grids: &[IqGrid]) -> String {
    grids.iter().map(|g| g.src()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [IqGrid; 8] = [
        IqGrid::Kmask,
        IqGrid::Ksigns,
        IqGrid::Iq2xxs,
        IqGrid::Iq2xs,
        IqGrid::Iq2s,
        IqGrid::Iq3xxs,
        IqGrid::Iq3s,
        IqGrid::Iq1s,
    ];

    /// Each table is declared at its CPU length, and its values parse
    /// back to the CPU's: the emitter cannot drop, reorder or misprint
    /// a row.
    ///
    /// Sabotage: format any row with `{v:07x}` or skip one and this
    /// names the table.
    #[test]
    fn every_table_is_the_cpu_table() {
        for g in ALL {
            let src = g.src();
            let open = src.find('{').unwrap();
            let close = src.rfind('}').unwrap();
            let vals: Vec<u64> = src[open + 1..close]
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let s = s.trim_end_matches("ull").trim_end_matches('u');
                    match s.strip_prefix("0x") {
                        Some(h) => u64::from_str_radix(h, 16).unwrap(),
                        None => s.parse().unwrap(),
                    }
                })
                .collect();
            let want: Vec<u64> = match g {
                IqGrid::Kmask => t::KMASK_IQ2XS.iter().map(|&v| v.into()).collect(),
                IqGrid::Ksigns => t::KSIGNS_IQ2XS.iter().map(|&v| v.into()).collect(),
                IqGrid::Iq2xxs => t::IQ2XXS_GRID.to_vec(),
                IqGrid::Iq2xs => t::IQ2XS_GRID.to_vec(),
                IqGrid::Iq2s => t::IQ2S_GRID.to_vec(),
                IqGrid::Iq3xxs => t::IQ3XXS_GRID.iter().map(|&v| v.into()).collect(),
                IqGrid::Iq3s => t::IQ3S_GRID.iter().map(|&v| v.into()).collect(),
                IqGrid::Iq1s => t::IQ1S_GRID.to_vec(),
            };
            assert_eq!(vals, want, "{}", g.c_name());
            assert!(src.contains(&format!("{}[{}]", g.c_name(), want.len())));
        }
    }
}
