//! The seven IQ kernels against `frink_quant`'s CPU dequant, on real
//! hardware (`--ignored`), through `crate::k_quant_check`'s shapes and
//! bounds (ragged row groups, one block a row, partial lane strides),
//! plus the table checks that run without a device.

use super::*;
use crate::k_quant_check::{check_gemm, check_matvec, CheckKind};

macro_rules! iq_check {
    ($kind:literal, $spec:ident, $test:ident, $dequant:path, $mv:ident, $mm:ident) => {
        fn $spec() -> CheckKind {
            fn blocks(seed: u32, n: usize) -> Vec<u8> {
                test_blocks($kind, seed, n)
            }
            CheckKind {
                name: $kind,
                block_bytes: crate::gen_kinds::by_name($kind).unwrap().block_bytes,
                block_elems: 256,
                dequant: $dequant,
                blocks,
                matvec: $mv,
                gemm: $mm,
            }
        }

        #[test]
        #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
        fn $test() {
            check_matvec(&$spec());
            check_gemm(&$spec());
        }
    };
}

iq_check!(
    "IQ2_XXS",
    iq2_xxs,
    iq2_xxs_matvec_and_gemm_match_the_cpu_dequant,
    frink_quant::dequant_iq2_xxs,
    launch_iq2_xxs_matvec,
    launch_iq2_xxs_mul_mm_sg
);
iq_check!(
    "IQ2_XS",
    iq2_xs,
    iq2_xs_matvec_and_gemm_match_the_cpu_dequant,
    frink_quant::dequant_iq2_xs,
    launch_iq2_xs_matvec,
    launch_iq2_xs_mul_mm_sg
);
iq_check!(
    "IQ2_S",
    iq2_s,
    iq2_s_matvec_and_gemm_match_the_cpu_dequant,
    frink_quant::dequant_iq2_s,
    launch_iq2_s_matvec,
    launch_iq2_s_mul_mm_sg
);
iq_check!(
    "IQ3_XXS",
    iq3_xxs,
    iq3_xxs_matvec_and_gemm_match_the_cpu_dequant,
    frink_quant::dequant_iq3_xxs,
    launch_iq3_xxs_matvec,
    launch_iq3_xxs_mul_mm_sg
);
iq_check!(
    "IQ3_S",
    iq3_s,
    iq3_s_matvec_and_gemm_match_the_cpu_dequant,
    frink_quant::dequant_iq3_s,
    launch_iq3_s_matvec,
    launch_iq3_s_mul_mm_sg
);
iq_check!(
    "IQ1_S",
    iq1_s,
    iq1_s_matvec_and_gemm_match_the_cpu_dequant,
    frink_quant::dequant_iq1_s,
    launch_iq1_s_matvec,
    launch_iq1_s_mul_mm_sg
);
iq_check!(
    "IQ1_M",
    iq1_m,
    iq1_m_matvec_and_gemm_match_the_cpu_dequant,
    frink_quant::dequant_iq1_m,
    launch_iq1_m_matvec,
    launch_iq1_m_mul_mm_sg
);

/// The table's block sizes are `frink_quant`'s and the source's: each
/// IQ `group` struct declares the `BLOCK_BYTES` its row names. Runs
/// without a device.
#[test]
fn every_kind_declares_its_block_size_in_its_functor() {
    for (k, group) in kinds() {
        let decl = format!(
            "struct {group} {{\n    static constexpr constant short BLOCK_BYTES = {};",
            k.block_bytes
        );
        assert!(IQ_FUNCTORS_MSL.contains(&decl), "{}: {decl}", k.name);
        assert!(
            gemm_functors_msl().contains(&format!("typedef IqDequant<{group}> {};", k.dequant)),
            "{}",
            k.name
        );
    }
    assert_eq!(kinds().count(), 7);
}

/// The grid tables are in each unit that reads them once: one copy of
/// the text. Runs without a device.
#[test]
fn the_grid_tables_are_in_each_unit_once() {
    for src in [iq_matvec_src(), *crate::gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC] {
        assert_eq!(src.matches("constant ulong IQ2XXS_GRID[256]").count(), 1);
    }
}
