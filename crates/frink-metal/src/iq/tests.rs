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
                block_bytes: by_name($kind).unwrap().block_bytes,
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
/// MSL struct declares the `BLOCK_BYTES` its row names. Runs without a
/// device.
#[test]
fn every_kind_declares_its_block_size_in_its_functor() {
    for k in &KINDS {
        let decl = format!(
            "struct {} {{\n    static constexpr constant short BLOCK_BYTES = {};",
            k.functor, k.block_bytes
        );
        assert!(IQ_FUNCTORS_MSL.contains(&decl), "{}: {decl}", k.name);
    }
}

/// Every entry point the table names is generated in the unit the
/// launch reads it from, once. Runs without a device.
#[test]
fn every_entry_point_is_generated_once_in_its_unit() {
    let mv = iq_matvec_src();
    let mm = *crate::gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC;
    let once = |src: &str, line: String| {
        assert_eq!(src.matches(&line).count(), 1, "{line}");
    };
    for k in &KINDS {
        once(mv, format!("IQ_MATVEC_ENTRY({}, {})", k.matvec, k.functor));
        once(
            mv,
            format!(
                "MOE_MV_ID_ENTRIES({}, iq_matvec_impl<{}>)",
                k.prefix, k.functor
            ),
        );
        let dq = dequant_typedef(k.functor);
        once(mm, format!("MUL_MM_SG_ENTRY({}, {dq})", k.mm_sg));
        once(mm, format!("MUL_MM_SG_F16_ENTRY({}, {dq})", k.mm_sg_f16));
        once(mm, format!("MUL_MM_ID_ENTRY({}, {dq})", k.mm_id));
        once(mm, format!("MUL_MM_ID_F16_ENTRY({}, {dq})", k.mm_id_f16));
        assert_eq!(aligned_fn(k.mm_sg), Some(k.mm_sg_a));
        assert_eq!(aligned_fn(k.mm_sg_f16), Some(k.mm_sg_f16_a));
        assert_eq!(f16_fn(k.mm_sg), Some(k.mm_sg_f16));
        assert_eq!(f16_fn(k.mm_id), Some(k.mm_id_f16));
    }
    // The grid tables are in each unit once: one copy of the text.
    for src in [mv, mm] {
        assert_eq!(src.matches("constant ulong IQ2XXS_GRID[256]").count(), 1);
    }
}
