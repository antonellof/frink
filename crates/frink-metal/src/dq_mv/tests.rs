//! The Q4_1, Q5_1 and TQ1_0 kernels against `frink_quant`'s CPU dequant,
//! on real hardware (`--ignored`), through `crate::k_quant_check`'s
//! shapes and bounds, plus a table check that runs without a device.

use super::*;
use crate::k_quant_check::{check_gemm, check_matvec, CheckKind};

fn dequant_tq1_0(b: &[u8]) -> Result<Vec<f32>, frink_quant::QuantError> {
    frink_quant::ternary::dequant_trits(b, frink_quant::ternary::TQ1_0)
}

macro_rules! dq_check {
    ($kind:literal, $spec:ident, $test:ident, $dequant:path, $mv:ident, $mm:ident) => {
        fn $spec() -> CheckKind {
            fn blocks(seed: u32, n: usize) -> Vec<u8> {
                test_blocks($kind, seed, n)
            }
            let k = crate::gen_kinds::by_name($kind).unwrap();
            CheckKind {
                name: $kind,
                block_bytes: k.block_bytes,
                block_elems: k.block_elems,
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

dq_check!(
    "Q4_1",
    q4_1,
    q4_1_matvec_and_gemm_match_the_cpu_dequant,
    frink_quant::dequant_q4_1,
    launch_q4_1_matvec,
    launch_q4_1_mul_mm_sg
);
dq_check!(
    "Q5_1",
    q5_1,
    q5_1_matvec_and_gemm_match_the_cpu_dequant,
    frink_quant::dequant_q5_1,
    launch_q5_1_matvec,
    launch_q5_1_mul_mm_sg
);
dq_check!(
    "TQ1_0",
    tq1_0,
    tq1_0_matvec_and_gemm_match_the_cpu_dequant,
    dequant_tq1_0,
    launch_tq1_0_matvec,
    launch_tq1_0_mul_mm_sg
);

/// Each functor declares the block size its table row names, and the
/// GEMM unit defines it once. Runs without a device.
#[test]
fn every_functor_declares_its_block_size() {
    let mm = *crate::gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC;
    for k in crate::gen_kinds::KINDS
        .iter()
        .filter(|k| k.family == crate::gen_kinds::Family::Functor)
    {
        let head = format!("struct {} {{", k.dequant);
        assert_eq!(mm.matches(&head).count(), 1, "{}", k.name);
        let at = mm.find(&head).unwrap();
        assert!(
            mm[at..at + 200].contains(&format!("BLOCK_BYTES = {};", k.block_bytes)),
            "{}",
            k.name
        );
    }
}
