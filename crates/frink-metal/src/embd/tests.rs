//! Every Metal gather against `frink_quant`'s CPU dequant (the host
//! embedding path, `WeightMatrix::dequant_row`), on real hardware
//! (`--ignored`), plus the table checks that run without a device.

use super::*;

/// Each gather is keyed by the matvec kernel the Metal table really
/// resolves for its kind, so the decode path's `from_fn_name` and the
/// coverage matrix's `from_kind_name` cannot disagree. Runs without a
/// device.
#[test]
fn every_gather_is_keyed_by_its_kinds_matvec_kernel() {
    for k in EmbdKind::ALL {
        let meta = crate::gpu::matvec_launch_meta(k.kind_name())
            .unwrap_or_else(|| panic!("{k:?}: a gather for a kind with no matvec kernel"));
        assert_eq!(meta.1, k.matvec_fn_name(), "{k:?}");
    }
}

/// The other direction: every quantized kind with a Metal matvec has a
/// gather, so a new matvec kind cannot leave its embedding on the host
/// unnoticed. `F32` is the one matvec kind with no block to dequant.
#[test]
fn every_matvec_kind_has_a_gather() {
    for &name in crate::gpu::MATVEC_KINDS {
        if name == "F32" {
            continue;
        }
        assert!(
            EmbdKind::from_kind_name(name).is_some(),
            "{name} has a Metal matvec and no get_rows"
        );
    }
}

/// The functor each gather instantiates is one the GEMM translation
/// unit defines, with that kind's block geometry: a gather cannot carry
/// a dequant of its own. Runs without a device.
#[test]
fn every_gather_reuses_its_kinds_gemm_functor() {
    let src: &str = &K_QUANT_MUL_MM_SG_KERNEL_SRC;
    for k in EmbdKind::ALL {
        let functor = k.dequant_functor();
        // The codebook formats share one templated functor; their
        // geometry lives in the format struct the typedef names.
        let geometry = match k {
            EmbdKind::Iq4Nl => {
                assert!(src.contains("typedef Codebook4Dequant<CB4IQ4NL> IQ4NLDequant;"));
                "CB4IQ4NL"
            }
            EmbdKind::Mxfp4 => {
                assert!(src.contains("typedef Codebook4Dequant<CB4MXFP4> MXFP4Dequant;"));
                "CB4MXFP4"
            }
            // The IQ formats share `IqDequant`; the geometry is the
            // `group` / `affine` struct it is instantiated over.
            _ if crate::iq::kinds().any(|(g, _)| g.name == k.kind_name()) => {
                let g = functor.strip_suffix("Dequant").unwrap();
                assert!(src.contains(&format!("typedef IqDequant<{g}> {functor};")));
                g
            }
            _ => functor,
        };
        let at = src
            .find(&format!("struct {geometry} {{"))
            .unwrap_or_else(|| panic!("{k:?}: no `struct {geometry}` in the GEMM unit"));
        let (_, _, block_bytes, _, _) = crate::gpu::matvec_launch_meta(k.kind_name()).unwrap();
        let head = &src[at..(at + 200).min(src.len())];
        assert!(
            head.contains(&format!("BLOCK_BYTES = {block_bytes};")),
            "{k:?}: `{geometry}` does not carry {block_bytes}-byte blocks"
        );
        assert!(
            src.contains(&format!(
                "GET_ROWS_ENTRY({}, {functor})",
                k.get_rows_fn_name()
            )),
            "{k:?}: no generated entry"
        );
    }
}

/// The CPU dequant of whole blocks of `kind`.
fn cpu_dequant(kind: EmbdKind, bytes: &[u8]) -> Vec<f32> {
    use frink_quant as q;
    match kind {
        EmbdKind::Q8_0 => q::dequant_q8_0(bytes),
        EmbdKind::Q4_0 => q::dequant_q4_0(bytes),
        EmbdKind::Q5_0 => q::dequant_q5_0(bytes),
        EmbdKind::Q4K => q::dequant_q4_k(bytes),
        EmbdKind::Q5K => q::dequant_q5_k(bytes),
        EmbdKind::Q6K => q::dequant_q6_k(bytes),
        EmbdKind::Iq4Xs => q::dequant_iq4_xs(bytes),
        EmbdKind::Ptq1_0 => q::ternary::dequant_trits(bytes, q::ternary::PTQ1_0),
        EmbdKind::Q2K => q::dequant_q2_k(bytes),
        EmbdKind::Q3K => q::dequant_q3_k(bytes),
        EmbdKind::Iq4Nl => q::dequant_iq4_nl(bytes),
        EmbdKind::Mxfp4 => q::dequant_mxfp4_gguf(bytes),
        EmbdKind::Iq2Xxs => q::dequant_iq2_xxs(bytes),
        EmbdKind::Iq2Xs => q::dequant_iq2_xs(bytes),
        EmbdKind::Iq2S => q::dequant_iq2_s(bytes),
        EmbdKind::Iq3Xxs => q::dequant_iq3_xxs(bytes),
        EmbdKind::Iq3S => q::dequant_iq3_s(bytes),
        EmbdKind::Iq1S => q::dequant_iq1_s(bytes),
        EmbdKind::Iq1M => q::dequant_iq1_m(bytes),
        EmbdKind::Q4_1 => q::dequant_q4_1(bytes),
        EmbdKind::Q5_1 => q::dequant_q5_1(bytes),
        EmbdKind::Tq1_0 => q::ternary::dequant_trits(bytes, q::ternary::TQ1_0),
    }
    .expect("whole blocks")
}

/// Pseudo-random block bytes with each block's scale fields forced to
/// realistic magnitudes (random scale bytes decode to values no
/// quantizer writes). The `half` scales sit where each format keeps
/// them; MXFP4's second block carries the subnormal exponent 0.
fn blocks(kind: EmbdKind, n_blocks: usize, block_bytes: usize, seed: u32) -> Vec<u8> {
    if crate::gen_kinds::by_name(kind.kind_name()).is_some() {
        return crate::gen_kinds::test_blocks(kind.kind_name(), seed, n_blocks);
    }
    let mut w = crate::k_quant_check::pseudo_bytes(seed, n_blocks * block_bytes);
    let halves: &[usize] = match kind {
        EmbdKind::Q8_0 | EmbdKind::Q4_0 | EmbdKind::Q5_0 | EmbdKind::Iq4Xs | EmbdKind::Iq4Nl => {
            &[0]
        }
        EmbdKind::Q4K | EmbdKind::Q5K => &[0, 2],
        EmbdKind::Q6K => &[208],
        EmbdKind::Ptq1_0 => &[26],
        EmbdKind::Q2K => &[80, 82],
        EmbdKind::Q3K => &[108],
        EmbdKind::Mxfp4 => &[],
        _ => unreachable!("the generated kinds returned above"),
    };
    for b in 0..n_blocks {
        let at = b * block_bytes;
        for (i, &off) in halves.iter().enumerate() {
            let v = 0.004 + 0.001 * ((b + 3 * i) % 7) as f32;
            w[at + off..at + off + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        if kind == EmbdKind::Mxfp4 {
            w[at] = if b == 1 { 0 } else { 118 + (b % 6) as u8 };
        }
    }
    w
}

/// Every kind, at row widths of 1, 3 and 7 blocks (an odd MXFP4 row is
/// an odd byte count, so later rows start at odd addresses; 7 blocks of
/// a 256-element kind is 112 threads, a ragged second threadgroup), on
/// seven rows read at the first, the second, a middle and the last
/// token id. Every value is held to the CPU's; prints the worst error
/// per kind.
#[test]
#[ignore = "needs a real Metal GPU"]
fn every_gather_matches_the_cpu_dequant() {
    const ROWS: usize = 7;
    for kind in EmbdKind::ALL {
        let (_, _, bb, be, _) = crate::gpu::matvec_launch_meta(kind.kind_name()).unwrap();
        let mut worst = 0f32;
        for (i, n_blocks) in [1usize, 3, 7].into_iter().enumerate() {
            let row_bytes = n_blocks * bb;
            let cols = n_blocks * be;
            let w = blocks(kind, ROWS * n_blocks, bb, 17 + i as u32);
            for row in [0usize, 1, 3, ROWS - 1] {
                let want = cpu_dequant(kind, &w[row * row_bytes..(row + 1) * row_bytes]);
                let got = launch_get_rows(kind, &w, ROWS, row_bytes, cols, row).unwrap();
                assert_eq!(got.len(), want.len());
                for (j, (&g, &c)) in got.iter().zip(&want).enumerate() {
                    let err = (g - c).abs();
                    assert!(
                        err <= 1e-6 * c.abs().max(1.0),
                        "{kind:?} {n_blocks} blocks, row {row}, col {j}: metal {g} cpu {c}"
                    );
                    worst = worst.max(err);
                }
            }
        }
        eprintln!(
            "get_rows {}: max |metal - cpu| = {worst:e}",
            kind.kind_name()
        );
    }
}
