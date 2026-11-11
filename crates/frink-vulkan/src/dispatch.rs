//! The typed matvec entry points, one per kind, over [`Runtime`].
//!
//! # The claim this file makes, and what checks it
//!
//! [`Runtime`] is FFI from end to end. Each call is checked by its
//! stated invariant; *collectively* they are checked here, against
//! `frink_quant`'s scalar twins (`dot_*_f32_scalar`). Only that can
//! catch a descriptor bound to the wrong binding, a push constant at
//! the wrong offset, a grid that misses rows, a reduction that drops a
//! partial, or a resident weight served for the wrong host buffer --
//! none memory-unsafe, all silently wrong numbers.

use crate::device::VulkanError;
pub use crate::runtime::Runtime;

/// The matvec for `kind` through its GLSL shader ([`crate::shaders`]):
/// `y[row] = dot(dequant(W[row]), x)`, where `weights` is `rows *
/// row_bytes` of verbatim GGUF blocks, exactly the slice
/// `WeightMatrix::Quantized::data` hands the Metal and CUDA launch
/// functions, and stays resident in `rt` under its address. One
/// argument list for every kind, so `frink-core` holds each as a plain
/// function pointer.
macro_rules! shader_matvec {
    ($($fn_name:ident => $kind:literal),* $(,)?) => {$(
        #[doc = concat!("`", $kind, "` matvec through its GLSL shader.")]
        pub fn $fn_name(
            rt: &mut Runtime,
            weights: &[u8],
            x: &[f32],
            rows: usize,
            row_bytes: usize,
            n_blocks_per_row: usize,
        ) -> Result<Vec<f32>, VulkanError> {
            let s = crate::shaders::matvec_shader($kind).expect("the table has the row");
            assert_eq!(x.len(), n_blocks_per_row * s.block_elems, "activation length");
            assert_eq!(row_bytes, n_blocks_per_row * s.block_bytes, "row stride");
            rt.matvec(s, weights, x, rows)
        }
    )*};
}

shader_matvec! {
    q8_0_matvec => "Q8_0",
    q4_0_matvec => "Q4_0",
    q4_k_matvec => "Q4_K",
    q5_k_matvec => "Q5_K",
    q6_k_matvec => "Q6_K",
    q4_1_matvec => "Q4_1",
    q5_0_matvec => "Q5_0",
    q5_1_matvec => "Q5_1",
    q2_k_matvec => "Q2_K",
    q3_k_matvec => "Q3_K",
    iq4_nl_matvec => "IQ4_NL",
    iq4_xs_matvec => "IQ4_XS",
    mxfp4_matvec => "MXFP4",
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn pseudo_random(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // The top 32 bits, so the value spans [-1, 1]. `>> 33`
                // kept 31, and every activation was <= 0.
                ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// A row of `blocks` blocks of `kind`: deterministic bytes, with each
    /// block's f16 scale fields set to small finite values so no block
    /// decodes to inf or NaN.
    pub(crate) fn kind_row(kind: &str, blocks: usize, seed: u64) -> Vec<u8> {
        let shader = crate::shaders::matvec_shader(kind).unwrap();
        let mut s = seed | 1;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        };
        let mut out = Vec::new();
        for b in 0..blocks {
            let mut blk: Vec<u8> = (0..shader.block_bytes).map(|_| next()).collect();
            let d = half::f16::from_f32(0.01 + 0.002 * (b % 7) as f32).to_le_bytes();
            let dmin = half::f16::from_f32(0.005 + 0.001 * (b % 5) as f32).to_le_bytes();
            match kind {
                "Q8_0" | "Q4_0" | "Q5_0" | "IQ4_NL" | "IQ4_XS" => blk[0..2].copy_from_slice(&d),
                "Q4_K" | "Q5_K" | "Q4_1" | "Q5_1" => {
                    blk[0..2].copy_from_slice(&d);
                    blk[2..4].copy_from_slice(&dmin);
                }
                "Q6_K" => blk[208..210].copy_from_slice(&d),
                "Q2_K" => {
                    blk[80..82].copy_from_slice(&d);
                    blk[82..84].copy_from_slice(&dmin);
                }
                "Q3_K" => blk[108..110].copy_from_slice(&d),
                // E8M0: 2^-9 .. 2^-2, and e = 0 (the subnormal case) on
                // every eighth block.
                "MXFP4" => blk[0] = if b % 8 == 7 { 0 } else { 118 + (b % 8) as u8 },
                other => panic!("no layout for {other}"),
            }
            out.extend(blk);
        }
        out
    }

    pub(crate) fn twin(kind: &str) -> fn(&[u8], &[f32]) -> f32 {
        match kind {
            "Q8_0" => frink_quant::dot_q8_0_f32_scalar,
            "Q4_0" => frink_quant::dot_q4_0_f32_scalar,
            "Q4_K" => frink_quant::dot_q4_k_f32_scalar,
            "Q5_K" => frink_quant::dot_q5_k_f32_scalar,
            "Q6_K" => frink_quant::dot_q6_k_f32_scalar,
            "Q4_1" => frink_quant::dot_q4_1_f32_scalar,
            "Q5_0" => frink_quant::dot_q5_0_f32_scalar,
            "Q5_1" => frink_quant::dot_q5_1_f32_scalar,
            "Q2_K" => frink_quant::dot_q2_k_f32_scalar,
            "Q3_K" => frink_quant::dot_q3_k_f32_scalar,
            "IQ4_NL" => frink_quant::dot_iq4_nl_f32_scalar,
            "IQ4_XS" => frink_quant::dot_iq4_xs_f32_scalar,
            "MXFP4" => frink_quant::dot_mxfp4_gguf_f32_scalar,
            other => panic!("no twin for {other}"),
        }
    }

    pub(crate) fn runtime() -> Option<Runtime> {
        match Runtime::new() {
            Ok(rt) => Some(rt),
            Err(e) => {
                crate::skipped(&format!("no Vulkan device: {e}"));
                None
            }
        }
    }

    type Launch =
        fn(&mut Runtime, &[u8], &[f32], usize, usize, usize) -> Result<Vec<f32>, VulkanError>;
    const LAUNCHES: [(&str, Launch); 13] = [
        ("Q8_0", q8_0_matvec),
        ("Q4_0", q4_0_matvec),
        ("Q4_K", q4_k_matvec),
        ("Q5_K", q5_k_matvec),
        ("Q6_K", q6_k_matvec),
        ("Q4_1", q4_1_matvec),
        ("Q5_0", q5_0_matvec),
        ("Q5_1", q5_1_matvec),
        ("Q2_K", q2_k_matvec),
        ("Q3_K", q3_k_matvec),
        ("IQ4_NL", iq4_nl_matvec),
        ("IQ4_XS", iq4_xs_matvec),
        ("MXFP4", mxfp4_matvec),
    ];

    /// Every shader in the table is exercised by [`LAUNCHES`], so a row
    /// added without a test cannot pass.
    #[test]
    fn every_shader_has_a_launch_under_test() {
        let tested: Vec<&str> = LAUNCHES.iter().map(|(k, _)| *k).collect();
        for s in crate::shaders::MATVEC {
            assert!(tested.contains(&s.kind), "{} has no test launch", s.kind);
        }
        assert_eq!(tested.len(), crate::shaders::MATVEC.len());
    }

    /// `rows` rows of `kind` and an activation, deterministic per shape.
    fn case(kind: &str, rows: usize, blocks: usize) -> (Vec<u8>, Vec<f32>, usize) {
        let shader = crate::shaders::matvec_shader(kind).unwrap();
        let row_bytes = blocks * shader.block_bytes;
        let mut w = Vec::with_capacity(rows * row_bytes);
        for r in 0..rows {
            w.extend(kind_row(kind, blocks, 31 * r as u64 + 7));
        }
        let x = pseudo_random(
            rows as u64 * 17 + blocks as u64,
            blocks * shader.block_elems,
        );
        (w, x, row_bytes)
    }

    fn assert_twin(kind: &str, w: &[u8], x: &[f32], got: &[f32], row_bytes: usize, what: &str) {
        for (r, g) in got.iter().enumerate() {
            let want = twin(kind)(&w[r * row_bytes..(r + 1) * row_bytes], x);
            let tol = 1e-4 * want.abs().max(1.0);
            assert!(
                (g - want).abs() <= tol,
                "{kind} {what} row {r}: gpu {g} vs twin {want}"
            );
        }
    }

    /// Every GLSL shader against its twin through one runtime, uploading
    /// weights the way `staging` says. Each shape runs TWICE: the second
    /// call is served from the resident buffer and the cached pipeline.
    fn every_shader_against_its_twin(staging: bool) {
        let Some(mut rt) = runtime() else { return };
        rt.force_staging(staging);
        for (kind, launch) in LAUNCHES {
            // One block; a row stride that is not word-aligned; more
            // units than a workgroup has invocations (so each sums
            // several); and fewer, leaving most of the group idle.
            for (rows, blocks) in [(1usize, 1usize), (9, 3), (65, 2), (130, 5), (3, 70)] {
                let (w, x, row_bytes) = case(kind, rows, blocks);
                for pass in ["upload", "resident"] {
                    let got = launch(&mut rt, &w, &x, rows, row_bytes, blocks).expect("dispatch");
                    assert_eq!(got.len(), rows);
                    assert_twin(
                        kind,
                        &w,
                        &x,
                        &got,
                        row_bytes,
                        &format!("({rows},{blocks}) {pass}"),
                    );
                }
            }
        }
    }

    /// **Every GLSL shader against its `frink_quant` scalar twin**, on
    /// whatever Vulkan device is present (MoltenVK on a Mac, RADV or
    /// AMDVLK on a Radeon), at shapes covering a row stride that is not
    /// word-aligned, a row count that is not a multiple of the 64-wide
    /// workgroup, and several workgroups. Skips loudly with no device.
    #[test]
    fn gpu_every_shader_matvec_matches_its_scalar_twin() {
        every_shader_against_its_twin(false);
    }

    /// The same, with every weight uploaded through a staging buffer and
    /// `vkCmdCopyBuffer` into `DEVICE_LOCAL` memory: the path a discrete
    /// Radeon without resizable BAR takes, which a unified-memory test
    /// device would otherwise never run.
    #[test]
    fn gpu_staged_weights_match_the_scalar_twin() {
        every_shader_against_its_twin(true);
    }

    /// The residency cache's identity rule. A weight is uploaded once and
    /// then served by address; but a different buffer at the same address
    /// and length (here: the same `Vec`, rewritten in place) must be
    /// re-uploaded, or the GPU multiplies by the previous tensor.
    #[test]
    fn gpu_a_rewritten_buffer_at_the_same_address_is_reuploaded() {
        let Some(mut rt) = runtime() else { return };
        let (mut w, x, row_bytes) = case("Q4_K", 9, 3);
        let first = q4_k_matvec(&mut rt, &w, &x, 9, row_bytes, 3).unwrap();
        q4_k_matvec(&mut rt, &w, &x, 9, row_bytes, 3).unwrap();
        assert_eq!(rt.resident_weights().0, 1, "one buffer, uploaded once");

        // Same address, same length, new bytes in the sampled head.
        let ptr = w.as_ptr();
        w[4..20].iter_mut().for_each(|b| *b = b.wrapping_add(0x5b));
        assert_eq!(ptr, w.as_ptr());
        let second = q4_k_matvec(&mut rt, &w, &x, 9, row_bytes, 3).unwrap();
        assert_ne!(first[0], second[0], "the test must change row 0's answer");
        assert_twin("Q4_K", &w, &x, &second, row_bytes, "rewritten");
        assert_eq!(
            rt.resident_weights().0,
            1,
            "the stale copy is replaced, not kept"
        );
    }

    /// **One submission, several kinds.** Q/K/V and gate/up go through
    /// [`Runtime::matvec_many`]: every output must be its own matvec's,
    /// at row counts (9, 65, 3, 130) whose byte sizes are not multiples
    /// of any storage-offset alignment, so a packing that ignores the
    /// alignment, or binds every output at offset 0, shows up as a wrong
    /// row. Run twice: the second submission is all resident.
    #[test]
    fn gpu_matvec_many_gives_every_job_its_own_answer() {
        let Some(mut rt) = runtime() else { return };
        let cols = 512;
        let x = pseudo_random(0xf05e, cols);
        let plan = [("Q4_K", 9usize), ("Q6_K", 65), ("Q8_0", 3), ("Q5_K", 130)];
        let weights: Vec<(&str, Vec<u8>, usize, usize)> = plan
            .iter()
            .map(|&(kind, rows)| {
                let s = crate::shaders::matvec_shader(kind).unwrap();
                let blocks = cols / s.block_elems;
                let w: Vec<u8> = (0..rows)
                    .flat_map(|r| kind_row(kind, blocks, 97 * r as u64 + rows as u64))
                    .collect();
                (kind, w, rows, blocks * s.block_bytes)
            })
            .collect();
        let jobs: Vec<crate::runtime::MatvecJob<'_>> = weights
            .iter()
            .map(|(kind, w, rows, _)| crate::runtime::MatvecJob {
                shader: crate::shaders::matvec_shader(kind).unwrap(),
                weights: w,
                rows: *rows,
            })
            .collect();
        for pass in ["upload", "resident"] {
            let outs = rt.matvec_many(&jobs, &x).expect("fused dispatch");
            assert_eq!(outs.len(), jobs.len());
            for ((kind, w, rows, row_bytes), got) in weights.iter().zip(&outs) {
                assert_eq!(got.len(), *rows);
                assert_twin(kind, w, &x, got, *row_bytes, &format!("fused {pass}"));
            }
        }

        // A shape that does not fit is an error the caller can fall back
        // from, not a panic in a rayon worker.
        let bad = crate::runtime::MatvecJob {
            shader: crate::shaders::matvec_shader("Q4_K").unwrap(),
            weights: &weights[0].1[..100],
            rows: 9,
        };
        assert!(rt.matvec_many(&[bad], &x).is_err());
        assert!(rt.matvec_many(&[], &x).is_err());
        assert!(rt
            .matvec_many(&[jobs[0]; crate::runtime::MAX_FUSED + 1], &x)
            .is_err());
    }
}
