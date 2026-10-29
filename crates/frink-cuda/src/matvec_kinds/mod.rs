//! The per-quant-kind matvec table: which formats CUDA can DECODE, and
//! the three strings a launch needs for each.
//!
//! This is the decode counterpart of [`crate::mul_mm::KINDS`], and it
//! is a table for the same reason. The same five rows were once written
//! out three times -- here, inline in `frink-core`'s `apply_gpu_multi`
//! and again in `apply_gpu_dense_ffn_swiglu` -- and a kind added to one
//! and not the others silently lost the fused launch while the
//! capability report kept saying GPU.
//!
//! # Always compiled
//!
//! Deliberately outside the `cuda` feature gate, like `mul_mm`. The
//! rows are CUDA C *text* and three `&'static str`s; nothing here needs
//! `cudarc`. That buys two things: `cargo test -p frink-cuda` on a
//! GPU-less host still checks that every row names an entry point its
//! source defines and that every embedded codebook agrees with the
//! GEMM's, and `frink-core` can ask "does CUDA decode this kind?" on
//! a build with no CUDA feature at all -- which is what lets
//! `Cuda::matvec_kernel` DERIVE from this table instead of restating
//! it. Restating it is how IQ4_XS ended up "supported" with no kernel.
//!
//! The launch itself ([`crate::gpu::matvec_launch_meta`] and the
//! `launch_*_matvec` functions) stays feature-gated.

pub mod coalesced;
pub mod codebook;
pub mod kquant;
pub mod legacy;

/// One quantized weight format the CUDA matvec path can consume.
///
/// The `__global__` entry point named by [`Self::fn_name`] must have
/// the signature every kernel in this directory shares:
///
/// ```text
/// void <fn_name>(const unsigned char* weights, const float* x,
///                float* out, int rows, int row_bytes,
///                int n_blocks_per_row)
/// ```
///
/// one threadblock per output row, 256 threads striding the row's
/// blocks, a tree reduction into `out[row]`. `launch_matvec` supplies
/// exactly that geometry, so a kernel written to a different one
/// returns wrong numbers rather than failing.
#[derive(Debug, Clone, Copy)]
pub struct MatvecKind {
    /// GGUF quant name -- the key `frink-core` looks a kind up by,
    /// which is `QuantKind::name()`.
    pub name: &'static str,
    /// NVRTC module cache key. Must be unique per kind.
    pub module_name: &'static str,
    /// The `__global__` entry point inside that module.
    pub fn_name: &'static str,
    /// The complete translation unit defining it.
    pub src: &'static str,
}

/// The dispatch table. A new format is one row here and nothing else.
///
/// Order is Q8_0, Q4_0, Q5_0, then the K-quants, then the codebook
/// kinds, matching [`crate::mul_mm::KINDS`] so the two read as the same
/// list -- which is what
/// `frink_core`'s `a_cuda_kind_with_a_matvec_also_has_a_gemm` requires
/// them to be.
pub const KINDS: &[MatvecKind] = &[
    MatvecKind {
        name: "Q8_0",
        module_name: "frink_q8_0",
        fn_name: "q8_0_matvec",
        src: legacy::Q8_0_MATVEC_KERNEL_SRC,
    },
    MatvecKind {
        name: "Q4_0",
        module_name: "frink_q4_0",
        fn_name: "q4_0_matvec",
        src: legacy::Q4_0_MATVEC_KERNEL_SRC,
    },
    MatvecKind {
        name: "Q5_0",
        module_name: "frink_q5_0",
        fn_name: "q5_0_matvec",
        src: legacy::Q5_0_MATVEC_KERNEL_SRC,
    },
    MatvecKind {
        name: "Q2_K",
        module_name: "frink_q2_k",
        fn_name: "q2_k_matvec",
        src: kquant::Q2_K_MATVEC_KERNEL_SRC,
    },
    MatvecKind {
        name: "Q3_K",
        module_name: "frink_q3_k",
        fn_name: "q3_k_matvec",
        src: kquant::Q3_K_MATVEC_KERNEL_SRC,
    },
    MatvecKind {
        name: "Q4_K",
        module_name: "frink_q4_k",
        fn_name: "q4_k_matvec",
        src: kquant::Q4_K_MATVEC_KERNEL_SRC,
    },
    MatvecKind {
        name: "Q5_K",
        module_name: "frink_q5_k",
        fn_name: "q5_k_matvec",
        src: kquant::Q5_K_MATVEC_KERNEL_SRC,
    },
    MatvecKind {
        name: "Q6_K",
        module_name: "frink_q6_k",
        fn_name: "q6_k_matvec",
        src: kquant::Q6_K_MATVEC_KERNEL_SRC,
    },
    MatvecKind {
        name: "IQ4_NL",
        module_name: "frink_iq4_nl",
        fn_name: "iq4_nl_matvec",
        src: codebook::IQ4_NL_MATVEC_KERNEL_SRC,
    },
    MatvecKind {
        name: "IQ4_XS",
        module_name: "frink_iq4_xs",
        fn_name: "iq4_xs_matvec",
        src: codebook::IQ4_XS_MATVEC_KERNEL_SRC,
    },
    MatvecKind {
        name: "MXFP4",
        module_name: "frink_mxfp4",
        fn_name: "mxfp4_matvec",
        src: codebook::MXFP4_MATVEC_KERNEL_SRC,
    },
];

/// Looks up a kind by its GGUF quant name. `None` means CUDA has no
/// matvec for that format and the caller must fall back and say so,
/// never compute something else.
pub fn kind_by_name(name: &str) -> Option<&'static MatvecKind> {
    KINDS.iter().find(|k| k.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row whose `fn_name` its source does not define is a
    /// `KernelCompile` error at a user's first token. Runnable without
    /// a device, because that failure needs no GPU to see.
    #[test]
    fn every_row_names_an_entry_point_its_source_defines() {
        for k in KINDS {
            assert!(
                k.src.contains(&format!("void {}(", k.fn_name)),
                "{}: the source in {} does not define {}",
                k.name,
                k.module_name,
                k.fn_name
            );
            assert!(
                k.src.contains("int n_blocks_per_row"),
                "{}: does not take the launch geometry every caller supplies",
                k.name
            );
        }
        for (i, a) in KINDS.iter().enumerate() {
            for b in &KINDS[i + 1..] {
                assert_ne!(
                    a.module_name, b.module_name,
                    "{} and {} collide in the process-wide NVRTC module cache",
                    a.name, b.name
                );
                assert_ne!(a.fn_name, b.fn_name, "{} vs {}", a.name, b.name);
            }
        }
    }

    /// A kind with no kernel must not resolve. Resolving sends a decode
    /// to a module that cannot compile, and the caller has no way to
    /// fall back honestly.
    #[test]
    fn a_kind_with_no_matvec_does_not_resolve() {
        for absent in ["Q5_1", "Q4_1", "Q8_1", "IQ1_S", "IQ2_XXS", "IQ3_S"] {
            assert!(
                kind_by_name(absent).is_none(),
                "{absent} resolved to a CUDA matvec that does not exist"
            );
        }
    }

    /// Every matvec kernel strides its row by a byte count written as
    /// a LITERAL in CUDA C (`row_ptr + (size_t)b * 84`), and that
    /// literal has to be the block size the format actually has.
    ///
    /// A wrong stride walks the row past the first block and every
    /// value after it is garbage -- the failure Q5_0 got a bespoke test
    /// for on 2026-09-05 (`the_q5_0_matvec_strides_by_the_real_block_
    /// geometry`), one kind at a time. This is that test for the whole
    /// table, driven from [`crate::mul_mm::KINDS`], which is where the
    /// GEMM takes the same number from. Five kinds landed on
    /// 2026-09-09 and none of them would have had one otherwise.
    ///
    /// The activation step is checked the same way: a kernel that steps
    /// the INPUT by the wrong count pairs every quant after the first
    /// block with the wrong activation.
    ///
    /// Matched per line rather than on one exact spelling, because the
    /// six kernels that predate this test write the same offset three
    /// ways (`b * 34`, `(size_t)b * 22`, `blk * 144`). Normalising them
    /// is a separate change; a test that only accepted one spelling
    /// would have to be weakened or would fail on code that is correct.
    ///
    /// Sabotage: change any stride or activation-step literal in a
    /// kernel source and this names the kind.
    #[test]
    fn every_matvec_strides_by_the_real_block_geometry() {
        for k in KINDS {
            let mm = crate::mul_mm::kind_by_name(k.name)
                .unwrap_or_else(|| panic!("{}: a matvec with no mul_mm row", k.name));

            let row_step = k
                .src
                .lines()
                .find(|l| l.contains("row_ptr +"))
                .unwrap_or_else(|| panic!("{}: no row-pointer arithmetic", k.name));
            assert!(
                row_step.contains(&format!("* {};", mm.block_bytes)),
                "{}: strides the row by something other than {} bytes: {}",
                k.name,
                mm.block_bytes,
                row_step.trim()
            );

            let x_step = k
                .src
                .lines()
                .find(|l| l.contains("base = ") && l.trim_end().ends_with(';'))
                .unwrap_or_else(|| panic!("{}: no activation-base arithmetic", k.name));
            assert!(
                x_step.contains(&format!("* {};", mm.block_elems)),
                "{}: steps the activation by something other than {} elements: {}",
                k.name,
                mm.block_elems,
                x_step.trim()
            );
        }
    }

    /// The codebook a matvec kernel embeds as a literal has to be the
    /// codebook the GEMM emits from its [`Codebook`] row.
    ///
    /// Two structures that must agree about sixteen arbitrary numbers,
    /// with the emitted half generated and the literal half typed by
    /// hand. Nothing else would catch a transposed pair: every value is
    /// plausible, every tensor would still decode, and only the
    /// arithmetic would be quietly wrong.
    ///
    /// The numbers are parsed back out of the kernel text rather than
    /// compared to a second copy of the formatting, so this fails on a
    /// bad literal and not merely on a bad `format!`.
    ///
    /// Sabotage: change one entry of either literal below and this
    /// names the kind and the index.
    #[test]
    fn every_embedded_codebook_is_the_mul_mm_codebook() {
        let mut checked = 0usize;
        for mm in crate::mul_mm::KINDS {
            let Some(cb) = mm.codebook else { continue };
            let Some(mv) = kind_by_name(mm.name) else {
                panic!(
                    "{}: has a mul_mm codebook kernel but no matvec, which is the \
                     split that decomposes a prefill into one launch per position",
                    mm.name
                );
            };
            let decl = format!("__constant__ float {}[16] = {{", cb.c_name);
            let at = mv.src.find(&decl).unwrap_or_else(|| {
                panic!("{}: the matvec does not declare {}", mm.name, cb.c_name)
            });
            let body = &mv.src[at + decl.len()..];
            let body = &body[..body.find('}').expect("unterminated codebook")];
            let got: Vec<f32> = body
                .split(',')
                .map(|t| {
                    t.trim()
                        .trim_end_matches('f')
                        .parse::<f32>()
                        .unwrap_or_else(|e| panic!("{}: {t:?}: {e}", mm.name))
                })
                .collect();
            assert_eq!(got.len(), 16, "{}: codebook is not 16 entries", mm.name);
            for (i, (g, w)) in got.iter().zip(cb.values.iter()).enumerate() {
                // Bit comparison: MXFP4's code 8 is negative zero, and
                // `-0.0 == 0.0` would let a sign flip through.
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{}: codebook entry {i}: matvec has {g}, mul_mm emits {w}",
                    mm.name
                );
            }
            checked += 1;
        }
        assert!(checked >= 3, "the codebook kinds stopped being checked");
    }

    /// A kind that can be prefilled on the GPU but not decoded there
    /// (or the reverse) splits a forward pass across two devices, which
    /// is the shape that cost Metal a Q5_0 decode path and CUDA a 325x
    /// K-quant prefill.
    ///
    /// `frink-core` asserts the same thing over `QuantKind`; this
    /// asserts it over the two kernel tables themselves, so it holds
    /// even for a format `QuantKind` has no variant for.
    #[test]
    fn the_matvec_table_and_the_mul_mm_table_name_the_same_kinds() {
        let mm: Vec<&str> = crate::mul_mm::KINDS.iter().map(|k| k.name).collect();
        let mv: Vec<&str> = KINDS.iter().map(|k| k.name).collect();
        assert_eq!(mm, mv, "the CUDA decode and prefill tables have diverged");
    }
}

/// The on-device check for every matvec kernel, in one loop.
///
/// Gated on `cuda` because it calls the launchers, and `#[ignore]`d
/// because it needs a device. It is deliberately ONE test over
/// [`KINDS`] rather than one per kernel: `gpu.rs` grew a hand-written
/// hardware test per kind, and the kinds that arrived later
/// (`Q5_0`, and everything added on 2026-09-09) each needed someone to
/// remember. A kind added to the table is on this list the moment it
/// exists.
#[cfg(all(test, feature = "cuda"))]
mod hardware_tests {
    /// Every CUDA matvec against `frink_quant`'s fused dot, on weights
    /// built by the shared `mul_mm` fixtures -- the same bytes the GEMM
    /// twin and the GEMM hardware test use, so a disagreement between
    /// decode and prefill shows up as one of them failing on data the
    /// other passed.
    ///
    /// Run on a machine with an actual CUDA device:
    ///   cargo test -p frink-cuda --features cuda -- --ignored
    #[test]
    #[ignore = "requires real CUDA hardware. Q2_K, Q3_K, IQ4_NL, IQ4_XS and MXFP4 have NEVER executed on a GPU; Q5_0 has not either. Run with --ignored on a CUDA-capable machine and record the result before any doc claims those kinds decode on CUDA"]
    fn every_cuda_matvec_matches_the_cpu_reference() {
        type Dot = fn(&[u8], &[f32]) -> f32;
        type Launch =
            fn(&[u8], &[f32], usize, usize, usize) -> Result<Vec<f32>, crate::gpu::CudaError>;

        // Keyed by name and checked to cover `KINDS`, so a kernel
        // without a CPU oracle fails here instead of shipping unchecked.
        let oracles: &[(&str, Dot, Launch)] = &[
            (
                "Q8_0",
                frink_quant::dot_q8_0_f32_scalar,
                crate::gpu::launch_q8_0_matvec,
            ),
            (
                "Q4_0",
                frink_quant::dot_q4_0_f32_scalar,
                crate::gpu::launch_q4_0_matvec,
            ),
            (
                "Q5_0",
                frink_quant::dot_q5_0_f32_scalar,
                crate::gpu::launch_q5_0_matvec,
            ),
            (
                "Q2_K",
                frink_quant::dot_q2_k_f32_scalar,
                crate::gpu::launch_q2_k_matvec,
            ),
            (
                "Q3_K",
                frink_quant::dot_q3_k_f32_scalar,
                crate::gpu::launch_q3_k_matvec,
            ),
            (
                "Q4_K",
                frink_quant::dot_q4_k_f32_scalar,
                crate::gpu::launch_q4_k_matvec,
            ),
            (
                "Q5_K",
                frink_quant::dot_q5_k_f32_scalar,
                crate::gpu::launch_q5_k_matvec,
            ),
            (
                "Q6_K",
                frink_quant::dot_q6_k_f32_scalar,
                crate::gpu::launch_q6_k_matvec,
            ),
            (
                "IQ4_NL",
                frink_quant::dot_iq4_nl_f32_scalar,
                crate::gpu::launch_iq4_nl_matvec,
            ),
            (
                "IQ4_XS",
                frink_quant::dot_iq4_xs_f32_scalar,
                crate::gpu::launch_iq4_xs_matvec,
            ),
            (
                "MXFP4",
                frink_quant::dot_mxfp4_gguf_f32_scalar,
                crate::gpu::launch_mxfp4_matvec,
            ),
        ];

        for mm in crate::mul_mm::KINDS {
            let (_, dot, launch) = oracles
                .iter()
                .find(|(name, _, _)| *name == mm.name)
                .unwrap_or_else(|| panic!("{}: a CUDA matvec with no CPU oracle", mm.name));

            let rows = 7;
            // Two whole super-blocks, so a kernel that strides the row
            // wrongly is visible rather than reading one block twice.
            let cols = mm.block_elems * 2;
            let blocks_per_row = cols / mm.block_elems;
            let row_bytes = blocks_per_row * mm.block_bytes;
            let weights = crate::mul_mm_ref::fixtures::weights(mm, rows, cols, 31337);
            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.09).sin()).collect();

            let expected: Vec<f32> = (0..rows)
                .map(|r| dot(&weights[r * row_bytes..(r + 1) * row_bytes], &x))
                .collect();
            let got = launch(&weights, &x, rows, row_bytes, blocks_per_row)
                .expect("kernel launch must succeed on real CUDA hardware");

            assert_eq!(got.len(), expected.len());
            for (i, (g, w)) in got.iter().zip(expected.iter()).enumerate() {
                // Relative, not absolute. Several of these kernels
                // factor the scale out of the inner loop where
                // `frink_quant`'s scalar dot multiplies it in per
                // element, so the two reassociate differently over 256
                // terms; an absolute 1e-2 would be a coin flip on a
                // large scale.
                let scale = w.abs().max(1.0);
                assert!(
                    (g - w).abs() <= 1e-3 * scale,
                    "{} row {i}: GPU={g} CPU reference={w}",
                    mm.name
                );
            }
        }
    }
}
