//! Which quant kinds the Metal embedding gather (`get_rows`) serves.
//!
//! Compiled on every build, unlike the kernels in [`crate::embd`]: the
//! set is a property of the kernel source, not of the build, and the
//! kernel coverage matrix (`frink_core::kernel_matrix`, #405) has to
//! read it from a CPU-only `cargo test --workspace`. The decode path
//! reads the same rows through [`EmbdKind::from_fn_name`], so the matrix
//! and the dispatch cannot name different sets.
//!
//! Every kind with a Metal matvec kernel has a gather. Each gather is
//! the GEMM's 16-value dequant functor for its kind (ggml's
//! `kernel_get_rows_q<block_q, nl, dequantize_func>`), so a kind's
//! dequant is written once; [`EmbdKind::dequant_functor`] names it and
//! [`crate::embd`] generates one entry point per row of [`EmbdKind::ALL`].

/// One `get_rows_*` kernel in [`crate::embd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbdKind {
    Q8_0,
    Q4_0,
    Q5_0,
    Q4K,
    Q5K,
    Q6K,
    Iq4Xs,
    Ptq1_0,
    Q2K,
    Q3K,
    Iq4Nl,
    Mxfp4,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
    Iq1M,
    Q4_1,
    Q5_1,
    Tq1_0,
}

impl EmbdKind {
    /// Every variant. `every_variant_is_listed_once` holds this to the
    /// exhaustive matches below.
    pub const ALL: [EmbdKind; 22] = [
        EmbdKind::Q8_0,
        EmbdKind::Q4_0,
        EmbdKind::Q5_0,
        EmbdKind::Q4K,
        EmbdKind::Q5K,
        EmbdKind::Q6K,
        EmbdKind::Iq4Xs,
        EmbdKind::Ptq1_0,
        EmbdKind::Q2K,
        EmbdKind::Q3K,
        EmbdKind::Iq4Nl,
        EmbdKind::Mxfp4,
        EmbdKind::Iq2Xxs,
        EmbdKind::Iq2Xs,
        EmbdKind::Iq2S,
        EmbdKind::Iq3Xxs,
        EmbdKind::Iq3S,
        EmbdKind::Iq1S,
        EmbdKind::Iq1M,
        EmbdKind::Q4_1,
        EmbdKind::Q5_1,
        EmbdKind::Tq1_0,
    ];

    /// The GGUF name of the stored type, as `frink_core::QuantKind::name`
    /// spells it.
    pub fn kind_name(self) -> &'static str {
        match self {
            EmbdKind::Q8_0 => "Q8_0",
            EmbdKind::Q4_0 => "Q4_0",
            EmbdKind::Q5_0 => "Q5_0",
            EmbdKind::Q4K => "Q4_K",
            EmbdKind::Q5K => "Q5_K",
            EmbdKind::Q6K => "Q6_K",
            EmbdKind::Iq4Xs => "IQ4_XS",
            EmbdKind::Ptq1_0 => "PTQ1_0",
            EmbdKind::Q2K => "Q2_K",
            EmbdKind::Q3K => "Q3_K",
            EmbdKind::Iq4Nl => "IQ4_NL",
            EmbdKind::Mxfp4 => "MXFP4",
            EmbdKind::Iq2Xxs => "IQ2_XXS",
            EmbdKind::Iq2Xs => "IQ2_XS",
            EmbdKind::Iq2S => "IQ2_S",
            EmbdKind::Iq3Xxs => "IQ3_XXS",
            EmbdKind::Iq3S => "IQ3_S",
            EmbdKind::Iq1S => "IQ1_S",
            EmbdKind::Iq1M => "IQ1_M",
            EmbdKind::Q4_1 => "Q4_1",
            EmbdKind::Q5_1 => "Q5_1",
            EmbdKind::Tq1_0 => "TQ1_0",
        }
    }

    /// The matvec kernel the decode path resolved for the embedding
    /// matrix, which is how it asks whether a gather exists.
    pub fn matvec_fn_name(self) -> &'static str {
        match self {
            EmbdKind::Q8_0 => "q8_0_matvec",
            EmbdKind::Q4_0 => "q4_0_matvec",
            EmbdKind::Q5_0 => "q5_0_matvec",
            EmbdKind::Q4K => "q4_k_matvec",
            EmbdKind::Q5K => "q5_k_matvec",
            EmbdKind::Q6K => "q6_k_matvec",
            EmbdKind::Iq4Xs => "iq4_xs_matvec",
            EmbdKind::Ptq1_0 => "ptq1_0_matvec",
            EmbdKind::Q2K => "q2_k_matvec",
            EmbdKind::Q3K => "q3_k_matvec",
            EmbdKind::Iq4Nl => "iq4_nl_matvec",
            EmbdKind::Mxfp4 => "mxfp4_matvec",
            EmbdKind::Iq2Xxs => "iq2_xxs_matvec",
            EmbdKind::Iq2Xs => "iq2_xs_matvec",
            EmbdKind::Iq2S => "iq2_s_matvec",
            EmbdKind::Iq3Xxs => "iq3_xxs_matvec",
            EmbdKind::Iq3S => "iq3_s_matvec",
            EmbdKind::Iq1S => "iq1_s_matvec",
            EmbdKind::Iq1M => "iq1_m_matvec",
            EmbdKind::Q4_1 => "q4_1_matvec",
            EmbdKind::Q5_1 => "q5_1_matvec",
            EmbdKind::Tq1_0 => "tq1_0_matvec",
        }
    }

    /// The gather's entry point in the GEMM translation unit.
    pub fn get_rows_fn_name(self) -> &'static str {
        match self {
            EmbdKind::Q8_0 => "get_rows_q8_0",
            EmbdKind::Q4_0 => "get_rows_q4_0",
            EmbdKind::Q5_0 => "get_rows_q5_0",
            EmbdKind::Q4K => "get_rows_q4_k",
            EmbdKind::Q5K => "get_rows_q5_k",
            EmbdKind::Q6K => "get_rows_q6_k",
            EmbdKind::Iq4Xs => "get_rows_iq4_xs",
            EmbdKind::Ptq1_0 => "get_rows_ptq1_0",
            EmbdKind::Q2K => "get_rows_q2_k",
            EmbdKind::Q3K => "get_rows_q3_k",
            EmbdKind::Iq4Nl => "get_rows_iq4_nl",
            EmbdKind::Mxfp4 => "get_rows_mxfp4",
            EmbdKind::Iq2Xxs => "get_rows_iq2_xxs",
            EmbdKind::Iq2Xs => "get_rows_iq2_xs",
            EmbdKind::Iq2S => "get_rows_iq2_s",
            EmbdKind::Iq3Xxs => "get_rows_iq3_xxs",
            EmbdKind::Iq3S => "get_rows_iq3_s",
            EmbdKind::Iq1S => "get_rows_iq1_s",
            EmbdKind::Iq1M => "get_rows_iq1_m",
            EmbdKind::Q4_1 => "get_rows_q4_1",
            EmbdKind::Q5_1 => "get_rows_q5_1",
            EmbdKind::Tq1_0 => "get_rows_tq1_0",
        }
    }

    /// The MSL dequant functor the kind's simdgroup GEMM instantiates
    /// (`gpu::K_QUANT_MUL_MM_SG_KERNEL_SRC`), which the gather reuses.
    pub fn dequant_functor(self) -> &'static str {
        match self {
            EmbdKind::Q8_0 => "Q8_0Dequant",
            EmbdKind::Q4_0 => "Q4_0Dequant",
            EmbdKind::Q5_0 => "Q5_0Dequant",
            EmbdKind::Q4K => "Q4KDequant",
            EmbdKind::Q5K => "Q5KDequant",
            EmbdKind::Q6K => "Q6KDequant",
            EmbdKind::Iq4Xs => "IQ4XSDequant",
            EmbdKind::Ptq1_0 => "PTQ1_0Dequant",
            EmbdKind::Q2K => "Q2KDequant",
            EmbdKind::Q3K => "Q3KDequant",
            EmbdKind::Iq4Nl => "IQ4NLDequant",
            EmbdKind::Mxfp4 => "MXFP4Dequant",
            EmbdKind::Iq2Xxs => "IQ2XXSDequant",
            EmbdKind::Iq2Xs => "IQ2XSDequant",
            EmbdKind::Iq2S => "IQ2SDequant",
            EmbdKind::Iq3Xxs => "IQ3XXSDequant",
            EmbdKind::Iq3S => "IQ3SDequant",
            EmbdKind::Iq1S => "IQ1SDequant",
            EmbdKind::Iq1M => "IQ1MDequant",
            EmbdKind::Q4_1 => "Q4_1Dequant",
            EmbdKind::Q5_1 => "Q5_1Dequant",
            EmbdKind::Tq1_0 => "TQ1_0Dequant",
        }
    }

    /// The gather for the kind whose matvec kernel is `fn_name`.
    pub fn from_fn_name(fn_name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|k| k.matvec_fn_name() == fn_name)
    }

    /// The gather for the GGUF type named `name`.
    pub fn from_kind_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.kind_name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ALL` must hold each variant once, or a gather exists that
    /// neither lookup can find.
    #[test]
    fn every_variant_is_listed_once() {
        for k in EmbdKind::ALL {
            let n = EmbdKind::ALL.iter().filter(|&&o| o == k).count();
            assert_eq!(n, 1, "{k:?}");
            // The exhaustive match: a new variant must answer here.
            match k {
                EmbdKind::Q8_0
                | EmbdKind::Q4_0
                | EmbdKind::Q5_0
                | EmbdKind::Q4K
                | EmbdKind::Q5K
                | EmbdKind::Q6K
                | EmbdKind::Iq4Xs
                | EmbdKind::Ptq1_0
                | EmbdKind::Q2K
                | EmbdKind::Q3K
                | EmbdKind::Iq4Nl
                | EmbdKind::Mxfp4
                | EmbdKind::Iq2Xxs
                | EmbdKind::Iq2Xs
                | EmbdKind::Iq2S
                | EmbdKind::Iq3Xxs
                | EmbdKind::Iq3S
                | EmbdKind::Iq1S
                | EmbdKind::Iq1M
                | EmbdKind::Q4_1
                | EmbdKind::Q5_1
                | EmbdKind::Tq1_0 => {}
            }
        }
    }

    /// The two lookups name the same set, so the matrix (by kind name)
    /// and the decode path (by matvec name) agree.
    #[test]
    fn both_lookups_find_every_gather() {
        for k in EmbdKind::ALL {
            assert_eq!(EmbdKind::from_kind_name(k.kind_name()), Some(k));
            assert_eq!(EmbdKind::from_fn_name(k.matvec_fn_name()), Some(k));
        }
        assert_eq!(EmbdKind::from_kind_name("Q8_1"), None);
        assert_eq!(EmbdKind::from_fn_name("f32_matvec"), None);
    }

    /// Entry-point names are distinct, or two gathers would share a
    /// cached pipeline (the cache keys on the function name).
    #[test]
    fn entry_points_are_distinct() {
        for a in EmbdKind::ALL {
            for b in EmbdKind::ALL {
                if a != b {
                    assert_ne!(a.get_rows_fn_name(), b.get_rows_fn_name());
                }
            }
        }
    }
}
