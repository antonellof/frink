//! Which quant kinds the Metal embedding gather (`get_rows`) serves.
//!
//! Compiled on every build, unlike the kernels in [`crate::embd`]: the
//! set is a property of the kernel source, not of the build, and the
//! kernel coverage matrix (`frink_core::kernel_matrix`, #405) has to
//! read it from a CPU-only `cargo test --workspace`. The decode path
//! reads the same rows through [`EmbdKind::from_fn_name`], so the matrix
//! and the dispatch cannot name different sets.

/// One `get_rows_*` kernel in [`crate::embd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbdKind {
    Q4K,
    Q6K,
}

impl EmbdKind {
    /// Every variant. `every_variant_is_listed_once` holds this to the
    /// exhaustive matches below.
    pub const ALL: [EmbdKind; 2] = [EmbdKind::Q4K, EmbdKind::Q6K];

    /// The GGUF name of the stored type, as `frink_core::QuantKind::name`
    /// spells it.
    pub fn kind_name(self) -> &'static str {
        match self {
            EmbdKind::Q4K => "Q4_K",
            EmbdKind::Q6K => "Q6_K",
        }
    }

    /// The matvec kernel the decode path resolved for the embedding
    /// matrix, which is how it asks whether a gather exists.
    pub fn matvec_fn_name(self) -> &'static str {
        match self {
            EmbdKind::Q4K => "q4_k_matvec",
            EmbdKind::Q6K => "q6_k_matvec",
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
                EmbdKind::Q4K | EmbdKind::Q6K => {}
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
        assert_eq!(EmbdKind::from_kind_name("Q8_0"), None);
        assert_eq!(EmbdKind::from_fn_name("q8_0_matvec"), None);
    }
}
