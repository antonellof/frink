//! `{arch}.attention.clamp_kqv`, for the architectures whose graph
//! applies it.
//!
//! `LLM_KV_ATTENTION_CLAMP_KQV` (`llama-arch.cpp:231`) is a symmetric
//! clamp on the Q, K and V projections, applied INSIDE the shared
//! `build_qkv` rather than in any one architecture's graph:
//!
//! ```text
//! // llama-graph.cpp:1611-1612, and again at :1631, :1641, :1651
//! if (hparams.f_clamp_kqv > 0.0f) {
//!     qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
//! }
//! ```
//!
//! The fused branch clamps the fused projection once; the split branch
//! clamps `Qcur`, `Kcur` and `Vcur` separately. Either way it happens
//! AFTER the bias add and before the reshape, before any QK-norm and
//! before RoPE. `build_qkv` clamps for ANY architecture whose
//! `f_clamp_kqv` is positive, but only three `load_arch_hparams` read
//! the key, so for every other architecture the key is dead metadata
//! and frink ignores it the same way.
//!
//! **Three architectures read the key**, and no other: `dbrx.cpp:5`
//! (REQUIRED -- no `false` argument, so a DBRX file without the key is
//! refused by llama.cpp's loader), `mpt.cpp:5` and `olmo.cpp:5` (both
//! optional). `mpt` never reaches the generic decoder (ALiBi), so this
//! module serves `olmo` and `dbrx`. It is live for both: the DBRX
//! converter writes `attn_config.clip_qkv` unconditionally
//! (`conversion/dbrx.py:28`, and every DBRX checkpoint sets it to 8),
//! and `conversion/olmo.py:23-25` writes the key whenever the HF config
//! has a `clip_qkv`, which OLMo-7B-Twin-2T and OLMo-1.7-7B do (`8.0`)
//! and the original OLMo-7B does not (`null`).
//!
//! **This used to be a refusal**, and the reason it was one is the
//! reason for the shape of the implementation. The projections are
//! computed in the CPU decode body, the prefill body and the
//! continuous-batching body, and each of those applied the QKV bias in
//! its own hand-written loop; a clamp added to some of them and not the
//! others would have been this repo's single most expensive defect
//! shape. So the three bias loops collapsed onto ONE helper,
//! `Decoder::apply_qkv_bias_and_clamp` (`decoder/qkv_bias.rs`), and the
//! clamp lives in that helper after the bias, where `build_qkv` puts
//! it. The fused Metal launches apply the bias inside their kernels via
//! `AttnExtras` and have no clamp, so `Decoder::metal_can_serve_model`
//! -- the one predicate every Metal eligibility check reads -- keeps a
//! clamped model on the host bodies rather than letting two backends
//! answer differently from the same weights.
//!
//! The evidence is `tests/olmo_graphs.rs`
//! (`olmo_clamped_tiny.gguf`, whose llama.cpp logits differ measurably
//! from the unclamped file's) and `tests/dbrx_graphs.rs`.

/// Why a file's clamp declaration cannot be honoured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClampError {
    /// The architecture reads the key as REQUIRED and the file has none.
    MissingRequired,
}

impl ClampError {
    /// The sentence the loader puts in its error, naming the key.
    pub fn message(&self, arch: &str) -> String {
        match self {
            ClampError::MissingRequired => format!(
                "`{arch}.attention.clamp_kqv` is REQUIRED for this architecture \
                 (src/models/dbrx.cpp:5 reads it with no default) and the file does not \
                 declare it; llama.cpp refuses the same file"
            ),
        }
    }
}

/// The clamp the decoder applies for `arch`, from what the file
/// declares (`None` when the key is absent).
///
/// `Ok(None)` is "no clamp": an architecture whose graph never clamps,
/// or a file declaring llama.cpp's own `0.0f` default or a negative
/// value -- `> 0.0f` is llama.cpp's test (llama-graph.cpp:1611), so
/// zero and a negative value both mean "no clamp" there and must mean
/// it here. Reading zero as a clamp would zero every projection.
pub fn resolve_clamp(arch: &str, declared: Option<f32>) -> Result<Option<f32>, ClampError> {
    resolve_clamp_for(
        crate::models::ArchFacts::of(arch).scaling.clamp_kqv,
        declared,
    )
}

/// The row: `None` where the graph reads no clamp, `Some(required)`
/// where it does, from [`CLAMPED_QKV_ARCHITECTURES`] alone
/// (`ArchFacts::from_tables`).
pub(crate) fn clamp_row_from_table(_arch: &str) -> Option<bool> {
    // No table: `olmo`, `dbrx` and `mpt` state theirs in `crate::models`.
    None
}

/// [`resolve_clamp`] given the row (`ArchFacts`).
pub fn resolve_clamp_for(
    row: Option<bool>,
    declared: Option<f32>,
) -> Result<Option<f32>, ClampError> {
    let Some(required) = row else {
        return Ok(None);
    };
    if declared.is_none() && required {
        return Err(ClampError::MissingRequired);
    }
    Ok(declared.filter(|v| *v > 0.0))
}

/// `ggml_clamp(x, -c, c)`, in place, over a slice that may hold one row
/// or a whole batch: the clamp is elementwise, so the two are the same
/// call.
#[inline]
pub fn clamp_in_place(x: &mut [f32], c: f32) {
    for v in x.iter_mut() {
        *v = v.clamp(-c, c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The resolution fires on exactly the values llama.cpp branches on.
    ///
    /// The absent and zero cases are the ones worth pinning: llama.cpp's
    /// default is `0.0f` and its test is `> 0.0f`, so treating either as
    /// a clamp would zero every projection of the original OLMo-7B,
    /// whose `clip_qkv` is null and whose converted file carries no key.
    #[test]
    fn only_a_positive_clamp_is_applied() {
        assert_eq!(resolve_clamp("olmo", Some(8.0)), Ok(Some(8.0)));
        assert_eq!(
            resolve_clamp("olmo", None),
            Ok(None),
            "an absent key is llama.cpp's 0.0 default: no clamp"
        );
        assert_eq!(
            resolve_clamp("olmo", Some(0.0)),
            Ok(None),
            "`> 0.0f` is llama.cpp's own test; zero means no clamp"
        );
        assert_eq!(
            resolve_clamp("olmo", Some(-1.0)),
            Ok(None),
            "a negative clamp is not a clamp in llama.cpp either"
        );
    }

    /// DBRX reads the key as REQUIRED, and only DBRX.
    ///
    /// `Some(0.0)` is still "no clamp" there: the key is present, which
    /// is all `get_key` checks, and the graph's `> 0.0f` then skips it.
    #[test]
    fn dbrx_refuses_a_file_that_omits_the_key_and_olmo_does_not() {
        assert_eq!(
            resolve_clamp("dbrx", None),
            Err(ClampError::MissingRequired)
        );
        assert_eq!(resolve_clamp("dbrx", Some(8.0)), Ok(Some(8.0)));
        assert_eq!(resolve_clamp("dbrx", Some(0.0)), Ok(None));
        assert_eq!(resolve_clamp("olmo", None), Ok(None));
        assert!(
            ClampError::MissingRequired
                .message("dbrx")
                .contains("dbrx.attention.clamp_kqv"),
            "the message must name the key"
        );
    }

    /// An architecture whose graph does not clamp is unaffected, even
    /// when the file declares the key -- llama.cpp's `load_arch_hparams`
    /// for it never reads the key, so `f_clamp_kqv` stays 0 there.
    #[test]
    fn an_architecture_whose_graph_does_not_clamp_ignores_the_key() {
        // `mpt` was here while it was refused; it clamps (`mpt.cpp:5`).
        assert_eq!(resolve_clamp("mpt", Some(8.0)), Ok(Some(8.0)));
        for arch in ["llama", "qwen3", "olmo2", "grok"] {
            assert_eq!(
                resolve_clamp(arch, Some(8.0)),
                Ok(None),
                "{arch} must not be clamped by this key"
            );
        }
    }

    /// The clamp is symmetric and leaves in-range values bit-identical.
    #[test]
    fn the_clamp_is_symmetric_and_inert_inside_the_range() {
        let mut x = vec![-9.5f32, -8.0, -0.25, 0.0, 3.0, 8.0, 12.0];
        clamp_in_place(&mut x, 8.0);
        assert_eq!(x, vec![-8.0, -8.0, -0.25, 0.0, 3.0, 8.0, 8.0]);
    }
}
