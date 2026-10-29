//! **A DENSE FFN SUMMED WITH THE ROUTED EXPERTS** -- the Grok-2 and
//! Arctic layer shape, served through the shared-expert slot, with the
//! two things that differ between the two graphs as one table.
//!
//! # What it is
//!
//! `src/models/grok.cpp:66-68` creates `ffn_gate` / `ffn_up` /
//! `ffn_down` as `TENSOR_NOT_REQUIRED` beside the routed experts, and
//! `:171-184` does this when they are present:
//!
//! ```text
//! if (model.layers[il].ffn_up) {
//!     ffn_out = build_ffn(cur, ffn_up, ffn_gate, ffn_down, LLM_FFN_GELU, LLM_FFN_PAR, il);
//!     cur = ggml_scale(ctx0, ggml_add(ctx0, ffn_out, moe_out), std::sqrt(2) / 2);
//! } else {
//!     cur = moe_out;
//! }
//! ```
//!
//! `src/models/arctic.cpp:38-42` creates the same three REQUIRED, sized
//! `{n_embd, n_embd}`, and `:118-154` runs `ffn_out = build_ffn(
//! ffn_norm(ffn_inp), ...SILU, PAR)`, `moe_out = build_moe_ffn(
//! ffn_norm_exps(inpSA), ...)` and sums them with no scale.
//!
//! # Reach -- MEASURED
//!
//! Over all 155 `src/models/*.cpp` (2026-09-12): every graph that calls
//! `build_moe_ffn` AND reads a dense `layers[il].ffn_up` was listed;
//! all but two use the dense triple on their LEADING dense layers
//! (`if (il < n_layer_dense_lead)` or `if (ffn_gate_inp == nullptr)`)
//! or as `_shexp`. The two that SUM a dense FFN with the routed output
//! on one layer are `grok.cpp:171-184` and `arctic.cpp:118-154`. So
//! [`PARALLEL_DENSE_FFN_ARCHITECTURES`] has two rows, and the two free
//! parameters are the columns: whether the triple is required
//! (`arctic`) or optional (`grok`: Grok-1 has none and takes the
//! `else`), and the scale on the sum (`sqrt(2)/2` for `grok`, none for
//! `arctic`). What does NOT differ and so is not a column: the dense
//! branch reads the normed FFN input `cur` in both, and its activation
//! is the architecture's dense activation (GELU for `grok`, SiLU for
//! `arctic`), which is what `ModelConfig::layer_ffn_acts(il).dense`
//! already answers for the shared-expert slot. Arctic's OTHER
//! difference -- the routed branch reading `ffn_norm_exps(inpSA)` --
//! is `crate::router_input::RouterInput::NormedLayerInput`, one graph
//! of 155, and not this module's business.
//!
//! # Why the shared-expert slot
//!
//! `MoeWeights::shared_experts` is already "a dense FFN that fires on
//! every token and is added to the routed sum", computed with the
//! architecture's dense activation on the normed FFN input, in the
//! row body and the batched body alike; every fused Metal MoE launch
//! refuses a layer that has one. What the slot lacked was the tensor
//! NAMES (it loads `_shexp`) and the scale on the sum, which is why
//! this module used to be a refusal: reading Grok-2's triple into it
//! would have dropped the `sqrt(2)/2`. The loader fills the slot from
//! the dense names for these two architectures and records the row's
//! scale on the layer (`MoeWeights::parallel_sum_scale`), which the
//! FFN bodies apply to the WHOLE branch output -- `ffn_out + moe_out`
//! -- before the post-FFN norm, where `grok.cpp:180` applies it. A
//! Grok-1 layer has no triple, loads no shared expert and carries no
//! scale, which is the `else` branch.

use std::f32::consts::FRAC_1_SQRT_2;

use frink_gguf::TensorSource;

use crate::LoadError;

/// Whether the dense triple must be present on a routed layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensePresence {
    /// `create_tensor(..., 0)`: absent is a load error upstream.
    Required,
    /// `TENSOR_NOT_REQUIRED`: absent means "no dense branch, no scale".
    Optional,
}

/// One architecture whose routed layers sum a dense FFN with the
/// experts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParallelDenseFfn {
    pub arch: &'static str,
    pub presence: DensePresence,
    /// The factor on `ffn_out + moe_out`, or `None` for a plain sum.
    pub sum_scale: Option<f32>,
    pub lines: &'static str,
}

/// Grok's row, stated by `crate::models::grok` (#403 step 6): the
/// dense triple OPTIONAL, the sum scaled by `1/sqrt(2)`.
pub const GROK: ParallelDenseFfn = ParallelDenseFfn {
    arch: "grok",
    presence: DensePresence::Optional,
    sum_scale: Some(FRAC_1_SQRT_2),
    lines: "src/models/grok.cpp:66-68,171-184",
};

/// The graphs not yet in `crate::models`, with the lines.
pub const PARALLEL_DENSE_FFN_ARCHITECTURES: &[ParallelDenseFfn] = &[ParallelDenseFfn {
    arch: "arctic",
    presence: DensePresence::Required,
    sum_scale: None,
    lines: "src/models/arctic.cpp:38-42,118-154",
}];

/// The scale on `moe_out + shexp_out` for `arch`, when its routed
/// layer carries a shared expert; `None` for a plain sum.
pub fn shared_expert_sum_scale(arch: &str, has_shared_expert: bool) -> Option<f32> {
    sum_scale_on_layer(shared_expert_sum_scale_row(arch), has_shared_expert)
}

/// `arch`'s shared-expert sum scale alone.
///
/// A row of the architecture's `crate::models` definition: `cohere2moe`
/// (`cohere2moe.cpp:248-260`, `* 0.5`) is the one graph with one.
pub fn shared_expert_sum_scale_row(arch: &str) -> Option<f32> {
    crate::models::ArchFacts::of(arch)
        .tensors
        .shared_expert_sum_scale
}

/// The row's scale on a layer that loaded a shared expert, else `None`.
pub fn sum_scale_on_layer(row: Option<f32>, has_shared_expert: bool) -> Option<f32> {
    row.filter(|_| has_shared_expert)
}

/// The row for an architecture, or `None` for one whose routed layers
/// have no dense branch.
pub fn parallel_dense_ffn(arch: &str) -> Option<&'static ParallelDenseFfn> {
    crate::models::ArchFacts::of(arch).tensors.parallel_dense
}

/// [`parallel_dense_ffn`] from [`PARALLEL_DENSE_FFN_ARCHITECTURES`]
/// alone, for `ArchFacts::from_tables`.
pub(crate) fn parallel_dense_ffn_from_table(arch: &str) -> Option<&'static ParallelDenseFfn> {
    PARALLEL_DENSE_FFN_ARCHITECTURES
        .iter()
        .find(|row| row.arch == arch)
}

/// Whether routed layer `l` of `arch` carries the dense triple, by the
/// row's presence rule: `Some(row)` when it does and the loader should
/// fill the shared-expert slot from `ffn_{gate,up,down}` and record the
/// scale, `None` when the layer runs the experts alone.
///
/// Refuses a REQUIRED triple that is missing (llama.cpp's loader would
/// fail on the same tensor), and an incomplete triple in either case,
/// because a graph with `ffn_up` and no `ffn_down` exists nowhere.
pub fn parallel_dense_for_layer(
    arch: &str,
    file: &impl TensorSource,
    l: usize,
) -> Result<Option<&'static ParallelDenseFfn>, LoadError> {
    let Some(row) = parallel_dense_ffn(arch) else {
        return Ok(None);
    };
    let names = ["ffn_gate", "ffn_up", "ffn_down"].map(|t| format!("blk.{l}.{t}.weight"));
    let present = names
        .iter()
        .filter(|n| file.find_tensor(n).is_some())
        .count();
    match (present, row.presence) {
        (3, _) => Ok(Some(row)),
        (0, DensePresence::Optional) => Ok(None),
        (0, DensePresence::Required) => Err(LoadError::Gguf(
            frink_gguf::GgufError::TensorNotFound(format!(
                "{} (the dense half of `{arch}`'s parallel dense + MoE layer, REQUIRED by {})",
                names[1], row.lines
            )),
        )),
        _ => Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!(
                "layer {l} carries {present} of the dense `ffn_gate` / `ffn_up` / `ffn_down` \
                 triple; {} reads all three or none",
                row.lines
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_source::StubSource;

    const TRIPLE: [&str; 3] = [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ];

    /// Grok-2's triple is served with the scale; Grok-1's absence is
    /// the `else` branch; Arctic's absence is a missing REQUIRED tensor.
    #[test]
    fn presence_follows_each_rows_rule() {
        let grok2 = parallel_dense_for_layer("grok", &StubSource::with_tensors(&TRIPLE), 0)
            .unwrap()
            .expect("Grok-2 has the branch");
        assert_eq!(grok2.sum_scale, Some(FRAC_1_SQRT_2));
        assert!(
            parallel_dense_for_layer("grok", &StubSource::with_tensors(&[]), 0)
                .unwrap()
                .is_none()
        );

        let arctic = parallel_dense_for_layer("arctic", &StubSource::with_tensors(&TRIPLE), 0)
            .unwrap()
            .expect("Arctic always has the branch");
        assert_eq!(arctic.sum_scale, None);
        let err = parallel_dense_for_layer("arctic", &StubSource::with_tensors(&[]), 0)
            .expect_err("REQUIRED");
        assert!(err.to_string().contains("arctic.cpp:38-42"), "{err}");

        // An incomplete triple is refused on both rows.
        for arch in ["grok", "arctic"] {
            let err = parallel_dense_for_layer(arch, &StubSource::with_tensors(&TRIPLE[..2]), 0)
                .err()
                .unwrap_or_else(|| panic!("{arch}: 2 of 3 refused"));
            assert!(err.to_string().contains("2 of the dense"), "{err}");
        }
    }

    /// Every other architecture with a dense `ffn_up` is a dense model
    /// or a leading-dense MoE, and neither is this table's business.
    #[test]
    fn a_dense_ffn_on_any_other_architecture_is_not_this_tables_business() {
        for arch in ["llama", "deepseek", "dbrx", "qwen3moe", "smallthinker"] {
            assert!(
                parallel_dense_for_layer(arch, &StubSource::with_tensors(&TRIPLE), 0)
                    .unwrap()
                    .is_none(),
                "{arch}"
            );
        }
    }

    /// Every row is an audited generic-path architecture, or the seam
    /// is unevidenced.
    #[test]
    fn every_row_is_an_audited_generic_row() {
        for row in PARALLEL_DENSE_FFN_ARCHITECTURES {
            let profile = crate::capability::resolve_profile(row.arch).unwrap_or_else(|| {
                panic!(
                    "`{}` ({}) is not a registered architecture",
                    row.arch, row.lines
                )
            });
            assert!(matches!(
                profile.path,
                crate::capability::ArchPath::GenericGqa { .. }
            ));
            assert!(
                crate::capability::AUDITED_GENERIC_GQA.contains(&row.arch),
                "{}",
                row.arch
            );
        }
    }
}
