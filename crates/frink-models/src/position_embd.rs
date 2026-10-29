//! **LEARNED ABSOLUTE POSITION EMBEDDINGS** -- `position_embd.weight`,
//! one row per trained position, ADDED to the token embedding before
//! layer 0 in place of any rotation.
//!
//! # What it is
//!
//! `gpt2.cpp:19` creates `pos_embd` as `{n_embd, n_ctx_train}` REQUIRED
//! and `:74-77` do `inpL = inpL + get_rows(pos_embd, inp_pos)`; the
//! graph calls no `ggml_rope`, and `llama_model_rope_type` answers
//! `LLAMA_ROPE_TYPE_NONE` for it. `starcoder.cpp:19,75-78` is the same
//! graph line for line (multi-query, `head_count_kv 1`; the rope-type
//! switch happens to list it under NORM, which nothing in its graph
//! reads). So the seam is two facts: a table the loader ADDS at the one
//! embedding site, and `crate::rope_layers::RopeLayers::Never`, so no
//! rotation site is reached.
//!
//! # Reach -- MEASURED
//!
//! `grep -l POS_EMBD src/models/*.cpp` over all 155 graphs
//! (2026-09-14): `gpt2`, `starcoder` (REQUIRED, decoders on the generic
//! path), `mpt` (`mpt.cpp:19,80-84`, OPTIONAL beside its ALiBi bias),
//! and `bert` / `nomic-bert` /
//! `nomic-bert-moe` / `jina-bert-v3` on the encoder engine. `mpt`'s
//! optional table is served too (its ALiBi row, `crate::alibi`).
//!
//! # Where the arithmetic is
//!
//! `Decoder::embed_token(token, pos)` -- the ONE embedding site, which
//! already applied `embedding_scale` and the skip-stream norm -- adds
//! row `pos` after the scale and before the norm, the order `gpt2.cpp:
//! 70-77` has (`build_inp_embd` scales, then the add; no graph here has
//! both a scale and a table). Every fused Metal launch is fenced off a
//! model with a table (`Decoder::metal_can_serve_model` on
//! `ModelConfig::learned_positions`): the GPU embedding gather has no
//! add, and the stacks never see `pos` for it.
//!
//! A position at or past the table is refused rather than clamped:
//! `ggml_get_rows` on `inp_pos` past `n_ctx_train` reads out of the
//! tensor upstream, and a context longer than the trained one is
//! already capped by `ModelConfig::context_length`.

use frink_core::WeightMatrix;
use frink_gguf::TensorSource;

use crate::loader::{load_weight_matrix, LoadError};
use crate::proj_bias::Presence;

/// Whether `arch`'s graph adds a learned position table to its token
/// embeddings (and, for the two REQUIRED rows, rotates nothing).
///
/// A row of the architecture's `crate::models` definition (`gpt2`,
/// `starcoder` REQUIRED; `mpt` optional).
pub fn learned_positions(arch: &str) -> bool {
    crate::models::ArchFacts::of(arch)
        .tensors
        .position_embd
        .is_some()
}

/// `position_embd.weight`: `Some` when the architecture's graph creates
/// it and the file has it, an error when the graph requires it and the
/// file lacks it or it is the wrong shape, `None` otherwise -- including
/// for an architecture whose graph never creates it, where a present
/// tensor stays unread and is refused as such.
pub fn load_position_embd(
    file: &impl TensorSource,
    arch: &str,
    row: Option<(Presence, &'static str)>,
    hidden_dim: usize,
    n_ctx_train: Option<usize>,
) -> Result<Option<WeightMatrix>, LoadError> {
    let Some((presence, lines)) = row else {
        return Ok(None);
    };
    const NAME: &str = "position_embd.weight";
    if file.find_tensor(NAME).is_none() {
        return match presence {
            Presence::Required => Err(LoadError::Gguf(frink_gguf::GgufError::TensorNotFound(
                format!("{NAME} (REQUIRED by `{arch}`'s graph, {lines})"),
            ))),
            Presence::Optional => Ok(None),
        };
    }
    // `{arch}.context_length` is `n_ctx_train`, the table's row count
    // upstream (`llama-model.cpp` reads it REQUIRED); both converters
    // write it.
    let Some(n_ctx_train) = n_ctx_train else {
        return Err(LoadError::MissingHparam(format!(
            "{arch}.context_length (sizes {NAME}, {lines})"
        )));
    };
    let table = load_weight_matrix(file, NAME)?;
    if table.rows() != n_ctx_train || table.cols() != hidden_dim {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!(
                "{NAME} is [{}, {}], expected [{n_ctx_train} (context_length), {hidden_dim}] \
                 ({lines} creates it {{n_embd, n_ctx_train}})",
                table.rows(),
                table.cols()
            ),
        ));
    }
    Ok(Some(table))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is a registered architecture; the REQUIRED rows are the
    /// audited generic ones and rotate nothing; the optional row is off
    /// the generic path for its own reason.
    #[test]
    fn every_row_is_registered_and_the_required_rows_rotate_nothing() {
        // Every row, over the catalog: they live in `crate::models`.
        let rows: Vec<(&str, Presence, &str)> = crate::capability::architecture_catalog()
            .iter()
            .filter_map(|p| {
                crate::models::ArchFacts::of(p.gguf_name)
                    .tensors
                    .position_embd
                    .map(|(presence, lines)| (p.gguf_name, presence, lines))
            })
            .collect();
        assert_eq!(rows.len(), 3, "gpt2, starcoder, mpt: {rows:?}");
        for (arch, presence, lines) in rows {
            assert!(
                crate::capability::resolve_profile(arch).is_some(),
                "`{arch}` ({lines}) is not a registered architecture"
            );
            let generic = matches!(
                crate::capability::resolve_architecture(arch),
                Some(crate::capability::ArchPath::GenericGqa { .. })
            );
            match presence {
                Presence::Required => {
                    assert!(
                        generic,
                        "`{arch}` has a golden (tests/position_embd_graphs.rs)"
                    );
                    assert_eq!(
                        crate::rope_layers::rope_layers(arch, 12, false, 0),
                        crate::rope_layers::RopeLayers::Never,
                        "`{arch}` adds positions and must rotate nothing"
                    );
                }
                // `mpt`: served since `crate::alibi`, its optional table
                // added when present (tests/alibi_graphs.rs, the
                // `mpt_posembd` fixture) and, like every ALiBi row,
                // rotating nothing.
                Presence::Optional => {
                    assert!(generic, "`{arch}` is audited on its ALiBi");
                    assert_eq!(
                        crate::rope_layers::rope_layers(arch, 32, false, 0),
                        crate::rope_layers::RopeLayers::Never
                    );
                }
            }
        }
        assert!(!learned_positions("llama"));
    }
}
