//! **THE EMBEDDING SKIP STREAM** -- Talkie's second residual: the normed
//! embedding, added back into every layer's output through a learned
//! per-layer scalar.
//!
//! # What it is
//!
//! `src/models/talkie.cpp:50-52` norm the embeddings before layer 0
//! (`build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1)`) and keep the
//! result as `embd_skip`; every layer then computes its attention and
//! FFN on the ordinary residual `inpL` and, AFTER the FFN residual add,
//! adds `embd_skip * out_scale` (`:123-126`), `out_scale` being the
//! layer's `blk.N.layer_output_scale.weight`, a `{1}` tensor (`:32`,
//! REQUIRED). So the embedding reaches every layer twice: once through
//! the residual stream as usual, and once more, scaled, from the side.
//!
//! # Reach -- MEASURED
//!
//! `LLM_TENSOR_LAYER_OUT_SCALE` is created by three graphs (`talkie.cpp`,
//! `gemma4.cpp`, `gemma4-assistant.cpp`), and only `talkie` feeds it
//! from the embedding: Gemma-4 (`gemma4.cpp:365-366`) multiplies its
//! own layer output by it, on its own engine, which has done so since
//! it existed. The embedding norm at `:50` is `talkie` alone. So
//! the row is one `bool` ([`has_skip_stream`], stated in
//! `crate::models::talkie`) on `ModelConfig` covering both halves,
//! because neither exists without the other in any graph.
//!
//! # Where it lives
//!
//! `Decoder::embed_token` is the ONE embedding site (the batch form
//! delegates to it), so the norm sits there and the vector it returns
//! IS the skip source. The three host bodies capture it once, before
//! layer 0, and hand it to the FFN body as a [`SkipStream`]; the two
//! FFN bodies add `skip * out_scale` after the residual add and before
//! the loop norm (`crate::layer_loops`; no graph has both, the order is
//! a convention). `LayerWeights::out_scale` is the per-layer scalar,
//! loaded only when the config says so and REQUIRED then. Every fused
//! Metal launch refuses the model: none norms the embedding or carries
//! a second residual.

use crate::loader::load_f32_vec;
use crate::LoadError;
use frink_gguf::TensorSource;

/// Whether this architecture norms its embeddings and keeps them as a
/// skip stream (`talkie.cpp:50-52,123-126`, stated in
/// `crate::models::talkie`).
pub fn has_skip_stream(arch: &str) -> bool {
    crate::models::ArchFacts::of(arch).topology.skip_stream
}

/// Layer `l`'s `layer_output_scale`, REQUIRED for a skip-stream model
/// (`talkie.cpp:32`) and untouched for every other, so a tensor of that
/// name on an architecture whose graph has no such op stays UNREAD and
/// is refused as such.
pub fn load_out_scale(
    file: &impl TensorSource,
    arch: &str,
    skip_stream: bool,
    l: usize,
) -> Result<Option<f32>, LoadError> {
    if !skip_stream {
        return Ok(None);
    }
    let name = format!("blk.{l}.layer_output_scale.weight");
    let v = load_f32_vec(file, &name)?;
    if v.len() != 1 {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!("{name} has {} entries, expected 1", v.len()),
        ));
    }
    Ok(Some(v[0]))
}

/// The skip source for one forward pass: the normed embedding rows the
/// bodies captured before layer 0, `[rows, hidden_dim]`. `None` for a
/// model without the stream -- an `Option` the FFN bodies take as an
/// argument, so a body cannot be reached with the question unasked.
#[derive(Debug, Clone, Copy)]
pub struct SkipStream<'a> {
    pub rows: &'a [f32],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_talkie_has_the_stream() {
        assert!(has_skip_stream("talkie"));
        for arch in ["llama", "gemma3", "gemma4", "olmo", "bitnet"] {
            assert!(!has_skip_stream(arch), "{arch}");
        }
    }

    #[test]
    fn every_skip_stream_row_is_an_audited_generic_row() {
        let mut count = 0;
        for def in crate::models::REGISTRY
            .iter()
            .filter(|d| d.facts.topology.skip_stream)
        {
            for arch in def.names {
                count += 1;
                let profile = crate::capability::resolve_profile(arch).unwrap_or_else(|| {
                    panic!("`{arch}` ({}) is not a registered architecture", def.source)
                });
                assert!(matches!(
                    profile.path,
                    crate::capability::ArchPath::GenericGqa { .. }
                ));
                assert!(crate::capability::AUDITED_GENERIC_GQA.contains(arch));
            }
        }
        assert!(count > 0, "at least talkie uses the skip stream");
    }
}
