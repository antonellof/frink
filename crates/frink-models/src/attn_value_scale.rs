//! **THE ATTENTION OUTPUT SCALE** -- `{arch}.attention.value_scale`, a
//! scalar one graph multiplies the attention branch by AFTER `wo`.
//!
//! # What it is
//!
//! `src/models/mimo2.cpp:14-17` reads `attention.value_scale` as
//! optional and keeps it only when it is not 1.0; `:104` binds it and
//! `:180-183` do `cur = ggml_scale(cur, v_scale)` on the output of
//! `build_attn` -- that is, after `wo` -- when it is nonzero. Then the
//! residual add. `conversion/mimo.py:163-165` writes it from
//! `attention_value_scale`; MiMo-V2-Flash sets `0.707`, so every real
//! export carries it, and a run without it attends at the wrong
//! magnitude on every layer.
//!
//! # Reach -- MEASURED
//!
//! `grep -rn ATTENTION_VALUE_SCALE src/` over all of llama.cpp
//! (2026-09-12): the key enum, the model-saver, one `print_info` line,
//! and `mimo2.cpp`. No other graph reads it, so on every other
//! architecture it is dead metadata upstream and stays dead here, as
//! `yarn_log_multiplier` does (`crate::yarn_magnitude`); a table of one
//! reader rather than a gate that would refuse a key llama.cpp ignores.
//!
//! # Where it is applied
//!
//! `Decoder::attn_out_to_residual_rows`, the ONE tail every host
//! attention body ends in, right after `o_proj` (and gpt-oss's
//! `o_bias`, which no graph has alongside) and before the Gemma
//! post-norm (which `mimo2` has not; the order is `build_attn`, scale,
//! residual, and a post-norm would sit between the scale and the
//! residual as it sits between `wo` and the residual upstream). The
//! fused Metal launches fold `wo` into their kernels with no scale
//! after it, and `metal_can_serve_model` refuses a model that has one.

/// The scale this model multiplies its attention output by after
/// `wo`, or `None` for no scale: the key absent, the key 1.0
/// (`mimo2.cpp:15`), the key 0 (`:180` skips), or an architecture
/// whose graph does not read it (`mimo2` states it in
/// `crate::models::mimo2`).
pub fn resolve_attn_value_scale(arch: &str, key: Option<f32>) -> Option<f32> {
    resolve_attn_value_scale_for(
        crate::models::ArchFacts::of(arch)
            .scaling
            .reads_attn_value_scale,
        key,
    )
}

/// [`resolve_attn_value_scale`] given `ScalingFacts::reads_attn_value_scale`.
pub(crate) fn resolve_attn_value_scale_for(reads: bool, key: Option<f32>) -> Option<f32> {
    if !reads {
        return None;
    }
    key.filter(|&v| v != 1.0 && v != 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mimo2_keeps_a_scale_that_is_neither_one_nor_zero() {
        assert_eq!(resolve_attn_value_scale("mimo2", Some(0.707)), Some(0.707));
        assert_eq!(resolve_attn_value_scale("mimo2", Some(1.0)), None);
        assert_eq!(resolve_attn_value_scale("mimo2", Some(0.0)), None);
        assert_eq!(resolve_attn_value_scale("mimo2", None), None);
    }

    /// Dead metadata everywhere else, as upstream: a `llama` file
    /// carrying the key is neither scaled nor refused.
    #[test]
    fn other_architectures_ignore_the_key_as_llama_cpp_does() {
        for arch in ["llama", "qwen3moe", "step35", "gpt-oss"] {
            assert_eq!(resolve_attn_value_scale(arch, Some(0.707)), None, "{arch}");
        }
    }

    #[test]
    fn every_reader_is_an_audited_generic_row() {
        let mut count = 0;
        for def in crate::models::REGISTRY
            .iter()
            .filter(|d| d.facts.scaling.reads_attn_value_scale)
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
        assert!(count > 0, "at least mimo2 reads attention.value_scale");
    }
}
