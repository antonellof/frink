//! The sampling a GGUF recommends for itself.

use super::*;

/// `general.sampling.*` is the checkpoint's own recommendation, and
/// only the keys the file carries become one: a file naming just
/// `top_k` must leave temperature and top_p to the server's
/// defaults.
#[test]
fn gguf_sampling_metadata_is_read_as_the_checkpoints_recommendation() {
    use crate::sampling::RecommendedSampling;
    let full = RecommendedSampling::from_gguf(&open_metadata_gguf(
        "sampling_full",
        &[
            ("general.architecture", Kv::Str("llama")),
            ("general.sampling.temp", Kv::F32(1.0)),
            ("general.sampling.top_k", Kv::U32(20)),
            ("general.sampling.top_p", Kv::F32(0.95)),
        ],
    ));
    assert_eq!(
        full,
        RecommendedSampling {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(20),
        }
    );

    let partial = RecommendedSampling::from_gguf(&open_metadata_gguf(
        "sampling_partial",
        &[
            ("general.architecture", Kv::Str("llama")),
            ("general.sampling.top_k", Kv::U32(40)),
        ],
    ));
    assert_eq!(partial.top_k, Some(40));
    assert_eq!(partial.temperature, None);
    assert_eq!(partial.top_p, None);
}

/// A converter that wrote `temp = 1` stores a GGUF integer, not a
/// float. Dropping it would serve a checkpoint that asked for
/// temperature 1.0 at the framework's greedy default -- the
/// repetition-loop failure the recommendation exists to prevent.
#[test]
fn an_integer_valued_sampling_temp_is_still_a_recommendation() {
    let recommended = crate::sampling::RecommendedSampling::from_gguf(&open_metadata_gguf(
        "sampling_int_temp",
        &[
            ("general.architecture", Kv::Str("llama")),
            ("general.sampling.temp", Kv::U32(1)),
        ],
    ));
    assert_eq!(recommended.temperature, Some(1.0));
}

/// The overwhelming majority of checkpoints recommend nothing, and
/// those must keep frink's existing defaults exactly.
#[test]
fn a_gguf_without_sampling_metadata_recommends_nothing() {
    let recommended = crate::sampling::RecommendedSampling::from_gguf(&open_metadata_gguf(
        "sampling_absent",
        &[("general.architecture", Kv::Str("llama"))],
    ));
    assert!(recommended.is_empty());
}
