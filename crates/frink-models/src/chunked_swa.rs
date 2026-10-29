//! `LLAMA_SWA_TYPE_CHUNKED`: a "window" that is the query's own chunk.
//!
//! llama.cpp's third window kind (`llama-hparams.h`, `is_masked_swa`):
//! under `STANDARD` a query at `p` sees keys `p - n_swa < k <= p`;
//! under `CHUNKED` it sees `k / n_swa == p / n_swa`, the keys in its
//! own chunk of `n_swa` positions and nothing before. So a chunked
//! layer's visible span RESETS at every chunk boundary, where a sliding
//! one slides.
//!
//! One graph of 155 sets it (`grep -l SWA_TYPE_CHUNKED src/models/
//! *.cpp` is `llama4.cpp:13`), from LITERALS: `n_swa = 8192`, a
//! 3-chunked-1-full period of 4, and the attention temperature's three
//! constants, on the branch a file takes unless it declares
//! `attention.sliding_window` PRESENT AND ZERO (`:5-11`). The file's
//! nonzero value is never read. The zero branch is what the converter
//! writes for a model whose every layer is full attention
//! (`conversion/llama.py:391-394`), and libllama ABORTS on it
//! (`llama-graph.cpp:159`, `GGML_ASSERT(f_attn_temp_scale != 0.0f)`:
//! the branch leaves the temperature scale at zero and the graph still
//! builds the temperature input; measured on
//! `scripts/make_llama4_fixture.py --noswa`), so frink refuses it by
//! name rather than answer where its reference cannot.
//!
//! # How it is served
//!
//! Chunked attention for a query at `p` is a sliding window of
//! `p % n_swa + 1` positions -- per query. The single-query kernels
//! take exactly that ([`ModelConfig::layer_window_for_query`]), and the
//! batched prefill body takes its per-query arm for a chunked layer
//! (the blocked kernel takes one window for the batch). `KvCache`
//! eviction keeps the last `n_swa` rows, a superset of any chunk, so a
//! chunked layer evicts like a sliding one and never under-retains.
//! The fused Metal launches take one sliding window per layer and
//! refuse the model (`Decoder::metal_can_serve_model`).
//!
//! [`ModelConfig::layer_window_for_query`]: crate::config::ModelConfig::layer_window_for_query

use crate::loader::LoadError;

/// The chunk `arch` attends within, given what the file declares for
/// `attention.sliding_window`, or `None` for every sliding-window
/// architecture. A declared zero is refused: llama.cpp's own graph
/// aborts on that branch (module doc).
///
/// The chunk is a row of the architecture's `crate::models` definition
/// (`llama4`, `llama4.cpp:5-21`, is the one graph with one).
pub fn chunked_window(arch: &str, declared: Option<u64>) -> Result<Option<usize>, LoadError> {
    chunked_window_for(
        arch,
        crate::models::ArchFacts::of(arch).swa.chunked,
        declared,
    )
}

/// [`chunked_window`] given the row (`ArchFacts`): the chunk and the
/// llama.cpp lines.
pub fn chunked_window_for(
    arch: &str,
    row: Option<(usize, &'static str)>,
    declared: Option<u64>,
) -> Result<Option<usize>, LoadError> {
    let Some((chunk, lines)) = row else {
        return Ok(None);
    };
    if declared == Some(0) {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!(
                "`{arch}.attention.sliding_window` is 0: {lines} take the no-window branch, \
                 which leaves `f_attn_temp_scale` at 0 while the graph still builds the \
                 attention-temperature input, and libllama aborts on it \
                 (llama-graph.cpp:159, `GGML_ASSERT(f_attn_temp_scale != 0.0f)`; measured on \
                 scripts/make_llama4_fixture.py --noswa). frink refuses the file rather than \
                 answer where its reference cannot"
            ),
        ));
    }
    Ok(Some(chunk))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_literal_wins_over_the_file_and_zero_is_refused() {
        assert_eq!(chunked_window("llama4", None).unwrap(), Some(8192));
        assert_eq!(chunked_window("llama4", Some(4096)).unwrap(), Some(8192));
        assert!(chunked_window("llama4", Some(0)).is_err());
        assert_eq!(chunked_window("llama", Some(4096)).unwrap(), None);
    }
}
