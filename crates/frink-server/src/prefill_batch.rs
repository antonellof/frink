//! How many prompt tokens go through one forward pass: llama.cpp's
//! `-b` / `-ub`, and the two environment variables frink already had
//! for the same number.
//!
//! # Why this module exists rather than two `set_var` calls
//!
//! frink has two prefill paths and each grew its own spelling of the
//! same knob:
//!
//! - the private `generate` loop reads `FRINK_CHUNKED_PREFILL`
//!   ([`generate::forward_prompt_batch`](crate::generate));
//! - the continuous-batching scheduler reads `FRINK_CB_PREFILL_CHUNK`
//!   ([`BatcherConfig::from_env`](crate::serving::batch::BatcherConfig)).
//!
//! That is this repo's dominant bug shape -- two structures that must
//! agree about one thing, with nothing enforcing it. An operator who
//! set one and served on the other path got the default and no warning.
//! So the names live here, in [`PREFILL_CHUNK_ENV_KEYS`], and both
//! readers index that array instead of writing the string themselves:
//! adding a third path means adding an entry, not remembering one.
//!
//! # The `-b` / `-ub` mapping
//!
//! llama.cpp splits the number in two: `-b`/`--batch-size` is the
//! *logical* maximum submitted to `llama_decode`, `-ub`/`--ubatch-size`
//! the *physical* maximum computed at once
//! (`common/arg.cpp:1616-1628`). frink has one stage, so it has one
//! number, and the value it takes is llama.cpp's own resolution of the
//! two -- `cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch
//! == 0 ? params.n_batch : params.n_ubatch)`
//! (`src/llama-context.cpp:265`), which is [`effective_chunk`].
//!
//! Naming neither flag leaves both paths on their own defaults, so a
//! command that does not mention batching is not silently re-tuned.

/// The environment variables that carry the prefill chunk, in the order
/// [`crate::generate`] and [`crate::serving::batch`] read them.
///
/// One array rather than two literals: see the module note. Both
/// readers index it, so a rename is a compile error at both sites
/// instead of a silent default at one of them.
pub(crate) const PREFILL_CHUNK_ENV_KEYS: [&str; 2] =
    ["FRINK_CHUNKED_PREFILL", "FRINK_CB_PREFILL_CHUNK"];

/// Index of the private `generate` path's spelling.
pub(crate) const PRIVATE_PATH_KEY: usize = 0;
/// Index of the continuous-batching scheduler's spelling.
pub(crate) const BATCH_PATH_KEY: usize = 1;

/// llama.cpp's resolution of `-b` and `-ub`, shared with `frink bench`
/// (`frink_core::argv::effective_ubatch`).
pub(crate) use frink_core::argv::effective_ubatch as effective_chunk;

/// What `-b` ALONE lowers to, for the one reader that needs llama.cpp's
/// logical batch rather than the chunk above: `/infill`, whose
/// `format_prompt_infill` takes three quarters of `n_batch` for the
/// prefix and one for the suffix. `-ub` does not move it upstream
/// either.
pub(crate) const LOGICAL_BATCH_ENV: &str = "FRINK_LOGICAL_BATCH";

/// llama.cpp's `n_batch` when no `-b` is given (`common/common.h`).
pub(crate) const LLAMA_DEFAULT_N_BATCH: usize = 2048;

/// `-b`, or llama.cpp's default without it.
pub(crate) fn logical_batch() -> usize {
    std::env::var(LOGICAL_BATCH_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(LLAMA_DEFAULT_N_BATCH)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two spellings are one concept. Before this array existed,
    /// `FRINK_CHUNKED_PREFILL` and `FRINK_CB_PREFILL_CHUNK` were
    /// independent literals at their two readers, so setting the one
    /// that belonged to the other serving path silently did nothing.
    #[test]
    fn the_two_prefill_chunk_spellings_are_distinct_and_indexed_by_name() {
        assert_eq!(
            PREFILL_CHUNK_ENV_KEYS[PRIVATE_PATH_KEY],
            "FRINK_CHUNKED_PREFILL"
        );
        assert_eq!(
            PREFILL_CHUNK_ENV_KEYS[BATCH_PATH_KEY],
            "FRINK_CB_PREFILL_CHUNK"
        );
        assert_ne!(
            PREFILL_CHUNK_ENV_KEYS[PRIVATE_PATH_KEY], PREFILL_CHUNK_ENV_KEYS[BATCH_PATH_KEY],
            "two readers indexing the same entry would leave one path unconfigurable"
        );
    }
}
