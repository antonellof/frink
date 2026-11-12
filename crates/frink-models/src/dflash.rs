//! DFlash / DFlash2 block-diffusion drafts: llama.cpp's `dflash`
//! architecture (`src/models/dflash.cpp`) and the drafting loop around
//! it (`common/speculative.cpp`, `common_speculative_impl_draft_dflash`).
//!
//! A DFlash draft is not a language model. It is a small stack that
//! reads the TARGET's residual stream at a few layers
//! ([`crate::decoder::TappedBatch::features`]), projects every
//! committed position of it into its own KV cache ("injection"), and
//! then predicts a whole block in one non-causal pass over
//! `[anchor, mask, mask, ...]`. The target verifies the block like any
//! other draft, so the tokens the user sees are the target's.
//!
//! DFlash2 adds two things (llama.cpp#27342): a two-tap grouped dynamic
//! convolution around each sublayer ([`forward::conv`]) and a candidate
//! selector that walks one coherent path through the top-k candidates
//! of each block position ([`selector`]).
//!
//! What refuses by name rather than runs: the DSpark variants (a Markov
//! head, or a DeepSeek-V4 backbone), LiLiCorr heads, reduced draft
//! vocabularies (`d2t`), M-RoPE targets, attention sinks, per-layer
//! output scales, rope frequency factors and non-SiLU activations.
//! Each is a different graph in `dflash.cpp`, and none has a fixture.

mod config;
mod drafter;
mod forward;
mod selector;

pub use config::{DFlashConfig, DFlashError};
pub use drafter::DFlashDrafter;
pub use forward::DFlashDraft;

/// `general.architecture` of a DFlash draft GGUF.
pub const DFLASH_ARCH: &str = "dflash";

/// Whether the GGUF at `file` is a DFlash draft, which is how llama.cpp
/// picks `--spec-type draft-dflash` when none is given
/// (`common_speculative_types_from_model`).
pub fn is_dflash_draft(file: &impl frink_gguf::TensorSource) -> bool {
    file.metadata_str("general.architecture") == Some(DFLASH_ARCH)
}
