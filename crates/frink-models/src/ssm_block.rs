//! The state-space block a layer carries, in either generation.
//!
//! `AttnWeights::ssm` is one `Option` because the decoder's three host
//! bodies ask one question of it -- "run this layer's block on this
//! state" -- and the answer must not depend on which generation the
//! file is. [`crate::mamba1`] and [`crate::mamba2`] own their tensors
//! and arithmetic; this enum owns nothing but the dispatch, so a third
//! generation is one variant here and one module there.

use frink_core::recurrent_state::RecurrentState;

use crate::gdn::Gdn;
use crate::lightning::Lightning;
use crate::mamba1::Mamba1;
use crate::mamba2::Mamba2;
use crate::plamo2_ssm::Plamo2Ssm;

pub enum SsmBlock {
    /// `build_mamba_layer` (`mamba-base.cpp:4-148`).
    Mamba1(Mamba1),
    /// `build_mamba2_layer` (`mamba-base.cpp:149-288`).
    Mamba2(Mamba2),
    /// `build_layer_attn_linear` (`qwen35.cpp:236-317`), the gated delta
    /// net.
    Gdn(Gdn),
    /// `build_plamo2_mamba_layer` (`plamo2.cpp:218-343`).
    Plamo2(Plamo2Ssm),
    /// MiniMax-01's lightning attention (`minimax-01.cpp:293-420`),
    /// `crate::lightning`. Its state is one `head_dim x head_dim` KV
    /// per head and it has no convolution, which is why
    /// `RecurrentState::conv` is empty for it -- as it is upstream,
    /// where `:296-299` allocate the conv rows only because the
    /// recurrent memory has no way to say a block does not want them.
    Lightning(Lightning),
}

impl SsmBlock {
    /// A fresh sequence's state for this layer, at the block's size.
    pub fn zero_state(&self) -> RecurrentState {
        match self {
            SsmBlock::Mamba1(m) => m.zero_state(),
            SsmBlock::Mamba2(m) => m.zero_state(),
            SsmBlock::Gdn(m) => m.zero_state(),
            SsmBlock::Plamo2(m) => m.zero_state(),
            SsmBlock::Lightning(m) => m.zero_state(),
        }
    }

    /// `rows` consecutive tokens of ONE sequence through the block,
    /// advancing `state` in place.
    pub fn forward_rows(
        &self,
        normed: &[f32],
        rows: usize,
        state: &mut RecurrentState,
        rms_eps: f32,
    ) -> Vec<f32> {
        match self {
            SsmBlock::Mamba1(m) => m.forward_rows(normed, rows, state, rms_eps),
            SsmBlock::Mamba2(m) => m.forward_rows(normed, rows, state, rms_eps),
            SsmBlock::Gdn(m) => m.forward_rows(normed, rows, state, rms_eps),
            SsmBlock::Plamo2(m) => m.forward_rows(normed, rows, state, rms_eps),
            SsmBlock::Lightning(m) => m.forward_rows(normed, rows, state, rms_eps),
        }
    }

    pub fn mamba2(&self) -> Option<&Mamba2> {
        match self {
            SsmBlock::Mamba2(m) => Some(m),
            _ => None,
        }
    }

    pub fn mamba2_mut(&mut self) -> Option<&mut Mamba2> {
        match self {
            SsmBlock::Mamba2(m) => Some(m),
            _ => None,
        }
    }

    pub fn mamba1(&self) -> Option<&Mamba1> {
        match self {
            SsmBlock::Mamba1(m) => Some(m),
            _ => None,
        }
    }

    pub fn gdn(&self) -> Option<&Gdn> {
        match self {
            SsmBlock::Gdn(m) => Some(m),
            _ => None,
        }
    }

    pub fn gdn_mut(&mut self) -> Option<&mut Gdn> {
        match self {
            SsmBlock::Gdn(m) => Some(m),
            _ => None,
        }
    }
}
