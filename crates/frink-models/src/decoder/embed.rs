//! Token embedding (the one site the position table, the embedding norm
//! and the scale are applied) and the hidden-to-logits helpers.

use crate::norm::NormOp;

use super::lm_head::Logits;
use super::Decoder;

impl Decoder {
    /// One token's embedding row, scaled if this checkpoint scales it.
    ///
    /// `embedding_scale` is `sqrt(hidden_dim)` on the Gemma family and
    /// `None` everywhere else, so a path that dequantizes the row and
    /// forgets the multiply is wrong on exactly one family and right on
    /// every other -- which is why it survived as a drift for as long as
    /// it did. The lookup and the scale live in one function so a caller
    /// cannot obtain the row without it.
    /// The embedding row, scaled by `embedding_scale` where the
    /// architecture has one, and -- for a skip-stream model
    /// (`crate::skip_stream`) -- RMS-normed without a weight, as
    /// `talkie.cpp:50` norms `inpL` before layer 0. The ONE embedding
    /// site; what it returns is both layer 0's input and the skip
    /// source.
    ///
    /// `pos` is the token's position, which a learned position table
    /// (`crate::position_embd`) indexes: `gpt2.cpp:74-77` add row `pos`
    /// after `build_inp_embd`'s scale and before anything else.
    pub(super) fn embed_token(&self, token_id: usize, pos: usize) -> Vec<f32> {
        let mut row = self.embedding.dequant_row(token_id);
        if let Some(scale) = self.config.embedding_scale {
            for v in row.iter_mut() {
                *v *= scale;
            }
        }
        if let Some(table) = &self.position_embd {
            assert!(
                pos < table.rows(),
                "position {pos} is past the {}-row learned position table \
                 (`position_embd.weight`); the trained context is the model's limit",
                table.rows()
            );
            for (v, p) in row.iter_mut().zip(table.dequant_row(pos)) {
                *v += p;
            }
        }
        // `bloom.cpp:77-80`: the embedding normed before layer 0. No
        // graph has both a position table and this norm, so their
        // order is not a graph's; it is the order the two arrived in.
        if !matches!(self.embedding_norm, NormOp::None) {
            row = self.embedding_norm.apply(&row, self.config.rms_norm_eps);
        }
        if self.config.skip_stream {
            row = crate::norm::rms_norm_no_params(&row, self.config.rms_norm_eps);
        }
        row
    }

    /// [`Self::embed_token`] for a whole batch: `[batch, hidden]`,
    /// flattened row-major, row `b` at `position(b)`.
    pub(super) fn embed_tokens(
        &self,
        tokens: &[usize],
        position: impl Fn(usize) -> usize,
    ) -> Vec<f32> {
        tokens
            .iter()
            .enumerate()
            .flat_map(|(b, &t)| self.embed_token(t, position(b)))
            .collect()
    }

    /// The `output_head` half of a single-position forward: project the
    /// final-normed hidden state and softcap the result if this
    /// checkpoint softcaps it.
    ///
    /// The counterpart to [`Self::logits_from_flat_hidden`] for the
    /// one-row case, and held here for the same reason: Gemma-2 caps its
    /// final logits at 30.0, so a path that projects and returns without
    /// capping produces a different distribution -- not an error, just a
    /// quietly wrong one.
    pub(super) fn logits_from_normed(&self, final_normed: &[f32]) -> Vec<f32> {
        Logits::project(
            &self.output_head,
            final_normed,
            self.output_bias.as_deref(),
            self.config.final_logit_softcap,
            self.config.logit_multiplier,
        )
        .into_vec()
    }

    /// The `output_head` half of [`Self::forward_batch`], split out so
    /// the hidden-state-returning variant cannot drift from it (a
    /// second copy of the softcap would be a silent quality bug).
    pub(super) fn logits_from_flat_hidden(
        &self,
        flat: Vec<f32>,
        batch_size: usize,
    ) -> Vec<Vec<f32>> {
        let vocab_size = self.output_head.rows();
        let logits_batch = Logits::from_output_head(
            self.output_head.apply_batch(&flat, batch_size),
            self.output_bias.as_deref(),
            self.config.final_logit_softcap,
            self.config.logit_multiplier,
        );
        logits_batch
            .as_slice()
            .chunks(vocab_size)
            .map(|c| c.to_vec())
            .collect()
    }
}
