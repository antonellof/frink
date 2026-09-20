//! The one place a host body norms the residual stream for a sublayer.
//!
//! Two facts have to be decided together at that point and were
//! decided in six places before this existed: what the sublayer reads
//! (the norm's output) and what its branch will be ADDED to. For every
//! architecture but one the second is "the stream, unchanged", so the
//! sites spelled only the first. `minimax-01` makes the second the
//! first, scaled (`crate::normed_residual`), and a site that kept
//! computing the norm by hand would silently keep the ordinary
//! topology while reading the right vector -- a difference no shape
//! check can see.
//!
//! So [`Decoder::pre_norm_residual`] returns the normed rows AND
//! updates the stream, and `no_host_body_norms_the_stream_by_hand`
//! pins that nothing else calls a pre-norm on `hidden` directly.

use rayon::prelude::*;

use super::Decoder;
use crate::norm::NormOp;

impl Decoder {
    /// Norms `hidden` (`[rows][hidden_dim]`) with `norm` for a
    /// sublayer, and replaces the stream with the scaled result where
    /// the architecture's residual IS that result.
    ///
    /// Returns the normed rows: the sublayer's input, which is the
    /// same vector whichever topology the model has.
    pub(crate) fn pre_norm_residual(
        &self,
        norm: &NormOp,
        hidden: &mut [f32],
        rows: usize,
    ) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        debug_assert_eq!(hidden.len(), rows * hidden_dim);
        let eps = self.config.rms_norm_eps;
        // `par_chunks` above one row, as the two batched bodies did
        // before this seam existed; a decode step is one row and pays
        // no scheduling for it.
        let normed: Vec<f32> = if rows > 1 {
            hidden
                .par_chunks(hidden_dim)
                .map(|row| norm.apply(row, eps))
                .flatten()
                .collect()
        } else {
            hidden
                .chunks(hidden_dim)
                .flat_map(|row| norm.apply(row, eps))
                .collect()
        };
        crate::normed_residual::adopt(hidden, &normed, self.config.normed_residual_scale);
        normed
    }
}

#[cfg(test)]
mod tests {
    /// The four host bodies must reach their pre-norms through
    /// [`Decoder::pre_norm_residual`] and not through `NormOp::apply`
    /// on the stream, because only the former answers the second
    /// question (`crate::normed_residual`). A site that norms by hand
    /// reads the right vector and keeps the wrong topology, which no
    /// shape check and no `Option` can see.
    ///
    /// Whitespace is stripped before the search: `cargo fmt` wraps a
    /// long call across three lines, and a grep test that a formatter
    /// can silence is a test that cannot fail -- this repo has shipped
    /// exactly that.
    #[test]
    fn no_host_body_norms_the_stream_by_hand() {
        // Not this file: the needle below is in its own source.
        const BODIES: [(&str, &str); 4] = [
            ("decoder.rs", include_str!("../decoder.rs")),
            ("decoder/ffn_block.rs", include_str!("ffn_block.rs")),
            ("decoder/attn_block.rs", include_str!("attn_block.rs")),
            (
                "decoder/recurrent_block.rs",
                include_str!("recurrent_block.rs"),
            ),
        ];
        for (name, src) in BODIES {
            let flat: String = src.chars().filter(|c| !c.is_whitespace()).collect();
            assert!(
                !flat.contains(".norm_weight.apply("),
                "{name} norms the residual stream by hand; call `Decoder::pre_norm_residual` \
                 so the architectures whose residual IS that norm's output \
                 (`crate::normed_residual`) are served there too"
            );
        }
    }
}
