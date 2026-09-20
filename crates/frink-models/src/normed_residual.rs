//! The residual stream a sublayer's branch joins, where it is not the
//! stream that entered the layer.
//!
//! Every decoder in this repo computes `h = x + branch(norm(x))`. ONE
//! graph of the 155 does not: `minimax-01.cpp` keeps the PRE-NORM's
//! output as the residual and throws the layer input away.
//!
//! ```text
//!   cur      = rms_norm(inpL, attn_norm)      // :248
//!   residual = cur                            // :249
//!   cur      = attention(cur)                 // :251-420
//!   residual = scale * residual               // :428
//!   ffn_inp  = cur + residual                 // :431
//!
//!   cur      = rms_norm(ffn_inp, ffn_norm)    // :434-438
//!   residual = cur                            // :440
//!   cur      = moe(cur)                       // :442-452
//!   residual = scale * residual               // :455
//!   out      = cur + residual                 // :458
//! ```
//!
//! `inpSA` -- the layer input -- is bound at `:244` and read at `:424`
//! only to be sliced by `inp_out_ids` alongside the others; nothing
//! ever adds it. That dead binding is the check on the reading: if the
//! stream were the ordinary one, `inpSA` would be the thing added.
//!
//! # Why it is not `residual_scale`
//!
//! Granite's `{arch}.residual_scale` multiplies each BRANCH OUTPUT
//! (`granite.cpp:213,238`), which is the same key name and a different
//! arithmetic. One column of [`crate::scalar_multipliers::
//! MultiplierSupport`] answers both, so an architecture cannot be given
//! the key twice with two meanings, and
//! [`crate::scalar_multipliers::ResidualScaleUse`] is that column.
//!
//! A scale of exactly `1.0` is still this topology: the layer input is
//! discarded whatever the multiplier is. So the resolved value is NOT
//! passed through `scale_or_none`, unlike every other multiplier here,
//! and [`crate::ModelConfig::normed_residual_scale`] is `Some(1.0)` for
//! a file that declares the identity.
//!
//! # Where it is applied
//!
//! At the PRE-NORM, not at the add: [`Decoder::pre_norm_residual`] is
//! the one function, called by every host body at the point it norms
//! the stream for a sublayer, and it returns the normed vector while
//! replacing the stream in place. A site that computed the norm itself
//! would silently keep the ordinary topology, so
//! `no_body_norms_the_residual_stream_by_hand` greps for that.
//!
//! Every fused Metal launch is refused for such a model
//! (`Decoder::metal_can_serve_model`): each bakes `x + branch` into its
//! kernel, with the pre-norm's output never leaving the device.

/// The architectures whose sublayers make their PRE-NORM OUTPUT the
/// residual stream.
///
/// Measured over the pin rather than assumed: `grep -ln f_residual_scale
/// src/models/*.cpp` is five graphs (`granite`, `granite-hybrid`,
/// `granite-swa`, `minicpm` through Granite's graph, `minimax-01`), and
/// reading each one's `ggml_scale` argument is what separates them --
/// the four Granite rows scale `cur` after a branch, `minimax-01`
/// scales a `build_norm` result.
pub const NORMED_RESIDUAL_ARCHITECTURES: &[&str] = &["minimax-01"];

/// Replaces `hidden` with `scale * normed` on an architecture whose
/// residual is the pre-norm output, and leaves it untouched otherwise.
///
/// `hidden` and `normed` are `[rows][hidden_dim]` and the same length;
/// the scale is per element, so rows need no separate treatment.
pub fn adopt(hidden: &mut [f32], normed: &[f32], scale: Option<f32>) {
    let Some(scale) = scale else { return };
    debug_assert_eq!(hidden.len(), normed.len());
    for (h, n) in hidden.iter_mut().zip(normed) {
        *h = scale * *n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_the_topology_the_stream_is_untouched() {
        let mut hidden = vec![1.0f32, 2.0, 3.0];
        adopt(&mut hidden, &[9.0, 9.0, 9.0], None);
        assert_eq!(hidden, vec![1.0, 2.0, 3.0]);
    }

    /// The identity scale is still the topology: the layer input is
    /// gone. A `scale_or_none` on this value would have turned the one
    /// architecture that has it back into every other one.
    #[test]
    fn an_identity_scale_still_discards_the_layer_input() {
        let mut hidden = vec![1.0f32, 2.0, 3.0];
        adopt(&mut hidden, &[9.0, 8.0, 7.0], Some(1.0));
        assert_eq!(hidden, vec![9.0, 8.0, 7.0]);
    }

    #[test]
    fn the_scale_multiplies_the_normed_value() {
        let mut hidden = vec![1.0f32, 2.0];
        adopt(&mut hidden, &[4.0, 6.0], Some(0.5));
        assert_eq!(hidden, vec![2.0, 3.0]);
    }
}
