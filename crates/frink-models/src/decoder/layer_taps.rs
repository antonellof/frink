//! The residual stream entering chosen layers, captured during a batch
//! forward: what a drafter conditioned on the target's intermediate
//! features reads (`dflash` drafts, `crate::dflash`).
//!
//! llama.cpp calls these the layer *inputs*: `res->t_layer_inp[il] =
//! inpL` at the top of each layer of the graphs that support it
//! (`qwen3.cpp:86`, `llama.cpp`, `qwen35moe.cpp:179`, ...), read back
//! with `llama_get_embeddings_layer_inp`. Index `n_layers` is the
//! stream after the last layer and before the final norm, which only a
//! few graphs expose (`nemotron-h.cpp:252`); it is accepted here
//! because the stream there is the same value every graph computes.
//!
//! The DFlash converter writes `target_layers` as the HF
//! `target_layer_ids` plus one (`conversion/qwen.py:880`), so the HF
//! "output of layer i" and llama.cpp's "input of layer i + 1" name the
//! same vector, and this module only ever deals in the second.

use super::Decoder;

/// Why a target cannot hand a drafter its layer inputs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LayerTapError {
    #[error(
        "the draft reads the input of target layer {layer}, and the target has only {n_layers} \
         layers (index {n_layers} is the stream after the last one)"
    )]
    OutOfRange { layer: usize, n_layers: usize },
    #[error(
        "the draft lists target layer {0} twice or out of order; `target_layers` is ascending"
    )]
    NotAscending(usize),
    #[error(
        "this target runs its layers on two residual streams (HRM), so a single layer input does \
         not exist to hand a drafter"
    )]
    TwoStreams,
}

/// One batch forward's captures: for each requested layer, the
/// `[batch, hidden]` residual entering it, row-major.
#[derive(Debug)]
pub(super) struct LayerTaps {
    layers: Vec<usize>,
    rows: Vec<Option<Vec<f32>>>,
}

impl LayerTaps {
    pub(super) fn new(layers: &[usize]) -> Self {
        Self {
            layers: layers.to_vec(),
            rows: vec![None; layers.len()],
        }
    }

    /// Records `hidden` if layer `l` is one of the taps. Called at the
    /// top of every layer, before the layer reads it.
    pub(super) fn capture(&mut self, l: usize, hidden: &[f32]) {
        if let Some(i) = self.layers.iter().position(|&t| t == l) {
            self.rows[i] = Some(hidden.to_vec());
        }
    }

    /// How many layers from `l` a fused multi-layer run may cover
    /// without stepping over a tap it would have to stop at. A tap AT
    /// `l` is already captured when this is asked, so it does not cut
    /// the run; the next one does.
    // Only the device bodies fuse runs of layers; a CPU-only build
    // walks one layer at a time and has nothing to cut.
    #[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
    pub(super) fn run_end(&self, l: usize, n_layers: usize) -> usize {
        self.layers
            .iter()
            .copied()
            .find(|&t| t > l)
            .unwrap_or(n_layers)
            .min(n_layers)
    }

    /// The captures interleaved per position: row `b` is the requested
    /// layers' inputs for position `b`, concatenated in the order they
    /// were asked for (`[layer0 | layer1 | ...]`), which is the order
    /// `common_speculative_impl_draft_dflash::process` packs them in.
    pub(super) fn into_rows(self, batch_size: usize, hidden_dim: usize) -> Vec<Vec<f32>> {
        let captured: Vec<Vec<f32>> = self
            .rows
            .into_iter()
            .zip(&self.layers)
            .map(|(r, l)| r.unwrap_or_else(|| panic!("layer {l} was not reached by the forward")))
            .collect();
        (0..batch_size)
            .map(|b| {
                captured
                    .iter()
                    .flat_map(|layer| &layer[b * hidden_dim..(b + 1) * hidden_dim])
                    .copied()
                    .collect()
            })
            .collect()
    }
}

/// [`Decoder::forward_batch_with_taps`]'s answer, every vector indexed
/// by position in the batch.
#[derive(Debug)]
pub struct TappedBatch {
    /// The target's logits after each position.
    pub logits: Vec<Vec<f32>>,
    /// The final-normed hidden state at each position.
    pub hidden: Vec<Vec<f32>>,
    /// The requested layers' inputs at each position, concatenated in
    /// the order requested: `layers.len() * hidden_dim` values a row.
    pub features: Vec<Vec<f32>>,
}

impl Decoder {
    /// Checks that `layers` can be tapped on this target, before any
    /// forward is run.
    pub fn check_layer_taps(&self, layers: &[usize]) -> Result<(), LayerTapError> {
        if self.hrm_z_l_init.is_some() || self.config.layer_loops.is_some() {
            return Err(LayerTapError::TwoStreams);
        }
        let n_layers = self.config.n_layers;
        for (i, &l) in layers.iter().enumerate() {
            if l > n_layers {
                return Err(LayerTapError::OutOfRange { layer: l, n_layers });
            }
            if i > 0 && layers[i - 1] >= l {
                return Err(LayerTapError::NotAscending(l));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_interleave_the_layers_in_the_order_requested() {
        let mut taps = LayerTaps::new(&[1, 3]);
        // Two positions of width 2.
        taps.capture(0, &[9.0, 9.0, 9.0, 9.0]);
        taps.capture(1, &[1.0, 2.0, 3.0, 4.0]);
        taps.capture(3, &[5.0, 6.0, 7.0, 8.0]);
        let rows = taps.into_rows(2, 2);
        assert_eq!(
            rows,
            vec![vec![1.0, 2.0, 5.0, 6.0], vec![3.0, 4.0, 7.0, 8.0]]
        );
    }

    #[test]
    fn a_fused_run_stops_before_the_next_tap_and_not_at_the_current_one() {
        let taps = LayerTaps::new(&[2, 5]);
        assert_eq!(taps.run_end(0, 8), 2);
        assert_eq!(taps.run_end(2, 8), 5);
        assert_eq!(taps.run_end(5, 8), 8);
        // A tap past the last layer (the post-stack stream) never cuts
        // a run short of the stack's end.
        assert_eq!(LayerTaps::new(&[8]).run_end(0, 8), 8);
    }
}
