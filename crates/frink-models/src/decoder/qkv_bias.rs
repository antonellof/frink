//! The QKV bias and the QKV clamp, applied to a projection ONCE, for
//! every host body.
//!
//! llama.cpp's `build_qkv` (`llama-graph.cpp:1591-1664`) does exactly
//! two things to each projection between the matmul and the reshape:
//! adds its bias if the layer has one, then clamps it if the model
//! declares `{arch}.attention.clamp_kqv`. Both happen before the
//! QK-norm and before RoPE.
//!
//! frink computes those projections in three host bodies -- the
//! single-row [`super::attn_block`], the prefill batch and the
//! continuous-batching prefill -- and each of them used to spell the
//! three bias loops out by hand. Three copies of one rule is how this
//! repo lost `attention_scale` from two of four sites, and it is why
//! `crate::clamp_kqv` was a REFUSAL rather than a feature: a clamp
//! added to some copies and not the others would have been silently
//! wrong on whichever path the token happened to take. So the copies
//! are one function now, and the clamp is a line in it.
//!
//! The Metal side applies the bias inside its kernels
//! (`frink_metal::attn::AttnExtras`) and has no clamp, which is why
//! `Decoder::metal_can_serve_model` refuses every fused launch for a
//! clamped model.

use crate::clamp_kqv::clamp_in_place;

use super::{Decoder, LayerWeights};

impl Decoder {
    /// Bias, then clamp, on Q, K and V -- for one row or for a batch of
    /// rows laid out contiguously (`q_width` / `kv_width` / `v_width`
    /// floats per row; K and V differ only on MiMo-V2,
    /// `crate::kv_head_dims`).
    ///
    /// Elementwise on the clamp and per-row on the bias, so a single
    /// row is the batch of one and both call shapes are the same body.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_qkv_bias_and_clamp(
        &self,
        layer: &LayerWeights,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        q_width: usize,
        kv_width: usize,
        v_width: usize,
    ) {
        let add_bias = |x: &mut [f32], bias: Option<&std::sync::Arc<[f32]>>, width: usize| {
            if let Some(bias) = bias {
                debug_assert_eq!(bias.len(), width);
                for row in x.chunks_mut(width) {
                    for (x, b) in row.iter_mut().zip(bias.iter()) {
                        *x += b;
                    }
                }
            }
        };
        add_bias(q, layer.attn.q_bias.as_ref(), q_width);
        add_bias(k, layer.attn.k_bias.as_ref(), kv_width);
        // V at its own width (`crate::kv_head_dims`).
        add_bias(v, layer.attn.v_bias.as_ref(), v_width);

        // AFTER the bias: llama-graph.cpp:1607-1612 adds `wqkv_b` and
        // only then clamps, and :1626-1652 does the same per projection.
        if let Some(c) = self.config.clamp_kqv {
            clamp_in_place(q, c);
            clamp_in_place(k, c);
            clamp_in_place(v, c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clamp is applied after the bias, not before it.
    ///
    /// A projection at 7.5 with a bias of 1.0 and a clamp of 8.0 lands
    /// on 8.0 in llama.cpp (7.5 + 1.0 = 8.5, clamped) and would land on
    /// 8.5 if the order were reversed (7.5 clamped to 7.5, plus 1.0).
    /// The two orders agree on every value that does not cross the
    /// clamp, which is most of them, so this is the value that can tell
    /// them apart.
    #[test]
    fn the_clamp_follows_the_bias() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.clamp_kqv = Some(8.0);
        let mut d = Decoder::new_random_small(cfg, 1, 32);
        let q_width = d.config.n_heads * d.config.head_dim;
        let kv_width = d.config.n_kv_heads * d.config.head_dim;
        d.layers[0].attn.q_bias = Some(vec![1.0; q_width].into());
        d.layers[0].attn.k_bias = Some(vec![-1.0; kv_width].into());

        let mut q = vec![7.5f32; q_width];
        let mut k = vec![-7.5f32; kv_width];
        let mut v = vec![100.0f32; kv_width];
        let layer = &d.layers[0];
        d.apply_qkv_bias_and_clamp(layer, &mut q, &mut k, &mut v, q_width, kv_width, kv_width);
        assert!(q.iter().all(|&x| x == 8.0), "{q:?}");
        assert!(k.iter().all(|&x| x == -8.0), "{k:?}");
        assert!(
            v.iter().all(|&x| x == 8.0),
            "V has no bias and is still clamped: {v:?}"
        );
    }

    /// With no clamp declared, the helper is exactly the old bias loops.
    #[test]
    fn no_clamp_means_the_bias_alone() {
        let mut d = Decoder::new_random_small(crate::config::test_dense_fixture(), 1, 32);
        assert_eq!(d.config.clamp_kqv, None);
        let q_width = d.config.n_heads * d.config.head_dim;
        let kv_width = d.config.n_kv_heads * d.config.head_dim;
        d.layers[0].attn.v_bias = Some(vec![0.5; kv_width].into());

        // Two rows, to exercise the batch shape.
        let mut q = vec![100.0f32; 2 * q_width];
        let mut k = vec![-100.0f32; 2 * kv_width];
        let mut v = vec![100.0f32; 2 * kv_width];
        let layer = &d.layers[0];
        d.apply_qkv_bias_and_clamp(layer, &mut q, &mut k, &mut v, q_width, kv_width, kv_width);
        assert!(q.iter().all(|&x| x == 100.0));
        assert!(k.iter().all(|&x| x == -100.0));
        assert!(v.iter().all(|&x| x == 100.5), "{v:?}");
    }
}
