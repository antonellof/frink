//! Q/K RMSNorm: how it is shaped, and which side of RoPE it goes on.
//!
//! Two per-architecture facts live here, and this repo has lost each
//! kind of fact at least once by leaving it inline:
//!
//! * **Shape.** `attn_q_norm` / `attn_k_norm` are either one whole-vector
//!   weight (OLMoE) or one `head_dim`-long weight applied per head
//!   (Qwen3, Gemma-3). `loader.rs` derives which from the weight length
//!   and refuses anything matching neither, and
//!   [`crate::capability::QkNormStyle`] carries the answer.
//! * **Order.** Almost every architecture norms Q and K and then rotates
//!   them (`qwen3moe.cpp:99,108`). `maincoder` and `hunyuan-moe` rotate
//!   and then norm (`maincoder.cpp:78-95`, `hunyuan-moe.cpp:93-118`).
//!   Rotating a normed vector and norming a rotated one give different
//!   attention scores on every layer, and no GGUF key distinguishes
//!   them: llama.cpp writes the order into each hand-written graph.
//!
//! The application itself was written out THREE times before this
//! module existed -- in `attn_block`, in `forward_batch_last`'s prefill
//! and in `forward_multi_seq` -- which is the shape that has already
//! cost this repo eight model features. Making the order configurable
//! would have turned three copies into six places to get an ordering
//! wrong, so the body moved here first.

use frink_core::matmul::{rms_norm, rms_norm_per_head};

use super::{Decoder, LayerWeights};

impl Decoder {
    /// Applies Q/K RMSNorm to one vector according to
    /// [`crate::config::ModelConfig::qk_norm_style`].
    pub(crate) fn apply_qk_norm(&self, x: &[f32], weight: &[f32]) -> Vec<f32> {
        use crate::capability::QkNormStyle;
        match self.config.qk_norm_style {
            QkNormStyle::WholeVector => rms_norm(x, weight, self.config.rms_norm_eps),
            QkNormStyle::PerHead => {
                rms_norm_per_head(x, weight, self.config.head_dim, self.config.rms_norm_eps)
            }
            // `talkie.cpp:82-84`: RMS over each head, then that head's one
            // scalar. `weight` is `[n_heads]`.
            QkNormStyle::PerHeadScalar => Self::rms_norm_per_head_scalar_gain(
                x,
                Some(weight),
                self.config.head_dim,
                self.config.rms_norm_eps,
            ),
            // `plamo2.cpp:163,166`: RMS over each head, times THAT head's
            // row of the weight. `weight` is `[n_heads * head_dim]`.
            QkNormStyle::PerHeadDistinct => {
                let head_dim = self.config.head_dim;
                assert_eq!(weight.len(), x.len(), "one weight row per head");
                let mut out = Vec::with_capacity(x.len());
                for (head, w) in x.chunks_exact(head_dim).zip(weight.chunks_exact(head_dim)) {
                    out.extend(rms_norm(head, w, self.config.rms_norm_eps));
                }
                out
            }
        }
    }

    /// Per-head RMSNorm with one gain per head (`Some`) or none (`None`,
    /// talkie's K at `:90`).
    fn rms_norm_per_head_scalar_gain(
        x: &[f32],
        gains: Option<&[f32]>,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        debug_assert_eq!(x.len() % head_dim, 0);
        let mut out = Vec::with_capacity(x.len());
        for (h, head) in x.chunks_exact(head_dim).enumerate() {
            let normed = crate::norm::rms_norm_no_params(head, eps);
            let gain = gains.map_or(1.0, |g| g[h]);
            out.extend(normed.iter().map(|v| v * gain));
        }
        out
    }

    /// Applies this layer's Q/K RMSNorm in place, row by row, to a whole
    /// batch of projections. `q_width` / `kv_width` are one row's widths,
    /// so a single-token caller passes the slice lengths.
    fn apply_qk_norms_batch(
        &self,
        layer: &LayerWeights,
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        q_width: usize,
        kv_width: usize,
    ) {
        if let Some(q_norm) = &layer.attn.q_norm {
            for row in q_batch.chunks_mut(q_width) {
                let normed = self.apply_qk_norm(row, q_norm);
                row.copy_from_slice(&normed);
            }
        }
        if let Some(k_norm) = &layer.attn.k_norm {
            for row in k_batch.chunks_mut(kv_width) {
                let normed = self.apply_qk_norm(row, k_norm);
                row.copy_from_slice(&normed);
            }
        } else if self.config.qk_norm_style == crate::capability::QkNormStyle::PerHeadScalar {
            // The weightless K norm (`talkie.cpp:90`): no tensor, and the
            // norm still runs. The style decides, not the tensor's
            // presence, because for this architecture there is none.
            for row in k_batch.chunks_mut(kv_width) {
                let normed = Self::rms_norm_per_head_scalar_gain(
                    row,
                    None,
                    self.config.head_dim,
                    self.config.rms_norm_eps,
                );
                row.copy_from_slice(&normed);
            }
        }
    }

    /// The QK norms that run BEFORE RoPE on this architecture -- i.e. all
    /// of them unless [`Decoder::qk_norm_after_rope`] is set.
    pub(crate) fn apply_qk_norms_pre_rope(
        &self,
        layer: &LayerWeights,
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        q_width: usize,
        kv_width: usize,
    ) {
        if !self.qk_norm_after_rope {
            self.apply_qk_norms_batch(layer, q_batch, k_batch, q_width, kv_width);
        }
    }

    /// The QK norms that run AFTER RoPE: the weighted ones for
    /// `maincoder` / `hunyuan-moe`, and Llama 4's weightless per-head
    /// pair on its rotating layers (`crate::weightless_qk_norm`).
    pub(crate) fn apply_qk_norms_post_rope(
        &self,
        layer: &LayerWeights,
        layer_idx: usize,
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        q_width: usize,
        kv_width: usize,
    ) {
        if self.qk_norm_after_rope {
            self.apply_qk_norms_batch(layer, q_batch, k_batch, q_width, kv_width);
        }
        if self.config.weightless_qk_norm && self.config.layer_rotates(layer_idx) {
            // `llama4.cpp:182-185`: no tensor, both projections, each
            // head on its own.
            for row in q_batch
                .chunks_mut(q_width)
                .chain(k_batch.chunks_mut(kv_width))
            {
                let normed = Self::rms_norm_per_head_scalar_gain(
                    row,
                    None,
                    self.config.head_dim,
                    self.config.rms_norm_eps,
                );
                row.copy_from_slice(&normed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::glm_5_2;
    use crate::Decoder;

    /// The weightless pair (`llama4.cpp:182-185`) norms every head of
    /// Q AND K on a rotating layer and touches nothing on an unrotated
    /// one, with no tensor on the layer.
    #[test]
    fn the_weightless_norm_runs_per_head_on_rotating_layers_only() {
        let mut cfg = glm_5_2();
        cfg.hidden_dim = 16;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 4;
        cfg.moe.hidden_dim = 16;
        cfg.moe.expert_ffn_dim = 8;
        cfg.n_layers = 4;
        cfg.weightless_qk_norm = true;
        // Layer 3 does not rotate (`crate::rope_layers`, llama4's rule).
        cfg.sliding_window = Some(8);
        cfg.rope_layers = crate::rope_layers::rope_layers("llama4", 4, true, 0);
        let decoder = Decoder::new_random_small(cfg, 4, 8);
        assert!(decoder.config.layer_rotates(0) && !decoder.config.layer_rotates(3));
        let (q_width, kv_width) = (16, 8);
        let q0: Vec<f32> = (0..q_width).map(|i| 0.5 + i as f32 * 0.25).collect();
        let k0: Vec<f32> = (0..kv_width).map(|i| 1.0 - i as f32 * 0.1).collect();
        let layer = &decoder.layers[0];
        assert!(layer.attn.q_norm.is_none() && layer.attn.k_norm.is_none());

        let (mut q, mut k) = (q0.clone(), k0.clone());
        decoder.apply_qk_norms_post_rope(layer, 0, &mut q, &mut k, q_width, kv_width);
        for (got, head) in q.chunks(4).zip(q0.chunks(4)) {
            let want = crate::norm::rms_norm_no_params(head, decoder.config.rms_norm_eps);
            assert_eq!(got, &want[..]);
        }
        for (got, head) in k.chunks(4).zip(k0.chunks(4)) {
            let want = crate::norm::rms_norm_no_params(head, decoder.config.rms_norm_eps);
            assert_eq!(got, &want[..]);
        }

        let (mut q, mut k) = (q0.clone(), k0.clone());
        decoder.apply_qk_norms_post_rope(layer, 3, &mut q, &mut k, q_width, kv_width);
        assert_eq!(q, q0);
        assert_eq!(k, k0);
    }

    /// Exactly one of the two hooks fires, whichever way the flag is
    /// set.
    ///
    /// The failure this rules out is the one the ordering flag invites:
    /// a site that calls the pre-RoPE hook and forgets the post-RoPE one
    /// leaves an architecture with NO QK norm at all, which is a quiet
    /// wrongness rather than an error. Here both hooks run over the same
    /// buffers, so "normed exactly once" is checkable without knowing
    /// which side did it.
    #[test]
    fn the_two_hooks_are_exclusive_and_together_always_norm_exactly_once() {
        for after in [false, true] {
            let mut cfg = glm_5_2();
            cfg.hidden_dim = 16;
            cfg.n_heads = 4;
            cfg.n_kv_heads = 2;
            cfg.head_dim = 4;
            cfg.moe.hidden_dim = 16;
            cfg.moe.expert_ffn_dim = 8;
            let mut decoder = Decoder::new_random_small(cfg, 1, 8);
            decoder.qk_norm_after_rope = after;
            let q_width = decoder.config.n_heads * decoder.config.head_dim;
            let kv_width = decoder.config.n_kv_heads * decoder.config.head_dim;
            decoder.layers[0].attn.q_norm = Some(vec![2.0; q_width].into());
            decoder.layers[0].attn.k_norm = Some(vec![3.0; kv_width].into());

            let q0: Vec<f32> = (0..q_width).map(|i| 0.5 + i as f32 * 0.25).collect();
            let k0: Vec<f32> = (0..kv_width).map(|i| 1.0 - i as f32 * 0.1).collect();

            // Once, through whichever hook this architecture uses.
            let mut q = q0.clone();
            let mut k = k0.clone();
            let layer = &decoder.layers[0];
            decoder.apply_qk_norms_pre_rope(layer, &mut q, &mut k, q_width, kv_width);
            decoder.apply_qk_norms_post_rope(layer, 0, &mut q, &mut k, q_width, kv_width);

            // The reference: the norm applied exactly once, directly.
            let want_q = decoder.apply_qk_norm(&q0, &vec![2.0; q_width]);
            let want_k = decoder.apply_qk_norm(&k0, &vec![3.0; kv_width]);

            for (got, want) in q.iter().zip(want_q.iter()) {
                assert!(
                    (got - want).abs() < 1e-6,
                    "after_rope={after}: Q normed {got} vs {want}"
                );
            }
            for (got, want) in k.iter().zip(want_k.iter()) {
                assert!(
                    (got - want).abs() < 1e-6,
                    "after_rope={after}: K normed {got} vs {want}"
                );
            }
            // And it really did move: a vacuous comparison would pass
            // even if both hooks were no-ops.
            assert!(
                q.iter().zip(q0.iter()).any(|(a, b)| (a - b).abs() > 1e-3),
                "after_rope={after}: the norm changed nothing"
            );
        }
    }
}

/// The Metal side of the ordering flag: there is no kernel that can
/// express it, so the layer must be refused rather than served wrong.
#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::*;

    /// `AttnExtras` hands the norm weights to kernels that apply them
    /// BEFORE their own RoPE, so every fused launch would compute a
    /// different attention than the host bodies for the same weights --
    /// the same checkpoint answering differently depending on which
    /// backend served the token. That is the failure
    /// `layer_supports_metal_attn` exists to prevent, and the four-way
    /// drift of that check has already produced it once.
    ///
    /// The first assertion is what makes this a fence test rather than a
    /// tautology: the layer is Metal-eligible with the flag clear, so
    /// the refusal is attributable to the flag and nothing else.
    #[test]
    fn metal_attention_refuses_a_layer_whose_qk_norm_runs_after_rope() {
        let mut decoder = Decoder::new_random_small(crate::config::test_dense_fixture(), 1, 32);
        let q_width = decoder.config.n_heads * decoder.config.head_dim;
        let kv_width = decoder.config.n_kv_heads * decoder.config.head_dim;
        decoder.layers[0].attn.q_norm = Some(vec![1.0; q_width].into());
        decoder.layers[0].attn.k_norm = Some(vec![1.0; kv_width].into());

        decoder.qk_norm_after_rope = false;
        assert!(
            decoder.layer_supports_metal_attn(&decoder.layers[0]),
            "the fence below would prove nothing if this layer were ineligible anyway"
        );
        decoder.qk_norm_after_rope = true;
        assert!(!decoder.layer_supports_metal_attn(&decoder.layers[0]));
    }
}
