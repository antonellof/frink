//! RoPE as it reaches one head, and the two scalar corrections that
//! ride beside it.
//!
//! Four per-architecture facts live here, and this repo has lost three
//! of them at least once by leaving them inline in a decode body:
//!
//! * **Pairing.** `Norm` rotates adjacent pairs, `Neox` rotates split
//!   halves. Applying NeoX to `llama` was the root cause of the
//!   Llama-3.1-8B early-stop bug.
//! * **Rotary width.** Phi-3/Phi-4 rotate `rope.dimension_count` of each
//!   head and pass the tail through untouched.
//! * **The per-LAYER pair.** The frequency base and the per-band
//!   divisors are ONE answer, and llama.cpp varies both per layer
//!   (`llama-model.cpp:2029-2035`, consumed at `gemma3.cpp:112-121`).
//!   frink varied only the base, which rotated Gemma-3 4B/12B/27B's
//!   sliding layers at scaled positions llama.cpp leaves unscaled -- see
//!   [`crate::config::RopeFreqs`].
//!   [`crate::config::ModelConfig::layer_rope`] now hands both out
//!   together so they cannot drift apart again.
//! * **`attn_factor` and `attention_scale`.** Two multipliers that are
//!   not the same thing: the first is YaRN's `mscale` on the rotated
//!   channels of q and k, the second is llama.cpp's `f_attention_scale`
//!   pre-baked into Q. `attention_scale` reached two of four host sites
//!   before it became one helper.
//! * **The per-position attention temperature.** A THIRD multiplier on
//!   Q, and the only one that varies by token
//!   (`crate::attn_temperature`); it is applied where `attention_scale`
//!   is, by one helper taking the row's position, so that the three
//!   host bodies cannot disagree about which position a row is at.

use frink_core::attention::{
    apply_rope, apply_rope_interleaved, apply_rope_interleaved_with_freq_factors,
    apply_rope_with_freq_factors,
};

use super::Decoder;

impl Decoder {
    /// Applies RoPE to one head's Q or K slice. Dispatches on both
    /// `rope_layout` (Norm = adjacent-pair / NeoX = split-half -- see
    /// `RopeLayout`) and whether this checkpoint carries a real
    /// `rope_freqs.weight` tensor (Llama 3/3.1/3.2's per-band frequency
    /// correction). Getting the layout wrong for `llama` was the real
    /// root cause of the Llama-3.1-8B early-stop bug: frink applied
    /// NeoX pairing to an architecture that needs Norm.
    ///
    /// `theta` and `freq_factors` are ONE answer, taken together from
    /// [`crate::config::ModelConfig::layer_rope`]. Passing them
    /// separately is how the sliding-window RoPE bug happened: the base
    /// varied per layer and the divisors did not.
    pub(crate) fn apply_rope_head_theta(
        &self,
        slice: &mut [f32],
        pos: usize,
        theta: f32,
        freq_factors: Option<&[f32]>,
        rot_dim: Option<usize>,
    ) {
        use crate::config::RopeLayout;
        // Partial rotary (llama.cpp `hparams.n_rot(il)` < `n_embd_head_k`,
        // GGUF `<arch>.rope.dimension_count` and its `_swa` twin):
        // Phi-3/Phi-4 rotate only the first 96 of each 128-wide head and
        // pass the remaining 32 through untouched; Step-3.5 rotates half
        // the head on its full-attention layers and all of it on the
        // sliding ones. Rotating the whole head instead is not a subtle
        // error -- it moves dimensions the model never trained to be
        // position-dependent. The width is THIS LAYER's, from
        // `ModelConfig::layer_rope`, never the model-wide field.
        let slice = match rot_dim {
            Some(rot) if rot < slice.len() => &mut slice[..rot],
            _ => slice,
        };
        match (self.config.rope_layout, freq_factors) {
            (RopeLayout::Norm, Some(freq_factors)) => {
                apply_rope_interleaved_with_freq_factors(slice, pos, theta, freq_factors)
            }
            (RopeLayout::Norm, None) => apply_rope_interleaved(slice, pos, theta),
            (RopeLayout::Neox, Some(freq_factors)) => {
                apply_rope_with_freq_factors(slice, pos, theta, freq_factors)
            }
            (RopeLayout::Neox, None) => apply_rope(slice, pos, theta),
        }
    }

    /// A layer llama.cpp does not rotate is a NO-OP here, not a
    /// rotation at some default base: `ModelConfig::layer_rope` answers
    /// `None` for it (`crate::rope_layers`), and the `let ... else`
    /// below is the CPU half of that rule. Rotating anyway is the
    /// silent failure the rule exists to stop -- fluent output from
    /// positions the checkpoint never encodes that way.
    pub(crate) fn apply_rope_head_layer(&self, slice: &mut [f32], pos: usize, layer_idx: usize) {
        let Some(crate::config::LayerRopeParams {
            theta,
            freq_factors,
            rot_dim,
        }) = self.config.layer_rope(layer_idx)
        else {
            return;
        };
        self.apply_rope_head_theta(slice, pos, theta, freq_factors.map(|f| &f[..]), rot_dim)
    }

    /// llama.cpp's RoPE `mscale` (ggml `rope_yarn`), applied where the
    /// QKV biases and QK-norms are: multiplying `cos`/`sin` by a constant
    /// is the same as scaling the vector RoPE rotates, and rotation is
    /// linear, so pre-scaling q and k here is exactly what the kernel
    /// would do post-hoc — without a new uniform on five backends' RoPE
    /// kernels.
    ///
    /// Both q and k are scaled, so attention logits carry `m²`, which is
    /// the whole observable effect (V is untouched, and k enters the
    /// cache scaled exactly as llama.cpp's does).
    ///
    /// `layer_idx` is here because `attn_factor` is an ARGUMENT to
    /// `ggml_rope_ext`, not a separate op: a layer llama.cpp does not
    /// rotate never reaches `rope_yarn` and never takes the scale
    /// either. Applying it to an unrotated layer would leave q and k
    /// scaled by `m` with no rotation to justify it, and attention
    /// logits carrying `m²` -- which is the whole observable effect of
    /// this function, so the error would be exactly as large as the
    /// feature. Taking the layer rather than a caller-computed bool is
    /// what stops this from being a second answer to the question
    /// `ModelConfig::layer_rope` already answers.
    #[inline]
    pub(crate) fn apply_rope_attn_factor(&self, q: &mut [f32], k: &mut [f32], layer_idx: usize) {
        let m = self.config.rope_attn_factor;
        let Some(rope) = self.config.layer_rope(layer_idx) else {
            return;
        };
        if m == 1.0 {
            return;
        }
        // ggml folds `attn_factor` into cos_theta/sin_theta inside
        // `rope_yarn` (ops.cpp), so it reaches ONLY the rotated channels;
        // `[n_rot, head_dim)` is then copied through untouched by the
        // "fill the remain channels with data from src tensor" loop.
        // Scaling the pass-through tail as well is a different graph, and
        // `frink parity` caught it as the one DRIFT verdict in a
        // 17-model sweep: Phi-4-mini rotates 96 of 128 dims with
        // attn_factor 1.1902, so 32 dims per head were scaled that
        // llama.cpp leaves alone.
        let head_dim = self.config.head_dim;
        let rot = rope.rot_dim.unwrap_or(head_dim).min(head_dim);
        for buf in [q, k] {
            for head in buf.chunks_mut(head_dim) {
                let n = rot.min(head.len());
                for v in head[..n].iter_mut() {
                    *v *= m;
                }
            }
        }
    }

    /// An architecture's explicit attention score scale
    /// (`ModelConfig::attention_scale`), applied to Q after RoPE.
    ///
    /// llama.cpp's Gemma graphs scale Q themselves and then call
    /// `build_attn(..., 1.0f)`; frink's kernels always apply their own
    /// `1/sqrt(head_dim)`, so the compensation has to be folded into Q
    /// here for the NET score scale to equal `attention_scale`.
    ///
    /// One helper rather than one copy per host body, because that is
    /// exactly how this feature came to be applied at two of four sites:
    /// `forward_hidden_batch_inner` and `forward_multi_seq_kv` computed a
    /// plausible distribution at the wrong temperature and nothing
    /// failed. Elementwise, so a `[batch, n_heads*head_dim]` Q batch is
    /// the same call as one row's.
    #[inline]
    pub(crate) fn apply_attention_scale(&self, q: &mut [f32]) {
        let Some(scale) = self.config.attention_scale else {
            return;
        };
        let compensate = scale * (self.config.head_dim as f32).sqrt();
        for v in q.iter_mut() {
            *v *= compensate;
        }
    }

    /// llama.cpp's per-position attention temperature
    /// (`ModelConfig::attn_temperature`), applied to a `[rows, q_width]`
    /// Q batch after RoPE and the post-RoPE QK-norm, where
    /// `mistral3.cpp:153-156` multiplies `Qcur` by the `[n_tokens]`
    /// input -- every head and channel of a token's Q by that token's
    /// scalar.
    ///
    /// Takes the position as a function of the row, because the three
    /// host bodies spell it three ways (`pos`, `start_pos + b`,
    /// `positions[b]`) and a helper that took a slice would have made
    /// the prefill body allocate one to say `start_pos + b`. A
    /// `[n_tokens]` input is exactly what llama.cpp hands the graph,
    /// so this is the same shape, not a rearrangement of it.
    #[inline]
    pub(crate) fn apply_attn_temperature(
        &self,
        layer_idx: usize,
        q: &mut [f32],
        q_width: usize,
        pos_of_row: impl Fn(usize) -> usize,
    ) {
        let Some(temp) = self.config.attn_temperature else {
            return;
        };
        // `llama4.cpp:163-177`: the multiply is the `else` of the RoPE
        // branch, so a rotating layer is left alone.
        if temp.unrotated_layers_only && self.config.layer_rotates(layer_idx) {
            return;
        }
        temp.apply_rows(q, q_width, pos_of_row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phi-3/Phi-4 rotate `rope.dimension_count` of each head and pass
    /// the rest through. The tail staying bit-identical is the whole
    /// property: rotating it would make dimensions position-dependent
    /// that the model never trained that way.
    #[test]
    fn partial_rotary_leaves_the_tail_untouched() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.rope_layout = crate::config::RopeLayout::Neox;
        cfg.rope_freqs = None;
        cfg.rope_dim = Some(4);
        let decoder = Decoder::new_random_small(cfg, 1, 32);

        let mut head: Vec<f32> = (0..8).map(|i| 1.0 + i as f32).collect();
        let before = head.clone();
        let rot_dim = decoder.config.layer_rope(0).expect("rotates").rot_dim;
        decoder.apply_rope_head_theta(&mut head, 3, 10000.0, None, rot_dim);

        assert_eq!(
            &head[4..],
            &before[4..],
            "dims at or past rope_dim must not rotate"
        );
        assert!(
            head[..4] != before[..4],
            "dims below rope_dim must rotate at a non-zero position"
        );
    }

    /// `attn_factor` is a magnitude scale folded into cos/sin inside
    /// ggml's `rope_yarn`, so it can only ever touch the rotated
    /// channels. The pass-through tail must come out bit-identical —
    /// scaling it is a different graph, and it was one, until
    /// `frink parity` reported Phi-4-mini as the single DRIFT in a
    /// 17-model sweep against llama.cpp.
    #[test]
    fn attn_factor_scales_only_the_rotated_channels() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.n_heads = 2;
        cfg.n_kv_heads = 2;
        cfg.rope_dim = Some(4);
        cfg.rope_attn_factor = 2.0;
        let decoder = Decoder::new_random_small(cfg, 1, 32);

        // Two heads, so a per-head slice bug cannot hide behind a single
        // head that happens to span the whole buffer.
        let mut q: Vec<f32> = (0..16).map(|i| 1.0 + i as f32).collect();
        let mut k: Vec<f32> = (0..16).map(|i| 1.0 + i as f32).collect();
        let before = q.clone();
        decoder.apply_rope_attn_factor(&mut q, &mut k, 0);

        for h in 0..2 {
            let base = h * 8;
            for i in 0..4 {
                assert_eq!(
                    q[base + i],
                    before[base + i] * 2.0,
                    "rotated channel {i} of head {h} must be scaled"
                );
            }
            for i in 4..8 {
                assert_eq!(
                    q[base + i],
                    before[base + i],
                    "pass-through channel {i} of head {h} must be untouched"
                );
            }
        }
        assert_eq!(q, k, "q and k take the same magnitude scale");
    }

    /// With no partial rotary the whole head is rotated, so the whole
    /// head takes the scale — the narrow case must not become the rule.
    #[test]
    fn attn_factor_scales_the_whole_head_without_partial_rotary() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.n_heads = 1;
        cfg.n_kv_heads = 1;
        cfg.rope_dim = None;
        cfg.rope_attn_factor = 3.0;
        let decoder = Decoder::new_random_small(cfg, 1, 32);

        let mut q: Vec<f32> = (0..8).map(|i| 1.0 + i as f32).collect();
        let mut k = q.clone();
        let before = q.clone();
        decoder.apply_rope_attn_factor(&mut q, &mut k, 0);
        for i in 0..8 {
            assert_eq!(q[i], before[i] * 3.0);
        }
    }

    /// The same call with no `rope_dim` must rotate everything, so the
    /// narrow case cannot silently become the default.
    #[test]
    fn full_rotary_still_rotates_the_whole_head() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.rope_layout = crate::config::RopeLayout::Neox;
        cfg.rope_freqs = None;
        cfg.rope_dim = None;
        let decoder = Decoder::new_random_small(cfg, 1, 32);

        let mut head: Vec<f32> = (0..8).map(|i| 1.0 + i as f32).collect();
        let before = head.clone();
        let rot_dim = decoder.config.layer_rope(0).expect("rotates").rot_dim;
        decoder.apply_rope_head_theta(&mut head, 3, 10000.0, None, rot_dim);
        assert!(head[4..] != before[4..]);
    }

    /// `attn_factor` is an argument to `ggml_rope_ext`, so a layer
    /// llama.cpp does not rotate never takes it. Before `layer_idx` was
    /// a parameter here, an unrotated layer of a model with a YaRN
    /// `mscale` would have had q and k scaled by `m` with no rotation
    /// behind it -- attention logits off by `m²` on exactly the layers
    /// the per-layer rule exists for.
    #[test]
    fn an_unrotated_layer_takes_no_attn_factor_either() {
        use crate::rope_layers::RopeLayers;
        let mut cfg = crate::config::test_dense_fixture();
        cfg.rope_attn_factor = 2.0;
        // `SlidingOnly` with no window at all: no layer slides, so no
        // layer rotates -- llama.cpp's degenerate `set_swa_pattern(1)`.
        cfg.sliding_window = None;
        cfg.rope_layers = RopeLayers::SlidingOnly;
        let decoder = Decoder::new_random_small(cfg, 1, 32);
        assert!(!decoder.config.layer_rotates(0), "the premise");

        let mut q = vec![1.0f32, -2.0, 3.0];
        let mut k = vec![0.5f32, 4.0];
        decoder.apply_rope_attn_factor(&mut q, &mut k, 0);
        assert_eq!(q, vec![1.0, -2.0, 3.0], "no rotation, no mscale");
        assert_eq!(k, vec![0.5, 4.0]);

        // And the same head DOES take it on a rotating layer, so the
        // test is about the gate and not about the scale being dead.
        let mut cfg = crate::config::test_dense_fixture();
        cfg.rope_attn_factor = 2.0;
        cfg.rope_layers = RopeLayers::All;
        let decoder = Decoder::new_random_small(cfg, 1, 32);
        let mut q = vec![1.0f32, -2.0, 3.0];
        let mut k = vec![0.5f32, 4.0];
        decoder.apply_rope_attn_factor(&mut q, &mut k, 0);
        assert_eq!(q, vec![2.0, -4.0, 6.0]);
    }

    /// `mscale` scales q and k and nothing else; `1.0` must be a literal
    /// no-op so every other model pays nothing.
    #[test]
    fn rope_attn_factor_scales_q_and_k_only() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.rope_attn_factor = 2.0;
        let decoder = Decoder::new_random_small(cfg, 1, 32);
        let mut q = vec![1.0f32, -2.0, 3.0];
        let mut k = vec![0.5f32, 4.0];
        decoder.apply_rope_attn_factor(&mut q, &mut k, 0);
        assert_eq!(q, vec![2.0, -4.0, 6.0]);
        assert_eq!(k, vec![1.0, 8.0]);

        let mut cfg = crate::config::test_dense_fixture();
        cfg.rope_attn_factor = 1.0;
        let decoder = Decoder::new_random_small(cfg, 1, 32);
        let mut q = vec![1.0f32, -2.0];
        let mut k = vec![3.0f32];
        decoder.apply_rope_attn_factor(&mut q, &mut k, 0);
        assert_eq!(q, vec![1.0, -2.0]);
        assert_eq!(k, vec![3.0]);
    }

    /// A sliding layer and a full-attention layer of the same model get
    /// DIFFERENT RoPE, in both halves: base and per-band divisors.
    ///
    /// llama.cpp splits both per layer (`llama-model.cpp:2029-2035`,
    /// consumed at `gemma3.cpp:112-121`). frink split only the base:
    /// `layer_rope_theta` varied per layer while `rope_freqs` was one
    /// global vector applied everywhere. On Gemma-3 4B/12B/27B --
    /// `rope_scaling: {linear, factor 8}`, `sliding_window_pattern = 6`
    /// -- that rotated FIVE LAYERS IN SIX at position `p/8` where
    /// llama.cpp rotates at `p`. Fluent output, worse the longer the
    /// prompt, no error. Gemma-3-1B, the audited fixture, is the one
    /// size with no `rope_scaling` at all, so it could not show this.
    #[test]
    fn a_sliding_layer_ropes_unscaled_where_a_full_attention_layer_ropes_scaled() {
        use crate::config::{RopeFreqs, RopeLayout};

        const FACTOR: f32 = 8.0;
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.rope_layout = RopeLayout::Norm;
        cfg.rope_theta = 1_000_000.0;
        // Gemma-3's SWA base: `gemma3.cpp:11` reads only the BASE key and
        // leaves `rope_freq_base_train_swa` at its 10000 default.
        cfg.rope_theta_swa = Some(10_000.0);
        cfg.sliding_window = Some(4);
        // Period 2, last-dense: layer 0 slides, layer 1 does not.
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(2, false);
        cfg.rope_freqs = Some(RopeFreqs {
            full: vec![FACTOR; 4].into(),
            swa: Some(vec![1.0; 4].into()),
        });
        assert_eq!(cfg.layer_sliding_window(0), Some(4), "layer 0 must slide");
        assert_eq!(cfg.layer_sliding_window(1), None, "layer 1 must be dense");

        let decoder = Decoder::new_random_small(cfg, 2, 32);
        let source: Vec<f32> = (0..8).map(|i| 1.0 + i as f32).collect();
        let pos = 7;

        // The sliding layer: llama.cpp's `freq_scale_l` is the 1.0
        // default, so no divisor at all.
        let mut sliding = source.clone();
        decoder.apply_rope_head_layer(&mut sliding, pos, 0);
        let mut want_sliding = source.clone();
        apply_rope_interleaved(&mut want_sliding, pos, 10_000.0);
        assert_eq!(
            sliding, want_sliding,
            "a Gemma-3 sliding layer rotates at the raw position, base 10000"
        );

        // The full-attention layer: the trained linear scale, folded
        // into the per-band divisors.
        let mut full = source.clone();
        decoder.apply_rope_head_layer(&mut full, pos, 1);
        let mut want_full = source.clone();
        apply_rope_interleaved_with_freq_factors(&mut want_full, pos, 1_000_000.0, &[FACTOR; 4]);
        assert_eq!(
            full, want_full,
            "a Gemma-3 full-attention layer rotates at p/8, base 1e6"
        );

        // Not vacuous: the two layers must actually disagree, which is
        // the whole property. Before the fix both took `full`.
        assert_ne!(
            sliding, full,
            "if these agree the test proves nothing about per-layer RoPE"
        );

        // And the sliding layer must NOT be the scaled rotation, which
        // is precisely what it used to get.
        let mut was_wrong = source.clone();
        apply_rope_interleaved_with_freq_factors(&mut was_wrong, pos, 10_000.0, &[FACTOR; 4]);
        assert_ne!(
            sliding, was_wrong,
            "the sliding layer must not carry the full-attention scale"
        );
    }

    /// `swa: None` means "inherit", which is what llama.cpp does for
    /// every architecture that assigns `rope_freq_scale_train_swa` from
    /// `rope_freq_scale_train` (`gemma2.cpp:11` and the ten others in
    /// `capability::swa_rope_scale_follows_model`). Gemma-2 must keep
    /// its scaling on the sliding layers.
    #[test]
    fn an_inheriting_architecture_ropes_both_kinds_of_layer_with_one_set() {
        use crate::config::{RopeFreqs, RopeLayout};

        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.rope_layout = RopeLayout::Norm;
        cfg.sliding_window = Some(4);
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(2, false);
        cfg.rope_freqs = Some(RopeFreqs {
            full: vec![8.0; 4].into(),
            swa: None,
        });
        assert!(
            !cfg.rope_freqs_vary_by_layer(),
            "an inheriting model resolves to ONE divisor set for every layer"
        );

        let decoder = Decoder::new_random_small(cfg, 2, 32);
        let source: Vec<f32> = (0..8).map(|i| 1.0 + i as f32).collect();
        let mut sliding = source.clone();
        let mut full = source.clone();
        decoder.apply_rope_head_layer(&mut sliding, 7, 0);
        decoder.apply_rope_head_layer(&mut full, 7, 1);
        assert_eq!(
            sliding, full,
            "both layers share one base and one divisor set here"
        );
    }
}
