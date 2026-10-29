//! A trait-based abstraction over the two structurally different but
//! text-in/text-out-shaped forward passes this crate has: `Decoder`
//! (GQA+RoPE, used by GLM-5.2/DeepSeek V4 Pro and every real GGUF
//! checkpoint) and Kimi K3's dedicated hybrid KDA/Gated-MLA stack
//! (`kimi_decoder`). This lets `frink-server` share one generic
//! generation loop across both engines (see
//! `frink-server::generate::generate_engine`) for the actual
//! sampling/stop-sequence logic, rather than hand-duplicating it --
//! while keeping GGUF-only features (the KV block pool, `PrefixCache`)
//! as `Decoder`-specific code layered on top, not forced into this
//! trait. The reason: Kimi's KDA state is
//! a fixed-size recurrent matrix that collapses history irreversibly,
//! so it cannot support the same restore/truncate operations
//! `KvCache` can -- unifying those too would mean either a leaky
//! abstraction or silently pretending Kimi supports something it
//! doesn't.
//!
//! `forward_token`'s `pos` parameter is meaningful for `Decoder` (used
//! directly for RoPE) but not for `KimiEngine`: Kimi's real forward
//! pass (`kimi_forward_token`) derives position purely from its own
//! per-layer state (KDA's recurrent state, MLA's growing K/V buffers)
//! -- its real signature has no `pos` parameter at all. `KimiEngine`
//! ignores the argument; this is a real architectural fact about the
//! model, not an oversight in this trait's design.

use crate::config::{KdaConfig, MlaConfig};
use crate::decoder::Decoder;
use crate::deepseek_v4_decoder::{
    deepseek_v4_forward_token, DeepseekV4DecodeState, DeepseekV4DecoderConfig,
    DeepseekV4DecoderWeights,
};
use crate::glm52_decoder::{
    glm52_forward_token, Glm52DecodeState, Glm52DecoderConfig, Glm52DecoderWeights,
};
use crate::kimi_decoder::{
    kimi_forward_token, KimiDecodeState, KimiDecoderConfig, KimiDecoderWeights,
};
use crate::kimi_tokenizer::KimiTokenizer;
use crate::tokenizer::SpecialTokens;
use frink_core::cache::KvCache;
use frink_core::weight_matrix::WeightMatrix;

mod entry;

pub use entry::Engine;

/// The shared pool-entry assertion each engine's own tests call, and the
/// probe for whether promotion happens in this process at all. See
/// `entry.rs` for why each is one function and not one copy per engine.
#[cfg(test)]
pub(crate) use entry::{assert_one_pool_entry_per_step, on_workers_promotes_here};

impl Engine for Decoder {
    type State = Vec<KvCache>;

    fn new_state(&self) -> Vec<KvCache> {
        self.config.new_kv_caches()
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn widest_layer_macs(&self) -> Option<usize> {
        Some(Decoder::widest_layer_macs(self))
    }

    /// Delegates to the inherent [`Decoder::forward_token`], which is
    /// itself promoted in `decoder/entry.rs`. The two wrappers nest, and
    /// nesting is free -- the inner one sees a rayon worker and returns
    /// the body directly -- so this stays a delegation rather than
    /// reaching past the entry module for a private body.
    fn forward_token_on_worker(
        &self,
        token_id: usize,
        pos: usize,
        state: &mut Self::State,
    ) -> Vec<f32> {
        Decoder::forward_token(self, token_id, pos, state)
    }
}

/// Bundles Kimi K3's weights with the three real config structs
/// `kimi_forward_token` needs, so `Engine::forward_token`'s three-
/// argument shape (`token_id`, `pos`, `state`) can wrap Kimi's real
/// four-config-argument function.
pub struct KimiEngine {
    pub weights: KimiDecoderWeights,
    pub cfg: KimiDecoderConfig,
    pub mla_cfg: MlaConfig,
    pub kda_cfg: KdaConfig,
}

impl Engine for KimiEngine {
    type State = KimiDecodeState;

    fn new_state(&self) -> KimiDecodeState {
        KimiDecodeState::new(&self.weights, &self.kda_cfg)
    }

    fn vocab_size(&self) -> usize {
        self.weights.output_head.rows()
    }

    fn forward_token_on_worker(
        &self,
        token_id: usize,
        _pos: usize,
        state: &mut Self::State,
    ) -> Vec<f32> {
        kimi_forward_token(
            &self.weights,
            &self.cfg,
            &self.mla_cfg,
            &self.kda_cfg,
            token_id,
            state,
        )
    }
}

/// GLM-5.2 dedicated DSA stack behind the same [`Engine`] trait as Kimi.
/// Synthetic / loader-backed weights only — no claim of a full real
/// ~744B serve path. Lets `generate_engine` exercise GLM without
/// forcing DSA into the GQA [`Decoder`].
pub struct Glm52Engine {
    pub weights: Glm52DecoderWeights,
    pub cfg: Glm52DecoderConfig,
}

impl Engine for Glm52Engine {
    type State = Glm52DecodeState;

    fn new_state(&self) -> Glm52DecodeState {
        Glm52DecodeState::new(&self.weights)
    }

    fn vocab_size(&self) -> usize {
        self.weights.output_head.rows()
    }

    fn forward_token_on_worker(
        &self,
        token_id: usize,
        _pos: usize,
        state: &mut Self::State,
    ) -> Vec<f32> {
        glm52_forward_token(&self.weights, &self.cfg, token_id, state)
    }
}

/// Multi-layer MLA stack for DeepSeek-2 / Mistral-4-style GGUF serve.
///
/// Uses [`crate::mla::mla_forward_token`] with asymmetric K/V caches
/// (plain `Vec<f32>`, not [`KvCache`]). Layers
/// `[0, leading_dense)` use dense SwiGLU; later layers use MoE
/// (`frink_moe`) when the GGUF carries experts — fail-closed at load if
/// expert tensors are missing.
pub struct MlaEngine {
    pub embedding: WeightMatrix,
    pub layers: Vec<MlaLayerWeights>,
    pub final_norm: Vec<f32>,
    pub output_head: WeightMatrix,
    pub mla_cfg: MlaConfig,
    pub rms_norm_eps: f32,
    pub hidden_dim: usize,
    /// Present when any layer uses [`MlaLayerFfn::Moe`].
    pub moe: Option<MlaMoeRuntime>,
    /// YaRN, when the file declares it (`crate::mla_yarn`): the `pe`
    /// frequency rewrite, its magnitude and the softmax scale.
    pub yarn: Option<crate::mla_yarn::MlaYarn>,
}

/// MoE routing knobs shared by every MoE layer (DeepSeek-2 / Mistral-4).
#[derive(Debug, Clone)]
pub struct MlaMoeRuntime {
    pub n_experts_active: usize,
    pub gating: frink_moe::GatingFunction,
    pub norm_topk_prob: bool,
    pub expert_weights_scale: f32,
}

/// One dense layer's FFN: the gate / up / down triple and the
/// activation that combines the first two, carried TOGETHER so the
/// forward pass cannot run a `plm` layer's ungated ReLU-squared
/// weights through SwiGLU. `act` is the architecture's
/// (`crate::mla_arch::MlaArch::dense_act`); for an ungated one the
/// loader aliases `gate` to `ffn_up`, as the generic loader does for
/// `arcee`, and `frink_moe::run_expert` skips the aliased matmul.
pub struct MlaDenseFfn {
    pub weights: frink_moe::ExpertWeights,
    pub act: frink_moe::GluAct,
}

pub struct MlaMoeFfn {
    pub router: WeightMatrix,
    pub experts: Vec<frink_moe::ExpertWeights>,
    pub shared_expert: frink_moe::ExpertWeights,
    /// Optional aux-loss-free bias (`blk.N.exp_probs_b.bias`).
    pub exp_probs_bias: Option<Vec<f32>>,
}

pub enum MlaLayerFfn {
    Dense(MlaDenseFfn),
    Moe(MlaMoeFfn),
}

pub struct MlaLayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn: crate::mla::MlaAttnWeights,
    pub ffn_norm: Vec<f32>,
    pub ffn: MlaLayerFfn,
}

pub struct MlaDecodeState {
    pub layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl MlaEngine {
    pub fn new_state(&self) -> MlaDecodeState {
        MlaDecodeState {
            layers: (0..self.layers.len())
                .map(|_| (Vec::new(), Vec::new()))
                .collect(),
        }
    }

    fn moe_ffn_forward(&self, ffn: &MlaMoeFfn, x: &[f32]) -> Vec<f32> {
        use frink_moe::{
            combine_expert_outputs, route_top_k, route_top_k_sigmoid_with_bias, run_expert,
            GatingFunction, GluAct,
        };
        let moe = self
            .moe
            .as_ref()
            .expect("MlaLayerFfn::Moe requires MlaEngine.moe");
        let router_logits = ffn.router.apply(x);
        let decision = match (moe.gating, ffn.exp_probs_bias.as_deref()) {
            (GatingFunction::Sigmoid, Some(bias)) => route_top_k_sigmoid_with_bias(
                &router_logits,
                bias,
                moe.n_experts_active,
                moe.norm_topk_prob,
                moe.expert_weights_scale,
            ),
            _ => {
                let mut d = route_top_k(
                    &router_logits,
                    moe.n_experts_active,
                    moe.gating,
                    moe.norm_topk_prob,
                );
                if (moe.expert_weights_scale - 1.0).abs() > f32::EPSILON {
                    for w in d.weights.iter_mut() {
                        *w *= moe.expert_weights_scale;
                    }
                }
                d
            }
        };
        let routed: Vec<(Vec<f32>, f32)> = decision
            .expert_ids
            .iter()
            .zip(decision.weights.iter())
            .map(|(&e, &w)| (run_expert(x, &ffn.experts[e], GluAct::Swiglu), w))
            .collect();
        let shared = run_expert(x, &ffn.shared_expert, GluAct::Swiglu);
        combine_expert_outputs(&routed, &[shared], x.len())
    }
}

impl Engine for MlaEngine {
    type State = MlaDecodeState;

    fn new_state(&self) -> MlaDecodeState {
        MlaEngine::new_state(self)
    }

    fn vocab_size(&self) -> usize {
        self.output_head.rows()
    }

    fn forward_token_on_worker(
        &self,
        token_id: usize,
        _pos: usize,
        state: &mut Self::State,
    ) -> Vec<f32> {
        use frink_core::matmul::rms_norm;
        let mut hidden = self.embedding.dequant_row(token_id);
        for (layer, (k_cache, v_cache)) in self.layers.iter().zip(state.layers.iter_mut()) {
            let normed = rms_norm(&hidden, &layer.attn_norm, self.rms_norm_eps);
            let attn_out = crate::mla::mla_forward_token(
                &layer.attn,
                &self.mla_cfg,
                self.yarn.as_ref(),
                &normed,
                self.rms_norm_eps,
                k_cache,
                v_cache,
            );
            for (h, a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }
            let ffn_in = rms_norm(&hidden, &layer.ffn_norm, self.rms_norm_eps);
            let down = match &layer.ffn {
                MlaLayerFfn::Dense(d) => frink_moe::run_expert(&ffn_in, &d.weights, d.act),
                MlaLayerFfn::Moe(m) => self.moe_ffn_forward(m, &ffn_in),
            };
            for (h, d) in hidden.iter_mut().zip(down.iter()) {
                *h += d;
            }
        }
        let final_normed = rms_norm(&hidden, &self.final_norm, self.rms_norm_eps);
        self.output_head.apply(&final_normed)
    }
}

/// DeepSeek V4 synthetic stack behind [`Engine`]. Preset `deepseek_v4_pro`
/// remains a sketch until a real GGUF loader + incremental DSV4 KV land.
pub struct DeepseekV4Engine {
    pub weights: DeepseekV4DecoderWeights,
    pub cfg: DeepseekV4DecoderConfig,
}

impl Engine for DeepseekV4Engine {
    type State = DeepseekV4DecodeState;

    fn new_state(&self) -> DeepseekV4DecodeState {
        DeepseekV4DecodeState::new(self.weights.embedding.shape[1])
    }

    fn vocab_size(&self) -> usize {
        self.weights.output_head.rows()
    }

    fn forward_token_on_worker(
        &self,
        token_id: usize,
        _pos: usize,
        state: &mut Self::State,
    ) -> Vec<f32> {
        deepseek_v4_forward_token(&self.weights, &self.cfg, token_id, state)
    }
}

/// A minimal text<->token-id interface shared by every real tokenizer
/// this crate has, regardless of each one's native id width
/// (`GgufBpeTokenizer`/`GgufSpmTokenizer`/`GgufUnigramTokenizer` use
/// `u32`, `KimiTokenizer` also uses `u32`) -- lets a generic generation
/// loop encode/decode without caring which concrete tokenizer it was
/// given.
pub trait TextTokenizer {
    /// `specials` is llama.cpp's `parse_special`: whether a literal
    /// marker in `text` is the token it names or the characters it is
    /// written with. See [`SpecialTokens`] for which callers want which.
    fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<usize>;
    fn decode(&self, ids: &[usize]) -> String;

    /// The raw bytes, before any UTF-8 decision is made about them.
    ///
    /// A caller decoding ONE token at a time needs these: a character
    /// split across two tokens is two invalid fragments, and `decode`
    /// resolves each to U+FFFD separately, losing the bytes (#124).
    ///
    /// The default is correct for any tokenizer whose tokens are whole
    /// text, and no worse than `decode` for one whose tokens are not --
    /// but such a tokenizer should override this.
    fn decode_bytes(&self, ids: &[usize]) -> Vec<u8> {
        self.decode(ids).into_bytes()
    }
}

impl TextTokenizer for KimiTokenizer {
    fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<usize> {
        KimiTokenizer::encode(self, text, specials)
            .into_iter()
            .map(|id| id as usize)
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        KimiTokenizer::decode(self, &ids32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_dense_fixture;

    /// Locks in the refactor: calling `Decoder` through the generic
    /// `Engine` trait must be bit-identical to calling its own
    /// `forward_token`/`forward_batch` directly -- the whole point of
    /// the trait is that `frink-server`'s generic generation loop can
    /// use it as a drop-in replacement with zero numeric difference.
    #[test]
    fn decoder_via_engine_trait_matches_direct_forward_token_calls() {
        let decoder = Decoder::new_random_small(test_dense_fixture(), 2, 64);
        let tokens = [3usize, 7, 1, 9];

        let mut direct_caches: Vec<KvCache> = decoder.config.new_kv_caches();
        let mut direct_logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            direct_logits = decoder.forward_token(tok, pos, &mut direct_caches);
        }

        let mut engine_state = Engine::new_state(&decoder);
        let mut engine_logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            engine_logits = Engine::forward_token(&decoder, tok, pos, &mut engine_state);
        }

        assert_eq!(engine_logits, direct_logits);
        assert_eq!(Engine::vocab_size(&decoder), decoder.config.vocab_size);
    }

    /// Same equivalence, but against `forward_batch`'s independent
    /// computation (the ground truth every other test in this
    /// workspace already uses) rather than a second manual loop.
    #[test]
    fn decoder_via_engine_trait_matches_forward_batch_ground_truth() {
        let decoder = Decoder::new_random_small(test_dense_fixture(), 2, 64);
        let tokens = vec![2usize, 5, 8];

        let mut batch_caches: Vec<KvCache> = decoder.config.new_kv_caches();
        let batch_logits = decoder.forward_batch(&tokens, 0, &mut batch_caches);
        let ground_truth = batch_logits.last().unwrap().clone();

        let mut engine_state = Engine::new_state(&decoder);
        let mut engine_logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            engine_logits = Engine::forward_token(&decoder, tok, pos, &mut engine_state);
        }

        // Tight tolerance, not bit equality: batched prefill runs the
        // blocked three-pass softmax while per-token decode keeps the
        // online accumulator, and the two round differently in the last
        // ulp (they were bit-identical only while both were online).
        assert_eq!(engine_logits.len(), ground_truth.len());
        for (i, (e, g)) in engine_logits.iter().zip(ground_truth.iter()).enumerate() {
            assert!(
                (e - g).abs() < 1e-5,
                "logit {i}: engine {e} vs forward_batch {g}"
            );
        }
    }
}
