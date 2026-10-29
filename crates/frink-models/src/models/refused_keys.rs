//! The metadata keys a file must not declare, derived from
//! [`ArchFacts`]: moved here from `capability.rs` (#403 step 1) so the
//! loader reads them off the facts. `capability::unsupported_feature_keys`
//! and `capability::unsupported_scaling_keys` remain as name-keyed
//! wrappers for the tests that call them.

use crate::capability::DecoderFamily;
use crate::scalar_multipliers::{AttentionScaleKey, LogitScaleUse, ResidualScaleUse};

use super::ArchFacts;

impl ArchFacts {
    /// Metadata keys that, when present with a nonzero value, require math
    /// frink's generic decoder does not implement *unless* the architecture
    /// profile opts into those features (Gemma family), or the architecture
    /// is named in [`crate::capability::LOGIT_SOFTCAP_ARCHITECTURES`] for the softcaps.
    pub fn unsupported_feature_keys(&self, arch: &str) -> Vec<(String, &'static str)> {
        // Gemma family implements softcap + SWA pattern; others still refuse.
        if matches!(self.family(), Some(DecoderFamily::GemmaFamily)) {
            return Vec::new();
        }
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let mut out = Vec::new();
        if !self.scaling.logit_softcap {
            out.push((
                key("attention.logit_softcapping"),
                "attention logit soft-capping (Gemma 2+); not implemented in the generic decoder",
            ));
            // The spelling llama.cpp's converters ACTUALLY write
            // (`llama-arch.cpp:213` is `%s.attn_logit_softcapping`). The
            // line above is a spelling no converter emits, so this gate has
            // never fired for any non-Gemma architecture -- while
            // `loader.rs` reads BOTH spellings and applies the value.
            //
            // A checkpoint declaring an attention softcap was therefore not
            // refused; it ran with the generic formula. For `grok` that is a
            // wrong answer rather than an approximation: `grok.cpp` folds
            // the real attention scale INTO the softcap and passes
            // `kq_scale = 1.0f`, which the generic path does not do.
            //
            // A gate that cannot fire is not a gate, and it looked exactly
            // like one.
            out.push((
                key("attn_logit_softcapping"),
                "attention logit soft-capping (Gemma 2+); not implemented in the generic decoder",
            ));
            out.push((
                key("final_logit_softcapping"),
                "final logit soft-capping (Gemma 2+); not implemented in the generic decoder",
            ));
        }
        // `{arch}.nextn_predict_layers` WAS refused here, for every
        // architecture, with the reason that frink's `n_layers` IS
        // `block_count` and it would run the MTP head as decoder layers.
        // `crate::mtp_blocks::trunk_layers` subtracts the blocks now for
        // exactly the seventeen graphs whose `load_arch_hparams` reads the
        // key (`NEXTN_READERS`, measured) and still refuses a nonzero value
        // on any other -- where llama.cpp itself would run every block and
        // then fail on the unread `nextn.*` tensors. One place decides both
        // halves, so the reader table and the refusal cannot drift apart.
        // `{arch}.attention.sliding_window_pattern` WAS refused here,
        // with the reason "not implemented in the generic decoder".
        // That reason was false, and had been for some time: the
        // alternating pattern lives in `ModelConfig::layer_sliding_window`,
        // which implements BOTH phases and which `gpt-oss` -- a
        // `StandardGqa` row, not a Gemma one -- has relied on since it
        // was audited against libllama.
        //
        // What the gate really did was make the loader's own read of
        // that key (`swa_pattern`) unreachable for every non-Gemma
        // architecture: llama.cpp lets the file override the
        // architecture's hardcoded period, frink refused any file that
        // tried. `plamo3` is the case that proves it -- its converter
        // writes the key verbatim (`conversion/plamo.py:178`) -- and
        // `tests/fixture_away_graphs.rs` now drives a period of 2 out of
        // a plamo3 fixture and compares against llama.cpp's own graph on
        // all three forward paths, with the phase and the window
        // sabotaged separately.
        //
        // The real gap the key could hide was NOT the pattern: it was
        // that llama.cpp accepts the value as a scalar OR an n_layer-long
        // ARRAY (`ml.get_key_or_arr`), and frink carried one scalar
        // period. The array is `crate::swa_layers` now, read the way each
        // graph reads it -- ignored, honoured, or broadcast -- so neither
        // shape is refused here or anywhere else.
        //
        // `{arch}.moe_latent_size` (`LLM_KV_MOE_LATENT_SIZE`,
        // `nemotron-h.cpp:21,36,82-85,206-208`): the routed experts run in a
        // LATENT width the layer projects into with `ffn_latent_down` and
        // out of with `ffn_latent_up`, while the router and the shared
        // expert read the unprojected input. Nemotron-3 Nano writes no such
        // key; Nemotron-3 Super does. The generic MoE bodies run their
        // experts at `hidden_dim`, so a nonzero value stops here, by name.
        out.push((
            key("moe_latent_size"),
            "a latent MoE (nemotron-h.cpp:206-208: the experts read `ffn_latent_down(x)` and \
             their sum is `ffn_latent_up`ed back), which the generic MoE bodies, which run \
             the experts at hidden_dim, do not have",
        ));
        // `{arch}.decision.block_count` (`LLM_KV_DECISION_BLOCK_COUNT`, new at
        // the `604cc430` pin): the file's LAST blocks are a decision head, not
        // decoder layers. `lfm2.cpp` (an architecture frink serves) reads it
        // at `load_arch_hparams`, requires `attention.causal = false` beside
        // it, builds no memory (`llama-model.cpp:2808`) and runs the trunk
        // bidirectionally into a biased-LayerNorm scorer that emits three
        // scores per token and no logits; `conversion/lfm2.py:258` writes it
        // for every LFM2-d1-omni export. The generic decoder would run those
        // head blocks as causal layers and sample from an LM head the file
        // does not have. `decision.type` ALONE is not refused: the
        // `lfm2-d1` / `openjev` kinds read the ordinary LM logits of a
        // causal model, which this path computes.
        out.push((
            key("decision.block_count"),
            "a DECISION head (lfm2.cpp: `decision.block_count` trailing blocks run as a \
             non-causal scorer that emits scores instead of logits, with no KV memory); frink \
             serves the causal trunk only",
        ));
        out
    }

    /// Scalar multipliers a checkpoint can declare in **metadata** that the
    /// generic decoder does not apply, with the value that means "no-op".
    ///
    /// These are the blind spot left by
    /// [`crate::loader::assert_every_tensor_consumed`]: that gate catches a
    /// missing *tensor*, but Granite / MiniCPM / Command-R style multipliers
    /// are hparams, not weights, so a checkpoint carrying them loads
    /// cleanly, runs at full speed, and computes a graph scaled differently
    /// from the one the checkpoint was trained as. Nothing says so.
    ///
    /// llama.cpp key names (`llama-arch.cpp`):
    /// `%s.logit_scale` (`LLM_KV_LOGIT_SCALE`), `%s.residual_scale`,
    /// `%s.embedding_scale`, `%s.attention.scale`. Granite reads all four
    /// (`src/models/granite.cpp::load_arch_hparams`); MiniCPM and
    /// Command-R/Cohere2 read the subset they use.
    ///
    /// **This list is DERIVED, never restated.** Which of the four an
    /// architecture applies lives in
    /// [`crate::scalar_multipliers::multiplier_support`], and this function
    /// is exactly its complement: a key appears here if and only if that
    /// table says the graph does not apply it. Two hand-written lists is the
    /// shape that once let this repo refuse a key it implemented and
    /// implement a key it refused, and the Gemma family used to be exempted
    /// from ALL FOUR of these wholesale on the strength of implementing two,
    /// so a hand-written `gemma3.residual_scale` would have loaded and been
    /// ignored.
    ///
    /// `residual_scale` is the one that reaches furthest: it multiplies the
    /// attention and FFN branch outputs before every residual add, so on an
    /// architecture that does not implement it a declared value would have
    /// to be dropped by every CPU decode/prefill/multi-seq path *and* by the
    /// fused Metal kernels that fold the residual in.
    ///
    /// The no-op value differs by key: the three `*_scale` multipliers are
    /// `1.0`, while llama.cpp's `f_attention_scale` uses `0.0` as its
    /// "unset, use 1/sqrt(head_dim)" sentinel.
    pub fn unsupported_scaling_keys(&self, arch: &str) -> Vec<(String, &'static str, f32)> {
        let support = self.scaling.multipliers;
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let mut out = Vec::new();
        if support.logit == LogitScaleUse::NotApplied {
            out.push((
                key("logit_scale"),
                "logit multiplier (Granite / Command-R `logits_scaling`); not applied by the generic decoder",
                1.0,
            ));
        }
        if support.residual == ResidualScaleUse::NotRead {
            out.push((
                key("residual_scale"),
                "residual multiplier (Granite `residual_multiplier`); not applied by the generic decoder",
                1.0,
            ));
        }
        if !support.embedding {
            out.push((
                key("embedding_scale"),
                "embedding multiplier (Granite / MiniCPM `embedding_multiplier`); the generic decoder only scales embeddings for the Gemma and Granite families",
                1.0,
            ));
        }
        // Two spellings of one slot, and an architecture reads at most one
        // of them: the OTHER stays refused. `grok` reads `output_scale` and
        // never `scale`; Granite the reverse; everyone else neither.
        if support.attention != AttentionScaleKey::Scale {
            out.push((
                key("attention.scale"),
                "explicit attention score scale (Granite `attention_multiplier`); the generic decoder always uses 1/sqrt(head_dim)",
                0.0,
            ));
        }
        if support.attention != AttentionScaleKey::OutputScale {
            // Applied as-is by the one graph that reads it (`grok.cpp`, no
            // sentinel), so there is no value that means "off" -- the
            // no-op here is the kernels' own scale expressed as a key, which
            // no converter writes for a non-Grok architecture. A file
            // declaring ANY other value is refused.
            out.push((
                key("attention.output_scale"),
                "attention output scale (Grok `attn_output_multiplier`, applied inside its tanh softcap); the generic decoder always uses 1/sqrt(head_dim)",
                0.0,
            ));
        }
        out
    }
}
