//! What the server currently has loaded, and the one seam where a
//! route that needs to *generate* finds out that it cannot.
//!
//! # Why an enum here and not a sixth `Model` variant
//!
//! [`crate::Model`] is the generation model: eighteen `match self` arms
//! asking for a chat template, stop tokens, a BOS policy, a vocabulary
//! size, an expert-cache reading. An encoder-only checkpoint (BERT/BGE,
//! see [`frink_models::EmbeddingModel`]) has an honest answer to none
//! of them — it has no output head, so there is no vocabulary to size,
//! nothing to stop, and no turn to template. Adding it as a variant
//! would mean writing eighteen invented answers, which is exactly the
//! "compute something else" failure this engine refuses everywhere
//! else.
//!
//! The alternative considered and rejected was making those eighteen
//! arms return `Option`/`Result`. That is eighteen new failure paths
//! for one new kind of model, and it pushes the refusal down to
//! `vocab_size()` — so a caller would learn "no vocabulary size" rather
//! than "this is an embedding model". The refusal has to name the
//! model, not the field.
//!
//! So the disjunction is one level up, in [`Loaded`], and there is
//! exactly ONE place that converts "what is loaded" into "the model I
//! can decode with": [`ActiveModel::generative`], which returns a
//! `Result`. A route cannot reach a decode without going through it,
//! and what it returns on an encoder is a refusal naming the
//! checkpoint, its architecture and the route that *would* serve it.
//!
//! Everything both kinds can genuinely answer — its name, its tokenizer
//! kind, whether it is the synthetic demo — is answered by
//! [`ActiveModel`] itself, so `/v1/models` and `/health` report an
//! encoder as the loaded model rather than reporting nothing.

use std::path::PathBuf;
use std::sync::Arc;

use axum::http::StatusCode;
use axum::Json;

use frink_models::tokenizer::SpecialTokens;
use frink_models::EmbeddingModel;

use crate::{budget, serving, ApiError, Model};

/// The checkpoint this server is serving: exactly one kind, never both.
pub(crate) enum Loaded {
    /// Something with an output head. Chat, completions, embeddings by
    /// pooling its hidden states.
    Generative(Arc<Model>),
    /// An encoder-only embedding checkpoint. `/v1/embeddings` only.
    Encoder(Arc<EmbeddingModel>),
}

/// The model the server is serving *right now*, together with the
/// pieces that are built from it and must be replaced with it.
///
/// The continuous batcher owns a worker thread holding an
/// `Arc<Decoder>`, so it belongs to one specific model: keeping it in a
/// separate field would let a swap leave a batcher decoding against the
/// old weights while `Model` named the new ones. Bundling them means
/// one `Arc` swap replaces a consistent pair.
/// What a `POST /sleep` remembers, so `POST /wake_up` can replay the
/// load that produced the model it put away.
///
/// Both halves are kept because a model reaches this server two ways:
/// by id through `/admin/models/load`, which discovers it in the
/// scanned directories, and by PATH at startup from
/// `FRINK_MODEL_PATH`, which never had an id. Remembering only the id
/// would make a startup-loaded deployment unwakeable, which is the
/// common case for a single-model server.
#[derive(Debug, Clone)]
pub(crate) struct SleptModel {
    pub(crate) id: Option<String>,
    pub(crate) path: std::path::PathBuf,
}

pub(crate) struct ActiveModel {
    /// Admin-surface id (see `admin::discover`), or `None` for a model
    /// that was not discovered through it -- the synthetic fallback, or
    /// a `FRINK_MODEL_PATH` outside the scanned directory.
    pub(crate) id: Option<String>,
    pub(crate) loaded: Loaded,
    /// Opt-in continuous-batching decode worker (`FRINK_CONTINUOUS_BATCHING=1`).
    /// Shares `forward_multi_seq` across concurrent GGUF requests. Disabled
    /// when a KV pool or prefix cache is configured (those keep the
    /// private-loop `generate` path). Always `None` for an encoder:
    /// there is no decode step to batch.
    pub(crate) batcher: Option<serving::batch::ContinuousBatcher>,
    /// The per-request context ceiling this model was priced for, or
    /// `None` when it could not be priced (see `crate::budget`).
    ///
    /// Lives on the *model* rather than on `AppState` because it is a
    /// property of the checkpoint plus the machine: `/admin/models/load`
    /// swapping in a different model must swap in its ceiling too,
    /// never keep the old model's arithmetic. The same `Arc` is inside
    /// this model's `batcher`, so the batched and private decode paths
    /// admit on one object.
    pub(crate) ceiling: Option<Arc<budget::ContextCeiling>>,
    /// The checkpoint file this model was loaded from, when it came
    /// from one.
    ///
    /// Held for [`crate::slots`], which fingerprints the checkpoint so
    /// a saved KV slot cannot be restored onto different weights.
    /// `FRINK_MODEL_PATH` would be the wrong source for that:
    /// `/admin/models/load` swaps the model without touching it, so a
    /// slot saved after a swap would be stamped with the identity of
    /// the checkpoint the process *started* on. Like `ceiling`, this is
    /// a property of the model and travels with it.
    ///
    /// `None` for the synthetic fallback and for a Kimi directory.
    pub(crate) checkpoint_path: Option<PathBuf>,
}

/// `FRINK_MODEL_NAME`, cached because `name()` returns a `&str` and is
/// called per request.
///
/// Read once: the server sets it before any worker thread starts, and a
/// name that changed under a running server would make two responses
/// disagree about which model answered them.
fn served_name_override() -> Option<&'static str> {
    static NAME: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        std::env::var("FRINK_MODEL_NAME")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    })
    .as_deref()
}

impl ActiveModel {
    /// The model to decode with, or a refusal naming the encoder that
    /// is loaded instead.
    ///
    /// **This is the only way to a `Model` from an `ActiveModel`**, and
    /// that is the point: an encoder cannot reach a decode by any other
    /// route, and a caller cannot forget to check, because there is
    /// nothing to forget — the `?` is the check.
    pub(crate) fn generative(&self) -> Result<&Arc<Model>, ApiError> {
        match &self.loaded {
            Loaded::Generative(m) => Ok(m),
            Loaded::Encoder(e) => Err(not_a_generative_model(e)),
        }
    }

    /// The model facts the sampler chain needs that a request body
    /// cannot carry: the vocabulary DRY's sequence breakers are
    /// tokenised against, and the context size `dry_penalty_last_n = -1`
    /// resolves to.
    ///
    /// Taken off the PINNED `ActiveModel` rather than looked up again,
    /// so a request cannot resolve its sampler against one checkpoint
    /// and decode against the one `/admin/models/load` swapped in
    /// afterwards.
    pub(crate) fn sampler_model(&self) -> crate::sampling_knobs::SamplerModel<'_> {
        let vocab = self
            .generative_opt()
            .filter(|m| m.has_real_vocabulary())
            .map(|m| m.as_ref() as &dyn frink_models::dry::DryVocab);
        crate::sampling_knobs::SamplerModel {
            vocab,
            // `usize::MAX` when this server could not price a ceiling:
            // see `SamplerModel::context_size` for why that is the
            // derived answer rather than a chosen constant.
            context_size: self
                .ceiling
                .as_ref()
                .and_then(|c| c.limit())
                .unwrap_or(usize::MAX),
        }
    }

    /// The generation model when there is one, for the reporting
    /// surfaces (`/v1/models`, `/health`) that describe whatever is
    /// loaded rather than requiring a particular kind. They say less
    /// about an encoder; they must not say something false.
    pub(crate) fn generative_opt(&self) -> Option<&Arc<Model>> {
        match &self.loaded {
            Loaded::Generative(m) => Some(m),
            Loaded::Encoder(_) => None,
        }
    }

    /// The encoder when the loaded checkpoint is one.
    pub(crate) fn encoder(&self) -> Option<&Arc<EmbeddingModel>> {
        match &self.loaded {
            Loaded::Generative(_) => None,
            Loaded::Encoder(e) => Some(e),
        }
    }

    /// Token ids for `text`, whichever kind of checkpoint is loaded.
    ///
    /// `/v1/tokenize` and `/v1/detokenize` went through
    /// `require_model()`, which is `generative()?`, so on an
    /// encoder-only server they answered 501 "not a generative model".
    /// That is the right refusal for a decode and the wrong one for a
    /// question about tokenization: an encoder has a real tokenizer,
    /// and asking what tokens it saw is exactly what a surprising
    /// embedding makes you want to do (issue #28).
    ///
    /// Deliberately NOT reached through `generative()`: routes that
    /// need a decode still go through it and still refuse, and this
    /// pair is the only thing that steps around it.
    ///
    /// `specials` reaches the generative tokenizer as is. An encoder's
    /// `token_ids` is the embedding input, which llama.cpp tokenizes
    /// with `parse_special = true` and which wraps its own `[CLS]` /
    /// `[SEP]`; there is no separate setting to honour on that side.
    pub(crate) fn encode_any(&self, text: &str, specials: SpecialTokens) -> Vec<usize> {
        match &self.loaded {
            Loaded::Generative(m) => m.encode(text, specials),
            Loaded::Encoder(e) => e.token_ids(text).into_iter().map(|t| t as usize).collect(),
        }
    }

    /// Text for `ids`, whichever kind of checkpoint is loaded. The
    /// counterpart to [`Self::encode_any`].
    pub(crate) fn decode_any(&self, ids: &[usize]) -> String {
        match &self.loaded {
            Loaded::Generative(m) => m.decode(ids),
            Loaded::Encoder(e) => {
                let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
                e.decode_tokens(&ids)
            }
        }
    }

    /// What `/v1/models` and `/health` call this checkpoint. Both kinds
    /// have a real name.
    ///
    /// `FRINK_MODEL_NAME` (llama.cpp's `--alias`) wins when set. It is
    /// read HERE rather than at load because three loader paths derive
    /// a name from `general.name` independently, and threading an
    /// override through all three would be three places that have to
    /// agree about one string. This is the single place the served name
    /// is decided, so the override cannot apply to some models and not
    /// others.
    pub(crate) fn name(&self) -> &str {
        if let Some(alias) = served_name_override() {
            return alias;
        }
        match &self.loaded {
            Loaded::Generative(m) => m.name(),
            Loaded::Encoder(e) => e.name(),
        }
    }

    /// The reasoning parser for this checkpoint under its served name:
    /// `PromptTemplate::reasoning_format`, the one place that decides
    /// it. An encoder has no template and no chain of thought.
    pub(crate) fn reasoning_format(&self) -> Option<crate::policy::parser::ReasoningFormat> {
        self.generative_opt()
            .and_then(|m| m.chat_template().reasoning_format(self.name()))
    }

    /// The tool-call grammar for this checkpoint under its served
    /// name: `PromptTemplate::tool_call_format`, the one place that
    /// decides it. An encoder emits no calls, so it takes the name's
    /// family and never reads one.
    pub(crate) fn tool_call_format(&self) -> crate::policy::parser::ToolCallFormat {
        match self.generative_opt() {
            Some(m) => m.chat_template().tool_call_format(self.name()),
            None => crate::policy::parser::ToolCallFormat::infer(self.name()),
        }
    }

    /// The tokenizer this checkpoint carries. An encoder's is real and
    /// is reported as such -- `EmbeddingModel` only accepts
    /// `tokenizer.ggml.model = "bert"`, so WordPiece is a fact about
    /// it, not a default.
    pub(crate) fn tokenizer_kind(&self) -> &'static str {
        match &self.loaded {
            Loaded::Generative(m) => m.tokenizer_kind(),
            Loaded::Encoder(_) => "gguf-wordpiece",
        }
    }

    /// Only the GGUF decoder path has a synthetic-weights fallback; an
    /// encoder is always a real checkpoint, because there is no
    /// random-weight encoder preset to fall back to.
    pub(crate) fn is_synthetic(&self) -> bool {
        self.generative_opt().is_some_and(|m| m.is_synthetic())
    }

    /// Live counters of the bounded expert cache, when the model
    /// streams routed experts. An encoder has no experts.
    pub(crate) fn expert_store_stats(&self) -> Option<frink_core::expert_store::ExpertStoreStats> {
        self.generative_opt().and_then(|m| m.expert_store_stats())
    }
}

/// The refusal a generation route gets when the loaded checkpoint is an
/// encoder.
///
/// 501 and not 503: 503 means "try again", and no amount of retrying
/// turns an encoder into a decoder. 501 with the model named is the
/// shape llama.cpp uses for the mirror-image case (`/v1/embeddings` on
/// a server started without `--embeddings`), and it says the thing that
/// is actually true -- this endpoint is not implemented *for this
/// model* -- rather than blaming a tensor.
fn not_a_generative_model(encoder: &EmbeddingModel) -> ApiError {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({"error": {
            "message": encoder_refusal(
                encoder.name(),
                encoder.architecture(),
                encoder.n_embd(),
                encoder.pooling_type().name(),
            ),
            "type": "unsupported",
            "param": "model",
        }})),
    )
}

/// The wording, split from [`not_a_generative_model`] only so a test
/// can check the facts survive a rewording without needing a 37 MB
/// checkpoint on disk to build an [`EmbeddingModel`] from.
fn encoder_refusal(name: &str, arch: &str, n_embd: usize, pooling: &str) -> String {
    format!(
        "the loaded model '{name}' is an embedding model ({arch} encoder, {n_embd} dims, \
         pooling {pooling}). An encoder has no output head, so it cannot generate text at \
         all. There is no next token for it to predict. POST /v1/embeddings to use it, or \
         load a generative checkpoint."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal must name the model that IS loaded and the route
    /// that WOULD serve it. "unsupported" alone sends an operator
    /// hunting a missing tensor that is not missing -- which is the
    /// exact failure this whole seam exists to prevent.
    #[test]
    fn the_encoder_refusal_names_the_model_and_the_route_that_works() {
        let msg = encoder_refusal("bge-small-en-v1.5", "bert", 384, "CLS");
        for fact in ["bge-small-en-v1.5", "bert", "384", "CLS", "/v1/embeddings"] {
            assert!(msg.contains(fact), "{msg} does not carry {fact}");
        }
        assert!(msg.contains("embedding model"), "{msg}");
    }
}
