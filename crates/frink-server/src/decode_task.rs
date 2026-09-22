//! The six handles a decode runs against, taken once and carried
//! together.
//!
//! Every generating route has to do the same thing before it can call
//! [`crate::run_generation`]: pin the model handle so a mid-flight
//! `/admin/models/load` cannot splice two checkpoints into one answer,
//! then clone the KV pool, the paged pool, the prefix cache, the
//! continuous batcher and the context ceiling off `AppState` and the
//! active model, then hand all six to an eight-argument function in the
//! right order.
//!
//! That block was written out four times -- `/v1/chat/completions`,
//! `/v1/completions`, `/v1/messages` and `/v1/responses` -- and the
//! *argument order* is the part that made it dangerous: `kv_pool` and
//! `paged_kv` are both `Option<&_>`, so transposing them is not a type
//! error. This repo's recorded lesson is that a copied path diverges
//! and nothing notices, so the block is one struct here and the routes
//! name their protocol instead.
//!
//! Two methods, matching the two shapes a route needs:
//!
//! - [`DecodeHandles::run`], the buffered one, wrapped by
//!   [`buffered`] which also does the `spawn_blocking` handoff and the
//!   two error mappings.
//! - [`DecodeHandles::run_emit`], which streams chunks to a sink. The
//!   sink is where the protocols genuinely differ (reasoning splits,
//!   tool-call parsing, plain text), so that part stays with each
//!   route; the handles do not.

use std::sync::{Arc, Mutex};

use frink_models::PrefixCache;

use crate::generate::GenerationParams;
use crate::{
    budget, decode_error_response, generate, join_error_response, serving, ActiveModel, ApiError,
    AppState, Model,
};

/// Everything a decode needs from the server, pinned to one model.
pub(crate) struct DecodeHandles {
    model: Arc<Model>,
    kv_pool: Option<generate::KvPoolConfig>,
    paged_kv: Option<generate::PagedKvConfig>,
    prefix_cache: Option<Arc<Mutex<PrefixCache>>>,
    batcher: Option<serving::batch::ContinuousBatcher>,
    ceiling: Option<Arc<budget::ContextCeiling>>,
    metal_private_decode_gate: Option<Arc<std::sync::Mutex<()>>>,
}

impl DecodeHandles {
    /// Take the handles for one request.
    ///
    /// `active` is the caller's own `require_active()` result rather
    /// than a fresh read of `AppState`, so the model, batcher and
    /// ceiling all come from the same swap generation -- a ceiling
    /// derived for a checkpoint that is no longer loaded prices the
    /// wrong KV geometry.
    ///
    /// Fails when `active` is an encoder-only checkpoint: there is no
    /// decode to take handles for. Every generating route passes
    /// through here, so this one `?` is what keeps an encoder off all
    /// five of them.
    pub(crate) fn take(state: &AppState, active: &ActiveModel) -> Result<Self, ApiError> {
        Ok(DecodeHandles {
            model: Arc::clone(active.generative()?),
            kv_pool: state.kv_pool.clone(),
            paged_kv: state.paged_kv.clone(),
            prefix_cache: state.prefix_cache.clone(),
            batcher: active.batcher.clone(),
            ceiling: active.ceiling.clone(),
            metal_private_decode_gate: state.metal_private_decode_gate.clone(),
        })
    }

    /// The model this decode is pinned to. Read for its name or its
    /// tokenizer; the handle itself never leaves.
    pub(crate) fn model(&self) -> &Model {
        &self.model
    }

    /// Whether a prefix cache is configured, which is the only
    /// circumstance under which prompt KV is reused across requests.
    ///
    /// Read by `/completion`, whose `cache_prompt: false` is a
    /// *requirement* not to reuse rather than a permission to.
    pub(crate) fn has_prefix_cache(&self) -> bool {
        self.prefix_cache.is_some()
    }

    /// Generate, collecting the whole answer. Blocking: call it from
    /// `spawn_blocking` (or through [`buffered`], which does that).
    pub(crate) fn run(
        &self,
        prompt: &str,
        params: &GenerationParams,
        // Per choice, choice 0 first. See `crate::run_generation_emit`.
    ) -> Result<generate::Generated, generate::DecodeError> {
        crate::run_generation(
            &self.model,
            prompt,
            params,
            self.kv_pool.as_ref(),
            self.paged_kv.as_ref(),
            self.prefix_cache.as_deref(),
            self.batcher.as_ref(),
            self.ceiling.as_deref(),
            self.metal_private_decode_gate.as_deref(),
        )
    }

    /// Generate, handing every decoded chunk to `emit` as it arrives.
    /// Blocking, for the same reason.
    pub(crate) fn run_emit(
        &self,
        prompt: &str,
        params: &GenerationParams,
        emit: impl FnMut(&str),
        // Per choice, choice 0 first. See `crate::run_generation_emit`.
    ) -> Result<generate::Generated, generate::DecodeError> {
        crate::run_generation_emit(
            &self.model,
            prompt,
            params,
            self.kv_pool.as_ref(),
            self.paged_kv.as_ref(),
            self.prefix_cache.as_deref(),
            self.batcher.as_ref(),
            self.ceiling.as_deref(),
            self.metal_private_decode_gate.as_deref(),
            emit,
        )
    }
}

/// One buffered generation, off the request thread.
///
/// Generation is CPU-bound and would otherwise block a Tokio worker for
/// the length of a completion, so it runs on `spawn_blocking`; a panic
/// in it becomes a 500 through `join_error_response` rather than a
/// hung request, and a decode error becomes the status
/// `decode_error_response` names.
pub(crate) async fn buffered(
    handles: DecodeHandles,
    prompt: String,
    params: GenerationParams,
) -> Result<generate::Generated, ApiError> {
    tokio::task::spawn_blocking(move || handles.run(&prompt, &params))
        .await
        .map_err(join_error_response)?
        .map_err(decode_error_response)
}
