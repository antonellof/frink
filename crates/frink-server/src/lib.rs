//! frink-server: OpenAI-compatible HTTP surface (`/health`,
//! `/v1/models`, `/v1/chat/completions`, `/v1/completions`,
//! `/v1/tokenize`, `/v1/detokenize`, `/v1/embeddings`) over the
//! frink-models decoder, plus a whole-response cache for exact-repeat
//! requests (see `cache` module). Loads a real GGUF checkpoint and its
//! own real tokenizer when `-m`/`--model` or `FRINK_MODEL_PATH` is set
//! (see `model` module). Supports sampling
//! (temperature/top_p/top_k/repetition_penalty), stop sequences, and SSE
//! streaming (see `generate` module).
//!
//! Concurrency: the loaded model
//! (`Model`) is immutable once loaded and shared via `Arc`, not locked
//! behind a `Mutex` -- there is no shared mutable decoder state for
//! concurrent requests to contend on or for one panicking request to
//! poison. The *pointer* to it is swappable (`AppState::active`, behind
//! an `RwLock` held only long enough to clone one `Arc`), which is what
//! `/admin/models/load` swaps; a request that has already cloned its
//! handle finishes against the exact weights it started on, and the old
//! model is freed when the last such request lets go.
//! Each request builds its own KV cache (see `generate::generate`)
//! and runs its decode loop on tokio's blocking-thread pool via
//! `spawn_blocking`, so CPU-bound generation no longer blocks the async
//! reactor threads -- multiple requests can decode genuinely
//! concurrently, bounded by that pool rather than serialized through one
//! lock. Only the small whole-response cache is still mutable shared
//! state, and it's locked only for the brief get/put around it, never
//! across a decode.
//!
//! Streaming scope: when `stream: true` and tools are inactive, each
//! decoded chunk is pushed through a bounded `mpsc` channel from the
//! blocking generate task into the SSE writer so time-to-first-byte
//! overlaps with ongoing decode. Under continuous batching the batch
//! worker emits the same incremental chunks as the private decode loop.

mod admin;
mod anthropic;
mod attribution;
mod budget;
mod cache_admin;
mod cancel;
mod chat_params;
mod chat_template;
mod cli;
mod completion;
mod continuation;
mod conversations;
mod decode_task;
mod embeddings;
mod generate;
mod grammar_request;
mod health;
mod journal;
mod json_mode;
mod limits;
mod loaded;
mod lora;
mod mcp;
mod model;
mod openai_extra;
mod output;
mod policy;
mod prefill_batch;
mod reasoning_budget;
mod reasoning_tokens;
mod rerank;
mod response_cache;
pub(crate) mod responses;
mod resume;
mod sample_step;
mod sampling_knobs;
mod sampling_loop;
mod security;
mod serving;
mod session;
mod slots;
mod sse;
mod stats;
mod stop;
mod stream_events;
mod tasks;
mod tool_grammar;
mod unsupported_sampling;
mod utf8_stream;

use std::cell::RefCell;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use cli::apply_cli_overrides;
pub use cli::{ServerArgs, BUILT_WITH_CUDA, BUILT_WITH_METAL};

use frink_core::cache::KvBlockPool;
use frink_models::kimi_tokenizer::KimiTokenizer;
use frink_models::sampling::SamplingParams;
use frink_models::tokenizer::{SpecialTokens, StopTokens};
use frink_models::{Decoder, Gemma4Engine, KimiEngine, MlaEngine, PrefixCache};
use generate::{FinishReason, GenerationParams};
pub(crate) use loaded::{ActiveModel, Loaded};
use model::ServerTokenizer;
use rerank::encoder_endpoints;
use response_cache::ResponseCache;
use sampling_knobs::SamplingKnobs;

/// The loaded model: immutable once built, so it needs no lock at all --
/// just cheap `Arc` sharing across concurrent request tasks. Two real
/// checkpoint shapes exist (see `model::LoadedModel`'s doc comment for
/// why `FRINK_MODEL_PATH` picks between them); everything that isn't
/// engine-specific (chat template, tokenizer kind reporting, whether
/// this is the synthetic demo) goes through the small inherent methods
/// below rather than being matched on ad hoc at every call site.
#[allow(clippy::large_enum_variant)] // KimiEngine/MlaEngine dwarf Arc<Decoder>; boxing would churn call sites
pub(crate) enum Model {
    Gguf(GgufModel),
    Kimi(KimiModel),
    Mla(MlaModel),
    Gemma4(Gemma4Model),
    Glm52(Glm52Model),
}

pub(crate) struct GgufModel {
    decoder: Arc<Decoder>,
    tokenizer: Arc<ServerTokenizer>,
    stop_tokens: StopTokens,
    bos_id: Option<usize>,
    is_synthetic: bool,
    chat_template: chat_template::PromptTemplate,
}

pub(crate) struct KimiModel {
    engine: KimiEngine,
    tokenizer: KimiTokenizer,
    stop_tokens: StopTokens,
    chat_template: chat_template::PromptTemplate,
}

pub(crate) struct MlaModel {
    engine: MlaEngine,
    tokenizer: ServerTokenizer,
    stop_tokens: StopTokens,
    bos_id: Option<usize>,
    name: String,
    chat_template: chat_template::PromptTemplate,
}

pub(crate) struct Gemma4Model {
    engine: Gemma4Engine,
    tokenizer: ServerTokenizer,
    stop_tokens: StopTokens,
    bos_id: Option<usize>,
    name: String,
    chat_template: chat_template::PromptTemplate,
}

pub(crate) struct Glm52Model {
    engine: frink_models::Glm52Engine,
    tokenizer: ServerTokenizer,
    stop_tokens: StopTokens,
    bos_id: Option<usize>,
    name: String,
    chat_template: chat_template::PromptTemplate,
}

impl Model {
    pub(crate) fn chat_template(&self) -> chat_template::PromptTemplate {
        match self {
            Model::Gguf(m) => m.chat_template.clone(),
            Model::Kimi(m) => m.chat_template.clone(),
            Model::Mla(m) => m.chat_template.clone(),
            Model::Gemma4(m) => m.chat_template.clone(),
            Model::Glm52(m) => m.chat_template.clone(),
        }
    }

    /// Kimi K3 / MLA / GLM-5.2 have no synthetic-weight demo path through this
    /// server (unlike GGUF, which falls back to one when
    /// `FRINK_MODEL_PATH` is unset) -- a loaded `Model::Kimi` /
    /// `Model::Mla` / `Model::Glm52` is always a real checkpoint.
    fn is_synthetic(&self) -> bool {
        match self {
            Model::Gguf(m) => m.is_synthetic,
            Model::Kimi(_) | Model::Mla(_) | Model::Gemma4(_) | Model::Glm52(_) => false,
        }
    }

    fn tokenizer_kind(&self) -> &'static str {
        match self {
            Model::Gguf(m) => m.tokenizer.kind(),
            Model::Kimi(_) => "kimi-tiktoken-bpe",
            Model::Mla(m) => m.tokenizer.kind(),
            Model::Gemma4(m) => m.tokenizer.kind(),
            Model::Glm52(m) => m.tokenizer.kind(),
        }
    }

    /// Live counters of the bounded expert cache, when the model
    /// streams routed experts (`FRINK_EXPERT_CACHE_BYTES`); `None`
    /// for fully resident models.
    fn expert_store_stats(&self) -> Option<frink_core::expert_store::ExpertStoreStats> {
        match self {
            Model::Gguf(m) => m.decoder.expert_store_stats(),
            Model::Kimi(m) => m.engine.weights.expert_store_stats(),
            Model::Mla(_) | Model::Gemma4(_) | Model::Glm52(_) => None,
        }
    }

    pub(crate) fn name(&self) -> &str {
        match self {
            Model::Gguf(m) => m.decoder.config.name,
            Model::Kimi(_) => "kimi-k3",
            Model::Mla(m) => m.name.as_str(),
            Model::Gemma4(m) => m.name.as_str(),
            Model::Glm52(m) => m.name.as_str(),
        }
    }

    /// `specials` is llama.cpp's `parse_special`, and each caller is
    /// matched to the llama.cpp server site it mirrors
    /// (`tools/server/server-context.cpp` unless said otherwise):
    ///
    /// * a prompt, rendered from a chat template or given raw --
    ///   `/v1/chat/completions`, `/v1/completions`, `/v1/messages`,
    ///   `count_tokens`, slot save: `Parse`, as
    ///   `tokenize_input_prompts(..., true, true)` does for both
    ///   completion routes. llama.cpp's server does NOT tokenize a
    ///   message's content separately from the template around it, so
    ///   neither does this one; a document that mentions `<|im_end|>`
    ///   inside a chat message is parsed on both engines. Doing better
    ///   would need the template renderer to hand back which spans are
    ///   content, and is deliberately not done here so the two engines
    ///   agree about the prompt.
    /// * pooled decoder embeddings: `Parse` (`handle_embeddings_impl`).
    /// * `/v1/tokenize`: the request's own `parse_special`, default
    ///   `true` (`json_value(body, "parse_special", true)`).
    /// * DRY sequence breakers: `AsText`
    ///   (`llama-sampler.cpp`: `vocab.tokenize(str, false, false)`).
    /// * a stop string that is one token: `Parse`. This is frink's own
    ///   mechanism (llama.cpp matches stop strings on decoded text and
    ///   tokenizes them only to trim `n_probs`), and a caller who names
    ///   `<|eot_id|>` as a stop means the token.
    /// * a tool-call opener that anchors the paged KV window: `Parse`,
    ///   because the opener is a special token where the family has one.
    pub(crate) fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<usize> {
        match self {
            Model::Gguf(m) => m.tokenizer.encode(text, specials),
            Model::Kimi(m) => m
                .tokenizer
                .encode(text, specials)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
            Model::Mla(m) => m.tokenizer.encode(text, specials),
            Model::Gemma4(m) => m.tokenizer.encode(text, specials),
            Model::Glm52(m) => m.tokenizer.encode(text, specials),
        }
    }

    /// The BOS id the generation path would prepend, or `None` when
    /// this checkpoint's own metadata says not to prepend one.
    ///
    /// Read by `/tokenize`'s `add_special`, so that endpoint reports
    /// the prompt the model would actually be given rather than a
    /// second opinion about it. Kimi has no BOS id plumbed through the
    /// server -- `run_generation` passes `None` for it -- and this
    /// agrees with that rather than inventing one.
    pub(crate) fn bos_id(&self) -> Option<usize> {
        match self {
            Model::Gguf(m) => m.bos_id,
            Model::Kimi(_) => None,
            Model::Mla(m) => m.bos_id,
            Model::Gemma4(m) => m.bos_id,
            Model::Glm52(m) => m.bos_id,
        }
    }

    pub(crate) fn decode(&self, ids: &[usize]) -> String {
        match self {
            Model::Gguf(m) => m.tokenizer.decode(ids),
            Model::Kimi(m) => {
                let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
                m.tokenizer.decode(&ids32)
            }
            Model::Mla(m) => m.tokenizer.decode(ids),
            Model::Gemma4(m) => m.tokenizer.decode(ids),
            Model::Glm52(m) => m.tokenizer.decode(ids),
        }
    }

    /// Final-normed last-layer hidden states for GGUF Decoder only.
    /// Returns `None` for engines without a hidden-state hook (e.g. Kimi/MLA/GLM).
    pub(crate) fn embed_tokens(&self, tokens: &[usize]) -> Option<Vec<Vec<f32>>> {
        match self {
            Model::Gguf(m) => {
                let mut caches: Vec<_> = m.decoder.config.new_kv_caches();
                Some(m.decoder.forward_hidden_batch(tokens, 0, &mut caches))
            }
            Model::Kimi(_) | Model::Mla(_) | Model::Gemma4(_) | Model::Glm52(_) => None,
        }
    }

    /// The generic GGUF decoder, when that is what is loaded.
    ///
    /// `None` for the dedicated engines (Kimi, MLA, Gemma-4, GLM-5.2):
    /// they hold their own KV in their own shape, and
    /// [`crate::slots`]'s file format describes the generic one.
    pub(crate) fn gguf_decoder(&self) -> Option<&Arc<Decoder>> {
        match self {
            Model::Gguf(m) => Some(&m.decoder),
            Model::Kimi(_) | Model::Mla(_) | Model::Gemma4(_) | Model::Glm52(_) => None,
        }
    }

    pub(crate) fn vocab_size(&self) -> Option<usize> {
        match self {
            Model::Gguf(m) => Some(m.decoder.config.vocab_size),
            Model::Kimi(m) => Some(m.tokenizer.vocab_size()),
            Model::Mla(m) => Some(frink_models::Engine::vocab_size(&m.engine)),
            Model::Gemma4(m) => Some(frink_models::Engine::vocab_size(&m.engine)),
            Model::Glm52(m) => Some(frink_models::Engine::vocab_size(&m.engine)),
        }
    }

    /// True when this checkpoint carries a real vocabulary rather than
    /// the byte-level fallback the synthetic-weight demo model uses.
    ///
    /// Read by the DRY sampler, whose sequence breakers are strings that
    /// only mean something against a real tokenizer; see
    /// [`frink_models::dry::DryVocabMissing`].
    fn has_real_vocabulary(&self) -> bool {
        match self {
            Model::Gguf(m) => !matches!(*m.tokenizer, model::ServerTokenizer::Byte),
            Model::Kimi(_) => true,
            Model::Mla(m) => !matches!(m.tokenizer, model::ServerTokenizer::Byte),
            Model::Gemma4(m) => !matches!(m.tokenizer, model::ServerTokenizer::Byte),
            Model::Glm52(m) => !matches!(m.tokenizer, model::ServerTokenizer::Byte),
        }
    }
}

/// What the DRY sampler needs to tokenise its sequence breakers.
///
/// One trait, two implementations (`frink_cli`'s `CliTokenizer` has the
/// other), so `--dry-sequence-breaker` and the `dry_sequence_breakers`
/// request field cannot come to mean different things.
impl frink_models::dry::DryVocab for Model {
    fn n_tokens(&self) -> usize {
        self.vocab_size().unwrap_or(0)
    }

    fn detokenize(&self, token: usize) -> String {
        self.decode(&[token])
    }

    fn tokenize(&self, text: &str) -> Vec<usize> {
        self.encode(text, SpecialTokens::AsText)
    }
}

pub(crate) struct AppState {
    /// A **side-car** embedding model (`FRINK_EMBEDDING_MODEL_PATH`),
    /// served by `/v1/embeddings` in preference to pooling a decoder's
    /// hidden states.
    ///
    /// This is now the *second* way an encoder gets here. The first is
    /// [`AppState::active`]: an encoder-only checkpoint at
    /// `FRINK_MODEL_PATH` (or swapped in through
    /// `/admin/models/load`) is the loaded model, as
    /// [`crate::loaded::Loaded::Encoder`]. This field is what a
    /// deployment uses when it wants a generative model active *and*
    /// embeddings from a real encoder at the same time -- one process,
    /// two checkpoints, which the active-model slot alone cannot
    /// express. See [`AppState::embedding_model`] for which wins.
    pub(crate) embedding: Option<Arc<frink_models::EmbeddingModel>>,
    /// The swappable active model.
    ///
    /// **A reader clones the `Arc` under the read lock and then runs;
    /// the lock is never held across a decode.** That is the whole
    /// design: `RwLock` guards the *pointer*, not the model, so
    /// `/admin/models/load` swapping in a new `Arc` cannot stall a
    /// request that is already generating, and a request that started
    /// against the old model keeps decoding against the exact weights
    /// it began with until it finishes -- the old `ActiveModel` (and
    /// its batcher thread) is dropped only when the last in-flight
    /// holder releases it, not when the swap happens. Requests that
    /// arrive after the swap see the new model. There is deliberately
    /// no attempt to migrate an in-flight request: half a completion
    /// from one checkpoint and half from another is worse than either.
    ///
    /// `None` means nothing is loaded (after `/admin/models/unload`, or
    /// a failed startup load): generation endpoints answer 503 rather
    /// than pretending, and `/health` reports `unavailable`.
    active: std::sync::RwLock<Option<Arc<ActiveModel>>>,
    /// Set while a load task is in flight, so a second load request is
    /// rejected instead of racing the first. A load is not cheap and
    /// two concurrent ones would fight for the same memory.
    pub(crate) load_in_progress: std::sync::atomic::AtomicBool,
    /// Long-running jobs (download, load) -- see the `tasks` module.
    pub(crate) tasks: Arc<tasks::TaskRegistry>,
    /// Generations that can currently be stopped by `POST /v1/cancel`
    /// -- see the `cancel` module for why a dropped socket alone is not
    /// enough.
    pub(crate) cancels: Arc<cancel::CancelRegistry>,
    /// Recent-request ring buffer and the counters behind
    /// `/admin/stats` -- see the `stats` module.
    pub(crate) stats: stats::Stats,
    /// Replay buffers for streams started with `stream_resumable`.
    /// See the `resume` module.
    pub(crate) streams: resume::StreamRegistry,
    /// The directory `/admin/models` scans, when one is configured.
    pub(crate) model_dir: Option<PathBuf>,
    /// The only shared *mutable* state in the server. Locked only for
    /// the brief get/put around a cache lookup, never held across a
    /// decode -- see the module doc comment.
    response_cache: Mutex<ResponseCache>,
    /// `Some` when `FRINK_KV_POOL_BLOCKS`/`FRINK_KV_POOL_BLOCK_SIZE`
    /// are set: every request's per-layer KV caches then draw from
    /// this one shared, bounded pool instead of each growing
    /// unboundedly. A request whose caches can't get their first block
    /// retries for up to `FRINK_KV_POOL_QUEUE_TIMEOUT_MS` (zero by
    /// default -- reject immediately) before being rejected with 503,
    /// rather than being admitted regardless of how many other
    /// requests are already decoding -- see
    /// `frink_core::cache::KvBlockPool` and `generate::KvPoolConfig`.
    /// `None` (the default) preserves the
    /// original unbounded-per-request behavior exactly.
    pub(crate) kv_pool: Option<generate::KvPoolConfig>,
    /// `Some` when `FRINK_PAGED_KV_BLOCKS` is set: per-layer paged KV
    /// storage every request draws pages from, rather than each request
    /// owning a private contiguous buffer.
    ///
    /// Mutually exclusive with BOTH `kv_pool` and `prefix_cache`, and
    /// refused at startup rather than silently preferred. Against
    /// `kv_pool` because they are two answers to the same question.
    /// Against `prefix_cache` because `PrefixCache` stores
    /// `Vec<KvCache>` snapshots, which a paged request has none of, so
    /// enabling both would give a cache that can never hit -- see
    /// `wire-radix-prefix-cache` in the plan, which is what removes
    /// that restriction.
    pub(crate) paged_kv: Option<generate::PagedKvConfig>,
    /// `Some` when `FRINK_PREFIX_CACHE_ENTRIES` is set: a shared,
    /// LRU-bounded store of previously processed prompt+KV-state
    /// snapshots (see `frink_models::PrefixCache`), consulted so a
    /// request that *extends* an earlier one -- the common multi-turn-
    /// chat case -- can skip recomputing the shared part. Mutually
    /// exclusive with `kv_pool` (see `generate::generate`'s doc
    /// comment for why); `None` (the default) means every request
    /// processes its full prompt from scratch, exactly as before this
    /// existed.
    pub(crate) prefix_cache: Option<Arc<Mutex<PrefixCache>>>,
    /// Server-side per-session conversation history -- see
    /// `session::SessionStore`'s doc comment.
    /// Always present (unlike `kv_pool`/`prefix_cache`, it's not
    /// opt-in): a request that never sends `session_id` simply never
    /// touches it, at negligible cost (one empty `HashMap`).
    sessions: session::SessionStore,
    requests_total: std::sync::atomic::AtomicU64,
    request_errors_total: std::sync::atomic::AtomicU64,
    started_at: std::time::Instant,
    /// Milliseconds after `started_at` at which the last request
    /// finished; 0 means none has. Reported by `/health` as an age, so a
    /// client that sees a slow health poll from a GPU-saturated server
    /// has positive evidence of liveness instead of declaring it dead.
    last_request_ms: std::sync::atomic::AtomicU64,
    /// Backend capability probe behind `/health` (see `health` module).
    detection: Arc<health::Detection>,
    /// Loaded MCP config (`--mcp-config`); tool invocation not wired yet.
    mcp: Option<mcp::LoadedMcpConfig>,
    /// Whether a swapped-in GGUF model should get a continuous-batching
    /// worker, decided once at startup from the same env var and
    /// exclusions as the initial load.
    pub(crate) continuous_batching_enabled: bool,
    /// Serializes private-loop Metal decodes when continuous batching is
    /// off. Shared `metal_attn_kv` is not safe across concurrent
    /// `forward_token` calls yet; see `docs/plans/metal-parallel-concurrency.md`.
    pub(crate) metal_private_decode_gate: Option<Arc<std::sync::Mutex<()>>>,
    /// The model id a load task is currently working on, so
    /// `/admin/models` can report `loading` for it. Separate from
    /// `load_in_progress` because that is a gate and this is a label.
    loading_model: Mutex<Option<String>>,
    /// The last failed load, as `(model id, message)`. Sticky until the
    /// next successful load so `/admin/models` can say *why* an entry
    /// is in `error` without the user retrying to find out.
    last_load_error: Mutex<Option<(String, String)>>,
    /// Live serving counters and the two sliding-window rates behind
    /// `/v1/stats` -- see `crate::stats::ServingStats`. Distinct from
    /// `stats`, which is the historical ring: this is what is happening
    /// *now*, and it decays to zero when nothing is.
    pub(crate) serving: Mutex<crate::stats::ServingStats>,
    /// The gate every request, cache rebuild and shutdown passes
    /// through -- see `crate::policy::maintenance::MaintenanceGate`. Held across none
    /// of them: each operation takes it, reads or moves the state, and
    /// releases before doing any work.
    pub(crate) maintenance: Mutex<crate::policy::maintenance::MaintenanceGate>,
    /// The live memory reading behind `/v1/stats`, re-probed at most
    /// once per [`FOOTPRINT_TTL_MS`] -- see
    /// `cache_admin::footprint_json`. A `Mutex` and not an atomic
    /// because holding it across the probe is what collapses concurrent
    /// pollers onto ONE VMA walk.
    pub(crate) footprint:
        Mutex<crate::policy::footprint::ProbeCache<crate::policy::footprint::Footprint>>,
    /// Wall-clock second this process started serving.
    ///
    /// Distinct from `started_at`, which is an `Instant` and has no
    /// wall clock at all. This exists so an accounting receipt's id can
    /// be derived from something stable for the life of THIS process
    /// and different in the next one: a pid alone is reused across
    /// restarts, and a restarted engine reusing a previous
    /// generation's receipt id would have its own receipt silently
    /// skipped as already written.
    pub(crate) started_unix: u64,
}

/// How long a memory reading is served before it is taken again.
///
/// Two seconds: long enough that a dashboard polling once a second
/// costs one probe rather than one per poll, short enough that an
/// operator watching a load ramp sees it move.
pub(crate) const FOOTPRINT_TTL_MS: u64 = 2_000;

impl AppState {
    /// Clones the active model's `Arc` and releases the lock before
    /// returning. Every caller then runs against its own handle, so no
    /// decode ever holds this lock -- see [`AppState::active`].
    pub(crate) fn active(&self) -> Option<Arc<ActiveModel>> {
        self.active
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// [`AppState::active`] for a request that cannot proceed without a
    /// model. 503 with a `Retry-After`-shaped explanation is the honest
    /// answer while nothing is loaded; the alternative -- keeping a
    /// stale model around so the endpoint never fails -- would serve
    /// tokens from a checkpoint the operator explicitly unloaded.
    pub(crate) fn require_active(&self) -> Result<Arc<ActiveModel>, ApiError> {
        self.active().ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": {
                    "message": "no model is loaded; POST /admin/models/load with an id from \
                                GET /admin/models",
                    "type": "model_not_loaded"
                }})),
            )
        })
    }

    /// [`AppState::active`]'s *generation* model only, for the many
    /// call sites that do not care about the batcher.
    ///
    /// Two refusals live behind this one `?`: nothing loaded (503, from
    /// [`AppState::require_active`]) and an encoder loaded (501, from
    /// [`ActiveModel::generative`]). They are different answers to
    /// different questions and neither may be given for the other.
    pub(crate) fn require_model(&self) -> Result<Arc<Model>, ApiError> {
        Ok(Arc::clone(self.require_active()?.generative()?))
    }

    /// Publishes a new active model (or `None` to unload) and returns
    /// the previous one.
    ///
    /// The write lock is held only for the pointer swap. The returned
    /// value is the caller's to drop *outside* the lock: dropping a
    /// multi-gigabyte model can take a moment, and doing it under the
    /// lock would block every reader for exactly as long.
    pub(crate) fn swap_active(&self, next: Option<Arc<ActiveModel>>) -> Option<Arc<ActiveModel>> {
        let mut guard = self.active.write().unwrap_or_else(|p| p.into_inner());
        std::mem::replace(&mut *guard, next)
    }

    /// Stamps "a request just finished" for `/health`'s liveness
    /// vouching. Relaxed: this is a freshness hint, not a
    /// synchronization point.
    fn mark_request_finished(&self) {
        let ms = self.started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
        self.last_request_ms
            .store(ms, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub(crate) fn requests_total(&self) -> u64 {
        self.requests_total
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn errors_total(&self) -> u64 {
        self.request_errors_total
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn cache_stats(&self) -> response_cache::CacheStats {
        lock_cache(&self.response_cache).stats()
    }

    /// Seconds since the last request finished, or `None` when none
    /// has. Same derivation `/health` uses, so the two agree.
    pub(crate) fn last_request_age_seconds(&self) -> Option<f64> {
        let last = self
            .last_request_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        (last > 0)
            .then(|| self.uptime().as_secs_f64() - (last as f64 / 1000.0))
            .map(|age| age.max(0.0))
    }

    pub(crate) fn loading_model_id(&self) -> Option<String> {
        self.loading_model
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub(crate) fn set_loading_model(&self, id: Option<String>) {
        *self.loading_model.lock().unwrap_or_else(|p| p.into_inner()) = id;
    }

    pub(crate) fn last_load_error(&self) -> Option<(String, String)> {
        self.last_load_error
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub(crate) fn set_last_load_error(&self, error: Option<(String, String)>) {
        *self
            .last_load_error
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = error;
    }

    /// Records one finished request in the `/admin/stats` ring buffer.
    ///
    /// `attribution` is threaded from the request's own headers rather
    /// than looked up here: by the time a generation task finishes, the
    /// request parts are long gone, and reconstructing "who was that"
    /// afterwards is exactly the guessing the monitor exists to avoid.
    /// The model that would serve a request right now, as `/v1/models`
    /// names it. `None` when nothing is loaded.
    pub(crate) fn active_model_name(&self) -> Option<String> {
        self.active().map(|a| a.name().to_string())
    }

    /// The encoder `/v1/embeddings` should use, from either of the two
    /// ways one gets here.
    ///
    /// `FRINK_EMBEDDING_MODEL_PATH` wins over an encoder loaded as the
    /// active model, and it has to: a deployment that names both has
    /// asked for the side-car explicitly, while the active model may
    /// have been swapped in by `/admin/models/load` since. Only one of
    /// the two is ever set in practice -- the side-car exists so a
    /// *generative* model can be active at the same time.
    pub(crate) fn embedding_model(&self) -> Option<Arc<frink_models::EmbeddingModel>> {
        self.embedding
            .clone()
            .or_else(|| self.active().and_then(|a| a.encoder().map(Arc::clone)))
    }

    /// What `/v1/embeddings` is actually charging against, for the
    /// `/admin/stats` ring: the embedding model when one is serving,
    /// otherwise whichever decoder is active.
    pub(crate) fn embedding_model_name(&self) -> Option<String> {
        match self.embedding_model() {
            Some(e) => Some(e.name().to_string()),
            None => self.active_model_name(),
        }
    }

    pub(crate) fn record_request(&self, record: stats::Record<'_>) {
        self.stats.record(stats::entry(record));
    }
}

/// Defense in depth: if a panic ever happened while this lock was held
/// (none of the CPU-bound decode work runs under it, so this should be
/// very unlikely), recovering the inner state on poison rather than
/// `.unwrap()`ing keeps the cache from permanently bricking the server.
fn lock_cache(cache: &Mutex<ResponseCache>) -> MutexGuard<'_, ResponseCache> {
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image_url: Option<serde_json::Value>,
}

impl MessageContent {
    fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.text.as_deref())
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    fn has_image(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts
                .iter()
                .any(|p| p.kind == "image_url" || p.image_url.is_some()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    /// `None` for an assistant message that made tool calls instead of
    /// replying with text (the real OpenAI convention: `content` and
    /// `tool_calls` are mutually exclusive on an assistant message).
    #[serde(default)]
    pub(crate) content: Option<MessageContent>,
    /// Present on a replayed assistant message that previously made
    /// one or more tool calls (conversation history a client sends
    /// back on a follow-up request).
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<ToolCallIn>>,
    /// Present on a `"tool"`-role message carrying a call's result
    /// (unused by rendering today -- `role` alone already
    /// distinguishes it -- but accepted so real OpenAI-shaped tool-
    /// result messages deserialize without error).
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) tool_call_id: Option<String>,
    /// A replayed assistant turn's chain of thought, kept out of
    /// `content` on the way in and handed back to the template on the
    /// way out.
    ///
    /// It has to be a field of its own rather than prose folded into
    /// `content`, because a template that knows about reasoning wraps
    /// it in the family's own markers -- and a template that does not
    /// must be able to drop it. Concatenating it into `content` would
    /// show a model its own scratchpad as if it had said it out loud,
    /// which is exactly what the markers exist to prevent.
    ///
    /// Accepted under both spellings clients use: `reasoning_content`
    /// (the DeepSeek convention frink emits) and `reasoning`
    /// (what the OpenAI Responses and Anthropic surfaces call it), so a
    /// client can replay a turn shaped the way it received it.
    #[serde(default, alias = "reasoning")]
    pub(crate) reasoning_content: Option<String>,
}

impl ChatMessage {
    /// The text this message actually contributes to a rendered
    /// prompt: `content` verbatim for an ordinary message, or (for a
    /// replayed assistant message carrying `tool_calls`) each call
    /// re-rendered as the same `<tool_call>{...}</tool_call>` marker
    /// text a model is asked to produce for a *new* call -- see
    /// `chat_template`'s module doc comment for why.
    fn rendered_content(&self) -> String {
        let mut out = self
            .content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default();
        if let Some(calls) = &self.tool_calls {
            for call in calls {
                out.push_str(&format!(
                    "<tool_call>{{\"name\": \"{}\", \"arguments\": {}}}</tool_call>",
                    call.function.name, call.function.arguments
                ));
            }
        }
        out
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallIn {
    #[serde(default)]
    #[allow(dead_code)]
    id: String,
    #[serde(rename = "type", default)]
    #[allow(dead_code)]
    kind: String,
    function: ToolCallFunctionIn,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolCallFunctionIn {
    name: String,
    /// A JSON-encoded string (the real OpenAI convention for
    /// `tool_calls[].function.arguments`), not a nested object --
    /// spliced directly into the re-rendered `<tool_call>{...}` marker
    /// text since it's already valid JSON.
    arguments: String,
}

/// A tool definition in the real OpenAI request shape:
/// `{"type": "function", "function": {"name", "description", "parameters"}}`.
#[derive(Debug, Clone, Deserialize)]
struct ToolDef {
    #[serde(rename = "type", default)]
    #[allow(dead_code)]
    kind: String,
    function: ToolFunctionDef,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolFunctionDef {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
}

/// OpenAI's `tool_choice`: `"auto"`/`"none"`/`"required"`, or an object
/// pinning one specific function.
///
/// All four are honoured now. `"none"` hides the tools from the prompt;
/// `"auto"` offers them; `"required"` and a named function FORCE a call,
/// by compiling the offered tools into a grammar the decode loop must
/// keep parseable (`crate::tool_grammar`). Before that grammar existed
/// the last two were a 501, because a server that is asked to force a
/// call and can only ask for one in the prompt has not done what it was
/// told.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ToolChoice {
    Mode(String),
    Specific(serde_json::Value),
}

/// OpenAI's `stop` field accepts either a single string or an array of
/// strings.
#[derive(Deserialize)]
#[serde(untagged)]
enum StopParam {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// llama.cpp's `--min-p`. Not an OpenAI field; accepted under the
    /// same spelling llama.cpp's server uses, because a client
    /// that sends it and is silently served an unfiltered distribution
    /// cannot tell that apart from having had it honoured.
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    /// llama.cpp's `typ_p`, `top_n_sigma`, `xtc_*` and `dry_*`, in ONE
    /// struct shared with the other two routes that take them. See
    /// `sampling_knobs::ExtraSamplerFields`.
    #[serde(flatten)]
    extra_samplers: crate::sampling_knobs::ExtraSamplerFields,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(default)]
    stream: Option<bool>,
    /// Frink extension. `true` asks the server to keep a replay buffer
    /// for this stream so a dropped connection can be resumed from the
    /// last `id:` seen, or drained over the JSON polling fallback.
    ///
    /// It also changes what a dropped socket *means*. Without it, the
    /// connection closing cancels the generation (see the `cancel`
    /// module). With it, the generation keeps running into the replay
    /// buffer -- which is the entire point, and the reason this is the
    /// caller's decision rather than the server's: a tab that navigated
    /// away wants the CPU back, and a tab whose proxy dropped a
    /// 90-second answer wants the answer. `POST /v1/cancel` stops a
    /// resumable stream either way.
    #[serde(default)]
    stream_resumable: Option<bool>,
    /// Run past the model's own end-of-generation tokens, so this
    /// request produces exactly `max_tokens`.
    ///
    /// A serving-benchmark knob, under the spelling the other
    /// OpenAI-compatible servers use. It
    /// exists because a benchmark whose requests stop at their own EOS
    /// finishes them at different lengths, and the slowest percentile
    /// is then whichever request happened to be asked for the most
    /// tokens -- a fact about the prompts, reported as a fact about the
    /// server. It does NOT withdraw the caller's own `stop` strings.
    #[serde(default)]
    ignore_eos: Option<bool>,
    #[serde(default)]
    tools: Vec<ToolDef>,
    #[serde(default)]
    tool_choice: Option<ToolChoice>,
    /// The OpenAI extension every reasoning-model deployment actually
    /// uses: whatever is in here becomes a top-level variable in the
    /// checkpoint's own chat template, which is how `enable_thinking`
    /// (Qwen3, gemma-4), `thinking` (DeepSeek) and `reasoning_effort`
    /// are really driven. Values here can never shadow the structural
    /// variables (`messages`, `tools`, `add_generation_prompt`) -- see
    /// `frink_models::chat_template::RenderOptions`.
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// OpenAI's own spelling of the same knob. It is folded into
    /// `chat_template_kwargs` before rendering, and loses to an explicit
    /// entry there: a caller who wrote both meant the specific one.
    ///
    /// `"none"` and `"off"` are not gears -- they mean *do not think*,
    /// and are handled by [`ChatCompletionRequest::thinking_direction`]
    /// before any quantization can round them onto a real one.
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// The DeepSeek wire's thinking switch: `{"type": "enabled"}` or
    /// `{"type": "disabled"}`. It decides the direction outright, and
    /// `disabled` beats any effort the same request also carries.
    #[serde(default)]
    thinking: Option<ThinkingSwitch>,
    /// Server-side conversation history key (see the `session`
    /// module): when set, `messages` is treated as
    /// *only the new turn(s)* to append to this session's stored
    /// history, not the whole conversation.
    #[serde(default)]
    session_id: Option<String>,
    /// llama.cpp's `continue_final_message`: render the LAST message,
    /// which must be an assistant turn, as a turn still being written
    /// rather than a closed one, so the model carries on from where
    /// it stopped. `true`, `"reasoning_content"`, `"content"`, or
    /// `false`; unset, a trailing assistant message is continued by
    /// default, as llama.cpp's server does. The whole rule, its
    /// refusals included, is [`continuation`].
    #[serde(default, deserialize_with = "continuation::deserialize")]
    continue_final_message: continuation::ContinueFinalMessage,
    /// llama.cpp's `reasoning_budget_tokens` (alias
    /// `thinking_budget_tokens`): a token budget for the chain of
    /// thought, enforced in the sampler. `-1` or absent takes the
    /// server's `--reasoning-budget`; `0` closes the block the moment it
    /// opens; `N` allows N tokens of thought and then forces the closer.
    /// The range is checked at deserialization, so an out-of-range
    /// value is a 400 naming the field. See [`crate::reasoning_budget`].
    #[serde(default, alias = "thinking_budget_tokens")]
    reasoning_budget_tokens: Option<reasoning_budget::BudgetTokens>,
    /// OpenAI fields we explicitly reject rather than silently ignore.
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<u32>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    /// Declared ONLY so it can be refused by name -- see
    /// [`crate::unsupported_sampling::refuse_logit_bias`], which
    /// `/v1/completions` calls with the same rules. Undeclared, serde
    /// dropped it and the caller got a 200 whose answer was sampled
    /// from unbiased logits, which is indistinguishable from having had
    /// the bias honoured.
    #[serde(default)]
    logit_bias: Option<serde_json::Value>,
    /// llama.cpp's per-request `lora: [{id, scale}]`: the scale of every
    /// loaded adapter for THIS request, unnamed adapters at 0. Resolved
    /// against the loaded adapters by `crate::lora::resolve_request`.
    #[serde(default)]
    lora: Option<Vec<frink_api::LoraScaleRequest>>,
    /// llama.cpp's `samplers`: the ORDER the sampler chain runs in,
    /// either a list of names or the one `;`-separated string
    /// `--samplers` takes.
    ///
    /// Read as `Value` and decided by
    /// [`crate::unsupported_sampling::parse_sampler_order`], shared with
    /// `/v1/completions` and `/completion`, so the three routes cannot
    /// disagree about which samplers exist. A sampler frink does not
    /// implement is refused BY NAME rather than dropped from the chain.
    #[serde(default)]
    samplers: Option<serde_json::Value>,
    /// A GBNF grammar every sampled token must keep parseable.
    ///
    /// llama.cpp's field, spelled the same way, because a client that
    /// already builds a grammar for `llama-server` should not have to
    /// build a second one. Not an OpenAI field: OpenAI states the same
    /// constraint as `response_format: {"type": "json_schema"}`, which
    /// is now compiled through the same grammar engine. Sending BOTH is
    /// two constraints on one generation and is refused -- see
    /// [`crate::grammar_request`], where every spelling is resolved.
    #[serde(default)]
    grammar: Option<String>,
}

/// The output budget a chat request gets when it names none.
///
/// Not OpenAI's legacy 16 -- that floor belongs to `/v1/completions`,
/// where a caller asking for a completion of a fragment usually wants a
/// fragment back. A chat client that omits `max_tokens` wants an
/// answer, and 16 tokens of one reads as a truncated server.
///
/// It is safe to be this large only because the context ceiling CLAMPS
/// rather than refuses (see `generate`): a request whose prompt leaves
/// less than this much room is served with what remains, not rejected
/// over a number the caller never set.
const DEFAULT_CHAT_MAX_TOKENS: usize = 32_768;

/// The DeepSeek-wire thinking switch.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ThinkingSwitch {
    #[serde(rename = "type")]
    pub(crate) kind: String,
}

/// Every spelling a caller can use to steer the template's thinking
/// themselves. If any of these is already present in
/// `chat_template_kwargs`, the protocol-level knobs stand down.
const THINKING_KWARG_KEYS: [&str; 4] = [
    "enable_thinking",
    "thinking",
    "thinking_mode",
    "reasoning_effort",
];

/// The efforts that mean "do not think" rather than naming a gear.
/// Compared after trimming and lowercasing, because a client that sends
/// `"None"` means the same thing.
const DISABLE_EFFORTS: [&str; 2] = ["none", "off"];

fn default_max_tokens() -> usize {
    DEFAULT_CHAT_MAX_TOKENS
}

impl ChatCompletionRequest {
    /// This request's sampler knobs. Resolved to `SamplingParams` by
    /// `sampling_knobs`, shared with `/v1/completions`, so the two
    /// routes cannot disagree about what a knob means or which ones
    /// exist.
    ///
    /// Fallible because `samplers` is parsed here: a chain naming a
    /// sampler this engine does not have is a refusal, never a chain
    /// built without it.
    fn sampling_knobs(&self) -> Result<SamplingKnobs, ApiError> {
        let mut knobs = SamplingKnobs {
            temperature: self.temperature,
            top_p: self.top_p,
            min_p: self.min_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            // The OpenAI wire has no field for the penalty window; only
            // llama.cpp's native `/completion` does. See
            // `SamplingKnobs::penalty_last_n`.
            penalty_last_n: None,
            sampler_order: unsupported_sampling::parse_sampler_order(
                self.samplers.as_ref(),
                "/v1/chat/completions",
            )?,
            ..SamplingKnobs::default()
        };
        self.extra_samplers.apply(&mut knobs);
        Ok(knobs)
    }

    fn sampling_params(
        &self,
        model: crate::sampling_knobs::SamplerModel<'_>,
    ) -> Result<SamplingParams, ApiError> {
        self.sampling_knobs()?.resolve(model).map_err(|e| {
            unsupported_feature(&format!("`dry_multiplier` on /v1/chat/completions: {e}"))
        })
    }

    fn stop_sequences(&self) -> Vec<String> {
        self.stop
            .as_ref()
            .map(|s| match s {
                StopParam::One(v) => vec![v.clone()],
                StopParam::Many(v) => v.clone(),
            })
            .unwrap_or_default()
    }

    /// Real tool-calling is only offered when `tools` is non-empty AND
    /// the client hasn't explicitly disabled it via `tool_choice:
    /// "none"` -- see `ToolChoice`'s doc comment for what the other
    /// values do (nothing different from `"auto"`).
    fn tools_active(&self) -> bool {
        !self.tools.is_empty()
            && !matches!(&self.tool_choice, Some(ToolChoice::Mode(m)) if m == "none")
    }

    /// Whether this request FORCES a tool call, and which tools it may
    /// choose between.
    ///
    /// `"required"` and a named function are the same question with a
    /// different answer set, so they are one function here and one
    /// grammar builder downstream. Everything else -- absent, `"auto"`,
    /// `"none"` -- forces nothing and returns `None`.
    ///
    /// An object `tool_choice` that names nothing is a 400 rather than a
    /// silent `None`: a client that sent `{"type": "function"}` and got
    /// an unforced answer cannot tell that apart from a served one.
    fn forced_tool_choice(&self) -> Result<Option<tool_grammar::Forced<'_>>, ApiError> {
        match &self.tool_choice {
            Some(ToolChoice::Mode(m)) if m == "required" => Ok(Some(tool_grammar::Forced::Any)),
            Some(ToolChoice::Specific(value)) => {
                // OpenAI's shape is `{"type":"function","function":{"name":…}}`;
                // several clients send `{"name":…}` flat, and both name
                // the same thing.
                let name = value
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .or_else(|| value.get("name"))
                    .and_then(|n| n.as_str());
                match name {
                    Some(name) => Ok(Some(tool_grammar::Forced::Named(name))),
                    None => Err(invalid_request(
                        "tool_choice must be \"auto\", \"none\", \"required\", or an object with \
                         function.name",
                        "tool_choice",
                    )),
                }
            }
            _ => Ok(None),
        }
    }

    /// The offered tools, reduced to what [`tool_grammar`] needs.
    fn tool_specs(&self) -> Vec<tool_grammar::ToolSpec<'_>> {
        self.tools
            .iter()
            .map(|t| tool_grammar::ToolSpec {
                name: &t.function.name,
                parameters: t.function.parameters.as_ref(),
            })
            .collect()
    }

    /// The `chat_template_kwargs` this request actually renders with.
    ///
    /// Five rules, all of them from `frink-edge`:
    ///
    /// * **An explicit knob wins wholesale.** A caller who already set
    ///   any of `enable_thinking` / `thinking` / `thinking_mode` /
    ///   `reasoning_effort` inside `chat_template_kwargs` has said what
    ///   they want; the protocol-level knobs are then ignored entirely
    ///   rather than merged, because a merge would let a default
    ///   contradict an explicit request.
    /// * **`none` and `off` are not gears.** `reasoning_effort: "none"`
    ///   means *turn thinking off* and broadcasts the off pair; it must
    ///   not be quantized onto the nearest gear, which would turn "do
    ///   not think" into "think a little". Same for the DeepSeek-wire
    ///   `thinking: {"type": "disabled"}`, which beats any effort.
    ///
    /// * **Thinking follows the tools.** Offering tools turns thinking
    ///   on even when the caller said nothing, because some encoders
    ///   emit well-formed tool calls only in thinking mode
    ///   ([`crate::policy::effort::resolve_thinking_mode`]).
    /// * **Effort is quantized onto what this checkpoint grades.** A
    ///   template that accepts only the OpenAI triple must not be sent
    ///   `minimal`; it is mapped to the nearest gear, or dropped when no
    ///   gear is close enough, rather than interpolated verbatim into
    ///   the prompt ([`crate::policy::effort::sanitize_effort`], against the
    ///   profile probed at load).
    /// * **One value, every spelling.** The graded-strength dialect
    ///   reads `reasoning_strength`; a Jinja template ignores variables
    ///   it does not declare, so broadcasting costs nothing and removes
    ///   a per-family routing table
    ///   ([`crate::policy::effort::broadcast_effort_spellings`]).
    ///
    /// Every render path has to do this identically -- a request that
    /// validates against one prompt and generates from another is the
    /// failure this returns a single value to prevent.
    /// Which way this request steers thinking, before any template is
    /// consulted: `Some(false)` off, `Some(true)` on, `None` unstated.
    ///
    /// `thinking: {"type": …}` decides outright and `disabled` wins over
    /// any effort, because a client that sent both a switch and a gear
    /// meant the switch -- the gear is what it would use *if* thinking
    /// were on.
    fn thinking_direction(&self) -> Option<bool> {
        if let Some(switch) = &self.thinking {
            return match switch.kind.trim().to_ascii_lowercase().as_str() {
                "disabled" => Some(false),
                "enabled" => Some(true),
                // An unrecognized type is not a silent default -- see
                // `validate_supported_fields`, which rejects it.
                _ => None,
            };
        }
        let effort = self.reasoning_effort.as_ref()?;
        DISABLE_EFFORTS
            .contains(&effort.trim().to_ascii_lowercase().as_str())
            .then_some(false)
    }

    fn resolve_template_kwargs(
        &self,
        template: &chat_template::PromptTemplate,
    ) -> serde_json::Map<String, serde_json::Value> {
        let mut kwargs = self.chat_template_kwargs.clone().unwrap_or_default();
        // Whether the caller steered the template themselves. Read
        // BEFORE anything is added, or every request looks explicit
        // from the second statement on.
        let caller_steered = THINKING_KWARG_KEYS.iter().any(|k| kwargs.contains_key(*k));

        if !caller_steered {
            match self.thinking_direction() {
                Some(false) => {
                    for (k, v) in crate::policy::effort::thinking_off_kwargs() {
                        kwargs.insert(k, v);
                    }
                    // Nothing below applies: an effort would re-enter a
                    // block this request just closed.
                    return kwargs;
                }
                Some(true) => {
                    for (k, v) in crate::policy::effort::thinking_on_kwargs() {
                        kwargs.insert(k, v);
                    }
                }
                None => {}
            }
            if let Some(effort) = &self.reasoning_effort {
                kwargs
                    .entry("reasoning_effort".to_string())
                    .or_insert_with(|| serde_json::json!(effort));
            }
        }

        let offered: Vec<serde_json::Value> = if self.tools_active() {
            self.tools.iter().map(chat_template::tool_json).collect()
        } else {
            Vec::new()
        };
        let thinking = crate::policy::effort::resolve_thinking_mode(Some(&kwargs), Some(&offered));
        if thinking == crate::policy::effort::ThinkingMode::Thinking {
            for (k, v) in crate::policy::effort::thinking_on_kwargs() {
                kwargs.entry(k).or_insert(v);
            }
        }
        match crate::policy::effort::sanitize_effort(&mut kwargs, template.efforts()) {
            crate::policy::effort::EffortMapping::Mapped(to) => {
                tracing::debug!("reasoning_effort quantized to {}", to.as_str());
            }
            crate::policy::effort::EffortMapping::Dropped => {
                tracing::debug!(
                    "reasoning_effort dropped: this checkpoint's template grades no gear close \
                     enough, so its own default applies"
                );
            }
            crate::policy::effort::EffortMapping::Unchanged => {}
        }
        crate::policy::effort::broadcast_effort_spellings(&mut kwargs);
        kwargs
    }

    /// Reject OpenAI fields we do not implement, and `tool_choice`
    /// values that would silently lie (required / named function).
    fn validate_supported_fields(&self) -> Result<(), ApiError> {
        // An explicit zero is a client error, not "unset". Serde already
        // told them apart -- an absent field became
        // `DEFAULT_CHAT_MAX_TOKENS` -- so a 0 here is one the caller
        // wrote, and the engine cannot serve a zero-token budget: the
        // request would never become decodable and the client would wait
        // for an answer that cannot arrive.
        if self.max_tokens == 0 {
            return Err(invalid_request(
                "max_tokens must be at least 1",
                "max_tokens",
            ));
        }
        // An unrecognized switch is refused rather than read as "on":
        // a client that misspells `disabled` and is served a thinking
        // model anyway has been silently given the opposite of what it
        // asked for.
        if let Some(switch) = &self.thinking {
            let kind = switch.kind.trim().to_ascii_lowercase();
            if kind != "enabled" && kind != "disabled" {
                return Err(invalid_request(
                    "thinking.type must be \"enabled\" or \"disabled\"",
                    "thinking.type",
                ));
            }
        }
        for msg in &self.messages {
            if msg.content.as_ref().is_some_and(MessageContent::has_image) {
                return Err(unsupported_feature(
                    "image_url content parts are not implemented (multimodal/VL deferred, see docs/API.md)",
                ));
            }
        }
        if self.logprobs == Some(true) || self.top_logprobs.is_some() {
            return Err(unsupported_feature(
                "logprobs / top_logprobs are not implemented yet (see docs/API.md)",
            ));
        }
        if self.n.is_some_and(|n| n > 1) {
            return Err(unsupported_feature(
                "n > 1 is not implemented (single completion only)",
            ));
        }
        unsupported_sampling::refuse_logit_bias(self.logit_bias.as_ref(), "/v1/chat/completions")?;
        // Parsed here as well as in `sampling_knobs` so a bad chain is
        // a 400/501 before any prompt is rendered. The same function
        // both times, so there is no second opinion to drift from.
        unsupported_sampling::parse_sampler_order(self.samplers.as_ref(), "/v1/chat/completions")?;
        // Every spelling of "constrain the output", resolved by the one
        // function that knows the rule: `grammar` is compiled and a
        // `response_format` is decided in full -- its schema converted,
        // its unhonoured members refused by name, its unknown types
        // refused by the type they named. Done here so all of that is a
        // 400 before any prompt is rendered. The result is recompiled in
        // `generation_params`, which is the only other caller: a grammar
        // is a small parse, and one rule in two places would be two
        // rules soon enough.
        //
        // Kept as ONE call rather than a second `match` on
        // `response_format` beside it. The one that used to be here
        // answered `json_schema` with "only json_object is supported"
        // and had to be kept in step with the module by hand.
        let stated_grammar =
            grammar_request::for_request(self.grammar.as_deref(), self.response_format.as_ref())?;
        // A forced `tool_choice` is served by compiling the offered tools
        // into a grammar (`tool_grammar`). What can be checked without
        // knowing which checkpoint is loaded is checked here, so the
        // caller's own mistakes are refused before a prompt is rendered;
        // the rest -- whether the served family's wire format has a
        // grammar at all -- needs the model and is refused in
        // `generation_params_for_template`.
        if let Some(forced) = self.forced_tool_choice()? {
            if self.tools.is_empty() {
                return Err(invalid_request(
                    "tool_choice forces a tool call, but no tools were offered",
                    "tool_choice",
                ));
            }
            if let tool_grammar::Forced::Named(name) = forced {
                if !self.tools.iter().any(|t| t.function.name == name) {
                    return Err(invalid_request(
                        &format!(
                            "tool_choice names {name:?}, which is not one of the tools offered"
                        ),
                        "tool_choice",
                    ));
                }
            }
            // Two different constraints on one generation. Serving the
            // one we happen to compile last is not answering either.
            //
            // Asked of the RESOLVED grammar rather than of
            // `self.grammar`: a `response_format` json_schema states one
            // too, and a check spelled against one field would have let
            // the other through -- `generation_params_for_template`
            // overwrites `params.grammar` with the tool-call grammar on
            // the strength of this refusal having happened.
            if stated_grammar.is_some() {
                return Err(invalid_request(
                    "a forced tool_choice and a \"grammar\" or response_format \"json_schema\" \
                     are two different constraints on the same generation; send one",
                    "tool_choice",
                ));
            }
            if self.json_object_mode() {
                return Err(invalid_request(
                    "a forced tool_choice cannot be combined with response_format json_object: \
                     the tool-call markers are not JSON",
                    "tool_choice",
                ));
            }
        }
        Ok(())
    }

    /// `stop_sequences()` plus `</tool_call>` when tool-calling is
    /// active -- reusing the existing stop-sequence machinery
    /// (`generate::generate`'s `earliest_stop_match`) to end generation
    /// right after a tool call's JSON body, rather than adding any new
    /// decode-time logic. See `tool_preamble`'s doc comment for the
    /// full real, disclosed approach.
    fn effective_stop_sequences(&self) -> Vec<String> {
        let mut stop = self.stop_sequences();
        if self.tools_active() {
            stop.push("</tool_call>".to_string());
        }
        stop
    }

    fn json_object_mode(&self) -> bool {
        self.response_format
            .as_ref()
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str())
            == Some("json_object")
    }
}

#[derive(Serialize)]
struct ChatCompletionChoice {
    index: usize,
    message: ChatCompletionResponseMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionResponseMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// A reasoning model's chain of thought, split out of `content`.
    /// Absent for a model that emitted none, which is also what a
    /// client that does not know the field sees.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallOut>>,
}

#[derive(Serialize, Clone)]
struct ToolCallOut {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: ToolCallFunctionOut,
}

/// One tool call as a **streamed delta**.
///
/// OpenAI's incremental shape: `index` correlates the pieces, and every
/// other field is optional because the first delta of a call carries
/// its identity and the ones after it carry only more argument text. A
/// buffered path expresses a whole call as a delta with every field
/// set, so there is one type on the wire rather than two.
#[derive(Serialize, Clone)]
struct ToolCallDelta {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    function: ToolCallFunctionDelta,
}

#[derive(Serialize, Clone, Default)]
struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// A literal continuation of this call's arguments JSON. A client
    /// concatenates them in `index` order and parses the result.
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

impl ToolCallDelta {
    /// The whole call in one delta, for a path that had it all along.
    fn whole(index: usize, name: String, arguments: String) -> Self {
        ToolCallDelta {
            index,
            id: Some(format!("call_{index}")),
            kind: Some("function"),
            function: ToolCallFunctionDelta {
                name: Some(name),
                arguments: Some(arguments),
            },
        }
    }

    /// The opening delta: identity, and no arguments yet.
    fn opening(index: usize, name: String) -> Self {
        ToolCallDelta {
            index,
            id: Some(format!("call_{index}")),
            kind: Some("function"),
            function: ToolCallFunctionDelta {
                name: Some(name),
                arguments: Some(String::new()),
            },
        }
    }

    /// A continuation: more argument text for a call already opened.
    fn arguments(index: usize, fragment: String) -> Self {
        ToolCallDelta {
            index,
            id: None,
            kind: None,
            function: ToolCallFunctionDelta {
                name: None,
                arguments: Some(fragment),
            },
        }
    }
}

#[derive(Serialize, Clone)]
struct ToolCallFunctionOut {
    name: String,
    /// A JSON-encoded string, matching the real OpenAI
    /// `tool_calls[].function.arguments` convention (see
    /// `ToolCallFunctionIn::arguments`'s doc comment).
    arguments: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    /// Non-standard extension: the same value as `id`, stated under the
    /// name the rest of frink keys by (metrics, logs, `POST /cancel`
    /// once it exists). `id` is OpenAI's completion id and a client has
    /// no way to know frink also uses it as the request key -- saying
    /// so costs one field and removes the guess.
    request_id: String,
    object: &'static str,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    /// OpenAI-convention token accounting (prompt/completion/total),
    /// counted from the exact ids the generation loop processed. On a
    /// whole-response cache hit, this is the original computation's
    /// accounting (same prompt, same deterministic outcome).
    usage: generate::Usage,
    /// Non-standard extension field (not part of the OpenAI API
    /// contract, but additive and harmless to OpenAI-compatible
    /// clients that ignore unknown fields): "hit" if this exact
    /// cacheable request was already computed, "miss" if this request
    /// just computed and cached a fresh completion, or "skip" if
    /// nothing was stored -- either the request wasn't cacheable at all
    /// (sampling without a seed -- see
    /// `ChatCompletionRequest::is_cacheable`) or the answer was not a
    /// complete one and may not be replayed to anybody (a cancelled
    /// generation -- see `response_cache::CachedCompletion::cacheable`).
    frink_cache: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// See `ChatCompletionResponseMessage::reasoning_content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Serialize)]
struct ChatCompletionChunkChoice {
    index: usize,
    delta: ChatCompletionChunkDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    /// Present on the **first** chunk of a stream (see
    /// `ChatCompletionResponse::request_id`). A client learns the key
    /// for this generation before any content arrives, so a live view
    /// can correlate metrics with the stream it is rendering instead of
    /// guessing which in-flight request is "probably mine" -- a guess
    /// that mis-attributes the moment two chats run at once.
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    object: &'static str,
    model: String,
    choices: Vec<ChatCompletionChunkChoice>,
    /// Present only on the final chunk (the one carrying
    /// `finish_reason`), mirroring OpenAI's stream `usage` shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<generate::Usage>,
}

/// Liveness, readiness and capabilities in one cheap answer (see the
/// `health` module for why detection is a visible state rather than a
/// gap). Never behind auth or rate limiting, and never blocking: this is
/// the endpoint a supervisor asks when it is deciding whether to kill
/// the process.
async fn health(State(state): State<Arc<AppState>>) -> Response {
    let snapshot = state.detection.snapshot();
    let mut capabilities = snapshot.capabilities;
    let active = state.active();

    // Model-derived capabilities need no probing, so they are answered
    // even while backend detection is still running.
    capabilities.push(match active.as_deref() {
        // `unavailable` was defined in Phase 1 but unreachable, because
        // the server only bound the port after a successful load. With
        // `/admin/models/unload` it is a state a client can actually
        // observe, and it must not read as "loaded but synthetic".
        None => frink_api::Capability::unavailable(
            frink_api::health::capability::REAL_WEIGHTS,
            frink_api::health::reason::MODEL_NOT_LOADED,
            "No model is loaded. POST /admin/models/load with an id from GET /admin/models.",
        ),
        Some(active) if active.is_synthetic() => frink_api::Capability::unavailable(
            frink_api::health::capability::REAL_WEIGHTS,
            frink_api::health::reason::MODEL_NOT_LOADED,
            "Serving synthetic random weights: set FRINK_MODEL_PATH (or -m) to a real \
             checkpoint. Output from this model is noise.",
        ),
        // An encoder is real weights and is genuinely serving, so this
        // is `available` -- but a supervisor reading "serving X" and
        // then getting 501 from /v1/chat/completions learned nothing.
        // The detail says which endpoint this checkpoint is for.
        // NOT a hard-coded /v1/embeddings any more: a reranker is an
        // encoder too, and its pooling_type is RANK, which
        // /v1/embeddings refuses and /v1/rerank is for. See
        // `rerank::encoder_endpoints`, which `/v1/models` reads as well
        // so the two cannot disagree.
        Some(active) if active.encoder().is_some() => {
            let endpoints = active
                .encoder()
                .map(|e| encoder_endpoints(e))
                .unwrap_or_default();
            let served_by = match endpoints.is_empty() {
                true => "no endpoint in this build serves it".to_string(),
                false => format!("served by {}", endpoints.join(" and ")),
            };
            frink_api::Capability::available(
                frink_api::health::capability::REAL_WEIGHTS,
                format!(
                    "Serving the real embedding checkpoint '{}'. This is an ENCODER, \
                     {served_by}; generation endpoints refuse it.",
                    active.name(),
                ),
            )
        }
        Some(active) => frink_api::Capability::available(
            frink_api::health::capability::REAL_WEIGHTS,
            format!("Serving the real checkpoint '{}'.", active.name()),
        ),
    });
    capabilities.push(if active.as_ref().is_some_and(|a| a.batcher.is_some()) {
        frink_api::Capability::available(
            frink_api::health::capability::CONTINUOUS_BATCHING,
            if state.continuous_batching_enabled && continuous_batching_env().is_none() {
                "On by default on Metal. Concurrent requests share one batched decode worker."
            } else {
                "Concurrent requests share one batched decode step."
            },
        )
    } else if state.metal_private_decode_gate.is_some() {
        frink_api::Capability::unavailable(
            frink_api::health::capability::CONTINUOUS_BATCHING,
            frink_api::health::reason::DISABLED,
            "Off; private Metal decodes serialize (one at a time). Set FRINK_CONTINUOUS_BATCHING=1 or --cont-batching for parallel serving.",
        )
    } else {
        frink_api::Capability::unavailable(
            frink_api::health::capability::CONTINUOUS_BATCHING,
            frink_api::health::reason::DISABLED,
            "Off; set FRINK_CONTINUOUS_BATCHING=1 (incompatible with a KV pool or prefix cache).",
        )
    });

    let last_request_ms = state
        .last_request_ms
        .load(std::sync::atomic::Ordering::Relaxed);
    let uptime = state.started_at.elapsed();
    // Readiness is "can this server generate", and with nothing loaded
    // it cannot -- so `unavailable` (503) wins over whatever the backend
    // probe concluded. Phase 1 defined this state but nothing could
    // reach it, because the process only bound the port after a
    // successful load; `/admin/models/unload` makes it reachable, and a
    // 200 `ready` here would tell a supervisor to send traffic that is
    // guaranteed to 503.
    let health_state = if active.is_none() {
        frink_api::HealthState::Unavailable
    } else {
        snapshot.state
    };
    let body = frink_api::HealthResponse {
        state: health_state,
        reason: match health_state {
            frink_api::HealthState::Ready => None,
            frink_api::HealthState::Unavailable => {
                Some(frink_api::health::reason::MODEL_NOT_LOADED.to_string())
            }
            frink_api::HealthState::Detecting => {
                Some(frink_api::health::reason::DETECTING.to_string())
            }
        },
        detail: match health_state {
            frink_api::HealthState::Ready => None,
            frink_api::HealthState::Unavailable => Some(
                "No model is loaded. POST /admin/models/load with an id from GET /admin/models."
                    .to_string(),
            ),
            frink_api::HealthState::Detecting => {
                Some("Probing available compute backends.".to_string())
            }
        },
        model: active
            .as_deref()
            .map(|active| frink_api::health::ModelSummary {
                id: active.name().to_string(),
                tokenizer: active.tokenizer_kind().to_string(),
                synthetic_weights: active.is_synthetic(),
            }),
        capabilities,
        version: env!("CARGO_PKG_VERSION").to_string(),
        pid: std::process::id(),
        uptime_seconds: uptime.as_secs_f64(),
        server_time_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis().min(u64::MAX as u128) as u64)
            .unwrap_or(0),
        last_request_age_seconds: (last_request_ms > 0)
            .then(|| uptime.as_secs_f64() - (last_request_ms as f64 / 1000.0))
            .map(|age| age.max(0.0)),
    };

    let status =
        StatusCode::from_u16(body.state.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(body)).into_response()
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // OpenAI's `/v1/models` lists what can be *used* right now, which
    // after an unload is nothing. The inventory of what is on disk is a
    // different question and lives at `/admin/models`.
    let Some(active) = state.active() else {
        return Json(serde_json::json!({ "object": "list", "data": [] }));
    };
    let mut model_entry = serde_json::json!({
        "id": active.name(),
        "object": "model",
        "frink_synthetic_weights": active.is_synthetic(),
        "frink_tokenizer": active.tokenizer_kind(),
    });
    // An encoder is listed -- it IS what is loaded, and a client asking
    // "what can I use" must be told about it -- but it is listed as
    // what it is. `frink_endpoints` is the machine-readable half of
    // the 501 a generation route would answer with: a client that reads
    // it never has to send the request to find out.
    if let Some(encoder) = active.encoder() {
        model_entry["frink_model_kind"] = serde_json::json!("embedding");
        model_entry["frink_endpoints"] = serde_json::json!(encoder_endpoints(encoder));
        model_entry["frink_n_embd"] = serde_json::json!(encoder.n_embd());
        model_entry["frink_pooling"] = serde_json::json!(encoder.pooling_type().name());
        model_entry["frink_context_length"] = serde_json::json!(encoder.n_ctx_train());
    }
    // Which reasoning gears this checkpoint really has, learned by
    // probing its own template at load. A checkpoint that says nothing
    // about thinking carries NEITHER field rather than an empty list:
    // an empty list reads as "asked, and it has no gears", which is a
    // different claim from "this is not a reasoning model". An encoder
    // is not asked at all, for the same reason -- it has no template to
    // probe, and `ThinkGears::default()` would be an invented answer.
    if let Some(model) = active.generative_opt() {
        let parser_configured = active.reasoning_format().is_some();
        let gears = model.chat_template().think_gears(parser_configured);
        if !gears.is_empty() {
            model_entry["supported_reasoning_efforts"] = serde_json::json!(gears.supported);
            if let Some(default) = &gears.default {
                model_entry["default_reasoning_effort"] = serde_json::json!(default);
            }
            // What to SEND for each gear, so a client selects one without
            // knowing that "off" is two booleans and "high" is a string.
            model_entry["reasoning_effort_kwargs"] = serde_json::json!(gears.kwargs);
        }
    }
    if let Some(mcp) = &state.mcp {
        model_entry["frink_mcp"] = mcp.models_metadata();
    }
    Json(serde_json::json!({
        "object": "list",
        "data": [model_entry]
    }))
}

/// `GET /v1/stats`: what is happening *now*.
///
/// Distinct from `/admin/stats`, which is the historical ring. The two
/// throughput figures come from sliding windows, so an idle server
/// reports 0 rather than the rate it managed while it was busy -- a
/// cumulative average never comes back down, and a status bar showing
/// one is reporting the past as the present.
///
/// Latency is the ring's p95, nearest-rank, so it names a request that
/// really took that long. Both it and the mean time-to-first-token are
/// `null` rather than `0` when nothing can be said: a non-streamed
/// request has no TTFT, and averaging those in as zero would make the
/// server look faster the fewer clients stream.
async fn serving_stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let now_ms = state.uptime().as_millis().min(u64::MAX as u128) as u64;
    let mut serving = state.serving.lock().unwrap_or_else(|p| p.into_inner());
    let active = state.active();
    Json(serde_json::json!({
        "model": active.as_ref().map(|a| a.name()),
        "state": state
            .maintenance
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .state()
            .as_str(),
        "uptime_s": state.uptime().as_secs(),
        "throughput": {
            "decode_tps": (serving.decode_tokens_per_second(now_ms) * 10.0).round() / 10.0,
            "prefill_tps": (serving.prefill_tokens_per_second(now_ms) * 10.0).round() / 10.0,
        },
        "requests": {
            "active": state.cancels.live_count(),
            "completed": state.stats.recorded_total(),
            "p95_ms": state.stats.p95_duration_ms(),
            "ttft_mean_ms": state.stats.ttft_mean_ms(),
            "prompt_tokens_total": state.stats.tokens_prompt_total(),
            "completion_tokens_total": state.stats.tokens_generated_total(),
        },
        // Served here so a status bar tracking throughput and pressure
        // makes ONE request rather than two. Upstream stamps the same
        // gauges on every reply of the batch; frink does not, because
        // the reply shapes here are OpenAI's and Anthropic's and a pool
        // gauge on a `chat.completion` is a field no client asked for.
        "pools": cache_admin::pool_gauges(&state),
        // What the engine is REALLY using, beside the budget it was
        // sized against. `null` when no live figure can be read.
        "memory": cache_admin::footprint_json(&state),
    }))
}

#[derive(Deserialize)]
struct RequestsQuery {
    #[serde(default)]
    since: u64,
    #[serde(default = "default_requests_limit")]
    limit: usize,
}

fn default_requests_limit() -> usize {
    stats::MAX_PAGE
}

/// `GET /v1/requests?since=&limit=`: an incremental page of the ring.
///
/// The cursor is all-time, so a poller that keeps up reads each row
/// exactly once and never re-reads. `missed` is the honest half: rows
/// that existed and were evicted before this poll could see them. A
/// client polling slower than the server finishes requests needs to
/// know that, rather than have it hidden by a shorter page.
async fn recent_requests(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<RequestsQuery>,
) -> Json<serde_json::Value> {
    let (rows, cursor, missed) = state.stats.page(q.since, q.limit);
    Json(serde_json::json!({
        "requests": rows,
        "next_cursor": cursor,
        "missed": missed,
        "total": state.stats.recorded_total(),
    }))
}

#[derive(Serialize)]
struct CombinedCacheStats {
    response_cache: response_cache::CacheStats,
    /// `None` when `FRINK_PREFIX_CACHE_ENTRIES` isn't set.
    prefix_cache: Option<frink_models::PrefixCacheStats>,
}

async fn cache_stats(State(state): State<Arc<AppState>>) -> Json<CombinedCacheStats> {
    Json(CombinedCacheStats {
        response_cache: lock_cache(&state.response_cache).stats(),
        prefix_cache: state
            .prefix_cache
            .as_ref()
            .map(|pc| pc.lock().unwrap_or_else(|p| p.into_inner()).stats()),
    })
}

/// Prometheus text-exposition format (`# HELP`/`# TYPE` plus
/// `name value` lines), so this endpoint can be scraped directly by a
/// Prometheus server or anything compatible with that format without
/// frink needing to speak any particular metrics client library.
async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    use std::sync::atomic::Ordering;

    let cache_stats = lock_cache(&state.response_cache).stats();
    let active = state.active();
    let requests_total = state.requests_total.load(Ordering::Relaxed);
    let errors_total = state.request_errors_total.load(Ordering::Relaxed);
    let uptime = state.started_at.elapsed().as_secs_f64();

    let body = format!(
        "# HELP frink_requests_total Total chat completion requests received.\n\
         # TYPE frink_requests_total counter\n\
         frink_requests_total {requests_total}\n\
         # HELP frink_request_errors_total Total chat completion requests that returned an error.\n\
         # TYPE frink_request_errors_total counter\n\
         frink_request_errors_total {errors_total}\n\
         # HELP frink_cache_hits_total Whole-response cache hits.\n\
         # TYPE frink_cache_hits_total counter\n\
         frink_cache_hits_total {}\n\
         # HELP frink_cache_misses_total Whole-response cache misses.\n\
         # TYPE frink_cache_misses_total counter\n\
         frink_cache_misses_total {}\n\
         # HELP frink_cache_entries Current whole-response cache entry count.\n\
         # TYPE frink_cache_entries gauge\n\
         frink_cache_entries {}\n\
         # HELP frink_synthetic_weights 1 if serving synthetic random weights instead of a real checkpoint.\n\
         # TYPE frink_synthetic_weights gauge\n\
         frink_synthetic_weights {}\n\
         # HELP frink_uptime_seconds Seconds since this server process started.\n\
         # TYPE frink_uptime_seconds gauge\n\
         frink_uptime_seconds {uptime}\n",
        cache_stats.hits,
        cache_stats.misses,
        cache_stats.entries,
        // With nothing loaded there are no weights at all, synthetic or
        // otherwise; 0 is the reading that keeps the gauge meaning
        // "serving noise" rather than "serving nothing".
        active
            .as_ref()
            .map(|a| a.is_synthetic() as u8)
            .unwrap_or(0),
    );

    // Expert-store counters, present only when the model streams
    // routed experts through the bounded cache
    // (FRINK_EXPERT_CACHE_BYTES).
    let body = match active
        .as_ref()
        .and_then(|a| a.expert_store_stats())
    {
        Some(es) => format!(
            "{body}\
             # HELP frink_expert_cache_hits_total Expert-store cache hits.\n\
             # TYPE frink_expert_cache_hits_total counter\n\
             frink_expert_cache_hits_total {}\n\
             # HELP frink_expert_cache_misses_total Expert-store cache misses (source reads).\n\
             # TYPE frink_expert_cache_misses_total counter\n\
             frink_expert_cache_misses_total {}\n\
             # HELP frink_expert_cache_evictions_total Expert-store LRU evictions.\n\
             # TYPE frink_expert_cache_evictions_total counter\n\
             frink_expert_cache_evictions_total {}\n\
             # HELP frink_expert_cache_pass_throughs_total Acquires served uncached (entry could not fit the budget).\n\
             # TYPE frink_expert_cache_pass_throughs_total counter\n\
             frink_expert_cache_pass_throughs_total {}\n\
             # HELP frink_expert_cache_bytes_read_total Bytes read from the checkpoint for expert misses.\n\
             # TYPE frink_expert_cache_bytes_read_total counter\n\
             frink_expert_cache_bytes_read_total {}\n\
             # HELP frink_expert_cache_resident_bytes Current expert-cache footprint in bytes.\n\
             # TYPE frink_expert_cache_resident_bytes gauge\n\
             frink_expert_cache_resident_bytes {}\n",
            es.hits, es.misses, es.evictions, es.pass_throughs, es.bytes_read, es.resident_bytes,
        ),
        None => body,
    };

    // Scheduler counters, present only under continuous batching
    // (FRINK_CONTINUOUS_BATCHING=1). `prefill_chunks` next to
    // `prefill_tokens` is what makes chunked prefill observable: their
    // ratio is the effective chunk size the worker actually ran.
    let body = match active.as_ref().and_then(|a| a.batcher.as_ref()) {
        Some(batcher) => {
            let sched = batcher.stats();
            format!(
                "{body}\
                 # HELP frink_prefill_chunks_total Bounded prefill chunks the batch scheduler has run.\n\
                 # TYPE frink_prefill_chunks_total counter\n\
                 frink_prefill_chunks_total {}\n\
                 # HELP frink_prefill_tokens_total Prompt tokens run through chunked prefill.\n\
                 # TYPE frink_prefill_tokens_total counter\n\
                 frink_prefill_tokens_total {}\n\
                 # HELP frink_decode_steps_total Batched decode steps the batch scheduler has run.\n\
                 # TYPE frink_decode_steps_total counter\n\
                 frink_decode_steps_total {}\n\
                 # HELP frink_scheduler_queue_depth Requests waiting for admission to the batch scheduler.\n\
                 # TYPE frink_scheduler_queue_depth gauge\n\
                 frink_scheduler_queue_depth {}\n\
                 # HELP frink_scheduler_queue_rejected_total Requests refused with 503 because the admission queue was full.\n\
                 # TYPE frink_scheduler_queue_rejected_total counter\n\
                 frink_scheduler_queue_rejected_total {}\n\
                 # HELP frink_kv_blocks_total KV blocks in the scheduler's admission budget (0 when unconfigured).\n\
                 # TYPE frink_kv_blocks_total gauge\n\
                 frink_kv_blocks_total {}\n\
                 # HELP frink_kv_blocks_free KV blocks not reserved by an in-flight request.\n\
                 # TYPE frink_kv_blocks_free gauge\n\
                 frink_kv_blocks_free {}\n\
                 # HELP frink_kv_block_size Token positions per KV block.\n\
                 # TYPE frink_kv_block_size gauge\n\
                 frink_kv_block_size {}\n\
                 # HELP frink_kv_rejected_too_large_total Requests refused with 400 because they exceed the whole KV block budget.\n\
                 # TYPE frink_kv_rejected_too_large_total counter\n\
                 frink_kv_rejected_too_large_total {}\n\
                 # HELP frink_kv_rejected_context_length_total Requests refused with 400 for exceeding the per-request context ceiling.\n\
                 # TYPE frink_kv_rejected_context_length_total counter\n\
                 frink_kv_rejected_context_length_total {}\n\
                 # HELP frink_scheduler_aborted_total Requests the batch scheduler stopped because they were cancelled.\n\
                 # TYPE frink_scheduler_aborted_total counter\n\
                 frink_scheduler_aborted_total {}\n\
                 # HELP frink_scheduler_max_seqs Cap on in-flight sequences (-np / FRINK_CB_MAX_SEQS); 0 when unlimited.\n\
                 # TYPE frink_scheduler_max_seqs gauge\n\
                 frink_scheduler_max_seqs {}\n\
                 # HELP frink_scheduler_prefill_chunk Prompt tokens per prefill chunk (-b / -ub / FRINK_CB_PREFILL_CHUNK).\n\
                 # TYPE frink_scheduler_prefill_chunk gauge\n\
                 frink_scheduler_prefill_chunk {}\n",
                sched.prefill_chunks,
                sched.prefill_tokens,
                sched.decode_steps,
                sched.queue_depth,
                sched.queue_rejected,
                sched.kv_blocks_total,
                sched.kv_blocks_free,
                sched.kv_block_size,
                sched.kv_rejected_too_large,
                sched.kv_rejected_context_length,
                sched.aborted,
                sched.max_seqs,
                sched.prefill_chunk,
            )
        }
        None => body,
    };

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    )
        .into_response()
}

pub(crate) type ApiError = (StatusCode, Json<serde_json::Value>);

/// A field the server understands but this value of which it cannot
/// serve. Distinct from [`unsupported_feature`] (501, "frink does not
/// implement this") -- a 400 says the request itself is wrong, which is
/// the difference between a client retrying elsewhere and a client
/// fixing its own body.
pub(crate) fn invalid_request(message: &str, param: &str) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": null,
        }})),
    )
}

pub(crate) fn unsupported_feature(message: &str) -> ApiError {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({"error": {"message": message, "type": "unsupported"}})),
    )
}

pub(crate) fn decode_error_response(e: generate::DecodeError) -> ApiError {
    let status = match e {
        generate::DecodeError::TokenOutOfVocab { .. } => StatusCode::BAD_REQUEST,
        // The request is bigger than the server can ever serve. That
        // is a property of the request, so it is the client's 400 --
        // answering 503 would send it into a retry loop that cannot
        // succeed.
        generate::DecodeError::KvBudgetExceeded { .. } => StatusCode::BAD_REQUEST,
        // Not the client's fault, and true of the exact same request a
        // moment later once capacity frees up -- 503, not 400. The
        // `Retry-After` header these need is stamped centrally by
        // `limits::retry_after`; see that function for why it lives in a
        // layer rather than here.
        generate::DecodeError::KvPoolExhausted | generate::DecodeError::QueueFull { .. } => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        // The caller's grammar against this model's vocabulary, and
        // nothing about the server's load: the same body fails the same
        // way on an idle box, so 400 rather than 503.
        generate::DecodeError::GrammarConstraint { .. } => StatusCode::BAD_REQUEST,
        // Meant to be unreachable -- the route refuses the family with
        // a 501 before rendering -- and a 500 when it is not, because
        // then it is this server's decode path that skipped a seam.
        generate::DecodeError::ReasoningBudget { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    };
    tracing::warn!("decode error: {e}");
    let mut body = serde_json::json!({"error": {"message": e.to_string()}});
    // A refusal against a ceiling names the ceiling and both sides of
    // the arithmetic. "Out of memory" (or a bare 400) tells a caller
    // that something did not fit; it does not tell them whether to
    // shorten the prompt or to run a bigger box, and those are the only
    // two actions available.
    if let generate::DecodeError::KvBudgetExceeded {
        binding,
        estimated_bytes,
        limit_bytes,
        positions,
        positions_limit,
        ..
    } = &e
    {
        body["error"]["type"] = serde_json::json!("invalid_request_error");
        body["error"]["code"] = serde_json::json!(binding);
        body["error"]["binding"] = serde_json::json!(binding);
        body["error"]["estimated_bytes"] = serde_json::json!(estimated_bytes);
        body["error"]["limit_bytes"] = serde_json::json!(limit_bytes);
        body["error"]["positions"] = serde_json::json!(positions);
        body["error"]["positions_limit"] = serde_json::json!(positions_limit);
    }
    // The header carries the same hint (stamped by `limits::retry_after`);
    // repeating it in the body is for clients that read JSON and never
    // look at headers, which is most of them.
    if let Some(secs) = e.retry_after_secs() {
        body["error"]["retry_after_seconds"] = serde_json::json!(secs);
    }
    (status, Json(body))
}

pub(crate) fn join_error_response(e: tokio::task::JoinError) -> ApiError {
    tracing::error!("generation task panicked: {e}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": {"message": "internal error during generation"}})),
    )
}

/// Runs generation for `params` against `model`, calling `emit` for each
/// decoded text chunk. Returns finish reason, usage, and the concatenated
/// text (for sessions / tool-call detection). Pure CPU-bound work with
/// no I/O and no shared lock: safe to run on `spawn_blocking`.
#[allow(clippy::too_many_arguments)] // one clear parameter per concern:
                                     // model + prompt + params, then the three optional shared
                                     // facilities (KV pool, prefix cache, batcher), the context
                                     // ceiling, and the sink. Bundling them would only move the
                                     // same list behind a struct at two call sites.
fn run_generation_emit(
    model: &Model,
    prompt: &str,
    params: &GenerationParams,
    kv_pool: Option<&generate::KvPoolConfig>,
    paged_kv: Option<&generate::PagedKvConfig>,
    prefix_cache: Option<&Mutex<PrefixCache>>,
    continuous_batcher: Option<&serving::batch::ContinuousBatcher>,
    ceiling: Option<&budget::ContextCeiling>,
    metal_private_decode_gate: Option<&std::sync::Mutex<()>>,
    mut emit: impl FnMut(&str),
) -> Result<(FinishReason, generate::Usage, String), generate::DecodeError> {
    let synthetic = model.is_synthetic();
    // Held for the whole generation: a `POST /lora-adapters`, or a
    // request whose `lora` field overrides the scales, waits for this
    // one to finish rather than changing the weights under it. See
    // `crate::lora`.
    let _lora_lease = lora::lease(model, params.lora.as_deref());
    let mut chunks = Vec::new();
    // Layer 1 of the stop machinery is resolved exactly here, because
    // this is the one place that has both the request's stop strings
    // and the model's tokenizer. Both the batched and the private
    // decode paths below read the result off the params, so there is
    // one answer rather than two that can drift.
    let params = &{
        let mut resolved = params.clone();
        resolved.stop_token_ids = crate::stop::resolve_stop_tokens(&resolved.stop, |text| {
            model.encode(text, SpecialTokens::Parse)
        });
        // The reasoning budget's markers, for the same reason and at
        // the same seam: `<think>` is a token id only to this model,
        // and whether the prompt already opened the block is a fact
        // about the rendered prompt, which this is the last place to
        // hold beside the tokenizer.
        resolved.reasoning_budget = resolved
            .reasoning_budget
            .armed(resolved.reasoning, prompt, |text| {
                model.encode(text, SpecialTokens::Parse)
            })
            .map_err(|detail| generate::DecodeError::ReasoningBudget { detail })?;
        resolved
    };
    let used_batcher = matches!((model, continuous_batcher), (Model::Gguf(_), Some(_)));
    let _metal_private_guard =
        acquire_metal_private_decode_gate(metal_private_decode_gate, used_batcher);
    let (finish, usage) = match model {
        Model::Gguf(m) => {
            if let Some(batcher) = continuous_batcher {
                let mut tokens = m.tokenizer.encode(prompt, SpecialTokens::Parse);
                frink_models::tokenizer::prepend_bos(&mut tokens, m.bos_id);
                let (finish, _generated_ids, text, usage) = if synthetic {
                    batcher.generate(tokens, params.clone(), m.stop_tokens.clone())?
                } else {
                    batcher.generate_streaming(
                        tokens,
                        params.clone(),
                        m.stop_tokens.clone(),
                        Some(|chunk: &str| {
                            if !chunk.is_empty() {
                                chunks.push(chunk.to_string());
                                emit(chunk);
                            }
                        }),
                    )?
                };
                if !text.is_empty() && chunks.is_empty() {
                    chunks.push(text);
                }
                (finish, usage)
            } else {
                generate::generate(
                    &m.decoder,
                    m.tokenizer.as_ref(),
                    &m.stop_tokens,
                    m.bos_id,
                    prompt,
                    params,
                    kv_pool,
                    paged_kv,
                    prefix_cache,
                    ceiling,
                    |chunk| {
                        chunks.push(chunk.to_string());
                        if !synthetic {
                            emit(chunk);
                        }
                    },
                )?
            }
        }
        Model::Kimi(m) => generate::generate_engine(
            &m.engine,
            &m.tokenizer,
            &m.stop_tokens,
            None,
            prompt,
            params,
            |chunk| {
                chunks.push(chunk.to_string());
                if !synthetic {
                    emit(chunk);
                }
            },
        )?,
        Model::Mla(m) => generate::generate_engine(
            &m.engine,
            &m.tokenizer,
            &m.stop_tokens,
            m.bos_id,
            prompt,
            params,
            |chunk| {
                chunks.push(chunk.to_string());
                if !synthetic {
                    emit(chunk);
                }
            },
        )?,
        Model::Gemma4(m) => generate::generate_engine(
            &m.engine,
            &m.tokenizer,
            &m.stop_tokens,
            m.bos_id,
            prompt,
            params,
            |chunk| {
                chunks.push(chunk.to_string());
                if !synthetic {
                    emit(chunk);
                }
            },
        )?,
        Model::Glm52(m) => generate::generate_engine(
            &m.engine,
            &m.tokenizer,
            &m.stop_tokens,
            m.bos_id,
            prompt,
            params,
            |chunk| {
                chunks.push(chunk.to_string());
                if !synthetic {
                    emit(chunk);
                }
            },
        )?,
    };

    let mut full = chunks.concat();
    if synthetic {
        full = format!(
            "[frink synthetic-weight demo: no real checkpoint loaded -- set FRINK_MODEL_PATH \
             to serve a real model. Decoded ids -> {full:?}]"
        );
        emit(&full);
    } else if used_batcher && !full.is_empty() && chunks.is_empty() {
        emit(&full);
    }

    Ok((finish, usage, full))
}

/// Collecting wrapper around [`run_generation_emit`] for non-streaming
/// paths and tests.
#[allow(clippy::too_many_arguments)] // mirrors `run_generation_emit`
                                     // exactly, minus the sink; see its note.
pub(crate) fn run_generation(
    model: &Model,
    prompt: &str,
    params: &GenerationParams,
    kv_pool: Option<&generate::KvPoolConfig>,
    paged_kv: Option<&generate::PagedKvConfig>,
    prefix_cache: Option<&Mutex<PrefixCache>>,
    continuous_batcher: Option<&serving::batch::ContinuousBatcher>,
    ceiling: Option<&budget::ContextCeiling>,
    metal_private_decode_gate: Option<&std::sync::Mutex<()>>,
) -> Result<(Vec<String>, FinishReason, generate::Usage), generate::DecodeError> {
    let (finish, usage, full) = run_generation_emit(
        model,
        prompt,
        params,
        kv_pool,
        paged_kv,
        prefix_cache,
        continuous_batcher,
        ceiling,
        metal_private_decode_gate,
        |_| {},
    )?;
    Ok((
        if full.is_empty() {
            Vec::new()
        } else {
            vec![full]
        },
        finish,
        usage,
    ))
}

/// Render a conversation into the prompt the served checkpoint expects.
///
/// Who describes the tools depends on the template: one that reads
/// `tools` is handed them structurally and owns the whole grammar, and
/// one that does not gets [`tool_preamble`] as an extra leading system
/// turn -- this server's original answer, and still the only one
/// available for a checkpoint whose template never mentions tools.
///
/// `extra` is the request's already-sanitized `chat_template_kwargs`
/// (see [`resolve_template_kwargs`]).
pub(crate) fn prompt_from_messages(
    messages: &[ChatMessage],
    template: &chat_template::PromptTemplate,
    tools: &[ToolDef],
    extra: serde_json::Map<String, serde_json::Value>,
) -> Result<String, ApiError> {
    let rendered = if tools.is_empty() || template.handles_tools() {
        template.render(messages, tools, extra)
    } else {
        let mut with_preamble = Vec::with_capacity(messages.len() + 1);
        with_preamble.push(ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text(tool_preamble(tools))),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
        with_preamble.extend_from_slice(messages);
        template.render(&with_preamble, &[], extra)
    };
    rendered.map_err(template_error_response)
}

/// A template that will not render is a request failure, never a
/// fallback to a guessed one: serving a checkpoint framing it has never
/// seen is the exact bug `chat_template` exists to delete, so the
/// compiler's own message goes back to the caller instead.
fn template_error_response(err: frink_models::chat_template::TemplateError) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": {
                "message": format!("chat template failed to render: {err}"),
                "type": "invalid_request_error",
                "param": "messages",
                "code": null,
            }
        })),
    )
}

/// Real, disclosed approach for tool-calling without grammar-
/// constrained decoding (which doesn't exist in this server):
/// describe each tool in plain text and ask the
/// model to wrap a call in a literal `<tool_call>{...}</tool_call>`
/// marker, then reuse the existing stop-sequence machinery (see
/// `ChatCompletionRequest::effective_stop_sequences`) to end
/// generation right after it, and parse the captured text for that
/// marker afterward (`output::parse_output`, which also accepts the
/// format the served checkpoint's own family emits). This is
/// stop-bounded,
/// prompt-engineered JSON extraction, not enforced-valid-JSON output --
/// a real limitation, not overclaimed.
fn tool_preamble(tools: &[ToolDef]) -> String {
    let mut out = String::from(
        "You can call tools to help answer the user. To call a tool, respond with \
         EXACTLY one line in this format and nothing else:\n\
         <tool_call>{\"name\": \"<tool name>\", \"arguments\": {<arguments as a JSON \
         object matching that tool's parameters>}}</tool_call>\n\n\
         Available tools:\n",
    );
    for t in tools {
        out.push_str(&format!(
            "- {}: {}\n  parameters (JSON schema): {}\n",
            t.function.name,
            t.function.description.as_deref().unwrap_or(""),
            t.function
                .parameters
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "{}".to_string()),
        ));
    }
    out
}

/// Fold one batch of parser events into the text to stream and the
/// tool-call deltas to stream beside it.
///
/// `opened` counts calls that have gone out, which is both the wire
/// `index` and how the terminal chunk knows whether this generation
/// ended in a tool call. `CallEnd` deliberately emits nothing: every
/// byte of the arguments has already gone out as a fragment, and
/// repeating them would make a client that concatenates deltas produce
/// the arguments twice.
fn tool_call_deltas(
    events: Vec<crate::policy::parser::ToolCallEvent>,
    opened: &std::cell::Cell<usize>,
) -> (String, Vec<ToolCallDelta>) {
    let mut text = String::new();
    let mut deltas = Vec::new();
    for event in events {
        match event {
            crate::policy::parser::ToolCallEvent::Text(chunk) => text.push_str(&chunk),
            crate::policy::parser::ToolCallEvent::CallStart { index, name } => {
                opened.set(opened.get().max(index + 1));
                deltas.push(ToolCallDelta::opening(index, name));
            }
            crate::policy::parser::ToolCallEvent::CallArguments { index, fragment } => {
                if !fragment.is_empty() {
                    deltas.push(ToolCallDelta::arguments(index, fragment));
                }
            }
            crate::policy::parser::ToolCallEvent::CallEnd { .. } => {}
        }
    }
    (text, deltas)
}

/// Builds the final response message + finish reason from raw
/// generated text.
///
/// Three things come out of the text: a reasoning block, when the
/// served checkpoint's family emits one; every tool call it made, in
/// whichever format it used; and whatever prose is left. `base_finish`
/// is promoted to `"tool_calls"` only when a call was actually found --
/// a model can answer in plain text despite tools being offered, and
/// that must fall through to an ordinary text response rather than an
/// error.
fn build_response_message(
    text: String,
    tools: &[ToolDef],
    posture: output::OutputPosture,
    base_finish: &'static str,
) -> (ChatCompletionResponseMessage, &'static str) {
    let parsed = output::parse_output(&text, tools, posture);
    let calls: Vec<ToolCallOut> = parsed
        .calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| ToolCallOut {
            id: format!("call_{index}"),
            kind: "function",
            function: ToolCallFunctionOut {
                name: call.name,
                arguments: call.arguments,
            },
        })
        .collect();
    if !calls.is_empty() {
        return (
            ChatCompletionResponseMessage {
                role: "assistant",
                content: None,
                reasoning_content: parsed.reasoning,
                tool_calls: Some(calls),
            },
            "tool_calls",
        );
    }
    (
        ChatCompletionResponseMessage {
            role: "assistant",
            content: Some(parsed.content),
            reasoning_content: parsed.reasoning,
            tool_calls: None,
        },
        base_finish,
    )
}

/// Resolves the full message history a prompt should be rendered
/// from: `req.messages` verbatim when no session is in play, or (see
/// `session` module) `req.messages` appended to `session_id`'s stored
/// history, returning the accumulated whole.
fn resolve_history(state: &AppState, req: &ChatCompletionRequest) -> Vec<ChatMessage> {
    let mut history = match &req.session_id {
        Some(id) => state.sessions.extend_and_get(id, &req.messages),
        None => req.messages.clone(),
    };
    if req.json_object_mode() {
        inject_json_object_system_hint(&mut history);
    }
    history
}

fn inject_json_object_system_hint(messages: &mut Vec<ChatMessage>) {
    const HINT: &str =
        "You must respond with valid JSON only (a single JSON object, no markdown fences).";
    if let Some(sys) = messages.iter_mut().find(|m| m.role == "system") {
        match &mut sys.content {
            Some(MessageContent::Text(s)) if !s.contains("JSON") => {
                s.push_str("\n\n");
                s.push_str(HINT);
            }
            None => {
                sys.content = Some(MessageContent::Text(HINT.to_string()));
            }
            _ => {}
        }
    } else {
        messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: Some(MessageContent::Text(HINT.to_string())),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        );
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let attribution = attribution::Attribution::from_headers(&headers);
    state
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let started = std::time::Instant::now();

    // One id per request, assigned before any work starts -- including
    // before validation -- so the streaming and non-streaming paths
    // agree and a rejected request is still nameable in the monitor.
    let request_id = frink_api::next_request_id();
    let stream = req.stream.unwrap_or(false);

    // The maintenance gate comes before validation: while the cache is
    // being resized or the server is draining, the honest answer is
    // "not now" whichever fields the body carries, and admitting a
    // request into a pool that is being rebuilt under it is worse than
    // refusing one that would have 400'd anyway.
    let refusal = cache_admin::check_admission(&state)
        .err()
        .or_else(|| req.validate_supported_fields().err());
    if let Some(err) = refusal {
        state
            .request_errors_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let response = err.into_response();
        state.record_request(stats::Record {
            request_id: &request_id,
            route: frink_api::routes::V1_CHAT_COMPLETIONS,
            model: state.active_model_name(),
            status: response.status().as_u16(),
            stream,
            duration_ms: started.elapsed().as_millis() as u64,
            usage: None,
            attribution: &attribution,
        });
        return response;
    }

    let response = if stream {
        chat_completions_stream(
            Arc::clone(&state),
            req,
            request_id.clone(),
            started,
            attribution.clone(),
        )
        .await
        .into_response()
    } else {
        chat_completions_full(
            Arc::clone(&state),
            req,
            request_id.clone(),
            started,
            attribution.clone(),
        )
        .await
        .into_response()
    };

    if response.status().is_client_error() || response.status().is_server_error() {
        state
            .request_errors_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Only failures are recorded here. A success has already
        // recorded itself from the path that knows the token counts --
        // and, for a stream, that has not even happened yet.
        state.record_request(stats::Record {
            request_id: &request_id,
            route: frink_api::routes::V1_CHAT_COMPLETIONS,
            // `None` here is the 503 case and says so: nothing was
            // loaded, so nothing served it.
            model: state.active_model_name(),
            status: response.status().as_u16(),
            stream,
            duration_ms: started.elapsed().as_millis() as u64,
            usage: None,
            attribution: &attribution,
        });
    }
    state.mark_request_finished();

    response
}

async fn chat_completions_full(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    request_id: String,
    started: std::time::Instant,
    attribution: attribution::Attribution,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let tools_active = req.tools_active();
    // Cloned once, up front: this request decodes against exactly this
    // model even if `/admin/models/load` swaps a different one in
    // halfway through (see `AppState::active`).
    let active = state.require_active()?;
    let history = resolve_history(&state, &req);
    let template = active.generative()?.chat_template();
    let kwargs = req.resolve_template_kwargs(&template);
    let prompt = req.render_prompt(&history, &template, &req.tools, kwargs, active.name())?;
    // Resolved BEFORE the lookup, because the constraint is part of the
    // key: a grammar, JSON mode and `ignore_eos` all change the answer
    // and none of them changes the prompt, so a cache consulted first
    // would answer a constrained request with an unconstrained
    // completion (#35). It also means an unparseable grammar is a 400
    // for the second caller too, rather than a 200 carrying prose
    // generated under no grammar at all.
    let mut params =
        req.generation_params_for_template(&template, active.name(), active.sampler_model())?;
    params.lora = lora::resolve_request(active.generative()?, req.lora.as_deref())?;
    let key = req.is_cacheable().then(|| req.cache_key(&prompt, &params));

    let (completion, cache_status) = if let Some(cached) = key
        .as_ref()
        .and_then(|key| lock_cache(&state.response_cache).get(key))
    {
        tracing::debug!("cache hit for key {}", key.as_ref().unwrap().digest());
        (cached, "hit")
    } else {
        let (chunks, finish, usage) = decode_task::buffered(
            decode_task::DecodeHandles::take(&state, &active)?,
            prompt.clone(),
            params,
        )
        .await?;

        let completion = response_cache::CachedCompletion {
            content: chunks.concat(),
            finish,
            usage,
        };
        // A cacheable KEY is not on its own permission to store an
        // answer: `cacheable` refuses a generation that did not run to
        // its own end, and is the only way to build the value `put`
        // takes, so a cancelled partial cannot become the cached answer
        // for the next caller (#57).
        let cache_status = match key {
            // Nothing is cloned unless there is a key to store it
            // under: the common path here is a sampled request, which
            // has none.
            Some(key) => match completion.clone().cacheable() {
                Some(cacheable) => {
                    tracing::debug!("cache miss for key {}", key.digest());
                    lock_cache(&state.response_cache).put(key, cacheable);
                    "miss"
                }
                None => "skip",
            },
            None => "skip",
        };
        (completion, cache_status)
    };
    let content = completion.content;

    if req.json_object_mode() {
        json_mode::validate_json_object_output(&content)?;
    }

    // Stored regardless of cache hit/miss, so a session's history is
    // always consistent with what a client would see, whether or not
    // this exact prompt happened to be served from cache.
    if let Some(id) = &req.session_id {
        state.sessions.store_reply(
            id,
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(MessageContent::Text(content.clone())),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        );
    }

    let (message, finish_reason) = build_response_message(
        content,
        if tools_active { &req.tools } else { &[] },
        output::OutputPosture::resolve_full(
            active.reasoning_format(),
            active.tool_call_format(),
            &prompt,
        ),
        completion.finish.as_str(),
    );

    state.record_request(stats::Record {
        request_id: &request_id,
        route: frink_api::routes::V1_CHAT_COMPLETIONS,
        // The handle this request decoded against, not `req.model`: a
        // swap mid-flight does not change which weights answered.
        model: Some(active.name().to_string()),
        status: 200,
        stream: false,
        duration_ms: started.elapsed().as_millis() as u64,
        usage: Some(&completion.usage),
        attribution: &attribution,
    });

    Ok(Json(ChatCompletionResponse {
        id: request_id.clone(),
        request_id,
        object: "chat.completion",
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message,
            finish_reason,
        }],
        usage: completion.usage,
        frink_cache: cache_status,
    }))
}

async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    request_id: String,
    started: std::time::Instant,
    attribution: attribution::Attribution,
) -> Result<Response, ApiError> {
    // Streaming requests are never served from or written to the response cache.
    let tools_active = req.tools_active();
    // See `chat_completions_full`: the handle is taken once and the
    // whole stream runs against it, so a mid-stream model swap cannot
    // splice two checkpoints into one completion.
    let active = state.require_active()?;
    let history = resolve_history(&state, &req);
    let template = active.generative()?.chat_template();
    let kwargs = req.resolve_template_kwargs(&template);
    let prompt = req.render_prompt(&history, &template, &req.tools, kwargs, active.name())?;
    let model_name = req.model.clone();
    let session_id = req.session_id.clone();
    let sessions = state.sessions.clone();

    let model = Arc::clone(active.generative()?);
    let kv_pool = state.kv_pool.clone();
    let paged_kv = state.paged_kv.clone();
    let prefix_cache = state.prefix_cache.clone();
    let batcher = active.batcher.clone();
    let ceiling = active.ceiling.clone();
    let metal_private_decode_gate = state.metal_private_decode_gate.clone();
    let mut params =
        req.generation_params_for_template(&template, active.name(), active.sampler_model())?;
    params.lora = lora::resolve_request(active.generative()?, req.lora.as_deref())?;
    let stats_state = Arc::clone(&state);
    // Read now, off the handle this stream will decode against. Read
    // later it would name whatever a swap had made current by then.
    let served_model = active.name().to_string();
    // How to read this stream, fixed before the first token: the family
    // from the served checkpoint, and whether the prompt that was
    // actually rendered left the model inside a reasoning block.
    let posture = output::OutputPosture::resolve_full(
        active.reasoning_format(),
        active.tool_call_format(),
        &prompt,
    );
    // The offered tools, captured for the terminal parse: the request
    // itself does not outlive the closure that consumes it.
    let offered_tools: Vec<ToolDef> = if tools_active {
        req.tools.clone()
    } else {
        Vec::new()
    };

    // Tier two of cancellation: the id is already on the wire, so the
    // client can name it. The guard rides with the generation task and
    // deregisters however that task ends, panic included -- see the
    // `cancel` module.
    let (cancel_token, cancel_guard) = state.cancels.register(&request_id);
    params.cancel = Some(cancel_token.clone());

    // Tool-call detection needs the full stop-bounded text; continuous
    // batching returns one string. Both stay buffered. Otherwise each
    // decoded chunk is pushed on a channel for overlapped SSE delivery.
    // Incremental streaming, including when tools are offered. It used
    // to be `!tools_active && ...`: finding a tool call needed the
    // whole text. `crate::policy::parser::ToolCallParser` streams prefix-stable
    // argument fragments, so that reason is gone, and a coding agent
    // now watches an argument arrive instead of waiting for it.
    let overlap = true;

    // Opt-in replay. Registering a buffer is also what decides whether a
    // dropped socket cancels this generation -- see `resume`'s module
    // doc for why that is the caller's call and not the server's.
    let slot = req
        .stream_resumable
        .unwrap_or(false)
        .then(|| state.streams.register(&request_id));
    let emitter = resume::Emitter::new(slot);

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    // Built here, where the id and model name are still owned by this
    // frame: the generation task takes both. Serialized once, because
    // it is byte-identical every time it goes out.
    let keepalive = sse::keepalive_event(&ChatCompletionChunk {
        id: request_id.clone(),
        request_id: None,
        object: "chat.completion.chunk",
        model: model_name.clone(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionChunkDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
        usage: None,
    });

    tokio::task::spawn_blocking(move || {
        // Held for the whole generation; dropping it is what takes the
        // id back out of the cancel registry.
        let _cancel_guard = cancel_guard;
        let tx_chunks = tx.clone();
        // The orphan deadline (see `crate::sse`): a client that is
        // neither reading nor disconnected must not park this blocking
        // thread -- and the model handle and cancel guard it holds --
        // for the life of the process.
        let orphan_timeout = sse::orphan_timeout_from_env();
        let mut first = true;
        let head_request_id = request_id.clone();
        // The chain-of-thought split, applied as the tokens arrive
        // rather than at the end. Without this an overlapped stream --
        // which is the default for a reasoning model with no tools --
        // would deliver the whole thinking block as `content` and then
        // the buffered path would deliver the same request's thinking
        // as `reasoning_content`, so the same question would answer
        // differently depending on a transport detail. Shared with the
        // terminal flush below, which releases whatever the parser is
        // still withholding against a marker that never arrived.
        let stream_reasoning: Rc<RefCell<Option<crate::policy::parser::ReasoningParser>>> =
            Rc::new(RefCell::new(posture.reasoning_parser()));
        let emit_reasoning = Rc::clone(&stream_reasoning);
        // The tool-call parser, fed whatever the reasoning parser
        // classified as content. Absent when the request offered no
        // tools, in which case marker-looking text is just text.
        let stream_tools: Rc<RefCell<Option<crate::policy::parser::ToolCallParser>>> = Rc::new(
            RefCell::new(tools_active.then(|| posture.tool_call_parser(&offered_tools))),
        );
        let emit_tools = Rc::clone(&stream_tools);
        // How many calls have been opened on the wire, so the terminal
        // chunk knows whether to say `tool_calls` and does not repeat
        // what already went out.
        let streamed_calls = Rc::new(std::cell::Cell::new(0usize));
        let emit_streamed_calls = Rc::clone(&streamed_calls);
        let result = run_generation_emit(
            &model,
            &prompt,
            &params,
            kv_pool.as_ref(),
            paged_kv.as_ref(),
            prefix_cache.as_deref(),
            batcher.as_ref(),
            ceiling.as_deref(),
            metal_private_decode_gate.as_deref(),
            |chunk| {
                if !overlap || chunk.is_empty() {
                    return;
                }
                let (reasoning, content) = match emit_reasoning.borrow_mut().as_mut() {
                    Some(parser) => {
                        let delta = parser.push(chunk);
                        (delta.reasoning, delta.content)
                    }
                    None => (String::new(), chunk.to_string()),
                };
                // Content goes through the tool parser, which holds
                // back anything that could still become a marker and
                // turns a recognized call into wire deltas.
                let (content, tool_calls) = match emit_tools.borrow_mut().as_mut() {
                    Some(parser) => {
                        let (text, calls) =
                            tool_call_deltas(parser.push(&content), &emit_streamed_calls);
                        (text, calls)
                    }
                    None => (content, Vec::new()),
                };
                // Both parsers withhold partial markers, so a chunk can
                // legitimately produce nothing at all this time round.
                if reasoning.is_empty() && content.is_empty() && tool_calls.is_empty() {
                    return;
                }
                let role = if first { Some("assistant") } else { None };
                let request_id = first.then(|| head_request_id.clone());
                first = false;
                let payload = ChatCompletionChunk {
                    id: head_request_id.clone(),
                    request_id,
                    object: "chat.completion.chunk",
                    model: model_name.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            role,
                            content: (!content.is_empty()).then_some(content),
                            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
                            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                // Tier one of cancellation. A failed send means the SSE
                // receiver is gone -- the browser tab closed, the
                // client aborted, the connection dropped -- and until
                // this was checked the return value was discarded and
                // the decode loop happily generated the remaining
                // hundreds of tokens into nothing. Flipping the same
                // flag `/v1/cancel` sets means there is one stop path,
                // not two.
                if let Err(why) =
                    sse::send_or_orphan(&tx_chunks, Ok(emitter.event(&payload)), orphan_timeout)
                {
                    if why == sse::SendFailure::Orphaned {
                        tracing::warn!(
                            "SSE stream {head_request_id} accepted nothing for the orphan \
                             deadline; treating it as abandoned"
                        );
                    }
                    // Two features met here and only one of them may
                    // win. The orphan deadline exists to stop work
                    // nobody is reading. A resumable stream is exactly
                    // the case where a gone receiver must NOT stop the
                    // work: the client said it may come back, the
                    // buffer is still being filled for it, and
                    // cancelling would make every reconnect resume into
                    // a truncated answer. So the deadline still detects
                    // and logs, and only a non-resumable stream is
                    // cancelled by it. `POST /v1/cancel` is the stop
                    // path for the resumable ones.
                    if !emitter.is_resumable() {
                        cancel_token.cancel();
                    }
                }
            },
        );

        // `first` is still true when nothing was streamed from the emit
        // closure (the buffered tool-call/batching path, or an empty
        // generation), so the id has not gone out yet. `take()` on the
        // way into each payload below guarantees it is announced
        // exactly once, on whichever chunk really is first.
        let mut pending_request_id = first.then(|| request_id.clone());

        match result {
            Ok((finish, usage, full_text)) => {
                if let Some(id) = &session_id {
                    sessions.store_reply(
                        id,
                        ChatMessage {
                            role: "assistant".to_string(),
                            content: Some(MessageContent::Text(full_text.clone())),
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                        },
                    );
                }
                // Both parsers may still be holding a run that could
                // have become a marker and did not. It is ordinary
                // output; dropping it would truncate every answer whose
                // tail happens to look like the start of a `</think>`
                // or a `<tool_call>`.
                let mut streamed_finish: Option<&'static str> = None;
                if overlap {
                    let tail = stream_reasoning
                        .borrow_mut()
                        .as_mut()
                        .map(|parser| parser.flush())
                        .unwrap_or_default();
                    let (mut content, mut tool_calls) = (tail.content, Vec::new());
                    if let Some(parser) = stream_tools.borrow_mut().as_mut() {
                        let mut events = parser.push(&content);
                        events.extend(parser.finish());
                        let (text, calls) = tool_call_deltas(events, &streamed_calls);
                        content = text;
                        tool_calls = calls;
                    }
                    if !content.is_empty() || !tail.reasoning.is_empty() || !tool_calls.is_empty() {
                        let payload = ChatCompletionChunk {
                            id: request_id.clone(),
                            request_id: pending_request_id.take(),
                            object: "chat.completion.chunk",
                            model: model_name.clone(),
                            choices: vec![ChatCompletionChunkChoice {
                                index: 0,
                                delta: ChatCompletionChunkDelta {
                                    role: None,
                                    content: (!content.is_empty()).then_some(content),
                                    reasoning_content: (!tail.reasoning.is_empty())
                                        .then_some(tail.reasoning),
                                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        let _ =
                            sse::send_or_orphan(&tx, Ok(emitter.event(&payload)), orphan_timeout);
                    }
                    if streamed_calls.get() > 0 {
                        streamed_finish = Some("tool_calls");
                    }
                } else {
                    // The batched path had no incremental stream to
                    // ride on, so the whole answer goes out at once.
                    let parsed = output::parse_output(&full_text, &offered_tools, posture);
                    let tool_calls: Vec<ToolCallDelta> = parsed
                        .calls
                        .iter()
                        .enumerate()
                        .map(|(index, call)| {
                            ToolCallDelta::whole(index, call.name.clone(), call.arguments.clone())
                        })
                        .collect();
                    if !tool_calls.is_empty() {
                        streamed_finish = Some("tool_calls");
                    }
                    if !tool_calls.is_empty()
                        || !parsed.content.is_empty()
                        || parsed.reasoning.is_some()
                    {
                        let payload = ChatCompletionChunk {
                            id: request_id.clone(),
                            request_id: pending_request_id.take(),
                            object: "chat.completion.chunk",
                            model: model_name.clone(),
                            choices: vec![ChatCompletionChunkChoice {
                                index: 0,
                                delta: ChatCompletionChunkDelta {
                                    role: Some("assistant"),
                                    content: (!parsed.content.is_empty() && tool_calls.is_empty())
                                        .then(|| parsed.content.clone()),
                                    reasoning_content: parsed.reasoning.clone(),
                                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        let _ =
                            sse::send_or_orphan(&tx, Ok(emitter.event(&payload)), orphan_timeout);
                    }
                }
                // A truncated generation is `length` even if it managed
                // to open a call: the client must not treat a
                // half-written call as one it should execute.
                let final_finish_reason = match streamed_finish {
                    Some(reason) if finish.as_str() != "length" => reason,
                    _ => finish.as_str(),
                };
                let final_payload = ChatCompletionChunk {
                    id: request_id.clone(),
                    request_id: pending_request_id.take(),
                    object: "chat.completion.chunk",
                    model: model_name,
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            role: None,
                            content: None,
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: Some(final_finish_reason),
                    }],
                    usage: Some(usage.clone()),
                };
                let _ = sse::send_or_orphan(&tx, Ok(emitter.event(&final_payload)), orphan_timeout);
                let _ = sse::send_or_orphan(&tx, Ok(emitter.done()), orphan_timeout);
                // Recorded here rather than where the handler returned:
                // the handler returns as soon as the SSE headers go out,
                // which is before a single token exists, so timing it
                // there would report every stream as instant.
                stats_state.record_request(stats::Record {
                    request_id: &request_id,
                    route: frink_api::routes::V1_CHAT_COMPLETIONS,
                    model: Some(served_model.clone()),
                    status: 200,
                    stream: true,
                    duration_ms: started.elapsed().as_millis() as u64,
                    usage: Some(&usage),
                    attribution: &attribution,
                });
            }
            Err(e) => {
                tracing::warn!("decode error on streamed request {request_id}: {e}");
                // The socket carried 200 -- SSE headers precede the
                // first token -- but the request produced no completion.
                // The monitor records outcomes, and a 200 row with zero
                // tokens would read as a successful empty answer, so the
                // failure is stated as 500 here and only here.
                stats_state.record_request(stats::Record {
                    request_id: &request_id,
                    route: frink_api::routes::V1_CHAT_COMPLETIONS,
                    model: Some(served_model.clone()),
                    status: 500,
                    stream: true,
                    duration_ms: started.elapsed().as_millis() as u64,
                    usage: None,
                    attribution: &attribution,
                });
                let payload = ChatCompletionChunk {
                    id: request_id.clone(),
                    request_id: pending_request_id.take(),
                    object: "chat.completion.chunk",
                    model: model_name,
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            role: Some("assistant"),
                            content: Some(format!("[error: {e}]")),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: Some("stop"),
                    }],
                    usage: None,
                };
                let _ = sse::send_or_orphan(&tx, Ok(emitter.event(&payload)), orphan_timeout);
                let _ = sse::send_or_orphan(&tx, Ok(emitter.done()), orphan_timeout);
            }
        }
        // The buffer is closed by dropping `emitter` here -- including
        // on a panic, which is the case an explicit call would miss.
        // See `resume::Emitter`'s `Drop`.
        drop(emitter);
    });

    let stream = sse::with_keepalive(rx, keepalive, sse::KEEPALIVE_INTERVAL);
    // `X-Accel-Buffering: no` is the one header that actually reaches
    // the problem the plan names: nginx (and the proxies that copied
    // its convention) buffer `text/event-stream` by default, which
    // turns a token-by-token stream into one silent wait followed by
    // the whole answer at once -- indistinguishable, from the browser,
    // from a hung backend. axum already sets `Cache-Control: no-cache`
    // on an `Sse` response, so that half is covered.
    //
    // The keepalive every 15s is the other half: it gives an
    // idle-but-healthy stream something to send, so a client's stall
    // timeout measures the *connection* rather than the model's
    // time-to-first-token on a long prompt.
    //
    // **Not `Sse::keep_alive`.** axum's keepalive is an SSE COMMENT,
    // and a comment does not reach a client's event handler -- codex's
    // 300s stream-idle timeout only resets on a data frame, so a
    // comment-kept stream is reconnected mid-answer on a long prefill.
    // `sse::with_keepalive` sends a real `chat.completion.chunk` with
    // an empty delta instead: a concatenating client adds nothing, and
    // the transport sees traffic. It also covers the silence BEFORE
    // the first token, which is exactly the queue-wait and long-prefill
    // window where this matters most.
    Ok((
        [(
            axum::http::HeaderName::from_static("x-accel-buffering"),
            axum::http::HeaderValue::from_static("no"),
        )],
        Sse::new(stream),
    )
        .into_response())
}

/// The axum pattern for one of the published path templates.
///
/// `frink_api::routes` writes placeholders in the OpenAPI style
/// because it is imported by clients that have never heard of this
/// server's router; axum 0.7 wants `:name`. Converting here keeps one
/// published spelling and one router spelling, and the test below fails
/// if they ever stop describing the same path.
///
/// This rewrites EVERY `{name}` it finds rather than one known
/// placeholder. The narrow version took `{request_id}` only, so the two
/// Responses templates were mounted with their braces intact and axum
/// read `{response_id}` as a literal segment: `GET /v1/responses/abc`
/// matched no route and got axum's bodiless 404 instead of the
/// handler's, and the one path that did match would have panicked on
/// `MissingPathParams`. Anything with a placeholder must go through
/// here.
/// Every route that sits behind `FRINK_API_KEY`, as ONE list.
///
/// Extracted because there were two of these: this one and a
/// hand-written copy in the test module, which had already drifted --
/// the test router was missing `/metrics`, `/cache/stats`, both rerank
/// spellings and half of `/admin`, so an HTTP test could pass against a
/// route the real server does not serve, or 404 on one it does. That is
/// this repo's dominant bug shape (two structures that must agree, with
/// nothing enforcing it) sitting inside the test harness, where it is
/// worst: it makes the tests agree with themselves.
///
/// `/health` is deliberately NOT here. It is the one route that must
/// stay reachable without a key, and it is registered separately for
/// that reason.
fn protected_routes() -> Router<Arc<AppState>> {
    use frink_api::routes;

    Router::new()
        .route(routes::V1_MODELS, get(list_models))
        // The Responses surface decodes tokens, so it sits behind the
        // same key as `/v1/chat/completions`: it must cost what
        // decoding tokens costs.
        .route(routes::V1_RESPONSES, post(responses::responses))
        .route(
            &axum_path(routes::V1_RESPONSE),
            get(responses::responses_get),
        )
        .route(
            &axum_path(routes::V1_RESPONSE_CANCEL),
            post(responses::responses_cancel),
        )
        .route(&axum_path(routes::SLOTS_ID), post(slots::post_slot))
        .route(routes::V1_STATS, get(serving_stats))
        .route(routes::V1_REQUESTS, get(recent_requests))
        .route(routes::V1_CACHE_STATUS, get(cache_admin::cache_status))
        .route(routes::V1_CACHE_REBUILD, post(cache_admin::cache_rebuild))
        .route(routes::ADMIN_PREPARE_STOP, post(cache_admin::prepare_stop))
        .route(
            routes::LORA_ADAPTERS,
            get(lora::get_lora_adapters).post(lora::post_lora_adapters),
        )
        .route(routes::V1_CHAT_COMPLETIONS, post(chat_completions))
        // Behind the same key as the endpoint that started the work:
        // an unauthenticated caller must not be able to stop someone
        // else's generation by guessing at request ids.
        .route(routes::V1_CANCEL, post(cancel_generation))
        // Reconnect and the polling fallback, both behind the same key
        // as the request that filled the buffer: the replay window holds
        // the model's output, so reading it must cost what producing it
        // cost.
        .route(&axum_path(routes::V1_STREAM), get(resume::resume))
        .route(&axum_path(routes::V1_STREAM_POLL), get(resume::poll))
        .route(routes::V1_MESSAGES, post(anthropic::messages))
        .route(
            routes::V1_MESSAGES_COUNT_TOKENS,
            post(anthropic::count_tokens),
        )
        .route(routes::V1_COMPLETIONS, post(openai_extra::completions))
        // llama.cpp's NATIVE completion endpoint, under both spellings
        // it mounts. Not an alias of the line above: different request
        // fields, a different response object, and a stream that ends
        // without `[DONE]`. See `crate::completion`.
        .route(routes::COMPLETION, post(completion::completion))
        .route(routes::COMPLETIONS, post(completion::completion))
        .route(routes::V1_TOKENIZE, post(openai_extra::tokenize))
        .route(routes::V1_DETOKENIZE, post(openai_extra::detokenize))
        // llama.cpp's unprefixed spelling of the same two, on the SAME
        // handlers -- not copies. The `/v1/` prefix was frink's
        // invention (OpenAI has no tokenize endpoint), so every
        // llama.cpp client was getting a 404 that named nothing. Behind
        // the key with their twins: they read the loaded vocabulary.
        .route(routes::TOKENIZE, post(openai_extra::tokenize))
        .route(routes::DETOKENIZE, post(openai_extra::detokenize))
        .route(routes::V1_EMBEDDINGS, post(embeddings::embeddings))
        // Cross-encoder reranking, under the `/v1` spelling Cohere and
        // Jina clients use and the unprefixed one llama.cpp mounts.
        // Same handler: this really is an alias, not a second dialect.
        .route(routes::V1_RERANK, post(rerank::rerank))
        .route(routes::RERANK, post(rerank::rerank))
        .route(routes::CACHE_STATS, get(cache_stats))
        .route(routes::METRICS, get(metrics))
        // The control surface. Registered inside `protected` on
        // purpose: these routes change what the server serves and write
        // to disk, so they get the same FRINK_API_KEY gate as /v1/*
        // and never the unauthenticated treatment /health has.
        .route(routes::ADMIN_MODELS, get(admin::models))
        .route(routes::ADMIN_MODELS_LOAD, post(admin::load_model))
        .route(routes::ADMIN_MODELS_UNLOAD, post(admin::unload_model))
        .route(routes::ADMIN_DOWNLOAD, post(admin::download))
        .route(routes::ADMIN_TASKS, get(admin::tasks))
        .route(&admin::cancel_route(), post(admin::cancel_task))
        .route(routes::ADMIN_STATS, get(admin::stats))
        // Server-side conversation storage, mounted here so it inherits
        // the same key gate as the endpoint that generated the text it
        // stores. Routes and store both live in `conversations`.
        .merge(conversations::router())
}

fn axum_path(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        let Some(close) = rest[open..].find('}').map(|c| open + c) else {
            break;
        };
        out.push_str(&rest[..open]);
        out.push(':');
        out.push_str(&rest[open + 1..close]);
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    out
}

/// `POST /v1/cancel` -- the explicit half of two-tier cancellation.
///
/// Answers `200` when a live generation was signalled and `404` when
/// the id names nothing that is running. That difference is the whole
/// point of the endpoint returning a body at all: "already finished"
/// and "stopped it" are both fine outcomes, but only one of them saved
/// any work, and a UI told `ok: true` for both will claim it stopped
/// something it did not.
async fn cancel_generation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<frink_api::CancelGenerationRequest>,
) -> Response {
    let cancelled = state.cancels.cancel(&req.request_id);
    let status = if cancelled {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    };
    let detail = if cancelled {
        "the generation was asked to stop; it ends at its next token".to_string()
    } else {
        "no generation with that request_id is running -- it has already \
         finished, was never issued, or was served by a path that does \
         not register for cancellation"
            .to_string()
    };
    (
        status,
        Json(frink_api::CancelGenerationResponse {
            request_id: req.request_id,
            cancelled,
            detail,
        }),
    )
        .into_response()
}

/// What a freshly loaded checkpoint becomes when it is published as the
/// active model: the model itself, its optional continuous-batching
/// worker, and the context ceiling both decode paths admit on.
type Activated = (
    Loaded,
    Option<serving::batch::ContinuousBatcher>,
    Option<Arc<budget::ContextCeiling>>,
);

/// The scheduler config for a freshly loaded GGUF, with the ceilings an
/// operator did not configure *derived* from the checkpoint instead of
/// left absent.
///
/// This is the server half of `mem-preload-kv-budget`: `frink run`
/// already priced weights + `n_ctx * per_token_kv` + headroom against
/// the device budget before loading, while `frink-server` admitted on
/// whatever `FRINK_CB_*` happened to be set and otherwise on nothing.
///
/// Precedence is one-directional and deliberate: an explicit
/// `FRINK_CB_MAX_CONTEXT` / `FRINK_CB_KV_BLOCKS` is never overridden,
/// because an operator who names a number has information this
/// arithmetic does not. Derivation only ever fills an *absent* ceiling,
/// where the alternative is no ceiling at all.
///
/// `path` is `None` for the synthetic-weights fallback, which has no
/// checkpoint on disk to price.
fn price_batcher_config(path: Option<&str>) -> serving::batch::BatcherConfig {
    let mut batcher = serving::batch::BatcherConfig::from_env();
    if batcher.max_context.is_some() && batcher.kv_blocks.is_some() {
        // Nothing left to derive, and pricing the checkpoint would only
        // print arithmetic that decides nothing.
        return batcher;
    }
    let Some(path) = path else {
        return batcher;
    };
    // `frink_core::cache::KvCache` is `Vec<f32>` on both decode paths,
    // so f32 is the width really kept, even under Metal attention where
    // the *device* also holds an f16 copy. Budgeting the host store is
    // the conservative reading: it over-charges KV and therefore
    // under-states the context that fits.
    let priced = budget::price_gguf(path, frink_models::KvElem::F32, 1);
    let Some((priced, gguf_ctx, source)) = priced else {
        return batcher;
    };
    let Some(derived) = budget::derive_limits(&priced, gguf_ctx, batcher.kv_block_size) else {
        // See `budget`'s module doc: a fit of zero tokens is not a
        // ceiling of zero, it is an estimate saying this model should
        // not have loaded -- and it did. Say so and admit as before.
        tracing::warn!(
            "this checkpoint's weights leave no room for KV inside the {source}: {} weight \
             bytes against a {} byte budget. Serving with no derived context ceiling -- set \
             FRINK_DEVICE_BUDGET_BYTES if the probe is wrong, or FRINK_CB_MAX_CONTEXT to \
             admit on a number you choose.",
            priced.weights_bytes,
            priced.device_budget_bytes,
        );
        return batcher;
    };
    tracing::info!("{source}");
    tracing::info!("{}", derived.fit);
    let adopted = budget::apply_derived(&mut batcher, &derived);
    if adopted.max_context {
        tracing::info!(
            "derived per-request context ceiling: {} token positions (prompt + max_tokens); \
             override with FRINK_CB_MAX_CONTEXT",
            derived.max_context
        );
    }
    if adopted.kv_blocks {
        tracing::info!(
            "derived KV block budget: {} blocks x {} positions; override with FRINK_CB_KV_BLOCKS",
            derived.kv_blocks,
            batcher.kv_block_size
        );
    }
    if let Some(narrowed) = adopted.max_context_narrowed {
        tracing::info!(
            "per-request context ceiling narrowed to {narrowed} token positions: the whole KV              ledger is {} blocks x {} positions, so a longer request could never be admitted",
            batcher.kv_blocks.unwrap_or_default(),
            batcher.kv_block_size
        );
    }
    batcher
}

/// Turns a freshly loaded checkpoint into the parts that get published
/// as the active model.
///
/// Extracted from `build_app_state` so `/admin/models/load` builds its
/// replacement exactly the way startup builds the first one -- a second
/// copy of this match would be a second place for a new engine variant
/// to be forgotten, and the difference would only show up as a model
/// that silently loses continuous batching after a swap.
pub(crate) fn activate_loaded_model(
    loaded: model::LoadedModel,
    enable_continuous_batching: bool,
    path: Option<&str>,
    paged_kv: Option<&generate::PagedKvConfig>,
) -> Activated {
    match loaded {
        model::LoadedModel::Gguf(g) => {
            let decoder = Arc::new(g.decoder);
            let tokenizer = Arc::new(g.tokenizer);
            let config = price_batcher_config(path);
            // Prefill is still a per-token `forward_token` loop on both
            // paths (see `sched-chunked-prefill`: chunking bought
            // fairness, not a batched prefill kernel), so a sliding
            // layer really does need only `window + 1 - 1` positions
            // live. `chunk = 1` here is the truth, not a simplification.
            let shape =
                frink_models::KvShape::from_config(&decoder.config, frink_models::KvElem::F32);
            let ceiling = Arc::new(budget::ContextCeiling::new(config.max_context, shape));
            let batcher = if enable_continuous_batching {
                tracing::info!(
                    "continuous batching enabled: decode steps share Decoder::forward_multi_seq \
                     (stop sequences use the same pending-buffer trim as the private generate loop)"
                );
                let tok = Arc::clone(&tokenizer);
                let decode = Arc::new(move |ids: &[usize]| tok.decode_bytes(ids));
                Some(serving::batch::ContinuousBatcher::spawn_with_ceiling(
                    Arc::clone(&decoder),
                    decode,
                    config,
                    Arc::clone(&ceiling),
                    paged_kv.cloned(),
                ))
            } else {
                None
            };
            (
                Loaded::Generative(Arc::new(Model::Gguf(GgufModel {
                    decoder,
                    tokenizer,
                    stop_tokens: g.stop_tokens,
                    bos_id: g.bos_id,
                    is_synthetic: g.is_synthetic,
                    chat_template: g.chat_template,
                }))),
                batcher,
                Some(ceiling),
            )
        }
        model::LoadedModel::Kimi(k) => (
            Loaded::Generative(Arc::new(Model::Kimi(KimiModel {
                engine: k.engine,
                tokenizer: k.tokenizer,
                stop_tokens: k.stop_tokens,
                chat_template: k.chat_template,
            }))),
            None,
            None,
        ),
        model::LoadedModel::Mla(m) => (
            Loaded::Generative(Arc::new(Model::Mla(MlaModel {
                engine: m.engine,
                tokenizer: m.tokenizer,
                stop_tokens: m.stop_tokens,
                bos_id: m.bos_id,
                name: m.name,
                chat_template: m.chat_template,
            }))),
            None,
            None,
        ),
        model::LoadedModel::Gemma4(m) => (
            Loaded::Generative(Arc::new(Model::Gemma4(Gemma4Model {
                engine: m.engine,
                tokenizer: m.tokenizer,
                stop_tokens: m.stop_tokens,
                bos_id: m.bos_id,
                name: m.name,
                chat_template: m.chat_template,
            }))),
            None,
            None,
        ),
        model::LoadedModel::Glm52(g) => (
            Loaded::Generative(Arc::new(Model::Glm52(Glm52Model {
                engine: g.engine,
                tokenizer: g.tokenizer,
                stop_tokens: g.stop_tokens,
                bos_id: g.bos_id,
                name: g.name,
                chat_template: g.chat_template,
            }))),
            None,
            None,
        ),
        // No batcher and no ceiling, and neither is an omission: an
        // encoder has no decode step to share between requests and no
        // KV cache to price a context against. Handing it either would
        // be pricing a cost it does not have.
        model::LoadedModel::Encoder(e) => (Loaded::Encoder(e), None, None),
    }
}

/// The models a server starts with: the generation model, and the
/// embedding model when `FRINK_EMBEDDING_MODEL_PATH` names one.
///
/// One struct rather than two parameters because they are chosen
/// together at startup and are the only two things `build_app_state`
/// takes that are a *model*.
struct StartupModels {
    loaded: model::LoadedModel,
    embedding: Option<Arc<frink_models::EmbeddingModel>>,
}

fn continuous_batching_env() -> Option<bool> {
    match std::env::var("FRINK_CONTINUOUS_BATCHING")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        None => None,
        Some("1" | "true" | "yes" | "on") => Some(true),
        Some("0" | "false" | "no" | "off") => Some(false),
        _ => None,
    }
}

fn metal_private_decode_active() -> bool {
    #[cfg(feature = "metal")]
    {
        BUILT_WITH_METAL
            && frink_metal::attn::metal_attn_enabled()
            && std::env::var("FRINK_METAL").ok().as_deref() != Some("0")
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

fn continuous_batching_compatible(
    loaded: &model::LoadedModel,
    kv_pool: &Option<generate::KvPoolConfig>,
    prefix_cache: &Option<Arc<Mutex<PrefixCache>>>,
    paged_kv: &Option<generate::PagedKvConfig>,
) -> bool {
    matches!(loaded, model::LoadedModel::Gguf(_))
        && (paged_kv.is_some() || (kv_pool.is_none() && prefix_cache.is_none()))
}

fn resolve_continuous_batching_enabled(
    loaded: &model::LoadedModel,
    kv_pool: &Option<generate::KvPoolConfig>,
    prefix_cache: &Option<Arc<Mutex<PrefixCache>>>,
    paged_kv: &Option<generate::PagedKvConfig>,
) -> bool {
    if !continuous_batching_compatible(loaded, kv_pool, prefix_cache, paged_kv) {
        return false;
    }
    match continuous_batching_env() {
        Some(true) => true,
        Some(false) => false,
        None => metal_private_decode_active(),
    }
}

fn acquire_metal_private_decode_gate(
    gate: Option<&std::sync::Mutex<()>>,
    used_batcher: bool,
) -> Option<std::sync::MutexGuard<'_, ()>> {
    if used_batcher {
        None
    } else {
        gate.map(|g| g.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

fn build_app_state(
    models: StartupModels,
    kv_pool: Option<generate::KvPoolConfig>,
    paged_kv: Option<generate::PagedKvConfig>,
    prefix_cache: Option<Arc<Mutex<PrefixCache>>>,
    enable_continuous_batching: bool,
    mcp: Option<mcp::LoadedMcpConfig>,
    detection: Arc<health::Detection>,
) -> AppState {
    let StartupModels { loaded, embedding } = models;
    let configured_path = std::env::var("FRINK_MODEL_PATH").ok();
    let (loaded, batcher, ceiling) = activate_loaded_model(
        loaded,
        enable_continuous_batching,
        configured_path.as_deref(),
        paged_kv.as_ref(),
    );
    // The startup model's admin id is whichever discovered entry sits
    // at the configured path; `None` when it was not discovered (the
    // synthetic fallback, or a path outside the scanned directories),
    // in which case `/admin/models` reports nothing as active rather
    // than inventing an id no `load` request could name.
    let id = startup_model_id();
    let metal_private_decode_gate = if enable_continuous_batching || !metal_private_decode_active()
    {
        None
    } else {
        tracing::info!(
            "Metal private-loop decode will serialize concurrent requests until \
             continuous batching is enabled (FRINK_CONTINUOUS_BATCHING=1 or --cont-batching)"
        );
        Some(Arc::new(std::sync::Mutex::new(())))
    };
    AppState {
        embedding,
        active: std::sync::RwLock::new(Some(Arc::new(ActiveModel {
            id,
            loaded,
            batcher,
            ceiling,
            checkpoint_path: configured_path.as_deref().map(PathBuf::from),
        }))),
        paged_kv,
        load_in_progress: std::sync::atomic::AtomicBool::new(false),
        tasks: Arc::new(tasks::TaskRegistry::new()),
        cancels: Arc::new(cancel::CancelRegistry::new()),
        stats: stats::Stats::new(),
        streams: resume::StreamRegistry::new(),
        model_dir: admin::model_dirs().into_iter().next(),
        response_cache: Mutex::new(ResponseCache::new(1000, Duration::from_secs(3600))),
        kv_pool,
        prefix_cache,
        sessions: session::SessionStore::new(),
        requests_total: std::sync::atomic::AtomicU64::new(0),
        request_errors_total: std::sync::atomic::AtomicU64::new(0),
        started_at: std::time::Instant::now(),
        last_request_ms: std::sync::atomic::AtomicU64::new(0),
        detection,
        mcp,
        continuous_batching_enabled: enable_continuous_batching,
        metal_private_decode_gate,
        loading_model: Mutex::new(None),
        last_load_error: Mutex::new(None),
        serving: Mutex::new(crate::stats::ServingStats::default()),
        maintenance: Mutex::new(crate::policy::maintenance::MaintenanceGate::serving()),
        footprint: Mutex::new(crate::policy::footprint::ProbeCache::new(FOOTPRINT_TTL_MS)),
        started_unix: unix_now(),
    }
}

/// Builds the `/v1/embeddings` encoder from
/// `FRINK_EMBEDDING_MODEL_PATH`, or `None` when the variable is unset.
///
/// A failure here is fatal rather than deferred: a server that starts
/// with a misspelt path and then answers embedding requests out of the
/// *decoder* would be handing back vectors from the wrong model with
/// nothing in the response saying so.
fn load_embedding_model() -> anyhow::Result<Option<Arc<frink_models::EmbeddingModel>>> {
    let Ok(path) = std::env::var("FRINK_EMBEDDING_MODEL_PATH") else {
        return Ok(None);
    };
    let model = frink_models::EmbeddingModel::from_gguf_path(&path)
        .map_err(|e| anyhow::anyhow!("FRINK_EMBEDDING_MODEL_PATH={path}: {e}"))?;
    tracing::info!(
        "loaded embedding model '{}' ({}, {} dims, pooling {}, max {} tokens)",
        model.name(),
        model.architecture(),
        model.n_embd(),
        model.pooling_type().name(),
        model.n_ctx_train(),
    );
    Ok(Some(Arc::new(model)))
}

/// Seconds since the epoch, or zero on a machine whose clock is set
/// before it. Only ever used to make an id distinct between process
/// generations, so a nonsense clock costs distinctness and nothing
/// else.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The `/admin/models` id of the checkpoint `FRINK_MODEL_PATH` names,
/// when discovery finds it. Matching on the resolved path rather than
/// on the filename keeps two same-named files in different directories
/// from claiming each other's id.
fn startup_model_id() -> Option<String> {
    let configured = std::env::var("FRINK_MODEL_PATH").ok()?;
    let configured = std::fs::canonicalize(&configured).ok()?;
    admin::discover(&admin::model_dirs())
        .into_iter()
        .find(|d| {
            std::fs::canonicalize(&d.path)
                .map(|p| p == configured)
                .unwrap_or(false)
        })
        .map(|d| d.id)
}

/// Builds the global rayon pool up front, on the main thread, with an
/// explicit width and QoS (see [`frink_core::threads`]).
///
/// Doing this from `main` rather than letting rayon build lazily is the
/// point: the first rayon call inside this server happens on a Tokio
/// `spawn_blocking` thread, so the workers used to inherit that thread's
/// QoS class -- which on macOS decides whether they land on performance
/// or efficiency cores.
fn init_cpu_pool() {
    match frink_core::threads::init_cpu_pool() {
        Some(n) => eprintln!(
            "frink-server: rayon pool {n} threads (perf cores {}; override with FRINK_CPU_THREADS)",
            frink_core::threads::perf_core_count()
        ),
        None => eprintln!("frink-server: global rayon pool already built; leaving it alone"),
    }
}

/// Prints the machine-readable ready line (see `frink_api::lifecycle`)
/// on stdout and flushes it.
///
/// This one line is what makes `--port 0` usable, and it deletes a whole
/// feature from any supervising process: no "is the port free" probe, no
/// `lsof` to work out whether an existing listener is a stale copy of
/// ourselves or a stranger's server, no dialog to explain the result.
/// The kernel picks the port and the child says what it got.
///
/// Shares stdout with the tracing subscriber on purpose -- a parent
/// reads stdout line by line and ignores anything that is not the ready
/// event, which `ServerReady::from_line` does for it.
fn announce_ready(addr: SocketAddr, scheme: &str) {
    use std::io::Write;
    let ready =
        frink_api::ServerReady::new(addr, scheme, env!("CARGO_PKG_VERSION"), std::process::id());
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{}", ready.to_line());
    let _ = stdout.flush();
}

/// Resolves when the server should stop serving.
///
/// Stdin-close is the one orphan-prevention mechanism that behaves
/// identically on macOS, Windows and Linux and survives a parent that
/// dies rather than exiting cleanly: the kernel closes the pipe either
/// way. The POSIX alternative -- a signal handler plus an exit hook plus
/// a reaper -- has no Windows equivalent at all, since there is no
/// SIGTERM there.
///
/// When disabled this future never resolves, which is exactly the
/// previous behaviour: serve until the process is stopped externally.
async fn shutdown_signal(exit_on_stdin_close: bool) {
    if !exit_on_stdin_close {
        std::future::pending::<()>().await;
        return;
    }
    let _ = tokio::task::spawn_blocking(|| {
        use std::io::Read;
        let mut sink = [0u8; 256];
        let mut stdin = std::io::stdin().lock();
        loop {
            match stdin.read(&mut sink) {
                // EOF: the parent is gone, or closed the pipe.
                Ok(0) => break,
                // Input on stdin is not a protocol here; drain it.
                Ok(_) => continue,
                Err(e) => {
                    tracing::warn!("stdin read failed ({e}); treating it as closed");
                    break;
                }
            }
        }
    })
    .await;
    tracing::info!("stdin closed; shutting down");
}

/// Tokio worker threads. The default is one per logical core, which on a
/// 10-core M2 Pro means 10 async workers oversubscribing the same cores
/// the rayon decode pool needs. Serving work here is almost entirely I/O
/// plus `spawn_blocking` handoff, so a small fixed pool is enough.
fn tokio_worker_threads() -> usize {
    std::env::var("FRINK_TOKIO_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(2)
}

/// Parses llama-server-style options and applies their environment
/// overrides before creating Tokio or Rayon worker threads. It then
/// brackets the async server lifecycle with journal records.
/// Install rustls' `ring` crypto provider as the process default.
///
/// `axum-server` is built with `tls-rustls-no-provider`, which
/// deliberately does NOT pick a backend -- see the comment on the
/// dependency in `Cargo.toml`. rustls then has no default provider, and
/// building a `ServerConfig` without one fails at ACCEPT time rather
/// than at compile time, which is the worst place for it to surface: a
/// server that started cleanly and refuses every TLS connection.
///
/// So this runs unconditionally at startup, not lazily in the TLS arm.
/// `install_default` returns `Err` if a provider is already installed,
/// which is not a failure -- it means something else got there first
/// and the invariant we care about (there IS a provider) already holds.
fn install_ring_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Runs the server to completion.
///
/// Takes already-parsed arguments so the same library backs both the
/// `frink-server` binary and frink-cli's optional `serve` feature,
/// and neither front end can drift into its own startup logic.
pub fn run_server(args: ServerArgs) -> anyhow::Result<()> {
    if args.list_devices {
        frink_models::devices::print_available_devices();
        return Ok(());
    }
    apply_cli_overrides(&args)?;

    // Before the model is loaded and before the port is bound: refuse
    // to be the second process holding weights on this host. Held for
    // the life of the process -- dropping it deregisters us.
    let _instance = {
        use frink_core::instance::{register, InstancePolicy};
        let policy = if args.allow_multiple_instances {
            InstancePolicy::Multi
        } else {
            InstancePolicy::from_env_or(InstancePolicy::Single)
        };
        let model = std::env::var("FRINK_MODEL_PATH").ok();
        register(
            "server",
            model.as_deref(),
            frink_core::instance::current_backend(),
            policy,
        )
        .map_err(|conflict| anyhow::anyhow!("{conflict}"))?
    };

    let journal = journal::Journal::from_env();
    eprintln!(
        "frink-server: process lifecycle journal at {:?} (override with FRINK_JOURNAL_PATH)",
        journal.path()
    );
    journal.append(&journal::Record::session_start(
        env!("CARGO_PKG_VERSION"),
        std::process::id(),
    ));
    journal::install_panic_hook(journal.clone());

    let mcp_config_path = args.mcp_config.clone();
    let exit_on_stdin_close = args.exit_on_stdin_close
        || std::env::var("FRINK_EXIT_ON_STDIN_CLOSE")
            .map(|v| v == "1")
            .unwrap_or(false);

    // Before Tokio exists, so the decode pool's threads are not spawned
    // from (and do not inherit the QoS of) a blocking-pool thread.
    // SAFETY: still single-threaded here.
    unsafe { frink_core::weight_matrix::default_cpu_int_dot_on() };
    init_cpu_pool();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(tokio_worker_threads())
        .enable_all()
        .build()?;
    let result = runtime.block_on(run(mcp_config_path, exit_on_stdin_close));

    let reason = match &result {
        Ok(()) => "normal".to_string(),
        Err(e) => e.to_string(),
    };
    journal.append(&journal::Record::session_exit(reason));

    // Dropping the runtime instead would wait for blocking tasks, and
    // the stdin watcher parks in a blocking read that may never return
    // (a terminal keeps stdin open forever). The serving future has
    // already finished by here, so nothing useful is being abandoned.
    runtime.shutdown_background();

    result
}

async fn run(mcp_config_path: Option<PathBuf>, exit_on_stdin_close: bool) -> anyhow::Result<()> {
    // `try_init`, not `init`. As a library this runs inside a process
    // that may already have a subscriber: frink-cli installs one
    // before it dispatches, so `frink serve` would panic on startup
    // with "a global default trace dispatcher has already been set".
    // Losing the race is not an error, it means logging is configured.
    let _ = tracing_subscriber::fmt::try_init();

    // Fail-closed listener check, before anything else (including
    // loading the model, so a misconfigured bind fails fast rather than
    // after however long that takes): refuse to start bound to a
    // non-loopback address with no API key configured, unless the
    // operator has explicitly opted into that via
    // FRINK_ALLOW_UNAUTHENTICATED_REMOTE=1 -- see
    // `security::check_bind_authorization`'s doc comment for why an
    // address that doesn't even parse as loopback is treated the same
    // as a confirmed non-loopback one.
    let addr = std::env::var("FRINK_ADDR").unwrap_or_else(|_| "127.0.0.1:8383".to_string());
    let api_key_configured = std::env::var("FRINK_API_KEY").is_ok();
    let allow_unauthenticated_remote = std::env::var("FRINK_ALLOW_UNAUTHENTICATED_REMOTE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if let Err(msg) =
        security::check_bind_authorization(&addr, api_key_configured, allow_unauthenticated_remote)
    {
        anyhow::bail!(msg);
    }

    // Loaded before the generation model, so a bad path fails the
    // start rather than the first `/v1/embeddings` request. This is the
    // SIDE-CAR: a second checkpoint beside a generative one. An encoder
    // at `FRINK_MODEL_PATH` needs none of this -- it goes through
    // `model::load()` below like any other checkpoint and becomes the
    // active model.
    let embedding_model = load_embedding_model()?;

    let mut loaded = model::load()?;
    match &loaded {
        model::LoadedModel::Gguf(g) => tracing::info!(
            "loaded GGUF model '{}' (synthetic={}, tokenizer={})",
            g.decoder.config.name,
            g.is_synthetic,
            g.tokenizer.kind()
        ),
        model::LoadedModel::Kimi(k) => tracing::info!(
            "loaded Kimi K3 checkpoint (tokenizer={} base tokens)",
            k.tokenizer.vocab_size()
        ),
        model::LoadedModel::Mla(m) => tracing::info!(
            "loaded MLA GGUF '{}' (tokenizer={})",
            m.name,
            m.tokenizer.kind()
        ),
        model::LoadedModel::Gemma4(m) => tracing::info!(
            "loaded Gemma4 GGUF '{}' (tokenizer={})",
            m.name,
            m.tokenizer.kind()
        ),
        model::LoadedModel::Glm52(g) => tracing::info!(
            "loaded GLM-5.2 GGUF '{}' (tokenizer={})",
            g.name,
            g.tokenizer.kind()
        ),
        // `model::load_encoder_checkpoint` has already logged the
        // dimensions, the pooling rule and which endpoint serves it.
        model::LoadedModel::Encoder(_) => {}
    }
    // Opt-in VRAM budget for GPU-resident MoE experts. When unset but
    // Metal is active, default to a large budget so routed experts that
    // have Metal-capable quants run via `run_expert_placed` (Metal
    // matvec) instead of staying on CPU after Metal attention. Explicit
    // `FRINK_GPU_VRAM_BUDGET_BYTES=0` keeps the historical all-CPU MoE
    // placement. CUDA builds still require an explicit budget (Vast /
    // multi-GPU hosts vary too much for a safe default).
    let metal_default_moe_budget = {
        #[cfg(feature = "metal")]
        {
            frink_core::metal_dense_enabled()
                && std::env::var("FRINK_GPU_VRAM_BUDGET_BYTES").is_err()
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    };
    if let Ok(budget_str) = std::env::var("FRINK_GPU_VRAM_BUDGET_BYTES") {
        let budget: u64 = budget_str
            .parse()
            .expect("FRINK_GPU_VRAM_BUDGET_BYTES must be a non-negative integer");
        match &mut loaded {
            model::LoadedModel::Gguf(g) => {
                tracing::info!(
                    "GPU expert placement enabled: {budget} byte VRAM budget for routed experts \
                     (CUDA and/or Metal matvecs when built with the matching feature)"
                );
                g.decoder.gpu_vram_budget_bytes = Some(budget);
            }
            model::LoadedModel::Kimi(_) => {
                tracing::warn!(
                    "FRINK_GPU_VRAM_BUDGET_BYTES is set but the loaded model is Kimi K3 -- not \
                     supported yet (its MoE stack isn't wired to PlacementPlan), ignoring"
                );
            }
            model::LoadedModel::Mla(_) => {
                tracing::warn!(
                    "FRINK_GPU_VRAM_BUDGET_BYTES is set but the loaded model is MLA -- dense \
                     FFN path only today; ignoring expert VRAM budget"
                );
            }
            model::LoadedModel::Gemma4(_) => {
                tracing::warn!(
                    "FRINK_GPU_VRAM_BUDGET_BYTES is set but the loaded model is Gemma4 -- \
                     ignoring expert VRAM budget"
                );
            }
            model::LoadedModel::Glm52(_) => {
                tracing::warn!(
                    "FRINK_GPU_VRAM_BUDGET_BYTES is set but the loaded model is GLM-5.2 DSA -- \
                     GPU expert placement not wired yet; ignoring"
                );
            }
            model::LoadedModel::Encoder(_) => {
                tracing::warn!(
                    "FRINK_GPU_VRAM_BUDGET_BYTES is set but the loaded model is an encoder -- \
                     it has no routed experts to place; ignoring"
                );
            }
        }
    } else if metal_default_moe_budget {
        // ~64 GiB sentinel: place as many experts as the planner allows;
        // Metal unified memory makes a hard VRAM split less meaningful
        // than on discrete CUDA cards.
        const METAL_DEFAULT_MOE_BUDGET: u64 = 64 * 1024 * 1024 * 1024;
        if let model::LoadedModel::Gguf(g) = &mut loaded {
            tracing::info!(
                "Metal MoE expert placement default-on ({METAL_DEFAULT_MOE_BUDGET} byte budget); \
                 set FRINK_GPU_VRAM_BUDGET_BYTES=0 to force CPU experts"
            );
            g.decoder.gpu_vram_budget_bytes = Some(METAL_DEFAULT_MOE_BUDGET);
        }
    }
    #[cfg(feature = "cuda")]
    {
        if frink_core::cuda_dense_enabled() {
            tracing::info!(
                "CUDA dense matvec enabled for WeightMatrix::apply \
                 (FRINK_CUDA=0|cpu forces CPU; weight buffers stay resident after first upload)"
            );
        } else {
            tracing::info!(
                "CUDA dense matvec disabled (FRINK_CUDA); dense decode uses CPU or Metal"
            );
        }
    }
    #[cfg(feature = "metal")]
    {
        if frink_core::metal_dense_enabled() {
            tracing::info!(
                "Metal dense matvec enabled for WeightMatrix::apply \
                 (FRINK_METAL=0|cpu forces CPU; weight buffers stay resident after first upload)"
            );
            match std::env::var("FRINK_METAL_ATTN").ok().as_deref() {
                Some("1") | Some("true") | Some("on") | Some("attn") => {
                    tracing::info!(
                        "Metal fused attention requested (FRINK_METAL_ATTN): \
                         QKV→RoPE→GQA→O on-GPU for Norm/NeoX decode without QKV bias/QK-norm"
                    );
                }
                _ => {}
            }
            tracing::info!(
                "Metal greedy GPU argmax: temperature<=0 folds \
                 final_norm+lm_head+argmax into the dense stack"
            );
        } else {
            tracing::info!("Metal dense matvec disabled (FRINK_METAL); dense decode uses CPU");
        }
    }
    // Both env vars are required together to enable pooling; unset ->
    // caches keep their original unbounded-per-request growth. This
    // mirrors the FRINK_API_KEY / FRINK_RATE_LIMIT_PER_MINUTE
    // pattern below: opt-in, off by default.
    //
    // Block count can be set explicitly (`FRINK_KV_POOL_BLOCKS` +
    // `FRINK_KV_POOL_BLOCK_SIZE`) or derived from a byte budget
    // (`FRINK_KV_BYTE_BUDGET` + `FRINK_KV_POOL_BLOCK_SIZE`, GGUF
    // models only). `FRINK_KV_POOL_BLOCKS` and
    // `FRINK_KV_BYTE_BUDGET` are mutually exclusive.
    let blocks_env = std::env::var("FRINK_KV_POOL_BLOCKS");
    let block_size_env = std::env::var("FRINK_KV_POOL_BLOCK_SIZE");
    let byte_budget_env = std::env::var("FRINK_KV_BYTE_BUDGET");
    if blocks_env.is_ok() && byte_budget_env.is_ok() {
        panic!(
            "FRINK_KV_POOL_BLOCKS and FRINK_KV_BYTE_BUDGET are mutually exclusive \
             (set one block-count source plus FRINK_KV_POOL_BLOCK_SIZE, or neither to disable)"
        );
    }
    let kv_pool = match (blocks_env, block_size_env, byte_budget_env) {
        (Ok(blocks), Ok(block_size), Err(_)) => {
            let total_blocks: usize = blocks
                .parse()
                .expect("FRINK_KV_POOL_BLOCKS must be a positive integer");
            let block_size: usize = block_size
                .parse()
                .expect("FRINK_KV_POOL_BLOCK_SIZE must be a positive integer");
            // Optional and independent of the two above: how long a
            // request retries before giving up when the pool is
            // momentarily exhausted, instead of rejecting on the very
            // first failed attempt. Zero (the default if unset)
            // preserves the original reject-immediately behavior.
            let queue_wait_ms: u64 = std::env::var("FRINK_KV_POOL_QUEUE_TIMEOUT_MS")
                .ok()
                .map(|v| {
                    v.parse()
                        .expect("FRINK_KV_POOL_QUEUE_TIMEOUT_MS must be a non-negative integer")
                })
                .unwrap_or(0);
            tracing::info!(
                "KV cache block pool enabled: {total_blocks} blocks x {block_size} positions \
                 each, shared across all concurrent requests, {queue_wait_ms}ms admission queue wait"
            );
            Some(generate::KvPoolConfig {
                pool: Arc::new(Mutex::new(KvBlockPool::new(block_size, total_blocks))),
                queue_wait: Duration::from_millis(queue_wait_ms),
            })
        }
        (Err(_), Ok(block_size), Ok(byte_budget)) => {
            let block_size: usize = block_size
                .parse()
                .expect("FRINK_KV_POOL_BLOCK_SIZE must be a positive integer");
            let budget: u64 = byte_budget
                .parse()
                .expect("FRINK_KV_BYTE_BUDGET must be a positive integer");
            let cfg = match &loaded {
                model::LoadedModel::Gguf(g) => &g.decoder.config,
                model::LoadedModel::Kimi(_)
                | model::LoadedModel::Mla(_)
                | model::LoadedModel::Gemma4(_)
                | model::LoadedModel::Glm52(_)
                | model::LoadedModel::Encoder(_) => {
                    panic!(
                        "FRINK_KV_BYTE_BUDGET requires a GGUF decoder model \
                         (set FRINK_MODEL_PATH to a generic-decoder .gguf file)"
                    );
                }
            };
            let bytes_per_block = block_size
                * cfg.kv_heads_all_layers()
                * (cfg.head_dim + cfg.v_head_dim())
                * std::mem::size_of::<f32>();
            assert!(
                bytes_per_block > 0,
                "derived KV block byte size must be positive (check model config and block size)"
            );
            let total_blocks = (budget as usize / bytes_per_block).max(1);
            let queue_wait_ms: u64 = std::env::var("FRINK_KV_POOL_QUEUE_TIMEOUT_MS")
                .ok()
                .map(|v| {
                    v.parse()
                        .expect("FRINK_KV_POOL_QUEUE_TIMEOUT_MS must be a non-negative integer")
                })
                .unwrap_or(0);
            tracing::info!(
                "KV cache block pool enabled from byte budget: {budget} bytes / \
                 {bytes_per_block} bytes per block ({block_size} positions x {} layers) -> \
                 {total_blocks} blocks, {queue_wait_ms}ms admission queue wait",
                cfg.n_layers
            );
            Some(generate::KvPoolConfig {
                pool: Arc::new(Mutex::new(KvBlockPool::new(block_size, total_blocks))),
                queue_wait: Duration::from_millis(queue_wait_ms),
            })
        }
        (Err(_), Err(_), Err(_)) => None,
        (Err(_), Ok(_), Err(_)) => panic!(
            "FRINK_KV_POOL_BLOCK_SIZE requires FRINK_KV_POOL_BLOCKS or FRINK_KV_BYTE_BUDGET \
             (or unset all three to disable KV cache pooling)"
        ),
        (Ok(_), Ok(_), Ok(_)) => {
            unreachable!("FRINK_KV_POOL_BLOCKS and FRINK_KV_BYTE_BUDGET are mutually exclusive")
        }
        (Ok(_), Err(_), _) | (Err(_), Err(_), Ok(_)) => panic!(
            "FRINK_KV_POOL_BLOCKS/FRINK_KV_BYTE_BUDGET and FRINK_KV_POOL_BLOCK_SIZE must be \
             set together (or neither, to disable KV cache pooling)"
        ),
    };
    // Paged KV: per-layer shared page storage rather than a private
    // contiguous buffer per request. Refused alongside the pool and the
    // prefix cache rather than silently preferred over either -- an
    // operator who set two of these meant one of them, and picking for
    // them is how a deployment ends up not running what it thinks.
    let paged_kv = match (
        std::env::var("FRINK_PAGED_KV_BLOCKS"),
        std::env::var("FRINK_PAGED_KV_BLOCK_SIZE"),
    ) {
        (Ok(blocks), Ok(block_size)) => {
            assert!(
                kv_pool.is_none(),
                "FRINK_PAGED_KV_BLOCKS and FRINK_KV_POOL_BLOCKS/FRINK_KV_BYTE_BUDGET are \
                 mutually exclusive: both bound the same KV memory, by different means. \
                 Set one."
            );
            // Paged KV used to be refused here on any GPU backend,
            // because it returned fluent wrong tokens on Metal: the
            // prefill left K/V on the device and filled the host cache
            // with `KvCache::advance_len` placeholders, and the paged
            // prefill then copied those placeholders into the page
            // store. The decode that followed attended over a prompt
            // the model never saw.
            //
            // Fixed in `frink_models::Decoder`, which now downloads
            // the real rows for the caller that reads them, and pinned
            // on hardware by `paged_metal_parity` -- greedy ids
            // identical between paged and contiguous KV on a dense
            // model, an MoE model and a sliding-window model.
            let blocks_per_layer: usize = blocks
                .parse()
                .expect("FRINK_PAGED_KV_BLOCKS must be a positive integer");
            let block_size: usize = block_size
                .parse()
                .expect("FRINK_PAGED_KV_BLOCK_SIZE must be a positive integer");
            let gguf = match &loaded {
                model::LoadedModel::Gguf(g) => g,
                _ => panic!(
                    "FRINK_PAGED_KV_BLOCKS requires a GGUF decoder model \
                     (set FRINK_MODEL_PATH to a generic-decoder .gguf file)"
                ),
            };
            let cfg = &gguf.decoder.config;
            let queue_wait_ms: u64 = std::env::var("FRINK_KV_POOL_QUEUE_TIMEOUT_MS")
                .ok()
                .map(|v| {
                    v.parse()
                        .expect("FRINK_KV_POOL_QUEUE_TIMEOUT_MS must be a non-negative integer")
                })
                .unwrap_or(0);
            tracing::info!(
                "Paged KV enabled: {blocks_per_layer} blocks x {block_size} positions per \
                 layer across {} layers, shared by all concurrent requests, \
                 {queue_wait_ms}ms admission queue wait",
                cfg.n_layers
            );
            // Prefix sharing rides on the same switch: paged KV is
            // what makes it possible at all, since sharing means two
            // sequences pointing at one page rather than one of them
            // holding a copy.
            let radix = Some(Arc::new(Mutex::new(crate::policy::radix::RadixCache::new(
                block_size,
            ))));
            // The anchor: the position an agentic turn will come back
            // to. Resolved ONCE here, from the served checkpoint's own
            // family and its own tokenizer, because it has to be a
            // single token id for the slide to recognize it on the hot
            // path for nothing. A checkpoint whose opener is more than
            // one token, or whose family has no opener at all (harmony
            // opens a call with an ordinary channel header), simply gets
            // no anchors and the slide follows the cursor.
            let anchor_token = crate::policy::anchor::resolve_anchor_token(
                crate::policy::parser::ToolCallFormat::infer(
                    &std::env::var("FRINK_MODEL_PATH").unwrap_or_default(),
                )
                .opener(),
                |text| {
                    gguf.tokenizer
                        .encode(text, SpecialTokens::Parse)
                        .into_iter()
                        .map(|t| t as u32)
                        .collect()
                },
            );
            if let Some(id) = anchor_token {
                tracing::info!(
                    "Paged KV window slide: tool-call anchor is token {id}, so a turn's \
                     window stops short of where its next turn rejoins"
                );
            }
            let slide_interval: usize = std::env::var("FRINK_PAGED_KV_SLIDE_INTERVAL")
                .ok()
                .map(|v| {
                    v.parse()
                        .expect("FRINK_PAGED_KV_SLIDE_INTERVAL must be a positive integer")
                })
                .unwrap_or(crate::policy::pool_budget::DEFAULT_SWA_EVICTION_INTERVAL);
            if let Some(window) = cfg.uniform_sliding_window() {
                tracing::info!(
                    "Paged KV window slide enabled: every layer slides by {window} every \
                     {slide_interval} decode steps, so a request holds its prompt and a \
                     window rather than its whole context"
                );
            } else if cfg.kv_block_window().is_some() {
                tracing::info!(
                    "Paged KV window slide NOT enabled: this model has full-attention layers, \
                     and a page group holds one block in every layer"
                );
            }
            Some(generate::PagedKvConfig {
                // Per layer, because a per-layer-shape model's layers do
                // not all cache the same width (`layer_shapes`).
                store: Arc::new(cfg.new_paged_kv(block_size, blocks_per_layer)),
                queue_wait: Duration::from_millis(queue_wait_ms),
                radix,
                anchor_token,
                slide_interval,
            })
        }
        (Err(_), Err(_)) => None,
        _ => panic!(
            "FRINK_PAGED_KV_BLOCKS and FRINK_PAGED_KV_BLOCK_SIZE must be set together \
             (or neither, to disable paged KV)"
        ),
    };
    // Mutually exclusive with kv_pool (see generate::generate's doc
    // comment on why a pool-backed cache can't safely be restored from
    // a prefix-cache clone): if both are set, the KV pool wins and
    // prefix caching is simply never consulted -- generate() already
    // enforces this per-request, so this is a heads-up for the
    // operator, not a hard failure.
    let prefix_cache = std::env::var("FRINK_PREFIX_CACHE_ENTRIES").ok().map(|v| {
        let max_entries: usize = v
            .parse()
            .expect("FRINK_PREFIX_CACHE_ENTRIES must be a positive integer");
        if kv_pool.is_some() {
            tracing::warn!(
                "FRINK_PREFIX_CACHE_ENTRIES is set but so is the KV pool -- prefix \
                     caching will never be consulted while a KV pool is configured"
            );
        }
        // A hard refusal rather than the warning above, because the
        // outcome is worse than "never consulted": `PrefixCache` stores
        // `Vec<KvCache>` snapshots, and a paged request has none to
        // give, so every store would be skipped and every lookup miss.
        // An operator would see a prefix cache configured, reporting
        // zero hits forever, with nothing saying why.
        assert!(
            paged_kv.is_none(),
            "FRINK_PREFIX_CACHE_ENTRIES and FRINK_PAGED_KV_BLOCKS are mutually exclusive: \
             the prefix cache stores contiguous KV snapshots, which a paged request does not \
             produce, so the cache could never hit. Set one."
        );
        tracing::info!(
            "KV-prefix cache enabled: up to {max_entries} stored prefixes, shared across \
                 all requests"
        );
        Arc::new(Mutex::new(PrefixCache::new(max_entries)))
    });
    if matches!(
        loaded,
        model::LoadedModel::Kimi(_) | model::LoadedModel::Mla(_) | model::LoadedModel::Glm52(_)
    ) && (kv_pool.is_some() || prefix_cache.is_some())
    {
        tracing::warn!(
            "KV pool / prefix cache are configured but the loaded model is Kimi, MLA, or GLM-5.2 -- \
             neither is consulted for those engines (state shapes differ from Decoder KV); see \
             frink_models::engine's module docs"
        );
    }
    let enable_cb =
        resolve_continuous_batching_enabled(&loaded, &kv_pool, &prefix_cache, &paged_kv);
    if enable_cb && continuous_batching_env().is_none() && metal_private_decode_active() {
        tracing::info!(
            "continuous batching enabled by default on Metal for safe parallel serving \
             (set FRINK_CONTINUOUS_BATCHING=0 or --no-cont-batching to use the private path)"
        );
    }
    if continuous_batching_env() == Some(true)
        && !continuous_batching_compatible(&loaded, &kv_pool, &prefix_cache, &paged_kv)
        && (kv_pool.is_some() || prefix_cache.is_some())
    {
        tracing::warn!(
            "FRINK_CONTINUOUS_BATCHING=1 ignored while KV pool or prefix cache is configured \
             (those modes keep the private generate path)"
        );
    }
    if let Ok(n) = std::env::var("FRINK_CHUNKED_PREFILL") {
        if let Ok(chunk) = n.parse::<usize>() {
            if chunk > 0 {
                tracing::info!("chunked prefill enabled: {chunk} tokens per forward_batch chunk");
            }
        }
    }
    if matches!(
        std::env::var("FRINK_CPU_KV_OFFLOAD").ok().as_deref(),
        Some("1")
    ) {
        tracing::warn!(
            "FRINK_CPU_KV_OFFLOAD=1: syncing Metal KV to host after each decode step \
             (minimal spill; full layer offload still planned)"
        );
    }

    let mcp = match mcp_config_path {
        Some(path) => {
            let loaded = mcp::load_mcp_config(&path)?;
            tracing::info!(
                "MCP config loaded from {} ({} server(s); invocation not wired yet)",
                loaded.path,
                loaded.servers.len()
            );
            Some(loaded)
        }
        None => None,
    };

    // Started before the router is built so the probe overlaps with
    // binding the port: by the time a client can ask, it has usually
    // already landed.
    let detection = health::Detection::spawn();

    let state = Arc::new(build_app_state(
        StartupModels {
            loaded,
            embedding: embedding_model,
        },
        kv_pool,
        paged_kv,
        prefix_cache,
        enable_cb,
        mcp,
        detection,
    ));

    // Paths come from `frink_api::routes` rather than string literals
    // so the UI, `frink chat` and this router cannot disagree about
    // what the surface is.
    use frink_api::routes;

    // Frink Studio is a separate app served by its own dev/static
    // server (see `ui/` at the repository root); it reaches this
    // process over the public HTTP API like any other client, so there
    // is nothing to mount here and `/` stays a 404.
    let public = Router::new().route(routes::HEALTH, get(health));

    let mut protected = protected_routes();

    // Both off by default; set the corresponding env var to enable.
    // route_layer (not layer) so these apply only to the routes above,
    // never to /health, which stays reachable for liveness/readiness
    // probes regardless of auth or rate-limit configuration.
    if let Ok(key) = std::env::var("FRINK_API_KEY") {
        tracing::info!("API key auth enabled");
        let auth = limits::AuthConfig {
            api_key: Arc::new(key),
        };
        protected = protected.route_layer(axum::middleware::from_fn_with_state(
            auth,
            limits::require_api_key,
        ));
    }
    if let Ok(rpm) = std::env::var("FRINK_RATE_LIMIT_PER_MINUTE") {
        let rpm: u32 = rpm
            .parse()
            .expect("FRINK_RATE_LIMIT_PER_MINUTE must be a positive integer");
        tracing::info!("rate limiting enabled: {rpm} requests/minute (global)");
        let limiter = Arc::new(limits::RateLimiter::per_minute(rpm));
        protected = protected.route_layer(axum::middleware::from_fn_with_state(
            limiter,
            limits::rate_limit,
        ));
    }
    // Off by default; set FRINK_CORS_ORIGINS (comma-separated exact
    // origins) to enable. No wildcard support by design -- see
    // `security::parse_cors_origins`'s doc comment. Added last (so it's
    // the outermost route_layer, run before auth/rate-limiting): a CORS
    // preflight (OPTIONS) request carries no Authorization header and
    // is answered directly by `CorsLayer` itself, so it must not be
    // blocked by the auth/rate-limit layers underneath.
    if let Ok(spec) = std::env::var("FRINK_CORS_ORIGINS") {
        let origins = security::parse_cors_origins(&spec)
            .unwrap_or_else(|e| panic!("FRINK_CORS_ORIGINS: {e}"));
        tracing::info!(
            "CORS enabled: {} allow-listed origin(s) ({})",
            origins.len(),
            spec
        );
        let cors = tower_http::cors::CorsLayer::new()
            .allow_origin(tower_http::cors::AllowOrigin::list(origins))
            .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
                // The self-declared client label the monitor records
                // (see `attribution`). A custom header makes every
                // cross-origin call preflighted, so omitting it here
                // would not merely drop the label -- it would fail the
                // request outright.
                axum::http::HeaderName::from_static(attribution::CLIENT_HEADER),
                // Set by hand rather than by `EventSource`, because
                // this API needs POST and a bearer token. Same
                // consequence if it is missing.
                axum::http::HeaderName::from_static("last-event-id"),
            ]);
        protected = protected.route_layer(cors);
    }

    // Outermost on purpose: every 503 this server can emit -- from a
    // handler, from `require_active`, or from the batch scheduler's
    // queue cap -- leaves with a `Retry-After` a client can act on.
    let app = public
        .merge(protected)
        .layer(axum::middleware::from_fn(limits::retry_after))
        .with_state(state);

    // TLS is off by default -- set FRINK_TLS_CERT and FRINK_TLS_KEY
    // together to serve HTTPS instead of plain HTTP; unset (either or
    // both) preserves the original plain-HTTP behavior exactly. See
    // `security::tls_paths_from_env`'s doc comment for why this can't
    // be meaningfully unit-tested here.
    let tls_paths = security::tls_paths_from_env().unwrap_or_else(|e| panic!("{e}"));
    install_ring_crypto_provider();
    // Both arms bind first and read the address back off the socket
    // rather than trusting the requested one: with `--port 0` the
    // requested port is a lie by construction, and the ready line has
    // to carry what the kernel actually handed out.
    match tls_paths {
        Some(paths) => {
            let config =
                axum_server::tls_rustls::RustlsConfig::from_pem_file(&paths.cert, &paths.key)
                    .await
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "failed to load TLS cert/key ({:?}, {:?}): {e}",
                            paths.cert,
                            paths.key
                        )
                    })?;
            let socket_addr: std::net::SocketAddr = addr
                .parse()
                .map_err(|e| anyhow::anyhow!("invalid FRINK_ADDR {addr:?} for TLS: {e}"))?;
            let listener = std::net::TcpListener::bind(socket_addr)?;
            // Tokio panics outright when handed a BLOCKING socket
            // ("Registering a blocking socket with the tokio runtime is
            // unsupported"), and axum-server registers this one
            // internally. Without this the TLS arm binds, prints its
            // ready line, and then panics on the first accept -- so the
            // failure looks like a healthy start followed by a server
            // that answers nothing.
            listener.set_nonblocking(true)?;
            let bound = listener.local_addr()?;
            tracing::info!("TLS enabled: frink-server listening on https://{bound}");
            announce_ready(bound, "https");

            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();
            tokio::spawn(async move {
                shutdown_signal(exit_on_stdin_close).await;
                shutdown_handle.graceful_shutdown(Some(Duration::from_secs(5)));
            });
            axum_server::from_tcp_rustls(listener, config)?
                .handle(handle)
                .serve(app.into_make_service())
                .await?;
        }
        None => {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            let bound = listener.local_addr()?;
            tracing::info!("frink-server listening on {bound}");
            announce_ready(bound, "http");
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal(exit_on_stdin_close))
                .await?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use frink_models::config::test_dense_fixture;

    #[test]
    fn the_ready_line_round_trips_through_a_parent_reading_stdout() {
        let addr: SocketAddr = "127.0.0.1:51999".parse().unwrap();
        let ready = frink_api::ServerReady::new(addr, "http", "0.5.0", std::process::id());
        let parsed = frink_api::ServerReady::from_line(&ready.to_line()).unwrap();
        assert_eq!(parsed.port, 51999);
        assert_eq!(parsed.base_url(), "http://127.0.0.1:51999");
        // A parent reads stdout line by line; tracing shares the stream.
        assert!(frink_api::ServerReady::from_line("INFO frink-server listening").is_none());
    }

    fn test_model() -> Model {
        // Tiny vocab (32): raw byte ids ≥32 (e.g. ASCII "hello") are OOV.
        // HTTP/chat-template tests that need full ASCII use
        // `test_model_full_byte_vocab` instead.
        let cfg = test_dense_fixture();
        Model::Gguf(GgufModel {
            decoder: Arc::new(Decoder::new_random_small(cfg, 2, 32)),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: None,
            is_synthetic: true,
            chat_template: chat_template::PromptTemplate::plain(),
        })
    }

    fn greedy_params(max_tokens: usize) -> GenerationParams {
        GenerationParams {
            reasoning: None,
            max_tokens,
            sampling: SamplingParams::default(),
            seed: 1,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            json_object: false,
            grammar: None,
            cancel: None,
            ignore_eos: false,
            reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
            lora: None,
        }
    }

    /// Declares a full 0..255 byte-compatible vocab so HTTP-level tests
    /// that render chat templates (ASCII role names) do not spuriously
    /// reject their own prompt prefixes.
    fn test_model_full_byte_vocab() -> Model {
        test_model_full_byte_vocab_with_eos(None)
    }

    /// [`test_model_full_byte_vocab`] with an end-of-generation id, so a
    /// test can tell a turn the MODEL ended from one that merely ran out
    /// of budget -- which is the only way `ignore_eos` is observable.
    ///
    /// Parameterised rather than copied: a second `Model` literal here
    /// is one more place a field has to be remembered.
    fn test_model_full_byte_vocab_with_eos(eos: Option<usize>) -> Model {
        let mut cfg = test_dense_fixture();
        cfg.vocab_size = 256;
        Model::Gguf(GgufModel {
            decoder: Arc::new(Decoder::new_random_small(cfg, 2, 256)),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::from_eos(eos),
            bos_id: None,
            is_synthetic: true,
            chat_template: chat_template::PromptTemplate::plain(),
        })
    }

    /// One `AppState` for the HTTP-level tests, so a new field on the
    /// struct is added in one place rather than in every test that
    /// builds one.
    pub(crate) fn test_state(model: Model, response_cache: ResponseCache) -> AppState {
        AppState {
            embedding: None,
            paged_kv: None,
            active: std::sync::RwLock::new(Some(Arc::new(ActiveModel {
                id: None,
                loaded: Loaded::Generative(Arc::new(model)),
                batcher: None,
                ceiling: None,
                checkpoint_path: None,
            }))),
            load_in_progress: std::sync::atomic::AtomicBool::new(false),
            tasks: Arc::new(tasks::TaskRegistry::new()),
            cancels: Arc::new(cancel::CancelRegistry::new()),
            stats: stats::Stats::new(),
            streams: resume::StreamRegistry::new(),
            model_dir: None,
            response_cache: Mutex::new(response_cache),
            kv_pool: None,
            prefix_cache: None,
            sessions: session::SessionStore::new(),
            requests_total: std::sync::atomic::AtomicU64::new(0),
            request_errors_total: std::sync::atomic::AtomicU64::new(0),
            started_at: std::time::Instant::now(),
            last_request_ms: std::sync::atomic::AtomicU64::new(0),
            detection: Arc::new(health::Detection::ready(health::probe_backends())),
            mcp: None,
            continuous_batching_enabled: false,
            metal_private_decode_gate: None,
            loading_model: Mutex::new(None),
            last_load_error: Mutex::new(None),
            serving: Mutex::new(crate::stats::ServingStats::default()),
            maintenance: Mutex::new(crate::policy::maintenance::MaintenanceGate::serving()),
            footprint: Mutex::new(crate::policy::footprint::ProbeCache::new(FOOTPRINT_TTL_MS)),
            started_unix: unix_now(),
        }
    }

    /// A real axum `Router` wired exactly like `main()`'s (minus auth/
    /// rate-limiting, which are orthogonal and already covered by
    /// `limits`'s own tests), backed by a fresh
    /// `test_model_full_byte_vocab()` -- so tool-calling/session tests
    /// exercise the real HTTP request/response path (JSON
    /// (de)serialization, routing, handler wiring, chat-template
    /// rendering) via `tower::ServiceExt::oneshot`, not just the inner
    /// functions directly.
    fn test_app() -> Router {
        test_app_with_state(Arc::new(test_state(
            test_model_full_byte_vocab(),
            ResponseCache::new(1000, Duration::from_secs(3600)),
        )))
    }

    /// [`test_app`] over a caller-owned state, so a test can reach in
    /// and swap or unload the model behind a live router.
    pub(crate) fn test_app_with_state(state: Arc<AppState>) -> Router {
        // The SAME route list the server builds, not a hand-written
        // copy of it. The copy that used to live here had drifted from
        // the real one, which is the failure mode that makes an HTTP
        // test worthless: it can only ever confirm that the tests agree
        // with the tests. See `protected_routes`.
        //
        // No auth, rate-limit or CORS layer: those are configured from
        // the environment in `run`, and a test that set the environment
        // would race every other test in the process.
        Router::new()
            .route(frink_api::routes::HEALTH, get(health))
            .merge(protected_routes())
            .with_state(state)
    }

    fn named_test_model(name: &'static str, vocab_size: usize) -> Model {
        let mut cfg = test_dense_fixture();
        cfg.name = name;
        cfg.vocab_size = vocab_size;
        Model::Gguf(GgufModel {
            decoder: Arc::new(Decoder::new_random_small(cfg, 2, 256)),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: None,
            is_synthetic: true,
            chat_template: chat_template::PromptTemplate::plain(),
        })
    }

    /// The same model, served through a real checkpoint's template
    /// rather than the role-labeled builtin -- so a test can ask what
    /// gets advertised for a checkpoint that actually has gears.
    fn model_with_template(name: &'static str, source: &str) -> Model {
        let mut cfg = test_dense_fixture();
        cfg.name = name;
        cfg.vocab_size = 256;
        Model::Gguf(GgufModel {
            decoder: Arc::new(Decoder::new_random_small(cfg, 2, 256)),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: None,
            is_synthetic: true,
            chat_template: chat_template::PromptTemplate::from_gguf_metadata(
                Some(source),
                Some("qwen3"),
                false,
                true,
                None,
                None,
            ),
        })
    }

    /// Once a `200` and `text/event-stream` are on the wire, a
    /// rejection can only ride *in* the stream, where several agents
    /// render it as an empty response. So the prompt is rendered before
    /// the stream is committed, and a template that rejects this
    /// particular conversation is an ordinary 400 with a body.
    ///
    /// Fails if `prompt_from_messages` moves back inside the spawned
    /// generation task.
    #[tokio::test]
    async fn a_template_that_rejects_the_conversation_is_a_400_on_the_streaming_path() {
        // Raises on a second user turn, the way a real strict template
        // rejects an ordering it was never trained on.
        let strict = "{% if messages | length > 1 %}\
             {{ raise_exception('this template takes one turn') }}\
             {% endif %}{{ messages[0].content }}";
        let state = Arc::new(test_state(
            model_with_template("strict", strict),
            ResponseCache::new(4, Duration::from_secs(60)),
        ));
        let app = test_app_with_state(state);

        let (status, body) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "strict",
                "stream": true,
                "messages": [
                    {"role": "user", "content": "one"},
                    {"role": "user", "content": "two"},
                ],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], serde_json::json!("messages"));
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("one turn"),
            "the template's own message must reach the caller: {body}"
        );

        // And the same template serves a conversation it accepts.
        let (status, _) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "strict",
                "stream": true,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "one"}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    /// A client should not have to guess which gears a checkpoint has.
    #[tokio::test]
    async fn models_advertises_the_gears_this_checkpoint_actually_has() {
        let reasoning = "{% if enable_thinking %}<think>{% endif %}\
             {% if reasoning_effort %}\
               {% if reasoning_effort not in ['low','medium','high'] %}\
                 {{ raise_exception('bad effort') }}\
               {% endif %}[{{ reasoning_effort }}]\
             {% endif %}{{ messages[0].content }}";
        let state = Arc::new(test_state(
            model_with_template("thinker", reasoning),
            ResponseCache::new(4, Duration::from_secs(60)),
        ));
        let app = test_app_with_state(state);
        let (status, models) = get_json(&app, frink_api::routes::V1_MODELS).await;
        assert_eq!(status, StatusCode::OK);
        let entry = &models["data"][0];
        assert_eq!(
            entry["supported_reasoning_efforts"],
            serde_json::json!(["off", "low", "medium", "high"])
        );
        assert_eq!(entry["default_reasoning_effort"], serde_json::json!("off"));
    }

    /// The other half of the acceptance criterion: neither field, not
    /// an empty one. An empty list would say the question was asked and
    /// the answer was "no gears"; absence says it is not that kind of
    /// model.
    #[tokio::test]
    async fn a_checkpoint_with_no_thinking_controls_advertises_neither_field() {
        let app = test_app();
        let (_, models) = get_json(&app, frink_api::routes::V1_MODELS).await;
        let entry = &models["data"][0];
        assert!(entry.get("supported_reasoning_efforts").is_none());
        assert!(entry.get("default_reasoning_effort").is_none());
    }

    fn active_model(state: &AppState, name: &'static str) -> Arc<ActiveModel> {
        Arc::new(ActiveModel {
            id: Some(name.to_string()),
            loaded: Loaded::Generative(Arc::new(named_test_model(name, 256))),
            batcher: None,
            ceiling: None,
            checkpoint_path: None,
        })
        .tap_into(state)
    }

    /// Small helper so the swap tests read as "publish this model".
    trait TapInto {
        fn tap_into(self, state: &AppState) -> Self;
    }
    impl TapInto for Arc<ActiveModel> {
        fn tap_into(self, state: &AppState) -> Self {
            state.swap_active(Some(Arc::clone(&self)));
            self
        }
    }

    /// The load-order guarantee the whole swap design exists to make:
    /// a request that has already taken its handle finishes against the
    /// weights it started on, even though a different model has since
    /// been published. Anything else would splice two checkpoints into
    /// one completion.
    #[test]
    fn an_in_flight_request_keeps_the_model_it_started_on() {
        let state = test_state(
            named_test_model("model-a", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        );

        // A request that has begun: it has cloned the handle and is
        // about to decode against it.
        let in_flight = state.active().expect("a model is loaded");
        assert_eq!(in_flight.name(), "model-a");

        active_model(&state, "model-b");

        // The swap is visible to anything that asks *now*...
        assert_eq!(state.active().unwrap().name(), "model-b");
        // ...and completely invisible to the request already running.
        assert_eq!(in_flight.name(), "model-a");
        let (_chunks, finish, _usage) = run_generation(
            in_flight.generative().unwrap(),
            "hi",
            &greedy_params(3),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("the old model must still decode after being swapped out");
        assert!(matches!(finish, FinishReason::Length | FinishReason::Stop));
    }

    /// The other half of the same guarantee: the old model is not freed
    /// at swap time, it is freed when the last holder lets go. A design
    /// that dropped it eagerly would free weights out from under a
    /// decode loop.
    #[test]
    fn a_swapped_out_model_lives_until_its_last_holder_releases_it() {
        let state = test_state(
            named_test_model("model-a", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        );
        let in_flight = state.active().expect("a model is loaded");
        let weights = Arc::clone(in_flight.generative().unwrap());
        assert!(Arc::strong_count(&weights) >= 2);

        let previous = state.swap_active(Some(Arc::new(ActiveModel {
            id: Some("model-b".to_string()),
            loaded: Loaded::Generative(Arc::new(named_test_model("model-b", 256))),
            batcher: None,
            ceiling: None,
            checkpoint_path: None,
        })));
        drop(previous);
        // The registry has let go; the in-flight request has not.
        assert!(Arc::strong_count(&weights) >= 2);
        drop(in_flight);
        assert_eq!(Arc::strong_count(&weights), 1);
    }

    /// Unload is not "keep serving the last thing loaded". A request
    /// that arrives afterwards must be told there is no model, not
    /// quietly served by a checkpoint the operator dropped.
    #[tokio::test]
    async fn unloading_answers_503_instead_of_serving_the_dropped_model() {
        let state = Arc::new(test_state(
            named_test_model("model-a", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        ));
        let app = test_app_with_state(Arc::clone(&state));

        let (status, body) = post_json_uri(
            &app,
            frink_api::routes::ADMIN_MODELS_UNLOAD,
            serde_json::json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["ok"], true);
        assert!(body["active"].is_null());
        assert!(state.active().is_none());

        let (status, _) = get_json(&app, frink_api::routes::V1_MODELS).await;
        assert_eq!(status, StatusCode::OK);
        let (_, models) = get_json(&app, frink_api::routes::V1_MODELS).await;
        assert_eq!(models["data"].as_array().unwrap().len(), 0);

        let (status, body) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["type"], "model_not_loaded");
    }

    /// `/health` must keep answering with nothing loaded -- a supervisor
    /// polls it to decide whether to kill the process, and "no model"
    /// is not "no server".
    #[tokio::test]
    async fn health_reports_the_unloaded_state_rather_than_going_silent() {
        let state = Arc::new(test_state(
            named_test_model("model-a", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        ));
        let app = test_app_with_state(Arc::clone(&state));
        state.swap_active(None);

        let (status, body) = get_json(&app, frink_api::routes::HEALTH).await;
        // Not `ready`: a supervisor reading 200 here would route traffic
        // that is guaranteed to 503 on arrival.
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["state"], "unavailable");
        assert_eq!(body["reason"], "model_not_loaded");
        assert!(body["model"].is_null());
        let real_weights = body["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == "real_weights")
            .cloned()
            .expect("real_weights is always reported");
        assert_eq!(real_weights["available"], false);
        assert_eq!(real_weights["reason"], "model_not_loaded");
    }

    /// The API-monitor contract: a finished request lands in the ring
    /// buffer keyed by the id the response carried, with the two
    /// durations reported separately.
    #[tokio::test]
    async fn a_finished_request_lands_in_the_stats_ring_with_both_durations() {
        let app = test_app();

        let (status, completion) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let request_id = completion["request_id"].as_str().unwrap().to_string();

        let (status, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        assert_eq!(status, StatusCode::OK);
        let recent = stats["recent"].as_array().unwrap();
        assert_eq!(recent.len(), 1);
        let row = &recent[0];
        assert_eq!(row["request_id"], request_id);
        assert_eq!(row["route"], frink_api::routes::V1_CHAT_COMPLETIONS);
        assert_eq!(row["status"], 200);
        assert_eq!(row["stream"], false);
        // Separate fields, and the decode phase is a real measurement
        // rather than a copy of the total.
        assert!(row["duration_ms"].is_number());
        assert!(row["decode_ms"].is_number());
        assert!(stats["tokens_generated_total"].as_u64().unwrap() > 0);
        assert_eq!(
            stats["tokens_prompt_total"].as_u64().unwrap(),
            row["prompt_tokens"].as_u64().unwrap()
        );
    }

    /// A rejected request is still a request the monitor should show;
    /// otherwise the screen quietly omits exactly the traffic someone
    /// is debugging.
    #[tokio::test]
    async fn a_rejected_request_is_recorded_too() {
        let state = Arc::new(test_state(
            named_test_model("model-a", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        ));
        let app = test_app_with_state(Arc::clone(&state));
        state.swap_active(None);

        let (status, _) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

        let (_, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        let recent = stats["recent"].as_array().unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0]["status"], 503);
        assert_eq!(recent[0]["completion_tokens"], 0);
        assert!(recent[0]["decode_ms"].is_null());
        assert_eq!(stats["errors_total"], 1);
    }

    /// POSTs with caller-supplied headers, so the attribution tests
    /// exercise the same header parsing a real client's request goes
    /// through rather than calling `Attribution::from_headers` twice.
    async fn post_json_with_headers(
        app: &Router,
        uri: &str,
        body: serde_json::Value,
        headers: &[(&str, &str)],
    ) -> (StatusCode, serde_json::Value) {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let mut builder = axum::http::Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json");
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let response = app
            .clone()
            .oneshot(
                builder
                    .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({}));
        (status, json)
    }

    /// The three small endpoints used to be served and never recorded,
    /// which made the monitor wrong rather than incomplete: an editor
    /// hammering `/v1/embeddings` showed up as an idle server.
    #[tokio::test]
    async fn tokenize_detokenize_and_embeddings_all_land_in_the_ring() {
        let app = test_app();

        let (status, _) = post_json_uri(
            &app,
            frink_api::routes::V1_TOKENIZE,
            serde_json::json!({"prompt": "hello"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) = post_json_uri(
            &app,
            frink_api::routes::V1_DETOKENIZE,
            serde_json::json!({"tokens": [104, 105]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) = post_json_uri(
            &app,
            frink_api::routes::V1_EMBEDDINGS,
            serde_json::json!({"input": "hello"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (_, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        let routes: Vec<&str> = stats["recent"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["route"].as_str().unwrap())
            .collect();
        for expected in [
            frink_api::routes::V1_TOKENIZE,
            frink_api::routes::V1_DETOKENIZE,
            frink_api::routes::V1_EMBEDDINGS,
        ] {
            assert!(
                routes.contains(&expected),
                "{expected} is missing: {routes:?}"
            );
        }

        let row = |route: &str| {
            stats["recent"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["route"] == route)
                .cloned()
                .unwrap()
        };
        // Embeddings run a forward pass, so their prompt tokens are
        // real prompt tokens. There is no decode loop, so `decode_ms`
        // stays null instead of borrowing the total.
        let embed = row(frink_api::routes::V1_EMBEDDINGS);
        assert!(embed["prompt_tokens"].as_u64().unwrap() > 0);
        assert!(embed["decode_ms"].is_null());
        assert_eq!(embed["completion_tokens"], 0);
        // Tokenizing runs the tokenizer and not the model, so it
        // contributes nothing to the token counters those counters
        // claim to measure.
        assert_eq!(row(frink_api::routes::V1_TOKENIZE)["prompt_tokens"], 0);
        assert_eq!(
            stats["tokens_prompt_total"].as_u64().unwrap(),
            embed["prompt_tokens"].as_u64().unwrap(),
            "only the forward pass counted"
        );
    }

    /// A router over a model that is NOT flagged synthetic, so the
    /// decode loop actually emits chunks: `run_generation_emit`
    /// suppresses `emit` for a synthetic model, and a streaming test
    /// against one would see only the terminal frame.
    fn streaming_test_app() -> Router {
        let mut cfg = test_dense_fixture();
        cfg.vocab_size = 256;
        let model = Model::Gguf(GgufModel {
            decoder: Arc::new(Decoder::new_random_small(cfg, 2, 256)),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: None,
            is_synthetic: false,
            chat_template: chat_template::PromptTemplate::plain(),
        });
        test_app_with_state(Arc::new(test_state(
            model,
            ResponseCache::new(1000, Duration::from_secs(3600)),
        )))
    }

    /// llama.cpp's native endpoint is a different WIRE, not a shorter
    /// path to the OpenAI one. If this ever starts answering `choices`,
    /// every llama.cpp client reading `content` breaks silently.
    #[tokio::test]
    async fn the_native_completion_wire_is_not_the_openai_one() {
        let app = test_app();

        let (status, native) = post_json_uri(
            &app,
            frink_api::routes::COMPLETION,
            serde_json::json!({"prompt": "hi", "n_predict": 4}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{native}");
        assert!(native["content"].is_string(), "{native}");
        assert_eq!(native["stop"], true);
        assert_eq!(native["stop_type"], "limit");
        assert_eq!(native["stopping_word"], "");
        assert_eq!(native["truncated"], false);
        assert_eq!(native["id_slot"], -1);
        assert!(native["timings"]["prompt_n"].is_number(), "{native}");
        assert!(native["generation_settings"]["n_predict"] == 4, "{native}");
        assert!(
            native.get("choices").is_none(),
            "the native shape has no `choices`: {native}"
        );

        let (status, openai) = post_json_uri(
            &app,
            frink_api::routes::V1_COMPLETIONS,
            serde_json::json!({"prompt": "hi", "max_tokens": 4}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(openai["choices"][0]["text"].is_string(), "{openai}");
        assert!(
            openai.get("content").is_none(),
            "the OpenAI shape has no top-level `content`: {openai}"
        );
    }

    /// llama.cpp mounts the native endpoint under both spellings
    /// (`server.cpp:240-241`), and its own web UI uses the plural. One
    /// handler, so the two cannot answer differently.
    #[tokio::test]
    async fn both_native_spellings_reach_the_same_handler() {
        let app = test_app();
        for route in [
            frink_api::routes::COMPLETION,
            frink_api::routes::COMPLETIONS,
        ] {
            let (status, body) = post_json_uri(
                &app,
                route,
                serde_json::json!({"prompt": "hi", "n_predict": 2, "seed": 1}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{route}: {body}");
            assert_eq!(body["stop"], true, "{route}");
            assert!(body["content"].is_string(), "{route}");
        }

        // And the ring records which one was called, so the split
        // between clients stays visible.
        let (_, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        let routes: Vec<&str> = stats["recent"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["route"].as_str().unwrap())
            .collect();
        assert!(
            routes.contains(&frink_api::routes::COMPLETION),
            "{routes:?}"
        );
        assert!(
            routes.contains(&frink_api::routes::COMPLETIONS),
            "{routes:?}"
        );
    }

    /// The native stream is not OpenAI's. Frames are bare objects with
    /// `content` and `stop`, the last one carries `stop: true` and the
    /// whole terminal body, and there is **no `[DONE]`** -- a client
    /// waiting for one would hang, and one that got it would try to
    /// parse it as JSON.
    #[tokio::test]
    async fn a_native_stream_ends_on_a_stop_frame_with_no_done_sentinel() {
        let app = streaming_test_app();
        let raw = post_sse_raw_uri(
            &app,
            frink_api::routes::COMPLETION,
            serde_json::json!({"prompt": "hi", "n_predict": 6, "stream": true, "seed": 7}),
        )
        .await;

        assert!(
            !raw.contains("[DONE]"),
            "llama.cpp's native stream has no sentinel: {raw}"
        );
        let frames: Vec<serde_json::Value> = raw
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|json| serde_json::from_str(json).expect("every frame is one JSON object"))
            .collect();
        assert!(frames.len() >= 2, "expected partials then a final: {raw}");

        let (last, partials) = frames.split_last().unwrap();
        assert_eq!(last["stop"], true, "the last frame closes the stream");
        assert!(last["timings"].is_object(), "{last}");
        assert!(last["stop_type"].is_string(), "{last}");
        for partial in partials {
            assert_eq!(partial["stop"], false, "{partial}");
            assert!(partial["content"].is_string(), "{partial}");
            // Upstream's documented partial carries content/tokens/stop
            // and nothing else; the terminal fields belong to the last
            // frame only.
            assert!(partial.get("timings").is_none(), "{partial}");
            assert!(partial.get("generation_settings").is_none(), "{partial}");
        }
        // The concatenated partials are the answer, so a client that
        // streams sees what a client that buffers would get.
        let streamed: String = partials
            .iter()
            .filter_map(|p| p["content"].as_str())
            .collect();
        assert_eq!(last["content"].as_str().unwrap(), streamed);
    }

    /// `n_predict: -1` is llama.cpp's default AND its "until the
    /// context is full". With no derived ceiling there is no context to
    /// be full of, and quietly substituting a small budget would hand a
    /// caller a truncated answer it never asked for.
    #[tokio::test]
    async fn an_unbounded_n_predict_is_refused_rather_than_quietly_shrunk() {
        let app = test_app();
        for body in [
            serde_json::json!({"prompt": "hi"}),
            serde_json::json!({"prompt": "hi", "n_predict": -1}),
        ] {
            let (status, refusal) =
                post_json_uri(&app, frink_api::routes::COMPLETION, body.clone()).await;
            assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}: {refusal}");
            assert!(
                refusal["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("n_predict"),
                "{refusal}"
            );
        }
        // An explicit budget is served, so the refusal is about the
        // unbounded case and not about the endpoint.
        let (status, _) = post_json_uri(
            &app,
            frink_api::routes::COMPLETION,
            serde_json::json!({"prompt": "hi", "n_predict": 2}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    /// A caller's `stop` must actually reach the sampler, and be named
    /// back in llama.cpp's own vocabulary. Dropping it is the dangerous
    /// silent failure: the caller believes generation halts at its
    /// sentinel and instead gets the whole budget of text past it.
    ///
    /// Deterministic without depending on what random weights say:
    /// generate once with no stop, then take a character out of that
    /// answer and demand the second run halt before it.
    #[tokio::test]
    async fn a_stop_string_halts_the_answer_and_is_named_back() {
        let app = streaming_test_app();
        let ask = |stop: serde_json::Value| {
            let app = app.clone();
            async move {
                post_json_uri(
                    &app,
                    frink_api::routes::COMPLETION,
                    serde_json::json!({
                        "prompt": "hi",
                        "n_predict": 64,
                        "ignore_eos": true,
                        "stop": stop,
                    }),
                )
                .await
                .1
            }
        };

        let baseline = ask(serde_json::json!([])).await;
        assert_eq!(baseline["stop_type"], "limit");
        assert_eq!(baseline["stopping_word"], "");
        let text = baseline["content"].as_str().unwrap().to_string();
        // Two characters, so the sentinel is more than one token in
        // this vocabulary and goes through the output-suffix layer that
        // reports WHICH string matched. A single-token stop is caught
        // by the token layer, which does not carry the string back --
        // see `stop_type`'s note and docs/API.md.
        let sentinel: String = text.chars().skip(1).take(2).collect();
        assert_eq!(
            sentinel.chars().count(),
            2,
            "the fixture must produce enough output to cut: {text:?}"
        );
        let cut = text.find(&sentinel).expect("it came out of this text");

        let stopped = ask(serde_json::json!([sentinel])).await;
        assert_eq!(stopped["stop_type"], "word", "{stopped}");
        assert_eq!(stopped["stopping_word"], sentinel);
        assert_eq!(
            stopped["content"].as_str().unwrap(),
            &text[..cut],
            "the answer must be cut at the sentinel, not run past it"
        );
    }

    /// llama.cpp mounts these two unprefixed and sends `content`, not
    /// `prompt`. frink mounted only the `/v1/` spelling it invented,
    /// so every llama.cpp client got a 404 that named nothing. The
    /// alias must reach the SAME handler -- identical ids for identical
    /// text -- rather than a second implementation of it.
    #[tokio::test]
    async fn the_llama_cpp_spelling_of_tokenize_reaches_the_same_handler() {
        let app = test_app();

        let (v1_status, v1) = post_json_uri(
            &app,
            frink_api::routes::V1_TOKENIZE,
            serde_json::json!({"prompt": "hello"}),
        )
        .await;
        let (alias_status, alias) = post_json_uri(
            &app,
            frink_api::routes::TOKENIZE,
            serde_json::json!({"content": "hello"}),
        )
        .await;
        assert_eq!(v1_status, StatusCode::OK);
        assert_eq!(alias_status, StatusCode::OK, "{alias}");
        assert_eq!(v1["tokens"], alias["tokens"]);
        assert!(!alias["tokens"].as_array().unwrap().is_empty());

        // And the reverse: frink's own field still works on llama.cpp's
        // path, so a client that switches URLs need not switch dialects.
        let (status, both_ways) = post_json_uri(
            &app,
            frink_api::routes::TOKENIZE,
            serde_json::json!({"prompt": "hello"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(both_ways["tokens"], v1["tokens"]);
    }

    /// llama.cpp answers detokenize under `content`
    /// (`server-context.cpp:4970`); frink has always answered under
    /// `text`. Both keys carry the same string, so neither dialect's
    /// client reads a null.
    #[tokio::test]
    async fn detokenize_answers_under_both_dialects_keys() {
        let app = test_app();
        for route in [
            frink_api::routes::DETOKENIZE,
            frink_api::routes::V1_DETOKENIZE,
        ] {
            let (status, body) =
                post_json_uri(&app, route, serde_json::json!({"tokens": [104, 105]})).await;
            assert_eq!(status, StatusCode::OK, "{route}");
            assert_eq!(body["text"], "hi", "{route}");
            assert_eq!(body["content"], body["text"], "{route}");
        }
    }

    /// The alias is one handler, so the ring must not attribute a
    /// llama.cpp client's traffic to the frink spelling: the row
    /// carries the path that was actually matched.
    #[tokio::test]
    async fn the_alias_is_recorded_under_the_path_the_client_called() {
        let app = test_app();
        let (status, _) = post_json_uri(
            &app,
            frink_api::routes::TOKENIZE,
            serde_json::json!({"content": "hello"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (_, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        let routes: Vec<&str> = stats["recent"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["route"].as_str().unwrap())
            .collect();
        assert!(
            routes.contains(&frink_api::routes::TOKENIZE),
            "the alias must be its own row: {routes:?}"
        );
        assert!(
            !routes.contains(&frink_api::routes::V1_TOKENIZE),
            "nothing called /v1/tokenize: {routes:?}"
        );
    }

    /// `add_special` is llama.cpp's "prepend BOS". Honoured, and with
    /// the id the generation path itself would prepend -- a tokenize
    /// endpoint that disagrees with the decoder about the prompt is
    /// worse than one that has no such option.
    #[tokio::test]
    async fn add_special_prepends_the_same_bos_the_decoder_would() {
        let mut cfg = test_dense_fixture();
        cfg.vocab_size = 256;
        let model = Model::Gguf(GgufModel {
            decoder: Arc::new(Decoder::new_random_small(cfg, 2, 256)),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: Some(7),
            is_synthetic: true,
            chat_template: chat_template::PromptTemplate::plain(),
        });
        let app = test_app_with_state(Arc::new(test_state(
            model,
            ResponseCache::new(1000, Duration::from_secs(3600)),
        )));

        let (_, plain) = post_json_uri(
            &app,
            frink_api::routes::TOKENIZE,
            serde_json::json!({"content": "hi"}),
        )
        .await;
        let (_, special) = post_json_uri(
            &app,
            frink_api::routes::TOKENIZE,
            serde_json::json!({"content": "hi", "add_special": true}),
        )
        .await;

        assert_eq!(plain["tokens"], serde_json::json!([104, 105]));
        assert_eq!(special["tokens"], serde_json::json!([7, 104, 105]));
        assert_eq!(special["count"], 3);
    }

    /// A failed small-endpoint call is still traffic. A 400 that leaves
    /// no row is indistinguishable from a request that was never sent.
    #[tokio::test]
    async fn a_rejected_embeddings_request_is_recorded_with_its_status() {
        let app = test_app();
        let (status, _) = post_json_uri(
            &app,
            frink_api::routes::V1_EMBEDDINGS,
            serde_json::json!({"input": "hi", "encoding_format": "base64"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (_, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        let recent = stats["recent"].as_array().unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0]["route"], frink_api::routes::V1_EMBEDDINGS);
        assert_eq!(recent[0]["status"], 400);
        assert_eq!(
            recent[0]["prompt_tokens"], 0,
            "a rejected call embedded nothing"
        );
    }

    /// Attribution: which key served a request, and what the caller
    /// says it is. The key itself must never appear.
    #[tokio::test]
    async fn a_row_names_the_key_that_served_it_without_carrying_the_key() {
        let app = test_app();
        let key = "sk-monitor-secret";
        let (status, _) = post_json_with_headers(
            &app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2
            }),
            &[
                ("authorization", &format!("Bearer {key}")),
                ("x-frink-client", "frink-studio"),
            ],
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (_, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        let row = stats["recent"].as_array().unwrap()[0].clone();
        let fingerprint = row["via_api_key"]
            .as_str()
            .expect("the row names the key that served it")
            .to_string();
        assert_eq!(fingerprint, attribution::key_fingerprint(key));
        assert!(!fingerprint.contains(key));
        assert!(
            !serde_json::to_string(&stats).unwrap().contains(key),
            "the stats payload must not carry the key in any form"
        );
        assert_eq!(row["client"], "frink-studio");
    }

    /// Two different keys are two different callers, and no key at all
    /// is a third answer -- not a copy of either.
    #[tokio::test]
    async fn different_keys_are_different_callers_and_no_key_is_null() {
        let app = test_app();
        let body = serde_json::json!({
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        });
        for headers in [
            vec![("authorization", "Bearer key-one")],
            vec![("authorization", "Bearer key-two")],
            vec![],
        ] {
            let (status, _) =
                post_json_with_headers(&app, "/v1/chat/completions", body.clone(), &headers).await;
            assert_eq!(status, StatusCode::OK);
        }

        let (_, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        let recent = stats["recent"].as_array().unwrap();
        assert_eq!(recent.len(), 3);
        let one = recent[0]["via_api_key"].as_str().unwrap();
        let two = recent[1]["via_api_key"].as_str().unwrap();
        assert_ne!(one, two, "two keys must not collapse into one caller");
        assert!(
            recent[2]["via_api_key"].is_null(),
            "an unauthenticated call is null, not a fingerprint of nothing"
        );
        assert!(recent[2]["client"].is_null());
    }

    /// The row names the model that SERVED the request. `req.model` is
    /// ignored by this server -- it decodes against whatever is loaded
    /// -- so echoing that string back would make the log agree with the
    /// caller's belief instead of with what happened.
    #[tokio::test]
    async fn a_row_names_the_model_that_served_it_not_the_one_requested() {
        let state = Arc::new(test_state(
            named_test_model("really-loaded", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        ));
        let app = test_app_with_state(Arc::clone(&state));

        let (status, _) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "gpt-4-turbo-that-is-not-here",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (_, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        assert_eq!(stats["recent"][0]["model"], "really-loaded");

        // Nothing loaded: nothing served it, and the row says so rather
        // than repeating what the request asked for.
        state.swap_active(None);
        let (status, _) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "gpt-4-turbo-that-is-not-here",
                "messages": [{"role": "user", "content": "hi"}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let (_, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        let recent = stats["recent"].as_array().unwrap();
        assert!(recent[recent.len() - 1]["model"].is_null());
    }

    /// A streamed request names its model too, and names the handle it
    /// decoded against rather than whatever a swap made current while it
    /// was running.
    #[tokio::test]
    async fn a_streamed_row_names_the_model_it_decoded_against() {
        let state = Arc::new(test_state(
            named_test_model("model-before", 256),
            ResponseCache::new(4, Duration::from_secs(60)),
        ));
        let app = test_app_with_state(Arc::clone(&state));
        let _ = post_sse_raw(&app, resumable_request()).await;
        // The stream has finished; a swap now must not rewrite history.
        active_model(&state, "model-after");

        let (_, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        assert_eq!(stats["recent"][0]["model"], "model-before");
    }

    /// The queue gauge reports a queue that exists or says there is
    /// none. `0` would claim an empty queue was measured.
    #[tokio::test]
    async fn the_queue_gauge_is_null_when_nothing_can_queue() {
        let app = test_app();
        let (status, stats) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            stats["queue_depth"].is_null(),
            "without continuous batching nothing queues, so there is nothing to measure"
        );
        assert!(stats["queue_rejected_total"].is_null());
        assert_eq!(
            stats["generating_now"], 0,
            "work in progress is measured and really is zero here"
        );
    }

    /// The raw SSE body, so the tests below can assert on the `id:` and
    /// `retry:` fields themselves rather than only on the JSON inside
    /// `data:`. Those two fields are the whole of the replay contract
    /// on the wire.
    async fn post_sse_raw(app: &Router, body: serde_json::Value) -> String {
        post_sse_raw_uri(app, frink_api::routes::V1_CHAT_COMPLETIONS, body).await
    }

    /// The same, on any route: `/completion` streams a different
    /// protocol over the same transport, and a second copy of this
    /// helper would be a second thing to keep in step.
    async fn post_sse_raw_uri(app: &Router, uri: &str, body: serde_json::Value) -> String {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    async fn get_json_with_headers(
        app: &Router,
        uri: &str,
        headers: &[(&str, &str)],
    ) -> (StatusCode, serde_json::Value) {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let mut builder = axum::http::Request::builder().method("GET").uri(uri);
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let response = app
            .clone()
            .oneshot(builder.body(axum::body::Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({})),
        )
    }

    fn sse_field<'a>(body: &'a str, field: &str) -> Vec<&'a str> {
        body.lines()
            .filter_map(|line| line.strip_prefix(field))
            .map(str::trim)
            .collect()
    }

    fn resumable_request() -> serde_json::Value {
        serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
            "max_tokens": 4,
            "temperature": 0,
            "stream": true,
            "stream_resumable": true,
        })
    }

    /// The wire half of the replay contract: every event is numbered,
    /// the numbers are qualified by the request so a `Last-Event-ID`
    /// cannot be mistaken for a position in another stream, and the
    /// reconnect delay is stated once.
    #[tokio::test]
    async fn a_resumable_stream_numbers_every_event_and_states_retry_once() {
        let app = test_app();
        let body = post_sse_raw(&app, resumable_request()).await;

        let request_id = body
            .lines()
            .find_map(|l| l.strip_prefix("data: "))
            .and_then(|d| serde_json::from_str::<serde_json::Value>(d).ok())
            .and_then(|v| v["request_id"].as_str().map(str::to_string))
            .expect("the first chunk names the request");

        let ids = sse_field(&body, "id:");
        let datas = sse_field(&body, "data:");
        assert_eq!(
            ids.len(),
            datas.len(),
            "every event carries an id, or a reconnect cannot name where it stopped"
        );
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(*id, format!("{request_id}:{i}"));
        }
        let retries = sse_field(&body, "retry:");
        assert_eq!(
            retries.len(),
            1,
            "the reconnect delay is stated once, not on every event"
        );
        assert_eq!(retries[0], "1500");
        assert!(
            body.contains("data: [DONE]"),
            "the end of stream is still stated"
        );
    }

    /// The refusal this feature was written around: an `id:` with no
    /// replay buffer behind it tells a client it may reconnect into
    /// something that does not exist.
    #[tokio::test]
    async fn a_plain_stream_carries_no_id_because_nothing_could_replay_it() {
        let app = test_app();
        let mut request = resumable_request();
        request["stream_resumable"] = serde_json::json!(false);
        let body = post_sse_raw(&app, request).await;
        assert!(!sse_field(&body, "data:").is_empty(), "it still streams");
        assert!(
            sse_field(&body, "id:").is_empty(),
            "an id promises a replay this stream cannot serve"
        );
        assert!(sse_field(&body, "retry:").is_empty());
    }

    /// The polling fallback, which is the answer to the proxy that
    /// buffers `text/event-stream`: the same events, over a short JSON
    /// response nothing can hold back.
    #[tokio::test]
    async fn the_polling_fallback_serves_exactly_what_the_stream_delivered() {
        let app = test_app();
        let body = post_sse_raw(&app, resumable_request()).await;
        let request_id = sse_field(&body, "id:")[0]
            .rsplit_once(':')
            .unwrap()
            .0
            .to_string();
        let streamed: Vec<String> = sse_field(&body, "data:")
            .iter()
            .map(|d| d.to_string())
            .collect();

        let (status, polled) = get_json(
            &app,
            &format!("{}?from=0", frink_api::routes::v1_stream_poll(&request_id)),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let events: Vec<String> = polled["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["data"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            events, streamed,
            "the fallback must deliver the same answer, not a re-run of it"
        );
        assert_eq!(polled["request_id"], request_id);
        assert_eq!(
            polled["done"], false,
            "events were still being handed out, so the client must ask again"
        );

        // Drained: only now is it done, so a client that stops on
        // `done` never discards events it was not given.
        let next = polled["next_index"].as_u64().unwrap();
        let (_, drained) = get_json(
            &app,
            &format!(
                "{}?from={next}",
                frink_api::routes::v1_stream_poll(&request_id)
            ),
        )
        .await;
        assert_eq!(drained["done"], true);
        assert_eq!(drained["events"].as_array().unwrap().len(), 0);
    }

    /// A resume returns what was missed and not what was already
    /// rendered -- repeating delivered tokens would make replay worse
    /// than starting over.
    #[tokio::test]
    async fn a_resume_continues_after_the_last_event_id_rather_than_repeating() {
        let app = test_app();
        let body = post_sse_raw(&app, resumable_request()).await;
        let ids = sse_field(&body, "id:");
        let datas: Vec<String> = sse_field(&body, "data:")
            .iter()
            .map(|d| d.to_string())
            .collect();
        assert!(
            ids.len() >= 3,
            "need a few events to resume into the middle"
        );
        let request_id = ids[0].rsplit_once(':').unwrap().0.to_string();

        let (status, resumed) = get_json_with_headers(
            &app,
            &format!("{}/poll", frink_api::routes::v1_stream(&request_id)),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(resumed["events"].as_array().unwrap().len(), datas.len());

        // Now from the middle, the way a reconnect would.
        let (_, tail) = get_json(
            &app,
            &format!("{}?from=2", frink_api::routes::v1_stream_poll(&request_id)),
        )
        .await;
        let tail_events: Vec<String> = tail["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["data"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(tail_events, datas[2..].to_vec());
    }

    /// Reconnecting over SSE picks up where the last id left off, with
    /// the ids still attached so a second drop can be resumed too.
    #[tokio::test]
    async fn an_sse_reconnect_resumes_from_the_last_event_id() {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let app = test_app();
        let body = post_sse_raw(&app, resumable_request()).await;
        let ids = sse_field(&body, "id:");
        let datas: Vec<String> = sse_field(&body, "data:")
            .iter()
            .map(|d| d.to_string())
            .collect();
        let request_id = ids[0].rsplit_once(':').unwrap().0.to_string();

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri(frink_api::routes::v1_stream(&request_id))
                    .header("last-event-id", format!("{request_id}:0"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("x-accel-buffering")
                .and_then(|v| v.to_str().ok()),
            Some("no"),
            "the reconnect needs the same anti-buffering header as the stream"
        );
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let resumed = String::from_utf8(bytes.to_vec()).unwrap();
        assert_eq!(
            sse_field(&resumed, "data:")
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>(),
            datas[1..].to_vec()
        );
        assert_eq!(sse_field(&resumed, "id:")[0], format!("{request_id}:1"));
    }

    /// A `Last-Event-ID` from another stream is refused rather than
    /// rounded down to zero: replaying a whole different answer would
    /// be a silent, confident lie.
    #[tokio::test]
    async fn a_last_event_id_from_another_stream_is_refused() {
        let app = test_app();
        let body = post_sse_raw(&app, resumable_request()).await;
        let request_id = sse_field(&body, "id:")[0]
            .rsplit_once(':')
            .unwrap()
            .0
            .to_string();

        let (status, err) = get_json_with_headers(
            &app,
            &frink_api::routes::v1_stream(&request_id),
            &[("last-event-id", "chatcmpl-someone-else:3")],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(err["error"]["code"], "bad_last_event_id");
    }

    /// A stream that was never resumable, or has been forgotten, is a
    /// 404 that says which -- not an empty stream that reads as an
    /// answer with no tokens in it.
    #[tokio::test]
    async fn resuming_a_stream_that_was_never_resumable_is_a_404_that_says_why() {
        let app = test_app();
        let mut request = resumable_request();
        request["stream_resumable"] = serde_json::json!(false);
        let body = post_sse_raw(&app, request).await;
        let request_id = body
            .lines()
            .find_map(|l| l.strip_prefix("data: "))
            .and_then(|d| serde_json::from_str::<serde_json::Value>(d).ok())
            .and_then(|v| v["request_id"].as_str().map(str::to_string))
            .unwrap();

        let (status, err) = get_json(&app, &frink_api::routes::v1_stream_poll(&request_id)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(err["error"]["code"], "stream_not_found");
        assert!(err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("stream_resumable"));
    }

    /// The published template and the router's pattern must describe
    /// the same path, or a client built from `frink_api::routes` asks
    /// for something this server does not serve.
    #[test]
    fn the_axum_stream_patterns_match_the_published_templates() {
        assert_eq!(
            axum_path(frink_api::routes::V1_STREAM),
            "/v1/stream/:request_id"
        );
        assert_eq!(
            axum_path(frink_api::routes::V1_STREAM_POLL),
            "/v1/stream/:request_id/poll"
        );
        assert_eq!(
            frink_api::routes::v1_stream("abc"),
            axum_path(frink_api::routes::V1_STREAM).replace(":request_id", "abc")
        );
    }

    /// Every published template goes through the converter, and what
    /// comes out has no braces left in it.
    ///
    /// The two Responses routes were mounted raw, so axum matched the
    /// literal segment `{response_id}` and a real id fell through to a
    /// bodiless 404. The test router had the same two lines, which is
    /// why nothing caught it. This walks the templates instead of
    /// naming them, so the next one added is covered without anybody
    /// remembering to come back here.
    #[test]
    fn no_published_template_reaches_the_router_with_its_braces() {
        for template in [
            frink_api::routes::V1_STREAM,
            frink_api::routes::V1_STREAM_POLL,
            frink_api::routes::V1_RESPONSE,
            frink_api::routes::V1_RESPONSE_CANCEL,
            frink_api::routes::ADMIN_TASK_CANCEL,
        ] {
            assert!(
                template.contains('{'),
                "{template} is in the template list but has no placeholder"
            );
            let mounted = axum_path(template);
            assert!(
                !mounted.contains('{') && !mounted.contains('}'),
                "{template} would be mounted as {mounted}, whose braces axum reads as a literal segment"
            );
            assert!(
                mounted.contains(':'),
                "{template} lost its placeholder entirely and would match one path only"
            );
        }
    }

    /// A real id must reach the handler, not axum's catch-all 404.
    ///
    /// The distinction is the whole point: axum answers an unmatched
    /// path with an empty body, while the handler answers an unknown id
    /// with a reasoned JSON error. Asserting on the body rather than
    /// the status is what separates "the route is missing" from "the
    /// response is not here".
    #[tokio::test]
    async fn an_unknown_response_id_gets_the_handler_not_a_bare_404() {
        let app = test_app();
        let (status, body) = get_json(&app, "/v1/responses/resp_nonexistent").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(
            !body.is_null(),
            "empty body means axum never matched the route, so the id was read as a literal segment"
        );
    }

    /// An empty task list is a list, not a missing key -- the UI renders
    /// "no jobs" from it rather than from an error.
    #[tokio::test]
    async fn the_task_list_starts_empty_rather_than_absent() {
        let app = test_app();
        let (status, body) = get_json(&app, frink_api::routes::ADMIN_TASKS).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["tasks"].as_array().unwrap().len(), 0);
    }

    /// The slots route exists, is reachable, and refuses by naming the
    /// flag that would turn it on -- rather than 404ing, which is what
    /// an unregistered route would do and is indistinguishable from
    /// "this build has no slots".
    ///
    /// The condition is reachable by default: `FRINK_SLOT_SAVE_PATH`
    /// is unset unless an operator passes `--slot-save-path`, so this
    /// is the answer every stock server gives.
    #[tokio::test]
    async fn the_slots_route_is_registered_and_refuses_by_naming_slot_save_path() {
        assert!(
            std::env::var("FRINK_SLOT_SAVE_PATH").is_err(),
            "this test asserts the unconfigured behaviour"
        );
        let app = test_app();
        let (status, body) = post_json_uri(
            &app,
            &format!("{}?action=save", frink_api::routes::slots_id(0)),
            serde_json::json!({"filename": "sys.fslot", "prompt": "hi"}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("--slot-save-path"),
            "{body}"
        );
    }

    pub(crate) async fn post_json_uri(
        app: &Router,
        uri: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({}));
        (status, json)
    }

    async fn post_json(app: &Router, body: serde_json::Value) -> serde_json::Value {
        post_json_uri(app, "/v1/chat/completions", body).await.1
    }

    /// The engine's live footprint, beside the budget it was sized
    /// against. Two things are asserted rather than the number itself,
    /// which is a property of the host: it is never a ZERO (an engine
    /// using no memory is not a thing that happens, so a zero would be
    /// a failed read presented as a fact), and it always says WHICH
    /// quantity it is -- a caller comparing a PSS figure with an RSS
    /// one is comparing two different things and will read the
    /// difference as a leak.
    #[tokio::test]
    async fn stats_says_what_the_engine_is_using_and_which_quantity_that_is() {
        let app = test_app();
        let (status, body) = get_json(&app, frink_api::routes::V1_STATS).await;
        assert_eq!(status, StatusCode::OK);

        let memory = &body["memory"];
        if memory.is_null() {
            // No `/proc`: absent is the honest answer, and the point of
            // this branch is that it is absent rather than zero.
            return;
        }
        assert!(
            memory["bytes"].as_u64().is_some_and(|b| b > 0),
            "a read that produced a zero is a broken read, not an idle \
             engine: {memory}"
        );
        assert!(
            ["pss", "rss"].contains(&memory["kind"].as_str().unwrap_or("")),
            "the quantity must travel with the number: {memory}"
        );
    }

    /// A pool this deployment does not have is reported `null`, never
    /// as a zero row. "No window pool" and "a window pool with nothing
    /// in it" are different facts, and an operator shown the second for
    /// the first sizes against a pool that does not exist. The test
    /// state runs with no shared KV pool, so all three are absent here.
    #[tokio::test]
    async fn stats_reports_a_pool_it_does_not_have_as_absent_and_not_as_zero() {
        let app = test_app();
        let (status, body) = get_json(&app, frink_api::routes::V1_STATS).await;
        assert_eq!(status, StatusCode::OK);
        for pool in ["kv_pages", "window_slots", "state_slots"] {
            assert!(
                body["pools"][pool].is_null(),
                "{pool} must be null rather than a zero row: {}",
                body["pools"]
            );
        }
    }

    /// A streamed `/v1/messages` can be cancelled only if the client
    /// can learn the id, and the Anthropic protocol has no field for
    /// it -- the `message_start` `msg_...` is a different identifier
    /// the cancel registry has never seen. So the header carries it,
    /// on the success path and on the error path alike, because a
    /// client that logs one id per call should not lose it exactly
    /// when something went wrong.
    #[tokio::test]
    async fn a_messages_response_states_the_id_that_v1_cancel_takes() {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let app = test_app();
        let send = |body: serde_json::Value| {
            let app = app.clone();
            async move {
                app.oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri(frink_api::routes::V1_MESSAGES)
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        };

        let ok = send(serde_json::json!({
            "model": "test",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .await;
        assert_eq!(ok.status(), StatusCode::OK);
        let id = ok
            .headers()
            .get("request-id")
            .expect("a served message names its id")
            .to_str()
            .unwrap()
            .to_string();
        assert!(!id.is_empty());

        // A rejected body still gets one, and a different one: two calls
        // must never collide in the ring.
        let bad = send(serde_json::json!({"model": "test"})).await;
        assert!(bad.status().is_client_error());
        let other = bad.headers().get("request-id").expect("errors too");
        assert_ne!(other.to_str().unwrap(), id);
        let _ = bad.into_body().collect().await.unwrap();
    }

    /// The gate is the point of the rebuild endpoint: a request that
    /// arrives while the KV pool is being re-split must be refused,
    /// because admitting it would let a decode allocate out of a pool
    /// whose block count is about to change under it. `503` and not
    /// `500` -- the caller should retry in a moment, and the body says
    /// which of the four closed states it hit so a client can tell
    /// "not yet" from "not ever".
    #[tokio::test]
    async fn a_request_that_arrives_mid_rebuild_is_refused_and_admitted_again_after() {
        let state = Arc::new(test_state(
            test_model_full_byte_vocab(),
            ResponseCache::new(1000, Duration::from_secs(3600)),
        ));
        let app = test_app_with_state(Arc::clone(&state));
        let body = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        });

        state
            .maintenance
            .lock()
            .unwrap()
            .begin_rebuild()
            .expect("a fresh server is serving, so the rebuild starts");
        let (status, refused) = post_json_uri(&app, "/v1/chat/completions", body.clone()).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(refused["error"]["type"], "cache_rebuilding");

        state.maintenance.lock().unwrap().finish_rebuild(true);
        let (status, _) = post_json_uri(&app, "/v1/chat/completions", body).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "the gate reopens; a rebuild is not a latch"
        );
    }

    /// Cancelling an id that is not generating must not answer `200`.
    /// A UI told "ok" for an already-finished request would report that
    /// it stopped work it did not stop, and the two outcomes are the
    /// only thing this endpoint exists to distinguish.
    #[tokio::test]
    async fn cancelling_an_id_that_is_not_generating_is_a_404_that_says_so() {
        let app = test_app();
        let (status, body) = post_json_uri(
            &app,
            frink_api::routes::V1_CANCEL,
            serde_json::json!({ "request_id": "chatcmpl-never-issued" }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["cancelled"], serde_json::json!(false));
        assert_eq!(body["request_id"], "chatcmpl-never-issued");
        assert!(
            body["detail"].as_str().is_some_and(|d| !d.is_empty()),
            "the verdict must carry a human reason: {body}"
        );
    }

    /// The endpoint reaches the registry the streaming path registers
    /// into -- not a second, parallel one. Registered by hand here
    /// because a `oneshot` router cannot hold a stream open.
    #[tokio::test]
    async fn cancelling_a_live_generation_signals_its_token_and_answers_200() {
        let state = Arc::new(test_state(
            test_model_full_byte_vocab(),
            ResponseCache::new(1000, Duration::from_secs(3600)),
        ));
        let app = test_app_with_state(Arc::clone(&state));
        let (token, _guard) = state.cancels.register("chatcmpl-live");

        let (status, before) = get_json(&app, frink_api::routes::ADMIN_STATS).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(before["generating_now"], serde_json::json!(1));

        let (status, body) = post_json_uri(
            &app,
            frink_api::routes::V1_CANCEL,
            serde_json::json!({ "request_id": "chatcmpl-live" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["cancelled"], serde_json::json!(true));
        assert!(
            token.is_cancelled(),
            "the endpoint answered ok without setting the flag the decode loop reads"
        );
    }

    #[tokio::test]
    async fn tokenize_detokenize_roundtrip_and_embeddings_mean() {
        let app = test_app();
        let (status, tok) =
            post_json_uri(&app, "/v1/tokenize", serde_json::json!({ "prompt": "Hi" })).await;
        assert_eq!(status, StatusCode::OK);
        let tokens = tok["tokens"].as_array().unwrap();
        assert_eq!(tok["count"], tokens.len());
        assert!(!tokens.is_empty());

        let (status, detok) = post_json_uri(
            &app,
            "/v1/detokenize",
            serde_json::json!({ "tokens": tokens }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(detok["text"], "Hi");

        let (status, emb) = post_json_uri(
            &app,
            "/v1/embeddings",
            serde_json::json!({
                "input": "Hi",
                "embedding_type": "mean"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let vec = emb["data"][0]["embedding"].as_array().unwrap();
        assert!(!vec.is_empty());
        assert!(vec.iter().all(|v| v.as_f64().is_some()));
    }

    /// The decoder path's accepted `embedding_type` set must not have
    /// widened when the encoder path arrived: `cls` is row 0 of a
    /// decoder's hidden states, which is its BOS position and means
    /// nothing, so it stays refused here and the refusal names what is
    /// accepted.
    #[tokio::test]
    async fn the_decoder_path_still_refuses_a_pooling_it_cannot_mean() {
        let app = test_app();
        let (status, body) = post_json_uri(
            &app,
            "/v1/embeddings",
            serde_json::json!({ "input": "Hi", "embedding_type": "cls" }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("mean") && msg.contains("last"), "{msg}");
    }

    /// A real BGE checkpoint served through the route: CLS by default
    /// because the file says `pooling_type = 2`, 384 dims, unit norm,
    /// and `usage.prompt_tokens` counting the `[CLS]`/`[SEP]` the model
    /// actually saw.
    #[tokio::test]
    #[ignore = "needs models/bge-small-en-v1.5-q8_0.gguf"]
    async fn a_real_embedding_model_serves_v1_embeddings() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/bge-small-en-v1.5-q8_0.gguf");
        if !path.exists() {
            eprintln!("SKIP: {} not present", path.display());
            return;
        }
        let encoder = frink_models::EmbeddingModel::from_gguf_path(&path).expect("load bge");
        let mut state = test_state(
            test_model_full_byte_vocab(),
            ResponseCache::new(1000, Duration::from_secs(3600)),
        );
        state.embedding = Some(Arc::new(encoder));
        let app = test_app_with_state(Arc::new(state));

        let (status, body) = post_json_uri(
            &app,
            "/v1/embeddings",
            serde_json::json!({ "input": ["Hello world", "a second input"] }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["model"], "bge-small-en-v1.5");
        let data = body["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        for (i, row) in data.iter().enumerate() {
            assert_eq!(row["index"], i);
            let v: Vec<f64> = row["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            assert_eq!(v.len(), 384, "the encoder\'s width, not the decoder\'s");
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "not L2-normalized: {norm}");
        }
        // "Hello world" is [CLS] hello world [SEP] = 4, and the second
        // input adds its own two specials.
        assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() >= 4 + 2);

        // The default came from the file. Asking for MEAN must give a
        // different vector, which is what proves CLS was not a
        // coincidence of this input.
        let (status, mean) = post_json_uri(
            &app,
            "/v1/embeddings",
            serde_json::json!({ "input": "Hello world", "embedding_type": "mean" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_ne!(mean["data"][0]["embedding"], data[0]["embedding"]);
    }

    /// The same BGE checkpoint as `FRINK_MODEL_PATH` -- the *loaded*
    /// model, not a side-car.
    ///
    /// Four claims, and the third is the one this whole seam exists
    /// for: the loader routes an encoder-only GGUF away from every
    /// decoder path, `/v1/embeddings` serves it, `/v1/chat/completions`
    /// refuses it NAMING IT AS AN EMBEDDING MODEL (before this, the
    /// same file died in `tokenizer_from_gguf` with a message about
    /// WordPiece being unreadable -- true, and the wrong thing to send
    /// a user after), and `/v1/models` says which endpoint it is for so
    /// a client need not send a request to find out.
    #[tokio::test]
    #[ignore = "needs models/bge-small-en-v1.5-q8_0.gguf"]
    async fn an_encoder_can_be_the_loaded_model() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/bge-small-en-v1.5-q8_0.gguf");
        if !path.exists() {
            eprintln!("SKIP: {} not present", path.display());
            return;
        }

        // Through the real `FRINK_MODEL_PATH` loader, not by
        // constructing an `EmbeddingModel` directly: the routing
        // decision is half of what is under test.
        let loaded = model::load_from_path(path.to_str().unwrap()).expect("load bge as the model");
        assert!(
            matches!(loaded, model::LoadedModel::Encoder(_)),
            "an encoder-only GGUF reached a decoder loader"
        );
        let (loaded, batcher, ceiling) = activate_loaded_model(loaded, true, None, None);
        assert!(
            matches!(loaded, Loaded::Encoder(_)),
            "the encoder did not stay an encoder through activation"
        );
        assert!(
            batcher.is_none() && ceiling.is_none(),
            "an encoder was given a decode batcher or a KV ceiling it has no use for"
        );

        let state = test_state(
            test_model_full_byte_vocab(),
            ResponseCache::new(1000, Duration::from_secs(3600)),
        );
        state.swap_active(Some(Arc::new(ActiveModel {
            id: None,
            loaded,
            batcher,
            ceiling,
            checkpoint_path: None,
        })));
        let app = test_app_with_state(Arc::new(state));

        // 1. It embeds.
        let (status, body) = post_json_uri(
            &app,
            "/v1/embeddings",
            serde_json::json!({ "input": "Hello world" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["model"], "bge-small-en-v1.5");
        let v = body["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(v.len(), 384, "the encoder's width, not the decoder's");

        // 2. It refuses to chat, by name.
        let (status, body) = post_json_uri(
            &app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "bge-small-en-v1.5",
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
        let msg = body["error"]["message"].as_str().unwrap();
        for fact in [
            "bge-small-en-v1.5",
            "bert",
            "embedding model",
            "/v1/embeddings",
        ] {
            assert!(msg.contains(fact), "the refusal does not say {fact}: {msg}");
        }

        // 3. `/v1/models` lists it as what it is.
        let (status, models) = get_json(&app, frink_api::routes::V1_MODELS).await;
        assert_eq!(status, StatusCode::OK);
        let entry = &models["data"][0];
        assert_eq!(entry["id"], "bge-small-en-v1.5");
        assert_eq!(entry["frink_model_kind"], "embedding");
        assert_eq!(entry["frink_tokenizer"], "gguf-wordpiece");
        assert_eq!(entry["frink_n_embd"], 384);
        assert_eq!(entry["frink_pooling"], "CLS");
        assert_eq!(
            entry["frink_endpoints"],
            serde_json::json!(["/v1/embeddings"])
        );
        // A reasoning-gear field here would be an invented answer about
        // a template the checkpoint does not have.
        assert!(entry.get("supported_reasoning_efforts").is_none());

        // 4. `/health` is ready, and says which endpoint is ready.
        let (status, health) = get_json(&app, frink_api::routes::HEALTH).await;
        assert_eq!(status, StatusCode::OK, "an encoder is a loaded model");
        assert_eq!(health["model"]["id"], "bge-small-en-v1.5");
        assert_eq!(health["model"]["synthetic_weights"], false);
        let weights = health["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == frink_api::health::capability::REAL_WEIGHTS)
            .expect("a real-weights capability row");
        let detail = weights["detail"].as_str().unwrap_or_default();
        assert!(detail.contains("ENCODER"), "{detail}");
        // 5. It tokenizes, and round-trips. An embedding model's whole
        // contract is the vector it returns for a string, so when that
        // vector surprises you the first question is what tokens it
        // actually saw. These routes used to go through
        // `generative()?` and answer 501 "not a generative model",
        // which left no way to ask without loading the checkpoint in a
        // second tool (issue #28).
        let (status, body) = post_json_uri(
            &app,
            frink_api::routes::V1_TOKENIZE,
            serde_json::json!({ "content": "hello world" }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "an encoder has a real tokenizer: {body}"
        );
        let tokens = body["tokens"].as_array().expect("tokens array").clone();
        assert!(!tokens.is_empty(), "WordPiece produced nothing: {body}");

        let (status, body) = post_json_uri(
            &app,
            frink_api::routes::V1_DETOKENIZE,
            serde_json::json!({ "tokens": tokens }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let round_tripped = body["content"].as_str().expect("content").to_string();
        assert!(
            round_tripped.contains("hello") && round_tripped.contains("world"),
            "the ids did not decode back through the encoder's own vocabulary: {round_tripped}"
        );

        // And the refusal that must NOT have been weakened: a decode is
        // still a decode, and this checkpoint still cannot do one.
        let (status, _) = post_json_uri(
            &app,
            "/v1/completions",
            serde_json::json!({ "model": "m", "prompt": "hi", "max_tokens": 1 }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::NOT_IMPLEMENTED,
            "tokenizing an encoder must not have opened a path to generating with one"
        );
    }

    /// The /metrics endpoint must expose the bounded expert cache's
    /// counters when the model streams routed experts, and the
    /// counters must reflect real decode activity (a forward pass
    /// through store-backed MoE layers produces misses/hits).
    #[tokio::test]
    async fn metrics_exposes_expert_store_counters_when_streaming_is_active() {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let fixture = concat!(
            "../frink-models/tests/fixtures/",
            "frink_real_moe_test.gguf"
        );
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(fixture);
        let decoder = Decoder::from_gguf_with_expert_cache(
            &fixture,
            frink_models::config::test_moe_fixture(),
            Some(1024 * 1024),
        )
        .expect("MoE fixture must load store-backed");

        // Drive one real forward pass so the store sees decode
        // activity (the fixture's tiny vocab can't survive the HTTP
        // path's template text, so decode directly).
        let mut caches: Vec<frink_core::cache::KvCache> = decoder.config.new_kv_caches();
        decoder.forward_token(1, 0, &mut caches);

        let model = Model::Gguf(GgufModel {
            decoder: Arc::new(decoder),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: None,
            is_synthetic: false,
            chat_template: chat_template::PromptTemplate::plain(),
        });
        let state = Arc::new(test_state(
            model,
            ResponseCache::new(16, Duration::from_secs(60)),
        ));
        let app = Router::new()
            .route("/metrics", axum::routing::get(metrics))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);

        let fetch_metrics = |app: Router| async move {
            let resp = app
                .oneshot(
                    axum::http::Request::builder()
                        .method("GET")
                        .uri("/metrics")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            String::from_utf8(bytes.to_vec()).unwrap()
        };

        let after = fetch_metrics(app.clone()).await;
        assert!(
            after.contains("frink_expert_cache_misses_total"),
            "streaming model must expose expert-cache metrics: {after}"
        );
        let misses: u64 = after
            .lines()
            .find(|l| l.starts_with("frink_expert_cache_misses_total"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .expect("misses metric line must parse");
        assert!(
            misses > 0,
            "decode must have read experts through the store: {after}"
        );
    }

    fn weather_tool() -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        })
    }

    fn weather_tool_def() -> ToolDef {
        ToolDef {
            kind: "function".to_string(),
            function: ToolFunctionDef {
                name: "get_weather".to_string(),
                description: Some("Get the current weather for a location.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                })),
            },
        }
    }

    #[test]
    fn tool_preamble_mentions_every_tool_name_and_description() {
        let preamble = tool_preamble(&[weather_tool_def()]);
        assert!(preamble.contains("get_weather"));
        assert!(preamble.contains("Get the current weather for a location."));
        assert!(preamble.contains("<tool_call>"));
        assert!(preamble.contains("</tool_call>"));
    }

    #[test]
    fn a_real_marker_becomes_a_structured_tool_call() {
        let text = "sure, let me check.<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}</tool_call>";
        let (message, finish) = build_response_message(
            text.to_string(),
            &[weather_tool_def()],
            output::OutputPosture::for_model("test-model"),
            "stop",
        );
        assert_eq!(finish, "tool_calls");
        let calls = message.tool_calls.expect("must carry a tool call");
        assert_eq!(calls[0].function.name, "get_weather");
        let parsed: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(parsed["location"], "Paris");
    }

    #[test]
    fn a_plain_answer_is_not_promoted_to_a_tool_call() {
        let (message, finish) = build_response_message(
            "just an answer".to_string(),
            &[weather_tool_def()],
            output::OutputPosture::for_model("test-model"),
            "stop",
        );
        assert_eq!(finish, "stop");
        assert!(message.tool_calls.is_none());
        assert_eq!(message.content.as_deref(), Some("just an answer"));
    }

    /// Malformed JSON inside the marker is not a call. Returning it as
    /// one would hand a client arguments it cannot parse.
    #[test]
    fn a_malformed_payload_is_not_a_tool_call() {
        let (message, finish) = build_response_message(
            "<tool_call>not valid json at all</tool_call>".to_string(),
            &[weather_tool_def()],
            output::OutputPosture::for_model("test-model"),
            "stop",
        );
        assert_eq!(finish, "stop");
        assert!(message.tool_calls.is_none());
    }

    /// A call to something the request never offered is refused: the
    /// client would be asked to execute a tool it does not have.
    #[test]
    fn a_tool_that_was_never_offered_is_not_returned() {
        let (message, finish) = build_response_message(
            "<tool_call>{\"name\": \"ping\", \"arguments\": {}}</tool_call>".to_string(),
            &[weather_tool_def()],
            output::OutputPosture::for_model("test-model"),
            "stop",
        );
        assert_eq!(finish, "stop");
        assert!(message.tool_calls.is_none());
    }

    /// With no tools offered at all, marker text is just text.
    #[test]
    fn marker_text_with_no_tools_offered_stays_content() {
        let (message, finish) = build_response_message(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>".to_string(),
            &[],
            output::OutputPosture::for_model("test-model"),
            "stop",
        );
        assert_eq!(finish, "stop");
        assert!(message.tool_calls.is_none());
        assert!(message.content.is_some());
    }

    /// The streaming contract a coding agent depends on: the call's
    /// identity arrives first, then its arguments in pieces, and the
    /// pieces concatenate to exactly the final arguments.
    #[test]
    fn a_streamed_call_opens_then_delivers_its_arguments_in_pieces() {
        let opened = std::cell::Cell::new(0usize);
        let mut parser = crate::policy::parser::ToolCallParser::new(
            crate::policy::parser::ToolCallFormat::Qwen3Coder,
            vec![
                crate::policy::parser::tool_call::ToolSchema::with_parameters(
                    "write_file",
                    serde_json::json!({"type": "object", "properties": {
                        "path": {"type": "string"},
                        "contents": {"type": "string"}
                    }}),
                ),
            ],
        );
        let wire = "<tool_call><function=write_file>\
                    <parameter=path>\n/tmp/x\n</parameter>\
                    <parameter=contents>\nhello world\n</parameter>\
                    </function></tool_call>";

        let mut deltas = Vec::new();
        let mut text = String::new();
        for piece in wire.as_bytes().chunks(7) {
            let chunk = String::from_utf8_lossy(piece).into_owned();
            let (more_text, more) = tool_call_deltas(parser.push(&chunk), &opened);
            text.push_str(&more_text);
            deltas.extend(more);
        }
        let (more_text, more) = tool_call_deltas(parser.finish(), &opened);
        text.push_str(&more_text);
        deltas.extend(more);

        assert_eq!(opened.get(), 1, "one call opened");
        assert!(text.is_empty(), "the markers are not content: {text:?}");

        let first = &deltas[0];
        assert_eq!(first.index, 0);
        assert_eq!(first.id.as_deref(), Some("call_0"));
        assert_eq!(first.kind, Some("function"));
        assert_eq!(first.function.name.as_deref(), Some("write_file"));

        // Everything after the opening delta is argument text only,
        // and it parses once concatenated.
        let joined: String = deltas
            .iter()
            .filter_map(|d| d.function.arguments.clone())
            .collect();
        let parsed: serde_json::Value =
            serde_json::from_str(&joined).expect("the fragments concatenate to valid JSON");
        assert_eq!(parsed["path"], serde_json::json!("/tmp/x"));
        assert_eq!(parsed["contents"], serde_json::json!("hello world"));
        assert!(
            deltas.len() >= 3,
            "the arguments arrived in pieces, not whole: {}",
            deltas.len()
        );
        assert!(
            deltas[1..].iter().all(|d| d.function.name.is_none()),
            "only the opening delta carries identity"
        );
    }

    /// Text either side of a call still streams as content, in order.
    #[test]
    fn text_around_a_streamed_call_is_still_content() {
        let opened = std::cell::Cell::new(0usize);
        let mut parser = crate::policy::parser::ToolCallParser::new(
            crate::policy::parser::ToolCallFormat::Qwen25,
            vec![crate::policy::parser::tool_call::ToolSchema::new(
                "get_weather",
            )],
        );
        let wire = "let me check. <tool_call>{\"name\": \"get_weather\", \
                    \"arguments\": {}}</tool_call> done";
        let mut text = String::new();
        for piece in wire.as_bytes().chunks(5) {
            let chunk = String::from_utf8_lossy(piece).into_owned();
            let (more, _) = tool_call_deltas(parser.push(&chunk), &opened);
            text.push_str(&more);
        }
        let (more, _) = tool_call_deltas(parser.finish(), &opened);
        text.push_str(&more);

        assert_eq!(opened.get(), 1);
        assert!(text.starts_with("let me check. "), "{text:?}");
        assert!(text.ends_with(" done"), "{text:?}");
        assert!(!text.contains("<tool_call>"), "markers leaked: {text:?}");
    }

    /// A reasoning model's thinking must not be returned as its
    /// answer.
    #[test]
    fn a_reasoning_block_is_split_out_of_the_answer() {
        let (message, finish) = build_response_message(
            "<think>weighing it up</think>The answer is 4.".to_string(),
            &[],
            output::OutputPosture::for_model("Qwen3-8B"),
            "stop",
        );
        assert_eq!(finish, "stop");
        assert_eq!(message.content.as_deref(), Some("The answer is 4."));
        assert_eq!(message.reasoning_content.as_deref(), Some("weighing it up"));
    }

    /// ... and a model with no reasoning format keeps its text intact,
    /// markers and all.
    #[test]
    fn a_non_reasoning_model_keeps_a_literal_marker_in_its_answer() {
        let (message, _) = build_response_message(
            "Use the <think> tag like this.".to_string(),
            &[],
            output::OutputPosture::for_model("llama-3.1-8b"),
            "stop",
        );
        assert_eq!(
            message.content.as_deref(),
            Some("Use the <think> tag like this.")
        );
        assert!(message.reasoning_content.is_none());
    }

    /// Zero-regression proof: an ordinary request with no `tools`/
    /// `session_id` produces the plain response shape -- `content` a
    /// string, no `tool_calls` field -- with an honest finish reason:
    /// this 4-token greedy request truncates at `max_tokens`, so
    /// `finish_reason` must be "length" (an earlier version hardcoded
    /// "stop" for every non-streaming response), and `usage` counts
    /// exactly the generated tokens.
    #[tokio::test]
    async fn a_request_with_no_tools_or_session_behaves_exactly_as_before() {
        let app = test_app();
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
            "max_tokens": 4,
            "temperature": 0,
        });
        let resp = post_json(&app, body).await;
        let message = &resp["choices"][0]["message"];
        assert!(message["content"].is_string());
        assert!(message.get("tool_calls").is_none());
        assert_eq!(resp["choices"][0]["finish_reason"], "length");
        assert_eq!(resp["usage"]["completion_tokens"], 4);
        assert_eq!(
            resp["usage"]["total_tokens"],
            resp["usage"]["prompt_tokens"].as_u64().unwrap() + 4
        );
    }

    pub(crate) async fn get_json(app: &Router, uri: &str) -> (StatusCode, serde_json::Value) {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn health_answers_a_capability_handshake_not_a_boolean() {
        let app = test_app();
        let (status, body) = get_json(&app, frink_api::routes::HEALTH).await;
        assert_eq!(status, StatusCode::OK);

        let health: frink_api::HealthResponse = serde_json::from_value(body).unwrap();
        assert_eq!(health.state, frink_api::HealthState::Ready);
        assert!(health.pid > 0);
        assert!(health.server_time_unix_ms > 0);
        // Nothing has been served yet: the field is absent rather than
        // claiming a request happened at time zero.
        assert_eq!(health.last_request_age_seconds, None);

        // Every control the UI might grey out has a code it can switch
        // on and a sentence it can show.
        for id in [
            frink_api::health::capability::CPU,
            frink_api::health::capability::METAL,
            frink_api::health::capability::CUDA,
            frink_api::health::capability::REAL_WEIGHTS,
            frink_api::health::capability::CONTINUOUS_BATCHING,
        ] {
            let cap = health
                .capability(id)
                .unwrap_or_else(|| panic!("{id} missing"));
            assert!(!cap.reason.is_empty(), "{cap:?}");
            assert!(!cap.detail.is_empty(), "{cap:?}");
        }
        // The test app serves synthetic random weights, and health must
        // say so: a UI that presents noise as a model invites a bug
        // report about "quality".
        let weights = health
            .capability(frink_api::health::capability::REAL_WEIGHTS)
            .unwrap();
        assert!(!weights.available);
        assert_eq!(weights.reason, frink_api::health::reason::MODEL_NOT_LOADED);
        assert!(health.model.as_ref().unwrap().synthetic_weights);
    }

    #[tokio::test]
    async fn health_vouches_for_liveness_after_a_request_has_been_served() {
        let app = test_app();
        let _ = post_json(
            &app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}"}],
                "max_tokens": 1,
                "temperature": 0,
            }),
        )
        .await;
        let (_status, body) = get_json(&app, frink_api::routes::HEALTH).await;
        let health: frink_api::HealthResponse = serde_json::from_value(body).unwrap();
        let age = health
            .last_request_age_seconds
            .expect("a served request is evidence of liveness");
        assert!((0.0..5.0).contains(&age), "implausible age {age}");
    }

    /// Every `data:` payload of an SSE response body, `[DONE]` excluded.
    async fn post_sse_chunks(app: &Router, body: serde_json::Value) -> Vec<serde_json::Value> {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|payload| *payload != "[DONE]")
            .map(|payload| serde_json::from_str(payload).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn a_stream_states_its_request_id_once_in_the_first_chunk() {
        let app = test_app();
        let chunks = post_sse_chunks(
            &app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": true,
            }),
        )
        .await;

        assert!(!chunks.is_empty());
        let request_id = chunks[0]["request_id"]
            .as_str()
            .expect("the first chunk names the request")
            .to_string();
        assert!(request_id.starts_with("chatcmpl-"), "{request_id}");
        // Once, and before any content: a client that reads the id from
        // chunk zero never has to correlate by heuristic.
        for (i, chunk) in chunks.iter().enumerate().skip(1) {
            assert!(
                chunk.get("request_id").is_none(),
                "chunk {i} repeats request_id"
            );
        }
        // Every chunk of one stream carries the same `id`, and it is
        // that request id -- not a shared constant.
        for chunk in &chunks {
            assert_eq!(chunk["id"], serde_json::json!(request_id));
        }

        let other = post_sse_chunks(
            &app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": true,
            }),
        )
        .await;
        assert_ne!(
            other[0]["request_id"].as_str().unwrap(),
            request_id,
            "two concurrent chats must not share an id"
        );
    }

    #[tokio::test]
    async fn a_non_streamed_response_names_the_same_request_id_as_its_completion_id() {
        let app = test_app();
        let resp = post_json(
            &app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "max_tokens": 2,
                "temperature": 0,
            }),
        )
        .await;
        assert_eq!(resp["id"], resp["request_id"]);
        assert!(resp["request_id"]
            .as_str()
            .unwrap()
            .starts_with("chatcmpl-"));
    }

    /// The whole point of server-reported timings: a client can tell
    /// prefill from decode without a stopwatch (see `frink_api::usage`).
    #[tokio::test]
    async fn usage_carries_separate_prefill_and_decode_timings() {
        let app = test_app();
        let resp = post_json(
            &app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "max_tokens": 4,
                "temperature": 0,
            }),
        )
        .await;
        let usage = &resp["usage"];
        assert!(usage["prompt_eval_duration_ms"].is_number(), "{usage}");
        assert!(usage["generation_duration_ms"].is_number(), "{usage}");
        assert!(usage["time_to_first_token_ms"].is_number(), "{usage}");
        assert!(usage["predicted_per_second"].is_number(), "{usage}");
        // No prefix cache in this app: the field must be absent, not 0.
        assert!(usage.get("cached_tokens").is_none(), "{usage}");
    }

    /// A real, deterministic small model with random weights will not
    /// spontaneously produce a `<tool_call>{...}</tool_call>` marker
    /// (whether a real deployed model does is a property of that
    /// model, not of frink's plumbing) -- so the real, testable
    /// end-to-end property here is that a `tools`-bearing request
    /// whose output does NOT contain the marker falls through cleanly
    /// to an ordinary text response instead of erroring or panicking.
    #[tokio::test]
    async fn a_tools_request_with_no_marker_in_the_output_falls_back_to_plain_content() {
        let app = test_app();
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
            "max_tokens": 4,
            "temperature": 0,
            "tools": [weather_tool()],
        });
        let resp = post_json(&app, body).await;
        let message = &resp["choices"][0]["message"];
        assert!(
            message["content"].is_string(),
            "must fall back to plain content when no real tool-call marker is present: {resp:?}"
        );
        assert!(message.get("tool_calls").is_none());
        // Truncated at max_tokens, so the honest finish reason is
        // "length" -- the point here is only that it is NOT
        // "tool_calls".
        assert_eq!(resp["choices"][0]["finish_reason"], "length");
    }

    /// A whole-response cache hit must be indistinguishable from
    /// recomputing: same content, same (honest) finish_reason, same
    /// usage counts -- only the `frink_cache` marker may differ.
    #[tokio::test]
    async fn a_cache_hit_reports_the_original_finish_reason_and_usage() {
        let app = test_app();
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "\u{1}\u{2}"}],
            "max_tokens": 3,
            "temperature": 0,
        });
        let first = post_json(&app, body.clone()).await;
        assert_eq!(first["frink_cache"], "miss");
        let second = post_json(&app, body).await;
        assert_eq!(second["frink_cache"], "hit");
        assert_eq!(
            first["choices"][0]["message"]["content"],
            second["choices"][0]["message"]["content"]
        );
        assert_eq!(
            first["choices"][0]["finish_reason"],
            second["choices"][0]["finish_reason"]
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(second["usage"]["completion_tokens"], 3);
    }

    /// The whole of #35 through the real router: a request that adds a
    /// GRAMMAR to a body already answered without one must be generated
    /// afresh, under that grammar.
    ///
    /// The cache used to be consulted before
    /// `generation_params_for_template` had even compiled the grammar,
    /// and the key held no trace of it, so the constrained request was
    /// handed the previous caller's unconstrained prose with a 200. The
    /// answer is asserted, not the key: a key that differs proves
    /// nothing if the lookup uses something else.
    #[tokio::test]
    async fn a_grammar_request_is_not_answered_from_an_unconstrained_cache_entry() {
        let app = test_app();
        let plain = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "\u{1}\u{2}"}],
            "max_tokens": 3,
            "temperature": 0,
        });

        let first = post_json(&app, plain.clone()).await;
        assert_eq!(first["frink_cache"], "miss");
        let unconstrained = first["choices"][0]["message"]["content"]
            .as_str()
            .expect("content")
            .to_string();

        let mut constrained = plain.clone();
        constrained["grammar"] = serde_json::json!("root ::= \"yes\"");
        let second = post_json(&app, constrained).await;
        assert_eq!(
            second["frink_cache"], "miss",
            "a grammar is part of the key, so this body has never been answered"
        );
        // The synthetic demo model wraps its decode in a banner, so the
        // assertion is on the decoded text inside it: `yes` is the only
        // string this grammar admits, and it is there.
        let constrained_answer = second["choices"][0]["message"]["content"]
            .as_str()
            .expect("content")
            .to_string();
        assert!(
            constrained_answer.contains("-> \"yes\"]"),
            "the grammar must have been compiled AND applied, not skipped \
             by a cache hit: {constrained_answer}"
        );
        assert_ne!(
            constrained_answer, unconstrained,
            "the constrained request was served the unconstrained answer"
        );

        // And the entry the first request made is still the first
        // request's: the miss above is the grammar, not a key that
        // fails to repeat.
        let third = post_json(&app, plain).await;
        assert_eq!(third["frink_cache"], "hit");
        assert_eq!(third["choices"][0]["message"]["content"], unconstrained);
    }

    /// The third of #35's fields, and the one whose old failure was
    /// LOUD: `validate_json_object_output` runs against whatever came
    /// back, so a `json_object` request answered from a cached prose
    /// entry got a hard 400 for a body that had never been generated
    /// under the JSON mask at all.
    ///
    /// The system message is what makes this reproducible, and it is the
    /// repo's own bug shape underneath. `inject_json_object_system_hint`
    /// usually leaves a fingerprint in the PROMPT, which happened to
    /// split the two keys apart -- a correctness property nothing stated
    /// or enforced, resting on a string edit made for a different
    /// reason. Its `!s.contains("JSON")` arm is the hole: a caller who
    /// already says "JSON" in their own system message gets NO hint
    /// appended, so the two requests render byte-identical prompts and
    /// the old key could not tell them apart.
    ///
    /// The synthetic model emits its demo banner under either mask, so
    /// the 400 is the same on both sides of this fix and cannot be the
    /// assertion; the cache-level twin in `response_cache` asserts the
    /// answer. What is asserted here is that the answer did not come
    /// from the other request's entry.
    #[tokio::test]
    async fn a_json_object_request_does_not_reuse_the_unconstrained_cache_entry() {
        let state = Arc::new(test_state(
            test_model_full_byte_vocab(),
            ResponseCache::new(1000, Duration::from_secs(3600)),
        ));
        let app = test_app_with_state(state.clone());
        let plain = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "Answer in JSON when it helps."},
                {"role": "user", "content": "\u{1}\u{2}"},
            ],
            "max_tokens": 3,
            "temperature": 0,
        });

        let first = post_json(&app, plain.clone()).await;
        assert_eq!(first["frink_cache"], "miss");
        assert_eq!(state.cache_stats().entries, 1);

        let mut as_json = plain.clone();
        as_json["response_format"] = serde_json::json!({"type": "json_object"});
        let (status, _) = post_json_uri(&app, "/v1/chat/completions", as_json).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "the demo banner is not a JSON object, whoever generated it"
        );
        assert_eq!(
            state.cache_stats().hits,
            0,
            "a json_object request must not be answered from an entry the \
             JSON mask never produced"
        );
        assert_eq!(
            state.cache_stats().entries,
            2,
            "json_object must key its own entry, not reuse the unconstrained \
             one it happens to render the same prompt as"
        );
    }

    /// The same failure for `ignore_eos`, whose whole purpose is that a
    /// benchmarking run produces EXACTLY `max_tokens`. Answered from a
    /// cache entry the model's own EOS had cut short, it produced the
    /// short answer instead -- the one outcome the field exists to rule
    /// out (#35).
    ///
    /// `0x77` is the id this model greedily emits SECOND for the prompt
    /// below, so with it as the EOS the plain request stops after one
    /// token and the `ignore_eos` one runs the whole budget. Asserted on
    /// the token count and the finish reason, which is where a replayed
    /// answer shows.
    #[tokio::test]
    async fn an_ignore_eos_request_is_not_answered_from_a_cache_entry_that_stopped_at_eos() {
        let app = test_app_with_state(Arc::new(test_state(
            test_model_full_byte_vocab_with_eos(Some(0x77)),
            ResponseCache::new(1000, Duration::from_secs(3600)),
        )));
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "\u{1}\u{2}"}],
            "max_tokens": 6,
            "temperature": 0,
        });

        let stopped = post_json(&app, body.clone()).await;
        assert_eq!(stopped["frink_cache"], "miss");
        assert_eq!(
            stopped["choices"][0]["finish_reason"], "stop",
            "the fixture is only meaningful if the model's EOS really fires here"
        );
        assert_eq!(stopped["usage"]["completion_tokens"], 1);

        let mut ignoring = body.clone();
        ignoring["ignore_eos"] = serde_json::json!(true);
        let ran_on = post_json(&app, ignoring).await;
        assert_eq!(
            ran_on["frink_cache"], "miss",
            "ignore_eos is part of the key, so this body has never been answered"
        );
        assert_eq!(
            ran_on["usage"]["completion_tokens"], 6,
            "ignore_eos must run the full budget, not replay the EOS-terminated answer"
        );
        assert_eq!(ran_on["choices"][0]["finish_reason"], "length");
        assert_ne!(
            ran_on["choices"][0]["message"]["content"],
            stopped["choices"][0]["message"]["content"]
        );
    }

    /// The real proof for session reuse:
    /// a two-request session where the second request sends only its
    /// new message must produce exactly the same output as manually
    /// resending the full history (built from the *real* first reply,
    /// not an assumed one) with no `session_id` at all.
    #[tokio::test]
    async fn session_reuse_produces_the_same_output_as_manually_resending_full_history() {
        let session_app = test_app();
        let manual_app = test_app();

        // Turn 1, via session.
        let turn1 = post_json(
            &session_app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "session_id": "s1",
                "max_tokens": 5,
                "temperature": 0,
            }),
        )
        .await;
        let reply1 = turn1["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();

        // Turn 1, manually, for comparison -- must match exactly
        // (trivially, since it's the literal same single-turn
        // request), confirming the session path's first turn isn't
        // doing anything different from a plain request.
        let manual_turn1 = post_json(
            &manual_app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{1}\u{2}\u{3}"}],
                "max_tokens": 5,
                "temperature": 0,
            }),
        )
        .await;
        assert_eq!(
            manual_turn1["choices"][0]["message"]["content"]
                .as_str()
                .unwrap(),
            reply1
        );

        // Turn 2, via session: sends ONLY the new message.
        let turn2 = post_json(
            &session_app,
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "\u{4}\u{5}"}],
                "session_id": "s1",
                "max_tokens": 5,
                "temperature": 0,
            }),
        )
        .await;
        let reply2 = turn2["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();

        // Turn 2, manually: the full three-message history
        // reconstructed using the REAL reply1 text, with no
        // session_id -- must produce byte-identical output.
        let manual_turn2 = post_json(
            &manual_app,
            serde_json::json!({
                "model": "m",
                "messages": [
                    {"role": "user", "content": "\u{1}\u{2}\u{3}"},
                    {"role": "assistant", "content": reply1},
                    {"role": "user", "content": "\u{4}\u{5}"},
                ],
                "max_tokens": 5,
                "temperature": 0,
            }),
        )
        .await;
        assert_eq!(
            manual_turn2["choices"][0]["message"]["content"]
                .as_str()
                .unwrap(),
            reply2,
            "resuming a session must produce identical output to manually resending the full history"
        );
    }

    /// `lock_cache` must return a usable guard even after the mutex was
    /// poisoned by a panic elsewhere.
    #[test]
    fn lock_cache_recovers_from_a_poisoned_mutex() {
        let cache = Arc::new(Mutex::new(ResponseCache::new(10, Duration::from_secs(60))));

        let poison_cache = Arc::clone(&cache);
        let _ = std::thread::spawn(move || {
            let _guard = poison_cache.lock().unwrap();
            panic!("simulated panic while holding the lock");
        })
        .join();

        // A plain `.lock().unwrap()` would panic here; lock_cache must not.
        let recovered = lock_cache(&cache);
        assert_eq!(recovered.stats().entries, 0);
    }

    #[test]
    fn is_cacheable_true_for_greedy_or_seeded_requests() {
        let mut req_body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let req: ChatCompletionRequest = serde_json::from_value(req_body.clone()).unwrap();
        assert!(
            req.is_cacheable(),
            "default (temperature 0) must be cacheable"
        );

        req_body["temperature"] = serde_json::json!(0.8);
        let req: ChatCompletionRequest = serde_json::from_value(req_body.clone()).unwrap();
        assert!(
            !req.is_cacheable(),
            "unseeded sampling must never be cacheable"
        );

        req_body["seed"] = serde_json::json!(42);
        let req: ChatCompletionRequest = serde_json::from_value(req_body).unwrap();
        assert!(
            req.is_cacheable(),
            "sampling with an explicit seed is deterministic and must be cacheable"
        );
    }

    /// A template that grades only the OpenAI triple. `raise_exception`
    /// is how a real one rejects a value it does not know, which is what
    /// makes the load-time probe able to learn the vocabulary at all.
    const GRADED: &str = "{% if reasoning_effort %}\
         {% if reasoning_effort not in ['low','medium','high'] %}\
           {{ raise_exception('unsupported effort') }}\
         {% endif %}E:{{ reasoning_effort }}|{% endif %}\
         {% if enable_thinking %}THINK|{% endif %}{{ messages[0].content }}";

    fn graded_template() -> chat_template::PromptTemplate {
        chat_template::PromptTemplate::from_gguf_metadata(
            Some(GRADED),
            Some("qwen3"),
            false,
            true,
            None,
            None,
        )
    }

    fn chat_request(value: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(value).expect("request")
    }

    /// The wire field reaches the sampler, compiled.
    ///
    /// Serde is the failure mode here, not the grammar engine: an
    /// undeclared field is dropped silently and the caller is served
    /// unconstrained text with a 200, which is exactly why `logit_bias`
    /// is declared on this struct only to be refused by name.
    #[test]
    fn a_grammar_on_the_chat_wire_reaches_the_generation_params() {
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": "root ::= \"a\"+",
        }));
        req.validate_supported_fields()
            .expect("a valid grammar is a valid request");
        let params = req
            .generation_params(crate::sampling_knobs::SamplerModel::absent())
            .expect("a valid grammar compiles at params time too");
        assert!(
            params.grammar.is_some(),
            "the grammar was dropped between the wire and the sampler"
        );
        assert!(
            params.needs_vocab_logits(),
            "a grammar request that may fold lm_head into a GPU argmax is \
             a grammar request served unconstrained"
        );

        let plain = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert!(plain
            .generation_params(crate::sampling_knobs::SamplerModel::absent())
            .unwrap()
            .grammar
            .is_none());
    }

    fn tool_request(tool_choice: serde_json::Value) -> ChatCompletionRequest {
        chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "weather in Rome?"}],
            "tools": [weather_tool()],
            "tool_choice": tool_choice,
        }))
    }

    /// `tool_choice: "required"` used to be a 501. It now compiles the
    /// offered tools into a grammar that rides on the params, which is
    /// the only thing every decode path shares.
    #[test]
    fn a_forced_tool_choice_puts_a_grammar_on_the_generation_params() {
        for choice in [
            serde_json::json!("required"),
            serde_json::json!({"type": "function", "function": {"name": "get_weather"}}),
        ] {
            let req = tool_request(choice.clone());
            req.validate_supported_fields()
                .unwrap_or_else(|e| panic!("{choice} is a valid request: {e:?}"));
            let params = req
                .generation_params_for_template(
                    &graded_template(),
                    "Qwen3-8B",
                    crate::sampling_knobs::SamplerModel::absent(),
                )
                .unwrap_or_else(|e| panic!("{choice} compiles: {e:?}"));
            let grammar = params
                .grammar
                .as_ref()
                .unwrap_or_else(|| panic!("{choice} was accepted and then not enforced"));
            assert!(
                grammar.is_awaiting_trigger(),
                "the model must be free to think before it calls"
            );
            assert!(
                !grammar.allows_eog(),
                "{choice} must not be able to end the turn without a call"
            );
            // The bug that has been fixed three times: a constrained
            // request that lets a backend fold lm_head+argmax on device
            // is a constrained request served unconstrained. A LAZY
            // grammar needs the vocabulary from the FIRST token, because
            // its trigger can fire on any of them.
            assert!(
                params.needs_vocab_logits(),
                "{choice} would let a backend return a token id instead of logits"
            );
            assert!(
                !generate::greedy_gpu_fold_allowed(&params),
                "{choice} at temperature 0 must still refuse the greedy GPU fold"
            );
        }
    }

    /// `auto` and `none` force nothing, and must not acquire a grammar.
    #[test]
    fn an_unforced_tool_choice_leaves_the_generation_unconstrained() {
        for choice in [serde_json::json!("auto"), serde_json::json!("none")] {
            let req = tool_request(choice.clone());
            req.validate_supported_fields().expect("still supported");
            let params = match req.generation_params_for_template(
                &graded_template(),
                "Qwen3-8B",
                crate::sampling_knobs::SamplerModel::absent(),
            ) {
                Ok(p) => p,
                Err((status, _)) => panic!("{choice} has no constraint to compile: {status}"),
            };
            assert!(
                params.grammar.is_none(),
                "{choice} does not force a call and must not be constrained"
            );
        }
    }

    /// Every refusal a forced choice can produce names the field, and
    /// none of them is a silent downgrade to `auto`.
    #[test]
    fn a_forced_tool_choice_refuses_rather_than_quietly_not_forcing() {
        // No tools to choose between.
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
        }));
        let (status, _) = req
            .validate_supported_fields()
            .expect_err("nothing to call");
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // A name that is not on offer.
        let req =
            tool_request(serde_json::json!({"type": "function", "function": {"name": "nope"}}));
        let (status, Json(body)) = req.validate_supported_fields().expect_err("no such tool");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "tool_choice");

        // An object that names nothing at all.
        let req = tool_request(serde_json::json!({"type": "function"}));
        let (status, _) = req.validate_supported_fields().expect_err("names nothing");
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // Two constraints on one generation.
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [weather_tool()],
            "tool_choice": "required",
            "grammar": "root ::= \"a\"+",
        }));
        let (status, _) = req
            .validate_supported_fields()
            .expect_err("a grammar and a forced call are two constraints");
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // A checkpoint whose wire format has no grammar is refused by
        // name at params time, when the served model is known. GLM and
        // gemma4 both used to stand here and are forced now;
        // muse_glimmer is the one `tool_grammar::wire::shape` still
        // refuses, and the refusal says which format and why.
        let req = tool_request(serde_json::json!("required"));
        let (status, Json(body)) = match req.generation_params_for_template(
            &graded_template(),
            "muse-glimmer-8b",
            crate::sampling_knobs::SamplerModel::absent(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("a muse_glimmer call's boundary is a channel, not a marker"),
        };
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("muse_glimmer"),
            "{body}"
        );

        // And the format this once refused is served: a served model
        // whose name resolves to gemma4 reaches a grammar rather than a
        // 501. `generation_params_for_template` is the only place a
        // forced choice becomes one, so this is the request-level
        // evidence that the wire work is wired.
        let req = tool_request(serde_json::json!("required"));
        let params = req
            .generation_params_for_template(
                &graded_template(),
                "gemma-4-E2B-it",
                crate::sampling_knobs::SamplerModel::absent(),
            )
            .expect("a gemma4 forced tool_choice is served");
        assert!(
            params.grammar.is_some(),
            "a forced tool_choice must arrive as the generation's grammar"
        );
    }

    /// A grammar that does not parse is refused before any work, and
    /// the refusal names the field and the parser's own diagnostic.
    #[test]
    fn an_unparseable_grammar_on_the_chat_wire_is_a_400() {
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": "root ::= \"a",
        }));
        let (status, Json(body)) = req
            .validate_supported_fields()
            .expect_err("this does not parse");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "grammar");
        assert!(
            req.generation_params(crate::sampling_knobs::SamplerModel::absent())
                .is_err(),
            "and again at params time"
        );
    }

    /// `response_format: json_schema` used to be a 501 naming the
    /// missing converter. It is served now, and the request-level
    /// evidence is that the schema reaches `generation_params` as a
    /// grammar -- there is exactly one place a `response_format` is
    /// decided, so a route that validated it and then forgot to apply
    /// it is the failure this asserts against.
    #[test]
    fn response_format_json_schema_becomes_the_requests_grammar() {
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {"type": "boolean"}},
            },
        }));
        req.validate_supported_fields()
            .expect("a boolean schema converts");
        let params = req
            .generation_params(crate::sampling_knobs::SamplerModel::absent())
            .expect("and compiles");
        let grammar = params.grammar.expect("the schema is the grammar");
        let mut g = (*grammar).clone();
        g.accept_token(0, b"true").expect("a boolean is accepted");
        assert!(g.allows_eog(), "and completes the parse");
        assert!(
            !params.json_object,
            "a schema is not the json_object character-class mask"
        );
    }

    /// A schema the converter will not compile is a 400 naming the
    /// keyword, at both the validation and the params seam -- never a
    /// 500, and never a grammar that is approximately the schema.
    #[test]
    fn an_unconvertible_response_format_schema_is_a_400_naming_the_keyword() {
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {"type": "integer", "minimum": 3}},
            },
        }));
        let (status, Json(body)) = req
            .validate_supported_fields()
            .expect_err("minimum has no grammar in this port");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]["message"]
                .as_str()
                .expect("a message")
                .contains("minimum"),
            "the refusal must name the keyword: {body}"
        );
        assert!(
            req.generation_params(crate::sampling_knobs::SamplerModel::absent())
                .is_err(),
            "and again at params time"
        );
    }

    /// A forced `tool_choice` and a `response_format` schema are two
    /// constraints on one generation. The refusal used to be spelled
    /// against `self.grammar` alone, so the schema spelling walked past
    /// it and `generation_params_for_template` overwrote the schema's
    /// grammar with the tool-call one.
    #[test]
    fn a_forced_tool_choice_and_a_schema_are_two_constraints() {
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
            "tools": [{
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}},
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {"type": "boolean"}},
            },
        }));
        let (status, Json(body)) = req
            .validate_supported_fields()
            .expect_err("two constraints, one generation");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "tool_choice");
    }

    /// A chat client that omits `max_tokens` wants an answer, not
    /// OpenAI's legacy 16-token completion fragment.
    #[test]
    fn an_omitted_output_budget_is_a_whole_answer_not_sixteen_tokens() {
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(req.max_tokens, DEFAULT_CHAT_MAX_TOKENS);
    }

    /// A knob the wire accepts must reach the sampler. Serde declaring
    /// `min_p` is only half of it: the field spent two commits resolved
    /// to a hardcoded `0.0` on both routes, which is exactly the
    /// silently-dropped-parameter bug, just one layer further in.
    #[test]
    fn min_p_reaches_the_sampler_from_the_chat_wire() {
        let asked = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "min_p": 0.07,
        }));
        assert_eq!(
            asked
                .sampling_params(crate::sampling_knobs::SamplerModel::absent())
                .expect("knobs")
                .min_p,
            0.07
        );

        let silent = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(
            silent
                .sampling_params(crate::sampling_knobs::SamplerModel::absent())
                .expect("knobs")
                .min_p,
            0.0,
            "an unset min_p must be off, not llama.cpp's CLI default"
        );
    }

    /// The whole-response cache is keyed on the sampler settings, and a
    /// setting left OUT of that key means two requests differing only in
    /// it share one answer: the second caller silently gets output
    /// computed under the first caller's parameters.
    ///
    /// Every knob the wire accepts is checked, not just the new one --
    /// this is the assertion that would have caught `min_p` being added
    /// to the sampler and forgotten here.
    #[test]
    fn no_sampler_knob_is_missing_from_the_cache_key() {
        let base = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "seed": 1,
        });
        let key_for = |body: serde_json::Value| {
            let req = chat_request(body);
            let params = req
                .generation_params(crate::sampling_knobs::SamplerModel::absent())
                .expect("params");
            req.cache_key("prompt", &params)
        };
        let baseline = key_for(base.clone());
        for (knob, value) in [
            ("temperature", serde_json::json!(0.5)),
            ("top_p", serde_json::json!(0.9)),
            ("min_p", serde_json::json!(0.05)),
            ("top_k", serde_json::json!(40)),
            ("repetition_penalty", serde_json::json!(1.1)),
            ("presence_penalty", serde_json::json!(0.3)),
            ("frequency_penalty", serde_json::json!(0.3)),
            (
                "samplers",
                serde_json::json!(["penalties", "top_p", "top_k", "min_p", "temperature"]),
            ),
        ] {
            let mut body = base.clone();
            body[knob] = value;
            assert_ne!(
                key_for(body),
                baseline,
                "`{knob}` is not in the cache key: two requests differing \
                 only in it would share one cached answer"
            );
        }
    }

    /// The sampler half's twin, for the constraints. Each of these
    /// changes the answer and changes NOTHING about the rendered
    /// prompt, so an omission is invisible until a caller compares two
    /// answers it never sees side by side (#35).
    ///
    /// `grammar` here is the wire field; `response_format:
    /// {"type":"json_schema"}` and a forced `tool_choice` compile to a
    /// grammar through the same `GenerationParams::grammar`, so they are
    /// keyed by the same field being keyed at all.
    #[test]
    fn no_constraint_is_missing_from_the_cache_key() {
        let base = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "pick one"}],
        });
        let key_for = |body: serde_json::Value| {
            let req = chat_request(body);
            let params = req
                .generation_params(crate::sampling_knobs::SamplerModel::absent())
                .expect("params");
            req.cache_key("prompt", &params)
        };
        let baseline = key_for(base.clone());
        for (field, value) in [
            ("grammar", serde_json::json!("root ::= \"yes\" | \"no\"")),
            (
                "response_format",
                serde_json::json!({"type": "json_object"}),
            ),
            (
                "response_format",
                serde_json::json!({"type": "json_schema", "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object", "properties": {"a": {"type": "string"}}}
                }}),
            ),
            ("ignore_eos", serde_json::json!(true)),
            ("stop", serde_json::json!(["\n"])),
            ("max_tokens", serde_json::json!(7)),
        ] {
            let mut body = base.clone();
            body[field] = value.clone();
            assert_ne!(
                key_for(body),
                baseline,
                "`{field}: {value}` is not in the cache key: two requests \
                 differing only in it would share one cached answer"
            );
        }
    }

    /// Serde already tells absent from zero -- an absent field became
    /// the default -- so a 0 here is one the caller wrote, and a
    /// zero-token budget is a request that can never become decodable.
    #[test]
    fn an_explicit_zero_output_budget_is_a_client_error() {
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        }));
        let (status, body) = req.validate_supported_fields().expect_err("rejected");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], serde_json::json!("max_tokens"));
    }

    /// The direction that had no wire path at all before: every request
    /// rendered in thinking mode because only the ON branch existed.
    #[test]
    fn a_request_can_turn_thinking_off() {
        let template = graded_template();
        for body in [
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "none",
            }),
            serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "disabled"},
            }),
        ] {
            let kwargs = chat_request(body).resolve_template_kwargs(&template);
            assert_eq!(kwargs["enable_thinking"], serde_json::json!(false));
            assert_eq!(kwargs["thinking_mode"], serde_json::json!("disabled"));
            // And `none` must not have been rounded onto a real gear on
            // the way: "do not think" is not "think a little".
            assert!(!kwargs.contains_key("reasoning_effort"));
        }
    }

    /// The switch is what the caller reached for last; the gear is what
    /// they would have used had thinking been on.
    #[test]
    fn a_disabled_switch_beats_an_effort_in_the_same_request() {
        let template = graded_template();
        let kwargs = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
            "thinking": {"type": "disabled"},
        }))
        .resolve_template_kwargs(&template);
        assert_eq!(kwargs["enable_thinking"], serde_json::json!(false));
        assert!(!kwargs.contains_key("reasoning_effort"));
    }

    /// Read as "on", a misspelled switch silently serves the opposite
    /// of what was asked for.
    #[test]
    fn an_unrecognized_thinking_switch_is_refused_rather_than_read_as_on() {
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "disable"},
        }));
        let (status, _) = req.validate_supported_fields().expect_err("rejected");
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// A caller who steered the template themselves has said what they
    /// want; merging a protocol default in would let it contradict them.
    #[test]
    fn an_explicit_template_kwarg_stands_the_protocol_knobs_down() {
        let template = graded_template();
        let kwargs = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": true},
        }))
        .resolve_template_kwargs(&template);
        assert_eq!(kwargs["enable_thinking"], serde_json::json!(true));
    }

    /// The acceptance criterion for effort plumbing: an off-vocabulary
    /// value is quantized onto the nearest gear the checkpoint really
    /// grades, and the request renders instead of failing.
    #[test]
    fn an_off_vocabulary_reasoning_effort_is_quantized_rather_than_interpolated() {
        let template = graded_template();
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "minimal",
        }));
        let kwargs = req.resolve_template_kwargs(&template);
        assert_eq!(kwargs["reasoning_effort"], serde_json::json!("low"));
        let prompt = prompt_from_messages(&req.messages, &template, &[], kwargs).expect("renders");
        assert!(prompt.starts_with("E:low|"), "{prompt}");
    }

    /// The other half of the same rule: a value no gear is close enough
    /// to is dropped, so the checkpoint's own default applies rather
    /// than an unknown string reaching the prompt.
    #[test]
    fn an_effort_with_no_near_gear_is_dropped_so_the_template_default_applies() {
        let template = graded_template();
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"reasoning_effort": "none"},
        }));
        let kwargs = req.resolve_template_kwargs(&template);
        assert!(!kwargs.contains_key("reasoning_effort"));
        let prompt = prompt_from_messages(&req.messages, &template, &[], kwargs).expect("renders");
        assert_eq!(prompt, "hi");
    }

    /// `chat_template_kwargs` is the specific spelling and wins over the
    /// top-level one, which is what a caller who wrote both meant.
    #[test]
    fn chat_template_kwargs_wins_over_the_top_level_reasoning_effort() {
        let template = graded_template();
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "high"},
        }));
        assert_eq!(
            req.resolve_template_kwargs(&template)["reasoning_effort"],
            serde_json::json!("high")
        );
    }

    /// Offering tools turns thinking on even when the caller asked for
    /// nothing: some encoders emit well-formed calls only in thinking
    /// mode.
    #[test]
    fn offering_tools_turns_thinking_on_by_itself() {
        let template = graded_template();
        let quiet = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert!(!quiet
            .resolve_template_kwargs(&template)
            .contains_key("enable_thinking"));

        let with_tools = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }));
        let kwargs = with_tools.resolve_template_kwargs(&template);
        assert_eq!(kwargs["enable_thinking"], serde_json::json!(true));
        let prompt =
            prompt_from_messages(&with_tools.messages, &template, &[], kwargs).expect("renders");
        assert!(prompt.starts_with("THINK|"), "{prompt}");
    }

    /// The reason `force_reasoning` could only ever be `false` before:
    /// no template could open a block in the prompt, because no kwargs
    /// reached one. Now that they do, the parser has to start inside it
    /// -- and the evidence is the rendered prompt, not the model name.
    #[test]
    fn a_prompt_that_opens_the_reasoning_block_makes_the_first_token_reasoning() {
        let opener = chat_template::PromptTemplate::from_gguf_metadata(
            Some("{{ messages[0].content }}{% if enable_thinking %}<think>{% endif %}"),
            Some("qwen3"),
            false,
            true,
            None,
            None,
        );
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": true},
        }));
        let kwargs = req.resolve_template_kwargs(&opener);
        let prompt = prompt_from_messages(&req.messages, &opener, &[], kwargs).expect("renders");
        assert!(prompt.ends_with("<think>"), "{prompt}");

        // No opening marker will ever arrive, so unparsed this whole
        // deliberation would have been served as the answer.
        let posture = output::OutputPosture::resolve("Qwen3-8B", &prompt);
        let (message, _) = build_response_message(
            "weighing it up</think>Paris.".to_string(),
            &[],
            posture,
            "stop",
        );
        assert_eq!(message.reasoning_content.as_deref(), Some("weighing it up"));
        assert_eq!(message.content.as_deref(), Some("Paris."));

        // Same text, a prompt that did not open the block: the model
        // wrote a stray closer and it stays content.
        let closed = output::OutputPosture::resolve("Qwen3-8B", "<|im_start|>assistant\n");
        let (message, _) = build_response_message(
            "weighing it up</think>Paris.".to_string(),
            &[],
            closed,
            "stop",
        );
        assert_eq!(message.reasoning_content, None);
    }

    #[test]
    fn stop_param_accepts_both_single_string_and_array() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "END",
        }))
        .unwrap();
        assert_eq!(req.stop_sequences(), vec!["END".to_string()]);

        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["A", "B"],
        }))
        .unwrap();
        assert_eq!(req.stop_sequences(), vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn run_generation_rejects_out_of_vocab_tokens_instead_of_panicking() {
        let model = test_model();
        let result = run_generation(
            &model,
            "hello",
            &greedy_params(4),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(matches!(
            result,
            Err(generate::DecodeError::TokenOutOfVocab { .. })
        ));
    }

    /// A pool that *could* serve this request but is momentarily fully
    /// held is the server being behind: 503, and retrying is honest
    /// advice because the blocks really do come back.
    #[test]
    fn run_generation_honors_an_exhausted_kv_pool_and_maps_it_to_a_503() {
        let model = test_model(); // 2 layers -> 2 blocks
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(frink_core::cache::KvBlockPool::new(64, 2)));

        let holder_pool = Arc::clone(&pool);
        let holder = std::thread::spawn(move || {
            let mut held = frink_core::cache::KvCache::with_pool(1, 1, holder_pool, 0).unwrap();
            held.push(&[0.0], &[0.0]).unwrap(); // crosses into the second block
            std::thread::sleep(Duration::from_millis(200));
            drop(held);
        });
        std::thread::sleep(Duration::from_millis(15));

        let config = generate::KvPoolConfig {
            pool,
            queue_wait: Duration::ZERO,
        };
        let result = run_generation(
            &model,
            &prompt,
            &greedy_params(4),
            Some(&config),
            None,
            None,
            None,
            None,
            None,
        );
        assert!(matches!(
            result,
            Err(generate::DecodeError::KvPoolExhausted)
        ));

        let (status, _body) = decode_error_response(result.unwrap_err());
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        holder.join().unwrap();
    }

    /// The same endpoint, the same pool size, a request too big for the
    /// *whole* pool: a 400 rather than a 503, because an idle server
    /// refuses it identically and `Retry-After` would be a promise
    /// nothing can keep.
    ///
    /// Confirmed to FAIL when `generate`'s `pool_immovable_refusal`
    /// check is removed: the status comes back 503.
    #[test]
    fn a_request_too_big_for_the_whole_pool_is_a_400_not_a_retryable_503() {
        let model = test_model(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        // One block, two layers: no schedule ever serves this.
        let pool = Arc::new(Mutex::new(frink_core::cache::KvBlockPool::new(64, 1)));
        let config = generate::KvPoolConfig {
            pool,
            queue_wait: Duration::ZERO,
        };

        let result = run_generation(
            &model,
            &prompt,
            &greedy_params(4),
            Some(&config),
            None,
            None,
            None,
            None,
            None,
        );
        let err = result.expect_err("one block cannot hold two layers' caches");
        assert!(
            matches!(
                &err,
                generate::DecodeError::KvBudgetExceeded { binding, .. }
                    if *binding == frink_models::Ceiling::DeviceMemory.code()
            ),
            "expected an immovable device-memory refusal, got {err:?}"
        );
        let (status, _body) = decode_error_response(err);
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// A full admission queue is the server being behind, not the
    /// client being wrong: 503, with the wait hint in the body (and the
    /// `Retry-After` header stamped by `limits::retry_after`) and the
    /// depth and cap named so an operator can tell a retry storm from a
    /// single oversized request.
    #[test]
    fn decode_error_response_maps_a_full_queue_to_a_retryable_503() {
        let (status, Json(body)) = decode_error_response(generate::DecodeError::QueueFull {
            queued: 512,
            cap: 512,
        });
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["retry_after_seconds"], 1);
        let message = body["error"]["message"].as_str().expect("message");
        assert!(message.contains("512"), "{message}");
    }

    #[test]
    fn decode_error_response_omits_a_retry_hint_for_an_unretryable_error() {
        let (_status, Json(body)) = decode_error_response(generate::DecodeError::TokenOutOfVocab {
            token: 99,
            vocab_size: 32,
        });
        assert!(
            body["error"]["retry_after_seconds"].is_null(),
            "retrying a prompt this model cannot tokenize never helps"
        );
    }

    #[test]
    fn decode_error_response_maps_token_out_of_vocab_to_bad_request() {
        let (status, _body) = decode_error_response(generate::DecodeError::TokenOutOfVocab {
            token: 99,
            vocab_size: 32,
        });
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn run_generation_succeeds_and_releases_blocks_when_the_pool_has_room() {
        let model = test_model(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(frink_core::cache::KvBlockPool::new(64, 2)));
        let config = generate::KvPoolConfig {
            pool: pool.clone(),
            queue_wait: Duration::ZERO,
        };

        let (_, finish, _usage) = run_generation(
            &model,
            &prompt,
            &greedy_params(4),
            Some(&config),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            2,
            "a completed request must return its blocks to the pool"
        );
    }

    /// The core concurrency claim: two requests using the *same* `Arc<Model>`
    /// must be able to run their (independent, per-call) KV caches
    /// concurrently without interfering with each other or needing any
    /// shared lock around the model itself.
    #[tokio::test]
    async fn concurrent_requests_against_the_same_model_do_not_interfere() {
        let model = Arc::new(test_model());
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let model = Arc::clone(&model);
            let prompt = prompt.clone();
            handles.push(tokio::task::spawn_blocking(move || {
                run_generation(
                    &model,
                    &prompt,
                    &greedy_params(6),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
            }));
        }

        let mut results = Vec::new();
        for h in handles {
            results.push(h.await.unwrap());
        }
        // Same prompt, same seed, same (greedy) sampling, same
        // immutable model -> every concurrent run must produce
        // identical output, proving no request's KV cache leaked into
        // another's.
        for r in &results[1..] {
            assert_eq!(r.0, results[0].0, "decoded chunks must match");
            assert_eq!(r.1, results[0].1, "finish reason must match");
            assert_eq!(
                r.2.prompt_tokens, results[0].2.prompt_tokens,
                "prompt token count must match"
            );
            assert_eq!(
                r.2.completion_tokens, results[0].2.completion_tokens,
                "completion token count must match"
            );
        }
    }

    /// A real, minimal safetensors shard: JSON header (name -> real
    /// dtype/shape/`data_offsets`) followed by the concatenated raw
    /// F32 bytes -- exactly the format `ShardedSafetensors::open_index`
    /// parses, hand-built here rather than depending on
    /// `frink-models::kimi_loader`'s own private test helpers (not
    /// visible across the crate boundary).
    fn write_safetensors_shard(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header_entries = Vec::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            let end = data.len();
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header_entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_str}],\"data_offsets\":[{start},{end}]}}"
            ));
        }
        let header = format!("{{{}}}", header_entries.join(","));
        let header_bytes = header.as_bytes();
        let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(&data);
        out
    }

    /// Builds a small but completely real Kimi K3 checkpoint directory
    /// on disk (real `model.safetensors.index.json` + shard bytes +
    /// `tiktoken.model`, the exact file layout `frink-cli`'s
    /// `run-kimi` command expects) and loads it through
    /// `model::load_kimi_checkpoint_with_config` (the same real loading
    /// logic `model::load()` uses for `FRINK_MODEL_PATH` pointing at a
    /// directory, parametrized here only so the checkpoint can be small
    /// -- see that function's doc comment). Shared by every test that
    /// needs a real, loaded `KimiLoaded` rather than duplicating this
    /// setup per test.
    fn build_synthetic_kimi_loaded() -> model::KimiLoaded {
        use frink_models::config::{AttentionKind, KdaConfig, KimiHybridAttention, MlaConfig};
        use frink_models::kimi_loader::KimiRealHparams;
        use frink_moe::{GatingFunction, MoeLayerConfig};

        let hidden_dim = 8;
        let kda_num_heads = 2;
        let kda_head_dim = 3;
        let kda_proj = kda_num_heads * kda_head_dim;
        let conv_kernel = 4;
        let dense_intermediate = 5;
        // One token per byte value -- enough to round-trip a simple
        // ASCII prompt through the real tiktoken-format vocab below,
        // matching `kimi_generate`'s own test convention.
        let vocab_size = 256;
        let mla_num_heads = 1;
        let mla_q_lora_rank = 2;
        let mla_kv_lora_rank = 2;
        let mla_qk_nope_head_dim = 2;
        let mla_qk_rope_head_dim = 2;
        let mla_v_head_dim = 2;

        let model_cfg = frink_models::ModelConfig {
            rope_layers: frink_models::rope_layers::RopeLayers::All,
            layer_shapes: frink_models::layer_shapes::LayerShapes::Uniform,
            name: "synthetic-kimi-server-test",
            n_layers: 1,
            n_mtp_blocks: 0,
            hidden_dim,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
            v_head_dim: None,
            vocab_size,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            post_norm_eps: 1e-5,
            sliding_window: None,
            moe: MoeLayerConfig {
                expert_weights_scale: 1.0,
                routed_weight_before_ffn: false,
                n_experts: 1,
                n_experts_active: 1,
                n_shared_experts: 0,
                hidden_dim,
                expert_ffn_dim: 4,
                gating: GatingFunction::Sigmoid,
                norm_topk_prob: true,
                expert_group_count: None,
                expert_group_used_count: None,
            },
            // Layer 0 is the sole dense leading layer, using KDA
            // attention (real Kimi K3's own layer-0 shape) -- the
            // 1-indexed `kda_layers`/`full_attn_layers` convention is
            // `ModelConfig::layer_attention_kind`'s, not this test's.
            n_dense_leading_layers: 1,
            moe_interleave_step: None,
            norm_function: frink_models::norm::NormFunction::Rms,
            attention: AttentionKind::KimiHybrid(KimiHybridAttention {
                kda_layers: vec![1],
                full_attn_layers: vec![],
                mla: MlaConfig {
                    num_heads: mla_num_heads,
                    q_lora_rank: mla_q_lora_rank,
                    kv_lora_rank: mla_kv_lora_rank,
                    qk_nope_head_dim: mla_qk_nope_head_dim,
                    qk_rope_head_dim: mla_qk_rope_head_dim,
                    v_head_dim: mla_v_head_dim,
                    use_output_gate: true,
                    rope: None,
                },
                kda: KdaConfig {
                    num_heads: kda_num_heads,
                    head_dim: kda_head_dim,
                    short_conv_kernel_size: conv_kernel,
                    gate_lower_bound: -5.0,
                    use_full_rank_gate: true,
                },
            }),
            rope_freqs: None,
            rope_attn_factor: 1.0,
            rope_dim: None,
            rope_dim_swa: None,
            rope_freqs_long: None,
            rope_freqs_short: None,
            rope_orig_ctx: None,
            rope_layout: frink_models::config::RopeLayout::Neox,
            qk_norm_style: frink_models::capability::QkNormStyle::WholeVector,
            swa_layers: frink_models::swa_layers::SwaLayers::All,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            embedding_scale: None,
            residual_scale: None,
            normed_residual_scale: None,
            clamp_kqv: None,
            attn_temperature: None,
            router_input: frink_models::router_input::RouterInput::NormedFfnInput,
            block_sub_norms: false,
            parallel_residual: false,
            learned_positions: false,
            attn_value_scale: None,
            alibi_max_bias: None,
            layer_loops: None,
            skip_stream: false,
            parallel_ssm: false,
            swa_chunked: false,
            weightless_qk_norm: false,
            logit_multiplier: None,
            attention_scale: None,
            rope_theta_swa: None,
            ffn_activation: frink_models::config::FfnActivation::Swiglu,
            best_effort_fields: &["synthetic test config, not a real preset"],
        };
        let hp = KimiRealHparams {
            hidden_dim,
            kda_num_heads,
            kda_head_dim,
            mla_num_heads,
            mla_q_lora_rank,
            mla_kv_lora_rank,
            mla_qk_nope_head_dim,
            mla_qk_rope_head_dim,
            mla_v_head_dim,
            dense_intermediate_dim: dense_intermediate,
            moe_hidden_dim: hidden_dim,
            moe_intermediate_dim: 4,
            n_experts: 1,
            num_shared_experts: 0,
        };

        // Every real tensor name `kimi_loader::load_kimi_layer` (dense
        // FFN + KDA attention + block residual) and
        // `load_kimi_checkpoint` (top-level) actually read.
        let prefix = "language_model.model.layers.0";
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let push = |tensors: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
                    name: String,
                    shape: Vec<usize>,
                    n: usize| {
            tensors.push((name, shape, vec![0.01f32; n]));
        };
        push(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attention_res_norm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attention_res_proj.weight"),
            vec![1, hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp_res_norm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp_res_proj.weight"),
            vec![1, hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.q_conv1d.weight"),
            vec![kda_proj, 1, conv_kernel],
            kda_proj * conv_kernel,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.k_conv1d.weight"),
            vec![kda_proj, 1, conv_kernel],
            kda_proj * conv_kernel,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.v_conv1d.weight"),
            vec![kda_proj, 1, conv_kernel],
            kda_proj * conv_kernel,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.A_log"),
            vec![kda_num_heads],
            kda_num_heads,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.f_a_proj.weight"),
            vec![kda_head_dim, hidden_dim],
            kda_head_dim * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.f_b_proj.weight"),
            vec![kda_proj, kda_head_dim],
            kda_proj * kda_head_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.dt_bias"),
            vec![kda_proj],
            kda_proj,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.b_proj.weight"),
            vec![kda_num_heads, hidden_dim],
            kda_num_heads * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.g_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.o_norm.weight"),
            vec![kda_head_dim],
            kda_head_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![hidden_dim, kda_proj],
            hidden_dim * kda_proj,
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![dense_intermediate, hidden_dim],
            dense_intermediate * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.up_proj.weight"),
            vec![dense_intermediate, hidden_dim],
            dense_intermediate * hidden_dim,
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.down_proj.weight"),
            vec![hidden_dim, dense_intermediate],
            hidden_dim * dense_intermediate,
        );
        push(
            &mut tensors,
            "language_model.model.embed_tokens.weight".to_string(),
            vec![vocab_size, hidden_dim],
            vocab_size * hidden_dim,
        );
        push(
            &mut tensors,
            "language_model.lm_head.weight".to_string(),
            vec![vocab_size, hidden_dim],
            vocab_size * hidden_dim,
        );
        push(
            &mut tensors,
            "language_model.model.norm.weight".to_string(),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            "language_model.model.output_attn_res_norm.weight".to_string(),
            vec![hidden_dim],
            hidden_dim,
        );
        push(
            &mut tensors,
            "language_model.model.output_attn_res_proj.weight".to_string(),
            vec![1, hidden_dim],
            hidden_dim,
        );

        // Unique per CALL, not per (pid, vocab_size). Both callers of
        // this helper use the same `vocab_size`, so keying on it gave
        // the two tests one directory -- and `fs::write` opens with
        // `O_TRUNC`, so one test rewriting the shard truncated it to
        // zero while the other's `frink-safetensors` MMAP of that
        // exact file was live. Touching a mapping past the end of its
        // file is SIGBUS, which kills the whole test binary rather than
        // failing one test, and only when the two happen to overlap --
        // so it showed up as an occasional unexplained CI crash.
        //
        // A counter and not a thread id: the harness reuses threads
        // across tests, so two sequential tests can share one.
        static FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "frink_server_kimi_e2e_test_{}_{}",
            std::process::id(),
            FIXTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let shard_bytes = write_safetensors_shard(&tensors);
        std::fs::write(dir.join("shard0.safetensors"), &shard_bytes).unwrap();
        let map_entries: Vec<String> = tensors
            .iter()
            .map(|(name, ..)| format!("\"{name}\":\"shard0.safetensors\""))
            .collect();
        let index = format!("{{\"weight_map\":{{{}}}}}", map_entries.join(","));
        std::fs::write(dir.join("model.safetensors.index.json"), &index).unwrap();

        // A real tiktoken-format vocab file: one base64-encoded byte
        // plus its rank per line -- enough to round-trip an ASCII
        // prompt without needing the real 163584-entry Kimi K3 vocab.
        use base64::Engine;
        let vocab_lines: Vec<String> = (0..vocab_size as u32)
            .map(|b| {
                let b64 = base64::engine::general_purpose::STANDARD.encode([b as u8]);
                format!("{b64} {b}")
            })
            .collect();
        std::fs::write(dir.join("tiktoken.model"), vocab_lines.join("\n")).unwrap();

        let loaded = model::load_kimi_checkpoint_with_config(dir.to_str().unwrap(), model_cfg, hp)
            .expect("must load the synthetic Kimi checkpoint end to end");
        std::fs::remove_dir_all(&dir).ok();
        loaded
    }

    /// The real end-to-end proof for Kimi-through-the-server: a real
    /// synthetic Kimi K3 checkpoint served through the exact same
    /// `run_generation` entry point the HTTP handlers call for the
    /// GGUF path. Proves the whole new plumbing end to end: directory-
    /// shaped checkpoint loading, `KimiEngine`/`KimiTokenizer` wired
    /// through the `Model` enum, and `generate::generate_engine`
    /// producing real, bounded generated text.
    #[test]
    fn kimi_model_serves_real_text_end_to_end_via_run_generation() {
        let loaded = build_synthetic_kimi_loaded();
        let state = build_app_state(
            StartupModels {
                loaded: model::LoadedModel::Kimi(loaded),
                embedding: None,
            },
            None,
            None,
            None,
            false,
            None,
            Arc::new(health::Detection::ready(health::probe_backends())),
        );
        let active = state.active().expect("a freshly built state has a model");
        assert_eq!(active.tokenizer_kind(), "kimi-tiktoken-bpe");
        assert!(!active.is_synthetic());

        let (_chunks, finish, _usage) = run_generation(
            active.generative().unwrap(),
            "hi",
            &greedy_params(5),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("a real Kimi checkpoint must generate without error");
        assert!(matches!(finish, FinishReason::Length | FinishReason::Stop));
    }

    /// The THIRD decode path: `generate_engine`, which serves every
    /// model that is not a `Decoder`.
    ///
    /// This is where a constraint gets dropped without anyone noticing.
    /// JSON mode was honoured on the `Decoder` path and silently not on
    /// this one, because this path had no tokenizer to hand the mask.
    /// A grammar must reach it too, and this checkpoint's vocabulary is
    /// one token per byte value, so `root ::= "a"+` has exactly one
    /// legal token (97) and the answer is decidable: all `a`, however
    /// the random weights would otherwise have decoded.
    ///
    /// The unconstrained run beside it is the vacuity check.
    #[test]
    fn a_grammar_constrains_the_engine_decode_path() {
        let loaded = build_synthetic_kimi_loaded();
        let state = build_app_state(
            StartupModels {
                loaded: model::LoadedModel::Kimi(loaded),
                embedding: None,
            },
            None,
            None,
            None,
            false,
            None,
            Arc::new(health::Detection::ready(health::probe_backends())),
        );
        let active = state.active().expect("a freshly built state has a model");

        let run = |grammar: Option<&str>| {
            let mut params = greedy_params(6);
            params.grammar = grammar.map(|src| {
                Arc::new(
                    frink_models::grammar::Grammar::from_str_with_root(src, "root")
                        .expect("test grammar parses"),
                )
            });
            run_generation(
                active.generative().unwrap(),
                "hi",
                &params,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        };

        let (chunks, _, _) = run(None).expect("the unconstrained run must serve");
        let unconstrained = chunks.concat();
        assert!(
            unconstrained.chars().any(|c| c != 'a'),
            "the unconstrained run produced only `a` ({unconstrained:?}), so the \
             constrained run below would prove nothing"
        );

        let (chunks, finish, _) =
            run(Some(r#"root ::= "a"+"#)).expect("a grammar this vocabulary can spell must serve");
        let constrained = chunks.concat();
        assert!(
            !constrained.is_empty() && constrained.chars().all(|c| c == 'a'),
            "the engine decode path served text its grammar forbids ({constrained:?}): \
             the constraint was dropped between `generate_engine` and the sampler"
        );
        assert!(matches!(finish, FinishReason::Length | FinishReason::Stop));
    }

    /// Explicit proof of the "gate, don't paper over" design decision
    /// (see `frink_models::engine`'s module docs): even when an operator configures
    /// a KV block pool and/or prefix cache, a Kimi request must never
    /// consult either -- `generate_engine`'s signature has no
    /// parameter for them at all, so this isn't just an unexercised
    /// code path, it's structurally impossible for a Kimi request to
    /// touch them. Confirmed here by observing both are completely
    /// untouched (pool blocks unchanged, cache stats unchanged) after a
    /// real Kimi generation runs alongside both.
    #[test]
    fn kv_pool_and_prefix_cache_are_never_consulted_for_a_kimi_model() {
        let loaded = build_synthetic_kimi_loaded();
        let state = build_app_state(
            StartupModels {
                loaded: model::LoadedModel::Kimi(loaded),
                embedding: None,
            },
            None,
            None,
            None,
            false,
            None,
            Arc::new(health::Detection::ready(health::probe_backends())),
        );

        let pool = Arc::new(Mutex::new(frink_core::cache::KvBlockPool::new(64, 4)));
        let kv_pool_config = generate::KvPoolConfig {
            pool: pool.clone(),
            queue_wait: Duration::ZERO,
        };
        let pc = Mutex::new(PrefixCache::new(4));

        run_generation(
            state
                .active()
                .expect("a freshly built state has a model")
                .generative()
                .unwrap(),
            "hi",
            &greedy_params(5),
            Some(&kv_pool_config),
            None,
            Some(&pc),
            None,
            None,
            None,
        )
        .expect("a real Kimi checkpoint must generate without error");

        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            4,
            "the KV pool must be completely untouched by a Kimi request"
        );
        let stats = pc.lock().unwrap().stats();
        assert_eq!(
            stats.hits + stats.misses,
            0,
            "the prefix cache must never be consulted for a Kimi request"
        );
    }
}
