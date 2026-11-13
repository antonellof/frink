//! The wire contract between `frink-server` and everything that talks
//! to it: the web UI served at `/`, the desktop shell that spawns the
//! server as a child process, `frink chat`, and any third-party client.
//!
//! Why a crate instead of literals at both ends: the UI is deliberately
//! "just another API client" (see `bc5074aa:docs/plans/on-hold/frink-ui.md`) -- it calls
//! the same public endpoints an IDE would, so the public contract cannot
//! rot without the UI breaking first. That only holds if there is
//! exactly one definition of each path and each payload shape. A
//! hand-copied `"/v1/chat/completions"` in a frontend is a contract that
//! drifts silently; a `pub const` that both sides import is one that
//! cannot.
//!
//! Scope rule: this crate owns *frink-specific* additions and control
//! surfaces (health/capabilities, the process-ready handshake, usage
//! timings, task progress). The OpenAI-compatible request/response
//! bodies stay in `frink-server` where they are validated -- mirroring
//! someone else's schema here would create a second place for it to be
//! wrong.
//!
//! Deliberately dependency-light (serde only): a desktop shell, a CLI
//! and a WASM frontend may all link it.

pub mod admin;
pub mod cancel;
pub mod health;
pub mod lifecycle;
pub mod lora;
pub mod progress;
pub mod request_id;
pub mod routes;
pub mod usage;

pub use admin::{
    CancelResponse, DownloadRequest, LoadModelRequest, ModelEntry, ModelState, ModelsResponse,
    ProgressState, RecentRequest, StatsResponse, TaskAccepted, TaskKind, TaskProgress, TaskStatus,
    TaskView, TasksResponse, UnloadResponse,
};
pub use cancel::{CancelGenerationRequest, CancelGenerationResponse};
pub use health::{Capability, HealthResponse, HealthState};
pub use lifecycle::{ServerReady, READY_EVENT};
pub use lora::{LoraAdapterInfo, LoraApplyResponse, LoraScaleRequest};
pub use progress::{RateEstimator, RateReport};
pub use request_id::next_request_id;
pub use usage::{CompletionTokensDetails, Usage};
