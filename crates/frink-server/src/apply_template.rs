//! llama-server's `POST /apply-template`: the prompt a chat request
//! renders to, with nothing generated (`tools/server/server-context.cpp`,
//! `post_apply_template`, which answers `{"prompt": ...}`).
//!
//! The body is a chat completions request, and the prompt is the one
//! `/v1/chat/completions` would decode from: the same history, the same
//! kwargs, the same template and tool rendering, through the same
//! `render_prompt`. A tool that shows a user the prompt shows the one
//! that would be served.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::{ApiError, AppState, ChatCompletionRequest};

pub(crate) async fn apply_template(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    req.validate_supported_fields()?;
    let active = state.require_active()?;
    let history = crate::resolve_history(&state, &req);
    let template = active.generative()?.chat_template();
    let kwargs = req.resolve_template_kwargs(&template);
    let prompt = req.render_prompt(&history, &template, &req.tools, kwargs, active.name())?;
    Ok(Json(serde_json::json!({ "prompt": prompt })))
}
