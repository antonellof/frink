//! The OpenAI **Responses** API (`POST /v1/responses`) -- the surface
//! `codex` speaks.
//!
//! Ported from FreeToken's `server/responses_api.py` (Apache-2.0; see
//! `docs/THIRD_PARTY_NOTICES.md`), and only its *simple* rail: the
//! gpt-oss Harmony rail is not ported, here or there.
//!
//! # This is a shaping, not a second engine
//!
//! Everything below reshapes a Responses request into the exact
//! [`crate::ChatCompletionRequest`] the chat path already renders and
//! decodes from, and reshapes what came back into Responses' typed
//! items and stream events. It deliberately does **not** grow its own
//! prompt renderer, its own effort sanitizer or its own stop set: those
//! live on that struct
//! ([`crate::ChatCompletionRequest::resolve_template_kwargs`],
//! [`crate::ChatCompletionRequest::generation_params_for_template`]),
//! and a second copy of them is a second set of answers that can drift
//! from the first. The failure that prevents is a checkpoint that
//! thinks through `/v1/chat/completions` and does not through
//! `/v1/responses`, on the same request, for no reason a user can see.
//!
//! # Scope
//!
//! A **stateless** subset, exactly as FreeToken ships it: `store`,
//! `previous_response_id` and `background` are accepted-and-ignored or
//! refused, and `GET /v1/responses/{id}` / `.../cancel` are 404 stubs.
//! That is enough to drive codex, which resends the whole conversation
//! as `input` when it is not storing.
//!
//! # The five rules that are load-bearing
//!
//! Each of these has a test below that fails if the rule is done the
//! obvious way instead.
//!
//! 1. **One leading system message.** The `developer` role folds to
//!    `system`, and the top-level `instructions` plus every
//!    system/developer input item merge into a single leading system
//!    turn. codex sends both at once; a strict template (Qwen3.5)
//!    answers a second system message with "System message must be at
//!    the beginning" and the request fails outright.
//! 2. **One assistant turn per assistant run.** A replayed
//!    `reasoning` + `message` + `function_call` run is *one* turn
//!    ([`merge_assistant_run`]). One message per input item shows the
//!    model phantom extra turns it never took.
//! 3. **Responses' own output budget.** `max_output_tokens` defaults to
//!    [`DEFAULT_MAX_OUTPUT_TOKENS`], named here rather than inherited
//!    from another surface's serde default. The floor FreeToken's
//!    comment warns about is still in this workspace -- 16 tokens, on
//!    `/v1/completions` (`openai_extra::default_max_tokens`) -- and a
//!    Responses turn served under it stops mid-sentence.
//! 4. **Truncation is `incomplete`, on the final item only.** A
//!    `length` finish emits `response.incomplete` with
//!    `incomplete_details.reason = "max_output_tokens"`, and marks only
//!    the *last* message item `incomplete`; items closed earlier in the
//!    stream completed normally and saying otherwise tells codex it
//!    holds a truncated tool result.
//! 5. **Keepalives are data frames.** After
//!    [`KEEPALIVE_INTERVAL`] of silence the stream emits a
//!    protocol-native `response.in_progress` -- never an SSE comment.
//!    codex's 300 s stream-idle timeout only resets on a *data* frame,
//!    so comment keepalives (what `Sse::keep_alive` sends, and what the
//!    chat path uses) leave it reconnecting in the middle of an answer.
//!    The interval also covers the silence *before* the first event,
//!    which is where a long prefill lives.
//!
//! # Errors
//!
//! A generation that fails mid-stream ends in `response.failed`
//! carrying `error.code`. codex reads that code to tell a blown context
//! window (`context_length_exceeded`, from
//! [`frink_models::Ceiling::code`] by way of
//! [`crate::generate::DecodeError`]) from anything else; a stream that
//! simply stops instead leaves it waiting out its idle timeout and then
//! reporting a network fault for a request the server rejected on
//! purpose.

// **Delete this the moment the routes below are registered.** Nothing
// in this module has a caller until `lib.rs`'s router mounts
// [`responses`], [`responses_get`] and [`responses_cancel`] -- which
// this change deliberately does not do -- so every item here reads as
// dead code to the compiler and `-D warnings` fails on the lot of it.
// Everything else in the module is reachable from those three, so this
// line stops being true the moment they are wired up.
#![allow(dead_code)]

use crate::stream_events::{map_tool_event, with_keepalive};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::generate::{DecodeError, Usage};
use crate::output::OutputPosture;
use crate::{
    attribution, output, run_generation_emit, sse, stats, ApiError, AppState,
    ChatCompletionRequest, ChatMessage, MessageContent, ToolCallFunctionIn, ToolCallIn, ToolChoice,
    ToolDef, ToolFunctionDef,
};

/// The path this module wants to be mounted at. Kept here only until it
/// can be `frink_api::routes::V1_RESPONSES`; the stats ring keys on it
/// like every other route.
pub(crate) const ROUTE: &str = "/v1/responses";

/// The Responses surface's own default output budget -- the value
/// FreeToken ships under this name, stated here rather than borrowed.
///
/// Borrowing is the failure it exists to prevent: the legacy 16-token
/// floor still lives in this workspace on `/v1/completions`
/// (`openai_extra::default_max_tokens`), where a caller completing a
/// fragment wants a fragment back, and a codex turn served under it
/// stops mid-sentence with `finish_reason: length` and no way for the
/// client to tell that from a model that had nothing more to say. That
/// the chat surface currently happens to default to the same number
/// this does is a coincidence, not a shared decision.
pub(crate) const DEFAULT_MAX_OUTPUT_TOKENS: usize = 32_768;

/// Event silence after which the stream emits a data-bearing
/// `response.in_progress`.
///
/// codex's stream-idle timeout (300 s by default) is reset by *data*,
/// not by SSE comments, so this is a real protocol frame and this
/// module never installs `Sse::keep_alive`.
pub(crate) const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);

// ---------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A `resp_` / `rs_` / `msg_` / `fc_` / `call_` id.
///
/// Same construction as [`frink_api::next_request_id`] -- a wall-clock
/// stamp plus a process-monotonic counter -- rather than a UUID: these
/// only have to be unique, and the workspace carries no RNG dependency
/// for the server to reach for.
fn new_id(prefix: &str) -> String {
    let stamp = unix_nanos();
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}{:012x}{:06x}", stamp & 0xffff_ffff_ffff, n)
}

fn unix_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------

/// `input` is either one bare user string or a list of typed items.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum ResponsesInput {
    Text(String),
    Items(Vec<Value>),
}

/// The subset of the Responses request this server honours. Unknown
/// fields deserialize away silently, which is the `extra="allow"` the
/// Python model declares.
///
/// Input *items* stay as raw [`Value`]s on purpose: there are six item
/// shapes, every one of them carries fields this server ignores, and a
/// typed enum would reject a future codex build over a field nobody
/// reads. [`convert_input_item`] is the only thing that looks inside.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResponsesRequest {
    #[serde(default)]
    model: String,
    input: ResponsesInput,
    #[serde(default)]
    instructions: Option<String>,
    /// Signed on purpose: `-1` must reach the "must be a positive
    /// integer" refusal below rather than die in serde as a type error,
    /// which would answer a different question than the one asked.
    #[serde(default)]
    max_output_tokens: Option<i64>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    /// `{"effort": "..."}`. Folded onto the chat request's
    /// `reasoning_effort`, so it goes through the same
    /// quantize-onto-what-this-checkpoint-grades path as every other
    /// surface.
    #[serde(default)]
    reasoning: Option<Value>,
    /// llama.cpp's `reasoning_budget_tokens` (alias
    /// `thinking_budget_tokens`), at the top level of the body where
    /// llama.cpp's Responses lowering leaves every field it does not
    /// rewrite (`server-chat.cpp:15`). Same type, same range, same
    /// sampler as on `/v1/chat/completions`.
    #[serde(default, alias = "thinking_budget_tokens")]
    reasoning_budget_tokens: Option<crate::reasoning_budget::BudgetTokens>,
    #[serde(default)]
    chat_template_kwargs: Option<Map<String, Value>>,
    /// Stateful features. Accepted so a client that always sends them
    /// still works; `background` and `previous_response_id` are refused
    /// (see [`responses`]) because honouring them is not possible here
    /// and pretending to would lose the client's turn.
    #[serde(default)]
    background: bool,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    store: bool,
    #[serde(default)]
    #[allow(dead_code)]
    metadata: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    parallel_tool_calls: Option<bool>,
}

// ---------------------------------------------------------------------
// Request conversion: Responses -> ChatCompletionRequest
// ---------------------------------------------------------------------

/// One converted call, before it becomes a [`ToolCallIn`] (which is
/// deserialize-only and has no `PartialEq`, so the merge rule could not
/// be asserted on it).
#[derive(Debug, Clone, PartialEq, Eq)]
struct ConvertedCall {
    id: String,
    name: String,
    arguments: String,
}

/// One conversation turn under construction.
///
/// `reasoning` exists here and nowhere downstream: it is what decides
/// whether an item *opens* a turn ([`merge_assistant_run`]), and it is
/// dropped when the turn is lowered to a [`ChatMessage`] -- see
/// [`Turn::lower`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Turn {
    role: String,
    content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Vec<ConvertedCall>,
    tool_call_id: Option<String>,
}

impl Turn {
    fn message(role: &str, content: String) -> Self {
        Turn {
            role: role.to_string(),
            content: Some(content),
            ..Turn::default()
        }
    }

    /// Python's truthiness on `m["content"]`: an empty string is not
    /// content, and merging on it would silently blank a turn that
    /// already had text.
    fn has_text(&self) -> bool {
        self.content.as_ref().is_some_and(|c| !c.is_empty())
    }

    /// The turn as the renderer wants it, or `None` for a turn that
    /// held nothing but replayed reasoning.
    ///
    /// A replayed reasoning item becomes
    /// [`ChatMessage::reasoning_content`], never `content`: putting a
    /// past chain of thought into the visible answer would show the
    /// model its own deliberation as something it *said*, which is what
    /// the family's thinking markers exist to prevent. A template that
    /// does not know about reasoning simply never reads the key.
    ///
    /// A turn that held NOTHING but reasoning still lowers to `None`.
    /// It has no content and no calls, so there is no message for the
    /// reasoning to hang off; what is kept is the half that shapes the
    /// conversation, since a reasoning item still opens its assistant
    /// turn and the following message merges into it.
    fn lower(self) -> Option<ChatMessage> {
        if !self.has_text() && self.tool_calls.is_empty() && self.reasoning.is_some() {
            return None;
        }
        let tool_calls: Vec<ToolCallIn> = self
            .tool_calls
            .into_iter()
            .map(|call| ToolCallIn {
                id: call.id,
                kind: "function".to_string(),
                function: ToolCallFunctionIn {
                    name: call.name,
                    arguments: call.arguments,
                },
            })
            .collect();
        Some(ChatMessage {
            role: self.role,
            // `None`, not `Some("")`, for a turn that only called tools:
            // that is the OpenAI convention `ChatMessage::content`
            // documents, and the one the templates are written against.
            content: match self.content {
                Some(text) if !text.is_empty() || tool_calls.is_empty() => {
                    Some(MessageContent::Text(text))
                }
                _ => None,
            },
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            tool_call_id: self.tool_call_id,
            reasoning_content: self.reasoning.filter(|r| !r.is_empty()),
        })
    }
}

/// One Responses input item as zero or more turns.
///
/// The `developer` role folds to `system` here: codex sends its
/// permissions block as `developer`, and a chat template knows only
/// system / user / assistant / tool. Rule 1 then merges it with
/// `instructions` (see [`to_chat_request`]).
fn convert_input_item(item: &Value) -> Vec<Turn> {
    let kind = match item.get("type") {
        Some(t) => t.as_str().unwrap_or("message"),
        None => "message",
    };
    match kind {
        "message" => {
            let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
            let role = if role == "developer" { "system" } else { role };
            vec![Turn::message(role, input_text(item.get("content")))]
        }
        "function_call" => {
            let id = item
                .get("call_id")
                .and_then(Value::as_str)
                .or_else(|| item.get("id").and_then(Value::as_str))
                .map(str::to_string)
                .unwrap_or_else(|| new_id("call_"));
            vec![Turn {
                role: "assistant".to_string(),
                tool_calls: vec![ConvertedCall {
                    id,
                    name: item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    arguments: item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                }],
                ..Turn::default()
            }]
        }
        "function_call_output" => vec![Turn {
            role: "tool".to_string(),
            content: Some(stringify(item.get("output"))),
            tool_call_id: Some(
                item.get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            ),
            ..Turn::default()
        }],
        "reasoning" => {
            // Summary-only and encrypted reasoning items carry no
            // recoverable text; they contribute no turn at all rather
            // than an empty one.
            let text: String = item
                .get("content")
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .filter(|p| p.get("type").and_then(Value::as_str) == Some("reasoning_text"))
                        .filter_map(|p| p.get("text").and_then(Value::as_str))
                        .collect()
                })
                .unwrap_or_default();
            if text.is_empty() {
                return Vec::new();
            }
            vec![Turn {
                role: "assistant".to_string(),
                reasoning: Some(text),
                ..Turn::default()
            }]
        }
        // Built-in tool items (web search, code interpreter, ...) are
        // not supported and are skipped rather than rendered as prose.
        _ => Vec::new(),
    }
}

/// Coalesce one assistant *run* -- reasoning, then a message, then any
/// function calls -- into a single assistant turn.
///
/// The slot rules are what keep genuinely distinct turns apart: a
/// reasoning item **always opens** a turn, and a message item only ever
/// fills a turn that has neither content nor calls yet. Merging on role
/// alone would fuse two separate assistant answers; not merging at all
/// shows the model three turns where it took one, and a model shown a
/// history it could not have produced answers the next question in that
/// shape.
fn merge_assistant_run(turns: Vec<Turn>) -> Vec<Turn> {
    let mut merged: Vec<Turn> = Vec::with_capacity(turns.len());
    for turn in turns {
        let mergeable = match merged.last() {
            Some(prev) => {
                turn.role == "assistant"
                    && prev.role == "assistant"
                    && turn.reasoning.is_none()
                    && (!turn.tool_calls.is_empty()
                        || (turn.has_text() && !prev.has_text() && prev.tool_calls.is_empty()))
            }
            None => false,
        };
        if !mergeable {
            merged.push(turn);
            continue;
        }
        let has_text = turn.has_text();
        let prev = merged
            .last_mut()
            .expect("mergeable is only true when there is a previous turn");
        if has_text {
            prev.content = turn.content;
        }
        prev.tool_calls.extend(turn.tool_calls);
    }
    merged
}

/// The text of a `content` field: a bare string, or the text parts of a
/// part list (`input_text` / `output_text` / `text`, plus anything else
/// that carries a `text`).
fn input_text(content: Option<&Value>) -> String {
    match content {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| match part {
                Value::Object(map) => {
                    let kind = map.get("type").and_then(Value::as_str);
                    let textual = matches!(kind, Some("input_text" | "output_text" | "text"))
                        || map.contains_key("text");
                    textual
                        .then(|| map.get("text").and_then(Value::as_str).unwrap_or_default())
                        .map(str::to_string)
                }
                Value::String(s) => Some(s.clone()),
                other => Some(other.to_string()),
            })
            .collect(),
        Some(other) => other.to_string(),
    }
}

/// A tool result as the text a template can render: a string verbatim,
/// anything structured as compact JSON.
fn stringify(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// Responses tool definitions in the flat shape (`{"type": "function",
/// "name", "parameters"}`) *and* the nested chat shape, since both are
/// in the wild. Built-in tools are skipped: this server has no web
/// search or code interpreter to offer, and describing one to the model
/// would invite a call nothing can answer.
fn convert_tools(tools: Option<&Vec<Value>>) -> Vec<ToolDef> {
    let Some(tools) = tools else {
        return Vec::new();
    };
    tools
        .iter()
        .filter(|tool| {
            matches!(
                tool.get("type").map(|t| t.as_str().unwrap_or_default()),
                None | Some("function")
            )
        })
        .map(|tool| {
            let function = match tool.get("function") {
                Some(inner @ Value::Object(_)) => inner,
                _ => tool,
            };
            ToolDef {
                kind: "function".to_string(),
                function: ToolFunctionDef {
                    name: function
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    description: function
                        .get("description")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    parameters: Some(
                        function
                            .get("parameters")
                            .filter(|p| !p.is_null())
                            .cloned()
                            .unwrap_or_else(|| json!({"type": "object"})),
                    ),
                },
            }
        })
        .collect()
}

/// The whole request conversion, and the only place the two structural
/// rules live.
///
/// Producing a [`ChatCompletionRequest`] rather than a prompt is the
/// point: everything after this -- template kwargs, the stop set, the
/// seed policy, the rejection of `tool_choice` values this server
/// cannot honour -- is then the *same code* the chat surface runs, so
/// the two surfaces cannot answer the same question differently.
fn to_chat_request(req: &ResponsesRequest) -> Result<ChatCompletionRequest, ApiError> {
    let max_tokens = match req.max_output_tokens {
        Some(n) if n < 1 => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "max_output_tokens must be a positive integer",
                None,
            ));
        }
        Some(n) => n as usize,
        // Rule 3. Written out rather than left to serde, which would
        // apply `default_max_tokens()` -- 16 -- and truncate every
        // answer this surface serves at a sentence and a half.
        None => DEFAULT_MAX_OUTPUT_TOKENS,
    };

    let mut system_texts: Vec<String> = Vec::new();
    if let Some(instructions) = &req.instructions {
        system_texts.push(instructions.clone());
    }
    let mut turns: Vec<Turn> = Vec::new();
    match &req.input {
        ResponsesInput::Text(text) => turns.push(Turn::message("user", text.clone())),
        ResponsesInput::Items(items) => {
            for item in items {
                for turn in convert_input_item(item) {
                    if turn.role == "system" {
                        system_texts.push(turn.content.unwrap_or_default());
                    } else {
                        turns.push(turn);
                    }
                }
            }
            turns = merge_assistant_run(turns);
        }
    }

    // Rule 1: one leading system message, never two. Pulling the system
    // items out above also means a system item sitting in the middle of
    // the input cannot break an assistant run in half.
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(turns.len() + 1);
    let system_text = system_texts
        .iter()
        .filter(|t| !t.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n\n");
    if !system_text.is_empty() {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text(system_text)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    messages.extend(turns.into_iter().filter_map(Turn::lower));

    let effort = req
        .reasoning
        .as_ref()
        .and_then(|r| r.get("effort"))
        .and_then(Value::as_str)
        .map(str::to_string);

    let request = ChatCompletionRequest {
        samplers: None,
        model: req.model.clone(),
        messages,
        max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        // Neither wire has a min_p; 0.0 (off) is what None resolves to.
        min_p: None,
        top_k: req.top_k,
        repetition_penalty: None,
        // Neither the Anthropic Messages wire nor the Responses wire
        // carries llama.cpp's extra samplers, so they resolve to their
        // neutral defaults and this chain is llama.cpp's default one
        // doing nothing extra.
        extra_samplers: Default::default(),
        // No seed field on this surface, so the chat path's policy
        // applies unchanged: an unseeded sampled request draws fresh
        // every time rather than replaying one draw forever.
        seed: None,
        stop: None,
        stream: Some(req.stream),
        // Replay is a frink extension on the chat surface with no
        // Responses spelling; codex reconnects by resending `input`.
        stream_resumable: None,
        // A serving-benchmark knob on the OpenAI surface only; this
        // protocol has no spelling for it.
        ignore_eos: None,
        tools: convert_tools(req.tools.as_ref()),
        tool_choice: req.tool_choice.clone().map(|choice| match choice {
            Value::String(mode) => ToolChoice::Mode(mode),
            other => ToolChoice::Specific(other),
        }),
        chat_template_kwargs: req.chat_template_kwargs.clone(),
        reasoning_effort: effort,
        // The DeepSeek wire's on/off switch has no Responses spelling:
        // `reasoning.effort` is the only knob this protocol carries, and
        // it arrives above.
        thinking: None,
        // Stateless surface: history comes in `input`, in full, on every
        // turn.
        session_id: None,
        // Says nothing, and gets the one rule every route gets: a
        // trailing assistant item is continued by default, exactly as
        // it is on `/v1/chat/completions` -- see `continuation`.
        continue_final_message: Default::default(),
        // llama.cpp copies every field of a Responses body onto the
        // chat body it lowers to (`server-chat.cpp:15`), so the budget
        // spelled at the top level of a Responses request reaches the
        // sampler there too.
        reasoning_budget_tokens: req.reasoning_budget_tokens,
        logprobs: None,
        top_logprobs: None,
        unimplemented: Default::default(),
        presence_penalty: None,
        frequency_penalty: None,
        response_format: None,
        // Neither surface has a grammar field of its own yet. Named
        // rather than defaulted so that adding one is a compile error
        // here first, instead of a constraint silently dropped on the
        // way through the chat request these translate into.
        grammar: None,
        // Not on the Responses wire.
        logit_bias: None,
        lora: None,
    };
    request.validate_supported_fields()?;
    Ok(request)
}

// ---------------------------------------------------------------------
// Response and event bodies
// ---------------------------------------------------------------------

fn error_response(status: StatusCode, message: &str, code: Option<&str>) -> ApiError {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
            }
        })),
    )
}

/// The machine-readable code that rides on `response.failed`.
///
/// `context_length_exceeded` is the one codex actually branches on --
/// it compacts the conversation and retries instead of surfacing an
/// error -- and it comes from [`frink_models::Ceiling::code`], which
/// this workspace already documents as safe for a client to match on.
/// The rest are stable but only diagnostic.
fn failure_code(error: &DecodeError) -> &'static str {
    match error {
        DecodeError::KvBudgetExceeded { binding, .. } => binding,
        DecodeError::TokenOutOfVocab { .. } => "invalid_prompt",
        // A valid request this server cannot serve, which is the same
        // thing the 501 on the wire says.
        DecodeError::Unsupported(_) => "unsupported",
        DecodeError::KvPoolExhausted | DecodeError::QueueFull { .. } => "server_overloaded",
        // The request named a constraint this model cannot satisfy, so
        // it is the request that is invalid -- retrying it unchanged
        // fails identically.
        DecodeError::GrammarConstraint { .. } => "invalid_grammar",
        // Likewise: the budget named a family, or reached the sampler
        // in a shape, this server cannot enforce.
        DecodeError::ReasoningBudget { .. } => "invalid_reasoning_budget",
    }
}

/// `input_tokens` stays inclusive of any cached prefix (OpenAI
/// semantics); `cached_tokens` is 0 when no prefix cache is configured,
/// which is the same thing `Usage::cached_tokens: None` means.
fn usage_json(usage: &Usage) -> Value {
    let mut v = json!({
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        "input_tokens_details": {
            "cached_tokens": usage.cached_tokens.unwrap_or(0),
            "cache_write_tokens": 0,
        },
    });
    // Present only when the split actually ran. This used to be a
    // hardcoded `0`, which told every caller that every model on every
    // request thought for exactly no tokens (#120).
    if let Some(d) = usage.completion_tokens_details.as_ref() {
        v["output_tokens_details"] = json!({"reasoning_tokens": d.reasoning_tokens});
    }
    v
}

fn reasoning_item(id: &str, text: Option<&str>, status: &str) -> Value {
    json!({
        "id": id,
        "type": "reasoning",
        "summary": [],
        "content": match text {
            Some(text) => json!([{"type": "reasoning_text", "text": text}]),
            None => json!([]),
        },
        "status": status,
    })
}

fn message_item(id: &str, text: Option<&str>, status: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": match text {
            Some(text) => json!([{"type": "output_text", "text": text, "annotations": []}]),
            None => json!([]),
        },
    })
}

fn function_call_item(id: &str, call_id: &str, name: &str, arguments: &str, status: &str) -> Value {
    json!({
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    })
}

fn output_text_part(text: &str) -> Value {
    json!({"type": "output_text", "text": text, "annotations": []})
}

/// One `Response` object, in every place one appears: the non-streaming
/// body and the `response` field of `created` / `in_progress` /
/// `completed` / `incomplete` / `failed`.
#[allow(clippy::too_many_arguments)]
fn response_object(
    response_id: &str,
    created: u64,
    model: &str,
    output: Vec<Value>,
    status: &str,
    usage: Option<Value>,
    error: Option<Value>,
    incomplete_reason: Option<&str>,
) -> Value {
    json!({
        "id": response_id,
        "created_at": created,
        "model": model,
        "object": "response",
        "output": output,
        "status": status,
        "usage": usage,
        "error": error,
        "incomplete_details": incomplete_reason.map(|reason| json!({"reason": reason})),
        "parallel_tool_calls": true,
        "tool_choice": "auto",
        "tools": [],
    })
}

/// The whole buffered answer as one `Response`.
///
/// Rule 4's non-streaming half: a `length` finish is `incomplete` with
/// `incomplete_details.reason = "max_output_tokens"`, and only the
/// *message* item carries the `incomplete` status -- a reasoning block
/// that finished before the budget ran out did finish, and a tool call
/// that parsed is a call the client may execute.
fn build_response(
    parsed: output::ParsedOutput,
    finish: &str,
    usage: &Usage,
    response_id: &str,
    created: u64,
    model: &str,
) -> Value {
    let truncated = finish == "length";
    let item_status = if truncated { "incomplete" } else { "completed" };
    let mut output = Vec::new();
    if let Some(reasoning) = parsed.reasoning.as_deref().filter(|r| !r.is_empty()) {
        output.push(reasoning_item(&new_id("rs_"), Some(reasoning), "completed"));
    }
    if !parsed.content.is_empty() {
        output.push(message_item(
            &new_id("msg_"),
            Some(&parsed.content),
            item_status,
        ));
    }
    for call in &parsed.calls {
        output.push(function_call_item(
            &new_id("fc_"),
            &new_id("call_"),
            &call.name,
            &call.arguments,
            "completed",
        ));
    }
    response_object(
        response_id,
        created,
        model,
        output,
        if truncated { "incomplete" } else { "completed" },
        Some(usage_json(usage)),
        None,
        truncated.then_some("max_output_tokens"),
    )
}

// ---------------------------------------------------------------------
// Streaming: semantic events -> Responses stream events
// ---------------------------------------------------------------------

/// What the generation thread tells the stream, in the vocabulary the
/// engine already speaks (the policy parser's events plus a
/// terminal). Keeping this protocol-neutral is what lets every
/// sequencing rule below be tested with no model and no socket.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GenEvent {
    Reasoning(String),
    Content(String),
    CallStart {
        index: usize,
        name: String,
    },
    CallArguments {
        index: usize,
        fragment: String,
    },
    CallEnd {
        index: usize,
        arguments: String,
    },
    /// A whole call from a path that never streamed it (the
    /// continuous-batching fallback, which returns one string).
    WholeCall {
        index: usize,
        name: String,
        arguments: String,
    },
    Done {
        finish: &'static str,
        usage: Usage,
    },
    Failed {
        code: &'static str,
        message: String,
    },
    /// Injected by [`with_keepalive`], never by the generator.
    Keepalive,
}

impl crate::stream_events::StreamEvent for GenEvent {
    fn keepalive() -> Self {
        GenEvent::Keepalive
    }
    fn content(text: String) -> Self {
        GenEvent::Content(text)
    }
    fn call_start(index: usize, name: String) -> Self {
        GenEvent::CallStart { index, name }
    }
    fn call_arguments(index: usize, fragment: String) -> Self {
        GenEvent::CallArguments { index, fragment }
    }
    fn call_end(index: usize, arguments: String) -> Self {
        GenEvent::CallEnd { index, arguments }
    }
}

/// One SSE frame: the event name that goes in `event:` and the object
/// that goes in `data:`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Frame {
    name: &'static str,
    data: Value,
}

impl Frame {
    fn into_event(self) -> Event {
        // Serializing a `Value` cannot fail; the fallback keeps a
        // serialization bug from taking the stream down with a panic.
        let data = serde_json::to_string(&self.data).unwrap_or_else(|e| {
            tracing::error!("failed to serialize a responses stream event: {e}");
            "{}".to_string()
        });
        Event::default().event(self.name).data(data)
    }
}

/// The item currently open on the stream. At most one is open at a
/// time; anything that starts a different kind closes it first.
enum Current {
    Reasoning {
        id: String,
        text: String,
    },
    Message {
        id: String,
        text: String,
    },
    Call {
        id: String,
        call_id: String,
        name: String,
        args: String,
        ordinal: usize,
    },
}

/// Turns the semantic event stream into Responses stream events.
///
/// Split out of the handler so the sequencing rules are testable
/// directly: every test below drives this with a `Vec<GenEvent>`.
pub(crate) struct ResponsesStream {
    response_id: String,
    created: u64,
    model: String,
    seq: u64,
    output_index: usize,
    output: Vec<Value>,
    current: Option<Current>,
    /// `"stop"` until the terminal event arrives, which is precisely
    /// what makes rule 4 work: an item closed before then cannot know
    /// about a truncation that has not happened yet, so it closes
    /// `completed` and only the final one is `incomplete`.
    finish: &'static str,
    usage: Option<Usage>,
}

impl ResponsesStream {
    pub(crate) fn new(response_id: String, created: u64, model: String) -> Self {
        ResponsesStream {
            response_id,
            created,
            model,
            seq: 0,
            output_index: 0,
            output: Vec::new(),
            current: None,
            finish: "stop",
            usage: None,
        }
    }

    fn next_seq(&mut self) -> u64 {
        let n = self.seq;
        self.seq += 1;
        n
    }

    /// Stamps `type` and `sequence_number` on every frame in one place,
    /// so no event can go out unnumbered (codex orders on it).
    fn frame(&mut self, name: &'static str, mut data: Value) -> Frame {
        let seq = self.next_seq();
        if let Some(object) = data.as_object_mut() {
            object.insert("type".to_string(), json!(name));
            object.insert("sequence_number".to_string(), json!(seq));
        }
        Frame { name, data }
    }

    fn snapshot(
        &self,
        status: &str,
        usage: Option<Value>,
        incomplete_reason: Option<&str>,
    ) -> Value {
        response_object(
            &self.response_id,
            self.created,
            &self.model,
            self.output.clone(),
            status,
            usage,
            None,
            incomplete_reason,
        )
    }

    /// `response.created` + `response.in_progress`, before anything has
    /// been generated. Emitted from the handler the moment the SSE
    /// headers go out, so codex has the response id even if the model
    /// never produces a token.
    pub(crate) fn opening(&mut self) -> Vec<Frame> {
        let created = self.snapshot("in_progress", None, None);
        let in_progress = self.snapshot("in_progress", None, None);
        vec![
            self.frame("response.created", json!({"response": created})),
            self.frame("response.in_progress", json!({"response": in_progress})),
        ]
    }

    fn close_current(&mut self) -> Vec<Frame> {
        let Some(current) = self.current.take() else {
            return Vec::new();
        };
        let output_index = self.output_index;
        let mut frames = Vec::new();
        let done_item = match current {
            Current::Reasoning { id, text } => {
                frames.push(self.frame(
                    "response.reasoning_text.done",
                    json!({
                        "item_id": id,
                        "output_index": output_index,
                        "content_index": 0,
                        "text": text,
                    }),
                ));
                reasoning_item(&id, Some(&text), "completed")
            }
            Current::Message { id, text } => {
                frames.push(self.frame(
                    "response.output_text.done",
                    json!({
                        "item_id": id,
                        "output_index": output_index,
                        "content_index": 0,
                        "text": text,
                        "logprobs": [],
                    }),
                ));
                frames.push(self.frame(
                    "response.content_part.done",
                    json!({
                        "item_id": id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": output_text_part(&text),
                    }),
                ));
                let status = if self.finish == "length" {
                    "incomplete"
                } else {
                    "completed"
                };
                message_item(&id, Some(&text), status)
            }
            Current::Call {
                id,
                call_id,
                name,
                args,
                ordinal: _,
            } => {
                frames.push(self.frame(
                    "response.function_call_arguments.done",
                    json!({
                        "item_id": id,
                        "output_index": output_index,
                        "name": name,
                        "arguments": args,
                    }),
                ));
                function_call_item(&id, &call_id, &name, &args, "completed")
            }
        };
        frames.push(self.frame(
            "response.output_item.done",
            json!({"output_index": output_index, "item": done_item.clone()}),
        ));
        self.output.push(done_item);
        self.output_index += 1;
        frames
    }

    fn open_call(&mut self, name: &str, ordinal: usize) -> Vec<Frame> {
        let mut frames = self.close_current();
        let id = new_id("fc_");
        let call_id = new_id("call_");
        let item = function_call_item(&id, &call_id, name, "", "in_progress");
        let output_index = self.output_index;
        self.current = Some(Current::Call {
            id,
            call_id,
            name: name.to_string(),
            args: String::new(),
            ordinal,
        });
        frames.push(self.frame(
            "response.output_item.added",
            json!({"output_index": output_index, "item": item}),
        ));
        frames
    }

    fn arguments_delta(&mut self, fragment: &str) -> Option<Frame> {
        let (id, output_index) = match &mut self.current {
            Some(Current::Call { id, .. }) => (id.clone(), self.output_index),
            // Defensive: a fragment with no call open is dropped rather
            // than opening a nameless item the client cannot execute.
            _ => return None,
        };
        if let Some(Current::Call { args, .. }) = &mut self.current {
            args.push_str(fragment);
        }
        Some(self.frame(
            "response.function_call_arguments.delta",
            json!({
                "item_id": id,
                "output_index": output_index,
                "delta": fragment,
            }),
        ))
    }

    /// One semantic event in, its stream events out.
    pub(crate) fn push(&mut self, event: GenEvent) -> Vec<Frame> {
        match event {
            // Rule 5. A data-bearing frame, so codex's stream-idle
            // countdown resets; an SSE comment would not reset it and
            // the client would reconnect mid-answer.
            GenEvent::Keepalive => {
                let snapshot = self.snapshot("in_progress", None, None);
                vec![self.frame("response.in_progress", json!({"response": snapshot}))]
            }
            GenEvent::Reasoning(text) => {
                if text.is_empty() {
                    return Vec::new();
                }
                let mut frames = Vec::new();
                if !matches!(self.current, Some(Current::Reasoning { .. })) {
                    frames.extend(self.close_current());
                    let id = new_id("rs_");
                    let item = reasoning_item(&id, None, "in_progress");
                    let output_index = self.output_index;
                    self.current = Some(Current::Reasoning {
                        id,
                        text: String::new(),
                    });
                    frames.push(self.frame(
                        "response.output_item.added",
                        json!({"output_index": output_index, "item": item}),
                    ));
                }
                let (id, output_index) = match &mut self.current {
                    Some(Current::Reasoning { id, text: buffered }) => {
                        buffered.push_str(&text);
                        (id.clone(), self.output_index)
                    }
                    _ => unreachable!("a reasoning item was just opened"),
                };
                frames.push(self.frame(
                    "response.reasoning_text.delta",
                    json!({
                        "item_id": id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": text,
                    }),
                ));
                frames
            }
            GenEvent::Content(text) => {
                if text.is_empty() {
                    return Vec::new();
                }
                let mut frames = Vec::new();
                if !matches!(self.current, Some(Current::Message { .. })) {
                    frames.extend(self.close_current());
                    let id = new_id("msg_");
                    let item = message_item(&id, None, "in_progress");
                    let output_index = self.output_index;
                    self.current = Some(Current::Message {
                        id: id.clone(),
                        text: String::new(),
                    });
                    frames.push(self.frame(
                        "response.output_item.added",
                        json!({"output_index": output_index, "item": item}),
                    ));
                    frames.push(self.frame(
                        "response.content_part.added",
                        json!({
                            "item_id": id,
                            "output_index": output_index,
                            "content_index": 0,
                            "part": output_text_part(""),
                        }),
                    ));
                }
                let (id, output_index) = match &mut self.current {
                    Some(Current::Message { id, text: buffered }) => {
                        buffered.push_str(&text);
                        (id.clone(), self.output_index)
                    }
                    _ => unreachable!("a message item was just opened"),
                };
                frames.push(self.frame(
                    "response.output_text.delta",
                    json!({
                        "item_id": id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": text,
                        "logprobs": [],
                    }),
                ));
                frames
            }
            GenEvent::CallStart { index, name } => self.open_call(&name, index),
            GenEvent::CallArguments { fragment, .. } => {
                self.arguments_delta(&fragment).into_iter().collect()
            }
            GenEvent::CallEnd { index, arguments } => {
                let open = match &self.current {
                    Some(Current::Call { ordinal, args, .. }) if *ordinal == index => {
                        Some(args.clone())
                    }
                    _ => None,
                };
                let Some(streamed) = open else {
                    // A close with nothing open: the call was never
                    // announced, so there is nothing to close.
                    return Vec::new();
                };
                let mut frames = Vec::new();
                // The final arguments are authoritative: top up whatever
                // has not gone out yet, so a client concatenating deltas
                // ends up with exactly this string.
                if let Some(remainder) = arguments.strip_prefix(streamed.as_str()) {
                    if !remainder.is_empty() {
                        frames.extend(self.arguments_delta(remainder));
                    }
                } else if let Some(Current::Call { args, .. }) = &mut self.current {
                    *args = arguments;
                }
                frames.extend(self.close_current());
                frames
            }
            GenEvent::WholeCall {
                index,
                name,
                arguments,
            } => {
                // Open, deliver, close immediately, so codex persists
                // the item even if the stream dies straight after.
                let mut frames = self.open_call(&name, index);
                frames.extend(self.arguments_delta(&arguments));
                frames.extend(self.close_current());
                frames
            }
            GenEvent::Done { finish, usage } => {
                // Set before `close_current`, which is the whole of rule
                // 4: the item still open when the budget ran out is the
                // one marked incomplete, and no earlier one is.
                self.finish = finish;
                let usage_value = usage_json(&usage);
                self.usage = Some(usage);
                let mut frames = self.close_current();
                if finish == "length" {
                    let snapshot =
                        self.snapshot("incomplete", Some(usage_value), Some("max_output_tokens"));
                    frames.push(self.frame("response.incomplete", json!({"response": snapshot})));
                } else {
                    let snapshot = self.snapshot("completed", Some(usage_value), None);
                    frames.push(self.frame("response.completed", json!({"response": snapshot})));
                }
                frames
            }
            GenEvent::Failed { code, message } => {
                let mut frames = self.close_current();
                let usage = self.usage.as_ref().map(usage_json);
                let response = response_object(
                    &self.response_id,
                    self.created,
                    &self.model,
                    self.output.clone(),
                    "failed",
                    usage,
                    Some(json!({"code": code, "message": message})),
                    None,
                );
                frames.push(self.frame("response.failed", json!({"response": response})));
                frames
            }
        }
    }
}

// ---------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------

/// `POST /v1/responses`.
///
/// Mount at [`ROUTE`], behind the same `FRINK_API_KEY` gate as
/// `/v1/chat/completions`: it decodes tokens, so it must cost what
/// decoding tokens costs.
pub(crate) async fn responses(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    let attribution = attribution::Attribution::from_headers(&headers);
    state
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let started = std::time::Instant::now();
    let request_id = frink_api::next_request_id();
    let stream = req.stream;

    let result = async {
        crate::cache_admin::check_admission(&state)?;
        if req.background {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "background mode is not supported",
                None,
            ));
        }
        if req.previous_response_id.is_some() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "previous_response_id is not supported (stateless server); resend the full \
                 context in 'input'",
                None,
            ));
        }
        let chat = to_chat_request(&req)?;
        if stream {
            responses_stream(
                state.clone(),
                chat,
                request_id.clone(),
                started,
                attribution.clone(),
            )
            .await
        } else {
            responses_full(
                state.clone(),
                chat,
                request_id.clone(),
                started,
                attribution.clone(),
            )
            .await
        }
    }
    .await;

    let response = match result {
        Ok(response) => response,
        Err(err) => err.into_response(),
    };
    if response.status().is_client_error() || response.status().is_server_error() {
        state
            .request_errors_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Only failures are recorded here; a success records itself from
        // the path that knows the token counts, and for a stream that
        // has not happened yet.
        state.record_request(stats::Record {
            request_id: &request_id,
            route: ROUTE,
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

/// `GET /v1/responses/{response_id}` -- a 404 with a reason.
///
/// The endpoint exists rather than 404-ing from the router because the
/// two 404s say different things: this server *has* the route and does
/// not keep responses, which tells a client to stop polling instead of
/// to check its base URL.
pub(crate) async fn responses_get(Path(response_id): Path<String>) -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        &format!("response {response_id:?} not found (stateless server)"),
        None,
    )
    .into_response()
}

/// `POST /v1/responses/{response_id}/cancel` -- see [`responses_get`].
/// A live generation is cancelled through `POST /v1/cancel` with the
/// `request_id`, or by dropping the connection.
pub(crate) async fn responses_cancel(Path(response_id): Path<String>) -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        &format!("response {response_id:?} not found (stateless server)"),
        None,
    )
    .into_response()
}

async fn responses_full(
    state: Arc<AppState>,
    chat: ChatCompletionRequest,
    request_id: String,
    started: std::time::Instant,
    attribution: attribution::Attribution,
) -> Result<Response, ApiError> {
    // Cloned once, up front: this request decodes against exactly this
    // model even if `/admin/models/load` swaps another one in halfway.
    let active = state.require_active()?;
    let template = active.generative()?.chat_template();
    let kwargs = chat.resolve_template_kwargs(&template);
    let offered = offered_tools(&chat);
    let prompt = chat.render_prompt(&chat.messages, &template, &offered, kwargs, active.name())?;
    let posture = OutputPosture::resolve_with(active.reasoning_format(), active.name(), &prompt);
    let params =
        chat.generation_params_for_template(&template, active.name(), active.sampler_model())?;

    let (choices, usage) = crate::decode_task::buffered(
        crate::decode_task::DecodeHandles::take(&state, &active)?,
        prompt,
        params,
    )
    .await?;

    // Choice 0: this wire has no `n`, and `crate::unimplemented_fields`
    // refuses the field.
    let (finish, generated) = choices
        .into_iter()
        .next()
        .expect("a generation produces at least one choice");
    let parsed = output::parse_output(&generated, &offered, posture);
    state.record_request(stats::Record {
        request_id: &request_id,
        route: ROUTE,
        // The handle this request decoded against, not `chat.model`.
        model: Some(active.name().to_string()),
        status: 200,
        stream: false,
        duration_ms: started.elapsed().as_millis() as u64,
        usage: Some(&usage),
        attribution: &attribution,
    });
    Ok(Json(build_response(
        parsed,
        finish.as_str(),
        &usage,
        &new_id("resp_"),
        unix_seconds(),
        &chat.model,
    ))
    .into_response())
}

/// The tools this request really offers: empty when `tool_choice:
/// "none"` disabled them, so the model is never told about tools the
/// caller asked it not to use.
fn offered_tools(chat: &ChatCompletionRequest) -> Vec<ToolDef> {
    if chat.tools_active() {
        chat.tools.clone()
    } else {
        Vec::new()
    }
}

async fn responses_stream(
    state: Arc<AppState>,
    chat: ChatCompletionRequest,
    request_id: String,
    started: std::time::Instant,
    attribution: attribution::Attribution,
) -> Result<Response, ApiError> {
    // See `responses_full`: the handle is taken once and the whole
    // stream runs against it, so a mid-stream model swap cannot splice
    // two checkpoints into one answer.
    let active = state.require_active()?;
    let template = active.generative()?.chat_template();
    let kwargs = chat.resolve_template_kwargs(&template);
    let offered = offered_tools(&chat);
    let served_model = active.name().to_string();
    let prompt = chat.render_prompt(&chat.messages, &template, &offered, kwargs, &served_model)?;
    let posture = OutputPosture::resolve_with(active.reasoning_format(), &served_model, &prompt);
    let mut params =
        chat.generation_params_for_template(&template, &served_model, active.sampler_model())?;

    // The same two-tier cancellation the chat stream has: the guard
    // rides with the generation task and deregisters however that task
    // ends, panic included.
    let (cancel_token, cancel_guard) = state.cancels.register(&request_id);
    params.cancel = Some(cancel_token.clone());

    let model = Arc::clone(active.generative()?);
    let kv_pool = state.kv_pool.clone();
    let paged_kv = state.paged_kv.clone();
    let prefix_cache = state.prefix_cache.clone();
    let batcher = active.batcher.clone();
    let ceiling = active.ceiling.clone();
    let metal_private_decode_gate = state.metal_private_decode_gate.clone();
    // Continuous batching returns one string, so there is no
    // incremental stream to ride on and the whole answer is parsed at
    // the end instead.
    let overlap = true;
    let stats_state = Arc::clone(&state);
    let stats_request_id = request_id.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);
    tokio::task::spawn_blocking(move || {
        let _cancel_guard = cancel_guard;
        let orphan = sse::orphan_timeout_from_env();
        let send = |event: GenEvent| {
            if sse::send_or_orphan(&tx, event, orphan).is_err() {
                // The reader is gone or has stopped reading. This
                // stream keeps no replay buffer, so there is nothing
                // left to generate for.
                cancel_token.cancel();
            }
        };
        let mut reasoning = posture.reasoning_parser();
        let mut tools = (!offered.is_empty()).then(|| posture.tool_call_parser(&offered));
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
                let (thought, content) = match reasoning.as_mut() {
                    Some(parser) => {
                        let delta = parser.push(chunk);
                        (delta.reasoning, delta.content)
                    }
                    None => (String::new(), chunk.to_string()),
                };
                if !thought.is_empty() {
                    send(GenEvent::Reasoning(thought));
                }
                match tools.as_mut() {
                    Some(parser) => {
                        for event in parser.push(&content) {
                            for mapped in map_tool_event(event) {
                                send(mapped);
                            }
                        }
                    }
                    None if !content.is_empty() => send(GenEvent::Content(content)),
                    None => {}
                }
            },
        );
        match result {
            // Streaming, so exactly one choice: `n` > 1 with `stream`
            // is refused at the route.
            Ok((choices, usage)) => {
                let (finish, full_text) = choices
                    .into_iter()
                    .next()
                    .expect("a generation produces at least one choice");
                if overlap {
                    // Both parsers may still be withholding a run that
                    // could have become a marker and did not. It is
                    // ordinary output; dropping it truncates every
                    // answer whose tail looks like half a `</think>`.
                    let tail = reasoning.as_mut().map(|p| p.flush()).unwrap_or_default();
                    if !tail.reasoning.is_empty() {
                        send(GenEvent::Reasoning(tail.reasoning));
                    }
                    match tools.as_mut() {
                        Some(parser) => {
                            let mut events = parser.push(&tail.content);
                            events.extend(parser.finish());
                            for event in events {
                                for mapped in map_tool_event(event) {
                                    send(mapped);
                                }
                            }
                        }
                        None if !tail.content.is_empty() => send(GenEvent::Content(tail.content)),
                        None => {}
                    }
                } else {
                    let parsed = output::parse_output(&full_text, &offered, posture);
                    if let Some(thought) = parsed.reasoning.filter(|r| !r.is_empty()) {
                        send(GenEvent::Reasoning(thought));
                    }
                    if !parsed.content.is_empty() {
                        send(GenEvent::Content(parsed.content));
                    }
                    for (index, call) in parsed.calls.into_iter().enumerate() {
                        send(GenEvent::WholeCall {
                            index,
                            name: call.name,
                            arguments: call.arguments,
                        });
                    }
                }
                stats_state.record_request(stats::Record {
                    request_id: &stats_request_id,
                    route: ROUTE,
                    model: Some(served_model.clone()),
                    status: 200,
                    stream: true,
                    duration_ms: started.elapsed().as_millis() as u64,
                    usage: Some(&usage),
                    attribution: &attribution,
                });
                send(GenEvent::Done {
                    finish: finish.as_str(),
                    usage,
                });
            }
            Err(e) => {
                tracing::warn!("decode error on streamed response {stats_request_id}: {e}");
                // The socket carried 200 -- SSE headers precede the
                // first token -- but the request produced no answer. A
                // 200 row with zero tokens would read as a successful
                // empty response, so the failure is stated as 500 here
                // and only here.
                stats_state.record_request(stats::Record {
                    request_id: &stats_request_id,
                    route: ROUTE,
                    model: Some(served_model.clone()),
                    status: 500,
                    stream: true,
                    duration_ms: started.elapsed().as_millis() as u64,
                    usage: None,
                    attribution: &attribution,
                });
                send(GenEvent::Failed {
                    code: failure_code(&e),
                    message: e.to_string(),
                });
            }
        }
    });

    let mut machine = ResponsesStream::new(new_id("resp_"), unix_seconds(), chat.model.clone());
    let queue: VecDeque<Frame> = machine.opening().into();
    // Boxed because `StreamExt::next` needs `Unpin` and an `unfold`
    // over an async block is not.
    let events = Box::pin(with_keepalive(rx, KEEPALIVE_INTERVAL));
    let stream = futures_util::stream::unfold(
        (machine, events, queue),
        |(mut machine, mut events, mut queue)| async move {
            loop {
                if let Some(frame) = queue.pop_front() {
                    return Some((
                        Ok::<Event, Infallible>(frame.into_event()),
                        (machine, events, queue),
                    ));
                }
                let event = events.next().await?;
                queue.extend(machine.push(event));
            }
        },
    );

    // `X-Accel-Buffering: no` for the same reason the chat stream sets
    // it: nginx and everything that copied its conventions buffer
    // `text/event-stream` by default, which turns a token stream into
    // one silent wait.
    //
    // **No `keep_alive`.** Rule 5: axum's keep-alive is an SSE comment,
    // and a comment does not reset codex's stream-idle timeout. The
    // keepalive here is a real `response.in_progress` frame, inserted
    // by `with_keepalive` above.
    Ok((
        [(
            axum::http::HeaderName::from_static("x-accel-buffering"),
            axum::http::HeaderValue::from_static("no"),
        )],
        Sse::new(stream),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: Value) -> ResponsesRequest {
        serde_json::from_value(value).expect("the fixture is a valid Responses request")
    }

    fn converted(value: Value) -> ChatCompletionRequest {
        to_chat_request(&request(value)).expect("the fixture converts")
    }

    fn text_of(message: &ChatMessage) -> String {
        message
            .content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default()
    }

    fn usage(prompt: usize, completion: usize) -> Usage {
        Usage::new(prompt, completion)
    }

    /// `/v1/responses` reported `reasoning_tokens: 0` for every answer
    /// from every model, which is a claim that the model did not think,
    /// not an admission that nobody counted (#120).
    #[test]
    fn a_model_that_did_not_think_omits_the_reasoning_field_entirely() {
        let v = usage_json(&usage(10, 20));
        assert_eq!(v["input_tokens"], 10);
        assert_eq!(v["output_tokens"], 20);
        assert!(
            v.get("output_tokens_details").is_none(),
            "a checkpoint with no reasoning format must report NO details \
             object, not a zero it did not earn: {v}"
        );
    }

    #[test]
    fn a_counted_reasoning_split_reaches_the_wire() {
        let v = usage_json(&usage(10, 20).with_reasoning_tokens(7));
        assert_eq!(
            v["output_tokens_details"]["reasoning_tokens"], 7,
            "the counted split must be what the wire carries: {v}"
        );
    }

    /// Zero is a real answer when the split RAN and found no reasoning,
    /// and it has to survive as a present zero rather than being folded
    /// back into "absent".
    #[test]
    fn a_split_that_ran_and_found_nothing_reports_a_present_zero() {
        let v = usage_json(&usage(10, 20).with_reasoning_tokens(0));
        assert_eq!(v["output_tokens_details"]["reasoning_tokens"], 0);
        assert!(v.get("output_tokens_details").is_some());
    }

    /// Drive the stream machine with a script of semantic events.
    fn run(events: Vec<GenEvent>) -> Vec<Frame> {
        let mut machine =
            ResponsesStream::new("resp_test".to_string(), 1_700_000_000, "m".to_string());
        let mut frames = machine.opening();
        for event in events {
            frames.extend(machine.push(event));
        }
        frames
    }

    fn names(frames: &[Frame]) -> Vec<&str> {
        frames.iter().map(|f| f.name).collect()
    }

    fn last_named<'a>(frames: &'a [Frame], name: &str) -> &'a Value {
        &frames
            .iter()
            .rev()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("no {name} frame in {:?}", names(frames)))
            .data
    }

    // -----------------------------------------------------------------
    // Rule 1: one leading system message
    // -----------------------------------------------------------------

    /// codex sends its system prompt as `instructions` **and** a
    /// `developer` message in the same request. Done the obvious way --
    /// one converted message per input item, `developer` passed through
    /// -- the model gets two system-ish turns, and a strict template
    /// (Qwen3.5) refuses the whole request with "System message must be
    /// at the beginning". This test fails on that shape: it asserts
    /// exactly one `system` message, in front, holding both texts.
    #[test]
    fn instructions_and_a_developer_message_merge_into_one_leading_system_message() {
        let chat = converted(json!({
            "model": "m",
            "instructions": "You are codex.",
            "input": [
                {"type": "message", "role": "developer", "content": "Never write to disk."},
                {"type": "message", "role": "user", "content": "hi"},
            ],
        }));
        let systems: Vec<&ChatMessage> = chat
            .messages
            .iter()
            .filter(|m| m.role == "system")
            .collect();
        assert_eq!(systems.len(), 1, "{:?}", chat.messages);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(
            text_of(&chat.messages[0]),
            "You are codex.\n\nNever write to disk."
        );
        assert_eq!(chat.messages[1].role, "user");
        assert!(
            !chat.messages.iter().any(|m| m.role == "developer"),
            "a chat template knows no developer role"
        );
    }

    /// A system item arriving *after* the conversation started still
    /// leads it, and -- because it is lifted out before the assistant
    /// run is coalesced -- it does not cut that run in half.
    #[test]
    fn a_system_item_in_the_middle_of_the_input_still_leads_the_conversation() {
        let chat = converted(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "message", "role": "system", "content": "Be brief."},
                {"type": "message", "role": "user", "content": "again"},
            ],
        }));
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(text_of(&chat.messages[0]), "Be brief.");
        assert_eq!(chat.messages.len(), 3);
    }

    // -----------------------------------------------------------------
    // Rule 2: one assistant turn per assistant run
    // -----------------------------------------------------------------

    /// The replay codex sends after a tool round-trip: a reasoning item,
    /// the assistant's text, and the call it made. Converted one message
    /// per item -- the obvious way -- the model is shown three assistant
    /// turns in a row for one turn it actually took, and answers the
    /// next question in that invented shape. This test asserts the run
    /// is one message carrying both the text and the call.
    #[test]
    fn a_reasoning_message_and_call_run_becomes_one_assistant_turn() {
        let chat = converted(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "weather?"},
                {"type": "reasoning", "content": [
                    {"type": "reasoning_text", "text": "I should look it up."}
                ]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Let me check."}]},
                {"type": "function_call", "call_id": "call_7", "name": "get_weather",
                 "arguments": "{\"city\": \"Rome\"}"},
                {"type": "function_call_output", "call_id": "call_7", "output": "sunny"},
            ],
        }));
        let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(
            roles,
            vec!["user", "assistant", "tool"],
            "{:?}",
            chat.messages
        );
        let assistant = &chat.messages[1];
        assert_eq!(text_of(assistant), "Let me check.");
        let calls = assistant
            .tool_calls
            .as_ref()
            .expect("the call is on the turn");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("call_7"));
        assert_eq!(text_of(&chat.messages[2]), "sunny");
    }

    /// The other half of the rule, and the reason merging cannot simply
    /// key on the role: two assistant *answers* are two turns. A merge
    /// that fused them would blank the first one's text with the
    /// second's.
    #[test]
    fn two_separate_assistant_messages_do_not_merge_into_one_turn() {
        let chat = converted(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "assistant", "content": "first"},
                {"type": "message", "role": "assistant", "content": "second"},
            ],
        }));
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(text_of(&chat.messages[0]), "first");
        assert_eq!(text_of(&chat.messages[1]), "second");
    }

    /// A reasoning item always OPENS a turn: the message before it
    /// belongs to the previous turn, not to the one the reasoning
    /// starts. Merging on role alone would collapse both turns into one.
    #[test]
    fn a_reasoning_item_always_opens_a_new_assistant_turn() {
        let chat = converted(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "assistant", "content": "done thinking"},
                {"type": "reasoning", "content": [
                    {"type": "reasoning_text", "text": "now the call"}
                ]},
                {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            ],
        }));
        assert_eq!(chat.messages.len(), 2, "{:?}", chat.messages);
        assert_eq!(text_of(&chat.messages[0]), "done thinking");
        assert!(chat.messages[0].tool_calls.is_none());
        assert!(chat.messages[1].tool_calls.is_some());
    }

    /// Two calls in one assistant turn stay in one turn, in order.
    #[test]
    fn consecutive_function_calls_accumulate_on_the_same_turn() {
        let chat = converted(json!({
            "model": "m",
            "input": [
                {"type": "function_call", "call_id": "a", "name": "one", "arguments": "{}"},
                {"type": "function_call", "call_id": "b", "name": "two", "arguments": "{}"},
            ],
        }));
        assert_eq!(chat.messages.len(), 1);
        let calls = chat.messages[0].tool_calls.as_ref().expect("both calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].function.name, "two");
    }

    /// A replayed chain of thought survives to the template, and lands
    /// on `reasoning_content` rather than on `content`.
    ///
    /// The distinction is the whole point: codex replays every turn's
    /// reasoning on the next request, and folding it into `content`
    /// would show the model its own deliberation as something it said
    /// out loud -- which is what the family's thinking markers exist to
    /// prevent. A template that does not know the key simply never
    /// reads it.
    #[test]
    fn a_replayed_reasoning_item_lands_beside_the_answer_and_not_inside_it() {
        let chat = converted(json!({
            "model": "m",
            "input": [
                {"type": "reasoning", "content": [
                    {"type": "reasoning_text", "text": "I should look it up."}
                ]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Let me check."}]},
            ],
        }));
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(
            chat.messages[0].reasoning_content.as_deref(),
            Some("I should look it up.")
        );
        assert_eq!(
            text_of(&chat.messages[0]),
            "Let me check.",
            "the visible answer must carry only what the model said"
        );
    }

    /// A reasoning item with nothing recoverable in it (summary-only or
    /// encrypted) contributes no turn at all, rather than an empty
    /// assistant turn the model would read as a silence it produced.
    #[test]
    fn a_reasoning_item_with_no_text_contributes_no_turn() {
        let chat = converted(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "reasoning", "summary": [], "encrypted_content": "..."},
            ],
        }));
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
    }

    #[test]
    fn a_bare_string_input_becomes_one_user_message() {
        let chat = converted(json!({"model": "m", "input": "hello"}));
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(text_of(&chat.messages[0]), "hello");
    }

    #[test]
    fn a_content_part_list_is_flattened_to_its_text() {
        let chat = converted(json!({
            "model": "m",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "a"},
                {"type": "input_text", "text": "b"},
            ]}],
        }));
        assert_eq!(text_of(&chat.messages[0]), "ab");
    }

    /// A structured tool result is JSON, not `[object Object]`.
    #[test]
    fn a_structured_function_call_output_is_stringified_as_json() {
        let chat = converted(json!({
            "model": "m",
            "input": [{"type": "function_call_output", "call_id": "c",
                       "output": {"temp": 20}}],
        }));
        assert_eq!(text_of(&chat.messages[0]), r#"{"temp":20}"#);
    }

    /// Responses tools are flat; chat tools are nested. Both convert,
    /// and a built-in tool this server cannot run is skipped rather than
    /// offered.
    #[test]
    fn flat_and_nested_function_tools_convert_and_built_ins_are_skipped() {
        let chat = converted(json!({
            "model": "m",
            "input": "hi",
            "tools": [
                {"type": "function", "name": "flat", "parameters": {"type": "object"}},
                {"type": "function", "function": {"name": "nested"}},
                {"type": "web_search"},
            ],
        }));
        let names: Vec<&str> = chat
            .tools
            .iter()
            .map(|t| t.function.name.as_str())
            .collect();
        assert_eq!(names, vec!["flat", "nested"]);
        assert!(chat.tools_active());
    }

    #[test]
    fn tool_choice_none_disables_the_offered_tools() {
        let chat = converted(json!({
            "model": "m",
            "input": "hi",
            "tool_choice": "none",
            "tools": [{"type": "function", "name": "t"}],
        }));
        assert!(!chat.tools_active());
        assert!(offered_tools(&chat).is_empty());
    }

    /// `reasoning.effort` reaches the same knob every other surface
    /// drives, so a checkpoint graded for effort is graded identically
    /// here.
    #[test]
    fn the_reasoning_effort_lands_on_the_shared_template_knob() {
        let chat = converted(json!({
            "model": "m",
            "input": "hi",
            "reasoning": {"effort": "high"},
        }));
        assert_eq!(chat.reasoning_effort.as_deref(), Some("high"));
    }

    // -----------------------------------------------------------------
    // Rule 3: Responses' own output budget
    // -----------------------------------------------------------------

    /// `max_tokens` on the converted request is filled in from this
    /// surface's own budget. Left to serde -- the obvious way, since the
    /// field has a `#[serde(default)]` already -- it takes whatever some
    /// other surface decided, and the legacy 16-token floor is still
    /// live in this crate on `/v1/completions`; under it every codex
    /// turn stops after a sentence and a half. This test fails on
    /// exactly that: it pins the converted budget to the Responses
    /// default and refuses the legacy floor by value.
    #[test]
    fn an_absent_max_output_tokens_uses_the_responses_surfaces_own_budget() {
        let chat = converted(json!({"model": "m", "input": "hi"}));
        assert_eq!(chat.max_tokens, DEFAULT_MAX_OUTPUT_TOKENS);
        assert_ne!(
            chat.max_tokens, 16,
            "the /v1/completions floor must never reach this surface"
        );
    }

    #[test]
    fn an_explicit_max_output_tokens_is_honoured() {
        let chat = converted(json!({"model": "m", "input": "hi", "max_output_tokens": 7}));
        assert_eq!(chat.max_tokens, 7);
    }

    #[test]
    fn a_non_positive_max_output_tokens_is_refused() {
        for budget in [0, -1] {
            let converted = to_chat_request(&request(
                json!({"model": "m", "input": "hi", "max_output_tokens": budget}),
            ));
            // Matched rather than `expect_err`: the Ok side is the chat
            // request struct, which has no `Debug` to unwrap through.
            match converted {
                Err((status, _)) => assert_eq!(status, StatusCode::BAD_REQUEST),
                Ok(_) => panic!("a budget of {budget} is a request error"),
            }
        }
    }

    // -----------------------------------------------------------------
    // Rule 4: truncation
    // -----------------------------------------------------------------

    /// A buffered answer cut off by the token budget must say so in the
    /// protocol's own vocabulary. Reporting `completed` (the obvious
    /// shape, since the request did not error) tells codex it holds the
    /// model's whole answer and it stops rather than continuing.
    #[test]
    fn a_truncated_buffered_answer_is_incomplete_with_the_max_output_tokens_reason() {
        let parsed = output::ParsedOutput {
            reasoning: Some("thinking".to_string()),
            content: "half an ans".to_string(),
            calls: Vec::new(),
        };
        let body = build_response(parsed, "length", &usage(3, 4), "resp_1", 1, "m");
        assert_eq!(body["status"], "incomplete");
        assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(body["output"][0]["type"], "reasoning");
        assert_eq!(
            body["output"][0]["status"], "completed",
            "a reasoning block that finished did finish"
        );
        assert_eq!(body["output"][1]["status"], "incomplete");
    }

    #[test]
    fn an_untruncated_buffered_answer_is_completed_with_no_incomplete_details() {
        let parsed = output::ParsedOutput {
            reasoning: None,
            content: "done".to_string(),
            calls: vec![output::ParsedToolCall {
                name: "t".to_string(),
                arguments: "{}".to_string(),
            }],
        };
        let body = build_response(parsed, "stop", &usage(3, 4), "resp_1", 1, "m");
        assert_eq!(body["status"], "completed");
        assert!(body["incomplete_details"].is_null());
        assert_eq!(body["output"][1]["type"], "function_call");
        assert_eq!(body["usage"]["input_tokens"], 3);
        assert_eq!(body["usage"]["total_tokens"], 7);
    }

    /// The streamed half of rule 4, and the one that is easy to get
    /// wrong: marking every message item by the final finish reason --
    /// the obvious implementation, since the reason is only known at the
    /// end -- marks a message that was closed *before* the truncation as
    /// `incomplete` too. codex reads that as a truncated earlier step.
    /// Here the first message is closed by a tool call and only the
    /// second one runs out of budget.
    #[test]
    fn only_the_final_message_item_is_marked_incomplete_by_a_truncation() {
        let frames = run(vec![
            GenEvent::Content("before".to_string()),
            GenEvent::WholeCall {
                index: 0,
                name: "t".to_string(),
                arguments: "{}".to_string(),
            },
            GenEvent::Content("after".to_string()),
            GenEvent::Done {
                finish: "length",
                usage: usage(1, 2),
            },
        ]);
        let final_response = &last_named(&frames, "response.incomplete")["response"];
        let output = final_response["output"].as_array().expect("output items");
        assert_eq!(output.len(), 3, "{output:#?}");
        assert_eq!(
            output[0]["status"], "completed",
            "closed before the cut-off"
        );
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(output[1]["status"], "completed");
        assert_eq!(output[2]["status"], "incomplete");
        assert_eq!(
            final_response["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        assert_eq!(final_response["status"], "incomplete");
    }

    // -----------------------------------------------------------------
    // The failure event
    // -----------------------------------------------------------------

    /// A generation that dies mid-stream must end in `response.failed`
    /// carrying a code. Ending the stream silently -- the obvious thing,
    /// since the error has already been logged and the socket is done --
    /// leaves codex waiting out its 300 s idle timeout and then blaming
    /// the network. Worse, without `error.code` it cannot tell a blown
    /// context window (which it fixes by compacting and retrying) from a
    /// server fault (which it must not retry).
    #[test]
    fn a_failed_generation_streams_a_failed_event_carrying_its_error_code() {
        let error = DecodeError::KvBudgetExceeded {
            binding: "context_length_exceeded",
            estimated_bytes: 9,
            limit_bytes: 4,
            positions: 90,
            positions_limit: 40,
            detail: "too long".to_string(),
        };
        assert_eq!(failure_code(&error), "context_length_exceeded");
        let frames = run(vec![
            GenEvent::Content("partial".to_string()),
            GenEvent::Failed {
                code: failure_code(&error),
                message: error.to_string(),
            },
        ]);
        assert_eq!(
            names(&frames).last(),
            Some(&"response.failed"),
            "the stream must end on a terminal event"
        );
        let response = &last_named(&frames, "response.failed")["response"];
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "context_length_exceeded");
        assert!(response["error"]["message"]
            .as_str()
            .expect("a message")
            .contains("too long"));
        assert_eq!(
            response["output"][0]["type"], "message",
            "what was already streamed is still in the failed snapshot"
        );
    }

    #[test]
    fn every_decode_error_has_a_stable_failure_code() {
        assert_eq!(
            failure_code(&DecodeError::TokenOutOfVocab {
                token: 1,
                vocab_size: 2
            }),
            "invalid_prompt"
        );
        assert_eq!(
            failure_code(&DecodeError::KvPoolExhausted),
            "server_overloaded"
        );
        assert_eq!(
            failure_code(&DecodeError::QueueFull { queued: 1, cap: 1 }),
            "server_overloaded"
        );
    }

    // -----------------------------------------------------------------
    // Rule 5: keepalives
    // -----------------------------------------------------------------

    /// The keepalive has to be a frame with data in it. An SSE comment
    /// -- what `Sse::keep_alive` sends, and what the chat stream uses --
    /// does not reset codex's stream-idle countdown, so a long prefill
    /// or a slow decode makes it reconnect in the middle of an answer.
    /// This test fails for a comment keepalive, because a comment
    /// carries no `type` and no `sequence_number`.
    #[test]
    fn a_keepalive_is_a_data_bearing_in_progress_frame() {
        let frames = run(vec![
            GenEvent::Content("a".to_string()),
            GenEvent::Keepalive,
        ]);
        let keepalive = frames.last().expect("a frame");
        assert_eq!(keepalive.name, "response.in_progress");
        assert_eq!(keepalive.data["type"], "response.in_progress");
        assert!(keepalive.data["sequence_number"].is_number());
        assert_eq!(keepalive.data["response"]["status"], "in_progress");
    }

    /// ... and the silence it covers includes the silence *before* the
    /// first event, which is where queueing and prefill on a long prompt
    /// live. A keepalive armed only after the first token would leave
    /// exactly that gap uncovered -- so this test sends nothing at all
    /// and still expects a keepalive.
    #[tokio::test]
    async fn the_keepalive_covers_the_silence_before_the_first_event() {
        let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(4);
        let mut events = Box::pin(with_keepalive(rx, Duration::from_millis(20)));
        assert_eq!(
            events.next().await,
            Some(GenEvent::Keepalive),
            "nothing has been generated yet and the client must still be fed"
        );
        tx.send(GenEvent::Content("first".to_string()))
            .await
            .expect("the receiver is live");
        assert_eq!(
            events.next().await,
            Some(GenEvent::Content("first".to_string()))
        );
        drop(tx);
        assert_eq!(
            events.next().await,
            None,
            "a closed generator ends the stream"
        );
    }

    // -----------------------------------------------------------------
    // Sequencing
    // -----------------------------------------------------------------

    #[test]
    fn a_stream_opens_with_created_and_in_progress() {
        let frames = run(Vec::new());
        assert_eq!(
            names(&frames),
            vec!["response.created", "response.in_progress"]
        );
        assert_eq!(frames[0].data["sequence_number"], 0);
        assert_eq!(frames[1].data["sequence_number"], 1);
        assert_eq!(frames[0].data["response"]["id"], "resp_test");
    }

    #[test]
    fn sequence_numbers_are_consecutive_from_zero_across_every_frame() {
        let frames = run(vec![
            GenEvent::Reasoning("t".to_string()),
            GenEvent::Content("a".to_string()),
            GenEvent::Keepalive,
            GenEvent::Done {
                finish: "stop",
                usage: usage(1, 1),
            },
        ]);
        let seen: Vec<u64> = frames
            .iter()
            .map(|f| f.data["sequence_number"].as_u64().expect("numbered"))
            .collect();
        assert_eq!(seen, (0..seen.len() as u64).collect::<Vec<_>>());
    }

    /// Reasoning and the answer are separate output items, in that
    /// order, each opened and closed exactly once -- the shape codex
    /// renders a thinking block from.
    #[test]
    fn reasoning_then_text_becomes_two_output_items_in_order() {
        let frames = run(vec![
            GenEvent::Reasoning("thinking".to_string()),
            GenEvent::Content("answer".to_string()),
            GenEvent::Done {
                finish: "stop",
                usage: usage(2, 3),
            },
        ]);
        assert_eq!(
            names(&frames),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let completed = &last_named(&frames, "response.completed")["response"];
        assert_eq!(completed["output"][0]["type"], "reasoning");
        assert_eq!(completed["output"][0]["content"][0]["text"], "thinking");
        assert_eq!(completed["output"][1]["type"], "message");
        assert_eq!(completed["output"][1]["content"][0]["text"], "answer");
        assert_eq!(completed["usage"]["output_tokens"], 3);
    }

    /// A streamed call: opened on `CallStart`, its arguments delivered
    /// as deltas, and closed once -- with the final arguments equal to
    /// the concatenation of the deltas, which is what a client that
    /// accumulates them ends up holding.
    #[test]
    fn a_streamed_tool_call_opens_delivers_and_closes_one_function_call_item() {
        let frames = run(vec![
            GenEvent::CallStart {
                index: 0,
                name: "get_weather".to_string(),
            },
            GenEvent::CallArguments {
                index: 0,
                fragment: "{\"city\":".to_string(),
            },
            GenEvent::CallArguments {
                index: 0,
                fragment: "\"Rome\"}".to_string(),
            },
            GenEvent::CallEnd {
                index: 0,
                arguments: "{\"city\":\"Rome\"}".to_string(),
            },
            GenEvent::Done {
                finish: "stop",
                usage: usage(1, 1),
            },
        ]);
        assert_eq!(
            names(&frames),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let item = &last_named(&frames, "response.output_item.done")["item"];
        assert_eq!(item["name"], "get_weather");
        assert_eq!(item["arguments"], "{\"city\":\"Rome\"}");
        assert_eq!(item["status"], "completed");
        assert!(item["call_id"]
            .as_str()
            .expect("a call id")
            .starts_with("call_"));
    }

    /// The close is authoritative: whatever of the final arguments never
    /// went out as a delta is topped up before the item closes, so a
    /// client that concatenates deltas and a client that reads the done
    /// item agree.
    #[test]
    fn a_call_close_tops_up_arguments_that_never_streamed() {
        let frames = run(vec![
            GenEvent::CallStart {
                index: 0,
                name: "t".to_string(),
            },
            GenEvent::CallEnd {
                index: 0,
                arguments: "{\"a\":1}".to_string(),
            },
        ]);
        let deltas: Vec<&str> = frames
            .iter()
            .filter(|f| f.name == "response.function_call_arguments.delta")
            .map(|f| f.data["delta"].as_str().expect("a delta"))
            .collect();
        assert_eq!(deltas, vec!["{\"a\":1}"]);
        let item = &last_named(&frames, "response.output_item.done")["item"];
        assert_eq!(item["arguments"], "{\"a\":1}");
    }

    /// Text after a call belongs to a new message item, not to the one
    /// the call interrupted.
    #[test]
    fn text_after_a_call_opens_a_second_message_item() {
        let frames = run(vec![
            GenEvent::Content("before".to_string()),
            GenEvent::WholeCall {
                index: 0,
                name: "t".to_string(),
                arguments: "{}".to_string(),
            },
            GenEvent::Content("after".to_string()),
            GenEvent::Done {
                finish: "stop",
                usage: usage(1, 1),
            },
        ]);
        let completed = &last_named(&frames, "response.completed")["response"];
        let output = completed["output"].as_array().expect("output items");
        assert_eq!(output.len(), 3);
        assert_eq!(output[0]["content"][0]["text"], "before");
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(output[2]["content"][0]["text"], "after");
    }

    #[test]
    fn an_empty_delta_produces_no_frame() {
        let frames = run(vec![
            GenEvent::Content(String::new()),
            GenEvent::Reasoning(String::new()),
        ]);
        assert_eq!(frames.len(), 2, "only the two opening frames");
    }

    /// A fragment with no call open is dropped rather than opening a
    /// nameless item a client would try to execute.
    #[test]
    fn a_stray_arguments_fragment_is_dropped() {
        let frames = run(vec![GenEvent::CallArguments {
            index: 0,
            fragment: "{}".to_string(),
        }]);
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn a_frame_serializes_as_a_named_sse_event() {
        let frames = run(vec![GenEvent::Keepalive]);
        let event = frames
            .into_iter()
            .last()
            .expect("a keepalive frame")
            .into_event();
        let wire = format!("{event:?}");
        assert!(!wire.is_empty());
    }

    #[test]
    fn ids_carry_their_kind_and_do_not_repeat() {
        let first = new_id("resp_");
        let second = new_id("resp_");
        assert!(first.starts_with("resp_"));
        assert_ne!(first, second);
    }
}
