//! The Anthropic **Messages** API (`POST /v1/messages`) and its token
//! counter (`POST /v1/messages/count_tokens`) -- the surface Claude Code
//! talks to.
//!
//! Ported from FreeToken's `server/anthropic_api.py` (Apache-2.0; see
//! `docs/THIRD_PARTY_NOTICES.md`).
//!
//! # This is a shaping, not a second engine
//!
//! Everything below reshapes a Messages request into the exact
//! [`crate::ChatCompletionRequest`] the chat path already renders and
//! decodes from, and reshapes what came back into Anthropic's content
//! blocks and stream events. It deliberately grows no prompt renderer,
//! no effort sanitizer and no stop set of its own -- those live on that
//! struct ([`crate::ChatCompletionRequest::resolve_template_kwargs`],
//! [`crate::ChatCompletionRequest::generation_params_for_template`]) --
//! for the same reason `responses` does not: a checkpoint that thinks
//! through `/v1/chat/completions` and does not through `/v1/messages`,
//! on the same conversation, is a difference no user can see the cause
//! of.
//!
//! # The rules that are load-bearing
//!
//! Each has a test below that fails if the rule is done the obvious way
//! instead.
//!
//! 1. **One leading system message.** The top-level `system` *and*
//!    every `system`-role message in the array merge into a single
//!    leading system turn. Claude Code sends both at once, and a strict
//!    template (Qwen3.5) answers a second system message with "System
//!    message must be at the beginning" -- the request then fails
//!    outright rather than degrading.
//! 2. **A `tool_result` is keyed by `tool_use_id`.** Not `id`: `id`
//!    names the *result* block, `tool_use_id` names the call it answers.
//!    Key on `id` and parallel tool results reach the template
//!    unattributable, so a model that ran three tools cannot tell which
//!    answer belongs to which call.
//! 3. **Unknown blocks are skipped, never refused.**
//!    `redacted_thinking`, `image`, and whatever Anthropic ships next
//!    are dropped from the conversation; the request still runs. A 4xx
//!    on an unknown block type means one new client version takes the
//!    whole endpoint down.
//! 4. **`tool_choice` splits into two lists.** `"none"` hides the tools
//!    from the template *and* from the output parser -- a caller who
//!    said "do not call tools" must not have marker-looking prose
//!    reinterpreted as a call. A named tool narrows what the *template*
//!    offers, while the parser keeps every tool, because the model can
//!    still emit a call the request did not force and its arguments
//!    must still be typed against a schema ([`split_tool_lists`]).
//! 5. **No `data: [DONE]`.** An Anthropic stream terminates on
//!    `message_stop`. The OpenAI sentinel is an unknown event to a
//!    strict client, which reports a protocol error on an answer that
//!    completed normally.
//! 6. **Usage excludes the cached prefix.** `input_tokens` is the
//!    prompt *minus* what the prefix cache served, and
//!    `cache_read_input_tokens` carries that remainder -- absent
//!    entirely when it is zero. Counting the cached prefix in
//!    `input_tokens` double-bills every cached turn.
//! 7. **`count_tokens` tokenizes what a generation would.** Same
//!    converter, same template tools, same `chat_template_kwargs`, so
//!    the number it answers is the `usage.input_tokens` the following
//!    generation reports rather than a second, differently-built
//!    estimate.
//!
//! # Ordering is the content
//!
//! The stream's shape *is* its meaning: `message_start`, then content
//! blocks opened and closed with a running index that advances on every
//! `content_block_stop`, then one `message_delta` carrying the stop
//! reason and usage, then `message_stop`. A thinking block closes with
//! an empty `signature_delta` first; a tool block streams
//! `input_json_delta` fragments and tops up the remainder of the
//! authoritative arguments before it closes. [`MessagesStream`] owns
//! all of that and is driven from tests with no model and no socket.

use crate::stream_events::{map_tool_event, with_keepalive};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::generate::Usage;
use crate::output::{OutputPosture, ParsedOutput};
use crate::{
    attribution, decode_error_response, output, run_generation_emit, sse, stats, ApiError,
    AppState, ChatCompletionRequest, ChatMessage, MessageContent, StopParam, ThinkingSwitch,
    ToolCallFunctionIn, ToolCallIn, ToolDef, ToolFunctionDef,
};
use frink_models::tokenizer::SpecialTokens;

/// Stream silence after which a protocol-native `ping` goes out.
///
/// A `ping` is a real Anthropic event, not an SSE comment: it is what
/// bridges the queue/prefill gap on a long prompt for a client with a
/// stream-idle timeout, and it is why this module never installs
/// `Sse::keep_alive` (whose keepalive is a comment).
pub(crate) const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);

// ---------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A `msg_` / `call_` id.
///
/// Same construction as [`frink_api::next_request_id`] -- a wall-clock
/// stamp plus a process-monotonic counter -- rather than the reference's
/// UUID: these only have to be unique, and the workspace carries no RNG
/// dependency for the server to reach for.
fn new_id(prefix: &str) -> String {
    let stamp = unix_nanos();
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}{:012x}{:06x}", stamp & 0xffff_ffff_ffff, n)
}

/// `toolu_<name>_<block index>_<unique>`, the shape the reference
/// builds so a human reading a transcript can see which tool a block
/// belongs to. The name is slugified because `_` is the field
/// separator.
fn tool_use_id(name: &str, index: usize) -> String {
    let slug: String = name.replace('_', "-").chars().take(24).collect();
    let slug = if slug.is_empty() {
        "tool".to_string()
    } else {
        slug
    };
    let stamp = unix_nanos();
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("toolu_{slug}_{index}_{:08x}", (stamp ^ n) as u32)
}

fn unix_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------

/// A message's content: one bare string, or a list of typed blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ContentIn {
    Text(String),
    Blocks(Vec<ContentBlockIn>),
}

/// One content block.
///
/// `type` stays a `String` rather than an enum, and every payload field
/// is optional, because rule 3 depends on it: Claude Code sends block
/// types beyond the five this converter knows, and a typed enum would
/// fail deserialization of the *whole request* over one block this
/// server would have skipped anyway.
#[derive(Debug, Clone, Deserialize)]
struct ContentBlockIn {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    /// `thinking` blocks.
    #[serde(default)]
    thinking: Option<String>,
    /// This block's own id (a `tool_use` block's call id).
    #[serde(default)]
    id: Option<String>,
    /// Rule 2: on a `tool_result`, the id of the `tool_use` it answers.
    #[serde(default)]
    tool_use_id: Option<String>,
    /// `tool_use` blocks.
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<Value>,
    /// `tool_result` payload: a string, or a list of blocks.
    #[serde(default)]
    content: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicMessage {
    /// Anthropic's own spec is user/assistant only, but Claude Code
    /// sends `system` inside the array as well as at the top level.
    /// Accepted here so rule 1 can merge it, rather than refused.
    role: String,
    content: ContentIn,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum SystemIn {
    Text(String),
    Blocks(Vec<ContentBlockIn>),
}

#[derive(Debug, Clone, Deserialize)]
struct ToolIn {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_schema: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolChoiceIn {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    name: Option<String>,
}

/// The prompt side of a Messages request: exactly the fields
/// `/v1/messages` and `/v1/messages/count_tokens` have in common.
///
/// Shared as one flattened struct rather than duplicated across the two
/// request types on purpose -- that is rule 7 made structural. A count
/// that is built from its own copy of these fields drifts from the
/// generation's the first time one of them grows a case.
#[derive(Debug, Clone, Deserialize)]
struct PromptFields {
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    system: Option<SystemIn>,
    #[serde(default)]
    tools: Option<Vec<ToolIn>>,
    #[serde(default)]
    tool_choice: Option<ToolChoiceIn>,
    /// Anthropic's extended-thinking toggle, `{"type": "enabled" |
    /// "disabled", "budget_tokens": N}`. The switch half is the same
    /// wire shape the chat surface already reads, so it is handed
    /// straight to [`crate::ChatCompletionRequest::resolve_template_kwargs`]
    /// and goes through the one thinking-resolution path; the budget
    /// half becomes llama.cpp's `thinking_budget_tokens` exactly as
    /// llama.cpp's own Anthropic lowering does it
    /// (`tools/server/server-chat.cpp:585-591`).
    #[serde(default)]
    thinking: Option<AnthropicThinking>,
}

/// Anthropic's `thinking` object. Its own struct rather than the shared
/// [`ThinkingSwitch`] because `budget_tokens` is this wire's field and
/// not the DeepSeek wire's: declared on the shared switch it would be
/// read on `/v1/chat/completions` too, where llama.cpp does not read
/// it.
#[derive(Debug, Clone, Deserialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    budget_tokens: Option<i64>,
}

/// llama.cpp's default when `thinking.type` is `enabled` and the
/// request names no `budget_tokens` (`server-chat.cpp:590`). Anthropic's
/// API makes the field mandatory, so a real client always sends one;
/// the number exists for the client that does not.
const DEFAULT_ANTHROPIC_THINKING_BUDGET: i64 = 10_000;

impl AnthropicThinking {
    fn switch(&self) -> ThinkingSwitch {
        ThinkingSwitch {
            kind: self.kind.clone(),
        }
    }

    /// The budget this object asks for, llama.cpp's way: only an
    /// `enabled` switch carries one, and one without a number gets
    /// [`DEFAULT_ANTHROPIC_THINKING_BUDGET`]. A `disabled` switch has
    /// no thought to bound, and is resolved by the shared switch.
    fn budget(&self) -> Result<Option<crate::reasoning_budget::BudgetTokens>, ApiError> {
        if self.kind != "enabled" {
            return Ok(None);
        }
        let value = self
            .budget_tokens
            .unwrap_or(DEFAULT_ANTHROPIC_THINKING_BUDGET);
        crate::reasoning_budget::BudgetTokens::parse(value)
            .map(Some)
            .map_err(|why| {
                crate::invalid_request(
                    &why.replace("reasoning_budget_tokens", "thinking.budget_tokens"),
                    "thinking.budget_tokens",
                )
            })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MessagesRequest {
    #[serde(default)]
    model: String,
    max_tokens: usize,
    #[serde(flatten)]
    prompt: PromptFields,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    metadata: Option<Value>,
}

/// `POST /v1/messages/count_tokens`: the input side of a Messages
/// request, with no output budget and no sampling knobs.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CountTokensRequest {
    #[serde(default)]
    model: String,
    #[serde(flatten)]
    prompt: PromptFields,
}

// ---------------------------------------------------------------------
// Request conversion: Anthropic Messages -> ChatCompletionRequest
// ---------------------------------------------------------------------

/// One converted assistant call, before it becomes a [`ToolCallIn`]
/// (which is deserialize-only and has no `PartialEq`, so the conversion
/// rules could not be asserted on it).
#[derive(Debug, Clone, PartialEq, Eq)]
struct ConvertedCall {
    id: String,
    name: String,
    arguments: String,
}

/// One conversation turn under construction.
///
/// `reasoning` is a `thinking` block's text, carried separately from
/// `content` all the way to [`ChatMessage::reasoning_content`] so a
/// template that knows the family's thinking markers can wrap it and
/// one that does not can drop it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Turn {
    role: String,
    content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Vec<ConvertedCall>,
    tool_call_id: Option<String>,
}

impl Turn {
    /// The turn as the renderer wants it, or `None` for a turn that
    /// held nothing a template can show the model.
    ///
    /// A `thinking` block becomes `reasoning_content`, as the
    /// reference has it -- never `content`. Folding it into `content`
    /// would show the model its own past deliberation as something it
    /// *said*, which is the whole reason the family markers exist.
    ///
    /// A turn holding ONLY thinking still lowers to nothing. It has no
    /// content and no calls, so there is no message for the reasoning
    /// to hang off, and emitting a contentless turn would show the
    /// model a blank turn it never took.
    fn lower(self) -> Option<ChatMessage> {
        let has_text = self.content.as_ref().is_some_and(|c| !c.is_empty());
        if !has_text && self.tool_calls.is_empty() && self.tool_call_id.is_none() {
            // Nothing usable: an image-only block list, or a turn that
            // carried only thinking. Emitting it anyway would show the
            // model a blank turn it never took.
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
            // documents and the one templates are written against.
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

/// The plain text of a message content: a bare string verbatim, or the
/// text blocks of a list concatenated.
fn content_text(content: &ContentIn) -> String {
    match content {
        ContentIn::Text(text) => text.clone(),
        ContentIn::Blocks(blocks) => blocks
            .iter()
            .filter(|b| b.kind == "text")
            .filter_map(|b| b.text.as_deref())
            .collect(),
    }
}

fn system_text(system: &SystemIn) -> String {
    match system {
        SystemIn::Text(text) => text.clone(),
        SystemIn::Blocks(blocks) => blocks
            .iter()
            .filter(|b| b.kind == "text")
            .filter_map(|b| b.text.as_deref())
            .collect(),
    }
}

/// A `tool_result`'s payload as the text a template can render.
///
/// A string is used verbatim; a list contributes each item's `text`
/// (the shape Claude Code sends); anything else is compact JSON rather
/// than being dropped, because a tool that answered with a number
/// answered something.
fn tool_result_text(content: Option<&Value>) -> String {
    match content {
        None => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::String(text) => text.clone(),
                Value::Object(_) => item
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                other => other.to_string(),
            })
            .collect(),
        Some(other) => other.to_string(),
    }
}

/// A `tool_use` block's `input` as the JSON-encoded string
/// `tool_calls[].function.arguments` is defined to carry.
///
/// Encoded, not passed through as an object: that is the real OpenAI
/// convention, and `chat_template::tool_call_json` decodes it back for
/// the templates that want a dict. A non-object input becomes `{}`
/// rather than a bare scalar, because a template iterating arguments
/// would otherwise be handed something it cannot iterate.
fn arguments_json(input: Option<&Value>) -> String {
    match input {
        Some(value @ Value::Object(_)) => value.to_string(),
        _ => "{}".to_string(),
    }
}

/// Rule 1 + rules 2 and 3: the whole content conversion.
///
/// Returns the messages in template order: at most one system turn,
/// first, then everything else in the order it arrived.
fn convert_prompt(prompt: &PromptFields) -> Vec<ChatMessage> {
    let mut system_texts: Vec<String> = Vec::new();
    if let Some(system) = &prompt.system {
        system_texts.push(system_text(system));
    }

    let mut rest: Vec<ChatMessage> = Vec::new();
    for message in &prompt.messages {
        if message.role == "system" {
            // Rule 1: hoisted, not left where it sat. A system turn in
            // the middle of the array is what a strict template refuses.
            system_texts.push(content_text(&message.content));
            continue;
        }

        let blocks = match &message.content {
            ContentIn::Text(text) => {
                rest.extend(
                    Turn {
                        role: message.role.clone(),
                        content: Some(text.clone()),
                        ..Turn::default()
                    }
                    .lower(),
                );
                continue;
            }
            ContentIn::Blocks(blocks) => blocks,
        };

        let mut turn = Turn {
            role: message.role.clone(),
            ..Turn::default()
        };
        let mut texts: Vec<String> = Vec::new();
        let mut thoughts: Vec<String> = Vec::new();
        for block in blocks {
            match block.kind.as_str() {
                "text" => {
                    if let Some(text) = block.text.as_ref().filter(|t| !t.is_empty()) {
                        texts.push(text.clone());
                    }
                }
                "thinking" => {
                    if let Some(text) = block.thinking.as_ref().filter(|t| !t.is_empty()) {
                        thoughts.push(text.clone());
                    }
                }
                "tool_use" => turn.tool_calls.push(ConvertedCall {
                    // A call with no id of its own still needs one: the
                    // tool result that answers it is matched by id.
                    id: block.id.clone().unwrap_or_else(|| new_id("call_")),
                    name: block.name.clone().unwrap_or_default(),
                    arguments: arguments_json(block.input.as_ref()),
                }),
                "tool_result" if message.role == "user" => {
                    // A tool result is its own `role: "tool"` message,
                    // emitted where it sits so a user turn that carries
                    // both a result and a question keeps that order.
                    //
                    // Rule 2: `tool_use_id` names the call, `id` names
                    // this block. `id` is consulted only as a fallback
                    // for a client that sent nothing else.
                    rest.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(MessageContent::Text(tool_result_text(
                            block.content.as_ref(),
                        ))),
                        tool_calls: None,
                        tool_call_id: Some(
                            block
                                .tool_use_id
                                .clone()
                                .or_else(|| block.id.clone())
                                .unwrap_or_default(),
                        ),
                        reasoning_content: None,
                    });
                }
                "tool_result" => {
                    // On an assistant turn a tool result has no `role:
                    // "tool"` slot to go to, so it is labelled prose.
                    texts.push(format!(
                        "Tool result: {}",
                        tool_result_text(block.content.as_ref())
                    ));
                }
                // Rule 3. `image` (this is a text-only server),
                // `redacted_thinking` (an opaque payload with no
                // plaintext to render), and anything Anthropic adds
                // next: skipped, never a rejection of the request.
                _ => {}
            }
        }
        if !texts.is_empty() {
            // Concatenated rather than kept as parts: `MessageContent`'s
            // own `as_text` joins parts with "" before any template sees
            // them, so this is the same string one step earlier.
            turn.content = Some(texts.concat());
        }
        if !thoughts.is_empty() {
            turn.reasoning = Some(thoughts.join("\n\n"));
        }
        rest.extend(turn.lower());
    }

    let mut messages: Vec<ChatMessage> = Vec::with_capacity(rest.len() + 1);
    let system = system_texts
        .iter()
        .filter(|t| !t.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n\n");
    if !system.is_empty() {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text(system)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    messages.extend(rest);
    messages
}

/// Anthropic tool definitions in the OpenAI request shape.
///
/// A schema with no `type` is given `"object"`, exactly as the
/// reference's validator does: a template that renders
/// `parameters.type` would otherwise describe a typeless tool to the
/// model.
fn tool_defs(tools: &[ToolIn]) -> Vec<ToolDef> {
    tools
        .iter()
        .map(|tool| {
            let mut schema = tool
                .input_schema
                .clone()
                .unwrap_or_else(|| json!({"type": "object"}));
            if let Some(object) = schema.as_object_mut() {
                object
                    .entry("type".to_string())
                    .or_insert_with(|| json!("object"));
            }
            ToolDef {
                kind: "function".to_string(),
                function: ToolFunctionDef {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: Some(schema),
                },
            }
        })
        .collect()
}

/// Rule 4's second half: `(template_tools, parser_tools)`.
///
/// The parser sees every tool; the template sees only the selected one
/// when `tool_choice` names a function. Narrowing the parser too would
/// make a call the model chose anyway -- which it can, since this
/// server has no constrained decoding to force the named one -- arrive
/// as unparsed prose.
fn split_tool_lists(all: Vec<ToolDef>, selected: Option<&str>) -> (Vec<ToolDef>, Vec<ToolDef>) {
    if all.is_empty() {
        return (Vec::new(), Vec::new());
    }
    match selected {
        Some(name) => (
            all.iter()
                .filter(|tool| tool.function.name == name)
                .cloned()
                .collect(),
            all,
        ),
        None => (all.clone(), all),
    }
}

/// A converted request: the chat request that renders and decodes it,
/// plus the tools its *output* parser gets (rule 4).
struct Prepared {
    chat: ChatCompletionRequest,
    parser_tools: Vec<ToolDef>,
}

/// The prompt half of the conversion, shared by both endpoints so a
/// counted prompt is the prompt a generation of the same body renders
/// (rule 7).
fn prepare_prompt(
    prompt: &PromptFields,
    model: String,
    max_tokens: usize,
) -> Result<Prepared, ApiError> {
    let reasoning_budget_tokens = prompt
        .thinking
        .as_ref()
        .map(AnthropicThinking::budget)
        .transpose()
        .map_err(anthropic_shape)?
        .flatten();
    let all = tool_defs(prompt.tools.as_deref().unwrap_or_default());
    let choice = prompt.tool_choice.as_ref();
    let (template_tools, parser_tools) = if choice.is_some_and(|c| c.kind == "none") {
        // Rule 4: hidden from the template AND from the parser.
        (Vec::new(), Vec::new())
    } else {
        let selected = choice
            .filter(|c| c.kind == "tool")
            .and_then(|c| c.name.as_deref());
        split_tool_lists(all, selected)
    };

    let chat = ChatCompletionRequest {
        samplers: None,
        model,
        messages: convert_prompt(prompt),
        max_tokens,
        temperature: None,
        top_p: None,
        // Neither wire has a min_p; 0.0 (off) is what None resolves to.
        min_p: None,
        top_k: None,
        repetition_penalty: None,
        // Neither the Anthropic Messages wire nor the Responses wire
        // carries llama.cpp's extra samplers, so they resolve to their
        // neutral defaults and this chain is llama.cpp's default one
        // doing nothing extra.
        extra_samplers: Default::default(),
        // No seed on this wire, so the chat path's policy applies
        // unchanged: an unseeded sampled request draws fresh every time
        // rather than replaying one draw forever.
        seed: None,
        stop: None,
        stream: None,
        // Replay is a frink extension with no Anthropic spelling.
        stream_resumable: None,
        // The template is offered only what `tool_choice` left it; the
        // parser's list rides beside this struct, not in it.
        tools: template_tools,
        // Already resolved into the two lists above. Left `None` so
        // `validate_supported_fields` does not refuse a named choice
        // the chat surface cannot honour but this one can (it narrows
        // the offer rather than forcing the call).
        tool_choice: None,
        chat_template_kwargs: None,
        reasoning_effort: None,
        thinking: prompt.thinking.as_ref().map(AnthropicThinking::switch),
        // Stateless surface: Claude Code resends the whole conversation.
        session_id: None,
        // Anthropic's API prefills a trailing assistant turn, and so does
        // the shared renderer by default now (`continuation`): this
        // lowering says nothing and gets the one rule every route gets.
        continue_final_message: Default::default(),
        reasoning_budget_tokens,
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
        // Not on the Anthropic wire.
        logit_bias: None,
        lora: None,
        // A serving-benchmark knob on the OpenAI surface only; this
        // protocol has no spelling for it.
        ignore_eos: None,
    };
    Ok(Prepared { chat, parser_tools })
}

/// The whole request conversion for `/v1/messages`.
fn to_chat_request(req: &MessagesRequest) -> Result<Prepared, ApiError> {
    let mut prepared = prepare_prompt(&req.prompt, req.model.clone(), req.max_tokens)?;
    prepared.chat.temperature = req.temperature;
    prepared.chat.top_p = req.top_p;
    prepared.chat.top_k = req.top_k;
    prepared.chat.stop = req.stop_sequences.clone().map(StopParam::Many);
    prepared.chat.stream = req.stream;
    // The shared validator, so `max_tokens: 0` and a misspelled
    // `thinking.type` are refused here exactly as they are on
    // `/v1/chat/completions` -- reshaped into the Anthropic envelope,
    // because a Claude client parses `{"type": "error", ...}` and shows
    // an OpenAI-shaped body as an unreadable protocol failure.
    prepared
        .chat
        .validate_supported_fields()
        .map_err(anthropic_shape)?;
    Ok(prepared)
}

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

/// The Anthropic error envelope: `{"type": "error", "error": {"type",
/// "message"}}`. Every failure this module answers with wears it,
/// including the ones raised by shared helpers -- see [`anthropic_shape`].
fn anthropic_error(status: StatusCode, message: &str) -> ApiError {
    (
        status,
        Json(json!({
            "type": "error",
            "error": {
                "type": error_type(status),
                "message": message,
            }
        })),
    )
}

/// Anthropic's error `type` for an HTTP status. The vocabulary is
/// closed, so an unmapped status is `api_error` rather than an invented
/// name a client would fail to match.
fn error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::REQUEST_TIMEOUT => "timeout_error",
        StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::NOT_IMPLEMENTED => "invalid_request_error",
        StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
        _ => "api_error",
    }
}

/// Re-dresses an OpenAI-shaped [`ApiError`] from a shared helper
/// (`validate_supported_fields`, `prompt_from_messages`,
/// `decode_error_response`, `require_model`) in the Anthropic envelope,
/// keeping its status and message.
///
/// The alternative -- letting the shared shape through -- is what makes
/// a Claude client report "unexpected response" for a request the
/// server refused for a reason it stated perfectly clearly.
fn anthropic_shape(err: ApiError) -> ApiError {
    let (status, Json(body)) = err;
    let message = body
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or("request failed")
        .to_string();
    anthropic_error(status, &message)
}

/// A body that is JSON but not a Messages request.
///
/// Answered here rather than by axum's own extractor rejection so it
/// arrives in the Anthropic envelope: this is the one class of 400 a
/// Claude client hits while its own request builder is wrong, which is
/// exactly when a readable error matters.
fn body_error(err: serde_json::Error) -> ApiError {
    anthropic_error(StatusCode::BAD_REQUEST, &err.to_string())
}

// ---------------------------------------------------------------------
// Output shaping
// ---------------------------------------------------------------------

/// The Anthropic stop reason for one frink finish reason, or `None`
/// for one Anthropic has no word for.
///
/// `"cancelled"` is deliberately unmapped: Anthropic's vocabulary is
/// `end_turn` / `max_tokens` / `stop_sequence` / `tool_use`, and every
/// one of them asserts the turn ended on its own terms. A generation
/// stopped by `POST /v1/cancel` did not, so it reports a null stop
/// reason -- which is the reference's own behaviour for a finish reason
/// outside its map, and the only value that does not claim something
/// untrue.
///
/// `stop_sequence` is produced only when a CALLER's stop string
/// matched, and never on its own: it is honest only beside
/// `stop_sequence: "<the string that matched>"`, so both come from the
/// one `matched_stop` value or neither is reported. See
/// [`terminal_stop`], which is where that decision is made -- this
/// function maps the finish reason alone.
fn stop_reason(finish: &str) -> Option<&'static str> {
    match finish {
        "stop" => Some("end_turn"),
        "length" => Some("max_tokens"),
        "tool_calls" => Some("tool_use"),
        _ => None,
    }
}

/// The `stop_reason` / `stop_sequence` pair for one ended generation.
///
/// Three rules, in this order:
///
/// 1. A truncated generation is `max_tokens` even if it parsed a call
///    and even if a stop matched -- a client must not execute a call
///    whose arguments may have been cut off mid-write.
/// 2. A generation that produced calls is `tool_use`.
/// 3. A caller's stop string that fired is `stop_sequence`, WITH the
///    string beside it. `matched_stop` is already filtered to what the
///    client asked for, so a template's own end-of-turn marker lands
///    here as `None` and reports the ordinary `end_turn`.
fn terminal_stop(
    finish: &str,
    calls: bool,
    matched_stop: Option<&str>,
) -> (Option<&'static str>, Value) {
    if finish == "length" {
        return (stop_reason(finish), Value::Null);
    }
    if calls {
        return (stop_reason("tool_calls"), Value::Null);
    }
    match matched_stop {
        Some(stop) => (Some("stop_sequence"), json!(stop)),
        None => (stop_reason(finish), Value::Null),
    }
}

/// The caller's own stop string, if that is what ended this
/// generation.
///
/// `FinishReason` names whichever stop the matcher hit, and the served
/// template contributes stops of its own -- a family's end-of-turn
/// marker among them. Reporting one of THOSE as `stop_sequence` would
/// tell an agent it ran into a fence it never put up, so the match is
/// kept only when it is a string the client actually sent in
/// `stop_sequences`.
fn caller_stop(finish: &crate::generate::FinishReason, caller: &[String]) -> Option<String> {
    let matched = finish.matched_stop()?;
    caller
        .iter()
        .any(|s| s == matched)
        .then(|| matched.to_string())
}

/// Rule 6. `input_tokens` excludes what the prefix cache served and
/// `cache_read_input_tokens` carries it, absent entirely when zero.
///
/// `Usage::cached_tokens` is `None` when no prefix cache is configured
/// and `Some(0)` when one was consulted and missed; both mean "nothing
/// was served from cache" here, and neither may add a zero row a client
/// would render as a cache that exists.
fn usage_json(usage: &Usage) -> Value {
    let cached = usage.cached_tokens.unwrap_or(0);
    let mut out = json!({
        "input_tokens": usage.prompt_tokens.saturating_sub(cached),
        "output_tokens": usage.completion_tokens,
    });
    if cached > 0 {
        out["cache_read_input_tokens"] = json!(cached);
    }
    out
}

/// A `tool_use` block's `input`, which is an object on this wire even
/// though the arguments travel as a string internally. Arguments that
/// do not parse as an object become `{}` rather than a string, because
/// the field is typed as an object and a client will index it.
fn parse_json_args(arguments: &str) -> Value {
    match serde_json::from_str::<Value>(arguments) {
        Ok(value @ Value::Object(_)) => value,
        _ => json!({}),
    }
}

/// The whole buffered answer as one Anthropic message.
///
/// The text block is emitted even when it is empty, as the reference
/// does: `content` is a list a client indexes, and a turn that only
/// called tools still has a (blank) assistant utterance in it.
fn message_body(
    parsed: ParsedOutput,
    finish: &str,
    matched_stop: Option<&str>,
    usage: &Usage,
    id: &str,
    model: &str,
) -> Value {
    let mut content: Vec<Value> = Vec::new();
    if let Some(reasoning) = parsed.reasoning.filter(|r| !r.is_empty()) {
        // `signature: ""`: this server has no signing key and never
        // verifies signatures on replayed thinking blocks, so an empty
        // one is the honest value for a shape that requires the field.
        content.push(json!({"type": "thinking", "thinking": reasoning, "signature": ""}));
    }
    content.push(json!({"type": "text", "text": parsed.content}));
    for (index, call) in parsed.calls.iter().enumerate() {
        content.push(json!({
            "type": "tool_use",
            "id": tool_use_id(&call.name, index),
            "name": call.name,
            "input": parse_json_args(&call.arguments),
        }));
    }
    let (reason, sequence) = terminal_stop(finish, !parsed.calls.is_empty(), matched_stop);
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": reason,
        "stop_sequence": sequence,
        "usage": usage_json(usage),
    })
}

// ---------------------------------------------------------------------
// Streaming: semantic events -> Anthropic stream events
// ---------------------------------------------------------------------

/// What the generation thread tells the stream, in the vocabulary the
/// engine already speaks (the policy parser's events plus a
/// terminal). Protocol-neutral, which is what lets every ordering rule
/// below be tested with no model and no socket.
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
        /// The caller's own stop string, when generation ran into one.
        ///
        /// Filtered to the strings the CLIENT sent in
        /// `stop_sequences`: the served template adds stops of its own
        /// (a family's end-of-turn marker), and reporting one of those
        /// as `stop_sequence` would tell an agent it hit a fence it
        /// never put up.
        matched_stop: Option<String>,
        usage: Usage,
    },
    Failed {
        status: StatusCode,
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
            tracing::error!("failed to serialize an anthropic stream event: {e}");
            "{}".to_string()
        });
        Event::default().event(self.name).data(data)
    }
}

/// The content block currently open. At most one is open at a time;
/// anything that starts a different kind closes it first.
#[derive(Debug, Clone, PartialEq, Eq)]
enum OpenBlock {
    Text,
    Thinking,
    Tool {
        /// Which call this block is, so a close for a different one is
        /// recognized as not belonging to it.
        ordinal: usize,
        /// Arguments already on the wire, which is what the close tops
        /// up against.
        sent: String,
    },
}

/// Turns the semantic event stream into Anthropic stream events.
///
/// Split out of the handler so the ordering rules are testable
/// directly: every stream test below drives this with a `Vec<GenEvent>`.
pub(crate) struct MessagesStream {
    message_id: String,
    model: String,
    /// The running `index` every content block carries. It advances on
    /// `content_block_stop` and nowhere else -- a client keys its
    /// assembled blocks on it, so an index that advanced on *open*
    /// would leave a hole for every block that never opened.
    index: usize,
    open: Option<OpenBlock>,
    /// How many tool blocks have gone out, which is how the terminal
    /// event knows whether this turn ended in a call.
    calls_opened: usize,
}

impl MessagesStream {
    pub(crate) fn new(message_id: String, model: String) -> Self {
        MessagesStream {
            message_id,
            model,
            index: 0,
            open: None,
            calls_opened: 0,
        }
    }

    /// Stamps `type` on every frame in one place, so an event's name
    /// and its body can never disagree.
    fn frame(&mut self, name: &'static str, mut data: Value) -> Frame {
        if let Some(object) = data.as_object_mut() {
            object.insert("type".to_string(), json!(name));
        }
        Frame { name, data }
    }

    /// `message_start`, emitted the moment the SSE headers go out so a
    /// client has the message id even if the model never produces a
    /// token. The usage here is a placeholder: the real counts ride on
    /// the terminal `message_delta`, which is the only event that can
    /// know them.
    pub(crate) fn opening(&mut self) -> Vec<Frame> {
        let message = json!({
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": self.model,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        });
        vec![self.frame("message_start", json!({"message": message}))]
    }

    fn close_open(&mut self) -> Vec<Frame> {
        let Some(open) = self.open.take() else {
            return Vec::new();
        };
        let index = self.index;
        let mut frames = Vec::new();
        if open == OpenBlock::Thinking {
            // A real thinking block ends with a signature_delta. This
            // server has no signing key, so the delta is empty -- but it
            // is still sent, because a client that expects the pair
            // treats a thinking block without one as truncated.
            frames.push(self.frame(
                "content_block_delta",
                json!({"index": index, "delta": {"type": "signature_delta", "signature": ""}}),
            ));
        }
        frames.push(self.frame("content_block_stop", json!({"index": index})));
        self.index += 1;
        frames
    }

    fn open_text(&mut self) -> Vec<Frame> {
        let mut frames = self.close_open();
        let index = self.index;
        self.open = Some(OpenBlock::Text);
        frames.push(self.frame(
            "content_block_start",
            json!({"index": index, "content_block": {"type": "text", "text": ""}}),
        ));
        frames
    }

    fn open_thinking(&mut self) -> Vec<Frame> {
        let mut frames = self.close_open();
        let index = self.index;
        self.open = Some(OpenBlock::Thinking);
        frames.push(self.frame(
            "content_block_start",
            json!({"index": index, "content_block": {"type": "thinking", "thinking": ""}}),
        ));
        frames
    }

    fn open_tool(&mut self, name: &str, ordinal: usize) -> Vec<Frame> {
        let mut frames = self.close_open();
        let index = self.index;
        let id = tool_use_id(name, index);
        self.open = Some(OpenBlock::Tool {
            ordinal,
            sent: String::new(),
        });
        self.calls_opened += 1;
        frames.push(self.frame(
            "content_block_start",
            json!({
                "index": index,
                "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}},
            }),
        ));
        frames
    }

    fn arguments_delta(&mut self, fragment: &str) -> Option<Frame> {
        match &mut self.open {
            Some(OpenBlock::Tool { sent, .. }) => sent.push_str(fragment),
            // Defensive: a fragment with no tool block open is dropped
            // rather than attached to a text block, where it would show
            // a user raw JSON.
            _ => return None,
        }
        let index = self.index;
        Some(self.frame(
            "content_block_delta",
            json!({
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": fragment},
            }),
        ))
    }

    /// One semantic event in, its stream events out.
    pub(crate) fn push(&mut self, event: GenEvent) -> Vec<Frame> {
        match event {
            // A protocol-native event, not an SSE comment: a comment
            // does not reach a client's event handler, so an idle
            // stream would still look dead to the thing timing it.
            GenEvent::Keepalive => vec![self.frame("ping", json!({}))],
            GenEvent::Reasoning(text) => {
                if text.is_empty() {
                    return Vec::new();
                }
                let mut frames = Vec::new();
                if self.open != Some(OpenBlock::Thinking) {
                    frames.extend(self.open_thinking());
                }
                let index = self.index;
                frames.push(self.frame(
                    "content_block_delta",
                    json!({"index": index, "delta": {"type": "thinking_delta", "thinking": text}}),
                ));
                frames
            }
            GenEvent::Content(text) => {
                if text.is_empty() {
                    return Vec::new();
                }
                let mut frames = Vec::new();
                if self.open != Some(OpenBlock::Text) {
                    frames.extend(self.open_text());
                }
                let index = self.index;
                frames.push(self.frame(
                    "content_block_delta",
                    json!({"index": index, "delta": {"type": "text_delta", "text": text}}),
                ));
                frames
            }
            GenEvent::CallStart { index, name } => self.open_tool(&name, index),
            GenEvent::CallArguments { fragment, .. } => {
                self.arguments_delta(&fragment).into_iter().collect()
            }
            GenEvent::CallEnd { index, arguments } => {
                let streamed = match &self.open {
                    Some(OpenBlock::Tool { ordinal, sent }) if *ordinal == index => sent.clone(),
                    // A close with nothing open, or with a different
                    // call open: there is nothing here to close.
                    _ => return Vec::new(),
                };
                let mut frames = Vec::new();
                // The final arguments are authoritative: top up whatever
                // has not gone out yet, so a client concatenating
                // `partial_json` ends with exactly this string.
                //
                // The `strip_prefix` guard is what enforces "only while
                // fragments are prefix-stable". the policy parser
                // emits literal continuations of the arguments JSON (or
                // nothing at all until the block completes), so the
                // guard holds; if a format ever broke that, this sends
                // no misleading top-up rather than a remainder that does
                // not concatenate.
                if let Some(remainder) = arguments.strip_prefix(streamed.as_str()) {
                    if !remainder.is_empty() {
                        frames.extend(self.arguments_delta(remainder));
                    }
                }
                frames.extend(self.close_open());
                frames
            }
            GenEvent::WholeCall {
                index,
                name,
                arguments,
            } => {
                // Open, deliver and close at once, so a client holds a
                // complete block even if the stream dies straight after.
                let mut frames = self.open_tool(&name, index);
                frames.extend(self.arguments_delta(&arguments));
                frames.extend(self.close_open());
                frames
            }
            GenEvent::Done {
                finish,
                matched_stop,
                usage,
            } => {
                let mut frames = self.close_open();
                let (reason, sequence) =
                    terminal_stop(finish, self.calls_opened > 0, matched_stop.as_deref());
                let mut delta = json!({});
                if let Some(reason) = reason {
                    delta["stop_reason"] = json!(reason);
                }
                // Only beside a `stop_sequence` reason. A null here on
                // every other ending is noise a client has to ignore,
                // and the buffered body carries it because its shape is
                // fixed; a delta's is not.
                if !sequence.is_null() {
                    delta["stop_sequence"] = sequence;
                }
                let usage = usage_json(&usage);
                frames.push(self.frame("message_delta", json!({"delta": delta, "usage": usage})));
                frames.push(self.frame("message_stop", json!({})));
                // Rule 5: nothing follows. An Anthropic stream ends on
                // `message_stop`, and `data: [DONE]` is an OpenAI
                // sentinel a strict client rejects as an unknown event.
                frames
            }
            GenEvent::Failed { status, message } => {
                let mut frames = self.close_open();
                let error = json!({"type": error_type(status), "message": message});
                frames.push(self.frame("error", json!({"error": error})));
                frames
            }
        }
    }
}

// ---------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------

/// `POST /v1/messages`.
///
/// Takes the body as a raw [`Value`] and deserializes it here so a
/// malformed request answers in the Anthropic error envelope instead of
/// axum's extractor rejection, which a Claude client cannot read.
pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let attribution = attribution::Attribution::from_headers(&headers);
    state
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let started = std::time::Instant::now();
    // Same server-assigned id scheme as /v1/chat/completions, so one
    // ring buffer keys both surfaces the same way.
    let request_id = frink_api::next_request_id();

    let parsed = serde_json::from_value::<MessagesRequest>(body).map_err(body_error);
    let stream = parsed
        .as_ref()
        .ok()
        .and_then(|req| req.stream)
        .unwrap_or(false);

    let result = async {
        // The maintenance gate, before the body is even looked at: a
        // request admitted while the KV pool is being re-split would
        // decode out of an allocation that is about to change size.
        crate::cache_admin::check_admission(&state).map_err(anthropic_shape)?;
        let req = parsed?;
        let _ = &req.metadata;
        let prepared = to_chat_request(&req)?;
        if stream {
            messages_stream(
                Arc::clone(&state),
                prepared,
                request_id.clone(),
                started,
                attribution.clone(),
            )
            .await
        } else {
            messages_full(
                Arc::clone(&state),
                prepared,
                request_id.clone(),
                started,
                attribution.clone(),
            )
            .await
        }
    }
    .await;

    let mut response = match result {
        Ok(response) => response,
        Err(err) => err.into_response(),
    };
    // The id `/v1/cancel` takes, stated on the response itself.
    //
    // The Anthropic message protocol has nowhere to put it: the
    // `message_start` id is a `msg_...` the server invents per message
    // and the cancel registry does not know, so without this header a
    // streamed `/v1/messages` could be started and never stopped except
    // by dropping the socket. The real Anthropic API spells the same
    // thing `request-id`, so a client that already reads it needs no
    // change.
    if let Ok(value) = axum::http::HeaderValue::from_str(&request_id) {
        response
            .headers_mut()
            .insert(axum::http::HeaderName::from_static("request-id"), value);
    }
    if response.status().is_client_error() || response.status().is_server_error() {
        state
            .request_errors_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Only failures are recorded here; a success records itself from
        // the path that knows the token counts, and for a stream that
        // has not happened yet.
        state.record_request(stats::Record {
            request_id: &request_id,
            route: frink_api::routes::V1_MESSAGES,
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

async fn messages_full(
    state: Arc<AppState>,
    prepared: Prepared,
    request_id: String,
    started: std::time::Instant,
    attribution: attribution::Attribution,
) -> Result<Response, ApiError> {
    // Cloned once, up front: this request decodes against exactly this
    // model even if `/admin/models/load` swaps another one in halfway.
    let active = state.require_active().map_err(anthropic_shape)?;
    let chat = prepared.chat;
    let template = active
        .generative()
        .map_err(anthropic_shape)?
        .chat_template();
    let kwargs = chat.resolve_template_kwargs(&template);
    let prompt = chat
        .render_prompt(
            &chat.messages,
            &template,
            &chat.tools,
            kwargs,
            active.name(),
        )
        .map_err(anthropic_shape)?;
    let posture = OutputPosture::resolve_with(active.reasoning_format(), active.name(), &prompt);
    // The client's own list, kept apart from `params.stop`, which the
    // template adds its end-of-turn marker to. See `caller_stop`.
    let caller_stops = chat.stop_sequences();
    let params =
        chat.generation_params_for_template(&template, active.name(), active.sampler_model())?;

    let (chunks, finish, usage) = crate::decode_task::buffered(
        crate::decode_task::DecodeHandles::take(&state, &active).map_err(anthropic_shape)?,
        prompt,
        params,
    )
    .await
    .map_err(anthropic_shape)?;

    let parsed = output::parse_output(&chunks.concat(), &prepared.parser_tools, posture);
    state.record_request(stats::Record {
        request_id: &request_id,
        route: frink_api::routes::V1_MESSAGES,
        // The handle this request decoded against, not `chat.model`: a
        // swap mid-flight does not change which weights answered.
        model: Some(active.name().to_string()),
        status: 200,
        stream: false,
        duration_ms: started.elapsed().as_millis() as u64,
        usage: Some(&usage),
        attribution: &attribution,
    });
    let matched_stop = caller_stop(&finish, &caller_stops);
    Ok(Json(message_body(
        parsed,
        finish.as_str(),
        matched_stop.as_deref(),
        &usage,
        &new_id("msg_"),
        &chat.model,
    ))
    .into_response())
}

async fn messages_stream(
    state: Arc<AppState>,
    prepared: Prepared,
    request_id: String,
    started: std::time::Instant,
    attribution: attribution::Attribution,
) -> Result<Response, ApiError> {
    // See `messages_full`: the handle is taken once and the whole stream
    // runs against it, so a mid-stream model swap cannot splice two
    // checkpoints into one answer.
    let active = state.require_active().map_err(anthropic_shape)?;
    let chat = prepared.chat;
    let template = active
        .generative()
        .map_err(anthropic_shape)?
        .chat_template();
    let kwargs = chat.resolve_template_kwargs(&template);
    let served_model = active.name().to_string();
    let prompt = chat
        .render_prompt(
            &chat.messages,
            &template,
            &chat.tools,
            kwargs,
            &served_model,
        )
        .map_err(anthropic_shape)?;
    let posture = OutputPosture::resolve_with(active.reasoning_format(), &served_model, &prompt);
    // See `messages_full`: the client's list, not the template's.
    let caller_stops = chat.stop_sequences();
    let mut params =
        chat.generation_params_for_template(&template, &served_model, active.sampler_model())?;

    // The same two-tier cancellation the chat stream has: the guard
    // rides with the generation task and deregisters however that task
    // ends, panic included.
    let (cancel_token, cancel_guard) = state.cancels.register(&request_id);
    params.cancel = Some(cancel_token.clone());

    let model = Arc::clone(active.generative().map_err(anthropic_shape)?);
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
    let offered = prepared.parser_tools;
    let stats_state = Arc::clone(&state);
    let stats_request_id = request_id.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);
    tokio::task::spawn_blocking(move || {
        let _cancel_guard = cancel_guard;
        let orphan = sse::orphan_timeout_from_env();
        let send = |event: GenEvent| {
            if sse::send_or_orphan(&tx, event, orphan).is_err() {
                // The reader is gone or has stopped reading. This stream
                // keeps no replay buffer, so there is nothing left to
                // generate for.
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
            Ok((finish, usage, full_text)) => {
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
                    route: frink_api::routes::V1_MESSAGES,
                    model: Some(served_model.clone()),
                    status: 200,
                    stream: true,
                    duration_ms: started.elapsed().as_millis() as u64,
                    usage: Some(&usage),
                    attribution: &attribution,
                });
                send(GenEvent::Done {
                    finish: finish.as_str(),
                    matched_stop: caller_stop(&finish, &caller_stops),
                    usage,
                });
            }
            Err(e) => {
                tracing::warn!("decode error on streamed message {stats_request_id}: {e}");
                // The socket carried 200 -- SSE headers precede the
                // first token -- but the request produced no answer. A
                // 200 row with zero tokens would read as a successful
                // empty response, so the failure is stated as 500 here
                // and only here.
                stats_state.record_request(stats::Record {
                    request_id: &stats_request_id,
                    route: frink_api::routes::V1_MESSAGES,
                    model: Some(served_model.clone()),
                    status: 500,
                    stream: true,
                    duration_ms: started.elapsed().as_millis() as u64,
                    usage: None,
                    attribution: &attribution,
                });
                // Classified exactly as the non-streaming path
                // classifies the same failure, so a client sees "your
                // request was too big" as a client error in-stream
                // rather than a server fault worth retrying.
                let (status, Json(body)) = decode_error_response(e);
                let message = body
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("generation failed")
                    .to_string();
                send(GenEvent::Failed { status, message });
            }
        }
    });

    let mut machine = MessagesStream::new(new_id("msg_"), chat.model.clone());
    let queue: VecDeque<Frame> = machine.opening().into();
    // Boxed because `StreamExt::next` needs `Unpin` and an `unfold` over
    // an async block is not.
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
    // **No `keep_alive`.** axum's keepalive is an SSE comment, which
    // never reaches a client's event handler; the keepalive here is a
    // real `ping` event inserted by `with_keepalive` above.
    Ok((
        [(
            axum::http::HeaderName::from_static("x-accel-buffering"),
            axum::http::HeaderValue::from_static("no"),
        )],
        Sse::new(stream),
    )
        .into_response())
}

/// `POST /v1/messages/count_tokens`.
///
/// Answers the number of prompt tokens a `/v1/messages` request with
/// this body would report as `usage.input_tokens`, by running the same
/// converter, the same template tools and the same
/// `chat_template_kwargs` and then rendering with the served
/// checkpoint's own template (rule 7). It decodes nothing, so it
/// answers while the engine is busy serving other requests.
pub(crate) async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let attribution = attribution::Attribution::from_headers(&headers);
    let started = std::time::Instant::now();
    let request_id = frink_api::next_request_id();

    let result = (|| {
        let req = serde_json::from_value::<CountTokensRequest>(body).map_err(body_error)?;
        let prepared = countable_prompt(&req)?;
        // Tokenizing needs the loaded vocabulary, so a server with
        // nothing loaded answers 503 here exactly as `/v1/tokenize`
        // does -- counting against a guessed vocabulary would answer a
        // number no generation could reproduce.
        //
        // The reference classifies a tokenizer that fails to
        // *initialize* as a 500. Frink has no such state to report: a
        // model is either loaded, tokenizer included (503 above
        // otherwise), and `Model::encode` is infallible from there. The
        // branch is deliberately not fabricated.
        let model = state.require_model().map_err(anthropic_shape)?;
        let template = model.chat_template();
        let kwargs = prepared.chat.resolve_template_kwargs(&template);
        // A template that refuses *this conversation* -- a tool result
        // with no call before it, a role order it forbids -- is a client
        // error, the same 400 `/v1/messages` answers for the same body.
        // Through the same renderer as the generation, so a trailing
        // assistant turn is counted as the continuation it will be.
        let prompt = prepared
            .chat
            .render_prompt(
                &prepared.chat.messages,
                &template,
                &prepared.chat.tools,
                kwargs,
                state.active_model_name().as_deref().unwrap_or_default(),
            )
            .map_err(anthropic_shape)?;
        // The count a generation would report is the encoded prompt
        // plus whatever BOS the decode path prepends
        // (`generate::generate` calls `prepend_bos`). `Model` exposes no
        // BOS id, so this can read one token low on a checkpoint that
        // has one; it is stated rather than guessed at.
        let input_tokens = model.encode(&prompt, SpecialTokens::Parse).len();
        Ok::<Value, ApiError>(json!({"input_tokens": input_tokens}))
    })();

    let response = match result {
        Ok(body) => Json(body).into_response(),
        Err(err) => err.into_response(),
    };
    if response.status().is_client_error() || response.status().is_server_error() {
        state
            .request_errors_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    // No `usage`, deliberately, exactly as `/v1/tokenize` records none:
    // this ran the tokenizer and not the model, and `prompt_tokens` here
    // feeds `tokens_prompt_total`, which means tokens that went through
    // a forward pass.
    state.record_request(stats::Record {
        request_id: &request_id,
        route: frink_api::routes::V1_MESSAGES_COUNT_TOKENS,
        model: state.active_model_name(),
        status: response.status().as_u16(),
        stream: false,
        duration_ms: started.elapsed().as_millis() as u64,
        usage: None,
        attribution: &attribution,
    });
    response
}

/// The two client-side refusals `count_tokens` makes before any
/// tokenizer is touched, kept separate from each other on purpose.
///
/// An empty `messages` array is a malformed request. A *non-empty* one
/// that converts to nothing -- an image-only turn on this text-only
/// server -- is a different client error and says so, because "at least
/// one message is required" would be visibly false to a caller who sent
/// one.
fn countable_prompt(req: &CountTokensRequest) -> Result<Prepared, ApiError> {
    if req.prompt.messages.is_empty() {
        return Err(anthropic_error(
            StatusCode::BAD_REQUEST,
            "messages: at least one message is required",
        ));
    }
    // `max_tokens: 1` is a placeholder: this endpoint has no output
    // budget, nothing here decodes, and the shared request type requires
    // a positive one.
    let prepared = prepare_prompt(&req.prompt, req.model.clone(), 1)?;
    if prepared.chat.messages.is_empty() {
        return Err(anthropic_error(
            StatusCode::BAD_REQUEST,
            "messages: no tokenizable content",
        ));
    }
    Ok(prepared)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn messages_request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("the fixture is a valid Messages request")
    }

    fn count_request(value: Value) -> CountTokensRequest {
        serde_json::from_value(value).expect("the fixture is a valid count_tokens request")
    }

    fn converted(value: Value) -> Prepared {
        to_chat_request(&messages_request(value)).expect("the fixture converts")
    }

    fn text_of(message: &ChatMessage) -> String {
        message
            .content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default()
    }

    /// The shape of a converted conversation, as a value two conversions
    /// can be compared on: `ChatMessage` is deserialize-only and has no
    /// `PartialEq`.
    fn shape(messages: &[ChatMessage]) -> Vec<(String, String, Option<String>, Vec<String>)> {
        messages
            .iter()
            .map(|m| {
                (
                    m.role.clone(),
                    text_of(m),
                    m.tool_call_id.clone(),
                    m.tool_calls
                        .as_ref()
                        .map(|calls| {
                            calls
                                .iter()
                                .map(|c| format!("{}({})", c.function.name, c.function.arguments))
                                .collect()
                        })
                        .unwrap_or_default(),
                )
            })
            .collect()
    }

    fn usage(prompt: usize, completion: usize) -> Usage {
        Usage::new(prompt, completion)
    }

    fn run(events: Vec<GenEvent>) -> Vec<Frame> {
        let mut machine = MessagesStream::new("msg_test".to_string(), "test-model".to_string());
        let mut frames = machine.opening();
        for event in events {
            frames.extend(machine.push(event));
        }
        frames
    }

    fn names(frames: &[Frame]) -> Vec<&str> {
        frames.iter().map(|f| f.name).collect()
    }

    fn only(frames: &[Frame], name: &str) -> Vec<Value> {
        frames
            .iter()
            .filter(|f| f.name == name)
            .map(|f| f.data.clone())
            .collect()
    }

    // -----------------------------------------------------------------
    // Rule 1: one leading system message
    // -----------------------------------------------------------------

    /// **Fails if the two system sources become two messages.** Claude
    /// Code sends a top-level `system` *and* system-role messages in the
    /// array; emitting both is what a strict template answers with
    /// "System message must be at the beginning", failing a request that
    /// is perfectly well formed.
    #[test]
    fn the_top_level_system_and_a_system_role_message_merge_into_one_leading_message() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "system": "you are terse",
            "messages": [
                {"role": "system", "content": "and precise"},
                {"role": "user", "content": "hi"},
            ],
        }));
        let messages = &prepared.chat.messages;
        assert_eq!(
            messages.iter().filter(|m| m.role == "system").count(),
            1,
            "exactly one system message must survive: {:?}",
            shape(messages)
        );
        assert_eq!(messages[0].role, "system", "and it must lead");
        assert_eq!(text_of(&messages[0]), "you are terse\n\nand precise");
        assert_eq!(messages[1].role, "user");
    }

    /// A system turn sitting *after* a user turn is still hoisted to the
    /// front rather than left where it sat.
    #[test]
    fn a_system_message_in_the_middle_of_the_array_still_leads_the_conversation() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "be terse"},
                {"role": "assistant", "content": "ok"},
            ],
        }));
        assert_eq!(
            shape(&prepared.chat.messages)
                .iter()
                .map(|(role, ..)| role.clone())
                .collect::<Vec<_>>(),
            vec!["system", "user", "assistant"]
        );
    }

    /// A `system` sent as text blocks concatenates, and an empty one
    /// contributes no message at all.
    #[test]
    fn a_block_list_system_concatenates_and_an_absent_one_adds_no_message() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(text_of(&prepared.chat.messages[0]), "ab");

        let bare = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(bare.chat.messages.len(), 1);
        assert_eq!(bare.chat.messages[0].role, "user");
    }

    // -----------------------------------------------------------------
    // Rule 2: tool_result is keyed by tool_use_id
    // -----------------------------------------------------------------

    /// **Fails if the result is keyed by `id`.** The fixture carries
    /// both fields with different values, which is the real wire shape:
    /// `id` names the result block and `tool_use_id` names the call it
    /// answers. Key on `id` and two parallel results become
    /// indistinguishable to the template, so the model is shown answers
    /// it cannot match to its own calls.
    #[test]
    fn a_tool_result_is_keyed_by_its_tool_use_id_and_not_by_its_own_id() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_read", "name": "read", "input": {"path": "a"}},
                    {"type": "tool_use", "id": "toolu_grep", "name": "grep", "input": {"q": "b"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "id": "block_1", "tool_use_id": "toolu_read",
                     "content": "file a"},
                    {"type": "tool_result", "id": "block_2", "tool_use_id": "toolu_grep",
                     "content": "no match"},
                ]},
            ],
        }));
        let tools: Vec<_> = prepared
            .chat
            .messages
            .iter()
            .filter(|m| m.role == "tool")
            .map(|m| (m.tool_call_id.clone().unwrap_or_default(), text_of(m)))
            .collect();
        assert_eq!(
            tools,
            vec![
                ("toolu_read".to_string(), "file a".to_string()),
                ("toolu_grep".to_string(), "no match".to_string()),
            ],
            "each result must name the call it answers, not its own block id"
        );
    }

    /// A `tool_use` becomes an OpenAI `tool_calls` entry whose
    /// `arguments` is a JSON *string* -- the real OpenAI convention, and
    /// what `chat_template::tool_call_json` decodes back for templates
    /// that want a dict. Passing the object through unencoded produces a
    /// double-encoded string in every such template.
    #[test]
    fn a_tool_use_block_becomes_a_tool_call_with_json_encoded_arguments() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a.rs"}},
            ]}],
        }));
        let calls = prepared.chat.messages[0]
            .tool_calls
            .as_ref()
            .expect("the assistant turn carries its call");
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"a.rs"}"#);
        assert!(
            prepared.chat.messages[0].content.is_none(),
            "a turn that only called tools has no content, per the OpenAI convention"
        );
    }

    /// A tool result carried as a block list flattens to its text, and a
    /// result on an *assistant* turn (which has no `role: \"tool\"` slot
    /// to go to) becomes labelled prose instead of vanishing.
    #[test]
    fn a_block_list_tool_result_flattens_and_an_assistant_side_result_becomes_prose() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": "late"},
                ]},
            ],
        }));
        let shaped = shape(&prepared.chat.messages);
        assert_eq!(shaped[0].0, "tool");
        assert_eq!(shaped[0].1, "onetwo");
        assert_eq!(shaped[1].0, "assistant");
        assert_eq!(shaped[1].1, "Tool result: late");
    }

    /// A user turn carrying a result *and* a follow-up question keeps
    /// that order: the result is its own message, emitted where it sat.
    #[test]
    fn a_tool_result_and_a_question_in_one_turn_keep_their_order() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
                {"type": "text", "text": "now what?"},
            ]}],
        }));
        let shaped = shape(&prepared.chat.messages);
        assert_eq!(shaped[0].0, "tool");
        assert_eq!(shaped[1].0, "user");
        assert_eq!(shaped[1].1, "now what?");
    }

    // -----------------------------------------------------------------
    // Rule 3: unknown blocks are skipped
    // -----------------------------------------------------------------

    /// **Fails if an unknown block type is a 4xx.** `redacted_thinking`
    /// and `image` are real blocks Claude Code sends today, and the list
    /// grows without this server's involvement: refusing the request
    /// takes the whole endpoint down for one new client version, while
    /// the surrounding text still answers the question perfectly well.
    #[test]
    fn redacted_thinking_images_and_unknown_blocks_are_skipped_rather_than_refused() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
                {"type": "some_future_block", "whatever": 1},
                {"type": "text", "text": "what is this?"},
            ]}],
        }));
        assert_eq!(
            prepared.chat.messages.len(),
            1,
            "the request still converts: {:?}",
            shape(&prepared.chat.messages)
        );
        assert_eq!(text_of(&prepared.chat.messages[0]), "what is this?");
    }

    /// A message left with nothing renderable contributes no turn --
    /// rather than a blank one the model reads as a turn it took.
    #[test]
    fn a_message_whose_blocks_are_all_skipped_contributes_no_turn() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "data": "..."}},
                ]},
                {"role": "user", "content": "still here"},
            ],
        }));
        assert_eq!(prepared.chat.messages.len(), 1);
        assert_eq!(text_of(&prepared.chat.messages[0]), "still here");
    }

    /// A `thinking` block reaches the template as `reasoning_content`
    /// and never as `content`. Claude Code replays the thinking of the
    /// turn that opened a tool loop, so it has to survive; shown as
    /// `content` it would become something the model believes it said
    /// out loud.
    #[test]
    fn a_thinking_block_is_replayed_beside_the_answer_and_never_as_content() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "the user probably means X"},
                {"type": "text", "text": "X."},
            ]}],
        }));
        assert_eq!(text_of(&prepared.chat.messages[0]), "X.");
        assert_eq!(
            prepared.chat.messages[0].reasoning_content.as_deref(),
            Some("the user probably means X"),
        );

        let thinking_only = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm"},
            ]}],
        }));
        assert!(
            thinking_only.chat.messages.is_empty(),
            "a turn that held only thinking contributes no message"
        );
    }

    // -----------------------------------------------------------------
    // Rule 4: tool_choice splits the two lists
    // -----------------------------------------------------------------

    fn tools_fixture() -> Value {
        json!([
            {"name": "read", "description": "read a file",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
            {"name": "write", "description": "write a file", "input_schema": {}},
        ])
    }

    /// **Fails if `\"none\"` only hides the tools from the template.**
    /// The parser must lose them too: with a parser still armed, prose
    /// that happens to contain a `<tool_call>` marker -- an assistant
    /// explaining tool syntax, say -- is reinterpreted as a call the
    /// caller explicitly asked not to have.
    #[test]
    fn tool_choice_none_hides_the_tools_from_the_template_and_from_the_parser() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "tools": tools_fixture(),
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert!(prepared.chat.tools.is_empty(), "template offers nothing");
        assert!(prepared.parser_tools.is_empty(), "parser is disarmed");
    }

    /// **Fails if a named tool narrows the parser too.** The template is
    /// offered only the named tool, but this server has no constrained
    /// decoding to force that choice, so the model can still call
    /// another one -- and a parser that has never heard of it cannot
    /// type its arguments against a schema.
    #[test]
    fn a_named_tool_choice_narrows_the_template_list_but_not_the_parser() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "tools": tools_fixture(),
            "tool_choice": {"type": "tool", "name": "write"},
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(
            prepared
                .chat
                .tools
                .iter()
                .map(|t| t.function.name.clone())
                .collect::<Vec<_>>(),
            vec!["write"]
        );
        assert_eq!(
            prepared
                .parser_tools
                .iter()
                .map(|t| t.function.name.clone())
                .collect::<Vec<_>>(),
            vec!["read", "write"]
        );
    }

    /// `auto` and `any` leave both lists whole, and a tool whose schema
    /// omits `type` is given `object` so a template that renders the
    /// schema does not describe a typeless tool.
    #[test]
    fn auto_offers_every_tool_and_a_typeless_schema_is_given_object() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "tools": tools_fixture(),
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(prepared.chat.tools.len(), 2);
        assert_eq!(prepared.parser_tools.len(), 2);
        let write = &prepared.chat.tools[1].function;
        assert_eq!(
            write.parameters.as_ref().unwrap()["type"],
            json!("object"),
            "a schema with no type is given one"
        );
    }

    // -----------------------------------------------------------------
    // Sampling and validation
    // -----------------------------------------------------------------

    /// The Anthropic sampling knobs land on the shared chat request, so
    /// they go through the one sampling path rather than a second copy.
    #[test]
    fn the_sampling_knobs_land_on_the_shared_chat_request() {
        let prepared = converted(json!({
            "model": "m",
            "max_tokens": 128,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(prepared.chat.max_tokens, 128);
        assert_eq!(prepared.chat.temperature, Some(0.5));
        assert_eq!(prepared.chat.top_p, Some(0.9));
        assert_eq!(prepared.chat.top_k, Some(40));
        assert_eq!(prepared.chat.stop_sequences(), vec!["END".to_string()]);
    }

    /// `max_tokens: 0` is refused, and the refusal wears the Anthropic
    /// envelope: a Claude client parses `{"type": "error", ...}` and
    /// shows anything else as an unreadable protocol failure.
    #[test]
    fn a_zero_max_tokens_is_refused_in_the_anthropic_error_envelope() {
        let req = messages_request(json!({
            "model": "m",
            "max_tokens": 0,
            "messages": [{"role": "user", "content": "hi"}],
        }));
        let Err((status, Json(body))) = to_chat_request(&req) else {
            panic!("max_tokens: 0 must be refused");
        };
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["type"], json!("error"));
        assert_eq!(body["error"]["type"], json!("invalid_request_error"));
        assert!(body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("max_tokens")));
    }

    /// The extended-thinking toggle is handed to the shared switch
    /// rather than re-implemented, so `/v1/messages` and
    /// `/v1/chat/completions` resolve thinking identically.
    #[test]
    fn the_thinking_toggle_is_handed_to_the_shared_switch() {
        let on = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(
            on.chat.thinking.as_ref().map(|t| t.kind.as_str()),
            Some("enabled")
        );

        let off = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(
            off.chat.thinking.as_ref().map(|t| t.kind.as_str()),
            Some("disabled")
        );
    }

    /// The budget half of `thinking`, llama.cpp's way
    /// (`server-chat.cpp:585-591`): `enabled` carries `budget_tokens`
    /// into `thinking_budget_tokens`, an `enabled` with no number gets
    /// 10,000, and `disabled` carries none.
    #[test]
    fn thinking_budget_tokens_becomes_the_reasoning_budget() {
        use crate::reasoning_budget::BudgetTokens;
        let on = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(
            on.chat.reasoning_budget_tokens,
            Some(BudgetTokens::Tokens(1024))
        );
        let unnumbered = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "thinking": {"type": "enabled"},
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(
            unnumbered.chat.reasoning_budget_tokens,
            Some(BudgetTokens::Tokens(10_000))
        );
        let off = converted(json!({
            "model": "m",
            "max_tokens": 16,
            "thinking": {"type": "disabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(off.chat.reasoning_budget_tokens, None);
        let bad: MessagesRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 16,
            "thinking": {"type": "enabled", "budget_tokens": -5},
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        let Err((status, body)) = to_chat_request(&bad) else {
            panic!("out of range");
        };
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body.0["error"]["message"]
                .as_str()
                .unwrap()
                .contains("thinking.budget_tokens"),
            "{}",
            body.0
        );
    }

    // -----------------------------------------------------------------
    // Rule 6: usage excludes the cached prefix
    // -----------------------------------------------------------------

    /// **Fails if the cached prefix stays in `input_tokens`.** Anthropic
    /// bills `input_tokens` as the *uncached* prompt with
    /// `cache_read_input_tokens` beside it; reporting the full prompt in
    /// both fields double-counts every cached turn, and a client adding
    /// them up sees more tokens than the prompt contains.
    #[test]
    fn usage_excludes_the_cached_prefix_from_input_tokens() {
        let cached = usage(100, 7).with_cached_tokens(40);
        assert_eq!(
            usage_json(&cached),
            json!({
                "input_tokens": 60,
                "output_tokens": 7,
                "cache_read_input_tokens": 40,
            })
        );
    }

    /// Zero cached tokens omit the field entirely rather than reporting
    /// a zero: `Some(0)` means the cache was consulted and missed, and a
    /// client renders a zero row as a cache that exists and did nothing.
    #[test]
    fn a_zero_cache_read_is_absent_rather_than_zero() {
        for usage in [usage(10, 2), usage(10, 2).with_cached_tokens(0)] {
            let json = usage_json(&usage);
            assert_eq!(json["input_tokens"], json!(10));
            assert!(
                json.get("cache_read_input_tokens").is_none(),
                "nothing was served from cache, so the field must be absent"
            );
        }
    }

    // -----------------------------------------------------------------
    // Streaming
    // -----------------------------------------------------------------

    /// **Fails if the stream ends with `data: [DONE]`.** That sentinel
    /// is OpenAI's; an Anthropic stream terminates on `message_stop`,
    /// and a strict client reports a protocol error on the extra event
    /// -- turning a completed answer into a failure.
    #[test]
    fn the_stream_terminates_on_message_stop_with_no_done_sentinel() {
        let frames = run(vec![
            GenEvent::Content("hi".into()),
            GenEvent::Done {
                finish: "stop",
                matched_stop: None,
                usage: usage(5, 1),
            },
        ]);
        assert_eq!(
            names(&frames).last().copied(),
            Some("message_stop"),
            "message_stop is the last event"
        );
        assert!(
            !frames
                .iter()
                .any(|f| f.name == "done" || f.data == json!("[DONE]")),
            "no OpenAI sentinel may follow it: {:?}",
            names(&frames)
        );
    }

    /// The opening and closing shape of an ordinary text answer, in
    /// order. Ordering *is* the content on this surface.
    #[test]
    fn a_text_answer_opens_and_closes_one_block_between_start_and_stop() {
        let frames = run(vec![
            GenEvent::Content("he".into()),
            GenEvent::Content("llo".into()),
            GenEvent::Done {
                finish: "stop",
                matched_stop: None,
                usage: usage(5, 2),
            },
        ]);
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let start = &only(&frames, "message_start")[0];
        assert_eq!(start["message"]["role"], json!("assistant"));
        assert_eq!(start["message"]["content"], json!([]));
        assert_eq!(start["message"]["usage"]["input_tokens"], json!(0));
    }

    /// **Fails if the index advances on `content_block_start`.** It
    /// advances on `content_block_stop` and nowhere else: a client keys
    /// its assembled blocks on this number, and an index bumped at open
    /// would number a thinking block and the text after it 0 and 2, with
    /// a hole where a client waits for a block that never comes.
    #[test]
    fn the_block_index_advances_on_every_content_block_stop() {
        let frames = run(vec![
            GenEvent::Reasoning("think".into()),
            GenEvent::Content("say".into()),
            GenEvent::CallStart {
                index: 0,
                name: "read".into(),
            },
            GenEvent::CallEnd {
                index: 0,
                arguments: "{}".into(),
            },
            GenEvent::Done {
                finish: "stop",
                matched_stop: None,
                usage: usage(5, 3),
            },
        ]);
        let indexes: Vec<i64> = frames
            .iter()
            .filter(|f| f.name == "content_block_start")
            .map(|f| f.data["index"].as_i64().unwrap())
            .collect();
        assert_eq!(indexes, vec![0, 1, 2], "three blocks, numbered in order");
        let stops: Vec<i64> = frames
            .iter()
            .filter(|f| f.name == "content_block_stop")
            .map(|f| f.data["index"].as_i64().unwrap())
            .collect();
        assert_eq!(stops, vec![0, 1, 2], "each closes the index it opened");
    }

    /// **Fails if a thinking block just stops.** A real thinking block
    /// ends with a `signature_delta`; a client that expects the pair
    /// treats a thinking block without one as truncated, so the empty
    /// signature this server can honestly produce is still sent.
    #[test]
    fn a_thinking_block_closes_with_an_empty_signature_delta_first() {
        let frames = run(vec![
            GenEvent::Reasoning("weighing it up".into()),
            GenEvent::Content("answer".into()),
            GenEvent::Done {
                finish: "stop",
                matched_stop: None,
                usage: usage(5, 4),
            },
        ]);
        let deltas = only(&frames, "content_block_delta");
        assert_eq!(deltas[0]["delta"]["type"], json!("thinking_delta"));
        assert_eq!(
            deltas[1]["delta"],
            json!({"type": "signature_delta", "signature": ""})
        );
        let order = names(&frames);
        let signature = order
            .iter()
            .position(|n| *n == "content_block_delta")
            .unwrap()
            + 1;
        assert_eq!(
            order[signature + 1],
            "content_block_stop",
            "the signature delta comes immediately before the stop"
        );
    }

    /// A tool block streams its arguments as `input_json_delta`
    /// fragments and closes only after the authoritative arguments have
    /// been topped up.
    #[test]
    fn a_tool_block_streams_its_arguments_and_tops_up_the_remainder_at_close() {
        let frames = run(vec![
            GenEvent::CallStart {
                index: 0,
                name: "read_file".into(),
            },
            GenEvent::CallArguments {
                index: 0,
                fragment: r#"{"path":"#.into(),
            },
            GenEvent::CallEnd {
                index: 0,
                arguments: r#"{"path":"a.rs"}"#.into(),
            },
            GenEvent::Done {
                finish: "stop",
                matched_stop: None,
                usage: usage(5, 5),
            },
        ]);
        let start = &only(&frames, "content_block_start")[0];
        assert_eq!(start["content_block"]["type"], json!("tool_use"));
        assert_eq!(start["content_block"]["name"], json!("read_file"));
        assert_eq!(start["content_block"]["input"], json!({}));
        assert!(
            start["content_block"]["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("toolu_read-file_0_")),
            "the id names its tool and block: {}",
            start["content_block"]["id"]
        );
        let fragments: Vec<String> = only(&frames, "content_block_delta")
            .iter()
            .map(|d| d["delta"]["partial_json"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            fragments.concat(),
            r#"{"path":"a.rs"}"#,
            "the concatenated fragments are exactly the final arguments"
        );
    }

    /// **Fails if a close re-sends the whole arguments string.** The
    /// top-up is the *remainder*: a client concatenating `partial_json`
    /// would otherwise end with the arguments twice, and parse neither.
    #[test]
    fn a_close_tops_up_only_the_remainder_of_what_already_streamed() {
        let frames = run(vec![
            GenEvent::CallStart {
                index: 0,
                name: "t".into(),
            },
            GenEvent::CallArguments {
                index: 0,
                fragment: r#"{"a":1"#.into(),
            },
            GenEvent::CallEnd {
                index: 0,
                arguments: r#"{"a":1}"#.into(),
            },
        ]);
        let fragments: Vec<String> = only(&frames, "content_block_delta")
            .iter()
            .map(|d| d["delta"]["partial_json"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(fragments, vec![r#"{"a":1"#.to_string(), "}".to_string()]);
    }

    /// A call that never streamed a fragment -- the buffered
    /// continuous-batching path -- still delivers its whole arguments
    /// before the block closes.
    #[test]
    fn a_whole_call_opens_delivers_and_closes_in_one_step() {
        let frames = run(vec![GenEvent::WholeCall {
            index: 0,
            name: "t".into(),
            arguments: r#"{"a":1}"#.into(),
        }]);
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
            ]
        );
        assert_eq!(
            only(&frames, "content_block_delta")[0]["delta"]["partial_json"],
            json!(r#"{"a":1}"#)
        );
    }

    /// A turn that made a call reports `tool_use`, which is what tells
    /// an agent to execute something rather than to show the user an
    /// answer.
    #[test]
    fn a_turn_that_opened_a_tool_block_reports_the_tool_use_stop_reason() {
        let frames = run(vec![
            GenEvent::WholeCall {
                index: 0,
                name: "t".into(),
                arguments: "{}".into(),
            },
            GenEvent::Done {
                finish: "stop",
                matched_stop: None,
                usage: usage(5, 6),
            },
        ]);
        assert_eq!(
            only(&frames, "message_delta")[0]["delta"]["stop_reason"],
            json!("tool_use")
        );
    }

    /// A truncated generation reports `max_tokens` even though it opened
    /// a call: a client must not execute a call whose arguments may have
    /// been cut off mid-write.
    #[test]
    fn a_truncated_turn_reports_max_tokens_even_after_opening_a_call() {
        let frames = run(vec![
            GenEvent::WholeCall {
                index: 0,
                name: "t".into(),
                arguments: "{}".into(),
            },
            GenEvent::Done {
                finish: "length",
                matched_stop: None,
                usage: usage(5, 7),
            },
        ]);
        assert_eq!(
            only(&frames, "message_delta")[0]["delta"]["stop_reason"],
            json!("max_tokens")
        );
    }

    /// A cancelled generation reports no stop reason at all. Every value
    /// in Anthropic's vocabulary asserts the turn ended on its own
    /// terms, and this one did not; a null is the only honest answer,
    /// and it is what the reference produces for the same unmapped
    /// finish reason.
    #[test]
    fn a_cancelled_generation_reports_no_stop_reason() {
        let frames = run(vec![GenEvent::Done {
            finish: "cancelled",
            matched_stop: None,
            usage: usage(5, 8),
        }]);
        let delta = &only(&frames, "message_delta")[0]["delta"];
        assert!(
            delta.get("stop_reason").is_none(),
            "an interrupted turn claims nothing: {delta}"
        );
    }

    /// The terminal usage carries the same cache split rule 6 states,
    /// which is where a client reads its billing from on a stream.
    #[test]
    fn the_terminal_message_delta_carries_the_cache_split_usage() {
        let frames = run(vec![GenEvent::Done {
            finish: "stop",
            matched_stop: None,
            usage: usage(100, 9).with_cached_tokens(40),
        }]);
        assert_eq!(
            only(&frames, "message_delta")[0]["usage"],
            json!({"input_tokens": 60, "output_tokens": 9, "cache_read_input_tokens": 40})
        );
    }

    /// An open block is closed before the terminal events, so a client
    /// never holds a block that was never stopped.
    #[test]
    fn an_open_block_is_closed_before_the_terminal_events() {
        let frames = run(vec![
            GenEvent::Content("half a sen".into()),
            GenEvent::Done {
                finish: "length",
                matched_stop: None,
                usage: usage(5, 10),
            },
        ]);
        let order = names(&frames);
        let stop = order.iter().position(|n| *n == "content_block_stop");
        let delta = order.iter().position(|n| *n == "message_delta");
        assert!(stop < delta, "the block closes first: {order:?}");
    }

    /// A failure mid-stream ends in an `error` event carrying the
    /// Anthropic error type for its status, so a client can tell a bad
    /// request from an overloaded server instead of waiting out an idle
    /// timeout on a stream that simply stopped.
    #[test]
    fn a_mid_stream_failure_closes_the_open_block_and_emits_an_error_event() {
        let frames = run(vec![
            GenEvent::Content("partial".into()),
            GenEvent::Failed {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: "kv pool exhausted".into(),
            },
        ]);
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "error",
            ]
        );
        assert_eq!(
            only(&frames, "error")[0]["error"],
            json!({"type": "overloaded_error", "message": "kv pool exhausted"})
        );
    }

    /// A keepalive is a protocol-native `ping`, never an SSE comment: a
    /// comment does not reach a client's event handler, so an idle but
    /// healthy stream still looks dead to whatever is timing it.
    #[test]
    fn a_keepalive_is_a_protocol_native_ping() {
        let frames = run(vec![GenEvent::Keepalive]);
        assert_eq!(names(&frames), vec!["message_start", "ping"]);
        assert_eq!(only(&frames, "ping")[0], json!({"type": "ping"}));
    }

    /// An empty delta produces no frame at all -- a client that renders
    /// every delta would otherwise show nothing, repeatedly.
    #[test]
    fn an_empty_delta_produces_no_frame() {
        let frames = run(vec![
            GenEvent::Content(String::new()),
            GenEvent::Reasoning(String::new()),
        ]);
        assert_eq!(names(&frames), vec!["message_start"]);
    }

    /// A stray arguments fragment with no tool block open is dropped
    /// rather than attached to a text block, where a user would be shown
    /// raw JSON.
    #[test]
    fn a_stray_arguments_fragment_is_dropped() {
        let frames = run(vec![GenEvent::CallArguments {
            index: 0,
            fragment: r#"{"a":1}"#.into(),
        }]);
        assert_eq!(names(&frames), vec!["message_start"]);
    }

    /// A frame serializes as a named SSE event, which is how an
    /// Anthropic client dispatches: `event: message_stop` with the
    /// matching `type` in the body.
    #[test]
    fn a_frame_serializes_as_a_named_sse_event() {
        let frames = run(vec![GenEvent::Keepalive]);
        let encoded = format!("{:?}", frames[1].clone().into_event());
        assert!(encoded.contains("ping"), "{encoded}");
        assert_eq!(frames[1].data["type"], json!("ping"));
    }

    // -----------------------------------------------------------------
    // The buffered response
    // -----------------------------------------------------------------

    /// The buffered body's block order: thinking, then text, then every
    /// call -- and the text block is present even when it is empty,
    /// because `content` is a list a client indexes.
    #[test]
    fn a_buffered_answer_orders_thinking_then_text_then_calls() {
        let parsed = ParsedOutput {
            reasoning: Some("thought".into()),
            content: String::new(),
            calls: vec![crate::output::ParsedToolCall {
                name: "read".into(),
                arguments: r#"{"path":"a"}"#.into(),
            }],
        };
        let body = message_body(parsed, "stop", None, &usage(11, 3), "msg_1", "m");
        let kinds: Vec<&str> = body["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, vec!["thinking", "text", "tool_use"]);
        assert_eq!(body["content"][0]["signature"], json!(""));
        assert_eq!(body["content"][2]["input"], json!({"path": "a"}));
        assert_eq!(body["stop_reason"], json!("tool_use"));
        assert_eq!(
            body["usage"],
            json!({"input_tokens": 11, "output_tokens": 3})
        );
    }

    /// Arguments that are not a JSON object become `{}` rather than a
    /// string: `input` is typed as an object and a client will index it.
    #[test]
    fn unparseable_tool_arguments_become_an_empty_input_object() {
        let parsed = ParsedOutput {
            reasoning: None,
            content: String::new(),
            calls: vec![crate::output::ParsedToolCall {
                name: "t".into(),
                arguments: "not json".into(),
            }],
        };
        let body = message_body(parsed, "stop", None, &usage(1, 1), "msg_1", "m");
        assert_eq!(body["content"][1]["input"], json!({}));
    }

    // -----------------------------------------------------------------
    // Rule 7: count_tokens
    // -----------------------------------------------------------------

    /// **Fails if `count_tokens` grows its own converter.** The number
    /// it answers has to be the `usage.input_tokens` the following
    /// generation reports, and it can only be that if both build the
    /// same messages, offer the template the same tools and render with
    /// the same kwargs -- so the same body must convert identically on
    /// both endpoints.
    #[test]
    fn count_tokens_converts_a_body_exactly_as_a_generation_of_it_would() {
        let body = json!({
            "system": "be terse",
            "tools": tools_fixture(),
            "tool_choice": {"type": "tool", "name": "write"},
            "thinking": {"type": "enabled"},
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
                    {"type": "text", "text": "and now?"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "tool_use", "id": "t2", "name": "read", "input": {"path": "a"}},
                ]},
            ],
        });
        let mut generation = body.clone();
        generation["model"] = json!("m");
        generation["max_tokens"] = json!(64);
        let generated = converted(generation);

        let mut counted_body = body;
        counted_body["model"] = json!("m");
        let counted = countable_prompt(&count_request(counted_body)).expect("converts");

        assert_eq!(
            shape(&counted.chat.messages),
            shape(&generated.chat.messages)
        );
        assert_eq!(
            counted
                .chat
                .tools
                .iter()
                .map(|t| t.function.name.clone())
                .collect::<Vec<_>>(),
            generated
                .chat
                .tools
                .iter()
                .map(|t| t.function.name.clone())
                .collect::<Vec<_>>(),
            "the template is offered the same tools"
        );
        assert_eq!(
            counted.chat.thinking.map(|t| t.kind),
            generated.chat.thinking.map(|t| t.kind),
            "and renders with the same thinking direction"
        );
    }

    /// An empty `messages` array is a malformed request.
    #[test]
    fn count_tokens_refuses_an_empty_messages_array() {
        let req = count_request(json!({"model": "m", "messages": []}));
        let Err((status, Json(body))) = countable_prompt(&req) else {
            panic!("an empty messages array must be refused");
        };
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["type"], json!("error"));
        assert!(body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("at least one message")));
    }

    /// **Fails if "nothing survived conversion" is answered as "no
    /// messages".** The caller *did* send a message; it was an
    /// image-only turn this text-only server dropped. Telling them "at
    /// least one message is required" is visibly false to them and
    /// points at the wrong fix, so the two refusals stay distinct.
    #[test]
    fn count_tokens_distinguishes_no_messages_from_no_tokenizable_content() {
        let req = count_request(json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ]}],
        }));
        let Err((status, Json(body))) = countable_prompt(&req) else {
            panic!("an image-only conversation must be refused");
        };
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body["error"]["message"],
            json!("messages: no tokenizable content")
        );
    }

    /// A body with content survives both checks and reaches the
    /// tokenizer.
    #[test]
    fn count_tokens_accepts_a_body_with_tokenizable_content() {
        let req = count_request(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        }));
        let prepared = countable_prompt(&req).expect("this one counts");
        assert_eq!(text_of(&prepared.chat.messages[0]), "hello");
    }

    // -----------------------------------------------------------------
    // Errors
    // -----------------------------------------------------------------

    /// A shared helper's OpenAI-shaped error is re-dressed in the
    /// Anthropic envelope with its status and message intact -- a Claude
    /// client cannot read the other shape and reports it as an
    /// unexpected response.
    #[test]
    fn a_shared_error_is_re_dressed_in_the_anthropic_envelope() {
        let (status, Json(body)) = anthropic_shape(crate::invalid_request("bad thing", "field"));
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body,
            json!({
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad thing"},
            })
        );
    }

    /// Every status this surface answers maps onto Anthropic's closed
    /// error vocabulary, and an unmapped one is `api_error` rather than
    /// an invented name a client would fail to match.
    #[test]
    fn every_status_maps_onto_the_anthropic_error_vocabulary() {
        assert_eq!(error_type(StatusCode::BAD_REQUEST), "invalid_request_error");
        assert_eq!(
            error_type(StatusCode::SERVICE_UNAVAILABLE),
            "overloaded_error"
        );
        assert_eq!(
            error_type(StatusCode::TOO_MANY_REQUESTS),
            "rate_limit_error"
        );
        assert_eq!(error_type(StatusCode::INTERNAL_SERVER_ERROR), "api_error");
        assert_eq!(error_type(StatusCode::IM_A_TEAPOT), "api_error");
    }

    /// A body that is JSON but not a Messages request is refused in the
    /// Anthropic envelope too, which is the one 400 a client hits while
    /// its own request builder is wrong.
    #[test]
    fn a_malformed_body_is_refused_in_the_anthropic_envelope() {
        let err = serde_json::from_value::<MessagesRequest>(json!({"model": "m"}))
            .expect_err("max_tokens and messages are required");
        let (status, Json(body)) = body_error(err);
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["type"], json!("error"));
        assert_eq!(body["error"]["type"], json!("invalid_request_error"));
    }

    // -----------------------------------------------------------------
    // stop_reason / stop_sequence
    // -----------------------------------------------------------------

    /// The whole point of carrying the matched string: an agent that put
    /// up its own fence needs to tell "I hit my fence" from "the model
    /// finished". Anthropic says that with `stop_reason:
    /// "stop_sequence"` and the string beside it, and the two are only
    /// ever reported together -- a `stop_sequence` reason with a null
    /// string is a fabrication to branch on.
    #[test]
    fn a_callers_own_stop_string_is_reported_as_a_stop_sequence_with_the_string() {
        let parsed = ParsedOutput {
            reasoning: None,
            content: "up to here".into(),
            calls: Vec::new(),
        };
        let body = message_body(parsed, "stop", Some("END"), &usage(4, 2), "msg_1", "m");
        assert_eq!(body["stop_reason"], "stop_sequence");
        assert_eq!(body["stop_sequence"], "END");
    }

    /// The model ending its own turn is `end_turn` with a null
    /// sequence, unchanged. A stop the SERVER added -- the served
    /// template's end-of-turn marker -- arrives here as `None` (see
    /// `caller_stop`) and lands in exactly this branch, because telling
    /// an agent it hit a fence it never put up is worse than telling it
    /// nothing.
    #[test]
    fn a_turn_the_model_ended_itself_stays_end_turn_with_no_sequence() {
        let parsed = ParsedOutput {
            reasoning: None,
            content: "done".into(),
            calls: Vec::new(),
        };
        let body = message_body(parsed, "stop", None, &usage(4, 2), "msg_1", "m");
        assert_eq!(body["stop_reason"], "end_turn");
        assert!(body["stop_sequence"].is_null());
    }

    /// Precedence, and it is not arbitrary. Truncation outranks
    /// everything: a client must not execute a call whose arguments may
    /// have been cut off mid-write, and must not be told the answer
    /// ended on a fence when it ended on the budget. Calls outrank a
    /// stop for the same reason the OpenAI surface reports
    /// `tool_calls`: what the client does next is run the call.
    #[test]
    fn truncation_outranks_a_stop_string_and_calls_outrank_it_too() {
        let call = || crate::output::ParsedToolCall {
            name: "read".into(),
            arguments: "{}".into(),
        };

        let truncated = ParsedOutput {
            reasoning: None,
            content: "half".into(),
            calls: vec![call()],
        };
        let body = message_body(truncated, "length", Some("END"), &usage(4, 2), "i", "m");
        assert_eq!(body["stop_reason"], "max_tokens");
        assert!(body["stop_sequence"].is_null());

        let with_call = ParsedOutput {
            reasoning: None,
            content: String::new(),
            calls: vec![call()],
        };
        let body = message_body(with_call, "stop", Some("END"), &usage(4, 2), "i", "m");
        assert_eq!(body["stop_reason"], "tool_use");
        assert!(body["stop_sequence"].is_null());
    }

    /// The stream says the same thing as the buffered body, which is
    /// the invariant that keeps a client from having to implement two
    /// readings of one generation. `stop_sequence` appears in the
    /// terminal `message_delta` ONLY beside its reason -- a null on
    /// every other ending is noise the client has to ignore.
    #[test]
    fn the_terminal_delta_carries_the_stop_sequence_only_when_there_is_one() {
        let frames = run(vec![GenEvent::Done {
            finish: "stop",
            matched_stop: Some("END".to_string()),
            usage: usage(4, 2),
        }]);
        let delta = &only(&frames, "message_delta")[0];
        assert_eq!(delta["delta"]["stop_reason"], "stop_sequence");
        assert_eq!(delta["delta"]["stop_sequence"], "END");

        let frames = run(vec![GenEvent::Done {
            finish: "stop",
            matched_stop: None,
            usage: usage(4, 2),
        }]);
        let delta = &only(&frames, "message_delta")[0];
        assert_eq!(delta["delta"]["stop_reason"], "end_turn");
        assert!(delta["delta"].get("stop_sequence").is_none());
    }

    /// A stop the client never sent must not be reported as one, and
    /// the served template really does add its own -- a family's
    /// end-of-turn marker goes into `params.stop` beside the caller's
    /// list. This is the filter that keeps the two apart.
    #[test]
    fn only_a_stop_the_client_asked_for_can_become_a_stop_sequence() {
        let caller = vec!["END".to_string()];
        assert_eq!(
            caller_stop(
                &crate::generate::FinishReason::StopSequence("END".into()),
                &caller
            ),
            Some("END".to_string())
        );
        assert_eq!(
            caller_stop(
                &crate::generate::FinishReason::StopSequence("<|im_end|>".into()),
                &caller
            ),
            None,
            "a template's own marker is not the caller's fence"
        );
        assert_eq!(
            caller_stop(&crate::generate::FinishReason::Stop, &caller),
            None
        );
    }
}
