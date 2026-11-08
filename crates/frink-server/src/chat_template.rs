//! The served checkpoint's own chat template, evaluated.
//!
//! # What this replaced
//!
//! This module used to *sniff* `tokenizer.chat_template` for literal
//! markers -- `<|im_start|>`, `<|start_header_id|>`, `<start_of_turn>`
//! -- and pick one of six hand-written renderers. That was a disclosed
//! scope decision when there was no Jinja evaluator in the workspace.
//! There is one now (`frink_models::chat_template`, which compiles the
//! checkpoint's real template with minijinja), and it had no caller, so
//! the server was still serving Mistral-Instruct `user: hi` because
//! `[INST] … [/INST]` matches none of those markers. The sniffer's three
//! silent failure modes are written up in that module's doc comment; all
//! three are gone here by construction, because the template is now
//! evaluated rather than recognised.
//!
//! # What this adds on top of it
//!
//! Three things the evaluator deliberately leaves to its caller:
//!
//! 1. **Message shape.** [`PromptTemplate::render`] converts the
//!    server's `ChatMessage` into the OpenAI-shaped JSON a real template
//!    reads (`message.role`, `message.content`, `message.tool_calls`,
//!    `message.tool_call_id`, and a replayed chain of thought under all
//!    three spellings families use for it).
//! 2. **Who describes the tools.** A template that really consumes
//!    `tools` -- established by rendering with and without one, not by
//!    looking for the word -- gets them as structured JSON and owns the
//!    whole tool grammar. A template that does not gets the text
//!    preamble this server has always used (`tool_preamble` in
//!    `lib.rs`), and replayed calls are folded back into `content` as
//!    the same `<tool_call>{…}</tool_call>` marker text the preamble
//!    asks for -- so a conversation looks consistent to the model on
//!    every turn regardless of which half of that split it lands in.
//! 3. **The effort vocabulary.** Probed once at load
//!    ([`crate::policy::effort::probe_effort_profile`]) rather than per request,
//!    because it costs ~30 renders and never changes for a checkpoint.
//! 4. **The end-of-turn marker.** Derived by rendering a conversation
//!    whose assistant turn is closed and diffing it against one that is
//!    not ([`crate::policy::turn::probe_end_of_turn`]). Gemma IT ends a
//!    turn with `<end_of_turn>` and Yi-1.5 with `<|im_end|>`, neither of
//!    which is that checkpoint's EOS, so both need the string in their
//!    stop set or the model talks past the end of its own answer.
//!
//! All four probes run once, at load, which is what lets them be renders
//! rather than guesses.
//!
//! # Nothing here matches template source any more
//!
//! The last substring check was the end-of-turn marker, which had an arm
//! per family -- `<start_of_turn>`, `<|im_end|>`, `<|turn>` -- and so
//! reported nothing for a family nobody had yet written an arm for. It
//! is now derived from the render like the other three, which is both
//! the same mechanism and the same reason: what a template PRINTS is the
//! question, and its source is a proxy that answers a different one.

use std::sync::Arc;

use crate::policy::effort::{
    derive_think_gears, probe_effort_profile, probe_thinking_profile, EffortProfile, ThinkGears,
};
use crate::policy::turn::probe_end_of_turn;
use frink_models::chat_template::{
    BuiltinTemplate, ChatTemplate as JinjaTemplate, RenderOptions, TemplateError,
};
use serde_json::{json, Map, Value};

use crate::{ChatMessage, ToolDef};

/// Everything the server needs to turn a conversation into a prompt.
///
/// Cheap to clone: every `*Loaded` struct carries one and hands it to
/// each request.
#[derive(Clone)]
pub(crate) struct PromptTemplate {
    inner: Arc<Inner>,
}

struct Inner {
    template: JinjaTemplate,
    end_of_turn: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    thinking: crate::policy::effort::ThinkingProfile,
    handles_tools: bool,
    /// The reasoning format the template's own text implies: `Think`
    /// when a thinking-on render of it contains `<think>`. Learned at
    /// load, like the gears, because the served NAME is not a family:
    /// Bonsai-2-27B's `general.name` is `Hf`, every Qwen3.5 export's is
    /// `Original`, and `--alias` can make it anything, so a parser
    /// picked from the name alone left `<think>` prose in `content`
    /// with `reasoning_content` empty on exactly those checkpoints.
    implied_reasoning: Option<crate::policy::parser::ReasoningFormat>,
    default_kwargs: Map<String, Value>,
    implied_tools: Option<crate::policy::parser::ToolCallFormat>,
    /// Whether llama.cpp calls the template thinking-capable, which is
    /// `enable_thinking`'s value in every render unless the request or a
    /// flag sets it ([`JinjaTemplate::supports_thinking`]). Probed once,
    /// at load, as llama-server does.
    supports_thinking: bool,
}

impl std::fmt::Debug for PromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PromptTemplate({})", self.inner.template.describe())
    }
}

/// What `--chat-template` and `--chat-template-file` lower to: the
/// template's text.
pub(crate) const TEMPLATE_ENV: &str = "FRINK_CHAT_TEMPLATE";
/// What `--chat-template-kwargs`, `-rea` and `--reasoning-effort` lower
/// to: their merged JSON object.
pub(crate) const TEMPLATE_KWARGS_ENV: &str = "FRINK_CHAT_TEMPLATE_KWARGS";

impl PromptTemplate {
    /// The template a GGUF is served with: the operator's, or whatever
    /// `tokenizer.chat_template` holds, or a builtin when the checkpoint
    /// ships none at all (llama.cpp `--jinja` defaults to ChatML the same
    /// way).
    ///
    /// These three constructors are the only ones outside tests, so a
    /// load path cannot miss the flags.
    pub(crate) fn for_gguf(
        file: &frink_gguf::ShardedGguf,
        arch: Option<&str>,
        byte_tokenizer: bool,
    ) -> anyhow::Result<Self> {
        Self::served(
            JinjaTemplate::from_gguf_metadata(
                file.metadata_str("tokenizer.chat_template"),
                arch,
                byte_tokenizer,
                JinjaTemplate::vocab_has_chatml(file),
            ),
            file.token_text("tokenizer.ggml.bos_token_id"),
            file.token_text("tokenizer.ggml.eos_token_id"),
        )
    }

    /// For checkpoints that carry their template outside GGUF metadata:
    /// Kimi K3 keeps it as a top-level string in `tokenizer_config.json`,
    /// the real HuggingFace convention. No vocabulary here: this path
    /// always carries its own template, so the ChatML fallback is
    /// unreachable.
    pub(crate) fn for_tokenizer_config(source: Option<&str>) -> anyhow::Result<Self> {
        Self::served(
            JinjaTemplate::from_gguf_metadata(source, None, false, false),
            None,
            None,
        )
    }

    /// Role-labeled lines, no special tokens: the synthetic-weights demo
    /// path, where there is no real vocabulary for markers to live in.
    pub(crate) fn for_synthetic_weights() -> anyhow::Result<Self> {
        Self::served(JinjaTemplate::builtin(BuiltinTemplate::Plain), None, None)
    }

    fn served(
        native: JinjaTemplate,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> anyhow::Result<Self> {
        let flags = operator_flags()?;
        Ok(Self::build(
            flags.template.unwrap_or(native),
            bos_token,
            eos_token,
            flags.kwargs,
        ))
    }

    #[cfg(test)]
    pub(crate) fn from_gguf_metadata(
        source: Option<&str>,
        arch: Option<&str>,
        byte_tokenizer: bool,
        chatml_tokens_present: bool,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        Self::build(
            JinjaTemplate::from_gguf_metadata(source, arch, byte_tokenizer, chatml_tokens_present),
            bos_token,
            eos_token,
            Map::new(),
        )
    }

    /// A template from text, as `for_tokenizer_config` has one, with the
    /// special tokens a test needs.
    #[cfg(test)]
    pub(crate) fn from_source(
        source: Option<&str>,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        Self::build(
            JinjaTemplate::from_gguf_metadata(source, None, false, false),
            bos_token,
            eos_token,
            Map::new(),
        )
    }

    #[cfg(test)]
    pub(crate) fn plain() -> Self {
        Self::build(
            JinjaTemplate::builtin(BuiltinTemplate::Plain),
            None,
            None,
            Map::new(),
        )
    }

    /// A template with these server-wide kwargs, as flags would set them.
    #[cfg(test)]
    pub(crate) fn with_default_kwargs(source: &str, default_kwargs: Map<String, Value>) -> Self {
        let template = JinjaTemplate::from_gguf_metadata(Some(source), None, false, true);
        Self::build(template, None, None, default_kwargs)
    }

    /// A GGUF's template with its tokens and the server-wide kwargs, as
    /// [`Self::for_gguf`] builds it, without reading the environment.
    #[cfg(test)]
    pub(crate) fn as_served(
        source: &str,
        bos_token: &str,
        eos_token: &str,
        default_kwargs: Map<String, Value>,
    ) -> Self {
        Self::build(
            JinjaTemplate::from_gguf_metadata(Some(source), None, false, false),
            Some(bos_token.to_string()),
            Some(eos_token.to_string()),
            default_kwargs,
        )
    }

    fn build(
        template: JinjaTemplate,
        bos_token: Option<String>,
        eos_token: Option<String>,
        default_kwargs: Map<String, Value>,
    ) -> Self {
        let (bos, eos) = (bos_token.as_deref(), eos_token.as_deref());
        // First, because the probes below render as a request would, and
        // a request's render carries its answer.
        let supports_thinking = template.supports_thinking(bos, eos);
        let thinking = probe_thinking_profile(
            probe_render(&template, bos, eos, supports_thinking),
            probe_efforts(&template, bos, eos, supports_thinking),
        );
        let handles_tools = probe_tools_consumed(&template, bos, eos);
        let implied_reasoning =
            probe_implied_reasoning(probe_render(&template, bos, eos, supports_thinking));
        let implied_tools = probe_implied_tool_format(&template, bos, eos);
        Self {
            inner: Arc::new(Inner {
                // A terminator that IS the EOS adds nothing to the stop
                // set: decoding already stops on that token, and
                // repeating it as a string only makes the withhold
                // buffer hold text it never has to.
                end_of_turn: probe_end_of_turn(turn_render(&template, bos, eos), bos)
                    .filter(|marker| Some(marker.as_str()) != eos),
                template,
                bos_token,
                eos_token,
                thinking,
                handles_tools,
                implied_reasoning,
                implied_tools,
                default_kwargs,
                supports_thinking,
            }),
        }
    }

    /// `--chat-template-kwargs` and friends: what every render starts
    /// from, under anything the request says.
    pub(crate) fn default_kwargs(&self) -> &Map<String, Value> {
        &self.inner.default_kwargs
    }

    /// The reasoning parser for a checkpoint served under `served_model`:
    /// the name's family when it has one
    /// ([`crate::policy::parser::ReasoningFormat::infer`]), else what the
    /// template's own text implies. ONE accessor, because the split, the
    /// token count, the budget and the continuation writer all have to
    /// agree about which family this is, and each used to ask the name
    /// on its own.
    pub(crate) fn reasoning_format(
        &self,
        served_model: &str,
    ) -> Option<crate::policy::parser::ReasoningFormat> {
        crate::policy::parser::ReasoningFormat::infer(served_model).or(self.inner.implied_reasoning)
    }

    /// The tool-call grammar for a checkpoint served under
    /// `served_model`: what the TEMPLATE asks for when it asks for
    /// anything, else the name's family.
    ///
    /// Template first, which is the opposite of `reasoning_format`'s
    /// order and deliberate: a served name comes from `--alias` and can
    /// be anything, while a template that prints `<function=` has
    /// stated the grammar it will emit.
    pub(crate) fn tool_call_format(
        &self,
        served_model: &str,
    ) -> crate::policy::parser::ToolCallFormat {
        self.inner
            .implied_tools
            .unwrap_or_else(|| crate::policy::parser::ToolCallFormat::infer(served_model))
    }

    /// Short human-readable identity, for the load-time log line.
    pub(crate) fn describe(&self) -> String {
        self.inner.template.describe()
    }

    /// The end-of-turn string this family emits before EOS, if it has
    /// one worth adding to the stop set.
    pub(crate) fn end_of_turn(&self) -> Option<&str> {
        self.inner.end_of_turn.as_deref()
    }

    /// Whether the template consumes `tools` itself. When it does not,
    /// the caller owes the model a text preamble describing them.
    pub(crate) fn handles_tools(&self) -> bool {
        self.inner.handles_tools
    }

    /// The effort vocabulary this checkpoint's template actually grades,
    /// probed at load.
    pub(crate) fn efforts(&self) -> &EffortProfile {
        &self.inner.thinking.efforts
    }

    /// The thinking controls to advertise on `/v1/models`, so a client
    /// picks a gear instead of guessing one.
    ///
    /// `parser_configured` covers the one case the template cannot
    /// speak for: an always-thinking family whose template has no
    /// observable knob, but whose output really is being split into
    /// `reasoning_content`. Such a checkpoint advertises a single `on`
    /// gear with no kwargs -- there is nothing to send, and the point is
    /// only to stop a reasoning model looking like one with no gears.
    pub(crate) fn think_gears(&self, parser_configured: bool) -> ThinkGears {
        derive_think_gears(&self.inner.thinking, parser_configured)
    }

    /// Render a conversation into a prompt.
    ///
    /// `extra` is the request's `chat_template_kwargs` -- already
    /// sanitized by the caller, since every render path has to quantize
    /// effort identically or a request validates against one prompt and
    /// generates from another.
    pub(crate) fn render(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        extra: Map<String, Value>,
    ) -> Result<String, TemplateError> {
        let structured = self.handles_tools() && !tools.is_empty();
        let json: Vec<Value> = messages
            .iter()
            .map(|m| message_json(m, structured))
            .collect();
        let opts = RenderOptions {
            add_generation_prompt: true,
            bos_token: self.inner.bos_token.clone(),
            eos_token: self.inner.eos_token.clone(),
            tools: if structured {
                tools.iter().map(tool_json).collect()
            } else {
                Vec::new()
            },
            extra,
            enable_thinking: Some(self.inner.supports_thinking),
        };
        self.inner.template.render(&json, &opts)
    }
}

/// What the operator's flags say, read from the variables they lower to
/// so that setting those directly is held to the same checks.
fn operator_flags() -> anyhow::Result<frink_models::template_flags::TemplateFlags> {
    let template = std::env::var(TEMPLATE_ENV).ok();
    let kwargs: Vec<String> = std::env::var(TEMPLATE_KWARGS_ENV)
        .ok()
        .into_iter()
        .collect();
    frink_models::template_flags::TemplateFlags::resolve(
        frink_models::template_flags::TemplateFlagValues {
            template: template.as_deref(),
            kwargs: &kwargs,
            ..Default::default()
        },
    )
    .map_err(anyhow::Error::msg)
}

/// One OpenAI-shaped message, as a template reads it.
///
/// `structured` says whether the template is going to iterate
/// `message.tool_calls` itself. When it is not, a replayed assistant
/// turn's calls are folded back into `content` as the marker text the
/// text preamble asked the model to produce -- otherwise a multi-turn
/// tool conversation would show the model its own past calls in a shape
/// it was never asked for.
fn message_json(m: &ChatMessage, structured: bool) -> Value {
    let mut obj = Map::new();
    obj.insert("role".into(), json!(m.role));
    let text = m
        .content
        .as_ref()
        .map(crate::MessageContent::as_text)
        .unwrap_or_default();
    // Keys in llama.cpp's order (`common_chat_msg::to_json_oaicompat`):
    // role, content, the reasoning, `tool_call_id`, `tool_calls`. A
    // template that prints a message, or walks its items, shows it.
    let tool_calls = match (&m.tool_calls, structured) {
        (Some(calls), true) => {
            obj.insert("content".into(), json!(text));
            Some(Value::Array(calls.iter().map(tool_call_json).collect()))
        }
        (Some(_), false) => {
            obj.insert("content".into(), json!(m.rendered_content()));
            None
        }
        (None, _) => {
            obj.insert("content".into(), json!(text));
            None
        }
    };
    // Under every spelling, because templates disagree and a template
    // reads a key it does not know as undefined rather than erroring:
    // the DeepSeek/GLM family iterates `message.reasoning_content`, the
    // Qwen lineage `message.reasoning`, harmony `message.thinking`.
    // Emitting all three costs two keys and means a replayed chain of
    // thought reaches whichever one the served checkpoint was written
    // against, instead of being silently dropped by the others.
    //
    // Empty is not the same as absent and is skipped: a template that
    // tests `if message.reasoning_content` would otherwise open the
    // family's thinking markers around nothing.
    if let Some(reasoning) = m.reasoning_content.as_deref().filter(|r| !r.is_empty()) {
        obj.insert("reasoning_content".into(), json!(reasoning));
        obj.insert("reasoning".into(), json!(reasoning));
        // `thinking` has ONE exception, and it is a raise rather than a
        // wrong render: gpt-oss's harmony template rejects an assistant
        // turn that makes tool calls while carrying both visible text
        // and thinking, because harmony puts those on channels that
        // cannot both be final. Visible text wins -- it is what the
        // user was actually shown -- and the other two spellings still
        // carry the reasoning for every family that reads them.
        let harmony_would_raise = m.tool_calls.is_some() && !text.is_empty();
        if !harmony_would_raise {
            obj.insert("thinking".into(), json!(reasoning));
        }
    }
    if let Some(id) = &m.tool_call_id {
        obj.insert("tool_call_id".into(), json!(id));
    }
    if let Some(calls) = tool_calls {
        obj.insert("tool_calls".into(), calls);
    }
    Value::Object(obj)
}

/// A replayed tool call.
///
/// `arguments` arrives on the wire as a JSON *string*, and templates
/// disagree about what they expect: HuggingFace's own
/// `apply_chat_template` hands the model a dict, so a template that
/// writes `{{ call.function.arguments | tojson }}` would emit a
/// double-encoded string if the wire form were passed through. Parsing
/// it back to an object when it is one gives every template the shape it
/// was written against; a value that is not a JSON object stays a
/// string, because that is all that can honestly be said about it.
fn tool_call_json(call: &crate::ToolCallIn) -> Value {
    let raw = call.function.arguments.as_str();
    let args = match serde_json::from_str::<Value>(raw) {
        Ok(v @ Value::Object(_)) => v,
        _ => json!(raw),
    };
    // llama.cpp's order (`common_chat_msg::to_json_oaicompat`), which a
    // template printing the call shows: the id last, and only when there
    // is one.
    let mut out = json!({
        "type": "function",
        "function": {"name": call.function.name, "arguments": args},
    });
    if !call.id.is_empty() {
        out["id"] = json!(call.id);
    }
    out
}

pub(crate) fn tool_json(t: &ToolDef) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.function.name,
            "description": t.function.description.as_deref().unwrap_or(""),
            "parameters": t.function.parameters.clone().unwrap_or_else(|| json!({
                "type": "object",
                "properties": {},
            })),
        },
    })
}

/// Does this template actually *consume* the tools it is handed?
///
/// `frink_models`'s own `handles_tools` is a substring check on the
/// template source, and its doc comment says as much -- llama.cpp
/// probes by rendering and keeps the source check only because it is
/// cheap enough to run per call. This runs once, at load, so it can
/// afford the real answer: render the same conversation with and
/// without a tool and see whether the template did anything with it.
///
/// The difference is not cosmetic. A template that merely *mentions*
/// the word (in a comment, or in prose it prints) would claim to handle
/// tools, the text preamble would be skipped on that basis, and the
/// model would be offered tools nobody ever described to it -- a
/// request that silently cannot call anything. A template that errors
/// when handed tools has not handled them either.
fn probe_tools_consumed(
    template: &JinjaTemplate,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> bool {
    let messages = vec![json!({"role": "user", "content": "probe"})];
    let render = |tools: Vec<Value>| {
        template.render(
            &messages,
            &RenderOptions {
                add_generation_prompt: true,
                bos_token: bos_token.map(str::to_string),
                eos_token: eos_token.map(str::to_string),
                tools,
                extra: Map::new(),
                enable_thinking: None,
            },
        )
    };
    let probe = json!({
        "type": "function",
        "function": {
            "name": "frink_probe_tool",
            "description": "No-op probe tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    });
    match (render(Vec::new()), render(vec![probe])) {
        // It did something with them, or it did not.
        (Ok(without), Ok(with)) => with != without,
        // Handed tools, it failed: it cannot express them.
        (_, Err(_)) => false,
        // It renders *only* when tools are present, which is as strong
        // a statement of dependence as a template can make.
        (Err(_), Ok(_)) => true,
    }
}

/// Which tool-call grammar the TEMPLATE itself asks for, rendered with
/// a tool in hand, or `None` when it names none.
///
/// The served name is a guess and the template is a statement. A
/// checkpoint served under `--alias bonsai-2-27b` is a Qwen3.5 file
/// whose template prints, verbatim, `<tool_call>\n<function=…>\n
/// <parameter=…>`; `ToolCallFormat::infer` sees a name with no "qwen"
/// in it and falls through to the Llama 3 fallback, which reads none
/// of those calls. Every call the model made then reached the client
/// as raw markup in `content` with `tool_calls` null, which is exactly
/// the failure that file's own comment records for Gemma 4.
///
/// Order is load-bearing for the same reason it is in `infer`: the
/// specific families have to be tested before the general ones they
/// look like. `<function=` is Qwen3-Coder's and appears INSIDE a
/// `<tool_call>`, so plain `<tool_call>` means Hermes JSON only once
/// that has been ruled out.
fn probe_implied_tool_format(
    template: &JinjaTemplate,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Option<crate::policy::parser::ToolCallFormat> {
    use crate::policy::parser::ToolCallFormat as F;
    let messages = vec![json!({"role": "user", "content": "probe"})];
    let probe = json!({
        "type": "function",
        "function": {
            "name": "frink_probe_tool",
            "description": "No-op probe tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    });
    let text = template
        .render(
            &messages,
            &RenderOptions {
                add_generation_prompt: true,
                bos_token: bos_token.map(str::to_string),
                eos_token: eos_token.map(str::to_string),
                tools: vec![probe],
                extra: Map::new(),
                enable_thinking: None,
            },
        )
        .ok()?;
    let has = |needle: &str| text.contains(needle);
    let format = if has("]<]minimax[>[") {
        F::MiniMaxM3
    } else if has("<minimax:tool_call>") {
        F::MiniMax
    } else if has("<arg_key>") {
        F::Glm47
    } else if has("<function=") && has("<parameter=") {
        F::Qwen3Coder
    } else if has("[TOOL_CALLS]") {
        F::Mistral
    } else if has("<|python_tag|>") {
        F::Llama3
    } else if has("<tool_call>") {
        F::Qwen25
    } else {
        return None;
    };
    Some(format)
}

/// One render closure over a fixed probe conversation.
///
/// Both probes take the same shape -- vary one thing, render, compare
/// -- so they share the conversation as well as the closure. Fixed, so
/// the only thing that differs between two renders is the thing the
/// probe varied.
fn probe_render<'a>(
    template: &'a JinjaTemplate,
    bos_token: Option<&'a str>,
    eos_token: Option<&'a str>,
    supports_thinking: bool,
) -> impl FnMut(&Map<String, Value>, Option<&[Value]>) -> Result<String, TemplateError> + 'a {
    let messages = vec![json!({"role": "user", "content": "probe"})];
    move |kwargs, tools| {
        let opts = RenderOptions {
            add_generation_prompt: true,
            bos_token: bos_token.map(str::to_string),
            eos_token: eos_token.map(str::to_string),
            tools: tools.map(<[Value]>::to_vec).unwrap_or_default(),
            extra: kwargs.clone(),
            enable_thinking: Some(supports_thinking),
        };
        template.render(&messages, &opts)
    }
}

/// What the template's markers say about its chain of thought: a render
/// with thinking on that contains `<think>` is the plain think block
/// ([`crate::policy::parser::ReasoningFormat::Think`]). A render that
/// fails, or shows no marker, implies nothing; the channel formats
/// (harmony, Gemma-4, MiniMax-M3) are not inferred here because their
/// names carry them and their markers are not a `<think>` pair.
fn probe_implied_reasoning(
    mut render: impl FnMut(&Map<String, Value>, Option<&[Value]>) -> Result<String, TemplateError>,
) -> Option<crate::policy::parser::ReasoningFormat> {
    let on = render(&crate::policy::effort::thinking_on_kwargs(), None).ok()?;
    on.contains("<think>")
        .then_some(crate::policy::parser::ReasoningFormat::Think)
}

/// Learn the effort vocabulary by rendering probes through the template.
///
/// A template that rejects the probe shape entirely is reported inert
/// by [`crate::policy::effort::probe_effort_profile`], which is the safe answer:
/// send no effort at all.
fn probe_efforts(
    template: &JinjaTemplate,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    supports_thinking: bool,
) -> EffortProfile {
    probe_effort_profile(probe_render(
        template,
        bos_token,
        eos_token,
        supports_thinking,
    ))
}

/// A render closure that varies the CONVERSATION, for
/// [`probe_end_of_turn`].
///
/// The other two probes hold the conversation fixed and vary a kwarg or
/// the tool list; this one is the other way round, so it takes messages
/// instead. `add_generation_prompt` is always false: the probe is asking
/// what CLOSES an assistant turn, and a generation prompt opens one.
pub(crate) fn turn_render<'a>(
    template: &'a JinjaTemplate,
    bos_token: Option<&'a str>,
    eos_token: Option<&'a str>,
) -> impl FnMut(&[Value]) -> Result<String, TemplateError> + 'a {
    move |messages: &[Value]| {
        template.render(
            messages,
            &RenderOptions {
                add_generation_prompt: false,
                bos_token: bos_token.map(str::to_string),
                eos_token: eos_token.map(str::to_string),
                tools: Vec::new(),
                extra: Map::new(),
                enable_thinking: None,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MessageContent;

    fn tool_def(name: &str) -> ToolDef {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": "a tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }))
        .expect("tool def")
    }

    fn tool_call_in(name: &str, arguments: &str) -> crate::ToolCallIn {
        serde_json::from_value(json!({
            "id": "call_0",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }))
        .expect("tool call")
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    /// The whole reason this module was rewritten: `[INST] … [/INST]`
    /// matches none of the old sniffer's markers, so a real Mistral
    /// checkpoint was served role-labeled lines it has never seen.
    /// The TEMPLATE decides the tool-call grammar, because the served
    /// name is whatever `--alias` said.
    ///
    /// Bonsai is the case that found this: a Qwen3.5 file served as
    /// `bonsai-2-27b`, whose template prints
    /// `<tool_call><function=…><parameter=…>` while
    /// `ToolCallFormat::infer` sees no "qwen" in the name and falls
    /// through to the Llama 3 fallback, which reads none of those
    /// calls. Every call reached the client as raw markup in `content`
    /// with `tool_calls` null.
    #[test]
    fn the_template_decides_the_tool_call_grammar_not_the_served_name() {
        use crate::policy::parser::ToolCallFormat;

        let qwen3_coder = "{% for m in messages %}{{ m.content }}{% endfor %}\
             {% if tools %}If you call a function reply as:\n<tool_call>\n\
             <function=name>\n<parameter=arg>\nvalue\n</parameter>\n\
             </function>\n</tool_call>{% endif %}";
        let tmpl =
            PromptTemplate::from_gguf_metadata(Some(qwen3_coder), None, false, true, None, None);
        // The name says nothing; the template says everything.
        assert_eq!(
            tmpl.tool_call_format("bonsai-2-27b"),
            ToolCallFormat::Qwen3Coder,
            "an alias that names no family must not decide the grammar"
        );
        assert_eq!(
            ToolCallFormat::infer("bonsai-2-27b"),
            ToolCallFormat::Llama3,
            "and the name alone really does resolve to the wrong one"
        );

        // Hermes JSON: a `<tool_call>` with no `<function=` inside it.
        let hermes = "{% for m in messages %}{{ m.content }}{% endfor %}\
             {% if tools %}Reply with <tool_call>{\"name\": …}</tool_call>{% endif %}";
        let tmpl = PromptTemplate::from_gguf_metadata(Some(hermes), None, false, true, None, None);
        assert_eq!(
            tmpl.tool_call_format("bonsai-2-27b"),
            ToolCallFormat::Qwen25
        );

        // A template that names no grammar leaves the name to decide.
        let silent = "{% for m in messages %}{{ m.content }}{% endfor %}";
        let tmpl = PromptTemplate::from_gguf_metadata(Some(silent), None, false, true, None, None);
        assert_eq!(
            tmpl.tool_call_format("Qwen3-Coder-30B"),
            ToolCallFormat::Qwen3Coder
        );
    }

    #[test]
    fn a_template_the_old_sniffer_could_not_recognise_now_renders_correctly() {
        let mistral = "{% for m in messages %}{% if m.role == 'user' %}[INST] {{ m.content }} [/INST]{% else %}{{ m.content }}</s>{% endif %}{% endfor %}";
        let tmpl = PromptTemplate::from_gguf_metadata(
            Some(mistral),
            Some("llama"),
            false,
            true,
            None,
            None,
        );
        let rendered = tmpl
            .render(&[msg("user", "hi")], &[], Map::new())
            .expect("renders");
        assert_eq!(rendered, "[INST] hi [/INST]");
    }

    #[test]
    fn a_checkpoint_with_no_template_falls_back_to_chatml_like_llama_cpp() {
        let tmpl = PromptTemplate::from_gguf_metadata(None, Some("olmoe"), false, true, None, None);
        let rendered = tmpl
            .render(&[msg("user", "hi")], &[], Map::new())
            .expect("renders");
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn a_byte_tokenizer_checkpoint_falls_back_to_role_labeled_lines() {
        let tmpl = PromptTemplate::from_gguf_metadata(None, Some("olmoe"), true, true, None, None);
        let rendered = tmpl
            .render(
                &[msg("user", "hello"), msg("assistant", "hi back")],
                &[],
                Map::new(),
            )
            .expect("renders");
        assert_eq!(rendered, "user: hello\nassistant: hi back");
    }

    #[test]
    fn chat_template_kwargs_reach_the_template() {
        let src = "{% if enable_thinking %}THINK{% endif %}{{ messages[0].content }}";
        let tmpl =
            PromptTemplate::from_gguf_metadata(Some(src), Some("qwen3"), false, true, None, None);
        let mut extra = Map::new();
        extra.insert("enable_thinking".into(), json!(true));
        assert_eq!(
            tmpl.render(&[msg("user", "hi")], &[], extra)
                .expect("renders"),
            "THINKhi"
        );
        assert_eq!(
            tmpl.render(&[msg("user", "hi")], &[], Map::new())
                .expect("renders"),
            "hi"
        );
    }

    /// The parser comes from the template when the name says nothing.
    /// Bonsai-2-27B is served as `Hf` and every Qwen3.5 export as
    /// `Original`; both templates open a `<think>` block, and before
    /// this the thinking landed in `content` for both.
    #[test]
    fn a_think_template_implies_the_think_parser_whatever_the_name() {
        use crate::policy::parser::ReasoningFormat;
        let src = "{{ messages[0].content }}{% if enable_thinking %}<think>\n{% endif %}";
        let tmpl =
            PromptTemplate::from_gguf_metadata(Some(src), Some("qwen3"), false, true, None, None);
        assert_eq!(tmpl.reasoning_format("Hf"), Some(ReasoningFormat::Think));
        assert_eq!(
            tmpl.reasoning_format("Original"),
            Some(ReasoningFormat::Think)
        );
        // The name still wins where it names a family the text cannot.
        assert_eq!(
            tmpl.reasoning_format("gpt-oss-20b"),
            Some(ReasoningFormat::GptOss)
        );
        let plain = PromptTemplate::from_gguf_metadata(
            Some("{{ messages[0].content }}"),
            Some("llama"),
            false,
            true,
            None,
            None,
        );
        assert_eq!(plain.reasoning_format("Hf"), None);
        assert_eq!(plain.reasoning_format("llama-3.1-8b"), None);
    }

    #[test]
    fn a_template_that_reads_tools_is_given_them_structurally() {
        let src =
            "{% for t in tools %}TOOL:{{ t.function.name }}{% endfor %}{{ messages[0].content }}";
        let tmpl =
            PromptTemplate::from_gguf_metadata(Some(src), Some("qwen3"), false, true, None, None);
        assert!(tmpl.handles_tools());
        let tools = vec![tool_def("get_weather")];
        assert_eq!(
            tmpl.render(&[msg("user", "hi")], &tools, Map::new())
                .expect("renders"),
            "TOOL:get_weatherhi"
        );
    }

    /// A source check says this template handles tools; rendering it
    /// says otherwise. The preamble is what the model would lose.
    #[test]
    fn a_template_that_only_mentions_tools_does_not_count_as_handling_them() {
        let mentions = "{# tools are described by the system prompt #}\
             {% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
        let tmpl = PromptTemplate::from_gguf_metadata(
            Some(mentions),
            Some("qwen2"),
            false,
            true,
            None,
            None,
        );
        assert!(
            !tmpl.handles_tools(),
            "the word alone must not skip the preamble"
        );
    }

    /// A template that never mentions `tools` cannot describe them, so
    /// the caller keeps the text preamble -- and a replayed call has to
    /// come back as the marker text that preamble asked for.
    #[test]
    fn a_template_that_ignores_tools_sees_replayed_calls_as_marker_text() {
        let tmpl = PromptTemplate::from_gguf_metadata(
            Some(CHATML),
            Some("qwen2"),
            false,
            true,
            None,
            None,
        );
        assert!(!tmpl.handles_tools());
        let replayed = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![tool_call_in("get_weather", r#"{"city":"Paris"}"#)]),
            tool_call_id: None,
            reasoning_content: None,
        };
        let rendered = tmpl.render(&[replayed], &[], Map::new()).expect("renders");
        assert!(
            rendered.contains(
                r#"<tool_call>{"name": "get_weather", "arguments": {"city":"Paris"}}</tool_call>"#
            ),
            "{rendered}"
        );
    }

    /// The wire form of `arguments` is a string; a template written
    /// against HuggingFace's `apply_chat_template` expects the parsed
    /// object, and `| tojson` on the string form would double-encode it.
    #[test]
    fn replayed_call_arguments_reach_a_structural_template_as_an_object() {
        let src = "{% for m in messages %}{% for c in m.tool_calls %}{{ c.function.arguments.city }}{% endfor %}{% endfor %}tools:{{ tools | length }}";
        let tmpl =
            PromptTemplate::from_gguf_metadata(Some(src), Some("qwen3"), false, true, None, None);
        let replayed = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![tool_call_in("get_weather", r#"{"city":"Paris"}"#)]),
            tool_call_id: None,
            reasoning_content: None,
        };
        let tools = vec![tool_def("get_weather")];
        assert_eq!(
            tmpl.render(&[replayed], &tools, Map::new())
                .expect("renders"),
            "Paristools:1"
        );
    }

    /// A message and a replayed call reach the template with their keys
    /// in llama.cpp's order, the call's id last and only when it has one.
    #[test]
    fn a_replayed_turn_keeps_llama_cpps_key_order() {
        let src = "{% for m in messages %}{{ m | tojson }}\n\
                   {% for c in m.tool_calls %}{% endfor %}{% endfor %}{{ tools | length }}";
        let tmpl =
            PromptTemplate::from_gguf_metadata(Some(src), Some("qwen3"), false, true, None, None);
        let mut no_id = tool_call_in("g", "{}");
        no_id.id.clear();
        let replayed = ChatMessage {
            role: "assistant".to_string(),
            content: Some(MessageContent::Text("hi".to_string())),
            tool_calls: Some(vec![tool_call_in("f", r#"{"b":1,"a":2}"#), no_id]),
            tool_call_id: None,
            reasoning_content: Some("why".to_string()),
        };
        let result = ChatMessage {
            tool_call_id: Some("call_0".to_string()),
            ..msg("tool", "18C")
        };
        let tools = vec![tool_def("f")];
        assert_eq!(
            tmpl.render(&[replayed, result], &tools, Map::new())
                .expect("renders"),
            concat!(
                r#"{"role": "assistant", "content": "hi", "reasoning_content": "why", "#,
                r#""reasoning": "why", "tool_calls": ["#,
                r#"{"type": "function", "function": {"name": "f", "arguments": {"b": 1, "a": 2}}, "id": "call_0"}, "#,
                r#"{"type": "function", "function": {"name": "g", "arguments": {}}}]}"#,
                "\n",
                r#"{"role": "tool", "content": "18C", "tool_call_id": "call_0"}"#,
                "\n1",
            )
        );
    }

    #[test]
    fn a_broken_template_fails_the_request_instead_of_guessing() {
        let tmpl = PromptTemplate::from_gguf_metadata(
            Some("{% for m in messages %}"),
            None,
            false,
            true,
            None,
            None,
        );
        assert!(tmpl.render(&[msg("user", "hi")], &[], Map::new()).is_err());
    }

    /// Every one of these used to need its own arm in a substring
    /// sniffer, and a family without an arm got no stop string at all.
    /// They are now derived from the render, so this test is about the
    /// families rather than about the list.
    #[test]
    fn the_end_of_turn_marker_is_derived_from_the_render() {
        let marker = |src: Option<&str>, arch: Option<&str>, eos: Option<&str>| {
            PromptTemplate::from_gguf_metadata(
                src,
                arch,
                false,
                true,
                Some("<bos>".to_string()),
                eos.map(str::to_string),
            )
            .end_of_turn()
            .map(str::to_string)
        };
        let turns = |open: &str, close: &str| {
            format!(
                "{{{{ bos_token }}}}{{% for m in messages %}}{open}{{{{ m.role }}}}\n\
                 {{{{ m.content }}}}{close}\n{{% endfor %}}"
            )
        };

        // Gemma IT: `<end_of_turn>` before `<eos>`, so decoding does not
        // stop on it and the stop set must.
        assert_eq!(
            marker(
                Some(&turns("<start_of_turn>", "<end_of_turn>")),
                None,
                Some("<eos>")
            ),
            Some("<end_of_turn>".to_string())
        );
        // gemma-4, a different marker in the same shape.
        assert_eq!(
            marker(Some(&turns("<|turn>", "<turn|>")), None, Some("<eos>")),
            Some("<turn|>".to_string())
        );
        // A marker no version of the sniffer ever had an arm for. This
        // is the point: nothing in the code names it.
        assert_eq!(
            marker(Some(&turns("[[", "]]/turn")), None, Some("<eos>")),
            Some("]]/turn".to_string())
        );

        // ChatML ends a turn with `<|im_end|>`, and MOST checkpoints
        // make that their EOS so decoding stops on it for free. Yi-1.5
        // does not: its template uses `<|im_end|>` (id 7) while
        // `eos_token_id` is 2, so nothing stopped and the marker was
        // emitted as literal text mid-answer.
        assert_eq!(
            marker(Some(CHATML), None, Some("</s>")),
            Some("<|im_end|>".to_string())
        );
        // The same template on a checkpoint that DOES make the marker
        // its EOS contributes nothing, because decoding already stops.
        assert_eq!(marker(Some(CHATML), None, Some("<|im_end|>")), None);

        // Mistral-Instruct, which has no per-turn marker at all: the
        // assistant turn is closed by `</s>`, which IS the EOS, so there
        // is nothing to add. The old sniffer reached the same answer by
        // recognising none of its markers, which is the right answer for
        // the wrong reason -- give the same template a different EOS and
        // the derivation still finds the terminator.
        const MISTRAL: &str = "{% for m in messages %}{% if m.role == 'user' %}\
             [INST] {{ m.content }} [/INST]{% else %}{{ m.content }}</s>{% endif %}{% endfor %}";
        assert_eq!(marker(Some(MISTRAL), None, Some("</s>")), None);
        assert_eq!(
            marker(Some(MISTRAL), None, Some("<eos>")),
            Some("</s>".to_string())
        );

        // No template at all, so the builtin prints `user:` /
        // `assistant:` lines and a base model continues that pattern to
        // the token cap because it has no EOS for a turn it never saw.
        // OLMoE-1B-7B answered "Paris." correctly and then wrote the
        // next `user:` line itself for 512 tokens. The marker the
        // builtin prints is the stop, and it is derived too.
        // `chatml_tokens_present: false` is what makes this Plain: a
        // vocabulary with no `<|im_start|>` cannot be wrapped in ChatML.
        let plain =
            PromptTemplate::from_gguf_metadata(None, Some("olmoe"), false, false, None, None);
        assert_eq!(plain.end_of_turn(), Some("\nuser:"));
    }

    /// The probe is what makes `reasoning_effort` mean anything: a
    /// template that grades only the OpenAI triple must not be sent
    /// `minimal`.
    #[test]
    fn the_effort_vocabulary_is_probed_at_load() {
        let graded = "{% set allowed = ['low','medium','high'] %}\
             {% if reasoning_effort %}\
               {% if reasoning_effort not in allowed %}{{ raise_exception('bad effort') }}{% endif %}\
               E:{{ reasoning_effort }}\
             {% endif %}{{ messages[0].content }}";
        let tmpl = PromptTemplate::from_gguf_metadata(
            Some(graded),
            Some("qwen3"),
            false,
            true,
            None,
            None,
        );
        let profile = tmpl.efforts();
        assert!(profile.consumes_effort);
        assert!(profile.validates);
        assert_eq!(
            profile
                .supported
                .iter()
                .map(|e| e.as_str())
                .collect::<Vec<_>>(),
            vec!["low", "medium", "high"]
        );

        let inert = PromptTemplate::from_gguf_metadata(
            Some(CHATML),
            Some("qwen2"),
            false,
            true,
            None,
            None,
        );
        assert!(!inert.efforts().consumes_effort);
    }

    /// Templates disagree about the key, so both are emitted. The
    /// DeepSeek/GLM lineage iterates `message.reasoning_content`, Qwen
    /// and gpt-oss `message.reasoning`; a template reads the one it
    /// does not know as undefined, so emitting both costs a key and
    /// loses nothing, while emitting one drops a replayed chain of
    /// thought on every checkpoint written against the other.
    #[test]
    fn a_replayed_chain_of_thought_reaches_a_template_under_both_spellings() {
        let mut m = msg("assistant", "the answer is 4");
        m.reasoning_content = Some("2 + 2".to_string());

        let json = message_json(&m, false);
        assert_eq!(json["reasoning_content"], "2 + 2");
        assert_eq!(json["reasoning"], "2 + 2");
        assert_eq!(json["thinking"], "2 + 2");
        assert_eq!(
            json["content"], "the answer is 4",
            "reasoning must never be folded into what the model said"
        );
    }

    /// The one case where a key is deliberately withheld. gpt-oss's
    /// harmony template RAISES on an assistant turn that makes tool
    /// calls while carrying both visible text and `thinking` -- the two
    /// land on channels that cannot both be final -- so a replayed
    /// agent turn would 500 the whole request rather than render
    /// oddly. Visible text wins, and the other two spellings still
    /// carry the reasoning for the families that read them.
    #[test]
    fn a_tool_call_turn_with_visible_text_withholds_only_the_harmony_spelling() {
        let mut m = msg("assistant", "checking the weather");
        m.reasoning_content = Some("I should call the tool".to_string());
        m.tool_calls = Some(vec![tool_call_in("get_weather", "{}")]);

        let json = message_json(&m, true);
        assert!(
            json.get("thinking").is_none(),
            "harmony would raise on this turn"
        );
        assert_eq!(json["reasoning_content"], "I should call the tool");
        assert_eq!(json["reasoning"], "I should call the tool");

        // A tool-call turn with NO visible text is the ordinary harmony
        // shape and keeps all three.
        m.content = None;
        assert_eq!(message_json(&m, true)["thinking"], "I should call the tool");
    }

    /// Empty is not absent. A template that opens the family's thinking
    /// markers on `if message.reasoning_content` would wrap them around
    /// nothing, so an empty string is skipped exactly like `None` --
    /// which also keeps a plain message's rendering byte-identical to
    /// what it was before the field existed.
    #[test]
    fn an_empty_or_absent_chain_of_thought_puts_no_key_in_front_of_a_template() {
        for empty in [None, Some(String::new())] {
            let mut m = msg("assistant", "hi");
            m.reasoning_content = empty;
            let json = message_json(&m, false);
            assert!(json.get("reasoning_content").is_none());
            assert!(json.get("reasoning").is_none());
            assert!(json.get("thinking").is_none());
        }
    }

    /// Both spellings are accepted on the way in, too. A client that
    /// received `reasoning` from `/v1/responses` or `/v1/messages` and
    /// replays the turn verbatim must not have it silently ignored
    /// because the chat surface spells the key the other way.
    #[test]
    fn a_replayed_turn_is_accepted_under_either_spelling_of_the_key() {
        for key in ["reasoning_content", "reasoning"] {
            let m: ChatMessage = serde_json::from_value(json!({
                "role": "assistant",
                "content": "4",
                key: "2 + 2",
            }))
            .expect("deserializes");
            assert_eq!(m.reasoning_content.as_deref(), Some("2 + 2"), "{key}");
        }
    }
}
