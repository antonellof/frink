//! What the model actually said: splitting generated text into
//! reasoning, answer, and tool calls.
//!
//! Two things were previously missing here and are not any more.
//!
//! **A reasoning model's chain of thought was part of its answer.** A
//! checkpoint trained to think inside `<think>` emitted the whole
//! block into `content`, so a client rendered several paragraphs of
//! deliberation as the reply. It now goes to `reasoning_content`,
//! which is where every client in the ecosystem already looks for it.
//!
//! **Only one tool-call format was understood, and only the first
//! call.** This server prompt-engineers the Hermes-style
//! `<tool_call>{…}</tool_call>` marker (see `tool_preamble`), and a
//! model trained on a *different* format frequently answers in its own
//! anyway -- correctly, in its own terms, and then went unrecognized.
//! Parsing now tries the format the served checkpoint's family
//! implies, then the one the preamble asked for, and returns every
//! call rather than the first.
//!
//! Both parsers live in `frink-edge`; this module is the request-shaped
//! layer over them -- which tools were offered, which format the served
//! model implies, and how to say the answer in OpenAI's response
//! shape.

use crate::policy::parser::tool_call::ToolSchema;
use crate::policy::parser::{ReasoningFormat, ReasoningParser, ToolCallFormat, ToolCallParser};

use crate::{ToolDef, ToolFunctionDef};

/// One recognized call, in the shape the wire wants: the arguments as
/// a JSON-encoded *string*, which is OpenAI's real convention even
/// though the model writes literal JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedToolCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

/// Everything a response body needs from the generated text.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ParsedOutput {
    /// The chain of thought, when the checkpoint emitted one.
    pub(crate) reasoning: Option<String>,
    /// What is left once the reasoning and the calls are taken out.
    pub(crate) content: String,
    pub(crate) calls: Vec<ParsedToolCall>,
}

/// The tools a request offered, in the form the parser types values
/// against.
pub(crate) fn tool_schemas(tools: &[ToolDef]) -> Vec<ToolSchema> {
    tools
        .iter()
        .map(|tool| {
            let ToolFunctionDef {
                name, parameters, ..
            } = &tool.function;
            match parameters {
                Some(schema) => ToolSchema::with_parameters(name.clone(), schema.clone()),
                None => ToolSchema::new(name.clone()),
            }
        })
        .collect()
}

/// How this request's output has to be read, resolved once.
///
/// Two facts, and they come from different places.
///
/// The *format* comes from the served checkpoint's name, which is all
/// there is to infer a family from -- frink does not carry a
/// per-checkpoint parser declaration. A name that implies nothing simply
/// gets no reasoning parser, which is the right answer for a model that
/// does not reason: an unconditional `<think>` splitter would silently
/// eat a literal `<think>` a non-reasoning model wrote in a code block.
///
/// Whether the block is *already open* comes from the prompt that was
/// actually rendered. Now that `chat_template_kwargs` reaches the
/// template, a template asked to think can open the reasoning block in
/// the prompt -- and then the model's first token is reasoning and no
/// opening marker will ever arrive. Reading it off the rendered text
/// (`ReasoningFormat::prompt_opens_reasoning`) is the only way to know
/// that is not a guess about a family.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OutputPosture {
    reasoning: Option<ReasoningFormat>,
    reasoning_open: bool,
    tools: ToolCallFormat,
    /// The request's `parallel_tool_calls: false`: at most one call
    /// comes back ([`Self::one_call`]).
    one_call: bool,
}

impl OutputPosture {
    /// The posture for a name alone: what the live routes did before the
    /// template's evidence joined the decision. Tests that pin a family
    /// by name use it; every route goes through [`Self::resolve_with`].
    #[cfg(test)]
    pub(crate) fn resolve(model_name: &str, prompt: &str) -> Self {
        Self::resolve_with(ReasoningFormat::infer(model_name), model_name, prompt)
    }

    /// The posture with the reasoning format decided elsewhere and the
    /// tool-call format taken from the NAME. Tests and the
    /// no-template paths; the serving path uses
    /// [`Self::resolve_full`].
    pub(crate) fn resolve_with(
        reasoning: Option<ReasoningFormat>,
        model_name: &str,
        prompt: &str,
    ) -> Self {
        Self::resolve_full(reasoning, ToolCallFormat::infer(model_name), prompt)
    }

    /// The posture with BOTH formats decided elsewhere
    /// (`PromptTemplate::reasoning_format` and
    /// `PromptTemplate::tool_call_format`, each of which also reads the
    /// template). The serving path takes this one: a served name comes
    /// from `--alias` and cannot be trusted to name the family.
    pub(crate) fn resolve_full(
        reasoning: Option<ReasoningFormat>,
        tools: ToolCallFormat,
        prompt: &str,
    ) -> Self {
        OutputPosture {
            reasoning,
            reasoning_open: reasoning.is_some_and(|f| f.prompt_opens_reasoning(prompt)),
            tools,
            one_call: false,
        }
    }

    /// `parallel_tool_calls: false`. The caller asked for at most one
    /// call, so the first is returned and any after it are dropped, on
    /// the whole-answer parse and on the stream alike. The default,
    /// `true`, returns every call the model made.
    pub(crate) fn one_call(mut self, one: bool) -> Self {
        self.one_call = one;
        self
    }

    /// Whether at most one call may come back.
    pub(crate) fn is_one_call(&self) -> bool {
        self.one_call
    }

    /// The posture for text with no prompt behind it: a checkpoint's own
    /// output read on its own terms.
    #[cfg(test)]
    pub(crate) fn for_model(model_name: &str) -> Self {
        Self::resolve(model_name, "")
    }

    /// A parser positioned where this request's prompt left the model.
    pub(crate) fn reasoning_parser(&self) -> Option<ReasoningParser> {
        self.reasoning
            .map(|format| ReasoningParser::new(format, self.reasoning_open, true))
    }

    pub(crate) fn tool_call_parser(&self, tools: &[ToolDef]) -> ToolCallParser {
        ToolCallParser::new(self.tools, tool_schemas(tools))
    }
}

/// Split generated text.
pub(crate) fn parse_output(text: &str, tools: &[ToolDef], posture: OutputPosture) -> ParsedOutput {
    let (reasoning, remainder) = split_reasoning(text, posture);
    let (content, mut calls) = extract_tool_calls(&remainder, tools, posture);
    if posture.one_call {
        calls.truncate(1);
    }
    ParsedOutput {
        reasoning,
        content,
        calls,
    }
}

/// Cut the chain of thought off the front of the answer.
fn split_reasoning(text: &str, posture: OutputPosture) -> (Option<String>, String) {
    let Some(parser) = posture.reasoning_parser() else {
        return (None, text.to_string());
    };
    let split = parser.parse_complete(text);
    if split.reasoning.is_empty() {
        return (None, split.content);
    }
    (Some(split.reasoning), split.content)
}

/// Find every tool call in `text`.
///
/// Two formats are tried, in this order and for this reason: the one
/// the served model's family emits natively, then the Hermes-style one
/// the preamble asked for. A model that ignores the preamble and
/// answers in its own format is the common case worth getting right,
/// and a model that follows the preamble is caught by the fallback --
/// so both work, and neither has to be configured.
fn extract_tool_calls(
    text: &str,
    tools: &[ToolDef],
    posture: OutputPosture,
) -> (String, Vec<ParsedToolCall>) {
    if tools.is_empty() || !ToolCallParser::text_may_contain_call(text) {
        return (text.to_string(), Vec::new());
    }
    let schemas = tool_schemas(tools);
    let native = posture.tools;
    let mut formats = vec![native];
    if native != ToolCallFormat::Qwen25 {
        formats.push(ToolCallFormat::Qwen25);
    }
    for format in formats {
        let parser = ToolCallParser::new(format, schemas.clone());
        let (content, calls) = parser.parse_complete(text);
        if !calls.is_empty() {
            return (
                content.trim().to_string(),
                calls
                    .into_iter()
                    .map(|call| ParsedToolCall {
                        name: call.name,
                        arguments: call.arguments,
                    })
                    .collect(),
            );
        }
    }
    (text.to_string(), Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<ToolDef> {
        vec![ToolDef {
            kind: "function".to_string(),
            function: ToolFunctionDef {
                name: "get_weather".to_string(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                })),
            },
        }]
    }

    /// The format the preamble asks for, from a model that has no
    /// format of its own.
    #[test]
    fn the_prompt_engineered_marker_is_still_understood() {
        let parsed = parse_output(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>",
            &tools(),
            OutputPosture::for_model("some-random-7b"),
        );
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "get_weather");
        assert_eq!(parsed.calls[0].arguments, r#"{"city":"Rome"}"#);
    }

    /// A model answering in its own native format used to go
    /// unrecognized and be returned as prose.
    #[test]
    fn a_models_own_format_is_understood_too() {
        let parsed = parse_output(
            "<tool_call><function=get_weather><parameter=city>\nRome\n</parameter>\
             </function></tool_call>",
            &tools(),
            OutputPosture::for_model("Qwen3-Coder-30B"),
        );
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "get_weather");
        assert_eq!(parsed.calls[0].arguments, r#"{"city":"Rome"}"#);
    }

    /// ... and a model whose family implies one format but which
    /// followed the preamble instead is still caught.
    #[test]
    fn a_native_family_that_followed_the_preamble_still_works() {
        let parsed = parse_output(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>",
            &tools(),
            OutputPosture::for_model("Qwen3-Coder-30B"),
        );
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].arguments, r#"{"city":"Oslo"}"#);
    }

    /// A family whose template opens the reasoning block in the prompt
    /// emits its close and then its call, and both are recognized.
    #[test]
    fn a_family_that_always_thinks_first_still_has_its_call_found() {
        let parsed = parse_output(
            "I need the weather.</think>\
             <minimax:tool_call><invoke name=\"get_weather\">\
             <parameter name=\"city\">Rome</parameter></invoke></minimax:tool_call>",
            &tools(),
            OutputPosture::for_model("MiniMax-M2"),
        );
        assert_eq!(parsed.reasoning.as_deref(), Some("I need the weather."));
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].arguments, r#"{"city":"Rome"}"#);
    }

    /// More than one call in a response is more than one call, not the
    /// first one.
    #[test]
    fn every_call_is_returned_not_just_the_first() {
        let parsed = parse_output(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>\n\
             <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>",
            &tools(),
            OutputPosture::for_model("qwen2.5-7b"),
        );
        assert_eq!(parsed.calls.len(), 2);
        assert_eq!(parsed.calls[1].arguments, r#"{"city":"Oslo"}"#);
    }

    /// `parallel_tool_calls: false` returns the first call and only the
    /// first, from the same text that returns both by default.
    #[test]
    fn one_call_returns_the_first_call_only() {
        let parsed = parse_output(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>\n\
             <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>",
            &tools(),
            OutputPosture::for_model("qwen2.5-7b").one_call(true),
        );
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].arguments, r#"{"city":"Rome"}"#);
    }

    #[test]
    fn a_plain_answer_with_tools_offered_stays_a_plain_answer() {
        let parsed = parse_output(
            "The weather is fine.",
            &tools(),
            OutputPosture::for_model("qwen2.5-7b"),
        );
        assert!(parsed.calls.is_empty());
        assert_eq!(parsed.content, "The weather is fine.");
        assert_eq!(parsed.reasoning, None);
    }

    /// A reasoning model's deliberation used to be returned as the
    /// answer.
    #[test]
    fn a_reasoning_block_leaves_the_answer() {
        let parsed = parse_output(
            "<think>The user wants weather. I should just say it.</think>It is sunny.",
            &[],
            OutputPosture::for_model("Qwen3-8B"),
        );
        assert_eq!(
            parsed.reasoning.as_deref(),
            Some("The user wants weather. I should just say it.")
        );
        assert_eq!(parsed.content, "It is sunny.");
    }

    /// A model with no reasoning format must not have one applied: a
    /// literal `<think>` in a code block is then just text.
    #[test]
    fn a_non_reasoning_model_keeps_its_markers_as_text() {
        let parsed = parse_output(
            "Use the tag <think> like this.",
            &[],
            OutputPosture::for_model("llama-3.1-8b-instruct"),
        );
        assert_eq!(parsed.reasoning, None);
        assert_eq!(parsed.content, "Use the tag <think> like this.");
    }

    /// Both at once: a reasoning model that thinks and then calls a
    /// tool.
    #[test]
    fn reasoning_and_a_call_are_separated_from_each_other() {
        let parsed = parse_output(
            "<think>I need the weather.</think>\
             <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>",
            &tools(),
            OutputPosture::for_model("Qwen3-8B"),
        );
        assert_eq!(parsed.reasoning.as_deref(), Some("I need the weather."));
        assert_eq!(parsed.calls.len(), 1);
        assert!(parsed.content.is_empty(), "{:?}", parsed.content);
    }

    #[test]
    fn a_tool_the_request_never_offered_is_not_returned() {
        let parsed = parse_output(
            "<tool_call>{\"name\": \"rm_rf\", \"arguments\": {}}</tool_call>",
            &tools(),
            OutputPosture::for_model("qwen2.5-7b"),
        );
        assert!(parsed.calls.is_empty());
    }

    #[test]
    fn schemas_carry_the_declared_parameter_types() {
        let schemas = tool_schemas(&tools());
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, "get_weather");
        assert!(schemas[0].parameters.is_some());
    }
}
