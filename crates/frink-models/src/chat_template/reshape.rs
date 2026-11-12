//! What llama.cpp does to a conversation before it renders it, for the
//! templates that need it (`common/chat.cpp`, `common_chat_templates_apply_jinja`).
//!
//! Three kinds, applied in upstream's order:
//!
//! 1. A system turn the template will not show is folded into the next
//!    message, or dropped when it is alone
//!    (`workaround::system_message_not_supported`). Gemma 2 raises
//!    "System role not supported" on one, and TranslateGemma insists the
//!    conversation start with the user.
//! 2. A replayed call's `arguments`, a JSON string on the wire, is parsed
//!    into an object for a template that reads one
//!    (`workaround::func_args_not_string`). Every other template gets the
//!    string: Functionary v3.2 concatenates it.
//! 3. The handler `common_chat_template_detect_params_init` picks from
//!    the template's source rewrites the messages for its family. Four
//!    of them do: [`Family`].

use serde_json::{json, Map, Value};

use super::{ChatTemplate, RenderOptions};

/// A family whose upstream handler rewrites the messages before
/// rendering. Each is detected as `common_chat_template_detect_params_init`
/// detects it, from markers in the template source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Ministral 3, Magistral and Devstral (`common/parsers/ministral3.cpp`):
    /// a system or assistant turn's reasoning and text become typed
    /// content blocks.
    Ministral3,
    /// K2 Horizon (`common/parsers/k2-horizon.cpp`): every assistant turn
    /// carries `reasoning_content`, empty when it had none, because the
    /// template raises on one without.
    K2Horizon,
    /// Gemma 4 before its OpenAI-shaped template
    /// (`common/parsers/gemma4.cpp`, `convert_tool_responses_gemma4`): a
    /// call, its results and the answer after them become one assistant
    /// turn with `tool_responses`.
    Gemma4Legacy,
    /// TranslateGemma (`common/parsers/translate-gemma.cpp`): a user
    /// turn becomes one typed item carrying the language codes.
    TranslateGemma,
}

impl Family {
    /// In upstream's order, which is first match wins. No rule upstream
    /// checks before these matches one of them on a template llama.cpp
    /// bundles, which the render census holds.
    pub fn detect(source: &str) -> Option<Self> {
        let has = |marker: &str| source.contains(marker);
        if has("[SYSTEM_PROMPT]") && has("[TOOL_CALLS]") && has("[ARGS]") && !has("[CALL_ID]") {
            Some(Self::Ministral3)
        } else if has("<|ifm|im_start|>") && has("<ifm|tool_calls>") {
            Some(Self::K2Horizon)
        } else if has("'<|tool_call>call:'") {
            (!has("{#- OpenAI Chat Completions:")).then_some(Self::Gemma4Legacy)
        } else if has("[source_lang_code]") && has("[target_lang_code]") {
            Some(Self::TranslateGemma)
        } else {
            None
        }
    }
}

/// How one template's conversations are reshaped. Learned at load.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reshape {
    /// The template does not show a system turn's content (llama.cpp's
    /// `supports_system_role` is false).
    pub fold_system: bool,
    /// The template reads a call's `arguments` as an object (llama.cpp's
    /// `supports_object_arguments`), so the wire string is parsed.
    pub object_arguments: bool,
    pub family: Option<Family>,
}

impl Reshape {
    /// What llama.cpp does to conversations it renders with `template`,
    /// learned once, by rendering it.
    pub fn for_template(template: &ChatTemplate, bos: Option<&str>, eos: Option<&str>) -> Self {
        Self {
            fold_system: !shows_system(template, bos, eos),
            object_arguments: reads_object_arguments(template, bos, eos),
            family: template.source().and_then(Family::detect),
        }
    }

    /// Reshapes `messages` for this template. `kwargs` are the render's
    /// `chat_template_kwargs`, which TranslateGemma's language codes come
    /// from. An error names what the reshape could not use.
    pub fn apply(
        &self,
        messages: &mut Vec<Value>,
        kwargs: &Map<String, Value>,
    ) -> Result<(), String> {
        if self.fold_system {
            fold_system(messages);
        }
        if self.object_arguments {
            parse_arguments(messages)?;
        }
        match self.family {
            Some(Family::Ministral3) => ministral3(messages),
            Some(Family::K2Horizon) => k2_horizon(messages),
            Some(Family::Gemma4Legacy) => *messages = gemma4_tool_responses(messages),
            Some(Family::TranslateGemma) => translate_gemma(messages, kwargs)?,
            None => {}
        }
        Ok(())
    }
}

/// llama.cpp's `supports_system_role` (`common/jinja/caps.cpp`): does a
/// system turn's text reach the prompt? Upstream renders the same two
/// turns and asks whether the template read the system text. This asks
/// whether it printed it, which is the same answer unless a template
/// reads the text and then throws it away. A render that fails, as
/// Gemma 2's raise on a system turn does, has not shown it.
fn shows_system(template: &ChatTemplate, bos: Option<&str>, eos: Option<&str>) -> bool {
    let messages = [
        json!({"role": "system", "content": "System message"}),
        json!({"role": "user", "content": "User message"}),
    ];
    let opts = RenderOptions {
        add_generation_prompt: true,
        bos_token: bos.map(str::to_string),
        eos_token: eos.map(str::to_string),
        ..Default::default()
    };
    template
        .render(&messages, &opts)
        .is_ok_and(|out| out.contains("System message"))
}

/// llama.cpp's `supports_object_arguments` (`common/jinja/caps.cpp`):
/// does the template read a replayed call's `arguments` as an object?
///
/// Upstream renders this conversation with `arguments` as
/// `{"arg": ...}` and asks whether the template read `arg`. minijinja
/// cannot say what a template read, so this asks whether the value
/// reached the prompt. `arguments.arg`, `arguments.items()`,
/// `arguments | tojson` and `arguments | string` agree with upstream. A
/// template that prints the whole map with `{{ arguments }}` gets an
/// object here and the string upstream, and no template llama.cpp
/// bundles does that, which frink-server's render census holds. A render
/// that fails, as Functionary v3.2's string plus map does, reads no
/// object. As upstream does, a failed render is tried once more with an
/// empty `reasoning_content` on each assistant turn, which K2 Horizon
/// insists on.
fn reads_object_arguments(template: &ChatTemplate, bos: Option<&str>, eos: Option<&str>) -> bool {
    const ARG: &str = "frink_probe_argument_value";
    let mut messages = vec![
        json!({"role": "user", "content": "User message"}),
        json!({"role": "assistant", "content": "", "tool_calls": [{
            "id": "call00001",
            "type": "function",
            "function": {"name": "tool1", "arguments": {"arg": ARG}},
        }]}),
        json!({"role": "tool", "content": "Tool response", "tool_call_id": "call00001"}),
        json!({"role": "assistant", "content": "The tool response was 'tool response'"}),
        json!({"role": "user", "content": "User message"}),
    ];
    let opts = RenderOptions {
        add_generation_prompt: true,
        bos_token: bos.map(str::to_string),
        eos_token: eos.map(str::to_string),
        tools: vec![json!({
            "name": "tool",
            "type": "function",
            "function": {
                "name": "tool1",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string", "description": "Arg description"}},
                    "required": ["arg"],
                },
            },
        })],
        ..Default::default()
    };
    let shown = |messages: &[Value]| {
        template
            .render(messages, &opts)
            .map(|out| out.contains(ARG))
    };
    shown(&messages).unwrap_or_else(|_| {
        for message in &mut messages {
            if role(message) == "assistant" && message.get("reasoning_content").is_none() {
                message["reasoning_content"] = json!("");
            }
        }
        shown(&messages).unwrap_or(false)
    })
}

/// `func_args_not_string`: each call's string `arguments` parsed as
/// JSON. A string that is not JSON is an error, as it is upstream.
fn parse_arguments(messages: &mut [Value]) -> Result<(), String> {
    for message in messages {
        let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) else {
            continue;
        };
        for call in calls {
            if let Some(args) = call.pointer_mut("/function/arguments") {
                if let Value::String(text) = args {
                    *args = serde_json::from_str(text)
                        .map_err(|e| format!("Failed to parse tool call arguments as JSON: {e}"))?;
                }
            }
        }
    }
    Ok(())
}

fn role(message: &Value) -> &str {
    message.get("role").and_then(Value::as_str).unwrap_or("")
}

fn text(message: &Value) -> &str {
    message.get("content").and_then(Value::as_str).unwrap_or("")
}

/// `system_message_not_supported`: the system text, a newline, and the
/// next message's text, or no system turn at all when it is the only
/// message.
fn fold_system(messages: &mut Vec<Value>) {
    if messages.first().map(role) != Some("system") {
        return;
    }
    let system = messages.remove(0);
    if let Some(next) = messages.first_mut() {
        let joined = format!("{}\n{}", text(&system), text(next));
        next["content"] = Value::String(joined);
    }
}

/// `common_chat_params_init_ministral_3`.
fn ministral3(messages: &mut [Value]) {
    for message in messages {
        if !matches!(role(message), "system" | "assistant") {
            continue;
        }
        let Some(obj) = message.as_object_mut() else {
            continue;
        };
        let mut blocks = Vec::new();
        if let Some(Value::String(thinking)) = obj.get("reasoning_content") {
            blocks.push(json!({"type": "thinking", "thinking": thinking}));
        }
        match obj.get("content") {
            Some(Value::String(text)) => blocks.push(json!({"type": "text", "text": text})),
            Some(Value::Array(parts)) => blocks.extend(parts.iter().cloned()),
            _ => {}
        }
        obj.insert("content".into(), Value::Array(blocks));
        // Upstream's message carries the reasoning once, and moves it.
        // frink's two other spellings of it go with it.
        for key in ["reasoning_content", "reasoning", "thinking"] {
            obj.shift_remove(key);
        }
    }
}

/// `common_chat_params_init_k2_horizon`.
fn k2_horizon(messages: &mut [Value]) {
    for message in messages {
        if role(message) == "assistant" {
            if let Some(obj) = message.as_object_mut() {
                obj.entry("reasoning_content")
                    .or_insert_with(|| Value::String(String::new()));
            }
        }
    }
}

fn has_tool_calls(message: &Value) -> bool {
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty())
}

/// `convert_tool_responses_gemma4`: an assistant turn with calls, the
/// `tool` turns after it, and the assistant answer after those (when it
/// has text and no calls of its own) become one assistant turn. The
/// first turn's own text is not kept, as upstream does not keep it.
fn gemma4_tool_responses(messages: &[Value]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    let mut i = 0;
    while i < messages.len() {
        let first = &messages[i];
        if role(first) != "assistant" || !has_tool_calls(first) {
            out.push(first.clone());
            i += 1;
            continue;
        }
        let calls = first["tool_calls"].as_array().cloned().unwrap_or_default();
        let reasoning = first
            .get("reasoning_content")
            .filter(|r| r.is_string())
            .cloned();
        i += 1;
        let mut responses = Vec::new();
        while i < messages.len() && role(&messages[i]) == "tool" {
            let result = &messages[i];
            let response = match result.get("content") {
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!(s)),
                Some(other) => other.clone(),
                None => Value::Null,
            };
            let name = calls
                .get(responses.len())
                .and_then(|call| call.pointer("/function/name"))
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())
                .or_else(|| result.get("tool_call_id").and_then(Value::as_str))
                .unwrap_or("");
            responses.push(json!({"name": name, "response": response}));
            i += 1;
        }
        let mut content = None;
        if let Some(next) = messages.get(i) {
            let has_content = match next.get("content") {
                Some(Value::String(s)) => !s.is_empty(),
                Some(Value::Array(parts)) => !parts.is_empty(),
                _ => false,
            };
            if role(next) == "assistant" && !has_tool_calls(next) && has_content {
                content = next.get("content").cloned();
                i += 1;
            }
        }
        let mut turn = Map::new();
        turn.insert("role".into(), json!("assistant"));
        turn.insert("tool_calls".into(), Value::Array(calls));
        if !responses.is_empty() {
            turn.insert("tool_responses".into(), Value::Array(responses));
        }
        if let Some(content) = content {
            turn.insert("content".into(), content);
        }
        if let Some(reasoning) = reasoning {
            turn.insert("reasoning_content".into(), reasoning);
        }
        out.push(Value::Object(turn));
    }
    out
}

/// `common_chat_params_init_translate_gemma`: the language codes come
/// from the kwargs, `en-GB` when a request names neither.
fn translate_gemma(messages: &mut [Value], kwargs: &Map<String, Value>) -> Result<(), String> {
    let lang = |key: &str| match kwargs.get(key) {
        None => Ok("en-GB".to_string()),
        Some(Value::String(code)) => Ok(code.clone()),
        Some(other) => Err(format!("{key} must be a string, not {other}")),
    };
    let (source, target) = (lang("source_lang_code")?, lang("target_lang_code")?);
    for message in messages {
        if role(message) != "user" {
            continue;
        }
        let text = match message.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
                }
                text
            }
            _ => String::new(),
        };
        message["content"] = json!([{
            "type": "text",
            "text": text,
            "source_lang_code": source,
            "target_lang_code": target,
        }]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reshaped(
        family: Option<Family>,
        fold_system: bool,
        messages: Value,
        kwargs: Value,
    ) -> Value {
        let (Value::Array(mut messages), Value::Object(kwargs)) = (messages, kwargs) else {
            panic!("messages are a list, kwargs a map")
        };
        Reshape {
            fold_system,
            object_arguments: false,
            family,
        }
        .apply(&mut messages, &kwargs)
        .expect("reshapes");
        Value::Array(messages)
    }

    /// Upstream's markers, and the two near misses it tells apart:
    /// Mistral Small 3.2 has `[CALL_ID]`, and a current Gemma 4 template
    /// renders `role: tool` itself.
    #[test]
    fn a_family_is_detected_from_its_markers() {
        let ministral = "[SYSTEM_PROMPT][TOOL_CALLS][ARGS]";
        assert_eq!(Family::detect(ministral), Some(Family::Ministral3));
        assert_eq!(Family::detect(&format!("{ministral}[CALL_ID]")), None);
        assert_eq!(
            Family::detect("<|ifm|im_start|><ifm|tool_calls>"),
            Some(Family::K2Horizon)
        );
        let gemma4 = "'<|tool_call>call:'";
        assert_eq!(Family::detect(gemma4), Some(Family::Gemma4Legacy));
        assert_eq!(
            Family::detect(&format!("{gemma4}{{#- OpenAI Chat Completions:")),
            None
        );
        assert_eq!(
            Family::detect("[source_lang_code][target_lang_code]"),
            Some(Family::TranslateGemma)
        );
        assert_eq!(Family::detect("{{ messages }}"), None);
    }

    /// The system text joins the next message's with a newline, and a
    /// system turn with nothing after it goes.
    #[test]
    fn a_system_turn_folds_into_the_next_message() {
        let folded = reshaped(
            None,
            true,
            json!([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]),
            json!({}),
        );
        assert_eq!(folded, json!([{"role": "user", "content": "S\nU"}]));
        let alone = reshaped(
            None,
            true,
            json!([{"role": "system", "content": "S"}]),
            json!({}),
        );
        assert_eq!(alone, json!([]));
        let kept = json!([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]);
        assert_eq!(reshaped(None, false, kept.clone(), json!({})), kept);
    }

    #[test]
    fn ministral_turns_reasoning_and_text_into_blocks() {
        let out = reshaped(
            Some(Family::Ministral3),
            false,
            json!([
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "4.", "reasoning_content": "why",
                 "reasoning": "why", "thinking": "why"},
                {"role": "assistant", "content": ""},
            ]),
            json!({}),
        );
        assert_eq!(
            out,
            json!([
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "why"},
                    {"type": "text", "text": "4."},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ])
        );
    }

    #[test]
    fn k2_horizon_gives_every_assistant_turn_a_reasoning_field() {
        let out = reshaped(
            Some(Family::K2Horizon),
            false,
            json!([
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "assistant", "content": "b", "reasoning_content": "r"},
            ]),
            json!({}),
        );
        assert_eq!(out[0], json!({"role": "user", "content": "q"}));
        assert_eq!(out[1]["reasoning_content"], json!(""));
        assert_eq!(out[2]["reasoning_content"], json!("r"));
    }

    /// One turn per call, its results named after the call they answer
    /// (or by id past the last call), a JSON result parsed, and the
    /// answer after them as the turn's content. The calling turn's own
    /// text goes, with an answer after it or without one.
    #[test]
    fn gemma4_folds_a_call_its_results_and_the_answer_into_one_turn() {
        let out = reshaped(
            Some(Family::Gemma4Legacy),
            false,
            json!([
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "dropped", "reasoning_content": "r",
                 "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {}}}]},
                {"role": "tool", "content": "Sunny", "tool_call_id": "c0"},
                {"role": "tool", "content": "{\"t\": 1}", "tool_call_id": "c1"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "again"},
                {"role": "assistant", "content": "also dropped",
                 "tool_calls": [{"type": "function", "function": {"name": "g", "arguments": {}}}]},
                {"role": "tool", "content": "x"},
            ]),
            json!({}),
        );
        assert_eq!(
            out,
            json!([
                {"role": "user", "content": "q"},
                {"role": "assistant",
                 "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {}}}],
                 "tool_responses": [
                     {"name": "f", "response": "Sunny"},
                     {"name": "c1", "response": {"t": 1}},
                 ],
                 "content": "done",
                 "reasoning_content": "r"},
                {"role": "user", "content": "again"},
                {"role": "assistant",
                 "tool_calls": [{"type": "function", "function": {"name": "g", "arguments": {}}}],
                 "tool_responses": [{"name": "g", "response": "x"}]},
            ])
        );
    }

    fn reads_arguments(source: &str) -> bool {
        ChatTemplate::from_jinja(source)
            .map(|t| Reshape::for_template(&t, None, None).object_arguments)
            .unwrap()
    }

    /// Reading a member, walking the items or `tojson` reads an object.
    /// Concatenating the arguments, ignoring them, or a render that
    /// fails does not. A template that insists on a reasoning field gets
    /// the second try upstream gives it.
    #[test]
    fn a_template_reads_object_arguments_when_their_value_reaches_the_prompt() {
        let each_call = |body: &str| {
            format!("{{% for m in messages %}}{{% for c in m.tool_calls %}}{body}{{% endfor %}}{{% endfor %}}")
        };
        assert!(reads_arguments(&each_call(
            "{{ c.function.arguments.arg }}"
        )));
        assert!(reads_arguments(&each_call(
            "{% for k, v in c.function.arguments.items() %}{{ v }}{% endfor %}"
        )));
        assert!(reads_arguments(&each_call(
            "{{ c.function.arguments | tojson }}"
        )));
        assert!(!reads_arguments(&each_call(
            "{{ '>>>' + c.function.arguments }}"
        )));
        assert!(!reads_arguments(&each_call("{{ c.function.name }}")));
        let insists = "{% for m in messages %}{% if m.role == 'assistant' and m.reasoning_content is not defined %}\
                       {{ raise_exception('missing a thinking field') }}{% endif %}\
                       {% for c in m.tool_calls %}{{ c.function.arguments | tojson }}{% endfor %}{% endfor %}";
        assert!(reads_arguments(insists));
    }

    /// The wire string becomes an object where the template reads one,
    /// a string that is not JSON is refused as upstream refuses it, and
    /// every other template keeps the string.
    #[test]
    fn arguments_are_parsed_only_for_a_template_that_reads_an_object() {
        let call = |args: &str| {
            json!([{"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": args}},
            ]}])
        };
        let parse = |object_arguments: bool, messages: Value| {
            let Value::Array(mut messages) = messages else {
                unreachable!()
            };
            Reshape {
                fold_system: false,
                object_arguments,
                family: None,
            }
            .apply(&mut messages, &Map::new())
            .map(|()| messages[0]["tool_calls"][0]["function"]["arguments"].clone())
        };
        assert_eq!(
            parse(true, call(r#"{"b": 1, "a": 2}"#)).unwrap(),
            json!({"b": 1, "a": 2})
        );
        assert_eq!(
            parse(false, call(r#"{"b": 1}"#)).unwrap(),
            json!(r#"{"b": 1}"#)
        );
        assert!(parse(true, call("not json")).is_err_and(|e| e.contains("as JSON")));
    }

    #[test]
    fn translategemma_types_each_user_turn_with_its_languages() {
        let messages =
            json!([{"role": "user", "content": "Hallo"}, {"role": "assistant", "content": "Hi"}]);
        let out = reshaped(
            Some(Family::TranslateGemma),
            false,
            messages.clone(),
            json!({"source_lang_code": "de"}),
        );
        assert_eq!(
            out[0]["content"],
            json!([{"type": "text", "text": "Hallo", "source_lang_code": "de", "target_lang_code": "en-GB"}])
        );
        assert_eq!(out[1], json!({"role": "assistant", "content": "Hi"}));
        let (Value::Array(mut messages), Value::Object(kwargs)) =
            (messages, json!({"target_lang_code": 7}))
        else {
            unreachable!()
        };
        let refused = Reshape {
            fold_system: false,
            object_arguments: false,
            family: Some(Family::TranslateGemma),
        }
        .apply(&mut messages, &kwargs);
        assert!(refused.is_err_and(|e| e.contains("target_lang_code")));
    }
}
