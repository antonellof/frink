//! Every chat template llama.cpp bundles at the pin, through frink's
//! Jinja engine.
//!
//! `chat_template_real_gguf.rs` holds byte goldens for fifteen families.
//! This holds the breadth: `models/templates/` at the pin is the corpus
//! upstream's own chat tests run on, and a template that does not render
//! here is a checkpoint that cannot chat. The first run of this found
//! plain chat failing on Qwen3-Coder and `tools` failing on Llama 3.x,
//! because the sweep over local GGUFs renders one plain conversation.
//!
//! `#[ignore]`d because the corpus lives in the llama.cpp checkout, which
//! `scripts/oracle.sh` has and CI's other jobs do not:
//!
//! ```text
//! FRINK_LLAMA_TEMPLATES=<llama.cpp>/models/templates \
//!   cargo test -p frink-models --test upstream_template_census -- --ignored --nocapture
//! ```
//!
//! The census proves a template renders, never that the bytes match.
//! [`upstream_templates_render_byte_for_byte_like_jinja2`] does that, on
//! every PR, for one template per construct in `tests/templates/upstream/`.

use frink_models::chat_template::reshape::Reshape;
use frink_models::chat_template::{ChatTemplate, RenderOptions};
use serde_json::{json, Map, Value};

/// A failure the census expects: the template, and the text of that one
/// failure. A template listed here is still held to every other shape,
/// and a row nothing matches any more fails the test.
const EXPECTED_REFUSALS: &[(&str, &str)] = &[
    // The template's own `raise_exception` on a shape it refuses, after
    // the reshape, which llama.cpp raises too: a tool turn where the
    // roles must alternate.
    ("google-gemma-2-2b-it.jinja", "roles must alternate"),
    (
        "google-translategemma-4b-it.jinja",
        "Conversation roles must alternate",
    ),
    // `functions`, which no caller sets, added to a string: llama.cpp's
    // engine reads it as empty and minijinja raises, a deviation
    // `docs/CLI.md` lists (#522).
    (
        "fireworks-ai-llama-3-firefunction-v2.jinja",
        "+ operator on unsupported types string and undefined",
    ),
];

fn opts(tools: Vec<Value>) -> RenderOptions {
    RenderOptions {
        add_generation_prompt: true,
        bos_token: Some("<s>".to_string()),
        eos_token: Some("</s>".to_string()),
        tools,
        extra: Map::new(),
        // These hold the engine to jinja2, which passes no such default.
        enable_thinking: None,
    }
}

fn weather_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "The city"}},
                "required": ["city"],
            },
        },
    })
}

/// Every conversation shape a server hands a template.
fn conversations() -> Vec<(&'static str, Vec<Value>, Vec<Value>)> {
    let plain = vec![
        json!({"role": "system", "content": "Be brief."}),
        json!({"role": "user", "content": "Hi"}),
        json!({"role": "assistant", "content": "Hello"}),
        json!({"role": "user", "content": "How are you?"}),
    ];
    let reasoning = vec![
        json!({"role": "user", "content": "Hi"}),
        json!({"role": "assistant", "content": "Hello", "reasoning_content": "They said hi"}),
        json!({"role": "user", "content": "How are you?"}),
    ];
    let tool_round_trip = vec![
        json!({"role": "user", "content": "Weather in Rome?"}),
        json!({"role": "assistant", "content": "", "tool_calls": [{
            "id": "call00001",
            "type": "function",
            // The wire's JSON string. The reshape parses it for a
            // template that reads an object, as a front end does.
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"},
        }]}),
        json!({"role": "tool", "tool_call_id": "call00001", "name": "get_weather", "content": "Sunny"}),
        json!({"role": "assistant", "content": "It is sunny."}),
        json!({"role": "user", "content": "And tomorrow?"}),
    ];
    vec![
        ("plain", plain.clone(), Vec::new()),
        ("reasoning", reasoning, Vec::new()),
        ("tools offered", plain, vec![weather_tool()]),
        ("tool round trip", tool_round_trip, vec![weather_tool()]),
    ]
}

/// Everything that failed on one template, after the reshape both front
/// ends apply ([`Reshape`]), so a shape the front ends never send does
/// not count as a failure.
fn failures(source: &str) -> Vec<String> {
    let template = match ChatTemplate::from_jinja(source) {
        Ok(t) => t,
        Err(e) => return vec![format!("compile: {e}")],
    };
    let reshape = Reshape::for_template(&template, Some("<s>"), Some("</s>"));
    conversations()
        .into_iter()
        .filter_map(|(shape, mut messages, tools)| {
            let rendered = reshape
                .apply(&mut messages, &Map::new())
                .map_err(|e| format!("reshape: {e}"))
                .and_then(|()| {
                    template
                        .render(&messages, &opts(tools))
                        .map_err(|e| e.to_string())
                });
            rendered.err().map(|e| format!("{shape}: {e}"))
        })
        .collect()
}

#[test]
#[ignore = "needs llama.cpp's models/templates: set FRINK_LLAMA_TEMPLATES"]
fn every_template_llama_cpp_bundles_renders_every_conversation_shape() {
    let dir = std::env::var("FRINK_LLAMA_TEMPLATES").expect("FRINK_LLAMA_TEMPLATES is not set");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "jinja"))
        .collect();
    paths.sort();
    assert!(
        paths.len() > 50,
        "{dir} holds {} templates: a census of a wrong directory proves nothing",
        paths.len()
    );

    let mut report = String::new();
    let mut matched = vec![false; EXPECTED_REFUSALS.len()];
    for path in &paths {
        let name = path.file_name().unwrap().to_str().unwrap();
        let source = std::fs::read_to_string(path).unwrap();
        for failure in failures(&source) {
            let expected = EXPECTED_REFUSALS
                .iter()
                .position(|(n, text)| *n == name && failure.contains(text));
            match expected {
                Some(row) => matched[row] = true,
                None => report.push_str(&format!("{name}\n    {failure}\n")),
            }
        }
    }
    println!("{} templates\n{report}", paths.len());
    assert!(report.is_empty(), "templates that do not render:\n{report}");
    let stale: Vec<_> = EXPECTED_REFUSALS
        .iter()
        .zip(&matched)
        .filter(|(_, hit)| !**hit)
        .map(|(row, _)| row)
        .collect();
    assert!(
        stale.is_empty(),
        "no failure matches these EXPECTED_REFUSALS rows, so remove them: {stale:?}"
    );
}

/// Kept byte-identical with `scripts/chat_template_goldens.py`.
fn golden_opts(tools: Vec<Value>) -> RenderOptions {
    let mut extra = Map::new();
    extra.insert("date_string".to_string(), json!("26 Jul 2024"));
    extra.insert("current_date".to_string(), json!("2024-07-26"));
    RenderOptions {
        extra,
        ..opts(tools)
    }
}

/// `scripts/chat_template_goldens.py`'s scenarios. The tool's keys are in
/// sorted order at every level, which that script explains.
fn golden_scenarios() -> Vec<(&'static str, Vec<Value>, Vec<Value>)> {
    let conversation = vec![
        json!({"role": "user", "content": "What is the capital of France?"}),
        json!({"role": "assistant", "content": "Paris."}),
        json!({"role": "user", "content": "And of Italy?"}),
    ];
    let system = vec![
        json!({"role": "system", "content": "Answer with one word."}),
        json!({"role": "user", "content": "What is the capital of France?"}),
    ];
    let tool = json!({
        "function": {
            "description": "The weather in a city",
            "name": "get_weather",
            "parameters": {
                "properties": {"city": {"description": "La citt\u{e0}", "type": "string"}},
                "required": ["city"],
                "type": "object",
            },
        },
        "type": "function",
    });
    let round_trip = vec![
        json!({"role": "user", "content": "Weather in Rome?"}),
        json!({"role": "assistant", "content": "", "tool_calls": [{
            "function": {"arguments": {"city": "Rome"}, "name": "get_weather"},
            "id": "call00001",
            "type": "function",
        }]}),
        json!({"role": "tool", "tool_call_id": "call00001", "name": "get_weather", "content": "Sunny"}),
        json!({"role": "assistant", "content": "It is sunny."}),
        json!({"role": "user", "content": "And tomorrow?"}),
    ];
    vec![
        ("", conversation.clone(), Vec::new()),
        (".system", system, Vec::new()),
        (".tools", conversation, vec![tool.clone()]),
        (".round_trip", round_trip, vec![tool]),
    ]
}

/// Each template here is the only one in the corpus reaching a construct
/// minijinja lacks: `tojson(indent=)` (Llama 3.2), `tojson(ensure_ascii=)`
/// (GLM), `.items()` (DeepSeek), `is iterable` on `none` (Qwen3-Coder),
/// `.append` and `.pop` (Kimi K2), `{% generation %}` (LFM3), `.replace`
/// (Nemotron), `.format` (Hy3) and an inline `if` in a keyword argument
/// (MUSE). Goldens are from real jinja2: regenerate with
/// `python3 scripts/chat_template_goldens.py`, never from a failure.
#[test]
fn upstream_templates_render_byte_for_byte_like_jinja2() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/templates/upstream");
    let mut wrong = Vec::new();
    let mut compared = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "jinja") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let source = std::fs::read_to_string(&path).unwrap();
        let template = ChatTemplate::from_jinja(&source)
            .unwrap_or_else(|e| panic!("{stem} did not compile: {e}"));
        for (suffix, messages, tools) in golden_scenarios() {
            let golden = dir.join(format!("{stem}{suffix}.expected"));
            let want =
                std::fs::read_to_string(&golden).unwrap_or_else(|e| panic!("{golden:?}: {e}"));
            compared += 1;
            match template.render(&messages, &golden_opts(tools)) {
                Ok(got) if got == want => {}
                Ok(got) => wrong.push(format!("{stem}{suffix}\n  want {want:?}\n  got  {got:?}")),
                Err(e) => wrong.push(format!("{stem}{suffix}: refused: {e}")),
            }
        }
    }
    assert_eq!(compared, 9 * 4, "a template or a scenario went missing");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
