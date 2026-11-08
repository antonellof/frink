//! The chat-template flags `frink run` and `frink-server` share, read
//! from llama.cpp's `common/arg.cpp`: `--chat-template`,
//! `--chat-template-file`, `--chat-template-kwargs`, `-rea` and
//! `--reasoning-effort`.
//!
//! One resolver, so the two entry points cannot disagree about what a
//! value means. Everything is checked here, before a model loads, as
//! llama.cpp checks it while parsing (`common_chat_verify_template`).

use std::path::Path;

use serde_json::{json, Map, Value};

use crate::chat_template::{BuiltinTemplate, ChatTemplate, RenderOptions};

/// The flags as given. `reasoning` is `-rea` already read: `Some(true)`
/// on, `Some(false)` off, `None` auto or absent.
#[derive(Debug, Default, Clone, Copy)]
pub struct TemplateFlagValues<'a> {
    pub template: Option<&'a str>,
    pub template_file: Option<&'a Path>,
    pub kwargs: &'a [String],
    pub reasoning: Option<bool>,
    pub reasoning_effort: Option<&'a str>,
}

/// What the flags mean.
#[derive(Debug, Clone, Default)]
pub struct TemplateFlags {
    /// The template text as given, which the server carries to its loaders.
    pub source: Option<String>,
    /// `source`, compiled and verified.
    pub template: Option<ChatTemplate>,
    /// Defaults for every render. A request's own keys win over these.
    pub kwargs: Map<String, Value>,
}

impl TemplateFlags {
    /// An empty template, or an empty file, is no template, as in
    /// llama.cpp. `--reasoning-effort` applies after
    /// `--chat-template-kwargs` whatever their order on the command line.
    pub fn resolve(values: TemplateFlagValues<'_>) -> Result<Self, String> {
        let source = match (values.template, values.template_file) {
            (Some(template), _) => Some(template.to_string()),
            (None, Some(path)) => Some(
                std::fs::read_to_string(path)
                    .map_err(|e| format!("--chat-template-file {}: {e}", path.display()))?,
            ),
            (None, None) => None,
        }
        .filter(|source| !source.is_empty());
        let template = source.as_deref().map(compile).transpose()?;

        let mut kwargs = Map::new();
        for value in values.kwargs {
            kwargs.extend(parse_kwargs(value)?);
        }
        if let Some(on) = values.reasoning {
            kwargs.insert("enable_thinking".to_string(), json!(on));
        }
        match values.reasoning_effort {
            None => {}
            Some("default") => {
                kwargs.remove("reasoning_effort");
            }
            Some(level) => {
                kwargs.insert("reasoning_effort".to_string(), json!(level));
            }
        }
        Ok(Self {
            source,
            template,
            kwargs,
        })
    }

    /// Whether the flags change anything.
    pub fn is_empty(&self) -> bool {
        self.source.is_none() && self.kwargs.is_empty()
    }
}

/// `chatml` is the one name llama.cpp special-cases in Jinja mode
/// (`common_chat_templates_init`). Any other value is compiled as Jinja
/// there, its default mode, so `--chat-template llama3` passes its check
/// and renders the word `llama3` as every prompt. Refused here: frink has
/// no named templates to fall back to, and no `--no-jinja` to reach them.
fn compile(value: &str) -> Result<ChatTemplate, String> {
    if value == "chatml" {
        return Ok(ChatTemplate::builtin(BuiltinTemplate::ChatMl));
    }
    if !value.contains("{{") && !value.contains("{%") {
        let head: String = value.chars().take(40).collect();
        return Err(format!(
            "--chat-template {head:?} is not Jinja source. frink's only named template is \
             chatml. Pass the template's text, or --chat-template-file"
        ));
    }
    let template = ChatTemplate::from_jinja(value).map_err(|e| format!("--chat-template: {e}"))?;
    // `common_chat_verify_template`: one user turn, with no model yet, so
    // the special tokens are empty strings.
    let opts = RenderOptions {
        add_generation_prompt: true,
        bos_token: Some(String::new()),
        eos_token: Some(String::new()),
        ..Default::default()
    };
    template
        .render(&[json!({"role": "user", "content": "test"})], &opts)
        .map_err(|e| format!("--chat-template: {e}"))?;
    Ok(template)
}

fn parse_kwargs(value: &str) -> Result<Map<String, Value>, String> {
    let parsed: Value =
        serde_json::from_str(value).map_err(|e| format!("--chat-template-kwargs: {e}"))?;
    let Value::Object(kwargs) = parsed else {
        return Err("--chat-template-kwargs: expected a JSON object".to_string());
    };
    if kwargs.contains_key("preserve_reasoning") {
        return Err(
            "--chat-template-kwargs: preserve_reasoning is not honoured yet. frink does not set \
             the template variables llama.cpp derives from it (#508)"
                .to_string(),
        );
    }
    Ok(kwargs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve(values: TemplateFlagValues<'_>) -> Result<TemplateFlags, String> {
        TemplateFlags::resolve(values)
    }

    fn template(value: &str) -> Result<TemplateFlags, String> {
        resolve(TemplateFlagValues {
            template: Some(value),
            ..Default::default()
        })
    }

    fn kwargs(values: &[&str]) -> Result<Map<String, Value>, String> {
        let values: Vec<String> = values.iter().map(|v| v.to_string()).collect();
        resolve(TemplateFlagValues {
            kwargs: &values,
            ..Default::default()
        })
        .map(|flags| flags.kwargs)
    }

    fn render(flags: &TemplateFlags) -> String {
        let opts = RenderOptions {
            add_generation_prompt: true,
            ..Default::default()
        };
        let messages = [json!({"role": "user", "content": "hi"})];
        let template = flags.template.as_ref().unwrap();
        template.render(&messages, &opts).unwrap()
    }

    #[test]
    fn chatml_is_the_builtin_and_other_source_is_compiled() {
        assert_eq!(
            render(&template("chatml").unwrap()),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        let flags = template("[{{ messages[0].content }}]").unwrap();
        assert_eq!(render(&flags), "[hi]");
        assert_eq!(flags.source.as_deref(), Some("[{{ messages[0].content }}]"));
    }

    /// `llama3` compiles as plain text upstream, and every prompt would be
    /// that word. A silent wrong prompt is what a refusal is for.
    #[test]
    fn a_bare_name_is_refused_by_name() {
        let error = template("llama3").unwrap_err();
        assert!(
            error.contains("llama3") && error.contains("chatml"),
            "{error}"
        );
    }

    /// `common_chat_verify_template` fails at startup on both of these,
    /// so neither waits for the first request.
    #[test]
    fn a_template_that_will_not_compile_or_render_is_refused_at_once() {
        assert!(template("{% if %}").is_err());
        let error = template("{{ raise_exception('no') }}").unwrap_err();
        assert!(error.contains("no"), "{error}");
    }

    #[test]
    fn an_empty_template_is_no_template() {
        assert!(template("").unwrap().is_empty());
        let path = std::env::temp_dir().join(format!("frink-tf-empty-{}", std::process::id()));
        std::fs::write(&path, "").unwrap();
        let flags = resolve(TemplateFlagValues {
            template_file: Some(&path),
            ..Default::default()
        });
        std::fs::remove_file(&path).unwrap();
        assert!(flags.unwrap().is_empty());
    }

    #[test]
    fn a_template_file_is_read_and_a_missing_one_named() {
        let path = std::env::temp_dir().join(format!("frink-tf-{}", std::process::id()));
        std::fs::write(&path, "<{{ messages[0].role }}>").unwrap();
        let flags = resolve(TemplateFlagValues {
            template_file: Some(&path),
            ..Default::default()
        })
        .unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(render(&flags), "<user>");

        let error = resolve(TemplateFlagValues {
            template_file: Some(&path),
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.contains(path.to_str().unwrap()), "{error}");
    }

    #[test]
    fn kwargs_merge_with_the_later_flag_winning() {
        let merged = kwargs(&[r#"{"a": 1, "b": true}"#, r#"{"a": 2}"#]).unwrap();
        assert_eq!(merged["a"], json!(2));
        assert_eq!(merged["b"], json!(true));
    }

    #[test]
    fn kwargs_must_be_a_json_object() {
        for bad in ["not json", "[1]", "3"] {
            assert!(kwargs(&[bad]).is_err(), "{bad} was accepted");
        }
    }

    /// Accepted and inert would be a flag not honoured (ADR 0005).
    #[test]
    fn preserve_reasoning_is_refused_until_it_is_honoured() {
        let error = kwargs(&[r#"{"preserve_reasoning": false}"#]).unwrap_err();
        assert!(error.contains("#508"), "{error}");
    }

    #[test]
    fn reasoning_sets_enable_thinking_and_auto_leaves_it_alone() {
        for on in [true, false] {
            let flags = resolve(TemplateFlagValues {
                reasoning: Some(on),
                ..Default::default()
            });
            assert_eq!(flags.unwrap().kwargs["enable_thinking"], json!(on));
        }
        let values = [r#"{"enable_thinking": true}"#.to_string()];
        let flags = resolve(TemplateFlagValues {
            kwargs: &values,
            reasoning: None,
            ..Default::default()
        });
        assert_eq!(flags.unwrap().kwargs["enable_thinking"], json!(true));
    }

    #[test]
    fn reasoning_effort_overrides_kwargs_and_default_erases() {
        let values = [r#"{"reasoning_effort": "low"}"#.to_string()];
        let with = |effort| {
            resolve(TemplateFlagValues {
                kwargs: &values,
                reasoning_effort: Some(effort),
                ..Default::default()
            })
            .unwrap()
            .kwargs
        };
        assert_eq!(with("high")["reasoning_effort"], json!("high"));
        assert!(!with("default").contains_key("reasoning_effort"));
    }
}
