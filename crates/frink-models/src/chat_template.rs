//! Chat-template rendering driven by the GGUF's own
//! `tokenizer.chat_template` Jinja2 string.
//!
//! # Why this exists
//!
//! The previous implementation (`frink-server`'s `chat_template.rs`, and
//! a near-identical copy in `frink-cli`'s `run.rs`) sniffed the template
//! string for literal markers — `<|im_start|>`, `<|start_header_id|>`,
//! `<start_of_turn>` — and picked one of six hand-written renderers. That
//! has three failure modes, all of them silent:
//!
//! 1. **Every unrecognised family renders as `Plain`.** Mistral-Instruct's
//!    real template is `[INST] … [/INST]`, which matches no marker, so a
//!    Mistral checkpoint was served `user: hi` — a prompt shape it has
//!    never seen. Same for Phi-3/Phi-4 (`<|user|>…<|end|>` uses the
//!    `<|user|>` marker but not the `</s>`-terminated framing the
//!    `GenericRoleMarkers` renderer emits), Yi, and DeepSeek-R1.
//! 2. **The tool-calling half of every template is unreachable.** No
//!    hand-written renderer ever consulted `tools`, so the `<tool_call>`
//!    / `<|tool▁calls▁begin|>` / Gemma `<|tool>` grammars a model was
//!    actually trained on could not be produced.
//! 3. **A recognised family is not the same as an implemented one.**
//!    `ChatTemplate::Gemma4` matched gemma-4's `<|turn>` marker and then
//!    rendered a three-line approximation of an 18 KB template: no
//!    thinking-channel injection, no `strip_thinking` on replayed
//!    assistant turns, no multimodal placeholders, no tool blocks.
//!
//! So this module evaluates the template instead of recognising it.
//! [`ChatTemplate::from_gguf_metadata`] compiles the checkpoint's own
//! Jinja source with [`minijinja`]; the hand-written renderers survive
//! only as [`BuiltinTemplate`], used for checkpoints that ship **no**
//! template at all (llama.cpp's `--jinja` does the same thing, defaulting
//! to ChatML) and for the server's synthetic-weights demo path.
//!
//! # Failing loudly
//!
//! A template that does not compile, or that uses a filter/test/function
//! this evaluator does not provide, produces a [`TemplateError`] that
//! propagates to the caller as a request failure. It does **not** fall
//! back to a hand-written renderer: silently serving a Mistral checkpoint
//! ChatML framing is exactly the class of bug this module exists to
//! delete, and the repo's rule is to land the refusal when the math is
//! not there. The one thing that *is* a fallback is "the checkpoint
//! carries no template", which is a genuine absence rather than a
//! guess.
//!
//! Known Jinja constructs and how they are handled:
//!
//! | Construct | Status |
//! |---|---|
//! | `{%- … -%}` whitespace control | supported by minijinja |
//! | `{% macro %}` / recursive macros | supported |
//! | `{% set ns = namespace(...) %}` and loop-scope writes through it | supported |
//! | `{% set x %}…{% endset %}` block set | supported |
//! | `loop.index0` / `loop.last` / `loop.first` | supported |
//! | `raise_exception(msg)` | provided here; aborts the render with the message |
//! | `strftime_now(fmt)` | provided here, local time as llama.cpp's, subset of `strftime` (see [`strftime_now`]) |
//! | `enable_thinking`, `date_string`, `datetime`, `reasoning_strength`, `preserve_thinking` and its kin | set under the kwargs, as every llama.cpp render sets them (see [`llama_defaults`]) |
//! | `dictsort`, `map`, `default`, `trim`, `reject`, `join`, slicing | minijinja builtins |
//! | `tojson` | reimplemented, with `indent`, `ensure_ascii` and `separators`; keys in the order they were written (see [`py_compat::tojson`]) |
//! | `{{ value }}`, `value \| string` | a list or a map in Python's literal syntax, as llama.cpp prints it (see [`py_compat::formatter`]) |
//! | `is iterable`, `{% generation %}`, a `namespace(...)` list that `.append`s | [`py_compat`] |
//! | Python methods `.get()`, `.split()`, `.strip()`, `.items()`, `.replace()`, `.format()` | provided here (see [`python_method`]) |
//! | anything else | **hard error**, never silently empty |
//!
//! The one deliberate difference from `jinja2`: undefined variables are
//! *lenient* (falsy in a condition, empty when printed) rather than
//! `StrictUndefined`, because that is what HuggingFace's
//! `apply_chat_template` uses and templates rely on it — e.g. gemma-3
//! tests `{%- if add_generation_prompt -%}` without the caller having to
//! define it.

use std::sync::Arc;

use minijinja::value::Value as JinjaValue;
use serde_json::Value;

mod llama_defaults;
mod py_compat;
pub mod reshape;

/// Everything that can go wrong between a GGUF's template string and a
/// rendered prompt. Every variant is a refusal, not a fallback.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TemplateError {
    /// `tokenizer.chat_template` is not valid Jinja2, or uses syntax
    /// minijinja does not parse.
    #[error("chat template does not compile: {0}")]
    Compile(String),
    /// The template compiled but the render failed: an unknown filter,
    /// test or function; an explicit `raise_exception(...)`; a type
    /// error inside the template.
    #[error("chat template failed to render: {0}")]
    Render(String),
}

/// Hand-written renderers, kept only for checkpoints that ship no
/// `tokenizer.chat_template` at all.
///
/// These are the six variants the sniffing implementation used. They are
/// no longer *selected* by sniffing — a checkpoint either has a template
/// (and it is evaluated) or it does not (and [`BuiltinTemplate::ChatMl`]
/// or [`BuiltinTemplate::Plain`] applies, matching llama.cpp `--jinja`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinTemplate {
    /// `<|im_start|>{role}\n{content}<|im_end|>\n`, ending with
    /// `<|im_start|>assistant\n`. llama.cpp's `CHATML_TEMPLATE_SRC`
    /// default for a real checkpoint with no template of its own.
    ChatMl,
    /// `{role}: {content}` lines, no special tokens — for byte/synthetic
    /// tokenizers where no real vocabulary exists to carry markers.
    Plain,
}

enum Kind {
    /// The checkpoint's own template, compiled.
    Jinja(Box<JinjaTemplate>),
    /// The checkpoint's own template, which did not compile. Kept as an
    /// error rather than replaced by a guess, so the failure surfaces at
    /// the request instead of as wrong-looking output.
    Broken(TemplateError),
    /// No template in the checkpoint.
    Builtin(BuiltinTemplate),
}

struct JinjaTemplate {
    env: minijinja::Environment<'static>,
    source: String,
}

/// A compiled chat template. Cheap to clone (`Arc` inside) because every
/// `*Loaded` struct in `frink-server` carries one and hands it to each
/// request.
#[derive(Clone)]
pub struct ChatTemplate(Arc<Kind>);

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &*self.0 {
            Kind::Jinja(t) => write!(f, "Jinja({} bytes)", t.source.len()),
            Kind::Broken(e) => write!(f, "Broken({e})"),
            Kind::Builtin(b) => write!(f, "Builtin({b:?})"),
        }
    }
}

/// Everything a template can read besides `messages`.
///
/// `extra` is the OpenAI-extension `chat_template_kwargs` passthrough:
/// whatever the client puts there becomes a top-level template variable,
/// which is how `enable_thinking` (Qwen3, gemma-4), `thinking`
/// (DeepSeek), and `preserve_thinking` are actually driven. Values in
/// `extra` never shadow `messages`/`tools`/`add_generation_prompt`.
#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    pub add_generation_prompt: bool,
    /// The vocabulary's BOS text (`<s>`, `<|begin_of_text|>`, `<bos>`).
    /// Templates that print `{{ bos_token }}` own BOS insertion; see
    /// `frink-server`'s `generate` for why that does not double-add.
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    /// OpenAI `tools` array, verbatim. Passed to every template; only
    /// templates that mention `tools` do anything with it (see
    /// [`ChatTemplate::handles_tools`]).
    pub tools: Vec<Value>,
    pub extra: serde_json::Map<String, Value>,
    /// llama.cpp's `inputs.enable_thinking`: passed to every render under
    /// a kwarg of the same name. llama-server and llama-cli set it to
    /// whether the template supports thinking
    /// ([`ChatTemplate::supports_thinking`]); `None` leaves it undefined.
    pub enable_thinking: Option<bool>,
}

impl ChatTemplate {
    /// Compiles `source` as Jinja2. Errors are returned, never swallowed.
    pub fn from_jinja(source: &str) -> Result<Self, TemplateError> {
        let mut env = new_environment();
        env.add_template_owned(
            "chat".to_string(),
            py_compat::neutralise_generation_tags(source),
        )
        .map_err(|e| TemplateError::Compile(format_jinja_error(&e)))?;
        Ok(Self(Arc::new(Kind::Jinja(Box::new(JinjaTemplate {
            env,
            source: source.to_string(),
        })))))
    }

    pub fn builtin(b: BuiltinTemplate) -> Self {
        Self(Arc::new(Kind::Builtin(b)))
    }

    /// The load-time entry point: what a GGUF's metadata says.
    ///
    /// * a non-empty `tokenizer.chat_template` is compiled, and a compile
    ///   failure is *recorded* (so the load still succeeds and
    ///   `/v1/completions` still works) but makes every chat render fail
    ///   with the compiler's message;
    /// * no template + a real tokenizer ⇒ ChatML, matching llama.cpp
    ///   `--jinja`'s `CHATML_TEMPLATE_SRC` default;
    /// * no template + a byte/synthetic tokenizer ⇒ `Plain`, since there
    ///   is no real vocabulary for markers to live in.
    pub fn from_gguf_metadata(
        chat_template: Option<&str>,
        arch: Option<&str>,
        byte_tokenizer: bool,
        chatml_tokens_present: bool,
    ) -> Self {
        match chat_template.filter(|t| !t.trim().is_empty()) {
            Some(t) => match Self::from_jinja(t) {
                Ok(tmpl) => tmpl,
                Err(e) => Self(Arc::new(Kind::Broken(e))),
            },
            // No template, and no `<|im_start|>` / `<|im_end|>` in the
            // vocabulary to build one out of.
            //
            // Falling back to ChatML here is worse than useless, and
            // OLMoE-1B-7B is the case that showed it: its vocab has
            // neither marker, so they tokenize as literal text the model
            // never saw in training, `<|im_end|>` can never be GENERATED
            // and so can never stop anything, and the model imitates the
            // `<|...|>` pattern it is being shown. The observed result
            // was 512 tokens of invented `<|area_key|>`-style markers
            // where the raw completion path answers correctly.
            //
            // `Plain` already existed for "no real vocabulary for
            // markers to live in". This is the same condition, checked
            // properly instead of assumed from the tokenizer kind.
            None if byte_tokenizer || arch.is_none() || !chatml_tokens_present => {
                Self::builtin(BuiltinTemplate::Plain)
            }
            None => Self::builtin(BuiltinTemplate::ChatMl),
        }
    }

    /// Does this vocabulary actually contain the ChatML markers?
    ///
    /// Both are required: a checkpoint with one and not the other cannot
    /// frame a turn either.
    pub fn vocab_has_chatml(file: &impl frink_gguf::TensorSource) -> bool {
        let Some(frink_gguf::GgufValue::Array(tokens)) = file.metadata("tokenizer.ggml.tokens")
        else {
            return false;
        };
        let (mut start, mut end) = (false, false);
        for t in tokens {
            match t.as_str() {
                Some("<|im_start|>") => start = true,
                Some("<|im_end|>") => end = true,
                _ => {}
            }
            if start && end {
                return true;
            }
        }
        false
    }

    /// True when this is the checkpoint's own compiled template.
    pub fn is_jinja(&self) -> bool {
        matches!(&*self.0, Kind::Jinja(_))
    }

    /// The compiled Jinja source, for callers that need to inspect it.
    pub fn source(&self) -> Option<&str> {
        match &*self.0 {
            Kind::Jinja(t) => Some(&t.source),
            _ => None,
        }
    }

    /// Whether the template itself renders `tools`.
    ///
    /// Templates that never mention `tools` cannot express a tool call,
    /// so a caller offering tools to such a checkpoint has to fall back
    /// to describing them in a system message (`frink-server`'s
    /// `tool_preamble`). This is a textual check on the template source,
    /// which is what llama.cpp's `common/chat.cpp` does too
    /// (`caps.supports_tools` is probed by rendering, but the cheap
    /// source check is what gates it here).
    pub fn handles_tools(&self) -> bool {
        match &*self.0 {
            Kind::Jinja(t) => t.source.contains("tools"),
            Kind::Broken(_) | Kind::Builtin(_) => false,
        }
    }

    /// Short human-readable identity, for the load-time log line.
    pub fn describe(&self) -> String {
        match &*self.0 {
            Kind::Jinja(t) => format!("jinja ({} bytes from the GGUF)", t.source.len()),
            Kind::Broken(e) => format!("BROKEN: {e}"),
            Kind::Builtin(b) => format!("builtin {b:?} (checkpoint ships no chat template)"),
        }
    }

    /// Renders `messages` (OpenAI-shaped JSON objects) into a prompt.
    pub fn render(
        &self,
        messages: &[Value],
        opts: &RenderOptions,
    ) -> Result<String, TemplateError> {
        match &*self.0 {
            Kind::Broken(e) => Err(e.clone()),
            Kind::Builtin(b) => Ok(render_builtin(*b, messages, opts)),
            Kind::Jinja(t) => {
                let tmpl = t
                    .env
                    .get_template("chat")
                    .map_err(|e| TemplateError::Compile(format_jinja_error(&e)))?;
                let mut ctx = serde_json::Map::new();
                // `chat_template_kwargs` first, so it can never shadow the
                // structural variables below.
                for (k, v) in &opts.extra {
                    ctx.insert(k.clone(), v.clone());
                }
                // What every llama.cpp render carries beside the kwargs,
                // under them.
                llama_defaults::fill(&mut ctx, opts.enable_thinking)
                    .map_err(TemplateError::Render)?;
                ctx.insert("messages".into(), Value::Array(messages.to_vec()));
                ctx.insert(
                    "add_generation_prompt".into(),
                    Value::Bool(opts.add_generation_prompt),
                );
                // `tools` is always bound, as JSON `null` when the
                // request offered none. Leaving it *undefined* is a real
                // bug: Llama-3.1's template gates its whole ipython
                // tool-calling preamble on `{%- if tools is not none %}`,
                // and an undefined value is not none, so a plain chat
                // request got a tool-calling system prompt it never asked
                // for. HuggingFace's `apply_chat_template` passes
                // `tools=None` explicitly for the same reason.
                ctx.insert(
                    "tools".into(),
                    if opts.tools.is_empty() {
                        Value::Null
                    } else {
                        Value::Array(opts.tools.clone())
                    },
                );
                for (name, tok) in [
                    ("bos_token", &opts.bos_token),
                    ("eos_token", &opts.eos_token),
                ] {
                    if let Some(tok) = tok {
                        ctx.insert(name.into(), Value::String(tok.clone()));
                    }
                }
                tmpl.render(JinjaValue::from_serialize(Value::Object(ctx)))
                    .map_err(|e| TemplateError::Render(format_jinja_error(&e)))
            }
        }
    }
}

/// minijinja reports the interesting part of a failure in the *cause*
/// chain (an unknown filter, or a `raise_exception` message), and the
/// `Display` of the top error alone often reads as a bare
/// "invalid operation". Flatten the whole chain so a refusal names what
/// the template actually asked for.
fn format_jinja_error(err: &minijinja::Error) -> String {
    let mut out = err.to_string();
    if let Some(line) = err.line() {
        out.push_str(&format!(" (line {line})"));
    }
    let mut src = std::error::Error::source(err);
    while let Some(e) = src {
        out.push_str(&format!(": {e}"));
        src = std::error::Error::source(e);
    }
    out
}

fn new_environment() -> minijinja::Environment<'static> {
    let mut env = minijinja::Environment::new();
    // HuggingFace's `apply_chat_template` uses jinja2's default
    // `Undefined`, not `StrictUndefined`: templates freely test
    // `{% if add_generation_prompt %}` or `{% if tools %}` without the
    // caller defining them. `Lenient` is minijinja's equivalent —
    // undefined is falsy and prints empty, but any *operation* on it
    // (indexing, arithmetic, calling) is still an error.
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
    // The two whitespace flags a chat template is authored against, and
    // the only two settings in this function whose absence is *silent*.
    //
    // HuggingFace compiles every chat template with
    // `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`
    // and llama.cpp's own Jinja engine hardcodes the same pair for chat
    // templates (`common/jinja/lexer.cpp:112-118`: "default config for
    // chat template: lstrip_blocks = true, trim_blocks = true").
    // minijinja defaults both to `false`, matching stock jinja2 rather
    // than either engine that actually renders these strings.
    //
    // A template written with explicit `{%- … -%}` markers is unaffected,
    // which is why most of `tests/templates/` renders identically either
    // way and this went unnoticed. TinyLlama-1.1B-Chat's real template is
    // not written that way: without these two flags its three-turn render
    // is `\n\n<|user|>\n…</s>\n\n\n\n\n<|assistant|>\n…`, thirteen bytes of
    // stray blank line that the checkpoint was never trained on and that
    // llama.cpp does not emit. `whitespace_control_matches_huggingface_and_llama_cpp`
    // pins it.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // Real templates are one giant expression; the default recursion
    // limit is fine, but gemma-4's `format_parameters` recurses through
    // nested JSON schemas, so keep the default rather than lowering it.
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", strftime_now);
    env.add_function("namespace", py_compat::namespace);
    env.add_filter("tojson", py_compat::tojson);
    env.add_filter("string", py_compat::string);
    env.set_formatter(py_compat::formatter);
    env.add_test("iterable", py_compat::is_iterable);
    env.set_unknown_method_callback(python_method);
    env
}

/// jinja2 runs on Python, so templates call Python *methods* on the
/// values the caller passed in — `message.get('tool_calls')`,
/// `content.split('</think>')[-1].lstrip('\n')`. minijinja has no such
/// methods (they are not Jinja, they are Python leaking through), so
/// they arrive here.
///
/// This implements the five the real templates in `tests/templates/`
/// actually use, with Python's semantics including the optional
/// `strip(chars)` argument. Anything else keeps minijinja's
/// `UnknownMethod` error, which surfaces as a [`TemplateError::Render`]
/// naming the method — a refusal, not an empty string.
fn python_method(
    _state: &minijinja::State,
    value: &JinjaValue,
    method: &str,
    args: &[JinjaValue],
) -> Result<JinjaValue, minijinja::Error> {
    fn unknown() -> minijinja::Error {
        minijinja::Error::from(minijinja::ErrorKind::UnknownMethod)
    }
    fn as_str(v: &JinjaValue) -> Result<&str, minijinja::Error> {
        v.as_str().ok_or_else(|| {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                "expected a string argument",
            )
        })
    }
    match method {
        // dict.get(key[, default])
        "get" => {
            if value.as_object().is_none() {
                return Err(unknown());
            }
            let (key, default) = match args {
                [k] => (k, JinjaValue::from(())),
                [k, d] => (k, d.clone()),
                _ => {
                    return Err(minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "get() takes 1 or 2 arguments",
                    ))
                }
            };
            Ok(value
                .get_item(key)
                .ok()
                .filter(|v| !v.is_undefined())
                .unwrap_or(default))
        }
        // dict.items() / .keys() / .values(): how tool schemas are walked.
        "items" | "keys" | "values" if args.is_empty() => {
            if !matches!(value.kind(), minijinja::value::ValueKind::Map) {
                return Err(unknown());
            }
            py_compat::map_view(value, method)
        }
        "append" => {
            let list = value
                .downcast_object_ref::<py_compat::PyList>()
                .ok_or_else(unknown)?;
            let [item] = args else {
                return Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    "append() takes exactly one argument",
                ));
            };
            list.append(item.clone());
            Ok(value.clone())
        }
        "pop" => {
            let list = value
                .downcast_object_ref::<py_compat::PyList>()
                .ok_or_else(unknown)?;
            let index = match args {
                [] => -1,
                [index] => i64::try_from(index.clone())?,
                _ => {
                    return Err(minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "pop() takes at most one argument",
                    ))
                }
            };
            list.pop(index)
        }
        "format" => py_compat::format(value.as_str().ok_or_else(unknown)?, args),
        "replace" => py_compat::replace(value.as_str().ok_or_else(unknown)?, args),
        // str.split(sep) -- Python's whitespace split when sep is absent.
        "split" => {
            let s = value.as_str().ok_or_else(unknown)?;
            let parts: Vec<JinjaValue> = match args {
                [] => s.split_whitespace().map(JinjaValue::from).collect(),
                [sep] => s.split(as_str(sep)?).map(JinjaValue::from).collect(),
                _ => {
                    return Err(minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "frink implements split() with at most one separator argument",
                    ))
                }
            };
            Ok(JinjaValue::from(parts))
        }
        "strip" | "lstrip" | "rstrip" => {
            let s = value.as_str().ok_or_else(unknown)?;
            let chars: Option<Vec<char>> = match args {
                [] => None,
                [c] => Some(as_str(c)?.chars().collect()),
                _ => {
                    return Err(minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "strip() takes at most one argument",
                    ))
                }
            };
            let pred = |c: char| match &chars {
                Some(set) => set.contains(&c),
                None => c.is_whitespace(),
            };
            Ok(JinjaValue::from(match method {
                "strip" => s.trim_matches(pred),
                "lstrip" => s.trim_start_matches(pred),
                _ => s.trim_end_matches(pred),
            }))
        }
        // str.startswith(prefix) / str.endswith(suffix), with Python's
        // "a string or a tuple of strings" argument. Qwen3.5's template
        // (`chat:72`) tests a user turn for `<tool_response>` wrapping
        // with both; without them the render failed on every request,
        // which is a model that cannot chat, not a template detail.
        "startswith" | "endswith" => {
            let s = value.as_str().ok_or_else(unknown)?;
            let [needle] = args else {
                return Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("{method}() takes exactly one argument"),
                ));
            };
            let test = |n: &str| {
                if method == "startswith" {
                    s.starts_with(n)
                } else {
                    s.ends_with(n)
                }
            };
            let hit = if let Some(n) = needle.as_str() {
                test(n)
            } else if let Ok(iter) = needle.try_iter() {
                let mut any = false;
                for n in iter {
                    any |= test(as_str(&n)?);
                }
                any
            } else {
                return Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("{method}() takes a string or a sequence of strings"),
                ));
            };
            Ok(JinjaValue::from(hit))
        }
        _ => Err(unknown()),
    }
}

/// `{{ raise_exception("...") }}` — the standard HuggingFace escape
/// hatch for "this conversation is not representable in this template"
/// (mistral and gemma-3 both use it to reject non-alternating roles).
/// Aborts the render; the message reaches the client.
fn raise_exception(msg: String) -> Result<JinjaValue, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        format!("template raised: {msg}"),
    ))
}

/// `{{ strftime_now("%d %b %Y") }}` — Llama-3.1's template stamps
/// today's date into its system preamble with it.
///
/// Local time, as llama.cpp's `std::localtime`, and a deliberately small `strftime` subset: `%Y %y %m %d %e %H
/// %M %S %j %B %b %A %a %F %T %%`, plus the `-` no-pad flag
/// (`%-d`, `%-m`). Anything else is an error rather than a silently
/// wrong date — a model told the wrong year is a real quality bug and it
/// would never show up as a crash.
///
/// `FRINK_TEST_CHAT_TEMPLATE_NOW` (Unix seconds, read as UTC) pins the
/// clock ([`llama_defaults::local_now`]).
fn strftime_now(fmt: String) -> Result<String, minijinja::Error> {
    let secs = llama_defaults::local_now()
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e))?;
    format_utc(secs, &fmt).map_err(|spec| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!(
                "strftime_now: unsupported format specifier `%{spec}` in {fmt:?} -- \
                 frink implements a subset (%Y %y %m %d %e %H %M %S %j %B %b %A %a %F %T %%) \
                 and refuses rather than stamping a wrong date into the prompt"
            ),
        )
    })
}

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const WEEKDAYS: [&str; 7] = [
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
];

/// Civil date from a Unix timestamp (Howard Hinnant's `civil_from_days`),
/// then a `strftime` subset. Returns `Err(spec)` naming the first
/// unsupported specifier.
fn format_utc(secs: i64, fmt: &str) -> Result<String, char> {
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (hour, minute, second) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    // 1970-01-01 was a Thursday, hence WEEKDAYS's rotation.
    let weekday = days.rem_euclid(7) as usize;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    // Day-of-year needs the calendar year's own Jan 1.
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    const CUM: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let yday = CUM[(month - 1) as usize] + day + i64::from(leap && month > 2);

    let mut out = String::with_capacity(fmt.len() + 8);
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let mut pad = true;
        let mut spec = chars.next().ok_or('%')?;
        if spec == '-' {
            pad = false;
            spec = chars.next().ok_or('-')?;
        }
        let num = |out: &mut String, v: i64, w: usize| {
            if pad {
                out.push_str(&format!("{v:0w$}"));
            } else {
                out.push_str(&v.to_string());
            }
        };
        match spec {
            'Y' => out.push_str(&year.to_string()),
            'y' => num(&mut out, year.rem_euclid(100), 2),
            'm' => num(&mut out, month, 2),
            'd' => num(&mut out, day, 2),
            // %e is space-padded day-of-month.
            'e' => out.push_str(&format!("{day:2}")),
            'H' => num(&mut out, hour, 2),
            'M' => num(&mut out, minute, 2),
            'S' => num(&mut out, second, 2),
            'j' => num(&mut out, yday, 3),
            'B' => out.push_str(MONTHS[(month - 1) as usize]),
            'b' => out.push_str(&MONTHS[(month - 1) as usize][..3]),
            'A' => out.push_str(WEEKDAYS[weekday]),
            'a' => out.push_str(&WEEKDAYS[weekday][..3]),
            'F' => out.push_str(&format!("{year:04}-{month:02}-{day:02}")),
            'T' => out.push_str(&format!("{hour:02}:{minute:02}:{second:02}")),
            '%' => out.push('%'),
            other => return Err(other),
        }
    }
    Ok(out)
}

/// Text a message contributes to a builtin render: `content` as a
/// string (or the concatenated `text` parts of an OpenAI content array),
/// plus any `tool_calls` re-rendered as the `<tool_call>{…}</tool_call>`
/// marker text a model is asked to emit for a *new* call.
fn builtin_message_text(m: &Value) -> String {
    let mut out = match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    if let Some(Value::Array(calls)) = m.get("tool_calls") {
        for call in calls {
            let f = call.get("function");
            let name = f
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let args = f
                .and_then(|f| f.get("arguments"))
                .map(|a| match a {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_else(|| "{}".to_string());
            out.push_str(&format!(
                "<tool_call>{{\"name\": \"{name}\", \"arguments\": {args}}}</tool_call>"
            ));
        }
    }
    out
}

fn builtin_role(m: &Value) -> &str {
    m.get("role").and_then(Value::as_str).unwrap_or("user")
}

fn render_builtin(b: BuiltinTemplate, messages: &[Value], opts: &RenderOptions) -> String {
    let mut out = String::new();
    match b {
        BuiltinTemplate::ChatMl => {
            for m in messages {
                out.push_str("<|im_start|>");
                out.push_str(builtin_role(m));
                out.push('\n');
                out.push_str(&builtin_message_text(m));
                out.push_str("<|im_end|>\n");
            }
            if opts.add_generation_prompt {
                out.push_str("<|im_start|>assistant\n");
            }
        }
        BuiltinTemplate::Plain => {
            let lines: Vec<String> = messages
                .iter()
                .map(|m| format!("{}: {}", builtin_role(m), builtin_message_text(m)))
                .collect();
            out.push_str(&lines.join("\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {

    /// A checkpoint with no template and no ChatML markers in its vocab
    /// must NOT be wrapped in ChatML.
    ///
    /// OLMoE-1B-7B is the case: its vocabulary contains neither
    /// `<|im_start|>` nor `<|im_end|>`. Wrapping it anyway made the
    /// markers tokenize as literal text the model never saw, left
    /// `<|im_end|>` impossible to generate so nothing could stop the
    /// run, and led the model to imitate the pattern it was shown. The
    /// observed output was 512 tokens of invented `<|area_key|>`-style
    /// markers, while the raw completion path answered correctly.
    #[test]
    fn a_vocab_without_chatml_markers_falls_back_to_plain() {
        let without = ChatTemplate::from_gguf_metadata(None, Some("olmoe"), false, false);
        let with = ChatTemplate::from_gguf_metadata(None, Some("olmoe"), false, true);
        assert_ne!(
            without.describe(),
            with.describe(),
            "the ChatML fallback must depend on the markers actually existing"
        );
        assert!(
            !without.describe().to_lowercase().contains("chatml"),
            "got {} for a vocab with no ChatML tokens",
            without.describe()
        );
    }
    use super::*;
    use serde_json::json;

    fn msg(role: &str, content: &str) -> Value {
        json!({"role": role, "content": content})
    }

    fn opts() -> RenderOptions {
        RenderOptions {
            add_generation_prompt: true,
            bos_token: Some("<s>".into()),
            eos_token: Some("</s>".into()),
            ..Default::default()
        }
    }

    // ---- the four constructs the plan named ------------------------

    /// `{{ bos_token }}`: the template, not the loader, decides where BOS
    /// goes. Mistral-7B-Instruct-v0.2's real GGUF template, verbatim.
    #[test]
    fn renders_bos_token_and_the_real_mistral_inst_framing() {
        let src = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";
        let t = ChatTemplate::from_jinja(src).unwrap();
        let out = t
            .render(
                &[
                    msg("user", "hi"),
                    msg("assistant", "hello"),
                    msg("user", "2+2?"),
                ],
                &opts(),
            )
            .unwrap();
        assert_eq!(out, "<s>[INST] hi [/INST]hello</s>[INST] 2+2? [/INST]");
    }

    /// The sniffing implementation matched *no* marker in that template
    /// and rendered `user: hi` instead. This is the bug, pinned.
    #[test]
    fn mistral_is_not_plain_role_labelled_lines() {
        let src = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% endif %}{% endfor %}";
        let out = ChatTemplate::from_jinja(src)
            .unwrap()
            .render(&[msg("user", "hi")], &opts())
            .unwrap();
        assert!(!out.contains("user: hi"), "{out}");
        assert!(out.contains("[INST]"), "{out}");
    }

    /// A system message: gemma-3's real GGUF template folds it into the
    /// first user turn, which the hand-written `Gemma` renderer only
    /// approximated (it always joined with `\n\n`, and never emitted
    /// `<bos>` or the multimodal `<start_of_image>` arm).
    #[test]
    fn renders_a_system_message_with_the_real_gemma3_template() {
        let src = GEMMA3_TEMPLATE;
        let t = ChatTemplate::from_jinja(src).unwrap();
        let out = t
            .render(
                &[msg("system", "be brief"), msg("user", "hi")],
                &RenderOptions {
                    add_generation_prompt: true,
                    bos_token: Some("<bos>".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            out,
            "<bos><start_of_turn>user\nbe brief\n\nhi<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    /// Multimodal content parts reach `<start_of_image>` — which the
    /// hand-written renderer dropped on the floor.
    #[test]
    fn gemma3_emits_the_image_placeholder_for_content_parts() {
        let out = ChatTemplate::from_jinja(GEMMA3_TEMPLATE)
            .unwrap()
            .render(
                &[json!({"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "what is this?"}
                ]})],
                &RenderOptions {
                    add_generation_prompt: true,
                    bos_token: Some("<bos>".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            out,
            "<bos><start_of_turn>user\n<start_of_image>what is this?<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    /// A tool-call block: Qwen2.5's real GGUF template, the `tools`
    /// preamble plus a replayed assistant `tool_calls` turn plus a
    /// `role: tool` result. None of this was reachable before — no
    /// hand-written renderer read `tools` at all.
    #[test]
    fn renders_a_tool_call_block_with_the_real_qwen25_template() {
        let t = ChatTemplate::from_jinja(QWEN25_TEMPLATE).unwrap();
        assert!(t.handles_tools());
        let out = t
            .render(
                &[
                    msg("user", "weather in Paris?"),
                    json!({"role": "assistant", "content": "", "tool_calls": [
                        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                    ]}),
                    json!({"role": "tool", "content": "18C"}),
                ],
                &RenderOptions {
                    add_generation_prompt: true,
                    tools: vec![json!({"type": "function", "function": {
                        "name": "get_weather",
                        "description": "Current weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                    }})],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            out,
            concat!(
                "<|im_start|>system\n",
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n",
                "# Tools\n\n",
                "You may call one or more functions to assist with the user query.\n\n",
                "You are provided with function signatures within <tools></tools> XML tags:\n",
                "<tools>\n",
                // `tojson`: the keys in the order the tool wrote them and
                // `", "` / `": "` separators, as llama.cpp prints it.
                "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", ",
                "\"description\": \"Current weather\", \"parameters\": {\"type\": \"object\", ",
                "\"properties\": {\"city\": {\"type\": \"string\"}}}}}\n",
                "</tools>\n\n",
                "For each function call, return a json object with function name and arguments ",
                "within <tool_call></tool_call> XML tags:\n",
                "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n",
                "</tool_call><|im_end|>\n",
                "<|im_start|>user\nweather in Paris?<|im_end|>\n",
                "<|im_start|>assistant\n",
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n",
                "</tool_call><|im_end|>\n",
                "<|im_start|>user\n<tool_response>\n18C\n</tool_response><|im_end|>\n",
                "<|im_start|>assistant\n",
            )
        );
    }

    /// `add_generation_prompt` is honoured both ways. The hand-written
    /// renderers appended the assistant header unconditionally, so a
    /// caller could not ask for a prefix-only render (what a
    /// prefill/scoring path or a "continue this reply" request needs).
    #[test]
    fn add_generation_prompt_is_honoured_both_ways() {
        let t = ChatTemplate::from_jinja(GEMMA3_TEMPLATE).unwrap();
        let with = t
            .render(
                &[msg("user", "hi")],
                &RenderOptions {
                    add_generation_prompt: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let without = t
            .render(
                &[msg("user", "hi")],
                &RenderOptions {
                    add_generation_prompt: false,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            with,
            "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n"
        );
        assert_eq!(without, "<start_of_turn>user\nhi<end_of_turn>\n");
    }

    /// `add_generation_prompt` + `{{ bos_token }}` + a system message on
    /// the real Llama-3.1-8B-Instruct template, which is also the
    /// regression for a bug this rewrite introduced and then fixed:
    /// leaving `tools` *undefined* rather than binding it to `null` made
    /// `{%- if tools is not none %}` true, so every plain chat request
    /// got Llama's ipython tool-calling preamble.
    #[test]
    fn llama31_binds_tools_to_null_so_a_plain_chat_gets_no_tool_preamble() {
        // `date_string` is today's, as in llama.cpp, so pin today.
        let _clock = llama_defaults::PinnedClock::at(1_721_952_000); // 2024-07-26
        let out = ChatTemplate::from_jinja(LLAMA31_TEMPLATE)
            .unwrap()
            .render(
                &[msg("system", "be brief"), msg("user", "hi")],
                &RenderOptions {
                    add_generation_prompt: true,
                    bos_token: Some("<|begin_of_text|>".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            out,
            concat!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
                "Cutting Knowledge Date: December 2023\n",
                "Today Date: 26 Jul 2024\n\n",
                "be brief<|eot_id|>",
                "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>",
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
            )
        );
        assert!(!out.contains("Environment: ipython"), "{out}");
    }

    // ---- sharp edges the plan named --------------------------------

    #[test]
    fn raise_exception_fails_the_render_and_keeps_the_message() {
        let src = "{{ bos_token }}{% for m in messages %}{% if (m['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% endfor %}";
        let err = ChatTemplate::from_jinja(src)
            .unwrap()
            .render(&[msg("assistant", "oops")], &opts())
            .unwrap_err();
        let text = err.to_string();
        assert!(matches!(err, TemplateError::Render(_)), "{text}");
        assert!(text.contains("roles must alternate"), "{text}");
    }

    #[test]
    fn strftime_now_stamps_a_pinned_clock() {
        // 2024-07-04T12:34:56Z
        let _clock = llama_defaults::PinnedClock::at(1_720_096_496);
        let out = ChatTemplate::from_jinja(
            "{{ strftime_now(\"%d %b %Y\") }}|{{ strftime_now('%A %F %T %j %-d') }}",
        )
        .unwrap()
        .render(&[], &opts())
        .unwrap();
        assert_eq!(out, "04 Jul 2024|Thursday 2024-07-04 12:34:56 186 4");
    }

    #[test]
    fn strftime_now_refuses_an_unimplemented_specifier() {
        let err = ChatTemplate::from_jinja("{{ strftime_now('%Z') }}")
            .unwrap()
            .render(&[], &opts())
            .unwrap_err();
        assert!(
            err.to_string().contains("unsupported format specifier"),
            "{err}"
        );
    }

    /// Whitespace control (`{%- … -%}`) is what makes gemma-3 render as
    /// one unbroken line despite being written across 40 indented lines.
    /// If it were ignored the prompt would be full of stray newlines.
    #[test]
    fn whitespace_control_is_respected() {
        let out = ChatTemplate::from_jinja(
            "{%- for m in messages -%}\n    {{- m['role'] -}}\n{%- endfor -%}",
        )
        .unwrap()
        .render(&[msg("user", "x"), msg("assistant", "y")], &opts())
        .unwrap();
        assert_eq!(out, "userassistant");
    }

    /// `namespace()` is the standard workaround for Jinja loop scoping —
    /// a plain `{% set %}` inside a `{% for %}` does not escape it.
    /// gemma-4's real template uses six namespaces.
    #[test]
    fn namespace_writes_escape_loop_scope() {
        let out = ChatTemplate::from_jinja(
            "{%- set ns = namespace(n=0) -%}{%- for m in messages -%}{%- set ns.n = ns.n + 1 -%}{%- endfor -%}{{ ns.n }}",
        )
        .unwrap()
        .render(&[msg("user", "a"), msg("user", "b"), msg("user", "c")], &opts())
        .unwrap();
        assert_eq!(out, "3");
    }

    /// An unknown filter must be a refusal, not an empty string.
    #[test]
    fn an_unsupported_construct_fails_loudly() {
        let err = ChatTemplate::from_jinja("{{ messages | no_such_filter }}")
            .unwrap()
            .render(&[msg("user", "hi")], &opts())
            .unwrap_err();
        let text = err.to_string();
        assert!(matches!(err, TemplateError::Render(_)), "{text}");
        assert!(text.contains("no_such_filter"), "{text}");
    }

    #[test]
    fn a_template_that_does_not_compile_is_recorded_not_replaced() {
        let t = ChatTemplate::from_gguf_metadata(
            Some("{% for m in messages %}{{ m }}"),
            Some("llama"),
            false,
            true,
        );
        assert!(!t.is_jinja());
        let err = t.render(&[msg("user", "hi")], &opts()).unwrap_err();
        assert!(matches!(err, TemplateError::Compile(_)), "{err}");
        // Specifically NOT a silent fallback to a hand-written renderer.
        assert!(t.describe().starts_with("BROKEN"), "{}", t.describe());
    }

    // ---- chat_template_kwargs passthrough --------------------------

    #[test]
    fn chat_template_kwargs_reach_the_template() {
        let src = "{%- if enable_thinking -%}THINK{%- else -%}PLAIN{%- endif -%}";
        let t = ChatTemplate::from_jinja(src).unwrap();
        let mut extra = serde_json::Map::new();
        extra.insert("enable_thinking".into(), Value::Bool(true));
        let on = t
            .render(
                &[msg("user", "hi")],
                &RenderOptions {
                    extra,
                    ..Default::default()
                },
            )
            .unwrap();
        let off = t
            .render(&[msg("user", "hi")], &RenderOptions::default())
            .unwrap();
        assert_eq!((on.as_str(), off.as_str()), ("THINK", "PLAIN"));
    }

    #[test]
    fn chat_template_kwargs_cannot_shadow_messages_or_tools() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "messages".into(),
            json!([{"role": "user", "content": "INJECTED"}]),
        );
        extra.insert("add_generation_prompt".into(), Value::Bool(true));
        let out = ChatTemplate::from_jinja(
            "{%- for m in messages -%}{{ m['content'] }}{%- endfor -%}|{{ add_generation_prompt }}",
        )
        .unwrap()
        .render(
            &[msg("user", "real")],
            &RenderOptions {
                add_generation_prompt: false,
                extra,
                ..Default::default()
            },
        )
        .unwrap();
        // `False`: jinja2 and llama.cpp print Python's spelling of a bool.
        assert_eq!(out, "real|False");
    }

    /// llama.cpp sets both spellings at render, so `frink run` needs this
    /// as much as the server does. A caller's own spelling stays.
    #[test]
    fn reasoning_effort_also_reads_as_reasoning_strength() {
        let render = |extra: serde_json::Value| {
            let serde_json::Value::Object(extra) = extra else {
                panic!("not an object");
            };
            ChatTemplate::from_jinja("[{{ reasoning_strength }}]")
                .unwrap()
                .render(
                    &[msg("user", "x")],
                    &RenderOptions {
                        extra,
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        assert_eq!(
            render(serde_json::json!({"reasoning_effort": "high"})),
            "[high]"
        );
        assert_eq!(render(serde_json::json!({"reasoning_effort": ""})), "[]");
        assert_eq!(
            render(serde_json::json!({"reasoning_effort": "high", "reasoning_strength": "low"})),
            "[low]"
        );
    }

    /// GLM-4.7-Flash keeps earlier thoughts only under
    /// `clear_thinking=false`, and llama.cpp's default keeps them.
    #[test]
    fn preserve_reasoning_also_reads_as_the_four_names_templates_use() {
        let render = |extra: serde_json::Value| {
            let serde_json::Value::Object(extra) = extra else {
                panic!("not an object");
            };
            ChatTemplate::from_jinja(
                "{{ preserve_thinking }}{{ clear_thinking }}\
                 {{ truncate_history_thinking }}{{ drop_thinking }}",
            )
            .unwrap()
            .render(
                &[msg("user", "x")],
                &RenderOptions {
                    extra,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        assert_eq!(
            render(serde_json::json!({"preserve_reasoning": true})),
            "TrueFalseFalseFalse"
        );
        assert_eq!(
            render(serde_json::json!({"preserve_reasoning": false})),
            "FalseTrueTrueTrue"
        );
        // Not a boolean: upstream skips it, so the template sees none.
        assert_eq!(render(serde_json::json!({"preserve_reasoning": "yes"})), "");
        assert_eq!(
            render(serde_json::json!({"preserve_reasoning": false, "clear_thinking": false})),
            "FalseFalseTrueTrue"
        );
    }

    // ---- gemma-4: the variant the plan says was never implemented ---

    /// The hand-written `ChatTemplate::Gemma4` rendered
    /// `<|turn>user\n…<turn|>\n<|turn>model\n` and nothing else. The real
    /// template injects a `<|think|>` channel into the first system turn
    /// when `enable_thinking` is set — driven by `chat_template_kwargs`,
    /// which had no path to it at all before.
    #[test]
    fn gemma4_thinking_injection_is_reachable_now() {
        let t = ChatTemplate::from_jinja(GEMMA4_TEMPLATE_CORE).unwrap();
        let mut extra = serde_json::Map::new();
        extra.insert("enable_thinking".into(), Value::Bool(true));
        let thinking = t
            .render(
                &[msg("user", "hi")],
                &RenderOptions {
                    add_generation_prompt: true,
                    bos_token: Some("<bos>".into()),
                    extra,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            thinking,
            "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
        );
        let plain = t
            .render(
                &[msg("user", "hi")],
                &RenderOptions {
                    add_generation_prompt: true,
                    bos_token: Some("<bos>".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(plain, "<bos><|turn>user\nhi<turn|>\n<|turn>model\n");
    }

    /// `strip_thinking`: a replayed assistant turn must have its
    /// `<|channel>…<channel|>` reasoning removed before it goes back into
    /// the prompt. The hand-written renderer replayed it verbatim.
    #[test]
    fn gemma4_strip_thinking_removes_replayed_reasoning() {
        let out = ChatTemplate::from_jinja(GEMMA4_TEMPLATE_CORE)
            .unwrap()
            .render(
                &[
                    msg("user", "hi"),
                    msg(
                        "assistant",
                        "<|channel>thought\nlet me think<channel|>the answer is 4",
                    ),
                    msg("user", "again?"),
                ],
                &RenderOptions {
                    add_generation_prompt: true,
                    bos_token: Some("<bos>".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!out.contains("let me think"), "{out}");
        assert!(
            out.contains("<|turn>model\nthe answer is 4<turn|>\n"),
            "{out}"
        );
    }

    // ---- builtins (no template in the checkpoint) -------------------

    #[test]
    fn a_checkpoint_with_no_template_gets_chatml_or_plain() {
        assert!(matches!(
            &*ChatTemplate::from_gguf_metadata(None, Some("olmoe"), false, true).0,
            Kind::Builtin(BuiltinTemplate::ChatMl)
        ));
        assert!(matches!(
            &*ChatTemplate::from_gguf_metadata(Some("   "), Some("olmoe"), false, true).0,
            Kind::Builtin(BuiltinTemplate::ChatMl)
        ));
        assert!(matches!(
            &*ChatTemplate::from_gguf_metadata(None, Some("olmoe"), true, true).0,
            Kind::Builtin(BuiltinTemplate::Plain)
        ));
        assert!(matches!(
            &*ChatTemplate::from_gguf_metadata(None, None, false, true).0,
            Kind::Builtin(BuiltinTemplate::Plain)
        ));
    }

    #[test]
    fn builtin_chatml_and_plain_render_as_before() {
        let msgs = [msg("system", "be helpful"), msg("user", "hi")];
        assert_eq!(
            ChatTemplate::builtin(BuiltinTemplate::ChatMl)
                .render(&msgs, &opts())
                .unwrap(),
            "<|im_start|>system\nbe helpful<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            ChatTemplate::builtin(BuiltinTemplate::Plain)
                .render(&msgs, &opts())
                .unwrap(),
            "system: be helpful\nuser: hi"
        );
    }

    #[test]
    fn builtin_renders_replayed_tool_calls_as_marker_text() {
        let msgs = [json!({"role": "assistant", "tool_calls": [
            {"function": {"name": "f", "arguments": "{\"a\": 1}"}}
        ]})];
        assert_eq!(
            ChatTemplate::builtin(BuiltinTemplate::Plain)
                .render(&msgs, &opts())
                .unwrap(),
            "assistant: <tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}}</tool_call>"
        );
    }

    #[test]
    fn handles_tools_is_a_property_of_the_template_not_a_guess() {
        assert!(ChatTemplate::from_jinja(QWEN25_TEMPLATE)
            .unwrap()
            .handles_tools());
        assert!(!ChatTemplate::from_jinja(GEMMA3_TEMPLATE)
            .unwrap()
            .handles_tools());
        assert!(!ChatTemplate::builtin(BuiltinTemplate::ChatMl).handles_tools());
    }

    /// Qwen3.5's template (`chat:72`) tests a user turn with
    /// `content.startswith('<tool_response>') and content.endswith(...)`;
    /// before the shim had them every request failed to render.
    #[test]
    fn python_startswith_and_endswith_take_a_string_or_a_tuple() {
        let t = ChatTemplate::from_jinja(
            "{% for m in messages %}{{ m.content.startswith('<tool_response>') }}\
             {{ m.content.endswith(('a', '</tool_response>')) }}\
             {{ m.content.startswith(('x', 'y')) }};{% endfor %}",
        )
        .unwrap();
        let out = t
            .render(
                &[serde_json::json!({"role": "user", "content": "<tool_response>ok</tool_response>"})],
                &RenderOptions::default(),
            )
            .unwrap();
        assert_eq!(out, "TrueTrueFalse;");
    }

    #[test]
    fn utc_calendar_math_matches_known_dates() {
        assert_eq!(
            format_utc(0, "%F %T %A %j").unwrap(),
            "1970-01-01 00:00:00 Thursday 001"
        );
        assert_eq!(
            format_utc(951_782_400, "%F %A %j").unwrap(),
            "2000-02-29 Tuesday 060"
        );
        assert_eq!(
            format_utc(1_709_164_800, "%F %A %j").unwrap(),
            "2024-02-29 Thursday 060"
        );
        assert_eq!(
            format_utc(1_767_225_599, "%F %T %j").unwrap(),
            "2025-12-31 23:59:59 365"
        );
        assert_eq!(
            format_utc(-86_400, "%F %A").unwrap(),
            "1969-12-31 Wednesday"
        );
    }

    // ---- real template strings, verbatim from local GGUFs -----------

    /// `tokenizer.chat_template` of `models/gemma-3-1b-it-Q8_0.gguf`,
    /// read out of the file's metadata, not paraphrased.
    const GEMMA3_TEMPLATE: &str = include_str!("../tests/templates/gemma-3-1b-it.jinja");
    /// `tokenizer.chat_template` of `models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`.
    const QWEN25_TEMPLATE: &str = include_str!("../tests/templates/qwen2.5-instruct.jinja");
    /// `tokenizer.chat_template` of `models/gemma-4-E2B-it-Q4_K_M.gguf`,
    /// all 18 KB of it: six macros, six namespaces, recursive
    /// `format_parameters`, `{% set … %}{% endset %}` block capture,
    /// string slicing and `.split()`. This is the template the plan
    /// records as "checked, and `ChatTemplate::Gemma4` does not
    /// implement it".
    const GEMMA4_TEMPLATE_CORE: &str = include_str!("../tests/templates/gemma-4-E2B-it.jinja");
    /// `tokenizer.chat_template` of `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`.
    const LLAMA31_TEMPLATE: &str = include_str!("../tests/templates/llama-3.1-8b-instruct.jinja");
}
