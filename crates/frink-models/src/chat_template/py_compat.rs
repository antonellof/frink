//! What chat templates expect of Python and jinja2 that minijinja does
//! not provide.
//!
//! Each piece is read from llama.cpp's own engine (`common/jinja/`), the
//! reference for these strings. `upstream_template_census.rs` runs every
//! template llama.cpp bundles through them.

use std::borrow::Cow;
use std::sync::{Arc, Mutex, MutexGuard};

use minijinja::value::{
    Enumerator, Kwargs, Object, ObjectRepr, Rest, Value as JinjaValue, ValueKind,
};
use minijinja::{Error, ErrorKind};
use serde_json::Value;

fn invalid(message: impl Into<Cow<'static, str>>) -> Error {
    Error::new(ErrorKind::InvalidOperation, message)
}

/// `{{ tool | tojson }}`, how every tool-calling template serialises a
/// function schema into the prompt, so its exact bytes are part of the
/// prompt the model was trained on.
///
/// llama.cpp's filter (`common/jinja/value.cpp`, `tojson`): `ensure_ascii`,
/// `indent`, `separators` and `sort_keys`, by name or in that order. The
/// default item separator is `", "`, or `","` once `indent` is given.
/// `sort_keys=true` is refused, as it is there.
///
/// Overrides minijinja's builtin, which emits `{"a":1}`. An unknown
/// argument is an error, because a filter that ignored one would change
/// the prompt without saying so.
///
/// Keys come out in the order they went in, as llama.cpp keeps them:
/// `serde_json` and minijinja are both built with `preserve_order`, so a
/// tool's schema prints in the order the request wrote it.
///
/// No `htmlsafe_json_dumps` escaping of `< > & '`. llama.cpp does not do
/// it either, and it is llama.cpp this engine is checked against.
pub(super) fn tojson(
    value: JinjaValue,
    positional: Rest<JinjaValue>,
    kwargs: Kwargs,
) -> Result<String, Error> {
    let arg = |name: &str, position: usize| -> Result<Option<JinjaValue>, Error> {
        Ok(kwargs
            .get::<Option<JinjaValue>>(name)?
            .or_else(|| positional.get(position).cloned()))
    };
    let ensure_ascii = arg("ensure_ascii", 0)?.is_some_and(|v| v.is_true());
    let indent = arg("indent", 1)?
        .filter(JinjaValue::is_integer)
        .and_then(|v| i64::try_from(v).ok())
        .unwrap_or(-1);
    let separators: Vec<String> = match arg("separators", 2)? {
        Some(v) if v.is_none() => Vec::new(),
        Some(v) => v.try_iter()?.map(|s| s.to_string()).collect(),
        None => Vec::new(),
    };
    if arg("sort_keys", 3)?.is_some_and(|v| v.is_true()) {
        return Err(invalid("tojson: sort_keys=true is not implemented"));
    }
    kwargs.assert_all_used()?;

    let style = JsonStyle {
        indent: usize::try_from(indent).ok(),
        item_sep: separators.first().map_or_else(
            || if indent < 0 { ", " } else { "," }.to_string(),
            Clone::clone,
        ),
        key_sep: separators
            .get(1)
            .map_or_else(|| ": ".to_string(), Clone::clone),
    };
    let json: Value = serde_json::to_value(&value)
        .map_err(|e| invalid(format!("tojson: value is not serialisable: {e}")))?;
    let mut out = String::new();
    style.write(&json, 0, &mut out);
    Ok(if ensure_ascii { ascii_json(&out) } else { out })
}

struct JsonStyle {
    /// `None` is llama.cpp's negative default: one line.
    indent: Option<usize>,
    item_sep: String,
    key_sep: String,
}

impl JsonStyle {
    fn line_break(&self, level: usize, out: &mut String) {
        if let Some(indent) = self.indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', level * indent));
        }
    }

    fn write(&self, value: &Value, level: usize, out: &mut String) {
        match value {
            Value::Object(map) if !map.is_empty() => {
                out.push('{');
                for (i, (key, value)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.line_break(level + 1, out);
                    out.push_str(&Value::String(key.clone()).to_string());
                    out.push_str(&self.key_sep);
                    self.write(value, level + 1, out);
                }
                self.line_break(level, out);
                out.push('}');
            }
            Value::Array(items) if !items.is_empty() => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.line_break(level + 1, out);
                    self.write(item, level + 1, out);
                }
                self.line_break(level, out);
                out.push(']');
            }
            other => out.push_str(&other.to_string()),
        }
    }
}

/// `ensure_ascii=true`: everything outside ASCII as a `\uXXXX` escape, a
/// surrogate pair above the basic plane, as `json.dumps` writes it.
fn ascii_json(json: &str) -> String {
    let mut out = String::with_capacity(json.len());
    let mut units = [0u16; 2];
    for c in json.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            for unit in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

/// jinja2's `is iterable`, which is false for `none`. minijinja's says
/// true, because it iterates `none` as empty.
///
/// Templates guard on it: `{% if tools is iterable and tools | length > 0 %}`
/// is how Qwen3-Coder opens, and `tools` is `none` on a request that
/// offers none, so the guard passed and `length` failed on every plain
/// chat.
pub(super) fn is_iterable(value: JinjaValue) -> bool {
    !value.is_none() && value.try_iter().is_ok()
}

/// What `{{ value }}` writes, as llama.cpp's engine writes it
/// (`common/jinja/value.h`, `as_string`): a list or a map in Python's
/// literal syntax (`{'type': 'function', 'name': 'f'}`), where
/// minijinja writes `{"type": "function", "name": "f"}`, and a float to
/// six places with the trailing zeros dropped. Anything else as
/// minijinja writes it.
pub(super) fn formatter(
    out: &mut minijinja::Output,
    state: &minijinja::State,
    value: &JinjaValue,
) -> Result<(), Error> {
    if is_python_shaped(value) {
        out.write_str(&py_str(value))
            .map_err(|e| Error::new(ErrorKind::WriteFailure, "formatter").with_source(e))
    } else {
        minijinja::escape_formatter(out, state, value)
    }
}

/// `value | string`, which is `str(value)`: the same text
/// [`formatter`] prints. Apriel 1.5 and 1.6 print each tool with
/// `{{ tool | string }}`.
pub(super) fn string(value: JinjaValue) -> JinjaValue {
    if is_python_shaped(&value) {
        JinjaValue::from(py_str(&value))
    } else if value.kind() == ValueKind::String {
        value
    } else {
        JinjaValue::from(value.to_string())
    }
}

fn is_python_shaped(value: &JinjaValue) -> bool {
    matches!(value.kind(), ValueKind::Map | ValueKind::Seq) || is_float(value)
}

fn is_float(value: &JinjaValue) -> bool {
    value.kind() == ValueKind::Number && !value.is_integer()
}

/// `str(value)` for a list, a map or a float.
fn py_str(value: &JinjaValue) -> String {
    let mut out = String::new();
    py_repr(value, &mut out);
    out
}

/// `value_to_string_repr` (`common/jinja/value.cpp`). A string goes in
/// single quotes unless it holds one, and then it is a JSON string,
/// which is upstream's rule and not quite Python's.
fn py_repr(value: &JinjaValue, out: &mut String) {
    match value.kind() {
        ValueKind::String => {
            let text = value.as_str().unwrap_or_default();
            if text.contains('\'') {
                out.push_str(&Value::String(text.to_string()).to_string());
            } else {
                out.push('\'');
                out.push_str(text);
                out.push('\'');
            }
        }
        ValueKind::Seq => {
            out.push('[');
            for (i, item) in value.try_iter().into_iter().flatten().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_repr(&item, out);
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (i, key) in value.try_iter().into_iter().flatten().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_repr(&key, out);
                out.push_str(": ");
                py_repr(&value.get_item(&key).unwrap_or_default(), out);
            }
            out.push('}');
        }
        ValueKind::Undefined => out.push_str("Undefined"),
        _ if is_float(value) => {
            // `std::to_string` on a double is `%f`.
            let f = f64::try_from(value.clone()).unwrap_or_default();
            let mut text = format!("{f:.6}");
            text.truncate(text.trim_end_matches('0').len());
            if text.ends_with('.') {
                text.push('0');
            }
            out.push_str(&text);
        }
        _ => out.push_str(&value.to_string()),
    }
}

/// `{% generation %}` and `{% endgeneration %}`, a transformers tag that
/// marks the span a model generated. llama.cpp parses both as no-ops
/// (`common/jinja/parser.cpp`). minijinja has no such statement and
/// cannot be taught one, so each becomes a statement that also renders
/// nothing, keeping its `-` markers and so the whitespace it trims.
pub(super) fn neutralise_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find("{%") {
        let (before, tag) = rest.split_at(open);
        out.push_str(before);
        let Some(close) = tag.find("%}") else {
            out.push_str(tag);
            return out;
        };
        let (statement, after) = tag.split_at(close + 2);
        let inner = &statement[2..close];
        if matches!(
            inner.trim_matches(['-', '+']).trim(),
            "generation" | "endgeneration"
        ) {
            let left = inner.strip_prefix(['-', '+']).map_or("", |_| &inner[..1]);
            let right = inner
                .strip_suffix(['-', '+'])
                .map_or("", |_| &inner[inner.len() - 1..]);
            out.push_str(&format!("{{%{left} set __generation = none {right}%}}"));
        } else {
            out.push_str(statement);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

/// A list `.append` and `.pop` can change.
///
/// minijinja's lists are immutable, and a Python template mutates one to
/// carry state out of a loop. The Kimi K2 family does, to pair each tool
/// response with its call. Every such list in llama.cpp's bundled
/// templates is held in a `namespace(...)`, so [`namespace`] makes these
/// and a list elsewhere stays immutable: `.append` on it is an error,
/// never a silent no-op.
#[derive(Debug)]
pub(super) struct PyList(Mutex<Vec<JinjaValue>>);

impl PyList {
    fn items(&self) -> MutexGuard<'_, Vec<JinjaValue>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `list.append(x)`. Returns the list, as llama.cpp does, where
    /// Python returns `None`.
    pub(super) fn append(&self, item: JinjaValue) {
        self.items().push(item);
    }

    /// `list.pop([index])`, with Python's negative indices.
    pub(super) fn pop(&self, index: i64) -> Result<JinjaValue, Error> {
        let mut items = self.items();
        let len = i64::try_from(items.len()).unwrap_or(i64::MAX);
        let at = if index < 0 { index + len } else { index };
        match usize::try_from(at) {
            Ok(at) if at < items.len() => Ok(items.remove(at)),
            _ => Err(invalid(format!(
                "pop: index {index} is out of range for a list of {len}"
            ))),
        }
    }
}

impl Object for PyList {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Seq
    }

    fn get_value(self: &Arc<Self>, key: &JinjaValue) -> Option<JinjaValue> {
        self.items().get(key.as_usize()?).cloned()
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Seq(self.items().len())
    }
}

/// jinja2's `namespace(...)`, with each list in it mutable. See [`PyList`].
pub(super) fn namespace(defaults: Option<JinjaValue>) -> Result<JinjaValue, Error> {
    let pairs = defaults
        .as_ref()
        .and_then(JinjaValue::as_object)
        .filter(|o| matches!(o.repr(), ObjectRepr::Map))
        .and_then(|o| o.try_iter_pairs());
    let Some(pairs) = pairs else {
        // Absent, or the wrong shape, which the builtin refuses by name.
        return minijinja::functions::namespace(defaults);
    };
    let mut mutable = Vec::new();
    for (key, value) in pairs {
        let value = if matches!(value.kind(), minijinja::value::ValueKind::Seq) {
            JinjaValue::from_object(PyList(Mutex::new(value.try_iter()?.collect())))
        } else {
            value
        };
        mutable.push((key, value));
    }
    minijinja::functions::namespace(Some(mutable.into_iter().collect()))
}

/// `str.format`, with `{}` placeholders only, which is all llama.cpp's
/// engine supports (`common/jinja/value.cpp`, `format`).
pub(super) fn format(template: &str, args: &[JinjaValue]) -> Result<JinjaValue, Error> {
    let mut out = String::with_capacity(template.len());
    let mut args = args.iter();
    let mut chars = template.chars();
    while let Some(c) = chars.next() {
        if c != '{' {
            out.push(c);
            continue;
        }
        if chars.next() != Some('}') {
            return Err(invalid("format() only supports simple '{}' placeholders"));
        }
        let arg = args
            .next()
            .ok_or_else(|| invalid("format() has more placeholders than arguments"))?;
        out.push_str(&arg.to_string());
    }
    Ok(JinjaValue::from(out))
}

/// `str.replace(old, new)`. A `count` is refused, as llama.cpp refuses it.
pub(super) fn replace(s: &str, args: &[JinjaValue]) -> Result<JinjaValue, Error> {
    let (old, new) = match args {
        [old, new] => (old, new),
        [old, new, count] if i64::try_from(count.clone()).is_ok_and(|c| c == -1) => (old, new),
        [_, _, _] => return Err(invalid("replace() with a count is not implemented")),
        _ => return Err(invalid("replace() takes an old and a new string")),
    };
    let text = |v: &'_ JinjaValue| -> Result<String, Error> {
        v.as_str()
            .map(str::to_string)
            .ok_or_else(|| invalid("replace() takes strings"))
    };
    Ok(JinjaValue::from(s.replace(&text(old)?, &text(new)?)))
}

/// `dict.items()`, `.keys()` and `.values()`, in the map's own order.
pub(super) fn map_view(map: &JinjaValue, method: &str) -> Result<JinjaValue, Error> {
    let keys = map.try_iter()?;
    let pairs = keys.map(|key| {
        let value = map.get_item(&key)?;
        Ok(match method {
            "keys" => key,
            "values" => value,
            _ => JinjaValue::from(vec![key, value]),
        })
    });
    Ok(JinjaValue::from(pairs.collect::<Result<Vec<_>, Error>>()?))
}

#[cfg(test)]
mod tests {
    use super::super::{ChatTemplate, RenderOptions, TemplateError};
    use serde_json::json;

    fn render(source: &str, tools: Vec<serde_json::Value>) -> Result<String, TemplateError> {
        let opts = RenderOptions {
            tools,
            ..Default::default()
        };
        ChatTemplate::from_jinja(source)?.render(&[json!({"role": "user", "content": "x"})], &opts)
    }

    fn rendered(source: &str) -> String {
        render(source, Vec::new()).unwrap()
    }

    /// Against `json.dumps`: a nested value, an empty container of each
    /// kind, and `indent=0`, which breaks lines without indenting.
    #[test]
    fn tojson_lays_out_indent_like_json_dumps() {
        assert_eq!(
            rendered(r#"{{ {"a": [1, 2], "b": {}, "c": []} | tojson(indent=2) }}"#),
            "{\n  \"a\": [\n    1,\n    2\n  ],\n  \"b\": {},\n  \"c\": []\n}"
        );
        assert_eq!(
            rendered(r#"{{ {"a": [1]} | tojson(indent=0) }}"#),
            "{\n\"a\": [\n1\n]\n}"
        );
        assert_eq!(
            rendered(r#"{{ {"a": [1, 2]} | tojson }}"#),
            r#"{"a": [1, 2]}"#
        );
    }

    /// A map's keys come out in the order they were written, nested and
    /// in a tool's schema alike, as llama.cpp keeps them.
    #[test]
    fn tojson_keeps_the_order_keys_were_written_in() {
        assert_eq!(
            rendered(r#"{{ {"b": 1, "a": {"z": 2, "y": 3}} | tojson }}"#),
            r#"{"b": 1, "a": {"z": 2, "y": 3}}"#
        );
        let tool = json!({"type": "function", "function": {"name": "f", "description": "d"}});
        assert_eq!(
            render("{{ tools[0] | tojson }}", vec![tool]).unwrap(),
            r#"{"type": "function", "function": {"name": "f", "description": "d"}}"#
        );
    }

    /// `{{ value }}` and `value | string` print a list or a map as
    /// llama.cpp does: Python's literal syntax, a string holding a single
    /// quote as JSON, and a float to six places less its trailing zeros.
    #[test]
    fn a_printed_map_or_list_is_pythons_literal() {
        let tool = json!({"type": "function", "function": {
            "name": "f", "required": ["a"], "strict": true, "default": null,
            "note": "it's", "min": 0.5, "max": 3, "step": 1e-7,
        }});
        let want = "{'type': 'function', 'function': {'name': 'f', 'required': ['a'], \
                    'strict': True, 'default': None, 'note': \"it's\", 'min': 0.5, \
                    'max': 3, 'step': 0.0}}";
        assert_eq!(render("{{ tools[0] }}", vec![tool.clone()]).unwrap(), want);
        assert_eq!(render("{{ tools[0] | string }}", vec![tool]).unwrap(), want);
        assert_eq!(
            rendered("{{ [1, 'x', [2.25]] }}|{{ 2.5 }}|{{ 'a' | string }}|{{ 7 }}"),
            "[1, 'x', [2.25]]|2.5|a|7"
        );
    }

    #[test]
    fn tojson_escapes_only_when_asked_and_uses_surrogate_pairs() {
        let source = r#"{{ "é😀" | tojson(ensure_ascii=ENSURE) }}"#;
        assert_eq!(rendered(&source.replace("ENSURE", "false")), "\"é😀\"");
        assert_eq!(
            rendered(&source.replace("ENSURE", "true")),
            "\"\\u00e9\\ud83d\\ude00\""
        );
    }

    #[test]
    fn tojson_takes_separators_by_name_and_arguments_by_position() {
        assert_eq!(
            rendered(r#"{{ {"a": 1, "b": 2} | tojson(separators=[",", ":"]) }}"#),
            r#"{"a":1,"b":2}"#
        );
        // llama.cpp's order: ensure_ascii, then indent.
        assert_eq!(rendered(r#"{{ [1] | tojson(false, 1) }}"#), "[\n 1\n]");
    }

    #[test]
    fn tojson_refuses_sort_keys_and_an_argument_it_does_not_know() {
        for source in [
            "{{ {} | tojson(sort_keys=true) }}",
            "{{ {} | tojson(pretty=true) }}",
        ] {
            assert!(render(source, Vec::new()).is_err(), "{source} rendered");
        }
    }

    /// What Qwen3-Coder and Nemotron open with. On a plain chat `tools`
    /// is `none`, which passed the guard and failed in `length`.
    #[test]
    fn none_is_not_iterable_so_a_tools_guard_holds_on_a_plain_chat() {
        let source = "{% if tools is iterable and tools | length > 0 %}T{% else %}plain{% endif %}";
        assert_eq!(render(source, Vec::new()).unwrap(), "plain");
        assert_eq!(render(source, vec![json!({"name": "f"})]).unwrap(), "T");
        for (value, iterable) in [("[1]", "y"), ("'s'", "y"), ("{}", "y"), ("3", "n")] {
            let source = format!("{{% if {value} is iterable %}}y{{% else %}}n{{% endif %}}");
            assert_eq!(rendered(&source), iterable, "{value}");
        }
    }

    /// A no-op that still trims: `-` on a tag removes the whitespace
    /// beside it, and a tag without `-` leaves it.
    #[test]
    fn generation_tags_render_their_body_and_keep_whitespace_control() {
        assert_eq!(
            rendered("a  {%- generation -%}  b  {%- endgeneration -%}  c"),
            "abc"
        );
        assert_eq!(
            rendered("a {% generation %}b{% endgeneration %} c"),
            "a b c"
        );
        // `set` inside one is visible after it, as with llama.cpp's no-op
        // statement, and a tag only NAMED generation stays what it was.
        assert_eq!(
            rendered("{% generation %}{% set x = 1 %}{% endgeneration %}{{ x }}"),
            "1"
        );
        assert!(ChatTemplate::from_jinja("{% regeneration %}").is_err());
    }

    /// The Kimi K2 queue: pushed by each tool call in one loop iteration
    /// and drained by a tool response in a later one.
    #[test]
    fn a_namespace_list_carries_state_out_of_a_loop() {
        assert_eq!(
            rendered(
                "{% set q = namespace(ids=[]) %}\
                 {% for i in [1, 2, 3] %}{% set _ = q.ids.append(i) %}{% endfor %}\
                 {{ q.ids | length }}:{{ q.ids.pop(0) }}:{{ q.ids.pop() }}:{{ q.ids | tojson }}"
            ),
            "3:1:3:[2]"
        );
        assert_eq!(
            rendered("{% set q = namespace(ids=[7, 8]) %}{{ q.ids[-1] }}{{ q.ids.pop(-2) }}"),
            "87"
        );
    }

    /// Refused rather than ignored: a dropped `.append` silently changes
    /// every later turn.
    #[test]
    fn mutating_a_list_outside_a_namespace_or_past_its_end_is_an_error() {
        for source in [
            "{% set x = [] %}{% set _ = x.append(1) %}",
            "{% set q = namespace(ids=[]) %}{{ q.ids.pop() }}",
            "{% set q = namespace(ids=[1]) %}{{ q.ids.pop(1) }}",
        ] {
            assert!(render(source, Vec::new()).is_err(), "{source} rendered");
        }
    }

    #[test]
    fn format_takes_bare_placeholders_and_refuses_the_rest() {
        assert_eq!(rendered("{{ '<a{}b{}>'.format(1, 'x') }}"), "<a1bx>");
        assert_eq!(
            rendered("{{ 'no placeholder'.format(1) }}"),
            "no placeholder"
        );
        for source in ["{{ '{0}'.format(1) }}", "{{ '{}{}'.format(1) }}"] {
            assert!(render(source, Vec::new()).is_err(), "{source} rendered");
        }
    }

    #[test]
    fn replace_matches_python_and_refuses_a_count() {
        assert_eq!(
            rendered("{{ 'a/think b/think'.replace('/think', '') }}"),
            "a b"
        );
        assert_eq!(rendered("{{ 'ab'.replace('', '-') }}"), "-a-b-");
        assert_eq!(rendered("{{ 'aa'.replace('a', 'b', -1) }}"), "bb");
        assert!(render("{{ 'aa'.replace('a', 'b', 1) }}", Vec::new()).is_err());
    }

    #[test]
    fn items_keys_and_values_walk_a_map_and_nothing_else() {
        let source = r#"{% for k, v in {"a": 1, "b": 2}.items() %}{{ k }}={{ v }},{% endfor %}"#;
        assert_eq!(rendered(source), "a=1,b=2,");
        assert_eq!(rendered(r#"{{ {"a": 1}.keys() | tojson }}"#), r#"["a"]"#);
        assert_eq!(rendered(r#"{{ {"a": 1}.values() | tojson }}"#), "[1]");
        assert!(render("{{ [1].items() }}", Vec::new()).is_err());
    }
}
