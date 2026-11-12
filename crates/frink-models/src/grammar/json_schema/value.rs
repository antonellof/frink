//! An order-preserving JSON value.
//!
//! `serde_json::Value` keeps an object's document order only while the
//! crate's `preserve_order` feature is on. This workspace turns it on
//! (#513), and this type keeps the order either way. Property order is
//! not cosmetic here: [`super::converter`] emits the *required* properties of
//! an object in the order `properties` declares them, so the order a
//! schema was written in is part of the language the grammar accepts.
//! llama.cpp gets this from `nlohmann::ordered_json`.
//!
//! So the converter runs on this type instead, which keeps objects as a
//! `Vec<(String, JsonValue)>` and therefore preserves document order when
//! parsed from text with [`JsonValue::parse`]. A `serde_json::Value`
//! converts in ([`JsonValue::from`]) with the order it has.

use serde::de::{Deserializer, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use std::fmt;
use std::fmt::Write as _;

/// A JSON document that remembers the order its object keys came in.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Parse JSON text, keeping object keys in document order.
    pub fn parse(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// The member named `key`, if this is an object that has one. On a
    /// duplicate key the first wins, matching `nlohmann::json`'s parser.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Whether this is an object with a member named `key`.
    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(entries) => Some(entries.as_slice()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The value as a non-negative integer, for `minItems` and friends.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Number(n) => n.as_u64(),
            _ => None,
        }
    }

    /// The JSON type name, for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            JsonValue::Null => "null",
            JsonValue::Bool(_) => "boolean",
            JsonValue::Number(_) => "number",
            JsonValue::String(_) => "string",
            JsonValue::Array(_) => "array",
            JsonValue::Object(_) => "object",
        }
    }

    /// `nlohmann::json::dump()`: compact, no spaces. The `const` and
    /// `enum` rules are literally this text wrapped in GBNF quotes, so it
    /// has to agree with what a JSON encoder would emit for the same
    /// value or the grammar forbids the document the schema allows.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        self.dump_into(&mut out);
        out
    }

    fn dump_into(&self, out: &mut String) {
        match self {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(true) => out.push_str("true"),
            JsonValue::Bool(false) => out.push_str("false"),
            JsonValue::Number(n) => {
                let _ = write!(out, "{n}");
            }
            JsonValue::String(s) => escape_json_string(s, out),
            JsonValue::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.dump_into(out);
                }
                out.push(']');
            }
            JsonValue::Object(entries) => {
                out.push('{');
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    escape_json_string(k, out);
                    out.push(':');
                    v.dump_into(out);
                }
                out.push('}');
            }
        }
    }
}

/// JSON string escaping as `nlohmann::json::dump()` does it: the seven
/// short escapes, `\u00xx` for the remaining C0 controls, everything else
/// verbatim. `/` is *not* escaped, and neither are non-ASCII codepoints.
fn escape_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

impl From<&serde_json::Value> for JsonValue {
    fn from(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => JsonValue::Null,
            serde_json::Value::Bool(b) => JsonValue::Bool(*b),
            serde_json::Value::Number(n) => JsonValue::Number(n.clone()),
            serde_json::Value::String(s) => JsonValue::String(s.clone()),
            serde_json::Value::Array(items) => {
                JsonValue::Array(items.iter().map(JsonValue::from).collect())
            }
            serde_json::Value::Object(map) => JsonValue::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), JsonValue::from(v)))
                    .collect(),
            ),
        }
    }
}

impl<'de> Deserialize<'de> for JsonValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonValueVisitor)
    }
}

struct JsonValueVisitor;

impl<'de> Visitor<'de> for JsonValueVisitor {
    type Value = JsonValue;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_unit<E>(self) -> Result<JsonValue, E> {
        Ok(JsonValue::Null)
    }

    fn visit_none<E>(self) -> Result<JsonValue, E> {
        Ok(JsonValue::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<JsonValue, D::Error> {
        d.deserialize_any(self)
    }

    fn visit_bool<E>(self, v: bool) -> Result<JsonValue, E> {
        Ok(JsonValue::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<JsonValue, E> {
        Ok(JsonValue::Number(v.into()))
    }

    fn visit_u64<E>(self, v: u64) -> Result<JsonValue, E> {
        Ok(JsonValue::Number(v.into()))
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<JsonValue, E> {
        // A non-finite float cannot be written back as JSON; serde_json
        // maps it to null on the way out, so do the same here rather than
        // inventing a literal no encoder would produce.
        Ok(match serde_json::Number::from_f64(v) {
            Some(n) => JsonValue::Number(n),
            None => JsonValue::Null,
        })
    }

    fn visit_str<E>(self, v: &str) -> Result<JsonValue, E> {
        Ok(JsonValue::String(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> Result<JsonValue, E> {
        Ok(JsonValue::String(v))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<JsonValue, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(JsonValue::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<JsonValue, A::Error> {
        // `next_entry` yields in document order, which is the whole point
        // of this type.
        let mut entries: Vec<(String, JsonValue)> = Vec::new();
        while let Some((k, v)) = map.next_entry::<String, JsonValue>()? {
            if !entries.iter().any(|(existing, _)| *existing == k) {
                entries.push((k, v));
            }
        }
        Ok(JsonValue::Object(entries))
    }
}
