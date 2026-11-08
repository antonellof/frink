//! JSON Schema to GBNF, a port of llama.cpp's
//! `common/json-schema-to-grammar.cpp` (MIT; see
//! `docs/THIRD_PARTY_NOTICES.md`).
//!
//! This is what `response_format: {"type": "json_schema"}` needs. The
//! output is GBNF text that [`crate::grammar::parse`] accepts, so it feeds
//! the same pushdown machine as a hand-written grammar -- and every
//! conversion re-parses its own output before returning it, because a
//! grammar that does not parse is a bug in this module and should not
//! reach a caller as a string.
//!
//! # What is ported
//!
//! | Keyword | Behaviour |
//! |---|---|
//! | `type` | `object` `array` `string` `number` `integer` `boolean` `null`, and an array of them as a union |
//! | `properties`, `required` | in declaration order, required first |
//! | `additionalProperties` | `false`/absent closes the object, a schema types the tail, `true` opens it |
//! | `items`, `prefixItems` | a single schema is a list, an array is a fixed tuple |
//! | `minItems`, `maxItems` | beside a single-schema `items` |
//! | `minLength`, `maxLength` | beside an explicit `"type": "string"` |
//! | `enum`, `const` | literal alternatives, JSON-encoded |
//! | `oneOf`, `anyOf` | union of alternatives |
//! | `$ref`, `$defs`, `definitions` | same-document `#/` pointers, recursion included |
//! | `format` | `date`, `time`, `date-time`, `uuid`, `uuid1`..`uuid5` |
//! | `pattern` | an anchored ECMA-262 regex, via [`pattern`] |
//!
//! Annotations (`title`, `description`, `default`, `examples`, `$schema`,
//! `$id`, `$comment`, `deprecated`, `readOnly`, `writeOnly`) are ignored,
//! which is safe: they constrain nothing.
//!
//! # What is refused, by name
//!
//! Everything else, via [`SchemaError::UnsupportedKeyword`]. The rule is
//! in [`branch`]: a keyword the chosen branch did not act on is refused
//! unless the schema's declared `type` makes it vacuous -- `items` beside
//! `"type": "null"` says nothing about a null, so it is dropped, while
//! `pattern` beside `"type": "string"` is refused.
//!
//! The notable refusals are `allOf` (schema intersection), the numeric
//! bounds `minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum`
//! (llama.cpp builds a digit-by-digit range grammar for integers), `not` /
//! `if` / `then` / `else`, `patternProperties`, `propertyNames`,
//! `uniqueItems`, `minProperties` / `maxProperties`, `multipleOf`, the
//! `dependent*` family, and any `format` outside the six above.
//!
//! **This is stricter than llama.cpp on purpose.** Upstream's `visit` is a
//! chain of `if`s: a keyword no branch tested falls off the end and is
//! discarded, so `{"type": "string", "pattern": "^[a-z]+$"}` compiles to a
//! grammar for *any* string. That grammar accepts documents the caller
//! declared invalid, which is a wrong answer, not a missing feature. Per
//! `CLAUDE.md`, a refusal is coverage.
//!
//! # Where this differs from llama.cpp on schemas it accepts
//!
//! - `_not_strings` writes property-name bytes into a GBNF character class
//!   unescaped, so upstream emits an unparseable grammar for a property
//!   named `a-b`, `a]b` or `añb`. This port escapes the class and keys the
//!   trie on `char` rather than `u8`.
//! - JSON Pointer escapes (`~1`, `~0`) in a `$ref` are decoded. Upstream
//!   splits on `/` and cannot address such a member at all.
//! - Remote (`https://`) `$ref`s are refused rather than fetched.
//! - `pattern` compiles the subset of ECMA-262 that upstream compiles, but
//!   refuses several inputs upstream mishandles rather than reproducing the
//!   mishandling: a lookaround group (upstream warns, then silently drops
//!   the group), an escape its own GBNF parser rejects (`\s`, `\b`, a
//!   backreference, a dangling `\`), a stray `]` or `}` (upstream loops
//!   forever), `*` with nothing before it (upstream reads past the end of a
//!   vector), and a top-level `)` (upstream returns early, discarding the
//!   rest of the pattern). `\d` / `\D` / `\w` / `\W` are translated to
//!   their exact ECMA-262 classes, where upstream copies them through into
//!   a grammar that does not parse. See [`pattern`].
//! - A top-level schema whose rule ends up named something other than
//!   `root` is refused ([`SchemaError::RootDisplaced`]) instead of silently
//!   producing a grammar that starts from a subschema; upstream reaches
//!   this whenever a property is named `""`.
//!
//! # Property order
//!
//! The order of `properties` is part of the accepted language: required
//! members are emitted in declaration order. [`json_schema_to_grammar`]
//! parses the schema text with [`value::JsonValue`], which keeps document
//! order the way llama.cpp's `nlohmann::ordered_json` does.
//! [`json_schema_to_grammar_value`] carries the order its
//! `serde_json::Value` has, which is document order: this workspace builds
//! `serde_json` with `preserve_order`.

mod branch;
mod converter;
mod error;
mod pattern;
mod primitives;
mod refs;
mod value;

#[cfg(test)]
mod tests;

pub use error::SchemaError;
pub use value::JsonValue;

use converter::Converter;

/// Convert JSON Schema text to a GBNF grammar.
///
/// This is the entry point to prefer: it keeps `properties` in the order
/// the schema declares them.
pub fn json_schema_to_grammar(schema_text: &str) -> Result<String, SchemaError> {
    let schema = JsonValue::parse(schema_text).map_err(|e| SchemaError::NotJson(e.to_string()))?;
    convert(&schema)
}

/// Convert an already-parsed `serde_json::Value` schema.
///
/// Properties keep the order the `Value` holds them in, which is the
/// schema's own (see the module docs on property order).
pub fn json_schema_to_grammar_value(schema: &serde_json::Value) -> Result<String, SchemaError> {
    convert(&JsonValue::from(schema))
}

/// Several schemas and some hand-written rules, compiled into ONE
/// grammar.
///
/// `common_grammar_builder` upstream (`common/chat.cpp` builds every
/// tool-call grammar through it): each schema becomes a NAMED rule
/// instead of `root`, and the caller writes the rule that composes them.
/// One converter, so the shared `space` / `char` / `string` builtins are
/// emitted once and a rule name used by two schemas is disambiguated
/// rather than silently redefined.
///
/// The single-schema [`json_schema_to_grammar`] is not a special case of
/// this and is left alone: it must produce `root` itself, and refuses if
/// a subschema takes that name first.
///
/// ```
/// use frink_models::grammar::json_schema::GrammarBuilder;
/// let mut b = GrammarBuilder::new();
/// let args = b
///     .add_schema_value("get-weather-args", &serde_json::json!({
///         "type": "object",
///         "properties": {"city": {"type": "string"}},
///         "required": ["city"],
///     }))
///     .unwrap();
/// b.add_rule("root", &format!("\"call \" {args}"));
/// let grammar = b.finish().unwrap();
/// assert!(grammar.contains("root ::="));
/// ```
pub struct GrammarBuilder {
    conv: Converter,
}

impl Default for GrammarBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GrammarBuilder {
    pub fn new() -> Self {
        Self {
            conv: Converter::new(std::collections::HashMap::new()),
        }
    }

    /// Compile `schema` into a rule named `name`, and return the name it
    /// actually got.
    ///
    /// The returned name is what the caller must reference: a name
    /// already bound to a different body gets a numeric suffix, which is
    /// how two tools with parameter sets that collapse to the same rule
    /// name stay distinct.
    pub fn add_schema(&mut self, name: &str, schema: &JsonValue) -> Result<String, SchemaError> {
        self.conv.add_refs(refs::collect(schema)?)?;
        self.conv.visit(schema, name)
    }

    /// As [`Self::add_schema`], for an already-parsed `serde_json::Value`.
    ///
    /// Properties keep the order the `Value` holds them in (see the
    /// module docs on property order).
    pub fn add_schema_value(
        &mut self,
        name: &str,
        schema: &serde_json::Value,
    ) -> Result<String, SchemaError> {
        self.add_schema(name, &JsonValue::from(schema))
    }

    /// Add a hand-written rule body, and return the name it got.
    pub fn add_rule(&mut self, name: &str, body: &str) -> String {
        self.conv.add_rule(name, body)
    }

    /// Emit the grammar text, checking it parses.
    ///
    /// Refuses a grammar with no `root`: it would compile to
    /// [`GrammarError::MissingRoot`](crate::grammar::GrammarError) at
    /// whatever call site tried to run it, which is further from the
    /// mistake than here.
    pub fn finish(self) -> Result<String, SchemaError> {
        if !self.conv.has_rule("root") {
            return Err(SchemaError::RootDisplaced {
                got: "<no root rule was added>".to_string(),
            });
        }
        let grammar = self.conv.finish();
        crate::grammar::parse(&grammar)?;
        Ok(grammar)
    }
}

/// `json_schema_to_grammar` / `build_grammar`: resolve refs, visit the
/// root, emit, and check the result against this repo's own GBNF parser.
fn convert(schema: &JsonValue) -> Result<String, SchemaError> {
    let refs = refs::collect(schema)?;
    let mut converter = Converter::new(refs);
    let root = converter.visit(schema, "")?;
    if root != "root" {
        return Err(SchemaError::RootDisplaced { got: root });
    }
    let grammar = converter.finish();
    // A grammar this module cannot hand to its own parser is a defect
    // here, and it is cheaper to find it now than in a sampler.
    crate::grammar::parse(&grammar)?;
    Ok(grammar)
}
