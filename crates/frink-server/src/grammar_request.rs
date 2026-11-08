//! Where a request's stated constraint becomes a compiled grammar.
//!
//! Exactly one function answers "what grammar does this request run
//! under", because a request can state the constraint several ways and
//! only one of them may win:
//!
//! | Request field | Today |
//! |---|---|
//! | `"grammar": "<GBNF>"` | compiled and applied (llama.cpp's field, same spelling) |
//! | `response_format: {"type": "json_schema", …}` | the `schema` is compiled to GBNF and applied |
//! | `response_format: {"type": "json_object"}` | not a grammar; the character-class mask in [`crate::json_mode`], decided by the caller of this module |
//! | `response_format` naming any other `type` | 400, naming the type |
//! | both a `grammar` and a `json_schema` | 400: two constraints on one generation, send one |
//!
//! The schema half used to be a 501, because a JSON schema has to become
//! a GBNF grammar before any of this can run and that converter did not
//! exist. It does now --
//! [`frink_models::grammar::json_schema`], a port of llama.cpp's
//! `common/json-schema-to-grammar.cpp` -- so everything after the
//! conversion (masking, accepting, both decode loops, the batcher) is the
//! same path `"grammar"` already took.
//!
//! # What inside `json_schema` is honoured, and what is refused
//!
//! The OpenAI object is `{"name": …, "description": …, "schema": {…},
//! "strict": bool}` and [`JsonSchemaSpec`] is destructured exhaustively,
//! so a field added to that struct cannot be dropped on the floor:
//!
//! | Member | Treatment |
//! |---|---|
//! | `schema` | compiled to GBNF; required, because there is nothing to enforce without it |
//! | `name`, `description` | labels. They constrain nothing, so ignoring them cannot widen or narrow what is accepted |
//! | `strict: true`, or absent | the schema is ENFORCED by the grammar |
//! | `strict: false` | 400, naming the field |
//! | anything else | 400, naming the field (`deny_unknown_fields`) |
//!
//! `strict: false` is a refusal rather than a shrug because OpenAI's two
//! modes are not the same promise: `false` asks for best-effort guidance
//! and permits schemas strict mode rejects, while this server has exactly
//! one behaviour for a schema, which is to enforce it token by token.
//! Serving that under `strict: false` is answering a question the caller
//! did not ask, and doing it silently is how a client learns the wrong
//! thing about what its schema bought it. An OMITTED `strict` is enforced
//! (OpenAI defaults it to `false`, and this table is the notice): a caller
//! who wrote nothing stated no preference, and the stronger guarantee is
//! the only one this engine has.
//!
//! A schema the converter will not compile is a **400 naming the keyword
//! and where it sits**, never a 500 and never a grammar that is only
//! nearly the caller's schema: that would be served with a 200 and read
//! as compliance. The converter is deliberately stricter than llama.cpp
//! here -- upstream discards a keyword no branch tested, so
//! `{"type": "string", "pattern": "^[a-z]+$"}` compiles upstream to a
//! grammar for *any* string.
//!
//! # Property order, stated because it is observable
//!
//! The grammar requires an object's `required` members in a fixed order,
//! the way llama.cpp's does: the order the schema declares them in. A
//! `response_format` reaches this module as a `serde_json::Value`, which
//! keeps that order because the workspace builds `serde_json` with
//! `preserve_order`. A forced tool call's `arguments`
//! ([`crate::tool_grammar`]) follow the same order.
//!
//! # Why a refusal is never a silent ignore
//!
//! `logit_bias` was declared on these requests only so it could be
//! refused by name, because serde dropped an undeclared field and the
//! caller got a 200 whose answer was sampled from unbiased logits. A
//! `json_schema` member that is accepted and not applied is the same
//! failure with more expensive consequences: the caller parses the
//! answer.

use std::sync::Arc;

use axum::http::StatusCode;
use axum::Json;
use frink_models::grammar::json_schema::json_schema_to_grammar_value;
use frink_models::grammar::{Grammar, SchemaError};
use serde::Deserialize;

use crate::ApiError;

/// The root rule name every grammar here starts from. llama.cpp's
/// convention, and the one every published `.gbnf` uses.
const ROOT: &str = "root";

/// `response_format`, as the wire spells it.
///
/// `deny_unknown_fields` is the point: a member this server does not act
/// on must be refused by name rather than dropped by serde, and the
/// derive is the only version of that rule which cannot fall out of date
/// with the struct.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseFormat {
    /// Optional here only so that its absence is answered with this
    /// module's own message instead of a serde one.
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    json_schema: Option<JsonSchemaSpec>,
}

/// The `json_schema` member of an OpenAI `response_format`.
///
/// Every field is destructured in [`schema_grammar`] with no `..`, so
/// adding one here stops compiling until somebody decides whether it
/// changes what is accepted.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonSchemaSpec {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    schema: Option<serde_json::Value>,
    #[serde(default)]
    strict: Option<bool>,
}

/// Compile GBNF text, or refuse with the parser's own diagnostic.
///
/// A 400: a grammar that does not parse is a fact about the request,
/// and the message carries the position the parser stopped at, which is
/// the only thing that makes one debuggable from the client side.
pub(crate) fn compile(src: &str) -> Result<Arc<Grammar>, ApiError> {
    Grammar::from_str_with_root(src, ROOT)
        .map(Arc::new)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("grammar: {e}"),
                        "type": "invalid_request_error",
                        "param": "grammar",
                    }
                })),
            )
        })
}

/// Compile a bare JSON Schema value to a grammar.
///
/// `param` is the field the caller sent it in, so the 400 points at the
/// request rather than at this module's idea of where a schema lives:
/// `/v1/chat/completions` spells it `response_format.json_schema.schema`
/// and llama.cpp's `/completion` spells it `json_schema`.
pub(crate) fn from_schema(
    schema: &serde_json::Value,
    param: &str,
) -> Result<Arc<Grammar>, ApiError> {
    let text = json_schema_to_grammar_value(schema).map_err(|e| schema_refused(&e, param))?;
    // The converter re-parses its own output before returning it, so a
    // failure here is a defect in the converter and not a fact about
    // the request. Saying 400 would send the caller looking at a schema
    // that is fine.
    Grammar::from_str_with_root(&text, ROOT)
        .map(Arc::new)
        .map_err(|e| {
            internal(format!(
                "the grammar generated from {param} does not compile: {e}"
            ))
        })
}

/// The grammar one request runs under, from every field that can state
/// one.
///
/// A request carrying BOTH a `grammar` and a `response_format` schema has
/// stated two different constraints on one generation, and serving
/// whichever we compiled last is not answering either question. That is a
/// 400, the same way a forced `tool_choice` beside a `grammar` already
/// is.
pub(crate) fn for_request(
    grammar: Option<&str>,
    response_format: Option<&serde_json::Value>,
) -> Result<Option<Arc<Grammar>>, ApiError> {
    // Whitespace is not a grammar. Left as `None` rather than sent to
    // the parser, which would 400 on an empty string a client library
    // sent by defaulting the field.
    let gbnf = grammar.map(str::trim).filter(|s| !s.is_empty());
    let from_format = match response_format {
        Some(fmt) => schema_grammar(fmt)?,
        None => None,
    };
    match (gbnf, from_format) {
        (Some(_), Some(_)) => Err(two_constraints()),
        (Some(src), None) => compile(src).map(Some),
        (None, from_format @ Some(_)) => Ok(from_format),
        (None, None) => Ok(None),
    }
}

/// The whole of what a `response_format` says about a grammar.
///
/// `Ok(None)` means "this format states no grammar", which today is
/// `json_object`: that is a character-class mask in [`crate::json_mode`],
/// decided by this module's caller, and it must not be caught by the
/// schema arm.
fn schema_grammar(fmt: &serde_json::Value) -> Result<Option<Arc<Grammar>>, ApiError> {
    let parsed: ResponseFormat = serde_json::from_value(fmt.clone())
        .map_err(|e| invalid(format!("response_format: {e}"), "response_format"))?;
    let ResponseFormat { kind, json_schema } = parsed;
    match kind.as_deref() {
        Some("json_schema") => {}
        Some("json_object") => {
            if json_schema.is_some() {
                return Err(invalid(
                    "response_format carries a \"json_schema\" but asks for type \"json_object\", \
                     which enforces nothing but the character class; send type \"json_schema\" to \
                     have the schema enforced",
                    "response_format",
                ));
            }
            return Ok(None);
        }
        Some(other) => {
            return Err(invalid(
                format!(
                    "response_format type {other:?} is not supported (only json_object and \
                     json_schema)"
                ),
                "response_format",
            ));
        }
        None => {
            return Err(invalid(
                "response_format must include \"type\" (only json_object and json_schema are \
                 supported)",
                "response_format",
            ));
        }
    }
    let spec = json_schema.ok_or_else(|| {
        invalid(
            "response_format type \"json_schema\" carries the schema in a \"json_schema\" object: \
             {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"…\", \"schema\": {…}}}",
            "response_format.json_schema",
        )
    })?;
    // Exhaustive, with no `..`: a member added to the wire struct must
    // be decided here rather than silently accepted and ignored.
    let JsonSchemaSpec {
        // Labels. They name the schema for the caller's own logs and
        // constrain no byte of the answer, so ignoring them cannot make
        // the grammar accept or reject anything.
        name: _name,
        description: _description,
        schema,
        strict,
    } = spec;
    // `Some(true)` and `None` both enforce -- absent is enforced, and
    // this module's table says so rather than leaving it to be
    // discovered. Only the explicit contradiction is refused.
    if strict == Some(false) {
        return Err(invalid(
            "response_format json_schema with \"strict\": false is not supported: this server has \
             one behaviour for a schema, which is to enforce it with a grammar, and serving that \
             under a flag asking for best-effort guidance would report a guarantee the caller \
             declined. Send \"strict\": true to have the schema enforced, or response_format \
             {\"type\": \"json_object\"} for the best-effort character-class mode.",
            "response_format.json_schema.strict",
        ));
    }
    let schema = schema.ok_or_else(|| {
        invalid(
            "response_format json_schema must carry a \"schema\"; there is nothing to enforce \
             without one",
            "response_format.json_schema.schema",
        )
    })?;
    from_schema(&schema, "response_format.json_schema.schema").map(Some)
}

/// A schema the converter will not compile, reported as what it refused.
///
/// [`SchemaError`]'s own `Display` names the keyword and the subschema it
/// sits in, which is the only thing that makes this fixable from the
/// client side.
fn schema_refused(err: &SchemaError, param: &str) -> ApiError {
    invalid(format!("{param}: {err}"), param)
}

/// Two constraints, one generation.
fn two_constraints() -> ApiError {
    invalid(
        "a \"grammar\" and a response_format \"json_schema\" are two different constraints on the \
         same generation; send one",
        "response_format",
    )
}

fn invalid(message: impl Into<String>, param: &str) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "param": param,
            }
        })),
    )
}

fn internal(message: String) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "server_error",
            }
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_format(schema: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        })
    }

    /// Drive `text` through the grammar the way a decode loop would and
    /// report whether the parse is complete.
    fn feed(grammar: &Grammar, pieces: &[&str]) -> Result<bool, String> {
        let mut g = grammar.clone();
        for (i, piece) in pieces.iter().enumerate() {
            g.accept_token(i as u32, piece.as_bytes())
                .map_err(|e| format!("piece {piece:?}: {e}"))?;
        }
        Ok(g.allows_eog())
    }

    #[test]
    fn a_gbnf_grammar_compiles_and_starts_at_root() {
        let g = for_request(Some(r#"root ::= "a"+"#), None)
            .expect("a valid grammar is not an error")
            .expect("a grammar was asked for");
        assert!(!g.stacks().is_empty(), "the machine has a live stack");
    }

    #[test]
    fn no_grammar_field_means_no_grammar() {
        assert!(for_request(None, None).unwrap().is_none());
        // Whitespace is not a grammar. Left as `None` rather than sent
        // to the parser, which would 400 on an empty string a client
        // library sent by defaulting the field.
        assert!(for_request(Some("   "), None).unwrap().is_none());
    }

    #[test]
    fn an_unparseable_grammar_is_a_400_naming_the_field() {
        // An unterminated string literal. Note that `root ::=` alone is
        // NOT this case: an empty alternate is legal GBNF, and it
        // compiles to a grammar that is satisfied before it starts.
        let (status, Json(body)) =
            for_request(Some(r#"root ::= "a"#), None).expect_err("this does not parse");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "grammar");
    }

    /// A grammar whose root rule is spelled something else is refused
    /// too, rather than starting from whichever rule came first.
    #[test]
    fn a_grammar_with_no_root_rule_is_refused() {
        let (status, _) =
            for_request(Some(r#"start ::= "a""#), None).expect_err("there is no root rule");
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// The headline, and what used to be a 501: the schema is not merely
    /// "some JSON", it is THIS schema. A required property cannot be
    /// omitted and an `enum` member cannot be invented.
    #[test]
    fn a_json_schema_compiles_to_a_grammar_that_constrains() {
        let fmt = schema_format(serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "hot": {"type": "boolean"},
            },
            "required": ["city", "hot"],
            "additionalProperties": false,
        }));
        let g = for_request(None, Some(&fmt))
            .expect("this schema converts")
            .expect("and it states a grammar");

        assert!(
            feed(&g, &[r#"{"city": "Rome", "hot": true}"#]).expect("the document is schema-valid"),
            "a complete document must finish the parse"
        );
        assert!(
            feed(&g, &[r#"{"city": "Rome"}"#]).is_err(),
            "a required property must not be droppable"
        );
        assert!(
            feed(&g, &[r#"{"city": "Rome", "hot": true, "extra"#]).is_err(),
            "additionalProperties: false must close the object"
        );
        assert!(
            feed(&g, &[r#"{"city": 3"#]).is_err(),
            "a property's own type must be enforced"
        );
    }

    /// A schema the converter refuses is a 400 that says WHICH keyword,
    /// not a 500 and not a grammar that is approximately the schema.
    /// `allOf` is a real keyword with no grammar (schema intersection),
    /// which is why upstream silently drops it.
    #[test]
    fn an_unconvertible_schema_is_a_400_naming_the_keyword() {
        let fmt = schema_format(serde_json::json!({
            "type": "object",
            "properties": {"x": {"allOf": [{"type": "string"}]}},
        }));
        let (status, Json(body)) = for_request(None, Some(&fmt)).expect_err("allOf has no grammar");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(
            message.contains("allOf"),
            "the refusal must name the keyword: {message}"
        );
        assert_eq!(body["error"]["param"], "response_format.json_schema.schema");
    }

    /// Two constraints on one generation. Compiling both and serving
    /// whichever came last answers neither question.
    #[test]
    fn a_grammar_and_a_schema_together_are_refused() {
        let fmt = schema_format(serde_json::json!({"type": "string"}));
        let (status, Json(body)) = for_request(Some(r#"root ::= "a""#), Some(&fmt))
            .expect_err("two constraints, one generation");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(message.contains("two different constraints"), "{message}");

        // Each alone is still fine, so the refusal is about the pair.
        assert!(for_request(Some(r#"root ::= "a""#), None)
            .unwrap()
            .is_some());
        assert!(for_request(None, Some(&fmt)).unwrap().is_some());
    }

    /// `strict: false` asks for best-effort guidance, which this server
    /// does not have. Enforcing anyway and answering 200 would report a
    /// guarantee the caller explicitly declined.
    #[test]
    fn strict_false_is_refused_rather_than_treated_as_strict_true() {
        let mut fmt = schema_format(serde_json::json!({"type": "string"}));
        fmt["json_schema"]["strict"] = serde_json::json!(false);
        let (status, Json(body)) =
            for_request(None, Some(&fmt)).expect_err("best-effort is not a mode here");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "response_format.json_schema.strict");

        // `true` and absent both enforce, and absent is the OpenAI
        // default -- which is why the table in the module docs says so
        // out loud rather than leaving it to be discovered.
        fmt["json_schema"]["strict"] = serde_json::json!(true);
        assert!(for_request(None, Some(&fmt)).unwrap().is_some());
        let absent = schema_format(serde_json::json!({"type": "string"}));
        assert!(for_request(None, Some(&absent)).unwrap().is_some());
    }

    /// Anything inside `json_schema` this server does not act on is
    /// refused BY NAME. Serde's `deny_unknown_fields` is what makes that
    /// true of a field nobody has thought of yet.
    #[test]
    fn an_unknown_member_of_json_schema_is_refused_by_name() {
        let mut fmt = schema_format(serde_json::json!({"type": "string"}));
        fmt["json_schema"]["max_depth"] = serde_json::json!(3);
        let (status, Json(body)) =
            for_request(None, Some(&fmt)).expect_err("nothing honours max_depth");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(message.contains("max_depth"), "{message}");

        // And the same at the top level of `response_format`.
        let stray = serde_json::json!({"type": "json_object", "schema": {"type": "string"}});
        let (status, Json(body)) = for_request(None, Some(&stray)).expect_err(
            "`schema` is not a \
             member of response_format",
        );
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"]
            .as_str()
            .expect("a message")
            .contains("schema"));
    }

    /// `name` and `description` are labels; honouring them by ignoring
    /// them is safe precisely because they constrain nothing, and the
    /// grammar must be the same with or without them.
    #[test]
    fn the_label_members_do_not_change_the_grammar() {
        let schema = serde_json::json!({"type": "boolean"});
        let bare = serde_json::json!({"type": "json_schema", "json_schema": {"schema": schema}});
        let labelled = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "yes_or_no",
                "description": "whether the answer is yes",
                "schema": schema,
                "strict": true,
            },
        });
        for fmt in [&bare, &labelled] {
            let g = for_request(None, Some(fmt)).unwrap().expect("a grammar");
            assert!(feed(&g, &["true"]).unwrap());
            assert!(
                feed(&g, &["\"true\""]).is_err(),
                "a boolean is not a string"
            );
        }
    }

    /// A `json_schema` type with nothing to compile is a 400 naming the
    /// member that is missing, not a grammar for "any JSON".
    #[test]
    fn a_json_schema_format_with_no_schema_is_refused() {
        let empty = serde_json::json!({"type": "json_schema"});
        let (status, Json(body)) =
            for_request(None, Some(&empty)).expect_err("there is no schema here");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "response_format.json_schema");

        let no_schema = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "answer"},
        });
        let (status, Json(body)) =
            for_request(None, Some(&no_schema)).expect_err("still nothing to enforce");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "response_format.json_schema.schema");
    }

    /// `json_object` is a different feature (a character-class mask) and
    /// is handled elsewhere; it must not be caught by the schema arm.
    #[test]
    fn json_object_is_not_the_schema_arm() {
        let fmt = serde_json::json!({"type": "json_object"});
        assert!(for_request(None, Some(&fmt)).unwrap().is_none());
        // And it composes with a `grammar` exactly as it did before the
        // schema arm existed: the mask is not a grammar, so there is no
        // pair of constraints to refuse.
        assert!(for_request(Some(r#"root ::= "a""#), Some(&fmt))
            .unwrap()
            .is_some());
    }

    /// A `response_format` this server cannot honour is refused by the
    /// type it named, and a `response_format` with no type at all says
    /// so, rather than being read as "unconstrained".
    #[test]
    fn an_unknown_response_format_type_is_refused_by_name() {
        let text = serde_json::json!({"type": "text"});
        let (status, Json(body)) =
            for_request(None, Some(&text)).expect_err("chat has no text arm");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"]
            .as_str()
            .expect("a message")
            .contains("\"text\""));

        let typeless = serde_json::json!({});
        let (status, Json(body)) =
            for_request(None, Some(&typeless)).expect_err("there is no type");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"]
            .as_str()
            .expect("a message")
            .contains("must include"));
    }

    /// The bare-schema entry point llama.cpp's `/completion` uses names
    /// the field the caller actually sent, so the 400 points at the
    /// request rather than at an OpenAI path that request never had.
    #[test]
    fn the_bare_schema_entry_point_names_the_callers_own_field() {
        let good = from_schema(&serde_json::json!({"type": "integer"}), "json_schema")
            .expect("integers have a grammar");
        assert!(feed(&good, &["42"]).unwrap());

        let (status, Json(body)) = from_schema(
            &serde_json::json!({"type": "object", "minProperties": 1}),
            "json_schema",
        )
        .expect_err("minProperties has no grammar");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "json_schema");
        assert!(body["error"]["message"]
            .as_str()
            .expect("a message")
            .contains("minProperties"));
    }
}
