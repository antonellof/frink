//! The golden corpus from llama.cpp's `tests/test-json-schema-to-grammar.cpp`.
//!
//! Every expectation below is upstream's, byte for byte, for a schema this
//! port accepts. Two are edited, and both edits are noted where they
//! appear. The refusal cases at the bottom are this port's own: upstream
//! either drops the keyword and emits a grammar wider than the schema, or
//! -- for the pattern cases in [`patterns`] -- emits one that does not
//! parse at all.
//!
//! [`super::convert`] already re-parses everything it emits, so a golden
//! that matches is also a grammar `crate::grammar::parse` accepts; the
//! `every_case_parses` test makes that guarantee explicit rather than
//! incidental.

use super::{json_schema_to_grammar, json_schema_to_grammar_value, SchemaError};

/// Upstream's comparison: leading indentation is not part of the grammar.
fn normalize(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[track_caller]
fn check(schema: &str, expected: &str) {
    match json_schema_to_grammar(schema) {
        Ok(got) => assert_eq!(normalize(&got), normalize(expected)),
        Err(e) => panic!("conversion failed: {e}"),
    }
}

#[track_caller]
fn refuse(schema: &str) -> SchemaError {
    match json_schema_to_grammar(schema) {
        Ok(g) => panic!("expected a refusal, got a grammar:\n{g}"),
        Err(e) => e,
    }
}

// -- primitives ------------------------------------------------------

#[test]
fn empty_schema_is_any_object() {
    check(
        "{}",
        r##"
        array ::= "[" space ( value ("," space value)* )? space "]"
        boolean ::= ("true" | "false")
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        null ::= "null"
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
        root ::= object
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        value ::= object | array | string | number | boolean | null
        "##,
    );
}

#[test]
fn string_number_integer_boolean() {
    check(
        r##"{"type": "string"}"##,
        r##"
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        root ::= "\"" char* "\""
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
    check(
        r##"{"type": "number"}"##,
        r##"
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        root ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
    check(
        r##"{"type": "integer"}"##,
        r##"
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        root ::= ("-"? integral-part)
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
    check(
        r##"{"type": "boolean"}"##,
        r##"
        root ::= ("true" | "false")
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

#[test]
fn string_lengths() {
    let char_rule = r##"char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"##;
    let space = r##"space ::= | " " | "\n"{1,2} [ \t]{0,20}"##;
    for (schema, root) in [
        (
            r##"{"type": "string", "minLength": 1}"##,
            r##"root ::= "\"" char+ "\"""##,
        ),
        (
            r##"{"type": "string", "minLength": 3}"##,
            r##"root ::= "\"" char{3,} "\"""##,
        ),
        (
            r##"{"type": "string", "maxLength": 3}"##,
            r##"root ::= "\"" char{0,3} "\"""##,
        ),
        (
            r##"{"type": "string", "minLength": 1, "maxLength": 4}"##,
            r##"root ::= "\"" char{1,4} "\"""##,
        ),
    ] {
        check(schema, &format!("{char_rule}\n{root}\n{space}"));
    }
}

#[test]
fn const_and_enum() {
    check(
        r##"{"const": "foo"}"##,
        r##"
        root ::= "\"foo\""
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
    check(
        r##"{"const": 123}"##,
        r##"
        root ::= "123"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
    check(
        r##"{"enum": ["red", "amber", "green", null, 42, ["foo"]]}"##,
        r##"
        root ::= ("\"red\"" | "\"amber\"" | "\"green\"" | "null" | "42" | "[\"foo\"]")
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

#[test]
fn literal_string_with_escapes() {
    check(
        r##"{
            "properties": {
                "code": {
                    "const": " \r \n \" \\ ",
                    "description": "Generated code",
                    "title": "Code",
                    "type": "string"
                }
            },
            "required": ["code"],
            "title": "DecoderResponse",
            "type": "object"
        }"##,
        r##"
        code ::= "\" \\r \\n \\\" \\\\ \""
        code-kv ::= "\"code\"" space ":" space code
        root ::= "{" space code-kv space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

#[test]
fn description_only_is_unconstrained() {
    check(
        r##"{"description": "The 0-based index of the last line."}"##,
        r##"
        array ::= "[" space ( value ("," space value)* )? space "]"
        boolean ::= ("true" | "false")
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        null ::= "null"
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
        root ::= value
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        value ::= object | array | string | number | boolean | null
        "##,
    );
}

mod patterns;

// -- string formats --------------------------------------------------

#[test]
fn exotic_formats() {
    check(
        r##"{
            "items": [
                { "format": "date" },
                { "format": "uuid" },
                { "format": "time" },
                { "format": "date-time" }
            ]
        }"##,
        r##"
        date ::= [0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )
        date-string ::= "\"" date "\""
        date-time ::= date "T" time
        date-time-string ::= "\"" date-time "\""
        root ::= "[" space tuple-0 "," space uuid "," space tuple-2 "," space tuple-3 space "]"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        time ::= ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )
        time-string ::= "\"" time "\""
        tuple-0 ::= date-string
        tuple-2 ::= time-string
        tuple-3 ::= date-time-string
        uuid ::= "\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""
        "##,
    );
}

// -- arrays ----------------------------------------------------------

#[test]
fn arrays_and_tuples() {
    check(
        r##"{"type": "array", "prefixItems": { "type": "string" }}"##,
        r##"
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        root ::= "[" space (string ("," space string)*)? space "]"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
    check(
        r##"{"prefixItems": [{ "type": "string" }]}"##,
        r##"
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        root ::= "[" space string space "]"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
    check(
        r##"{"prefixItems": [{ "type": "string" }, { "type": "number" }]}"##,
        r##"
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        root ::= "[" space string "," space number space "]"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
    check(
        r##"{"type": "array", "items": {}}"##,
        r##"
        array ::= "[" space ( value ("," space value)* )? space "]"
        boolean ::= ("true" | "false")
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        item ::= object
        null ::= "null"
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
        root ::= "[" space (item ("," space item)*)? space "]"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        value ::= object | array | string | number | boolean | null
        "##,
    );
}

#[test]
fn item_counts() {
    let boolean = r##"boolean ::= ("true" | "false")"##;
    let space = r##"space ::= | " " | "\n"{1,2} [ \t]{0,20}"##;
    for (schema, root) in [
        (
            r##"{"items": {"type": "boolean"}, "minItems": 2}"##,
            r##"root ::= "[" space boolean ("," space boolean)+ space "]""##,
        ),
        (
            r##"{"items": {"type": "boolean"}, "maxItems": 0}"##,
            r##"root ::= "[" space  space "]""##,
        ),
        (
            r##"{"items": {"type": "boolean"}, "maxItems": 1}"##,
            r##"root ::= "[" space boolean? space "]""##,
        ),
        (
            r##"{"items": {"type": "boolean"}, "maxItems": 2}"##,
            r##"root ::= "[" space (boolean ("," space boolean)?)? space "]""##,
        ),
    ] {
        check(schema, &format!("{boolean}\n{root}\n{space}"));
    }
}

#[test]
fn item_type_union() {
    check(
        r##"{
            "items": { "type": ["number", "integer"] },
            "minItems": 3,
            "maxItems": 5
        }"##,
        r##"
        decimal-part ::= [0-9]{1,16}
        integer ::= ("-"? integral-part)
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        item ::= number | integer
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        root ::= "[" space item ("," space item){2,4} space "]"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

#[test]
fn nullable_array_via_type_union() {
    check(
        r##"{"type": ["array", "null"], "prefixItems": { "type": "string" }}"##,
        r##"
        alternative-0 ::= "[" space (string ("," space string)*)? space "]"
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        null ::= "null"
        root ::= alternative-0 | null
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
}

// -- objects ---------------------------------------------------------

#[test]
fn required_props_in_declaration_order() {
    // Not alphabetical: `b`, `c`, `a` is the order `properties` declares.
    check(
        r##"{
            "type": "object",
            "properties": {
                "b": {"type": "string"},
                "c": {"type": "string"},
                "a": {"type": "string"}
            },
            "required": ["a", "b", "c"],
            "additionalProperties": false,
            "definitions": {}
        }"##,
        r##"
        a-kv ::= "\"a\"" space ":" space string
        b-kv ::= "\"b\"" space ":" space string
        c-kv ::= "\"c\"" space ":" space string
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        root ::= "{" space b-kv "," space c-kv "," space a-kv space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
}

#[test]
fn optional_props() {
    check(
        r##"{"properties": {"a": {"type": "string"}}, "additionalProperties": false}"##,
        r##"
        a-kv ::= "\"a\"" space ":" space string
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        root ::= "{" space  (a-kv )? space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
    check(
        r##"{
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"}
            },
            "additionalProperties": false
        }"##,
        r##"
        a-kv ::= "\"a\"" space ":" space string
        a-rest ::= ( "," space b-kv )? b-rest
        b-kv ::= "\"b\"" space ":" space string
        b-rest ::= ( "," space c-kv )?
        c-kv ::= "\"c\"" space ":" space string
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        root ::= "{" space  (a-kv a-rest | b-kv b-rest | c-kv )? space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
}

#[test]
fn required_and_optional_each_in_declaration_order() {
    check(
        r##"{
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "string"},
                "d": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b"],
            "additionalProperties": false
        }"##,
        r##"
        a-kv ::= "\"a\"" space ":" space string
        b-kv ::= "\"b\"" space ":" space string
        c-kv ::= "\"c\"" space ":" space string
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        d-kv ::= "\"d\"" space ":" space string
        d-rest ::= ( "," space c-kv )?
        root ::= "{" space b-kv "," space a-kv ( "," space ( d-kv d-rest | c-kv ) )? space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
}

#[test]
fn empty_object_variants() {
    let any_object = r##"
        array ::= "[" space ( value ("," space value)* )? space "]"
        boolean ::= ("true" | "false")
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        null ::= "null"
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
        root ::= object
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        value ::= object | array | string | number | boolean | null
        "##;
    check(r##"{"type": "object"}"##, any_object);
    check(
        r##"{"type": "object", "additionalProperties": true}"##,
        any_object,
    );
    check(
        r##"{"type": "object", "additionalProperties": false}"##,
        r##"
        root ::= "{" space  space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

#[test]
fn additional_props_typed() {
    check(
        r##"{
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "number"}}
        }"##,
        r##"
        additional-kv ::= string ":" space additional-value
        additional-value ::= "[" space (number ("," space number)*)? space "]"
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        root ::= "{" space  (additional-kv ( "," space additional-kv )* )? space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
}

#[test]
fn additional_props_beside_declared_props() {
    check(
        r##"{
            "type": "object",
            "properties": {"a": {"type": "number"}},
            "required": ["a"],
            "additionalProperties": {"type": "string"}
        }"##,
        r##"
        a-kv ::= "\"a\"" space ":" space number
        additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["]
        additional-kv ::= additional-k ":" space string
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        root ::= "{" space a-kv ( "," space ( additional-kv ( "," space additional-kv )* ) )? space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
    check(
        r##"{
            "type": "object",
            "properties": {"a": {"type": "number"}},
            "additionalProperties": {"type": "number"}
        }"##,
        r##"
        a-kv ::= "\"a\"" space ":" space number
        a-rest ::= ( "," space additional-kv )*
        additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["]
        additional-kv ::= additional-k ":" space number
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        root ::= "{" space  (a-kv a-rest | additional-kv ( "," space additional-kv )* )? space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

#[test]
fn not_strings_trie_shapes() {
    // The additional-properties key rule must exclude every declared name,
    // which is where upstream's trie earns its keep.
    check(
        r##"{
            "type": "object",
            "properties": {
                "and": {"type": "number"},
                "also": {"type": "number"}
            },
            "required": ["and"],
            "additionalProperties": {"type": "number"}
        }"##,
        r##"
        additional-k ::= ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]
        additional-kv ::= additional-k ":" space number
        also-kv ::= "\"also\"" space ":" space number
        also-rest ::= ( "," space additional-kv )*
        and-kv ::= "\"and\"" space ":" space number
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        root ::= "{" space and-kv ( "," space ( also-kv also-rest | additional-kv ( "," space additional-kv )* ) )? space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
    check(
        r##"{
            "properties": {"ab": {"type": "integer"}, "ac": {"type": "integer"}},
            "additionalProperties": {"type": "integer"}
        }"##,
        r##"
        ab-kv ::= "\"ab\"" space ":" space integer
        ab-rest ::= ( "," space ac-kv )? ac-rest
        ac-kv ::= "\"ac\"" space ":" space integer
        ac-rest ::= ( "," space additional-kv )*
        additional-k ::= ["] ( [a] ([b] char+ | [c] char+ | [^"bc] char*) | [^"a] char* )? ["]
        additional-kv ::= additional-k ":" space integer
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        integer ::= ("-"? integral-part)
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        root ::= "{" space  (ab-kv ab-rest | ac-kv ac-rest | additional-kv ( "," space additional-kv )* )? space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

#[test]
fn conflicting_names_get_suffixes() {
    check(
        r##"{
            "type": "object",
            "properties": {
                "number": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "object",
                            "properties": {"root": {"type": "number"}},
                            "required": ["root"],
                            "additionalProperties": false
                        }
                    },
                    "required": ["number"],
                    "additionalProperties": false
                }
            },
            "required": ["number"],
            "additionalProperties": false,
            "definitions": {}
        }"##,
        r##"
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        number- ::= "{" space number-number-kv space "}"
        number-kv ::= "\"number\"" space ":" space number-
        number-number ::= "{" space number-number-root-kv space "}"
        number-number-kv ::= "\"number\"" space ":" space number-number
        number-number-root-kv ::= "\"root\"" space ":" space number
        root ::= "{" space number-kv space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

// -- refs ------------------------------------------------------------

#[test]
fn top_level_ref() {
    check(
        r##"{
            "$ref": "#/definitions/foo",
            "definitions": {
                "foo": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                    "additionalProperties": false
                }
            }
        }"##,
        r##"
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        ref-definitions-foo ::= "{" space ref-definitions-foo-a-kv space "}"
        ref-definitions-foo-a-kv ::= "\"a\"" space ":" space string
        root ::= ref-definitions-foo
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
}

#[test]
fn any_of_over_refs() {
    // Upstream's schema also carries a top-level `"type": "object"`, which
    // it silently drops. This port refuses that (see `type_beside_any_of`),
    // so the expectation below is for the same schema without it.
    check(
        r##"{
            "anyOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
                "foo": {"properties": {"a": {"type": "number"}}},
                "bar": {"properties": {"b": {"type": "number"}}}
            }
        }"##,
        r##"
        alternative-0 ::= ref-definitions-foo
        alternative-1 ::= ref-definitions-bar
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        ref-definitions-bar ::= "{" space  (ref-definitions-bar-b-kv )? space "}"
        ref-definitions-bar-b-kv ::= "\"b\"" space ":" space number
        ref-definitions-foo ::= "{" space  (ref-definitions-foo-a-kv )? space "}"
        ref-definitions-foo-a-kv ::= "\"a\"" space ":" space number
        root ::= alternative-0 | alternative-1
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

#[test]
fn ref_into_a_sibling_branch() {
    check(
        r##"{
            "properties": {
                "a": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                "b": {"anyOf": [{"$ref": "#/properties/a/anyOf/0"}, {"type": "boolean"}]}
            },
            "type": "object"
        }"##,
        r##"
        a ::= string | number
        a-kv ::= "\"a\"" space ":" space a
        a-rest ::= ( "," space b-kv )?
        b ::= b-0 | boolean
        b-0 ::= string
        b-kv ::= "\"b\"" space ":" space b
        boolean ::= ("true" | "false")
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        decimal-part ::= [0-9]{1,16}
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
        root ::= "{" space  (a-kv a-rest | b-kv )? space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        string ::= "\"" char* "\""
        "##,
    );
}

#[test]
fn recursive_schema_terminates() {
    // A linked list: the ref back to `Node` must reuse the rule the outer
    // resolution is in the middle of defining rather than expand again.
    check(
        r##"{
            "$ref": "#/$defs/Node",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/$defs/Node"}},
                    "additionalProperties": false
                }
            }
        }"##,
        r##"
        ref-defs-Node ::= "{" space  (ref-defs-Node-next-kv )? space "}"
        ref-defs-Node-next ::= ref-defs-Node
        ref-defs-Node-next-kv ::= "\"next\"" space ":" space ref-defs-Node-next
        root ::= ref-defs-Node
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

// -- refusals --------------------------------------------------------

#[test]
fn unknown_and_invalid_types() {
    assert!(matches!(
        refuse(r##"{"type": "kaboom"}"##),
        SchemaError::UnknownType { .. }
    ));
    assert!(matches!(
        refuse(r##"{"type": 123}"##),
        SchemaError::UnknownType { .. }
    ));
}

#[test]
fn numeric_bounds_are_refused() {
    for schema in [
        r##"{"type": "integer", "minimum": 0}"##,
        r##"{"type": "integer", "exclusiveMaximum": 10}"##,
        r##"{"type": "number", "maximum": 1}"##,
    ] {
        assert!(matches!(
            refuse(schema),
            SchemaError::UnsupportedKeyword { .. }
        ));
    }
}

#[test]
fn intersection_and_conditional_keywords_are_refused() {
    for (schema, keyword) in [
        (
            r##"{"allOf": [{"type": "string"}, {"minLength": 1}]}"##,
            "allOf",
        ),
        (r##"{"not": {"type": "string"}}"##, "not"),
        (
            r##"{"type": "object", "properties": {}, "patternProperties": {"^a": {}}}"##,
            "patternProperties",
        ),
        (
            r##"{"type": "array", "items": {}, "uniqueItems": true}"##,
            "uniqueItems",
        ),
        (
            r##"{"type": "object", "properties": {}, "minProperties": 1}"##,
            "minProperties",
        ),
    ] {
        match refuse(schema) {
            SchemaError::UnsupportedKeyword { keyword: k, .. } => assert_eq!(k, keyword),
            other => panic!("expected {keyword:?} to be refused, got {other}"),
        }
    }
}

#[test]
fn constraints_upstream_silently_drops_are_refused() {
    // Each of these compiles upstream, to a grammar wider than the schema.
    for (schema, keyword) in [
        (r##"{"minLength": 3}"##, "minLength"),
        (
            r##"{"items": [{"type": "string"}], "minItems": 1}"##,
            "minItems",
        ),
        (
            r##"{"items": {}, "prefixItems": {"type": "string"}}"##,
            "prefixItems",
        ),
    ] {
        match refuse(schema) {
            SchemaError::UnsupportedKeyword { keyword: k, .. } => assert_eq!(k, keyword),
            other => panic!("expected {keyword:?} to be refused, got {other}"),
        }
    }
}

#[test]
fn type_beside_any_of() {
    match refuse(r##"{"anyOf": [{"type": "string"}], "type": "object"}"##) {
        SchemaError::UnsupportedKeyword { keyword, .. } => assert_eq!(keyword, "type"),
        other => panic!("expected the sibling type to be refused, got {other}"),
    }
}

#[test]
fn unsupported_format_is_named() {
    match refuse(r##"{"type": "string", "format": "email"}"##) {
        SchemaError::UnsupportedFormat { format, .. } => assert_eq!(format, "email"),
        other => panic!("expected an email format refusal, got {other}"),
    }
}

#[test]
fn required_must_be_declared() {
    match refuse(
        r##"{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"], "additionalProperties": false}"##,
    ) {
        SchemaError::BadValue { keyword, .. } => assert_eq!(keyword, "required"),
        other => panic!("expected an undeclared-required refusal, got {other}"),
    }
}

#[test]
fn remote_refs_are_refused() {
    assert!(matches!(
        refuse(r##"{"$ref": "https://example.com/schema.json#/foo"}"##),
        SchemaError::UnsupportedRef { .. }
    ));
}

#[test]
fn a_property_named_empty_displaces_root() {
    // Upstream emits `root0` for the object and leaves `root` bound to the
    // property's own rule, so its grammar starts from the wrong place.
    assert!(matches!(
        refuse(
            r##"{"properties": {"": {"type": "integer"}, "a": {"type": "integer"}}, "additionalProperties": {"type": "integer"}}"##
        ),
        SchemaError::RootDisplaced { .. }
    ));
}

#[test]
fn boolean_schemas_are_refused() {
    assert!(matches!(
        refuse(r##"{"type": "array", "items": true}"##),
        SchemaError::NotAnObject { .. }
    ));
}

#[test]
fn malformed_json_is_refused() {
    assert!(matches!(refuse(r##"{"type": "##), SchemaError::NotJson(_)));
}

// -- keywords that are vacuous, not ignored ---------------------------

#[test]
fn type_scoped_keywords_outside_their_scope_are_vacuous() {
    // `format` says nothing about an integer, so the OpenAPI habit of
    // tagging integers `int64` converts rather than refusing.
    check(
        r##"{"type": "integer", "format": "int64"}"##,
        r##"
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        root ::= ("-"? integral-part)
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
    // `prefixItems` says nothing about a null. This is the branch the
    // `["array", "null"]` union above depends on.
    check(
        r##"{"type": "null", "prefixItems": {"type": "string"}}"##,
        r##"
        root ::= "null"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

// -- the contract with this repo's own parser -------------------------

#[test]
fn every_case_parses_as_gbnf() {
    // `convert` re-parses what it emits, so a schema that converts is a
    // grammar this engine accepts. Assert that the guarantee is real by
    // parsing again here, and that the rules a caller would start from
    // exist.
    for schema in [
        "{}",
        r##"{"type": "string"}"##,
        r##"{"type": ["string", "null"]}"##,
        r##"{"format": "date-time"}"##,
        r##"{"enum": ["a", "b"]}"##,
        r##"{"properties": {"a": {"type": "string"}}, "additionalProperties": {"type": "number"}}"##,
        r##"{"$ref": "#/$defs/N", "$defs": {"N": {"type": "object", "properties": {"n": {"$ref": "#/$defs/N"}}, "additionalProperties": false}}}"##,
    ] {
        let grammar = json_schema_to_grammar(schema).expect("schema converts");
        let parsed = crate::grammar::parse(&grammar).expect("emitted grammar parses");
        assert!(
            parsed.symbol_id("root").is_some(),
            "no root rule in:\n{grammar}"
        );
    }
}

#[test]
fn serde_json_value_entry_point_keeps_the_declaration_order() {
    // `serde_json` keeps document order (`preserve_order`), so a parsed
    // `Value` converts as its text does, an unsorted schema included:
    // `b` is required before `a`, as llama.cpp requires it.
    for text in [
        r##"{"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"], "additionalProperties": false}"##,
        r##"{"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "string"}}, "required": ["a", "b"], "additionalProperties": false}"##,
    ] {
        let value: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(
            json_schema_to_grammar(text).expect("text converts"),
            json_schema_to_grammar_value(&value).expect("value converts"),
            "{text}"
        );
    }
}

#[test]
fn a_hyphen_in_a_property_name_stays_out_of_a_range() {
    // `-` is the one character llama.cpp's GRAMMAR_RANGE_LITERAL_ESCAPES
    // table spells `\-` while its GBNF *parser* has no such escape. Written
    // bare it would open a range instead. `\x2D` is the spelling both
    // halves accept, and this grammar has to parse.
    check(
        r##"{"properties": {"a-b": {"type": "integer"}}, "additionalProperties": {"type": "integer"}}"##,
        r##"
        a-b-kv ::= "\"a-b\"" space ":" space integer
        a-b-rest ::= ( "," space additional-kv )*
        additional-k ::= ["] ( [a] ([\x2D] ([b] char+ | [^"b] char*) | [^"\x2D] char*) | [^"a] char* )? ["]
        additional-kv ::= additional-k ":" space integer
        char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        integer ::= ("-"? integral-part)
        integral-part ::= [0] | [1-9] [0-9]{0,15}
        root ::= "{" space  (a-b-kv a-b-rest | additional-kv ( "," space additional-kv )* )? space "}"
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}
