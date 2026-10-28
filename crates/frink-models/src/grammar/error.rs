//! Errors this grammar engine returns.
//!
//! llama.cpp's parser throws `std::runtime_error`, catches it in
//! `llama_grammar_parser::parse`, prints to stderr, clears the rule table
//! and returns `false` -- so a caller learns only "it failed". Two of its
//! stack-machine invariants are `GGML_ABORT`, which kills the process.
//!
//! Per ADR 0002 ("return `Result` and name what is missing"), every one
//! of those becomes a typed variant carrying the byte offset and the
//! remaining input, so a server can hand the message back to the client
//! that sent the grammar.

use std::fmt;

/// What the grammar engine refused, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarError {
    /// A syntax error in the GBNF source. `offset` is the byte offset into
    /// the grammar text where llama.cpp's parser would have thrown, and
    /// `rest` is the (truncated) input from there on, matching the
    /// `"expecting X at " + src` shape of the upstream messages.
    Syntax {
        expected: String,
        offset: usize,
        rest: String,
    },
    /// A rule was referenced but never given a `::=` definition.
    UndefinedRule { name: String, rule_id: u32 },
    /// The grammar parsed but has no rule with the requested root name.
    MissingRoot { name: String },
    /// A rule can reach itself without consuming input. llama.cpp logs
    /// "unsupported grammar, left recursion detected" and returns null.
    LeftRecursion { rule_id: u32, name: Option<String> },
    /// A repetition operator would expand to more rules than the engine
    /// is willing to build. llama.cpp's `MAX_REPETITION_THRESHOLD`.
    RepetitionTooLarge {
        requested: u64,
        limit: u64,
        offset: usize,
    },
    /// `<name>` token syntax was used but no vocabulary was supplied to
    /// resolve it. `<[42]>` needs no vocabulary and always works.
    TokenNeedsVocabulary { token: String, offset: usize },
    /// A `<name>` token did not tokenize to exactly one token.
    TokenNotSingle { token: String, n_tokens: usize },
    /// The generated text left no viable parse: every stack died. Carries
    /// the piece that killed it, as llama.cpp's thrown message does.
    NoViableStack { piece: String },
    /// A lazy grammar's trigger pattern does not compile. llama.cpp builds
    /// a `std::regex` in the constructor, which throws.
    TriggerPatternInvalid { pattern: String, reason: String },
    /// A trigger pattern compiled but failed while matching -- the
    /// backtracking limit, in practice. Upstream's `std::regex` has the
    /// same failure mode and does not report it.
    TriggerPatternFailed { pattern: String, reason: String },
    /// A grammar was made lazy with nothing that could ever switch it on,
    /// which is an unconstrained generation wearing a grammar. llama.cpp
    /// permits it; this engine will not pretend to constrain.
    LazyWithoutTriggers,
    /// A not-yet-triggered lazy grammar was asked which tokens it forbids.
    /// It forbids nothing, and answering "nothing" would be
    /// indistinguishable from a grammar that allows everything -- so the
    /// question is refused. Callers test
    /// [`Grammar::is_awaiting_trigger`](super::Grammar::is_awaiting_trigger)
    /// first, as `llama_grammar_apply_impl` does.
    AwaitingTrigger,
    /// An invariant llama.cpp asserts with `GGML_ABORT`. Reaching one is a
    /// bug in this engine, not in the caller's grammar.
    Internal(&'static str),
}

/// How much of the remaining input a syntax error quotes.
const REST_CLIP: usize = 40;

impl GrammarError {
    /// Build a [`GrammarError::Syntax`] from a position in the source.
    pub(crate) fn syntax(expected: impl Into<String>, src: &[u8], offset: usize) -> Self {
        let start = offset.min(src.len());
        let end = (start + REST_CLIP).min(src.len());
        // The offset can land mid-codepoint on malformed input; take the
        // longest valid prefix rather than panicking inside an error path.
        let rest = match std::str::from_utf8(&src[start..end]) {
            Ok(s) => s.to_string(),
            Err(e) => String::from_utf8_lossy(&src[start..start + e.valid_up_to()]).into_owned(),
        };
        GrammarError::Syntax {
            expected: expected.into(),
            offset: start,
            rest,
        }
    }
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrammarError::Syntax {
                expected,
                offset,
                rest,
            } => {
                write!(
                    f,
                    "grammar syntax error at byte {offset}: {expected}, at {rest:?}"
                )
            }
            GrammarError::UndefinedRule { name, rule_id } => write!(
                f,
                "grammar references rule {name:?} (id {rule_id}) which is never defined with ::="
            ),
            GrammarError::MissingRoot { name } => {
                write!(f, "grammar does not contain a {name:?} rule to start from")
            }
            GrammarError::LeftRecursion { rule_id, name } => match name {
                Some(n) => write!(
                    f,
                    "unsupported grammar: rule {n:?} (id {rule_id}) is left-recursive"
                ),
                None => write!(
                    f,
                    "unsupported grammar: rule id {rule_id} is left-recursive"
                ),
            },
            GrammarError::RepetitionTooLarge {
                requested,
                limit,
                offset,
            } => write!(
                f,
                "grammar repetition at byte {offset} would expand to {requested} rules, over the \
                 limit of {limit}; reduce the repetition count or the rule complexity"
            ),
            GrammarError::TokenNeedsVocabulary { token, offset } => write!(
                f,
                "grammar token {token:?} at byte {offset} names a token but no vocabulary was \
                 supplied; use the <[id]> form or pass a vocabulary"
            ),
            GrammarError::TokenNotSingle { token, n_tokens } => write!(
                f,
                "grammar token {token:?} tokenizes to {n_tokens} tokens, but must be exactly 1"
            ),
            GrammarError::NoViableStack { piece } => write!(
                f,
                "no grammar parse survives the piece {piece:?}; it should have been masked out \
                 before it was sampled"
            ),
            GrammarError::TriggerPatternInvalid { pattern, reason } => write!(
                f,
                "lazy grammar trigger pattern {pattern:?} does not compile: {reason}"
            ),
            GrammarError::TriggerPatternFailed { pattern, reason } => write!(
                f,
                "lazy grammar trigger pattern {pattern:?} failed while matching the output so \
                 far: {reason}"
            ),
            GrammarError::LazyWithoutTriggers => write!(
                f,
                "a lazy grammar needs at least one trigger token or trigger pattern; with none \
                 it can never switch on, and nothing would be constrained"
            ),
            GrammarError::AwaitingTrigger => write!(
                f,
                "this lazy grammar has not been triggered yet and constrains nothing; check \
                 is_awaiting_trigger before asking which tokens it rejects"
            ),
            GrammarError::Internal(what) => {
                write!(f, "internal grammar engine invariant violated: {what}")
            }
        }
    }
}

impl std::error::Error for GrammarError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax_error_clips_and_quotes_the_rest_of_the_input() {
        let src =
            b"root ::= \"a\" @@@ trailing garbage that runs on and on and on and on past the clip";
        let e = GrammarError::syntax("expecting newline or end", src, 13);
        match &e {
            GrammarError::Syntax {
                expected,
                offset,
                rest,
            } => {
                assert_eq!(expected, "expecting newline or end");
                assert_eq!(*offset, 13);
                assert!(rest.starts_with("@@@ trailing"));
                assert_eq!(rest.len(), REST_CLIP);
            }
            other => panic!("wrong variant: {other:?}"),
        }
        assert!(e.to_string().contains("byte 13"));
    }

    #[test]
    fn syntax_error_offset_past_the_end_is_clamped() {
        let src = b"root";
        let e = GrammarError::syntax("expecting ::=", src, 999);
        match e {
            GrammarError::Syntax { offset, rest, .. } => {
                assert_eq!(offset, 4);
                assert_eq!(rest, "");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn syntax_error_does_not_panic_on_a_split_codepoint() {
        // "aé" -- offset 2 lands between the two bytes of 'é'.
        let src = "aé".as_bytes();
        let e = GrammarError::syntax("expecting name", src, 2);
        match e {
            GrammarError::Syntax { rest, .. } => assert_eq!(rest, ""),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn every_variant_says_what_is_missing() {
        let cases: Vec<GrammarError> = vec![
            GrammarError::UndefinedRule {
                name: "ws".into(),
                rule_id: 4,
            },
            GrammarError::MissingRoot {
                name: "root".into(),
            },
            GrammarError::LeftRecursion {
                rule_id: 1,
                name: Some("expr".into()),
            },
            GrammarError::LeftRecursion {
                rule_id: 1,
                name: None,
            },
            GrammarError::RepetitionTooLarge {
                requested: 10_000,
                limit: 2000,
                offset: 7,
            },
            GrammarError::TokenNeedsVocabulary {
                token: "<think>".into(),
                offset: 9,
            },
            GrammarError::TokenNotSingle {
                token: "<think>".into(),
                n_tokens: 3,
            },
            GrammarError::NoViableStack { piece: "}".into() },
            GrammarError::TriggerPatternInvalid {
                pattern: "(unclosed".into(),
                reason: "unclosed group".into(),
            },
            GrammarError::TriggerPatternFailed {
                pattern: "(a+)+b".into(),
                reason: "backtrack limit exceeded".into(),
            },
            GrammarError::LazyWithoutTriggers,
            GrammarError::AwaitingTrigger,
            GrammarError::Internal("stack rested on CHAR_ALT"),
        ];
        for c in cases {
            let msg = c.to_string();
            assert!(msg.len() > 20, "message too thin for {c:?}: {msg}");
        }
    }
}
