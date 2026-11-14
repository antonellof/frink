//! GBNF grammar-constrained decoding.
//!
//! A port of llama.cpp's `src/llama-grammar.cpp` and `src/llama-grammar.h`
//! (MIT; see `docs/THIRD_PARTY_NOTICES.md`). The algorithm is transcribed
//! from that source rather than reconstructed from how EBNF engines
//! usually work, because the two differ in places that matter -- the
//! negation of a character class lives on its first element only, the
//! repetition rewrites produce observable rule ids, and a token piece that
//! ends mid-codepoint has to stay viable rather than be rejected.
//!
//! What this replaces, and why it is not the same kind of thing:
//! `crates/frink-server/src/json_mode.rs` masks logits by a fixed
//! character class. It has no state, so it cannot know whether a `}`
//! closes an object that was opened. This is a pushdown machine over a
//! parsed grammar, so it can.
//!
//! # Layout
//!
//! | Module | Role |
//! |---|---|
//! | [`element`] | the compiled element / rule / cursor types |
//! | [`error`] | every refusal, naming what is missing |
//! | [`utf8`] | llama.cpp's two UTF-8 decoders, partial sequences included |
//! | [`parser`] | GBNF text to a rule table |
//! | [`machine`] | the pushdown stack machine over that table |
//! | [`candidates`] | which candidate tokens no viable stack accepts |
//! | [`lazy`] | trigger tokens / patterns: a grammar that switches on mid-generation |
//! | [`json_schema`] | JSON Schema to GBNF text, for `response_format` |
//!
//! # Status
//!
//! Steps 1 and 2 of the three-step spine in
//! `1ebda4fc:docs/plans/llama-cpp-gap-inventory.md` §8 item 8: the parser and the
//! stack machine, with the candidate-rejection core the sampler hook would
//! sit on. **Not wired into sampling or the server.** The seam a caller
//! needs is [`machine::Grammar::accept_token`] after each accepted token
//! and [`candidates::reject_candidates`] before each sample; nothing calls
//! either yet.
//!
//! [`json_schema`] now ports `common/json-schema-to-grammar.cpp`, which is
//! what `response_format: json_schema` needs on top of a grammar. It is
//! stricter than upstream by design -- a keyword it cannot honour is a
//! typed refusal rather than a grammar that accepts too much -- and its
//! module docs list what is ported, what is refused by name, and where its
//! output differs from llama.cpp's. `frink-server` now answers
//! `response_format: {"type": "json_schema"}` through it (see that crate's
//! `grammar_request`), so a schema and a hand-written grammar reach the
//! same machine.
//!
//! **Lazy grammars** ([`lazy`]) are ported: `trigger_tokens` and
//! `trigger_patterns`, the accumulated trigger buffer, and the replay that
//! feeds the grammar from the match onward. That is what `tool_choice`
//! needs on top of a grammar -- a model may talk before it calls a tool,
//! and a grammar applied from token zero forbids the talking.

pub mod candidates;
pub mod element;
pub mod error;
pub mod json_schema;
pub mod lazy;
pub mod machine;
pub mod parser;
pub mod utf8;

#[cfg(test)]
mod grammars;
#[cfg(test)]
mod machine_tests;
#[cfg(test)]
mod parser_tests;

pub use candidates::{reject_candidates, Candidate, DecodedPiece};
pub use element::{GrammarElement, GrammarRule, GrammarStack, GreType, RulePos};
pub use error::GrammarError;
pub use json_schema::{json_schema_to_grammar, json_schema_to_grammar_value, SchemaError};
pub use lazy::{LazyState, LazyTriggers, TriggerPattern, TriggerStep};
pub use machine::Grammar;
pub use parser::{parse, parse_with_vocab, GrammarVocab, ParsedGrammar};
pub use utf8::PartialUtf8;
