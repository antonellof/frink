//! Request fields that CHANGE WHAT COMES BACK and that this server does
//! not implement, refused BY NAME on every route that could carry them.
//!
//! Serde drops an undeclared field silently, and a caller cannot tell
//! that apart from having had it honoured: they get a 200 and an answer
//! computed under different rules than they asked for. That is the same
//! argument [`crate::unsupported_sampling`] makes for `logit_bias` and
//! `samplers`; this module is the rest of the surface, measured rather
//! than guessed.
//!
//! # What the measurement found
//!
//! Every field below was sent to a running server on 2026-09-22 and
//! answered **200** while appearing NOWHERE in `crates/frink-server`
//! or `crates/frink-api` (`grep -c` over both trees: zero). Two of them
//! were worse than absent -- they were refused on ONE route and dropped
//! on another:
//!
//! ```text
//! POST /v1/chat/completions {"n": 3}        -> 501  "n > 1 is not implemented"
//! POST /v1/completions      {"n": 3}        -> 200  one choice, no mention
//! ```
//!
//! which is this repo's dominant bug shape (two structures that must
//! agree about one thing, with nothing enforcing it) and is the exact
//! defect `sampling_knobs::ExtraSamplerFields` was built to close for
//! the knobs that ARE implemented. `n` was hand-written into the chat
//! route's validator and never reached the other two.
//!
//! # The line this table draws
//!
//! Only fields that change the TOKENS or the TEXT returned. A field
//! that a server may ignore without changing its answer -- a scheduling
//! hint, a tracing tag -- is not refused, because refusing it would
//! break a caller for whom ignoring it was correct.
//!
//! So `priority` is deliberately absent, and `cache_salt` is absent
//! with a reason worth writing down: it changes which prefix-cache
//! entries a request may reuse, so a server that IGNORES it can serve
//! one caller from another's cached prefix. frink's radix cache is
//! keyed by token ids and shared across requests
//! (`policy::radix`), so the field is not merely unimplemented here,
//! it names an isolation property this server does not yet offer. That
//! is a row of its own rather than a line in this table.
//!
//! # Why a flattened struct and not a `Value` walk
//!
//! The three generation routes take genuinely different bodies and
//! cannot share a request struct. They can share this one, flattened
//! into each, exactly as `ExtraSamplerFields` is -- so a field added
//! here reaches all three at once and none of them can be given a
//! refusal the others lack. [`UnimplementedFields::refuse`]
//! destructures exhaustively with no `..`, so a field added to the wire
//! struct and not answered is an unused variable and
//! `cargo clippy -- -D warnings` is a gate.

use serde_json::Value;

use crate::{unsupported_feature, ApiError};

/// The routes whose response shape can carry more than one answer.
///
/// Not "every OpenAI route": llama.cpp's native `/completion` returns a
/// single `content` string, and neither the Anthropic nor the Responses
/// wire has a `choices` array, so `n` has nowhere to go on them and is
/// refused by name rather than silently collapsed to one.
const SERVES_SEVERAL_CHOICES: &[&str] = &[frink_api::routes::V1_COMPLETIONS];

/// Fields deserialized purely in order to be refused.
///
/// Every member is `Option`, and `None` is "the caller said nothing",
/// which is the only reading that leaves an existing client working.
/// Where a field has a default that matches what frink already does
/// (`n: 1`, `echo: false`, `skip_special_tokens: true`), that value is
/// accepted and only the other values are refused -- a caller who spells
/// out the default asked for the behaviour they are getting.
#[derive(Debug, Default, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub(crate) struct UnimplementedFields {
    /// OpenAI's `n`: how many completions to return. Refused above 1.
    pub(crate) n: Option<u32>,
    /// OpenAI's `best_of`: generate `k` and return the best-scoring
    /// one. Refused above 1. Was dropped on ALL THREE routes.
    pub(crate) best_of: Option<u32>,
    /// Per-token logprobs for the PROMPT, not the completion. frink
    /// scores prompt tokens in `frink perplexity` but exposes nothing
    /// for them over HTTP.
    pub(crate) prompt_logprobs: Option<Value>,
    /// Prepend the prompt to the returned text. Refused when true.
    pub(crate) echo: Option<bool>,
    /// Beam search instead of the sampler chain.
    pub(crate) use_beam_search: Option<bool>,
    /// Silently drop the prompt to the last `k` tokens. The most
    /// dangerous member of this table: ignoring it answers a DIFFERENT
    /// prompt than the caller believes they sent, with no error.
    pub(crate) truncate_prompt_tokens: Option<Value>,
    /// Pre-computed embeddings in place of text. A different input
    /// path entirely, not a knob on this one.
    pub(crate) prompt_embeds: Option<Value>,
    /// Restrict sampling to these ids.
    pub(crate) allowed_token_ids: Option<Value>,
    /// Forbid these strings. `stop` is implemented and is not this:
    /// `stop` ENDS the generation, this one steers around a token.
    pub(crate) bad_words: Option<Value>,
    /// Include special tokens in the returned text. frink always skips
    /// them, so `true` is accepted and `false` is refused.
    pub(crate) skip_special_tokens: Option<bool>,
    /// Return `"token_id:123"` strings in place of text pieces.
    pub(crate) return_tokens_as_token_ids: Option<bool>,
}

impl UnimplementedFields {
    /// Refuse anything the caller asked for that this server does not
    /// do, naming the field and what to reach for instead.
    ///
    /// **Exhaustive destructure, no `..`.** A member added above and
    /// not answered here fails the build.
    pub(crate) fn refuse(&self, route: &str) -> Result<(), ApiError> {
        let UnimplementedFields {
            n,
            best_of,
            prompt_logprobs,
            echo,
            use_beam_search,
            truncate_prompt_tokens,
            prompt_embeds,
            allowed_token_ids,
            bad_words,
            skip_special_tokens,
            return_tokens_as_token_ids,
        } = self;

        // `n` > 1 is SERVED on the routes whose response has a
        // `choices[]` array to put the extra answers in, and refused on
        // the ones that do not: llama.cpp's native `/completion`
        // returns one `content`, and the Anthropic and Responses wires
        // have no such field at all. `n: 1` is every route.
        if n.is_some_and(|v| v > 1) && !SERVES_SEVERAL_CHOICES.contains(&route) {
            return Err(refusal(
                route,
                "n",
                "more than one completion per request on this wire, which has no `choices` array \
                 to return them in; use /v1/completions or send the request again",
            ));
        }
        if best_of.is_some_and(|v| v > 1) {
            return Err(refusal(
                route,
                "best_of",
                "generating several completions and returning the best-scoring one",
            ));
        }
        if prompt_logprobs.is_some() {
            return Err(refusal(
                route,
                "prompt_logprobs",
                "logprobs for the PROMPT's own tokens",
            ));
        }
        if echo == &Some(true) {
            return Err(refusal(
                route,
                "echo",
                "prepending the prompt to the completion",
            ));
        }
        if use_beam_search == &Some(true) {
            return Err(refusal(
                route,
                "use_beam_search",
                "beam search; this server samples",
            ));
        }
        if truncate_prompt_tokens.is_some() {
            return Err(refusal(
                route,
                "truncate_prompt_tokens",
                "truncating the prompt server-side -- send the prompt you want answered",
            ));
        }
        if prompt_embeds.is_some() {
            return Err(refusal(
                route,
                "prompt_embeds",
                "embeddings as input in place of text",
            ));
        }
        if allowed_token_ids.is_some() {
            return Err(refusal(
                route,
                "allowed_token_ids",
                "restricting sampling to a token-id set; `response_format` constrains output here",
            ));
        }
        if bad_words.is_some() {
            return Err(refusal(
                route,
                "bad_words",
                "forbidding strings during sampling; `stop` ends a generation but does not steer it",
            ));
        }
        if skip_special_tokens == &Some(false) {
            return Err(refusal(
                route,
                "skip_special_tokens",
                "returning special tokens in the text; this server always skips them",
            ));
        }
        if return_tokens_as_token_ids == &Some(true) {
            return Err(refusal(
                route,
                "return_tokens_as_token_ids",
                "returning token ids in place of text pieces; `/v1/tokenize` returns ids",
            ));
        }
        Ok(())
    }
}

/// One sentence, one shape, naming the field and the route.
///
/// A 501 rather than a 400: the request is well-formed and a server
/// that implemented the field would serve it, which is the distinction
/// `unsupported_sampling` already draws between a typo and a gap.
fn refusal(route: &str, field: &str, what: &str) -> ApiError {
    unsupported_feature(&format!(
        "`{field}` is not implemented on {route}: {what} (see docs/API.md)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: serde_json::Value) -> UnimplementedFields {
        serde_json::from_value(body).expect("the struct is all-optional")
    }

    /// The defaults a caller may legitimately spell out are SERVED, not
    /// refused: they describe what this server already does.
    #[test]
    fn spelling_out_the_defaults_is_not_a_refusal() {
        for body in [
            serde_json::json!({ "n": 1 }),
            serde_json::json!({ "best_of": 1 }),
            serde_json::json!({ "echo": false }),
            serde_json::json!({ "use_beam_search": false }),
            serde_json::json!({ "skip_special_tokens": true }),
            serde_json::json!({ "return_tokens_as_token_ids": false }),
            serde_json::json!({}),
        ] {
            assert!(
                parse(body.clone())
                    .refuse(frink_api::routes::COMPLETION)
                    .is_ok(),
                "{body} should be served"
            );
        }
    }

    /// Every member refuses, and the message names the field. Driven
    /// from a list so a member added to the struct and forgotten here
    /// is visible as a count.
    #[test]
    fn every_field_refuses_by_name() {
        let cases: [(&str, serde_json::Value); 11] = [
            ("n", serde_json::json!({ "n": 2 })),
            ("best_of", serde_json::json!({ "best_of": 2 })),
            (
                "prompt_logprobs",
                serde_json::json!({ "prompt_logprobs": 1 }),
            ),
            ("echo", serde_json::json!({ "echo": true })),
            (
                "use_beam_search",
                serde_json::json!({ "use_beam_search": true }),
            ),
            (
                "truncate_prompt_tokens",
                serde_json::json!({ "truncate_prompt_tokens": 8 }),
            ),
            (
                "prompt_embeds",
                serde_json::json!({ "prompt_embeds": "AA==" }),
            ),
            (
                "allowed_token_ids",
                serde_json::json!({ "allowed_token_ids": [1, 2] }),
            ),
            ("bad_words", serde_json::json!({ "bad_words": ["x"] })),
            (
                "skip_special_tokens",
                serde_json::json!({ "skip_special_tokens": false }),
            ),
            (
                "return_tokens_as_token_ids",
                serde_json::json!({ "return_tokens_as_token_ids": true }),
            ),
        ];
        // The count is the struct's field count: a member added without
        // a case here changes it.
        assert_eq!(
            cases.len(),
            serde_json::to_value(UnimplementedFields::default())
                .expect("serializes")
                .as_object()
                .expect("an object")
                .len(),
            "every field of the struct needs a case"
        );
        for (field, body) in cases {
            // The native wire, which serves none of them: `n` is
            // SERVED on `/v1/completions` (see
            // `a_route_with_a_choices_array_serves_n`), and picking a
            // route that refuses everything keeps this test about the
            // table rather than about the exception.
            let err = parse(body)
                .refuse(frink_api::routes::COMPLETION)
                .expect_err("{field} must refuse");
            let msg = format!("{err:?}");
            assert!(msg.contains(field), "{field} not named in {msg}");
        }
    }

    /// `n` is the one field with a per-route answer, and both halves
    /// are pinned: served where the response has a `choices` array to
    /// put the answers in, refused by name where it does not.
    #[test]
    fn a_route_with_a_choices_array_serves_n() {
        let four = parse(serde_json::json!({ "n": 4 }));
        assert!(
            four.refuse(frink_api::routes::V1_COMPLETIONS).is_ok(),
            "the route that renders several choices must serve `n`"
        );
        for route in [
            frink_api::routes::COMPLETION,
            frink_api::routes::V1_CHAT_COMPLETIONS,
        ] {
            let err = four.refuse(route).expect_err("no choices array");
            assert!(format!("{err:?}").contains('n'), "{route}");
        }
        // And `n: 1` is every route, including the ones that refuse
        // more: a caller spelling out the default asked for what they
        // are getting.
        let one = parse(serde_json::json!({ "n": 1 }));
        for route in [
            frink_api::routes::COMPLETION,
            frink_api::routes::V1_CHAT_COMPLETIONS,
            frink_api::routes::V1_COMPLETIONS,
        ] {
            assert!(one.refuse(route).is_ok(), "{route} refused n = 1");
        }
    }

    /// The route is in the message, because the same field may be
    /// implemented on one wire and not another later and a caller
    /// reading the error should not have to guess which one refused.
    #[test]
    fn the_message_names_the_route() {
        let err = parse(serde_json::json!({ "n": 4 }))
            .refuse("/v1/chat/completions")
            .expect_err("refuses");
        assert!(format!("{err:?}").contains("/v1/chat/completions"));
    }
}
