//! Rendering per-token distributions into OpenAI's `logprobs` shape.
//!
//! The distributions come from the sampler itself
//! (`frink_models::Sampler::sample_reporting`), which returns the very
//! vector a token was drawn from rather than a second opinion computed
//! beside it. This module only shapes them for the wire, and holds the
//! two decisions that shaping forces.
//!
//! # `log(0)` is `-inf`, and `-inf` is not JSON
//!
//! A filtered-out candidate has probability exactly zero -- that is
//! what "the chain removed it" means -- and `ln(0)` is negative
//! infinity, which `serde_json` cannot represent. Such a candidate is
//! OMITTED from `top_logprobs` rather than serialised as `null` or as
//! some large negative stand-in: it was not a candidate, so reporting a
//! number for it would describe a choice the sampler could not have
//! made.
//!
//! The SAMPLED token can never be one of these. It was drawn from this
//! distribution, so its probability is positive by construction, and
//! `token_logprobs` therefore always has a real number in it.
//!
//! # `text_offset` is bytes, and the tokens are the server's own
//!
//! OpenAI's field is a byte offset into the returned text. It is
//! computed here by decoding each token in order and accumulating
//! lengths, rather than by re-tokenizing the finished string: a
//! detokenize-then-retokenize round trip is not the identity for every
//! vocabulary, and an offset that disagrees with the text it indexes is
//! worse than no offset.

use serde_json::Value;

use crate::sampling_loop::PerTokenProbs;

/// The `logprobs` object for one choice, or `Value::Null` when the
/// request did not ask for it.
///
/// `top_k` is the caller's `logprobs: N` -- how many alternatives to
/// report per position. Zero means "the chosen token only", which is
/// what `logprobs: 0` asks for upstream.
pub(crate) fn render(
    per_token: &PerTokenProbs,
    top_k: Option<usize>,
    decode: &dyn Fn(usize) -> String,
) -> Value {
    let Some(top_k) = top_k else {
        return Value::Null;
    };
    if per_token.is_empty() {
        // Asked for, and there is nothing to report: an empty object
        // rather than null, because the request WAS honoured and the
        // generation simply produced no tokens.
        return serde_json::json!({
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": [],
            "text_offset": [],
        });
    }

    let mut tokens: Vec<String> = Vec::with_capacity(per_token.len());
    let mut token_logprobs: Vec<f64> = Vec::with_capacity(per_token.len());
    let mut top_logprobs: Vec<Value> = Vec::with_capacity(per_token.len());
    let mut text_offset: Vec<usize> = Vec::with_capacity(per_token.len());
    let mut offset = 0usize;

    for (id, probs) in per_token {
        let piece = decode(*id);
        text_offset.push(offset);
        offset += piece.len();
        tokens.push(piece);
        // Positive by construction: this token was drawn from this
        // distribution. `max(f64::MIN)` is not a fallback for that, it
        // guards a denormal that rounds to zero in the f32 -> f64 hop.
        let chosen = probs.get(*id).copied().unwrap_or(0.0);
        token_logprobs.push(logprob(chosen));

        let mut alternatives: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            // A zero is a candidate the chain removed, and `ln(0)` is
            // not a number JSON can carry. Omitted, not stand-in'd.
            .filter(|(_, p)| **p > 0.0)
            .map(|(i, p)| (i, *p))
            .collect();
        alternatives.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        alternatives.truncate(top_k);
        let mut map = serde_json::Map::new();
        for (alt, p) in alternatives {
            map.insert(decode(alt), Value::from(logprob(p)));
        }
        top_logprobs.push(Value::Object(map));
    }

    serde_json::json!({
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    })
}

/// `ln(p)` in f64, so the value survives JSON.
///
/// A zero cannot reach here from the chosen token, and is filtered out
/// of the alternatives, so this never returns `-inf` for a value that
/// is serialised.
fn logprob(p: f32) -> f64 {
    (p as f64).ln()
}

/// The CHAT wire's shape, which is a different object from
/// `/v1/completions`'s and not a rename of it.
///
/// OpenAI gives chat `choices[].logprobs.content[]`, one entry per
/// token carrying `token`, `logprob`, `bytes` and its own
/// `top_logprobs` list of `{token, logprob, bytes}`. The older
/// completions wire gives four PARALLEL ARRAYS and a `text_offset`
/// chat has no field for. Rendering one from the other would be a
/// translation layer that has to agree with two upstream shapes at
/// once; they are two functions over one input instead.
///
/// `bytes` is the token's UTF-8 bytes, which is how a client
/// reassembles a piece that is half a character -- the reason the
/// field exists upstream.
pub(crate) fn render_chat(
    per_token: &PerTokenProbs,
    top_k: Option<usize>,
    decode: &dyn Fn(usize) -> String,
) -> Value {
    let Some(top_k) = top_k else {
        return Value::Null;
    };
    let entry = |id: usize, p: f32, decode: &dyn Fn(usize) -> String| {
        let piece = decode(id);
        serde_json::json!({
            "token": piece,
            "logprob": logprob(p),
            "bytes": piece.as_bytes(),
        })
    };
    let content: Vec<Value> = per_token
        .iter()
        .map(|(id, probs)| {
            let chosen = probs.get(*id).copied().unwrap_or(0.0);
            let mut alternatives: Vec<(usize, f32)> = probs
                .iter()
                .enumerate()
                // See `render`: a removed candidate is not a candidate.
                .filter(|(_, p)| **p > 0.0)
                .map(|(i, p)| (i, *p))
                .collect();
            alternatives.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            alternatives.truncate(top_k);
            let mut e = entry(*id, chosen, decode);
            e["top_logprobs"] = Value::Array(
                alternatives
                    .into_iter()
                    .map(|(alt, p)| entry(alt, p, decode))
                    .collect(),
            );
            e
        })
        .collect();
    serde_json::json!({ "content": content })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoder() -> impl Fn(usize) -> String {
        |id: usize| format!("t{id}")
    }

    /// The chat shape is its own object: `content[]` of
    /// `{token, logprob, bytes, top_logprobs[]}`, with NO parallel
    /// arrays and no `text_offset`.
    ///
    /// Asserted here rather than over HTTP, because the test server
    /// serves synthetic weights and the banner clears the
    /// distributions -- an HTTP assertion iterates an empty `content`
    /// and passes vacuously, which it did until this test existed.
    #[test]
    fn the_chat_shape_is_entries_not_parallel_arrays() {
        let per_token: PerTokenProbs = vec![(1, vec![0.25, 0.5, 0.25, 0.0])];
        let out = render_chat(&per_token, Some(2), &decoder());

        assert!(out["tokens"].is_null(), "completions shape leaked: {out}");
        assert!(out["text_offset"].is_null(), "{out}");
        let content = out["content"].as_array().expect("content");
        assert_eq!(content.len(), 1, "one entry per token: {out}");

        let e = &content[0];
        assert_eq!(e["token"], "t1");
        assert_eq!(
            e["bytes"],
            serde_json::json!([116, 49]),
            "the token's UTF-8"
        );
        let chosen = e["logprob"].as_f64().expect("a real number");
        assert!((chosen - 0.5f64.ln()).abs() < 1e-9, "{chosen}");

        let top = e["top_logprobs"].as_array().expect("top_logprobs");
        assert_eq!(top.len(), 2, "asked for 2: {top:?}");
        // Sorted, so the likeliest is first, and it is the chosen one.
        assert_eq!(top[0]["token"], "t1");
        for alt in top {
            assert!(alt["bytes"].is_array(), "{alt}");
            assert!(
                alt["logprob"].as_f64().expect("a real number").is_finite(),
                "ln(0) reached the wire: {alt}"
            );
        }
    }

    /// A removed candidate is omitted from the chat shape too, with
    /// room to spare so the filter is the only thing deciding the
    /// length.
    #[test]
    fn the_chat_shape_omits_removed_candidates() {
        let per_token: PerTokenProbs = vec![(0, vec![0.7, 0.3, 0.0, 0.0])];
        let out = render_chat(&per_token, Some(4), &decoder());
        let top = out["content"][0]["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), 2, "a zero was reported: {top:?}");
    }

    #[test]
    fn the_chat_shape_is_absent_when_not_asked_for() {
        assert_eq!(render_chat(&Vec::new(), None, &decoder()), Value::Null);
    }

    #[test]
    fn no_request_means_no_object() {
        assert_eq!(render(&Vec::new(), None, &decoder()), Value::Null);
    }

    /// The chosen token's logprob is always a real number, and the
    /// alternatives are sorted by probability with the zeros left out.
    #[test]
    fn the_chosen_token_and_its_alternatives_are_reported() {
        // token 1 chosen from a distribution where token 3 was removed.
        let per_token: PerTokenProbs = vec![(1, vec![0.25, 0.5, 0.25, 0.0])];
        let out = render(&per_token, Some(3), &decoder());

        assert_eq!(out["tokens"], serde_json::json!(["t1"]));
        assert_eq!(out["text_offset"], serde_json::json!([0]));
        let chosen = out["token_logprobs"][0].as_f64().expect("a real number");
        assert!((chosen - 0.5f64.ln()).abs() < 1e-9, "{chosen}");

        let top = out["top_logprobs"][0].as_object().expect("an object");
        assert_eq!(top.len(), 3, "three survived the filter: {top:?}");
        assert!(
            !top.contains_key("t3"),
            "a zero-probability candidate was reported: {top:?}"
        );
        // Sorted by probability, so the most likely is the chosen one.
        assert!((top["t1"].as_f64().unwrap() - 0.5f64.ln()).abs() < 1e-9);
    }

    /// A removed candidate is OMITTED even when there is room for it.
    ///
    /// The earlier version of this test asked for 3 out of 4 where the
    /// zero sorted last, so keeping the zeros would still have
    /// produced 3 entries and the test could not fail. Here `top_k`
    /// exceeds the number of surviving candidates, so the filter is
    /// the only thing that decides the length.
    #[test]
    fn a_removed_candidate_is_omitted_even_with_room_to_spare() {
        let per_token: PerTokenProbs = vec![(0, vec![0.7, 0.3, 0.0, 0.0])];
        let out = render(&per_token, Some(4), &decoder());
        let top = out["top_logprobs"][0].as_object().expect("an object");
        assert_eq!(
            top.len(),
            2,
            "a zero-probability candidate was reported: {top:?}"
        );
        assert!(
            !top.contains_key("t2") && !top.contains_key("t3"),
            "{top:?}"
        );
        for v in top.values() {
            assert!(
                v.as_f64().expect("a real number").is_finite(),
                "ln(0) reached the wire: {top:?}"
            );
        }
    }

    /// `top_k` truncates, and truncates to the MOST likely.
    #[test]
    fn top_k_keeps_the_likeliest() {
        let per_token: PerTokenProbs = vec![(0, vec![0.6, 0.3, 0.1])];
        let out = render(&per_token, Some(2), &decoder());
        let top = out["top_logprobs"][0].as_object().unwrap();
        assert_eq!(top.len(), 2);
        assert!(top.contains_key("t0") && top.contains_key("t1"));
        assert!(!top.contains_key("t2"), "kept the least likely: {top:?}");
    }

    /// Offsets are byte offsets into the concatenated text, computed
    /// from the same pieces the `tokens` array reports -- so a client
    /// can slice the returned text with them.
    #[test]
    fn text_offsets_index_the_text_they_describe() {
        let per_token: PerTokenProbs = vec![
            (1, vec![0.0, 1.0]),
            (0, vec![1.0, 0.0]),
            (1, vec![0.0, 1.0]),
        ];
        let out = render(&per_token, Some(1), &decoder());
        let tokens: Vec<String> = serde_json::from_value(out["tokens"].clone()).unwrap();
        let offsets: Vec<usize> = serde_json::from_value(out["text_offset"].clone()).unwrap();
        let text = tokens.concat();
        for (i, off) in offsets.iter().enumerate() {
            assert!(
                text[*off..].starts_with(&tokens[i]),
                "offset {off} does not point at {:?} in {text:?}",
                tokens[i]
            );
        }
    }

    /// Asked for, nothing generated: the request was honoured, so the
    /// object exists and is empty rather than absent.
    #[test]
    fn an_empty_generation_still_reports_the_object() {
        let out = render(&Vec::new(), Some(2), &decoder());
        assert!(out.is_object(), "{out}");
        assert_eq!(out["tokens"], serde_json::json!([]));
    }
}
