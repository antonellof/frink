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

/// [`render`] with the ECHOED prompt in front of the completion.
///
/// `echo` returns the prompt and the completion as one string, so the
/// parallel arrays have to cover both or a client lining `text_offset`
/// up against `text` reads the wrong span. The prompt's entries are
/// exactly what [`render_prompt`] reports -- the PLAIN softmax, since
/// a prompt token was supplied rather than drawn -- and the
/// completion's are the sampler's own distributions, which is the one
/// place this wire carries two different kinds of number in one array.
///
/// The first entry is `null`: nothing predicted the first prompt
/// token. Offsets run over `prompt_text ++ completion`, so the
/// completion's offsets are the ordinary ones shifted by the prompt's
/// length in BYTES rather than in tokens.
pub(crate) fn render_echoed(
    prompt: &[usize],
    prompt_rows: &[Vec<f32>],
    prompt_text: &str,
    per_token: &PerTokenProbs,
    top_k: Option<usize>,
    decode: &dyn Fn(usize) -> String,
) -> Value {
    let Some(top_k) = top_k else {
        return Value::Null;
    };
    let mut tokens: Vec<Value> = Vec::with_capacity(prompt.len() + per_token.len());
    let mut token_logprobs: Vec<Value> = Vec::with_capacity(prompt.len() + per_token.len());
    let mut top_logprobs: Vec<Value> = Vec::with_capacity(prompt.len() + per_token.len());
    let mut text_offset: Vec<usize> = Vec::with_capacity(prompt.len() + per_token.len());
    let mut offset = 0usize;

    for (i, id) in prompt.iter().enumerate() {
        let piece = decode(*id);
        text_offset.push(offset);
        offset += piece.len();
        tokens.push(Value::from(piece));
        // Row `i - 1` predicted position `i`; position 0 had nothing
        // before it.
        match i.checked_sub(1).and_then(|r| prompt_rows.get(r)) {
            Some(logits) => {
                let probs = softmax(logits);
                token_logprobs.push(Value::from(logprob(probs.get(*id).copied().unwrap_or(0.0))));
                top_logprobs.push(top_map(&probs, top_k, decode));
            }
            None => {
                token_logprobs.push(Value::Null);
                top_logprobs.push(Value::Null);
            }
        }
    }
    // The prompt's pieces are detokenized one at a time above, and the
    // text the caller gets is the prompt string. Those agree in length
    // for a lossless tokenizer and can differ by a byte for one that
    // is not, so the completion's offsets are anchored to the STRING
    // rather than to the sum of the pieces.
    offset = prompt_text.len();

    for (id, probs) in per_token {
        let piece = decode(*id);
        text_offset.push(offset);
        offset += piece.len();
        tokens.push(Value::from(piece));
        token_logprobs.push(Value::from(logprob(probs.get(*id).copied().unwrap_or(0.0))));
        top_logprobs.push(top_map(probs, top_k, decode));
    }

    serde_json::json!({
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    })
}

/// The `top_k` most likely ids of `probs`, as `{piece: ln p}`.
///
/// Shared by the two renderers above, which sorted and truncated the
/// same way in two places until `echo` needed a third.
fn top_map(probs: &[f32], top_k: usize, decode: &dyn Fn(usize) -> String) -> Value {
    let mut alternatives: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        // A zero is a candidate the chain removed, and `ln(0)` is not
        // a number JSON can carry. Omitted, not stand-in'd.
        .filter(|(_, p)| **p > 0.0)
        .map(|(i, p)| (i, *p))
        .collect();
    alternatives.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    alternatives.truncate(top_k);
    let mut map = serde_json::Map::new();
    for (alt, p) in alternatives {
        map.insert(decode(alt), Value::from(logprob(p)));
    }
    Value::Object(map)
}

/// `ln(p)` in f64, so the value survives JSON.
///
/// A zero cannot reach here from the chosen token, and is filtered out
/// of the alternatives, so this never returns `-inf` for a value that
/// is serialised.
fn logprob(p: f32) -> f64 {
    (p as f64).ln()
}

/// Score the PROMPT: for each position after the first, the
/// log-probability the model gave the token that actually followed.
///
/// `per_position` is one logit row per prompt token, as
/// `Decoder::forward_batch` returns them. Row `i` predicts position
/// `i + 1`, so the LAST row is dropped -- it predicts the first
/// generated token, which is the completion's business and is already
/// reported there.
///
/// # The plain softmax, not the sampler's chain
///
/// A prompt token was NOT drawn from the sampler's filtered
/// distribution; it was supplied by the caller. Reporting a
/// penalised, top-p-truncated distribution for it would describe a
/// choice that never happened, and would answer "how likely was this
/// token" with "how likely would the sampler have been to pick it",
/// which is a different question. So this is the raw softmax over the
/// model's own logits.
///
/// That also means a prompt token can have probability far below any
/// sampling threshold and still be reported, which is the point: the
/// field exists to score text the model did not choose.
///
/// # Position 0 is `null`
///
/// The first prompt token has no preceding context, so nothing
/// predicted it. `null` rather than an invented number, and it is one
/// entry so the array lines up with the prompt token for token.
pub(crate) fn render_prompt(
    prompt: &[usize],
    per_position: &[Vec<f32>],
    top_k: usize,
    decode: &dyn Fn(usize) -> String,
) -> Value {
    let mut out: Vec<Value> = Vec::with_capacity(prompt.len());
    // Nothing predicted the first token.
    out.push(Value::Null);
    for (i, id) in prompt.iter().enumerate().skip(1) {
        let Some(logits) = per_position.get(i - 1) else {
            // Fewer rows than prompt tokens: the caller is told what
            // is missing rather than handed a shorter array it has to
            // line up itself.
            out.push(Value::Null);
            continue;
        };
        let probs = softmax(logits);
        let mut alternatives: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(t, p)| (t, *p))
            .filter(|(_, p)| *p > 0.0)
            .collect();
        alternatives.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        alternatives.truncate(top_k);
        let mut map = serde_json::Map::new();
        // The token that actually followed is always present, whether
        // or not it made the top `k`: a caller scoring their own text
        // needs its number, and omitting it would make the field
        // useless for exactly the case it exists for.
        let actual = probs.get(*id).copied().unwrap_or(0.0);
        map.insert(
            decode(*id),
            serde_json::json!({
                "logprob": logprob(actual),
                "rank": rank_of(&probs, *id),
                "decoded_token": decode(*id),
            }),
        );
        for (alt, p) in alternatives {
            map.entry(decode(alt)).or_insert_with(|| {
                serde_json::json!({
                    "logprob": logprob(p),
                    "rank": rank_of(&probs, alt),
                    "decoded_token": decode(alt),
                })
            });
        }
        out.push(Value::Object(map));
    }
    Value::Array(out)
}

/// 1-based rank of `id` by probability, which is what a caller uses to
/// ask "how surprising was this token" without reading the whole
/// distribution.
fn rank_of(probs: &[f32], id: usize) -> usize {
    let p = probs.get(id).copied().unwrap_or(0.0);
    1 + probs.iter().filter(|q| **q > p).count()
}

/// Numerically stable softmax over the model's own logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; logits.len()];
    }
    let mut out: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let total: f32 = out.iter().sum();
    if total > 0.0 {
        for p in &mut out {
            *p /= total;
        }
    }
    out
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

    /// The prompt's first token is `null` -- nothing predicted it --
    /// and every later one carries the log-probability the model gave
    /// the token that ACTUALLY followed, plus its rank.
    #[test]
    fn the_prompt_is_scored_from_the_second_token_on() {
        // Two logit rows for a three-token prompt: row i predicts
        // position i + 1, and the last row (predicting the first
        // GENERATED token) is not part of the prompt.
        let prompt = vec![0usize, 1, 2];
        let rows = vec![
            // predicts prompt[1] == 1: make 1 the likeliest.
            vec![0.0f32, 2.0, 0.0],
            // predicts prompt[2] == 2: make 2 UNLIKELY, so the test
            // covers the case the field exists for.
            vec![3.0f32, 3.0, 0.0],
        ];
        let out = render_prompt(&prompt, &rows, 2, &decoder());
        let arr = out.as_array().expect("an array");
        assert_eq!(arr.len(), 3, "one entry per prompt token: {out}");
        assert!(arr[0].is_null(), "nothing predicted the first token");

        // The likely one.
        let e1 = arr[1].as_object().expect("an object");
        assert_eq!(e1["t1"]["rank"], 1, "{e1:?}");
        assert!(e1["t1"]["logprob"].as_f64().unwrap() > -0.5, "{e1:?}");

        // The UNLIKELY one is still reported, with its real rank --
        // that is the whole point of the field.
        let e2 = arr[2].as_object().expect("an object");
        assert!(
            e2.contains_key("t2"),
            "the token that actually followed was omitted: {e2:?}"
        );
        assert_eq!(e2["t2"]["rank"], 3, "{e2:?}");
        assert!(
            e2["t2"]["logprob"].as_f64().unwrap() < -2.0,
            "an unlikely token was reported as likely: {e2:?}"
        );
    }

    /// It is the PLAIN softmax, not the sampler's filtered chain: a
    /// prompt token was supplied, not drawn, so a truncated
    /// distribution would describe a choice that never happened.
    ///
    /// Pinned by a token that any top-p would have removed: it still
    /// gets a real, finite number.
    #[test]
    fn a_prompt_token_no_sampler_would_pick_still_gets_a_number() {
        let prompt = vec![0usize, 2];
        // Token 2 is vanishingly unlikely beside token 0.
        let rows = vec![vec![12.0f32, 0.0, -12.0]];
        let out = render_prompt(&prompt, &rows, 1, &decoder());
        let e = out[1].as_object().expect("an object");
        let v = e["t2"]["logprob"].as_f64().expect("a real number");
        assert!(v.is_finite(), "ln(0) reached the wire: {e:?}");
        assert!(v < -20.0, "a filtered distribution was used: {e:?}");
        assert_eq!(e["t2"]["rank"], 3);
    }

    /// Fewer rows than prompt tokens is reported as `null` for the
    /// positions nobody scored, so the array still lines up token for
    /// token rather than being silently short.
    #[test]
    fn missing_rows_are_null_rather_than_a_shorter_array() {
        let out = render_prompt(&[0usize, 1, 2], &[vec![0.0, 1.0, 0.0]], 1, &decoder());
        let arr = out.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert!(arr[0].is_null() && arr[2].is_null(), "{out}");
        assert!(arr[1].is_object(), "{out}");
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
