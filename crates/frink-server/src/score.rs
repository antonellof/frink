//! `/v1/score` (and `/score`): how related are these two texts?
//!
//! One pair in, one number out, per pair. It is the endpoint
//! `crate::rerank` is built on top of, exposed on its own, and it
//! differs from that route in exactly one way that matters: it admits
//! BOTH kinds of checkpoint, and says which one answered.
//!
//! # Two regimes, named in the response
//!
//! A **cross-encoder** reads the pair together, `[CLS] a [SEP] b
//! [SEP]`, and reports a relevance logit from a trained classification
//! head. That is what `/v1/rerank` requires, and when the loaded
//! checkpoint has one it is what this route reports too.
//!
//! A **bi-encoder** embeds the two texts independently; the only
//! comparison available afterwards is the angle between the vectors.
//! `crate::rerank` REFUSES to substitute that, and it is right to: a
//! rerank promises the model's own ranking. A score does not. A caller
//! who loaded an embedding model and asked how related two texts are
//! is asking exactly what a bi-encoder computes, so the cosine is the
//! answer rather than a stand-in for one.
//!
//! The two are not on the same scale and never will be, so the
//! response carries `frink_score_kind` -- `"cross_encoder"` or
//! `"cosine_similarity"` -- for the same reason `/v1/rerank` carries
//! `frink_score_head`: a client thresholding a number has to know
//! which number it got. A cosine is in `[-1, 1]`; a head's logit is
//! whatever the head was trained to emit.
//!
//! # Pairing
//!
//! Upstream's rule, which is two rules: a single `text_1` against
//! every `text_2` (the common case -- one query, many candidates), or
//! two lists of the SAME length paired element by element. Two lists
//! of different lengths are a 400 rather than a zip that silently
//! drops the tail, which is the one mistake in this shape that returns
//! a plausible shorter answer.

use std::sync::Arc;

use axum::{extract::State, Json};
use frink_models::EmbeddingModel;

use crate::openai_extra::Call;
use crate::{join_error_response, ApiError, AppState};

/// `{"text_1": str|[str], "text_2": str|[str]}`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ScoreRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub text_1: TextInput,
    pub text_2: TextInput,
}

/// One text or several, which is how upstream spells both sides.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum TextInput {
    One(String),
    Many(Vec<String>),
}

impl TextInput {
    fn as_slice(&self) -> &[String] {
        match self {
            TextInput::One(_) => std::slice::from_ref(self.single()),
            TextInput::Many(v) => v,
        }
    }

    fn single(&self) -> &String {
        match self {
            TextInput::One(s) => s,
            TextInput::Many(v) => &v[0],
        }
    }
}

/// The pairs a request names, or the refusal saying why it names none.
///
/// Pure, so the rule is testable without a checkpoint -- which matters
/// because the mistake it guards against (zipping two lists of
/// different lengths) returns a shorter answer rather than an error.
pub(crate) fn pairs(a: &[String], b: &[String]) -> Result<Vec<(usize, usize)>, ApiError> {
    if a.is_empty() || b.is_empty() {
        return Err(crate::invalid_request(
            "`text_1` and `text_2` must each name at least one text",
            "text_1",
        ));
    }
    if a.len() == 1 {
        return Ok((0..b.len()).map(|j| (0, j)).collect());
    }
    if b.len() == 1 {
        return Ok((0..a.len()).map(|i| (i, 0)).collect());
    }
    if a.len() != b.len() {
        return Err(crate::invalid_request(
            &format!(
                "`text_1` has {} entries and `text_2` has {}; give one text on either side, \
                 or the same number on both",
                a.len(),
                b.len()
            ),
            "text_2",
        ));
    }
    Ok((0..a.len()).map(|i| (i, i)).collect())
}

/// Cosine of two vectors, or `None` when the angle is undefined.
///
/// `None` rather than zero: a zero vector has no direction, and
/// reporting `0.0` would say "unrelated" where the honest answer is
/// "undefined". The caller turns it into a refusal naming the pair.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    // No separate zero check: a zero vector makes `dot` zero too,
    // so the division is `0.0 / 0.0` and `is_finite` already answers
    // false. A guard was written here first and a sabotage that
    // deleted it left every test green, which is what a redundant
    // check looks like -- two places that must agree about one
    // condition, with only one of them load-bearing.
    let c = dot / (na * nb);
    c.is_finite().then_some(c)
}

pub async fn score(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ScoreRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let call = Call::new(&headers);
    let result = score_inner(&state, req).await;
    // The encoder passes are real prompt tokens and there is no decode
    // loop, so the completion half is 0 rather than borrowing from the
    // total. Same rule as `/v1/embeddings` and `/v1/rerank`.
    let usage = result
        .as_ref()
        .ok()
        .map(|(_, prompt_tokens)| frink_api::Usage::new(*prompt_tokens, 0));
    call.record(
        &state,
        frink_api::routes::V1_SCORE,
        state.embedding_model_name(),
        &result,
        usage.as_ref(),
    );
    result.map(|(body, _)| Json(body))
}

/// The encoder that can answer this route.
///
/// Unlike `/v1/rerank` a head is NOT required: a bi-encoder answers
/// with a cosine. What is still refused is a generative model, which
/// has neither.
fn require_encoder(state: &AppState) -> Result<Arc<EmbeddingModel>, ApiError> {
    state.embedding_model().ok_or_else(|| {
        state.require_active().err().unwrap_or_else(|| {
            crate::unsupported_feature(&format!(
                "the loaded model '{}' is not an encoder. {} needs an embedding or \
                 cross-encoder checkpoint, loaded as FRINK_MODEL_PATH or beside this one as \
                 FRINK_EMBEDDING_MODEL_PATH",
                state.active_model_name().unwrap_or_else(|| "?".to_string()),
                frink_api::routes::V1_SCORE,
            ))
        })
    })
}

async fn score_inner(
    state: &AppState,
    req: ScoreRequest,
) -> Result<(serde_json::Value, usize), ApiError> {
    let left: Vec<String> = req.text_1.as_slice().to_vec();
    let right: Vec<String> = req.text_2.as_slice().to_vec();
    let wanted = pairs(&left, &right)?;
    let encoder = require_encoder(state)?;
    let model_name = req
        .model
        .clone()
        .unwrap_or_else(|| encoder.name().to_string());
    let cross = encoder.rank_head().is_some();

    let (scores, prompt_tokens) = tokio::task::spawn_blocking({
        let encoder = Arc::clone(&encoder);
        let wanted = wanted.clone();
        move || -> Result<(Vec<f32>, usize), ApiError> {
            let mut scores = Vec::with_capacity(wanted.len());
            let mut prompt_tokens = 0usize;
            // Embeddings are memoised per TEXT, not per pair: the
            // common shape is one text against many, and embedding it
            // once per pair would run the encoder `n` times for one
            // vector. A cross-encoder cannot memoise anything, because
            // its input IS the pair.
            let mut memo: std::collections::HashMap<usize, Vec<f32>> =
                std::collections::HashMap::new();
            for (n, &(i, j)) in wanted.iter().enumerate() {
                if cross {
                    let pair = encoder
                        .rerank_input(&left[i], &right[j])
                        .map_err(|e| pair_error(n, e))?;
                    prompt_tokens += pair.tokens.len();
                    scores.push(encoder.rerank_score(&pair).map_err(|e| pair_error(n, e))?);
                    continue;
                }
                for (key, text) in [(i, &left[i]), (usize::MAX - j, &right[j])] {
                    if let std::collections::hash_map::Entry::Vacant(slot) = memo.entry(key) {
                        prompt_tokens += encoder.token_ids(text).len();
                        slot.insert(
                            encoder
                                .embed(text, /* normalize = */ false)
                                .map_err(|e| pair_error(n, e))?,
                        );
                    }
                }
                let c = cosine(&memo[&i], &memo[&(usize::MAX - j)]).ok_or_else(|| {
                    crate::invalid_request(
                        &format!(
                            "pair {n} embeds to a zero-length vector, so the angle between \
                             the two texts is undefined"
                        ),
                        "text_1",
                    )
                })?;
                scores.push(c);
            }
            Ok((scores, prompt_tokens))
        }
    })
    .await
    .map_err(join_error_response)??;

    let data: Vec<serde_json::Value> = scores
        .iter()
        .enumerate()
        .map(|(index, s)| serde_json::json!({ "index": index, "object": "score", "score": s }))
        .collect();
    Ok((
        serde_json::json!({
            "object": "list",
            "model": model_name,
            "data": data,
            // Which of the two regimes answered. A cosine is in
            // [-1, 1]; a head's logit is whatever it was trained to
            // emit, so a client thresholding one has to know which it
            // got.
            "frink_score_kind": if cross { "cross_encoder" } else { "cosine_similarity" },
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            }
        }),
        prompt_tokens,
    ))
}

fn pair_error(index: usize, e: frink_models::EmbedError) -> ApiError {
    crate::invalid_request(&format!("pair {index}: {e}"), "text_1")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    /// One text against many is the common shape, and the pairing has
    /// to hold whichever side is the single one.
    #[test]
    fn one_against_many_pairs_in_both_directions() {
        assert_eq!(
            pairs(&v(&["q"]), &v(&["a", "b", "c"])).expect("valid"),
            vec![(0, 0), (0, 1), (0, 2)]
        );
        assert_eq!(
            pairs(&v(&["a", "b"]), &v(&["q"])).expect("valid"),
            vec![(0, 0), (1, 0)]
        );
    }

    #[test]
    fn equal_lists_pair_element_by_element() {
        assert_eq!(
            pairs(&v(&["a", "b"]), &v(&["x", "y"])).expect("valid"),
            vec![(0, 0), (1, 1)]
        );
    }

    /// **The one mistake in this shape that returns a plausible
    /// answer.** Zipping two lists of different lengths silently drops
    /// the tail, so the caller gets fewer scores than pairs and no
    /// error. It is a 400 naming both counts.
    #[test]
    fn two_lists_of_different_lengths_are_a_bad_request() {
        let err = pairs(&v(&["a", "b", "c"]), &v(&["x", "y"])).expect_err("refused");
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
        let message = err.1["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains('3') && message.contains('2'), "{message}");
    }

    #[test]
    fn an_empty_side_is_a_bad_request() {
        assert!(pairs(&[], &v(&["a"])).is_err());
        assert!(pairs(&v(&["a"]), &[]).is_err());
    }

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let a = [1.0f32, 2.0, 3.0];
        let c = cosine(&a, &a).expect("defined");
        assert!((c - 1.0).abs() < 1e-6, "{c}");
        let opposite = [-1.0f32, -2.0, -3.0];
        let c = cosine(&a, &opposite).expect("defined");
        assert!((c + 1.0).abs() < 1e-6, "{c}");
    }

    /// A zero vector has no direction, so the angle is UNDEFINED
    /// rather than zero. Reporting `0.0` would say "unrelated".
    #[test]
    fn a_zero_vector_has_no_cosine() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), None);
        assert_eq!(cosine(&[1.0], &[1.0, 1.0]), None, "mismatched widths");
    }
}
