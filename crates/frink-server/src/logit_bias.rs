//! `logit_bias`: a per-token additive shift, applied before every mask.
//!
//! OpenAI's field, and the last sampler knob this server refused. The
//! refusal named three blockers and all three are closed now, which is
//! why this is a row rather than a note:
//!
//! - the whole-response cache keys on the sampler settings, and a bias
//!   outside that key means two requests differing only in their bias
//!   share one answer. `GenerationKey` carries it, as it carries
//!   `crate::token_mask`;
//! - the continuous-batching worker samples through its own call site,
//!   so a bias wired into the private decode loop alone would be
//!   honoured or ignored depending on `FRINK_CONTINUOUS_BATCHING`.
//!   Both loops go through `crate::sample_step::sample_next` now;
//! - on Metal at `temperature <= 0` the decoder folds `lm_head` and
//!   argmax into the GPU stack and returns a ONE-element vector
//!   holding the chosen id, with no vocabulary left to bias.
//!   `GenerationParams::needs_vocab_logits` answers true for a biased
//!   request, which is what refuses the fold.
//!
//! # A constraint always wins, and the ORDER is not what makes it so
//!
//! A mask writes `-f32::INFINITY`; a bias is finite, clamped to
//! `+-100`. So `-inf + 100.0` is still `-inf` and `x + 100.0` is still
//! masked afterwards: the intersection is the same whichever runs
//! first, which is the property the whole mask closure already rests
//! on ("no mask here ever clears a `-inf`").
//!
//! That is worth writing down because the obvious claim -- "the bias
//! must run first or `+100` beats a grammar" -- is FALSE, and was in
//! this file until a sabotage that moved the bias after the masks
//! left every test green. The bias runs first because llama.cpp
//! applies it first, not because anything here depends on it.
//!
//! What a test CAN pin, and does, is the intersection itself: a token
//! forced by a bias and forbidden by `allowed_token_ids` does not come
//! back.

use serde_json::Value;

use crate::ApiError;

/// Upstream's range. A value outside it is a 400 rather than a clamp:
/// clamping answers a question the caller did not ask, and `1e9` is
/// far more likely to be a units mistake than an intent.
const LIMIT: f32 = 100.0;

/// Per-token additive shifts for one request.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct LogitBias {
    /// Sorted by id and deduplicated, so applying is a walk rather
    /// than a scan of a caller-supplied map per token.
    entries: Vec<(usize, f32)>,
}

impl Eq for LogitBias {}

impl std::hash::Hash for LogitBias {
    /// By the BITS of each bias, because two requests that name the
    /// same shift must key the same and `f32` is not `Hash`. `NaN`
    /// cannot reach here -- [`Self::parse`] refuses it -- so there is
    /// no `NaN != NaN` case for the key to disagree with `PartialEq`
    /// about.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for (id, bias) in &self.entries {
            id.hash(state);
            bias.to_bits().hash(state);
        }
    }
}

impl LogitBias {
    /// OpenAI's shape: `{"<token id>": <bias>}`.
    ///
    /// `null` and `{}` are the empty bias, which several clients send
    /// on every request as a default. Refusing those would be a false
    /// refusal: there is no token whose logit they would have moved.
    pub(crate) fn parse(value: Option<&Value>, route: &str) -> Result<Self, ApiError> {
        let Some(value) = value else {
            return Ok(LogitBias::default());
        };
        if value.is_null() {
            return Ok(LogitBias::default());
        }
        let Some(map) = value.as_object() else {
            return Err(crate::invalid_request(
                &format!("`logit_bias` on {route} must be an object of token id to bias"),
                "logit_bias",
            ));
        };
        let mut entries = Vec::with_capacity(map.len());
        for (key, raw) in map {
            let id: usize = key.parse().map_err(|_| {
                crate::invalid_request(
                    &format!("`logit_bias` key {key:?} is not a token id"),
                    "logit_bias",
                )
            })?;
            let bias = raw.as_f64().ok_or_else(|| {
                crate::invalid_request(
                    &format!("`logit_bias[{key}]` is not a number"),
                    "logit_bias",
                )
            })?;
            if !bias.is_finite() || bias.abs() > LIMIT as f64 {
                return Err(crate::invalid_request(
                    &format!(
                        "`logit_bias[{key}]` is {bias}, outside the range -{LIMIT} to {LIMIT}"
                    ),
                    "logit_bias",
                ));
            }
            entries.push((id, bias as f32));
        }
        entries.sort_unstable_by_key(|(id, _)| *id);
        entries.dedup_by_key(|(id, _)| *id);
        Ok(LogitBias { entries })
    }

    /// Whether this request has anything to shift.
    ///
    /// Read by `GenerationParams::needs_vocab_logits`: a backend that
    /// folded `lm_head + argmax` onto the device returns a token id
    /// rather than a vocabulary, and there would be nothing to bias.
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Adds each named bias to its token's logit.
    ///
    /// An id past the end of the vocabulary is skipped rather than a
    /// panic: the ids come from a caller and the logits from a
    /// checkpoint, and a mismatch is a refusal's job, not an index's.
    pub(crate) fn apply(&self, scores: &mut [f32]) {
        for (id, bias) in &self.entries {
            if let Some(score) = scores.get_mut(*id) {
                *score += *bias;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: serde_json::Value) -> LogitBias {
        LogitBias::parse(Some(&v), "/t").expect("valid")
    }

    #[test]
    fn a_bias_shifts_only_the_named_tokens() {
        let bias = parse(serde_json::json!({"1": 5.0, "3": -2.5}));
        let mut scores = vec![0.0f32; 4];
        bias.apply(&mut scores);
        assert_eq!(scores, vec![0.0, 5.0, 0.0, -2.5]);
    }

    /// **`-inf + 100` is still `-inf`.**
    ///
    /// The arithmetic that makes a constraint win regardless of which
    /// runs first. Pinned on its own because the module's first draft
    /// claimed the ORDER was what made it so, and it is not.
    #[test]
    fn a_bias_cannot_lift_a_masked_token() {
        let bias = parse(serde_json::json!({"0": 100.0}));
        let mut scores = vec![f32::NEG_INFINITY, 0.0];
        bias.apply(&mut scores);
        assert_eq!(scores[0], f32::NEG_INFINITY, "a mask was lifted by a bias");
    }

    /// The empty forms several clients send on every request, which
    /// must not be refused: there is no token they would have moved.
    #[test]
    fn the_empty_forms_are_accepted_and_change_nothing() {
        for v in [serde_json::json!({}), serde_json::Value::Null] {
            let bias = LogitBias::parse(Some(&v), "/t").expect("accepted");
            assert!(bias.is_empty());
        }
        assert!(LogitBias::parse(None, "/t").expect("accepted").is_empty());
    }

    /// Out of range is a 400 rather than a clamp: clamping answers a
    /// question the caller did not ask, and a huge value is far more
    /// likely to be a units mistake than an intent.
    #[test]
    fn a_bias_outside_the_range_is_a_bad_request() {
        for v in [
            serde_json::json!({"1": 101.0}),
            serde_json::json!({"1": -100.5}),
        ] {
            let err = LogitBias::parse(Some(&v), "/t").expect_err("refused");
            assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
        }
    }

    #[test]
    fn a_key_that_is_not_a_token_id_is_a_bad_request() {
        let err =
            LogitBias::parse(Some(&serde_json::json!({"abc": 1.0})), "/t").expect_err("refused");
        assert_eq!(err.0, axum::http::StatusCode::BAD_REQUEST);
    }

    /// An id past the vocabulary is skipped rather than panicking.
    #[test]
    fn an_out_of_range_id_does_not_panic() {
        let bias = parse(serde_json::json!({"99": 10.0}));
        let mut scores = vec![0.0f32; 4];
        bias.apply(&mut scores);
        assert_eq!(scores, vec![0.0; 4]);
    }
}
