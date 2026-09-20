//! The token loop: draw, decode, check for a stop, repeat.
//!
//! Split out of `generate.rs` when the server's speculative decoding
//! row began, because that change adds a SECOND loop beside this one
//! (verify a block of drafted tokens rather than draw a single next
//! token) and `generate.rs` was 4,583 lines. The repo's rule is that a
//! new file beats a new section, and the split comes first.
//!
//! The one invariant worth stating here: every token this loop emits
//! comes from `crate::sample_step::sample_next`, with the penalty
//! window over `prompt ++ generated` and the grammar machine advanced
//! in order. The speculative loop beside it will emit tokens from the
//! SAME function for the same reason -- a second sampler would be two
//! structures that must agree about what the model said.

use frink_models::tokenizer::StopTokens;

use crate::generate::{DecodeError, FinishReason, GenerationParams};

/// Fallible since constrained decoding landed: a grammar that cannot be
/// continued ends the generation with an error rather than with an
/// answer, because the alternative is to emit a token the caller's
/// grammar forbids and report it as constrained output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_until_stop(
    mut logits: Vec<f32>,
    mut pos: usize,
    // The prompt this generation continues. Passed rather than derived
    // because the penalties window is the tail of `prompt ++ generated`
    // (llama-server seeds its sampler with the prompt before the first
    // draw), and this seam previously had no way to see it, so the HTTP
    // API and `frink run` disagreed about what one flag means (#73).
    prompt_ids: &[usize],
    stop_tokens: &StopTokens,
    params: &GenerationParams,
    // Raw BYTES, not text. A character can straddle two tokens, and
    // deciding UTF-8 per token destroys it -- see `crate::utf8_stream`.
    mut decode_one: impl FnMut(&[usize]) -> Vec<u8>,
    mut step: impl FnMut(usize, usize) -> Vec<f32>,
    mut emit: impl FnMut(&str),
    decode_token: &dyn Fn(usize) -> String,
) -> Result<(FinishReason, Vec<usize>, Vec<f32>), DecodeError> {
    let mut matcher = crate::stop::StopMatcher::new(&params.stop, &params.stop_token_ids);
    // Sits BEFORE the stop matcher: a stop string is text, so it can
    // only be matched against whole characters, and half of one is not
    // text yet.
    let mut utf8 = crate::utf8_stream::Utf8Stream::default();
    let mut state = crate::sample_step::SampleState::new(params.seed);
    // NOT `with_capacity(params.max_tokens)`. That is a caller-supplied
    // number sizing an allocation, and it reached
    // `Vec::with_capacity(usize::MAX)` from one unauthenticated POST.
    // The vector grows as tokens are produced, so the reservation only
    // ever saved reallocations on a path that performs a full model
    // forward pass per element. A cap keeps that saving for the sizes
    // it was worth having for, and refuses to pre-size beyond them.
    const PREALLOC_CAP: usize = 4096;
    let mut generated_ids: Vec<usize> = Vec::with_capacity(params.max_tokens.min(PREALLOC_CAP));
    let mut finish = FinishReason::Length;

    for _ in 0..params.max_tokens {
        // The one place cancellation is honoured, shared by `generate`
        // and `generate_engine`. Checked before sampling so a cancel
        // that lands between two tokens costs no further work, and
        // whatever `pending` already holds is still flushed below --
        // an interrupted answer keeps the tokens it earned.
        if params.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }
        let next = match crate::sample_step::sample_next(
            &mut state,
            &logits,
            params,
            prompt_ids,
            &generated_ids,
            stop_tokens,
            decode_token,
        )? {
            crate::sample_step::Step::Token(next) => next,
            // The grammar's parse is complete and nothing may follow
            // it. A finished answer, so `Stop` -- the same reason the
            // model's own end-of-generation token gives, since it is
            // the same statement made by the constraint instead of by
            // the model.
            crate::sample_step::Step::GrammarComplete => {
                finish = FinishReason::Stop;
                break;
            }
        };
        if !params.ignore_eos && stop_tokens.contains(next) {
            finish = FinishReason::Stop;
            break;
        }
        // Layer 1: before the token is detokenized or counted. A
        // control token the client asked to stop on is not part of the
        // answer, so it contributes neither an id nor a character --
        // exactly how `eos_id` is treated one line above.
        if matcher.is_stop_token(next) {
            finish = FinishReason::Stop;
            break;
        }
        generated_ids.push(next);
        logits = step(next, pos);
        pos += 1;

        // Layer 2: only text that can no longer become part of a stop
        // string leaves here.
        match matcher.push(&utf8.push(&decode_one(&[next]))) {
            crate::stop::StopStep::Emit(text) => {
                if !text.is_empty() {
                    emit(&text);
                }
            }
            crate::stop::StopStep::Matched { text, stop } => {
                if !text.is_empty() {
                    emit(&text);
                }
                finish = FinishReason::StopSequence(stop);
                break;
            }
        }
    }

    // A generation that stopped mid-character cannot complete it, so
    // the held bytes surface as U+FFFD rather than vanishing -- that
    // goes through the matcher like any other text.
    let partial = utf8.flush();
    if !partial.is_empty() {
        let (crate::stop::StopStep::Emit(text) | crate::stop::StopStep::Matched { text, .. }) =
            matcher.push(&partial);
        if !text.is_empty() {
            emit(&text);
        }
    }

    // Ended for some other reason (length, EOS, a cancel): whatever is
    // still withheld was output that no stop ever claimed.
    let tail = matcher.flush();
    if !tail.is_empty() {
        emit(&tail);
    }

    Ok((finish, generated_ids, logits))
}
/// The earliest byte offset in `text` at which any of `stops` begins,
/// or `None` if none match yet.
pub(crate) fn earliest_stop_match<'a>(text: &str, stops: &'a [String]) -> Option<(usize, &'a str)> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()).map(|at| (at, s.as_str())))
        // Leftmost wins, because that is where the answer is cut. Two
        // stops starting at the same place cut identically, so the tie
        // is broken on LENGTH, longest first: `"</tool_call>"` and
        // `"</tool"` both match at the same index and the longer one is
        // the more specific claim about what the model produced. Some
        // rule is needed either way -- without one the reported stop
        // would depend on the order the caller happened to list them.
        .min_by_key(|(at, s)| (*at, std::cmp::Reverse(s.len())))
}

/// The largest char boundary `<= idx`. `str::floor_char_boundary` is
/// still nightly-only in stable Rust as of this writing; the
/// walk-backward-to-a-boundary logic lives in `frink-edge`, next to
/// the byte-length withhold rules that produce the indices it is
/// applied to.
pub(crate) fn floor_char_boundary(s: &str, idx: usize) -> usize {
    crate::policy::detokenize::floor_char_boundary(s, idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Moved here with `earliest_stop_match` itself: a test that stays
    /// behind when its subject moves is a test nobody runs against the
    /// thing it names.
    #[test]
    fn earliest_stop_match_finds_the_leftmost_match_across_multiple_stops() {
        assert_eq!(
            earliest_stop_match("hello world", &["world".to_string(), "hello".to_string()]),
            Some((0, "hello")),
            "the leftmost match wins, not the caller's first entry"
        );
        assert_eq!(
            earliest_stop_match("hello world", &["nope".to_string()]),
            None
        );
    }
}
