//! One completion's decode state, steppable one token at a time.
//!
//! `sample_until_stop` used to hold this as eight locals inside a
//! `for` loop, which made "run this completion to its end" the only
//! thing the sampler could do. That is exactly why `n` > 1 with
//! `stream` was refused: a client reading `choices[].index` expects
//! the choices interleaved, and a loop that runs choice 0 to its end
//! before choice 1 begins cannot interleave anything.
//!
//! So the state is a value now, and the SCHEDULE is the caller's:
//! `sample_until_stop` steps one of these until it is done, and
//! `crate::round_robin` steps several a token at a time. One
//! implementation of the per-token rules, two orders -- rather than a
//! second copy of the decode loop, which is how this repo lost five
//! model features from one duplicated path.

use frink_models::tokenizer::StopTokens;

use crate::generate::{DecodeError, FinishReason, GenerationParams};
use crate::sampling_loop::PerTokenProbs;

/// What a committed token means for the rest of the completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Committed {
    Continue,
    Stopped(FinishReason),
}

/// The request-wide facts every step reads and none of them changes.
///
/// One value rather than four arguments repeated at each call, and
/// borrowed rather than owned so a round-robin driver can hand the
/// same context to every choice: the prompt, the stop rules and the
/// tokenizer are the REQUEST's, and only the seed differs per choice.
pub(crate) struct StepContext<'a> {
    pub(crate) params: &'a GenerationParams,
    pub(crate) prompt_ids: &'a [usize],
    pub(crate) stop_tokens: &'a StopTokens,
    pub(crate) decode_token: &'a dyn Fn(usize) -> String,
}

/// One completion in progress.
pub(crate) struct ChoiceStream {
    matcher: crate::stop::StopMatcher,
    /// Sits BEFORE the stop matcher: a stop string is text, so it can
    /// only be matched against whole characters, and half of one is
    /// not text yet.
    utf8: crate::utf8_stream::Utf8Stream,
    state: crate::sample_step::SampleState,
    pub(crate) generated_ids: Vec<usize>,
    pub(crate) finish: FinishReason,
    pub(crate) per_token_probs: PerTokenProbs,
    /// The prediction for the position after the last committed token.
    pub(crate) logits: Vec<f32>,
    pub(crate) pos: usize,
    /// Set by whatever ended this completion. A done stream is never
    /// stepped again, which is the property a round-robin driver rests
    /// on: it skips them and stops when every one is done.
    done: bool,
}

impl ChoiceStream {
    pub(crate) fn new(logits: Vec<f32>, pos: usize, params: &GenerationParams) -> Self {
        // NOT `with_capacity(params.max_tokens)`. That is a
        // caller-supplied number sizing an allocation, and it reached
        // `Vec::with_capacity(usize::MAX)` from one unauthenticated
        // POST. The vector grows as tokens are produced, so the
        // reservation only ever saved reallocations on a path that
        // performs a full model forward pass per element. A cap keeps
        // that saving for the sizes it was worth having for, and
        // refuses to pre-size beyond them.
        const PREALLOC_CAP: usize = 4096;
        ChoiceStream {
            matcher: crate::stop::StopMatcher::new(&params.stop, &params.stop_token_ids),
            utf8: crate::utf8_stream::Utf8Stream::default(),
            state: crate::sample_step::SampleState::new(params.seed),
            generated_ids: Vec::with_capacity(params.max_tokens.min(PREALLOC_CAP)),
            finish: FinishReason::Length,
            per_token_probs: PerTokenProbs::new(),
            logits,
            pos,
            done: false,
        }
    }

    /// The three things the speculative verifier needs, as ONE split
    /// borrow.
    ///
    /// It draws a whole block against this stream's sampler state and
    /// must leave it where an ordinary draw would have, so the state
    /// goes out by mutable reference rather than being copied. The
    /// history is a CLONE because the verifier's own walk appends to
    /// it while `commit` is what really appends here; one clone per
    /// BLOCK, not per token.
    pub(crate) fn verify_inputs(
        &mut self,
    ) -> (&mut crate::sample_step::SampleState, &[f32], Vec<usize>) {
        (&mut self.state, &self.logits, self.generated_ids.clone())
    }

    /// Ends this completion for `reason`, so no later step runs.
    pub(crate) fn stop(&mut self, reason: FinishReason) {
        self.finish = reason;
        self.done = true;
    }

    /// Whether the budget or a cancel has already ended this one.
    ///
    /// Checked before sampling so a cancel that lands between two
    /// tokens costs no further work, and whatever the matcher already
    /// holds is still flushed by [`Self::flush`] -- an interrupted
    /// answer keeps the tokens it earned.
    pub(crate) fn check_budget(&mut self, params: &GenerationParams) -> bool {
        if self.done {
            return false;
        }
        // The budget is a number of TOKENS, and a speculative round
        // commits a whole block, so the count is asked directly rather
        // than inferred from an iteration count. Without this a
        // `max_tokens` of 6 returned nine tokens.
        if self.generated_ids.len() >= params.max_tokens {
            self.stop(FinishReason::Length);
            return false;
        }
        if params.is_cancelled() {
            self.stop(FinishReason::Cancelled);
            return false;
        }
        true
    }

    /// The per-token work every path shares: the two stop layers, the
    /// id, and the text.
    ///
    /// Note what is NOT here: feeding the token forward. A speculated
    /// token's row may already exist, and the caller knows which.
    pub(crate) fn commit(
        &mut self,
        next: usize,
        ctx: &StepContext<'_>,
        decode_one: &mut impl FnMut(&[usize]) -> Vec<u8>,
        emit: &mut impl FnMut(&str),
    ) -> Committed {
        if !ctx.params.ignore_eos && ctx.stop_tokens.contains(next) {
            // `skip_special_tokens: false`: the marker that ended the
            // answer is part of the answer. It still ends it -- the
            // field is about what comes back, not about when to stop.
            if ctx.params.keep_special_tokens {
                self.generated_ids.push(next);
                let text = self
                    .matcher
                    .flush_with(&self.utf8.push(&decode_one(&[next])));
                if !text.is_empty() {
                    emit(&text);
                }
            }
            return Committed::Stopped(FinishReason::Stop);
        }
        // Layer 1: before the token is detokenized or counted. A
        // control token the client asked to stop on is not part of the
        // answer, so it contributes neither an id nor a character --
        // exactly how `eos_id` is treated one line above.
        if self.matcher.is_stop_token(next) {
            return Committed::Stopped(FinishReason::Stop);
        }
        self.generated_ids.push(next);

        // Layer 2: only text that can no longer become part of a stop
        // string leaves here.
        match self.matcher.push(&self.utf8.push(&decode_one(&[next]))) {
            crate::stop::StopStep::Emit(text) => {
                if !text.is_empty() {
                    emit(&text);
                }
                Committed::Continue
            }
            crate::stop::StopStep::Matched { text, stop } => {
                if !text.is_empty() {
                    emit(&text);
                }
                Committed::Stopped(FinishReason::StopSequence(stop))
            }
        }
    }

    /// Draws one token, commits it, and feeds it forward.
    ///
    /// The whole of the ordinary decode step, so a caller that wants a
    /// different ORDER of completions does not get a different set of
    /// stop rules with it.
    pub(crate) fn step(
        &mut self,
        ctx: &StepContext<'_>,
        engine: &mut dyn crate::sampling_loop::DecodeEngine,
        decode_one: &mut impl FnMut(&[usize]) -> Vec<u8>,
        emit: &mut impl FnMut(&str),
    ) -> Result<(), DecodeError> {
        let (next, next_probs) = match crate::sample_step::sample_next(
            &mut self.state,
            &self.logits,
            ctx.params,
            ctx.prompt_ids,
            &self.generated_ids,
            ctx.stop_tokens,
            ctx.decode_token,
        )? {
            crate::sample_step::Step::Token { id, probs } => (id, probs),
            // The grammar's parse is complete and nothing may follow
            // it. A finished answer, so `Stop` -- the same reason the
            // model's own end-of-generation token gives, since it is
            // the same statement made by the constraint instead of by
            // the model.
            crate::sample_step::Step::GrammarComplete => {
                self.stop(FinishReason::Stop);
                return Ok(());
            }
        };
        match self.commit(next, ctx, decode_one, emit) {
            Committed::Continue => {
                // Recorded only for tokens that were KEPT: a stop
                // token is not part of the answer, so a logprob for it
                // would describe a position no `choices[]` entry has.
                if let Some(p) = next_probs {
                    self.per_token_probs.push((next, p));
                }
            }
            Committed::Stopped(reason) => {
                self.stop(reason);
                return Ok(());
            }
        }
        self.logits = engine.step(next, self.pos);
        self.pos += 1;
        Ok(())
    }

    /// Everything the matcher and the decoder were still holding.
    ///
    /// A generation that stopped mid-character cannot complete it, so
    /// the held bytes surface as U+FFFD rather than vanishing, and
    /// that goes through the matcher like any other text. Whatever is
    /// still withheld after it was output no stop ever claimed.
    pub(crate) fn flush(&mut self, emit: &mut impl FnMut(&str)) {
        let partial = self.utf8.flush();
        if !partial.is_empty() {
            let (crate::stop::StopStep::Emit(text) | crate::stop::StopStep::Matched { text, .. }) =
                self.matcher.push(&partial);
            if !text.is_empty() {
                emit(&text);
            }
        }
        let tail = self.matcher.flush();
        if !tail.is_empty() {
            emit(&tail);
        }
    }

    pub(crate) fn into_parts(self) -> (FinishReason, Vec<usize>, Vec<f32>, PerTokenProbs) {
        (
            self.finish,
            self.generated_ids,
            self.logits,
            self.per_token_probs,
        )
    }
}
