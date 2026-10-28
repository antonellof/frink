//! What one request has produced so far, as a single value.
//!
//! A row's state splits into two kinds. The KV, the position and the
//! logits are a function of the tokens the row has run, so they can be
//! recomputed from those tokens. Everything here cannot: the sampler's
//! RNG has advanced, the grammar has consumed every generated token,
//! the stop matcher is withholding a tail the client has not seen, the
//! UTF-8 buffer may hold half a character, and some text has already
//! been streamed. Recompute any of that and a client sees a duplicate
//! or missing delta, or a seeded request that no longer reproduces.
//!
//! That split is what preemption (`super::preempt`) needs: a row is
//! evicted by dropping the recomputable half and carrying this one back
//! to the queue. Keeping it as ONE type, rather than as fields on
//! `Slot` that a preemption path copies across one at a time, is what
//! stops the two from disagreeing about which fields survive -- a field
//! added here survives by construction, and a field added to `Slot` has
//! to be classified in `Slot::into_parts`, whose destructure has no
//! `..`.

use std::sync::mpsc::Sender;

use crate::generate::{DecodeError, FinishReason, GenerationParams};
use crate::sample_step::SampleState;
use crate::stop::StopMatcher;
use crate::utf8_stream::Utf8Stream;

use super::clock::RowClock;
use super::config::{send_finished, BatcherEvent};

pub(super) struct Answer {
    /// The seeded sampler AND this row's live grammar parse, in one
    /// value: a row that could hold the first without the second is a
    /// row that can sample a token its grammar forbids. See
    /// [`crate::sample_step`].
    pub(super) sample: SampleState,
    pub(super) generated_ids: Vec<usize>,
    /// Detokenized text already safe to expose (past the stop
    /// hold-back), and therefore already sent to a streaming client.
    pub(super) visible: String,
    /// Both stop layers for this row, including the text it is
    /// withholding against a stop string that may still complete.
    pub(super) stops: StopMatcher,
    /// This row's half-finished character, if a token ended inside one.
    /// Per row: rows interleave, and one shared buffer would splice one
    /// answer's bytes into another's.
    pub(super) utf8: Utf8Stream,
    /// The row's own clock, started at FIRST admission. A preempted row
    /// keeps it, so its `Usage` spans the whole request and its TTFT is
    /// the one the client actually waited.
    pub(super) clock: RowClock,
    /// The *real* prompt length, for `Usage`. Not the length of the
    /// tokens a prefill runs, which is 1 for an empty prompt and, on a
    /// resumed row, includes everything it had generated.
    pub(super) prompt_tokens: usize,
    /// Prompt positions adopted from the radix tree at FIRST admission.
    ///
    /// Read off the lease where it was taken rather than at the end: a
    /// finished row has already published its own prefix, and so has a
    /// preempted one, so asking a later lease would count this
    /// request's own contribution as its reuse.
    ///
    /// `None` when no radix tree is configured, which is the same
    /// distinction `usage.cached_tokens` carries everywhere else.
    pub(super) cached_tokens: Option<usize>,
    /// Times this request has been evicted and re-queued. Bounded by
    /// [`super::preempt::MAX_PREEMPTIONS`], which is what makes the
    /// policy terminate.
    pub(super) preemptions: u32,
}

impl Answer {
    /// The state of a request that has produced nothing yet. The clock
    /// starts here, at admission.
    pub(super) fn fresh(
        params: &GenerationParams,
        prompt_tokens: usize,
        cached_tokens: Option<usize>,
    ) -> Self {
        Answer {
            sample: SampleState::new(params.seed),
            generated_ids: Vec::with_capacity(params.max_tokens),
            visible: String::new(),
            stops: StopMatcher::new(&params.stop, &params.stop_token_ids),
            utf8: Utf8Stream::default(),
            clock: RowClock::start(),
            prompt_tokens,
            cached_tokens,
            preemptions: 0,
        }
    }

    /// Sends this answer to its caller as finished. The one exit every
    /// admitted request takes, however it ended: finished, cancelled
    /// mid-prefill, cancelled while waiting after a preemption, or
    /// failed.
    pub(super) fn reply(
        mut self,
        reply: &Sender<BatcherEvent>,
        finish: FinishReason,
        error: Option<DecodeError>,
    ) {
        // A row that failed reports the failure, not the partial text it
        // had produced: the caller asked for output under a constraint,
        // and text that stops short of satisfying it is not a shorter
        // answer to that question.
        if let Some(error) = error {
            send_finished(reply, Err(error));
            return;
        }
        // Text withheld against a stop that never arrived is ordinary
        // output; dropping it would truncate every answer whose tail
        // looks like the start of a stop string.
        // A row cut off mid-character cannot finish it, so the held
        // bytes surface as U+FFFD instead of vanishing -- through the
        // stop matcher, like any other text this row produced.
        let partial = self.utf8.flush();
        if !partial.is_empty() {
            self.stops.push(&partial);
        }
        let tail = self.stops.flush();
        if !tail.is_empty() {
            self.visible.push_str(&tail);
            let _ = reply.send(BatcherEvent::Chunk(tail));
        }
        let usage = self.clock.usage(
            self.prompt_tokens,
            self.generated_ids.len(),
            self.cached_tokens,
        );
        send_finished(reply, Ok((finish, self.generated_ids, self.visible, usage)));
    }
}
