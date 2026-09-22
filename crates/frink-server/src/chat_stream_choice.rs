//! One streamed chat choice's incremental parser state.
//!
//! Three things travel together per completion: the reasoning split,
//! the tool-call parser fed by it, and how many calls have already
//! gone out on the wire. They were three locals in
//! `chat_completions_stream`, which made the route a ONE-CHOICE route
//! -- `index: 0` was written into every chunk it sent, and `n` > 1
//! with `stream` was refused partly for that reason.
//!
//! As a value there is one per choice, and the route indexes them. The
//! rules are unchanged; what changed is that there can be several.

use std::cell::Cell;

use crate::{tool_call_deltas, ChatCompletionChunkChoice, ChatCompletionChunkDelta, ToolCallDelta};

/// The wire pieces one chunk of one choice carries.
pub(crate) struct Delta {
    pub(crate) reasoning: String,
    pub(crate) content: String,
    pub(crate) tool_calls: Vec<ToolCallDelta>,
}

impl Delta {
    /// Whether there is anything to send. Both parsers withhold
    /// partial markers, so a chunk can legitimately produce nothing at
    /// all this time round.
    pub(crate) fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty() && self.tool_calls.is_empty()
    }

    /// The wire choice, at `index`, announcing the role on the first
    /// chunk this choice sends.
    pub(crate) fn into_choice(self, index: usize, first: bool) -> ChatCompletionChunkChoice {
        ChatCompletionChunkChoice {
            index,
            delta: ChatCompletionChunkDelta {
                role: first.then_some("assistant"),
                content: (!self.content.is_empty()).then_some(self.content),
                reasoning_content: (!self.reasoning.is_empty()).then_some(self.reasoning),
                tool_calls: (!self.tool_calls.is_empty()).then_some(self.tool_calls),
            },
            finish_reason: None,
        }
    }
}

/// One choice's parsers, and whether it has sent anything yet.
pub(crate) struct ChoiceEmitter {
    reasoning: Option<crate::policy::parser::ReasoningParser>,
    tools: Option<crate::policy::parser::ToolCallParser>,
    /// How many calls have been opened on the wire, so the terminal
    /// chunk knows whether to say `tool_calls` and does not repeat what
    /// already went out.
    calls: Cell<usize>,
    /// False until this choice has sent a chunk, which is what decides
    /// where its `role` goes. PER CHOICE: a client reading
    /// `choices[].index` gets a role on the first chunk of each, not
    /// one role for the whole stream.
    started: bool,
}

impl ChoiceEmitter {
    pub(crate) fn new(
        reasoning: Option<crate::policy::parser::ReasoningParser>,
        tools: Option<crate::policy::parser::ToolCallParser>,
    ) -> Self {
        ChoiceEmitter {
            reasoning,
            tools,
            calls: Cell::new(0),
            started: false,
        }
    }

    /// Marks the role as sent, returning whether this call was the one
    /// that did it.
    pub(crate) fn start(&mut self) -> bool {
        let first = !self.started;
        self.started = true;
        first
    }

    pub(crate) fn opened_calls(&self) -> usize {
        self.calls.get()
    }

    /// One generated chunk, split into what the wire carries.
    pub(crate) fn push(&mut self, chunk: &str) -> Delta {
        let (reasoning, content) = match self.reasoning.as_mut() {
            Some(parser) => {
                let delta = parser.push(chunk);
                (delta.reasoning, delta.content)
            }
            None => (String::new(), chunk.to_string()),
        };
        // Content goes through the tool parser, which holds back
        // anything that could still become a marker and turns a
        // recognized call into wire deltas.
        let (content, tool_calls) = match self.tools.as_mut() {
            Some(parser) => tool_call_deltas(parser.push(&content), &self.calls),
            None => (content, Vec::new()),
        };
        Delta {
            reasoning,
            content,
            tool_calls,
        }
    }

    /// Whatever both parsers are still withholding.
    ///
    /// A run that could have become a marker and did not is ordinary
    /// output; dropping it would truncate every answer whose tail
    /// happens to look like the start of a `</think>` or a
    /// `<tool_call>`.
    pub(crate) fn flush(&mut self) -> Delta {
        let tail = self
            .reasoning
            .as_mut()
            .map(|parser| parser.flush())
            .unwrap_or_default();
        let (mut content, mut tool_calls) = (tail.content, Vec::new());
        if let Some(parser) = self.tools.as_mut() {
            let mut events = parser.push(&content);
            events.extend(parser.finish());
            let (text, calls) = tool_call_deltas(events, &self.calls);
            content = text;
            tool_calls = calls;
        }
        Delta {
            reasoning: tail.reasoning,
            content,
            tool_calls,
        }
    }
}
