//! Stop sequences, in two layers.
//!
//! A stop sequence is a promise about what reaches the client: the
//! text up to the stop is the answer, and the stop itself and
//! everything after it never existed. Streaming makes that promise hard
//! to keep, because output leaves in pieces and a stop string can
//! straddle any number of them. Emit each piece as it arrives and a
//! client watching a `</tool_call>` stop sees `</tool` appear on screen
//! and then get taken back -- if the transport even allows taking it
//! back, which SSE does not.
//!
//! # Layer 1: the token matcher
//!
//! When a stop string is exactly one token in this model's vocabulary
//! -- which is the usual case for the chat-control strings people
//! actually stop on, `<|im_end|>`, `<end_of_turn>`, `<|eot_id|>` -- the
//! honest test is on the token id, and it is made *before* the token is
//! detokenized or appended to anything.
//!
//! This is not an optimization of layer 2. It answers a question layer
//! 2 cannot: whether the model emitted that control token. The text
//! layer can only ask whether the token's rendered form happens to
//! spell the stop string, and a rendered form is a tokenizer's choice
//! -- byte-fallback pieces, added-token display forms, and specials
//! that render to nothing all break the spelling while leaving the id
//! exact. Matching the id also cannot be fooled from the other
//! direction, by ordinary text that happens to spell a control string.
//!
//! # Layer 2: output-suffix buffering
//!
//! For everything else, the emitted text is buffered so that **no
//! suffix which could still become a stop string is ever released**.
//! After each piece:
//!
//! 1. If a stop matches, emit up to the match and finish.
//! 2. Otherwise find the longest suffix of the buffer that is a proper
//!    prefix of some stop string, and hold back exactly that much.
//!    Everything before it can never be part of a match, so it goes out
//!    immediately.
//!
//! Step 2 is the part worth being precise about. The obvious
//! implementation withholds a fixed `longest_stop - 1` bytes, which is
//! *safe* -- no partial match escapes -- but withholds those bytes from
//! every request with a stop sequence whether or not anything is
//! actually pending. With `stop: ["<|im_end|>"]` that is nine bytes of
//! answer permanently one step behind the model, for a match that in
//! almost every chunk is not even beginning. Withholding the real
//! partial instead means text goes out the moment it is provably safe,
//! and the buffer is empty in the common case.
//!
//! Both layers are needed. Layer 1 alone misses every multi-token and
//! user-supplied stop string; layer 2 alone misses the control tokens
//! and streams fragments of the ones it does catch.

use std::collections::HashSet;

use crate::sampling_loop::{earliest_stop_match, floor_char_boundary};

/// What a piece of decoded output should do to the stream.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum StopStep {
    /// Text that is provably safe to send. May be empty when the whole
    /// piece is still a possible partial match.
    Emit(String),
    /// A stop string matched. `text` (possibly empty) is the last text
    /// of the answer; generation should end.
    ///
    /// `stop` is WHICH string matched, kept because two protocols ask:
    /// Anthropic reports it as `stop_sequence`, and an agent branches
    /// on it to tell its own fence from the model ending its turn. It
    /// is the stop as the caller spelled it, which is the form the
    /// caller can compare against its own list.
    Matched { text: String, stop: String },
}

/// Resolves which stop strings are single tokens in this vocabulary.
///
/// A stop string that encodes to exactly one id gets layer 1. Anything
/// else -- multi-token strings, and strings the tokenizer splits -- is
/// layer 2's business, and no attempt is made to guess at a token
/// sequence: a multi-token encoding is not a reliable statement about
/// how the *model* will emit that text, since a different tokenization
/// of the same string is still the same string.
pub(crate) fn resolve_stop_tokens(
    stops: &[String],
    encode: impl Fn(&str) -> Vec<usize>,
) -> Vec<usize> {
    let mut ids = Vec::new();
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        let encoded = encode(stop);
        if encoded.len() == 1 && !ids.contains(&encoded[0]) {
            ids.push(encoded[0]);
        }
    }
    ids
}

/// Both layers, as one object a decode loop can hold per request.
pub(crate) struct StopMatcher {
    stops: Vec<String>,
    stop_tokens: HashSet<usize>,
    /// Output withheld because it might still complete a stop string.
    pending: String,
}

impl StopMatcher {
    pub(crate) fn new(stops: &[String], stop_tokens: &[usize]) -> Self {
        StopMatcher {
            // An empty stop string can never be matched against and
            // would make `earliest_stop_match` fire at position 0 on
            // everything.
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            stop_tokens: stop_tokens.iter().copied().collect(),
            pending: String::new(),
        }
    }

    /// Layer 1. Checked before the token is detokenized, so a stop
    /// token contributes nothing to the answer whatever it renders as.
    pub(crate) fn is_stop_token(&self, token: usize) -> bool {
        self.stop_tokens.contains(&token)
    }

    /// Layer 2. Feeds one decoded piece through the buffer.
    pub(crate) fn push(&mut self, piece: &str) -> StopStep {
        if self.stops.is_empty() {
            // Nothing to match, so nothing to withhold. Not merely an
            // optimization: buffering with no stop strings would delay
            // output for no reason at all.
            return StopStep::Emit(piece.to_string());
        }
        self.pending.push_str(piece);

        if let Some((cut, stop)) = earliest_stop_match(&self.pending, &self.stops) {
            let matched = StopStep::Matched {
                text: self.pending[..cut].to_string(),
                stop: stop.to_string(),
            };
            self.pending.clear();
            return matched;
        }

        let keep = partial_suffix_len(&self.pending, &self.stops);
        let split = floor_char_boundary(&self.pending, self.pending.len() - keep);
        let out: String = self.pending.drain(..split).collect();
        StopStep::Emit(out)
    }

    /// Whatever is still held back when generation ends for a reason
    /// other than a stop match -- length, EOS, cancellation.
    ///
    /// It is released, not discarded: it was withheld against a match
    /// that never came, so it is ordinary output, and dropping it would
    /// silently truncate every answer whose tail happens to look like
    /// the start of a stop string.
    pub(crate) fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Everything withheld, plus `tail`, as one string.
    ///
    /// For the LAST piece of an answer, where there is nothing after
    /// it that could still turn the held bytes into a stop string.
    /// `push` would withhold a suffix against a match that can no
    /// longer arrive, so the final token of a `skip_special_tokens:
    /// false` answer would vanish.
    pub(crate) fn flush_with(&mut self, tail: &str) -> String {
        let mut out = std::mem::take(&mut self.pending);
        out.push_str(tail);
        out
    }
}

/// The length, in bytes, of the longest suffix of `pending` that is a
/// *proper* prefix of some stop string.
///
/// Proper: a full match is layer 2's other branch and has already been
/// checked, so a suffix equal to a whole stop string is not what this
/// looks for.
///
/// The rule itself lives in `crate::policy::detokenize::stop_prefix_holdback`, which
/// the streaming detokenizer and both output parsers also withhold
/// against. One implementation, because three copies of a rule this
/// exact would disagree eventually, and the disagreement would show up
/// as a partial stop string reaching a client on one code path and not
/// another.
fn partial_suffix_len(pending: &str, stops: &[String]) -> usize {
    crate::policy::detokenize::stop_prefix_holdback(pending, stops)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn matcher(list: &[&str]) -> StopMatcher {
        StopMatcher::new(&stops(list), &[])
    }

    /// The promise the whole module exists to keep: no prefix of a stop
    /// string ever reaches the wire, not even for one chunk.
    ///
    /// `</tool` must not appear and then be taken back -- SSE has no
    /// mechanism for taking anything back.
    #[test]
    fn a_partial_match_is_never_emitted() {
        let mut m = matcher(&["</tool_call>"]);
        assert_eq!(m.push("answer "), StopStep::Emit("answer ".into()));
        // Every one of these is a growing prefix of the stop string and
        // must be held.
        for piece in ["<", "/", "tool", "_ca"] {
            assert_eq!(
                m.push(piece),
                StopStep::Emit(String::new()),
                "a growing partial match escaped at {piece:?}"
            );
        }
        assert_eq!(
            m.push("ll>"),
            StopStep::Matched {
                text: String::new(),
                stop: "</tool_call>".into()
            }
        );
    }

    /// The precision half: text that cannot possibly be part of a stop
    /// goes out at once, rather than trailing `longest_stop - 1` bytes
    /// behind the model forever.
    #[test]
    fn text_that_cannot_match_is_not_withheld() {
        let mut m = matcher(&["<|im_end|>"]);
        assert_eq!(m.push("hello"), StopStep::Emit("hello".into()));
        assert_eq!(m.push(" world"), StopStep::Emit(" world".into()));
        assert_eq!(
            m.flush(),
            "",
            "nothing should still be held when nothing could match"
        );
    }

    /// A partial that turns out not to be one is released as soon as it
    /// is disproved, not at the end of the answer.
    #[test]
    fn an_abandoned_partial_match_is_released_immediately() {
        let mut m = matcher(&["<|im_end|>"]);
        assert_eq!(m.push("a<|im"), StopStep::Emit("a".into()));
        // `_x` cannot continue `<|im`, so the whole thing is ordinary
        // text again.
        assert_eq!(m.push("_x"), StopStep::Emit("<|im_x".into()));
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn a_stop_in_the_middle_of_a_piece_cuts_it_there() {
        let mut m = matcher(&["STOP"]);
        assert_eq!(
            m.push("keep thisSTOPdrop this"),
            StopStep::Matched {
                text: "keep this".into(),
                stop: "STOP".into()
            }
        );
        assert_eq!(m.flush(), "", "everything after the stop is gone");
    }

    #[test]
    fn the_leftmost_stop_wins_when_several_could_match() {
        let mut m = matcher(&["world", "hello"]);
        assert_eq!(
            m.push("say hello world"),
            StopStep::Matched {
                text: "say ".into(),
                // The leftmost match is "hello", and that is the stop
                // reported -- not "world", which the caller listed
                // first and which the text never reached.
                stop: "hello".into()
            }
        );
    }

    /// Withheld text is output that was never disproved, so it must be
    /// released when generation ends for any other reason. Dropping it
    /// would truncate every answer ending in something that looks like
    /// the start of a stop string.
    #[test]
    fn a_pending_partial_is_released_when_generation_ends_otherwise() {
        let mut m = matcher(&["<|im_end|>"]);
        assert_eq!(m.push("done<|im"), StopStep::Emit("done".into()));
        assert_eq!(m.flush(), "<|im");
        assert_eq!(m.flush(), "", "flushing twice must not duplicate it");
    }

    #[test]
    fn no_stop_sequences_means_no_buffering_at_all() {
        let mut m = matcher(&[]);
        assert_eq!(m.push("<|im"), StopStep::Emit("<|im".into()));
        assert_eq!(m.flush(), "");
    }

    /// An empty stop string matches everywhere and nowhere. Keeping it
    /// would end every generation at its first token.
    #[test]
    fn an_empty_stop_string_is_ignored() {
        let mut m = matcher(&[""]);
        assert_eq!(m.push("hello"), StopStep::Emit("hello".into()));
    }

    /// The buffer must never split a character, whatever the byte
    /// arithmetic says.
    #[test]
    fn multibyte_text_is_never_cut_mid_character() {
        let mut m = matcher(&["éx"]);
        // "é" is a two-byte prefix of the stop string.
        assert_eq!(m.push("aé"), StopStep::Emit("a".into()));
        assert_eq!(m.push("y"), StopStep::Emit("éy".into()));

        let mut m = matcher(&["ありがとう"]);
        assert_eq!(m.push("あり"), StopStep::Emit(String::new()));
        assert_eq!(
            m.push("がとう"),
            StopStep::Matched {
                text: String::new(),
                stop: "ありがとう".into()
            }
        );
    }

    #[test]
    fn partial_suffix_length_is_the_longest_real_partial() {
        let s = stops(&["abc"]);
        assert_eq!(partial_suffix_len("xxab", &s), 2);
        assert_eq!(partial_suffix_len("xxa", &s), 1);
        assert_eq!(partial_suffix_len("xxb", &s), 0);
        // A whole match is not a *proper* prefix and is not this
        // function's job.
        assert_eq!(partial_suffix_len("abc", &s), 0);
        // Longest wins across several stops.
        let s = stops(&["ab", "abcd"]);
        assert_eq!(partial_suffix_len("xabc", &s), 3);
    }

    // ---------------- layer 1 -------------------------------------

    #[test]
    fn a_single_token_stop_string_becomes_a_stop_token() {
        // `<|im_end|>` is one token; `hello world` is three.
        let encode = |text: &str| match text {
            "<|im_end|>" => vec![100usize],
            "hello world" => vec![1, 2, 3],
            _ => vec![],
        };
        let ids = resolve_stop_tokens(&stops(&["<|im_end|>", "hello world"]), encode);
        assert_eq!(
            ids,
            vec![100],
            "only the single-token stop is a token-level stop"
        );
    }

    #[test]
    fn duplicate_and_empty_stop_strings_do_not_duplicate_ids() {
        let encode = |text: &str| match text {
            "<|end|>" => vec![7usize],
            _ => vec![],
        };
        let ids = resolve_stop_tokens(&stops(&["<|end|>", "<|end|>", ""]), encode);
        assert_eq!(ids, vec![7]);
    }

    /// The case layer 2 cannot cover: a control token whose rendered
    /// form is not the string the client asked to stop on. The id is
    /// exact; the spelling is the tokenizer's choice.
    #[test]
    fn a_stop_token_is_matched_by_id_not_by_how_it_renders() {
        let m = StopMatcher::new(&stops(&["<|im_end|>"]), &[100]);
        assert!(m.is_stop_token(100));
        assert!(!m.is_stop_token(101));

        // Same matcher, and the text layer would never see it: this
        // model renders token 100 as the empty string.
        let mut m = StopMatcher::new(&stops(&["<|im_end|>"]), &[100]);
        assert_eq!(m.push(""), StopStep::Emit(String::new()));
        assert_eq!(
            m.flush(),
            "",
            "the text layer has nothing to go on -- that is layer 1's job"
        );
    }

    /// Layer 1 works with no stop strings at all -- a caller may have
    /// resolved a control token without the client naming any text.
    #[test]
    fn stop_tokens_work_without_any_stop_strings() {
        let mut m = StopMatcher::new(&[], &[42]);
        assert!(m.is_stop_token(42));
        assert!(!m.is_stop_token(43));
        assert_eq!(m.push("free text"), StopStep::Emit("free text".into()));
    }
}
