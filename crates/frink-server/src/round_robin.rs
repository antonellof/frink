//! Several completions of one prompt, interleaved a token at a time.
//!
//! The schedule a STREAMING `n` needs. A client reading
//! `choices[].index` expects the choices to arrive together, and the
//! sequential schedule -- choice 0 to its end, then choice 1 -- gives
//! it choice 0's whole answer before choice 1 says anything. That is
//! why `n` > 1 with `stream` was refused by name: the refusal was
//! about the ORDER, not about the sampling.
//!
//! So this is a scheduler and nothing else. Every per-token rule lives
//! on [`crate::choice_stream::ChoiceStream`], which
//! `sampling_loop::sample_until_stop` drives for the sequential order,
//! and a stop rule cannot fire on one order and not the other because
//! there is one implementation of it.
//!
//! # No drafting here
//!
//! A speculative round commits a BLOCK of tokens at once. In the
//! sequential order that is invisible; interleaved, it would deliver
//! choice 0 five tokens, then choice 1 five tokens, which is the
//! bursty delivery this schedule exists to avoid. The forward passes
//! are already `n`-way on a multi-choice request, so what the drafter
//! saves is smaller here than it is for one sequence.
//!
//! # One engine per choice
//!
//! Each choice has its own KV, so each needs its own engine. They are
//! stepped in index order within a round, which is what makes the
//! interleaving deterministic: a seeded request streams its choices in
//! the same order on every run.

use crate::choice_stream::{ChoiceStream, StepContext};
use crate::generate::{DecodeError, FinishReason};
use crate::sampling_loop::{DecodeEngine, PerTokenProbs};

/// What one completion produced.
pub(crate) type ChoiceOutcome = (FinishReason, Vec<usize>, Vec<f32>, PerTokenProbs);

/// Steps every unfinished choice once per round until all are done.
///
/// `emit` carries the choice index, because an interleaved stream is
/// only readable if each piece of text says which completion it
/// belongs to. That is the same signature `generate` already hands its
/// callers, so the routes did not have to learn a new one.
pub(crate) fn sample_round_robin(
    mut streams: Vec<ChoiceStream>,
    engines: &mut [&mut dyn DecodeEngine],
    contexts: &[StepContext<'_>],
    decode_one: &mut impl FnMut(&[usize]) -> Vec<u8>,
    emit: &mut impl FnMut(usize, &str),
) -> Result<Vec<ChoiceOutcome>, DecodeError> {
    assert_eq!(
        streams.len(),
        engines.len(),
        "one engine per choice, or a choice would decode into another's KV"
    );
    assert_eq!(
        streams.len(),
        contexts.len(),
        "one context per choice: the seed differs per choice and the sampler reads it"
    );
    loop {
        let mut stepped = false;
        for (i, stream) in streams.iter_mut().enumerate() {
            if !stream.check_budget(contexts[i].params) {
                continue;
            }
            stepped = true;
            let mut emit_here = |text: &str| emit(i, text);
            stream.step(&contexts[i], engines[i], decode_one, &mut emit_here)?;
        }
        // Every choice is finished. Checked after a whole round rather
        // than per choice, so a choice that ends early does not end the
        // others with it.
        if !stepped {
            break;
        }
    }
    let mut out = Vec::with_capacity(streams.len());
    for (i, mut stream) in streams.into_iter().enumerate() {
        let mut emit_here = |text: &str| emit(i, text);
        stream.flush(&mut emit_here);
        out.push(stream.into_parts());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::GenerationParams;
    use frink_models::tokenizer::StopTokens;

    /// A decoder that always predicts the next id in its script,
    /// regardless of what it is fed. One per choice, so a choice that
    /// steps out of turn shows up as the wrong script being read.
    struct Scripted {
        script: Vec<usize>,
        vocab: usize,
        steps: usize,
    }

    impl DecodeEngine for Scripted {
        fn step(&mut self, _token: usize, pos: usize) -> Vec<f32> {
            self.steps += 1;
            peaked(self.vocab, self.script.get(pos + 1).copied().unwrap_or(0))
        }
    }

    fn peaked(vocab: usize, winner: usize) -> Vec<f32> {
        let mut row = vec![0.0; vocab];
        row[winner] = 10.0;
        row
    }

    fn params(max_tokens: usize, stop: &[&str]) -> GenerationParams {
        GenerationParams {
            cache_salt: None,
            prompt_logprobs: None,
            wants_logprobs: false,
            n: 2,
            interleave_choices: true,
            reasoning: None,
            max_tokens,
            sampling: frink_models::sampling::SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            seed: 1,
            stop: stop.iter().map(|s| s.to_string()).collect(),
            stop_token_ids: Vec::new(),
            json_object: false,
            grammar: None,
            cancel: None,
            ignore_eos: false,
            reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
            lora: None,
        }
    }

    fn bytes(ids: &[usize]) -> Vec<u8> {
        ids.iter()
            .map(|i| format!("<{i}>"))
            .collect::<String>()
            .into_bytes()
    }

    /// **The choices arrive a token at a time, in index order.**
    ///
    /// The whole reason this module exists. Asserted on the ORDER the
    /// emissions arrive in, because a scheduler that ran choice 0 to
    /// its end would produce exactly the same text.
    #[test]
    fn a_round_steps_every_choice_once_in_index_order() {
        let scripts = [vec![1usize, 2, 3, 4], vec![5usize, 6, 7, 8]];
        let vocab = 16;
        let per = params(3, &[]);
        let stops = StopTokens::default();
        let contexts: Vec<StepContext<'_>> = (0..2)
            .map(|_| StepContext {
                params: &per,
                prompt_ids: &[],
                stop_tokens: &stops,
                decode_token: &|_| String::new(),
            })
            .collect();
        let streams: Vec<ChoiceStream> = scripts
            .iter()
            .map(|s| ChoiceStream::new(peaked(vocab, s[0]), 0, &per))
            .collect();
        let mut engines: Vec<Scripted> = scripts
            .iter()
            .map(|s| Scripted {
                script: s.clone(),
                vocab,
                steps: 0,
            })
            .collect();
        let mut dyns: Vec<&mut dyn DecodeEngine> = engines
            .iter_mut()
            .map(|e| e as &mut dyn DecodeEngine)
            .collect();

        let mut order: Vec<usize> = Vec::new();
        let outcomes = sample_round_robin(
            streams,
            &mut dyns,
            &contexts,
            &mut |ids| bytes(ids),
            &mut |choice, _text| order.push(choice),
        )
        .expect("scripted");

        assert_eq!(
            order,
            vec![0, 1, 0, 1, 0, 1],
            "the choices were not stepped one token each per round"
        );
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].1, vec![1, 2, 3], "choice 0 read another script");
        assert_eq!(outcomes[1].1, vec![5, 6, 7], "choice 1 read another script");
    }

    /// **A choice that stops early does not stop the others.**
    ///
    /// The failure a same-length test cannot see: with every choice
    /// ending on the same round, skipping a finished one and BREAKING
    /// out of the round are the same thing. Here choice 0 hits its
    /// stop string three tokens before choice 1 runs out of budget, so
    /// the two differ.
    #[test]
    fn a_choice_that_finishes_early_leaves_the_others_running() {
        let scripts = [vec![1usize, 2, 3, 4, 5, 6], vec![7usize, 8, 9, 10, 11, 12]];
        let vocab = 16;
        // `<2>` is what the second token of choice 0 decodes to, so
        // choice 0 stops there and choice 1 never sees the string.
        let short = params(6, &["<2>"]);
        let long = params(6, &[]);
        let stops = StopTokens::default();
        let contexts = vec![
            StepContext {
                params: &short,
                prompt_ids: &[],
                stop_tokens: &stops,
                decode_token: &|_| String::new(),
            },
            StepContext {
                params: &long,
                prompt_ids: &[],
                stop_tokens: &stops,
                decode_token: &|_| String::new(),
            },
        ];
        let streams: Vec<ChoiceStream> = vec![
            ChoiceStream::new(peaked(vocab, scripts[0][0]), 0, &short),
            ChoiceStream::new(peaked(vocab, scripts[1][0]), 0, &long),
        ];
        let mut engines: Vec<Scripted> = scripts
            .iter()
            .map(|s| Scripted {
                script: s.clone(),
                vocab,
                steps: 0,
            })
            .collect();
        let mut dyns: Vec<&mut dyn DecodeEngine> = engines
            .iter_mut()
            .map(|e| e as &mut dyn DecodeEngine)
            .collect();

        let mut text = [String::new(), String::new()];
        let outcomes = sample_round_robin(
            streams,
            &mut dyns,
            &contexts,
            &mut |ids| bytes(ids),
            &mut |choice, s| text[choice].push_str(s),
        )
        .expect("scripted");

        assert!(
            matches!(outcomes[0].0, FinishReason::StopSequence(_)),
            "choice 0 should have hit its stop string: {:?}",
            outcomes[0].0
        );
        assert_eq!(
            outcomes[1].1.len(),
            6,
            "choice 1 was cut short by choice 0 finishing: {:?}",
            outcomes[1].1
        );
    }
}
