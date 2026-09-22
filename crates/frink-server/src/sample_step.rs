//! One token, chosen under every per-request constraint on the logits.
//!
//! There are two decode loops in this server -- the private
//! `generate::sample_until_stop` loop and the continuous batcher's
//! per-row step in `serving::batch::worker` -- and for as long as each
//! held its own call to `Sampler`, they disagreed about what a request
//! meant. `response_format: {"type": "json_object"}` was honoured on one
//! and dropped on the other, so the same body got structured output or
//! not depending on `FRINK_CONTINUOUS_BATCHING`: an env var the caller
//! cannot see deciding a per-request feature.
//!
//! So the choice lives here once and both loops call it. Adding a
//! constraint means adding it to [`sample_next`], and it is then in
//! effect on both paths or on neither -- which is the only way the two
//! can be made to agree about a constraint that does not exist yet.
//!
//! # Why the state is a struct and not a `Sampler`
//!
//! A grammar is not a filter, it is a parse in progress: the mask before
//! the sample and the accept after it are one operation split in half,
//! and a loop that keeps only the first half emits text that satisfies
//! the grammar's FIRST token forever. So a decode loop no longer holds a
//! `Sampler`, it holds a [`SampleState`], and the only thing it can do
//! with one is call [`sample_next`] -- which does both halves. A third
//! decode loop cannot repeat the JSON-mode bug here, because there is no
//! shorter call for it to reach for.

use frink_models::grammar_sampler::{ConstraintError, GrammarSampler, MaskOutcome};
use frink_models::penalty_window::PenaltyWindow;
use frink_models::sampling::reasoning_budget::{lossy_piece_is_complete, ReasoningBudget};
use frink_models::sampling::Sampler;
use frink_models::tokenizer::StopTokens;

use crate::generate::{DecodeError, GenerationParams};
use crate::json_mode::mask_logits_for_json;

/// Everything a decode loop carries between token steps for ONE request:
/// the seeded sampler, and the live grammar parse when there is one.
pub(crate) struct SampleState {
    sampler: Sampler,
    /// Built on the first constrained step rather than at construction:
    /// the vocabulary view a grammar needs (every token's piece, every
    /// token's end-of-generation flag) is only knowable once a logits
    /// vector has said how big the vocabulary is, and the batcher builds
    /// its rows before it has ever decoded one.
    grammar: Option<GrammarSampler>,
    /// The reasoning budget machine, built on the first step from the
    /// armed plan on the params -- the same lazy shape as `grammar`,
    /// and for the same reason: the batcher builds its rows before it
    /// has a step to hand them.
    budget: Option<ReasoningBudget>,
}

impl SampleState {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            sampler: Sampler::new(seed),
            grammar: None,
            budget: None,
        }
    }

    /// The budget machine's state, for tests: `None` before the first
    /// step and for a request without a budget.
    #[cfg(test)]
    pub(crate) fn budget_state(
        &self,
    ) -> Option<frink_models::sampling::reasoning_budget::BudgetState> {
        self.budget.as_ref().map(ReasoningBudget::state)
    }

    /// Whether a grammar is being applied. For tests and diagnostics;
    /// `false` before the first constrained step even for a request that
    /// has a grammar.
    #[cfg(test)]
    pub(crate) fn grammar_started(&self) -> bool {
        self.grammar.is_some()
    }
}

/// What one step of a decode loop got out of the sampler.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Step {
    /// Sampled this token id, and every constraint has been advanced
    /// over it.
    ///
    /// `probs` is the distribution it was drawn from, present only
    /// when the request asked for logprobs -- computing it costs the
    /// greedy fast path, so it is not paid for by requests that will
    /// not read it. `None` also for a device-folded argmax, which has
    /// no vocabulary behind it (`Sampler::sample_reporting`).
    Token { id: usize, probs: Option<Vec<f32>> },
    /// No token: the grammar's parse is complete and nothing may follow
    /// it, so the generation is finished with what it already has.
    ///
    /// Distinct from a stop token, which is a token this model chose;
    /// nothing was chosen here. Distinct from an error, which is what a
    /// grammar that is *unsatisfied* and stuck returns.
    GrammarComplete,
}

/// Sample the next token id from `logits` under `params`, and advance
/// whatever state that token changes.
///
/// `decode_token` renders one vocabulary entry as text. It is required
/// rather than optional on purpose: the previous shape took an
/// `Option`, and the `None` arm silently fell through to *unconstrained*
/// sampling for a request that had asked to be constrained. Both callers
/// already hold a tokenizer, so there is no arm to fall through to.
///
/// `stop_tokens` is the model's end-of-generation set, which a grammar
/// needs for a reason no other caller has: EOG is the one token that
/// asserts the parse is FINISHED, so it is masked out until the grammar
/// says a parse is complete, and is never offered to the grammar as an
/// ordinary piece of text.
pub(crate) fn sample_next(
    state: &mut SampleState,
    logits: &[f32],
    params: &GenerationParams,
    prompt: &[usize],
    history: &[usize],
    stop_tokens: &StopTokens,
    decode_token: &dyn Fn(usize) -> String,
) -> Result<Step, DecodeError> {
    // Both halves, so `penalty_last_n` slides over `prompt ++
    // generated` exactly as llama-server's does: it seeds its sampler
    // with every prompt token before the first draw
    // (`tools/server/server-context.cpp:386-390`). The prompt used to
    // be `&[]` here because the ids did not reach this seam, which made
    // the HTTP API disagree with `frink run` about what the same flag
    // means (#73).
    let window = PenaltyWindow::new(prompt, history);
    if !params.needs_vocab_logits() {
        let (id, probs) = state
            .sampler
            .sample_reporting(logits, &params.sampling, window, None);
        return Ok(Step::Token {
            id,
            probs: params.wants_logprobs.then_some(probs).flatten(),
        });
    }
    // A backend that folded lm_head+argmax onto the device returns one
    // element holding the chosen id, not a vocabulary, and masking it
    // would zero a token id rather than a logit. That is unreachable:
    // `generate::greedy_gpu_fold_allowed` refuses the fold for exactly
    // the requests that reach this branch. Asserted rather than assumed,
    // because the failure it guards is silent -- plausible-looking
    // unstructured text, served with a 200.
    debug_assert!(
        logits.len() > 1,
        "a request needing vocabulary-shaped logits was handed {} of them; \
         greedy_gpu_fold_allowed and needs_vocab_logits have drifted apart",
        logits.len()
    );

    if state.grammar.is_none() {
        if let Some(grammar) = &params.grammar {
            state.grammar = Some(GrammarSampler::new(
                grammar.as_ref().clone(),
                logits.len(),
                |id| decode_token(id).into_bytes(),
                |id| stop_tokens.contains(id),
            ));
        }
    }
    // Bad words that never met a tokenizer. The same refusal shape as
    // an unresolved reasoning budget below, and for the same reason:
    // running on would serve an UNFILTERED answer to a caller who
    // asked for a filtered one, with a 200.
    if let [first, ..] = params.token_mask.unresolved() {
        return Err(DecodeError::Unsupported(format!(
            "`bad_words` reached the sampler unresolved (first: {first:?}); the decode path \
             did not tokenize them"
        )));
    }
    if state.budget.is_none() {
        match &params.reasoning_budget {
            crate::reasoning_budget::ReasoningBudget::Unrestricted => {}
            crate::reasoning_budget::ReasoningBudget::Armed(plan) => {
                state.budget = Some(ReasoningBudget::new(plan));
            }
            // A number nobody tokenized. The seam that turns it into a
            // plan (`run_generation_emit`) was skipped, and running on
            // would serve an unbounded thought as a bounded one.
            crate::reasoning_budget::ReasoningBudget::Requested(n) => {
                return Err(DecodeError::ReasoningBudget {
                    detail: format!(
                        "a budget of {n} tokens reached the sampler unresolved; the decode \
                         path did not tokenize the reasoning markers"
                    ),
                });
            }
        }
    }

    // Destructured so the sampler can be borrowed mutably while the
    // grammar and the budget are read by the mask closure it is handed.
    let SampleState {
        sampler,
        grammar,
        budget,
    } = state;
    let json_object = params.json_object;
    let grammar_ref = grammar.as_ref();
    let budget_ref = budget.as_ref();
    // The mask cannot return an error through a callback the sampler
    // calls, so it parks one here and the token is discarded below. It
    // must be discarded: a mask that failed left every logit at `-inf`,
    // and the "token" that comes back out of that is an artefact of
    // argmax over negative infinity, not a choice.
    let mut refusal: Option<ConstraintError> = None;
    let mut outcome = MaskOutcome::Allowed;
    let (next, drawn_probs) = {
        let mut mask = |scores: &mut [f32]| {
            // First because llama.cpp applies it first, and NOT
            // because the result depends on it: a bias is finite and
            // every mask below writes `-f32::INFINITY`, so the
            // intersection is the same whichever runs first. That is
            // the same argument the comment below makes for the three
            // masks, and `crate::logit_bias` records the sabotage that
            // proved it.
            params.logit_bias.apply(scores);
            // Order does not matter and must not: no mask here ever
            // clears a `-inf`, so the result is the intersection either
            // way. The budget goes first only because llama.cpp applies
            // it first (`common/sampling.cpp:616-620`), and JSON mode
            // next because it is the cheaper of the other two.
            if let Some(b) = budget_ref {
                b.mask(scores);
            }
            if json_object {
                mask_logits_for_json(scores, decode_token);
            }
            // `allowed_token_ids` / `bad_words`. `history` rather than
            // the window: a bad word's prefix is what THIS completion
            // has generated, and a prompt that happens to end with one
            // was supplied rather than drawn.
            params.token_mask.mask(scores, history);
            if let Some(g) = grammar_ref {
                match g.mask_logits(scores) {
                    Ok(o) => outcome = o,
                    Err(e) => refusal = Some(e),
                }
            }
        };
        sampler.sample_reporting(logits, &params.sampling, window, Some(&mut mask))
    };
    if let Some(e) = refusal {
        return Err(DecodeError::GrammarConstraint {
            detail: e.to_string(),
        });
    }
    if outcome == MaskOutcome::Complete {
        // Every logit was `-inf`, so `next` is an artefact of argmax
        // over negative infinity rather than a token anything chose.
        return Ok(Step::GrammarComplete);
    }

    // The other half of the hook. Done HERE, not in the decode loops,
    // because a loop that can forget it is a loop that will: the mask
    // would then keep answering the question "what may the FIRST token
    // be?" for every token in the answer.
    //
    // Including for a token the loop is about to throw away as a stop:
    // every such token ends the generation, so a parse advanced one step
    // past its end is a parse nothing asks another question of.
    if let Some(g) = grammar.as_mut() {
        g.accept(next).map_err(|e| DecodeError::GrammarConstraint {
            detail: e.to_string(),
        })?;
    }
    // And the budget's: every generated token, the forced ones
    // included, walks the machine (`common/sampling.cpp:482-483`
    // accepts each into `rbudget` the same way). The piece decides
    // whether the closer may be forced yet, or must wait for the byte
    // that finishes a character.
    if let Some(b) = budget.as_mut() {
        b.accept(next, lossy_piece_is_complete(&decode_token(next)));
    }
    Ok(Step::Token {
        id: next,
        // The distribution AFTER every mask, which is the one the token
        // came from: a grammar-constrained request's logprobs describe
        // the candidates the grammar allowed, not the ones the model
        // would have had without it.
        probs: params.wants_logprobs.then_some(drawn_probs).flatten(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_models::grammar::Grammar;
    use frink_models::sampling::SamplingParams;
    use std::sync::Arc;

    /// Token 0 renders as something no JSON document may contain;
    /// token 1 is a brace. Any masked sampler must refuse 0.
    fn decode_token(id: usize) -> String {
        match id {
            0 => "<html>".to_string(),
            1 => "{".to_string(),
            _ => "0".to_string(),
        }
    }

    /// **The flag is what decides whether the distribution comes
    /// back**, on BOTH paths through `sample_next` -- the plain one
    /// and the masked one.
    ///
    /// This is the test the end-to-end HTTP one cannot be: the test
    /// server serves synthetic weights, and the synthetic banner
    /// clears the distributions with the text it replaces, so an
    /// HTTP-level assertion takes the empty branch and would pass with
    /// the sampler reporting nothing at all. Confirmed by sabotage:
    /// hard-coding `probs: None` left the HTTP test green.
    #[test]
    fn wants_logprobs_decides_whether_the_distribution_is_reported() {
        let logits = vec![0.5f32, 2.0, 0.25, 1.0];
        let stop = StopTokens::default();

        for (json_object, path) in [(false, "plain"), (true, "masked")] {
            for (wants, expect_some) in [(false, false), (true, true)] {
                let mut p = params(json_object, 0.8);
                p.wants_logprobs = wants;
                let mut state = SampleState::new(p.seed);
                let step = sample_next(
                    &mut state,
                    &logits,
                    &p,
                    &[],
                    &[],
                    &stop,
                    &(decode_token as fn(usize) -> String),
                )
                .expect("the sampler answers");
                let Step::Token { probs, .. } = step else {
                    panic!("{path}: expected a token");
                };
                assert_eq!(
                    probs.is_some(),
                    expect_some,
                    "{path} path, wants_logprobs = {wants}"
                );
                if let Some(probs) = probs {
                    assert_eq!(probs.len(), logits.len(), "{path}: one entry per token");
                    let total: f32 = probs.iter().sum();
                    assert!((total - 1.0).abs() < 1e-4, "{path}: not normalised");
                }
            }
        }
    }

    fn params(json_object: bool, temperature: f32) -> GenerationParams {
        GenerationParams {
            cache_salt: None,
            prompt_logprobs: None,
            wants_logprobs: false,
            n: 1,
            interleave_choices: false,
            logit_bias: crate::logit_bias::LogitBias::default(),
            keep_special_tokens: false,
            truncate_prompt_tokens: None,
            token_mask: crate::token_mask::TokenMask::default(),
            reasoning: None,
            max_tokens: 8,
            sampling: SamplingParams {
                temperature,
                ..SamplingParams::default()
            },
            seed: 7,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            json_object,
            grammar: None,
            cancel: None,
            ignore_eos: false,
            reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
            lora: None,
        }
    }

    /// A vocabulary of single letters plus an end-of-generation token,
    /// so a grammar over letters has something to be right about.
    fn letter(id: usize) -> String {
        match id {
            0 => "a".to_string(),
            1 => "b".to_string(),
            2 => "c".to_string(),
            3 => "d".to_string(),
            _ => "</s>".to_string(),
        }
    }
    const LETTER_EOG: usize = 4;

    fn letter_stops() -> StopTokens {
        StopTokens::from_eos(Some(LETTER_EOG))
    }

    fn with_grammar(mut p: GenerationParams, src: &str) -> GenerationParams {
        p.grammar = Some(Arc::new(
            Grammar::from_str_with_root(src, "root").expect("test grammar parses"),
        ));
        p
    }

    /// The id a step produced. A test that asks for a token and is told
    /// the grammar is finished has found a different bug than the one
    /// it was written for, so it says which.
    fn token(step: Step) -> usize {
        match step {
            Step::Token { id, .. } => id,
            Step::GrammarComplete => {
                panic!("the grammar ended the generation where a token was expected")
            }
        }
    }

    /// One step, with the arguments every test here shares.
    fn step(
        state: &mut SampleState,
        logits: &[f32],
        params: &GenerationParams,
        history: &[usize],
        stops: &StopTokens,
        decode: &dyn Fn(usize) -> String,
    ) -> Result<Step, DecodeError> {
        // These tests are about masking and stops, not penalties, so
        // the prompt half is empty and says so.
        sample_next(state, logits, params, &[], history, stops, decode)
    }

    /// The assertion neither decode path had. Token 0 wins the argmax by
    /// a mile and is not JSON-safe; a masked greedy sample must not
    /// return it. `temperature: 0` because that is how callers actually
    /// ask for structured output, and it is the case the Metal greedy
    /// fold used to break.
    #[test]
    fn json_mode_masks_at_temperature_zero() {
        let logits = vec![9.0, 1.0, 0.5];
        let mut state = SampleState::new(7);
        let chosen = token(
            step(
                &mut state,
                &logits,
                &params(true, 0.0),
                &[],
                &StopTokens::default(),
                &decode_token,
            )
            .expect("json mode has a legal token here"),
        );
        assert_ne!(chosen, 0, "a non-JSON-safe token was sampled in json mode");
        assert_eq!(chosen, 1);
    }

    /// Same logits, sampled rather than greedy: the mask is not a
    /// greedy-only path.
    #[test]
    fn json_mode_masks_when_sampling_too() {
        let logits = vec![9.0, 1.0, 0.5];
        for seed in 0..16u64 {
            let mut state = SampleState::new(seed);
            let chosen = token(
                step(
                    &mut state,
                    &logits,
                    &params(true, 1.0),
                    &[],
                    &StopTokens::default(),
                    &decode_token,
                )
                .expect("json mode has a legal token here"),
            );
            assert_ne!(chosen, 0, "seed {seed} sampled a non-JSON-safe token");
        }
    }

    /// And the mask is not applied to a request that never asked for it:
    /// the unconstrained argmax is token 0.
    #[test]
    fn a_plain_request_is_not_masked() {
        let logits = vec![9.0, 1.0, 0.5];
        let mut state = SampleState::new(7);
        let chosen = token(
            step(
                &mut state,
                &logits,
                &params(false, 0.0),
                &[],
                &StopTokens::default(),
                &decode_token,
            )
            .expect("an unconstrained request cannot fail"),
        );
        assert_eq!(chosen, 0);
    }

    /// The grammar's mask half: "b" wins the argmax by a mile, and
    /// `root ::= "ac"` forbids it.
    #[test]
    fn a_grammar_masks_the_argmax_it_forbids() {
        let logits = vec![1.0, 9.0, 0.5, 0.1, 0.0];
        let mut state = SampleState::new(7);
        let chosen = token(
            step(
                &mut state,
                &logits,
                &with_grammar(params(false, 0.0), r#"root ::= "ac""#),
                &[],
                &letter_stops(),
                &letter,
            )
            .expect("\"a\" is legal here"),
        );
        assert_eq!(chosen, 0, "the grammar's only legal first token is \"a\"");
        assert!(state.grammar_started());
    }

    /// The grammar's accept half, which is the one a decode loop can
    /// drop without noticing: the second step must be constrained by
    /// what the FIRST step chose. Without the accept, "a" is the only
    /// legal token forever and this returns 0 again.
    #[test]
    fn a_grammar_advances_between_steps() {
        let logits = vec![1.0, 9.0, 0.5, 0.1, 0.0];
        let params = with_grammar(params(false, 0.0), r#"root ::= "ac""#);
        let mut state = SampleState::new(7);
        let first = token(
            step(&mut state, &logits, &params, &[], &letter_stops(), &letter)
                .expect("\"a\" is legal here"),
        );
        let second = token(
            step(
                &mut state,
                &logits,
                &params,
                &[first],
                &letter_stops(),
                &letter,
            )
            .expect("\"c\" is legal here"),
        );
        assert_eq!(
            second, 2,
            "the grammar did not advance past its first token"
        );
    }

    /// End-of-generation is masked until the grammar is satisfied, and
    /// allowed the moment it is. Token 4 wins the argmax throughout, so
    /// an unconstrained sampler would end the answer immediately.
    #[test]
    fn a_grammar_holds_end_of_generation_back_until_it_is_satisfied() {
        let logits = vec![1.0, 0.5, 0.4, 0.1, 9.0];
        let params = with_grammar(params(false, 0.0), r#"root ::= "a""#);
        let mut state = SampleState::new(7);
        let first = token(
            step(&mut state, &logits, &params, &[], &letter_stops(), &letter)
                .expect("\"a\" is legal here"),
        );
        assert_eq!(first, 0, "generation was allowed to end unsatisfied");
        let second = token(
            step(
                &mut state,
                &logits,
                &params,
                &[first],
                &letter_stops(),
                &letter,
            )
            .expect("eog is legal once the parse is complete"),
        );
        assert_eq!(second, LETTER_EOG);
    }

    /// A satisfied grammar in a vocabulary with no end-of-generation
    /// token has finished the answer, and says so rather than failing.
    #[test]
    fn a_finished_grammar_with_nothing_left_to_say_ends_the_generation() {
        let logits = vec![1.0, 9.0, 0.5, 0.1, 0.0];
        let params = with_grammar(params(false, 0.0), r#"root ::= "a""#);
        let mut state = SampleState::new(7);
        let first = token(
            step(
                &mut state,
                &logits,
                &params,
                &[],
                &StopTokens::default(),
                &letter,
            )
            .expect("\"a\" is legal here"),
        );
        assert_eq!(first, 0);
        assert_eq!(
            step(
                &mut state,
                &logits,
                &params,
                &[first],
                &StopTokens::default(),
                &letter,
            )
            .expect("a completed parse is not an error"),
            Step::GrammarComplete
        );
    }

    /// A grammar this vocabulary cannot spell is a typed refusal, not a
    /// token sampled out of an all-`-inf` distribution.
    #[test]
    fn a_grammar_no_token_can_satisfy_stops_the_generation() {
        let logits = vec![1.0, 9.0, 0.5, 0.1, 0.0];
        let mut state = SampleState::new(7);
        let err = step(
            &mut state,
            &logits,
            &with_grammar(params(false, 0.0), r#"root ::= "zz""#),
            &[],
            &letter_stops(),
            &letter,
        )
        .expect_err("no token in this vocabulary starts with z");
        assert!(
            matches!(err, DecodeError::GrammarConstraint { .. }),
            "{err}"
        );
    }

    /// The two masks intersect rather than replace each other. Piece 0
    /// is `<html>`, which json mode forbids; piece 1 is `{`, which the
    /// grammar below forbids; piece 2 is `0`, allowed by both.
    #[test]
    fn a_grammar_and_json_mode_intersect_rather_than_replace_each_other() {
        let logits = vec![9.0, 8.0, 0.5];
        let mut state = SampleState::new(7);
        let chosen = token(
            step(
                &mut state,
                &logits,
                &with_grammar(params(true, 0.0), r#"root ::= [0-9]+"#),
                &[],
                &StopTokens::default(),
                &decode_token,
            )
            .expect("\"0\" is legal under both"),
        );
        assert_eq!(chosen, 2);
    }

    /// The coupling that keeps [`sample_next`]'s `debug_assert` true,
    /// checked in the default build because the fold it constrains is
    /// `#[cfg(feature = "metal")]`.
    ///
    /// `temperature <= 0` alone used to permit the fold. It must not: a
    /// JSON-mode request at temperature 0 would then be handed a
    /// one-element vector and masked into nothing. A grammar request is
    /// the same statement and is checked beside it, because a folded
    /// grammar request is unconstrained text served as constrained.
    #[test]
    fn a_constrained_request_may_never_fold_lm_head_into_a_gpu_argmax() {
        use crate::generate::greedy_gpu_fold_allowed;

        assert!(greedy_gpu_fold_allowed(&params(false, 0.0)));
        assert!(!greedy_gpu_fold_allowed(&params(true, 0.0)));
        assert!(!greedy_gpu_fold_allowed(&with_grammar(
            params(false, 0.0),
            r#"root ::= "a""#
        )));
        // Not greedy: never folded either way, whatever the constraints.
        assert!(!greedy_gpu_fold_allowed(&params(false, 0.8)));
        assert!(!greedy_gpu_fold_allowed(&params(true, 0.8)));
        // A LAZY grammar is the case that looks unconstrained and is
        // not: its trigger can fire on any token, so it needs the whole
        // vocabulary from the first one.
        assert!(!greedy_gpu_fold_allowed(&with_lazy_grammar(
            params(false, 0.0),
            r#"root ::= "cd""#,
            "c",
        )));

        // And the third thing that needs the vocabulary: a SAMPLER that
        // can move the argmax. `xtc` removes the top candidates, `typ_p`
        // can drop the most likely one, and `dry` changes which logit is
        // the maximum -- so a folded argmax at temperature 0 would be the
        // answer to a chain the caller did not configure.
        for live in [
            frink_models::SamplingParams {
                xtc_probability: 1.0,
                xtc_threshold: 0.1,
                ..frink_models::SamplingParams::default()
            },
            frink_models::SamplingParams {
                typical_p: 0.5,
                ..frink_models::SamplingParams::default()
            },
        ] {
            let mut p = params(false, 0.0);
            p.sampling = live;
            assert!(
                !greedy_gpu_fold_allowed(&p),
                "a chain that can move the argmax must stop the fold: {:?}",
                p.sampling
            );
        }

        // And the FOURTH: the repetition / presence / frequency
        // penalties (GitHub issue #170). They are applied to the whole
        // vocabulary on the HOST, so a device argmax over raw logits
        // skips them and answers a chain the caller did not configure.
        //
        // This route defaults them off, so the bug was only reachable
        // here for a client that sent one; `frink run` defaults
        // `--repeat-penalty` to 1.1 and had it live on every greedy
        // `--ngl 99` run.
        //
        // Sabotage: revert `needs_vocab_logits` to the old
        // `greedy_equals_argmax`; all three rows go red.
        for penalised in [
            frink_models::SamplingParams {
                repetition_penalty: 1.1,
                ..frink_models::SamplingParams::default()
            },
            frink_models::SamplingParams {
                presence_penalty: 0.5,
                ..frink_models::SamplingParams::default()
            },
            frink_models::SamplingParams {
                frequency_penalty: 0.5,
                ..frink_models::SamplingParams::default()
            },
        ] {
            let mut p = params(false, 0.0);
            p.sampling = penalised;
            assert!(
                !greedy_gpu_fold_allowed(&p),
                "a live penalty must stop the fold: {:?}",
                p.sampling
            );
        }
        // A penalty with no window is no penalty, so the fold stands.
        let mut windowless = params(false, 0.0);
        windowless.sampling = frink_models::SamplingParams {
            repetition_penalty: 1.1,
            penalty_last_n: 0,
            ..frink_models::SamplingParams::default()
        };
        assert!(greedy_gpu_fold_allowed(&windowless));
    }

    /// A grammar that waits for a trigger, with the trigger MANDATORY --
    /// what `tool_choice: "required"` compiles to.
    fn with_lazy_grammar(mut p: GenerationParams, src: &str, trigger: &str) -> GenerationParams {
        use frink_models::grammar::LazyTriggers;
        p.grammar = Some(Arc::new(
            Grammar::from_str_with_root(src, "root")
                .expect("test grammar parses")
                .into_lazy(
                    LazyTriggers::new()
                        .with_word(trigger)
                        .expect("the trigger compiles")
                        .mandatory(),
                )
                .expect("there is a trigger"),
        ));
        p
    }

    /// The `tool_choice: "required"` shape, on the one step both decode
    /// loops and `generate_engine` share.
    ///
    /// Three things at once, because they only mean anything together:
    /// free text before the trigger, an ENDING that is forbidden until
    /// the trigger fires, and a grammar that constrains hard once it
    /// does.
    #[test]
    fn a_mandatory_lazy_grammar_forbids_only_the_ending_until_it_fires() {
        let params = with_lazy_grammar(params(false, 0.0), r#"root ::= "cd""#, "c");
        let stops = letter_stops();
        let mut state = SampleState::new(7);

        // End-of-generation wins the argmax by a mile, and "a" -- which
        // the grammar could never accept -- is the best of the rest.
        let logits = vec![5.0, 1.0, 0.5, 0.5, 9.0];
        let chosen = token(step(&mut state, &logits, &params, &[], &stops, &letter).unwrap());
        assert_eq!(
            chosen, 0,
            "free text before the trigger, but not the end of the turn"
        );

        // "c" is the trigger. After it the grammar is live and wants "d".
        let logits = vec![1.0, 1.0, 9.0, 0.5, 5.0];
        let chosen = token(step(&mut state, &logits, &params, &[0], &stops, &letter).unwrap());
        assert_eq!(chosen, 2, "the trigger token itself");

        let logits = vec![9.0, 9.0, 9.0, 0.0, 9.0];
        let chosen = token(step(&mut state, &logits, &params, &[0, 2], &stops, &letter).unwrap());
        assert_eq!(
            chosen, 3,
            "once triggered the grammar constrains: only \"d\" continues \"c\""
        );
    }

    /// The same grammar WITHOUT `mandatory` lets the turn end. The pair
    /// is what shows the first test is measuring the flag and not the
    /// mask in general.
    #[test]
    fn an_optional_lazy_grammar_lets_the_turn_end_before_the_trigger() {
        use frink_models::grammar::LazyTriggers;
        let mut params = params(false, 0.0);
        params.grammar = Some(Arc::new(
            Grammar::from_str_with_root(r#"root ::= "cd""#, "root")
                .unwrap()
                .into_lazy(LazyTriggers::new().with_word("c").unwrap())
                .unwrap(),
        ));
        let logits = vec![5.0, 1.0, 0.5, 0.5, 9.0];
        let mut state = SampleState::new(7);
        let chosen =
            token(step(&mut state, &logits, &params, &[], &letter_stops(), &letter).unwrap());
        assert_eq!(chosen, LETTER_EOG, "an optional trigger constrains nothing");
    }

    /// **The server must penalise the prompt, as llama-server does.**
    ///
    /// llama-server seeds its sampler with every prompt token before
    /// the first draw (`tools/server/server-context.cpp:386-390`), so
    /// `penalty_last_n` slides over `prompt ++ generated` there. This
    /// seam passed `&[]` until #73, because the prompt ids did not
    /// reach it, so the SAME request body produced different text from
    /// llama-server, and from `frink run`, which was fixed first.
    ///
    /// Asserted on the sampled token. A test on the window's length
    /// would pass with the window handed to the wrong distribution.
    #[test]
    fn a_prompt_token_is_penalised_by_the_server_before_it_is_generated() {
        // Token 0 wins the argmax outright. A penalty heavy enough to
        // dethrone it can only come from the PROMPT, since nothing has
        // been generated yet.
        let logits = vec![9.0, 8.0, 0.5];
        let mut p = params(false, 0.0);
        p.sampling.repetition_penalty = 50.0;
        p.sampling.penalty_last_n = 64;

        let mut state = SampleState::new(7);
        let with_prompt = sample_next(&mut state, &logits, &p, &[0], &[], &letter_stops(), &|id| {
            letter(id)
        })
        .expect("sampling succeeds");

        let mut state = SampleState::new(7);
        let without_prompt =
            sample_next(&mut state, &logits, &p, &[], &[], &letter_stops(), &|id| {
                letter(id)
            })
            .expect("sampling succeeds");

        assert_eq!(
            token(without_prompt),
            0,
            "with no prompt the argmax stands, which is what makes the other half meaningful"
        );
        assert_eq!(
            token(with_prompt),
            1,
            "a token in the prompt must be penalised on its FIRST generated occurrence"
        );
    }

    /// A five-token vocabulary with an opener and a closer, for the
    /// budget tests: 0..=2 are words, 3 opens the block, 4 closes it.
    fn think_vocab(id: usize) -> String {
        match id {
            3 => "<think>".to_string(),
            4 => "</think>".to_string(),
            other => format!("w{other}"),
        }
    }
    const OPEN: usize = 3;
    const CLOSE: usize = 4;

    fn with_budget(mut p: GenerationParams, budget: u32, prefill: &[usize]) -> GenerationParams {
        use frink_models::sampling::reasoning_budget::ReasoningBudgetPlan;
        p.reasoning_budget =
            crate::reasoning_budget::ReasoningBudget::Armed(Arc::new(ReasoningBudgetPlan {
                start_seqs: vec![vec![OPEN]],
                end_seqs: vec![vec![CLOSE]],
                forced: vec![CLOSE],
                budget,
                prefill: prefill.to_vec(),
            }));
        p
    }

    /// The whole hook under a budget, at temperature 0 with the closer
    /// the LEAST likely token every step: the model opens the block,
    /// spends its budget on its own argmax, and the next draw is the
    /// closer whatever the logits say. Then the mask is gone and the
    /// argmax is back. This is the property both decode loops get by
    /// calling here: a loop that kept its own `Sampler` would think
    /// forever.
    #[test]
    fn the_budget_forces_the_closer_after_n_tokens_of_thought_on_both_loops_path() {
        use frink_models::sampling::reasoning_budget::BudgetState;
        // Argmax is the opener first, then word 1 forever; the closer
        // is always last.
        let opener_first = vec![1.0, 5.0, 0.5, 9.0, -9.0];
        let words = vec![1.0, 5.0, 0.5, 0.2, -9.0];
        let p = with_budget(params(false, 0.0), 2, &[]);
        let mut state = SampleState::new(7);
        let stops = StopTokens::default();
        let mut out = Vec::new();
        let mut draw = |state: &mut SampleState, logits: &[f32]| {
            let id = token(step(state, logits, &p, &out, &stops, &think_vocab).expect("draws"));
            out.push(id);
            id
        };
        assert_eq!(draw(&mut state, &opener_first), OPEN);
        assert_eq!(state.budget_state(), Some(BudgetState::Counting));
        assert_eq!(draw(&mut state, &words), 1, "first of two");
        assert_eq!(
            draw(&mut state, &words),
            1,
            "second of two, still the model's own"
        );
        assert_eq!(state.budget_state(), Some(BudgetState::Forcing));
        assert_eq!(
            draw(&mut state, &words),
            CLOSE,
            "the least likely token, forced over the argmax"
        );
        assert_eq!(state.budget_state(), Some(BudgetState::Done));
        assert_eq!(draw(&mut state, &words), 1, "the answer is unconstrained");
    }

    /// Sampled rather than greedy, and the budget still lands: the
    /// mask leaves one finite logit, so the draw has one candidate.
    #[test]
    fn the_forced_closer_survives_sampling_at_temperature_one() {
        let words = vec![1.0, 5.0, 0.5, 0.2, -9.0];
        for seed in 0..8u64 {
            let p = with_budget(params(false, 1.0), 0, &[OPEN]);
            let mut state = SampleState::new(seed);
            let chosen = token(
                step(
                    &mut state,
                    &words,
                    &p,
                    &[],
                    &StopTokens::default(),
                    &think_vocab,
                )
                .expect("draws"),
            );
            assert_eq!(chosen, CLOSE, "seed {seed}");
        }
    }

    /// A budget nobody tokenized is a refusal at the sampler, not a
    /// thought served unbounded: the seam that arms it was skipped.
    #[test]
    fn a_requested_budget_that_was_never_armed_is_an_error_not_a_passthrough() {
        let mut p = params(false, 0.0);
        p.reasoning_budget = crate::reasoning_budget::ReasoningBudget::Requested(5);
        let mut state = SampleState::new(7);
        let err = step(
            &mut state,
            &[1.0, 5.0, 0.5, 0.2, -9.0],
            &p,
            &[],
            &StopTokens::default(),
            &think_vocab,
        )
        .expect_err("unresolved");
        assert!(matches!(err, DecodeError::ReasoningBudget { .. }), "{err}");
    }
}
