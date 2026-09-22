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

/// One `(token id, distribution it was drawn from)` per KEPT token.
///
/// The id travels with the distribution so a renderer cannot line the
/// two up wrongly: a stop token is dropped from the answer, and an
/// index into `generated_ids` would then be off by one for every
/// position after it.
pub(crate) type PerTokenProbs = Vec<(usize, Vec<f32>)>;

/// Fallible since constrained decoding landed: a grammar that cannot be
/// continued ends the generation with an error rather than with an
/// answer, because the alternative is to emit a token the caller's
/// grammar forbids and report it as constrained output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_until_stop(
    logits: Vec<f32>,
    pos: usize,
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
    engine: &mut dyn DecodeEngine,
    mut emit: impl FnMut(&str),
    decode_token: &dyn Fn(usize) -> String,
    // Tokens a round may draft. Zero is the ordinary path, and so is
    // an engine whose `draft` returns nothing.
    draft_max: usize,
    // The finish reason, the generated ids, the final logits, and the
    // per-token distributions when the request asked for them --
    // EMPTY otherwise, because a request that did not ask does not pay
    // for the vector.
) -> Result<(FinishReason, Vec<usize>, Vec<f32>, PerTokenProbs), DecodeError> {
    let ctx = crate::choice_stream::StepContext {
        params,
        prompt_ids,
        stop_tokens,
        decode_token,
    };
    let mut choice = crate::choice_stream::ChoiceStream::new(logits, pos, params);

    while choice.check_budget(params) {
        // A speculative round commits a BLOCK: the drafts that agreed
        // with the sampler, plus one token that did not (or the bonus
        // one, when every draft agreed). Every token still goes
        // through `ChoiceStream::commit` in order, so the stop rules
        // cannot differ between the two paths.
        //
        // The rows for accepted drafts are already in the store; the
        // LAST committed token has not been fed, exactly as in the
        // ordinary path, so the tail of this branch feeds it.
        if draft_max > 0
            && speculate(
                &mut choice,
                &ctx,
                engine,
                &mut decode_one,
                &mut emit,
                draft_max,
            )?
        {
            continue;
        }
        choice.step(&ctx, engine, &mut decode_one, &mut emit)?;
    }

    choice.flush(&mut emit);
    Ok(choice.into_parts())
}

/// One speculative round, or `false` when there was nothing to draft
/// and the caller should take the ordinary step.
///
/// Split out when `ChoiceStream` did, because the round-robin
/// scheduler does NOT draft: a block commits several tokens at once,
/// and a client reading `choices[].index` asked for the choices
/// interleaved rather than in bursts. Keeping it here rather than on
/// the stream is what says which of the two schedules it belongs to.
fn speculate(
    choice: &mut crate::choice_stream::ChoiceStream,
    ctx: &crate::choice_stream::StepContext<'_>,
    engine: &mut dyn DecodeEngine,
    decode_one: &mut impl FnMut(&[usize]) -> Vec<u8>,
    emit: &mut impl FnMut(&str),
    draft_max: usize,
) -> Result<bool, DecodeError> {
    // The budget is per TOKEN and the caller's loop counts iterations,
    // which used to be the same thing. A block commits its accepted
    // drafts plus one, so a round may only draft `remaining - 1`:
    // without this a `max_tokens` of 6 with a 3-token drafter returned
    // more than six tokens, and the test that compares speculative
    // output against plain output caught it.
    let remaining = ctx
        .params
        .max_tokens
        .saturating_sub(choice.generated_ids.len());
    let room = remaining.saturating_sub(1);
    if room == 0 {
        return Ok(false);
    }
    let draft = engine.draft(ctx.prompt_ids, &choice.generated_ids, draft_max.min(room));
    if draft.is_empty() {
        return Ok(false);
    }
    let Some(rows) = engine.batch(&draft, choice.pos) else {
        return Ok(false);
    };
    let (state, logits, mut history) = choice.verify_inputs();
    let block = verify_block(
        state,
        logits,
        &rows,
        &draft,
        ctx.params,
        ctx.prompt_ids,
        &mut history,
        ctx.stop_tokens,
        ctx.decode_token,
    )?;
    engine.observe(block.accepted, block.drafted);
    // Rejected drafts wrote rows that describe a prefix that never
    // happened.
    if block.accepted < draft.len() {
        engine.truncate(choice.pos + block.accepted);
    }
    choice.pos += block.accepted;

    let mut stopped = None;
    let mut last: Option<usize> = None;
    for (i, &t) in block.tokens.iter().enumerate() {
        match choice.commit(t, ctx, decode_one, emit) {
            crate::choice_stream::Committed::Continue => last = Some(t),
            crate::choice_stream::Committed::Stopped(reason) => {
                // The tokens after this one are not part of the
                // answer, and neither are their rows.
                let kept = choice.pos - block.accepted + i.min(block.accepted);
                engine.truncate(kept);
                choice.pos = kept;
                stopped = Some(reason);
                break;
            }
        }
    }
    if let Some(reason) = stopped {
        choice.stop(reason);
        return Ok(true);
    }
    if block.grammar_complete {
        choice.stop(FinishReason::Stop);
        return Ok(true);
    }
    // Only the final committed token still needs feeding; the accepted
    // drafts already have rows.
    if let Some(t) = last {
        choice.logits = engine.step(t, choice.pos);
        choice.pos += 1;
    }
    Ok(true)
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

/// Tokens a prompt-lookup round drafts.
///
/// llama.cpp's `--draft-max` default is 16 and its drafter is a second
/// MODEL; a prompt-lookup drafter is right far less often, and every
/// rejected draft is a position the target paid for and threw away.
/// Five is `frink run`'s default for the same drafter, so the two
/// front ends agree.
pub(crate) const DEFAULT_DRAFT_MAX: usize = 5;

/// N-gram length the prompt-lookup drafter matches on.
///
/// Shorter matches more often and is wrong more often. Two is
/// `frink run`'s.
pub(crate) const DRAFT_NGRAM: usize = 2;

/// Everything the loop asks of the engine, including speculation.
///
/// `step` lives here rather than staying a closure for a reason found
/// by trying the other way: a speculative round needs `batch` and
/// `truncate` on the SAME KV store that `step` writes to, and a
/// closure capturing `&mut Kv` holds that borrow for its whole life,
/// so the caller could not hand out both. One object owns the store
/// and answers every question about it.
///
/// Speculation is opt-in through defaults: an engine that only
/// implements `step` drafts nothing and takes the ordinary path, which
/// is what every caller did before this trait existed. A blanket impl
/// keeps plain closures working for the tests that only need a step.
pub(crate) trait DecodeEngine {
    /// Feed one token at `pos`, returning the next position's logits.
    fn step(&mut self, token: usize, pos: usize) -> Vec<f32>;

    /// Up to `max` tokens continuing `history`, or empty for none.
    fn draft(&mut self, _prompt: &[usize], _history: &[usize], _max: usize) -> Vec<usize> {
        Vec::new()
    }

    /// Feed `tokens` at `pos`, one logit row per token.
    ///
    /// `None` means this store must not be speculated into at all --
    /// see `Kv::step_batch`, which asks whether it can roll back
    /// BEFORE it writes.
    fn batch(&mut self, _tokens: &[usize], _pos: usize) -> Option<Vec<Vec<f32>>> {
        None
    }

    /// Drop every row past `pos`, undoing rejected drafts.
    fn truncate(&mut self, _pos: usize) {}

    /// Accepted and drafted, for the acceptance metric.
    fn observe(&mut self, _accepted: usize, _drafted: usize) {}
}

/// A plain step closure is an engine that never speculates, so the
/// callers that only have a forward keep working unchanged.
impl<F: FnMut(usize, usize) -> Vec<f32>> DecodeEngine for F {
    fn step(&mut self, token: usize, pos: usize) -> Vec<f32> {
        self(token, pos)
    }
}

/// What one verification round committed.
///
/// UNWIRED, deliberately, and this says where it is going rather than
/// leaving a reader to guess: the rule is one row and the wiring is
/// the next. What still has to land for a request to reach it is
/// listed in `docs/plans/server-speculative-decoding.md` -- a drafter
/// chosen per request, the KV truncated to `accepted` rows when a
/// draft is rejected, and `stats::requests::with_speculation` finally
/// given the producer it has never had. Landing the rule first is what
/// lets that wiring be reviewed against a tested definition instead of
/// alongside one.
///
/// `accepted` is the number of DRAFTED tokens that survived, which is
/// also how many KV rows the caller keeps: the drafts past that point
/// were fed to the model and their rows are now wrong, and the
/// corrective token has no row yet because it was never fed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    dead_code,
    reason = "wired by the server speculative-decoding row; see above"
)]
pub(crate) struct VerifiedBlock {
    /// Committed tokens, in order. Always at least one unless the
    /// grammar completed immediately.
    pub tokens: Vec<usize>,
    /// Drafted tokens that matched what the sampler drew.
    pub accepted: usize,
    /// Drafted tokens offered this round.
    pub drafted: usize,
    /// The grammar's parse completed inside the block; nothing may
    /// follow, and the caller stops.
    pub grammar_complete: bool,
}

/// Verify a block of drafted tokens by AGREEMENT with the sampler.
///
/// The rule, decided in `docs/plans/server-speculative-decoding.md`:
/// draw with [`crate::sample_step::sample_next`] at every position --
/// the same sampler, state, penalty window and grammar machine the
/// non-speculative loop uses -- and accept a drafted token if and only
/// if it EQUALS that draw. The token emitted is always the sampler's,
/// never the draft's, so a drafter can only ever save a forward pass
/// and can never change an answer.
///
/// That is what makes this lossless by construction rather than by
/// proof. There is no `p(x)`/`q(x)` bookkeeping that could be subtly
/// wrong, and no rollback: state advances only over committed tokens,
/// in order, and the walk stops at the first disagreement, so the
/// grammar machine never has to be checkpointed.
///
/// # Logits
///
/// `first_logits` is the row the caller already holds for the position
/// before `draft[0]`. `draft_logits[i]` is what the model produced
/// after being fed `draft[i]`, so it is only MEANINGFUL while every
/// draft up to and including `i` was accepted -- which is exactly why
/// the walk stops at the first mismatch rather than scoring the rest.
///
/// When every draft is accepted there is one extra row left over, and
/// the token drawn from it is free: that is the round's profit.
#[allow(clippy::too_many_arguments)]
#[allow(
    dead_code,
    reason = "wired by the server speculative-decoding row; see VerifiedBlock"
)]
pub(crate) fn verify_block(
    state: &mut crate::sample_step::SampleState,
    first_logits: &[f32],
    draft_logits: &[Vec<f32>],
    draft: &[usize],
    params: &GenerationParams,
    prompt: &[usize],
    history: &mut Vec<usize>,
    stop_tokens: &StopTokens,
    decode_token: &dyn Fn(usize) -> String,
) -> Result<VerifiedBlock, DecodeError> {
    debug_assert_eq!(
        draft.len(),
        draft_logits.len(),
        "one logit row per drafted token, or the rows and the drafts have drifted"
    );

    let mut out = VerifiedBlock {
        tokens: Vec::new(),
        accepted: 0,
        drafted: draft.len(),
        grammar_complete: false,
    };

    for i in 0..=draft.len() {
        let logits = if i == 0 {
            first_logits
        } else {
            &draft_logits[i - 1]
        };
        match crate::sample_step::sample_next(
            state,
            logits,
            params,
            prompt,
            history,
            stop_tokens,
            decode_token,
        )? {
            crate::sample_step::Step::GrammarComplete => {
                out.grammar_complete = true;
                return Ok(out);
            }
            crate::sample_step::Step::Token { id: t, .. } => {
                history.push(t);
                out.tokens.push(t);
                // The last iteration has no draft to compare against:
                // it is the bonus token every draft being right earns.
                if i == draft.len() {
                    break;
                }
                if t == draft[i] {
                    out.accepted += 1;
                } else {
                    break;
                }
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::sample_step::{sample_next, SampleState, Step};
    use frink_models::sampling::SamplingParams;

    fn greedy_params() -> GenerationParams {
        GenerationParams {
            cache_salt: None,
            prompt_logprobs: None,
            wants_logprobs: false,
            n: 1,
            interleave_choices: false,
            reasoning: None,
            max_tokens: 64,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            seed: 1,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            json_object: false,
            grammar: None,
            cancel: None,
            ignore_eos: false,
            reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
            lora: None,
        }
    }

    /// One-hot logits, so the greedy draw at a row is known by
    /// construction and the test is about the WALK, not the sampler.
    fn peaked(vocab: usize, winner: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab];
        v[winner] = 10.0;
        v
    }

    fn no_stops() -> StopTokens {
        StopTokens::default()
    }

    fn decode(_: usize) -> String {
        String::new()
    }

    /// Every draft right: `k` accepted plus the bonus token the last
    /// logit row pays out, which is the entire point of the round.
    #[test]
    fn all_drafts_accepted_earn_one_extra_token() {
        let params = greedy_params();
        let mut state = SampleState::new(params.seed);
        let mut history = Vec::new();
        let draft = [3usize, 4, 5];
        let first = peaked(8, 3);
        let rows = vec![peaked(8, 4), peaked(8, 5), peaked(8, 6)];

        let b = verify_block(
            &mut state,
            &first,
            &rows,
            &draft,
            &params,
            &[],
            &mut history,
            &no_stops(),
            &decode,
        )
        .expect("verify");

        assert_eq!(b.accepted, 3);
        assert_eq!(b.drafted, 3);
        assert_eq!(b.tokens, vec![3, 4, 5, 6], "three drafts plus the bonus");
        assert_eq!(history, vec![3, 4, 5, 6]);
    }

    /// A wrong draft ends the round THERE, and the token committed at
    /// that position is the sampler's, not the draft's. The rows after
    /// it were produced from a prefix that never happened, so nothing
    /// past the mismatch is scored.
    #[test]
    fn a_wrong_draft_commits_the_samplers_token_and_stops() {
        let params = greedy_params();
        let mut state = SampleState::new(params.seed);
        let mut history = Vec::new();
        let draft = [3usize, 99, 5];
        let first = peaked(8, 3);
        // Row 0 says 4; the draft claimed 99, so 4 is committed and the
        // walk stops. Row 1 would have said 7 and must not be reached.
        let rows = vec![peaked(8, 4), peaked(8, 7), peaked(8, 1)];

        let b = verify_block(
            &mut state,
            &first,
            &rows,
            &draft,
            &params,
            &[],
            &mut history,
            &no_stops(),
            &decode,
        )
        .expect("verify");

        assert_eq!(b.accepted, 1, "only the first draft matched");
        assert_eq!(b.tokens, vec![3, 4], "the sampler's token, not 99");
        assert!(
            !b.tokens.contains(&99),
            "a draft must never be emitted as itself"
        );
        assert!(!b.tokens.contains(&7), "rows past the mismatch are invalid");
    }

    /// THE correctness claim, checked rather than argued: the tokens a
    /// verified block commits are exactly the tokens the ordinary
    /// per-token loop would have drawn from the same logits, because
    /// both call the same sampler over the same growing history.
    ///
    /// This is what "lossless by construction" has to mean in a test,
    /// and it holds whether the drafts were right or wrong -- the two
    /// cases below differ only in how many forward passes it took.
    #[test]
    fn a_verified_block_commits_what_sequential_sampling_would_have() {
        for draft in [
            vec![3usize, 4, 5], // all correct
            vec![3, 99, 5],     // wrong in the middle
            vec![42, 4, 5],     // wrong immediately
        ] {
            let params = greedy_params();
            let first = peaked(8, 3);
            let rows = vec![peaked(8, 4), peaked(8, 5), peaked(8, 6)];

            let mut spec_state = SampleState::new(params.seed);
            let mut spec_history = Vec::new();
            let b = verify_block(
                &mut spec_state,
                &first,
                &rows,
                &draft,
                &params,
                &[],
                &mut spec_history,
                &no_stops(),
                &decode,
            )
            .expect("verify");

            // The same rows, drawn one at a time, stopping after as
            // many tokens as the block committed.
            let mut seq_state = SampleState::new(params.seed);
            let mut seq_history: Vec<usize> = Vec::new();
            for i in 0..b.tokens.len() {
                let logits = if i == 0 { &first } else { &rows[i - 1] };
                let Step::Token { id: t, .. } = sample_next(
                    &mut seq_state,
                    logits,
                    &params,
                    &[],
                    &seq_history,
                    &no_stops(),
                    &decode,
                )
                .expect("sequential") else {
                    panic!("grammar completed in a grammarless test")
                };
                seq_history.push(t);
            }

            assert_eq!(
                b.tokens, seq_history,
                "draft {draft:?}: a verified block must commit the sequential answer"
            );

            // Checked independently of the walk's own length, because
            // the comparison above cannot see this: it drives its
            // reference loop from `b.tokens.len()`, so a block that
            // kept walking past a mismatch would agree with it
            // trivially. A sabotage that removed the `break` was
            // caught by exactly one test until this line existed.
            //
            // The invariant: a round commits its accepted drafts plus
            // ONE token, the corrective one at the mismatch or the
            // bonus one after the last draft. Never more, because
            // every row after the first disagreement was produced from
            // a prefix that never happened.
            assert_eq!(
                b.tokens.len(),
                b.accepted + 1,
                "draft {draft:?}: committed {} tokens for {} accepted drafts",
                b.tokens.len(),
                b.accepted
            );
        }
    }

    /// A scripted engine: position `p` produces a one-hot row whose
    /// winner is `script[p]`, so the answer is known and both paths
    /// must find it.
    struct Scripted {
        script: Vec<usize>,
        vocab: usize,
        /// Drafts to offer, one per round; empty entry means none.
        drafts: Vec<Vec<usize>>,
        round: usize,
        batches: usize,
        accepted: usize,
        drafted: usize,
        /// Forwards taken, which is the number this row exists to move.
        steps: usize,
    }

    impl Scripted {
        fn new(script: &[usize], vocab: usize, drafts: Vec<Vec<usize>>) -> Self {
            Scripted {
                script: script.to_vec(),
                vocab,
                drafts,
                round: 0,
                batches: 0,
                accepted: 0,
                drafted: 0,
                steps: 0,
            }
        }

        fn row(&self, pos: usize) -> Vec<f32> {
            peaked(self.vocab, self.script.get(pos).copied().unwrap_or(0))
        }
    }

    impl DecodeEngine for Scripted {
        fn step(&mut self, _token: usize, pos: usize) -> Vec<f32> {
            self.steps += 1;
            self.row(pos + 1)
        }

        fn draft(&mut self, _p: &[usize], _h: &[usize], max: usize) -> Vec<usize> {
            let d = self.drafts.get(self.round).cloned().unwrap_or_default();
            self.round += 1;
            d.into_iter().take(max).collect()
        }
        fn batch(&mut self, tokens: &[usize], pos: usize) -> Option<Vec<Vec<f32>>> {
            self.batches += 1;
            Some((0..tokens.len()).map(|i| self.row(pos + i + 1)).collect())
        }
        fn truncate(&mut self, _pos: usize) {}
        fn observe(&mut self, accepted: usize, drafted: usize) {
            self.accepted += accepted;
            self.drafted += drafted;
        }
    }

    /// THE claim the whole row rests on: speculation changes how many
    /// forwards it takes to get an answer, never the answer.
    ///
    /// Same scripted engine, same prompt, same sampler seed; once with
    /// no drafting and once with a drafter that is right, wrong, and
    /// silent in turn. The ids and the text have to match exactly, and
    /// the forward count has to DROP where drafts are accepted, or
    /// speculation is costing work rather than saving it.
    ///
    /// Comparing against the plain path rather than checking
    /// speculation alone is what caught the real bug here:
    /// `max_tokens` counts TOKENS and the loop counts ITERATIONS,
    /// which were the same thing until a round could commit a block,
    /// and a limit of 6 returned nine.
    #[test]
    fn speculation_changes_the_forward_count_and_not_the_answer() {
        let script = vec![1usize, 2, 3, 4, 5, 6, 7];
        let vocab = 16;
        let params = GenerationParams {
            max_tokens: 6,
            ..greedy_params()
        };
        let bytes = |ids: &[usize]| {
            ids.iter()
                .map(|i| format!("<{i}>"))
                .collect::<String>()
                .into_bytes()
        };

        let mut plain = Scripted::new(&script, vocab, Vec::new());
        let mut plain_text = String::new();
        let (_, plain_ids, _, _probs) = sample_until_stop(
            peaked(vocab, script[0]),
            0,
            &[],
            &no_stops(),
            &params,
            bytes,
            &mut plain,
            |c| plain_text.push_str(c),
            &decode,
            0,
        )
        .expect("plain");

        // `draft[0]` is the token drawn from the CURRENT logits, not
        // the one after it: verification starts at the position the
        // caller already holds a row for. Getting this off by one made
        // the first version of this test vacuous -- every draft was
        // rejected at index 0 and the speed assertion never ran.
        for (label, drafts) in [
            ("always right", vec![vec![1usize, 2, 3], vec![5, 6, 7]]),
            ("always wrong", vec![vec![9usize, 9, 9], vec![9, 9, 9]]),
            (
                "mixed, and silent",
                vec![vec![1usize, 9, 3], vec![], vec![5, 6]],
            ),
        ] {
            let mut engine = Scripted::new(&script, vocab, drafts);
            let mut spec_text = String::new();
            let (_, spec_ids, _, _probs) = sample_until_stop(
                peaked(vocab, script[0]),
                0,
                &[],
                &no_stops(),
                &params,
                bytes,
                &mut engine,
                |c| spec_text.push_str(c),
                &decode,
                3,
            )
            .expect("speculative");

            println!(
                "CASE {label} accepted={} drafted={} spec_steps={} plain_steps={}",
                engine.accepted, engine.drafted, engine.steps, plain.steps
            );
            assert_eq!(
                spec_ids, plain_ids,
                "{label}: speculation changed the ids (plain={plain_ids:?} spec={spec_ids:?})"
            );
            assert_eq!(
                spec_text, plain_text,
                "{label}: speculation changed the text"
            );

            if label == "always wrong" {
                assert_eq!(engine.accepted, 0, "a wrong draft must never be accepted");
            } else {
                // Asserted unconditionally for the cases that DO
                // accept, so the claim cannot go untested the way it
                // did while the drafts were off by one.
                assert!(
                    engine.accepted > 0,
                    "{label}: nothing was accepted, so this case proves nothing"
                );
                assert!(
                    engine.steps < plain.steps,
                    "{label}: accepted {} drafts and still took {} forwards against {}",
                    engine.accepted,
                    engine.steps,
                    plain.steps
                );
            }
        }
    }

    /// An empty draft is the ordinary loop: one row, one token, nothing
    /// accepted and nothing saved. Worth pinning because it is the
    /// boundary a caller hits when the drafter has nothing to offer.
    #[test]
    fn an_empty_draft_commits_exactly_one_token() {
        let params = greedy_params();
        let mut state = SampleState::new(params.seed);
        let mut history = Vec::new();

        let b = verify_block(
            &mut state,
            &peaked(8, 2),
            &[],
            &[],
            &params,
            &[],
            &mut history,
            &no_stops(),
            &decode,
        )
        .expect("verify");

        assert_eq!(b.tokens, vec![2]);
        assert_eq!(b.accepted, 0);
        assert_eq!(b.drafted, 0);
    }

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
