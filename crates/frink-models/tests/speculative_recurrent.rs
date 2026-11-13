//! Speculative decoding on targets with recurrent layers: a rejected
//! block cannot be truncated out of a recurrent state, so the caches are
//! checkpointed before each verify batch and a partial accept restores
//! them and replays the committed tokens (llama.cpp's server, the
//! `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` path). The answer must be plain
//! greedy decoding's, token for token.

mod common;

use common::{graph_caches, load_graph_fixture, GRAPH_PROMPT};
use frink_models::sampling::SamplingParams;
use frink_models::speculative::{speculative_decode_with, DraftBlock, Drafter, SpeculativeOptions};
use frink_models::Decoder;

/// Proposes `right` tokens of the known greedy answer and then one
/// wrong one, so every round with room for it is a partial accept: the
/// restore and the replay run every round.
struct Scripted {
    answer: Vec<usize>,
    right: usize,
    vocab: usize,
}

impl Drafter for Scripted {
    fn propose(&mut self, history: &[usize], _hidden: &[f32], max_len: usize) -> DraftBlock {
        let next = history.len();
        let mut tokens: Vec<usize> = self
            .answer
            .iter()
            .skip(next)
            .take(self.right)
            .copied()
            .collect();
        if let Some(&want) = self.answer.get(next + tokens.len()) {
            tokens.push((want + 1) % self.vocab);
        }
        tokens.truncate(max_len);
        DraftBlock::deterministic(tokens)
    }
}

fn greedy(target: &Decoder, n: usize) -> Vec<usize> {
    let mut caches = graph_caches(target);
    let mut logits = target
        .forward_batch(&GRAPH_PROMPT, 0, &mut caches)
        .pop()
        .unwrap();
    let mut out = Vec::new();
    for pos in (GRAPH_PROMPT.len()..).take(n) {
        let t = common::argmax(&logits);
        out.push(t);
        logits = target.forward_token(t, pos, &mut caches);
    }
    out
}

fn assert_speculation_is_greedy(name: &str) {
    let target = load_graph_fixture(name);
    assert!(
        target.config.has_recurrent_layers(),
        "{name} is meant to be a recurrent target"
    );
    let n = 24;
    let plain = greedy(&target, n);
    for right in [0, 2] {
        let mut answer = GRAPH_PROMPT.to_vec();
        answer.extend_from_slice(&plain);
        let mut drafter = Scripted {
            answer,
            right,
            vocab: target.config.vocab_size,
        };
        let mut caches = graph_caches(&target);
        let result = speculative_decode_with(
            &target,
            &GRAPH_PROMPT,
            &mut caches,
            &mut drafter,
            &SpeculativeOptions {
                max_new_tokens: n,
                start_pos: 0,
                sampling: SamplingParams::default(),
                seed: 0,
            },
        );
        assert_eq!(
            result.generated_tokens, plain,
            "{name}, {right} right per block: speculation changed the answer"
        );
        // Every block was cut short, so each round past the prefill ran
        // a verify and a replay; and with `right` > 0 some drafts were
        // committed through the restore.
        assert!(
            result.forward_calls > result.verification_steps + 1,
            "{name}: no replay ran, so the restore was never exercised: {result:?}"
        );
        assert_eq!(result.accepted_tokens > 0, right > 0, "{name}: {result:?}");
    }
}

#[test]
fn speculation_on_a_gated_delta_net_target_is_greedy_decoding() {
    assert_speculation_is_greedy("qwen35");
}

#[test]
fn speculation_on_a_mamba2_target_is_greedy_decoding() {
    assert_speculation_is_greedy("mamba2");
}

#[test]
fn speculation_on_a_hybrid_mamba_attention_target_is_greedy_decoding() {
    assert_speculation_is_greedy("jamba");
}

#[test]
fn speculation_on_a_lightning_attention_target_is_greedy_decoding() {
    assert_speculation_is_greedy("minimax_01");
}

#[test]
fn speculation_on_a_plamo2_target_is_greedy_decoding() {
    assert_speculation_is_greedy("plamo2");
}

/// The same on Metal, where a GDN or Mamba kernel reads and writes the
/// recurrent state through a no-copy wrap of its host bytes: the restore
/// must land in those bytes, not in a new allocation the wrap does not
/// see.
///
/// `cargo test -p frink-models --features metal --test speculative_recurrent -- --ignored`
#[test]
#[ignore = "needs Apple Metal GPU"]
fn metal_speculation_on_recurrent_targets_is_greedy_decoding() {
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    {
        common::pin_backend(common::Backend::Metal);
        for name in ["qwen35", "mamba2", "jamba"] {
            assert_speculation_is_greedy(name);
        }
    }
}
