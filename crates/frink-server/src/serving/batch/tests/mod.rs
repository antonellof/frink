//! The batcher's tests, grouped by what they hold the scheduler to.
//!
//! They live under `batch` rather than beside one submodule because
//! what almost all of them cover is the *tick* -- admission, prefill,
//! decode and flush composed -- which is the whole module and not any
//! one file in it. The fixtures they share are here; each group below
//! uses them.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use frink_core::cache::KvCache;
use frink_models::config::test_dense_fixture;
use frink_models::sampling::{Sampler, SamplingParams};
use frink_models::tokenizer::StopTokens;
use frink_models::Decoder;

use crate::budget::ContextCeiling;
use crate::cancel::CancelToken;
use crate::generate::{DecodeError, FinishReason, GenerationParams, PagedKvConfig};
use crate::stop::StopMatcher;

use super::batcher::ContinuousBatcher;
use super::block_budget::BlockBudget;
use super::config::{BatcherConfig, BatcherEvent, DecodeFn, JobResult, DEFAULT_KV_BLOCK_SIZE};
use super::prefill::{Prefill, PrefillState};
use super::queue::{AbortId, AbortInbox, QueueGate};
use super::row::{Job, RowKv, Rows, Slot};
use super::status::{PoolUsage, PrefillSnapshot};
use super::worker::{admit, apply_aborts, batch_status};

mod admission;
mod batching;
mod cancel;
mod queue;
mod radix;
mod rows;
mod status;

fn tiny_decoder() -> Arc<Decoder> {
    let cfg = test_dense_fixture();
    let vocab = cfg.vocab_size;
    Arc::new(Decoder::new_random_small(cfg, 2, vocab))
}

fn greedy_params(max_tokens: usize, seed: u64) -> GenerationParams {
    GenerationParams {
        cache_salt: None,
        prompt_logprobs: None,
        wants_logprobs: false,
        n: 1,
        reasoning: None,
        max_tokens,
        // `SamplingParams::default()` IS greedy with every filter
        // off, so restating its fields here would be a second copy of
        // the defaults that could drift from the first.
        sampling: SamplingParams::default(),
        seed,
        stop: vec![],
        stop_token_ids: Vec::new(),
        json_object: false,
        grammar: None,
        cancel: None,
        ignore_eos: false,
        reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
        lora: None,
    }
}

fn identity_decode() -> DecodeFn {
    Arc::new(|ids: &[usize]| ids.iter().map(|id| b'A' + (*id as u8 % 26)).collect())
}

/// `identity_decode`'s output as text, for tests that need to reason
/// about characters rather than bytes. ASCII by construction, so this
/// cannot be the lossy step the code under test is about.
fn identity_decode_text(ids: &[usize]) -> String {
    String::from_utf8(identity_decode()(ids)).expect("identity_decode is ASCII")
}

fn sequential_ids(decoder: &Decoder, prompt: &[usize], params: &GenerationParams) -> Vec<usize> {
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    let mut pos = 0;
    let mut logits = Vec::new();
    for &tok in prompt {
        logits = decoder.forward_token(tok, pos, &mut caches);
        pos += 1;
    }
    let mut sampler = Sampler::new(params.seed);
    let mut generated = Vec::new();
    for _ in 0..params.max_tokens {
        // The empty prompt half mirrors `sample_step::sample_next`,
        // which is what the batcher goes through: this reference decode
        // has to make the SAME penalty-window choice the server makes,
        // or it stops being a reference. See the note there.
        let next = sampler.sample(
            &logits,
            &params.sampling,
            frink_models::PenaltyWindow::new(&[], &generated),
        );
        generated.push(next);
        logits = decoder.forward_token(next, pos, &mut caches);
        pos += 1;
    }
    generated
}

/// The KV shape of `tiny_decoder`, for pricing a refusal in bytes.
fn test_shape() -> frink_models::KvShape {
    frink_models::KvShape::from_config(&test_dense_fixture(), frink_models::KvElem::F32)
}

/// A ledger with no budget configured, for tests that are not
/// about admission.
fn no_budget() -> BlockBudget {
    BlockBudget::new(
        DEFAULT_KV_BLOCK_SIZE,
        None,
        Arc::new(ContextCeiling::new(None, test_shape())),
    )
}

fn budget(block_size: usize, total: Option<usize>) -> BlockBudget {
    BlockBudget::new(
        block_size,
        total,
        Arc::new(ContextCeiling::new(None, test_shape())),
    )
}

fn finished_result(event: BatcherEvent) -> JobResult {
    match event {
        BatcherEvent::Finished(result) => *result,
        BatcherEvent::Chunk(_) => panic!("expected finished event, got chunk"),
    }
}

fn test_slot(max_tokens: usize, seed: u64) -> (Slot, mpsc::Receiver<BatcherEvent>) {
    let (tx, rx) = mpsc::channel();
    let params = greedy_params(max_tokens, seed);
    (
        Slot {
            prompt_ids: Vec::new(),
            kv: RowKv::Contiguous(Vec::new()),
            pos: 0,
            logits: Vec::new(),
            sample: crate::sample_step::SampleState::new(seed),
            generated_ids: Vec::new(),
            visible: String::new(),
            stops: StopMatcher::new(&params.stop, &params.stop_token_ids),
            prompt_tokens: 0,
            max_tokens,
            stop_tokens: StopTokens::default(),
            params,
            reply: tx,
            abort: AbortId(0),
            blocks: 1,
            finish: None,
            error: None,
            clock: super::clock::RowClock::start(),
            utf8: Default::default(),
        },
        rx,
    )
}

fn budget_config(block_size: usize, blocks: usize) -> BatcherConfig {
    BatcherConfig {
        prefill_chunk: 1,
        kv_block_size: block_size,
        kv_blocks: Some(blocks),
        ..BatcherConfig::default()
    }
}

fn cancellable_params(max_tokens: usize, seed: u64) -> (GenerationParams, CancelToken) {
    let token = CancelToken::new();
    let mut params = greedy_params(max_tokens, seed);
    params.cancel = Some(token.clone());
    (params, token)
}

fn abortable_job(abort: AbortId, prompt: Vec<usize>) -> (Job, mpsc::Receiver<BatcherEvent>) {
    let (tx, rx) = mpsc::channel();
    (
        Job {
            prompt_tokens: prompt,
            params: greedy_params(4, 1),
            stop_tokens: StopTokens::from_eos(None),
            reply: tx,
            abort,
            blocks: 1,
        },
        rx,
    )
}
