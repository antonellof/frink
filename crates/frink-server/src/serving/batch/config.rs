//! The batcher's knobs, and where each one is read from.
//!
//! Every value here is a scalar with a default. Tests pass a
//! [`BatcherConfig`] explicitly rather than setting process environment,
//! which two tests running in parallel cannot do without racing.

use std::sync::mpsc::Sender;
use std::sync::Arc;

use crate::generate::{DecodeError, FinishReason, Usage};

use super::status::DEFAULT_DECODE_LOG_INTERVAL;

/// Detokenize to RAW BYTES, not text.
///
/// Bytes because a character can straddle two tokens and a batched row
/// is decoded one token at a time: resolving UTF-8 per token turns each
/// half into U+FFFD and loses it (#124). Each row buffers its own tail
/// through `crate::utf8_stream::Utf8Stream` -- per row, because rows
/// interleave and one shared buffer would splice two answers together.
pub(super) type DecodeFn = Arc<dyn Fn(&[usize]) -> Vec<u8> + Send + Sync>;

/// Finish reason, generated token ids, detokenized text (stop-trimmed),
/// and usage. Callers should prefer `text` for the response body when
/// stop sequences may have cut the decoded string short of a full
/// `decode(ids)`.
pub(super) type JobResult = Result<(FinishReason, Vec<usize>, String, Usage), DecodeError>;

/// Worker → caller messages. Chunks carry incremental detokenized text;
/// `Finished` ends the job with the same payload as before.
pub(super) enum BatcherEvent {
    Chunk(String),
    Finished(Box<JobResult>),
}

pub(super) fn send_finished(reply: &Sender<BatcherEvent>, result: JobResult) {
    let _ = reply.send(BatcherEvent::Finished(Box::new(result)));
}

/// Prompt tokens run per prefill chunk when `FRINK_CB_PREFILL_CHUNK`
/// is unset. Large enough that a short prompt still prefills in one
/// tick, small enough that a long one cannot monopolize the worker.
pub const DEFAULT_PREFILL_CHUNK: usize = 128;

/// Token positions per KV block when `FRINK_CB_KV_BLOCK_SIZE` is
/// unset. The block is the admission quantum: smaller wastes less on
/// the rounding-up of each request, larger keeps the ledger cheap.
pub const DEFAULT_KV_BLOCK_SIZE: usize = 256;

/// Jobs allowed to wait for admission when `FRINK_CB_MAX_QUEUE` is
/// unset. Deep enough that a normal burst queues instead of failing,
/// shallow enough that a retry storm is refused while the server can
/// still refuse cheaply.
pub const DEFAULT_MAX_QUEUE: usize = 512;

/// Scheduler knobs, read from the environment by `from_env` and passed
/// explicitly by tests (which must not race each other over process
/// environment).
#[derive(Clone, Copy, Debug)]
pub struct BatcherConfig {
    /// Cap on in-flight sequences, counting prompts still prefilling.
    pub max_seqs: usize,
    /// Prompt tokens per `PrefillState::step_chunk` call.
    pub prefill_chunk: usize,
    /// Tokens one tick may run, prompt and generated together.
    ///
    /// The prefill chunk is sized against this minus the decode width,
    /// so a tick's cost stays near it instead of growing with the
    /// batch. See [`super::step_budget`].
    pub max_batch_tokens: usize,
    /// Jobs that may wait for admission before new ones are refused.
    pub max_queue: usize,
    /// Token positions per KV block, the admission quantum.
    pub kv_block_size: usize,
    /// Total KV blocks the scheduler may hand out, or `None` for no
    /// block budget (sequence count alone, the pre-budget behaviour).
    pub kv_blocks: Option<usize>,
    /// Token positions (prompt + `max_tokens`) any single request may
    /// ask for, or `None` for no per-request ceiling.
    pub max_context: Option<usize>,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        BatcherConfig {
            max_seqs: usize::MAX,
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
            max_batch_tokens: super::step_budget::DEFAULT_MAX_BATCH_TOKENS,
            max_queue: DEFAULT_MAX_QUEUE,
            kv_block_size: DEFAULT_KV_BLOCK_SIZE,
            kv_blocks: None,
            max_context: None,
        }
    }
}

impl BatcherConfig {
    pub fn from_env() -> Self {
        BatcherConfig {
            max_seqs: env_positive("FRINK_CB_MAX_SEQS").unwrap_or(usize::MAX),
            prefill_chunk: env_positive(
                crate::prefill_batch::PREFILL_CHUNK_ENV_KEYS[crate::prefill_batch::BATCH_PATH_KEY],
            )
            .unwrap_or(DEFAULT_PREFILL_CHUNK),
            max_batch_tokens: env_positive("FRINK_CB_MAX_BATCH_TOKENS")
                .unwrap_or(super::step_budget::DEFAULT_MAX_BATCH_TOKENS),
            max_queue: env_positive("FRINK_CB_MAX_QUEUE").unwrap_or(DEFAULT_MAX_QUEUE),
            kv_block_size: env_positive("FRINK_CB_KV_BLOCK_SIZE").unwrap_or(DEFAULT_KV_BLOCK_SIZE),
            kv_blocks: env_positive("FRINK_CB_KV_BLOCKS"),
            max_context: env_positive("FRINK_CB_MAX_CONTEXT"),
        }
    }
}

pub(super) fn env_positive(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    let value: usize = raw
        .parse()
        .unwrap_or_else(|_| panic!("{name} must be a positive integer"));
    assert!(value > 0, "{name} must be a positive integer");
    Some(value)
}

/// How many decode forwards pass between two status lines.
///
/// `FRINK_DECODE_LOG_INTERVAL=0` means "every forward", which the
/// reporter clamps to one rather than dividing by zero. An unparseable
/// value takes the default rather than failing a server to start over a
/// log setting.
pub(super) fn decode_log_interval_from_env() -> usize {
    std::env::var("FRINK_DECODE_LOG_INTERVAL")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_DECODE_LOG_INTERVAL)
}
