//! What the worker counts as it runs, and the snapshot `/metrics` and
//! `/admin/stats` read it through.

use std::sync::atomic::{AtomicU64, AtomicUsize};

/// Counters the worker keeps as it runs, exposed through
/// `ContinuousBatcher::stats` so prefill/decode interleaving is
/// *observable* rather than merely intended.
#[derive(Default)]
pub(super) struct Counters {
    pub(super) prefill_chunks: AtomicU64,
    pub(super) prefill_tokens: AtomicU64,
    pub(super) decode_steps: AtomicU64,
    /// High-water mark of KV blocks actually held by in-flight work.
    ///
    /// Deliberately measured from the rows and prefills themselves
    /// rather than derived from `BlockBudget::free`: a ledger-derived
    /// peak cannot exceed the budget however broken admission is, so it
    /// would be a gauge that reports the invariant instead of checking
    /// it.
    pub(super) peak_blocks: AtomicUsize,
    /// Decoding rows evicted back to the queue to admit a waiting job.
    pub(super) preempted: AtomicU64,
}

/// A snapshot of the worker's counters.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct BatcherStats {
    /// `PrefillState::step_chunk` calls the worker has made.
    pub prefill_chunks: u64,
    /// Prompt tokens run through prefill.
    pub prefill_tokens: u64,
    /// Batched decode steps (one per tick that had an active row,
    /// regardless of how many rows that step covered).
    pub decode_steps: u64,
    /// Jobs currently waiting for admission.
    pub queue_depth: usize,
    /// Jobs refused because the queue was full.
    pub queue_rejected: u64,
    /// Total KV blocks in the budget; 0 when none is configured.
    pub kv_blocks_total: usize,
    /// KV blocks not currently reserved by an in-flight request.
    pub kv_blocks_free: usize,
    /// Token positions per KV block.
    pub kv_block_size: usize,
    /// Jobs refused because they exceed the whole KV block budget.
    /// Distinct from `queue_rejected`, which is momentary pressure.
    pub kv_rejected_too_large: u64,
    /// Jobs refused for asking for more context than one request may
    /// have. Distinct again: the fix is in the request, not the box.
    pub kv_rejected_context_length: u64,
    /// Most KV blocks ever held by in-flight work at one moment,
    /// counted from the rows themselves.
    pub kv_blocks_peak: usize,
    /// Requests the scheduler actually stopped because they were
    /// cancelled.
    pub aborted: u64,
    /// Decoding rows preempted back to the queue to admit a waiting
    /// job (`FRINK_CB_PREEMPT_AFTER_TICKS`). Each one costs a
    /// re-prefill, so a rising count is a budget too small for the
    /// load rather than free fairness.
    pub preempted: u64,
    /// The configured cap on in-flight sequences (llama.cpp `-np` /
    /// `FRINK_CB_MAX_SEQS`), or 0 when unlimited.
    ///
    /// Reported because a knob nobody can read back is a knob nobody
    /// can verify: `-np` set an environment variable that no route
    /// echoed, so "the flag is wired" and "the flag is documented"
    /// were separate claims with nothing joining them.
    pub max_seqs: usize,
    /// Prompt tokens per prefill chunk (llama.cpp `-ub`), for the same
    /// reason.
    pub prefill_chunk: usize,
}
