//! What one scheduler step reports about itself.
//!
//! These types used to sit in a separate module from the batcher that
//! is their only caller, because they were ported and it was not. They
//! are the batcher's status line: [`StatusReporter`] renders one prefill
//! line per admitted chunk and one decode line every Nth forward, and
//! [`PoolUsage`] is the single convention every pool gauge follows.
//!
//! Ported from FreeToken's `scheduler/status.py` and the pool-occupancy
//! helpers of `scheduler/scheduler.py` (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`.

use crate::serving::admission::AdmittedChunk;

/// A pool's occupancy, for a status line.
pub fn usage_ratio(used: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    used as f64 / total as f64
}

/// Tokens per second over a window, given the tokens produced and the
/// seconds they took. A non-positive interval reports zero rather than
/// an infinity.
pub fn throughput(tokens: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        return 0.0;
    }
    tokens as f64 / seconds
}

/// One pool's occupancy, in whatever unit that pool is denominated:
/// pages for the KV pool, tokens for the window pool, slots for the
/// recurrent-state pool.
///
/// Both numbers follow one convention, and the convention is the whole
/// point of the type:
///
/// * `total` **excludes the pool's reserved sentinel**. Window slot 0
///   and recurrent slot 0 are never handed out, so counting them claims
///   a unit of capacity nothing can ever occupy.
/// * `used` **excludes free units and evictable cache entries alike**.
///   An evictable prefix nobody is reading is memory, not occupancy --
///   exactly the rule [`crate::policy::cache_manager::CacheManager::page_usage`]
///   already applies to KV pages, and the rule
///   [`crate::policy::cache_manager::CacheManager::available_tokens`] admits
///   against.
///
/// Counting evictable entries as used is the failure this type exists
/// to prevent. A healthy idle server *fills* its pools with reusable
/// prefixes, so every gauge would sit near 100% while the machine is
/// doing nothing; the three gauges would disagree with each other and
/// with admission, which is happily seating requests into the memory
/// they claim is full. An operator reading that grows a pool that was
/// never full, and a live client watching the same gauges is told the
/// same lie.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolUsage {
    /// Units held by something that cannot be evicted: live requests
    /// and locked prefixes.
    pub used: usize,
    /// Units that could ever be handed out, sentinel excluded.
    pub total: usize,
}

impl PoolUsage {
    /// Occupancy of a pool with `total` usable units of which
    /// `available` could be served right now -- free units **plus**
    /// evictable ones.
    ///
    /// The subtraction saturates rather than wrapping: `available` is
    /// summed from two independent sources (a free list and a prefix
    /// tree), so a caller that double-counts a unit for one step would
    /// otherwise wrap `used` to something near `usize::MAX` and render
    /// a gauge of astronomical nonsense. Reading as an empty pool for
    /// that step is a far smaller lie.
    pub fn from_available(total: usize, available: usize) -> Self {
        PoolUsage {
            used: total.saturating_sub(available),
            total,
        }
    }

    /// Fraction of the pool that is occupied, `0.0` for a pool the
    /// engine did not allocate.
    pub fn ratio(&self) -> f64 {
        usage_ratio(self.used, self.total)
    }
}

/// The gauges every status line carries, whichever phase it reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchStatus {
    /// Requests decoding right now.
    pub running_reqs: usize,
    /// Prompts still waiting to be admitted.
    pub queue_reqs: usize,
    /// KV occupancy **in pages**.
    pub kv_pages: PoolUsage,
    pub page_size: usize,
    /// `None` for a model with no window pool.
    pub window: Option<PoolUsage>,
    /// `None` for a model with no recurrent-state pool.
    pub recurrent: Option<PoolUsage>,
}

/// What a prefill batch looked like **when it was scheduled**.
///
/// Taken before the forward runs, and that timing is the entire reason
/// the type exists. By the time a batch is reported, the forward has
/// advanced every request's cached length to its device length, so
/// reading the live requests logs the *decode* state of a *prefill*
/// batch: `#new-token` equal to the number of requests -- one token
/// each, as though they were decoding -- and `#cached-token` equal to
/// the whole prompt, on every line. Both numbers look plausible, both
/// are fiction, and together they make a prefill log useless for the
/// one thing it is read for: telling a cache hit from a cache miss.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefillSnapshot {
    /// Chunks in the batch, continuations included.
    pub new_seqs: usize,
    /// Prompt tokens this batch computes.
    pub new_tokens: usize,
    /// Prefix-cache hits, counted once per prompt.
    pub cached_tokens: usize,
}

impl PrefillSnapshot {
    /// Snapshot the chunks a [`PrefillPass`] has just admitted, at the
    /// only moment the numbers are still true.
    ///
    /// Cached tokens come from [`AdmittedChunk::admission`], which is
    /// set on a prompt's **first** chunk only: a prompt cut into ten
    /// chunks reports its prefix hit once, not ten times. New tokens
    /// come from every chunk, continuations included -- those are
    /// tokens this batch really computes.
    pub fn from_chunks(chunks: &[AdmittedChunk]) -> Self {
        PrefillSnapshot {
            new_seqs: chunks.len(),
            new_tokens: chunks.iter().map(|chunk| chunk.chunk_len).sum(),
            cached_tokens: chunks
                .iter()
                .filter_map(|chunk| chunk.admission)
                .map(|admission| admission.cached_tokens)
                .sum(),
        }
    }
}

/// Decode forwards between two status lines, by default.
pub const DEFAULT_DECODE_LOG_INTERVAL: usize = 40;

/// The operator-facing batch log.
///
/// `frink-edge` holds no clock and no logger, so this holds neither:
/// the caller passes the time and receives the line to log, or `None`
/// when there is nothing to say. That keeps every rule below testable
/// on any host, with time as an ordinary parameter.
///
/// # Prefill lines report a snapshot
///
/// Always the [`PrefillSnapshot`] taken at schedule time, never the
/// live requests -- see that type for what reading the live ones
/// produces.
///
/// # Decode lines are throttled, and report the gap
///
/// A decode forward happens tens of times a second, so a line per
/// forward would bury everything else in the log. One line is emitted
/// every `decode_log_interval` forwards, and it reports throughput over
/// the interval **since the last emitted line**, resetting the token
/// counter each time.
///
/// A lifetime average would be the naive reading, and it is worse than
/// no number at all: what an operator watches a decode log for is a
/// *change* -- a batch that has begun to thrash, a pool that has
/// started evicting -- and a running mean over a long-lived server is
/// precisely the statistic that cannot show one.
#[derive(Debug)]
pub struct StatusReporter {
    decode_log_interval: usize,
    last_prefill_time: f64,
    last_decode_time: f64,
    decode_forwards: usize,
    decode_tokens: usize,
}

impl StatusReporter {
    /// A reporter whose first interval starts at `now`.
    ///
    /// `decode_log_interval` is clamped to at least one forward: zero
    /// would make the `forwards % interval` throttle a division by
    /// zero, and a caller that passes it means "log every forward".
    pub fn new(decode_log_interval: usize, now: f64) -> Self {
        StatusReporter {
            decode_log_interval: decode_log_interval.max(1),
            last_prefill_time: now,
            last_decode_time: now,
            decode_forwards: 0,
            decode_tokens: 0,
        }
    }

    pub fn decode_log_interval(&self) -> usize {
        self.decode_log_interval
    }

    /// The line for a prefill batch. Always emitted: prefill batches are
    /// rare enough to be worth one line each.
    ///
    /// Input throughput covers the gap since the last prefill line and
    /// counts the snapshot's tokens -- the ones this batch computed.
    pub fn report_prefill(
        &mut self,
        now: f64,
        snapshot: &PrefillSnapshot,
        status: &BatchStatus,
    ) -> String {
        let gap = now - self.last_prefill_time;
        self.last_prefill_time = now;
        let input_throughput = throughput(snapshot.new_tokens, gap);
        format!(
            "Prefill batch, \
             #new-seq: {}, \
             #new-token: {}, \
             #cached-token: {}, \
             token usage: {:.2}, \
             {}{}#running-req: {}, \
             #queue-req: {}, \
             input throughput (token/s): {:.2}",
            snapshot.new_seqs,
            snapshot.new_tokens,
            snapshot.cached_tokens,
            status.kv_pages.ratio(),
            window_field(status.window),
            recurrent_field(status.recurrent),
            status.running_reqs,
            status.queue_reqs,
            input_throughput,
        )
    }

    /// The line for a decode forward, or `None` because this forward is
    /// not one of the ones that log.
    ///
    /// `batch_reqs` is the number of requests in the forward, which is
    /// also the number of tokens it produced -- one each.
    ///
    /// Every call counts, whether or not it emits: the tokens of the
    /// silent forwards are exactly what the emitted line's throughput is
    /// measuring.
    pub fn report_decode(
        &mut self,
        now: f64,
        batch_reqs: usize,
        status: &BatchStatus,
    ) -> Option<String> {
        self.decode_forwards += 1;
        self.decode_tokens += batch_reqs;
        if !self
            .decode_forwards
            .is_multiple_of(self.decode_log_interval)
        {
            return None;
        }
        let gap = now - self.last_decode_time;
        self.last_decode_time = now;
        let gen_throughput = throughput(self.decode_tokens, gap);
        // Reset, so the next line measures its own interval rather than
        // the whole run.
        self.decode_tokens = 0;
        Some(format!(
            "Decode batch, \
             #running-req: {}, \
             #token: {}, \
             token usage: {:.2}, \
             {}{}gen throughput (token/s): {:.2}, \
             #queue-req: {}",
            status.running_reqs,
            status.kv_pages.used * status.page_size,
            status.kv_pages.ratio(),
            window_field(status.window),
            recurrent_field(status.recurrent),
            gen_throughput,
            status.queue_reqs,
        ))
    }
}

/// One pool's field, or nothing at all.
///
/// A pool the model does not have is **omitted from the line**, not
/// printed as `0/0`: a zero row says the engine allocated a pool and
/// left it empty, which is a different -- and much more alarming --
/// fact than not having one. Same rule as `cache_report`'s dropped
/// columns.
fn pool_field(name: &str, unit: &str, usage: Option<PoolUsage>) -> String {
    match usage {
        Some(usage) => format!(
            "#{name}-{unit}: {}/{}, {name} usage: {:.2}, ",
            usage.used,
            usage.total,
            usage.ratio()
        ),
        None => String::new(),
    }
}

/// The window pool keeps upstream's `swa` wording, so a frink log line
/// diffs against a FreeToken one without translation.
fn window_field(usage: Option<PoolUsage>) -> String {
    pool_field("swa", "token", usage)
}

/// Likewise `mamba` for the recurrent-state pool.
fn recurrent_field(usage: Option<PoolUsage>) -> String {
    pool_field("mamba", "slot", usage)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::serving::admission::tests::{geometry, roomy};
    use crate::serving::admission::{ChunkState, PendingRequest, PrefillPass};

    #[test]
    fn a_status_line_never_divides_by_zero() {
        assert_eq!(usage_ratio(3, 4), 0.75);
        assert_eq!(usage_ratio(3, 0), 0.0);
        assert_eq!(throughput(120, 2.0), 60.0);
        assert_eq!(throughput(120, 0.0), 0.0);
        assert_eq!(throughput(120, -1.0), 0.0);
    }

    fn batch_status() -> BatchStatus {
        BatchStatus {
            running_reqs: 2,
            queue_reqs: 1,
            kv_pages: PoolUsage {
                used: 50,
                total: 200,
            },
            page_size: 16,
            window: None,
            recurrent: None,
        }
    }

    /// The rule every pool gauge shares: an evictable cache entry is
    /// memory, not occupancy. Counting it as used -- the naive way --
    /// reports an idle-but-warm pool at 6/7 (0.86) instead of 1/7
    /// (0.14), and an operator grows a pool that was never full.
    ///
    /// Stated against [`PoolUsage::from_available`] directly. The
    /// window and recurrent slot pools that used to supply the numbers
    /// went with the multi-currency prefix cache; the arithmetic they
    /// shared with the KV gauge is what mattered and is still here.
    #[test]
    fn evictable_entries_are_memory_and_not_occupancy() {
        // Seven usable slots: five sit on unlocked tree nodes a new
        // request could evict, one is free, one is held by a live
        // request.
        let usage = PoolUsage::from_available(7, 1 + 5);
        assert_eq!(usage.used, 1, "5 evictable + 1 free are all available");
        assert_eq!(usage.total, 7);
        assert!(
            (usage.ratio() - 1.0 / 7.0).abs() < 1e-9,
            "{}",
            usage.ratio()
        );

        let naive = PoolUsage::from_available(7, 1);
        assert_eq!(
            naive.used, 6,
            "counting evictable as used is what this test rejects"
        );
    }

    /// Every pool is gauged the same way, whatever it holds: 200 units,
    /// 50 free and 30 evictable is 120 used, on the KV pool and on any
    /// other.
    #[test]
    fn every_pool_gauge_counts_availability_the_same_way() {
        let kv = PoolUsage::from_available(200, 50 + 30);
        assert_eq!((kv.used, kv.total), (120, 200));
    }

    /// A pool the model does not have is left out of the line entirely.
    /// Printing `#swa-token: 0/0` -- the naive way -- claims the engine
    /// allocated a window pool and left it empty, which is a different
    /// and much more alarming fact than not having one.
    #[test]
    fn a_pool_the_model_does_not_have_is_omitted_from_the_line() {
        let mut reporter = StatusReporter::new(1, 0.0);
        let dense = batch_status();
        let line = reporter.report_prefill(1.0, &PrefillSnapshot::default(), &dense);
        assert!(!line.contains("swa"), "{line}");
        assert!(!line.contains("mamba"), "{line}");
        assert!(!line.contains("0/0"), "{line}");

        let hybrid = BatchStatus {
            window: Some(PoolUsage {
                used: 8448,
                total: 76800,
            }),
            recurrent: Some(PoolUsage {
                used: 37,
                total: 256,
            }),
            ..dense
        };
        let line = reporter.report_prefill(2.0, &PrefillSnapshot::default(), &hybrid);
        assert!(
            line.contains("#swa-token: 8448/76800, swa usage: 0.11, "),
            "{line}"
        );
        assert!(
            line.contains("#mamba-slot: 37/256, mamba usage: 0.14, "),
            "{line}"
        );

        // Both phases carry them.
        let line = reporter.report_decode(3.0, 1, &hybrid).unwrap();
        assert!(line.contains("#swa-token: 8448/76800"), "{line}");
        assert!(line.contains("#mamba-slot: 37/256"), "{line}");
        let line = reporter.report_decode(4.0, 1, &dense).unwrap();
        assert!(!line.contains("swa"), "{line}");
        assert!(!line.contains("mamba"), "{line}");
    }

    /// Decode lines are throttled: one every `decode_log_interval`
    /// forwards, and the token count is `#used pages * page size`.
    #[test]
    fn decode_lines_are_emitted_only_every_nth_forward() {
        let mut reporter = StatusReporter::new(3, 0.0);
        let status = BatchStatus {
            kv_pages: PoolUsage {
                used: 60,
                total: 200,
            },
            queue_reqs: 0,
            ..batch_status()
        };
        assert_eq!(reporter.report_decode(1.0, 2, &status), None);
        assert_eq!(reporter.report_decode(1.5, 2, &status), None);

        let status = BatchStatus {
            queue_reqs: 4,
            kv_pages: PoolUsage {
                used: 62,
                total: 200,
            },
            ..status
        };
        let line = reporter
            .report_decode(2.0, 2, &status)
            .expect("the third forward logs");
        assert!(line.starts_with("Decode batch, "), "{line}");
        assert!(line.contains("#running-req: 2, "), "{line}");
        assert!(line.contains("#token: 992, "), "62 pages of 16: {line}");
        assert!(line.contains("token usage: 0.31, "), "{line}");
        assert!(line.contains("#queue-req: 4"), "{line}");
        assert!(
            line.contains("gen throughput (token/s): 3.00"),
            "6 tokens over 2.0s: {line}"
        );
    }

    /// Decode throughput covers the interval since the last emitted
    /// line, not the run. A lifetime average -- the naive way -- would
    /// report 4.00 here (16 tokens over 4.0s) and could never show the
    /// slowdown from 5 tok/s to 3 tok/s that this test contains, which
    /// is the only thing an operator watches a decode log for.
    #[test]
    fn decode_throughput_covers_the_interval_and_not_the_whole_run() {
        let mut reporter = StatusReporter::new(2, 0.0);
        let status = BatchStatus {
            kv_pages: PoolUsage { used: 1, total: 10 },
            page_size: 1,
            ..batch_status()
        };
        assert_eq!(reporter.report_decode(1.0, 5, &status), None);
        let line = reporter.report_decode(2.0, 5, &status).unwrap();
        assert!(
            line.contains("gen throughput (token/s): 5.00"),
            "10 tokens over 2.0s: {line}"
        );

        assert_eq!(reporter.report_decode(3.0, 3, &status), None);
        let line = reporter.report_decode(4.0, 3, &status).unwrap();
        assert!(
            line.contains("gen throughput (token/s): 3.00"),
            "6 tokens over the 2.0s since the last line: {line}"
        );
        assert!(
            !line.contains("gen throughput (token/s): 4.00"),
            "a lifetime average would read 16 tokens over 4.0s: {line}"
        );
    }

    /// A server that has allocated no pool at all, reported at the
    /// instant it started, still produces a line rather than a NaN or a
    /// division by zero.
    #[test]
    fn a_zero_gap_and_an_unallocated_pool_still_render() {
        let mut reporter = StatusReporter::new(1, 0.0);
        let status = BatchStatus {
            kv_pages: PoolUsage::default(),
            page_size: 1,
            ..batch_status()
        };
        let line = reporter.report_decode(0.0, 4, &status).unwrap();
        assert!(line.contains("gen throughput (token/s): 0.00"), "{line}");
        assert!(line.contains("token usage: 0.00"), "{line}");
        assert!(line.contains("#token: 0, "), "{line}");

        let line = reporter.report_prefill(0.0, &PrefillSnapshot::default(), &status);
        assert!(line.contains("input throughput (token/s): 0.00"), "{line}");
    }

    /// An interval of zero means "log every forward", not "divide by
    /// zero on the first one".
    #[test]
    fn the_decode_interval_is_clamped_to_at_least_one_forward() {
        let mut reporter = StatusReporter::new(0, 0.0);
        assert_eq!(reporter.decode_log_interval(), 1);
        assert!(reporter.report_decode(1.0, 1, &batch_status()).is_some());
    }

    /// The prefill line reads the schedule-time snapshot. Reading the
    /// live requests instead -- the naive way -- reports the state the
    /// forward has already advanced them to: one "new" token per
    /// request and the whole computed prefix as cached. Neither of those
    /// numbers may appear in the line.
    #[test]
    fn a_prefill_line_reports_the_schedule_time_snapshot() {
        let mut pass = PrefillPass::new(512, 0, geometry());
        let request = PendingRequest::new(1, 1000, 100);
        let chunk = pass.take_chunk(&request, 300, 0, &roomy()).unwrap();
        let snapshot = PrefillSnapshot::from_chunks(&[chunk]);
        assert_eq!(snapshot.new_seqs, 1);
        assert_eq!(snapshot.new_tokens, 512);
        assert_eq!(snapshot.cached_tokens, 300);

        // What the request looks like by the time the batch is reported:
        // the forward has advanced its cached length to its device
        // length, and it now produces one token per step.
        let post_forward_new_tokens = 1;
        let post_forward_cached_tokens = chunk.computed_len + chunk.chunk_len;
        assert_eq!(post_forward_cached_tokens, 812);

        let mut reporter = StatusReporter::new(DEFAULT_DECODE_LOG_INTERVAL, 0.0);
        let line = reporter.report_prefill(0.5, &snapshot, &batch_status());
        assert!(line.starts_with("Prefill batch, "), "{line}");
        assert!(line.contains("#new-seq: 1, "), "{line}");
        assert!(line.contains("#new-token: 512, "), "{line}");
        assert!(line.contains("#cached-token: 300, "), "{line}");
        assert!(
            !line.contains(&format!("#new-token: {post_forward_new_tokens},")),
            "{line}"
        );
        assert!(
            !line.contains(&format!("#cached-token: {post_forward_cached_tokens},")),
            "{line}"
        );
        assert!(line.contains("token usage: 0.25, "), "{line}");
        assert!(line.contains("#running-req: 2, "), "{line}");
        assert!(line.contains("#queue-req: 1, "), "{line}");
        assert!(
            line.contains("input throughput (token/s): 1024.00"),
            "512 tokens over 0.5s: {line}"
        );
    }

    /// A prefix hit belongs to a prompt, not to a chunk: a continuation
    /// contributes its tokens to `#new-token` and nothing at all to
    /// `#cached-token`, which it would otherwise re-report on every
    /// chunk of a long prompt.
    #[test]
    fn a_prefix_hit_is_reported_once_across_a_prompts_chunks() {
        let mut pass = PrefillPass::new(1024, 0, geometry());
        let fresh = PendingRequest::new(1, 600, 100);
        let first = pass.take_chunk(&fresh, 200, 0, &roomy()).unwrap();
        assert_eq!(first.chunk_len, 400);

        let continued = PendingRequest {
            chunk: Some(ChunkState {
                computed_len: 512,
                slot: 1,
                swa_evicted_len: 0,
                locked_prefix_len: 0,
            }),
            ..PendingRequest::new(2, 2000, 100)
        };
        let second = pass.take_chunk(&continued, 0, 1, &roomy()).unwrap();
        assert_eq!(second.chunk_len, 624);
        assert_eq!(second.admission, None);

        let snapshot = PrefillSnapshot::from_chunks(&[first, second]);
        assert_eq!(snapshot.new_seqs, 2, "continuations are batch entries");
        assert_eq!(snapshot.new_tokens, 1024, "and their tokens are computed");
        assert_eq!(
            snapshot.cached_tokens, 200,
            "only the first chunk's prompt reports a hit"
        );
    }
}
