//! Where a Metal submission's time goes, measured rather than inferred.
//!
//! Two instruments, both opt-in by environment variable and both
//! process-wide:
//!
//! - `FRINK_METAL_MM_TIMING`: prefill GEMM wall-clock totals
//!   (setup / GPU wait / readback), see [`mm_timing_add`].
//! - `FRINK_METAL_GPU_TIMING`: per-tag GPU-clock time of each command
//!   buffer, the host's encode time and submit latency around it, and
//!   the encode work that bought it, see [`commit_wait_note`].
//!
//! Moved out of `gpu.rs` along the seam GitHub issue #149 touches, and
//! then given the two host phases it lacked, because lacking them is
//! what let that issue mistake a second command buffer's GPU time for
//! host time.

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;
use std::sync::atomic::{AtomicU64, Ordering as AtomOrd};
use std::time::{Duration, Instant};
static MM_SETUP_US: AtomicU64 = AtomicU64::new(0);
static MM_GPU_US: AtomicU64 = AtomicU64::new(0);
static MM_READ_US: AtomicU64 = AtomicU64::new(0);
static MM_CALLS: AtomicU64 = AtomicU64::new(0);

/// Accumulate into the `FRINK_METAL_MM_TIMING` counters (prefill evidence).
pub(crate) fn mm_timing_add(setup: u128, gpu: u128, read: u128) {
    MM_SETUP_US.fetch_add(setup as u64, AtomOrd::Relaxed);
    MM_GPU_US.fetch_add(gpu as u64, AtomOrd::Relaxed);
    MM_READ_US.fetch_add(read as u64, AtomOrd::Relaxed);
    let n = MM_CALLS.fetch_add(1, AtomOrd::Relaxed) + 1;
    // First call + every 224: stack prefill is often one timed launch.
    if n == 1 || n.is_multiple_of(224) {
        eprintln!(
            "frink: mul_mm {n} calls -- setup {:.1} ms, gpu {:.1} ms, readback {:.1} ms",
            MM_SETUP_US.load(AtomOrd::Relaxed) as f64 / 1000.0,
            MM_GPU_US.load(AtomOrd::Relaxed) as f64 / 1000.0,
            MM_READ_US.load(AtomOrd::Relaxed) as f64 / 1000.0,
        );
    }
}

/// Per-tag accumulators for `FRINK_METAL_GPU_TIMING`.
///
/// Wall-clock (`FRINK_METAL_MM_TIMING`) measures how long the host waited,
/// which on a loaded host is mostly scheduler noise. `MTLCommandBuffer`'s
/// `GPUStartTime`/`GPUEndTime` measure the command buffer's own occupancy
/// and stay usable while the machine is busy, so A/B evidence in
/// `4dc195e5:docs/plans/llama-cpp-parity-push.md` is taken from these.
///
/// GPU time alone was how GitHub issue #149 mis-read its own numbers:
/// "host = wall minus GPU" charged to the host everything that was not
/// the ONE timed command buffer, including a second, untimed one (the
/// lm_head, `launch_matvec_fused`) that runs on the GPU for 1.2 ms per
/// token on Llama-3.2-1B and 2.8 ms on Gemma-2-2B. So every submission
/// now reports three phases from the same clock, and a phase that is
/// not measured is not attributed to anything:
///
/// - **encode**: from [`SubmitClock::start`] to `commit`, the host
///   building the command buffer;
/// - **gpu**: `GPUEndTime - GPUStartTime`, the GPU executing it;
/// - **latency**: the wall time from `commit` to `waitUntilCompleted`
///   returning, MINUS the GPU time -- submit-to-start plus
///   end-to-wakeup, the part no lever on encoding can touch.
static GPU_TIMING: std::sync::Mutex<Ledger> = std::sync::Mutex::new(Ledger::new());

/// Every tag's slot, plus the encode-counter snapshot at the last noted
/// submission OF ANY TAG.
///
/// The encode counters in [`crate::dispatch`] are process-wide, so a
/// per-submission figure has to come from a DELTA. The delta used to be
/// taken per tag, between consecutive submissions of the same tag, and
/// that is wrong as soon as two tags alternate: the dense stack and its
/// lm_head run once each per token, so the lm_head's line reported the
/// stack's 210 dispatches as its own. One snapshot for the whole ledger
/// attributes each dispatch to the next submission noted after it.
struct Ledger {
    slots: Vec<TimingSlot>,
    prev: crate::dispatch::EncodeStats,
}

impl Ledger {
    const fn new() -> Self {
        Self {
            slots: Vec::new(),
            prev: crate::dispatch::EncodeStats {
                dispatches: 0,
                barriers: 0,
                begin_ops: 0,
            },
        }
    }

    /// Account one submission and, every `every` submissions under its
    /// tag, return the figures to print.
    fn note(&mut self, tag: &'static str, sample: Sample, every: u64) -> Option<Report> {
        let now = sample.encode_stats;
        let delta = crate::dispatch::EncodeStats {
            dispatches: now.dispatches.saturating_sub(self.prev.dispatches),
            barriers: now.barriers.saturating_sub(self.prev.barriers),
            begin_ops: now.begin_ops.saturating_sub(self.prev.begin_ops),
        };
        self.prev = now;
        let slot = match self.slots.iter_mut().position(|s| s.tag == tag) {
            Some(i) => &mut self.slots[i],
            None => {
                self.slots.push(TimingSlot::new(tag));
                self.slots.last_mut().expect("just pushed")
            }
        };
        slot.record(sample, delta);
        if slot.n.is_multiple_of(every.max(1)) {
            Some(slot.report())
        } else {
            None
        }
    }
}

/// One submission's three phases plus the counters at its completion.
#[derive(Clone, Copy)]
struct Sample {
    gpu_ns: u64,
    encode_ns: u64,
    wait_ns: u64,
    encode_stats: crate::dispatch::EncodeStats,
}

/// One tag's running totals, plus what the host encoded to produce them.
struct TimingSlot {
    tag: &'static str,
    n: u64,
    /// GPU-clock nanoseconds.
    acc_gpu_ns: u64,
    /// Host wall nanoseconds from encode start to commit.
    acc_encode_ns: u64,
    /// Host wall nanoseconds from commit to the wait returning.
    acc_wait_ns: u64,
    /// The same three, over the submissions since the last report only.
    /// A cumulative mean over a decode run carries every priming token
    /// and every cold buffer in it; the steady state is the window.
    win_gpu_ns: u64,
    win_encode_ns: u64,
    win_wait_ns: u64,
    win_n: u64,
    /// GPU nanoseconds of the most recent submission.
    last_gpu_ns: u64,
    /// Encode work attributed to this tag, summed over `n` submissions.
    acc_dispatches: u64,
    acc_barriers: u64,
    acc_begin_ops: u64,
}

/// The per-submission means a report line prints, in milliseconds.
#[derive(Debug, Clone, PartialEq)]
struct Report {
    tag: &'static str,
    n: u64,
    gpu_ms: f64,
    last_gpu_ms: f64,
    encode_ms: f64,
    latency_ms: f64,
    win_n: u64,
    win_gpu_ms: f64,
    win_encode_ms: f64,
    win_latency_ms: f64,
    dispatches: f64,
    barriers: f64,
    begin_ops: f64,
}

impl TimingSlot {
    fn new(tag: &'static str) -> Self {
        Self {
            tag,
            n: 0,
            acc_gpu_ns: 0,
            acc_encode_ns: 0,
            acc_wait_ns: 0,
            win_gpu_ns: 0,
            win_encode_ns: 0,
            win_wait_ns: 0,
            win_n: 0,
            last_gpu_ns: 0,
            acc_dispatches: 0,
            acc_barriers: 0,
            acc_begin_ops: 0,
        }
    }

    fn record(&mut self, s: Sample, delta: crate::dispatch::EncodeStats) {
        self.n += 1;
        self.acc_gpu_ns += s.gpu_ns;
        self.acc_encode_ns += s.encode_ns;
        self.acc_wait_ns += s.wait_ns;
        self.win_gpu_ns += s.gpu_ns;
        self.win_encode_ns += s.encode_ns;
        self.win_wait_ns += s.wait_ns;
        self.win_n += 1;
        self.acc_dispatches += delta.dispatches;
        self.acc_barriers += delta.barriers;
        self.acc_begin_ops += delta.begin_ops;
        self.last_gpu_ns = s.gpu_ns;
    }

    /// The report for the state so far. Resets the window.
    fn report(&mut self) -> Report {
        let n = self.n.max(1);
        let per = |acc: u64| acc as f64 / n as f64;
        let ms = |acc: u64| per(acc) / 1e6;
        let win_n = self.win_n.max(1);
        let win_ms = |acc: u64| acc as f64 / win_n as f64 / 1e6;
        let r = Report {
            tag: self.tag,
            n: self.n,
            gpu_ms: ms(self.acc_gpu_ns),
            last_gpu_ms: self.last_gpu_ns as f64 / 1e6,
            encode_ms: ms(self.acc_encode_ns),
            latency_ms: (ms(self.acc_wait_ns) - ms(self.acc_gpu_ns)).max(0.0),
            win_n: self.win_n,
            win_gpu_ms: win_ms(self.win_gpu_ns),
            win_encode_ms: win_ms(self.win_encode_ns),
            win_latency_ms: (win_ms(self.win_wait_ns) - win_ms(self.win_gpu_ns)).max(0.0),
            dispatches: per(self.acc_dispatches),
            barriers: per(self.acc_barriers),
            begin_ops: per(self.acc_begin_ops),
        };
        self.win_gpu_ns = 0;
        self.win_encode_ns = 0;
        self.win_wait_ns = 0;
        self.win_n = 0;
        r
    }
}

impl Report {
    fn print(&self) {
        eprintln!(
            "frink: metal gpu[{}] {:.3} ms avg over {} (last {:.3} ms); \
             host/submission: encode {:.3} ms, latency beyond gpu {:.3} ms; \
             last {}: gpu {:.3} ms, encode {:.3} ms, latency {:.3} ms; \
             encode/submission: {:.1} dispatches, {:.1} barriers, \
             {:.1} hazard checks ({:.2} bar/op)",
            self.tag,
            self.gpu_ms,
            self.n,
            self.last_gpu_ms,
            self.encode_ms,
            self.latency_ms,
            self.win_n,
            self.win_gpu_ms,
            self.win_encode_ms,
            self.win_latency_ms,
            self.dispatches,
            self.barriers,
            self.begin_ops,
            self.barriers / self.begin_ops.max(f64::MIN_POSITIVE),
        );
    }
}

/// True when `FRINK_METAL_GPU_TIMING` is set (cached; read once).
pub(crate) fn gpu_timing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FRINK_METAL_GPU_TIMING").is_some())
}

/// The host-side clock of one command buffer, started before its first
/// encode call. Consumed by [`commit_wait_note`], which is the only way
/// to finish it: a submission cannot be timed with the encode phase
/// left out, because leaving it out is how the phase was mis-attributed.
#[must_use = "a SubmitClock that is not passed to commit_wait_note times nothing"]
pub(crate) struct SubmitClock {
    encode_start: Instant,
}

impl SubmitClock {
    pub(crate) fn start() -> Self {
        Self {
            encode_start: Instant::now(),
        }
    }
}

/// Commit `cmd_buf`, wait for it, and account its three phases under
/// `tag`, logging a running mean every `every` submissions. Returns how
/// long the host waited (commit to completion), for callers that keep a
/// wall-clock total of their own.
/// Waits for a command buffer that was committed EARLIER, and notes it.
///
/// [`commit_wait_note`] commits and waits together, which is right when
/// one buffer is one unit of work. A RUN of buffers committed back to
/// back is one unit of work spread over several, and waiting for the
/// last one waits for all of them: the GPU time noted is that last
/// buffer's, and `count` says how many submissions shared the single
/// wake-up, which is the number the run exists to reduce.
pub(crate) fn note_wait(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    tag: &'static str,
    count: usize,
    every: u64,
    clock: SubmitClock,
) -> Duration {
    let committed = Instant::now();
    cmd_buf.waitUntilCompleted();
    let waited = committed.elapsed();
    if !gpu_timing_enabled() || count == 0 {
        return waited;
    }
    let sample = Sample {
        gpu_ns: ((cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e9).max(0.0) as u64,
        encode_ns: committed.duration_since(clock.encode_start).as_nanos() as u64,
        wait_ns: waited.as_nanos() as u64,
        encode_stats: crate::dispatch::metal_encode_stats(),
    };
    let report = match GPU_TIMING.lock() {
        Ok(mut ledger) => ledger.note(tag, sample, every),
        Err(_) => None,
    };
    if let Some(r) = report {
        r.print();
    }
    waited
}

pub(crate) fn commit_wait_note(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    tag: &'static str,
    every: u64,
    clock: SubmitClock,
) -> Duration {
    let committed = Instant::now();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    let waited = committed.elapsed();
    if !gpu_timing_enabled() {
        return waited;
    }
    let sample = Sample {
        gpu_ns: ((cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e9).max(0.0) as u64,
        encode_ns: committed.duration_since(clock.encode_start).as_nanos() as u64,
        wait_ns: waited.as_nanos() as u64,
        encode_stats: crate::dispatch::metal_encode_stats(),
    };
    let report = match GPU_TIMING.lock() {
        Ok(mut ledger) => ledger.note(tag, sample, every),
        Err(_) => None,
    };
    if let Some(r) = report {
        r.print();
    }
    waited
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::EncodeStats;

    fn stats(dispatches: u64, barriers: u64) -> EncodeStats {
        EncodeStats {
            dispatches,
            barriers,
            begin_ops: barriers,
        }
    }

    fn sample(gpu_ms: f64, encode_ms: f64, wait_ms: f64, s: EncodeStats) -> Sample {
        Sample {
            gpu_ns: (gpu_ms * 1e6) as u64,
            encode_ns: (encode_ms * 1e6) as u64,
            wait_ns: (wait_ms * 1e6) as u64,
            encode_stats: s,
        }
    }

    /// Two command buffers per token, one tag each. The lm_head's line
    /// used to report the dense stack's 210 dispatches, because the
    /// delta was taken between consecutive submissions of the SAME tag
    /// and the stack ran in between. Each submission gets exactly the
    /// work encoded since the previous noted submission, whatever its
    /// tag was.
    #[test]
    fn dispatches_are_attributed_to_the_submission_that_encoded_them() {
        let mut l = Ledger::new();
        // Counters are process-wide and monotone: 210 for the stack,
        // then 1 more for the lm_head, per token.
        let mut d = 0;
        for _ in 0..3 {
            d += 210;
            let _ = l.note("stack", sample(5.0, 0.1, 5.2, stats(d, 160)), 3);
            d += 1;
            let _ = l.note("head", sample(1.0, 0.02, 1.2, stats(d, 160)), 3);
        }
        let stack = l
            .slots
            .iter_mut()
            .find(|s| s.tag == "stack")
            .unwrap()
            .report();
        let head = l
            .slots
            .iter_mut()
            .find(|s| s.tag == "head")
            .unwrap()
            .report();
        assert_eq!(stack.dispatches, 210.0);
        assert_eq!(
            head.dispatches, 1.0,
            "the head encodes one matvec, not the stack's 210"
        );
    }

    /// The cumulative mean carries every priming token in it. The window
    /// is the steady state, so it must start over after each report
    /// while the cumulative figures keep counting.
    #[test]
    fn the_window_resets_at_each_report_and_the_cumulative_mean_does_not() {
        let mut l = Ledger::new();
        // A 100 ms cold first submission, then 3 warm ones at 5 ms.
        assert!(l
            .note("t", sample(100.0, 10.0, 101.0, stats(1, 1)), 2)
            .is_none());
        let first = l.note("t", sample(5.0, 0.1, 5.2, stats(2, 2)), 2).unwrap();
        assert_eq!(first.win_n, 2);
        assert!((first.win_gpu_ms - 52.5).abs() < 1e-9);
        assert!(l.note("t", sample(5.0, 0.1, 5.2, stats(3, 3)), 2).is_none());
        let second = l.note("t", sample(5.0, 0.1, 5.2, stats(4, 4)), 2).unwrap();
        assert_eq!(second.n, 4);
        assert_eq!(
            second.win_n, 2,
            "the window holds only the two since the last report"
        );
        assert!(
            (second.win_gpu_ms - 5.0).abs() < 1e-9,
            "warm steady state, no cold token in it"
        );
        assert!(
            (second.gpu_ms - 28.75).abs() < 1e-9,
            "cumulative still carries the cold one"
        );
        assert!((second.win_encode_ms - 0.1).abs() < 1e-9);
    }

    /// Latency is what the wait spent beyond the GPU's own occupancy,
    /// i.e. submit-to-start plus end-to-wakeup. Reported per phase, not
    /// folded into "host", because folding it in is the mis-attribution
    /// this module exists to prevent; and never negative, since a wait
    /// cannot return before the GPU is done.
    #[test]
    fn latency_is_the_wait_beyond_gpu_time_and_never_negative() {
        let mut l = Ledger::new();
        let r = l.note("t", sample(4.0, 0.15, 4.4, stats(1, 1)), 1).unwrap();
        assert!((r.latency_ms - 0.4).abs() < 1e-9);
        assert!((r.win_latency_ms - 0.4).abs() < 1e-9);
        assert!((r.encode_ms - 0.15).abs() < 1e-9);
        // A clock skew that puts GPU time above the wait clamps to zero
        // rather than reporting a negative host cost.
        let r = l.note("u", sample(4.0, 0.1, 3.9, stats(2, 2)), 1).unwrap();
        assert_eq!(r.latency_ms, 0.0);
    }
}
