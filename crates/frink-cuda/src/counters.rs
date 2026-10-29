//! How many times one decode token (or one prefill step) crosses the
//! bus and waits for the device, counted where it happens.
//!
//! #133 and #259 each spent a day on a host-side theory that one
//! measurement would have settled, and the per-token inventory of the
//! decode path was first counted by READING code, which is exactly the kind of claim
//! this repo has been wrong about before. This is the measurement: every
//! host-to-device copy, device-to-host copy, device-to-device copy and
//! host wait in `frink-cuda` goes through [`crate::xfer`], which records
//! it here, and the decoder marks the end of each step. With
//! `FRINK_CUDA_COUNT=1` the per-step difference is printed to stderr
//! the first time a step of its kind is seen and again whenever it
//! changes, so a steady decode prints one line and a regression prints
//! a second.
//!
//! Always compiled: the counters are plain atomics and the reporting
//! is plain arithmetic, so both are tested on a host with no GPU. The
//! recording costs a relaxed atomic add per transfer whether or not the
//! report is on, so a counted run measures the same binary as an
//! uncounted one.
//!
//! What it does NOT count: kernel launches (there are a few dozen call
//! sites and none of them waits), and `frink-core`'s compile-only
//! expert pool, which talks to `cudarc` directly and has never run.
//! The counters are process-wide, so two requests decoding at once
//! are counted together.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static H2D: AtomicU64 = AtomicU64::new(0);
static H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static D2H: AtomicU64 = AtomicU64::new(0);
static D2H_BYTES: AtomicU64 = AtomicU64::new(0);
static D2D: AtomicU64 = AtomicU64::new(0);
static SYNCS: AtomicU64 = AtomicU64::new(0);

/// One host-to-device copy of `bytes`. Does not wait.
pub fn record_h2d(bytes: usize) {
    H2D.fetch_add(1, Ordering::Relaxed);
    H2D_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// One device-to-host copy of `bytes`. Every one of these waits for the
/// stream (`cudarc`'s `dtoh_sync_copy`), so it is a sync as well, and
/// is recorded as one: a caller cannot download without the wait
/// showing up in [`Counts::syncs`].
pub fn record_d2h(bytes: usize) {
    D2H.fetch_add(1, Ordering::Relaxed);
    D2H_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    record_sync();
}

/// One device-to-device copy. Enqueued; does not wait.
pub fn record_d2d() {
    D2D.fetch_add(1, Ordering::Relaxed);
}

/// One host wait on the stream that is not a download.
pub fn record_sync() {
    SYNCS.fetch_add(1, Ordering::Relaxed);
}

/// Totals since the process started, or a difference between two of
/// them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub h2d: u64,
    pub h2d_bytes: u64,
    pub d2h: u64,
    pub d2h_bytes: u64,
    pub d2d: u64,
    pub syncs: u64,
}

impl Counts {
    /// What happened between `earlier` and `self`. Saturating, because
    /// the counters only grow and a caller passing them the wrong way
    /// round should read zeros rather than wrap to 16 exabytes.
    pub fn since(self, earlier: Counts) -> Counts {
        // Exhaustive on both sides: a field added to `Counts` and not
        // to the difference fails to compile here.
        let Counts {
            h2d,
            h2d_bytes,
            d2h,
            d2h_bytes,
            d2d,
            syncs,
        } = self;
        let Counts {
            h2d: h2d0,
            h2d_bytes: h2d_bytes0,
            d2h: d2h0,
            d2h_bytes: d2h_bytes0,
            d2d: d2d0,
            syncs: syncs0,
        } = earlier;
        Counts {
            h2d: h2d.saturating_sub(h2d0),
            h2d_bytes: h2d_bytes.saturating_sub(h2d_bytes0),
            d2h: d2h.saturating_sub(d2h0),
            d2h_bytes: d2h_bytes.saturating_sub(d2h_bytes0),
            d2d: d2d.saturating_sub(d2d0),
            syncs: syncs.saturating_sub(syncs0),
        }
    }
}

/// The totals right now.
pub fn snapshot() -> Counts {
    Counts {
        h2d: H2D.load(Ordering::Relaxed),
        h2d_bytes: H2D_BYTES.load(Ordering::Relaxed),
        d2h: D2H.load(Ordering::Relaxed),
        d2h_bytes: D2H_BYTES.load(Ordering::Relaxed),
        d2d: D2D.load(Ordering::Relaxed),
        syncs: SYNCS.load(Ordering::Relaxed),
    }
}

/// Which kind of step just ended. A decode token and a prefill step
/// have different steady states, so each is compared with its own
/// previous step, never with the other's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    Decode,
    Prefill,
}

impl StepKind {
    fn name(self) -> &'static str {
        match self {
            StepKind::Decode => "decode token",
            StepKind::Prefill => "prefill step",
        }
    }

    fn index(self) -> usize {
        match self {
            StepKind::Decode => 0,
            StepKind::Prefill => 1,
        }
    }
}

/// Turns a stream of totals into the lines worth printing: a step's
/// difference the first time its kind is seen and whenever it differs
/// from that kind's previous step. Pure, so the rule is tested on the
/// host.
#[derive(Debug, Default)]
pub struct StepReporter {
    /// The totals at the end of the previous step of either kind, so a
    /// step is charged only for what happened since then.
    last_total: Counts,
    /// Per kind: how many steps have ended, and the difference the last
    /// one had.
    seen: [(u64, Option<Counts>); 2],
}

impl StepReporter {
    /// A reporter whose first step is charged from `start`.
    pub fn starting_at(start: Counts) -> Self {
        Self {
            last_total: start,
            seen: [(0, None), (0, None)],
        }
    }

    /// Records that a step of `kind` ended with the totals at `now`, and
    /// returns the line to print, if this step is worth one.
    pub fn observe(&mut self, kind: StepKind, now: Counts) -> Option<String> {
        let step = now.since(self.last_total);
        self.last_total = now;
        let (count, prev) = &mut self.seen[kind.index()];
        let index = *count;
        *count += 1;
        let changed = *prev != Some(step);
        *prev = Some(step);
        changed.then(|| format_step(kind, index, step))
    }
}

fn format_step(kind: StepKind, index: u64, c: Counts) -> String {
    let Counts {
        h2d,
        h2d_bytes,
        d2h,
        d2h_bytes,
        d2d,
        syncs,
    } = c;
    format!(
        "frink: cuda {} {index}: syncs={syncs} h2d={h2d} ({h2d_bytes} B) \
         d2h={d2h} ({d2h_bytes} B) d2d={d2d}",
        kind.name()
    )
}

/// `FRINK_CUDA_COUNT=1|true|on` prints the per-step counts. Read once.
pub fn report_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("FRINK_CUDA_COUNT").as_deref(),
            Ok("1" | "true" | "on")
        )
    })
}

static REPORTER: Mutex<Option<StepReporter>> = Mutex::new(None);

/// Marks the end of one step. A no-op unless [`report_enabled`]; when
/// it is, prints the step's counts if [`StepReporter`] says the line is
/// new. The first step is charged from the first call, so the one-time
/// weight uploads of the first token land in that token's line and the
/// steady state shows from the second.
pub fn step_end(kind: StepKind) {
    if !report_enabled() {
        return;
    }
    let now = snapshot();
    let mut guard = REPORTER.lock().unwrap_or_else(|p| p.into_inner());
    let reporter = guard.get_or_insert_with(|| StepReporter::starting_at(Counts::default()));
    if let Some(line) = reporter.observe(kind, now) {
        eprintln!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(h2d: u64, d2h: u64, syncs: u64) -> Counts {
        Counts {
            h2d,
            h2d_bytes: h2d * 100,
            d2h,
            d2h_bytes: d2h * 10,
            d2d: 0,
            syncs,
        }
    }

    /// The point of the reporter is that a steady decode prints ONE
    /// line: if identical steps printed again the log would bury the
    /// one that changed, and if a changed step did not print the
    /// regression this exists to catch would be silent.
    #[test]
    fn a_steady_step_prints_once_and_a_changed_one_prints_again() {
        let mut r = StepReporter::starting_at(Counts::default());
        // First decode token: one-time uploads.
        let l0 = r
            .observe(StepKind::Decode, at(100, 2, 2))
            .expect("first step prints");
        assert!(l0.contains("decode token 0"), "{l0}");
        assert!(l0.contains("h2d=100"), "{l0}");
        // Second: the steady state, a different difference.
        let l1 = r.observe(StepKind::Decode, at(102, 4, 4)).expect("changed");
        assert!(
            l1.contains("decode token 1: syncs=2 h2d=2 (200 B) d2h=2 (20 B)"),
            "{l1}"
        );
        // Third and fourth: the same difference, silent.
        assert_eq!(r.observe(StepKind::Decode, at(104, 6, 6)), None);
        assert_eq!(r.observe(StepKind::Decode, at(106, 8, 8)), None);
        // A regression: one more sync.
        let l4 = r
            .observe(StepKind::Decode, at(108, 10, 11))
            .expect("changed");
        assert!(l4.contains("decode token 4: syncs=3"), "{l4}");
    }

    /// A prefill step between two decode tokens is charged to itself,
    /// and the decode token after it is compared with the previous
    /// decode token, not with the prefill.
    #[test]
    fn each_kind_is_compared_with_its_own_previous_step() {
        let mut r = StepReporter::starting_at(at(10, 0, 0));
        assert!(r.observe(StepKind::Decode, at(12, 2, 2)).is_some());
        let p = r
            .observe(StepKind::Prefill, at(42, 5, 5))
            .expect("first prefill");
        assert!(p.contains("prefill step 0: syncs=3 h2d=30"), "{p}");
        // Same difference as the first decode token: silent.
        assert_eq!(r.observe(StepKind::Decode, at(44, 7, 7)), None);
    }

    /// The counter is only a measurement if nothing goes around it.
    /// Every non-test line of this crate that copies across the bus or
    /// waits on the stream must do it through `crate::xfer`; a new
    /// raw call anywhere else is a step the counts silently miss.
    #[test]
    fn no_transfer_bypasses_the_counted_door() {
        const RAW: &[&str] = &[
            ".htod_copy(",
            ".htod_sync_copy",
            ".dtoh_sync_copy",
            ".dtod_copy(",
            ".synchronize()",
            ".sync_reclaim(",
            "cuMemcpy",
            "cuStreamSynchronize",
        ];
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk(&src, &mut files);
        assert!(files.len() > 10, "found the crate's sources: {files:?}");
        let mut offenders = Vec::new();
        for f in &files {
            let name = f.file_name().unwrap().to_string_lossy();
            if name == "xfer.rs" || name == "tests.rs" {
                continue;
            }
            let text = std::fs::read_to_string(f).unwrap();
            // Everything from the first test module on is test code.
            let code = text.split("#[cfg(test)]").next().unwrap();
            for (i, line) in code.lines().enumerate() {
                let t = line.trim_start();
                if t.starts_with("//") {
                    continue;
                }
                if let Some(raw) = RAW.iter().find(|r| line.contains(*r)) {
                    offenders.push(format!("{}:{}: {raw}", f.display(), i + 1));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "raw transfers outside crate::xfer:\n{}",
            offenders.join("\n")
        );
    }

    #[test]
    fn a_difference_taken_the_wrong_way_round_reads_zero_rather_than_wrapping() {
        assert_eq!(at(1, 1, 1).since(at(5, 5, 5)), Counts::default());
    }

    #[test]
    fn a_download_is_counted_as_a_sync_too() {
        let before = snapshot();
        record_d2h(64);
        record_h2d(32);
        record_d2d();
        let d = snapshot().since(before);
        // Other tests in this binary record concurrently, so these are
        // lower bounds.
        assert!(
            d.d2h >= 1 && d.syncs >= 1 && d.h2d >= 1 && d.d2d >= 1,
            "{d:?}"
        );
        assert!(d.d2h_bytes >= 64 && d.h2d_bytes >= 32, "{d:?}");
    }
}
