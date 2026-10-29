//! Sealed kernel-lookup registry: makes a *missing* kernel loud instead
//! of silently slow.
//!
//! ## Why this exists
//!
//! The worst bug class this engine has shipped is not a wrong kernel but
//! an absent one, silently replaced by a correct-but-slow fallback. Two
//! real examples:
//!
//! - IQ4_XS batched prefill ran on the **CPU** because
//!   [`crate::weight_matrix::WeightMatrix`]'s `metal_kind_supported`
//!   predicate and `apply_gpu_batch`'s per-kind dispatch table disagreed
//!   about one kind. The only symptom was a benchmark row 13.7x behind.
//! - Gemma-4-E2B is slower on Metal than on CPU. Output is correct, so
//!   nothing fails; the model simply never reaches a batched path.
//!
//! Both are *lookups that missed*. Neither produced a diagnostic.
//!
//! ## The shape
//!
//! Every dispatch decision that asks "is there a kernel for this
//! (backend, op, quant kind)?" records the answer here.
//!
//! - **Build phase.** While the model is constructed, each weight matrix
//!   is probed eagerly ([`crate::weight_matrix::WeightMatrix::probe_kernels`]):
//!   the same predicates the hot path will consult are evaluated once per
//!   weight and recorded. This is what turns "why is this row slow" into
//!   a startup line.
//! - **[`seal`].** Called once the model is loaded. It summarises the
//!   build picture, warns about quantized weights that will run off the
//!   selected accelerator, and switches on post-seal checking.
//! - **Run phase.** After sealing, a dispatch-site lookup that misses a
//!   *combination the build probe never saw* is by definition an
//!   unpredicted slow path. It warns once, loudly, naming the call site
//!   ([`#[track_caller]`](std::panic::Location)) and the quant kind, and
//!   is a hard error under `FRINK_STRICT_KERNELS=1` so CI and benchmarks
//!   can run closed.
//!
//! ## Cost
//!
//! This sits next to dispatch, so it follows the `OnceLock` discipline
//! used by `metal_dense_enabled` / `min_task_macs`: the environment is
//! read exactly once per process, never per dispatch.
//!
//! On the hot path the added cost is **zero instructions on a hit** —
//! hits are only ever recorded by the build probe, never by a dispatch
//! site. A dispatch site records only when it is *about to take a
//! fallback*, i.e. only when it is already paying orders of magnitude
//! more than the bookkeeping costs.
//!
//! And that bookkeeping never takes an exclusive lock in steady state: a
//! repeat lookup is a shared `RwLock` read plus one relaxed
//! `fetch_add`. The write lock is taken once per distinct call site, to
//! create the row and decide whether to warn. A dispatch path must not
//! be able to serialize rayon workers behind a mutex.
//!
//! `FRINK_KERNEL_REGISTRY=0` reduces every entry point to one relaxed
//! atomic load and a return.
//!
//! ## What it must not do
//!
//! Observe only. Nothing in this module may change a dispatch decision;
//! the predicates it calls are the same ones the dispatch takes, read a
//! second time, and their results are recorded rather than acted on.

use crate::weight_matrix::QuantKind;
use std::collections::{HashMap, HashSet};
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

/// Which execution backend a lookup was resolved against.
///
/// **The GPU variants are generated**, from
/// [`crate::weight_matrix::gpu_backend::gpu_backend_table`] — the same
/// ordered list `WeightMatrix::apply_gpu` dispatches over and
/// `active_backend` reads. That is the beachhead verdict's point 4:
/// "a backend cannot be dispatched to without being reportable". It
/// used to be a hand-kept list beside the dispatch order, i.e. two
/// structures that had to agree about one thing with nothing enforcing
/// it, which is the failure this file exists to catch in *kernels* and
/// was quietly repeating in *backends*.
///
/// `Cpu` is written out rather than generated: it is the fallthrough
/// every accelerator miss lands on, not a backend anything dispatches
/// to, and it has no row in the table for the same reason it has no
/// cargo feature.
///
/// This works only because the registry and the seam are one crate. A
/// `macro_rules!` cannot add a variant to an enum in another crate, so
/// an out-of-tree backend could not own its own variant; if one is ever
/// wanted, this becomes a trait object or a `&'static str`, not a
/// second list.
macro_rules! define_backend_enum {
    ([] $(($variant:ident, $name:literal, $ty:ident)),* $(,)?) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
        pub enum Backend {
            Cpu,
            $($variant,)*
        }

        impl Backend {
            pub fn as_str(self) -> &'static str {
                match self {
                    Backend::Cpu => "cpu",
                    $(Backend::$variant => $name,)*
                }
            }

            /// Every backend this build knows about, `Cpu` first. Lets a
            /// test iterate the set instead of restating it.
            pub const ALL: &'static [Backend] = &[Backend::Cpu, $(Backend::$variant,)*];
        }
    };
}
crate::weight_matrix::gpu_backend::gpu_backend_table!(define_backend_enum);

impl Backend {
    /// True for the accelerator backends. A miss here means work the
    /// user asked to run on a GPU is running somewhere else.
    pub fn is_accelerator(self) -> bool {
        !matches!(self, Backend::Cpu)
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Canonical op names. Free-form `&'static str` is accepted everywhere,
/// but sticking to these keeps the report groupable.
pub mod op {
    /// One activation against a whole weight matrix (decode).
    pub const MATVEC: &str = "matvec";
    /// Several independent matvecs sharing one activation (fused q/k/v).
    pub const MATVEC_MULTI: &str = "matvec_multi";
    /// A real batched GEMM over `batch` activations (prefill).
    pub const GEMM_PREFILL: &str = "gemm_prefill";
    /// Whole gate/up/down SwiGLU fused into one dispatch.
    pub const FFN_SWIGLU: &str = "ffn_swiglu";
    /// An engine-level capability rather than a per-tensor kernel: does
    /// this engine have a batched prefill path at all?
    pub const ENGINE_PREFILL_BATCH: &str = "engine.prefill_batch";
}

/// The names of the host paths an accelerator miss lands on. A
/// fallback named here leaves the device; any other fallback name stays
/// on it (the per-position matvec loops). The kernel coverage matrix
/// (`crate::kernel_matrix`) reads that split from here, and its tests
/// hold every backend's `GEMM_FALLBACK` to it, so a new host path
/// cannot be counted as staying on the GPU.
pub mod host_fallback {
    /// A matvec with no accelerator kernel: `WeightMatrix::apply_cpu`.
    pub const MATVEC: &str = "CPU apply_cpu";
    /// A batched matmul with no accelerator path at all.
    pub const BATCH: &str = "CPU apply_batch";
    /// Every name above.
    pub const ALL: &[&str] = &[MATVEC, BATCH];

    /// Whether a fallback named `name` runs on the host.
    pub fn is_host(name: &str) -> bool {
        ALL.contains(&name)
    }
}

/// The identity of one lookup. Deduplicated on this; a repeated lookup
/// bumps [`Entry::count`] instead of adding a row.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Key {
    pub backend: Backend,
    /// See [`op`].
    pub op: &'static str,
    /// Tensor role (`"attn_q"`, `"ffn_down"`, `"moe_router"`, ...) or
    /// `"(dispatch)"` when recorded from the hot path, where the role is
    /// not known.
    pub role: &'static str,
    /// `None` for a non-quantized (F32 / MXFP4-pair) matrix.
    pub kind: Option<QuantKind>,
    pub file: &'static str,
    pub line: u32,
}

impl Key {
    /// The part of the identity that decides whether a kernel exists.
    /// Call site and tensor role are diagnostics, not part of the
    /// question being asked.
    fn shape(&self) -> (Backend, &'static str, Option<QuantKind>) {
        (self.backend, self.op, self.kind)
    }

    fn kind_name(&self) -> &'static str {
        self.kind.map_or("f32", QuantKind::name)
    }
}

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {} ({}) at {}:{}",
            self.backend,
            self.op,
            self.kind_name(),
            self.role,
            self.file,
            self.line
        )
    }
}

/// How bad a miss is. Recorded at the call site, which is the only
/// place that knows — inferring it at report time from the kind or the
/// backend is how a signal turns into noise nobody reads.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Severity {
    /// The fallback *is* the intended path. An MoE router is a lone
    /// small F32 matvec that costs more to ship to a GPU than to compute
    /// on the host; that is a decision, not an omission.
    ByDesign,
    /// No kernel exists, so the work runs somewhere slower than the
    /// backend the user selected, with correct output and no symptom.
    /// This is the class the registry exists to surface.
    SlowPath,
}

/// What the lookup resolved to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Outcome {
    /// A real kernel exists and will be used.
    Hit,
    /// No kernel; the caller takes `fallback` instead. The string is the
    /// fallback's name, so the report reads as a sentence.
    Miss {
        fallback: &'static str,
        severity: Severity,
    },
}

impl Outcome {
    /// A miss that is a silent slow path.
    pub fn slow_path(fallback: &'static str) -> Self {
        Outcome::Miss {
            fallback,
            severity: Severity::SlowPath,
        }
    }

    /// A miss whose fallback is the deliberate, documented choice.
    pub fn by_design(fallback: &'static str) -> Self {
        Outcome::Miss {
            fallback,
            severity: Severity::ByDesign,
        }
    }

    pub fn is_miss(self) -> bool {
        matches!(self, Outcome::Miss { .. })
    }

    pub fn is_slow_path(self) -> bool {
        matches!(
            self,
            Outcome::Miss {
                severity: Severity::SlowPath,
                ..
            }
        )
    }

    pub fn fallback(self) -> Option<&'static str> {
        match self {
            Outcome::Miss { fallback, .. } => Some(fallback),
            Outcome::Hit => None,
        }
    }
}

/// Whether the lookup happened while the model was being built or after
/// [`seal`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    Build,
    Run,
}

/// A lookup being reported, minus the call site (which `#[track_caller]`
/// supplies).
#[derive(Clone, Copy, Debug)]
pub struct Lookup {
    pub backend: Backend,
    pub op: &'static str,
    pub role: &'static str,
    pub kind: Option<QuantKind>,
}

impl Lookup {
    pub fn new(backend: Backend, op: &'static str, kind: Option<QuantKind>) -> Self {
        Lookup {
            backend,
            op,
            role: "(dispatch)",
            kind,
        }
    }

    pub fn with_role(mut self, role: &'static str) -> Self {
        self.role = role;
        self
    }
}

/// One deduplicated row of the registry.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub key: Key,
    pub outcome: Outcome,
    pub phase: Phase,
    /// How many lookups collapsed into this row.
    pub count: u64,
}

impl std::fmt::Display for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.outcome {
            Outcome::Hit => write!(f, "hit   {}  x{}", self.key, self.count),
            Outcome::Miss {
                fallback,
                severity: Severity::ByDesign,
            } => write!(f, "host  {} -> {}  x{}", self.key, fallback, self.count),
            Outcome::Miss {
                fallback,
                severity: Severity::SlowPath,
            } => write!(f, "MISS  {} -> {}  x{}", self.key, fallback, self.count),
        }
    }
}

/// The picture at [`seal`] time.
#[derive(Clone, Debug, Default)]
pub struct SealReport {
    /// Every build-phase row, sorted for stable printing.
    pub entries: Vec<Entry>,
    /// Build-phase rows whose outcome was a miss, of either severity.
    pub misses: Vec<Entry>,
    /// The subset of `misses` that is a real silent slow path
    /// ([`Severity::SlowPath`]) on an *accelerator* backend the process
    /// actually selected — work the user asked to run on a GPU that will
    /// not. A CPU-backend miss is informational: there is nothing to
    /// fall off.
    pub violations: Vec<Entry>,
}

impl SealReport {
    /// One line per row, for `FRINK_KERNEL_REGISTRY=1`.
    pub fn render(&self) -> String {
        let mut s = String::new();
        for e in &self.entries {
            s.push_str("frink kernels: ");
            s.push_str(&e.to_string());
            s.push('\n');
        }
        s
    }

    /// The loud one-paragraph summary printed whenever a violation
    /// exists, registry verbose or not.
    pub fn render_violations(&self) -> String {
        let mut s = String::new();
        for e in &self.violations {
            // Engine-level capabilities are counted once, not per
            // weight, so do not label their count "weights".
            let unit = if e.key.op.starts_with("engine.") {
                String::new()
            } else {
                format!(", {} weights", e.count)
            };
            let kind = match e.key.kind {
                Some(k) => format!(" {}", k.name()),
                None => String::new(),
            };
            s.push_str(&format!(
                "frink: NO KERNEL for {} {}{} ({}) -> falls back to {} [{}:{}{}]\n",
                e.key.backend,
                e.key.op,
                kind,
                e.key.role,
                e.outcome.fallback().unwrap_or("(hit)"),
                e.key.file,
                e.key.line,
                unit,
            ));
        }
        s
    }
}

/// Raised by [`seal_or_error`] under `FRINK_STRICT_KERNELS=1`.
#[derive(Clone, Debug)]
pub struct StrictKernelError {
    pub report: SealReport,
}

impl std::fmt::Display for StrictKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FRINK_STRICT_KERNELS=1 and {} kernel lookup(s) missed:\n{}",
            self.report.violations.len(),
            self.report.render_violations()
        )
    }
}

impl std::error::Error for StrictKernelError {}

/// One stored row. `count` is atomic so a repeat lookup -- the only
/// thing a dispatch site ever does after the first call -- needs a
/// *shared* read lock and a relaxed increment, never exclusive access.
/// A dispatch path must not be able to serialize rayon workers on a
/// mutex, whatever else it does.
struct Row {
    outcome: Outcome,
    phase: Phase,
    count: AtomicU64,
}

#[derive(Default)]
struct State {
    rows: HashMap<Key, Row>,
    /// Lookup *shapes* seen during the build phase. A post-seal miss on
    /// a shape in here was predicted at load time and already reported;
    /// one that is not is the thing this registry exists to find.
    known: HashSet<(Backend, &'static str, Option<QuantKind>)>,
    /// Post-seal unpredicted misses, in discovery order.
    surprises: Vec<Entry>,
}

impl State {
    fn snapshot(&self, build_only: bool) -> Vec<Entry> {
        let mut entries: Vec<Entry> = self
            .rows
            .iter()
            .filter(|(_, row)| !build_only || row.phase == Phase::Build)
            .map(|(key, row)| Entry {
                key: *key,
                outcome: row.outcome,
                phase: row.phase,
                count: row.count.load(Ordering::Relaxed),
            })
            .collect();
        entries.sort_by_key(|e| {
            (
                e.key.backend,
                e.key.op,
                e.key.kind.map(|k| k.name()).unwrap_or("f32"),
                e.key.role,
                e.key.line,
            )
        });
        entries
    }
}

/// A registry instance. There is one [`global`] instance; tests build
/// their own so they never race the process-wide one.
pub struct Registry {
    inner: RwLock<State>,
    sealed: AtomicBool,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        Registry {
            inner: RwLock::new(State::default()),
            sealed: AtomicBool::new(false),
        }
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, State> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, State> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Shared-lock increment for a row that already exists. Returns
    /// false when the key is new and the caller must take the write
    /// lock. This is the whole steady-state cost of the registry on a
    /// dispatch path.
    fn bump_existing(&self, key: &Key) -> bool {
        match self.read().rows.get(key) {
            Some(row) => {
                row.count.fetch_add(1, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    pub fn is_sealed(&self) -> bool {
        self.sealed.load(Ordering::Relaxed)
    }

    /// Records a build-phase lookup. Always accepted, even after
    /// [`Self::seal`] — a process that loads a second model (draft +
    /// target for speculative decoding) probes it too, and those
    /// lookups are predictions, not surprises.
    pub fn record_build_at(&self, loc: &'static Location<'static>, l: Lookup, outcome: Outcome) {
        let key = Key {
            backend: l.backend,
            op: l.op,
            role: l.role,
            kind: l.kind,
            file: loc.file(),
            line: loc.line(),
        };
        if self.bump_existing(&key) {
            return;
        }
        let mut st = self.write();
        st.known.insert(key.shape());
        st.rows
            .entry(key)
            .or_insert_with(|| Row {
                outcome,
                phase: Phase::Build,
                count: AtomicU64::new(0),
            })
            .count
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records a dispatch-site lookup. Before sealing this is just
    /// bookkeeping; after sealing, a miss on a shape the build probe
    /// never predicted warns once and is collected into
    /// [`Self::surprises`].
    pub fn record_at(&self, loc: &'static Location<'static>, l: Lookup, outcome: Outcome) {
        let key = Key {
            backend: l.backend,
            op: l.op,
            role: l.role,
            kind: l.kind,
            file: loc.file(),
            line: loc.line(),
        };
        // Steady state: the key already exists, so the warn-or-not
        // decision was settled when it was created. A shared read lock
        // and one relaxed add, and nothing here blocks anything.
        if self.bump_existing(&key) {
            return;
        }
        let sealed = self.is_sealed();
        let mut st = self.write();
        // Re-check under the write lock: another thread may have raced
        // us to this key, in which case it already made the decision.
        if let Some(row) = st.rows.get(&key) {
            row.count.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let phase = if sealed { Phase::Run } else { Phase::Build };
        st.rows.insert(
            key,
            Row {
                outcome,
                phase,
                count: AtomicU64::new(1),
            },
        );
        if !sealed {
            st.known.insert(key.shape());
            return;
        }
        if !outcome.is_slow_path() || st.known.contains(&key.shape()) {
            return;
        }
        st.surprises.push(Entry {
            key,
            outcome,
            phase: Phase::Run,
            count: 1,
        });
        drop(st);
        let fallback = outcome
            .fallback()
            .unwrap_or("(unknown)" /* unreachable: guarded by is_slow_path */);
        eprintln!(
            "frink: SILENT SLOW PATH — kernel lookup missed after the model was sealed.\n\
             frink:   {} {} for {} has no kernel; falling back to {}.\n\
             frink:   call site {}:{} (role {}).\n\
             frink:   this was not predicted at load time, so no startup diagnostic covered it.\n\
             frink:   set FRINK_STRICT_KERNELS=1 to make this a hard error.",
            key.backend,
            key.op,
            key.kind_name(),
            fallback,
            key.file,
            key.line,
            key.role,
        );
    }

    /// Freezes the build picture and switches on post-seal checking.
    /// Idempotent: re-sealing recomputes the report over everything
    /// recorded so far.
    pub fn seal(&self) -> SealReport {
        self.sealed.store(true, Ordering::Relaxed);
        let entries = self.read().snapshot(true);
        let misses: Vec<Entry> = entries
            .iter()
            .copied()
            .filter(|e| e.outcome.is_miss())
            .collect();
        let violations: Vec<Entry> = misses
            .iter()
            .copied()
            .filter(|e| e.key.backend.is_accelerator() && e.outcome.is_slow_path())
            .collect();
        SealReport {
            entries,
            misses,
            violations,
        }
    }

    /// Post-seal misses that the build probe did not predict.
    pub fn surprises(&self) -> Vec<Entry> {
        self.read().surprises.clone()
    }

    /// Every row, build and run, sorted like [`SealReport::entries`].
    pub fn entries(&self) -> Vec<Entry> {
        self.read().snapshot(false)
    }
}

static GLOBAL: OnceLock<Registry> = OnceLock::new();

/// The process-wide registry.
pub fn global() -> &'static Registry {
    GLOBAL.get_or_init(Registry::new)
}

/// Whether the registry records at all.
///
/// Default **on**: a warning nobody enabled is the entire point. Read
/// once per process (`FRINK_KERNEL_REGISTRY=0|false|off` disables), so
/// no dispatch ever touches the environment.
pub fn enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        !matches!(
            std::env::var("FRINK_KERNEL_REGISTRY").ok().as_deref(),
            Some("0") | Some("false") | Some("off")
        )
    })
}

/// Whether the full build-phase table is printed at [`seal`].
pub fn verbose() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            std::env::var("FRINK_KERNEL_REGISTRY").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("verbose")
        )
    })
}

/// Whether a missing kernel is a hard error. Set this in CI and in
/// benchmark harnesses so a slow path cannot be published as a number.
pub fn strict() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            std::env::var("FRINK_STRICT_KERNELS").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}

/// Record an eager, load-time lookup against the global registry.
#[track_caller]
pub fn record_build(l: Lookup, outcome: Outcome) {
    if !enabled() {
        return;
    }
    global().record_build_at(Location::caller(), l, outcome);
}

/// Record a dispatch-site lookup that found a kernel. Dispatch sites do
/// not normally call this — hits cost nothing precisely because they are
/// not recorded on the hot path — but it exists for completeness.
#[track_caller]
pub fn hit(l: Lookup) {
    if !enabled() {
        return;
    }
    global().record_at(Location::caller(), l, Outcome::Hit);
}

/// Record a dispatch-site lookup that missed and is taking a slower
/// `fallback` than the selected backend.
///
/// Call this on the fallback branch only. By construction the branch is
/// already about to do far more work than a hash and a lock.
#[track_caller]
pub fn miss(l: Lookup, fallback: &'static str) {
    if !enabled() {
        return;
    }
    global().record_at(Location::caller(), l, Outcome::slow_path(fallback));
}

/// Record a dispatch-site lookup that missed, where the fallback is the
/// deliberate choice rather than a gap. Never warns; recorded so the
/// report is a complete picture instead of a filtered one.
#[track_caller]
pub fn miss_by_design(l: Lookup, fallback: &'static str) {
    if !enabled() {
        return;
    }
    global().record_at(Location::caller(), l, Outcome::by_design(fallback));
}

/// Seal the global registry: print what the build probe found, warn
/// loudly about quantized weights that will run off the selected
/// accelerator, and switch on post-seal checking.
pub fn seal() -> SealReport {
    let report = global().seal();
    if !enabled() {
        return report;
    }
    if verbose() {
        eprint!("{}", report.render());
    }
    if !report.violations.is_empty() && !strict() {
        eprint!("{}", report.render_violations());
        eprintln!(
            "frink: {} kernel lookup(s) above will run on a slower path than the \
             selected backend. Set FRINK_STRICT_KERNELS=1 to refuse to run instead.",
            report.violations.len()
        );
    }
    report
}

/// [`seal`], but returns `Err` under `FRINK_STRICT_KERNELS=1` when a
/// quantized weight has no kernel on the selected accelerator.
pub fn seal_or_error() -> Result<SealReport, StrictKernelError> {
    let report = seal();
    if strict() && !report.violations.is_empty() {
        return Err(StrictKernelError { report });
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(backend: Backend, kind: Option<QuantKind>) -> Lookup {
        Lookup::new(backend, op::GEMM_PREFILL, kind).with_role("ffn_down")
    }

    #[test]
    fn a_build_hit_is_recorded_once_per_shape_with_a_count() {
        let r = Registry::new();
        let loc = Location::caller();
        for _ in 0..5 {
            r.record_build_at(
                loc,
                lookup(Backend::Metal, Some(QuantKind::Q4K)),
                Outcome::Hit,
            );
        }
        let report = r.seal();
        assert_eq!(report.entries.len(), 1);
        assert_eq!(report.entries[0].count, 5);
        assert!(report.misses.is_empty());
        assert!(report.violations.is_empty());
    }

    /// A quantized weight with no accelerator kernel is exactly the
    /// IQ4_XS bug: correct output, silent CPU fallback, no diagnostic.
    /// Seal must call it a violation.
    #[test]
    fn a_quantized_weight_with_no_accelerator_kernel_is_a_violation() {
        let r = Registry::new();
        r.record_build_at(
            Location::caller(),
            lookup(Backend::Metal, Some(QuantKind::IQ4XS)),
            Outcome::slow_path("CPU apply_batch"),
        );
        let report = r.seal();
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].key.kind, Some(QuantKind::IQ4XS));
        assert!(report.render_violations().contains("IQ4_XS"));
    }

    /// An F32 matrix has no quantized kernel by construction and its
    /// host GEMV is a deliberate decision. It is a miss, but not a
    /// violation — otherwise every MoE router would fail the gate and
    /// the signal would be worthless.
    #[test]
    fn an_f32_miss_is_reported_but_is_not_a_violation() {
        let r = Registry::new();
        r.record_build_at(
            Location::caller(),
            lookup(Backend::Metal, None),
            Outcome::by_design("host GEMV"),
        );
        let report = r.seal();
        assert_eq!(report.misses.len(), 1);
        assert!(report.violations.is_empty());
    }

    /// A CPU-only process has no accelerator to fall off, so its misses
    /// are informational.
    #[test]
    fn a_cpu_backend_miss_is_not_a_violation() {
        let r = Registry::new();
        r.record_build_at(
            Location::caller(),
            lookup(Backend::Cpu, Some(QuantKind::IQ2XXS)),
            Outcome::slow_path("f32 dequant-dot"),
        );
        assert!(r.seal().violations.is_empty());
    }

    #[test]
    fn a_post_seal_miss_on_a_predicted_shape_is_not_a_surprise() {
        let r = Registry::new();
        let loc = Location::caller();
        r.record_build_at(
            loc,
            lookup(Backend::Metal, Some(QuantKind::IQ4XS)),
            Outcome::slow_path("CPU apply_batch"),
        );
        r.seal();
        r.record_at(
            loc,
            lookup(Backend::Metal, Some(QuantKind::IQ4XS)),
            Outcome::slow_path("CPU apply_batch"),
        );
        assert!(r.surprises().is_empty(), "seal already reported this shape");
    }

    /// The registry's whole purpose: a lookup that misses on a shape the
    /// build probe never saw is an unpredicted slow path.
    #[test]
    fn a_post_seal_miss_on_an_unpredicted_shape_is_a_surprise_reported_once() {
        let r = Registry::new();
        let loc = Location::caller();
        r.record_build_at(
            loc,
            lookup(Backend::Metal, Some(QuantKind::Q4K)),
            Outcome::Hit,
        );
        r.seal();
        for _ in 0..3 {
            r.record_at(
                loc,
                lookup(Backend::Metal, Some(QuantKind::Q2K)),
                Outcome::slow_path("CPU apply_batch"),
            );
        }
        let surprises = r.surprises();
        assert_eq!(surprises.len(), 1, "warned once, not once per dispatch");
        assert_eq!(surprises[0].key.kind, Some(QuantKind::Q2K));
        assert_eq!(surprises[0].phase, Phase::Run);
    }

    #[test]
    fn a_post_seal_hit_is_never_a_surprise() {
        let r = Registry::new();
        r.seal();
        r.record_at(
            Location::caller(),
            lookup(Backend::Metal, Some(QuantKind::Q4K)),
            Outcome::Hit,
        );
        assert!(r.surprises().is_empty());
    }

    /// Probing a second model after the first was sealed must not
    /// manufacture surprises (speculative decoding loads two).
    #[test]
    fn build_records_after_seal_extend_the_predicted_set() {
        let r = Registry::new();
        let loc = Location::caller();
        r.seal();
        r.record_build_at(
            loc,
            lookup(Backend::Metal, Some(QuantKind::Q6K)),
            Outcome::slow_path("CPU apply_batch"),
        );
        r.record_at(
            loc,
            lookup(Backend::Metal, Some(QuantKind::Q6K)),
            Outcome::slow_path("CPU apply_batch"),
        );
        assert!(r.surprises().is_empty());
    }

    /// The dispatch path runs on rayon workers, so the first-sighting
    /// write must be raced-into by many threads and still warn exactly
    /// once, with every lookup counted.
    #[test]
    fn concurrent_dispatch_misses_warn_once_and_count_all() {
        let r = std::sync::Arc::new(Registry::new());
        let loc = Location::caller();
        r.seal();
        std::thread::scope(|s| {
            for _ in 0..8 {
                let r = std::sync::Arc::clone(&r);
                s.spawn(move || {
                    for _ in 0..250 {
                        r.record_at(
                            loc,
                            lookup(Backend::Metal, Some(QuantKind::IQ1S)),
                            Outcome::slow_path("CPU apply_batch"),
                        );
                    }
                });
            }
        });
        assert_eq!(r.surprises().len(), 1, "warned once across 8 threads");
        let counted: u64 = r
            .entries()
            .iter()
            .filter(|e| e.key.kind == Some(QuantKind::IQ1S))
            .map(|e| e.count)
            .sum();
        assert_eq!(counted, 2000, "every lookup counted exactly once");
    }

    #[test]
    fn the_report_names_the_call_site_and_the_quant_kind() {
        let r = Registry::new();
        r.record_build_at(
            Location::caller(),
            lookup(Backend::Metal, Some(QuantKind::Q5K)),
            Outcome::slow_path("CPU apply_batch"),
        );
        let rendered = r.seal().render();
        assert!(rendered.contains("Q5_K"), "{rendered}");
        assert!(rendered.contains("kernel_registry.rs"), "{rendered}");
        assert!(rendered.contains("ffn_down"), "{rendered}");
        assert!(rendered.contains("CPU apply_batch"), "{rendered}");
    }
}
